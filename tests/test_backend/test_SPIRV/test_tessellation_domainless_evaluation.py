"""A tessellation evaluation stage that states no domain leaves it, and every unstated mode, to its control stage.

SPIR-V lets either tessellation stage declare the domain, spacing, vertex order and ``PointMode`` (SPIR-V
specification, Execution Mode), so an evaluation entry may declare none of them. SPIR-V output used to write
``Triangles``, ``SpacingEqual`` and ``VertexOrderCcw`` for such an entry; it now declares only the modes the stage
states. GLSL output refuses the stage, because a GLSL evaluation shader must declare its primitive mode (GLSL 4.60
section 4.4.1.2), and a program whose two tessellation stages both leave the domain out is refused for SPIR-V.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_refusals import OpenGLLoweringRefused
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

from .test_tessellation_evaluation_modes import _module

DOMAINLESS_MODE_SETS = {
    "none": (),
    "fractional-odd": ("SpacingFractionalOdd",),
    "cw": ("VertexOrderCw",),
    "point-mode": ("PointMode",),
    "fractional-even-cw-point": (
        "SpacingFractionalEven",
        "VertexOrderCw",
        "PointMode",
    ),
}

PROGRAM = """shader Probe {{
    {control}
    tessellation_evaluation {{
        {layout}
        void ds() @stage_entry {{
            gl_Position = vec4(gl_TessCoord, 1.0);
        }}
    }}
}}
"""

CONTROL = """tessellation_control {{
        layout(vertices = 3) out;
        void hs() @stage_entry {attributes} {{
            gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
            if (gl_InvocationID == 0) {{
                gl_TessLevelOuter[0] = 2.0;
                gl_TessLevelOuter[1] = 2.0;
                gl_TessLevelOuter[2] = 2.0;
                gl_TessLevelInner[0] = 2.0;
            }}
        }}
    }}"""


def _write(tmp_path, text, name):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _validate(tmp_path, assembly, name):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    binary = tmp_path / f"{name}.spv"
    for command in (
        [
            "spirv-as",
            "--target-env",
            "vulkan1.1",
            _write(tmp_path, assembly, f"{name}.spvasm"),
            "-o",
            str(binary),
        ],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr


def _modes(spirv, model="TessellationEvaluation"):
    entry = re.search(rf"OpEntryPoint {model} %(\w+)", spirv)
    assert entry is not None, spirv
    return sorted(
        re.findall(rf"OpExecutionMode %{re.escape(entry.group(1))} (\w+)", spirv)
    )


def _imported(tmp_path, modes):
    path = _write(tmp_path, _module(modes), "evaluation.spvasm")
    crossgl = crosstl.translate(path, backend="cgl", format_output=False)
    return _write(tmp_path, crossgl, "evaluation.cgl")


def _program(tmp_path, layout="", control=None):
    control_text = "" if control is None else CONTROL.format(attributes=control)
    return _write(
        tmp_path, PROGRAM.format(control=control_text, layout=layout), "probe.cgl"
    )


def _translate(path, backend):
    return crosstl.translate(path, backend=backend, format_output=False)


@pytest.mark.parametrize("name", sorted(DOMAINLESS_MODE_SETS))
def test_spirv_round_trip_declares_only_the_modules_modes(tmp_path, name):
    modes = DOMAINLESS_MODE_SETS[name]
    _validate(tmp_path, _module(modes), "fixture")

    spirv = _translate(_imported(tmp_path, modes), "vulkan")

    _validate(tmp_path, spirv, "round-trip")
    assert _modes(spirv) == sorted(modes)


@pytest.mark.parametrize("name", sorted(DOMAINLESS_MODE_SETS))
def test_glsl_output_of_a_domainless_evaluation_is_refused(tmp_path, name):
    path = _imported(tmp_path, DOMAINLESS_MODE_SETS[name])

    with pytest.raises(OpenGLLoweringRefused, match="states no domain"):
        _translate(path, "opengl")


def test_glsl_layout_without_a_domain_states_only_its_own_modes(tmp_path):
    path = _program(tmp_path, "layout(fractional_odd_spacing, cw) in;")

    spirv = _translate(path, "vulkan")

    _validate(tmp_path, spirv, "layout")
    assert _modes(spirv) == ["SpacingFractionalOdd", "VertexOrderCw"]


def test_glsl_layout_with_a_domain_keeps_glsls_defaults(tmp_path):
    spirv = _translate(_program(tmp_path, "layout(quads) in;"), "vulkan")

    assert _modes(spirv) == ["Quads", "SpacingEqual", "VertexOrderCcw"]


def test_domain_from_the_control_stage_is_not_refused(tmp_path):
    control = '@domain("tri") @partitioning("integer") @outputtopology("triangle_cw")'
    path = _program(tmp_path, control=control)

    spirv = _translate(path, "vulkan")
    glsl = _translate(path, "opengl")

    _validate(tmp_path, spirv, "with-control")
    assert _modes(spirv, "TessellationControl") == [
        "OutputVertices",
        "SpacingEqual",
        "Triangles",
        "VertexOrderCw",
    ]
    assert "Triangles" in _modes(spirv)
    assert "layout(triangles, equal_spacing, cw) in;" in glsl


def test_program_whose_stages_both_leave_the_domain_out_is_refused(tmp_path):
    path = _program(tmp_path, "layout(fractional_odd_spacing) in;", control="")

    with pytest.raises(SpirvLoweringRefused, match="states no domain"):
        _translate(path, "vulkan")
