"""SPIR-V input keeps exactly the primitive-generator modes a tessellation evaluation entry point declares.

DXC writes an HLSL domain shader's ``[domain]`` as the ``TessellationEvaluation`` entry's only mode and leaves the
spacing and the vertex order to the hull's control entry (SPIR-V specification, Execution Mode). The importer used
to write such an entry as a GLSL input layout, ``layout(triangles) in;``, which means ``equal_spacing`` and ``ccw``
as well, so SPIR-V output declared two modes the module never had. An entry that leaves the spacing or the vertex
order out now takes the HLSL-style ``@domain``/``@partitioning``/``@outputtopology`` attributes, which state only
what they name. An entry that states all three keeps its layout.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.SPIRV.tessellation_evaluation_import import (
    evaluation_leaves_modes_to_control_stage,
    tessellation_evaluation_mode_attributes,
)
from crosstl.translator.codegen.hlsl_glsl_tessellation_interface import (
    DirectXGLSLTessellationInterfaceError,
)

EVALUATION_TEMPLATE = """               OpCapability Tessellation
               OpMemoryModel Logical GLSL450
               OpEntryPoint TessellationEvaluation %main "main" %gl_TessCoord %out_pos
{modes}
               OpName %main "main"
               OpName %out_pos "out_pos"
               OpDecorate %gl_TessCoord BuiltIn TessCoord
               OpDecorate %out_pos Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v3float = OpTypeVector %float 3
    %v4float = OpTypeVector %float 4
    %float_1 = OpConstant %float 1
  %ptr_in_v3 = OpTypePointer Input %v3float
 %ptr_out_v4 = OpTypePointer Output %v4float
%gl_TessCoord = OpVariable %ptr_in_v3 Input
    %out_pos = OpVariable %ptr_out_v4 Output
       %main = OpFunction %void None %fn
      %label = OpLabel
      %coord = OpLoad %v3float %gl_TessCoord
          %x = OpCompositeExtract %float %coord 0
          %y = OpCompositeExtract %float %coord 1
          %z = OpCompositeExtract %float %coord 2
        %pos = OpCompositeConstruct %v4float %x %y %z %float_1
               OpStore %out_pos %pos
               OpReturn
               OpFunctionEnd
"""

# Mode sets that leave the spacing or the vertex order to the control stage, as DXC writes a domain shader's.
PARTIAL_MODE_SETS = {
    "dxc-tri": (("Triangles",), ['@domain("tri")']),
    "dxc-quad": (("Quads",), ['@domain("quad")']),
    "dxc-isoline": (("Isolines",), ['@domain("isoline")']),
    "tri-fractional-odd": (
        ("Triangles", "SpacingFractionalOdd"),
        ['@domain("tri")', '@partitioning("fractional_odd")'],
    ),
    "quad-cw": (
        ("Quads", "VertexOrderCw"),
        ['@domain("quad")', '@outputtopology("triangle_cw")'],
    ),
}

# Mode sets that state everything GLSL would otherwise default, as glslang writes an evaluation shader's.
COMPLETE_MODE_SETS = {
    "glslang-tri": (
        ("Triangles", "SpacingEqual", "VertexOrderCcw"),
        "layout(triangles, equal_spacing, ccw) in;",
    ),
    "quad-fractional-even-cw": (
        ("Quads", "SpacingFractionalEven", "VertexOrderCw"),
        "layout(quads, fractional_even_spacing, cw) in;",
    ),
}


def _module(modes):
    return EVALUATION_TEMPLATE.format(
        modes="\n".join(
            f"               OpExecutionMode %main {mode}" for mode in modes
        )
    )


def _write(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return str(path)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def _validate(tmp_path, assembly, name):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    source = _write(tmp_path, assembly, f"{name}.spvasm")
    binary = tmp_path / f"{name}.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", source, "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def _crossgl(tmp_path, modes):
    return crosstl.translate(
        _write(tmp_path, _module(modes), "evaluation.spvasm"),
        backend="cgl",
        format_output=False,
    )


def _round_trip(tmp_path, modes, backend="vulkan"):
    path = _write(tmp_path, _crossgl(tmp_path, modes), "evaluation.cgl")
    return crosstl.translate(path, backend=backend, format_output=False)


def _evaluation_modes(spirv):
    entry = re.search(r"OpEntryPoint TessellationEvaluation %(\w+)", spirv)
    assert entry is not None, spirv
    return sorted(
        re.findall(rf"OpExecutionMode %{re.escape(entry.group(1))} (\w+)", spirv)
    )


def _main_attributes(crossgl):
    main = re.search(r"void main\(\)([^{]*)\{", crossgl)
    assert main is not None, crossgl
    return main.group(1).split()


@pytest.mark.parametrize("name", sorted({**PARTIAL_MODE_SETS, **COMPLETE_MODE_SETS}))
def test_fixtures_are_valid_spirv(tmp_path, name):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    modes = {**PARTIAL_MODE_SETS, **COMPLETE_MODE_SETS}[name][0]
    _validate(tmp_path, _module(modes), name)


@pytest.mark.parametrize("name", sorted(PARTIAL_MODE_SETS))
def test_partial_modes_import_as_hlsl_style_attributes(tmp_path, name):
    modes, attributes = PARTIAL_MODE_SETS[name]
    crossgl = _crossgl(tmp_path, modes)

    assert "tessellation_evaluation {" in crossgl
    assert not re.search(r"layout\([^)]*\) in;", crossgl), crossgl
    assert _main_attributes(crossgl) == attributes


@pytest.mark.parametrize("name", sorted(PARTIAL_MODE_SETS))
def test_spirv_output_declares_only_the_modules_modes(tmp_path, name):
    modes = PARTIAL_MODE_SETS[name][0]
    spirv = _round_trip(tmp_path, modes)

    _validate(tmp_path, spirv, "round-trip")
    assert _evaluation_modes(spirv) == sorted(modes)


@pytest.mark.parametrize("name", sorted(COMPLETE_MODE_SETS))
def test_complete_modes_keep_the_glsl_layout(tmp_path, name):
    modes, layout = COMPLETE_MODE_SETS[name]
    crossgl = _crossgl(tmp_path, modes)

    assert layout in crossgl
    assert _main_attributes(crossgl) == []
    spirv = _round_trip(tmp_path, modes)
    _validate(tmp_path, spirv, "round-trip")
    assert _evaluation_modes(spirv) == sorted(modes)


def test_hlsl_output_of_a_glsl_style_evaluation_stage_is_refused(tmp_path):
    # The attributes satisfy the domain shader's [domain] requirement, but the stage still reads
    # gl_TessCoord and writes a stage-scope output, which an HLSL domain shader cannot.
    with pytest.raises(DirectXGLSLTessellationInterfaceError) as error:
        _round_trip(tmp_path, ("Triangles",), backend="directx")

    assert error.value.stage == "tessellation_evaluation"
    assert "gl_TessCoord" in error.value.names


@pytest.mark.parametrize(
    "modes, leaves",
    [
        (("Triangles",), True),
        (("Triangles", "SpacingEqual"), True),
        (("Triangles", "VertexOrderCcw", "PointMode"), True),
        (("Triangles", "SpacingEqual", "VertexOrderCw"), False),
        (("Isolines", "SpacingFractionalOdd", "VertexOrderCcw", "PointMode"), False),
        # An entry without a domain mode keeps the input layout; DXC and glslang always state the domain.
        (("SpacingEqual",), False),
        ((), False),
    ],
)
def test_which_entries_leave_modes_to_the_control_stage(modes, leaves):
    execution_modes = [{"mode": mode} for mode in modes]
    assert evaluation_leaves_modes_to_control_stage(execution_modes) is leaves
    assert bool(tessellation_evaluation_mode_attributes(execution_modes)) is leaves
