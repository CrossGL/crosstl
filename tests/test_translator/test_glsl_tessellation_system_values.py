"""CrossGL tessellation stages that name their system values by HLSL semantics translate to GLSL and SPIR-V.

CrossGL accepts ``@SV_TessFactor``, ``@SV_InsideTessFactor``, ``@SV_OutputControlPointID`` and ``@SV_DomainLocation``
as well as the GLSL built-ins the HLSL importer writes for them. The GLSL and SPIR-V tessellation lowering read only
the GLSL names, so an HLSL-style hull or domain stage spelled with the semantics was refused (``member
'PatchConstants.edges' carries SV_TessFactor``). Both spellings now give the same output.

A quad patch tessellates into triangles, so a hull with a quad domain and ``triangle_cw`` or ``triangle_ccw``
topology gives its evaluation shader the ``cw`` or ``ccw`` vertex order (GLSL 4.60 section 4.4.1.2); only an isoline
patch, which tessellates into lines, refuses a triangle topology.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import TessellationPatchUnsupportedError

_SEMANTICS = {
    "outer": ("SV_TessFactor", "gl_TessLevelOuter"),
    "inner": ("SV_InsideTessFactor", "gl_TessLevelInner"),
    "point": ("SV_OutputControlPointID", "gl_InvocationID"),
    "coord": ("SV_DomainLocation", "gl_TessCoord"),
}


def _program(spelling, domain="tri", topology="triangle_cw", edges=3, inside=""):
    names = {key: pair[spelling] for key, pair in _SEMANTICS.items()}
    size = 3 if domain == "tri" else 2
    return f"""
shader Probe {{
    struct ControlPoint {{
        vec4 position @ POSITION;
    }}
    struct PatchConstants {{
        float edges[{edges}] @ {names["outer"]};
        float inside{inside} @ {names["inner"]};
        vec4 extra @ EXTRA;
    }}
    struct DomainOutput {{
        vec4 position @ SV_Position;
    }}
    tessellation_control {{
        PatchConstants constants(InputPatch<ControlPoint, 3> patch) {{
            PatchConstants output;
            output.edges[0] = 2.0;
            output.edges[1] = 3.0;
            output.extra = patch[0].position;
            return output;
        }}
        ControlPoint main(InputPatch<ControlPoint, 3> patch, uint id @ {names["point"]})
            @domain({domain})
            @partitioning(integer)
            @outputtopology({topology})
            @outputcontrolpoints(3)
            @patchconstantfunc(constants) {{
            ControlPoint output;
            output.position = patch[id].position * 2.0;
            return output;
        }}
    }}
    tessellation_evaluation {{
        DomainOutput main(PatchConstants constants, vec{size} coord @ {names["coord"]},
                          OutputPatch<ControlPoint, 3> patch) @domain({domain}) {{
            DomainOutput output;
            output.position = patch[0].position * coord.x + constants.extra * constants.edges[1];
            return output;
        }}
    }}
}}
"""


def _translate(tmp_path, source, backend, name="probe.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


_MACROS = {"tesc": "GL_TESS_CONTROL_SHADER", "tese": "GL_TESS_EVALUATION_SHADER"}


def _glslang(tmp_path, code, stage):
    """Compile the ``stage`` part of a two-stage GLSL output, which guards each stage by its stage macro."""
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / f"out.{stage}"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", "-l", f"-D{_MACROS[stage]}=1", "-S", stage, str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["opengl", "vulkan"])
def test_semantic_spelling_translates_like_the_glsl_built_ins(tmp_path, backend):
    hlsl = _translate(tmp_path, _program(0), backend, "hlsl.cgl")
    glsl = _translate(tmp_path, _program(1), backend, "glsl.cgl")

    assert hlsl == glsl


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        (
            "tessellation_control",
            [
                "layout(vertices = 3) out;",
                "main_control_point(uint(gl_InvocationID))",
                "gl_TessLevelOuter[1] = ",
                "gl_TessLevelInner[0] = ",
                "patch out vec4",
            ],
        ),
        (
            "tessellation_evaluation",
            [
                "layout(triangles, equal_spacing, cw) in;",
                "gl_TessCoord",
                "gl_TessLevelOuter[1]",
                "patch in vec4",
            ],
        ),
    ],
)
def test_semantic_spelling_reaches_the_glsl_tessellation_interface(
    tmp_path, stage, expected
):
    code = _translate(tmp_path, _program(0), "opengl")

    for text in expected:
        assert text in code
    assert "SV_" not in code
    _glslang(tmp_path, code, "tesc" if stage == "tessellation_control" else "tese")


@pytest.mark.parametrize(("topology", "order"), [("triangle_cw", "cw"), ("triangle_ccw", "ccw")])
def test_quad_patch_with_triangle_topology_takes_its_vertex_order(
    tmp_path, topology, order
):
    source = _program(0, domain="quad", topology=topology, edges=4, inside="[2]")

    code = _translate(tmp_path, source, "opengl")

    assert f"layout(quads, equal_spacing, {order}) in;" in code
    _glslang(tmp_path, code, "tese")


def test_semantic_index_other_than_zero_is_still_refused(tmp_path):
    source = _program(0).replace("@ SV_TessFactor;", "@ SV_TessFactor0;")
    _translate(tmp_path, source, "opengl")

    with pytest.raises(Exception, match="SV_TessFactor1"):
        _translate(tmp_path, _program(0).replace("@ SV_TessFactor;", "@ SV_TessFactor1;"), "opengl")


def test_unrelated_system_value_on_a_patch_member_is_still_refused(tmp_path):
    source = _program(0).replace("vec4 extra @ EXTRA;", "uint view @ SV_ViewportArrayIndex;")
    source = source.replace("output.extra = patch[0].position;", "")
    source = source.replace("constants.extra", "patch[1].position")

    with pytest.raises(TessellationPatchUnsupportedError, match="gl_ViewportIndex"):
        _translate(tmp_path, source, "opengl")
