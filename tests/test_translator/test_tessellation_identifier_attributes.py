"""A tessellation attribute argument written as a bare identifier means the same as the quoted string.

CrossGL accepts ``@domain(tri)`` and ``@patchconstantfunc(Constants)`` as well as ``@domain("tri")`` and
``@patchconstantfunc("Constants")``; the DirectX generator already prints both forms the same way. The GLSL and
SPIR-V tessellation lowering and the HLSL tessellation-factor sizing read the argument of a bare identifier as the
identifier's name, so the bare and quoted forms translate to the same program.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.semantics.stage_varyings import attribute_text

PROGRAM = """
shader Patches {{
    struct ControlPoint {{
        vec4 position @ gl_Position;
        vec3 normal @ NORMAL;
    }}
    struct Constants {{
        float edges[{edges}] @ gl_TessLevelOuter;
        float inside @ gl_TessLevelInner;
        float weight @ WEIGHT;
    }}
    struct Vertex {{
        vec4 position @ gl_Position;
        vec3 normal @ NORMAL;
    }}

    Constants PatchConstants(InputPatch<ControlPoint, 3> patch) {{
        Constants c;
        c.edges[0] = 4.0;
        c.edges[1] = 4.0;
        c.edges[2] = 4.0;
        c.inside = 4.0;
        c.weight = patch[0].position.w;
        return c;
    }}

    tessellation_control {{
        @domain({domain})
        @partitioning({partitioning})
        @outputtopology({topology})
        @outputcontrolpoints(3)
        @patchconstantfunc({function})
        ControlPoint HullMain(InputPatch<ControlPoint, 3> patch, uint id @ gl_InvocationID) {{
            ControlPoint o;
            o.position = patch[id].position;
            o.normal = patch[id].normal;
            return o;
        }}
    }}

    tessellation_evaluation {{
        @domain({domain})
        Vertex DomainMain(Constants c, vec3 coord @ gl_TessCoord, const OutputPatch<ControlPoint, 3> patch) {{
            Vertex v;
            v.position = (patch[0].position * coord.x + patch[1].position * coord.y
                + patch[2].position * coord.z) * c.weight * c.edges[0] * c.inside;
            v.normal = normalize(patch[0].normal * coord.x + patch[1].normal * coord.y + patch[2].normal * coord.z);
            return v;
        }}
    }}
}}
"""

ARGUMENTS = {
    "domain": "tri",
    "partitioning": "fractional_odd",
    "topology": "triangle_cw",
    "function": "PatchConstants",
}


def _program(bare, edges=3):
    """The program with every tessellation attribute argument quoted, or only those named in ``bare`` bare."""
    values = {
        key: value if key in bare else f'"{value}"' for key, value in ARGUMENTS.items()
    }
    return PROGRAM.format(edges=edges, **values)


def _translate(tmp_path, source, backend, entry_point=None, name="patches.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, entry_point=entry_point
    )


def _glslang(tmp_path, code, stage):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / f"checked.{stage}"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bare", sorted(ARGUMENTS))
@pytest.mark.parametrize(
    "entry_point, stage", [("HullMain", "tesc"), ("DomainMain", "tese")]
)
def test_bare_attribute_argument_lowers_to_the_glsl_stage_of_the_quoted_one(
    tmp_path, bare, entry_point, stage
):
    quoted = _translate(tmp_path, _program(()), "opengl", entry_point)
    code = _translate(tmp_path, _program({bare}), "opengl", entry_point)

    assert code == quoted
    if stage == "tesc":
        assert "layout(vertices = 3) out;" in code
        assert "Constants hs_patch = PatchConstants(" in code
        assert "gl_TessLevelOuter[2] = hs_patch.edges[2];" in code
    else:
        assert "layout(triangles, fractional_odd_spacing, cw) in;" in code
        assert "ds_patch.inside = gl_TessLevelInner[0];" in code
    _glslang(tmp_path, code, stage)


def test_every_attribute_bare_lowers_to_the_glsl_stages_of_the_quoted_program(
    tmp_path,
):
    for entry_point, stage in (("HullMain", "tesc"), ("DomainMain", "tese")):
        quoted = _translate(tmp_path, _program(()), "opengl", entry_point)
        code = _translate(tmp_path, _program(set(ARGUMENTS)), "opengl", entry_point)
        assert code == quoted
        _glslang(tmp_path, code, stage)


@pytest.mark.parametrize("bare", sorted(ARGUMENTS))
def test_bare_attribute_argument_lowers_to_the_spirv_of_the_quoted_one(tmp_path, bare):
    quoted = _translate(tmp_path, _program(()), "vulkan")
    code = _translate(tmp_path, _program({bare}), "vulkan")

    assert code == quoted
    assert "OpExecutionMode" in code and "Triangles" in code
    assert "SpacingFractionalOdd" in code


def test_hlsl_sizes_tessellation_factors_to_a_bare_domain(tmp_path):
    """Four outer factors for a triangle domain narrow to the three DXIL requires, as with ``@domain("tri")``."""
    quoted = _translate(tmp_path, _program((), edges=4), "directx")
    code = _translate(tmp_path, _program({"domain"}, edges=4), "directx")

    assert code == quoted
    assert "float edges[3] : SV_TessFactor;" in code


def test_attribute_text_reads_strings_and_identifiers():
    assert attribute_text(A.IdentifierNode("tri")) == "tri"
    assert attribute_text(A.LiteralNode('"quad"', "string")) == "quad"
    assert attribute_text(A.LiteralNode("isoline", "string")) == "isoline"
    assert attribute_text("'integer'") == "integer"
