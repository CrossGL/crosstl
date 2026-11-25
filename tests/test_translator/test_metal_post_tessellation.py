"""CrossGL tessellation evaluation stages and Metal post-tessellation vertex functions, in both directions.

Metal has no domain stage: a ``[[patch(triangle|quad, N)]] vertex`` function reads its control points through a
``patch_control_point<T>`` member of its ``[[stage_in]]`` structure, the per-patch values through the other members,
and the position in the patch through ``[[position_in_patch]]`` (MSL 4.1 sections 5.1.5 and 5.2.3.2). The Metal
generator refused every tessellation evaluation stage, and the Metal importer kept ``patch_control_point`` and
``@patch`` verbatim and refused a subscript of the control points.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.metal_post_tessellation import (
    MetalPostTessellationError,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

TRIANGLE = """
shader Probe {
    struct ControlPoint {
        vec4 position @ POSITION;
        vec2 uv @ TEXCOORD0;
    }
    struct PatchConstants {
        float edges[3] @ SV_TessFactor;
        float inside @ SV_InsideTessFactor;
        vec4 extra @ EXTRA;
    }
    struct DomainOutput {
        vec4 position @ SV_Position;
        vec2 uv @ TEXCOORD0;
    }
    vec4 blend(OutputPatch<ControlPoint, 3> points, vec3 w) {
        return points[0].position * w.x + points[1].position * w.y + points[2].position * w.z;
    }
    tessellation_evaluation {
        DomainOutput main(PatchConstants constants, vec3 coord @ SV_DomainLocation,
                          OutputPatch<ControlPoint, 3> patch, uint id @ SV_PrimitiveID) @domain(tri) {
            DomainOutput output;
            output.position = blend(patch, coord) + constants.extra * float(id);
            output.uv = patch[0].uv * coord.x + patch[2].uv * coord.z;
            return output;
        }
    }
    fragment {
        vec4 main(DomainOutput input) @ gl_FragColor {
            return vec4(input.uv, 0.0, 1.0);
        }
    }
}
"""

QUAD = """
shader Probe {
    struct ControlPoint {
        vec4 position @ POSITION;
    }
    struct PatchConstants {
        float edges[4] @ SV_TessFactor;
        float inside[2] @ SV_InsideTessFactor;
    }
    struct DomainOutput {
        vec4 position @ SV_Position;
    }
    tessellation_evaluation {
        DomainOutput main(PatchConstants constants, vec2 coord @ SV_DomainLocation,
                          OutputPatch<ControlPoint, 4> patch) @domain(quad) @partitioning(integer) {
            DomainOutput output;
            output.position = patch[0].position * coord.x + patch[3].position * coord.y;
            return output;
        }
    }
}
"""

METAL = textwrap.dedent("""\
    #include <metal_stdlib>
    using namespace metal;

    struct ControlPoint {
        float4 position [[attribute(2)]];
        float2 uv [[attribute(5)]];
    };

    struct Patch {
        patch_control_point<ControlPoint> points;
        float4 extra [[attribute(0)]];
    };

    [[patch(quad, 4)]]
    vertex float4 probe(Patch patch [[stage_in]], float2 coord [[position_in_patch]], uint id [[patch_id]]) {
        float4 sum = 0.0;
        for (uint i = 0; i < patch.points.size(); ++i) {
            sum += patch.points[i].position * patch.points[i].uv.x;
        }
        return sum * coord.x + patch.extra * coord.y + float(id);
    }
    """)


def parse(code):
    return Parser(Lexer(code).get_tokens()).parse()


def compile_metal(tmp_path, source):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    (tmp_path / "patch.metal").write_text(source)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "patch.metal", "-o", "patch.air"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    assert result.returncode == 0, result.stderr


def test_triangle_domain_becomes_a_patch_vertex_function(tmp_path):
    code = MetalCodeGen().generate(parse(TRIANGLE))

    assert (
        "[[patch(triangle, 3)]]\nvertex DomainOutput tessellation_evaluation_main("
        in code
    )
    assert (
        "patch_control_point<tessellation_evaluation_main_PatchPoint> points;" in code
    )
    assert "float4 position [[attribute(0)]];" in code
    assert "float2 uv [[attribute(1)]];" in code
    # Patch constants continue after the control points; the tessellation factors are not stage inputs.
    assert "float4 extra [[attribute(2)]];" in code
    assert "edges [[" not in code and "inside [[" not in code
    assert "float3 coord [[position_in_patch]]" in code
    assert "uint id [[patch_id]]" in code
    assert "ControlPoint patch[3];" in code
    assert "constants.extra = _crossglPatch.extra;" in code
    # A helper that takes the patch takes an array of its control points.
    assert "OutputPatch" not in code
    assert "float2 uv [[user(TEXCOORD0)]];" in code
    compile_metal(tmp_path, code)


def test_quad_domain_drops_the_host_set_partitioning(tmp_path):
    code = MetalCodeGen().generate(parse(QUAD))

    assert "[[patch(quad, 4)]]" in code
    assert "float2 coord [[position_in_patch]]" in code
    assert "partitioning" not in code
    # The body never names its patch constants, so no copy of them is made.
    assert "PatchConstants constants;" not in code
    compile_metal(tmp_path, code)


@pytest.mark.parametrize(
    ("body", "phrase"),
    [
        (
            "output.position = patch[0].position * constants.edges[0];",
            "tessellation factor",
        ),
        ("output.position = use(constants);", "as a whole"),
    ],
)
def test_reading_the_tessellation_factors_is_refused(body, phrase):
    source = QUAD.replace(
        "output.position = patch[0].position * coord.x + patch[3].position * coord.y;",
        body,
    ).replace(
        "    tessellation_evaluation {",
        "    vec4 use(PatchConstants c) { return vec4(c.edges[0]); }\n    tessellation_evaluation {",
    )
    with pytest.raises(MetalPostTessellationError) as raised:
        MetalCodeGen().generate(parse(source))
    assert raised.value.code == "project.translate.metal-post-tessellation-unsupported"
    assert phrase in str(raised.value)


def test_isoline_domain_is_refused():
    source = QUAD.replace("@domain(quad)", "@domain(isoline)")
    with pytest.raises(MetalPostTessellationError, match="triangle and quad"):
        MetalCodeGen().generate(parse(source))


def test_metal_patch_function_imports_as_a_tessellation_evaluation_stage(tmp_path):
    path = tmp_path / "patch.metal"
    path.write_text(METAL)
    crossgl = crosstl.translate(str(path), backend="cgl")

    assert "tessellation_evaluation {" in crossgl
    assert "patch_control_point" not in crossgl
    assert (
        "vec4 probe(Patch patch, OutputPatch<ControlPoint, 4> patch_points, "
        "vec2 coord @SV_DomainLocation, uint id @gl_PrimitiveID) @domain(quad)"
    ) in crossgl
    assert "patch_points[i].position" in crossgl
    assert "i < 4;" in crossgl


def test_metal_patch_function_round_trip_keeps_attribute_indices(tmp_path):
    path = tmp_path / "patch.metal"
    path.write_text(METAL)
    code = crosstl.translate(str(path), backend="metal")

    assert "[[patch(quad, 4)]]" in code
    assert "float4 position [[attribute(2)]];" in code
    assert "float2 uv [[attribute(5)]];" in code
    assert "float4 extra [[attribute(0)]];" in code
    compile_metal(tmp_path, code)


def test_metal_patch_function_translates_to_hlsl_domain_shader(tmp_path):
    path = tmp_path / "patch.metal"
    path.write_text(METAL)
    hlsl = crosstl.translate(str(path), backend="directx")

    assert '[domain("quad")]' in hlsl
    assert "OutputPatch<ControlPoint, 4>" in hlsl
    assert "SV_DomainLocation" in hlsl
