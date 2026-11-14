"""Metal refuses the stages it cannot write as entry points, with a typed diagnostic.

Metal has no geometry stage, and its tessellation pipeline (a compute kernel writing tessellation factors and a
``[[patch]]`` post-tessellation vertex function) is not lowered yet, so these stages must not come out as plain
functions that no Metal pipeline can call.
"""

import pytest

import crosstl
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.metal_stage_refusals import MetalStageUnsupportedError
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

CODE = "project.translate.metal-stage-unsupported"

GEOMETRY = """
shader GeometryExpansion {
    struct VOut {
        vec4 position @ gl_Position;
        vec3 color @ COLOR;
    };

    vertex {
        VOut main() {
            VOut o;
            o.position = vec4(0.0, 0.0, 0.0, 1.0);
            o.color = vec3(1.0);
            return o;
        }
    }

    geometry {
        void main(triangle VOut input[3], inout TriangleStream<VOut> stream)
            @maxvertexcount(3) {
            stream.Append(input[0]);
            stream.RestartStrip();
        }
    }

    fragment {
        vec4 main(VOut input) @ gl_FragColor {
            return vec4(input.color, 1.0);
        }
    }
}
"""

TESSELLATION_CONTROL = """
shader TessControl {
    struct PatchInput { vec3 position @ POSITION; };
    struct PatchOutput { vec3 position @ POSITION; };
    struct PatchConstants {
        vec3 edgeFactors @ SV_TessFactor;
        float insideFactor @ SV_InsideTessFactor;
    };

    tessellation_control {
        PatchConstants patchConstants(InputPatch<PatchInput, 3> patch) {
            PatchConstants constants;
            constants.edgeFactors = vec3(1.0);
            constants.insideFactor = 1.0;
            return constants;
        }

        PatchOutput main(InputPatch<PatchInput, 3> patch, uint pointID @ SV_OutputControlPointID)
            @domain(tri) @partitioning(integer) @outputtopology(triangle_cw)
            @outputcontrolpoints(3) @patchconstantfunc(patchConstants) {
            PatchOutput output;
            output.position = patch[pointID].position;
            return output;
        }
    }
}
"""

TESSELLATION_EVALUATION = """
shader TessEval {
    struct DomainInput { vec3 position @ POSITION; };

    tessellation_evaluation {
        vec4 main(OutputPatch<DomainInput, 3> patch, vec3 uvw @ SV_DomainLocation) @domain(tri) {
            vec3 p = patch[0].position * uvw.x + patch[1].position * uvw.y + patch[2].position * uvw.z;
            return vec4(p, 1.0);
        }
    }
}
"""

HLSL_GEOMETRY = """
struct VSOut { float4 pos : SV_Position; float3 color : COLOR0; };

[maxvertexcount(3)]
void main(triangle VSOut input[3], inout TriangleStream<VSOut> stream) {
    for (int i = 0; i < 3; i++) {
        stream.Append(input[i]);
    }
    stream.RestartStrip();
}
"""


def parse(code):
    return Parser(Lexer(code).get_tokens()).parse()


@pytest.mark.parametrize(
    ("code", "stage", "phrase"),
    [
        (GEOMETRY, "geometry", "Metal has no geometry stage"),
        (TESSELLATION_CONTROL, "tessellation_control", "compute kernel"),
        (TESSELLATION_EVALUATION, "tessellation_evaluation", "[[patch]]"),
    ],
)
def test_metal_refuses_stage_without_metal_entry_point(code, stage, phrase):
    with pytest.raises(MetalStageUnsupportedError) as raised:
        MetalCodeGen().generate(parse(code))
    error = raised.value
    assert isinstance(error, TranslationDiagnostic)
    assert error.code == CODE
    assert error.project_diagnostic_code == CODE
    assert error.stage == stage
    assert error.target == "metal"
    assert phrase in str(error)

    with pytest.raises(MetalStageUnsupportedError):
        MetalCodeGen().generate_stage(parse(code), stage)


def test_metal_emits_the_other_stages_of_a_program_with_a_geometry_stage():
    ast = parse(GEOMETRY)
    vertex = MetalCodeGen().generate_stage(ast, "vertex")
    fragment = MetalCodeGen().generate_stage(parse(GEOMETRY), "fragment")

    assert "vertex VOut vertex_main" in vertex
    assert "fragment float4 fragment_main" in fragment
    for code in (vertex, fragment):
        assert "Stream" not in code
        assert "geometry_main" not in code


GLSL_GEOMETRY = """#version 450
layout(points) in;
layout(points, max_vertices = 1) out;
layout(location = 0) in vec3 inColor[];
layout(location = 0) out vec3 outColor;
void main() {
    outColor = inColor[0];
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();
    EndPrimitive();
}
"""

GLSL_TESSELLATION_EVALUATION = """#version 450
layout(triangles, equal_spacing, cw) in;
void main() {
    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position
        + gl_TessCoord.z * gl_in[2].gl_Position;
}
"""


@pytest.mark.parametrize(
    ("name", "source", "stage"),
    [
        ("points.geom", GLSL_GEOMETRY, "geometry"),
        ("triangles.tese", GLSL_TESSELLATION_EVALUATION, "tessellation_evaluation"),
    ],
)
def test_glsl_stage_to_metal_is_a_typed_refusal(tmp_path, name, source, stage):
    path = tmp_path / name
    path.write_text(source)
    with pytest.raises(MetalStageUnsupportedError) as raised:
        crosstl.translate(str(path), backend="metal")
    assert raised.value.stage == stage


def test_hlsl_geometry_shader_to_metal_is_a_typed_refusal(tmp_path):
    source = tmp_path / "expand.gs.hlsl"
    source.write_text(HLSL_GEOMETRY)
    with pytest.raises(TranslationDiagnostic) as raised:
        crosstl.translate(str(source), backend="metal")
    assert raised.value.code == CODE
