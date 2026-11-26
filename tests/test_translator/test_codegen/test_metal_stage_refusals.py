"""Metal refuses the stages it cannot write as entry points, with a typed diagnostic.

Metal has no geometry stage, and the compute kernel that writes Metal's tessellation factors is not lowered from a
GLSL-form tessellation control stage yet, so these stages must not come out as plain functions that no Metal pipeline
can call. A domain stage becomes a ``[[patch]]`` post-tessellation vertex function (``test_metal_post_tessellation.py``
and ``test_metal_glsl_tessellation.py``) and an HLSL-form hull stage a compute kernel
(``test_metal_tessellation_control.py``).
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
    in vec4 inPosition @location(0)[];
    out vec4 outPosition @location(0)[];

    tessellation_control {
        layout(vertices = 3) out;

        void main() {
            outPosition[gl_InvocationID] = inPosition[gl_InvocationID] * 2.0;
            barrier();
            gl_TessLevelOuter[0] = 1.0;
            gl_TessLevelInner[0] = 1.0;
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


@pytest.mark.parametrize(
    ("name", "source", "stage"),
    [
        ("points.geom", GLSL_GEOMETRY, "geometry"),
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
