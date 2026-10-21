"""OpenGL refuses entry interfaces it cannot lower instead of emitting invalid GLSL."""

import pytest

import crosstl
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.glsl_hlsl_resources import HlslRegisterOverlapError
from crosstl.translator.codegen.glsl_refusals import OpenGLLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

STAGELESS_STRUCT_ENTRY = """
shader main {
    struct VSInput {
        vec2 Pos @ Position;
    }
    struct VSOutput {
        vec4 Pos @ gl_Position;
    }
    VSOutput main(VSInput input) {
        VSOutput output = VSOutput(vec4(0.0));
        output.Pos = vec4(input.Pos, 0.0, 1.0);
        return output;
    }
}
"""

STAGELESS_VOID_ENTRY = """
shader main {
    float scale(float value) {
        return value * 2.0;
    }
    void main() {
        float doubled = scale(1.0);
    }
}
"""


def _glsl(source):
    ast = Parser(Lexer(source).get_tokens()).parse()
    return GLSLCodeGen().generate(ast)


def test_stageless_entry_with_struct_interface_is_refused():
    with pytest.raises(OpenGLLoweringRefused, match="has no shader stage"):
        _glsl(STAGELESS_STRUCT_ENTRY)


def test_stageless_void_main_is_still_emitted():
    glsl = _glsl(STAGELESS_VOID_ENTRY)

    assert "void main()" in glsl
    assert "float doubled = scale(1.0);" in glsl


def test_hlsl_entry_without_stage_is_refused(tmp_path):
    # No semantic names a stage: neither the result nor the parameter is a
    # system value or a structure of stage semantics.
    source = tmp_path / "stageless.hlsl"
    source.write_text("""
struct Result { float4 Value : RESULT0; };
Result main(float2 Pos : POSITION0)
{
    Result output = (Result)0;
    output.Value = float4(Pos, 0.0, 1.0);
    return output;
}
""")

    with pytest.raises(OpenGLLoweringRefused) as refusal:
        crosstl.translate(str(source), backend="opengl", source_backend="directx")

    assert refusal.value.project_diagnostic_code.endswith("opengl-lowering-unsupported")


def test_hlsl_entry_returning_a_position_structure_is_a_vertex_stage(tmp_path):
    # Only a vertex or domain shader returns SV_Position, and a domain shader
    # carries [domain]; the returned structure names the vertex stage.
    source = tmp_path / "vertex.hlsl"
    source.write_text("""
struct VSInput { float2 Pos : POSITION0; };
struct VSOutput { float4 Pos : SV_POSITION; };
VSOutput main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    output.Pos = float4(input.Pos, 0.0, 1.0);
    return output;
}
""")

    glsl = crosstl.translate(str(source), backend="opengl", source_backend="directx")

    assert "layout(location = 0) in vec2 Pos;" in glsl
    assert "gl_Position = vec4(Pos, 0.0, 1.0);" in glsl


def test_hlsl_domain_patch_parameter_is_refused(tmp_path):
    source = tmp_path / "domain.hlsl"
    source.write_text("""
struct HSOutput { float4 Pos : POSITION; };
struct ConstantsHSOutput
{
    float TessLevelOuter[3] : SV_TessFactor;
    float TessLevelInner : SV_InsideTessFactor;
};
struct DSOutput { float4 Pos : SV_POSITION; };

[domain("tri")]
DSOutput main(ConstantsHSOutput input, float3 TessCoord : SV_DomainLocation,
              const OutputPatch<HSOutput, 3> patch)
{
    DSOutput output = (DSOutput)0;
    output.Pos = TessCoord.x * patch[0].Pos + TessCoord.y * patch[1].Pos
        + TessCoord.z * patch[2].Pos;
    return output;
}
""")

    with pytest.raises(OpenGLLoweringRefused, match="patch parameter 'patch'"):
        crosstl.translate(str(source), backend="opengl", source_backend="directx")


def test_overlapping_hlsl_registers_raise_a_typed_diagnostic(tmp_path):
    source = tmp_path / "overlap.hlsl"
    source.write_text("""
Texture2D textures[4] : register(t1);
StructuredBuffer<float4> values : register(t2);
SamplerState linearSampler : register(s0);
RWStructuredBuffer<float4> results : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    results[0] = values[0] + textures[1].SampleLevel(linearSampler, float2(0.5, 0.5), 0);
}
""")

    with pytest.raises(
        HlslRegisterOverlapError, match="register t 2 overlaps"
    ) as error:
        crosstl.translate(str(source), backend="opengl", source_backend="directx")

    assert error.value.project_diagnostic_code == (
        "project.translate.hlsl-register-overlap"
    )
