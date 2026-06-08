import textwrap

from crosstl.backend.DirectX.DirectxCrossGLCodeGen import HLSLToCrossGLConverter
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser


def generate_crossgl(code: str) -> str:
    tokens = HLSLLexer(code).tokenize()
    ast = HLSLParser(tokens).parse()
    return HLSLToCrossGLConverter().generate(ast)


def parse_crossgl(code: str):
    tokens = CrossGLLexer(code).get_tokens()
    return CrossGLParser(tokens).parse()


def test_vk_push_constant_struct_global_reparses():
    # Source shape: SaschaWillems/Vulkan samples conditional_rendering model.vert.hlsl
    hlsl = textwrap.dedent("""
        struct UBO
        {
            float4x4 projection;
            float4x4 view;
        };
        [[vk::binding(0, 0)]]
        ConstantBuffer<UBO> ubo : register(b0);

        struct PushConstants
        {
            float4x4 model;
            float4 color;
        };
        [[vk::push_constant]] PushConstants push_constants;

        struct VSInput
        {
            [[vk::location(0)]] float3 Pos : POSITION0;
            [[vk::location(1)]] float3 Normal : NORMAL0;
        };

        struct VSOutput
        {
            float4 Pos : SV_POSITION;
            [[vk::location(0)]] float3 Color : COLOR0;
        };

        VSOutput main(VSInput input)
        {
            VSOutput output = (VSOutput) 0;
            float4 localPos = mul(ubo.view, mul(push_constants.model, float4(input.Pos, 1.0)));
            output.Color = push_constants.color.rgb;
            output.Pos = mul(ubo.projection, localPos);
            return output;
        }
    """).strip()

    crossgl = generate_crossgl(hlsl)

    assert "@ vk::push_constant" in crossgl
    assert "var PushConstants push_constants;" in crossgl
    parse_crossgl(crossgl)
