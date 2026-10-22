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

    assert "@ push_constant" in crossgl
    assert "ConstantBuffer<PushConstants> push_constants;" in crossgl
    parse_crossgl(crossgl)


def test_fixed_cbuffer_register_is_reserved_before_push_constants(tmp_path):
    # Source shape: SaschaWillems/Vulkan hlsl/pushconstants/pushconstants.vert. The push-constant block has no
    # register, so it must not take b0 from the cbuffer declared after it.
    import re

    import crosstl

    source = tmp_path / "shader.vert.hlsl"
    source.write_text(textwrap.dedent("""
            struct UBO { float4x4 projection; };
            cbuffer ubo : register(b0) { UBO ubo; }
            struct PushConsts { float4 color; };
            [[vk::push_constant]] PushConsts pushConsts;
            struct VSOutput { float4 Pos : SV_POSITION; };
            VSOutput main(float3 pos : POSITION0)
            {
                VSOutput output = (VSOutput)0;
                output.Pos = mul(ubo.projection, float4(pos + pushConsts.color.xyz, 1.0));
                return output;
            }
        """))

    hlsl = crosstl.translate(str(source), backend="directx", format_output=False)

    assert re.search(r"cbuffer ubo : register\(b0\)", hlsl)
    push = re.search(r"pushConsts : register\(b(\d+)\)", hlsl)
    assert push is not None and push.group(1) != "0"
