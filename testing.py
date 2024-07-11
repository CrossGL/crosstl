import unittest
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.codegen import directx_codegen, metal_codegen, vulkan_codegen
import re


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


class TestCodeGeneration(unittest.TestCase):
    def setUp(self):
        self.code = "shader main { input vec3 position; output vec4 color; void main() { color = vec4(position, 1.0); } }"
        lexer = Lexer(self.code)
        parser = Parser(lexer.tokens)
        self.ast = parser.parse()

    def test_vulkan_codegen(self):
        codegen = vulkan_codegen.GLSLCodeGen()
        glsl_code = codegen.generate(self.ast)

        expected_glsl_code = """
        #version 450

        layout(location = 0) in vec3 position;
        layout(location = 0) out vec4 color;

        void main() {
            color = vec4(position, 1.0);
        }
        """
        self.assertEqual(
            normalize_whitespace(glsl_code), normalize_whitespace(expected_glsl_code)
        )
        print("Success: Vulkan codegen test passed")

    def test_metal_codegen(self):
        codegen = metal_codegen.MetalCodeGen()
        metal_code = codegen.generate(self.ast)

        expected_metal_code = """
        #include <metal_stdlib>
        using namespace metal;

        struct VertexInput {
            float3 position [[attribute(0)]];
        };

        struct FragmentOutput {
            float4 color [[color(0)]];
        };

        fragment FragmentOutput main(VertexInput input [[stage_in]]) {
            FragmentOutput output;
            color = vec4(position, 1.0);
            return output;
        }
        """
        self.assertEqual(
            normalize_whitespace(metal_code), normalize_whitespace(expected_metal_code)
        )
        print("Success: Metal codegen test passed")

    def test_directx_codegen(self):
        codegen = directx_codegen.HLSLCodeGen()
        hlsl_code = codegen.generate(self.ast)

        expected_hlsl_code = """
        struct VS_INPUT {
            float3 position : POSITION;
        };

        struct PS_OUTPUT {
            float4 color : SV_TARGET;
        };

        void main(VS_INPUT input) {
            color = vec4(position, 1.0);
            return;
        }
        """
        self.assertEqual(
            normalize_whitespace(hlsl_code),
            normalize_whitespace(expected_hlsl_code),
        )
        print("Success: DirectX codegen test passed")


if __name__ == "__main__":
    unittest.main()
