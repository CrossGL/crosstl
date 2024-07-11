import unittest
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.codegen import (
    directx_codegen,
    metal_codegen,
    vulkan_codegen,
    opengl_codegen,
)
import re


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


class TestCodeGeneration(unittest.TestCase):
    def setUp(self):
        self.code = "shader main { input vec3 position; output vec4 color; void main() { color = vec4(position, 1.0); } }"
        lexer = Lexer(self.code)
        parser = Parser(lexer.tokens)
        self.ast = parser.parse()

    def test_tokens(self):
        code = "shader main { input vec3 position; output vec4 color; void main() { color = vec4(position, 1.0); } }"
        lexer = Lexer(code)
        expected_tokens = [
            ("SHADER", "shader"),
            ("MAIN", "main"),
            ("LBRACE", "{"),
            ("INPUT", "input"),
            ("VECTOR", "vec3"),
            ("IDENTIFIER", "position"),
            ("SEMICOLON", ";"),
            ("OUTPUT", "output"),
            ("VECTOR", "vec4"),
            ("IDENTIFIER", "color"),
            ("SEMICOLON", ";"),
            ("VOID", "void"),
            ("MAIN", "main"),
            ("LPAREN", "("),
            ("RPAREN", ")"),
            ("LBRACE", "{"),
            ("IDENTIFIER", "color"),
            ("EQUALS", "="),
            ("VECTOR", "vec4"),
            ("LPAREN", "("),
            ("IDENTIFIER", "position"),
            ("COMMA", ","),
            ("NUMBER", "1.0"),
            ("RPAREN", ")"),
            ("SEMICOLON", ";"),
            ("RBRACE", "}"),
            ("RBRACE", "}"),
            ("EOF", None),
        ]
        for token in lexer.tokens:
            print(token)
        parser = Parser(lexer.tokens)
        ast = parser.parse()
        # print(ast)
        codegen = directx_codegen.HLSLCodeGen()
        hlsl_code = codegen.generate(ast)
        print(hlsl_code)

    def test_vulkan_codegen(self):
        codegen = vulkan_codegen.SPIRVCodeGen()
        spir_code = codegen.generate(self.ast)
        print(spir_code)
        expected_spir_code = """
            ; SPIR-V
            ; Version: 1.0
            ; Generator: Custom SPIR-V CodeGen
            ; Bound: 31
            ; Schema: 0
            OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
            OpMemoryModel Logical GLSL450
            OpEntryPoint Vertex %main "main" %position %color
            OpSource GLSL 450
            OpName %main "main"
            OpName %position "position"
            OpName %color "color"
            OpDecorate %position Location 0
            OpDecorate %color Location 1
            %void = OpTypeVoid
            %1 = OpTypeFunction %void
            %float = OpTypeFloat 32
            %vec3 = OpTypeVector %float 3
            %vec4 = OpTypeVector %float 4
            %_ptr_Output_vec4 = OpTypePointer Output %vec4
            %color = OpVariable %_ptr_Output_vec4 Output
            %_ptr_Input_vec3 = OpTypePointer Input %vec3
            %position = OpVariable %_ptr_Input_vec3 Input
            %float_1 = OpConstant %float 1
            %main = OpFunction %void None %3
            %8 = OpLabel
            %9 = OpLoad %v3float %position
            %10 = OpCompositeExtract %float %8 0
            %11 = OpCompositeExtract %float %9 1
            %12 = OpCompositeExtract %float %10 2
            %13 = OpCompositeConstruct %v4float %11 %12 %13 %float_1
            OpStore %color %13
            OpReturn
            OpFunctionEnd
         """
        self.assertEqual(
            normalize_whitespace(spir_code), normalize_whitespace(expected_spir_code)
        )
        print("Success: Vulkan codegen test passed")
        print("\n------------------\n")

    def test_opengl_codegen(self):
        codegen = opengl_codegen.GLSLCodeGen()
        glsl_code = codegen.generate(self.ast)
        print(glsl_code)

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
        print("Success: OpenGL codegen test passed")
        print("\n------------------\n")

    def test_metal_codegen(self):
        codegen = metal_codegen.MetalCodeGen()
        metal_code = codegen.generate(self.ast)
        print(metal_code)

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
        print("\n------------------\n")

    def test_directx_codegen(self):
        codegen = directx_codegen.HLSLCodeGen()
        hlsl_code = codegen.generate(self.ast)

        print(hlsl_code)

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
        print("\n------------------\n")


if __name__ == "__main__":
    unittest.main()
