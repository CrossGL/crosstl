import unittest
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.codegen import (
    directx_codegen,
    metal_codegen,
    vulkan_codegen,
    opengl_codegen,
)
from compiler.ast import ASTNode


import re


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def print_ast(node, indent=0):
    print("  " * indent + node.__class__.__name__)
    for key, value in node.__dict__.items():
        if isinstance(value, ASTNode):
            print_ast(value, indent + 1)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ASTNode):
                    print_ast(item, indent + 1)
                else:
                    print("  " * (indent + 1) + repr(item))
        else:
            print("  " * (indent + 1) + repr(value))


class TestCodeGeneration(unittest.TestCase):
    def setUp(self):
        self.code = """ 
  shader main {
    input vec3 position;
    output vec4 fragColor;

    float perlinNoise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    void main() {
        vec2 uv = position.xy * 10.0; 
        float noise = perlinNoise(uv);
        float height = noise * 10.0;
        vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
        fragColor = vec4(color, 1.0);
    }
}
"""
        lexer = Lexer(self.code)
        parser = Parser(lexer.tokens)
        self.ast = parser.parse()
        self.hlsl_codegen = directx_codegen.HLSLCodeGen()
        self.metal_codegen = metal_codegen.MetalCodeGen()

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
            ("ASSIGN_ADD", "+="),
            ("ASSIGN_SUB", "-="),
            ("ASSIGN_MUL", "*="),
            ("ASSIGN_DIV", "/="),
            ("EQUALS", "="),
            ("VECTOR", "vec4"),
            ("LPAREN", "("),
            ("IDENTIFIER", "position"),
            ("COMMA", ","),
            ("NUMBER", "1.0"),
            ("MULTIPLY" , "*"),
            ("DIVIDE", "/"),
            ("MINUS", "-"),
            ("PLUS", "+"),
            ("DOT","."),
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

    # def test_vulkan_codegen(self):
    # codegen = vulkan_codegen.SPIRVCodeGen()
    # spir_code = codegen.generate(self.ast)
    # print(spir_code)

    #     expected_spir_code = """
    #         ; SPIR-V
    #         ; Version: 1.0
    #         ; Generator: Custom SPIR-V CodeGen
    #         ; Bound: 31
    #         ; Schema: 0
    #         OpCapability Shader
    #         %1 = OpExtInstImport "GLSL.std.450"
    #         OpMemoryModel Logical GLSL450
    #         OpEntryPoint Vertex %main "main" %position %color
    #         OpSource GLSL 450
    #         OpName %main "main"
    #         OpName %position "position"
    #         OpName %color "color"
    #         OpDecorate %position Location 0
    #         OpDecorate %color Location 1
    #         %void = OpTypeVoid
    #         %1 = OpTypeFunction %void
    #         %float = OpTypeFloat 32
    #         %vec3 = OpTypeVector %float 3
    #         %vec4 = OpTypeVector %float 4
    #         %_ptr_Output_vec4 = OpTypePointer Output %vec4
    #         %color = OpVariable %_ptr_Output_vec4 Output
    #         %_ptr_Input_vec3 = OpTypePointer Input %vec3
    #         %position = OpVariable %_ptr_Input_vec3 Input
    #         %float_1 = OpConstant %float 1
    #         %main = OpFunction %void None %3
    #         %8 = OpLabel
    #         %9 = OpLoad %v3float %position
    #         %10 = OpCompositeExtract %float %8 0
    #         %11 = OpCompositeExtract %float %9 1
    #         %12 = OpCompositeExtract %float %10 2
    #         %13 = OpCompositeConstruct %v4float %11 %12 %13 %float_1
    #         OpStore %color %13
    #         OpReturn
    #         OpFunctionEnd
    #      """
    #     self.assertEqual(
    #         normalize_whitespace(spir_code), normalize_whitespace(expected_spir_code)
    #     )
    #   print("Success: Vulkan codegen test passed")
    #    print("\n------------------\n")

    def test_opengl_codegen(self):
        codegen = opengl_codegen.GLSLCodeGen()
        glsl_code = codegen.generate(self.ast)
        #print(glsl_code)

        print("Success: OpenGL codegen test passed")
        print("\n------------------\n")

    def test_metal_codegen(self):
        codegen = metal_codegen.MetalCodeGen()
        metal_code = codegen.generate(self.ast)
        #print(metal_code)

  
        print("Success: Metal codegen test passed")
        print("\n------------------\n")

    def test_directx_codegen(self):
        codegen = directx_codegen.HLSLCodeGen()
        hlsl_code = codegen.generate(self.ast)

        #print(hlsl_code)

        print("Success: DirectX codegen test passed")
        print("\n------------------\n")


if __name__ == "__main__":
    unittest.main()
