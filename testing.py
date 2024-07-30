import unittest
from src.translator.lexer import Lexer
from src.translator.parser import Parser
from src.translator.codegen import (
    directx_codegen,
    metal_codegen,
    vulkan_codegen,
    opengl_codegen,
)
from src.translator.ast import ASTNode
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

shader PerlinNoise {

// 1. Perlin Noise Terrain Generation.

vertex {

input vec3 position;
output vec2 vUV;

void main()
{
    vUV = position.xy * 10.0;
    gl_Position = vec4(position, 1.0);
}

}

float perlinNoise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

fragment {

input vec2 vUV;
output vec4 fragColor;


void main()
{
    float noise = perlinNoise(vUV);
    float height = noise * 10.0;
    vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
    fragColor = vec4(color, 1.0);
}


}

}

"""
        lexer = Lexer(self.code)
        parser = Parser(lexer.tokens)
        self.ast = parser.parse()
        self.hlsl_codegen = directx_codegen.HLSLCodeGen()
        self.metal_codegen = metal_codegen.MetalCodeGen()

    def test_opengl_codegen(self):
        codegen = opengl_codegen.GLSLCodeGen()
        opengl_code = codegen.generate(self.ast)
        print(opengl_code)

        print("Success: OpenGL codegen test passed")
        print("\n------------------\n")

    def test_metal_codegen(self):
        codegen = metal_codegen.MetalCodeGen()
        metal_code = codegen.generate(self.ast)
        print(metal_code)

        print("Success: Metal codegen test passed")
        print("\n------------------\n")

    def test_directx_codegen(self):
        codegen = directx_codegen.HLSLCodeGen()
        hlsl_code = codegen.generate(self.ast)

        print(hlsl_code)

        print("Success: DirectX codegen test passed")
        print("\n------------------\n")


if __name__ == "__main__":
    unittest.main()
