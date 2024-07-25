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
    shader nestedLoopsShader {
    input vec3 position;
    input vec2 texCoord;
    input vec3 normal;
    output vec4 fragColor;

    uniform sampler2D texture;
    uniform vec3 lightPos;
    uniform vec3 viewPos;
    uniform vec3 ambientColor;
    uniform float shininess;
    uniform float specularIntensity;
    uniform int outerIterations;
    uniform int innerIterations;

    vec3 calculateSpecular(vec3 normal, vec3 lightDir, vec3 viewDir) {
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
        return spec * vec3(1.0, 1.0, 1.0) * specularIntensity;
    }

    void main() {
        vec4 texColor = texture(texture, texCoord);
        vec3 lightDir = normalize(lightPos - position);
        vec3 viewDir = normalize(viewPos - position);

        vec3 ambient = ambientColor;
        vec3 diffuse = texColor.rgb;
        vec3 specular = vec3(0.0);

        // Outer loop
        for (int i = 0; i < outerIterations; i++) {
            // Inner loop
            for (int j = 0; j < innerIterations; j++) {
                // Compute lighting
                vec3 tempSpecular = calculateSpecular(normal, lightDirModified, viewDirModified);
                specular += tempSpecular;
            }

            // Reduce diffuse intensity progressively
            diffuse *= 0.9;
        }

        vec3 lighting = ambient + diffuse + specular;
        fragColor = vec4(lighting, 1.0);
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
        codegen.generate(self.ast)
        # print(glsl_code)

        print("Success: OpenGL codegen test passed")
        print("\n------------------\n")

    def test_metal_codegen(self):
        codegen = metal_codegen.MetalCodeGen()
        codegen.generate(self.ast)
        # print(metal_code)

        print("Success: Metal codegen test passed")
        print("\n------------------\n")

    def test_directx_codegen(self):
        codegen = directx_codegen.HLSLCodeGen()
        codegen.generate(self.ast)

        # print(hlsl_code)

        print("Success: DirectX codegen test passed")
        print("\n------------------\n")


if __name__ == "__main__":
    unittest.main()
