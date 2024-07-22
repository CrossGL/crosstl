from compiler.codegen import (
    directx_codegen,
    metal_codegen,
    opengl_codegen,
    vulkan_codegen,
)
from compiler.lexer import Lexer
from compiler.parser import Parser


class transpiler:
    def __init__(self, content, backend):
        self.content = content
        self.backend = backend
        # Implement parsing logic here

    def __repr__(self):
        return self.transpile()

    def transpile(self):
        if self.backend == "metal":
            return self.parse_metal()
        elif self.backend == "directx":
            return self.parse_directx()
        elif self.backend == "opengl":
            return self.parse_opengl()
        elif self.backend == "vulkan":
            return self.parse_vulkan()
        else:
            raise ValueError(
                f"Unknown backend: {self.backend} the supported backends are metal, directx, opengl, vulkan"
            )

    def parse_metal(self):
        code = self.content
        lexer = Lexer(code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = metal_codegen.MetalCodeGen()
        metal_code = codegen.generate(ast)
        return f"{metal_code}"

    def parse_directx(self):
        code = self.content
        lexer = Lexer(code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = directx_codegen.HLSLCodeGen()
        hlsl_code = codegen.generate(ast)
        return f"{hlsl_code}"

    def parse_opengl(self):
        code = self.content
        lexer = Lexer(code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = opengl_codegen.GLSLCodeGen()
        glsl_code = codegen.generate(ast)
        return f"{glsl_code}"

    def parse_vulkan(self):
        code = self.content
        lexer = Lexer(code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = vulkan_codegen.VulkanSPIRVCodeGen()
        vulkan_code = codegen.generate(ast)
        return f"{vulkan_code}"
