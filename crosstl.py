from compiler.codegen import (
    directx_codegen,
    metal_codegen,
    opengl_codegen,
    vulkan_codegen,
)
from compiler.lexer import Lexer
from compiler.parser import Parser
import os


class Transpiler:
    """
    A class used to transpile CGL code to various backend-specific shader languages.

    Attributes
    ----------
    content : str
        The CGL code content to be transpiled or a path to a CGL file.
    backend : str
        The target backend for the transpilation. Supported values are "metal", "directx", "opengl", "vulkan".

    Methods
    -------
    transpile()
        Transpiles the CGL code to the specified backend.
    parse_metal()
        Parses the CGL code and generates Metal shader code.
    parse_directx()
        Parses the CGL code and generates DirectX (HLSL) shader code.
    parse_opengl()
        Parses the CGL code and generates OpenGL (GLSL) shader code.
    parse_vulkan()
        Parses the CGL code and generates Vulkan (SPIR-V) shader code.
    """

    def __init__(self, content, backend):
        """
        Constructs the necessary attributes for the Transpiler object.

        Parameters
        ----------
        content : str
            The CGL code content to be transpiled or a path to a CGL file.
        backend : str
            The target backend for the transpilation. Supported values are "metal", "directx", "opengl", "vulkan".
        """
        self.content = content
        self.backend = backend

    def __repr__(self):
        return self.transpile()

    def transpile(self):
        """
        Transpiles the CGL code to the specified backend.

        Returns
        -------
        str
            The transpiled shader code for the specified backend.
        """
        # Check if content is a file path
        if os.path.isfile(self.content):
            with open(self.content, "r") as file:
                code = file.read()
        else:
            code = self.content

        if self.backend == "metal":
            transpiled_code = self.parse_metal(code)
        elif self.backend == "directx":
            transpiled_code = self.parse_directx(code)
        elif self.backend == "opengl":
            transpiled_code = self.parse_opengl(code)
        elif self.backend == "vulkan":
            transpiled_code = self.parse_vulkan(code)
        else:
            raise ValueError(
                f"Unknown backend: {self.backend}. Supported backends are: metal, directx, opengl, vulkan."
            )

        # If input was a file, save the transpiled code to the corresponding backend output file
        if os.path.isfile(self.content):
            base_name, _ = os.path.splitext(self.content)
            extensions = {
                "metal": ".metal",
                "directx": ".hlsl",
                "opengl": ".glsl",
                "vulkan": ".spv",
            }
            output_path = base_name + extensions[self.backend]
            with open(output_path, "w") as output_file:
                output_file.write(transpiled_code)
            print(f"Transpiled code written to {output_path}")
            return output_path

        return transpiled_code

    def parse_metal(self, code):
        """
        Parses the CGL code and generates Metal shader code.

        Parameters
        ----------
        code : str
            The CGL code content to be parsed.

        Returns
        -------
        str
            The generated Metal shader code.
        """
        lexer = Lexer(code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = metal_codegen.MetalCodeGen()
        metal_code = codegen.generate(ast)
        return f"{metal_code}"

    def parse_directx(self, code):
        """
        Parses the CGL code and generates DirectX (HLSL) shader code.

        Parameters
        ----------
        code : str
            The CGL code content to be parsed.

        Returns
        -------
        str
            The generated DirectX (HLSL) shader code.
        """
        lexer = Lexer(code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = directx_codegen.HLSLCodeGen()
        hlsl_code = codegen.generate(ast)
        return f"{hlsl_code}"

    def parse_opengl(self, code):
        """
        Parses the CGL code and generates OpenGL (GLSL) shader code.

        Parameters
        ----------
        code : str
            The CGL code content to be parsed.

        Returns
        -------
        str
            The generated OpenGL (GLSL) shader code.
        """
        lexer = Lexer(code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = opengl_codegen.GLSLCodeGen()
        glsl_code = codegen.generate(ast)
        return f"{glsl_code}"

    def parse_vulkan(self, code):
        """
        Parses the CGL code and generates Vulkan (SPIR-V) shader code.

        Parameters
        ----------
        code : str
            The CGL code content to be parsed.

        Returns
        -------
        str
            The generated Vulkan (SPIR-V) shader code.
        """
        lexer = Lexer(code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = vulkan_codegen.VulkanSPIRVCodeGen()
        vulkan_code = codegen.generate(ast)
        return f"{vulkan_code}"
