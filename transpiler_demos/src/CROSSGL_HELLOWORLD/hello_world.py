import argparse
import os
from compiler.codegen import (
    directx_codegen,
    metal_codegen,
    opengl_codegen,
    vulkan_codegen,
)
from compiler.lexer import Lexer
from compiler.parser import Parser


class BackendParser:
    def __init__(self, content, backend):
        self.content = content
        self.backend = backend

    def parse(self):
        # Implement parsing logic here
        if self.backend == "metal":
            return self.parse_metal()
        elif self.backend == "directx":
            return self.parse_directx()
        elif self.backend == "opengl":
            return self.parse_opengl()
        elif self.backend == "vulkan":
            return self.parse_vulkan()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

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


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="CGL to Backend Transpiler")
    parser.add_argument(
        "cgl_file",
        nargs="?",
        default="hello_world.cgl",
        help="Path to the CGL file (default: hello_world.cgl)",
    )
    parser.add_argument(
        "--backend",
        choices=["metal", "directx", "opengl", "vulkan"],
        required=True,
        help="Specify the backend",
    )
    parser.add_argument(
        "--output",
        nargs="?",
        default=None,
        help="Optional output file (default: same as input file with backend-specific extension)",
    )

    args = parser.parse_args()

    # Read the content of the .cgl file
    cgl_path = os.path.join("transpiler_demos/src/CROSSGL_HELLOWORLD", args.cgl_file)
    with open(cgl_path, "r") as file:
        content = file.read()

    # Initialize the parser with the file content and backend
    backend_parser = BackendParser(content, args.backend)

    # Parse the content
    parsed_content = backend_parser.parse()

    # Determine output file name and extension
    if args.output:
        output_path = args.output
    else:
        base_name, _ = os.path.splitext(cgl_path)
        extensions = {
            "metal": ".metal",
            "directx": ".hlsl",
            "opengl": ".glsl",
            "vulkan": ".spv",
        }
        output_path = base_name + extensions[args.backend]

    # Write the parsed content to the output file
    with open(output_path, "w") as output_file:
        output_file.write(parsed_content)

    # Print the output path
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
