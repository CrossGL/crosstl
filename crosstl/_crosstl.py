from . import translator
from .translator.lexer import Lexer
from .translator.parser import Parser
from .translator.codegen import (
    directx_codegen,
    metal_codegen,
    opengl_codegen,
    vulkan_codegen,
)
from .translator.ast import ASTNode
import argparse
import sys
import os

try:
    from .formatter import format_shader_code

    FORMATTER_AVAILABLE = True
except ImportError:
    FORMATTER_AVAILABLE = False


def translate(
    file_path: str,
    backend: str = "cgl",
    save_shader: str = None,
    format_output: bool = True,
) -> str:
    """Translate a shader file to another language.

    Args:
        file_path (str): The path to the shader file
        backend (str, optional): The target language to translate to. Defaults to "cgl".
        save_shader (str, optional): The path to save the translated shader. Defaults to None.
        format_output (bool, optional): Whether to format the generated code. Defaults to True.

    Returns:
        str: The translated shader code
    """
    backend = backend.lower()

    with open(file_path, "r") as file:
        shader_code = file.read()

    # Determine the input shader type based on the file extension
    if file_path.endswith(".cgl"):
        lexer = Lexer(shader_code)
        parser = Parser(lexer.tokens)
    elif file_path.endswith(".hlsl"):
        from .backend.DirectX import HLSLLexer, HLSLParser

        lexer = HLSLLexer(shader_code)
        parser = HLSLParser(lexer.tokenize())
    elif file_path.endswith(".metal"):
        from .backend.Metal import MetalLexer, MetalParser

        lexer = MetalLexer(shader_code)
        parser = MetalParser(lexer.tokenize())
    elif file_path.endswith(".glsl"):
        from .backend.Opengl import GLSLLexer, GLSLParser

        lexer = GLSLLexer(shader_code)
        parser = GLSLParser(lexer.tokenize())
    elif file_path.endswith(".slang"):
        from .backend.slang import SlangLexer, SlangParser

        lexer = SlangLexer(shader_code)
        parser = SlangParser(lexer.tokenize())
    elif file_path.endswith(".spv"):
        from .backend.Vulkan import VulkanLexer, VulkanParser

        lexer = VulkanLexer(shader_code)
        parser = VulkanParser(lexer.tokenize())
    else:
        raise ValueError(f"Unsupported shader file type: {file_path}")

    ast = parser.parse()

    if file_path.endswith(".cgl"):
        if backend == "metal":
            codegen = metal_codegen.MetalCodeGen()
        elif backend == "directx":
            codegen = directx_codegen.HLSLCodeGen()
        elif backend == "opengl":
            codegen = opengl_codegen.GLSLCodeGen()
        elif backend == "vulkan":
            codegen = vulkan_codegen.VulkanSPIRVCodeGen()
        else:
            raise ValueError(f"Unsupported backend for CrossGL file: {backend}")
    else:
        if backend == "cgl":
            if file_path.endswith(".hlsl"):
                from .backend.DirectX.DirectxCrossGLCodeGen import (
                    HLSLToCrossGLConverter,
                )

                codegen = HLSLToCrossGLConverter()
            elif file_path.endswith(".metal"):
                from .backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter

                codegen = MetalToCrossGLConverter()
            elif file_path.endswith(".glsl"):
                from .backend.Opengl.openglCrossglCodegen import GLSLToCrossGLConverter

                codegen = GLSLToCrossGLConverter()
            elif file_path.endswith(".slang"):
                from .backend.slang.SlangCrossGLCodeGen import SlangToCrossGLConverter

                codegen = SlangToCrossGLConverter()
            else:
                raise ValueError(f"Reverse translation not supported for: {file_path}")
        else:
            raise ValueError(
                f"Unsupported translation scenario: {file_path} to {backend}"
            )

    # Generate the code
    generated_code = codegen.generate(ast)

    # Format the code if requested and the formatter is available
    if format_output and FORMATTER_AVAILABLE:
        generated_code = format_shader_code(generated_code, backend, save_shader)

    # Write to the file if a path is provided
    if save_shader is not None:
        with open(save_shader, "w") as file:
            file.write(generated_code)

    return generated_code


def main():
    """Command-line entry point for CrossGL translation."""
    parser = argparse.ArgumentParser(description="CrossGL Shader Translator")

    parser.add_argument("input", help="Input shader file path")
    parser.add_argument(
        "--backend",
        "-b",
        default="cgl",
        help="Target backend (metal, directx, opengl, vulkan, cgl)",
    )
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument(
        "--no-format", action="store_true", help="Disable code formatting"
    )

    args = parser.parse_args()

    try:
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found")
            return 1

        # Determine output path if not specified
        output_path = args.output
        if not output_path:
            base, _ = os.path.splitext(args.input)
            ext_map = {
                "metal": ".metal",
                "directx": ".hlsl",
                "opengl": ".glsl",
                "vulkan": ".spirv",
                "cgl": ".cgl",
            }
            output_path = base + ext_map.get(args.backend, ".out")

        # Perform translation
        code = translate(
            args.input,
            backend=args.backend,
            save_shader=output_path,
            format_output=not args.no_format,
        )

        print(f"Successfully translated to {output_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
