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

# Import backend modules with careful handling of case sensitivity
try:
    # Import each needed symbol explicitly to avoid issues with case sensitivity
    from .backend.DirectX.HLSLLexer import HLSLLexer
    from .backend.DirectX.HLSLParser import HLSLParser
    from .backend.DirectX.DirectXCrossGLCodeGen import HLSLToCrossGLConverter

    from .backend.Metal.MetalLexer import MetalLexer
    from .backend.Metal.MetalParser import MetalParser
    from .backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter

    from .backend.OpenGL.OpenglLexer import GLSLLexer
    from .backend.OpenGL.OpenglParser import GLSLParser
    from .backend.OpenGL.OpenGLCrossGLCodeGen import GLSLToCrossGLConverter

    from .backend.Slang.SlangLexer import SlangLexer
    from .backend.Slang.SlangParser import SlangParser
    from .backend.Slang.SlangCrossGLCodeGen import SlangToCrossGLConverter

    from .backend.Vulkan.VulkanLexer import VulkanLexer
    from .backend.Vulkan.VulkanParser import VulkanParser
except ImportError as e:
    # Log the import error to help with debugging
    import sys

    print(f"Import error in _crosstl.py: {e}", file=sys.stderr)
    raise


def translate(file_path: str, backend: str = "cgl", save_shader: str = None) -> str:
    """Translate a shader file to another language.

    Args:
        file_path (str): The path to the shader file
        backend (str, optional): The target language to translate to. Defaults to "cgl".
        save_shader (str, optional): The path to save the translated shader. Defaults to None.

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
        lexer = HLSLLexer(shader_code)
        parser = HLSLParser(lexer.tokenize())
    elif file_path.endswith(".metal"):
        lexer = MetalLexer(shader_code)
        parser = MetalParser(lexer.tokenize())
    elif file_path.endswith(".glsl"):
        lexer = GLSLLexer(shader_code)
        parser = GLSLParser(lexer.tokenize())
    elif file_path.endswith(".slang"):
        lexer = SlangLexer(shader_code)
        parser = SlangParser(lexer.tokenize())
    elif file_path.endswith(".spv"):
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
                codegen = HLSLToCrossGLConverter()
            elif file_path.endswith(".metal"):
                codegen = MetalToCrossGLConverter()
            elif file_path.endswith(".glsl"):
                codegen = GLSLToCrossGLConverter()
            elif file_path.endswith(".slang"):
                codegen = SlangToCrossGLConverter()
            else:
                raise ValueError(f"Reverse translation not supported for: {file_path}")
        else:
            raise ValueError(
                f"Unsupported translation scenario: {file_path} to {backend}"
            )

    # Generate the code and write to the file
    generated_code = codegen.generate(ast)

    if save_shader is not None:
        with open(save_shader, "w") as file:
            file.write(generated_code)

    return generated_code
