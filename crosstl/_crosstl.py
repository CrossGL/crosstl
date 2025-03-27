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

# Import backend modules
# The imports below reference the class names that are re-exported in each backend's __init__.py
from .backend import DirectX, Metal, OpenGL, Slang, Vulkan, Mojo

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
        lexer = DirectX.HLSLLexer(shader_code)
        parser = DirectX.HLSLParser(lexer.tokenize())
    elif file_path.endswith(".metal"):
        lexer = Metal.MetalLexer(shader_code)
        parser = Metal.MetalParser(lexer.tokenize())
    elif file_path.endswith(".glsl"):
        lexer = OpenGL.GLSLLexer(shader_code)
        parser = OpenGL.GLSLParser(lexer.tokenize())
    elif file_path.endswith(".slang"):
        lexer = Slang.SlangLexer(shader_code)
        parser = Slang.SlangParser(lexer.tokenize())
    elif file_path.endswith(".spv"):
        lexer = Vulkan.VulkanLexer(shader_code)
        parser = Vulkan.VulkanParser(lexer.tokenize())
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
                codegen = DirectX.HLSLToCrossGLConverter()
            elif file_path.endswith(".metal"):
                codegen = Metal.MetalToCrossGLConverter()
            elif file_path.endswith(".glsl"):
                codegen = OpenGL.GLSLToCrossGLConverter()
            elif file_path.endswith(".slang"):
                codegen = Slang.SlangToCrossGLConverter()
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
