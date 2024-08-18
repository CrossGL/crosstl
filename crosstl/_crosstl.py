from .src import translator
from .src.translator.lexer import Lexer
from .src.translator.parser import Parser
from .src.translator.codegen import directx_codegen, metal_codegen, opengl_codegen
from .src.translator.ast import ASTNode
from .src.backend.DirectX import *
from .src.backend.Metal import *
from .src.backend.Opengl import *


def translate(file_path: str, backend: str = "cgl", save_shader: str = None) -> str:
    backend = backend.lower()

    with open(file_path, "r") as file:
        shader_code = file.read()

    # Determine the input shader type based on the file extension
    if file_path.endswith(".cgl"):
        lexer = Lexer(shader_code)
        parser = Parser(lexer.tokens)
    elif file_path.endswith(".hlsl"):
        lexer = HLSLLexer(shader_code)
        parser = HLSLParser(lexer.tokens)
    elif file_path.endswith(".metal"):
        lexer = MetalLexer(shader_code)
        parser = MetalParser(lexer.tokens)
    elif file_path.endswith(".glsl"):
        lexer = GLSLLexer(shader_code)
        parser = GLSLParser(lexer.tokens)
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
