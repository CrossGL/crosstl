from .DirectxLexer import HLSLLexer
from .DirectxParser import HLSLParser
from .DirectxCrossGLCodeGen import HLSLToCrossGLConverter


def process_shader(shader_code):
    """
    Process an HLSL shader code through the DirectX pipeline:
    1. Preprocess the code.
    2. Tokenize with the lexer.
    3. Parse with the parser.
    4. Convert to CrossGL using the code generator."""
    lexer = HLSLLexer(shader_code)
    tokens = lexer.tokenize()
    parser = HLSLParser(tokens)
    ast = parser.parse()
    converter = HLSLToCrossGLConverter()
    return converter.convert(ast)
