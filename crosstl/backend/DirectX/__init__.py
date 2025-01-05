from .DirectxLexer import HLSLLexer
from .DirectxParser import HLSLParser
from .DirectxCrossGLCodeGen import HLSLToCrossGLConverter
def process_shader(shader_code):
    lexer = HLSLLexer(shader_code)
    tokens = lexer.tokenize()
    parser = HLSLParser(tokens)
    ast = parser.parse()
    converter = HLSLToCrossGLConverter()
    return converter.convert(ast)

