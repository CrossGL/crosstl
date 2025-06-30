from . import lexer
from . import parser
from . import codegen
from .lexer import Lexer
from .parser import Parser


def parse(shader_code):
    """Parse shader code and return the AST.

    Args:
        shader_code (str): The shader code to parse

    Returns:
        The abstract syntax tree
    """
    lexer = Lexer(shader_code)
    tokens = lexer.tokens
    parser = Parser(tokens)
    return parser.parse()
