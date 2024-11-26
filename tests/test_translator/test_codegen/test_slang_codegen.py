from crosstl.src.translator.lexer import Lexer
import pytest
from typing import List
from crosstl.src.translator.parser import Parser
from crosstl.src.translator.codegen.slang_codegen import SlangCodeGen


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = Lexer(code)
    return lexer.tokens


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST.

    Args:
        tokens (List): The list of tokens to parse
    Returns:
        AST: The abstract syntax tree generated from the parser
    """
    parser = Parser(tokens)
    return parser.parse()


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = SlangCodeGen()
    return codegen.generate(ast_node)

# ToDO: Implement the tests
def test_struct():
    pass


if __name__ == "__main__":
    pytest.main()
