from crosstl.backend.Mojo import MojoLexer
from crosstl.backend.Mojo import MojoParser
import pytest
from typing import List


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    pass


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = MojoLexer(code)
    return lexer.tokens


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST."""
    parser = MojoParser(tokens)
    return parser.parse()

# ToDO: Implement the tests
def test_struct():
    pass


if __name__ == "__main__":
    pytest.main()
