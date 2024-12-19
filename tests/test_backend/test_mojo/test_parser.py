import pytest
from typing import List
from crosstl.backend.Mojo.MojoLexer import MojoLexer
from crosstl.backend.Mojo import MojoParser


def parse_code(tokens: List):
    """Test the parser
    Args:
        tokens (List): The list of tokens generated from the lexer

    Returns:
        ASTNode: The abstract syntax tree generated from the code
    """
    parser = MojoParser(tokens)
    return parser.parse()


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = MojoLexer(code)
    return lexer.tokens


# ToDO: Implement the tests
def test_struct():
    pass


if __name__ == "__main__":
    pytest.main()
