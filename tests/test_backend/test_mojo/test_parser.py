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


def test_mod_parsing():
    code = """
    fn main():
        let a: Int = 10 % 3  # Basic modulus
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Modulus operator parsing not implemented")


if __name__ == "__main__":
    pytest.main()
