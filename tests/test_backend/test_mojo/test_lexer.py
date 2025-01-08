import pytest
from typing import List
from crosstl.backend.Mojo.MojoLexer import MojoLexer


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = MojoLexer(code)
    return lexer.tokenize()


def test_mod_tokenization():
    code = """
        int a = 10 % 3;  // Basic modulus
    """
    tokens = tokenize_code(code)

    # Find the modulus operator in tokens
    has_mod = False
    for token in tokens:
        if token == ("MOD", "%"):
            has_mod = True
            break

    assert has_mod, "Modulus operator (%) not tokenized correctly"


def test_bitwise_not_tokenization():
    code = """
        int a = ~5;  // Bitwise NOT
    """
    tokens = tokenize_code(code)
    has_not = False
    for token in tokens:
        if token == ("BITWISE_NOT", "~"):
            has_not = True
            break
    assert has_not, "Bitwise NOT operator (~) not tokenized correctly"


if __name__ == "__main__":
    pytest.main()
