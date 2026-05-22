import pytest
from typing import List
from crosstl.backend.SPIRV.VulkanLexer import VulkanLexer


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = VulkanLexer(code)
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


def test_array_bracket_tokenization():
    code = """
        mat4 transforms[64];
    """
    tokens = tokenize_code(code)

    assert ("LBRACKET", "[") in tokens
    assert ("RBRACKET", "]") in tokens


def test_array_postfix_update_tokenization():
    tokens = tokenize_code("items[i]++; values[index]--;")

    assert ("POST_INCREMENT", "++") in tokens
    assert ("POST_DECREMENT", "--") in tokens
    assert ("PLUS", "+") not in tokens
    assert ("MINUS", "-") not in tokens


def test_preprocessor_version_directive_tokenization():
    tokens = tokenize_code(
        """
        #version 450
        void main() {}
        """
    )

    assert tokens[:4] == [
        ("VOID", "void"),
        ("IDENTIFIER", "main"),
        ("LPAREN", "("),
        ("RPAREN", ")"),
    ]
    assert all(token_type != "PREPROCESSOR" for token_type, _ in tokens)


def test_preprocessor_extension_directive_tokenization():
    tokens = tokenize_code(
        """
        #extension GL_EXT_nonuniform_qualifier : enable
        layout(set = 0, binding = 0) uniform sampler2D albedoTex;
        """
    )

    assert tokens[:2] == [("LAYOUT", "layout"), ("LPAREN", "(")]
    assert ("IDENTIFIER", "GL_EXT_nonuniform_qualifier") not in tokens
    assert all(token_type != "PREPROCESSOR" for token_type, _ in tokens)


def test_one_dimensional_sampler_tokenization():
    tokens = tokenize_code(
        """
        uniform sampler1D ramp;
        uniform sampler1DArray ramps;
        """
    )

    assert ("SAMPLER1D", "sampler1D") in tokens
    assert ("SAMPLER1DARRAY", "sampler1DArray") in tokens


if __name__ == "__main__":
    pytest.main()
