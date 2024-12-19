import pytest
from typing import List
from crosstl.backend.Vulkan.VulkanLexer import VulkanLexer
from crosstl.backend.Vulkan import VulkanParser


def parse_code(tokens: List):
    """Test the parser
    Args:
        tokens (List): The list of tokens generated from the lexer

    Returns:
        ASTNode: The abstract syntax tree generated from the code
    """
    parser = VulkanParser(tokens)
    return parser.parse()


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = VulkanLexer(code)
    return lexer.tokens


# ToDO: Implement the tests
def test_struct():
    pass


if __name__ == "__main__":
    pytest.main()
