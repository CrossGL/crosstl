import pytest
from typing import List
from crosstl.backend.Vulkan.VulkanLexer import VulkanLexer


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = VulkanLexer(code)
    return lexer.tokens


# ToDO: Implement the tests
def test_struct():
    pass


if __name__ == "__main__":
    pytest.main()
