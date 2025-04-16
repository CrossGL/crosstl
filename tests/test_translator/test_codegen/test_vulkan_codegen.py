from crosstl.translator.lexer import Lexer
import pytest
from typing import List
from crosstl.translator.parser import Parser
from crosstl.translator.codegen.vulkan_codegen import VulkanSPIRVCodeGen
from ..test_utils.array_test_helper import ARRAY_TEST_SHADER, tokenize_code, parse_code


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
    codegen = VulkanSPIRVCodeGen()
    return codegen.generate(ast_node)


def test_struct():
    code = """
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };
    struct VSOutput {
        vec4 color @ COLOR;
    };
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        assert code is not None
    except SyntaxError:
        pytest.fail("Vulkan struct codegen not implemented.")


def test_basic_shader():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        assert code is not None
    except SyntaxError:
        pytest.fail("Vulkan basic shader codegen not implemented.")


def test_vulkan_array_handling():
    """Test the Vulkan code generator's handling of array types and array access."""
    try:
        tokens = tokenize_code(ARRAY_TEST_SHADER)
        ast = parse_code(tokens)
        code_gen = VulkanSPIRVCodeGen()
        code_gen.generate(ast)
        # We don't check actual SPIR-V output here as it's binary/complex,
        # but ensure it doesn't crash with array handling
    except SyntaxError as e:
        pytest.fail(f"Vulkan array codegen failed: {e}")


if __name__ == "__main__":
    pytest.main()
