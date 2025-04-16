"""Utilities for testing array handling across different code generators.

This module provides common test cases and verification utilities for arrays.
"""

import pytest
from crosstl.translator.parser import Parser
from crosstl.translator.lexer import Lexer
from typing import List, Dict, Any

# Common shader template with comprehensive array handling scenarios
ARRAY_TEST_SHADER = """
shader ArrayTest {
    struct Particle {
        vec3 position;
        vec3 velocity;
    };

    struct Material {
        float values[4];  // Fixed-size array
        vec3 colors[];    // Dynamic array
    };

    cbuffer Constants {
        float weights[8];
        int indices[10];
    };

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            
            // Array access in various forms
            float value = weights[2];
            int index = indices[5];
            
            // Array member access
            Material material;
            float x = material.values[0];
            vec3 color = material.colors[index];
            
            // Nested array access
            Particle particles[10];
            vec3 pos = particles[3].position;
            particles[index].velocity = vec3(1.0, 0.0, 0.0);
            
            // Array access in expressions
            float sum = weights[0] + weights[1] + weights[2];
            
            // Write to array
            weights[index] = sum;
            
            // Nested array dimension
            float multiDim[3][4];  // Not all backends may support this
            multiDim[0][1] = 2.0;
            
            return output;
        }
    }
}
"""


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = Lexer(code)
    return lexer.tokens


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST."""
    parser = Parser(tokens)
    return parser.parse()


def run_array_test(code_generator_class, verification_tests: Dict[str, Any] = None):
    """Run standard array handling tests on a given code generator.

    Args:
        code_generator_class: The code generator class to test
        verification_tests: Optional dict mapping test names to test cases

    Returns:
        The generated code for further testing
    """
    try:
        tokens = tokenize_code(ARRAY_TEST_SHADER)
        ast = parse_code(tokens)
        code_gen = code_generator_class()
        generated_code = code_gen.generate(ast)

        # Run verification tests
        if verification_tests:
            for test_name, test_case in verification_tests.items():
                if isinstance(test_case, list):
                    # List of strings to check for in the output
                    for expected in test_case:
                        assert (
                            expected in generated_code
                        ), f"Test '{test_name}' failed: '{expected}' not found in generated code"
                elif isinstance(test_case, str):
                    # Single string to check for
                    assert (
                        test_case in generated_code
                    ), f"Test '{test_name}' failed: '{test_case}' not found in generated code"
                elif callable(test_case):
                    # Custom verification function
                    assert test_case(
                        generated_code
                    ), f"Test '{test_name}' failed: custom verification function returned false"

        return generated_code
    except Exception as e:
        pytest.fail(f"Array test failed: {e}")
        return None


# Common verification tests for different backends
DIRECTX_VERIFICATION = {
    "array_type_declarations": [
        "float values[4]",
        "float weights[8]",
    ],
    "array_access": [
        "weights[2]",
        "material.values[0]",
        "material.colors[index]",
        "particles[3].position",
    ],
    "nested_array_access": "particles[index].velocity",
}

METAL_VERIFICATION = {
    "array_type_declarations": [
        "array<float, 4> values",
        "array<float, 8> weights",
    ],
    "array_access": [
        "weights[2]",
        "material.values[0]",
        "material.colors[index]",
        "particles[3].position",
    ],
    "nested_array_access": "particles[index].velocity",
}

GLSL_VERIFICATION = {
    "array_type_declarations": [
        "float values[4]",
        "float weights[8]",
    ],
    "array_access": [
        "weights[2]",
        "material.values[0]",
        "material.colors[index]",
        "particles[3].position",
    ],
    "nested_array_access": "particles[index].velocity",
}
