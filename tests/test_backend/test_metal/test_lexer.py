import pytest
from typing import List
from crosstl.src.backend.Metal.MetalLexer import MetalLexer


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = MetalLexer(code)
    return lexer.tokens


def test_struct_tokenization():
    code = """
    struct Vertex_INPUT {
    float3 position [[attribute(0)]];
    };

    struct Vertex_OUTPUT {
        float4 position [[position]];
        float2 vUV;
    };
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Struct tokenization not implemented.")


def test_if_tokenization():
    code = """
    float perlinNoise(float2 p) {
    if (p.x == p.y) {
        return 0.0;
    }
    return fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("If statement tokenization not implemented.")


def test_for_tokenization():
    code = """
    float perlinNoise(float2 p) {
    if (p.x == p.y) {
        return 0.0;
    }
    return fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("For loop tokenization not implemented.")


def test_else_tokenization():
    code = """
    float perlinNoise(float2 p) {
    if (p.x == p.y) {
        return 0.0;
    }
    else {
        return fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Else statement tokenization not implemented.")


def test_function_call_tokenization():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    float perlinNoise(float2 p) {
        return fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
    }


    struct Fragment_INPUT {
        float2 vUV [[stage_in]];
    };

    struct Fragment_OUTPUT {
        float4 fragColor [[color(0)]];
    };

    fragment Fragment_OUTPUT fragment_main(Fragment_INPUT input [[stage_in]]) {
        Fragment_OUTPUT output;
        float noise = perlinNoise(input.vUV);
        float height = noise * 10.0;
        float3 color = float3(height / 10.0, 1.0 - height / 10.0, 0.0);
        output.fragColor = float4(color, 1.0);
        return output;
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Function call tokenization not implemented.")


def test_if_else_tokenization():
    code = """
    float perlinNoise(float2 p) {
        if (p.x == p.y) {
            return 0.0;
        }
        else if (p.x == 0.0) {
            return 1.0;
        }

        else {
            return fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
        }
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("If-else statement tokenization not implemented.")


if __name__ == "__main__":
    pytest.main()
