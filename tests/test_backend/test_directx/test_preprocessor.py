import pytest
from unittest.mock import patch
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.backend.DirectX.DirectxCrossGLCodeGen import HLSLToCrossGLConverter


@pytest.fixture
def converter():
    return HLSLToCrossGLConverter()


@patch("crosstl.backend.DirectX.DirectxPreprocessor.DirectxPreprocessor.handle_include")
def test_include_codegen(mock_handle_include, converter):
    mock_handle_include.return_value = "// Mocked content for common.hlsl"

    shader_code = """
    #include "common.hlsl"
    struct VSInput {
        float4 position : POSITION;
        float4 color : TEXCOORD0;
    };

    struct VSOutput {
        float4 out_position : TEXCOORD0;
    };

    VSOutput VSMain(VSInput input) {
        VSOutput output;
        output.out_position = input.position;
        return output;
    }
    """
    lexer = HLSLLexer(shader_code)
    tokens = lexer.tokenize()
    parser = HLSLParser(tokens)
    ast = parser.parse()
    output = converter.convert(ast)

    assert "// Mocked content for common.hlsl" in output
