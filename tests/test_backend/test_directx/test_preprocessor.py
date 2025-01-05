import pytest
from unittest.mock import patch
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.backend.DirectX.DirectxCrossGLCodeGen import HLSLToCrossGLConverter

@pytest.fixture
def converter():
    return HLSLToCrossGLConverter()

# Mocking the handle_include method to bypass file system checks
@patch('crosstl.backend.Directx.DirectxPreprocessor.DirectxPreprocessor.handle_include')
def test_include_directive(mock_handle_include, converter):
    # Mock the content that would be returned from the #include directive
    mock_handle_include.return_value = "// Mocked content of common.hlsl"

    shader_code = '#include "common.hlsl"\nfloat4 main() : SV_Target { return 0; }'
    
    # Expected output should include the mocked content
    expected_output = (
        "// Mocked content of common.hlsl\nfloat4 main() : SV_Target { return 0; }"
    )

    lexer = HLSLLexer(shader_code)
    tokens = lexer.tokenize()
    parser = HLSLParser(tokens)
    ast = parser.parse()
    output = converter.convert(ast)

    # Check if the mocked content is part of the output
    assert "// Mocked content of common.hlsl" in output
    # Additional assertions can be added here to verify the correctness of the output