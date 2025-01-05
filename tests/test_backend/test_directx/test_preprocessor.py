import pytest
from crosstl.backend.Directx.DirectxLexer import HLSLLexer
from DirectxParser import HLSLParser
from DirectxCrossGLCodeGen import HLSLToCrossGLConverter

@pytest.fixture
def converter():
    return HLSLToCrossGLConverter()

def test_include_directive(converter):
    shader_code = '#include "common.hlsl"\nfloat4 main() : SV_Target { return 0; }'
    expected_output = (
        "// Included file: common.hlsl\nfloat4 main() : SV_Target { return 0; }"
    )
    lexer = HLSLLexer(shader_code)
    tokens = lexer.tokenize()
    parser = HLSLParser(tokens)
    ast = parser.parse()
    output = converter.convert(ast)
    
    # Check if the included file path is part of the output
    assert "// Included file: common.hlsl" in output
    # Additional assertions can be added here to verify the correctness of the output
