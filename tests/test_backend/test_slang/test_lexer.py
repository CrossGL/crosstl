import pytest
from crosstl.backend.Slang.SlangLexer import SlangLexer


def test_basic_lexing():
    """Test basic Slang lexing functionality."""
    code = """
    float4 main(float2 uv : TEXCOORD) : SV_TARGET {
        float x = 1.0;
        return float4(x, 0.0, 0.0, 1.0);
    }
    """
    lexer = SlangLexer(code)
    tokens = lexer.tokenize()
    
    # Verify we have tokens
    assert len(tokens) > 0
    
    # Check for specific tokens
    token_types = [t.type for t in tokens]
    assert 'IDENTIFIER' in token_types  # 'main', 'float4', etc.
    assert 'FLOAT' in token_types or 'FLOAT_KW' in token_types
    assert 'EQUALS' in token_types
    assert 'NUMBER' in token_types  # 1.0


if __name__ == "__main__":
    pytest.main()
