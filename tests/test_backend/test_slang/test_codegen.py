import pytest

from crosstl.backend.Slang.SlangLexer import SlangLexer
from crosstl.backend.Slang.SlangParser import SlangParser
from crosstl.backend.Slang.SlangCrossGLCodeGen import SlangToCrossGLConverter


def test_struct_definition():
    """Test that struct definitions are properly converted to CrossGL."""
    slang_code = """
    struct Material {
        float3 diffuse;
        float shininess;
    };
    
    cbuffer MaterialBuffer {
        Material material;
    };
    
    float4 main(float2 uv : TEXCOORD) : SV_TARGET {
        float3 color = material.diffuse;
        return float4(color, 1.0);
    }
    """
    lexer = SlangLexer(slang_code)
    tokens = lexer.tokenize()
    parser = SlangParser(tokens)
    ast = parser.parse()
    converter = SlangToCrossGLConverter()
    result = converter.generate(ast)

    assert "struct Material" in result
    assert "diffuse" in result
    assert "shininess" in result


if __name__ == "__main__":
    pytest.main()
