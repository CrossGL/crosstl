import pytest
from crosstl.backend.Directx.DirectxLexer import HLSLLexer
from crosstl.backend.Directx.DirectxParser import HLSLParser
from crosstl.backend.Directx.DirectxCrossGLCodeGen import HLSLToCrossGLConverter


class TestHLSLPreprocessor:
    def setup_method(self):
        self.converter = HLSLToCrossGLConverter()

    def test_include_directive(self):
        shader_code = '#include "common.hlsl"\nfloat4 main() : SV_Target { return 0; }'
        expected_output = (
            "// Included file: common.hlsl\nfloat4 main() : SV_Target { return 0; }"
        )
        lexer = HLSLLexer(shader_code)
        tokens = lexer.tokenize()
        parser = HLSLParser(tokens)
        ast = parser.parse()
        output = self.converter.convert(ast)
        assert "// Included file: common.hlsl" in output

    def test_define_directive(self):
        shader_code = (
            "#define MAX_LIGHTS 10\nfloat4 main() : SV_Target { return MAX_LIGHTS; }"
        )
        expected_output = "float4 main() : SV_Target { return 10; }"
        lexer = HLSLLexer(shader_code)
        tokens = lexer.tokenize()
        parser = HLSLParser(tokens)
        ast = parser.parse()
        output = self.converter.convert(ast)
        assert "float4 main() : SV_Target { return 10; }" in output

    def test_ifdef_directive(self):
        shader_code = "#ifdef MAX_LIGHTS\nfloat4 main() : SV_Target { return MAX_LIGHTS; }\n#endif"
        expected_output = "float4 main() : SV_Target { return MAX_LIGHTS; }"
        lexer = HLSLLexer(shader_code)
        tokens = lexer.tokenize()
        parser = HLSLParser(tokens)
        ast = parser.parse()
        output = self.converter.convert(ast)
        assert "float4 main() : SV_Target { return MAX_LIGHTS; }" in output

    def test_else_directive(self):
        shader_code = """#ifdef MAX_LIGHTS
float4 main() : SV_Target { return MAX_LIGHTS; }
#else
float4 main() : SV_Target { return 0; }
#endif"""
        expected_output = "float4 main() : SV_Target { return MAX_LIGHTS; }"
        lexer = HLSLLexer(shader_code)
        tokens = lexer.tokenize()
        parser = HLSLParser(tokens)
        ast = parser.parse()
        output = self.converter.convert(ast)
        assert "float4 main() : SV_Target { return MAX_LIGHTS; }" in output

    def test_endif_directive(self):
        shader_code = """#ifdef MAX_LIGHTS
float4 main() : SV_Target { return MAX_LIGHTS; }
#endif"""
        expected_output = "float4 main() : SV_Target { return MAX_LIGHTS; }"
        lexer = HLSLLexer(shader_code)
        tokens = lexer.tokenize()
        parser = HLSLParser(tokens)
        ast = parser.parse()
        output = self.converter.convert(ast)
        assert "float4 main() : SV_Target { return MAX_LIGHTS; }" in output


if __name__ == "__main__":
    pytest.main()
