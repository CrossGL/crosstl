import unittest
from DirectxLexer import HLSLLexer
from DirectxParser import HLSLParser
from DirectxCrossGLCodeGen import HLSLToCrossGLConverter


class TestHLSLPreprocessor(unittest.TestCase):
    def setUp(self):
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
        self.assertIn("// Included file: common.hlsl", output)
        # Additional assertions can be added here to verify the correctness of the output


if __name__ == "__main__":
    unittest.main()
