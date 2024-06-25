import unittest
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.codegen import directx_codegen

class TestLexer(unittest.TestCase):
    def test_tokens(self):
        code = "shader main { input vec3 position; output vec4 color; void main() { color = vec4(position, 1.0); } }"
        lexer = Lexer(code)
        expected_tokens = [
            ('SHADER', 'shader'),
            ('MAIN', 'main'),
            ('LBRACE', '{'),
            ('INPUT', 'input'),
            ('VECTOR', 'vec3'),
            ('IDENTIFIER', 'position'),
            ('SEMICOLON', ';'),
            ('OUTPUT', 'output'),
            ('VECTOR', 'vec4'),
            ('IDENTIFIER', 'color'),
            ('SEMICOLON', ';'),
            ('VOID', 'void'),
            ('MAIN', 'main'),
            ('LPAREN', '('),
            ('RPAREN', ')'),
            ('LBRACE', '{'),
            ('IDENTIFIER', 'color'),
            ('EQUALS', '='),
            ('VECTOR', 'vec4'),
            ('LPAREN', '('),
            ('IDENTIFIER', 'position'),
            ('COMMA', ','),
            ('NUMBER', '1.0'),
            ('RPAREN', ')'),
            ('SEMICOLON', ';'),
            ('RBRACE', '}'),
            ('RBRACE', '}'),
            ('EOF', None)
        ]
        for token in lexer.tokens:
            print(token)
        parser = Parser(lexer.tokens)
        ast = parser.parse()
        #print(ast)
        codegen = directx_codegen.HLSLCodeGen()
        hlsl_code = codegen.generate(ast)
        print(hlsl_code)
if __name__ == "__main__":
    unittest.main()