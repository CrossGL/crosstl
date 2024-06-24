import re

TOKENS = [
    ('SHADER', r'shader'),
    ('INPUT', r'input'),
    ('OUTPUT', r'output'),
    ('VOID', r'void'),
    ('MAIN', r'main'),
    ('VECTOR', r'vec[2-4]'),
    ('IDENTIFIER', r'[a-zA-Z_][a-zA-Z_0-9]*'),
    ('NUMBER', r'\d+(\.\d+)?'),
    ('LBRACE', r'\{'),
    ('RBRACE', r'\}'),
    ('LPAREN', r'\('),
    ('RPAREN', r'\)'),
    ('SEMICOLON', r';'),
    ('COMMA', r','),
    ('EQUALS', r'='),
    ('WHITESPACE', r'\s+'),
]

KEYWORDS = {
    'shader': 'SHADER',
    'input': 'INPUT',
    'output': 'OUTPUT',
    'void': 'VOID',
    'main': 'MAIN',
}

class Lexer:
    def __init__(self, code):
        self.code = code
        self.tokens = []
        self.tokenize()

    def tokenize(self):
        pos = 0
        while pos < len(self.code):
            match = None
            for token_type, pattern in TOKENS:
                regex = re.compile(pattern)
                match = regex.match(self.code, pos)
                if match:
                    text = match.group(0)
                    if token_type == 'IDENTIFIER' and text in KEYWORDS:
                        token_type = KEYWORDS[text]
                    if token_type != 'WHITESPACE':  # Ignore whitespace tokens
                        token = (token_type, text)
                        self.tokens.append(token)
                    pos = match.end(0)
                    break
            if not match:
                raise SyntaxError(f'Illegal character at position {pos}')
        self.tokens.append(('EOF', None))  # End of file token

if __name__ == "__main__":
    code = "shader main { input vec3 position; output vec4 color; void main() { color = vec4(position, 1.0); } }"
    lexer = Lexer(code)
    for token in lexer.tokens:
        print(token)
