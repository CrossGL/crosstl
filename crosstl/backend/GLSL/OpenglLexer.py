import re
from typing import Iterator, Tuple, List
from enum import Enum

# using sets for faster lookup
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}

TOKENS = tuple(
    [
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        ("VERSION", r"#version"),
        ("PREPROCESSOR", r"#\w+"),
        ("CONSTANT", r"\bconst\b"),
        ("STRUCT", r"\bstruct\b"),
        ("UNIFORM", r"\buniform\b"),
        ("SAMPLER2D", r"\bsampler2D\b"),
        ("SAMPLERCUBE", r"\bsamplerCube\b"),
        ("BUFFER", r"\bbuffer\b"),
        ("VECTOR", r"\b(vec|ivec|uvec|bvec)[234]\b"),
        ("MATRIX", r"\bmat[234](x[234])?\b"),
        ("FLOAT", r"\bfloat\b"),
        ("INT", r"\bint\b"),
        ("DOUBLE", r"\bdouble\b"),
        ("UINT", r"\buint\b"),
        ("BOOL", r"\bbool\b"),
        ("VOID", r"\bvoid\b"),
        ("RETURN", r"\breturn\b"),
        ("IF", r"\bif\b"),
        ("ELSE", r"\belse\b"),
        ("FOR", r"\bfor\b"),
        ("WHILE", r"\bwhile\b"),
        ("DO", r"\bdo\b"),
        ("IN", r"\bin\b"),
        ("OUT", r"\bout\b"),
        ("INOUT", r"\binout\b"),
        ("LAYOUT", r"\blayout\b"),
        ("ATTRIBUTE", r"\battribute\b"),
        ("VARYING", r"\bvarying\b"),
        ("CONST", r"\bconst\b"),
        ("SWITCH", r"\bswitch\b"),
        ("CASE", r"\bcase\b"),
        ("DEFAULT", r"\bdefault\b"),
        ("BREAK", r"\bbreak\b"),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("NUMBER", r"\d+(\.\d+)?([eE][+-]?\d+)?u?"),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
        ("SEMICOLON", r";"),
        ("STRING", r'"[^"]*"'),
        ("COMMA", r","),
        ("COLON", r":"),
        ("QUESTION", r"\?"),
        ("LESS_EQUAL", r"<="),
        ("GREATER_EQUAL", r">="),
        ("LESS_THAN", r"<"),
        ("GREATER_THAN", r">"),
        ("EQUAL", r"=="),
        ("NOT_EQUAL", r"!="),
        ("ASSIGN_AND", r"&="),
        ("ASSIGN_OR", r"\|="),
        ("ASSIGN_XOR", r"\^="),
        ("LOGICAL_AND", r"&&"),
        ("LOGICAL_OR", r"\|\|"),
        ("ASSIGN_MOD", r"%="),
        ("MOD", r"%"),
        ("PLUS_EQUALS", r"\+="),
        ("MINUS_EQUALS", r"-="),
        ("MULTIPLY_EQUALS", r"\*="),
        ("DIVIDE_EQUALS", r"/="),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("DOT", r"\."),
        ("EQUALS", r"="),
        ("BITWISE_AND", r"&"),
        ("BITWISE_OR", r"\|"),
        ("BITWISE_XOR", r"\^"),
        ("BITWISE_NOT", r"~"),
        ("WHITESPACE", r"\s+"),
    ]
)

KEYWORDS = {
    "struct": "STRUCT",
    "uniform": "UNIFORM",
    "sampler2D": "SAMPLER2D",
    "samplerCube": "SAMPLERCUBE",
    "float": "FLOAT",
    "int": "INT",
    "uint": "UINT",
    "bool": "BOOL",
    "void": "VOID",
    "double": "DOUBLE",
    "return": "RETURN",
    "else if": "ELSE_IF",
    "if": "IF",
    "else": "ELSE",
    "for": "FOR",
    "while": "WHILE",
    "do": "DO",
    "in": "IN",
    "out": "OUT",
    "inout": "INOUT",
    "layout": "LAYOUT",
    "attribute": "ATTRIBUTE",
    "varying": "VARYING",
    "const": "CONST",
    "switch": "SWITCH",
    "case": "CASE",
    "default": "DEFAULT",
    "break": "BREAK",
}


class TokenType(Enum):
    ID = "ID"
    INT = "INT"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    HALF = "HALF"
    STRING = "STRING"

    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    MOD = "MOD"
    ASSIGN = "ASSIGN"

    # Bitwise operators
    AMPERSAND = "AMPERSAND"  # &
    PIPE = "PIPE"  # |
    CARET = "CARET"  # ^
    TILDE = "TILDE"  # ~

    EQ = "EQ"
    NEQ = "NEQ"
    GT = "GT"
    GTE = "GTE"
    LT = "LT"
    LTE = "LTE"

    AND = "AND"
    OR = "OR"
    NOT = "NOT"

    BIT_AND = "BIT_AND"
    BIT_OR = "BIT_OR"
    BIT_XOR = "BIT_XOR"
    BIT_NOT = "BIT_NOT"

    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LCBR = "LCBR"
    RCBR = "RCBR"
    LSBR = "LSBR"
    RSBR = "RSBR"

    DOT = "DOT"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"
    COLON = "COLON"
    QUESTION = "QUESTION"

    IF = "IF"
    ELSE = "ELSE"
    FOR = "FOR"
    WHILE = "WHILE"
    DO = "DO"
    BREAK = "BREAK"
    CONTINUE = "CONTINUE"
    RETURN = "RETURN"

    SWITCH = "SWITCH"
    CASE = "CASE"
    DEFAULT = "DEFAULT"

    STRUCT = "STRUCT"

    COMMENT = "COMMENT"

    # Type specifiers
    TYPE_INT = "TYPE_INT"
    TYPE_FLOAT = "TYPE_FLOAT"
    TYPE_BOOL = "TYPE_BOOL"
    TYPE_VEC2 = "TYPE_VEC2"
    TYPE_VEC3 = "TYPE_VEC3"
    TYPE_VEC4 = "TYPE_VEC4"
    TYPE_MAT2 = "TYPE_MAT2"
    TYPE_MAT3 = "TYPE_MAT3"
    TYPE_MAT4 = "TYPE_MAT4"
    TYPE_SAMPLER2D = "TYPE_SAMPLER2D"
    TYPE_SAMPLER3D = "TYPE_SAMPLER3D"
    TYPE_VOID = "TYPE_VOID"

    # GLSL Qualifiers
    QUALIFIER_CONST = "QUALIFIER_CONST"
    QUALIFIER_IN = "QUALIFIER_IN"
    QUALIFIER_OUT = "QUALIFIER_OUT"
    QUALIFIER_INOUT = "QUALIFIER_INOUT"
    QUALIFIER_UNIFORM = "QUALIFIER_UNIFORM"
    QUALIFIER_VARYING = "QUALIFIER_VARYING"
    QUALIFIER_ATTRIBUTE = "QUALIFIER_ATTRIBUTE"

    PRECISION_LOW = "PRECISION_LOW"
    PRECISION_MEDIUM = "PRECISION_MEDIUM"
    PRECISION_HIGH = "PRECISION_HIGH"
    PRECISION = "PRECISION"

    EOF = "EOF"


class Token:
    """Token class for the GLSL lexer"""

    def __init__(self, token_type, value, line, column):
        self.token_type = token_type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.token_type}, '{self.value}', line={self.line}, col={self.column})"


class GLSLLexer:
    def __init__(self, code: str):
        self._token_patterns = [(name, re.compile(pattern)) for name, pattern in TOKENS]
        self.code = code
        self._length = len(code)
        self.position = 0
        self.line = 1
        self.column = 0
        self.current_char = (
            self.code[self.position] if self.position < len(self.code) else None
        )

        self.reserved_keywords = {
            "if": TokenType.IF,
            "else": TokenType.ELSE,
            "for": TokenType.FOR,
            "while": TokenType.WHILE,
            "do": TokenType.DO,
            "break": TokenType.BREAK,
            "continue": TokenType.CONTINUE,
            "return": TokenType.RETURN,
            "switch": TokenType.SWITCH,
            "case": TokenType.CASE,
            "default": TokenType.DEFAULT,
            "struct": TokenType.STRUCT,
            "int": TokenType.TYPE_INT,
            "float": TokenType.TYPE_FLOAT,
            "bool": TokenType.TYPE_BOOL,
            "vec2": TokenType.TYPE_VEC2,
            "vec3": TokenType.TYPE_VEC3,
            "vec4": TokenType.TYPE_VEC4,
            "mat2": TokenType.TYPE_MAT2,
            "mat3": TokenType.TYPE_MAT3,
            "mat4": TokenType.TYPE_MAT4,
            "sampler2D": TokenType.TYPE_SAMPLER2D,
            "sampler3D": TokenType.TYPE_SAMPLER3D,
            "void": TokenType.TYPE_VOID,
            "const": TokenType.QUALIFIER_CONST,
            "in": TokenType.QUALIFIER_IN,
            "out": TokenType.QUALIFIER_OUT,
            "inout": TokenType.QUALIFIER_INOUT,
            "uniform": TokenType.QUALIFIER_UNIFORM,
            "varying": TokenType.QUALIFIER_VARYING,
            "attribute": TokenType.QUALIFIER_ATTRIBUTE,
            "lowp": TokenType.PRECISION_LOW,
            "mediump": TokenType.PRECISION_MEDIUM,
            "highp": TokenType.PRECISION_HIGH,
            "precision": TokenType.PRECISION,
        }

    def tokenize(self) -> List[Tuple[str, str]]:
        # tokenize the input code and return list of tokens
        return list(self.token_generator())

    def token_generator(self) -> Iterator[Tuple[str, str]]:
        # function that yields tokens one at a time
        while self.position < len(self.code):
            # Ensure current_char is set correctly
            self.current_char = (
                self.code[self.position] if self.position < len(self.code) else None
            )

            if self.current_char is None:
                break

            token = self.get_next_token()
            if token is None:
                self.advance()  # Skip this character and try again
                continue

            new_pos, token_type, text = token

            if token_type == "IDENTIFIER" and text in KEYWORDS:
                token_type = KEYWORDS[text]

            # Only skip whitespace and comments
            if token_type not in ["WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"]:
                yield (token_type, text)

            self.position = new_pos

        yield ("EOF", "")

    def get_next_token(self):
        """
        Lexical analyzer (tokenizer) - this method is responsible for
        breaking a sentence apart into tokens
        """
        token = None
        while self.current_char is not None and token is None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == "/" and self.peek() == "/":
                self.skip_comment()
                continue

            if self.current_char == "/" and self.peek() == "*":
                self.skip_multiline_comment()
                continue

            if self.current_char.isalpha() or self.current_char == "_":
                return self.identifier()

            if self.current_char.isdigit() or (
                self.current_char == "." and self.peek().isdigit()
            ):
                return self.number()

            if self.current_char == "+":
                # token = Token(TokenType.ADD, "+", self.line, self.column)
                token = (self.position + 1, "PLUS", "+")
                self.advance()

            elif self.current_char == "-":
                # token = Token(TokenType.SUB, "-", self.line, self.column)
                token = (self.position + 1, "MINUS", "-")
                self.advance()

            elif self.current_char == "*":
                # token = Token(TokenType.MUL, "*", self.line, self.column)
                token = (self.position + 1, "MULTIPLY", "*")
                self.advance()

            elif self.current_char == "/":
                # token = Token(TokenType.DIV, "/", self.line, self.column)
                token = (self.position + 1, "DIVIDE", "/")
                self.advance()

            elif self.current_char == "%":
                # token = Token(TokenType.MOD, "%", self.line, self.column)
                token = (self.position + 1, "MOD", "%")
                self.advance()

            elif self.current_char == "=":
                if self.peek() == "=":
                    # token = Token(TokenType.EQ, "==", self.line, self.column)
                    token = (self.position + 2, "EQUAL", "==")
                    self.advance()
                    self.advance()
                else:
                    # token = Token(TokenType.ASSIGN, "=", self.line, self.column)
                    token = (self.position + 1, "EQUALS", "=")
                    self.advance()

            elif self.current_char == "!":
                if self.peek() == "=":
                    # token = Token(TokenType.NEQ, "!=", self.line, self.column)
                    token = (self.position + 2, "NOT_EQUAL", "!=")
                    self.advance()
                    self.advance()
                else:
                    # token = Token(TokenType.NOT, "!", self.line, self.column)
                    token = (self.position + 1, "LOGICAL_NOT", "!")
                    self.advance()

            elif self.current_char == ">":
                if self.peek() == "=":
                    # token = Token(TokenType.GTE, ">=", self.line, self.column)
                    token = (self.position + 2, "GREATER_EQUAL", ">=")
                    self.advance()
                    self.advance()
                else:
                    # token = Token(TokenType.GT, ">", self.line, self.column)
                    token = (self.position + 1, "GREATER_THAN", ">")
                    self.advance()

            elif self.current_char == "<":
                if self.peek() == "=":
                    # token = Token(TokenType.LTE, "<=", self.line, self.column)
                    token = (self.position + 2, "LESS_EQUAL", "<=")
                    self.advance()
                    self.advance()
                else:
                    # token = Token(TokenType.LT, "<", self.line, self.column)
                    token = (self.position + 1, "LESS_THAN", "<")
                    self.advance()

            elif self.current_char == "&":
                if self.peek() == "&":
                    # token = Token(TokenType.AND, "&&", self.line, self.column)
                    token = (self.position + 2, "LOGICAL_AND", "&&")
                    self.advance()
                    self.advance()
                else:
                    # token = Token(TokenType.AMPERSAND, "&", self.line, self.column)
                    token = (self.position + 1, "BITWISE_AND", "&")
                    self.advance()

            elif self.current_char == "|":
                if self.peek() == "|":
                    # token = Token(TokenType.OR, "||", self.line, self.column)
                    token = (self.position + 2, "LOGICAL_OR", "||")
                    self.advance()
                    self.advance()
                else:
                    # token = Token(TokenType.PIPE, "|", self.line, self.column)
                    token = (self.position + 1, "BITWISE_OR", "|")
                    self.advance()

            elif self.current_char == "^":
                # token = Token(TokenType.CARET, "^", self.line, self.column)
                token = (self.position + 1, "BITWISE_XOR", "^")
                self.advance()

            elif self.current_char == "~":
                # token = Token(TokenType.TILDE, "~", self.line, self.column)
                token = (self.position + 1, "BITWISE_NOT", "~")
                self.advance()

            elif self.current_char == "(":
                # token = Token(TokenType.LPAREN, "(", self.line, self.column)
                token = (self.position + 1, "LPAREN", "(")
                self.advance()

            elif self.current_char == ")":
                # token = Token(TokenType.RPAREN, ")", self.line, self.column)
                token = (self.position + 1, "RPAREN", ")")
                self.advance()

            elif self.current_char == "{":
                # token = Token(TokenType.LCBR, "{", self.line, self.column)
                token = (self.position + 1, "LBRACE", "{")
                self.advance()

            elif self.current_char == "}":
                # token = Token(TokenType.RCBR, "}", self.line, self.column)
                token = (self.position + 1, "RBRACE", "}")
                self.advance()

            elif self.current_char == "[":
                # token = Token(TokenType.LSBR, "[", self.line, self.column)
                token = (self.position + 1, "LBRACKET", "[")
                self.advance()

            elif self.current_char == "]":
                # token = Token(TokenType.RSBR, "]", self.line, self.column)
                token = (self.position + 1, "RBRACKET", "]")
                self.advance()

            elif self.current_char == ".":
                # token = Token(TokenType.DOT, ".", self.line, self.column)
                token = (self.position + 1, "DOT", ".")
                self.advance()

            elif self.current_char == ",":
                # token = Token(TokenType.COMMA, ",", self.line, self.column)
                token = (self.position + 1, "COMMA", ",")
                self.advance()

            elif self.current_char == ";":
                # token = Token(TokenType.SEMICOLON, ";", self.line, self.column)
                token = (self.position + 1, "SEMICOLON", ";")
                self.advance()

            elif self.current_char == ":":
                # token = Token(TokenType.COLON, ":", self.line, self.column)
                token = (self.position + 1, "COLON", ":")
                self.advance()

            elif self.current_char == "?":
                # token = Token(TokenType.QUESTION, "?", self.line, self.column)
                token = (self.position + 1, "QUESTION", "?")
                self.advance()

            elif self.current_char == "#":
                # Handle preprocessor directive starting with #
                self.position
                directive = "#"
                self.advance()

                # Collect the directive name (like 'version', 'include', etc.)
                while self.current_char is not None and not self.current_char.isspace():
                    directive += self.current_char
                    self.advance()

                token = (self.position, "PREPROCESSOR", directive)

            if token is None:
                raise SyntaxError(f"Unexpected character: {self.current_char}")

            return token

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def skip_comment(self):
        while self.current_char != "\n":
            self.advance()

    def skip_multiline_comment(self):
        while not (self.current_char == "*" and self.peek() == "/"):
            self.advance()
        self.advance()  # Skip the closing */

    def advance(self):
        """Advance the position pointer and set the current character"""
        self.position += 1
        self.column += 1
        if self.position < len(self.code):
            self.current_char = self.code[self.position]
        else:
            self.current_char = None

    def peek(self):
        peek_pos = self.position + 1
        if peek_pos < len(self.code):
            return self.code[peek_pos]
        return None

    def identifier(self):
        start = self.position
        while self.current_char is not None and (
            self.current_char.isalnum() or self.current_char == "_"
        ):
            self.advance()
        return (self.position, "IDENTIFIER", self.code[start : self.position])

    def number(self):
        start = self.position
        while self.current_char is not None and (
            self.current_char.isdigit()
            or self.current_char == "."
            or self.current_char == "e"
            or self.current_char == "E"
            or self.current_char == "+"
            or self.current_char == "-"
        ):
            self.advance()
        return (self.position, "NUMBER", self.code[start : self.position])

    @classmethod
    def from_file(cls, filepath: str, chunk_size: int = 8192) -> "GLSLLexer":
        # create a lexer instance from a file, reading in chunks
        with open(filepath, "r") as f:
            return cls(f.read())
