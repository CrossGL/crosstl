import re
from typing import Iterator, Tuple, List
from enum import Enum, auto

# using sets for faster lookup
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}

# use tuple for immutable token types that won't change
TOKENS = tuple(
    [
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        ("BITWISE_NOT", r"~"),
        ("USE", r"\buse\b"),
        ("MOD", r"\bmod\b"),
        ("PUB", r"\bpub\b"),
        ("STRUCT", r"\bstruct\b"),
        ("IMPL", r"\bimpl\b"),
        ("TRAIT", r"\btrait\b"),
        ("ENUM", r"\benum\b"),
        ("TYPE", r"\btype\b"),
        ("CONST", r"\bconst\b"),
        ("STATIC", r"\bstatic\b"),
        ("MUT", r"\bmut\b"),
        ("REF", r"\bref\b"),
        ("LET", r"\blet\b"),
        ("FN", r"\bfn\b"),
        ("RETURN", r"\breturn\b"),
        ("SELF", r"\bself\b"),
        ("SUPER", r"\bsuper\b"),
        ("CRATE", r"\bcrate\b"),
        ("EXTERN", r"\bextern\b"),
        ("UNSAFE", r"\bunsafe\b"),
        ("ASYNC", r"\basync\b"),
        ("AWAIT", r"\bawait\b"),
        ("MOVE", r"\bmove\b"),
        ("BOX", r"\bbox\b"),
        ("WHERE", r"\bwhere\b"),
        ("AS", r"\bas\b"),
        ("MATCH", r"\bmatch\b"),
        ("IF", r"\bif\b"),
        ("ELSE", r"\belse\b"),
        ("FOR", r"\bfor\b"),
        ("WHILE", r"\bwhile\b"),
        ("LOOP", r"\bloop\b"),
        ("BREAK", r"\bbreak\b"),
        ("CONTINUE", r"\bcontinue\b"),
        ("IN", r"\bin\b"),
        ("TRUE", r"\btrue\b"),
        ("FALSE", r"\bfalse\b"),
        # Rust types for GPU/shader programming
        ("F32", r"\bf32\b"),
        ("F64", r"\bf64\b"),
        ("I32", r"\bi32\b"),
        ("I64", r"\bi64\b"),
        ("U32", r"\bu32\b"),
        ("U64", r"\bu64\b"),
        ("I8", r"\bi8\b"),
        ("U8", r"\bu8\b"),
        ("I16", r"\bi16\b"),
        ("U16", r"\bu16\b"),
        ("BOOL", r"\bbool\b"),
        ("STR", r"\bstr\b"),
        ("CHAR", r"\bchar\b"),
        ("USIZE", r"\busize\b"),
        ("ISIZE", r"\bisize\b"),
        ("VEC2", r"\bVec2\b"),
        ("VEC3", r"\bVec3\b"),
        ("VEC4", r"\bVec4\b"),
        ("MAT2", r"\bMat2\b"),
        ("MAT3", r"\bMat3\b"),
        ("MAT4", r"\bMat4\b"),
        ("OPTION", r"\bOption\b"),
        ("RESULT", r"\bResult\b"),
        ("SOME", r"\bSome\b"),
        ("NONE", r"\bNone\b"),
        ("OK", r"\bOk\b"),
        ("ERR", r"\bErr\b"),
        # Note: shader-specific words like vertex, fragment are not Rust keywords
        # They can be used as identifiers and only have meaning in attributes
        # Numeric literals must come before identifiers to handle type suffixes
        ("NUMBER", r"\d+(\.\d+)?((i|u)(8|16|32|64|128|size)|f(32|64))?"),
        # Underscore must come before identifier to match wildcard patterns
        ("UNDERSCORE", r"_"),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("STRING", r'"([^"\\]|\\.)*"'),
        ("CHAR_LIT", r"'([^'\\]|\\.)'"),
        ("RAW_STRING", r'r#*"[^"]*"#*'),
        # Punctuation
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
        ("SEMICOLON", r";"),
        ("COMMA", r","),
        ("DOUBLE_COLON", r"::"),
        ("COLON", r":"),
        ("QUESTION", r"\?"),
        # Range operators MUST come before DOT
        ("RANGE_INCLUSIVE", r"\.\.="),
        ("RANGE", r"\.\."),
        ("DOT", r"\."),
        ("ARROW", r"->"),
        ("FAT_ARROW", r"=>"),
        ("POUND", r"#"),
        ("EXCLAMATION", r"!"),
        ("AT", r"@"),
        ("CARET", r"\^"),
        ("TILDE", r"~"),
        # Operators (multi-character operators MUST come before single-character ones)
        ("SHIFT_LEFT", r"<<"),
        ("SHIFT_RIGHT", r">>"),
        ("LESS_EQUAL", r"<="),
        ("GREATER_EQUAL", r">="),
        ("EQUAL", r"=="),
        ("NOT_EQUAL", r"!="),
        ("LOGICAL_AND", r"&&"),
        ("LOGICAL_OR", r"\|\|"),
        ("LESS_THAN", r"<"),
        ("GREATER_THAN", r">"),
        ("AMPERSAND", r"&"),
        ("PIPE", r"\|"),
        ("PLUS_EQUALS", r"\+="),
        ("MINUS_EQUALS", r"-="),
        ("MULTIPLY_EQUALS", r"\*="),
        ("DIVIDE_EQUALS", r"/="),
        ("MOD_EQUALS", r"%="),
        ("BITWISE_AND_EQUALS", r"&="),
        ("BITWISE_OR_EQUALS", r"\|="),
        ("BITWISE_XOR_EQUALS", r"\^="),
        ("SHIFT_LEFT_EQUALS", r"<<="),
        ("SHIFT_RIGHT_EQUALS", r">>="),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("MODULO", r"%"),
        ("EQUALS", r"="),
        ("WHITESPACE", r"\s+"),
    ]
)

KEYWORDS = {
    "use": "USE",
    "mod": "MOD",
    "pub": "PUB",
    "struct": "STRUCT",
    "impl": "IMPL",
    "trait": "TRAIT",
    "enum": "ENUM",
    "type": "TYPE",
    "const": "CONST",
    "static": "STATIC",
    "mut": "MUT",
    "ref": "REF",
    "let": "LET",
    "fn": "FN",
    "return": "RETURN",
    "self": "SELF",
    "super": "SUPER",
    "crate": "CRATE",
    "extern": "EXTERN",
    "unsafe": "UNSAFE",
    "async": "ASYNC",
    "await": "AWAIT",
    "move": "MOVE",
    "box": "BOX",
    "where": "WHERE",
    "as": "AS",
    "match": "MATCH",
    "if": "IF",
    "else": "ELSE",
    "for": "FOR",
    "while": "WHILE",
    "loop": "LOOP",
    "break": "BREAK",
    "continue": "CONTINUE",
    "in": "IN",
    "true": "TRUE",
    "false": "FALSE",
    "f32": "F32",
    "f64": "F64",
    "i32": "I32",
    "i64": "I64",
    "u32": "U32",
    "u64": "U64",
    "i8": "I8",
    "u8": "U8",
    "i16": "I16",
    "u16": "U16",
    "bool": "BOOL",
    "str": "STR",
    "char": "CHAR",
    "usize": "USIZE",
    "isize": "ISIZE",
    "Vec2": "VEC2",
    "Vec3": "VEC3",
    "Vec4": "VEC4",
    "Mat2": "MAT2",
    "Mat3": "MAT3",
    "Mat4": "MAT4",
    "Option": "OPTION",
    "Result": "RESULT",
    "Some": "SOME",
    "None": "NONE",
    "Ok": "OK",
    "Err": "ERR",
    # Removed shader keywords - they should be identifiers, not reserved words
}


class TokenType(Enum):
    COMMENT_SINGLE = auto()
    COMMENT_MULTI = auto()
    BITWISE_NOT = auto()
    USE = auto()
    MOD = auto()
    PUB = auto()
    STRUCT = auto()
    IMPL = auto()
    TRAIT = auto()
    ENUM = auto()
    TYPE = auto()
    CONST = auto()
    STATIC = auto()
    MUT = auto()
    REF = auto()
    LET = auto()
    FN = auto()
    RETURN = auto()
    SELF = auto()
    SUPER = auto()
    CRATE = auto()
    EXTERN = auto()
    UNSAFE = auto()
    ASYNC = auto()
    AWAIT = auto()
    MOVE = auto()
    BOX = auto()
    WHERE = auto()
    AS = auto()
    MATCH = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    LOOP = auto()
    BREAK = auto()
    CONTINUE = auto()
    IN = auto()
    TRUE = auto()
    FALSE = auto()
    F32 = auto()
    F64 = auto()
    I32 = auto()
    I64 = auto()
    U32 = auto()
    U64 = auto()
    I8 = auto()
    U8 = auto()
    I16 = auto()
    U16 = auto()
    BOOL = auto()
    STR = auto()
    CHAR = auto()
    USIZE = auto()
    ISIZE = auto()
    VEC2 = auto()
    VEC3 = auto()
    VEC4 = auto()
    MAT2 = auto()
    MAT3 = auto()
    MAT4 = auto()
    OPTION = auto()
    RESULT = auto()
    SOME = auto()
    NONE = auto()
    OK = auto()
    ERR = auto()
    # Removed shader-specific tokens - they should be identifiers
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()
    CHAR_LIT = auto()
    RAW_STRING = auto()
    LBRACE = auto()
    RBRACE = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMICOLON = auto()
    COMMA = auto()
    COLON = auto()
    DOUBLE_COLON = auto()
    QUESTION = auto()
    DOT = auto()
    RANGE = auto()
    RANGE_INCLUSIVE = auto()
    ARROW = auto()
    FAT_ARROW = auto()
    POUND = auto()
    EXCLAMATION = auto()
    AT = auto()
    AMPERSAND = auto()
    PIPE = auto()
    CARET = auto()
    TILDE = auto()
    UNDERSCORE = auto()
    SHIFT_LEFT = auto()
    SHIFT_RIGHT = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LOGICAL_AND = auto()
    LOGICAL_OR = auto()
    PLUS_EQUALS = auto()
    MINUS_EQUALS = auto()
    MULTIPLY_EQUALS = auto()
    DIVIDE_EQUALS = auto()
    MOD_EQUALS = auto()
    BITWISE_AND_EQUALS = auto()
    BITWISE_OR_EQUALS = auto()
    BITWISE_XOR_EQUALS = auto()
    SHIFT_LEFT_EQUALS = auto()
    SHIFT_RIGHT_EQUALS = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    EQUALS = auto()
    WHITESPACE = auto()


class Token:
    def __init__(self, token_type: TokenType, text: str):
        self.token_type = token_type
        self.text = text


class RustLexer:
    def __init__(self, code: str):
        self._token_patterns = [(name, re.compile(pattern)) for name, pattern in TOKENS]
        self.code = code
        self._length = len(code)
        self.reserved_keywords = {
            **KEYWORDS,
            "match": TokenType.MATCH,
            "loop": TokenType.LOOP,
            "break": TokenType.BREAK,
            "continue": TokenType.CONTINUE,
        }

    def tokenize(self) -> List[Tuple[str, str]]:
        # tokenize the input code and return list of tokens
        return list(self.token_generator())

    def token_generator(self) -> Iterator[Tuple[str, str]]:
        # function that yields tokens one at a time
        pos = 0
        while pos < self._length:
            token = self._next_token(pos)
            if token is None:
                raise SyntaxError(
                    f"Illegal character '{self.code[pos]}' at position {pos}"
                )
            new_pos, token_type, text = token

            # Skip comments and whitespace
            if token_type == "IDENTIFIER" and text in self.reserved_keywords:
                token_type = self.reserved_keywords[text]

            if token_type not in SKIP_TOKENS:
                yield (token_type, text)

            pos = new_pos

        yield ("EOF", "")

    def _next_token(self, pos: int) -> Tuple[int, str, str]:
        # find the next token starting at the given position
        for token_type, pattern in self._token_patterns:
            match = pattern.match(self.code, pos)
            if match:
                return match.end(0), token_type, match.group(0)
        return None

    @classmethod
    def from_file(cls, filepath: str, chunk_size: int = 8192) -> "RustLexer":
        # create a lexer instance from a file, reading in chunks
        with open(filepath, "r") as f:
            return cls(f.read())


class Lexer:
    """Compatibility wrapper around RustLexer"""

    def __init__(self, input_str):
        self.lexer = RustLexer(input_str)
        self.tokens = self.lexer.tokenize()
        self.current_pos = 0

    def next(self):
        if self.current_pos < len(self.tokens):
            token = self.tokens[self.current_pos]
            self.current_pos += 1
            return token
        return ("EOF", "")

    def peek(self):
        if self.current_pos < len(self.tokens):
            return self.tokens[self.current_pos]
        return ("EOF", "")
