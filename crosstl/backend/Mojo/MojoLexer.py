"""Lexer for importing Mojo source into CrossGL Translator."""

import re
from typing import Iterator, List, Tuple

# using sets for faster lookup
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}

TOKENS = tuple(
    [
        ("COMMENT_SINGLE", r"#.*"),
        ("COMMENT_MULTI", r'"""[\s\S]*?"""'),
        ("BITWISE_NOT", r"~"),
        ("STRUCT", r"\bstruct\b"),
        ("LET", r"\blet\b"),
        ("VAR", r"\bvar\b"),
        ("FN", r"\bfn\b"),
        ("RETURN", r"\breturn\b"),
        ("RAISE", r"\braise\b"),
        ("IF", r"\bif\b"),
        ("ELIF", r"\belif\b"),
        ("ELSE", r"\belse\b"),
        ("FOR", r"\bfor\b"),
        ("WHILE", r"\bwhile\b"),
        ("WITH", r"\bwith\b"),
        ("SWITCH", r"\bswitch\b"),
        ("CASE", r"\bcase\b"),
        ("DEFAULT", r"\bdefault\b"),
        ("BREAK", r"\bbreak\b"),
        ("CONTINUE", r"\bcontinue\b"),
        ("FROM", r"\bfrom\b"),
        ("IMPORT", r"\bimport\b"),
        ("AS", r"\bas\b"),
        ("IN", r"\bin\b"),
        ("PASS", r"\bpass\b"),
        ("DEF", r"\bdef\b"),
        ("CLASS", r"\bclass\b"),
        ("COMPTIME", r"\bcomptime\b"),
        ("CONSTANT", r"\bconstant\b"),
        ("INT", r"\bInt\b"),
        ("FLOAT", r"\bFloat\b"),
        ("BOOL", r"\bBool\b"),
        ("STRING", r"\bString\b"),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        (
            "NUMBER",
            r"0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+(\.\d+)?([eE][+-]?\d+)?",
        ),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("ATTRIBUTE", r"\[\[[^\]]*\]\]"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
        ("SEMICOLON", r";"),
        ("STRING_LITERAL", r'"[^"]*"'),
        ("COMMA", r","),
        ("COLON", r":"),
        ("QUESTION", r"\?"),
        ("ASSIGN_SHIFT_LEFT", r"<<="),
        ("ASSIGN_SHIFT_RIGHT", r">>="),
        ("SHIFT_LEFT", r"<<"),
        ("SHIFT_RIGHT", r">>"),
        ("LESS_EQUAL", r"<="),
        ("GREATER_EQUAL", r">="),
        ("LESS_THAN", r"<"),
        ("GREATER_THAN", r">"),
        ("EQUAL", r"=="),
        ("NOT_EQUAL", r"!="),
        ("NOT", r"!"),
        ("PLUS_EQUALS", r"\+="),
        ("MINUS_EQUALS", r"-="),
        ("MULTIPLY_EQUALS", r"\*="),
        ("DIVIDE_EQUALS", r"/="),
        ("ASSIGN_XOR", r"\^="),
        ("ASSIGN_OR", r"\|="),
        ("ASSIGN_AND", r"\&="),
        ("ASSIGN_MOD", r"%="),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("AND", r"&&"),
        ("OR", r"\|\|"),
        ("BITWISE_AND", r"&"),
        ("BITWISE_OR", r"\|"),
        ("BITWISE_XOR", r"\^"),
        ("DOT", r"\."),
        ("EQUALS", r"="),
        ("WHITESPACE", r"\s+"),
        ("MOD", r"%"),
        ("AT", r"@"),
    ]
)


KEYWORDS = {
    "struct": "STRUCT",
    "let": "LET",
    "var": "VAR",
    "fn": "FN",
    "return": "RETURN",
    "raise": "RAISE",
    "if": "IF",
    "elif": "ELIF",
    "else": "ELSE",
    "for": "FOR",
    "while": "WHILE",
    "with": "WITH",
    "switch": "SWITCH",
    "case": "CASE",
    "default": "DEFAULT",
    "break": "BREAK",
    "continue": "CONTINUE",
    "from": "FROM",
    "import": "IMPORT",
    "as": "AS",
    "in": "IN",
    "and": "AND",
    "or": "OR",
    "not": "NOT",
    "True": "BOOL_LITERAL",
    "False": "BOOL_LITERAL",
    "true": "BOOL_LITERAL",
    "false": "BOOL_LITERAL",
    "pass": "PASS",
    "def": "DEF",
    "class": "CLASS",
    "comptime": "COMPTIME",
    "constant": "CONSTANT",
    "Int": "INT",
    "Float": "FLOAT",
    "Bool": "BOOL",
    "String": "STRING",
}


class MojoLexer:
    """Tokenize Mojo source for the Mojo backend parser."""

    def __init__(self, code: str):
        self._token_patterns = [(name, re.compile(pattern)) for name, pattern in TOKENS]
        self.code = code

    def tokenize(self) -> List[Tuple[str, str]]:
        return list(self.token_generator())

    def token_generator(self) -> Iterator[Tuple[str, str]]:
        indent_stack = [0]
        layout_code = self._remove_multiline_comments(self.code)

        for line in layout_code.splitlines():
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue

            indent_text = line[: len(line) - len(line.lstrip(" \t"))]
            indent_width = len(indent_text.expandtabs(4))
            while indent_width < indent_stack[-1]:
                indent_stack.pop()
                yield ("DEDENT", "")

            if indent_width != indent_stack[-1]:
                if indent_width > indent_stack[-1]:
                    indent_stack.append(indent_width)
                    yield ("INDENT", indent_text)
                else:
                    raise SyntaxError("Inconsistent Mojo indentation")

            pos = len(indent_text)
            while pos < len(line):
                token = self._next_token(line, pos)
                if token is None:
                    raise SyntaxError(
                        f"Illegal character '{line[pos]}' at position {pos}"
                    )
                new_pos, token_type, text = token

                if token_type == "COMMENT_SINGLE":
                    break

                if token_type == "IDENTIFIER" and text in KEYWORDS:
                    token_type = KEYWORDS[text]

                if token_type == "AND":
                    text = "&&"
                elif token_type == "OR":
                    text = "||"
                elif token_type == "NOT":
                    text = "!"
                elif token_type == "BOOL_LITERAL":
                    text = text.lower()

                if token_type not in SKIP_TOKENS:
                    yield (token_type, text)

                pos = new_pos

            yield ("NEWLINE", "\n")

        while len(indent_stack) > 1:
            indent_stack.pop()
            yield ("DEDENT", "")

        yield ("EOF", "")

    def _remove_multiline_comments(self, code: str) -> str:
        """Remove triple-quoted comments while preserving line layout."""

        def preserve_layout(match):
            return "".join("\n" if char == "\n" else " " for char in match.group(0))

        return re.sub(r'"""[\s\S]*?"""', preserve_layout, code)

    def _next_token(self, source: str, pos: int) -> Tuple[int, str, str]:
        """Match the next token in ``source`` at ``pos`` and return its end offset."""
        for token_type, pattern in self._token_patterns:
            match = pattern.match(source, pos)
            if match:
                return match.end(0), token_type, match.group(0)
        return None

    @classmethod
    def from_file(cls, filepath: str, chunk_size: int = 8192) -> "MojoLexer":
        """Create a lexer instance from a Mojo source file."""
        with open(filepath) as f:
            return cls(f.read())
