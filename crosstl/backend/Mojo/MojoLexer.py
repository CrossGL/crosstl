"""Lexer for importing Mojo source into CrossGL Translator."""

import re
from typing import Iterator, List, Optional, Tuple

# using sets for faster lookup
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}
TRIPLE_STRING_PREFIXES = {
    "",
    "r",
    "R",
    "t",
    "T",
    "rt",
    "rT",
    "Rt",
    "RT",
    "tr",
    "tR",
    "Tr",
    "TR",
}

TOKENS = tuple(
    [
        ("COMMENT_SINGLE", r"#.*"),
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
        ("TRAIT", r"\btrait\b"),
        ("COMPTIME", r"\bcomptime\b"),
        ("ALIAS", r"\balias\b"),
        ("CONSTANT", r"\bconstant\b"),
        ("INT", r"\bInt\b"),
        ("FLOAT", r"\bFloat\b"),
        ("BOOL", r"\bBool\b"),
        ("STRING", r"\bString\b"),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("BACKTICK_IDENTIFIER", r"`(?:[^`\\]|\\.)*`"),
        (
            "NUMBER",
            r"0[xX](?=[0-9a-fA-F_]*[0-9a-fA-F])[0-9a-fA-F_]+|"
            r"0[bB](?=[01_]*[01])[01_]+|"
            r"0[oO](?=[0-7_]*[0-7])[0-7_]+|"
            r"((\d[\d_]*)?\.\d[\d_]*|\d[\d_]*\.|\d[\d_]*)[eE][+-]?\d[\d_]*|"
            r"(\d[\d_]*)?\.\d[\d_]*|\d[\d_]*\.|\d[\d_]*",
        ),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("ATTRIBUTE", r"\[\[[^\]]*\]\]"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
        ("SEMICOLON", r";"),
        ("STRING_LITERAL", r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''),
        ("COMMA", r","),
        ("WALRUS", r":="),
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
        ("FLOOR_DIVIDE_EQUALS", r"//="),
        ("DIVIDE_EQUALS", r"/="),
        ("FLOOR_DIVIDE", r"//"),
        ("POWER", r"\*\*"),
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
    "trait": "TRAIT",
    "comptime": "COMPTIME",
    "alias": "ALIAS",
    "constant": "CONSTANT",
    "Int": "INT",
    "Float": "FLOAT",
    "Bool": "BOOL",
    "String": "STRING",
}


class MojoLexer:
    """Tokenize Mojo source for the Mojo backend parser."""

    def __init__(self, code: str):
        code = code.lstrip("\ufeff")
        self._token_patterns = [(name, re.compile(pattern)) for name, pattern in TOKENS]
        self.code = code

    def tokenize(self) -> List[Tuple[str, str]]:
        return list(self.token_generator())

    def token_generator(self) -> Iterator[Tuple[str, str]]:
        indent_stack = [0]
        layout_code = self._join_line_continuations(
            self._normalize_triple_string_literals(self.code)
        )

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

    def _normalize_triple_string_literals(self, code: str) -> str:
        """Collapse Mojo triple-quoted string literals for line-oriented lexing."""
        pieces = []
        pos = 0

        while pos < len(code):
            prefix, quote = self._match_triple_string_start(code, pos)
            if quote is None:
                pieces.append(code[pos])
                pos += 1
                continue

            body_start = pos + len(prefix) + len(quote)
            body_end = code.find(quote, body_start)
            if body_end == -1:
                raise SyntaxError("Unterminated Mojo triple-quoted string literal")

            literal_source = code[pos : body_end + len(quote)]
            if self._is_standalone_triple_string(code, pos, prefix):
                pieces.append(self._preserve_layout(literal_source))
                pos = body_end + len(quote)
                continue

            body = code[body_start:body_end]
            pieces.append(f'{prefix}"{self._escape_triple_string_body(body)}"')
            pos = body_end + len(quote)

        return "".join(pieces)

    def _match_triple_string_start(
        self, code: str, pos: int
    ) -> Tuple[str, Optional[str]]:
        """Return an optional Mojo string prefix and triple quote delimiter."""
        for prefix_len in (2, 1, 0):
            prefix = code[pos : pos + prefix_len]
            if prefix not in TRIPLE_STRING_PREFIXES:
                continue
            if prefix and pos > 0 and re.match(r"[A-Za-z0-9_]", code[pos - 1]):
                continue

            quote_pos = pos + prefix_len
            quote = code[quote_pos : quote_pos + 3]
            if quote in {'"""', "'''"}:
                return prefix, quote

        return "", None

    def _is_standalone_triple_string(self, code: str, pos: int, prefix: str) -> bool:
        """Detect unprefixed doc/comment-style triple-quoted blocks."""
        if prefix:
            return False
        line_start = code.rfind("\n", 0, pos) + 1
        return not code[line_start:pos].strip()

    def _preserve_layout(self, text: str) -> str:
        return "".join("\n" if char == "\n" else " " for char in text)

    def _escape_triple_string_body(self, body: str) -> str:
        """Escape a triple-quoted body as a normal double-quoted literal."""
        return (
            body.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\n", "\\n")
        )

    def _join_line_continuations(self, code: str) -> str:
        """Join explicit Mojo line continuations before indentation analysis."""
        return re.sub(r"\\[ \t]*(?:\r?\n)[ \t]*", " ", code)

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
