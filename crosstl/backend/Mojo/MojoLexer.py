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

SOFT_CONTINUATION_SUFFIXES = ("+",)
SOFT_CONTINUATION_PREFIXES = ("+",)

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
        ("POWER_EQUALS", r"\*\*="),
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
        ("AT_EQUALS", r"@="),
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
            self._normalize_prefixed_string_literals(
                self._normalize_triple_string_literals(self.code)
            )
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
            if code[pos] == "#":
                line_end = code.find("\n", pos)
                if line_end == -1:
                    pieces.append(code[pos:])
                    break
                pieces.append(code[pos : line_end + 1])
                pos = line_end + 1
                continue

            if code[pos : pos + 3] not in {'"""', "'''"}:
                if code[pos : pos + 1] in {'"', "'"}:
                    body_end = self._find_simple_string_end(code, pos + 1, code[pos])
                    if body_end is None:
                        raise SyntaxError("Unterminated Mojo string literal")
                    pieces.append(code[pos : body_end + 1])
                    pos = body_end + 1
                    continue

                prefix, quote = self._match_prefixed_string_start(code, pos)
                if quote is not None:
                    body_start = pos + len(prefix) + 1
                    body_end = self._find_prefixed_string_end(
                        code,
                        body_start,
                        quote,
                        interpolation="t" in prefix.lower(),
                    )
                    if body_end is None:
                        raise SyntaxError("Unterminated Mojo prefixed string literal")
                    pieces.append(code[pos : body_end + 1])
                    pos = body_end + 1
                    continue

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
        previous = self._previous_significant_char(code, pos)
        if previous in {"(", "[", "{", ",", "=", ":"}:
            return False
        line_start = code.rfind("\n", 0, pos) + 1
        return not code[line_start:pos].strip()

    def _previous_significant_char(self, code: str, pos: int) -> Optional[str]:
        index = pos - 1
        while index >= 0 and code[index].isspace():
            index -= 1
        if index < 0:
            return None
        return code[index]

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

    def _normalize_prefixed_string_literals(self, code: str) -> str:
        """Collapse Mojo-prefixed string literals for line-oriented lexing."""
        pieces = []
        pos = 0

        while pos < len(code):
            if code[pos] == "#":
                line_end = code.find("\n", pos)
                if line_end == -1:
                    pieces.append(code[pos:])
                    break
                pieces.append(code[pos : line_end + 1])
                pos = line_end + 1
                continue

            if code[pos : pos + 1] in {'"', "'"}:
                body_end = self._find_simple_string_end(code, pos + 1, code[pos])
                if body_end is None:
                    raise SyntaxError("Unterminated Mojo string literal")
                pieces.append(code[pos : body_end + 1])
                pos = body_end + 1
                continue

            prefix, quote = self._match_prefixed_string_start(code, pos)
            if quote is None:
                pieces.append(code[pos])
                pos += 1
                continue

            body_start = pos + len(prefix) + 1
            body_end = self._find_prefixed_string_end(
                code,
                body_start,
                quote,
                interpolation="t" in prefix.lower(),
            )
            if body_end is None:
                raise SyntaxError("Unterminated Mojo prefixed string literal")

            body = code[body_start:body_end]
            pieces.append(f'{prefix}"{self._escape_simple_string_body(body)}"')
            pos = body_end + 1

        return "".join(pieces)

    def _match_prefixed_string_start(
        self, code: str, pos: int
    ) -> Tuple[str, Optional[str]]:
        for prefix_len in (2, 1):
            prefix = code[pos : pos + prefix_len]
            if prefix not in TRIPLE_STRING_PREFIXES:
                continue
            if pos > 0 and re.match(r"[A-Za-z0-9_]", code[pos - 1]):
                continue

            quote_pos = pos + prefix_len
            quote = code[quote_pos : quote_pos + 1]
            if quote not in {'"', "'"}:
                continue
            if code[quote_pos : quote_pos + 3] in {'"""', "'''"}:
                continue
            return prefix, quote

        return "", None

    def _find_prefixed_string_end(
        self, code: str, pos: int, quote: str, interpolation: bool
    ) -> Optional[int]:
        escaped = False
        brace_depth = 0

        while pos < len(code):
            char = code[pos]
            if escaped:
                escaped = False
                pos += 1
                continue
            if char == "\\":
                escaped = True
                pos += 1
                continue

            if interpolation:
                if char == "{" and code[pos : pos + 2] != "{{":
                    brace_depth += 1
                    pos += 1
                    continue
                if char == "}" and code[pos : pos + 2] != "}}":
                    if brace_depth:
                        brace_depth -= 1
                    pos += 1
                    continue
                if brace_depth and self._match_nested_string_start(code, pos):
                    pos = self._skip_nested_string_literal(code, pos)
                    continue

            if char == quote and brace_depth == 0:
                return pos
            pos += 1

        return None

    def _match_nested_string_start(self, code: str, pos: int) -> bool:
        if code[pos : pos + 1] in {'"', "'"}:
            return True
        return self._match_prefixed_string_start(code, pos)[1] is not None

    def _skip_nested_string_literal(self, code: str, pos: int) -> int:
        prefix, quote = self._match_prefixed_string_start(code, pos)
        if quote is None:
            prefix = ""
            quote = code[pos]
        body_start = pos + len(prefix) + 1
        body_end = self._find_simple_string_end(code, body_start, quote)
        if body_end is None:
            raise SyntaxError("Unterminated Mojo nested string literal")
        return body_end + 1

    def _find_simple_string_end(self, code: str, pos: int, quote: str) -> Optional[int]:
        escaped = False
        while pos < len(code):
            char = code[pos]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                return pos
            pos += 1
        return None

    def _escape_simple_string_body(self, body: str) -> str:
        return body.replace("\\", "\\\\").replace('"', '\\"')

    def _join_line_continuations(self, code: str) -> str:
        """Join explicit Mojo line continuations before indentation analysis."""
        joined_lines = []
        pending = ""
        lines = code.splitlines(keepends=True)

        for line_index, line in enumerate(lines):
            content, newline = self._split_line_ending(line)
            if pending:
                content = content.lstrip(" \t")

            stripped = content.rstrip(" \t")
            continuation_pos = len(stripped) - 1
            if stripped.endswith("\\") and not self._has_comment_before(
                content, continuation_pos
            ):
                pending += stripped[:-1] + " "
                continue

            if self._is_soft_continuation_line(stripped, lines, line_index):
                pending += stripped + " "
                continue

            joined_lines.append(pending + content + newline)
            pending = ""

        if pending:
            joined_lines.append(pending)

        return "".join(joined_lines)

    def _is_soft_continuation_line(self, stripped: str, lines, line_index: int) -> bool:
        """Detect unparenthesized expression continuations around infix operators."""
        if not stripped:
            return False
        if self._ends_with_soft_continuation_operator(stripped):
            return True
        next_line = self._next_significant_line(lines, line_index + 1)
        return bool(
            next_line and self._starts_with_soft_continuation_operator(next_line)
        )

    def _ends_with_soft_continuation_operator(self, stripped: str) -> bool:
        if self._has_comment_before(stripped, len(stripped)):
            return False
        if stripped.lstrip(" \t").startswith(("from ", "import ")):
            return False
        return stripped.endswith(SOFT_CONTINUATION_SUFFIXES)

    def _starts_with_soft_continuation_operator(self, stripped: str) -> bool:
        return stripped.lstrip(" \t").startswith(SOFT_CONTINUATION_PREFIXES)

    def _next_significant_line(self, lines, start_index: int) -> str:
        for line in lines[start_index:]:
            content, _ = self._split_line_ending(line)
            stripped = content.strip()
            if stripped and not stripped.startswith("#"):
                return content.rstrip(" \t")
        return ""

    def _split_line_ending(self, line: str) -> Tuple[str, str]:
        if line.endswith("\r\n"):
            return line[:-2], "\r\n"
        if line.endswith(("\n", "\r")):
            return line[:-1], line[-1]
        return line, ""

    def _has_comment_before(self, line: str, pos: int) -> bool:
        quote = None
        escaped = False

        for char in line[:pos]:
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue

            if char in {'"', "'"}:
                quote = char
            elif char == "#":
                return True

        return False

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
