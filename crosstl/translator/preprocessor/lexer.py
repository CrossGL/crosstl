"""Preprocessing tokens: translation phases 1 to 3 of C++ ([lex.phases]) and GLSL (section 3.1 to 3.3).

Line endings are normalized, backslash-newline splices are removed, comments become one space, and the logical
text is cut into preprocessing tokens. Each token keeps its offset in the logical text; :class:`SourceFile` maps
that offset back to the physical line and column, across splices, for diagnostics and the output line map.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache

from ..diagnostics import SourceSpan, TranslationDiagnostic

IDENT = "identifier"
NUMBER = "pp-number"
CHAR = "character-literal"
STRING = "string-literal"
PUNCT = "punctuator"
OTHER = "other"
UNTERMINATED = "unterminated-literal"
NEWLINE = "newline"

LEX_CODE = "project.translate.preprocessor-invalid-token"


class SourceFile:
    """One file's logical text (line endings normalized, splices removed) and its map back to physical lines."""

    __slots__ = ("path", "text", "_line_starts", "_splice_logical", "_splice_physical")

    def __init__(self, path: str | None, raw: str, *, splice_whitespace: bool = False):
        self.path = path
        physical = raw.replace("\r\n", "\n").replace("\r", "\n")
        if physical.startswith("\ufeff"):
            physical = " " + physical[1:]
        self._line_starts = [0] + [m.end() for m in re.finditer("\n", physical)]
        # A splice is a backslash immediately before a newline; Clang (and so DXC and the Metal compiler) also
        # accepts horizontal whitespace between them, with a warning.
        splice = re.compile(r"\\[ \t\f\v]*\n" if splice_whitespace else r"\\\n")
        pieces, logical, last = [], 0, 0
        self._splice_logical, self._splice_physical = [], []
        for match in splice.finditer(physical):
            pieces.append(physical[last : match.start()])
            logical += match.start() - last
            last = match.end()
            self._splice_logical.append(logical)
            self._splice_physical.append(last)
        pieces.append(physical[last:])
        self.text = "".join(pieces)

    def physical(self, offset: int) -> int:
        index = bisect_right(self._splice_logical, offset) - 1
        if index < 0:
            return offset
        return self._splice_physical[index] + offset - self._splice_logical[index]

    def position(self, offset: int):
        """1-based (line, column) of a logical offset."""
        physical = self.physical(offset)
        line = bisect_right(self._line_starts, physical)
        return line, physical - self._line_starts[line - 1] + 1

    def span(self, offset: int) -> SourceSpan:
        line, column = self.position(offset)
        return SourceSpan(self.path, line, column)


class Token:
    """A preprocessing token. ``space`` records whitespace before it on its line; ``hide`` is its hide set, and
    ``noexpand`` marks an identifier that may never be expanded again (the standard's "painted blue").
    A token produced by macro replacement is located at the invocation; ``end`` is then the (source, offset)
    where the outermost invocation ends, which is the line ``__LINE__`` reports in Clang and GCC.
    """

    __slots__ = ("kind", "text", "source", "offset", "space", "hide", "noexpand", "end")

    def __init__(
        self, kind, text, source=None, offset=0, space=False, hide=frozenset()
    ):
        self.kind = kind
        self.text = text
        self.source = source
        self.offset = offset
        self.space = space
        self.hide = hide
        self.noexpand = False
        self.end = None

    def copy(self, **changes) -> Token:
        token = Token(
            self.kind, self.text, self.source, self.offset, self.space, self.hide
        )
        token.noexpand = self.noexpand
        token.end = self.end
        for name, value in changes.items():
            setattr(token, name, value)
        return token

    def span(self) -> SourceSpan | None:
        return self.source.span(self.offset) if self.source is not None else None

    def end_location(self):
        """(source, offset) of the end of the outermost macro invocation, or of the token itself."""
        return self.end if self.end is not None else (self.source, self.offset)

    def __repr__(self) -> str:
        return f"Token({self.kind}, {self.text!r})"


_PUNCTUATORS = r"\.\.\.|<=>|<<=|>>=|->\*|->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||\*=|/=|%=|\+=|-=|&=|\^=|\|=|##|::|\.\*"
_DIGRAPHS = r"%:%:|<:|:>|<%|%>|%:|"
_SINGLE = r"[{}\[\]();:?.~!+\-*/%^&|=<>,#]"


@dataclass(frozen=True)
class LexMode:
    """The lexical grammar of one language: which C++ token forms exist.

    ``cpp`` selects C++ preprocessing tokens over the reduced GLSL set. The other switches follow the language
    standard each compiler implements; Clang derives them from its language options, so they differ between
    Clang's C++20, the Metal compiler (C++11 to C++17 by ``-std``, never raw strings) and DXC's HLSL mode (Clang
    3.7: no digraphs, raw strings, digit separators, ``u8''`` or ``and``-style operator names).
    """

    cpp: bool = True
    digraphs: bool = True
    raw_strings: bool = True
    digit_separators: bool = True
    unicode_literals: bool = True  # u"", U"", u8"", u'' and U'' (C++11)
    utf8_characters: bool = True  # u8'' (C++17)
    dollar_identifiers: bool = True


CPP_MODE = LexMode()
GLSL_MODE = LexMode(
    cpp=False,
    digraphs=False,
    raw_strings=False,
    digit_separators=False,
    dollar_identifiers=False,
)


@lru_cache(maxsize=None)
def _master(mode: LexMode):
    separator = r"'[0-9A-Za-z_]|" if mode.digit_separators else ""
    number = r"\.?[0-9](?:[eEpP][+-]|" + separator + r"[0-9A-Za-z_.])*"
    identifier = (
        r"[A-Za-z_$][A-Za-z0-9_$]*"
        if mode.dollar_identifiers
        else r"[A-Za-z_][A-Za-z0-9_]*"
    )
    string_prefix = r"(?:u8|u|U|L)?" if mode.unicode_literals else r"L?"
    char_prefix = (
        string_prefix
        if mode.utf8_characters or not mode.unicode_literals
        else r"(?:u|U|L)?"
    )
    parts = [
        ("ws", r"[ \t\f\v]+"),
        ("nl", r"\n"),
        ("block", r"/\*"),
        ("line", r"//[^\n]*"),
    ]
    if mode.raw_strings:
        parts.append(("raw", string_prefix + r'R"([^ ()\\\t\v\f\n"]{0,16})\('))
    parts += [
        ("string", string_prefix + r'"(?:[^"\\\n]|\\.)*"'),
        ("char", char_prefix + r"'(?:[^'\\\n]|\\.)*'"),
        ("number", number),
        ("ident", identifier),
        (
            "punct",
            ((_DIGRAPHS if mode.digraphs else "") + _PUNCTUATORS) + "|" + _SINGLE,
        ),
        ("quote", r"[\"']"),
        ("other", r"."),
    ]
    return re.compile(
        "|".join(f"(?P<{name}>{pattern})" for name, pattern in parts), re.S
    )


def resolve_mode(cpp: bool = True, mode: LexMode | None = None) -> LexMode:
    """``mode`` if given, else full C++ (``cpp``) or GLSL."""
    if mode is not None:
        return mode
    return CPP_MODE if cpp else GLSL_MODE


_KINDS = {
    "string": STRING,
    "char": CHAR,
    "number": NUMBER,
    "ident": IDENT,
    "punct": PUNCT,
    "quote": UNTERMINATED,
    "other": OTHER,
}


def lex(
    source: SourceFile, *, cpp: bool = True, mode: LexMode | None = None
) -> list[Token]:
    """Tokenize ``source`` in ``mode`` (default: C++ when ``cpp``, else GLSL); the result ends with a NEWLINE."""
    text = source.text
    mode = resolve_mode(cpp, mode)
    master = _master(mode)
    tokens: list[Token] = []
    position, space = 0, False
    end = len(text)
    while position < end:
        match = master.match(text, position)
        group = match.lastgroup
        if group == "ws":
            space = True
            position = match.end()
            continue
        if group == "nl":
            tokens.append(Token(NEWLINE, "\n", source, position))
            space = False
            position = match.end()
            continue
        if group == "line":
            space = True
            position = match.end()
            continue
        if group == "block":
            close = text.find("*/", match.end())
            if close < 0:
                raise TranslationDiagnostic(
                    "unterminated /* comment", span=source.span(position), code=LEX_CODE
                )
            space = True
            position = close + 2
            continue
        if group == "raw":
            delimiter = match.group(match.lastindex + 1)
            close = text.find(")" + delimiter + '"', match.end())
            if close < 0:
                raise TranslationDiagnostic(
                    "unterminated raw string literal",
                    span=source.span(position),
                    code=LEX_CODE,
                )
            stop = close + len(delimiter) + 2
            tokens.append(Token(STRING, text[position:stop], source, position, space))
            space = False
            position = stop
            continue
        spelling = match.group()
        if group == "punct" and mode.digraphs and spelling == "<:":
            # [lex.pptoken]: '<::' not followed by ':' or '>' is '<' then '::'.
            after = text[match.end() : match.end() + 2]
            if after[:1] == ":" and after[1:2] not in (":", ">"):
                spelling = "<"
        tokens.append(Token(_KINDS[group], spelling, source, position, space))
        space = False
        position += len(spelling)
    if not tokens or tokens[-1].kind != NEWLINE:
        tokens.append(Token(NEWLINE, "\n", source, end))
    return tokens


def lex_text(
    text: str, *, cpp: bool = True, mode: LexMode | None = None
) -> list[Token]:
    """Tokens of a string that is not a file (a pasted or stringized spelling); no trailing NEWLINE."""
    tokens = lex(SourceFile(None, text), cpp=cpp, mode=mode)
    return [token for token in tokens if token.kind != NEWLINE]


def spell(tokens) -> str:
    """Spelling of a token sequence with single spaces where whitespace separated tokens."""
    out = []
    for index, token in enumerate(tokens):
        if index and token.space:
            out.append(" ")
        out.append(token.text)
    return "".join(out)
