"""The WGSL lexer: the tokens of a WGSL module (W3C WGSL, section 3).

Blankspace and comments separate tokens; block comments nest (section 3.2). Numeric literals keep their spelling,
suffix included, so the importer can tell an abstract literal from an ``i32``, ``u32``, ``f32`` or ``f16`` one
(section 3.4). A ``<`` that opens a template list and the ``>`` that closes it are found as the template list
discovery algorithm of section 3.9 finds them, on the source text before it is split into tokens, and become
``template_start`` and ``template_end`` tokens; any other ``<`` and ``>`` are operators. Text the grammar cannot
start a token with is refused with :class:`WgslSyntaxError`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["KEYWORDS", "Token", "WgslLexer", "WgslSyntaxError", "discover_templates"]


class WgslSyntaxError(ValueError):
    """Text that is not a WGSL module, with the line it was found on."""

    def __init__(self, message: str, line: int | None = None):
        self.line = line
        super().__init__(f"line {line}: {message}" if line else message)


# WGSL section 15.1 keywords. ``true`` and ``false`` are boolean literals.
KEYWORDS = frozenset(
    """alias break case const const_assert continue continuing default diagnostic discard else enable false fn
    for if let loop override requires return struct switch true var while""".split()
)

# Section 3.4: hexadecimal and decimal floating-point literals, then integer literals. The longest match wins.
_NUMBER_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"0[xX][0-9a-fA-F]*\.[0-9a-fA-F]+(?:[pP][+-]?[0-9]+[fh]?)?",
        r"0[xX][0-9a-fA-F]+\.[0-9a-fA-F]*(?:[pP][+-]?[0-9]+[fh]?)?",
        r"0[xX][0-9a-fA-F]+[pP][+-]?[0-9]+[fh]?",
        r"0[xX][0-9a-fA-F]+[iu]?",
        r"[0-9]*\.[0-9]+(?:[eE][+-]?[0-9]+)?[fh]?",
        r"[0-9]+\.[0-9]*(?:[eE][+-]?[0-9]+)?[fh]?",
        r"[0-9]+[eE][+-]?[0-9]+[fh]?",
        r"(?:0|[1-9][0-9]*)[fh]",
        r"(?:0|[1-9][0-9]*)[iu]?",
    )
)
_INTEGER = re.compile(r"(?:0[xX][0-9a-fA-F]+|0|[1-9][0-9]*)[iu]?")
_IDENT = re.compile(r"[^\W\d]\w*", re.UNICODE)
_BLANK = re.compile(r"[ \t\n\r\v\f\u0085\u200e\u200f\u2028\u2029]+")
# Longest first, so that ``<<=`` is one token, not ``<`` ``<`` ``=``.
_OPERATORS = (
    "<<=",
    ">>=",
    "->",
    "&&",
    "||",
    "==",
    "!=",
    "<=",
    ">=",
    "<<",
    ">>",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "++",
    "--",
    *"&|^~!+-*/%<>=()[]{},.;:@",
)


@dataclass(frozen=True)
class Token:
    """One token: ``kind`` is ``ident``, ``keyword``, ``int``, ``float``, ``op``, ``template_start``,
    ``template_end`` or ``eof``."""

    kind: str
    text: str
    line: int

    def is_op(self, *texts: str) -> bool:
        return self.kind == "op" and self.text in texts

    def is_keyword(self, *texts: str) -> bool:
        return self.kind == "keyword" and self.text in texts


def _skip_comment(text: str, index: int, line: int) -> tuple[int, int]:
    """The position and line after the comment at ``index``, which must start one."""
    if text.startswith("//", index):
        end = text.find("\n", index)
        return (len(text) if end < 0 else end), line
    depth, index = 1, index + 2
    while depth:
        if index >= len(text):
            raise WgslSyntaxError("unterminated block comment", line)
        if text.startswith("/*", index):
            depth, index = depth + 1, index + 2
        elif text.startswith("*/", index):
            depth, index = depth - 1, index + 2
        else:
            line += text[index] == "\n"
            index += 1
    return index, line


def _skip_blank_and_comments(text: str, index: int, line: int = 0) -> tuple[int, int]:
    while index < len(text):
        blank = _BLANK.match(text, index)
        if blank:
            line += blank.group().count("\n")
            index = blank.end()
        elif text.startswith("//", index) or text.startswith("/*", index):
            index, line = _skip_comment(text, index, line)
        else:
            break
    return index, line


def _number(text: str, index: int) -> str | None:
    best = None
    for pattern in _NUMBER_PATTERNS:
        found = pattern.match(text, index)
        if found and (best is None or len(found.group()) > len(best)):
            best = found.group()
    return best


def _pop_deeper(pending: list, depth: int) -> None:
    while pending and pending[-1][1] >= depth:
        pending.pop()


def discover_templates(text: str) -> tuple[frozenset, frozenset]:
    """The positions of the ``<`` that start and the ``>`` that end template lists (WGSL section 3.9)."""
    starts, ends = set(), set()
    pending: list[tuple[int, int]] = []
    index, depth = 0, 0
    while index < len(text):
        index, _ = _skip_blank_and_comments(text, index)
        if index >= len(text):
            break
        char = text[index]
        if char.isdigit() or (
            char == "." and index + 1 < len(text) and text[index + 1].isdigit()
        ):
            index += len(_number(text, index) or char)
            continue
        ident = _IDENT.match(text, index)
        if ident:
            index, _ = _skip_blank_and_comments(text, ident.end())
            if text.startswith("<", index):
                pending.append((index, depth))
                index += 1
                if text.startswith(("<", "="), index):
                    pending.pop()
                    index += 1
            continue
        if char == ">":
            if pending and pending[-1][1] == depth:
                starts.add(pending.pop()[0])
                ends.add(index)
                index += 1
            else:
                index += 2 if text.startswith(">=", index) else 1
        elif char in "([":
            depth += 1
            index += 1
        elif char in ")]":
            _pop_deeper(pending, depth)
            depth = max(0, depth - 1)
            index += 1
        elif char == "!":
            index += 2 if text.startswith("!=", index) else 1
        elif char == "=":
            if text.startswith("==", index):
                index += 2
            else:
                depth, index = 0, index + 1
                pending.clear()
        elif char in ";{:":
            depth, index = 0, index + 1
            pending.clear()
        elif text.startswith(("&&", "||"), index):
            _pop_deeper(pending, depth)
            index += 2
        else:
            index += 1
    return frozenset(starts), frozenset(ends)


class WgslLexer:
    """Splits WGSL source text into :class:`Token` values; :meth:`tokenize` returns them, ending with ``eof``."""

    def __init__(self, code: str):
        self.code = code
        self.tokens: list[Token] = []

    def tokenize(self) -> list[Token]:
        text = self.code
        starts, ends = discover_templates(text)
        tokens, index, line = [], 0, 1
        while True:
            index, line = _skip_blank_and_comments(text, index, line)
            if index >= len(text):
                break
            char = text[index]
            if index in starts:
                tokens.append(Token("template_start", "<", line))
                index += 1
                continue
            if index in ends:
                tokens.append(Token("template_end", ">", line))
                index += 1
                continue
            if char.isdigit() or (
                char == "." and text[index + 1 : index + 2].isdigit()
            ):
                spelling = _number(text, index)
                kind = "int" if _INTEGER.fullmatch(spelling) else "float"
                tokens.append(Token(kind, spelling, line))
                index += len(spelling)
                continue
            ident = _IDENT.match(text, index)
            if ident:
                word = ident.group()
                tokens.append(
                    Token("keyword" if word in KEYWORDS else "ident", word, line)
                )
                index = ident.end()
                continue
            for operator in _OPERATORS:
                if text.startswith(operator, index):
                    tokens.append(Token("op", operator, line))
                    index += len(operator)
                    break
            else:
                raise WgslSyntaxError(f"unexpected character {char!r}", line)
        tokens.append(Token("eof", "", line))
        self.tokens = tokens
        return tokens
