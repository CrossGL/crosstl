"""Text output of the shared preprocessor, with a line map back to the source.

Tokens are written with one space where the source had whitespace, and with a space wherever two adjacent tokens
would otherwise lex as a different token (``+`` ``+``, ``x`` ``1``, ``"a"`` ``s``). An output line holds the
tokens between two newlines of the macro-replaced stream and maps to the source line of its first token. Within
one file, blank lines keep every output line at the line it came from, so a front end's line numbers match the
source for code without includes; ``line_map`` gives the (file, line) of every output line in all cases.
"""

from __future__ import annotations

from .lexer import (
    CHAR,
    IDENT,
    NEWLINE,
    NUMBER,
    PUNCT,
    STRING,
    LexMode,
    Token,
    lex_text,
    resolve_mode,
)

_WORDS = (IDENT, NUMBER)
_LITERALS = (STRING, CHAR)


def _mode(mode: bool | LexMode) -> LexMode:
    return resolve_mode(mode) if isinstance(mode, bool) else mode


def needs_space(previous: Token, token: Token, mode: bool | LexMode) -> bool:
    """Whether writing ``token`` right after ``previous`` would change how the pair lexes in ``mode``."""
    if previous.kind in _WORDS and token.kind in _WORDS:
        return True
    if (previous.kind in _WORDS and token.kind in _LITERALS) or (
        previous.kind in _LITERALS and token.kind == IDENT
    ):
        return True  # an encoding prefix or a user-defined-literal suffix would attach
    if previous.kind == NUMBER and (
        token.text[:1] in (".", "'")
        or (token.text[:1] in ("+", "-") and previous.text[-1:] in "eEpP")
    ):
        return True  # the pair would continue the pp-number
    if previous.text == "." and token.kind == NUMBER:
        return True
    if previous.text == "/" and token.text[:1] in ("/", "*"):
        return True
    if previous.kind == PUNCT and token.kind == PUNCT:
        joined = lex_text(previous.text + token.text, mode=_mode(mode))
        return len(joined) != 2 or joined[0].text != previous.text
    return False


class Writer:
    def __init__(self, mode: bool | LexMode):
        self.mode = _mode(mode)
        self.lines: list[str] = []
        self.line_map: list[tuple[str | None, int]] = []
        self._parts: list[str] = []
        self._origin: tuple[object, int] | None = None
        self._previous: Token | None = None
        self._last: tuple[object, int] | None = None

    @staticmethod
    def _location(token: Token):
        if token.source is None:
            return None, 0
        return token.source, token.source.position(token.offset)[0]

    def _open(self, token: Token) -> None:
        source, line = self._location(token)
        if self._last is None and source is not None:
            self._last = (source, 0)
        if self._last is not None and source is not None and self._last[0] is source:
            for padding in range(self._last[1] + 1, line):
                self.lines.append("")
                self.line_map.append((source.path, padding))
        self._origin = (source, line)
        self._parts = []
        self._previous = None

    def _close(self) -> None:
        if self._origin is None:
            return
        source, line = self._origin
        self.lines.append("".join(self._parts))
        self.line_map.append((getattr(source, "path", None), line))
        self._last = self._origin
        self._origin = None

    def write(self, tokens: list[Token]) -> None:
        for token in tokens:
            if token.kind == NEWLINE:
                self._close()
                continue
            if self._origin is None:
                self._open(token)
            elif token.space or needs_space(self._previous, token, self.mode):
                self._parts.append(" ")
            self._parts.append(token.text)
            self._previous = token

    def directive(self, tokens: list[Token]) -> None:
        """Write a kept directive (``#pragma``, ``#version``, ``#extension``, a kept ``#include``) as a line."""
        self._close()
        self._open(tokens[0])
        for index, token in enumerate(tokens):
            if index == 2 or (
                index > 2
                and (token.space or needs_space(tokens[index - 1], token, self.mode))
            ):
                self._parts.append(" ")
            self._parts.append(token.text)
        self._close()

    def text(self) -> str:
        self._close()
        return "".join(line + "\n" for line in self.lines)
