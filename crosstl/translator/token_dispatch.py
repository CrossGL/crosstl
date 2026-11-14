"""First-character dispatch for ordered regular-expression token tables.

The CrossGL, Metal, HLSL and GLSL lexers match an ordered table of token
patterns at each position and keep the first pattern that matches. Trying
every pattern at every position makes each token cost a pass over the whole
table: a space walks past every keyword before it reaches ``WHITESPACE``.

This module computes, for every pattern, a superset of the ASCII characters a
match can begin with. Only the patterns that can begin with the character at
the current position are tried, still in table order, so the first matching
pattern is the same one the full table would have chosen. Positions holding a
non-ASCII character use the full table, because Unicode categories are not
analysed. Whenever the analysis meets a construct it does not model it
assumes every ASCII character, which costs speed but never changes a result.
"""

from __future__ import annotations

import functools
import re
from typing import Pattern, Sequence, Tuple

try:  # Python 3.11+ exposes the regular-expression parser here.
    from re import _constants as _sre
    from re import _parser as _sre_parse
except ImportError:  # pragma: no cover - only an unexpected runtime lacks it
    _sre = None
    _sre_parse = None

ASCII_LIMIT = 128
_ALL_ASCII: frozenset[int] = frozenset(range(ASCII_LIMIT))
_NOTHING: frozenset[int] = frozenset()

TokenTable = Tuple[Tuple[str, str], ...]


def _category_members() -> dict:
    if _sre is None:
        return {}
    spellings = {
        _sre.CATEGORY_DIGIT: r"\d",
        _sre.CATEGORY_NOT_DIGIT: r"\D",
        _sre.CATEGORY_SPACE: r"\s",
        _sre.CATEGORY_NOT_SPACE: r"\S",
        _sre.CATEGORY_WORD: r"\w",
        _sre.CATEGORY_NOT_WORD: r"\W",
    }
    return {
        category: frozenset(
            code
            for code in range(ASCII_LIMIT)
            if re.match(spelling, chr(code)) is not None
        )
        for category, spelling in spellings.items()
    }


_CATEGORY_MEMBERS = _category_members()


def _set_first(items) -> frozenset[int]:
    """ASCII members of a parsed ``[...]`` set, or every ASCII character."""

    members = set()
    for op, value in items:
        if op is _sre.LITERAL:
            if value < ASCII_LIMIT:
                members.add(value)
        elif op is _sre.RANGE:
            low, high = value
            members.update(range(low, min(high, ASCII_LIMIT - 1) + 1))
        elif op is _sre.CATEGORY and value in _CATEGORY_MEMBERS:
            members.update(_CATEGORY_MEMBERS[value])
        else:
            # NEGATE and anything not modelled: assume every character.
            return _ALL_ASCII
    return frozenset(members)


def _item_first(op, value) -> tuple[frozenset[int], bool]:
    """Return (first characters, can match empty) for one parsed item."""

    if op is _sre.LITERAL:
        return (frozenset({value}) if value < ASCII_LIMIT else _NOTHING), False
    if op is _sre.IN:
        return _set_first(value), False
    if op in (_sre.NOT_LITERAL, _sre.ANY):
        return _ALL_ASCII, False
    if op in (_sre.AT, _sre.ASSERT, _sre.ASSERT_NOT):
        # Zero-width: the next item supplies the first character.
        return _NOTHING, True
    if op is _sre.BRANCH:
        first: set = set()
        nullable = False
        for alternative in value[1]:
            alternative_first, alternative_nullable = _sequence_first(alternative)
            first |= alternative_first
            nullable = nullable or alternative_nullable
        return frozenset(first), nullable
    if op is _sre.SUBPATTERN:
        _group, add_flags, _del_flags, pattern = value
        if add_flags & _sre.SRE_FLAG_IGNORECASE:
            return _ALL_ASCII, True
        return _sequence_first(pattern)
    if op in _REPEATS:
        minimum, _maximum, pattern = value
        first, nullable = _sequence_first(pattern)
        return first, nullable or minimum == 0
    if op is _ATOMIC_GROUP:
        return _sequence_first(value)
    return _ALL_ASCII, True


def _sequence_first(pattern) -> tuple[frozenset[int], bool]:
    first: set = set()
    for op, value in pattern:
        item_first, nullable = _item_first(op, value)
        first |= item_first
        if not nullable:
            return frozenset(first), False
    return frozenset(first), True


if _sre is not None:
    _REPEATS = tuple(
        getattr(_sre, name)
        for name in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT")
        if hasattr(_sre, name)
    )
    _ATOMIC_GROUP = getattr(_sre, "ATOMIC_GROUP", object())
else:  # pragma: no cover
    _REPEATS = ()
    _ATOMIC_GROUP = object()


@functools.lru_cache(maxsize=None)
def pattern_first_chars(pattern: str) -> frozenset[int]:
    """Return a superset of the ASCII code points a match can start with.

    A pattern that can match the empty string, uses case-insensitive
    matching, or contains a construct the analysis does not model is reported
    as able to start with any ASCII character.
    """

    if _sre_parse is None:
        return _ALL_ASCII
    try:
        parsed = _sre_parse.parse(pattern)
    except (re.error, RecursionError, TypeError):
        return _ALL_ASCII
    if parsed.state.flags & _sre.SRE_FLAG_IGNORECASE:
        return _ALL_ASCII
    try:
        first, nullable = _sequence_first(parsed)
    except (AttributeError, TypeError, ValueError):
        return _ALL_ASCII
    return _ALL_ASCII if nullable else first


def _ascii_buckets(table: TokenTable) -> tuple[tuple[int, ...], ...]:
    firsts = [pattern_first_chars(pattern) for _name, pattern in table]
    return tuple(
        tuple(index for index, first in enumerate(firsts) if code in first)
        for code in range(ASCII_LIMIT)
    )


class OrderedPatternDispatch:
    """Per-character candidate lists for a first-match-wins pattern list."""

    def __init__(self, table: TokenTable) -> None:
        self.patterns: tuple[tuple[str, Pattern[str]], ...] = tuple(
            (name, re.compile(pattern)) for name, pattern in table
        )
        self.buckets: tuple[tuple[tuple[str, Pattern[str]], ...], ...] = tuple(
            tuple(self.patterns[index] for index in bucket)
            for bucket in _ascii_buckets(table)
        )

    def candidates(self, char: str) -> tuple[tuple[str, Pattern[str]], ...]:
        """Return the patterns, in table order, that can match at ``char``."""

        code = ord(char)
        return self.buckets[code] if code < ASCII_LIMIT else self.patterns

    def match(self, text: str, pos: int):
        """Return ``(name, match)`` for the first table pattern matching at pos."""

        for name, pattern in self.candidates(text[pos]):
            match = pattern.match(text, pos)
            if match:
                return name, match
        return None


class AlternationDispatch:
    """Per-character named-group alternations for a first-match-wins table."""

    def __init__(self, table: TokenTable) -> None:
        self._table = table
        self.full: Pattern[str] = self._alternation(range(len(table)))
        self._indices = _ascii_buckets(table)
        self._compiled: list[Pattern[str] | None] = [None] * ASCII_LIMIT
        self._ready = [False] * ASCII_LIMIT

    def _alternation(self, indices) -> Pattern[str] | None:
        parts = [
            f"(?P<{self._table[index][0]}>{self._table[index][1]})" for index in indices
        ]
        return re.compile("|".join(parts)) if parts else None

    def pattern_for(self, char: str) -> Pattern[str] | None:
        """Return the alternation to match at ``char`` (``None``: no token)."""

        code = ord(char)
        if code >= ASCII_LIMIT:
            return self.full
        if not self._ready[code]:
            self._compiled[code] = self._alternation(self._indices[code])
            self._ready[code] = True
        return self._compiled[code]

    def match(self, text: str, pos: int):
        pattern = self.pattern_for(text[pos])
        return pattern.match(text, pos) if pattern is not None else None


def _freeze(tokens) -> TokenTable:
    items = tokens.items() if hasattr(tokens, "items") else tokens
    return tuple((str(name), str(pattern)) for name, pattern in items)


@functools.lru_cache(maxsize=32)
def _ordered_pattern_dispatch(table: TokenTable) -> OrderedPatternDispatch:
    return OrderedPatternDispatch(table)


@functools.lru_cache(maxsize=32)
def _alternation_dispatch(table: TokenTable) -> AlternationDispatch:
    return AlternationDispatch(table)


def ordered_pattern_dispatch(tokens: Sequence) -> OrderedPatternDispatch:
    """Return the shared dispatch for a ``[(name, pattern), ...]`` table."""

    return _ordered_pattern_dispatch(_freeze(tokens))


def alternation_dispatch(tokens) -> AlternationDispatch:
    """Return the shared dispatch for a ``{name: pattern}`` table."""

    return _alternation_dispatch(_freeze(tokens))
