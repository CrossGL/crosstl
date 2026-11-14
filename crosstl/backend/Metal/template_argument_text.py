"""Normalized spellings of Metal template arguments, memoized by their text.

The Metal importer compares template-ids by the normalized spelling of their arguments: comments and redundant
whitespace removed, and redundant parentheses dropped around an integral or Boolean literal. Template
materialization asks for the same few hundred spellings millions of times on a library of a few hundred kernels,
once for every specialization it compares a template-id with, so the spelling is computed once per distinct text.

The normalization is a pure function of the text, so a memo shared by every preprocessor returns exactly what a
fresh computation does. It is bounded and starts over once it holds :data:`TEMPLATE_ARGUMENT_TEXT_MEMO_LIMIT`
spellings.
"""

from __future__ import annotations

import re
from typing import Callable

__all__ = [
    "TEMPLATE_ARGUMENT_TEXT_MEMO_LIMIT",
    "normalize_template_argument_text",
]

#: Distinct argument spellings the memo keeps before it starts over.
TEMPLATE_ARGUMENT_TEXT_MEMO_LIMIT = 1 << 16

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_SPACING_RE = re.compile(r"\s*([<>,:*&\[\](){}])\s*")
_LITERAL_RE = re.compile(
    r"(?:true|false|[-+]?(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|\d+)[uUlL]*)"
)

_memo: dict[str, str] = {}


def normalize_template_argument_text(
    value: str,
    strip_comments: Callable[[str], str],
    find_matching_delimiter: Callable[[str, int, str, str], int | None],
) -> str:
    """The normalized spelling of one template argument.

    ``strip_comments`` removes comments outside string literals and ``find_matching_delimiter(text, start, opener,
    closer)`` returns the index of the delimiter that closes the one at ``start``. Both must be pure functions of
    their arguments, as the Metal preprocessor's scanners are.
    """
    try:
        return _memo[value]
    except KeyError:
        pass
    normalized = _normalize(value, strip_comments, find_matching_delimiter)
    if len(_memo) >= TEMPLATE_ARGUMENT_TEXT_MEMO_LIMIT:
        _memo.clear()
    _memo[value] = normalized
    return normalized


def _normalize(
    value: str,
    strip_comments: Callable[[str], str],
    find_matching_delimiter: Callable[[str, int, str, str], int | None],
) -> str:
    value = strip_comments(value).strip()
    if not value:
        return ""
    collapsed = _WHITESPACE_RE.sub(" ", value)
    normalized = _PUNCTUATION_SPACING_RE.sub(r"\1", collapsed).strip()
    # Redundant parentheses around a literal non-type template argument do not change its value. Strip them only
    # after proving that the complete nested spelling reduces to an integral or Boolean literal. Grouping remains
    # semantic for expressions such as `(2 > 2)`: removing that pair would turn a recursive
    # `Struct<..., (2 > 2)>` substitution into the ambiguous token sequence `Struct<..., 2 > 2>`.
    candidate = normalized
    while candidate.startswith("(") and candidate.endswith(")"):
        closing = find_matching_delimiter(candidate, 0, "(", ")")
        if closing != len(candidate) - 1:
            break
        inner = candidate[1:-1].strip()
        if not inner:
            break
        candidate = inner
    if _LITERAL_RE.fullmatch(candidate):
        normalized = candidate
    return normalized
