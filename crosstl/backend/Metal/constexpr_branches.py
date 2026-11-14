"""Compile-time facts the Metal static-assertion pass needs from instantiated templates.

Two rules from C++17, which Metal Shading Language follows, decide whether a ``static_assert`` in a materialized
function template can be checked and whether it must be:

* ``decltype(local)::member`` and ``Owner::member`` name a static data member. When that member is a constant
  integral expression, so is the use ([expr.const]), so a ``constexpr`` local such as
  ``constexpr int stride = decltype(tile)::kElemsPerTile * 32;`` has a translation-time value.
* During the instantiation of a templated entity, the substatement an ``if constexpr`` discards is not instantiated
  ([stmt.if]/2), so a ``static_assert`` inside it is never evaluated. Outside a template a discarded statement is
  still fully checked, so callers only pass the bodies of materialized function templates.

The helpers work on text whose comments and literals the caller has masked with spaces (same length as the source).
Any construct they cannot delimit exactly is left alone, so the caller keeps evaluating, and refusing, as before.
"""

import re
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

from .local_decltypes import substitute_local_decltypes

__all__ = [
    "discarded_constexpr_branch_spans",
    "substitute_static_member_constants",
]

Span = Tuple[int, int]

_IF_CONSTEXPR_RE = re.compile(r"(?<![\w:.])if\s+constexpr\s*\(")
_IF_RE = re.compile(r"if(?:\s+constexpr)?\s*\(")
_ELSE_RE = re.compile(r"else(?![\w])")
_STATIC_MEMBER_RE = re.compile(
    r"(?<![\w:.])(?P<owner>[A-Za-z_]\w*(?:\s*<[^<>;(){}]*>)?"
    r"(?:\s*::\s*[A-Za-z_]\w*(?:\s*<[^<>;(){}]*>)?)*)\s*::\s*"
    r"(?P<member>[A-Za-z_]\w*)\b(?!\s*[(<])"
)
_INTEGER_RE = re.compile(r"\d+")
# Else-if chains deeper than this are not delimited.
_MAX_CHAIN_DEPTH = 64


def substitute_static_member_constants(
    expression: str,
    declared_types: Mapping[str, str],
    lookup: Callable[[str, str], Optional[str]],
) -> str:
    """Replace ``decltype(local)::member`` and ``Owner::member`` with the value of the static member.

    ``declared_types`` maps a visible local to its declared type, and ``lookup(owner, member)`` returns the folded
    value of a constant static data member, an integer literal optionally wrapped in a functional cast to the
    member's type, or ``None``. The owner may be a template-id such as ``Frag<float, 1>``. Only folded values are
    inserted, so no identifier from the owner's scope leaks into the caller's scope. A use followed by ``(`` or
    ``<`` names a function or template, never a data member, and stays untouched.
    """
    text = substitute_local_decltypes(expression, declared_types)
    if "::" not in text:
        return expression if text == expression else text

    pieces: List[str] = []
    cursor = 0
    for match in _STATIC_MEMBER_RE.finditer(text):
        owner = re.sub(r"\s*::\s*", "::", match.group("owner"))
        owner = re.sub(r"\s+", "", owner)
        value = lookup(owner, match.group("member"))
        if value is None:
            continue
        pieces.append(text[cursor : match.start()])
        pieces.append(value if _INTEGER_RE.fullmatch(value) else f"({value})")
        cursor = match.end()
    if not pieces:
        return text
    pieces.append(text[cursor:])
    return "".join(pieces)


def _skip_space(masked: str, index: int, end: int) -> int:
    while index < end and masked[index].isspace():
        index += 1
    return index


def _matching(masked: str, start: int, end: int, opener: str, closer: str) -> int:
    """Index just past the delimiter closing ``masked[start]``, or ``-1``."""
    depth = 0
    for index in range(start, end):
        character = masked[index]
        if character == opener:
            depth += 1
        elif character == closer:
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def _statement_end(masked: str, start: int, end: int, depth: int = 0) -> int:
    """End of the compound statement or brace-bodied ``if`` chain at ``start``, or ``-1``."""
    index = _skip_space(masked, start, end)
    if index >= end or depth > _MAX_CHAIN_DEPTH:
        return -1
    if masked[index] == "{":
        return _matching(masked, index, end, "{", "}")
    match = _IF_RE.match(masked, index, end)
    if match is None or (
        index > 0 and (masked[index - 1].isalnum() or masked[index - 1] == "_")
    ):
        return -1
    condition_end = _matching(masked, match.end() - 1, end, "(", ")")
    if condition_end < 0:
        return -1
    then_end = _statement_end(masked, condition_end, end, depth + 1)
    if then_end < 0:
        return -1
    else_start, else_end = _else_branch(masked, then_end, end, depth + 1)
    if else_end == -2:
        return -1
    return then_end if else_start < 0 else else_end


def _else_branch(masked: str, then_end: int, end: int, depth: int) -> Tuple[int, int]:
    """``(start, end)`` of the ``else`` substatement after ``then_end``; ``(-1, -1)`` when absent, ``(-1, -2)`` when it cannot be delimited."""
    index = _skip_space(masked, then_end, end)
    match = _ELSE_RE.match(masked, index, end)
    if match is None:
        return -1, -1
    statement_end = _statement_end(masked, match.end(), end, depth)
    if statement_end < 0:
        return -1, -2
    return _skip_space(masked, match.end(), end), statement_end


def discarded_constexpr_branch_spans(
    masked: str,
    regions: Sequence[Span],
    evaluate: Callable[[int, int], Optional[bool]],
) -> List[Span]:
    """Spans of the substatements each ``if constexpr`` in ``regions`` discards.

    ``evaluate(start, end)`` folds the condition text at ``masked[start:end]`` to ``True``/``False`` or returns ``None``. Only brace-bodied
    branches and ``else if`` chains of them are delimited; an ``if constexpr`` whose condition does not fold or
    whose branches cannot be delimited contributes nothing. Nested statements of a discarded branch are covered by
    its span. The result is sorted.
    """
    spans: List[Span] = []
    for region_start, region_end in sorted(regions):
        for match in _IF_CONSTEXPR_RE.finditer(masked, region_start, region_end):
            if any(start <= match.start() < stop for start, stop in spans):
                continue
            open_paren = match.end() - 1
            condition_end = _matching(masked, open_paren, region_end, "(", ")")
            if condition_end < 0:
                continue
            then_start = _skip_space(masked, condition_end, region_end)
            then_end = _statement_end(masked, then_start, region_end)
            if then_end < 0:
                continue
            else_start, else_end = _else_branch(masked, then_end, region_end, 0)
            if else_end == -2:
                continue
            value = evaluate(open_paren + 1, condition_end - 1)
            if value is None:
                continue
            if not value:
                spans.append((then_start, then_end))
            elif else_start >= 0:
                spans.append((else_start, else_end))
    return sorted(spans)
