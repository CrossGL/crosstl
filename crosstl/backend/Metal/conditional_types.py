"""Select the branch a Metal ``conditional_t`` type spelling names.

``metal::conditional_t<C, A, B>`` and ``metal::conditional<C, A, B>::type``
name ``A`` when the constant expression ``C`` is true and ``B`` otherwise.
Materialized templates often select a helper struct this way, for example a
loader chosen from template parameters, and then call its methods. The
helpers here pick the branch only when ``C`` folds to a constant; every other
spelling yields ``None`` so callers keep failing closed.
"""

import re
from typing import Callable, List, Optional, Tuple

__all__ = [
    "select_conditional_type",
    "select_nested_conditional_types",
    "split_template_arguments",
]

_CONDITIONAL_RE = re.compile(
    r"(?:typename\s+)?(?:::\s*)?(?:metal\s*::\s*)?"
    r"(?P<name>conditional_t|conditional)\s*<"
)
_TYPENAME_RE = re.compile(r"^typename\s+")
# A selection nested in a larger spelling, e.g. a template argument of
# ``Frag<metal::conditional_t<C, A, B>, 8, 8>``.
_NESTED_CONDITIONAL_RE = re.compile(
    r"(?<![\w:])(?:typename\s+)?(?:::\s*)?(?:metal\s*::\s*)?"
    r"(?P<name>conditional_t|conditional)\s*<"
)
_TYPE_MEMBER_RE = re.compile(r"\s*::\s*type\b")
# Nested selections deeper than this are not followed.
_MAX_DEPTH = 16


def _closing_angle(text: str, start: int) -> Optional[int]:
    """Index of the ``>`` closing the template-argument list opened at ``start``.

    ``<=``, ``>=``, ``<<`` and ``->`` are operators. ``>>`` closes two lists,
    as in C++11. A ``<`` opens a nested list only after an identifier, so a
    comparison such as ``0 < 4`` in a folded condition stays an operator.
    Angles inside parentheses, brackets or braces belong to the value
    expression there and never close the outer list.
    """
    depth = 1
    nesting = 0
    index = start + 1
    while index < len(text):
        pair = text[index : index + 2]
        character = text[index]
        if pair in ("<=", ">=", "<<", "->"):
            index += 2
            continue
        if character in "([{":
            nesting += 1
        elif character in ")]}":
            nesting -= 1
            if nesting < 0:
                return None
        elif nesting == 0 and character == "<":
            previous = text[:index].rstrip()
            if previous and (previous[-1].isalnum() or previous[-1] == "_"):
                if not re.search(r"(?:^|[^\w])\d+$", previous):
                    depth += 1
        elif nesting == 0 and character == ">":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def split_template_arguments(text: str) -> Optional[List[str]]:
    """Split a template-argument list body at its top-level commas.

    Returns ``None`` when a nested list or a parenthesized expression does
    not close inside ``text``.
    """
    parts: List[str] = []
    current_start = 0
    nesting = 0
    index = 0
    while index < len(text):
        pair = text[index : index + 2]
        character = text[index]
        if pair in ("<=", ">=", "<<", "->", ">>"):
            index += 2
            continue
        if character in "([{":
            nesting += 1
        elif character in ")]}":
            nesting -= 1
            if nesting < 0:
                return None
        elif nesting == 0 and character == "<":
            closing = _closing_angle(text, index)
            previous = text[:index].rstrip()
            opens = (
                previous
                and (previous[-1].isalnum() or previous[-1] == "_")
                and not re.search(r"(?:^|[^\w])\d+$", previous)
            )
            if opens:
                if closing is None:
                    return None
                index = closing + 1
                continue
        elif nesting == 0 and character == ",":
            parts.append(text[current_start:index].strip())
            current_start = index + 1
        index += 1
    if nesting:
        return None
    parts.append(text[current_start:].strip())
    return parts


def _conditional_parts(text: str) -> Optional[Tuple[str, str, str]]:
    match = _CONDITIONAL_RE.match(text)
    if match is None:
        return None
    opening = match.end() - 1
    closing = _closing_angle(text, opening)
    if closing is None:
        return None
    rest = text[closing + 1 :].strip()
    if match.group("name") == "conditional_t":
        if rest:
            return None
    elif re.fullmatch(r"::\s*type", rest) is None:
        # The bare metafunction names a type only through ``::type``.
        return None
    arguments = split_template_arguments(text[opening + 1 : closing])
    if arguments is None or len(arguments) != 3 or not all(arguments):
        return None
    return arguments[0], arguments[1], arguments[2]


def select_conditional_type(
    text: str, evaluate: Callable[[str], Tuple[bool, Optional[int]]]
) -> Optional[str]:
    """Return the type a ``conditional_t`` spelling selects, or ``None``.

    ``evaluate`` folds a condition and returns ``(folded, value)``. Nested
    selections in the chosen branch are followed. The result has no leading
    ``typename``. ``text`` must already be free of comments.
    """
    current = _TYPENAME_RE.sub("", text.strip())
    selected = False
    for _ in range(_MAX_DEPTH):
        parts = _conditional_parts(current)
        if parts is None:
            return current if selected else None
        condition, when_true, when_false = parts
        folded, value = evaluate(condition)
        if not folded or value is None:
            return None
        current = _TYPENAME_RE.sub("", (when_true if value else when_false).strip())
        selected = True
    return None


def select_nested_conditional_types(
    text: str, evaluate: Callable[[str], Tuple[bool, Optional[int]]]
) -> str:
    """Replace every selection inside ``text`` whose condition folds.

    ``Frag<metal::conditional_t<1 == 2, half, float>, 8, 8>`` becomes
    ``Frag<float, 8, 8>``. Selections whose condition does not fold, and
    spellings that are not selections, are kept unchanged.
    """
    result: List[str] = []
    position = 0
    for _ in range(len(text)):
        match = _NESTED_CONDITIONAL_RE.search(text, position)
        if match is None:
            break
        closing = _closing_angle(text, match.end() - 1)
        if closing is None:
            break
        end = closing + 1
        if match.group("name") == "conditional":
            member = _TYPE_MEMBER_RE.match(text, end)
            end = member.end() if member is not None else end
        selected = select_conditional_type(text[match.start() : end], evaluate)
        result.append(text[position : match.start()])
        if selected is None:
            result.append(text[match.start() : match.end()])
            position = match.end()
            continue
        result.append(select_nested_conditional_types(selected, evaluate))
        position = end
    result.append(text[position:])
    return "".join(result)
