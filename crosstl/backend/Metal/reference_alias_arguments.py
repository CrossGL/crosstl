"""Which call arguments can hand a named object to the callee by reference.

A reference alias such as ``thread const auto& r = tile.frag_at(i, j);`` is
replaced by its storage ``tile.frags[i * C + j]`` only while ``i`` and ``j``
keep their values. A call can change a variable only through a reference or a
pointer to it. A pointer needs ``&i``, which callers reject separately. A
reference parameter binds only an argument that designates the object: the
whole argument is ``i``, possibly parenthesized or followed by member and
subscript accesses ([dcl.init.ref]/5). An operand of arithmetic, such as the
``i`` in ``f(a[(i) * 4 + j])``, is converted to a prvalue first
([conv.lval]/1, [expr.arith.conv]), so no callee can bind it.

A conditional or comma expression may designate one of its operands, so an
argument that contains either at its top level is treated as designating.
"""

from typing import Callable, Optional, Tuple

_OPENERS = "([{"
_CLOSERS = ")]}"


def _argument_bounds(masked: str, open_paren: int, close_paren: int, position: int):
    """Return the ``[start, end)`` bounds of the argument that holds ``position``."""
    depth = 0
    start = open_paren + 1
    for index in range(open_paren + 1, close_paren):
        character = masked[index]
        if character in _OPENERS:
            depth += 1
        elif character in _CLOSERS:
            depth -= 1
        elif character == "," and depth == 0:
            if index > position:
                return start, index
            start = index + 1
    return start, close_paren


def _strip_enclosing_parentheses(masked: str, start: int, end: int) -> Tuple[int, int]:
    while True:
        while start < end and masked[start].isspace():
            start += 1
        while end > start and masked[end - 1].isspace():
            end -= 1
        if not (masked[start : start + 1] == "(" and masked[end - 1 : end] == ")"):
            return start, end
        depth = 0
        for index in range(start, end):
            if masked[index] == "(":
                depth += 1
            elif masked[index] == ")":
                depth -= 1
                if depth == 0 and index != end - 1:
                    return start, end
        start, end = start + 1, end - 1


def _has_top_level_conditional_or_comma(masked: str, start: int, end: int) -> bool:
    depth = 0
    for index in range(start, end):
        character = masked[index]
        if character in _OPENERS:
            depth += 1
        elif character in _CLOSERS:
            depth -= 1
        elif depth == 0 and character in "?,":
            return True
    return False


def call_argument_designates(
    masked: str,
    call_span: Tuple[int, int],
    name_start: int,
    name_end: int,
    access_end: Callable[[int, int], int],
) -> Optional[bool]:
    """Whether the argument of ``call_span`` around a name can bind it by reference.

    ``call_span`` holds the offsets of the call's parentheses, and
    ``access_end(start, limit)`` returns the offset after a member and subscript
    access chain that starts at ``start``. Returns ``None`` when the name is
    not inside the call's parentheses.
    """
    open_paren, close_paren = call_span
    if not open_paren < name_start < close_paren:
        return None
    start, end = _argument_bounds(masked, open_paren, close_paren, name_start)
    start, end = _strip_enclosing_parentheses(masked, start, end)
    if _has_top_level_conditional_or_comma(masked, start, end):
        return True
    return start == name_start and access_end(name_end, end) >= end
