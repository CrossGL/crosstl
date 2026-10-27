"""Delimit an array-element receiver such as ``tiles[k]`` in ``tiles[k].frag_at(0)``.

Indexing an array of class type yields an lvalue of that class ([expr.sub]), so a member function called on the
element is called on that object exactly as on a named variable. Struct-method lowering passes the receiver as the
first argument of the lowered function, so the element expression only has to be delimited.

The receiver is accepted only when the declaration supplies exactly one dimension per subscript, so the subscripted
expression is a struct value and never a sub-array, and each index is free of calls, member accesses and
assignments: lowering moves the receiver text without rescanning it, so an index must contain nothing that would
itself need lowering or could change meaning when evaluated as part of an argument list.
"""

import re
from typing import Optional

__all__ = ["array_rank_at", "array_element_receiver_end"]

_UNSAFE_INDEX_RE = re.compile(r"[(){}.;,?]|->|<<=|>>=|(?<![=!<>])=(?!=)|\+\+|--")


def _skip_space(code: str, index: int) -> int:
    while index < len(code) and code[index].isspace():
        index += 1
    return index


def _closing_bracket(code: str, open_index: int) -> Optional[int]:
    depth = 0
    for index in range(open_index, len(code)):
        character = code[index]
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return index
        elif character in ";{}":
            return None
    return None


def array_rank_at(code: str, position: int) -> int:
    """Number of ``[extent]`` groups of the declarator that continues at ``position``."""
    rank = 0
    index = _skip_space(code, position)
    while index < len(code) and code[index] == "[":
        close = _closing_bracket(code, index)
        if close is None:
            return 0
        rank += 1
        index = _skip_space(code, close + 1)
    return rank


def array_element_receiver_end(
    code: str, subscript_open: int, rank: int
) -> Optional[int]:
    """End of ``name[i]...[j]`` starting at ``subscript_open`` when it indexes all ``rank`` dimensions and a ``.``
    member access follows; ``None`` otherwise."""
    if rank < 1:
        return None
    index = subscript_open
    for _ in range(rank):
        if index >= len(code) or code[index] != "[":
            return None
        close = _closing_bracket(code, index)
        if close is None:
            return None
        subscript = code[index + 1 : close]
        if not subscript.strip() or _UNSAFE_INDEX_RE.search(subscript):
            return None
        end = close + 1
        index = _skip_space(code, end)
    if index >= len(code) or code[index] != "." or code.startswith("...", index):
        return None
    return end
