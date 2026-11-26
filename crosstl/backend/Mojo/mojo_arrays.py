"""Mojo's fixed-size arrays and compile-time values as the CrossGL arrays and constants they hold.

``Array[T, N]`` and ``InlineArray[T, N]`` (the standard library's ``collections.inline_array``) hold ``N`` values of
``T`` in place, as a CrossGL ``T[N]`` does; a list literal initializes either one element by element, as a braced
list initializes a CrossGL array. ``materialize[value]()`` (``std.builtin``) makes a run-time copy of the
compile-time ``value``, which in CrossGL is ``value`` itself, since a CrossGL constant is read as a value.
"""

from __future__ import annotations

import re

from ..common_ast import ArrayAccessNode, CallNode, VariableNode

__all__ = ["is_array_type_text", "materialized_value", "mojo_array_type"]

_ARRAY = re.compile(r"^(?:Inline)?Array\[(?P<element>.+),\s*(?P<size>\d+)\]$")


def mojo_array_type(generator, mojo_type):
    """The CrossGL ``T[N]`` of a Mojo ``Array[T, N]`` or ``InlineArray[T, N]``, else ``None``."""
    if not isinstance(mojo_type, str):
        return None
    match = _ARRAY.match(mojo_type.strip())
    if match is None:
        return None
    element = generator.map_type(match.group("element").strip())
    return f"{element}[{int(match.group('size'))}]"


def is_array_type_text(type_text) -> bool:
    """Whether the CrossGL type ``type_text`` is a sized array, such as ``float[3]``."""
    return isinstance(type_text, str) and bool(re.search(r"\[\d+\]$", type_text))


def materialized_value(expr):
    """The value ``materialize[value]()`` copies, else ``None``."""
    callee = getattr(expr, "callee", None)
    if not isinstance(expr, CallNode) or getattr(expr, "args", None):
        return None
    if not isinstance(callee, ArrayAccessNode):
        return None
    function = callee.array
    if isinstance(function, VariableNode) and function.name == "materialize":
        return callee.index
    return None
