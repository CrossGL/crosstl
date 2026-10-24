"""The integer values of the enumerators a Metal program declares, for attributes that name one.

Metal sources name colour attachments, buffer slots and vertex attributes with the enumerators of the header they
share with the host (``[[color(RenderTargetDepth)]]``, where ``typedef enum { ..., RenderTargetDepth = 3 }``).
:func:`enumerator_values` gives each plain enumerator's integer: its own value when that is an integer literal, or
the previous enumerator's value plus one, as in C (C++17 section 10.2).
"""

from __future__ import annotations

import re

__all__ = ["enumerator_values"]

_INTEGER = re.compile(r"(0[xX][0-9a-fA-F]+|\d+)[uUlL]*")


def enumerator_values(enums, render) -> dict:
    """``{name: value}`` for the enumerators of ``enums``; ``render`` spells an enumerator's value expression."""
    values = {}
    for enum in enums or ():
        next_value = 0
        for name, value in getattr(enum, "members", None) or ():
            if value is not None:
                match = _INTEGER.fullmatch(str(render(value)).strip().strip("()"))
                if match is None:
                    # Later enumerators count on from a value that is no literal; leave them unresolved.
                    next_value = None
                    continue
                next_value = int(match.group(1), 0)
            if next_value is None:
                continue
            values.setdefault(name, next_value)
            next_value += 1
    return values
