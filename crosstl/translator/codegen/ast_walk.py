"""Iterative pre-order walk over every object reachable from an AST value.

The code generators used to walk with nested ``yield from`` generators. Each
yielded node then passed through every enclosing generator frame, so a walk cost
the node count times the nesting depth, and a deep expression could exceed the
recursion limit. This walk keeps an explicit stack of iterators instead and
yields exactly the same objects in exactly the same order:

- ``None``, ``str``, ``int``, ``float`` and ``bool`` values are skipped;
- ``dict`` values, and the items of a ``list``, ``tuple`` or ``set``, are
  visited in iteration order, and the containers themselves are not yielded;
- any other object is yielded once (by identity), then its ``vars()`` values are
  visited in attribute order.

Child iterators are created only when the walk resumes after yielding their
owner, as before, so a caller may still change the attributes of the node it has
just received.
"""

from typing import Any, Iterator

_SCALARS = (str, int, float, bool)
_SEQUENCES = (list, tuple, set)

# How the walk treats a value, decided once per concrete type: the walk runs
# millions of times per large translation unit, and one dictionary lookup on
# ``type(value)`` replaces a chain of ``isinstance`` and ``hasattr`` checks.
_SKIP = 0
_MAPPING = 1
_SEQUENCE = 2
_OBJECT = 3
_OPAQUE = 4
_KINDS = {type(None): _SKIP}


def _value_kind(value: Any) -> int:
    value_type = type(value)
    if isinstance(value, _SCALARS):
        kind = _SKIP
    elif isinstance(value, dict):
        kind = _MAPPING
    elif isinstance(value, _SEQUENCES):
        kind = _SEQUENCE
    else:
        # A type provides ``__dict__`` for all of its instances or for none, so
        # the answer for this instance holds for every value of its type.
        kind = _OBJECT if hasattr(value, "__dict__") else _OPAQUE
    _KINDS[value_type] = kind
    return kind


def walk_ast_values(root: Any) -> Iterator[Any]:
    """Yield every non-container object reachable from ``root`` once, in pre-order."""
    visited = set()
    stack = [iter((root,))]
    kinds = _KINDS
    while stack:
        for value in stack[-1]:
            kind = kinds.get(type(value))
            if kind is None:
                kind = _value_kind(value)
            if kind == _SKIP:
                continue
            if kind == _MAPPING:
                stack.append(iter(value.values()))
                break
            if kind == _SEQUENCE:
                stack.append(iter(value))
                break
            value_id = id(value)
            if value_id in visited:
                continue
            visited.add(value_id)
            yield value
            if kind == _OBJECT:
                stack.append(iter(vars(value).values()))
                break
        else:
            stack.pop()
