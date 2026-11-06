"""Pre-order walk over every object reachable through the attributes of an AST value.

Code generators ask which nodes a function body, an expression or a whole shader contains many times per program.
A recursive generator that delegates with ``yield from`` resumes one frame per nesting level for every node it
produces, so a walk over a deeply nested body costs its depth for each node. This walk keeps an explicit stack of
iterators instead, so each node costs a constant amount of work, and it produces the same nodes in the same order
at the same moments as the recursive form.

Order and timing:

* A node is produced before anything it contains. Its attributes are visited in ``vars`` order and the items of a
  list, tuple or set in iteration order.
* The attributes of a node are read only when the consumer asks for the item after that node, and lists are
  iterated live, so a consumer that edits a node it has just received sees the edit reflected in the walk exactly
  as before.
* ``None``, strings, numbers and Booleans are skipped, as are objects without a ``__dict__`` such as dictionaries.
  The ``parent`` and ``annotations`` attributes are never followed.
* A value may be reached along several paths and is then produced once per path. A value that contains itself has
  no finite walk; the recursive form ran out of frames on it, and this walk raises ``RecursionError`` as soon as a
  value recurs inside itself. Nesting deeper than the interpreter's recursion limit is walked normally.
"""

from __future__ import annotations

from typing import Any, Iterator

__all__ = ["iter_ast_values"]

_SCALAR_TYPES = (str, int, float, bool)
_SEQUENCE_TYPES = (list, tuple, set)
_SKIPPED_ATTRIBUTES = frozenset({"parent", "annotations"})


def _attribute_values(node: Any) -> Iterator[Any]:
    for key, value in vars(node).items():
        if key not in _SKIPPED_ATTRIBUTES:
            yield value


def iter_ast_values(root: Any) -> Iterator[Any]:
    """Every object with a ``__dict__`` reachable from ``root``, in pre-order."""
    stack = [iter((root,))]
    # Identities of the values whose iterators are on the stack, aligned with ``stack[1:]``.
    path: list[int] = []
    open_values: set[int] = set()
    while stack:
        for value in stack[-1]:
            if value is None or isinstance(value, _SCALAR_TYPES):
                continue
            if isinstance(value, _SEQUENCE_TYPES):
                _enter(value, path, open_values)
                stack.append(iter(value))
                break
            if not hasattr(value, "__dict__"):
                continue
            _enter(value, path, open_values)
            yield value
            stack.append(_attribute_values(value))
            break
        else:
            stack.pop()
            if path:
                open_values.discard(path.pop())


def _enter(value: Any, path: list[int], open_values: set[int]) -> None:
    identity = id(value)
    if identity in open_values:
        raise RecursionError(
            f"AST value walk reached a {type(value).__name__} value inside itself; the value graph contains a cycle"
        )
    open_values.add(identity)
    path.append(identity)
