"""Whether a Metal function that returns a ``device`` or ``constant`` pointer returns whole buffers (#2194).

The importer types a function returning ``device T*`` as a CrossGL ``RWStructuredBuffer<T>`` (``StructuredBuffer<T>``
for a ``const`` or ``constant`` pointee), a buffer handle. A handle names a whole buffer and carries no offset, so
that type is right only when every value the function returns is a pointer parameter it receives and never moves,
as in a selector that returns one of several buffers. ``const device int* advance(const device int* v) { v += 1;
return v; }`` returns an interior pointer; typed as a handle, ``advance(values)[i]`` read ``values[i]`` instead of
``values[i + 1]`` on targets that pass buffers as handles, and Metal output could not write the handle type back.
Such a function keeps its Metal pointer type.
"""

from __future__ import annotations

from ..common_ast import (
    AssignmentNode,
    PostfixOpNode,
    ReturnNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)

__all__ = ["returns_whole_buffer_parameters"]

_MOVING_OPERATORS = frozenset({"++", "--"})


def _parameter_names(function):
    return {
        parameter.name
        for parameter in getattr(function, "params", None) or ()
        if isinstance(getattr(parameter, "name", None), str)
    }


def _walk(node, children):
    stack = [node]
    while stack:
        current = stack.pop()
        if current is None or isinstance(current, (str, int, float, bool)):
            continue
        yield current
        stack.extend(children(current))


def _returned_names(value, names):
    """Add the parameter names ``value`` returns to ``names``; ``False`` if it returns anything else."""
    while isinstance(value, TernaryOpNode):
        if not _returned_names(getattr(value, "true_expr", None), names):
            return False
        value = getattr(value, "false_expr", None)
    if isinstance(value, VariableNode) and isinstance(value.name, str):
        names.add(value.name)
        return True
    return False


def _moved_names(node):
    if isinstance(node, AssignmentNode):
        target = node.left
    elif (
        isinstance(node, (UnaryOpNode, PostfixOpNode)) and node.op in _MOVING_OPERATORS
    ):
        target = node.operand
    else:
        return None
    if isinstance(target, VariableNode) and isinstance(target.name, str):
        return target.name
    return None


def returns_whole_buffer_parameters(function, children):
    """True when every ``return`` of ``function`` yields a parameter that the function never reassigns or steps.

    ``children(node)`` lists the child nodes of an AST node. A function without a body, without a ``return`` value
    or with any other returned value answers False.
    """
    body = getattr(function, "body", None)
    if body is None:
        return False
    parameters = _parameter_names(function)
    returned = set()
    moved = set()
    saw_return = False
    for node in _walk(body, children):
        if isinstance(node, ReturnNode):
            saw_return = True
            if not _returned_names(getattr(node, "value", None), returned):
                return False
            continue
        name = _moved_names(node)
        if name is not None:
            moved.add(name)
    return saw_return and returned <= parameters and not returned & moved
