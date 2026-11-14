"""Generic value parameters in the body of a specialized generic function.

A CrossGL generic parameter can stand for a value as well as a type: ``generic<T, N> T scale(T x) { return x * T(N);
}`` called as ``scale<float, 3>(x)`` binds ``T`` to ``float`` and ``N`` to ``3``, as a C++ or Metal non-type template
parameter does. Every monomorphizing target clones the function for the call
(:func:`.generic_function_utils._ensure_generic_function_specialization`), and
:func:`.generic_body_types.substitute_body_types` gives the clone its concrete types. The value parameters stayed
identifiers, so the specialization read the undeclared ``N``.

:func:`substitute_value_parameters` replaces each use of a value parameter in the clone's body, including array sizes
of local declarations, by the expression the call bound it to. A parameter bound to a type and used as a value is
refused rather than emitted.
"""

from __future__ import annotations

from copy import deepcopy

from ..ast import (
    ArrayType,
    ASTNode,
    ExpressionNode,
    FunctionCallNode,
    IdentifierNode,
    ParameterNode,
    VariableNode,
)
from ..refusals import TargetRefusal
from ..semantics.rewrite import child_slots, get_child, set_child

__all__ = ["substitute_value_parameters", "value_argument_expression"]

# Fields of a call whose identifier names the callee rather than reading a value.
_CALLEE_FIELDS = frozenset({"function", "func", "callee"})
# Calls whose first argument names a type, as in ``sizeof(T)``; the same set as :mod:`..semantics.type_aliases`.
_TYPE_OPERAND_CALLS = frozenset({"sizeof", "alignof", "metal_sizeof"})


def _type_operand_ids(body):
    """The ids of the nodes in the type operands of ``sizeof(T)``-like calls, such as ``T`` in ``sizeof(T[3])``."""
    ids = set()
    for node in _nodes(body):
        callee = getattr(node, "function", None)
        if not (
            isinstance(node, FunctionCallNode)
            and isinstance(callee, IdentifierNode)
            and callee.name in _TYPE_OPERAND_CALLS
        ):
            continue
        arguments = getattr(node, "arguments", getattr(node, "args", None)) or []
        if arguments:
            ids.update(id(item) for item in _nodes(arguments[0]))
    return ids


def value_argument_expression(text):
    """The CrossGL expression a value generic argument spells, else ``None`` when ``text`` spells a type."""
    from ..lexer import Lexer
    from ..parser import Parser

    try:
        parser = Parser(Lexer(str(text)).get_tokens())
        first = parser.current_token
        if first[0] != "IDENTIFIER" and parser.current_token_starts_type():
            # ``float``, ``vec3`` and ``vec3<float>`` are type arguments.
            return None
        expression = parser.parse_expression()
    except (SyntaxError, ValueError, IndexError):
        return None
    if parser.current_token[0] != "EOF" or not isinstance(expression, ExpressionNode):
        return None
    return expression


def _declared_names(body):
    """Names the body declares; a use of one reads the declaration, not a generic parameter."""
    names = set()
    for node in _nodes(body):
        if isinstance(node, (VariableNode, ParameterNode)):
            name = getattr(node, "name", None)
            if isinstance(name, str):
                names.add(name)
    return names


def _nodes(node):
    stack, seen = [node], set()
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, ASTNode) and id(item) not in seen:
            seen.add(id(item))
            yield item
            stack.extend(
                get_child(item, field, key) for field, key in child_slots(item)
            )


def _value_uses(body, names):
    """``(parent, field, key, name, type_operand)`` for each identifier of ``names`` read in ``body``.

    ``type_operand`` marks the operand of ``sizeof(T)``, which may name a type parameter as well as a value.
    """
    uses = []
    type_operands = _type_operand_ids(body)
    for node in _nodes(body):
        for field, key in child_slots(node):
            child = get_child(node, field, key)
            if not isinstance(child, IdentifierNode) or child.name not in names:
                continue
            if isinstance(node, FunctionCallNode) and field in _CALLEE_FIELDS:
                continue
            uses.append((node, field, key, child.name, id(child) in type_operands))
    return uses


def _array_size_slots(body, names):
    """Local array declarations whose size is a value parameter: ``T values[N]``."""
    slots = []
    for node in _nodes(body):
        for field in ("var_type", "vtype"):
            value = getattr(node, field, None)
            while isinstance(value, ArrayType):
                size = value.size
                if isinstance(size, IdentifierNode) and size.name in names:
                    slots.append((value, size.name))
                elif isinstance(size, str) and size.strip() in names:
                    slots.append((value, size.strip()))
                value = value.element_type
    return slots


def substitute_value_parameters(body, substitutions, function_name="") -> int:
    """Replace the value parameters read in ``body`` by their bound expressions; returns how many uses changed.

    ``substitutions`` maps each generic parameter to its argument's CrossGL spelling. Raises ``ValueError`` when a
    parameter read as a value is bound to a type.
    """
    if body is None or not substitutions:
        return 0
    candidates = set(substitutions) - _declared_names(body)
    if not candidates:
        return 0
    uses = _value_uses(body, candidates)
    sizes = _array_size_slots(body, candidates)
    expressions = {
        name: value_argument_expression(substitutions[name])
        for name in sorted({use[3] for use in uses} | {slot[1] for slot in sizes})
    }
    # A type parameter is left to the type passes where it is a type operand; anywhere else it is no value.
    uses = [use for use in uses if expressions[use[3]] is not None or not use[4]]
    for name in sorted({use[3] for use in uses} | {slot[1] for slot in sizes}):
        if expressions[name] is None:
            label = f" of generic function '{function_name}'" if function_name else ""
            raise TargetRefusal(
                f"Generic parameter '{name}'{label} is read as a value but bound to the type "
                f"'{substitutions[name]}'"
            )
    for parent, field, key, name, _type_operand in uses:
        set_child(parent, field, key, deepcopy(expressions[name]))
    for array_type, name in sizes:
        array_type.size = deepcopy(expressions[name])
    return len(uses) + len(sizes)
