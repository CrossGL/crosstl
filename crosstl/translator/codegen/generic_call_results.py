"""Types of expressions that read the result of a generic call, for generic-call deduction.

In ``twice(twice(v).y)`` the outer call deduces ``T`` from ``twice(v).y``. Generators type a call by its
definition's declared return type, which for a generic definition is the open parameter ``T``. Collection then
bound the outer ``T`` to ``T`` itself and created the specialization ``T twice_T(T x)``, which Metal and HLSL
emitted verbatim and OpenGL refused.

A generic call is typed here by the return type of the specialization its own arguments select. An expression
that reads such a call (a member, element, operator or ternary branch) is typed from that result, and a type
spelled with an open parameter of the generic it reads is discarded, so the call stays unresolved and is
refused instead of specialized for a parameter.
"""

from __future__ import annotations

import re

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
)

__all__ = [
    "generic_call_argument_type_name",
    "generic_call_result_type_name",
    "generic_calls_read_by",
]

_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def _call_name(call):
    callee = getattr(call, "function", getattr(call, "name", None))
    name = getattr(callee, "name", callee)
    return name if isinstance(name, str) else None


def _call_arguments(call):
    return list(getattr(call, "arguments", getattr(call, "args", [])) or [])


def _generic_definition_name(generator, call):
    from .generic_function_utils import generic_function_definition_name

    name = _call_name(call)
    if not name:
        return None
    definitions = getattr(generator, "generic_function_definitions", {}) or {}
    definition = generic_function_definition_name(generator, name)
    return definition if definition in definitions else None


def generic_calls_read_by(generator, expr):
    """The generic definitions whose call results decide the type of ``expr``, in source order."""
    found = []
    pending = [expr]
    while pending:
        node = pending.pop()
        if isinstance(node, FunctionCallNode):
            definition = _generic_definition_name(generator, node)
            if definition is not None:
                found.append(definition)
            else:
                pending.extend(reversed(_call_arguments(node)))
        elif isinstance(node, MemberAccessNode):
            pending.append(getattr(node, "object", getattr(node, "object_expr", None)))
        elif isinstance(node, ArrayAccessNode):
            pending.append(getattr(node, "array", None))
        elif isinstance(node, BinaryOpNode):
            pending.extend([getattr(node, "right", None), getattr(node, "left", None)])
        elif isinstance(node, UnaryOpNode):
            pending.append(getattr(node, "operand", None))
        elif isinstance(node, TernaryOpNode):
            pending.extend(
                [getattr(node, "false_expr", None), getattr(node, "true_expr", None)]
            )
    return found


def generic_call_result_type_name(generator, call, call_key):
    """The return type of the specialization ``call`` selects, or ``None`` when its arguments do not select one.

    ``call_key`` is the deduction that maps a call to its specialization key; it creates the specialization while
    specializations are collected.
    """
    definition = _generic_definition_name(generator, call)
    if definition is None:
        return None
    key = call_key(generator, _call_name(call), _call_arguments(call))
    if key is None:
        return None
    specialization = (
        getattr(generator, "generic_function_specializations", {}) or {}
    ).get(key)
    if specialization is None:
        return None
    return generator.type_name_string(getattr(specialization, "return_type", None))


def _open_parameters(generator, definitions):
    from .generic_function_utils import generic_function_parameters

    candidates = getattr(generator, "generic_function_definitions", {}) or {}
    names = set()
    for definition in definitions:
        for func in candidates.get(definition) or []:
            names.update(str(param) for param in generic_function_parameters(func))
    return names


def generic_call_argument_type_name(generator, definitions, fallback_type, actual_type):
    """The type of an argument that reads the results of the generic ``definitions``.

    ``fallback_type`` is typed from the results of the specializations the calls select; ``actual_type`` is the
    generator's own typing, which reads the definitions' declared return types. A type spelled with one of the
    definitions' open parameters is not a type of the argument.
    """
    open_parameters = _open_parameters(generator, definitions)
    for candidate in (actual_type, fallback_type):
        if not candidate:
            continue
        if open_parameters & set(_IDENTIFIER.findall(str(candidate))):
            continue
        return candidate
    return None
