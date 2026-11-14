"""Recognize expressions whose only calls are read-only resource-aggregate helpers.

Resource-aggregate lowering marks the ``make``, ``shift`` and ``load`` helpers it emits, and every function
that calls only such helpers while writing nothing but private scalars, with ``resource_aggregate_nonmutating``.
An expression built from side-effect-free parts and calls of such helpers has no effect on any other value, so
an interval summary may treat it as an opaque value instead of an unmodeled effect.

This predicate is deliberately separate from ``glsl_side_effect_free_expression``: a helper call still reads
memory, so it must not be duplicated or reordered the way a pure expression may be.
"""

from __future__ import annotations

from typing import Any

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
)

_MUTATING_UNARY_OPERATORS = frozenset({"++", "--"})


def is_nonmutating_resource_helper_call(codegen: Any, expression: Any) -> bool:
    """Whether ``expression`` calls a nonmutating resource helper with read-only arguments."""
    if not isinstance(expression, FunctionCallNode):
        return False
    name = codegen.function_call_name(expression)
    candidates = [
        candidate
        for candidate in codegen.glsl_function_overloads_by_name.get(name, ())
        if getattr(candidate, "body", None) is not None
    ]
    definition = codegen.function_definitions.get(name)
    if definition is not None and definition not in candidates:
        candidates.append(definition)
    if not candidates or not all(
        getattr(candidate, "resource_aggregate_nonmutating", False)
        for candidate in candidates
    ):
        return False
    arguments = getattr(expression, "arguments", getattr(expression, "args", [])) or []
    return all(
        is_read_only_helper_expression(codegen, argument) for argument in arguments
    )


def is_read_only_helper_expression(codegen: Any, expression: Any) -> bool:
    """Whether ``expression`` writes nothing, allowing calls of nonmutating resource helpers."""
    if codegen.glsl_side_effect_free_expression(expression):
        return True
    if isinstance(expression, FunctionCallNode):
        return is_nonmutating_resource_helper_call(codegen, expression)
    if isinstance(expression, MemberAccessNode):
        return is_read_only_helper_expression(codegen, expression.object)
    if isinstance(expression, ArrayAccessNode):
        return is_read_only_helper_expression(
            codegen, expression.array
        ) and is_read_only_helper_expression(codegen, expression.index)
    if isinstance(expression, BinaryOpNode):
        return is_read_only_helper_expression(
            codegen, expression.left
        ) and is_read_only_helper_expression(codegen, expression.right)
    if isinstance(expression, UnaryOpNode):
        return codegen.map_operator(
            expression.op
        ) not in _MUTATING_UNARY_OPERATORS and is_read_only_helper_expression(
            codegen, expression.operand
        )
    if isinstance(expression, TernaryOpNode):
        return all(
            is_read_only_helper_expression(codegen, part)
            for part in (
                expression.condition,
                expression.true_expr,
                expression.false_expr,
            )
        )
    return False
