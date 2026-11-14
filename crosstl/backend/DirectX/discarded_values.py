"""HLSL expression statements whose value is discarded with a cast to ``void``.

``(void)e;`` evaluates ``e`` and discards its value (C++14 section 5.4/5.2.9 [expr.static.cast]/6, which HLSL
follows), so ``(void)bump();`` still runs ``bump``. CrossGL carries the evaluation as the statement ``e;``. When
``e`` is proven free of side effects (names, literals, member, element and swizzle reads, and operators over them)
the statement is dropped, because it only silences an unused-value warning and some targets (GLSL, WGSL) reject a
bare value statement.

A statement whose root is a conditional, an element or member access or a cast (``c ? f() : g();``,
``data[i++];``) is an expression statement too (C++14 section 6.2 [stmt.expr]): it evaluates the expression and
discards its value. :data:`VALUE_STATEMENT_TYPES` lists those roots, which CrossGL carries as ``e;``.
"""

from __future__ import annotations

from ..common_ast import (
    ArrayAccessNode,
    BinaryOpNode,
    CastNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)

__all__ = ["VALUE_STATEMENT_TYPES", "discarded_cast_operand"]

VALUE_STATEMENT_TYPES = (TernaryOpNode, ArrayAccessNode, MemberAccessNode, CastNode)

_STEP_OPERATORS = frozenset({"++", "--"})


def _is_void_cast(node) -> bool:
    if not isinstance(node, CastNode):
        return False
    target = node.target_type
    name = getattr(target, "name", target)
    return isinstance(name, str) and name.strip() == "void"


def _is_side_effect_free(expression) -> bool:
    """Whether evaluating ``expression`` provably writes nothing and calls nothing."""
    if expression is None or isinstance(expression, (str, int, float, bool)):
        return True
    if isinstance(expression, VariableNode):
        return expression.value is None and not getattr(expression, "vtype", None)
    if isinstance(expression, MemberAccessNode):
        return _is_side_effect_free(expression.object)
    if isinstance(expression, ArrayAccessNode):
        return _is_side_effect_free(expression.array) and _is_side_effect_free(
            expression.index
        )
    if isinstance(expression, BinaryOpNode):
        return _is_side_effect_free(expression.left) and _is_side_effect_free(
            expression.right
        )
    if isinstance(expression, UnaryOpNode):
        return expression.op not in _STEP_OPERATORS and _is_side_effect_free(
            expression.operand
        )
    if isinstance(expression, TernaryOpNode):
        return all(
            _is_side_effect_free(part)
            for part in (
                expression.condition,
                expression.true_expr,
                expression.false_expr,
            )
        )
    if isinstance(expression, CastNode):
        return _is_side_effect_free(expression.expression)
    return False


def discarded_cast_operand(statement):
    """The statement ``(void)e;`` stands for: ``e``, or ``None`` when ``e`` has no side effects.

    Any other statement is returned unchanged.
    """
    if not _is_void_cast(statement):
        return statement
    expression = statement.expression
    while _is_void_cast(expression):
        expression = expression.expression
    return None if _is_side_effect_free(expression) else expression
