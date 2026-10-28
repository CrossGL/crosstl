"""Comma expressions, ``(a, b)``, in code generation.

The CrossGL comma operator evaluates its operands from left to right, discards every value but the
last, and yields the last operand's value. It is never an l-value. The C-family targets share the
operator and its sequencing: C11 6.5.17 (OpenCL C, CUDA and HIP device code), C++ [expr.comma] (the
Metal Shading Language is C++14 based), GLSL 4.60 and GLSL ES 3.00 section 5.9 ("sequence"
operator), HLSL and Slang, which follow C. Those emitters print it with explicit parentheses, because
the comma binds more loosely than assignment and would otherwise split an initializer, an argument
list or a ternary arm.

Targets without the operator lower it instead. SPIR-V evaluates the operands in order into the
current block, so an operand's side effects run before the next operand is read, and returns the
last operand's id. A target that can do neither refuses the construct with its typed diagnostic.
"""

from __future__ import annotations

import copy

from ..ast import (
    AssignmentNode,
    CommaExpressionNode,
    ExpressionStatementNode,
    FunctionCallNode,
    UnaryOpNode,
)
from .lowering_refusals import refuse_unsupported

__all__ = [
    "CommaExpressionVisitorMixin",
    "c_family_comma_expression",
    "comma_result_operand",
    "discarded_comma_operands_call",
    "is_comma_expression",
    "refuse_comma_expression",
    "spirv_comma_expression",
    "split_comma_initializer",
]

# Qualifiers of a declaration whose initializer does not run where the declaration stands, or runs once.
_DEFERRED_INITIALIZER_QUALIFIERS = frozenset(
    {"static", "groupshared", "shared", "threadgroup", "workgroup", "buffer"}
)


def is_comma_expression(expr) -> bool:
    return isinstance(expr, CommaExpressionNode)


def _operands(expr):
    operands = list(expr.expressions or ())
    if len(operands) < 2:
        raise ValueError(
            "CrossGL comma expression needs at least two operands, got "
            f"{len(operands)}"
        )
    return operands


def comma_result_operand(expr):
    """The operand whose value and type a comma expression yields: its last one."""
    return _operands(expr)[-1]


def _fully_parenthesized(text) -> bool:
    """True when ``text`` is one parenthesized group, ``(...)``, and not ``(a) + (b)``."""
    if not (text.startswith("(") and text.endswith(")")):
        return False
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
    return depth == 0


def c_family_comma_expression(expr, generate, discard=None, discarded=False):
    """``(a, b, ...)`` with each operand rendered by ``generate``; None when ``expr`` is no comma.

    ``generate`` is the emitter's expression renderer. The operands are rendered in source order, so
    an emitter that records state while rendering (temporaries, helper requests) sees them in the
    order the target evaluates them. ``discard``, when given, is the emitter's renderer for a value
    that is unused (an expression statement's): it renders every operand but the last, and the last
    one too when the comma expression is itself ``discarded``. A postfix update the emitter lowers to
    an update followed by the old value then does not leave that value unused.
    """
    if not isinstance(expr, CommaExpressionNode):
        return None
    operands = _operands(expr)

    def render_discarded(operand):
        if discard is None:
            return generate(operand)
        # A statement renderer may end the update with ``;`` and leaves an assignment bare.
        rendered = discard(operand).strip().rstrip(";").rstrip()
        return rendered if _fully_parenthesized(rendered) else f"({rendered})"

    rendered = [render_discarded(operand) for operand in operands[:-1]]
    rendered.append((render_discarded if discarded else generate)(operands[-1]))
    return f"({', '.join(rendered)})"


def discarded_comma_operands_call(expr) -> bool:
    """True when an operand whose value a comma discards contains a function call.

    A call may write memory, so a lowering that drops a discarded operand must not drop one with a
    call in it.
    """
    for operand in _operands(expr)[:-1]:
        nodes = operand.walk() if hasattr(operand, "walk") else ()
        if any(isinstance(node, FunctionCallNode) for node in nodes):
            return True
    return False


def _has_effect(operand) -> bool:
    nodes = operand.walk() if hasattr(operand, "walk") else ()
    for node in nodes:
        if isinstance(node, (AssignmentNode, FunctionCallNode)):
            return True
        if isinstance(node, UnaryOpNode) and node.operator in ("++", "--"):
            return True
    return False


def split_comma_initializer(stmt):
    """``T x = (a, b);`` as the statements ``a;`` and ``T x = b;``; None for any other declaration.

    The initializer of a local declaration runs where the declaration stands, before anything else
    in it, so running the discarded operands as statements first is the same program. A discarded
    operand without an assignment, an increment, a decrement or a call has no effect and is dropped.
    DXC warns on the unsplit form (``-Wcomma-in-init``: "comma expression used where a constructor
    list may have been intended"). Declarations whose initializer runs once or elsewhere (``static``,
    ``groupshared``) are left alone.
    """
    value = getattr(stmt, "initial_value", None)
    if not isinstance(value, CommaExpressionNode):
        return None
    qualifiers = {str(item).lower() for item in getattr(stmt, "qualifiers", None) or ()}
    if qualifiers & _DEFERRED_INITIALIZER_QUALIFIERS:
        return None
    *discarded, result = _operands(value)
    declaration = copy.copy(stmt)
    declaration.initial_value = result
    effects = [
        ExpressionStatementNode(operand)
        for operand in discarded
        if _has_effect(operand)
    ]
    return effects, declaration


def spirv_comma_expression(codegen, expr):
    """Evaluate ``expr``'s operands in order into the open block and return the last one's id.

    Each discarded operand is still evaluated for its side effects; its result id is unused, as the
    value of an expression statement is.
    """
    operands = _operands(expr)
    for operand in operands[:-1]:
        codegen.process_expression(operand)
    return codegen.process_expression(operands[-1])


def refuse_comma_expression(target, expr, placeholder="0"):
    """Refuse ``expr`` for ``target``, which has no comma operator and no lowering for it here."""
    count = len(_operands(expr))
    return refuse_unsupported(
        target,
        "comma expression",
        f"{target} has no comma operator, and this emitter has no sequencing lowering for "
        f"a comma expression of {count} operands",
        placeholder,
    )


class CommaExpressionVisitorMixin:
    """``visit_CommaExpressionNode`` for an emitter that dispatches on ``visit_<node type>``."""

    def visit_CommaExpressionNode(self, node):
        return c_family_comma_expression(node, self.generate_expression)
