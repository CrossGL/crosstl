"""Emit CrossGL assignments that are used as values.

CrossGL assignments are expressions: ``(e = 4) + 1`` assigns ``e`` and
yields the assigned value.  In every C-family target the assignment
operators bind more loosely than any other operator except the comma, so
an assignment that appears as an operand must be parenthesized.  Without
the parentheses ``(e = 4) + 1`` is emitted as ``e = 4 + 1``, which assigns
five instead of four.
"""

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
)


def is_fully_parenthesized(code: str) -> bool:
    """Return whether one pair of parentheses encloses all of ``code``."""

    text = code.strip()
    if len(text) < 2 or text[0] != "(" or text[-1] != ")":
        return False
    depth = 0
    quote = None
    for index, char in enumerate(text):
        if quote is not None:
            if char == quote and text[index - 1] != "\\":
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
    return depth == 0


def assignment_value_expression(code: str, *, discarded: bool = False) -> str:
    """Parenthesize an emitted assignment unless its value is discarded."""

    if discarded or is_fully_parenthesized(code):
        return code
    return f"({code})"


_PURE_UNARY_OPERATORS = frozenset({"-", "+", "!", "~"})


def is_reevaluable_place(node) -> bool:
    """Return whether reading ``node`` again has no side effects.

    Targets without an assignment expression, such as Rust, store first and
    then read the target back.  That is only faithful when evaluating the
    target twice is indistinguishable from evaluating it once, so the target
    may name variables, members and elements whose indices are themselves
    free of calls, assignments, increments and decrements.
    """

    if isinstance(node, str):
        return bool(node)
    if isinstance(node, (IdentifierNode, LiteralNode)):
        return True
    if isinstance(node, VariableNode):
        return getattr(node, "initial_value", getattr(node, "value", None)) is None
    if isinstance(node, MemberAccessNode):
        return is_reevaluable_place(node.object)
    if isinstance(node, ArrayAccessNode):
        return is_reevaluable_place(node.array) and is_reevaluable_place(node.index)
    if isinstance(node, BinaryOpNode):
        return is_reevaluable_place(node.left) and is_reevaluable_place(node.right)
    if isinstance(node, UnaryOpNode):
        return node.op in _PURE_UNARY_OPERATORS and is_reevaluable_place(node.operand)
    return False
