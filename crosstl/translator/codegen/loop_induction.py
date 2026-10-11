"""Checked finite induction for collective control-flow proofs."""

from ..arithmetic_conversions import ArithmeticScalarKind, source_integer_shape
from ..ast import BinaryOpNode, IdentifierNode, LiteralNode, UnaryOpNode
from .array_utils import _UnsignedLiteralInt, evaluate_literal_int_expression


def fixed_stride_loop_terminates(type_name, start, bound, update, constants, operator):
    """Prove the terminating update as well as every visited induction value."""
    if not isinstance(type_name, str) or not type_name.isidentifier():
        return False
    shape = source_integer_shape(
        {"min16int": "short", "min16uint": "ushort"}.get(type_name, type_name)
    )
    if shape is None or shape[2] != 1 or shape[1] > 32:
        return False
    unary = isinstance(update, UnaryOpNode)
    if update.operator not in (
        {"++", "--"} if unary else {"+=", "-="}
    ) or operator not in {"<", "<=", ">", ">="}:
        return False
    unsigned = shape[0] == ArithmeticScalarKind.UNSIGNED_INTEGER
    minimum = 0 if unsigned else -(1 << (shape[1] - 1))
    maximum = (1 << (shape[1] - (not unsigned))) - 1

    def checked_value(expression):
        if isinstance(expression, (LiteralNode, IdentifierNode)):
            pass
        elif isinstance(expression, UnaryOpNode):
            if expression.operator not in {"+", "-"}:
                return None
            if checked_value(expression.operand) is None:
                return None
        elif isinstance(expression, BinaryOpNode):
            if expression.operator not in {"+", "-", "*", "/", "%"}:
                return None
            left = checked_value(expression.left)
            right = checked_value(expression.right)
            if left is None or right is None:
                return None
            if expression.operator in {"/", "%"} and left == -(1 << 31) and right == -1:
                return None
        else:
            return None
        value = evaluate_literal_int_expression(expression, constants)
        if value is None:
            return None
        low, high = (
            (0, (1 << 32) - 1)
            if isinstance(value, _UnsignedLiteralInt)
            else (-(1 << 31), (1 << 31) - 1)
        )
        return value if low <= value <= high else None

    start = checked_value(start)
    stop = checked_value(bound)
    amount = 1 if unary else checked_value(update.value)
    if any(value is None for value in (start, stop, amount)):
        return False
    if not all(minimum <= value <= maximum for value in (start, stop, amount)):
        return False
    # Mixed unsigned arithmetic can change comparisons before assignment.
    if any(isinstance(value, _UnsignedLiteralInt) for value in (start, stop, amount)):
        if min(start, stop, amount) < 0:
            return False
    step = int(amount) * (1 if update.operator in {"+=", "++"} else -1)
    if step == 0 or (step > 0) != (operator in {"<", "<="}):
        return False
    distance = int(stop) - int(start) if step > 0 else int(start) - int(stop)
    distance += operator in {"<=", ">="}
    trips = max(0, (distance + abs(step) - 1) // abs(step))
    final = int(start) + trips * step
    return minimum <= final <= maximum
