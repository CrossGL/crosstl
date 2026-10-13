"""Checked finite induction for collective control-flow proofs."""

from ..arithmetic_conversions import ArithmeticScalarKind, source_integer_shape
from ..ast import (
    BinaryOpNode,
    BlockNode,
    ForNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    UnaryOpNode,
    VariableNode,
)
from ..resource_identity import source_resource_identity
from .array_utils import _UnsignedLiteralInt, evaluate_literal_int_expression


def checked_induction_value(expression, constants):
    """Evaluate integer proof operands without hiding overflowing subexpressions."""
    if isinstance(expression, (LiteralNode, IdentifierNode)):
        pass
    elif isinstance(expression, UnaryOpNode):
        if expression.operator not in {"+", "-"}:
            return None
        if checked_induction_value(expression.operand, constants) is None:
            return None
    elif isinstance(expression, BinaryOpNode):
        if expression.operator not in {"+", "-", "*", "/", "%"}:
            return None
        left = checked_induction_value(expression.left, constants)
        right = checked_induction_value(expression.right, constants)
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


def loop_constant_bindings(function, constants, mutable_names, type_name):
    """Snapshot immutable local integer constants at each lexical for loop.

    The caller supplies names invalidated by writes, references, address escapes
    and resolved call effects. Other control-flow forms gain no new proof.
    """
    result = {}

    def visit(body, incoming):
        visible = dict(incoming)
        statements = body.statements if isinstance(body, BlockNode) else body
        if statements is None:
            return
        if not isinstance(statements, (list, tuple)):
            statements = [statements]
        for statement in statements:
            if isinstance(statement, VariableNode):
                visible.pop(statement.name, None)
                qualifiers = set(statement.qualifiers or []) | set(
                    getattr(statement.var_type, "qualifiers", []) or []
                )
                if (
                    statement.name in mutable_names
                    or not qualifiers & {"const", "constant"}
                    or "volatile" in qualifiers
                ):
                    continue
                shape = source_integer_shape(type_name(statement.var_type))
                if shape is None or shape[2] != 1 or shape[1] > 32:
                    continue
                value = checked_induction_value(statement.initial_value, visible)
                unsigned = shape[0] == ArithmeticScalarKind.UNSIGNED_INTEGER
                low = 0 if unsigned else -(1 << (shape[1] - 1))
                high = (1 << (shape[1] - (not unsigned))) - 1
                if value is not None and low <= value <= high:
                    visible[statement.name] = (
                        _UnsignedLiteralInt(value) if unsigned else int(value)
                    )
            elif isinstance(statement, BlockNode):
                visit(statement, visible)
            elif isinstance(statement, IfNode):
                for branch in [
                    statement.then_branch,
                    *statement.else_if_bodies,
                    statement.else_branch,
                ]:
                    visit(branch, visible)
            elif isinstance(statement, ForNode):
                result[id(statement)] = dict(visible)
                nested = dict(visible)
                if isinstance(statement.init, VariableNode):
                    nested.pop(statement.init.name, None)
                visit(statement.body, nested)

    visit(
        function.body,
        {name: value for name, value in constants.items() if name not in mutable_names},
    )
    return result


def metadata_loop_bindings(function, records, mutable_names, walk_ast):
    """Keep only immutable, unshadowed source-owned scalar parameter ranges."""
    return {
        name: record
        for name, record in metadata_parameter_bindings(
            function, records, mutable_names, walk_ast
        )
        if "element" not in record
    }


def metadata_parameter_bindings(function, records, mutable_names, walk_ast):
    """Resolve immutable metadata without conflating buffer contents and offsets."""
    invalid = set(mutable_names) | {
        node.name for node in walk_ast(function.body) if isinstance(node, VariableNode)
    }
    for parameter in function.parameters:
        if parameter.name in invalid:
            continue
        reference = parameter.param_type
        if any(
            "volatile" in (getattr(node, "qualifiers", ()) or ())
            for node in (
                parameter,
                reference,
                getattr(reference, "referenced_type", None),
            )
        ):
            continue
        origin = source_resource_identity(parameter)
        if origin is None:
            continue
        for record in records:
            if record["sourceResource"] == {"schemaVersion": 1, **origin}:
                yield parameter.name, record


def fixed_stride_loop_terminates(
    type_name, start, bound, update, constants, operator, *, metadata_bounds=None
):
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

    start = checked_induction_value(start, constants)
    stop = checked_induction_value(bound, constants)
    amount = 1 if unary else checked_induction_value(update.value, constants)
    if start is None or amount is None:
        return False
    stops = [stop]
    if stop is None:
        record = (metadata_bounds or {}).get(
            bound.name if isinstance(bound, IdentifierNode) else None
        )
        if record is None:
            return False
        convert = _UnsignedLiteralInt if record["dtype"].startswith("u") else int
        stops = [convert(record["minimum"]), convert(record["maximum"])]
    if not all(minimum <= value <= maximum for value in (start, amount, *stops)):
        return False
    # Mixed unsigned arithmetic can change comparisons before assignment.
    if unsigned or any(
        isinstance(value, _UnsignedLiteralInt) for value in (start, amount, *stops)
    ):
        if min(start, amount, *stops) < 0:
            return False
    step = int(amount) * (1 if update.operator in {"+=", "++"} else -1)
    if step == 0 or (step > 0) != (operator in {"<", "<="}):
        return False
    # For a fixed start and stride, the final value is monotone in the bound.
    # Checking both endpoints includes zero-trip paths and the terminating update.
    for stop in stops:
        distance = int(stop) - int(start) if step > 0 else int(start) - int(stop)
        distance += operator in {"<=", ">="}
        trips = max(0, (distance + abs(step) - 1) // abs(step))
        final = int(start) + trips * step
        if not minimum <= final <= maximum:
            return False
    return True


def geometric_loop_terminates(type_name, start, bound, update, constants, operator):
    """Prove positive geometric induction, including the terminating update."""
    if not isinstance(type_name, str) or not type_name.isidentifier():
        return False
    shape = source_integer_shape(
        {"min16int": "short", "min16uint": "ushort"}.get(type_name, type_name)
    )
    if (
        shape is None
        or shape[2] != 1
        or shape[1] > 32
        or getattr(update, "operator", None) not in {"*=", "<<="}
        or operator not in {"<", "<="}
    ):
        return False
    unsigned = shape[0] == ArithmeticScalarKind.UNSIGNED_INTEGER
    maximum = (1 << (shape[1] - (not unsigned))) - 1
    start = checked_induction_value(start, constants)
    stop = checked_induction_value(bound, constants)
    amount = checked_induction_value(update.value, constants)
    if any(value is None for value in (start, stop, amount)):
        return False
    if not (0 < start <= maximum and 0 <= stop <= maximum):
        return False
    if update.operator == "<<=":
        if not 0 < amount < shape[1]:
            return False
        factor = 1 << int(amount)
    else:
        if not 1 < amount <= maximum:
            return False
        factor = int(amount)
    value = int(start)
    # Growth is at least twofold, so at most the source bit width is visited.
    while value < stop or operator == "<=" and value == stop:
        value *= factor
        if value > maximum:
            return False
    return True
