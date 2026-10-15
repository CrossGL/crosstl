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


def walk_control_scopes(body, names):
    """Yield ``(node, visible)`` for ``body`` and every node below it.

    ``visible`` holds the ``names`` that still denote the enclosing loop's
    controls at ``node``. A nested scope may hide one: a nested ``for`` that
    declares it, as MLX's ``for (int i = 0; i < qk_per_thread; i++)`` inside
    its key loop over ``i``, or a declaration in a nested block, for the rest
    of that block. Writes to the hiding variable then leave the control
    unchanged. A declaration directly in ``body``'s own statement list, the
    loop's outermost scope, hides nothing: C++ and GLSL reject it, so callers
    still see it and reject the loop. Initializers are visited with the names
    of the enclosing scope, which only over-approximates the writes they make.
    """
    names = frozenset(names)
    visited = set()

    def walk(value, visible, outermost=False):
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        if isinstance(value, dict):
            for item in value.values():
                yield from walk(item, visible)
            return
        if isinstance(value, (list, tuple)):
            yield from sequence(value, visible, outermost)
            return
        if isinstance(value, set):
            for item in value:
                yield from walk(item, visible)
            return
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, VariableNode) and not outermost and value.name in visible:
            yield value, visible - {value.name}
        else:
            yield value, visible
        if isinstance(value, BlockNode):
            for field, child in vars(value).items():
                if field == "statements":
                    yield from sequence(child or (), visible, outermost)
                else:
                    yield from walk(child, visible)
            return
        if isinstance(value, ForNode):
            initializer = value.init
            inner = visible
            if isinstance(initializer, VariableNode) and initializer.name in visible:
                inner = visible - {initializer.name}
            for field, child in vars(value).items():
                if field == "init":
                    yield from walk(child, visible)
                else:
                    yield from walk(child, inner)
            return
        if hasattr(value, "__dict__"):
            for child in vars(value).values():
                yield from walk(child, visible)

    def sequence(items, visible, outermost):
        current = visible
        for item in items:
            yield from walk(
                item, current, outermost and not isinstance(item, BlockNode)
            )
            if (
                not outermost
                and isinstance(item, VariableNode)
                and getattr(item, "name", None) in current
            ):
                current = current - {item.name}

    yield from walk(body, names, True)


def fixed_stride_loop_terminates(
    type_name,
    start,
    bound,
    update,
    constants,
    operator,
    *,
    metadata_bounds=None,
    any_start=False,
):
    """Prove the terminating update as well as every visited induction value.

    With ``any_start`` the caller has proved the start uniform but not its
    value, so the proof must hold for every value of the counter's type, as in
    MLX's ``for (int i = block_idx; i < N; i += blocks)``. A loop then visits
    only values before the bound, so only the terminating update can leave
    the type: ``bound - 1 + stride`` for an ascending ``<`` loop.
    """
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

    known_start = checked_induction_value(start, constants)
    if known_start is None and any_start:
        starts = None
    elif known_start is None:
        return False
    else:
        starts = (known_start,)
    amount = 1 if unary else checked_induction_value(update.value, constants)
    if amount is None:
        return False
    stop = checked_induction_value(bound, constants)
    stops = [stop]
    if stop is None:
        record = (metadata_bounds or {}).get(
            bound.name if isinstance(bound, IdentifierNode) else None
        )
        if record is None:
            return False
        convert = _UnsignedLiteralInt if record["dtype"].startswith("u") else int
        stops = [convert(record["minimum"]), convert(record["maximum"])]
    if not all(
        minimum <= value <= maximum for value in (*(starts or ()), amount, *stops)
    ):
        return False
    # Mixed unsigned arithmetic can change comparisons before assignment.
    if unsigned or any(
        isinstance(value, _UnsignedLiteralInt)
        for value in (*(starts or ()), amount, *stops)
    ):
        if min(*(starts or ()), amount, *stops) < 0:
            return False
        if starts is None and not unsigned:
            # An unknown signed start may convert to a large unsigned value.
            return False
    step = int(amount) * (1 if update.operator in {"+=", "++"} else -1)
    if step == 0 or (step > 0) != (operator in {"<", "<="}):
        return False
    if starts is None:
        # Every value before termination satisfies the condition, so the
        # largest terminating value follows the extreme bound by one stride.
        for stop in stops:
            inclusive = operator in {"<=", ">="}
            final = (
                int(stop) - (not inclusive) + step
                if step > 0
                else int(stop) + (not inclusive) + step
            )
            if not minimum <= final <= maximum:
                return False
        return True
    (start,) = starts
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
