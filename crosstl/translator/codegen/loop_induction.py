"""Checked finite induction for collective control-flow proofs."""

from ..arithmetic_conversions import ArithmeticScalarKind, source_integer_shape
from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    ReturnNode,
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


def metadata_bound_key(bound):
    """The ``metadata_loop_bindings`` key a loop bound reads, if any.

    A scalar parameter is keyed by its name; one field of a record parameter,
    MLX's ``params.C``, by ``parameter.field``, which no identifier can spell.
    """
    if isinstance(bound, IdentifierNode):
        return bound.name
    if (
        isinstance(bound, MemberAccessNode)
        and isinstance(bound.object_expr, IdentifierNode)
        and isinstance(bound.member, str)
        and bound.member.isidentifier()
    ):
        return f"{bound.object_expr.name}.{bound.member}"
    return None


def metadata_loop_bindings(function, records, mutable_names, walk_ast):
    """Keep only immutable, unshadowed source-owned scalar parameter ranges."""
    result = {}
    for name, record in metadata_parameter_bindings(
        function, records, mutable_names, walk_ast
    ):
        if "element" in record:
            continue
        result[f"{name}.{record['member']}" if "member" in record else name] = record
    return result


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
        record = (metadata_bounds or {}).get(metadata_bound_key(bound))
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


def subgroup_strided_trips(type_name, bound, step, operator, subgroup_count):
    """The trip count shared by every subgroup of a subgroup-strided loop.

    In ``for (i = subgroup_id; i < bound; i += step)`` subgroup ``s`` of
    ``subgroup_count`` takes ``ceil((bound - s) / step)`` trips. That count
    can still agree for every ``s``: MLX's Winograd transforms run
    ``for (c = simd_group_id; c < 32; c += 4)`` in four SIMD-groups, eight
    trips each. Only concrete nonnegative bounds and positive strides qualify;
    the count must agree for every subgroup and every visited and terminating
    counter value must fit the counter's type. Returns the common count, or
    None when that is not proved.
    """
    if not isinstance(type_name, str) or not type_name.isidentifier():
        return None
    shape = source_integer_shape(
        {"min16int": "short", "min16uint": "ushort"}.get(type_name, type_name)
    )
    if shape is None or shape[2] != 1 or shape[1] > 32:
        return None
    if (
        operator not in {"<", "<="}
        or type(subgroup_count) is not int
        or subgroup_count < 1
        or not isinstance(bound, int)
        or isinstance(bound, bool)
        or not isinstance(step, int)
        or isinstance(step, bool)
    ):
        return None
    unsigned = shape[0] == ArithmeticScalarKind.UNSIGNED_INTEGER
    maximum = (1 << (shape[1] - (not unsigned))) - 1
    if not (0 <= int(bound) <= maximum and 0 < int(step) <= maximum):
        return None
    inclusive = operator == "<="

    def trips(start):
        return max(0, -(-(int(bound) + inclusive - start) // int(step)))

    # Trips never increase with the start, so the extreme subgroups decide.
    count = trips(0)
    if trips(subgroup_count - 1) != count:
        return None
    if subgroup_count - 1 + count * int(step) > maximum:
        return None
    return count


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


_INCREMENT_OPERATORS = {"++", "PRE_INCREMENT", "POST_INCREMENT"}
# Unary operators that write their operand or take its address.
_COUNTER_WRITE_OPERATORS = {
    *_INCREMENT_OPERATORS,
    "--",
    "PRE_DECREMENT",
    "POST_DECREMENT",
    "&",
}


def _counter_update_name(update):
    """The counter an ascending ``c += step`` or ``c++`` update writes."""
    if isinstance(update, AssignmentNode):
        if update.operator == "+=" and isinstance(update.target, IdentifierNode):
            return update.target.name
        return None
    if isinstance(update, UnaryOpNode):
        if update.operator in _INCREMENT_OPERATORS and isinstance(
            update.operand, IdentifierNode
        ):
            return update.operand.name
    return None


def _identifier_names(walk_ast, root):
    return {
        node.name
        for node in walk_ast(root)
        if isinstance(node, IdentifierNode) and getattr(node, "name", None)
    }


def init_less_counter_loops(function, walk_ast, writes=None):
    """Find MLX's ``int t = 0; for (; t + C <= T; t += C)`` counter loops.

    The counter is a local declared once, with an initializer, in the
    function's outermost statement list. The only write to it anywhere in the
    function is the update of one later outermost ``for`` that has no
    initializer, and no ``return`` precedes that loop. Every invocation then
    reaches the loop with the counter's initial value; whether the loop is
    uniform, and so whether the counter stays uniform after it, is the
    caller's proof. Returns ``{id(loop): (name, declaration, loop)}``.

    ``writes(node)`` names what a call or a pointer or reference declaration
    may write or alias. Without it, every name such a node reads counts, so a
    counter passed to a call or bound to a reference never qualifies.
    """
    if writes is None:

        def writes(node):
            if isinstance(node, VariableNode):
                return _identifier_names(walk_ast, node.initial_value)
            names = set()
            for argument in node.arguments or []:
                names |= _identifier_names(walk_ast, argument)
            return names

    body = getattr(function, "body", None)
    statements = body.statements if isinstance(body, BlockNode) else body
    if statements is None:
        return {}
    if not isinstance(statements, (list, tuple)):
        statements = [statements]
    parameters = {
        parameter.name
        for parameter in getattr(function, "parameters", None) or []
        if getattr(parameter, "name", None)
    }
    nodes = list(walk_ast(body))
    counts = {}
    for node in nodes:
        if isinstance(node, VariableNode) and node.name:
            counts[node.name] = counts.get(node.name, 0) + 1
    candidates = {}
    declared = {}
    for statement in statements:
        if (
            isinstance(statement, VariableNode)
            and statement.name
            and statement.initial_value is not None
        ):
            declared[statement.name] = statement
        elif isinstance(statement, ForNode) and statement.init is None:
            name = _counter_update_name(statement.update)
            declaration = declared.get(name)
            if (
                declaration is not None
                and counts.get(name) == 1
                and name not in parameters
                and name not in {item[0] for item in candidates.values()}
            ):
                candidates[id(statement)] = (name, declaration, statement)
        if any(isinstance(node, ReturnNode) for node in walk_ast(statement)):
            break
    result = {}
    for loop_id, (name, declaration, loop) in candidates.items():
        writes_elsewhere = False
        for node in nodes:
            if node is loop.update:
                continue
            if (
                isinstance(node, AssignmentNode)
                and isinstance(node.target, IdentifierNode)
                and node.target.name == name
            ):
                writes_elsewhere = True
            elif (
                isinstance(node, UnaryOpNode)
                and node.operator in _COUNTER_WRITE_OPERATORS
                and isinstance(node.operand, IdentifierNode)
                and node.operand.name == name
            ):
                writes_elsewhere = True
            elif isinstance(node, FunctionCallNode) or (
                isinstance(node, VariableNode)
                and isinstance(node.var_type, (PointerType, ReferenceType))
            ):
                # A reference parameter or alias may write the counter.
                writes_elsewhere = name in (writes(node) or set())
            if writes_elsewhere:
                break
        if not writes_elsewhere:
            result[loop_id] = (name, declaration, loop)
    return result


def offset_counter_loop_terminates(type_name, start, offset, stops, step, operator):
    """Prove ``for (; c + offset OP stop; c += step)`` from a known start.

    For each extreme ``stop`` the loop visits ``start, start + step, ...``
    while ``c + offset`` satisfies the condition. Both the visited counters
    and every evaluated ``c + offset``, the last one included, must fit the
    counter's type; the values grow monotonically, so the first and the
    terminating counter decide.
    """
    if not isinstance(type_name, str) or not type_name.isidentifier():
        return False
    shape = source_integer_shape(
        {"min16int": "short", "min16uint": "ushort"}.get(type_name, type_name)
    )
    if shape is None or shape[2] != 1 or shape[1] > 32:
        return False
    if operator not in {"<", "<="}:
        return False
    values = (start, offset, step, *stops)
    if not stops or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        return False
    unsigned = shape[0] == ArithmeticScalarKind.UNSIGNED_INTEGER
    minimum = 0 if unsigned else -(1 << (shape[1] - 1))
    maximum = (1 << (shape[1] - (not unsigned))) - 1
    if int(step) <= 0 or int(offset) < 0:
        return False
    if (
        unsigned or any(isinstance(value, _UnsignedLiteralInt) for value in values)
    ) and min(int(value) for value in values) < 0:
        # Mixed unsigned arithmetic can change comparisons before assignment.
        return False
    start, offset, step = int(start), int(offset), int(step)
    for stop in stops:
        stop = int(stop)
        if not minimum <= stop <= maximum:
            return False
        limit = stop - offset - (operator == "<")
        trips = 0 if start > limit else (limit - start) // step + 1
        final = start + trips * step
        for value in (start, final, start + offset, final + offset):
            if not minimum <= value <= maximum:
                return False
    return True


_STATIC_COMPARISONS = {
    "<": lambda left, right: left < right,
    "<=": lambda left, right: left <= right,
    ">": lambda left, right: left > right,
    ">=": lambda left, right: left >= right,
    "==": lambda left, right: left == right,
    "!=": lambda left, right: left != right,
}
_SIGNED_32 = (-(1 << 31), (1 << 31) - 1)


def _checked_static_value(expression, environment):
    """Evaluate a signed 32-bit integer expression, rejecting any overflow.

    Only literals, names bound in ``environment``, arithmetic, shifts by
    in-range amounts, bitwise operators and comparisons qualify; every
    intermediate value must fit ``int`` so C++ undefined behavior never
    decides the result.
    """
    if isinstance(expression, LiteralNode):
        value = expression.value
        literal_type = getattr(getattr(expression, "literal_type", None), "name", None)
        if isinstance(value, bool):
            return int(value)
        if (
            isinstance(value, _UnsignedLiteralInt)
            or not isinstance(value, int)
            or literal_type not in {None, "int"}
        ):
            # ``8u`` keeps an int value with a uint literal type; unsigned or
            # wider operands change the comparison and arithmetic rules.
            return None
    elif isinstance(expression, IdentifierNode):
        value = environment.get(expression.name)
        if value is None or isinstance(value, _UnsignedLiteralInt):
            return None
    elif isinstance(expression, UnaryOpNode):
        operand = _checked_static_value(expression.operand, environment)
        if operand is None:
            return None
        value = {
            "+": operand,
            "-": -operand,
            "~": ~operand,
            "!": int(not operand),
        }.get(expression.operator)
    elif isinstance(expression, BinaryOpNode):
        left = _checked_static_value(expression.left, environment)
        right = _checked_static_value(expression.right, environment)
        if left is None or right is None:
            return None
        operator = expression.operator
        if operator in _STATIC_COMPARISONS:
            value = int(_STATIC_COMPARISONS[operator](left, right))
        elif operator in {"&&", "||"}:
            both = bool(left) and bool(right)
            value = int(both if operator == "&&" else bool(left) or bool(right))
        elif operator == "+":
            value = left + right
        elif operator == "-":
            value = left - right
        elif operator == "*":
            value = left * right
        elif operator in {"/", "%"}:
            if right == 0 or (left == _SIGNED_32[0] and right == -1):
                return None
            quotient = abs(left) // abs(right)
            quotient = -quotient if (left < 0) != (right < 0) else quotient
            value = quotient if operator == "/" else left - quotient * right
        elif operator in {"<<", ">>"}:
            if not 0 <= right < 31 or left < 0:
                return None
            value = left << right if operator == "<<" else left >> right
        elif operator in {"&", "|", "^"}:
            value = {"&": left & right, "|": left | right, "^": left ^ right}[operator]
        else:
            return None
    else:
        return None
    if value is None or not _SIGNED_32[0] <= value <= _SIGNED_32[1]:
        return None
    return value


def static_loop_trip_count(
    type_name, start, condition, update, loop_name, constants, limit=4096
):
    """Run a loop's control on constants alone and return its trip count.

    ``start``, the condition and the update may read only ``loop_name`` and
    the immutable literal ``constants``, as MLX's
    ``for (int step = 1; (1 << step) < C; step++)``. Every invocation then
    computes the same sequence. Each value, the terminating update included,
    must fit both ``int`` and the counter's type; a loop that exceeds
    ``limit`` trips or reads anything else returns None.
    """
    if not isinstance(type_name, str) or not type_name.isidentifier():
        return None
    shape = source_integer_shape(
        {"min16int": "short", "min16uint": "ushort"}.get(type_name, type_name)
    )
    if (
        shape is None
        or shape[2] != 1
        or shape[1] > 32
        or shape[0] == ArithmeticScalarKind.UNSIGNED_INTEGER
    ):
        return None
    minimum = -(1 << (shape[1] - 1))
    maximum = (1 << (shape[1] - 1)) - 1
    environment = {
        name: value
        for name, value in constants.items()
        if name != loop_name and not isinstance(value, _UnsignedLiteralInt)
    }
    value = _checked_static_value(start, environment)
    if value is None or not minimum <= value <= maximum:
        return None
    if isinstance(update, UnaryOpNode):
        if update.operator in _INCREMENT_OPERATORS:
            step = 1
        elif update.operator in {"--", "PRE_DECREMENT", "POST_DECREMENT"}:
            step = -1
        else:
            return None
        if not (
            isinstance(update.operand, IdentifierNode)
            and update.operand.name == loop_name
        ):
            return None
        operator, amount = "+=", step
    elif isinstance(update, AssignmentNode):
        if not (
            isinstance(update.target, IdentifierNode)
            and update.target.name == loop_name
        ):
            return None
        operator = update.operator
        amount = _checked_static_value(update.value, environment)
        if amount is None or operator not in {"+=", "-=", "*=", "<<=", ">>="}:
            return None
    else:
        return None
    for trips in range(limit + 1):
        taken = _checked_static_value(condition, {**environment, loop_name: value})
        if taken is None:
            return None
        if not taken:
            return trips
        if operator == "+=":
            value += amount
        elif operator == "-=":
            value -= amount
        elif operator == "*=":
            value *= amount
        elif not 0 <= amount < 31 or value < 0:
            return None
        elif operator == "<<=":
            value <<= amount
        else:
            value >>= amount
        if not minimum <= value <= maximum:
            return None
    return None
