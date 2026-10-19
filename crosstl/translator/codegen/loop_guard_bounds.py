"""Guarded entry ranges for finite loops with invariant additive updates."""

from collections import Counter

from ..arithmetic_conversions import ArithmeticScalarKind, target_arithmetic_type
from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
)


def guarded_additive_loop_entries(
    generator,
    function,
    loop,
    incoming,
    entries,
    count,
    *,
    constants,
    escaped,
    mutated,
    lexical,
    checked_integer,
    checked_components,
    mutation_keys,
):
    """Tighten entry hulls only; retain the original loop and conservative exits."""
    if not entries or count is None or not 0 < count <= 64:
        return {}
    statements = list(getattr(function.body, "statements", function.body) or [])
    if loop not in statements or len(statements) > 256:
        return {}
    prefix = statements[: statements.index(loop)]
    body = [
        item.expression if isinstance(item, ExpressionStatementNode) else item
        for item in getattr(loop.body, "statements", loop.body) or []
    ]
    if not all(
        isinstance(item, (AssignmentNode, IdentifierNode))
        or isinstance(item, FunctionCallNode)
        and generator.function_call_name(item) == "buffer_store"
        and "buffer_store" not in generator.function_return_types
        for item in body
    ):
        return {}
    counter = loop.init.name
    guarded_names = generator.glsl_interval_dependency_names(loop.condition)
    overlap = generator.glsl_interval_key_sets_overlap
    if overlap(guarded_names - {counter}, mutated | escaped):
        return {}

    declarations = list(lexical["declarations"].values())
    frequencies = Counter(item["node"].name for item in declarations)
    source_types = {
        item["node"].name: item["source_type"]
        for item in declarations
        if item["source_type"] is not None and frequencies[item["node"].name] == 1
    }
    ambiguous = {name for name, frequency in frequencies.items() if frequency != 1}
    if guarded_names & ambiguous:
        return {}
    definitions = {}
    later_mutations = set()
    for statement in reversed(prefix):
        if isinstance(statement, VariableNode):
            name = statement.name
            dependencies = generator.glsl_interval_dependency_names(
                statement.initial_value
            )
            info = generator.glsl_value_type_info(statement.var_type)
            if (
                info is not None
                and info["family"] in {"int", "uint"}
                and name in source_types
                and name not in dependencies
                and not overlap({name} | dependencies, later_mutations | escaped)
                and not overlap(dependencies, mutated)
                and not (dependencies & ambiguous)
                and not set(statement.qualifiers or ()) & {"volatile", "out", "inout"}
            ):
                definitions[name] = statement
        later_mutations.update(mutation_keys(statement))
    definitions = dict(reversed(list(definitions.items())))
    updates = {}
    for statement in body:
        if not isinstance(statement, AssignmentNode):
            continue
        target = statement.target
        key = generator.glsl_private_pointer_interval_key(target, incoming, constants)
        if key in entries:
            owner = target.object if isinstance(target, MemberAccessNode) else target
            if isinstance(owner, IdentifierNode) and owner.name in definitions:
                step = checked_integer(statement.value, incoming)
                operator = generator.map_operator(statement.operator)
                if step is not None and step[0] == step[1] and operator in {"+=", "-="}:
                    updates[key] = (
                        owner.name,
                        step[0] * (-1 if operator == "-=" else 1),
                    )
    if not updates:
        return {}
    required = guarded_names | {owner for owner, _step in updates.values()}
    for _ in range(len(definitions)):
        dependencies = set(required)
        for name in required:
            if name in definitions:
                dependencies.update(
                    generator.glsl_interval_dependency_names(
                        definitions[name].initial_value
                    )
                )
        if dependencies == required:
            break
        required = dependencies
    definitions = {
        name: value for name, value in definitions.items() if name in required
    }

    def preserving_operands(expression, frame):
        plan = generator.glsl_arithmetic_conversion_plan(
            generator.glsl_source_expression_type(expression.left),
            generator.glsl_source_expression_type(expression.right),
            generator.map_operator(expression.op),
        )
        if plan is None:
            return False
        for operand, target in (
            (expression.left, plan.left_target_type),
            (expression.right, plan.right_target_type),
        ):
            bounds = checked_integer(operand, frame)
            info = target_arithmetic_type(target)
            if bounds is None or info is None or info.lanes != 1:
                return False
            if info.kind not in {
                ArithmeticScalarKind.SIGNED_INTEGER,
                ArithmeticScalarKind.UNSIGNED_INTEGER,
            }:
                return False
            signed = info.kind == ArithmeticScalarKind.SIGNED_INTEGER
            minimum = -(1 << (info.bits - 1)) if signed else 0
            maximum = (1 << (info.bits - int(signed))) - 1
            if not minimum <= bounds[0] <= bounds[1] <= maximum:
                return False
        return True

    def preserving_arithmetic(expression, frame):
        operator = generator.map_operator(expression.op)
        if operator not in {"+", "-", "*"} or not preserving_operands(
            expression, frame
        ):
            return False
        left = checked_integer(expression.left, frame)
        right = checked_integer(expression.right, frame)
        if operator == "+":
            bounds = left[0] + right[0], left[1] + right[1]
        elif operator == "-":
            bounds = left[0] - right[1], left[1] - right[0]
        else:
            products = [a * b for a in left for b in right]
            bounds = min(products), max(products)
        domain = generator.glsl_integer_type_domain(expression, allow_wide=True)
        return (
            domain is not None
            and domain.minimum <= bounds[0] <= bounds[1] <= domain.maximum
        )

    budget = [4096]

    def affine(expression, frame, active=(), depth=0):
        budget[0] -= 1
        if budget[0] < 0 or depth > 32:
            return None
        bounds = checked_integer(expression, frame)
        if bounds is None:
            return None
        if bounds[0] == bounds[1]:
            return bounds[0], {}
        if isinstance(expression, IdentifierNode) and expression.name in definitions:
            name = expression.name
            if name in active:
                return None
            initial = definitions[name].initial_value
            if checked_integer(initial, frame) != bounds:
                return None
            return affine(initial, frame, (*active, name), depth + 1)
        if isinstance(expression, (IdentifierNode, MemberAccessNode)):
            key = generator.glsl_private_pointer_interval_key(
                expression, frame, constants
            )
            return (0, {key: 1}) if key in frame else None
        cast = generator.glsl_integer_scalar_cast_call(expression)
        if cast is not None:
            operand = cast[1][0]
            return (
                affine(operand, frame, active, depth + 1)
                if checked_integer(operand, frame) == bounds
                else None
            )
        if isinstance(expression, UnaryOpNode):
            operator = generator.map_operator(expression.op)
            operand = affine(expression.operand, frame, active, depth + 1)
            if operand is None or operator not in {"+", "-"}:
                return None
            sign = -1 if operator == "-" else 1
            source = checked_integer(expression.operand, frame)
            if source is None:
                return None
            domain = generator.glsl_integer_type_domain(expression, allow_wide=True)
            endpoints = [sign * value for value in source]
            if (
                domain is None
                or not domain.minimum
                <= min(endpoints)
                <= max(endpoints)
                <= domain.maximum
            ):
                return None
            return sign * operand[0], {
                key: sign * value for key, value in operand[1].items()
            }
        if not isinstance(expression, BinaryOpNode) or not preserving_arithmetic(
            expression, frame
        ):
            return None
        operator = generator.map_operator(expression.op)
        left, right = affine(expression.left, frame, active, depth + 1), affine(
            expression.right, frame, active, depth + 1
        )
        if left is None or right is None:
            return None
        if operator == "*":
            if not left[1]:
                left, right = right, left
            if right[1]:
                return None
            return left[0] * right[0], {
                key: value * right[0] for key, value in left[1].items()
            }
        if operator not in {"+", "-"}:
            return None
        sign = -1 if operator == "-" else 1
        terms = dict(left[1])
        for key, coefficient in right[1].items():
            terms[key] = terms.get(key, 0) + sign * coefficient
        return left[0] + sign * right[0], {
            key: value for key, value in terms.items() if value
        }

    def refine(condition, frame):
        if not isinstance(condition, BinaryOpNode):
            return True
        operator = generator.map_operator(condition.op)
        if operator == "&&":
            return refine(condition.left, frame) and refine(condition.right, frame)
        if operator not in {"<", "<=", ">", ">="} or not preserving_operands(
            condition, frame
        ):
            return True
        left, right = affine(condition.left, frame), affine(condition.right, frame)
        if left is None or right is None:
            return True
        if operator in {">", ">="}:
            left, right = right, left
        constant = left[0] - right[0] + int(operator in {"<", ">"})
        terms = dict(left[1])
        for key, coefficient in right[1].items():
            terms[key] = terms.get(key, 0) - coefficient
        terms = {key: value for key, value in terms.items() if value}
        if not terms:
            return constant <= 0
        if len(terms) != 1:
            return True
        ((key, coefficient),) = terms.items()
        lower, upper = frame[key]
        if coefficient > 0:
            upper = min(upper, -constant // coefficient)
        else:
            lower = max(lower, -(constant // coefficient))
        frame[key] = lower, upper
        return lower <= upper

    def recompute(frame):
        def retain(key, value):
            if value is not None and key in frame:
                value = max(frame[key][0], value[0]), min(frame[key][1], value[1])
                if value[0] > value[1]:
                    return False
            frame.pop(key, None)
            if value is not None:
                frame[key] = value
            return True

        for name, definition in definitions.items():
            budget[0] -= 1
            if budget[0] < 0:
                return False
            info = generator.glsl_value_type_info(definition.var_type)
            if info["width"] == 1:
                value = checked_integer(definition.initial_value, frame)
                domain = generator.glsl_integer_type_domain(
                    IdentifierNode(name), allow_wide=True
                )
                if value is not None and (
                    domain is None
                    or not domain.minimum <= value[0] <= value[1] <= domain.maximum
                ):
                    value = None
                if not retain(name, value):
                    return False
            else:
                values = checked_components(
                    definition.initial_value, definition.var_type, frame
                )
                for component in range(info["width"]):
                    key = generator.glsl_index_component_interval_key(name, component)
                    value = values[component] if values is not None else None
                    domain = generator.glsl_integer_type_domain(
                        MemberAccessNode(IdentifierNode(name), "xyzw"[component]),
                        allow_wide=True,
                    )
                    if value is not None and (
                        domain is None
                        or not domain.minimum <= value[0] <= value[1] <= domain.maximum
                    ):
                        value = None
                    if not retain(key, value):
                        return False
        return True

    result = {}
    rejected = set()
    start = checked_integer(loop.init.initial_value, incoming)
    if start is None or start[0] != start[1]:
        return {}
    with generator.glsl_lexical_source_type_scope(source_types):
        for iteration in range(count):
            frame = dict(incoming)
            frame[counter] = (start[0] + iteration,) * 2
            if not refine(loop.condition, frame):
                continue
            if not recompute(frame):
                return {}
            for key, (_owner, step) in updates.items():
                initial = frame.get(key)
                if initial is None:
                    rejected.add(key)
                    continue
                bounds = initial[0] + iteration * step, initial[1] + iteration * step
                old = result.get(key, bounds)
                result[key] = min(old[0], bounds[0]), max(old[1], bounds[1])
    if budget[0] < 0:
        return {}
    return {
        key: bounds
        for key, bounds in result.items()
        if key not in rejected
        and entries[key][0] <= bounds[0] <= bounds[1] <= entries[key][1]
    }
