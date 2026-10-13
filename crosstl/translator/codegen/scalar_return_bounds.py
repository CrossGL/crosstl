"""Source-typed range summaries for pure, single-return integer helpers."""

from dataclasses import dataclass

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    ReturnNode,
    UnaryOpNode,
)
from .integer_intervals import (
    convert_integer_intervals,
    integer_arithmetic_intervals,
)


@dataclass(frozen=True)
class _Value:
    source_type: object
    intervals: list


class _ReturnAnalysis:
    def __init__(self, generator, intervals, constants):
        self.generator = generator
        self.intervals = intervals
        self.constants = constants
        self.remaining = 512

    def shape(self, source_type):
        info = self.generator.glsl_value_type_info(source_type)
        if info is None or info["width"] != 1 or info["family"] not in {"int", "uint"}:
            return None
        bits = info["narrow"]["bits"] if info["narrow"] else info["bits"]
        if bits not in {8, 16, 32, 64}:
            return None
        return bits, info["family"] == "int"

    def convert(self, value, source_type):
        shape = self.shape(source_type)
        if value is None or shape is None:
            return None
        return _Value(source_type, convert_integer_intervals(value.intervals, *shape))

    def leaf(self, expression):
        generator = self.generator
        if not generator.glsl_side_effect_free_expression(expression):
            return None
        source_type = generator.glsl_source_expression_type(expression)
        shape = self.shape(source_type)
        type_name = generator.type_name_string(source_type) or ""
        if (
            shape is None
            or "volatile" in (getattr(source_type, "qualifiers", ()) or ())
            or "volatile" in type_name.replace("&", " ").split()
        ):
            return None
        bounds = generator.glsl_checked_private_pointer_interval(
            expression, self.intervals, self.constants, require_arithmetic_types=True
        )
        if bounds is None:
            return None
        bits, signed = shape
        minimum, maximum = (
            (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
            if signed
            else (0, (1 << bits) - 1)
        )
        if not minimum <= bounds[0] <= bounds[1] <= maximum:
            return None
        return _Value(source_type, [bounds])

    def expression(self, expression, bindings=None, active=()):
        self.remaining -= 1
        if self.remaining < 0 or len(active) > 32:
            return None
        generator = self.generator
        if isinstance(expression, IdentifierNode):
            # A callee sees only its own parameters, never caller constants.
            return (
                self.leaf(expression)
                if bindings is None
                else bindings.get(expression.name)
            )
        if isinstance(expression, LiteralNode):
            value = generator.literal_int_value(expression, {})
            source_type = generator.glsl_source_expression_type(expression)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or self.shape(source_type) is None
            ):
                return None
            value = int(value)
            converted = self.convert(_Value(source_type, [(value, value)]), source_type)
            return converted if converted.intervals == [(value, value)] else None
        if isinstance(expression, (ArrayAccessNode, MemberAccessNode)):
            return self.leaf(expression) if bindings is None else None
        if isinstance(expression, UnaryOpNode):
            operator = generator.map_operator(expression.op)
            if operator not in {"+", "-"}:
                return None
            value = self.expression(expression.operand, bindings, active)
            if value is None:
                return None
            info = generator.glsl_value_type_info(value.source_type)
            source_type = (
                generator.glsl_promoted_integer_type(info) or value.source_type
            )
            value = self.convert(value, source_type)
            if operator == "+":
                return value
            result = integer_arithmetic_intervals(
                "-", [(0, 0)], value.intervals, *self.shape(source_type)
            )
            return _Value(source_type, result) if result is not None else None
        if isinstance(expression, BinaryOpNode):
            operator = generator.map_operator(expression.op)
            if operator not in {"+", "-", "*"}:
                return None
            left = self.expression(expression.left, bindings, active)
            right = self.expression(expression.right, bindings, active)
            if left is None or right is None:
                return None
            source_type = generator.glsl_common_arithmetic_type(
                left.source_type, right.source_type, operator
            )
            shape = self.shape(source_type)
            if shape is None:
                return None
            result = integer_arithmetic_intervals(
                operator, left.intervals, right.intervals, *shape
            )
            return _Value(source_type, result) if result is not None else None
        if isinstance(expression, FunctionCallNode):
            if not isinstance(expression.function, IdentifierNode):
                return None
            name = generator.function_call_name(expression)
            arguments = list(expression.arguments or ())
            values = [
                self.expression(argument, bindings, active) for argument in arguments
            ]
            if any(value is None for value in values):
                return None
            if (
                self.shape(name) is not None
                and name not in generator.function_return_types
            ):
                return self.convert(values[0], name) if len(values) == 1 else None
            return self.call(expression, values, active)
        return None

    def call(self, expression, values, active):
        generator = self.generator
        name = generator.function_call_name(expression)
        candidates = list(generator.glsl_function_overloads_by_name.get(name, ()))
        fallback = generator.function_definitions.get(name)
        if fallback is not None:
            candidates.append(fallback)
        unique = {}
        for function in candidates:
            if len(function.parameters or ()) != len(values):
                continue
            signature = generator.glsl_source_function_signature(function)
            previous = unique.get(signature)
            if previous is None or previous.body is None:
                unique[signature] = function
        scored = []
        for function in unique.values():
            scores = [
                generator.glsl_function_type_match_score(
                    value.source_type, parameter.param_type
                )
                for value, parameter in zip(values, function.parameters)
            ]
            if all(score is not None for score in scores):
                scored.append((sum(scores), function))
        if not scored:
            return None
        best = max(score for score, _ in scored)
        winners = [function for score, function in scored if score == best]
        if len(winners) != 1:
            return None
        function = winners[0]
        if (
            id(function) in active
            or isinstance(function.return_type, (PointerType, ReferenceType))
            or self.shape(function.return_type) is None
        ):
            return None
        statements = generator.glsl_software_subgroup_statement_list(function.body)
        if len(statements) != 1 or not isinstance(statements[0], ReturnNode):
            return None
        bindings = {}
        for parameter, value in zip(function.parameters, values):
            source_type = parameter.param_type
            nodes = [parameter, source_type]
            if isinstance(source_type, ReferenceType):
                if source_type.is_mutable:
                    return None
                source_type = source_type.referenced_type
                nodes.append(source_type)
            if (
                isinstance(source_type, (PointerType, ReferenceType))
                or set(generator.glsl_parameter_qualifiers(parameter))
                & {"out", "inout"}
                or any(
                    "volatile" in (getattr(node, "qualifiers", ()) or ())
                    for node in nodes
                )
                or parameter.name in bindings
            ):
                return None
            converted = self.convert(value, source_type)
            if converted is None:
                return None
            bindings[parameter.name] = _Value(parameter.param_type, converted.intervals)
        value = self.expression(statements[0].value, bindings, (*active, id(function)))
        return self.convert(value, function.return_type)


def scalar_return_interval(generator, expression, intervals, constants):
    """Bound a helper result without rewriting it or changing evaluation counts."""
    if not isinstance(expression, FunctionCallNode) or getattr(
        generator, "_glsl_return_bounds_active", False
    ):
        return None
    name = generator.function_call_name(expression)
    if (
        name not in generator.function_definitions
        and name not in generator.glsl_function_overloads_by_name
    ):
        return None
    generator._glsl_return_bounds_active = True
    try:
        value = _ReturnAnalysis(generator, intervals, constants).expression(expression)
    finally:
        generator._glsl_return_bounds_active = False
    if value is None or not value.intervals:
        return None
    return min(low for low, _ in value.intervals), max(
        high for _, high in value.intervals
    )
