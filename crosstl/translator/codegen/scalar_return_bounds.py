"""Source-typed return bounds for bounded, side-effect-free integer helpers."""

from dataclasses import dataclass

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayType,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    ConstructorNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MemberAccessNode,
    PointerType,
    PrimitiveType,
    ReferenceType,
    ReturnNode,
    UnaryOpNode,
    VariableNode,
)
from .integer_intervals import (
    convert_integer_intervals,
    integer_arithmetic_intervals,
    merge_integer_intervals,
)


@dataclass(frozen=True)
class _Value:
    source_type: object
    intervals: list


@dataclass(frozen=True)
class _Vector:
    source_type: object
    components: tuple


@dataclass(frozen=True)
class _MetadataArray:
    source_type: object
    elements: dict


@dataclass(frozen=True)
class _Aggregate:
    source_type: object
    fields: dict


@dataclass(frozen=True)
class _Handle:
    """An opaque pointer, buffer or texture: it may be passed on, but reads through it are unknown."""

    source_type: object


@dataclass(frozen=True)
class _CallResult:
    value: object
    updates: dict
    function: object


@dataclass(frozen=True)
class _Flow:
    continuing: object
    returned: list


class _ReturnAnalysis:
    def __init__(self, generator, intervals, constants):
        self.generator = generator
        self.intervals = intervals
        self.constants = constants
        self.remaining = 4096

    def aggregate_fields(self, source_type):
        if isinstance(source_type, (PointerType, ReferenceType, ArrayType)):
            return {}
        name = self.generator.type_name_string(source_type) or ""
        if "volatile" in name.split():
            return {}
        return self.generator.glsl_struct_member_types(name)

    def is_empty_aggregate(self, source_type):
        """Whether ``source_type`` is a declared struct without instance members.

        Such a value, for example a tag or an ``integral_constant`` whose only member is
        static, carries no state, so it is fully known even when it is a caller's value.
        """
        if source_type is None or isinstance(
            source_type, (PointerType, ReferenceType, ArrayType)
        ):
            return False
        generator = self.generator
        name = generator.glsl_source_type_identifier_name(
            generator.type_name_string(source_type) or ""
        )
        members = generator.struct_member_types.get(name)
        return members is not None and not members

    def is_handle_type(self, source_type):
        """Whether ``source_type`` is a pointer, a structured buffer or an opaque texture or sampler."""
        if isinstance(source_type, PointerType):
            return True
        if source_type is None or isinstance(source_type, (ReferenceType, ArrayType)):
            return False
        generator = self.generator
        return bool(
            generator.is_structured_buffer_type(source_type)
            or generator.is_opaque_resource_type(
                generator.type_name_string(source_type)
            )
        )

    def owned_path(self, expression):
        if isinstance(expression, IdentifierNode):
            return (expression.name,)
        if isinstance(expression, MemberAccessNode):
            owner = self.owned_path(expression.object)
            if owner is not None:
                return (*owner, str(expression.member))
        return None

    def store(self, target, value, bindings, readonly):
        path = self.owned_path(target)
        if path is None or path[0] not in bindings or path[0] in readonly:
            return False

        def replace(owner, members):
            if not members:
                return (
                    self.convert(value, owner.source_type)
                    if owner is not None
                    else None
                )
            member, *remaining = members
            if isinstance(owner, _Aggregate):
                source_type = self.aggregate_fields(owner.source_type).get(member)
                if source_type is None or "const" in (
                    getattr(source_type, "qualifiers", ()) or ()
                ):
                    return None
                updated = (
                    replace(owner.fields.get(member), remaining)
                    if remaining
                    else self.convert(value, source_type)
                )
                if updated is not None:
                    return _Aggregate(
                        owner.source_type, {**owner.fields, member: updated}
                    )
            elif isinstance(owner, _Vector) and not remaining:
                index = self.generator.glsl_index_component_index(member)
                if index is not None and index < len(owner.components):
                    updated = self.convert(value, owner.components[index].source_type)
                    if updated is not None:
                        components = list(owner.components)
                        components[index] = updated
                        return _Vector(owner.source_type, tuple(components))
            return None

        updated = replace(bindings[path[0]], path[1:])
        if updated is None:
            return False
        bindings[path[0]] = updated
        return True

    def component_type(self, source_type):
        info = self.generator.glsl_value_type_info(source_type)
        if info is None or info["family"] not in {"int", "uint"}:
            return None
        if info["width"] == 1:
            return source_type
        if 2 <= info["width"] <= 4:
            if info["narrow"] is not None:
                return info["family"] + str(info["narrow"]["bits"])
            return self.generator.vector_component_type(info["mapped"])
        return None

    def join(self, values):
        if not values or any(value is None for value in values):
            return None
        first = values[0]
        if any(
            type(value) is not type(first) or value.source_type != first.source_type
            for value in values
        ):
            return None
        if isinstance(first, (_MetadataArray, _Handle)):
            return first if all(value == first for value in values) else None
        if isinstance(first, _Aggregate):
            return _Aggregate(
                first.source_type,
                {
                    name: self.join([value.fields.get(name) for value in values])
                    for name in first.fields
                },
            )
        if isinstance(first, _Vector):
            components = tuple(
                self.join([value.components[i] for value in values])
                for i in range(len(first.components))
            )
            return _Vector(first.source_type, components) if all(components) else None
        return _Value(
            first.source_type,
            merge_integer_intervals(
                [bounds for value in values for bounds in value.intervals]
            ),
        )

    def shape(self, source_type):
        info = self.generator.glsl_value_type_info(source_type)
        if info is None or info["width"] != 1 or info["family"] not in {"int", "uint"}:
            return None
        bits = info["narrow"]["bits"] if info["narrow"] else info["bits"]
        if bits not in {8, 16, 32, 64}:
            return None
        return bits, info["family"] == "int"

    def convert(self, value, source_type):
        if isinstance(value, _Aggregate):
            name = self.generator.type_name_string
            return (
                _Aggregate(source_type, dict(value.fields))
                if name(value.source_type) == name(source_type)
                else None
            )
        info = self.generator.glsl_value_type_info(source_type)
        if info is not None and 2 <= info["width"] <= 4 and isinstance(value, _Vector):
            if len(value.components) != info["width"]:
                return None
            component_type = self.component_type(source_type)
            components = tuple(
                self.convert(item, component_type) for item in value.components
            )
            return _Vector(source_type, components) if all(components) else None
        shape = self.shape(source_type)
        if not isinstance(value, _Value) or shape is None:
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

    def caller_value(self, expression, active=()):
        self.remaining -= 1
        if self.remaining < 0:
            return None
        generator = self.generator
        source_type = generator.glsl_source_expression_type(expression)
        fields = self.aggregate_fields(source_type)
        if fields:
            name = generator.type_name_string(source_type)
            if name in active or len(active) >= 8:
                return None
            return _Aggregate(
                source_type,
                {
                    member: self.caller_value(
                        MemberAccessNode(expression, member), (*active, name)
                    )
                    for member in fields
                },
            )
        if self.is_empty_aggregate(
            source_type
        ) and generator.glsl_side_effect_free_expression(expression):
            return _Aggregate(source_type, {})
        if isinstance(expression, IdentifierNode):
            prefix = f"@metadata:{expression.name}["
            elements = {}
            for key, bounds in self.intervals.items():
                if (
                    not isinstance(key, str)
                    or not key.startswith(prefix)
                    or not key.endswith("]")
                ):
                    continue
                index = key[len(prefix) : -1]
                if not index.isdecimal():
                    return None
                read = ArrayAccessNode(
                    expression, LiteralNode(int(index), PrimitiveType("int"))
                )
                value = self.leaf(read)
                if value is None:
                    return None
                elements[int(index)] = value
            if elements:
                return _MetadataArray(source_type, elements)
            if self.is_handle_type(source_type):
                return _Handle(source_type)
        info = generator.glsl_value_type_info(source_type)
        if info is not None and 2 <= info["width"] <= 4:
            components = tuple(
                self.leaf(MemberAccessNode(expression, name))
                for name in "xyzw"[: info["width"]]
            )
            return _Vector(source_type, components) if all(components) else None
        return self.leaf(expression)

    def arithmetic(self, operator, left, right):
        if (
            left is None
            or right is None
            or isinstance(left, (_MetadataArray, _Aggregate, _Handle))
            or isinstance(right, (_MetadataArray, _Aggregate, _Handle))
        ):
            return None
        source_type = self.generator.glsl_common_arithmetic_type(
            left.source_type, right.source_type, operator
        )
        if isinstance(left, _Vector) or isinstance(right, _Vector):
            count = (
                len(left.components)
                if isinstance(left, _Vector)
                else len(right.components)
            )
            a = left.components if isinstance(left, _Vector) else (left,) * count
            b = right.components if isinstance(right, _Vector) else (right,) * count
            if len(a) != len(b) or self.component_type(source_type) is None:
                return None
            values = tuple(self.arithmetic(operator, x, y) for x, y in zip(a, b))
            return (
                self.convert(_Vector(source_type, values), source_type)
                if all(values)
                else None
            )
        shape = self.shape(source_type)
        if shape is None:
            return None
        result = integer_arithmetic_intervals(
            operator, left.intervals, right.intervals, *shape
        )
        return _Value(source_type, result) if result is not None else None

    def read(self, value, index):
        if not isinstance(index, _Value):
            return None
        lower, upper = min(x for x, _ in index.intervals), max(
            y for _, y in index.intervals
        )
        elements = (
            value.elements
            if isinstance(value, _MetadataArray)
            else dict(enumerate(value.components)) if isinstance(value, _Vector) else {}
        )
        if lower < 0 or upper - lower >= len(elements):
            return None
        return self.join([elements.get(i) for i in range(lower, upper + 1)])

    def constructor(self, source_type, values):
        if any(value is None for value in values):
            return None
        fields = self.aggregate_fields(source_type)
        if fields:
            if len(values) != len(fields):
                return None
            converted = {
                name: self.convert(value, field_type)
                for (name, field_type), value in zip(fields.items(), values)
            }
            return (
                _Aggregate(source_type, converted) if all(converted.values()) else None
            )
        info = self.generator.glsl_value_type_info(source_type)
        if info is None:
            return None
        if info["width"] == 1:
            return self.convert(values[0], source_type) if len(values) == 1 else None
        components = tuple(
            component
            for value in values
            for component in (
                value.components if isinstance(value, _Vector) else (value,)
            )
        )
        if len(components) == 1:
            components *= info["width"]
        return self.convert(_Vector(source_type, components), source_type)

    def expression(self, expression, bindings=None, active=()):
        self.remaining -= 1
        if self.remaining < 0 or len(active) > 32:
            return None
        generator = self.generator
        if isinstance(expression, IdentifierNode):
            # A callee sees only its own parameters, never caller constants.
            return (
                self.caller_value(expression)
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
        if isinstance(expression, ArrayLiteralNode):
            values = tuple(
                self.expression(item, bindings, active) for item in expression.elements
            )
            return (
                _Vector(None, values)
                if all(isinstance(value, _Value) for value in values)
                else None
            )
        if isinstance(expression, ConstructorNode):
            return self.constructor(
                expression.constructor_type,
                [
                    self.expression(item, bindings, active)
                    for item in expression.arguments
                ],
            )
        if isinstance(expression, ArrayAccessNode):
            result = self.read(
                self.expression(expression.array, bindings, active),
                self.expression(expression.index, bindings, active),
            )
            return (
                result
                if result is not None or bindings is not None
                else self.leaf(expression)
            )
        if isinstance(expression, MemberAccessNode):
            owner = self.expression(expression.object, bindings, active)
            if isinstance(owner, _Aggregate):
                return owner.fields.get(str(expression.member))
            index = generator.glsl_index_component_index(expression.member)
            if (
                isinstance(owner, _Vector)
                and index is not None
                and index < len(owner.components)
            ):
                return owner.components[index]
            return self.leaf(expression) if bindings is None else None
        if isinstance(expression, UnaryOpNode):
            operator = generator.map_operator(expression.op)
            if operator not in {"+", "-"}:
                return None
            value = self.expression(expression.operand, bindings, active)
            if not isinstance(value, _Value):
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
            if operator not in {"+", "-", "*", "/", "%", "&", "|", "^"}:
                return None
            left = self.expression(expression.left, bindings, active)
            right = self.expression(expression.right, bindings, active)
            return self.arithmetic(operator, left, right)
        if isinstance(expression, FunctionCallNode):
            if not isinstance(expression.function, IdentifierNode):
                return None
            name = generator.function_call_name(expression)
            arguments = list(expression.arguments or ())
            values = [
                self.argument(argument, bindings, active) for argument in arguments
            ]
            if any(value is None for value in values):
                return None
            if (
                self.component_type(name) is not None or self.aggregate_fields(name)
            ) and name not in generator.function_return_types:
                return self.constructor(name, values)
            if (
                name == "buffer_load"
                and name not in generator.function_return_types
                and len(values) == 2
            ):
                return self.read(*values)
            return self.call(expression, values, active)
        return None

    def argument(self, argument, bindings, active):
        """Evaluate a call argument; an untyped caller name may still bind a handle parameter.

        Resource specialization can leave a caller's hidden buffer argument without a source
        type. Such a name can only bind a pointer or resource parameter (see ``score``), and
        every read through that parameter is unknown, so no value is invented for it.
        """
        value = self.expression(argument, bindings, active)
        if (
            value is None
            and bindings is None
            and isinstance(argument, IdentifierNode)
            and self.generator.glsl_source_expression_type(argument) is None
        ):
            return _Handle(None)
        return value

    def score(self, value, parameter):
        if isinstance(value, _Handle):
            if not self.is_handle_type(parameter.param_type):
                return None
            if value.source_type is None:
                return 0
        if isinstance(
            value, _MetadataArray
        ) and self.generator.is_structured_buffer_type(value.source_type):
            source_type = parameter.param_type
            element = (
                source_type.pointee_type
                if isinstance(source_type, PointerType)
                else (
                    source_type.element_type
                    if isinstance(source_type, ArrayType)
                    else None
                )
            )
            if self.shape(element) is not None and all(
                self.shape(item.source_type) == self.shape(element)
                for item in value.elements.values()
            ):
                return 2
            return None
        if isinstance(value, _Vector) and value.source_type is None:
            converted = self.convert(value, parameter.param_type)
            return 1 if converted is not None else None
        return self.generator.glsl_function_type_match_score(
            value.source_type, parameter.param_type
        )

    def call(self, expression, values, active):
        result = self.invoke(expression, values, active, allow_updates=False)
        return result.value if result is not None else None

    def invoke(self, expression, values, active, *, allow_updates):
        self.remaining -= 1
        if self.remaining < 0 or len(active) > 32:
            return None
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
            # Emitted type spellings can erase source pointer address spaces.
            signature = (
                generator.glsl_source_function_signature(function),
                tuple(
                    (
                        getattr(parameter.param_type, "address_space", None),
                        getattr(parameter.param_type, "is_mutable", None),
                        getattr(parameter.param_type, "access_mode", None),
                        tuple(sorted(str(item) for item in parameter.qualifiers)),
                        tuple(
                            sorted(
                                (item.kind, item.scope or "")
                                for item in generator.typed_resource_memory_qualifiers(
                                    parameter
                                )
                            )
                        ),
                    )
                    for parameter in function.parameters
                ),
            )
            previous = unique.get(signature)
            if previous is None or previous.body is None:
                unique[signature] = function
        scored = []
        for function in unique.values():
            scores = [
                self.score(value, parameter)
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
        if allow_updates:
            arguments = list(expression.arguments or ())
            for index, (parameter, value) in enumerate(
                zip(function.parameters, values)
            ):
                if not isinstance(
                    value, _Aggregate
                ) or "inout" not in generator.glsl_parameter_qualifiers(parameter):
                    continue
                argument = arguments[index]
                path = self.owned_path(argument)
                if path is None:
                    return None
                for other_index, other in enumerate(function.parameters):
                    if other_index == index:
                        continue
                    if isinstance(other.param_type, ReferenceType) or set(
                        generator.glsl_parameter_qualifiers(other)
                    ) & {"inout", "out"}:
                        dependencies = generator.glsl_interval_dependency_names(
                            arguments[other_index]
                        )
                        if generator.glsl_interval_key_sets_overlap(
                            {".".join(path)}, dependencies
                        ):
                            return None
        is_void = generator.type_name_string(function.return_type) == "void"
        if (
            id(function) in active
            or isinstance(function.return_type, (PointerType, ReferenceType))
            or not (
                self.component_type(function.return_type) is not None
                or self.aggregate_fields(function.return_type)
                or is_void
                and allow_updates
            )
        ):
            return None
        statements = generator.glsl_software_subgroup_statement_list(function.body)
        if function.body is None:
            return None
        # Value parameters and locals belong to this call's frame. References
        # and metadata keep their source identity and remain read-only below.
        bindings = {}
        readonly = set()
        mutable = {}
        for index, (parameter, value) in enumerate(zip(function.parameters, values)):
            source_type = parameter.param_type
            nodes = [parameter, source_type]
            if parameter.name in bindings:
                return None
            if isinstance(source_type, ReferenceType):
                if source_type.is_mutable:
                    return None
                source_type = source_type.referenced_type
                nodes.append(source_type)
                readonly.add(parameter.name)
            qualifiers = set(generator.glsl_parameter_qualifiers(parameter))
            if (
                "out" in qualifiers
                or "inout" in qualifiers
                and not isinstance(value, _Aggregate)
                or any(
                    "volatile" in (getattr(node, "qualifiers", ()) or ())
                    for node in nodes
                )
                or any(
                    qualifier.kind == "volatile"
                    for qualifier in generator.typed_resource_memory_qualifiers(
                        parameter
                    )
                )
            ):
                return None
            if "inout" in qualifiers:
                if allow_updates:
                    mutable[index] = parameter.name
                else:
                    readonly.add(parameter.name)
            if "const" in (getattr(parameter, "qualifiers", ()) or ()):
                readonly.add(parameter.name)
            if isinstance(value, _MetadataArray):
                if not isinstance(
                    source_type, (PointerType, ArrayType)
                ) or "volatile" in (getattr(parameter, "qualifiers", ()) or ()):
                    return None
                if isinstance(source_type, PointerType):
                    if (
                        source_type.address_space != "constant"
                        and source_type.is_mutable
                    ):
                        return None
                    element_type = source_type.pointee_type
                else:
                    if "constant" not in (getattr(parameter, "qualifiers", ()) or ()):
                        return None
                    element_type = source_type.element_type
                if "volatile" in (getattr(element_type, "qualifiers", ()) or ()):
                    return None
                if any(
                    self.shape(item.source_type) != self.shape(element_type)
                    for item in value.elements.values()
                ):
                    return None
                bindings[parameter.name] = _MetadataArray(source_type, value.elements)
                readonly.add(parameter.name)
                continue
            if isinstance(value, _Handle):
                if not self.is_handle_type(source_type):
                    return None
                bindings[parameter.name] = _Handle(source_type)
                readonly.add(parameter.name)
                continue
            if (
                isinstance(source_type, (PointerType, ReferenceType))
                or parameter.name in bindings
            ):
                return None
            converted = self.convert(value, source_type)
            if converted is None:
                return None
            bindings[parameter.name] = (
                _Aggregate(parameter.param_type, dict(converted.fields))
                if isinstance(converted, _Aggregate)
                else (
                    _Vector(parameter.param_type, converted.components)
                    if isinstance(converted, _Vector)
                    else _Value(parameter.param_type, converted.intervals)
                )
            )
        active = (*active, id(function))
        flow = self.execute(
            statements, bindings, readonly, active, function.return_type
        )
        if flow is None:
            return None
        exits = list(flow.returned)
        if flow.continuing is not None:
            if not is_void:
                return None
            exits.append((None, flow.continuing))
        if not exits:
            return None
        value = None if is_void else self.join([value for value, _ in exits])
        if not is_void and value is None:
            return None
        updates = {
            index: self.join([frame[name] for _, frame in exits])
            for index, name in mutable.items()
        }
        if any(value is None for value in updates.values()):
            return None
        return _CallResult(value, updates, function)

    def truth(self, expression, bindings, active):
        self.remaining -= 1
        if self.remaining < 0:
            return None
        if isinstance(expression, BinaryOpNode):
            op = self.generator.map_operator(expression.op)
            if op in {"&&", "||"}:
                left = self.truth(expression.left, bindings, active)
                if left is None:
                    return None
                if left == {False} and op == "&&" or left == {True} and op == "||":
                    return left
                right = self.truth(expression.right, bindings, active)
                return (
                    {a and b if op == "&&" else a or b for a in left for b in right}
                    if right is not None
                    else None
                )
            if op in {"<", "<=", ">", ">=", "==", "!="}:
                left = self.expression(expression.left, bindings, active)
                right = self.expression(expression.right, bindings, active)
                if not isinstance(left, _Value) or not isinstance(right, _Value):
                    return None
                common = self.generator.glsl_common_arithmetic_type(
                    left.source_type, right.source_type, "+"
                )
                left, right = self.convert(left, common), self.convert(right, common)
                if left is None or right is None:
                    return None
                a, b = min(x for x, _ in left.intervals), max(
                    y for _, y in left.intervals
                )
                c, d = min(x for x, _ in right.intervals), max(
                    y for _, y in right.intervals
                )
                if op == "<":
                    return {True} if b < c else {False} if a >= d else {True, False}
                if op == "<=":
                    return {True} if b <= c else {False} if a > d else {True, False}
                if op == ">":
                    return {True} if a > d else {False} if b <= c else {True, False}
                if op == ">=":
                    return {True} if a >= d else {False} if b < c else {True, False}
                equal = (
                    {True}
                    if a == b == c == d
                    else {False} if b < c or d < a else {True, False}
                )
                return equal if op == "==" else {not item for item in equal}
        value = self.expression(expression, bindings, active)
        if not isinstance(value, _Value):
            return None
        return ({False} if any(a <= 0 <= b for a, b in value.intervals) else set()) | (
            {True} if any(a != 0 or b != 0 for a, b in value.intervals) else set()
        )

    def statements(self, statements, bindings, readonly, active):
        readonly = set(readonly)
        for statement in statements:
            self.remaining -= 1
            if self.remaining < 0:
                return False
            if isinstance(statement, ExpressionStatementNode):
                statement = statement.expression
            if isinstance(statement, VariableNode):
                if (
                    statement.name in bindings
                    or isinstance(
                        statement.var_type, (ReferenceType, PointerType, ArrayType)
                    )
                    or "volatile" in statement.qualifiers
                ):
                    return False
                fields = self.aggregate_fields(statement.var_type)
                value = (
                    _Aggregate(statement.var_type, {name: None for name in fields})
                    if fields and statement.initial_value is None
                    else self.convert(
                        self.expression(statement.initial_value, bindings, active),
                        statement.var_type,
                    )
                )
                if value is None:
                    return False
                bindings[statement.name] = value
                if "const" in statement.qualifiers:
                    readonly.add(statement.name)
            elif isinstance(statement, FunctionCallNode):
                if not self.apply_call(statement, bindings, readonly, active):
                    return False
            else:
                if isinstance(statement, UnaryOpNode) and self.generator.map_operator(
                    statement.op
                ) in {"++", "--"}:
                    target = statement.operand
                    op = (
                        "+"
                        if self.generator.map_operator(statement.op) == "++"
                        else "-"
                    )
                    value = self.arithmetic(
                        op,
                        self.expression(target, bindings, active),
                        _Value("int", [(1, 1)]),
                    )
                elif isinstance(statement, AssignmentNode):
                    target = statement.target
                    value = self.expression(statement.value, bindings, active)
                    op = self.generator.map_operator(statement.operator)
                    if op != "=":
                        if op not in {"+=", "-=", "*=", "/=", "%="}:
                            return False
                        value = self.arithmetic(
                            op[:-1], self.expression(target, bindings, active), value
                        )
                else:
                    return False
                if not self.store(target, value, bindings, readonly):
                    return False
        return True

    def join_frames(self, names, frames):
        joined = {
            name: self.join([frame.get(name) for frame in frames]) for name in names
        }
        return joined if all(value is not None for value in joined.values()) else None

    def execute(self, statements, bindings, readonly, active, return_type):
        readonly = set(readonly)
        returned = []
        for statement in statements:
            if isinstance(statement, (ReturnNode, BlockNode, IfNode, ForNode)):
                self.remaining -= 1
                if self.remaining < 0:
                    return None
            if isinstance(statement, ReturnNode):
                if self.generator.type_name_string(return_type) == "void":
                    if statement.value is not None:
                        return None
                    value = None
                else:
                    value = self.convert(
                        self.expression(statement.value, bindings, active), return_type
                    )
                    if value is None:
                        return None
                returned.append((value, dict(bindings)))
                return _Flow(None, returned)
            if isinstance(statement, BlockNode):
                flow = self.execute(
                    statement.statements, dict(bindings), readonly, active, return_type
                )
            elif isinstance(statement, IfNode):
                branches = [(statement.condition, statement.if_body)] + list(
                    zip(
                        statement.else_if_conditions or (),
                        statement.else_if_bodies or (),
                    )
                )
                continuing, exits = [], []
                fallthrough = True
                for condition, body in branches:
                    if not fallthrough:
                        break
                    truth = self.truth(condition, bindings, active)
                    if truth is None:
                        return None
                    if True in truth:
                        branch = self.execute(
                            self.generator.glsl_software_subgroup_statement_list(body),
                            dict(bindings),
                            readonly,
                            active,
                            return_type,
                        )
                        if branch is None:
                            return None
                        exits.extend(branch.returned)
                        if branch.continuing is not None:
                            continuing.append(branch.continuing)
                    fallthrough = False in truth
                if fallthrough:
                    branch = self.execute(
                        self.generator.glsl_software_subgroup_statement_list(
                            statement.else_body
                        ),
                        dict(bindings),
                        readonly,
                        active,
                        return_type,
                    )
                    if branch is None:
                        return None
                    exits.extend(branch.returned)
                    if branch.continuing is not None:
                        continuing.append(branch.continuing)
                joined = self.join_frames(bindings, continuing) if continuing else None
                if continuing and joined is None:
                    return None
                flow = _Flow(joined, exits)
            elif isinstance(statement, ForNode):
                flow = self.loop(statement, bindings, readonly, active, return_type)
            else:
                if not self.statements([statement], bindings, readonly, active):
                    return None
                if (
                    isinstance(statement, VariableNode)
                    and "const" in statement.qualifiers
                ):
                    readonly.add(statement.name)
                continue
            if flow is None:
                return None
            returned.extend(flow.returned)
            if flow.continuing is None:
                return _Flow(None, returned)
            for name in bindings:
                bindings[name] = flow.continuing[name]
        return _Flow(bindings, returned)

    def apply_call(self, expression, bindings, readonly, active):
        arguments = list(expression.arguments or ())
        values = [self.expression(argument, bindings, active) for argument in arguments]
        if any(value is None for value in values):
            return False
        result = self.invoke(expression, values, active, allow_updates=True)
        if result is None:
            return False
        owners = [
            self.owned_path(argument)
            for argument, value in zip(arguments, values)
            if isinstance(value, _Aggregate)
        ]
        if any(owner is None for owner in owners) or len(owners) != len(set(owners)):
            return False
        updated = dict(bindings)
        for index, value in result.updates.items():
            if not self.store(arguments[index], value, updated, readonly):
                return False
        bindings.update(updated)
        return True

    def loop(self, node, bindings, readonly, active, return_type):
        if (
            not isinstance(node.init, VariableNode)
            or node.condition is None
            or node.update is None
        ):
            return None
        current = dict(bindings)
        if not self.statements([node.init], current, readonly, active):
            return None
        if "const" in node.init.qualifiers:
            readonly = readonly | {node.init.name}
        exits = []
        returned = []
        for iteration in range(65):
            condition = self.truth(node.condition, current, active)
            if condition is None:
                return None
            if False in condition:
                exits.append(dict(current))
            if True not in condition:
                break
            if iteration == 64:
                return None
            frame = dict(current)
            body = self.generator.glsl_software_subgroup_statement_list(node.body)
            flow = self.execute(body, frame, readonly, active, return_type)
            if flow is None:
                return None
            returned.extend(flow.returned)
            if flow.continuing is None:
                break
            current = {name: flow.continuing[name] for name in current}
            if not self.statements([node.update], current, readonly, active):
                return None
        joined = self.join_frames(bindings, exits) if exits else None
        if exits and joined is None:
            return None
        return _Flow(joined, returned)


def _return_value(generator, expression, intervals, constants):
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
    return value


def scalar_return_interval(generator, expression, intervals, constants):
    """Bound a helper result without rewriting it or changing evaluation counts."""
    value = _return_value(generator, expression, intervals, constants)
    if not isinstance(value, _Value) or not value.intervals:
        return None
    return min(low for low, _ in value.intervals), max(
        high for _, high in value.intervals
    )


def aggregate_call_intervals(
    generator, expression, intervals, constants, *, additive=False
):
    """Return proved field updates; caller ownership and escape checks are required."""
    if not isinstance(expression, FunctionCallNode) or getattr(
        generator, "_glsl_return_bounds_active", False
    ):
        return None
    generator._glsl_return_bounds_active = True
    try:
        analysis = _ReturnAnalysis(generator, intervals, constants)
        arguments = list(expression.arguments or ())
        values = [analysis.expression(argument) for argument in arguments]
        if any(value is None for value in values):
            return None
        result = analysis.invoke(expression, values, (), allow_updates=True)
        if result is None:
            return None
        owners = [
            argument.name
            for argument, value in zip(arguments, values)
            if isinstance(argument, IdentifierNode) and isinstance(value, _Aggregate)
        ]
        if len(owners) != len(set(owners)):
            return None
        if additive:
            body = generator.glsl_software_subgroup_statement_list(result.function.body)
            if len(body) != 1:
                return None
            statement = body[0]
            if isinstance(statement, ExpressionStatementNode):
                statement = statement.expression
            if not isinstance(statement, AssignmentNode) or generator.map_operator(
                statement.operator
            ) not in {"+=", "-="}:
                return None
            target = statement.target
            if not isinstance(target, MemberAccessNode) or not isinstance(
                target.object, IdentifierNode
            ):
                return None
            parameters = result.function.parameters
            indices = [
                index
                for index in result.updates
                if parameters[index].name == target.object.name
            ]
            if len(indices) != 1:
                return None
            index = indices[0]
            if not isinstance(arguments[index], IdentifierNode):
                return None
            aggregate_names = {
                parameter.name
                for parameter, value in zip(parameters, values)
                if isinstance(value, _Aggregate)
            }
            if any(
                isinstance(node, IdentifierNode) and node.name in aggregate_names
                for node in generator.walk_ast(statement.value)
            ):
                return None
            original = values[index]
            field = str(target.member)
            stored = original.fields.get(field)
            if not isinstance(stored, _Value):
                return None
            values[index] = _Aggregate(
                original.source_type,
                {**original.fields, field: _Value(stored.source_type, [(0, 0)])},
            )
            shifted = analysis.invoke(expression, values, (), allow_updates=True)
            if shifted is None:
                return None
            delta = shifted.updates[index].fields.get(field)
            if not isinstance(delta, _Value):
                return None
            return {
                f"{arguments[index].name}.{field}": (
                    min(a for a, _ in delta.intervals),
                    max(b for _, b in delta.intervals),
                )
            }
        updates = {}

        def fields(name, value):
            if isinstance(value, _Aggregate):
                for member, item in value.fields.items():
                    fields(f"{name}.{member}", item)
            elif isinstance(value, _Vector):
                for index, item in enumerate(value.components):
                    fields(
                        generator.glsl_index_component_interval_key(name, index), item
                    )
            elif isinstance(value, _Value):
                updates[name] = (
                    min(a for a, _ in value.intervals),
                    max(b for _, b in value.intervals),
                )

        for index, value in result.updates.items():
            if not isinstance(arguments[index], IdentifierNode):
                return None
            fields(arguments[index].name, value)
        return updates
    finally:
        generator._glsl_return_bounds_active = False


def vector_return_intervals(generator, expression, vector_type, intervals, constants):
    value = _return_value(generator, expression, intervals, constants)
    if not isinstance(value, _Vector):
        return None
    value = _ReturnAnalysis(generator, intervals, constants).convert(value, vector_type)
    if value is None:
        return None
    return tuple(
        (min(a for a, _ in item.intervals), max(b for _, b in item.intervals))
        for item in value.components
    )
