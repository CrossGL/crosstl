"""Function signatures and indices in the GLSL generator: index value ranges, function graphs, rebased resources, parameter signatures, overloads and boolean reductions."""

import re

from ..ast import (
    BinaryOpNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    StructMemberNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..validation import expression_debug_name
from .array_utils import evaluate_literal_int_expression
from .glsl_exact_lookups import glsl_call_names_resolving_to
from .glsl_guarded_offsets import guarded_rebased_subscript
from .index_range_contracts import (
    INDEX_RANGE_CONSTANT,
    INDEX_RANGE_STATIC,
    IntegerRange,
    decide_index_narrowing,
    nonnegative_remainder_interval,
    truncating_division_interval,
)
from .opengl_values import (
    OpenGLMappedOverloadError,
    OpenGLRecursiveFunctionError,
)
from .stage_utils import function_stage_name


class OpenGLFunctionSignatureMixin:
    """Function signature members of ``GLSLCodeGen``."""

    def glsl_index_value_range(self, expression, _active=None, *, intervals=None):
        if expression is None:
            return None
        active = set() if _active is None else _active
        expression_id = id(expression)
        if expression_id in active:
            return None
        active.add(expression_id)
        try:
            literal = self.literal_int_value(
                expression,
                self.glsl_active_literal_int_constants(),
            )
            if isinstance(literal, int) and not isinstance(literal, bool):
                return IntegerRange.exact(
                    literal,
                    provenance=expression_debug_name(expression),
                )

            asserted = self.glsl_index_asserted_range(expression)
            if asserted is not None:
                return asserted

            flow_range = self.glsl_index_flow_range(expression, intervals)
            if flow_range is not None:
                return flow_range

            if isinstance(expression, MemberAccessNode):
                component_index = self.glsl_index_component_index(expression.member)
                if component_index is not None:
                    vector_value = self.glsl_integer_vector_literal_value(
                        expression.object
                    )
                    if vector_value is not None and component_index < len(vector_value):
                        return IntegerRange.exact(
                            vector_value[component_index],
                            provenance=expression_debug_name(expression),
                        )
                return self.glsl_integer_type_domain(expression)

            if isinstance(expression, UnaryOpNode):
                operator_name = self.map_operator(expression.op)
                operand_range = self.glsl_index_value_range(
                    expression.operand, active, intervals=intervals
                )
                if operator_name == "+":
                    return operand_range
                if operator_name == "-" and operand_range is not None:
                    candidate = IntegerRange(
                        -operand_range.maximum,
                        -operand_range.minimum,
                        INDEX_RANGE_STATIC,
                        expression_debug_name(expression),
                    )
                    domain = self.glsl_integer_type_domain(expression, allow_wide=True)
                    return (
                        candidate
                        if domain is None or candidate.is_within(domain)
                        else None
                    )
                if operator_name in {"++", "--"} or expression.op in {
                    "PRE_INCREMENT",
                    "PRE_DECREMENT",
                    "POST_INCREMENT",
                    "POST_DECREMENT",
                }:
                    return self.glsl_integer_type_domain(expression)
                return None

            if isinstance(expression, BinaryOpNode):
                operator_name = self.map_operator(expression.op)
                left = self.glsl_index_value_range(
                    expression.left, active, intervals=intervals
                )
                right = self.glsl_index_value_range(
                    expression.right, active, intervals=intervals
                )
                if operator_name == "%" and right is not None and right.minimum > 0:
                    left = left or self.glsl_integer_type_domain(
                        expression.left, allow_wide=True
                    )
                if operator_name == "&":
                    mask = None
                    if right is not None and right.is_exact and right.minimum >= 0:
                        mask = right.minimum
                    elif left is not None and left.is_exact and left.minimum >= 0:
                        mask = left.minimum
                    if mask is not None:
                        return IntegerRange(
                            0,
                            mask,
                            INDEX_RANGE_STATIC,
                            expression_debug_name(expression),
                        )
                if left is None or right is None:
                    return None
                if operator_name == "+":
                    bounds = (
                        left.minimum + right.minimum,
                        left.maximum + right.maximum,
                    )
                elif operator_name == "-":
                    bounds = (
                        left.minimum - right.maximum,
                        left.maximum - right.minimum,
                    )
                elif operator_name == "*":
                    products = (
                        left.minimum * right.minimum,
                        left.minimum * right.maximum,
                        left.maximum * right.minimum,
                        left.maximum * right.maximum,
                    )
                    bounds = (min(products), max(products))
                elif operator_name in {"/", "%"}:
                    operation = (
                        truncating_division_interval
                        if operator_name == "/"
                        else nonnegative_remainder_interval
                    )
                    bounds = operation(
                        (left.minimum, left.maximum),
                        (right.minimum, right.maximum),
                    )
                    if bounds is None:
                        return None
                elif operator_name in {"<<", ">>"}:
                    bounds = self.glsl_unsigned_shift_interval(
                        expression,
                        (left.minimum, left.maximum),
                        (right.minimum, right.maximum),
                    )
                    if bounds is None:
                        return None
                else:
                    return None
                candidate = IntegerRange(
                    *bounds,
                    INDEX_RANGE_STATIC,
                    expression_debug_name(expression),
                )
                domain = self.glsl_integer_type_domain(expression, allow_wide=True)
                return (
                    candidate if domain is None or candidate.is_within(domain) else None
                )

            if isinstance(expression, TernaryOpNode):
                true_range = self.glsl_index_value_range(
                    expression.true_expr, active, intervals=intervals
                )
                false_range = self.glsl_index_value_range(
                    expression.false_expr, active, intervals=intervals
                )
                if true_range is None or false_range is None:
                    return None
                return IntegerRange(
                    min(true_range.minimum, false_range.minimum),
                    max(true_range.maximum, false_range.maximum),
                    INDEX_RANGE_STATIC,
                    expression_debug_name(expression),
                )

            if isinstance(expression, (FunctionCallNode, ConstructorNode)):
                if isinstance(expression, FunctionCallNode):
                    function_name = self.function_call_name(expression)
                    arguments = list(expression.arguments or [])
                else:
                    function_name = self.type_name_string(expression.constructor_type)
                    arguments = list(expression.arguments or [])
                if function_name in {"min", "max"} and len(arguments) == 2:
                    left = self.glsl_index_value_range(
                        arguments[0], active, intervals=intervals
                    ) or self.glsl_integer_type_domain(arguments[0], allow_wide=True)
                    right = self.glsl_index_value_range(
                        arguments[1], active, intervals=intervals
                    ) or self.glsl_integer_type_domain(arguments[1], allow_wide=True)
                    if left is not None and right is not None:
                        if function_name == "min":
                            bounds = (
                                min(left.minimum, right.minimum),
                                min(left.maximum, right.maximum),
                            )
                        else:
                            bounds = (
                                max(left.minimum, right.minimum),
                                max(left.maximum, right.maximum),
                            )
                        return IntegerRange(
                            *bounds,
                            INDEX_RANGE_STATIC,
                            expression_debug_name(expression),
                        )
                if function_name == "clamp" and len(arguments) == 3:
                    lower = self.glsl_index_value_range(
                        arguments[1], active, intervals=intervals
                    )
                    upper = self.glsl_index_value_range(
                        arguments[2], active, intervals=intervals
                    )
                    if (
                        lower is not None
                        and lower.is_exact
                        and upper is not None
                        and upper.is_exact
                        and lower.minimum <= upper.minimum
                    ):
                        return IntegerRange(
                            lower.minimum,
                            upper.minimum,
                            INDEX_RANGE_STATIC,
                            expression_debug_name(expression),
                        )
                destination = self.glsl_value_type_info(function_name)
                if (
                    destination is not None
                    and destination["width"] == 1
                    and len(arguments) == 1
                ):
                    argument_range = self.glsl_index_value_range(
                        arguments[0], active, intervals=intervals
                    )
                    if argument_range is None:
                        return None
                    bits = (
                        destination["narrow"]["bits"]
                        if destination["narrow"] is not None
                        else destination["bits"]
                    )
                    if destination["family"] == "int":
                        destination_range = IntegerRange(
                            -(1 << (bits - 1)),
                            (1 << (bits - 1)) - 1,
                        )
                    elif destination["family"] == "uint":
                        destination_range = IntegerRange(0, (1 << bits) - 1)
                    else:
                        return None
                    if argument_range.is_within(destination_range):
                        return argument_range.with_status(
                            INDEX_RANGE_STATIC,
                            expression_debug_name(expression),
                        )
                return None

            return self.glsl_integer_type_domain(expression)
        finally:
            active.remove(expression_id)

    def glsl_indexed_extent(self, indexed_expr):
        indexed_type = self.type_name_string(
            self.glsl_source_expression_type(indexed_expr)
            or self.expression_result_type(indexed_expr)
        )
        outer_array = self.glsl_outer_array_type(indexed_type)
        if outer_array is not None:
            extent = evaluate_literal_int_expression(
                outer_array[0],
                self.glsl_active_literal_int_constants(),
            )
            if isinstance(extent, int) and not isinstance(extent, bool) and extent >= 0:
                return extent

        value_info = self.glsl_value_type_info(indexed_type)
        if value_info is not None and value_info["width"] > 1:
            return value_info["width"]

        mapped_type = self.map_type(indexed_type) if indexed_type else ""
        matrix_match = re.fullmatch(r"(?:d?mat)([234])(?:x[234])?", mapped_type)
        if matrix_match is not None:
            return int(matrix_match.group(1))

        indexed_name = self.expression_name(indexed_expr)
        extent = self.resource_variable_array_sizes.get(indexed_name)
        return extent if isinstance(extent, int) and extent >= 0 else None

    def glsl_target_index_literal(self, value, target_type):
        return f"{value}u" if target_type == "uint" else str(value)

    def glsl_index_type_error(
        self,
        index_expr,
        indexed_expr,
        index_type,
        target_index_type,
        reason,
        *,
        range_status=None,
        source_range=None,
        accepted_range=None,
    ):
        rendered_index_type = self.type_name_string(index_type) or "unknown"
        indexed_value = expression_debug_name(indexed_expr)
        target_profile = self.glsl_index_target_profile()
        raise self.GLSL_INDEX_RANGE_ERROR(
            f"{self.GLSL_TARGET_DISPLAY_NAME} cannot preserve subscript index type "
            f"'{rendered_index_type}' for '{indexed_value}': "
            f"{reason.replace('-', ' ')}",
            index_type=rendered_index_type,
            target_index_type=target_index_type,
            target_profile=target_profile.name,
            indexed_value=indexed_value,
            index_expression=expression_debug_name(index_expr),
            range_status=range_status,
            source_range=source_range,
            accepted_range=accepted_range,
            reason=reason,
            source_location=(
                getattr(index_expr, "source_location", None)
                or getattr(indexed_expr, "source_location", None)
            ),
        )

    def glsl_rebased_access_asserted_range(self, indexed_expr, index_expr):
        """Return an explicit complete-offset assertion for ``pointer[index]``.

        ``pointer[index]`` denotes ``*(pointer + index)``. Pointer-valued source
        expressions already use ``pointer + delta`` as their explicit range key
        for the complete logical offset, so the same key covers a subscript
        through a rebased storage pointer. Assertions on ``index`` alone still
        do not establish the sum.
        """
        pointer_name = expression_debug_name(indexed_expr)
        index_name = expression_debug_name(index_expr)
        if not pointer_name or not index_name:
            return None
        return self.glsl_index_asserted_range(f"{pointer_name} + {index_name}")

    def glsl_asserted_rebased_index_expression(
        self,
        rendered,
        index_expr,
        indexed_expr,
        index_type,
        index_info,
        source_bits,
        source_range,
        complete_range,
    ):
        """Render a rebased subscript whose complete offset is asserted.

        The caller-trusted range must lie in the signed target offset domain.
        The relative index is still added to an ``int`` logical-offset local, so
        it must be representable as ``int`` from its source type or from a
        proved or asserted range.
        """
        accepted = IntegerRange(0, (1 << 31) - 1)
        signed_int = IntegerRange(-(1 << 31), (1 << 31) - 1)
        operand = source_range
        if operand is None:
            operand = self.glsl_integer_type_domain(index_expr, allow_wide=True)
        if (
            not complete_range.is_within(accepted)
            or operand is None
            or not operand.is_within(signed_int)
        ):
            self.glsl_index_type_error(
                index_expr,
                indexed_expr,
                index_type,
                "int",
                "composed-pointer-index-range-unproven",
                range_status="unproven",
                source_range=complete_range,
                accepted_range=accepted,
            )
        if index_info["family"] == "int" and source_bits <= 32:
            return rendered
        if operand.is_exact and operand.status == INDEX_RANGE_CONSTANT:
            return self.glsl_target_index_literal(operand.minimum, "int")
        return f"int({rendered})"

    def glsl_guarded_rebased_index_expression(
        self, rendered, index_expr, indexed_expr, binding, index_info
    ):
        """Render a subscript whose dominating guards bound its complete offset."""
        if not guarded_rebased_subscript(
            self,
            getattr(self, "current_glsl_emitted_function", None),
            index_expr,
            indexed_expr,
            binding,
        ):
            return None
        bits = (
            index_info["narrow"]["bits"]
            if index_info["narrow"] is not None
            else index_info["bits"]
        )
        if index_info["family"] == "int" and bits <= 32:
            return rendered
        return f"int({rendered})"

    def glsl_index_expression(self, index_expr, indexed_expr=None, **options):
        """Render a subscript index, scoping assertions to the indexed storage."""
        previous = getattr(self, "current_glsl_indexed_storage", None)
        if isinstance(indexed_expr, IdentifierNode):
            storage = indexed_expr.name
        elif isinstance(indexed_expr, str):
            storage = indexed_expr
        else:
            storage = None
        self.current_glsl_indexed_storage = storage
        try:
            return self.glsl_storage_index_expression(
                index_expr, indexed_expr, **options
            )
        finally:
            self.current_glsl_indexed_storage = previous

    def glsl_storage_index_expression(
        self,
        index_expr,
        indexed_expr=None,
        *,
        indexed_extent=None,
        target_signed=None,
        assertion_expression=None,
    ):
        rendered = self.generate_expression_with_expected(index_expr, None)
        index_type = self.glsl_source_expression_type(index_expr)
        index_info = self.glsl_value_type_info(index_type)
        if index_info is None:
            return rendered

        if index_info["width"] != 1:
            self.glsl_index_type_error(
                index_expr,
                indexed_expr,
                index_type,
                None,
                "vector-index-unsupported",
                range_status="unproven",
            )
        if index_info["family"] not in {"int", "uint"}:
            self.glsl_index_type_error(
                index_expr,
                indexed_expr,
                index_type,
                None,
                "non-integer-index",
                range_status="unproven",
            )
        source_bits = (
            index_info["narrow"]["bits"]
            if index_info["narrow"] is not None
            else index_info["bits"]
        )
        source_range = (
            self.glsl_index_asserted_range(assertion_expression)
            if assertion_expression is not None
            else None
        )
        if source_range is None:
            source_range = self.glsl_index_value_range(index_expr)
        base_range = None
        if indexed_expr is not None and not isinstance(indexed_expr, str):
            binding = self.glsl_storage_pointer_binding(
                indexed_expr, self.glsl_storage_pointer_aliases()
            )
            if (
                binding is not None
                and binding.get("offset", 0) not in (0, "0", "0u")
                and (
                    source_bits > 32
                    or binding.get("wide_offset")
                    or source_range is not None
                    and source_range.is_exact
                    and source_range.minimum < 0
                )
            ):
                relative_range = source_range

                def asserted_rebased_subscript():
                    # Only an explicit assertion on the complete pointer
                    # expression replaces a missing composed-offset proof.
                    complete_range = self.glsl_rebased_access_asserted_range(
                        indexed_expr, index_expr
                    )
                    if complete_range is None:
                        return None
                    return self.glsl_asserted_rebased_index_expression(
                        rendered,
                        index_expr,
                        indexed_expr,
                        index_type,
                        index_info,
                        source_bits,
                        relative_range,
                        complete_range,
                    )

                intervals = getattr(indexed_expr, "_glsl_control_flow_intervals", {})
                base = self.glsl_checked_private_pointer_interval(
                    indexed_expr, intervals, self.glsl_active_literal_int_constants()
                )
                if base is None:
                    base = self.glsl_checked_private_pointer_interval(
                        binding.get("offset"),
                        intervals,
                        self.glsl_active_literal_int_constants(),
                    )
                accepted = IntegerRange(0, (1 << 31) - 1)
                if base is None:
                    guarded = self.glsl_guarded_rebased_index_expression(
                        rendered, index_expr, indexed_expr, binding, index_info
                    )
                    if guarded is not None:
                        return guarded
                    asserted = asserted_rebased_subscript()
                    if asserted is not None:
                        return asserted
                    self.glsl_index_type_error(
                        index_expr,
                        indexed_expr,
                        index_type,
                        "int",
                        "composed-pointer-index-range-unproven",
                        range_status="unproven",
                        source_range=None,
                        accepted_range=accepted,
                    )
                # A proved zero base is an ordinary subscript. Keep explicit
                # assertion provenance rather than treating it as a rebase proof.
                if base != (0, 0):
                    base_range = IntegerRange(*base)
                    if source_range is not None and (
                        source_range.minimum < 0
                        or any(
                            self.glsl_index_asserted_range(node) is not None
                            for node in self.walk_ast(index_expr)
                        )
                    ):
                        checked = self.glsl_checked_private_pointer_interval(
                            index_expr,
                            intervals,
                            self.glsl_active_literal_int_constants(),
                            require_arithmetic_types=True,
                        )
                        source_range = (
                            IntegerRange(*checked) if checked is not None else None
                        )
                target_signed = True
        decision = decide_index_narrowing(
            source_signed=index_info["family"] == "int",
            source_bits=source_bits,
            source_width=index_info["width"],
            profile=self.glsl_index_target_profile(),
            value_range=source_range,
            indexed_extent=(
                self.glsl_indexed_extent(indexed_expr)
                if indexed_extent is None
                else indexed_extent
            ),
            target_signed=target_signed,
            base_range=base_range,
        )
        target_type = decision.target_type.name if decision.target_type else None
        if not decision.accepted:
            if base_range is not None:
                asserted = asserted_rebased_subscript()
                if asserted is not None:
                    return asserted
            self.glsl_index_type_error(
                index_expr,
                indexed_expr,
                index_type,
                target_type,
                decision.reason,
                range_status=decision.range_status,
                source_range=decision.source_range,
                accepted_range=decision.accepted_range,
            )
        if decision.action == "identity":
            return rendered
        if (
            source_range is not None
            and source_range.is_exact
            and source_range.status == INDEX_RANGE_CONSTANT
        ):
            return self.glsl_target_index_literal(source_range.minimum, target_type)
        return f"{target_type}({rendered})"

    def glsl_resource_array_index_expression(self, expr):
        index_expr = getattr(expr, "index", getattr(expr, "index_expr", None))
        array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
        literal_index = self.literal_int_value(index_expr, self.literal_int_constants)
        array_type = self.glsl_source_expression_type(
            array_expr
        ) or self.expression_result_type(array_expr)
        if (
            isinstance(literal_index, int)
            and not isinstance(literal_index, bool)
            and literal_index < 0
            and (
                self.is_resource_array_access(expr) or self.is_sampler_type(array_type)
            )
        ):
            return "0"
        return self.glsl_index_expression(index_expr, array_expr)

    def is_resource_array_access(self, expr):
        array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
        array_name = self.expression_name(array_expr)
        if not array_name:
            return False
        if array_name in self.texture_variable_types:
            return True
        if array_name in self.current_texture_parameters:
            return True
        array_type = self.expression_result_type(array_expr)
        return self.is_resource_array_hint_type(array_type)

    def collect_function_parameter_types(self, functions):
        parameter_types = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            types = []
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                raw_type = getattr(param, "param_type", getattr(param, "vtype", None))
                types.append(self.type_name_string(raw_type))
            parameter_types[func_name] = types
        return parameter_types

    def collect_function_parameter_infos(self, functions):
        parameter_infos = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            parameter_infos[func_name] = self.glsl_function_parameter_infos(func)
        return parameter_infos

    def glsl_function_parameter_infos(self, function):
        infos = []
        for parameter in getattr(
            function, "parameters", getattr(function, "params", [])
        ):
            raw_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            infos.append(
                (
                    getattr(parameter, "name", None),
                    self.type_name_string(raw_type),
                    tuple(self.glsl_parameter_qualifiers(parameter))
                    + (
                        ("in",)
                        if "in" in (getattr(parameter, "qualifiers", []) or [])
                        else ()
                    ),
                )
            )
        return infos

    def prepare_glsl_mapped_function_overloads(self, functions, reserved_names=()):
        """Assign target names where distinct source overloads collapse in GLSL."""
        groups = {}
        for function in functions or []:
            function_name = getattr(function, "name", None)
            if not function_name or function_stage_name(function):
                continue
            groups.setdefault(function_name, []).append(function)

        self.glsl_function_overloads_by_name = groups
        self.glsl_function_target_names = {}

        for function_name, declarations in groups.items():
            target_name = self.glsl_module_identifier_name(function_name)
            if target_name == function_name:
                continue
            for declaration in declarations:
                self.glsl_function_target_names[id(declaration)] = target_name
            representative = next(
                (
                    declaration
                    for declaration in declarations
                    if getattr(declaration, "body", None) is not None
                ),
                declarations[0],
            )
            self.register_glsl_function_target_metadata(target_name, representative)

        for function_name, overloads in groups.items():
            mapped_groups = {}
            for function in overloads:
                target_key = self.glsl_mapped_function_signature_key(function)
                collision_key = tuple(parameter[0] for parameter in target_key[1])
                mapped_groups.setdefault(collision_key, []).append(function)

            for mapped_overloads in mapped_groups.values():
                source_groups = {}
                for function in mapped_overloads:
                    source_key = self.glsl_source_function_signature(function)
                    source_groups.setdefault(source_key, []).append(function)
                if len(source_groups) < 2:
                    continue

                if any(
                    self.glsl_function_has_resource_contract(function)
                    for function in mapped_overloads
                ):
                    candidates = [
                        self.glsl_function_candidate_signature(declarations[0])
                        for declarations in source_groups.values()
                    ]
                    raise OpenGLMappedOverloadError(
                        "OpenGL codegen cannot preserve mapped overloads with "
                        f"resource parameters for '{function_name}'; candidates "
                        f"are {', '.join(candidates)}",
                        function_name=function_name,
                        candidates=candidates,
                        reason="resource-parameter-collision",
                    )

                for source_key, declarations in source_groups.items():
                    suffix = self.glsl_function_overload_suffix(source_key)
                    base_name = f"{function_name}_{suffix}"
                    target_name = self.glsl_generated_module_identifier(
                        ("mapped-overload", function_name, source_key),
                        base_name,
                    )
                    for declaration in declarations:
                        self.glsl_function_target_names[id(declaration)] = target_name
                    representative = next(
                        (
                            declaration
                            for declaration in declarations
                            if getattr(declaration, "body", None) is not None
                        ),
                        declarations[0],
                    )
                    self.register_glsl_function_target_metadata(
                        target_name, representative
                    )

    def glsl_source_function_signature(self, function):
        return tuple(
            self.glsl_source_type_signature(
                getattr(parameter, "param_type", getattr(parameter, "vtype", None))
            )
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
        )

    def glsl_source_type_signature(self, value_type):
        type_name = self.type_name_string(value_type) or "void"
        resolved = self.resolve_glsl_type_alias(type_name)
        return re.sub(r"\s+", " ", str(resolved).strip())

    def glsl_overload_reserved_names(self, ast):
        names = set()
        for directive in getattr(ast, "preprocessors", []) or []:
            directive_name = getattr(directive, "directive", None)
            if directive_name != "define":
                continue
            content = str(getattr(directive, "content", "") or "")
            match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", content)
            if match is not None:
                names.add(match.group(1))
        for node in self.walk_ast(ast):
            is_declaration = (
                isinstance(node, (StructMemberNode, StructNode, VariableNode))
                or hasattr(node, "parameters")
                or hasattr(node, "param_type")
                or hasattr(node, "const_type")
            )
            name = getattr(node, "name", None)
            if is_declaration and isinstance(name, str) and name:
                names.add(name)
        return names

    def glsl_function_has_resource_contract(self, function):
        for parameter in (
            getattr(function, "parameters", getattr(function, "params", [])) or []
        ):
            raw_type = self.type_name_string(
                getattr(parameter, "param_type", getattr(parameter, "vtype", None))
            )
            buffer_type, _size = self.structured_buffer_array_base_type(raw_type)
            if (
                self.canonical_resource_type(raw_type) is not None
                or self.is_storage_image_type(raw_type)
                or self.is_structured_buffer_type(raw_type)
                or (
                    buffer_type is not None
                    and self.is_structured_buffer_type(buffer_type)
                )
            ):
                return True
        return False

    def glsl_mapped_function_signature_key(self, function):
        """Return the complete mapped signature used to diagnose collisions."""
        return_type = self.map_type(
            self.type_name_string(getattr(function, "return_type", None)) or "void"
        )
        parameters = []
        function_name = getattr(function, "name", None)
        unsupported_indices = self.unsupported_structured_buffer_array_functions.get(
            function_name, {}
        ).get("indices", set())
        for index, parameter in enumerate(
            getattr(function, "parameters", getattr(function, "params", [])) or []
        ):
            if index in unsupported_indices:
                continue
            raw_type = self.type_name_string(
                getattr(parameter, "param_type", getattr(parameter, "vtype", None))
            )
            if self.is_sampler_type(
                raw_type
            ) and not self.should_preserve_sampler_parameter(
                function_name, getattr(parameter, "name", None)
            ):
                continue

            buffer_array = self.structured_buffer_array_parameter_info(
                raw_type, getattr(parameter, "name", None), function_name
            )
            if buffer_array is not None:
                resource_kind = self.resource_base_type(buffer_array["base_type"])
                for _array_index in range(buffer_array["count"]):
                    parameters.append(
                        (f"{buffer_array['element_type']}[]", resource_kind, ())
                    )
                if self.structured_buffer_requires_counter(buffer_array["base_type"]):
                    parameters.extend(
                        ("uint[]", "counter", ("inout",))
                        for _array_index in range(buffer_array["count"])
                    )
                continue

            mapped_type = self.map_resource_parameter_type_with_hint(
                raw_type, parameter, function_name
            )
            resource_kind = self.canonical_resource_type(raw_type) or ""
            access_qualifiers = set(self.glsl_parameter_qualifiers(parameter))
            memory_qualifiers = self.resource_memory_qualifiers(parameter)
            if memory_qualifiers:
                access_qualifiers.update(memory_qualifiers.split())
            parameters.append(
                (
                    re.sub(r"\s+", " ", str(mapped_type).strip()),
                    resource_kind,
                    tuple(sorted(access_qualifiers)),
                )
            )
            if self.structured_buffer_requires_counter(raw_type):
                parameters.append(("uint[]", "counter", ("inout",)))
        return return_type, tuple(parameters)

    def glsl_function_overload_suffix(self, source_signature):
        parts = []
        for type_name in source_signature:
            part = re.sub(r"[^0-9A-Za-z]+", "_", type_name).strip("_")
            parts.append(part or "value")
        return "_".join(parts) or "void"

    def register_glsl_function_target_metadata(self, target_name, function):
        parameters = (
            getattr(function, "parameters", getattr(function, "params", [])) or []
        )
        self.function_return_types[target_name] = self.type_name_string(
            getattr(function, "return_type", "void")
        )
        self.function_parameter_names[target_name] = [
            getattr(parameter, "name", None) for parameter in parameters
        ]
        self.function_parameter_types[target_name] = [
            self.type_name_string(
                getattr(parameter, "param_type", getattr(parameter, "vtype", None))
            )
            for parameter in parameters
        ]
        self.function_parameter_infos[target_name] = self.glsl_function_parameter_infos(
            function
        )
        self.function_definitions[target_name] = function
        source_name = getattr(function, "name", None)
        sampler_indices = self.collect_function_sampler_parameter_indices(
            [function]
        ).get(source_name)
        if sampler_indices is not None:
            self.function_sampler_parameter_indices[target_name] = sampler_indices
        if source_name in self.function_fragment_only_requirements:
            self.function_fragment_only_requirements[target_name] = (
                self.function_fragment_only_requirements[source_name]
            )

    def glsl_function_declaration_name(self, function):
        target_name = self.glsl_function_target_names.get(id(function))
        if target_name is not None:
            return target_name
        return self.glsl_module_identifier_name(getattr(function, "name", ""))

    def register_glsl_function_graph_scope(self, function, *, stage_context=None):
        if function is not None:
            self.glsl_function_stage_contexts[id(function)] = stage_context

    def glsl_function_graph_key(
        self,
        function,
        target_name,
        *,
        stage_context=None,
    ):
        if function is None:
            source_signature = ()
        else:
            source_signature = self.glsl_source_function_signature(function)
            if id(function) in self.glsl_function_stage_contexts:
                stage_context = self.glsl_function_stage_contexts[id(function)]
        return (stage_context or "", target_name, source_signature)

    @staticmethod
    def glsl_function_graph_name(function_key):
        return function_key[1]

    def record_glsl_function_call(
        self,
        target_name,
        source_location=None,
        *,
        target_function=None,
        source_target_name=None,
    ):
        source_key = self.current_glsl_function_key
        if not source_key or not target_name:
            return
        target_key = self.glsl_function_graph_key(
            target_function,
            target_name,
            stage_context=source_key[0] or None,
        )
        self.glsl_function_call_graph.setdefault(source_key, set()).add(target_key)
        self.glsl_function_call_locations.setdefault(
            (source_key, target_key),
            source_location,
        )
        source_function_name = self.current_glsl_source_function_name
        if source_function_name and source_target_name:
            self.glsl_function_call_source_edges.setdefault(
                (source_key, target_key),
                (source_function_name, source_target_name),
            )

    def validate_glsl_function_call_graph(self):
        emitted_keys = set(self.glsl_function_call_graph)
        graph = {
            source_key: {
                target_key for target_key in target_keys if target_key in emitted_keys
            }
            for source_key, target_keys in self.glsl_function_call_graph.items()
        }
        visited = set()
        active_indices = {}
        stack = []

        def visit(source_key):
            active_indices[source_key] = len(stack)
            stack.append(source_key)
            for target_key in sorted(graph.get(source_key, ())):
                if target_key in active_indices:
                    cycle_keys = stack[active_indices[target_key] :] + [target_key]
                    cycle = [
                        self.glsl_function_graph_name(function_key)
                        for function_key in cycle_keys
                    ]
                    cycle_text = " -> ".join(cycle)
                    raise OpenGLRecursiveFunctionError(
                        "OpenGL does not support recursive function calls; "
                        f"detected cycle '{cycle_text}'. Refactor the cycle into "
                        "an iterative helper before GLSL generation.",
                        cycle=cycle,
                        source_edge=self.glsl_function_call_source_edges.get(
                            (source_key, target_key)
                        ),
                        source_location=self.glsl_function_call_locations.get(
                            (source_key, target_key)
                        ),
                    )
                if target_key not in visited:
                    visit(target_key)
            stack.pop()
            active_indices.pop(source_key, None)
            visited.add(source_key)

        for source_key in sorted(graph):
            if source_key not in visited:
                visit(source_key)

    def glsl_function_visible_from_current_stage(self, function):
        """Whether a call from the function being emitted can name ``function``.

        A helper declared inside one stage block is emitted only in that stage, so
        a call from another stage never resolves to it. A program-scope function is
        emitted in every stage, so it sees only program-scope helpers: a call from
        it never resolves to a stage-local helper either, and names the built-in
        when one of that name exists.
        """
        owner = getattr(self, "glsl_stage_local_function_owners", {}).get(id(function))
        if owner is None:
            return True
        current_stage = getattr(self, "current_glsl_function_stage", None)
        return current_stage is not None and owner == current_stage

    def resolve_glsl_function_overload(
        self, function_name, arguments, *, call_node=None
    ):
        candidates = [
            function
            for function in self.glsl_function_overloads_by_name.get(function_name, ())
            if len(
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
            == len(arguments)
            and self.glsl_function_visible_from_current_stage(function)
        ]
        if not candidates:
            return None

        unique_candidates = {}
        for function in candidates:
            source_key = self.glsl_source_function_signature(function)
            previous = unique_candidates.get(source_key)
            if previous is None or (
                getattr(previous, "body", None) is None
                and getattr(function, "body", None) is not None
            ):
                unique_candidates[source_key] = function
        candidates = list(unique_candidates.values())
        if len(candidates) == 1:
            return candidates[0]

        argument_types = [
            self.glsl_source_expression_type(argument) for argument in arguments
        ]
        scored = []
        for function in candidates:
            parameters = (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
            score = 0
            compatible = True
            for argument, argument_type, parameter in zip(
                arguments,
                argument_types,
                parameters,
            ):
                if argument_type is None and not (
                    isinstance(argument, UnaryOpNode)
                    and self.map_operator(argument.op) == "&"
                ):
                    continue
                match_score = self.glsl_function_argument_match_score(
                    argument,
                    argument_type,
                    parameter,
                )
                if match_score is None:
                    compatible = False
                    break
                score += match_score
            if not compatible:
                continue
            scored.append((score, function))

        if scored:
            highest_score = max(score for score, _function in scored)
            winners = [function for score, function in scored if score == highest_score]
            if len(winners) == 1:
                return winners[0]
        else:
            winners = candidates

        if self.is_plain_metal_simd_group_helper_name(function_name) and not any(
            id(function) in self.glsl_function_target_names for function in candidates
        ):
            return None

        argument_names = [
            (
                self.glsl_source_type_signature(argument_type)
                if argument_type is not None
                else "unknown"
            )
            for argument_type in argument_types
        ]
        candidate_names = [
            self.glsl_function_candidate_signature(function) for function in winners
        ]
        raise OpenGLMappedOverloadError(
            "OpenGL codegen cannot preserve source overload binding for "
            f"'{function_name}({', '.join(argument_names)})' after GLSL type mapping; "
            f"candidates are {', '.join(candidate_names)}",
            function_name=function_name,
            argument_types=argument_names,
            candidates=candidate_names,
            reason="call-binding-ambiguous",
            source_location=getattr(call_node, "source_location", None),
        )

    def glsl_plain_simd_call_result_type(self, function_name, arguments):
        if (
            not self.is_plain_metal_simd_group_helper_name(function_name)
            or not arguments
        ):
            return None
        argument_types = [
            self.glsl_source_expression_type(argument) for argument in arguments
        ]
        if any(argument_type is None for argument_type in argument_types):
            return self.glsl_source_expression_type(arguments[0])
        argument_signature = tuple(
            self.glsl_source_type_signature(argument_type)
            for argument_type in argument_types
        )
        exact_matches = [
            function
            for function in self.glsl_function_overloads_by_name.get(function_name, ())
            if self.glsl_source_function_signature(function) == argument_signature
        ]
        return_types = {
            self.type_name_string(getattr(function, "return_type", None))
            for function in exact_matches
        }
        return_types.discard(None)
        if len(return_types) == 1:
            return next(iter(return_types))
        return self.glsl_source_expression_type(arguments[0])

    def glsl_componentwise_unary_result_type(self, function_name, arguments):
        if len(arguments) != 1:
            return None
        if function_name not in glsl_call_names_resolving_to(
            self, "componentwise-unary", self.GLSL_COMPONENTWISE_UNARY_FUNCTIONS
        ):
            return None
        return self.glsl_source_expression_type(arguments[0])

    def glsl_componentwise_boolean_result_type(
        self,
        function_name,
        arguments,
        *,
        source_type=False,
    ):
        if len(arguments) != 1:
            return None
        if function_name not in glsl_call_names_resolving_to(
            self, "componentwise-boolean", self.GLSL_COMPONENTWISE_BOOLEAN_FUNCTIONS
        ):
            return None
        argument_type = (
            self.glsl_source_expression_type(arguments[0])
            if source_type
            else self.expression_result_type(arguments[0])
        )
        argument = self.glsl_value_type_info(argument_type)
        if argument is None or argument["family"] != "float":
            return None
        if argument["width"] == 1:
            return "bool"
        return f"bvec{argument['width']}"

    def glsl_boolean_reduction_result_type(self, function_name, arguments):
        if len(arguments) != 1:
            return None
        if function_name not in glsl_call_names_resolving_to(
            self, "boolean-reduction", self.GLSL_BOOLEAN_REDUCTION_FUNCTIONS
        ):
            return None
        return "bool"
