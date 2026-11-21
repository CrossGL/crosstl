"""Private pointer intervals in the GLSL generator: symbolic offsets, integer keys, conditions, arrays and storage intervals."""

from ..arithmetic_conversions import (
    ArithmeticScalarKind,
    target_arithmetic_type,
)
from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PointerReinterpretNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..metadata_preconditions import MAX_METADATA_PRECONDITIONS
from ..validation import expression_debug_name
from .congruent_intervals import congruent_ring_interval
from .index_range_contracts import (
    nonnegative_remainder_interval,
    truncating_division_interval,
)
from .integer_intervals import convert_integer_intervals
from .private_array_rows import array_row_view
from .private_member_views import (
    glsl_member_view_root,
    member_view_chain,
)
from .scalar_return_bounds import scalar_return_interval


class OpenGLPrivatePointerIntervalMixin:
    """Private pointer interval members of ``GLSLCodeGen``."""

    def glsl_private_pointer_symbolic_offset(
        self,
        expression,
        pointer_roots,
        forms,
        constants,
    ):
        if isinstance(expression, str):
            return (expression, (0, ())) if expression in pointer_roots else None
        if isinstance(expression, (IdentifierNode, VariableNode)):
            name = getattr(expression, "name", None)
            return (name, (0, ())) if name in pointer_roots else None
        if isinstance(expression, UnaryOpNode) and expression.op == "&":
            if not isinstance(expression.operand, ArrayAccessNode):
                return None
            binding = self.glsl_private_pointer_symbolic_offset(
                expression.operand.array,
                pointer_roots,
                forms,
                constants,
            )
            delta = self.glsl_private_pointer_affine_form(
                expression.operand.index, forms, constants
            )
            if binding is None or delta is None:
                return None
            return (
                binding[0],
                self.glsl_private_pointer_combine_affine_forms(binding[1], delta),
            )
        if not isinstance(expression, BinaryOpNode) or expression.op not in {"+", "-"}:
            return None
        binding = self.glsl_private_pointer_symbolic_offset(
            expression.left,
            pointer_roots,
            forms,
            constants,
        )
        delta_expression = expression.right
        operator = expression.op
        if binding is None and expression.op == "+":
            binding = self.glsl_private_pointer_symbolic_offset(
                expression.right,
                pointer_roots,
                forms,
                constants,
            )
            delta_expression = expression.left
        delta = self.glsl_private_pointer_affine_form(
            delta_expression, forms, constants
        )
        if binding is None or delta is None:
            return None
        return (
            binding[0],
            self.glsl_private_pointer_combine_affine_forms(
                binding[1], delta, -1 if operator == "-" else 1
            ),
        )

    def glsl_private_pointer_correlated_loop_contract(
        self,
        node,
        intervals,
        value_keys,
        constants,
        active_loops,
    ):
        init = getattr(node, "init", None)
        loop_name = getattr(init, "name", None)
        start = intervals.get(loop_name)
        if not loop_name or start is None or start[0] != start[1]:
            return None
        condition = getattr(node, "condition", None)
        if not isinstance(condition, BinaryOpNode):
            return None
        left_name = self.expression_name(condition.left)
        right_name = self.expression_name(condition.right)
        operator = condition.op
        if left_name == loop_name:
            bound_expression = condition.right
        elif right_name == loop_name:
            bound_expression = condition.left
            operator = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(operator)
        else:
            return None
        update = getattr(node, "update", None)
        if not (
            isinstance(update, UnaryOpNode)
            and self.expression_name(update.operand) == loop_name
            and update.op == "++"
            and operator in {"<", "<="}
        ):
            return None
        bound_key = self.glsl_private_pointer_contract_expression_key(
            bound_expression, value_keys, constants
        )
        if bound_key is None:
            return None
        interval_name, interval = self.glsl_private_pointer_loop_interval(
            node, intervals, constants
        )
        if interval_name != loop_name:
            interval = None
        parent_keys = tuple(loop["key"] for loop in active_loops.values())
        key = (loop_name, start[0], operator, bound_key, 1, parent_keys)
        symbol = repr(("loop", key))
        return {
            "key": key,
            "symbol": symbol,
            "interval": interval,
            "lower_bound": start[0],
        }

    def glsl_private_pointer_offset_is_nonnegative(self, footprint):
        constant, terms = footprint["form"]
        minimum = constant
        for symbol, coefficient in terms:
            loop = footprint["loops"].get(symbol)
            if loop is None or coefficient < 0 or loop["lower_bound"] < 0:
                return False
            minimum += coefficient * loop["lower_bound"]
        return minimum >= 0

    def glsl_private_pointer_access_covered_span(self, footprint, access):
        if (
            access["root"] != footprint["root"]
            or access["predicates"] != footprint["predicates"]
        ):
            return None
        for symbol, _coefficient in footprint["form"][1]:
            loop = footprint["loops"].get(symbol)
            if (
                loop is not None
                and access["loops"].get(symbol, {}).get("key") != loop["key"]
            ):
                return None
        difference = self.glsl_private_pointer_combine_affine_forms(
            access["form"], footprint["form"], -1
        )
        if difference == (0, ()):
            return 1
        if difference is None or difference[0] != 0 or len(difference[1]) != 1:
            return None
        coverage_symbol, coefficient = difference[1][0]
        coverage_loop = access["loops"].get(coverage_symbol)
        if (
            coefficient != 1
            or coverage_symbol in dict(footprint["form"][1])
            or coverage_loop is None
            or coverage_loop["interval"] is None
            or coverage_loop["interval"][0] != 0
        ):
            return None
        return coverage_loop["interval"][1] + 1

    def glsl_private_pointer_expression_has_side_effects(self, expression):
        if expression is None or isinstance(expression, (str, int, float, bool)):
            return False
        if isinstance(expression, AssignmentNode):
            return True
        if isinstance(expression, UnaryOpNode) and expression.op in {"++", "--"}:
            return True
        if isinstance(expression, FunctionCallNode):
            function_name = self.function_call_name(expression)
            if function_name not in {
                "int",
                "uint",
                "short",
                "ushort",
                "int16",
                "uint16",
                "int32_t",
                "uint32_t",
                "size_t",
                "ptrdiff_t",
            }:
                return True
        if not hasattr(expression, "__dict__"):
            return False
        return any(
            self.glsl_private_pointer_expression_has_side_effects(child)
            for key, child in vars(expression).items()
            if key not in {"parent", "annotations", "array", "index", "args", "name"}
        )

    def glsl_private_pointer_condition_value(self, expression, intervals, constants):
        if self.glsl_private_pointer_expression_has_side_effects(expression):
            return None

        value = self.literal_int_value(expression, constants)
        if value is not None:
            return value != 0

        # Bounds intervals use unbounded arithmetic and cannot prove branch
        # outcomes across source-width overflow.
        if isinstance(expression, UnaryOpNode) and expression.op == "!":
            operand = self.glsl_private_pointer_condition_value(
                expression.operand, intervals, constants
            )
            return None if operand is None else not operand

        if isinstance(expression, BinaryOpNode):
            operator = self.map_operator(expression.op)
            if operator in {"&&", "and", "||", "or"}:
                left = self.glsl_private_pointer_condition_value(
                    expression.left, intervals, constants
                )
                if operator in {"&&", "and"} and left is False:
                    return False
                if operator in {"||", "or"} and left is True:
                    return True
                right = self.glsl_private_pointer_condition_value(
                    expression.right, intervals, constants
                )
                if operator in {"&&", "and"}:
                    if right is False:
                        return False
                    return True if left is True and right is True else None
                if right is True:
                    return True
                return False if left is False and right is False else None
        return None

    def glsl_guard_addend_operand(self, candidate, lower, upper, constants):
        """Move constant addends of a compared operand to its bounds."""
        while isinstance(candidate, BinaryOpNode) and self.map_operator(
            candidate.op
        ) in {"+", "-"}:
            operator = self.map_operator(candidate.op)
            right = self.literal_int_value(candidate.right, constants)
            left = self.literal_int_value(candidate.left, constants)
            if isinstance(right, int) and not isinstance(right, bool):
                delta = right if operator == "+" else -right
                candidate = candidate.left
            elif (
                operator == "+" and isinstance(left, int) and not isinstance(left, bool)
            ):
                delta = left
                candidate = candidate.right
            else:
                break
            lower, upper = lower - delta, upper - delta
        return candidate, lower, upper

    def glsl_private_pointer_refined_condition_intervals(
        self,
        expression,
        intervals,
        constants,
        expected,
        *,
        excluded_names=None,
        check_source_types=False,
    ):
        refined = dict(intervals)
        if self.glsl_private_pointer_expression_has_side_effects(expression):
            return refined
        excluded_names = set(excluded_names or ())
        if self.glsl_interval_dependency_names(expression) & excluded_names:
            return refined
        condition = self.glsl_private_pointer_condition_value(
            expression,
            refined,
            constants,
        )
        if condition is not None:
            return refined if condition is expected else None

        if (
            isinstance(expression, UnaryOpNode)
            and self.map_operator(expression.op) == "!"
        ):
            return self.glsl_private_pointer_refined_condition_intervals(
                expression.operand,
                refined,
                constants,
                not expected,
                excluded_names=excluded_names,
                check_source_types=check_source_types,
            )

        if not isinstance(expression, BinaryOpNode):
            return refined
        operator = self.map_operator(expression.op)
        if operator in {"&&", "and"}:
            if not expected:
                left = self.glsl_private_pointer_condition_value(
                    expression.left,
                    refined,
                    constants,
                )
                if left is True:
                    return self.glsl_private_pointer_refined_condition_intervals(
                        expression.right,
                        refined,
                        constants,
                        False,
                        excluded_names=excluded_names,
                        check_source_types=check_source_types,
                    )
                right = self.glsl_private_pointer_condition_value(
                    expression.right,
                    refined,
                    constants,
                )
                if right is True:
                    return self.glsl_private_pointer_refined_condition_intervals(
                        expression.left,
                        refined,
                        constants,
                        False,
                        excluded_names=excluded_names,
                        check_source_types=check_source_types,
                    )
                return refined
            refined = self.glsl_private_pointer_refined_condition_intervals(
                expression.left,
                refined,
                constants,
                True,
                excluded_names=excluded_names,
                check_source_types=check_source_types,
            )
            if refined is None:
                return None
            return self.glsl_private_pointer_refined_condition_intervals(
                expression.right,
                refined,
                constants,
                True,
                excluded_names=excluded_names,
                check_source_types=check_source_types,
            )
        if operator in {"||", "or"}:
            if expected:
                left = self.glsl_private_pointer_condition_value(
                    expression.left,
                    refined,
                    constants,
                )
                if left is False:
                    return self.glsl_private_pointer_refined_condition_intervals(
                        expression.right,
                        refined,
                        constants,
                        True,
                        excluded_names=excluded_names,
                        check_source_types=check_source_types,
                    )
                right = self.glsl_private_pointer_condition_value(
                    expression.right,
                    refined,
                    constants,
                )
                if right is False:
                    return self.glsl_private_pointer_refined_condition_intervals(
                        expression.left,
                        refined,
                        constants,
                        True,
                        excluded_names=excluded_names,
                        check_source_types=check_source_types,
                    )
                return refined
            refined = self.glsl_private_pointer_refined_condition_intervals(
                expression.left,
                refined,
                constants,
                False,
                excluded_names=excluded_names,
                check_source_types=check_source_types,
            )
            if refined is None:
                return None
            return self.glsl_private_pointer_refined_condition_intervals(
                expression.right,
                refined,
                constants,
                False,
                excluded_names=excluded_names,
                check_source_types=check_source_types,
            )
        if operator not in {"==", "!=", "<", "<=", ">", ">="}:
            return refined

        if not expected:
            operator = {
                "==": "!=",
                "!=": "==",
                "<": ">=",
                "<=": ">",
                ">": "<=",
                ">=": "<",
            }[operator]
        left_expression = expression.left
        right_expression = expression.right
        left = self.glsl_private_pointer_interval(
            left_expression,
            refined,
            constants,
        )
        right = self.glsl_private_pointer_interval(
            right_expression,
            refined,
            constants,
        )
        if left is None or right is None:
            return refined

        if check_source_types:
            plan = self.glsl_arithmetic_conversion_plan(
                self.glsl_source_expression_type(left_expression),
                self.glsl_source_expression_type(right_expression),
                operator,
            )
            if plan is None:
                return refined
            # Mathematical comparison bounds are valid only when the source
            # promotions preserve both operands, including intermediate values.
            for operand, bounds, promoted in (
                (left_expression, left, plan.left_target_type),
                (right_expression, right, plan.right_target_type),
            ):
                checked = self.glsl_checked_private_pointer_interval(
                    operand, refined, constants, require_arithmetic_types=True
                )
                info = target_arithmetic_type(promoted)
                if (
                    checked != bounds
                    or info is None
                    or info.lanes != 1
                    or info.kind not in {
                        ArithmeticScalarKind.SIGNED_INTEGER,
                        ArithmeticScalarKind.UNSIGNED_INTEGER,
                    }
                ):
                    return refined
                signed = info.kind == ArithmeticScalarKind.SIGNED_INTEGER
                minimum = -(1 << (info.bits - 1)) if signed else 0
                maximum = (1 << (info.bits - int(signed))) - 1
                if not minimum <= bounds[0] <= bounds[1] <= maximum:
                    return refined

        def narrow(candidate, lower, upper):
            if check_source_types:
                # The operands were checked not to wrap, so a bound on
                # ``x + c`` or ``x - c`` is a bound on ``x``.
                candidate, lower, upper = self.glsl_guard_addend_operand(
                    candidate, lower, upper, constants
                )
            name = self.glsl_private_pointer_interval_key(candidate, refined, constants)
            if name not in refined:
                return True
            current = refined[name]
            replacement = max(current[0], lower), min(current[1], upper)
            if replacement[0] > replacement[1]:
                return False
            refined[name] = replacement
            return True

        if operator == "==":
            overlap = max(left[0], right[0]), min(left[1], right[1])
            if overlap[0] > overlap[1]:
                return None
            if not narrow(left_expression, *overlap) or not narrow(
                right_expression,
                *overlap,
            ):
                return None
            return refined
        if operator == "!=":
            if left[0] == left[1] == right[0] == right[1]:
                return None
            left_name = self.glsl_private_pointer_interval_key(
                left_expression, refined, constants
            )
            right_name = self.glsl_private_pointer_interval_key(
                right_expression, refined, constants
            )
            if right[0] == right[1] and left_name in refined:
                excluded = right[0]
                lower, upper = refined[left_name]
                if lower == excluded:
                    lower += 1
                elif upper == excluded:
                    upper -= 1
                if not narrow(left_expression, lower, upper):
                    return None
            if left[0] == left[1] and right_name in refined:
                excluded = left[0]
                lower, upper = refined[right_name]
                if lower == excluded:
                    lower += 1
                elif upper == excluded:
                    upper -= 1
                if not narrow(right_expression, lower, upper):
                    return None
            return refined

        if operator == ">":
            left_expression, right_expression = right_expression, left_expression
            left, right = right, left
            operator = "<"
        elif operator == ">=":
            left_expression, right_expression = right_expression, left_expression
            left, right = right, left
            operator = "<="
        if operator == "<":
            if not narrow(left_expression, left[0], right[1] - 1) or not narrow(
                right_expression,
                left[0] + 1,
                right[1],
            ):
                return None
        elif operator == "<=":
            if not narrow(left_expression, left[0], right[1]) or not narrow(
                right_expression,
                left[0],
                right[1],
            ):
                return None
        return refined

    @staticmethod
    def glsl_private_pointer_array_row(expression, local_arrays, excluded_names):
        """The row of a fixed multidimensional local array ``expression`` selects, or None."""

        def shape_of(name):
            info = local_arrays.get(name)
            if name in excluded_names or info is None or info.get("extent") is None:
                return None
            return info.get("rows")

        return array_row_view(expression, shape_of)

    def glsl_private_pointer_static_binding(
        self,
        expression,
        parameter_element_types,
        local_arrays,
        constants,
        lexical_bindings=None,
        intervals=None,
    ):
        if isinstance(expression, PointerReinterpretNode):
            binding = self.glsl_private_pointer_static_binding(
                expression.expression,
                parameter_element_types,
                local_arrays,
                constants,
                lexical_bindings,
                intervals,
            )
            if binding is None or binding.get("root_kind") is None:
                return None
            return self.glsl_private_pointer_reinterpret_binding(
                expression,
                binding,
                constants,
            )

        if isinstance(expression, (str, IdentifierNode, VariableNode)):
            name = (
                expression
                if isinstance(expression, str)
                else getattr(expression, "name", None)
            )
            if lexical_bindings is not None and name in lexical_bindings:
                return dict(lexical_bindings[name])
            if name in parameter_element_types:
                return {
                    "root_kind": "parameter",
                    "root": name,
                    "offset": 0,
                    "offset_interval": (0, 0),
                    "side_effecting": False,
                    "element_type": parameter_element_types[name],
                }
            if name in local_arrays:
                return {
                    "root_kind": "local",
                    "root": name,
                    "extent": local_arrays[name]["extent"],
                    "offset": 0,
                    "offset_interval": (0, 0),
                    "side_effecting": False,
                    "element_type": local_arrays[name]["element_type"],
                }
            return None

        if isinstance(expression, ArrayAccessNode):
            row = self.glsl_private_pointer_array_row(
                expression,
                local_arrays,
                set(parameter_element_types)
                | {
                    name
                    for name, binding in (lexical_bindings or {}).items()
                    if binding.get("root_kind") != "local"
                    or binding.get("root") != name
                },
            )
            if row is None:
                return None
            root_name, extent, element_type, indices = row
            return {
                "root_kind": "local",
                "root": root_name,
                "extent": extent,
                "offset": 0,
                "offset_interval": (0, 0),
                "side_effecting": any(
                    self.glsl_private_pointer_expression_has_side_effects(index)
                    for index in indices
                ),
                "element_type": element_type,
                "row_view": True,
            }

        if isinstance(expression, UnaryOpNode) and expression.op == "&":
            operand = expression.operand
            if isinstance(operand, (str, IdentifierNode, VariableNode)):
                binding = self.glsl_private_pointer_static_binding(
                    operand,
                    parameter_element_types,
                    local_arrays,
                    constants,
                    lexical_bindings,
                    intervals,
                )
                if binding is not None and binding.get("root_kind") == "scalar_value":
                    binding["root_kind"] = "scalar"
                    return binding
            if isinstance(operand, MemberAccessNode):
                root, _members = member_view_chain(operand)
                return glsl_member_view_root(
                    self,
                    operand,
                    (lexical_bindings or {}).get(root),
                )
            if not isinstance(operand, ArrayAccessNode):
                return None
            binding = self.glsl_private_pointer_static_binding(
                operand.array,
                parameter_element_types,
                local_arrays,
                constants,
                lexical_bindings,
                intervals,
            )
            if binding is None:
                return None
            delta_expression = operand.index
            operator = "+"
        elif isinstance(expression, BinaryOpNode) and expression.op in {"+", "-"}:
            binding = self.glsl_private_pointer_static_binding(
                expression.left,
                parameter_element_types,
                local_arrays,
                constants,
                lexical_bindings,
                intervals,
            )
            delta_expression = expression.right
            operator = expression.op
            if binding is None and expression.op == "+":
                binding = self.glsl_private_pointer_static_binding(
                    expression.right,
                    parameter_element_types,
                    local_arrays,
                    constants,
                    lexical_bindings,
                    intervals,
                )
                delta_expression = expression.left
            if binding is None:
                return None
        else:
            return None

        delta = self.literal_int_value(delta_expression, constants)
        delta_interval = (
            (delta, delta)
            if delta is not None
            else self.glsl_private_pointer_affine_interval(
                delta_expression, intervals or {}, constants
            )
        )
        if operator == "-":
            if delta is not None:
                delta = -delta
            if delta_interval is not None:
                delta_interval = (-delta_interval[1], -delta_interval[0])
        current = binding.get("offset")
        binding["offset"] = (
            None if current is None or delta is None else current + delta
        )
        current_interval = binding.get("offset_interval")
        if current_interval is None and isinstance(current, int):
            current_interval = (current, current)
        binding["offset_interval"] = (
            None
            if current_interval is None or delta_interval is None
            else (
                current_interval[0] + delta_interval[0],
                current_interval[1] + delta_interval[1],
            )
        )
        binding["side_effecting"] = binding.get(
            "side_effecting", False
        ) or self.glsl_private_pointer_expression_has_side_effects(delta_expression)
        return binding

    def glsl_private_pointer_builtin_range_interval(
        self,
        function_name,
        arguments,
        intervals,
        constants,
        interval_resolver,
    ):
        expected_arity = {"min": 2, "max": 2, "clamp": 3}.get(function_name)
        if (
            expected_arity is None
            or len(arguments) != expected_arity
            or self.glsl_function_overloads_by_name.get(function_name)
        ):
            return None

        if function_name == "clamp":
            lower = interval_resolver(arguments[1], intervals, constants)
            upper = interval_resolver(arguments[2], intervals, constants)
            if lower is None or upper is None or lower[1] > upper[0]:
                return None
            value = interval_resolver(arguments[0], intervals, constants)
            if value is None:
                return lower[0], upper[1]
            lower_clamped = (
                max(value[0], lower[0]),
                max(value[1], lower[1]),
            )
            return (
                min(lower_clamped[0], upper[0]),
                min(lower_clamped[1], upper[1]),
            )

        argument_intervals = [
            interval_resolver(argument, intervals, constants) for argument in arguments
        ]
        if any(interval is None for interval in argument_intervals):
            return None
        left, right = argument_intervals
        if function_name == "min":
            return min(left[0], right[0]), min(left[1], right[1])
        return max(left[0], right[0]), max(left[1], right[1])

    def glsl_integer_storage_interval(self, interval, source_type, *, component=False):
        """Apply the destination's integer conversion before retaining flow facts."""
        if interval is None:
            return None
        info = self.glsl_value_type_info(source_type)
        if info is None or info["family"] not in {"int", "uint"}:
            return interval
        if info["width"] != 1 and not component:
            return None
        bits = info["narrow"]["bits"] if info["narrow"] else info["bits"]
        converted = convert_integer_intervals([interval], bits, info["family"] == "int")
        return converted[0][0], converted[-1][1]

    def glsl_congruent_storage_interval(
        self, expression, intervals, constants, source_type
    ):
        """Bound wrapped unsigned ring arithmetic stored into ``source_type``.

        The result is mathematical; callers apply the destination conversion.
        """
        info = self.glsl_value_type_info(source_type)
        if info is None or info["width"] != 1 or info["family"] not in {"int", "uint"}:
            return None
        bits = info["narrow"]["bits"] if info["narrow"] else info["bits"]
        return congruent_ring_interval(
            expression,
            bits,
            bounds=lambda node: self.glsl_private_pointer_interval(
                node, intervals, constants
            ),
            domain=lambda node: self.glsl_integer_type_domain(node, allow_wide=True),
            operator=lambda node: self.map_operator(node.op),
            walk=self.walk_ast,
        )

    def glsl_integer_cast_interval(
        self,
        function_name,
        arguments,
        intervals,
        constants,
        interval_resolver,
    ):
        if len(arguments) != 1:
            return None
        destination = self.glsl_value_type_info(function_name)
        if (
            destination is None
            or destination["family"] not in {"int", "uint"}
            or destination["width"] != 1
        ):
            return None
        source_interval = interval_resolver(arguments[0], intervals, constants)
        if source_interval is None:
            return None
        bits = (
            destination["narrow"]["bits"]
            if destination["narrow"] is not None
            else destination["bits"]
        )
        if destination["family"] == "int":
            domain = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        else:
            domain = (0, (1 << bits) - 1)
        if source_interval[0] < domain[0] or source_interval[1] > domain[1]:
            return None
        return source_interval

    def glsl_integer_scalar_cast_call(self, expression):
        if not isinstance(expression, FunctionCallNode):
            return None
        function_name = self.function_call_name(expression)
        arguments = list(
            getattr(
                expression,
                "arguments",
                getattr(expression, "args", []),
            )
            or []
        )
        destination = self.glsl_value_type_info(function_name)
        if (
            len(arguments) != 1
            or destination is None
            or destination["family"] not in {"int", "uint"}
            or destination["width"] != 1
        ):
            return None
        return function_name, arguments

    def glsl_constant_return_integer_interval(self, expression, constants):
        """Summarize constants using the same source-typed helper contract."""
        result = scalar_return_interval(self, expression, {}, constants)
        return result if result is not None and result[0] == result[1] else None

    def glsl_private_pointer_affine_interval(self, expression, intervals, constants):
        if expression is None:
            return None
        integer_cast = self.glsl_integer_scalar_cast_call(expression)
        if integer_cast is not None:
            function_name, arguments = integer_cast
            return self.glsl_integer_cast_interval(
                function_name,
                arguments,
                intervals,
                constants,
                self.glsl_private_pointer_affine_interval,
            )
        value = self.literal_int_value(expression, constants)
        if value is not None:
            return value, value
        name = self.expression_name(expression)
        if name in intervals:
            return intervals[name]
        if isinstance(expression, UnaryOpNode):
            if expression.op in {"++", "--"}:
                return None
            operand = self.glsl_private_pointer_affine_interval(
                expression.operand, intervals, constants
            )
            if operand is None:
                return None
            if expression.op == "+":
                return operand
            if expression.op == "-":
                return -operand[1], -operand[0]
            return None
        if isinstance(expression, BinaryOpNode):
            if expression.op in {"+", "-"}:
                left = self.glsl_private_pointer_affine_interval(
                    expression.left, intervals, constants
                )
                right = self.glsl_private_pointer_affine_interval(
                    expression.right, intervals, constants
                )
                if left is None or right is None:
                    return None
                if expression.op == "+":
                    return left[0] + right[0], left[1] + right[1]
                return left[0] - right[1], left[1] - right[0]
            if expression.op != "*":
                return None
            left_coefficient = self.literal_int_value(expression.left, constants)
            right_coefficient = self.literal_int_value(expression.right, constants)
            if left_coefficient is not None:
                coefficient = left_coefficient
                affine_expression = expression.right
            elif right_coefficient is not None:
                coefficient = right_coefficient
                affine_expression = expression.left
            else:
                return None
            affine = self.glsl_private_pointer_affine_interval(
                affine_expression, intervals, constants
            )
            if affine is None:
                return None
            products = (coefficient * affine[0], coefficient * affine[1])
            return min(products), max(products)
        if isinstance(expression, FunctionCallNode):
            function_name = self.function_call_name(expression)
            arguments = list(
                getattr(
                    expression,
                    "arguments",
                    getattr(expression, "args", []),
                )
                or []
            )
            cast_interval = self.glsl_integer_cast_interval(
                function_name,
                arguments,
                intervals,
                constants,
                self.glsl_private_pointer_affine_interval,
            )
            if cast_interval is not None:
                return cast_interval
            constant_return = self.glsl_constant_return_integer_interval(
                expression,
                constants,
            )
            if constant_return is not None:
                return constant_return
            return self.glsl_private_pointer_builtin_range_interval(
                function_name,
                arguments,
                intervals,
                constants,
                self.glsl_private_pointer_affine_interval,
            )
        return None

    @staticmethod
    def glsl_metadata_element_key(name, element):
        return f"@metadata:{name}[{element}]"

    def glsl_metadata_element_interval(self, expression, intervals, constants):
        if (
            isinstance(expression, FunctionCallNode)
            and self.function_call_name(expression) == "buffer_load"
            and "buffer_load" not in self.function_return_types
        ):
            arguments = list(expression.arguments or [])
            if len(arguments) != 2:
                return None
            resource, index = arguments
        elif isinstance(expression, ArrayAccessNode):
            resource, index = expression.array, expression.index
        else:
            return None
        if not isinstance(resource, IdentifierNode):
            return None
        prefix = f"@metadata:{resource.name}["
        if not any(key.startswith(prefix) for key in intervals if isinstance(key, str)):
            return None
        if not self.glsl_side_effect_free_expression(index):
            return None
        index_range = self.glsl_checked_private_pointer_interval(
            index, intervals, constants, require_arithmetic_types=True
        )
        if (
            index_range is None
            or index_range[0] < 0
            or index_range[1] - index_range[0] >= MAX_METADATA_PRECONDITIONS
        ):
            return None
        values = [
            intervals.get(self.glsl_metadata_element_key(resource.name, element))
            for element in range(index_range[0], index_range[1] + 1)
        ]
        if not values or any(value is None for value in values):
            return None
        return min(value[0] for value in values), max(value[1] for value in values)

    def glsl_private_pointer_interval_key(self, expression, intervals, constants):
        field_key = self.glsl_declared_struct_member_interval_key(expression)
        if field_key is not None:
            return field_key if field_key in intervals else None
        if isinstance(expression, MemberAccessNode):
            component_index = self.glsl_index_component_index(expression.member)
            key = self.glsl_index_component_interval_key(
                expression.object,
                component_index,
            )
            if key is not None and key in intervals:
                return key
            member_name = expression_debug_name(expression)
            if member_name in intervals:
                return member_name
        if isinstance(expression, ArrayAccessNode):
            component_index = self.literal_int_value(expression.index, constants)
            key = self.glsl_index_component_interval_key(
                expression.array,
                component_index,
            )
            if key is not None and key in intervals:
                return key
            return None
        name = self.expression_name(expression)
        return name if name in intervals else None

    @staticmethod
    def glsl_pointer_array_interval_key(name, index):
        if (
            name
            and isinstance(index, int)
            and not isinstance(index, bool)
            and index >= 0
        ):
            return f"{name}.[{index}]"
        return None

    def glsl_private_pointer_interval(self, expression, intervals, constants):
        if expression is None:
            return None
        if isinstance(expression, ArrayAccessNode):
            bounds = getattr(expression, "_glsl_lowered_pointer_offset_interval", None)
            if bounds is not None:
                return bounds
        returned_interval = scalar_return_interval(
            self, expression, intervals, constants
        )
        if returned_interval is not None:
            return returned_interval
        element_interval = self.glsl_metadata_element_interval(
            expression, intervals, constants
        )
        if element_interval is not None:
            return element_interval
        if isinstance(expression, (MemberAccessNode, ArrayAccessNode)):
            if isinstance(expression, ArrayAccessNode):
                name = self.expression_name(expression.array)
                index = self.glsl_private_pointer_interval(
                    expression.index, intervals, constants
                )
                if (
                    name
                    and index is not None
                    and 0 <= index[0] <= index[1]
                    and index[1] - index[0] < len(intervals)
                ):
                    bounds = [
                        intervals.get(
                            self.glsl_pointer_array_interval_key(name, element)
                        )
                        for element in range(index[0], index[1] + 1)
                    ]
                    if bounds and all(item is not None for item in bounds):
                        return min(item[0] for item in bounds), max(
                            item[1] for item in bounds
                        )
            key = self.glsl_private_pointer_interval_key(
                expression, intervals, constants
            )
            if key is not None:
                return intervals[key]
            if isinstance(expression, ArrayAccessNode):
                return None
        integer_cast = self.glsl_integer_scalar_cast_call(expression)
        if integer_cast is not None:
            function_name, arguments = integer_cast
            return self.glsl_integer_cast_interval(
                function_name,
                arguments,
                intervals,
                constants,
                self.glsl_private_pointer_interval,
            )
        value = self.literal_int_value(expression, constants)
        if value is not None:
            return value, value
        # Flow facts belong to source identifiers, not emitted GLSL aliases.
        name = (
            expression.name
            if isinstance(expression, IdentifierNode)
            else self.expression_name(expression)
        )
        if name in intervals:
            return intervals[name]
        if isinstance(expression, UnaryOpNode):
            if expression.op in {"++", "--"}:
                return None
            operand = self.glsl_private_pointer_interval(
                expression.operand, intervals, constants
            )
            if operand is None:
                return None
            if expression.op == "+":
                return operand
            if expression.op == "-":
                return -operand[1], -operand[0]
            return None
        if isinstance(expression, BinaryOpNode):
            if expression.op == "&":
                left_mask = self.literal_int_value(expression.left, constants)
                right_mask = self.literal_int_value(expression.right, constants)
                mask = left_mask if left_mask is not None else right_mask
                if mask is not None and mask >= 0:
                    return 0, mask
            left = self.glsl_private_pointer_interval(
                expression.left, intervals, constants
            )
            right = self.glsl_private_pointer_interval(
                expression.right, intervals, constants
            )
            if left is None or right is None:
                return None
            if self.map_operator(expression.op) in {"<<", ">>"}:
                return self.glsl_unsigned_shift_interval(expression, left, right)
            if expression.op == "+":
                return left[0] + right[0], left[1] + right[1]
            if expression.op == "-":
                return left[0] - right[1], left[1] - right[0]
            if expression.op == "*":
                products = (
                    left[0] * right[0],
                    left[0] * right[1],
                    left[1] * right[0],
                    left[1] * right[1],
                )
                return min(products), max(products)
            if expression.op == "/" and right[0] > 0:
                return truncating_division_interval(left, right)
            if expression.op == "%" and right[0] > 0 and left[0] >= 0:
                remainder_interval = nonnegative_remainder_interval(left, right)
                if right[0] != right[1] or left[0] == left[1]:
                    return remainder_interval
                divisor = right[0]
                maximum = remainder_interval[1]
                dividend = expression.left
                if (
                    isinstance(dividend, BinaryOpNode)
                    and self.map_operator(dividend.op) == "*"
                ):
                    factors = (
                        self.literal_int_value(dividend.left, constants),
                        self.literal_int_value(dividend.right, constants),
                    )
                    exact_factor = next(
                        (
                            abs(factor)
                            for factor in factors
                            if isinstance(factor, int)
                            and not isinstance(factor, bool)
                            and factor != 0
                        ),
                        None,
                    )
                    domain = self.glsl_integer_type_domain(
                        dividend,
                        allow_wide=True,
                    )
                    if (
                        exact_factor is not None
                        and domain is not None
                        and left[0] >= domain.minimum
                        and left[1] <= domain.maximum
                    ):
                        # With no source overflow, every dividend remains a
                        # multiple of the exact factor.  Its non-negative
                        # remainders are therefore spaced by gcd(factor,
                        # divisor), so the largest reachable residue can be
                        # smaller than divisor - 1 (for example,
                        # (4 * lane) % 8 reaches only 0 or 4).
                        a, b = exact_factor, divisor
                        while b:
                            a, b = b, a % b
                        maximum = min(maximum, divisor - a)
                return 0, maximum
            return None
        if isinstance(expression, TernaryOpNode):
            condition = self.glsl_private_pointer_condition_value(
                expression.condition,
                intervals,
                constants,
            )
            if condition is not None:
                selected = expression.true_expr if condition else expression.false_expr
                return self.glsl_private_pointer_interval(
                    selected,
                    intervals,
                    constants,
                )
            true_interval = self.glsl_private_pointer_interval(
                expression.true_expr, intervals, constants
            )
            false_interval = self.glsl_private_pointer_interval(
                expression.false_expr, intervals, constants
            )
            if true_interval is None or false_interval is None:
                return None
            return (
                min(true_interval[0], false_interval[0]),
                max(true_interval[1], false_interval[1]),
            )
        if isinstance(expression, FunctionCallNode):
            function_name = self.function_call_name(expression)
            arguments = list(
                getattr(
                    expression,
                    "arguments",
                    getattr(expression, "args", []),
                )
                or []
            )
            cast_interval = self.glsl_integer_cast_interval(
                function_name,
                arguments,
                intervals,
                constants,
                self.glsl_private_pointer_interval,
            )
            if cast_interval is not None:
                return cast_interval
            constant_return = self.glsl_constant_return_integer_interval(
                expression,
                constants,
            )
            if constant_return is not None:
                return constant_return
            return self.glsl_private_pointer_builtin_range_interval(
                function_name,
                arguments,
                intervals,
                constants,
                self.glsl_private_pointer_interval,
            )
        return None
