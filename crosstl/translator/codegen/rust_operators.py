"""Rust binary, ternary, vector and matrix operators, array literals, constructors and assignments."""

from .assignment_values import is_reevaluable_place
from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    BlockNode,
    LiteralNode,
    MatchNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
)
from ..refusals import RustRefusal


class RustOperatorMixin:
    """Operator and assignment members of ``RustCodeGen``."""

    def generate_binary_expression(self, expr, target_type=None):
        left_expr = getattr(expr, "left", "")
        right_expr = getattr(expr, "right", "")
        op = getattr(expr, "operator", getattr(expr, "op", "+"))
        mapped_op = self.map_operator(op)

        left_type = self.expression_result_type(left_expr)
        right_type = self.expression_result_type(right_expr)
        bool_vector_logical = self.generate_bool_vector_logical_expression(
            left_expr, right_expr, left_type, right_type, mapped_op
        )
        if bool_vector_logical is not None:
            return bool_vector_logical

        if mapped_op in {"&&", "||"}:
            left = self.generate_condition_expression(left_expr)
            right = self.generate_condition_expression(right_expr)
            return f"({left} {mapped_op} {right})"

        vector_comparison = self.generate_vector_comparison_expression(
            left_expr, right_expr, left_type, right_type, mapped_op
        )
        if vector_comparison is not None:
            return vector_comparison

        vector_arithmetic = self.generate_vector_arithmetic_expression(
            left_expr, right_expr, left_type, right_type, mapped_op
        )
        if vector_arithmetic is not None:
            return vector_arithmetic

        matrix_multiply = self.generate_matrix_multiply_expression(
            left_expr, right_expr, left_type, right_type, mapped_op
        )
        if matrix_multiply is not None:
            return matrix_multiply

        matrix_vector_plan = self.binary_matrix_vector_plan(
            left_type, right_type, mapped_op, target_type
        )
        if matrix_vector_plan is not None:
            matrix_vector = self.generate_matrix_vector_expression(
                left_expr,
                right_expr,
                left_type,
                right_type,
                matrix_vector_plan,
            )
            if matrix_vector is not None:
                return matrix_vector
            left = self.generate_binary_composite_operand(
                left_expr, left_type, matrix_vector_plan["left_target_type"]
            )
            right = self.generate_binary_composite_operand(
                right_expr, right_type, matrix_vector_plan["right_target_type"]
            )
            return f"({left} {mapped_op} {right})"

        scalar_left_vector = self.generate_scalar_left_vector_binary_expression(
            left_expr, right_expr, left_type, right_type, mapped_op
        )
        if scalar_left_vector is not None:
            return scalar_left_vector

        composite_operand_type = self.binary_composite_operand_type(
            left_type, right_type, mapped_op
        )
        if composite_operand_type is not None:
            left = self.generate_binary_composite_operand(
                left_expr, left_type, composite_operand_type
            )
            right = self.generate_binary_composite_operand(
                right_expr, right_type, composite_operand_type
            )
            return f"({left} {mapped_op} {right})"

        operand_type = self.binary_scalar_operand_type(
            left_type,
            right_type,
            mapped_op,
        )
        left = self.generate_expression(left_expr)
        right = self.generate_expression(right_expr)
        if operand_type is not None:
            left = self.normalize_binary_scalar_operand(
                left_expr,
                left,
                left_type,
                operand_type,
            )
            right = self.normalize_binary_scalar_operand(
                right_expr,
                right,
                right_type,
                operand_type,
            )

        return f"({left} {mapped_op} {right})"

    def generate_scalar_mul_add_call(self, func_name, args):
        if func_name != "fma" or len(args or []) != 3:
            return None

        arg_types = [self.expression_result_type(arg) for arg in args]
        if any(
            self.vector_type_info(arg_type) is not None
            or self.matrix_type_info(arg_type) is not None
            for arg_type in arg_types
        ):
            return None

        if any(self.normalize_scalar_type(arg_type) is None for arg_type in arg_types):
            return None

        target_type = self.normalize_scalar_type(self.promoted_argument_type(arg_types))
        if target_type not in {"f32", "f64"}:
            return None

        value = self.generate_mul_add_receiver(args[0], arg_types[0], target_type)
        multiplier = self.generate_mul_add_scalar_operand(
            args[1], arg_types[1], target_type
        )
        addend = self.generate_mul_add_scalar_operand(
            args[2], arg_types[2], target_type
        )
        return f"{value}.mul_add({multiplier}, {addend})"

    def generate_mul_add_receiver(self, expr, source_type, target_type):
        value = self.generate_mul_add_scalar_operand(expr, source_type, target_type)
        if self.is_numeric_literal_expression(expr):
            return f"({value} as {target_type})"
        return f"({value})"

    def generate_mul_add_scalar_operand(self, expr, source_type, target_type):
        generated = self.generate_expression(expr)
        return self.normalize_binary_scalar_operand(
            expr,
            generated,
            source_type,
            target_type,
        )

    def generate_reciprocal_call(self, func_name, args):
        if func_name != "rcp" or len(args or []) != 1:
            return None

        arg = args[0]
        arg_type = self.expression_result_type(arg)
        scalar_type = self.normalize_scalar_type(arg_type)
        if scalar_type in {"f32", "f64"}:
            return f"(1.0 / {self.generate_expression(arg)})"

        vector_info = self.vector_type_info(arg_type)
        if vector_info is not None:
            component_type = self.normalize_scalar_type(vector_info["component_type"])
            if component_type not in {"f32", "f64"}:
                return None
            temp_bindings = []
            lanes = self.vector_argument_lane_expressions(
                arg, vector_info, temp_bindings, component_type
            )
            lanes = [f"(1.0 / {lane})" for lane in lanes]
            return self.generate_constructor_call(
                self.map_type(arg_type), lanes, temp_bindings
            )

        matrix_info = self.matrix_type_info(arg_type)
        if matrix_info is not None:
            component_type = self.normalize_scalar_type(matrix_info["component_type"])
            if component_type not in {"f32", "f64"}:
                return None
            temp_bindings = []
            lanes = self.matrix_argument_lane_expressions(
                arg, matrix_info, temp_bindings, component_type
            )
            lanes = [f"(1.0 / {lane})" for lane in lanes]
            return self.generate_constructor_call(
                self.map_type(arg_type), lanes, temp_bindings
            )

        return None

    def is_numeric_literal_expression(self, expr):
        if isinstance(expr, (int, float)) and not isinstance(expr, bool):
            return True
        return (
            isinstance(expr, LiteralNode)
            and isinstance(expr.value, (int, float))
            and not isinstance(expr.value, bool)
        )

    def generate_glam_matrix_vector_product(
        self, left_expr, right_expr, left_type, right_type, plan
    ):
        """``M * v`` and ``v * M`` with glam's square matrices: ``v * M`` is ``transpose(M) * v``."""
        matrix_type = left_type if plan["matrix_on_left"] else right_type
        vector_type = right_type if plan["matrix_on_left"] else left_type
        matrix_info = self.matrix_type_info(matrix_type)
        vector_info = self.vector_type_info(vector_type)
        if matrix_info is None or vector_info is None:
            return None
        if matrix_info["columns"] != matrix_info["rows"]:
            return None
        matrix_scalar = self.normalize_scalar_type(matrix_info["component_type"])
        if matrix_scalar != self.normalize_scalar_type(vector_info["component_type"]):
            return None
        left = self.generate_expression(left_expr)
        right = self.generate_expression(right_expr)
        if plan["matrix_on_left"]:
            return f"({left} * {right})"
        return f"(({right}).transpose() * {left})"

    def generate_matrix_multiply_expression(
        self, left_expr, right_expr, left_type, right_type, operator
    ):
        plan = self.matrix_multiply_plan(left_type, right_type, operator)
        if plan is None:
            return None
        left_info = self.matrix_type_info(left_type)
        right_info = self.matrix_type_info(right_type)
        if left_info == right_info and left_info["columns"] == left_info["rows"]:
            # glam multiplies two square matrices of one type (OpMatrixTimesMatrix).
            return (
                f"({self.generate_expression(left_expr)} * "
                f"{self.generate_expression(right_expr)})"
            )

        temp_bindings = []
        left_lanes = self.matrix_argument_lane_expressions(
            left_expr,
            plan["left_matrix"],
            temp_bindings,
            plan["component_type"],
        )
        right_lanes = self.matrix_argument_lane_expressions(
            right_expr,
            plan["right_matrix"],
            temp_bindings,
            plan["component_type"],
        )

        left_cell = self.matrix_lane_accessor(left_lanes, plan["left_matrix"])
        right_cell = self.matrix_lane_accessor(right_lanes, plan["right_matrix"])
        result_lanes = []
        for column in range(plan["result_columns"]):
            for row in range(plan["result_rows"]):
                terms = [
                    f"({left_cell(k, row)} * {right_cell(column, k)})"
                    for k in range(plan["shared_size"])
                ]
                result_lanes.append(self.sum_rust_terms(terms))

        return self.generate_constructor_call(
            self.map_type(plan["result_type"]),
            result_lanes,
            temp_bindings,
        )

    def generate_matrix_vector_expression(
        self, left_expr, right_expr, left_type, right_type, plan
    ):
        if not self.rust_matrix_vector_requires_lanes(plan):
            return None
        glam_product = self.generate_glam_matrix_vector_product(
            left_expr, right_expr, left_type, right_type, plan
        )
        if glam_product is not None:
            return glam_product

        temp_bindings = []
        matrix_expr = left_expr if plan["matrix_on_left"] else right_expr
        vector_expr = right_expr if plan["matrix_on_left"] else left_expr
        matrix_type = left_type if plan["matrix_on_left"] else right_type
        vector_type = right_type if plan["matrix_on_left"] else left_type
        matrix_info = self.matrix_type_info(matrix_type)
        vector_info = self.vector_type_info(vector_type)
        if matrix_info is None or vector_info is None:
            return None

        result_info = self.vector_type_info(plan["result_type"])
        if result_info is None:
            return None
        component_type = result_info["component_type"]

        matrix_lanes = self.matrix_argument_lane_expressions(
            matrix_expr,
            matrix_info,
            temp_bindings,
            component_type,
        )
        vector_lanes = self.vector_argument_lane_expressions(
            vector_expr,
            vector_info,
            temp_bindings,
            component_type,
        )
        matrix_cell = self.matrix_lane_accessor(matrix_lanes, matrix_info)

        result_lanes = []
        if plan["matrix_on_left"]:
            for row in range(matrix_info["rows"]):
                terms = [
                    f"({matrix_cell(column, row)} * {vector_lanes[column]})"
                    for column in range(matrix_info["columns"])
                ]
                result_lanes.append(self.sum_rust_terms(terms))
        else:
            for column in range(matrix_info["columns"]):
                terms = [
                    f"({vector_lanes[row]} * {matrix_cell(column, row)})"
                    for row in range(matrix_info["rows"])
                ]
                result_lanes.append(self.sum_rust_terms(terms))

        return self.generate_constructor_call(
            self.map_type(plan["result_type"]),
            result_lanes,
            temp_bindings,
        )

    def matrix_lane_accessor(self, lanes, matrix_info):
        rows = matrix_info["rows"]
        return lambda column, row: lanes[column * rows + row]

    def sum_rust_terms(self, terms):
        if not terms:
            return "0"
        expression = terms[0]
        for term in terms[1:]:
            expression = f"({expression} + {term})"
        return expression

    def generate_vector_arithmetic_expression(
        self, left_expr, right_expr, left_type, right_type, operator
    ):
        plan = self.vector_arithmetic_plan(left_type, right_type, operator)
        if plan is None or not self.rust_vector_arithmetic_requires_lanes(plan):
            return None

        temp_bindings = []
        left_lanes = self.vector_comparison_operand_lanes(
            left_expr,
            left_type,
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        right_lanes = self.vector_comparison_operand_lanes(
            right_expr,
            right_type,
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        lanes = [
            f"({left_lane} {operator} {right_lane})"
            for left_lane, right_lane in zip(left_lanes, right_lanes)
        ]
        return self.generate_constructor_call(
            self.map_type(plan["result_type"]),
            lanes,
            temp_bindings,
        )

    def generate_scalar_left_vector_binary_expression(
        self, left_expr, right_expr, left_type, right_type, operator
    ):
        if operator not in {"+", "-", "*", "/", "%"}:
            return None

        left_scalar = self.normalize_scalar_type(left_type)
        right_vector = self.vector_type_info(right_type)
        if left_scalar is None or right_vector is None:
            return None

        result_type = self.vector_type_for_promoted_scalar(right_vector, left_scalar)
        result_info = self.vector_type_info(result_type)
        if result_info is None:
            return None

        component_type = result_info["component_type"]
        if self.normalize_scalar_type(component_type) == "bool":
            return None

        temp_bindings = []
        left_lanes = self.vector_comparison_operand_lanes(
            left_expr,
            left_type,
            component_type,
            result_info["size"],
            temp_bindings,
        )
        right_lanes = self.vector_argument_lane_expressions(
            right_expr,
            right_vector,
            temp_bindings,
            component_type,
        )
        lanes = [
            f"({left_lane} {operator} {right_lane})"
            for left_lane, right_lane in zip(left_lanes, right_lanes)
        ]
        return self.generate_constructor_call(
            self.map_type(result_type),
            lanes,
            temp_bindings,
        )

    def generate_binary_composite_operand(self, expr, source_type, composite_type):
        if self.vector_type_info(source_type) or self.matrix_type_info(source_type):
            operand = self.generate_expression_with_type(expr, composite_type)
            return self.normalize_typed_expression_value(expr, operand, composite_type)

        component_type = self.composite_component_type(composite_type)
        if component_type is not None and self.normalize_scalar_type(source_type):
            operand = self.generate_expression_with_type(expr, component_type)
            return self.normalize_scalar_assignment_value(
                expr, operand, source_type, component_type
            )

        return self.generate_expression(expr)

    def composite_component_type(self, composite_type):
        vector_info = self.vector_type_info(composite_type)
        if vector_info is not None:
            return vector_info["component_type"]

        matrix_info = self.matrix_type_info(composite_type)
        if matrix_info is not None:
            return matrix_info["component_type"]

        return None

    def generate_bool_vector_logical_expression(
        self, left_expr, right_expr, left_type, right_type, operator
    ):
        plan = self.bool_vector_logical_plan(left_type, right_type, operator)
        if plan is None:
            return None

        temp_bindings = []
        left_lanes = self.vector_comparison_operand_lanes(
            left_expr, left_type, "bool", plan["size"], temp_bindings
        )
        right_lanes = self.vector_comparison_operand_lanes(
            right_expr, right_type, "bool", plan["size"], temp_bindings
        )
        lanes = [
            f"({left_lane} {operator} {right_lane})"
            for left_lane, right_lane in zip(left_lanes, right_lanes)
        ]
        return self.generate_constructor_call(
            self.map_type(plan["result_type"]), lanes, temp_bindings
        )

    def generate_bool_vector_not_expression(self, expr, operator):
        if operator != "!":
            return None

        vector_info = self.vector_type_info(self.expression_result_type(expr))
        if vector_info is None or vector_info["component_type"] != "bool":
            return None

        temp_bindings = []
        lanes = self.vector_argument_lane_expressions(
            expr, vector_info, temp_bindings, "bool"
        )
        lanes = [f"(!{lane})" for lane in lanes[: vector_info["size"]]]
        result_type = self.vector_type_for_components("bool", vector_info["size"])
        return self.generate_constructor_call(
            self.map_type(result_type), lanes, temp_bindings
        )

    def generate_vector_comparison_expression(
        self, left_expr, right_expr, left_type, right_type, operator
    ):
        plan = self.vector_comparison_plan(left_type, right_type, operator)
        if plan is None:
            return None

        temp_bindings = []
        left_lanes = self.vector_comparison_operand_lanes(
            left_expr,
            left_type,
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        right_lanes = self.vector_comparison_operand_lanes(
            right_expr,
            right_type,
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        lanes = [
            f"({left_lane} {operator} {right_lane})"
            for left_lane, right_lane in zip(left_lanes, right_lanes)
        ]
        return self.generate_constructor_call(
            self.map_type(plan["result_type"]), lanes, temp_bindings
        )

    def vector_comparison_operand_lanes(
        self, expr, source_type, target_component_type, size, temp_bindings
    ):
        source_info = self.vector_type_info(source_type)
        if source_info is not None:
            return self.vector_argument_lane_expressions(
                expr, source_info, temp_bindings, target_component_type
            )

        scalar_expr = self.generate_expression_with_type(expr, target_component_type)
        scalar_expr = self.normalize_scalar_assignment_value(
            expr, scalar_expr, source_type, target_component_type
        )
        if not self.is_repeat_safe_expression(expr):
            temp_name = self.next_vector_arg_temp_name()
            temp_bindings.append((temp_name, scalar_expr))
            scalar_expr = temp_name
        return [scalar_expr] * size

    def generate_condition_expression(self, expr):
        if isinstance(expr, UnaryOpNode):
            op = self.map_operator(getattr(expr, "operator", getattr(expr, "op", "")))
            if op == "!":
                operand = getattr(expr, "operand", "")
                return f"!({self.generate_condition_expression(operand)})"

        condition = self.generate_expression(expr)
        return self.normalize_condition_expression(expr, condition)

    def normalize_condition_expression(self, expr, generated_expr):
        condition_type = self.normalize_scalar_type(self.expression_result_type(expr))
        if condition_type is None or condition_type == "bool":
            return generated_expr

        zero_literal = "0.0" if condition_type in {"f16", "f32", "f64"} else "0"
        return f"({generated_expr} != {zero_literal})"

    def generate_ternary_expression(self, expr, target_type=None):
        condition_expr = getattr(expr, "condition", "")
        true_expr = getattr(expr, "true_expr", "")
        false_expr = getattr(expr, "false_expr", "")

        bool_vector_ternary = self.generate_bool_vector_ternary_expression(
            condition_expr, true_expr, false_expr, target_type
        )
        if bool_vector_ternary is not None:
            return bool_vector_ternary

        condition = self.generate_condition_expression(condition_expr)
        branch_type = target_type or self.expression_result_type(expr)

        true_value = self.generate_expression_with_type(true_expr, branch_type)
        false_value = self.generate_expression_with_type(false_expr, branch_type)
        true_value = self.normalize_typed_expression_value(
            true_expr, true_value, branch_type
        )
        false_value = self.normalize_typed_expression_value(
            false_expr, false_value, branch_type
        )
        return f"(if {condition} {{ {true_value} }} else {{ {false_value} }})"

    def generate_bool_vector_ternary_expression(
        self, condition_expr, true_expr, false_expr, target_type=None
    ):
        plan = self.bool_vector_ternary_plan(
            condition_expr, true_expr, false_expr, target_type
        )
        if plan is None:
            return None

        temp_bindings = []
        condition_info = self.vector_type_info(
            self.expression_result_type(condition_expr)
        )
        condition_lanes = self.vector_argument_lane_expressions(
            condition_expr, condition_info, temp_bindings, "bool"
        )
        true_lanes = self.vector_comparison_operand_lanes(
            true_expr,
            self.expression_result_type(true_expr),
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        false_lanes = self.vector_comparison_operand_lanes(
            false_expr,
            self.expression_result_type(false_expr),
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        lanes = [
            f"(if {condition_lane} {{ {true_lane} }} else {{ {false_lane} }})"
            for condition_lane, true_lane, false_lane in zip(
                condition_lanes, true_lanes, false_lanes
            )
        ]
        return self.generate_constructor_call(
            self.map_type(plan["result_type"]), lanes, temp_bindings
        )

    def is_array_type_name(self, type_name):
        return type_name is not None and "[" in str(type_name) and "]" in str(type_name)

    def c_array_type_parts(self, type_name):
        if type_name is None:
            return None, []
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        if "[" not in type_name or "]" not in type_name:
            return type_name, []

        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket]
        suffix = type_name[open_bracket:]
        sizes = []
        while suffix.startswith("["):
            close_bracket = suffix.find("]")
            if close_bracket < 0:
                return type_name, []

            size_text = suffix[1:close_bracket].strip()
            if not size_text:
                sizes.append(None)
            else:
                try:
                    sizes.append(int(size_text))
                except ValueError:
                    sizes.append(None)
            suffix = suffix[close_bracket + 1 :]

        if suffix:
            return type_name, []
        return base_type, sizes

    def format_c_array_type(self, base_type, sizes):
        suffix = "".join(f"[{'' if size is None else size}]" for size in sizes)
        return f"{base_type}{suffix}"

    def c_array_outer_element_type_and_size(self, type_name):
        base_type, sizes = self.c_array_type_parts(type_name)
        if not sizes:
            return None, None
        element_type = (
            self.format_c_array_type(base_type, sizes[1:])
            if len(sizes) > 1
            else base_type
        )
        return element_type, sizes[0]

    def generate_array_literal_expression(
        self, expr, target_type=None, static_context=False
    ):
        if not (getattr(expr, "elements", None) or []):
            enum_initializer = self.rust_empty_brace_enum_initializer(target_type)
            if enum_initializer is not None:
                return enum_initializer

        element_target_type = None
        target_size = None

        if self.is_array_type_name(target_type):
            element_target_type, target_size = self.c_array_outer_element_type_and_size(
                target_type
            )

        elements = []
        for element in expr.elements:
            if element_target_type is None:
                elements.append(self.generate_expression(element))
                continue

            element_expr = self.generate_expression_with_type(
                element,
                element_target_type,
                static_context=static_context,
            )
            elements.append(
                self.normalize_typed_expression_value(
                    element,
                    element_expr,
                    element_target_type,
                )
            )

        if element_target_type is not None:
            if target_size is None:
                return f"vec![{', '.join(elements)}]"

            elements = elements[:target_size]
            padding = self.rust_array_padding_expression(
                element_target_type, static_context=static_context
            )
            while len(elements) < target_size:
                elements.append(padding)

        return f"[{', '.join(elements)}]"

    def generate_constructor_expression(self, expr, target_type=None):
        type_name = self.normalize_turbofish_type_name(
            self.convert_type_node_to_string(expr.constructor_type)
        )
        rust_type = self.map_type(type_name)

        named_arguments = getattr(expr, "named_arguments", {}) or {}
        if named_arguments:
            variant_info = self.enum_variant_struct_constructor_info(
                type_name,
                target_type,
            )
            if variant_info is not None:
                enum_type, variant_name, field_types = variant_info
                constructor_path = self.enum_variant_call_path(enum_type, variant_name)
                fields = self.generate_constructor_named_fields(
                    named_arguments,
                    lambda name: field_types.get(name),
                )
                return f"{constructor_path} {{ {', '.join(fields)} }}"

            constructor_path = self.rust_constructor_path(rust_type)
            fields = self.generate_constructor_named_fields(
                named_arguments,
                lambda name: self.resolve_struct_member_type(type_name, name),
            )
            return f"{constructor_path} {{ {', '.join(fields)} }}"

        arguments = getattr(expr, "arguments", []) or []
        member_types = self.resolve_struct_positional_member_types(type_name)
        args = self.generate_constructor_call_args_with_types(arguments, member_types)
        return f"{self.rust_constructor_path(rust_type)}::new({', '.join(args)})"

    def generate_constructor_named_fields(self, named_arguments, field_type_for_name):
        argument_values = list(named_arguments.values())
        later_roots = self.return_call_arg_later_roots(argument_values)
        lambda_conflict_roots = self.return_call_arg_lambda_conflict_roots(
            argument_values
        )
        fields = []
        for index, (name, value) in enumerate(named_arguments.items()):
            field_type = field_type_for_name(name)
            field_value = self.generate_with_lambda_capture_move_blocks(
                value,
                later_roots[index],
                lambda value=value, field_type=field_type: (
                    self.generate_constructor_argument_with_type(
                        value,
                        field_type,
                    )
                ),
                lambda_conflict_roots[index],
            )
            fields.append(f"{self.rust_identifier(name)}: {field_value}")
        return fields

    def rust_array_padding_expression(self, base_type, static_context=False):
        enum_initializer = self.rust_empty_brace_enum_array_element(base_type)
        if enum_initializer is not None:
            return enum_initializer
        if static_context:
            rust_type = self.map_type(base_type)
            scalar_initializer = self.rust_scalar_default_literal(rust_type)
            if scalar_initializer is not None:
                return scalar_initializer
            if self.is_rust_shader_pod_value_type(rust_type):
                return self.rust_gpu_default_value(rust_type)
        return "Default::default()"

    def generate_assignment(self, node):
        # Handle both old and new AST assignment structures
        if hasattr(node, "target") and hasattr(node, "value"):
            # New AST structure
            target_expr = node.target
            value_expr = node.value
            op = getattr(node, "operator", "=")
        else:
            # Old AST structure
            target_expr = node.left
            value_expr = node.right
            op = getattr(node, "operator", "=")

        padded_store = self.generate_rust_gpu_padded_slice_assignment(
            target_expr, value_expr, op
        )
        if padded_store is not None:
            return padded_store

        swizzle_assignment = self.generate_swizzle_assignment(
            target_expr, value_expr, op
        )
        if swizzle_assignment is not None:
            return swizzle_assignment

        pointer_assignment = self.generate_pointer_index_assignment(
            target_expr, value_expr, op
        )
        if pointer_assignment is not None:
            return pointer_assignment

        glsl_block_store = self.generate_glsl_buffer_block_assignment(
            target_expr, value_expr, op
        )
        if glsl_block_store is not None:
            return glsl_block_store

        vector_compound_assignment = self.generate_vector_compound_assignment(
            target_expr,
            value_expr,
            op,
        )
        if vector_compound_assignment is not None:
            return vector_compound_assignment

        if hasattr(node, "target") and hasattr(node, "value"):
            # New AST structure
            lhs = self.generate_assignment_target_expression(node.target)
            lhs_type = self.expression_result_type(node.target)
            rhs = self.generate_expression_with_type(node.value, lhs_type)
            rhs = self.normalize_assignment_rhs(
                lhs_type, node.value, rhs, getattr(node, "operator", "=")
            )
            op = getattr(node, "operator", "=")
        else:
            # Old AST structure
            lhs = self.generate_assignment_target_expression(node.left)
            lhs_type = self.expression_result_type(node.left)
            rhs = self.generate_expression_with_type(node.right, lhs_type)
            rhs = self.normalize_assignment_rhs(
                lhs_type, node.right, rhs, getattr(node, "operator", "=")
            )
            op = getattr(node, "operator", "=")
        return f"{lhs} {op} {rhs}"

    def generate_vector_compound_assignment(self, target, value, operator):
        binary_operator = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
            "%=": "%",
        }.get(operator)
        if binary_operator is None:
            return None

        target_type = self.expression_result_type(target)
        if self.vector_type_info(target_type) is None:
            return None

        value_type = self.expression_result_type(value)
        if binary_operator == "*" and self.matrix_type_info(value_type) is not None:
            # v *= m is v = v * m, a row-vector product glam has no MulAssign for.
            if not is_reevaluable_place(target):
                return self.refuse_lowering(
                    "vector *= matrix: the target has side effects and would be evaluated twice"
                )
            lhs = self.generate_assignment_target_expression(target)
            expression = BinaryOpNode(target, "*", value)
            return f"{lhs} = {self.generate_binary_expression(expression, target_type)}"
        if (
            self.vector_type_info(value_type) is None
            and self.normalize_scalar_type(value_type) is None
        ):
            return None

        lhs = self.generate_assignment_target_expression(target)
        rhs = self.generate_vector_arithmetic_expression(
            target,
            value,
            target_type,
            value_type,
            binary_operator,
        )
        if rhs is None:
            expression = BinaryOpNode(target, binary_operator, value)
            rhs = self.generate_binary_expression(expression, target_type)
            rhs = self.normalize_typed_expression_value(expression, rhs, target_type)
        return f"{lhs} = {rhs}"

    def generate_swizzle_assignment(self, target, value, operator):
        if not isinstance(target, MemberAccessNode):
            return None

        components = self.member_swizzle_components(target)
        if components is None or len(components) <= 1:
            return None

        self.validate_swizzle_assignment_target(target, components)

        target_type = self.expression_result_type(target)
        rhs = self.generate_expression_with_type(value, target_type)
        rhs = self.normalize_assignment_rhs(target_type, value, rhs, operator)
        rhs_info = self.vector_type_info(self.expression_result_type(value))
        target_info = self.vector_type_info(target_type)
        if rhs_info is None and target_info is not None:
            rhs = self.normalize_typed_expression_value(value, rhs, target_type)

        object_expr = getattr(target, "object_expr", getattr(target, "object", None))
        self.assignment_lhs_depth += 1
        try:
            obj = self.generate_expression(object_expr)
        finally:
            self.assignment_lhs_depth -= 1

        rhs_name = self.next_swizzle_temp_name()
        assignments = []
        if rhs_info is not None:
            rhs_components = ("x", "y", "z", "w")
            for index, component in enumerate(components):
                assignments.append(
                    f"{obj}.{component} {operator} "
                    f"{rhs_name}.{rhs_components[index]}"
                )
        else:
            for component in components:
                assignments.append(f"{obj}.{component} {operator} {rhs_name}")
        return f"{{ let {rhs_name} = {rhs}; {'; '.join(assignments)}; }}"

    def validate_swizzle_assignment_target(self, target, components):
        if len(components) > 4:
            raise RustRefusal(
                "Rust swizzle assignment target cannot exceed 4 components"
            )
        if len(set(components)) != len(components):
            raise RustRefusal("Rust swizzle assignment target cannot repeat components")

        object_expr = getattr(target, "object_expr", getattr(target, "object", None))
        object_type = self.expression_result_type(object_expr)
        vector_info = self.vector_type_info(object_type)
        if vector_info is None:
            return

        component_indices = {"x": 0, "y": 1, "z": 2, "w": 3}
        for component in components:
            if component_indices[component] >= vector_info["size"]:
                raise RustRefusal(
                    "Rust swizzle assignment target component "
                    f"'{component}' is not available on "
                    f"{self.map_type(object_type)}"
                )

    def generate_pointer_index_assignment(self, target, value, operator):
        if not isinstance(target, ArrayAccessNode):
            return None

        array_expr = getattr(target, "array_expr", getattr(target, "array", None))
        pointer_type = self.pointer_type_parts(
            self.map_type(self.expression_result_type(array_expr))
        )
        if pointer_type is None:
            return None

        index_expr = getattr(target, "index_expr", getattr(target, "index", None))
        pointer = self.generate_expression(array_expr)
        index = self.generate_array_index_expression(index_expr)
        lhs_type = self.expression_result_type(target)
        rhs = self.generate_expression_with_type(value, lhs_type)
        rhs = self.normalize_assignment_rhs(lhs_type, value, rhs, operator)
        return f"unsafe {{ *({pointer}).add({index}) {operator} {rhs} }}"

    def generate_glsl_buffer_block_assignment(self, target, value, op):
        access = self.glsl_buffer_block_access(target)
        if access is None:
            return None
        if access == "readonly":
            return self.unsupported_glsl_buffer_block_statement(
                "store", "a readonly block cannot be written"
            )
        root_name = self.glsl_buffer_block_root_name(target)
        if root_name in self.rust_entry_resource_infos:
            # The block reaches the entry point and its helpers as a value of the block's struct type, so a store
            # would change a copy and never reach the bound buffer.
            return self.unsupported_glsl_buffer_block_statement(
                "store",
                "the block is passed by value, so a store would not reach the buffer",
            )
        if root_name in self.static_variable_names and root_name not in (
            self.local_variable_names
        ):
            return self.unsupported_glsl_buffer_block_statement(
                "store", "the block is an immutable Rust static"
            )
        lhs = self.generate_assignment_target_expression(target)
        lhs_type = self.expression_result_type(target)
        rhs = self.generate_expression_with_type(value, lhs_type)
        rhs = self.normalize_assignment_rhs(lhs_type, value, rhs, op)
        return f"{lhs} {op} {rhs}"

    def generate_assignment_target_expression(self, target):
        self.assignment_lhs_depth += 1
        try:
            return self.generate_expression(target)
        finally:
            self.assignment_lhs_depth -= 1

    def normalize_assignment_rhs(self, lhs_type, rhs_expr, generated_rhs, operator):
        if operator == "=":
            return self.normalize_typed_expression_value(
                rhs_expr, generated_rhs, lhs_type
            )

        compound_ops = {
            "+=",
            "-=",
            "*=",
            "/=",
            "%=",
            "^=",
            "|=",
            "&=",
            "<<=",
            ">>=",
        }
        if operator not in compound_ops:
            return generated_rhs

        target_type = self.normalize_scalar_type(lhs_type)
        source_type = self.expression_result_type(rhs_expr)
        if target_type is None or self.normalize_scalar_type(source_type) is None:
            return generated_rhs
        return self.normalize_binary_scalar_operand(
            rhs_expr, generated_rhs, source_type, target_type
        )

    def normalize_typed_expression_value(self, expr, generated_expr, target_type):
        if isinstance(expr, TernaryOpNode):
            return generated_expr
        if self.generated_binary_expression_matches_target(expr, target_type):
            return generated_expr

        reference_expr = self.normalize_reference_typed_expression(
            expr,
            generated_expr,
            target_type,
        )
        if reference_expr is not None:
            return reference_expr

        vector_expr = self.normalize_vector_typed_expression(expr, target_type)
        if vector_expr is not None:
            return vector_expr
        matrix_expr = self.normalize_matrix_typed_expression(expr, target_type)
        if matrix_expr is not None:
            return matrix_expr
        return self.normalize_scalar_assignment_value(
            expr, generated_expr, self.expression_result_type(expr), target_type
        )

    def normalize_reference_typed_expression(self, expr, generated_expr, target_type):
        target_reference = self.reference_type_parts_for_type(target_type)
        if target_reference is None:
            return None
        if isinstance(expr, (MatchNode, BlockNode)):
            return generated_expr

        generated_text = str(generated_expr).strip()
        if not generated_text:
            return generated_expr

        target_is_mutable, _ = target_reference
        source_reference = self.reference_type_parts_for_type(
            self.expression_result_type(expr)
        )
        if source_reference is not None:
            source_is_mutable, _ = source_reference
            if not target_is_mutable or source_is_mutable:
                return generated_expr

        if generated_text.startswith("&mut "):
            return generated_expr
        if not target_is_mutable and generated_text.startswith("&"):
            return generated_expr

        borrow_prefix = "&mut " if target_is_mutable else "&"
        return f"{borrow_prefix}{generated_expr}"

    def generated_binary_expression_matches_target(self, expr, target_type):
        if not isinstance(expr, BinaryOpNode) or target_type is None:
            return False

        left_type = self.expression_result_type(expr.left)
        right_type = self.expression_result_type(expr.right)
        operator = self.map_operator(
            getattr(expr, "operator", getattr(expr, "op", None))
        )
        matrix_vector_plan = self.binary_matrix_vector_plan(
            left_type, right_type, operator, target_type
        )
        if matrix_vector_plan is None:
            return False
        return self.type_names_match(matrix_vector_plan["result_type"], target_type)

    def type_names_match(self, left_type, right_type):
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)
        if left_scalar is not None or right_scalar is not None:
            return left_scalar == right_scalar
        return self.map_type(left_type) == self.map_type(right_type)

    def normalize_scalar_assignment_value(
        self, expr, generated_expr, source_type, target_type
    ):
        source_type = self.normalize_scalar_type(source_type)
        target_type = self.normalize_scalar_type(target_type)
        if source_type is None or target_type is None or source_type == target_type:
            return generated_expr

        if target_type == "bool":
            zero_literal = "0.0" if source_type in {"f16", "f32", "f64"} else "0"
            return f"({generated_expr} != {zero_literal})"

        if source_type == "bool" and target_type in {"f16", "f32", "f64"}:
            return f"(if {generated_expr} {{ 1.0 }} else {{ 0.0 }})"

        if self.is_integer_literal_expression(expr):
            if target_type in {"f32", "f64"}:
                return f"{self.integer_literal_expression_value(expr)}.0"
            if target_type == "f16":
                return f"({generated_expr} as f16)"
            return generated_expr

        return f"({generated_expr} as {target_type})"
