"""Expression kinds of the SPIR-V generator: string literals and named constants, literal nodes, variables, array subscripts, binary, unary and ternary operators, and member accesses."""

from ..ast import (
    AssignmentNode,
    FunctionCallNode,
    IdentifierNode,
)
from .spirv_literal_context import (
    is_literal_operand,
)
from .spirv_ray_builtins import (
    ray_constant_value,
    transposed_ray_builtin,
)
from .spirv_short_circuit import (
    SHORT_CIRCUIT_OPERATORS,
    lower_short_circuit_logical,
    lower_short_circuit_ternary,
)
from .spirv_single_evaluation import (
    evaluate_subscript_index,
)
from .spirv_standard_constants import (
    spirv_standard_float_constant,
)
from .spirv_value_subscripts import (
    vector_value_subscript,
)


class SpirvExpressionKindMixin:
    """Per-kind members of ``VulkanSPIRVCodeGen.process_expression``."""

    def spirv_string_expression(
        self,
        expr,
    ):
        """Lower a bare string: a named constant, a variable, a built-in or a literal."""
        if expr.rsplit("::", 1)[-1] == "None":
            none_default = self.option_none_default_value()
            if none_default is not None:
                return none_default

        if self.enum_variant_is_payload_path(expr):
            enum_value = self.process_enum_variant_constructor(expr, [])
            if enum_value is not None:
                return enum_value
        else:
            enum_constant = self.enum_variant_constant(expr)
            if enum_constant is not None:
                return enum_constant

        if expr in self.local_variables:
            var_id = self.local_variables[expr]
            return self.get_variable_value(var_id)
        generic_literal = self.current_generic_literal_int_constants().get(expr)
        if generic_literal is not None:
            primitive_type = self.integer_literal_type_for_value("int", generic_literal)
            return self.register_constant(
                generic_literal,
                self.register_primitive_type(primitive_type),
            )
        elif expr in self.named_constants:
            return self.named_constants[expr]
        else:
            var_id = self.resolve_global_variable(expr)
            if var_id is not None:
                self.mark_interface_variable_if_needed(var_id)
                self.mark_builtin_interface_variable(var_id)
                return self.get_variable_value(var_id)

            if expr in self.cbuffer_members:
                member_pointer = self.cbuffer_member_pointer(expr)
                if member_pointer is not None:
                    return self.get_variable_value(member_pointer)

            patch_input = self.ensure_tessellation_patch_constant_input(expr)
            if patch_input is not None:
                return self.get_variable_value(patch_input)

            builtin_component = self.process_dotted_builtin(expr)
            if builtin_component is not None:
                return builtin_component

            ray_constant = ray_constant_value(expr)
            if ray_constant is not None:
                return self.register_constant(
                    ray_constant, self.register_primitive_type("uint")
                )

            transposed = transposed_ray_builtin(expr)
            if transposed is not None:
                # gl_ObjectToWorld3x4EXT and gl_WorldToObject3x4EXT transpose a built-in (spirv_ray_builtins).
                return self.process_expression(
                    FunctionCallNode(
                        IdentifierNode("transpose"), [IdentifierNode(transposed)]
                    )
                )

            builtin = self.ensure_builtin_variable(expr)
            if builtin is not None:
                return self.get_variable_value(builtin)

            builtin_limit = self.glsl_builtin_limit_constant(expr)
            if builtin_limit is not None:
                return builtin_limit

            standard_constant = spirv_standard_float_constant(self, expr)
            if standard_constant is not None:
                return standard_constant

            # Create a default float constant for missing variables in examples
            # This is to make the SPIR-V code valid even if we can't find the variable
            if expr.replace(".", "", 1).isdigit():  # Check if it's a numeric string
                float_type = self.register_primitive_type("float")
                try:
                    value = float(expr)
                    return self.register_constant(value, float_type)
                except ValueError:
                    pass

            self.refuse_lowering(f"Unknown variable {expr}")
            float_type = self.register_primitive_type("float")
            return self.register_constant(0.0, float_type)

    def spirv_literal_expression(
        self,
        expr,
    ):
        """Lower a literal node to a typed constant."""
        literal_type = self.convert_type_node_to_string(expr.literal_type)
        primitive_type_name = self.normalize_primitive_name(literal_type)
        if primitive_type_name in {"float", "double"}:
            literal_type_id = self.register_primitive_type(primitive_type_name)
            return self.register_constant(float(expr.value), literal_type_id)
        if primitive_type_name in self.INTEGER_TYPE_NAMES:
            literal_value = int(expr.value)
            if (
                primitive_type_name in self.SIGNED_INTEGER_TYPES
                and self.expected_primitive_type_name() in self.UNSIGNED_INTEGER_TYPES
                and literal_value >= 0
            ):
                primitive_type_name = self.expected_primitive_type_name()
            primitive_type_name = self.integer_literal_type_for_value(
                primitive_type_name, literal_value
            )
            literal_type_id = self.register_primitive_type(primitive_type_name)
            return self.register_constant(literal_value, literal_type_id)
        if primitive_type_name == "bool":
            literal_type_id = self.register_primitive_type("bool")
            if isinstance(expr.value, str):
                value = expr.value.lower() == "true"
            else:
                value = bool(expr.value)
            return self.register_constant(value, literal_type_id)
        return self.process_expression(expr.value)

    def spirv_variable_expression(
        self,
        expr,
    ):
        """Load a variable node's value."""
        if expr.name in self.local_variables:
            var_id = self.local_variables[expr.name]
            return self.get_variable_value(var_id)
        elif expr.name in self.named_constants:
            return self.named_constants[expr.name]
        else:
            var_id = self.resolve_global_variable(expr.name)
            if var_id is not None:
                self.mark_interface_variable_if_needed(var_id)
                self.mark_builtin_interface_variable(var_id)
                return self.get_variable_value(var_id)

            if expr.name in self.cbuffer_members:
                member_pointer = self.cbuffer_member_pointer(expr.name)
                if member_pointer is not None:
                    return self.get_variable_value(member_pointer)

            patch_input = self.ensure_tessellation_patch_constant_input(expr.name)
            if patch_input is not None:
                return self.get_variable_value(patch_input)

            builtin = self.ensure_builtin_variable(expr.name)
            if builtin is not None:
                return self.get_variable_value(builtin)

            builtin_limit = self.glsl_builtin_limit_constant(expr.name)
            if builtin_limit is not None:
                return builtin_limit

            standard_constant = spirv_standard_float_constant(self, expr.name)
            if standard_constant is not None:
                return standard_constant

            self.refuse_lowering(f"Unknown variable {expr.name}")
            float_type = self.register_primitive_type("float")
            return self.register_constant(0.0, float_type)

    def spirv_array_access_expression(
        self,
        expr,
    ):
        """Lower an array subscript."""
        index = evaluate_subscript_index(self, expr.index)

        if index is None:
            self.refuse_lowering(f"Failed to evaluate array access")
            float_type = self.register_primitive_type("float")
            return self.register_constant(0.0, float_type)

        reinterpret_load = self.process_storage_pointer_reinterpret_load(
            expr.array, index
        )
        if reinterpret_load is not None:
            return reinterpret_load
        component = vector_value_subscript(self, expr.array, index, expr.index)
        if component is not None:
            return component

        access, element_type = self.create_array_element_access(
            expr.array, index, expr.index
        )

        if access is None or element_type is None:
            self.refuse_lowering(
                "Could not determine array element type for "
                f"{self.diagnostic_expression(expr.array)}"
            )
            element_type = self.primitive_types["float"]
            return self.register_constant(0.0, element_type)

        return self.load_from_variable(access, element_type)

    def spirv_binary_expression(
        self,
        expr,
    ):
        """Lower a binary operator."""
        if expr.op == ",":
            if isinstance(expr.left, AssignmentNode):
                self.process_assignment(expr.left)
            else:
                self.process_expression(expr.left)
            return self.process_expression(expr.right)
        if expr.op in SHORT_CIRCUIT_OPERATORS:
            short_circuit = lower_short_circuit_logical(self, expr)
            if short_circuit is not None:
                return short_circuit
        if expr.op in {"&", "|", "^"}:
            expected_type = self.bitwise_expression_operand_type(
                self.infer_expression_result_type(expr.left),
                self.infer_expression_result_type(expr.right),
            )
            if expected_type is not None:
                left = self.process_expression_with_expected_type(
                    expr.left, expected_type
                )
                right = self.process_expression_with_expected_type(
                    expr.right, expected_type
                )
            else:
                left = self.process_expression(expr.left)
                right = self.process_expression(expr.right)
        else:
            # A literal operand takes the other operand's type, or its own when that is unknown, never the
            # enclosing expression's: in uint(v % 3) and (v > 0) ? 1u : 0u the literal stays int, so the
            # operation stays signed. Operations without a literal operand keep the enclosing context, which
            # loads and constructors read.
            previous_expected_type = self.current_expression_expected_type
            if is_literal_operand(expr.left) or is_literal_operand(expr.right):
                operand_type = self.bitwise_expression_operand_type(
                    self.infer_expression_result_type(expr.left),
                    self.infer_expression_result_type(expr.right),
                )
                self.current_expression_expected_type = (
                    self.type_name_string(operand_type)
                    if operand_type is not None
                    else None
                )
            try:
                left = self.process_expression(expr.left)
                right = self.process_expression(expr.right)
            finally:
                self.current_expression_expected_type = previous_expected_type

        if left is None or right is None:
            self.refuse_lowering(
                f"Operand of {expr.op!r} has no value: "
                f"{self.diagnostic_expression(expr.left if left is None else expr.right)}"
            )
            float_type = self.register_primitive_type("float")
            return self.register_constant(0.0, float_type)

        left_type = self.registered_value_type(left) or self.ensure_registered_type(
            left.type
        )
        right_type = self.registered_value_type(right) or self.ensure_registered_type(
            right.type
        )
        result_type = self.binary_expression_result_type(expr.op, left_type, right_type)
        if result_type is None:
            result_type = left_type

        return self.binary_operation(expr.op, result_type, left, right)

    def spirv_unary_expression(
        self,
        expr,
    ):
        """Lower a unary operator."""
        if expr.op in {"++", "--"}:
            return self.process_increment_expression(expr)
        if expr.op == "*" and not getattr(expr, "is_postfix", False):
            pointer = self.dereference_pointer_from_expression(expr.operand)
            value_type = (
                self.variable_value_types.get(pointer.id)
                if pointer is not None
                else None
            )
            if pointer is None or value_type is None:
                raise self.unsupported_pointer_dereference_error(expr)
            return self.load_from_variable(pointer, value_type)

        operand = self.process_expression(expr.operand)
        if operand is None:
            self.refuse_lowering(
                f"Operand of {expr.op!r} has no value: "
                f"{self.diagnostic_expression(expr.operand)}"
            )
            float_type = self.register_primitive_type("float")
            return self.register_constant(0.0, float_type)

        return self.unary_operation(expr.op, operand.type, operand)

    def spirv_ternary_expression(
        self,
        expr,
    ):
        """Lower a conditional expression."""
        short_circuit = lower_short_circuit_ternary(self, expr)
        if short_circuit is not None:
            return short_circuit
        condition = self.process_expression(expr.condition)
        true_value = self.process_expression(expr.true_expr)
        false_value = self.process_expression(expr.false_expr)

        if condition is None:
            self.refuse_lowering(
                "Conditional operator condition has no value: "
                f"{self.diagnostic_expression(expr.condition)}"
            )
            condition = self.register_constant(
                False, self.register_primitive_type("bool")
            )
        if true_value is None or false_value is None:
            self.refuse_lowering(
                "Conditional operator branch has no value: "
                + self.diagnostic_expression(
                    expr.true_expr if true_value is None else expr.false_expr
                )
            )
            float_type = self.register_primitive_type("float")
            fallback = self.register_constant(0.0, float_type)
            true_value = true_value or fallback
            false_value = false_value or fallback

        result_type, true_value, false_value = self.normalize_ternary_values(
            true_value, false_value
        )
        return self.select_operation(result_type, condition, true_value, false_value)

    def spirv_member_access_expression(
        self,
        expr,
    ):
        """Lower a member access."""
        member_name = expr.member
        base_pointer = self.variable_pointer_from_expression(expr.object)
        if base_pointer is not None:
            access = self.create_member_access_pointer(base_pointer, member_name)
            if access is not None:
                member_type = self.variable_value_types.get(access.id)
                return self.load_from_variable(access, member_type)

        base = self.process_expression(expr.object)
        if base is None:
            return None

        struct_type = base.type.base_type
        member_info = self.struct_member_info(struct_type, member_name)
        if member_info is not None:
            member_index, member_type = member_info
            return self.composite_extract(base, member_type, member_index)

        member_info = self.vector_member_info(struct_type, member_name)
        if member_info is not None:
            member_index, member_type = member_info
            return self.composite_extract(base, member_type, member_index)

        swizzle_info = self.vector_swizzle_info(struct_type, member_name)
        if swizzle_info is not None:
            indices, _, result_type = swizzle_info
            return self.vector_shuffle(base, result_type, indices)

        if (
            self.vector_component_type_and_count(struct_type) is not None
            and len(member_name) > 1
        ):
            self.refuse_lowering(
                f"Invalid vector swizzle {member_name} for {struct_type}"
            )
            return None

        selected = self.scalar_swizzle(base, member_name)
        if selected is not None:
            return selected

        self.refuse_lowering(f"Could not find member {member_name} in {struct_type}")
        return None
