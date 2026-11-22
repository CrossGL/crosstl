"""GLSL renderings of the CrossGL expression forms that ``GLSLCodeGen.generate_expression`` dispatches on: names, literals, operators, subscripts, constructors, member and pointer accesses, and conditional expressions."""

from ..ast import (
    ArrayLiteralNode,
)
from .generic_struct_utils import (
    generate_struct_constructor_expression,
)
from .glsl_double_literals import binary64_literal_arms
from .glsl_shared_array_values import (
    is_shared_array_value,
)
from .opengl_values import (
    OpenGLStoragePointerError,
    OpenGLWorkgroupPointerError,
)
from .relational_builtins import (
    glsl_vector_logical_not,
)
from .storage_image_texels import (
    is_whole_texel_load,
)


class OpenGLExpressionFormMixin:
    """Expression-form members of ``GLSLCodeGen.generate_expression``."""

    def generate_glsl_raw_string_expression(
        self,
        expr,
    ):
        """Render an expression that is already a source string."""
        scalar_array_element = self.glsl_scalar_array_view_element_expression(expr)
        if scalar_array_element is not None:
            return scalar_array_element
        if self.is_glsl_builtin_option_none_expression(expr):
            none_value = self.generate_glsl_builtin_option_none_value()
            if none_value is not None:
                return none_value
        binding = self.current_glsl_workgroup_pointer_aliases.get(expr)
        if (
            binding is not None
            and binding.get("kind") == "workgroup-pointer"
            and not is_shared_array_value(self, expr)
        ):
            raise OpenGLWorkgroupPointerError(
                "OpenGL cannot emit a workgroup pointer as a first-class value: "
                f"{expr}",
                parameter_name=expr,
                reason="bare-pointer-expression",
            )
        storage_binding = self.current_glsl_storage_pointer_aliases.get(expr)
        if (
            storage_binding is not None
            and storage_binding.get("kind") == "storage-pointer"
            and not storage_binding.get("resource_root")
        ):
            raise OpenGLStoragePointerError(
                "OpenGL cannot emit a storage pointer as a first-class value: "
                f"{expr}",
                parameter_name=expr,
                reason="bare-pointer-expression",
            )
        return expr

    def generate_glsl_variable_expression(
        self,
        expr,
    ):
        """Render a variable reference."""
        if hasattr(expr, "name"):
            scalar_array_element = self.glsl_scalar_array_view_element_expression(
                expr.name
            )
            if scalar_array_element is not None:
                return scalar_array_element
            if self.is_glsl_builtin_option_none_expression(expr):
                none_value = self.generate_glsl_builtin_option_none_value()
                if none_value is not None:
                    return none_value
            self.glsl_workgroup_pointer_bare_expression_error(expr)
            self.glsl_storage_pointer_bare_expression_error(expr)
            if expr.name in self.current_resource_aliases:
                return self.current_resource_aliases[expr.name]["expression"]
            if expr.name in getattr(self, "enum_variant_constants", {}):
                return self.glsl_enum_value_expression(expr.name)
            substitution = self.glsl_compile_time_bitcast_substitution_expression(
                expr.name
            )
            if substitution is not None:
                return substitution
            if expr.name in self.current_identifier_aliases:
                return self.current_identifier_aliases[expr.name]
            return self.generate_glsl_identifier_expression(expr.name)
        else:
            return str(expr)

    def generate_glsl_identifier_node_expression(
        self,
        expr,
    ):
        """Render an identifier node."""
        scalar_array_element = self.glsl_scalar_array_view_element_expression(expr.name)
        if scalar_array_element is not None:
            return scalar_array_element
        if self.is_glsl_builtin_option_none_expression(expr):
            none_value = self.generate_glsl_builtin_option_none_value()
            if none_value is not None:
                return none_value
        self.glsl_workgroup_pointer_bare_expression_error(expr)
        self.glsl_storage_pointer_bare_expression_error(expr)
        if expr.name in self.current_resource_aliases:
            return self.current_resource_aliases[expr.name]["expression"]
        if expr.name in getattr(self, "enum_variant_constants", {}):
            return self.glsl_enum_value_expression(expr.name)
        substitution = self.glsl_compile_time_bitcast_substitution_expression(expr.name)
        if substitution is not None:
            return substitution
        if expr.name in self.current_identifier_aliases:
            return self.current_identifier_aliases[expr.name]
        return self.generate_glsl_identifier_expression(expr.name)

    def generate_glsl_literal_expression(
        self,
        expr,
    ):
        """Render a literal."""
        literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
        if literal_type == "string":
            return f'"{expr.value}"'
        if literal_type == "char":
            return f"'{expr.value}'"
        mapped_literal_type = self.map_type(literal_type) if literal_type else None
        if (
            mapped_literal_type == "uint64_t"
            and isinstance(expr.value, int)
            and not isinstance(expr.value, bool)
        ):
            return f"{expr.value}ul"
        if (
            mapped_literal_type == "int64_t"
            and isinstance(expr.value, int)
            and not isinstance(expr.value, bool)
        ):
            return f"{expr.value}l"
        if (
            literal_type == "uint"
            and isinstance(expr.value, int)
            and not isinstance(expr.value, bool)
        ):
            return f"{expr.value}u"
        if isinstance(expr.value, bool):
            return "true" if expr.value else "false"
        if isinstance(expr.value, int):
            if expr.value > 0xFFFFFFFF:
                return f"{expr.value}ul"
            if expr.value > 0x7FFFFFFF:
                return f"{expr.value}u"
            if expr.value < -0x80000000:
                return f"{expr.value}l"
        return str(expr.value)

    def generate_glsl_binary_expression(
        self,
        expr,
    ):
        """Render a binary operator expression."""
        op = self.map_operator(expr.op)
        emulated_matrix = self.generate_glsl_emulated_matrix_binary(expr, op)
        if emulated_matrix is not None:
            return emulated_matrix
        if op in {"&&", "||"}:
            self.reject_glsl_software_subgroup_control_flow(expr)
            left = self.generate_glsl_boolean_context(expr.left)
            right = self.generate_glsl_boolean_context(expr.right)
            return f"({left} {op} {right})"
        option_none_comparison = self.generate_glsl_builtin_option_none_comparison(
            expr.left, op, expr.right
        )
        if option_none_comparison is not None:
            return option_none_comparison
        vector_relational = self.generate_vector_relational_expression(
            expr.left, op, expr.right, source_node=expr
        )
        if vector_relational is not None:
            return vector_relational
        vector_equality = self.generate_glsl_componentwise_equality(
            expr.left, op, expr.right
        )
        if vector_equality is not None:
            return vector_equality
        float_remainder = self.generate_glsl_float_remainder(expr.left, op, expr.right)
        if float_remainder is not None:
            return float_remainder
        left_type = self.glsl_source_expression_type(expr.left)
        right_type = self.glsl_source_expression_type(expr.right)
        if self.is_glsl_registered_complex64_type(
            left_type
        ) or self.is_glsl_registered_complex64_type(right_type):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            return self.glsl_complex64_binary_expression(
                expr,
                left,
                right,
                op,
                left_type,
                right_type,
            )
        operand_types = self.glsl_binary_operand_conversion_types(
            left_type,
            right_type,
            op,
            source_node=expr,
            fail_closed=True,
        )
        if operand_types is not None:
            left_expected, right_expected = operand_types
            left = self.generate_expression_with_expected(expr.left, left_expected)
            right = self.generate_expression_with_expected(expr.right, right_expected)
        else:
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
        if op in {"+", "-", "*", "/"} and operand_types is None:
            expected_vector = self.expected_float_vector_expression_type()
            if expected_vector:
                left = self.cast_integer_vector_expression_for_expected_float(
                    left, left_type, expected_vector
                )
                right = self.cast_integer_vector_expression_for_expected_float(
                    right, right_type, expected_vector
                )
        if op == "%":
            remainder = self.generate_glsl_signed_remainder(
                expr, left, right, left_type, right_type
            )
            if remainder is not None:
                return remainder
        if op == "-":
            difference = self.generate_glsl_identical_difference(expr, left, right)
            if difference is not None:
                return self.glsl_apply_narrow_float_contract(
                    difference, self.glsl_source_expression_type(expr), expr
                )
        if self.current_glsl_ieee_operator_arithmetic:
            ieee_arithmetic = self.generate_glsl_ieee_operator_arithmetic(
                expr, op, left, right
            )
            if ieee_arithmetic is not None:
                return ieee_arithmetic
        return self.glsl_apply_narrow_float_contract(
            self.glsl_narrow_vector_wrap(expr, op, f"({left} {op} {right})"),
            self.glsl_source_expression_type(expr),
            expr,
        )

    def generate_glsl_unary_expression(
        self,
        expr,
    ):
        """Render a unary operator expression."""
        op = self.map_operator(expr.op)
        emulated_matrix = self.generate_glsl_emulated_matrix_unary(expr, op)
        if emulated_matrix is not None:
            return emulated_matrix
        if op == "*":
            aggregate = self.generate_glsl_private_scalar_struct_reinterpret_read(expr)
            if aggregate is not None:
                return aggregate
            pointee = self.generate_glsl_private_pointer_view_access(
                expr.operand, access_expression=expr
            )
            if pointee is not None:
                return pointee
            pointee = self.glsl_workgroup_pointer_element_expression(expr.operand)
            if pointee is not None:
                return pointee
            pointee = self.glsl_storage_pointer_element_expression(expr.operand)
            if pointee is not None:
                return pointee
        if op == "&":
            operand_name = self.expression_name(expr.operand)
            if operand_name in getattr(self, "current_glsl_scalar_array_views", {}):
                return self.generate_glsl_identifier_expression(operand_name)
            binding = self.glsl_workgroup_pointer_binding(
                expr,
                self.glsl_workgroup_pointer_aliases(),
            )
            if binding is not None:
                raise OpenGLWorkgroupPointerError(
                    "OpenGL cannot emit a workgroup address as a first-class value",
                    parameter_name=self.expression_name(expr.operand),
                    reason="address-expression-unsupported",
                    source_location=getattr(expr, "source_location", None),
                )
            storage_binding = self.glsl_storage_pointer_binding(
                expr,
                self.glsl_storage_pointer_aliases(),
            )
            if storage_binding is not None:
                raise OpenGLStoragePointerError(
                    "OpenGL cannot emit a storage-buffer address as a "
                    "first-class value",
                    parameter_name=self.expression_name(expr.operand),
                    reason="address-expression-unsupported",
                    source_location=getattr(expr, "source_location", None),
                )
        if op in {"++", "--"} or expr.op in {
            "PRE_INCREMENT",
            "PRE_DECREMENT",
            "POST_INCREMENT",
            "POST_DECREMENT",
        }:
            pointer_step = self.generate_glsl_storage_pointer_step(expr, op)
            if pointer_step is not None:
                return pointer_step
            self.validate_glsl_private_pointer_byte_view_mutation(expr.operand)
            self.validate_glsl_storage_pointer_mutation_target(expr.operand)
            self.validate_glsl_buffer_block_member_access(expr.operand, "read_write")
            storage_update = self.float_atomic_storage.unary_update(expr, op)
            if storage_update is not None:
                return storage_update
            narrow_update = self.generate_glsl_narrow_unary_update(expr, op)
            if narrow_update is not None:
                return narrow_update
            operand = self.generate_glsl_buffer_block_mutation_target(expr.operand)
        elif op == "!":
            vector_not = glsl_vector_logical_not(self, expr.operand)
            if vector_not is not None:
                return vector_not
            operand = self.generate_glsl_boolean_context(expr.operand)
        else:
            operand = self.generate_expression_with_expected(
                expr.operand,
                self.glsl_unary_operand_conversion_type(expr),
            )
        complex_unary = self.glsl_complex64_unary_expression(expr, operand, op)
        if complex_unary is not None:
            return complex_unary
        is_postfix = bool(getattr(expr, "is_postfix", False)) or expr.op in {
            "POST_INCREMENT",
            "POST_DECREMENT",
        }
        if is_postfix:
            return f"({operand}{op})"
        return self.glsl_narrow_vector_wrap(expr, op, f"({op}{operand})")

    def generate_glsl_mesh_operation_expression(
        self,
        expr,
    ):
        """Render a mesh-shader operation."""
        mesh_output_counts_call = self.generate_mesh_output_counts_call(
            expr.operation, expr.arguments
        )
        if mesh_output_counts_call is not None:
            return mesh_output_counts_call
        arguments = expr.arguments
        if expr.operation == "DispatchMesh":
            if self.current_stage_entry_type in {
                "task",
                "amplification",
                "object",
            }:
                self.validate_dispatch_mesh_arguments(arguments)
            if len(arguments) == 4:
                arguments = arguments[:3]
        args = [self.generate_expression(arg) for arg in arguments]
        return self.map_mesh_intrinsic(expr.operation, args)

    def generate_glsl_array_access_expression(
        self,
        expr,
    ):
        """Render a subscript expression."""
        if hasattr(expr, "array") and hasattr(expr, "index"):
            emulated_matrix_column = self.generate_glsl_emulated_matrix_index(expr)
            if emulated_matrix_column is not None:
                return emulated_matrix_column
            structured_storage = self.generate_glsl_hlsl_structured_read(expr)
            if structured_storage is not None:
                return structured_storage
            fixed_vector_view_access = (
                self.generate_glsl_fixed_array_vector_pointer_view_access(expr)
            )
            if fixed_vector_view_access is not None:
                return fixed_vector_view_access
            private_access = self.generate_glsl_private_pointer_view_access(
                expr.array, expr.index, access_expression=expr
            )
            if private_access is not None:
                return private_access
            workgroup_access = self.glsl_workgroup_pointer_access_expression(expr)
            if workgroup_access is not None:
                return workgroup_access
            storage_access = self.glsl_storage_pointer_access_expression(expr)
            if storage_access is not None:
                return storage_access
            indexed_aggregate = self.glsl_indexed_aggregate_literal_expression(expr)
            if indexed_aggregate is not None:
                return indexed_aggregate
            if is_whole_texel_load(expr.array, self.function_call_name):
                # The subscript selects a component of the whole texel.
                array = self.generate_expression_with_expected(expr.array, None)
            else:
                array = self.generate_expression(expr.array)
            array = self.glsl_precise_selected_value(expr.array, array)
            index = self.glsl_resource_array_index_expression(expr)
            instance_member = self.glsl_structured_buffer_array_element_member(
                expr.array
            )
            if instance_member is not None:
                # One buffer of an array of storage blocks holds its elements in a member.
                return f"{array}.{instance_member}[{index}]"
            return f"{array}[{index}]"
        else:
            return str(expr)

    def generate_glsl_constructor_expression(
        self,
        expr,
    ):
        """Render a constructor expression."""
        union = self.union_storage.construct(
            expr, expr.constructor_type, expr.arguments
        )
        if union is not None:
            return union
        if getattr(expr, "is_braced_constructor", False):
            vector_initializer = self.generate_glsl_vector_aggregate_initializer(
                ArrayLiteralNode(list(expr.arguments)),
                self.map_type(expr.constructor_type),
            )
            if vector_initializer is not None:
                return vector_initializer
        enum_constructor = self.glsl_enum_constructor_expression(expr)
        if enum_constructor is not None:
            return enum_constructor
        constructor = generate_struct_constructor_expression(self, expr)
        if constructor is not None:
            return constructor
        return str(expr)

    def generate_glsl_member_access_expression(
        self,
        expr,
    ):
        """Render a member access."""
        structured_storage = self.generate_glsl_hlsl_structured_read(expr)
        if structured_storage is not None:
            return structured_storage
        flattened_member = self.flattened_stage_member_name(expr)
        if flattened_member is not None:
            return flattened_member
        object_type = self.type_name_string(self.expression_result_type(expr.object))
        if object_type and object_type.rstrip().endswith("*"):
            pointer_member = self.glsl_private_pointer_member_expression(
                expr.object, expr.member
            )
            if pointer_member is None:
                pointer_member = self.glsl_workgroup_pointer_member_expression(
                    expr.object,
                    expr.member,
                    getattr(expr, "source_location", None),
                )
            if pointer_member is not None:
                return pointer_member
            pointer_member = self.glsl_storage_pointer_member_expression(
                expr.object,
                expr.member,
                getattr(expr, "source_location", None),
            )
            if pointer_member is not None:
                return pointer_member
        if not self.glsl_buffer_block_read_validation_suppression:
            self.validate_glsl_buffer_block_member_access(expr, "read")
        obj = self.generate_expression_with_expected(expr.object, None)
        obj = self.glsl_precise_selected_value(expr.object, obj)
        member_name = self.glsl_member_access_name(expr.object, str(expr.member))
        cast_member = self.signed_compute_builtin_member_cast(obj, member_name)
        if cast_member is not None:
            return cast_member
        return f"{obj}.{member_name}"

    def generate_glsl_pointer_access_expression(
        self,
        expr,
    ):
        """Render a pointer member access."""
        pointer_member = self.glsl_private_pointer_member_expression(
            expr.pointer_expr, expr.member
        )
        if pointer_member is None:
            pointer_member = self.glsl_workgroup_pointer_member_expression(
                expr.pointer_expr,
                expr.member,
                getattr(expr, "source_location", None),
            )
        if pointer_member is None:
            pointer_member = self.glsl_storage_pointer_member_expression(
                expr.pointer_expr,
                expr.member,
                getattr(expr, "source_location", None),
            )
        if pointer_member is None:
            pointer_name = self.expression_name(expr.pointer_expr)
            if pointer_name in self.glsl_storage_pointer_aliases():
                raise OpenGLStoragePointerError(
                    "OpenGL pointer member access has no concrete storage-buffer "
                    "backing object",
                    parameter_name=pointer_name,
                    reason="pointer-member-backing-unresolved",
                    source_location=getattr(expr, "source_location", None),
                )
            raise OpenGLWorkgroupPointerError(
                "OpenGL pointer member access has no concrete shared-array "
                "backing object",
                reason="pointer-member-backing-unresolved",
                source_location=getattr(expr, "source_location", None),
            )
        return pointer_member

    def generate_glsl_ternary_expression(
        self,
        expr,
    ):
        """Render a conditional expression."""
        self.reject_glsl_software_subgroup_control_flow(expr)
        condition = self.glsl_precise_condition(
            expr.condition, self.generate_glsl_boolean_context(expr.condition)
        )
        true_type = self.glsl_source_expression_type(expr.true_expr)
        false_type = self.glsl_source_expression_type(expr.false_expr)
        operand_types = self.glsl_binary_operand_conversion_types(
            true_type,
            false_type,
            "?:",
            source_node=expr,
            fail_closed=True,
        )
        binary64_arms = binary64_literal_arms(self, expr)
        if binary64_arms is not None:
            # Two double constants: a plain conditional selects one exactly.
            return f"({condition} ? {binary64_arms[0]} : {binary64_arms[1]})"
        if operand_types is None:
            true_expr = self.generate_expression(expr.true_expr)
            false_expr = self.generate_expression(expr.false_expr)
        else:
            true_expr = self.generate_expression_with_expected(
                expr.true_expr, operand_types[0]
            )
            false_expr = self.generate_expression_with_expected(
                expr.false_expr, operand_types[1]
            )
        true_expr = self.glsl_copied_storage_structure_arm(expr.true_expr, true_expr)
        false_expr = self.glsl_copied_storage_structure_arm(expr.false_expr, false_expr)
        selection = self.generate_glsl_float_selection(
            expr, condition, true_expr, false_expr
        )
        if selection is not None:
            return selection
        return f"({condition} ? {true_expr} : {false_expr})"
