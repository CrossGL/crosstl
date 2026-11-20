"""HLSL renderings of the CrossGL expression forms that ``HLSLCodeGen.generate_expression`` dispatches on: cooperative-matrix operations, literals, names, operators, subscripts, constructors, member accesses and conditional expressions."""

from ..ast import (
    ArrayLiteralNode,
    BinaryOpNode,
    FunctionCallNode,
    LiteralNode,
    PointerReinterpretNode,
)
from .directx_values import (
    DirectXCooperativeMatrixUnsupportedError,
    DirectXPrivatePointerParameterError,
    _directx_cooperative_matrix_fragment_contract,
    _directx_cooperative_matrix_fragment_contract_message,
)
from .enum_utils import (
    enum_value_expression,
    generate_enum_constructor_expression,
)
from .generic_struct_utils import (
    generate_struct_constructor_expression,
)
from .hlsl_wave_extended import (
    hlsl_subgroup_lane_mask_expression,
)
from .storage_image_texels import (
    image_load_swizzle_extent,
)


class DirectXExpressionFormMixin:
    """Expression-form members of ``HLSLCodeGen.generate_expression``."""

    def generate_hlsl_cooperative_matrix_operation_expression(
        self,
        expr,
    ):
        """Render a cooperative-matrix operation."""
        if self.cooperative_matrix_software_lowering:
            return self.directx_cooperative_matrix_operation_lowering(expr)
        operation = expr.operation
        matrix_type = getattr(expr, "result_type", None)
        fragment_contract = _directx_cooperative_matrix_fragment_contract(matrix_type)
        fragment_contract_message = (
            _directx_cooperative_matrix_fragment_contract_message(fragment_contract)
        )
        raise DirectXCooperativeMatrixUnsupportedError(
            "DirectX cooperative matrix lowering is unavailable for "
            f"operation '{operation}'; emitting a regular HLSL expression "
            "would not preserve distributed matrix semantics; "
            f"{fragment_contract_message}",
            operation=operation,
            matrix_type=matrix_type,
            **fragment_contract,
            reason="unsupported-operation",
            source_location=getattr(expr, "source_location", None),
        )

    def generate_hlsl_literal_expression(
        self,
        expr,
    ):
        """Render a literal."""
        if hasattr(expr, "value"):
            value = expr.value
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            mapped_literal_type = self.map_type(literal_type) if literal_type else None
            if (
                mapped_literal_type == "uint64_t"
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                return f"{value}ull"
            if (
                mapped_literal_type == "int64_t"
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                return f"{value}ll"
            if (
                literal_type == "uint"
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                return f"{value}u"
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, int):
                if value > 0xFFFFFFFF:
                    return f"{value}ull"
                if value > 0x7FFFFFFF:
                    return f"{value}u"
                if value < -0x80000000:
                    return f"{value}ll"
            if isinstance(value, str) and not (
                value.startswith('"') and value.endswith('"')
            ):
                return f'"{value}"'
            return str(value)
        return str(expr)

    def generate_hlsl_identifier_node_expression(
        self,
        expr,
    ):
        """Render an identifier node."""
        if self.is_hlsl_builtin_option_none_expression(expr):
            none_value = self.generate_hlsl_builtin_option_none_value()
            if none_value is not None:
                return none_value
        unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
        if unsupported_value is not None:
            return unsupported_value
        name = getattr(expr, "name", str(expr))
        enum_value = enum_value_expression(self, name)
        if enum_value is not None:
            return enum_value
        standard_constant = self.hlsl_standard_math_constant_expression(name)
        if standard_constant is not None:
            return standard_constant
        lane_mask = hlsl_subgroup_lane_mask_expression(self, name)
        if lane_mask is not None:
            return lane_mask
        self.hlsl_resource_pointer_array_bare_expression_error(expr)
        self.hlsl_workgroup_pointer_bare_expression_error(expr)
        return self.hlsl_identifier_name(name)

    def generate_hlsl_variable_expression(
        self,
        expr,
    ):
        """Render a variable reference."""
        if self.is_hlsl_builtin_option_none_expression(expr):
            none_value = self.generate_hlsl_builtin_option_none_value()
            if none_value is not None:
                return none_value
        unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
        if unsupported_value is not None:
            return unsupported_value
        enum_value = enum_value_expression(self, expr.name)
        if enum_value is not None:
            return enum_value
        standard_constant = self.hlsl_standard_math_constant_expression(expr.name)
        if standard_constant is not None:
            return standard_constant
        lane_mask = hlsl_subgroup_lane_mask_expression(self, expr.name)
        if lane_mask is not None:
            return lane_mask
        self.hlsl_resource_pointer_array_bare_expression_error(expr)
        self.hlsl_workgroup_pointer_bare_expression_error(expr)
        return self.hlsl_identifier_name(expr.name)

    def generate_hlsl_binary_expression(
        self,
        expr,
    ):
        """Render a binary operator expression."""
        op = getattr(expr, "operator", getattr(expr, "op", "+"))
        mapped_op = self.map_operator(op)
        if mapped_op == ",":
            left = self.generate_expression_with_expected(expr.left, None)
            right = self.generate_expression_with_expected(expr.right, None)
            sequence = f"({left}, {right})"
            result_type = self.expression_result_type(expr.right)
            if self.hlsl_struct_constructor_fields(result_type) is not None:
                # Explicit same-type construction distinguishes an aggregate
                # value sequence from a misplaced HLSL initializer list.
                return f"{self.map_type(result_type)}({sequence})"
            return sequence
        if mapped_op in {"+", "-"}:
            private_pointer_binding = self.hlsl_private_pointer_view_binding(expr)
            if private_pointer_binding is not None and private_pointer_binding.get(
                "root_kind"
            ) in {"parameter", "scalar_parameter", "word_view_parameter"}:
                parameter_name = next(
                    (
                        name
                        for name in self.current_hlsl_private_pointer_base_names
                        if self.hlsl_identifier_name(name)
                        == private_pointer_binding.get("backing")
                    ),
                    None,
                )
                raise DirectXPrivatePointerParameterError(
                    "DirectX cannot emit fixed private array parameter "
                    "arithmetic as a first-class pointer expression",
                    function_name=self.current_function_name,
                    parameter_name=parameter_name,
                    reason="bare-pointer-expression",
                    source_location=getattr(expr, "source_location", None),
                )
        if mapped_op in {"==", "!=", "<", "<=", ">", ">=", "&&", "||", "<<", ">>"}:
            previous_expected_type = self.current_expression_expected_type
            self.current_expression_expected_type = None
            try:
                left = self.generate_expression(getattr(expr, "left", ""))
                right = self.generate_expression(getattr(expr, "right", ""))
            finally:
                self.current_expression_expected_type = previous_expected_type
        else:
            left = self.generate_expression(getattr(expr, "left", ""))
            right = self.generate_expression(getattr(expr, "right", ""))
        if mapped_op in {"<<", ">>"} and self.hlsl_untyped_int_expression(expr.left):
            # Fix the source operand type before shifting; the count and
            # enclosing expression must not choose its width or signedness.
            left = f"int({left})"
        option_none_comparison = self.generate_hlsl_builtin_option_none_comparison(
            getattr(expr, "left", ""),
            mapped_op,
            getattr(expr, "right", ""),
        )
        if option_none_comparison is not None:
            return option_none_comparison
        self_comparison = self.hlsl_floating_self_comparison(expr, left, mapped_op)
        if self_comparison is not None:
            return self_comparison
        if mapped_op == "-":
            precise_difference = self.hlsl_precise_difference_call(expr, left, right)
            if precise_difference is not None:
                return precise_difference
        if self.hlsl_binary_multiply_uses_mul(expr):
            return f"mul({left}, {right})"
        complex_binary = self.hlsl_complex64_binary_expression(
            expr,
            left,
            right,
            mapped_op,
        )
        if complex_binary is not None:
            return complex_binary
        bfloat_binary = self.hlsl_bfloat16_binary_expression(
            expr,
            left,
            right,
            mapped_op,
        )
        if bfloat_binary is not None:
            return bfloat_binary
        native_16_bit_binary = self.hlsl_native_16_bit_binary_expression(
            expr,
            left,
            right,
            mapped_op,
        )
        if native_16_bit_binary is not None:
            return native_16_bit_binary
        integer_floating = self.hlsl_integer_floating_binary_expression(
            expr,
            left,
            right,
            mapped_op,
        )
        if integer_floating is not None:
            return integer_floating
        wide_integer = self.hlsl_wide_integer_binary_expression(
            expr, left, right, mapped_op
        )
        if wide_integer is not None:
            return wide_integer
        bool_arithmetic = self.hlsl_bool_arithmetic_expression(
            expr,
            left,
            right,
            mapped_op,
        )
        if bool_arithmetic is not None:
            return bool_arithmetic
        minimum_precision_integer = (
            self.hlsl_minimum_precision_integer_binary_expression(
                expr,
                left,
                right,
                mapped_op,
            )
        )
        if minimum_precision_integer is not None:
            return minimum_precision_integer
        return f"({left} {mapped_op} {right})"

    def generate_hlsl_unary_expression(
        self,
        expr,
    ):
        """Render a unary operator expression."""
        pointer_array_base_unary = self.generate_hlsl_resource_pointer_array_base_unary(
            expr,
            statement_context=False,
        )
        if pointer_array_base_unary is not None:
            return pointer_array_base_unary
        self.generate_hlsl_resource_pointer_step(expr, statement_context=False)
        op = getattr(expr, "operator", getattr(expr, "op", "+"))
        mapped_op = self.map_operator(op)
        if mapped_op in {"++", "--"}:
            self.hlsl_tall_matrix_rows.mark_chain(getattr(expr, "operand", None))
            self.hlsl_private_pointer_word_view_write_error(
                getattr(expr, "operand", None)
            )
            ancestor = self.hlsl_union_access_ancestor(getattr(expr, "operand", None))
            if ancestor is not None and ancestor["member"]["kind"] not in {
                "uint32",
                "uint32_array",
            }:
                raise self.hlsl_union_usage_error(
                    ancestor,
                    expr,
                    "interpreted-increment-unsupported",
                    "increment and decrement require a canonical union lvalue",
                )
        if mapped_op == "*" and not getattr(expr, "is_postfix", False):
            postincrement_load = self.hlsl_resource_pointer_postincrement_load(expr)
            if postincrement_load is not None:
                return postincrement_load
            aggregate = self.generate_hlsl_private_scalar_struct_reinterpret_read(expr)
            if aggregate is not None:
                return aggregate
            private_pointee = self.generate_hlsl_private_pointer_view_access(
                expr.operand, access_expression=expr
            )
            if private_pointee is not None:
                return private_pointee
            pointee = self.generate_hlsl_resource_pointer_access(expr.operand)
            if pointee is not None:
                return pointee
        operand = self.generate_expression(getattr(expr, "operand", ""))
        if mapped_op in {"++", "--"}:
            byte_update = self.hlsl_byte_update_expression(
                expr, expr.operand, mapped_op
            )
            if byte_update is not None:
                return byte_update
        complex_unary = self.hlsl_complex64_unary_expression(
            expr,
            operand,
            mapped_op,
        )
        if complex_unary is not None:
            return complex_unary
        bfloat_unary = self.hlsl_bfloat16_unary_expression(
            expr,
            operand,
            mapped_op,
        )
        if bfloat_unary is not None:
            return bfloat_unary
        narrow_scalar_unary = self.hlsl_narrow_scalar_unary_expression(
            expr, operand, mapped_op
        )
        if narrow_scalar_unary is not None:
            return narrow_scalar_unary
        if mapped_op == "-" and not isinstance(expr.operand, LiteralNode):
            floating = self.hlsl_floating_arithmetic_type_info(
                self.hlsl_source_expression_type(expr.operand)
            )
            if floating is not None and floating["base_type"] == "float":
                # Arithmetic negation can flush subnormals under DirectX FTZ.
                # Toggle the sign without arithmetic or repeated evaluation.
                width = floating["width"]
                mask = "0x80000000u"
                if width != 1:
                    mask = f"uint{width}({', '.join([mask] * width)})"
                return f"asfloat(asuint({operand}) ^ {mask})"
        if mapped_op in {"++", "--"} and getattr(expr, "is_postfix", False):
            return f"{operand}{mapped_op}"
        if mapped_op in {"+", "-"} and operand.startswith(("+", "-")):
            # Keep adjacent signs from becoming an increment or decrement.
            operand = f"({operand})"
        return f"{mapped_op}{operand}"

    def generate_hlsl_array_access_expression(
        self,
        expr,
    ):
        """Render a subscript expression."""
        array_expr = getattr(expr, "array_expr", getattr(expr, "array", ""))
        index_expr = getattr(expr, "index_expr", getattr(expr, "index", ""))
        union_array_read = self.generate_hlsl_union_array_read(expr)
        if union_array_read is not None:
            return union_array_read
        pointer_array_binding = self.hlsl_resource_pointer_array_for_access(expr)
        if pointer_array_binding is not None:
            self.hlsl_resource_pointer_array_index(pointer_array_binding, index_expr)
            raise self.hlsl_resource_pointer_array_error(
                "DirectX cannot emit a selected storage pointer array element "
                f"from '{pointer_array_binding['array_name']}' as a first-class "
                "pointer",
                binding=pointer_array_binding,
                node=expr,
                reason="pointer-element-escape",
            )
        private_pointer_access = self.generate_hlsl_private_pointer_view_access(
            array_expr, index_expr, access_expression=expr
        )
        if private_pointer_access is not None:
            return private_pointer_access
        pointer_access = None
        if (
            self.hlsl_resource_pointer_alias_expression(array_expr)
            or isinstance(array_expr, PointerReinterpretNode)
            or isinstance(array_expr, BinaryOpNode)
            and array_expr.op in {"+", "-"}
        ):
            pointer_access = self.generate_hlsl_resource_pointer_access(
                array_expr, index_expr
            )
        if pointer_access is not None:
            return pointer_access
        unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
        if unsupported_value is not None:
            return unsupported_value
        block_load = self.generate_glsl_buffer_block_array_load(expr)
        if block_load is not None:
            return block_load
        fixed_array_access = self.generate_hlsl_fixed_array_return_access(
            array_expr,
            index_expr,
        )
        if fixed_array_access is not None:
            return fixed_array_access
        matrix_access = self.hlsl_matrix_orientation.subscript(array_expr, index_expr)
        if matrix_access is not None:
            return matrix_access
        if isinstance(array_expr, FunctionCallNode) and self.expression_name(
            array_expr.name
        ) in {"imageLoad", "texelFetch"}:
            # The subscript selects a component of the whole texel, so the
            # element's context does not narrow the load.
            array = self.generate_expression_with_expected(array_expr, None)
        else:
            array = self.generate_expression(array_expr)
        index = self.generate_expression(index_expr)
        index = self.hlsl_resource_pointer_offset_index(array, index, array_expr)
        index = self.hlsl_resource_index_expression(
            array_expr,
            index_expr,
            index,
            rendered_resource=array,
        )
        array = (
            self.hlsl_tall_matrix_rows.row_read(array_expr, index_expr, array) or array
        )
        return f"{array}[{index}]"

    def generate_hlsl_constructor_expression(
        self,
        expr,
    ):
        """Render a constructor expression."""
        if getattr(expr, "is_braced_constructor", False):
            vector_initializer = self.hlsl_vector_aggregate_initializer(
                ArrayLiteralNode(list(expr.arguments)),
                self.map_type(expr.constructor_type),
            )
            if vector_initializer is not None:
                return vector_initializer
        registered_conversion = self.hlsl_registered_structure_scalar_constructor_call(
            self.type_name_string(getattr(expr, "constructor_type", None)),
            list(getattr(expr, "arguments", []) or []),
            source_location=getattr(expr, "source_location", None),
        )
        if registered_conversion is not None:
            return registered_conversion
        enum_constructor = generate_enum_constructor_expression(self, expr)
        if enum_constructor is not None:
            return enum_constructor
        self.validate_hlsl_struct_constructor_expression_context(expr)
        constructor = generate_struct_constructor_expression(self, expr)
        if constructor is not None:
            return constructor
        return str(expr)

    def generate_hlsl_member_access_expression(
        self,
        expr,
    ):
        """Render a member access."""
        union_member_read = self.generate_hlsl_union_member_read(expr)
        if union_member_read is not None:
            return union_member_read
        struct_buffer_access = self.hlsl_struct_buffer_member_access_expression(expr)
        if struct_buffer_access is not None:
            return struct_buffer_access
        lowered_resource_member = self.hlsl_lowered_struct_resource_member_global_name(
            expr
        )
        if lowered_resource_member is not None:
            return lowered_resource_member
        unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
        if unsupported_value is not None:
            return unsupported_value
        block_load = self.generate_glsl_buffer_block_member_load(expr)
        if block_load is not None:
            return block_load
        obj_expr = getattr(expr, "object_expr", getattr(expr, "object", ""))
        member = getattr(expr, "member", "")
        previous_extent = self.hlsl_image_load_swizzle_extent
        self.hlsl_image_load_swizzle_extent = image_load_swizzle_extent(
            obj_expr, member, self.expression_name
        )
        try:
            obj = self.generate_expression_with_expected(obj_expr, None)
        finally:
            self.hlsl_image_load_swizzle_extent = previous_extent
        return f"{obj}.{member}"

    def generate_hlsl_ternary_expression(
        self,
        expr,
    ):
        """Render a conditional expression."""
        expected_type = self.current_expression_expected_type
        aggregate_type = self.hlsl_aggregate_conditional_type(expr, expected_type)
        if aggregate_type is not None:
            self.hlsl_aggregate_conditional_error(
                expr,
                aggregate_type=aggregate_type,
                context="expression",
                reason="statement-context-required",
                detail=(
                    "HLSL aggregate selection requires a temporary or direct "
                    "statement control flow"
                ),
            )
        half_selection = self.generate_hlsl_half_selection(expr)
        if half_selection is not None:
            return half_selection
        source_type = self.expression_result_type(expr)
        source_info = self.hlsl_floating_arithmetic_type_info(source_type)
        if (source_info is not None and source_info["base_type"] == "float16_t") or any(
            self.is_hlsl_bfloat16_type(self.expression_result_type(arm))
            for arm in (expr.true_expr, expr.false_expr)
        ):
            # Convert arms to the logical common type before selection;
            # the enclosing context converts the selected value once.
            expected_type = source_type
        condition = self.hlsl_precise_condition(
            getattr(expr, "condition", None),
            self.generate_expression_with_expected(
                getattr(expr, "condition", ""), "bool"
            ),
        )
        wide_integer_contract = self.hlsl_wide_integer_arithmetic_contract(
            expr,
            "?:",
            self.expression_result_type(getattr(expr, "true_expr", None)),
            self.expression_result_type(getattr(expr, "false_expr", None)),
        )
        if wide_integer_contract is not None:
            expected_type = None
        true_expr = self.generate_expression_with_expected(
            getattr(expr, "true_expr", ""), expected_type
        )
        false_expr = self.generate_expression_with_expected(
            getattr(expr, "false_expr", ""), expected_type
        )
        if wide_integer_contract is not None:
            true_expr, false_expr = self.hlsl_wide_integer_operands(
                wide_integer_contract, true_expr, false_expr
            )
        return f"({condition} ? {true_expr} : {false_expr})"
