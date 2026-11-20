"""Result types of binary, subscript, member-access and call expressions for ``HLSLCodeGen.expression_result_type``."""

import re

from ..arithmetic_conversions import ArithmeticScalarKind
from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    MemberAccessNode,
)
from ..unhandled import UNHANDLED
from .array_utils import split_array_type_suffix
from .builtin_result_types import (
    C_FAMILY,
    component_access_type,
)
from .directx_call_result_types import HLSLCallResultTypeMixin
from .glsl_buffer_array_values import glsl_buffer_array_value_type
from .glsl_buffer_block_arrays import (
    glsl_buffer_indexed_value_type,
    glsl_buffer_whole_array_reason,
)


class HLSLResultTypeMixin(HLSLCallResultTypeMixin):
    """Expression-family result types of ``HLSLCodeGen.expression_result_type``."""

    def hlsl_binary_expression_result_type(
        self,
        expr,
    ):
        """Return the HLSL result type of a binary expression, or ``UNHANDLED`` when ``expr`` is not one."""
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            operator = str(getattr(expr, "operator", getattr(expr, "op", "")))
            if operator == ",":
                return right_type
            if operator in {"==", "!=", "<", "<=", ">", ">=", "&&", "||"}:
                return self.hlsl_boolean_expression_result_type(left_type, right_type)
            pointer_type = self.hlsl_pointer_offset_result_type(
                operator, left_type, right_type
            )
            if pointer_type is not None:
                return pointer_type
            if self.is_hlsl_complex64_type(left_type) or self.is_hlsl_complex64_type(
                right_type
            ):
                return "complex64_t"
            vector_result_type = self.hlsl_bfloat16_vector_binary_result_type(
                left_type, right_type
            )
            if vector_result_type is not None:
                return vector_result_type
            left_is_bfloat = self.is_hlsl_bfloat16_type(left_type)
            right_is_bfloat = self.is_hlsl_bfloat16_type(right_type)
            if left_is_bfloat or right_is_bfloat:
                if left_is_bfloat and right_is_bfloat:
                    return left_type or right_type
                other_type = right_type if left_is_bfloat else left_type
                mapped_other = self.map_type(other_type)
                if mapped_other in {
                    "double",
                    "float",
                    "half",
                    "min10float",
                    "min16float",
                }:
                    return other_type
                return "float"
            integer_floating = self.hlsl_integer_floating_binary_contract(
                expr,
                operator,
                left_type,
                right_type,
            )
            if integer_floating is not None:
                return integer_floating["result_type"]
            mapped_operator = self.map_operator(operator)
            native_16_bit_contract = self.hlsl_native_16_bit_arithmetic_contract(
                expr,
                mapped_operator,
                left_type,
                right_type,
            )
            if native_16_bit_contract is not None:
                return native_16_bit_contract["operation_type"]
            wide_integer_contract = self.hlsl_wide_integer_arithmetic_contract(
                expr, mapped_operator, left_type, right_type
            )
            if wide_integer_contract is not None:
                return wide_integer_contract["operation_type"]
            if mapped_operator in {"<<", ">>"}:
                left_integer = self.hlsl_integer_arithmetic_type_info(left_type)
                right_integer = self.hlsl_integer_arithmetic_type_info(right_type)
                if (
                    left_integer is not None
                    and right_integer is not None
                    and left_integer["width"] == right_integer["width"] == 1
                ):
                    return self.hlsl_promoted_integer_arithmetic_base_type(left_integer)
            if mapped_operator in {"/", "%"}:
                integer_contract = (
                    self.hlsl_minimum_precision_integer_operation_contract(
                        expr,
                        mapped_operator,
                        left_type,
                        right_type,
                        context="binary expression",
                    )
                )
                if integer_contract is not None:
                    width = integer_contract["result_width"]
                    suffix = "" if width == 1 else str(width)
                    return f"{integer_contract['base_type']}{suffix}"
            if mapped_operator == "*":
                product_type = self.hlsl_matrix_orientation.product_type(
                    left_type, right_type
                )
                if product_type is not None:
                    return product_type
            matrix_type = self.hlsl_matrix_orientation.arithmetic_type(
                mapped_operator, left_type, right_type
            )
            if matrix_type is not None:
                return matrix_type
            if self.is_vector_value_type(left_type):
                return left_type
            if self.is_vector_value_type(right_type):
                return right_type
            if left_type == "float" or right_type == "float":
                return "float"
            left_mapped = self.map_type(left_type)
            right_mapped = self.map_type(right_type)
            if "uint64_t" in {left_mapped, right_mapped}:
                return "uint64_t"
            if "int64_t" in {left_mapped, right_mapped}:
                return "int64_t"
            if "uint" in {left_mapped, right_mapped}:
                return "uint"
            if "int" in {left_mapped, right_mapped}:
                return "int"
            return left_type or right_type
        return UNHANDLED

    def hlsl_access_expression_result_type(
        self,
        expr,
    ):
        """Return the HLSL result type of a subscript or member-access expression, or ``UNHANDLED`` when ``expr`` is neither."""
        if isinstance(expr, ArrayAccessNode):
            block_access = self.glsl_buffer_block_array_access(expr)
            if block_access is not None:
                return glsl_buffer_indexed_value_type(block_access)
            array_type = self.type_name_string(self.expression_result_type(expr.array))
            if not array_type:
                array_type = self.unsupported_glsl_buffer_block_expression_type(
                    expr.array
                )
            array_metadata = self.hlsl_fixed_array_return_struct_metadata.get(
                self.map_type(array_type)
            )
            if array_metadata is not None:
                return array_metadata["base_type"]
            if array_type and "[" in array_type and "]" in array_type:
                # Indexing removes the outermost extent only: g[i] of int[3][2] is int[2].
                base_type, array_suffix = split_array_type_suffix(array_type)
                inner = re.fullmatch(r"\[[^\]]*\](.*)", array_suffix or "")
                return f"{base_type}{inner.group(1)}" if inner else base_type
            buffer_element_type = self.hlsl_typed_buffer_element_type(
                array_type,
                {"Buffer", "RWBuffer", "StructuredBuffer", "RWStructuredBuffer"},
            )
            if buffer_element_type is not None:
                return buffer_element_type
            pointer_element_type = self.hlsl_pointer_element_type(array_type)
            if pointer_element_type is not None:
                return pointer_element_type
            byte_shape = self.hlsl_byte_integer_shape(array_type)
            if byte_shape is not None and byte_shape[2] > 1:
                return (
                    "int8"
                    if byte_shape[0] is ArithmeticScalarKind.SIGNED_INTEGER
                    else "uint8"
                )
            column_type = self.hlsl_matrix_orientation.column_type(array_type)
            if column_type is not None:
                return column_type
            component_type = component_access_type(array_type, C_FAMILY)
            if component_type is not None:
                return component_type
            return array_type
        if isinstance(expr, MemberAccessNode):
            block_access = self.glsl_buffer_block_member_access(expr)
            if block_access is not None:
                if glsl_buffer_whole_array_reason(block_access, indexed=False):
                    return glsl_buffer_array_value_type(block_access)
                return block_access["type"]
            object_type = self.expression_result_type(
                expr.object
            ) or self.unsupported_glsl_buffer_block_expression_type(expr.object)
            element_type = self.hlsl_typed_buffer_element_type(
                object_type, {"StructuredBuffer", "RWStructuredBuffer"}
            )
            if element_type is not None:
                object_type = element_type
            member = str(expr.member)
            if object_type and all(ch in "xyzwrgba" for ch in member):
                byte_shape = self.hlsl_byte_integer_shape(object_type)
                component_type = (
                    (
                        "char"
                        if byte_shape[0] is ArithmeticScalarKind.SIGNED_INTEGER
                        else "uchar"
                    )
                    if byte_shape is not None and byte_shape[2] > 1
                    else self.vector_component_type(object_type)
                )
                if component_type and len(member) == 1:
                    return component_type
                if component_type:
                    return f"{component_type}{len(member)}"
            if object_type:
                member_type = self.struct_member_types.get(
                    self.type_name_string(object_type), {}
                ).get(member)
                if member_type:
                    return member_type
            member_types = {
                self.type_name_string(members[member])
                for members in self.struct_member_types.values()
                if member in members
            }
            if len(member_types) == 1:
                return next(iter(member_types))
            return None
        return UNHANDLED

    def hlsl_call_expression_result_type(
        self,
        expr,
    ):
        """Return the HLSL result type of a function call, or ``UNHANDLED`` when ``expr`` is not a call or no call rule applies."""
        if not isinstance(expr, FunctionCallNode):
            return UNHANDLED
        ray_query_call = self.hlsl_ray_query_call_parts(expr)
        if ray_query_call is not None:
            operation, _query_expr, _args = ray_query_call
            return self.hlsl_ray_query_method_return_type(operation)

        func_expr = getattr(expr, "function", None) or getattr(expr, "name", None)
        func_name = getattr(func_expr, "name", func_expr)
        args = getattr(expr, "arguments", getattr(expr, "args", []))
        handled = self.hlsl_method_call_result_type(expr, args, func_expr, func_name)
        if handled is not UNHANDLED:
            return handled
        handled = self.hlsl_math_call_result_type(expr, args, func_name)
        if handled is not UNHANDLED:
            return handled
        handled = self.hlsl_bit_and_constructor_call_result_type(args, func_name)
        if handled is not UNHANDLED:
            return handled
        handled = self.hlsl_function_and_resource_call_result_type(args, func_name)
        if handled is not UNHANDLED:
            return handled
        return UNHANDLED
