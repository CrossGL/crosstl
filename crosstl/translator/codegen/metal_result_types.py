"""Result types of binary, subscript, member-access, pointer-access and call expressions for ``MetalCodeGen.expression_result_type``."""

import re

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    MemberAccessNode,
    PointerAccessNode,
)
from ..unhandled import (
    UNHANDLED,
)
from .builtin_result_types import (
    C_FAMILY,
    component_access_type,
)
from .glsl_buffer_array_values import (
    glsl_buffer_array_value_type,
)
from .glsl_buffer_block_arrays import (
    glsl_buffer_indexed_value_type,
    glsl_buffer_whole_array_reason,
)
from .metal_call_result_types import MetalCallResultTypeMixin
from .metal_hlsl_matrix import (
    hlsl_emulated_matrix_shape,
)


class MetalResultTypeMixin(MetalCallResultTypeMixin):
    """Expression-family result types of ``MetalCodeGen.expression_result_type``."""

    def metal_binary_expression_result_type(
        self,
        expr,
    ):
        """Return the Metal result type of a binary expression, or ``UNHANDLED`` when ``expr`` is not one."""
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            operator = self.map_operator(getattr(expr, "op", ""))
            emulated_result = self.metal_emulated_matrix_binary_result_type(
                operator, left_type, right_type
            )
            if emulated_result is not None:
                return emulated_result
            pointer_operand = self.pointer_arithmetic_operand(
                expr, (left_type, right_type)
            )
            if pointer_operand is not None:
                return left_type if pointer_operand is expr.left else right_type
            if operator in {"<", ">", "<=", ">=", "==", "!=", "&&", "||"}:
                for candidate_type in (left_type, right_type):
                    mapped_type = self.map_type(candidate_type)
                    component_type = self.vector_component_type(mapped_type)
                    if component_type is None:
                        continue
                    width = mapped_type[-1] if mapped_type[-1:].isdigit() else ""
                    if width:
                        return f"bool{width}"
                return "bool"
            matrix_result = self.metal_matrix_binary_result_type(
                operator, left_type, right_type
            )
            if matrix_result is not None:
                return matrix_result
            if (
                self.is_vector_value_type(left_type)
                and self.is_vector_value_type(right_type)
                and operator not in {"<<", ">>"}
            ):
                left_mapped = self.map_type(left_type)
                right_mapped = self.map_type(right_type)
                left_component = self.vector_component_type(left_mapped)
                right_component = self.vector_component_type(right_mapped)
                if (
                    left_component != right_component
                    and self.vector_value_width(left_mapped)
                    == self.vector_value_width(right_mapped)
                    and left_component in self.METAL_ARITHMETIC_COMPONENT_RANKS
                    and right_component in self.METAL_ARITHMETIC_COMPONENT_RANKS
                ):
                    common = self.metal_binary_common_component(
                        operator, left_component, right_component
                    )
                    if common is not None:
                        return f"{common}{self.vector_value_width(left_mapped)}"
            if self.is_vector_value_type(left_type) != self.is_vector_value_type(
                right_type
            ):
                # metal_hlsl_values.py: vector op scalar has the common component type.
                vector_type, scalar = (
                    (left_type, expr.right)
                    if self.is_vector_value_type(left_type)
                    else (right_type, expr.left)
                )
                vector_mapped = self.map_type(vector_type)
                component = self.vector_component_type(vector_mapped)
                common = (
                    self.metal_vector_scalar_common_component(
                        operator, component, scalar
                    )
                    if component in self.METAL_ARITHMETIC_COMPONENT_RANKS
                    else None
                )
                if common is not None and common != component:
                    return f"{common}{self.vector_value_width(vector_mapped)}"
            if self.is_vector_value_type(left_type):
                return left_type
            if self.is_vector_value_type(right_type):
                return right_type
            if left_type == "float" or right_type == "float":
                return "float"
            return left_type or right_type
        return UNHANDLED

    def metal_access_expression_result_type(
        self,
        expr,
    ):
        """Return the Metal result type of a subscript, member-access or pointer-access expression, or ``UNHANDLED`` when ``expr`` is none of them."""
        if isinstance(expr, ArrayAccessNode):
            block_access = self.glsl_buffer_block_array_access(expr)
            if block_access is not None:
                return glsl_buffer_indexed_value_type(block_access)
            array_type = self.expression_result_type(expr.array)
            if not array_type:
                array_type = self.unsupported_glsl_buffer_block_expression_type(
                    expr.array
                )
            array_type_name = self.type_name_string(array_type)
            if array_type_name and "[" in array_type_name and "]" in array_type_name:
                return self.metal_for_in_array_element_type(
                    array_type_name, allow_unsized=True
                )
            metal_array_element_type = self.metal_array_element_type(array_type)
            if metal_array_element_type is not None:
                return metal_array_element_type
            if self.is_structured_buffer_type(array_type):
                return self.structured_buffer_element_type(array_type)
            pointee_type = self.pointer_pointee_type_name(array_type)
            if pointee_type is not None:
                return pointee_type
            emulated_matrix = (
                hlsl_emulated_matrix_shape(array_type_name) if array_type_name else None
            )
            if emulated_matrix is not None:
                # Like Metal matrices, indexing selects a column.
                scalar, rows, _columns = emulated_matrix
                return f"{scalar}{rows}"
            if self.is_vector_value_type(array_type):
                native = self.metal_native_narrow_bitcast_storage_type(array_type)
                if native is not None and native[-1:] in {"2", "3", "4"}:
                    return native[:-1]
                return self.vector_component_type(array_type)
            matrix_dimensions = self.metal_matrix_dimensions(
                self.type_name_string(array_type) or ""
            )
            if matrix_dimensions is not None:
                # An MSL floatCxR subscript is a column of R components, as in CrossGL.
                return f"{matrix_dimensions[0]}{matrix_dimensions[2]}"
            component_type = component_access_type(
                self.type_name_string(array_type), C_FAMILY
            )
            if component_type is not None:
                return component_type
            mapped_array_type = (
                self.map_type(array_type) if array_type is not None else None
            )
            if isinstance(mapped_array_type, str):
                if self.is_vector_value_type(mapped_array_type):
                    return self.vector_component_type(mapped_array_type)
                if self.is_matrix_value_type(mapped_array_type):
                    scalar, dimensions = re.fullmatch(
                        r"(float|half|double)(\d)x(\d)", mapped_array_type
                    ).group(1, 3)
                    return f"{scalar}{dimensions}"
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
            object_type = self.member_lookup_type_name(object_type)
            member = str(expr.member)
            if object_type and all(ch in "xyzwrgba" for ch in member):
                native = self.metal_native_narrow_bitcast_storage_type(object_type)
                component_type = (
                    native[:-1]
                    if native is not None and native[-1:] in {"2", "3", "4"}
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
        if isinstance(expr, PointerAccessNode):
            object_type = self.expression_result_type(
                expr.pointer_expr
            ) or self.unsupported_glsl_buffer_block_expression_type(expr.pointer_expr)
            object_type = self.member_lookup_type_name(object_type)
            member = str(expr.member)
            if object_type:
                member_type = self.struct_member_types.get(
                    self.type_name_string(object_type), {}
                ).get(member)
                if member_type:
                    return member_type
            return None
        return UNHANDLED

    def metal_call_expression_result_type(
        self,
        expr,
    ):
        """Return the Metal result type of a function call, or ``UNHANDLED`` when ``expr`` is not a call or no call rule applies."""
        if not isinstance(expr, FunctionCallNode):
            return UNHANDLED
        buffer_source = self.structured_buffer_load_source(expr)
        if buffer_source is not None:
            return self.structured_buffer_element_type(
                self.expression_result_type(buffer_source)
            )
        ray_query_operation = self.ray_query_operation_from_function_call(expr)
        if ray_query_operation is not None:
            return self.metal_ray_query_method_return_type(ray_query_operation)
        func_expr = getattr(expr, "function", None) or getattr(expr, "name", None)
        func_name = getattr(func_expr, "name", func_expr)
        args = getattr(expr, "arguments", getattr(expr, "args", []))
        handled = self.metal_math_call_result_type(expr, args, func_name)
        if handled is not UNHANDLED:
            return handled
        handled = self.metal_conversion_and_texture_call_result_type(args, func_name)
        if handled is not UNHANDLED:
            return handled
        handled = self.metal_function_and_constructor_call_result_type(args, func_name)
        if handled is not UNHANDLED:
            return handled
        return UNHANDLED
