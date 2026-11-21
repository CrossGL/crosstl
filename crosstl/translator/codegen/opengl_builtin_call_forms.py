"""Calls that ``OpenGLCallExpressionMixin.generate_glsl_function_call_expression`` renders as a GLSL built-in form, a lowering or a typed refusal before it resolves a user function: matrix, logical, mesh, ray-query, synchronization, subgroup, atomic, math, bit-cast, packing, texture, interpolation, derivative and buffer calls, and type constructors."""

import re

from ..ast import (
    MemberAccessNode,
)
from ..unhandled import (
    UNHANDLED,
)
from .exponent_builtins import (
    glsl_exponent_call,
)
from .extended_arithmetic_builtins import (
    glsl_extended_arithmetic_call,
)
from .generic_function_utils import (
    generate_static_generic_numeric_call,
)
from .glsl_double_literals import binary64_constructor_argument
from .glsl_tessellation_input_arrays import (
    tessellation_input_length,
)
from .opengl_operand_forms import glsl_operand_form_call
from .sizeof_calls import (
    sizeof_call_bytes,
)
from .texture_lod_clamp import (
    glsl_texture_lod_clamp_call,
    is_texture_lod_clamp_operation,
)
from .texture_residency import (
    TEXTURE_RESIDENCY_OPERATION,
    TEXTURE_RESIDENT_OPERATION,
    glsl_texture_residency_call,
    glsl_texture_resident_call,
)


class OpenGLBuiltinCallFormMixin:
    """Built-in call forms of ``OpenGLCallExpressionMixin.generate_glsl_function_call_expression``."""

    def generate_glsl_lowered_builtin_call(
        self,
        expr,
        func_expr,
        original_func_name,
    ):
        """Render a call that a matrix, logical, mesh, ray-query, synchronization, subgroup, atomic or math lowering handles, or return ``UNHANDLED``."""
        operand_form_call = glsl_operand_form_call(self, expr, original_func_name)
        if operand_form_call is not UNHANDLED:
            return operand_form_call
        emulated_matrix_call = self.generate_glsl_emulated_matrix_call(
            expr, original_func_name, list(expr.args or [])
        )
        if emulated_matrix_call is not None:
            return emulated_matrix_call
        logical_intrinsic = self.generate_glsl_hlsl_logical_intrinsic(
            original_func_name, list(expr.args or [])
        )
        if logical_intrinsic is not None:
            return logical_intrinsic

        mesh_output_counts_call = self.generate_mesh_output_counts_call(
            original_func_name, expr.args
        )
        union = self.union_storage.call(original_func_name, expr.args)
        if union is not None:
            return union
        if mesh_output_counts_call is not None:
            return mesh_output_counts_call
        self.reject_mesh_output_helper_expression_context(original_func_name, expr.args)

        ray_query_member_call = self.ray_query_member_function_call(
            func_expr, expr.args
        )
        if ray_query_member_call is not None:
            return ray_query_member_call
        if (
            isinstance(func_expr, MemberAccessNode)
            and func_expr.member == "length"
            and not expr.args
        ):
            written_length = tessellation_input_length(self, func_expr)
            if written_length is not None:
                return written_length
            instance_member = self.glsl_structured_buffer_array_element_member(
                func_expr.object
            )
            if instance_member is not None:
                # One buffer of an array of storage blocks holds its elements in a member.
                buffer = self.generate_expression(func_expr.object)
                return f"{buffer}.{instance_member}.length()"

        get_dimensions = self.glsl_structured_buffer_get_dimensions(
            func_expr, expr.args
        )
        if get_dimensions is not None:
            return get_dimensions

        synchronization_call = self.synchronization_function_call(
            original_func_name,
            expr.args,
            source_location=getattr(expr, "source_location", None),
        )
        if synchronization_call is not None:
            return synchronization_call

        wave_operation = self.glsl_wave_operation_name(original_func_name)
        if wave_operation in self.GLSL_WAVE_INTRINSIC_ARITIES:
            return self.generate_glsl_wave_operation(
                wave_operation,
                expr.args,
                original_func_name,
                source_location=getattr(expr, "source_location", None),
            )
        if self.is_metal_simd_group_helper_name(original_func_name):
            return self.glsl_wave_diagnostic_expression(
                original_func_name, "has no OpenGL subgroup equivalent"
            )

        hlsl_converted_call = self.generate_glsl_converted_builtin_call(
            original_func_name, expr.args
        )
        if hlsl_converted_call is not None:
            return hlsl_converted_call

        glsl_memory_atomic_call = self.generate_glsl_memory_atomic_call(
            original_func_name, expr.args
        )
        if glsl_memory_atomic_call is not None:
            return glsl_memory_atomic_call

        saturate_call = self.generate_saturate_call(original_func_name, expr.args)
        if saturate_call is not None:
            return saturate_call

        reciprocal_call = self.generate_reciprocal_call(original_func_name, expr.args)
        if reciprocal_call is not None:
            return reciprocal_call

        log10_call = self.generate_glsl_log10_call(original_func_name, expr.args)
        if log10_call is not None:
            return log10_call

        metal_math_call = self.generate_glsl_metal_math_call(
            original_func_name, expr.args, call_node=expr
        )
        if metal_math_call is not None:
            return metal_math_call

        ieee_math_call = self.generate_glsl_ieee_math_call(
            original_func_name, expr.args, call_node=expr
        )
        if ieee_math_call is not None:
            return ieee_math_call

        trailing_zero_call = self.generate_glsl_trailing_zero_call(
            original_func_name,
            expr.args,
            source_location=getattr(expr, "source_location", None),
        )
        if trailing_zero_call is not None:
            return trailing_zero_call

        boolean_order_call = self.glsl_ordered_boolean_minmax_call(
            original_func_name,
            expr.args,
            source_location=getattr(expr, "source_location", None),
        )
        if boolean_order_call is not None:
            return boolean_order_call
        return UNHANDLED

    def generate_glsl_rewritten_builtin_call(
        self,
        expr,
        original_func_name,
    ):
        """Render a call that an exponent, extended-arithmetic, classification, sign, remainder, bit-cast, multiply or packing rewrite handles, or return ``UNHANDLED``."""
        exponent_call = glsl_exponent_call(self, original_func_name, expr.args)
        if exponent_call is not None:
            return exponent_call
        extended_arithmetic_refusal = glsl_extended_arithmetic_call(
            self, original_func_name, expr.args
        )
        if extended_arithmetic_refusal is not None:
            return extended_arithmetic_refusal

        isfinite_call = self.generate_glsl_isfinite_call(
            original_func_name,
            expr.args,
            call_node=expr,
        )
        if isfinite_call is not None:
            return isfinite_call

        signbit_call = self.generate_glsl_signbit_call(
            original_func_name,
            expr.args,
            call_node=expr,
        )
        if signbit_call is not None:
            return signbit_call

        copysign_call = self.generate_glsl_copysign_call(
            original_func_name,
            expr.args,
            call_node=expr,
        )
        if copysign_call is not None:
            return copysign_call

        remainder_call = self.generate_glsl_floating_remainder_call(
            original_func_name, expr.args, call_node=expr
        )
        if remainder_call is not None:
            return remainder_call

        bitcast_call = self.generate_bitcast_call(original_func_name, expr.args)
        if bitcast_call is not None:
            return bitcast_call

        metal_as_type_call = self.generate_metal_as_type_call(
            original_func_name,
            expr.args,
        )
        if metal_as_type_call is not None:
            return metal_as_type_call

        mul_call = self.generate_mul_call(original_func_name, expr.args)
        if mul_call is not None:
            return mul_call

        packed_bytes = self.generate_packed_byte_call(original_func_name, expr.args)
        if packed_bytes is not None:
            return packed_bytes

        mad_call = self.generate_mad_call(original_func_name, expr.args)
        if mad_call is not None:
            return mad_call

        packed_dot = self.generate_packed_dot4add_call(original_func_name, expr.args)
        if packed_dot is not None:
            return packed_dot
        return UNHANDLED

    def generate_glsl_resource_builtin_call(
        self,
        expr,
        func_name,
        original_func_name,
    ):
        """Render a call that a size, generic numeric, enum, texture, interpolation, derivative or buffer rewrite handles, or return ``UNHANDLED``."""
        sizeof_bytes = sizeof_call_bytes(self, original_func_name, expr.args, "opengl")
        if sizeof_bytes is not None:
            return str(sizeof_bytes)

        static_generic_call = generate_static_generic_numeric_call(
            self,
            original_func_name,
        )
        if static_generic_call is not None:
            return static_generic_call

        enum_constructor = self.glsl_enum_constructor_call(
            original_func_name, expr.args
        )
        if enum_constructor is not None:
            return enum_constructor

        if original_func_name == TEXTURE_RESIDENCY_OPERATION:
            return glsl_texture_residency_call(self, expr.args, expr)
        if original_func_name == TEXTURE_RESIDENT_OPERATION:
            return glsl_texture_resident_call(self, expr.args, expr)

        if is_texture_lod_clamp_operation(original_func_name):
            return glsl_texture_lod_clamp_call(
                self, original_func_name, expr.args, expr
            )

        texture_call = self.generate_texture_call(func_name, expr.args)
        if texture_call is not None:
            return texture_call

        interpolation_call = self.generate_interpolation_call(func_name, expr.args)
        if interpolation_call is not None:
            return interpolation_call

        derivative_call = self.generate_derivative_call(func_name, expr.args)
        if derivative_call is not None:
            return derivative_call

        buffer_call = self.generate_buffer_call(func_name, expr.args)
        if buffer_call is not None:
            return buffer_call
        return UNHANDLED

    def generate_glsl_constructor_call(
        self,
        expr,
        func_name,
        original_func_name,
    ):
        """Render a call that names a type constructor, or return ``UNHANDLED``."""
        constructor = self.glsl_constructor_type(func_name)
        if constructor:
            if self.glsl_half_width(original_func_name) is not None:
                for arg in expr.args:
                    info = self.glsl_value_type_info(
                        self.glsl_source_expression_type(arg)
                    )
                    if (
                        info is not None
                        and info["family"] == "float"
                        and info["bits"] > 32
                    ):
                        self.glsl_scalar_conversion_error(
                            expr,
                            info["source"],
                            original_func_name,
                            "half-double-rounding",
                        )
            complex_scalar_constructor = (
                self.glsl_explicit_complex64_scalar_constructor(
                    expr,
                    original_func_name,
                )
            )
            if complex_scalar_constructor is not None:
                return complex_scalar_constructor
            partial_vector_constructor = self.generate_glsl_partial_vector_constructor(
                expr,
                constructor,
                original_func_name,
            )
            if partial_vector_constructor is not None:
                return self.glsl_apply_narrow_float_contract(
                    partial_vector_constructor, original_func_name, expr
                )
            converted_constructor = self.glsl_struct_constructor_conversion(
                expr,
                constructor,
                expr.args,
            )
            if converted_constructor is not None:
                return converted_constructor
            integer = (
                self.glsl_float_to_integer_conversion_info(expr.args[0])
                if len(expr.args) == 1
                else None
            )
            if self.glsl_integer_float_conversion_applies(integer, constructor):
                converted_integer = self.generate_glsl_integer_float_conversion(
                    expr,
                    self.generate_expression_with_expected(expr.args[0], None),
                    integer,
                    constructor,
                )
                return self.glsl_apply_narrow_float_contract(
                    converted_integer, original_func_name, expr
                )
            constructor_fields = list(
                self.glsl_struct_member_types(constructor).values()
            )
            if re.fullmatch(r"[biud]?vec[234]", str(constructor)):
                # Vector constructors take scalars and vectors of any width
                # and component type; GLSL converts their components.
                constructor_fields = []
            args = ", ".join(
                binary64_constructor_argument(constructor, arg)
                or self.generate_expression_with_expected(
                    arg,
                    (
                        constructor_fields[index]
                        if index < len(constructor_fields)
                        else None
                    ),
                )
                for index, arg in enumerate(expr.args)
            )
            generated_constructor = f"{constructor}({args})"
            narrowed = self.glsl_apply_narrow_integer_contract(
                generated_constructor,
                original_func_name,
            )
            return self.glsl_apply_narrow_float_contract(
                narrowed, original_func_name, expr
            )
        return UNHANDLED
