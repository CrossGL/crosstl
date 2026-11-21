"""Calls that ``DirectXCallExpressionMixin.generate_hlsl_function_call_expression`` renders as an HLSL built-in form, a lowering or a typed refusal before it resolves a user function."""

from ..unhandled import (
    UNHANDLED,
)
from .bitfield_builtins import (
    BITFIELD_ARITIES,
    hlsl_bitfield_call,
)
from .boolean_mix import (
    boolean_mix_selector,
    hlsl_boolean_mix,
)
from .enum_utils import (
    generate_enum_constructor_call,
)
from .exponent_builtins import (
    hlsl_exponent_call,
)
from .extended_arithmetic_builtins import (
    hlsl_extended_arithmetic_call,
)
from .generic_function_utils import (
    generate_static_generic_numeric_call,
)
from .hlsl_byte_address_atomics import (
    hlsl_byte_address_atomic_expression,
)
from .pack_builtins import (
    PACK_FUNCTIONS,
    hlsl_pack_call,
)
from .sizeof_calls import (
    sizeof_call_bytes,
)


class DirectXBuiltinCallFormMixin:
    """Built-in call forms of ``DirectXCallExpressionMixin.generate_hlsl_function_call_expression``."""

    def generate_hlsl_storage_and_constructor_call(
        self,
        expr,
        args,
        func_expr,
        func_name,
    ):
        """Render a call that a union, byte-address, buffer-length, size, generic numeric, enum, constructor, synchronization or interpolation form handles, or return ``UNHANDLED``."""
        union_small_vector_set = self.generate_hlsl_union_small_vector_set_call(
            func_name, args
        )
        if union_small_vector_set is not None:
            return union_small_vector_set

        byteaddress_interlocked_call = (
            self.generate_hlsl_byteaddress_interlocked_member_call(
                func_expr,
                args,
            )
        )
        if byteaddress_interlocked_call is not None:
            return byteaddress_interlocked_call

        block_length_call = self.generate_glsl_buffer_block_length_call(func_expr, args)
        if block_length_call is not None:
            return block_length_call
        matrix_length = self.hlsl_matrix_orientation.length_call(func_expr, args)
        if matrix_length is not None:
            return matrix_length

        byteaddress_member_call = self.generate_hlsl_byteaddress_member_call(
            func_expr,
            args,
        )
        if byteaddress_member_call is not None:
            return byteaddress_member_call

        sizeof_bytes = sizeof_call_bytes(self, func_name, args, "directx")
        if sizeof_bytes is not None:
            return str(sizeof_bytes)

        static_generic_call = generate_static_generic_numeric_call(self, func_name)
        if static_generic_call is not None:
            return static_generic_call

        enum_constructor = generate_enum_constructor_call(self, func_name, args)
        if enum_constructor is not None:
            return enum_constructor

        registered_conversion = self.hlsl_registered_structure_scalar_constructor_call(
            func_name,
            args,
            source_location=getattr(expr, "source_location", None),
        )
        if registered_conversion is not None:
            return registered_conversion

        bfloat_constructor = self.hlsl_bfloat16_constructor_call(
            func_name,
            args,
            source_location=getattr(expr, "source_location", None),
        )
        if bfloat_constructor is not None:
            return bfloat_constructor

        complex64_constructor = self.hlsl_complex64_constructor_call(func_name, args)
        if complex64_constructor is not None:
            return complex64_constructor

        self.validate_hlsl_struct_constructor_expression_context(expr)
        struct_constructor = self.generate_hlsl_struct_constructor_call(
            func_name,
            args,
            getattr(expr, "named_arguments", None),
        )
        if struct_constructor is not None:
            return struct_constructor

        unsupported_call = self.unsupported_glsl_buffer_block_function_call(func_name)
        if unsupported_call is not None:
            return unsupported_call

        input_attachment_call = self.unsupported_input_attachment_call(func_name)
        if input_attachment_call is not None:
            return input_attachment_call

        synchronization_call = self.synchronization_function_call(
            func_name,
            args,
            source_location=getattr(expr, "source_location", None),
        )
        if synchronization_call is not None:
            return synchronization_call

        interpolation_call = self.interpolation_function_call(func_name, args)
        if interpolation_call is not None:
            return interpolation_call
        return UNHANDLED

    def generate_hlsl_rewritten_intrinsic_call(
        self,
        expr,
        args,
        func_name,
    ):
        """Render a call that a packing, bit-field, exponent, arithmetic, subgroup, Metal, math, bit-cast, texture, atomic, buffer or matrix rewrite handles, or return ``UNHANDLED``."""
        if func_name in PACK_FUNCTIONS and func_name not in getattr(
            self, "function_return_types", {}
        ):
            return hlsl_pack_call(
                self, func_name, [self.generate_expression(arg) for arg in args]
            )
        if func_name in BITFIELD_ARITIES and func_name not in getattr(
            self, "function_return_types", {}
        ):
            return hlsl_bitfield_call(
                self,
                func_name,
                args,
                [self.generate_expression(arg) for arg in args],
            )
        exponent_refusal = hlsl_exponent_call(self, func_name, args)
        if exponent_refusal is not None:
            return exponent_refusal
        extended_arithmetic = hlsl_extended_arithmetic_call(self, func_name, args)
        if extended_arithmetic is not None:
            return extended_arithmetic
        boolean_mix = boolean_mix_selector(
            self, func_name, args, getattr(self, "function_return_types", {})
        )
        if boolean_mix is not None:
            return hlsl_boolean_mix(self, args, boolean_mix)

        if func_name in self.HLSL_WAVE_INTRINSIC_ARITIES:
            return self.generate_hlsl_wave_intrinsic_call(func_name, args)

        metal_simd_call = self.generate_hlsl_metal_simd_shuffle_call(func_name, args)
        if metal_simd_call is not None:
            return metal_simd_call

        metal_as_type_call = self.generate_hlsl_metal_as_type_call(func_name, args)
        if metal_as_type_call is not None:
            return metal_as_type_call

        signbit_call = self.generate_hlsl_signbit_call(
            func_name,
            args,
            call_node=expr,
        )
        if signbit_call is not None:
            return signbit_call

        copysign_call = self.generate_hlsl_copysign_call(
            func_name,
            args,
            call_node=expr,
        )
        if copysign_call is not None:
            return copysign_call

        inverse_hyperbolic_call = self.generate_hlsl_inverse_hyperbolic_call(
            func_name,
            args,
            call_node=expr,
        )
        if inverse_hyperbolic_call is not None:
            return inverse_hyperbolic_call

        precise_hyperbolic_call = self.generate_hlsl_precise_hyperbolic_call(
            func_name,
            args,
            call_node=expr,
        )
        if precise_hyperbolic_call is not None:
            return precise_hyperbolic_call

        atan2_call = self.generate_hlsl_atan2_call(func_name, args, call_node=expr)
        if atan2_call is not None:
            return atan2_call

        bitcast_call = self.generate_hlsl_bitcast_call(func_name, args)
        if bitcast_call is not None:
            return bitcast_call

        texture_call = self.generate_texture_call(func_name, args, call_node=expr)
        if texture_call is not None:
            return texture_call

        glsl_block_atomic_call = self.generate_glsl_buffer_block_atomic_call(
            func_name, args
        )
        if glsl_block_atomic_call is not None:
            return glsl_block_atomic_call

        byte_address_atomic_call = hlsl_byte_address_atomic_expression(
            self, func_name, args
        )
        if byte_address_atomic_call is not None:
            return byte_address_atomic_call

        typed_buffer_atomic_call = self.generate_hlsl_typed_buffer_atomic_expression(
            func_name, args
        )
        if typed_buffer_atomic_call is not None:
            return typed_buffer_atomic_call

        buffer_call = self.generate_buffer_call(func_name, args)
        if buffer_call is not None:
            return buffer_call

        inverse_call = self.hlsl_inverse_function_call(func_name, args)
        if inverse_call is not None:
            return inverse_call
        matrix_builtin = self.hlsl_matrix_orientation.builtin_call(func_name, args)
        if matrix_builtin is not None:
            return matrix_builtin
        return UNHANDLED
