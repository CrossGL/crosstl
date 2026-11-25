"""Calls that ``MetalCallExpressionMixin.generate_metal_function_call_expression`` renders as a Metal built-in form, a lowering, a type constructor or a typed refusal before it emits a user-function call: packing, bit-field, exponent, extended-arithmetic, vector, size, ray-query, enum, atomic, synchronization, wave, mesh, texture, buffer, geometric, matrix, derivative, bit-cast and conversion calls."""

import re

from ..ast import (
    ArrayAccessNode,
    MemberAccessNode,
)
from ..matrix_shape import (
    length_method_object,
)
from ..unhandled import (
    UNHANDLED,
)
from .bitfield_builtins import (
    BITFIELD_ARITIES,
    metal_bitfield_call,
)
from .boolean_mix import (
    boolean_mix_selector,
    metal_boolean_mix,
)
from .enum_utils import (
    generate_enum_constructor_call,
)
from .exponent_builtins import (
    metal_exponent_call,
)
from .extended_arithmetic_builtins import (
    metal_extended_arithmetic_call,
)
from .generic_function_utils import (
    generate_static_generic_numeric_call,
)
from .metal_bit_scans import (
    metal_bit_scan_call,
)
from .metal_constructor_names import (
    METAL_BUILTIN_CONSTRUCTOR_NAMES,
)
from .metal_integer_sign import (
    metal_integer_sign_call,
)
from .pack_builtins import (
    PACK_FUNCTIONS,
    metal_pack_call,
)
from .sizeof_calls import (
    sizeof_call_bytes,
)
from .texture_lod_clamp import (
    is_texture_lod_clamp_operation,
    metal_texture_lod_clamp_call,
)
from .texture_residency import (
    TEXTURE_RESIDENCY_OPERATION,
    TEXTURE_RESIDENT_OPERATION,
    metal_texture_residency_call,
    metal_texture_resident_call,
)


class MetalBuiltinCallFormMixin:
    """Built-in call forms of ``MetalCallExpressionMixin.generate_metal_function_call_expression``."""

    def generate_metal_numeric_builtin_call(
        self,
        expr,
        func_expr,
        func_name,
    ):
        """Render a call that a packing, bit-field, exponent, bit-scan, extended-arithmetic, vector, size, ray-query or static generic rewrite handles, or return ``UNHANDLED``."""
        if func_name in PACK_FUNCTIONS and func_name not in self.user_function_names:
            return metal_pack_call(
                self, func_name, [self.generate_expression(a) for a in expr.args]
            )
        packed_dot = self.generate_packed_dot4add_call(func_name, expr.args)
        if packed_dot is not None:
            return packed_dot
        if func_name in BITFIELD_ARITIES and func_name not in self.user_function_names:
            return metal_bitfield_call(
                self, func_name, [self.generate_expression(a) for a in expr.args]
            )
        boolean_mix = boolean_mix_selector(
            self, func_name, expr.args, self.user_function_names
        )
        if boolean_mix is not None:
            return metal_boolean_mix(self, expr.args, boolean_mix)
        exponent_call = metal_exponent_call(self, func_name, expr.args)
        if exponent_call is not None:
            return exponent_call
        bit_scan_call = metal_bit_scan_call(self, func_name, expr.args)
        if bit_scan_call is not None:
            return bit_scan_call
        integer_sign = metal_integer_sign_call(self, func_name, expr.args)
        if integer_sign is not None:
            return integer_sign
        extended_arithmetic = metal_extended_arithmetic_call(self, func_name, expr.args)
        if extended_arithmetic is not None:
            return extended_arithmetic
        packed_bytes = self.generate_packed_byte_call(func_name, expr.args)
        if packed_bytes is not None:
            return packed_bytes

        vector_set = re.fullmatch(
            r"CrossGLMetalVectorIndex_([iu](?:8|16)vec[234])_set",
            str(func_name),
        )
        if vector_set and len(expr.args) == 3:
            member = expr.args[0]
            while isinstance(member, ArrayAccessNode):
                member = member.array
            if (
                isinstance(member, MemberAccessNode)
                and self.member_lookup_type_name(
                    self.expression_result_type(member.object)
                )
                in self.structs_by_name
            ):
                native = self.metal_native_narrow_bitcast_storage_type(vector_set[1])
                target = self.generate_expression(expr.args[0])
                lane = self.generate_expression_with_expected(expr.args[1], "uint")
                selected = self.generate_expression(expr.args[2])
                return f"({target}[{lane}] = {native[:-1]}({selected}))"

        block_length = self.glsl_buffer_block_length_call(func_expr, expr.args)
        if block_length is not None:
            return block_length
        length_object = length_method_object(func_expr, expr.args)
        matrix_dimensions = (
            self.metal_matrix_dimensions(
                self.expression_result_type(length_object) or ""
            )
            if length_object is not None
            else None
        )
        if matrix_dimensions is not None:
            # A matrix's length() is its column count (GLSL 4.60 section 5.10).
            return f"int({matrix_dimensions[1]})"

        unsupported_table_call = (
            self.unsupported_metal_ray_function_table_array_member_call(func_expr)
        )
        if unsupported_table_call is not None:
            return unsupported_table_call

        ray_query_operation = self.ray_query_operation_from_function_call(expr)
        if ray_query_operation is not None and isinstance(func_expr, MemberAccessNode):
            return self.generate_metal_ray_query_call(
                ray_query_operation,
                getattr(
                    func_expr,
                    "object",
                    getattr(func_expr, "object_expr", None),
                ),
                expr.args,
            )

        sizeof_bytes = sizeof_call_bytes(self, func_name, expr.args, "metal")
        if sizeof_bytes is not None:
            return str(sizeof_bytes)

        static_generic_call = generate_static_generic_numeric_call(self, func_name)
        if static_generic_call is not None:
            return static_generic_call
        return UNHANDLED

    def generate_metal_resource_builtin_call(
        self,
        expr,
        func_name,
    ):
        """Render a call that a shadowing, ray-description, buffer-block, input-attachment, enum, atomic, synchronization, wave, mesh, texture or buffer rewrite handles, or return ``UNHANDLED``."""
        if isinstance(func_name, str) and not self.metal_function_name_is_shadowed(
            func_name
        ):
            matrix_constructor = self.generate_metal_matrix_constructor_override(
                func_name, expr.args
            )
            if matrix_constructor is not None:
                return matrix_constructor
            logical_intrinsic = self.generate_metal_hlsl_logical_intrinsic(
                func_name, expr.args
            )
            if logical_intrinsic is not None:
                return logical_intrinsic

        if func_name == "RayDesc" and self.requires_metal_builtin_ray_desc(func_name):
            self.required_metal_ray_desc_runtime = True
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            return f"CglRayDesc({args})"

        unsupported_call = self.unsupported_glsl_buffer_block_function_call(func_name)
        if unsupported_call is not None:
            return unsupported_call

        input_attachment_call = self.unsupported_input_attachment_call(func_name)
        if input_attachment_call is not None:
            return input_attachment_call

        enum_constructor = generate_enum_constructor_call(self, func_name, expr.args)
        if enum_constructor is not None:
            return enum_constructor

        glsl_block_atomic_call = self.generate_glsl_buffer_block_atomic_call(
            func_name, expr.args
        )
        if glsl_block_atomic_call is not None:
            return glsl_block_atomic_call

        buffer_atomic_call = self.generate_metal_buffer_resource_atomic_call(
            func_name, expr.args
        )
        if buffer_atomic_call is not None:
            return buffer_atomic_call

        synchronization_call = self.synchronization_function_call(
            func_name,
            expr.args,
            source_location=getattr(expr, "source_location", None),
        )
        if synchronization_call is not None:
            return synchronization_call

        wave_call = self.generate_metal_wave_operation(func_name, expr.args)
        if wave_call is not None:
            return wave_call

        atomic_call = self.generate_atomic_function_call(func_name, expr.args)
        if atomic_call is not None:
            return atomic_call

        mesh_output_call = self.generate_metal_mesh_output_call(func_name, expr.args)
        if mesh_output_call is not None:
            return mesh_output_call

        if func_name == TEXTURE_RESIDENCY_OPERATION:
            return metal_texture_residency_call(self, expr.args, expr)
        if func_name == TEXTURE_RESIDENT_OPERATION:
            return metal_texture_resident_call(self, expr.args, expr)

        if is_texture_lod_clamp_operation(func_name):
            return metal_texture_lod_clamp_call(self, func_name, expr.args, expr)

        texture_call = self.generate_texture_call(func_name, expr.args)
        if texture_call is not None:
            return texture_call

        buffer_call = self.generate_buffer_call(func_name, expr.args)
        if buffer_call is not None:
            return buffer_call
        return UNHANDLED

    def generate_metal_math_builtin_call(
        self,
        expr,
        func_name,
    ):
        """Render a call that a geometric, matrix, derivative, bit-cast, integer-bit or angle-conversion rewrite handles, or return ``UNHANDLED``."""
        scalar_geometric_call = self.metal_scalar_geometric_call(func_name, expr.args)
        if scalar_geometric_call is not None:
            return scalar_geometric_call
        if func_name == "normalize" and func_name not in self.user_function_names:
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            call_name = (
                "metal::normalize"
                if self.metal_function_name_is_shadowed("normalize")
                else "normalize"
            )
            return f"{call_name}({args})"
        matrix_builtin = self.metal_matrix_builtins.call(func_name, expr.args)
        if matrix_builtin is not None:
            return matrix_builtin
        if (
            func_name == "inverse"
            and len(expr.args) == 1
            and func_name not in self.user_function_names
        ):
            arg_type = self.map_type(self.expression_result_type(expr.args[0]))
            if arg_type in {"float2x2", "float3x3", "float4x4"}:
                self.required_metal_inverse_helpers.add(arg_type)
                arg = self.generate_expression(expr.args[0])
                return f"__crossgl_inverse_{arg_type}({arg})"
        if (
            func_name == "mod"
            and len(expr.args) == 2
            and func_name not in self.user_function_names
        ):
            left = self.generate_expression(expr.args[0])
            right = self.generate_expression(expr.args[1])
            return f"(({left}) - (({right}) * floor(({left}) / ({right}))))"
        if (
            func_name == "frac"
            and len(expr.args) == 1
            and func_name not in self.user_function_names
        ):
            arg = self.generate_expression(expr.args[0])
            fract_name = (
                "metal::fract"
                if self.metal_function_name_is_shadowed("fract")
                else "fract"
            )
            return f"{fract_name}({arg})"
        if (
            func_name == "lerp"
            and len(expr.args) == 3
            and func_name not in self.user_function_names
        ):
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            mix_name = (
                "metal::mix" if self.metal_function_name_is_shadowed("mix") else "mix"
            )
            return f"{mix_name}({args})"
        if (
            func_name == "atan"
            and len(expr.args) == 2
            and func_name not in self.user_function_names
        ):
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            atan2_name = (
                "metal::atan2"
                if self.metal_function_name_is_shadowed("atan2")
                else "atan2"
            )
            return f"{atan2_name}({args})"
        if (
            func_name in {"inverseSqrt", "inversesqrt"}
            and len(expr.args) == 1
            and func_name not in self.user_function_names
        ):
            arg = self.generate_expression(expr.args[0])
            rsqrt_name = (
                "metal::rsqrt"
                if self.metal_function_name_is_shadowed("rsqrt")
                else "rsqrt"
            )
            return f"{rsqrt_name}({arg})"
        if (
            func_name == "roundEven"
            and len(expr.args) == 1
            and func_name not in self.user_function_names
        ):
            # MSL rint() rounds halfway cases to the nearest even integer (MSL 4.1 section 6.5).
            # It has only float and half overloads, so integer operands are converted to float.
            unified_args = self.generate_metal_unified_builtin_arguments(
                func_name, expr.args
            )
            arg = (
                unified_args[0]
                if unified_args is not None
                else self.generate_expression(expr.args[0])
            )
            rint_name = (
                "metal::rint"
                if self.metal_function_name_is_shadowed("rint")
                else "rint"
            )
            return f"{rint_name}({arg})"
        derivative_name = self.METAL_DERIVATIVE_FUNCTION_ALIASES.get(func_name)
        if (
            derivative_name is not None
            and len(expr.args) == 1
            and func_name not in self.user_function_names
        ):
            arg = self.generate_expression(expr.args[0])
            derivative_call_name = (
                f"metal::{derivative_name}"
                if self.metal_function_name_is_shadowed(derivative_name)
                else derivative_name
            )
            return f"{derivative_call_name}({arg})"
        explicit_as_type = self.generate_metal_explicit_as_type_call(
            func_name, expr.args
        )
        if explicit_as_type is not None:
            return explicit_as_type
        bitcast_call = self.generate_metal_bitcast_call(func_name, expr.args)
        if bitcast_call is not None:
            return bitcast_call
        integer_bit_call = self.generate_metal_integer_bit_call(func_name, expr.args)
        if integer_bit_call is not None:
            return integer_bit_call
        angle_conversion = self.generate_metal_angle_conversion_call(
            func_name, expr.args
        )
        if angle_conversion is not None:
            return angle_conversion
        return UNHANDLED

    def generate_metal_type_constructor_call(
        self,
        expr,
        func_name,
    ):
        """Render a call that constructs a Metal scalar, vector, matrix, registered structure or aggregate value, or return ``UNHANDLED``."""
        if (
            func_name in self.METAL_STDLIB_BUILTIN_FUNCTIONS
            and func_name not in self.user_function_names
        ):
            integer_dot = self.generate_metal_integer_dot_call(func_name, expr.args)
            if integer_dot is not None:
                return integer_dot
            unified_args = self.generate_metal_unified_builtin_arguments(
                func_name, expr.args
            )
            if unified_args is not None:
                args = ", ".join(unified_args)
            elif func_name in self.METAL_BFLOAT_ARGUMENT_PROMOTED_FUNCTIONS:
                args = ", ".join(
                    self.generate_metal_bfloat_promoted_argument(arg)
                    for arg in expr.args
                )
            else:
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            if self.metal_fast_math_call(expr, func_name):
                return f"metal::fast::{func_name}({args})"
            if self.metal_precise_math_call(
                expr, func_name
            ) or self.metal_safe_math_call(expr, func_name):
                return f"metal::precise::{func_name}({args})"
            call_name = (
                f"metal::{func_name}"
                if self.metal_function_name_is_shadowed(func_name)
                else func_name
            )
            return f"{call_name}({args})"
        registered_conversion = self.metal_registered_structure_scalar_conversion(
            expr,
            func_name,
            expr.args,
        )
        if registered_conversion is not None:
            return registered_conversion
        aggregate_constructor = self.generate_metal_aggregate_constructor_call(
            expr,
            func_name,
        )
        if aggregate_constructor is not None:
            return aggregate_constructor
        if (
            func_name in self.metal_type_aliases
            or (
                func_name not in self.user_function_names
                and self.is_vector_value_type(func_name)
            )
            or re.fullmatch(
                r"(?:u?int(?:8|16|32|64)(?:_t)?|"
                r"(?:float(?:16|32|64)|bfloat16)(?:_t)?)[234]",
                str(func_name),
            )
            is not None
            or func_name in METAL_BUILTIN_CONSTRUCTOR_NAMES
        ):
            metal_type = self.map_type(func_name)
            matrix_resize = self.generate_metal_matrix_resize_constructor(
                metal_type, expr.args
            ) or self.metal_matrix_builtins.component_constructor(metal_type, expr.args)
            if matrix_resize is not None:
                return matrix_resize
            args = ", ".join(
                self.generate_metal_value_constructor_arguments(metal_type, expr.args)
            )
            direct = self.metal_direct_narrow_float_conversion(
                func_name, expr.args, args
            )
            if direct is not None:
                return direct
            rendered = f"{metal_type}({args})"
            return (
                self.metal_narrow_integer_conversion_expression(rendered, func_name)
                or rendered
            )
        return UNHANDLED
