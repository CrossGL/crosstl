"""Call-expression result-type rules for ``HLSLResultTypeMixin.hlsl_call_expression_result_type``, tried in order."""

from ..ast import MemberAccessNode
from ..unhandled import UNHANDLED
from .directx_constructor_names import HLSL_CONSTRUCTOR_RESULT_TYPE_NAMES
from .directx_values import HLSL_RAY_SYSTEM_VALUE_TYPES
from .generic_concrete_call_results import concrete_overload_results_differ
from .generic_function_utils import (
    generic_function_call_name,
    numeric_trait_method_result_type,
)
from .image_access_contracts import (
    is_image_atomic_operation,
    is_texel_fetch_basic_operation,
    is_texel_fetch_offset_operation,
    is_texture_compare_operation,
    is_texture_gather_compare_operation,
    is_texture_gather_operation,
    is_texture_query_lod_operation,
    is_texture_sample_operation,
)


class HLSLCallResultTypeMixin:
    """Families of call-expression result-type rules, each returning ``UNHANDLED`` when none of its rules applies."""

    def hlsl_method_call_result_type(
        self,
        expr,
        args,
        func_expr,
        func_name,
    ):
        """Result types of method calls on resources and objects, built-in option calls and bfloat16 vector built-ins."""
        if (
            not args
            and isinstance(func_name, str)
            and func_name in HLSL_RAY_SYSTEM_VALUE_TYPES
            and func_name not in getattr(self, "function_return_types", {})
        ):
            return HLSL_RAY_SYSTEM_VALUE_TYPES[func_name]
        if (
            isinstance(func_name, str)
            and func_name not in getattr(self, "function_return_types", {})
            and self.hlsl_bfloat16_vector_lanes(func_name) is not None
        ):
            return func_name
        if isinstance(func_expr, MemberAccessNode) or (
            hasattr(func_expr, "__class__")
            and "MemberAccess" in str(func_expr.__class__)
        ):
            member = str(getattr(func_expr, "member", ""))
            object_expr = getattr(
                func_expr, "object", getattr(func_expr, "object_expr", None)
            )
            object_type = self.hlsl_buffer_helper_resource_type(
                object_expr
            ) or self.expression_result_type(object_expr)
            if member == "Load":
                element_type = self.hlsl_typed_buffer_element_type(
                    object_type,
                    {
                        "Buffer",
                        "StructuredBuffer",
                        "RWBuffer",
                        "RWStructuredBuffer",
                        "RasterizerOrderedBuffer",
                        "RasterizerOrderedStructuredBuffer",
                    },
                )
                if element_type is not None:
                    return element_type
            if member in {"Load", "Load2", "Load3", "Load4"}:
                resource_name = self.hlsl_resource_type_name(object_type)
                if resource_name == "ByteAddressBuffer":
                    width = {"Load": "", "Load2": "2", "Load3": "3", "Load4": "4"}[
                        member
                    ]
                    return f"uint{width}" if width else "uint"
        option_call_type = self.hlsl_builtin_option_call_type(expr, func_name)
        if option_call_type is not None:
            return option_call_type
        bfloat_argument_types = [
            self.hlsl_source_expression_type(argument) for argument in args
        ]
        vector_builtin_type = self.hlsl_bfloat16_vector_builtin_result_type(
            func_name, bfloat_argument_types
        )
        if vector_builtin_type is not None:
            return vector_builtin_type
        if any(
            self.is_hlsl_bfloat16_type(argument_type)
            for argument_type in bfloat_argument_types
        ):
            bfloat_builtin_result_type = self.hlsl_bfloat16_builtin_result_type(
                func_name,
                bfloat_argument_types,
                len(args),
            )
            if bfloat_builtin_result_type is not None:
                return bfloat_builtin_result_type
        return UNHANDLED

    def hlsl_math_call_result_type(
        self,
        expr,
        args,
        func_name,
    ):
        """Result types of wave, SIMD-shuffle, buffer-load, non-uniform-index, numeric-trait, hyperbolic, copysign, signbit and matrix built-ins."""
        if func_name in self.HLSL_WAVE_INTRINSIC_ARITIES:
            return self.hlsl_wave_intrinsic_return_type(func_name, args)
        metal_simd_result_type = self.hlsl_metal_simd_shuffle_result_type(
            func_name, args
        )
        if metal_simd_result_type is not None:
            return metal_simd_result_type
        if func_name == "buffer_load" and args:
            resource_type = self.hlsl_buffer_helper_resource_type(args[0])
            element_type = self.hlsl_typed_buffer_element_type(
                resource_type,
                {
                    "Buffer",
                    "StructuredBuffer",
                    "RWBuffer",
                    "RWStructuredBuffer",
                    "RasterizerOrderedBuffer",
                    "RasterizerOrderedStructuredBuffer",
                },
            )
            if element_type is not None:
                return element_type
        if func_name in self.HLSL_NONUNIFORM_RESOURCE_INDEX_NAMES:
            return self.hlsl_nonuniform_resource_index_return_type(args)
        numeric_result_type = numeric_trait_method_result_type(self, expr)
        if numeric_result_type:
            return numeric_result_type
        if func_name in self.HLSL_INVERSE_HYPERBOLIC_ALIASES:
            resolved_inverse_hyperbolic = self.resolve_hlsl_function_overload(
                func_name,
                args,
                call_node=expr,
            )
            if resolved_inverse_hyperbolic is not None:
                return self.type_name_string(
                    getattr(resolved_inverse_hyperbolic, "return_type", None)
                )
            if args:
                return self.expression_result_type(args[0])
        if func_name in self.HLSL_COPYSIGN_NAMES:
            resolved_copysign = self.resolve_hlsl_function_overload(
                func_name,
                args,
                call_node=expr,
            )
            if resolved_copysign is not None:
                return self.type_name_string(
                    getattr(resolved_copysign, "return_type", None)
                )
            if args:
                return self.expression_result_type(args[0])
        if func_name in self.HLSL_SIGNBIT_NAMES:
            resolved_signbit = self.resolve_hlsl_function_overload(
                func_name,
                args,
                call_node=expr,
            )
            if resolved_signbit is not None:
                return self.type_name_string(
                    getattr(resolved_signbit, "return_type", None)
                )
            signbit_result_type = self.hlsl_signbit_result_type(args)
            if signbit_result_type is not None:
                return signbit_result_type
        if func_name == "inverse" and args:
            return self.expression_result_type(args[0])
        if func_name == "transpose" and args:
            return self.hlsl_transpose_result_type(self.expression_result_type(args[0]))
        if func_name in {"matrixCompMult", "outerProduct"} and func_name not in (
            getattr(self, "function_return_types", {})
        ):
            matrix_call_type = self.hlsl_matrix_orientation.call_type(
                func_name, [self.expression_result_type(arg) for arg in args]
            )
            if matrix_call_type is not None:
                return matrix_call_type
        return UNHANDLED

    def hlsl_bit_and_constructor_call_result_type(
        self,
        args,
        func_name,
    ):
        """Result types of bit-manipulation and bitcast built-ins, type constructors and the geometric built-ins."""
        if (
            func_name in {
                "frac",
                "fract",
                "inverseSqrt",
                "inversesqrt",
                "rsqrt",
                "mod",
                # Derivatives return their operand's type (GLSL 4.60 section 8.14).
                "dFdx",
                "dFdy",
                "dFdxFine",
                "dFdyFine",
                "dFdxCoarse",
                "dFdyCoarse",
                "fwidth",
                "fwidthFine",
                "fwidthCoarse",
                "fwidth_fine",
                "fwidth_coarse",
                "ddx",
                "ddy",
                "ddx_fine",
                "ddy_fine",
                "ddx_coarse",
                "ddy_coarse",
            } and args and func_name not in getattr(self, "function_return_types", {})
        ):
            return self.expression_result_type(args[0])
        if func_name in {
            "quiet_NaN",
            "signaling_NaN",
            "infinity",
        } and func_name not in getattr(self, "function_return_types", {}):
            return "float"
        if (
            func_name in {
                "clamp",
                "lerp",
                "max",
                "min",
                "mix",
                "fabs",
                "fmax",
                "fmin",
                "select",
                "atan2",
            }
            and args
            and func_name not in getattr(self, "function_return_types", {})
        ):
            return self.expression_result_type(args[0])
        trailing_zero_result_type = self.hlsl_trailing_zero_result_type(func_name, args)
        if trailing_zero_result_type is not None:
            return trailing_zero_result_type
        explicit_bitcast_type = self.hlsl_metal_as_type_target_type(func_name)
        if explicit_bitcast_type is not None:
            source_target_type = self.hlsl_metal_as_type_source_target_type(func_name)
            if self.hlsl_bfloat16_lanes(source_target_type) is not None:
                return source_target_type
            return explicit_bitcast_type
        bitcast_result_type = self.hlsl_bitcast_result_type(func_name, args)
        if bitcast_result_type is not None:
            return bitcast_result_type
        countbits_result_type = self.hlsl_countbits_result_type(func_name, args)
        if countbits_result_type is not None:
            return countbits_result_type
        reversebits_result_type = self.hlsl_reversebits_result_type(func_name, args)
        if reversebits_result_type is not None:
            return reversebits_result_type
        firstbit_result_type = self.hlsl_firstbit_result_type(func_name, args)
        if firstbit_result_type is not None:
            return firstbit_result_type
        if func_name in {"normalize", "reflect"} and args:
            return self.expression_result_type(args[0])
        if self.is_hlsl_bfloat16_type(func_name):
            return self.hlsl_bfloat16_scalar_type_name(func_name)
        constructor_type = self.hlsl_type_constructor_name(func_name)
        if constructor_type is not None:
            return constructor_type
        if (
            func_name not in getattr(self, "function_return_types", {})
            and self.hlsl_struct_constructor_fields(func_name) is not None
        ):
            return self.map_type(func_name)
        if func_name == "dot" and args:
            return (
                self.vector_component_type(self.expression_result_type(args[0]))
                or "float"
            )
        if func_name == "cross" and args:
            return self.expression_result_type(args[0])
        return UNHANDLED

    def hlsl_function_and_resource_call_result_type(
        self,
        args,
        func_name,
    ):
        """Result types of user and generic functions, overloads, predicates, and image, texture and buffer operations, and of calls of type names."""
        specialized_func_name = generic_function_call_name(self, func_name, args)
        if specialized_func_name in getattr(self, "function_return_types", {}):
            return self.function_return_types[specialized_func_name]
        overloaded_return_type = self.hlsl_resolve_overloaded_return_type(
            func_name, args
        )
        if overloaded_return_type is not None:
            return overloaded_return_type
        if getattr(
            self, "collecting_generic_function_specializations", False
        ) and concrete_overload_results_differ(
            self,
            func_name,
            lambda generator, type_name: generator.map_type(type_name),
        ):
            # Generic deduction must not bind a type parameter to a guessed overload;
            # it resolves the overload from the argument types it tracks.
            return None
        if func_name in getattr(self, "function_return_types", {}):
            return self.function_return_types[func_name]
        if func_name in {"isnan", "isinf", "isfinite"} and len(args) == 1:
            argument_type = self.expression_result_type(args[0])
            if self.hlsl_floating_arithmetic_type_info(argument_type) is not None:
                return self.hlsl_boolean_expression_result_type(argument_type)
        if func_name in {"any", "all"} and len(args) == 1:
            return "bool"
        if len(args) == 1 and func_name in self.hlsl_componentwise_float_unary_calls:
            argument_type = self.expression_result_type(args[0])
            if argument_type is not None:
                mapped_argument = self.map_type(argument_type)
                if mapped_argument in {"float", "float2", "float3", "float4"}:
                    return mapped_argument
        unsupported_functions = getattr(
            self, "unsupported_glsl_buffer_block_functions", {}
        )
        if func_name in unsupported_functions:
            return unsupported_functions[func_name].get("return_type")
        if func_name == "buffer_consume" and args:
            resource_type = self.hlsl_buffer_helper_resource_type(args[0])
            return self.hlsl_typed_buffer_element_type(
                resource_type, {"ConsumeStructuredBuffer"}
            )
        if func_name in {
            "buffer_increment_counter",
            "buffer_decrement_counter",
        }:
            return "uint"
        if is_image_atomic_operation(func_name) and args:
            return self.image_atomic_result_type(func_name, args[0])
        if func_name == "imageLoad" and args:
            return self.image_load_result_type(args[0])
        if func_name == "subpassLoad":
            return "vec4"
        if func_name == "textureSamplePosition":
            return "vec2"
        if is_texture_query_lod_operation(func_name):
            return "vec2"
        if is_texture_compare_operation(func_name):
            return "float"
        if (
            is_texture_sample_operation(func_name)
            or is_texture_gather_operation(func_name)
            or is_texture_gather_compare_operation(func_name)
            or is_texel_fetch_basic_operation(func_name)
            or is_texel_fetch_offset_operation(func_name)
        ):
            return "vec4"
        if func_name in HLSL_CONSTRUCTOR_RESULT_TYPE_NAMES:
            return str(func_name)
        return UNHANDLED
