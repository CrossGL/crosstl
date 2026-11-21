"""Call-expression result-type rules for ``MetalResultTypeMixin.metal_call_expression_result_type``, tried in order."""

from ..unhandled import (
    UNHANDLED,
)
from .builtin_result_types import (
    C_FAMILY,
    boolean_builtin_result_type,
)
from .generic_function_utils import (
    generic_function_call_name,
    numeric_trait_method_result_type,
)
from .image_access_contracts import (
    is_resource_size_query_operation,
    is_texel_fetch_operation,
    is_texture_sampling_operation,
)
from .metal_constructor_names import (
    METAL_CONSTRUCTOR_RESULT_TYPE_NAMES,
)
from .metal_hlsl_matrix import (
    hlsl_emulated_matrix_shape,
)


class MetalCallResultTypeMixin:
    """Families of call-expression result-type rules, each returning ``UNHANDLED`` when none of its rules applies."""

    def metal_math_call_result_type(
        self,
        expr,
        args,
        func_name,
    ):
        """Result types of option and ray-description constructors, emulated matrices, logical and select built-ins, wave intrinsics, numeric traits, and the geometric, matrix and component-wise math built-ins."""
        if isinstance(func_name, str) and func_name.rsplit("::", 1)[-1] == "Some":
            payload_type = self.expected_option_payload_type_name()
            if payload_type is not None:
                return payload_type
            if self.option_payload_type_name(self.current_expression_expected_type):
                return None
            return self.expression_result_type(args[0]) if args else None
        if func_name == "RayDesc" and self.requires_metal_builtin_ray_desc(func_name):
            return "RayDesc"
        if isinstance(func_name, str) and hlsl_emulated_matrix_shape(func_name):
            return func_name
        if func_name in {"and", "or"} and len(args) == 2:
            for arg in args:
                shape = self.metal_expression_emulated_matrix_shape(arg)
                if shape is not None:
                    return f"bool{shape[1]}x{shape[2]}"
            logical_type = self.metal_hlsl_logical_result_type(func_name, args)
            if logical_type is not None:
                return logical_type
        if func_name == "select" and len(args) == 3:
            for arg in args[:2]:
                if self.metal_expression_emulated_matrix_shape(arg) is not None:
                    return self.expression_result_type(arg)
        if func_name in self.METAL_WAVE_INTRINSIC_ARITIES:
            return self.metal_wave_result_type(func_name, args)
        numeric_result_type = numeric_trait_method_result_type(self, expr)
        if numeric_result_type:
            return numeric_result_type
        if func_name in {"normalize", "reflect"} and args:
            return self.expression_result_type(args[0])
        if func_name == "inverse" and args:
            return self.expression_result_type(args[0])
        if func_name == "transpose" and args:
            arg_type = self.expression_result_type(args[0])
            dimensions = self.metal_matrix_dimensions(arg_type)
            if dimensions is not None:
                component_prefix, columns, rows = dimensions
                return f"{component_prefix}{rows}x{columns}"
            return arg_type
        if func_name == "dot" and args:
            unified_type = self.metal_unified_builtin_result_type(func_name, args)
            if unified_type is not None:
                return unified_type
            return (
                self.vector_component_type(self.expression_result_type(args[0]))
                or "float"
            )
        if func_name == "cross" and args:
            return self.expression_result_type(args[0])
        if func_name == "mod" and args and func_name not in self.user_function_names:
            return self.expression_result_type(args[0])
        if func_name == "frac" and args and func_name not in self.user_function_names:
            return self.expression_result_type(args[0])
        if func_name == "lerp" and args and func_name not in self.user_function_names:
            return self.expression_result_type(args[0])
        componentwise_type = self.metal_componentwise_builtin_result_type(
            func_name, args
        )
        if componentwise_type is not None:
            return componentwise_type
        if (
            func_name in {"inverseSqrt", "inversesqrt", "roundEven"}
            and args
            and func_name not in self.user_function_names
        ):
            unified_type = self.metal_unified_builtin_result_type(func_name, args)
            if unified_type is not None:
                return unified_type
            return self.expression_result_type(args[0])
        if (
            func_name in self.METAL_DERIVATIVE_FUNCTION_ALIASES
            and args
            and func_name not in self.user_function_names
        ):
            return self.expression_result_type(args[0])
        return UNHANDLED

    def metal_conversion_and_texture_call_result_type(
        self,
        args,
        func_name,
    ):
        """Result types of reinterpretation, bitcast, integer-bit, packed-byte and angle-conversion built-ins, the unified min/max/mix family, and texture size queries, samples and fetches."""
        explicit_as_type = self.metal_explicit_as_type_target(func_name, args)
        if explicit_as_type is not None:
            return explicit_as_type
        bitcast_result_type = self.metal_bitcast_result_type(func_name, args)
        if bitcast_result_type is not None:
            return bitcast_result_type
        integer_bit_result_type = self.metal_integer_bit_result_type(func_name, args)
        if integer_bit_result_type is not None:
            return integer_bit_result_type
        if (
            func_name in self.PACKED_BYTE_INTRINSICS
            and func_name not in self.user_function_names
        ):
            return self.PACKED_BYTE_INTRINSICS[func_name][0]
        if (
            func_name in self.METAL_ANGLE_CONVERSION_FACTORS
            and len(args or []) == 1
            and func_name not in self.user_function_names
        ):
            argument_type = self.map_type(self.expression_result_type(args[0]))
            if self.vector_component_type(argument_type) in {"int", "uint"}:
                width = self.vector_value_width(argument_type)
                return f"float{width or ''}"
            return self.expression_result_type(args[0])
        if func_name in {"mix", "clamp", "min", "max", "saturate"} and args:
            unified_type = self.metal_unified_builtin_result_type(func_name, args)
            if unified_type is not None:
                return unified_type
            return self.expression_result_type(args[0])
        if is_resource_size_query_operation(func_name) and args:
            texture_type = self.texture_argument_resource_type(args[0])
            raw_type = self.expression_result_type(args[0])
            descriptor = self.texture_query_size_descriptor_for_argument(
                texture_type, raw_type
            )
            if descriptor is not None:
                return descriptor["return_type"]
        if (
            is_texture_sampling_operation(func_name)
            or is_texel_fetch_operation(func_name)
        ) and args:
            if self.is_plain_depth_read(func_name, args):
                # generate_texture_call widens the scalar MSL depth read to (d, 0, 0, 1).
                return "float4"
            return self.texture_sample_result_type(args[0])
        return UNHANDLED

    def metal_function_and_constructor_call_result_type(
        self,
        args,
        func_name,
    ):
        """Result types of user and generic functions, overloads, boolean built-ins, aggregate constructors, image and subpass loads, and calls of type names."""
        specialized_func_name = generic_function_call_name(self, func_name, args)
        if specialized_func_name in getattr(self, "function_return_types", {}):
            return self.function_return_types[specialized_func_name]
        overload = self.metal_user_function_overload_for_call(func_name, args)
        if overload is not None:
            return self.metal_effective_function_return_type(overload)
        overloads = self.function_overloads_by_name.get(func_name, [])
        if overloads:
            argument_count = len(args or [])
            viable_return_types = {
                self.metal_effective_function_return_type(candidate)
                for candidate in overloads
                if self.metal_user_function_accepts_argument_count(
                    candidate,
                    argument_count,
                )
            }
            if len(viable_return_types) == 1:
                return next(iter(viable_return_types))
            # Type-aware source overload resolution is unavailable here.
            # Differing same-arity results must remain unknown rather than
            # consuming the last declaration in function_return_types.
            return None
        if func_name in getattr(self, "function_return_types", {}):
            return self.function_return_types[func_name]
        boolean_type = boolean_builtin_result_type(
            func_name,
            [self.expression_result_type(arg) for arg in args or ()],
            C_FAMILY,
        )
        if boolean_type is not None:
            return boolean_type
        aggregate_constructor = self.metal_aggregate_constructor_name(func_name)
        if aggregate_constructor is not None:
            return aggregate_constructor
        unsupported_functions = getattr(
            self, "unsupported_glsl_buffer_block_functions", {}
        )
        if func_name in unsupported_functions:
            return unsupported_functions[func_name].get("return_type")
        if func_name == "imageLoad" and args:
            return self.image_load_result_type(args[0])
        if func_name == "subpassLoad":
            return "vec4"
        if self.is_builtin_value_constructor_type(func_name):
            return str(func_name)
        if func_name in METAL_CONSTRUCTOR_RESULT_TYPE_NAMES:
            return str(func_name)
        return UNHANDLED
