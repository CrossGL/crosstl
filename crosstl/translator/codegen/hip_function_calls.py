"""Function-call lowering of the HIP generator: the call visitor, math names, bit casts, integer bit functions, shader clocks and warp-sync built-ins."""

from ..ast import (
    FunctionCallNode,
    MemberAccessNode,
)
from ..refusals import HipRefusal
from .c_narrow_integer_vectors import generic_narrow_vector_spelling
from .c_pointer_math import generate_pointer_math_call
from .c_vector_reductions import generate_vector_reduction_call
from .c_wide_integer_vectors import wide_integer_vector_constructor
from .cuda_componentwise_math import generate_componentwise_math_call
from .cuda_pointer_operands import hip_pointer_operand_call
from .cuda_type_reinterpretation import (
    as_type_result_type,
    is_as_type_call,
    lower_as_type,
)
from .enum_utils import generate_enum_constructor_call
from .generic_function_utils import (
    generate_numeric_trait_method_call,
    generate_static_generic_numeric_call,
    generic_function_call_name,
    reject_unresolved_generic_function_call,
)
from .hip_cuda_warps import hip_cuda_warp_call
from .hip_library_calls import (
    ffs64_argument,
    generate_hip_library_call,
)
from .hip_refusals import HipLoweringRefused
from .hip_tables import (
    HIP_64BIT_BITCAST_FUNCTIONS,
    HIP_BITCAST_FUNCTION_TARGETS,
    HIP_INTEGER_BIT_FUNCTION_ALIASES,
    HIP_WAVE_OP_ARITIES,
)


class HipFunctionCallMixin:
    """Function call members of ``HipCodeGen``."""

    def visit_FunctionCallNode(self, node) -> str:
        func_expr = (
            node.function if hasattr(node, "function") else getattr(node, "name", None)
        )
        func_name = None
        if isinstance(func_expr, MemberAccessNode):
            callee = self.visit(func_expr)
            func_name = callee
        elif hasattr(func_expr, "name") and getattr(func_expr, "name", None):
            func_name = func_expr.name
            callee = func_name
        elif isinstance(func_expr, str):
            func_name = func_expr
            callee = func_expr
        else:
            callee = self.visit(func_expr)
        raw_args = getattr(node, "args", getattr(node, "arguments", []))

        numeric_trait_call = generate_numeric_trait_method_call(
            self,
            func_expr,
            raw_args,
        )
        if numeric_trait_call is not None:
            return numeric_trait_call

        static_generic_call = generate_static_generic_numeric_call(self, func_name)
        if static_generic_call is not None:
            return static_generic_call

        specialized_func_name = generic_function_call_name(self, func_name, raw_args)
        if specialized_func_name is not None:
            func_name = specialized_func_name
            callee = specialized_func_name
        else:
            reject_unresolved_generic_function_call(self, func_name, "HIP")

        enum_constructor = generate_enum_constructor_call(self, func_name, raw_args)
        if enum_constructor is not None:
            return enum_constructor

        args = [self.visit(arg) for arg in raw_args]

        half_constructor = self.hip_half_constructor_expression(
            func_name, raw_args, args
        )
        if half_constructor is not None:
            return half_constructor

        if func_name == "lambda":
            return self.generate_lambda_expression(raw_args)
        if func_name in HIP_WAVE_OP_ARITIES:
            return self.generate_wave_operation(func_name, raw_args, args)
        if func_name in {"SetMeshOutputCounts", "DispatchMesh"}:
            return self.generate_mesh_task_call_expression(func_name, raw_args, args)

        is_user_function = self.is_user_defined_function(func_name)
        if not is_user_function:
            scalar_alias_constructor = self.hip_scalar_alias_constructor_expression(
                func_name, args
            )
            if scalar_alias_constructor is not None:
                return scalar_alias_constructor
        if not is_user_function:
            warp_sync_builtin_call = self.generate_warp_sync_builtin_call(
                func_name, raw_args, args
            )
            if warp_sync_builtin_call is not None:
                return warp_sync_builtin_call
            pointer_operand_call = hip_pointer_operand_call(
                self, func_name, raw_args, args, HipRefusal
            )
            if pointer_operand_call is not None:
                return pointer_operand_call
        if not is_user_function:
            ray_tracing_call = self.generate_ray_tracing_call_expression(
                func_name, raw_args
            )
            if ray_tracing_call is not None:
                return ray_tracing_call
        if not is_user_function:
            buffer_call = self.generate_buffer_call(
                func_expr, func_name, raw_args, args
            )
            if buffer_call is not None:
                return buffer_call

            structured_atomic_call = self.generate_structured_buffer_atomic_call(
                func_name, raw_args, args
            )
            if structured_atomic_call is not None:
                return structured_atomic_call

            plain_atomic_call = self.generate_plain_atomic_call(
                func_name, raw_args, args
            )
            if plain_atomic_call is not None:
                return plain_atomic_call

            resource_call = self.generate_resource_call(func_name, raw_args, args)
            if resource_call is not None:
                return resource_call

            bitcast_call = self.generate_hip_bitcast_call(func_name, raw_args, args)
            if bitcast_call is not None:
                return bitcast_call

            integer_bit_call = self.generate_hip_integer_bit_call(
                func_name, raw_args, args
            )
            if integer_bit_call is not None:
                return integer_bit_call

            shader_clock_call = self.generate_hip_shader_clock_call(func_name, raw_args)
            if shader_clock_call is not None:
                return shader_clock_call

        if is_user_function:
            args = self.hip_user_function_call_arguments(func_name, raw_args, args)
            return f"{callee}({', '.join(args)})"

        pointer_math_call = generate_pointer_math_call(
            self, HipLoweringRefused, func_name, raw_args, args
        )
        if pointer_math_call is not None:
            return pointer_math_call

        library_call = generate_hip_library_call(self, func_name, raw_args, args)
        if library_call is not None:
            return library_call

        if func_name in self.synchronization_builtins and raw_args:
            raise HipRefusal(
                f"HIP synchronization builtin '{func_name}' requires 0 "
                f"arguments; got {len(raw_args)}"
            )

        args = self.query_metadata_call_arguments(func_name, raw_args, args)

        union_constructor = self.native_unions.constructor(func_name, raw_args, None)
        if union_constructor is not None:
            return union_constructor
        if func_name in self.structs_by_name:
            # The structure has no constructor; aggregate initialization sets its members in order.
            return f"{self.map_type(func_name)}{{{', '.join(args)}}}"

        func_name = generic_narrow_vector_spelling(func_name) or func_name
        mapped_name = wide_integer_vector_constructor(func_name)
        if mapped_name is None:
            # A name function_map does not rename is a program function or a HIP library function, spelled as
            # it is; a built-in function of another language left in the output is refused by
            # render_checked_program.
            mapped_name = (
                self.function_map[func_name]
                if func_name in self.function_map
                else func_name
            )

        reduction_call = generate_vector_reduction_call(
            self, "HIP", func_name, raw_args, args
        )
        if reduction_call is not None:
            return reduction_call

        if func_name == "abs":
            if len(args) == 1:
                abs_call = self.generate_abs_call(raw_args, args)
                if abs_call is not None:
                    return abs_call
        elif func_name == "sign":
            if len(args) == 1:
                sign_call = self.generate_sign_call(raw_args, args)
                if sign_call is not None:
                    return sign_call
        elif func_name == "mod":
            if len(args) == 2:
                mod_call = self.generate_mod_call(raw_args, args)
                if mod_call is not None:
                    return mod_call
        elif func_name in {"fract", "frac"}:
            if len(args) == 1:
                return self.generate_fract_call(raw_args, args)
        elif func_name in {"min", "max"}:
            if len(args) == 2:
                min_max_call = self.generate_min_max_call(func_name, raw_args, args)
                if min_max_call is not None:
                    return min_max_call
        elif func_name in {"atan", "atan2"}:
            if len(args) == 2:
                atan2_call = self.generate_atan2_call(raw_args, args)
                if atan2_call is not None:
                    return atan2_call
                if func_name == "atan":
                    scalar_atan2_call = self.generate_scalar_atan2_call(raw_args, args)
                    if scalar_atan2_call is not None:
                        return scalar_atan2_call
        elif func_name in {"fma", "mad"}:
            if len(args) == 3:
                result_type = self.fused_multiply_add_result_type(raw_args)
                fma_call = self.generate_fused_multiply_add_call(raw_args, args)
                if fma_call is not None:
                    if result_type is not None:
                        node.expression_type = result_type
                    return fma_call
        elif func_name in {
            "dot",
            "cross",
            "distance",
            "length",
            "normalize",
            "reflect",
            "refract",
            "faceforward",
        }:
            geometric_call = self.generate_vector_geometric_call(
                func_name,
                raw_args,
                args,
            )
            if geometric_call is not None:
                return geometric_call
        elif func_name == "clamp":
            if len(args) == 3:
                return self.generate_clamp_call(raw_args, args)
        elif func_name in {"mix", "lerp"}:
            if len(args) == 3:
                mix_call = self.generate_mix_call(raw_args, args)
                if mix_call is not None:
                    return mix_call
        elif func_name == "saturate":
            if len(args) == 1:
                saturate_call = self.generate_saturate_call(raw_args, args)
                if saturate_call is not None:
                    return saturate_call
        elif func_name == "step":
            if len(args) == 2:
                step_call = self.generate_step_call(raw_args, args)
                if step_call is not None:
                    return step_call
        elif func_name == "smoothstep":
            if len(args) == 3:
                smoothstep_call = self.generate_smoothstep_call(raw_args, args)
                if smoothstep_call is not None:
                    return smoothstep_call
        elif func_name in ["texture", "tex2D"]:
            if len(args) >= 2:
                return (
                    f"tex2D<float4>({args[0]}, "
                    f"{self.coord_component(args[1], 'x')}, "
                    f"{self.coord_component(args[1], 'y')})"
                )

        componentwise_call = generate_componentwise_math_call(
            self, func_name, raw_args, args
        )
        if componentwise_call is not None:
            return componentwise_call

        math_func_name = self.hip_math_function_name(func_name)
        vector_math_call = self.generate_vector_scalar_math_call(
            math_func_name,
            raw_args,
            args,
        )
        if vector_math_call is not None:
            return vector_math_call

        vector_info = self.vector_type_info(func_name)
        if vector_info:
            splat_call = self.generate_vector_scalar_splat_call(
                vector_info, raw_args, args
            )
            if splat_call is not None:
                return splat_call
            constructor_call = self.generate_vector_constructor_single_eval_call(
                vector_info, raw_args, args
            )
            if constructor_call is not None:
                return constructor_call
            args = self.generate_vector_constructor_args(vector_info, raw_args, args)

        scalar_math_call = self.generate_scalar_math_call(
            math_func_name, raw_args, args
        )
        if scalar_math_call is not None:
            return scalar_math_call

        args_str = ", ".join(args)
        target = mapped_name if mapped_name is not None else callee
        return f"{target}({args_str})"

    def hip_math_function_name(self, func_name):
        """The math lowering's name for ``func_name``: ``inverseSqrt`` and ``rsqrt`` are spellings of ``inversesqrt``.

        Any other name is its own math lowering name; the math lowerings return ``None`` for a name they do not
        handle, and the call is then refused if it names another language's built-in function.
        """
        if func_name in {"inverseSqrt", "rsqrt"}:
            return "inversesqrt"
        return func_name

    def normalize_hip_integer_bit_function(self, func_name):
        if func_name in HIP_INTEGER_BIT_FUNCTION_ALIASES:
            return HIP_INTEGER_BIT_FUNCTION_ALIASES[func_name]
        # Already the CrossGL spelling, or not an integer bit function; callers check the result.
        return func_name

    def hip_bitcast_result_type(self, func_name, raw_args):
        if is_as_type_call(func_name):
            return as_type_result_type(self, func_name)
        wide_bitcast = HIP_64BIT_BITCAST_FUNCTIONS.get(func_name)
        if (
            wide_bitcast is not None
            and not self.is_user_defined_function(func_name)
            and len(raw_args or []) == 1
        ):
            return wide_bitcast[1]

        target_component = HIP_BITCAST_FUNCTION_TARGETS.get(func_name)
        if (
            target_component is None
            or self.is_user_defined_function(func_name)
            or len(raw_args or []) != 1
        ):
            return None

        source_type = self.expression_result_type(raw_args[0])
        source_info = self.vector_type_info(source_type)
        if source_info is not None:
            if source_info["component_type"] not in {"float", "int", "uint"}:
                return None
            return self.vector_type_for_components(
                target_component,
                len(source_info["components"]),
            )

        source_component = self.scalar_component_type(source_type)
        if source_component in {"float", "int", "uint"}:
            return target_component
        return None

    def generate_hip_bitcast_call(self, func_name, raw_args, args):
        if is_as_type_call(func_name) and not self.is_user_defined_function(func_name):
            return lower_as_type(
                self,
                func_name,
                raw_args,
                args,
                HipLoweringRefused,
                self.require_hip_vector_bitcast_helper,
            )
        result_type = self.hip_bitcast_result_type(func_name, raw_args)
        if result_type is None:
            return None

        wide_bitcast = HIP_64BIT_BITCAST_FUNCTIONS.get(func_name)
        if wide_bitcast is not None:
            intrinsic = wide_bitcast[0]
            # The intrinsic takes one scalar, and HIP converts any arithmetic scalar to
            # its parameter type. It has no vector or matrix form.
            source_type = self.expression_result_type(raw_args[0])
            if (
                source_type is not None
                and self.scalar_component_type(source_type) is None
            ):
                raise HipLoweringRefused(
                    [
                        f"bit reinterpretation: {func_name} of "
                        f"{self.type_name_string(source_type)}: "
                        f"{intrinsic} takes one scalar"
                    ]
                )
            return f"{intrinsic}({args[0]})"

        source_type = self.expression_result_type(raw_args[0])
        source_info = self.vector_type_info(source_type)
        result_info = self.vector_type_info(result_type)
        if source_info is not None and result_info is not None:
            helper_name = self.require_hip_vector_bitcast_helper(
                source_info,
                result_info,
            )
            return f"{helper_name}({args[0]})"

        target_component = HIP_BITCAST_FUNCTION_TARGETS[func_name]
        source_component = self.scalar_component_type(source_type)
        return self.format_hip_scalar_bitcast(
            source_component,
            target_component,
            args[0],
        )

    def hip_integer_bit_result_type(self, func_name, raw_args):
        original_func_name = func_name
        func_name = self.normalize_hip_integer_bit_function(func_name)
        if func_name not in {
            "bitCount",
            "bitfieldReverse",
            "findLSB",
            "findMSB",
            "countLeadingZeros",
        } or self.is_user_defined_function(original_func_name):
            return None

        if len(raw_args or []) != 1:
            return "uint"

        source_type = self.expression_result_type(raw_args[0])
        source_info = self.vector_type_info(source_type)
        if func_name == "countLeadingZeros":
            # __clz and __clzll both return int.
            if source_info is not None:
                return self.vector_type_for_components(
                    "int",
                    len(source_info["components"]),
                )
            return "int"
        if source_info is not None:
            if source_info["component_type"] not in {"int", "uint"}:
                return self.vector_type_for_components(
                    "uint",
                    len(source_info["components"]),
                )
            if func_name == "bitCount":
                # GLSL 4.60, 8.8: bitCount returns genIType, int components for int and uint operands alike.
                return self.vector_type_for_components(
                    "int",
                    len(source_info["components"]),
                )
            return source_type

        source_component = self.scalar_component_type(source_type)
        if source_component in {"int", "uint"}:
            if func_name == "bitCount":
                return "uint"
            if func_name == "bitfieldReverse":
                return source_type or source_component
            return source_component
        return "uint"

    def generate_hip_integer_bit_call(self, func_name, raw_args, args):
        result_type = self.hip_integer_bit_result_type(func_name, raw_args)
        if result_type is None:
            return None
        if len(raw_args or []) != 1 or len(args) != 1:
            return None

        operation = self.normalize_hip_integer_bit_function(func_name)
        source_type = self.expression_result_type(raw_args[0])
        if source_type is None and operation == "countLeadingZeros":
            # A sized cast such as uint64_t(x) has no inferred type; read its width from the cast
            # so that __clz does not truncate a 64-bit value.
            source_type = self.hip_integer_cast_type(raw_args[0])
        source_info = self.vector_type_info(source_type)
        result_info = self.vector_type_info(result_type)
        if source_info is not None:
            if result_info is None or source_info["component_type"] not in {
                "int",
                "uint",
            }:
                return None
            helper_name = self.require_hip_integer_bit_vector_helper(
                operation,
                source_info,
                result_info,
            )
            return f"{helper_name}({args[0]})"

        source_component = self.scalar_component_type(source_type)
        if source_component not in {"int", "uint", None}:
            return None
        return self.format_hip_integer_bit_component(
            operation,
            args[0],
            source_component,
            self.scalar_component_type(result_type),
            source_type,
            result_type,
        )

    def hip_integer_cast_type(self, node):
        """The target type of a scalar integer cast such as ``uint64_t(x)``, or None."""
        if not isinstance(node, FunctionCallNode):
            return None
        function_expr = getattr(node, "function", getattr(node, "name", None))
        type_name = getattr(function_expr, "name", function_expr)
        if not isinstance(type_name, str) or self.is_user_defined_function(type_name):
            return None
        if self.scalar_component_type(type_name) in {"int", "uint"}:
            return type_name
        return None

    HIP_SHADER_CLOCK_SOURCES = {
        # GL_ARB_shader_clock reads a per-multiprocessor counter that does not run at a constant rate.
        "clockARB": "clock64()",
        "clock2x32ARB": "clock64()",
        # GL_EXT_shader_realtime_clock reads the device-wide, constant-rate counter.
        "clockRealtimeEXT": "wall_clock64()",
        "clockRealtime2x32EXT": "wall_clock64()",
    }

    # The CrossGL result types, so arithmetic on a 2x32 pair goes through the
    # uint2 helpers as it does for any other uvec2 value.
    HIP_SHADER_CLOCK_RESULT_TYPES = {
        "clockARB": "uint64_t",
        "clock2x32ARB": "uvec2",
        "clockRealtimeEXT": "uint64_t",
        "clockRealtime2x32EXT": "uvec2",
    }

    def generate_hip_shader_clock_call(self, func_name, raw_args):
        """The shader clock built-ins; the 2x32 forms return the 64-bit counter as a uint2 (low, high)."""
        source = self.HIP_SHADER_CLOCK_SOURCES.get(func_name)
        if source is None:
            return None
        if raw_args:
            raise HipRefusal(
                f"HIP shader clock builtin {func_name} requires 0 arguments; "
                f"got {len(raw_args)}"
            )
        if not func_name.endswith("2x32ARB") and not func_name.endswith("2x32EXT"):
            return source
        helper_name = (
            "cgl_hip_clock2x32"
            if func_name == "clock2x32ARB"
            else "cgl_hip_realtime_clock2x32"
        )
        if helper_name not in self.helper_functions:
            self.helper_functions[helper_name] = (
                f"__device__ inline uint2 {helper_name}()\n"
                "{\n"
                f"    unsigned long long ticks = {source};\n"
                "    return make_uint2((unsigned int)ticks, "
                "(unsigned int)(ticks >> 32));\n"
                "}"
            )
        return f"{helper_name}()"

    def require_hip_integer_bit_vector_helper(
        self,
        operation,
        source_info,
        result_info,
    ):
        helper_name = self.sanitize_helper_name(
            f"cgl_{source_info['type']}_{operation}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        components = [
            self.format_hip_integer_bit_component(
                operation,
                f"value.{component}",
                source_info["component_type"],
                result_info["component_type"],
            )
            for component in source_info["components"]
        ]
        helper = (
            f"__device__ inline {result_info['type']} {helper_name}"
            f"({source_info['type']} value)\n"
            "{\n"
            f"    return {result_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_hip_integer_bit_component(
        self,
        operation,
        value,
        source_component=None,
        result_component=None,
        source_type=None,
        result_type=None,
    ):
        bit_width = self.hip_integer_bit_width(source_type)
        unsigned_cast_type = "unsigned long long" if bit_width == 64 else "unsigned int"
        signed_cast_type = "long long" if bit_width == 64 else "int"
        popc_intrinsic = "__popcll" if bit_width == 64 else "__popc"
        ffs_intrinsic = "__ffsll" if bit_width == 64 else "__ffs"
        clz_intrinsic = "__clzll" if bit_width == 64 else "__clz"
        high_bit_index = 63 if bit_width == 64 else 31
        unsigned_value = (
            f"static_cast<{unsigned_cast_type}>({value})"
            if source_component == "int"
            else value
        )
        if operation == "bitCount":
            return f"{popc_intrinsic}({unsigned_value})"
        if operation == "bitfieldReverse":
            intrinsic_name = "__brevll" if bit_width == 64 else "__brev"
            reversed_value = f"{intrinsic_name}({unsigned_value})"
            if result_component == "int":
                return f"static_cast<{signed_cast_type}>({reversed_value})"
            return reversed_value
        if operation == "countLeadingZeros":
            return f"{clz_intrinsic}({value})"
        if operation == "findLSB":
            if bit_width == 64:
                scalar_type = self.map_type(source_type) if source_type else None
                return f"(__ffsll({ffs64_argument(value, scalar_type)}) - 1)"
            return f"({ffs_intrinsic}({value}) - 1)"
        if operation == "findMSB":
            clz_value = (
                f"static_cast<{unsigned_cast_type}>({value})"
                if bit_width == 64 and source_component == "int"
                else value
            )
            if source_component == "int":
                inverted_value = (
                    f"static_cast<{unsigned_cast_type}>(~({value}))"
                    if bit_width == 64
                    else f"~({value})"
                )
                return (
                    f"(({value}) < 0 ? ({high_bit_index} - "
                    f"{clz_intrinsic}({inverted_value})) : "
                    f"(({value}) == 0 ? -1 : ({high_bit_index} - "
                    f"{clz_intrinsic}({clz_value}))))"
                )
            return (
                f"(({value}) == 0 ? -1 : "
                f"({high_bit_index} - {clz_intrinsic}({value})))"
            )
        return f"{operation}({value})"

    def hip_integer_bit_width(self, type_name):
        if type_name is None:
            return 32
        type_text = self.type_name_string(type_name)
        mapped_type = self.map_type(type_text)
        if type_text in {"i64", "int64", "int64_t", "u64", "uint64", "uint64_t"}:
            return 64
        if mapped_type in {
            "long long",
            "long long int",
            "unsigned long long",
            "unsigned long long int",
        }:
            return 64
        return 32

    def require_hip_vector_bitcast_helper(self, source_info, result_info):
        helper_name = self.sanitize_helper_name(
            f"cgl_{source_info['type']}_to_{result_info['type']}_bitcast"
        )
        if helper_name in self.helper_functions:
            return helper_name

        components = [
            self.format_hip_scalar_bitcast(
                source_info["component_type"],
                result_info["component_type"],
                f"value.{component}",
            )
            for component in source_info["components"]
        ]
        helper = (
            f"__device__ inline {result_info['type']} {helper_name}"
            f"({source_info['type']} value)\n"
            "{\n"
            f"    return {result_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_hip_scalar_bitcast(self, source_component, target_component, value):
        if source_component == target_component:
            return value
        if source_component == "float" and target_component == "int":
            return f"__float_as_int({value})"
        if source_component == "float" and target_component == "uint":
            return f"__float_as_uint({value})"
        if source_component == "int" and target_component == "float":
            return f"__int_as_float({value})"
        if source_component == "uint" and target_component == "float":
            return f"__uint_as_float({value})"
        if source_component == "int" and target_component == "uint":
            return f"static_cast<unsigned int>({value})"
        if source_component == "uint" and target_component == "int":
            return f"static_cast<int>({value})"
        return value

    def generate_warp_sync_builtin_call(self, func_name, raw_args, args):
        return hip_cuda_warp_call(self, func_name, raw_args, args)
