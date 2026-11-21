"""Function-call lowering of the CUDA generator: the call visitor, math names, bit casts and integer bit functions."""

from ..ast import FunctionCallNode
from ..refusals import CudaRefusal
from .c_pointer_math import generate_pointer_math_call
from .c_vector_reductions import generate_vector_reduction_call
from .cuda_pointer_operands import cuda_pointer_operand_call
from .cuda_refusals import CudaLoweringRefused
from .cuda_tables import (
    CUDA_64BIT_BITCAST_FUNCTIONS,
    CUDA_BITCAST_FUNCTION_TARGETS,
    CUDA_INTEGER_BIT_FUNCTION_ALIASES,
    CUDA_WAVE_OP_ARITIES,
)
from .generic_function_utils import (
    generate_numeric_trait_method_call,
    generate_static_generic_numeric_call,
    generic_function_call_name,
    reject_unresolved_generic_function_call,
)


class CudaFunctionCallMixin:
    """Function call members of ``CudaCodeGen``."""

    def visit_FunctionCallNode(self, node):
        """Visit function call"""
        function_expr = getattr(node, "function", getattr(node, "name", None))
        if hasattr(node, "function"):
            func_name = self.visit(function_expr)
        else:
            func_name = getattr(node, "name", "unknown")

        raw_args = []
        if hasattr(node, "arguments"):
            raw_args = node.arguments
        elif hasattr(node, "args"):
            raw_args = node.args

        numeric_trait_call = generate_numeric_trait_method_call(
            self,
            function_expr,
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
        else:
            reject_unresolved_generic_function_call(self, func_name, "CUDA")

        args = [self.visit(arg) for arg in raw_args]

        half_constructor = self.cuda_half_constructor_expression(
            func_name, raw_args, args
        )
        if half_constructor is not None:
            return half_constructor

        if func_name == "lambda":
            return self.generate_lambda_expression(raw_args)
        if func_name in CUDA_WAVE_OP_ARITIES:
            return self.generate_wave_operation(func_name, raw_args, args)
        if func_name in {"SetMeshOutputCounts", "DispatchMesh"}:
            return self.generate_mesh_task_call_expression(func_name, raw_args, args)

        is_user_function = self.is_user_defined_function(func_name)
        if not is_user_function:
            scalar_alias_constructor = self.cuda_scalar_alias_constructor_expression(
                func_name, args
            )
            if scalar_alias_constructor is not None:
                return scalar_alias_constructor
        if not is_user_function:
            ray_tracing_call = self.generate_ray_tracing_call_expression(
                func_name, raw_args
            )
            if ray_tracing_call is not None:
                return ray_tracing_call
        if not is_user_function:
            buffer_call = self.generate_buffer_call(
                function_expr, func_name, raw_args, args
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

            bitcast_call = self.generate_cuda_bitcast_call(func_name, raw_args, args)
            if bitcast_call is not None:
                return bitcast_call

            integer_bit_call = self.generate_cuda_integer_bit_call(
                func_name, raw_args, args
            )
            if integer_bit_call is not None:
                return integer_bit_call

            warp_sync_call = self.generate_warp_sync_builtin_call(
                func_name, raw_args, args
            )
            if warp_sync_call is not None:
                return warp_sync_call

            pointer_operand_call = cuda_pointer_operand_call(
                self, func_name, raw_args, args, CudaRefusal
            )
            if pointer_operand_call is not None:
                return pointer_operand_call

            shader_clock_call = self.generate_cuda_shader_clock_call(
                func_name, raw_args
            )
            if shader_clock_call is not None:
                return shader_clock_call

        args = self.cuda_user_function_call_arguments(func_name, raw_args, args)
        if is_user_function:
            return f"{func_name}({', '.join(args)})"

        if func_name in self.synchronization_builtins and raw_args:
            raise CudaRefusal(
                f"CUDA synchronization builtin '{func_name}' requires 0 "
                f"arguments; got {len(raw_args)}"
            )

        reduction_call = generate_vector_reduction_call(
            self, "CUDA", func_name, raw_args, args
        )
        if reduction_call is not None:
            return reduction_call

        if func_name == "abs" and len(args) == 1:
            abs_call = self.generate_abs_call(raw_args, args)
            if abs_call is not None:
                return abs_call

        if func_name == "sign" and len(args) == 1:
            sign_call = self.generate_sign_call(raw_args, args)
            if sign_call is not None:
                return sign_call

        if func_name == "mod" and len(args) == 2:
            mod_call = self.generate_mod_call(raw_args, args)
            if mod_call is not None:
                return mod_call

        if func_name in {"fract", "frac"} and len(args) == 1:
            fract_call = self.generate_fract_call(raw_args, args)
            if fract_call is not None:
                return fract_call

        if func_name == "clamp" and len(args) == 3:
            return self.generate_clamp_call(raw_args, args)

        if func_name in {"min", "max"} and len(args) == 2:
            min_max_call = self.generate_min_max_call(func_name, raw_args, args)
            if min_max_call is not None:
                return min_max_call

        if func_name in {"atan", "atan2"} and len(args) == 2:
            atan2_call = self.generate_atan2_call(raw_args, args)
            if atan2_call is not None:
                return atan2_call
            if func_name == "atan":
                scalar_atan2_call = self.generate_scalar_atan2_call(raw_args, args)
                if scalar_atan2_call is not None:
                    return scalar_atan2_call

        if func_name in {"fma", "mad"} and len(args) == 3:
            result_type = self.fused_multiply_add_result_type(raw_args)
            fma_call = self.generate_fused_multiply_add_call(raw_args, args)
            if fma_call is not None:
                if result_type is not None:
                    node.expression_type = result_type
                return fma_call

        if func_name in {"mix", "lerp"} and len(args) == 3:
            mix_call = self.generate_mix_call(raw_args, args)
            if mix_call is not None:
                return mix_call

        if func_name == "saturate" and len(args) == 1:
            saturate_call = self.generate_saturate_call(raw_args, args)
            if saturate_call is not None:
                return saturate_call

        if func_name == "step" and len(args) == 2:
            step_call = self.generate_step_call(raw_args, args)
            if step_call is not None:
                return step_call

        if func_name == "smoothstep" and len(args) == 3:
            smoothstep_call = self.generate_smoothstep_call(raw_args, args)
            if smoothstep_call is not None:
                return smoothstep_call

        pointer_math_call = generate_pointer_math_call(
            self, CudaLoweringRefused, func_name, raw_args, args
        )
        if pointer_math_call is not None:
            return pointer_math_call

        if func_name in {
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

        math_func_name = self.cuda_math_function_name(func_name)
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

        union_constructor = self.native_unions.constructor(func_name, raw_args, None)
        if union_constructor is not None:
            return union_constructor
        if func_name in self.struct_member_types:
            args = self.struct_constructor_arguments(func_name, raw_args, args)
            if func_name in self.structs_by_name:
                # The structure has no constructor; aggregate initialization sets its members in order.
                mapped_type = self.convert_crossgl_type_to_cuda(func_name)
                return f"{mapped_type}{{{', '.join(args)}}}"

        scalar_math_call = self.generate_scalar_math_call(
            math_func_name, raw_args, args
        )
        if scalar_math_call is not None:
            return scalar_math_call

        args_str = ", ".join(args)

        euclidean_norm_name = self.cuda_euclidean_norm_function_name(
            func_name, raw_args
        )
        if euclidean_norm_name is not None:
            return f"{euclidean_norm_name}({args_str})"

        func_name = self.convert_builtin_function(func_name)
        return f"{func_name}({args_str})"

    def cuda_math_function_name(self, func_name):
        """The math lowering's name for ``func_name``: ``inverseSqrt`` and ``rsqrt`` are spellings of ``inversesqrt``.

        Any other name is its own math lowering name; the math lowerings return ``None`` for a name they do not
        handle, and the call is then refused if it names another language's built-in function.
        """
        if func_name in {"inverseSqrt", "rsqrt"}:
            return "inversesqrt"
        return func_name

    def normalize_cuda_integer_bit_function(self, func_name):
        if func_name in CUDA_INTEGER_BIT_FUNCTION_ALIASES:
            return CUDA_INTEGER_BIT_FUNCTION_ALIASES[func_name]
        # Already the CrossGL spelling, or not an integer bit function; callers check the result.
        return func_name

    def cuda_bitcast_result_type(self, func_name, raw_args):
        wide_bitcast = CUDA_64BIT_BITCAST_FUNCTIONS.get(func_name)
        if (
            wide_bitcast is not None
            and not self.is_user_defined_function(func_name)
            and len(raw_args or []) == 1
        ):
            return wide_bitcast[1]
        target_component = CUDA_BITCAST_FUNCTION_TARGETS.get(func_name)
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

    def generate_cuda_bitcast_call(self, func_name, raw_args, args):
        result_type = self.cuda_bitcast_result_type(func_name, raw_args)
        if result_type is None:
            return None

        wide_bitcast = CUDA_64BIT_BITCAST_FUNCTIONS.get(func_name)
        if wide_bitcast is not None:
            intrinsic = wide_bitcast[0]
            # The intrinsic takes one scalar, and CUDA converts any arithmetic scalar to
            # its parameter type. It has no vector or matrix form.
            source_type = self.expression_result_type(raw_args[0])
            if (
                source_type is not None
                and self.scalar_component_type(source_type) is None
            ):
                raise CudaLoweringRefused(
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
            helper_name = self.require_cuda_vector_bitcast_helper(
                source_info,
                result_info,
            )
            return f"{helper_name}({args[0]})"

        target_component = CUDA_BITCAST_FUNCTION_TARGETS[func_name]
        source_component = self.scalar_component_type(source_type)
        return self.format_cuda_scalar_bitcast(
            source_component,
            target_component,
            args[0],
        )

    def cuda_integer_bit_result_type(self, func_name, raw_args):
        original_func_name = func_name
        func_name = self.normalize_cuda_integer_bit_function(func_name)
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
                return self.vector_type_for_components(
                    "uint",
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

    def generate_cuda_integer_bit_call(self, func_name, raw_args, args):
        result_type = self.cuda_integer_bit_result_type(func_name, raw_args)
        if result_type is None:
            return None
        if len(raw_args or []) != 1 or len(args) != 1:
            return None

        operation = self.normalize_cuda_integer_bit_function(func_name)
        source_type = self.expression_result_type(raw_args[0])
        if source_type is None and operation == "countLeadingZeros":
            # A sized cast such as u64(x) has no inferred type; read its width from the cast so that
            # __clz does not truncate a 64-bit value.
            source_type = self.cuda_integer_cast_type(raw_args[0])
        source_info = self.vector_type_info(source_type)
        result_info = self.vector_type_info(result_type)
        if source_info is not None:
            if result_info is None or source_info["component_type"] not in {
                "int",
                "uint",
            }:
                return None
            helper_name = self.require_cuda_integer_bit_vector_helper(
                operation,
                source_info,
                result_info,
            )
            return f"{helper_name}({args[0]})"

        source_component = self.scalar_component_type(source_type)
        if source_component not in {"int", "uint", None}:
            return None
        return self.format_cuda_integer_bit_component(
            operation,
            args[0],
            source_component,
            self.scalar_component_type(result_type),
            source_type,
            result_type,
        )

    def cuda_integer_cast_type(self, node):
        """The target type of a scalar integer cast such as ``u64(x)``, or None."""
        if not isinstance(node, FunctionCallNode):
            return None
        function_expr = getattr(node, "function", getattr(node, "name", None))
        type_name = getattr(function_expr, "name", function_expr)
        if not isinstance(type_name, str) or self.is_user_defined_function(type_name):
            return None
        if self.scalar_component_type(type_name) in {"int", "uint"}:
            return type_name
        return None

    def require_cuda_integer_bit_vector_helper(
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
            self.format_cuda_integer_bit_component(
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

    def format_cuda_integer_bit_component(
        self,
        operation,
        value,
        source_component=None,
        result_component=None,
        source_type=None,
        result_type=None,
    ):
        bit_width = self.cuda_integer_bit_width(source_type)
        unsigned_cast_type = "unsigned long long" if bit_width == 64 else "unsigned int"
        signed_cast_type = "long long" if bit_width == 64 else "int"
        unsigned_value = (
            f"static_cast<{unsigned_cast_type}>({value})"
            if source_component == "int"
            else value
        )
        if operation == "bitCount":
            return f"__popc({unsigned_value})"
        if operation == "bitfieldReverse":
            intrinsic_name = "__brevll" if bit_width == 64 else "__brev"
            reversed_value = f"{intrinsic_name}({unsigned_value})"
            if result_component == "int":
                return f"static_cast<{signed_cast_type}>({reversed_value})"
            return reversed_value
        if operation == "countLeadingZeros":
            intrinsic_name = "__clzll" if bit_width == 64 else "__clz"
            return f"{intrinsic_name}({value})"
        if operation == "findLSB":
            return f"(__ffs({value}) - 1)"
        if operation == "findMSB":
            if source_component == "int":
                return (
                    f"(({value}) < 0 ? (31 - __clz(~({value}))) : "
                    f"(({value}) == 0 ? -1 : (31 - __clz({value}))))"
                )
            return f"(({value}) == 0 ? -1 : (31 - __clz({value})))"
        return f"{operation}({value})"

    def cuda_integer_bit_width(self, type_name):
        if type_name is None:
            return 32
        type_text = self.type_name_string(type_name)
        mapped_type = self.convert_crossgl_type_to_cuda(type_text)
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

    def require_cuda_vector_bitcast_helper(self, source_info, result_info):
        helper_name = self.sanitize_helper_name(
            f"cgl_{source_info['type']}_to_{result_info['type']}_bitcast"
        )
        if helper_name in self.helper_functions:
            return helper_name

        components = [
            self.format_cuda_scalar_bitcast(
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

    def format_cuda_scalar_bitcast(self, source_component, target_component, value):
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
