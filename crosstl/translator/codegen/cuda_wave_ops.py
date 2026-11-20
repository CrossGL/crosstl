"""Mesh, ray-tracing, wave and warp operations of the CUDA generator, plus shader clocks."""

from ..refusals import CudaRefusal
from .cuda_tables import (
    CUDA_WARP_SYNC_BUILTIN_ARITIES,
    CUDA_WAVE_BOOL_RESULT_OPS,
    CUDA_WAVE_OP_ARITIES,
    CUDA_WAVE_PREDICATE_ARGUMENT_OPS,
    CUDA_WAVE_UINT_RESULT_OPS,
    CUDA_WAVE_UVEC4_RESULT_OPS,
)
from .warp_wave_helpers import (
    CUDA_WARP,
    wave_first_lane_test,
    wave_helper_body,
)


class CudaWaveOpMixin:
    """Wave op members of ``CudaCodeGen``."""

    def visit_MeshOpNode(self, node):
        operation = getattr(node, "operation", "")
        raw_args = getattr(node, "arguments", []) or []
        args = [self.visit(arg) for arg in raw_args]
        return self.generate_mesh_task_call_expression(operation, raw_args, args)

    def generate_mesh_task_call_expression(self, operation, raw_args, args):
        stage_name = getattr(self, "current_stage_name", None)
        if operation == "SetMeshOutputCounts":
            if stage_name != "mesh":
                raise CudaRefusal(
                    "CUDA SetMeshOutputCounts is only valid in mesh stages"
                )
            if len(raw_args) != 2:
                raise CudaRefusal(
                    "CUDA SetMeshOutputCounts requires exactly two arguments"
                )
            return f"cgl_cuda_set_mesh_output_counts({', '.join(args)})"

        if operation == "DispatchMesh":
            if stage_name not in {"task", "object", "amplification"}:
                raise CudaRefusal(
                    "CUDA DispatchMesh is only valid in task, object, or "
                    "amplification stages"
                )
            if len(raw_args) not in {3, 4}:
                raise CudaRefusal(
                    "CUDA DispatchMesh requires exactly three arguments, or four "
                    "with a task payload"
                )
            return f"cgl_cuda_dispatch_mesh({', '.join(args)})"

        return None

    def visit_RayTracingOpNode(self, node):
        operation = getattr(node, "operation", "")
        if operation in self.function_return_types:
            args = ", ".join(
                self.visit(arg) for arg in getattr(node, "arguments", []) or []
            )
            return f"{operation}({args})"
        return self.generate_ray_tracing_call_expression(
            operation,
            getattr(node, "arguments", []),
        )

    def generate_ray_tracing_call_expression(self, operation, arguments):
        if operation not in {
            "TraceRay",
            "CallShader",
            "ReportHit",
            "IgnoreHit",
            "AcceptHitAndEndSearch",
        }:
            return None

        self.require_cuda_ray_runtime_helpers()
        generated_args = [self.visit(arg) for arg in arguments or []]
        if operation == "TraceRay" and len(generated_args) == 11:
            ray_desc = (
                "CglRayDesc("
                f"{generated_args[6]}, {generated_args[7]}, "
                f"{generated_args[8]}, {generated_args[9]}"
                ")"
            )
            generated_args = generated_args[:6] + [ray_desc, generated_args[10]]
        elif operation == "ReportHit" and len(generated_args) == 2:
            generated_args.append("CglBuiltInTriangleIntersectionAttributes{}")

        helper_name = self.convert_builtin_function(operation)
        return f"{helper_name}({', '.join(generated_args)})"

    def visit_RayQueryOpNode(self, node):
        operation = getattr(node, "operation", "")
        helper_name = self.require_cuda_ray_query_helper(operation)
        args = [self.visit(getattr(node, "query_expr", None))]
        args.extend(self.visit(arg) for arg in getattr(node, "arguments", []) or [])
        return f"{helper_name}({', '.join(args)})"

    def visit_WaveOpNode(self, node):
        raw_args = list(getattr(node, "arguments", []) or [])
        args = [self.visit(arg) for arg in raw_args]
        return self.generate_wave_operation(
            getattr(node, "operation", "WaveOp"), raw_args, args
        )

    def generate_wave_operation(self, operation, raw_args, args):
        expected_count = CUDA_WAVE_OP_ARITIES.get(operation)
        if expected_count is None:
            raise CudaRefusal(f"Unsupported CUDA wave intrinsic {operation}")
        if len(raw_args) != expected_count:
            raise CudaRefusal(
                f"CUDA wave intrinsic {operation} requires {expected_count} "
                f"argument{'s' if expected_count != 1 else ''}, got {len(raw_args)}"
            )

        if operation == "WaveGetLaneCount":
            return "warpSize"
        if operation == "WaveGetLaneIndex":
            return self.cuda_warp_lane_index_expression()
        if operation == "WaveIsFirstLane":
            return wave_first_lane_test(
                CUDA_WARP, self.cuda_warp_lane_index_expression()
            )

        diagnostic = self.cuda_wave_operation_diagnostic(operation, raw_args)
        if diagnostic is not None:
            return diagnostic

        helper_name = self.require_wave_helper(operation, raw_args)
        return f"{helper_name}({', '.join(args)})"

    def require_wave_helper(self, operation, raw_args):
        result_type = self.map_type(self.wave_result_type(operation, raw_args))
        arg_types = [
            self.wave_argument_type(operation, index, arg)
            for index, arg in enumerate(raw_args)
        ]
        helper_name = self.wave_helper_name(operation, result_type, arg_types)
        if helper_name in self.helper_functions:
            return helper_name

        parameter_names = self.wave_helper_parameter_names(operation, len(arg_types))
        params = [
            f"{arg_type} {parameter_name}"
            for parameter_name, arg_type in zip(parameter_names, arg_types)
        ]
        body = wave_helper_body(
            CUDA_WARP,
            operation,
            result_type,
            arg_types[0] if arg_types else result_type,
            self.cuda_warp_lane_index_expression(),
        )
        if body is None:
            raise CudaRefusal(f"Unsupported CUDA wave intrinsic {operation}")
        helper = (
            f"__device__ inline {result_type} {helper_name}({', '.join(params)})\n"
            "{\n"
            f"{body}"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def cuda_wave_operation_diagnostic(self, operation, raw_args):
        if operation in {
            "WaveActiveAllTrue",
            "WaveActiveAnyTrue",
            "WaveActiveBallot",
            "WaveActiveCountBits",
            "WavePrefixCountBits",
            "WaveMultiPrefixCountBits",
        }:
            arg_type = self.type_name_string(self.expression_result_type(raw_args[0]))
            if arg_type is not None and self.map_type(arg_type) != "bool":
                return self.unsupported_cuda_wave_operation(
                    operation, raw_args, "requires bool predicate input"
                )
            if (
                operation.startswith("WaveMultiPrefix")
                and len(raw_args) > 1
                and self.map_type(self.expression_result_type(raw_args[1])) != "uint4"
            ):
                return self.unsupported_cuda_wave_operation(
                    operation, raw_args, "requires uvec4 partition mask"
                )
            return None

        scalar_ops = {
            "WaveActiveSum",
            "WaveActiveProduct",
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
            "WaveActiveMin",
            "WaveActiveMax",
            "WaveActiveAllEqual",
            "WaveReadLaneAt",
            "WaveReadLaneFirst",
            "WavePrefixSum",
            "WavePrefixProduct",
            "QuadReadAcrossX",
            "QuadReadAcrossY",
            "QuadReadAcrossDiagonal",
            "QuadReadLaneAt",
            "WaveMatch",
            "WaveMultiPrefixSum",
            "WaveMultiPrefixProduct",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        }
        if operation not in scalar_ops or not raw_args:
            return None

        value_type = self.wave_argument_type(operation, 0, raw_args[0])
        if self.vector_type_info(value_type) is not None:
            return self.unsupported_cuda_wave_operation(
                operation, raw_args, "requires scalar input"
            )

        kind = self.cuda_wave_scalar_kind(value_type)
        if kind is None:
            return self.unsupported_cuda_wave_operation(
                operation, raw_args, f"does not support {value_type or 'unknown'} input"
            )

        if operation in {
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        } and kind not in {"signed", "unsigned"}:
            return self.unsupported_cuda_wave_operation(
                operation, raw_args, "requires integer scalar input"
            )

        if (
            operation in {
                "WaveActiveSum",
                "WaveActiveProduct",
                "WaveActiveMin",
                "WaveActiveMax",
                "WavePrefixSum",
                "WavePrefixProduct",
                "WaveMultiPrefixSum",
                "WaveMultiPrefixProduct",
            }
            and kind == "bool"
        ):
            return self.unsupported_cuda_wave_operation(
                operation, raw_args, "requires numeric scalar input"
            )

        if (
            operation.startswith("WaveMultiPrefix")
            and len(raw_args) > 1
            and self.map_type(self.expression_result_type(raw_args[1])) != "uint4"
        ):
            return self.unsupported_cuda_wave_operation(
                operation, raw_args, "requires uvec4 partition mask"
            )

        return None

    def unsupported_cuda_wave_operation(self, operation, raw_args, reason):
        result_type = self.wave_result_type(operation, raw_args)
        fallback = self.diagnostic_zero_value_for_type(result_type)
        return self.refuse_lowering(
            f"CUDA wave intrinsic {operation}: {reason}", fallback
        )

    def cuda_wave_scalar_kind(self, type_name):
        mapped_type = self.map_type(type_name)
        if mapped_type in {"bool"}:
            return "bool"
        if mapped_type in {
            "char",
            "short",
            "int",
            "long",
            "long long",
            "i8",
            "i16",
            "i32",
            "i64",
        }:
            return "signed"
        if mapped_type in {
            "unsigned char",
            "unsigned short",
            "unsigned int",
            "unsigned long",
            "unsigned long long",
            "uint",
            "u8",
            "u16",
            "u32",
            "u64",
        }:
            return "unsigned"
        if mapped_type in {"float", "double", "half"}:
            return "float"
        return None

    def wave_result_type(self, operation, raw_args):
        if operation in CUDA_WAVE_UINT_RESULT_OPS:
            return "uint"
        if operation in CUDA_WAVE_BOOL_RESULT_OPS:
            return "bool"
        if operation in CUDA_WAVE_UVEC4_RESULT_OPS:
            return "uvec4"
        if raw_args:
            return self.expression_result_type(raw_args[0]) or "uint"
        return "uint"

    def wave_argument_type(self, operation, index, arg):
        if index == 0 and operation in CUDA_WAVE_PREDICATE_ARGUMENT_OPS:
            return self.map_type("bool")
        if index == 1 and operation.startswith("WaveMultiPrefix"):
            return self.map_type("uvec4")
        if index == 1 and operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            return self.map_type("uint")
        return self.map_type(self.expression_result_type(arg) or "uint")

    def wave_helper_name(self, operation, result_type, arg_types):
        suffix_parts = [self.wave_type_suffix(result_type)]
        suffix_parts.extend(self.wave_type_suffix(arg_type) for arg_type in arg_types)
        return (
            f"cgl_cuda_{self.wave_operation_suffix(operation)}_{'_'.join(suffix_parts)}"
        )

    def wave_operation_suffix(self, operation):
        name = operation[0].lower() + operation[1:]
        suffix = []
        for char in name:
            if char.isupper():
                suffix.append("_")
                suffix.append(char.lower())
            else:
                suffix.append(char)
        return "".join(suffix)

    def wave_type_suffix(self, type_name):
        suffix = type_name.replace("unsigned int", "uint")
        suffix = suffix.replace(" ", "_").replace("*", "_ptr")
        suffix = suffix.replace("&", "_ref").replace("::", "_")
        return "".join(char if char.isalnum() else "_" for char in suffix).strip("_")

    CUDA_SHADER_CLOCK_BUILTINS = {
        "clockARB",
        "clock2x32ARB",
        "clockRealtimeEXT",
        "clockRealtime2x32EXT",
    }

    # The CrossGL result types, so arithmetic on a 2x32 pair goes through the
    # uint2 helpers as it does for any other uvec2 value.
    CUDA_SHADER_CLOCK_RESULT_TYPES = {
        "clockARB": "uint64_t",
        "clock2x32ARB": "uvec2",
        "clockRealtimeEXT": "uint64_t",
        "clockRealtime2x32EXT": "uvec2",
    }

    def generate_cuda_shader_clock_call(self, func_name, raw_args):
        """GL_ARB_shader_clock and GL_EXT_shader_realtime_clock built-ins.

        clockARB reads the per-multiprocessor counter, which is CUDA's clock64().
        clockRealtimeEXT reads the device-wide nanosecond timer, the PTX %globaltimer
        register. The 2x32 forms return the same counter as a uint2 (low, high).
        """
        if func_name not in self.CUDA_SHADER_CLOCK_BUILTINS:
            return None
        if raw_args:
            raise CudaRefusal(
                f"CUDA shader clock builtin {func_name} requires 0 arguments; "
                f"got {len(raw_args)}"
            )
        if func_name == "clockARB":
            return "clock64()"
        if func_name == "clockRealtimeEXT":
            return self.require_cuda_realtime_clock_helper()
        if func_name == "clock2x32ARB":
            return self.require_cuda_clock_pair_helper(
                "cgl_cuda_clock2x32", "(unsigned long long)clock64()"
            )
        realtime = self.require_cuda_realtime_clock_helper()
        return self.require_cuda_clock_pair_helper(
            "cgl_cuda_realtime_clock2x32", realtime
        )

    def require_cuda_realtime_clock_helper(self):
        helper_name = "cgl_cuda_realtime_clock"
        if helper_name not in self.helper_functions:
            self.helper_functions[helper_name] = (
                f"__device__ inline unsigned long long {helper_name}()\n"
                "{\n"
                "    unsigned long long timer;\n"
                '    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));\n'
                "    return timer;\n"
                "}"
            )
        return f"{helper_name}()"

    def require_cuda_clock_pair_helper(self, helper_name, source):
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

    def generate_warp_sync_builtin_call(self, func_name, raw_args, args):
        expected_counts = CUDA_WARP_SYNC_BUILTIN_ARITIES.get(func_name)
        if expected_counts is None:
            return None
        if len(raw_args) not in expected_counts:
            expected_list = " or ".join(str(count) for count in expected_counts)
            raise CudaRefusal(
                f"CUDA warp sync builtin {func_name} requires {expected_list} "
                f"arguments, got {len(raw_args)}"
            )
        if func_name == "__shfl_down_sync":
            helper_name = self.require_cuda_shfl_down_sync_helper(
                raw_args[1],
                has_width=len(raw_args) == 4,
            )
            return f"{helper_name}({', '.join(args)})"
        return None

    def require_cuda_shfl_down_sync_helper(self, value_arg, has_width=False):
        value_type = self.map_type(self.expression_result_type(value_arg) or "uint")
        width_suffix = "_width" if has_width else ""
        helper_name = (
            f"cgl_cuda_shfl_down_sync{width_suffix}_"
            f"{self.wave_type_suffix(value_type)}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        width_param = ", int width" if has_width else ""
        width_arg = ", width" if has_width else ""
        width_ignore = "    (void)width;\n" if has_width else ""
        helper = (
            f"__device__ inline {value_type} {helper_name}"
            f"(unsigned int mask, {value_type} value, unsigned int delta{width_param})\n"
            "{\n"
            "#if defined(__CUDA_ARCH__)\n"
            f"    return __shfl_down_sync(mask, value, delta{width_arg});\n"
            "#else\n"
            "    (void)mask;\n"
            "    (void)delta;\n"
            f"{width_ignore}"
            "    return value;\n"
            "#endif\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def wave_helper_parameter_names(self, operation, arg_count):
        if arg_count == 1:
            if operation in CUDA_WAVE_PREDICATE_ARGUMENT_OPS:
                return ["predicate"]
            return ["value"]
        if operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            return ["value", "lane"]
        if operation == "WaveMultiPrefixCountBits":
            return ["predicate", "mask"]
        return ["value", "mask"]
