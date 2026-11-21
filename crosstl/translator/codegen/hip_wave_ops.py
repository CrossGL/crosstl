"""Mesh, ray-tracing and wave operations of the HIP generator."""

from ..ast import (
    MeshOpNode,
    RayQueryOpNode,
    RayTracingOpNode,
    WaveOpNode,
)
from ..refusals import HipRefusal
from .gpu_subgroup_builtins import warp_lane_index_expression
from .hip_tables import (
    HIP_WAVE_BOOL_RESULT_OPS,
    HIP_WAVE_OP_ARITIES,
    HIP_WAVE_PREDICATE_ARGUMENT_OPS,
    HIP_WAVE_UINT_RESULT_OPS,
    HIP_WAVE_UVEC4_RESULT_OPS,
)
from .warp_wave_helpers import (
    HIP_WAVEFRONT,
    wave_first_lane_test,
    wave_helper_body,
)


class HipWaveOpMixin:
    """Wave op members of ``HipCodeGen``."""

    def visit_MeshOpNode(self, node: MeshOpNode) -> str:
        operation = getattr(node, "operation", "")
        raw_args = getattr(node, "arguments", []) or []
        args = [self.visit(arg) for arg in raw_args]
        return self.generate_mesh_task_call_expression(operation, raw_args, args)

    def generate_mesh_task_call_expression(self, operation, raw_args, args):
        stage_name = getattr(self, "current_stage_name", None)
        if operation == "SetMeshOutputCounts":
            if stage_name != "mesh":
                raise HipRefusal("HIP SetMeshOutputCounts is only valid in mesh stages")
            if len(raw_args) != 2:
                raise HipRefusal(
                    "HIP SetMeshOutputCounts requires exactly two arguments"
                )
            return f"cgl_hip_set_mesh_output_counts({', '.join(args)})"

        if operation == "DispatchMesh":
            if stage_name not in {"task", "object", "amplification"}:
                raise HipRefusal(
                    "HIP DispatchMesh is only valid in task, object, or "
                    "amplification stages"
                )
            if len(raw_args) not in {3, 4}:
                raise HipRefusal(
                    "HIP DispatchMesh requires exactly three arguments, or four "
                    "with a task payload"
                )
            return f"cgl_hip_dispatch_mesh({', '.join(args)})"

        return None

    def visit_RayTracingOpNode(self, node: RayTracingOpNode) -> str:
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

        self.require_hip_ray_runtime_helpers()
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

        # function_map names a runtime helper for each of the five operations accepted above.
        helper_name = self.function_map[operation]
        return f"{helper_name}({', '.join(generated_args)})"

    def visit_RayQueryOpNode(self, node: RayQueryOpNode) -> str:
        operation = getattr(node, "operation", "")
        helper_name = self.require_hip_ray_query_helper(operation)
        args = [self.visit(getattr(node, "query_expr", None))]
        args.extend(self.visit(arg) for arg in getattr(node, "arguments", []) or [])
        return f"{helper_name}({', '.join(args)})"

    def visit_WaveOpNode(self, node: WaveOpNode) -> str:
        raw_args = list(getattr(node, "arguments", []) or [])
        args = [self.visit(arg) for arg in raw_args]
        return self.generate_wave_operation(
            getattr(node, "operation", "WaveOp"), raw_args, args
        )

    def generate_wave_operation(self, operation, raw_args, args):
        expected_count = HIP_WAVE_OP_ARITIES.get(operation)
        if expected_count is None:
            raise HipRefusal(f"Unsupported HIP wave intrinsic {operation}")
        if len(raw_args) != expected_count:
            raise HipRefusal(
                f"HIP wave intrinsic {operation} requires {expected_count} "
                f"argument{'s' if expected_count != 1 else ''}, got {len(raw_args)}"
            )

        if operation == "WaveGetLaneCount":
            return "warpSize"
        if operation == "WaveGetLaneIndex":
            # Warps are formed from linear thread IDs, so a block of more than one dimension needs all three.
            return warp_lane_index_expression()
        if operation == "WaveIsFirstLane":
            return wave_first_lane_test(HIP_WAVEFRONT, warp_lane_index_expression())

        diagnostic = self.wave_operation_diagnostic(operation, raw_args)
        if diagnostic is not None:
            return diagnostic

        helper_name = self.require_wave_helper(operation, raw_args)
        return f"{helper_name}({', '.join(args)})"

    def wave_operation_diagnostic(self, operation, raw_args):
        reason = self.unsupported_wave_operation_reason(operation, raw_args)
        if reason is None:
            return None
        fallback = self.diagnostic_zero_value_for_type(
            self.wave_result_type(operation, raw_args)
        )
        return self.refuse_lowering(
            f"{self.resource_backend_name()} wave intrinsic: {operation} {reason}",
            fallback,
        )

    def unsupported_wave_operation_reason(self, operation, raw_args):
        if not raw_args:
            return None

        if operation in HIP_WAVE_PREDICATE_ARGUMENT_OPS:
            predicate_type = self.wave_diagnostic_type_name(raw_args[0])
            if not self.wave_is_scalar_component_type(predicate_type, {"bool"}):
                return f"predicate must be bool, got {predicate_type}"

        if operation in {"WaveReadLaneAt", "QuadReadLaneAt"} and len(raw_args) > 1:
            lane_type = self.wave_diagnostic_type_name(raw_args[1])
            if not self.wave_is_scalar_component_type(lane_type, {"int", "uint"}):
                return f"lane index must be scalar int or uint, got {lane_type}"

        if operation.startswith("WaveMultiPrefix") and len(raw_args) > 1:
            mask_type = self.wave_diagnostic_type_name(raw_args[1])
            if self.map_type(mask_type) != "uint4":
                return f"partition mask must be uvec4, got {mask_type}"

        if operation in {
            "WaveActiveSum",
            "WaveActiveProduct",
            "WaveActiveMin",
            "WaveActiveMax",
            "WavePrefixSum",
            "WavePrefixProduct",
            "WaveMultiPrefixSum",
            "WaveMultiPrefixProduct",
        }:
            value_type = self.wave_diagnostic_type_name(raw_args[0])
            if not self.wave_has_scalar_or_vector_component(
                value_type, {"float", "double", "int", "uint"}
            ):
                return "value must be numeric scalar or vector, " f"got {value_type}"

        if operation in {
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        }:
            value_type = self.wave_diagnostic_type_name(raw_args[0])
            if not self.wave_has_scalar_or_vector_component(
                value_type, {"int", "uint"}
            ):
                return "value must be integer scalar or vector, " f"got {value_type}"

        if operation in {
            "WaveActiveAllEqual",
            "WaveMatch",
            "WaveReadLaneAt",
            "WaveReadLaneFirst",
            "QuadReadLaneAt",
            "QuadReadAcrossX",
            "QuadReadAcrossY",
            "QuadReadAcrossDiagonal",
        }:
            value_type = self.wave_diagnostic_type_name(raw_args[0])
            if not self.wave_has_scalar_or_vector_component(
                value_type, {"bool", "float", "double", "int", "uint"}
            ):
                return "value must be scalar or vector, " f"got {value_type}"

        return None

    def wave_diagnostic_type_name(self, arg):
        type_name = self.expression_result_type(arg)
        if type_name is None:
            return "unknown"
        return self.map_type(type_name)

    def wave_is_scalar_component_type(self, type_name, components):
        if self.vector_type_info(type_name) is not None:
            return False
        return self.scalar_component_type(type_name) in components

    def wave_has_scalar_or_vector_component(self, type_name, components):
        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            return vector_info["component_type"] in components
        return self.scalar_component_type(type_name) in components

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
            HIP_WAVEFRONT,
            operation,
            result_type,
            arg_types[0] if arg_types else result_type,
            warp_lane_index_expression(),
        )
        if body is None:
            raise HipRefusal(f"Unsupported HIP wave intrinsic {operation}")
        helper = (
            f"__device__ inline {result_type} {helper_name}({', '.join(params)})\n"
            "{\n"
            f"{body}"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def wave_result_type(self, operation, raw_args):
        if operation in HIP_WAVE_UINT_RESULT_OPS:
            return "uint"
        if operation in HIP_WAVE_BOOL_RESULT_OPS:
            return "bool"
        if operation in HIP_WAVE_UVEC4_RESULT_OPS:
            return "uvec4"
        if raw_args:
            return self.expression_result_type(raw_args[0]) or "uint"
        return "uint"

    def wave_argument_type(self, operation, index, arg):
        if index == 0 and operation in HIP_WAVE_PREDICATE_ARGUMENT_OPS:
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
            f"cgl_hip_{self.wave_operation_suffix(operation)}_{'_'.join(suffix_parts)}"
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

    def wave_helper_parameter_names(self, operation, arg_count):
        if arg_count == 1:
            if operation in HIP_WAVE_PREDICATE_ARGUMENT_OPS:
                return ["predicate"]
            return ["value"]
        if operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            return ["value", "lane"]
        if operation == "WaveMultiPrefixCountBits":
            return ["predicate", "mask"]
        return ["value", "mask"]
