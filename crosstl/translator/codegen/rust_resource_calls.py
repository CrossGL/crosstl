"""Ray-tracing, ray-query and texture operations and mapped built-in function names in Rust output."""

import re

from ..ast import (
    FunctionCallNode,
    MemberAccessNode,
)
from ..refusals import RustRefusal
from .rust_exact_lookups import RUST_RAY_TRACING_INTRINSICS
from .rust_gpu_images import generate_rust_gpu_image_call


class RustResourceCallMixin:
    """Resource-operation and built-in call members of ``RustCodeGen``."""

    def generate_ray_tracing_op_expression(self, expr):
        operation = getattr(expr, "operation", "")
        if operation in self.user_function_names:
            args = ", ".join(
                self.generate_user_function_call_args(
                    operation,
                    getattr(expr, "arguments", []),
                )
            )
            return f"{self.rust_callable_name(operation)}({args})"

        return self.generate_ray_tracing_call_expression(
            operation,
            getattr(expr, "arguments", []),
        )

    def generate_ray_tracing_call_expression(self, operation, arguments):
        intrinsic = RUST_RAY_TRACING_INTRINSICS.get(operation)
        if intrinsic is None:
            return None

        func_name, argument_counts = intrinsic
        if len(arguments or []) not in argument_counts:
            counts = " or ".join(str(count) for count in argument_counts)
            return self.refuse_lowering(
                f"{operation} with {len(arguments or [])} arguments: the Rust runtime's "
                f"{func_name} takes {counts}",
                f"{func_name}()",
            )
        args = ", ".join(self.generate_expression(arg) for arg in arguments or [])
        if operation == "TraceRay" and len(arguments or []) == 11:
            generated_args = [self.generate_expression(arg) for arg in arguments]
            ray_desc = (
                "RayDesc { "
                f"origin: {generated_args[6]}, "
                f"t_min: {generated_args[7]}, "
                f"direction: {generated_args[8]}, "
                f"t_max: {generated_args[9]} "
                "}"
            )
            args = ", ".join(generated_args[:6] + [ray_desc, generated_args[10]])
        elif operation == "ReportHit" and len(arguments or []) == 2:
            generated_args = [self.generate_expression(arg) for arg in arguments]
            args = ", ".join(generated_args + ["()"])
        return f"{func_name}({args})"

    def generate_ray_query_op_expression(self, expr):
        operation = getattr(expr, "operation", "")
        receiver = self.generate_expression(getattr(expr, "query_expr", None))
        args = [receiver]
        args.extend(
            self.generate_expression(arg)
            for arg in getattr(expr, "arguments", []) or []
        )
        func_name = f"ray_query_{self.rust_snake_case_name(operation)}"
        return f"{func_name}({', '.join(args)})"

    def generate_texture_node_expression(self, expr):
        args = [expr.texture_expr]
        if getattr(expr, "sampler_expr", None) is not None:
            args.append(expr.sampler_expr)
        args.append(expr.coordinates)
        operation = "texture"
        if getattr(expr, "level", None) is not None:
            args.append(expr.level)
            operation = "textureLod"
        if getattr(expr, "offset", None) is not None:
            args.append(expr.offset)
            operation = (
                "textureLodOffset" if operation == "textureLod" else "textureOffset"
            )
        return self.generate_texture_operation_call(operation, args)

    def generate_texture_op_node_expression(self, expr):
        operation = getattr(expr, "operation", "")
        sampler_expr = getattr(expr, "sampler_expr", None)
        arguments = getattr(expr, "arguments", []) or []
        if operation in self.rust_hlsl_texture_member_operations():
            member_args = []
            if sampler_expr is not None:
                member_args.append(sampler_expr)
            member_args.extend(arguments)
            call = self.generate_hlsl_texture_member_call(
                operation,
                expr.texture_expr,
                member_args,
            )
            if call is not None:
                return call

        args = [expr.texture_expr]
        if sampler_expr is not None:
            args.append(sampler_expr)
        args.extend(arguments)
        call = self.generate_texture_operation_call(operation, args)
        if call is not None:
            return call
        raise RustRefusal(f"Unsupported Rust texture operation {operation}")

    def generate_texture_member_function_call(self, func_expr, args):
        member = getattr(func_expr, "member", None)
        obj_expr = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        if not self.is_texture_resource_type(self.expression_result_type(obj_expr)):
            return None
        call = self.generate_hlsl_texture_member_call(member, obj_expr, args or [])
        if call is not None:
            return call
        raise RustRefusal(f"Unsupported Rust texture member operation {member}")

    def generate_texture_operation_call(self, operation, args):
        image_call = generate_rust_gpu_image_call(self, operation, args)
        if image_call is not None:
            return image_call
        if operation in self.rust_hlsl_texture_member_operations():
            if not args:
                raise RustRefusal(f"Unsupported Rust texture operation {operation}")
            call = self.generate_hlsl_texture_member_call(
                operation,
                args[0],
                list(args[1:]),
            )
            if call is not None:
                return call
        if operation in {"Sample", "sample", "texture"}:
            helper_name = self.texture_sample_helper_name(args)
            return self.generate_texture_helper_call(helper_name, args)
        if operation == "SampleCmp":
            helper_name = (
                "texture_compare_sampler"
                if self.call_has_explicit_sampler_argument(args)
                else "texture_compare"
            )
            return self.generate_texture_helper_call(helper_name, args)
        if operation in self.function_map or str(operation).startswith("texture"):
            helper_name = self.mapped_function_name(operation, len(args or []), args)
            if helper_name != operation:
                return self.generate_texture_helper_call(helper_name, args)
        return None

    def texture_sample_helper_name(self, args):
        has_sampler = self.call_has_explicit_sampler_argument(args)
        shadow = self.call_uses_shadow_texture(args)
        if shadow:
            return "sample_shadow_sampler" if has_sampler else "sample_shadow"
        return "sample_sampler" if has_sampler else "sample"

    def texture_member_function_result_type(self, func_expr, args):
        member = getattr(func_expr, "member", None)
        obj_expr = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        texture_type = self.expression_result_type(obj_expr)
        if not self.is_texture_resource_type(texture_type):
            return None
        if member == "SampleCmp":
            return "float"
        if member in {
            "SampleCmpLevel",
            "SampleCmpLevelZero",
            "SampleCmpGrad",
            "CalculateLevelOfDetail",
            "CalculateLevelOfDetailUnclamped",
        }:
            return "float"
        if member == "GetDimensions":
            return "void"
        if member in {"Sample", "sample", "texture"}:
            return (
                "float"
                if self.is_shadow_texture_type(texture_type)
                else self.texture_sample_result_type(texture_type)
            )
        if str(member).startswith("Gather"):
            return self.texture_gather_result_type(texture_type)
        if member in self.rust_hlsl_texture_member_operations():
            return self.texture_sample_result_type(texture_type)
        return None

    def generate_texture_helper_call(self, helper_name, args):
        # rust-gpu output defines no texture helper functions: a texture read reaches here only when
        # rust_gpu_images has no spirv-std method for it, so it is refused rather than written as a call to a
        # helper that does not exist.
        self.validate_rust_texture_helper_call(helper_name, args)
        return self.refuse_lowering(
            f"texture operation: '{helper_name}' has no spirv-std lowering for these operands",
            "Default::default()",
        )

    def rust_hlsl_texture_member_operations(self):
        return {
            "Sample",
            "SampleLevel",
            "SampleLOD",
            "SampleGrad",
            "SampleBias",
            "SampleCmp",
            "SampleCmpLevel",
            "SampleCmpLevelZero",
            "SampleCmpGrad",
            "Gather",
            "GatherRed",
            "GatherGreen",
            "GatherBlue",
            "GatherAlpha",
            "GatherCmp",
            "GatherCmpRed",
            "GatherCmpGreen",
            "GatherCmpBlue",
            "GatherCmpAlpha",
            "Load",
            "GetDimensions",
            "CalculateLevelOfDetail",
            "CalculateLevelOfDetailUnclamped",
        }

    def generate_hlsl_texture_member_call(self, operation, texture_expr, member_args):
        member_args = list(member_args or [])
        texture_type = self.expression_result_type(texture_expr)

        def helper(helper_name, args):
            return self.generate_texture_helper_call(helper_name, args)

        if operation == "Sample":
            self.require_rust_texture_member_arg_count(operation, member_args, {2, 3})
            helper_name = (
                "sample_shadow_sampler"
                if self.is_shadow_texture_type(texture_type)
                else "sample_sampler"
            )
            if len(member_args) == 3:
                helper_name = "sample_offset_sampler"
            return helper(helper_name, [texture_expr, *member_args])

        if operation in {"sample", "texture"}:
            return self.generate_rust_lowercase_texture_sample_member_call(
                operation,
                texture_expr,
                member_args,
                texture_type,
                helper,
            )

        if operation in {"SampleLevel", "SampleLOD"}:
            self.require_rust_texture_member_arg_count(operation, member_args, {3, 4})
            helper_name = (
                "sample_lod_offset_sampler"
                if len(member_args) == 4
                else "sample_lod_sampler"
            )
            return helper(helper_name, [texture_expr, *member_args])

        if operation == "SampleGrad":
            self.require_rust_texture_member_arg_count(operation, member_args, {4, 5})
            helper_name = (
                "sample_grad_offset_sampler"
                if len(member_args) == 5
                else "sample_grad_sampler"
            )
            return helper(helper_name, [texture_expr, *member_args])

        if operation == "SampleBias":
            self.require_rust_texture_member_arg_count(operation, member_args, {3, 4})
            sampler, coord, bias = member_args[:3]
            if len(member_args) == 4:
                offset = member_args[3]
                return helper(
                    "sample_offset_bias_sampler",
                    [texture_expr, sampler, coord, offset, bias],
                )
            return helper("sample_bias_sampler", [texture_expr, sampler, coord, bias])

        if operation == "SampleCmp":
            self.require_rust_texture_member_arg_count(operation, member_args, {3, 4})
            helper_name = (
                "texture_compare_offset_sampler"
                if len(member_args) == 4
                else "texture_compare_sampler"
            )
            return helper(helper_name, [texture_expr, *member_args])

        if operation == "SampleCmpLevel":
            self.require_rust_texture_member_arg_count(operation, member_args, {4, 5})
            helper_name = (
                "texture_compare_lod_offset_sampler"
                if len(member_args) == 5
                else "texture_compare_lod_sampler"
            )
            return helper(helper_name, [texture_expr, *member_args])

        if operation == "SampleCmpLevelZero":
            self.require_rust_texture_member_arg_count(operation, member_args, {3, 4})
            helper_name = (
                "texture_compare_offset_sampler"
                if len(member_args) == 4
                else "texture_compare_sampler"
            )
            return helper(helper_name, [texture_expr, *member_args])

        if operation == "SampleCmpGrad":
            self.require_rust_texture_member_arg_count(operation, member_args, {5, 6})
            helper_name = (
                "texture_compare_grad_offset_sampler"
                if len(member_args) == 6
                else "texture_compare_grad_sampler"
            )
            return helper(helper_name, [texture_expr, *member_args])

        gather_components = {
            "GatherRed": "0",
            "GatherGreen": "1",
            "GatherBlue": "2",
            "GatherAlpha": "3",
        }
        if operation in {"Gather", *gather_components}:
            self.require_rust_texture_member_arg_count(operation, member_args, {2, 3})
            if operation == "Gather":
                helper_name = (
                    "texture_gather_offset_sampler"
                    if len(member_args) == 3
                    else "texture_gather_sampler"
                )
                return helper(helper_name, [texture_expr, *member_args])

            component = gather_components[operation]
            sampler, coord = member_args[:2]
            if len(member_args) == 3:
                offset = member_args[2]
                return helper(
                    "texture_gather_offset_component_sampler",
                    [texture_expr, sampler, coord, offset, component],
                )
            return helper(
                "texture_gather_component_sampler",
                [texture_expr, sampler, coord, component],
            )

        if operation in {
            "GatherCmp",
            "GatherCmpRed",
            "GatherCmpGreen",
            "GatherCmpBlue",
            "GatherCmpAlpha",
        }:
            self.require_rust_texture_member_arg_count(operation, member_args, {3, 4})
            helper_name = (
                "texture_gather_compare_offset_sampler"
                if len(member_args) == 4
                else "texture_gather_compare_sampler"
            )
            return helper(helper_name, [texture_expr, *member_args])

        if operation == "Load":
            self.require_rust_texture_member_arg_count(operation, member_args, {1, 2})
            if len(member_args) == 2 and not self.is_multisample_texture_type(
                texture_type
            ):
                return helper(
                    "texel_fetch_offset",
                    [texture_expr, member_args[0], "0", member_args[1]],
                )
            lod_or_sample = member_args[1] if len(member_args) == 2 else "0"
            return helper("texel_fetch", [texture_expr, member_args[0], lod_or_sample])

        if operation == "GetDimensions":
            generated_args = ", ".join(
                self.generate_expression(arg) for arg in member_args
            )
            if len(member_args) == 1:
                generated_args += ","
            return (
                f"texture_dimensions({self.generate_expression(texture_expr)}, "
                f"({generated_args}))"
            )

        if operation in {"CalculateLevelOfDetail", "CalculateLevelOfDetailUnclamped"}:
            self.require_rust_texture_member_arg_count(operation, member_args, {2})
            return helper("texture_calculate_lod", [texture_expr, *member_args])

        return None

    def generate_rust_lowercase_texture_sample_member_call(
        self,
        operation,
        texture_expr,
        member_args,
        texture_type,
        helper,
    ):
        self.require_rust_texture_member_arg_count(
            operation,
            member_args,
            {1, 2, 3, 4},
        )
        has_sampler = self.is_sampler_type(self.expression_result_type(member_args[0]))
        coord_index = 1 if has_sampler else 0
        if len(member_args) <= coord_index:
            raise RustRefusal(
                self.rust_texture_member_error(operation, "missing coordinates")
            )

        sampler_suffix = "_sampler" if has_sampler else ""
        base_args = [texture_expr]
        if has_sampler:
            base_args.append(member_args[0])
        coord = member_args[coord_index]
        tail_args = member_args[coord_index + 1 :]

        if not tail_args:
            helper_name = (
                f"sample_shadow{sampler_suffix}"
                if self.is_shadow_texture_type(texture_type)
                else f"sample{sampler_suffix}"
            )
            return helper(helper_name, [*base_args, coord])

        if len(tail_args) > 2:
            raise RustRefusal(
                self.rust_texture_member_error(
                    operation,
                    "expected at most 2 sample option argument(s), "
                    f"got {len(tail_args)}",
                )
            )

        option = self.rust_texture_sample_option(tail_args[0])
        offset = tail_args[1] if len(tail_args) == 2 else None
        shadow = self.is_shadow_texture_type(texture_type)

        if option is None:
            if offset is not None:
                raise RustRefusal(
                    self.rust_texture_member_error(
                        operation,
                        "unrecognized sample option before offset",
                    )
                )
            if shadow:
                raise RustRefusal(
                    self.rust_texture_member_error(
                        operation,
                        "shadow texture offsets are unavailable",
                    )
                )
            return helper(
                f"sample_offset{sampler_suffix}",
                [*base_args, coord, tail_args[0]],
            )

        option_name, option_args = option
        if option_name == "bias":
            if len(option_args) != 1:
                raise RustRefusal(
                    self.rust_texture_member_error(
                        operation,
                        "bias sample option requires 1 argument",
                    )
                )
            if offset is not None:
                if shadow:
                    raise RustRefusal(
                        self.rust_texture_member_error(
                            operation,
                            "shadow texture offsets are unavailable",
                        )
                    )
                return helper(
                    f"sample_offset_bias{sampler_suffix}",
                    [*base_args, coord, offset, option_args[0]],
                )
            helper_name = (
                f"sample_shadow_bias{sampler_suffix}"
                if shadow
                else f"sample_bias{sampler_suffix}"
            )
            return helper(helper_name, [*base_args, coord, option_args[0]])

        if shadow:
            raise RustRefusal(
                self.rust_texture_member_error(
                    operation,
                    f"{option_name} sample option is unavailable for shadow textures",
                )
            )

        if option_name == "level":
            if len(option_args) != 1:
                raise RustRefusal(
                    self.rust_texture_member_error(
                        operation,
                        "level sample option requires 1 argument",
                    )
                )
            helper_name = (
                f"sample_lod_offset{sampler_suffix}"
                if offset is not None
                else f"sample_lod{sampler_suffix}"
            )
            helper_args = [*base_args, coord, option_args[0]]
            if offset is not None:
                helper_args.append(offset)
            return helper(helper_name, helper_args)

        if option_name == "gradient":
            if len(option_args) != 2:
                raise RustRefusal(
                    self.rust_texture_member_error(
                        operation,
                        "gradient sample option requires 2 arguments",
                    )
                )
            helper_name = (
                f"sample_grad_offset{sampler_suffix}"
                if offset is not None
                else f"sample_grad{sampler_suffix}"
            )
            helper_args = [*base_args, coord, *option_args]
            if offset is not None:
                helper_args.append(offset)
            return helper(helper_name, helper_args)

        raise RustRefusal(
            self.rust_texture_member_error(
                operation,
                f"unrecognized sample option {option_name}",
            )
        )

    def rust_texture_member_error(self, operation, detail):
        return "Unsup" f"ported Rust texture member operation {operation}; {detail}"

    def rust_texture_sample_option(self, arg):
        if not isinstance(arg, FunctionCallNode):
            return None
        option_name = self.function_call_name(getattr(arg, "function", None))
        if option_name is None:
            return None
        option_name = str(option_name).rsplit("::", 1)[-1]
        option_args = list(getattr(arg, "arguments", getattr(arg, "args", [])) or [])
        if option_name == "level":
            return "level", option_args
        if option_name == "bias":
            return "bias", option_args
        if option_name in {"gradient2d", "gradient3d", "gradientcube"}:
            return "gradient", option_args
        return option_name, option_args

    def require_rust_texture_member_arg_count(self, operation, args, expected_counts):
        if len(args) not in expected_counts:
            expected = ", ".join(str(count) for count in sorted(expected_counts))
            raise RustRefusal(
                f"Unsupported Rust texture member operation {operation}; "
                f"expected {expected} argument(s), got {len(args)}"
            )

    def validate_rust_texture_helper_call(self, helper_name, args):
        if not args:
            return
        if not self.is_rust_texture_helper_name(helper_name):
            return

        texture_type = self.expression_result_type(args[0])
        if texture_type is not None and not self.is_texture_resource_type(texture_type):
            raise RustRefusal(
                f"Unsupported {helper_name} for Rust codegen; texture resource "
                f"required: {self.map_type(texture_type)}"
            )

        if helper_name.startswith(
            ("texture_compare", "texture_gather_compare")
        ) and not self.is_shadow_texture_type(texture_type):
            raise RustRefusal(
                f"Unsupported {helper_name} for Rust codegen; shadow texture "
                f"required: {self.map_type(texture_type)}"
            )

    def is_rust_texture_helper_name(self, helper_name):
        helper_name = str(helper_name)
        return helper_name.startswith(
            (
                "sample",
                "texel_fetch",
                "texture_size",
                "texture_query",
                "texture_samples",
                "texture_gather",
                "texture_compare",
                "texture_dimensions",
                "texture_calculate_lod",
            )
        )

    def rust_snake_case_name(self, name):
        name = str(name)
        name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        return name.replace("__", "_").strip("_").lower()

    def mapped_function_name(self, func_name, arg_count=None, arguments=None):
        if not isinstance(func_name, str):
            return func_name
        if arguments is not None:
            arg_count = len(arguments)
        has_sampler = self.call_has_explicit_sampler_argument(arguments)
        if func_name == "atan" and arg_count == 2:
            return "atan2"
        if func_name in {"textureSize", "textureDimensions"}:
            return "texture_size" if arg_count == 1 else "texture_size_lod"
        if func_name == "texture":
            if self.call_uses_shadow_texture(arguments):
                if has_sampler:
                    return (
                        "sample_shadow_bias_sampler"
                        if arg_count and arg_count >= 4
                        else "sample_shadow_sampler"
                    )
                return (
                    "sample_shadow_bias"
                    if arg_count and arg_count >= 3
                    else "sample_shadow"
                )
            if has_sampler:
                return (
                    "sample_bias_sampler"
                    if arg_count and arg_count >= 4
                    else "sample_sampler"
                )
            return "sample_bias" if arg_count and arg_count >= 3 else "sample"
        if func_name == "textureOffset":
            if has_sampler:
                return (
                    "sample_offset_bias_sampler"
                    if arg_count and arg_count >= 5
                    else "sample_offset_sampler"
                )
            return (
                "sample_offset_bias"
                if arg_count and arg_count >= 4
                else "sample_offset"
            )
        if func_name == "textureProj":
            if has_sampler:
                return (
                    "sample_projected_bias_sampler"
                    if arg_count and arg_count >= 4
                    else "sample_projected_sampler"
                )
            return (
                "sample_projected_bias"
                if arg_count and arg_count >= 3
                else "sample_projected"
            )
        if func_name == "textureProjOffset":
            if has_sampler:
                return (
                    "sample_projected_offset_bias_sampler"
                    if arg_count and arg_count >= 5
                    else "sample_projected_offset_sampler"
                )
            return (
                "sample_projected_offset_bias"
                if arg_count and arg_count >= 4
                else "sample_projected_offset"
            )
        sampler_texture_map = {
            "textureLod": "sample_lod_sampler",
            "textureLodOffset": "sample_lod_offset_sampler",
            "textureGrad": "sample_grad_sampler",
            "textureGradOffset": "sample_grad_offset_sampler",
            "textureProjLod": "sample_projected_lod_sampler",
            "textureProjLodOffset": "sample_projected_lod_offset_sampler",
            "textureProjGrad": "sample_projected_grad_sampler",
            "textureProjGradOffset": "sample_projected_grad_offset_sampler",
            "textureQueryLod": "texture_query_lod_sampler",
        }
        if has_sampler and func_name in sampler_texture_map:
            return sampler_texture_map[func_name]
        if func_name == "imageLoad" and arg_count == 3:
            return "image_load_sample"
        if func_name == "imageStore" and arg_count == 4:
            return "image_store_sample"
        sample_indexed_atomics = {
            "imageAtomicAdd": "image_atomic_add_sample",
            "imageAtomicMin": "image_atomic_min_sample",
            "imageAtomicMax": "image_atomic_max_sample",
            "imageAtomicAnd": "image_atomic_and_sample",
            "imageAtomicOr": "image_atomic_or_sample",
            "imageAtomicXor": "image_atomic_xor_sample",
            "imageAtomicExchange": "image_atomic_exchange_sample",
        }
        if func_name in sample_indexed_atomics and arg_count == 4:
            return sample_indexed_atomics[func_name]
        if func_name in {"imageAtomicCompSwap", "imageAtomicCompareExchange"} and (
            arg_count == 5
        ):
            return "image_atomic_comp_swap_sample"
        if func_name == "textureGather":
            if has_sampler:
                return (
                    "texture_gather_component_sampler"
                    if arg_count and arg_count >= 4
                    else "texture_gather_sampler"
                )
            return (
                "texture_gather_component"
                if arg_count and arg_count >= 3
                else "texture_gather"
            )
        if func_name == "textureGatherOffset":
            if has_sampler:
                return (
                    "texture_gather_offset_component_sampler"
                    if arg_count and arg_count >= 5
                    else "texture_gather_offset_sampler"
                )
            return (
                "texture_gather_offset_component"
                if arg_count and arg_count >= 4
                else "texture_gather_offset"
            )
        if func_name == "textureGatherOffsets":
            if has_sampler:
                return (
                    "texture_gather_offsets_component_sampler"
                    if arg_count in {5, 8}
                    else "texture_gather_offsets_sampler"
                )
            return (
                "texture_gather_offsets_component"
                if arg_count in {4, 7}
                else "texture_gather_offsets"
            )
        compare_sampler_thresholds = {
            "textureCompare": (4, "texture_compare_sampler"),
            "textureCompareOffset": (5, "texture_compare_offset_sampler"),
            "textureCompareLod": (5, "texture_compare_lod_sampler"),
            "textureCompareLodOffset": (6, "texture_compare_lod_offset_sampler"),
            "textureCompareGrad": (6, "texture_compare_grad_sampler"),
            "textureCompareGradOffset": (7, "texture_compare_grad_offset_sampler"),
            "textureCompareProj": (4, "texture_compare_projected_sampler"),
            "textureCompareProjOffset": (5, "texture_compare_projected_offset_sampler"),
            "textureCompareProjLod": (5, "texture_compare_projected_lod_sampler"),
            "textureCompareProjLodOffset": (
                6,
                "texture_compare_projected_lod_offset_sampler",
            ),
            "textureCompareProjGrad": (6, "texture_compare_projected_grad_sampler"),
            "textureCompareProjGradOffset": (
                7,
                "texture_compare_projected_grad_offset_sampler",
            ),
            "textureGatherCompare": (4, "texture_gather_compare_sampler"),
            "textureGatherCompareOffset": (5, "texture_gather_compare_offset_sampler"),
        }
        sampler_mapping = compare_sampler_thresholds.get(func_name)
        if sampler_mapping is not None:
            threshold, sampler_name = sampler_mapping
            if has_sampler or (arg_count and arg_count >= threshold):
                return sampler_name
        mapped_name = self.function_map.get(func_name)
        if mapped_name is not None:
            return mapped_name
        # Not renamed: a constructor, a declared function or a runtime function spelled as in CrossGL. The call
        # branch refuses a built-in of another language that reaches the output this way.
        return func_name

    def call_has_explicit_sampler_argument(self, arguments):
        if not arguments or len(arguments) < 2:
            return False
        return self.is_sampler_type(self.expression_result_type(arguments[1]))

    def call_uses_shadow_texture(self, arguments):
        if not arguments:
            return False
        return self.is_shadow_texture_type(self.expression_result_type(arguments[0]))

    def is_sampler_type(self, type_name):
        if type_name is None:
            return False
        type_name = str(type_name)
        return (
            type_name in {"sampler", "Sampler"} or self.map_type(type_name) == "Sampler"
        )

    def is_texture_resource_type(self, type_name):
        if type_name is None:
            return False
        mapped_type = self.map_type(type_name)
        base_type, _generic_args = self.generic_type_parts(mapped_type)
        base_name = base_type.rsplit("::", 1)[-1]
        return base_name.startswith(("Texture", "DepthTexture"))

    def is_shadow_texture_type(self, type_name):
        if type_name is None:
            return False
        source_type = str(type_name)
        mapped_type = self.map_type(source_type)
        names = {source_type, mapped_type}
        return any("Shadow" in name or "DepthTexture" in name for name in names)

    def is_multisample_texture_type(self, type_name):
        if type_name is None:
            return False
        source_type = str(type_name)
        mapped_type = self.map_type(source_type)
        names = {source_type, mapped_type}
        return any("MS" in name for name in names)

    def generate_qualified_generic_trait_method_call(self, func_expr, args):
        if not isinstance(func_expr, MemberAccessNode):
            return None

        method_name = getattr(func_expr, "member", "")
        object_expr = getattr(
            func_expr, "object_expr", getattr(func_expr, "object", None)
        )
        receiver_type = self.expression_result_type(object_expr)
        if receiver_type not in self.current_generic_param_names:
            return None

        trait_name = self.generic_trait_method_owner(receiver_type, method_name)
        if trait_name is None:
            return None

        receiver = self.generate_expression(object_expr)
        args_str = ", ".join(self.generate_expression(arg) for arg in args)
        if args_str:
            return f"{trait_name}::{method_name}({receiver}, {args_str})"
        return f"{trait_name}::{method_name}({receiver})"

    def generic_trait_method_owner(self, generic_name, method_name):
        for constraint in self.current_function_generic_constraints.get(
            generic_name, []
        ):
            trait_name = str(constraint).split("<", 1)[0]
            if method_name in self.trait_methods.get(trait_name, set()):
                return trait_name
        return None
