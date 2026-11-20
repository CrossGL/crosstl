"""Texture offsets, projection, gather and comparison in the Slang generator, with their rank and type checks."""

from ..ast import LiteralNode
from .array_utils import (
    evaluate_literal_int_expression,
    split_array_type_suffix,
)
from .c_family_operands import postfix_operand


class SlangTextureSamplingMixin:
    """Texture-sampling members of ``SlangCodeGen``."""

    def generate_texture_offset(self, func_name, args):
        sample_args = self.sampled_texture_args(args)
        if sample_args is None:
            return self.unsupported_texture_offset_call(
                func_name, "requires texture and coordinate arguments"
            )

        if not self.sampled_texture_sampling_accepts_resource(args[0]):
            return self.unsupported_texture_offset_call(
                func_name,
                "requires a non-shadow non-multisampled sampled texture resource",
            )

        texture_name, coord, extra_args = sample_args
        coord_node = args[self.sampled_texture_coord_index(args)]
        coord_reason = self.sampled_texture_coordinate_rank_unsupported_reason(
            args[0], coord_node
        )
        if coord_reason:
            return self.unsupported_texture_offset_call(func_name, coord_reason)

        if func_name == "textureOffset":
            if len(extra_args) not in {1, 2}:
                return self.unsupported_texture_offset_call(
                    func_name, "requires offset and optional bias arguments"
                )
            offset_reason = self.texture_offset_rank_unsupported_reason(
                args[0], extra_args[0]
            )
            if offset_reason:
                return self.unsupported_texture_offset_call(func_name, offset_reason)
            if len(extra_args) == 2:
                bias_reason = self.scalar_texture_bias_unsupported_reason(extra_args[1])
                if bias_reason:
                    return self.unsupported_texture_offset_call(func_name, bias_reason)
            expected_reason = self.texture_result_expected_type_unsupported_reason(
                func_name, "float4"
            )
            if expected_reason:
                return self.unsupported_texture_offset_call(func_name, expected_reason)
            offset = self.generate_expression(extra_args[0])
            if len(extra_args) == 2:
                bias = self.generate_expression(extra_args[1])
                return self.slang_texture_method_call(
                    texture_name, "SampleBias", args, coord, bias, offset
                )
            return self.slang_texture_method_call(
                texture_name, "Sample", args, coord, offset
            )

        if func_name == "textureLodOffset":
            if len(extra_args) != 2:
                return self.unsupported_texture_offset_call(
                    func_name, "requires lod and offset arguments"
                )
            lod_reason = self.scalar_texture_lod_unsupported_reason(extra_args[0])
            if lod_reason:
                return self.unsupported_texture_offset_call(func_name, lod_reason)
            offset_reason = self.texture_offset_rank_unsupported_reason(
                args[0], extra_args[1]
            )
            if offset_reason:
                return self.unsupported_texture_offset_call(func_name, offset_reason)
            expected_reason = self.texture_result_expected_type_unsupported_reason(
                func_name, "float4"
            )
            if expected_reason:
                return self.unsupported_texture_offset_call(func_name, expected_reason)
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return self.slang_texture_method_call(
                texture_name, "SampleLevel", args, coord, lod, offset
            )

        if len(extra_args) != 3:
            return self.unsupported_texture_offset_call(
                func_name, "requires gradient x, gradient y, and offset arguments"
            )
        grad_reason = self.texture_gradient_rank_unsupported_reason(
            args[0], extra_args[0]
        ) or self.texture_gradient_rank_unsupported_reason(args[0], extra_args[1])
        if grad_reason:
            return self.unsupported_texture_offset_call(func_name, grad_reason)
        offset_reason = self.texture_offset_rank_unsupported_reason(
            args[0], extra_args[2]
        )
        if offset_reason:
            return self.unsupported_texture_offset_call(func_name, offset_reason)
        expected_reason = self.texture_result_expected_type_unsupported_reason(
            func_name, "float4"
        )
        if expected_reason:
            return self.unsupported_texture_offset_call(func_name, expected_reason)
        ddx = self.generate_expression(extra_args[0])
        ddy = self.generate_expression(extra_args[1])
        offset = self.generate_expression(extra_args[2])
        return self.slang_texture_method_call(
            texture_name, "SampleGrad", args, coord, ddx, ddy, offset
        )

    def unsupported_texture_offset_call(self, func_name, reason):
        fallback = self.texture_result_diagnostic_fallback("float4")
        return self.refuse_lowering(
            f"Slang texture offset: {func_name} {reason}", fallback
        )

    def texture_result_expected_type_unsupported_reason(self, func_name, result_type):
        expected_type = self.convert_type(self.current_expression_expected_type)
        if not expected_type or expected_type == "auto":
            return None
        if not (
            self.is_scalar_value_type(expected_type)
            or self.is_vector_value_type(expected_type)
        ):
            return None
        result_type = self.convert_type(result_type)
        if expected_type == result_type:
            return None
        return f"returns {result_type} but target expects {expected_type}"

    def texture_result_diagnostic_fallback(self, default_type):
        expected_type = self.convert_type(self.current_expression_expected_type)
        if (
            expected_type
            and expected_type != "auto"
            and (
                self.is_scalar_value_type(expected_type)
                or self.is_vector_value_type(expected_type)
            )
        ):
            return self.zero_value_for_type(expected_type)
        return self.zero_value_for_type(default_type)

    def generate_texture_projected(self, func_name, args):
        sample_args = self.sampled_texture_args(args)
        if sample_args is None:
            return self.unsupported_texture_projected_call(
                func_name, "requires texture and projected coordinate arguments"
            )

        if not self.projected_texture_accepts_resource(args[0]):
            return self.unsupported_texture_projected_call(
                func_name, "requires sampler1D/2D/3D texture resource"
            )

        texture_name, coord, extra_args = sample_args
        coord_node = args[self.sampled_texture_coord_index(args)]
        projected_coord = self.projected_texture_coord(args[0], coord_node, coord)
        if projected_coord is None:
            return self.unsupported_texture_projected_call(
                func_name, "requires sampler1D/2D/3D projection coordinates"
            )

        if func_name == "textureProj":
            if not extra_args:
                expected_reason = self.texture_result_expected_type_unsupported_reason(
                    func_name, "float4"
                )
                if expected_reason:
                    return self.unsupported_texture_projected_call(
                        func_name, expected_reason
                    )
                return self.slang_texture_method_call(
                    texture_name, "Sample", args, projected_coord
                )
            if len(extra_args) == 1:
                bias_reason = self.scalar_texture_bias_unsupported_reason(extra_args[0])
                if bias_reason:
                    return self.unsupported_texture_projected_call(
                        func_name, bias_reason
                    )
                expected_reason = self.texture_result_expected_type_unsupported_reason(
                    func_name, "float4"
                )
                if expected_reason:
                    return self.unsupported_texture_projected_call(
                        func_name, expected_reason
                    )
                bias = self.generate_expression(extra_args[0])
                return self.slang_texture_method_call(
                    texture_name, "SampleBias", args, projected_coord, bias
                )
            return self.unsupported_texture_projected_call(
                func_name, "accepts at most one bias argument"
            )

        if func_name == "textureProjOffset":
            if len(extra_args) == 1:
                offset_reason = self.texture_offset_rank_unsupported_reason(
                    args[0], extra_args[0]
                )
                if offset_reason:
                    return self.unsupported_texture_projected_call(
                        func_name, offset_reason
                    )
                expected_reason = self.texture_result_expected_type_unsupported_reason(
                    func_name, "float4"
                )
                if expected_reason:
                    return self.unsupported_texture_projected_call(
                        func_name, expected_reason
                    )
                offset = self.generate_expression(extra_args[0])
                return self.slang_texture_method_call(
                    texture_name, "Sample", args, projected_coord, offset
                )
            if len(extra_args) == 2:
                offset_reason = self.texture_offset_rank_unsupported_reason(
                    args[0], extra_args[0]
                )
                if offset_reason:
                    return self.unsupported_texture_projected_call(
                        func_name, offset_reason
                    )
                bias_reason = self.scalar_texture_bias_unsupported_reason(extra_args[1])
                if bias_reason:
                    return self.unsupported_texture_projected_call(
                        func_name, bias_reason
                    )
                expected_reason = self.texture_result_expected_type_unsupported_reason(
                    func_name, "float4"
                )
                if expected_reason:
                    return self.unsupported_texture_projected_call(
                        func_name, expected_reason
                    )
                offset = self.generate_expression(extra_args[0])
                bias = self.generate_expression(extra_args[1])
                return self.slang_texture_method_call(
                    texture_name, "SampleBias", args, projected_coord, bias, offset
                )
            return self.unsupported_texture_projected_call(
                func_name, "requires offset and optional bias arguments"
            )

        if func_name == "textureProjLod":
            if len(extra_args) != 1:
                return self.unsupported_texture_projected_call(
                    func_name, "requires one lod argument"
                )
            lod_reason = self.scalar_texture_lod_unsupported_reason(extra_args[0])
            if lod_reason:
                return self.unsupported_texture_projected_call(func_name, lod_reason)
            expected_reason = self.texture_result_expected_type_unsupported_reason(
                func_name, "float4"
            )
            if expected_reason:
                return self.unsupported_texture_projected_call(
                    func_name, expected_reason
                )
            lod = self.generate_expression(extra_args[0])
            return self.slang_texture_method_call(
                texture_name, "SampleLevel", args, projected_coord, lod
            )

        if func_name == "textureProjLodOffset":
            if len(extra_args) != 2:
                return self.unsupported_texture_projected_call(
                    func_name, "requires lod and offset arguments"
                )
            lod_reason = self.scalar_texture_lod_unsupported_reason(extra_args[0])
            if lod_reason:
                return self.unsupported_texture_projected_call(func_name, lod_reason)
            offset_reason = self.texture_offset_rank_unsupported_reason(
                args[0], extra_args[1]
            )
            if offset_reason:
                return self.unsupported_texture_projected_call(func_name, offset_reason)
            expected_reason = self.texture_result_expected_type_unsupported_reason(
                func_name, "float4"
            )
            if expected_reason:
                return self.unsupported_texture_projected_call(
                    func_name, expected_reason
                )
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return self.slang_texture_method_call(
                texture_name, "SampleLevel", args, projected_coord, lod, offset
            )

        if func_name == "textureProjGrad":
            if len(extra_args) != 2:
                return self.unsupported_texture_projected_call(
                    func_name, "requires gradient x and gradient y arguments"
                )
            grad_reason = self.texture_gradient_rank_unsupported_reason(
                args[0], extra_args[0]
            ) or self.texture_gradient_rank_unsupported_reason(args[0], extra_args[1])
            if grad_reason:
                return self.unsupported_texture_projected_call(func_name, grad_reason)
            expected_reason = self.texture_result_expected_type_unsupported_reason(
                func_name, "float4"
            )
            if expected_reason:
                return self.unsupported_texture_projected_call(
                    func_name, expected_reason
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            return self.slang_texture_method_call(
                texture_name, "SampleGrad", args, projected_coord, ddx, ddy
            )

        if len(extra_args) != 3:
            return self.unsupported_texture_projected_call(
                func_name, "requires gradient x, gradient y, and offset arguments"
            )
        grad_reason = self.texture_gradient_rank_unsupported_reason(
            args[0], extra_args[0]
        ) or self.texture_gradient_rank_unsupported_reason(args[0], extra_args[1])
        if grad_reason:
            return self.unsupported_texture_projected_call(func_name, grad_reason)
        offset_reason = self.texture_offset_rank_unsupported_reason(
            args[0], extra_args[2]
        )
        if offset_reason:
            return self.unsupported_texture_projected_call(func_name, offset_reason)
        expected_reason = self.texture_result_expected_type_unsupported_reason(
            func_name, "float4"
        )
        if expected_reason:
            return self.unsupported_texture_projected_call(func_name, expected_reason)
        ddx = self.generate_expression(extra_args[0])
        ddy = self.generate_expression(extra_args[1])
        offset = self.generate_expression(extra_args[2])
        return self.slang_texture_method_call(
            texture_name, "SampleGrad", args, projected_coord, ddx, ddy, offset
        )

    def projected_texture_accepts_resource(self, texture_node):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        if resource_type is None:
            return True
        return resource_type in {"sampler1D", "sampler2D", "sampler3D"}

    def projected_texture_coord(self, texture_node, coord_node, coord):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        coord_type = self.resource_base_type(
            self.get_expression_type(coord_node)
            or self.type_name_string(self.expression_result_type(coord_node))
        )
        specs = {
            "sampler1D": {
                "vec2": ("x", "y"),
                "float2": ("x", "y"),
                "vec4": ("x", "w"),
                "float4": ("x", "w"),
            },
            "sampler2D": {
                "vec3": ("xy", "z"),
                "float3": ("xy", "z"),
                "vec4": ("xy", "w"),
                "float4": ("xy", "w"),
            },
            "sampler3D": {
                "vec4": ("xyz", "w"),
                "float4": ("xyz", "w"),
            },
        }
        resource_specs = specs.get(resource_type)
        if resource_specs is None:
            return None
        coord_spec = resource_specs.get(coord_type)
        if coord_spec is None:
            return None
        numerator, divisor = coord_spec
        coord = postfix_operand(coord)
        return f"{coord}.{numerator} / {coord}.{divisor}"

    def unsupported_texture_projected_call(self, func_name, reason):
        fallback = self.texture_result_diagnostic_fallback("float4")
        return self.refuse_lowering(
            f"Slang projected texture: {func_name} {reason}", fallback
        )

    def generate_texture_gather(self, func_name, args):
        gather_args = self.sampled_texture_args(args)
        if gather_args is None:
            return self.unsupported_texture_gather_call(
                func_name, "requires texture and coordinate arguments"
            )

        if not self.sampled_texture_sampling_accepts_resource(args[0]):
            return self.unsupported_texture_gather_call(
                func_name,
                "requires a non-shadow non-multisampled sampled texture resource",
            )

        texture_name, coord, extra_args = gather_args
        coord_node = args[self.sampled_texture_coord_index(args)]
        coord_reason = self.sampled_texture_coordinate_rank_unsupported_reason(
            args[0], coord_node
        )
        if coord_reason:
            return self.unsupported_texture_gather_call(func_name, coord_reason)
        offset_args = []
        component_arg = None

        if func_name == "textureGather":
            if len(extra_args) > 1:
                return self.unsupported_texture_gather_call(
                    func_name, "accepts at most one component argument"
                )
            if extra_args:
                component_arg = extra_args[0]
        elif func_name == "textureGatherOffset":
            if len(extra_args) not in {1, 2}:
                return self.unsupported_texture_gather_call(
                    func_name, "requires offset and optional component arguments"
                )
            offset_reason = self.gather_offset_rank_unsupported_reason(
                args[0], extra_args[0]
            )
            if offset_reason:
                return self.unsupported_texture_gather_call(func_name, offset_reason)
            offset_args = [extra_args[0]]
            if len(extra_args) == 2:
                component_arg = extra_args[1]
        else:
            offsets_reason = self.gather_offsets_rank_unsupported_reason(
                args[0], extra_args
            )
            if offsets_reason:
                return self.unsupported_texture_gather_call(func_name, offsets_reason)
            offset_args, component_arg = self.texture_gather_offsets_args(extra_args)
            if offset_args is None:
                return self.unsupported_texture_gather_call(
                    func_name,
                    "requires a typed offsets array or four offset arguments",
                )

        component_reason = self.texture_gather_component_unsupported_reason(
            component_arg
        )
        if component_reason:
            return self.unsupported_texture_gather_call(func_name, component_reason)

        method_args = self.slang_texture_method_args(
            args,
            coord,
            *[self.generate_expression(offset_arg) for offset_arg in offset_args],
        )
        method = self.texture_gather_method(component_arg)
        if method is not None:
            expected_reason = self.texture_result_expected_type_unsupported_reason(
                func_name, "float4"
            )
            if expected_reason:
                return self.unsupported_texture_gather_call(func_name, expected_reason)
            return f"{texture_name}.{method}({', '.join(method_args)})"
        if isinstance(component_arg, LiteralNode):
            return self.unsupported_texture_gather_call(
                func_name, "component literal must be 0, 1, 2, or 3"
            )

        expected_reason = self.texture_result_expected_type_unsupported_reason(
            func_name, "float4"
        )
        if expected_reason:
            return self.unsupported_texture_gather_call(func_name, expected_reason)

        component = self.generate_expression(component_arg)
        return self.texture_gather_component_expression(
            texture_name, method_args, component
        )

    def texture_gather_offsets_args(self, extra_args):
        if len(extra_args) in {1, 2} and self.is_array_expression(extra_args[0]):
            offsets_name = self.generate_expression(extra_args[0])
            offset_args = [f"{offsets_name}[{index}]" for index in range(4)]
            component_arg = extra_args[1] if len(extra_args) == 2 else None
            return offset_args, component_arg

        if len(extra_args) in {4, 5}:
            component_arg = extra_args[4] if len(extra_args) == 5 else None
            return extra_args[:4], component_arg

        return None, None

    def texture_gather_method(self, component_arg):
        if component_arg is None:
            return "Gather"

        methods = {
            0: "GatherRed",
            1: "GatherGreen",
            2: "GatherBlue",
            3: "GatherAlpha",
        }
        return methods.get(self.literal_int_value(component_arg))

    def texture_gather_component_unsupported_reason(self, component_arg):
        if component_arg is None:
            return None

        component_type = self.expression_result_type(component_arg)
        if component_type is None:
            return None

        mapped_type = self.convert_type(component_type)
        if mapped_type in {"int", "uint"}:
            return None

        return f"component argument must be scalar int or uint, got {mapped_type}"

    def texture_gather_component_expression(self, texture_name, method_args, component):
        arg_list = ", ".join(method_args)
        component_calls = [
            f"{texture_name}.{method}({arg_list})"
            for method in (
                "GatherRed",
                "GatherGreen",
                "GatherBlue",
                "GatherAlpha",
            )
        ]
        return (
            f"({component} == 0 ? {component_calls[0]} : "
            f"{component} == 1 ? {component_calls[1]} : "
            f"{component} == 2 ? {component_calls[2]} : "
            f"{component_calls[3]})"
        )

    def unsupported_texture_gather_call(self, func_name, reason):
        fallback = self.texture_result_diagnostic_fallback("float4")
        return self.refuse_lowering(
            f"Slang texture gather: {func_name} {reason}", fallback
        )

    def generate_texture_compare(self, func_name, args):
        compare_args = self.texture_compare_args(func_name, args)
        if compare_args is None:
            return self.unsupported_texture_compare_call(
                func_name, "requires texture, coordinate, and compare arguments"
            )

        texture_name, coord, compare, extra_args = compare_args
        if not self.is_shadow_compare_resource(args[0]):
            return self.unsupported_texture_compare_call(
                func_name, "requires a shadow sampler resource"
            )
        coord_index = self.sampled_texture_coord_index(args)
        coord_reason = self.shadow_compare_coordinate_rank_unsupported_reason(
            args[0], args[coord_index]
        )
        if coord_reason:
            return self.unsupported_texture_compare_call(func_name, coord_reason)
        compare_reason = self.compare_reference_unsupported_reason(
            args[coord_index + 1]
        )
        if compare_reason:
            return self.unsupported_texture_compare_call(func_name, compare_reason)
        expected_reason = self.resource_query_expected_type_unsupported_reason(
            func_name, "float"
        )
        if expected_reason:
            return self.unsupported_texture_compare_call(
                func_name,
                expected_reason,
                self.zero_value_for_type(self.current_expression_expected_type),
            )

        if func_name == "textureCompare":
            if extra_args:
                return self.unsupported_texture_compare_call(
                    func_name, "accepts no extra arguments"
                )
            return self.slang_texture_method_call(
                texture_name, "SampleCmp", args, coord, compare
            )

        if func_name == "textureCompareOffset":
            if len(extra_args) != 1:
                return self.unsupported_texture_compare_call(
                    func_name, "requires one offset argument"
                )
            offset_reason = self.shadow_compare_offset_rank_unsupported_reason(
                args[0], extra_args[0]
            )
            if offset_reason:
                return self.unsupported_texture_compare_call(func_name, offset_reason)
            offset = self.generate_expression(extra_args[0])
            return self.slang_texture_method_call(
                texture_name, "SampleCmp", args, coord, compare, offset
            )

        if func_name == "textureCompareLod":
            if len(extra_args) != 1:
                return self.unsupported_texture_compare_call(
                    func_name, "requires one lod argument"
                )
            lod_reason = self.scalar_numeric_texture_argument_unsupported_reason(
                extra_args[0], "lod argument"
            )
            if lod_reason:
                return self.unsupported_texture_compare_call(func_name, lod_reason)
            lod = self.generate_expression(extra_args[0])
            return self.slang_texture_method_call(
                texture_name, "SampleCmpLevel", args, coord, compare, lod
            )

        if func_name == "textureCompareLodOffset":
            if len(extra_args) != 2:
                return self.unsupported_texture_compare_call(
                    func_name, "requires lod and offset arguments"
                )
            lod_reason = self.scalar_numeric_texture_argument_unsupported_reason(
                extra_args[0], "lod argument"
            )
            if lod_reason:
                return self.unsupported_texture_compare_call(func_name, lod_reason)
            offset_reason = self.shadow_compare_offset_rank_unsupported_reason(
                args[0], extra_args[1]
            )
            if offset_reason:
                return self.unsupported_texture_compare_call(func_name, offset_reason)
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return self.slang_texture_method_call(
                texture_name, "SampleCmpLevel", args, coord, compare, lod, offset
            )

        if len(extra_args) != 2:
            return self.unsupported_texture_compare_call(
                func_name, "requires gradient x and gradient y arguments"
            )
        grad_reason = self.shadow_compare_gradient_rank_unsupported_reason(
            args[0], extra_args[0]
        ) or self.shadow_compare_gradient_rank_unsupported_reason(
            args[0], extra_args[1]
        )
        if grad_reason:
            return self.unsupported_texture_compare_call(func_name, grad_reason)
        ddx = self.generate_expression(extra_args[0])
        ddy = self.generate_expression(extra_args[1])
        return self.slang_texture_method_call(
            texture_name, "SampleCmpGrad", args, coord, compare, ddx, ddy
        )

    def generate_texture_gather_compare(self, func_name, args):
        compare_args = self.texture_compare_args(func_name, args)
        if compare_args is None:
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires texture, coordinate, and compare arguments"
            )

        texture_name, coord, compare, extra_args = compare_args
        if not self.is_shadow_compare_resource(args[0]):
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires a shadow sampler resource"
            )
        coord_index = self.sampled_texture_coord_index(args)
        coord_reason = self.shadow_compare_coordinate_rank_unsupported_reason(
            args[0], args[coord_index]
        )
        if coord_reason:
            return self.unsupported_texture_gather_compare_call(func_name, coord_reason)
        compare_reason = self.compare_reference_unsupported_reason(
            args[coord_index + 1]
        )
        if compare_reason:
            return self.unsupported_texture_gather_compare_call(
                func_name, compare_reason
            )
        expected_reason = self.resource_query_expected_type_unsupported_reason(
            func_name, "float4"
        )
        if expected_reason:
            return self.unsupported_texture_gather_compare_call(
                func_name,
                expected_reason,
                self.zero_value_for_type(self.current_expression_expected_type),
            )

        if func_name == "textureGatherCompare":
            if extra_args:
                return self.unsupported_texture_gather_compare_call(
                    func_name, "accepts no extra arguments"
                )
            return self.slang_texture_method_call(
                texture_name, "GatherCmp", args, coord, compare
            )

        if len(extra_args) != 1:
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires one offset argument"
            )
        offset_reason = self.shadow_compare_offset_rank_unsupported_reason(
            args[0], extra_args[0]
        )
        if offset_reason:
            return self.unsupported_texture_gather_compare_call(
                func_name, offset_reason
            )
        offset = self.generate_expression(extra_args[0])
        return self.slang_texture_method_call(
            texture_name, "GatherCmp", args, coord, compare, offset
        )

    def texture_compare_args(self, func_name, args):
        coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
        if len(args) <= coord_index + 1:
            return None

        texture_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[coord_index])
        compare = self.generate_expression(args[coord_index + 1])
        return texture_name, coord, compare, args[coord_index + 2 :]

    def sampled_texture_coordinate_rank_unsupported_reason(self, texture_node, coord):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        expected_rank = self.sampled_texture_coordinate_rank(resource_type)
        return self.texture_rank_unsupported_reason(
            coord, expected_rank, resource_type, "coordinate"
        )

    def texel_fetch_coordinate_rank_unsupported_reason(self, texture_node, coord):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        expected_rank = self.texel_fetch_coordinate_rank(resource_type)
        return self.texture_rank_unsupported_reason(
            coord, expected_rank, resource_type, "coordinate"
        )

    def texel_fetch_offset_rank_unsupported_reason(self, texture_node, offset):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        if resource_type is None:
            return None
        expected_rank = self.texture_offset_rank(resource_type)
        if expected_rank is None:
            return (
                "requires an offset-capable sampler1D/1DArray/2D/2DArray/3D "
                "texture resource"
            )
        rank_reason = self.texture_rank_unsupported_reason(
            offset, expected_rank, resource_type, "offset"
        )
        if rank_reason:
            return rank_reason
        return self.texture_offset_type_unsupported_reason(offset)

    def texture_offset_rank_unsupported_reason(self, texture_node, offset):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        if resource_type is None:
            return None
        expected_rank = self.texture_offset_rank(resource_type)
        if expected_rank is None:
            return "requires an offset-capable sampler1D/2D/3D texture resource"
        rank_reason = self.texture_rank_unsupported_reason(
            offset, expected_rank, resource_type, "offset"
        )
        if rank_reason:
            return rank_reason
        return self.texture_offset_type_unsupported_reason(offset)

    def texture_gradient_rank_unsupported_reason(self, texture_node, gradient):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        expected_rank = self.texture_gradient_rank(resource_type)
        rank_reason = self.texture_rank_unsupported_reason(
            gradient, expected_rank, resource_type, "gradient"
        )
        if rank_reason:
            return rank_reason
        return self.texture_gradient_type_unsupported_reason(gradient)

    def gather_offset_rank_unsupported_reason(self, texture_node, offset):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        if resource_type is None:
            return None
        expected_rank = self.gather_offset_rank(resource_type)
        if expected_rank is None:
            return "requires a gather-offset-capable sampler2D/2DArray texture resource"
        rank_reason = self.texture_rank_unsupported_reason(
            offset, expected_rank, resource_type, "offset"
        )
        if rank_reason:
            return rank_reason
        return self.texture_offset_type_unsupported_reason(offset)

    def gather_offsets_rank_unsupported_reason(self, texture_node, extra_args):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        if resource_type is None:
            return None
        expected_rank = self.gather_offset_rank(resource_type)
        if expected_rank is None:
            return "requires a gather-offset-capable sampler2D/2DArray texture resource"

        if len(extra_args) in {1, 2} and self.is_array_expression(extra_args[0]):
            rank_reason = self.texture_rank_unsupported_reason(
                extra_args[0],
                expected_rank,
                resource_type,
                "offset",
                array_element=True,
            )
            if rank_reason:
                return rank_reason
            return self.texture_offset_type_unsupported_reason(
                extra_args[0], array_element=True
            )
        if len(extra_args) in {4, 5}:
            for offset in extra_args[:4]:
                reason = self.texture_rank_unsupported_reason(
                    offset, expected_rank, resource_type, "offset"
                )
                if reason:
                    return reason
                reason = self.texture_offset_type_unsupported_reason(offset)
                if reason:
                    return reason
        return None

    def shadow_compare_coordinate_rank_unsupported_reason(self, texture_node, coord):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        expected_rank = self.shadow_compare_coordinate_rank(resource_type)
        return self.texture_rank_unsupported_reason(
            coord, expected_rank, resource_type, "coordinate"
        )

    def shadow_compare_offset_rank_unsupported_reason(self, texture_node, offset):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        if resource_type is None:
            return None
        expected_rank = self.shadow_compare_offset_rank(resource_type)
        if expected_rank is None:
            return "requires an offset-capable sampler2DShadow/2DArrayShadow resource"
        rank_reason = self.texture_rank_unsupported_reason(
            offset, expected_rank, resource_type, "offset"
        )
        if rank_reason:
            return rank_reason
        return self.texture_offset_type_unsupported_reason(offset)

    def shadow_compare_gradient_rank_unsupported_reason(self, texture_node, gradient):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        expected_rank = self.shadow_compare_gradient_rank(resource_type)
        rank_reason = self.texture_rank_unsupported_reason(
            gradient, expected_rank, resource_type, "gradient"
        )
        if rank_reason:
            return rank_reason
        return self.texture_gradient_type_unsupported_reason(gradient)

    def compare_reference_unsupported_reason(self, compare):
        type_name = self.type_name_string(self.expression_result_type(compare))
        if type_name is None:
            return None

        mapped_type = self.convert_type(type_name)
        if mapped_type in {"float", "double"}:
            return None

        return f"compare reference must be scalar float or double, got {mapped_type}"

    def texture_offset_type_unsupported_reason(self, offset, array_element=False):
        type_name = self.type_name_string(self.expression_result_type(offset))
        if type_name is None:
            return None
        if array_element and "[" in type_name and "]" in type_name:
            type_name, _suffix = split_array_type_suffix(type_name)

        mapped_type = self.convert_type(type_name)
        if mapped_type == "int":
            return None

        info = self.vector_value_info(type_name)
        if info is not None and info["component_type"] == "int":
            return None

        return f"offset must be scalar or vector int, got {mapped_type}"

    def texture_gradient_type_unsupported_reason(self, gradient):
        type_name = self.type_name_string(self.expression_result_type(gradient))
        if type_name is None:
            return None

        mapped_type = self.convert_type(type_name)
        if mapped_type in {"float", "double"}:
            return None

        info = self.vector_value_info(type_name)
        if info is not None and info["component_type"] in {"float", "double"}:
            return None

        return f"gradient must be scalar or vector float or double, got {mapped_type}"

    def scalar_texture_argument_rank_unsupported_reason(self, node, role):
        actual_rank = self.expression_value_rank(node)
        if actual_rank is None or actual_rank == 1:
            return None
        return f"requires {self.texture_rank_phrase(1, role)}"

    def scalar_texture_lod_unsupported_reason(self, node):
        return self.scalar_texture_argument_rank_unsupported_reason(
            node, "lod argument"
        ) or self.scalar_numeric_texture_argument_unsupported_reason(
            node, "lod argument"
        )

    def scalar_texture_bias_unsupported_reason(self, node):
        return self.scalar_texture_argument_rank_unsupported_reason(
            node, "bias argument"
        ) or self.scalar_numeric_texture_argument_unsupported_reason(
            node, "bias argument"
        )

    def scalar_texture_mip_unsupported_reason(self, node):
        return self.scalar_texture_argument_rank_unsupported_reason(
            node, "mip argument"
        ) or self.scalar_integer_texture_argument_unsupported_reason(
            node, "mip argument"
        )

    def scalar_integer_texture_argument_unsupported_reason(self, node, role):
        type_name = self.type_name_string(self.expression_result_type(node))
        if type_name is None:
            return None

        mapped_type = self.convert_type(type_name)
        if mapped_type in {"int", "uint"}:
            return None

        return f"{role} must be scalar int or uint, got {mapped_type}"

    def scalar_numeric_texture_argument_unsupported_reason(self, node, role):
        type_name = self.type_name_string(self.expression_result_type(node))
        if type_name is None:
            return None

        mapped_type = self.convert_type(type_name)
        if mapped_type in {"int", "uint", "float", "double"}:
            return None

        return f"{role} must be scalar int, uint, float, or double, got {mapped_type}"

    def texture_rank_unsupported_reason(
        self, node, expected_rank, resource_type, role, array_element=False
    ):
        if expected_rank is None or resource_type is None:
            return None
        actual_rank = self.expression_value_rank(node, array_element=array_element)
        if actual_rank is None or actual_rank == expected_rank:
            return None
        return (
            f"requires {self.texture_rank_phrase(expected_rank, role)} "
            f"for {resource_type}"
        )

    def texture_rank_phrase(self, rank, role):
        if rank == 1:
            return f"a scalar {role}"
        return f"a {rank}-component {role}"

    def expression_value_rank(self, node, array_element=False):
        type_name = self.type_name_string(self.expression_result_type(node))
        if not type_name:
            return None
        if array_element and "[" in type_name and "]" in type_name:
            type_name, _suffix = split_array_type_suffix(type_name)
        if self.is_scalar_value_type(type_name):
            return 1
        info = self.vector_value_info(type_name)
        if info is None:
            return None
        return info["size"]

    def sampled_texture_coordinate_rank(self, resource_type):
        resource_type = self.sampled_texture_shape_type(resource_type)
        return {
            "sampler1D": 1,
            "sampler1DArray": 2,
            "sampler2D": 2,
            "sampler2DArray": 3,
            "sampler3D": 3,
            "samplerCube": 3,
            "samplerCubeArray": 4,
        }.get(resource_type)

    def texel_fetch_coordinate_rank(self, resource_type):
        resource_type = self.sampled_texture_shape_type(resource_type)
        return {
            "sampler1D": 1,
            "sampler1DArray": 2,
            "sampler2D": 2,
            "sampler2DArray": 3,
            "sampler3D": 3,
            "sampler2DMS": 2,
            "sampler2DMSArray": 3,
        }.get(resource_type)

    def texture_offset_rank(self, resource_type):
        resource_type = self.sampled_texture_shape_type(resource_type)
        return {
            "sampler1D": 1,
            "sampler1DArray": 1,
            "sampler2D": 2,
            "sampler2DArray": 2,
            "sampler3D": 3,
        }.get(resource_type)

    def texture_gradient_rank(self, resource_type):
        resource_type = self.sampled_texture_shape_type(resource_type)
        return {
            "sampler1D": 1,
            "sampler1DArray": 1,
            "sampler2D": 2,
            "sampler2DArray": 2,
            "sampler3D": 3,
            "samplerCube": 3,
            "samplerCubeArray": 3,
        }.get(resource_type)

    def gather_offset_rank(self, resource_type):
        resource_type = self.sampled_texture_shape_type(resource_type)
        return {
            "sampler2D": 2,
            "sampler2DArray": 2,
        }.get(resource_type)

    def shadow_compare_coordinate_rank(self, resource_type):
        return {
            "sampler2DShadow": 2,
            "sampler2DArrayShadow": 3,
            "samplerCubeShadow": 3,
            "samplerCubeArrayShadow": 4,
        }.get(resource_type)

    def shadow_compare_offset_rank(self, resource_type):
        return {
            "sampler2DShadow": 2,
            "sampler2DArrayShadow": 2,
        }.get(resource_type)

    def shadow_compare_gradient_rank(self, resource_type):
        return {
            "sampler2DShadow": 2,
            "sampler2DArrayShadow": 2,
            "samplerCubeShadow": 3,
            "samplerCubeArrayShadow": 3,
        }.get(resource_type)

    def is_shadow_compare_resource(self, node):
        resource_type = self.resource_base_type(self.get_expression_type(node))
        return resource_type is None or resource_type in {
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
        }

    def unsupported_texture_compare_call(self, func_name, reason, fallback="0.0"):
        return self.refuse_lowering(
            f"Slang shadow compare: {func_name} {reason}", fallback
        )

    def unsupported_texture_gather_compare_call(
        self, func_name, reason, fallback="float4(0.0)"
    ):
        return self.refuse_lowering(
            f"Slang shadow gather compare: {func_name} {reason}", fallback
        )

    def literal_int_value(self, node, constants=None):
        return evaluate_literal_int_expression(
            node, self.literal_int_constants if constants is None else constants
        )

    def is_array_expression(self, node):
        type_name = self.type_name_string(self.expression_result_type(node))
        return isinstance(type_name, str) and "[" in type_name and "]" in type_name
