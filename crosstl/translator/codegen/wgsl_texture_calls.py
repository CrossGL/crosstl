"""Texture and barrier calls in the WGSL generator: sampled texture array dispatch, sampling, comparison, dimension queries, texel fetches, image loads and stores, coordinate splitting and barriers."""

import re

from ..ast import (
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    NamedType,
)
from ..refusals import WGSLRefusal
from ..semantics.implicit_lod import ImplicitLodStageUnsupportedError
from .wgsl_comparison_samplers import WGSLComparisonSamplerUnsupportedError
from .wgsl_integer_textures import (
    WGSLIntegerTextureSampleUnsupportedError,
    is_integer_sampler_type_name,
)
from .wgsl_texture_bias import (
    WGSLTextureBiasUnsupportedError,
    has_texture_sample_bias,
)
from .wgsl_texture_grad_1d import wgsl_texture_1d_grad_call


class WGSLTextureCallMixin:
    """Texture call members of ``WGSLCodeGen``."""

    def generate_sampled_texture_array_dispatch_call(self, node, resolved_function):
        if resolved_function is None:
            return None
        parameters = list(getattr(resolved_function, "parameters", []) or [])
        dynamic_texture_args = []
        for index, (argument, parameter) in enumerate(zip(node.arguments, parameters)):
            if (
                self.sampled_texture_type(getattr(parameter, "param_type", None))
                is None
            ):
                continue
            array_info = self.sampled_texture_array_access(argument)
            if array_info is None:
                continue
            if self.sampled_texture_array_element_binding(argument) is not None:
                continue
            dynamic_texture_args.append((index, argument, parameter, array_info))
        if not dynamic_texture_args:
            return None
        if len(dynamic_texture_args) > 1:
            names = ", ".join(
                arg.array_expr.name for _idx, arg, _param, _info in dynamic_texture_args
            )
            raise WGSLRefusal(
                "WGSL target cannot lower a call with multiple dynamic sampled "
                f"texture resource arrays ({names}); split the call into one "
                "array-dispatched texture parameter"
            )

        texture_index, argument, _parameter, array_info = dynamic_texture_args[0]
        helper_name = self.sampled_texture_array_dispatch_helper_name(
            resolved_function, array_info["source_name"]
        )
        self._sampled_texture_array_dispatch_helpers[helper_name] = {
            "helper_name": helper_name,
            "function": resolved_function,
            "array_info": array_info,
            "texture_parameter_index": texture_index,
        }
        rendered_args = [f"i32({self.generate_expression(argument.index_expr)})"]
        parameter_types = [
            getattr(parameter, "param_type", None) for parameter in parameters
        ]
        for index, argument in enumerate(node.arguments):
            if index == texture_index:
                continue
            rendered_args.append(
                self.generate_expression_for_target(argument, parameter_types[index])
                if index < len(parameter_types)
                else self.generate_expression(argument)
            )
        return f"{helper_name}({', '.join(rendered_args)})"

    def sampled_texture_array_dispatch_helper_name(self, function, array_name):
        function_name = self.function_declaration_identifier_name(function)
        return self.safe_wgsl_identifier(f"{function_name}_{array_name}")

    def generate_sampled_texture_array_dispatch_helpers(self):
        if not self._sampled_texture_array_dispatch_helpers:
            return ""
        helpers = []
        for info in self._sampled_texture_array_dispatch_helpers.values():
            if info.get("kind") == "texture_sample":
                helpers.append(self.generate_sampled_texture_array_sample_helper(info))
            else:
                helpers.append(
                    self.generate_sampled_texture_array_dispatch_helper(info)
                )
        return "\n\n".join(helpers)

    def generate_sampled_texture_array_sample_helper(self, info):
        array_info = info["array_info"]
        helper_name = info["helper_name"]
        coord_type = self.sampled_texture_coordinate_type(array_info["element_type"])
        lines = [
            f"fn {helper_name}({array_info['root_name']}_index: i32, "
            f"coords: {coord_type}) -> vec4<f32> {{",
            f"    switch ({array_info['root_name']}_index) {{",
        ]
        for binding in array_info["bindings"]:
            lines.extend(
                [
                    f"        case {binding['index']}: {{",
                    "            return textureSample("
                    f"{binding['texture_name']}, {binding['sampler_name']}, coords);",
                    "        }",
                ]
            )
        first = array_info["bindings"][0]
        lines.extend(
            [
                "        default: {",
                "            return textureSample("
                f"{first['texture_name']}, {first['sampler_name']}, coords);",
                "        }",
                "    }",
                "}",
            ]
        )
        return "\n".join(lines)

    def sampled_texture_coordinate_type(self, element_type):
        type_name = self.resource_type_name(element_type)
        if type_name in {"sampler3d", "samplercube", "samplercubearray"}:
            return "vec3<f32>"
        return "vec2<f32>"

    def generate_sampled_texture_array_dispatch_helper(self, info):
        function = info["function"]
        array_info = info["array_info"]
        texture_parameter_index = info["texture_parameter_index"]
        helper_name = info["helper_name"]
        parameters = list(getattr(function, "parameters", []) or [])
        helper_parameters = [
            f"{array_info['root_name']}_index: i32",
            *(
                f"{self.identifier_name(parameter.name)}: "
                f"{self.type_name_string(parameter.param_type)}"
                for index, parameter in enumerate(parameters)
                if index != texture_parameter_index
            ),
        ]
        return_type = self.type_name_string(getattr(function, "return_type", None))
        signature = f"fn {helper_name}({', '.join(helper_parameters)})"
        if return_type != "void":
            signature += f" -> {return_type}"
        lines = [signature + " {", f"    switch ({array_info['root_name']}_index) {{"]
        for binding in array_info["bindings"]:
            lines.extend(
                self.sampled_texture_array_dispatch_case(
                    function, array_info, texture_parameter_index, binding
                )
            )
        lines.extend(
            self.sampled_texture_array_dispatch_case(
                function,
                array_info,
                texture_parameter_index,
                array_info["bindings"][0],
                label="default",
            )
        )
        lines.append("    }")
        lines.append("}")
        return "\n".join(lines)

    def sampled_texture_array_dispatch_case(
        self, function, array_info, texture_parameter_index, binding, label=None
    ):
        case_label = label if label is not None else f"case {binding['index']}"
        call_args = []
        for index, parameter in enumerate(getattr(function, "parameters", []) or []):
            if index == texture_parameter_index:
                call_args.append(binding["texture_name"])
                call_args.append(binding["sampler_name"])
            else:
                call_args.append(self.identifier_name(parameter.name))
        call = (
            f"{self.function_declaration_identifier_name(function)}("
            f"{', '.join(call_args)})"
        )
        return [
            f"        {case_label}: {{",
            f"            return {call};",
            "        }",
        ]

    def refuse_integer_texture_sample(self, texture_expr, function_name):
        type_name = self.resource_type_name(self.expression_type(texture_expr))
        if is_integer_sampler_type_name(type_name):
            raise WGSLIntegerTextureSampleUnsupportedError(
                function_name, self.generate_expression(texture_expr)
            )

    def refuse_comparison_sampler(
        self, sampler, function_name, texture, builtin_name, *, bias=False
    ):
        if not self.is_comparison_sampler_type(self.expression_type(sampler)):
            return
        sampler_name = self.generate_expression(sampler)
        texture_name = (
            texture if isinstance(texture, str) else self.generate_expression(texture)
        )
        if bias:
            raise WGSLTextureBiasUnsupportedError(
                function_name,
                texture_name,
                f"textureSampleBias takes a sampler, but {sampler_name} is "
                "a comparison sampler",
            )
        raise WGSLComparisonSamplerUnsupportedError(
            function_name,
            texture_name,
            builtin_name,
            sampler_name,
        )

    def generate_texture_call_args(self, args, *, function_name, implicit, explicit):
        # The coordinate follows the texture, or the texture and sampler. On
        # an array texture it becomes ``coordinate, layer``.
        if len(args) == implicit:
            texture = args[0]
            return [
                self.generate_expression(texture),
                self.texture_sampler_expression(texture),
                *self.texture_lookup_args(texture, args[1:], function_name),
            ]
        if len(args) == explicit:
            texture, sampler = args[:2]
            return [
                self.generate_expression(texture),
                self.generate_expression(sampler),
                *self.texture_lookup_args(texture, args[2:], function_name),
            ]
        raise WGSLRefusal(
            f"WGSL target supports {function_name}() calls with {implicit} or "
            f"{explicit} argument(s); got {len(args)}"
        )

    def texture_lookup_args(self, texture, args, function_name):
        if not args:
            return []
        coords, *rest = args
        return [
            self.sampled_texture_coordinates(texture, coords, function_name),
            *(self.generate_expression(arg) for arg in rest),
        ]

    def generate_texture_builtin_call(
        self, builtin_name, function_name, args, *, implicit, explicit
    ):
        return (
            f"{builtin_name}("
            + ", ".join(
                self.generate_texture_call_args(
                    args,
                    function_name=function_name,
                    implicit=implicit,
                    explicit=explicit,
                )
            )
            + ")"
        )

    def generate_texture_sample_call(self, function_name, args):
        if len(args) == 2:
            texture, coords = args
            array_info = self.sampled_texture_array_access(texture)
            if (
                array_info is not None
                and self.sampled_texture_array_element_binding(texture) is None
            ):
                return self.generate_sampled_texture_array_sample_call(
                    texture, coords, array_info
                )
            sample_function = (
                "textureSampleLevel"
                if self._current_stage_name == "compute"
                else "textureSample"
            )
            level_argument = ", 0.0" if sample_function == "textureSampleLevel" else ""
            return (
                f"{sample_function}({self.generate_expression(texture)}, "
                f"{self.texture_sampler_expression(texture)}, "
                f"{self.sampled_texture_coordinates(texture, coords, function_name)}"
                f"{level_argument})"
            )
        if len(args) == 3:
            texture, second, third = args
            if self.is_sampler_type(self.expression_type(second)):
                comparison_call = self.explicit_shadow_comparison_call(
                    function_name, texture, second, third
                )
                if comparison_call is not None:
                    return comparison_call
                sample_function = (
                    "textureSampleLevel"
                    if self._current_stage_name == "compute"
                    else "textureSample"
                )
                self.refuse_comparison_sampler(
                    second, function_name, texture, sample_function
                )
                level_argument = (
                    ", 0.0" if sample_function == "textureSampleLevel" else ""
                )
                return (
                    f"{sample_function}({self.generate_expression(texture)}, "
                    f"{self.generate_expression(second)}, "
                    f"{self.sampled_texture_coordinates(texture, third, function_name)}"
                    f"{level_argument})"
                )
            # GLSL texture(sampler, coords, bias): the second argument is the
            # coordinate, not a sampler, so the split sampler the declaration
            # provides is used and the bias goes to textureSampleBias
            # (issue #2462).
            return self.generate_texture_sample_bias_call(
                function_name, texture, second, third
            )
        if len(args) == 4:
            texture, sampler, coords, bias = args
            if self.is_sampler_type(self.expression_type(sampler)):
                # HLSL SampleBias, Metal sample(s, uv, bias) and the CrossGL
                # form texture(t, s, coords, bias): the explicit sampler
                # goes to textureSampleBias (issue #3584).
                return self.generate_texture_sample_bias_call(
                    function_name, texture, coords, bias, sampler=sampler
                )
            raise WGSLTextureBiasUnsupportedError(
                function_name,
                self.generate_expression(texture),
                "texture() with four arguments takes "
                "texture/sampler/coords/bias, but the second argument is "
                "not a sampler: "
                f"{self.raw_type_name_string(self.expression_type(sampler)) or 'an unknown type'}",
            )
        raise WGSLRefusal(
            "WGSL target supports texture() calls with texture/coords, "
            "texture/sampler/coords, texture/coords/bias or "
            "texture/sampler/coords/bias arguments; got "
            f"{len(args)} argument(s) for {function_name}"
        )

    def generate_texture_sample_bias_call(
        self, function_name, texture, coords, bias, sampler=None
    ):
        """A biased ``texture()`` as ``textureSampleBias`` (see :mod:`.wgsl_texture_bias`).

        With an explicit ``sampler`` (the four-argument form of issue
        #3584) that sampler is used; otherwise the split sampler the
        declaration provides is used (the three-argument form of issue
        #2462).
        """
        if self._current_stage_name == "compute":
            raise ImplicitLodStageUnsupportedError(
                "wgsl",
                function_name,
                "compute",
                "a level-of-detail bias needs implicit derivatives, which WGSL "
                "has only in fragment shaders, where textureSampleBias is "
                "available",
            )
        texture_name = self.generate_expression(texture)
        array_info = self.sampled_texture_array_access(texture)
        if (
            array_info is not None
            and self.sampled_texture_array_element_binding(texture) is None
        ):
            raise WGSLTextureBiasUnsupportedError(
                function_name,
                texture_name,
                "an element of a sampled texture array selected at run time is "
                "read through a dispatch helper that takes no bias; index the "
                "array with a constant",
            )
        texture_type = self.sampled_texture_type(self.expression_type(texture))
        if not has_texture_sample_bias(texture_type):
            raise WGSLTextureBiasUnsupportedError(
                function_name,
                texture_name,
                "textureSampleBias has no overload for "
                f"{texture_type or 'a texture of unknown type'}; it takes 2D, "
                "2D array, 3D, cube and cube array textures",
            )
        if sampler is None:
            sampler_name = self.texture_sampler_expression(texture)
        else:
            sampler_name = self.generate_expression(sampler)
            self.refuse_comparison_sampler(
                sampler, function_name, texture_name, "textureSampleBias", bias=True
            )
        bias_type = self.expression_type(bias)
        if bias_type is not None and self.scalar_type_name(bias_type) not in {
            "f32",
            "i32",
            "u32",
        }:
            bias_position = "third" if sampler is None else "fourth"
            raise WGSLTextureBiasUnsupportedError(
                function_name,
                texture_name,
                f"the bias is a float scalar, but the {bias_position} argument is "
                f"{self.raw_type_name_string(bias_type)}",
            )
        return (
            f"textureSampleBias({texture_name}, {sampler_name}, "
            f"{self.sampled_texture_coordinates(texture, coords, function_name)}, "
            f"{self.generate_expression_for_target(bias, 'float')})"
        )

    def generate_sampled_texture_array_sample_call(self, texture, coords, array_info):
        helper_name = self.sampled_texture_array_sample_helper_name(
            array_info["source_name"]
        )
        self._sampled_texture_array_dispatch_helpers[helper_name] = {
            "helper_name": helper_name,
            "array_info": array_info,
            "kind": "texture_sample",
        }
        return (
            f"{helper_name}(i32({self.generate_expression(texture.index_expr)}), "
            f"{self.generate_expression(coords)})"
        )

    def sampled_texture_array_sample_helper_name(self, array_name):
        return self.safe_wgsl_identifier(f"{array_name}_sample")

    def generate_texture_sample_level_call(self, function_name, args):
        if len(args) == 3:
            texture, coords, level = args
            return (
                f"textureSampleLevel({self.generate_expression(texture)}, "
                f"{self.texture_sampler_expression(texture)}, "
                f"{self.sampled_texture_coordinates(texture, coords, function_name)}, "
                f"{self.generate_expression(level)})"
            )
        if len(args) == 4:
            texture, sampler, coords, level = args
            comparison_call = self.explicit_shadow_comparison_call(
                function_name, texture, sampler, coords, lod=level
            )
            if comparison_call is not None:
                return comparison_call
            self.refuse_comparison_sampler(
                sampler, function_name, texture, "textureSampleLevel"
            )
            return (
                f"textureSampleLevel({self.generate_expression(texture)}, "
                f"{self.generate_expression(sampler)}, "
                f"{self.sampled_texture_coordinates(texture, coords, function_name)}, "
                f"{self.generate_expression(level)})"
            )
        raise WGSLRefusal(
            "WGSL target supports textureLod() calls with texture/coords/lod or "
            "texture/sampler/coords/lod arguments; got "
            f"{len(args)} argument(s) for {function_name}"
        )

    def generate_texture_sample_level_offset_call(self, function_name, args):
        if len(args) == 5:
            comparison_call = self.explicit_shadow_comparison_call(
                function_name, args[0], args[1], args[2], lod=args[3], offset=args[4]
            )
            if comparison_call is not None:
                return comparison_call
            self.refuse_comparison_sampler(
                args[1], function_name, args[0], "textureSampleLevel"
            )
        return self.generate_texture_builtin_call(
            "textureSampleLevel", function_name, args, implicit=4, explicit=5
        )

    def generate_texture_sample_grad_call(self, function_name, args):
        if len(args) == 5:
            self.refuse_comparison_sampler(
                args[1], function_name, args[0], "textureSampleGrad"
            )
        one_dimensional = wgsl_texture_1d_grad_call(
            self, function_name, args, offset=False
        )
        if one_dimensional is not None:
            return one_dimensional
        return self.generate_texture_builtin_call(
            "textureSampleGrad", function_name, args, implicit=4, explicit=5
        )

    def generate_texture_sample_grad_offset_call(self, function_name, args):
        if len(args) == 6:
            self.refuse_comparison_sampler(
                args[1], function_name, args[0], "textureSampleGrad"
            )
        wgsl_texture_1d_grad_call(self, function_name, args, offset=True)
        return self.generate_texture_builtin_call(
            "textureSampleGrad", function_name, args, implicit=5, explicit=6
        )

    def generate_texture_sample_offset_call(self, function_name, args):
        if len(args) == 3:
            return self.generate_texture_builtin_call(
                "textureSample", function_name, args, implicit=3, explicit=4
            )
        if len(args) == 4:
            if self.is_sampler_type(self.expression_type(args[1])):
                comparison_call = self.explicit_shadow_comparison_call(
                    function_name, args[0], args[1], args[2], offset=args[3]
                )
                if comparison_call is not None:
                    return comparison_call
                self.refuse_comparison_sampler(
                    args[1], function_name, args[0], "textureSample"
                )
                return self.generate_texture_builtin_call(
                    "textureSample", function_name, args, implicit=3, explicit=4
                )
            texture, coords, offset, bias = args
            return (
                "textureSampleBias("
                f"{self.generate_expression(texture)}, "
                f"{self.texture_sampler_expression(texture)}, "
                f"{self.sampled_texture_coordinates(texture, coords, function_name)}, "
                f"{self.generate_expression(bias)}, {self.generate_expression(offset)})"
            )
        if len(args) == 5:
            self.refuse_comparison_sampler(
                args[1], function_name, args[0], "textureSampleBias", bias=True
            )
            call_args = self.generate_texture_call_args(
                args, function_name=function_name, implicit=4, explicit=5
            )
            texture, sampler, coords, offset, bias = call_args
            return (
                "textureSampleBias("
                f"{texture}, {sampler}, {coords}, {bias}, {offset})"
            )
        raise WGSLRefusal(
            "WGSL target supports textureOffset() calls with texture/coords/offset, "
            "texture/sampler/coords/offset, or texture/sampler/coords/offset/bias "
            f"arguments; got {len(args)} argument(s)"
        )

    def explicit_shadow_comparison_call(
        self, function_name, texture, sampler, coords, *, lod=None, offset=None
    ):
        """Lower a shadow lookup that pairs a depth texture with a comparison_sampler.

        Returns ``None`` when the call is not such a pairing, so the caller keeps its
        ordinary path. Otherwise the vec3 coordinate is split into the texture
        coordinate (``xy``) and the comparison reference (``z``). Outside the fragment
        stage, and for a literal zero LOD, the level-zero ``textureSampleCompareLevel``
        is used, since ``textureSampleCompare`` needs implicit derivatives.
        """
        if not (
            self.is_depth_texture_type(self.expression_type(texture))
            and self.is_comparison_sampler_type(self.expression_type(sampler))
        ):
            return None
        texture_text = self.generate_expression(texture)
        sampler_text = self.generate_expression(sampler)
        texture_type = self.expression_type(texture)
        type_name = self.resource_type_name(texture_type)
        type_label = (
            texture_type.name if isinstance(texture_type, NamedType) else type_name
        )

        def refuse(reason):
            raise WGSLComparisonSamplerUnsupportedError(
                function_name,
                texture_text,
                "textureSampleCompare",
                sampler_text,
                reason=reason,
            )

        if "cube" in type_name or "array" in type_name:
            refuse(
                f"{type_label} lookups take a vec4 coordinate, and WGSL output "
                "lowers only 2D shadow textures through a comparison_sampler"
            )
        coordinate_shape = self.vector_shape(self.expression_type(coords))
        if coordinate_shape is None or coordinate_shape[1] != 3:
            refuse(
                f"{type_label} lookups need a vec3 coordinate (uv plus the "
                "comparison reference)"
            )
        if not self.is_repeatable_matrix_operand(coords):
            refuse(
                "the coordinate is split into a texture coordinate and a comparison "
                "reference, so a coordinate with side effects must be assigned to a "
                "local variable first"
            )
        if lod is not None and not (
            isinstance(lod, LiteralNode) and lod.value in {0, 0.0}
        ):
            refuse(
                "textureSampleCompareLevel samples mip level 0, so only a literal "
                "0 LOD can be lowered"
            )

        arguments = [
            texture,
            sampler,
            MemberAccessNode(coords, "xy"),
            MemberAccessNode(coords, "z"),
        ]
        if offset is not None:
            arguments.append(offset)
        level_only = lod is not None or self._current_stage_name in {
            "compute",
            "vertex",
        }
        builtin = "textureSampleCompareLevel" if level_only else "textureSampleCompare"
        return self.generate_texture_builtin_call(
            builtin,
            function_name,
            arguments,
            implicit=len(arguments) - 1,
            explicit=len(arguments),
        )

    def generate_texture_sample_compare_call(self, function_name, args):
        self.require_depth_texture_operand(args[0], function_name)
        self.require_comparison_sampler_operand(args, function_name, implicit=3)
        return self.generate_texture_builtin_call(
            "textureSampleCompare", function_name, args, implicit=3, explicit=4
        )

    def generate_texture_sample_compare_offset_call(self, function_name, args):
        self.require_depth_texture_operand(args[0], function_name)
        self.require_comparison_sampler_operand(args, function_name, implicit=4)
        return self.generate_texture_builtin_call(
            "textureSampleCompare", function_name, args, implicit=4, explicit=5
        )

    def generate_texture_sample_compare_level_call(self, function_name, args):
        self.require_depth_texture_operand(args[0], function_name)
        self.require_comparison_sampler_operand(args, function_name, implicit=4)
        self.require_zero_compare_level_operand(args, function_name, implicit=4)
        call_args = self.generate_texture_call_args(
            args, function_name=function_name, implicit=4, explicit=5
        )
        call_args.pop()
        return "textureSampleCompareLevel(" + ", ".join(call_args) + ")"

    def generate_texture_sample_compare_level_offset_call(self, function_name, args):
        self.require_depth_texture_operand(args[0], function_name)
        self.require_comparison_sampler_operand(args, function_name, implicit=5)
        self.require_zero_compare_level_operand(args, function_name, implicit=5)
        call_args = self.generate_texture_call_args(
            args, function_name=function_name, implicit=5, explicit=6
        )
        offset = call_args.pop()
        call_args.pop()
        call_args.append(offset)
        return "textureSampleCompareLevel(" + ", ".join(call_args) + ")"

    def generate_texture_dimensions_call(self, args):
        if len(args) not in {1, 2}:
            raise WGSLRefusal(
                "WGSL target supports textureSize() calls with texture or "
                f"texture/lod arguments; got {len(args)} argument(s)"
            )
        texture_expr = self.generate_expression(args[0])
        dimensions = (
            "textureDimensions("
            + ", ".join(
                [texture_expr] + [self.generate_expression(arg) for arg in args[1:]]
            )
            + ")"
        )
        expected_type = self._current_expression_expected_type
        expected_shape = self.vector_shape(expected_type)
        texture_type = self.sampled_texture_type(
            self.expression_type(args[0])
        ) or self.multisampled_texture_type(self.expression_type(args[0]))
        if texture_type is None:
            if expected_shape is not None:
                element_type, size = expected_shape
                if element_type == "f32":
                    return f"vec{size}<f32>({dimensions})"
            return dimensions

        # CrossGL's textureSize follows GLSL: it returns int/ivecN, and an
        # array texture's last component is its layer count. WGSL's
        # textureDimensions returns u32/vecN<u32> without the layers.
        base_type = re.sub(r"<[^>]*>$", "", texture_type)
        if base_type.endswith("_array"):
            dimensions = f"vec3<u32>({dimensions}, textureNumLayers({texture_expr}))"
            size = 3
        elif base_type.endswith("_1d"):
            size = 1
        elif base_type.endswith("_3d"):
            size = 3
        else:
            size = 2

        if expected_shape is not None:
            element_type = expected_shape[0]
        else:
            try:
                element_type = (
                    self.type_name_string(expected_type)
                    if expected_type is not None
                    else None
                )
            except ValueError:
                element_type = None
        if element_type not in {"i32", "u32", "f32"}:
            element_type = "i32"
        if element_type == "u32":
            return dimensions
        if size == 1:
            return f"{element_type}({dimensions})"
        return f"vec{size}<{element_type}>({dimensions})"

    def generate_image_dimensions_call(self, args):
        if len(args) != 1:
            raise WGSLRefusal(
                "WGSL target supports imageSize() calls with exactly 1 "
                f"argument; got {len(args)}"
            )
        image = args[0]
        if self.storage_texture_dimension_for_expression(image) is None:
            raise WGSLRefusal(
                "WGSL target requires imageSize() to use a storage image resource"
            )
        image_expr = self.generate_expression(image)
        dimensions = f"textureDimensions({image_expr})"

        # GLSL's imageSize returns int for 1D images, ivec2 for 2D and cube
        # images, and ivec3 for 3D and array images with the array size last.
        # Cube images are lowered to 2D array storage textures with six layers
        # per cube, so a cube array's last component is the number of cubes.
        image_type = self.resource_type_name(self.expression_type(image)) or ""
        base_type = (
            image_type[1:]
            if image_type.startswith(("iimage", "uimage"))
            else image_type
        )
        if base_type == "image1d":
            return f"i32({dimensions})"
        if base_type == "image3d":
            return f"vec3<i32>({dimensions})"
        if base_type == "image2darray":
            return (
                f"vec3<i32>(vec3<u32>({dimensions}, "
                f"textureNumLayers({image_expr})))"
            )
        if base_type == "imagecubearray":
            return (
                f"vec3<i32>(vec3<u32>({dimensions}, "
                f"textureNumLayers({image_expr}) / 6u))"
            )
        return f"vec2<i32>({dimensions})"

    def generate_texel_fetch_call(self, function_name, args):
        if len(args) != 3:
            raise WGSLRefusal(
                "WGSL target supports texelFetch() calls with exactly 3 "
                f"arguments; got {len(args)} for {function_name}"
            )
        texture, coords, level = args
        split = self.split_array_texture_coordinates(texture, coords, function_name)
        if split is not None:
            texel, layer = split
            return (
                f"textureLoad({self.generate_expression(texture)}, {texel}, "
                f"{layer}, {self.generate_expression(level)})"
            )
        return (
            f"textureLoad({self.generate_expression(texture)}, "
            f"{self.generate_expression(coords)}, "
            f"{self.generate_expression(level)})"
        )

    def generate_texel_fetch_offset_call(self, function_name, args):
        """``texelFetchOffset(t, P, lod, offset)`` reads the texel at ``P + offset`` (GLSL 4.60 section 8.9.2)."""
        if len(args) != 4:
            raise WGSLRefusal(
                "WGSL target supports texelFetchOffset() calls with exactly 4 "
                f"arguments; got {len(args)} for {function_name}"
            )
        texture, coords, level, offset = args
        split = self.split_array_texture_coordinates(texture, coords, function_name)
        if split is not None:
            texel, layer = split
            return (
                f"textureLoad({self.generate_expression(texture)}, "
                f"({texel} + {self.generate_expression(offset)}), {layer}, "
                f"{self.generate_expression(level)})"
            )
        return (
            f"textureLoad({self.generate_expression(texture)}, "
            f"({self.generate_expression(coords)} + {self.generate_expression(offset)}), "
            f"{self.generate_expression(level)})"
        )

    def generate_image_load_call(self, node, function_name):
        if len(node.arguments) != 2:
            raise WGSLRefusal(
                "WGSL target supports imageLoad() calls with exactly 2 "
                f"arguments; got {len(node.arguments)} for {function_name}"
            )
        image, coords = node.arguments
        self.require_readable_storage_texture_resource(image, function_name)
        call_args = self.storage_texture_coordinate_args(image, coords)
        return (
            f"textureLoad({self.generate_expression(image)}, "
            f"{', '.join(call_args)})"
        )

    def generate_image_store_call(self, node, function_name):
        if len(node.arguments) != 3:
            raise WGSLRefusal(
                "WGSL target supports imageStore() calls with exactly 3 "
                f"arguments; got {len(node.arguments)} for {function_name}"
            )
        image, coords, value = node.arguments
        self.require_writable_storage_texture_resource(image, function_name)
        call_args = self.storage_texture_coordinate_args(image, coords)
        return (
            f"textureStore({self.generate_expression(image)}, "
            f"{', '.join(call_args)}, "
            f"{self.generate_expression(value)})"
        )

    # Array textures whose GLSL coordinate carries the layer in its last
    # component. WGSL passes the layer as a separate array_index argument.
    ARRAY_TEXTURE_COORDINATE_SIZES = {
        "texture_2d_array": 2,
        "texture_cube_array": 3,
        "texture_depth_2d_array": 2,
        "texture_depth_cube_array": 3,
    }

    def array_texture_coordinate_size(self, texture_expr):
        texture_type = self.sampled_texture_type(self.expression_type(texture_expr))
        if texture_type is None:
            return None
        return self.ARRAY_TEXTURE_COORDINATE_SIZES.get(
            re.sub(r"<[^>]*>$", "", texture_type)
        )

    def split_array_texture_coordinates(self, texture_expr, coords_expr, function_name):
        """``(coordinate, layer)`` for a lookup on an array texture, else None.

        GLSL and HLSL put the layer in the last coordinate component, WGSL
        takes it as its own ``array_index`` argument.
        """
        size = self.array_texture_coordinate_size(texture_expr)
        if size is None:
            return None
        if not self.is_repeatable_matrix_operand(coords_expr):
            raise WGSLRefusal(
                f"WGSL target splits the layer out of the {function_name}() "
                "coordinate of an array texture, which reads the coordinate twice; "
                "assign the coordinate to a local variable first"
            )
        rendered = self.generate_expression(coords_expr)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", rendered):
            rendered = f"({rendered})"
        components = "xyzw"
        return (
            f"{rendered}.{components[:size]}",
            f"{rendered}.{components[size]}",
        )

    def sampled_texture_coordinates(self, texture_expr, coords_expr, function_name):
        """The coordinate arguments of a ``textureSample*`` call.

        For an array texture this is ``coordinate, layer``. GLSL selects the
        layer of a filtered lookup as ``floor(layer + 0.5)`` (GLSL 4.60
        section 8.9), so the float layer is rounded before the conversion.
        """
        split = self.split_array_texture_coordinates(
            texture_expr, coords_expr, function_name
        )
        if split is None:
            return self.generate_expression(coords_expr)
        coordinate, layer = split
        return f"{coordinate}, i32(floor({layer} + 0.5))"

    def texture_sampler_expression(self, texture_expr):
        array_binding = self.sampled_texture_array_element_binding(texture_expr)
        if array_binding is not None:
            return array_binding["sampler_name"]
        resource_binding = self.resource_member_binding_for_access(texture_expr)
        if resource_binding is not None:
            return self.texture_sampler_name(resource_binding["binding_name"])
        if isinstance(texture_expr, IdentifierNode):
            return self.texture_sampler_name(self.identifier_name(texture_expr.name))
        raise WGSLRefusal(
            "WGSL target cannot infer a companion sampler for texture expression "
            f"{self.generate_expression(texture_expr)}; pass an explicit sampler"
        )

    def require_depth_texture_operand(self, texture_expr, function_name):
        texture_type = self.expression_type(texture_expr)
        if not self.is_depth_texture_type(texture_type):
            raise WGSLRefusal(
                f"WGSL target requires {function_name}() to use a shadow/depth "
                "texture resource"
            )

    def require_comparison_sampler_operand(self, args, function_name, *, implicit):
        if len(args) == implicit:
            return
        if len(args) != implicit + 1:
            return
        sampler_type = self.expression_type(args[1])
        if not self.is_comparison_sampler_type(sampler_type):
            raise WGSLRefusal(
                f"WGSL target requires {function_name}() explicit sampler operand "
                "to be a comparison_sampler"
            )

    def require_zero_compare_level_operand(self, args, function_name, *, implicit):
        if self.semantic_key(function_name).endswith("offset"):
            level_index = len(args) - 2
        else:
            level_index = len(args) - 1
        if len(args) <= level_index:
            return
        level_expr = args[level_index]
        if isinstance(level_expr, LiteralNode) and level_expr.value in {0, 0.0}:
            return
        raise WGSLRefusal(
            f"WGSL target only lowers {function_name}() when the explicit LOD "
            "operand is literal 0 because textureSampleCompareLevel samples "
            "mip level 0"
        )

    def pointer_argument_expression(self, pointer_expr):
        if isinstance(pointer_expr, IdentifierNode) and self.is_pointer_identifier(
            pointer_expr.name
        ):
            return self.identifier_name(pointer_expr.name)
        return f"&{self.generate_expression(pointer_expr)}"

    def is_multi_barrier_memory_lowering(self, expression):
        if not isinstance(expression, FunctionCallNode):
            return False
        function_name = self.expression_name(expression.function)
        if not isinstance(function_name, str):
            return False
        lowering = self.WGSL_MEMORY_BARRIER_LOWERING.get(
            self.semantic_key(function_name)
        )
        return lowering is not None and len(lowering) > 1

    def generate_barrier_call(self, node, function_name):
        if node.arguments:
            raise WGSLRefusal(
                f"WGSL target does not support arguments for barrier function {function_name}"
            )
        if self._current_stage_name != "compute":
            raise WGSLRefusal(
                "WGSL target only supports barrier() inside compute stages"
            )
        lowering = self.WGSL_MEMORY_BARRIER_LOWERING.get(
            self.semantic_key(function_name)
        )
        if lowering is not None:
            return "; ".join(f"{barrier}()" for barrier in lowering)
        return "workgroupBarrier()"
