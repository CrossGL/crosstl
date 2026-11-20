"""Samplers, texture sample options, compare, gather and size queries, and sampled and storage texture reads."""

from ....translator.semantics.array_layers import metal_sampled_layer_source_argument
from ....translator.semantics.sampler_states import (
    SAMPLER_STATE_CALL_ARGUMENTS,
    canonical_sampler_argument,
)
from ..MetalAst import (
    FunctionCallNode,
    MethodCallNode,
    VariableNode,
)
from ..texture_types import combined_texture_type


class MetalTextureAccessMixin:
    """Members of ``MetalToCrossGLConverter``: samplers, texture sample options, compare, gather and size queries, and sampled and storage texture reads."""

    def generate_sampler_constructor_arg(self, arg, is_main=False):
        if isinstance(arg, VariableNode) and self.is_scoped_identifier(arg.name):
            return canonical_sampler_argument(arg.name)
        if isinstance(arg, FunctionCallNode) and isinstance(arg.name, str):
            name = canonical_sampler_argument(arg.name)
            if name in SAMPLER_STATE_CALL_ARGUMENTS:
                arguments = ", ".join(
                    self.generate_expression(item, is_main) for item in arg.args
                )
                return f"{name}({arguments})"
        return self.generate_expression(arg, is_main)

    def program_scope_sampler_constant(self, variable, initializer):
        """``const sampler NAME = sampler(...);`` for a program-scope constexpr sampler."""
        name = self.sanitize_identifier(variable.name)
        extents = [
            self.generate_expression(size, False)
            for size in getattr(variable, "array_sizes", None) or []
        ]
        array_type = self.metal_array_type_parts(
            self.normalized_metal_type(getattr(variable, "vtype", None))
        )
        if array_type is not None:
            extents.insert(0, array_type[1])
        if initializer is None:
            value = "sampler()"
        else:
            value = self.generate_expression(initializer, False)
        suffix = "".join(f"[{extent}]" for extent in extents)
        return f"    const sampler {name}{suffix} = {value};\n"

    def is_scoped_identifier(self, name):
        return (
            isinstance(name, str)
            and "::" in name
            and all(
                self.crossgl_identifier_pattern.match(part)
                and part not in self.crossgl_reserved_identifiers
                for part in name.split("::")
            )
        )

    def unwrap_texture_option_argument(self, expr, option_name):
        if (
            isinstance(expr, FunctionCallNode)
            and self.unscoped_function_name(expr.name) == option_name
            and len(expr.args) == 1
        ):
            return expr.args[0]
        return expr

    def unscoped_function_name(self, name):
        return str(name).split("::")[-1]

    def texture_sample_options_call(self, options, sample_args, is_main=False):
        if not options:
            return f"texture({', '.join(sample_args)})"
        clamp_split = self.texture_min_lod_clamp_options(options, is_main)
        if clamp_split is not None:
            return self.texture_min_lod_clamp_sample_call(
                clamp_split, sample_args, is_main
            )
        if len(options) > 2:
            return None

        option = options[0]
        offset = options[1] if len(options) == 2 else None

        if (
            isinstance(option, FunctionCallNode)
            and self.unscoped_function_name(option.name) == "bias"
            and len(option.args) == 1
        ):
            bias = self.generate_expression(option.args[0], is_main)
            if offset is not None:
                rendered_offset = self.generate_expression(offset, is_main)
                args = sample_args + [rendered_offset, bias]
                return f"textureOffset({', '.join(args)})"
            return f"texture({', '.join(sample_args + [bias])})"

        gradient_args = self.texture_gradient_option_arguments(option)
        if gradient_args is not None:
            ddx = self.generate_expression(gradient_args[0], is_main)
            ddy = self.generate_expression(gradient_args[1], is_main)
            if offset is not None:
                rendered_offset = self.generate_expression(offset, is_main)
                return (
                    f"textureGradOffset("
                    f"{', '.join(sample_args + [ddx, ddy, rendered_offset])})"
                )
            return f"textureGrad({', '.join(sample_args + [ddx, ddy])})"

        level_arg = self.unwrap_texture_option_argument(option, "level")
        if level_arg is not option or not self.texture_sample_option_is_offset(option):
            lod = self.generate_expression(level_arg, is_main)
            if offset is not None:
                rendered_offset = self.generate_expression(offset, is_main)
                return (
                    f"textureLodOffset("
                    f"{', '.join(sample_args + [lod, rendered_offset])})"
                )
            return f"textureLod({', '.join(sample_args + [lod])})"

        rendered_offset = self.generate_expression(option, is_main)
        return f"textureOffset({', '.join(sample_args + [rendered_offset])})"

    METAL_SAMPLE_OPTION_NAMES = frozenset(
        {
            "bias",
            "level",
            "min_lod_clamp",
            "gradient2d",
            "gradient3d",
            "gradientcube",
        }
    )

    def texture_min_lod_clamp_options(self, options, is_main=False):
        """``(leading option, clamp, offset)`` for MSL options that hold ``min_lod_clamp``, otherwise ``None``.

        MSL accepts ``min_lod_clamp`` alone or after ``bias`` or a ``gradient*`` option, and before the offset
        (MSL specification, section 6.12). CrossGL's ``texture*Clamp`` operations carry the same clamp.
        """
        positions = [
            index
            for index, option in enumerate(options)
            if self.unwrap_texture_option_argument(option, "min_lod_clamp")
            is not option
        ]
        if not positions:
            return None
        index = positions[0]
        leading, trailing = options[:index], options[index + 1 :]
        if (
            len(positions) > 1
            or len(leading) > 1
            or len(trailing) > 1
            or any(
                isinstance(option, FunctionCallNode)
                and self.unscoped_function_name(option.name)
                in self.METAL_SAMPLE_OPTION_NAMES
                for option in trailing
            )
        ):
            return False
        clamp = self.generate_expression(
            self.unwrap_texture_option_argument(options[index], "min_lod_clamp"),
            is_main,
        )
        offset = self.generate_expression(trailing[0], is_main) if trailing else None
        return (leading[0] if leading else None), clamp, offset

    def texture_min_lod_clamp_sample_call(self, clamp_split, sample_args, is_main):
        if clamp_split is False:
            return self.refuse_import(
                "Metal sample options around min_lod_clamp: MSL takes at most one bias or gradient option before "
                "min_lod_clamp and an offset after it",
                "vec4(0.0)",
            )
        option, clamp, offset = clamp_split
        if option is None:
            if offset is not None:
                return f"textureOffsetClamp({', '.join(sample_args + [offset, clamp])})"
            return f"textureClamp({', '.join(sample_args + [clamp])})"
        bias_arg = self.unwrap_texture_option_argument(option, "bias")
        if bias_arg is not option:
            bias = self.generate_expression(bias_arg, is_main)
            if offset is not None:
                return (
                    "textureOffsetClamp("
                    f"{', '.join(sample_args + [offset, clamp, bias])})"
                )
            return f"textureClamp({', '.join(sample_args + [clamp, bias])})"
        gradient_args = self.texture_gradient_option_arguments(option)
        if gradient_args is not None:
            ddx = self.generate_expression(gradient_args[0], is_main)
            ddy = self.generate_expression(gradient_args[1], is_main)
            if offset is not None:
                return (
                    "textureGradOffsetClamp("
                    f"{', '.join(sample_args + [ddx, ddy, offset, clamp])})"
                )
            return f"textureGradClamp({', '.join(sample_args + [ddx, ddy, clamp])})"
        return self.refuse_import(
            "Metal sample option combined with min_lod_clamp: MSL clamps only a computed level of detail, so "
            "min_lod_clamp combines with bias or gradient options, not with an explicit level",
            "vec4(0.0)",
        )

    def texture_sample_option_is_offset(self, option):
        mapped_type = self.expression_mapped_type(option)
        if self.is_integer_vector_type(mapped_type):
            return True
        constructor_type = getattr(option, "name", None) or getattr(
            option, "type_name", None
        )
        if constructor_type is None:
            return False
        return self.is_integer_vector_type(self.map_type(str(constructor_type)))

    def is_integer_vector_type(self, type_name):
        return type_name in {
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "i16vec2",
            "i16vec3",
            "i16vec4",
            "u16vec2",
            "u16vec3",
            "u16vec4",
            "i8vec2",
            "i8vec3",
            "i8vec4",
            "u8vec2",
            "u8vec3",
            "u8vec4",
        }

    def texture_gradient_option_arguments(self, option):
        if (
            isinstance(option, FunctionCallNode)
            and self.unscoped_function_name(option.name)
            in {"gradient2d", "gradient3d", "gradientcube"}
            and len(option.args) == 2
        ):
            return option.args
        return None

    def resource_classification_type(self, mapped_type):
        """The combined CrossGL spelling of a mapped resource type, without pointer or reference suffixes."""
        if not mapped_type:
            return mapped_type
        base = str(mapped_type).strip()
        while base.endswith("*") or base.endswith("&"):
            base = base[:-1].strip()
        return combined_texture_type(base)

    def sampled_array_coordinate_constructor(self, texture_expr):
        mapped_type = self.resource_classification_type(
            self.expression_mapped_type(texture_expr)
        )
        return self.sampled_array_coordinate_constructor_for_type(mapped_type)

    def sampled_array_coordinate_constructor_for_type(self, mapped_type):
        return {
            "sampler1DArray": "vec2",
            "isampler1DArray": "vec2",
            "usampler1DArray": "vec2",
            "sampler2DArray": "vec3",
            "isampler2DArray": "vec3",
            "usampler2DArray": "vec3",
            "sampler2DArrayShadow": "vec3",
            "samplerCubeArray": "vec4",
            "samplerCubeArrayShadow": "vec4",
        }.get(mapped_type)

    def texture_sample_coordinate_and_options(
        self, texture_expr, coords_expr, options, is_main=False
    ):
        coords = self.generate_expression(coords_expr, is_main)
        remaining_options = list(options)
        constructor = self.sampled_array_coordinate_constructor(texture_expr)
        if constructor and remaining_options:
            layer = self.sampled_array_layer_argument(remaining_options.pop(0), is_main)
            coords = f"{constructor}({coords}, {layer})"
        return coords, remaining_options

    def sampled_array_layer_argument(self, layer_expr, is_main=False):
        """Render a Metal sampled-array layer argument as a CrossGL coordinate."""
        layer = self.generate_expression(layer_expr, is_main)
        metal_type = self.resolve_type_alias(self.expression_metal_type(layer_expr))
        return metal_sampled_layer_source_argument(layer, metal_type)

    def texture_compare_method_base_arguments(
        self, obj_expr, method_args, is_main=False
    ):
        mapped_type = self.resource_classification_type(
            self.expression_mapped_type(obj_expr)
        )
        if mapped_type == "sampler2DArrayShadow" and len(method_args) >= 4:
            sampler = self.generate_expression(method_args[0], is_main)
            coord = self.generate_expression(method_args[1], is_main)
            layer = self.sampled_array_layer_argument(method_args[2], is_main)
            compare = self.generate_expression(method_args[3], is_main)
            return [sampler, f"vec3({coord}, {layer})", compare], 4
        if mapped_type == "samplerCubeArrayShadow" and len(method_args) >= 4:
            sampler = self.generate_expression(method_args[0], is_main)
            coord = self.generate_expression(method_args[1], is_main)
            layer = self.sampled_array_layer_argument(method_args[2], is_main)
            compare = self.generate_expression(method_args[3], is_main)
            return [sampler, f"vec4({coord}, {layer})", compare], 4
        return [self.generate_expression(arg, is_main) for arg in method_args[:3]], 3

    def texture_compare_option_method_call(
        self, obj, obj_expr, method_args, is_main=False
    ):
        compare_args, compare_arg_count = self.texture_compare_method_base_arguments(
            obj_expr, method_args, is_main
        )
        if len(method_args) == compare_arg_count and compare_arg_count == 4:
            return f"textureCompare({obj}, {', '.join(compare_args)})"
        clamp_split = self.texture_min_lod_clamp_options(
            method_args[compare_arg_count:], is_main
        )
        if clamp_split is not None:
            return self.texture_compare_min_lod_clamp_call(
                obj, compare_args, clamp_split, is_main
            )
        if len(method_args) not in {compare_arg_count + 1, compare_arg_count + 2}:
            return None

        option = method_args[compare_arg_count]
        unsupported_option = self.unsupported_texture_compare_lod_option(obj, option)
        if unsupported_option is not None:
            return unsupported_option

        level_arg = self.unwrap_texture_option_argument(option, "level")
        if level_arg is not option:
            lod = self.generate_expression(level_arg, is_main)
            if len(method_args) == compare_arg_count + 1:
                return f"textureCompareLod({obj}, {', '.join(compare_args + [lod])})"
            offset = self.generate_expression(
                method_args[compare_arg_count + 1], is_main
            )
            return (
                f"textureCompareLodOffset("
                f"{obj}, {', '.join(compare_args + [lod, offset])})"
            )

        gradient_args = self.texture_gradient_option_arguments(option)
        if gradient_args is not None:
            ddx = self.generate_expression(gradient_args[0], is_main)
            ddy = self.generate_expression(gradient_args[1], is_main)
            if len(method_args) == compare_arg_count + 1:
                return (
                    f"textureCompareGrad({obj}, {', '.join(compare_args + [ddx, ddy])})"
                )
            offset = self.generate_expression(
                method_args[compare_arg_count + 1], is_main
            )
            return (
                f"textureCompareGradOffset("
                f"{obj}, {', '.join(compare_args + [ddx, ddy, offset])})"
            )

        if len(method_args) == compare_arg_count + 1 and not isinstance(
            option, FunctionCallNode
        ):
            offset = self.generate_expression(option, is_main)
            return f"textureCompareOffset({obj}, {', '.join(compare_args + [offset])})"

        return None

    def texture_compare_min_lod_clamp_call(
        self, obj, compare_args, clamp_split, is_main
    ):
        if clamp_split is False:
            return self.refuse_import(
                f"Metal depth compare options around min_lod_clamp on {obj}: MSL takes at most one gradient "
                "option before min_lod_clamp and an offset after it",
                "0.0",
            )
        option, clamp, offset = clamp_split
        gradient_args = (
            self.texture_gradient_option_arguments(option)
            if option is not None
            else None
        )
        if option is not None and gradient_args is None:
            return self.unsupported_texture_compare_lod_option(obj, option) or (
                self.refuse_import(
                    f"Metal depth compare option combined with min_lod_clamp on {obj}: MSL clamps only a "
                    "computed level of detail",
                    "0.0",
                )
            )
        args = [obj, *compare_args]
        if gradient_args is not None:
            args += [
                self.generate_expression(gradient_args[0], is_main),
                self.generate_expression(gradient_args[1], is_main),
            ]
        function = (
            "textureCompareGrad" if gradient_args is not None else "textureCompare"
        )
        if offset is not None:
            return f"{function}OffsetClamp({', '.join(args + [offset, clamp])})"
        return f"{function}Clamp({', '.join(args + [clamp])})"

    def unsupported_texture_compare_lod_option(self, obj, option):
        if not isinstance(option, FunctionCallNode):
            return None
        option_name = self.unscoped_function_name(option.name)
        if option_name != "bias":
            return None
        return self.refuse_import(
            f"Metal depth compare LOD option {option_name} on {obj}: CrossGL depth "
            "compares have no bias operand",
            "0.0",
        )

    def texture_gather_call(self, obj, texture_expr, method_args, is_main=False):
        args, offset = self.texture_gather_arguments(texture_expr, method_args, is_main)
        function = "textureGatherOffset" if offset else "textureGather"
        return f"{function}({obj}, {', '.join(args)})"

    def texture_gather_arguments(self, texture_expr, method_args, is_main=False):
        if len(method_args) < 2:
            return [
                self.texture_gather_argument(arg, is_main) for arg in method_args
            ], False

        sampler = self.generate_expression(method_args[0], is_main)
        coords = self.generate_expression(method_args[1], is_main)
        tail = list(method_args[2:])
        constructor = self.sampled_array_coordinate_constructor(texture_expr)
        if constructor and tail:
            layer = self.sampled_array_layer_argument(tail.pop(0), is_main)
            coords = f"{constructor}({coords}, {layer})"

        args = [sampler, coords]
        has_offset = False
        if tail:
            if self.texture_gather_component_index(tail[0]) is None:
                args.append(self.generate_expression(tail.pop(0), is_main))
                has_offset = True
            if tail:
                args.append(self.texture_gather_argument(tail.pop(0), is_main))

        args.extend(self.texture_gather_argument(arg, is_main) for arg in tail)
        return args, has_offset

    def texture_gather_compare_call(self, obj, obj_expr, method_args, is_main=False):
        compare_args, consumed = self.texture_gather_compare_base_arguments(
            obj_expr, method_args, is_main
        )
        tail = list(method_args[consumed:])
        if len(tail) > 1:
            return None
        if tail:
            offset = self.generate_expression(tail[0], is_main)
            return (
                "textureGatherCompareOffset("
                f"{obj}, {', '.join(compare_args + [offset])})"
            )
        return f"textureGatherCompare({obj}, {', '.join(compare_args)})"

    def texture_gather_compare_base_arguments(
        self, obj_expr, method_args, is_main=False
    ):
        if len(method_args) < 3:
            return [self.generate_expression(arg, is_main) for arg in method_args], len(
                method_args
            )

        mapped_type = self.resource_classification_type(
            self.expression_mapped_type(obj_expr)
        )
        if (
            mapped_type in {"sampler2DArrayShadow", "samplerCubeArrayShadow"}
            and len(method_args) >= 4
        ):
            constructor = self.sampled_array_coordinate_constructor_for_type(
                mapped_type
            )
            sampler = self.generate_expression(method_args[0], is_main)
            coord = self.generate_expression(method_args[1], is_main)
            layer = self.sampled_array_layer_argument(method_args[2], is_main)
            compare = self.generate_expression(method_args[3], is_main)
            return [sampler, f"{constructor}({coord}, {layer})", compare], 4

        return [self.generate_expression(arg, is_main) for arg in method_args[:3]], 3

    def texture_gather_argument(self, arg, is_main=False):
        component = self.texture_gather_component_index(arg)
        if component is not None:
            return component
        return self.generate_expression(arg, is_main)

    def texture_gather_component_index(self, arg):
        if not isinstance(arg, VariableNode):
            return None
        name = str(getattr(arg, "name", ""))
        parts = name.split("::")
        if len(parts) < 2 or parts[-2] != "component":
            return None
        return self.texture_gather_components.get(parts[-1])

    def is_multisample_resource_type(self, mapped_type):
        return bool(mapped_type and "MS" in str(mapped_type))

    def scalar_size_resource_type(self, mapped_type):
        mapped_type = self.resource_classification_type(mapped_type)
        return mapped_type in {
            "sampler1D",
            "isampler1D",
            "usampler1D",
            "samplerBuffer",
            "isamplerBuffer",
            "usamplerBuffer",
            "image1D",
            "iimage1D",
            "uimage1D",
            "imageBuffer",
            "iimageBuffer",
            "uimageBuffer",
        }

    def array_size_component(self, mapped_type):
        return "y" if mapped_type and "1DArray" in str(mapped_type) else "z"

    def resource_size_query_info(self, expr, is_main=False):
        if not isinstance(expr, MethodCallNode):
            return None
        method = expr.method
        if method not in self.texture_size_query_methods:
            return None

        obj = self.generate_expression(expr.object, is_main)
        mapped_type = self.expression_mapped_type(expr.object)
        query_function = (
            "imageSize"
            if self.is_storage_image_expression(expr.object)
            else "textureSize"
        )
        multisample = self.is_multisample_resource_type(mapped_type)

        lod = None
        if (
            query_function == "textureSize"
            and not multisample
            and method != "get_array_size"
        ):
            lod = self.generate_expression(expr.args[0], is_main) if expr.args else "0"

        components = {
            "get_width": "" if self.scalar_size_resource_type(mapped_type) else "x",
            "get_height": "y",
            "get_depth": "z",
            "get_array_size": self.array_size_component(mapped_type),
        }
        return {
            "object": obj,
            "function": query_function,
            "component": components[method],
            "lod": lod,
            "multisample": multisample,
        }

    def resource_size_query_call(self, info, lod=None):
        args = [info["object"]]
        if info["function"] == "textureSize" and not info["multisample"]:
            args.append(lod if lod is not None else info["lod"] or "0")
        return f"{info['function']}({', '.join(args)})"

    def resource_size_method_expression(self, expr, is_main=False):
        info = self.resource_size_query_info(expr, is_main)
        if info is None:
            return None
        size_call = self.resource_size_query_call(info)
        component = info["component"]
        return f"{size_call}.{component}" if component else size_call

    def texture_size_constructor_expression(self, expr, is_main=False):
        infos = [
            self.resource_size_query_info(arg, is_main)
            for arg in getattr(expr, "args", [])
        ]
        if not infos or any(info is None for info in infos):
            return None

        first = infos[0]
        if any(
            info["object"] != first["object"]
            or info["function"] != first["function"]
            or info["multisample"] != first["multisample"]
            for info in infos
        ):
            return None

        components = [info["component"] for info in infos]
        if components not in (["x", "y"], ["x", "y", "z"]):
            return None

        lods = {info["lod"] for info in infos if info["lod"] is not None}
        if len(lods) > 1:
            return None
        lod = next(iter(lods), None)
        return self.resource_size_query_call(first, lod)

    def is_storage_image_expression(self, expr):
        mapped_type = self.expression_mapped_type(expr)
        return bool(
            mapped_type and str(mapped_type).startswith(("image", "iimage", "uimage"))
        )

    def unsupported_storage_texture_sampled_method(self, texture_expr, method, is_main):
        if not self.is_storage_image_expression(texture_expr):
            return None
        texture = self.generate_expression(texture_expr, is_main)
        fallback = (
            "0.0"
            if method in {"sample_compare", "sample_compare_level"}
            else "vec4(0.0)"
        )
        return self.refuse_import(
            f"Metal storage texture sampled method {method} on {texture}: a texture "
            "without access::sample cannot be sampled",
            fallback,
        )

    def unsupported_sampled_texture_write(self, texture_expr, method, is_main):
        texture = self.generate_expression(texture_expr, is_main)
        return self.refuse_import(
            f"Metal sampled texture write {method} on {texture}: a texture with "
            "access::sample cannot be written"
        )

    def storage_image_coordinate_expression(self, image_expr, args):
        if not args:
            return ""
        rendered_args = [self.generate_expression(arg) for arg in args]
        image_type = self.expression_mapped_type(image_expr)
        if (
            image_type in {"image1DArray", "iimage1DArray", "uimage1DArray"}
            and len(rendered_args) >= 2
        ):
            return f"uvec2({rendered_args[0]}, {rendered_args[1]})"
        if (
            image_type in {"image2DArray", "iimage2DArray", "uimage2DArray"}
            and len(rendered_args) >= 2
        ):
            return f"uvec3({rendered_args[0]}, {rendered_args[1]})"
        return rendered_args[0]

    def sampled_texture_read_expression(self, texture_expr, args, is_main=False):
        mapped_type = self.expression_mapped_type(texture_expr)
        if not self.is_sampled_texture_type(mapped_type) or not args:
            return None

        texture = self.generate_expression(texture_expr, is_main)
        cube_read = self.unsupported_sampled_cube_texture_read_expression(
            texture, mapped_type
        )
        if cube_read is not None:
            return cube_read

        rendered_args = [self.generate_expression(arg, is_main) for arg in args]
        coord, tail = self.sampled_texture_read_coordinate_and_tail(
            mapped_type, rendered_args
        )
        if coord is None:
            return None

        if self.is_multisample_resource_type(mapped_type):
            if not tail:
                return None
            return f"texelFetch({texture}, {coord}, {tail[0]})"

        if self.is_sampled_buffer_texture_type(mapped_type):
            return f"texelFetch({texture}, {coord})"

        lod = tail[0] if tail else "0"
        return f"texelFetch({texture}, {coord}, {lod})"

    def unsupported_sampled_cube_texture_read_expression(self, texture, mapped_type):
        if not self.is_sampled_cube_texture_type(mapped_type):
            return None
        fallback = self.sampled_texture_read_zero_value(mapped_type)
        return self.refuse_import(
            f"Metal sampled cube texture read on {texture}: CrossGL has no "
            "face-addressed texel fetch for cube textures",
            fallback,
        )

    def is_sampled_cube_texture_type(self, mapped_type):
        mapped_type = self.resource_classification_type(mapped_type)
        return bool(mapped_type and "Cube" in str(mapped_type))

    def sampled_texture_read_zero_value(self, mapped_type):
        mapped_type = str(self.resource_classification_type(mapped_type) or "")
        if mapped_type.startswith("usampler"):
            return "uvec4(0u)"
        if mapped_type.startswith("isampler"):
            return "ivec4(0)"
        return "vec4(0.0)"

    def is_sampled_buffer_texture_type(self, mapped_type):
        mapped_type = self.resource_classification_type(mapped_type)
        return mapped_type in {"samplerBuffer", "isamplerBuffer", "usamplerBuffer"}

    def is_sampled_texture_type(self, mapped_type):
        mapped_type = self.resource_classification_type(mapped_type)
        if mapped_type == "sampler":
            return False
        return bool(
            mapped_type
            and str(mapped_type).startswith(("sampler", "isampler", "usampler"))
        )

    def sampled_texture_read_coordinate_and_tail(self, mapped_type, rendered_args):
        if not rendered_args:
            return None, []

        coord = rendered_args[0]
        tail = rendered_args[1:]
        constructor = self.sampled_texture_array_coordinate_constructor(mapped_type)
        if constructor is not None:
            if not tail:
                return None, []
            coord = f"{constructor}({coord}, {tail[0]})"
            tail = tail[1:]

        return coord, tail

    def sampled_texture_array_coordinate_constructor(self, mapped_type):
        mapped_type = str(mapped_type)
        if "1DArray" in mapped_type:
            return "uvec2"
        if "2DArray" in mapped_type or "2DMSArray" in mapped_type:
            return "uvec3"
        if "CubeArray" in mapped_type:
            return "vec4"
        return None
