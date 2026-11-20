"""Texture validation in the Metal generator: sample index, multisample and gather component arguments, call arity, texture resource kinds and coordinate parts, sampling capabilities, the 1D, offset, projected, gather, multisample and cube refusals, depth textures, shadow compares and gather compares."""

import re

from ..refusals import MetalRefusal
from ..semantics.array_layers import metal_sampled_array_layer
from ..validation import (
    IMAGE_RESOURCE_INTRINSIC_NAMES,
    expression_debug_name,
    texture_gather_component_argument_index,
    texture_sample_index_argument_index,
)
from .image_access_contracts import (
    explicit_image_format,
    image_multisample_sample_argument_index,
    image_multisample_sample_type_error,
    image_multisample_sample_type_mismatch,
    image_resource_metadata,
    is_metal_storage_image_resource,
    is_projected_texture_basic_offset_operation,
    is_projected_texture_basic_operation,
    is_projected_texture_compare_operation,
    is_projected_texture_grad_offset_operation,
    is_projected_texture_grad_operation,
    is_projected_texture_lod_offset_operation,
    is_projected_texture_lod_operation,
    is_storage_image_texture_comparison_operation,
    is_storage_image_texture_operation,
    is_texture_compare_basic_operation,
    is_texture_compare_grad_offset_operation,
    is_texture_compare_grad_operation,
    is_texture_compare_lod_offset_operation,
    is_texture_compare_lod_operation,
    is_texture_compare_offset_operation,
    is_texture_gather_basic_operation,
    is_texture_gather_multi_offset_operation,
    is_texture_gather_single_offset_operation,
    is_texture_sample_basic_offset_operation,
    is_texture_sample_grad_offset_operation,
    is_texture_sample_lod_offset_operation,
    operation_argument_type_error,
    projected_texture_compare_reference,
    projected_texture_extra_argument_count_error,
    projected_texture_offset_capability_error,
    texture_compare_argument_error,
    texture_compare_extra_argument_count_error,
    texture_compare_offset_capability_error,
    texture_compare_projected_coordinate_error,
    texture_gather_capability_error,
    texture_gather_compare_extra_argument_count_error,
    texture_gather_component_count_error,
    texture_gather_component_literal_error,
    texture_gather_offset_argument_count_error,
    texture_gather_offset_capability_error,
    texture_gather_offsets_argument_count_error,
    texture_gather_operation_error,
    texture_image_resource_operation_names,
    texture_multisample_sample_type_error,
    texture_query_lod_coordinate_swizzle,
    texture_sample_offset_capability_error,
    texture_sample_offset_extra_argument_count_error,
    unsupported_cube_texel_fetch_expression,
    unsupported_multisample_texture_call_vector_expression,
    unsupported_multisample_texture_compare_scalar_expression,
    unsupported_multisample_texture_gather_compare_vector_expression,
    unsupported_multisample_texture_query_lod_expression,
    unsupported_projected_texture_call_expression,
    unsupported_projected_texture_operation_error,
    unsupported_storage_image_texture_comparison_scalar_expression,
    unsupported_storage_image_texture_operation_vector_expression,
    unsupported_texture_compare_operation_error,
    unsupported_texture_compare_scalar_expression,
    unsupported_texture_gather_call_expression,
    unsupported_texture_gather_compare_call_expression,
    unsupported_texture_offset_call_expression,
    unsupported_texture_offset_operation_error,
    unsupported_texture_query_levels_expression,
    unsupported_texture_query_lod_expression,
    validate_texture_operation_arity,
)


class MetalTextureValidationMixin:
    """Texture validation members of ``MetalCodeGen``."""

    def validate_sample_index_argument(self, func_name, args):
        sample_index = texture_sample_index_argument_index(func_name, len(args))
        if sample_index is None:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        if not self.is_multisample_texture_resource(texture_type):
            return
        sample_type = self.texture_argument_diagnostic_type(args[sample_index])
        if sample_type is None or self.is_scalar_integer_type(sample_type):
            return
        raise MetalRefusal(
            texture_multisample_sample_type_error(
                "Metal",
                func_name,
                expression_debug_name(args[sample_index]),
                self.type_name_string(sample_type),
            )
        )

    def validate_image_multisample_arguments(self, func_name, args):
        if not args:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        sample_index = image_multisample_sample_argument_index(
            func_name,
            len(args),
            self.is_multisample_storage_image_resource(texture_type),
            "Metal",
        )
        if sample_index is None:
            return
        sample_type = self.texture_argument_diagnostic_type(args[sample_index])
        sample_type = image_multisample_sample_type_mismatch(
            sample_type, self.is_scalar_integer_type
        )
        if sample_type is None:
            return
        raise MetalRefusal(
            image_multisample_sample_type_error(
                "Metal",
                func_name,
                expression_debug_name(args[sample_index]),
                self.type_name_string(sample_type),
            )
        )

    def validate_gather_component_argument(self, func_name, args):
        component_index = texture_gather_component_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if component_index is None:
            return
        component_type = self.texture_argument_diagnostic_type(args[component_index])
        if component_type is None or self.is_scalar_integer_type(component_type):
            return
        raise MetalRefusal(
            operation_argument_type_error(
                "Metal",
                "texture gather",
                func_name,
                "a scalar integer",
                "component",
                expression_debug_name(args[component_index]),
                self.type_name_string(component_type),
            )
        )

    def validate_texture_call_arity(self, func_name, args):
        if self.is_packed_shadow_texture_sample_call(func_name, args):
            return
        validate_texture_operation_arity(
            "Metal",
            func_name,
            args,
            self.texture_resource_operation_names(),
            self.texture_call_uses_explicit_sampler,
        )

    def texture_resource_operation_names(self):
        return texture_image_resource_operation_names(IMAGE_RESOURCE_INTRINSIC_NAMES)

    def image_resource_format(self, texture_arg):
        image_format = image_resource_metadata(
            texture_arg,
            self.expression_name,
            self.current_image_format_parameters,
            self.image_variable_formats,
        )
        if image_format is not None:
            return image_format
        member = self.struct_resource_member_node(texture_arg)
        if member is None:
            return None
        return explicit_image_format(member, self.attribute_value_to_string)

    def is_array_texture_resource(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if self.is_storage_image_resource(texture_type):
            return False
        return texture_type.startswith(
            (
                "texture1d_array<",
                "texture2d_array<",
                "depth2d_array<",
                "texturecube_array<",
                "depthcube_array<",
            )
        )

    def is_multisample_texture_resource(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if self.is_storage_image_resource(texture_type):
            return False
        return texture_type.startswith(("texture2d_ms<", "texture2d_ms_array<"))

    def is_multisample_storage_image_resource(self, texture_type):
        texture_type = self.storage_image_access_agnostic_type(texture_type)
        return texture_type.startswith(("texture2d_ms<", "texture2d_ms_array<"))

    def is_storage_image_resource(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return is_metal_storage_image_resource(texture_type)

    def storage_image_access_mode(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if "access::read_write>" in texture_type:
            return "read_write"
        if "access::read>" in texture_type:
            return "read"
        if "access::write>" in texture_type:
            return "write"
        return None

    def vector_component(self, expression, component):
        if all(char.isalnum() or char in "_.[]" for char in expression):
            return f"{expression}.{component}"
        return f"({expression}).{component}"

    def array_texture_coordinate_parts(self, coord):
        coord_xy = self.vector_component(coord, "xy")
        layer = metal_sampled_array_layer(self.vector_component(coord, "z"))
        return coord_xy, layer

    def cube_array_texture_coordinate_parts(self, coord):
        coord_xyz = self.vector_component(coord, "xyz")
        layer = metal_sampled_array_layer(self.vector_component(coord, "w"))
        return coord_xyz, layer

    def texture_coordinate_parts(self, texture_type, coord):
        texture_type = self.resource_base_type(texture_type)
        if texture_type.startswith("texture1d_array<"):
            coord_x = self.vector_component(coord, "x")
            layer = metal_sampled_array_layer(self.vector_component(coord, "y"))
            return coord_x, layer
        if texture_type.startswith(("texturecube_array<", "depthcube_array<")):
            return self.cube_array_texture_coordinate_parts(coord)
        return self.array_texture_coordinate_parts(coord)

    def texture_read_coordinate_components(self, texture_type, coord):
        texture_type = self.resource_base_type(texture_type)
        if texture_type.startswith("texture1d_array<"):
            coord_x = self.vector_component(coord, "x")
            layer = f"uint({self.vector_component(coord, 'y')})"
            return coord_x, layer, 1
        if texture_type.startswith(
            ("texture2d_array<", "depth2d_array<", "texture2d_ms_array<")
        ):
            coord_xy = self.vector_component(coord, "xy")
            layer = f"uint({self.vector_component(coord, 'z')})"
            return coord_xy, layer, 2
        if texture_type.startswith("texture1d<"):
            return coord, None, 1
        if texture_type.startswith("texture3d<"):
            return coord, None, 3
        return coord, None, 2

    def texture_read_coordinate_parts(self, texture_type, coord):
        texel_coord, layer, dimensions = self.texture_read_coordinate_components(
            texture_type, coord
        )
        return self.unsigned_coordinate_expression(texel_coord, dimensions), layer

    def texture_query_lod_coordinate(self, texture_type, coord):
        texture_type = self.resource_base_type(texture_type)
        swizzle = texture_query_lod_coordinate_swizzle("Metal", texture_type)
        if swizzle:
            return self.vector_component(coord, swizzle)
        return coord

    def texture_gradient_options(self, texture_type, ddx, ddy):
        texture_type = self.resource_base_type(texture_type)
        if texture_type.startswith(
            ("texturecube<", "depthcube<", "texturecube_array<", "depthcube_array<")
        ):
            return f"gradientcube({ddx}, {ddy})"
        if texture_type.startswith("texture3d<"):
            return f"gradient3d({ddx}, {ddy})"
        return f"gradient2d({ddx}, {ddy})"

    def texture_sampling_capabilities(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        is_sampled_texture = not self.is_storage_image_resource(texture_type)
        is_texture2d = is_sampled_texture and texture_type.startswith("texture2d<")
        is_texture2d_array = is_sampled_texture and texture_type.startswith(
            "texture2d_array<"
        )
        is_texture3d = is_sampled_texture and texture_type.startswith("texture3d<")
        is_texturecube = is_sampled_texture and texture_type.startswith("texturecube<")
        is_texturecube_array = is_sampled_texture and texture_type.startswith(
            "texturecube_array<"
        )
        is_depth2d = texture_type.startswith("depth2d<")
        is_depth2d_array = texture_type.startswith("depth2d_array<")
        # MSL depth textures gather (one depth per texel) and sample with offsets like their colour
        # counterparts (MSL specification 6.12.6-6.12.9).
        is_depth = is_depth2d or is_depth2d_array
        return {
            "texture_type": texture_type,
            "gather": (
                is_texture2d
                or is_texture2d_array
                or is_texturecube
                or is_texturecube_array
                or is_depth
                or texture_type.startswith(("depthcube<", "depthcube_array<"))
            ),
            "gather_offset": is_texture2d or is_texture2d_array or is_depth,
            "sample_offset": (
                is_texture2d or is_texture2d_array or is_texture3d or is_depth
            ),
            "projected_offset": is_texture2d or is_texture2d_array or is_texture3d,
            "compare_offset": is_depth2d or is_depth2d_array,
            "gather_compare_offset": is_depth2d or is_depth2d_array,
        }

    def texture_gather_supports_offset(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_offset"]

    def texture_gather_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather"]

    def texture_sample_supports_offset(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["sample_offset"]

    def is_texture1d_sample_resource(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if self.is_storage_image_resource(texture_type):
            return False
        return texture_type.startswith(("texture1d<", "texture1d_array<"))

    def unsupported_texture1d_sampling_option_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return self.refuse_lowering(
            f"texture sampling: {func_name} on {texture_type} supports only implicit "
            "sampling",
            "float4(0.0)",
        )

    def unsupported_texture1d_texel_fetch_lod_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return self.refuse_lowering(
            f"texel fetch: {func_name} on {texture_type} requires a compile-time "
            "literal mip level",
            "float4(0.0)",
        )

    def unsupported_texture1d_size_lod_call(self, texture_type, return_type):
        texture_type = self.resource_base_type(texture_type)
        zero_value = "0" if return_type == "int" else f"{return_type}(0)"
        return self.refuse_lowering(
            f"texture size query: textureSize on {texture_type} requires a "
            "compile-time literal mip level",
            zero_value,
        )

    def metal_texture1d_read_lod_argument(self, lod_arg):
        if lod_arg is None:
            return "uint(0)"
        value = self.literal_int_value(lod_arg, self.literal_int_constants)
        if value is None:
            return None
        return f"uint({value})"

    def unsupported_texture_sample_offset_call(self, func_name, reason):
        return unsupported_texture_offset_call_expression("Metal", func_name, reason)

    def texture_sample_offset_coord_args(self, texture_type, coord):
        if self.is_array_texture_resource(texture_type):
            return self.texture_coordinate_parts(texture_type, coord)
        return (coord,)

    def generate_texture_sample_offset_call(
        self, func_name, texture_name, sampler_arg, coord, extra_args, texture_type
    ):
        if not self.texture_sample_supports_offset(texture_type):
            return self.unsupported_texture_sample_offset_call(
                func_name, texture_sample_offset_capability_error("Metal")
            )

        coord_args = self.texture_sample_offset_coord_args(texture_type, coord)

        if is_texture_sample_basic_offset_operation(func_name):
            count_error = texture_sample_offset_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_sample_offset_call(
                    func_name, count_error
                )
            offset = self.generate_expression(extra_args[0])
            args = [sampler_arg] + list(coord_args)
            if len(extra_args) == 2:
                bias = self.generate_expression(extra_args[1])
                args.append(f"bias({bias})")
            args.append(offset)
            return f"{texture_name}.sample({', '.join(args)})"

        if is_texture_sample_lod_offset_operation(func_name):
            count_error = texture_sample_offset_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_sample_offset_call(
                    func_name, count_error
                )
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            args = [sampler_arg] + list(coord_args) + [f"level({lod})", offset]
            return f"{texture_name}.sample({', '.join(args)})"

        if is_texture_sample_grad_offset_operation(func_name):
            count_error = texture_sample_offset_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_sample_offset_call(
                    func_name, count_error
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            offset = self.generate_expression(extra_args[2])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            args = [sampler_arg] + list(coord_args) + [gradient_options, offset]
            return f"{texture_name}.sample({', '.join(args)})"

        return self.unsupported_texture_sample_offset_call(
            func_name, unsupported_texture_offset_operation_error()
        )

    def unsupported_texture_projected_call(self, func_name, reason):
        return unsupported_projected_texture_call_expression("Metal", func_name, reason)

    def projected_texture_coord(self, texture_arg, coord_arg, coord):
        texture_type = self.resource_base_type(self.texture_resource_type(texture_arg))
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))
        specs = {
            "texture1d<float>": {
                "vec2": ("x", "y"),
                "float2": ("x", "y"),
                "vec4": ("x", "w"),
                "float4": ("x", "w"),
            },
            "texture2d<float>": {
                "vec3": ("xy", "z"),
                "float3": ("xy", "z"),
                "vec4": ("xy", "w"),
                "float4": ("xy", "w"),
            },
            "texture2d_array<float>": {
                "vec4": ("xy", "w"),
                "float4": ("xy", "w"),
            },
            "texture3d<float>": {
                "vec4": ("xyz", "w"),
                "float4": ("xyz", "w"),
            },
            "texturecube<float>": {
                "vec4": ("xyz", "w"),
                "float4": ("xyz", "w"),
            },
        }
        texture_specs = specs.get(texture_type)
        if texture_specs is None:
            return None
        coord_spec = texture_specs.get(coord_type)
        if coord_spec is None:
            return None
        numerator, divisor = coord_spec
        projected_coord = (
            f"{self.vector_component(coord, numerator)} / "
            f"{self.vector_component(coord, divisor)}"
        )
        if texture_type == "texture2d_array<float>":
            layer = metal_sampled_array_layer(self.vector_component(coord, "z"))
            return f"{projected_coord}, {layer}"
        return projected_coord

    def projected_texture_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["projected_offset"]

    def generate_texture_projected_call(
        self,
        func_name,
        texture_name,
        sampler_arg,
        coord,
        extra_args,
        texture_type,
        args,
    ):
        coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
        projected_coord = self.projected_texture_coord(
            args[0], args[coord_index], coord
        )
        if projected_coord is None:
            return self.unsupported_texture_projected_call(
                func_name, "requires 1D, 2D, or 3D projection coordinates"
            )

        if is_projected_texture_basic_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            if extra_args and self.is_texture1d_sample_resource(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, "bias is not supported for Metal 1D textures"
                )
            if not extra_args:
                return f"{texture_name}.sample({sampler_arg}, {projected_coord})"
            if len(extra_args) == 1:
                bias = self.generate_expression(extra_args[0])
                return (
                    f"{texture_name}.sample("
                    f"{sampler_arg}, {projected_coord}, bias({bias}))"
                )

        if is_projected_texture_basic_offset_operation(func_name):
            if not self.projected_texture_offset_supported(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, projected_texture_offset_capability_error()
                )
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            if len(extra_args) == 1:
                offset = self.generate_expression(extra_args[0])
                return (
                    f"{texture_name}.sample({sampler_arg}, {projected_coord}, {offset})"
                )
            if len(extra_args) == 2:
                offset = self.generate_expression(extra_args[0])
                bias = self.generate_expression(extra_args[1])
                return (
                    f"{texture_name}.sample("
                    f"{sampler_arg}, {projected_coord}, bias({bias}), {offset})"
                )

        if is_projected_texture_lod_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            if self.is_texture1d_sample_resource(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, "explicit LOD is not supported for Metal 1D textures"
                )
            lod = self.generate_expression(extra_args[0])
            return (
                f"{texture_name}.sample({sampler_arg}, {projected_coord}, level({lod}))"
            )

        if is_projected_texture_lod_offset_operation(func_name):
            if not self.projected_texture_offset_supported(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, projected_texture_offset_capability_error()
                )
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return (
                f"{texture_name}.sample("
                f"{sampler_arg}, {projected_coord}, level({lod}), {offset})"
            )

        if is_projected_texture_grad_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            if self.is_texture1d_sample_resource(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, "gradients are not supported for Metal 1D textures"
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            return (
                f"{texture_name}.sample("
                f"{sampler_arg}, {projected_coord}, {gradient_options})"
            )

        if is_projected_texture_grad_offset_operation(func_name):
            if not self.projected_texture_offset_supported(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, projected_texture_offset_capability_error()
                )
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            offset = self.generate_expression(extra_args[2])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            return (
                f"{texture_name}.sample("
                f"{sampler_arg}, {projected_coord}, {gradient_options}, {offset})"
            )

        return self.unsupported_texture_projected_call(
            func_name, unsupported_projected_texture_operation_error()
        )

    def is_array_expression(self, node):
        type_name = self.expression_result_type(node)
        return isinstance(type_name, str) and "[" in type_name and "]" in type_name

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

    def texture_gather_component_option(self, component_arg):
        if component_arg is None:
            return None

        components = {
            0: "component::x",
            1: "component::y",
            2: "component::z",
            3: "component::w",
        }
        return components.get(self.literal_int_value(component_arg))

    def texture_gather_coord_args(self, texture_type, coord):
        if self.is_array_texture_resource(texture_type):
            coord_part, layer = self.texture_coordinate_parts(texture_type, coord)
            return [coord_part, layer]
        return [coord]

    def texture_gather_call_expression(
        self,
        texture_name,
        sampler_arg,
        coord_args,
        offset_arg=None,
        component=None,
        default_offset_for_component=False,
    ):
        args = [sampler_arg] + coord_args
        if offset_arg is not None:
            args.append(offset_arg)
        elif component is not None and default_offset_for_component:
            args.append("int2(0)")
        if component is not None:
            args.append(component)
        return f"{texture_name}.gather({', '.join(args)})"

    def texture_gather_offsets_expression(
        self,
        texture_name,
        sampler_arg,
        coord_args,
        offset_args,
        component,
        texture_type=None,
    ):
        """Lower ``textureGatherOffsets`` to one ``gather`` per offset.

        Component ``i`` of ``textureGatherOffsets`` is texel ``T_i0_j0`` of the
        footprint at offset ``i`` (GLSL 4.60, section 8.9.4); ``gather`` returns
        the four footprint texels in the same order as GLSL, so that texel is ``w``.
        """
        component_values = []
        for offset_arg in offset_args:
            gather = self.texture_gather_call_expression(
                texture_name,
                sampler_arg,
                coord_args,
                self.generate_expression(offset_arg),
                component,
                default_offset_for_component=True,
            )
            component_values.append(f"{gather}.w")
        vector_type = self.texture_gather_result_vector_type(texture_type)
        return f"{vector_type}({', '.join(component_values)})"

    def texture_gather_result_vector_type(self, texture_type):
        match = re.search(
            r"<\s*(float|half|int|uint|short|ushort)\b", str(texture_type)
        )
        return f"{match.group(1)}4" if match else "float4"

    def texture_gather_dynamic_component_expression(
        self, build_expression, component_expr
    ):
        component_options = (
            "component::x",
            "component::y",
            "component::z",
            "component::w",
        )
        component_calls = [
            build_expression(component) for component in component_options
        ]
        return (
            f"({component_expr} == 0 ? {component_calls[0]} : "
            f"{component_expr} == 1 ? {component_calls[1]} : "
            f"{component_expr} == 2 ? {component_calls[2]} : {component_calls[3]})"
        )

    def unsupported_texture_gather_call(self, func_name, reason):
        return unsupported_texture_gather_call_expression("Metal", func_name, reason)

    def unsupported_multisample_texture_call(self, func_name, texture_type):
        return unsupported_multisample_texture_call_vector_expression(
            "Metal", func_name, texture_type
        )

    def unsupported_multisample_texture_compare_call(self, func_name, texture_type):
        return unsupported_multisample_texture_compare_scalar_expression(
            "Metal", func_name, texture_type
        )

    def unsupported_multisample_texture_gather_compare_call(
        self, func_name, texture_type
    ):
        return unsupported_multisample_texture_gather_compare_vector_expression(
            "Metal", func_name, texture_type
        )

    def unsupported_multisample_texture_query_lod_call(self, texture_type):
        return unsupported_multisample_texture_query_lod_expression(
            "Metal", texture_type
        )

    def unsupported_texture_query_levels_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_texture_query_levels_expression("Metal", texture_type)

    def unsupported_texture_query_lod_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_texture_query_lod_expression("Metal", texture_type)

    def storage_image_texture_operation_expression(self, func_name, texture_type):
        if not self.is_storage_image_resource(texture_type):
            return None

        texture_type = self.resource_base_type(texture_type)
        if is_storage_image_texture_comparison_operation(func_name):
            return unsupported_storage_image_texture_comparison_scalar_expression(
                "Metal", func_name, texture_type
            )

        if is_storage_image_texture_operation(func_name):
            return unsupported_storage_image_texture_operation_vector_expression(
                "Metal", func_name, texture_type
            )

        return None

    def is_cube_texture_resource(self, texture_type):
        return texture_type in {
            "texturecube<float>",
            "texturecube_array<float>",
            "depthcube<float>",
            "depthcube_array<float>",
        }

    def unsupported_cube_texel_fetch_call(self, func_name, texture_type):
        return unsupported_cube_texel_fetch_expression("Metal", func_name, texture_type)

    def generate_texture_gather_call(
        self, func_name, texture_name, sampler_arg, coord, extra_args, texture_type
    ):
        if self.is_multisample_texture_resource(texture_type):
            return self.unsupported_multisample_texture_call(func_name, texture_type)
        if is_texture_gather_basic_operation(
            func_name
        ) and not self.texture_gather_supported(texture_type):
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_capability_error()
            )

        coord_args = self.texture_gather_coord_args(texture_type, coord)
        supports_offset = self.texture_gather_supports_offset(texture_type)
        offset_args = []
        component_arg = None

        if is_texture_gather_basic_operation(func_name):
            if len(extra_args) > 1:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_component_count_error()
                )
            if extra_args:
                component_arg = extra_args[0]
        elif is_texture_gather_single_offset_operation(func_name):
            if len(extra_args) not in {1, 2}:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_offset_argument_count_error()
                )
            if not supports_offset:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_offset_capability_error()
                )
            offset_args = [extra_args[0]]
            if len(extra_args) == 2:
                component_arg = extra_args[1]
        elif is_texture_gather_multi_offset_operation(func_name):
            if not supports_offset:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_offset_capability_error()
                )
            offset_args, component_arg = self.texture_gather_offsets_args(extra_args)
            if offset_args is None:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_offsets_argument_count_error()
                )
        else:
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_operation_error()
            )

        component = self.texture_gather_component_option(component_arg)
        if component is not None or component_arg is None:
            if is_texture_gather_multi_offset_operation(func_name):
                return self.texture_gather_offsets_expression(
                    texture_name,
                    sampler_arg,
                    coord_args,
                    offset_args,
                    component,
                    texture_type,
                )
            offset_arg = (
                self.generate_expression(offset_args[0]) if offset_args else None
            )
            return self.texture_gather_call_expression(
                texture_name,
                sampler_arg,
                coord_args,
                offset_arg,
                component,
                default_offset_for_component=supports_offset,
            )

        if self.literal_int_value(component_arg) is not None:
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_component_literal_error()
            )

        component_expr = self.generate_expression(component_arg)
        if is_texture_gather_multi_offset_operation(func_name):
            return self.texture_gather_dynamic_component_expression(
                lambda option: self.texture_gather_offsets_expression(
                    texture_name,
                    sampler_arg,
                    coord_args,
                    offset_args,
                    option,
                    texture_type,
                ),
                component_expr,
            )

        offset_arg = self.generate_expression(offset_args[0]) if offset_args else None
        return self.texture_gather_dynamic_component_expression(
            lambda option: self.texture_gather_call_expression(
                texture_name,
                sampler_arg,
                coord_args,
                offset_arg,
                option,
                default_offset_for_component=supports_offset,
            ),
            component_expr,
        )

    def texture_compare_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_offset"]

    def unsupported_texture_compare_call(self, func_name, reason):
        return unsupported_texture_compare_scalar_expression("Metal", func_name, reason)

    def is_depth_texture_resource(self, texture_type):
        return self.resource_base_type(texture_type).startswith("depth")

    def is_packed_shadow_texture_sample_call(self, func_name, args):
        if func_name not in {
            "texture",
            "textureLod",
            "textureOffset",
            "textureLodOffset",
        }:
            return False
        if not args:
            return False
        return self.is_depth_texture_resource(
            self.texture_argument_resource_type(args[0])
        ) and not self.is_plain_depth_read(func_name, args)

    # ``depthcube_array`` is absent: CrossGL's ``texture(t, P, ref)`` on it passes the reference after a full
    # coordinate (GLSL's samplerCubeArrayShadow overload), so the coordinate size cannot tell a plain read.
    DEPTH_SAMPLE_COORDINATE_SIZES = {
        "depth2d<": 2,
        "depth2d_array<": 3,
        "depthcube<": 3,
    }
    PLAIN_DEPTH_READ_OPERATIONS = frozenset(
        {
            "texture",
            "textureLod",
            "textureOffset",
            "textureLodOffset",
            "textureGrad",
            "textureGradOffset",
            "texelFetch",
            "texelFetchOffset",
        }
    )

    def is_plain_depth_read(self, func_name, args):
        """Whether a sample or fetch reads a depth texture's value rather than comparing against it.

        A coordinate with exactly the texture's sampling dimensions carries no packed reference (GLSL's
        ``texture(sampler2DShadow, vec3)`` packs one in ``.z``), and fetches never compare. MSL returns the depth
        as a scalar where HLSL and GLSL return ``(d, 0, 0, 1)``.
        """
        if func_name not in self.PLAIN_DEPTH_READ_OPERATIONS or len(args) < 2:
            return False
        texture_type = self.resource_base_type(
            self.texture_argument_resource_type(args[0])
        )
        if not self.is_depth_texture_resource(texture_type):
            return False
        if func_name.startswith("texelFetch"):
            return True
        size = next(
            (
                count
                for prefix, count in self.DEPTH_SAMPLE_COORDINATE_SIZES.items()
                if texture_type.startswith(prefix)
            ),
            None,
        )
        coord_index = 2 if self.texture_call_uses_explicit_sampler(args) else 1
        if size is None or len(args) <= coord_index:
            return False
        if func_name.startswith("textureGrad"):
            return True
        return self.expression_component_count(args[coord_index]) == size

    def packed_shadow_compare_coord_args(self, texture_type, coord):
        texture_type = self.resource_base_type(texture_type)
        if texture_type.startswith("depth2d_array<"):
            return [
                self.vector_component(coord, "xy"),
                metal_sampled_array_layer(self.vector_component(coord, "z")),
            ], self.vector_component(coord, "w")
        if texture_type.startswith("depth2d<"):
            return [self.vector_component(coord, "xy")], self.vector_component(
                coord, "z"
            )
        if texture_type.startswith("depthcube_array<"):
            return [
                self.vector_component(coord, "xyz"),
                metal_sampled_array_layer(self.vector_component(coord, "w")),
            ], None
        if texture_type.startswith("depthcube<"):
            return [self.vector_component(coord, "xyz")], self.vector_component(
                coord, "w"
            )
        return None, None

    def generate_packed_shadow_texture_sample_call(
        self, func_name, texture_name, sampler_arg, coord, extra_args, texture_type
    ):
        if not self.is_depth_texture_resource(texture_type):
            return None

        coord_args, packed_compare = self.packed_shadow_compare_coord_args(
            texture_type, coord
        )
        if coord_args is None:
            return self.unsupported_texture_compare_call(
                func_name, "requires supported shadow texture coordinates"
            )

        is_cube_array = self.resource_base_type(texture_type).startswith(
            "depthcube_array<"
        )
        sample_options = []
        trailing_args = []

        if func_name == "texture":
            if is_cube_array:
                if len(extra_args) not in {1, 2}:
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "cube-array shadow sampling requires compare and optional bias arguments",
                    )
                compare = self.generate_expression(extra_args[0])
                if len(extra_args) == 2:
                    bias = self.generate_expression(extra_args[1])
                    sample_options.append(f"bias({bias})")
            else:
                if len(extra_args) > 1:
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "packed shadow sampling accepts at most one bias argument",
                    )
                compare = packed_compare
                if extra_args:
                    bias = self.generate_expression(extra_args[0])
                    sample_options.append(f"bias({bias})")
        elif func_name == "textureLod":
            if is_cube_array:
                if len(extra_args) != 2:
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "cube-array shadow LOD sampling requires compare and lod arguments",
                    )
                compare = self.generate_expression(extra_args[0])
                lod_arg = extra_args[1]
            else:
                if len(extra_args) != 1:
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "packed shadow LOD sampling requires one lod argument",
                    )
                compare = packed_compare
                lod_arg = extra_args[0]
            sample_options.append(f"level({self.generate_expression(lod_arg)})")
        elif func_name == "textureOffset":
            if is_cube_array or not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("Metal")
                )
            if len(extra_args) not in {1, 2}:
                return self.unsupported_texture_compare_call(
                    func_name,
                    "packed shadow offset sampling requires offset and optional bias arguments",
                )
            compare = packed_compare
            if len(extra_args) == 2:
                bias = self.generate_expression(extra_args[1])
                sample_options.append(f"bias({bias})")
            trailing_args.append(self.generate_expression(extra_args[0]))
        elif func_name == "textureLodOffset":
            if is_cube_array or not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("Metal")
                )
            if len(extra_args) != 2:
                return self.unsupported_texture_compare_call(
                    func_name,
                    "packed shadow LOD offset sampling requires lod and offset arguments",
                )
            compare = packed_compare
            sample_options.append(f"level({self.generate_expression(extra_args[0])})")
            trailing_args.append(self.generate_expression(extra_args[1]))
        else:
            return None

        args = [sampler_arg] + coord_args + [compare] + sample_options + trailing_args
        return f"{texture_name}.sample_compare({', '.join(args)})"

    def texture_compare_projected_coord_args(
        self, texture_type, coord_arg, coord, compare
    ):
        """The ``sample_compare`` coordinate arguments then the reference, divided by ``q``; else ``None``."""
        texture_type = self.resource_base_type(texture_type)
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))

        if texture_type == "depth2d<float>":
            if coord_type in {"vec3", "float3"}:
                divisor = self.vector_component(coord, "z")
            elif coord_type in {"vec4", "float4"}:
                divisor = self.vector_component(coord, "w")
            else:
                return None
            coord_args = [f"{self.vector_component(coord, 'xy')} / {divisor}"]
        elif texture_type == "depthcube<float>":
            if coord_type not in {"vec4", "float4"}:
                return None
            divisor = self.vector_component(coord, "w")
            coord_args = [f"{self.vector_component(coord, 'xyz')} / {divisor}"]
        elif texture_type == "depth2d_array<float>" and coord_type in {
            "vec4",
            "float4",
        }:
            divisor = self.vector_component(coord, "w")
            coord_args = [
                f"{self.vector_component(coord, 'xy')} / {divisor}",
                metal_sampled_array_layer(self.vector_component(coord, "z")),
            ]
        else:
            return None
        return coord_args + [projected_texture_compare_reference(compare, divisor)]

    def generate_texture_compare_call(
        self,
        func_name,
        texture_name,
        sampler_arg,
        coord,
        extra_args,
        texture_type,
        args=None,
    ):
        if not extra_args:
            return self.unsupported_texture_compare_call(
                func_name, texture_compare_argument_error()
            )

        if self.is_multisample_texture_resource(texture_type):
            return self.unsupported_multisample_texture_compare_call(
                func_name, texture_type
            )

        compare = self.generate_expression(extra_args[0])
        if is_projected_texture_compare_operation(func_name):
            coord_index = 2 if self.is_explicit_sampler_argument(args or []) else 1
            coord_arg = (args or [None, None])[coord_index]
            coord_args = self.texture_compare_projected_coord_args(
                texture_type, coord_arg, coord, compare
            )
            if coord_args is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_projected_coordinate_error("Metal")
                )
            projected_args = [sampler_arg] + coord_args

            if is_texture_compare_basic_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                return f"{texture_name}.sample_compare({', '.join(projected_args)})"

            if is_texture_compare_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_offset_capability_error("Metal")
                    )
                offset = self.generate_expression(extra_args[1])
                args = projected_args + [offset]
                return f"{texture_name}.sample_compare({', '.join(args)})"

            if is_texture_compare_lod_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                lod = self.generate_expression(extra_args[1])
                args = projected_args + [f"level({lod})"]
                return f"{texture_name}.sample_compare({', '.join(args)})"

            if is_texture_compare_lod_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_offset_capability_error("Metal")
                    )
                lod = self.generate_expression(extra_args[1])
                offset = self.generate_expression(extra_args[2])
                args = projected_args + [f"level({lod})", offset]
                return f"{texture_name}.sample_compare({', '.join(args)})"

            if is_texture_compare_grad_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
                args = projected_args + [gradient_options]
                return f"{texture_name}.sample_compare({', '.join(args)})"

            if is_texture_compare_grad_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_offset_capability_error("Metal")
                    )
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
                offset = self.generate_expression(extra_args[3])
                args = projected_args + [gradient_options, offset]
                return f"{texture_name}.sample_compare({', '.join(args)})"

            return self.unsupported_texture_compare_call(
                func_name, unsupported_texture_compare_operation_error(projected=True)
            )

        coord_args = (
            self.texture_coordinate_parts(texture_type, coord)
            if self.is_array_texture_resource(texture_type)
            else (coord,)
        )

        if is_texture_compare_basic_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            args = [sampler_arg] + list(coord_args) + [compare]
            return f"{texture_name}.sample_compare({', '.join(args)})"

        if is_texture_compare_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("Metal")
                )
            offset = self.generate_expression(extra_args[1])
            args = [sampler_arg] + list(coord_args) + [compare, offset]
            return f"{texture_name}.sample_compare({', '.join(args)})"

        if is_texture_compare_lod_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            lod = self.generate_expression(extra_args[1])
            args = [sampler_arg] + list(coord_args) + [compare, f"level({lod})"]
            return f"{texture_name}.sample_compare({', '.join(args)})"

        if is_texture_compare_lod_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("Metal")
                )
            lod = self.generate_expression(extra_args[1])
            offset = self.generate_expression(extra_args[2])
            args = (
                [sampler_arg]
                + list(coord_args)
                + [
                    compare,
                    f"level({lod})",
                    offset,
                ]
            )
            return f"{texture_name}.sample_compare({', '.join(args)})"

        if is_texture_compare_grad_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            args = [sampler_arg] + list(coord_args) + [compare, gradient_options]
            return f"{texture_name}.sample_compare({', '.join(args)})"

        if is_texture_compare_grad_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("Metal")
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            offset = self.generate_expression(extra_args[3])
            args = (
                [sampler_arg]
                + list(coord_args)
                + [
                    compare,
                    gradient_options,
                    offset,
                ]
            )
            return f"{texture_name}.sample_compare({', '.join(args)})"

        return self.unsupported_texture_compare_call(
            func_name, unsupported_texture_compare_operation_error()
        )

    def texture_gather_compare_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_compare_offset"]

    def unsupported_texture_gather_compare_call(self, func_name, reason):
        return unsupported_texture_gather_compare_call_expression(
            "Metal", func_name, reason
        )

    def generate_texture_gather_compare_call(
        self, func_name, texture_name, sampler_arg, coord, extra_args, texture_type
    ):
        if not extra_args:
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_compare_argument_error()
            )

        if self.is_multisample_texture_resource(texture_type):
            return self.unsupported_multisample_texture_gather_compare_call(
                func_name, texture_type
            )

        compare = self.generate_expression(extra_args[0])
        coord_args = self.texture_gather_coord_args(texture_type, coord)
        if func_name == "textureGatherCompare":
            count_error = texture_gather_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_gather_compare_call(
                    func_name, count_error
                )
            args = [sampler_arg] + coord_args + [compare]
            return f"{texture_name}.gather_compare({', '.join(args)})"

        count_error = texture_gather_compare_extra_argument_count_error(
            func_name, len(extra_args)
        )
        if count_error:
            return self.unsupported_texture_gather_compare_call(func_name, count_error)
        if not self.texture_gather_compare_offset_supported(texture_type):
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_compare_offset_capability_error("Metal")
            )
        offset = self.generate_expression(extra_args[1])
        args = [sampler_arg] + coord_args + [compare, offset]
        return f"{texture_name}.gather_compare({', '.join(args)})"

    def texture_query_resource_descriptor(self, texture_arg):
        texture_type = self.texture_argument_resource_type(texture_arg)
        raw_type = self.texture_argument_raw_resource_type(texture_arg)
        return {
            "texture_type": texture_type,
            "storage_image": self.is_storage_image_resource(texture_type),
            "multisample": (
                self.is_multisample_texture_resource(texture_type)
                or self.is_multisample_storage_image_resource(texture_type)
            ),
            "size_descriptor": self.texture_query_size_descriptor_for_argument(
                texture_type, raw_type
            ),
        }
