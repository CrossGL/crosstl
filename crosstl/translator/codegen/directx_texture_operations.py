"""Texture operations in the HLSL generator: gather, query, compare, sample offsets, multisample access, stores, projected coordinates and refusals of unsupported forms."""

from ..ast import LiteralNode
from ..refusals import DirectXRefusal
from ..validation import expression_debug_name
from .enum_utils import default_value_expression
from .image_access_contracts import (
    default_storage_image_channel_count,
    image_format_or_default_channel_count,
    image_load_result_kind_error,
    image_load_result_kind_mismatch,
    image_load_result_shape_error,
    image_store_value_kind_error,
    image_store_value_kind_mismatch,
    image_store_value_shape_error,
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
    is_texture_gather_compare_offset_operation,
    is_texture_gather_multi_offset_operation,
    is_texture_gather_offset_operation,
    is_texture_gather_single_offset_operation,
    is_texture_sample_basic_offset_operation,
    is_texture_sample_basic_operation,
    is_texture_sample_grad_offset_operation,
    is_texture_sample_lod_offset_operation,
    is_texture_sample_lod_operation,
    is_texture_sample_offset_operation,
    projected_texture_compare_reference,
    projected_texture_extra_argument_count_error,
    should_validate_image_load_result_shape,
    storage_image_atomic_zero_value,
    texture_compare_argument_error,
    texture_compare_extra_argument_count_error,
    texture_compare_offset_capability_error,
    texture_compare_projected_coordinate_error,
    texture_coordinate_arguments_error,
    texture_gather_capability_error,
    texture_gather_compare_extra_argument_count_error,
    texture_gather_component_count_error,
    texture_gather_component_literal_error,
    texture_gather_offset_argument_count_error,
    texture_gather_offset_capability_error,
    texture_gather_offsets_argument_count_error,
    texture_gather_operation_error,
    texture_query_lod_coordinate_swizzle,
    texture_sample_offset_capability_error,
    texture_sample_offset_extra_argument_count_error,
    texture_samples_query_expression,
    unsupported_cube_texel_fetch_expression,
    unsupported_image_atomic_expression,
    unsupported_multisample_image_atomic_expression,
    unsupported_multisample_image_store_expression,
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
    unsupported_texture_samples_query_call_expression,
)
from .storage_image_texels import (
    load_context_allowed,
    store_value_allowed,
)


class DirectXTextureOperationMixin:
    """Texture operation members of ``HLSLCodeGen``."""

    def hlsl_constructor_expression_from_rendered_args(
        self, constructor_type, args, rendered_args
    ):
        mapped_type = self.map_type(constructor_type)
        info = self.hlsl_floating_arithmetic_type_info(constructor_type)
        if info is not None and info["base_type"] in {"float", "double"}:
            rendered_args = list(rendered_args)
            for index, arg in enumerate(args):
                argument_type = self.expression_result_type(arg)
                bfloat_width = self.hlsl_bfloat16_vector_width(argument_type)
                if info["base_type"] == "float" and bfloat_width is not None:
                    # bfloat16-to-binary32 widening is exact: shift the payload.
                    rendered_args[index] = (
                        self.hlsl_bfloat16_vector_to_float_expression(
                            rendered_args[index], bfloat_width
                        )
                    )
                    if len(args) == 1 and bfloat_width == info["width"]:
                        return rendered_args[index]
                    continue
                integer_info = self.hlsl_integer_arithmetic_type_info(argument_type)
                if info["base_type"] == "float" and integer_info is not None:
                    width = integer_info["width"]
                    converted = self.hlsl_unsigned_float_conversion_expression(
                        rendered_args[index],
                        "float" + (str(width) if width > 1 else ""),
                        argument_type,
                        source_location=getattr(arg, "source_location", None),
                    )
                    if converted is not None:
                        rendered_args[index] = converted
                        if len(args) == 1 and width == info["width"]:
                            return converted
                source_info = self.hlsl_floating_arithmetic_type_info(argument_type)
                if source_info is not None and source_info["base_type"] == "float16_t":
                    rendered_args[index] = self.hlsl_native_16_bit_arithmetic_operand(
                        rendered_args[index], source_info, info["base_type"]
                    )
                    if len(args) == 1 and source_info["width"] == info["width"]:
                        return rendered_args[index]
        if info is not None and info["base_type"] == "float16_t":
            source_types = [self.expression_result_type(arg) for arg in args]
            if not all(
                self.hlsl_half_literal_is_exact(arg)
                or (source_info := self.hlsl_floating_arithmetic_type_info(source))
                and source_info["base_type"] == "float16_t"
                for source, arg in zip(source_types, args)
            ):
                width = info["width"]
                # Construct at binary32 precision before rounding, not after a
                # native narrowing cast has already discarded the low bits.
                widened = self.hlsl_constructor_expression_from_rendered_args(
                    "float" + (str(width) if width > 1 else ""), args, rendered_args
                )
                return self.hlsl_half_conversion_expression(
                    widened,
                    width,
                    source_types,
                    source_location=(
                        getattr(args[0], "source_location", None) if args else None
                    ),
                )
        component_count = self.value_component_count(constructor_type)
        if component_count and component_count > 1 and len(args) == 1:
            arg_component_count = self.expression_component_count(args[0])
            if arg_component_count == 1:
                rendered_arg = rendered_args[0]
                if not self.hlsl_expression_is_repeatable(args[0]):
                    splat = self.hlsl_scalar_splat_cast(mapped_type, rendered_arg)
                    return (
                        self.hlsl_byte_conversion_expression(splat, constructor_type)
                        or splat
                    )
                rendered_args = [rendered_arg] * component_count
            elif arg_component_count and arg_component_count > component_count:
                rendered_args = [
                    f"{rendered_args[0]}.{self.hlsl_vector_swizzle(component_count)}"
                ]
            elif arg_component_count is None:
                # HLSL has no one-argument vector constructor for a scalar ("too few elements in vector
                # initialization"); a cast splats a scalar and keeps a vector of the same width.
                splat = self.hlsl_scalar_splat_cast(mapped_type, rendered_args[0])
                return (
                    self.hlsl_byte_conversion_expression(splat, constructor_type)
                    or splat
                )
        if (
            component_count
            and component_count > 1
            and len(args) > 1
            and len(rendered_args) > 1
        ):
            arg_component_counts = [
                self.expression_component_count(arg) for arg in args
            ]
            # Only pad when every argument's width is known. Treating an
            # unknown width as 1 (the previous `or 1`) under-counts multi-
            # component members whose type is not resolved -- e.g.
            # `float3(attributes.barycentrics, 1.0)` where the DXR builtin
            # `barycentrics` is a float2 -- and appended a spurious element,
            # producing an over-wide vector that the target compiler rejects.
            if all(count for count in arg_component_counts):
                provided_component_count = sum(arg_component_counts)
                if provided_component_count < component_count:
                    component_type = self.vector_component_type(mapped_type)
                    rendered_args = [
                        *rendered_args,
                        *(
                            default_value_expression(self, component_type)
                            for _ in range(component_count - provided_component_count)
                        ),
                    ]
        rendered = f"{mapped_type}({', '.join(rendered_args)})"
        return (
            self.hlsl_byte_conversion_expression(rendered, constructor_type) or rendered
        )

    def hlsl_constructor_expression(self, constructor_type, args):
        matrix_resize = self.generate_hlsl_matrix_resize_constructor(
            constructor_type, args
        )
        if matrix_resize is not None:
            return matrix_resize
        matrix_constructor = self.hlsl_matrix_orientation.constructor(
            constructor_type,
            args,
            lambda arg: self.generate_expression_with_expected(arg, None),
        )
        if matrix_constructor is not None:
            return matrix_constructor

        info = self.hlsl_floating_arithmetic_type_info(constructor_type)
        float_constructor = info is not None and info["base_type"] == "float"
        rendered_args = []
        for arg in args:
            # Only an immediate float-vector conversion may consume a bfloat16
            # vector view: DirectX has no bfloat16 vector arithmetic lowering.
            converts_bfloat_vector = (
                float_constructor
                and self.hlsl_bfloat16_vector_width(self.expression_result_type(arg))
                is not None
            )
            if converts_bfloat_vector:
                self.hlsl_bfloat16_vector_conversion_depth = (
                    getattr(self, "hlsl_bfloat16_vector_conversion_depth", 0) + 1
                )
            try:
                rendered_args.append(self.generate_expression_with_expected(arg, None))
            finally:
                if converts_bfloat_vector:
                    self.hlsl_bfloat16_vector_conversion_depth -= 1
        return self.hlsl_constructor_expression_from_rendered_args(
            constructor_type, args, rendered_args
        )

    def image_store_channel_count(self, image_type, image_format):
        component_type = self.hlsl_storage_image_component_type(image_type)
        component_count = self.value_component_count(component_type)
        if image_format is None and component_count == 1:
            default_count = default_storage_image_channel_count(
                self.hlsl_storage_image_component_kind(image_type)
            )
            if default_count is not None:
                return default_count
        return image_format_or_default_channel_count(
            image_format,
            component_count,
        )

    def validate_image_store_value_shape(self, image_type, image_format, value_arg):
        expected_channels = self.image_store_channel_count(image_type, image_format)
        value_channels = self.expression_component_count(value_arg)
        if store_value_allowed(expected_channels, value_channels):
            return
        format_label = image_format or self.resource_base_type(image_type)
        raise DirectXRefusal(
            image_store_value_shape_error(
                "DirectX",
                format_label,
                expression_debug_name(value_arg),
                expected_channels,
                value_channels,
            )
        )

    def validate_image_store_value_type(self, image_type, image_format, value_arg):
        self.validate_image_store_value_shape(image_type, image_format, value_arg)
        expected_kind = self.image_store_expected_value_type(image_type, image_format)
        if self.hlsl_expression_contains_image_atomic(value_arg):
            previous_expected_type = self.current_expression_expected_type
            self.current_expression_expected_type = expected_kind
            try:
                self.generate_expression(value_arg)
            finally:
                self.current_expression_expected_type = previous_expected_type
        value_kind = image_store_value_kind_mismatch(
            expected_kind, self.expression_component_kind(value_arg)
        )
        if value_kind is None:
            return
        format_label = image_format or self.resource_base_type(image_type)
        raise DirectXRefusal(
            image_store_value_kind_error(
                "DirectX",
                format_label,
                expression_debug_name(value_arg),
                expected_kind,
                value_kind,
            )
        )

    def validate_image_load_result_type(self, image_type, image_format):
        expected_kind = self.expected_component_kind()
        component_kind = self.image_load_component_kind(image_type, image_format)
        format_label = image_format or self.resource_base_type(image_type)
        if not should_validate_image_load_result_shape(expected_kind, component_kind):
            return
        expected_kind = image_load_result_kind_mismatch(expected_kind, component_kind)
        if expected_kind is not None:
            raise DirectXRefusal(
                image_load_result_kind_error(
                    "DirectX", format_label, component_kind, expected_kind
                )
            )
        expected_channels = self.expected_component_count()
        loaded_channels = self.image_load_channel_count(image_type, image_format)
        if load_context_allowed(loaded_channels, expected_channels):
            return
        raise DirectXRefusal(
            image_load_result_shape_error(
                "DirectX", format_label, loaded_channels, expected_channels
            )
        )

    def three_component_image_store_constructor(self, texture_type):
        constructor = self.four_component_image_store_constructor(texture_type)
        if constructor is None:
            return None
        zero_value = self.image_store_zero_values_by_kind().get(
            self.hlsl_storage_image_component_kind(texture_type)
        )
        if zero_value is None:
            return None
        return constructor, zero_value

    def texture_query_dimension(self, texture_type):
        texture_type = self.sampled_texture_shape_type(texture_type)
        if texture_type in {"Texture1D"}:
            return 1
        if texture_type in {"Texture1DArray"}:
            return 2
        if texture_type in {
            "Texture2D",
            "TextureCube",
        }:
            return 2
        if texture_type.startswith("Texture2DMSArray<"):
            return 3
        if texture_type.startswith("Texture2DMS<"):
            return 2
        return 3

    def is_storage_image_resource_type(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return texture_type.startswith(
            (
                "RWTexture1D<",
                "RWTexture1DArray<",
                "RWTexture2D<",
                "RWTexture3D<",
                "RWTexture2DArray<",
                "RWTexture2DMS<",
                "RWTexture2DMSArray<",
                "RWTextureCube<",
                "RasterizerOrderedTexture1D<",
                "RasterizerOrderedTexture1DArray<",
                "RasterizerOrderedTexture2D<",
                "RasterizerOrderedTexture3D<",
                "RasterizerOrderedTexture2DArray<",
            )
        )

    def texture_query_helper_key(self, helper_name, texture_type):
        if not texture_type:
            return None
        return helper_name, texture_type

    def texture_query_resource_descriptor(self, texture_arg):
        texture_type = self.texture_resource_type(texture_arg)
        storage_image = self.is_storage_image_resource_type(texture_type)
        return {
            "texture_type": texture_type,
            "storage_image": storage_image,
            "multisample": (
                self.is_multisample_texture_resource_type(texture_type)
                or self.is_multisample_storage_image_resource_type(texture_type)
            ),
            "size_descriptor": (
                self.image_size_helper_descriptor(texture_type)
                if storage_image
                else self.texture_size_helper_descriptor(texture_type)
            ),
            "levels_descriptor": (
                None
                if storage_image
                else self.texture_query_levels_helper_descriptor(texture_type)
            ),
            "samples_descriptor": self.texture_samples_helper_descriptor(texture_type),
        }

    def texture_query_size_expression(self, texture_arg, lod_arg=None):
        texture_name = self.generate_expression(texture_arg)
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        texture_type = descriptor["texture_type"]
        if descriptor["storage_image"]:
            key = self.texture_query_helper_key("imageSize", texture_type)
            if key:
                self.required_texture_query_helpers.add(key)
            return f"imageSize({texture_name})"
        key = self.texture_query_helper_key("textureSize", texture_type)
        if key:
            self.required_texture_query_helpers.add(key)
        if descriptor["multisample"]:
            return f"textureSize({texture_name})"
        lod = self.generate_expression(lod_arg) if lod_arg is not None else "0"
        return f"textureSize({texture_name}, {lod})"

    def texture_query_levels_expression(self, texture_arg):
        texture_name = self.generate_expression(texture_arg)
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        texture_type = descriptor["texture_type"]
        if descriptor["storage_image"]:
            return self.unsupported_texture_query_levels_call(texture_type)
        key = self.texture_query_helper_key("textureQueryLevels", texture_type)
        if key:
            self.required_texture_query_helpers.add(key)
        return f"textureQueryLevels({texture_name})"

    def texture_samples_expression(self, texture_arg):
        texture_name = self.generate_expression(texture_arg)
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        texture_type = descriptor["texture_type"]
        if not descriptor["multisample"]:
            return unsupported_texture_samples_query_call_expression("DirectX")
        key = self.texture_query_helper_key("textureSamples", texture_type)
        if key:
            self.required_texture_query_helpers.add(key)
        return texture_samples_query_expression("DirectX", texture_name)

    def texture_sample_position_expression(self, texture_arg, sample_arg):
        texture_name = self.generate_expression(texture_arg)
        texture_type = self.texture_resource_type(texture_arg)
        if not self.is_multisample_texture_resource_type(texture_type):
            resource = texture_type or "resource"
            return self.refuse_lowering(
                f"texture sample-position query: textureSamplePosition on {resource} "
                "requires a sampled multisample texture",
                "float2(0.0, 0.0)",
            )
        sample = self.generate_expression(sample_arg)
        return f"{texture_name}.GetSamplePosition({sample})"

    def vector_component(self, expression, component):
        if all(char.isalnum() or char in "_.[]" for char in expression):
            return f"{expression}.{component}"
        return f"({expression}).{component}"

    def texture_query_lod_coordinate(self, texture_type, coord):
        texture_type = self.sampled_texture_shape_type(texture_type)
        swizzle = texture_query_lod_coordinate_swizzle("DirectX", texture_type)
        if swizzle:
            return self.vector_component(coord, swizzle)
        return coord

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
            f"{component} == 2 ? {component_calls[2]} : {component_calls[3]})"
        )

    def unsupported_texture_gather_call(self, func_name, reason):
        return unsupported_texture_gather_call_expression("DirectX", func_name, reason)

    def texture_sampling_capabilities(self, texture_type):
        texture_type = self.sampled_texture_shape_type(texture_type)
        gather_types = {
            "Texture2D",
            "Texture2DArray",
            "TextureCube",
            "TextureCubeArray",
        }
        offset_types = {"Texture2D", "Texture2DArray"}
        compare_offset_types = {
            "Texture1D",
            "Texture1DArray",
            "Texture2D",
            "Texture2DArray",
        }
        sample_offset_types = {
            "",
            "Texture1D",
            "Texture1DArray",
            "Texture2D",
            "Texture2DArray",
            "Texture3D",
        }
        return {
            "texture_type": texture_type,
            "gather": texture_type in gather_types,
            "gather_offset": texture_type in offset_types,
            "sample_offset": texture_type in sample_offset_types,
            "compare_offset": texture_type in compare_offset_types,
            "gather_compare_offset": texture_type in offset_types,
        }

    def texture_gather_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather"]

    def texture_gather_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_offset"]

    def unsupported_multisample_texture_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return unsupported_multisample_texture_call_vector_expression(
            "DirectX", func_name, texture_type
        )

    def unsupported_multisample_texture_query_lod_call(self, texture_type):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return unsupported_multisample_texture_query_lod_expression(
            "DirectX", texture_type
        )

    def unsupported_multisample_texture_compare_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return unsupported_multisample_texture_compare_scalar_expression(
            "DirectX", func_name, texture_type
        )

    def unsupported_multisample_texture_gather_compare_call(
        self, func_name, texture_type
    ):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return unsupported_multisample_texture_gather_compare_vector_expression(
            "DirectX", func_name, texture_type
        )

    def unsupported_texture_query_levels_call(self, texture_type):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return unsupported_texture_query_levels_expression("DirectX", texture_type)

    def unsupported_texture_query_lod_call(self, texture_type):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return unsupported_texture_query_lod_expression("DirectX", texture_type)

    def storage_image_texture_operation_expression(self, func_name, texture_type):
        if not self.is_storage_image_resource_type(texture_type):
            return None

        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        if is_storage_image_texture_comparison_operation(func_name):
            return unsupported_storage_image_texture_comparison_scalar_expression(
                "DirectX", func_name, texture_type
            )

        if is_storage_image_texture_operation(func_name):
            return unsupported_storage_image_texture_operation_vector_expression(
                "DirectX", func_name, texture_type
            )

        return None

    def is_cube_texture_resource_type(self, texture_type):
        texture_type = self.sampled_texture_shape_type(texture_type)
        return texture_type in {"TextureCube", "TextureCubeArray"}

    def unsupported_cube_texel_fetch_call(self, func_name, texture_type):
        texture_type = self.sampled_texture_shape_type(texture_type)
        return unsupported_cube_texel_fetch_expression(
            "DirectX", func_name, texture_type
        )

    def texture_sample_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["sample_offset"]

    def unsupported_texture_sample_offset_call(self, func_name, reason):
        return unsupported_texture_offset_call_expression("DirectX", func_name, reason)

    def generate_texture_sample_offset_call(self, func_name, args, *, call_node=None):
        parts = self.texture_call_parts(args, func_name)
        if parts is None:
            return self.unsupported_texture_sample_offset_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, sampler_name, coord, extra_args = parts
        texture_type = self.texture_resource_type(args[0])
        if not self.texture_sample_offset_supported(texture_type):
            return self.unsupported_texture_sample_offset_call(
                func_name, texture_sample_offset_capability_error("DirectX")
            )

        if is_texture_sample_basic_offset_operation(func_name):
            count_error = texture_sample_offset_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_sample_offset_call(
                    func_name, count_error
                )
            if len(extra_args) == 2:
                bias = self.generate_expression(extra_args[1])
                offset = self.hlsl_immediate_texture_offset(
                    func_name,
                    extra_args[0],
                    texture_type,
                    "SampleBias",
                    call_node=call_node,
                )
                return (
                    f"{texture_name}.SampleBias("
                    f"{sampler_name}, {coord}, {bias}, {offset})"
                )
            offset = self.hlsl_immediate_texture_offset(
                func_name,
                extra_args[0],
                texture_type,
                "Sample",
                call_node=call_node,
            )
            return f"{texture_name}.Sample({sampler_name}, {coord}, {offset})"

        if is_texture_sample_lod_offset_operation(func_name):
            count_error = texture_sample_offset_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_sample_offset_call(
                    func_name, count_error
                )
            lod = self.generate_expression(extra_args[0])
            offset = self.hlsl_immediate_texture_offset(
                func_name,
                extra_args[1],
                texture_type,
                "SampleLevel",
                call_node=call_node,
            )
            return (
                f"{texture_name}.SampleLevel({sampler_name}, {coord}, {lod}, {offset})"
            )

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
            offset = self.hlsl_immediate_texture_offset(
                func_name,
                extra_args[2],
                texture_type,
                "SampleGrad",
                call_node=call_node,
            )
            return (
                f"{texture_name}.SampleGrad("
                f"{sampler_name}, {coord}, {ddx}, {ddy}, {offset})"
            )

        return self.unsupported_texture_sample_offset_call(
            func_name, unsupported_texture_offset_operation_error()
        )

    def unsupported_texture_projected_call(self, func_name, reason):
        return unsupported_projected_texture_call_expression(
            "DirectX", func_name, reason
        )

    def projected_texture_coord(self, texture_arg, coord_arg, coord):
        texture_type = self.sampled_texture_shape_type(
            self.texture_resource_type(texture_arg)
        )
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))
        specs = {
            "Texture1D": {
                "vec2": ("x", "y"),
                "float2": ("x", "y"),
                "vec4": ("x", "w"),
                "float4": ("x", "w"),
            },
            "Texture2D": {
                "vec3": ("xy", "z"),
                "float3": ("xy", "z"),
                "vec4": ("xy", "w"),
                "float4": ("xy", "w"),
            },
            "Texture2DArray": {
                "vec4": ("xy", "w"),
                "float4": ("xy", "w"),
            },
            "Texture3D": {
                "vec4": ("xyz", "w"),
                "float4": ("xyz", "w"),
            },
            "TextureCube": {
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
        if texture_type == "Texture2DArray":
            return f"float3({projected_coord}, {self.vector_component(coord, 'z')})"
        return projected_coord

    def generate_texture_projected_call(self, func_name, args, *, call_node=None):
        parts = self.texture_call_parts(args, func_name)
        if parts is None:
            return self.unsupported_texture_projected_call(
                func_name, "requires texture and projected coordinate arguments"
            )

        texture_name, sampler_name, coord, extra_args = parts
        texture_type = self.texture_resource_type(args[0])
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
            if not extra_args:
                return f"{texture_name}.Sample({sampler_name}, {projected_coord})"
            if len(extra_args) == 1:
                bias = self.generate_expression(extra_args[0])
                return (
                    f"{texture_name}.SampleBias("
                    f"{sampler_name}, {projected_coord}, {bias})"
                )

        if is_projected_texture_basic_offset_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            if not self.texture_sample_offset_supported(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, texture_sample_offset_capability_error("DirectX")
                )
            if len(extra_args) == 1:
                offset = self.hlsl_immediate_texture_offset(
                    func_name,
                    extra_args[0],
                    texture_type,
                    "Sample",
                    call_node=call_node,
                )
                return (
                    f"{texture_name}.Sample("
                    f"{sampler_name}, {projected_coord}, {offset})"
                )
            if len(extra_args) == 2:
                bias = self.generate_expression(extra_args[1])
                offset = self.hlsl_immediate_texture_offset(
                    func_name,
                    extra_args[0],
                    texture_type,
                    "SampleBias",
                    call_node=call_node,
                )
                return (
                    f"{texture_name}.SampleBias("
                    f"{sampler_name}, {projected_coord}, {bias}, {offset})"
                )

        if is_projected_texture_lod_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            lod = self.generate_expression(extra_args[0])
            return (
                f"{texture_name}.SampleLevel({sampler_name}, {projected_coord}, {lod})"
            )

        if is_projected_texture_lod_offset_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            if not self.texture_sample_offset_supported(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, texture_sample_offset_capability_error("DirectX")
                )
            lod = self.generate_expression(extra_args[0])
            offset = self.hlsl_immediate_texture_offset(
                func_name,
                extra_args[1],
                texture_type,
                "SampleLevel",
                call_node=call_node,
            )
            return (
                f"{texture_name}.SampleLevel("
                f"{sampler_name}, {projected_coord}, {lod}, {offset})"
            )

        if is_projected_texture_grad_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            return (
                f"{texture_name}.SampleGrad("
                f"{sampler_name}, {projected_coord}, {ddx}, {ddy})"
            )

        if is_projected_texture_grad_offset_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            if not self.texture_sample_offset_supported(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, texture_sample_offset_capability_error("DirectX")
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            offset = self.hlsl_immediate_texture_offset(
                func_name,
                extra_args[2],
                texture_type,
                "SampleGrad",
                call_node=call_node,
            )
            return (
                f"{texture_name}.SampleGrad("
                f"{sampler_name}, {projected_coord}, {ddx}, {ddy}, {offset})"
            )

        return self.unsupported_texture_projected_call(
            func_name, unsupported_projected_texture_operation_error()
        )

    def generate_texture_gather_call(self, func_name, args):
        parts = self.texture_call_parts(args, func_name)
        if parts is None:
            return self.unsupported_texture_gather_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, sampler_name, coord, extra_args = parts
        texture_type = self.texture_resource_type(args[0])
        if self.is_multisample_texture_resource_type(texture_type):
            return self.unsupported_multisample_texture_call(func_name, texture_type)
        if is_texture_gather_basic_operation(
            func_name
        ) and not self.texture_gather_supported(texture_type):
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_capability_error()
            )
        if is_texture_gather_offset_operation(
            func_name
        ) and not self.texture_gather_offset_supported(texture_type):
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_offset_capability_error()
            )

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
            offset_args = [extra_args[0]]
            if len(extra_args) == 2:
                component_arg = extra_args[1]
        elif is_texture_gather_multi_offset_operation(func_name):
            offset_args, component_arg = self.texture_gather_offsets_args(extra_args)
            if offset_args is None:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_offsets_argument_count_error()
                )
        else:
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_operation_error()
            )

        method_args = [sampler_name, coord] + [
            self.generate_expression(offset_arg) for offset_arg in offset_args
        ]
        method = self.texture_gather_method(component_arg)
        if method is not None:
            return f"{texture_name}.{method}({', '.join(method_args)})"
        if self.literal_int_value(component_arg) is not None:
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_component_literal_error()
            )

        component = self.generate_expression(component_arg)
        return self.texture_gather_component_expression(
            texture_name, method_args, component
        )

    def texture_compare_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_offset"]

    def unsupported_texture_compare_call(self, func_name, reason):
        return unsupported_texture_compare_scalar_expression(
            "DirectX", func_name, reason
        )

    def texture_compare_projected_coordinate(
        self, texture_type, coord_arg, coord, compare
    ):
        """The projected coordinate and reference, both divided by ``q``, or ``None`` for an unsupported shape."""
        texture_type = self.sampled_texture_shape_type(texture_type)
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))

        if texture_type == "Texture2D":
            if coord_type in {"vec3", "float3"}:
                divisor = self.vector_component(coord, "z")
            elif coord_type in {"vec4", "float4"}:
                divisor = self.vector_component(coord, "w")
            else:
                return None
            projected_coord = f"{self.vector_component(coord, 'xy')} / {divisor}"
        elif texture_type == "TextureCube":
            if coord_type not in {"vec4", "float4"}:
                return None
            divisor = self.vector_component(coord, "w")
            projected_coord = f"{self.vector_component(coord, 'xyz')} / {divisor}"
        elif texture_type == "Texture2DArray" and coord_type in {"vec4", "float4"}:
            divisor = self.vector_component(coord, "w")
            projected_coord = (
                f"float3({self.vector_component(coord, 'xy')} / {divisor}, "
                f"{self.vector_component(coord, 'z')})"
            )
        else:
            return None
        return projected_coord, projected_texture_compare_reference(compare, divisor)

    def hlsl_sample_cmp_level(
        self, texture_name, sampler_name, coord, compare, lod_arg, offset=None
    ):
        """An explicit-LOD compare: SampleCmpLevelZero (Shader Model 4) at a literal level 0, else SampleCmpLevel
        (Shader Model 6.7)."""
        arguments = [sampler_name, coord, compare]
        if self.hlsl_is_zero_literal(lod_arg):
            method = "SampleCmpLevelZero"
        else:
            method = "SampleCmpLevel"
            arguments.append(self.generate_expression(lod_arg))
        if offset is not None:
            arguments.append(offset)
        return f"{texture_name}.{method}({', '.join(arguments)})"

    @staticmethod
    def hlsl_is_zero_literal(node):
        if not isinstance(node, LiteralNode) or isinstance(node.value, bool):
            return False
        value = node.value
        if isinstance(value, str):
            value = value.strip().rstrip("fFhHlLuU")
        try:
            return float(value) == 0.0
        except (TypeError, ValueError):
            return False

    def generate_explicit_shadow_sample_call(self, func_name, args):
        """Lower a shadow texture sampled with an explicit comparison sampler."""
        if len(args) < 3 or not self.is_shadow_sampler_type(
            self.expression_result_type(args[0])
        ):
            return None
        sampler_type = self.expression_result_type(args[1])
        if (
            self.map_type(self.resource_base_type(sampler_type))
            != "SamplerComparisonState"
        ):
            return None
        if not (
            is_texture_sample_basic_operation(func_name)
            or is_texture_sample_lod_operation(func_name)
            or is_texture_sample_offset_operation(func_name)
        ):
            return None

        coords = args[2]
        if self.expression_component_count(coords) != 3:
            return self.unsupported_texture_compare_call(
                func_name,
                "shadow texture sampling requires vec3 coordinates (uv plus comparison reference)",
            )
        if not self.hlsl_expression_is_repeatable(coords):
            return self.unsupported_texture_compare_call(
                func_name,
                "shadow coordinates with side effects must be assigned to a local variable first",
            )
        expected_count = 3 if is_texture_sample_basic_operation(func_name) else 4
        if len(args) != expected_count:
            return self.unsupported_texture_compare_call(
                func_name, "unsupported number of shadow sampling arguments"
            )

        texture_name = self.generate_expression(args[0])
        sampler_name = self.hlsl_comparison_sampler_alias_expression(
            args[1], self.generate_expression(args[1])
        )
        rendered_coords = self.generate_expression(coords)
        texture_coord = self.vector_component(rendered_coords, "xy")
        compare = self.vector_component(rendered_coords, "z")
        offset = None
        if is_texture_sample_offset_operation(func_name):
            offset = self.hlsl_immediate_texture_offset(
                func_name,
                args[-1],
                self.texture_resource_type(args[0]),
                "SampleCmp",
            )
        if is_texture_sample_lod_operation(func_name):
            return self.hlsl_sample_cmp_level(
                texture_name, sampler_name, texture_coord, compare, args[3], offset
            )
        call_args = [sampler_name, texture_coord, compare]
        if offset is not None:
            call_args.append(offset)
        return f"{texture_name}.SampleCmp({', '.join(call_args)})"

    def generate_texture_compare_call(self, func_name, args, *, call_node=None):
        parts = self.texture_call_parts(args, func_name)
        if parts is None:
            return self.unsupported_texture_compare_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, sampler_name, coord, extra_args = parts
        if not extra_args:
            return self.unsupported_texture_compare_call(
                func_name, texture_compare_argument_error()
            )

        texture_type = self.texture_resource_type(args[0])
        if self.is_multisample_texture_resource_type(texture_type):
            return self.unsupported_multisample_texture_compare_call(
                func_name, texture_type
            )

        compare = self.generate_expression(extra_args[0])
        if is_projected_texture_compare_operation(func_name):
            coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
            projected = self.texture_compare_projected_coordinate(
                texture_type, args[coord_index], coord, compare
            )
            if projected is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_projected_coordinate_error("DirectX")
                )
            projected_coord, compare = projected

            if is_texture_compare_basic_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                return (
                    f"{texture_name}.SampleCmp("
                    f"{sampler_name}, {projected_coord}, {compare})"
                )

            if is_texture_compare_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_offset_capability_error("DirectX")
                    )
                offset = self.hlsl_immediate_texture_offset(
                    func_name,
                    extra_args[1],
                    texture_type,
                    "SampleCmp",
                    call_node=call_node,
                )
                return (
                    f"{texture_name}.SampleCmp("
                    f"{sampler_name}, {projected_coord}, {compare}, {offset})"
                )

            if is_texture_compare_lod_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                return self.hlsl_sample_cmp_level(
                    texture_name, sampler_name, projected_coord, compare, extra_args[1]
                )

            if is_texture_compare_lod_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_offset_capability_error("DirectX")
                    )
                offset = self.generate_expression(extra_args[2])
                return self.hlsl_sample_cmp_level(
                    texture_name,
                    sampler_name,
                    projected_coord,
                    compare,
                    extra_args[1],
                    offset,
                )

            if is_texture_compare_grad_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                return (
                    f"{texture_name}.SampleCmpGrad("
                    f"{sampler_name}, {projected_coord}, {compare}, {ddx}, {ddy})"
                )

            if is_texture_compare_grad_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_offset_capability_error("DirectX")
                    )
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                offset = self.generate_expression(extra_args[3])
                return (
                    f"{texture_name}.SampleCmpGrad("
                    f"{sampler_name}, {projected_coord}, {compare}, {ddx}, {ddy}, {offset})"
                )

            return self.unsupported_texture_compare_call(
                func_name, unsupported_texture_compare_operation_error(projected=True)
            )

        if is_texture_compare_basic_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            return f"{texture_name}.SampleCmp({sampler_name}, {coord}, {compare})"

        if is_texture_compare_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("DirectX")
                )
            offset = self.hlsl_immediate_texture_offset(
                func_name,
                extra_args[1],
                texture_type,
                "SampleCmp",
                call_node=call_node,
            )
            return f"{texture_name}.SampleCmp({sampler_name}, {coord}, {compare}, {offset})"

        if is_texture_compare_lod_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            return self.hlsl_sample_cmp_level(
                texture_name, sampler_name, coord, compare, extra_args[1]
            )

        if is_texture_compare_lod_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("DirectX")
                )
            offset = self.generate_expression(extra_args[2])
            return self.hlsl_sample_cmp_level(
                texture_name, sampler_name, coord, compare, extra_args[1], offset
            )

        if is_texture_compare_grad_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            return (
                f"{texture_name}.SampleCmpGrad("
                f"{sampler_name}, {coord}, {compare}, {ddx}, {ddy})"
            )

        if is_texture_compare_grad_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("DirectX")
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            offset = self.generate_expression(extra_args[3])
            return (
                f"{texture_name}.SampleCmpGrad("
                f"{sampler_name}, {coord}, {compare}, {ddx}, {ddy}, {offset})"
            )

        return self.unsupported_texture_compare_call(
            func_name, unsupported_texture_compare_operation_error()
        )

    def texture_gather_compare_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_compare_offset"]

    def unsupported_texture_gather_compare_call(self, func_name, reason):
        return unsupported_texture_gather_compare_call_expression(
            "DirectX", func_name, reason
        )

    def generate_texture_gather_compare_call(self, func_name, args):
        parts = self.texture_call_parts(args, func_name)
        if parts is None:
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, sampler_name, coord, extra_args = parts
        if not extra_args:
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_compare_argument_error()
            )

        texture_type = self.texture_resource_type(args[0])
        if self.is_multisample_texture_resource_type(texture_type):
            return self.unsupported_multisample_texture_gather_compare_call(
                func_name, texture_type
            )
        if not is_texture_gather_compare_offset_operation(
            func_name
        ) and not self.texture_gather_supported(texture_type):
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_gather_capability_error()
            )

        compare = self.generate_expression(extra_args[0])
        if func_name == "textureGatherCompare":
            count_error = texture_gather_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_gather_compare_call(
                    func_name, count_error
                )
            return f"{texture_name}.GatherCmp({sampler_name}, {coord}, {compare})"

        count_error = texture_gather_compare_extra_argument_count_error(
            func_name, len(extra_args)
        )
        if count_error:
            return self.unsupported_texture_gather_compare_call(func_name, count_error)
        if not self.texture_gather_compare_offset_supported(texture_type):
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_gather_offset_capability_error()
            )
        offset = self.generate_expression(extra_args[1])
        return f"{texture_name}.GatherCmp({sampler_name}, {coord}, {compare}, {offset})"

    def generate_texture_query_helpers(self):
        if not self.required_texture_query_helpers:
            return ""

        helpers = []
        seen_helpers = set()
        for helper_name, texture_type in sorted(self.required_texture_query_helpers):
            helper = ""
            if helper_name == "textureSize":
                helper = self.generate_texture_size_helper(texture_type)
            elif helper_name == "imageSize":
                helper = self.generate_image_size_helper(texture_type)
            elif helper_name == "textureQueryLevels":
                helper = self.generate_texture_query_levels_helper(texture_type)
            elif helper_name == "textureSamples":
                helper = self.generate_texture_samples_helper(texture_type)

            if helper and helper not in seen_helpers:
                seen_helpers.add(helper)
                helpers.append(helper)

        return "".join(helpers)

    def image_atomic_helper_name(self, operation, texture_type):
        descriptor = self.image_atomic_helper_descriptor(operation, texture_type)
        return descriptor["helper_name"] if descriptor else None

    def image_atomic_helper_return_type(self, texture_type):
        descriptor = self.image_atomic_helper_descriptor("imageAtomicAdd", texture_type)
        return descriptor["return_type"] if descriptor else None

    def image_atomic_helper_coord_type(self, texture_type):
        descriptor = self.image_atomic_helper_descriptor("imageAtomicAdd", texture_type)
        return descriptor["coord_type"] if descriptor else None

    def image_atomic_intrinsic(self, operation):
        return {
            "imageAtomicAdd": "InterlockedAdd",
            "imageAtomicMin": "InterlockedMin",
            "imageAtomicMax": "InterlockedMax",
            "imageAtomicAnd": "InterlockedAnd",
            "imageAtomicOr": "InterlockedOr",
            "imageAtomicXor": "InterlockedXor",
            "imageAtomicExchange": "InterlockedExchange",
            "imageAtomicCompSwap": "InterlockedCompareExchange",
        }.get(operation)

    def image_atomic_zero_value(self, image_type):
        return storage_image_atomic_zero_value(
            self.hlsl_storage_image_component_type(image_type)
        )

    def unsupported_image_atomic_call(self, operation, image_type):
        return unsupported_image_atomic_expression(
            "DirectX",
            operation,
            self.resource_base_type(image_type),
            self.image_atomic_zero_value(image_type),
        )

    def unsupported_multisample_image_atomic_call(self, operation, image_type):
        return unsupported_multisample_image_atomic_expression(
            "DirectX",
            operation,
            self.resource_base_type(image_type),
            self.image_atomic_zero_value(image_type),
        )

    def unsupported_multisample_image_store_call(self, image_type):
        return unsupported_multisample_image_store_expression(
            "DirectX", self.resource_base_type(image_type)
        )
