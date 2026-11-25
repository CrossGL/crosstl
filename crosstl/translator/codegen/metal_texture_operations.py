"""Texture operations in the Metal generator: size, level and sample queries, image coordinates and formats, image stores and loads with their value shapes, image and buffer atomic compare helpers, image and texture calls, expression text and precedence, and type aliases."""

import re

from ..ast import (
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import MetalRefusal
from ..validation import expression_debug_name
from .array_utils import prepend_array_dimension
from .crossgl_vector_names import crossgl_named_vector_type
from .image_access_contracts import (
    default_storage_image_channel_count,
    image_atomic_helper_descriptor_fields,
    image_atomic_helper_resource_metadata,
    image_format_component_kind,
    image_format_or_default_channel_count,
    image_load_result_kind_error,
    image_load_result_kind_mismatch,
    image_load_result_shape_error,
    image_store_value_kind_error,
    image_store_value_kind_mismatch,
    image_store_value_shape_error,
    is_metal_float_image_resource,
    is_metal_integer_image_type,
    is_projected_texture_operation,
    is_resource_samples_query_operation,
    is_resource_size_query_operation,
    is_scalar_image_format,
    is_texel_fetch_basic_operation,
    is_texel_fetch_offset_operation,
    is_texture_compare_operation,
    is_texture_gather_compare_operation,
    is_texture_gather_operation,
    is_texture_query_levels_operation,
    is_texture_query_lod_operation,
    is_texture_sample_basic_operation,
    is_texture_sample_grad_operation,
    is_texture_sample_lod_operation,
    is_texture_sample_offset_operation,
    is_texture_sample_operation,
    is_two_component_image_format,
    metal_storage_image_access_agnostic_type,
    metal_storage_image_component_type,
    numeric_component_count_from_type,
    numeric_component_kind_from_type,
    numeric_expression_component_count,
    numeric_expression_component_kind,
    resource_query_method_size_descriptor,
    should_validate_image_load_result_shape,
    storage_image_atomic_zero_value,
    storage_image_format_store_constructor,
    storage_image_store_constructors,
    storage_image_two_component_store_expression,
    storage_image_zero_values,
    texture_query_levels_multisample_expression,
    texture_samples_query_expression,
    unsupported_image_atomic_expression,
    unsupported_multisample_image_atomic_expression,
    unsupported_multisample_image_store_expression,
    unsupported_multisample_texel_fetch_offset_expression,
    unsupported_texture_samples_query_call_expression,
)
from .metal_buffer_coherence import memory_helper_token
from .metal_compare_exchange import compare_exchange_loop
from .metal_texel_buffers import texel_buffer_component_type
from .storage_image_texels import (
    leading_components_suffix,
    load_context_allowed,
    store_value_allowed,
    texel_store_value,
    texel_vector_type,
)
from .texture_call_names import names_sampled_texture_operation


class MetalTextureOperationMixin:
    """Texture operation members of ``MetalCodeGen``."""

    def texture_query_size_expression(self, texture_arg, lod_arg=None):
        texture_name = self.generate_expression(texture_arg)
        query_descriptor = self.texture_query_resource_descriptor(texture_arg)
        texture_type = query_descriptor["texture_type"]
        size_descriptor = query_descriptor["size_descriptor"]
        if size_descriptor is None:
            # MSL has no free textureSize function: emitting the call verbatim would not compile.
            return self.refuse_lowering(
                f"texture size query: textureSize on {texture_type or texture_name} "
                "has no Metal size query",
                "int2(0)",
            )

        if self.is_texture1d_sample_resource(texture_type):
            lod_arg_string = self.metal_texture1d_read_lod_argument(lod_arg)
            if lod_arg_string is None:
                return self.unsupported_texture1d_size_lod_call(
                    texture_type, size_descriptor["return_type"]
                )
        else:
            lod = self.generate_expression(lod_arg) if lod_arg is not None else "0"
            lod_arg_string = f"uint({lod})"
        return self.texture_query_size_descriptor_expression(
            texture_name, size_descriptor, lod_arg_string
        )

    def texture_query_size_descriptor_expression(
        self, texture_name, descriptor, lod_arg_string
    ):
        dimension_expressions = [
            self.texture_query_size_dimension_expression(
                texture_name, method, use_lod, lod_arg_string
            )
            for method, use_lod in descriptor["dimensions"]
        ]
        if descriptor["return_type"] == "int":
            return f"int({dimension_expressions[0]})"
        return f"{descriptor['return_type']}({', '.join(dimension_expressions)})"

    def texture_query_size_dimension_expression(
        self, texture_name, method, use_lod, lod_arg_string
    ):
        args = lod_arg_string if use_lod else ""
        return f"{texture_name}.{method}({args})"

    def texture_query_size_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if self.is_storage_image_resource(texture_type):
            return self.storage_image_size_descriptor(texture_type)
        return self.sampled_texture_size_descriptor(texture_type)

    def texture_query_size_descriptor_for_argument(self, texture_type, raw_type=None):
        raw_base_type = self.resource_base_type(raw_type)
        if raw_base_type == "imageCube":
            return resource_query_method_size_descriptor(
                "int2", (("get_width", False), ("get_height", False))
            )
        return self.texture_query_size_descriptor(texture_type)

    def storage_image_size_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        make_descriptor = resource_query_method_size_descriptor
        if texture_type.startswith("texture1d_array<"):
            return make_descriptor(
                "int2", (("get_width", False), ("get_array_size", False))
            )
        if texture_type.startswith("texture1d<"):
            return make_descriptor("int", (("get_width", False),))
        if texture_type.startswith("texture2d_array<"):
            return make_descriptor(
                "int3",
                (
                    ("get_width", False),
                    ("get_height", False),
                    ("get_array_size", False),
                ),
            )
        if texture_type.startswith("texture2d_ms_array<"):
            return make_descriptor(
                "int3",
                (
                    ("get_width", False),
                    ("get_height", False),
                    ("get_array_size", False),
                ),
            )
        if texture_type.startswith("texture2d_ms<"):
            return make_descriptor(
                "int2", (("get_width", False), ("get_height", False))
            )
        if texture_type.startswith("texture3d<"):
            return make_descriptor(
                "int3",
                (
                    ("get_width", False),
                    ("get_height", False),
                    ("get_depth", False),
                ),
            )
        return make_descriptor("int2", (("get_width", False), ("get_height", False)))

    def sampled_texture_size_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        # The size query depends on the image shape alone, so an integer or half texture
        # (texture2d<uint>, texturecube<half>) has the descriptor of its float form.
        texture_type = re.sub(
            r"^(\w+)<(?:int|uint|half|short|ushort|bfloat)>$",
            r"\1<float>",
            texture_type,
        )
        make_descriptor = resource_query_method_size_descriptor
        descriptors = {
            "texture1d<float>": make_descriptor("int", (("get_width", True),)),
            "texture1d_array<float>": make_descriptor(
                "int2", (("get_width", True), ("get_array_size", False))
            ),
            "texture2d<float>": make_descriptor(
                "int2", (("get_width", True), ("get_height", True))
            ),
            "depth2d<float>": make_descriptor(
                "int2", (("get_width", True), ("get_height", True))
            ),
            "texturecube<float>": make_descriptor(
                "int2", (("get_width", True), ("get_height", True))
            ),
            "depthcube<float>": make_descriptor(
                "int2", (("get_width", True), ("get_height", True))
            ),
            "texture2d_array<float>": make_descriptor(
                "int3",
                (
                    ("get_width", True),
                    ("get_height", True),
                    ("get_array_size", False),
                ),
            ),
            "depth2d_array<float>": make_descriptor(
                "int3",
                (
                    ("get_width", True),
                    ("get_height", True),
                    ("get_array_size", False),
                ),
            ),
            "texturecube_array<float>": make_descriptor(
                "int3",
                (
                    ("get_width", True),
                    ("get_height", True),
                    ("get_array_size", False),
                ),
            ),
            "depthcube_array<float>": make_descriptor(
                "int3",
                (
                    ("get_width", True),
                    ("get_height", True),
                    ("get_array_size", False),
                ),
            ),
            "texture3d<float>": make_descriptor(
                "int3",
                (
                    ("get_width", True),
                    ("get_height", True),
                    ("get_depth", True),
                ),
            ),
            "texture2d_ms<float>": make_descriptor(
                "int2", (("get_width", False), ("get_height", False))
            ),
            "texture2d_ms_array<float>": make_descriptor(
                "int3",
                (
                    ("get_width", False),
                    ("get_height", False),
                    ("get_array_size", False),
                ),
            ),
        }
        return descriptors.get(texture_type)

    def texture_query_levels_expression(self, texture_arg):
        texture_name = self.generate_expression(texture_arg)
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        texture_type = descriptor["texture_type"]
        if descriptor["storage_image"]:
            return self.unsupported_texture_query_levels_call(texture_type)
        if descriptor["multisample"]:
            return texture_query_levels_multisample_expression()
        return f"int({texture_name}.get_num_mip_levels())"

    def texture_samples_expression(self, texture_arg):
        texture_name = self.generate_expression(texture_arg)
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        if not descriptor["multisample"]:
            return unsupported_texture_samples_query_call_expression("Metal")
        return texture_samples_query_expression("Metal", texture_name)

    def image_coordinate_expression(self, image_type, coord):
        image_type = self.storage_image_access_agnostic_type(image_type)
        if image_type in {
            "texture1d_array<float, access::read_write>",
            "texture1d_array<int, access::read_write>",
            "texture1d_array<uint, access::read_write>",
        }:
            coord_x = self.unsigned_coordinate_expression(
                self.vector_component(coord, "x"), 1
            )
            layer = f"uint({self.vector_component(coord, 'y')})"
            return coord_x, layer
        if image_type in {
            "texture1d<float, access::read_write>",
            "texture1d<int, access::read_write>",
            "texture1d<uint, access::read_write>",
        }:
            return self.unsigned_coordinate_expression(coord, 1), None
        if image_type in {
            "texture2d_array<float, access::read_write>",
            "texture2d_array<int, access::read_write>",
            "texture2d_array<uint, access::read_write>",
            "texture2d_ms_array<float, access::read_write>",
            "texture2d_ms_array<int, access::read_write>",
            "texture2d_ms_array<uint, access::read_write>",
        }:
            coord_xy = self.unsigned_coordinate_expression(
                self.vector_component(coord, "xy"), 2
            )
            layer = f"uint({self.vector_component(coord, 'z')})"
            return coord_xy, layer
        if image_type in {
            "texture2d_ms<float, access::read_write>",
            "texture2d_ms<int, access::read_write>",
            "texture2d_ms<uint, access::read_write>",
        }:
            return self.unsigned_coordinate_expression(coord, 2), None
        if image_type in {
            "texture3d<float, access::read_write>",
            "texture3d<int, access::read_write>",
            "texture3d<uint, access::read_write>",
        }:
            return self.unsigned_coordinate_expression(coord, 3), None
        return self.unsigned_coordinate_expression(coord, 2), None

    def unsigned_coordinate_expression(self, coord, dimensions):
        constructor = "uint" if dimensions == 1 else f"uint{dimensions}"
        coord_text = str(coord).strip()
        if coord_text.startswith(f"{constructor}("):
            return coord_text
        return f"{constructor}({coord_text})"

    def storage_image_access_agnostic_type(self, image_type):
        image_type = self.resource_base_type(image_type)
        return metal_storage_image_access_agnostic_type(image_type)

    def is_integer_image_type(self, image_type):
        image_type = self.storage_image_access_agnostic_type(image_type)
        return is_metal_integer_image_type(image_type)

    def is_scalar_image_format(self, image_format):
        return is_scalar_image_format(image_format)

    def is_two_component_image_format(self, image_format):
        return is_two_component_image_format(image_format)

    def is_scalar_integer_image_resource(self, image_type, image_format):
        if image_format is not None:
            return self.is_scalar_image_format(image_format)
        return self.is_integer_image_type(image_type)

    def is_float_image_resource(self, image_type):
        image_type = self.storage_image_access_agnostic_type(image_type)
        return is_metal_float_image_resource(image_type)

    def image_store_constructors_by_kind(self):
        return storage_image_store_constructors("float4", "int4", "uint4")

    def image_store_zero_values_by_kind(self):
        return storage_image_zero_values()

    def image_load_component_suffix(self, image_type, image_format):
        if self.suppress_image_load_component_suffix:
            return ""
        # ``read`` yields four components; a narrower context takes the leading ones.
        return leading_components_suffix(self.expected_component_count())

    def image_format_store_constructor(self, image_format):
        return storage_image_format_store_constructor(
            image_format, self.image_store_constructors_by_kind()
        )

    def integer_image_store_constructor(self, image_type):
        image_type = self.storage_image_access_agnostic_type(image_type)
        if image_type in {
            "texture1d<int, access::read_write>",
            "texture1d_array<int, access::read_write>",
            "texture2d<int, access::read_write>",
            "texture3d<int, access::read_write>",
            "texture2d_array<int, access::read_write>",
            "texture2d_ms<int, access::read_write>",
            "texture2d_ms_array<int, access::read_write>",
        }:
            return "int4"
        if image_type in {
            "texture1d<uint, access::read_write>",
            "texture1d_array<uint, access::read_write>",
            "texture2d<uint, access::read_write>",
            "texture3d<uint, access::read_write>",
            "texture2d_array<uint, access::read_write>",
            "texture2d_ms<uint, access::read_write>",
            "texture2d_ms_array<uint, access::read_write>",
        }:
            return "uint4"
        return None

    def two_component_image_store_expression(
        self, image_format, value, value_type=None
    ):
        return storage_image_two_component_store_expression(
            image_format,
            value,
            self.is_scalar_value_type(value_type),
            self.image_store_constructors_by_kind(),
            self.image_store_zero_values_by_kind(),
        )

    def image_store_value_expression(
        self, image_type, image_format, value, value_type=None
    ):
        # ``write`` takes four components and ignores those the format lacks.
        component_kind = metal_storage_image_component_type(
            self.storage_image_access_agnostic_type(image_type)
        ) or image_format_component_kind(image_format)
        constructor = self.image_store_constructors_by_kind().get(component_kind)
        if constructor is None:
            return value
        channels = self.value_component_count(value_type)
        value = self.image_store_value_with_texel_elements(
            value, value_type, image_type, channels
        )
        return texel_store_value(constructor, component_kind, value, channels)

    METAL_TEXEL_ELEMENT_TYPES = frozenset(
        {"float", "half", "int", "uint", "short", "ushort"}
    )

    def image_store_value_with_texel_elements(
        self, value, value_type, image_type, channels
    ):
        """``value`` with the element type of the texel ``write`` takes.

        ``texture<T>::write`` takes ``vec<T, 4>`` of the texture's own ``T``
        (``half4`` for ``texture2d<half>``) and MSL has no implicit conversion
        between vector types, while GLSL converts a ``f16vec4`` argument to
        ``vec4`` and HLSL converts on store. A vector of another element type
        is converted explicitly before it is padded; a scalar converts inside
        the splatting constructor.
        """
        if channels is None or channels < 2 or not value_type:
            return value
        agnostic = self.storage_image_access_agnostic_type(image_type)
        if "<" not in str(agnostic):
            return value
        texel = str(agnostic).split("<", 1)[1].split(",", 1)[0].split(">", 1)[0]
        texel = texel.strip()
        if texel not in self.METAL_TEXEL_ELEMENT_TYPES:
            return value
        element = self.vector_component_type(value_type)
        if element is None or element == texel:
            return value
        return f"{texel}{channels}({value})"

    def image_store_expected_value_type(self, image_type, image_format, value_arg=None):
        if value_arg is not None and self.is_vector_value_type(
            self.expression_result_type(value_arg)
        ):
            return None
        if image_format:
            return image_format_component_kind(image_format)
        component_type = metal_storage_image_component_type(image_type)
        if component_type in {"float", "int", "uint"}:
            return component_type
        return None

    def image_load_result_type(self, image_arg):
        # imageLoad yields four components whatever the format, as ``read`` does.
        image_format = self.image_resource_format(image_arg)
        if image_format:
            component_kind = image_format_component_kind(image_format)
        else:
            image_type = self.texture_argument_resource_type(image_arg)
            component_kind = metal_storage_image_component_type(
                image_type
            ) or texel_buffer_component_type(image_type)
        return texel_vector_type(
            component_kind, {"float": "float", "int": "int", "uint": "uint"}
        )

    def image_load_component_kind(self, image_type, image_format):
        if image_format:
            return image_format_component_kind(image_format)
        component_type = metal_storage_image_component_type(image_type)
        if component_type in {"float", "int", "uint"}:
            return component_type
        return None

    def image_load_channel_count(self, image_type, image_format):
        return image_format_or_default_channel_count(
            image_format,
            default_storage_image_channel_count(
                metal_storage_image_component_type(image_type)
            ),
        )

    def expected_component_kind(self):
        return numeric_component_kind_from_type(
            self.current_expression_expected_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
        )

    def expected_component_count(self):
        return self.value_component_count(self.current_expression_expected_type)

    def expression_component_kind(self, expr):
        return numeric_expression_component_kind(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
        )

    def value_component_count(self, vtype):
        return numeric_component_count_from_type(
            vtype,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
            scalar_types={
                "float",
                "half",
                "double",
                "int",
                "uint",
                "long",
                "ulong",
                "int64_t",
                "uint64_t",
                "bool",
            },
            excluded_type_markers=("x",),
        )

    def expression_component_count(self, expr):
        return numeric_expression_component_count(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
            scalar_types={
                "float",
                "half",
                "double",
                "int",
                "uint",
                "long",
                "ulong",
                "int64_t",
                "uint64_t",
                "bool",
            },
            excluded_type_markers=("x",),
        )

    def image_store_channel_count(self, image_type, image_format):
        return image_format_or_default_channel_count(
            image_format,
            default_storage_image_channel_count(
                metal_storage_image_component_type(image_type)
            ),
        )

    def validate_image_store_value_shape(self, image_type, image_format, value_arg):
        expected_channels = self.image_store_channel_count(image_type, image_format)
        value_channels = self.expression_component_count(value_arg)
        if store_value_allowed(expected_channels, value_channels):
            return
        format_label = image_format or self.resource_base_type(image_type)
        raise MetalRefusal(
            image_store_value_shape_error(
                "Metal",
                format_label,
                expression_debug_name(value_arg),
                expected_channels,
                value_channels,
            )
        )

    def validate_image_store_value_type(self, image_type, image_format, value_arg):
        self.validate_image_store_value_shape(image_type, image_format, value_arg)
        expected_kind = self.image_store_expected_value_type(image_type, image_format)
        value_kind = image_store_value_kind_mismatch(
            expected_kind, self.expression_component_kind(value_arg)
        )
        if value_kind is None:
            return
        format_label = image_format or self.resource_base_type(image_type)
        raise MetalRefusal(
            image_store_value_kind_error(
                "Metal",
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
            raise MetalRefusal(
                image_load_result_kind_error(
                    "Metal", format_label, component_kind, expected_kind
                )
            )
        expected_channels = self.expected_component_count()
        loaded_channels = self.image_load_channel_count(image_type, image_format)
        if load_context_allowed(loaded_channels, expected_channels):
            return
        raise MetalRefusal(
            image_load_result_shape_error(
                "Metal", format_label, loaded_channels, expected_channels
            )
        )

    def image_atomic_method(self, func_name):
        return {
            "imageAtomicAdd": "atomic_fetch_add",
            "imageAtomicMin": "atomic_fetch_min",
            "imageAtomicMax": "atomic_fetch_max",
            "imageAtomicAnd": "atomic_fetch_and",
            "imageAtomicOr": "atomic_fetch_or",
            "imageAtomicXor": "atomic_fetch_xor",
            "imageAtomicExchange": "atomic_exchange",
        }.get(func_name)

    def unsupported_multisample_image_atomic_call(self, func_name, image_type):
        resource_type = self.resource_base_type(image_type)
        agnostic_type = self.storage_image_access_agnostic_type(image_type)
        component_type = metal_storage_image_component_type(agnostic_type)
        return unsupported_multisample_image_atomic_expression(
            "Metal",
            func_name,
            resource_type,
            storage_image_atomic_zero_value(component_type),
        )

    def unsupported_multisample_image_store_call(self, image_type):
        return unsupported_multisample_image_store_expression(
            "Metal", self.resource_base_type(image_type)
        )

    def unsupported_image_atomic_call(self, func_name, image_type):
        image_type = self.storage_image_access_agnostic_type(image_type)
        component_type = metal_storage_image_component_type(image_type)
        return unsupported_image_atomic_expression(
            "Metal",
            func_name,
            self.resource_base_type(image_type),
            storage_image_atomic_zero_value(component_type),
        )

    def image_atomic_compare_descriptor(self, texture_type):
        texture_type = self.storage_image_access_agnostic_type(texture_type)
        component_type = metal_storage_image_component_type(texture_type)
        if "<" not in texture_type:
            return None

        texture_family = texture_type.split("<", 1)[0]
        metadata = image_atomic_helper_resource_metadata(
            texture_family,
            {
                "texture1d": "image1D",
                "texture1d_array": "image1DArray",
                "texture2d": "image2D",
                "texture3d": "image3D",
                "texture2d_array": "image2DArray",
            },
            {
                "texture1d": "int",
                "texture1d_array": "int2",
                "texture2d": "int2",
                "texture3d": "int3",
                "texture2d_array": "int3",
            },
            extra_fields_by_family={
                "texture1d": {
                    "exchange_expr": (
                        "image.atomic_compare_exchange_weak(uint(coord), &original, value)"
                    )
                },
                "texture1d_array": {
                    "exchange_expr": (
                        "image.atomic_compare_exchange_weak(uint(coord.x), uint(coord.y), &original, value)"
                    )
                },
                "texture2d": {
                    "exchange_expr": (
                        "image.atomic_compare_exchange_weak(uint2(coord), &original, value)"
                    )
                },
                "texture3d": {
                    "exchange_expr": (
                        "image.atomic_compare_exchange_weak(uint3(coord), &original, value)"
                    )
                },
                "texture2d_array": {
                    "exchange_expr": (
                        "image.atomic_compare_exchange_weak(uint2(coord.xy), uint(coord.z), &original, value)"
                    )
                },
            },
        )
        if metadata is None:
            return None

        descriptor = image_atomic_helper_descriptor_fields(
            "imageAtomicCompSwap",
            component_type,
            metadata["suffix_family"],
            metadata["coord_type"],
        )
        if descriptor is None:
            return None
        descriptor.update(
            {
                "vector_type": f"{component_type}4",
                "exchange_expr": metadata["exchange_expr"],
            }
        )
        return descriptor

    def image_atomic_compare_helper_name(self, texture_type):
        descriptor = self.image_atomic_compare_descriptor(texture_type)
        return descriptor["helper_name"] if descriptor else None

    def image_atomic_compare_return_type(self, texture_type):
        descriptor = self.image_atomic_compare_descriptor(texture_type)
        return descriptor["return_type"] if descriptor else None

    def image_atomic_compare_vector_type(self, texture_type):
        descriptor = self.image_atomic_compare_descriptor(texture_type)
        return descriptor["vector_type"] if descriptor else None

    def image_atomic_compare_coord_type(self, texture_type):
        descriptor = self.image_atomic_compare_descriptor(texture_type)
        return descriptor["coord_type"] if descriptor else None

    def image_atomic_compare_exchange_expression(self, texture_type):
        descriptor = self.image_atomic_compare_descriptor(texture_type)
        return descriptor["exchange_expr"] if descriptor else None

    def generate_image_atomic_compare_helpers(self):
        if not self.required_image_atomic_compare_helpers:
            return ""

        helpers = []
        for texture_type in sorted(self.required_image_atomic_compare_helpers):
            helper_name = self.image_atomic_compare_helper_name(texture_type)
            return_type = self.image_atomic_compare_return_type(texture_type)
            vector_type = self.image_atomic_compare_vector_type(texture_type)
            coord_type = self.image_atomic_compare_coord_type(texture_type)
            exchange_expr = self.image_atomic_compare_exchange_expression(texture_type)
            if (
                not helper_name
                or not return_type
                or not vector_type
                or not coord_type
                or not exchange_expr
            ):
                continue
            helpers.append(
                f"static {return_type} {helper_name}({texture_type} image, {coord_type} coord, {return_type} compareValue, {return_type} value) {{\n"
                f"    {vector_type} original;\n"
                "    do {\n"
                "        original.x = compareValue;\n"
                f"    }} while (!{exchange_expr} && original.x == compareValue);\n"
                "    return original.x;\n"
                "}\n\n"
            )
        return "".join(helpers)

    def buffer_atomic_compare_helper_name(self, component_type, memory=""):
        return (
            "__crossgl_buffer_atomic_compare_exchange_"
            f"{memory_helper_token(memory)}{component_type}"
        )

    def generate_buffer_atomic_compare_helpers(self):
        if not self.required_buffer_atomic_compare_helpers:
            return ""

        helpers = []
        for component_type, memory in sorted(
            self.required_buffer_atomic_compare_helpers
        ):
            value_type = self.map_type(component_type)
            helper_name = self.buffer_atomic_compare_helper_name(component_type, memory)
            helpers.append(
                f"static {value_type} {helper_name}({memory}device uchar* buffer, uint offset, {value_type} compareValue, {value_type} value) {{\n"
                + compare_exchange_loop(
                    f"{memory}device atomic_",
                    "buffer + offset",
                    component_type,
                    name="target",
                )
                + "}\n\n"
            )
        return "".join(helpers)

    def generate_image_call(self, func_name, args):
        if func_name == "imageAtomicCompSwap" and len(args) >= 4:
            image_type = self.texture_resource_type(args[0])
            if self.is_multisample_storage_image_resource(image_type):
                return self.unsupported_multisample_image_atomic_call(
                    func_name, image_type
                )
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            compare = self.generate_expression(args[2])
            value = self.generate_expression(args[3])
            helper_name = self.image_atomic_compare_helper_name(image_type)
            if not helper_name:
                return self.unsupported_image_atomic_call(func_name, image_type)
            self.required_image_atomic_compare_helpers.add(image_type)
            # The helper takes signed coordinates; Metal converts no vector implicitly.
            coord_type = self.image_atomic_compare_coord_type(image_type)
            coord_value_type = self.expression_result_type(args[1])
            if coord_value_type and self.map_type(str(coord_value_type)) != coord_type:
                coord = f"{coord_type}({coord})"
            return f"{helper_name}({image_name}, {coord}, {compare}, {value})"

        atomic_method = self.image_atomic_method(func_name)
        if atomic_method and len(args) >= 3:
            image_type = self.texture_resource_type(args[0])
            if self.is_multisample_storage_image_resource(image_type):
                return self.unsupported_multisample_image_atomic_call(
                    func_name, image_type
                )
            if metal_storage_image_component_type(image_type) == "float":
                return self.unsupported_image_atomic_call(func_name, image_type)
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            value = self.generate_expression(args[2])
            texel_coord, layer = self.image_coordinate_expression(image_type, coord)
            if layer is not None:
                return (
                    f"{image_name}.{atomic_method}({texel_coord}, {layer}, {value}).x"
                )
            return f"{image_name}.{atomic_method}({texel_coord}, {value}).x"

        if func_name == "imageLoad" and len(args) >= 2:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            image_type = self.texture_resource_type(args[0])
            texel_coord, layer = self.image_coordinate_expression(image_type, coord)
            if len(args) >= 3 and self.is_multisample_storage_image_resource(
                image_type
            ):
                sample = f"uint({self.generate_expression(args[2])})"
                if layer is not None:
                    load_expr = f"{image_name}.read({texel_coord}, {layer}, {sample})"
                else:
                    load_expr = f"{image_name}.read({texel_coord}, {sample})"
            elif layer is not None:
                load_expr = f"{image_name}.read({texel_coord}, {layer})"
            else:
                load_expr = f"{image_name}.read({texel_coord})"
            image_format = self.image_resource_format(args[0])
            self.validate_image_load_result_type(image_type, image_format)
            return f"{load_expr}{self.image_load_component_suffix(image_type, image_format)}"

        if func_name == "imageStore" and len(args) >= 3:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            image_type = self.texture_resource_type(args[0])
            if len(args) >= 4 and self.is_multisample_storage_image_resource(
                image_type
            ):
                sample = f"uint({self.generate_expression(args[2])})"
                value_arg = args[3]
            else:
                sample = None
                value_arg = args[2]
            image_format = self.image_resource_format(args[0])
            self.validate_image_store_value_type(image_type, image_format, value_arg)
            value = self.generate_expression_with_expected(
                value_arg,
                self.image_store_expected_value_type(
                    image_type, image_format, value_arg
                ),
            )
            value = self.image_store_value_expression(
                image_type, image_format, value, self.expression_result_type(value_arg)
            )
            if self.is_multisample_storage_image_resource(image_type):
                return self.unsupported_multisample_image_store_call(image_type)
            texel_coord, layer = self.image_coordinate_expression(image_type, coord)
            if sample is not None and layer is not None:
                store = f"{image_name}.write({value}, {texel_coord}, {layer}, {sample})"
            elif sample is not None:
                store = f"{image_name}.write({value}, {texel_coord}, {sample})"
            elif layer is not None:
                store = f"{image_name}.write({value}, {texel_coord}, {layer})"
            else:
                store = f"{image_name}.write({value}, {texel_coord})"
            return store

        return None

    def generate_texture_call(self, func_name, args):
        expression = self.generate_texture_call_value(func_name, args)
        if expression is not None and self.is_plain_depth_read(func_name, args):
            # An MSL depth read is a scalar; HLSL and GLSL read a depth texel as (d, 0, 0, 1).
            return f"float4({expression}, 0.0, 0.0, 1.0)"
        return expression

    def generate_texture_call_value(self, func_name, args):
        outermost = self.explicit_sampler_answers is None
        if outermost:
            self.explicit_sampler_answers = {}
        try:
            return self.attempt_texture_call_value(func_name, args)
        finally:
            if outermost:
                self.explicit_sampler_answers = None

    def attempt_texture_call_value(self, func_name, args):
        if not func_name:
            return None
        texel_buffer_call = self.metal_texel_buffers.generate_call(func_name, args)
        if texel_buffer_call is not None:
            return texel_buffer_call

        self.validate_texture_call_arity(func_name, args)
        self.validate_image_resource_argument(func_name, args)
        self.validate_image_access_argument(func_name, args)
        self.validate_texture_resource_argument(func_name, args)
        self.validate_image_multisample_arguments(func_name, args)
        self.validate_image_atomic_format_argument(func_name, args)
        self.validate_integer_coordinate_argument(func_name, args)
        self.validate_coordinate_dimension_argument(func_name, args)
        self.validate_query_lod_coordinate_argument(func_name, args)
        self.validate_compare_argument(func_name, args)
        self.validate_lod_argument(func_name, args)
        self.validate_bias_argument(func_name, args)
        self.validate_sample_index_argument(func_name, args)
        self.validate_mip_level_argument(func_name, args)
        self.validate_gradient_dimension_arguments(func_name, args)
        self.validate_offset_dimension_argument(func_name, args)
        self.validate_gather_component_argument(func_name, args)

        image_call = self.generate_image_call(func_name, args)
        if image_call is not None:
            return image_call

        if is_resource_size_query_operation(func_name) and args:
            lod_arg = args[1] if len(args) > 1 else None
            return self.texture_query_size_expression(args[0], lod_arg)

        if is_texture_query_levels_operation(func_name) and args:
            return self.texture_query_levels_expression(args[0])

        if is_resource_samples_query_operation(func_name) and args:
            return self.texture_samples_expression(args[0])

        if len(args) < 2:
            return None

        texture_type = self.texture_argument_resource_type(args[0])
        if texture_type is None and not names_sampled_texture_operation(func_name):
            # Nothing below applies; the caller generates this call's arguments (texture_call_names).
            return None

        parts = self.texture_call_parts(args)
        if parts is None:
            return None

        texture_name, sampler_arg, coord, extra_args = parts
        storage_image_operation = self.storage_image_texture_operation_expression(
            func_name, texture_type
        )
        if storage_image_operation is not None:
            return storage_image_operation

        packed_shadow_call = (
            self.generate_packed_shadow_texture_sample_call(
                func_name,
                texture_name,
                sampler_arg,
                coord,
                extra_args,
                texture_type,
            )
            if not self.is_plain_depth_read(func_name, args)
            else None
        )
        if packed_shadow_call is not None:
            return packed_shadow_call

        is_array_texture = self.is_array_texture_resource(texture_type)
        if is_array_texture:
            coord_xy, layer = self.texture_coordinate_parts(texture_type, coord)

        if is_texture_sample_operation(
            func_name
        ) and self.is_multisample_texture_resource(texture_type):
            return self.unsupported_multisample_texture_call(func_name, texture_type)

        if is_texture_sample_basic_operation(func_name):
            if extra_args:
                if self.is_texture1d_sample_resource(texture_type):
                    return self.unsupported_texture1d_sampling_option_call(
                        func_name, texture_type
                    )
                bias = self.generate_expression(extra_args[0])
                if is_array_texture:
                    return (
                        f"{texture_name}.sample("
                        f"{sampler_arg}, {coord_xy}, {layer}, bias({bias}))"
                    )
                return f"{texture_name}.sample({sampler_arg}, {coord}, bias({bias}))"
            if is_array_texture:
                return f"{texture_name}.sample({sampler_arg}, {coord_xy}, {layer})"
            return f"{texture_name}.sample({sampler_arg}, {coord})"
        if is_texture_sample_lod_operation(func_name) and extra_args:
            if self.is_texture1d_sample_resource(texture_type):
                if self.metal_texture1d_explicit_lod_matches_kernel_sampling(
                    extra_args[0]
                ):
                    if is_array_texture:
                        return (
                            f"{texture_name}.sample({sampler_arg}, {coord_xy}, {layer})"
                        )
                    return f"{texture_name}.sample({sampler_arg}, {coord})"
                return self.unsupported_texture1d_sampling_option_call(
                    func_name, texture_type
                )
            lod = self.generate_expression(extra_args[0])
            if is_array_texture:
                return f"{texture_name}.sample({sampler_arg}, {coord_xy}, {layer}, level({lod}))"
            return f"{texture_name}.sample({sampler_arg}, {coord}, level({lod}))"
        if is_texture_sample_grad_operation(func_name) and len(extra_args) >= 2:
            if self.is_texture1d_sample_resource(texture_type):
                return self.unsupported_texture1d_sampling_option_call(
                    func_name, texture_type
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            if is_array_texture:
                return f"{texture_name}.sample({sampler_arg}, {coord_xy}, {layer}, {gradient_options})"
            return f"{texture_name}.sample({sampler_arg}, {coord}, {gradient_options})"
        if is_texture_sample_offset_operation(func_name):
            return self.generate_texture_sample_offset_call(
                func_name,
                texture_name,
                sampler_arg,
                coord,
                extra_args,
                texture_type,
            )
        if is_projected_texture_operation(func_name):
            return self.generate_texture_projected_call(
                func_name,
                texture_name,
                sampler_arg,
                coord,
                extra_args,
                texture_type,
                args,
            )
        if is_texture_gather_operation(func_name):
            return self.generate_texture_gather_call(
                func_name, texture_name, sampler_arg, coord, extra_args, texture_type
            )
        if is_texture_compare_operation(func_name):
            return self.generate_texture_compare_call(
                func_name,
                texture_name,
                sampler_arg,
                coord,
                extra_args,
                texture_type,
                args,
            )
        if is_texture_gather_compare_operation(func_name):
            return self.generate_texture_gather_compare_call(
                func_name, texture_name, sampler_arg, coord, extra_args, texture_type
            )
        if is_texture_query_lod_operation(func_name):
            if self.is_multisample_texture_resource(texture_type):
                return self.unsupported_multisample_texture_query_lod_call(texture_type)
            if self.is_storage_image_resource(texture_type):
                return self.unsupported_texture_query_lod_call(texture_type)
            if self.is_texture1d_sample_resource(texture_type):
                return self.unsupported_texture_query_lod_call(texture_type)
            lod_coord = self.texture_query_lod_coordinate(texture_type, coord)
            return (
                f"float2({texture_name}.calculate_clamped_lod({sampler_arg}, {lod_coord}), "
                f"{texture_name}.calculate_unclamped_lod({sampler_arg}, {lod_coord}))"
            )
        if is_texel_fetch_basic_operation(func_name) and len(args) >= 3:
            if self.is_cube_texture_resource(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            texel_coord, layer = self.texture_read_coordinate_parts(texture_type, coord)
            if self.is_multisample_texture_resource(texture_type):
                sample = self.unsigned_coordinate_expression(
                    self.generate_expression(args[2]), 1
                )
                if layer is not None:
                    return f"{texture_name}.read({texel_coord}, {layer}, {sample})"
                return f"{texture_name}.read({texel_coord}, {sample})"
            if self.is_texture1d_sample_resource(texture_type):
                lod = self.metal_texture1d_read_lod_argument(args[2])
                if lod is None:
                    return self.unsupported_texture1d_texel_fetch_lod_call(
                        func_name, texture_type
                    )
                if layer is not None:
                    return f"{texture_name}.read({texel_coord}, {layer}, {lod})"
                return f"{texture_name}.read({texel_coord}, {lod})"
            lod = self.unsigned_coordinate_expression(
                self.generate_expression(args[2]), 1
            )
            if layer is not None:
                return f"{texture_name}.read({texel_coord}, {layer}, {lod})"
            return f"{texture_name}.read({texel_coord}, {lod})"

        if is_texel_fetch_offset_operation(func_name) and len(args) >= 4:
            if self.is_cube_texture_resource(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            if self.is_multisample_texture_resource(texture_type):
                return unsupported_multisample_texel_fetch_offset_expression("Metal")
            if self.is_texture1d_sample_resource(texture_type):
                lod = self.metal_texture1d_read_lod_argument(args[2])
                if lod is None:
                    return self.unsupported_texture1d_texel_fetch_lod_call(
                        func_name, texture_type
                    )
                offset = self.generate_expression(args[3])
                if is_array_texture:
                    texel_coord, layer, _ = self.texture_read_coordinate_components(
                        texture_type, coord
                    )
                    return (
                        f"{texture_name}.read(uint(({texel_coord} + {offset})), "
                        f"{layer}, {lod})"
                    )
                return f"{texture_name}.read(uint(({coord} + {offset})), {lod})"
            lod = self.unsigned_coordinate_expression(
                self.generate_expression(args[2]), 1
            )
            offset = self.generate_expression(args[3])
            (
                texel_coord,
                layer,
                dimensions,
            ) = self.texture_read_coordinate_components(texture_type, coord)
            if layer is not None:
                offset_coord = self.unsigned_coordinate_expression(
                    f"({texel_coord} + {offset})", dimensions
                )
                return f"{texture_name}.read({offset_coord}, {layer}, {lod})"
            offset_coord = self.unsigned_coordinate_expression(
                f"({texel_coord} + {offset})", dimensions
            )
            return f"{texture_name}.read({offset_coord}, {lod})"

        return None

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if isinstance(type_node, FunctionCallNode):
            return self.safe_expression_to_string(type_node)
        if isinstance(type_node, IdentifierNode):
            return type_node.name
        if isinstance(type_node, PointerType):
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            return f"{pointee_type}*"
        if isinstance(type_node, ReferenceType):
            referenced_type = self.convert_type_node_to_string(
                type_node.referenced_type
            )
            return f"{referenced_type}&"
        generic_args = getattr(type_node, "generic_args", [])
        if getattr(type_node, "name", None) == "array" and len(generic_args) == 2:
            element, extent = generic_args
            if isinstance(element, PointerType):
                pointee = self.metal_standard_array_element_type(
                    self.convert_type_node_to_string(element.pointee_type)
                )
                space = element.address_space or "thread"
                readonly = (
                    "const "
                    if (
                        "const" in getattr(element, "qualifiers", [])
                        or element.access_mode in {"read", "readonly"}
                    )
                    else ""
                )
                memory = self.resource_memory_qualifier_prefix(raw_type=element)
                element_text = f"{memory}{readonly}{space} {pointee}*"
            else:
                element_text = self.metal_standard_array_element_type(
                    self.convert_type_node_to_string(element)
                )
                qualifiers = [
                    str(value)
                    for value in getattr(element, "qualifiers", [])
                    if str(value) in {"const", "volatile"}
                ]
                if qualifiers:
                    element_text = " ".join([*qualifiers, element_text])
            return f"array<{element_text}, {self.safe_expression_to_string(extent)}>"
        if hasattr(type_node, "name") and generic_args:
            args = ", ".join(
                self.convert_type_node_to_string(arg)
                or self.metal_tessellation_patch_argument_text(arg)
                for arg in generic_args
            )
            return f"{type_node.name}<{args}>"
        if hasattr(type_node, "name"):
            return type_node.name
        elif hasattr(type_node, "rows") and hasattr(type_node, "cols"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            if type_node.rows == type_node.cols:
                return f"float{type_node.rows}x{type_node.rows}"
            else:
                return f"float{type_node.cols}x{type_node.rows}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            if str(type(type_node)).find("ArrayType") != -1:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    if isinstance(type_node.size, int):
                        return prepend_array_dimension(element_type, type_node.size)
                    else:
                        size_str = self.safe_expression_to_string(type_node.size)
                        return prepend_array_dimension(element_type, size_str)
                else:
                    return prepend_array_dimension(element_type)
            else:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
                if element_type == "float":
                    return f"float{size}"
                elif element_type == "int":
                    return f"int{size}"
                elif element_type == "uint":
                    return f"uint{size}"
                elif element_type == "bool":
                    return f"bool{size}"
                named_vector = crossgl_named_vector_type(element_type, size)
                if named_vector is not None:
                    return named_vector
                return f"{element_type}{size}"
        else:
            return str(type_node)

    def safe_expression_to_string(self, expr):
        """Convert an expression node to a string representation safely (avoid infinite recursion)."""
        return self.safe_expression_to_string_with_precedence(expr)

    def safe_expression_to_string_with_precedence(self, expr, parent_precedence=0):
        if hasattr(expr, "value"):
            return str(expr.value)
        elif isinstance(expr, FunctionCallNode):
            callee = self.safe_expression_to_string_with_precedence(expr.function)
            generic_args = getattr(expr, "generic_args", []) or []
            if generic_args:
                args = ", ".join(
                    self.convert_type_node_to_string(arg) for arg in generic_args
                )
                callee = f"{callee}<{args}>"
            arguments = ", ".join(
                self.safe_expression_to_string_with_precedence(argument)
                for argument in getattr(expr, "arguments", [])
            )
            return f"{callee}({arguments})"
        elif isinstance(expr, MemberAccessNode):
            object_expr = self.safe_expression_to_string_with_precedence(
                expr.object_expr
            )
            return f"{object_expr}.{expr.member}"
        elif getattr(expr, "name", None) is not None:
            return str(expr.name)
        elif isinstance(expr, int) or isinstance(expr, float):
            return str(expr)
        elif isinstance(expr, str):
            return expr
        elif isinstance(expr, BinaryOpNode):
            operator = self.map_operator(expr.op)
            precedence = self.expression_precedence(operator)
            left = self.safe_expression_to_string_with_precedence(expr.left, precedence)
            right = self.safe_expression_to_string_with_precedence(
                expr.right, precedence + 1
            )
            expression = f"{left} {operator} {right}"
            if precedence < parent_precedence:
                return f"({expression})"
            return expression
        elif isinstance(expr, UnaryOpNode):
            operand = self.safe_expression_to_string_with_precedence(
                expr.operand, self.expression_precedence("unary")
            )
            operator = self.map_operator(expr.op)
            if getattr(expr, "is_postfix", False):
                return f"{operand}{operator}"
            return f"{operator}{operand}"
        elif isinstance(expr, TernaryOpNode):
            # A conditional extent such as `[size == 0 ? 1 : size]` binds below
            # every binary operator, so it is parenthesized inside one.
            condition = self.safe_expression_to_string_with_precedence(
                expr.condition, 1
            )
            true_value = self.safe_expression_to_string_with_precedence(expr.true_expr)
            false_value = self.safe_expression_to_string_with_precedence(
                expr.false_expr
            )
            expression = f"{condition} ? {true_value} : {false_value}"
            return f"({expression})" if parent_precedence > 0 else expression
        else:
            # Fallback - avoid calling generate_expression to prevent infinite recursion
            return str(expr)

    def expression_precedence(self, operator):
        return {
            "||": 1,
            "&&": 2,
            "|": 3,
            "^": 4,
            "&": 5,
            "==": 6,
            "!=": 6,
            "<": 7,
            ">": 7,
            "<=": 7,
            ">=": 7,
            "<<": 8,
            ">>": 8,
            "+": 9,
            "-": 9,
            "*": 10,
            "/": 10,
            "%": 10,
            "unary": 11,
        }.get(operator, 0)

    def expression_to_string(self, expr):
        """Convert an expression node to a string representation."""
        return self.safe_expression_to_string(expr)

    def collect_metal_type_aliases(self, ast):
        return {
            node.name: getattr(node, "var_type", None)
            for node in getattr(ast, "global_variables", []) or []
            if self.is_metal_type_alias_declaration(node)
        }

    @staticmethod
    def is_metal_type_alias_declaration(node):
        return isinstance(node, VariableNode) and bool(
            getattr(node, "is_type_alias", False)
        )

    def resolve_metal_type_alias(self, type_name):
        aliases = getattr(self, "metal_type_aliases", {})
        resolved = type_name
        seen = []
        while resolved in aliases:
            if resolved in seen:
                cycle = " -> ".join([*seen, resolved])
                raise MetalRefusal(f"Cyclic Metal type alias: {cycle}")
            seen.append(resolved)
            resolved = self.type_name_string(aliases[resolved])
        return resolved
