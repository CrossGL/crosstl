"""Resource types and storage images in the Slang generator: generic resource mapping, image formats, access and image load/store."""

from .array_utils import split_array_type_suffix
from .hlsl_image_formats import (
    declared_texel_type,
    normalized_texel_type,
    texel_attribute,
    texel_shape,
)


class SlangImageResourcesMixin:
    """Image-resource members of ``SlangCodeGen``."""

    def map_slang_generic_resource_type(self, type_name):
        """Map generic resource element aliases while preserving resource spelling."""
        if not isinstance(type_name, str):
            return None

        base_type = self.resource_base_type(type_name)
        generic_resource_types = {
            "Texture1D",
            "Texture1DArray",
            "Texture2D",
            "Texture2DArray",
            "Texture2DMS",
            "Texture2DMSArray",
            "Texture3D",
            "TextureCube",
            "TextureCubeArray",
            "RWTexture1D",
            "RWTexture1DArray",
            "RWTexture2D",
            "RWTexture2DArray",
            "RWTexture2DMS",
            "RWTexture2DMSArray",
            "RWTexture3D",
            "Sampler1D",
            "Sampler1DArray",
            "Sampler2D",
            "Sampler2DArray",
            "Sampler2DMS",
            "Sampler2DMSArray",
            "Sampler3D",
            "SamplerCube",
            "SamplerCubeArray",
            "Buffer",
            "RWBuffer",
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
            "ConstantBuffer",
            "ParameterBlock",
            "PointStream",
            "LineStream",
            "TriangleStream",
        }
        for resource_type in generic_resource_types:
            prefix = f"{resource_type}<"
            if not base_type.startswith(prefix) or not base_type.endswith(">"):
                continue
            element_type = base_type[len(prefix) : -1].strip()
            if not element_type:
                return None
            mapped_element_type = self.convert_type(element_type)
            return type_name.replace(
                base_type, f"{resource_type}<{mapped_element_type}>", 1
            )
        return None

    def supported_image_formats(self):
        return {
            "r8",
            "r8_snorm",
            "r8i",
            "r8ui",
            "r16",
            "r16_snorm",
            "r16f",
            "r16i",
            "r16ui",
            "r32f",
            "r32i",
            "r32ui",
            "rg8",
            "rg8_snorm",
            "rg8i",
            "rg8ui",
            "rg16",
            "rg16_snorm",
            "rg16f",
            "rg16i",
            "rg16ui",
            "rg32f",
            "rg32i",
            "rg32ui",
            "rgba8",
            "rgba8_snorm",
            "rgba8i",
            "rgba8ui",
            "rgba16",
            "rgba16_snorm",
            "rgba16f",
            "rgba16i",
            "rgba16ui",
            "rgba32f",
            "rgba32i",
            "rgba32ui",
            "rgb10_a2",
            "rgb10_a2ui",
            "r11f_g11f_b10f",
        }

    def scalar_image_format_components(self):
        return {
            "r8": "float",
            "r8_snorm": "float",
            "r16": "float",
            "r16_snorm": "float",
            "r16f": "float",
            "r32f": "float",
            "r8i": "int",
            "r16i": "int",
            "r32i": "int",
            "r8ui": "uint",
            "r16ui": "uint",
            "r32ui": "uint",
        }

    def vector_image_format_components(self):
        return {
            "rg8": "float2",
            "rg8_snorm": "float2",
            "rg16": "float2",
            "rg16_snorm": "float2",
            "rg16f": "float2",
            "rg8i": "int2",
            "rg16i": "int2",
            "rg8ui": "uint2",
            "rg16ui": "uint2",
            "rg32f": "float2",
            "rg32i": "int2",
            "rg32ui": "uint2",
            "rgba8": "float4",
            "rgba8_snorm": "float4",
            "rgba16": "float4",
            "rgba16_snorm": "float4",
            "rgba16f": "float4",
            "rgba32f": "float4",
            "rgba8i": "int4",
            "rgba16i": "int4",
            "rgba32i": "int4",
            "rgba8ui": "uint4",
            "rgba16ui": "uint4",
            "rgba32ui": "uint4",
            "rgb10_a2": "float4",
            "rgb10_a2ui": "uint4",
            "r11f_g11f_b10f": "float3",
        }

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "name") and value.name is not None:
            return str(value.name)
        if hasattr(value, "value") and value.value is not None:
            return str(value.value).strip('"')
        return str(value)

    def explicit_image_format(self, node):
        if not hasattr(node, "attributes"):
            return None
        supported_formats = self.supported_image_formats()
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in supported_formats:
                return attr_name
            if attr_name == "format":
                arguments = getattr(attr, "arguments", []) or []
                if not arguments:
                    continue
                format_name = self.attribute_value_to_string(arguments[0])
                if format_name is None:
                    continue
                format_name = str(format_name).lower()
                if format_name in supported_formats:
                    return format_name
        return None

    def explicit_resource_access(self, node):
        if node is None:
            return None

        access_names = {
            "read": "readonly",
            "readonly": "readonly",
            "write": "writeonly",
            "writeonly": "writeonly",
            "read_write": "readwrite",
            "readwrite": "readwrite",
            "access::read": "readonly",
            "access::write": "writeonly",
            "access::read_write": "readwrite",
        }
        for qualifier in getattr(node, "qualifiers", []) or []:
            access = access_names.get(str(qualifier).lower())
            if access is not None:
                return access

        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name == "access":
                arguments = getattr(attr, "arguments", []) or []
                if not arguments:
                    continue
                raw_access = self.attribute_value_to_string(arguments[0])
                if raw_access is None:
                    continue
                access = access_names.get(str(raw_access).lower())
            else:
                access = access_names.get(attr_name)
            if access is not None:
                return access
        return None

    def map_resource_type_with_format(self, type_name, node=None):
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return self.convert_type(type_name)

        if "[" in type_name and "]" in type_name:
            base_type, array_suffix = split_array_type_suffix(type_name)
            mapped_base = self.map_resource_base_type_with_format(base_type, node)
            return f"{mapped_base}{array_suffix}"
        return self.map_resource_base_type_with_format(type_name, node)

    def map_resource_base_type_with_format(self, type_name, node=None):
        mapped_type = self.map_explicit_sampler_base_type(type_name, node)
        if mapped_type is None:
            mapped_type = self.map_image_base_type_with_format(type_name, node)
        return self.map_buffer_base_type_with_access(mapped_type, node)

    def map_explicit_sampler_base_type(self, type_name, node=None):
        node_name = getattr(node, "name", getattr(node, "variable_name", None))
        base_type = self.resource_base_type(type_name)
        if (
            node_name in self.explicit_comparison_sampler_names
            and base_type == "sampler"
        ):
            return "SamplerComparisonState"
        if node_name not in self.explicit_sampler_texture_names:
            return None
        return self.separated_slang_texture_type(base_type)

    def separated_slang_texture_type(self, base_type):
        texture_types = {
            "sampler1D": "Texture1D<float4>",
            "sampler1DArray": "Texture1DArray<float4>",
            "sampler2D": "Texture2D<float4>",
            "sampler2DArray": "Texture2DArray<float4>",
            "sampler3D": "Texture3D<float4>",
            "samplerCube": "TextureCube<float4>",
            "samplerCubeArray": "TextureCubeArray<float4>",
            "isampler1D": "Texture1D<int4>",
            "isampler1DArray": "Texture1DArray<int4>",
            "isampler2D": "Texture2D<int4>",
            "isampler2DArray": "Texture2DArray<int4>",
            "isampler3D": "Texture3D<int4>",
            "isamplerCube": "TextureCube<int4>",
            "isamplerCubeArray": "TextureCubeArray<int4>",
            "isampler2DMS": "Texture2DMS<int4>",
            "isampler2DMSArray": "Texture2DMSArray<int4>",
            "usampler1D": "Texture1D<uint4>",
            "usampler1DArray": "Texture1DArray<uint4>",
            "usampler2D": "Texture2D<uint4>",
            "usampler2DArray": "Texture2DArray<uint4>",
            "usampler3D": "Texture3D<uint4>",
            "usamplerCube": "TextureCube<uint4>",
            "usamplerCubeArray": "TextureCubeArray<uint4>",
            "usampler2DMS": "Texture2DMS<uint4>",
            "usampler2DMSArray": "Texture2DMSArray<uint4>",
            "sampler2DShadow": "Texture2D<float>",
            "sampler2DArrayShadow": "Texture2DArray<float>",
            "samplerCubeShadow": "TextureCube<float>",
            "samplerCubeArrayShadow": "TextureCubeArray<float>",
        }
        return texture_types.get(base_type)

    def map_image_base_type_with_format(self, type_name, node=None):
        base_type = self.resource_base_type(type_name)
        explicit_format = self.explicit_image_format(node) if node is not None else None
        if explicit_format is None:
            parameter_type = self.storage_parameter_type(node)
            if parameter_type is not None:
                return parameter_type
        component_type = self.scalar_image_format_components().get(
            explicit_format
        ) or self.vector_image_format_components().get(explicit_format)
        texture_types = {
            "image1D": "RWTexture1D",
            "iimage1D": "RWTexture1D",
            "uimage1D": "RWTexture1D",
            "image2D": "RWTexture2D",
            "iimage2D": "RWTexture2D",
            "uimage2D": "RWTexture2D",
            "image3D": "RWTexture3D",
            "iimage3D": "RWTexture3D",
            "uimage3D": "RWTexture3D",
            "imageCube": "RWTexture2DArray",
            "iimageCube": "RWTexture2DArray",
            "uimageCube": "RWTexture2DArray",
            "image1DArray": "RWTexture1DArray",
            "iimage1DArray": "RWTexture1DArray",
            "uimage1DArray": "RWTexture1DArray",
            "image2DArray": "RWTexture2DArray",
            "iimage2DArray": "RWTexture2DArray",
            "uimage2DArray": "RWTexture2DArray",
            "imageCubeArray": "RWTexture2DArray",
            "iimageCubeArray": "RWTexture2DArray",
            "uimageCubeArray": "RWTexture2DArray",
            "image2DMS": "RWTexture2DMS",
            "iimage2DMS": "RWTexture2DMS",
            "uimage2DMS": "RWTexture2DMS",
            "image2DMSArray": "RWTexture2DMSArray",
            "iimage2DMSArray": "RWTexture2DMSArray",
            "uimage2DMSArray": "RWTexture2DMSArray",
        }
        texture_type = texture_types.get(base_type)
        declared_texel = (
            self.declared_storage_texel_type(node, base_type, explicit_format)
            if texture_type
            else None
        )
        if component_type and texture_type:
            texel_type = normalized_texel_type(explicit_format, component_type)
            return f"{texture_type}<{texel_type}>"
        if declared_texel:
            return f"{texture_type}<{declared_texel}>"
        return self.convert_type(type_name)

    def storage_parameter_type(self, node):
        """The Slang type of the storage images a parameter receives, or ``None``."""
        texels = getattr(self, "storage_parameter_texels", None)
        if node is None or texels is None or texel_attribute(node) is not None:
            return None
        try:
            return texels.parameter_type(
                node,
                lambda declaration: self.map_image_base_type_with_format(
                    declaration.vtype, declaration
                ),
            )
        except ValueError as error:
            return self.refuse_lowering(f"Slang {error}", None)

    def declared_storage_texel_type(self, node, image_type, explicit_format):
        """The Slang texel type a storage image's ``@texel`` declares, or ``None``."""
        if node is None:
            return None
        try:
            return declared_texel_type(
                node, image_type, explicit_format, self.attribute_value_to_string
            )
        except ValueError as error:
            return self.refuse_lowering(f"Slang {error}", None)

    def map_buffer_base_type_with_access(self, type_name, node=None):
        access = self.explicit_resource_access(node)
        if access is None:
            return type_name

        base_type = self.resource_base_type(type_name)
        if not isinstance(base_type, str):
            return type_name

        if base_type in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            mapped_base = (
                "ByteAddressBuffer" if access == "readonly" else "RWByteAddressBuffer"
            )
            return type_name.replace(base_type, mapped_base, 1)

        structured_prefixes = ("StructuredBuffer<", "RWStructuredBuffer<")
        if not base_type.startswith(structured_prefixes) or not base_type.endswith(">"):
            return type_name

        element_type = base_type[base_type.find("<") + 1 : -1].strip()
        if not element_type:
            return type_name

        resource_type = (
            "StructuredBuffer" if access == "readonly" else "RWStructuredBuffer"
        )
        return type_name.replace(base_type, f"{resource_type}<{element_type}>", 1)

    def is_storage_image_type(self, type_name):
        base_type = self.resource_base_type(type_name)
        return isinstance(base_type, str) and base_type in {
            "image1D",
            "iimage1D",
            "uimage1D",
            "image2D",
            "iimage2D",
            "uimage2D",
            "image3D",
            "iimage3D",
            "uimage3D",
            "imageCube",
            "iimageCube",
            "uimageCube",
            "image1DArray",
            "iimage1DArray",
            "uimage1DArray",
            "image2DArray",
            "iimage2DArray",
            "uimage2DArray",
            "imageCubeArray",
            "iimageCubeArray",
            "uimageCubeArray",
            "image2DMS",
            "iimage2DMS",
            "uimage2DMS",
            "image2DMSArray",
            "iimage2DMSArray",
            "uimage2DMSArray",
        }

    def image_resource_type(self, image_arg):
        image_name = self.get_expression_name(image_arg)
        if not image_name:
            return None
        return self.image_resource_types.get(image_name)

    def image_resource_access(self, image_arg):
        image_name = self.get_expression_name(image_arg)
        if not image_name:
            return None
        return self.image_resource_accesses.get(image_name)

    def image_resource_element_type(self, image_type):
        image_type = self.resource_base_type(image_type)
        if not image_type or "<" not in image_type or not image_type.endswith(">"):
            return None
        return image_type[image_type.find("<") + 1 : -1]

    def vector_size(self, type_name):
        if not isinstance(type_name, str) or not type_name[-1:].isdigit():
            return None
        size = int(type_name[-1])
        return size if size in {2, 3, 4} else None

    def vector_zero_value(self, type_name):
        if isinstance(type_name, str) and type_name.startswith("bool"):
            return "false"
        if isinstance(type_name, str) and type_name.startswith("uint"):
            return "0u"
        if isinstance(type_name, str) and type_name.startswith("int"):
            return "0"
        return "0.0"

    def unsupported_image_access_call(self, operation, reason, result_type=None):
        placeholder = (
            "" if result_type is None else self.zero_value_for_type(result_type)
        )
        return self.refuse_lowering(
            f"Slang image access: {operation} {reason}", placeholder
        )

    def is_multisample_storage_image_type(self, type_name):
        return self.resource_base_type(type_name) in {
            "image2DMS",
            "iimage2DMS",
            "uimage2DMS",
            "image2DMSArray",
            "iimage2DMSArray",
            "uimage2DMSArray",
        }

    def image_coordinate_rank(self, image_type):
        return {
            "image1D": 1,
            "iimage1D": 1,
            "uimage1D": 1,
            "image1DArray": 2,
            "iimage1DArray": 2,
            "uimage1DArray": 2,
            "image2D": 2,
            "iimage2D": 2,
            "uimage2D": 2,
            "image2DMS": 2,
            "iimage2DMS": 2,
            "uimage2DMS": 2,
            "imageCube": 3,
            "iimageCube": 3,
            "uimageCube": 3,
            "image2DArray": 3,
            "iimage2DArray": 3,
            "uimage2DArray": 3,
            "imageCubeArray": 3,
            "iimageCubeArray": 3,
            "uimageCubeArray": 3,
            "image2DMSArray": 3,
            "iimage2DMSArray": 3,
            "uimage2DMSArray": 3,
            "image3D": 3,
            "iimage3D": 3,
            "uimage3D": 3,
        }.get(self.resource_base_type(image_type))

    def image_coordinate_unsupported_reason(self, image_type, coord):
        return self.texture_rank_unsupported_reason(
            coord,
            self.image_coordinate_rank(image_type),
            self.resource_base_type(image_type),
            "coordinate",
        )

    def image_sample_index_unsupported_reason(self, sample):
        return self.scalar_integer_texture_argument_unsupported_reason(
            sample, "sample argument"
        )

    def image_load_arity_unsupported_reason(self, image_type, args):
        if self.is_multisample_storage_image_type(image_type):
            if len(args) != 3:
                return "requires image, coordinate, and sample arguments"
            return None
        if len(args) != 2:
            return "accepts image and coordinate arguments"
        return None

    def image_store_arity_unsupported_reason(self, image_type, args):
        if self.is_multisample_storage_image_type(image_type):
            if len(args) != 4:
                return "requires image, coordinate, sample, and value arguments"
            return None
        if len(args) != 3:
            return "requires image, coordinate, and value arguments"
        return None

    def image_access_result_type(self, image_arg):
        return (
            self.image_resource_element_type(self.image_resource_type(image_arg))
            or self.current_expression_expected_type
            or "uint"
        )

    def image_load_expression(self, args):
        if len(args) < 2:
            return self.unsupported_image_access_call(
                "imageLoad",
                "requires image and coordinate arguments",
                self.current_expression_expected_type or "uint",
            )

        source_type = self.resource_base_type(self.get_expression_type(args[0]))
        if not self.is_storage_image_type(source_type):
            return self.unsupported_image_access_call(
                "imageLoad",
                "requires an image resource",
                self.current_expression_expected_type or "uint",
            )

        arity_reason = self.image_load_arity_unsupported_reason(source_type, args)
        if arity_reason:
            return self.unsupported_image_access_call(
                "imageLoad",
                arity_reason,
                self.image_access_result_type(args[0]),
            )

        coord_reason = self.image_coordinate_unsupported_reason(source_type, args[1])
        if coord_reason:
            return self.unsupported_image_access_call(
                "imageLoad",
                coord_reason,
                self.image_access_result_type(args[0]),
            )

        if self.is_multisample_storage_image_type(source_type):
            sample_reason = self.image_sample_index_unsupported_reason(args[2])
            if sample_reason:
                return self.unsupported_image_access_call(
                    "imageLoad",
                    sample_reason,
                    self.image_access_result_type(args[0]),
                )

        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
        image_type = self.image_resource_type(args[0])
        element_type = self.image_resource_element_type(image_type)
        if self.image_resource_access(args[0]) == "writeonly":
            return self.unsupported_image_access_call(
                "imageLoad",
                "requires readable image resource",
                element_type or self.current_expression_expected_type or "uint",
            )

        if len(args) >= 3:
            sample = self.generate_expression(args[2])
            load_expr = f"{image_name}[{coord}, {sample}]"
        else:
            load_expr = f"{image_name}[{coord}]"

        if self.vector_size(element_type) and self.is_scalar_value_type(
            self.current_expression_expected_type
        ):
            return f"{load_expr}.x"
        shape = texel_shape(element_type)
        expected_type = self.current_expression_expected_type
        expected_size = (
            self.vector_size(self.convert_type(expected_type) or "")
            if isinstance(expected_type, str)
            else None
        )
        if (
            shape is not None
            and shape[1] < 4
            and not self.is_scalar_value_type(self.current_expression_expected_type)
            and not (expected_size and expected_size <= shape[1])
        ):
            # imageLoad returns four components; Slang returns the texel's own,
            # so fill the missing ones as an image read does: zero, then one in w.
            kind, count = shape
            zero = self.vector_zero_value(kind)
            one = {"float": "1.0", "int": "1", "uint": "1u"}[kind]
            fill = ", ".join([zero] * (3 - count) + [one])
            return f"{kind}4({load_expr}, {fill})"
        return load_expr

    def image_store_value_expression(self, image_arg, value_arg):
        value = self.generate_expression(value_arg)
        image_type = self.image_resource_type(image_arg)
        element_type = self.image_resource_element_type(image_type)
        shape = texel_shape(element_type)
        value_type = self.type_name_string(self.expression_result_type(value_arg))
        value_size = self.vector_size(self.convert_type(value_type) or "")
        if shape is not None and value_size and value_size > shape[1]:
            # imageStore takes four components and writes the texel's own.
            return f"({value}).{'xyzw'[: shape[1]]}"
        if not self.vector_size(element_type):
            return value
        if not self.is_scalar_value_type(value_type):
            return value

        texel_type = f"{shape[0]}{shape[1]}" if shape is not None else element_type
        if self.vector_size(element_type) == 2:
            return f"{texel_type}({value}, {self.vector_zero_value(texel_type)})"
        return f"{texel_type}({value})"

    def image_store_expression(self, args):
        if len(args) < 3:
            return self.unsupported_image_access_call(
                "imageStore",
                "requires image, coordinate, and value arguments",
            )

        source_type = self.resource_base_type(self.get_expression_type(args[0]))
        if not self.is_storage_image_type(source_type):
            return self.unsupported_image_access_call(
                "imageStore", "requires an image resource"
            )

        arity_reason = self.image_store_arity_unsupported_reason(source_type, args)
        if arity_reason:
            return self.unsupported_image_access_call("imageStore", arity_reason)

        coord_reason = self.image_coordinate_unsupported_reason(source_type, args[1])
        if coord_reason:
            return self.unsupported_image_access_call("imageStore", coord_reason)

        if self.is_multisample_storage_image_type(source_type):
            sample_reason = self.image_sample_index_unsupported_reason(args[2])
            if sample_reason:
                return self.unsupported_image_access_call("imageStore", sample_reason)

        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
        if self.image_resource_access(args[0]) == "readonly":
            return self.unsupported_image_access_call(
                "imageStore", "requires writable image resource"
            )

        if len(args) >= 4:
            sample = self.generate_expression(args[2])
            value = self.image_store_value_expression(args[0], args[3])
            return f"{image_name}[{coord}, {sample}] = {value}"

        value = self.image_store_value_expression(args[0], args[2])
        return f"{image_name}[{coord}] = {value}"
