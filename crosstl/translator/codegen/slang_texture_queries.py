"""Texture queries and query helpers of the Slang generator, plus sampler and texture type predicates."""

from ..ast import (
    ArrayAccessNode,
    IdentifierNode,
    VariableNode,
)


class SlangTextureQueriesMixin:
    """Texture-query members of ``SlangCodeGen``."""

    def generate_texture_query_levels(self, args):
        if not args:
            return self.unsupported_resource_query_call(
                "textureQueryLevels", "requires a resource argument"
            )
        if len(args) != 1:
            return self.unsupported_resource_query_call(
                "textureQueryLevels", "accepts only a resource argument"
            )

        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        spec = self.dimension_query_spec(resource_type)
        if (
            spec is None
            or not spec["mip"]
            or not self.is_sampled_texture_resource_type(resource_type)
        ):
            return self.unsupported_resource_query_call(
                "textureQueryLevels",
                "requires a mipmapped sampled texture resource",
            )

        expected_reason = self.resource_query_expected_type_unsupported_reason(
            "textureQueryLevels", "int"
        )
        if expected_reason:
            return self.unsupported_resource_query_call(
                "textureQueryLevels",
                expected_reason,
                self.zero_value_for_type(self.current_expression_expected_type),
            )

        resource_name = self.generate_expression(args[0])
        base_helper_name = f"cgl_textureQueryLevels_{resource_type}"
        helper_name = self.helper_function_name(base_helper_name)
        self.register_helper_function(
            helper_name,
            self.build_texture_query_levels_helper(helper_name, resource_type, spec),
        )
        return f"{helper_name}({resource_name})"

    def sample_count_query_accepts_resource(self, func_name, resource_type):
        if func_name == "textureSamples":
            return self.is_sampled_texture_resource_type(resource_type)
        if func_name == "imageSamples":
            return self.is_storage_image_type(resource_type)
        return False

    def sample_count_query_requirement(self, func_name):
        if func_name == "textureSamples":
            return "requires a multisampled texture resource"
        if func_name == "imageSamples":
            return "requires a multisampled image resource"
        return "requires a multisampled resource"

    def unsupported_resource_query_call(self, func_name, reason, fallback="0"):
        return self.refuse_lowering(
            f"Slang resource query: {func_name} {reason}", fallback
        )

    def generate_texture_query_lod(self, args):
        query_args = self.texture_query_lod_args(args)
        if query_args is None:
            return self.unsupported_texture_query_lod_call(
                self.texture_query_lod_unsupported_reason(args)
            )

        texture_name, coord = query_args
        coord_node = args[self.texture_query_lod_coord_index(args)]
        coord_reason = self.sampled_texture_coordinate_rank_unsupported_reason(
            args[0], coord_node
        )
        if coord_reason:
            return self.unsupported_texture_query_lod_call(coord_reason)
        expected_reason = self.resource_query_expected_type_unsupported_reason(
            "textureQueryLod", "float2"
        )
        if expected_reason:
            return self.unsupported_texture_query_lod_call(
                expected_reason,
                self.zero_value_for_type(self.current_expression_expected_type),
            )
        lod_args = ", ".join(self.slang_texture_method_args(args, coord))
        clamped = f"{texture_name}.CalculateLevelOfDetail({lod_args})"
        unclamped = f"{texture_name}.CalculateLevelOfDetailUnclamped({lod_args})"
        return f"float2({clamped}, {unclamped})"

    def texture_query_lod_args(self, args):
        coord_index = self.texture_query_lod_coord_index(args)
        if len(args) <= coord_index or len(args) != coord_index + 1:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        if not self.is_lod_query_sampler_type(resource_type):
            return None

        texture_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[coord_index])
        return texture_name, coord

    def texture_query_lod_coord_index(self, args):
        if len(args) >= 2 and self.is_sampler_state_type(
            self.get_expression_type(args[1])
        ):
            return 2
        return 1

    def texture_query_lod_unsupported_reason(self, args):
        if not args:
            return "requires texture and coordinate arguments"

        coord_index = self.texture_query_lod_coord_index(args)
        if len(args) <= coord_index:
            return "requires texture and coordinate arguments"
        if len(args) != coord_index + 1:
            return "accepts only texture, optional sampler, and coordinate arguments"

        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        if not self.is_lod_query_sampler_type(resource_type):
            return "requires a non-shadow non-multisampled sampled texture resource"

        return "requires texture and coordinate arguments"

    def unsupported_texture_query_lod_call(self, reason, fallback="float2(0.0, 0.0)"):
        return self.unsupported_resource_query_call("textureQueryLod", reason, fallback)

    def register_helper_function(self, name, source):
        if name not in self.helper_functions:
            self.helper_functions[name] = source

    def helper_function_name(self, base_name):
        if base_name in self.helper_name_aliases:
            return self.helper_name_aliases[base_name]

        candidate = base_name
        suffix = 1
        used_helper_names = set(self.helper_functions) | set(
            self.helper_name_aliases.values()
        )
        while candidate in self.user_symbol_names or candidate in used_helper_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1

        self.helper_name_aliases[base_name] = candidate
        return candidate

    def build_dimension_query_helper(
        self, helper_name, resource_type, spec, resource_slang_type=None
    ):
        resource_slang_type = resource_slang_type or self.convert_type(resource_type)
        return_type = self.query_return_type(spec["dimensions"])
        params = f"{resource_slang_type} tex"
        if spec["mip"]:
            params += ", uint mipLevel"

        declarations = self.query_local_declarations(spec)
        get_dimensions_args = self.get_dimensions_args(spec)
        dimensions = ", ".join(spec["dimensions"])
        if len(spec["dimensions"]) == 1:
            return_value = spec["dimensions"][0]
        else:
            return_value = f"{return_type}({dimensions})"

        return (
            f"{return_type} {helper_name}({params})\n"
            "{\n"
            f"{declarations}"
            f"    tex.GetDimensions({get_dimensions_args});\n"
            f"    return {return_value};\n"
            "}"
        )

    def build_sample_count_query_helper(
        self, helper_name, resource_type, spec, resource_slang_type=None
    ):
        resource_slang_type = resource_slang_type or self.convert_type(resource_type)
        declarations = self.query_local_declarations(spec)
        get_dimensions_args = self.get_dimensions_args(spec)
        return (
            f"int {helper_name}({resource_slang_type} tex)\n"
            "{\n"
            f"{declarations}"
            f"    tex.GetDimensions({get_dimensions_args});\n"
            "    return samples;\n"
            "}"
        )

    def build_texture_query_levels_helper(self, helper_name, resource_type, spec):
        resource_slang_type = self.convert_type(resource_type)
        declarations = self.query_local_declarations(spec)
        get_dimensions_args = self.texture_query_levels_args(spec)
        return (
            f"int {helper_name}({resource_slang_type} tex)\n"
            "{\n"
            f"{declarations}"
            f"    tex.GetDimensions({get_dimensions_args});\n"
            "    return levels;\n"
            "}"
        )

    def query_return_type(self, dimensions):
        if len(dimensions) == 1:
            return "int"
        return f"int{len(dimensions)}"

    def query_local_declarations(self, spec):
        names = list(spec.get("query_dimensions", spec["dimensions"]))
        if spec["samples"]:
            names.append("samples")
        if spec["mip"]:
            names.append("levels")
        return "".join(f"    int {name};\n" for name in names)

    def get_dimensions_args(self, spec):
        args = []
        if spec["mip"]:
            args.append("mipLevel")
        args.extend(spec.get("query_dimensions", spec["dimensions"]))
        if spec["samples"]:
            args.append("samples")
        if spec["mip"]:
            args.append("levels")
        return ", ".join(args)

    def texture_query_levels_args(self, spec):
        args = []
        if spec["mip"]:
            args.append("0u")
        args.extend(spec["dimensions"])
        args.append("levels")
        return ", ".join(args)

    def dimension_query_spec(self, type_name):
        specs = {
            "sampler1D": (("width",), True, False),
            "sampler1DArray": (("width", "elements"), True, False),
            "sampler2D": (("width", "height"), True, False),
            "sampler2DShadow": (("width", "height"), True, False),
            "sampler2DArray": (("width", "height", "elements"), True, False),
            "sampler2DArrayShadow": (
                ("width", "height", "elements"),
                True,
                False,
            ),
            "sampler3D": (("width", "height", "depth"), True, False),
            "samplerCube": (("width", "height"), True, False),
            "samplerCubeShadow": (("width", "height"), True, False),
            "samplerCubeArray": (("width", "height", "elements"), True, False),
            "samplerCubeArrayShadow": (
                ("width", "height", "elements"),
                True,
                False,
            ),
            "sampler2DMS": (("width", "height"), False, True),
            "sampler2DMSArray": (("width", "height", "elements"), False, True),
            "image1D": (("width",), False, False),
            "iimage1D": (("width",), False, False),
            "uimage1D": (("width",), False, False),
            "image1DArray": (("width", "elements"), False, False),
            "iimage1DArray": (("width", "elements"), False, False),
            "uimage1DArray": (("width", "elements"), False, False),
            "image2D": (("width", "height"), False, False),
            "iimage2D": (("width", "height"), False, False),
            "uimage2D": (("width", "height"), False, False),
            "image2DArray": (("width", "height", "elements"), False, False),
            "iimage2DArray": (("width", "height", "elements"), False, False),
            "uimage2DArray": (("width", "height", "elements"), False, False),
            "image3D": (("width", "height", "depth"), False, False),
            "iimage3D": (("width", "height", "depth"), False, False),
            "uimage3D": (("width", "height", "depth"), False, False),
            "imageCube": (
                ("width", "height"),
                False,
                False,
                ("width", "height", "elements"),
            ),
            "iimageCube": (
                ("width", "height"),
                False,
                False,
                ("width", "height", "elements"),
            ),
            "uimageCube": (
                ("width", "height"),
                False,
                False,
                ("width", "height", "elements"),
            ),
            "imageCubeArray": (("width", "height", "elements"), False, False),
            "iimageCubeArray": (("width", "height", "elements"), False, False),
            "uimageCubeArray": (("width", "height", "elements"), False, False),
            "image2DMS": (("width", "height"), False, True),
            "iimage2DMS": (("width", "height"), False, True),
            "uimage2DMS": (("width", "height"), False, True),
            "image2DMSArray": (("width", "height", "elements"), False, True),
            "iimage2DMSArray": (("width", "height", "elements"), False, True),
            "uimage2DMSArray": (("width", "height", "elements"), False, True),
        }
        spec = specs.get(type_name)
        if spec is None:
            return None
        dimensions, mip, samples, *rest = spec
        result = {
            "dimensions": dimensions,
            "mip": mip,
            "samples": samples,
        }
        if rest:
            result["query_dimensions"] = rest[0]
        return result

    def is_explicit_sampler_argument(self, args):
        if len(args) < 2:
            return False
        return self.is_sampler_state_type(self.get_expression_type(args[1]))

    def is_sampler_state_type(self, type_name):
        return self.resource_base_type(type_name) in {
            "sampler",
            "comparison_sampler",
            "SamplerState",
            "SamplerComparisonState",
        }

    def is_lod_query_sampler_type(self, type_name):
        resource_type = self.resource_base_type(type_name)
        shape_type = self.sampled_texture_shape_type(resource_type)
        return (
            isinstance(shape_type, str)
            and shape_type.startswith("sampler")
            and shape_type != "sampler"
            and "MS" not in shape_type
            and "Shadow" not in shape_type
        )

    def sampled_texture_shape_type(self, type_name):
        resource_type = self.resource_base_type(type_name)
        if not isinstance(resource_type, str):
            return None
        for prefix in ("isampler", "usampler"):
            if resource_type.startswith(prefix):
                return f"sampler{resource_type[len(prefix):]}"
        return resource_type

    def sampled_texture_value_type(self, type_name):
        resource_type = self.resource_base_type(type_name)
        if isinstance(resource_type, str) and resource_type.startswith("isampler"):
            return "int4"
        if isinstance(resource_type, str) and resource_type.startswith("usampler"):
            return "uint4"
        return "float4"

    def is_sampled_texture_resource_type(self, type_name):
        resource_type = self.resource_base_type(type_name)
        return (
            isinstance(resource_type, str)
            and resource_type.startswith(("sampler", "isampler", "usampler"))
            and resource_type != "sampler"
        )

    def get_expression_type(self, node):
        name = self.get_expression_name(node)
        if name is None:
            return None
        return self.variable_types.get(name)

    def get_expression_name(self, node):
        if isinstance(node, IdentifierNode):
            return node.name
        if isinstance(node, VariableNode):
            return node.name
        if isinstance(node, str):
            return node
        if isinstance(node, ArrayAccessNode):
            return self.get_expression_name(
                getattr(node, "array", getattr(node, "array_expr", None))
            )
        return None

    def resource_base_type(self, type_name):
        type_name = self.type_name_string(type_name)
        if not isinstance(type_name, str):
            return None
        return type_name.split("[", 1)[0]

    def is_multisample_sampler_type(self, type_name):
        return self.sampled_texture_shape_type(type_name) in {
            "sampler2DMS",
            "sampler2DMSArray",
        }

    def is_texel_fetch_sampler_type(self, type_name):
        return self.sampled_texture_shape_type(type_name) in {
            "sampler1D",
            "sampler1DArray",
            "sampler2D",
            "sampler2DArray",
            "sampler3D",
            "sampler2DMS",
            "sampler2DMSArray",
        }

    def is_texel_fetch_offset_sampler_type(self, type_name):
        return self.sampled_texture_shape_type(type_name) in {
            "sampler1D",
            "sampler1DArray",
            "sampler2D",
            "sampler2DArray",
            "sampler3D",
        }

    def texel_fetch_coord_constructor(self, type_name):
        base_type = self.sampled_texture_shape_type(type_name)
        if base_type in {"sampler1D", "sampler1DArray"}:
            return "int2" if base_type == "sampler1D" else "int3"
        if base_type in {"sampler3D", "sampler2DArray"}:
            return "int4"
        return "int3"
