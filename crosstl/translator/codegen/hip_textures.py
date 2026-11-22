"""Texture sampling, fetch, gather and query lowering of the HIP generator, plus image access diagnostics."""

from ..ast import (
    ArrayAccessNode,
    LiteralNode,
    UnaryOpNode,
)
from .resource_query import ResourceQueryMixin


class HipTextureMixin:
    """Texture members of ``HipCodeGen``."""

    def vector_component_count_for_type(self, type_name):
        if not isinstance(type_name, str):
            return None

        base_type = type_name.split("[", 1)[0].strip()
        scalar_types = {
            "bool",
            "double",
            "f16",
            "f32",
            "f64",
            "float",
            "half",
            "i32",
            "int",
            "u32",
            "uint",
        }
        if base_type in scalar_types:
            return 1

        vector_prefixes = (
            "bvec",
            "dvec",
            "ivec",
            "uvec",
            "vec",
            "bool",
            "double",
            "float",
            "half",
            "int",
            "uint",
        )
        for prefix in vector_prefixes:
            if not base_type.startswith(prefix):
                continue
            suffix = base_type[len(prefix) :]
            if suffix and suffix[0] in {"2", "3", "4"}:
                return int(suffix[0])
        return None

    def texel_fetch_coordinate_count(self, raw_coord):
        coord_type = self.get_expression_type(raw_coord)
        if coord_type is None:
            coord_type = self.expression_result_type(raw_coord)
        return self.vector_component_count_for_type(coord_type)

    def sampled_texture_shape_type(self, texture_type):
        """Return the float sampler shape spelling for typed sampled textures."""
        texture_type = self.resource_base_type(texture_type)
        if not isinstance(texture_type, str):
            return None
        for prefix in ("isampler", "usampler"):
            if texture_type.startswith(prefix):
                return f"sampler{texture_type[len(prefix):]}"
        return texture_type

    def sampled_texture_value_type(self, texture_type):
        """Return the HIP vector type read from a sampled texture."""
        texture_type = self.resource_base_type(texture_type)
        if isinstance(texture_type, str) and texture_type.startswith("isampler"):
            return "int4"
        if isinstance(texture_type, str) and texture_type.startswith("usampler"):
            return "uint4"
        return "float4"

    def is_sampled_resource_type(self, type_name):
        """Whether ``type_name`` is a sampled texture, with float, int or uint texels."""
        return isinstance(type_name, str) and type_name.startswith(
            ("sampler", "isampler", "usampler")
        )

    def is_sampled_texture_family_type(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return (
            isinstance(texture_type, str)
            and texture_type != "sampler"
            and texture_type.startswith(("sampler", "isampler", "usampler"))
        )

    def expected_texel_fetch_coordinate_count(self, texture_type):
        texture_type = self.sampled_texture_shape_type(texture_type)
        return {
            "sampler1D": 1,
            "sampler1DArray": 2,
            "sampler2D": 2,
            "sampler2DArray": 3,
            "sampler3D": 3,
        }.get(texture_type)

    def expected_texture_coordinate_count(self, texture_type):
        texture_type = self.sampled_texture_shape_type(texture_type)
        return {
            "sampler1D": 1,
            "sampler1DArray": 2,
            "sampler2D": 2,
            "sampler2DArray": 3,
            "sampler3D": 3,
            "samplerCube": 3,
            "samplerCubeArray": 4,
        }.get(texture_type)

    def texture_coordinate_rank_diagnostic(self, func_name, texture_type, raw_coord):
        expected_coord_count = self.expected_texture_coordinate_count(texture_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_sampled_resource_call(
                f"{func_name} coordinate rank",
                texture_type,
                [],
            )
        return None

    def expected_texture_gradient_coordinate_counts(self, texture_type):
        texture_type = self.sampled_texture_shape_type(texture_type)
        return {
            "sampler1D": {1},
            "sampler1DArray": {1},
            "sampler2D": {2},
            "sampler2DArray": {2},
            "sampler3D": {3, 4},
            "samplerCube": {3, 4},
            "samplerCubeArray": {3, 4},
        }.get(texture_type)

    def texture_gradient_rank_diagnostic(self, texture_type, *raw_gradients):
        expected_gradient_counts = self.expected_texture_gradient_coordinate_counts(
            texture_type
        )
        if expected_gradient_counts is None:
            return None

        for raw_gradient in raw_gradients:
            actual_gradient_count = self.texel_fetch_coordinate_count(raw_gradient)
            if (
                actual_gradient_count is not None
                and actual_gradient_count not in expected_gradient_counts
            ):
                return self.unsupported_sampled_resource_call(
                    "textureGrad derivative rank",
                    texture_type,
                    [],
                )
        return None

    def expected_projected_texture_coordinate_count(self, texture_type):
        texture_type = self.sampled_texture_shape_type(texture_type)
        return {
            "sampler1D": 2,
            "sampler2D": 3,
            "sampler3D": 4,
            "samplerCube": 4,
        }.get(texture_type)

    def projected_texture_coordinate_rank_diagnostic(
        self, func_name, texture_type, raw_coord
    ):
        expected_coord_count = self.expected_projected_texture_coordinate_count(
            texture_type
        )
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_sampled_resource_call(
                f"{func_name} coordinate rank",
                texture_type,
                [],
            )
        return None

    def projected_coord_args(self, texture_type, texture_name, coord):
        texture_type = self.sampled_texture_shape_type(texture_type)
        projection_component = {
            "sampler1D": "y",
            "sampler2D": "z",
            "sampler3D": "w",
            "samplerCube": "w",
        }.get(texture_type)
        if projection_component is None:
            return None

        projection = self.coord_component(coord, projection_component)
        components = ["x"]
        if texture_type in {"sampler2D", "sampler3D", "samplerCube"}:
            components.append("y")
        if texture_type in {"sampler3D", "samplerCube"}:
            components.append("z")
        projected_components = [
            f"({self.coord_component(coord, component)} / {projection})"
            for component in components
        ]
        return ", ".join([texture_name, *projected_components])

    def expected_texel_fetch_offset_count(self, texture_type):
        texture_type = self.sampled_texture_shape_type(texture_type)
        return {
            "sampler1D": 1,
            "sampler1DArray": 1,
            "sampler2D": 2,
            "sampler2DArray": 2,
            "sampler3D": 3,
        }.get(texture_type)

    def texel_fetch_offset_rank_diagnostic(self, texture_type, raw_offset):
        expected_offset_count = self.expected_texel_fetch_offset_count(texture_type)
        actual_offset_count = self.texel_fetch_coordinate_count(raw_offset)
        if (
            expected_offset_count is not None
            and actual_offset_count is not None
            and actual_offset_count != expected_offset_count
        ):
            return self.unsupported_sampled_resource_call(
                "texelFetchOffset offset rank",
                texture_type,
                [],
            )
        return None

    def offset_coord_component(self, coord, offset, component):
        return (
            f"({self.coord_component(coord, component)} + "
            f"{self.coord_component(offset, component)})"
        )

    def hip_texture_gradient_argument(self, texture_type, raw_gradient, gradient):
        texture_type = self.sampled_texture_shape_type(texture_type)
        if texture_type not in {"sampler3D", "samplerCube", "samplerCubeArray"}:
            return gradient

        gradient_count = self.texel_fetch_coordinate_count(raw_gradient)
        if gradient_count == 3:
            return (
                f"make_float4({self.coord_component(gradient, 'x')}, "
                f"{self.coord_component(gradient, 'y')}, "
                f"{self.coord_component(gradient, 'z')}, 0.0f)"
            )
        return gradient

    def expected_image_coordinate_count(self, image_type):
        image_type = self.resource_base_type(image_type)
        if not isinstance(image_type, str):
            return None
        if "1DArray" in image_type:
            return 2
        if "1D" in image_type:
            return 1
        if "CubeArray" in image_type:
            return 4
        if "3D" in image_type:
            return 3
        if "Cube" in image_type:
            return 3
        if "Array" in image_type:
            return 3
        if "2D" in image_type:
            return 2
        return None

    def unsupported_image_coordinate_rank_call(self, func_name, image_type):
        fallback = "((void)0)"
        if func_name == "imageLoad":
            fallback = self.zero_value_for_type(self.image_value_type(image_type))
        return self.refuse_lowering(
            f"{self.resource_backend_name()} image resource call: {func_name} coordinate rank on {image_type}",
            fallback,
        )

    def image_coordinate_rank_diagnostic(self, func_name, image_type, raw_coord):
        expected_coord_count = self.expected_image_coordinate_count(image_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_image_coordinate_rank_call(func_name, image_type)
        return None

    def image_resource_access(self, image_arg):
        if isinstance(image_arg, ArrayAccessNode):
            array_node = getattr(
                image_arg, "array", getattr(image_arg, "array_expr", None)
            )
            return self.image_resource_access(array_node)

        image_name = self.get_expression_name(image_arg)
        if not image_name:
            return None
        return self.image_resource_accesses.get(image_name)

    def unsupported_image_access_call(self, func_name, image_type, reason):
        image_type = image_type or "unknown resource"
        fallback = "((void)0)"
        if func_name == "imageLoad":
            fallback = self.zero_value_for_type(self.image_value_type(image_type))
        return self.refuse_lowering(
            f"{self.resource_backend_name()} image access: {func_name} {reason} on {image_type}",
            fallback,
        )

    def image_access_diagnostic(self, func_name, image_type, raw_image):
        access = self.image_resource_access(raw_image)
        if func_name == "imageLoad" and access == "writeonly":
            return self.unsupported_image_access_call(
                func_name,
                image_type,
                "requires readable image resource",
            )
        if func_name == "imageStore" and access == "readonly":
            return self.unsupported_image_access_call(
                func_name,
                image_type,
                "requires writable image resource",
            )
        return None

    def unsupported_image_atomic_coordinate_rank_call(self, func_name, image_type):
        image_type = image_type or "unknown resource"
        return self.refuse_lowering(
            f"{self.resource_backend_name()} image atomic resource call: {func_name} coordinate rank on {image_type}",
            self.image_atomic_zero_value(image_type),
        )

    def image_atomic_coordinate_rank_diagnostic(self, func_name, image_type, raw_coord):
        expected_coord_count = self.expected_image_coordinate_count(image_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_image_atomic_coordinate_rank_call(
                func_name, image_type
            )
        return None

    def unsupported_image_atomic_access_call(self, func_name, image_type):
        image_type = image_type or "unknown resource"
        return self.refuse_lowering(
            f"{self.resource_backend_name()} image atomic resource call: {func_name} requires readwrite image resource on {image_type}",
            self.image_atomic_zero_value(image_type),
        )

    def image_atomic_access_diagnostic(self, func_name, image_type, raw_image):
        if self.image_resource_access(raw_image) in {"readonly", "writeonly"}:
            return self.unsupported_image_atomic_access_call(func_name, image_type)
        return None

    def image_surface_x_offset(self, coord, value_type, image_type):
        if "1D" in image_type and "Array" not in image_type:
            return f"{coord} * sizeof({value_type})"
        return self.surface_x_offset(coord, value_type)

    def is_storage_image_type(self, type_name):
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        base_type = self.resource_base_type(type_name)
        if not isinstance(base_type, str):
            return False
        image_shapes = {
            "image1D",
            "image1DArray",
            "image2D",
            "image2DArray",
            "image3D",
            "imageCube",
            "imageCubeArray",
            "image2DMS",
            "image2DMSArray",
        }
        return base_type in image_shapes or any(
            base_type == f"{prefix}{shape}"
            for prefix in ("i", "u")
            for shape in image_shapes
        )

    def unsupported_scalar_resource_query_call(self, func_name, resource_type):
        resource_type = resource_type or "unknown resource"
        return self.refuse_lowering(
            f"{self.resource_backend_name()} resource query: {func_name} on {resource_type}",
            "0",
        )

    def unsupported_dimension_resource_query_call(self, func_name, resource_type):
        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return None
        return_type = self.query_return_type(spec["dimensions"])
        fallback = self.query_constructor(
            return_type,
            ["0"] * len(spec["dimensions"]),
        )
        return self.refuse_lowering(
            f"{self.resource_backend_name()} resource query: {func_name} on {resource_type}",
            fallback,
        )

    def generate_dimension_query(self, func_name, raw_args, args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if resource_type is None:
            return None
        if func_name == "textureSize" and not self.is_sampled_resource_type(
            resource_type
        ):
            return self.unsupported_dimension_resource_query_call(
                func_name, resource_type
            )
        if func_name == "imageSize" and not self.is_storage_image_type(resource_type):
            return self.unsupported_dimension_resource_query_call(
                func_name, resource_type
            )
        return ResourceQueryMixin.generate_dimension_query(
            self, func_name, raw_args, args
        )

    def generate_sample_count_query(self, func_name, raw_args, args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if resource_type is None:
            return None

        expected_resource = (
            self.is_sampled_resource_type(resource_type)
            if func_name == "textureSamples"
            else self.is_storage_image_type(resource_type)
        )
        sample_count_query = None
        if expected_resource:
            sample_count_query = ResourceQueryMixin.generate_sample_count_query(
                self, func_name, raw_args, args
            )
        if sample_count_query is not None:
            return sample_count_query
        return self.unsupported_scalar_resource_query_call(func_name, resource_type)

    def generate_texture_query_levels(self, raw_args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if resource_type is None:
            return None
        spec = self.dimension_query_spec(resource_type)
        if (
            not self.is_sampled_resource_type(resource_type)
            or spec is None
            or not spec["mip"]
        ):
            return self.unsupported_scalar_resource_query_call(
                "textureQueryLevels", resource_type
            )
        return ResourceQueryMixin.generate_texture_query_levels(self, raw_args)

    def is_sampler_state_resource_type(self, type_name):
        return self.resource_base_type(type_name) == "sampler"

    def has_explicit_sampler_argument(self, raw_args):
        if len(raw_args) < 3:
            return False
        return self.is_sampler_state_resource_type(
            self.resource_expression_type(raw_args[1])
        )

    def normalize_split_sampler_texture_args(self, func_name, raw_args, args):
        split_sampler_functions = {
            "texture",
            "textureLod",
            "textureGrad",
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
            "textureCompare",
            "textureCompareLod",
            "textureCompareGrad",
            "textureCompareOffset",
            "textureCompareLodOffset",
            "textureCompareGradOffset",
            "textureCompareProj",
            "textureCompareProjOffset",
            "textureCompareProjLod",
            "textureCompareProjLodOffset",
            "textureCompareProjGrad",
            "textureCompareProjGradOffset",
            "textureGatherCompare",
            "textureGatherCompareOffset",
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
            "textureQueryLod",
        }
        if (
            func_name not in split_sampler_functions
            or not self.has_explicit_sampler_argument(raw_args)
        ):
            return raw_args, args
        return [raw_args[0], *raw_args[2:]], [args[0], *args[2:]]

    def generate_texel_fetch_offset_call(self, raw_args, args):
        texture_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if texture_type is not None and not self.is_sampled_texture_family_type(
            texture_type
        ):
            return self.unsupported_sampled_resource_call(
                "texelFetchOffset", texture_type, args
            )
        if self.is_shadow_resource_type(texture_type):
            return self.unsupported_shadow_resource_call(
                "texelFetchOffset", texture_type, args
            )
        if self.is_multisample_resource_type(texture_type):
            return self.unsupported_multisample_resource_call(
                "texelFetchOffset", texture_type, args
            )
        texture_shape = self.sampled_texture_shape_type(texture_type)
        if texture_shape in {"samplerCube", "samplerCubeArray"}:
            return self.unsupported_sampled_resource_call(
                "texelFetchOffset", texture_type, args
            )

        coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
            "texelFetchOffset",
            texture_type,
            raw_args[1],
        )
        if coordinate_diagnostic is not None:
            return coordinate_diagnostic

        offset_diagnostic = self.texel_fetch_offset_rank_diagnostic(
            texture_type,
            raw_args[3],
        )
        if offset_diagnostic is not None:
            return offset_diagnostic

        texture_name = args[0]
        coord = args[1]
        offset = args[3]
        value_type = self.sampled_texture_value_type(texture_type)
        if texture_shape == "sampler1D":
            return f"tex1Dfetch<{value_type}>({texture_name}, ({coord} + {offset}))"
        if texture_shape == "sampler1DArray":
            return (
                f"tex1DLayered<{value_type}>({texture_name}, "
                f"({self.coord_component(coord, 'x')} + {offset}), "
                f"{self.coord_component(coord, 'y')})"
            )
        if texture_shape == "sampler2D":
            return (
                f"tex2D<{value_type}>({texture_name}, "
                f"{self.offset_coord_component(coord, offset, 'x')}, "
                f"{self.offset_coord_component(coord, offset, 'y')})"
            )
        if texture_shape == "sampler2DArray":
            return (
                f"tex2DLayered<{value_type}>({texture_name}, "
                f"{self.offset_coord_component(coord, offset, 'x')}, "
                f"{self.offset_coord_component(coord, offset, 'y')}, "
                f"{self.coord_component(coord, 'z')})"
            )
        if texture_shape == "sampler3D":
            return (
                f"tex3D<{value_type}>({texture_name}, "
                f"{self.offset_coord_component(coord, offset, 'x')}, "
                f"{self.offset_coord_component(coord, offset, 'y')}, "
                f"{self.offset_coord_component(coord, offset, 'z')})"
            )
        return None

    def generate_projected_texture_call(self, func_name, raw_args, args):
        texture_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if texture_type is None:
            return None
        if self.is_shadow_resource_type(texture_type):
            return self.unsupported_shadow_resource_call(func_name, texture_type, args)
        if self.is_multisample_resource_type(texture_type):
            return self.unsupported_multisample_resource_call(
                func_name, texture_type, args
            )

        coord = args[1]
        coordinate_diagnostic = self.projected_texture_coordinate_rank_diagnostic(
            func_name, texture_type, raw_args[1]
        )
        if coordinate_diagnostic is not None:
            return coordinate_diagnostic

        grad_x = None
        grad_y = None
        if func_name == "textureProjGrad" and len(args) >= 4:
            gradient_diagnostic = self.texture_gradient_rank_diagnostic(
                texture_type,
                raw_args[2],
                raw_args[3],
            )
            if gradient_diagnostic is not None:
                return gradient_diagnostic
            grad_x = self.hip_texture_gradient_argument(
                texture_type,
                raw_args[2],
                args[2],
            )
            grad_y = self.hip_texture_gradient_argument(
                texture_type,
                raw_args[3],
                args[3],
            )

        texture_name = args[0]
        coord_args = self.projected_coord_args(texture_type, texture_name, coord)
        if coord_args is None:
            return self.unsupported_sampled_resource_call(func_name, texture_type, args)
        texture_shape = self.sampled_texture_shape_type(texture_type)
        value_type = self.sampled_texture_value_type(texture_type)

        if texture_shape == "sampler1D":
            if func_name == "textureProj":
                return f"tex1D<{value_type}>({coord_args})"
            if func_name == "textureProjLod" and len(args) >= 3:
                return f"tex1DLod<{value_type}>({coord_args}, {args[2]})"
            if func_name == "textureProjGrad" and grad_x is not None:
                return f"tex1DGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

        if texture_shape == "sampler2D":
            if func_name == "textureProj":
                return f"tex2D<{value_type}>({coord_args})"
            if func_name == "textureProjLod" and len(args) >= 3:
                return f"tex2DLod<{value_type}>({coord_args}, {args[2]})"
            if func_name == "textureProjGrad" and grad_x is not None:
                return f"tex2DGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

        if texture_shape == "sampler3D":
            if func_name == "textureProj":
                return f"tex3D<{value_type}>({coord_args})"
            if func_name == "textureProjLod" and len(args) >= 3:
                return f"tex3DLod<{value_type}>({coord_args}, {args[2]})"
            if func_name == "textureProjGrad" and grad_x is not None:
                return f"tex3DGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

        if texture_shape == "samplerCube":
            if func_name == "textureProj":
                return f"texCubemap<{value_type}>({coord_args})"
            if func_name == "textureProjLod" and len(args) >= 3:
                return f"texCubemapLod<{value_type}>({coord_args}, {args[2]})"
            if func_name == "textureProjGrad" and grad_x is not None:
                return f"texCubemapGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

        return self.unsupported_sampled_resource_call(func_name, texture_type, args)

    def generate_texture_gather_call(self, raw_args, args):
        texture_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if texture_type != "sampler2D":
            return None

        coordinate_index = 1
        component = None
        component_arg = None
        if len(args) == 2:
            pass
        elif len(args) == 3:
            component = args[2]
            component_arg = raw_args[2]
        else:
            return None

        coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
            "textureGather",
            texture_type,
            raw_args[coordinate_index],
        )
        if coordinate_diagnostic is not None:
            return coordinate_diagnostic

        component_diagnostic = self.texture_gather_component_diagnostic(
            texture_type, component_arg
        )
        if component_diagnostic is not None:
            return component_diagnostic

        texture_name = args[0]
        coord = args[coordinate_index]
        gather_args = (
            f"{texture_name}, "
            f"{self.coord_component(coord, 'x')}, "
            f"{self.coord_component(coord, 'y')}"
        )
        if component is not None:
            return f"tex2Dgather<float4>({gather_args}, {component})"
        return f"tex2Dgather<float4>({gather_args})"

    def texture_gather_component_diagnostic(self, texture_type, component_arg):
        if not self.is_texture_gather_literal_component(component_arg):
            return None
        component_value = self.literal_int_value(component_arg)
        if component_value in {0, 1, 2, 3}:
            return None
        return self.unsupported_sampled_resource_call(
            "textureGather component literal must be 0, 1, 2, or 3",
            texture_type,
            [],
        )

    def is_texture_gather_literal_component(self, node):
        if isinstance(node, LiteralNode):
            return True
        if isinstance(node, UnaryOpNode) and not getattr(node, "is_postfix", False):
            operator = getattr(node, "operator", getattr(node, "op", None))
            if operator in {"+", "-"}:
                return self.is_texture_gather_literal_component(node.operand)
        return False
