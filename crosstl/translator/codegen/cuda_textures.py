"""Texture sampling and fetch lowering of the CUDA generator, plus image access diagnostics."""

from ..ast import (
    ArrayAccessNode,
    FunctionCallNode,
    MemberAccessNode,
    PointerAccessNode,
)


class CudaTextureMixin:
    """Texture members of ``CudaCodeGen``."""

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
        """Return the CUDA vector type read from a sampled texture."""
        texture_type = self.resource_base_type(texture_type)
        if isinstance(texture_type, str) and texture_type.startswith("isampler"):
            return "int4"
        if isinstance(texture_type, str) and texture_type.startswith("usampler"):
            return "uint4"
        return "float4"

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

    def texture_offset_argument_index(self, func_name):
        return {
            "textureOffset": 2,
            "textureLodOffset": 3,
            "textureGradOffset": 4,
            "textureProjOffset": 2,
            "textureProjLodOffset": 3,
            "textureProjGradOffset": 4,
            "texelFetchOffset": 3,
        }.get(func_name)

    def expected_texture_offset_coordinate_count(self, texture_type):
        texture_type = self.sampled_texture_shape_type(texture_type)
        return {
            "sampler1D": 1,
            "sampler1DArray": 1,
            "sampler2D": 2,
            "sampler2DArray": 2,
            "sampler3D": 3,
        }.get(texture_type)

    def texture_offset_rank_diagnostic(self, func_name, texture_type, raw_offset):
        expected_offset_count = self.expected_texture_offset_coordinate_count(
            texture_type
        )
        actual_offset_count = self.texel_fetch_coordinate_count(raw_offset)
        if (
            expected_offset_count is not None
            and actual_offset_count is not None
            and actual_offset_count != expected_offset_count
        ):
            return self.unsupported_sampled_resource_call(
                f"{func_name} offset rank",
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

    def sampled_texture_sampler_argument_offset(self, raw_args):
        if len(raw_args) > 2 and self.is_sampler_state_argument(raw_args[1]):
            return 1
        return 0

    def cuda_texture_gradient_argument(self, texture_type, raw_gradient, gradient):
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

    def cuda_texture_coord_components(self, texture_type, coord):
        """Return CUDA texture coordinate components for a sampled resource."""
        texture_type = self.sampled_texture_shape_type(texture_type)
        if texture_type == "sampler1D":
            return [coord]
        if texture_type == "sampler1DArray":
            return [self.coord_component(coord, "x"), self.coord_component(coord, "y")]
        if texture_type == "sampler2D":
            return [self.coord_component(coord, "x"), self.coord_component(coord, "y")]
        if texture_type == "sampler2DArray":
            return [
                self.coord_component(coord, "x"),
                self.coord_component(coord, "y"),
                self.coord_component(coord, "z"),
            ]
        if texture_type in {"sampler3D", "samplerCube"}:
            return [
                self.coord_component(coord, "x"),
                self.coord_component(coord, "y"),
                self.coord_component(coord, "z"),
            ]
        if texture_type == "samplerCubeArray":
            return [
                self.coord_component(coord, "x"),
                self.coord_component(coord, "y"),
                self.coord_component(coord, "z"),
                self.coord_component(coord, "w"),
            ]
        return None

    def cuda_texture_offset_components(self, texture_type, coord, offset):
        """Return texture coordinate components with a GLSL-style texel offset."""
        components = self.cuda_texture_coord_components(texture_type, coord)
        if components is None:
            return None

        offset_component_count = self.expected_texture_offset_coordinate_count(
            texture_type
        )
        if offset_component_count is None:
            return None

        if offset_component_count == 1:
            components[0] = f"({components[0]} + {offset})"
            return components

        for index, component in enumerate(("x", "y", "z")[:offset_component_count]):
            components[index] = (
                f"({components[index]} + {self.coord_component(offset, component)})"
            )
        return components

    def cuda_projected_texture_coord_components(self, texture_type, raw_coord, coord):
        """Return projected CUDA texture coordinate components."""
        texture_type = self.sampled_texture_shape_type(texture_type)
        coord_count = self.texel_fetch_coordinate_count(raw_coord)

        if texture_type == "sampler1D" and coord_count in {2, 4}:
            divisor = "y" if coord_count == 2 else "w"
            return [
                f"({self.coord_component(coord, 'x')} / "
                f"{self.coord_component(coord, divisor)})"
            ]

        if texture_type == "sampler2D" and coord_count in {3, 4}:
            divisor = "z" if coord_count == 3 else "w"
            return [
                f"({self.coord_component(coord, 'x')} / "
                f"{self.coord_component(coord, divisor)})",
                f"({self.coord_component(coord, 'y')} / "
                f"{self.coord_component(coord, divisor)})",
            ]

        if texture_type == "sampler2DArray" and coord_count == 4:
            return [
                f"({self.coord_component(coord, 'x')} / "
                f"{self.coord_component(coord, 'w')})",
                f"({self.coord_component(coord, 'y')} / "
                f"{self.coord_component(coord, 'w')})",
                self.coord_component(coord, "z"),
            ]

        if texture_type in {"sampler3D", "samplerCube"} and coord_count == 4:
            return [
                f"({self.coord_component(coord, 'x')} / "
                f"{self.coord_component(coord, 'w')})",
                f"({self.coord_component(coord, 'y')} / "
                f"{self.coord_component(coord, 'w')})",
                f"({self.coord_component(coord, 'z')} / "
                f"{self.coord_component(coord, 'w')})",
            ]

        return None

    def cuda_apply_texture_offset_components(self, texture_type, components, offset):
        """Apply a texture offset to existing coordinate components."""
        offset_component_count = self.expected_texture_offset_coordinate_count(
            texture_type
        )
        if offset_component_count is None:
            return None

        components = list(components)
        if offset_component_count == 1:
            components[0] = f"({components[0]} + {offset})"
            return components

        for index, component in enumerate(("x", "y", "z")[:offset_component_count]):
            components[index] = (
                f"({components[index]} + {self.coord_component(offset, component)})"
            )
        return components

    def cuda_texture_sample_call_from_components(
        self,
        func_name,
        texture_type,
        texture_name,
        components,
        lod=None,
        grad_x=None,
        grad_y=None,
    ):
        """Return a CUDA texture sample call from precomputed components."""
        value_type = self.sampled_texture_value_type(texture_type)
        texture_type = self.sampled_texture_shape_type(texture_type)
        if texture_type == "sampler1D" and len(components) >= 1:
            coord_args = f"{texture_name}, {components[0]}"
            if func_name == "texture":
                return f"tex1D<{value_type}>({coord_args})"
            if func_name == "textureLod" and lod is not None:
                return f"tex1DLod<{value_type}>({coord_args}, {lod})"
            if func_name == "textureGrad" and grad_x is not None:
                return f"tex1DGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

        if texture_type == "sampler1DArray" and len(components) >= 2:
            coord_args = f"{texture_name}, {components[0]}, {components[1]}"
            if func_name == "texture":
                return f"tex1DLayered<{value_type}>({coord_args})"
            if func_name == "textureLod" and lod is not None:
                return f"tex1DLayeredLod<{value_type}>({coord_args}, {lod})"
            if func_name == "textureGrad" and grad_x is not None:
                return (
                    f"tex1DLayeredGrad<{value_type}>"
                    f"({coord_args}, {grad_x}, {grad_y})"
                )

        if texture_type == "sampler2D" and len(components) >= 2:
            coord_args = f"{texture_name}, {components[0]}, {components[1]}"
            if func_name == "texture":
                return f"tex2D<{value_type}>({coord_args})"
            if func_name == "textureLod" and lod is not None:
                return f"tex2DLod<{value_type}>({coord_args}, {lod})"
            if func_name == "textureGrad" and grad_x is not None:
                return f"tex2DGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

        if texture_type == "sampler2DArray" and len(components) >= 3:
            coord_args = (
                f"{texture_name}, {components[0]}, {components[1]}, {components[2]}"
            )
            if func_name == "texture":
                return f"tex2DLayered<{value_type}>({coord_args})"
            if func_name == "textureLod" and lod is not None:
                return f"tex2DLayeredLod<{value_type}>({coord_args}, {lod})"
            if func_name == "textureGrad" and grad_x is not None:
                return (
                    f"tex2DLayeredGrad<{value_type}>"
                    f"({coord_args}, {grad_x}, {grad_y})"
                )

        if texture_type == "sampler3D" and len(components) >= 3:
            coord_args = (
                f"{texture_name}, {components[0]}, {components[1]}, {components[2]}"
            )
            if func_name == "texture":
                return f"tex3D<{value_type}>({coord_args})"
            if func_name == "textureLod" and lod is not None:
                return f"tex3DLod<{value_type}>({coord_args}, {lod})"
            if func_name == "textureGrad" and grad_x is not None:
                return f"tex3DGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

        if texture_type == "samplerCube" and len(components) >= 3:
            coord_args = (
                f"{texture_name}, {components[0]}, {components[1]}, {components[2]}"
            )
            if func_name == "texture":
                return f"texCubemap<{value_type}>({coord_args})"
            if func_name == "textureLod" and lod is not None:
                return f"texCubemapLod<{value_type}>({coord_args}, {lod})"
            if func_name == "textureGrad" and grad_x is not None:
                return (
                    f"texCubemapGrad<{value_type}>"
                    f"({coord_args}, {grad_x}, {grad_y})"
                )

        if texture_type == "samplerCubeArray" and len(components) >= 4:
            coord_args = (
                f"{texture_name}, {components[0]}, {components[1]}, "
                f"{components[2]}, {components[3]}"
            )
            if func_name == "texture":
                return f"texCubemapLayered<{value_type}>({coord_args})"
            if func_name == "textureLod" and lod is not None:
                return f"texCubemapLayeredLod<{value_type}>({coord_args}, {lod})"
            if func_name == "textureGrad" and grad_x is not None:
                return (
                    f"texCubemapLayeredGrad<{value_type}>"
                    f"({coord_args}, {grad_x}, {grad_y})"
                )

        return None

    def cuda_texel_fetch_call_from_components(
        self, texture_type, texture_name, components
    ):
        """Return a CUDA texel-fetch expression from precomputed components."""
        value_type = self.sampled_texture_value_type(texture_type)
        texture_type = self.sampled_texture_shape_type(texture_type)
        if texture_type == "sampler1D" and len(components) >= 1:
            return f"tex1Dfetch<{value_type}>({texture_name}, {components[0]})"
        if texture_type == "sampler1DArray" and len(components) >= 2:
            return (
                f"tex1DLayered<{value_type}>"
                f"({texture_name}, {components[0]}, {components[1]})"
            )
        if texture_type == "sampler2D" and len(components) >= 2:
            return (
                f"tex2D<{value_type}>"
                f"({texture_name}, {components[0]}, {components[1]})"
            )
        if texture_type == "sampler2DArray" and len(components) >= 3:
            return (
                f"tex2DLayered<{value_type}>"
                f"({texture_name}, {components[0]}, {components[1]}, {components[2]})"
            )
        if texture_type == "sampler3D" and len(components) >= 3:
            return (
                f"tex3D<{value_type}>"
                f"({texture_name}, {components[0]}, {components[1]}, {components[2]})"
            )
        return None

    def generate_texture_offset_call(self, func_name, raw_args, args):
        """Lower textureOffset/textureLodOffset/textureGradOffset to CUDA calls."""
        texture_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        if texture_type is None:
            return None
        if self.is_shadow_resource_type(texture_type):
            return self.unsupported_shadow_resource_call(func_name, texture_type, args)
        if self.is_multisample_resource_type(texture_type):
            return self.unsupported_multisample_resource_call(
                func_name, texture_type, args
            )
        if texture_type not in {
            "sampler1D",
            "sampler1DArray",
            "sampler2D",
            "sampler2DArray",
            "sampler3D",
        }:
            return self.unsupported_sampled_resource_call(func_name, texture_type, args)

        sampler_arg_offset = self.sampled_texture_sampler_argument_offset(raw_args)
        coord_index = 1 + sampler_arg_offset
        if len(args) <= coord_index:
            return None

        coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
            func_name, texture_type, raw_args[coord_index]
        )
        if coordinate_diagnostic is not None:
            return coordinate_diagnostic

        texture_name = args[0]
        coord = args[coord_index]
        lod = None
        grad_x = None
        grad_y = None
        offset_index = None
        sample_func = "texture"

        if func_name == "textureOffset":
            offset_index = coord_index + 1
            if len(args) > offset_index + 1:
                return self.unsupported_sampled_resource_call(
                    "textureOffset bias", texture_type, args
                )
        elif func_name == "textureLodOffset":
            sample_func = "textureLod"
            lod_index = coord_index + 1
            offset_index = coord_index + 2
            if len(args) <= lod_index:
                return None
            lod = args[lod_index]
        elif func_name == "textureGradOffset":
            sample_func = "textureGrad"
            grad_x_index = coord_index + 1
            grad_y_index = coord_index + 2
            offset_index = coord_index + 3
            if len(args) <= grad_y_index:
                return None
            gradient_diagnostic = self.texture_gradient_rank_diagnostic(
                texture_type,
                raw_args[grad_x_index],
                raw_args[grad_y_index],
            )
            if gradient_diagnostic is not None:
                return gradient_diagnostic
            grad_x = self.cuda_texture_gradient_argument(
                texture_type,
                raw_args[grad_x_index],
                args[grad_x_index],
            )
            grad_y = self.cuda_texture_gradient_argument(
                texture_type,
                raw_args[grad_y_index],
                args[grad_y_index],
            )

        if offset_index is None or len(args) <= offset_index:
            return None

        offset_diagnostic = self.texture_offset_rank_diagnostic(
            func_name, texture_type, raw_args[offset_index]
        )
        if offset_diagnostic is not None:
            return offset_diagnostic

        components = self.cuda_texture_offset_components(
            texture_type, coord, args[offset_index]
        )
        if components is None:
            return self.unsupported_sampled_resource_call(func_name, texture_type, args)

        return self.cuda_texture_sample_call_from_components(
            sample_func,
            texture_type,
            texture_name,
            components,
            lod=lod,
            grad_x=grad_x,
            grad_y=grad_y,
        )

    def generate_texture_projected_call(self, func_name, raw_args, args):
        """Lower supported projected texture operations to CUDA texture calls."""
        texture_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        if texture_type is None:
            return None
        if self.is_shadow_resource_type(texture_type):
            return self.unsupported_shadow_resource_call(func_name, texture_type, args)
        if self.is_multisample_resource_type(texture_type):
            return self.unsupported_multisample_resource_call(
                func_name, texture_type, args
            )
        if texture_type not in {
            "sampler1D",
            "sampler2D",
            "sampler2DArray",
            "sampler3D",
            "samplerCube",
        }:
            return self.unsupported_sampled_resource_call(func_name, texture_type, args)

        sampler_arg_offset = self.sampled_texture_sampler_argument_offset(raw_args)
        coord_index = 1 + sampler_arg_offset
        if len(args) <= coord_index:
            return None

        components = self.cuda_projected_texture_coord_components(
            texture_type, raw_args[coord_index], args[coord_index]
        )
        if components is None:
            return self.unsupported_sampled_resource_call(
                f"{func_name} coordinate rank", texture_type, args
            )

        texture_name = args[0]
        sample_func = "texture"
        lod = None
        grad_x = None
        grad_y = None
        offset_index = None

        if func_name == "textureProj":
            if len(args) > coord_index + 1:
                return self.unsupported_sampled_resource_call(
                    "textureProj bias", texture_type, args
                )
        elif func_name == "textureProjOffset":
            offset_index = coord_index + 1
            if len(args) > offset_index + 1:
                return self.unsupported_sampled_resource_call(
                    "textureProjOffset bias", texture_type, args
                )
        elif func_name == "textureProjLod":
            sample_func = "textureLod"
            lod_index = coord_index + 1
            if len(args) <= lod_index:
                return None
            lod = args[lod_index]
        elif func_name == "textureProjLodOffset":
            sample_func = "textureLod"
            lod_index = coord_index + 1
            offset_index = coord_index + 2
            if len(args) <= lod_index:
                return None
            lod = args[lod_index]
        elif func_name == "textureProjGrad":
            sample_func = "textureGrad"
            grad_x_index = coord_index + 1
            grad_y_index = coord_index + 2
            if len(args) <= grad_y_index:
                return None
            gradient_diagnostic = self.texture_gradient_rank_diagnostic(
                texture_type,
                raw_args[grad_x_index],
                raw_args[grad_y_index],
            )
            if gradient_diagnostic is not None:
                return gradient_diagnostic
            grad_x = self.cuda_texture_gradient_argument(
                texture_type,
                raw_args[grad_x_index],
                args[grad_x_index],
            )
            grad_y = self.cuda_texture_gradient_argument(
                texture_type,
                raw_args[grad_y_index],
                args[grad_y_index],
            )
        elif func_name == "textureProjGradOffset":
            sample_func = "textureGrad"
            grad_x_index = coord_index + 1
            grad_y_index = coord_index + 2
            offset_index = coord_index + 3
            if len(args) <= grad_y_index:
                return None
            gradient_diagnostic = self.texture_gradient_rank_diagnostic(
                texture_type,
                raw_args[grad_x_index],
                raw_args[grad_y_index],
            )
            if gradient_diagnostic is not None:
                return gradient_diagnostic
            grad_x = self.cuda_texture_gradient_argument(
                texture_type,
                raw_args[grad_x_index],
                args[grad_x_index],
            )
            grad_y = self.cuda_texture_gradient_argument(
                texture_type,
                raw_args[grad_y_index],
                args[grad_y_index],
            )

        if offset_index is not None:
            if len(args) <= offset_index:
                return None
            if texture_type == "samplerCube":
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )
            offset_diagnostic = self.texture_offset_rank_diagnostic(
                func_name, texture_type, raw_args[offset_index]
            )
            if offset_diagnostic is not None:
                return offset_diagnostic
            components = self.cuda_apply_texture_offset_components(
                texture_type, components, args[offset_index]
            )
            if components is None:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

        return self.cuda_texture_sample_call_from_components(
            sample_func,
            texture_type,
            texture_name,
            components,
            lod=lod,
            grad_x=grad_x,
            grad_y=grad_y,
        )

    def generate_texel_fetch_offset_call(self, raw_args, args):
        """Lower texelFetchOffset by applying the offset before CUDA fetch."""
        texture_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        if texture_type is None:
            return None
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

        sampler_arg_offset = self.sampled_texture_sampler_argument_offset(raw_args)
        coord_index = 1 + sampler_arg_offset
        offset_index = coord_index + 2
        if len(args) <= offset_index:
            return None

        expected_coord_count = self.expected_texel_fetch_coordinate_count(texture_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_args[coord_index])
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_sampled_resource_call(
                "texelFetchOffset coordinate rank",
                texture_type,
                args,
            )

        offset_diagnostic = self.texture_offset_rank_diagnostic(
            "texelFetchOffset", texture_type, raw_args[offset_index]
        )
        if offset_diagnostic is not None:
            return offset_diagnostic

        components = self.cuda_texture_offset_components(
            texture_shape, args[coord_index], args[offset_index]
        )
        if components is None:
            return self.unsupported_sampled_resource_call(
                "texelFetchOffset", texture_type, args
            )
        return self.cuda_texel_fetch_call_from_components(
            texture_type, args[0], components
        )

    def expected_image_coordinate_count(self, image_type):
        image_type = self.resource_base_type(image_type)
        if not isinstance(image_type, str):
            return None
        if "1DArray" in image_type:
            return 2
        if "1D" in image_type:
            return 1
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
            f"CUDA image resource call: {func_name} coordinate rank on {image_type}",
            fallback,
        )

    def image_resource_access(self, image_arg):
        if isinstance(image_arg, FunctionCallNode):
            callee_name = self.raw_function_call_name(image_arg)
            return_source = self.query_return_sources.get(callee_name)
            if return_source is None:
                return None
            raw_args = getattr(
                image_arg,
                "arguments",
                getattr(image_arg, "args", []),
            )
            return self.query_return_source_image_access(return_source, raw_args)

        if isinstance(image_arg, ArrayAccessNode):
            array_node = getattr(
                image_arg, "array", getattr(image_arg, "array_expr", None)
            )
            return self.image_resource_access(array_node)

        if isinstance(image_arg, MemberAccessNode):
            object_node = getattr(
                image_arg,
                "object_expr",
                getattr(image_arg, "object", None),
            )
            object_type = self.struct_member_lookup_type(
                self.expression_result_type(object_node)
            )
            member = getattr(image_arg, "member", None)
            access = self.struct_member_image_accesses.get(object_type, {}).get(member)
            if access is not None:
                return access

        if isinstance(image_arg, PointerAccessNode):
            pointer_expr = getattr(image_arg, "pointer_expr", None)
            pointer_type = self.expression_result_type(pointer_expr)
            pointer_info = self.cuda_indirect_type_info(pointer_type)
            if pointer_info is not None:
                object_type = self.struct_member_lookup_type(
                    pointer_info["pointee_type"]
                )
                member = getattr(image_arg, "member", None)
                access = self.struct_member_image_accesses.get(object_type, {}).get(
                    member
                )
                if access is not None:
                    return access

        image_name = self.get_expression_name(image_arg)
        if not image_name:
            return None
        return self.image_resource_accesses.get(image_name)

    def query_return_source_image_access(self, return_source, raw_args):
        """Return storage-image access for a traceable returned resource."""
        kind = return_source.get("kind")
        if kind == "ternary":
            true_access = self.query_return_source_image_access(
                return_source["true_source"],
                raw_args,
            )
            false_access = self.query_return_source_image_access(
                return_source["false_source"],
                raw_args,
            )
            if true_access is None or true_access != false_access:
                return None
            return true_access

        if kind == "global":
            return self.image_resource_access(return_source.get("name"))

        if kind == "parameter":
            index = return_source.get("index")
            if index is None or index >= len(raw_args):
                return None
            return self.image_resource_access(raw_args[index])

        if kind == "member":
            object_type = return_source.get("object_type")
            member = return_source.get("member")
            return self.struct_member_image_accesses.get(object_type, {}).get(member)

        return None

    def unsupported_image_access_call(self, func_name, image_type, reason):
        image_type = image_type or "unknown resource"
        fallback = "((void)0)"
        if func_name == "imageLoad":
            fallback = self.zero_value_for_type(self.image_value_type(image_type))
        return self.refuse_lowering(
            f"CUDA image access: {func_name} {reason} on {image_type}", fallback
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

    def unsupported_image_atomic_coordinate_rank_call(self, func_name, image_type):
        image_type = image_type or "unknown resource"
        return self.refuse_lowering(
            f"CUDA image atomic resource call: {func_name} coordinate rank on {image_type}",
            self.image_atomic_zero_value(image_type),
        )

    def unsupported_image_atomic_access_call(self, func_name, image_type):
        image_type = image_type or "unknown resource"
        return self.refuse_lowering(
            f"CUDA image atomic resource call: {func_name} requires readwrite image resource on {image_type}",
            self.image_atomic_zero_value(image_type),
        )

    def image_atomic_access_diagnostic(self, func_name, image_type, raw_image):
        if self.image_resource_access(raw_image) in {"readonly", "writeonly"}:
            return self.unsupported_image_atomic_access_call(func_name, image_type)
        return None

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

    def unsupported_scalar_resource_query_call(self, func_name, resource_type):
        resource_type = resource_type or "unknown resource"
        return self.refuse_lowering(
            f"{self.resource_backend_name()} resource query: {func_name} on {resource_type}",
            "0",
        )
