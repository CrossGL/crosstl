"""Texture and storage-image calls of the CUDA generator: size, sample-count, level and LOD queries, the refused and specially lowered sampling forms, offset and projective sampling, image atomics, implicit-LOD, explicit-LOD and gradient sampling, texel fetches, and image loads and stores."""

from ..unhandled import (
    UNHANDLED,
)
from .cuda_surface_texels import UnsupportedSurfaceTexel


class CudaResourceCallMixin:
    """Texture and storage-image members of ``CudaCodeGen.generate_resource_call``."""

    def cuda_texture_query_call(
        self,
        func_name,
        raw_args,
        args,
    ):
        """Emit texture and image size, sample-count, level-count and LOD queries."""
        if func_name in {"textureSize", "imageSize"}:
            return self.generate_dimension_query(func_name, raw_args, args)

        if func_name in {"textureSamples", "imageSamples"}:
            sample_count_query = self.generate_sample_count_query(
                func_name, raw_args, args
            )
            if sample_count_query is not None:
                return sample_count_query
            if raw_args:
                resource_type = self.resource_base_type(
                    self.resource_expression_type(raw_args[0])
                )
                if resource_type is not None:
                    return self.unsupported_scalar_resource_query_call(
                        func_name, resource_type
                    )
            return None

        if func_name == "textureQueryLevels":
            return self.generate_texture_query_levels(raw_args)

        if func_name == "textureQueryLod" and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_resource_query_call(
                    func_name, texture_type, args
                )
        return UNHANDLED

    def cuda_texture_form_call(
        self,
        func_name,
        raw_args,
        args,
    ):
        """Refuse or lower the gather, compare, offset, projective and offset-fetch forms that need special handling."""
        if (
            func_name in {
                "texture",
                "textureLod",
                "textureGrad",
                "textureGather",
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
            }
            and len(args) >= 2
        ):
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if self.is_shadow_resource_type(texture_type):
                return self.unsupported_shadow_resource_call(
                    func_name, texture_type, args
                )

        if func_name in {"textureGather", "textureGatherOffset"} and len(args) >= 2:
            texture_gather = self.generate_texture_gather_call(
                func_name, raw_args, args
            )
            if texture_gather is not None:
                return texture_gather

        if (
            func_name in {
                "textureGather",
                "textureGatherOffset",
                "textureGatherOffsets",
            }
            and len(args) >= 2
        ):
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

        if (
            func_name in {
                "textureOffset",
                "textureLodOffset",
                "textureGradOffset",
            }
            and raw_args
        ):
            return self.generate_texture_offset_call(func_name, raw_args, args)

        if (
            func_name in {
                "textureProj",
                "textureProjOffset",
                "textureProjLod",
                "textureProjLodOffset",
                "textureProjGrad",
                "textureProjGradOffset",
            }
            and raw_args
        ):
            return self.generate_texture_projected_call(func_name, raw_args, args)

        if func_name == "texelFetchOffset" and raw_args:
            return self.generate_texel_fetch_offset_call(raw_args, args)
        return UNHANDLED

    def cuda_offset_texture_call(
        self,
        func_name,
        raw_args,
        args,
    ):
        """Emit offset and projective texture sampling."""
        if (
            func_name in {
                "textureOffset",
                "textureLodOffset",
                "textureGradOffset",
                "textureProj",
                "textureProjOffset",
                "textureProjLod",
                "textureProjLodOffset",
                "textureProjGrad",
                "textureProjGradOffset",
                "texelFetchOffset",
            }
            and raw_args
        ):
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if texture_type is not None:
                if self.is_shadow_resource_type(texture_type):
                    return self.unsupported_shadow_resource_call(
                        func_name, texture_type, args
                    )
                if self.is_multisample_resource_type(texture_type):
                    return self.unsupported_multisample_resource_call(
                        func_name, texture_type, args
                    )
                offset_arg_index = self.texture_offset_argument_index(func_name)
                if offset_arg_index is not None:
                    offset_arg_index += self.sampled_texture_sampler_argument_offset(
                        raw_args
                    )
                if offset_arg_index is not None and len(raw_args) > offset_arg_index:
                    offset_diagnostic = self.texture_offset_rank_diagnostic(
                        func_name,
                        texture_type,
                        raw_args[offset_arg_index],
                    )
                    if offset_diagnostic is not None:
                        return offset_diagnostic
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )
        return UNHANDLED

    def cuda_image_atomic_call(
        self,
        func_name,
        raw_args,
        args,
    ):
        """Emit atomic operations on storage images."""
        if func_name in {
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        }:
            image_type = None
            if raw_args:
                image_type = self.resource_base_type(
                    self.resource_expression_type(raw_args[0])
                )
                access_diagnostic = self.image_atomic_access_diagnostic(
                    func_name, image_type, raw_args[0]
                )
                if access_diagnostic is not None:
                    return access_diagnostic
            if len(raw_args) >= 2:
                coordinate_diagnostic = self.image_atomic_coordinate_rank_diagnostic(
                    func_name, image_type, raw_args[1]
                )
                if coordinate_diagnostic is not None:
                    return coordinate_diagnostic
            return self.unsupported_image_atomic_resource_call(
                func_name, image_type, args
            )
        return UNHANDLED

    def cuda_sampled_texture_call(
        self,
        func_name,
        raw_args,
        args,
    ):
        """Emit implicit-LOD, explicit-LOD and gradient texture sampling."""
        if func_name in {"texture", "textureLod", "textureGrad"} and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            texture_shape = self.sampled_texture_shape_type(texture_type)
            value_type = self.sampled_texture_value_type(texture_type)
            if self.is_multisample_resource_type(texture_type):
                return self.unsupported_multisample_resource_call(
                    func_name, texture_type, args
                )
            sampler_arg_offset = self.sampled_texture_sampler_argument_offset(raw_args)
            coord_index = 1 + sampler_arg_offset
            if len(args) <= coord_index:
                return None
            coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
                func_name, texture_type, raw_args[coord_index]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            if func_name == "texture" and len(args) > coord_index + 1:
                return self.unsupported_sampled_resource_call(
                    "texture bias", texture_type, args
                )
            grad_x = None
            grad_y = None
            if func_name == "textureGrad" and len(args) > coord_index + 2:
                gradient_diagnostic = self.texture_gradient_rank_diagnostic(
                    texture_type,
                    raw_args[coord_index + 1],
                    raw_args[coord_index + 2],
                )
                if gradient_diagnostic is not None:
                    return gradient_diagnostic
                grad_x = self.cuda_texture_gradient_argument(
                    texture_type,
                    raw_args[coord_index + 1],
                    args[coord_index + 1],
                )
                grad_y = self.cuda_texture_gradient_argument(
                    texture_type,
                    raw_args[coord_index + 2],
                    args[coord_index + 2],
                )

            texture_name = args[0]
            coord = args[coord_index]
            lod_index = coord_index + 1
            if texture_shape == "sampler1D":
                if func_name == "texture":
                    return f"tex1D<{value_type}>({texture_name}, {coord})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return (
                        f"tex1DLod<{value_type}>"
                        f"({texture_name}, {coord}, {args[lod_index]})"
                    )
                if func_name == "textureGrad" and grad_x is not None:
                    return (
                        f"tex1DGrad<{value_type}>"
                        f"({texture_name}, {coord}, {grad_x}, {grad_y})"
                    )

            if texture_shape == "sampler1DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex1DLayered<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return f"tex1DLayeredLod<{value_type}>({coord_args}, {args[lod_index]})"
                if func_name == "textureGrad" and grad_x is not None:
                    return (
                        f"tex1DLayeredGrad<{value_type}>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_shape == "sampler2D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex2D<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return f"tex2DLod<{value_type}>({coord_args}, {args[lod_index]})"
                if func_name == "textureGrad" and grad_x is not None:
                    return f"tex2DGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

            if texture_shape == "sampler2DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex2DLayered<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return f"tex2DLayeredLod<{value_type}>({coord_args}, {args[lod_index]})"
                if func_name == "textureGrad" and grad_x is not None:
                    return (
                        f"tex2DLayeredGrad<{value_type}>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_shape == "sampler3D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex3D<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return f"tex3DLod<{value_type}>({coord_args}, {args[lod_index]})"
                if func_name == "textureGrad" and grad_x is not None:
                    return f"tex3DGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

            if texture_shape == "samplerCube":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"texCubemap<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return (
                        f"texCubemapLod<{value_type}>({coord_args}, {args[lod_index]})"
                    )
                if func_name == "textureGrad" and grad_x is not None:
                    return f"texCubemapGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

            if texture_shape == "samplerCubeArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}, "
                    f"{self.coord_component(coord, 'w')}"
                )
                if func_name == "texture":
                    return f"texCubemapLayered<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return (
                        f"texCubemapLayeredLod<{value_type}>"
                        f"({coord_args}, {args[lod_index]})"
                    )
                if func_name == "textureGrad" and grad_x is not None:
                    return (
                        f"texCubemapLayeredGrad<{value_type}>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )
        return UNHANDLED

    def cuda_texel_fetch_call(
        self,
        func_name,
        raw_args,
        args,
    ):
        """Emit texel fetches."""
        if func_name == "texelFetch" and len(args) >= 3:
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if texture_type is not None and not self.is_sampled_texture_family_type(
                texture_type
            ):
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )
            if self.is_shadow_resource_type(texture_type):
                return self.unsupported_shadow_resource_call(
                    func_name, texture_type, args
                )
            if self.is_multisample_resource_type(texture_type):
                return self.unsupported_multisample_resource_call(
                    func_name, texture_type, args
                )
            texture_shape = self.sampled_texture_shape_type(texture_type)
            if texture_shape in {"samplerCube", "samplerCubeArray"}:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

            sampler_arg_offset = self.sampled_texture_sampler_argument_offset(raw_args)
            coord_index = 1 + sampler_arg_offset
            lod_index = coord_index + 1
            if len(args) <= lod_index:
                return None
            texture_name = args[0]
            coord = args[coord_index]
            expected_coord_count = self.expected_texel_fetch_coordinate_count(
                texture_shape
            )
            actual_coord_count = self.texel_fetch_coordinate_count(
                raw_args[coord_index]
            )
            if (
                expected_coord_count is not None
                and actual_coord_count is not None
                and actual_coord_count != expected_coord_count
            ):
                return self.unsupported_sampled_resource_call(
                    "texelFetch coordinate rank",
                    texture_type,
                    args,
                )
            components = self.cuda_texture_coord_components(texture_shape, coord)
            if components is not None:
                return self.cuda_texel_fetch_call_from_components(
                    texture_type, texture_name, components
                )
        return UNHANDLED

    def cuda_storage_image_call(
        self,
        func_name,
        raw_args,
        args,
    ):
        """Emit storage-image loads and stores.

        The surface element is the image's texel as memory holds it (:mod:`.cuda_surface_texels`); without a
        format or texel type it is the four-component default of :meth:`image_value_type`.
        """
        if func_name not in {"imageLoad", "imageStore"}:
            return UNHANDLED
        if len(args) < (2 if func_name == "imageLoad" else 3):
            return UNHANDLED
        image_type = self.resource_base_type(self.resource_expression_type(raw_args[0]))
        if image_type is None:
            return None
        if self.is_multisample_resource_type(image_type):
            return self.unsupported_multisample_resource_call(
                func_name, image_type, args
            )
        access_diagnostic = self.image_access_diagnostic(
            func_name, image_type, raw_args[0]
        )
        if access_diagnostic is not None:
            return access_diagnostic
        coordinate_diagnostic = self.image_coordinate_rank_diagnostic(
            func_name, image_type, raw_args[1]
        )
        if coordinate_diagnostic is not None:
            return coordinate_diagnostic

        image_name = args[0]
        coord = args[1]
        texel = self.storage_image_texel(raw_args[0])
        if isinstance(texel, UnsupportedSurfaceTexel):
            placeholder = (
                self.zero_value_for_type(self.image_value_type(image_type))
                if func_name == "imageLoad"
                else "((void)0)"
            )
            return self.surface_texel_refusal(func_name, image_name, texel, placeholder)
        value_type = (
            texel.element if texel is not None else self.image_value_type(image_type)
        )
        if func_name == "imageLoad":
            read = self.cuda_surface_read(image_type, image_name, coord, value_type)
            if read is None:
                return UNHANDLED
            width = self.surface_texel_load_width(raw_args)
            return self.surface_texel_load(read, texel, width)
        value = self.surface_texel_store_value(raw_args[2], args[2], texel)
        return self.cuda_surface_write(image_type, image_name, coord, value, value_type)

    def cuda_surface_read(self, image_type, image_name, coord, value_type):
        """The ``surf*read<value_type>`` call that reads the texel at ``coord``, or ``None``."""
        x = self.surface_x_offset(coord, value_type)
        y = self.coord_component(coord, "y")
        if "CubeArray" in image_type:
            layer_face = self.coord_component(coord, "z")
            return (
                f"surfCubemapLayeredread<{value_type}>"
                f"({image_name}, {x}, {y}, {layer_face})"
            )
        if "Cube" in image_type:
            face = self.coord_component(coord, "z")
            return f"surfCubemapread<{value_type}>({image_name}, {x}, {y}, {face})"
        if "3D" in image_type:
            z = self.coord_component(coord, "z")
            return f"surf3Dread<{value_type}>({image_name}, {x}, {y}, {z})"
        if "1DArray" in image_type:
            layer = self.coord_component(coord, "y")
            return f"surf1DLayeredread<{value_type}>({image_name}, {x}, {layer})"
        if "Array" in image_type:
            layer = self.coord_component(coord, "z")
            return f"surf2DLayeredread<{value_type}>({image_name}, {x}, {y}, {layer})"
        if "1D" in image_type:
            x = f"{coord} * sizeof({value_type})"
            return f"surf1Dread<{value_type}>({image_name}, {x})"
        if "2D" in image_type:
            return f"surf2Dread<{value_type}>({image_name}, {x}, {y})"
        return None

    def cuda_surface_write(self, image_type, image_name, coord, value, value_type):
        """The ``surf*write`` call that stores ``value`` at ``coord``, or ``UNHANDLED``."""
        x = self.surface_x_offset(coord, value_type)
        y = self.coord_component(coord, "y")
        if "CubeArray" in image_type:
            layer_face = self.coord_component(coord, "z")
            return (
                f"surfCubemapLayeredwrite"
                f"({value}, {image_name}, {x}, {y}, {layer_face})"
            )
        if "Cube" in image_type:
            face = self.coord_component(coord, "z")
            return f"surfCubemapwrite({value}, {image_name}, {x}, {y}, {face})"
        if "3D" in image_type:
            z = self.coord_component(coord, "z")
            return f"surf3Dwrite({value}, {image_name}, {x}, {y}, {z})"
        if "1DArray" in image_type:
            layer = self.coord_component(coord, "y")
            return f"surf1DLayeredwrite({value}, {image_name}, {x}, {layer})"
        if "Array" in image_type:
            layer = self.coord_component(coord, "z")
            return f"surf2DLayeredwrite({value}, {image_name}, {x}, {y}, {layer})"
        if "1D" in image_type:
            x = f"{coord} * sizeof({value_type})"
            return f"surf1Dwrite({value}, {image_name}, {x})"
        if "2D" in image_type:
            return f"surf2Dwrite({value}, {image_name}, {x}, {y})"
        return UNHANDLED
