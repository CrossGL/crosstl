"""Texture and storage-image calls of the HIP generator: LOD queries, the refused and specially lowered sampling forms, offset and projective sampling, image atomics, implicit-LOD, explicit-LOD and gradient sampling, texel fetches, and image loads and stores."""

from ..unhandled import (
    UNHANDLED,
)
from .cuda_surface_texels import UnsupportedSurfaceTexel


class HipResourceCallMixin:
    """Texture and storage-image members of ``HipCodeGen.generate_resource_call``."""

    def hip_texture_form_call(
        self,
        func_name,
        raw_args,
        args,
    ):
        """Emit LOD queries, and refuse or lower the gather, compare, projective and offset-fetch forms that need special handling."""
        if func_name == "textureQueryLod" and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_resource_query_call(
                    func_name, texture_type, args
                )

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
                self.get_expression_type(raw_args[0])
            )
            if self.is_shadow_resource_type(texture_type):
                return self.unsupported_shadow_resource_call(
                    func_name, texture_type, args
                )

        if func_name == "textureGather" and len(args) >= 2:
            texture_gather = self.generate_texture_gather_call(raw_args, args)
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
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

        if (
            func_name in {
                "textureProj",
                "textureProjLod",
                "textureProjGrad",
            }
            and len(args) >= 2
        ):
            projected_texture = self.generate_projected_texture_call(
                func_name, raw_args, args
            )
            if projected_texture is not None:
                return projected_texture

        if func_name == "texelFetchOffset" and len(args) >= 4:
            texel_fetch_offset = self.generate_texel_fetch_offset_call(raw_args, args)
            if texel_fetch_offset is not None:
                return texel_fetch_offset
        return UNHANDLED

    def hip_offset_texture_call(
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
                self.get_expression_type(raw_args[0])
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
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )
        return UNHANDLED

    def hip_image_atomic_call(
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
                    self.get_expression_type(raw_args[0])
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

    def hip_sampled_texture_call(
        self,
        func_name,
        raw_args,
        args,
    ):
        """Emit implicit-LOD, explicit-LOD and gradient texture sampling."""
        if func_name in {"texture", "textureLod", "textureGrad"} and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            texture_shape = self.sampled_texture_shape_type(texture_type)
            value_type = self.sampled_texture_value_type(texture_type)
            if self.is_multisample_resource_type(texture_type):
                return self.unsupported_multisample_resource_call(
                    func_name, texture_type, args
                )
            coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
                func_name, texture_type, raw_args[1]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            grad_x = None
            grad_y = None
            if func_name == "textureGrad" and len(args) >= 4:
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
            coord = args[1]
            if texture_shape == "sampler1D":
                if func_name == "texture":
                    return f"tex1D<{value_type}>({texture_name}, {coord})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex1DLod<{value_type}>({texture_name}, {coord}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"tex1DGrad<{value_type}>"
                        f"({texture_name}, {coord}, {grad_x}, {grad_y})"
                    )

            if texture_shape == "sampler2D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex2D<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex2DLod<{value_type}>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return f"tex2DGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

            if texture_shape == "sampler1DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex1DLayered<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex1DLayeredLod<{value_type}>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"tex1DLayeredGrad<{value_type}>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_shape == "sampler2DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex2DLayered<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex2DLayeredLod<{value_type}>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
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
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex3DLod<{value_type}>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
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
                if func_name == "textureLod" and len(args) >= 3:
                    return f"texCubemapLod<{value_type}>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"texCubemapGrad<{value_type}>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

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
                if func_name == "textureLod" and len(args) >= 3:
                    return (
                        f"texCubemapLayeredLod<{value_type}>({coord_args}, {args[2]})"
                    )
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"texCubemapLayeredGrad<{value_type}>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )
        return UNHANDLED

    def hip_texel_fetch_call(
        self,
        func_name,
        raw_args,
        args,
    ):
        """Emit texel fetches."""
        if func_name == "texelFetch" and len(args) >= 3:
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
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

            texture_name = args[0]
            coord = args[1]
            value_type = self.sampled_texture_value_type(texture_type)
            expected_coord_count = self.expected_texel_fetch_coordinate_count(
                texture_shape
            )
            actual_coord_count = self.texel_fetch_coordinate_count(raw_args[1])
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
            if texture_shape == "sampler1D":
                return f"tex1Dfetch<{value_type}>({texture_name}, {coord})"
            if texture_shape == "sampler1DArray":
                return (
                    f"tex1DLayered<{value_type}>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')})"
                )
            if texture_shape == "sampler2D":
                return (
                    f"tex2D<{value_type}>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')})"
                )
            if texture_shape == "sampler2DArray":
                return (
                    f"tex2DLayered<{value_type}>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')})"
                )
            if texture_shape == "sampler3D":
                return (
                    f"tex3D<{value_type}>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')})"
                )
        return UNHANDLED

    def hip_storage_image_call(
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
        image_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
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
            read = self.hip_surface_read(image_type, image_name, coord, value_type)
            if read is None:
                return UNHANDLED
            width = self.surface_texel_load_width(raw_args)
            return self.surface_texel_load(read, texel, width)
        value = self.surface_texel_store_value(raw_args[2], args[2], texel)
        return self.hip_surface_write(image_type, image_name, coord, value, value_type)

    def hip_surface_read(self, image_type, image_name, coord, value_type):
        """The ``cgl_surf*read<value_type>`` call that reads the texel at ``coord``, or ``None``."""
        x = self.image_surface_x_offset(coord, value_type, image_type)
        y = self.coord_component(coord, "y")
        if "1DArray" in image_type:
            self.require_surface_read_helper("cgl_surf1DLayeredread")
            layer = self.coord_component(coord, "y")
            return f"cgl_surf1DLayeredread<{value_type}>({image_name}, {x}, {layer})"
        if "1D" in image_type:
            self.require_surface_read_helper("cgl_surf1Dread")
            return f"cgl_surf1Dread<{value_type}>({image_name}, {x})"
        if "CubeArray" in image_type:
            self.require_surface_read_helper("cgl_surfCubemapLayeredread")
            face = self.coord_component(coord, "z")
            layer = self.coord_component(coord, "w")
            return (
                f"cgl_surfCubemapLayeredread<{value_type}>"
                f"({image_name}, {x}, {y}, {face}, {layer})"
            )
        if "Cube" in image_type:
            self.require_surface_read_helper("cgl_surfCubemapread")
            face = self.coord_component(coord, "z")
            return f"cgl_surfCubemapread<{value_type}>({image_name}, {x}, {y}, {face})"
        if "3D" in image_type:
            self.require_surface_read_helper("cgl_surf3Dread")
            z = self.coord_component(coord, "z")
            return f"cgl_surf3Dread<{value_type}>({image_name}, {x}, {y}, {z})"
        if "Array" in image_type:
            self.require_surface_read_helper("cgl_surf2DLayeredread")
            layer = self.coord_component(coord, "z")
            return (
                f"cgl_surf2DLayeredread<{value_type}>"
                f"({image_name}, {x}, {y}, {layer})"
            )
        if "2D" in image_type:
            self.require_surface_read_helper("cgl_surf2Dread")
            return f"cgl_surf2Dread<{value_type}>({image_name}, {x}, {y})"
        return None

    def hip_surface_write(self, image_type, image_name, coord, value, value_type):
        """The ``surf*write`` call that stores ``value`` at ``coord``, or ``UNHANDLED``."""
        x = self.image_surface_x_offset(coord, value_type, image_type)
        y = self.coord_component(coord, "y")
        if "1DArray" in image_type:
            layer = self.coord_component(coord, "y")
            return f"surf1DLayeredwrite({value}, {image_name}, {x}, {layer})"
        if "1D" in image_type:
            return f"surf1Dwrite({value}, {image_name}, {x})"
        if "CubeArray" in image_type:
            face = self.coord_component(coord, "z")
            layer = self.coord_component(coord, "w")
            return (
                f"surfCubemapLayeredwrite"
                f"({value}, {image_name}, {x}, {y}, {face}, {layer})"
            )
        if "Cube" in image_type:
            face = self.coord_component(coord, "z")
            return f"surfCubemapwrite({value}, {image_name}, {x}, {y}, {face})"
        if "3D" in image_type:
            z = self.coord_component(coord, "z")
            return f"surf3Dwrite({value}, {image_name}, {x}, {y}, {z})"
        if "Array" in image_type:
            layer = self.coord_component(coord, "z")
            return f"surf2DLayeredwrite({value}, {image_name}, {x}, {y}, {layer})"
        if "2D" in image_type:
            return f"surf2Dwrite({value}, {image_name}, {x}, {y})"
        return UNHANDLED
