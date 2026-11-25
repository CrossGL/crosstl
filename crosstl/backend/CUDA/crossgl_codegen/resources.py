"""Texture and surface calls in the CUDA to CrossGL converter."""


class CudaImportResourceMixin:
    """Resource members of ``CudaToCrossGLConverter``."""

    def format_cuda_texture_call(self, function_name, args, vector_name, dimensions):
        if len(args) < 2:
            return None

        if self.is_sparse_cuda_texture_call(function_name, args):
            return self.format_unsupported_cuda_texture_sparse_residency_call(
                function_name
            )

        extra_count = (
            2 if "Grad" in function_name else 1 if "Lod" in function_name else 0
        )
        coordinate_count = len(args) - 1 - extra_count
        if coordinate_count <= 0:
            return None

        texture_name = args[0]
        coordinate_args = args[1 : 1 + coordinate_count]
        if coordinate_count == 1:
            coordinate = coordinate_args[0]
            consumed = 2
        elif coordinate_count == dimensions:
            coordinate = self.format_vector_constructor(vector_name, coordinate_args)
            consumed = 1 + dimensions
        else:
            return None

        remaining = args[consumed:]
        if "Grad" in function_name:
            if len(remaining) < 2:
                return None
            return f"textureGrad({texture_name}, {coordinate}, {remaining[0]}, {remaining[1]})"
        if "Lod" in function_name:
            if not remaining:
                return None
            return f"textureLod({texture_name}, {coordinate}, {remaining[0]})"
        return f"texture({texture_name}, {coordinate})"

    def is_sparse_cuda_texture_call(self, function_name, args):
        sparse_arg_counts = {
            "tex1D": 3,
            "tex1DLod": 4,
            "tex1DGrad": 5,
            "tex2D": 4,
            "tex2DLod": 5,
            "tex2DGrad": 6,
            "tex3D": 5,
            "tex3DLod": 6,
            "tex3DGrad": 7,
            "texCubemap": 5,
            "texCubemapLod": 6,
            "texCubemapGrad": 7,
            "tex1DLayered": 4,
            "tex1DLayeredLod": 5,
            "tex1DLayeredGrad": 6,
            "tex2DLayered": 5,
            "tex2DLayeredLod": 6,
            "tex2DLayeredGrad": 7,
            "texCubemapLayered": 6,
            "texCubemapLayeredLod": 7,
            "texCubemapLayeredGrad": 8,
        }
        return len(args) == sparse_arg_counts.get(function_name)

    def format_unsupported_cuda_texture_sparse_residency_call(self, function_name):
        return self.format_unsupported_cuda_resource_expression(
            "texture",
            f"{function_name} sparse residency",
            "vec4<f32>(0.0, 0.0, 0.0, 0.0)",
        )

    def format_cuda_texture_gather_call(self, args):
        if len(args) == 2:
            texture_name, coordinate = args
            component = None
        elif len(args) in {3, 4}:
            texture_name = args[0]
            coordinate = self.format_vector_constructor("vec2", args[1:3])
            component = args[3] if len(args) == 4 else None
        else:
            return self.format_unsupported_cuda_resource_expression(
                "texture",
                "tex2Dgather sparse residency",
                "vec4<f32>(0.0, 0.0, 0.0, 0.0)",
            )

        if component is not None:
            return f"textureGather({texture_name}, {coordinate}, {component})"
        return f"textureGather({texture_name}, {coordinate})"

    def format_cuda_texture_fetch_call(self, args):
        if len(args) == 2:
            texture_name, coordinate = args
            # A texture bound to linear memory is a texel buffer, read by element (texture_elements).
            return f"texelFetch({texture_name}, {coordinate})"
        return self.format_unsupported_cuda_resource_expression(
            "texture",
            "tex1Dfetch sparse residency",
            "vec4<f32>(0.0, 0.0, 0.0, 0.0)",
        )

    def format_cuda_surface_coordinate(
        self, base_name, surface_name, coord_args, dimensions, mode_args
    ):
        """The image coordinate of a surface access, or ``None`` to keep the CUDA call as it is.

        ``mode_args`` holds the arguments after the coordinates: none, or the boundary mode. With no mode or
        ``cudaBoundaryModeTrap``, the default, the coordinate is used as it is. ``cudaBoundaryModeClamp`` clamps it
        to the surface. ``cudaBoundaryModeZero``, under which a load out of range reads zero and a store out of
        range is skipped, has no CrossGL form yet, and neither does a mode that is not one of the three names (a
        variable, say). The call then stays as CUDA wrote it, and every target refuses it as an undeclared
        function instead of reading or writing the surface without the mode.
        """
        if not mode_args:
            mode = "trap"
        elif len(mode_args) == 1:
            mode = self.cuda_surface_boundary_mode(mode_args[0])
        else:
            mode = None
        if mode == "trap":
            return self.format_vector_constructor(f"vec{dimensions}", coord_args, "i32")
        if mode == "clamp":
            return self.format_cuda_surface_clamped_coordinate(
                base_name, surface_name, coord_args, dimensions
            )
        return None

    def cuda_surface_boundary_mode(self, mode_text):
        """Return 'trap', 'clamp' or 'zero' for a boundary-mode argument."""
        if not isinstance(mode_text, str):
            return None
        text = str(mode_text).strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        if text.startswith("::"):
            text = text[2:]
        if "::" in text:
            text = text.rsplit("::", 1)[-1]
        return self.CUDA_SURFACE_BOUNDARY_MODES.get(text)

    def format_cuda_surface_clamped_coordinate(
        self, base_name, surface_name, coord_args, dimensions
    ):
        """``coord_args`` clamped to the texels of ``surface_name``, as ``cudaBoundaryModeClamp`` reads them.

        Each texel coordinate is clamped to ``[0, size - 1]``, so a negative one reads the first texel. A layer
        index or a cubemap face is not a texel coordinate and is left as it is: ``imageSize`` of a cubemap has no
        face component, and that of a layered cubemap counts cubes, not faces.
        """
        texel_dimensions = self.CUDA_SURFACE_TEXEL_DIMENSIONS[
            base_name.removeprefix("surf").removesuffix("read").removesuffix("write")
        ]
        size = f"imageSize({surface_name})"
        if texel_dimensions == dimensions:
            coord = self.format_vector_constructor(
                f"vec{dimensions}", coord_args, "i32"
            )
            if dimensions == 1:
                return f"clamp({coord}, 0, {size} - 1)"
            zeros = ", ".join(["0"] * dimensions)
            ones = ", ".join(["1"] * dimensions)
            return (
                f"clamp({coord}, vec{dimensions}<i32>({zeros}), "
                f"{size} - vec{dimensions}<i32>({ones}))"
            )
        if texel_dimensions == 1:
            clamped = [f"clamp({coord_args[0]}, 0, {size}.x - 1)"]
        else:
            clamped = [
                f"clamp({coord_args[index]}, 0, {size}.{'xyz'[index]} - 1)"
                for index in range(texel_dimensions)
            ]
        return self.format_vector_constructor(
            f"vec{dimensions}", clamped + list(coord_args[texel_dimensions:]), "i32"
        )

    def format_unsupported_cuda_resource_expression(self, kind, member, fallback):
        return self.refuse_import(f"CUDA {kind}.{member} has no CrossGL form", fallback)
