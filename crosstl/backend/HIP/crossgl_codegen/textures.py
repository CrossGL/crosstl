"""Texture calls and coordinate ranks in the HIP to CrossGL converter."""

from ...CUDA.texture_elements import TEXTURE_SAMPLE_COORDINATES, TextureElementMixin


class HipTextureMixin(TextureElementMixin):
    """Texture members of ``HipToCrossGLConverter``."""

    def texture_element_crossgl_type(self, template_args):
        """The CrossGL type of a texture fetch's element type ``T``, or ``None`` when the fetch names none."""
        if not template_args:
            return None
        return self.convert_hip_type_to_crossgl(template_args[0])

    def format_hip_resource_call(self, function_name, args, raw_args=None):
        base_name, template_args = self.parse_cpp_template(function_name)
        if self.is_user_defined_function(base_name):
            return None

        value_type = template_args[0] if template_args else None
        if base_name == "tex1Dfetch":
            return self.format_typed_texture_result(
                self.format_hip_texture_fetch_call(args), value_type, raw_args
            )
        if base_name == "tex2Dgather":
            return self.format_typed_texture_result(
                self.format_hip_texture_gather_call(args), value_type, raw_args
            )
        if base_name in TEXTURE_SAMPLE_COORDINATES:
            vector_name, dimensions = TEXTURE_SAMPLE_COORDINATES[base_name]
            return self.format_typed_texture_result(
                self.format_hip_texture_call(
                    base_name, args, vector_name, dimensions, raw_args
                ),
                value_type,
                raw_args,
            )

        if base_name in {
            "surf1Dread",
            "surf1DLayeredread",
            "surf2Dread",
            "surf3Dread",
            "surf2DLayeredread",
            "surfCubemapread",
            "surfCubemapLayeredread",
        }:
            dimensions = {
                "surf1Dread": 1,
                "surf1DLayeredread": 2,
                "surf2Dread": 2,
                "surf3Dread": 3,
                "surf2DLayeredread": 3,
                "surfCubemapread": 3,
                "surfCubemapLayeredread": 4,
            }[base_name]
            return self.format_hip_surface_read(
                base_name, args, dimensions, value_type, raw_args
            )

        if base_name in {
            "surf1Dwrite",
            "surf1DLayeredwrite",
            "surf2Dwrite",
            "surf3Dwrite",
            "surf2DLayeredwrite",
            "surfCubemapwrite",
            "surfCubemapLayeredwrite",
        }:
            dimensions = {
                "surf1Dwrite": 1,
                "surf1DLayeredwrite": 2,
                "surf2Dwrite": 2,
                "surf3Dwrite": 3,
                "surf2DLayeredwrite": 3,
                "surfCubemapwrite": 3,
                "surfCubemapLayeredwrite": 4,
            }[base_name]
            return self.format_hip_surface_write(
                base_name,
                args,
                dimensions,
                value_type,
                value_is_pointer=base_name == "surfCubemapLayeredwrite",
                raw_args=raw_args,
            )

        return None

    def format_hip_texture_call(
        self, function_name, args, vector_name, dimensions, raw_args=None
    ):
        if len(args) < 2:
            return None

        if self.is_sparse_hip_texture_call(function_name, args):
            return self.format_unsupported_hip_texture_sparse_residency_call(
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
        raw_coordinate_args = (raw_args or [])[1 : 1 + coordinate_count]
        if coordinate_count == 1:
            rank = self.infer_texture_coordinate_rank(
                coordinate_args[0],
                raw_coordinate_args[0] if raw_coordinate_args else None,
            )
            if dimensions > 1 and rank is not None and rank != dimensions:
                return self.format_hip_texture_coordinate_rank_diagnostic(function_name)
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
            return (
                f"textureGrad({texture_name}, {coordinate}, "
                f"{remaining[0]}, {remaining[1]})"
            )
        if "Lod" in function_name:
            if not remaining:
                return None
            return f"textureLod({texture_name}, {coordinate}, {remaining[0]})"
        return f"texture({texture_name}, {coordinate})"

    def infer_texture_coordinate_rank(self, coordinate, raw_coordinate=None):
        expression_type = self.lookup_texture_coordinate_expression_type(
            coordinate, raw_coordinate
        )
        rank = self.vector_type_rank(expression_type)
        if rank is not None:
            return rank
        if self.is_scalar_type_name(expression_type):
            return 1

        text = self.strip_wrapping_parentheses(str(coordinate).strip())
        rank = self.vector_constructor_rank(text)
        if rank is not None:
            return rank
        if self.is_swizzle_expression(text):
            return len(text.rsplit(".", 1)[1])
        if self.is_scalar_literal(text):
            return 1
        return None

    def lookup_texture_coordinate_expression_type(self, coordinate, raw_coordinate):
        name = self.get_resource_object_expression_name(raw_coordinate)
        if name is None:
            name = self.strip_wrapping_parentheses(str(coordinate).strip())
        if self.is_simple_identifier(name):
            return self.lookup_variable_type(name)
        return None

    def vector_type_rank(self, type_name):
        if not type_name:
            return None
        text = str(type_name).strip()
        if text.startswith("vec") and len(text) >= 4 and text[3].isdigit():
            return int(text[3])
        mapped_type = self.VECTOR_TYPE_MAPPING.get(text)
        if mapped_type is not None:
            return self.vector_type_rank(mapped_type)
        return None

    def vector_constructor_rank(self, text):
        if text.startswith("vec") and len(text) >= 4 and text[3].isdigit():
            return int(text[3])
        return None

    def is_scalar_type_name(self, type_name):
        return type_name in {
            "bool",
            "f32",
            "f64",
            "i8",
            "u8",
            "i16",
            "u16",
            "i32",
            "u32",
            "i64",
            "u64",
        }

    def is_simple_identifier(self, text):
        if not isinstance(text, str) or not text:
            return False
        return text.replace("_", "").isalnum() and not text[0].isdigit()

    def is_swizzle_expression(self, text):
        if "." not in text:
            return False
        suffix = text.rsplit(".", 1)[1]
        return bool(suffix) and all(char in "xyzwrgba" for char in suffix)

    def is_scalar_literal(self, text):
        if not text:
            return False
        literal = text.lower()
        while literal.endswith(("f", "u", "l")):
            literal = literal[:-1]
        try:
            float(literal)
        except ValueError:
            return False
        return True

    def strip_wrapping_parentheses(self, text):
        while self.has_wrapping_parentheses(text):
            inner = text[1:-1].strip()
            if not inner:
                break
            text = inner
        return text

    def has_wrapping_parentheses(self, text):
        if not text.startswith("(") or not text.endswith(")"):
            return False
        depth = 0
        for index, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(text) - 1:
                    return False
            if depth < 0:
                return False
        return depth == 0

    def is_sparse_hip_texture_call(self, function_name, args):
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

    def format_unsupported_hip_texture_sparse_residency_call(self, function_name):
        return self.format_unsupported_hip_resource_expression(
            "texture",
            f"{function_name} sparse residency",
            "vec4<f32>(0.0, 0.0, 0.0, 0.0)",
        )

    def format_hip_texture_gather_call(self, args):
        if len(args) == 2:
            texture_name, coordinate = args
            component = None
        elif len(args) in {3, 4}:
            texture_name = args[0]
            coordinate = self.format_vector_constructor("vec2", args[1:3])
            component = args[3] if len(args) == 4 else None
        else:
            return self.format_unsupported_hip_texture_sparse_residency_call(
                "tex2Dgather"
            )

        if component is not None:
            return f"textureGather({texture_name}, {coordinate}, {component})"
        return f"textureGather({texture_name}, {coordinate})"

    def format_hip_texture_coordinate_rank_diagnostic(self, function_name):
        return self.format_unsupported_hip_resource_expression(
            "texture",
            f"{function_name} coordinate rank mismatch",
            "vec4<f32>(0.0, 0.0, 0.0, 0.0)",
        )

    def format_hip_texture_fetch_call(self, args):
        if len(args) == 3:
            return self.format_unsupported_hip_resource_expression(
                "texture",
                "tex1Dfetch sparse residency",
                "vec4<f32>(0.0, 0.0, 0.0, 0.0)",
            )
        if len(args) != 2:
            return None
        texture_name, coordinate = args
        # A texture bound to linear memory is a texel buffer, read by element (CUDA/texture_elements).
        return f"texelFetch({texture_name}, {coordinate})"
