"""CrossGL emission of HLSL texel reads and writes.

HLSL resources declare the component count of their texels, while CrossGL
image and texture operations return four components. These members narrow
loaded texels and widen stored values to the declared type, update single
components of storage-image texels, recognize storage textures whose format
the runtime chooses, and map sampled HLSL texture types.
"""

from ..common_ast import (
    ArrayAccessNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    MemberAccessNode,
    TextureSampleNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
)


class HLSLTexelAccessMixin:
    """HLSL texture and storage-image members of ``HLSLToCrossGLConverter``."""

    def is_structured_buffer_resource_type(self, type_name):
        """Whether a resource stores elements with the structured-buffer layout."""
        if type_name is None:
            return False
        return self.raw_type_base(type_name) in {
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
            "RasterizerOrderedStructuredBuffer",
        }

    TEXEL_VALUE_FUNCTIONS = frozenset(
        {
            "imageLoad",
            "texelFetch",
            "texelFetchOffset",
            "texture",
            "textureClamp",
            "textureOffset",
            "textureOffsetClamp",
            "textureLod",
            "textureLodOffset",
            "textureGrad",
            "textureGradClamp",
            "textureGradOffset",
            "textureGradOffsetClamp",
        }
    )

    def is_texture_resource_type(self, type_name):
        return self.raw_type_base(type_name).startswith(
            ("Texture", "RWTexture", "RasterizerOrderedTexture")
        )

    def texel_component_count(self, resource_type):
        """Component count of a texture's declared element type, if known."""
        if not self.is_texture_resource_type(resource_type):
            return None
        element = self.first_template_argument(resource_type)
        if element is None:
            return None
        return self.storage_image_component_count(element)

    def texel_narrowing_suffix(self, resource_type):
        """Swizzle that narrows a four-component texel to the element type.

        HLSL texture reads return the declared element type while CrossGL
        texel functions return four components, as in GLSL.
        """
        return {1: ".x", 2: ".xy", 3: ".xyz"}.get(
            self.texel_component_count(resource_type), ""
        )

    def texel_vector_type(self, resource_type):
        """Four-component CrossGL vector type of a texture's texels."""
        element = self.normalize_resource_component_type(
            self.first_template_argument(resource_type)
        )
        family = self.numeric_type_family(element)
        if family == "uint":
            return "uvec4"
        if family == "int":
            return "ivec4"
        return "vec4"

    def widen_storage_image_value(self, resource_type, value, rendered_value):
        """Widen a stored value to the four components image stores take."""
        count = self.texel_component_count(resource_type)
        if count is None or count == 4:
            return rendered_value
        vector_type = self.texel_vector_type(resource_type)
        zero = {"vec4": "0.0", "ivec4": "0", "uvec4": "0u"}[vector_type]
        scalar = {"vec4": "float", "ivec4": "int", "uvec4": "uint"}[vector_type]
        value_type = self.expression_value_raw_type(value)
        value_width = self.storage_image_component_count(value_type)
        if value_width is not None and value_width > count:
            narrowing = {1: ".x", 2: ".xy", 3: ".xyz"}[count]
            rendered_value = self.maybe_parenthesize(value, rendered_value) + narrowing
            value_type = None
        if count > 1 and self.numeric_type_family(
            value_type
        ) != self.numeric_type_family(scalar):
            rendered_value = f"{vector_type[:-1]}{count}({rendered_value})"
        padding = ", ".join([zero] * (4 - count))
        return f"{vector_type}({rendered_value}, {padding})"

    def expression_is_side_effect_free(self, expr):
        """Whether evaluating ``expr`` twice is observably the same as once."""
        if expr is None or isinstance(expr, (bool, int, float, str)):
            return True
        if isinstance(expr, VariableNode):
            return True
        if isinstance(expr, MemberAccessNode):
            return self.expression_is_side_effect_free(expr.object)
        if isinstance(expr, ArrayAccessNode):
            return self.expression_is_side_effect_free(
                expr.array
            ) and self.expression_is_side_effect_free(expr.index)
        if isinstance(expr, BinaryOpNode):
            return self.expression_is_side_effect_free(
                expr.left
            ) and self.expression_is_side_effect_free(expr.right)
        if isinstance(expr, UnaryOpNode):
            return str(expr.op) not in {
                "++",
                "--",
            } and self.expression_is_side_effect_free(expr.operand)
        if isinstance(expr, CastNode):
            return self.expression_is_side_effect_free(expr.expression)
        if isinstance(expr, VectorConstructorNode):
            return all(
                self.expression_is_side_effect_free(arg)
                for arg in getattr(expr, "args", []) or []
            )
        if isinstance(expr, FunctionCallNode) and isinstance(expr.name, str):
            type_name = self.canonical_composite_type(expr.name)
            is_constructor = self.map_type(type_name) != self.sanitize_type_name(
                type_name
            )
            return is_constructor and all(
                self.expression_is_side_effect_free(arg) for arg in expr.args
            )
        return False

    def generate_storage_image_component_update(self, node, indent, is_main=False):
        """Lower ``image[coord].yz = value`` to a texel read-modify-write.

        Storage image texels are not addressable, so a partial component store
        loads the texel, updates the components and stores the whole texel.
        Returns ``None`` for other assignments.
        """
        component_access = self.storage_image_component_assignment_access(node.left)
        if component_access is None:
            return None
        access, component = component_access
        resource_type = self.expression_raw_type(access.array)
        count = self.texel_component_count(resource_type)
        if count == 1 and component in {"x", "r"}:
            return None
        if not self.expression_is_side_effect_free(access.index):
            raise ValueError(
                "DirectX partial storage image store needs a side-effect-free "
                f"texel coordinate: {self.generate_expression(access.index, is_main)}"
            )
        image, coord = self.generate_storage_image_access_parts(access, is_main)
        self.storage_image_texel_temp_count += 1
        temp = f"_texel{self.storage_image_texel_temp_count}"
        value = self.generate_expression(node.right, is_main)
        indent_text = "    " * indent
        stored = self.unorm_store_value(access.array, resource_type, temp)
        return (
            f"{self.texel_vector_type(resource_type)} {temp} = "
            f"imageLoad({image}, {coord});\n"
            f"{indent_text}{temp}.{component} {node.operator} {value};\n"
            f"{indent_text}imageStore({image}, {coord}, {stored});\n"
        )

    def generate_texel_access(self, expr, is_main=False, narrow=True):
        """Render an indexed texture read, or ``None`` for other accesses."""
        if not isinstance(expr, ArrayAccessNode):
            return None
        multisample_fetch = self.generate_multisample_texture_operator_fetch(
            expr, is_main, narrow
        )
        if multisample_fetch is not None:
            return multisample_fetch
        sampled_fetch = self.generate_sampled_texture_operator_fetch(
            expr, is_main, narrow
        )
        if sampled_fetch is not None:
            return sampled_fetch
        if (
            not self.suppress_storage_image_index_lowering
            and self.is_storage_image_texel_access(expr)
        ):
            return self.generate_storage_image_load(expr, is_main, narrow)
        return None

    def texel_swizzle_reads_element_components(self, texel_expr, member):
        """Whether a swizzle of a texel read only selects element components."""
        if not isinstance(member, str):
            return False
        if isinstance(texel_expr, TextureSampleNode):
            return self.texel_swizzle_selects_components(
                self.texel_component_count(
                    self.expression_raw_type(texel_expr.texture)
                ),
                member,
            )
        if (
            isinstance(texel_expr, FunctionCallNode)
            and isinstance(texel_expr.name, MemberAccessNode)
            and self.is_texture_resource_type(
                self.expression_raw_type(texel_expr.name.object)
            )
        ):
            resource = texel_expr.name.object
            return self.texel_swizzle_selects_components(
                self.texel_component_count(self.expression_raw_type(resource)), member
            )
        if not isinstance(texel_expr, ArrayAccessNode):
            return False
        resource = texel_expr.array
        if (
            isinstance(resource, ArrayAccessNode)
            and isinstance(resource.array, MemberAccessNode)
            and str(resource.array.member).lower() == "sample"
        ):
            resource = resource.array.object
        return self.texel_swizzle_selects_components(
            self.texel_component_count(self.expression_raw_type(resource)), member
        )

    def texel_swizzle_selects_components(self, count, member):
        if count is None or not member or len(member) > 4:
            return False
        if count == 1:
            return member in {"x", "r"}
        return any(
            all(ch in names[:count] for ch in member) for names in ("xyzw", "rgba")
        )

    RUNTIME_FORMAT_STORAGE_TEXTURES = (
        "RWTexture1D",
        "RWTexture1DArray",
        "RWTexture2D",
        "RWTexture2DArray",
        "RWTexture3D",
        "RasterizerOrderedTexture1D",
        "RasterizerOrderedTexture1DArray",
        "RasterizerOrderedTexture2D",
        "RasterizerOrderedTexture2DArray",
        "RasterizerOrderedTexture3D",
    )

    def is_runtime_format_storage_texture(self, node):
        """True for typed storage textures, whose format the runtime chooses.

        HLSL declares only the element type of a typed UAV; the bound view
        decides the storage format. Unless the source names a format (for
        example ``[[vk::image_format]]``), the declaration carries
        ``@ format(unknown)`` so targets that need a format do not guess one.
        """
        if self.raw_type_base(getattr(node, "vtype", None)) not in (
            self.RUNTIME_FORMAT_STORAGE_TEXTURES
        ):
            return False
        for attribute in getattr(node, "attributes", None) or []:
            name = str(getattr(attribute, "name", "") or "").lower()
            if name == "format" or "image_format" in name:
                return False
        return True

    SAMPLED_TEXTURE_TYPES = {
        "Texture1D": "texture1D",
        "Texture1DArray": "texture1DArray",
        "Texture2D": "texture2D",
        "Texture2DArray": "texture2DArray",
        "Texture3D": "texture3D",
        "TextureCube": "textureCube",
        "TextureCubeArray": "textureCubeArray",
        "Texture2DMS": "texture2DMS",
        "Texture2DMSArray": "texture2DMSArray",
    }

    def map_sampled_texture_type(self, base_type, generic_type):
        """Integer sampled textures use the integer sampler types."""
        sampler_type = self.SAMPLED_TEXTURE_TYPES.get(base_type)
        if sampler_type is None:
            return None
        arguments = self.split_generic_arguments(generic_type)
        if not arguments:
            return None
        family = self.numeric_type_family(
            self.normalize_resource_component_type(arguments[0])
        )
        if family == "uint":
            return f"u{sampler_type}"
        if family == "int":
            return f"i{sampler_type}"
        return None
