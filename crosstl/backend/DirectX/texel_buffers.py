"""CrossGL import of HLSL typed (texel) buffers.

``Buffer<T>`` is a read-only texel buffer and ``RWBuffer<T>`` (or
``RasterizerOrderedBuffer<T>``) a storage texel buffer. CrossGL spells them as
the GLSL texel-buffer types ``samplerBuffer`` and ``imageBuffer`` with the
``i``/``u`` prefix of the element family, so they reach every target as
texel buffers rather than as untyped structured storage:

- ``buffer[i]`` and ``buffer.Load(i)`` become ``texelFetch(buffer, i)``;
- ``rwBuffer[i]`` reads and writes become ``imageLoad`` and ``imageStore``,
  and ``Interlocked*`` on an element becomes ``imageAtomic*``;
- ``GetDimensions(width)`` becomes ``textureSize`` or ``imageSize``.

CrossGL texel functions return four components, so reads are narrowed to the
declared element width and stores are widened, exactly as for textures.
"""

from ..common_ast import ArrayAccessNode

SAMPLED_TEXEL_BUFFER_TYPES = frozenset({"Buffer"})
STORAGE_TEXEL_BUFFER_TYPES = frozenset({"RWBuffer", "RasterizerOrderedBuffer"})


class HLSLTexelBufferMixin:
    """Texel-buffer members of ``HLSLToCrossGLConverter``."""

    def is_sampled_texel_buffer_type(self, type_name):
        return self.raw_type_base(type_name) in SAMPLED_TEXEL_BUFFER_TYPES

    def is_storage_texel_buffer_type(self, type_name):
        return self.raw_type_base(type_name) in STORAGE_TEXEL_BUFFER_TYPES

    def is_texel_buffer_type(self, type_name):
        return self.is_sampled_texel_buffer_type(
            type_name
        ) or self.is_storage_texel_buffer_type(type_name)

    def map_texel_buffer_type(self, base_type, generic_type):
        """CrossGL texel-buffer type of ``Buffer<T>`` or ``RWBuffer<T>``."""
        if base_type in SAMPLED_TEXEL_BUFFER_TYPES:
            crossgl_type = "samplerBuffer"
        elif base_type in STORAGE_TEXEL_BUFFER_TYPES:
            crossgl_type = "imageBuffer"
        else:
            return None
        arguments = self.split_generic_arguments(generic_type)
        if len(arguments) != 1:
            return None
        family = self.numeric_type_family(
            self.normalize_resource_component_type(arguments[0])
        )
        if family is None or self.storage_image_component_count(arguments[0]) is None:
            self.refuse_import(
                f"DirectX {base_type}<{generic_type}> needs a scalar or vector "
                "element type for a texel buffer"
            )
        if family == "uint":
            return f"u{crossgl_type}"
        if family == "int":
            return f"i{crossgl_type}"
        return crossgl_type

    def is_sampled_texel_buffer_element_access(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return False
        if not self.is_sampled_texel_buffer_type(self.expression_raw_type(expr.array)):
            return False
        return (
            self.array_access_depth(expr)
            == self.expression_resource_array_dims(expr) + 1
        )

    def generate_sampled_texel_buffer_fetch(self, expr, is_main=False, narrow=True):
        """Render ``buffer[i]`` on a ``Buffer<T>`` as a narrowed ``texelFetch``."""
        if not self.is_sampled_texel_buffer_element_access(expr):
            return None
        buffer = self.generate_expression(expr.array, is_main)
        coord = self.generate_expression(expr.index, is_main)
        suffix = (
            self.texel_narrowing_suffix(self.expression_raw_type(expr.array))
            if narrow
            else ""
        )
        return f"texelFetch({buffer}, {coord}){suffix}"

    def texel_buffer_load_arguments(self, obj, rendered_args, descriptor):
        """``Buffer<T>.Load(location)`` takes no mip level."""
        if not self.is_sampled_texel_buffer_type(descriptor.get("resource_type")):
            return None
        return [obj, *rendered_args]

    def generate_texel_buffer_dimensions(self, stmt, indent=0, is_main=False):
        """Lower ``GetDimensions(width)`` on a texel buffer to a size query."""
        resource_type = self.expression_raw_type(stmt.name.object)
        if not self.is_texel_buffer_type(resource_type):
            return None
        base = self.raw_type_base(resource_type)
        if len(stmt.args) != 1:
            return self.refuse_import(
                f"DirectX GetDimensions on {base} takes one width output, "
                f"got {len(stmt.args)} arguments"
            )
        query = (
            "textureSize"
            if self.is_sampled_texel_buffer_type(resource_type)
            else "imageSize"
        )
        obj = self.generate_expression(stmt.name.object, is_main)
        target = self.generate_expression(stmt.args[0], is_main)
        target_type = self.map_type(
            self.raw_type_base(self.expression_raw_type(stmt.args[0])) or "uint"
        )
        if target_type not in {"uint", "int", "float"}:
            return self.refuse_import(
                f"DirectX GetDimensions on {base} writes a scalar width, got "
                f"{target_type} {target}"
            )
        size = f"{query}({obj})"
        if target_type != "int":
            size = f"{target_type}({size})"
        return f"{'    ' * indent}{target} = {size};\n"
