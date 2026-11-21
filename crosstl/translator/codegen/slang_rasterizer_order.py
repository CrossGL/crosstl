"""Slang rasterizer-ordered views for CrossGL's ``@rasterizer_ordered`` resources.

CrossGL marks a storage buffer or storage image whose accesses must happen in rasterization order with
``@ rasterizer_ordered`` (HLSL's rasterizer-ordered views, GLSL's GL_ARB_fragment_shader_interlock critical
section). Slang has HLSL's ``RasterizerOrdered*`` resource types, which slangc compiles for SPIR-V to the
``PixelInterlockOrderedEXT`` execution mode with ``OpBeginInvocationInterlockEXT`` and
``OpEndInvocationInterlockEXT`` around the fragment shader. The generator declared the plain ``RW*`` type, so
overlapping fragments raced on the resource.

A ``@ rasterizer_ordered`` declaration whose Slang type has no rasterizer-ordered form, such as a read-only
resource, is refused.
"""

from __future__ import annotations

from .array_utils import split_array_type_suffix

__all__ = ["is_slang_rasterizer_ordered", "slang_rasterizer_ordered_type"]

_RASTERIZER_ORDERED_TYPES = {
    "RWTexture1D": "RasterizerOrderedTexture1D",
    "RWTexture1DArray": "RasterizerOrderedTexture1DArray",
    "RWTexture2D": "RasterizerOrderedTexture2D",
    "RWTexture2DArray": "RasterizerOrderedTexture2DArray",
    "RWTexture3D": "RasterizerOrderedTexture3D",
    "RWBuffer": "RasterizerOrderedBuffer",
    "RWStructuredBuffer": "RasterizerOrderedStructuredBuffer",
    "RWByteAddressBuffer": "RasterizerOrderedByteAddressBuffer",
}
_ORDERED_TYPES = frozenset(_RASTERIZER_ORDERED_TYPES.values())


def is_slang_rasterizer_ordered(node) -> bool:
    """Whether the declaration ``node`` carries ``@ rasterizer_ordered``."""
    return any(
        str(getattr(attribute, "name", "") or "").lower() == "rasterizer_ordered"
        for attribute in getattr(node, "attributes", None) or ()
    )


def slang_rasterizer_ordered_type(codegen, mapped_type, node):
    """``mapped_type`` as its ``RasterizerOrdered*`` form when ``node`` is ``@ rasterizer_ordered``."""
    if node is None or not isinstance(mapped_type, str):
        return mapped_type
    if not is_slang_rasterizer_ordered(node):
        return mapped_type
    base_type, array_suffix = (
        split_array_type_suffix(mapped_type)
        if "[" in mapped_type and mapped_type.endswith("]")
        else (mapped_type, "")
    )
    resource_name, bracket, generic_arguments = base_type.partition("<")
    resource_name = resource_name.strip()
    if resource_name in _ORDERED_TYPES:
        return mapped_type
    ordered = _RASTERIZER_ORDERED_TYPES.get(resource_name)
    if ordered is None:
        name = getattr(node, "name", None)
        return codegen.refuse_lowering(
            f"Slang @rasterizer_ordered: '{name}' is a {resource_name}, which has no rasterizer-ordered "
            "form; only read-write buffers and storage textures do",
            mapped_type,
        )
    return f"{ordered}{bracket}{generic_arguments}{array_suffix}"
