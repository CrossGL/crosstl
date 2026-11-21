"""Raster order groups for CrossGL's rasterizer-ordered resources in Metal output.

A rasterizer-ordered resource (CrossGL ``@rasterizer_ordered``, HLSL's ``RasterizerOrderedBuffer`` and
``RasterizerOrderedTexture2D``, a SPIR-V or GLSL pixel interlock) orders the accesses that the fragments covering one
pixel make to it by the order of their primitives. MSL states the same with the ``raster_order_group(index)``
attribute on a fragment function's device buffer or texture argument (Metal Shading Language Specification,
"Rasterization order groups"): the accesses to the resources of one group are ordered between the fragments that
overlap.

Every rasterizer-ordered resource is placed in group 0. One group orders every pair of accesses that HLSL's views or
a SPIR-V critical section order, so nothing the source orders runs unordered. The Metal importer reads any group as
``@rasterizer_ordered``; merging several Metal groups into one orders more accesses than the source required, which
keeps every result the source can produce.

The attribute is written only where Metal accepts it and where it has a meaning:

- on a fragment function's arguments. Metal rejects it on a kernel or vertex argument ("invalid
  'raster_order_group' attribute for input declaration in a kernel function"), and no other stage has overlapping
  fragments to order, so its accesses keep their meaning without it;
- on device buffers and textures. Metal rejects it in the ``constant`` address space, which no fragment writes, so
  a read of it already returns the same value in every order.
"""

from __future__ import annotations

__all__ = [
    "RASTERIZER_ORDERED",
    "is_rasterizer_ordered_declaration",
    "metal_raster_order_group_attribute",
]

RASTERIZER_ORDERED = "rasterizer_ordered"
_ATTRIBUTE = " [[raster_order_group(0)]]"


def is_rasterizer_ordered_declaration(node):
    """Whether ``node`` (a global or parameter declaration) carries ``@rasterizer_ordered``."""
    return any(
        str(getattr(attribute, "name", "") or "").lower() == RASTERIZER_ORDERED
        for attribute in getattr(node, "attributes", None) or ()
    )


def metal_raster_order_group_attribute(node, shader_type, declaration):
    """The ``[[raster_order_group(0)]]`` that the Metal argument ``declaration`` of ``node`` takes, or ``""``.

    ``shader_type`` is the stage of the entry function the argument belongs to, and ``declaration`` the argument as
    the generator writes it, without its attributes.
    """
    if shader_type != "fragment" or not is_rasterizer_ordered_declaration(node):
        return ""
    if str(declaration).lstrip().startswith("constant "):
        return ""
    return _ATTRIBUTE
