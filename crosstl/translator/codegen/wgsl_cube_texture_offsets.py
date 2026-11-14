"""Offset texture lookups on cube array textures, which WGSL output refuses.

``textureOffset``, ``textureLodOffset``, ``textureGradOffset`` and the CrossGL compare forms ``textureCompareOffset``
and ``textureCompareLodOffset`` add a constant texel offset to a lookup. WGSL's ``textureSample``,
``textureSampleBias``, ``textureSampleLevel``, ``textureSampleGrad``, ``textureSampleCompare`` and
``textureSampleCompareLevel`` take that offset only for 2D, 2D array and 3D textures, and GLSL defines no offset
lookup on cube textures either, so no WGSL built-in performs one on a ``texture_cube_array`` or
``texture_depth_cube_array``. The WGSL emitter refuses these lookups with
:class:`WGSLCubeTextureOffsetUnsupportedError` rather than write a call that naga rejects.
"""

from __future__ import annotations

import re

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "OFFSET_TEXTURE_LOOKUPS",
    "WGSLCubeTextureOffsetUnsupportedError",
    "is_cube_array_texture_type",
]

# Normalized names of the lookups that take a texel offset.
OFFSET_TEXTURE_LOOKUPS = frozenset(
    {
        "texturecompareoffset",
        "texturecomparelodoffset",
        "texturegradoffset",
        "texturelodoffset",
        "textureoffset",
    }
)

_CUBE_ARRAY_TEXTURE_TYPES = frozenset(
    {"texture_cube_array", "texture_depth_cube_array"}
)


class WGSLCubeTextureOffsetUnsupportedError(TranslationDiagnostic):
    """An offset lookup on a cube array texture, which no WGSL built-in performs."""

    def __init__(self, function_name, texture):
        self.function_name = function_name
        self.texture = texture
        self.missing_capabilities = ("wgsl.cube-texture-offset",)
        super().__init__(
            f"WGSL target cannot lower {function_name}() on a cube array texture ({texture}): WGSL texture lookups "
            "take an offset only on 2D, 2D array and 3D textures",
            target="wgsl",
            code=f"{CODE_PREFIX}wgsl-cube-texture-offset-unsupported",
        )


def is_cube_array_texture_type(texture_type):
    """Whether ``texture_type``, a WGSL type such as ``texture_cube_array<f32>``, is a cube array texture."""
    return (
        bool(texture_type)
        and re.sub(r"<[^>]*>$", "", texture_type) in _CUBE_ARRAY_TEXTURE_TYPES
    )
