"""CrossGL spellings of MSL sampled and depth textures.

An MSL ``texture2d<T>`` or ``depth2d<T>`` is a texture object separate from the ``sampler`` its ``sample`` calls
take (MSL 3.2 sections 2.9 and 6.12), so it imports as a CrossGL separate texture: ``texture2D``, ``itexture2D``,
``utexture2D``, and the depth family ``texture2DShadow``. A multisampled depth texture has no sampler state to
compare with (``depth2d_ms`` offers only ``read``), so it imports as the plain multisampled texture. Texel buffers
(``texture_buffer``) never take a sampler and keep the combined ``samplerBuffer`` spelling.
"""

import re

__all__ = [
    "METAL_DEPTH_TEXTURE_TYPES",
    "METAL_SAMPLED_TEXTURE_TYPES",
    "combined_texture_type",
]

# MSL base name -> CrossGL texture type of a floating-point element; integer elements add an ``i``/``u`` prefix.
METAL_SAMPLED_TEXTURE_TYPES = {
    "texture1d": "texture1D",
    "texture1d_array": "texture1DArray",
    "texture2d": "texture2D",
    "texture2d_array": "texture2DArray",
    "texture2d_ms": "texture2DMS",
    "texture2d_ms_array": "texture2DMSArray",
    "texture3d": "texture3D",
    "texturecube": "textureCube",
    "texturecube_array": "textureCubeArray",
    "texture_buffer": "samplerBuffer",
}

# MSL depth texture base name -> CrossGL type. Depth textures hold floating-point depth only.
METAL_DEPTH_TEXTURE_TYPES = {
    "depth2d": "texture2DShadow",
    "depth2d_array": "texture2DArrayShadow",
    "depthcube": "textureCubeShadow",
    "depthcube_array": "textureCubeArrayShadow",
    "depth2d_ms": "texture2DMS",
    "depth2d_ms_array": "texture2DMSArray",
}

_SEPARATE_TEXTURE = re.compile(r"([iu]?)texture(1D|2D|3D|Cube)(MS)?(Array)?(Shadow)?")


def combined_texture_type(type_name):
    """The combined CrossGL spelling (``sampler2D``, ``usampler2DArray``) of a separate texture type name.

    The importer classifies resources by image shape, which both spellings share; any other name is returned
    unchanged.
    """
    match = _SEPARATE_TEXTURE.fullmatch(str(type_name))
    if match is None:
        return type_name
    prefix, dim, multisampled, arrayed, shadow = match.groups()
    return f"{prefix}sampler{dim}{multisampled or ''}{arrayed or ''}{shadow or ''}"
