"""SPIR-V image types of separate CrossGL textures.

A separate texture (``texture2D``, ``utexture2DArray``, ``texture2DMS``, the depth ``texture2DShadow``; HLSL
``Texture2D<T>`` and MSL ``texture2d<T>``/``depth2d<T>``) is an ``OpTypeImage`` with ``Sampled`` 1 bound as a
``SAMPLED_IMAGE`` descriptor, not a ``COMBINED_IMAGE_SAMPLER``. Sampling pairs it with a separate sampler through
``OpSampledImage``; fetches and queries read the image itself (SPIR-V 1.0 section 3.32.6, Vulkan "Sampled Image"
descriptor type).
"""

import re
from typing import Dict, Optional

__all__ = ["separate_texture_type_info"]

_SEPARATE_TEXTURE = re.compile(r"([iu]?)texture(1D|2D|3D|Cube)(MS)?(Array)?(Shadow)?")
_COMPONENT_TYPES = {"": "float", "i": "int", "u": "uint"}


def separate_texture_type_info(type_name: str) -> Optional[Dict[str, object]]:
    """Image metadata of a separate texture type name, or ``None`` when ``type_name`` is not one.

    Multisampling exists only for 2D images and 3D images have no array form (SPIR-V ``OpTypeImage``). A depth
    texture (``Shadow``) holds floating-point depth and is declared with ``Depth`` 1; it has no multisampled
    CrossGL spelling.
    """
    match = _SEPARATE_TEXTURE.fullmatch(str(type_name))
    if match is None:
        return None
    prefix, dim, multisampled, arrayed, shadow = match.groups()
    if multisampled and dim != "2D":
        return None
    if arrayed and dim == "3D":
        return None
    if shadow and (prefix or multisampled or dim == "3D"):
        return None
    return {
        "kind": "texture",
        "component_type": _COMPONENT_TYPES[prefix],
        "dim": dim,
        "depth": 1 if shadow else 0,
        "arrayed": 1 if arrayed else 0,
        "multisampled": 1 if multisampled else 0,
        "sampled": 1,
        "format": "Unknown",
    }
