"""Storage image texel formats in HLSL and Slang declarations.

A CrossGL storage image names its texel format, such as ``layout(rgba8)``.
HLSL and Slang spell a format through the texel type and, where the type
alone does not determine the format, through an attribute:

- an 8- or 16-bit normalized format is a ``unorm`` or ``snorm`` float texel,
  so that ``rgba8`` is ``RWTexture2D<unorm float4>`` rather than the 32-bit
  float image ``RWTexture2D<float4>``;
- DXC derives the SPIR-V image format from the texel's scalar type only, as
  32-bit float, int or uint, and ignores ``unorm`` and ``snorm``; any other
  format needs ``[[vk::image_format(...)]]``, which DXC accepts only when it
  targets SPIR-V (``__spirv__``) and otherwise warns about;
- Slang reads ``[format(...)]`` on every target.
"""

import re

from .image_access_contracts import (
    image_format_channel_count,
    image_format_component_kind,
)

NORMALIZED_FORMAT_MODIFIERS = {
    "r8": "unorm",
    "rg8": "unorm",
    "rgba8": "unorm",
    "r16": "unorm",
    "rg16": "unorm",
    "rgba16": "unorm",
    "r8_snorm": "snorm",
    "rg8_snorm": "snorm",
    "rgba8_snorm": "snorm",
    "r16_snorm": "snorm",
    "rg16_snorm": "snorm",
    "rgba16_snorm": "snorm",
}

# The formats DXC gives a storage image whose texel is a 32-bit scalar or a
# vector of them, keyed by the texel's component kind and channel count.
DXC_DERIVED_FORMATS = {
    ("float", 1): "r32f",
    ("float", 2): "rg32f",
    ("float", 4): "rgba32f",
    ("int", 1): "r32i",
    ("int", 2): "rg32i",
    ("int", 4): "rgba32i",
    ("uint", 1): "r32ui",
    ("uint", 2): "rg32ui",
    ("uint", 4): "rgba32ui",
}


def normalized_texel_type(image_format, component_type):
    """Return ``component_type`` with the format's unorm/snorm modifier."""
    modifier = NORMALIZED_FORMAT_MODIFIERS.get(image_format)
    if modifier is None or component_type is None:
        return component_type
    return f"{modifier} {component_type}"


def dxc_derived_format(image_format):
    """Return the format DXC derives from the texel type CrossTL writes."""
    key = (
        image_format_component_kind(image_format),
        image_format_channel_count(image_format),
    )
    return DXC_DERIVED_FORMATS.get(key)


def dxc_vulkan_image_format_prefix(image_format):
    """Return the guarded ``vk::image_format`` lines a declaration needs."""
    if image_format is None or dxc_derived_format(image_format) == image_format:
        return ""
    # DXC spells a signed normalized format without the underscore: rg8snorm.
    dxc_format = image_format.replace("_snorm", "snorm")
    return "#ifdef __spirv__\n" f'[[vk::image_format("{dxc_format}")]]\n' "#endif\n"


def slang_image_format_prefix(image_format):
    """Return the Slang ``[format(...)]`` attribute for an explicit format."""
    if image_format is None:
        return ""
    return f'[format("{image_format}")] '


# ``@texel(T)`` or ``@texel(unorm, T)`` names the texel type of a storage image
# whose format the runtime chooses (``@format(unknown)``), as an HLSL or Slang
# source declared it: a typed UAV's element type decides how many components a
# load returns and a store writes, which loads the device must support, and
# whether the image can be an atomic's target, while the bound view decides the
# format. Without the attribute the texel has four components.
TEXEL_ATTRIBUTE = "texel"
TEXEL_MODIFIERS = frozenset({"unorm", "snorm"})
CROSSGL_TEXEL_TYPES = {
    "float": ("float", 1),
    "vec2": ("float", 2),
    "vec3": ("float", 3),
    "vec4": ("float", 4),
    "int": ("int", 1),
    "ivec2": ("int", 2),
    "ivec3": ("int", 3),
    "ivec4": ("int", 4),
    "uint": ("uint", 1),
    "uvec2": ("uint", 2),
    "uvec3": ("uint", 3),
    "uvec4": ("uint", 4),
}


def crossgl_image_component_kind(image_type):
    """Return ``float``, ``int`` or ``uint``: the component kind of a CrossGL image type."""
    text = str(image_type).strip()
    if text.startswith("iimage"):
        return "int"
    if text.startswith("uimage"):
        return "uint"
    return "float"


def texel_attribute(node):
    """A declaration's ``@texel`` attribute, or ``None``."""
    return next(
        (
            attribute
            for attribute in getattr(node, "attributes", None) or []
            if str(getattr(attribute, "name", "")).lower() == TEXEL_ATTRIBUTE
        ),
        None,
    )


def declared_texel_type(node, image_type, explicit_format, attribute_value_to_string):
    """Return the HLSL texel type a storage image's ``@texel`` declares, or ``None``.

    Raises ``ValueError`` for a malformed attribute, for one whose component kind
    is not the image's, and for one on an image that names its format, whose
    texel type the format decides.
    """
    attribute = texel_attribute(node)
    if attribute is None:
        return None
    name = getattr(node, "name", None) or "<unnamed>"
    arguments = [
        str(attribute_value_to_string(argument)).strip()
        for argument in getattr(attribute, "arguments", None) or []
    ]
    spelled = f"@texel({', '.join(arguments)})"
    modifier = arguments[0] if len(arguments) == 2 else None
    shape = CROSSGL_TEXEL_TYPES.get(arguments[-1]) if arguments else None
    if (
        len(arguments) not in (1, 2)
        or shape is None
        or (modifier is not None and modifier not in TEXEL_MODIFIERS)
    ):
        raise ValueError(
            f"storage image '{name}': {spelled} must name a scalar or vector "
            "texel type, optionally after unorm or snorm"
        )
    if explicit_format is not None:
        raise ValueError(
            f"storage image '{name}': {spelled} applies only to an image whose "
            f"format is unknown, but its format is {explicit_format}"
        )
    kind, count = shape
    image_kind = crossgl_image_component_kind(image_type)
    if kind != image_kind or (modifier is not None and kind != "float"):
        raise ValueError(
            f"storage image '{name}': {image_type} holds {image_kind} texels, "
            f"which {spelled} does not describe"
        )
    texel = kind if count == 1 else f"{kind}{count}"
    return f"{modifier} {texel}" if modifier else texel


_TEXEL_MODIFIER_PREFIX = re.compile(r"^(?:unorm|snorm)\s+")
_TEXEL_SHAPE = re.compile(r"(float|int|uint)([1-4]?)")


def strip_texel_modifier(element_type):
    """An HLSL texel type without its ``unorm`` or ``snorm`` modifier.

    A normalized texel has the shape and component kind of its float type.
    """
    if not isinstance(element_type, str):
        return element_type
    return _TEXEL_MODIFIER_PREFIX.sub("", element_type.strip())


def texel_shape(element_type):
    """``(component kind, count)`` of an HLSL or Slang texel type, or ``None``."""
    if not isinstance(element_type, str):
        return None
    match = _TEXEL_SHAPE.fullmatch(strip_texel_modifier(element_type))
    if match is None:
        return None
    return match.group(1), int(match.group(2) or 1)
