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
