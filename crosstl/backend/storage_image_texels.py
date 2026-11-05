"""CrossGL format and texel metadata for HLSL and Slang storage images.

A typed UAV such as ``RWTexture2D<unorm float4>`` declares its texel type, and
the view bound at run time decides its format. Unless the source names the
format (DXC's ``[[vk::image_format(...)]]``, Slang's ``[format(...)]``), the
importers write ``@format(unknown)`` and keep the texel type in
``@texel(...)`` whenever it is not the four-component default, so that HLSL
and Slang output declare the original texel type again: it decides how many
components a load returns and a store writes, which typed loads the device has
to support, and whether the image can be an atomic's target.
"""

from __future__ import annotations

import re

from crosstl.translator.codegen.image_access_contracts import SUPPORTED_IMAGE_FORMATS

__all__ = [
    "IMAGE_FORMAT_ATTRIBUTES",
    "crossgl_image_format",
    "storage_image_element_type",
    "texel_attribute_arguments",
]

# Source attributes that name a storage image's format: DXC and Slang read
# ``vk::image_format``, and Slang also reads ``format``.
IMAGE_FORMAT_ATTRIBUTES = frozenset({"vk::image_format", "format"})

_TEXEL_SCALARS = {
    "float": ("float", "vec"),
    "int": ("int", "ivec"),
    "uint": ("uint", "uvec"),
    "dword": ("uint", "uvec"),
}

_TEXEL_TYPE = re.compile(
    r"(?:(?P<modifier>unorm|snorm)\s*)?(?P<scalar>float|int|uint|dword)(?P<count>[1-4]?)"
)
_TEXEL_VECTOR = re.compile(
    r"(?:(?P<modifier>unorm|snorm)\s*)?vector\s*<\s*(?P<scalar>float|int|uint|dword)"
    r"\s*,\s*(?P<count>[1-4])\s*>"
)


def crossgl_image_format(name):
    """CrossGL spelling of a source format name, ``unknown``, or ``None``.

    Slang spells formats as GLSL does (``rg8_snorm``); DXC writes the signed
    normalized ones without the underscore (``rg8snorm``). ``None`` means
    CrossGL has no such storage image format.
    """
    text = str(name).strip().strip('"').strip().lower()
    if text == "unknown" or text in SUPPORTED_IMAGE_FORMATS:
        return text
    if text.endswith("snorm") and not text.endswith("_snorm"):
        candidate = f"{text[: -len('snorm')]}_snorm"
        if candidate in SUPPORTED_IMAGE_FORMATS:
            return candidate
    return None


def storage_image_element_type(type_name):
    """The element type between the angle brackets of ``RWTexture2D<...>``, or ``None``."""
    text = str(type_name or "")
    start = text.find("<")
    end = text.rfind(">")
    if start < 0 or end <= start:
        return None
    return text[start + 1 : end].strip()


def texel_attribute_arguments(element_type):
    """``@texel`` arguments for a storage image's element type, or ``None``.

    ``None`` stands for the four-component default and for element types
    CrossGL has no ``@texel`` spelling for, such as ``half4``.
    """
    text = " ".join(str(element_type or "").split())
    match = _TEXEL_TYPE.fullmatch(text) or _TEXEL_VECTOR.fullmatch(text)
    if match is None:
        return None
    modifier = match.group("modifier")
    kind, prefix = _TEXEL_SCALARS[match.group("scalar")]
    count = int(match.group("count") or 1)
    if modifier is not None and kind != "float":
        return None
    if count == 4 and modifier is None:
        return None
    texel = kind if count == 1 else f"{prefix}{count}"
    return f"{modifier}, {texel}" if modifier else texel
