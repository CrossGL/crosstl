"""Same-width word views of 32-bit storage for typed-buffer targets.

DirectX structured buffers and OpenGL storage blocks keep one physical 32-bit
element per backing word. A reinterpreted view whose element is also one
32-bit scalar therefore addresses whole backing words: reads and writes are
bit casts of one element, and an atomic through the view is an atomic on that
element. Narrower or wider views need byte assembly and stay outside this
contract.

A backing record whose only member is one 32-bit scalar has that scalar's
layout (see ``scalar_wrapper_views``), so its member is the backing word.
"""

from .pointer_reinterpret import scalar_storage_layout
from .scalar_wrapper_views import scalar_wrapper_member


def wrapper_scalar_type_name(type_name, structs_by_name):
    """Return ``(member, scalar)`` when ``type_name`` names a scalar wrapper."""

    if not isinstance(type_name, str):
        return None
    member = scalar_wrapper_member((structs_by_name or {}).get(type_name.strip()))
    if member is None:
        return None
    name, member_type = member
    return name, member_type.name


def reinterpret_source_layout(type_name, structs_by_name):
    """Return ``(layout, member)`` of a backing element, looking through wrappers."""

    layout = scalar_storage_layout(type_name)
    if layout is not None:
        return layout, None
    wrapper = wrapper_scalar_type_name(type_name, structs_by_name)
    if wrapper is None:
        return None, None
    member, scalar = wrapper
    return scalar_storage_layout(scalar), member


def is_same_width_word_view(reinterpretation):
    """Whether a reinterpretation views each 32-bit backing word as one scalar."""

    if not reinterpretation:
        return False
    source = reinterpretation.get("source_layout")
    target = reinterpretation.get("target_layout")
    return (
        source is not None
        and target is not None
        and source.bit_width == 32
        and target.bit_width == 32
        and reinterpretation.get("target_width", 1) == 1
        and not reinterpretation.get("lane_gather")
        and reinterpretation.get("private_word_view") is None
    )


def word_view_index(byte_offset, element_index):
    """Return the backing word index of view element ``element_index``.

    ``byte_offset`` is the view's rendered byte offset into the backing
    storage. Both operands are rendered target expressions.
    """

    element_index = str(element_index)
    element_bytes = (
        element_index if element_index in {"0", "0u"} else f"({element_index} * 4)"
    )
    byte_offset = str(byte_offset or "0")
    if byte_offset in {"0", "0u"}:
        byte_index = element_bytes
    elif element_bytes in {"0", "0u"}:
        byte_index = byte_offset
    else:
        byte_index = f"({byte_offset} + {element_bytes})"
    if byte_index in {"0", "0u"}:
        return "0"
    return f"(({byte_index}) / 4)"


HLSL_BIT_CASTS = {"float": "asfloat", "int": "asint", "uint": "asuint"}
GLSL_FROM_UINT = {"float": "uintBitsToFloat", "int": "int", "uint": ""}
GLSL_TO_UINT = {"float": "floatBitsToUint", "int": "uint", "uint": ""}


def layout_scalar_kind(layout):
    """Return ``float``, ``int`` or ``uint`` for a 32-bit layout."""

    if layout.kind == "floating":
        return "float"
    return "int" if layout.signed else "uint"


def hlsl_word_bit_cast(value, source_layout, target_layout):
    """Reinterpret a 32-bit HLSL value from one layout to another."""

    if layout_scalar_kind(source_layout) == layout_scalar_kind(target_layout):
        return value
    return f"{HLSL_BIT_CASTS[layout_scalar_kind(target_layout)]}({value})"


def glsl_word_bit_cast(value, source_layout, target_layout):
    """Reinterpret a 32-bit GLSL value from one layout to another."""

    source_kind = layout_scalar_kind(source_layout)
    target_kind = layout_scalar_kind(target_layout)
    if source_kind == target_kind:
        return value
    if source_kind != "uint":
        value = f"{GLSL_TO_UINT[source_kind]}({value})"
    if target_kind == "uint":
        return value
    return f"{GLSL_FROM_UINT[target_kind]}({value})"
