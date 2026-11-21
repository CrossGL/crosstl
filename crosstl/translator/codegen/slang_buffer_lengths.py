"""``length()`` of an array member of a GLSL buffer block in Slang output.

The Slang emitter lowers std140 and std430 GLSL buffer blocks to byte-address buffers, but it printed
``block.member.length()`` unchanged, and slangc refused the member access on a ``RWByteAddressBuffer``. A sized
array's length is its declared size. A runtime array's length is the number of whole elements that the bound buffer
holds after the array's offset (GLSL 4.60 section 4.1.9). As in DirectX output (``buffer_length``), it is computed
from ``ByteAddressBuffer.GetDimensions`` through a helper function: ``int((cgl_buffer_bytes(b) - offset) / stride)``.

A CrossGL ``buffer T* b`` is a Slang structured buffer, and ``b.length()`` is the number of elements bound, which
``GetDimensions`` writes as its count. Slang's structured buffers have no ``length`` member, so slangc refused the
call as written; it is rendered as ``int`` of the element-count helper that ``buffer_dimensions(b)`` calls.
"""

from __future__ import annotations

from ..ast import MemberAccessNode

__all__ = [
    "slang_glsl_buffer_block_length",
    "slang_structured_buffer_length",
]


def slang_glsl_buffer_block_length(codegen, func_expr, args):
    """Slang for ``block.member.length()`` when ``block`` is a lowered GLSL buffer block, or ``None``."""
    if not (
        isinstance(func_expr, MemberAccessNode)
        and getattr(func_expr, "member", None) == "length"
        and not args
    ):
        return None
    object_expr = getattr(func_expr, "object_expr", getattr(func_expr, "object", None))
    access = codegen.glsl_buffer_block_member_access(object_expr)
    if access is None or not access.get("is_array"):
        return None
    size = access.get("array_size")
    if size is not None:
        return f"int({codegen.generate_expression(size)})"
    resource = "ByteAddressBuffer" if access.get("readonly") else "RWByteAddressBuffer"
    rendered = codegen.slang_buffer_length.runtime_array_length(
        access["buffer"], resource, access.get("offset"), access.get("stride")
    )
    if rendered is None:
        return codegen.refuse_lowering(
            f"Slang GLSL buffer block: length() of runtime array '{access.get('member')}' has no "
            "constant offset and stride in its byte-address buffer",
            "0",
        )
    return rendered


def slang_structured_buffer_length(codegen, func_expr, args):
    """Slang for ``b.length()`` when ``b`` is a structured buffer, or ``None``."""
    if not (
        isinstance(func_expr, MemberAccessNode)
        and getattr(func_expr, "member", None) == "length"
        and not args
    ):
        return None
    object_expr = getattr(func_expr, "object_expr", getattr(func_expr, "object", None))
    buffer_type = codegen.structured_buffer_resource_type(object_expr)
    if not codegen.is_structured_buffer_resource_type(buffer_type):
        return None
    helper = codegen.buffer_dimensions_helper_name(buffer_type)
    return f"int({helper}({codegen.generate_expression(object_expr)}))"
