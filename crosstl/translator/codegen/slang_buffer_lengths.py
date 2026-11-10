"""``length()`` of an array member of a GLSL buffer block in Slang output.

The Slang emitter lowers std140 and std430 GLSL buffer blocks to byte-address buffers, but it printed
``block.member.length()`` unchanged, and slangc refused the member access on a ``RWByteAddressBuffer``. A sized
array's length is its declared size. A runtime array's length is the number of whole elements that the bound buffer
holds after the array's offset (GLSL 4.60 section 4.1.9). As in DirectX output (``buffer_length``), it is computed
from ``ByteAddressBuffer.GetDimensions`` through a helper function: ``int((cgl_buffer_bytes(b) - offset) / stride)``.
"""

from __future__ import annotations

from ..ast import MemberAccessNode

__all__ = ["slang_glsl_buffer_block_length"]


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
