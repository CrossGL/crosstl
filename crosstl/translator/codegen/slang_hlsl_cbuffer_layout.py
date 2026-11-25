"""Slang constant buffers declared with HLSL constant-buffer packing.

A CrossGL ``cbuffer`` marked ``@ buffer_layout(hlsl_cbuffer)`` holds its members where HLSL packs them into 16-byte
registers: ``float a; float3 b;`` puts ``b`` at byte 4. slangc lays a constant buffer out by HLSL packing only for
Direct3D; for SPIR-V it uses std140, so ``b`` moved to byte 16 and every read after it took other bytes than the
host wrote.

slangc keeps a member's ``packoffset(c<N>.<component>)`` for SPIR-V as well, so the generator writes one for every
member of such a buffer whose offsets differ between the two rule sets. ``packoffset`` places only the buffer's own
members. A buffer is refused instead when its members cannot be placed that way: a struct, array or matrix inside a
member that std140 lays out differently from HLSL (its inside stays std140), or a member that HLSL packs into the
padding std140 reserves after an array, matrix or struct. ``float a[2]; float b;`` puts ``b`` at byte 20, inside the
32 bytes a std140 ``float[2]`` takes, and Vulkan's layout rules forbid that overlap (Vulkan 1.3 section 15.6.4).
"""

from __future__ import annotations

from .buffer_member_layout import (
    member_explicit_offset,
    member_is_row_major,
)
from .glsl_block_layout import (
    STD140_LAYOUT,
    block_size,
    block_struct_placements,
    check_block_interior_matches_hlsl,
    crossgl_struct_layout_type,
    declared_buffer_layout,
    member_explicit_alignment,
)
from .hlsl_buffer_layout import (
    CBUFFER_REGISTER_SIZE,
    HLSL_CBUFFER_LAYOUT,
    HlslLayoutError,
    cbuffer_struct_placements,
)

__all__ = ["slang_hlsl_cbuffer_packoffsets"]

_COMPONENTS = "xyzw"


def _member_tuples(members, placed):
    return [
        (
            member.name,
            member.member_type,
            member_is_row_major(member),
            member_explicit_offset(member) if placed else None,
            member_explicit_alignment(member) if placed else None,
        )
        for member in members
        if hasattr(member, "member_type")
    ]


def _packoffset(offset):
    register, byte = divmod(offset, CBUFFER_REGISTER_SIZE)
    component = byte // 4
    if component == 0:
        return f"c{register}"
    return f"c{register}.{_COMPONENTS[component]}"


def _hlsl_placements(codegen, node):
    """The HLSL placements of the buffer's members; raises :class:`HlslLayoutError` when std140 cannot hold them."""

    def struct_members(name):
        struct = codegen.user_structs_by_name.get(name)
        if struct is None:
            return None
        return _member_tuples(getattr(struct, "members", None) or [], True)

    layout_type = crossgl_struct_layout_type(
        node.name, _member_tuples(node.members, True), struct_members, (node.name,)
    )
    hlsl, _size = cbuffer_struct_placements(layout_type)
    std140_end = 0
    for placement in hlsl:
        check_block_interior_matches_hlsl(
            placement.name, placement.layout_type, STD140_LAYOUT
        )
        if placement.offset < std140_end:
            raise HlslLayoutError(
                f"member '{placement.name}' sits at byte {placement.offset} under HLSL packing, inside the "
                f"{STD140_LAYOUT} extent of the member before it, which ends at byte {std140_end}"
            )
        std140_end = placement.offset + block_size(placement.layout_type, STD140_LAYOUT)
    plain = crossgl_struct_layout_type(
        node.name, _member_tuples(node.members, False), struct_members, (node.name,)
    )
    std140, _std140_size = block_struct_placements(plain, STD140_LAYOUT, root=True)
    return hlsl, [placement.offset for placement in std140]


def slang_hlsl_cbuffer_packoffsets(codegen, node):
    """``{member name: " : packoffset(...)"}`` for an HLSL-packed constant buffer, or ``None`` when none is needed.

    ``None`` stands for a buffer without ``@ buffer_layout(hlsl_cbuffer)``, one whose members name their own
    ``packoffset``, and one whose HLSL offsets are its std140 offsets. A buffer that cannot be placed is refused.
    """
    if declared_buffer_layout(getattr(node, "attributes", None)) != HLSL_CBUFFER_LAYOUT:
        return None
    members = [member for member in node.members if hasattr(member, "member_type")]
    if any(
        str(getattr(attr, "name", "")).lower() == "packoffset"
        for member in members
        for attr in getattr(member, "attributes", None) or ()
    ):
        return None
    try:
        hlsl, std140_offsets = _hlsl_placements(codegen, node)
    except HlslLayoutError as exc:
        codegen.refuse_lowering(
            f"Slang constant buffer '{node.name}' with HLSL packing: {exc}; slangc lays a SPIR-V constant "
            "buffer out by std140, and packoffset places only the buffer's own members"
        )
        return None
    if [placement.offset for placement in hlsl] == std140_offsets:
        return None
    return {
        placement.name: f" : packoffset({_packoffset(placement.offset)})"
        for placement in hlsl
    }
