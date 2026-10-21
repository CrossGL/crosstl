"""Entry-input members naming a built-in the stage does not have.

An HLSL struct shared by two stages carries the semantics of both: a vertex output struct read as the pixel input
keeps ``[[vk::builtin("PointSize")]] float size : PSIZE``. SPIR-V has no ``PointSize`` input in a fragment shader
(Vulkan's ``VUID-PointSize-PointSize-04314``), so the member cannot become an interface variable there. When no
function of the program reads a member of that name, the member's value cannot be observed and the entry's copy
of the struct takes an undefined value for it; a member that is read is still refused by the emitter.
"""

from __future__ import annotations

from ..ast import MemberAccessNode
from ..semantics.rewrite import child_slots, get_child

__all__ = ["unread_unavailable_builtin_input"]


def _reads_member(program, name):
    stack, seen = [program], set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, MemberAccessNode) and node.member == name:
            return True
        stack.extend(get_child(node, field, key) for field, key in child_slots(node))
    return False


def unread_unavailable_builtin_input(
    emitter, program, execution_model, node, type_id, member_name
):
    """Whether entry-input member ``member_name`` names a built-in ``execution_model`` lacks and nothing reads it."""
    if program is None or execution_model is None or node is None:
        return False
    semantic = emitter.entry_point_interface_semantic(
        execution_model, "Input", node, type_id, member_name
    )
    info = emitter.stage_builtin_info(str(semantic)) if semantic is not None else None
    if info is None or execution_model in info[3]:
        return False
    return not _reads_member(program, member_name)
