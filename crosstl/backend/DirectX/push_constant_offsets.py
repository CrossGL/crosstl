"""DXC's push-constant member offsets, imported as CrossGL ``@offset`` qualifiers.

DXC declares ``[[vk::push_constant]] ConstantBuffer<T> name;`` (or ``[[vk::push_constant]] T name;``) in the
``PushConstant`` storage class and, by default, lays ``T`` out with its relaxed std430 rule (DXC SPIR-V codegen
reference, "Memory layout rules"): a vector of 32-bit or narrower components is aligned to its component size, and
moved to the next 16-byte boundary only when it would cross one there. Arrays, matrices and structs keep their
std430 alignment. The rule is Vulkan's relaxed block layout (Vulkan 1.3 section 15.6.4, VK_KHR_relaxed_block_layout,
core in Vulkan 1.1). ``struct T { float scale; float3 v; };`` thus puts ``v`` at byte 4, where std430 puts it at 16.

CrossGL lays a push-constant block out by the std430 rules, so the importer used to declare ``v`` at byte 16 in every
output while the host wrote it at byte 4. It also dropped a member's ``[[vk::offset(N)]]``. :func:`member_offsets`
now gives each member of a push-constant struct the ``@offset(N)`` that places it where DXC does, when std430 would
place it elsewhere, and every ``[[vk::offset(N)]]`` member its offset. A generator then reproduces the offsets or
refuses them (:mod:`crosstl.translator.codegen.glsl_block_layout`).

Offsets of members of a nested struct would apply wherever that struct is laid out, so a push-constant struct holding
a struct whose relaxed layout differs from std430 is refused, as is a push-constant struct that needs offsets and is
also laid out somewhere else (a buffer element, a constant-buffer member or a member of another struct). The command
line options ``-fvk-use-gl-layout``, ``-fvk-use-scalar-layout`` and ``-fvk-use-dx-layout`` select other rules; the
importer assumes DXC's default.
"""

from __future__ import annotations

import re

from crosstl.translator.codegen.glsl_block_layout import (
    STD430_LAYOUT,
    VEC4_ALIGNMENT,
    block_alignment,
    block_size,
    block_struct_placements,
    crossgl_layout_type,
    improperly_straddles,
)
from crosstl.translator.codegen.hlsl_buffer_layout import HlslLayoutError, align_to

__all__ = ["HLSLPushConstantLayoutError", "member_offsets", "vk_offset"]

PUSH_CONSTANT_ATTRIBUTE = "vk::push_constant"
VK_OFFSET_ATTRIBUTE = "vk::offset"
_CONSTANT_BUFFER = re.compile(r"ConstantBuffer\s*<\s*(\w+)\s*>")
# A vector of components this wide or narrower takes its component alignment under DXC's relaxed rule.
RELAXED_COMPONENT_BYTES = 4


class HLSLPushConstantLayoutError(ValueError):
    """A DXC push-constant struct whose relaxed layout CrossGL cannot record."""

    project_diagnostic_code = "project.translate.hlsl-push-constant-layout-unsupported"

    def __init__(self, name, reason):
        self.reason = reason
        super().__init__(
            f"Cannot import the layout of DXC push-constant buffer '{name}': {reason}"
        )


def _attribute(node, name):
    for attribute in getattr(node, "attributes", None) or []:
        if str(getattr(attribute, "name", "")) == name:
            return attribute
    return None


def vk_offset(member):
    """The byte offset of a member's ``[[vk::offset(N)]]``, or ``None``."""
    attribute = _attribute(member, VK_OFFSET_ATTRIBUTE)
    if attribute is None:
        return None
    arguments = list(
        getattr(attribute, "args", None) or getattr(attribute, "arguments", None) or []
    )
    value = arguments[0] if len(arguments) == 1 else None
    value = getattr(value, "value", value)
    try:
        offset = int(str(value).strip(), 0)
    except ValueError:
        offset = -1
    if offset < 0:
        raise HLSLPushConstantLayoutError(
            str(getattr(member, "name", "")),
            f"[[vk::offset]] of member '{getattr(member, 'name', '')}' is not a constant byte offset",
        )
    return offset


def _push_constant_struct(node, structs_by_name):
    """The name of the struct a ``[[vk::push_constant]]`` global holds, or ``None``."""
    if _attribute(node, PUSH_CONSTANT_ATTRIBUTE) is None:
        return None
    vtype = re.sub(r"\s+", "", str(getattr(node, "vtype", "") or ""))
    match = _CONSTANT_BUFFER.fullmatch(vtype)
    name = match.group(1) if match else vtype
    return name if name in structs_by_name else None


class _Planner:
    """CrossGL layout types of HLSL structs, built with the importer's own type spellings."""

    def __init__(self, codegen, structs_by_name):
        self.codegen = codegen
        self.structs_by_name = structs_by_name
        self.crossgl_names = {
            codegen.render_type_identifier(name): name for name in structs_by_name
        }

    def spelling(self, member):
        spelling = self.codegen.map_variable_type(member)
        for size in getattr(member, "array_sizes", None) or []:
            count = (
                size
                if isinstance(size, int)
                else self.codegen.evaluate_integer_constant_node(
                    size, self.codegen.integer_constant_values
                )
            )
            if not isinstance(count, int):
                raise HlslLayoutError(
                    f"member '{member.name}' has an array size that is not a constant"
                )
            spelling = f"{spelling}[{count}]"
        return spelling

    def members(self, crossgl_name):
        struct = self.structs_by_name.get(self.crossgl_names.get(crossgl_name))
        if struct is None:
            return None
        return [
            (
                str(member.name),
                self.spelling(member),
                "row_major" in self.codegen.matrix_layout_qualifiers(member),
                vk_offset(member),
                None,
            )
            for member in getattr(struct, "members", None) or []
            if getattr(member, "name", None) and hasattr(member, "vtype")
        ]

    def layout_type(self, struct_name):
        return crossgl_layout_type(
            self.codegen.render_type_identifier(struct_name), self.members
        )


def _relaxed_offset(layout_type, cursor):
    """Where DXC's relaxed rule places a member of ``layout_type`` at or after ``cursor``."""
    if (
        layout_type.kind != "vector"
        or layout_type.packed
        or layout_type.scalar_size > RELAXED_COMPONENT_BYTES
    ):
        return align_to(cursor, block_alignment(layout_type, STD430_LAYOUT))
    offset = align_to(cursor, layout_type.scalar_size)
    if improperly_straddles(offset, block_size(layout_type, STD430_LAYOUT)):
        offset = align_to(offset, VEC4_ALIGNMENT)
    return offset


def _relaxed_placements(layout_type):
    """``(name, layout type, offset, moved)`` of each member of a struct under DXC's relaxed rule; ``moved`` marks a
    member std430 places elsewhere after the same previous members."""
    offsets = layout_type.offsets or (None,) * len(layout_type.members)
    placements = []
    cursor = 0
    for (name, member_type), explicit in zip(layout_type.members, offsets):
        std430 = align_to(cursor, block_alignment(member_type, STD430_LAYOUT))
        offset = (
            explicit if explicit is not None else _relaxed_offset(member_type, cursor)
        )
        placements.append((name, member_type, offset, offset != std430))
        cursor = offset + block_size(member_type, STD430_LAYOUT)
    return placements


def _innermost(layout_type):
    while layout_type.kind == "array":
        layout_type = layout_type.element
    return layout_type


def _check_nested(block, path, layout_type):
    """Refuse a struct inside the push-constant struct that DXC lays out differently from std430."""
    inner = _innermost(layout_type)
    if inner.kind != "struct":
        return
    std430, _size = block_struct_placements(inner, STD430_LAYOUT)
    for (name, member_type, offset, _moved), placement in zip(
        _relaxed_placements(inner), std430
    ):
        if offset != placement.offset:
            raise HLSLPushConstantLayoutError(
                block,
                f"member '{path}.{name}' of struct '{inner.struct_name}' is at byte {offset} under DXC's relaxed "
                f"rule and at byte {placement.offset} under std430; CrossGL places the members of a nested "
                "struct by std430",
            )
        _check_nested(block, f"{path}.{name}", member_type)


def _laid_out_elsewhere(ast, struct_name, push_constants):
    """A declaration other than the push constants that lays ``struct_name`` out, or ``None``."""
    pattern = re.compile(rf"\b{re.escape(struct_name)}\b")
    for node in getattr(ast, "global_variables", None) or []:
        if id(node) in push_constants or getattr(node, "value", None) is not None:
            continue
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", None) or []}
        if qualifiers & {"static", "groupshared", "const"}:
            continue
        if pattern.search(str(getattr(node, "vtype", "") or "")):
            return f"global '{node.name}'"
    for cbuffer in getattr(ast, "cbuffers", None) or []:
        for member in getattr(cbuffer, "members", None) or []:
            if pattern.search(str(getattr(member, "vtype", "") or "")):
                return f"member '{member.name}' of constant buffer '{cbuffer.name}'"
    for struct in getattr(ast, "structs", None) or []:
        for member in getattr(struct, "members", None) or []:
            if pattern.search(str(getattr(member, "vtype", "") or "")):
                return f"member '{member.name}' of struct '{struct.name}'"
    return None


def _struct_offsets(planner, ast, struct_name, block, push_constants):
    try:
        layout_type = planner.layout_type(struct_name)
        placements = _relaxed_placements(layout_type)
        for name, member_type, _offset, _moved in placements:
            _check_nested(block, name, member_type)
    except HlslLayoutError as exc:
        raise HLSLPushConstantLayoutError(block, str(exc)) from exc
    explicit = layout_type.offsets or (None,) * len(placements)
    offsets = {
        name: offset
        for (name, _type, offset, moved), declared in zip(placements, explicit)
        if moved or declared is not None
    }
    if offsets:
        user = _laid_out_elsewhere(ast, struct_name, push_constants)
        if user is not None:
            raise HLSLPushConstantLayoutError(
                block,
                f"struct '{struct_name}' needs member offsets for DXC's relaxed rule and is also laid out by {user}",
            )
    return offsets


def member_offsets(codegen, ast, structs_by_name):
    """``{(struct name, member name): offset}`` for every member that needs an ``@offset`` in CrossGL.

    That is every ``[[vk::offset(N)]]`` member, and every member of a push-constant struct that DXC's relaxed rule
    places where std430 does not.
    """
    planner = _Planner(codegen, structs_by_name)
    offsets = {}
    for struct_name, struct in structs_by_name.items():
        for member in getattr(struct, "members", None) or []:
            offset = vk_offset(member)
            if offset is not None:
                offsets[(struct_name, str(member.name))] = offset
    push_constants = {}
    for node in getattr(ast, "global_variables", None) or []:
        struct_name = _push_constant_struct(node, structs_by_name)
        if struct_name is not None:
            push_constants[id(node)] = (struct_name, str(node.name))
    for struct_name, block in sorted(set(push_constants.values())):
        found = _struct_offsets(planner, ast, struct_name, block, push_constants)
        for member_name, offset in found.items():
            offsets[(struct_name, member_name)] = offset
    return offsets
