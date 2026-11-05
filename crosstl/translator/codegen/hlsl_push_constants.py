"""Push-constant blocks in HLSL output: ``[[vk::push_constant]]`` for SPIR-V, a constant buffer for Direct3D.

A CrossGL block marked ``@push_constant`` (:mod:`.push_constant_blocks`) holds constants the host pushes through the
command buffer (``vkCmdPushConstants``), laid out by the std430 rules unless ``@buffer_layout`` names std140 or
scalar (:mod:`..semantics.push_constant_layouts` marks the default). The HLSL generator declared it as an ordinary
``cbuffer`` at a ``register(bN)``, which DXC compiles for Vulkan to a uniform buffer at a descriptor binding the host
never binds for push constants: the program read its constants from an unbound descriptor.

DXC declares push constants only on a ``ConstantBuffer<T>`` or a struct variable marked ``[[vk::push_constant]]``,
not on a ``cbuffer``, and lays them out by its relaxed std430 rules, which align a vector to its scalar when it does
not cross a 16-byte boundary (``float s; float3 v;`` puts ``v`` at byte 4, std430 at byte 16). So the block is
declared twice. When DXC compiles for SPIR-V (it defines ``__spirv__``):

* a struct holds the block's members, each with the ``[[vk::offset(N)]]`` of its std430 (or declared) offset, and a
  struct a member holds is declared again as ``__crossgl_push_T`` with its members' offsets, so nested members keep
  their offsets as well;
* ``[[vk::push_constant]] ConstantBuffer<...> Name`` declares the block in the ``PushConstant`` storage class;
* every member is a ``static`` global of its own name and type, which a loader that each stage entry calls first
  assigns from the block (as for the blocks of :mod:`.hlsl_glsl_block_registers`), so member reads are unchanged.

Otherwise the constant buffer the generator has always written, with its members at the same offsets, stands for the
block, and its loader, when it has none, is empty.

DXC lays out the inside of an array or a matrix by the std430 rules, which ``[[vk::offset]]`` cannot change, and
cannot place a vector across a 16-byte boundary. A block with such a member (a std140 array of scalars or a std140
``mat2``, a scalar-layout ``vec3[]`` or ``mat3``) is declared as the 32-bit words it spans instead, copied into the
``uint4`` registers its Direct3D register loader reads (:mod:`.hlsl_glsl_block_registers`). A second push-constant
block and a layout other than std430, std140 or scalar are refused with
:class:`.push_constant_blocks.PushConstantBlockError`.
"""

from __future__ import annotations

import re

from ..ast import FunctionNode
from .glsl_block_layout import (
    block_alignment,
    block_array_stride,
    block_matrix_vector_stride,
    block_size,
    block_struct_placements,
    crossgl_struct_layout_type,
)
from .hlsl_buffer_layout import HlslLayoutError, align_to
from .push_constant_blocks import (
    PushConstantBlockError,
    _claim,
    is_push_constant,
    push_constant_block_layout,
)

__all__ = ["SPIRV_GUARD", "hlsl_push_constant_block", "is_push_constant"]

TARGET = "directx"
# DXC defines __spirv__ when it compiles for SPIR-V (-spirv).
SPIRV_GUARD = "#ifdef __spirv__\n"
DXC_LAYOUT = "std430"
VECTOR_BOUNDARY = 16
WORD_BYTES = 4
_PACKOFFSET = re.compile(r"\s*:\s*packoffset\s*\([^)]*\)\s*$")
_ORIENTATION = re.compile(r"^(?:row_major|column_major)\s+")


def _refuse(block, reason):
    return PushConstantBlockError(TARGET, block, reason)


class _Unplaceable(Exception):
    """A member DXC's push-constant rules cannot place where the block's layout does."""


def _members(node):
    return [
        member
        for member in getattr(node, "members", None) or []
        if getattr(member, "name", None) and not isinstance(member, FunctionNode)
    ]


def _struct_nodes(ast):
    structs = {}
    collections = [getattr(ast, "structs", None) or []]
    stages = getattr(ast, "stages", None) or {}
    for stage in stages.values() if isinstance(stages, dict) else stages:
        collections.append(getattr(stage, "local_structs", None) or [])
    for collection in collections:
        for struct in collection:
            name = getattr(struct, "name", None)
            if isinstance(name, str):
                structs.setdefault(name, struct)
    return structs


def _innermost(layout_type):
    while layout_type.kind == "array":
        layout_type = layout_type.element
    return layout_type


def _array_suffix(layout_type):
    suffix = ""
    while layout_type.kind == "array":
        suffix += f"[{layout_type.count}]"
        layout_type = layout_type.element
    return suffix


# Layout checks -------------------------------------------------------------------------------------------------------


def _dxc_size(layout_type, layout):
    """The bytes DXC gives a value whose struct members carry their ``layout`` offsets."""
    kind = layout_type.kind
    if kind == "struct":
        placements, _size = block_struct_placements(layout_type, layout)
        end = max(
            (
                placement.offset + _dxc_size(placement.layout_type, layout)
                for placement in placements
            ),
            default=0,
        )
        return align_to(end, block_alignment(layout_type, DXC_LAYOUT))
    if kind == "array":
        return _dxc_array_stride(layout_type, layout) * layout_type.count
    return block_size(layout_type, DXC_LAYOUT)


def _dxc_array_stride(layout_type, layout):
    return align_to(
        _dxc_size(layout_type.element, layout),
        block_alignment(layout_type.element, DXC_LAYOUT),
    )


def _check_offset(block, path, layout_type, offset):
    """Reject an offset DXC's relaxed std430 rules cannot give a member (VK_KHR_relaxed_block_layout)."""
    if layout_type.kind == "vector":
        size = block_size(layout_type, DXC_LAYOUT)
        straddles = (
            offset // VECTOR_BOUNDARY != (offset + size - 1) // VECTOR_BOUNDARY
            if size <= VECTOR_BOUNDARY
            else offset % VECTOR_BOUNDARY != 0
        )
        if offset % layout_type.scalar_size == 0 and not straddles:
            return
        reason = "a vector there crosses a 16-byte boundary"
    else:
        alignment = block_alignment(layout_type, DXC_LAYOUT)
        if offset % alignment == 0:
            return
        reason = f"it is not a multiple of the member's std430 alignment {alignment}"
    raise _Unplaceable(
        f"member '{path}' of '{block}' is at byte {offset}, which DXC cannot place in a push-constant block: "
        f"{reason}"
    )


def _check_interior(block, path, layout_type, layout):
    """Reject an array or matrix whose ``layout`` strides differ from the std430 strides DXC gives it."""
    kind = layout_type.kind
    if kind == "matrix":
        stride = block_matrix_vector_stride(layout_type, layout)
        dxc = block_matrix_vector_stride(layout_type, DXC_LAYOUT)
        if stride != dxc:
            raise _Unplaceable(
                f"matrix member '{path}' of '{block}' has a {stride}-byte vector stride under the {layout} rules, "
                f"and DXC lays push-constant matrices out with the std430 stride {dxc}"
            )
    elif kind == "array":
        stride = block_array_stride(layout_type, layout)
        dxc = _dxc_array_stride(layout_type, layout)
        if stride != dxc:
            raise _Unplaceable(
                f"array member '{path}' of '{block}' has a {stride}-byte stride under the {layout} rules, and "
                f"DXC lays push-constant arrays out with the std430 stride {dxc}"
            )
        _check_interior(block, f"{path}[]", layout_type.element, layout)
    elif kind == "struct":
        placements, _size = block_struct_placements(layout_type, layout)
        for placement in placements:
            member_path = f"{path}.{placement.name}"
            _check_offset(block, member_path, placement.layout_type, placement.offset)
            _check_interior(block, member_path, placement.layout_type, layout)


# Declarations --------------------------------------------------------------------------------------------------------


def _mirror_name(struct_name):
    return f"__crossgl_push_{struct_name}"


def _declaration(codegen, member):
    """A member's HLSL declaration without its ``packoffset``."""
    return _PACKOFFSET.sub("", codegen.generate_hlsl_cbuffer_member_declaration(member))


def _placed_declaration(codegen, member, layout_type, offset):
    inner = _innermost(layout_type)
    if inner.kind == "struct":
        declaration = f"{_mirror_name(inner.struct_name)} {member.name}{_array_suffix(layout_type)}"
    else:
        declaration = _declaration(codegen, member)
    return f"    [[vk::offset({offset})]] {declaration};\n"


def _mirror_structs(codegen, block, structs, layout_type, layout, emitted, code):
    """Declare ``__crossgl_push_T`` for every struct ``layout_type`` holds, nested ones first."""
    inner = _innermost(layout_type)
    if inner.kind != "struct":
        return code
    placements, _size = block_struct_placements(inner, layout)
    for placement in placements:
        code = _mirror_structs(
            codegen, block, structs, placement.layout_type, layout, emitted, code
        )
    if inner.struct_name in emitted:
        return code
    struct = structs.get(inner.struct_name)
    if struct is None:
        raise _refuse(block, f"struct '{inner.struct_name}' has no declaration")
    emitted.add(inner.struct_name)
    members = {member.name: member for member in _members(struct)}
    code += f"struct {_mirror_name(inner.struct_name)}\n{{\n"
    for placement in placements:
        code += _placed_declaration(
            codegen, members[placement.name], placement.layout_type, placement.offset
        )
    return code + "};\n"


def _copy(lines, layout_type, target, source, depth):
    """Assign ``target`` from ``source``, member by member where a mirrored struct stands for its type."""
    indent = "    " * depth
    if _innermost(layout_type).kind != "struct":
        lines.append(f"{indent}{target} = {source};")
        return
    if layout_type.kind == "struct":
        for name, member_type in layout_type.members:
            _copy(lines, member_type, f"{target}.{name}", f"{source}.{name}", depth)
        return
    index = f"__crossgl_push_i{depth - 1}"
    lines.append(
        f"{indent}for (int {index} = 0; {index} < {layout_type.count}; ++{index})"
    )
    lines.append(f"{indent}{{")
    _copy(
        lines,
        layout_type.element,
        f"{target}[{index}]",
        f"{source}[{index}]",
        depth + 1,
    )
    lines.append(f"{indent}}}")


def _layout_type(codegen, node, structs):
    def struct_members(name):
        struct = structs.get(name)
        if struct is None:
            return None
        return [codegen.hlsl_block_member(member) for member in _members(struct)]

    return crossgl_struct_layout_type(
        node.name,
        [codegen.hlsl_block_member(member) for member in _members(node)],
        struct_members,
        (node.name,),
        codegen.hlsl_constant_int_value,
    )


def _placed_block(codegen, node, structs, layout, placements):
    """The block as a struct whose members carry ``[[vk::offset(N)]]``, with ``static`` copies and their loader."""
    members = _members(node)
    emitted = set()
    code = ""
    for placement in placements:
        code = _mirror_structs(
            codegen, node.name, structs, placement.layout_type, layout, emitted, code
        )
    block_type = f"__crossgl_{node.name}_push_constants"
    code += f"struct {block_type}\n{{\n"
    for member, placement in zip(members, placements):
        code += _placed_declaration(
            codegen, member, placement.layout_type, placement.offset
        )
    code += "};\n"
    code += f"[[vk::push_constant]] ConstantBuffer<{block_type}> {node.name};\n"
    lines = []
    for member, placement in zip(members, placements):
        declaration = _ORIENTATION.sub("", _declaration(codegen, member))
        code += f"static {declaration};\n"
        _copy(
            lines, placement.layout_type, member.name, f"{node.name}.{member.name}", 1
        )
    body = "".join(f"{line}\n" for line in lines)
    return code + f"void __crossgl_load_{node.name}()\n{{\n{body}}}\n"


def _word_block(codegen, node, layout, layout_type):
    """The block as the 32-bit words it spans, copied into the registers the block's register loader reads.

    DXC lays the inside of a push-constant array or matrix out by the std430 rules; a block whose layout places them
    otherwise is read word by word, as Direct3D reads such a block (:mod:`.hlsl_glsl_block_registers`). The words
    start at the block's first member, so a block placed after another stage's range keeps within its own.
    """
    code = codegen.hlsl_glsl_block_register_code(node, layout, layout_type)
    first = code.first_offset // WORD_BYTES
    count = -(-code.size // WORD_BYTES) - first
    block_type = f"__crossgl_{node.name}_push_constants"
    index = "__crossgl_push_word"
    word = f"{index} + {first}" if first else index
    return (
        f"struct {block_type}\n{{\n"
        f"    [[vk::offset({first * WORD_BYTES})]] uint words[{count}];\n"
        "};\n"
        f"[[vk::push_constant]] ConstantBuffer<{block_type}> {node.name};\n"
        f"static uint4 {code.registers}[{code.register_count}];\n"
        f"{code.statics}"
        f"{code.word_reader}"
        f"void {code.loader_name}()\n{{\n"
        f"    for (int {index} = 0; {index} < {count}; ++{index})\n"
        "    {\n"
        f"        {code.registers}[({word}) >> 2][({word}) & 3] = {node.name}.words[{index}];\n"
        "    }\n"
        f"{code.loader_body}}}\n"
    )


def _spirv_block(codegen, ast, node, layout):
    """The SPIR-V declaration of a push-constant block, its ``static`` members and their loader."""
    structs = _struct_nodes(ast)
    try:
        layout_type = _layout_type(codegen, node, structs)
        placements, _size = block_struct_placements(layout_type, layout, root=True)
    except HlslLayoutError as exc:
        raise _refuse(node.name, str(exc)) from exc
    try:
        for placement in placements:
            _check_offset(
                node.name, placement.name, placement.layout_type, placement.offset
            )
            _check_interior(node.name, placement.name, placement.layout_type, layout)
    except _Unplaceable:
        return _word_block(codegen, node, layout, layout_type)
    return _placed_block(codegen, node, structs, layout, placements)


def hlsl_push_constant_block(codegen, ast, node, direct3d):
    """The declarations of a ``cbuffer`` marked ``@push_constant``: its SPIR-V push-constant block and, for
    Direct3D, ``direct3d``, the constant buffer the generator writes for it.

    Both declare the block's members as module-scope values read by name; a loader the stage entries call assigns
    them where they are copies.
    """
    layout = push_constant_block_layout(node, TARGET, node.name)
    _claim(codegen, TARGET, node.name)
    if not hasattr(codegen, "hlsl_glsl_block_loaders"):
        codegen.reset_hlsl_glsl_block_registers()
    loader_name = f"__crossgl_load_{node.name}"
    spirv = _spirv_block(codegen, ast, node, layout)
    if loader_name not in codegen.hlsl_glsl_block_loaders:
        # Direct3D reads the members from the constant buffer itself.
        direct3d += f"void {loader_name}() {{}}\n"
        codegen.hlsl_glsl_block_loaders.append(loader_name)
        codegen.hlsl_glsl_block_patch_functions.update(
            codegen.hlsl_patch_constant_function_names(ast)
        )
    return f"{SPIRV_GUARD}{spirv}#else\n{direct3d}#endif\n"
