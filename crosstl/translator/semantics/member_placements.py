"""Constant-buffer members placed by ``@offset`` or ``@align`` on targets that would drop the placement.

A member of a constant block, or of the struct a ``ConstantBuffer<T>`` holds, may fix its own byte offset with
``@ offset(N)`` and its alignment with ``@ align(N)`` (:mod:`..codegen.buffer_member_layout`). SPIR-V and HLSL output
reproduce those placements in push-constant blocks, and every generator does in a block whose declared layout it
plans. Two generators wrote other declarations silently:

* Metal declares a constant block, or the ``T`` of a ``ConstantBuffer<T>``, as a plain MSL struct in its natural
  layout (MSL 3.2 section 2.2) unless the block declares a layout it plans (``hlsl_cbuffer``, ``std140``, ``std430``
  or ``scalar``) or is a push-constant block, which it plans with the push-constant layout.
  ``float scale; vec3 v @offset(32);`` then read ``v`` at byte 16.
* Slang declares the ``T`` of a push-constant ``ConstantBuffer<T>`` with no offsets, so slangc lays it out by std430;
  and it writes the ``[[vk::offset(N)]]`` of a push-constant ``cbuffer`` member, which slangc moves to the member's
  std430 alignment when ``N`` is not a multiple of it.

HLSL and GLSL output declare a struct once, by their own rules, so they read a constant block whose structs carry
placements that move a scalar from its 16-byte registers (:mod:`..codegen.glsl_hlsl_cbuffer`,
:mod:`..codegen.hlsl_glsl_block_registers`); they place a push-constant block's own members only, not those of a
struct it holds. Every target but SPIR-V declared the ``T`` of an array of ``ConstantBuffer<T>`` without its
placements. For those targets
:func:`refuse_unplaced_member_offsets` raises :class:`MemberPlacementError` when the declared placements put any
scalar somewhere other than the target's own rules would. Placements that agree with them, such as the offsets an
importer writes for every member, are accepted.

SPIR-V output decorates every struct type's members with their ``Offset`` (SPIR-V 1.6 section 2.16.2), so it places
the members of a block, of the structs it holds at any depth and of the ``T`` of a ``ConstantBuffer<T>`` or an array
of them at the bytes they declare (:mod:`..codegen.spirv_placed_blocks`), and refuses a placement its layout rules
reject with its own typed diagnostic.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.buffer_member_layout import (
    member_explicit_offset,
    member_has_explicit_placement,
    member_is_row_major,
)
from ..codegen.glsl_block_layout import (
    GLSL_BLOCK_LAYOUTS,
    METAL_LAYOUT,
    STD140_LAYOUT,
    STD430_LAYOUT,
    block_alignment,
    crossgl_struct_layout_type,
    declared_buffer_layout,
    member_explicit_alignment,
)
from ..codegen.glsl_hlsl_cbuffer import layout_scalar_offsets
from ..codegen.hlsl_buffer_layout import HLSL_CBUFFER_LAYOUT, HlslLayoutError
from ..codegen.push_constant_blocks import is_push_constant
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .constant_buffer_layouts import TARGET_LAYOUTS
from .constant_buffers import constant_buffer_element
from .hub_types import iter_structs

__all__ = [
    "ConstantBufferArrayLayoutError",
    "MemberPlacementError",
    "refuse_unplaced_member_offsets",
]

METAL_TARGETS = frozenset({"metal"})
SLANG_TARGETS = frozenset({"slang"})
SPIRV_TARGETS = frozenset({"vulkan", "spirv"})
# Declared layouts the Metal generator plans, placing explicit offsets with padding and packed storage.
METAL_PLANNED_LAYOUTS = frozenset({HLSL_CBUFFER_LAYOUT, *GLSL_BLOCK_LAYOUTS})


class MemberPlacementError(TranslationDiagnostic):
    """A constant-buffer member placement the target's declaration would not reproduce."""

    code = f"{CODE_PREFIX}member-placement-unsupported"


class ConstantBufferArrayLayoutError(TranslationDiagnostic):
    """An array of ``ConstantBuffer<T>`` whose element bytes the target's declaration of ``T`` would move."""

    code = f"{CODE_PREFIX}constant-buffer-array-layout-unsupported"


class _Structs:
    def __init__(self, program):
        self.structs = {}
        for struct in iter_structs(program):
            name = getattr(struct, "name", None)
            if isinstance(name, str) and not getattr(struct, "is_cbuffer", False):
                self.structs.setdefault(name, struct)

    def tuples(self, members, placed):
        return [
            (
                member.name,
                member.member_type,
                member_is_row_major(member),
                member_explicit_offset(member) if placed else None,
                member_explicit_alignment(member) if placed else None,
            )
            for member in members
            if isinstance(member, A.StructMemberNode)
        ]

    def members(self, name, placed):
        struct = self.structs.get(name)
        if struct is None:
            return None
        return self.tuples(getattr(struct, "members", None) or [], placed)

    def holds_placements(self, members, stack=()):
        for member in members or []:
            if not isinstance(member, A.StructMemberNode):
                continue
            if member_has_explicit_placement(member):
                return True
            inner = _struct_name(member.member_type)
            if inner is not None and inner not in stack and inner in self.structs:
                if self.holds_placements(self.structs[inner].members, (*stack, inner)):
                    return True
        return False

    def layout_type(self, name, members, placed):
        return crossgl_struct_layout_type(
            name,
            self.tuples(members, placed),
            lambda inner: self.members(inner, placed),
            (name,),
        )

    def placements_move(self, name, members, layout):
        """Whether the explicit placements of ``members`` put a scalar where ``layout`` alone would not."""
        try:
            placed = layout_scalar_offsets(
                self.layout_type(name, members, True), layout
            )
            plain = layout_scalar_offsets(
                self.layout_type(name, members, False), layout
            )
        except HlslLayoutError:
            return True
        return placed != plain

    def nested_placements_move(self, name, members, layout):
        """Whether placements inside the structs ``members`` hold put a scalar where ``layout`` alone would not.

        The block's own member placements are kept on both sides; the targets that call this place those.
        """
        top = self.tuples(members, True)
        try:
            placed = layout_scalar_offsets(
                crossgl_struct_layout_type(
                    name, top, lambda inner: self.members(inner, True), (name,)
                ),
                layout,
            )
            plain = layout_scalar_offsets(
                crossgl_struct_layout_type(
                    name, top, lambda inner: self.members(inner, False), (name,)
                ),
                layout,
            )
        except HlslLayoutError:
            return True
        return placed != plain


def _check_nested_placements(
    structs, program, target, native, push_constants_only=False
):
    """Refuse a constant block whose members' structs carry placements that move a scalar under the block's layout.

    HLSL and GLSL output place a block's own members (``packoffset`` and ``layout(offset)``), but declare the
    structs it holds once, by the layout's rules alone: GLSL ``layout(offset)`` and HLSL ``packoffset`` apply to
    block members only (GLSL 4.60 section 4.4.5). Both read any other constant block whose structs move a scalar
    from its registers (:mod:`..codegen.glsl_hlsl_cbuffer`, :mod:`..codegen.hlsl_glsl_block_registers`), so with
    ``push_constants_only`` only a push-constant block is checked.
    """
    for block in iter_structs(program):
        if not getattr(block, "is_cbuffer", False):
            continue
        if push_constants_only and not is_push_constant(block):
            continue
        nested = [
            member
            for member in getattr(block, "members", None) or []
            if isinstance(member, A.StructMemberNode)
            and _struct_name(member.member_type) in structs.structs
            and structs.holds_placements(
                structs.structs[_struct_name(member.member_type)].members
            )
        ]
        if not nested:
            continue
        layout = declared_buffer_layout(getattr(block, "attributes", None)) or native
        if structs.nested_placements_move(block.name, block.members, layout):
            names = ", ".join(sorted({_struct_name(m.member_type) for m in nested}))
            _refuse(
                target,
                block.name,
                f"the members of struct {names} carry @offset or @align, and {target} output declares a struct "
                "inside a constant buffer by the buffer's layout rules alone",
            )


def _struct_name(type_node):
    while isinstance(type_node, A.ArrayType):
        type_node = type_node.element_type
    if isinstance(type_node, A.NamedType) and not type_node.generic_args:
        return type_node.name
    return None


def _refuse(target, name, reason):
    raise MemberPlacementError(f"constant buffer '{name}': {reason}", target=target)


def _variables(program):
    yield from getattr(program, "global_variables", None) or []
    stages = getattr(program, "stages", None) or {}
    for stage in stages.values() if isinstance(stages, dict) else stages:
        yield from getattr(stage, "local_variables", None) or []


def _element_struct(structs, variable):
    """The struct a ``ConstantBuffer<T>``, an array of them, or a push-constant ``T`` holds, or ``None``."""
    var_type = getattr(variable, "var_type", None)
    element = constant_buffer_element(var_type)
    if element is None and isinstance(var_type, A.ArrayType):
        inner = var_type
        while isinstance(inner, A.ArrayType):
            inner = inner.element_type
        element = constant_buffer_element(inner)
    if element is None and is_push_constant(variable):
        element = var_type
    name = _struct_name(element) if element is not None else None
    return structs.structs.get(name) if name is not None else None


def _check_constant_buffer_arrays(structs, program, target, native):
    """Refuse an array of ``ConstantBuffer<T>`` whose ``T`` the target would lay out at other bytes than declared.

    Each element of the array is a buffer of its own, so the block :mod:`.constant_buffer_layouts` declares for one
    buffer with a declared layout cannot hold it, and HLSL and GLSL output declare the array's ``T`` once, by the
    target's own constant-buffer rules (``native``) and without member placements. The host writes each element by
    the declared ``@ buffer_layout``, or by ``native`` when there is none, with the placements of ``T`` and of the
    structs it holds.
    """
    for variable in _variables(program):
        if not isinstance(getattr(variable, "var_type", None), A.ArrayType):
            continue
        struct = _element_struct(structs, variable)
        if struct is None:
            continue
        declared = declared_buffer_layout(getattr(variable, "attributes", None))
        layout = declared or native
        try:
            wanted = layout_scalar_offsets(
                structs.layout_type(struct.name, struct.members, True), layout
            )
            written = layout_scalar_offsets(
                structs.layout_type(struct.name, struct.members, False), native
            )
        except HlslLayoutError as exc:
            reason = f"its elements cannot be laid out by the {layout} rules: {exc}"
        else:
            if wanted == written:
                continue
            cause = (
                f"the {declared} layout it declares"
                if declared not in (None, native)
                else f"the @offset or @align of the members of struct '{struct.name}'"
            )
            reason = (
                f"{target} output declares the struct '{struct.name}' of every element by the {native} rules "
                f"without placements, which moves members that {cause} places elsewhere"
            )
        raise ConstantBufferArrayLayoutError(
            f"array of constant buffers '{variable.name}': {reason}", target=target
        )


def _check_blocks(structs, program, target, layouts, planned, push_constants):
    """Refuse a constant block, other than a push-constant block unless ``push_constants``, that declares no layout
    in ``planned`` and whose placements move a scalar under one of ``layouts``."""
    for block in iter_structs(program):
        if not getattr(block, "is_cbuffer", False):
            continue
        if is_push_constant(block) and not push_constants:
            continue
        layout = declared_buffer_layout(getattr(block, "attributes", None))
        if layout in planned or not structs.holds_placements(block.members):
            continue
        if any(
            structs.placements_move(block.name, block.members, native)
            for native in layouts
        ):
            _refuse(
                target,
                block.name,
                f"its members' @offset or @align place them where {target} output does not, and the block "
                "declares no layout the generator plans",
            )


def _check_elements(structs, program, target, native, skip):
    """Refuse a ``ConstantBuffer<T>`` whose ``T`` carries placements that move a scalar under the buffer's declared
    layout, or under one of the ``native`` layouts when it declares none. A buffer with a declared layout is
    declared as a block holding one ``T`` (:mod:`.constant_buffer_layouts`), and the block planners place the members
    of a nested struct by the layout's rules alone. ``skip(variable, declared)`` exempts the buffers the generator
    places itself."""
    for variable in _variables(program):
        struct = _element_struct(structs, variable)
        if struct is None or not structs.holds_placements(struct.members):
            continue
        declared = declared_buffer_layout(getattr(variable, "attributes", None))
        if skip(variable, declared):
            continue
        layouts = (declared,) if declared is not None else native
        if any(
            structs.placements_move(struct.name, struct.members, layout)
            for layout in layouts
        ):
            _refuse(
                target,
                variable.name,
                f"the members of struct '{struct.name}' carry @offset or @align, and {target} output declares "
                f"'{struct.name}' without them",
            )


def _check_slang_cbuffer(structs, block, target):
    """slangc keeps a push-constant member's [[vk::offset(N)]] only at a multiple of its std430 alignment."""
    for member in block.members or []:
        if not isinstance(member, A.StructMemberNode):
            continue
        inner = _struct_name(member.member_type)
        if inner in structs.structs and structs.holds_placements(
            structs.structs[inner].members
        ):
            _refuse(
                target,
                block.name,
                f"member '{member.name}' holds struct '{inner}', whose @offset or @align members Slang does not "
                "place",
            )
    try:
        layout_type = structs.layout_type(block.name, block.members, True)
    except HlslLayoutError as exc:
        _refuse(target, block.name, str(exc))
    for (name, member_type), offset in zip(
        layout_type.members, layout_type.offsets or ()
    ):
        if offset is None:
            continue
        alignment = block_alignment(member_type, STD430_LAYOUT)
        if offset % alignment:
            _refuse(
                target,
                block.name,
                f"member '{name}' is at byte {offset}, which is not a multiple of its std430 alignment "
                f"{alignment}; slangc moves such a [[vk::offset]] to that alignment",
            )


def _check_slang(structs, program, target):
    for block in iter_structs(program):
        if getattr(block, "is_cbuffer", False) and is_push_constant(block):
            _check_slang_cbuffer(structs, block, target)
    # slangc lays a push constant out by std430, and a constant buffer by std140 for SPIR-V and by HLSL packing for
    # Direct3D.
    native = (STD140_LAYOUT, HLSL_CBUFFER_LAYOUT)
    _check_blocks(structs, program, target, native, frozenset(), False)
    for variable in _variables(program):
        if not is_push_constant(variable):
            continue
        struct = _element_struct(structs, variable)
        if struct is None or not structs.holds_placements(struct.members):
            continue
        if structs.placements_move(struct.name, struct.members, STD430_LAYOUT):
            _refuse(
                target,
                variable.name,
                f"the members of struct '{struct.name}' carry @offset or @align, and Slang declares a "
                "push-constant ConstantBuffer<T> with std430 offsets only",
            )
    _check_elements(
        structs,
        program,
        target,
        native,
        lambda variable, _declared: is_push_constant(variable),
    )


def refuse_unplaced_member_offsets(program, target) -> None:
    """Raise :class:`MemberPlacementError` for member placements the target's declarations would drop."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode):
        return
    structs = _Structs(program)
    if target in METAL_TARGETS:
        native = (METAL_LAYOUT,)
        # Metal output plans push-constant blocks and buffers with their push-constant layout, and the structs a
        # planned block holds with their members' placements (metal_hlsl_resources).
        _check_blocks(structs, program, target, native, METAL_PLANNED_LAYOUTS, False)
        _check_elements(
            structs,
            program,
            target,
            native,
            lambda variable, declared: is_push_constant(variable)
            or declared in METAL_PLANNED_LAYOUTS,
        )
    elif target in SLANG_TARGETS:
        _check_slang(structs, program, target)
    elif target in SPIRV_TARGETS:
        # SPIR-V output places every member at its declared bytes (spirv_placed_blocks).
        return
    elif target in TARGET_LAYOUTS:
        # GLSL output places or refuses the members of a push-constant T itself (push_constant_blocks). HLSL output places them (hlsl_push_constants)
        # and reads a buffer that declares a layout from registers at the planned offsets
        # (hlsl_glsl_block_registers).
        hlsl = TARGET_LAYOUTS[target] == HLSL_CBUFFER_LAYOUT
        _check_constant_buffer_arrays(structs, program, target, TARGET_LAYOUTS[target])
        _check_nested_placements(
            structs, program, target, TARGET_LAYOUTS[target], push_constants_only=True
        )
        _check_elements(
            structs,
            program,
            target,
            (TARGET_LAYOUTS[target],),
            lambda variable, declared: is_push_constant(variable)
            or (hlsl and declared is not None),
        )
