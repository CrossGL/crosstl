"""Packed vector members in storage whose layout rules cannot place them.

A ``@ packed`` struct member is an MSL packed vector (``packed_float3``): aligned to its scalar and taking exactly its
components (MSL 3.2 section 2.2.3), where a ``float3`` takes 16 bytes. Only some buffer layouts place such a member:

* MSL's own layout (``@ buffer_layout(metal)``) and the scalar block layout of GL_EXT_scalar_block_layout, which the
  block planners and the structured-buffer lowering reproduce on every target;
* the HLSL structured-buffer rules, which align every vector to its scalar.

std140, std430 and HLSL constant packing align a three- or four-component vector to 16 bytes, so a generator that
declares the struct in those rules moves the member and everything after it. For OpenGL, SPIR-V and HLSL,
:func:`refuse_unplaced_packed_members` raises :class:`PackedMemberLayoutError` for a constant buffer, uniform,
structured buffer or stage-entry constant reference whose element type holds packed members that move it, that
is, whose MSL layout differs from the one it has with plain vectors, unless its declared layout places them. The
check is conservative: it does not ask whether the target's own rules happen to put every member where MSL would.
A helper's reference parameter names a value its caller holds and is not checked.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.buffer_member_layout import member_is_packed, member_is_row_major
from ..codegen.glsl_block_layout import (
    METAL_LAYOUT,
    SCALAR_BLOCK_LAYOUT,
    crossgl_struct_layout_type,
    declared_buffer_layout,
)
from ..codegen.glsl_hlsl_cbuffer import layout_scalar_offsets
from ..codegen.hlsl_buffer_layout import HLSL_STRUCTURED_LAYOUT, HlslLayoutError
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .constant_reference_layouts import TARGETS
from .hub_types import iter_functions, iter_structs

__all__ = ["PackedMemberLayoutError", "refuse_unplaced_packed_members"]

# Layouts that place a packed vector member at its MSL offset.
PLACING_LAYOUTS = frozenset({METAL_LAYOUT, SCALAR_BLOCK_LAYOUT, HLSL_STRUCTURED_LAYOUT})
_STORAGE_QUALIFIERS = frozenset({"uniform", "buffer", "push_constant", "constant"})


class PackedMemberLayoutError(TranslationDiagnostic):
    """Storage holding packed vector members in a layout that moves them."""

    code = f"{CODE_PREFIX}packed-member-layout-unsupported"


class _Structs:
    """Which structs hold packed members that move them under MSL's rules."""

    def __init__(self, program):
        self.structs = {}
        for struct in iter_structs(program):
            name = getattr(struct, "name", None)
            if isinstance(name, str) and not getattr(struct, "is_cbuffer", False):
                self.structs.setdefault(name, struct)
        self.cache = {}

    def members(self, name, packed=True):
        struct = self.structs.get(name)
        if struct is None:
            return None
        return [
            (
                member.name,
                member.member_type,
                member_is_row_major(member),
                None,
                None,
                packed and member_is_packed(member),
            )
            for member in getattr(struct, "members", None) or []
            if isinstance(member, A.StructMemberNode)
        ]

    def holds_packed(self, name, stack=()):
        struct = self.structs.get(name)
        if struct is None or name in stack:
            return False
        for member in getattr(struct, "members", None) or []:
            if not isinstance(member, A.StructMemberNode):
                continue
            if member_is_packed(member):
                return True
            inner = _struct_name(member.member_type)
            if inner is not None and self.holds_packed(inner, (*stack, name)):
                return True
        return False

    def packed_moves(self, name):
        """Whether the packed members of struct ``name`` (or of the structs it holds) change its MSL layout."""
        if name not in self.cache:
            self.cache[name] = self.holds_packed(name) and self._offsets_differ(name)
        return self.cache[name]

    def _offsets_differ(self, name):
        try:
            packed = crossgl_struct_layout_type(
                name, self.members(name), lambda inner: self.members(inner), (name,)
            )
            plain = crossgl_struct_layout_type(
                name,
                self.members(name, packed=False),
                lambda inner: self.members(inner, packed=False),
                (name,),
            )
            return layout_scalar_offsets(packed, METAL_LAYOUT) != layout_scalar_offsets(
                plain, METAL_LAYOUT
            )
        except HlslLayoutError:
            # A layout the planner cannot compute is not known to keep the packed members in place.
            return True


def _struct_name(type_node):
    while isinstance(type_node, (A.ArrayType, A.ReferenceType)):
        type_node = getattr(type_node, "element_type", None) or getattr(
            type_node, "referenced_type", None
        )
    if isinstance(type_node, A.NamedType) and not type_node.generic_args:
        return type_node.name
    return None


def _stored_types(type_node):
    """The value types a declaration of ``type_node`` stores in a buffer: a buffer's element, or the value."""
    if isinstance(type_node, A.ReferenceType):
        return [type_node.referenced_type]
    if isinstance(type_node, A.NamedType) and type_node.generic_args:
        return list(type_node.generic_args)
    return [type_node]


def _check(structs, declaration, type_nodes, layout, target):
    if layout in PLACING_LAYOUTS:
        return
    for type_node in type_nodes:
        name = _struct_name(type_node)
        if name is not None and structs.packed_moves(name):
            raise PackedMemberLayoutError(
                f"'{declaration}' stores struct '{name}', whose packed vector members the "
                f"{layout or 'default'} layout of the target moves; declare it with @buffer_layout(metal)",
                target=target,
            )


def _is_storage_parameter(parameter, stage_entry):
    """Whether a parameter binds storage: a buffer, or a ``constant``/``device`` reference of a stage entry (a
    helper's reference names a value its caller already holds)."""
    param_type = getattr(parameter, "param_type", None)
    if isinstance(param_type, A.ReferenceType):
        return stage_entry and str(
            getattr(param_type, "address_space", "") or ""
        ).lower() in ("constant", "device")
    return isinstance(param_type, A.NamedType) and bool(param_type.generic_args)


def _stage_entry_ids(program):
    stages = getattr(program, "stages", None) or {}
    values = stages.values() if isinstance(stages, dict) else stages
    return {
        id(stage.entry_point)
        for stage in values
        if getattr(stage, "entry_point", None) is not None
    }


def _is_stage_entry(function, entries):
    return id(function) in entries or any(
        str(getattr(attribute, "name", "")).lower() == "stage_entry"
        for attribute in getattr(function, "attributes", None) or []
    )


def refuse_unplaced_packed_members(program, target) -> None:
    """Raise :class:`PackedMemberLayoutError` for storage whose layout moves its packed vector members."""
    target = str(target or "")
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return
    structs = _Structs(program)
    if not any(structs.holds_packed(name) for name in structs.structs):
        return
    for block in iter_structs(program):
        if not getattr(block, "is_cbuffer", False):
            continue
        layout = declared_buffer_layout(getattr(block, "attributes", None))
        _check(
            structs,
            block.name,
            [getattr(member, "member_type", None) for member in block.members or []],
            layout,
            target,
        )
    for variable in getattr(program, "global_variables", None) or []:
        qualifiers = {
            str(q).lower() for q in getattr(variable, "qualifiers", None) or []
        }
        var_type = getattr(variable, "var_type", None)
        if not qualifiers & _STORAGE_QUALIFIERS and not (
            isinstance(var_type, A.NamedType) and var_type.generic_args
        ):
            continue
        layout = declared_buffer_layout(getattr(variable, "attributes", None))
        _check(structs, variable.name, _stored_types(var_type), layout, target)
    entries = _stage_entry_ids(program)
    for function in iter_functions(program):
        stage_entry = _is_stage_entry(function, entries)
        for parameter in getattr(function, "parameters", None) or []:
            if not _is_storage_parameter(parameter, stage_entry):
                continue
            layout = declared_buffer_layout(getattr(parameter, "attributes", None))
            _check(
                structs,
                f"{function.name}.{parameter.name}",
                _stored_types(parameter.param_type),
                layout,
                target,
            )
