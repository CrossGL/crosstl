"""Row-major block members of structure type, as row-major copies of the structure.

GLSL lets a uniform or buffer block, or one of its members, declare ``row_major`` (GLSL 4.60 section 4.4.5); the
order reaches every matrix the member holds, including the matrices of a structure member and of the structures it
nests. The GLSL import keeps the block's order on such a member as ``Light light @row_major;``. The targets spell a
matrix order on a matrix only:

* HLSL and Slang accept ``row_major`` on matrix declarations, and DXC rejects it elsewhere ("'row_major' can only be
  used with a matrix type"), so the HLSL output of a uniform block did not compile;
* MSL has no matrix order and reads a matrix as it is declared, so the Metal output read each nested matrix
  transposed;
* SPIR-V decorates ``RowMajor`` per structure member, and the SPIR-V output refused a uniform-block member as holding
  no matrix.

Buffer blocks, which HLSL and Slang read through byte-address buffers, read every nested matrix column-major on all
four targets.

All of them already lay out a structure member declared ``@row_major``: HLSL and Slang ``row_major`` in constant
blocks, a transposed declaration read through ``transpose`` in buffer blocks (:mod:`.row_major_members`), MSL a
transposed declaration, SPIR-V ``RowMajor``. :func:`orient_row_major_struct_members` (constant blocks) and
:func:`orient_row_major_buffer_struct_members` (buffer blocks, before :mod:`.row_major_members` runs) therefore give
each such block member a copy of its structure, ``<Name>_row_major``, whose matrices without an order of their own
are ``@row_major`` and whose nested structures holding matrices are copied the same way, and drop the member's own
``@row_major``. A member is read through its fields (``light.m``, ``lights[i].m``, ``block.light.m``). A read of the
whole member would give the copy where the program expects the original structure, and is refused with
:class:`RowMajorStructMemberUnsupportedError`. GLSL output keeps ``layout(row_major)`` on the member, which GLSL
defines.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .stage_varyings import slots
from .stage_varyings import stages as _stages

__all__ = [
    "RowMajorStructMemberUnsupportedError",
    "orient_row_major_buffer_struct_members",
    "orient_row_major_struct_members",
]

TARGETS = frozenset(
    {
        "directx",
        "hlsl",
        "dx11",
        "dx12",
        "d3d11",
        "d3d12",
        "slang",
        "metal",
        "vulkan",
        "spirv",
    }
)
_ORIENTATIONS = ("row_major", "column_major")


class RowMajorStructMemberUnsupportedError(TranslationDiagnostic):
    """A row-major block member of structure type that a row-major copy of the structure cannot carry."""

    def __init__(self, target, detail):
        self.reason_code = "whole-member-read"
        self.missing_capabilities = (f"{target}.row-major-struct-member",)
        super().__init__(
            f"{detail}; on {target} a row-major member of structure type takes a row-major copy of the "
            "structure, which only reads through its fields can use",
            target=target,
            code=f"{CODE_PREFIX}row-major-struct-member-unsupported",
        )


def _names(node):
    names = {
        str(getattr(attribute, "name", "")).lower()
        for attribute in getattr(node, "attributes", None) or []
    }
    names.update(str(q).lower() for q in getattr(node, "qualifiers", None) or [])
    return names


def _orientation(node):
    names = _names(node)
    for orientation in _ORIENTATIONS:
        if orientation in names:
            return orientation
    return None


def _innermost(type_node):
    while isinstance(type_node, A.ArrayType):
        type_node = type_node.element_type
    return type_node


def _struct_name(type_node, structs):
    inner = _innermost(type_node)
    name = getattr(inner, "name", None)
    if isinstance(inner, A.NamedType) and name in structs:
        return name
    return None


def _retyped(type_node, name):
    if isinstance(type_node, A.ArrayType):
        return A.ArrayType(_retyped(type_node.element_type, name), type_node.size)
    return A.NamedType(name, list(getattr(type_node, "generic_args", None) or []))


class _Orientation:
    def __init__(self, program, target):
        self.program = program
        self.target = target
        self.structs = {
            struct.name: struct
            for struct in getattr(program, "structs", None) or []
            if isinstance(struct, A.StructNode) and getattr(struct, "name", None)
        }
        self.copies = {}
        self.instances = {}
        self.taken = set(self.structs)
        self.holding = {}

    def holds_matrix(self, type_node, seen=()):
        inner = _innermost(type_node)
        if isinstance(inner, A.MatrixType):
            return True
        name = _struct_name(inner, self.structs)
        if name is None or name in seen:
            return False
        if name not in self.holding:
            self.holding[name] = any(
                self.holds_matrix(member.member_type, seen + (name,))
                for member in self.structs[name].members or []
                if getattr(member, "member_type", None) is not None
            )
        return self.holding[name]

    def row_major_copy(self, name):
        """The name of the row-major copy of struct ``name``, declared after it on first use."""
        if name in self.copies:
            return self.copies[name]
        copy_name = f"{name}_row_major"
        suffix = 1
        while copy_name in self.taken:
            copy_name = f"{name}_row_major_{suffix}"
            suffix += 1
        self.taken.add(copy_name)
        self.copies[name] = copy_name
        original = self.structs[name]
        members = []
        for member in original.members or []:
            member_type = getattr(member, "member_type", None)
            attributes = list(getattr(member, "attributes", None) or [])
            if member_type is not None and _orientation(member) is None:
                if isinstance(_innermost(member_type), A.MatrixType):
                    attributes.append(A.AttributeNode("row_major"))
                elif self.holds_matrix(member_type):
                    member_type = _retyped(
                        member_type,
                        self.row_major_copy(_struct_name(member_type, self.structs)),
                    )
            copied = A.StructMemberNode(
                member.name,
                member_type,
                default_value=getattr(member, "default_value", None),
                attributes=attributes,
                resource_qualifiers=list(
                    getattr(member, "resource_qualifiers", None) or []
                ),
            )
            copied.source_location = getattr(member, "source_location", None)
            members.append(copied)
        copy = A.StructNode(
            copy_name,
            members,
            attributes=list(getattr(original, "attributes", None) or []),
        )
        copy.source_location = getattr(original, "source_location", None)
        structs = self.program.structs
        structs.insert(structs.index(original) + 1, copy)
        self.structs[copy_name] = copy
        return copy_name

    def refuse_whole_reads(self, member_name, block_name, instances=()):
        functions = list(getattr(self.program, "functions", None) or [])
        for stage in _stages(self.program):
            functions.extend(getattr(stage, "local_functions", None) or [])
            entry = getattr(stage, "entry_point", None)
            if entry is not None:
                functions.append(entry)
        for function in functions:
            parents = {}
            for parent, _field, _key, node in slots(getattr(function, "body", None)):
                parents.setdefault(id(node), parent)
            for parent, _field, _key, node in slots(getattr(function, "body", None)):
                # A constant block's member is named directly; a buffer block's through its instance.
                direct = isinstance(node, A.IdentifierNode) and node.name == member_name
                through_instance = (
                    isinstance(node, A.MemberAccessNode)
                    and node.member == member_name
                    and isinstance(node.object, A.IdentifierNode)
                    and node.object.name in instances
                )
                if not (direct or through_instance):
                    continue
                if isinstance(parent, A.MemberAccessNode) and parent.object is node:
                    continue
                if (
                    isinstance(parent, A.ArrayAccessNode)
                    and parent.array is node
                    and isinstance(parents.get(id(parent)), A.MemberAccessNode)
                    and parents[id(parent)].object is parent
                ):
                    continue
                raise RowMajorStructMemberUnsupportedError(
                    self.target,
                    f"'{getattr(function, 'name', '')}' reads the whole row-major member '{member_name}' of "
                    f"'{block_name}'",
                )

    def buffer_blocks(self):
        """The structs that GLSL buffer blocks (``@ glsl_buffer_block``) are declared with, and their names."""
        declarations = list(getattr(self.program, "global_variables", None) or [])
        for stage in _stages(self.program):
            declarations.extend(getattr(stage, "local_variables", None) or [])
        blocks = []
        for declaration in declarations:
            if "glsl_buffer_block" not in _names(declaration):
                continue
            name = _struct_name(getattr(declaration, "var_type", None), self.structs)
            if name is None:
                continue
            if self.structs[name] not in blocks:
                blocks.append(self.structs[name])
            self.instances.setdefault(name, set()).add(declaration.name)
        return blocks

    def run(self, buffers=False):
        if buffers:
            blocks = self.buffer_blocks()
        else:
            blocks = list(getattr(self.program, "cbuffers", None) or [])
            for stage in _stages(self.program):
                blocks.extend(getattr(stage, "local_cbuffers", None) or [])
        changed = 0
        for block in blocks:
            for member in getattr(block, "members", None) or []:
                member_type = getattr(member, "member_type", None)
                if member_type is None or _orientation(member) != "row_major":
                    continue
                name = _struct_name(member_type, self.structs)
                if name is None or not self.holds_matrix(member_type):
                    continue
                self.refuse_whole_reads(
                    member.name,
                    getattr(block, "name", ""),
                    self.instances.get(getattr(block, "name", None), ()),
                )
                member.member_type = _retyped(member_type, self.row_major_copy(name))
                member.attributes = [
                    attribute
                    for attribute in member.attributes or []
                    if str(getattr(attribute, "name", "")).lower() != "row_major"
                ]
                if getattr(member, "qualifiers", None):
                    member.qualifiers = [
                        q for q in member.qualifiers if str(q).lower() != "row_major"
                    ]
                changed += 1
        return changed


def orient_row_major_struct_members(program, target) -> int:
    """Give row-major constant-block members of structure type row-major structure copies; returns how many."""
    target = str(target or "").lower()
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    return _Orientation(program, target).run()


def orient_row_major_buffer_struct_members(program, target) -> int:
    """The same for GLSL buffer blocks, before :mod:`.row_major_members` stores the copies' matrices transposed.

    HLSL, Slang and Metal read a buffer block's bytes through their own lowerings and SPIR-V transposes a row-major
    struct member, so a copy's ``@ row_major`` matrices reach each of them through that pass.
    """
    target = str(target or "").lower()
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    return _Orientation(program, target).run(buffers=True)
