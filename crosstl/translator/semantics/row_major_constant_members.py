"""Row-major members of constant blocks that Metal declares in its own layout.

A ``@ row_major`` member of a constant block (``cbuffer`` or ``uniform`` block) stores its matrix row by row. A block
marked ``@ buffer_layout(std140)``, ``std430``, ``scalar`` or ``hlsl_cbuffer`` gets Metal storage that reproduces that
layout, and the Metal generator reads its row-major matrices through ``transpose``. A block without a layout marker
keeps the target's own layout (ADR 0005), and the Metal generator declared its row-major members as column-major
``floatCxR`` matrices: every read used the matrix's transpose, and a non-square matrix took the size of the wrong
shape, which moved the members after it.

MSL has no row-major matrices. A row-major ``matCxR`` is laid out as ``R`` row vectors of ``C`` components, which is
the layout of its column-major transpose ``matRxC``, so for Metal :func:`store_row_major_constant_members_transposed`
declares such a member of an unmarked block as its transpose and rewrites every read with the rules of
:mod:`.row_major_members`: ``m`` becomes ``transpose(m)``, a column ``m[i]`` ``transpose(m)[i]``, and an element
``m[i][j]`` ``m[j][i]`` when neither index has side effects. An array of matrices used without selecting one matrix
raises :class:`.row_major_members.RowMajorMemberError`. Constant block member names are unique in a program (the
parser rejects a name two blocks declare), so a name that no declaration shadows reads that member.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.buffer_member_layout import member_is_row_major
from ..codegen.glsl_block_layout import declared_buffer_layout
from .hub_types import iter_functions
from .row_major_members import (
    _matrix_dims,
    _Member,
    _Rewriter,
    _transposed,
)
from .scoped_walk import ScopedWalker

__all__ = ["TARGETS", "store_row_major_constant_members_transposed"]

TARGETS = frozenset({"metal"})

_LAYOUT_ATTRIBUTES = frozenset({"row_major", "column_major"})


def _member_type(member):
    member_type = getattr(member, "member_type", None)
    return member_type if member_type is not None else getattr(member, "var_type", None)


def _set_member_type(member, member_type):
    if getattr(member, "member_type", None) is not None:
        member.member_type = member_type
    else:
        member.var_type = member_type


def _unmarked_blocks(program):
    return [
        cbuffer
        for cbuffer in getattr(program, "cbuffers", None) or []
        if declared_buffer_layout(getattr(cbuffer, "attributes", None)) is None
    ]


def _collect_members(program):
    """Declare each row-major matrix member of an unmarked block transposed; returns ``{name: (type, _Member)}``."""
    found = {}
    for cbuffer in _unmarked_blocks(program):
        for member in getattr(cbuffer, "members", None) or []:
            if not member_is_row_major(member):
                continue
            matrix, dims = _matrix_dims(_member_type(member))
            if matrix is None:
                continue
            transposed = _transposed(_member_type(member))
            _set_member_type(member, transposed)
            member.attributes = [
                attribute
                for attribute in getattr(member, "attributes", None) or []
                if str(getattr(attribute, "name", "")).lower() not in _LAYOUT_ATTRIBUTES
            ]
            found[member.name] = (transposed, _Member(cbuffer.name, member, dims))
    return found


class _Sites(ScopedWalker):
    """Finds the names that read a transposed block member, skipping the declarations that shadow it."""

    def __init__(self, program, members):
        super().__init__(program)
        self.members = members
        self.sites = {}

    def visit(self, node):
        if not isinstance(node, A.IdentifierNode):
            return
        entry = self.members.get(node.name)
        if entry is not None and self.scope.lookup(node.name) is entry[0]:
            self.sites[id(node)] = entry[1]


def store_row_major_constant_members_transposed(program, target) -> int:
    """Store the row-major members of unmarked constant blocks transposed for ``target``; returns how many changed."""
    target = str(target or "").lower()
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    members = _collect_members(program)
    if not members:
        return 0
    walker = _Sites(program, members)
    walker.run_program()
    functions = {}
    for function in iter_functions(program):
        functions.setdefault(function.name, function)
    rewriter = _Rewriter(walker.sites, {}, [], functions, {}, target)
    for function in iter_functions(program):
        if getattr(function, "body", None) is not None:
            rewriter.children(function.body)
    return len(members)
