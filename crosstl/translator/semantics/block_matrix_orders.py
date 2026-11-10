"""Block-level matrix orders: ``@ row_major`` on a constant buffer or on a block instance.

GLSL lets a uniform or storage block name the order of its matrices, ``layout(row_major) buffer B { ... } b;``. The
order applies to every matrix in the block, including the matrices in structures and arrays to all depths of
nesting, and a member that names its own order keeps it (GLSL 4.60 section 4.4.5). CrossGL writes that order on the
block: ``cbuffer U @ row_major { ... }`` for a constant buffer, and ``B b @ row_major @ glsl_buffer_block(std430);``
for a block instance, whose struct ``B`` holds the members. Every generator reads matrix orders from the members
(:func:`..codegen.buffer_member_layout.member_is_row_major`) and ignored the block's, so such a block was declared
column-major and every matrix in it read transposed.

For every target but CrossGL, :func:`distribute_block_matrix_orders` places ``@ row_major`` on each member that
names no order and whose type holds a matrix, directly, in an array or through a struct. That is the form the GLSL
importer writes. ``@ column_major`` is the order an unmarked member already has, so it changes nothing. A struct that
module-scope declarations use both with and without a block-level ``row_major`` cannot carry both orders and raises
:class:`BlockMatrixOrderError`.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.buffer_member_layout import member_attribute_names, member_is_row_major
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .scoped_walk import program_stages

__all__ = [
    "BlockMatrixOrderError",
    "KEEPING_TARGETS",
    "distribute_block_matrix_orders",
]

# CrossGL output writes the block-level order back as it was declared.
KEEPING_TARGETS = frozenset({"cgl", "crossgl", ""})

_ORDERS = frozenset({"row_major", "column_major"})


class BlockMatrixOrderError(TranslationDiagnostic):
    """A struct declared by blocks with different matrix orders."""

    code = f"{CODE_PREFIX}block-matrix-order-conflict"


def _element_type(node):
    while isinstance(node, A.ArrayType):
        node = node.element_type
    if isinstance(node, A.ReferenceType):
        return _element_type(node.referenced_type)
    return node


def _type_name(node):
    node = _element_type(node)
    if isinstance(node, (A.NamedType, A.PrimitiveType)):
        return str(node.name)
    return None


def _holds_matrix(type_node, structs, visiting=frozenset()):
    """Whether a value of ``type_node`` contains a matrix, directly, in an array or in a struct member."""
    node = _element_type(type_node)
    if isinstance(node, A.MatrixType):
        return True
    name = _type_name(node)
    struct = structs.get(name) if name else None
    if struct is None or name in visiting:
        return False
    visiting = visiting | {name}
    return any(
        _holds_matrix(getattr(member, "member_type", None), structs, visiting)
        for member in getattr(struct, "members", None) or []
        if isinstance(member, A.StructMemberNode)
    )


def _place_row_major(members, structs) -> int:
    """Mark ``row_major`` on each member that names no order and holds a matrix; returns how many changed."""
    changed = 0
    for member in members:
        if not isinstance(member, A.StructMemberNode):
            continue
        if member_attribute_names(member) & _ORDERS:
            continue
        if not _holds_matrix(member.member_type, structs):
            continue
        member.attributes = [*(member.attributes or []), A.AttributeNode("row_major")]
        changed += 1
    return changed


def _structs(program):
    structs = {}
    declared = list(getattr(program, "structs", None) or [])
    for stage in program_stages(program):
        declared.extend(getattr(stage, "local_structs", None) or [])
    for struct in declared:
        if isinstance(struct, A.StructNode) and not getattr(
            struct, "is_cbuffer", False
        ):
            structs.setdefault(str(struct.name), struct)
    return structs


def _declarations(program, field):
    declared = list(getattr(program, field, None) or [])
    stage_field = {"cbuffers": "local_cbuffers", "global_variables": "local_variables"}
    for stage in program_stages(program):
        declared.extend(getattr(stage, stage_field[field], None) or [])
    return declared


def _row_major_block_structs(program, structs, target):
    """Names of the structs that row-major block instances declare, after checking no other declaration disagrees."""
    orders = {}
    for declaration in _declarations(program, "global_variables"):
        if not isinstance(declaration, A.VariableNode):
            continue
        name = _type_name(getattr(declaration, "var_type", None))
        if name not in structs:
            continue
        orders.setdefault(name, []).append(
            (declaration, member_is_row_major(declaration))
        )
    names = []
    for name in sorted(orders):
        uses = orders[name]
        if not any(row_major for _, row_major in uses):
            continue
        unmarked = [
            member
            for member in structs[name].members or []
            if isinstance(member, A.StructMemberNode)
            and not member_attribute_names(member) & _ORDERS
            and _holds_matrix(member.member_type, structs)
        ]
        if not unmarked:
            continue
        others = [declaration.name for declaration, row_major in uses if not row_major]
        if others:
            row_major_names = [
                declaration.name for declaration, row_major in uses if row_major
            ]
            raise BlockMatrixOrderError(
                f"struct '{name}' is declared row_major by "
                f"{', '.join(repr(n) for n in row_major_names)} and column-major by "
                f"{', '.join(repr(n) for n in others)}; its member "
                f"'{unmarked[0].name}' cannot carry both orders",
                target=target,
            )
        names.append(name)
    return names


def distribute_block_matrix_orders(program, target) -> int:
    """Place block-level ``row_major`` orders on the block members for ``target``; returns how many members changed."""
    if not isinstance(program, A.ShaderNode):
        return 0
    if str(target or "").lower() in KEEPING_TARGETS:
        return 0
    structs = _structs(program)
    changed = 0
    for cbuffer in _declarations(program, "cbuffers"):
        if isinstance(cbuffer, A.StructNode) and member_is_row_major(cbuffer):
            changed += _place_row_major(cbuffer.members or [], structs)
    for name in _row_major_block_structs(program, structs, str(target)):
        changed += _place_row_major(structs[name].members or [], structs)
    return changed
