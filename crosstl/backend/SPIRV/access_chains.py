"""Typed SPIR-V access chains.

``OpAccessChain`` indexes its base by walking the pointee type (SPIR-V 1.6 section 3.49.8): an index into a
``struct`` must be an ``OpConstant`` naming the member, an index into an array, matrix or vector selects an element.
CrossGL spells the first as a member access and the second as a subscript, so the importer has to know the type at
every step. Without the type, ``camera.pos`` of a struct member of a uniform block was imported as ``camera[0]``,
which no target compiles, and a struct passed to a function lost its member names the same way.

:func:`walk_indices` continues an already built expression through the remaining indices of a chain;
:func:`chain_base_pointee` finds the type the first index applies to.
"""

from __future__ import annotations

from .VulkanAst import ArrayAccessNode, MemberAccessNode

__all__ = ["chain_base_pointee", "member_name", "walk_indices"]


def member_name(member_names, struct_type_id, member_index):
    """The name the importer declares a struct member under (``memberN`` when the module has no ``OpMemberName``)."""
    key = str(member_index)
    return member_names.get(struct_type_id, {}).get(key, f"member{key}")


def chain_base_pointee(base_operand, types, variables_by_id, expression_type_ids):
    """The pointee type id of an access chain's base pointer, or ``None`` when it is not known."""
    variable = (variables_by_id or {}).get(base_operand)
    pointer_type_id = (
        variable["pointer_type_id"]
        if variable is not None
        else (expression_type_ids or {}).get(base_operand)
    )
    pointer_type = types.get(pointer_type_id, {})
    if pointer_type.get("kind") != "pointer":
        return None
    return pointer_type.get("type_id")


def walk_indices(
    parser,
    access,
    type_id,
    index_operands,
    operand_expression,
    member_names,
    types,
    constants,
):
    """Apply ``index_operands`` to ``access``, whose type is ``type_id``.

    A struct index becomes a member access; any other index becomes a subscript. When ``type_id`` is unknown the
    remaining indices stay subscripts, which is exact for arrays, vectors and matrices. A struct index that is not an
    in-range constant makes the module invalid, so it is refused rather than imported as a subscript.
    """
    for index_operand in index_operands:
        type_info = types.get(type_id, {})
        if type_info.get("kind") == "struct":
            member_index = parser.spirv_integer_constant_operand(
                index_operand, constants
            )
            member_types = type_info.get("member_types", [])
            if member_index is None or not 0 <= member_index < len(member_types):
                raise SyntaxError(
                    f"SPIR-V access chain indexes struct {type_id} with {index_operand}: a struct index must be an "
                    "in-range OpConstant (SPIR-V 1.6 section 3.49.8, OpAccessChain)"
                )
            access = MemberAccessNode(
                access, member_name(member_names, type_id, member_index)
            )
            type_id = member_types[member_index]
            continue
        access = ArrayAccessNode(access, operand_expression(index_operand))
        type_id = parser.spirv_composite_index_type_id(type_id, index_operand, types)
    return access
