"""GLSL.std.450 ``ModfStruct`` and ``FrexpStruct`` imported as CrossGL ``modf`` and ``frexp`` with an out operand.

``ModfStruct`` returns a struct whose member 0 is the fractional part and member 1 the whole number; ``FrexpStruct``
returns one whose member 0 is the significand and member 1 the exponent (GLSL.std.450 instructions 36 and 52).
CrossGL spells both as GLSL does, ``modf(x, out whole)`` and ``frexp(x, out exponent)``, each returning member 0
(GLSL 4.60 section 8.3). The importer used to write ``modf(x)[1]``, an index into a call no CrossGL emitter accepts.

The instruction now becomes two locals: the out operand, declared first, and the call's result. Each
``OpCompositeExtract`` of the struct reads one of them, so later indices into a vector member still apply. The
struct value itself stays the one-operand call, which every emitter refuses, so any other use of the struct (a
store, a function argument, ``OpCopyObject``) fails closed instead of being mistranslated.
"""

from __future__ import annotations

from .VulkanAst import AssignmentNode, FunctionCallNode, VariableNode

__all__ = ["spirv_struct_result_call", "spirv_struct_result_member"]

GLSL_STD_450 = "GLSL.std.450"

# Instruction name -> (CrossGL function, suffix of member 0's local, suffix of member 1's local).
STRUCT_RESULT_INSTRUCTIONS = {
    "ModfStruct": ("modf", "fract", "whole"),
    "FrexpStruct": ("frexp", "significand", "exponent"),
}
STRUCT_RESULT_INSTRUCTION_IDS = {"36": "ModfStruct", "52": "FrexpStruct"}


def _instruction_name(instruction_set, instruction):
    if instruction_set != GLSL_STD_450:
        return None
    text = str(instruction)
    if text in STRUCT_RESULT_INSTRUCTIONS:
        return text
    return STRUCT_RESULT_INSTRUCTION_IDS.get(text)


def _unique_name(base, names, taken):
    used = set(names.values()) | taken
    if base not in used:
        return base
    suffix = 1
    while f"{base}_{suffix}" in used:
        suffix += 1
    return f"{base}_{suffix}"


def spirv_struct_result_call(
    parser,
    result_id,
    result_type_id,
    instruction_set,
    instruction,
    argument,
    statements,
    names,
    decorations,
    types,
    constants,
):
    """The struct value of a ``ModfStruct`` or ``FrexpStruct`` instruction, or ``None`` for any other instruction.

    Appends the declarations of the two member locals to ``statements``. The returned call carries the member
    locals in ``spirv_struct_members`` for :func:`spirv_struct_result_member`.
    """
    instruction_name = _instruction_name(instruction_set, instruction)
    if instruction_name is None:
        return None
    member_types = types.get(result_type_id, {}).get("member_types", [])
    if len(member_types) != 2:
        return None
    function, result_suffix, out_suffix = STRUCT_RESULT_INSTRUCTIONS[instruction_name]
    base = parser.spirv_assembly_value_name(
        result_id, names, decorations, prefix="value"
    )
    taken = set()
    members = []
    for suffix, type_id in zip((result_suffix, out_suffix), member_types):
        type_name, _array_suffix = parser.spirv_type_name_and_suffix(
            type_id, types, constants, names=names
        )
        name = _unique_name(f"{base}_{suffix}", names, taken)
        taken.add(name)
        members.append((name, type_name, type_id))
    (result_name, result_type, result_member_type), (
        out_name,
        out_type,
        out_member_type,
    ) = members
    statements.append(VariableNode(out_type, out_name, spirv_type_id=out_member_type))
    statements.append(
        AssignmentNode(
            VariableNode(result_type, result_name, spirv_type_id=result_member_type),
            FunctionCallNode(function, [argument, VariableNode("", out_name)]),
        )
    )
    struct_value = FunctionCallNode(function, [argument])
    struct_value.spirv_struct_members = (
        VariableNode("", result_name),
        VariableNode("", out_name),
    )
    return struct_value


def spirv_struct_result_member(expression, member_index):
    """The local holding member ``member_index`` of a struct from :func:`spirv_struct_result_call`, else ``None``."""
    members = getattr(expression, "spirv_struct_members", None)
    if members is None or not 0 <= member_index < len(members):
        return None
    return members[member_index]
