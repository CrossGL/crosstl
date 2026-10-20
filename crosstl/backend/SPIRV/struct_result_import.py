"""GLSL.std.450 ``ModfStruct`` and ``FrexpStruct`` imported as CrossGL ``modf`` and ``frexp`` with an out operand.

``ModfStruct`` returns a struct whose member 0 is the fractional part and member 1 the whole number; ``FrexpStruct``
returns one whose member 0 is the significand and member 1 the exponent (GLSL.std.450 instructions 36 and 52).
CrossGL spells both as GLSL does, ``modf(x, out whole)`` and ``frexp(x, out exponent)``, each returning member 0
(GLSL 4.60 section 8.3). The importer used to write ``modf(x)[1]``, an index into a call no CrossGL emitter accepts.

The instruction now becomes two locals: the out operand, declared first, and the call's result. Each
``OpCompositeExtract`` of the struct reads one of them, so later indices into a vector member still apply. The
struct value itself stays the one-operand call, which every emitter refuses, so any other use of the struct (a
store, a function argument, ``OpCopyObject``) fails closed instead of being mistranslated.

The core instructions ``OpIAddCarry``, ``OpISubBorrow``, ``OpUMulExtended`` and ``OpSMulExtended`` (SPIR-V 1.6
section 3.52.13) return the same kind of two-member struct and import the same way, as GLSL's ``uaddCarry``,
``usubBorrow``, ``umulExtended`` and ``imulExtended`` (GLSL 4.60 section 8.8). The multiplications return nothing
and write the high word (struct member 1) and then the low word (member 0) to out operands; their struct value is
the two-operand call, which every emitter refuses.
"""

from __future__ import annotations

from .integer_signedness_import import spirv_integer_type_name
from .VulkanAst import AssignmentNode, FunctionCallNode, VariableNode

__all__ = [
    "EXTENDED_ARITHMETIC_INSTRUCTIONS",
    "spirv_extended_arithmetic_call",
    "spirv_struct_result_call",
    "spirv_struct_result_member",
]

GLSL_STD_450 = "GLSL.std.450"

# Instruction name -> (CrossGL function, suffix of member 0's local, suffix of member 1's local).
STRUCT_RESULT_INSTRUCTIONS = {
    "ModfStruct": ("modf", "fract", "whole"),
    "FrexpStruct": ("frexp", "significand", "exponent"),
}
STRUCT_RESULT_INSTRUCTION_IDS = {"36": "ModfStruct", "52": "FrexpStruct"}

# Core opcode -> (CrossGL function, suffix of member 0's local, suffix of member 1's local, signedness GLSL needs).
EXTENDED_ARITHMETIC_INSTRUCTIONS = {
    "IAddCarry": ("uaddCarry", "sum", "carry", False),
    "ISubBorrow": ("usubBorrow", "difference", "borrow", False),
    "UMulExtended": ("umulExtended", "lsb", "msb", False),
    "SMulExtended": ("imulExtended", "lsb", "msb", True),
}
_MULTIPLICATIONS = {"UMulExtended", "SMulExtended"}


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


def _reinterpreted(expression, type_name, wanted_name):
    if type_name == wanted_name:
        return expression
    return FunctionCallNode(wanted_name, [expression])


def spirv_extended_arithmetic_call(
    parser,
    result_id,
    result_type_id,
    operation,
    arguments,
    statements,
    names,
    decorations,
    types,
):
    """The struct value of an extended-arithmetic instruction, or ``None`` for any other opcode.

    Appends the member locals and the call that writes them to ``statements``. ``OpSMulExtended`` may have unsigned
    members (its operands are read as signed whatever their type); ``imulExtended`` takes ``int`` operands, so the
    operands, the locals and the members read back are reinterpreted with same-width constructors.
    """
    entry = EXTENDED_ARITHMETIC_INSTRUCTIONS.get(operation)
    if entry is None or len(arguments) != 2:
        return None
    function, first_suffix, second_suffix, signed = entry
    struct_value = FunctionCallNode(function, list(arguments))
    member_types = types.get(result_type_id, {}).get("member_types", [])
    if len(member_types) != 2 or member_types[0] != member_types[1]:
        return struct_value
    member_type_id = member_types[0]
    member_name = parser.spirv_type_name(member_type_id, types)
    local_type = spirv_integer_type_name(
        member_type_id, types, signed, parser.spirv_vector_type_name
    )
    if member_name is None or local_type is None:
        return struct_value
    base = parser.spirv_assembly_value_name(
        result_id, names, decorations, prefix="value"
    )
    first = _unique_name(f"{base}_{first_suffix}", names, set())
    second = _unique_name(f"{base}_{second_suffix}", names, {first})
    operands = [
        _reinterpreted(argument, member_name, local_type) for argument in arguments
    ]
    statements.append(VariableNode(local_type, second))
    if operation in _MULTIPLICATIONS:
        statements.append(VariableNode(local_type, first))
        statements.append(
            FunctionCallNode(
                function, operands + [VariableNode("", second), VariableNode("", first)]
            )
        )
    else:
        statements.append(
            AssignmentNode(
                VariableNode(local_type, first),
                FunctionCallNode(function, operands + [VariableNode("", second)]),
            )
        )
    struct_value.spirv_struct_members = tuple(
        _reinterpreted(VariableNode("", name), local_type, member_name)
        for name in (first, second)
    )
    return struct_value
