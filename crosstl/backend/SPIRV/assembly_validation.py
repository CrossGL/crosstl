"""Checks that a SPIR-V module is well formed before it is imported.

The assembly reader is lenient about layout, so a malformed module could otherwise reach the importer and fail
with an internal error, or be imported with a guessed meaning. These checks refuse it instead. Each rule is one
that ``spirv-as`` or ``spirv-val`` (SPIRV-Tools) enforces:

* every ``<id>`` operand of a type, constant, variable or function-body instruction is the result of some
  instruction of the module (``spirv-val``: "The following forward referenced IDs have not been defined", or
  "Operand ... requires a previous definition");
* a branch or merge instruction has the operands its grammar requires (``spirv-as``: "Expected operand for
  OpBranch instruction, but found the next instruction instead");
* each label operand of a branch or merge names a block of the same function (``spirv-val``: "'Target Label'
  operands for OpBranch must be the ID of an OpLabel instruction", "Block(s) ... are referenced but not defined
  in function");
* each result ``<id>`` is defined once (``spirv-val``: "Id ... is defined more than once"; SPIR-V 1.6 section 2.2.1,
  the module is in static single-assignment form);
* a type declaration names only types and constants declared before it, except a pointer type announced by
  ``OpTypeForwardPointer`` (``spirv-val``: "Operand ... requires a previous definition"; SPIR-V 1.6 section 2.4,
  logical layout item 9). A type that refers to itself or to a later type would otherwise make the reader recurse
  without end.
"""

from __future__ import annotations

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "SpirvInvalidModuleError",
    "check_control_flow_operands",
    "check_defined_ids",
    "check_type_declaration_order",
    "check_unique_result_ids",
]

# Instructions that only attach metadata to an <id>: entry points, execution modes, debug names and annotations.
# CrossTL keeps their records keyed by <id> and never dereferences them, and it deliberately accepts stub modules
# made of such records alone (an OpEntryPoint whose function is not in the module), so they are not checked.
METADATA_OPCODES = frozenset(
    {
        "OpEntryPoint",
        "OpExecutionMode",
        "OpExecutionModeId",
        "OpSource",
        "OpSourceContinued",
        "OpSourceExtension",
        "OpString",
        "OpName",
        "OpMemberName",
        "OpModuleProcessed",
        "OpLine",
        "OpNoLine",
        "OpDecorate",
        "OpDecorateId",
        "OpDecorateString",
        "OpMemberDecorate",
        "OpMemberDecorateString",
        "OpDecorationGroup",
        "OpGroupDecorate",
        "OpGroupMemberDecorate",
    }
)


class SpirvInvalidModuleError(TranslationDiagnostic):
    """A SPIR-V module that the SPIR-V specification itself rejects."""

    def __init__(self, detail):
        super().__init__(
            f"invalid SPIR-V module: {detail}",
            target="cgl",
            code=f"{CODE_PREFIX}spirv-invalid-module",
        )


def check_defined_ids(id_operands, defined_ids):
    """Refuse an ``<id>`` operand of a declaration or function-body instruction that no instruction defines.

    ``id_operands`` holds ``(line, opcode, ids)`` for each instruction, where ``ids`` are its unquoted operands
    spelled ``%name``; ``defined_ids`` holds every result ``<id>`` of the module.
    """
    for line, opcode, ids in id_operands:
        if opcode in METADATA_OPCODES:
            continue
        for operand in ids:
            if operand not in defined_ids:
                raise SpirvInvalidModuleError(
                    f"{opcode} on line {line} uses {operand}, which no instruction defines"
                )


# Operand counts after the result type and result id, per the SPIR-V grammar: OpBranchConditional may add two
# literal branch weights, and OpSwitch is a selector, a default label, then (literal, label) pairs.
def _operand_count_valid(opcode, count):
    if opcode == "OpBranch":
        return count == 1
    if opcode == "OpBranchConditional":
        return count in (3, 5)
    if opcode == "OpSwitch":
        return count >= 2 and count % 2 == 0
    if opcode == "OpSelectionMerge":
        return count == 2
    if opcode == "OpLoopMerge":
        return count >= 3
    return True


def _label_operands(opcode, operands):
    if opcode == "OpBranch":
        return operands[:1]
    if opcode == "OpBranchConditional":
        return operands[1:3]
    if opcode == "OpSwitch":
        return [operands[1], *operands[3::2]]
    if opcode == "OpSelectionMerge":
        return operands[:1]
    if opcode == "OpLoopMerge":
        return operands[:2]
    return []


def check_control_flow_operands(block_label, opcode, operands, blocks):
    """Refuse a branch or merge instruction with the wrong operands, or a label operand outside ``blocks``."""
    if not _operand_count_valid(opcode, len(operands)):
        raise SpirvInvalidModuleError(
            f"{opcode} in block {block_label} has {len(operands)} operands: {' '.join(operands) or '(none)'}"
        )
    for label in _label_operands(opcode, operands):
        if label not in blocks:
            raise SpirvInvalidModuleError(
                f"{opcode} in block {block_label} names {label}, which is not a block of the function"
            )


def check_unique_result_ids(instructions):
    """Refuse a result ``<id>`` that two instructions define.

    ``instructions`` holds ``(result_id, opcode, operands, line)`` for each instruction in module order.
    """
    defined_on = {}
    for result_id, opcode, _operands, line in instructions:
        if not result_id:
            continue
        if result_id in defined_on:
            raise SpirvInvalidModuleError(
                f"{opcode} on line {line} defines {result_id}, which line {defined_on[result_id]} already defines"
            )
        defined_on[result_id] = line


def check_type_declaration_order(instructions):
    """Refuse a type declaration whose ``<id>`` operand is not declared before it.

    The one exception is a pointer type that an ``OpTypeForwardPointer`` announced. ``instructions`` holds
    ``(result_id, opcode, operands, line)`` for each instruction in module order.
    """
    forward_pointers = {
        operands[0]
        for _result_id, opcode, operands, _line in instructions
        if opcode == "OpTypeForwardPointer" and operands
    }
    declared = set()
    for result_id, opcode, operands, line in instructions:
        if opcode.startswith("OpType") and opcode != "OpTypeForwardPointer":
            for operand in operands:
                if (
                    isinstance(operand, str)
                    and operand.startswith("%")
                    and operand not in declared
                    and operand not in forward_pointers
                ):
                    raise SpirvInvalidModuleError(
                        f"{opcode} on line {line} uses {operand} before any instruction declares it"
                    )
        if result_id:
            declared.add(result_id)
