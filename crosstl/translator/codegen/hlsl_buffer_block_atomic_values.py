"""HLSL assignments of atomic results to members of GLSL buffer blocks.

The DirectX generator stores a GLSL buffer block as a ``RWByteAddressBuffer``, so a write to one of its members is a
``Store`` call, not an assignment (HLSL intrinsic reference, "RWByteAddressBuffer::Store"). An atomic whose result
is assigned to such a member, as in ``block.value = atomicAdd(shared, 1u)`` or ``block.data[i] = x + atomicLoad(c)``,
went through the atomic statement lowering, which renders the target as an expression: the generator printed the
member's ``Load`` call as the ``original_value`` argument of an ``Interlocked*`` call, or on the left of an
assignment, which DXC rejects ("expression is not assignable").

:func:`generate_hlsl_buffer_block_atomic_value_assignment` evaluates the atomic value into a temporary through the
ordinary atomic lowering, then writes the temporary through the buffer block store, which applies a compound
operator, the member's layout and the readonly refusal as any other store does.
:func:`generate_hlsl_buffer_block_interlocked_statement` does the same for an ``Interlocked*`` call whose
``original_value`` argument, an ``out`` parameter, is a buffer block member, the form the semantic pass gives
``member = atomicAdd(shared, value)``.
"""

from __future__ import annotations

import copy

from ..ast import (
    AssignmentNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
)

__all__ = [
    "generate_hlsl_buffer_block_atomic_value_assignment",
    "generate_hlsl_buffer_block_interlocked_statement",
]

# Interlocked function -> position of its ``original_value`` out parameter.
_INTERLOCKED_ORIGINAL_INDEX = {
    "InterlockedAdd": 2,
    "InterlockedMin": 2,
    "InterlockedMax": 2,
    "InterlockedAnd": 2,
    "InterlockedOr": 2,
    "InterlockedXor": 2,
    "InterlockedExchange": 2,
    "InterlockedCompareExchange": 3,
}


def _assignment_parts(stmt):
    if hasattr(stmt, "target") and hasattr(stmt, "value"):
        return stmt.target, stmt.value, getattr(stmt, "operator", "=")
    return stmt.left, stmt.right, getattr(stmt, "operator", "=")


def _buffer_block_access(codegen, target):
    """The byte-address access ``target`` names, as the buffer block store resolves it, or None."""
    access = codegen.glsl_buffer_block_array_access(target)
    if access is not None:
        return access
    access = codegen.glsl_buffer_block_member_access(target)
    if access is None or access.get("runtime_array"):
        return None
    return access


def _declare_temporary(codegen, access, indent):
    value_type = access["type"]
    temporary = codegen.next_hlsl_temp_variable("atomic_value")
    codegen.local_variable_types[temporary] = value_type
    return temporary, f"{'    ' * indent}{codegen.map_type(value_type)} {temporary};\n"


def generate_hlsl_buffer_block_atomic_value_assignment(codegen, stmt, indent):
    """HLSL for ``stmt`` when it assigns a value holding an atomic to a buffer block member, else None."""
    target, value, op = _assignment_parts(stmt)
    if not codegen.hlsl_expression_contains_typed_buffer_atomic(value):
        return None
    access = _buffer_block_access(codegen, target)
    if access is None or not access.get("type"):
        return None

    temporary, code = _declare_temporary(codegen, access, indent)
    code += codegen.generate_statement(
        AssignmentNode(IdentifierNode(temporary), value, "="), indent
    )
    store = codegen.generate_glsl_buffer_block_store(
        target, IdentifierNode(temporary), op
    )
    return code + codegen.generate_statement_code(store, indent)


def generate_hlsl_buffer_block_interlocked_statement(codegen, expression, indent):
    """HLSL for an ``Interlocked*`` call statement whose original value goes to a buffer block member, else None."""
    if not isinstance(expression, FunctionCallNode):
        return None
    index = _INTERLOCKED_ORIGINAL_INDEX.get(codegen.function_call_name(expression))
    arguments = list(expression.arguments or [])
    if index is None or len(arguments) != index + 1:
        return None
    original = arguments[index]
    access = _buffer_block_access(codegen, original)
    if access is None or not access.get("type"):
        return None

    temporary, code = _declare_temporary(codegen, access, indent)
    call = copy.copy(expression)
    call.arguments = call.args = [*arguments[:index], IdentifierNode(temporary)]
    code += codegen.generate_statement(ExpressionStatementNode(call), indent)
    store = codegen.generate_glsl_buffer_block_store(
        original, IdentifierNode(temporary), "="
    )
    return code + codegen.generate_statement_code(store, indent)
