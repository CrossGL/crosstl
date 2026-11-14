"""SPIR-V emission of CrossGL byte-address buffer atomics.

CrossGL spells an atomic on a byte-address buffer ``atomicAdd(buffer, offset, value)`` (likewise ``atomicMin``,
``atomicMax``, ``atomicAnd``, ``atomicOr``, ``atomicXor`` and ``atomicExchange``) and
``atomicCompareExchange(buffer, offset, compare, value)`` (``atomicCompSwap`` is a synonym), each returning the
previous value of the 32-bit word at ``offset``. The generator binds a byte-address buffer as a runtime array of
uint words, so each call is the matching ``OpAtomic*`` instruction on the word ``offset / 4``, as for the HLSL
``RWByteAddressBuffer`` method form the generator already lowers.

The word is unsigned. As in HLSL, ``atomicMin`` and ``atomicMax`` with an ``int`` value are the signed
``OpAtomicSMin``/``OpAtomicSMax`` (SPIR-V 1.6 section 3.42.18; their operand and result types may be unsigned,
since the opcode alone fixes the signedness) and return ``int``; a non-negative ``int`` literal is the unsigned form.
"""

from __future__ import annotations

from ..ast import IdentifierNode, LiteralNode, VariableNode

__all__ = ["process_byte_address_atomic_call", "process_byte_address_member_min_max"]

# CrossGL name: HLSL RWByteAddressBuffer method whose lowering the call shares.
_METHODS = {
    "atomicAdd": "InterlockedAdd",
    "atomicMin": "InterlockedMin",
    "atomicMax": "InterlockedMax",
    "atomicAnd": "InterlockedAnd",
    "atomicOr": "InterlockedOr",
    "atomicXor": "InterlockedXor",
    "atomicExchange": "InterlockedExchange",
    # The previous value comes back as the instruction's result, which is what CompareStore lowers to.
    "atomicCompareExchange": "InterlockedCompareStore",
    "atomicCompSwap": "InterlockedCompareStore",
}
_SIGNED_OPCODES = {
    "InterlockedMin": "OpAtomicSMin",
    "InterlockedMax": "OpAtomicSMax",
}


def _call_name(expr):
    func_expr = getattr(expr, "function", getattr(expr, "name", None))
    if isinstance(func_expr, str):
        return func_expr
    if isinstance(func_expr, (IdentifierNode, VariableNode)):
        name = getattr(func_expr, "name", None)
        return name if isinstance(name, str) else None
    return None


def _is_unsigned_literal(expr):
    value = getattr(expr, "value", None) if isinstance(expr, LiteralNode) else None
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        text = value.rstrip("uU")
        if not text.isdigit():
            return False
        value = int(text)
    return isinstance(value, int) and value >= 0


def _value_kind(codegen, value):
    value_type = codegen.value_types.get(
        value.id
    ) or codegen.find_registered_type_by_base(value.type.base_type)
    if value_type is None:
        return None
    if (
        codegen.vector_type_info_from_type(value_type) is not None
        or codegen.matrix_type_info_from_type(value_type) is not None
    ):
        return None
    return codegen.normalize_primitive_name(value_type.type.base_type)


def _original_pointer(codegen, expr, diagnostic_name):
    """The pointer to a scalar ``int`` or ``uint`` ``out`` target receiving the previous word."""
    pointer = codegen.assignable_pointer_from_expression(expr)
    if pointer is None:
        codegen.refuse_lowering(
            f"{diagnostic_name} original operand must be an assignable scalar int or uint target"
        )
        return None, None
    pointee = codegen.pointer_pointee_type(pointer)
    if (
        pointee is None
        or codegen.vector_type_info_from_type(pointee) is not None
        or codegen.matrix_type_info_from_type(pointee) is not None
        or codegen.normalize_primitive_name(pointee.type.base_type)
        not in {"int", "uint"}
    ):
        codegen.refuse_lowering(
            f"{diagnostic_name} original operand must be scalar int or uint"
        )
        return None, None
    return pointer, pointee


def _signed_min_max(codegen, buffer_pointer, metadata, method, args):
    """``OpAtomicSMin``/``OpAtomicSMax`` (or the unsigned form for a uint value) on the word at ``args[0]``.

    ``args[2]``, when present, is the HLSL method's ``out`` original value.
    """
    diagnostic_name = f"RWByteAddressBuffer.{method}"
    uint_type = codegen.register_primitive_type("uint")
    if metadata.get("readonly") or metadata.get("writeonly"):
        codegen.refuse_lowering(f"{diagnostic_name} requires a read-write buffer")
        return codegen.byte_address_interlocked_default_value()
    element_index = codegen.byte_address_buffer_element_index(args[0])
    if element_index is None:
        return codegen.byte_address_interlocked_default_value()
    value = codegen.process_expression(args[1])
    if value is None:
        codegen.refuse_lowering(
            f"{diagnostic_name} value operand could not be evaluated"
        )
        return codegen.byte_address_interlocked_default_value()
    kind = _value_kind(codegen, value)
    if kind not in {"int", "uint"}:
        codegen.refuse_lowering(
            f"{diagnostic_name} value operand must be a scalar integer"
        )
        return codegen.byte_address_interlocked_default_value()
    signed = kind == "int" and not _is_unsigned_literal(args[1])
    original_pointer = original_type = None
    if len(args) > 2:
        original_pointer, original_type = _original_pointer(
            codegen, args[2], diagnostic_name
        )
        if original_pointer is None:
            return codegen.byte_address_interlocked_default_value()
    target_pointer = codegen.structured_buffer_element_pointer(
        buffer_pointer, element_index
    )
    if target_pointer is None:
        codegen.refuse_lowering(
            f"{diagnostic_name} requires a byte-address buffer element"
        )
        return codegen.byte_address_interlocked_default_value()
    opcode = (
        _SIGNED_OPCODES[method]
        if signed
        else codegen.byte_address_method_interlocked_info(method)["opcode"]
    )
    result = codegen.emit_byte_address_interlocked_atomic(
        opcode, target_pointer, [codegen.convert_value_to_type(value, uint_type)]
    )
    if original_pointer is not None:
        codegen.store_to_variable(
            original_pointer, codegen.convert_value_to_type(result, original_type)
        )
    if signed:
        return codegen.convert_value_to_type(
            result, codegen.register_primitive_type("int")
        )
    return result


def process_byte_address_member_min_max(
    codegen, buffer_pointer, metadata, method, args
):
    """``(handled, result)`` for ``RWByteAddressBuffer.InterlockedMin/Max(offset, value[, original])``.

    The method's generic lowering is always unsigned; an ``int`` value selects the signed opcode, as DXC does.
    Other argument counts stay with the generic lowering, which refuses them.
    """
    if method not in _SIGNED_OPCODES or len(args) not in (2, 3):
        return False, None
    return True, _signed_min_max(codegen, buffer_pointer, metadata, method, args)


def process_byte_address_atomic_call(codegen, expr):
    """``(handled, result)`` for a CrossGL atomic call on a byte-address buffer."""
    func_name = _call_name(expr)
    method = _METHODS.get(func_name)
    args = list(getattr(expr, "args", None) or getattr(expr, "arguments", None) or [])
    if method is None or not args or func_name in codegen.function_signatures:
        return False, None
    # A byte-address buffer is named directly. Probing any other destination, such as ``counters[i]``, would
    # emit its access chain a second time before the ordinary atomic lowering evaluates it.
    if not isinstance(args[0], (IdentifierNode, VariableNode)):
        return False, None
    buffer_pointer = codegen.variable_pointer_from_expression(args[0])
    if buffer_pointer is None:
        return False, None
    metadata = codegen.structured_buffer_metadata_for_pointer(buffer_pointer)
    if metadata is None or not metadata.get("byte_address"):
        return False, None

    expected = 4 if method == "InterlockedCompareStore" else 3
    if len(args) != expected:
        operands = "a compare value and a value" if expected == 4 else "a value"
        codegen.refuse_lowering(
            f"{func_name} on a byte-address buffer takes the buffer, a byte offset and {operands}"
        )
        return True, codegen.byte_address_interlocked_default_value()
    if method in _SIGNED_OPCODES:
        return True, _signed_min_max(
            codegen, buffer_pointer, metadata, method, args[1:]
        )
    return True, codegen.process_byte_address_buffer_interlocked_call(
        buffer_pointer, metadata, method, args[1:]
    )
