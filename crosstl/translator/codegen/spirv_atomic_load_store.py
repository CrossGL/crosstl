"""SPIR-V lowering of the CrossGL atomic loads and stores.

CrossGL ``atomicLoad(target)`` and ``atomicStore(target, value)`` are relaxed atomic accesses to one scalar integer,
as Metal's ``atomic_load_explicit`` and ``atomic_store_explicit`` with ``memory_order_relaxed`` and GLSL's
``atomicLoad`` and ``atomicStore`` from GL_KHR_memory_scope_semantics are. They become ``OpAtomicLoad`` and
``OpAtomicStore`` (SPIR-V 1.6, section 3.52.18) with relaxed (``None``) memory semantics, at ``Device`` scope for
storage buffers and at ``Workgroup`` scope for workgroup storage, the scopes the other buffer atomics use.

The operand rules follow the other buffer atomics: the target is a scalar ``int`` or ``uint`` in a storage buffer
or in workgroup storage. A load needs a readable target and a store a writable one, so a store to a ``readonly``
buffer and a load from a ``writeonly`` buffer are refused.
"""

from __future__ import annotations

__all__ = [
    "SPIRV_ATOMIC_LOAD_STORE_FUNCTIONS",
    "call_spirv_atomic_load_store",
    "spirv_atomic_load_store_access",
]

# CrossGL name -> number of operands.
SPIRV_ATOMIC_LOAD_STORE_FUNCTIONS = {"atomicLoad": 1, "atomicStore": 2}


def spirv_atomic_load_store_access(function_name: str) -> str | None:
    """The storage-buffer access an atomic load or store needs, or None for other functions."""
    if function_name == "atomicLoad":
        return "read"
    if function_name == "atomicStore":
        return "write"
    return None


def _target_type(codegen, target_pointer):
    target_type = codegen.variable_value_types.get(target_pointer.id)
    if target_type is None and target_pointer.type.storage_class:
        target_type = codegen.find_registered_type_by_base(
            target_pointer.type.base_type.replace("ptr_", "", 1)
        )
    return target_type


def _operand_error(codegen, function_name: str, target_pointer) -> str | None:
    """Why ``target_pointer`` cannot be the target of ``function_name``, or None when it can."""
    target_type = _target_type(codegen, target_pointer)
    if target_type is None:
        return f"{function_name} requires an addressable target"
    metadata = codegen.storage_buffer_access_metadata_for_pointer(target_pointer)
    workgroup = metadata is None and target_pointer.type.storage_class == "Workgroup"
    if metadata is None and not workgroup:
        return f"{function_name} requires a storage buffer or workgroup target"
    if metadata is not None:
        if function_name == "atomicStore" and metadata.get("readonly"):
            return f"{function_name} requires a writable storage buffer"
        if function_name == "atomicLoad" and metadata.get("writeonly"):
            return f"{function_name} requires a readable storage buffer"
    if (
        codegen.vector_type_info_from_type(target_type) is not None
        or codegen.matrix_type_info_from_type(target_type) is not None
    ):
        return f"{function_name} requires a scalar int or uint target"
    type_name = codegen.normalize_primitive_name(target_type.type.base_type)
    if type_name not in {"int", "uint"}:
        return f"{function_name} currently supports only int or uint targets"
    return None


def call_spirv_atomic_load_store(
    codegen, function_name: str, args: list
) -> object | None:
    """Emit ``atomicLoad`` or ``atomicStore`` through ``codegen``.

    Returns the loaded value for ``atomicLoad`` and None for ``atomicStore``. A refused call records the reason
    through ``codegen.refuse_lowering`` and returns a placeholder so generation can reach later refusals.
    """
    operand_count = SPIRV_ATOMIC_LOAD_STORE_FUNCTIONS[function_name]
    if len(args) != operand_count:
        operands = "a target" if operand_count == 1 else "a target and a value"
        codegen.refuse_lowering(f"{function_name} requires exactly {operands}")
        return _placeholder(codegen, function_name, args)

    target_pointer = args[0]
    error = _operand_error(codegen, function_name, target_pointer)
    if error is not None:
        codegen.refuse_lowering(error)
        return _placeholder(codegen, function_name, args)

    target_type = _target_type(codegen, target_pointer)
    workgroup = target_pointer.type.storage_class == "Workgroup"
    scope = codegen.spirv_scope_constant("Workgroup" if workgroup else "Device")
    semantics = codegen.spirv_memory_semantics_constant()
    if function_name == "atomicStore":
        value = codegen.convert_value_to_type(args[1], target_type)
        codegen.emit(
            f"OpAtomicStore %{target_pointer.id} %{scope.id} %{semantics.id} "
            f"%{value.id}"
        )
        return None

    from .SPIRV_codegen import (
        SpirvId,  # imported here: the host module imports this one
    )

    id_value = codegen.get_id()
    codegen.emit(
        f"%{id_value} = OpAtomicLoad %{target_type.id} %{target_pointer.id} "
        f"%{scope.id} %{semantics.id}"
    )
    codegen.value_types[id_value] = target_type
    return SpirvId(id_value, target_type.type)


def _placeholder(codegen, function_name: str, args: list):
    if function_name == "atomicStore":
        return None
    target_type = _target_type(codegen, args[0]) if args else None
    if target_type is not None:
        return codegen.default_value_for_type(target_type)
    return codegen.register_constant(0, codegen.register_primitive_type("int"))
