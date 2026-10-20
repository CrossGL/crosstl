"""SPIR-V ``OpControlBarrier`` and ``OpMemoryBarrier`` imported as CrossGL barrier intrinsics.

The importer used to spell each barrier as ``spirvControlBarrier(exec, mem, semantics)``, which no CrossGL emitter
lowers: every target either failed or printed the call verbatim. Each barrier now becomes the CrossGL barrier, or
sequence of barriers, whose synchronization covers the SPIR-V instruction's (SPIR-V 1.6 sections 3.27 and 3.25):

* CrossGL ``memoryBarrierShared`` orders workgroup memory at workgroup scope; ``groupMemoryBarrier`` orders
  buffer, image and workgroup memory at workgroup scope (GLSL 4.60 section 8.16); ``memoryBarrierBuffer``,
  ``memoryBarrierImage`` and ``deviceMemoryBarrier`` order buffer, image, and buffer and image memory at device
  scope; ``memoryBarrier`` orders all three at device scope.
* ``workgroupBarrier`` is a workgroup control barrier that orders workgroup memory, ``workgroupExecutionBarrier``
  one without memory semantics, and ``barrier`` in a tessellation control stage the patch barrier
  (``OpControlBarrier Workgroup Invocation None``, as glslang emits it).

The mapping never weakens a barrier. Workgroup memory is visible within one workgroup only, so a barrier limited to
it is a workgroup-scope barrier whatever its memory scope. Otherwise a memory barrier picks the narrowest CrossGL fence whose scope and storage
classes cover the instruction's; Acquire, Release and AcquireRelease all become the CrossGL fences' acquire-release
ordering. A control barrier whose memory semantics reach beyond workgroup memory, or beyond workgroup scope,
becomes ``fence; workgroupBarrier(); fence``, so that both the writes before the barrier and the reads after it are
ordered. A barrier whose operands are not constants (specialization constants included), whose execution scope is
not the workgroup, or whose memory scope is ``CrossDevice`` has no CrossGL equivalent and is refused.
"""

from __future__ import annotations

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

from .VulkanAst import FunctionCallNode

__all__ = ["SpirvBarrierUnsupportedError", "spirv_barrier_statements"]

# SPIR-V Scope values (SPIR-V 1.6 section 3.27).
CROSS_DEVICE, DEVICE, WORKGROUP, SUBGROUP, INVOCATION, QUEUE_FAMILY, SHADER_CALL = (
    range(7)
)
# SPIR-V Memory Semantics storage-class bits (SPIR-V 1.6 section 3.25).
UNIFORM_MEMORY = 0x40
SUBGROUP_MEMORY = 0x80
WORKGROUP_MEMORY = 0x100
CROSS_WORKGROUP_MEMORY = 0x200
ATOMIC_COUNTER_MEMORY = 0x400
IMAGE_MEMORY = 0x800
OUTPUT_MEMORY = 0x1000
# Buffer memory as GLSL and HLSL see it: Uniform and StorageBuffer storage, plus atomic counters.
_BUFFER_BITS = UNIFORM_MEMORY | CROSS_WORKGROUP_MEMORY | ATOMIC_COUNTER_MEMORY
_WORKGROUP_BITS = WORKGROUP_MEMORY | SUBGROUP_MEMORY
_STORAGE_BITS = _BUFFER_BITS | _WORKGROUP_BITS | IMAGE_MEMORY | OUTPUT_MEMORY
_SCOPE_NAMES = {
    CROSS_DEVICE: "CrossDevice",
    DEVICE: "Device",
    WORKGROUP: "Workgroup",
    SUBGROUP: "Subgroup",
    INVOCATION: "Invocation",
    QUEUE_FAMILY: "QueueFamily",
    SHADER_CALL: "ShaderCallKHR",
}


class SpirvBarrierUnsupportedError(TranslationDiagnostic):
    """A SPIR-V barrier whose synchronization no CrossGL barrier expresses."""

    def __init__(self, message: str):
        self.missing_capabilities = ("spirv.import.barrier",)
        super().__init__(
            message, target="cgl", code=f"{CODE_PREFIX}spirv-barrier-unsupported"
        )


def _constant_value(value, opcode: str, operand_name: str) -> int:
    text = str(value).strip() if value is not None else ""
    try:
        return int(text, 0)
    except ValueError:
        raise SpirvBarrierUnsupportedError(
            f"{opcode} {operand_name} '{value}' is not a constant: CrossGL barriers have fixed scopes and "
            "memory semantics"
        ) from None


def _memory_fence(scope: int, storage: int, opcode: str):
    """The narrowest CrossGL memory barrier ordering ``storage`` at ``scope`` or wider, or ``None``."""
    if not storage:
        return None
    if not storage & ~_WORKGROUP_BITS:
        # Workgroup memory is shared within one workgroup only, so any wider scope orders it as workgroup scope
        # does; glslang emits memoryBarrierShared() at Device scope.
        return "memoryBarrierShared"
    if scope in {WORKGROUP, SUBGROUP, INVOCATION}:
        if not storage & ~_WORKGROUP_BITS:
            return "memoryBarrierShared"
        return "groupMemoryBarrier"
    if scope in {DEVICE, QUEUE_FAMILY, SHADER_CALL}:
        buffer = bool(storage & _BUFFER_BITS)
        image = bool(storage & IMAGE_MEMORY)
        other = bool(storage & ~(_BUFFER_BITS | IMAGE_MEMORY))
        if other:
            return "memoryBarrier"
        if buffer and image:
            return "deviceMemoryBarrier"
        return "memoryBarrierBuffer" if buffer else "memoryBarrierImage"
    raise SpirvBarrierUnsupportedError(
        f"{opcode} memory scope {_SCOPE_NAMES.get(scope, scope)} is wider than any CrossGL barrier "
        "(the widest is Device)"
    )


def _call(name: str):
    return FunctionCallNode(name, [])


def spirv_barrier_statements(opcode: str, operand_values, execution_models):
    """CrossGL statements for one ``OpControlBarrier`` or ``OpMemoryBarrier``.

    ``operand_values`` are the instruction's constant operand values in SPIR-V order and ``execution_models`` the
    module's entry point execution models.
    """
    if opcode == "OpMemoryBarrier":
        scope = _constant_value(operand_values[0], opcode, "Memory scope")
        semantics = _constant_value(operand_values[1], opcode, "Semantics")
        fence = _memory_fence(scope, semantics & _STORAGE_BITS, opcode)
        # A memory barrier without storage classes orders nothing (SPIR-V 1.6 section 3.25).
        return [_call(fence)] if fence is not None else []

    execution = _constant_value(operand_values[0], opcode, "Execution scope")
    scope = _constant_value(operand_values[1], opcode, "Memory scope")
    semantics = _constant_value(operand_values[2], opcode, "Semantics")
    if execution != WORKGROUP:
        raise SpirvBarrierUnsupportedError(
            f"OpControlBarrier execution scope {_SCOPE_NAMES.get(execution, execution)}: CrossGL control "
            "barriers synchronize a workgroup"
        )
    storage = semantics & _STORAGE_BITS
    tessellation_control = set(execution_models) == {"TessellationControl"}
    if tessellation_control and not storage & ~OUTPUT_MEMORY:
        # The patch barrier orders the patch's output writes (GLSL 4.60 section 8.16).
        return [_call("barrier")]
    if not storage:
        return [_call("workgroupExecutionBarrier")]
    if not storage & ~_WORKGROUP_BITS:
        return [_call("workgroupBarrier")]
    fence = _memory_fence(scope, storage, opcode)
    return [_call(fence), _call("workgroupBarrier"), _call(fence)]
