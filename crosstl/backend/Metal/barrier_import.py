"""Metal ``threadgroup_barrier`` and ``simdgroup_barrier`` imported as CrossGL barrier intrinsics.

A Metal barrier is an execution barrier across the threadgroup or SIMD-group that also orders the memory named by
its ``mem_flags`` operand for the threads it synchronizes (Metal Shading Language 3.2 section 6.9.1). The importer
used to keep the memory part only for device and texture flags (``threadgroup_barrier(mem_device)`` became
``memoryBarrierBuffer()``, which GLSL, HLSL and SPIR-V emit as a fence without the execution barrier), to widen a
SIMD-group barrier to the whole threadgroup, and to print any other flag combination verbatim.

Each barrier now becomes the CrossGL statements whose synchronization covers it:

* ``threadgroup_barrier(mem_none)`` is ``workgroupExecutionBarrier()`` and ``threadgroup_barrier(mem_threadgroup)``
  ``workgroupBarrier()``.
* A threadgroup barrier that also orders device or texture memory is ``fence; barrier; fence``, where the fence is
  ``memoryBarrierBuffer``, ``memoryBarrierImage`` or ``deviceMemoryBarrier`` and the barrier ``workgroupBarrier``
  when ``mem_threadgroup`` is among the flags and ``workgroupExecutionBarrier`` otherwise. The fence before the
  barrier releases the writes made before it and the fence after acquires them, as a single SPIR-V
  ``OpControlBarrier`` with acquire-release semantics over those storage classes does. The Metal and HLSL
  emitters merge the three statements back into one barrier.
* ``simdgroup_barrier(mem_none)`` is ``subgroupExecutionBarrier()``; a SIMD-group barrier with memory flags is
  ``subgroupBarrier()``, the subgroup control barrier that orders buffer, image and shared memory
  (GL_KHR_shader_subgroup).

Imageblock and mesh object-data memory have no CrossGL barrier, so barriers naming them are refused, as are flags
that are not ``mem_flags`` constants.
"""

from __future__ import annotations

__all__ = ["metal_barrier_statements"]

_SUPPORTED_FLAGS = frozenset(
    {"mem_none", "mem_device", "mem_threadgroup", "mem_texture"}
)
_KNOWN_FLAGS = _SUPPORTED_FLAGS | {"mem_threadgroup_imageblock", "mem_object_data"}


def _device_fence(flags) -> str | None:
    device = "mem_device" in flags
    texture = "mem_texture" in flags
    if device and texture:
        return "deviceMemoryBarrier"
    if device:
        return "memoryBarrierBuffer"
    if texture:
        return "memoryBarrierImage"
    return None


def metal_barrier_statements(name: str, flags):
    """CrossGL barrier calls, in order, for one Metal barrier, or the reason it has no CrossGL form.

    ``name`` is ``threadgroup_barrier`` or ``simdgroup_barrier`` and ``flags`` the set of ``mem_flags`` names, or
    ``None`` when the operand is not a combination of ``mem_flags`` constants. Returns a tuple of zero-argument
    CrossGL function names, or a string explaining the refusal.
    """
    if not flags:
        return f"{name} requires a mem_flags constant operand"
    unknown = sorted(set(flags) - _KNOWN_FLAGS)
    if unknown:
        return f"{name} memory flag {', '.join(unknown)} is not a Metal mem_flags value"
    unsupported = sorted(set(flags) - _SUPPORTED_FLAGS)
    if unsupported:
        return (
            f"{name}({' | '.join(sorted(flags))}) orders {', '.join(unsupported)} memory, which no CrossGL "
            "barrier orders"
        )
    memory = set(flags) - {"mem_none"}
    if name == "simdgroup_barrier":
        return ("subgroupBarrier",) if memory else ("subgroupExecutionBarrier",)
    if not memory:
        return ("workgroupExecutionBarrier",)
    fence = _device_fence(memory)
    if fence is None:
        return ("workgroupBarrier",)
    # workgroupBarrier orders threadgroup memory as well; without mem_threadgroup the execution barrier alone
    # sits between the fences.
    control = (
        "workgroupBarrier"
        if "mem_threadgroup" in memory
        else "workgroupExecutionBarrier"
    )
    return (fence, control, fence)
