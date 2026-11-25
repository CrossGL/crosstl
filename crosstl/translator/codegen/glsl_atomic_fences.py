"""The GLSL memory barrier that carries a CrossGL ``atomicThreadFence``.

CrossGL's ``atomicThreadFence(flags, order, scope)`` is Metal's ``atomic_thread_fence``: it orders the calling
thread's accesses to the memory ``flags`` names (``mem_device``, ``mem_threadgroup``, ``mem_texture``,
``mem_threadgroup_imageblock``, ``mem_object_data`` or ``mem_none``) with ``order`` for the threads of ``scope``. The
Metal importer also writes one before and after each atomic function that is sequentially consistent over all memory
(MSL section 6.15.4), so a Metal ``atomic_fetch_add`` reaches CrossGL as a relaxed ``atomicAdd`` between two
``atomicThreadFence(mem_device | mem_threadgroup | mem_threadgroup_imageblock | mem_texture | mem_object_data,
memory_order_seq_cst, thread_scope_device)``.

GLSL 4.60 section 8.17 (and GLSL ES 3.10, the same functions) has one barrier per kind of memory: ``memoryBarrierBuffer``,
``memoryBarrierImage``, ``memoryBarrierShared`` and ``memoryBarrierAtomicCounter``, ``memoryBarrier`` for all of them,
and ``groupMemoryBarrier`` for all of them as seen by the invocations of the workgroup. Each one completes before the
invocation continues, and makes the invocation's earlier stores to coherent memory of its kinds visible to every later
access by the other invocations of its scope. That is at least the ordering of an acquire, release, acquire-release
or sequentially consistent fence, and a relaxed fence orders nothing, so the fence becomes the narrowest barrier whose
kinds and scope cover the fence's:

* ``mem_device`` is buffer memory (with atomic counter memory when the program declares atomic counters), and
  ``mem_texture`` image memory.
* ``mem_threadgroup`` is shared memory, which only compute, mesh and task shaders have; elsewhere it names no memory.
  ``mem_threadgroup_imageblock`` names no memory GLSL has: a program with imageblocks does not reach GLSL output.
  ``mem_object_data`` is the task payload of a mesh pipeline, which no GLSL barrier is specified to order, so a fence on
  it in a mesh or task shader is refused; other stages have no object data.
* ``thread_scope_thread``, ``thread_scope_simdgroup`` and ``thread_scope_threadgroup`` are covered by the workgroup,
  and ``thread_scope_device`` by the device-wide barriers. GLSL has no barrier toward the host or other devices, so
  ``thread_scope_system`` is refused.

``memoryBarrierShared`` and ``groupMemoryBarrier`` exist in compute, mesh and task shaders only; a stage without
workgroups uses the device-wide barriers, which are stronger, and a workgroup-scope fence there that names only
shared memory is refused, because the stage has none to order. A fence in a helper function that stages of different
kinds call is placed as in a stage without workgroups.
"""

from __future__ import annotations

__all__ = [
    "WORKGROUP_STAGES",
    "glsl_atomic_fence_barrier",
]

# Stages whose invocations form workgroups with shared memory (GLSL 4.60 section 4.3.8; GL_EXT_mesh_shader).
WORKGROUP_STAGES = frozenset({"compute", "mesh", "amplification"})

_WORKGROUP_SCOPES = frozenset(
    {"thread_scope_thread", "thread_scope_simdgroup", "thread_scope_threadgroup"}
)
_SHARED_FLAGS = frozenset({"mem_threadgroup"})
_SINGLE_KIND_BARRIERS = {
    frozenset({"buffer"}): "memoryBarrierBuffer()",
    frozenset({"image"}): "memoryBarrierImage()",
}


class GLSLAtomicFenceRefusal(Exception):
    """The fence has no GLSL barrier; ``reason`` names why."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _memory_kinds(flags, has_workgroups, has_atomic_counters):
    kinds = set()
    if "mem_device" in flags:
        kinds.add("buffer")
        if has_atomic_counters:
            kinds.add("atomic_counter")
    if "mem_texture" in flags:
        kinds.add("image")
    if "mem_threadgroup" in flags and has_workgroups:
        kinds.add("shared")
    return frozenset(kinds)


def glsl_atomic_fence_barrier(
    flags, order, scope, stage, *, has_atomic_counters=False
) -> str:
    """The GLSL barrier call (without ``;``) that carries ``atomicThreadFence(flags, order, scope)`` in ``stage``.

    ``flags`` is the set of memory flag names, ``order`` and ``scope`` the order and scope names, all already known
    to CrossGL, and ``stage`` the canonical name of the stage the fence runs in, or ``None`` when it is not one stage.
    Raises :class:`GLSLAtomicFenceRefusal` when GLSL has no barrier for the fence.
    """
    del order  # Every GLSL barrier orders at least as strongly as any fence order (module docstring).
    if scope == "thread_scope_system":
        raise GLSLAtomicFenceRefusal("unsupported-system-thread-scope")
    has_workgroups = stage in WORKGROUP_STAGES
    if "mem_object_data" in flags and stage in {"mesh", "amplification"}:
        raise GLSLAtomicFenceRefusal("unsupported-object-data-memory")
    kinds = _memory_kinds(flags, has_workgroups, has_atomic_counters)
    if has_workgroups:
        if kinds <= {"shared"}:
            return "memoryBarrierShared()"
        if scope in _WORKGROUP_SCOPES:
            return "groupMemoryBarrier()"
    elif scope in _WORKGROUP_SCOPES and not kinds and flags & _SHARED_FLAGS:
        raise GLSLAtomicFenceRefusal("unsupported-shader-stage")
    return _SINGLE_KIND_BARRIERS.get(kinds - {"shared"}, "memoryBarrier()")
