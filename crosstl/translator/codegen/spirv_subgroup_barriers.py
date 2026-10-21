"""SPIR-V lowering of the CrossGL subgroup barriers.

CrossGL takes its subgroup barriers from GL_KHR_shader_subgroup: ``subgroupBarrier`` is a subgroup control barrier
that orders buffer, image and shared memory, and ``subgroupMemoryBarrier``, ``subgroupMemoryBarrierBuffer``,
``subgroupMemoryBarrierShared`` and ``subgroupMemoryBarrierImage`` are subgroup-scope fences over all, buffer,
shared and image memory. ``subgroupExecutionBarrier`` is the control barrier without memory semantics, as Metal's
``simdgroup_barrier(mem_none)`` is. Each becomes the instruction glslang emits for it: ``OpControlBarrier Subgroup
Subgroup`` or ``OpMemoryBarrier Subgroup`` with acquire-release semantics over the named storage classes. The
barriers require the ``GroupNonUniform`` capability, as the GL_KHR_shader_subgroup_basic functions do.
"""

from __future__ import annotations

__all__ = ["SPIRV_SUBGROUP_BARRIERS", "emit_spirv_subgroup_barrier"]

# CrossGL name -> (control barrier, storage classes ordered).
SPIRV_SUBGROUP_BARRIERS = {
    "subgroupExecutionBarrier": (True, ()),
    "subgroupBarrier": (True, ("UniformMemory", "WorkgroupMemory", "ImageMemory")),
    "subgroupMemoryBarrier": (
        False,
        ("UniformMemory", "WorkgroupMemory", "ImageMemory"),
    ),
    "subgroupMemoryBarrierBuffer": (False, ("UniformMemory",)),
    "subgroupMemoryBarrierShared": (False, ("WorkgroupMemory",)),
    "subgroupMemoryBarrierImage": (False, ("ImageMemory",)),
}


def emit_spirv_subgroup_barrier(codegen, function_name: str, args) -> bool:
    """Emit the CrossGL subgroup barrier ``function_name`` through ``codegen``; ``False`` if it is not one."""
    entry = SPIRV_SUBGROUP_BARRIERS.get(function_name)
    if entry is None:
        return False
    if args:
        codegen.refuse_lowering(
            f"synchronization builtin '{function_name}' requires 0 operands"
        )
        return True
    control, storage = entry
    scope = codegen.subgroup_scope_id()
    if control:
        semantics = (
            codegen.spirv_memory_semantics_constant("AcquireRelease", *storage)
            if storage
            else codegen.spirv_memory_semantics_constant()
        )
        codegen.emit(f"OpControlBarrier %{scope.id} %{scope.id} %{semantics.id}")
    elif storage:
        semantics = codegen.spirv_memory_semantics_constant("AcquireRelease", *storage)
        codegen.emit(f"OpMemoryBarrier %{scope.id} %{semantics.id}")
    return True
