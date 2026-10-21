"""SPIR-V built-in variables of the GL_EXT_ray_tracing stages that the shared stage table does not describe.

GLSL_EXT_ray_tracing (section 7.1) gives the ray stages built-ins that keep a rasterisation stage's name with a
different meaning: ``gl_PrimitiveID`` and ``gl_InstanceID`` are the index of the hit primitive and of its instance in
the top-level acceleration structure. SPIR-V decorates them ``PrimitiveId`` and ``InstanceId`` there (the latter is
not the vertex stage's ``InstanceIndex``), and both are 32-bit integers (Vulkan 1.3 section 15.9). It also adds
``gl_RayTmaxEXT``, ``gl_IncomingRayFlagsEXT`` and the object/world transforms, which have no rasterisation
counterpart. :func:`ray_builtin_info` returns their ``(type, BuiltIn, storage class, execution models)`` entry in
the shape :meth:`VulkanSPIRVCodeGen.stage_builtin_info` uses, so the DXR intrinsics that
:mod:`..semantics.dxr_ray_interfaces` rewrites into these names reach SPIR-V as the same decorated inputs glslang
declares for them.
"""

from __future__ import annotations

__all__ = ["RAY_EXECUTION_MODELS", "ray_builtin_info"]

RAY_EXECUTION_MODELS = frozenset(
    {
        "RayGenerationKHR",
        "IntersectionKHR",
        "AnyHitKHR",
        "ClosestHitKHR",
        "MissKHR",
        "CallableKHR",
    }
)

# Stages that see a hit candidate or committed hit (GLSL_EXT_ray_tracing section 7.1).
_HIT_STAGES = frozenset({"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR"})
# Stages that see the incoming ray.
_RAY_STAGES = _HIT_STAGES | {"MissKHR"}

# Built-ins that only exist in ray stages.
_RAY_ONLY = {
    "gl_RayTmaxEXT": ("float", "RayTmaxKHR", "Input", _RAY_STAGES),
    "gl_IncomingRayFlagsEXT": ("uint", "IncomingRayFlagsKHR", "Input", _RAY_STAGES),
    "gl_ObjectToWorldEXT": ("mat4x3", "ObjectToWorldKHR", "Input", _HIT_STAGES),
    "gl_WorldToObjectEXT": ("mat4x3", "WorldToObjectKHR", "Input", _HIT_STAGES),
}
# Rasterisation built-ins whose ray stage meaning has a different decoration or stage set.
_RAY_MEANING = {
    "gl_PrimitiveID": ("int", "PrimitiveId", "Input", _HIT_STAGES),
    "gl_InstanceID": ("int", "InstanceId", "Input", _HIT_STAGES),
}


def ray_builtin_info(generator, name):
    """Stage built-in entry for ``name`` in the current ray stage, or ``None`` when the shared table applies."""
    info = _RAY_ONLY.get(name)
    if info is not None:
        return info
    info = _RAY_MEANING.get(name)
    if info is None:
        return None
    if generator.current_stage_builtin_execution_model() in RAY_EXECUTION_MODELS:
        return info
    return None
