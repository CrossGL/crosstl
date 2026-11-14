"""SPIR-V built-in variables that only the ray tracing stages declare (SPV_KHR_ray_tracing, GLSL_EXT_ray_tracing).

:meth:`VulkanSPIRVCodeGen.stage_builtin_info` maps a GLSL built-in name to ``(type, BuiltIn, storage class,
execution models)``. The ray tracing names below were missing, so a GLSL or DXR (through
:mod:`..semantics.dxr_ray_interfaces`) ray shader reading them was refused as an unknown variable. Two names mean
another built-in in a ray stage than in a graphics stage: GLSL_EXT_ray_tracing section 7.1 gives the hit stages
``gl_PrimitiveID`` (SPIR-V ``PrimitiveId``) and ``gl_InstanceID``, the index of the instance in the top-level
acceleration structure (SPIR-V ``InstanceId``, not the vertex stage's ``InstanceIndex``); both are ``int`` there.

``gl_ObjectToWorldEXT`` and ``gl_WorldToObjectEXT`` are ``mat4x3``, four columns of three components, the type of
the ``ObjectToWorldKHR`` and ``WorldToObjectKHR`` built-ins. ``gl_ObjectToWorld3x4EXT`` and ``gl_WorldToObject3x4EXT``
are their transposes; SPIR-V has no built-in for them, so :func:`transposed_ray_builtin` names the built-in an
identifier transposes and the generator loads it and emits ``OpTranspose``.

The ray flag and hit kind constants (GLSL_EXT_ray_tracing section 7.2, GL_EXT_ray_flags_primitive_culling) are
``uint`` constants with the values of SPIR-V's ``RayFlags`` operand bits and the triangle hit kinds.
"""

from __future__ import annotations

from ..ray_query_vocabulary import ray_query_constant

__all__ = [
    "RAY_CONSTANTS",
    "RAY_EXECUTION_MODELS",
    "ray_builtin_info",
    "ray_constant_value",
    "transposed_ray_builtin",
]

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
_HIT = frozenset({"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR"})
_TRAVERSAL = _HIT | {"MissKHR"}

# Built-ins whose name means another variable in the graphics stages.
_RAY_STAGE_MEANINGS = {
    "gl_PrimitiveID": ("int", "PrimitiveId", "Input", _HIT),
    "gl_InstanceID": ("int", "InstanceId", "Input", _HIT),
}
_RAY_BUILTINS = {
    "gl_RayTmaxEXT": ("float", "RayTmaxKHR", "Input", _TRAVERSAL),
    "gl_IncomingRayFlagsEXT": ("uint", "IncomingRayFlagsKHR", "Input", _TRAVERSAL),
    "gl_ObjectToWorldEXT": ("mat4x3", "ObjectToWorldKHR", "Input", _HIT),
    "gl_WorldToObjectEXT": ("mat4x3", "WorldToObjectKHR", "Input", _HIT),
}
_TRANSPOSED = {
    "gl_ObjectToWorld3x4EXT": "gl_ObjectToWorldEXT",
    "gl_WorldToObject3x4EXT": "gl_WorldToObjectEXT",
}


def ray_builtin_info(codegen, name):
    """``stage_builtin_info`` entry for a ray tracing built-in, or ``None`` for other names."""
    name = str(name)
    if name in _RAY_BUILTINS:
        return _RAY_BUILTINS[name]
    if name in _RAY_STAGE_MEANINGS:
        model = codegen.current_stage_builtin_execution_model()
        if model in RAY_EXECUTION_MODELS:
            return _RAY_STAGE_MEANINGS[name]
    return None


def transposed_ray_builtin(name):
    """The built-in whose transpose the identifier ``name`` reads (``gl_ObjectToWorld3x4EXT``), or ``None``."""
    return _TRANSPOSED.get(str(name))


# GLSL_EXT_ray_tracing section 7.2 and GL_EXT_ray_flags_primitive_culling; SPIR-V "Ray Flags" and hit kinds.
RAY_CONSTANTS = {
    "gl_RayFlagsNoneEXT": 0x0,
    "gl_RayFlagsOpaqueEXT": 0x1,
    "gl_RayFlagsNoOpaqueEXT": 0x2,
    "gl_RayFlagsTerminateOnFirstHitEXT": 0x4,
    "gl_RayFlagsSkipClosestHitShaderEXT": 0x8,
    "gl_RayFlagsCullBackFacingTrianglesEXT": 0x10,
    "gl_RayFlagsCullFrontFacingTrianglesEXT": 0x20,
    "gl_RayFlagsCullOpaqueEXT": 0x40,
    "gl_RayFlagsCullNoOpaqueEXT": 0x80,
    "gl_RayFlagsSkipTrianglesEXT": 0x100,
    "gl_RayFlagsSkipAABBEXT": 0x200,
    "gl_HitKindFrontFacingTriangleEXT": 0xFE,
    "gl_HitKindBackFacingTriangleEXT": 0xFF,
}


def ray_constant_value(name):
    """The ``uint`` value of a ray flag, intersection kind or hit kind constant, or ``None``.

    Both the GLSL and the DXR spellings are accepted.
    """
    value = RAY_CONSTANTS.get(str(name))
    if value is not None:
        return value
    constant = ray_query_constant(name)
    return None if constant is None else constant.value
