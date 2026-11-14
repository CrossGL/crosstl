"""Ray flags of a ``TraceRay`` call, applied to the Metal ``intersector<>`` that traces the ray.

DXR's ``TraceRay`` and GLSL's ``traceRayEXT`` take a ray flags word whose bits select opacity overrides, culling and
early termination (DXR ``RAY_FLAG`` enumeration; ``GL_EXT_ray_tracing`` section 8.19 defines ``gl_RayFlags*EXT``
with the same values). Metal has no flags word: the intersector carries the same controls as setters (Metal Shading
Language Specification, section 6.18.2 "Intersector"). The Metal generator lowered ``TraceRay`` to
``intersector<>::intersect`` but dropped the flags, so a trace that asked for back-face culling, forced opacity or
first-hit termination ran with none of them, and the intersection functions it invoked saw other candidates.

:func:`metal_trace_ray_flag_statements` returns the setter calls for one trace:

=========================================  ====================================================
Ray flag (value)                           Metal intersector setter
=========================================  ====================================================
``FORCE_OPAQUE`` (0x1)                     ``force_opacity(forced_opacity::opaque)``
``FORCE_NON_OPAQUE`` (0x2)                 ``force_opacity(forced_opacity::non_opaque)``
``ACCEPT_FIRST_HIT_AND_END_SEARCH`` (0x4)  ``accept_any_intersection(true)``
``SKIP_CLOSEST_HIT_SHADER`` (0x8)          none: ``intersect`` never runs a closest-hit function
``CULL_BACK_FACING_TRIANGLES`` (0x10)      ``set_triangle_cull_mode(triangle_cull_mode::back)``
``CULL_FRONT_FACING_TRIANGLES`` (0x20)     ``set_triangle_cull_mode(triangle_cull_mode::front)``
``CULL_OPAQUE`` (0x40)                     ``set_opacity_cull_mode(opacity_cull_mode::opaque)``
``CULL_NON_OPAQUE`` (0x80)                 ``set_opacity_cull_mode(opacity_cull_mode::non_opaque)``
``SKIP_TRIANGLES`` (0x100)                 ``set_geometry_cull_mode(geometry_cull_mode::triangle)``
``SKIP_PROCEDURAL_PRIMITIVES`` (0x200)     ``set_geometry_cull_mode(geometry_cull_mode::bounding_box)``
``FORCE_OMM_2_STATE`` (0x400)              none: Metal acceleration structures have no opacity micromaps
=========================================  ====================================================

Front and back faces agree without a setter: DXR and Vulkan treat clockwise triangles as front-facing unless an
instance flag says otherwise, and the Metal intersector's default winding is ``winding::clockwise``.

A flags word known at translation time yields only the setters its bits need. Any other flags word is evaluated once
into a ``uint`` temporary, and every setter selects its mode from that value at run time.

A constant word that DXR and SPIR-V leave undefined is refused rather than given one of its possible meanings: more
than one of the four opacity bits (FORCE_OPAQUE, FORCE_NON_OPAQUE, CULL_OPAQUE, CULL_NON_OPAQUE), both face culling
bits, SKIP_TRIANGLES with SKIP_PROCEDURAL_PRIMITIVES or with a face culling bit, and bits outside the enumeration.
"""

from __future__ import annotations

from .array_utils import evaluate_literal_int_expression

__all__ = [
    "RAY_FLAG_CONSTANTS",
    "constant_ray_flags",
    "invalid_ray_flags_reason",
    "metal_trace_ray_flag_statements",
]

FORCE_OPAQUE = 0x01
FORCE_NON_OPAQUE = 0x02
ACCEPT_FIRST_HIT_AND_END_SEARCH = 0x04
SKIP_CLOSEST_HIT_SHADER = 0x08
CULL_BACK_FACING_TRIANGLES = 0x10
CULL_FRONT_FACING_TRIANGLES = 0x20
CULL_OPAQUE = 0x40
CULL_NON_OPAQUE = 0x80
SKIP_TRIANGLES = 0x100
SKIP_PROCEDURAL_PRIMITIVES = 0x200
FORCE_OMM_2_STATE = 0x400
_ALL_FLAGS = 0x7FF

# DXR spellings and the GLSL spellings of the same values (GL_EXT_ray_tracing 8.19, GL_EXT_opacity_micromap).
RAY_FLAG_CONSTANTS = {
    "RAY_FLAG_NONE": 0,
    "RAY_FLAG_FORCE_OPAQUE": FORCE_OPAQUE,
    "RAY_FLAG_FORCE_NON_OPAQUE": FORCE_NON_OPAQUE,
    "RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH": ACCEPT_FIRST_HIT_AND_END_SEARCH,
    "RAY_FLAG_SKIP_CLOSEST_HIT_SHADER": SKIP_CLOSEST_HIT_SHADER,
    "RAY_FLAG_CULL_BACK_FACING_TRIANGLES": CULL_BACK_FACING_TRIANGLES,
    "RAY_FLAG_CULL_FRONT_FACING_TRIANGLES": CULL_FRONT_FACING_TRIANGLES,
    "RAY_FLAG_CULL_OPAQUE": CULL_OPAQUE,
    "RAY_FLAG_CULL_NON_OPAQUE": CULL_NON_OPAQUE,
    "RAY_FLAG_SKIP_TRIANGLES": SKIP_TRIANGLES,
    "RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES": SKIP_PROCEDURAL_PRIMITIVES,
    "RAY_FLAG_FORCE_OMM_2_STATE": FORCE_OMM_2_STATE,
    "gl_RayFlagsNoneEXT": 0,
    "gl_RayFlagsOpaqueEXT": FORCE_OPAQUE,
    "gl_RayFlagsNoOpaqueEXT": FORCE_NON_OPAQUE,
    "gl_RayFlagsTerminateOnFirstHitEXT": ACCEPT_FIRST_HIT_AND_END_SEARCH,
    "gl_RayFlagsSkipClosestHitShaderEXT": SKIP_CLOSEST_HIT_SHADER,
    "gl_RayFlagsCullBackFacingTrianglesEXT": CULL_BACK_FACING_TRIANGLES,
    "gl_RayFlagsCullFrontFacingTrianglesEXT": CULL_FRONT_FACING_TRIANGLES,
    "gl_RayFlagsCullOpaqueEXT": CULL_OPAQUE,
    "gl_RayFlagsCullNoOpaqueEXT": CULL_NON_OPAQUE,
    "gl_RayFlagsSkipTrianglesEXT": SKIP_TRIANGLES,
    "gl_RayFlagsSkipAABBEXT": SKIP_PROCEDURAL_PRIMITIVES,
    "gl_RayFlagsForceOpacityMicromap2StateEXT": FORCE_OMM_2_STATE,
}

_OPACITY_BITS = (FORCE_OPAQUE, FORCE_NON_OPAQUE, CULL_OPAQUE, CULL_NON_OPAQUE)
_FACE_CULL_BITS = CULL_BACK_FACING_TRIANGLES | CULL_FRONT_FACING_TRIANGLES

# (setter, ((bit, mode), ...), no-bit mode): the first matching bit selects the mode.
_MODE_SETTERS = (
    (
        "force_opacity",
        (
            (FORCE_OPAQUE, "forced_opacity::opaque"),
            (FORCE_NON_OPAQUE, "forced_opacity::non_opaque"),
        ),
        "forced_opacity::none",
    ),
    (
        "set_opacity_cull_mode",
        (
            (CULL_OPAQUE, "opacity_cull_mode::opaque"),
            (CULL_NON_OPAQUE, "opacity_cull_mode::non_opaque"),
        ),
        "opacity_cull_mode::none",
    ),
    (
        "set_triangle_cull_mode",
        (
            (CULL_BACK_FACING_TRIANGLES, "triangle_cull_mode::back"),
            (CULL_FRONT_FACING_TRIANGLES, "triangle_cull_mode::front"),
        ),
        "triangle_cull_mode::none",
    ),
    (
        "set_geometry_cull_mode",
        (
            (SKIP_TRIANGLES, "geometry_cull_mode::triangle"),
            (SKIP_PROCEDURAL_PRIMITIVES, "geometry_cull_mode::bounding_box"),
        ),
        "geometry_cull_mode::none",
    ),
)


def constant_ray_flags(expr, constants=None):
    """The value of a ray flags argument known at translation time, or ``None``."""
    visible = dict(constants or {})
    visible.update(RAY_FLAG_CONSTANTS)
    value = evaluate_literal_int_expression(expr, visible)
    return None if value is None else int(value)


def invalid_ray_flags_reason(flags):
    """Why a constant flags word has no defined meaning in DXR or SPIR-V, or ``None`` when it is valid."""
    if flags < 0 or flags & ~_ALL_FLAGS:
        return f"ray flags {flags:#x} set bits outside the DXR and GL_EXT_ray_tracing enumerations"
    if sum(1 for bit in _OPACITY_BITS if flags & bit) > 1:
        return (
            f"ray flags {flags:#x} combine more than one of FORCE_OPAQUE, FORCE_NON_OPAQUE, CULL_OPAQUE and "
            "CULL_NON_OPAQUE, which DXR and SPIR-V leave undefined"
        )
    if flags & _FACE_CULL_BITS == _FACE_CULL_BITS:
        return (
            f"ray flags {flags:#x} cull both front-facing and back-facing triangles, which DXR and SPIR-V "
            "leave undefined"
        )
    if flags & SKIP_TRIANGLES and flags & (
        SKIP_PROCEDURAL_PRIMITIVES | _FACE_CULL_BITS
    ):
        return (
            f"ray flags {flags:#x} combine SKIP_TRIANGLES with SKIP_PROCEDURAL_PRIMITIVES or a face culling "
            "flag, which DXR and SPIR-V leave undefined"
        )
    return None


def _constant_statements(intersector, flags):
    statements = []
    for setter, choices, _none in _MODE_SETTERS:
        for bit, mode in choices:
            if flags & bit:
                statements.append(f"{intersector}.{setter}({mode})")
                break
    if flags & ACCEPT_FIRST_HIT_AND_END_SEARCH:
        statements.append(f"{intersector}.accept_any_intersection(true)")
    return statements


def _runtime_mode(flags_name, choices, none_mode):
    selected = none_mode
    for bit, mode in reversed(choices):
        selected = f"(({flags_name} & {bit}u) != 0u ? {mode} : {selected})"
    return selected


def _runtime_statements(intersector, flags_name):
    statements = [
        f"{intersector}.{setter}({_runtime_mode(flags_name, choices, none_mode)})"
        for setter, choices, none_mode in _MODE_SETTERS
    ]
    statements.append(
        f"{intersector}.accept_any_intersection(({flags_name} & "
        f"{ACCEPT_FIRST_HIT_AND_END_SEARCH}u) != 0u)"
    )
    return statements


def metal_trace_ray_flag_statements(
    intersector, flags_expr, rendered_flags, constants, new_temporary
):
    """Statements that apply a trace's ray flags to ``intersector``, as ``(statements, refusal reason)``.

    ``new_temporary`` returns a fresh local name; it is called only when the flags are not a translation-time
    constant, and the first statement then declares that name.
    """
    flags = constant_ray_flags(flags_expr, constants)
    if flags is not None:
        reason = invalid_ray_flags_reason(flags)
        if reason is not None:
            return [], reason
        return _constant_statements(intersector, flags), None
    flags_name = new_temporary()
    statements = [f"uint {flags_name} = uint({rendered_flags})"]
    statements.extend(_runtime_statements(intersector, flags_name))
    return statements, None
