"""The shared vocabulary of inline ray queries.

CrossGL spells ray queries the DXR way: ``RayQuery<FLAGS>`` objects with
methods such as ``TraceRayInline``, ``Proceed`` and ``CommittedStatus``. The
GLSL importer also produces ``Initialize``, which takes the GLSL
``rayQueryInitializeEXT`` operands (origin, tMin, direction, tMax) in place of
a ``RayDesc``.

Ray flags and intersection kinds have the same values in DXR, GL_EXT_ray_query
and SPV_KHR_ray_query. Only their spellings differ, so each table row maps
one value to both spellings.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = (
    "RAY_QUERY_CONSTANTS",
    "canonical_ray_query_type",
    "RayQueryConstant",
    "glsl_ray_query_constant",
    "hlsl_ray_query_constant",
    "ray_flag_value",
    "ray_query_constant",
    "ray_query_type_flags",
)


@dataclass(frozen=True)
class RayQueryConstant:
    """One ray flag or intersection kind, with its DXR and GLSL spellings."""

    value: int
    hlsl: str
    glsl: str


# DXR (D3D12_RAY_FLAG_*, COMMITTED_STATUS, CANDIDATE_TYPE) against
# GLSL_EXT_ray_tracing / GLSL_EXT_ray_query section 7 built-in constants.
RAY_QUERY_CONSTANTS = (
    RayQueryConstant(0x0, "RAY_FLAG_NONE", "gl_RayFlagsNoneEXT"),
    RayQueryConstant(0x1, "RAY_FLAG_FORCE_OPAQUE", "gl_RayFlagsOpaqueEXT"),
    RayQueryConstant(0x2, "RAY_FLAG_FORCE_NON_OPAQUE", "gl_RayFlagsNoOpaqueEXT"),
    RayQueryConstant(
        0x4,
        "RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH",
        "gl_RayFlagsTerminateOnFirstHitEXT",
    ),
    RayQueryConstant(
        0x8, "RAY_FLAG_SKIP_CLOSEST_HIT_SHADER", "gl_RayFlagsSkipClosestHitShaderEXT"
    ),
    RayQueryConstant(
        0x10,
        "RAY_FLAG_CULL_BACK_FACING_TRIANGLES",
        "gl_RayFlagsCullBackFacingTrianglesEXT",
    ),
    RayQueryConstant(
        0x20,
        "RAY_FLAG_CULL_FRONT_FACING_TRIANGLES",
        "gl_RayFlagsCullFrontFacingTrianglesEXT",
    ),
    RayQueryConstant(0x40, "RAY_FLAG_CULL_OPAQUE", "gl_RayFlagsCullOpaqueEXT"),
    RayQueryConstant(0x80, "RAY_FLAG_CULL_NON_OPAQUE", "gl_RayFlagsCullNoOpaqueEXT"),
    RayQueryConstant(0x100, "RAY_FLAG_SKIP_TRIANGLES", "gl_RayFlagsSkipTrianglesEXT"),
    RayQueryConstant(
        0x200, "RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES", "gl_RayFlagsSkipAABBEXT"
    ),
    RayQueryConstant(0, "COMMITTED_NOTHING", "gl_RayQueryCommittedIntersectionNoneEXT"),
    RayQueryConstant(
        1, "COMMITTED_TRIANGLE_HIT", "gl_RayQueryCommittedIntersectionTriangleEXT"
    ),
    RayQueryConstant(
        2,
        "COMMITTED_PROCEDURAL_PRIMITIVE_HIT",
        "gl_RayQueryCommittedIntersectionGeneratedEXT",
    ),
    RayQueryConstant(
        0,
        "CANDIDATE_NON_OPAQUE_TRIANGLE",
        "gl_RayQueryCandidateIntersectionTriangleEXT",
    ),
    RayQueryConstant(
        1, "CANDIDATE_PROCEDURAL_PRIMITIVE", "gl_RayQueryCandidateIntersectionAABBEXT"
    ),
)

_BY_NAME = {
    name: constant
    for constant in RAY_QUERY_CONSTANTS
    for name in (constant.hlsl, constant.glsl)
}


def ray_query_constant(name) -> RayQueryConstant | None:
    """The ray flag or intersection kind spelled *name* in DXR or GLSL, or ``None``."""
    return _BY_NAME.get(str(name)) if name is not None else None


def hlsl_ray_query_constant(name) -> str | None:
    """The DXR spelling of the ray query constant *name*, or ``None``."""
    constant = ray_query_constant(name)
    return None if constant is None else constant.hlsl


def glsl_ray_query_constant(name) -> str | None:
    """The GLSL spelling of the ray query constant *name*, or ``None``."""
    constant = ray_query_constant(name)
    return None if constant is None else constant.glsl


def ray_query_type_flags(type_name) -> tuple[str, ...] | None:
    """The flag operands of ``RayQuery<A | B>``, ``()`` for none, ``None`` otherwise.

    Each operand is a DXR or GLSL ray flag name or an integer literal. Any
    other operand makes the result ``None``, so callers refuse the type instead
    of dropping a flag.
    """
    text = "".join(str(type_name).split())
    if text in {"RayQuery", "rayQuery", "rayQueryEXT"}:
        return ()
    if not (text.startswith("RayQuery<") and text.endswith(">")):
        return None
    # CrossGL renders a flag expression in parentheses: RayQuery<(A | B)>.
    inner = text[len("RayQuery<") : -1].replace("(", "").replace(")", "")
    operands = tuple(part for part in inner.split("|") if part)
    for operand in operands:
        if _flag_value(operand) is None:
            return None
    return operands


def ray_flag_value(operands) -> int | None:
    """The combined value of ray flag *operands*, or ``None`` if one is unknown."""
    value = 0
    for operand in operands:
        operand_value = _flag_value(operand)
        if operand_value is None:
            return None
        value |= operand_value
    return value


def _flag_value(operand) -> int | None:
    text = str(operand).strip()
    lowered = text.lower()
    try:
        return int(lowered.rstrip("u"), 0)
    except ValueError:
        pass
    constant = ray_query_constant(text)
    if constant is None or not text.upper().startswith(("RAY_FLAG_", "GL_RAYFLAGS")):
        return None
    return constant.value


def canonical_ray_query_type(type_name) -> str | None:
    """The CrossGL spelling ``RayQuery<FLAGS>`` of a ray query type, or ``None``.

    The flags keep their source spelling; an unrecognized flag operand is kept
    as written so the generators refuse it rather than lose it.
    """
    if not isinstance(type_name, str):
        return None
    text = " ".join(type_name.split())
    compact = text.replace(" ", "")
    if compact in {"RayQuery", "rayQuery", "rayQueryEXT"}:
        return "RayQuery<RAY_FLAG_NONE>"
    if not (compact.startswith("RayQuery<") and compact.endswith(">")):
        return None
    inner = compact[len("RayQuery<") : -1].replace("(", "").replace(")", "")
    operands = [part for part in inner.split("|") if part]
    if not operands:
        return "RayQuery<RAY_FLAG_NONE>"
    return "RayQuery<" + " | ".join(operands) + ">"
