"""HLSL lowering of CrossGL ray queries that came from GLSL or SPIR-V.

CrossGL spells ray queries the DXR way. Imports from GLSL still carry three
GLSL shapes, and DXR has no direct spelling for any of them:

* the type ``rayQueryEXT``, or ``rayQuery`` with no flags. Both become
  ``RayQuery<RAY_FLAG_NONE>``, and GLSL flag names in ``RayQuery<...>`` become
  their DXR names;
* ``TraceRayInline(tlas, flags, mask, origin, tMin, direction, tMax)``, with
  the seven ``rayQueryInitializeEXT`` operands. Its four GLSL ray operands
  become one ``RayDesc``, built by a generated helper;
* the GLSL ray flag and intersection kind constants, which become the DXR
  names with the same values.
"""

from __future__ import annotations

from ..ast import FunctionCallNode, IdentifierNode, MemberAccessNode, RayQueryOpNode
from ..ray_query_vocabulary import hlsl_ray_query_constant, ray_query_type_flags

RAY_DESC_HELPER = "__crossgl_make_ray_desc"

__all__ = (
    "RAY_DESC_HELPER",
    "hlsl_ray_query_expression",
    "hlsl_ray_query_helpers",
    "hlsl_ray_query_type",
)


def hlsl_ray_query_type(type_name: str) -> str | None:
    """The DXR spelling of ray query type *type_name*, or ``None`` for other types.

    A ``RayQuery<...>`` whose flags are not ray flags is refused with
    ``ValueError`` rather than passed through.
    """
    text = "".join(str(type_name).split())
    if not (
        text in {"RayQuery", "rayQuery", "rayQueryEXT"} or text.startswith("RayQuery<")
    ):
        return None
    flags = ray_query_type_flags(text)
    if flags is None:
        raise ValueError(
            f"DirectX RayQuery type '{type_name}' has template flags that are not ray flags"
        )
    if not flags:
        return "RayQuery<RAY_FLAG_NONE>"
    return (
        "RayQuery<" + " | ".join(hlsl_ray_query_constant(f) or f for f in flags) + ">"
    )


def _constant_name(expr) -> str | None:
    if isinstance(expr, str):
        return expr
    if isinstance(expr, IdentifierNode):
        return getattr(expr, "name", None)
    return None


def _initialize_call(expr):
    """``(query, arguments)`` for a seven-operand ``TraceRayInline``, or ``None``."""
    if isinstance(expr, RayQueryOpNode):
        if getattr(expr, "operation", None) != "TraceRayInline":
            return None
        arguments = list(getattr(expr, "arguments", []) or [])
        return (expr.query_expr, arguments) if len(arguments) == 7 else None
    if not isinstance(expr, FunctionCallNode):
        return None
    callee = getattr(expr, "function", getattr(expr, "name", None))
    if not isinstance(callee, MemberAccessNode) or callee.member != "TraceRayInline":
        return None
    arguments = list(getattr(expr, "arguments", None) or [])
    return (callee.object, arguments) if len(arguments) == 7 else None


def _is_ray_query(codegen, query) -> bool:
    base_type, array_suffix = codegen.hlsl_expression_mapped_base_and_array_suffix(
        query
    )
    return (
        base_type is not None
        and not array_suffix
        and (base_type == "RayQuery" or base_type.startswith("RayQuery<"))
    )


def hlsl_ray_query_expression(codegen, expr) -> str | None:
    """HLSL text for a GLSL-shaped ray query construct, or ``None`` for other expressions."""
    name = _constant_name(expr)
    if name is not None and name.startswith("gl_"):
        spelled = hlsl_ray_query_constant(name)
        return spelled if spelled is not None and spelled != name else None

    initialize = _initialize_call(expr)
    if initialize is None:
        return None
    query, arguments = initialize
    if not _is_ray_query(codegen, query):
        return None
    tlas, flags, mask, origin, t_min, direction, t_max = (
        codegen.generate_expression(argument) for argument in arguments
    )
    codegen.hlsl_ray_desc_helper_needed = True
    ray = f"{RAY_DESC_HELPER}({origin}, {t_min}, {direction}, {t_max})"
    return (
        f"{codegen.generate_expression(query)}.TraceRayInline("
        f"{tlas}, {flags}, {mask}, {ray})"
    )


def hlsl_ray_query_helpers(codegen) -> str:
    """The generated ``RayDesc`` helper, when a seven-operand call needed it."""
    if not getattr(codegen, "hlsl_ray_desc_helper_needed", False):
        return ""
    codegen.hlsl_ray_desc_helper_needed = False
    return (
        f"RayDesc {RAY_DESC_HELPER}(float3 origin, float tMin, float3 direction, "
        "float tMax) {\n"
        "    RayDesc ray;\n"
        "    ray.Origin = origin;\n"
        "    ray.TMin = tMin;\n"
        "    ray.Direction = direction;\n"
        "    ray.TMax = tMax;\n"
        "    return ray;\n"
        "}\n\n"
    )
