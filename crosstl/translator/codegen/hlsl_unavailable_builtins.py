"""GLSL built-in inputs with no HLSL counterpart.

``gl_PointCoord`` is the fragment's position inside a point sprite. Direct3D 10
and later rasterize points as single pixels and have no point sprites, so HLSL
has no system value for it (the D3D10+ system-value semantics list no point
coordinate). Emitting the GLSL name left an undeclared identifier, which DXC
rejects. Its use is now refused.
"""

from __future__ import annotations

from ..ast import IdentifierNode, VariableNode

__all__ = ("HLSL_UNAVAILABLE_BUILTINS", "hlsl_unavailable_builtin")

HLSL_UNAVAILABLE_BUILTINS = {
    "gl_PointCoord": (
        "gl_PointCoord: Direct3D 10 and later have no point sprites, so HLSL has no "
        "point coordinate input"
    ),
}


def hlsl_unavailable_builtin(codegen, expr) -> str | None:
    """Record the refusal of a GLSL-only built-in and return its placeholder, or ``None``."""
    if isinstance(expr, str):
        name = expr
    elif isinstance(expr, (IdentifierNode, VariableNode)):
        name = getattr(expr, "name", None)
    else:
        return None
    reason = HLSL_UNAVAILABLE_BUILTINS.get(name)
    if reason is None:
        return None
    for table in ("local_variable_types", "global_variable_types"):
        if name in (getattr(codegen, table, None) or {}):
            return None
    return codegen.refuse_lowering(reason, "float2(0.0, 0.0)")
