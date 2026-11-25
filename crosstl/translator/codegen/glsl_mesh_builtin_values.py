"""Values a GLSL mesh stage writes to the ``int`` per-primitive built-ins.

GL_EXT_mesh_shader declares ``gl_PrimitiveID``, ``gl_Layer`` and ``gl_ViewportIndex`` of ``gl_MeshPerPrimitiveEXT``
as ``int``, and GLSL converts no ``uint`` to ``int`` implicitly (GLSL 4.60 section 4.1.10). HLSL
(``SV_PrimitiveID``, ``SV_RenderTargetArrayIndex``, ``SV_ViewportArrayIndex``) and MSL (``[[primitive_id]]``,
``[[render_target_array_index]]``, ``[[viewport_array_index]]``) declare these members ``uint``, so a value of the
member's type is written through ``int(...)``. The conversion keeps the bits, which is the value the source writes
for every index the pipeline can use.
"""

from __future__ import annotations

import re

__all__ = ["glsl_mesh_builtin_value"]

_INT_PRIMITIVE_BUILTIN = re.compile(
    r"^gl_MeshPrimitivesEXT\[.*\]\.(?:gl_PrimitiveID|gl_Layer|gl_ViewportIndex)$",
    re.S,
)
_INT_LITERAL = re.compile(r"^-?\d+$")


def glsl_mesh_builtin_value(codegen, target: str, value: str, value_type) -> str:
    """``value`` as the ``int`` the built-in ``target`` holds, when ``value_type`` is another type."""
    if not _INT_PRIMITIVE_BUILTIN.match(target) or _INT_LITERAL.match(value):
        return value
    mapped = codegen.map_type(value_type) if value_type is not None else None
    if mapped == "int":
        return value
    return f"int({value})"
