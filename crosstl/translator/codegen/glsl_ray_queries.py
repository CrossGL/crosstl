"""GLSL lowering of CrossGL ray queries that came from HLSL.

CrossGL spells ray queries the DXR way. GLSL has no direct spelling for three
parts of that vocabulary:

* ``RayDesc`` is a DXR built-in type. GLSL gets an equivalent struct, and
  ``TraceRayInline`` passes its fields to ``rayQueryInitializeEXT``;
* the flags of ``RayQuery<FLAGS>`` belong to the query type in DXR. In GLSL
  they are combined into the flags operand of ``rayQueryInitializeEXT``,
  which is what DXR does with them (D3D12 raytracing spec, "RayQuery");
* the DXR ray flag and intersection kind constants become the GLSL built-in
  constants with the same values.
"""

from __future__ import annotations

from ..ast import IdentifierNode
from ..ray_query_vocabulary import glsl_ray_query_constant, ray_query_type_flags

RAY_DESC_DECLARATION = (
    "struct RayDesc {\n"
    "  vec3 Origin;\n"
    "  float TMin;\n"
    "  vec3 Direction;\n"
    "  float TMax;\n"
    "};\n"
)

_DXR_CONSTANT_PREFIXES = ("RAY_FLAG_", "COMMITTED_", "CANDIDATE_")
_TYPE_ATTRIBUTES = ("vtype", "var_type", "param_type", "const_type", "return_type")

__all__ = (
    "glsl_ray_desc_declaration",
    "glsl_ray_query_constant_expression",
    "glsl_ray_query_initialize_flags",
)


def glsl_ray_query_constant_expression(codegen, expr) -> str | None:
    """The GLSL constant for a DXR ray flag or intersection kind, or ``None``."""
    name = expr if isinstance(expr, str) else None
    if isinstance(expr, IdentifierNode):
        name = getattr(expr, "name", None)
    if not isinstance(name, str) or not name.startswith(_DXR_CONSTANT_PREFIXES):
        return None
    # A program's own declaration of the name wins over the built-in.
    for table in ("local_variable_types", "global_variable_types"):
        if name in (getattr(codegen, table, None) or {}):
            return None
    return glsl_ray_query_constant(name)


def glsl_ray_query_initialize_flags(codegen, query, flags_text: str) -> str:
    """*flags_text* combined with the flags of the query's ``RayQuery<...>`` type."""
    type_name = codegen.type_name_string(codegen.expression_result_type(query))
    type_flags = ray_query_type_flags(type_name) if type_name else ()
    if type_flags is None:
        raise ValueError(
            f"GLSL ray query type '{type_name}' has template flags that are not ray flags"
        )
    spelled = [glsl_ray_query_constant(flag) or flag for flag in type_flags]
    spelled = [flag for flag in spelled if flag not in {"0", "gl_RayFlagsNoneEXT"}]
    if not spelled:
        return flags_text
    return f"({flags_text}) | {' | '.join(spelled)}"


def _type_names(codegen, node):
    for attribute in _TYPE_ATTRIBUTES:
        value = getattr(node, attribute, None)
        if value is not None:
            yield codegen.type_name_string(value)


def glsl_ray_desc_declaration(codegen, ast, struct_nodes) -> str:
    """The ``RayDesc`` struct, when the program uses the type without declaring it."""
    if any(getattr(node, "name", None) == "RayDesc" for node in struct_nodes):
        return ""
    for node in codegen.walk_ast(ast):
        if any(name == "RayDesc" for name in _type_names(codegen, node)):
            return RAY_DESC_DECLARATION
    return ""
