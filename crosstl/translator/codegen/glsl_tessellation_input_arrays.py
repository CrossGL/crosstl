"""GLSL tessellation inputs leave their per-vertex dimension implicit.

GLSL 4.60 section 4.3.4 (Input Variables): the inputs of a tessellation control or evaluation shader that are
not ``patch`` are arrays indexed by vertex, and "if the array is declared with a size, the size must be
gl_MaxPatchVertices"; an implicitly sized one takes that size. SPIR-V and HLSL size the same arrays by the
patch's control-point count (DXC writes ``float4 in_var_POSITION[3]`` for an ``InputPatch<T, 3>``), which
glslang rejects with "tessellation input array size must be gl_MaxPatchVertices or implicitly sized".

The declaration therefore drops a written outer size. ``.length()`` of such an array would then count
gl_MaxPatchVertices elements instead of the written ones, so the codegen spells it as the written size.
"""

from typing import Optional

_TESSELLATION_STAGES = frozenset({"tessellation_control", "tessellation_evaluation"})
_MAX_PATCH_VERTICES = "gl_MaxPatchVertices"


def _outer_dimension_end(array_suffix: str) -> int:
    """The index just past the ``]`` closing the first dimension of ``array_suffix``, or -1."""
    depth = 0
    for index, char in enumerate(array_suffix):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def implicit_tessellation_input_type(
    codegen, node, qualifier: str, mapped_type: str, stage_name
) -> str:
    """``mapped_type`` with the per-vertex size of a tessellation input left implicit.

    The written size is recorded for ``tessellation_input_length``. Types that are not arrays, ``patch``
    inputs, outputs and the inputs of other stages come back unchanged, as does a size already spelled
    ``gl_MaxPatchVertices``.
    """
    if str(stage_name or "").lower() not in _TESSELLATION_STAGES:
        return mapped_type
    if codegen.global_stage_io_layout_direction(qualifier) != "input":
        return mapped_type
    if "patch" in str(qualifier).split():
        return mapped_type
    text = str(mapped_type)
    open_bracket = text.find("[")
    if open_bracket < 0:
        return mapped_type
    base_type, array_suffix = text[:open_bracket], text[open_bracket:]
    end = _outer_dimension_end(array_suffix)
    if end < 0:
        return mapped_type
    size = array_suffix[1 : end - 1].strip()
    if not size or size == _MAX_PATCH_VERTICES:
        return mapped_type
    name = codegen.resource_node_name(node, "")
    if name:
        codegen.glsl_tessellation_input_extents[name] = size
    return f"{base_type}[]{array_suffix[end:]}"


def tessellation_input_length(codegen, func_expr) -> Optional[str]:
    """The written size of a tessellation input array whose ``.length()`` ``func_expr`` reads, or ``None``."""
    extents = codegen.glsl_tessellation_input_extents
    if not extents:
        return None
    target = getattr(func_expr, "object", None)
    if type(target).__name__ != "IdentifierNode":
        return None
    name = getattr(target, "name", None)
    if name not in extents or name in (codegen.local_variable_types or {}):
        return None
    size = extents[name]
    return size if size.isdigit() else f"int({size})"
