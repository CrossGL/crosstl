"""Names that the GLSL emitters give to the resource parameters of entry points.

An entry point's buffer, texture and sampler parameters become GLSL globals: a texture becomes
``uniform sampler2D colorMap;`` and a buffer becomes an unnamed block whose member keeps the parameter's
name, ``buffer colorBuffer { uvec4 color[]; };``. Stage inputs and outputs are globals too, and GLSL gives
all of them, and the members of unnamed blocks, one namespace (GLSL 4.60 section 4.2.7). A varying that a
structure member names ``color`` therefore collides with a buffer parameter ``color``; glslang reports
``nameless block contains a member that already has a name at global scope``.

The varyings avoid every name returned here, which covers the entry points of all stages so that stages
translated one at a time name their shared varyings alike.
"""

from __future__ import annotations

from .stage_utils import function_stage_name

__all__ = ["glsl_entry_resource_parameter_names"]


def _entry_points(ast):
    for func in getattr(ast, "functions", None) or []:
        if function_stage_name(func):
            yield func
    stages = getattr(ast, "stages", None) or {}
    for _stage_type, stage in sorted(stages.items(), key=lambda item: str(item[0])):
        entry_point = getattr(stage, "entry_point", None)
        if entry_point is not None:
            yield entry_point


def glsl_entry_resource_parameter_names(codegen, ast) -> set[str]:
    """The source names of the resource parameters of every entry point in ``ast``."""
    names = set()
    for func in _entry_points(ast):
        parameters = getattr(func, "parameters", getattr(func, "params", None)) or []
        for parameter in parameters:
            name = getattr(parameter, "name", None)
            if isinstance(name, str) and codegen.is_stage_entry_resource_parameter(
                parameter
            ):
                names.add(name)
    return names
