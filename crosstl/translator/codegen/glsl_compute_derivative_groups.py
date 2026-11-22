"""The derivative group a GLSL compute, mesh or task stage declares.

GL_KHR_compute_shader_derivatives, and GL_NV_compute_shader_derivatives before it, let these stages say how their
invocations form quads:

- ``layout(derivative_group_quadsKHR) in;`` takes 2x2 blocks of the x and y dimensions, and needs both dimensions
  to be even;
- ``layout(derivative_group_linearKHR) in;`` takes four consecutive local invocation indices, and needs the
  invocation count to be a multiple of four.

Quad operations and derivatives read those quads, so a quad swap in an 8x8 workgroup reads a different invocation
under each group. A CrossGL stage that declares a group (the GLSL frontend keeps the source's layout) keeps it in
GLSL output, with the extension that declares it. GLSL output used to write the extension directive the source
enabled but drop the layout, which turned 2x2 quads into the default quads of four consecutive subgroup
invocations.

Both extensions apply to GLSL 4.50 and later; glslang declares the compute derivative functions from 4.50 only.
"""

from __future__ import annotations

from .stage_utils import compute_local_size, normalize_stage_name, stage_matches

__all__ = [
    "DERIVATIVE_GROUP_EXTENSIONS",
    "compute_derivative_group_extension_lines",
    "compute_derivative_group_layout",
    "declared_derivative_group",
]

DERIVATIVE_GROUP_EXTENSIONS = {
    "derivative_group_quadsKHR": "GL_KHR_compute_shader_derivatives",
    "derivative_group_linearKHR": "GL_KHR_compute_shader_derivatives",
    "derivative_group_quadsNV": "GL_NV_compute_shader_derivatives",
    "derivative_group_linearNV": "GL_NV_compute_shader_derivatives",
}
_COMPUTE_LIKE_STAGES = frozenset({"compute", "mesh", "task", "amplification", "object"})


def declared_derivative_group(codegen, stage_layout_qualifiers):
    """The derivative group layout that the stage's ``layout(...) in;`` qualifiers name, or None."""
    groups = []
    for entry in codegen.glsl_stage_layout_entries(stage_layout_qualifiers, "in"):
        name = str(getattr(entry, "name", "") or "")
        if name not in DERIVATIVE_GROUP_EXTENSIONS:
            continue
        if getattr(entry, "arguments", None):
            codegen.refuse_lowering(f"GLSL layout {name} does not take arguments")
        if name not in groups:
            groups.append(name)
    if len(groups) > 1:
        codegen.refuse_lowering(
            "a GLSL stage declares one derivative group; this one declares "
            + " and ".join(groups)
        )
    return groups[0] if groups else None


def _literal_size(execution_config):
    try:
        return tuple(
            int(str(value).strip()) for value in compute_local_size(execution_config)
        )
    except (TypeError, ValueError):
        return None


def compute_derivative_group_layout(
    codegen, shader_type, stage_layout_qualifiers, execution_config
):
    """The stage's ``layout(derivative_group_*) in;`` line, or an empty string."""
    if shader_type not in _COMPUTE_LIKE_STAGES:
        return ""
    group = declared_derivative_group(codegen, stage_layout_qualifiers)
    if group is None:
        return ""
    size = _literal_size(execution_config)
    if size is not None:
        x, y, z = size
        if "quads" in group and (x % 2 or y % 2):
            codegen.refuse_lowering(
                f"GLSL {group} needs even local_size_x and local_size_y; the workgroup is {x}x{y}x{z}, "
                "and changing it would change the dispatch"
            )
        if "linear" in group and (x * y * z) % 4:
            codegen.refuse_lowering(
                f"GLSL {group} needs a multiple of four invocations; the workgroup is {x}x{y}x{z}, "
                "and changing it would change the dispatch"
            )
    return f"layout({group}) in;\n"


def compute_derivative_group_extension_lines(codegen, ast, target_stage=None):
    """The ``#extension`` lines of the derivative groups the emitted stages declare, unless the program enables them."""
    enabled = codegen.glsl_preprocessor_extensions(ast)
    lines = []
    for stage_type, stage in (getattr(ast, "stages", None) or {}).items():
        stage_name = normalize_stage_name(stage_type)
        if stage_name not in _COMPUTE_LIKE_STAGES or not stage_matches(
            target_stage, stage_name
        ):
            continue
        group = declared_derivative_group(
            codegen, getattr(stage, "layout_qualifiers", None)
        )
        if group is None:
            continue
        extension = DERIVATIVE_GROUP_EXTENSIONS[group]
        line = f"#extension {extension} : require"
        if extension not in enabled and line not in lines:
            lines.append(line)
    return lines
