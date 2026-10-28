"""``invariant`` built-in outputs in GLSL and GLSL ES.

A CrossGL output that carries a built-in semantic and ``@invariant``, such as ``vec4 position @ gl_Position
@ invariant`` in an entry's return structure, becomes the built-in itself in GLSL, so no declaration of it is
written and the qualifier used to be lost. GLSL 4.60 section 4.8.1 and GLSL ES 3.00 section 4.6.1 make an existing
built-in output invariant by redeclaring it at global scope: ``invariant gl_Position;``.

:func:`invariant_builtin_redeclarations` writes those redeclarations for every vertex-processing stage entry whose
return structure, return value or ``out`` parameters carry the qualifier on a built-in. A built-in the stage does not
declare as a plain variable, such as ``gl_Position`` of a tessellation control stage, which is written through
``gl_out[]``, has no global redeclaration form and is refused.
"""

from __future__ import annotations

from .interface_qualifiers import has_invariant_attribute

__all__ = [
    "has_invariant_attribute",
    "invariant_builtin_redeclaration",
    "invariant_builtin_redeclarations",
]

# Built-in outputs each stage declares as plain variables (GLSL 4.60 sections 7.1.1-7.1.5, GLSL ES 3.00 section
# 7.1); the tessellation control and mesh stages write theirs through ``gl_out[]`` and ``gl_MeshVerticesEXT[]``.
_VERTEX_PROCESSING_OUTPUTS = frozenset(
    {"gl_Position", "gl_PointSize", "gl_ClipDistance", "gl_CullDistance"}
)
_REDECLARABLE_BUILTINS = {
    "vertex": _VERTEX_PROCESSING_OUTPUTS,
    "tessellation_evaluation": _VERTEX_PROCESSING_OUTPUTS,
    "geometry": _VERTEX_PROCESSING_OUTPUTS,
    "fragment": frozenset({"gl_FragDepth"}),
}


def invariant_builtin_redeclaration(codegen, builtin, stage_name, owner) -> str:
    """``invariant <builtin>;`` for an invariant built-in output of ``stage_name``, or ``""`` after a refusal."""
    if builtin in _REDECLARABLE_BUILTINS.get(stage_name, ()):
        return f"invariant {builtin};\n"
    codegen.refuse_lowering(
        f"GLSL cannot make the built-in output '{builtin}' of '{owner}' invariant in the "
        f"{stage_name or 'unknown'} stage: the stage does not declare it as a variable that a global "
        "'invariant' redeclaration can name (GLSL 4.60 section 4.8.1)"
    )
    return ""


def _builtin_output_name(codegen, node, semantic=None):
    if semantic is None:
        semantic = codegen.semantic_from_node(node)
    mapped = codegen.map_semantic(semantic) if semantic else None
    if not mapped or not str(mapped).startswith("gl_"):
        return None
    return str(mapped)


def _entry_invariant_builtins(codegen, func):
    """``(builtin, owner)`` pairs of ``func``'s invariant built-in outputs, in declaration order."""
    found = []
    return_type = codegen.type_node_name(getattr(func, "return_type", None))
    struct = codegen.structs_by_name.get(return_type) if return_type else None
    if struct is not None:
        for member in getattr(struct, "members", None) or ():
            if not has_invariant_attribute(member):
                continue
            builtin = _builtin_output_name(
                codegen, member, codegen.stage_io_semantic_from_node(member)
            )
            if builtin is not None:
                found.append((builtin, f"{struct.name}.{member.name}"))
    elif has_invariant_attribute(func):
        builtin = _builtin_output_name(
            codegen, func, codegen.function_return_semantic(func)
        )
        if builtin is not None:
            found.append((builtin, f"{func.name} return value"))
    for param in getattr(func, "parameters", None) or ():
        qualifiers = {str(q).lower() for q in getattr(param, "qualifiers", None) or ()}
        if "out" not in qualifiers or not has_invariant_attribute(param):
            continue
        builtin = _builtin_output_name(codegen, param)
        if builtin is not None:
            found.append((builtin, f"{func.name} parameter {param.name}"))
    return found


def invariant_builtin_redeclarations(codegen, target_stage) -> str:
    """Global ``invariant`` redeclarations for the built-in outputs of the stage entries being generated."""
    code = ""
    entries = getattr(codegen, "stage_entry_functions_by_stage", None) or {}
    for stage_name in sorted(entries):
        if target_stage is not None and stage_name != target_stage:
            continue
        lines = []
        for builtin, owner in _entry_invariant_builtins(codegen, entries[stage_name]):
            line = invariant_builtin_redeclaration(codegen, builtin, stage_name, owner)
            if line and line not in lines:
                lines.append(line)
        code += codegen.wrap_stage_guard("".join(lines), stage_name)
    return code
