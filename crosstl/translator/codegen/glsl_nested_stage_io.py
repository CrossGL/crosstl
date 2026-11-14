"""Structure-typed GLSL stage inputs and outputs.

GLSL allows a vertex output or a fragment input of structure type (``out Inner inner;``), which links with the
other stage's variable of the same name and type. The generator used to declare such variables before the
structures they name, so every shader that used one failed to compile; the declarations now follow the data
structures.

A field of such a structure is a user varying, so a built-in field cannot stay inside it: Metal nests the vertex and
primitive structures of a mesh pipeline's fragment ``[[stage_in]]``, whose vertex structure carries the
``[[position]]`` built-in. Declared whole, the vertex stage would write a user varying instead of ``gl_Position`` and
the fragment would read one instead of ``gl_FragCoord``. Until such members are flattened to their fields, they are
refused with :class:`GLSLNestedStageBuiltinError`.
"""

from __future__ import annotations

__all__ = ["GLSLNestedStageBuiltinError", "refuse_nested_stage_builtins"]


class GLSLNestedStageBuiltinError(ValueError):
    """A structure-typed GLSL stage variable holds a built-in field, which a user varying cannot carry."""

    project_diagnostic_code = "project.translate.glsl-nested-stage-builtin"
    missing_capabilities = ("glsl.stage-io.nested-builtin-flattening",)

    def __init__(self, stage, direction, member, path, builtin):
        self.path = path
        self.builtin = builtin
        super().__init__(
            f"GLSL {stage} {direction} '{member}' is a structure whose field '{path}' is the built-in {builtin}; "
            "a structure-typed varying cannot carry a built-in"
        )


def _builtin_path(codegen, struct_name, prefix, seen):
    struct = codegen.structs_by_name.get(struct_name)
    if struct is None or struct_name in seen:
        return None
    seen = seen | {struct_name}
    for field in getattr(struct, "members", []) or []:
        path = f"{prefix}.{field.name}"
        mapped = str(
            codegen.map_semantic(codegen.stage_io_semantic_from_node(field)) or ""
        )
        if mapped.startswith("gl_"):
            return path, mapped
        nested = _builtin_path(codegen, codegen.member_type_name(field), path, seen)
        if nested is not None:
            return nested
    return None


def refuse_nested_stage_builtins(codegen, member, stage, direction):
    """Raise :class:`GLSLNestedStageBuiltinError` when a structure-typed stage member holds a built-in field."""
    found = _builtin_path(
        codegen, codegen.member_type_name(member), member.name, frozenset()
    )
    if found is not None:
        raise GLSLNestedStageBuiltinError(stage, direction, member.name, *found)
