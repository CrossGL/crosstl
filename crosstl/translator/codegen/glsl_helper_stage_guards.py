"""Stage guards for the helper functions of a combined GLSL program.

A program with several stages is emitted as one GLSL file whose stage-specific parts sit under the stage's macro
(``#ifdef GL_FRAGMENT_SHADER``), and the file is compiled once per stage. Helper functions used to be emitted
unguarded, so every stage compiled every helper. A helper that only a fragment entry point calls may use operations
GLSL allows in fragment shaders only: derivatives (``fwidth``), implicit-level-of-detail sampling with a clamp
(``textureClampARB``, GL_ARB_sparse_texture_clamp) or ``discard``. The vertex compile then rejected the file with
``no matching overloaded function found``, although no vertex entry point calls the helper.

Each helper is now guarded by the macros of the stages whose entry points reach it, when those are not all the
stages of the file. A helper that no entry point reaches, or that every stage reaches, stays unguarded. Prototypes
stay unguarded: an uncalled prototype without a definition is valid GLSL (GLSL 4.60 section 6.1).
"""

from __future__ import annotations

from .stage_utils import collect_stage_entry_records

__all__ = ["glsl_helper_stage_macros", "wrap_glsl_helper_stage_guard"]


def glsl_helper_stage_macros(codegen, ast) -> dict:
    """``{helper name: stage guard macros}`` for the helpers some, but not all, stages of ``ast`` reach."""
    if not getattr(codegen, "emit_stage_guards", False):
        return {}
    records = collect_stage_entry_records(
        ast, None, codegen.combined_stage_entry_types()
    )
    all_macros = {codegen.glsl_stage_guard_macro(stage) for _, stage, _ in records}
    if None in all_macros or len(all_macros) < 2:
        return {}
    reached = {}
    for _id, stage, function in records:
        macro = codegen.glsl_stage_guard_macro(stage)
        for name in codegen.reachable_function_names(ast, [function]):
            reached.setdefault(name, set()).add(macro)
    return {
        name: tuple(sorted(macros))
        for name, macros in reached.items()
        if macros != all_macros
    }


def wrap_glsl_helper_stage_guard(codegen, code, func, helper_macros) -> str:
    """``code`` for the helper ``func``, guarded by the macros of the stages that reach it."""
    macros = helper_macros.get(getattr(func, "name", None)) if code else None
    if not macros or getattr(codegen, "current_target_stage", None) is not None:
        return code
    if not code.endswith("\n"):
        code += "\n"
    if len(macros) == 1:
        return f"#ifdef {macros[0]}\n{code}#endif\n"
    condition = " || ".join(f"defined({macro})" for macro in macros)
    return f"#if {condition}\n{code}#endif\n"
