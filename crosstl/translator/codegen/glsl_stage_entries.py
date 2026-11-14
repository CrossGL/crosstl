"""At most one GLSL entry point per stage in one generated compilation unit.

A GLSL shader's entry point is the function ``main`` (GLSL 4.60 and GLSL ES 3.20 section 6.1), and a program object
links at most one executable per stage (OpenGL 4.6 section 7.3). A source program can hold several entries of one
stage, as a Metal, HLSL or CUDA file with two kernels does, and its whole-file GLSL translation has no faithful
form: the generator kept the first entry as ``main`` and emitted the others as functions that nothing calls, or,
for one requested stage, defined ``main`` once per entry. It now refuses such a program and names its entries.
Translating with ``entry_point`` set emits each entry as its own artifact, with its own bindings, and
``crosstl.translate_entries`` emits every entry's artifact from one reading of the source; the refusal names the
program's entries as ``stage_entries`` for it.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX
from ..refusals import OpenGLRefusal
from .stage_utils import collect_stage_entry_records

__all__ = ["OpenGLRepeatedStageEntryError", "refuse_repeated_stage_entries"]


class OpenGLRepeatedStageEntryError(OpenGLRefusal):
    """The program has more than one entry point of a stage that one GLSL compilation unit can hold once."""

    code = f"{CODE_PREFIX}opengl-repeated-stage-entry"
    missing_capabilities = ("opengl.entry-points-per-stage",)

    def __init__(self, stage, entry_points, stage_entries=()):
        self.stage = stage
        self.entry_points = tuple(entry_points)
        # Every (stage, entry point) of the program in source order: what a caller translates one at a time.
        self.stage_entries = tuple(stage_entries) or tuple(
            (stage, name) for name in self.entry_points
        )
        names = ", ".join(f"'{name}'" for name in self.entry_points)
        super().__init__(
            f"the program has {len(self.entry_points)} {stage} entry points ({names}), but a GLSL compilation "
            f"unit has one {stage} entry point, main (GLSL 4.60 section 6.1); translate each entry on its own "
            "with entry_point, or all of them at once with crosstl.translate_entries"
        )


def refuse_repeated_stage_entries(ast, target_stage, stage_entry_types) -> None:
    """Refuse ``ast`` when the generated unit would hold two entries of one stage."""
    by_stage: dict[str, list[str]] = {}
    stage_entries = []
    for _entry_id, stage_name, function in collect_stage_entry_records(
        ast, target_stage, stage_entry_types
    ):
        name = getattr(function, "name", None) or "<unnamed>"
        by_stage.setdefault(stage_name, []).append(name)
        stage_entries.append((stage_name, name))
    for stage_name, names in by_stage.items():
        if len(names) > 1:
            raise OpenGLRepeatedStageEntryError(stage_name, names, stage_entries)
