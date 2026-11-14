"""Translate a source with several entries of one stage to a target whose compilation unit holds one per stage.

A GLSL compilation unit has one ``main`` per stage, so the GLSL generators refuse a whole file with two kernels
(:class:`OpenGLRepeatedStageEntryError`). Tests that check what every entry of such a file becomes on every target
translate it for those targets one entry at a time, as ``crosstl.translate_entries`` does for callers.
"""

from __future__ import annotations

import crosstl
from crosstl.translator.codegen.glsl_stage_entries import OpenGLRepeatedStageEntryError


def translate_each_entry(path, backend, **options):
    """``{None: code}`` for a target that takes the whole file, else ``{(stage, entry): code}`` per entry.

    The refusal is checked to name every entry of the program before the entries are translated one at a time.
    """
    try:
        return {None: crosstl.translate(str(path), backend=backend, **options)}
    except OpenGLRepeatedStageEntryError as refusal:
        stage_entries = refusal.stage_entries
    names = [name for _stage, name in stage_entries]
    assert len(set(names)) == len(names), stage_entries
    format_output = options.pop("format_output", True)
    artifacts = crosstl.translate_entries(
        str(path), backend, names, format_output, **options
    )
    return {(stage, name): artifacts[name] for stage, name in stage_entries}


def joined_entries(path, backend, **options):
    """Every artifact :func:`translate_each_entry` returns, in entry order, as one text for content checks."""
    return "\n".join(translate_each_entry(path, backend, **options).values())
