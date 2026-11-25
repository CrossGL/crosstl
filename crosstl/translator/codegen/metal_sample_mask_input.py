"""GLSL's and SPIR-V's ``int`` coverage-mask input on Metal's ``uint [[sample_mask]]``.

GLSL declares the fragment input ``int gl_SampleMaskIn[]``, one bit per sample in 32-bit words (GLSL 4.60 section
7.1.5), and SPIR-V's ``SampleMask`` built-in is an array of 32-bit integers (SPIR-V 1.6 section 3.21), which the
SPIR-V importer writes as ``int gl_SampleMaskIn[1] @input @gl_SampleMaskIn``. Metal reads the same mask as one
``uint`` fragment input on ``[[sample_mask]]`` (Metal Shading Language Specification 3.2, section 5.2.3.4). The Metal
generator required a ``uint`` parameter and refused the ``int`` and ``int[1]`` forms, so no GLSL or SPIR-V fragment
shader that reads its coverage reached Metal.

:func:`metal_sample_mask_input_declaration` gives the body-local that stands for such a parameter: the entry point
receives ``uint`` on ``[[sample_mask]]`` under a fresh name, and the local of the declared type is initialised from
it, ``int mask[1] = {int(_crossglSampleMaskIn)};``. One word holds the whole mask, because Metal's mask is one 32-bit
``uint``; a declaration of more than one word is left to the existing refusal. The conversion keeps
every bit (two's complement).
"""

from __future__ import annotations

import re

__all__ = ["METAL_SAMPLE_MASK_INPUT_NAME", "metal_sample_mask_input_declaration"]

# The base of the name the ``uint [[sample_mask]]`` entry parameter takes.
METAL_SAMPLE_MASK_INPUT_NAME = "_crossglSampleMaskIn"

_ONE_WORD_ARRAY = re.compile(r"(?P<element>u?int)\s*\[\s*1\s*\]")


def metal_sample_mask_input_declaration(
    stage_name, metal_semantic, mapped_type, name, source
):
    """``name``'s declaration from the ``uint`` mask ``source``, or ``None`` when ``mapped_type`` needs none."""
    if stage_name != "fragment" or metal_semantic != "sample_mask":
        return None
    mapped_type = str(mapped_type or "").strip()
    if mapped_type == "int":
        return f"int {name} = int({source});"
    array = _ONE_WORD_ARRAY.fullmatch(mapped_type)
    if array is None:
        return None
    element = array.group("element")
    value = source if element == "uint" else f"int({source})"
    return f"{element} {name}[1] = {{{value}}};"
