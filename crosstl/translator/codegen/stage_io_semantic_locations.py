"""The locations CrossTL gives user varyings that carry an HLSL-style semantic but no explicit location.

GLSL and SPIR-V link stage outputs to the next stage's inputs by location, so every generator that declares one side
of a stage interface must give a semantic the location the other side uses.

* GLSL places the classic vertex semantics at fixed locations: ``POSITION`` 0, ``NORMAL`` 1, ``TANGENT`` 2,
  ``BINORMAL`` 3, ``TEXCOORD`` 4, ``TEXCOORD0``..``TEXCOORD7`` 5..12 and ``COLOR``/``COLOR0``..``COLOR7`` 13..20, in
  both the HLSL spelling and the CrossGL one the HLSL importer writes (``Normal``, ``TexCoord0``, ``Color0``). A
  varying with another semantic has no location and links by name.
* SPIR-V places the upper-case spellings at the same fixed locations, any ``TEXCOORDn`` at ``5 + n`` and any
  ``COLORn`` at ``13 + n``; a structure member with another semantic takes its running position in the structure.
"""

from __future__ import annotations

import re

__all__ = [
    "GLSL_SEMANTIC_LOCATIONS",
    "glsl_semantic_location",
    "spirv_semantic_location",
]


def _glsl_table():
    table = {
        "Position": 0,
        "POSITION": 0,
        "Normal": 1,
        "NORMAL": 1,
        "Tangent": 2,
        "TANGENT": 2,
        "Binormal": 3,
        "BINORMAL": 3,
        "TexCoord": 4,
        "TEXCOORD": 4,
        "Color": 13,
        "COLOR": 13,
    }
    for index in range(8):
        table[f"TexCoord{index}"] = 5 + index
        table[f"TEXCOORD{index}"] = 5 + index
        table[f"Color{index}"] = 13 + index
        table[f"COLOR{index}"] = 13 + index
    return table


GLSL_SEMANTIC_LOCATIONS = _glsl_table()

_SPIRV_FIXED_LOCATIONS = {
    "POSITION": 0,
    "NORMAL": 1,
    "TANGENT": 2,
    "BINORMAL": 3,
    "TEXCOORD": 4,
    "COLOR": 13,
}


def glsl_semantic_location(semantic) -> int | None:
    """The GLSL location of a user varying with ``semantic``, or ``None`` when it links by name."""
    if semantic is None:
        return None
    return GLSL_SEMANTIC_LOCATIONS.get(str(semantic))


def spirv_semantic_location(semantic) -> int | None:
    """The fixed SPIR-V location of ``semantic``, or ``None`` when the member's position decides it."""
    if semantic is None:
        return None
    text = str(semantic)
    if text in _SPIRV_FIXED_LOCATIONS:
        return _SPIRV_FIXED_LOCATIONS[text]
    match = re.fullmatch(r"TEXCOORD(\d+)", text)
    if match:
        return 5 + int(match.group(1))
    match = re.fullmatch(r"COLOR(\d+)", text)
    if match:
        return 13 + int(match.group(1))
    return None
