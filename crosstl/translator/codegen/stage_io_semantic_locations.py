"""The locations CrossTL gives user varyings and vertex inputs that carry no explicit location.

GLSL, SPIR-V and WGSL link stage outputs to the next stage's inputs by location, and the host binds vertex inputs by
location (Metal by ``[[attribute(n)]]``, which CrossTL keeps equal to the location). Every generator therefore gives a
stage-interface value the same location, so one host binding fits every target:

* An explicit ``location`` wins.
* The classic semantics take fixed locations (:func:`semantic_location`): ``POSITION`` 0, ``NORMAL`` 1, ``TANGENT`` 2,
  ``BINORMAL`` 3, ``TEXCOORD0``..``TEXCOORD7`` 5..12 and ``COLOR``/``COLOR0``..``COLOR7`` 13..20, in
  both the HLSL spelling and the CrossGL one the HLSL importer writes (``Normal``, ``TexCoord0``, ``Color0``). A
  semantic without an index is index 0, as in HLSL: ``TEXCOORD`` is ``TEXCOORD0`` and ``COLOR`` is ``COLOR0``.
* Any other user value of an interface (another semantic, or none) takes, in declaration order, the first run of
  free locations at or after the end of the previous one, skipping every fixed location of the interface
  (:func:`allocate_locations`). Built-in values take no location.

A fragment output's ``COLORn`` names colour attachment ``n`` instead, as ``SV_Targetn`` does.

GLSL links a varying whose semantic has no fixed location by name, so its generator leaves such a varying unlocated.
"""

from __future__ import annotations

from typing import Iterable, Sequence

__all__ = [
    "SEMANTIC_LOCATIONS",
    "allocate_locations",
    "semantic_location",
]


def _table():
    table = {
        "Position": 0,
        "POSITION": 0,
        "Normal": 1,
        "NORMAL": 1,
        "Tangent": 2,
        "TANGENT": 2,
        "Binormal": 3,
        "BINORMAL": 3,
        # HLSL reads a semantic without an index as index 0.
        "TexCoord": 5,
        "TEXCOORD": 5,
        "Color": 13,
        "COLOR": 13,
    }
    for spelling in list(table):
        if spelling not in ("TexCoord", "TEXCOORD", "Color", "COLOR"):
            table[f"{spelling}0"] = table[spelling]
    for index in range(8):
        table[f"TexCoord{index}"] = 5 + index
        table[f"TEXCOORD{index}"] = 5 + index
        table[f"Color{index}"] = 13 + index
        table[f"COLOR{index}"] = 13 + index
    return table


SEMANTIC_LOCATIONS = _table()


def semantic_location(semantic) -> int | None:
    """The fixed location of a user value with ``semantic``, or ``None`` when its interface allocates one."""
    if semantic is None:
        return None
    return SEMANTIC_LOCATIONS.get(str(semantic))


def allocate_locations(
    entries: Sequence[tuple[int | None, int]], occupied: Iterable[int] = ()
) -> list[int]:
    """The locations of an interface's user values, given as ``(fixed location or None, span)`` in declaration order.

    A value with a fixed location keeps it. The others take, in order, the first ``span`` consecutive locations at or
    after the end of the previous allocated value that neither a fixed value nor ``occupied`` covers.
    """
    used = set(occupied)
    for fixed, span in entries:
        if fixed is not None:
            used.update(range(fixed, fixed + max(span, 1)))
    locations = []
    next_location = 0
    for fixed, span in entries:
        if fixed is not None:
            locations.append(fixed)
            continue
        span = max(span, 1)
        while any(slot in used for slot in range(next_location, next_location + span)):
            next_location += 1
        locations.append(next_location)
        used.update(range(next_location, next_location + span))
        next_location += span
    return locations
