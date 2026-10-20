"""Where each clip or cull distance value lands in the target's single distance array.

HLSL declares clip and cull distances as separate values, ``float2 a : SV_ClipDistance0`` and
``float b : SV_ClipDistance1``, each a float or a float vector of up to four components. GLSL, SPIR-V and Metal
have one float array per kind (``gl_ClipDistance``, ``BuiltIn ClipDistance``, ``[[clip_distance]]``). DXC packs
the HLSL values into that array in ascending semantic index, whatever order the structure declares them in, so
the example above writes elements 0 and 1 from ``a`` and element 2 from ``b`` of a three-element array.

CrossGL spells semantic index 0 ``gl_ClipDistance`` and index *N* ``gl_ClipDistanceN`` (likewise for
``gl_CullDistance``). A float array value with index 0 is the GLSL form: it fills the array from element 0.

:func:`distance_layout` assigns every value its first element and element count. The result does not depend on
declaration order, because two values of one kind may not share a semantic index; such a pair is an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "DISTANCE_BUILTINS",
    "DistanceLayout",
    "DistanceSlot",
    "distance_component_count",
    "distance_layout",
    "distance_semantic",
    "distance_semantic_spelling",
]

DISTANCE_BUILTINS = ("gl_ClipDistance", "gl_CullDistance")

_SEMANTIC = re.compile(r"(gl_ClipDistance|gl_CullDistance)(\d*)")
_VECTOR = re.compile(r"(?:vec|float)([234])")
_ARRAY = re.compile(r"(float)\s*\[\s*(\d+)\s*\]")


def distance_semantic(semantic) -> tuple[str, int] | None:
    """``(builtin, semantic index)`` for a CrossGL clip or cull distance semantic, or ``None``."""
    if semantic is None:
        return None
    match = _SEMANTIC.fullmatch(str(semantic).strip())
    if match is None:
        return None
    return match.group(1), int(match.group(2) or 0)


def distance_semantic_spelling(builtin: str, index: int) -> str:
    """The CrossGL semantic for ``builtin`` at semantic ``index``: index 0 carries no suffix."""
    return builtin if index == 0 else f"{builtin}{index}"


def distance_component_count(type_text, array_size=None) -> tuple[str, int] | None:
    """``(shape, element count)`` of a distance value's type, or ``None`` when it is not float-based.

    ``shape`` is ``"scalar"``, ``"vector"`` or ``"array"``. ``array_size`` is the declared size of an array
    value whose ``type_text`` names only the element type.
    """
    text = re.sub(r"\s+", "", str(type_text or ""))
    if array_size is not None:
        try:
            size = int(str(array_size))
        except ValueError:
            return None
        return ("array", size) if text == "float" and size > 0 else None
    if text == "float":
        return "scalar", 1
    vector = _VECTOR.fullmatch(text)
    if vector is not None:
        return "vector", int(vector.group(1))
    array = _ARRAY.fullmatch(text)
    if array is not None and int(array.group(2)) > 0:
        return "array", int(array.group(2))
    return None


@dataclass(frozen=True)
class DistanceSlot:
    """One distance value: ``count`` elements of ``builtin`` starting at ``offset``."""

    name: str
    builtin: str
    semantic_index: int
    shape: str
    offset: int
    count: int


@dataclass(frozen=True)
class DistanceLayout:
    """The distance values of one interface (a structure or an entry point's parameters)."""

    slots: tuple[DistanceSlot, ...]

    def slot(self, name) -> DistanceSlot | None:
        for slot in self.slots:
            if slot.name == name:
                return slot
        return None

    def size(self, builtin: str) -> int:
        return sum(slot.count for slot in self.slots if slot.builtin == builtin)

    def builtins(self) -> tuple[str, ...]:
        return tuple(
            b for b in DISTANCE_BUILTINS if any(s.builtin == b for s in self.slots)
        )


def distance_layout(
    entries: Iterable[tuple[str, object, object, object]],
) -> DistanceLayout:
    """Lay out ``(name, semantic, type_text, array_size)`` entries; non-distance entries are skipped.

    Raises ``ValueError`` for a distance value that is not a float, float vector or float array, and for two
    values of one kind that share a semantic index.
    """
    by_builtin = {}
    for name, semantic, type_text, array_size in entries:
        parsed = distance_semantic(semantic)
        if parsed is None:
            continue
        builtin, index = parsed
        shape_count = distance_component_count(type_text, array_size)
        if shape_count is None:
            raise ValueError(
                f"{builtin} value '{name}' of type '{type_text}': a clip or cull distance is a float, "
                "a float vector or a float array"
            )
        values = by_builtin.setdefault(builtin, [])
        for other_name, other_index, _shape, _count in values:
            if other_index == index:
                raise ValueError(
                    f"{builtin} values '{other_name}' and '{name}' share semantic index {index}"
                )
        values.append((name, index, *shape_count))
    slots = []
    for builtin in DISTANCE_BUILTINS:
        offset = 0
        for name, index, shape, count in sorted(
            by_builtin.get(builtin, ()), key=lambda value: value[1]
        ):
            slots.append(DistanceSlot(name, builtin, index, shape, offset, count))
            offset += count
    return DistanceLayout(tuple(slots))
