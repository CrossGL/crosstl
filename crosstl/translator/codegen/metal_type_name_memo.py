"""Memoized Metal spellings of bare type names.

``MetalCodeGen.map_type`` runs millions of times on large programs, and nearly every call asks for one of a few
bare names (``float4``, ``uint``, a struct name). Each uncached call walks about twenty checks before it reaches
the type table, so the walk dominates expression typing.

An entry is reused only while everything the mapping read still holds, and a hit repeats everything the mapping
did to the generator:

- Reads. The spelling of a bare name depends on the type table entry of the name, on whether the program
  declares the name as an alias, enumeration or generic type, on the program's declared type names and on
  whether a rendering is in progress. A mapping can map further names (``float16_t`` maps ``half``); their
  entries are read too. An entry records the type table entry of every name the mapping read, the declared type
  names object and whether a rendering was in progress, and is reused only while they are unchanged and none of
  those names is declared.
- Effects. A mapping can add an emulated HLSL matrix shape and record a foreign built-in type name of the
  rendering in progress. A miss collects both and a hit adds them again, as the uncached mapping does.
- A mapping is not memoized when it records a lowering refusal, raises, rebinds or resizes any attribute of the
  generator, maps a name that is not bare or is declared by the program, or maps a ray query, ray description,
  function table, patch, stream, cooperative matrix or ``Option`` name, whose mappings set runtime requirements.
"""

from __future__ import annotations

import re

from .lowering_refusals import active_refusals

__all__ = ("memoized_bare_metal_type", "note_unmemoized_metal_mapping")

_BARE_TYPE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MEMO = "_metal_bare_type_memo"
_MISSES = "_metal_bare_type_misses"
_NOT_MEMOIZED = object()
_SHAPES = "metal_emulated_matrix_shapes"
_FOREIGN = "written_foreign_type_names"
# Bare names whose mappings set runtime requirements or refuse; always mapped uncached.
_STATEFUL_NAMES = frozenset(
    {
        "RayQuery",
        "rayQueryEXT",
        "RayDesc",
        "visible_function_table",
        "visibleFunctionTable",
        "intersection_function_table",
        "intersectionFunctionTable",
        "InputPatch",
        "OutputPatch",
        "PointStream",
        "LineStream",
        "TriangleStream",
        "CooperativeMatrix",
        "Option",
    }
)


class _Miss:
    """One uncached mapping in progress: the names it read, and whether its result may be stored."""

    __slots__ = ("names", "storable")

    def __init__(self, name):
        self.names = [name]
        self.storable = True


def _declared_name(generator, name: str) -> bool:
    """Whether the program declares *name* in a way that changes its spelling."""
    return (
        name in getattr(generator, "metal_type_aliases", ())
        or name in getattr(generator, "enum_type_names", ())
        or name in getattr(generator, "enum_struct_type_names", ())
        or name in getattr(generator, "generic_struct_definitions", ())
        or name in getattr(generator, "generic_enum_struct_definitions", ())
    )


def note_unmemoized_metal_mapping(generator) -> None:
    """Keep the mappings in progress from being stored: they read state no entry records."""
    misses = generator.__dict__.get(_MISSES)
    if misses:
        for miss in misses:
            miss.storable = False


def _uncached(generator, name: str) -> str:
    note_unmemoized_metal_mapping(generator)
    return generator.map_type(name, memoized=False)


def _valid(generator, entry) -> bool:
    if entry[0] is not getattr(generator, "program_type_names", None):
        return False
    if entry[1] != (getattr(generator, _FOREIGN, None) is None):
        return False
    type_mapping = generator.type_mapping
    for name, mapped in entry[2]:
        if type_mapping.get(name) != mapped or _declared_name(generator, name):
            return False
    return True


def _replay(generator, entry) -> None:
    if entry[4]:
        getattr(generator, _SHAPES).update(entry[4])
    if entry[5]:
        foreign = getattr(generator, _FOREIGN, None)
        if foreign is not None:
            foreign.update(entry[5])


def memoized_bare_metal_type(generator, name: str) -> str:
    """``generator.map_type(name)``, reused while the state it read is unchanged."""
    state = generator.__dict__
    memo = state.get(_MEMO)
    if memo is None:
        memo = state[_MEMO] = {}
    entry = memo.get(name)
    if entry is _NOT_MEMOIZED:
        return _uncached(generator, name)
    if entry is not None and _valid(generator, entry):
        misses = state.get(_MISSES)
        if misses:
            for miss in misses:
                miss.names.extend(read for read, _ in entry[2])
        _replay(generator, entry)
        return entry[3]
    if _declared_name(generator, name):
        return _uncached(generator, name)
    if entry is None and (
        name in _STATEFUL_NAMES or _BARE_TYPE_NAME.fullmatch(name) is None
    ):
        memo[name] = _NOT_MEMOIZED
        return _uncached(generator, name)
    return _map_and_store(generator, name, memo)


def _refusal_sinks(generator):
    """The refusal sinks a mapping can record into: the generator's and the one of the generation in progress."""
    sinks = []
    for sink in (getattr(generator, "refusals", None), active_refusals()):
        if isinstance(getattr(sink, "reasons", None), list) and all(
            sink is not seen for seen, _ in sinks
        ):
            sinks.append((sink, sink.reasons))
    return sinks


def _map_and_store(generator, name: str, memo: dict) -> str:
    """Map *name* uncached, collecting what it read and did, and store it when nothing escaped the record."""
    state = generator.__dict__
    misses = state.get(_MISSES)
    if misses is None:
        misses = state[_MISSES] = []
    miss = _Miss(name)
    misses.append(miss)
    sinks = _refusal_sinks(generator)
    for sink, _ in sinks:
        sink.reasons = []
    collected = {}
    for key in (_SHAPES, _FOREIGN):
        original = getattr(generator, key, None)
        if isinstance(original, set):
            collected[key] = (original, set())
            setattr(generator, key, collected[key][1])
    before = {key: (value, _size(value)) for key, value in state.items()}
    try:
        result = generator.map_type(name, memoized=False)
    except BaseException:
        miss.storable = False
        raise
    else:
        miss.storable = miss.storable and _unchanged(state, before)
    finally:
        misses.pop()
        if not misses:
            del state[_MISSES]
        for sink, recorded in sinks:
            if sink.reasons:
                miss.storable = False
                for reason in sink.reasons:
                    if reason not in recorded:
                        recorded.append(reason)
            sink.reasons = recorded
        for key, (original, added) in collected.items():
            if getattr(generator, key, None) is added:
                setattr(generator, key, original)
            else:
                miss.storable = False
            original.update(added)
        if misses:
            outer = misses[-1]
            outer.names.extend(miss.names)
            outer.storable = outer.storable and miss.storable
    if miss.storable:
        type_mapping = generator.type_mapping
        memo[name] = (
            getattr(generator, "program_type_names", None),
            getattr(generator, _FOREIGN, None) is None,
            tuple((read, type_mapping.get(read)) for read in dict.fromkeys(miss.names)),
            result,
            tuple(sorted(collected.get(_SHAPES, (None, ()))[1])),
            tuple(sorted(collected.get(_FOREIGN, (None, ()))[1])),
        )
    return result


def _size(value):
    return len(value) if isinstance(value, (list, set, dict)) else None


def _unchanged(state: dict, before: dict) -> bool:
    """Whether no attribute of the generator was added, rebound or resized, ignoring the memo's own records."""
    if state.keys() - before.keys() - {_MEMO, _MISSES}:
        return False
    for key, (value, size) in before.items():
        if key in (_MEMO, _MISSES, _SHAPES, _FOREIGN):
            continue
        current = state.get(key, before)
        if current is not value or _size(current) != size:
            return False
    return True
