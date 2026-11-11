"""Memoized HLSL spellings of bare type names.

``HLSLCodeGen.map_type`` runs millions of times on large programs, and nearly
every call asks for one of a few bare names (``float``, ``uint3``, a struct
name). The spelling of a bare name depends on generator state that can change
between renders and during one: the type table, type aliases, enum and generic
declarations, the target profile and float16 widening.

A memo entry therefore records the state it was computed under and is reused
only while that state still holds. Names that are aliases, enums or generic
declarations are never memoized, because their spelling depends on more than
the name. A mapping that requires native 16-bit types sets
``requires_hlsl_native_16_bit_types`` again on every hit, exactly as the
uncached mapping does. Mappings that raise are not memoized.
"""

from __future__ import annotations

import re

__all__ = ("memoized_bare_hlsl_type",)

_BARE_TYPE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NOT_BARE = object()
_NATIVE_16_BIT_FLAG = "requires_hlsl_native_16_bit_types"


def _declared_name(generator, name: str) -> bool:
    """Whether *name* is declared by the program in a way that changes its spelling."""
    return (
        name in getattr(generator, "hlsl_type_aliases", ())
        or name in getattr(generator, "enum_type_names", ())
        or name in getattr(generator, "enum_struct_type_names", ())
        or name in getattr(generator, "generic_struct_definitions", ())
        or name in getattr(generator, "generic_enum_struct_definitions", ())
    )


def _mapping_state(generator, name: str):
    return (
        generator.type_mapping.get(name),
        getattr(generator, "widen_native_float16", False),
        getattr(generator, "target_profile", None),
    )


def _uncached(generator, name: str) -> str:
    return generator.map_type(name, memoized=False)


def memoized_bare_hlsl_type(generator, name: str) -> str:
    """``generator.map_type(name)``, reused while the state it read is unchanged."""
    memo = generator.__dict__.get("_hlsl_bare_type_memo")
    if memo is None:
        memo = generator.__dict__["_hlsl_bare_type_memo"] = {}
    entry = memo.get(name)
    if entry is _NOT_BARE or _declared_name(generator, name):
        return _uncached(generator, name)
    state = _mapping_state(generator, name)
    if entry is not None and entry[0] == state:
        if entry[2]:
            setattr(generator, _NATIVE_16_BIT_FLAG, True)
        return entry[1]
    if entry is None and _BARE_TYPE_NAME.fullmatch(name) is None:
        memo[name] = _NOT_BARE
        return _uncached(generator, name)

    # Observe whether this mapping raises the native 16-bit requirement, even
    # when an earlier mapping already raised it, so a hit can replay it.
    previous = getattr(generator, _NATIVE_16_BIT_FLAG, False)
    setattr(generator, _NATIVE_16_BIT_FLAG, False)
    try:
        result = _uncached(generator, name)
        requires_native_16_bit = getattr(generator, _NATIVE_16_BIT_FLAG, False)
    finally:
        setattr(
            generator,
            _NATIVE_16_BIT_FLAG,
            previous or getattr(generator, _NATIVE_16_BIT_FLAG, False),
        )
    memo[name] = (state, result, requires_native_16_bit)
    return result
