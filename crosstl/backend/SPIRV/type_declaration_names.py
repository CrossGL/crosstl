"""Names for the struct types the SPIR-V importer declares.

An ``OpName`` may give a struct type the name of a built-in type: slangc names the block of every
``RWStructuredBuffer<float>`` it compiles ``RWStructuredBuffer``. Declared under that name, the CrossGL struct shadows
the built-in type, and ``RWStructuredBuffer<RWStructuredBuffer>`` reads as a buffer of buffers. Such a struct takes the
name with a trailing underscore instead, the spelling the importer already gives identifiers that are CrossGL keywords.
"""

from __future__ import annotations

from ...translator.semantics.builtin_type_names import BUILTIN_TYPE_NAMES

__all__ = ["declarable_struct_name"]


def declarable_struct_name(name: str, taken=frozenset()) -> str:
    """``name``, or the first of ``name_``, ``name__``, ... not in ``taken`` when ``name`` is a built-in type."""
    if name not in BUILTIN_TYPE_NAMES:
        return name
    candidate = f"{name}_"
    while candidate in BUILTIN_TYPE_NAMES or candidate in taken:
        candidate = f"{candidate}_"
    return candidate
