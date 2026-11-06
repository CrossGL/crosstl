"""Block-scope aggregate declarations that also declare objects.

A C++ simple declaration may define a class and declare objects of it at once, as in
``union { uint i; half t[2]; } bits;`` ([dcl.pre], [class.pre]). The class may be unnamed only when the declaration
declares at least one object or typedef name; an unnamed union that declares nothing is an anonymous union whose
members join the enclosing scope ([class.union.anon]), which this importer does not model.

The Metal importer hoists block-scope aggregates to program scope, so the declaration is split into the aggregate
followed by ordinary declarations of its objects, in source order. An unnamed aggregate receives a name that
contains a double underscore. Such names are reserved to the implementation ([lex.name]), so no program can declare
or spell it, and the hoisted declaration is renamed again before emission.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "UNNAMED_AGGREGATE_PREFIX",
    "is_unnamed_aggregate_name",
    "split_block_aggregate_declaration",
    "unnamed_aggregate_display_name",
    "unnamed_aggregate_name",
    "union_layout",
]

UNNAMED_AGGREGATE_PREFIX = "__crossgl_unnamed_"


def unnamed_aggregate_name(kind: str, ordinal: int) -> str:
    """The reserved source-level name of the ``ordinal``-th unnamed ``kind`` in a translation unit."""

    return f"{UNNAMED_AGGREGATE_PREFIX}{kind}_{int(ordinal)}"


def is_unnamed_aggregate_name(name: object) -> bool:
    return isinstance(name, str) and name.startswith(UNNAMED_AGGREGATE_PREFIX)


def unnamed_aggregate_display_name(kind: str) -> str:
    """The base of the emitted name of a hoisted unnamed aggregate, such as ``UnnamedUnion``."""

    return "Unnamed" + (str(kind or "aggregate").capitalize())


def split_block_aggregate_declaration(aggregate: Any) -> Any:
    """The block statements of an aggregate definition: the aggregate, then the objects it declares.

    ``None`` (a forward declaration) and an aggregate that declares no objects are returned unchanged.
    """

    if aggregate is None:
        return None
    declarations = list(getattr(aggregate, "trailing_declarations", None) or ())
    if not declarations:
        return aggregate
    aggregate.trailing_declarations = []
    return [aggregate, *declarations]


def union_layout(member_layouts: list[tuple[int, int]]) -> tuple[int, int]:
    """The size and alignment of a union whose members have ``member_layouts``.

    Every member starts at offset 0 ([class.union]), so the union is as large as its largest member rounded up to
    its strictest alignment.
    """

    size = max((member_size for member_size, _ in member_layouts), default=0)
    alignment = max(
        (member_alignment for _, member_alignment in member_layouts), default=1
    )
    size = -(-size // alignment) * alignment
    return max(size, 1), alignment
