"""Members of named uniform blocks that the import flattens.

GLSL code reads a member of a uniform block with an instance name only through the instance, ``ubo.color``, so a
local, parameter or global may carry the same name (GLSL 4.60 section 4.3.9: the members of a block with an instance
name are not in the global scope). The importer flattens such a block into a CrossGL ``cbuffer`` whose members are
global names, which turned ``push_constants.color`` into ``color``. That name then read the function's local
``color`` instead of the block member. :func:`unqualified_names` collects every name the program's functions use or
declare without qualification; the importer gives a flattened member that collides with one of them a name of its own.
"""

from __future__ import annotations

from .OpenglAst import MemberAccessNode, VariableNode

__all__ = ["unqualified_names"]


def _walk(value, names, seen):
    if isinstance(value, (list, tuple)):
        for item in value:
            _walk(item, names, seen)
        return
    if not hasattr(value, "__dict__") or id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, VariableNode) and isinstance(value.name, str):
        names.add(value.name.split("[", 1)[0])
    for field, child in vars(value).items():
        if field == "parent":
            continue
        if isinstance(value, MemberAccessNode) and field == "member":
            continue
        _walk(child, names, seen)


def unqualified_names(shader) -> frozenset:
    """Names the functions of ``shader`` declare or read without a qualifier, parameters included."""
    names, seen = set(), set()
    for function in getattr(shader, "functions", None) or []:
        _walk(getattr(function, "params", None), names, seen)
        _walk(getattr(function, "body", None), names, seen)
    return frozenset(names)
