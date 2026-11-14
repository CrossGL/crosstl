"""Locals that share their name with a member of a named uniform block.

GLSL code reads a member of a uniform block with an instance name only through the instance, ``ubo.color``, so a local
or parameter may carry the same name (GLSL 4.60 section 4.3.9: the members of a block with an instance name are not in
the global scope). The importer flattens such a block into a CrossGL ``cbuffer`` whose members are global names, which
turned ``push_constants.color`` into ``color``; in a function with a local ``color`` it then read the local. The member
keeps its name, which the host may bind it by, so :func:`rename_shadowing_locals` renames the local instead: every
parameter or local of such a name, and its uses, in the function that declares it.
"""

from __future__ import annotations

from .OpenglAst import MemberAccessNode, VariableNode

__all__ = [
    "function_variable_names",
    "instance_block_member_names",
    "rename_shadowing_locals",
]


def instance_block_member_names(shader) -> frozenset:
    """Member names of the uniform blocks of ``shader`` that have an instance name."""
    names = set()
    for struct in getattr(shader, "structs", None) or []:
        if not getattr(struct, "interface_block", False):
            continue
        if not getattr(struct, "interface_instance_name", None):
            continue
        qualifiers = {
            str(q).lower() for q in getattr(struct, "interface_qualifiers", None) or []
        }
        if "uniform" not in qualifiers:
            continue
        for member in getattr(struct, "members", None) or []:
            name = getattr(member, "name", None)
            if isinstance(name, str):
                names.add(name.split("[", 1)[0])
    return frozenset(names)


def _variables(value, found, seen):
    """Every ``VariableNode`` under ``value`` that names a variable (not a member selector)."""
    if isinstance(value, (list, tuple)):
        for item in value:
            _variables(item, found, seen)
        return
    if not hasattr(value, "__dict__") or id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, VariableNode) and isinstance(value.name, str):
        found.append(value)
    for field, child in vars(value).items():
        if field == "parent" or (
            isinstance(value, MemberAccessNode) and field == "member"
        ):
            continue
        _variables(child, found, seen)


def function_variable_names(shader) -> set:
    """Names of the parameters, locals and other variables that ``shader``'s functions declare or read."""
    nodes = []
    for function in getattr(shader, "functions", None) or []:
        _variables(
            [getattr(function, "params", None), getattr(function, "body", None)],
            nodes,
            set(),
        )
    return {node.name.split("[", 1)[0] for node in nodes}


def _declares(node) -> bool:
    return bool(getattr(node, "vtype", None))


def rename_shadowing_locals(shader, member_names) -> int:
    """Rename the parameters and locals of ``shader``'s functions named after ``member_names``; returns how many."""
    if not member_names:
        return 0
    taken = set(member_names)
    every = []
    for function in getattr(shader, "functions", None) or []:
        _variables([function.params, function.body], every, set())
    taken.update(node.name for node in every)
    count = 0
    for function in getattr(shader, "functions", None) or []:
        nodes = []
        _variables(
            [getattr(function, "params", None), getattr(function, "body", None)],
            nodes,
            set(),
        )
        declared = {
            node.name.split("[", 1)[0]
            for node in nodes
            if _declares(node) and node.name.split("[", 1)[0] in member_names
        }
        for name in sorted(declared):
            index = 1
            while f"{name}_{index}" in taken:
                index += 1
            fresh = f"{name}_{index}"
            taken.add(fresh)
            for node in nodes:
                base, bracket, rest = node.name.partition("[")
                if base == name:
                    node.name = fresh + bracket + rest
            count += 1
    return count
