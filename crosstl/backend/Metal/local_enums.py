"""Enums declared in a Metal function body become module-scope enums.

MSL follows C++, where an enum may be declared in a block; its name and, for an unscoped enum, its enumerators are
names of that block (C++17 section 10.2). CrossGL declares enums only at module scope and spells an enumerator
``E::A``. A type has no run-time state, so moving the declaration out of the function changes nothing the program
computes, provided the moved names do not meet another declaration there: :func:`hoist_local_enums` moves each
block-scope enum to the module's enums, where the importer writes it and qualifies its enumerators as it does for any
unscoped enum, and refuses one whose name or enumerator a module-scope declaration or another moved enum already
uses. Local variables still hide an enumerator of the same name, as they hide a module-scope one.
"""

from __future__ import annotations

from .MetalAst import EnumNode

__all__ = ["hoist_local_enums"]

_SKIPPED_FIELDS = frozenset({"parent", "annotations", "source_location"})


def hoist_local_enums(ast, refuse) -> int:
    """Move enums declared in function bodies to ``ast.enums``; returns how many moved.

    ``refuse(reason)`` records a clash; the enum then stays where it is.
    """
    functions = list(getattr(ast, "functions", None) or [])
    if not functions:
        return 0
    taken = _module_names(ast)
    moved = 0
    for function in functions:
        for container, enum in _block_enums(getattr(function, "body", None)):
            clash = sorted(_declared_names(enum) & taken)
            if clash:
                refuse(
                    f"the enum '{enum.name or '<anonymous>'}' declared in function "
                    f"'{getattr(function, 'name', '')}' declares '{clash[0]}', which another declaration at "
                    "module scope already names, so it cannot move to module scope"
                )
                continue
            container.remove(enum)
            ast.enums = [*(getattr(ast, "enums", None) or []), enum]
            taken |= _declared_names(enum)
            moved += 1
    return moved


def _declared_names(enum):
    names = {member for member, _value in getattr(enum, "members", None) or ()}
    if getattr(enum, "is_scoped", False):
        names = set()
    if enum.name:
        names.add(enum.name)
    return names


def _module_names(ast):
    names = set()
    for enum in getattr(ast, "enums", None) or []:
        if isinstance(enum, EnumNode):
            names |= _declared_names(enum)
    for field in (
        "structs",
        "functions",
        "global_variables",
        "global_vars",
        "typedefs",
    ):
        for item in getattr(ast, field, None) or []:
            name = getattr(item, "name", None)
            if isinstance(name, str) and name:
                names.add(name)
    return names


def _block_enums(body):
    """``(statement list, enum)`` for each enum declared in ``body`` or a block nested in it, in source order."""
    found = []

    def visit(value):
        if isinstance(value, list):
            for item in list(value):
                if isinstance(item, EnumNode):
                    found.append((value, item))
                else:
                    visit(item)
        elif hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
            for field, child in vars(value).items():
                if field not in _SKIPPED_FIELDS:
                    visit(child)

    visit(body)
    return found
