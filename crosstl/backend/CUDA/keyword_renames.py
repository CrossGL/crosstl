"""Renames of source identifiers that CrossGL reads as keywords (#3386).

The CUDA, HIP and OpenCL C importers rename an identifier that CrossGL reserves, such as a
struct named ``let``, by appending an underscore. The new name must be one the source program
does not use: ``struct let`` next to ``struct let_`` would declare ``let_`` twice, and a local
``let`` inside the scope of a local ``let_`` would hide it.
"""

import re

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def source_identifier_names(ast_node):
    """Every identifier spelled in the strings of the parsed program ``ast_node``.

    That covers declared names, type spellings and member names. Words of a literal count too,
    which only makes a rename avoid a few more names.
    """
    names = set()
    visited = set()
    pending = [ast_node]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            names.update(_IDENTIFIER.findall(current))
        elif isinstance(current, dict):
            pending.extend(current.items())
        elif isinstance(current, (list, tuple, set, frozenset)):
            pending.extend(current)
        elif hasattr(current, "__dict__") and id(current) not in visited:
            visited.add(id(current))
            pending.extend(vars(current).values())
    return frozenset(names)


def unused_identifier_name(name, used_names):
    """``name`` followed by the fewest underscores, at least one, that keep it out of ``used_names``."""
    candidate = f"{name}_"
    while candidate in used_names:
        candidate = f"{candidate}_"
    return candidate
