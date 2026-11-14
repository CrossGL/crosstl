"""Distinct names for a SPIR-V function's parameters and local variables.

``OpName`` is debug information: SPIR-V lets any number of ids carry the same name (SPIR-V 1.6 section 3.32.2), and
glslang names every temporary it creates for an ``in`` argument ``param``. The importer declares each
``Function``-storage ``OpVariable`` as a CrossGL local by its name, so two locals called ``param`` became one
redeclared variable, which every target compiler rejects. :func:`function_local_names` gives the function its own
name table in which each parameter and local keeps its name the first time it appears and later ones take the first
``name_N`` no id in the module is named.
"""

from __future__ import annotations

__all__ = ["function_local_names"]


def _local_ids(raw_instructions):
    for result_id, opcode, operands, _line in raw_instructions:
        if not result_id:
            continue
        if opcode == "OpFunctionParameter":
            yield result_id
        elif (
            opcode == "OpVariable" and len(operands) >= 2 and operands[1] == "Function"
        ):
            yield result_id


def function_local_names(raw_instructions, names):
    """``names``, or a copy of it in which the function's locals and parameters have distinct names."""
    local_ids = list(dict.fromkeys(_local_ids(raw_instructions)))
    seen = {}
    for value_id in local_ids:
        name = names.get(value_id)
        if name:
            seen[name] = seen.get(name, 0) + 1
    if all(count == 1 for count in seen.values()):
        return names

    local = dict(names)
    used = {name for name in names.values() if name}
    kept = set()
    for value_id in local_ids:
        name = names.get(value_id)
        if not name:
            continue
        if name not in kept:
            kept.add(name)
            continue
        suffix = 1
        while f"{name}_{suffix}" in used:
            suffix += 1
        unique = f"{name}_{suffix}"
        used.add(unique)
        local[value_id] = unique
    return local
