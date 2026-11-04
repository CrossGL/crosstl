"""``__constant__``, ``__device__`` and ``__managed__`` on CUDA and HIP local variables.

CUDA and HIP allow these memory spaces only on variables with static storage: at namespace scope, or in a function
body with ``static`` or as a block-scope ``extern`` declaration. A function-body variable with automatic storage and
one of them is not valid CUDA or HIP; clang rejects it with ``__constant__, __device__, and __managed__ are not
allowed on non-static local variables``. The importers used to read it as a resource declaration inside the function,
which Metal output turned into an uninitialized local array.

A ``static __constant__`` local without an initializer is zero-initialized, and device code cannot write it, so the
importers declare it as a function-scope constant of :func:`crossgl_zero_value`.
"""

from __future__ import annotations

import re

from ...translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["LocalMemorySpaceError", "crossgl_zero_value", "refuse_local_memory_space"]

STATIC_STORAGE_MEMORY_SPACES = ("__constant__", "__device__", "__managed__")


class LocalMemorySpaceError(TranslationDiagnostic):
    """A CUDA or HIP local variable with automatic storage in the ``__constant__``, ``__device__`` or ``__managed__`` space."""

    code = f"{CODE_PREFIX}local-memory-space-invalid"


def refuse_local_memory_space(qualifiers, name):
    """Refuse the function-body declaration of ``name`` when ``qualifiers`` give it automatic storage in a memory
    space that needs static storage."""
    spaces = [space for space in STATIC_STORAGE_MEMORY_SPACES if space in qualifiers]
    if spaces and "static" not in qualifiers and "extern" not in qualifiers:
        raise LocalMemorySpaceError(
            f"local variable '{name}' is declared {' '.join(spaces)}: __constant__, __device__ and __managed__ "
            "are not allowed on non-static local variables; declare it at namespace scope"
        )


def crossgl_zero_value(var_type):
    """Zero value for a CrossGL type, for zero-initialized constants.

    Scalars map to their literal zero, vectors and matrices to a constructor of scalar zeros, and fixed-size arrays
    with a literal extent to a brace list of element zeros. Anything else (a non-literal extent, a struct) falls back
    to ``{}``, which still value-initializes instead of leaving the storage uninitialized.
    """
    var_type = var_type.strip()
    if var_type in {"f16", "f32", "f64"}:
        return "0.0"
    if var_type.startswith(("i", "u")) and var_type[1:].isdigit():
        return "0"
    if var_type == "bool":
        return "false"
    match = re.fullmatch(r"vec(\d+)<(.+)>", var_type)
    if match is not None:
        count = int(match.group(1))
        element_zero = crossgl_zero_value(match.group(2))
        return f"{var_type}({', '.join([element_zero] * count)})"
    match = re.fullmatch(r"mat(\d+)x(\d+)<(.+)>", var_type)
    if match is not None:
        columns = int(match.group(1))
        rows = int(match.group(2))
        element_zero = crossgl_zero_value(match.group(3))
        column = f"vec{rows}<{match.group(3)}>({', '.join([element_zero] * rows)})"
        return f"{var_type}({', '.join([column] * columns)})"
    if var_type.startswith("array<") and var_type.endswith(">"):
        inner = var_type[len("array<") : -1]
        element_type, extent = _split_crossgl_array_inner(inner)
        if element_type is not None and extent.isdigit():
            element_zero = crossgl_zero_value(element_type)
            return "{" + ", ".join([element_zero] * int(extent)) + "}"
    return "{}"


def _split_crossgl_array_inner(inner):
    """Split ``array<...>`` inner text into (element type, extent)."""
    depth = 0
    for index, char in enumerate(inner):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == "," and depth == 0:
            return inner[:index].strip(), inner[index + 1 :].strip()
    return None, None
