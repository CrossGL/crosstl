"""Refusal of ``__constant__``, ``__device__`` and ``__managed__`` on CUDA and HIP local variables.

CUDA and HIP allow these memory spaces only on variables with static storage: at namespace scope, or in a function
body with ``static`` or as a block-scope ``extern`` declaration. A function-body variable with automatic storage and
one of them is not valid CUDA or HIP; clang rejects it with ``__constant__, __device__, and __managed__ are not
allowed on non-static local variables``. The importers used to read it as a resource declaration inside the function,
which Metal output turned into an uninitialized local array.
"""

from __future__ import annotations

from ...translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["LocalMemorySpaceError", "refuse_local_memory_space"]

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
