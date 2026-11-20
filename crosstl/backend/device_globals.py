"""Namespace-scope ``__device__`` and ``__managed__`` variables of CUDA and HIP, imported as CrossGL storage buffers.

A namespace-scope variable declared ``__device__`` (or ``__managed__``, which is also ``__device__``) is one object
in global memory, shared by every thread of every grid and reachable from the host through the symbol API
(``cudaMemcpyToSymbol``, ``hipGetSymbolAddress``) or, for managed memory, directly (CUDA C++ Programming Guide,
"Variable Memory Space Specifiers"; HIP follows CUDA). The importers wrote it as a module-scope ``var``, CrossGL's
per-invocation private variable, so each thread read and wrote its own copy and nothing reached memory.

:func:`device_memory_declaration` writes it as a storage buffer with the next resource binding,
``@group(0) @binding(n) var<storage, read_write> name: T;``, the CrossGL declaration of one object in device memory
that the host binds. A variable with an initializer is refused: CUDA initializes it when the module loads, and a
buffer the host binds carries no initial contents. A ``__device__`` pointer, which the host points at a buffer
through the symbol API, keeps its import as a pointer variable.
"""

from __future__ import annotations

__all__ = ["DEVICE_MEMORY_QUALIFIERS", "device_memory_declaration", "is_device_memory"]

DEVICE_MEMORY_QUALIFIERS = frozenset({"__device__", "__managed__"})
_OTHER_SPACES = frozenset({"__constant__", "__shared__", "constexpr", "const"})


def is_device_memory(node) -> bool:
    """Whether a namespace-scope declaration is a mutable ``__device__`` or ``__managed__`` value variable."""
    qualifiers = set(getattr(node, "qualifiers", None) or [])
    return (
        bool(qualifiers & DEVICE_MEMORY_QUALIFIERS)
        and not qualifiers & _OTHER_SPACES
        and "*" not in str(getattr(node, "vtype", ""))
    )


def device_memory_declaration(codegen, node, name, crossgl_type, binding, language):
    """The CrossGL declaration of a ``__device__`` variable, or a recorded refusal (and ``None``)."""
    if getattr(node, "value", None) is not None:
        codegen.refuse_import(
            f"{language} __device__ variable '{name}' has an initializer: device memory is initialized "
            "when the module loads, and a storage buffer the host binds carries no initial contents"
        )
        return None
    return f"@group(0) @binding({binding}) var<storage, read_write> {name}: {crossgl_type};"
