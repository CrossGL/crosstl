"""Namespace-scope ``const`` and ``constexpr`` constants of CUDA and HIP sources.

A CUDA or HIP kernel can read a namespace-scope ``constexpr`` variable, and a ``const`` one of arithmetic type
with a constant initializer, without ``__device__`` or ``__constant__``. Importing such a constant as a mutable
CrossGL global (``var N: i32 = 3;``) loses that: the CUDA and HIP outputs then write a plain host variable, which
device code cannot read, and the other targets treat it as writable state. The importers write a CrossGL
``const`` instead.
"""

from __future__ import annotations

# Declarations with a memory-space or linkage qualifier keep their own lowering.
_NON_CONSTANT_QUALIFIERS = frozenset(
    {"__constant__", "__device__", "__shared__", "__managed__", "extern"}
)


def is_c_family_namespace_constant(node) -> bool:
    """Return whether a namespace-scope variable declaration is a compile-time constant.

    Pointers and references are left alone: ``const char*`` is a pointer to const data, not a constant.
    """
    if getattr(node, "value", None) is None:
        return False
    qualifiers = set(getattr(node, "qualifiers", None) or [])
    if qualifiers & _NON_CONSTANT_QUALIFIERS:
        return False
    vtype = str(getattr(node, "vtype", "") or "")
    if "*" in vtype or "&" in vtype or "[" in vtype:
        return False
    if getattr(node, "array_size", None) is not None or getattr(
        node, "array_sizes", None
    ):
        return False
    words = vtype.split()
    return (
        "constexpr" in qualifiers
        or "constexpr" in words
        or getattr(node, "is_constexpr", False)
        or "const" in qualifiers
        or "const" in words
    )
