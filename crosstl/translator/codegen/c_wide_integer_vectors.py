"""Vectors of 64-bit integers in the CUDA and HIP generators.

CrossGL spells a vector of 64-bit integers by name, ``i64vec2`` and ``u64vec3``, or generically, ``vec2<i64>``
(the spelling the CUDA and HIP importers give ``longlong2`` and ``ulonglong3``). The two generators mapped only
vectors of 8- to 32-bit elements, so a 64-bit vector came out as ``u64vec2``, ``vec2<i64>`` or ``i642``, none of
which names a CUDA or HIP type, and a CUDA source using ``longlong2`` could not even be translated back to CUDA.

CUDA's ``vector_types.h`` and HIP's ``hip_vector_types.h`` both declare ``longlong2`` to ``longlong4`` and
``ulonglong2`` to ``ulonglong4``, with ``make_longlong2`` and its relatives as constructors. Their components are
``long long`` and ``unsigned long long``. The vector helpers describe those components as the CrossGL scalars
``i64`` and ``u64``, so a component read keeps all 64 bits instead of narrowing to ``int`` or ``uint``.
"""

from __future__ import annotations

import re

__all__ = [
    "WIDE_INTEGER_COMPONENT_C_TYPES",
    "is_integer_vector_component",
    "integer_vector_component_signedness",
    "wide_integer_vector_constructor",
    "wide_integer_vector_details",
    "wide_integer_vector_prefix",
    "wide_integer_vector_type",
]

# CrossGL component type of a 64-bit vector -> the C type of one component.
WIDE_INTEGER_COMPONENT_C_TYPES = {
    "i64": "long long",
    "u64": "unsigned long long",
}

# Element spellings CrossGL accepts for a 64-bit integer -> component type.
_WIDE_ELEMENTS = {
    "i64": "i64",
    "int64": "i64",
    "int64_t": "i64",
    "u64": "u64",
    "uint64": "u64",
    "uint64_t": "u64",
}

_NATIVE_PREFIXES = {"i64": "longlong", "u64": "ulonglong"}

_NAMED = re.compile(r"^([iu])64vec([234])$")
_GENERIC = re.compile(r"^vec([234])<([A-Za-z0-9_]+)>$")
_NATIVE = re.compile(r"^(u?)longlong([234])$")

_COMPONENTS = ("x", "y", "z", "w")


def _wide_parts(type_name):
    """``(component_type, size)`` for a 64-bit integer vector spelling, else ``None``."""
    if not isinstance(type_name, str):
        return None
    compact = "".join(type_name.split())
    match = _NAMED.match(compact)
    if match is not None:
        return f"{match.group(1)}64", int(match.group(2))
    match = _GENERIC.match(compact)
    if match is not None:
        component = _WIDE_ELEMENTS.get(match.group(2))
        if component is None:
            return None
        return component, int(match.group(1))
    match = _NATIVE.match(compact)
    if match is not None:
        return ("u64" if match.group(1) else "i64"), int(match.group(2))
    return None


def wide_integer_vector_type(type_name):
    """The CUDA/HIP type of a 64-bit integer vector spelling (``u64vec2`` -> ``ulonglong2``), else ``None``."""
    parts = _wide_parts(type_name)
    if parts is None:
        return None
    component, size = parts
    return f"{_NATIVE_PREFIXES[component]}{size}"


def wide_integer_vector_constructor(type_name):
    """The constructor of a 64-bit integer vector spelling (``vec2<i64>`` -> ``make_longlong2``), else ``None``."""
    native = wide_integer_vector_type(type_name)
    return None if native is None else f"make_{native}"


def wide_integer_vector_details():
    """``vector_type_info`` rows for the native 64-bit vectors: type -> (constructor, component, components)."""
    details = {}
    for component, prefix in sorted(_NATIVE_PREFIXES.items()):
        for size in (2, 3, 4):
            native = f"{prefix}{size}"
            details[native] = (f"make_{native}", component, _COMPONENTS[:size])
    return details


def wide_integer_vector_prefix(component_type):
    """The CrossGL named-vector prefix of a 64-bit component (``u64`` -> ``u64vec``), else ``None``."""
    if component_type in WIDE_INTEGER_COMPONENT_C_TYPES:
        return f"{component_type}vec"
    return None


def is_integer_vector_component(component_type):
    """Whether a vector component type (``int``, ``uint``, ``i64``, ``u64``) is an integer."""
    return component_type in {"int", "uint", "i64", "u64"}


def integer_vector_component_signedness(component_type):
    """``"signed"`` or ``"unsigned"`` for an integer component type, else ``None``."""
    if component_type in {"int", "i64"}:
        return "signed"
    if component_type in {"uint", "u64"}:
        return "unsigned"
    return None
