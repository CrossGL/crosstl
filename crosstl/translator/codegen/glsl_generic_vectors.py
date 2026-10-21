"""GLSL spellings of CrossGL's generic vector constructors.

CrossGL accepts the generic spelling ``vec3<f32>(x, y, z)`` for a vector constructor, which the CUDA and HIP
importers write for ``make_float3`` and its relatives. GLSL names its vector types by component type (``vec3``,
``ivec3``, ``uvec3``, ``dvec3``, ``bvec3``; GLSL 4.60 section 4.1), so the generic spelling is not GLSL and was
emitted unchanged.
"""

from __future__ import annotations

import re

__all__ = ["glsl_generic_vector_constructor_name"]

_GENERIC_VECTOR = re.compile(r"^vec([234])\s*<\s*([A-Za-z0-9_]+)\s*>$")
_PREFIXES = {
    "f32": "",
    "float": "",
    "i32": "i",
    "int": "i",
    "u32": "u",
    "uint": "u",
    "f64": "d",
    "double": "d",
    "bool": "b",
}


def glsl_generic_vector_constructor_name(name):
    """The GLSL constructor for a generic vector constructor name (``vec3<i32>`` -> ``ivec3``), else ``name``.

    Component types without a core GLSL vector (16-bit and 8-bit values) keep their spelling for the generator's
    narrow-type handling to accept or refuse.
    """
    if not isinstance(name, str):
        return name
    match = _GENERIC_VECTOR.match(name)
    if match is None or match.group(2) not in _PREFIXES:
        return name
    return f"{_PREFIXES[match.group(2)]}vec{match.group(1)}"
