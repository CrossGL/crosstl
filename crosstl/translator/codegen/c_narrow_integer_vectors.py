"""Named vectors of 8- and 16-bit integers in the CUDA and HIP generators.

CrossGL spells a vector of 8- or 16-bit integers by name, ``i8vec2`` and ``u16vec3`` (the spelling the GLSL importer
keeps from ``GL_EXT_shader_explicit_arithmetic_types``), or generically, ``vec2<i8>`` (the spelling the CUDA and HIP
importers give ``char2`` and ``ushort3``). The two generators mapped only the generic spelling, so a named vector came
out verbatim as ``i8vec2``, which names no CUDA or HIP type.

CUDA's ``vector_types.h`` and HIP's ``hip_vector_types.h`` both declare ``char2`` to ``char4``, ``uchar2`` to
``uchar4``, ``short2`` to ``short4`` and ``ushort2`` to ``ushort4``, with ``make_char2`` and its relatives as
constructors; the generators already map the generic spelling to them and lower their arithmetic. A named spelling
is rewritten to the generic one first, so both take the same path.
"""

from __future__ import annotations

import re

__all__ = ["generic_narrow_vector_spelling"]

_NAMED = re.compile(r"^([iu])(8|16)vec([234])$")


def generic_narrow_vector_spelling(type_name):
    """The generic spelling of a named 8- or 16-bit integer vector (``u16vec3`` -> ``vec3<u16>``), else ``None``."""
    if not isinstance(type_name, str):
        return None
    match = _NAMED.match(type_name.strip())
    if match is None:
        return None
    kind, bits, size = match.groups()
    return f"vec{size}<{kind}{bits}>"
