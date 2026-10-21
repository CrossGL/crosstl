"""Addresses of vector components for Metal atomics.

MSL does not let a program take the address of a vector component ("address of vector element requested"), so an
atomic on ``v[i].z`` cannot be written as ``reinterpret_cast<device atomic_uint*>(&v[i].z)``. A vector's components
are stored in order, one scalar after another, in both the natural and the packed vector types (Metal Shading
Language Specification, sections 2.2 and 2.2.3), so the component is addressed through the vector instead:
``reinterpret_cast<device uint*>(&v[i]) + 2``.
"""

from __future__ import annotations

import re

from ..ast import MemberAccessNode

__all__ = ["metal_vector_component_address"]

_COMPONENTS = {"x": 0, "y": 1, "z": 2, "w": 3, "r": 0, "g": 1, "b": 2, "a": 3}
_INTEGER_VECTOR = re.compile(r"(?:packed_)?(?:int|uint)([234])")


def metal_vector_component_address(codegen, target, address_space, scalar_type):
    """The address of ``target`` when it is one component of an integer vector, or ``None``."""
    if not isinstance(target, MemberAccessNode):
        return None
    index = _COMPONENTS.get(str(target.member))
    if index is None:
        return None
    vector = target.object_expr
    mapped = codegen.map_type(codegen.expression_result_type(vector)) or ""
    match = _INTEGER_VECTOR.fullmatch(mapped)
    if match is None or index >= int(match.group(1)):
        return None
    vector_expr = codegen.generate_expression(vector)
    if not codegen.is_metal_address_expression(vector, vector_expr):
        vector_expr = f"&{vector_expr}"
    return (
        f"(reinterpret_cast<{address_space} {scalar_type}*>({vector_expr}) + {index})"
    )
