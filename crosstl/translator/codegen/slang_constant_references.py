"""Slang entry parameters for Metal ``constant T&`` buffer arguments.

A Metal kernel takes a small read-only argument by reference to the constant address space:
``constant float& start [[buffer(0)]]`` or ``constant Params& params [[buffer(1)]]``. CrossGL keeps it as a
``ReferenceType`` with ``address_space='constant'``. The Slang generator printed ``float &start``, which slangc does
not accept as an entry parameter. Every Metal kernel with such an argument was invalid.

Slang's ``ConstantBuffer<T>`` is the same argument. A read of ``start`` or ``params.member`` dereferences it
implicitly, and slangc lowers it to a Metal ``constant T* [[buffer(n)]]`` argument, an HLSL ``cbuffer`` and a SPIR-V
``Uniform`` block. A reference in any other address space has no such form and is refused by the caller.
"""

from __future__ import annotations

from ..ast import ReferenceType

__all__ = ["slang_constant_reference_parameter"]


def slang_constant_reference_parameter(codegen, parameter):
    """``(declaration, value_type)`` of a ``constant T&`` entry parameter, or ``None`` for any other parameter."""
    reference = getattr(parameter, "param_type", None) or getattr(
        parameter, "vtype", None
    )
    if not isinstance(reference, ReferenceType):
        return None
    if str(getattr(reference, "address_space", "") or "").lower() != "constant":
        return None
    value_type = codegen.convert_type(
        codegen.convert_type_node_to_string(reference.referenced_type)
    )
    return f"ConstantBuffer<{value_type}> {parameter.name}", value_type
