"""Structured buffer handles as function results and locals in Metal output (#2194).

Metal passes a CrossGL ``StructuredBuffer<T>`` as ``const device T*`` and a ``RWStructuredBuffer<T>`` as
``device T*``, the spelling the generator gives a helper's buffer parameter. A function that returns a buffer
handle, as the Metal importer types a selector ``device float* pick(int k, device float* a, device float* b)``,
or a local that holds one, takes the same pointer type; the handle type itself is no Metal type and its spelling
did not compile. Append and consume buffers carry a counter beside the elements, which a returned pointer cannot,
so a function or local of those types is refused.
"""

from __future__ import annotations

from ..refusals import MetalRefusal

__all__ = ["is_metal_buffer_handle_type", "metal_buffer_handle_pointer_type"]

_COUNTED = frozenset({"AppendStructuredBuffer", "ConsumeStructuredBuffer"})
_HANDLES = frozenset({"StructuredBuffer", "RWStructuredBuffer"})


def _handle_name(codegen, vtype):
    if vtype is None:
        return None, None
    type_name = codegen.type_name_string(vtype)
    if not isinstance(type_name, str) or "<" not in type_name:
        return None, None
    return type_name, codegen.structured_buffer_type_name(type_name)


def is_metal_buffer_handle_type(codegen, vtype):
    """Whether ``vtype`` is a ``StructuredBuffer<T>`` or ``RWStructuredBuffer<T>``, which Metal holds in a ``device`` pointer."""
    return _handle_name(codegen, vtype)[1] in _HANDLES


def metal_buffer_handle_pointer_type(codegen, vtype, role):
    """The Metal pointer type of a structured buffer handle ``vtype``, else ``None``.

    ``role`` names the declaration for the refusal of a counted buffer, as in ``"result of 'pick'"``.
    """
    type_name, base_name = _handle_name(codegen, vtype)
    if base_name in _COUNTED:
        raise MetalRefusal(
            f"Metal output cannot write the {role} as a {base_name}: the pointer Metal passes for the buffer "
            "has no room for its counter"
        )
    if base_name == "StructuredBuffer":
        space = "const device"
    elif base_name == "RWStructuredBuffer":
        space = "device"
    else:
        return None
    return f"{space} {codegen.structured_buffer_element_type(type_name)}*"
