"""``[[threadgroup(index)]]`` on Metal entry-point parameters that point to threadgroup memory.

A kernel or tile function receives host-sized threadgroup memory through a ``threadgroup`` (or
``threadgroup_imageblock``) pointer or reference parameter, located by ``[[threadgroup(index)]]`` (Metal Shading
Language Specification 5.2.1, "Locating Buffer, Texture, and Sampler Arguments", table 5.1). The host sizes each
slot with ``setThreadgroupMemoryLength:atIndex:``, so the index is part of the function's ABI: emitting the
parameter without it lets the compiler assign slots in declaration order, which moves memory the host sized for
one index to another.

The Metal importer keeps the index as a ``@threadgroup(index)`` attribute on the CrossGL parameter. This module
turns it back into the Metal attribute and refuses an attribute it cannot read as one non-negative index.
"""

from __future__ import annotations

__all__ = ["metal_threadgroup_parameter_attribute"]

_THREADGROUP_ADDRESS_SPACES = frozenset({"threadgroup", "threadgroup_imageblock"})


def _is_threadgroup_memory_parameter(node):
    qualifiers = {str(qualifier) for qualifier in getattr(node, "qualifiers", ()) or ()}
    return bool(qualifiers & _THREADGROUP_ADDRESS_SPACES)


def _threadgroup_attributes(codegen, node):
    return [
        attribute
        for attribute in getattr(node, "attributes", ()) or ()
        if codegen.normalized_metal_abi_attribute_name(attribute) == "threadgroup"
    ]


def metal_threadgroup_parameter_attribute(codegen, node):
    """Return `` [[threadgroup(n)]]`` for a threadgroup-memory parameter that carries its index, else ``""``.

    A parameter outside threadgroup memory, or one without an index, gets no attribute. More than one index, or
    an index that is not a non-negative integer literal, is recorded as a refusal through
    :meth:`MetalCodeGen.refuse_lowering`.
    """
    if node is None or not _is_threadgroup_memory_parameter(node):
        return ""
    attributes = _threadgroup_attributes(codegen, node)
    if not attributes:
        return ""
    name = getattr(node, "name", "<anonymous>")
    if len(attributes) > 1:
        return codegen.refuse_lowering(
            f"threadgroup parameter '{name}' carries {len(attributes)} [[threadgroup(index)]] attributes; a "
            "Metal argument has one threadgroup memory index"
        )
    arguments = getattr(attributes[0], "arguments", None) or []
    index = codegen.binding_index_value(arguments[0]) if len(arguments) == 1 else None
    if index is None:
        return codegen.refuse_lowering(
            f"threadgroup parameter '{name}' has a threadgroup memory index that is not one non-negative "
            "integer literal; [[threadgroup(index)]] needs an integral constant (Metal Shading Language "
            "Specification 5.2.1)"
        )
    return f" [[threadgroup({index})]]"
