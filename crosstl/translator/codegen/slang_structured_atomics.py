"""The location an atomic call names inside a Slang structured buffer.

HLSL's ``Interlocked*`` functions, which Slang declares, take an ``int`` or ``uint`` destination in groupshared
memory or a writable resource. An ``int`` or ``uint`` member of a ``RWStructuredBuffer`` element is such a
destination: the HLSL generator writes ``InterlockedAdd(buf[0].m, v, original)``, and slangc compiles the same call
for SPIR-V and DXIL. The Slang generator typed the location of ``atomicAdd(buf[i].count, 1)`` as the buffer's
element, the structure, so it refused every atomic on a structure member. :func:`structured_buffer_atomic_location`
types the location itself instead, and names the buffer variable whose access the call needs.
"""

from __future__ import annotations

from ..ast import ArrayAccessNode, IdentifierNode, MemberAccessNode, VariableNode

__all__ = ["structured_buffer_atomic_location"]


def _inner(node):
    if isinstance(node, ArrayAccessNode):
        return getattr(node, "array", getattr(node, "array_expr", None))
    if isinstance(node, MemberAccessNode):
        return getattr(node, "object", getattr(node, "object_expr", None))
    return None


def structured_buffer_atomic_location(gen, target, buffer_type):
    """``(buffer name, location type)`` of the atomic ``target`` inside the structured buffer ``buffer_type``.

    A target that is an element, ``buf[i]``, has the element type. A target that reaches into the element through
    a member, ``buf[i].m`` or ``buf[i].a[j]``, has the type of that member. The buffer name is ``None`` when the
    target does not start at a named buffer.
    """
    through_member = False
    node = target
    while isinstance(node, (ArrayAccessNode, MemberAccessNode)):
        through_member = through_member or isinstance(node, MemberAccessNode)
        node = _inner(node)
    name = node.name if isinstance(node, (IdentifierNode, VariableNode)) else None
    if not through_member:
        return name, gen.structured_buffer_element_type(buffer_type)
    location_type = gen.expression_result_type(target)
    return name, gen.convert_type(location_type) if location_type else None
