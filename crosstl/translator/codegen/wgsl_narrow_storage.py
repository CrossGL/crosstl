"""8- and 16-bit integer elements of WGSL storage buffers.

WGSL has no 8- or 16-bit integer types (WGSL section 6.2.4). :mod:`..semantics.narrow` widens narrow registers to
``i32`` and leaves narrow elements of storage buffers and device pointees to the emitters, which lay that storage out
one 32-bit word per element as the SPIR-V emitter does: ``RWStructuredBuffer<uint8>`` and ``device uchar*`` become
``array<u32>``, ``int16`` elements ``i32`` words and narrow vectors ``vec<N>`` of those words. The WGSL emitter used to
write the element verbatim (``array<uint8>``), which no WGSL implementation accepts.

An access to such storage has the word type, so stores convert their values to it. Generic deduction, however, binds
type parameters to the CrossGL element type: ``bump(buffer_load(p, i))`` on a ``uint8`` buffer selects the explicit
specialization ``bump<uint8>``, never ``bump<uint>``. A generic primary instantiated for a narrow type argument would
compute in 32 bits without wrapping at the narrow width, so WGSL refuses that instantiation.
"""

from __future__ import annotations

from ..ast import (
    ArrayAccessNode,
    FunctionCallNode,
    PointerType,
    PrimitiveType,
    VectorType,
)
from ..semantics.narrow import (
    NarrowIntegerUnsupportedError,
    narrow_from_name,
    narrow_of,
)

__all__ = [
    "narrow_storage_element",
    "narrow_storage_word_type",
    "refuse_narrow_generic_instantiation",
]


def narrow_storage_word_type(element):
    """The 32-bit word type that holds a narrow storage ``element``, or ``None`` if it is not narrow."""
    narrow = narrow_of(element)
    if narrow is None:
        return None
    word = PrimitiveType("int" if narrow.kind == "i" else "uint")
    return word if narrow.lanes == 1 else VectorType(word, narrow.lanes)


def _storage_element(generator, resource):
    """The declared element type of the storage that ``resource[i]`` reads, or ``None``."""
    resource_type = generator.expression_type(resource)
    element = generator.structured_buffer_element_type(resource_type)
    if element is not None:
        return element
    if isinstance(resource_type, PointerType):
        return resource_type.pointee_type
    return None


def narrow_storage_element(generator, expr):
    """The declared narrow element type a storage load ``expr`` reads, or ``None``.

    ``expr`` is ``buffer[i]``, ``buffer_load(buffer, i)`` or ``buffer.Load(i)`` on a storage buffer or device pointer.
    """
    if isinstance(expr, FunctionCallNode):
        name = generator.expression_name(expr.function)
        expr = generator.structured_buffer_load_access(expr, name)
    if not isinstance(expr, ArrayAccessNode):
        return None
    element = _storage_element(generator, expr.array_expr)
    return element if narrow_of(element) is not None else None


def refuse_narrow_generic_instantiation(func, concrete):
    """Refuse a clone of the generic ``func`` for a narrow type argument in ``concrete``."""
    for argument in concrete:
        if narrow_from_name(argument) is not None:
            raise NarrowIntegerUnsupportedError(
                "wgsl",
                f"generic function '{getattr(func, 'name', '?')}' cannot be instantiated for the 8- or 16-bit "
                f"integer type '{argument}' without an explicit specialization",
                reason_code="generic-instantiation",
            )
