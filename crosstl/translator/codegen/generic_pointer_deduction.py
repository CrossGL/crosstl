"""Deduce a generic function's type parameter from a ``T*`` parameter that receives a storage buffer.

A helper such as ``generic<T> void clear(device T* data, uint i)`` views a structured storage buffer element by
element, so a call ``clear(buffer, i)`` with ``RWStructuredBuffer<int> buffer`` passes ``int`` storage. Deduction
only matched a pointer parameter against a pointer argument, so a type parameter that occurs nowhere else stayed
unbound and every generator refused the call as an unspecialized generic.

The pointee is bound to the buffer's element type instead. Whether the specialized call is valid is still decided
by each generator's storage-buffer parameter compatibility check.

A specialization then gave ``device T*`` its concrete type by spelling: ``int*`` lost the pointer's address space,
so the HLSL generator no longer saw the parameter or a local ``device T* p = data + offset`` as a view into a storage
buffer. :func:`concrete_generic_pointer_type` keeps the pointer node and replaces only its pointee.
"""

from __future__ import annotations

import copy

from ..ast import PointerType
from .enum_utils import generic_type_parts

__all__ = ["concrete_generic_pointer_type", "structured_buffer_element_type"]

# Structured buffers whose elements a pointer parameter can address. Append and consume buffers expose no element
# addresses, and typed or byte-address buffers do not store elements of their declared type one per index.
STRUCTURED_BUFFER_TYPE_NAMES = frozenset(
    {
        "StructuredBuffer",
        "RWStructuredBuffer",
        "RasterizerOrderedStructuredBuffer",
    }
)


def structured_buffer_element_type(type_text):
    """The element type of ``type_text`` when it names a structured storage buffer, else ``None``."""
    base, arguments = generic_type_parts(str(type_text or "").strip())
    if base in STRUCTURED_BUFFER_TYPE_NAMES and len(arguments) == 1:
        element_type = str(arguments[0]).strip()
        return element_type or None
    return None


def _parsed_type_node(type_text):
    """The CrossGL type node spelled ``type_text``, or ``None`` when it does not parse as one type."""
    from ..lexer import Lexer
    from ..parser import Parser

    try:
        parser = Parser(Lexer(f"{type_text} __crossgl_type;").tokens)
        type_node = parser.parse_type()
    except Exception:
        return None
    if parser.current_token[0] != "IDENTIFIER":
        return None
    return type_node


def concrete_generic_pointer_type(type_value, substitute):
    """A copy of the pointer ``type_value`` with a concrete pointee, or ``None``.

    ``substitute`` returns the concrete pointee, a type node or its spelling, or ``None`` when the pointee names no
    type parameter. ``None`` is also returned when ``type_value`` is not a pointer or the concrete spelling does not
    parse, so the caller keeps its spelling-based substitution.
    """
    if not isinstance(type_value, PointerType):
        return None
    pointee = substitute(type_value.pointee_type)
    if pointee is None:
        return None
    if isinstance(pointee, str):
        pointee = _parsed_type_node(pointee)
        if pointee is None:
            return None
    concrete = copy.copy(type_value)
    concrete.pointee_type = pointee
    return concrete
