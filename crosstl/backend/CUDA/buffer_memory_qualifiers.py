"""The memory-qualifier attributes of a CUDA or HIP kernel buffer parameter: ``@volatile`` and ``@restrict``.

A kernel's ``T*`` parameter imports as a CrossGL storage buffer. Two qualifiers of its type say something about that
buffer's memory, and CrossGL carries both as attributes of the buffer, which the GLSL, SPIR-V, Metal, CUDA and HIP
generators write back (GLSL 4.60 section 4.10, SPIR-V 1.6 section 3.20 ``Volatile`` and ``Restrict``):

- ``volatile T*``: every access to the pointed-to memory is a side effect that the compiler keeps and orders
  (C++ [intro.execution]/7). The qualifier belongs to the element type, so it comes before the last ``*``.
- ``T* __restrict__``: no other pointer the kernel uses reaches this memory (CUDA C++ Programming Guide,
  "__restrict__"; HIP takes the same keyword, and ``__restrict`` and ``restrict`` are clang's spellings). The
  qualifier belongs to the pointer, so it follows the last ``*``.

``T* volatile`` makes the pointer variable volatile, not its memory, and ``__restrict__`` on an inner pointer of
``T* __restrict__*`` restricts an element of the buffer, not the buffer; neither marks the buffer. The importers
dropped both qualifiers, so every translation of such a kernel lost them.
"""

import re

__all__ = ["buffer_memory_attributes"]

_RESTRICT = re.compile(r"\b(?:__restrict__|__restrict|restrict)\b")
_VOLATILE = re.compile(r"\bvolatile\b")


def buffer_memory_attributes(type_text):
    """``"@volatile @restrict "`` (or one of them, or ``""``) for a buffer parameter of pointer type ``type_text``."""
    if not isinstance(type_text, str) or "*" not in type_text:
        return ""
    pointee, pointer = type_text.rsplit("*", 1)
    attributes = []
    if "*" not in pointee and _VOLATILE.search(pointee):
        attributes.append("@volatile ")
    if _RESTRICT.search(pointer):
        attributes.append("@restrict ")
    return "".join(attributes)
