"""Memory qualifiers of OpenCL C kernel pointer parameters.

A kernel's ``__global`` or ``__constant`` pointer parameter becomes a CrossGL storage buffer. Two of the C type
qualifiers it may carry change how the kernel accesses that memory (OpenCL C 3.0, section 6.7, and C11 6.7.3):

- ``volatile`` on the pointed-to type (``__global volatile float* p``): every access is performed as written, so
  the compiler may not merge, reorder or remove them. CrossGL spells it ``@volatile`` on the buffer, which SPIR-V
  carries as ``Decoration.Volatile``, GLSL as ``volatile buffer``, Metal as ``volatile device float*`` and HIP as
  ``volatile float*``.
- ``restrict`` on the pointer itself (``__global float* restrict p``): no other pointer the kernel uses reaches the
  memory it accesses. CrossGL spells it ``@restrict``, which SPIR-V carries as ``Decoration.Restrict`` and GLSL as
  ``restrict buffer``.

A ``volatile`` on the pointer itself (``float* volatile p``) qualifies only the parameter's own copy of the
address, not the memory it points to, so it gives the buffer no attribute. The same holds for ``const`` there:
``__global float* const p`` is a constant address of writable memory, so its buffer stays ``read_write``.
"""

from __future__ import annotations

__all__ = ["opencl_pointee_qualifiers", "opencl_pointer_memory_attributes"]

_RESTRICT = frozenset({"__restrict__", "__restrict", "restrict"})


def _split(raw_type):
    """The qualifier words of the pointed-to type and of the pointer, split at the last ``*``."""
    text = str(raw_type)
    star = text.rfind("*")
    if star < 0:
        return set(text.split()), set()
    return set(text[:star].replace("*", " ").split()), set(text[star + 1 :].split())


def opencl_pointee_qualifiers(raw_type) -> set[str]:
    """The words of a pointer parameter type before its last ``*``: the address space, qualifiers and type."""
    return _split(raw_type)[0]


def opencl_pointer_memory_attributes(raw_type) -> str:
    """The ``@volatile``/``@restrict`` attributes, each followed by a space, of a kernel pointer parameter type.

    ``raw_type`` is the parser's type string, such as ``"__global__ const volatile int * __restrict__"``: the
    qualifiers before the last ``*`` qualify the pointed-to type and those after it the pointer.
    """
    if "*" not in str(raw_type):
        return ""
    pointee, pointer = _split(raw_type)
    attributes = []
    if "volatile" in pointee:
        attributes.append("@volatile ")
    if pointer & _RESTRICT:
        attributes.append("@restrict ")
    return "".join(attributes)
