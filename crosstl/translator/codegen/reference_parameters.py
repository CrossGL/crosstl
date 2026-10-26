"""Pass ``out`` and ``inout`` value parameters by reference in C++-like targets.

CUDA and HIP have no ``out``/``inout`` parameter qualifiers. A helper function
that writes such a parameter must receive a reference, otherwise the write is
made to a local copy and never reaches the caller.
"""

from ..ast import (
    ArrayType,
    FunctionType,
    GenericType,
    MatrixType,
    NamedType,
    PointerType,
    PrimitiveType,
    ReferenceType,
    VectorType,
)

WRITABLE_PARAMETER_QUALIFIERS = frozenset({"out", "inout"})

_RESOURCE_NAME_FRAGMENTS = (
    "buffer",
    "texture",
    "sampler",
    "image",
    "stream",
    "acceleration",
    "rayquery",
)


def parameter_is_writable(parameter):
    """Return whether ``parameter`` is declared ``out`` or ``inout``."""
    qualifiers = {
        str(qualifier).lower()
        for qualifier in getattr(parameter, "qualifiers", []) or []
    }
    return bool(qualifiers & WRITABLE_PARAMETER_QUALIFIERS)


def is_reference_value_type(raw_type, type_name, struct_names=()):
    """Return whether a parameter of this type can be bound to ``T&``.

    Arrays, pointers, references, resources and generic types keep their
    existing lowering; only scalar, vector, matrix and struct values qualify.
    """
    if isinstance(raw_type, (ArrayType, PointerType, ReferenceType, FunctionType)):
        return False
    if isinstance(raw_type, GenericType):
        return False
    if isinstance(raw_type, (PrimitiveType, VectorType, MatrixType)):
        return True
    name = str(type_name or "").strip()
    if not name or any(token in name for token in ("[", "*", "&", "<")):
        return False
    if isinstance(raw_type, NamedType) or not isinstance(raw_type, str):
        return name in struct_names
    lowered = name.lower()
    if any(fragment in lowered for fragment in _RESOURCE_NAME_FRAGMENTS):
        return False
    return True


def reference_declarator(declaration, name):
    """Turn ``T name`` into ``T& name``."""
    suffix = f" {name}"
    if not declaration.endswith(suffix):
        return declaration
    return f"{declaration[: -len(suffix)]}& {name}"
