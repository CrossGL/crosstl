"""Deduce a generic function's type parameter from a ``vecN<T>`` parameter.

A generic helper such as ``generic<T> T sum(vec2<T> v)`` stores its parameter as ``VectorType(T, 2)``. Generators
render that node by joining element and size (``T2``), so deduction compared ``T2`` with the argument's ``int2`` or
``ivec2`` and bound nothing: GLSL and SPIR-V refused the call, while HLSL and Metal printed the generic name as if it
were declared.

The signature is read as ``vec2<T>`` instead, and a concrete vector argument in any CrossGL, GLSL, HLSL, MSL or
SPIR-V-generator spelling binds ``T`` to its element type. The specialization then substitutes the concrete vector
into its signature, locals and ``vec2<T>(...)`` constructors.
"""

from __future__ import annotations

import re

from ..ast import PrimitiveType, VectorType
from .crossgl_vector_names import crossgl_named_vector_type
from .enum_utils import substitute_generic_type_name

__all__ = [
    "bind_generic_vector_parameter",
    "concrete_generic_vector_name",
    "concrete_generic_vector_type",
    "generic_vector_signature_name",
    "substituted_generic_type",
    "substituted_generic_type_name",
    "vector_element_and_size",
]

_GENERIC_VECTOR = re.compile(r"^vec([234])\s*<\s*(.+?)\s*>$")

# Prefixes of CrossGL and GLSL named vectors (``ivec3``, ``u16vec2``) -> element type.
_NAMED_VECTOR_ELEMENTS = {
    "": "float",
    "i": "int",
    "u": "uint",
    "b": "bool",
    "d": "double",
    "f16": "f16",
    "f32": "f32",
    "f64": "f64",
    "i8": "i8",
    "u8": "u8",
    "i16": "i16",
    "u16": "u16",
    "i32": "i32",
    "u32": "u32",
    "i64": "i64",
    "u64": "u64",
    "bfloat16": "bfloat16",
}
_NAMED_VECTOR = re.compile(r"^([a-z0-9]*?)vec([234])$")

# Scalars that HLSL and MSL suffix with the component count (``int2``, ``uint16_t3``, ``ushort4``) -> the CrossGL
# element, so that one deduced ``T`` names the same type on every generator.
_SUFFIXED_VECTOR_ELEMENTS = {
    "float": "float",
    "float32_t": "float",
    "double": "double",
    "float64_t": "double",
    "half": "half",
    "float16_t": "f16",
    "bfloat": "bfloat16",
    "int": "int",
    "int32_t": "int",
    "uint": "uint",
    "uint32_t": "uint",
    "bool": "bool",
    "char": "i8",
    "int8_t": "i8",
    "uchar": "u8",
    "uint8_t": "u8",
    "short": "i16",
    "int16_t": "i16",
    "ushort": "u16",
    "uint16_t": "u16",
    "long": "i64",
    "int64_t": "i64",
    "ulong": "u64",
    "uint64_t": "u64",
    # HLSL minimum-precision types have no CrossGL spelling of their own.
    "min16float": "min16float",
    "min10float": "min10float",
    "min16int": "min16int",
    "min12int": "min12int",
    "min16uint": "min16uint",
}
_SUFFIXED_VECTOR = re.compile(r"^([a-z0-9_]+?)([234])$")

# The SPIR-V generator's own vector names (``v2int``, ``v3u64``), which put the count first.
_SPIRV_VECTOR = re.compile(r"^v([234])(float|double|int|uint|bool|i64|u64|bfloat16)$")


def generic_vector_signature_name(type_value, generic_params):
    """``vecN<T>`` for a vector parameter whose element is one of ``generic_params``, else ``None``."""
    element = getattr(type_value, "element_type", None)
    size = getattr(type_value, "size", None)
    if element is None or str(size) not in {"2", "3", "4"}:
        return None
    if getattr(type_value, "rows", None) is not None:
        return None
    name = getattr(element, "name", element)
    if not isinstance(name, str) or name not in generic_params:
        return None
    return f"vec{size}<{name}>"


def vector_element_and_size(type_name):
    """``(element, size)`` for a concrete vector spelling (``ivec2``, ``int2``, ``vec2<i64>``, ``v2int``), else ``None``."""
    if not isinstance(type_name, str):
        return None
    text = type_name.strip()
    generic = _GENERIC_VECTOR.match(text)
    if generic is not None:
        return generic.group(2), int(generic.group(1))
    named = _NAMED_VECTOR.match(text)
    if named is not None and named.group(1) in _NAMED_VECTOR_ELEMENTS:
        return _NAMED_VECTOR_ELEMENTS[named.group(1)], int(named.group(2))
    spirv = _SPIRV_VECTOR.match(text)
    if spirv is not None:
        return spirv.group(2), int(spirv.group(1))
    suffixed = _SUFFIXED_VECTOR.match(text)
    if suffixed is not None and suffixed.group(1) in _SUFFIXED_VECTOR_ELEMENTS:
        return _SUFFIXED_VECTOR_ELEMENTS[suffixed.group(1)], int(suffixed.group(2))
    return None


def bind_generic_vector_parameter(expected_type, actual_type, bind):
    """Bind the element of a ``vecN<X>`` signature to a concrete vector argument's element.

    ``bind(expected_element, actual_element)`` continues deduction on the elements. Returns ``False`` when
    ``expected_type`` is not a generic vector or ``actual_type`` is not a vector of the same size.
    """
    generic = _GENERIC_VECTOR.match(str(expected_type).strip())
    if generic is None:
        return False
    actual = vector_element_and_size(str(actual_type))
    if actual is None or actual[1] != int(generic.group(1)):
        return False
    bind(generic.group(2), actual[0])
    return True


def _concrete_vector(element, size, substitutions):
    concrete = substitutions.get(element)
    if not isinstance(concrete, str):
        return None
    named = crossgl_named_vector_type(concrete, size)
    return (
        named if named is not None else VectorType(PrimitiveType(concrete), int(size))
    )


def concrete_generic_vector_type(type_value, substitutions):
    """The concrete type of a ``vecN<T>`` type node or spelling once ``T`` is substituted, else ``None``.

    Elements with a named CrossGL vector give its name (``ivec2``, ``i64vec2``); others, such as the HLSL
    minimum-precision scalars, give a ``VectorType`` node that the generator spells itself (``min16float2``).
    """
    if not substitutions:
        return None
    if isinstance(type_value, str):
        generic = _GENERIC_VECTOR.match(type_value.strip())
        if generic is None:
            return None
        return _concrete_vector(generic.group(2), generic.group(1), substitutions)
    element = getattr(type_value, "element_type", None)
    size = getattr(type_value, "size", None)
    if element is None or str(size) not in {"2", "3", "4"}:
        return None
    if getattr(type_value, "rows", None) is not None:
        return None
    return _concrete_vector(getattr(element, "name", element), size, substitutions)


def concrete_generic_vector_name(name, substitutions):
    """The constructor name for a ``vecN<T>(...)`` call once ``T`` is substituted, else ``None``."""
    concrete = concrete_generic_vector_type(name, substitutions)
    if concrete is None or isinstance(concrete, str):
        return concrete
    return f"vec{concrete.size}<{concrete.element_type.name}>"


def substituted_generic_type(generator, type_value, substitutions):
    """``type_value`` with its type parameters replaced by ``substitutions``.

    A ``vecN<T>`` node becomes its concrete vector (see :func:`concrete_generic_vector_type`); any other type goes
    through :func:`.enum_utils.substitute_generic_type_name` on the generator's spelling of it.
    """
    concrete = concrete_generic_vector_type(type_value, substitutions)
    if concrete is not None:
        return concrete
    return substitute_generic_type_name(
        generator.type_name_string(type_value), substitutions
    )


def substituted_generic_type_name(generator, type_value, substitutions):
    """The spelling of :func:`substituted_generic_type`, for the generator's local type tables."""
    concrete = substituted_generic_type(generator, type_value, substitutions)
    return (
        concrete if isinstance(concrete, str) else generator.type_name_string(concrete)
    )
