"""Typed and value-deduced byte-address buffer accesses, imported as CrossGL word accesses.

CrossGL's byte-address accesses move 32-bit words: ``buffer_load(b, a)`` to ``buffer_load4`` return ``uint`` to
``uvec4``, and ``buffer_store(b, a, v)`` to ``buffer_store4`` store ``uint`` words. HLSL has two more forms:

- the templated ``Load<T>(a)`` and ``Store<T>(a, v)`` (HLSL 2018, Shader Model 6.2 for 16-bit ``T``), which move
  ``sizeof(T)`` bytes as they are, with no value conversion;
- ``Store(a, v)`` without a template argument, which deduces ``T`` from ``v``: DXC stores the bits of a ``float``
  value (``bufferStore.f32``), every component of a ``float4`` value, and ``1`` for ``true``. ``Store2`` to
  ``Store4`` take ``uint`` vectors, so their values convert numerically.

Dropping the template argument, as the importer did, turned ``Load<float>(a)`` into a numeric conversion of the
word and ``Load<float4>(a)`` into one word, so every target read different values. This module writes a 32-bit
scalar or vector ``T`` of ``float``, ``int``, ``uint`` or ``bool`` components as word accesses of ``T``'s width
plus a bit cast: ``Load<float4>(a)`` is ``uintBitsToFloat(buffer_load4(b, a))``, ``Store(a, f)`` for a ``float f``
is ``buffer_store(b, a, floatBitsToUint(f))``. A ``bool`` is stored as ``0u`` or ``1u`` and loaded as the word
compared with zero, as DXC does. Other ``T`` (16- or 64-bit components, matrices, structures, arrays) and stored
values whose type is not known are refused.
"""

from __future__ import annotations

import re

from ..common_ast import BinaryOpNode, FunctionCallNode, TernaryOpNode, UnaryOpNode

__all__ = ["byte_address_method_expression", "word_shape"]

_FAMILIES = {
    "float": "float",
    "int": "int",
    "uint": "uint",
    "bool": "bool",
    "dword": "uint",
}
_VECTOR = re.compile(r"(float|int|uint|bool|dword)([1-4])?")
_TEMPLATE_VECTOR = re.compile(
    r"vector\s*<\s*(float|int|uint|bool|dword)\s*,\s*([1-4])\s*>"
)
_PREFIXES = {"float": "", "int": "i", "uint": "u", "bool": "b"}


def word_shape(raw_type):
    """``(component family, lanes)`` of a 32-bit scalar or vector HLSL type, or ``None``."""
    text = str(raw_type or "").strip()
    match = _VECTOR.fullmatch(text) or _TEMPLATE_VECTOR.fullmatch(text)
    if match is None:
        return None
    return _FAMILIES[match.group(1)], int(match.group(2) or 1)


# HLSL intrinsics whose result shape the importer does not track, for deducing a stored value's type.
_BITCASTS = {"asuint": "uint", "asint": "int", "asfloat": "float"}
_UINT_LANES = frozenset({"f32tof16", "countbits"})
_FLOAT_LANES = frozenset({"f16tof32"})
_UINT_SCALARS = frozenset(
    {
        "WaveGetLaneIndex",
        "WaveGetLaneCount",
        "WaveActiveCountBits",
        "WavePrefixCountBits",
    }
)
_FIRST_ARGUMENT = frozenset(
    {
        "abs",
        "reversebits",
        "saturate",
        "WaveReadLaneFirst",
        "WaveReadLaneAt",
        "WaveActiveSum",
        "WaveActiveProduct",
        "WaveActiveMin",
        "WaveActiveMax",
        "WaveActiveBitAnd",
        "WaveActiveBitOr",
        "WaveActiveBitXor",
        "WavePrefixSum",
        "WavePrefixProduct",
        "QuadReadAcrossX",
        "QuadReadAcrossY",
        "QuadReadAcrossDiagonal",
        "QuadReadLaneAt",
    }
)
_SAME_ARGUMENTS = frozenset({"min", "max", "clamp"})
_COMPARISONS = frozenset({"<", "<=", ">", ">=", "==", "!=", "&&", "||"})
_ARITHMETIC = frozenset({"+", "-", "*", "/", "%", "&", "|", "^"})
_SHIFTS = frozenset({"<<", ">>"})
# Usual arithmetic conversions between 32-bit families: bool promotes to int, int converts to uint.
_RANK = {"bool": 0, "int": 1, "uint": 2, "float": 3}


def _combined_lanes(left, right):
    if left == right or right == 1:
        return left
    return right if left == 1 else None


def _intrinsic_shape(importer, name, arguments):
    shapes = [_value_shape(importer, argument) for argument in arguments]
    if name in _UINT_SCALARS:
        return ("uint", 1)
    if not shapes or shapes[0] is None:
        return None
    lanes = shapes[0][1]
    if name in _BITCASTS and len(shapes) == 1:
        return (_BITCASTS[name], lanes)
    if name in _UINT_LANES and len(shapes) == 1:
        return ("uint", lanes)
    if name in _FLOAT_LANES and len(shapes) == 1:
        return ("float", lanes)
    if name in _FIRST_ARGUMENT:
        return shapes[0]
    if name in _SAME_ARGUMENTS and all(shape == shapes[0] for shape in shapes):
        return shapes[0]
    return None


def _binary_shape(importer, expression):
    operator = importer.binary_operator_text(expression.op)
    left = _value_shape(importer, expression.left)
    right = _value_shape(importer, expression.right)
    if left is None or right is None:
        return None
    lanes = _combined_lanes(left[1], right[1])
    if lanes is None:
        return None
    if operator in _COMPARISONS:
        return ("bool", lanes)
    if operator in _SHIFTS:
        family = "int" if left[0] == "bool" else left[0]
        return None if family == "float" else (family, lanes)
    if operator in _ARITHMETIC:
        family = max(left[0], right[0], key=_RANK.__getitem__)
        return ("int" if family == "bool" else family, lanes)
    return None


def _value_shape(importer, expression):
    """``(family, lanes)`` of a stored value, for ``Store(a, v)``, which moves the bytes of ``v``'s type."""
    shape = word_shape(
        importer.resolve_integer_alias(importer.expression_value_raw_type(expression))
    )
    if shape is not None:
        return shape
    if isinstance(expression, FunctionCallNode) and isinstance(expression.name, str):
        name = importer.normalize_hlsl_intrinsic_name(expression.name)
        if name in getattr(importer, "source_functions", {}):
            return None
        return _intrinsic_shape(importer, name, list(expression.args or []))
    if isinstance(expression, BinaryOpNode):
        return _binary_shape(importer, expression)
    if isinstance(expression, TernaryOpNode):
        true_shape = _value_shape(importer, expression.true_expr)
        false_shape = _value_shape(importer, expression.false_expr)
        return true_shape if true_shape == false_shape else None
    if isinstance(expression, UnaryOpNode):
        if importer.binary_operator_text(expression.op) == "!":
            operand = _value_shape(importer, expression.operand)
            return None if operand is None else ("bool", operand[1])
        return _value_shape(importer, expression.operand)
    return None


def _crossgl_type(family, lanes):
    return family if lanes == 1 else f"{_PREFIXES[family]}vec{lanes}"


def _suffix(lanes):
    return "" if lanes == 1 else str(lanes)


def _typed_load(shape, buffer, address):
    family, lanes = shape
    words = f"buffer_load{_suffix(lanes)}({buffer}, {address})"
    if family == "uint":
        return words
    if family == "float":
        return f"uintBitsToFloat({words})"
    # int keeps the bits; bool is the word compared with zero.
    return f"{_crossgl_type(family, lanes)}({words})"


def _word_store(shape, buffer, address, value, value_shape):
    family, lanes = shape
    converted = (
        value if value_shape == shape else f"{_crossgl_type(family, lanes)}({value})"
    )
    if family == "uint":
        payload = converted
    elif family == "float":
        payload = f"floatBitsToUint({converted})"
    else:
        payload = f"{_crossgl_type('uint', lanes)}({converted})"
    return f"buffer_store{_suffix(lanes)}({buffer}, {address}, {payload})"


def byte_address_method_expression(
    importer, member, buffer, rendered_args, raw_args, resource_type
):
    """CrossGL for ``Load<T>``, ``Store<T>`` or a value-deduced ``Store``; ``None`` for the plain word forms.

    ``importer`` is the DirectX importer, which resolves types and records refusals.
    """
    if not importer.is_byte_address_buffer_type(resource_type):
        return None
    base = importer.templated_method_base(member)
    if base not in ("Load", "Store"):
        return None
    template = importer.first_template_argument(member) if "<" in member else None
    if template is not None:
        template = importer.resolve_integer_alias(template)
    if base == "Load":
        if template is None:
            return None
        shape = word_shape(template)
        if shape is None:
            return importer.refuse_import(
                f"DirectX ByteAddressBuffer Load<{template}>: only 32-bit scalar and vector types of float, int, "
                "uint or bool components are imported",
                f"buffer_load({buffer}, 0u)",
            )
        if len(rendered_args) != 1:
            if shape == ("uint", 1):
                return None
            return importer.refuse_import(
                f"DirectX ByteAddressBuffer Load<{template}> with a tiled-resource status output"
            )
        return _typed_load(shape, buffer, rendered_args[0])

    if len(rendered_args) != 2:
        return None
    value_shape = _value_shape(importer, raw_args[1])
    shape = word_shape(template) if template is not None else value_shape
    if shape == ("uint", 1) and value_shape == shape:
        return None
    if shape is None:
        if template is not None:
            described = f"Store<{template}>"
        else:
            value_type = importer.expression_value_raw_type(raw_args[1])
            described = f"Store of a {value_type or 'value of unknown type'}"
        return importer.refuse_import(
            f"DirectX ByteAddressBuffer {described}: Store moves the bytes of its value's type, and only 32-bit "
            "scalar and vector types of float, int, uint or bool components are imported",
            f"buffer_store({buffer}, {rendered_args[0]}, 0u)",
        )
    return _word_store(shape, buffer, rendered_args[0], rendered_args[1], value_shape)
