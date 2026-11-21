"""Static initializers for the file-scope vector and array constants that CUDA and HIP device code reads.

Device code may read a file-scope constant vector or array only from device memory, so the CUDA and HIP generators
declare it ``__device__ const`` (see :mod:`.gpu_device_globals`). A ``__device__`` variable must be initialized
statically: CUDA C++ Programming Guide, "Initialization" (dynamic initialization is not supported for ``__device__``,
``__constant__`` and ``__shared__`` variables), and hipcc follows the same rule. The generators spell vector
arithmetic through helper functions such as ``cgl_float2_mul_scalar`` and vectors through ``make_float2``; neither
is constexpr. ``const vec2 V = vec2(1.0, 2.0) * S;`` written as it reads therefore stayed a host variable, and every
kernel that named it failed to compile (nvcc: ``identifier "V" is undefined in device code``).

This module evaluates such an initializer while generating code, with the rounding of the value's own type, and
spells the result as a brace list of literals: ``__device__ const float2 V = {3.5f, 7.0f};``. It folds literals,
earlier file-scope constants, the scalar and vector constructors, unary, binary and conditional operators, swizzles,
indexing and array literals. Anything else, or an operation whose result C++ leaves undefined (signed overflow, a
division by zero, a shift by the width or more, a float outside the target range), does not fold. A vector or array
constant that neither folds nor already has a static initializer is refused, because no spelling of it compiles in
device code.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
)
from .gpu_device_globals import (
    braced_vector_constructors,
    constant_in_device_memory,
    in_device_memory,
    names_execution_space,
)

__all__ = [
    "device_constant_declaration",
    "device_variable_declaration",
    "is_device_variable",
]

_SCALAR_KINDS = {
    "float": "float",
    "f32": "float",
    "float32_t": "float",
    "double": "double",
    "f64": "double",
    "float64_t": "double",
    "int": "int",
    "i32": "int",
    "int32_t": "int",
    "uint": "uint",
    "u32": "uint",
    "uint32_t": "uint",
    "unsigned int": "uint",
    "bool": "bool",
}
_VECTOR_PREFIXES = {"": "float", "d": "double", "i": "int", "u": "uint", "b": "bool"}
_CROSSGL_VECTOR = re.compile(r"([diub]?)vec([2-4])")
_C_VECTOR = re.compile(r"(float|double|int|uint|bool)([2-4])")
_ARRAY = re.compile(r"(.+?)\[(\d*)\]")
_SWIZZLE_SETS = ("xyzw", "rgba", "stpq")
# Functional casts that are constant expressions, the only calls a static initializer may contain.
_CONSTANT_CALLEES = frozenset({"float", "double", "int", "unsigned", "bool"})
_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_INT_MIN, _INT_MAX = -(2**31), 2**31 - 1
_FLOAT_MAX = struct.unpack("<f", b"\xff\xff\x7f\x7f")[0]


class _NotConstant(Exception):
    """The initializer has no value this module can compute."""


@dataclass(frozen=True)
class _Scalar:
    kind: str
    value: object


@dataclass(frozen=True)
class _Vector:
    kind: str
    values: tuple


@dataclass(frozen=True)
class _Array:
    elements: tuple


def _type_shape(type_value):
    """``("scalar", kind)``, ``("vector", kind, n)``, ``("array", element, size)`` or None."""
    if type_value is None:
        return None
    if not isinstance(type_value, str):
        name = type(type_value).__name__
        if name == "PrimitiveType":
            kind = _SCALAR_KINDS.get(type_value.name)
            return ("scalar", kind) if kind else None
        if name == "VectorType":
            element = _type_shape(type_value.element_type)
            if element is None or element[0] != "scalar":
                return None
            return ("vector", element[1], int(type_value.size))
        if name == "ArrayType":
            element = _type_shape(type_value.element_type)
            size = type_value.size
            if isinstance(size, LiteralNode):
                size = size.value
            if element is None or (size is not None and not isinstance(size, int)):
                return None
            return ("array", element, size)
        return None
    text = " ".join(type_value.split())
    array = _ARRAY.fullmatch(text)
    if array:
        element = _type_shape(array.group(1))
        size = int(array.group(2)) if array.group(2) else None
        return ("array", element, size) if element else None
    if text in _SCALAR_KINDS:
        return ("scalar", _SCALAR_KINDS[text])
    vector = _CROSSGL_VECTOR.fullmatch(text)
    if vector:
        return ("vector", _VECTOR_PREFIXES[vector.group(1)], int(vector.group(2)))
    vector = _C_VECTOR.fullmatch(text)
    if vector:
        return ("vector", vector.group(1), int(vector.group(2)))
    return None


def _round(kind, value):
    """``value`` as the C++ type ``kind`` holds it, or raise when C++ leaves the result undefined."""
    if kind == "bool":
        return bool(value)
    if kind in ("int", "uint"):
        if isinstance(value, float):
            if not math.isfinite(value):
                raise _NotConstant
            value = math.trunc(value)
            if kind == "uint" and value < 0:
                raise _NotConstant
        value = int(value)
        if kind == "uint":
            return value % 2**32
        if not _INT_MIN <= value <= _INT_MAX:
            raise _NotConstant
        return value
    value = float(value)
    if not math.isfinite(value):
        raise _NotConstant
    if kind == "float":
        if abs(value) > _FLOAT_MAX:
            raise _NotConstant
        value = struct.unpack("<f", struct.pack("<f", value))[0]
    return value


def _convert(value, kind):
    if isinstance(value, _Scalar):
        if value.kind == "int" and kind == "uint":
            return _Scalar(kind, value.value % 2**32)
        return _Scalar(kind, _round(kind, value.value))
    if isinstance(value, _Vector):
        return _Vector(
            kind,
            tuple(_convert(_Scalar(value.kind, v), kind).value for v in value.values),
        )
    raise _NotConstant


def _common_kind(left, right):
    kinds = {left, right}
    if "bool" in kinds:
        return "bool" if kinds == {"bool"} else None
    for kind in ("double", "float", "uint"):
        if kind in kinds:
            return kind
    return "int"


def _arithmetic(operator, kind, a, b):
    if kind == "bool":
        if operator == "&&":
            return a and b
        if operator == "||":
            return a or b
        if operator in ("==", "!="):
            return (a == b) == (operator == "==")
        raise _NotConstant
    if operator in ("+", "-", "*"):
        return {"+": a + b, "-": a - b, "*": a * b}[operator]
    if operator == "/":
        if b == 0:
            raise _NotConstant
        if kind in ("float", "double"):
            return a / b
        return abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
    if kind in ("float", "double"):
        raise _NotConstant
    if operator == "%":
        if b == 0:
            raise _NotConstant
        return a - b * _arithmetic("/", kind, a, b)
    if operator in ("&", "|", "^"):
        return {"&": a & b, "|": a | b, "^": a ^ b}[operator]
    if operator in ("<<", ">>"):
        if not 0 <= b < 32 or (operator == "<<" and a < 0):
            raise _NotConstant
        return a << b if operator == "<<" else a >> b
    raise _NotConstant


_COMPARISONS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}


def _binary(operator, left, right):
    if isinstance(left, _Array) or isinstance(right, _Array):
        raise _NotConstant
    if operator in ("<<", ">>"):
        # A shift has the type of its promoted left operand (C++ [expr.shift]), not the common type.
        if left.kind not in ("int", "uint") or right.kind not in ("int", "uint"):
            raise _NotConstant
        if not isinstance(right, _Scalar):
            right = _Vector(left.kind, right.values)
        else:
            right = _Scalar(left.kind, right.value)
    kind = _common_kind(left.kind, right.kind)
    if kind is None:
        raise _NotConstant
    if operator in _COMPARISONS and kind != "bool":
        if not (isinstance(left, _Scalar) and isinstance(right, _Scalar)):
            raise _NotConstant
        a, b = _convert(left, kind).value, _convert(right, kind).value
        return _Scalar("bool", _COMPARISONS[operator](a, b))
    left, right = _convert(left, kind), _convert(right, kind)
    if isinstance(left, _Scalar) and isinstance(right, _Scalar):
        return _Scalar(
            kind, _round(kind, _arithmetic(operator, kind, left.value, right.value))
        )
    if operator in _COMPARISONS or operator in ("&&", "||"):
        raise _NotConstant
    lefts = left.values if isinstance(left, _Vector) else None
    rights = right.values if isinstance(right, _Vector) else None
    count = len(lefts or rights)
    if lefts is not None and rights is not None and len(lefts) != len(rights):
        raise _NotConstant
    lefts = lefts or (left.value,) * count
    rights = rights or (right.value,) * count
    return _Vector(
        kind,
        tuple(
            _round(kind, _arithmetic(operator, kind, a, b))
            for a, b in zip(lefts, rights)
        ),
    )


def _unary(operator, operand):
    if not isinstance(operand, (_Scalar, _Vector)):
        raise _NotConstant
    values = operand.values if isinstance(operand, _Vector) else (operand.value,)
    kind = operand.kind
    if operator == "+" and kind != "bool":
        result = values
    elif operator == "-" and kind != "bool":
        result = tuple(_round(kind, -v) for v in values)
    elif operator == "!" and kind == "bool" and isinstance(operand, _Scalar):
        result = (not values[0],)
    elif operator == "~" and kind in ("int", "uint"):
        result = tuple(_round(kind, ~v) for v in values)
    else:
        raise _NotConstant
    return (
        _Vector(kind, result)
        if isinstance(operand, _Vector)
        else _Scalar(kind, result[0])
    )


def _components(value):
    if isinstance(value, _Scalar):
        return [value]
    if isinstance(value, _Vector):
        return [_Scalar(value.kind, v) for v in value.values]
    raise _NotConstant


def _construct(shape, arguments):
    if shape[0] == "scalar":
        if len(arguments) != 1 or not isinstance(arguments[0], _Scalar):
            raise _NotConstant
        return _convert(arguments[0], shape[1])
    _, kind, size = shape
    parts = [part for argument in arguments for part in _components(argument)]
    if len(parts) == 1 and isinstance(arguments[0], _Scalar):
        parts = parts * size
    if len(parts) != size:
        raise _NotConstant
    return _Vector(kind, tuple(_convert(part, kind).value for part in parts))


def _swizzle(value, member):
    if not isinstance(value, _Vector) or not member:
        raise _NotConstant
    for letters in _SWIZZLE_SETS:
        if all(letter in letters[: len(value.values)] for letter in member):
            picked = tuple(value.values[letters.index(letter)] for letter in member)
            if len(picked) == 1:
                return _Scalar(value.kind, picked[0])
            if len(picked) > 4:
                raise _NotConstant
            return _Vector(value.kind, picked)
    raise _NotConstant


def _index(value, index):
    if not isinstance(index, _Scalar) or index.kind not in ("int", "uint"):
        raise _NotConstant
    items = (
        value.elements
        if isinstance(value, _Array)
        else _components(value) if isinstance(value, _Vector) else None
    )
    if items is None or not 0 <= index.value < len(items):
        raise _NotConstant
    return items[index.value]


def _callee_name(node):
    function = getattr(node, "function", None)
    if isinstance(function, IdentifierNode):
        return function.name
    if isinstance(function, str):
        return function
    return getattr(node, "name", None) if function is None else None


class _Folder:
    def __init__(self, constants):
        self.constants = constants

    def fold(self, node):
        if isinstance(node, LiteralNode):
            shape = _type_shape(node.literal_type)
            if shape is None or shape[0] != "scalar" or isinstance(node.value, str):
                raise _NotConstant
            return _Scalar(shape[1], _round(shape[1], node.value))
        if isinstance(node, IdentifierNode):
            value = self.constants.get(node.name)
            if value is None:
                raise _NotConstant
            return value
        if isinstance(node, UnaryOpNode) and not node.is_postfix:
            return _unary(node.operator, self.fold(node.operand))
        if isinstance(node, BinaryOpNode):
            return _binary(node.operator, self.fold(node.left), self.fold(node.right))
        if isinstance(node, TernaryOpNode):
            condition = self.fold(node.condition)
            if not isinstance(condition, _Scalar) or condition.kind != "bool":
                raise _NotConstant
            return self.fold(node.true_expr if condition.value else node.false_expr)
        if isinstance(node, MemberAccessNode):
            return _swizzle(self.fold(node.object_expr), node.member)
        if isinstance(node, SwizzleNode):
            return _swizzle(self.fold(node.vector_expr), node.components)
        if isinstance(node, ArrayAccessNode):
            return _index(self.fold(node.array_expr), self.fold(node.index_expr))
        if isinstance(node, ArrayLiteralNode):
            return _Array(tuple(self.fold(element) for element in node.elements))
        if isinstance(node, FunctionCallNode):
            shape = _type_shape(_callee_name(node) or "")
            if shape is None or shape[0] == "array":
                raise _NotConstant
            arguments = getattr(node, "arguments", None) or getattr(node, "args", [])
            return _construct(shape, [self.fold(argument) for argument in arguments])
        raise _NotConstant


def _conform(value, shape):
    """``value`` converted to the declared ``shape``, or raise when it does not have that shape."""
    if shape[0] == "scalar":
        if not isinstance(value, _Scalar):
            raise _NotConstant
        return _convert(value, shape[1])
    if shape[0] == "vector":
        if not isinstance(value, _Vector) or len(value.values) != shape[2]:
            raise _NotConstant
        return _convert(value, shape[1])
    _, element, size = shape
    if not isinstance(value, _Array) or (
        size is not None and len(value.elements) > size
    ):
        raise _NotConstant
    return _Array(tuple(_conform(item, element) for item in value.elements))


def _float_literal(value):
    for digits in range(1, 18):
        text = f"{value:.{digits}g}"
        if struct.unpack("<f", struct.pack("<f", float(text)))[0] == value:
            break
    if not any(mark in text for mark in ".en"):
        text += ".0"
    return f"{text}f"


def _scalar_literal(kind, value):
    if kind == "bool":
        return "true" if value else "false"
    if kind == "uint":
        return f"{value}u"
    if kind == "int":
        return "(-2147483647 - 1)" if value == _INT_MIN else str(value)
    if kind == "float":
        return _float_literal(value)
    return repr(value)


def _literal(value):
    if isinstance(value, _Scalar):
        return _scalar_literal(value.kind, value.value)
    if isinstance(value, _Vector):
        return (
            "{" + ", ".join(_scalar_literal(value.kind, v) for v in value.values) + "}"
        )
    return "{" + ", ".join(_literal(element) for element in value.elements) + "}"


def _has_dynamic_call(initializer):
    return any(name not in _CONSTANT_CALLEES for name in _CALL.findall(initializer))


def _uses_vector_kind(shape, kind):
    if shape[0] == "vector":
        return shape[1] == kind
    if shape[0] == "array":
        return _uses_vector_kind(shape[1], kind)
    return False


def device_constant_declaration(codegen, name, var_type, value, emitted, refusal):
    """The file-scope declaration of the constant ``name``, placed where device code reads it.

    ``var_type`` is the CrossGL type, ``value`` the CrossGL initializer and ``emitted`` the generator's own
    ``const T name = ...`` declaration. A scalar, matrix or structure keeps ``emitted`` as
    :func:`.gpu_device_globals.constant_in_device_memory` places it; a scalar is also remembered so that later
    vector and array initializers can fold it. ``refusal`` is the generator's refusal class.
    """
    constants = getattr(codegen, "folded_constants", None)
    if constants is None:
        constants = codegen.folded_constants = {}
    shape = _type_shape(var_type)
    folded = None
    if shape is not None and value is not None:
        try:
            folded = _conform(_Folder(constants).fold(value), shape)
        except _NotConstant:
            folded = None
    constants.pop(name, None)
    if folded is not None:
        constants[name] = folded
    placed = constant_in_device_memory(emitted)
    if shape is None or shape[0] == "scalar" or value is None:
        return placed
    head, _, initializer = emitted.partition(" = ")
    if folded is not None and not _uses_vector_kind(shape, "bool"):
        return in_device_memory(f"{head} = {_literal(folded)}")
    if not names_execution_space(placed) or _has_dynamic_call(initializer):
        raise refusal(
            f"File-scope constant '{name}' needs a static initializer in device memory, "
            f"and CrossTL cannot evaluate '{initializer}' while generating code"
        )
    return placed


def device_variable_declaration(codegen, name, var_type, value, emitted, refusal):
    """The declaration of the file-scope ``device`` variable ``name`` with a static initializer.

    ``emitted`` is the generator's ``__device__ T name = ...`` declaration. The initializer folds as a constant's
    does; one that does not fold keeps its text when it is static already, with vector constructors written as
    brace lists, and is refused otherwise.
    """
    if value is None:
        return emitted
    shape = _type_shape(var_type)
    head, _, initializer = emitted.partition(" = ")
    if shape is not None:
        try:
            folded = _conform(
                _Folder(getattr(codegen, "folded_constants", {})).fold(value), shape
            )
        except _NotConstant:
            folded = None
        if folded is not None and not _uses_vector_kind(shape, "bool"):
            return f"{head} = {_literal(folded)}"
    braced = braced_vector_constructors(initializer)
    if _has_dynamic_call(braced):
        raise refusal(
            f"Device variable '{name}' needs a static initializer, "
            f"and CrossTL cannot evaluate '{initializer}' while generating code"
        )
    return f"{head} = {braced}"


def is_device_variable(node) -> bool:
    """Whether a CrossGL variable carries the ``device`` qualifier of a program-scope device-memory variable."""
    return any(
        str(qualifier).lower() == "device"
        for qualifier in getattr(node, "qualifiers", None) or []
    )
