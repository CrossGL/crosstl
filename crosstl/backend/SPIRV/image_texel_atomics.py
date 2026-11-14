"""Atomics through ``OpImageTexelPointer``, imported as CrossGL image atomics.

SPIR-V reaches a storage-image texel atomically in two steps. ``OpImageTexelPointer`` forms a pointer to the texel
(storage class ``Image``), and an ``OpAtomic*`` instruction operates through it (SPIR-V 3.37.18 and 3.37.10). CrossGL
has one call per operation that takes the image and the coordinate, as GLSL does (``imageAtomicAdd`` and its kin,
GLSL 4.60 section 8.12). The importer writes the pointer as ``spirvImageTexelPointer(image, coordinate, sample)``,
and :func:`fold_image_texel_atomic` turns an atomic through it into the image atomic:

* ``OpAtomicIAdd``, ``OpAtomicISub``, ``OpAtomicIIncrement`` and ``OpAtomicIDecrement`` become ``imageAtomicAdd``
  (with the value negated, ``1`` or ``-1``, as for other pointers);
* ``OpAtomicAnd``, ``OpAtomicOr``, ``OpAtomicXor`` and ``OpAtomicExchange`` become ``imageAtomicAnd``,
  ``imageAtomicOr``, ``imageAtomicXor`` and ``imageAtomicExchange``;
* ``OpAtomicCompareExchange`` and ``OpAtomicCompareExchangeWeak`` become ``imageAtomicCompSwap(image, coordinate,
  comparator, value)``;
* ``OpAtomicSMin``, ``OpAtomicSMax``, ``OpAtomicUMin`` and ``OpAtomicUMax`` become ``imageAtomicMin`` and
  ``imageAtomicMax`` when the result type has the opcode's signedness. The image atomic compares as the image's
  texel type, which is the result type (VUID-StandaloneSpirv-OpImageTexelPointer-04658 ties the pointer's type to
  the image's Sampled Type), so ``OpAtomicUMin`` on a signed texel has no image-atomic spelling.

An integer literal among the data operands of a ``uint`` atomic is spelled unsigned (``3u``), since the image
atomic takes data of the texel's kind and CrossGL reads an unsuffixed literal as ``int``.

The pointer is left as it is when its Sample operand is not the constant 0, which only a multisampled image may
take, and so is an atomic load or store through it, which CrossGL has no image form of. No target defines
``spirvImageTexelPointer``, so :func:`~crosstl.translator.semantics.spirv_instructions.refuse_spirv_instructions`
refuses what is left before generation.
"""

from __future__ import annotations

import re

__all__ = ["IMAGE_TEXEL_POINTER", "fold_image_texel_atomic"]

IMAGE_TEXEL_POINTER = "spirvImageTexelPointer"

_IMAGE_ATOMICS = {
    "atomicAdd": "imageAtomicAdd",
    "atomicMin": "imageAtomicMin",
    "atomicMax": "imageAtomicMax",
    "atomicAnd": "imageAtomicAnd",
    "atomicOr": "imageAtomicOr",
    "atomicXor": "imageAtomicXor",
    "atomicExchange": "imageAtomicExchange",
    "atomicCompSwap": "imageAtomicCompSwap",
}
_SIGNED_OPCODES = frozenset({"OpAtomicSMin", "OpAtomicSMax"})
_UNSIGNED_OPCODES = frozenset({"OpAtomicUMin", "OpAtomicUMax"})
_SIGNED_TYPES = frozenset({"int", "int64_t", "long"})
_UNSIGNED_TYPES = frozenset({"uint", "uint64_t", "ulong"})


def _is_zero(expression) -> bool:
    text = getattr(expression, "value", expression)
    if not isinstance(text, (str, int)):
        return False
    text = str(text).strip().lower().rstrip("u")
    try:
        return int(text, 0) == 0
    except ValueError:
        return False


_INTEGER_LITERAL = re.compile(r"(?:0[xX][0-9A-Fa-f]+|[0-9]+)")


def _unsigned_operand(value):
    """``value``, a data operand of an atomic on a ``uint`` texel, with integer literals spelled unsigned.

    SPIR-V types every data operand as the result type (SPIR-V 3.37.18), but the importer writes a ``uint`` constant
    without a suffix, which CrossGL reads as ``int``; image atomics take data of the image's texel kind exactly.
    """
    if isinstance(value, str) and _INTEGER_LITERAL.fullmatch(value.strip()):
        return f"{value.strip()}u"
    if getattr(value, "op", None) == "-" and hasattr(value, "operand"):
        operand = _unsigned_operand(value.operand)
        if operand is not value.operand:
            return type(value)("-", operand)
    return value


def _signedness_matches(opcode, result_type) -> bool:
    if opcode in _SIGNED_OPCODES:
        return result_type in _SIGNED_TYPES
    if opcode in _UNSIGNED_OPCODES:
        return result_type in _UNSIGNED_TYPES
    return True


def fold_image_texel_atomic(call, opcode, result_type):
    """Return ``call``, an atomic of SPIR-V ``opcode``, as an image atomic when it goes through a texel pointer.

    ``call`` is the importer's ``FunctionCallNode`` (``atomicAdd(pointer, value)`` and its kin), ``result_type`` the
    CrossGL name of the instruction's result type. Any other call is returned unchanged.
    """
    arguments = list(getattr(call, "args", None) or ())
    image_atomic = _IMAGE_ATOMICS.get(getattr(call, "name", None))
    if image_atomic is None or not arguments:
        return call
    pointer = arguments[0]
    if getattr(pointer, "name", None) != IMAGE_TEXEL_POINTER:
        return call
    pointer_arguments = list(getattr(pointer, "args", None) or ())
    if len(pointer_arguments) != 3 or not _is_zero(pointer_arguments[2]):
        return call
    if not _signedness_matches(opcode, str(result_type or "")):
        return call
    image, coordinate, _sample = pointer_arguments
    values = arguments[1:]
    if result_type == "uint":
        values = [_unsigned_operand(value) for value in values]
    return type(call)(image_atomic, [image, coordinate, *values])
