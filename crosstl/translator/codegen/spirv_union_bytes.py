"""Byte views of a union's storage words for :mod:`.spirv_union_storage`.

A ``u8vec4`` view (``uchar4 bytes[2]`` over a ``uint2``) is one storage word per vector, its lanes the word's bytes
from the least significant up (the ``@union_layout`` byte order is ``little_endian``). The SPIR-V, WGSL and Slang
lowering reads and writes such a view with 32-bit word arithmetic:

- a read of an element unpacks the word: ``(uvec4(w) >> uvec4(0u, 8u, 16u, 24u)) & uvec4(255u)``, as the
  register type the target uses for ``u8vec4`` (``ivec4`` where narrow-integer legalization widened it, ``u8vec4``
  for Slang);
- a write of an element packs four bytes into the word, so the value must be a four-component constructor or an
  expression that can be evaluated once per lane;
- a write of one lane (``b.bytes[1].x = v``, ``b.bytes[1][i] = v`` or the Metal importer's
  ``CrossGLMetalVectorIndex_u8vec4_set`` helper) replaces that byte of the word and keeps the other three.

The finder refuses what these forms cannot express exactly; this module only builds the expressions.
"""

from __future__ import annotations

import re

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PrimitiveType,
    SwizzleNode,
    UnaryOpNode,
)
from ..semantics.hub_types import call_name
from ..semantics.rewrite import clone, is_pure

__all__ = [
    "BYTE_KINDS",
    "byte_decode",
    "byte_encode",
    "byte_lane_write",
    "is_byte_lane_helper",
    "lane_of",
    "packable",
]

# Member views lowered here: four bytes of one storage word, as union_layout.member_storage_kind names them.
BYTE_KINDS = frozenset({"u8x4"})

_LANE_LETTERS = {letter: index % 4 for index, letter in enumerate("xyzwrgba")}
_SHIFTS = (0, 8, 16, 24)
# The Metal importer writes an assignment to one lane of a small vector as a call to a generated helper whose name
# it suffixes with _1, _2, ... when the program already uses the name (vector_types.small_vector_index_type).
_LANE_HELPER = re.compile(r"CrossGLMetalVectorIndex_u8vec4(?:_[0-9]+)?_set")
_VECTOR_CONSTRUCTORS = frozenset({"u8vec4", "i8vec4", "uvec4", "ivec4"})


def _call(name, arguments):
    return FunctionCallNode(IdentifierNode(name), list(arguments))


def _uint(value):
    return LiteralNode(value, PrimitiveType("uint"))


def _uvec4(values):
    return _call("uvec4", [_uint(value) for value in values])


def _literal_int(node):
    if isinstance(node, LiteralNode) and isinstance(node.value, int):
        return node.value
    if isinstance(node, LiteralNode) and isinstance(node.value, str):
        text = node.value.rstrip("uU")
        if text.isdigit():
            return int(text)
    return None


def is_byte_lane_helper(name, functions):
    """Whether ``name`` is the importer's lane-write helper for ``u8vec4`` and the program defines it."""
    if not name or not _LANE_HELPER.fullmatch(name):
        return False
    return any(
        len(getattr(function, "parameters", None) or []) == 3
        for function in functions.get(name, ())
    )


def lane_of(target):
    """``(vector, lane)`` when ``target`` writes one lane of a vector (``v.x``, ``v[i]``), else ``None``."""
    if isinstance(target, ArrayAccessNode):
        return target.array_expr, target.index_expr
    if isinstance(target, MemberAccessNode) and target.member in _LANE_LETTERS:
        return target.object_expr, _uint(_LANE_LETTERS[target.member])
    if isinstance(target, SwizzleNode) and target.components in _LANE_LETTERS:
        return target.vector_expr, _uint(_LANE_LETTERS[target.components])
    return None


def _components(value):
    """The four lane expressions of a ``u8vec4`` constructor, or ``None``."""
    if isinstance(value, ConstructorNode):
        name = str(getattr(value.constructor_type, "name", value.constructor_type))
        arguments = list(value.arguments or [])
    elif isinstance(value, FunctionCallNode):
        name = call_name(value)
        arguments = list(value.arguments or [])
    else:
        return None
    if name in _VECTOR_CONSTRUCTORS and len(arguments) == 4:
        return arguments
    return None


def packable(value):
    """Whether a write of ``value`` to a byte view evaluates each part of it exactly once."""
    components = _components(value)
    return components is not None or is_pure(value)


def byte_decode(storage, narrow_registers):
    """The four bytes of the word ``storage`` as the target's ``u8vec4`` register value."""
    shifted = BinaryOpNode(_call("uvec4", [storage]), ">>", _uvec4(_SHIFTS))
    lanes = BinaryOpNode(shifted, "&", _call("uvec4", [_uint(255)]))
    return _call("ivec4" if narrow_registers else "u8vec4", [lanes])


def _byte(value, shift):
    byte = BinaryOpNode(_call("uint", [value]), "&", _uint(255))
    return byte if shift == 0 else BinaryOpNode(byte, "<<", _uint(shift))


def byte_encode(value):
    """The storage word that holds the four lanes of ``value``; the finder checked :func:`packable`."""
    components = _components(value)
    if components is None:
        components = [MemberAccessNode(clone(value), letter) for letter in "xyzw"]
    word = None
    for component, shift in zip(components, _SHIFTS):
        part = _byte(component, shift)
        word = part if word is None else BinaryOpNode(word, "|", part)
    return word


def byte_lane_write(storage, lane, value):
    """``storage`` with the byte ``lane`` (0 to 3) replaced by the low eight bits of ``value``.

    A lane above 3 selects the last lane, as the importer's helper does (its ``else`` branch writes ``.w``).
    """
    constant = _literal_int(lane)
    if constant is not None:
        shift = _uint(8 * min(constant, 3))
    else:
        selected = _call("min", [_call("uint", [lane]), _uint(3)])
        shift = BinaryOpNode(_uint(8), "*", selected)
    mask = UnaryOpNode("~", BinaryOpNode(_uint(255), "<<", shift))
    kept = BinaryOpNode(clone(storage), "&", mask)
    byte = BinaryOpNode(
        BinaryOpNode(_call("uint", [value]), "&", _uint(255)), "<<", clone(shift)
    )
    return BinaryOpNode(kept, "|", byte)
