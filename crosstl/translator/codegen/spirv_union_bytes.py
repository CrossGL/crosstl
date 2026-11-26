"""Sub-word lane views of a union's storage words for :mod:`.spirv_union_storage`.

A ``u8vec4`` view (``uchar4 bytes[2]`` over a ``uint2``) is one storage word per vector, its lanes the word's bytes
from the least significant up (the ``@union_layout`` byte order is ``little_endian``). A ``u16vec2`` view
(``ushort2 halves[2]`` over a ``uint2``) is the same shape with two 16-bit lanes per word. The SPIR-V, WGSL and
Slang lowering reads and writes such a view with 32-bit word arithmetic:

- a read of an element unpacks the word: ``(uvec4(w) >> uvec4(0u, 8u, 16u, 24u)) & uvec4(255u)`` for bytes,
  ``(uvec2(w) >> uvec2(0u, 16u)) & uvec2(65535u)`` for 16-bit lanes, as the register type the target uses for the
  view (``ivec4``/``ivec2`` where narrow-integer legalization widened it, ``u8vec4``/``u16vec2`` for Slang);
- a write of an element packs its lanes into the word, so the value must be a constructor with one component per
  lane or an expression that can be evaluated once per lane;
- a write of one lane (``b.bytes[1].x = v``, ``b.halves[1][i] = v`` or the Metal importer's
  ``CrossGLMetalVectorIndex_u8vec4_set`` / ``CrossGLMetalVectorIndex_u16vec2_set`` helper) replaces that lane of
  the word and keeps the others.

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
    "LANE_KINDS",
    "SUBWORD_KINDS",
    "SUBWORD_LANES",
    "byte_decode",
    "byte_encode",
    "byte_lane_write",
    "is_byte_lane_helper",
    "is_subword_lane_helper",
    "is_u16_lane_helper",
    "lane_of",
    "packable",
    "subword_decode",
    "subword_encode",
    "subword_lane_write",
    "u16_decode",
    "u16_encode",
    "u16_lane_write",
]

# Member views lowered here: four bytes of one storage word (u8x4), or two 16-bit lanes of one word (u16x2),
# as union_layout.member_storage_kind names them.
BYTE_KINDS = frozenset({"u8x4"})
LANE_KINDS = frozenset({"u16x2"})
SUBWORD_KINDS = BYTE_KINDS | LANE_KINDS
SUBWORD_LANES = {"u8x4": 4, "u16x2": 2}

_LANE_LETTERS = {letter: index % 4 for index, letter in enumerate("xyzwrgba")}
# Per view kind: (lane count, lane width in bits, lane mask, shifts, narrow register, native register,
# constructors with one component per lane, Metal importer lane-write helper). The importer suffixes the helper
# name with _1, _2, ... when the program already uses the name (vector_types.small_vector_index_type).
_VIEW_SPECS = {
    "u8x4": (
        4,
        8,
        255,
        (0, 8, 16, 24),
        "ivec4",
        "u8vec4",
        frozenset({"u8vec4", "i8vec4", "uvec4", "ivec4"}),
        re.compile(r"CrossGLMetalVectorIndex_u8vec4(?:_[0-9]+)?_set"),
    ),
    "u16x2": (
        2,
        16,
        65535,
        (0, 16),
        "ivec2",
        "u16vec2",
        frozenset({"u16vec2", "i16vec2", "uvec2", "ivec2"}),
        re.compile(r"CrossGLMetalVectorIndex_u16vec2(?:_[0-9]+)?_set"),
    ),
}


def _call(name, arguments):
    return FunctionCallNode(IdentifierNode(name), list(arguments))


def _uint(value):
    return LiteralNode(value, PrimitiveType("uint"))


def _uvec(values, lanes):
    return _call(f"uvec{lanes}", [_uint(value) for value in values])


def _literal_int(node):
    if isinstance(node, LiteralNode) and isinstance(node.value, int):
        return node.value
    if isinstance(node, LiteralNode) and isinstance(node.value, str):
        text = node.value.rstrip("uU")
        if text.isdigit():
            return int(text)
    return None


def _is_lane_helper(name, functions, kind):
    """Whether ``name`` is the importer's lane-write helper for ``kind`` and the program defines it."""
    if not name or not _VIEW_SPECS[kind][7].fullmatch(name):
        return False
    return any(
        len(getattr(function, "parameters", None) or []) == 3
        for function in functions.get(name, ())
    )


def is_byte_lane_helper(name, functions):
    """Whether ``name`` is the importer's lane-write helper for ``u8vec4`` and the program defines it."""
    return _is_lane_helper(name, functions, "u8x4")


def is_u16_lane_helper(name, functions):
    """Whether ``name`` is the importer's lane-write helper for ``u16vec2`` and the program defines it."""
    return _is_lane_helper(name, functions, "u16x2")


def is_subword_lane_helper(name, functions):
    """Whether ``name`` is the importer's lane-write helper for a lowered sub-word view."""
    return any(_is_lane_helper(name, functions, kind) for kind in ("u8x4", "u16x2"))


def lane_of(target):
    """``(vector, lane)`` when ``target`` writes one lane of a vector (``v.x``, ``v[i]``), else ``None``."""
    if isinstance(target, ArrayAccessNode):
        return target.array_expr, target.index_expr
    if isinstance(target, MemberAccessNode) and target.member in _LANE_LETTERS:
        return target.object_expr, _uint(_LANE_LETTERS[target.member])
    if isinstance(target, SwizzleNode) and target.components in _LANE_LETTERS:
        return target.vector_expr, _uint(_LANE_LETTERS[target.components])
    return None


def _components(value, lanes):
    """The lane expressions of a constructor with one component per lane, or ``None``."""
    if isinstance(value, ConstructorNode):
        name = str(getattr(value.constructor_type, "name", value.constructor_type))
        arguments = list(value.arguments or [])
    elif isinstance(value, FunctionCallNode):
        name = call_name(value)
        arguments = list(value.arguments or [])
    else:
        return None
    constructors = next(spec[6] for spec in _VIEW_SPECS.values() if spec[0] == lanes)
    if name in constructors and len(arguments) == lanes:
        return arguments
    return None


def packable(value, lanes=4):
    """Whether a write of ``value`` to a sub-word view evaluates each part of it exactly once."""
    components = _components(value, lanes)
    return components is not None or is_pure(value)


def subword_decode(kind, storage, narrow_registers):
    """The lanes of the word ``storage`` as the target's register value for the ``kind`` view."""
    (
        lanes,
        _bits,
        mask,
        shifts,
        narrow_register,
        native_register,
        _constructors,
        _helper,
    ) = _VIEW_SPECS[kind]
    shifted = BinaryOpNode(_call(f"uvec{lanes}", [storage]), ">>", _uvec(shifts, lanes))
    masked = BinaryOpNode(shifted, "&", _call(f"uvec{lanes}", [_uint(mask)]))
    return _call(narrow_register if narrow_registers else native_register, [masked])


def byte_decode(storage, narrow_registers):
    """The four bytes of the word ``storage`` as the target's ``u8vec4`` register value."""
    return subword_decode("u8x4", storage, narrow_registers)


def u16_decode(storage, narrow_registers):
    """The two 16-bit lanes of the word ``storage`` as the target's ``u16vec2`` register value."""
    return subword_decode("u16x2", storage, narrow_registers)


def _lane(value, shift, mask):
    lane = BinaryOpNode(_call("uint", [value]), "&", _uint(mask))
    return lane if shift == 0 else BinaryOpNode(lane, "<<", _uint(shift))


def subword_encode(kind, value):
    """The storage word that holds the lanes of ``value``; the finder checked :func:`packable`."""
    lanes, _bits, mask, shifts, _narrow, _native, _constructors, _helper = _VIEW_SPECS[
        kind
    ]
    components = _components(value, lanes)
    if components is None:
        components = [
            MemberAccessNode(clone(value), letter) for letter in "xyzw"[:lanes]
        ]
    word = None
    for component, shift in zip(components, shifts):
        part = _lane(component, shift, mask)
        word = part if word is None else BinaryOpNode(word, "|", part)
    return word


def byte_encode(value):
    """The storage word that holds the four lanes of ``value``; the finder checked :func:`packable`."""
    return subword_encode("u8x4", value)


def u16_encode(value):
    """The storage word that holds the two lanes of ``value``; the finder checked :func:`packable`."""
    return subword_encode("u16x2", value)


def subword_lane_write(kind, storage, lane, value):
    """``storage`` with the lane ``lane`` replaced by the low bits of ``value``.

    A lane above the last selects the last lane, as the importer's helper does (its ``else`` branch writes the
    last lane).
    """
    lanes, bits, mask, _shifts, _narrow, _native, _constructors, _helper = _VIEW_SPECS[
        kind
    ]
    last = lanes - 1
    constant = _literal_int(lane)
    if constant is not None:
        shift = _uint(bits * min(constant, last))
    else:
        selected = _call("min", [_call("uint", [lane]), _uint(last)])
        shift = BinaryOpNode(_uint(bits), "*", selected)
    mask_node = UnaryOpNode("~", BinaryOpNode(_uint(mask), "<<", shift))
    kept = BinaryOpNode(clone(storage), "&", mask_node)
    lane_value = BinaryOpNode(
        BinaryOpNode(_call("uint", [value]), "&", _uint(mask)), "<<", clone(shift)
    )
    return BinaryOpNode(kept, "|", lane_value)


def byte_lane_write(storage, lane, value):
    """``storage`` with the byte ``lane`` (0 to 3) replaced by the low eight bits of ``value``.

    A lane above 3 selects the last lane, as the importer's helper does (its ``else`` branch writes ``.w``).
    """
    return subword_lane_write("u8x4", storage, lane, value)


def u16_lane_write(storage, lane, value):
    """``storage`` with the 16-bit lane ``lane`` (0 or 1) replaced by the low sixteen bits of ``value``.

    A lane above 1 selects the last lane, as the importer's helper does (its ``else`` branch writes ``.y``).
    """
    return subword_lane_write("u16x2", storage, lane, value)
