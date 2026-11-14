"""Exact-width integer types and integer promotion in the Rust generator.

The CrossGL IR names its 8- and 16-bit integers ``int8``/``uint8``/``int16``/``uint16`` (with the ``_t`` and
``uchar``/``ushort`` spellings importers write), and computes integer arithmetic on them after integer promotion: an
operand narrower than 32 bits computes as ``int`` (:func:`~crosstl.translator.semantics.hub_types.promote`), as C,
C++, CUDA and MSL do.

The Rust generator knew only ``i8``/``u8``/``i16``/``u16``. An IR spelling was written as it stands (``let a: int16 =
int16(30000);``, which names nothing in Rust), and ``u8``/``i8`` operands crashed the operand-type ranking with an
untyped ``KeyError`` (defect X3). Rust also has no integer promotion: ``a + b`` on two ``i16`` values computes in
``i16`` and overflows where the IR's ``int`` arithmetic does not.

:data:`RUST_EXACT_WIDTH_INTEGER_TYPES` maps every IR spelling to its Rust type, :data:`RUST_SCALAR_RANKS` ranks every
Rust scalar, and :func:`promoted_operand_type` widens a narrow integer operand of an arithmetic, bitwise or shift
operator to ``i32`` before the operands are unified, so ``a + b`` is written ``((a as i32) + (b as i32))``.
"""

from __future__ import annotations

__all__ = [
    "INTEGER_PROMOTION_OPERATORS",
    "RUST_EXACT_WIDTH_INTEGER_TYPES",
    "RUST_SCALAR_RANKS",
    "promoted_operand_type",
]

# IR spelling -> Rust type (Rust Reference, "Integer types").
RUST_EXACT_WIDTH_INTEGER_TYPES = {
    "int8": "i8",
    "int8_t": "i8",
    "uint8": "u8",
    "uint8_t": "u8",
    "uchar": "u8",
    "int16": "i16",
    "int16_t": "i16",
    "uint16": "u16",
    "uint16_t": "u16",
    "int32": "i32",
    "int32_t": "i32",
    "uint32": "u32",
    "uint32_t": "u32",
    "int64": "i64",
    "uint64": "u64",
}

# The rank of each Rust scalar for unifying two operand types. After integer promotion it orders the integers as the
# usual arithmetic conversions do: the wider type wins, and of two types of one width the unsigned one.
RUST_SCALAR_RANKS = {
    "bool": 0,
    "i8": 1,
    "u8": 2,
    "i16": 3,
    "u16": 4,
    "i32": 5,
    "u32": 6,
    "i64": 7,
    "u64": 8,
    "f16": 9,
    "f32": 10,
    "f64": 11,
}

# The operators whose integer operands are promoted (C11 6.5.5-6.5.12; the shift's right operand only sets the
# count, and promoting it does not change the result).
INTEGER_PROMOTION_OPERATORS = frozenset(
    {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"}
)

_NARROW_INTEGERS = frozenset({"i8", "u8", "i16", "u16"})


def promoted_operand_type(rust_scalar, operator):
    """The Rust type an operand of type ``rust_scalar`` computes in under ``operator``: ``i32`` for a narrow integer."""
    if operator in INTEGER_PROMOTION_OPERATORS and rust_scalar in _NARROW_INTEGERS:
        return "i32"
    return rust_scalar
