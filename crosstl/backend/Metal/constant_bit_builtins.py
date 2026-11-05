"""Fold Clang's bit-counting builtins on concrete integral operands.

Metal is compiled by Clang, so constant expressions such as ``constexpr short logN = __builtin_ctz(N);`` are valid in
Metal sources and commonly size template arguments. After template materialization the operand is an integer literal,
and the call is a core constant expression whose value is fixed by the builtin's definition (Clang "Language
Extensions", "Builtin Functions"; GCC "Other Built-in Functions"):

* ``__builtin_ctz(x)`` and ``__builtin_clz(x)`` count the trailing and leading zero bits of ``x``. The result is
  undefined when ``x`` is zero, and Clang rejects such a call in a constant expression, so a zero operand is never
  folded.
* ``__builtin_popcount(x)`` counts the one bits of ``x``.

The unsuffixed builtins take ``unsigned int`` and the ``l``/``ll`` forms take ``unsigned long`` and
``unsigned long long``. Metal's ``int`` is 32 bits and its ``long`` is 64 bits (MSL 3.2 section 2.1, "Scalar Data
Types"), so the operands are 32 and 64 bits wide. A signed operand converts to the unsigned parameter type modulo
``2**W`` ([conv.integral]).
"""

from typing import Optional

# Each builtin's operand width in bits.
BIT_BUILTIN_WIDTHS = {
    "__builtin_ctz": 32,
    "__builtin_ctzl": 64,
    "__builtin_ctzll": 64,
    "__builtin_clz": 32,
    "__builtin_clzl": 64,
    "__builtin_clzll": 64,
    "__builtin_popcount": 32,
    "__builtin_popcountl": 64,
    "__builtin_popcountll": 64,
}


def fold_bit_builtin(name: str, operand: int) -> Optional[int]:
    """The value of builtin ``name`` applied to ``operand``, or None when it has no defined constant value."""
    width = BIT_BUILTIN_WIDTHS.get(name)
    if width is None or type(operand) is not int:
        return None
    if not -(1 << (width - 1)) <= operand < (1 << width):
        # Outside both the signed and unsigned ranges of the parameter type; the
        # source expression would have been narrowed before the call.
        return None
    value = operand % (1 << width)
    kind = name[len("__builtin_") :].rstrip("l")
    if kind == "popcount":
        return value.bit_count()
    if value == 0:
        return None
    if kind == "ctz":
        return (value & -value).bit_length() - 1
    return width - value.bit_length()
