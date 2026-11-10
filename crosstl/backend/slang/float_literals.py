"""Range checks for imported Slang floating-point literals.

slangc types an unsuffixed or ``f``-suffixed floating literal as ``float`` and an ``h``-suffixed one as ``half``.
A literal outside the type's finite range is "unrepresentable, converted to 'inf'" (warning E40009), and a nonzero
literal that rounds to zero is "smaller than the smallest representable value ... converted to '0'" (E40010).
CrossGL has no literal spelling for infinity, and an exponent such as ``e4294967296`` cannot be written out in
fixed notation at all, so the importer refuses both cases instead of handing the target compiler a different number.
"""

from __future__ import annotations

from decimal import Decimal

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["SlangFloatLiteralUnrepresentableError", "check_float_literal_range"]


# IEEE 754 round-to-nearest-even: a magnitude at or above max + ulp/2 becomes infinity, and a nonzero magnitude
# at or below half the smallest subnormal becomes zero. The limits are built exactly: Decimal arithmetic would
# round them to the context precision. 2**-n is 5**n * 10**-n.
def _power_of_two_reciprocal(exponent):
    return Decimal((0, tuple(int(digit) for digit in str(5**exponent)), -exponent))


_LIMITS = {
    "float": (Decimal(2**128 - 2**103), _power_of_two_reciprocal(150)),
    "half": (Decimal(65520), _power_of_two_reciprocal(25)),
}


class SlangFloatLiteralUnrepresentableError(TranslationDiagnostic):
    """A Slang floating-point literal whose value its type cannot hold."""

    def __init__(self, literal, type_name, result):
        super().__init__(
            f"Slang floating-point literal {literal!r} is not representable as {type_name}; "
            f"slangc converts it to {result}, which CrossGL cannot spell as a literal",
            target="cgl",
            code=f"{CODE_PREFIX}slang-float-literal-unrepresentable",
        )


def check_float_literal_range(literal, value, suffix):
    """Raise when ``value`` (the literal's digits as a ``Decimal``) overflows or underflows its Slang type."""
    type_name = "half" if any(char in "hH" for char in suffix) else "float"
    overflow, underflow = _LIMITS[type_name]
    magnitude = (
        value.copy_abs()
    )  # abs() rounds to the context and overflows on exponents like e4294967296
    if magnitude >= overflow:
        raise SlangFloatLiteralUnrepresentableError(literal, type_name, "inf")
    if magnitude and magnitude <= underflow:
        raise SlangFloatLiteralUnrepresentableError(literal, type_name, "0")
