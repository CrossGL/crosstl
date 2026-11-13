"""Metal's predefined ``half`` and ``bfloat`` constants.

``<metal_math>`` and ``<metal_limits>`` define ``M_PI_H``, ``HALF_MAX``, ``M_PI_BF``, ``BFLT_MAX`` and their
relatives as macros over the compiler's ``__HALF_*__`` and ``__BFLT16_*__`` constants (MSL specification,
section 6.6.1). The importer used to keep these names as identifiers, and no target defines them, so they reached
HLSL and GLSL verbatim. The importer now writes each one as a conversion of its exact value: the source decimal
(the ``H`` or ``BF16`` literal recorded in ``crosstl/translator/preprocessor/data/msl.json``) rounded to the
nearest ``half`` or ``bfloat`` value, ties to even. Every such value is exactly representable as a ``float``, so
the conversion is exact in every target. The integer-valued limits (``HALF_DIG``, ``BFLT_MAX_EXP`` and so on)
are ``int`` constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from crosstl.translator.refusals import MetalSourceRefusal

from ...translator.float_limit_constants import float_limit_constant
from ...translator.standard_constants import standard_math_constant

__all__ = [
    "NarrowFloatConstant",
    "metal_predefined_constant_type",
    "narrow_float_constant",
    "render_narrow_float_constant",
    "round_to_format",
]

# (name, decimal) pairs of the compiler's literals, without their H / BF16 suffix.
_HALF_DECIMALS = {
    "HALF_MAX": "6.5504e+4",
    "MAXHALF": "6.5504e+4",
    "HALF_MIN": "6.103515625e-5",
    "HALF_EPSILON": "9.765625e-4",
    "M_E_H": "2.71828182845904523536028747135266250",
    "M_LOG2E_H": "1.44269504088896340735992468100189214",
    "M_LOG10E_H": "0.434294481903251827651128918916605082",
    "M_LN2_H": "0.693147180559945309417232121458176568",
    "M_LN10_H": "2.3025850929940456840179914546843642",
    "M_PI_H": "3.14159265358979323846264338327950288",
    "M_PI_2_H": "1.57079632679489661923132169163975144",
    "M_PI_4_H": "0.785398163397448309615660845819875721",
    "M_1_PI_H": "0.318309886183790671537767526745028724",
    "M_2_PI_H": "0.636619772367581343075535053490057448",
    "M_2_SQRTPI_H": "1.12837916709551257389615890312154517",
    "M_SQRT2_H": "1.41421356237309504880168872420969808",
    "M_SQRT1_2_H": "0.707106781186547524400844362104849039",
}
_BFLOAT_DECIMALS = {
    "BFLT_MAX": "3.389531e+38",
    "BFLT_MIN": "1.175494e-38",
    "BFLT_EPSILON": "7.812500e-03",
    "M_E_BF": "2.71828182845904523536028747135266250",
    "M_LOG2E_BF": "1.44269504088896340735992468100189214",
    "M_LOG10E_BF": "0.434294481903251827651128918916605082",
    "M_LN2_BF": "0.693147180559945309417232121458176568",
    "M_LN10_BF": "2.3025850929940456840179914546843642",
    "M_PI_BF": "3.14159265358979323846264338327950288",
    "M_PI_2_BF": "1.57079632679489661923132169163975144",
    "M_PI_4_BF": "0.785398163397448309615660845819875721",
    "M_1_PI_BF": "0.318309886183790671537767526745028724",
    "M_2_PI_BF": "0.636619772367581343075535053490057448",
    "M_2_SQRTPI_BF": "1.12837916709551257389615890312154517",
    "M_SQRT2_BF": "1.41421356237309504880168872420969808",
    "M_SQRT1_2_BF": "0.707106781186547524400844362104849039",
}
_INTEGERS = {
    "HALF_DIG": 3,
    "HALF_DECIMAL_DIG": 5,
    "HALF_MANT_DIG": 11,
    "HALF_MAX_EXP": 16,
    "HALF_MAX_10_EXP": 4,
    "HALF_MIN_EXP": -13,
    "HALF_MIN_10_EXP": -4,
    "HALF_RADIX": 2,
    "BFLT_DIG": 2,
    "BFLT_DECIMAL_DIG": 4,
    "BFLT_MANT_DIG": 8,
    "BFLT_MAX_EXP": 128,
    "BFLT_MAX_10_EXP": 38,
    "BFLT_MIN_EXP": -125,
    "BFLT_MIN_10_EXP": -37,
    "BFLT_RADIX": 2,
}

# Significand bits (with the implicit bit), smallest normal exponent, largest exponent.
_FORMATS = {"half": (11, -14, 15), "bfloat": (8, -126, 127)}


@dataclass(frozen=True)
class NarrowFloatConstant:
    """A predefined constant: its Metal type and an exact literal of its value."""

    name: str
    metal_type: str
    literal: str


def _exponent(value: Fraction) -> int:
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    while Fraction(2) ** exponent > value:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= value:
        exponent += 1
    return exponent


def round_to_format(decimal: str, metal_type: str) -> Fraction:
    """Round ``decimal`` to the nearest value of ``metal_type``, ties to even."""
    significand_bits, min_exponent, max_exponent = _FORMATS[metal_type]
    value = Fraction(decimal)
    if value <= 0:
        raise MetalSourceRefusal(f"{decimal} is not a positive constant")
    exponent = max(_exponent(value), min_exponent)
    quantum = Fraction(2) ** (exponent - (significand_bits - 1))
    quotient = value / quantum
    whole = quotient.numerator // quotient.denominator
    remainder = quotient - whole
    if remainder > Fraction(1, 2) or (remainder == Fraction(1, 2) and whole % 2):
        whole += 1
    rounded = whole * quantum
    largest = (2 - Fraction(2) ** (1 - significand_bits)) * Fraction(2) ** max_exponent
    if rounded > largest:
        raise MetalSourceRefusal(f"{decimal} overflows {metal_type}")
    return rounded


def _literal(value: Fraction) -> str:
    # The value is a binary fraction that a double holds exactly, and repr gives
    # the shortest decimal that reads back as that double, hence as that float.
    as_float = float(value)
    if Fraction(as_float) != value:
        raise MetalSourceRefusal(f"{value} is not exact as a double")
    text = repr(as_float)
    return text if any(mark in text for mark in ".e") else f"{text}.0"


def _build() -> dict:
    constants = {}
    for metal_type, table in (("half", _HALF_DECIMALS), ("bfloat", _BFLOAT_DECIMALS)):
        for name, decimal in table.items():
            constants[name] = NarrowFloatConstant(
                name, metal_type, _literal(round_to_format(decimal, metal_type))
            )
    for name, value in _INTEGERS.items():
        constants[name] = NarrowFloatConstant(
            name, "int", str(value) if value >= 0 else f"({value})"
        )
    return constants


_CONSTANTS = _build()


def narrow_float_constant(name: object) -> NarrowFloatConstant | None:
    """The predefined ``half``, ``bfloat`` or limit constant ``name``, if it is one."""
    if not isinstance(name, str):
        return None
    return _CONSTANTS.get(name)


def render_narrow_float_constant(converter, constant: NarrowFloatConstant) -> str:
    """CrossGL for ``constant``: an ``int`` literal, or a conversion of a float literal."""
    if constant.metal_type == "int":
        return constant.literal
    return f"{converter.map_type(constant.metal_type)}({constant.literal}f)"


def metal_predefined_constant_type(name: object) -> str | None:
    """The Metal type of a predefined constant: ``FLT_MIN``, ``M_PI_F``, ``HALF_MAX``, ``HALF_DIG``..."""
    narrow = narrow_float_constant(name)
    if narrow is not None:
        return narrow.metal_type
    # INFINITY, NAN, FLT_MAX and the other binary32 limits (MSL section 6.6.1).
    if float_limit_constant(name) is not None:
        return "float"
    constant = standard_math_constant(name) if isinstance(name, str) else None
    if constant is not None and "metal" in constant.source_environments:
        return constant.scalar_type
    return None
