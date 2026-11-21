"""Exact IEEE 754 binary32 and binary64 operations rounded down, up or toward zero, on bit patterns.

Every result is computed from the operands' exact rational values with :class:`fractions.Fraction` and rounded once,
so it is independent of the host's floating-point unit and of the CrossGL functions under test. NaN results are
CUDA's canonical encodings.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

FORMATS = {"f32": (32, 24, 8), "f64": (64, 53, 11)}
CANONICAL_NAN = {"f32": 0x7FFFFFFF, "f64": 0xFFF8000000000000}


class _Format:
    def __init__(self, name):
        self.name = name
        self.width, self.precision, exponent_bits = FORMATS[name]
        self.bias = (1 << (exponent_bits - 1)) - 1
        self.emin = 1 - self.bias
        self.emax = self.bias
        self.sign = 1 << (self.width - 1)
        self.infinity = ((1 << exponent_bits) - 1) << (self.precision - 1)
        self.nan = CANONICAL_NAN[name]

    def classify(self, bits):
        magnitude = bits & (self.sign - 1)
        negative = bool(bits & self.sign)
        if magnitude > self.infinity:
            return "nan", negative, None
        if magnitude == self.infinity:
            return "inf", negative, None
        if magnitude == 0:
            return "zero", negative, Fraction(0)
        field = magnitude >> (self.precision - 1)
        mantissa = magnitude & ((1 << (self.precision - 1)) - 1)
        if field:
            mantissa |= 1 << (self.precision - 1)
        exponent = max(field, 1) - self.bias - (self.precision - 1)
        value = Fraction(mantissa) * Fraction(2) ** exponent
        return "finite", negative, -value if negative else value

    def zero(self, negative):
        return self.sign if negative else 0

    def infinite(self, negative):
        return self.infinity | (self.sign if negative else 0)

    def round(self, value, mode):
        """Bits of the nonzero rational ``value`` rounded in ``mode`` ('rd', 'ru' or 'rz')."""
        negative = value < 0
        magnitude = -value if negative else value
        away = (mode == "rd" and negative) or (mode == "ru" and not negative)
        exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
        if Fraction(2) ** exponent > magnitude:
            exponent -= 1
        quantum = max(exponent, self.emin) - (self.precision - 1)
        scaled = magnitude / Fraction(2) ** quantum
        integer = scaled.numerator // scaled.denominator
        if away and integer != scaled:
            integer += 1
        if integer == 1 << self.precision:
            integer >>= 1
            quantum += 1
        if quantum + self.precision - 1 > self.emax:
            bits = self.infinity if away else self.infinity - 1
        elif integer < 1 << (self.precision - 1):
            bits = integer
        else:
            field = quantum + self.precision - 1 + self.bias
            bits = (field << (self.precision - 1)) | (
                integer - (1 << (self.precision - 1))
            )
        return bits | (self.sign if negative else 0)

    def exact_zero_sum(self, mode):
        return self.zero(mode == "rd")


def add(fmt, a, b, mode):
    f = _Format(fmt)
    (ka, na, va), (kb, nb, vb) = f.classify(a), f.classify(b)
    if "nan" in (ka, kb) or (ka == kb == "inf" and na != nb):
        return f.nan
    if ka == "inf":
        return f.infinite(na)
    if kb == "inf":
        return f.infinite(nb)
    if ka == kb == "zero":
        return f.zero(na or nb if mode == "rd" else na and nb)
    total = va + vb
    return f.exact_zero_sum(mode) if total == 0 else f.round(total, mode)


def sub(fmt, a, b, mode):
    return add(fmt, a, b ^ _Format(fmt).sign, mode)


def mul(fmt, a, b, mode):
    f = _Format(fmt)
    (ka, na, va), (kb, nb, vb) = f.classify(a), f.classify(b)
    negative = na != nb
    if "nan" in (ka, kb) or {ka, kb} == {"inf", "zero"}:
        return f.nan
    if "inf" in (ka, kb):
        return f.infinite(negative)
    if "zero" in (ka, kb):
        return f.zero(negative)
    return f.round(va * vb, mode)


def div(fmt, a, b, mode):
    f = _Format(fmt)
    (ka, na, va), (kb, nb, vb) = f.classify(a), f.classify(b)
    negative = na != nb
    if "nan" in (ka, kb) or ka == kb == "inf" or ka == kb == "zero":
        return f.nan
    if ka == "inf" or kb == "zero":
        return f.infinite(negative)
    if kb == "inf" or ka == "zero":
        return f.zero(negative)
    return f.round(va / vb, mode)


def rcp(fmt, a, mode):
    f = _Format(fmt)
    return div(fmt, f.bias << (f.precision - 1), a, mode)


def fma(fmt, a, b, c, mode):
    f = _Format(fmt)
    (ka, na, va), (kb, nb, vb), (kc, nc, vc) = (
        f.classify(a),
        f.classify(b),
        f.classify(c),
    )
    product_negative = na != nb
    if "nan" in (ka, kb, kc) or {ka, kb} == {"inf", "zero"}:
        return f.nan
    if "inf" in (ka, kb):
        if kc == "inf" and nc != product_negative:
            return f.nan
        return f.infinite(product_negative)
    if kc == "inf":
        return f.infinite(nc)
    if "zero" in (ka, kb) and kc == "zero":
        return f.zero(
            product_negative or nc if mode == "rd" else product_negative and nc
        )
    total = va * vb + vc
    return f.exact_zero_sum(mode) if total == 0 else f.round(total, mode)


def sqrt(fmt, a, mode):
    f = _Format(fmt)
    kind, negative, value = f.classify(a)
    if kind == "nan" or (negative and kind != "zero"):
        return f.nan
    if kind in ("zero", "inf"):
        return a
    # floor(sqrt(v) / 2^q) = isqrt(floor(v / 4^q)) with q low enough to keep precision + 2 bits.
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    q = exponent // 2 - f.precision - 4
    scaled = value / Fraction(4) ** q
    whole = scaled.numerator // scaled.denominator
    root = isqrt(whole)
    exact = root * root == whole and whole == scaled
    # Fold an inexact remainder into one more bit that rounding treats as sticky.
    approximation = Fraction(2 * root + (0 if exact else 1), 2) * Fraction(2) ** q
    return f.round(approximation, mode)


def from_integer(fmt, value, mode):
    if value == 0:
        return 0
    return _Format(fmt).round(Fraction(value), mode)


def narrow(a, mode):
    kind, negative, value = _Format("f64").classify(a)
    f = _Format("f32")
    if kind == "nan":
        return f.nan
    if kind == "inf":
        return f.infinite(negative)
    if kind == "zero":
        return f.zero(negative)
    return f.round(value, mode)
