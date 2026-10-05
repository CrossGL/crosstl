"""Portable binary32 exponential with explicit subnormal result construction."""

from .source_licenses import SOURCE_LICENSES

# Adapted from OpenLibm src/e_expf.c, revision
# 5fe399749f9276eaa0b8403e507470da05cbbb3f (fdlibm).
# Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
# Developed at SunPro, a Sun Microsystems, Inc. business.
# Permission to use, copy, modify, and distribute this software is freely
# granted, provided that this notice is preserved.
# Integer exponent scaling replaces floating-point underflow multiplication.

_EXP_SUPPORT = """@source_license(fdlibm)
@precise
@metal_static
float $evaluate(float value) {
    uint word = asuint(value);
    uint magnitude = word & 0x7fffffffu;
    bool negative = (word >> 31u) != 0u;
    if (magnitude > 0x7f800000u) { return asfloat(0x7fc00000u); }
    if (magnitude == 0x7f800000u) { return negative ? 0.0 : value; }
    if (!negative && magnitude >= 0x42b17218u) { return asfloat(0x7f800000u); }
    if (negative && magnitude >= 0x42cff1b5u) { return 0.0; }
    if (magnitude < 0x39000000u) { return 1.0 + value; }
    float high @precise = 0.0;
    float low @precise = 0.0;
    int exponent = 0;
    float reduced @precise = value;
    if (magnitude > 0x3eb17218u) {
        if (magnitude < 0x3f851592u) {
            high = value - (negative ? -6.9314575195e-1 : 6.9314575195e-1);
            low = negative ? -1.4286067653e-6 : 1.4286067653e-6;
            exponent = negative ? -1 : 1;
        } else {
            float quotient @precise = value * 1.4426950216;
            exponent = int(quotient + (negative ? -0.5 : 0.5));
            high = value - float(exponent) * 6.9314575195e-1;
            low = float(exponent) * 1.4286067653e-6;
        }
        reduced = high - low;
    }
    float squared @precise = reduced * reduced;
    float correction @precise = reduced
        - squared * (1.6666625440e-1 + squared * -2.7667332906e-3);
    float ratio @precise = (reduced * correction) / (2.0 - correction);
    float result @precise = exponent == 0 ? 1.0 + (ratio + reduced)
        : 1.0 - ((low - ratio) - high);
    uint result_word = asuint(result);
    int scale = int(result_word >> 23u) - 127 + exponent;
    if (scale > 127) { return asfloat(0x7f800000u); }
    uint mantissa = (result_word & 0x7fffffu) | 0x800000u;
    if (scale >= -126) {
        return asfloat((uint(scale + 127) << 23u) | (mantissa & 0x7fffffu));
    }
    // Round the scaled significand without executing subnormal arithmetic.
    int shift = -126 - scale;
    if (shift > 24) { return 0.0; }
    uint retained = mantissa >> uint(shift);
    uint remainder = mantissa & ((1u << uint(shift)) - 1u);
    uint midpoint = 1u << uint(shift - 1);
    retained += uint(remainder > midpoint
                    || (remainder == midpoint && (retained & 1u) != 0u));
    return asfloat(retained);
}
"""


def binary32_exp_support(name):
    """Render a scalar approximation, not a correctly rounded exp contract.

    Range reduction and rational evaluation use binary32 arithmetic. Result
    scaling preserves gradual underflow; NaN payloads and exception flags are
    not represented. The caller supplies a collision-safe function name.
    """
    notice = "".join(f"// {line}\n" for line in SOURCE_LICENSES["fdlibm"].splitlines())
    return notice + _EXP_SUPPORT.replace("$evaluate", name)
