"""Portable binary32 division with explicit rounding and underflow policies."""

_DIVISION_SUPPORT = """
@metal_static
uint $bits(uint a, uint b, bool flush_denormals) {
    if (flush_denormals) {
        if ((a & 2139095040u) == 0u) { a &= 2147483648u; }
        if ((b & 2139095040u) == 0u) { b &= 2147483648u; }
    }
    uint sign = (a ^ b) & 2147483648u;
    uint magnitude_a = a & 2147483647u;
    uint magnitude_b = b & 2147483647u;
    if (magnitude_a > 2139095040u || magnitude_b > 2139095040u
        || (magnitude_a == 0u && magnitude_b == 0u)
        || (magnitude_a == 2139095040u && magnitude_b == 2139095040u)) {
        return 2143289344u;
    }
    if (magnitude_a == 2139095040u || magnitude_b == 0u) {
        return sign | 2139095040u;
    }
    if (magnitude_a == 0u || magnitude_b == 2139095040u) { return sign; }
    int exponent_a = int(a >> 23u & 255u);
    int exponent_b = int(b >> 23u & 255u);
    uint significand_a = a & 8388607u;
    uint significand_b = b & 8388607u;
    if (exponent_a == 0) {
        exponent_a = 1;
        while (significand_a < 8388608u) { significand_a <<= 1u; exponent_a -= 1; }
    } else { significand_a |= 8388608u; }
    if (exponent_b == 0) {
        exponent_b = 1;
        while (significand_b < 8388608u) { significand_b <<= 1u; exponent_b -= 1; }
    } else { significand_b |= 8388608u; }
    int exponent = exponent_a - exponent_b + 127;
    if (significand_a < significand_b) { significand_a <<= 1u; exponent -= 1; }
    if (exponent > 254) { return sign | 2139095040u; }
    if (flush_denormals && exponent < 1) { return sign; }
    // The quotient is in [1, 2). Below half the least subnormal it rounds to zero.
    if (exponent < -23) { return sign; }
    if (exponent == -23) { return sign | uint(significand_a > significand_b); }
    int fractional_bits = exponent > 0 ? 23 : exponent + 22;
    uint quotient = 1u;
    uint remainder = significand_a - significand_b;
    // The remainder stays below the 24-bit divisor; no wide integer is needed.
    for (int bit = 0; bit < fractional_bits; bit += 1) {
        remainder <<= 1u;
        quotient <<= 1u;
        if (remainder >= significand_b) { remainder -= significand_b; quotient |= 1u; }
    }
    uint twice_remainder = remainder << 1u;
    if (twice_remainder > significand_b
        || (twice_remainder == significand_b && (quotient & 1u) != 0u)) {
        quotient += 1u;
    }
    // Addition carries into the exponent at normal and overflow boundaries.
    uint result = exponent > 0 ? (uint(exponent - 1) << 23u) + quotient : quotient;
    return sign | result;
}
"""


def binary32_division_support(name):
    """Render an integer-word division helper owned by the caller.

    Results round to nearest with ties to even. ``flush_denormals`` selects
    signed-zero flushing of subnormal operands and results before rounding;
    otherwise underflow is gradual. NaNs are canonicalized and exception flags
    are not represented. This helper does not select a source execution profile.
    """
    return _DIVISION_SUPPORT.replace("$bits", name)
