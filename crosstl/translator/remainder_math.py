"""Portable binary32 remainder with explicit subnormal operation policies."""

_REMAINDER_SUPPORT = """
@metal_static
uint $bits(uint a, uint b, bool flush_arithmetic) {
    uint sign = a & 0x80000000u;
    uint x = a & 0x7fffffffu;
    uint y = b & 0x7fffffffu;
    if (x >= 0x7f800000u || y > 0x7f800000u || y == 0u
        || (flush_arithmetic && y < 0x00800000u)) {
        return 0x7fc00000u;
    }
    // The no-division path preserves the numerator, including subnormal bits.
    if (x < y) { return a; }
    if (x == y) { return sign; }
    int x_exponent = int(x >> 23u);
    int y_exponent = int(y >> 23u);
    uint remainder = x & 0x007fffffu;
    uint divisor = y & 0x007fffffu;
    if (x_exponent == 0) {
        x_exponent = 1;
        while (remainder < 0x00800000u) { remainder <<= 1u; x_exponent -= 1; }
    } else { remainder |= 0x00800000u; }
    if (y_exponent == 0) {
        y_exponent = 1;
        while (divisor < 0x00800000u) { divisor <<= 1u; y_exponent -= 1; }
    } else { divisor |= 0x00800000u; }
    // Significands remain below 25 bits, even for the largest exponent gap.
    for (int shift = x_exponent - y_exponent; shift >= 0; shift -= 1) {
        if (remainder >= divisor) { remainder -= divisor; }
        if (remainder == 0u) { return sign; }
        if (shift > 0) { remainder <<= 1u; }
    }
    while (remainder < 0x00800000u) { remainder <<= 1u; y_exponent -= 1; }
    if (y_exponent > 0) {
        remainder = (uint(y_exponent) << 23u) | (remainder & 0x007fffffu);
    } else {
        if (flush_arithmetic) { return sign; }
        remainder >>= uint(1 - y_exponent);
    }
    return sign | remainder;
}
"""


def binary32_remainder_support(name):
    """Render an exact integer-significand remainder helper.

    Preserving mode computes truncating remainder, including subnormal results.
    Arithmetic-flush mode treats subnormal divisors as zero and flushes computed
    subnormal remainders, but retains the numerator when its magnitude is less
    than a valid divisor. Both modes preserve zero signs and canonicalize NaNs.
    This helper does not select a source compiler or device contract.
    """
    return _REMAINDER_SUPPORT.replace("$bits", name)
