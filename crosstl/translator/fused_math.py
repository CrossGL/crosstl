"""Portable binary32 fused rounding using pairs of 32-bit integer words."""

# Adapted from Berkeley SoftFloat Release 3e, s_mulAddF32.c and s_roundPackToF32.c.
# Upstream revision: a0c6494cdc11865811dec815d5c0049fba9d82a8.
# Generated definitions carry the registered softfloat license notice.
FMA_HELPER_KEYS = (
    "add",
    "subtract",
    "shift_left",
    "shift_jam",
    "leading_zeros",
    "round_pack",
    "bits",
)

_FMA_SUPPORT = """
@source_license(softfloat)
uvec2 $add(uvec2 a, uvec2 b) {
    uint low = a.x + b.x;
    return uvec2(low, a.y + b.y + uint(low < a.x));
}
uvec2 $subtract(uvec2 a, uvec2 b) {
    return uvec2(a.x - b.x, a.y - b.y - uint(a.x < b.x));
}
uvec2 $shift_left(uvec2 value, int distance) {
    if (distance == 0) { return value; }
    if (distance < 32) {
        return uvec2(value.x << uint(distance),
                     (value.y << uint(distance)) | (value.x >> uint(32 - distance)));
    }
    if (distance < 64) { return uvec2(0u, value.x << uint(distance - 32)); }
    return uvec2(0u, 0u);
}
uvec2 $shift_jam(uvec2 value, int distance) {
    if (distance == 0) { return value; }
    if (distance < 32) {
        uint low = (value.x >> uint(distance)) | (value.y << uint(32 - distance));
        uint sticky = uint((value.x << uint(32 - distance)) != 0u);
        return uvec2(low | sticky, value.y >> uint(distance));
    }
    if (distance == 32) { return uvec2(value.y | uint(value.x != 0u), 0u); }
    if (distance < 64) {
        uint sticky = uint(value.x != 0u || (value.y << uint(64 - distance)) != 0u);
        return uvec2((value.y >> uint(distance - 32)) | sticky, 0u);
    }
    return uvec2(uint(value.x != 0u || value.y != 0u), 0u);
}
int $leading_zeros(uvec2 value) {
    uint word = value.y;
    int count = 0;
    if (word == 0u) { word = value.x; count = 32; }
    if (word == 0u) { return 64; }
    while ((word & 2147483648u) == 0u) { word <<= 1u; count += 1; }
    return count;
}
uint $round_pack(uint sign, int exponent, uint significand, bool flush_denormals) {
    if (exponent < 0) {
        if (flush_denormals) { return sign; }
        significand = $shift_jam(uvec2(significand, 0u), -exponent).x;
        exponent = 0;
    }
    if (exponent > 253 || (exponent == 253 && significand >= 2147483584u)) {
        return sign | 2139095040u;
    }
    uint rounding = significand & 127u;
    significand = (significand + 64u) >> 7u;
    if (rounding == 64u) { significand &= 4294967294u; }
    if (significand == 0u) { return sign; }
    uint result = sign + (uint(exponent) << 23u) + significand;
    if (flush_denormals && (result & 2139095040u) == 0u) { return sign; }
    return result;
}
uint $bits(uint a, uint b, uint c, bool flush_denormals) {
    if (flush_denormals) {
        if ((a & 2139095040u) == 0u) { a &= 2147483648u; }
        if ((b & 2139095040u) == 0u) { b &= 2147483648u; }
        if ((c & 2139095040u) == 0u) { c &= 2147483648u; }
    }
    uint sign = (a ^ b) & 2147483648u;
    uint sign_c = c & 2147483648u;
    int exponent_a = int((a >> 23u) & 255u);
    int exponent_b = int((b >> 23u) & 255u);
    int exponent_c = int((c >> 23u) & 255u);
    uint fraction_a = a & 8388607u;
    uint fraction_b = b & 8388607u;
    uint fraction_c = c & 8388607u;
    if ((exponent_a == 255 && fraction_a != 0u)
        || (exponent_b == 255 && fraction_b != 0u)
        || (exponent_c == 255 && fraction_c != 0u)) { return 2143289344u; }
    if (exponent_a == 255 || exponent_b == 255) {
        if ((a & 2147483647u) == 0u || (b & 2147483647u) == 0u
            || (exponent_c == 255 && sign != sign_c)) { return 2143289344u; }
        return sign | 2139095040u;
    }
    if (exponent_c == 255) { return c; }
    if ((a & 2147483647u) == 0u || (b & 2147483647u) == 0u) {
        if ((c & 2147483647u) != 0u) { return c; }
        return sign & sign_c;
    }
    if (exponent_a == 0) {
        int distance = $leading_zeros(uvec2(fraction_a, 0u)) - 40;
        fraction_a <<= uint(distance);
        exponent_a = 1 - distance;
    }
    if (exponent_b == 0) {
        int distance = $leading_zeros(uvec2(fraction_b, 0u)) - 40;
        fraction_b <<= uint(distance);
        exponent_b = 1 - distance;
    }
    uint significand_a = fraction_a | 8388608u;
    uint significand_b = fraction_b | 8388608u;
    // Form the exact 48-bit product using bounded 16-bit partial products.
    uint low_product = (significand_a & 65535u) * (significand_b & 65535u);
    uint middle = (significand_a >> 16u) * (significand_b & 65535u)
                + (significand_b >> 16u) * (significand_a & 65535u);
    uint low = low_product + (middle << 16u);
    uint high = (significand_a >> 16u) * (significand_b >> 16u)
                + (middle >> 16u) + uint(low < low_product);
    uvec2 product = $shift_left(uvec2(low, high), 14);
    // Align product and addend with sticky bits before the single final rounding.
    int exponent_product = exponent_a + exponent_b - 126;
    if (product.y < 536870912u) { product = $shift_left(product, 1); exponent_product -= 1; }
    if ((c & 2147483647u) == 0u) {
        return $round_pack(sign, exponent_product - 1, $shift_jam(product, 31).x, flush_denormals);
    }
    if (exponent_c == 0) {
        int distance = $leading_zeros(uvec2(fraction_c, 0u)) - 40;
        fraction_c <<= uint(distance);
        exponent_c = 1 - distance;
    }
    uint significand_c = (fraction_c | 8388608u) << 6u;
    int difference = exponent_product - exponent_c;
    int exponent;
    uint significand;
    if (sign == sign_c) {
        if (difference <= 0) {
            exponent = exponent_c;
            significand = significand_c + $shift_jam(product, 32 - difference).x;
        } else {
            exponent = exponent_product;
            uvec2 sum = $add(product, $shift_jam(uvec2(0u, significand_c), difference));
            significand = $shift_jam(sum, 32).x;
        }
        if (significand < 1073741824u) { exponent -= 1; significand <<= 1u; }
    } else {
        uvec2 addend = uvec2(0u, significand_c);
        uvec2 difference_bits;
        if (difference < 0) {
            sign = sign_c;
            exponent = exponent_c;
            difference_bits = $subtract(addend, $shift_jam(product, -difference));
        } else if (difference == 0) {
            exponent = exponent_product;
            difference_bits = $subtract(product, addend);
            if (difference_bits.x == 0u && difference_bits.y == 0u) { return 0u; }
            if ((difference_bits.y & 2147483648u) != 0u) {
                sign ^= 2147483648u;
                difference_bits = $subtract(uvec2(0u, 0u), difference_bits);
            }
        } else {
            exponent = exponent_product;
            difference_bits = $subtract(product, $shift_jam(addend, difference));
        }
        int distance = $leading_zeros(difference_bits) - 1;
        exponent -= distance;
        distance -= 32;
        if (distance < 0) { significand = $shift_jam(difference_bits, -distance).x; }
        else { significand = difference_bits.x << uint(distance); }
    }
    return $round_pack(sign, exponent, significand, flush_denormals);
}
"""


def binary32_fma_support(names):
    """Render binary32 RNE support, optionally flushing subnormals before rounding.

    ``flush_denormals`` also flushes subnormal inputs, retaining signed zero.
    NaN results use the canonical quiet NaN; payloads and exception flags are
    not represented. All generated names belong to the caller.
    """
    code = _FMA_SUPPORT
    for key in FMA_HELPER_KEYS:
        code = code.replace("$" + key, names[key])
    return code
