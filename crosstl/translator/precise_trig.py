"""Binary32 trigonometry with integer range reduction across shader targets."""

from .source_licenses import SOURCE_LICENSES

# Adapted from Arm optimized-routines math/sincosf.h and math/sincosf_data.c,
# revision 375f32ed2f7098090f41795ad363822752a25a65.
# Copyright (c) 2018-2024, Arm Limited. Used under the MIT license; its full
# notice is registered in source_licenses.py and retained in generated sources.
# The reduction uses uint32 pairs instead of uint64. Conversion and polynomial
# evaluation use binary32 instead of upstream double arithmetic.

TRIG_HELPER_KEYS = ("word", "multiply_high", "evaluate")

_INV_PIO4 = (
    0xA2,
    0xA2F9,
    0xA2F983,
    0xA2F9836E,
    0xF9836E4E,
    0x836E4E44,
    0x6E4E4415,
    0x4E441529,
    0x441529FC,
    0x1529FC27,
    0x29FC2757,
    0xFC2757D1,
    0x2757D1F5,
    0x57D1F534,
    0xD1F534DD,
    0xF534DDC0,
    0x34DDC0DB,
    0xDDC0DB62,
    0xC0DB6295,
    0xDB629599,
    0x6295993C,
    0x95993C43,
    0x993C4390,
    0x3C439041,
)

_TRIG_SUPPORT = """
@metal_static
uint $multiply_high(uint a, uint b) {
    uint low = (a & 65535u) * (b & 65535u);
    uint middle = (a >> 16u) * (b & 65535u) + (low >> 16u);
    uint upper = middle >> 16u;
    middle = (a & 65535u) * (b >> 16u) + (middle & 65535u);
    return (a >> 16u) * (b >> 16u) + upper + (middle >> 16u);
}
@precise
@metal_static
float $evaluate(float value, bool cosine) {
    uint bits = asuint(value);
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) { return asfloat(0x7fc00000u); }
    if (magnitude < 0x39800000u) { return cosine ? 1.0 : value; }
    float reduced @precise = asfloat(magnitude);
    uint quadrant = 0u;
    if (magnitude >= 0x40000000u) {
        // Keep the fractional remainder in integer form until after rounding
        // the quadrant. Subtracting rounded floats loses near-axis results.
        uint index = (magnitude >> 26u) & 15u;
        uint significand = ((magnitude & 0xffffffu) | 0x800000u)
                           << ((magnitude >> 23u) & 7u);
        uint first = significand * $word(index);
        uint middle_word = $word(index + 4u);
        uint last_high = $multiply_high(significand, $word(index + 8u));
        uint middle_low = significand * middle_word;
        uint low = last_high + middle_low;
        uint high = first + $multiply_high(significand, middle_word)
                    + uint(low < last_high);
        quadrant = (high + 0x20000000u) >> 30u;
        bool negative = (high & 0x20000000u) != 0u;
        high &= 0x3fffffffu;
        if (negative) {
            high = ((~high) + uint(low == 0u)) & 0x3fffffffu;
            low = 0u - low;
        }
        float fraction @precise = float(high) * 9.31322574615478515625e-10
                                  + float(low) * 2.16840434497100886801e-19;
        reduced = fraction * 1.57079632679489661923;
        if (negative) { reduced = -reduced; }
    } else if (magnitude > 0x3f490fdbu) {
        reduced = (reduced - 1.57079625129699707031) - 7.54978941586159635335e-8;
        quadrant = 1u;
    }
    float squared @precise = reduced * reduced;
    float cubed @precise = reduced * squared;
    float fourth @precise = squared * squared;
    float sine @precise = (reduced + cubed * -0.16666655242443084717)
        + (cubed * squared) * (0.00833217799663543701
                              + squared * -0.00019517299369908869);
    float cos_value @precise = ((1.0 + squared * -0.5)
                               + fourth * 0.04166662320494651794)
        + (fourth * squared) * (-0.00138867634814232588
                               + squared * 0.00002439045056235045);
    float result @precise = ((quadrant & 1u) != 0u) != cosine ? cos_value : sine;
    uint sign = cosine ? ((quadrant + 1u) & 2u) : (quadrant & 2u);
    if (!cosine) { sign ^= (bits >> 30u) & 2u; }
    return sign != 0u ? -result : result;
}
"""


def binary32_trig_support(names):
    """Render finite-domain reduction and scalar sine/cosine evaluation.

    Signed zero is preserved for sine. Nonfinite operands return a quiet NaN;
    NaN payloads and floating-point exception flags are not represented.
    Every generated name belongs to the caller's collision-safe namespace.
    """
    code = "".join(
        f"// {line}\n" for line in SOURCE_LICENSES["arm_optimized"].splitlines()
    )
    code += "@source_license(arm_optimized)\n@metal_static\n"
    code += "uint $word(uint index) {\n"
    for index, word in enumerate(_INV_PIO4[:-1]):
        code += f"    if (index == {index}u) {{ return 0x{word:08x}u; }}\n"
    code += f"    return 0x{_INV_PIO4[-1]:08x}u;\n}}\n"
    code += _TRIG_SUPPORT
    for key in TRIG_HELPER_KEYS:
        code = code.replace("$" + key, names[key])
    return code
