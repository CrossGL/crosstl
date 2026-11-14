"""Binary32 sine and cosine in 32-bit integer arithmetic: the polyfill of the libm rule family.

C-family sources (CUDA, HIP, OpenCL C) bound the error of ``sinf`` and ``cosf`` over the whole binary32 domain. Graphics
targets bound it on ``[-pi, pi]`` at best (Vulkan: an absolute error of 2^-11 there, nothing outside; GLSL 4.60 section
4.7.1 leaves the trigonometric functions without any precision). The helpers below evaluate the functions without a
single floating-point operation, so neither fast-math compilation, contraction, reassociation nor flushing of
subnormals can change their result on any target. Every operation is unsigned 32-bit arithmetic, a comparison, a
select or a bit reinterpretation, which every target defines exactly, and every shift amount is below 32. The top
bit is tested with a shift rather than a comparison with 0x80000000u, which KosmicKrisp (Vulkan SDK 1.4.363)
miscompiles to a signed test, and the significand is normalised by constant shifts: KosmicKrisp's range analysis
drops the top bit of a value shifted left by a variable amount.

1. ``|x| < 2^-12``: ``sin(x) = x`` and ``cos(x) = 1`` are within half an ulp. Infinities and NaN give a quiet NaN.
2. Payne-Hanek reduction: the significand times a 96-bit window of ``2/pi`` gives ``x * 2/pi mod 4`` as a 2.62 fixed
   point value, whose top bits are the quadrant. The window table starts with two zero words, so exponents down to
   2^-15 use the same reduction (the words of ``2/pi`` above its binary point are zero).
3. The remainder ``f`` (``|f| <= 1/2``) is normalised to a 32-bit significand, by shifts of constant amounts only,
   and multiplied by ``pi/2``, giving
   ``r = f * pi/2`` with ``|r| <= pi/4``, and ``z = r^2`` as a 0.32 fixed point value.
4. Taylor polynomials in ``z`` with 0.32 fixed point coefficients: the series alternate with decreasing terms, so
   the first omitted term bounds the truncation (below 2^-36 relative for both).
5. The 64-bit product ``r * (1 - z * p(z))`` (or ``1 - z * q(z)``) is rounded to nearest into the binary32
   significand, and the exponent is added as an integer.

``tests/conformance/libm_exhaustive.py`` evaluates exactly this CrossGL text for all 2^32 operands and records the
largest error in ``tests/conformance/libm_proofs.json``.

The window table is the one of Arm optimized-routines math/sincosf_data.c (MIT license, registered in
source_licenses.py), revision 375f32ed2f7098090f41795ad363822752a25a65; the rest is original.
"""

from __future__ import annotations

import re

from ..precise_trig import _INV_PIO4

# Reserved names (the ``cgl_rule_`` prefix): the program may not define them (helpers.py refuses a collision).
WORD = "cgl_rule_libm_word"
MULTIPLY_HIGH = "cgl_rule_libm_mulhi"
TRIG = "cgl_rule_libm_trig"
SUPPORT_FUNCTIONS = (WORD, MULTIPLY_HIGH, TRIG)

# floor(pi/2 * 2^31): the product with a normalised 32-bit significand keeps 32 significant bits.
_PI_OVER_2 = 0xC90FDAA2
# The smallest significand whose product with _PI_OVER_2 has its top bit set: the normalisation of the product
# tests the significand, because KosmicKrisp (Vulkan SDK 1.4.363) drops the top bit of a sum of 16-bit products
# before a shift.
_REDUCED_TOP_BIT = -(-(1 << 63) // _PI_OVER_2)
# round(2^32 / (2k+1)!) for k = 1..5 and round(2^32 / (2k)!) for k = 1..6.
_SINE = (0x2AAAAAAB, 35791394, 852176, 11836, 108)
_COSINE = (0x80000000, 178956971, 5965232, 106522, 1184, 9)


def _window_function():
    words = (0, 0) + tuple(_INV_PIO4)
    lines = [f"uint {WORD}(uint index) {{"]
    for index, word in enumerate(words[:-1]):
        lines.append(f"    if (index == {index}u) {{ return 0x{word:08x}u; }}")
    lines.append(f"    return 0x{words[-1]:08x}u;")
    lines.append("}")
    return "\n".join(lines)


_MULTIPLY_HIGH = f"""
uint {MULTIPLY_HIGH}(uint a, uint b) {{
    uint low = (a & 65535u) * (b & 65535u);
    uint middle = (a >> 16u) * (b & 65535u) + (low >> 16u);
    uint upper = middle >> 16u;
    uint cross = (a & 65535u) * (b >> 16u) + (middle & 65535u);
    return (a >> 16u) * (b >> 16u) + upper + (cross >> 16u);
}}
"""


def _horner(variable, coefficients):
    """Lines evaluating the polynomial with ``coefficients`` (highest last) in ``z`` into ``variable``."""
    lines = [f"        uint {variable} = {coefficients[-1]}u;"]
    for coefficient in reversed(coefficients[:-1]):
        lines.append(
            f"        {variable} = {coefficient}u - {MULTIPLY_HIGH}(z, {variable});"
        )
    return "\n".join(lines)


_TRIG = f"""
float {TRIG}(float value, bool cosine) {{
    uint bits = asuint(value);
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {{ return asfloat(0x7fc00000u); }}
    if (magnitude < 0x39800000u) {{ return cosine ? 1.0 : value; }}
    uint exponent = magnitude >> 23u;
    uint index = (exponent >> 3u) - 14u;
    uint significand = ((magnitude & 0x007fffffu) | 0x00800000u) << (exponent & 7u);
    uint first = significand * {WORD}(index);
    uint middle_word = {WORD}(index + 4u);
    uint last_high = {MULTIPLY_HIGH}(significand, {WORD}(index + 8u));
    uint middle_low = significand * middle_word;
    uint low = last_high + middle_low;
    uint high = first + {MULTIPLY_HIGH}(significand, middle_word) + (low < last_high ? 1u : 0u);
    uint quadrant = (high + 0x20000000u) >> 30u;
    bool negative = (high & 0x20000000u) != 0u;
    high = high & 0x3fffffffu;
    if (negative) {{
        high = ((~high) + (low == 0u ? 1u : 0u)) & 0x3fffffffu;
        low = 0u - low;
    }}
    bool use_cosine = ((quadrant & 1u) != 0u) != cosine;
    uint sign = cosine ? ((quadrant + 1u) & 2u) : ((quadrant & 2u) ^ ((bits >> 30u) & 2u));
    if (negative && !use_cosine) {{ sign = sign ^ 2u; }}
    uint result = 0x3f800000u;
    if ((high | low) == 0u) {{
        result = use_cosine ? 0x3f800000u : 0u;
    }} else {{
        uint mantissa = high;
        uint rest = low;
        uint scale = 30u;
        if (mantissa == 0u) {{
            mantissa = rest;
            rest = 0u;
            scale = 62u;
        }}
        if ((mantissa >> 16u) == 0u) {{
            mantissa = (mantissa << 16u) | (rest >> 16u);
            rest = rest << 16u;
            scale = scale + 16u;
        }}
        if ((mantissa >> 24u) == 0u) {{
            mantissa = (mantissa << 8u) | (rest >> 24u);
            rest = rest << 8u;
            scale = scale + 8u;
        }}
        if ((mantissa >> 28u) == 0u) {{
            mantissa = (mantissa << 4u) | (rest >> 28u);
            rest = rest << 4u;
            scale = scale + 4u;
        }}
        if ((mantissa >> 30u) == 0u) {{
            mantissa = (mantissa << 2u) | (rest >> 30u);
            rest = rest << 2u;
            scale = scale + 2u;
        }}
        if ((mantissa >> 31u) == 0u) {{
            mantissa = (mantissa << 1u) | (rest >> 31u);
            rest = rest << 1u;
            scale = scale + 1u;
        }}
        uint reduced = {MULTIPLY_HIGH}(mantissa, 0x{_PI_OVER_2:08x}u);
        if (mantissa < 0x{_REDUCED_TOP_BIT:08x}u) {{
            reduced = (reduced << 1u) | ((mantissa * 0x{_PI_OVER_2:08x}u) >> 31u);
        }} else {{
            scale = scale - 1u;
        }}
        uint square = {MULTIPLY_HIGH}(reduced, reduced);
        uint z = 2u * scale - 64u < 32u ? square >> (2u * scale - 64u < 32u ? 2u * scale - 64u : 0u) : 0u;
        if (use_cosine) {{
{_horner("q", _COSINE)}
            uint drop = {MULTIPLY_HIGH}(z, q);
            uint fraction = 0u - drop;
            uint rounded = (fraction >> 8u) + ((fraction >> 7u) & 1u);
            result = drop == 0u ? 0x3f800000u : (125u << 23u) + rounded;
        }} else {{
{_horner("p", _SINE)}
            uint drop = {MULTIPLY_HIGH}(z, p);
            uint product_low = reduced * drop;
            uint sine_low = 0u - product_low;
            uint sine = reduced - {MULTIPLY_HIGH}(reduced, drop) - (product_low != 0u ? 1u : 0u);
            uint sine_scale = scale;
            if ((sine >> 31u) == 0u) {{
                sine = (sine << 1u) | (sine_low >> 31u);
                sine_scale = scale + 1u;
            }}
            uint rounded = (sine >> 8u) + ((sine >> 7u) & 1u);
            result = ((157u - sine_scale) << 23u) + rounded;
        }}
    }}
    return asfloat(sign != 0u ? result ^ 0x80000000u : result);
}}
"""


# Parameters and locals of the functions above. They take the reserved prefix too, so that no helper declaration
# shadows a global of the program (which Metal output refuses) on any target.
_LOCALS = (
    "index a b value cosine low middle upper cross rest bits magnitude exponent significand first "
    "middle_word last_high middle_low high quadrant negative use_cosine sign result mantissa scale "
    "reduced square z q p drop fraction rounded product_low sine_low sine sine_scale"
).split()
_LOCAL_PATTERN = re.compile(r"\b(" + "|".join(_LOCALS) + r")\b")


def libm_trig_support() -> str:
    """The CrossGL definitions the sine and cosine helpers call, in dependency order."""
    text = "\n".join(
        part.strip() for part in (_window_function(), _MULTIPLY_HIGH, _TRIG)
    )
    return _LOCAL_PATTERN.sub(lambda match: "cgl_rule_" + match.group(1), text)
