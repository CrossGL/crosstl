"""Binary32 power approximation with unchanged native range-boundary behavior."""

from .source_licenses import SOURCE_LICENSES

# Adapted from OpenLibm src/e_powf.c, revision
# 5fe399749f9276eaa0b8403e507470da05cbbb3f (fdlibm).
# Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
# Developed at SunPro, a Sun Microsystems, Inc. business.
# Permission to use, copy, modify, and distribute this software is freely
# granted, provided that this notice is preserved.

_POWER_SUPPORT = """@source_license(fdlibm)
@precise
@metal_static
float $evaluate(float base, float exponent) {
    uint ix = asuint(base);
    uint iy = asuint(exponent) & 0x7fffffffu;
    float t1 @precise = 0.0;
    float t2 @precise = 0.0;
    if (ix == 0x3f800000u || iy == 0u) { return 1.0; }
    if ($near_one) {
$far_from_one        float t @precise = base - 1.0;
        float w @precise = (t * t) * (0.5 - t * (3.33333343e-1 - t * 0.25));
        float u @precise = 1.4426879883 * t;
        float v @precise = t * 7.0526075433e-6 - w * 1.4426950216;
        t1 = asfloat(asuint(u + v) & 0xfffff000u);
        t2 = v - (t1 - u);
    } else {
        int n = int(ix >> 23u) - 127;
        uint significand = ix & 0x7fffffu;
        ix = significand | 0x3f800000u;
        uint interval = 0u;
        if (significand > 0x1cc471u) {
            if (significand < 0x5db3d7u) {
                interval = 1u;
            } else {
                n += 1;
                ix -= 0x800000u;
            }
        }
        float ax = asfloat(ix);
        float center = interval == 0u ? 1.0 : 1.5;
        float offset_high = interval == 0u ? 0.0 : 5.84960938e-1;
        float offset_low = interval == 0u ? 0.0 : 1.56322085e-6;
        float u @precise = ax - center;
        float v @precise = 1.0 / (ax + center);
        float s @precise = u * v;
        float s_high = asfloat(asuint(s) & 0xfffff000u);
        uint high_word = ((ix >> 1u) & 0xfffff000u) | 0x20000000u;
        float t_high = asfloat(high_word + 0x400000u + (interval << 21u));
        float t_low @precise = ax - (t_high - center);
        float s_low @precise = v * ((u - s_high * t_high) - s_high * t_low);
        float squared @precise = s * s;
        float r @precise = squared * squared * (6.0000002384e-1 + squared *
            (4.2857143283e-1 + squared * (3.3333334327e-1 + squared *
            (2.7272811532e-1 + squared * (2.3066075146e-1 + squared * 2.0697501302e-1)))));
        r += s_low * (s_high + s);
        squared = s_high * s_high;
        t_high = asfloat(asuint(3.0 + squared + r) & 0xfffff000u);
        t_low = r - ((t_high - 3.0) - squared);
        u = s_high * t_high;
        v = s_low * t_high + t_low * s;
        float p_high = asfloat(asuint(u + v) & 0xfffff000u);
        float p_low @precise = v - (p_high - u);
        float z_high @precise = 9.6191406250e-1 * p_high;
        float z_low @precise = -1.1736857402e-4 * p_high + p_low * 9.6179670095e-1 + offset_low;
        float exponent_part = float(n);
        t1 = asfloat(asuint(((z_high + z_low) + offset_high) + exponent_part) & 0xfffff000u);
        t2 = z_low - (((t1 - exponent_part) - offset_high) - z_high);
    }
    float y_high = asfloat(asuint(exponent) & 0xfffff000u);
    float product_low @precise = (exponent - y_high) * t1 + exponent * t2;
    float product_high @precise = y_high * t1;
    float z @precise = product_low + product_high;
$range    uint word = asuint(z);
    uint magnitude = word & 0x7fffffffu;
    int n = 0;
    if (magnitude > 0x3f000000u) {
        int k = int(magnitude >> 23u) - 127;
        uint rounded = word + (0x800000u >> uint(k + 1));
        k = int((rounded & 0x7fffffffu) >> 23u) - 127;
        float integral = asfloat(rounded & ~(0x7fffffu >> uint(k)));
        n = int(((rounded & 0x7fffffu) | 0x800000u) >> uint(23 - k));
        if ((word >> 31u) != 0u) { n = -n; }
        product_high -= integral;
    }
    float t = asfloat(asuint(product_low + product_high) & 0xffff8000u);
    float u @precise = t * 6.93145752e-1;
    float v @precise = (product_low - (t - product_high)) * 6.9314718246e-1 + t * 1.42860654e-6;
    z = u + v;
    float w @precise = v - (z - u);
    t = z * z;
    t1 = z - t * (1.6666667163e-1 + t * (-2.7777778450e-3 + t *
        (6.6137559770e-5 + t * (-1.6533901999e-6 + t * 4.1381369442e-8))));
    float r @precise = (z * t1) / (t1 - 2.0) - (w + z * w);
    z = 1.0 - (r - z);
    uint result_word = asuint(z);
$scale}
"""


# Range boundaries of the ``portable-finite`` profile: a full binade on each side of the binary32 range keeps the
# target's own ``pow``.
_NATIVE_BOUNDARIES = {
    "$near_one": "iy > 0x4d000000u",
    "$far_from_one": (
        """        if (ix < 0x3f7ffff6u || ix > 0x3f800007u) {
            return pow(base, exponent);
        }
"""
    ),
    "$range": (
        """    // Range-boundary behavior remains native until a source result policy is resolved.
    // A full binade on each side separates this approximation from those decisions.
    if (!(z > -125.0 && z < 127.0)) { return pow(base, exponent); }
"""
    ),
    "$scale": (
        """    int scale = int(result_word >> 23u) - 127 + n;
    return asfloat((uint(scale + 127) << 23u) | (result_word & 0x7fffffu));
"""
    ),
}

# Range boundaries of ``precise::pow``: every normal result is computed and overflow is infinite, as in e_powf.c.
# Exponents above 2^27 take the near-one series only near one; elsewhere the general logarithm keeps the result,
# which e_powf.c's shortcut would overflow although x^y can still be finite for x just below one.
# Subnormal results keep the target's own ``pow``, because whether they flush to zero is the target's choice
# (Apple GPUs flush them, Mesa keeps them) and no source result policy selects one yet.
_COMPUTED_BOUNDARIES = {
    "$near_one": "iy > 0x4d000000u && ix >= 0x3f7ffff6u && ix <= 0x3f800007u",
    "$far_from_one": "",
    "$range": (
        """    // Overflow at 2^128, with e_powf.c's rounding margin when z rounds to 128 exactly.
    if (z > 128.0 || (z == 128.0 && product_low + 4.2995665694e-8 > z - product_high)) {
        return asfloat(0x7f800000u);
    }
    if (!(z > -127.0)) { return pow(base, exponent); }
"""
    ),
    "$scale": (
        """    int biased = int(result_word >> 23u) + n;
    if (biased >= 255) { return asfloat(0x7f800000u); }
    if (biased <= 0) { return pow(base, exponent); }
    return asfloat((uint(biased) << 23u) | (result_word & 0x7fffffu));
"""
    ),
}


_BOUNDARY_POLICIES = {"native": _NATIVE_BOUNDARIES, "computed": _COMPUTED_BOUNDARIES}


def binary32_power_support(name, boundaries="native", notice=True):
    """Render a positive-normal-base, finite-exponent approximation.

    The caller handles operand profiles and the power domain. Split logarithm
    and exponent products avoid losing near-one accuracy at large exponents.
    With ``boundaries="native"`` results near underflow and overflow retain
    target-native behavior. ``boundaries="computed"`` computes every normal
    result and overflow, leaving only subnormal results to the target; neither
    defines a cross-target result flushing policy. ``notice=False`` leaves out
    the fdlibm notice comment when an earlier helper already carries it.
    """
    chosen = _BOUNDARY_POLICIES[boundaries]
    text = _POWER_SUPPORT.replace("$evaluate", name)
    for placeholder, replacement in chosen.items():
        text = text.replace(placeholder, replacement)
    if not notice:
        return text
    lines = SOURCE_LICENSES["fdlibm"].splitlines()
    return "".join(f"// {line}\n" for line in lines) + text
