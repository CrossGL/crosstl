"""Binary64 ``pow``, ``exp2`` and ``log2`` in binary64 arithmetic: fdlibm's e_pow.c.

``__ieee754_pow`` of fdlibm 5.3 (the notice is in :mod:`crosstl.translator.source_licenses`) computes ``log2|x|``
as a head ``t1`` with a zero low word plus a tail ``t2``, multiplies it by ``y`` split the same way, and evaluates
``2^(p_h + p_l)`` from the extended product. Those two halves are helpers of their own here, so ``exp2`` and
``log2`` use them directly:

* :data:`LOG2_PARTS` is ``log2(ax) = t1 + t2`` for a positive finite ``ax`` (subnormals scaled by 2^53 first);
* :data:`EXP2_PARTS` is ``2^(p_h + p_l)`` with overflow to infinity and underflow to zero, carrying the sign word of
  the result, and ends with one rounding to a normal or subnormal result;
* ``pow`` follows e_pow.c, with the C99 Annex F special values CUDA and HIP give: ``pow(x, +-0) = 1`` and
  ``pow(+1, y) = 1`` for every ``x`` and ``y`` (NaN included), ``pow(-1, +-inf) = 1``, and a NaN for a negative
  finite ``x`` and a finite non-integral ``y``;
* ``exp2(x)`` is ``2^(x + 0)``, and ``log2(x)`` is ``t1 + t2`` rounded once, with the special values of ``log``.

Three departures from e_pow.c keep the bound and the target independence. For ``|y| > 2^31`` and ``x`` within
2^-20 of one, e_pow.c replaces the logarithm by the series ``t - t^2/2 + t^3/3 - t^4/4``, whose head ``ivln2_h * t``
(24 by up to 32 significant bits) is not exact: ``pow(0.9999999145748363, -4048175887.0471315)`` was 313 ulp off. The
general logarithm, accurate enough for every ``z = y log2|x|`` that has a finite nonzero result, is used instead. The
shortcut for ``y = 0.5`` (``sqrt``, which HLSL lacks for ``double``) is left out, and the words are tested unsigned. ``tests/conformance/libm64_accuracy.py`` measures
the CrossGL text below.
"""

from __future__ import annotations

from .bits import BITS, HIGH, LOW, PREFIX, SCALE, WITH_HIGH, WITH_LOW, render

__all__ = ["EXP2", "EXP2_PARTS", "LOG2", "LOG2_PARTS", "POW", "sources"]

LOG2_PARTS = PREFIX + "log2_parts"
EXP2_PARTS = PREFIX + "exp2_parts"
POW = PREFIX + "pow"
EXP2 = PREFIX + "exp2"
LOG2 = PREFIX + "log2"

_CONSTANTS = {
    "TWO53": (0x43400000, 0),
    "DP_H1": (0x3FE2B803, 0x40000000),
    "DP_L1": (0x3E4CFDEB, 0x43CFD006),
    "L1": (0x3FE33333, 0x33333303),
    "L2": (0x3FDB6DB6, 0xDB6FABFF),
    "L3": (0x3FD55555, 0x518F264D),
    "L4": (0x3FD17460, 0xA91D4101),
    "L5": (0x3FCD864A, 0x93C9DB65),
    "L6": (0x3FCA7E28, 0x4A454EEF),
    "P1": (0x3FC55555, 0x5555553E),
    "P2": (0xBF66C16C, 0x16BEBD93),
    "P3": (0x3F11566A, 0xAF25DE2C),
    "P4": (0xBEBBBD41, 0xC5D26BF1),
    "P5": (0x3E663769, 0x72BEA4D0),
    "LG2": (0x3FE62E42, 0xFEFA39EF),
    "LG2_H": (0x3FE62E43, 0x00000000),
    "LG2_L": (0xBE205C61, 0x0CA86C39),
    "OVT": (0x3C971547, 0x652B82FE),
    "CP": (0x3FEEC709, 0xDC3A03FD),
    "CP_H": (0x3FEEC709, 0xE0000000),
    "CP_L": (0xBE3E2FE0, 0x145B01F5),
    "INFINITY": (0x7FF00000, 0),
    "NAN": (0x7FF80000, 0),
}

_LOG2_PARTS = f"""@source_license(fdlibm)
@precise
dvec2 {LOG2_PARTS}(double ax) {{
    double v = ax;
    uint ix = {HIGH}(ax);
    int n = 0;
    if (ix < 0x00100000u) {{
        v = ax * $TWO53;
        n = -53;
        ix = {HIGH}(v);
    }}
    n += int(ix >> 20u) - 1023;
    uint j = ix & 0x000FFFFFu;
    ix = j | 0x3FF00000u;
    uint k = 0u;
    if (j > 0x3988Eu) {{
        if (j < 0xBB67Au) {{
            k = 1u;
        }} else {{
            n += 1;
            ix -= 0x00100000u;
        }}
    }}
    v = {WITH_HIGH}(v, ix);
    double bp = k == 1u ? 1.5 : 1.0;
    double dp_h = k == 1u ? $DP_H1 : 0.0;
    double dp_l = k == 1u ? $DP_L1 : 0.0;
    double u = v - bp;
    double w = 1.0 / (v + bp);
    double ss = u * w;
    double s_h = {WITH_LOW}(ss, 0u);
    double t_h = {BITS}(((ix >> 1u) | 0x20000000u) + 0x00080000u + (k << 18u), 0u);
    double t_l = v - (t_h - bp);
    double s_l = w * ((u - s_h * t_h) - s_h * t_l);
    double s2 = ss * ss;
    double r = s2 * s2 * ($L1 + s2 * ($L2 + s2 * ($L3 + s2 * ($L4 + s2 * ($L5 + s2 * $L6)))));
    r += s_l * (s_h + ss);
    s2 = s_h * s_h;
    t_h = {WITH_LOW}(3.0 + s2 + r, 0u);
    t_l = r - ((t_h - 3.0) - s2);
    u = s_h * t_h;
    w = s_l * t_h + t_l * ss;
    double p_h = {WITH_LOW}(u + w, 0u);
    double p_l = w - (p_h - u);
    double z_h = $CP_H * p_h;
    double z_l = $CP_L * p_h + p_l * $CP + dp_l;
    double t = double(n);
    double t1 = {WITH_LOW}(((z_h + z_l) + dp_h) + t, 0u);
    double t2 = z_l - (((t1 - t) - dp_h) - z_h);
    return dvec2(t1, t2);
}}"""

_EXP2_PARTS = f"""@source_license(fdlibm)
@precise
double {EXP2_PARTS}(double p_h, double p_l, uint sign) {{
    double z = p_h + p_l;
    uint j = {HIGH}(z);
    uint i = {LOW}(z);
    uint magnitude = j & 0x7FFFFFFFu;
    if ((j >> 31u) == 0u && magnitude >= 0x40900000u) {{
        if (magnitude != 0x40900000u || i != 0u || p_l + $OVT > z - p_h) {{
            return {BITS}(sign | 0x7FF00000u, 0u);
        }}
    }} else if ((j >> 31u) != 0u && magnitude >= 0x4090CC00u) {{
        if (magnitude != 0x4090CC00u || i != 0u || p_l <= z - p_h) {{
            return {BITS}(sign, 0u);
        }}
    }}
    double head = p_h;
    int n = 0;
    if (magnitude > 0x3FE00000u) {{
        int k = int(magnitude >> 20u) - 1023;
        uint rounded = j + (0x00100000u >> uint(k + 1));
        k = int((rounded & 0x7FFFFFFFu) >> 20u) - 1023;
        double integral = {BITS}(rounded & ~(0x000FFFFFu >> uint(k)), 0u);
        n = int(((rounded & 0x000FFFFFu) | 0x00100000u) >> uint(20 - k));
        if ((j >> 31u) != 0u) {{
            n = -n;
        }}
        head -= integral;
    }}
    double t = {WITH_LOW}(p_l + head, 0u);
    double u = t * $LG2_H;
    double v = (p_l - (t - head)) * $LG2 + t * $LG2_L;
    z = u + v;
    double w = v - (z - u);
    t = z * z;
    double t1 = z - t * ($P1 + t * ($P2 + t * ($P3 + t * ($P4 + t * $P5))));
    double r = (z * t1) / (t1 - 2.0) - (w + z * w);
    double y = {SCALE}(1.0 - (r - z), n);
    return {BITS}({HIGH}(y) ^ sign, {LOW}(y));
}}"""

_POW = f"""@source_license(fdlibm)
@precise
double {POW}(double x, double y) {{
    uint hx = {HIGH}(x);
    uint lx = {LOW}(x);
    uint hy = {HIGH}(y);
    uint ly = {LOW}(y);
    uint ix = hx & 0x7FFFFFFFu;
    uint iy = hy & 0x7FFFFFFFu;
    if ((iy | ly) == 0u || (hx == 0x3FF00000u && lx == 0u)) {{
        return 1.0;
    }}
    if (ix > 0x7FF00000u || (ix == 0x7FF00000u && lx != 0u)) {{
        return x;
    }}
    if (iy > 0x7FF00000u || (iy == 0x7FF00000u && ly != 0u)) {{
        return y;
    }}
    bool negative_x = (hx >> 31u) != 0u;
    bool negative_y = (hy >> 31u) != 0u;
    uint yisint = 0u;
    if (negative_x) {{
        if (iy >= 0x43400000u) {{
            yisint = 2u;
        }} else if (iy >= 0x3FF00000u) {{
            uint e = (iy >> 20u) - 0x3FFu;
            if (e > 20u) {{
                uint m = ly >> (52u - e);
                if ((m << (52u - e)) == ly) {{
                    yisint = 2u - (m & 1u);
                }}
            }} else if (ly == 0u) {{
                uint m = iy >> (20u - e);
                if ((m << (20u - e)) == iy) {{
                    yisint = 2u - (m & 1u);
                }}
            }}
        }}
    }}
    if (ly == 0u) {{
        if (iy == 0x7FF00000u) {{
            if (ix == 0x3FF00000u && lx == 0u) {{
                return 1.0;
            }}
            if (ix >= 0x3FF00000u) {{
                return negative_y ? 0.0 : y;
            }}
            return negative_y ? -y : 0.0;
        }}
        if (iy == 0x3FF00000u) {{
            return negative_y ? 1.0 / x : x;
        }}
        if (hy == 0x40000000u) {{
            return x * x;
        }}
    }}
    double ax = {BITS}(ix, lx);
    if (lx == 0u && (ix == 0x7FF00000u || ix == 0u || ix == 0x3FF00000u)) {{
        double z = 1.0;
        if (ix == 0u) {{
            z = negative_y ? $INFINITY : 0.0;
        }} else if (ix == 0x7FF00000u) {{
            z = negative_y ? 0.0 : $INFINITY;
        }}
        if (negative_x) {{
            if (ix == 0x3FF00000u && yisint == 0u) {{
                return $NAN;
            }}
            if (yisint == 1u) {{
                return -z;
            }}
        }}
        return z;
    }}
    if (negative_x && yisint == 0u) {{
        return $NAN;
    }}
    uint sign = negative_x && yisint == 1u ? 0x80000000u : 0u;
    if (iy > 0x41E00000u) {{
        if (iy > 0x43F00000u) {{
            if (ix <= 0x3FEFFFFFu) {{
                return negative_y ? $INFINITY : 0.0;
            }}
            if (ix >= 0x3FF00000u) {{
                return negative_y ? 0.0 : $INFINITY;
            }}
        }}
        if (ix < 0x3FEFFFFFu) {{
            return {BITS}(sign | (negative_y ? 0x7FF00000u : 0u), 0u);
        }}
        if (ix > 0x3FF00000u) {{
            return {BITS}(sign | (negative_y ? 0u : 0x7FF00000u), 0u);
        }}
    }}
    dvec2 parts = {LOG2_PARTS}(ax);
    double t1 = parts.x;
    double t2 = parts.y;
    double y1 = {WITH_LOW}(y, 0u);
    double p_l = (y - y1) * t1 + y * t2;
    double p_h = y1 * t1;
    return {EXP2_PARTS}(p_h, p_l, sign);
}}"""

_EXP2 = f"""@source_license(fdlibm)
@precise
double {EXP2}(double x) {{
    uint hx = {HIGH}(x);
    uint ix = hx & 0x7FFFFFFFu;
    if (ix >= 0x7FF00000u) {{
        if (ix > 0x7FF00000u || {LOW}(x) != 0u) {{
            return x;
        }}
        return (hx >> 31u) == 0u ? x : 0.0;
    }}
    return {EXP2_PARTS}(x, 0.0, 0u);
}}"""

_LOG2 = f"""@source_license(fdlibm)
@precise
double {LOG2}(double x) {{
    uint hx = {HIGH}(x);
    uint lx = {LOW}(x);
    uint ix = hx & 0x7FFFFFFFu;
    if ((ix | lx) == 0u) {{
        return -$INFINITY;
    }}
    if (ix > 0x7FF00000u || (ix == 0x7FF00000u && lx != 0u)) {{
        return x;
    }}
    if ((hx >> 31u) != 0u) {{
        return $NAN;
    }}
    if (ix == 0x7FF00000u) {{
        return x;
    }}
    dvec2 parts = {LOG2_PARTS}(x);
    return parts.x + parts.y;
}}"""


def sources() -> list[tuple[str, str]]:
    """``(name, CrossGL definition)`` of the helpers, each after the helpers it calls."""
    return [
        (name, render(text, _CONSTANTS))
        for name, text in (
            (LOG2_PARTS, _LOG2_PARTS),
            (EXP2_PARTS, _EXP2_PARTS),
            (POW, _POW),
            (EXP2, _EXP2),
            (LOG2, _LOG2),
        )
    ]
