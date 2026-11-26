"""Binary64 ``sinh``, ``cosh``, ``tanh``, ``asinh``, ``acosh`` and ``atanh`` in binary64 arithmetic.

``asinh``, ``acosh`` and ``atanh`` are fdlibm 5.3's s_asinh.c, e_acosh.c and e_atanh.c with their s_log1p.c, and the
smallest ``cosh`` arguments use its s_expm1.c (the notice is in :mod:`crosstl.translator.source_licenses`), written with
unsigned word arithmetic where fdlibm tests signed words, with ``log`` from :mod:`.exp_log` and the square root of
:mod:`.roots`. ``sinh``, ``cosh`` and ``tanh`` round once: fdlibm's forms reach two ulps, above CUDA's one-ulp bound for
``cosh`` and ``tanh``. Every function has the special values of C99 Annex F.9.2:

* ``expm1`` (support): ``exp(x) - 1`` without cancellation. ``x = k ln2 + r`` as in ``exp``, a degree-5 rational
  approximation in ``r^2 / 2`` and a reconstruction that adds ``-1`` before or after the scaling by ``2^k``.
* ``log1p`` (support): ``log(1 + x)`` from ``1 + x = 2^k (1 + f)``, with the rounding error of ``1 + x`` carried as
  a correction term, and the polynomial of ``log``.
* ``sinh`` and ``tanh`` below 0.5 and 0.25: their Taylor series to the term below 2^-66 of the result.
* ``sinh``, ``cosh`` and ``tanh`` up to 22: ``e^|x| / 2 +- e^-|x| / 2`` and ``1 - 2 / (e^(2|x|) + 1)``, from the
  double-double ``e^x`` of :mod:`.extended`, with each sum, product and quotient carried as an unevaluated pair.
  ``cosh`` below ``ln2 / 2`` keeps fdlibm's ``1 + expm1(|x|)^2 / (2 (1 + expm1(|x|)))``.
* above 22: ``+-e^|x| / 2`` as ``(hi + lo) 2^(k - 1)``, which stays finite up to the overflow threshold of 710.48,
  and ``tanh(x) = +-1``.
* ``asinh(x) = sign(x) log1p(|x| + x^2 / (1 + sqrt(1 + x^2)))`` below 2, the ``log`` form up to 2^28 and
  ``log(|x|) + ln2`` above; ``acosh`` likewise for ``x >= 1``; ``atanh(x) = sign(x) log1p(2|x| / (1 - |x|)) / 2``.

``tests/conformance/libm64_accuracy.py`` measures the CrossGL text below: ``sinh``, ``cosh`` and ``tanh`` within 0.6
ulp, ``asinh``, ``acosh`` and ``atanh`` within 1.5.
"""

from __future__ import annotations

from .bits import ABS, BITS, HIGH, LOW, PREFIX, SCALE, WITH_HIGH, render
from .exp_log import LOG
from .extended import EXP_DD, SPLIT_PRODUCT_ERROR
from .roots import SQRT

__all__ = ["ACOSH", "ASINH", "ATANH", "COSH", "SINH", "TANH", "sources"]

EXPM1 = PREFIX + "expm1"
LOG1P = PREFIX + "log1p"
SINH = PREFIX + "sinh"
COSH = PREFIX + "cosh"
TANH = PREFIX + "tanh"
ASINH = PREFIX + "asinh"
ACOSH = PREFIX + "acosh"
ATANH = PREFIX + "atanh"

_COMMON = {
    "LN2_HI": (0x3FE62E42, 0xFEE00000),
    "LN2_LO": (0x3DEA39EF, 0x35793C76),
    "LN2": (0x3FE62E42, 0xFEFA39EF),
    "INFINITY": (0x7FF00000, 0),
    "NAN": (0x7FF80000, 0),
}

_EXPM1_CONSTANTS = {
    **_COMMON,
    "O_THRESHOLD": (0x40862E42, 0xFEFA39EF),
    "INV_LN2": (0x3FF71547, 0x652B82FE),
    "Q1": (0xBFA11111, 0x111110F4),
    "Q2": (0x3F5A01A0, 0x19FE5585),
    "Q3": (0xBF14CE19, 0x9EAADBB7),
    "Q4": (0x3ED0CFCA, 0x86E65239),
    "Q5": (0xBE8AFDB7, 0x6E09C32D),
}

_EXPM1 = f"""@source_license(fdlibm)
@precise
double {EXPM1}(double x) {{
    uint hx = {HIGH}(x);
    uint xsb = hx >> 31u;
    uint ax = hx & 0x7FFFFFFFu;
    if (ax >= 0x4043687Au) {{
        if (ax >= 0x40862E42u) {{
            if (ax >= 0x7FF00000u) {{
                if (((ax & 0x000FFFFFu) | {LOW}(x)) != 0u) {{
                    return x;
                }}
                return xsb == 0u ? x : -1.0;
            }}
            if (x > $O_THRESHOLD) {{
                return $INFINITY;
            }}
        }}
        if (xsb != 0u) {{
            return -1.0;
        }}
    }}
    double r = x;
    double c = 0.0;
    int k = 0;
    if (ax > 0x3FD62E42u) {{
        double hi = 0.0;
        double lo = 0.0;
        if (ax < 0x3FF0A2B2u) {{
            if (xsb == 0u) {{
                hi = x - $LN2_HI;
                lo = $LN2_LO;
                k = 1;
            }} else {{
                hi = x + $LN2_HI;
                lo = -$LN2_LO;
                k = -1;
            }}
        }} else {{
            k = int($INV_LN2 * x + (xsb == 0u ? 0.5 : -0.5));
            double kd = double(k);
            hi = x - kd * $LN2_HI;
            lo = kd * $LN2_LO;
        }}
        r = hi - lo;
        c = (hi - r) - lo;
    }} else if (ax < 0x3C900000u) {{
        return x;
    }}
    double hfx = 0.5 * r;
    double hxs = r * hfx;
    double r1 = 1.0 + hxs * ($Q1 + hxs * ($Q2 + hxs * ($Q3 + hxs * ($Q4 + hxs * $Q5))));
    double t = 3.0 - r1 * hfx;
    double e = hxs * ((r1 - t) / (6.0 - r * t));
    if (k == 0) {{
        return r - (r * e - hxs);
    }}
    e = r * (e - c) - c;
    e -= hxs;
    if (k == -1) {{
        return 0.5 * (r - e) - 0.5;
    }}
    if (k == 1) {{
        if (r < -0.25) {{
            return -2.0 * (e - (r + 0.5));
        }}
        return 1.0 + 2.0 * (r - e);
    }}
    if (k <= -2 || k > 56) {{
        return {SCALE}(1.0 - (e - r), k) - 1.0;
    }}
    if (k < 20) {{
        double one_minus = {BITS}(0x3FF00000u - (0x00200000u >> uint(k)), 0u);
        return {SCALE}(one_minus - (e - r), k);
    }}
    double power = {BITS}(uint(1023 - k) << 20u, 0u);
    return {SCALE}((r - (e + power)) + 1.0, k);
}}"""

_LOG1P_CONSTANTS = {
    **_COMMON,
    "TWO_THIRDS": (0x3FE55555, 0x55555555),
    "LP1": (0x3FE55555, 0x55555593),
    "LP2": (0x3FD99999, 0x9997FA04),
    "LP3": (0x3FD24924, 0x94229359),
    "LP4": (0x3FCC71C5, 0x1D8E78AF),
    "LP5": (0x3FC74664, 0x96CB03DE),
    "LP6": (0x3FC39A09, 0xD078C69F),
    "LP7": (0x3FC2F112, 0xDF3E5244),
}

_LOG1P = f"""@source_license(fdlibm)
@precise
double {LOG1P}(double x) {{
    uint hx = {HIGH}(x);
    uint ax = hx & 0x7FFFFFFFu;
    uint sign = hx >> 31u;
    int k = 1;
    double f = 0.0;
    double c = 0.0;
    uint hu = 0u;
    if (sign != 0u || hx < 0x3FDA827Au) {{
        if (ax >= 0x3FF00000u) {{
            if (x == -1.0) {{
                return -$INFINITY;
            }}
            if (ax > 0x7FF00000u || (ax == 0x7FF00000u && {LOW}(x) != 0u)) {{
                return x;
            }}
            return $NAN;
        }}
        if (ax < 0x3E200000u) {{
            if (ax < 0x3C900000u) {{
                return x;
            }}
            return x - x * x * 0.5;
        }}
        if (sign == 0u || hx <= 0xBFD2BEC3u) {{
            k = 0;
            f = x;
            hu = 1u;
        }}
    }}
    if (sign == 0u && hx >= 0x7FF00000u) {{
        return x;
    }}
    if (k != 0) {{
        double u = x;
        if (sign != 0u || hx < 0x43400000u) {{
            u = 1.0 + x;
            hu = {HIGH}(u);
            k = int(hu >> 20u) - 1023;
            c = k > 0 ? 1.0 - (u - x) : x - (u - 1.0);
            c /= u;
        }} else {{
            hu = {HIGH}(u);
            k = int(hu >> 20u) - 1023;
            c = 0.0;
        }}
        hu = hu & 0x000FFFFFu;
        if (hu < 0x6A09Eu) {{
            u = {WITH_HIGH}(u, hu | 0x3FF00000u);
        }} else {{
            k += 1;
            u = {WITH_HIGH}(u, hu | 0x3FE00000u);
            hu = (0x00100000u - hu) >> 2u;
        }}
        f = u - 1.0;
    }}
    double hfsq = 0.5 * f * f;
    double dk = double(k);
    if (hu == 0u) {{
        if (f == 0.0) {{
            if (k == 0) {{
                return 0.0;
            }}
            c += dk * $LN2_LO;
            return dk * $LN2_HI + c;
        }}
        double rf = hfsq * (1.0 - $TWO_THIRDS * f);
        if (k == 0) {{
            return f - rf;
        }}
        return dk * $LN2_HI - ((rf - (dk * $LN2_LO + c)) - f);
    }}
    double s = f / (2.0 + f);
    double z = s * s;
    double rs = z * ($LP1 + z * ($LP2 + z * ($LP3 + z * ($LP4 + z * ($LP5 + z * ($LP6 + z * $LP7))))));
    if (k == 0) {{
        return f - (hfsq - s * (hfsq + rs));
    }}
    return dk * $LN2_HI - ((hfsq - (s * (hfsq + rs) + (dk * $LN2_LO + c))) - f);
}}"""

_SERIES = {
    **_COMMON,
    # 1/(2n+1)!, n = 1 .. 8: sinh(x) = x + x z (S1 + z (S2 + ...)) with z = x^2.
    "S1": (0x3FC55555, 0x55555555),
    "S2": (0x3F811111, 0x11111111),
    "S3": (0x3F2A01A0, 0x1A01A01A),
    "S4": (0x3EC71DE3, 0xA556C734),
    "S5": (0x3E5AE645, 0x67F544E4),
    "S6": (0x3DE61246, 0x13A86D09),
    "S7": (0x3D6AE7F3, 0xE733B81F),
    "S8": (0x3CE952C7, 0x7030AD4A),
    # 2^(2n) (2^(2n) - 1) B(2n) / (2n)!, n = 2 .. 13: tanh(x) = x + x z (T1 + z (T2 + ...)).
    "T1": (0xBFD55555, 0x55555555),
    "T2": (0x3FC11111, 0x11111111),
    "T3": (0xBFABA1BA, 0x1BA1BA1C),
    "T4": (0x3F9664F4, 0x882C10FA),
    "T5": (0xBF8226E3, 0x55E6C23D),
    "T6": (0x3F6D6D3D, 0x0E157DE0),
    "T7": (0xBF57DA36, 0x452B75E3),
    "T8": (0x3F435582, 0x48036744),
    "T9": (0xBF2F57D7, 0x734D1664),
    "T10": (0x3F1967E1, 0x8AFCAFAD),
    "T11": (0xBF0497D8, 0xEEA25259),
    "T12": (0x3EF0B132, 0xD39A6050),
}

# e^a / 2 and e^-a / 2 for 0.25 <= a < 22, each as an unevaluated sum: ``big + big_lo`` and ``small + small_lo``.
# 1 / (hi + lo) is the quotient q = 1 / hi and its correction q (1 - q (hi + lo)), whose residual 1 - q hi is exact.
_HALF_EXPONENTIALS = f"""        dvec3 e = {EXP_DD}(a);
        int k = int(e.z);
        double big = {SCALE}(e.x, k - 1);
        double big_lo = {SCALE}(e.y, k - 1);
        double q = 1.0 / e.x;
        double p = q * e.x;
        double residual = ((1.0 - p) - {SPLIT_PRODUCT_ERROR}(q, e.x, p)) - q * e.y;
        double small = {SCALE}(q, -k - 1);
        double small_lo = {SCALE}(q * residual, -k - 1);"""

_OVERFLOW_EDGE = (
    """(ax < 0x408633CEu || (ax == 0x408633CEu && {LOW}(x) <= 0x8FB9F87Du))"""
)

_SINH = f"""@precise
double {SINH}(double x) {{
    uint hx = {HIGH}(x);
    uint ax = hx & 0x7FFFFFFFu;
    if (ax >= 0x7FF00000u || ax < 0x3E300000u) {{
        return x;
    }}
    if (ax < 0x3FE00000u) {{
        double z = x * x;
        return x + x * (z * ($S1 + z * ($S2 + z * ($S3 + z * ($S4 + z * ($S5 + z * ($S6 + z * ($S7 + z * ($S8)))))))));
    }}
    double sign = (hx >> 31u) == 0u ? 1.0 : -1.0;
    double a = {ABS}(x);
    if (ax < 0x40360000u) {{
{_HALF_EXPONENTIALS}
        double s = big - small;
        double s_error = (big - s) - small;
        return sign * (s + (s_error + (big_lo - small_lo)));
    }}
    if {_OVERFLOW_EDGE.format(LOW=LOW)} {{
        dvec3 e = {EXP_DD}(a);
        return sign * {SCALE}(e.x, int(e.z) - 1);
    }}
    return sign * $INFINITY;
}}"""

_COSH = f"""@source_license(fdlibm)
@precise
double {COSH}(double x) {{
    uint ax = {HIGH}(x) & 0x7FFFFFFFu;
    if (ax >= 0x7FF00000u) {{
        return x * x;
    }}
    double a = {ABS}(x);
    if (ax < 0x3FD62E43u) {{
        double t = {EXPM1}(a);
        double w = 1.0 + t;
        if (ax < 0x3C800000u) {{
            return w;
        }}
        return 1.0 + (t * t) / (w + w);
    }}
    if (ax < 0x40360000u) {{
{_HALF_EXPONENTIALS}
        double s = big + small;
        double s_error = (big - s) + small;
        return s + (s_error + (big_lo + small_lo));
    }}
    if {_OVERFLOW_EDGE.format(LOW=LOW)} {{
        dvec3 e = {EXP_DD}(a);
        return {SCALE}(e.x, int(e.z) - 1);
    }}
    return $INFINITY;
}}"""

# tanh(a) = 1 - 2 / (e^(2a) + 1) for 0.25 <= a < 22: the denominator and the quotient are carried as unevaluated
# sums, and 1 - q loses at most two bits, so one rounding at the end leaves the result within 0.6 ulp.
_TANH = f"""@precise
double {TANH}(double x) {{
    uint hx = {HIGH}(x);
    uint ax = hx & 0x7FFFFFFFu;
    double sign = (hx >> 31u) == 0u ? 1.0 : -1.0;
    if (ax >= 0x7FF00000u) {{
        if (ax > 0x7FF00000u || {LOW}(x) != 0u) {{
            return x;
        }}
        return sign;
    }}
    if (ax >= 0x40360000u) {{
        return sign;
    }}
    if (ax < 0x3E300000u) {{
        return x;
    }}
    if (ax < 0x3FD00000u) {{
        double z = x * x;
        return x + x * (z * ($T1 + z * ($T2 + z * ($T3 + z * ($T4 + z * ($T5 + z * ($T6 + z * ($T7 + z * ($T8 + z * ($T9 + z * ($T10 + z * ($T11 + z * ($T12)))))))))))));
    }}
    dvec3 e = {EXP_DD}(2.0 * {ABS}(x));
    int k = int(e.z);
    double big = {SCALE}(e.x, k);
    double d = big + 1.0;
    double d_lo = ((big - d) + 1.0) + {SCALE}(e.y, k);
    double q = 2.0 / d;
    double p = q * d;
    double q_lo = (((2.0 - p) - {SPLIT_PRODUCT_ERROR}(q, d, p)) - q * d_lo) / d;
    double s = 1.0 - q;
    double s_error = (1.0 - s) - q;
    return sign * (s + (s_error - q_lo));
}}"""

_ASINH = f"""@source_license(fdlibm)
@precise
double {ASINH}(double x) {{
    uint hx = {HIGH}(x);
    uint ax = hx & 0x7FFFFFFFu;
    if (ax >= 0x7FF00000u || ax < 0x3E300000u) {{
        return x;
    }}
    double a = {ABS}(x);
    double w = 0.0;
    if (ax > 0x41B00000u) {{
        w = {LOG}(a) + $LN2;
    }} else if (ax > 0x40000000u) {{
        w = {LOG}(2.0 * a + 1.0 / ({SQRT}(x * x + 1.0) + a));
    }} else {{
        double t = x * x;
        w = {LOG1P}(a + t / (1.0 + {SQRT}(1.0 + t)));
    }}
    return (hx >> 31u) == 0u ? w : -w;
}}"""

_ACOSH = f"""@source_license(fdlibm)
@precise
double {ACOSH}(double x) {{
    uint hx = {HIGH}(x);
    uint ax = hx & 0x7FFFFFFFu;
    if ((hx >> 31u) != 0u || hx < 0x3FF00000u) {{
        if (ax > 0x7FF00000u || (ax == 0x7FF00000u && {LOW}(x) != 0u)) {{
            return x;
        }}
        return $NAN;
    }}
    if (hx >= 0x41B00000u) {{
        if (hx >= 0x7FF00000u) {{
            return x;
        }}
        return {LOG}(x) + $LN2;
    }}
    if (hx == 0x3FF00000u && {LOW}(x) == 0u) {{
        return 0.0;
    }}
    if (hx > 0x40000000u) {{
        double t = x * x;
        return {LOG}(2.0 * x - 1.0 / (x + {SQRT}(t - 1.0)));
    }}
    double t = x - 1.0;
    return {LOG1P}(t + {SQRT}(2.0 * t + t * t));
}}"""

_ATANH = f"""@source_license(fdlibm)
@precise
double {ATANH}(double x) {{
    uint hx = {HIGH}(x);
    uint lx = {LOW}(x);
    uint ax = hx & 0x7FFFFFFFu;
    if (ax > 0x3FF00000u || (ax == 0x3FF00000u && lx != 0u)) {{
        if (ax > 0x7FF00000u || (ax == 0x7FF00000u && lx != 0u)) {{
            return x;
        }}
        return $NAN;
    }}
    if (ax == 0x3FF00000u) {{
        return (hx >> 31u) == 0u ? $INFINITY : -$INFINITY;
    }}
    if (ax < 0x3E300000u) {{
        return x;
    }}
    double a = {ABS}(x);
    double t = 0.0;
    if (ax < 0x3FE00000u) {{
        double twice = a + a;
        t = 0.5 * {LOG1P}(twice + twice * a / (1.0 - a));
    }} else {{
        t = 0.5 * {LOG1P}((a + a) / (1.0 - a));
    }}
    return (hx >> 31u) == 0u ? t : -t;
}}"""


def sources() -> list[tuple[str, str]]:
    """``(name, CrossGL definition)`` of the hyperbolic functions and their support, each after what it calls."""
    return [
        (EXPM1, render(_EXPM1, _EXPM1_CONSTANTS)),
        (LOG1P, render(_LOG1P, _LOG1P_CONSTANTS)),
        (SINH, render(_SINH, _SERIES)),
        (COSH, render(_COSH, _SERIES)),
        (TANH, render(_TANH, _SERIES)),
        (ASINH, render(_ASINH, _COMMON)),
        (ACOSH, render(_ACOSH, _COMMON)),
        (ATANH, render(_ATANH, _COMMON)),
    ]
