"""Binary64 ``exp`` and ``log`` in binary64 arithmetic: fdlibm's e_exp.c and e_log.c.

The algorithms and coefficients are those of fdlibm 5.3 (``__ieee754_exp``, ``__ieee754_log``; the notice is in
:mod:`crosstl.translator.source_licenses`), written with unsigned word arithmetic where fdlibm tests signed words, and
with C99 Annex F special values:

* ``exp``: NaN gives NaN, ``exp(+inf) = +inf``, ``exp(-inf) = +0``, overflow above 709.78 gives ``+inf`` and
  underflow below -745.13 gives ``+0``. The argument is reduced to ``r = x - k ln2`` with ``|r| <= ln2/2`` in two
  parts (``ln2`` split into a 32-bit head, so ``k * head`` is exact), a degree-5 rational approximation gives
  ``exp(r)``, and the result is scaled by ``2^k`` with one rounding.
* ``log``: ``log(+-0) = -inf``, ``log(x < 0) = NaN``, ``log(+inf) = +inf``, NaN gives NaN, ``log(1) = +0``.
  ``x = 2^k (1 + f)`` with ``sqrt(2)/2 < 1 + f < sqrt(2)``, subnormal operands scaled by 2^54 first, and
  ``log(1 + f)`` from ``s = f / (2 + f)`` and a degree-14 polynomial in ``s``.

fdlibm documents both within one ulp; ``tests/conformance/libm64_accuracy.py`` measures the CrossGL text below.
"""

from __future__ import annotations

from .bits import HIGH, LOW, PREFIX, SCALE, WITH_HIGH, render

__all__ = ["EXP", "LOG", "sources"]

EXP = PREFIX + "exp"
LOG = PREFIX + "log"

_EXP_CONSTANTS = {
    "O_THRESHOLD": (0x40862E42, 0xFEFA39EF),
    "U_THRESHOLD": (0xC0874910, 0xD52D3051),
    "LN2_HI": (0x3FE62E42, 0xFEE00000),
    "LN2_LO": (0x3DEA39EF, 0x35793C76),
    "INV_LN2": (0x3FF71547, 0x652B82FE),
    "P1": (0x3FC55555, 0x5555553E),
    "P2": (0xBF66C16C, 0x16BEBD93),
    "P3": (0x3F11566A, 0xAF25DE2C),
    "P4": (0xBEBBBD41, 0xC5D26BF1),
    "P5": (0x3E663769, 0x72BEA4D0),
    "INFINITY": (0x7FF00000, 0),
}

_EXP = f"""@source_license(fdlibm)
@precise
double {EXP}(double x) {{
    uint hx = {HIGH}(x);
    uint xsb = hx >> 31u;
    uint ax = hx & 0x7FFFFFFFu;
    if (ax >= 0x40862E42u) {{
        if (ax >= 0x7FF00000u) {{
            if (((ax & 0x000FFFFFu) | {LOW}(x)) != 0u) {{
                return x;
            }}
            return xsb == 0u ? x : 0.0;
        }}
        if (x > $O_THRESHOLD) {{
            return $INFINITY;
        }}
        if (x < $U_THRESHOLD) {{
            return 0.0;
        }}
    }}
    double r = x;
    double hi = 0.0;
    double lo = 0.0;
    int k = 0;
    if (ax > 0x3FD62E42u) {{
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
    }} else if (ax < 0x3E300000u) {{
        return 1.0 + x;
    }}
    double t = r * r;
    double c = r - t * ($P1 + t * ($P2 + t * ($P3 + t * ($P4 + t * $P5))));
    if (k == 0) {{
        return 1.0 - ((r * c) / (c - 2.0) - r);
    }}
    double y = 1.0 - ((lo - (r * c) / (2.0 - c)) - hi);
    return {SCALE}(y, k);
}}"""

_LOG_CONSTANTS = {
    "LN2_HI": (0x3FE62E42, 0xFEE00000),
    "LN2_LO": (0x3DEA39EF, 0x35793C76),
    "TWO54": (0x43500000, 0),
    "THIRD": (0x3FD55555, 0x55555555),
    "LG1": (0x3FE55555, 0x55555593),
    "LG2": (0x3FD99999, 0x9997FA04),
    "LG3": (0x3FD24924, 0x94229359),
    "LG4": (0x3FCC71C5, 0x1D8E78AF),
    "LG5": (0x3FC74664, 0x96CB03DE),
    "LG6": (0x3FC39A09, 0xD078C69F),
    "LG7": (0x3FC2F112, 0xDF3E5244),
    "INFINITY": (0x7FF00000, 0),
    "NAN": (0x7FF80000, 0),
}

_LOG = f"""@source_license(fdlibm)
@precise
double {LOG}(double x) {{
    uint hx = {HIGH}(x);
    uint lx = {LOW}(x);
    uint ax = hx & 0x7FFFFFFFu;
    if ((ax | lx) == 0u) {{
        return -$INFINITY;
    }}
    if (ax > 0x7FF00000u || (ax == 0x7FF00000u && lx != 0u)) {{
        return x;
    }}
    if ((hx >> 31u) != 0u) {{
        return $NAN;
    }}
    if (hx == 0x7FF00000u) {{
        return x;
    }}
    double v = x;
    int k = 0;
    if (hx < 0x00100000u) {{
        k = -54;
        v = x * $TWO54;
        hx = {HIGH}(v);
    }}
    k += int(hx >> 20u) - 1023;
    hx = hx & 0x000FFFFFu;
    uint i = (hx + 0x95F64u) & 0x00100000u;
    v = {WITH_HIGH}(v, hx | (i ^ 0x3FF00000u));
    k += int(i >> 20u);
    double f = v - 1.0;
    double dk = double(k);
    if (((hx + 2u) & 0x000FFFFFu) < 3u) {{
        if (f == 0.0) {{
            if (k == 0) {{
                return 0.0;
            }}
            return dk * $LN2_HI + dk * $LN2_LO;
        }}
        double q = f * f * (0.5 - $THIRD * f);
        if (k == 0) {{
            return f - q;
        }}
        return dk * $LN2_HI - ((q - dk * $LN2_LO) - f);
    }}
    double s = f / (2.0 + f);
    double z = s * s;
    double w = z * z;
    double t1 = w * ($LG2 + w * ($LG4 + w * $LG6));
    double t2 = z * ($LG1 + w * ($LG3 + w * ($LG5 + w * $LG7)));
    double R = t2 + t1;
    if (hx >= 0x6147Au && hx <= 0x6B851u) {{
        double hfsq = 0.5 * f * f;
        if (k == 0) {{
            return f - (hfsq - s * (hfsq + R));
        }}
        return dk * $LN2_HI - ((hfsq - (s * (hfsq + R) + dk * $LN2_LO)) - f);
    }}
    if (k == 0) {{
        return f - s * (f - R);
    }}
    return dk * $LN2_HI - ((s * (f - R) - dk * $LN2_LO) - f);
}}"""


def sources() -> list[tuple[str, str]]:
    """``(name, CrossGL definition)`` of ``exp`` and ``log``."""
    return [
        (EXP, render(_EXP, _EXP_CONSTANTS)),
        (LOG, render(_LOG, _LOG_CONSTANTS)),
    ]
