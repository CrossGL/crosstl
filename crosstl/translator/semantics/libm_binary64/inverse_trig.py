"""Binary64 ``asin``, ``acos``, ``atan`` and ``atan2`` in binary64 arithmetic: fdlibm's e_asin.c, e_acos.c, s_atan.c
and e_atan2.c.

The algorithms and coefficients are those of fdlibm 5.3 (the notice is in :mod:`crosstl.translator.source_licenses`),
written with unsigned word arithmetic where fdlibm tests signed words, with the square root of :mod:`.roots`, and
with the special values of C99 Annex F.9.1:

* ``asin``: ``asin(+-0) = +-0``, ``asin(+-1) = +-pi/2``, NaN for ``|x| > 1`` and NaN operands. Below 0.5 it is
  ``x + x R(x^2)`` with a rational approximation ``R``; above, ``pi/2 - 2 asin(sqrt((1 - |x|) / 2))``, with the
  square root's low bits carried separately below 0.975.
* ``acos``: ``acos(1) = +0``, ``acos(-1) = pi``, NaN for ``|x| > 1``; ``pi/2 - asin(x)`` below 0.5 and
  ``2 asin(sqrt((1 -+ x) / 2))`` (from ``pi`` for negative ``x``) above, with the same ``R``.
* ``atan``: odd, ``+-pi/2`` for infinite ``x``; ``|x|`` is reduced to within 7/16 of 0, 0.5, 1 or 1.5 (or replaced
  by ``-1/|x|``) and a degree-22 odd polynomial completes the table value ``atan(c)``, which is carried in two parts.
* ``atan2(y, x)``: C99 F.9.1.4 for zeros and infinities, ``atan(y)`` for ``x = 1`` and otherwise
  ``atan(|y / x|)`` placed in the quadrant of ``(x, y)`` with ``pi`` in two parts.

fdlibm documents each within one ulp; ``tests/conformance/libm64_accuracy.py`` measures the CrossGL text below.
"""

from __future__ import annotations

from .bits import ABS, HIGH, LOW, PREFIX, WITH_LOW, render
from .roots import SQRT

__all__ = ["ACOS", "ASIN", "ATAN", "ATAN2", "sources"]

ASIN = PREFIX + "asin"
ACOS = PREFIX + "acos"
ATAN = PREFIX + "atan"
ATAN2 = PREFIX + "atan2"
# The rational approximation asin(x) = x + x R(x^2) on [0, 0.5] that asin and acos share.
_ASIN_R = PREFIX + "asin_r"

_PI = {
    "PI": (0x400921FB, 0x54442D18),
    "PIO2_HI": (0x3FF921FB, 0x54442D18),
    "PIO2_LO": (0x3C91A626, 0x33145C07),
    "PIO4_HI": (0x3FE921FB, 0x54442D18),
    "PI_LO": (0x3CA1A626, 0x33145C07),
    "NAN": (0x7FF80000, 0),
}

_R_CONSTANTS = {
    "PS0": (0x3FC55555, 0x55555555),
    "PS1": (0xBFD4D612, 0x03EB6F7D),
    "PS2": (0x3FC9C155, 0x0E884455),
    "PS3": (0xBFA48228, 0xB5688F3B),
    "PS4": (0x3F49EFE0, 0x7501B288),
    "PS5": (0x3F023DE1, 0x0DFDF709),
    "QS1": (0xC0033A27, 0x1C8A2D4B),
    "QS2": (0x40002AE5, 0x9C598AC8),
    "QS3": (0xBFE6066C, 0x1B8D0159),
    "QS4": (0x3FB3B8C5, 0xB12E9282),
}

_R = f"""@source_license(fdlibm)
@precise
double {_ASIN_R}(double t) {{
    double p = t * ($PS0 + t * ($PS1 + t * ($PS2 + t * ($PS3 + t * ($PS4 + t * $PS5)))));
    double q = 1.0 + t * ($QS1 + t * ($QS2 + t * ($QS3 + t * $QS4)));
    return p / q;
}}"""

_ASIN = f"""@source_license(fdlibm)
@precise
double {ASIN}(double x) {{
    uint hx = {HIGH}(x);
    uint lx = {LOW}(x);
    uint ax = hx & 0x7FFFFFFFu;
    if (ax >= 0x3FF00000u) {{
        if (ax == 0x3FF00000u && lx == 0u) {{
            return x * $PIO2_HI + x * $PIO2_LO;
        }}
        if (ax > 0x7FF00000u || (ax == 0x7FF00000u && lx != 0u)) {{
            return x;
        }}
        return $NAN;
    }}
    if (ax < 0x3FE00000u) {{
        if (ax < 0x3E400000u) {{
            return x;
        }}
        return x + x * {_ASIN_R}(x * x);
    }}
    double t = (1.0 - {ABS}(x)) * 0.5;
    double s = {SQRT}(t);
    double r = {_ASIN_R}(t);
    double result = 0.0;
    if (ax >= 0x3FEF3333u) {{
        result = $PIO2_HI - (2.0 * (s + s * r) - $PIO2_LO);
    }} else {{
        double w = {WITH_LOW}(s, 0u);
        double c = (t - w * w) / (s + w);
        double p = 2.0 * s * r - ($PIO2_LO - 2.0 * c);
        double q = $PIO4_HI - 2.0 * w;
        result = $PIO4_HI - (p - q);
    }}
    return (hx >> 31u) == 0u ? result : -result;
}}"""

_ACOS = f"""@source_license(fdlibm)
@precise
double {ACOS}(double x) {{
    uint hx = {HIGH}(x);
    uint lx = {LOW}(x);
    uint ax = hx & 0x7FFFFFFFu;
    if (ax >= 0x3FF00000u) {{
        if (ax == 0x3FF00000u && lx == 0u) {{
            if ((hx >> 31u) == 0u) {{
                return 0.0;
            }}
            return $PI + 2.0 * $PIO2_LO;
        }}
        if (ax > 0x7FF00000u || (ax == 0x7FF00000u && lx != 0u)) {{
            return x;
        }}
        return $NAN;
    }}
    if (ax < 0x3FE00000u) {{
        if (ax <= 0x3C600000u) {{
            return $PIO2_HI + $PIO2_LO;
        }}
        return $PIO2_HI - (x - ($PIO2_LO - x * {_ASIN_R}(x * x)));
    }}
    if ((hx >> 31u) != 0u) {{
        double z = (1.0 + x) * 0.5;
        double s = {SQRT}(z);
        double w = {_ASIN_R}(z) * s - $PIO2_LO;
        return $PI - 2.0 * (s + w);
    }}
    double z = (1.0 - x) * 0.5;
    double s = {SQRT}(z);
    double df = {WITH_LOW}(s, 0u);
    double c = (z - df * df) / (s + df);
    double w = {_ASIN_R}(z) * s + c;
    return 2.0 * (df + w);
}}"""

_ATAN_CONSTANTS = {
    "ATANHI0": (0x3FDDAC67, 0x0561BB4F),
    "ATANHI1": (0x3FE921FB, 0x54442D18),
    "ATANHI2": (0x3FEF730B, 0xD281F69B),
    "ATANHI3": (0x3FF921FB, 0x54442D18),
    "ATANLO0": (0x3C7A2B7F, 0x222F65E2),
    "ATANLO1": (0x3C81A626, 0x33145C07),
    "ATANLO2": (0x3C700788, 0x7AF0CBBD),
    "ATANLO3": (0x3C91A626, 0x33145C07),
    "AT0": (0x3FD55555, 0x5555550D),
    "AT1": (0xBFC99999, 0x9998EBC4),
    "AT2": (0x3FC24924, 0x920083FF),
    "AT3": (0xBFBC71C6, 0xFE231671),
    "AT4": (0x3FB745CD, 0xC54C206E),
    "AT5": (0xBFB3B0F2, 0xAF749A6D),
    "AT6": (0x3FB10D66, 0xA0D03D51),
    "AT7": (0xBFADDE2D, 0x52DEFD9A),
    "AT8": (0x3FA97B4B, 0x24760DEB),
    "AT9": (0xBFA2B444, 0x2C6A6C2F),
    "AT10": (0x3F90AD3A, 0xE322DA11),
}

_ATAN = f"""@source_license(fdlibm)
@precise
double {ATAN}(double x) {{
    uint hx = {HIGH}(x);
    uint ax = hx & 0x7FFFFFFFu;
    if (ax >= 0x44100000u) {{
        if (ax > 0x7FF00000u || (ax == 0x7FF00000u && {LOW}(x) != 0u)) {{
            return x;
        }}
        return (hx >> 31u) == 0u ? $ATANHI3 + $ATANLO3 : -$ATANHI3 - $ATANLO3;
    }}
    double t = x;
    int id = -1;
    if (ax < 0x3FDC0000u) {{
        if (ax < 0x3E400000u) {{
            return x;
        }}
    }} else {{
        t = {ABS}(x);
        if (ax < 0x3FF30000u) {{
            if (ax < 0x3FE60000u) {{
                id = 0;
                t = (2.0 * t - 1.0) / (2.0 + t);
            }} else {{
                id = 1;
                t = (t - 1.0) / (t + 1.0);
            }}
        }} else if (ax < 0x40038000u) {{
            id = 2;
            t = (t - 1.5) / (1.0 + 1.5 * t);
        }} else {{
            id = 3;
            t = -1.0 / t;
        }}
    }}
    double z = t * t;
    double w = z * z;
    double s1 = z * ($AT0 + w * ($AT2 + w * ($AT4 + w * ($AT6 + w * ($AT8 + w * $AT10)))));
    double s2 = w * ($AT1 + w * ($AT3 + w * ($AT5 + w * ($AT7 + w * $AT9))));
    if (id < 0) {{
        return t - t * (s1 + s2);
    }}
    double hi = id == 0 ? $ATANHI0 : (id == 1 ? $ATANHI1 : (id == 2 ? $ATANHI2 : $ATANHI3));
    double lo = id == 0 ? $ATANLO0 : (id == 1 ? $ATANLO1 : (id == 2 ? $ATANLO2 : $ATANLO3));
    double result = hi - ((t * (s1 + s2) - lo) - t);
    return (hx >> 31u) == 0u ? result : -result;
}}"""

# atan2's quotient test: fdlibm's k = (iy - ix) >> 20 above 60 or below -60, as bounds on iy - ix.
_ATAN2 = f"""@source_license(fdlibm)
@precise
double {ATAN2}(double y, double x) {{
    uint hx = {HIGH}(x);
    uint lx = {LOW}(x);
    uint ax = hx & 0x7FFFFFFFu;
    uint hy = {HIGH}(y);
    uint ly = {LOW}(y);
    uint ay = hy & 0x7FFFFFFFu;
    if (ax > 0x7FF00000u || (ax == 0x7FF00000u && lx != 0u)) {{
        return x + y;
    }}
    if (ay > 0x7FF00000u || (ay == 0x7FF00000u && ly != 0u)) {{
        return x + y;
    }}
    if (hx == 0x3FF00000u && lx == 0u) {{
        return {ATAN}(y);
    }}
    uint m = ((hy >> 31u) & 1u) | ((hx >> 30u) & 2u);
    if ((ay | ly) == 0u) {{
        if (m < 2u) {{
            return y;
        }}
        return m == 2u ? $PI : -$PI;
    }}
    if ((ax | lx) == 0u) {{
        return (hy >> 31u) == 0u ? $PIO2_HI : -$PIO2_HI;
    }}
    if (ax == 0x7FF00000u) {{
        if (ay == 0x7FF00000u) {{
            if (m == 0u) {{
                return $PIO4_HI;
            }}
            if (m == 1u) {{
                return -$PIO4_HI;
            }}
            if (m == 2u) {{
                return 3.0 * $PIO4_HI;
            }}
            return -3.0 * $PIO4_HI;
        }}
        if (m == 0u) {{
            return 0.0;
        }}
        if (m == 1u) {{
            return -0.0;
        }}
        return m == 2u ? $PI : -$PI;
    }}
    if (ay == 0x7FF00000u) {{
        return (hy >> 31u) == 0u ? $PIO2_HI : -$PIO2_HI;
    }}
    int d = int(ay) - int(ax);
    double z = 0.0;
    if (d >= 63963136) {{
        z = $PIO2_HI + 0.5 * $PI_LO;
    }} else if ((hx >> 31u) != 0u && d < -62914560) {{
        z = 0.0;
    }} else {{
        z = {ATAN}({ABS}(y / x));
    }}
    if (m == 0u) {{
        return z;
    }}
    if (m == 1u) {{
        return -z;
    }}
    if (m == 2u) {{
        return $PI - (z - $PI_LO);
    }}
    return (z - $PI_LO) - $PI;
}}"""


def sources() -> list[tuple[str, str]]:
    """``(name, CrossGL definition)`` of ``asin``, ``acos``, ``atan`` and ``atan2``, each after what it calls."""
    return [
        (_ASIN_R, render(_R, _R_CONSTANTS)),
        (ASIN, render(_ASIN, _PI)),
        (ACOS, render(_ACOS, _PI)),
        (ATAN, render(_ATAN, _ATAN_CONSTANTS)),
        (ATAN2, render(_ATAN2, _PI)),
    ]
