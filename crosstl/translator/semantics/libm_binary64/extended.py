"""``exp`` in double-double arithmetic, for the hyperbolic functions that combine two exponentials.

CUDA bounds binary64 ``cosh`` and ``tanh`` at one ulp and ``sinh`` at two (CUDA C++ Programming Guide, the
double-precision table of the mathematical functions appendix). fdlibm's formulas, ``(e^x + e^-x) / 2`` and
``expm1(2x) / (expm1(2x) + 2)`` with every step rounded to binary64, reach almost two ulps, so these helpers compute
``e^x`` with an error near 2^-60 instead and round once at the end:

* ``x = k ln2 + r`` with ``|r| <= ln2/2 + 2^-40``: ``k`` is ``x / ln2`` rounded, ``x - k ln2_hi`` is exact (``ln2_hi``
  has 32 significant bits and Sterbenz's lemma covers the subtraction) and ``r`` is that difference minus ``k ln2_lo``,
  kept as an unevaluated sum ``r_hi + r_lo`` (Dekker's fast two-sum);
* ``e^r = 1 + r + r^2 / 2 + r^3 P(r)`` with the Taylor coefficients ``1/3!`` to ``1/16!``, whose remainder is below
  2^-68 for ``|r| <= 0.35``. ``1 + r_hi`` and ``r_hi^2 / 2`` are carried exactly (two-sum, and Dekker's product with
  Veltkamp's split, since no fused multiply-add is assumed); only ``r^3 P(r)``, ``r_lo`` terms and the last two
  additions round, about 2^-58 in total;
* the result is ``dvec3(hi, lo, k)`` with ``e^x = 2^k (hi + lo)`` and ``hi + lo`` in ``[0.7, 1.42]``, so a caller can
  scale ``hi + lo`` by ``2^(k - 1)`` near the overflow threshold, where ``e^x`` itself overflows.

The callers pass ``0.25 <= x <= 710.48``; :mod:`.hyperbolic` handles smaller arguments with Taylor series.
"""

from __future__ import annotations

from .bits import PREFIX, render

__all__ = ["EXP_DD", "SPLIT_PRODUCT_ERROR", "sources"]

EXP_DD = PREFIX + "exp_dd"
# The rounding error of a * b: a * b - fl(a * b), exact (Dekker's product with Veltkamp's split).
SPLIT_PRODUCT_ERROR = PREFIX + "product_error"

_CONSTANTS = {
    # 2^27 + 1, Veltkamp's splitting factor for 53-bit significands.
    "SPLIT": (0x41A00000, 0x02000000),
    "LN2_HI": (0x3FE62E42, 0xFEE00000),
    "LN2_LO": (0x3DEA39EF, 0x35793C76),
    "INV_LN2": (0x3FF71547, 0x652B82FE),
    # 1/j! for j = 3 .. 16, each rounded to binary64.
    "C3": (0x3FC55555, 0x55555555),
    "C4": (0x3FA55555, 0x55555555),
    "C5": (0x3F811111, 0x11111111),
    "C6": (0x3F56C16C, 0x16C16C17),
    "C7": (0x3F2A01A0, 0x1A01A01A),
    "C8": (0x3EFA01A0, 0x1A01A01A),
    "C9": (0x3EC71DE3, 0xA556C734),
    "C10": (0x3E927E4F, 0xB7789F5C),
    "C11": (0x3E5AE645, 0x67F544E4),
    "C12": (0x3E21EED8, 0xEFF8D898),
    "C13": (0x3DE61246, 0x13A86D09),
    "C14": (0x3DA93974, 0xA8C07C9D),
    "C15": (0x3D6AE7F3, 0xE733B81F),
    "C16": (0x3D2AE7F3, 0xE733B81F),
}

_PRODUCT_ERROR = f"""@precise
double {SPLIT_PRODUCT_ERROR}(double a, double b, double p) {{
    double sa = $SPLIT * a;
    double ah = sa - (sa - a);
    double al = a - ah;
    double sb = $SPLIT * b;
    double bh = sb - (sb - b);
    double bl = b - bh;
    return ((ah * bh - p) + ah * bl + al * bh) + al * bl;
}}"""

_EXP_DD = f"""@source_license(fdlibm)
@precise
dvec3 {EXP_DD}(double x) {{
    int k = int($INV_LN2 * x + 0.5);
    double kd = double(k);
    double hi = x - kd * $LN2_HI;
    double lo = kd * $LN2_LO;
    double r = hi - lo;
    double rl = (hi - r) - lo;
    double tail = r * r * r * ($C3 + r * ($C4 + r * ($C5 + r * ($C6 + r * ($C7 + r * ($C8 + r * ($C9 + r * ($C10
        + r * ($C11 + r * ($C12 + r * ($C13 + r * ($C14 + r * ($C15 + r * $C16)))))))))))));
    double square = r * r;
    double square_error = {SPLIT_PRODUCT_ERROR}(r, r, square);
    double half_square = 0.5 * square;
    double head = 1.0 + r;
    double head_error = (1.0 - head) + r;
    double sum = head + half_square;
    double sum_error = (head - sum) + half_square;
    double low = (sum_error + head_error) + ((0.5 * square_error + rl * sum) + tail);
    double result = sum + low;
    return dvec3(result, (sum - result) + low, kd);
}}"""


def sources() -> list[tuple[str, str]]:
    """``(name, CrossGL definition)`` of the double-double support, each after what it calls."""
    return [
        (SPLIT_PRODUCT_ERROR, render(_PRODUCT_ERROR, _CONSTANTS)),
        (EXP_DD, render(_EXP_DD, _CONSTANTS)),
    ]
