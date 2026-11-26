"""A correctly rounded binary64 square root in binary64 arithmetic, for the helpers that need one.

``asin``, ``acos``, ``asinh`` and ``acosh`` take square roots of binary64 values. GLSL and SPIR-V declare ``sqrt``
for ``double``, but give double-precision operations no precision beyond that of single precision (GLSL 4.60
section 4.7.1), and DXIL has no binary64 square root, so the helpers compute their own from the operations the
package already relies on (correctly rounded addition, subtraction, multiplication and division, :mod:`.bits`).

``x = m 2^(2k)`` with ``1 <= m < 4`` (a subnormal ``x`` is scaled by 2^54 first). Four Newton steps
``y <- (y + m / y) / 2`` from a linear seed with a relative error below 7 % leave ``y`` within 0.75 ulp of
``sqrt(m)``: the iteration's own error is below 2^-80, and the last step rounds twice. The residual ``m - y^2`` is
then computed with Dekker's product (Veltkamp's split, no fused multiply-add) and Sterbenz's lemma, exactly wherever
it decides the result, and compared with ``y 2^-52``, which tells whether ``sqrt(m)`` lies within half an ulp of
``y``, above it or below it (Tuckerman's test). The result is ``y``, or ``y`` moved by one ulp, scaled by ``2^k``
exactly.
"""

from __future__ import annotations

from .bits import HIGH, LOW, PREFIX, SCALE, WITH_HIGH, render

__all__ = ["SQRT", "sources"]

SQRT = PREFIX + "sqrt"

_CONSTANTS = {
    # 2^27 + 1, Veltkamp's splitting factor for 53-bit significands.
    "SPLIT": (0x41A00000, 0x02000000),
    "TWO54": (0x43500000, 0),
    "TWO_M52": (0x3CB00000, 0),
    "NAN": (0x7FF80000, 0),
}

_SQRT = f"""@precise
double {SQRT}(double x) {{
    uint hx = {HIGH}(x);
    uint lx = {LOW}(x);
    uint ax = hx & 0x7FFFFFFFu;
    if ((ax | lx) == 0u) {{
        return x;
    }}
    if (ax > 0x7FF00000u || (ax == 0x7FF00000u && lx != 0u)) {{
        return x;
    }}
    if ((hx >> 31u) != 0u) {{
        return $NAN;
    }}
    if (ax == 0x7FF00000u) {{
        return x;
    }}
    double v = x;
    int adjust = 0;
    if (ax < 0x00100000u) {{
        v = x * $TWO54;
        adjust = 27;
        ax = {HIGH}(v);
    }}
    uint biased = ax >> 20u;
    uint odd = (biased & 1u) ^ 1u;
    double m = {WITH_HIGH}(v, (ax & 0x000FFFFFu) | ((1023u + odd) << 20u));
    int k = int((biased + 1u - odd) >> 1u) - 512 - adjust;
    double y = 0.75 + 0.3125 * m;
    y = 0.5 * (y + m / y);
    y = 0.5 * (y + m / y);
    y = 0.5 * (y + m / y);
    y = 0.5 * (y + m / y);
    if (y < 1.0) {{
        y = 1.0;
    }}
    if (y > 2.0) {{
        y = 2.0;
    }}
    double split = $SPLIT * y;
    double yh = split - (split - y);
    double yl = y - yh;
    double p = y * y;
    double e = ((yh * yh - p) + 2.0 * yh * yl) + yl * yl;
    double r = (m - p) - e;
    double bound = y * $TWO_M52;
    if (r > bound) {{
        y = y + $TWO_M52;
    }} else if (r <= -bound) {{
        y = y - $TWO_M52;
    }}
    return {SCALE}(y, k);
}}"""


def sources() -> list[tuple[str, str]]:
    """``(name, CrossGL definition)`` of the square root."""
    return [(SQRT, render(_SQRT, _CONSTANTS))]
