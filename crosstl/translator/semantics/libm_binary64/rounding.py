"""Binary64 rounding to an integral value with halfway cases away from zero (C11 7.12.9.6 ``round``), in word
arithmetic.

CUDA, HIP and OpenCL C programs write ``round`` on ``double`` values, which the importers spell ``__cgl_round_away``
(the ``round_away`` rule family, :mod:`..rule_table`). The family's binary32 helper rounds with ``trunc``, ``abs``
and ``sign``, but GLSL.std.450 ``Trunc`` takes 16- and 32-bit operands only and DXIL has no binary64 ``Round_z``, so
the binary64 helper works on the two words of the value instead:

* ``|x| < 1``: ``+-1`` from 0.5 on and ``+-0`` below it, with the sign of ``x``;
* ``|x| >= 2^52``, infinities and NaNs: ``x`` itself;
* otherwise the bit of weight 0.5 is added to the significand, the carry propagating into the exponent, and the
  fraction bits are cleared, unless they are all zero already.

The result is exact; ``tests/conformance/libm64_accuracy.py`` compares it with a decimal reference.
"""

from __future__ import annotations

from .bits import BITS, HIGH, LOW, PREFIX

__all__ = ["ROUND_AWAY", "sources"]

ROUND_AWAY = PREFIX + "round_away"

_ROUND_AWAY = f"""double {ROUND_AWAY}(double x) {{
    uint hx = {HIGH}(x);
    uint lx = {LOW}(x);
    uint sign = hx & 0x80000000u;
    int e = int((hx & 0x7FFFFFFFu) >> 20u) - 1023;
    if (e < 0) {{
        if (e == -1) {{
            return {BITS}(sign | 0x3FF00000u, 0u);
        }}
        return {BITS}(sign, 0u);
    }}
    if (e >= 52) {{
        return x;
    }}
    if (e < 20) {{
        uint high_fraction = 0x000FFFFFu >> uint(e);
        if (((hx & high_fraction) | lx) == 0u) {{
            return x;
        }}
        return {BITS}((hx + (0x00080000u >> uint(e))) & ~high_fraction, 0u);
    }}
    uint low_fraction = 0xFFFFFFFFu >> uint(e - 20);
    if ((lx & low_fraction) == 0u) {{
        return x;
    }}
    uint low = lx + (1u << uint(51 - e));
    if (low < lx) {{
        hx += 1u;
    }}
    return {BITS}(hx, low & ~low_fraction);
}}"""


def sources() -> list[tuple[str, str]]:
    """``(name, CrossGL definition)`` of the rounding helper."""
    return [(ROUND_AWAY, _ROUND_AWAY)]
