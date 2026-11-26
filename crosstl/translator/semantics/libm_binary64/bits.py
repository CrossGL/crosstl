"""Bit access and exact constants shared by the binary64 math helpers.

Every helper reads and writes the two 32-bit words of a binary64 value with ``unpackDouble2x32`` and
``packDouble2x32``, which GLSL 4.00, GLSL.std.450 (``UnpackDouble2x32``, ``PackDouble2x32``) and HLSL (``asuint``,
``asdouble``) define exactly. Constants that binary32 cannot represent are written as their two words too: an
unsuffixed CrossGL literal reaches GLSL and SPIR-V as a binary32 constant, which would round a binary64 coefficient.
Literals in the helpers are limited to values binary32 represents exactly (``0.25``, ``0.5``, ``1.0``, ``2.0``, ...).
"""

from __future__ import annotations

import re
import struct

__all__ = [
    "ABS",
    "BITS",
    "HIGH",
    "LOW",
    "PREFIX",
    "SCALE",
    "WITH_HIGH",
    "WITH_LOW",
    "constant",
    "render",
    "support_sources",
    "value_of",
]

# Reserved names (the ``cgl_rule_`` prefix): a program that defines one is refused (lowering.py).
PREFIX = "cgl_rule_libm64_"
BITS = PREFIX + "bits"
HIGH = PREFIX + "high"
LOW = PREFIX + "low"
WITH_HIGH = PREFIX + "with_high"
WITH_LOW = PREFIX + "with_low"
ABS = PREFIX + "abs"
SCALE = PREFIX + "scale"


def constant(high: int, low: int) -> str:
    """The CrossGL expression of the binary64 value with words ``high`` and ``low``."""
    return f"{BITS}(0x{high:08X}u, 0x{low:08X}u)"


def value_of(high: int, low: int) -> float:
    """The binary64 value with words ``high`` and ``low``."""
    return struct.unpack("<d", struct.pack("<II", low, high))[0]


_PLACEHOLDER = re.compile(r"\$([A-Z][A-Z0-9_]*)")


def render(text: str, constants: dict[str, tuple[int, int]]) -> str:
    """``text`` with each ``$NAME`` replaced by the exact expression of ``constants[NAME]``."""
    return _PLACEHOLDER.sub(lambda match: constant(*constants[match.group(1)]), text)


# 2^1023 and 2^-1000, the factors that keep a scaled result's rounding single (see SCALE).
_SCALE_CONSTANTS = {"TWO_1023": (0x7FE00000, 0), "TWO_M1000": (0x01700000, 0)}

_SUPPORT = (
    (
        BITS,
        f"""double {BITS}(uint high, uint low) {{
    return packDouble2x32(uvec2(low, high));
}}""",
    ),
    (
        HIGH,
        f"""uint {HIGH}(double x) {{
    return unpackDouble2x32(x).y;
}}""",
    ),
    (
        LOW,
        f"""uint {LOW}(double x) {{
    return unpackDouble2x32(x).x;
}}""",
    ),
    (
        WITH_HIGH,
        f"""double {WITH_HIGH}(double x, uint high) {{
    return packDouble2x32(uvec2(unpackDouble2x32(x).x, high));
}}""",
    ),
    (
        WITH_LOW,
        f"""double {WITH_LOW}(double x, uint low) {{
    return packDouble2x32(uvec2(low, unpackDouble2x32(x).y));
}}""",
    ),
    # |x| with the sign bit cleared, exact for every value (fdlibm's fabs).
    (
        ABS,
        f"""double {ABS}(double x) {{
    uvec2 words = unpackDouble2x32(x);
    return packDouble2x32(uvec2(words.x, words.y & 0x7FFFFFFFu));
}}""",
    ),
    # y * 2^k with one rounding, for k <= 1024 and a y * 2^k at least 2^-1075: the scalbn that exp and pow end
    # with. A power of two is exact while the product stays normal, so a result that becomes subnormal is scaled in
    # two steps, of which only the last rounds.
    (
        SCALE,
        f"""@precise
double {SCALE}(double y, int k) {{
    if (k >= -1021) {{
        if (k == 1024) {{
            return y * 2.0 * $TWO_1023;
        }}
        return y * {BITS}(uint(k + 1023) << 20u, 0u);
    }}
    return y * {BITS}(uint(k + 2023) << 20u, 0u) * $TWO_M1000;
}}""",
    ),
)


def support_sources() -> list[tuple[str, str]]:
    """``(name, CrossGL definition)`` of each shared support function, in dependency order."""
    return [(name, render(text, _SCALE_CONSTANTS)) for name, text in _SUPPORT]
