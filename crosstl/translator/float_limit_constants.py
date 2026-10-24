"""Binary32 limit and non-finite constants supplied by source environments.

Metal (MSL specification, section 6.6.1 "Floating-point math constants" and the
``<metal_limits>`` macros) and C provide ``INFINITY``, ``HUGE_VALF``, ``NAN``,
``MAXFLOAT``, ``FLT_MAX``, ``FLT_MIN`` and ``FLT_EPSILON`` as predefined names.
CrossGL keeps them as identifiers. HLSL, GLSL and SPIR-V define none of them, so
each generator renders them from the exact IEEE 754 binary32 encoding recorded
here instead of emitting the source spelling.

Finite values keep the exact decimal spelling of the C ``__FLT_*__`` macros so
the rendered literal is a constant expression in every target. Non-finite values
are rendered as bit casts, because neither HLSL nor GLSL has an infinity or NaN
literal.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class FloatLimitConstant:
    """A binary32 constant identified by its source-environment name."""

    name: str
    bits: int
    decimal: str | None

    @property
    def is_finite(self) -> bool:
        return self.decimal is not None

    @property
    def value(self) -> float:
        """Return the constant as a Python float with the exact binary32 value."""
        if self.name in _VALUE_CACHE:
            return _VALUE_CACHE[self.name]
        return struct.unpack("<f", struct.pack("<I", self.bits))[0]


_POSITIVE_INFINITY_BITS = 0x7F800000
_QUIET_NAN_BITS = 0x7FC00000
_FLT_MAX_BITS = 0x7F7FFFFF
_FLT_MAX_DECIMAL = "3.40282346638528859811704183484516925e+38"

_FLOAT_LIMITS = (
    ("INFINITY", _POSITIVE_INFINITY_BITS, None),
    ("HUGE_VALF", _POSITIVE_INFINITY_BITS, None),
    ("NAN", _QUIET_NAN_BITS, None),
    ("MAXFLOAT", _FLT_MAX_BITS, _FLT_MAX_DECIMAL),
    ("FLT_MAX", _FLT_MAX_BITS, _FLT_MAX_DECIMAL),
    ("FLT_MIN", 0x00800000, "1.17549435082228750796873653722224568e-38"),
    ("FLT_EPSILON", 0x34000000, "1.19209289550781250000000000000000000e-7"),
)

FLOAT_LIMIT_CONSTANTS: Mapping[str, FloatLimitConstant] = MappingProxyType(
    {
        name: FloatLimitConstant(name=name, bits=bits, decimal=decimal)
        for name, bits, decimal in _FLOAT_LIMITS
    }
)

# One float object per name, so NaN constants compare identical when they are
# used as cache keys by a generator.
_VALUE_CACHE: dict[str, float] = {}
for _constant in FLOAT_LIMIT_CONSTANTS.values():
    _VALUE_CACHE[_constant.name] = struct.unpack(
        "<f", struct.pack("<I", _constant.bits)
    )[0]
del _constant

_BITCAST_FUNCTIONS = MappingProxyType(
    {
        "hlsl": "asfloat",
        "directx": "asfloat",
        "glsl": "uintBitsToFloat",
        "opengl": "uintBitsToFloat",
    }
)


def float_limit_constant(name: str) -> FloatLimitConstant | None:
    """Return the binary32 limit constant named *name*, if known."""
    if not isinstance(name, str):
        return None
    return FLOAT_LIMIT_CONSTANTS.get(name)


def render_float_limit_constant(name: str, target: str) -> str | None:
    """Render a known constant as an HLSL or GLSL float expression."""
    constant = float_limit_constant(name)
    if constant is None or not isinstance(target, str):
        return None
    bitcast = _BITCAST_FUNCTIONS.get(target.strip().lower())
    if bitcast is None:
        return None
    if constant.is_finite:
        return f"{constant.decimal}f"
    return f"{bitcast}(0x{constant.bits:08x}u)"


__all__ = (
    "FLOAT_LIMIT_CONSTANTS",
    "FloatLimitConstant",
    "float_limit_constant",
    "render_float_limit_constant",
)
