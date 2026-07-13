"""Standard source-environment mathematical constants.

The decimal spellings in this module are source data.  Keep them as strings so
constant lowering never depends on the host's floating-point or locale rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


@dataclass(frozen=True)
class StandardMathConstant:
    """A typed mathematical constant supplied by a source environment."""

    name: str
    scalar_type: Literal["float", "double"]
    decimal: str
    source_environments: frozenset[str]

    @property
    def decimal_spelling(self) -> str:
        """Return the verbatim decimal spelling retained by the registry."""
        return self.decimal


_DOUBLE_SOURCE_ENVIRONMENTS = frozenset({"c", "cpp", "opencl"})
_FLOAT_SOURCE_ENVIRONMENTS = frozenset({"metal", "opencl"})

_STANDARD_DECIMALS = (
    ("M_E", "2.71828182845904523536028747135266250"),
    ("M_LOG2E", "1.44269504088896340735992468100189214"),
    ("M_LOG10E", "0.434294481903251827651128918916605082"),
    ("M_LN2", "0.693147180559945309417232121458176568"),
    ("M_LN10", "2.30258509299404568401799145468436421"),
    ("M_PI", "3.14159265358979323846264338327950288"),
    ("M_PI_2", "1.57079632679489661923132169163975144"),
    ("M_PI_4", "0.785398163397448309615660845819875721"),
    ("M_1_PI", "0.318309886183790671537767526745028724"),
    ("M_2_PI", "0.636619772367581343075535053490057448"),
    ("M_2_SQRTPI", "1.12837916709551257389615890312154517"),
    ("M_SQRT2", "1.41421356237309504880168872420969808"),
    ("M_SQRT1_2", "0.707106781186547524400844362104849039"),
)


def _build_standard_math_constants() -> Mapping[str, StandardMathConstant]:
    constants = {}
    for name, decimal in _STANDARD_DECIMALS:
        constants[name] = StandardMathConstant(
            name=name,
            scalar_type="double",
            decimal=decimal,
            source_environments=_DOUBLE_SOURCE_ENVIRONMENTS,
        )
        float_name = f"{name}_F"
        constants[float_name] = StandardMathConstant(
            name=float_name,
            scalar_type="float",
            decimal=decimal,
            source_environments=_FLOAT_SOURCE_ENVIRONMENTS,
        )
    return MappingProxyType(constants)


STANDARD_MATH_CONSTANTS = _build_standard_math_constants()

_TARGET_ALIASES = MappingProxyType(
    {
        "directx": "hlsl",
        "hlsl": "hlsl",
        "opengl": "glsl",
        "glsl": "glsl",
    }
)
_TARGET_SUFFIXES = MappingProxyType(
    {
        "hlsl": MappingProxyType({"float": "f", "double": "L"}),
        "glsl": MappingProxyType({"float": "f", "double": "LF"}),
    }
)


def standard_math_constant(name: str) -> StandardMathConstant | None:
    """Return the standard mathematical constant named *name*, if known."""
    if not isinstance(name, str):
        return None
    return STANDARD_MATH_CONSTANTS.get(name)


def render_standard_math_constant(name: str, target: str) -> str | None:
    """Render a known constant as an explicitly typed target literal."""
    constant = standard_math_constant(name)
    if constant is None or not isinstance(target, str):
        return None

    normalized_target = _TARGET_ALIASES.get(target.strip().lower())
    if normalized_target is None:
        return None

    suffix = _TARGET_SUFFIXES[normalized_target][constant.scalar_type]
    return f"{constant.decimal}{suffix}"


__all__ = (
    "STANDARD_MATH_CONSTANTS",
    "StandardMathConstant",
    "render_standard_math_constant",
    "standard_math_constant",
)
