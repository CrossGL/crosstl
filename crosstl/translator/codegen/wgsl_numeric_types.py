"""Numeric types that WGSL output cannot represent.

WGSL's scalar types are ``bool``, ``i32``, ``u32``, ``f32`` and, behind ``enable f16``, ``f16`` (WGSL section 6.2,
plain types). It has no 64-bit floating-point type, no 64-bit integer type and no bfloat16 type. A CrossGL value of one
of those types therefore has no WGSL spelling: writing it as ``f32`` or ``i32`` changes its range and precision, and
writing it verbatim produces a module that no WGSL implementation accepts. The WGSL emitter refuses such types with
:class:`WGSLNumericTypeUnsupportedError` instead.

8- and 16-bit integers are not refused here. :mod:`..semantics.narrow` legalizes them to ``i32`` with explicit
wrapping before WGSL emission, and :mod:`..semantics.half` computes ``half`` in binary32 with binary16 rounding.
"""

from __future__ import annotations

import re

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "WGSLNumericTypeUnsupportedError",
    "refuse_unsupported_wgsl_numeric_type",
    "unsupported_wgsl_numeric_family",
]

FLOAT64 = "64-bit floating-point"
INT64 = "64-bit integer"
BFLOAT16 = "bfloat16"

_SCALARS = {
    FLOAT64: ("double", "f64", "float64", "float64_t"),
    INT64: (
        "long",
        "ulong",
        "signed long",
        "unsigned long",
        "long long",
        "unsigned long long",
        "i64",
        "u64",
        "int64",
        "int64_t",
        "uint64",
        "uint64_t",
        "size_t",
        "ptrdiff_t",
    ),
    BFLOAT16: ("bfloat16", "bfloat16_t", "bf16"),
}
_SCALAR_FAMILIES = {
    spelling: family for family, spellings in _SCALARS.items() for spelling in spellings
}
# GLSL-style vector prefixes (``dvec2``, ``i64vec3``) besides the HLSL/MSL-style ``<scalar>N`` spellings.
_VECTOR_PREFIXES = {
    "dvec": FLOAT64,
    "i64vec": INT64,
    "u64vec": INT64,
    "bf16vec": BFLOAT16,
}
_SIZED = re.compile(r"(?P<base>[a-z_][a-z0-9_]*?)(?P<rows>[234])(?:x(?P<cols>[234]))?")
_DMAT = re.compile(r"dmat[234](?:x[234])?")


class WGSLNumericTypeUnsupportedError(TranslationDiagnostic):
    """A CrossGL numeric type that has no WGSL scalar, vector or matrix spelling."""

    def __init__(self, spelling, family):
        self.spelling = spelling
        self.family = family
        self.missing_capabilities = (f"wgsl.{family.replace(' ', '-')}-type",)
        super().__init__(
            f"WGSL target cannot represent {family} type '{spelling}': WGSL scalars are bool, i32, u32, f32 "
            "and f16 (WGSL section 6.2), and narrowing the value would change its range and precision",
            target="wgsl",
            code=f"{CODE_PREFIX}wgsl-numeric-type-unsupported",
        )


def unsupported_wgsl_numeric_family(spelling):
    """The numeric family of ``spelling`` when WGSL cannot represent it, else ``None``.

    ``spelling`` is a CrossGL scalar (``double``, ``int64_t``), vector (``dvec2``, ``long3``, ``i64vec4``) or matrix
    (``dmat3``, ``double2x4``) type name. Generic spellings such as ``vec2<f64>`` are resolved by the caller, which
    checks their element type.
    """
    name = " ".join(str(spelling).split())
    family = _SCALAR_FAMILIES.get(name)
    if family is not None:
        return family
    if _DMAT.fullmatch(name):
        return FLOAT64
    sized = _SIZED.fullmatch(name)
    if sized is None:
        return None
    base = sized.group("base")
    if sized.group("cols") is not None:
        # HLSL-style matrices: ``double2x3``, ``int64_t4x4``.
        return _SCALAR_FAMILIES.get(base)
    return _VECTOR_PREFIXES.get(base) or _SCALAR_FAMILIES.get(base)


def refuse_unsupported_wgsl_numeric_type(spelling):
    """Raise :class:`WGSLNumericTypeUnsupportedError` when WGSL cannot represent ``spelling``."""
    family = unsupported_wgsl_numeric_family(spelling)
    if family is not None:
        raise WGSLNumericTypeUnsupportedError(" ".join(str(spelling).split()), family)
