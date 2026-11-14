"""Scalar types of the predefined constants generators render as literals.

``M_SQRT1_2_F`` is a ``float`` and ``M_SQRT1_2`` a ``double`` (C, OpenCL and MSL section 6.6); ``FLT_MAX``,
``INFINITY`` and the other binary32 limits are ``float``. A generator that types ``x * M_SQRT1_2_F`` for a
``float16`` ``x`` needs the constant's own type: the usual arithmetic conversions then make the product a
``float``, not a ``float16``.
"""

from __future__ import annotations

from ..float_limit_constants import float_limit_constant
from ..standard_constants import standard_math_constant

__all__ = ["standard_constant_type"]


def standard_constant_type(name: object) -> str | None:
    """The scalar type of the predefined constant ``name``, or ``None`` when ``name`` names none."""
    if not isinstance(name, str):
        return None
    constant = standard_math_constant(name)
    if constant is not None:
        return constant.scalar_type
    if float_limit_constant(name) is not None:
        return "float"
    return None
