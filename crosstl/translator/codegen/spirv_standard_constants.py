"""SPIR-V lowering of source-environment standard constants.

CrossGL keeps C and Metal predefined constants (``M_PI_F``, ``INFINITY``,
``FLT_MAX``, ...) as identifiers. SPIR-V has no names for them, so they lower to
``OpConstant`` instructions carrying the exact value recorded in
:mod:`crosstl.translator.standard_constants` and
:mod:`crosstl.translator.float_limit_constants`.
"""

from __future__ import annotations

import math

from ..float_limit_constants import float_limit_constant
from ..standard_constants import standard_math_constant

# SPIR-V assembly writes infinities and NaNs as hexadecimal floats whose
# exponent overflows the type (SPIRV-Tools hex-float syntax), keyed by the
# exponent bias limit of each floating-point width.
_NON_FINITE_EXPONENTS = {"float": 128, "double": 1024}


def spirv_standard_float_constant(generator, name):
    """Return the SPIR-V constant for a standard constant *name*, or ``None``."""
    limit = float_limit_constant(name)
    if limit is not None:
        float_type = generator.register_primitive_type("float")
        return generator.register_constant(limit.value, float_type)

    constant = standard_math_constant(name)
    if constant is None:
        return None
    scalar_type = generator.register_primitive_type(constant.scalar_type)
    return generator.register_constant(float(constant.decimal), scalar_type)


def spirv_non_finite_float_literal(value, type_name):
    """Return the SPIR-V assembly text for a non-finite float, or ``None``."""
    if not isinstance(value, float) or math.isfinite(value):
        return None
    exponent = _NON_FINITE_EXPONENTS.get(type_name)
    if exponent is None:
        return None
    if math.isnan(value):
        return f"0x1.8p+{exponent}"
    sign = "-" if value < 0 else ""
    return f"{sign}0x1p+{exponent}"


__all__ = ("spirv_non_finite_float_literal", "spirv_standard_float_constant")
