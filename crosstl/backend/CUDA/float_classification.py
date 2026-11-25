"""CrossGL forms of the CUDA and HIP classification intrinsics ``__isinff``, ``__isnanf``, ``__finitef``, ``__signbitf``.

CUDA declares these device built-ins in ``crt/math_functions.h`` with an ``int`` result, and the ``bool`` functions of
``<cmath>`` (``isinf``, ``isnan``, ``isfinite``) call them (``crt/math_functions.hpp``). HIP declares the same functions
over OCML (``hip/amd_detail/amd_math_functions.h``). Each returns 1 when the test holds and 0 otherwise: nvcc writes
``selp.u32 %r, 1, 0`` after the comparison, and the sign-bit tests shift the sign bit down (``shr.u32 %r, %r, 31``).

CrossGL has the ``bool`` tests, so each intrinsic becomes the ``i32`` of its ``bool`` test. The operand passes
through the intrinsic's parameter type first (``f32`` for the ``f`` forms, ``f64`` for the others), as a call
argument does (C++ [expr.call]/7): ``__isinff`` of a double too large for a float is 1. The sign bit is the top bit
of the operand's representation, which ``floatBitsToUint`` and ``as_type<u64>`` read, so ``__signbitf(-0.0f)`` and
the sign of a NaN are kept.
"""

__all__ = ["FLOAT_CLASSIFICATION_INTRINSICS", "format_float_classification_intrinsic"]

# name -> (parameter type, CrossGL text with ``{x}`` for the converted operand)
FLOAT_CLASSIFICATION_INTRINSICS = {
    "__isinff": ("f32", "i32(isinf({x}))"),
    "__isnanf": ("f32", "i32(isnan({x}))"),
    "__finitef": ("f32", "i32(isfinite({x}))"),
    "__signbitf": ("f32", "i32(floatBitsToUint({x}) >> 31u)"),
    "__isinf": ("f64", "i32(isinf({x}))"),
    "__isnan": ("f64", "i32(isnan({x}))"),
    "__finite": ("f64", "i32(isfinite({x}))"),
    "__signbit": ("f64", "i32(as_type<u64>({x}) >> 63u)"),
}


def format_float_classification_intrinsic(name, args):
    """The CrossGL text of a classification intrinsic call, or ``None`` when ``name`` is not one with one argument."""
    if not isinstance(name, str):
        return None
    bare = name[2:] if name.startswith("::") else name
    form = FLOAT_CLASSIFICATION_INTRINSICS.get(bare)
    if form is None or len(args) != 1:
        return None
    parameter_type, template = form
    return template.format(x=f"{parameter_type}({args[0]})")
