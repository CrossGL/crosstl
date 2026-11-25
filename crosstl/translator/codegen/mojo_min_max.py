"""CrossGL ``min``, ``max`` and ``clamp`` written with Mojo's ``std.math.min`` and ``std.math.max``.

Mojo's ``min`` and ``max`` take two SIMD values of one DType and width (a scalar is a one-lane SIMD), so they cover
every integer, ``Float16``, ``Float32`` and ``Float64`` scalar and vector once both operands have the result's type.
The generator therefore converts each operand to the result type instead of declaring module-level overloads, which
would hide the standard ones for every type they do not list.

The result type is the type of the first vector operand; with scalar operands only it is their common type under
the usual arithmetic conversions (C17 section 6.3.1.8, GLSL 4.60 section 4.1.10), a literal taking the type of the
other operands. A scalar next to a vector is splatted into the vector's lanes, keeping a ``vec3``'s padding lane at
zero. ``clamp(x, lo, hi)`` is ``min(max(x, lo), hi)``, its definition in GLSL 4.60 section 8.3.
"""

from functools import reduce

from .mojo_arithmetic_conversions import (
    common_scalar_dtype,
    mojo_dtype_type_name,
    mojo_scalar_kind,
)
from .mojo_tables import MOJO_DTYPE_INFO, MOJO_SCALAR_DTYPES

__all__ = ["generate_mojo_min_max", "mojo_min_max_result_type"]

_ARITIES = {"min": 2, "max": 2, "clamp": 3}
_FLOAT_DTYPES = frozenset({"DType.float16", "DType.float32", "DType.float64"})


def _is_literal(generator, expr):
    return generator.is_literal_expression(expr)


def mojo_min_max_result_type(generator, args):
    """The CrossGL type of ``min``/``max``/``clamp`` of ``args``, or ``None`` when it is not a numeric type."""
    arg_types = [generator.expression_result_type(arg) for arg in args]
    for arg_type in arg_types:
        if generator.vector_type_info(arg_type) is not None:
            return arg_type
    kinds = [mojo_scalar_kind(generator, arg_type) for arg_type in arg_types]
    if any(kind is None for kind in kinds):
        return None
    operands = [
        kind for arg, kind in zip(args, kinds) if not _is_literal(generator, arg)
    ]
    if not operands:
        operands = kinds
    common = reduce(common_scalar_dtype, operands)
    if common not in _FLOAT_DTYPES and any(kind in _FLOAT_DTYPES for kind in kinds):
        # A float literal next to integer operands converts them to float, as an arithmetic operator does.
        common = "DType.float32"
    return mojo_dtype_type_name(common)


def _vector_operand(generator, arg, info):
    """``arg`` as a value of the vector type ``info`` describes, or ``None`` when its shape cannot be one."""
    dtype, source_width, storage_width, pad_literal = info
    arg_info = generator.vector_type_info(generator.expression_result_type(arg))
    if arg_info is not None:
        if arg_info[1] != source_width:
            return None
        return generator.cast_vector_component(
            generator.generate_expression(arg), arg_info[0], dtype
        )
    scalar = generator.generate_constructor_scalar_expression(arg, dtype)
    if source_width == storage_width:
        return generator.scalar_splat_expression(dtype, storage_width, scalar)
    if generator.is_duplicate_sensitive_expression(arg):
        generator.required_splat_helpers.add(dtype)
        return f"{generator.vec3_splat_helper_name(dtype)}({scalar})"
    lanes = [scalar] * source_width + [pad_literal or MOJO_DTYPE_INFO[dtype][2]]
    return f"SIMD[{dtype}, {storage_width}]({', '.join(lanes)})"


def _scalar_operand(generator, arg, result_type):
    """``arg`` as a value of the scalar ``result_type``."""
    code = generator.generate_expression(arg)
    if not _is_literal(generator, arg) and MOJO_SCALAR_DTYPES.get(
        generator.type_name(generator.expression_result_type(arg))
    ) == MOJO_SCALAR_DTYPES.get(result_type):
        return code
    return f"{generator.map_type(result_type)}({code})"


def generate_mojo_min_max(generator, func_name, args):
    """The Mojo expression of the CrossGL call ``func_name(args)``, or ``None`` when it is not a numeric min/max/clamp."""
    if _ARITIES.get(func_name) != len(args):
        return None
    result_type = mojo_min_max_result_type(generator, args)
    if result_type is None:
        return None
    info = generator.vector_type_info(result_type)
    if info is not None:
        operands = [_vector_operand(generator, arg, info) for arg in args]
        if any(operand is None for operand in operands):
            return None
    else:
        operands = [_scalar_operand(generator, arg, result_type) for arg in args]
    if func_name == "clamp":
        value, lower, upper = operands
        return f"min(max({value}, {lower}), {upper})"
    return f"{func_name}({operands[0]}, {operands[1]})"
