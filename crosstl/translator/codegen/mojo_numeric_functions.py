"""CrossGL's floating-point classification, bit-count and wide-multiply built-ins in Mojo output.

The generator wrote these calls as they stand, and the foreign-call check then refused them, because Mojo spells
them differently or not at all. They are functions of Mojo 1.x's standard library, which the output imports whole
(``from std.math import *``; ``std.math`` re-exports ``isnan``, ``isinf``, ``isfinite``, ``nan`` and ``nextafter``
from ``std.utils.numerics``) or by name (``std.bit``):

- ``isnan(x)``, ``isinf(x)`` and ``isfinite(x)`` keep their names, for a floating-point or integer operand, scalar
  or SIMD; each returns a ``Bool`` for each lane. Mojo classifies an integer as a finite number, as converting it to
  a float first would;
- ``nextafter(x, y)`` keeps its name, for two floating-point operands of one type;
- ``rcp(x)`` is ``recip(x)`` (``1 / x``) and ``tgamma(x)`` is ``gamma(x)``, for a floating-point operand;
- ``nan()`` is ``nan[dtype]()`` of the floating-point type the context expects, ``Float32`` by default; OpenCL C's
  ``nan(nancode)``, whose payload Mojo cannot set, is left to the foreign-call refusal;
- ``clz(x)`` and ``ctz(x)`` are ``count_leading_zeros(x)`` and ``count_trailing_zeros(x)`` of ``std.bit``, for a
  32-bit integer operand, scalar or SIMD: like WGSL's ``countLeadingZeros`` and ``countTrailingZeros`` they count
  within the operand's 32 bits, so 0 gives 32, and they keep its type;
- ``mulhi(x, y)`` is the high 32 bits of the 64-bit product of two 32-bit integers of one signedness, scalar or SIMD
  (Metal's ``mulhi``, MSL 3.2 section 6.4): both are widened to 64 bits, multiplied, shifted right by 32 (an
  arithmetic shift for a signed product) and narrowed back. For two 64-bit scalars it is ``std.math.mulhi``, which
  Mojo defines only for scalar ``Int64`` and ``UInt64``.

Any other operand type, such as a boolean operand of ``isnan``, a 64-bit ``clz`` or a float ``mulhi``, gets
``None``. ``clz``, ``ctz``, ``rcp``, ``tgamma`` and ``nan`` are not Mojo functions, so such a call is refused by the
foreign-call check; ``isnan`` and the other names Mojo shares are refused here instead, by :class:`MojoRefusal`.
"""

from __future__ import annotations

from ..refusals import MojoRefusal
from .mojo_tables import MOJO_SCALAR_DTYPES

__all__ = [
    "MOJO_NUMERIC_CALLABLE_NAMES",
    "MOJO_NUMERIC_FUNCTIONS",
    "generate_mojo_numeric_function_call",
    "mojo_numeric_function_result_type",
]

_FLOAT_DTYPES = frozenset(
    {"DType.float16", "DType.bfloat16", "DType.float32", "DType.float64"}
)
_INTEGER_32 = {"DType.int32": "DType.int64", "DType.uint32": "DType.uint64"}
_INTEGER_64 = frozenset({"DType.int64", "DType.uint64"})
_INTEGER_DTYPES = frozenset(
    f"DType.{sign}int{bits}" for sign in ("", "u") for bits in (8, 16, 32, 64)
)
# CrossGL's floating-point scalar types, as the dtype nan() is instantiated with.
_NAN_DTYPES = {
    "half": "DType.float16",
    "float16_t": "DType.float16",
    "float": "DType.float32",
    "double": "DType.float64",
}

_SAME_NAME = frozenset({"isnan", "isinf", "isfinite"})
_FLOAT_RENAMES = {"rcp": "recip", "tgamma": "gamma"}
_BIT_COUNTS = {"clz": "count_leading_zeros", "ctz": "count_trailing_zeros"}

MOJO_NUMERIC_FUNCTIONS = frozenset(
    _SAME_NAME | set(_FLOAT_RENAMES) | set(_BIT_COUNTS) | {"nextafter", "nan", "mulhi"}
)
# The functions the lowerings write that the Mojo callable-name list does not already have.
MOJO_NUMERIC_CALLABLE_NAMES = frozenset(
    {"isnan", "isinf", "isfinite", "nextafter", "mulhi"}
)


def generate_mojo_numeric_function_call(generator, func_name, args, target_type=None):
    """The Mojo spelling of a CrossGL numeric built-in call, or ``None`` when its operands have no Mojo form."""
    if func_name not in MOJO_NUMERIC_FUNCTIONS:
        return None
    if generator.is_user_defined_function(func_name):
        return None
    if func_name == "nan":
        if args:
            return None
        dtype = _NAN_DTYPES.get(generator.type_name(target_type), "DType.float32")
        return f"nan[{dtype}]()"
    dtypes = [generator.expression_mojo_dtype(arg) for arg in args]
    if func_name == "nextafter":
        return _next_after(generator, args, dtypes)
    if func_name == "mulhi":
        return _multiply_high(generator, args, dtypes)
    if len(args) != 1:
        return _refuse(func_name, dtypes)
    if func_name in _SAME_NAME:
        if dtypes[0] not in _FLOAT_DTYPES | _INTEGER_DTYPES:
            return _refuse(func_name, dtypes)
        return f"{func_name}({_operands(generator, args)})"
    if func_name in _BIT_COUNTS:
        if dtypes[0] not in _INTEGER_32:
            return None
        name = _BIT_COUNTS[func_name]
        generator.required_bit_imports.add(name)
        return f"{name}({_operands(generator, args)})"
    if dtypes[0] not in _FLOAT_DTYPES:
        return None
    # Every name left here is one of the floating-point renames.
    name = _FLOAT_RENAMES[func_name]
    return f"{name}({_operands(generator, args)})"


def _refuse(func_name, dtypes):
    """Refuse a call to a function Mojo also defines, whose operands Mojo's function does not take as CrossGL's."""
    if func_name not in MOJO_NUMERIC_CALLABLE_NAMES:
        return None
    operands = ", ".join(str(dtype) for dtype in dtypes) or "no operands"
    raise MojoRefusal(
        f"Mojo has no {func_name} for operands of type {operands}; its {func_name} takes "
        + (
            "32-bit integers or integer vectors of one width, or two 64-bit integer scalars of one type"
            if func_name == "mulhi"
            else (
                "floating-point or integer values"
                if func_name in _SAME_NAME
                else "two floating-point values of one type"
            )
        )
    )


def _next_after(generator, args, dtypes):
    """``nextafter(x, y)`` for floats; a scalar ``y`` of another float width is converted to ``x``'s type first."""
    if (
        len(args) != 2
        or dtypes[0] not in _FLOAT_DTYPES
        or dtypes[1] not in _FLOAT_DTYPES
    ):
        return _refuse("nextafter", dtypes)
    x_type = generator.expression_result_type(args[0])
    y_type = generator.expression_result_type(args[1])
    x, y = (generator.generate_expression(arg) for arg in args)
    if dtypes[0] != dtypes[1]:
        if generator.vector_type_info(x_type) or generator.vector_type_info(y_type):
            return _refuse("nextafter", dtypes)
        y = f"{generator.map_type(x_type)}({y})"
    elif generator.vector_type_info(x_type) != generator.vector_type_info(y_type):
        return _refuse("nextafter", dtypes)
    return f"nextafter({x}, {y})"


def _operands(generator, args):
    return ", ".join(generator.generate_expression(arg) for arg in args)


def _multiply_high_type(generator, args):
    """The CrossGL type ``mulhi`` multiplies its operands in, or ``None`` when Mojo has no form of the product.

    Two 32-bit operands of different signedness meet in the unsigned type, as C's usual arithmetic conversions
    make them; a scalar operand of a vector product is broadcast to the vector's width.
    """
    if len(args) != 2:
        return None
    types = [generator.expression_result_type(arg) for arg in args]
    dtypes = [generator.expression_mojo_dtype(arg) for arg in args]
    infos = [generator.vector_type_info(type_name) for type_name in types]
    widths = {info[1] for info in infos if info is not None}
    if len(widths) > 1:
        return None
    if dtypes[0] in _INTEGER_64 or dtypes[1] in _INTEGER_64:
        if dtypes[0] != dtypes[1] or widths:
            return None
        return types[0]
    if dtypes[0] not in _INTEGER_32 or dtypes[1] not in _INTEGER_32:
        return None
    dtype = dtypes[0] if dtypes[0] == dtypes[1] else "DType.uint32"
    if widths:
        return generator.vector_type_name_for_dtype_width(dtype, widths.pop())
    return "int" if dtype == "DType.int32" else "uint"


def _multiply_high(generator, args, dtypes):
    product_type = _multiply_high_type(generator, args)
    if product_type is None:
        return _refuse("mulhi", dtypes)
    mojo_type = generator.map_type(product_type)
    # Each operand is converted to the product's Mojo type first, so a literal operand is a value with ``cast``.
    left, right = (f"{mojo_type}({generator.generate_expression(arg)})" for arg in args)
    info = generator.vector_type_info(product_type)
    dtype = info[0] if info is not None else MOJO_SCALAR_DTYPES[product_type]
    if dtype in _INTEGER_64:
        return f"mulhi({left}, {right})"
    wide = _INTEGER_32[dtype]
    return f"(({left}.cast[{wide}]() * {right}.cast[{wide}]()) >> 32).cast[{dtype}]()"


def mojo_numeric_function_result_type(generator, func_name, args):
    """The CrossGL type of a numeric built-in call the generator lowers, or ``None``.

    The classifications return a ``bool`` per lane of their operand; ``nan()`` is a ``float``; the others keep the
    type of their first operand. The assignment and arithmetic conversions read it, so ``int c = clz(u)`` converts
    the ``UInt32`` count as CrossGL does.
    """
    if func_name not in MOJO_NUMERIC_FUNCTIONS or generator.is_user_defined_function(
        func_name
    ):
        return None
    if func_name == "nan":
        return None if args else "float"
    if not args:
        return None
    if func_name == "mulhi":
        return _multiply_high_type(generator, args)
    operand_type = generator.expression_result_type(args[0])
    if func_name not in _SAME_NAME:
        return operand_type
    info = generator.vector_type_info(operand_type)
    if info is None:
        return "bool" if operand_type is not None else None
    return generator.vector_type_name_for_dtype_width("DType.bool", info[1])
