"""Prove every current MLX unary kernel natively on Metal, OpenGL and DirectX.

``unary.metal`` instantiates 877 kernels: four base layouts and a
work-per-thread variant of ``unary_v`` for thirty-seven operators over MLX's
dtypes, complex64 and the FP8 conversions. Each entry translates with the
configuration of its complete-family translation contract and must reproduce
the artifact that contract pins; the seventy-five entries whose operator or
helper changed upstream reproduce the current artifacts this gate's contract
pins.
Every entry then executes two layout cases on Metal, Mesa EGL and Direct3D 12
WARP.

Inputs come from per-operator domains whose results are exactly defined:
integer and rounding operators take every boundary value that C++ defines,
FP8 conversions follow their bit-level algorithm exhaustively, and
transcendental operators take only the points where IEEE-754, C99 or the MLX
helper fixes the result (signed zeros, infinities, NaN, poles, overflow and
saturation). Each output element is compared bit for bit with an independent
model, NaN results accept any NaN, and every guard after the output must stay
unchanged. Metal also runs the unchanged upstream kernel, which must produce
the same bytes. Accuracy away from those points is covered by the
per-operator unary gates.
"""

import functools
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    load_project_config,
    translate_project,
    validate_project_report,
)
from crosstl.project.runtime_value_encoding import FLOAT16_BITS
from demos.integrations.mlx.tests.kernels import (
    test_unary_complete_directx as complete_directx,
)
from demos.integrations.mlx.tests.kernels import (
    test_unary_complete_opengl as complete_opengl,
)
from demos.integrations.mlx.tests.kernels import (
    test_unary_native_loader as complete_metal,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_binary_family import (
    ANY_NAN,
    NAN,
    _add,
    _check,
    _complex_model,
    _concrete,
    _decoys,
    _different,
    _divide,
    _f,
    _fill,
    _is_zero,
    _less,
    _metal_wanted,
    _metal_words,
    _multiply,
    _negate,
    _value,
    _word,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.kernels.test_current_copy_family import (
    EDGES,
    INTEGER_RANGES,
    _float_value,
    _nan_bits,
    _round_float,
    _wrap,
)
from demos.integrations.mlx.tests.kernels.test_current_ternary import (
    GUARDS,
    ITEM_SIZES,
    _element_type,
    _metal_bytes,
    _runtime_value,
)
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/unary.metal"
SOURCE_SHA256 = "51af04126d68e1f5baee5f467268408650d24a68db66e8c044f7f0be3f15368b"
HEADERS = {
    "mlx/backend/metal/kernels/unary.h": (
        "fb952a2c30a6d0cd01f6c61e852cc15e426b286475de8ff139bb65b8d9001c8c"
    ),
    "mlx/backend/metal/kernels/unary_ops.h": (
        "af710925f066b029f67289a0bbe4edd0ba6c5e7f51c9df609abb111216fa963f"
    ),
    "mlx/backend/metal/kernels/utils.h": (
        "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf"
    ),
    "mlx/backend/metal/kernels/fp8.h": (
        "9a9d77a6295f66b4ada4311c3b855b6ecb7a2b38f28623109e3c61d2e17e81ae"
    ),
    "mlx/backend/metal/kernels/erf.h": (
        "31d88004390bbe08636c5f64a3cb3180ebc8d79a89267347de9f6c33aba0e9f9"
    ),
    "mlx/backend/metal/kernels/expm1f.h": (
        "dbdf551f45723639da54d72e327008bde6ef01a2f1ed561bb065e251ab6c9f1a"
    ),
    "mlx/backend/metal/kernels/cexpf.h": (
        "88b6e15a52a5800d98d9bc6da840ca5cf70bf572fda136409580c1f17b1e0aab"
    ),
    "mlx/backend/metal/kernels/complex.h": (
        "16e8a815b2cbdb6070e0824e64fe33fccb6e918f1b84ea5c792bd89d33e57bf1"
    ),
}
# The headers at the complete-family commit. ``unary_ops.h`` moved complex
# ``Abs`` and ``Sign`` to ``hypot`` and ``Sigmoid`` to the precise exponential;
# ``utils.h`` added that ``hypot``; ``fp8.h`` rewrote the E4M3 decoding.
COMPLETE_HEADERS = {
    "mlx/backend/metal/kernels/unary_ops.h": (
        "0a5492b65ae39ecb6d8b04e64ea5007e0a4ff60d8d9428559bfbfd03387ece2a"
    ),
    "mlx/backend/metal/kernels/utils.h": (
        "5e1568e9edde9d05dbf86f68fa0d6c6240f2c32b973c7c6a76166b9c0d91543d"
    ),
    "mlx/backend/metal/kernels/fp8.h": (
        "7c25dcdf0dc49af5fda0bc379e495b196347c6eaceb6bafcb4e5848c4ec851b7"
    ),
}
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_UNARY_FAMILY"
SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_UNARY_FAMILY_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_UNARY_FAMILY_SHARD_COUNT"
CI_SHARD_COUNT = 2
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/unary.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
INTEGERS = ("uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64")
FLOATS = ("float16", "float32", "bfloat16")
# ``instantiate_unary_int`` uses the base layouts only for 8-byte integers.
WIDE = ("uint64", "int64")
TYPES = ("bool_", *INTEGERS, *FLOATS)
BASE_KINDS = ("v", "v2", "gn1", "gn4large")
ALL_KINDS = (*BASE_KINDS, "vn")


def _same(operator, dtypes):
    return [(operator, dtype, dtype, dtype not in WIDE) for dtype in dtypes]


COMPLEX_OPERATORS = (
    "Abs",
    "ArcCos",
    "ArcSin",
    "ArcTan",
    "Conjugate",
    "Cos",
    "Cosh",
    "Exp",
    "Log",
    "Log1p",
    "Log2",
    "Log10",
    "Negative",
    "Sign",
    "Sin",
    "Sinh",
    "Square",
    "Sqrt",
    "Rsqrt",
    "Tan",
    "Tanh",
    "Round",
)
# (operator, input dtype, output dtype, has the work-per-thread kind), in the
# order of ``unary.metal``.
INSTANTIATIONS = (
    *_same("Abs", TYPES),
    *_same("ArcCos", FLOATS),
    *_same("ArcCosh", FLOATS),
    *_same("ArcSin", FLOATS),
    *_same("ArcSinh", FLOATS),
    *_same("ArcTan", FLOATS),
    *_same("ArcTanh", FLOATS),
    *_same("Ceil", TYPES),
    *_same("Cos", FLOATS),
    *_same("Cosh", FLOATS),
    *_same("Exp", FLOATS),
    *_same("Expm1", FLOATS),
    *_same("Floor", TYPES),
    *_same("Log", FLOATS),
    *_same("Log2", FLOATS),
    *_same("Log10", FLOATS),
    *_same("Log1p", FLOATS),
    *_same("Negative", TYPES),
    *_same("Sigmoid", FLOATS),
    *_same("Erf", FLOATS),
    *_same("ErfInv", FLOATS),
    *_same("Sign", TYPES),
    *_same("Sin", FLOATS),
    *_same("Sinh", FLOATS),
    *_same("Square", TYPES),
    *_same("Sqrt", FLOATS),
    *_same("Rsqrt", FLOATS),
    *_same("Tan", FLOATS),
    *_same("Tanh", FLOATS),
    *_same("Round", FLOATS),
    *_same("BitwiseInvert", INTEGERS),
    *((operator, "complex64", "complex64", False) for operator in COMPLEX_OPERATORS),
    ("Real", "complex64", "float32", False),
    ("Imag", "complex64", "float32", False),
    ("LogicalNot", "bool_", "bool_", True),
    *(("ToFP8", dtype, "uint8", True) for dtype in ("float16", "bfloat16", "float32")),
    *(
        ("FromFP8", "uint8", dtype, True)
        for dtype in ("float16", "bfloat16", "float32")
    ),
)
ENTRIES = tuple(
    f"{kind}_{operator}{source}{output}"
    for operator, source, output, work in INSTANTIATIONS
    for kind in (ALL_KINDS if work else BASE_KINDS)
)
ENTRY_TYPES = {
    f"{kind}_{operator}{source}{output}": (kind, operator, source, output)
    for operator, source, output, work in INSTANTIATIONS
    for kind in (ALL_KINDS if work else BASE_KINDS)
}
# kind: (kernel template, elements per thread or None for WorkPerThread<T>::n).
KINDS = {
    "v": ("unary_v", 1),
    "vn": ("unary_v", None),
    "v2": ("unary_v2", None),
    "gn1": ("unary_g", 1),
    "gn4large": ("unary_g", 4),
}
CASES = {
    "unary_v": {"tail": {"size": 37}, "exact": {"size": 64}},
    "unary_v2": {"tail": {"grid": [3, 2]}, "wide": {"grid": [4, 3]}},
    # ``unary_g`` reads ``strides[ndim - 2]``, so shapes have two dimensions
    # or more; the innermost extents (5 and 3) end inside a four-wide tile.
    "unary_g": {
        "transpose": {"shape": [3, 5], "strides": [1, 3]},
        "nd5-broadcast": {"shape": [2, 2, 1, 3, 3], "strides": [0, 9, 0, 3, 1]},
    },
}
# Metal ``[[buffer(n)]]`` slots of each kernel's constants.
BUFFER_INDICES = {
    "unary_v": {"size": 2},
    "unary_v2": {"size": 2},
    "unary_g": {"in_shape": 2, "in_strides": 3, "ndim": 4},
}
CONSTANT_FORMATS = {"uint32": "I", "int32": "i", "int64": "q"}
GUARD_COUNT = 16
# Inputs read once per dispatch start at a different domain rank per kind.
KIND_SALTS = {kind: index * 11 for index, kind in enumerate(KINDS)}


def _split(entry):
    """Return the kind, operator, input dtype and output dtype of an entry."""
    return ENTRY_TYPES[entry]


def _kernel(entry):
    return KINDS[_split(entry)[0]][0]


def _work(entry):
    """Return the elements one thread of the entry computes."""
    kind, _, source, _ = _split(entry)
    explicit = KINDS[kind][1]
    return explicit if explicit is not None else 8 // ITEM_SIZES[source]


def _output(entry):
    return _split(entry)[3]


# ---------------------------------------------------------------------------
# Exact values: ("nan",), ("inf", sign) or ("finite", sign, |x|).
# ---------------------------------------------------------------------------

ZERO = ("finite", 0, Fraction(0))
NEGATIVE_ZERO = ("finite", 1, Fraction(0))
ONE = ("finite", 0, Fraction(1))
HALF = ("finite", 0, Fraction(1, 2))
INF = ("inf", 0)
NEGATIVE_INF = ("inf", 1)


class Undefined(AssertionError):
    """The operator has no exactly defined result at this input."""


def _exact(value):
    """Require a binary32 intermediate that needs no rounding."""
    if value != NAN and value[0] == "finite":
        try:
            _word("float32", value)
        except AssertionError as error:
            raise Undefined(f"inexact binary32 intermediate {value}") from error
    return value


def _signed(sign, value):
    return value if value == NAN else (value[0], sign, *value[2:])


def _negative(value):
    """``x < 0`` (false for -0 and NaN)."""
    return _less(value, ZERO)


def _positive(value):
    return _less(ZERO, value)


def _magnitude_at_least(value, bound):
    return value != NAN and (value[0] == "inf" or value[2] >= bound)


def _sum(x, y):
    return _exact(_add(x, y))


def _difference(x, y):
    return _exact(_add(x, _negate(y)))


def _product(x, y):
    return _exact(_multiply(x, y))


def _quotient(x, y):
    return _exact(_divide(x, y))


def _integral(operator, x):
    """``ceil``, ``floor`` and ``rint``: exact, keeping the sign of zero."""
    if x == NAN or x[0] == "inf":
        return x
    magnitude, sign = x[2], x[1]
    whole, fraction = divmod(magnitude, 1)
    if operator == "Round":
        if fraction > Fraction(1, 2) or (fraction == Fraction(1, 2) and whole % 2):
            whole += 1
    elif fraction and (operator == "Ceil") != bool(sign):
        whole += 1
    return ("finite", sign, Fraction(whole))


def _sqrt(x):
    """Correctly rounded ``sqrt`` where the root is exact."""
    if x == NAN or _is_zero(x) or x == INF:
        return x
    if x[1]:
        return NAN
    numerator, denominator = x[2].numerator, x[2].denominator
    root_n, root_d = math.isqrt(numerator), math.isqrt(denominator)
    if root_n * root_n != numerator or root_d * root_d != denominator:
        raise Undefined(f"sqrt {x} is not exact")
    return ("finite", 0, Fraction(root_n, root_d))


def _rsqrt(x):
    if x == NAN:
        return NAN
    if _is_zero(x):
        return ("inf", x[1])
    if x == INF:
        return ZERO
    if x[1]:
        return NAN
    return _divide(ONE, _sqrt(x))


def _log(x):
    """``log``, ``log2`` and ``log10`` at their exactly defined points."""
    if x == NAN or _negative(x) or x == NEGATIVE_INF:
        return NAN
    if _is_zero(x):
        return NEGATIVE_INF
    if x == INF:
        return INF
    if x == ONE:
        return ZERO
    raise Undefined(f"log {x}")


def _atan2(y, x):
    """``atan2(y, x)`` where the angle is a signed zero."""
    if NAN in (y, x):
        return NAN
    if _is_zero(y) and _positive(x):
        return y
    if y[0] == "finite" and x == INF:
        return ("finite", y[1], Fraction(0))
    raise Undefined(f"atan2 {y} {x}")


def _hypot(x, y):
    """MLX ``hypot`` from ``utils.h`` where it is exact."""
    if "inf" in (x[0], y[0]):
        return INF
    if NAN in (x, y):
        return NAN
    a, b = max(x[2], y[2]), min(x[2], y[2])
    if a == 0:
        return ZERO
    quotient = _exact(("finite", 0, b / a))
    ratio = _product(quotient, quotient)
    root = _sqrt(_sum(ONE, ratio))
    if a == b:
        raise Undefined("hypot selects sqrt(2) * a")
    if root == ONE and ratio != ZERO:
        raise Undefined("hypot selects a + a * r / 2")
    return _product(("finite", 0, a), root)


def _copysign(magnitude, sign_source):
    if magnitude == NAN:
        return NAN
    assert sign_source != NAN, "copysign from a NaN sign is not modelled"
    return _signed(sign_source[1], magnitude)


# Real-function thresholds in every evaluation width the backends use for a
# dtype: |x| at and beyond these overflow exp, sinh and cosh, round exp(-|x|)
# to zero, saturate tanh to one, or make Sigmoid exactly one or zero.
EXP_OVERFLOW = {"float16": 12, "float32": 89, "bfloat16": 89}
SINH_OVERFLOW = {"float16": 12, "float32": 90, "bfloat16": 90}
EXP_UNDERFLOW = {"float16": 18, "float32": 110, "bfloat16": 110}
TANH_SATURATION = {"float16": 10, "float32": 20, "bfloat16": 20}
SIGMOID_ONE = {"float16": 12, "float32": 20, "bfloat16": 20}
SIGMOID_ZERO = {"float16": 18, "float32": 89, "bfloat16": 89}
# MLX ``erf`` and ``expm1f`` evaluate binary32 polynomials; past these the
# ``pow(2, a)`` and ``expm1f`` saturation paths make the result exact.
ERF_SATURATION = 10
EXPM1_OVERFLOW = {"float16": 12, "float32": 90, "bfloat16": 90}
EXPM1_SATURATION = {"float16": 18, "float32": 88, "bfloat16": 88}
LOG1P_EXACT = Fraction(1, 2**25)


def _real(operator, dtype, x):
    """Return the exact result of a real operator, or raise ``Undefined``."""
    nan = x == NAN
    if operator == "Abs":
        return NAN if nan else _signed(0, x)
    if operator == "Negative":
        return _negate(x)
    if operator in ("Ceil", "Floor", "Round"):
        return _integral(operator, x)
    if operator == "Sign":
        # (x > 0) - (x < 0) converts back from int: NaN and zeros give +0.
        return ONE if _positive(x) else _signed(1, ONE) if _negative(x) else ZERO
    if operator == "Square":
        return _multiply(x, x)
    if operator == "Sqrt":
        return _sqrt(x)
    if operator == "Rsqrt":
        return _rsqrt(x)
    if nan:
        return NAN
    zero = _is_zero(x)
    if operator in ("Sin", "Tan", "ArcTan", "ArcSinh", "Erf", "ErfInv", "Tanh"):
        if zero:
            return x
    if operator in ("Sin", "Cos", "Tan"):
        if x[0] == "inf":
            return NAN
        if operator == "Cos" and zero:
            return ONE
    elif operator in ("ArcSin", "ArcCos", "ArcTanh", "ErfInv"):
        if x[0] == "inf" or x[2] > 1:
            return NAN
        if operator == "ArcSin" and zero:
            return x
        if operator == "ArcCos" and x == ONE:
            return ZERO
        if operator in ("ArcTanh", "ErfInv"):
            if zero:
                return x
            if x[2] == 1:
                return ("inf", x[1])
    elif operator == "ArcSinh":
        if x[0] == "inf":
            return x
    elif operator == "ArcCosh":
        if _less(x, ONE) or x == NEGATIVE_INF:
            return NAN
        if x == ONE:
            return ZERO
        if x == INF:
            return INF
    elif operator in ("Sinh", "Cosh"):
        if zero:
            return x if operator == "Sinh" else ONE
        if _magnitude_at_least(x, SINH_OVERFLOW[dtype]):
            return ("inf", x[1] if operator == "Sinh" else 0)
    elif operator == "Tanh":
        if _magnitude_at_least(x, TANH_SATURATION[dtype]):
            return _signed(x[1], ONE)
    elif operator == "Exp":
        if zero:
            return ONE
        if not x[1] and _magnitude_at_least(x, EXP_OVERFLOW[dtype]):
            return INF
        if x[1] and _magnitude_at_least(x, EXP_UNDERFLOW[dtype]):
            return ZERO
    elif operator == "Expm1":
        if zero:
            return x
        if not x[1] and _magnitude_at_least(x, EXPM1_OVERFLOW[dtype]):
            return INF
        if x[1] and _magnitude_at_least(x, EXPM1_SATURATION[dtype]):
            return _signed(1, ONE)
    elif operator in ("Log", "Log2", "Log10"):
        return _log(x)
    elif operator == "Log1p":
        # utils.h: 1 + x == 1 returns x; 1 + x == Limits<float>::max, which
        # is +inf, returns +inf; otherwise x * (log(1 + x) / x).
        if zero or (x[0] == "finite" and x[2] < LOG1P_EXACT and dtype != "float16"):
            return x
        if x == INF:
            return INF
        if x[0] == "inf" or (x[1] and x[2] > 1):
            return NAN
        if x == _signed(1, ONE):
            return NEGATIVE_INF
    elif operator == "Erf":
        if _magnitude_at_least(x, ERF_SATURATION):
            return _signed(x[1], ONE)
    elif operator == "Sigmoid":
        if zero:
            return HALF
        if not x[1] and _magnitude_at_least(x, SIGMOID_ONE[dtype]):
            return ONE
        if x[1] and _magnitude_at_least(x, SIGMOID_ZERO[dtype]):
            return ZERO
    raise Undefined(f"{operator} {dtype} {x}")


def _float_model(operator, dtype, word):
    result = _real(operator, dtype, _value(dtype, word))
    return ANY_NAN if result == NAN else _overflowing_word(dtype, result)


def _overflowing_word(dtype, value):
    """Encode an exact value; magnitudes past the format's range round to inf."""
    if value[0] == "finite" and value[2]:
        bits, overflowed = _round_float(dtype, value[1], value[2])
        if overflowed:
            return _word(dtype, ("inf", value[1]))
    return _word(dtype, value)


# ---------------------------------------------------------------------------
# Integer and bool operators (C++ semantics after integral promotion).
# ---------------------------------------------------------------------------


def _signed_type(dtype):
    return INTEGER_RANGES[dtype][0] < 0


def _int_defined(operator, dtype, x):
    low, high = INTEGER_RANGES[dtype]
    narrow = ITEM_SIZES[dtype] < 4
    if operator == "Abs":
        return x != low if _signed_type(dtype) else True
    if operator == "Negative":
        # Narrow types negate in int; 32- and 64-bit signed minimums overflow.
        return narrow or not _signed_type(dtype) or x != low
    if operator == "Square":
        if narrow:
            return x * x <= 2**31 - 1
        return _signed_type(dtype) is False or x * x <= high
    return True


def _int_model(operator, dtype, x):
    assert _int_defined(operator, dtype, x), (operator, dtype, x)
    wrap = functools.partial(_wrap, dtype)
    if operator == "Abs":
        return abs(x)
    if operator in ("Ceil", "Floor"):
        return x
    if operator == "Negative":
        return wrap(-x)
    if operator == "Sign":
        return (x > 0) - (x < 0)
    if operator == "Square":
        return wrap(x * x)
    if operator == "BitwiseInvert":
        return wrap(~x)
    raise AssertionError(operator)


def _bool_model(operator, x):
    if operator == "LogicalNot":
        return 1 - x
    # Abs, Ceil, Floor, Negative, Sign and Square all map bool to itself.
    assert operator in ("Abs", "Ceil", "Floor", "Negative", "Sign", "Square")
    return x


# ---------------------------------------------------------------------------
# complex64 operators: unary_ops.h, complex.h and utils.h formulas with exact
# binary32 intermediates.
# ---------------------------------------------------------------------------


def _c_add(a, b):
    return (_sum(a[0], b[0]), _sum(a[1], b[1]))


def _c_mul(a, b):
    return (
        _difference(_product(a[0], b[0]), _product(a[1], b[1])),
        _sum(_product(a[0], b[1]), _product(a[1], b[0])),
    )


def _c_div(a, b):
    denominator = _sum(_product(b[0], b[0]), _product(b[1], b[1]))
    real = _sum(_product(a[0], b[0]), _product(a[1], b[1]))
    imag = _difference(_product(a[1], b[0]), _product(a[0], b[1]))
    return (_quotient(real, denominator), _quotient(imag, denominator))


def _c_scalar_div(scalar, b):
    """``operator/(U, complex_t)``: ``{s * b.re, -s * b.im} / |b|^2``."""
    denominator = _sum(_product(b[0], b[0]), _product(b[1], b[1]))
    real = _product(scalar, b[0])
    imag = _product(_negate(scalar), b[1])
    return (_quotient(real, denominator), _quotient(imag, denominator))


def _c_scalar_sub(scalar, b):
    return (_difference(scalar, b[0]), _negate(b[1]))


def _c_scalar_add(scalar, b):
    return (_sum(scalar, b[0]), b[1])


IMAGINARY_UNIT = (ZERO, ONE)


def _c_is_zero(x):
    return _is_zero(x[0]) and _is_zero(x[1])


def _c_log(x):
    return (_log(_hypot(*x)), _atan2(x[1], x[0]))


def _c_sqrt(x):
    if _c_is_zero(x):
        return (ZERO, ZERO)
    radius = _hypot(*x)
    two = ("finite", 0, Fraction(2))
    real = _sqrt(_quotient(_sum(radius, x[0]), two))
    imag = _sqrt(_quotient(_difference(radius, x[0]), two))
    return (real, _copysign(imag, x[1]))


def _divide_by_log_base(value):
    """``y / M_LN2_F`` or ``y / M_LN10_F``: exact for zeros, infinities, NaN."""
    if value == NAN or value[0] == "inf" or _is_zero(value):
        return value
    raise Undefined(f"{value} / ln(base)")


def _complex(operator, x):
    """Return the exact complex64 result of an operator, or raise."""
    re, im = x
    real = functools.partial(_real, dtype="float32")
    if operator == "Abs":
        return (_hypot(re, im), ZERO)
    if operator == "Negative":
        return (_negate(re), _negate(im))
    if operator == "Conjugate":
        return (re, _negate(im))
    if operator == "Round":
        return (_integral("Round", re), _integral("Round", im))
    if operator == "Square":
        return _c_mul(x, x)
    if operator == "Sign":
        if _c_is_zero(x):
            return x
        radius = _hypot(re, im)
        return (_quotient(re, radius), _quotient(im, radius))
    if operator == "Sqrt":
        return _c_sqrt(x)
    if operator == "Rsqrt":
        return _c_scalar_div(ONE, _c_sqrt(x))
    if operator == "Log":
        return _c_log(x)
    if operator in ("Log2", "Log10"):
        return tuple(_divide_by_log_base(part) for part in _c_log(x))
    if operator == "Log1p":
        return _complex_log1p(re, im)
    if operator == "Exp":
        return _cexpf(re, im)
    if operator == "Cos":
        return (
            _product(real("Cos", x=re), real("Cosh", x=im)),
            _product(_negate(real("Sin", x=re)), real("Sinh", x=im)),
        )
    if operator == "Cosh":
        return (
            _product(real("Cosh", x=re), real("Cos", x=im)),
            _product(real("Sinh", x=re), real("Sin", x=im)),
        )
    if operator == "Sin":
        return (
            _product(real("Sin", x=re), real("Cosh", x=im)),
            _product(real("Cos", x=re), real("Sinh", x=im)),
        )
    if operator == "Sinh":
        return (
            _product(real("Sinh", x=re), real("Cos", x=im)),
            _product(real("Cosh", x=re), real("Sin", x=im)),
        )
    if operator == "Tan":
        tan_a, tanh_b = real("Tan", x=re), real("Tanh", x=im)
        t1 = _product(tan_a, tanh_b)
        denominator = _sum(ONE, _product(t1, t1))
        return (
            _quotient(_difference(tan_a, _product(tanh_b, t1)), denominator),
            _quotient(_sum(tanh_b, _product(tan_a, t1)), denominator),
        )
    if operator == "Tanh":
        tanh_a, tan_b = real("Tanh", x=re), real("Tan", x=im)
        t1 = _product(tanh_a, tan_b)
        denominator = _sum(ONE, _product(t1, t1))
        return (
            _quotient(_sum(tanh_a, _product(tan_b, t1)), denominator),
            _quotient(_difference(tan_b, _product(tanh_a, t1)), denominator),
        )
    if operator in ("ArcCos", "ArcSin"):
        root = _c_sqrt(_c_scalar_sub(ONE, _c_mul(x, x)))
        if operator == "ArcCos":
            y = _c_log(_c_add(x, _c_mul(IMAGINARY_UNIT, root)))
        else:
            y = _c_log(_c_add(_c_mul(IMAGINARY_UNIT, x), root))
        return (y[1], _negate(y[0]))
    if operator == "ArcTan":
        ix = _c_mul(IMAGINARY_UNIT, x)
        ratio = _c_div(_c_scalar_add(ONE, ix), _c_scalar_sub(ONE, ix))
        scale = _c_scalar_div(ONE, (ZERO, ("finite", 0, Fraction(2))))
        return _c_mul(scale, _c_log(ratio))
    raise AssertionError(operator)


def _complex_log1p(x, y):
    """``log1p(complex64_t)`` from ``utils.h`` at exactly defined points."""
    radius = _sqrt(_sum(_product(x, x), _product(y, y)))
    theta = _atan2(y, _sum(x, ONE))
    if _less(radius, HALF):
        r = _sum(_product(x, _sum(("finite", 0, Fraction(2)), x)), _product(y, y))
        if _is_zero(r):
            return (x, theta)
        raise Undefined("complex log1p of a small nonzero radius")
    shifted = _sum(x, ONE)
    z0 = _sqrt(_sum(_product(shifted, shifted), _product(y, y)))
    return (_log(z0), theta)


def _cexpf(x, y):
    """``cexpf`` from ``cexpf.h`` at exactly defined points."""
    real = functools.partial(_real, dtype="float32")
    if y != NAN and _is_zero(y):
        return (real("Exp", x=x), y)
    if x != NAN and _is_zero(x):
        return (real("Cos", x=y), real("Sin", x=y))
    if y == NAN or y[0] == "inf":
        if x == NAN or x[0] != "inf":
            return (NAN, NAN)
        return (ZERO, ZERO) if x[1] else (x, NAN)
    if x == NAN:
        return (NAN, NAN)
    raise Undefined(f"cexpf {x} {y}")


def _complex_model(operator, word):
    x = tuple(_value("float32", part) for part in word)
    result = _complex(operator, x)
    return tuple(ANY_NAN if part == NAN else _word("float32", part) for part in result)


# ---------------------------------------------------------------------------
# FP8 (E4M3) conversions from fp8.h.
# ---------------------------------------------------------------------------


def _binary32_bits(dtype, word):
    """Widen a float16, bfloat16 or float32 word to binary32 bits exactly."""
    if dtype == "float32":
        return word
    if dtype == "bfloat16":
        return word << 16
    sign = (word & 0x8000) << 16
    exponent, mantissa = (word >> 10) & 0x1F, word & 0x3FF
    if exponent == 0x1F:
        return sign | 0x7F800000 | (mantissa << 13)
    if exponent == 0 and mantissa == 0:
        return sign
    value = struct.unpack("<e", struct.pack("<H", word))[0]
    return struct.unpack("<I", struct.pack("<f", value))[0] | sign


def _to_fp8(dtype, word):
    """``fp8_e4m3(f).bits``: saturating, round-to-nearest-even E4M3."""
    bits32 = _binary32_bits(dtype, word)
    sign = bits32 & 0x80000000
    bits32 ^= sign
    if bits32 >= 543 << 21:
        bits = 0x7E
    elif bits32 < 121 << 23:
        # Adding 2**14 rounds the magnitude to a multiple of 2**-9.
        _, _, magnitude = _float_value("float32", bits32)
        total, overflowed = _round_float("float32", 0, magnitude + 2**14)
        assert not overflowed
        bits = (total - (141 << 23)) & 0xFF
    else:
        odd = (bits32 >> 20) & 1
        bits32 = (bits32 + (((7 - 127) << 23) & 0xFFFFFFFF) + 0x7FFFF) & 0xFFFFFFFF
        bits = ((bits32 + odd) >> 20) & 0xFF
    return bits | (sign >> 24)


def _from_fp8(output, byte):
    """``float(fp8_e4m3)`` via the binary16 decoding, scaled by 256."""
    v = byte & 127
    half = (v << 7) | (((v + 1) >> 7) << 14) | (((byte >> 7) & 1) << 15)
    kind, sign, magnitude = _float_value("float16", half)
    if kind == "nan":
        return ANY_NAN
    assert kind == "finite"
    return _word(output, ("finite", sign, magnitude * 256))


def _model(entry, value):
    _, operator, source, output = _split(entry)
    if operator == "ToFP8":
        return _to_fp8(source, value)
    if operator == "FromFP8":
        return _from_fp8(output, value)
    if operator in ("Real", "Imag"):
        part = value[0 if operator == "Real" else 1]
        return ANY_NAN if _value("float32", part) == NAN else part
    if source == "complex64":
        return _complex_model(operator, value)
    if source == "bool_":
        return _bool_model(operator, value)
    if source in INTEGERS:
        return _int_model(operator, source, value)
    return _float_model(operator, source, value)


# ---------------------------------------------------------------------------
# Input domains: every value is defined and exactly modelled.
# ---------------------------------------------------------------------------

SPECIALS = ["0", "-0", "inf", "-inf", "nan"]
NEGATIVE_NANS = {"float16": 0xFE00, "float32": 0xFFC00000, "bfloat16": 0xFFC0}
MAX_FINITE = {"float16": 0x7BFF, "float32": 0x7F7FFFFF, "bfloat16": 0x7F7F}
MIN_NORMAL = {"float16": 0x0400, "float32": 0x00800000, "bfloat16": 0x0080}


def _both_signs(values):
    return [sign * Fraction(value) for value in values for sign in (1, -1)]


# Fractions, ties and integers that ceil, floor and rint treat differently.
ROUNDING = _both_signs(
    ["1/4", "1/2", "3/4", "1", "5/4", "3/2", "5/2", "7/2", "31/4", "63/2", "255/2"]
)
# The last fractional values and first integers of each precision.
PRECISION_EDGES = {
    "float16": ["1023.5", "-1023.5", "2047", "-1024"],
    "float32": ["8388607.5", "-8388607.5", "8388609", "-16777215"],
    "bfloat16": ["127.5", "-127.5", "255", "-256"],
}
# Products of these need at most six significant bits.
SMALL = _both_signs(k * Fraction(2) ** e for k in (1, 3, 5, 7) for e in range(-2, 4))
SQUARES = [
    "1/16",
    "1/4",
    "1",
    "9/4",
    "4",
    "25/4",
    "9",
    "49/4",
    "16",
    "121/4",
    "64",
    "100",
    "1024",
]
POWERS_OF_FOUR = ["1/16", "1/4", "1", "4", "16", "64"]
# Magnitudes past every overflow, underflow and saturation threshold above.
LARGE = {
    "float16": ["20", "100", "1000"],
    "float32": ["110", "1000", str(2**64)],
    "bfloat16": ["110", "1000", str(2**64)],
}
OUTSIDE_UNIT = ["3/2", "-3/2", "2", "-2", "100", "-100"]


def _words(dtype, values):
    return [value if isinstance(value, int) else _f(dtype, value) for value in values]


def _large(dtype):
    magnitudes = _words(dtype, LARGE[dtype]) + [MAX_FINITE[dtype]]
    sign = 1 << (16 if dtype != "float32" else 32) - 1
    return magnitudes + [word | sign for word in magnitudes]


def _float_domain(operator, dtype):
    words = functools.partial(_words, dtype)
    specials = words(SPECIALS) + [NEGATIVE_NANS[dtype]]
    if operator in ("Abs", "Negative", "Sign", "Ceil", "Floor", "Round"):
        extremes = [
            MAX_FINITE[dtype],
            MAX_FINITE[dtype] | (0x8000 if dtype != "float32" else 0x80000000),
        ]
        extremes += [
            MIN_NORMAL[dtype],
            MIN_NORMAL[dtype] | (0x8000 if dtype != "float32" else 0x80000000),
        ]
        return specials + words(ROUNDING + PRECISION_EDGES[dtype]) + extremes
    if operator == "Square":
        return specials + words(SMALL) + [MAX_FINITE[dtype]]
    if operator == "Sqrt":
        return specials + words(SQUARES + ["-1", "-4", "-1/4", "-100"])
    if operator == "Rsqrt":
        return specials + words(POWERS_OF_FOUR + ["-1", "-4", "-1/4"])
    if operator in ("Sin", "Cos", "Tan", "ArcSinh"):
        return specials
    if operator == "ArcTan":
        # atan(+-inf) is +-pi/2, which binary formats round, and Apple GPUs
        # return +0 from ``metal::precise::atan(-0)``.
        return words(["0", "nan"]) + [NEGATIVE_NANS[dtype]]
    if operator == "ArcSin":
        return specials + words(OUTSIDE_UNIT)
    if operator == "ArcCos":
        # acos(+-0) is pi/2; only acos(1) is exact.
        return words(["inf", "-inf", "nan", "1"] + OUTSIDE_UNIT) + [
            NEGATIVE_NANS[dtype]
        ]
    if operator in ("ArcTanh", "ErfInv"):
        return specials + words(["1", "-1"] + OUTSIDE_UNIT)
    if operator == "ArcCosh":
        return specials + words(["1", "1/2", "1/4", "-1", "-5/2"])
    if operator in ("Sinh", "Cosh", "Tanh", "Exp", "Expm1", "Erf", "Sigmoid"):
        return specials + _large(dtype)
    if operator in ("Log", "Log2", "Log10"):
        return specials + words(["1", "-1", "-1/4", "-5/2", "-100"])
    if operator == "Log1p":
        extra = ["-1", "-3/2", "-2", "-100"]
        if dtype != "float16":
            extra += _both_signs([Fraction(1, 2**30), Fraction(1, 2**40)])
        # 1 + FLT_MAX rounds to FLT_MAX, which takes the logarithm path.
        return specials + words(extra)
    raise AssertionError(operator)


def _fp8_sources(dtype):
    """Every E4M3 magnitude, every midpoint and values beside the midpoints."""
    values = set()
    for code in range(126):
        low = _from_fp8("float32", code)
        high = _from_fp8("float32", code + 1)
        low_value = _float_value("float32", low)[2]
        high_value = _float_value("float32", high)[2]
        values.update({low_value, (low_value + high_value) / 2})
    values.update(
        {
            Fraction(448),
            Fraction(464),
            Fraction(480),
            Fraction(1000),
            Fraction(1, 2**12),
        }
    )
    words = _words(dtype, sorted(values)[::2] + sorted(values)[1::4])
    words += _words(
        dtype, ["-448", "-464", "-480", "-1/512", "-3/1024", "-1", "-0.9375"]
    )
    words += _words(dtype, SPECIALS)
    # ``static_cast<float>(half)`` on Apple GPUs drops the sign of a NaN,
    # which ``fp8_e4m3`` then reads.
    if dtype != "float16":
        words.append(NEGATIVE_NANS[dtype])
    return list(dict.fromkeys(words))


def _int_domain(operator, dtype):
    low, high = INTEGER_RANGES[dtype]
    values = list(dict.fromkeys(int(value) for value in EDGES[dtype]))
    if operator == "Square":
        limit = math.isqrt(2**31 - 1 if ITEM_SIZES[dtype] < 4 else high)
        values += [
            value for value in (limit, -limit, limit + 1) if low <= value <= high
        ]
    return [
        value for value in dict.fromkeys(values) if _int_defined(operator, dtype, value)
    ]


def _complex_pairs(pairs):
    return [(_f("float32", re), _f("float32", im)) for re, im in pairs]


SIGNED_ZEROS = [(re, im) for re in ("0", "-0") for im in ("0", "-0")]
COMPLEX_NANS = [("nan", "0"), ("0", "nan")]
COMPLEX_PARTS = [
    "0",
    "-0",
    "1",
    "-1",
    "5/2",
    "-125/4",
    "1/2",
    "-3/2",
    "inf",
    "-inf",
    "nan",
]
COMPLEX_DOMAINS = {
    "Abs": [
        ("3", "4"),
        ("-3", "4"),
        ("4", "-3"),
        ("-3/4", "-1"),
        ("12", "-16"),
        ("3/8", "1/2"),
        ("5/2", "0"),
        ("-125/4", "-0"),
        ("0", "7"),
        ("-0", "-1/4"),
        *SIGNED_ZEROS,
        ("inf", "nan"),
        ("nan", "-inf"),
        ("-inf", "2"),
        ("nan", "1"),
        ("1", "nan"),
        ("nan", "nan"),
    ],
    "Sign": [
        ("5/2", "0"),
        ("-125/4", "-0"),
        ("1/4", "0"),
        ("-7", "0"),
        ("0", "1"),
        ("-0", "-5/2"),
        ("0", "125/4"),
        *SIGNED_ZEROS,
        ("inf", "1"),
        ("-3", "inf"),
        ("nan", "1"),
        ("1", "nan"),
    ],
    "Square": [
        ("1/4", "3"),
        ("-6", "5/2"),
        ("7", "-1/2"),
        ("-14", "-28"),
        ("3/4", "7/4"),
        ("5", "0"),
        ("-0", "6"),
        *SIGNED_ZEROS,
        ("inf", "0"),
        ("0", "inf"),
        ("nan", "1"),
    ],
    "Sqrt": [
        ("4", "0"),
        ("4", "-0"),
        ("-4", "0"),
        ("-4", "-0"),
        ("3", "4"),
        ("3", "-4"),
        ("-3", "4"),
        ("-3", "-4"),
        ("0", "2"),
        ("0", "-2"),
        ("-0", "8"),
        ("12", "16"),
        ("3/4", "1"),
        ("9/4", "0"),
        ("-25/4", "-0"),
        *SIGNED_ZEROS,
        ("inf", "0"),
        ("1", "inf"),
        ("1", "-inf"),
        ("nan", "1"),
        ("1", "nan"),
    ],
    "Rsqrt": [
        ("4", "0"),
        ("4", "-0"),
        ("-4", "0"),
        ("-4", "-0"),
        ("0", "2"),
        ("0", "-8"),
        ("16", "0"),
        ("1/4", "-0"),
        *SIGNED_ZEROS,
        ("inf", "0"),
        ("nan", "1"),
    ],
    "Exp": [
        *SIGNED_ZEROS,
        ("-inf", "0"),
        ("inf", "-0"),
        ("nan", "0"),
        ("110", "0"),
        ("-110", "-0"),
        ("0", "inf"),
        ("-0", "nan"),
        ("1", "inf"),
        ("-inf", "inf"),
        ("-inf", "nan"),
        ("inf", "nan"),
        ("inf", "-inf"),
        ("nan", "1"),
    ],
    "Log": [
        ("1", "0"),
        ("1", "-0"),
        ("inf", "0"),
        ("inf", "-0"),
        ("inf", "1"),
        ("inf", "-5/2"),
        ("inf", "nan"),
        ("nan", "0"),
        ("0", "nan"),
        ("1", "nan"),
    ],
    "Log1p": [*SIGNED_ZEROS, ("inf", "0"), ("inf", "-0"), ("inf", "1"), *COMPLEX_NANS],
    "Cos": [
        *SIGNED_ZEROS,
        ("inf", "0"),
        ("0", "inf"),
        ("-0", "-inf"),
        ("0", "110"),
        *COMPLEX_NANS,
    ],
    "Cosh": [
        *SIGNED_ZEROS,
        ("inf", "0"),
        ("-inf", "-0"),
        ("110", "0"),
        ("0", "inf"),
        *COMPLEX_NANS,
    ],
    "Sin": [
        *SIGNED_ZEROS,
        ("0", "inf"),
        ("-0", "-inf"),
        ("0", "110"),
        ("inf", "0"),
        *COMPLEX_NANS,
    ],
    "Sinh": [
        *SIGNED_ZEROS,
        ("inf", "0"),
        ("-inf", "-0"),
        ("-110", "0"),
        ("0", "inf"),
        *COMPLEX_NANS,
    ],
    "Tan": [
        *SIGNED_ZEROS,
        ("0", "inf"),
        ("-0", "inf"),
        ("0", "-inf"),
        ("0", "20"),
        ("inf", "0"),
        *COMPLEX_NANS,
    ],
    "Tanh": [
        *SIGNED_ZEROS,
        ("inf", "0"),
        ("-inf", "0"),
        ("inf", "-0"),
        ("20", "0"),
        ("0", "inf"),
        *COMPLEX_NANS,
    ],
    "ArcCos": [("1", "0"), ("1", "-0"), *COMPLEX_NANS, ("nan", "nan")],
    "ArcSin": [*SIGNED_ZEROS, *COMPLEX_NANS],
    "ArcTan": [*SIGNED_ZEROS, *COMPLEX_NANS],
}
COMPLEX_DOMAINS["Log2"] = COMPLEX_DOMAINS["Log10"] = COMPLEX_DOMAINS["Log"]


@functools.lru_cache(maxsize=None)
def _domain(operator, source):
    """Return the family's input domain (words, integers or word pairs)."""
    if operator == "FromFP8":
        domain = list(range(256))
    elif operator == "ToFP8":
        domain = _fp8_sources(source)
    elif source == "bool_":
        domain = [0, 1]
    elif source in INTEGERS:
        domain = _int_domain(operator, source)
    elif source == "complex64":
        pairs = COMPLEX_DOMAINS.get(operator)
        if pairs is None:
            pairs = [(re, im) for re in COMPLEX_PARTS for im in COMPLEX_PARTS[::2]]
        domain = _complex_pairs(pairs)
    else:
        domain = _float_domain(operator, source)
    return tuple(dict.fromkeys(domain))


# ---------------------------------------------------------------------------
# Layouts and cases.
# ---------------------------------------------------------------------------


def _layout(kernel, spec, work, name):
    """Return the grid, constants and (output, input) index pairs."""
    constants = {}
    if kernel in ("unary_v", "unary_v2"):
        if kernel == "unary_v":
            size = spec["size"]
            grid = [-(-size // work), 1, 1]
            constants["size"] = ("uint32", [size])
        else:
            threads = spec["grid"][0] * spec["grid"][1]
            # A short last tile exercises the tail loop; one element per
            # thread has no tail, so the grid then covers the size exactly.
            size = threads * work - (1 if name == "tail" and work > 1 else 0)
            grid = [spec["grid"][0], spec["grid"][1], 1]
            constants["size"] = ("int64", [size])
        return grid, constants, [(index, index) for index in range(size)]
    shape, strides = spec["shape"], spec["strides"]
    points = itertools.product(*(range(extent) for extent in shape))
    pairs = [
        (output, sum(i * s for i, s in zip(point, strides)))
        for output, point in enumerate(points)
    ]
    grid = [-(-shape[-1] // work), shape[-2], math.prod(shape[:-2])]
    constants["in_shape"] = ("int32", list(shape))
    constants["in_strides"] = ("int64", list(strides))
    constants["ndim"] = ("int32", [len(shape)])
    return grid, constants, pairs


def _extent(entry, name):
    kernel = _kernel(entry)
    _, _, pairs = _layout(kernel, CASES[kernel][name], _work(entry), name)
    return max(read for _, read in pairs) + 1


@functools.lru_cache(maxsize=None)
def _family_offsets(family):
    """Start each case where the previous case of its family stopped reading."""
    offsets, total = {}, 0
    for entry in ENTRIES:
        if _split(entry)[1:] != family:
            continue
        for name in CASES[_kernel(entry)]:
            offsets[entry, name] = total
            total += _extent(entry, name)
    return offsets, total


def _case(entry, name, target="metal"):
    """Return the dispatch, input and expected output of one case."""
    kind, operator, source, output = _split(entry)
    kernel = KINDS[kind][0]
    grid, constants, pairs = _layout(kernel, CASES[kernel][name], _work(entry), name)
    offsets, _ = _family_offsets(_split(entry)[1:])
    values = _fill(
        [read for _, read in pairs], _domain(operator, source), offsets[entry, name]
    )
    written = {out: _model(entry, values[read]) for out, read in pairs}
    assert len(written) == len(pairs), "every output is written once"
    initial, expected = [], []
    for index in range(len(pairs) + GUARD_COUNT):
        if output == "bool_" and target == "metal":
            # One-byte Metal bools hold canonical values only.
            first = 1 - written[index] if index in written else index % 2
        elif index in written:
            first = _different(output, written[index])
        else:
            first = GUARDS[output]
        initial.append(first)
        expected.append(written.get(index, first))
    return {
        "grid": grid,
        "constants": constants,
        "in": values + _decoys(source),
        "initial": initial,
        "expected": expected,
        "pairs": pairs,
        "writtenCount": len(pairs),
    }


# ---------------------------------------------------------------------------
# Translation identity.
# ---------------------------------------------------------------------------

WORKLOADS = {
    "metal": {item.entry_point: item for item in complete_metal.UNARY_METAL_WORKLOADS},
    "opengl": {
        item.entry_point: item for item in complete_opengl.UNARY_OPENGL_WORKLOADS
    },
    "directx": {
        item.entry_point: item for item in complete_directx.UNARY_DIRECTX_WORKLOADS
    },
}
ARTIFACT_CONTRACTS = {
    "metal": complete_metal.UNARY_METAL_CONTRACT_PATH,
    "opengl": complete_opengl.UNARY_OPENGL_CONTRACT_PATH,
    "directx": complete_directx.UNARY_DIRECTX_CONTRACT_PATH,
}
# Operators whose source changed since the complete-family commit: Sigmoid
# and FP8 decoding directly, the complex operators through ``hypot``. The
# uint32 ``Sign`` kernels also materialize the complex ``Sign`` overload.
HYPOT_OPERATORS = ("Abs", "Sign", "Log", "Log2", "Log10", "Sqrt", "Rsqrt")
CHANGED_ENTRIES = tuple(
    entry
    for entry in ENTRIES
    if _split(entry)[1] in ("Sigmoid", "FromFP8")
    or (
        _split(entry)[2] == "complex64"
        and _split(entry)[1] in (*HYPOT_OPERATORS, "ArcCos", "ArcSin", "ArcTan")
    )
    or _split(entry)[1:3] == ("Sign", "uint32")
)


def _config(target, entry):
    """Return the complete-family translation configuration of an entry."""
    workload = WORKLOADS[target][entry]
    if target == "metal":
        return complete_metal._project_config("metal", workload)
    module = complete_opengl if target == "opengl" else complete_directx
    return module._project_config(workload)


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _pinned(target, entry):
    workload = WORKLOADS[target][entry]
    if target == "metal":
        record = workload.generated_artifacts["metal"]
        return record["sha256"], record["sizeBytes"]
    return workload.sha256, workload.size_bytes


def _artifact(target, entry):
    """Return the (sha256, size) the entry's artifact must reproduce."""
    if entry in CHANGED_ENTRIES:
        record = _contract()["currentArtifacts"][target][entry]
        return record["sha256"], record["sizeBytes"]
    return _pinned(target, entry)


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"unary family shard {index} is outside [0, {count})")
    return ENTRIES[index::count]


SELECTED_ENTRIES = _selected_entries()


def _families():
    return list(dict.fromkeys(_split(entry)[1:] for entry in ENTRIES))


def _jsonable(value):
    return list(value) if isinstance(value, tuple) else value


def _domain_digest():
    """Pin every input domain without spelling 183 lists in the contract."""
    payload = {
        "/".join(family): [_jsonable(value) for value in _domain(*family[:2])]
        for family in _families()
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dispatch_count():
    return sum(len(CASES[_kernel(entry)]) for entry in ENTRIES)


def _element_types():
    dtypes = ("bool_", *INTEGERS, *FLOATS, "complex64")
    return {
        target: {dtype: _element_type(target, dtype) for dtype in dtypes}
        for target in TARGETS
    }


def _index_range_assertions():
    return {
        "metal": [],
        **{
            target: [
                {"expression": expression, "minimum": low, "maximum": high}
                for expression, low, high in module.INDEX_RANGE_ASSERTIONS
            ]
            for target, module in (
                ("opengl", complete_opengl),
                ("directx", complete_directx),
            )
        },
    }


def expected_contract(current_artifacts=None):
    """Return the contract this module pins, for review and regeneration."""
    if current_artifacts is None:
        current_artifacts = _contract()["currentArtifacts"]
    return {
        "schemaVersion": 1,
        "kind": "crosstl-mlx-current-tree-unary-family-contract",
        "provenance": {
            "repository": "https://github.com/ml-explore/mlx.git",
            "commit": MLX_COMMIT,
            "source": SOURCE,
            "sourceSha256": SOURCE_SHA256,
            "headers": HEADERS,
        },
        "corpusCensus": {
            "metalUnitCount": 49,
            "entryCount": 17832,
            "discoveryDiagnosticCount": 0,
        },
        "entries": list(ENTRIES),
        # unary.metal and unary.h are byte-identical at the corpus commit; the
        # entries outside ``currentArtifacts`` must reproduce the artifacts
        # their complete-family contracts pin there.
        "artifactContracts": {
            target: {
                "path": ARTIFACT_CONTRACTS[target].relative_to(ROOT).as_posix(),
                "commit": complete_opengl.MLX_COMMIT,
                "sourceSha256": SOURCE_SHA256,
                "changedHeaders": COMPLETE_HEADERS,
            }
            for target in TARGETS
        },
        "currentArtifacts": {
            target: {
                entry: current_artifacts[target][entry] for entry in CHANGED_ENTRIES
            }
            for target in TARGETS
        },
        "validationContract": {
            "workgroupSize": [1, 1, 1],
            "indexRangeAssertions": _index_range_assertions(),
            "cases": CASES,
            "outputGuardCount": GUARD_COUNT,
            "inputDecoyCount": len(_decoys("float32")),
            "elementTypes": _element_types(),
            "inputDomains": {
                "families": len(_families()),
                "valueCount": sum(len(_domain(*family[:2])) for family in _families()),
                "sha256": _domain_digest(),
                "policy": (
                    "every value is defined and exactly modelled; each case "
                    "reads the domain where the previous case of its family "
                    "stopped, so a family's cases cover its whole domain"
                ),
            },
            "operatorModel": {
                "integers": (
                    "C++ semantics after integral promotion: two's complement "
                    "wrap; Abs, Negative and Square inputs that overflow int "
                    "or the signed type are excluded"
                ),
                "bool": "promoted to int, converted back to bool",
                "rounding": "ceil, floor and rint exact with the sign of zero kept",
                "transcendental": (
                    "only at signed zeros, infinities, NaN, poles, domain "
                    "errors and overflow, underflow or saturation thresholds "
                    "valid in every evaluation width"
                ),
                "complex64": (
                    "unary_ops.h, complex.h, cexpf.h and utils.h formulas with "
                    "exact binary32 intermediates"
                ),
                "fp8": (
                    "fp8.h E4M3 bit algorithm: saturating round-to-nearest-even "
                    "encoding, binary16-based decoding of all 256 codes"
                ),
            },
            "compilerValidationRequiredInCi": list(TARGETS),
            "runtimeRequiredInCi": list(TARGETS),
            "upstreamMetallibParityTargets": ["metal"],
            "ciShardCount": CI_SHARD_COUNT,
        },
        "scope": {
            "coveredEntryCount": len(ENTRIES),
            "runtimeCoveredEntryCount": len(ENTRIES),
            "currentArtifactEntryCount": len(CHANGED_ENTRIES),
            "dispatchCountByTarget": {target: _dispatch_count() for target in TARGETS},
            "completeCorpusEntryCount": 17832,
            "upstreamMlxTestSuiteExecuted": False,
            "mlxHostRuntimeRedirectionImplemented": False,
        },
    }


# ---------------------------------------------------------------------------
# Unit tests.
# ---------------------------------------------------------------------------

TYPE_NAMES = {
    "bool_": "bool",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "uint32": "uint32_t",
    "uint64": "uint64_t",
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "int64": "int64_t",
    "float16": "half",
    "float32": "float",
    "bfloat16": "bfloat16_t",
    "complex64": "complex64_t",
}


def test_unary_entries_match_the_upstream_instantiation():
    assert len(ENTRIES) == 877 and len(set(ENTRIES)) == 877
    assert len(_families()) == 183
    for target in TARGETS:
        assert set(WORKLOADS[target]) == set(ENTRIES)
    for entry in ENTRIES:
        kind, operator, source, output = _split(entry)
        workload = WORKLOADS["opengl"][entry]
        assert (workload.shape, workload.operator_type) == (kind, operator)
        assert workload.template_name == KINDS[kind][0], entry
        # ``unary.metal`` spells binary16 ``half`` or ``float16_t``.
        for dtype, name in (
            (source, workload.input_type),
            (output, workload.output_type),
        ):
            spellings = {TYPE_NAMES[dtype]} | (
                {"float16_t"} if dtype == "float16" else set()
            )
            assert name in spellings, (entry, name)
    works = {kind: set() for kind in KINDS}
    for entry in ENTRIES:
        works[_split(entry)[0]].add(_work(entry))
    assert works["vn"] == {2, 4, 8} and works["v2"] == {1, 2, 4, 8}
    assert works["v"] == works["gn1"] == {1} and works["gn4large"] == {4}
    assert len(CHANGED_ENTRIES) == 75


def test_current_mlx_unary_family_contract_is_exact():
    assert _contract() == json.loads(json.dumps(expected_contract()))


def test_unary_family_reuses_the_complete_translation_identities():
    for target, path in ARTIFACT_CONTRACTS.items():
        contract = json.loads(path.read_text(encoding="utf-8"))
        assert contract["commit"] == complete_opengl.MLX_COMMIT
        assert contract["source"] == SOURCE
        assert contract["sourceSha256"] == SOURCE_SHA256
        assert {item["entryPoint"] for item in contract["entries"]} == set(ENTRIES)
        changed = _contract()["currentArtifacts"][target]
        assert set(changed) == set(CHANGED_ENTRIES)
        for entry in CHANGED_ENTRIES:
            record = changed[entry]
            assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
            assert record["sha256"] != _pinned(target, entry)[0]
    assert complete_opengl.MLX_UNARY_SHA256 == SOURCE_SHA256


@pytest.mark.parametrize("target", TARGETS)
def test_unary_family_configs_match_the_complete_contracts(tmp_path, target):
    entry = "gn4large_Sqrtfloat32float32"
    path = tmp_path / "crosstl.toml"
    path.write_text(_config(target, entry) + "\n", encoding="utf-8")
    config = load_project_config(tmp_path, path)
    assert list(config.targets) == [target]
    assert dict(config.entry_points) == {SOURCE: entry}
    assert [item.to_json() for item in config.index_range_assertions] == [
        {"source": SOURCE, **item} for item in _index_range_assertions()[target]
    ]
    assert not config.metadata_preconditions


def test_input_domains_are_defined_and_exactly_modelled():
    for operator, source, output in _families():
        domain = _domain(operator, source)
        assert domain, (operator, source)
        _, total = _family_offsets((operator, source, output))
        assert total >= len(domain), (operator, source, total, len(domain))
        entry = next(e for e in ENTRIES if _split(e)[1:] == (operator, source, output))
        results = set()
        for value in domain:
            if source in INTEGERS:
                low, high = INTEGER_RANGES[source]
                assert low <= value <= high
            results.add(json.dumps(_model(entry, value)))
        if output == "bool_":
            assert results == {"0", "1"}, (operator, results)
        elif operator not in ("Ceil", "Floor") or source in FLOATS:
            assert len(results) > 1, (operator, source)


@pytest.mark.parametrize("family", _families(), ids="/".join)
def test_unary_cases_cover_every_domain_value(family):
    entries = [entry for entry in ENTRIES if _split(entry)[1:] == family]
    seen = set()
    for entry in entries:
        for name in CASES[_kernel(entry)]:
            case = _case(entry, name)
            for out, read in case["pairs"]:
                seen.add(case["in"][read])
            assert len(case["expected"]) == len(case["initial"])
            written = {out for out, _ in case["pairs"]}
            for slot, (first, final) in enumerate(
                zip(case["initial"], case["expected"])
            ):
                assert (first != final) is (slot in written)
    domain = set(_domain(*family[:2]))
    assert domain <= seen, sorted(domain - seen, key=str)[:5]


@pytest.mark.parametrize("kernel", CASES)
def test_unary_layouts_write_each_output_once(kernel):
    works = {_work(entry) for entry in ENTRIES if _kernel(entry) == kernel}
    for work in sorted(works):
        for name, spec in CASES[kernel].items():
            grid, constants, pairs = _layout(kernel, spec, work, name)
            outputs = [out for out, _ in pairs]
            assert outputs == list(range(len(outputs)))
            assert all(extent > 0 for extent in grid)
            if kernel == "unary_v":
                size = constants["size"][1][0]
                assert (grid[0] - 1) * work < size <= grid[0] * work
            if kernel == "unary_g":
                assert len(spec["shape"]) >= 2
    if kernel == "unary_v":
        tails = [
            _layout(kernel, spec, 4, name)[1]["size"][1][0] % 4
            for name, spec in CASES[kernel].items()
        ]
        assert any(tails) and not all(tails)


F32 = {"one": 0x3F800000, "nan": ANY_NAN}


@pytest.mark.parametrize(
    "entry,value,expected",
    [
        ("v_Absfloat32float32", 0x80000000, 0x00000000),
        ("v_Absfloat16float16", 0xFE00, ANY_NAN),
        ("v_Negativefloat16float16", 0x0000, 0x8000),
        ("v_Ceilfloat32float32", 0xBF000000, 0x80000000),
        ("v_Floorfloat32float32", 0xBF000000, 0xBF800000),
        ("v_Roundfloat32float32", 0x40200000, 0x40000000),
        ("v_Roundfloat32float32", 0xBF000000, 0x80000000),
        ("v_Roundbfloat16bfloat16", 0x42FF, 0x4300),
        ("v_Roundfloat16float16", 0x63FF, 0x6400),
        ("v_Signfloat32float32", 0x7FC00000, 0x00000000),
        ("v_Signfloat32float32", 0x80000000, 0x00000000),
        ("v_Signbfloat16bfloat16", 0xFF80, 0xBF80),
        ("v_Squarefloat16float16", 0x7BFF, 0x7C00),
        ("v_Sqrtfloat32float32", 0x80000000, 0x80000000),
        ("v_Sqrtfloat32float32", 0x41100000, 0x40400000),
        ("v_Sqrtfloat32float32", 0xBF800000, ANY_NAN),
        ("v_Rsqrtfloat32float32", 0x80000000, 0xFF800000),
        ("v_Rsqrtfloat16float16", 0x4400, 0x3800),
        ("v_ArcCosfloat32float32", 0x3F800000, 0x00000000),
        ("v_ArcTanhbfloat16bfloat16", 0xBF80, 0xFF80),
        ("v_ArcCoshfloat32float32", 0x3F000000, ANY_NAN),
        ("v_Expfloat32float32", 0xFF800000, 0x00000000),
        ("v_Expfloat16float16", 0xCD00, 0x0000),
        ("v_Expm1float32float32", 0xFF800000, 0xBF800000),
        ("v_Expm1float32float32", 0x80000000, 0x80000000),
        # ``Limits<float>::max`` is +inf, which log1p returns for +inf.
        ("v_Log1pfloat32float32", 0x7F800000, 0x7F800000),
        ("v_Log1pfloat16float16", 0x7C00, 0x7C00),
        ("v_Log1pfloat32float32", 0xFF800000, ANY_NAN),
        ("v_Log1pfloat32float32", 0xBF800000, 0xFF800000),
        ("v_Log2float16float16", 0x8000, 0xFC00),
        ("v_Sigmoidfloat32float32", 0x80000000, 0x3F000000),
        ("v_Sigmoidbfloat16bfloat16", 0xC2DC, 0x0000),
        ("v_Sigmoidfloat16float16", 0x4D00, 0x3C00),
        ("v_Erffloat16float16", 0xC900, 0xBC00),
        ("v_ErfInvfloat32float32", 0x3F800000, 0x7F800000),
        ("v_Tanhfloat32float32", 0xC1A00000, 0xBF800000),
        ("v_Coshfloat16float16", 0xCA00, 0x7C00),
        ("v_Absint8int8", -127, 127),
        ("v_Negativeint8int8", -128, -128),
        ("v_Negativeuint16uint16", 1, 0xFFFF),
        ("v_Signint16int16", -5, -1),
        ("v_Signuint32uint32", 7, 1),
        ("v_Squareuint8uint8", 255, 1),
        ("v_Squareint16int16", -32768, 0),
        ("v_Squareuint32uint32", 0xFFFFFFFF, 1),
        ("v_BitwiseInvertuint8uint8", 5, 250),
        ("v_BitwiseInvertint32int32", 0, -1),
        ("v_Negativebool_bool_", 1, 1),
        ("v_LogicalNotbool_bool_", 0, 1),
        ("v_Abscomplex64complex64", (0x40400000, 0xC0800000), (0x40A00000, 0)),
        ("v_Abscomplex64complex64", (0x7F800000, 0x7FC00000), (0x7F800000, 0)),
        (
            "v_Signcomplex64complex64",
            (0x80000000, 0x80000000),
            (0x80000000, 0x80000000),
        ),
        ("v_Signcomplex64complex64", (0x7F800000, 0x3F800000), (ANY_NAN, 0)),
        (
            "v_Sqrtcomplex64complex64",
            (0xC0400000, 0xC0800000),
            (0x3F800000, 0xC0000000),
        ),
        ("v_Sqrtcomplex64complex64", (0x80000000, 0x80000000), (0, 0)),
        ("v_Rsqrtcomplex64complex64", (0x40800000, 0), (0x3F000000, 0x80000000)),
        ("v_Expcomplex64complex64", (0xFF800000, 0x7F800000), (0, 0)),
        ("v_Expcomplex64complex64", (0x7F800000, 0x7FC00000), (0x7F800000, ANY_NAN)),
        ("v_Logcomplex64complex64", (0x3F800000, 0x80000000), (0, 0x80000000)),
        (
            "v_Log1pcomplex64complex64",
            (0x80000000, 0x80000000),
            (0x80000000, 0x80000000),
        ),
        ("v_Tancomplex64complex64", (0x80000000, 0x7F800000), (0, 0x3F800000)),
        ("v_Tanhcomplex64complex64", (0xFF800000, 0), (0xBF800000, 0)),
        ("v_Coscomplex64complex64", (0, 0x7F800000), (0x7F800000, ANY_NAN)),
        ("v_ArcCoscomplex64complex64", (0x3F800000, 0), (0, 0x80000000)),
        (
            "v_ArcSincomplex64complex64",
            (0x80000000, 0x80000000),
            (0x80000000, 0x80000000),
        ),
        ("v_ArcTancomplex64complex64", (0x80000000, 0), (0, 0)),
        ("v_Squarecomplex64complex64", (0x80000000, 0), (0, 0x80000000)),
        ("v_Realcomplex64float32", (0x7FC00000, 0x3F800000), ANY_NAN),
        ("v_Imagcomplex64float32", (0x7FC00000, 0xBF800000), 0xBF800000),
        # ``fp8_max`` is 543 << 21 = 448.0: every magnitude from 448 up,
        # infinities and NaN included, saturates to 0x7E.
        ("v_ToFP8float32uint8", 0x43E80000, 0x7E),
        ("v_ToFP8float32uint8", 0x43EC0000, 0x7E),
        ("v_ToFP8float32uint8", 0x43DFFFFF, 0x7E),
        # Midpoints between 384, 416 and 448 round to the even code.
        ("v_ToFP8float32uint8", 0x43C80000, 0x7C),
        ("v_ToFP8float32uint8", 0x43D80000, 0x7E),
        ("v_ToFP8float32uint8", 0xC3F00000, 0xFE),
        ("v_ToFP8float32uint8", 0x7FC00000, 0x7E),
        ("v_ToFP8float16uint8", 0x7E00, 0x7E),
        # The smallest E4M3 subnormal is 2**-9; 2**-10 and 1.5 * 2**-9 are
        # ties that round to the even code.
        ("v_ToFP8bfloat16uint8", 0x3B00, 0x01),
        ("v_ToFP8bfloat16uint8", 0x3B40, 0x02),
        ("v_ToFP8float32uint8", 0x3A800000, 0x00),
        ("v_FromFP8uint8float32", 0x7E, 0x43E00000),
        ("v_FromFP8uint8float16", 0x01, 0x1800),
        ("v_FromFP8uint8bfloat16", 0xFF, ANY_NAN),
        ("v_FromFP8uint8float32", 0x80, 0x80000000),
    ],
)
def test_operator_models_pin_mlx_semantics(entry, value, expected):
    """Pin the IEEE, C99, C++ and MLX helper boundaries the gate checks."""
    assert _model(entry, value) == expected


@pytest.mark.parametrize(
    "entry,value",
    [
        ("v_Sinfloat32float32", 0x3F800000),
        ("v_Expfloat32float32", 0x3F800000),
        ("v_Expfloat16float16", 0x4800),
        ("v_Sqrtfloat32float32", 0x40000000),
        ("v_Logfloat32float32", 0x40000000),
        ("v_ArcSinfloat32float32", 0x3F800000),
        ("v_Tanhfloat32float32", 0x41200000),
        ("v_Sigmoidfloat32float32", 0xC1A00000),
        ("v_Log1pfloat16float16", 0x0400),
        ("v_Signcomplex64complex64", (0x40400000, 0x40800000)),
        ("v_Logcomplex64complex64", (0xBF800000, 0)),
        ("v_Abscomplex64complex64", (0x3F800000, 0x3F800000)),
        ("v_Expcomplex64complex64", (0x3F800000, 0x3F800000)),
        ("v_Rsqrtcomplex64complex64", (0x40400000, 0x40800000)),
    ],
)
def test_models_refuse_results_that_are_not_exact(entry, value):
    with pytest.raises(AssertionError):
        _model(entry, value)


@pytest.mark.parametrize("negative", [False, True])
def test_fp8_encoding_round_trips_every_code(negative):
    for code in range(127):
        byte = code | (0x80 if negative else 0)
        word = _from_fp8("float32", byte)
        assert _to_fp8("float32", word) == byte
        if code < 126:
            half = _f(
                "float16",
                (
                    _float_value("float32", word)[2]
                    + _float_value("float32", _from_fp8("float32", code + 1))[2]
                )
                / 2
                * (-1 if negative else 1),
            )
            # Midpoints round to the even code.
            assert _to_fp8("float16", half) == (code + (code & 1)) | (
                0x80 if negative else 0
            )


def test_unary_shards_partition_the_family(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert {len(shard) for shard in shards} == {438, 439}
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


@pytest.mark.parametrize("fault", ["value", "missing", "guard", "nan", "zero-sign"])
def test_unary_readback_rejects_corruption(fault):
    entry = "v_Sqrtfloat16float16"
    case = _case(entry, "tail")
    wanted, nans = _metal_wanted("float16", case["expected"])
    assert nans
    actual = list(wanted)
    finite = next(
        index
        for index, word in enumerate(wanted)
        if index < case["writtenCount"] and index not in nans and word & 0x7FFF
    )
    if fault == "value":
        actual[finite] ^= 0x0400
    elif fault == "missing":
        actual[finite] = case["initial"][finite]
    elif fault == "guard":
        actual[-1] ^= 1
    elif fault == "nan":
        actual[nans[0]] = 0x7C00
    else:
        zero = next(i for i, w in enumerate(wanted) if w == 0x8000)
        actual[zero] = 0
    with pytest.raises(AssertionError):
        _check(actual, wanted, nans, "float16")
    _check(wanted, wanted, nans, "float16")


# ---------------------------------------------------------------------------
# Native execution.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native unary family gate")
    assert root_value and target in TARGETS
    for name in {
        "directx": ["dxc"],
        "opengl": ["glslangValidator", "spirv-val"],
        "metal": ["xcrun"],
    }[target]:
        assert shutil.which(name), f"Required tool is missing: {name}"
    root = Path(root_value).resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert revision == MLX_COMMIT
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    for path, digest in ((SOURCE, SOURCE_SHA256), *HEADERS.items()):
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("unary-family-metal-reference")
    runner = work / "readback"
    _run(
        [
            "xcrun",
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            runner,
        ],
        work,
        "build-runner",
    )
    return runner, _metal_library(
        root / SOURCE, work / "upstream.metallib", root, upstream=True
    )


def _upstream_metal(work, reference, entry, case, name):
    runner, library = reference
    kind, _, source, output = _split(entry)
    kernel = KINDS[kind][0]
    payloads = {
        0: _metal_bytes(source, case["in"]),
        1: _metal_bytes(output, case["initial"]),
    }
    for parameter, (dtype_name, values) in case["constants"].items():
        payloads[BUFFER_INDICES[kernel][parameter]] = struct.pack(
            f"<{len(values)}{CONSTANT_FORMATS[dtype_name]}", *values
        )
    files = []
    for index in range(max(payloads) + 1):
        path = work / f"{name}-upstream-input-{index}.bin"
        path.write_bytes(payloads.get(index, b"\0\0\0\0"))
        files.append(str(path))
    request = work / f"{name}-upstream-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": case["grid"],
                "workgroupSize": [1, 1, 1],
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    readback = work / f"{name}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, readback], work, f"{name}-upstream")
    )
    for index, path in enumerate(files):
        if index != 1:
            assert (readback / f"buffer-{index}.bin").read_bytes() == Path(
                path
            ).read_bytes()
    return device, (readback / "buffer-1.bin").read_bytes()


def _dispatch(descriptor, package, target, entry, case):
    _, _, source, output = _split(entry)
    concrete, nans = _concrete(output, case["expected"])
    wanted = _runtime_value(target, output, concrete)["values"]
    data = {
        "in": _runtime_value(target, source, case["in"]),
        "out": _runtime_value(target, output, case["initial"]),
    }
    for parameter, (dtype_name, values) in case["constants"].items():
        data[parameter] = {
            "dtype": dtype_name,
            "shape": [len(values)],
            "values": values,
        }
    inputs, observed, output_name = {}, [], None
    # DirectX constant members carry the sanitized entry name as a prefix.
    prefix = re.sub(r"_+", "_", entry + "_")
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        layout = binding["scalarLayout"]
        name = layout.get("memberName", binding["name"]).removeprefix(prefix)
        name = {"in_": "in", "out_": "out"}.get(name, name)
        value = data[name]
        storage = layout.get("storageEncoding")
        if storage is None:
            assert layout["elementType"] == value["dtype"], (name, layout)
        else:
            assert layout["elementType"] == "uint16", (name, layout)
            assert storage == {
                "encoding": FLOAT16_BITS,
                "logicalElementType": value["dtype"],
            }
            assert value["encoding"] == FLOAT16_BITS
        inputs[binding["name"]] = value
        if name == "out":
            output_name = binding["name"]
        else:
            observed.append(binding["name"])
    assert output_name is not None and len(inputs) == len(data), sorted(inputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output_name: dict(data["out"])},
        {"workgroupCount": case["grid"], "workgroupSize": [1, 1, 1]},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output_name, wanted, nans, len(observed)


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_current_mlx_unary_family_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    output = _output(entry)
    sha256, size_bytes = _artifact(target, entry)
    bits = _nan_bits(target, output) if output in FLOATS else 32
    metal_bits = _nan_bits("metal", output) if output in FLOATS else 32
    with tempfile.TemporaryDirectory(prefix=".current-unary-", dir=root) as directory:
        work = Path(directory)
        try:
            config = work / "crosstl.toml"
            config.write_text(_config(target, entry) + "\n", encoding="utf-8")
            translation = translate_project(
                load_project_config(root, config),
                targets=(target,),
                output_dir=f"{work.name}/out",
                format_output=False,
                validate=True,
                run_toolchains=False,
            )
            report = translation.to_json()
            translation.write_json(work / "report.json")
            validation = validate_project_report(work / "report.json")
            assert validation["success"] is True, validation
            assert report["diagnostics"] == []
            assert report["summary"]["translatedCount"] == 1
            assert report["summary"]["failedCount"] == 0
            (artifact,) = report["artifacts"]
            assert artifact["sourceHash"] == {
                "algorithm": "sha256",
                "value": SOURCE_SHA256,
            }
            check_generated_artifact(artifact, sha256=sha256, size_bytes=size_bytes)
            (execution,) = artifact["execution"]["entryPoints"]
            assert execution["workgroupSize"] == [1, 1, 1]
            generated = root / artifact["path"]
            assert "unsupported" not in generated.read_text(encoding="utf-8")
            if target == "metal":
                _metal_library(generated, work / "translated.metallib", root)
            else:
                _validate_generated_artifact(generated, work, target)
            descriptor, package = _prepare_native_package(translation, work)
            executor = _executor(target)
            evidence = []
            try:
                for name in CASES[_kernel(entry)]:
                    case = _case(entry, name, target)
                    request, output_name, wanted, nans, observed = _dispatch(
                        descriptor, package, target, entry, case
                    )
                    availability = executor.is_available(request)
                    assert availability.available, availability.reason
                    result = executor.run(request)
                    if result.status != "ok":
                        (work / f"{name}-result.json").write_text(
                            json.dumps(
                                {"status": result.status, "details": result.details},
                                indent=2,
                                default=str,
                            ),
                            encoding="utf-8",
                        )
                    assert result.status == "ok"
                    actual = result.outputs[output_name]["values"]
                    if output == "bool_" and target == "metal":
                        actual = [int(value) for value in actual]
                        wanted = [int(value) for value in wanted]
                    _check(actual, wanted, nans, bits)
                    observations = result.details["inputObservations"]
                    assert len(observations) == observed
                    assert all(item["unchanged"] for item in observations.values())
                    upstream = None
                    if target == "metal":
                        device, raw = _upstream_metal(
                            work, metal_reference, entry, case, name
                        )
                        words, metal_nans = _metal_wanted(output, case["expected"])
                        _check(_metal_words(output, raw), words, metal_nans, metal_bits)
                        upstream = {"device": device, "matched": True}
                    evidence.append(
                        {
                            "case": name,
                            "grid": case["grid"],
                            "constants": case["constants"],
                            "writtenCount": case["writtenCount"],
                            "outputGuardCount": GUARD_COUNT,
                            "anyNanSlots": len(nans),
                            "observedInputCount": observed,
                            "upstream": upstream,
                        }
                    )
            finally:
                close = getattr(executor.runtime_adapter.runtime, "close", None)
                if callable(close):
                    close()
        except BaseException:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)
            raise
    (tmp_path / "parity.json").write_text(
        json.dumps(
            {
                "commit": MLX_COMMIT,
                "entry": entry,
                "target": target,
                "artifactSha256": sha256,
                "cases": evidence,
                "upstreamMlxTestSuiteExecuted": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
