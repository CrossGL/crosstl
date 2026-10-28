"""Prove every current MLX binary kernel natively on Metal, OpenGL and DirectX.

``binary.metal`` instantiates 4,122 kernels: fifteen base layouts and three
work-per-thread variants for twenty-four operators over MLX's dtypes. Each
entry translates with the configuration of its complete-family translation
contract and must reproduce the artifact that contract pins; the fifteen
complex power entries, whose operator changed upstream, reproduce the
current artifacts this gate's contract pins. Every entry then executes two
layout cases on Metal, Mesa EGL and Direct3D 12 WARP.

Operands come from per-operator domains whose results are exactly defined:
every finite result is representable without rounding and every special
value follows IEEE-754 or the MLX operator, and no pair is undefined in C++.
Each output element is compared bit for bit with an independent model of the
operator, NaN results accept any NaN, and every guard after the output must
stay unchanged. Metal also runs the unchanged upstream kernel, which must
produce the same bytes. Rounding, subnormal and accuracy behaviour of the
arithmetic operators is covered by the per-operator gates.
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
    test_binary_complete_directx as complete_directx,
)
from demos.integrations.mlx.tests.kernels import (
    test_binary_complete_metal_roundtrip as complete_metal,
)
from demos.integrations.mlx.tests.kernels import (
    test_binary_complete_opengl as complete_opengl,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.kernels.test_current_copy_family import (
    CANONICAL_NANS,
    EDGES,
    FLOAT_LAYOUTS,
    INTEGER_RANGES,
    _float_value,
    _is_nan,
    _nan_bits,
    _round_float,
    _wrap,
)
from demos.integrations.mlx.tests.kernels.test_current_ternary import (
    GUARDS,
    ITEM_SIZES,
    METAL_FORMATS,
    _element_type,
    _metal_bytes,
    _runtime_value,
)
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/binary.metal"
SOURCE_SHA256 = "4dadb612a9b768f9d51b3b394b32fc0129d361a55b35d545b3c014c87e00897e"
HEADERS = {
    "mlx/backend/metal/kernels/binary.h": (
        "9a367967a48907ac6792be685bab4cc062f3ad916cccacb441c627d87a4b7898"
    ),
    "mlx/backend/metal/kernels/binary_ops.h": (
        "6dbe789faafbf7259f411c7fab7c0fcdd192df47a78f8f0bd408228f7a30e5c7"
    ),
    "mlx/backend/metal/kernels/complex.h": (
        "16e8a815b2cbdb6070e0824e64fe33fccb6e918f1b84ea5c792bd89d33e57bf1"
    ),
    "mlx/backend/metal/kernels/utils.h": (
        "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf"
    ),
}
# ``binary_ops.h`` at the complete-family commit; only complex ``Power``
# differs (``log(hypot(re, im))`` replaced ``0.5 * log(re * re + im * im)``).
COMPLETE_OPS_HEADER_SHA256 = (
    "2dd13c2496f5d6f0856e4ca99db2ceb5c6d7dc0c344b9ba8e41ef3b7d7ed8a97"
)
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_BINARY_FAMILY"
SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_BINARY_FAMILY_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_BINARY_FAMILY_SHARD_COUNT"
CI_SHARD_COUNT = 12
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/binary.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
INTEGERS = ("uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64")
FLOATS = ("float16", "float32", "bfloat16")
# Instantiation orders of ``instantiate_binary_types`` and ``_types_bool``.
ALL_TYPES = ("bool_", *INTEGERS, "complex64", *FLOATS)
BOOL_OUTPUT_TYPES = ("bool_", *INTEGERS, *FLOATS, "complex64")
# ``instantiate_binary_base`` only: the work-per-thread variants skip them.
BASE_ONLY = ("uint64", "int64", "complex64")
OPERATORS = (
    ("Add", ALL_TYPES),
    ("Divide", ALL_TYPES),
    ("Equal", BOOL_OUTPUT_TYPES),
    ("Greater", BOOL_OUTPUT_TYPES),
    ("GreaterEqual", BOOL_OUTPUT_TYPES),
    ("Less", BOOL_OUTPUT_TYPES),
    ("LessEqual", BOOL_OUTPUT_TYPES),
    ("NotEqual", BOOL_OUTPUT_TYPES),
    ("LogAddExp", (*FLOATS, "complex64")),
    ("Maximum", ALL_TYPES),
    ("Minimum", ALL_TYPES),
    ("Multiply", ALL_TYPES),
    ("Subtract", ALL_TYPES),
    ("Power", ALL_TYPES),
    ("Remainder", ALL_TYPES),
    ("ArcTan2", FLOATS),
    ("NaNEqual", (*FLOATS, "complex64")),
    ("LogicalOr", ("bool_",)),
    ("LogicalAnd", ("bool_",)),
    ("BitwiseAnd", (*INTEGERS, "bool_")),
    ("BitwiseOr", (*INTEGERS, "bool_")),
    ("BitwiseXor", (*INTEGERS, "bool_")),
    ("LeftShift", INTEGERS),
    ("RightShift", INTEGERS),
)
COMPARISONS = ("Equal", "Greater", "GreaterEqual", "Less", "LessEqual", "NotEqual")
BOOL_RESULTS = (*COMPARISONS, "NaNEqual")
BASE_KINDS = (
    "ss",
    "sv",
    "vs",
    "vv",
    "sv2",
    "vs2",
    "vv2",
    "gn2",
    "gn4large",
    "g1",
    "g1large",
    "g2",
    "g2large",
    "g3",
    "g3large",
)
WORK_KINDS = ("svn", "vsn", "vvn")
ENTRIES = tuple(
    f"{kind}_{operator}{dtype}"
    for operator, dtypes in OPERATORS
    for dtype in dtypes
    for kind in BASE_KINDS + (() if dtype in BASE_ONLY else WORK_KINDS)
)
# kind: (kernel template, elements per thread or None for WorkPerThread<T>::n).
KINDS = {
    "ss": ("binary_ss", 1),
    "sv": ("binary_sv", 1),
    "vs": ("binary_vs", 1),
    "vv": ("binary_vv", 1),
    "svn": ("binary_sv", None),
    "vsn": ("binary_vs", None),
    "vvn": ("binary_vv", None),
    "sv2": ("binary_sv2", None),
    "vs2": ("binary_vs2", None),
    "vv2": ("binary_vv2", None),
    "g1": ("binary_g_nd1", 1),
    "g1large": ("binary_g_nd1", 1),
    "g2": ("binary_g_nd2", 1),
    "g2large": ("binary_g_nd2", 1),
    "g3": ("binary_g_nd3", 1),
    "g3large": ("binary_g_nd3", 1),
    "gn2": ("binary_g", 2),
    "gn4large": ("binary_g", 4),
}
# Flat kernels take element counts or (x, y) grids; strided kernels take a
# shape (outermost first) and per-operand strides for ``a`` and ``b``.
_FLAT = {"tail": {"size": 37}, "exact": {"size": 64}}
_GRID = {"tail": {"grid": [3, 2]}, "wide": {"grid": [4, 3]}}
CASES = {
    # MLX dispatches one thread; more threads must each write their index.
    "binary_ss": {"single": {"grid": [1, 1, 1]}, "spread": {"grid": [5, 1, 1]}},
    "binary_sv": _FLAT,
    "binary_vs": _FLAT,
    "binary_vv": _FLAT,
    "binary_sv2": _GRID,
    "binary_vs2": _GRID,
    "binary_vv2": _GRID,
    "binary_g_nd1": {
        "strided": {"shape": [17], "strides": [[3], [1]]},
        "broadcast": {"shape": [9], "strides": [[0], [2]]},
    },
    "binary_g_nd2": {
        "transpose": {"shape": [3, 5], "strides": [[1, 3], [5, 1]]},
        "broadcast": {"shape": [4, 2], "strides": [[0, 1], [2, 0]]},
    },
    "binary_g_nd3": {
        "permute": {"shape": [2, 2, 3], "strides": [[1, 2, 4], [6, 3, 1]]},
        "broadcast": {"shape": [3, 1, 4], "strides": [[4, 0, 1], [0, 0, 1]]},
    },
    # The innermost extents (5 and 3) end inside a two- and four-wide tile.
    "binary_g": {
        "nd4": {"shape": [2, 3, 2, 5], "strides": [[1, 2, 30, 6], [30, 10, 5, 1]]},
        "nd5-broadcast": {
            "shape": [2, 2, 1, 3, 3],
            "strides": [[0, 9, 0, 3, 1], [9, 0, 0, 1, 3]],
        },
    },
}
# Metal ``[[buffer(n)]]`` slots of each kernel's constants.
BUFFER_INDICES = {
    "binary_g": {"shape": 3, "a_strides": 4, "b_strides": 5, "ndim": 6},
    "binary_g_nd1": {"a_stride": 3, "b_stride": 4},
    "binary_g_nd2": {"a_strides": 3, "b_strides": 4},
    "binary_g_nd3": {"a_strides": 3, "b_strides": 4},
    **{
        kernel: {"size": 3}
        for kernel in (
            "binary_sv",
            "binary_vs",
            "binary_vv",
            "binary_sv2",
            "binary_vs2",
            "binary_vv2",
        )
    },
    "binary_ss": {},
}
CONSTANT_FORMATS = {"uint32": "I", "int32": "i", "int64": "q"}
GUARD_COUNT = 16
DECOY_COUNT = 4
ANY_NAN = "any-nan"
# Scalar operands read one value per dispatch; each kind starts elsewhere.
KIND_SALTS = {kind: index * 7 for index, kind in enumerate(KINDS)}


def _split(entry):
    """Return the kind, operator and input dtype of an entry."""
    kind, rest = entry.split("_", 1)
    for operator, dtypes in OPERATORS:
        if rest.startswith(operator) and rest[len(operator) :] in dtypes:
            return kind, operator, rest[len(operator) :]
    raise AssertionError(entry)


def _output(operator, dtype):
    return "bool_" if operator in BOOL_RESULTS else dtype


def _kernel(entry):
    return KINDS[_split(entry)[0]][0]


def _work(entry):
    """Return the elements one thread of the entry computes."""
    kind, _, dtype = _split(entry)
    explicit = KINDS[kind][1]
    return explicit if explicit is not None else 8 // ITEM_SIZES[dtype]


# ---------------------------------------------------------------------------
# Exact IEEE-754 values: ("nan",), ("inf", sign) or ("finite", sign, |x|).
# ---------------------------------------------------------------------------

NAN = ("nan",)


def _value(dtype, word):
    kind, sign, magnitude = _float_value(dtype, word)
    if kind == "nan":
        return NAN
    if kind == "inf":
        return ("inf", sign)
    return ("finite", sign, magnitude)


def _normal(dtype, word):
    exponent_bits, mantissa_bits = FLOAT_LAYOUTS[dtype]
    exponent = word >> mantissa_bits & ((1 << exponent_bits) - 1)
    return exponent != 0 or word & ((1 << mantissa_bits) - 1) == 0


def _word(dtype, value):
    """Encode a value the format holds exactly; NaN becomes ``ANY_NAN``."""
    if value == NAN:
        return ANY_NAN
    exponent_bits, mantissa_bits = FLOAT_LAYOUTS[dtype]
    sign = value[1] << (exponent_bits + mantissa_bits)
    if value[0] == "inf":
        return sign | ((1 << exponent_bits) - 1) << mantissa_bits
    bits, overflowed = _round_float(dtype, value[1], value[2])
    assert not overflowed, (dtype, value)
    assert _float_value(dtype, bits)[2] == value[2], f"{dtype} rounds {value}"
    assert _normal(dtype, bits), f"{dtype} {value} is subnormal"
    return bits


def _number(value):
    """Return a comparable number for a non-NaN value."""
    if value[0] == "inf":
        return -math.inf if value[1] else math.inf
    return -value[2] if value[1] else value[2]


def _negate(value):
    return value if value == NAN else (value[0], 1 - value[1], *value[2:])


def _is_zero(value):
    return value[0] == "finite" and value[2] == 0


def _add(x, y):
    if NAN in (x, y):
        return NAN
    if x[0] == "inf" or y[0] == "inf":
        if x[0] == y[0] == "inf" and x[1] != y[1]:
            return NAN
        return x if x[0] == "inf" else y
    total = _number(x) + _number(y)
    if total == 0:
        # Round to nearest gives +0 unless both addends are -0.
        return ("finite", int(x[1] and y[1]), Fraction(0))
    return ("finite", int(total < 0), abs(total))


def _multiply(x, y):
    if NAN in (x, y):
        return NAN
    sign = x[1] ^ y[1]
    if x[0] == "inf" or y[0] == "inf":
        return NAN if _is_zero(x) or _is_zero(y) else ("inf", sign)
    return ("finite", sign, x[2] * y[2])


def _divide(x, y):
    if NAN in (x, y):
        return NAN
    sign = x[1] ^ y[1]
    if x[0] == "inf":
        return NAN if y[0] == "inf" else ("inf", sign)
    if y[0] == "inf":
        return ("finite", sign, Fraction(0))
    if y[2] == 0:
        return NAN if x[2] == 0 else ("inf", sign)
    return ("finite", sign, x[2] / y[2])


def _fmod(x, y):
    """IEEE ``fmod``: exact, with the sign of ``x``."""
    if NAN in (x, y) or x[0] == "inf" or _is_zero(y):
        return NAN
    if y[0] == "inf":
        return x
    quotient = int(x[2] / y[2])
    return ("finite", x[1], x[2] - quotient * y[2])


def _half_fmod(x, y):
    """Metal's binary16 ``fmod``: ``x - trunc(x / y) * y`` in binary32.

    Unlike IEEE ``fmod`` it returns NaN for infinite divisors, and an exact
    zero difference is +0 even when ``x`` is negative.
    """
    if NAN in (x, y) or "inf" in (x[0], y[0]) or _is_zero(y):
        return NAN
    quotient = _exactly("float32", _divide(x, y))
    integral = ("finite", quotient[1], Fraction(int(quotient[2])))
    product = _exactly("float32", _multiply(integral, y))
    return _exactly("float32", _add(x, _negate(product)))


def _less(x, y):
    return NAN not in (x, y) and _number(x) < _number(y)


def _equal(x, y):
    return NAN not in (x, y) and _number(x) == _number(y)


def _exactly(dtype, value):
    """Require that an intermediate result needs no rounding."""
    if value != NAN and value[0] == "finite":
        _word(dtype, value)
    return value


# ---------------------------------------------------------------------------
# Operator models.
# ---------------------------------------------------------------------------


def _float_model(operator, dtype, a, b):
    x, y = _value(dtype, a), _value(dtype, b)
    if operator == "Add":
        return _word(dtype, _add(x, y))
    if operator == "Subtract":
        return _word(dtype, _add(x, _negate(y)))
    if operator == "Multiply":
        return _word(dtype, _multiply(x, y))
    if operator == "Divide":
        return _word(dtype, _divide(x, y))
    if operator == "Remainder":
        result = _half_fmod(x, y) if dtype == "float16" else _fmod(x, y)
        nonzero = result == NAN or not _is_zero(result)
        if nonzero and _less(result, ("finite", 0, 0)) != _less(y, ("finite", 0, 0)):
            result = _add(result, y)
        return _word(dtype, result)
    if operator in ("Maximum", "Minimum"):
        if x == NAN:
            return ANY_NAN
        chosen = _less(y, x) if operator == "Maximum" else _less(x, y)
        return a if chosen else (ANY_NAN if y == NAN else b)
    if operator in COMPARISONS:
        return int(_compare(operator, x, y))
    if operator == "NaNEqual":
        return int(_equal(x, y) or x == y == NAN)
    if operator == "LogAddExp":
        if NAN in (x, y):
            return ANY_NAN
        larger, smaller = (x, y) if _less(y, x) else (y, x)
        if smaller == ("inf", 1) or larger == ("inf", 0):
            return _word(dtype, larger)
        # exp(-32) is far below half an ulp of any magnitude of at least 16.
        assert larger[0] == smaller[0] == "finite"
        assert _number(larger) - _number(smaller) >= 32 and larger[2] >= 16
        return _word(dtype, larger)
    if operator == "ArcTan2":
        # atan2(y, x) with y = a and x = b.
        if NAN in (x, y):
            return ANY_NAN
        if y == ("inf", 0) and x[0] == "finite":
            return _word(dtype, ("finite", x[1], Fraction(0)))
        assert _is_zero(x) and y[1] == 0, (a, b)
        return a
    if operator == "Power":
        return _word(dtype, _power(x, y))
    raise AssertionError(operator)


def _compare(operator, x, y):
    return {
        "Equal": lambda: _equal(x, y),
        "NotEqual": lambda: not _equal(x, y),
        "Greater": lambda: _less(y, x),
        "GreaterEqual": lambda: _less(y, x) or _equal(x, y),
        "Less": lambda: _less(x, y),
        "LessEqual": lambda: _less(x, y) or _equal(x, y),
    }[operator]()


def _power(x, y):
    """C99 ``pow`` for the exactly defined cases the domains use."""
    one = ("finite", 0, Fraction(1))
    if _is_zero(y) or x == one:
        return one
    if NAN in (x, y):
        return NAN
    if y == one:
        return x
    if y[0] == "inf":
        if x[0] == "finite" and x[2] == 1:
            return one
        larger = x[0] == "inf" or x[2] > 1
        return ("inf", 0) if larger != bool(y[1]) else ("finite", 0, Fraction(0))
    integral = y[2].denominator == 1
    odd = integral and y[2].numerator % 2 == 1
    sign = x[1] if odd else 0
    if _is_zero(x) or x[0] == "inf":
        infinite = (x[0] == "inf") != bool(y[1])
        return ("inf", sign) if infinite else ("finite", sign, Fraction(0))
    assert x[2] == 1, (x, y)
    return ("finite", sign, Fraction(1)) if integral else NAN


def _signed(dtype):
    return INTEGER_RANGES[dtype][0] < 0


def _promoted(dtype):
    """C++ promotes the narrow types to ``int`` before arithmetic."""
    return ITEM_SIZES[dtype] < 4


def _int_defined(operator, dtype, x, y):
    """Return whether C++ defines the operator for these integers."""
    low, high = INTEGER_RANGES[dtype]
    bits = ITEM_SIZES[dtype] * 8
    if _promoted(dtype):
        low, high, bits = -(2**31), 2**31 - 1, 32

    def fits(value):
        # Unsigned arithmetic wraps; signed and promoted overflow is undefined.
        return low <= value <= high if _signed(dtype) or _promoted(dtype) else True

    if operator in ("Add", "Subtract", "Multiply"):
        return fits({"Add": x + y, "Subtract": x - y, "Multiply": x * y}[operator])
    if operator in ("Divide", "Remainder"):
        # Only MIN / -1 overflows; the remainder is undefined with it.
        return y != 0 and fits(_truncated_quotient(x, y))
    if operator == "LeftShift":
        if not 0 <= y < bits or x < 0:
            return False
        # A signed shift must stay representable in the unsigned type.
        return x << y < 2**bits if _signed(dtype) or _promoted(dtype) else True
    if operator == "RightShift":
        return 0 <= y < bits
    if operator == "Power":
        if y < 0:
            return True
        result, base, exponent = 1, x, y
        while exponent:
            if exponent & 1:
                if not fits(result * base):
                    return False
                result = _wrap(dtype, result * base)
            exponent >>= 1
            if not fits(base * base):
                return False
            base = _wrap(dtype, base * base)
        return True
    return True


def _truncated_quotient(x, y):
    quotient = abs(x) // abs(y)
    return quotient if (x < 0) == (y < 0) else -quotient


def _int_model(operator, dtype, x, y):
    assert _int_defined(operator, dtype, x, y), (operator, dtype, x, y)
    wrap = functools.partial(_wrap, dtype)
    if operator == "Add":
        return wrap(x + y)
    if operator == "Subtract":
        return wrap(x - y)
    if operator == "Multiply":
        return wrap(x * y)
    if operator == "Divide":
        return wrap(_truncated_quotient(x, y))
    if operator == "Remainder":
        remainder = x - _truncated_quotient(x, y) * y
        if _signed(dtype) and remainder != 0 and (remainder < 0) != (y < 0):
            remainder += y
        return wrap(remainder)
    if operator == "Maximum":
        return max(x, y)
    if operator == "Minimum":
        return min(x, y)
    if operator in COMPARISONS:
        return int(
            {
                "Equal": x == y,
                "NotEqual": x != y,
                "Greater": x > y,
                "GreaterEqual": x >= y,
                "Less": x < y,
                "LessEqual": x <= y,
            }[operator]
        )
    if operator == "BitwiseAnd":
        return wrap(x & y)
    if operator == "BitwiseOr":
        return wrap(x | y)
    if operator == "BitwiseXor":
        return wrap(x ^ y)
    if operator == "LeftShift":
        return wrap(x << y)
    if operator == "RightShift":
        return wrap(x >> y)
    if operator == "Power":
        if y < 0:
            return 0
        result, base, exponent = 1, x, y
        while exponent:
            if exponent & 1:
                result = wrap(result * base)
            exponent >>= 1
            base = wrap(base * base)
        return result
    raise AssertionError(operator)


def _bool_model(operator, x, y):
    """``bool`` operands promote to ``int``; results convert back to bool."""
    if operator in ("Divide", "Remainder"):
        assert y == 1, "division by false is undefined"
        return x if operator == "Divide" else 0
    return int(
        {
            "Add": bool(x + y),
            "Subtract": bool(x - y),
            "Multiply": bool(x * y),
            "Maximum": max(x, y),
            "Minimum": min(x, y),
            "Power": x if y else 1,
            "Equal": x == y,
            "NotEqual": x != y,
            "Greater": x > y,
            "GreaterEqual": x >= y,
            "Less": x < y,
            "LessEqual": x <= y,
            "LogicalAnd": x and y,
            "LogicalOr": x or y,
            "BitwiseAnd": x & y,
            "BitwiseOr": x | y,
            "BitwiseXor": x ^ y,
        }[operator]
    )


def _complex_words(result):
    return tuple(_word("float32", part) for part in result)


def _complex_greater(x, y):
    """``complex.h`` orders by real part, then imaginary part."""
    return _less(y[0], x[0]) or (_equal(x[0], y[0]) and _less(y[1], x[1]))


def _complex_model(operator, a, b):
    x = tuple(_value("float32", part) for part in a)
    y = tuple(_value("float32", part) for part in b)
    exact = functools.partial(_exactly, "float32")
    nan_x, nan_y = NAN in x, NAN in y
    if operator in ("Add", "Subtract"):
        flip = _negate if operator == "Subtract" else (lambda value: value)
        return _complex_words(tuple(_add(p, flip(q)) for p, q in zip(x, y)))
    if operator == "Multiply":
        products = [exact(_multiply(p, q)) for p in x for q in y]
        real = _add(products[0], _negate(products[3]))
        imag = _add(products[1], products[2])
        return _complex_words((real, imag))
    if operator == "Divide":
        denominator = exact(
            _add(exact(_multiply(y[0], y[0])), exact(_multiply(y[1], y[1])))
        )
        real = exact(_add(exact(_multiply(x[0], y[0])), exact(_multiply(x[1], y[1]))))
        imag = exact(
            _add(exact(_multiply(x[1], y[0])), _negate(exact(_multiply(x[0], y[1]))))
        )
        return _complex_words((_divide(real, denominator), _divide(imag, denominator)))
    if operator == "Remainder":
        parts = []
        for p, q in zip(x, y):
            assert p[0] == q[0] == "finite" and q[2] != 0, (a, b)
            quotient = int(_number(p) / _number(q))
            product = exact(
                _multiply(q, ("finite", int(quotient < 0), Fraction(abs(quotient))))
            )
            part = exact(_add(p, _negate(product)))
            zero = ("finite", 0, Fraction(0))
            if not _is_zero(part) and _less(part, zero) != _less(q, zero):
                part = exact(_add(part, q))
            parts.append(part)
        return _complex_words(parts)
    if operator in COMPARISONS:

        def at_least(p, q):
            # complex.h: p >= q orders by real part, then imaginary part.
            return _less(q[0], p[0]) or (
                _equal(p[0], q[0]) and (_less(q[1], p[1]) or _equal(p[1], q[1]))
            )

        return int(
            {
                "Equal": _equal(x[0], y[0]) and _equal(x[1], y[1]),
                "NotEqual": not _equal(x[0], y[0]) or not _equal(x[1], y[1]),
                "Greater": _complex_greater(x, y),
                "GreaterEqual": at_least(x, y),
                "Less": _complex_greater(y, x),
                "LessEqual": at_least(y, x),
            }[operator]
        )
    if operator == "NaNEqual":
        return int(
            (_equal(x[0], y[0]) and _equal(x[1], y[1]))
            or (x[0] == y[0] == x[1] == y[1] == NAN)
            or (_equal(x[0], y[0]) and x[1] == y[1] == NAN)
            or (x[0] == y[0] == NAN and _equal(x[1], y[1]))
        )
    if operator in ("Maximum", "Minimum"):
        if nan_x:
            chosen = a
        elif operator == "Maximum":
            chosen = a if _complex_greater(x, y) else b
        else:
            chosen = a if _complex_greater(y, x) else b
        return tuple(
            ANY_NAN if _value("float32", part) == NAN else part for part in chosen
        )
    if operator == "Power":
        zero = x[0][0] == x[1][0] == "finite" and x[0][2] == x[1][2] == 0
        if zero:
            return (ANY_NAN, ANY_NAN) if nan_y else (0, 0)
        # Nonzero bases have finite magnitude above one and angle in [0, pi].
        assert x[0][0] == x[1][0] == "finite" and x[1][1] == 0
        assert x[0][2] ** 2 + x[1][2] ** 2 > 1
        if nan_y:
            return (ANY_NAN, ANY_NAN)
        assert y == (("finite", 0, 0), ("finite", 0, 0)), b
        return (0x3F800000, 0)
    if operator == "LogAddExp":
        if nan_x or nan_y:
            return (ANY_NAN, 0)
        larger, smaller = (x, y) if _complex_greater(x, y) else (y, x)
        result = a if larger is x else b
        if smaller[0] == ("inf", 1) or larger[0] == ("inf", 0):
            return result
        # The correction is below 2**-46; every imaginary part is at least 1/4.
        assert _number(larger[0]) - _number(smaller[0]) >= 32
        assert larger[0][2] >= 16 and larger[1][0] == "finite" and larger[1][2] >= 0.25
        return result
    raise AssertionError(operator)


def _model(operator, dtype, a, b):
    if dtype == "bool_":
        return _bool_model(operator, a, b)
    if dtype in INTEGERS:
        return _int_model(operator, dtype, a, b)
    if dtype == "complex64":
        return _complex_model(operator, a, b)
    return _float_model(operator, dtype, a, b)


# ---------------------------------------------------------------------------
# Operand domains: every pair in A x B is defined and exactly modelled.
# ---------------------------------------------------------------------------


def _f(dtype, value):
    """Return the word of an exact value, "-0", "inf", "-inf" or "nan"."""
    if value == "nan":
        return CANONICAL_NANS[dtype]
    if value in ("inf", "-inf"):
        return _word(dtype, ("inf", int(value == "-inf")))
    if value == "-0":
        return _word(dtype, ("finite", 1, Fraction(0)))
    value = Fraction(value)
    return _word(dtype, ("finite", int(value < 0), abs(value)))


def _signed_values(values):
    return [sign * Fraction(value) for value in values for sign in (1, -1)]


SPECIALS = ["0", "-0", "inf", "-inf", "nan"]
# Multiples of 1/4 up to 31.25: sums and differences stay below 64 and so
# need at most eight significant bits, which bfloat16 holds.
QUARTERS = _signed_values(
    Fraction(k, 4)
    for k in (1, 2, 3, 4, 5, 7, 9, 10, 13, 16, 22, 31, 40, 49, 64, 99, 125)
)
# Products of these need at most six significant bits.
SMALL = _signed_values(k * Fraction(2) ** e for k in (1, 3, 5, 7) for e in range(-2, 4))
POWERS = _signed_values(Fraction(2) ** e for e in range(-3, 4))
LARGE = [Fraction(v) for v in ("16", "17.5", "20", "24.25", "31.75", "48", "63.5")]
FLOAT_EXTREMES = {
    "float16": [0x7BFF, 0xFBFF, 0x0400, 0x8400, 0x3C01, 0x3BFF, 0x5BFC],
    "float32": [0x7F7FFFFF, 0xFF7FFFFF, 0x00800000, 0x80800000, 0x3F800001, 0x4B800001],
    "bfloat16": [0x7F7F, 0xFF7F, 0x0080, 0x8080, 0x3F81, 0x4780, 0x4049],
}


def _rich(dtype):
    """Normal values for selection and comparison, plus the special values."""
    words = [_f(dtype, value) for value in SPECIALS]
    words += [_f(dtype, value) for value in QUARTERS[::3] + LARGE[::2]]
    return list(dict.fromkeys(words + FLOAT_EXTREMES[dtype]))


def _float_domains(operator, dtype):
    def words(values):
        return [_f(dtype, value) for value in values]

    if operator in ("Add", "Subtract"):
        values = words(SPECIALS + QUARTERS)
        return [(values, values)]
    if operator == "Multiply":
        values = words(SPECIALS + SMALL)
        return [(values, values)]
    if operator == "Divide":
        return [(words(SPECIALS + SMALL), words(SPECIALS + POWERS))]
    if operator == "Remainder":
        return [(words(SPECIALS + QUARTERS), words(SPECIALS + POWERS))]
    if operator in ("Maximum", "Minimum", "NaNEqual", *COMPARISONS):
        values = _rich(dtype)
        return [(values, values)]
    if operator == "LogAddExp":
        large = words(["inf", "-inf", "nan"] + LARGE)
        small = words(["inf", "-inf", "nan"] + [-value for value in LARGE])
        return [(large, small), (small, large)]
    if operator == "ArcTan2":
        positive = [value for value in QUARTERS + LARGE if value > 0]
        return [
            (words(["0", "-0", "nan"]), words(["0", "inf", "nan"] + positive)),
            (words(["0", "-0", "nan"] + QUARTERS), words(["inf", "nan"])),
        ]
    if operator == "Power":
        return [
            (_rich(dtype), words(["0", "-0", "1"])),
            (
                words(["0", "-0", "inf", "-inf", "1", "-1", "nan"]),
                words(SPECIALS + ["1", "-1", "2", "-2", "3", "-3", "0.5", "-0.5"]),
            ),
        ]
    raise AssertionError(operator)


def _int_edges(dtype):
    return list(dict.fromkeys(int(value) for value in EDGES[dtype]))


def _within(dtype, values, limit):
    return [value for value in values if abs(value) <= limit]


def _int_domains(operator, dtype):
    edges = _int_edges(dtype)
    low, high = INTEGER_RANGES[dtype]
    bits = ITEM_SIZES[dtype] * 8
    wide_signed = _signed(dtype) and not _promoted(dtype)
    if operator in ("Add", "Subtract"):
        if wide_signed:
            values = _within(dtype, edges, high // 2) + [high // 2, low // 2]
            return [(values, values)]
        return [(edges, edges)]
    if operator == "Multiply":
        if wide_signed or dtype == "uint16":
            limit = math.isqrt(2**31 - 1 if _promoted(dtype) else high)
            values = _within(dtype, edges, limit) + [
                limit,
                -limit if _signed(dtype) else 0,
            ]
            return [(list(dict.fromkeys(values)), list(dict.fromkeys(values)))]
        return [(edges, edges)]
    if operator in ("Divide", "Remainder"):
        divisors = [value for value in edges if value != 0]
        if wide_signed:
            divisors = [value for value in divisors if value != -1]
        return [(edges, divisors)]
    if operator == "Power":
        if not (_signed(dtype) or _promoted(dtype)):
            bases = [0, 1, 2, 3, 5, 7, 0x10, high, high - 1]
            exponents = [0, 1, 2, 3, 5, 8, 13, bits - 1, bits, 2 * bits + 1]
            return [(bases, exponents)]
        if dtype == "uint16":
            return [([0, 1, 2, 3], list(range(16)))]
        if _promoted(dtype):
            # Products of narrow operands always fit the promoted int.
            bases = [0, 1, 2, 3, 5, 7, -1, -2, -3, 0x5A, low, high]
            exponents = [-3, -1, 0, 1, 2, 3, 4, 7, 9, 15, 16, 21]
            return [
                (
                    [value for value in bases if low <= value <= high],
                    [value for value in exponents if low <= value <= high],
                )
            ]
        limit = 3 if dtype == "int32" else 7
        bases = [0, 1, -1, 2, -2, 3, -3] + ([5, -7, 7] if limit == 7 else [])
        return [(bases, [-3, -1, 0, 1, 2, 3, 5, 7, 10, 15])]
    if operator == "LeftShift":
        amounts = sorted({0, 1, 2, 3, 4, 7} | {bits // 2, bits - 1})
        if _signed(dtype) or _promoted(dtype):
            width = 32 if _promoted(dtype) else bits
            small = [value for value in edges if 0 <= value < 2 ** (width - bits + 1)]
            wide = [value for value in edges if 0 <= value < 2 ** (width - bits // 2)]
            return [
                (wide, [value for value in amounts if value <= bits // 2]),
                (small, amounts),
            ]
        return [(edges, amounts)]
    if operator == "RightShift":
        return [(edges, sorted({0, 1, 2, 3, 4, 7} | {bits // 2, bits - 1}))]
    return [(edges, edges)]


def _complex(re, im):
    return (_f("float32", re), _f("float32", im))


def _complex_domains(operator):
    quarters = QUARTERS[::2]
    if operator in ("Add", "Subtract"):
        values = [
            _complex(re, im) for re, im in zip(quarters, quarters[3:] + quarters[:3])
        ]
        values += [
            _complex(*pair)
            for pair in (("-0", "0"), ("inf", "1"), ("nan", "2"), ("1", "-inf"))
        ]
        return [(values, list(reversed(values)))]
    if operator == "Multiply":
        small = SMALL[::3]
        values = [_complex(re, im) for re, im in zip(small, small[5:] + small[:5])]
        values += [_complex(*pair) for pair in (("0", "-0"), ("-0", "3"), ("nan", "1"))]
        return [(values, list(reversed(values)))]
    if operator == "Divide":
        small = SMALL[::3]
        numerators = [_complex(re, im) for re, im in zip(small, small[4:] + small[:4])]
        numerators += [_complex(*pair) for pair in (("0", "-0"), ("-0", "5"))]
        divisors = [
            _complex(*pair)
            for pair in (
                ("2", "0"),
                ("-0.5", "0"),
                ("0", "4"),
                ("-0", "-1"),
                ("1", "1"),
                ("2", "-2"),
                ("-0.25", "0.25"),
                ("0", "0"),
            )
        ]
        return [(numerators, divisors)]
    if operator == "Remainder":
        values = [
            _complex(re, im) for re, im in zip(QUARTERS, QUARTERS[5:] + QUARTERS[:5])
        ]
        values += [_complex(*pair) for pair in (("0", "-0"), ("-0", "4"), ("-4", "8"))]
        divisors = [
            _complex(re, im) for re in ("2", "-0.5", "4") for im in ("1", "-2", "0.25")
        ]
        return [(values, divisors)]
    if operator in (*COMPARISONS, "NaNEqual", "Maximum", "Minimum"):
        parts = ["0", "-0", "1", "-1", "2.5", "-31.25", "inf", "-inf", "nan"]
        values = [_complex(re, im) for re in parts for im in parts[::2]]
        values += [_complex("1", "-0"), _complex("-0", "nan")]
        return [(values, values[7:] + values[:7])]
    if operator == "Power":
        bases = [
            _complex(*pair)
            for pair in (
                ("0", "0"),
                ("-0", "0"),
                ("0", "-0"),
                ("2", "1"),
                ("-3", "0.5"),
                ("1.5", "2"),
                ("-2", "3"),
                ("0.25", "4"),
                ("2", "0"),
                ("-3", "0"),
            )
        ]
        exponents = [_complex("0", "0"), _complex("nan", "0"), _complex("0", "nan")]
        return [(bases, exponents)]
    if operator == "LogAddExp":
        imag = ["1", "-2.5", "0.25", "-31.25", "7"]
        large = [_complex(re, im) for re, im in zip(LARGE, imag + imag)]
        large += [_complex("inf", "1"), _complex("nan", "1"), _complex("20", "nan")]
        small = [_complex(-re, im) for re, im in zip(LARGE, imag[::-1] + imag)]
        small += [_complex("-inf", "2"), _complex("-inf", "-3")]
        return [(large, small), (small, large)]
    raise AssertionError(operator)


@functools.lru_cache(maxsize=None)
def _domains(operator, dtype):
    if dtype == "bool_":
        divisor = [1] if operator in ("Divide", "Remainder") else [0, 1]
        return (([0, 1], divisor),)
    if dtype in INTEGERS:
        domains = _int_domains(operator, dtype)
    elif dtype == "complex64":
        domains = _complex_domains(operator)
    else:
        domains = _float_domains(operator, dtype)
    return tuple(
        (tuple(dict.fromkeys(first)), tuple(dict.fromkeys(second)))
        for first, second in domains
    )


# ---------------------------------------------------------------------------
# Layouts and cases.
# ---------------------------------------------------------------------------


def _layout(kernel, spec, work, name):
    """Return the grid, constants and (output, a, b) index triples."""
    constants = {}
    if kernel == "binary_ss":
        grid = list(spec["grid"])
        return grid, constants, [(index, 0, 0) for index in range(grid[0])]
    if kernel in ("binary_sv", "binary_vs", "binary_vv"):
        size = spec["size"]
        grid = [-(-size // work), 1, 1]
        constants["size"] = ("uint32", [size])
    elif kernel in ("binary_sv2", "binary_vs2", "binary_vv2"):
        threads = spec["grid"][0] * spec["grid"][1]
        # A short last tile exercises the tail loop; one element per thread
        # has no tail, so the grid then covers the size exactly.
        size = threads * work - (1 if name == "tail" and work > 1 else 0)
        grid = [spec["grid"][0], spec["grid"][1], 1]
        constants["size"] = ("int64", [size])
    if "size" in constants:
        scalar_a = kernel.startswith("binary_sv")
        scalar_b = kernel.startswith("binary_vs")
        return (
            grid,
            constants,
            [
                (index, 0 if scalar_a else index, 0 if scalar_b else index)
                for index in range(size)
            ],
        )
    shape = spec["shape"]
    strides_a, strides_b = spec["strides"]
    points = list(itertools.product(*(range(extent) for extent in shape)))
    triples = [
        (
            output,
            sum(i * s for i, s in zip(point, strides_a)),
            sum(i * s for i, s in zip(point, strides_b)),
        )
        for output, point in enumerate(points)
    ]
    if kernel == "binary_g":
        grid = [-(-shape[-1] // work), shape[-2], math.prod(shape[:-2])]
        constants["shape"] = ("int32", list(shape))
        constants["a_strides"] = ("int64", list(strides_a))
        constants["b_strides"] = ("int64", list(strides_b))
        constants["ndim"] = ("int32", [len(shape)])
    else:
        grid = list(reversed(shape)) + [1] * (3 - len(shape))
        if kernel == "binary_g_nd1":
            constants["a_stride"] = ("int64", list(strides_a))
            constants["b_stride"] = ("int64", list(strides_b))
        else:
            constants["a_strides"] = ("int64", list(strides_a))
            constants["b_strides"] = ("int64", list(strides_b))
    return grid, constants, triples


def _fill(reads, domain, salt):
    """Assign domain values to an operand in first-read order."""
    extent = max(reads) + 1
    order = list(dict.fromkeys(reads))
    seen = set(order)
    order += [index for index in range(extent) if index not in seen]
    values = [None] * extent
    for rank, index in enumerate(order):
        values[index] = domain[(salt + rank) % len(domain)]
    return values


def _different(dtype, value):
    """Return a value of ``dtype`` that differs from ``value``."""
    guard = GUARDS[dtype]
    if dtype == "complex64":
        return guard if value != guard else (guard[0] ^ 1, guard[1])
    return guard if value != guard else guard ^ 1


def _decoys(dtype):
    return (
        [1, 0] * (DECOY_COUNT // 2)
        if dtype == "bool_"
        else [GUARDS[dtype]] * DECOY_COUNT
    )


def _case(entry, name, target="metal"):
    """Return the dispatch, operands and expected output of one case."""
    kind, operator, dtype = _split(entry)
    output = _output(operator, dtype)
    kernel = KINDS[kind][0]
    names = list(CASES[kernel])
    position = names.index(name)
    grid, constants, triples = _layout(kernel, CASES[kernel][name], _work(entry), name)
    domains = _domains(operator, dtype)
    first, second = domains[position % len(domains)]
    salt = position * 23 + KIND_SALTS[kind]
    # The first case pairs equal ranks, the second shifts ``b``.
    a = _fill([item[1] for item in triples], first, salt)
    b = _fill([item[2] for item in triples], second, salt + 5 * position)
    written = {
        out: _model(operator, dtype, a[left], b[right]) for out, left, right in triples
    }
    assert len(written) == len(triples), "every output is written once"
    initial, expected = [], []
    for index in range(len(triples) + GUARD_COUNT):
        if output == "bool_" and target == "metal":
            # One-byte Metal bools hold canonical values only.
            first_value = 1 - written[index] if index in written else index % 2
        elif index in written:
            first_value = _different(output, written[index])
        else:
            first_value = GUARDS[output]
        initial.append(first_value)
        expected.append(written.get(index, first_value))
    return {
        "grid": grid,
        "constants": constants,
        "a": a + _decoys(dtype),
        "b": b + _decoys(dtype),
        "initial": initial,
        "expected": expected,
        "triples": triples,
        "writtenCount": len(triples),
    }


# ---------------------------------------------------------------------------
# Encoding and comparison.
# ---------------------------------------------------------------------------


def _concrete(dtype, values):
    """Replace ``ANY_NAN`` with canonical NaNs; return values and NaN slots."""
    width = 2 if dtype == "complex64" else 1
    nans, concrete = [], []
    for index, value in enumerate(values):
        if dtype == "complex64":
            parts = []
            for part_index, part in enumerate(value):
                if part == ANY_NAN:
                    nans.append(index * width + part_index)
                    part = CANONICAL_NANS["float32"]
                parts.append(part)
            value = tuple(parts)
        elif value == ANY_NAN:
            nans.append(index)
            value = CANONICAL_NANS[dtype]
        concrete.append(value)
    return concrete, nans


def _check(actual, wanted, nans, bits):
    """Compare readback words exactly, accepting any NaN in NaN-only slots."""
    assert len(actual) == len(wanted), (len(actual), len(wanted))
    nan_slots = set(nans)
    mismatches = [
        (index, got, want)
        for index, (got, want) in enumerate(zip(actual, wanted))
        if (not _is_nan(bits, got) if index in nan_slots else got != want)
    ]
    assert not mismatches, mismatches[:8]


def _metal_words(dtype, raw):
    if dtype == "complex64":
        return [word for pair in struct.iter_unpack("<II", raw) for word in pair]
    return [value for (value,) in struct.iter_unpack("<" + METAL_FORMATS[dtype], raw)]


def _metal_wanted(dtype, values):
    concrete, nans = _concrete(dtype, values)
    if dtype == "complex64":
        return [part for value in concrete for part in value], nans
    return [int(value) for value in concrete], nans


# ---------------------------------------------------------------------------
# Translation identity.
# ---------------------------------------------------------------------------

WORKLOADS = {
    "metal": {item.entry_point: item for item in complete_metal.BINARY_METAL_WORKLOADS},
    "opengl": {
        item.entry_point: item for item in complete_opengl.BINARY_OPENGL_WORKLOADS
    },
    "directx": {
        item.entry_point: item for item in complete_directx.BINARY_DIRECTX_WORKLOADS
    },
}
ARTIFACT_CONTRACTS = {
    "metal": complete_metal.BINARY_METAL_CONTRACT_PATH,
    "opengl": complete_opengl.BINARY_OPENGL_CONTRACT_PATH,
    "directx": complete_directx.BINARY_DIRECTX_CONTRACT_PATH,
}
# Complex power reads ``hypot`` at the current commit; its artifacts differ.
CHANGED_ENTRIES = tuple(entry for entry in ENTRIES if entry.endswith("_Powercomplex64"))


def _config(target, entry):
    """Return the complete-family translation configuration of an entry."""
    workload = WORKLOADS[target][entry]
    return {
        "metal": complete_metal._project_config,
        "opengl": complete_opengl._project_config,
        "directx": complete_directx._project_config,
    }[target](workload)


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _artifact(target, entry):
    """Return the (sha256, size) the entry's artifact must reproduce."""
    if entry in CHANGED_ENTRIES:
        record = _contract()["currentArtifacts"][target][entry]
        return record["sha256"], record["sizeBytes"]
    workload = WORKLOADS[target][entry]
    return workload.sha256, workload.size_bytes


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"binary family shard {index} is outside [0, {count})")
    return ENTRIES[index::count]


SELECTED_ENTRIES = _selected_entries()


def _families():
    return sorted({_split(entry)[1:] for entry in ENTRIES})


def _domain_digest():
    """Pin every operand domain without spelling 238 lists in the contract."""
    payload = {
        f"{operator}/{dtype}": [
            [list(map(_jsonable, first)), list(map(_jsonable, second))]
            for first, second in _domains(operator, dtype)
        ]
        for operator, dtype in _families()
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value):
    return list(value) if isinstance(value, tuple) else value


def _dispatch_count():
    return sum(len(CASES[_kernel(entry)]) for entry in ENTRIES)


def _element_types():
    return {
        target: {dtype: _element_type(target, dtype) for dtype in ALL_TYPES}
        for target in TARGETS
    }


def _index_range_assertions():
    return {
        "metal": [],
        "opengl": [
            {"expression": expression, "minimum": low, "maximum": high}
            for expression, low, high in complete_opengl.INDEX_RANGE_ASSERTIONS
        ],
        "directx": [
            {"expression": expression, "minimum": low, "maximum": high}
            for expression, low, high in complete_directx.INDEX_RANGE_ASSERTIONS
        ],
    }


def expected_contract(current_artifacts=None):
    """Return the contract this module pins, for review and regeneration."""
    if current_artifacts is None:
        current_artifacts = _contract()["currentArtifacts"]
    return {
        "schemaVersion": 1,
        "kind": "crosstl-mlx-current-tree-binary-family-contract",
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
        # binary.metal and binary.h are byte-identical at the corpus commit;
        # binary_ops.h changes complex Power only, so every other entry must
        # reproduce the artifact its complete-family contract pins there.
        "artifactContracts": {
            target: {
                "path": ARTIFACT_CONTRACTS[target].relative_to(ROOT).as_posix(),
                "commit": complete_metal.MLX_COMMIT,
                "sourceSha256": SOURCE_SHA256,
                "operatorsHeaderSha256": COMPLETE_OPS_HEADER_SHA256,
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
            "inputDecoyCount": DECOY_COUNT,
            "elementTypes": _element_types(),
            "operandDomains": {
                "families": len(_families()),
                "domainCount": sum(
                    len(_domains(operator, dtype)) for operator, dtype in _families()
                ),
                "sha256": _domain_digest(),
                "policy": (
                    "every pair in A x B is defined in C++ and exactly modelled; "
                    "case n draws from domain n modulo the family's domain count"
                ),
            },
            "operatorModel": {
                "integers": (
                    "C++ semantics after promotion: two's complement wrap, "
                    "truncating division, MLX floored remainder and power; "
                    "undefined pairs excluded"
                ),
                "bool": "promoted to int, converted back to bool",
                "floatingPoint": (
                    "IEEE-754 results that need no rounding and are normal or "
                    "zero; signed zeros and infinities exact; NaN accepts any NaN"
                ),
                "float16Remainder": (
                    "Metal binary16 fmod: x - trunc(x / y) * y in binary32, NaN "
                    "for non-finite operands or a zero divisor"
                ),
                "complex64": "complex.h formulas with exact float32 intermediates",
                "selection": "Maximum and Minimum return operand bits",
                "transcendental": (
                    "LogAddExp, ArcTan2 and Power only at exactly defined values"
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


def test_binary_entries_match_the_upstream_instantiation():
    assert len(ENTRIES) == 4122 and len(set(ENTRIES)) == 4122
    assert len(_families()) == 238
    for target in TARGETS:
        assert set(WORKLOADS[target]) == set(ENTRIES)
    for entry in ENTRIES:
        kind, operator, dtype = _split(entry)
        workload = WORKLOADS["metal"][entry]
        assert (workload.shape, workload.operator_type) == (kind, operator)
        assert workload.template_name == KINDS[kind][0], entry
    works = {kind: set() for kind in KINDS}
    for entry in ENTRIES:
        works[_split(entry)[0]].add(_work(entry))
    # Work-per-thread kinds skip the 8-byte types that ``*2`` kinds cover.
    assert works["svn"] == works["vvn"] == {2, 4, 8}
    assert works["sv2"] == works["vv2"] == {1, 2, 4, 8}
    assert works["gn4large"] == {4} and works["gn2"] == {2}
    assert all(works[kind] == {1} for kind in ("ss", "sv", "vv", "g1", "g3large"))
    assert len(CHANGED_ENTRIES) == 15


def test_current_mlx_binary_family_contract_is_exact():
    assert _contract() == json.loads(json.dumps(expected_contract()))


def test_binary_family_reuses_the_complete_translation_identities():
    for target, path in ARTIFACT_CONTRACTS.items():
        contract = json.loads(path.read_text(encoding="utf-8"))
        assert contract["commit"] == complete_metal.MLX_COMMIT
        assert contract["source"] == SOURCE
        assert contract["sourceSha256"] == SOURCE_SHA256
        assert {item["entryPoint"] for item in contract["entries"]} == set(ENTRIES)
        changed = _contract()["currentArtifacts"][target]
        assert set(changed) == set(CHANGED_ENTRIES)
        for entry in CHANGED_ENTRIES:
            record = changed[entry]
            assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
            assert record["sha256"] != WORKLOADS[target][entry].sha256
    assert complete_metal.MLX_BINARY_SHA256 == SOURCE_SHA256


@pytest.mark.parametrize("target", TARGETS)
def test_binary_family_configs_match_the_complete_contracts(tmp_path, target):
    entry = "gn4large_Remainderfloat32"
    path = tmp_path / "crosstl.toml"
    path.write_text(_config(target, entry) + "\n", encoding="utf-8")
    config = load_project_config(tmp_path, path)
    assert list(config.targets) == [target]
    assert dict(config.entry_points) == {SOURCE: entry}
    assert [item.to_json() for item in config.index_range_assertions] == [
        {"source": SOURCE, **item} for item in _index_range_assertions()[target]
    ]
    assert not config.metadata_preconditions


def test_operand_domains_are_defined_and_exactly_modelled():
    """Every pair of every domain is modelled without rounding or UB."""
    for operator, dtype in _families():
        domains = _domains(operator, dtype)
        assert 1 <= len(domains) <= 2
        for first, second in domains:
            assert first and second, (operator, dtype)
            for a, b in itertools.product(first, second):
                if dtype in INTEGERS:
                    assert _int_defined(operator, dtype, a, b), (operator, dtype, a, b)
                    low, high = INTEGER_RANGES[dtype]
                    assert low <= a <= high and low <= b <= high
                result = _model(operator, dtype, a, b)
                if _output(operator, dtype) == "bool_":
                    assert result in (0, 1)


@pytest.mark.parametrize("operator,dtype", _families())
def test_binary_cases_cover_every_domain_value(operator, dtype):
    entries = [entry for entry in ENTRIES if _split(entry)[1:] == (operator, dtype)]
    domains = _domains(operator, dtype)
    seen = [(set(), set()) for _ in domains]
    results = set()
    for entry in entries:
        names = list(CASES[_kernel(entry)])
        for position, name in enumerate(names):
            case = _case(entry, name)
            index = position % len(domains)
            for out, left, right in case["triples"]:
                seen[index][0].add(case["a"][left])
                seen[index][1].add(case["b"][right])
                results.add(json.dumps(case["expected"][out]))
            assert len(case["expected"]) == len(case["initial"])
            written = {out for out, _, _ in case["triples"]}
            for slot, (first, final) in enumerate(
                zip(case["initial"], case["expected"])
            ):
                assert (first != final) is (slot in written)
    for (first, second), (got_a, got_b) in zip(domains, seen):
        assert set(first) <= got_a, sorted(set(first) - got_a, key=str)[:5]
        assert set(second) <= got_b, sorted(set(second) - got_b, key=str)[:5]
    if (operator, dtype) == ("Remainder", "bool_"):
        # ``x % true`` is always false; dividing by false is undefined.
        assert results == {"0"}
    elif _output(operator, dtype) == "bool_":
        assert results == {"0", "1"}, (operator, dtype)
    else:
        assert len(results) > 1


@pytest.mark.parametrize("kernel", CASES)
def test_binary_layouts_write_each_output_once(kernel):
    works = {_work(entry) for entry in ENTRIES if _kernel(entry) == kernel}
    for work in sorted(works):
        for name, spec in CASES[kernel].items():
            grid, constants, triples = _layout(kernel, spec, work, name)
            outputs = [out for out, _, _ in triples]
            assert outputs == list(range(len(outputs)))
            assert all(extent > 0 for extent in grid)
            assert min(min(a, b) for _, a, b in triples) >= 0
            if kernel in ("binary_sv", "binary_vs", "binary_vv"):
                size = constants["size"][1][0]
                assert (grid[0] - 1) * work < size <= grid[0] * work
            if kernel == "binary_g":
                assert spec["shape"][-1] % work != 0
    if kernel in ("binary_sv", "binary_vs", "binary_vv", "binary_sv2"):
        tails = [
            _layout(kernel, spec, 4, name)[1]["size"][1][0] % 4
            for name, spec in CASES[kernel].items()
        ]
        assert any(tails) and not all(tails)


@pytest.mark.parametrize(
    "operator,dtype,a,b,expected",
    [
        ("Add", "float32", 0x80000000, 0x80000000, 0x80000000),
        ("Add", "float32", 0x3F800000, 0xBF800000, 0x00000000),
        ("Subtract", "float16", 0x7C00, 0x7C00, ANY_NAN),
        ("Multiply", "bfloat16", 0x7F80, 0x8000, ANY_NAN),
        ("Multiply", "float32", 0xC0400000, 0x00000000, 0x80000000),
        ("Divide", "float32", 0x3F800000, 0x80000000, 0xFF800000),
        ("Divide", "float16", 0xC500, 0x3800, 0xC900),
        ("Remainder", "float32", 0xC0800000, 0x40000000, 0x80000000),
        ("Remainder", "float32", 0x40400000, 0xC0000000, 0xBF800000),
        ("Remainder", "float32", 0xBFC00000, 0x7F800000, 0x7F800000),
        ("Remainder", "float32", 0x3FC00000, 0x00000000, ANY_NAN),
        # Metal's binary16 fmod, read back from the upstream kernel: exact
        # zero differences are +0 and infinite divisors give NaN.
        ("Remainder", "float16", 0xCC00, 0xC400, 0x0000),
        ("Remainder", "float16", 0xB400, 0xB400, 0x0000),
        ("Remainder", "float16", 0x8000, 0xFC00, ANY_NAN),
        ("Remainder", "float16", 0xBC00, 0x7C00, ANY_NAN),
        ("Remainder", "float16", 0x4A20, 0x4000, 0x3400),
        ("Remainder", "float32", 0xC1800000, 0xC0800000, 0x80000000),
        ("Remainder", "float32", 0x80000000, 0xFF800000, 0x80000000),
        ("Remainder", "bfloat16", 0xBF80, 0x7F80, 0x7F80),
        ("Maximum", "float32", 0x80000000, 0x00000000, 0x00000000),
        ("Maximum", "float32", 0x00000000, 0x80000000, 0x80000000),
        ("Minimum", "float16", 0x3C00, 0x7E00, ANY_NAN),
        ("Maximum", "float16", 0x7E00, 0x3C00, ANY_NAN),
        ("Equal", "float32", 0x80000000, 0x00000000, 1),
        ("NotEqual", "float32", 0x7FC00000, 0x7FC00000, 1),
        ("NaNEqual", "bfloat16", 0x7FC0, 0x7FC0, 1),
        ("GreaterEqual", "float16", 0x7C00, 0x7C00, 1),
        ("LogAddExp", "float32", 0x41A00000, 0xC1A00000, 0x41A00000),
        ("LogAddExp", "float16", 0xFC00, 0xFC00, 0xFC00),
        ("ArcTan2", "float32", 0x80000000, 0x00000000, 0x80000000),
        ("ArcTan2", "float32", 0xC1F80000, 0x7F800000, 0x80000000),
        ("Power", "float32", 0x7FC00000, 0x00000000, 0x3F800000),
        ("Power", "float32", 0x80000000, 0xC0400000, 0xFF800000),
        ("Power", "float32", 0x80000000, 0xC0000000, 0x7F800000),
        ("Power", "float32", 0xBF800000, 0x7F800000, 0x3F800000),
        ("Power", "float32", 0xBF800000, 0x3F000000, ANY_NAN),
        ("Power", "float32", 0xFF800000, 0x40400000, 0xFF800000),
        ("Power", "float32", 0x00000000, 0xFF800000, 0x7F800000),
        ("Divide", "int32", -7, 2, -3),
        ("Remainder", "int32", -7, 2, 1),
        ("Remainder", "int8", 7, -2, -1),
        ("Remainder", "uint16", 7, 2, 1),
        ("Divide", "int8", -128, -1, -128),
        ("LeftShift", "int8", 1, 7, -128),
        ("LeftShift", "uint16", 0xFFFF, 15, 0x8000),
        ("RightShift", "int16", -32768, 15, -1),
        ("Power", "int32", -3, 3, -27),
        ("Power", "int32", 2, -1, 0),
        ("Power", "uint8", 3, 5, 243),
        ("Power", "uint32", 0xFFFFFFFF, 2, 1),
        ("Subtract", "uint8", 0, 1, 255),
        ("Multiply", "int16", -32768, -1, -32768),
        ("Subtract", "bool_", 0, 1, 1),
        ("Add", "bool_", 1, 1, 1),
        ("Power", "bool_", 0, 0, 1),
        ("Remainder", "bool_", 1, 1, 0),
        (
            "Multiply",
            "complex64",
            (0x40000000, 0x3F800000),
            (0x40400000, 0xC0000000),
            (0x41000000, 0xBF800000),
        ),
        (
            "Divide",
            "complex64",
            (0x40800000, 0x40000000),
            (0x00000000, 0x40000000),
            (0x3F800000, 0xC0000000),
        ),
        (
            "Remainder",
            "complex64",
            (0xC0800000, 0x40A00000),
            (0x40000000, 0xC0000000),
            (0x00000000, 0xBF800000),
        ),
        ("Greater", "complex64", (0x3F800000, 0x40000000), (0x3F800000, 0x3F800000), 1),
        (
            "LessEqual",
            "complex64",
            (0x3F800000, 0x7FC00000),
            (0x3F800000, 0x3F800000),
            0,
        ),
        (
            "NaNEqual",
            "complex64",
            (0x3F800000, 0x7FC00000),
            (0x3F800000, 0x7FC00000),
            1,
        ),
        (
            "Maximum",
            "complex64",
            (0x7FC00000, 0x3F800000),
            (0x40000000, 0x0),
            (ANY_NAN, 0x3F800000),
        ),
        ("Power", "complex64", (0x40000000, 0x3F800000), (0, 0), (0x3F800000, 0)),
        ("Power", "complex64", (0x80000000, 0), (0, 0x7FC00000), (ANY_NAN, ANY_NAN)),
        (
            "LogAddExp",
            "complex64",
            (0x7FC00000, 0x3F800000),
            (0xC1A00000, 0x3F800000),
            (ANY_NAN, 0),
        ),
        (
            "LogAddExp",
            "complex64",
            (0xC1A00000, 0x3E800000),
            (0x41A00000, 0xBE800000),
            (0x41A00000, 0xBE800000),
        ),
    ],
)
def test_operator_models_pin_mlx_semantics(operator, dtype, a, b, expected):
    """Pin the IEEE, C++ and ``binary_ops.h`` boundaries the gate checks."""
    assert _model(operator, dtype, a, b) == expected


@pytest.mark.parametrize(
    "operator,dtype,a,b",
    [
        ("Add", "int32", 2**31 - 1, 1),
        ("Divide", "int64", -(2**63), -1),
        ("Remainder", "uint8", 3, 0),
        ("LeftShift", "int32", -1, 1),
        ("LeftShift", "uint32", 1, 32),
        ("Multiply", "uint16", 0xFFFF, 0xFFFF),
        ("Power", "int32", 7, 15),
    ],
)
def test_undefined_integer_pairs_are_rejected(operator, dtype, a, b):
    assert not _int_defined(operator, dtype, a, b)


@pytest.mark.parametrize(
    "operator,dtype,a,b",
    [
        ("Add", "float16", 0x7BFF, 0x7BFF),
        ("Multiply", "bfloat16", 0x3F81, 0x3F81),
        ("Divide", "float32", 0x3F800000, 0x40400000),
        ("ArcTan2", "float32", 0x3F800000, 0x3F800000),
        ("Power", "float32", 0x40400000, 0x40000000),
        ("LogAddExp", "float32", 0x3F800000, 0x40000000),
    ],
)
def test_models_refuse_results_that_are_not_exact(operator, dtype, a, b):
    with pytest.raises(AssertionError):
        _model(operator, dtype, a, b)


@pytest.mark.parametrize(
    "fault", ["value", "missing", "guard", "nan", "zero-sign", "imaginary"]
)
def test_binary_readback_rejects_corruption(fault):
    imaginary = fault == "imaginary"
    entry = "vv_Dividecomplex64" if imaginary else "vv_Dividefloat16"
    dtype = "complex64" if imaginary else "float16"
    width = 2 if imaginary else 1
    magnitude = 0x7FFFFFFF if imaginary else 0x7FFF
    case = _case(entry, "tail")
    wanted, nans = _metal_wanted(dtype, case["expected"])
    assert nans
    actual = list(wanted)
    finite = next(
        index
        for index, word in enumerate(wanted)
        if index < case["writtenCount"] * width
        and index not in nans
        and word & magnitude
        and (index % 2 == 1 or not imaginary)
    )
    if fault in ("value", "imaginary"):
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
    bits = 32 if imaginary else "float16"
    with pytest.raises(AssertionError):
        _check(actual, wanted, nans, bits)
    _check(wanted, wanted, nans, bits)


def test_binary_shards_partition_the_family(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert {len(shard) for shard in shards} == {343, 344}
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


# ---------------------------------------------------------------------------
# Native execution.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native binary family gate")
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
    work = tmp_path_factory.mktemp("binary-family-metal-reference")
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
    kind, operator, dtype = _split(entry)
    output = _output(operator, dtype)
    kernel = KINDS[kind][0]
    payloads = {
        0: _metal_bytes(dtype, case["a"]),
        1: _metal_bytes(dtype, case["b"]),
        2: _metal_bytes(output, case["initial"]),
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
        if index != 2:
            assert (readback / f"buffer-{index}.bin").read_bytes() == Path(
                path
            ).read_bytes()
    return device, (readback / "buffer-2.bin").read_bytes()


def _dispatch(descriptor, package, target, entry, case):
    _, operator, dtype = _split(entry)
    output = _output(operator, dtype)
    concrete, nans = _concrete(output, case["expected"])
    wanted = _runtime_value(target, output, concrete)["values"]
    data = {
        "a": _runtime_value(target, dtype, case["a"]),
        "b": _runtime_value(target, dtype, case["b"]),
        "c": _runtime_value(target, output, case["initial"]),
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
        if name == "c":
            output_name = binding["name"]
        else:
            observed.append(binding["name"])
    assert output_name is not None and len(inputs) == len(data), sorted(inputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output_name: dict(data["c"])},
        {"workgroupCount": case["grid"], "workgroupSize": [1, 1, 1]},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output_name, wanted, nans, len(observed)


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_current_mlx_binary_family_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    _, operator, dtype = _split(entry)
    output = _output(operator, dtype)
    sha256, size_bytes = _artifact(target, entry)
    bits = _nan_bits(target, output) if output in FLOATS else 32
    metal_bits = _nan_bits("metal", output) if output in FLOATS else 32
    with tempfile.TemporaryDirectory(prefix=".current-binary-", dir=root) as directory:
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


def test_ci_requires_native_binary_family_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX binary family native parity",
        family="binary",
        count=CI_SHARD_COUNT,
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        f"{SHARD_INDEX_ENV}: ${{{{ matrix.shard }}}}",
        f'{SHARD_COUNT_ENV}: "{CI_SHARD_COUNT}"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/${{ matrix.group }}",
        "--junitxml=mlx-current-results/${{ matrix.group }}.xml",
        "demos/integrations/mlx/tests/kernels/test_current_binary_family.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_binary_family.py"
    )
