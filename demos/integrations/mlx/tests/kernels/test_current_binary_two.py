"""Execute every pinned MLX ``binary_two`` DivMod kernel against exact oracles."""

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
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    load_project_config,
    translate_project,
    validate_project_report,
)
from crosstl.project.runtime_value_encoding import FLOAT16_BITS
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.kernels.test_current_division import (
    _expected as _quotient,
)
from demos.integrations.mlx.tests.kernels.test_current_floating_remainder import (
    _expected as _float_remainder,
)
from demos.integrations.mlx.tests.kernels.test_current_half_remainder import (
    _remainder as _half_remainder,
)
from demos.integrations.mlx.tests.kernels.test_current_ternary import (
    GUARDS,
    ITEM_SIZES,
    _element_type,
    _metal_bytes,
    _operand_offsets,
    _runtime_value,
)
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/binary_two.metal"
SOURCE_SHA256 = "a425eb847614267e04a92ee70009ef9250914a327848c8e3a71519a233266142"
HEADER = "mlx/backend/metal/kernels/binary_two.h"
HEADER_SHA256 = "af26413a6161c1b37667fc5c3a802510d8b425074ee2ca0a8fd7e31646f52cd8"
OPS_HEADER = "mlx/backend/metal/kernels/binary_ops.h"
OPS_HEADER_SHA256 = "6dbe789faafbf7259f411c7fab7c0fcdd192df47a78f8f0bd408228f7a30e5c7"
COMPLEX_HEADER = "mlx/backend/metal/kernels/complex.h"
COMPLEX_HEADER_SHA256 = (
    "16e8a815b2cbdb6070e0824e64fe33fccb6e918f1b84ea5c792bd89d33e57bf1"
)
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_BINARY_TWO"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/binary_two.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
# ``instantiate_binary_types(DivMod)`` order.
DTYPES = (
    "bool_",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "complex64",
    "float16",
    "float32",
    "bfloat16",
)
# ``instantiate_binary_base`` covers every type; the work-per-thread ``*n``
# variants of ``instantiate_binary_all`` skip the 8-byte types.
BASE_ONLY_DTYPES = ("uint64", "int64", "complex64")
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
    "g2",
    "g3",
    "g1large",
    "g2large",
    "g3large",
)
WORK_KINDS = ("svn", "vsn", "vvn")
ENTRIES = tuple(
    f"{kind}_DivMod{dtype}"
    for dtype in DTYPES
    for kind in BASE_KINDS + (() if dtype in BASE_ONLY_DTYPES else WORK_KINDS)
)
# (kernel, case family, scalar a, scalar b, explicit work per thread or None
# for ``WorkPerThread<T>::n``, generic tile width).
KINDS = {
    "ss": ("binary_ss", "binary_ss", True, True, 1, None),
    "sv": ("binary_sv", "binary_flat", True, False, 1, None),
    "vs": ("binary_vs", "binary_flat", False, True, 1, None),
    "vv": ("binary_vv", "binary_flat", False, False, 1, None),
    "svn": ("binary_sv", "binary_flat", True, False, None, None),
    "vsn": ("binary_vs", "binary_flat", False, True, None, None),
    "vvn": ("binary_vv", "binary_flat", False, False, None, None),
    "sv2": ("binary_sv2", "binary_flat2", True, False, None, None),
    "vs2": ("binary_vs2", "binary_flat2", False, True, None, None),
    "vv2": ("binary_vv2", "binary_flat2", False, False, None, None),
    "g1": ("binary_g_nd1", "binary_g_nd1", False, False, 1, None),
    "g1large": ("binary_g_nd1", "binary_g_nd1", False, False, 1, None),
    "g2": ("binary_g_nd2", "binary_g_nd2", False, False, 1, None),
    "g2large": ("binary_g_nd2", "binary_g_nd2", False, False, 1, None),
    "g3": ("binary_g_nd3", "binary_g_nd3", False, False, 1, None),
    "g3large": ("binary_g_nd3", "binary_g_nd3", False, False, 1, None),
    "gn2": ("binary_g", "binary_g", False, False, 1, 2),
    "gn4large": ("binary_g", "binary_g", False, False, 1, 4),
}
KIND_ORDER = BASE_KINDS + WORK_KINDS
# Kernels that read both operands at the output index, in pair-tiling order.
VECTOR_KINDS = ("vv", "vv2", "vvn")
# Output counts for the scalar kernel and flat kernels, (x, y) grids for the
# ``*2`` kernels and shapes with per-operand strides (outermost first) for
# the strided kernels.
CASES = {
    "binary_ss": {"single": {"size": 1}, "broadcast": {"size": 6}},
    "binary_flat": {"tail": {"size": 37}, "exact": {"size": 80}},
    "binary_flat2": {"tail": {"grid": [3, 2]}, "wide": {"grid": [4, 3]}},
    "binary_g_nd1": {
        "strided": {"shape": [19], "strides": [[1], [2]]},
        "broadcast": {"shape": [11], "strides": [[3], [0]]},
    },
    "binary_g_nd2": {
        "transpose": {"shape": [3, 5], "strides": [[5, 1], [1, 3]]},
        "broadcast": {"shape": [4, 2], "strides": [[1, 4], [0, 1]]},
    },
    "binary_g_nd3": {
        "permute": {"shape": [2, 2, 3], "strides": [[6, 3, 1], [1, 2, 4]]},
        "broadcast": {"shape": [3, 1, 4], "strides": [[4, 4, 1], [0, 0, 1]]},
    },
    "binary_g": {
        "nd4": {"shape": [2, 3, 2, 5], "strides": [[30, 10, 5, 1], [0, 10, 1, 2]]},
        "nd5-broadcast": {
            "shape": [2, 2, 1, 3, 3],
            "strides": [[18, 9, 9, 3, 1], [0, 3, 0, 1, 6]],
        },
    },
}
GUARD_COUNT = 16
DECOY_COUNT = 4
# OpenGL indexes 32-bit storage arrays. These host preconditions bound every
# 64-bit source index of the ``large``, ``*2`` and generic kernels to signed
# int range; each fixture index stays far below the bound.
OPENGL_INDEX_RANGE_ASSERTIONS = tuple(
    (expression, 0, 2**31 - 1)
    for expression in (
        "offset + i",
        "a_idx",
        "b_idx",
        "out_idx",
        "out_idx++",
        "idx.x",
        "idx.y",
    )
)
# The pinned per-operator profiles reproduce Metal's binary32 division,
# remainder, comparison and addition and its binary16 remainder exactly.
BINARY32_PROFILES = (
    ("binary32_division_profile", "rne-flush"),
    ("binary32_comparison_profile", "flush-subnormals"),
    ("binary32_remainder_profile", "flush-arithmetic-subnormals"),
    ("binary32_additive_profile", "rne-flush"),
)
PROFILES = {
    "float32": BINARY32_PROFILES,
    "bfloat16": BINARY32_PROFILES,
    "float16": (("binary16_remainder_profile", "binary32-quotient"),),
}
PROVENANCE_FIELDS = {
    "binary32_division_profile": "binary32DivisionProfile",
    "binary32_comparison_profile": "binary32ComparisonProfile",
    "binary32_remainder_profile": "binary32RemainderProfile",
    "binary32_additive_profile": "binary32AdditiveProfile",
    "binary16_remainder_profile": "binary16RemainderProfile",
}
METAL_FORMATS = {
    "bool_": "B",
    "uint8": "B",
    "uint16": "H",
    "uint32": "I",
    "uint64": "Q",
    "int8": "b",
    "int16": "h",
    "int32": "i",
    "int64": "q",
    "float16": "H",
    "float32": "I",
    "bfloat16": "H",
}
INTEGER_BITS = {
    "bool_": 1,
    "uint8": 8,
    "uint16": 16,
    "uint32": 32,
    "uint64": 64,
    "int8": 8,
    "int16": 16,
    "int32": 32,
    "int64": 64,
}
FLOAT_FORMATS = {"float16": (16, 5, 10), "bfloat16": (16, 8, 7), "float32": (32, 8, 23)}


def _signed_limits(bits):
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def _integer_pairs(dtype):
    """Return dividend/divisor pairs with every sign and extreme combination."""
    if dtype == "bool_":
        # Bool division promotes to int; only a true divisor is defined.
        return [(0, 1), (1, 1)]
    bits = INTEGER_BITS[dtype]
    if dtype.startswith("uint"):
        top = (1 << bits) - 1
        dividends = [0, 1, 2, 7, 100, top // 3, (1 << (bits - 1)) + 5, top - 1, top]
        divisors = [1, 2, 3, 7, (1 << (bits - 1)) + 1, top]
        return list(itertools.product(dividends, divisors))
    low, high = _signed_limits(bits)
    dividends = [0, 7, -7, 1, -1, 6, -6, 100, -100, high, low, low + 1, high - 1]
    divisors = [1, -1, 2, -2, 3, -3, 7, -7, high, low]
    pairs = list(itertools.product(dividends, divisors))
    if bits >= 32:
        # Without integer promotion, MIN / -1 overflows (undefined in Metal).
        pairs.remove((low, -1))
    return pairs


def _float_word(value, dtype):
    if dtype == "float32":
        return struct.unpack("<I", struct.pack("<f", value))[0]
    if dtype == "float16":
        return struct.unpack("<H", struct.pack("<e", value))[0]
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    assert word & 0xFFFF == 0, value
    return word >> 16


FLOAT_POOLS = {
    # Signed zeros, unit and integral values, values with fractional
    # quotients of both signs, infinities, quiet and signaling NaNs, the
    # smallest and largest subnormals and normals, and the largest finite.
    "float32": (
        [
            0x00000000,
            0x80000000,
            0x3F800000,
            0xBF800000,
            0x40E00000,
            0xC0E00000,
            0x40200000,
            0xC0700000,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0x7FA00001,
            0x00000001,
            0x80800000,
            0x7F7FFFFF,
            0xCB7FFFFF,
            0x3EAAAAAB,
            0x501502F9,
        ],
        [
            0x40000000,
            0xC0000000,
            0x40400000,
            0xBF000000,
            0x00000000,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0x00000003,
            0x3DCCCCCD,
            0x3F7FFFFF,
        ],
    ),
    "bfloat16": (
        [
            0x0000,
            0x8000,
            0x3F80,
            0xBF80,
            0x40E0,
            0xC0E0,
            0x4020,
            0xC070,
            0x7F80,
            0xFF80,
            0x7FC0,
            0x7F81,
            0x0001,
            0x8080,
            0x7F7F,
            0xCB7F,
            0x3EAB,
            0x5015,
        ],
        [
            0x4000,
            0xC000,
            0x4040,
            0xBF00,
            0x0000,
            0x8000,
            0x7F80,
            0xFF80,
            0x7FC0,
            0x0003,
            0x3DCD,
            0x3F7F,
        ],
    ),
    # 39.90625 / 6.65234375 rounds to 6.0 in binary16 before the floor, the
    # upstream half semantics documented in METAL_HOST.md.
    "float16": (
        [
            0x0000,
            0x8000,
            0x3C00,
            0xBC00,
            0x4700,
            0xC700,
            0x4100,
            0xC380,
            0x7C00,
            0xFC00,
            0x7E00,
            0x7D01,
            0x0001,
            0x8400,
            0x7BFF,
            0xD3FF,
            0x3555,
            0x50FD,
        ],
        [
            0x4000,
            0xC000,
            0x4200,
            0xB800,
            0x0000,
            0x8000,
            0x7C00,
            0xFC00,
            0x7E00,
            0x0003,
            0x2E66,
            0x46A7,
        ],
    ),
}
# Dyadic components keep every complex product, sum, quotient and
# truncation exact, so the result does not depend on contraction; both
# divisor components are nonzero and equal in magnitude.
COMPLEX_POOLS = (
    [
        (7.0, -9.0),
        (-6.5, 3.25),
        (0.75, -0.25),
        (-0.0, 5.0),
        (12.0, 0.0),
        (-3.0, -3.0),
        (1.5, 2.5),
        (100.0, -100.0),
        (0.0, -0.0),
        (-17.25, 9.5),
    ],
    [
        (2.0, 2.0),
        (-2.0, 2.0),
        (4.0, -4.0),
        (0.5, 0.5),
        (-1.0, -1.0),
        (8.0, 8.0),
        (2.0, -2.0),
        (-0.25, 0.25),
    ],
)


def _pairs(dtype):
    if dtype in INTEGER_BITS:
        return _integer_pairs(dtype)
    if dtype == "complex64":
        return [
            (
                tuple(_float_word(part, "float32") for part in x),
                tuple(_float_word(part, "float32") for part in y),
            )
            for x, y in itertools.product(*COMPLEX_POOLS)
        ]
    dividends, divisors = FLOAT_POOLS[dtype]
    return list(itertools.product(dividends, divisors))


def _wrap(value, dtype):
    bits = INTEGER_BITS[dtype]
    if dtype == "bool_":
        return int(bool(value))
    value %= 1 << bits
    if dtype.startswith("int") and value >> (bits - 1):
        value -= 1 << bits
    return value


def _integer_divmod(x, y, dtype):
    """FloorDivide and Remainder on the promoted operands, then narrowed."""
    if dtype == "bool_":
        x, y = int(bool(x)), int(bool(y))
    assert y != 0
    quotient = abs(x) // abs(y) * (1 if (x < 0) == (y < 0) else -1)
    remainder = x - quotient * y
    floor_quotient = quotient
    if remainder != 0 and (x < 0) != (y < 0):
        floor_quotient -= 1
    floor_remainder = remainder
    if remainder != 0 and (remainder < 0) != (y < 0):
        floor_remainder += y
    return _wrap(floor_quotient, dtype), _wrap(floor_remainder, dtype)


def _float_value(word, dtype):
    if dtype == "float32":
        return struct.unpack("<f", struct.pack("<I", word))[0]
    if dtype == "float16":
        return struct.unpack("<e", struct.pack("<H", word))[0]
    return struct.unpack("<f", struct.pack("<I", word << 16))[0]


def _floor_word(word, dtype):
    """Floor a binary16, bfloat16 or binary32 value exactly."""
    width, exponent_bits, fraction_bits = FLOAT_FORMATS[dtype]
    magnitude = word & ((1 << (width - 1)) - 1)
    if magnitude == 0 or magnitude >> fraction_bits == (1 << exponent_bits) - 1:
        # Signed zeros, infinities and NaNs are their own floor.
        return word
    return _float_word(float(math.floor(_float_value(word, dtype))), dtype)


def _float_divmod(x, y, dtype):
    quotient = _floor_word(_quotient(x, y, dtype), dtype)
    if dtype == "float16":
        return quotient, _half_remainder(x, y)
    return quotient, _float_remainder(x, y, dtype)


def _exact32(value):
    word = _float_word(value, "float32")
    assert _float_value(word, "float32") == value or math.isnan(value), value
    return word


def _complex_divmod(x, y):
    """``complex64_t`` operator/ and operator% with exact dyadic operands."""
    ar, ai = (_float_value(word, "float32") for word in x)
    br, bi = (_float_value(word, "float32") for word in y)
    denominator = br * br + bi * bi
    quotient = (
        (ar * br + ai * bi) / denominator,
        (ai * br - ar * bi) / denominator,
    )
    real = ar - br * float(math.trunc(ar / br))
    imag = ai - bi * float(math.trunc(ai / bi))
    if real != 0 and (real < 0) != (br < 0):
        real += br
    if imag != 0 and (imag < 0) != (bi < 0):
        imag += bi
    return (
        tuple(_exact32(part) for part in quotient),
        (_exact32(real), _exact32(imag)),
    )


def _divmod(x, y, dtype):
    if dtype in INTEGER_BITS:
        return _integer_divmod(x, y, dtype)
    if dtype == "complex64":
        return _complex_divmod(x, y)
    return _float_divmod(x, y, dtype)


def _kind(entry):
    return entry.split("_DivMod", 1)[0]


def _dtype(entry):
    return entry.split("_DivMod", 1)[1]


def _kernel(entry):
    return KINDS[_kind(entry)][0]


def _family(entry):
    return KINDS[_kind(entry)][1]


def _work(entry):
    """Return the elements each thread of a flat kernel handles."""
    explicit = KINDS[_kind(entry)][4]
    return explicit if explicit is not None else 8 // ITEM_SIZES[_dtype(entry)]


def _layout(entry, name):
    """Return grid, constants and per-output operand offsets of one case."""
    kernel, family, scalar_a, scalar_b, _, tile = KINDS[_kind(entry)]
    spec = CASES[family][name]
    work = _work(entry)
    constants = {}
    if family in {"binary_ss", "binary_flat", "binary_flat2"}:
        if family == "binary_ss":
            size = spec["size"]
            grid = [size, 1, 1]
        elif family == "binary_flat":
            size = spec["size"]
            grid = [math.ceil(size / work), 1, 1]
            constants["size"] = size
        else:
            threads = spec["grid"][0] * spec["grid"][1]
            # A short last tile exercises the tail loop; one element per
            # thread has no tail, so the grid then covers the size exactly.
            size = threads * work - (1 if name == "tail" and work > 1 else 0)
            grid = [spec["grid"][0], spec["grid"][1], 1]
            constants["size"] = size
        offsets = {
            "a": [0] * size if scalar_a else list(range(size)),
            "b": [0] * size if scalar_b else list(range(size)),
        }
    else:
        shape = spec["shape"]
        strides = dict(zip("ab", spec["strides"]))
        offsets = {key: _operand_offsets(shape, strides[key]) for key in "ab"}
        if kernel == "binary_g":
            grid = [math.ceil(shape[-1] / tile), shape[-2], math.prod(shape[:-2])]
            constants["shape"] = list(shape)
            constants["ndim"] = len(shape)
        else:
            grid = list(reversed(shape)) + [1] * (3 - len(shape))
        for key in "ab":
            if kernel == "binary_g_nd1":
                constants[f"{key}_stride"] = strides[key][0]
            else:
                constants[f"{key}_strides"] = list(strides[key])
    return {
        "grid": grid,
        "constants": constants,
        "offsets": offsets,
        "count": len(offsets["a"]),
    }


def _pair_start(entry, name):
    """Return the pair index of a case's first operand element.

    The same-index vector kernels take consecutive pairs, so together they
    realize every pair of each dtype exactly; the other layouts recombine
    dividends and divisors.
    """
    kind, dtype = _kind(entry), _dtype(entry)
    if kind not in VECTOR_KINDS:
        return 37 * list(CASES[_family(entry)]).index(name) + 101 * KIND_ORDER.index(
            kind
        )
    start = 0
    for vector_kind in VECTOR_KINDS:
        vector_entry = f"{vector_kind}_DivMod{dtype}"
        if vector_entry not in ENTRIES:
            continue
        for case_name in CASES[KINDS[vector_kind][1]]:
            if (vector_kind, case_name) == (kind, name):
                return start
            start += _layout(vector_entry, case_name)["count"]
    raise AssertionError(entry)


def _case(entry, name):
    """Return grid, constants, operand storage and expected logical outputs."""
    dtype = _dtype(entry)
    case = _layout(entry, name)
    offsets, count = case["offsets"], case["count"]
    pairs = _pairs(dtype)
    start = _pair_start(entry, name)
    operands = {}
    for position, key in enumerate("ab"):
        length = max(offsets[key]) + 1
        operands[key] = [
            pairs[(start + index) % len(pairs)][position] for index in range(length)
        ]
    if dtype in {"int32", "int64"}:
        # A broadcast can pair MIN with -1 across pairs; keep it defined.
        low = _signed_limits(INTEGER_BITS[dtype])[0]
        for index in range(count):
            a_index, b_index = offsets["a"][index], offsets["b"][index]
            if operands["a"][a_index] == low and operands["b"][b_index] == -1:
                operands["a"][a_index] = low + 1
    outputs = [
        _divmod(operands["a"][offsets["a"][i]], operands["b"][offsets["b"][i]], dtype)
        for i in range(count)
    ]
    for key in "ab":
        operands[key] = operands[key] + [GUARDS[dtype]] * DECOY_COUNT
    return {
        **case,
        "operands": operands,
        "expected": {
            "c": [quotient for quotient, _ in outputs],
            "d": [remainder for _, remainder in outputs],
        },
    }


def _guard_words(dtype, target=None):
    """Return logical guard values that follow each output region."""
    if dtype == "bool_" and target in {None, "metal"}:
        # One-byte Metal bools hold canonical values; alternate them instead.
        return [index % 2 for index in range(GUARD_COUNT)]
    return [GUARDS[dtype]] * GUARD_COUNT


def _output_initial(target, dtype, expected):
    """Fill an output so that every unwritten element differs from its result."""
    if dtype == "bool_" and target == "metal":
        return [1 - int(value) for value in expected]
    return [GUARDS[dtype]] * len(expected)


def _nan_format(target, dtype):
    """Return (magnitude mask, infinity) of the physical floating encoding."""
    if dtype in {"float32", "complex64"} or (
        target == "opengl" and dtype in {"float16", "bfloat16"}
    ):
        return 0x7FFFFFFF, 0x7F800000
    if dtype == "float16":
        return 0x7FFF, 0x7C00
    if dtype == "bfloat16":
        return 0x7FFF, 0x7F80
    return None


def _mismatches(target, dtype, actual, expected):
    """Compare exact words; any NaN matches any NaN for floating results."""
    assert len(actual) == len(expected)
    nan = _nan_format(target, dtype)
    mismatches = []
    for index, (got, want) in enumerate(zip(actual, expected)):
        if got == want and type(got) is type(want):
            continue
        if nan is not None and type(got) is int and type(want) is int:
            mask, infinity = nan
            if got & mask > infinity and want & mask > infinity:
                continue
        mismatches.append({"index": index, "expected": want, "actual": got})
    return mismatches


def _config(output, target, entry):
    profiles = "".join(
        f'{option} = "{value}"\n' for option, value in PROFILES.get(_dtype(entry), ())
    )
    assertions = ""
    if target == "opengl":
        for expression, low, high in OPENGL_INDEX_RANGE_ASSERTIONS:
            assertions += f"""
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "{expression}"
minimum = {low}
maximum = {high}
"""
    return f"""
[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
output_dir = "{output}"
[project.entry_points]
"{SOURCE}" = "{entry}"
[project.entry_workgroup_size_rules."{SOURCE}"]
"{entry}" = [1, 1, 1]
[project.source_options.metal]
max_template_specializations = 512
max_template_materialization_work = 65536
{profiles}{assertions}"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _case_contract():
    return {
        family: {name: dict(spec) for name, spec in cases.items()}
        for family, cases in CASES.items()
    }


def _dispatch_count():
    return sum(len(CASES[_family(entry)]) for entry in ENTRIES)


def _profile_contract():
    return {
        dtype: {PROVENANCE_FIELDS[option]: value for option, value in profiles}
        for dtype, profiles in PROFILES.items()
    }


def test_binary_two_entries_match_the_upstream_instantiation():
    assert len(ENTRIES) == 225 and len(set(ENTRIES)) == 225
    assert sum(1 for entry in ENTRIES if _kind(entry) in WORK_KINDS) == 30
    assert {
        _work(f"vvn_DivMod{dtype}") for dtype in DTYPES if dtype not in BASE_ONLY_DTYPES
    } == {2, 4, 8}
    assert _work("vv2_DivModcomplex64") == 1 and _work("vv_DivModuint8") == 1
    assert set(KINDS) == set(KIND_ORDER)


def test_current_mlx_binary_two_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-binary-two-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "header": HEADER,
        "headerSha256": HEADER_SHA256,
        "opsHeader": OPS_HEADER,
        "opsHeaderSha256": OPS_HEADER_SHA256,
        "complexHeader": COMPLEX_HEADER,
        "complexHeaderSha256": COMPLEX_HEADER_SHA256,
    }
    assert contract["corpusCensus"] == {
        "metalUnitCount": 49,
        "entryCount": 17832,
        "discoveryDiagnosticCount": 0,
    }
    assert tuple(contract["entries"]) == ENTRIES
    assert set(contract["artifacts"]) == set(TARGETS)
    for target in TARGETS:
        artifacts = contract["artifacts"][target]
        assert tuple(artifacts) == ENTRIES
        for record in artifacts.values():
            assert set(record) == {"sha256", "sizeBytes"}
            assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
    validation = contract["validationContract"]
    assert validation["workgroupSize"] == [1, 1, 1]
    assert validation["indexRangeAssertions"] == {
        "opengl": [
            {"expression": expression, "minimum": low, "maximum": high}
            for expression, low, high in OPENGL_INDEX_RANGE_ASSERTIONS
        ],
        "metal": [],
        "directx": [],
    }
    assert validation["sourceProfiles"] == _profile_contract()
    assert validation["cases"] == _case_contract()
    assert validation["pairCounts"] == {dtype: len(_pairs(dtype)) for dtype in DTYPES}
    assert validation["outputGuardCount"] == GUARD_COUNT
    assert validation["inputDecoyCount"] == DECOY_COUNT
    assert validation["comparison"] == (
        "exact words; any NaN matches any NaN for floating results"
    )
    assert validation["elementTypes"] == {
        target: {dtype: _element_type(target, dtype) for dtype in DTYPES}
        for target in TARGETS
    }
    assert validation["compilerValidationRequiredInCi"] == list(TARGETS)
    assert validation["runtimeRequiredInCi"] == list(TARGETS)
    assert validation["upstreamMetallibParityTargets"] == ["metal"]
    assert contract["scope"] == {
        "coveredEntryCount": 225,
        "runtimeCoveredEntryCount": 225,
        "dispatchCountByTarget": {target: _dispatch_count() for target in TARGETS},
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


@pytest.mark.parametrize("entry", ENTRIES)
def test_binary_two_cases_cover_layouts_and_keep_guards_distinct(entry):
    dtype = _dtype(entry)
    for name in CASES[_family(entry)]:
        case = _case(entry, name)
        assert case["count"] > 0
        for key in "cd":
            assert len(case["expected"][key]) == case["count"]
            # Guards follow every output and never equal a valid result.
            if dtype != "bool_":
                assert GUARDS[dtype] not in case["expected"][key]
        for key in "ab":
            assert case["operands"][key][-DECOY_COUNT:] == [GUARDS[dtype]] * 4
        if _kind(entry) in {"ss", "sv", "svn", "sv2"}:
            assert len(case["operands"]["a"]) == 1 + DECOY_COUNT
        if _kind(entry) in {"ss", "vs", "vsn", "vs2"}:
            assert len(case["operands"]["b"]) == 1 + DECOY_COUNT
    family = _family(entry)
    if family == "binary_flat" and _work(entry) > 1:
        # The 37-element case ends inside a tile.
        assert 37 % _work(entry) != 0
    if family == "binary_flat2" and _work(entry) > 1:
        assert _case(entry, "tail")["count"] % _work(entry) == _work(entry) - 1


@pytest.mark.parametrize("dtype", DTYPES)
def test_binary_two_vector_layouts_realize_every_pair(dtype):
    realized = set()
    for kind in KIND_ORDER:
        entry = f"{kind}_DivMod{dtype}"
        if entry not in ENTRIES or KINDS[kind][2] or KINDS[kind][3]:
            continue
        for name in CASES[KINDS[kind][1]]:
            case = _case(entry, name)
            realized.update(
                (case["operands"]["a"][a], case["operands"]["b"][b])
                for a, b in zip(case["offsets"]["a"], case["offsets"]["b"])
            )
    assert set(_pairs(dtype)) <= realized


@pytest.mark.parametrize(
    "dtype,x,y,expected",
    (
        ("int8", -128, -1, (-128, 0)),
        ("int8", -7, 2, (-4, 1)),
        ("int16", 7, -2, (-4, -1)),
        ("int32", -(2**31), 3, (-715827883, 1)),
        ("int64", 2**63 - 1, -(2**63), (-1, -1)),
        ("uint8", 255, 7, (36, 3)),
        ("uint64", 2**64 - 1, 2**63 + 1, (1, 2**63 - 2)),
        ("bool_", 1, 1, (1, 0)),
        # floor(-0.5) is -1 and the remainder takes the divisor's sign.
        ("float32", 0xBF800000, 0x40000000, (0xBF800000, 0x3F800000)),
        ("float32", 0x80000000, 0x3F800000, (0x80000000, 0x80000000)),
        # -2^-126 / 2 flushes to -0.0 before the floor; the negative
        # remainder -2^-126 then adds the divisor and rounds to 2.0.
        ("float32", 0x80800000, 0x40000000, (0x80000000, 0x40000000)),
        ("float32", 0x40E00000, 0x00000000, (0x7F800000, 0x7FC00000)),
        # 39.90625 / 6.65234375 is 6.0 in binary16, so the floor is 6 while
        # the binary32 quotient truncates to 5 for the remainder.
        ("float16", 0x50FD, 0x46A7, (0x4600, 0x46A5)),
        ("float16", 0xC700, 0x4000, (0xC400, 0x3C00)),
        ("bfloat16", 0xC0E0, 0x4000, (0xC080, 0x3F80)),
    ),
)
def test_divmod_reference_floors_and_adjusts_signs(dtype, x, y, expected):
    assert _divmod(x, y, dtype) == expected


def test_complex_reference_matches_upstream_operators():
    one = _float_word(1.0, "float32")
    quotient, remainder = _complex_divmod(
        (_float_word(7.0, "float32"), _float_word(-9.0, "float32")),
        (_float_word(2.0, "float32"), _float_word(2.0, "float32")),
    )
    # (7 - 9i) / (2 + 2i) = (-0.5, -4); 7 % 2 = 1 and -9 % 2 = 1.
    assert quotient == (_float_word(-0.5, "float32"), _float_word(-4.0, "float32"))
    assert remainder == (one, one)
    for x, y in _pairs("complex64"):
        _complex_divmod(x, y)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "dtype,nan,other,negative_zero",
    (
        ("float32", 0x7FC00000, 0xFFA00001, 0x80000000),
        ("float16", 0x7E00, 0xFD01, 0x8000),
        ("bfloat16", 0x7FC0, 0xFF81, 0x8000),
    ),
)
def test_float_comparison_classifies_nan_and_keeps_zero_signs(
    target, dtype, nan, other, negative_zero
):
    nan, other, zero, negative_zero = _runtime_value(
        target, dtype, [nan, other, 0, negative_zero]
    )["values"]
    assert _mismatches(target, dtype, [other], [nan]) == []
    assert _mismatches(target, dtype, [zero], [nan])
    assert _mismatches(target, dtype, [negative_zero], [zero])
    assert _mismatches(target, "int32", [1], [2])
    assert _mismatches(target, "bool_", [True], [1])


@pytest.mark.parametrize("target", TARGETS)
def test_binary_two_configs_scope_profiles_and_index_preconditions(tmp_path, target):
    for entry in ("vv_DivModint32", "vv_DivModfloat32", "vv_DivModfloat16"):
        path = tmp_path / "crosstl.toml"
        path.write_text(_config("out", target, entry), encoding="utf-8")
        config = load_project_config(tmp_path, path)
        assert [item.to_json() for item in config.index_range_assertions] == (
            [
                {
                    "source": SOURCE,
                    "expression": expression,
                    "minimum": low,
                    "maximum": high,
                }
                for expression, low, high in OPENGL_INDEX_RANGE_ASSERTIONS
            ]
            if target == "opengl"
            else []
        )
        options = config.source_options["metal"]
        for option in PROVENANCE_FIELDS:
            assert options.get(option) == dict(PROFILES.get(_dtype(entry), ())).get(
                option
            )
        assert list(config.targets) == [target]


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native binary_two gate")
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
    for path, digest in (
        (SOURCE, SOURCE_SHA256),
        (HEADER, HEADER_SHA256),
        (OPS_HEADER, OPS_HEADER_SHA256),
        (COMPLEX_HEADER, COMPLEX_HEADER_SHA256),
    ):
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("binary-two-metal-reference")
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


def _metal_constant_payloads(kernel, constants):
    if kernel == "binary_ss":
        return []
    if kernel in {"binary_sv", "binary_vs", "binary_vv"}:
        return [struct.pack("<I", constants["size"])]
    if kernel in {"binary_sv2", "binary_vs2", "binary_vv2"}:
        return [struct.pack("<q", constants["size"])]
    if kernel == "binary_g_nd1":
        return [struct.pack("<q", constants[f"{key}_stride"]) for key in "ab"]
    strides = [
        struct.pack(f"<{len(constants[name])}q", *constants[name])
        for name in ("a_strides", "b_strides")
    ]
    if kernel != "binary_g":
        return strides
    shape = struct.pack(f"<{len(constants['shape'])}i", *constants["shape"])
    return [shape, *strides, struct.pack("<i", constants["ndim"])]


def _metal_words(dtype, payload):
    """Decode Metal storage bytes into logical words (complex flattened)."""
    code = "I" if dtype == "complex64" else METAL_FORMATS[dtype]
    count = len(payload) // struct.calcsize("<" + code)
    return list(struct.unpack(f"<{count}{code}", payload))


def _logical_words(dtype, values):
    if dtype == "complex64":
        return [part for value in values for part in value]
    return [int(value) for value in values]


def _upstream_metal(work, reference, entry, case, name):
    runner, library = reference
    dtype = _dtype(entry)
    payloads = [
        _metal_bytes(dtype, case["operands"]["a"]),
        _metal_bytes(dtype, case["operands"]["b"]),
        *(
            _metal_bytes(
                dtype,
                _output_initial("metal", dtype, case["expected"][key])
                + _guard_words(dtype),
            )
            for key in "cd"
        ),
        *_metal_constant_payloads(_kernel(entry), case["constants"]),
    ]
    files = []
    for index, payload in enumerate(payloads):
        path = work / f"{name}-upstream-input-{index}.bin"
        path.write_bytes(payload)
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
    for index, payload in enumerate(payloads):
        if index not in (2, 3):
            assert (readback / f"buffer-{index}.bin").read_bytes() == payload
    outputs = {
        key: _metal_words(dtype, (readback / f"buffer-{index}.bin").read_bytes())
        for key, index in (("c", 2), ("d", 3))
    }
    return device, outputs


def _dispatch(descriptor, package, target, entry, case):
    dtype = _dtype(entry)
    kernel = _kernel(entry)
    guards = _guard_words(dtype, target)
    data = {key: _runtime_value(target, dtype, case["operands"][key]) for key in "ab"}
    for key in "cd":
        data[key] = _runtime_value(
            target,
            dtype,
            _output_initial(target, dtype, case["expected"][key]) + guards,
        )
    constants = case["constants"]
    if kernel in {"binary_sv", "binary_vs", "binary_vv"}:
        data["size"] = {"dtype": "uint32", "shape": [1], "values": [constants["size"]]}
    elif kernel in {"binary_sv2", "binary_vs2", "binary_vv2"}:
        data["size"] = {"dtype": "int64", "shape": [1], "values": [constants["size"]]}
    elif kernel == "binary_g_nd1":
        for key in "ab":
            data[f"{key}_stride"] = {
                "dtype": "int64",
                "shape": [1],
                "values": [constants[f"{key}_stride"]],
            }
    elif kernel != "binary_ss":
        for name in ("a_strides", "b_strides"):
            data[name] = {
                "dtype": "int64",
                "shape": [len(constants[name])],
                "values": list(constants[name]),
            }
        if kernel == "binary_g":
            data["shape"] = {
                "dtype": "int32",
                "shape": [len(constants["shape"])],
                "values": list(constants["shape"]),
            }
            data["ndim"] = {
                "dtype": "int32",
                "shape": [1],
                "values": [constants["ndim"]],
            }
    inputs, observed, output_names = {}, [], {}
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
        if name in {"c", "d"}:
            output_names[name] = binding["name"]
        else:
            observed.append(binding["name"])
    assert set(output_names) == {"c", "d"} and len(inputs) == len(data)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output_names[key]: dict(data[key]) for key in "cd"},
        {"workgroupCount": case["grid"], "workgroupSize": [1, 1, 1]},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    wanted = {
        key: _runtime_value(target, dtype, case["expected"][key] + guards)["values"]
        for key in "cd"
    }
    return request, output_names, wanted, len(observed)


@pytest.mark.parametrize("entry", ENTRIES)
def test_current_mlx_binary_two_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    dtype = _dtype(entry)
    record = _contract()["artifacts"][target][entry]
    with tempfile.TemporaryDirectory(
        prefix=".current-binary-two-", dir=root
    ) as directory:
        work = Path(directory)
        try:
            config = work / "crosstl.toml"
            config.write_text(
                _config(f"{work.name}/out", target, entry), encoding="utf-8"
            )
            translation = translate_project(
                load_project_config(root, config),
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
            assert artifact["generatedHash"] == {
                "algorithm": "sha256",
                "value": record["sha256"],
            }
            assert artifact["generatedSizeBytes"] == record["sizeBytes"]
            for option, value in PROFILES.get(dtype, ()):
                assert artifact["provenance"][PROVENANCE_FIELDS[option]] == value
            (execution,) = artifact["execution"]["entryPoints"]
            assert execution["workgroupSize"] == [1, 1, 1]
            generated = root / artifact["path"]
            text = generated.read_text(encoding="utf-8")
            assert "unsupported" not in text
            if target == "metal":
                _metal_library(generated, work / "translated.metallib", root)
            else:
                _validate_generated_artifact(generated, work, target)
            descriptor, package = _prepare_native_package(translation, work)
            executor = _executor(target)
            evidence = []
            try:
                for name in CASES[_family(entry)]:
                    case = _case(entry, name)
                    request, output_names, wanted, observed = _dispatch(
                        descriptor, package, target, entry, case
                    )
                    availability = executor.is_available(request)
                    assert availability.available, availability.reason
                    result = executor.run(request)
                    (work / f"{name}-result.json").write_text(
                        json.dumps(
                            {
                                "status": result.status,
                                "outputs": result.outputs,
                                "details": result.details,
                            },
                            indent=2,
                            default=str,
                        ),
                        encoding="utf-8",
                    )
                    assert result.status == "ok"
                    for key in "cd":
                        values = result.outputs[output_names[key]]["values"]
                        mismatches = _mismatches(target, dtype, values, wanted[key])
                        assert not mismatches, (name, key, mismatches[:8])
                        # Guards are exact; no NaN class applies to them.
                        assert values[-GUARD_COUNT:] == wanted[key][-GUARD_COUNT:]
                    observations = result.details["inputObservations"]
                    assert len(observations) == observed
                    assert all(item["unchanged"] for item in observations.values())
                    upstream = None
                    if target == "metal":
                        device, raw = _upstream_metal(
                            work, metal_reference, entry, case, name
                        )
                        for key in "cd":
                            expected_words = _logical_words(
                                dtype, case["expected"][key] + _guard_words(dtype)
                            )
                            mismatches = _mismatches(
                                "metal", dtype, raw[key], expected_words
                            )
                            assert not mismatches, (name, key, mismatches[:8])
                        upstream = {"device": device, "matched": True}
                    evidence.append(
                        {
                            "case": name,
                            "grid": case["grid"],
                            "constants": case["constants"],
                            "outputCount": case["count"],
                            "outputGuardCount": GUARD_COUNT,
                            "observedInputCount": observed,
                            "upstream": upstream,
                        }
                    )
            finally:
                close = getattr(executor.runtime_adapter.runtime, "close", None)
                if callable(close):
                    close()
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entry": entry,
                        "target": target,
                        "artifactSha256": record["sha256"],
                        "profiles": dict(PROFILES.get(dtype, ())),
                        "cases": evidence,
                        "comparison": (
                            "exact words; any NaN matches any NaN for "
                            "floating results"
                        ),
                        "indexRangeAssertions": (
                            [list(item) for item in OPENGL_INDEX_RANGE_ASSERTIONS]
                            if target == "opengl"
                            else []
                        ),
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)


def test_ci_requires_native_binary_two_parity():
    from tests.ci_helpers import (
        assert_native_step_prerequisites,
        assert_workflow_triggers,
    )
    from tools import ci_coverage

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Prove current MLX binary_two native parity"
    )
    assert_native_step_prerequisites(step, job="mlx-metal-porting")
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/binary-two",
        "--junitxml=mlx-current-results/binary-two.xml",
        "demos/integrations/mlx/tests/kernels/test_current_binary_two.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_binary_two.py"
    )
    upload = ci_coverage.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Upload current MLX runtime diagnostics"
    )
    assert "if: always()" in upload
    assert "path: mlx-current-results" in upload
    assert workflow.index("Prove current MLX binary_two native parity") < (
        workflow.index("Upload current MLX runtime diagnostics")
    )
