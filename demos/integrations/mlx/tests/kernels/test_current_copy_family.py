"""Execute every pinned MLX copy kernel natively against an exact cast model.

``copy.metal`` instantiates 2,496 kernels: twelve layouts plus two
work-per-thread variants over every pairing of MLX's thirteen dtypes, and
sixteen same-type general and dynamic-offset copies per dtype. Each entry
translates with the configuration of its complete-family translation contract,
whose pinned artifact it must reproduce byte for byte, then executes on Metal,
Mesa EGL and Direct3D 12 WARP. Every output element, every untouched padding
element and every guard after the output is checked against an independent
model of ``cast_to<U>(T)``. Metal also runs the unchanged upstream kernel,
which must produce the same bytes.
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
    test_copy_complete_directx as complete_directx,
)
from demos.integrations.mlx.tests.kernels import (
    test_copy_complete_metal_roundtrip as complete_metal,
)
from demos.integrations.mlx.tests.kernels import (
    test_copy_complete_opengl as complete_opengl,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
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

SOURCE = "mlx/backend/metal/kernels/copy.metal"
SOURCE_SHA256 = "ed8a579eb6fe6a14c36560d2c8b548baf99e66fa77d300fb4ad7554883820eba"
HEADER = "mlx/backend/metal/kernels/copy.h"
HEADER_SHA256 = "a9fad1b262ffc25f4ccda4e09d0c76821cd580f12871acd2a0daa98151e1587c"
UTILS_HEADER = "mlx/backend/metal/kernels/utils.h"
UTILS_HEADER_SHA256 = "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_COPY_FAMILY"
SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_COPY_FAMILY_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_COPY_FAMILY_SHARD_COUNT"
CI_SHARD_COUNT = 6
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/copy.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
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
    "float16",
    "float32",
    "bfloat16",
    "complex64",
)
# ``instantiate_copy_base`` covers every output type; the work-per-thread
# ``sn``/``vn`` variants of ``instantiate_copy_all`` skip 8-byte outputs.
BASE_ONLY_OUTPUTS = ("uint64", "int64", "complex64")
SAME_TYPE_KINDS = (
    "gg1",
    "gg2",
    "gg3",
    "ggn2",
    "gg1large",
    "gg2large",
    "gg3large",
    "ggn4large",
    "gg1_dynamic",
    "gg2_dynamic",
    "gg3_dynamic",
    "ggn2_dynamic",
    "gg1large_dynamic",
    "gg2large_dynamic",
    "gg3large_dynamic",
    "ggn4large_dynamic",
)
BASE_KINDS = (
    "s",
    "v",
    "s2",
    "v2",
    "g1",
    "g2",
    "g3",
    "gn2",
    "g1large",
    "g2large",
    "g3large",
    "gn4large",
)
WORK_KINDS = ("sn", "vn")
ENTRIES = tuple(
    entry
    for source in DTYPES
    for entry in (
        *(f"{kind}_copy{source}{source}" for kind in SAME_TYPE_KINDS),
        *(
            f"{kind}_copy{source}{output}"
            for output in DTYPES
            for kind in BASE_KINDS + (() if output in BASE_ONLY_OUTPUTS else WORK_KINDS)
        ),
    )
)
# kind: (kernel template, elements per thread or None for WorkPerThread<U>::n).
KINDS = {
    "s": ("copy_s", 1),
    "sn": ("copy_s", None),
    "v": ("copy_v", 1),
    "vn": ("copy_v", None),
    "s2": ("copy_s2", None),
    "v2": ("copy_v2", None),
    "g1": ("copy_g_nd1", 1),
    "g1large": ("copy_g_nd1", 1),
    "g2": ("copy_g_nd2", 1),
    "g2large": ("copy_g_nd2", 1),
    "g3": ("copy_g_nd3", 1),
    "g3large": ("copy_g_nd3", 1),
    "gn2": ("copy_g", 2),
    "gn4large": ("copy_g", 4),
    "gg1": ("copy_gg_nd1", 1),
    "gg1large": ("copy_gg_nd1", 1),
    "gg2": ("copy_gg_nd2", 1),
    "gg2large": ("copy_gg_nd2", 1),
    "gg3": ("copy_gg_nd3", 1),
    "gg3large": ("copy_gg_nd3", 1),
    "ggn2": ("copy_gg", 2),
    "ggn4large": ("copy_gg", 4),
    "gg1_dynamic": ("copy_gg_dynamic_nd1", 1),
    "gg1large_dynamic": ("copy_gg_dynamic_nd1", 1),
    "gg2_dynamic": ("copy_gg_dynamic_nd2", 1),
    "gg2large_dynamic": ("copy_gg_dynamic_nd2", 1),
    "gg3_dynamic": ("copy_gg_dynamic_nd3", 1),
    "gg3large_dynamic": ("copy_gg_dynamic_nd3", 1),
    "ggn2_dynamic": ("copy_gg_dynamic", 2),
    "ggn4large_dynamic": ("copy_gg_dynamic", 4),
}
_GENERAL_CASES = {
    "copy_g_nd1": {
        "strided": {"shape": [17], "srcStrides": [3]},
        "broadcast": {"shape": [9], "srcStrides": [0]},
    },
    "copy_g_nd2": {
        "transpose": {"shape": [3, 5], "srcStrides": [1, 3]},
        "broadcast": {"shape": [4, 2], "srcStrides": [0, 1]},
    },
    "copy_g_nd3": {
        "permute": {"shape": [2, 2, 3], "srcStrides": [1, 2, 4]},
        "broadcast": {"shape": [3, 1, 4], "srcStrides": [4, 0, 1]},
    },
    # The innermost extents (5 and 3) end inside a two- and four-wide tile.
    "copy_g": {
        "nd4": {"shape": [2, 3, 2, 5], "srcStrides": [1, 2, 30, 6]},
        "nd5-broadcast": {"shape": [2, 2, 1, 3, 3], "srcStrides": [0, 9, 0, 3, 1]},
    },
    # Destination strides leave padding that the kernel must not touch.
    "copy_gg_nd1": {
        "strided": {"shape": [13], "srcStrides": [2], "dstStrides": [3]},
        "broadcast": {"shape": [9], "srcStrides": [0], "dstStrides": [1]},
    },
    "copy_gg_nd2": {
        "transpose": {"shape": [3, 5], "srcStrides": [1, 3], "dstStrides": [5, 1]},
        "padded": {"shape": [4, 2], "srcStrides": [0, 1], "dstStrides": [3, 1]},
    },
    "copy_gg_nd3": {
        "permute": {
            "shape": [2, 2, 3],
            "srcStrides": [1, 2, 4],
            "dstStrides": [6, 3, 1],
        },
        "padded": {
            "shape": [3, 1, 4],
            "srcStrides": [4, 0, 1],
            "dstStrides": [10, 5, 2],
        },
    },
    "copy_gg": {
        "nd4": {
            "shape": [2, 3, 2, 5],
            "srcStrides": [1, 2, 30, 6],
            "dstStrides": [40, 12, 6, 1],
        },
        "nd5-broadcast": {
            "shape": [2, 2, 1, 3, 3],
            "srcStrides": [0, 9, 0, 3, 1],
            "dstStrides": [24, 12, 12, 4, 1],
        },
    },
}
_DYNAMIC_OFFSETS = ({"srcOffset": 5, "dstOffset": 3}, {"srcOffset": 0, "dstOffset": 7})
CASES = {
    "copy_s": {"tail": {"size": 37}, "exact": {"size": 64}},
    "copy_v": {"tail": {"size": 37}, "exact": {"size": 64}},
    "copy_s2": {"tail": {"grid": [3, 2]}, "wide": {"grid": [4, 3]}},
    "copy_v2": {"tail": {"grid": [3, 2]}, "wide": {"grid": [4, 3]}},
    **_GENERAL_CASES,
    **{
        kernel.replace("copy_gg", "copy_gg_dynamic"): {
            name: {**spec, **offsets}
            for (name, spec), offsets in zip(cases.items(), _DYNAMIC_OFFSETS)
        }
        for kernel, cases in _GENERAL_CASES.items()
        if kernel.startswith("copy_gg")
    },
}
# Metal ``[[buffer(n)]]`` slots; ``constant uint& size`` takes the free slot 2.
BUFFER_INDICES = {
    "src": 0,
    "dst": 1,
    "size": 2,
    "src_shape": 2,
    "src_stride": 3,
    "src_strides": 3,
    "dst_stride": 4,
    "dst_strides": 4,
    "ndim": 5,
    "src_offset": 6,
    "dst_offset": 7,
}
CONSTANT_FORMATS = {"uint32": "I", "int32": "i", "int64": "q"}
GUARD_COUNT = 16
DECOY_COUNT = 4
FLOAT_LAYOUTS = {"float16": (5, 10), "bfloat16": (8, 7), "float32": (8, 23)}
CANONICAL_NANS = {"float16": 0x7E00, "bfloat16": 0x7FC0, "float32": 0x7FC00000}
INTEGER_RANGES = {
    "uint8": (0, 2**8 - 1),
    "uint16": (0, 2**16 - 1),
    "uint32": (0, 2**32 - 1),
    "uint64": (0, 2**64 - 1),
    "int8": (-(2**7), 2**7 - 1),
    "int16": (-(2**15), 2**15 - 1),
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
}
# Boundary inputs: integer extremes and wrap points, values at and around
# every narrower range, float32 rounding ties and overflow thresholds of
# float16 and bfloat16, signed zeros, infinities, quiet and signaling NaNs,
# subnormals and the largest finite values.
EDGES = {
    "bool_": [0, 1],
    "uint8": [0, 1, 2, 0x7F, 0x80, 0xFE, 0xFF, 0x5A, 0xA5, 0x0F],
    "int8": [0, 1, -1, 2, -2, 127, -128, 126, -127, 90, -91],
    "uint16": (
        [0, 1, 0x7F, 0x80, 0xFF, 0x100, 0x7FFF, 0x8000, 0xFFFF]
        + [0x1234, 0xFEDC, 2049, 2051, 4097]
    ),
    "int16": (
        [0, 1, -1, 127, -128, 128, -129, 255, 256, 32767, -32768]
        + [2049, -2049, 4660, -4661]
    ),
    "uint32": (
        [0, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0x7FFFFFFF, 0x80000000]
        + [0xFFFFFFFF, 0x12345678, 0xFEDCBA98, 16777217, 16777219, 16842753]
        + [0x7FFFFF80, 65504, 65519, 2049]
    ),
    "int32": (
        [0, 1, -1, 127, -128, 255, 256, 32767, -32768, 65535, 65536]
        + [2**31 - 1, -(2**31), 16777217, -16777219, 16842753, -16842753]
        + [0x7FFFFF80, -0x7FFFFF80, 65504, -65504, 2049, -2051]
    ),
    "uint64": (
        [0, 1, 0xFF, 0x100, 0xFFFFFFFF, 2**32, 2**63 - 1, 2**63]
        + [2**64 - 1, 2**53 + 1, 0x0123456789ABCDEF, 0xFEDCBA9876543210, 65504]
        + [2049, 16842753, 2**63 + 2**39, 2**63 + 2**39 + 1, 2**63 + 2**55]
    ),
    "int64": (
        [0, 1, -1, 127, -128, 2**31, -(2**31) - 1, 2**32, 2**63 - 1]
        + [-(2**63), 2**53 + 1, -(2**53) - 1, 0x0123456789ABCDEF]
        + [-0x0123456789ABCDEF, 65504, -65504, 2049, -16842753]
    ),
    "float16": (
        [0x0000, 0x8000, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E00, 0x7D01]
        + [0xFE13, 0x0001, 0x83FF, 0x0400, 0x7BFF, 0xFBFF, 0x3E00, 0xBE00, 0x3800]
        + [0xBA00, 0x5BF8, 0x5BFC, 0x5C04, 0xD800, 0xD808, 0x57F0, 0x3555, 0x4248]
        + [0x3C01, 0x3C03]
    ),
    "float32": (
        [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000]
        + [0xFF800000, 0x7FC00000, 0x7FA00001, 0xFFC12345, 0x00000001, 0x807FFFFF]
        + [0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x3FC00000, 0xBF400000, 0x3F000000]
        + [0xBF7FFFFF, 0x437F0000, 0x437F8000, 0x43800000, 0x42FE0000, 0xC3000000]
        + [0xC3010000, 0x46FFFE00, 0xC7000000, 0x477FFF00, 0x47800000, 0x4EFFFFFF]
        + [0xCF000000, 0x4F000000, 0x4F7FFFFF, 0x5EFFFFFF, 0xDF000000, 0x5F7FFFFF]
        + [0x477FE000, 0x477FEFFF, 0x477FF000, 0x3F801000, 0x3F803000, 0x38800000]
        + [0x387FC000, 0x33800000, 0x33000000, 0x33400000, 0x35800000, 0x3F808000]
        + [0x3F818000, 0x3F80C000, 0x7F7F8000, 0x7F7F7FFF, 0x00018000, 0x00008000]
        + [0x4B800001, 0x40490FDB]
    ),
    "bfloat16": (
        [0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC0, 0x7F81]
        + [0xFFC5, 0x0001, 0x807F, 0x0080, 0x7F7F, 0xFF7F, 0x3FC0, 0xBF40, 0x3F00]
        + [0x437F, 0x4380, 0x42FE, 0xC300, 0xC301, 0x477F, 0x4780, 0x3F81, 0x3880]
        + [0x3380, 0x3300, 0x3340, 0x3381, 0x4F00, 0x4EFF, 0xCF00, 0x5F00, 0x5F7F]
        + [0xDF00, 0x4049]
    ),
    "complex64": [
        (0x3F800000, 0x80000000),
        (0x7FC00001, 0x40000000),
        (0x00000001, 0xFF800000),
        (0x7FA00001, 0x7F7FFFFF),
        (0x80000000, 0x00000000),
        (0x00000000, 0x80000000),
        (0x80000000, 0x00000001),
        (0xC0490FDB, 0x807FFFFF),
        (0x7F800000, 0xFFC12345),
        (0x437F8000, 0x3F800000),
        (0xBF400000, 0x40000000),
        (0x477FF000, 0x00000000),
        (0x3F801000, 0x3F800000),
        (0x3F818000, 0x00000000),
        (0x4F000000, 0x00000000),
        (0xC3010000, 0x00000000),
        (0x00018000, 0x00000000),
    ],
}
UNDEFINED = "undefined"
ANY_NAN = "any-nan"
# Scalar-source kinds read a single value per dispatch; give each its own.
SCALAR_SALTS = {"s": 0, "sn": 5, "s2": 11}


def _split(entry):
    """Return the kind, source dtype and destination dtype of an entry."""
    kind, types = entry.split("_copy", 1)
    for source in DTYPES:
        if types.startswith(source) and types[len(source) :] in DTYPES:
            return kind, source, types[len(source) :]
    raise AssertionError(entry)


def _kernel(entry):
    return KINDS[_split(entry)[0]][0]


def _work(entry):
    """Return the elements one thread of the entry copies."""
    kind, _, output = _split(entry)
    explicit = KINDS[kind][1]
    return explicit if explicit is not None else 8 // ITEM_SIZES[output]


def _float_value(dtype, word):
    """Decode float bits as ("nan"|"inf"|"finite", sign, exact magnitude)."""
    exponent_bits, mantissa_bits = FLOAT_LAYOUTS[dtype]
    sign = word >> (exponent_bits + mantissa_bits) & 1
    exponent = word >> mantissa_bits & ((1 << exponent_bits) - 1)
    mantissa = word & ((1 << mantissa_bits) - 1)
    if exponent == (1 << exponent_bits) - 1:
        return ("nan" if mantissa else "inf"), sign, None
    bias = (1 << (exponent_bits - 1)) - 1
    if exponent == 0:
        return "finite", sign, Fraction(mantissa, 1 << (bias + mantissa_bits - 1))
    significand = (1 << mantissa_bits) | mantissa
    return (
        "finite",
        sign,
        significand * Fraction(2) ** (exponent - bias - mantissa_bits),
    )


def _round_float(dtype, sign, magnitude):
    """Round an exact magnitude to nearest-even; return (bits, overflowed)."""
    exponent_bits, mantissa_bits = FLOAT_LAYOUTS[dtype]
    bias = (1 << (exponent_bits - 1)) - 1
    sign_bit = sign << (exponent_bits + mantissa_bits)
    infinity = sign_bit | ((1 << exponent_bits) - 1) << mantissa_bits
    if magnitude == 0:
        return sign_bit, False
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    exponent = max(exponent, 1 - bias)
    scaled = magnitude / Fraction(2) ** (exponent - mantissa_bits)
    rounded, remainder = divmod(scaled.numerator, scaled.denominator)
    if 2 * remainder > scaled.denominator or (
        2 * remainder == scaled.denominator and rounded & 1
    ):
        rounded += 1
    if rounded == 1 << (mantissa_bits + 1):
        rounded >>= 1
        exponent += 1
    if rounded < 1 << mantissa_bits:
        return sign_bit | rounded, False
    biased = exponent + bias
    if biased >= (1 << exponent_bits) - 1:
        return infinity, True
    return sign_bit | biased << mantissa_bits | (rounded - (1 << mantissa_bits)), False


def _wrap(dtype, value):
    bits = ITEM_SIZES[dtype] * 8
    value &= (1 << bits) - 1
    if INTEGER_RANGES[dtype][0] < 0 and value >> (bits - 1):
        value -= 1 << bits
    return value


def _cast(source, destination, value):
    """Model MLX ``cast_to<destination>(source value)``.

    Returns a logical destination value, ``ANY_NAN`` where only a NaN is
    defined, or ``UNDEFINED`` where C++ leaves the conversion undefined
    (non-finite or out-of-range float to integer, integer to float overflow).
    """
    if source == destination:
        return value
    if source == "complex64":
        if destination == "bool_":
            return int(bool(value[0] & 0x7FFFFFFF or value[1] & 0x7FFFFFFF))
        return _cast("float32", destination, value[0])
    if destination == "complex64":
        real = _cast(source, "float32", value)
        return UNDEFINED if real == UNDEFINED else (real, 0)
    if destination == "bool_":
        if source in FLOAT_LAYOUTS:
            # MLX tests float and bfloat16 bits; half compares with zero.
            exponent_bits, mantissa_bits = FLOAT_LAYOUTS[source]
            return int(bool(value & ((1 << (exponent_bits + mantissa_bits)) - 1)))
        return int(value != 0)
    if source in FLOAT_LAYOUTS:
        kind, sign, magnitude = _float_value(source, value)
    else:
        kind, sign, magnitude = "finite", int(value < 0), Fraction(abs(int(value)))
    if destination in INTEGER_RANGES:
        if source not in FLOAT_LAYOUTS:
            return _wrap(destination, int(value))
        if kind != "finite":
            return UNDEFINED
        truncated = -int(magnitude) if sign else int(magnitude)
        low, high = INTEGER_RANGES[destination]
        return truncated if low <= truncated <= high else UNDEFINED
    if kind == "nan":
        return ANY_NAN
    exponent_bits, mantissa_bits = FLOAT_LAYOUTS[destination]
    if kind == "inf":
        return (
            sign << (exponent_bits + mantissa_bits)
            | ((1 << exponent_bits) - 1) << mantissa_bits
        )
    bits, overflowed = _round_float(destination, sign, magnitude)
    # Float conversions round past the largest value to infinity; integer
    # conversions out of range are undefined.
    return UNDEFINED if overflowed and source not in FLOAT_LAYOUTS else bits


def _ordinary(dtype):
    """Return distinct unremarkable values that fill the rest of a buffer."""
    steps = [(index * 37 + 3) % 251 - 125 for index in range(128)]
    if dtype == "bool_":
        return []
    if dtype in INTEGER_RANGES:
        low = INTEGER_RANGES[dtype][0]
        return [step if low < 0 else step + 125 for step in steps]
    quarters = [
        _round_float("float32", int(step < 0), Fraction(abs(step), 4))[0]
        for step in steps
    ]
    if dtype == "float32":
        return quarters
    if dtype == "complex64":
        return list(zip(quarters, quarters[64:] + quarters[:64]))
    # Quarter integers up to 31.25 are exact in both 16-bit formats.
    return [
        _round_float(dtype, *_float_value("float32", word)[1:])[0] for word in quarters
    ]


@functools.lru_cache(maxsize=None)
def _domain(source, destination):
    """Return the source values whose cast to ``destination`` is defined."""
    values = []
    for value in EDGES[source] + _ordinary(source):
        if value not in values and _cast(source, destination, value) != UNDEFINED:
            values.append(value)
    return tuple(values)


def _layout(kernel, spec, work, name):
    """Return the grid, constants and the (destination, source) pairs written."""
    constants = {}
    if kernel in {"copy_s", "copy_v"}:
        size = spec["size"]
        grid = [-(-size // work), 1, 1]
        constants["size"] = ("uint32", [size])
        pairs = [(index, 0 if kernel == "copy_s" else index) for index in range(size)]
    elif kernel in {"copy_s2", "copy_v2"}:
        threads = spec["grid"][0] * spec["grid"][1]
        # A short last tile exercises the tail loop; one element per thread
        # has no tail, so the grid then covers the size exactly.
        size = threads * work - (1 if name == "tail" and work > 1 else 0)
        grid = [spec["grid"][0], spec["grid"][1], 1]
        constants["size"] = ("int64", [size])
        pairs = [(index, 0 if kernel == "copy_s2" else index) for index in range(size)]
    else:
        shape = spec["shape"]
        source_strides = spec["srcStrides"]
        destination_strides = spec.get("dstStrides")
        points = list(itertools.product(*(range(extent) for extent in shape)))
        if destination_strides is None:
            destinations = list(range(len(points)))
        else:
            destinations = [
                sum(i * s for i, s in zip(point, destination_strides))
                for point in points
            ]
        sources = [
            sum(i * s for i, s in zip(point, source_strides)) for point in points
        ]
        source_offset = spec.get("srcOffset", 0)
        destination_offset = spec.get("dstOffset", 0)
        pairs = [
            (destination + destination_offset, source + source_offset)
            for destination, source in zip(destinations, sources)
        ]
        if kernel in {"copy_g", "copy_gg", "copy_gg_dynamic"}:
            grid = [-(-shape[-1] // work), shape[-2], math.prod(shape[:-2])]
            constants["src_shape"] = ("int32", list(shape))
            constants["ndim"] = ("int32", [len(shape)])
        else:
            grid = list(reversed(shape)) + [1] * (3 - len(shape))
        # One-dimensional kernels take a scalar stride reference.
        stride = "stride" if kernel.endswith("_nd1") else "strides"
        constants[f"src_{stride}"] = ("int64", list(source_strides))
        if destination_strides is not None:
            constants[f"dst_{stride}"] = ("int64", list(destination_strides))
        if "srcOffset" in spec:
            constants["src_offset"] = ("int64", [source_offset])
            constants["dst_offset"] = ("int64", [destination_offset])
    return grid, constants, pairs


def _different(dtype, value):
    """Return a value of ``dtype`` that differs from ``value``."""
    guard = GUARDS[dtype]
    if dtype == "complex64":
        return guard if value != guard else (guard[0] ^ 1, guard[1])
    return guard if value != guard else guard ^ 1


def _case(entry, name, target="metal"):
    """Return the dispatch, buffers and expected output of one case."""
    kind, source, destination = _split(entry)
    kernel = KINDS[kind][0]
    work = _work(entry)
    names = list(CASES[kernel])
    grid, constants, pairs = _layout(kernel, CASES[kernel][name], work, name)
    domain = _domain(source, destination)
    # Read elements take consecutive domain values in first-read order, so a
    # case starts at the boundary values and the second case continues where
    # it left off. Scalar sources read one value; each kind takes another.
    salt = names.index(name) * 23 + SCALAR_SALTS.get(kind, 0)
    source_extent = max(src for _, src in pairs) + 1
    order = list(dict.fromkeys(src for _, src in pairs))
    read = set(order)
    order += [index for index in range(source_extent) if index not in read]
    values = [None] * source_extent
    for rank, index in enumerate(order):
        values[index] = domain[(salt + rank) % len(domain)]
    decoys = (
        [1, 0] * (DECOY_COUNT // 2)
        if source == "bool_"
        else [GUARDS[source]] * DECOY_COUNT
    )
    destination_extent = max(dst for dst, _ in pairs) + 1
    written = {dst: _cast(source, destination, values[src]) for dst, src in pairs}
    assert len(written) == len(pairs), "every destination is written once"
    initial, expected = [], []
    for index in range(destination_extent + GUARD_COUNT):
        if destination == "bool_" and target == "metal":
            # One-byte Metal bools hold canonical values only.
            first = 1 - written[index] if index in written else index % 2
        elif index in written:
            first = _different(destination, written[index])
        else:
            first = GUARDS[destination]
        initial.append(first)
        expected.append(written.get(index, first))
    return {
        "grid": grid,
        "constants": constants,
        "source": values + decoys,
        "initial": initial,
        "expected": expected,
        "pairs": pairs,
        "writtenCount": len(pairs),
    }


def _encoded(target, dtype, values):
    """Encode logical values; return the runtime value and its any-NaN slots."""
    width = 2 if dtype == "complex64" else 1
    nans = []
    concrete = []
    for index, value in enumerate(values):
        if dtype == "complex64" and value[0] == ANY_NAN:
            nans.append(index * width)
            value = (CANONICAL_NANS["float32"], value[1])
        elif value == ANY_NAN:
            nans.append(index)
            value = CANONICAL_NANS[dtype]
        concrete.append(value)
    return _runtime_value(target, dtype, concrete), nans


def _nan_bits(target, dtype):
    """Return the word format that represents ``dtype`` on ``target``."""
    if dtype not in {"float16", "bfloat16"}:
        return 32
    encoded = _runtime_value(target, dtype, [CANONICAL_NANS[dtype]])
    return 32 if encoded["dtype"] == "float32" else dtype


def _is_nan(bits, word):
    if bits == 32:
        return (word & 0x7FFFFFFF) > 0x7F800000
    exponent_bits, mantissa_bits = FLOAT_LAYOUTS[bits]
    infinity = ((1 << exponent_bits) - 1) << mantissa_bits
    return (word & ((1 << (exponent_bits + mantissa_bits)) - 1)) > infinity


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
    """Return the Metal words of logical values and their any-NaN slots."""
    width = 2 if dtype == "complex64" else 1
    words, nans = [], []
    for index, value in enumerate(values):
        if dtype == "complex64":
            if value[0] == ANY_NAN:
                nans.append(index * width)
                value = (CANONICAL_NANS["float32"], value[1])
            words.extend(value)
        elif value == ANY_NAN:
            nans.append(index)
            words.append(CANONICAL_NANS[dtype])
        else:
            words.append(int(value))
    return words, nans


def _config(target, entry):
    """Return the complete-family translation configuration of an entry."""
    workload = WORKLOADS[target][entry]
    return {
        "metal": complete_metal._project_config,
        "opengl": complete_opengl._project_config,
        "directx": complete_directx._project_config,
    }[target](workload)


WORKLOADS = {
    "metal": {item.entry_point: item for item in complete_metal.COPY_METAL_WORKLOADS},
    "opengl": {
        item.entry_point: item for item in complete_opengl.COPY_OPENGL_WORKLOADS
    },
    "directx": {
        item.entry_point: item for item in complete_directx.COPY_DIRECTX_WORKLOADS
    },
}
ARTIFACT_CONTRACTS = {
    "metal": complete_metal.COPY_METAL_CONTRACT_PATH,
    "opengl": complete_opengl.COPY_OPENGL_CONTRACT_PATH,
    "directx": complete_directx.COPY_DIRECTX_CONTRACT_PATH,
}


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"copy family shard {index} is outside [0, {count})")
    return ENTRIES[index::count]


SELECTED_ENTRIES = _selected_entries()


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _dispatch_count():
    return sum(len(CASES[_kernel(entry)]) for entry in ENTRIES)


def _element_types():
    return {
        target: {dtype: _element_type(target, dtype) for dtype in DTYPES}
        for target in TARGETS
    }


def _index_range_assertions():
    return {
        "metal": [],
        "opengl": [
            {"expression": expression, "minimum": low, "maximum": high}
            for expression, low, high in complete_opengl.INDEX_RANGE_ASSERTIONS
        ],
        "directx": [],
    }


def expected_contract():
    """Return the contract this module pins, for review and regeneration."""
    return {
        "schemaVersion": 1,
        "kind": "crosstl-mlx-current-tree-copy-family-contract",
        "provenance": {
            "repository": "https://github.com/ml-explore/mlx.git",
            "commit": MLX_COMMIT,
            "source": SOURCE,
            "sourceSha256": SOURCE_SHA256,
            "header": HEADER,
            "headerSha256": HEADER_SHA256,
            "utilsHeader": UTILS_HEADER,
            "utilsHeaderSha256": UTILS_HEADER_SHA256,
        },
        "corpusCensus": {
            "metalUnitCount": 49,
            "entryCount": 17832,
            "discoveryDiagnosticCount": 0,
        },
        "entries": list(ENTRIES),
        # copy.metal and copy.h are byte-identical at the corpus commit, and
        # each entry must reproduce the artifact its complete-family contract
        # pins there.
        "artifactContracts": {
            target: {
                "path": ARTIFACT_CONTRACTS[target].relative_to(ROOT).as_posix(),
                "commit": complete_metal.MLX_COMMIT,
                "sourceSha256": SOURCE_SHA256,
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
            "castModel": {
                "sameType": "bit-exact",
                "rounding": (
                    "IEEE-754 binary16, bfloat16 and binary32, "
                    "round to nearest, ties to even, from the exact source value"
                ),
                "floatOverflow": "infinity",
                "integerToFloatOverflow": "excluded as undefined",
                "floatToInteger": (
                    "truncate toward zero; non-finite or "
                    "out-of-range inputs excluded as undefined"
                ),
                "integerToInteger": "two's complement wrap",
                "nan": "any NaN when the types differ",
                "toBool": "nonzero, float32, bfloat16 and complex64 by bits",
                "complex": "real part to scalars, zero imaginary from scalars",
            },
            "compilerValidationRequiredInCi": list(TARGETS),
            "runtimeRequiredInCi": list(TARGETS),
            "upstreamMetallibParityTargets": ["metal"],
            "ciShardCount": CI_SHARD_COUNT,
        },
        "scope": {
            "coveredEntryCount": len(ENTRIES),
            "runtimeCoveredEntryCount": len(ENTRIES),
            "dispatchCountByTarget": {target: _dispatch_count() for target in TARGETS},
            "completeCorpusEntryCount": 17832,
            "upstreamMlxTestSuiteExecuted": False,
            "mlxHostRuntimeRedirectionImplemented": False,
        },
    }


def test_copy_entries_match_the_upstream_instantiation():
    assert len(ENTRIES) == 2496 and len(set(ENTRIES)) == 2496
    assert len({_split(entry)[1:] for entry in ENTRIES}) == 169
    assert sum(1 for entry in ENTRIES if _split(entry)[0] in SAME_TYPE_KINDS) == 208
    assert set(KINDS) == set(SAME_TYPE_KINDS + BASE_KINDS + WORK_KINDS)
    works = {kind: set() for kind in KINDS}
    for entry in ENTRIES:
        works[_split(entry)[0]].add(_work(entry))
    assert works["vn"] == works["sn"] == {2, 4, 8}
    assert works["v2"] == works["s2"] == {1, 2, 4, 8}
    assert works["gn4large"] == works["ggn4large_dynamic"] == {4}
    assert all(works[kind] == {1} for kind in ("s", "v", "g3large", "gg2"))
    for target in TARGETS:
        assert set(WORKLOADS[target]) == set(ENTRIES)


def test_current_mlx_copy_family_contract_is_exact():
    assert _contract() == json.loads(json.dumps(expected_contract()))


def test_copy_family_reuses_the_complete_translation_identities():
    """Each entry reproduces the artifact its complete-family contract pins."""
    for target, path in ARTIFACT_CONTRACTS.items():
        contract = json.loads(path.read_text(encoding="utf-8"))
        assert contract["commit"] == complete_metal.MLX_COMMIT
        assert contract["source"] == SOURCE
        assert contract["sourceSha256"] == SOURCE_SHA256
        assert {item["entryPoint"] for item in contract["entries"]} == set(ENTRIES)
        for entry in ENTRIES:
            workload = WORKLOADS[target][entry]
            kind, source, destination = _split(entry)
            assert workload.shape == kind, entry
            assert len(workload.sha256) == 64 and workload.size_bytes > 0
    assert complete_metal.MLX_COPY_SHA256 == SOURCE_SHA256


@pytest.mark.parametrize("target", TARGETS)
def test_copy_family_configs_match_the_complete_contracts(tmp_path, target):
    entry = "gn4large_copyuint8float16"
    path = tmp_path / "crosstl.toml"
    path.write_text(_config(target, entry) + "\n", encoding="utf-8")
    config = load_project_config(tmp_path, path)
    assert list(config.targets) == [target]
    assert dict(config.entry_points) == {SOURCE: entry}
    assert [item.to_json() for item in config.index_range_assertions] == [
        {"source": SOURCE, **item} for item in _index_range_assertions()[target]
    ]
    assert not config.metadata_preconditions


@pytest.mark.parametrize(
    "dtype,word",
    [
        ("float16", word)
        for word in (0x0000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x7BFF, 0xFBFF, 0x8001)
    ]
    + [
        ("float32", word)
        for word in (0x00000001, 0x007FFFFF, 0x3F800000, 0x7F7FFFFF, 0xC0490FDB)
    ],
)
def test_float_codec_round_trips_every_finite_class(dtype, word):
    kind, sign, magnitude = _float_value(dtype, word)
    assert kind == "finite"
    assert _round_float(dtype, sign, magnitude) == (word, False)


def test_rounding_matches_the_host_binary16_and_binary32_codecs():
    # The host codecs round from binary64, which holds these values exactly.
    for numerator in range(-65519, 65520, 97):
        for denominator in (1, 1024, 2**20, 2**36):
            value = Fraction(numerator, denominator)
            sign, magnitude = int(value < 0), abs(value)
            half = struct.unpack("<H", struct.pack("<e", float(value)))[0]
            single = struct.unpack("<I", struct.pack("<f", float(value)))[0]
            assert _round_float("float16", sign, magnitude)[0] == half, value
            assert _round_float("float32", sign, magnitude)[0] == single, value
    assert _round_float("float16", 0, Fraction(65520)) == (0x7C00, True)
    assert _round_float("float16", 1, Fraction(65519)) == (0xFBFF, False)


def test_bfloat16_rounding_matches_the_bit_level_reference():
    for word in list(range(0, 2**32, 0x10001 * 37)) + [0x3F808000, 0x3F818000]:
        kind, sign, magnitude = _float_value("float32", word)
        if kind != "finite":
            continue
        reference = (word + 0x7FFF + (word >> 16 & 1)) >> 16
        assert _round_float("bfloat16", sign, magnitude)[0] == reference, hex(word)


@pytest.mark.parametrize(
    "source,destination,value,expected",
    [
        ("float32", "float16", 0x3F801000, 0x3C00),
        ("float32", "float16", 0x3F803000, 0x3C02),
        ("float32", "float16", 0x477FF000, 0x7C00),
        ("float32", "float16", 0x33400000, 0x0001),
        ("float32", "float16", 0xFFC12345, ANY_NAN),
        ("float32", "bfloat16", 0x7F7F8000, 0x7F80),
        ("float32", "bfloat16", 0x00018000, 0x0002),
        ("int32", "float16", 2049, 0x6800),
        ("int32", "float16", 70000, UNDEFINED),
        ("int32", "bfloat16", 16842753, 0x4B81),
        ("uint64", "bfloat16", 2**63 + 2**55, 0x5F00),
        ("int32", "float32", 16777217, 0x4B800000),
        ("uint64", "float32", 2**64 - 1, 0x5F800000),
        ("uint64", "float32", 2**63 + 2**39 + 1, 0x5F000001),
        ("float32", "uint8", 0xBF400000, 0),
        ("float32", "uint8", 0x437F8000, 255),
        ("float32", "int8", 0xC3010000, UNDEFINED),
        ("float32", "int32", 0x7FC00000, UNDEFINED),
        ("float16", "uint8", 0x5BFC, 255),
        ("int64", "uint8", -1, 255),
        ("uint16", "int8", 0x80, -128),
        ("float32", "bool_", 0x80000000, 0),
        ("float32", "bool_", 0x7FC00000, 1),
        ("float16", "bool_", 0x0001, 1),
        ("bfloat16", "bool_", 0x8000, 0),
        ("complex64", "bool_", (0x80000000, 0x00000001), 1),
        ("complex64", "bool_", (0x80000000, 0x80000000), 0),
        ("complex64", "int16", (0xC3010000, 0x7FC00000), -129),
        ("bool_", "complex64", 1, (0x3F800000, 0)),
        ("int8", "complex64", -2, (0xC0000000, 0)),
        ("bfloat16", "float16", 0x4780, 0x7C00),
        ("bfloat16", "float16", 0x3381, 0x0001),
        ("float16", "bfloat16", 0x7BFF, 0x4780),
        ("float16", "float32", 0x83FF, 0xB87FC000),
        ("float16", "float32", 0x7D01, ANY_NAN),
        ("float16", "float16", 0x7D01, 0x7D01),
    ],
)
def test_cast_model_matches_upstream_metal_conversions(
    source, destination, value, expected
):
    """Pin rounding, overflow, truncation, wrap, bool and NaN boundaries.

    The unchanged kernels produce these results on Apple GPUs; the native
    gate compares every modelled element against them on Metal.
    """
    assert _cast(source, destination, value) == expected


@pytest.mark.parametrize("kernel", CASES)
def test_copy_layouts_write_each_destination_once(kernel):
    works = {_work(entry) for entry in ENTRIES if _kernel(entry) == kernel}
    padded = False
    for work in sorted(works):
        for name, spec in CASES[kernel].items():
            grid, constants, pairs = _layout(kernel, spec, work, name)
            destinations = [dst for dst, _ in pairs]
            assert len(set(destinations)) == len(destinations)
            assert min(destinations) >= 0 and min(src for _, src in pairs) >= 0
            assert all(extent > 0 for extent in grid)
            padded |= max(destinations) + 1 > len(destinations)
            if kernel in {"copy_s", "copy_v"}:
                size = constants["size"][1][0]
                assert (grid[0] - 1) * work < size <= grid[0] * work
            if kernel in {"copy_g", "copy_gg", "copy_gg_dynamic"}:
                # The innermost extent ends inside the last tile of a thread.
                assert spec["shape"][-1] % work != 0
            if "srcOffset" in spec:
                assert min(src for _, src in pairs) == spec["srcOffset"]
                assert min(destinations) == spec["dstOffset"]
        if kernel in {"copy_s", "copy_v", "copy_s2", "copy_v2"} and work > 1:
            tails = [
                constants["size"][1][0] % work
                for name, spec in CASES[kernel].items()
                for _, constants, _ in [_layout(kernel, spec, work, name)]
            ]
            assert any(tails) and not all(tails)
    # Strided destinations leave padding the kernel must not touch.
    assert padded is kernel.startswith("copy_gg")


@pytest.mark.parametrize("pair", sorted({_split(entry)[1:] for entry in ENTRIES}))
def test_copy_cases_cover_every_defined_boundary(pair):
    source, destination = pair
    entries = [entry for entry in ENTRIES if _split(entry)[1:] == pair]
    covered = []
    for entry in entries:
        for name in CASES[_kernel(entry)]:
            case = _case(entry, name)
            covered.extend(case["source"][src] for _, src in case["pairs"])
            assert len(case["expected"]) == len(case["initial"])
            written = {dst for dst, _ in case["pairs"]}
            for index, (first, final) in enumerate(
                zip(case["initial"], case["expected"])
            ):
                # Unwritten elements keep their value; written ones change.
                assert (first != final) is (index in written)
    boundaries = [
        value
        for value in EDGES[source]
        if _cast(source, destination, value) != UNDEFINED
    ]
    missing = [value for value in boundaries if value not in covered]
    assert not missing, missing


@pytest.mark.parametrize(
    "fault", ["value", "missing", "guard", "nan", "zero-sign", "rounding"]
)
def test_copy_readback_rejects_corruption(fault):
    entry = "v_copyfloat32float16"
    case = _case(entry, "tail")
    wanted, nans = _metal_wanted("float16", case["expected"])
    assert nans
    actual = list(wanted)
    finite = next(i for i, w in enumerate(wanted) if i not in nans and w & 0x7FFF)
    if fault == "value":
        actual[finite] ^= 0x0400
    elif fault == "missing":
        actual[finite] = case["initial"][finite]
    elif fault == "guard":
        actual[-1] ^= 1
    elif fault == "nan":
        actual[nans[0]] = 0x7C00
    elif fault == "zero-sign":
        zero = next(i for i, w in enumerate(wanted) if w == 0x8000)
        actual[zero] = 0
    else:
        actual[finite] += 1
    with pytest.raises(AssertionError):
        _check(actual, wanted, nans, "float16")
    _check(wanted, wanted, nans, "float16")


def test_copy_shards_partition_the_family(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert {len(shard) for shard in shards} == {416}
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native copy family gate")
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
        (UTILS_HEADER, UTILS_HEADER_SHA256),
    ):
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("copy-family-metal-reference")
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
    _, source, destination = _split(entry)
    payloads = {
        0: _metal_bytes(source, case["source"]),
        1: _metal_bytes(destination, case["initial"]),
    }
    for parameter, (dtype, values) in case["constants"].items():
        payloads[BUFFER_INDICES[parameter]] = struct.pack(
            f"<{len(values)}{CONSTANT_FORMATS[dtype]}", *values
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
    _, source, destination = _split(entry)
    output, nans = _encoded(target, destination, case["expected"])
    data = {
        "src": _runtime_value(target, source, case["source"]),
        "dst": _runtime_value(target, destination, case["initial"]),
    }
    for parameter, (dtype, values) in case["constants"].items():
        data[parameter] = {"dtype": dtype, "shape": [len(values)], "values": values}
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
        if name == "dst":
            output_name = binding["name"]
        else:
            observed.append(binding["name"])
    assert output_name is not None and len(inputs) == len(data), sorted(inputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output_name: dict(data["dst"])},
        {"workgroupCount": case["grid"], "workgroupSize": [1, 1, 1]},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output_name, output["values"], nans, len(observed)


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_current_mlx_copy_family_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    _, source, destination = _split(entry)
    workload = WORKLOADS[target][entry]
    bits = _nan_bits(target, destination)
    metal_bits = _nan_bits("metal", destination)
    with tempfile.TemporaryDirectory(prefix=".current-copy-", dir=root) as directory:
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
            check_generated_artifact(
                artifact, sha256=workload.sha256, size_bytes=workload.size_bytes
            )
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
                    if destination == "bool_" and target == "metal":
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
                        words, metal_nans = _metal_wanted(destination, case["expected"])
                        _check(
                            _metal_words(destination, raw),
                            words,
                            metal_nans,
                            metal_bits,
                        )
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
                "artifactSha256": workload.sha256,
                "cases": evidence,
                "upstreamMlxTestSuiteExecuted": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_ci_requires_native_copy_family_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX copy family native parity",
        family="copy",
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
        "demos/integrations/mlx/tests/kernels/test_current_copy_family.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_copy_family.py"
    )
