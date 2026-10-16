"""Pinned MLX Winograd transforms on Metal, OpenGL and DirectX.

``conv.cpp``'s ``winograd_conv_2D_gpu`` computes stride-one 3x3 convolutions as
F(6x6, 3x3) Winograd products. ``winograd_conv_2d_weight_transform`` moves
every filter slice into the 8x8 Winograd domain as ``G g G^T``,
``winograd_conv_2d_input_transform`` moves every overlapping 8x8 input tile as
``B^T d B``, and after the batched GEMM ``winograd_conv_2d_output_transform``
returns every tile to a 6x6 output block as ``A^T m A``. Each SIMD-group forms
these with two ``simdgroup_matrix<float, 8, 8>`` products per channel. This
gate translates all nine entries, the three transforms for float32, bfloat16
and float16, to Metal, OpenGL and DirectX and executes them natively with
``conv.cpp``'s launch geometry and ``MLXConvParams<2>`` record.

OpenGL and DirectX lower each product as the software multiply-accumulate from
a +0.0 accumulator with fused, ties-to-even binary32 steps in increasing k,
which Apple's SIMD-group matrices match. ``G`` holds non-dyadic constants such
as -2/9 and ``B`` constants such as 21/4, so products and sums round: the
reference words come from an exact fused multiply-add oracle and every output
word is compared exactly. ``A`` holds powers of two, so the output transform's
products are exact and it checks summation order and rounding, not fusion.

Every channel loop holds barriers, so OpenGL and DirectX check metadata
preconditions that bound it: the weight transform's ``C`` and the ``params.C``
and ``params.O`` members of the input and output transforms' records. Those
transforms' matrix loops start at the SIMD-group index and step by the
SIMD-group count, with equal trip counts in every SIMD-group, which
translation proves for the software subgroup barriers.
"""

import functools
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import zlib
from fractions import Fraction
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    load_project_config,
    pack_storage_records,
    pack_uniform_block,
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
from demos.integrations.mlx.tests.kernels.test_current_conv import (
    HEADERS,
    PARAMS_BLOCK_SIZES,
    PARAMS_SIZES,
    SOURCE,
    SOURCE_SHA256,
    WINOGRAD_ENTRIES,
    _contiguous,
    _params_bytes,
)
from demos.integrations.mlx.tests.kernels.test_current_cross_entropy import (
    _bits_float32,
    _float32_bits,
    _quantize,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _element_type,
    _metal_bytes,
    _runtime_value,
)
from demos.integrations.mlx.tests.kernels.test_current_sort import (
    _assert_words,
    _execute,
)
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_fused_math import _oracle
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_WINOGRAD"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/winograd.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
# instantiate_winograd_conv_2d emits the weight, input and output transforms
# for float, bfloat16_t and half, in that order.
DTYPES = ("float32", "bfloat16", "float16")
STAGES = ("weight", "input", "output")
ENTRIES = WINOGRAD_ENTRIES
ENTRY_PATTERN = re.compile(
    r"winograd_conv_2d_(?P<stage>weight|input|output)_transform_"
    r"(?P<dtype>float32|float16|bfloat16)_(?P<block>bc|bo)32"
)
SIMD_SIZE = 32
SIGNED_MAX = 2**31 - 1
# Every transform moves 32 channels per block: bc for the weight and input
# transforms, bo for the output transform.
BLOCK_CHANNELS = 32
# winograd_conv_2D_gpu: the weight transform runs bo = 4 filters, one per
# SIMD-group, in (32, bo, 1) groups over (O / bo, 1, 1). The input and output
# transforms run (32, wn, wm) = (32, 2, 2) groups over (tiles_w, tiles_h, N):
# each SIMD-group loads or stores one quadrant of the tile and transforms
# every fourth channel.
GROUP_FILTERS = 4
TILE_SPLIT = 2
SIMD_GROUPS = 4
WORKGROUPS = {
    "weight": (SIMD_SIZE, GROUP_FILTERS, 1),
    "input": (SIMD_SIZE, TILE_SPLIT, TILE_SPLIT),
    "output": (SIMD_SIZE, TILE_SPLIT, TILE_SPLIT),
}
GRID_RULES = {
    "weight": "[O / 4, 1, 1] groups",
    "input": "[tiles_w, tiles_h, N] groups",
    "output": "[tiles_w, tiles_h, N] groups",
}
FILTER = 3
OUT_TILE = 6
TILE = 8


def _table(*rows):
    """A ``WinogradTransforms<6, 3, 8>`` table as exact rationals; Metal
    zero-fills the columns a row leaves out."""
    return tuple(
        tuple(Fraction(item) for item in row) + (Fraction(0),) * (TILE - len(row))
        for row in rows
    )


# G: wt_transform, with three live columns.
WT_TRANSFORM = _table(
    (1, 0, 0),
    ("-2/9", "-2/9", "-2/9"),
    ("-2/9", "2/9", "-2/9"),
    ("1/90", "1/45", "2/45"),
    ("1/90", "-1/45", "2/45"),
    ("32/45", "16/45", "8/45"),
    ("32/45", "-16/45", "8/45"),
    (0, 0, 1),
)
# B: in_transform.
IN_TRANSFORM = _table(
    (1, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, -1, "1/2", "-1/2", 2, -2, -1),
    ("-21/4", 1, 1, "1/4", "1/4", 4, 4, 0),
    (0, "-17/4", "17/4", "-5/2", "5/2", "-5/2", "5/2", "21/4"),
    ("21/4", "-17/4", "-17/4", "-5/4", "-5/4", -5, -5, 0),
    (0, 1, -1, 2, -2, "1/2", "-1/2", "-21/4"),
    (-1, 1, 1, 1, 1, 1, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 1),
)
# A: out_transform, with six live columns.
OUT_TRANSFORM = _table(
    (1, 0, 0, 0, 0, 0),
    (1, 1, 1, 1, 1, 1),
    (1, -1, 1, -1, 1, -1),
    (1, 2, 4, 8, 16, 32),
    (1, -2, 4, -8, 16, -32),
    (1, "1/2", "1/4", "1/8", "1/16", "1/32"),
    (1, "-1/2", "1/4", "-1/8", "1/16", "-1/32"),
    (0, 0, 0, 0, 0, 1),
)
# Each transform's table: its conv.metal name, values and live columns.
TABLES = {
    "weight": ("wt_transform", WT_TRANSFORM, FILTER),
    "input": ("in_transform", IN_TRANSFORM, TILE),
    "output": ("out_transform", OUT_TRANSFORM, OUT_TILE),
}
# conv.metal's two products per channel. The output transform names its
# out_transform fragment B, which is the A above.
PRODUCTS = {
    "weight": "(G * g) * Gt",
    "input": "(Bt * I) * B",
    "output": "At * (O * A)",
}
# The kernels index lanes the way Apple lays out an 8x8 SIMD-group matrix:
# four 4x4 tiles in row-major order, each lane holding a row pair.
FRAGMENT_MAPPING = "tile_4x4_row_pair"
FRAGMENT_MAPPING_PROVENANCE = "mlx_conv_winograd_lane_coordinates"
SOFTWARE_MATRIX_TARGETS = ("opengl", "directx")
MMA_PROFILE = "fma-rne-gradual"
BUFFER_ORDER = {
    "weight": ("wt_in", "wt_out", "C", "O"),
    "input": ("inp_in", "inp_out", "params"),
    "output": ("out_in", "out_out", "params"),
}
INPUT_BUFFER = {"weight": "wt_in", "input": "inp_in", "output": "out_in"}
OUTPUT_BUFFER = {"weight": "wt_out", "input": "inp_out", "output": "out_out"}
# Every channel loop, `for (int bc = 0; bc < C; bc += 32)` and its params.C
# and params.O forms, holds barriers, so OpenGL and DirectX must prove it
# terminates without overflow: the bound is at most INT_MAX - 31.
LOOP_BOUND = {"minimum": 0, "maximum": SIGNED_MAX - 31}
METADATA_PRECONDITIONS = {
    "weight": {"parameter": "C", **LOOP_BOUND},
    "input": {"parameter": "params", "member": "C", **LOOP_BOUND},
    "output": {"parameter": "params", "member": "O", **LOOP_BOUND},
}
# MLXConvParams<2> starts with int N, C and O on every target's layout.
PARAMS_MEMBER_OFFSETS = {"N": 0, "C": 4, "O": 8}
# OpenGL indexes 32-bit storage arrays. These host preconditions bound the
# composed element offsets to signed int range; the cases stay far below it.
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "weight": (
        "ko * 3 * 3 * C",
        "ohw_0 * C * O + ko",
        "ohw_1 * C * O + ko",
        "wt_in + kh * 3 * C + kw * C + kc",
        "wt_out_0 + c * O",
        "wt_out_1 + c * O",
    ),
    "input": (
        "tid.z * params.in_strides[0] + bh * params.in_strides[1]"
        " + bw * params.in_strides[2]",
        "ohw_0 * N_TILES * params.C + tile_id * params.C",
        "ohw_1 * N_TILES * params.C + tile_id * params.C",
        "in_ptr + c",
        "inp_out_0 + c",
        "inp_out_1 + c",
    ),
    "output": (
        "tid.z * params.out_strides[0] + bh * params.out_strides[1]"
        " + bw * params.out_strides[2]",
        "ohw_0 * N_TILES * params.O + tile_id * params.O",
        "ohw_1 * N_TILES * params.O + tile_id * params.O",
        "out_in_0 + c",
        "out_in_1 + c",
        "out_ptr + c",
    ),
}
# Cases: name -> batch N, output extent oS, channels C and filters O.
# conv.cpp only takes Winograd when C and O are multiples of 32. The first
# case has whole 6x6 output tiles; the second leaves partial tiles in both
# directions for the output transform's bounds and runs three channel blocks
# through the weight and input transforms. Both tile grids are wider than they
# are tall, and C != O, so swapped tile coordinates or a transform bounded by
# the other count fail.
CASES = {
    "whole-tiles": {"N": 1, "oS": (12, 18), "C": 32, "O": 64},
    "partial-tiles": {"N": 2, "oS": (10, 15), "C": 96, "O": 32},
}
# Slices with known transforms, as (filter or tile, channel): all -0.0 inputs
# must give +0.0 words, since every product chain starts from +0.0, and a unit
# impulse at (kh, kw) gives left[r][kh] * right[kw][s], the outer product of a
# column of the left factor and a row of the right one. Tile 0 is a whole
# output tile in every case.
NEGATIVE_ZERO_SLICE = {"weight": (1, 3), "input": (0, 3), "output": (0, 3)}
IMPULSE_SLICE = {"weight": (2, 5), "input": (1, 5), "output": (0, 5)}
IMPULSE_TAP = {"weight": (1, 2), "input": (2, 3), "output": (3, 4)}
# -2^15 is exact in every type and no transform reaches it.
GUARD_VALUE = -32768.0
GUARD_COUNT = 16
INPUT_GUARD_COUNT = 16


def _stage(entry):
    return ENTRY_PATTERN.fullmatch(entry).group("stage")


def _dtype(entry):
    return ENTRY_PATTERN.fullmatch(entry).group("dtype")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _ceil_div(numerator, denominator):
    return -(-numerator // denominator)


@functools.lru_cache(maxsize=None)
def _geometry(name):
    """conv.cpp's padded input, tiling, grids and ``MLXConvParams<2>``."""
    case = CASES[name]
    batch, channels, filters = case["N"], case["C"], case["O"]
    out_h, out_w = case["oS"]
    tiles = (_ceil_div(out_h, OUT_TILE), _ceil_div(out_w, OUT_TILE))
    # winograd_padded_size, 6 * ceildiv(iS + 2 * pad - 2, 6) + 2, for a valid
    # convolution's unpadded iS = oS + 2 input.
    padded = tuple(
        OUT_TILE * _ceil_div(extent + FILTER - 1 - 2, OUT_TILE) + 2
        for extent in (out_h, out_w)
    )
    in_shape = (batch, padded[0], padded[1], channels)
    out_shape = (batch, out_h, out_w, filters)
    tile_grid = [tiles[1], tiles[0], batch]
    return {
        "tiles": tiles,
        "tileCount": batch * tiles[0] * tiles[1],
        "inShape": in_shape,
        "outShape": out_shape,
        # conv_params_updated: stride one, no padding or dilation.
        "params": {
            "N": batch,
            "C": channels,
            "O": filters,
            "iS": list(padded),
            "wS": [FILTER, FILTER],
            "oS": [out_h, out_w],
            "str": [1, 1],
            "pad": [0, 0],
            "kdil": [1, 1],
            "idil": [1, 1],
            "in_strides": _contiguous(in_shape),
            "wt_strides": _contiguous((filters, FILTER, FILTER, channels)),
            "out_strides": _contiguous(out_shape),
            "groups": 1,
            "flip": False,
        },
        "grids": {
            "weight": [filters // GROUP_FILTERS, 1, 1],
            "input": tile_grid,
            "output": tile_grid,
        },
    }


def _tile_position(name, tile):
    """conv.metal's tile_id = tid.z * gx * gy + tid.y * gx + tid.x."""
    tiles_h, tiles_w = _geometry(name)["tiles"]
    batch, rest = divmod(tile, tiles_h * tiles_w)
    tile_y, tile_x = divmod(rest, tiles_w)
    return batch, tile_y, tile_x


def _tap_index(channels, filter_index, kh, kw, channel):
    """conv.cpp's (O, 3, 3, C) weight layout."""
    return ((filter_index * FILTER + kh) * FILTER + kw) * channels + channel


def _weight_output_index(channels, filters, row, column, channel, filter_index):
    """wt_out is stored transposed: (8 * 8, C, O)."""
    return ((row * TILE + column) * channels + channel) * filters + filter_index


def _input_index(name, tile, row, column, channel):
    """Element (row, column) of an 8x8 input tile in the padded (N, H, W, C)
    input; neighbouring tiles overlap by two rows or columns."""
    strides = _geometry(name)["params"]["in_strides"]
    batch, tile_y, tile_x = _tile_position(name, tile)
    return (
        batch * strides[0]
        + (OUT_TILE * tile_y + row) * strides[1]
        + (OUT_TILE * tile_x + column) * strides[2]
        + channel
    )


def _domain_index(name, channels, row, column, tile, channel):
    """inp_out and out_in interleave the tiles: (8 * 8, tiles, channels)."""
    tiles = _geometry(name)["tileCount"]
    return ((row * TILE + column) * tiles + tile) * channels + channel


def _output_index(name, tile, row, column, channel):
    """Element (row, column) of a 6x6 output tile in out_out's (N, oH, oW, O),
    or None outside the tile or past oS."""
    params = _geometry(name)["params"]
    batch, tile_y, tile_x = _tile_position(name, tile)
    y, x = OUT_TILE * tile_y + row, OUT_TILE * tile_x + column
    if row >= OUT_TILE or column >= OUT_TILE:
        return None
    if y >= params["oS"][0] or x >= params["oS"][1]:
        return None
    strides = params["out_strides"]
    return batch * strides[0] + y * strides[1] + x * strides[2] + channel


def _shape(stage, name):
    """(slices, channels): the filters or tiles a transform visits, and the
    channels of each."""
    case, geometry = CASES[name], _geometry(name)
    if stage == "weight":
        return case["O"], case["C"]
    return geometry["tileCount"], case["C" if stage == "input" else "O"]


def _input_count(stage, name):
    case, geometry = CASES[name], _geometry(name)
    if stage == "weight":
        return case["O"] * FILTER * FILTER * case["C"]
    if stage == "input":
        return math.prod(geometry["inShape"])
    return TILE * TILE * geometry["tileCount"] * case["O"]


def _output_count(stage, name):
    case, geometry = CASES[name], _geometry(name)
    if stage == "weight":
        return TILE * TILE * case["C"] * case["O"]
    if stage == "input":
        return TILE * TILE * geometry["tileCount"] * case["C"]
    return math.prod(geometry["outShape"])


def _value_index(stage, name, outer, row, column, channel):
    """Where input (row, column) of slice (outer, channel) is stored."""
    if stage == "weight":
        return _tap_index(CASES[name]["C"], outer, row, column, channel)
    if stage == "input":
        return _input_index(name, outer, row, column, channel)
    return _domain_index(name, CASES[name]["O"], row, column, outer, channel)


def _result_index(stage, name, outer, row, column, channel):
    """Where result (row, column) of slice (outer, channel) is stored, if
    anywhere."""
    case = CASES[name]
    if stage == "weight":
        return _weight_output_index(case["C"], case["O"], row, column, channel, outer)
    if stage == "input":
        return _domain_index(name, case["C"], row, column, outer, channel)
    return _output_index(name, outer, row, column, channel)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def _round32(value):
    """Round an exact rational in binary32's normal range, ties to even."""
    value = Fraction(value)
    if value == 0:
        return 0
    sign = 0x80000000 if value < 0 else 0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    assert -126 <= exponent <= 127
    scaled = magnitude * Fraction(2) ** (23 - exponent)
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    if 2 * remainder > scaled.denominator or (
        2 * remainder == scaled.denominator and integer & 1
    ):
        integer += 1
    if integer == 1 << 24:
        integer >>= 1
        exponent += 1
    return sign | ((exponent + 127) << 23) | (integer & 0x7FFFFF)


def _words(table):
    """A table as binary32 words: each entry rounded once."""
    return tuple(tuple(_round32(item) for item in row) for row in table)


def _transpose(matrix):
    return tuple(tuple(row) for row in zip(*matrix))


G = _words(WT_TRANSFORM)
B = _words(IN_TRANSFORM)
A = _words(OUT_TRANSFORM)
GT, BT, AT = (_transpose(matrix) for matrix in (G, B, A))
# The left and right factors of each transform's products.
FACTORS = {"weight": (G, GT), "input": (BT, B), "output": (AT, A)}


def _finite(word):
    return word & 0x7F800000 != 0x7F800000


def _fused(left, right, accumulator):
    """A binary32 fused multiply-add on words: ties to even, gradual underflow."""
    if (
        accumulator & 0x7FFFFFFF
        and _finite(left)
        and _finite(right)
        and _finite(accumulator)
        and not (left & 0x7FFFFFFF and right & 0x7FFFFFFF)
    ):
        # x + (+-0) is exactly x for every finite nonzero x.
        return accumulator
    return _oracle(left, right, accumulator)


def _unfused(left, right, accumulator):
    """A rounded product, then a rounded sum: the defect the gate must reject."""
    if not (left & 0x7FFFFFFF and right & 0x7FFFFFFF):
        return _fused(left, right, accumulator)
    return _oracle(_oracle(left, right, 0), 0x3F800000, accumulator)


def _product(left, right, step=_fused):
    """``simdgroup_matrix`` ``left * right``: steps from +0.0 in increasing k."""
    result = []
    for row in left:
        values = []
        for column in range(TILE):
            accumulator = 0
            for inner in range(TILE):
                accumulator = step(row[inner], right[inner][column], accumulator)
            values.append(accumulator)
        result.append(tuple(values))
    return tuple(result)


def _reversed(left, right, step=_fused):
    """The same product with every chain in decreasing k."""
    return _product(tuple(row[::-1] for row in left), right[::-1], step)


def _transform(
    stage, matrix, *, step=_fused, product=_product, factors=None, reassociated=False
):
    """One channel's two products: left factor first, except that the output
    transform multiplies its right factor first."""
    left, right = factors or FACTORS[stage]
    if (stage == "output") != reassociated:
        return product(left, product(matrix, right, step), step)
    return product(product(left, matrix, step), right, step)


def _draw(dtype, label, count):
    """Values exact in ``dtype``: a sign, a binade in [2^-6, 2) and a random
    significand."""
    state = zlib.crc32(f"{dtype}-{label}".encode()) | 1
    values = []
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        sign = -1.0 if state >> 63 else 1.0
        binade = (state >> 56) % 7 - 6
        significand = 1 + ((state >> 24) & 0xFFFFFF) / 2**24
        values.append(_quantize(dtype, sign * significand * 2.0**binade))
    return values


def _slice_taps(stage):
    extent = FILTER if stage == "weight" else TILE
    return [(row, column) for row in range(extent) for column in range(extent)]


@functools.lru_cache(maxsize=None)
def _values(stage, dtype, name):
    """A transform's input values in buffer order, with the special slices."""
    values = _draw(dtype, f"{stage}-{name}", _input_count(stage, name))
    zero_outer, zero_channel = NEGATIVE_ZERO_SLICE[stage]
    impulse_outer, impulse_channel = IMPULSE_SLICE[stage]
    for row, column in _slice_taps(stage):
        index = _value_index(stage, name, zero_outer, row, column, zero_channel)
        values[index] = -0.0
        index = _value_index(stage, name, impulse_outer, row, column, impulse_channel)
        values[index] = 1.0 if (row, column) == IMPULSE_TAP[stage] else 0.0
    return tuple(values)


def _matrix(stage, dtype, name, outer, channel):
    """The 8x8 binary32 matrix one SIMD-group multiplies for slice (outer,
    channel); a 3x3 filter fills the top left and zeros the rest."""
    values = _values(stage, dtype, name)
    extent = FILTER if stage == "weight" else TILE
    return tuple(
        tuple(
            (
                _float32_bits(
                    values[_value_index(stage, name, outer, row, column, channel)]
                )
                if row < extent and column < extent
                else 0
            )
            for column in range(TILE)
        )
        for row in range(TILE)
    )


@functools.lru_cache(maxsize=None)
def _reference_words(stage, dtype, name):
    """Binary32 transform words in the output buffer's order."""
    slices, channels = _shape(stage, name)
    words = [None] * _output_count(stage, name)
    for outer in range(slices):
        for channel in range(channels):
            result = _transform(stage, _matrix(stage, dtype, name, outer, channel))
            for row in range(TILE):
                for column in range(TILE):
                    index = _result_index(stage, name, outer, row, column, channel)
                    if index is not None:
                        assert words[index] is None
                        words[index] = result[row][column]
    assert None not in words
    return tuple(words)


@functools.lru_cache(maxsize=None)
def _reference(stage, dtype, name):
    """The stored values: conv.metal's ``static_cast<T>`` rounds to nearest even."""
    return tuple(
        _quantize(dtype, _bits_float32(word))
        for word in _reference_words(stage, dtype, name)
    )


def _case_data(entry, name):
    stage, dtype = _stage(entry), _dtype(entry)
    case, geometry = CASES[name], _geometry(name)
    live = list(_reference(stage, dtype, name))
    data = {
        "arrays": {
            INPUT_BUFFER[stage]: (
                list(_values(stage, dtype, name)) + [math.nan] * INPUT_GUARD_COUNT
            ),
            OUTPUT_BUFFER[stage]: [GUARD_VALUE] * (len(live) + GUARD_COUNT),
        },
        "scalars": {},
        "params": None,
        "expected": live + [GUARD_VALUE] * GUARD_COUNT,
        "grid": geometry["grids"][stage],
    }
    if stage == "weight":
        data["scalars"] = {"C": ("int32", [case["C"]]), "O": ("int32", [case["O"]])}
    else:
        data["params"] = geometry["params"]
    return data


# ---------------------------------------------------------------------------
# Kernel index arithmetic
# ---------------------------------------------------------------------------


def _lane_coordinates(lane):
    """conv.metal's ``sm`` and ``sn`` for one SIMD lane."""
    qid = lane // 4
    sm = (qid & 4) + (lane // 2) % 4
    sn = (qid & 2) * 2 + (lane % 2) * 2
    return sm, sn


def _groups(grid):
    for z in range(grid[2]):
        for y in range(grid[1]):
            for x in range(grid[0]):
                yield z, y, x, (z * grid[1] + y) * grid[0] + x


def _simulate_weight(name):
    """Replay the weight transform's stores: wt_out index -> (row, column,
    filter, channel)."""
    channels, filters = CASES[name]["C"], CASES[name]["O"]
    written = {}
    for tid in range(_geometry(name)["grids"]["weight"][0]):
        for simd_group in range(GROUP_FILTERS):
            for lane in range(SIMD_SIZE):
                sm, sn = _lane_coordinates(lane)
                ko = GROUP_FILTERS * tid + simd_group
                out_0 = (sm * 8 + sn) * channels * filters + ko
                out_1 = (sm * 8 + sn + 1) * channels * filters + ko
                for bc in range(0, channels, BLOCK_CHANNELS):
                    for c in range(BLOCK_CHANNELS):
                        for column, base in ((sn, out_0), (sn + 1, out_1)):
                            index = base + (bc * filters) + c * filters
                            assert index not in written
                            written[index] = (sm, column, ko, bc + c)
    return written


def _simulate_input(name):
    """Replay the input transform: its shared-tile loads, (tile, row, column,
    channel) -> inp_in index, and its stores, inp_out index -> (row, column,
    tile, channel)."""
    params = _geometry(name)["params"]
    channels, strides = params["C"], params["in_strides"]
    grid = _geometry(name)["grids"]["input"]
    tiles = grid[0] * grid[1] * grid[2]
    span = TILE // TILE_SPLIT
    loaded, written = {}, {}
    for z, y, x, tile_id in _groups(grid):
        for simd_group in range(SIMD_GROUPS):
            kh = span * (simd_group // TILE_SPLIT)
            kw = span * (simd_group % TILE_SPLIT)
            base = (
                z * strides[0]
                + (OUT_TILE * y + kh) * strides[1]
                + (OUT_TILE * x + kw) * strides[2]
            )
            for lane in range(SIMD_SIZE):
                for bc in range(0, channels, BLOCK_CHANNELS):
                    for h in range(span):
                        for w in range(span):
                            jump = h * strides[1] + w * strides[2]
                            for c in range(lane, BLOCK_CHANNELS, SIMD_SIZE):
                                key = (tile_id, kh + h, kw + w, bc + c)
                                assert key not in loaded
                                loaded[key] = base + bc + jump + c
                sm, sn = _lane_coordinates(lane)
                out_0 = (sm * 8 + sn) * tiles * channels + tile_id * channels
                out_1 = (sm * 8 + sn + 1) * tiles * channels + tile_id * channels
                for bc in range(0, channels, BLOCK_CHANNELS):
                    for c in range(simd_group, BLOCK_CHANNELS, SIMD_GROUPS):
                        for column, base_out in ((sn, out_0), (sn + 1, out_1)):
                            index = base_out + bc + c
                            assert index not in written
                            written[index] = (sm, column, tile_id, bc + c)
    return loaded, written


def _simulate_output(name):
    """Replay the output transform: its matrix loads, (tile, row, column,
    channel) -> out_in index, and its guarded stores, out_out index -> (tile,
    row, column, channel)."""
    params = _geometry(name)["params"]
    filters, strides = params["O"], params["out_strides"]
    out_h, out_w = params["oS"]
    grid = _geometry(name)["grids"]["output"]
    tiles = grid[0] * grid[1] * grid[2]
    span = OUT_TILE // TILE_SPLIT
    loaded, written = {}, {}
    for z, y, x, tile_id in _groups(grid):
        shared = set()
        for simd_group in range(SIMD_GROUPS):
            for lane in range(SIMD_SIZE):
                sm, sn = _lane_coordinates(lane)
                in_0 = (sm * 8 + sn) * tiles * filters + tile_id * filters
                in_1 = (sm * 8 + sn + 1) * tiles * filters + tile_id * filters
                for bo in range(0, filters, BLOCK_CHANNELS):
                    for c in range(simd_group, BLOCK_CHANNELS, SIMD_GROUPS):
                        for column, base_in in ((sn, in_0), (sn + 1, in_1)):
                            key = (tile_id, sm, column, bo + c)
                            assert key not in loaded
                            loaded[key] = base_in + bo + c
                            if sm < OUT_TILE and column < OUT_TILE:
                                shared.add((sm, column, bo + c))
        assert len(shared) == OUT_TILE * OUT_TILE * filters
        for simd_group in range(SIMD_GROUPS):
            kh = span * (simd_group // TILE_SPLIT)
            kw = span * (simd_group % TILE_SPLIT)
            bh, bw = OUT_TILE * y + kh, OUT_TILE * x + kw
            base = z * strides[0] + bh * strides[1] + bw * strides[2]
            for h in range(span):
                for w in range(span):
                    if bh + h >= out_h or bw + w >= out_w:
                        continue
                    jump = h * strides[1] + w * strides[2]
                    for lane in range(SIMD_SIZE):
                        for bo in range(0, filters, BLOCK_CHANNELS):
                            for c in range(lane, BLOCK_CHANNELS, SIMD_SIZE):
                                assert (kh + h, kw + w, bo + c) in shared
                                index = base + jump + bo + c
                                assert index not in written
                                written[index] = (tile_id, kh + h, kw + w, bo + c)
    return loaded, written


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def _config(output, target, entry):
    stage = _stage(entry)
    options = ""
    if target in SOFTWARE_MATRIX_TARGETS:
        options += f"""
[project.source_options.metal.target_options.{target}]
cooperative_matrix_software_lowering = true
software_subgroup_width = {SIMD_SIZE}
cooperative_matrix_mma_profile = "{MMA_PROFILE}"
"""
        if target == "directx":
            options += 'relative_wave_shuffle_out_of_range = "self"\n'
        precondition = METADATA_PRECONDITIONS[stage]
        member = (
            f'member = "{precondition["member"]}"\n' if "member" in precondition else ""
        )
        options += f"""
[[project.metadata_preconditions]]
source = "{SOURCE}"
entryPoint = "{entry}"
parameter = "{precondition["parameter"]}"
{member}minimum = {precondition["minimum"]}
maximum = {precondition["maximum"]}
"""
    if target == "opengl":
        for expression in OPENGL_INDEX_RANGE_ASSERTIONS[stage]:
            options += f"""
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "{expression}"
minimum = 0
maximum = {SIGNED_MAX}
"""
    size = ", ".join(str(item) for item in WORKGROUPS[stage])
    return f"""[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
output_dir = "{output}"
[project.entry_points]
"{SOURCE}" = "{entry}"
[project.entry_workgroup_size_rules."{SOURCE}"]
"{entry}" = [{size}]
[project.source_options.metal]
max_template_specializations = 256
max_template_materialization_work = 16384
preserve_resource_origins = true
cooperative_matrix_fragment_mapping = "{FRAGMENT_MAPPING}"
cooperative_matrix_fragment_mapping_provenance = "{FRAGMENT_MAPPING_PROVENANCE}"
{options}"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _checked_elements(entry):
    return sum(_output_count(_stage(entry), name) for name in CASES)


def _case_contract(name):
    geometry = _geometry(name)
    case = CASES[name]
    return {
        "N": case["N"],
        "oS": list(case["oS"]),
        "C": case["C"],
        "O": case["O"],
        "tiles": list(geometry["tiles"]),
        "paddedInput": list(geometry["params"]["iS"]),
        "grids": {stage: list(geometry["grids"][stage]) for stage in STAGES},
        "checkedValues": {stage: _output_count(stage, name) for stage in STAGES},
    }


def _expected_validation_contract():
    return {
        "comparison": "exact words",
        "matrixProducts": {
            "profile": MMA_PROFILE,
            "definition": (
                "each element is a chain of fused binary32 multiply-adds from "
                "+0.0 in increasing k, ties to even, with gradual underflow"
            ),
            "softwareTargets": list(SOFTWARE_MATRIX_TARGETS),
        },
        "fragmentMapping": {
            "mapping": FRAGMENT_MAPPING,
            "provenance": FRAGMENT_MAPPING_PROVENANCE,
        },
        "softwareSubgroupWidth": {
            "targets": list(SOFTWARE_MATRIX_TARGETS),
            "width": SIMD_SIZE,
        },
        "stages": {
            stage: {
                "products": PRODUCTS[stage],
                "table": TABLES[stage][0],
                "workgroupSize": list(WORKGROUPS[stage]),
                "grid": GRID_RULES[stage],
                "buffers": list(BUFFER_ORDER[stage]),
                "metadataPreconditions": {
                    target: (
                        [dict(METADATA_PRECONDITIONS[stage])]
                        if target in SOFTWARE_MATRIX_TARGETS
                        else []
                    )
                    for target in TARGETS
                },
                "indexRangeAssertions": {
                    target: (
                        [
                            {
                                "expression": expression,
                                "minimum": 0,
                                "maximum": SIGNED_MAX,
                            }
                            for expression in OPENGL_INDEX_RANGE_ASSERTIONS[stage]
                        ]
                        if target == "opengl"
                        else []
                    )
                    for target in TARGETS
                },
                "specialSlices": {
                    "negativeZero": list(NEGATIVE_ZERO_SLICE[stage]),
                    "impulse": {
                        "slice": list(IMPULSE_SLICE[stage]),
                        "tap": list(IMPULSE_TAP[stage]),
                    },
                },
            }
            for stage in STAGES
        },
        "cases": {name: _case_contract(name) for name in CASES},
        "channelRule": "C % 32 == 0, O % 32 == 0",
        "transformWords": {
            table: [[f"0x{word:08x}" for word in row[:live]] for row in _words(values)]
            for table, values, live in TABLES.values()
        },
        "paramsBindings": {
            "record": PARAMS_SIZES[2],
            "block": dict(PARAMS_BLOCK_SIZES),
            "memberOffsets": dict(PARAMS_MEMBER_OFFSETS),
        },
        "outputGuard": GUARD_VALUE,
        "outputGuardCount": GUARD_COUNT,
        "inputGuard": "nan",
        "inputGuardCount": INPUT_GUARD_COUNT,
        "elementTypes": {
            target: {dtype: _element_type(target, dtype) for dtype in DTYPES}
            for target in TARGETS
        },
        "dispatchCountByTarget": len(ENTRIES) * len(CASES),
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": len(WINOGRAD_ENTRIES),
        "coveredEntryCount": len(ENTRIES),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "artifactCountByTarget": len(ENTRIES),
        "dispatchCountByTarget": len(ENTRIES) * len(CASES),
        "checkedElementCountByEntry": {
            entry: _checked_elements(entry) for entry in ENTRIES
        },
        "completeCorpusEntryCount": 17832,
        "partialOutputTilesExercised": True,
        "partialThreadgroupsExercised": False,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_winograd_contract_is_exact():
    contract = _contract()
    assert set(contract) == {
        "schemaVersion",
        "kind",
        "provenance",
        "entries",
        "artifacts",
        "validationContract",
        "scope",
    }
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-winograd-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "headers": HEADERS,
    }
    assert tuple(contract["entries"]) == ENTRIES
    assert set(contract["artifacts"]) == set(TARGETS)
    for target in TARGETS:
        artifacts = contract["artifacts"][target]
        assert tuple(artifacts) == ENTRIES
        for record in artifacts.values():
            assert set(record) == {"sha256", "sizeBytes"}
            assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
        assert len({record["sha256"] for record in artifacts.values()}) == len(ENTRIES)
    assert contract["validationContract"] == _expected_validation_contract()
    assert contract["scope"] == _expected_scope()


def test_winograd_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "winograd.current-tree.translation.json" in readme
    assert len(ENTRIES) == len(set(ENTRIES)) == 9
    # instantiate_winograd_conv_2d emits weight, input and output per type;
    # the output transform's block is bo, the others' bc.
    for index, entry in enumerate(ENTRIES):
        match = ENTRY_PATTERN.fullmatch(entry)
        assert match, entry
        assert match.group("stage") == STAGES[index % 3]
        assert match.group("dtype") == DTYPES[index // 3]
        assert match.group("block") == ("bo" if _stage(entry) == "output" else "bc")


def test_winograd_cases_follow_mlx_dispatch():
    for name, case in CASES.items():
        # conv_2D_gpu's Winograd conditions on the channel counts.
        assert case["C"] % 32 == 0 and case["O"] % 32 == 0, name
        assert case["C"] != case["O"], name
        geometry = _geometry(name)
        out_h, out_w = case["oS"]
        # winograd_conv_2D_gpu: N_tiles_h, N_tiles_w and the padded input of
        # a valid 3x3 convolution, whose input extent is oS + 2.
        assert geometry["tiles"] == (math.ceil(out_h / 6), math.ceil(out_w / 6))
        assert geometry["params"]["iS"] == [
            6 * math.ceil((extent + 2 - 2) / 6) + 2 for extent in (out_h, out_w)
        ]
        tiles_h, tiles_w = geometry["tiles"]
        assert geometry["grids"] == {
            "weight": [case["O"] // GROUP_FILTERS, 1, 1],
            "input": [tiles_w, tiles_h, case["N"]],
            "output": [tiles_w, tiles_h, case["N"]],
        }
        # Every 8x8 input tile lies inside the padded input.
        assert OUT_TILE * (tiles_h - 1) + TILE <= geometry["params"]["iS"][0]
        assert OUT_TILE * (tiles_w - 1) + TILE <= geometry["params"]["iS"][1]
        for entry in ENTRIES[:3]:
            stage = _stage(entry)
            data = _case_data(entry, name)
            assert len(data["arrays"][INPUT_BUFFER[stage]]) == (
                _input_count(stage, name) + INPUT_GUARD_COUNT
            )
            assert data["grid"] == geometry["grids"][stage]
            assert all(0 < item <= SIGNED_MAX for item in data["grid"])
    assert WORKGROUPS["weight"] == (32, GROUP_FILTERS, 1)
    assert WORKGROUPS["input"] == WORKGROUPS["output"] == (32, 2, 2)
    assert math.prod(WORKGROUPS["input"]) == SIMD_GROUPS * SIMD_SIZE
    assert any(case["C"] > BLOCK_CHANNELS for case in CASES.values())
    assert any(case["O"] > BLOCK_CHANNELS for case in CASES.values())
    assert any(case["N"] > 1 for case in CASES.values())
    # Partial output tiles in both directions, and a case without any.
    assert any(all(extent % 6 for extent in case["oS"]) for case in CASES.values())
    assert any(not any(extent % 6 for extent in case["oS"]) for case in CASES.values())


def test_winograd_params_follow_the_mlx_struct_layout():
    for name in CASES:
        params = _geometry(name)["params"]
        data = _params_bytes(params)
        assert len(data) == PARAMS_SIZES[2]
        for member, offset in PARAMS_MEMBER_OFFSETS.items():
            assert struct.unpack_from("<i", data, offset)[0] == params[member]
        assert params["in_strides"][3] == params["out_strides"][3] == 1


def test_winograd_tables_round_once():
    for row, items in zip(G, WT_TRANSFORM):
        for word, item in zip(row, items):
            # Metal and Mesa divide binary32 literals, which rounds once, and
            # DXC folds literal arithmetic in double first; both give RN32.
            assert word == _float32_bits(item.numerator / item.denominator)
        assert row[FILTER:] == (0,) * (TILE - FILTER)
    assert G[1][0] == 0xBE638E39  # -2/9
    assert G[3][0] == 0x3C360B61  # 1/90
    assert G[5][0] == 0x3F360B61  # 32/45
    # The input and output tables are dyadic, so binary32 holds them exactly.
    for words, table in ((B, IN_TRANSFORM), (A, OUT_TRANSFORM)):
        for row, items in zip(words, table):
            for word, item in zip(row, items):
                assert Fraction(_bits_float32(word)) == item
    assert B[2][0] == 0xC0A80000  # -21/4
    assert all(row[OUT_TILE:] == (0, 0) for row in A)
    for transposed, matrix in ((GT, G), (BT, B), (AT, A)):
        assert transposed == tuple(
            tuple(matrix[column][row] for column in range(TILE)) for row in range(TILE)
        )


def test_winograd_output_products_are_exact():
    # A holds zeros, ones and signed powers of two, so every output-transform
    # product is exact and fused and unfused steps agree: that gate checks
    # summation order and rounding. The weight and input transforms check
    # fusion.
    for row in OUT_TRANSFORM:
        for item in row:
            magnitude = abs(item)
            assert item == 0 or not (
                magnitude.numerator & (magnitude.numerator - 1)
                or magnitude.denominator & (magnitude.denominator - 1)
            )
    for dtype in DTYPES:
        for outer, channel in ((0, 0), (3, 7), (7, 31)):
            matrix = _matrix("output", dtype, "partial-tiles", outer, channel)
            assert _transform("output", matrix, step=_unfused) == _transform(
                "output", matrix
            )


def test_winograd_reference_matches_kernel_index_arithmetic():
    for name, case in CASES.items():
        written = _simulate_weight(name)
        assert set(written) == set(range(_output_count("weight", name)))
        for index, (row, column, filter_index, channel) in written.items():
            assert index == _result_index(
                "weight", name, filter_index, row, column, channel
            )
        loaded, written = _simulate_input(name)
        assert set(written) == set(range(_output_count("input", name)))
        for index, (row, column, tile, channel) in written.items():
            assert index == _result_index("input", name, tile, row, column, channel)
        tiles = _geometry(name)["tileCount"]
        assert len(loaded) == tiles * TILE * TILE * case["C"]
        for (tile, row, column, channel), index in loaded.items():
            assert index == _value_index("input", name, tile, row, column, channel)
        # Neighbouring input tiles share two columns.
        assert _input_index(name, 0, 0, OUT_TILE, 0) == _input_index(name, 1, 0, 0, 0)
        loaded, written = _simulate_output(name)
        assert set(written) == set(range(_output_count("output", name)))
        for index, (tile, row, column, channel) in written.items():
            assert index == _result_index("output", name, tile, row, column, channel)
        assert len(loaded) == tiles * TILE * TILE * case["O"]
        for (tile, row, column, channel), index in loaded.items():
            assert index == _value_index("output", name, tile, row, column, channel)
    # Every lane owns a distinct row pair of the 8x8 matrix.
    pairs = {
        (sm, sn + offset)
        for lane in range(SIMD_SIZE)
        for sm, sn in [_lane_coordinates(lane)]
        for offset in (0, 1)
    }
    assert len(pairs) == TILE * TILE


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_winograd_values_are_exact_and_guards_distinct(stage, dtype):
    left, right = FACTORS[stage]
    for name in CASES:
        values = _values(stage, dtype, name)
        assert all(_quantize(dtype, value) == value for value in values)
        # bfloat16 holds 1,792 values in these binades; the others far more.
        assert len(set(values)) >= 1024
        live = _reference(stage, dtype, name)
        assert GUARD_VALUE not in live
        assert all(math.isfinite(value) for value in live)
        assert all(_quantize(dtype, value) == value for value in live)
        words = _reference_words(stage, dtype, name)
        zero_outer, zero_channel = NEGATIVE_ZERO_SLICE[stage]
        impulse_outer, impulse_channel = IMPULSE_SLICE[stage]
        kh, kw = IMPULSE_TAP[stage]
        checked = 0
        for row in range(TILE):
            for column in range(TILE):
                index = _result_index(
                    stage, name, zero_outer, row, column, zero_channel
                )
                if index is not None:
                    # The chains start from +0.0, so -0.0 inputs give +0.0.
                    assert words[index] == 0
                    checked += 1
                index = _result_index(
                    stage, name, impulse_outer, row, column, impulse_channel
                )
                if index is not None:
                    expected = _oracle(left[row][kh], right[kw][column], 0)
                    assert words[index] == expected
                    checked += 1
        assert checked >= 2 * OUT_TILE * OUT_TILE
    assert _quantize(dtype, GUARD_VALUE) == GUARD_VALUE


# A's powers of two make every output-transform product exact, so fused and
# unfused steps agree there; test_winograd_output_products_are_exact records
# that limit.
MUTATIONS = {
    stage: (
        ("unfused",) * (stage != "output")
        + (
            "reversed-k",
            "reassociated",
            "transposed-input",
            "untransposed-factor",
            "neighbor-channel",
            "neighbor-slice",
        )
    )
    for stage in STAGES
}


def _mutated(stage, dtype, name, mutation, outer, channel):
    slices, channels = _shape(stage, name)
    matrix = _matrix(stage, dtype, name, outer, channel)
    if mutation == "unfused":
        return _transform(stage, matrix, step=_unfused)
    if mutation == "reversed-k":
        return _transform(stage, matrix, product=_reversed)
    if mutation == "reassociated":
        return _transform(stage, matrix, reassociated=True)
    if mutation == "transposed-input":
        return _transform(stage, _transpose(matrix))
    if mutation == "untransposed-factor":
        table = {"weight": G, "input": B, "output": A}[stage]
        return _transform(stage, matrix, factors=(table, table))
    if mutation == "neighbor-channel":
        return _transform(
            stage, _matrix(stage, dtype, name, outer, (channel + 1) % channels)
        )
    if mutation == "neighbor-slice":
        return _transform(
            stage, _matrix(stage, dtype, name, (outer + 1) % slices, channel)
        )
    raise AssertionError(mutation)


@pytest.mark.parametrize(
    "stage,mutation",
    [(stage, mutation) for stage in STAGES for mutation in MUTATIONS[stage]],
)
def test_winograd_reference_rejects_defects(stage, mutation):
    dtype, name = DTYPES[0], "partial-tiles"
    slices, channels = _shape(stage, name)
    words = _reference_words(stage, dtype, name)
    differing = 0
    for outer, channel in (
        (0, 0),
        (5 % slices, 40 % channels),
        (slices - 1, channels - 1),
        (slices // 2, 17 % channels),
    ):
        result = _mutated(stage, dtype, name, mutation, outer, channel)
        for row in range(TILE):
            for column in range(TILE):
                index = _result_index(stage, name, outer, row, column, channel)
                if index is not None:
                    differing += result[row][column] != words[index]
    assert differing >= 4, (stage, mutation)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_winograd_configs_scope_target_options(tmp_path, target, entry):
    stage = _stage(entry)
    config_path = tmp_path / "crosstl.toml"
    config_path.write_text(_config("out", target, entry), encoding="utf-8")
    config = load_project_config(tmp_path, config_path)
    assert list(config.targets) == [target]
    metal = config.source_options["metal"]
    assert metal["cooperative_matrix_fragment_mapping"] == FRAGMENT_MAPPING
    assert metal["preserve_resource_origins"] is True
    options = metal.get("target_options", {}).get(target, {})
    if target in SOFTWARE_MATRIX_TARGETS:
        assert options["cooperative_matrix_software_lowering"] is True
        assert options["software_subgroup_width"] == SIMD_SIZE
        assert options["cooperative_matrix_mma_profile"] == MMA_PROFILE
        assert [item.to_json() for item in config.metadata_preconditions] == [
            {"source": SOURCE, "entryPoint": entry, **METADATA_PRECONDITIONS[stage]}
        ]
    else:
        assert options == {}
        assert not config.metadata_preconditions
    expressions = [item.expression for item in config.index_range_assertions]
    assert expressions == (
        list(OPENGL_INDEX_RANGE_ASSERTIONS[stage]) if target == "opengl" else []
    )
    size = ", ".join(str(item) for item in WORKGROUPS[stage])
    assert f'"{entry}" = [{size}]' in config_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Native parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native Winograd gate")
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
    assert hashlib.sha256((root / SOURCE).read_bytes()).hexdigest() == SOURCE_SHA256
    for header, digest in HEADERS.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("winograd-metal-reference")
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


def _literal_quotient(text):
    """An entry such as ``-2.0 / 9.00`` or ``-5.25f``: decimal literals
    divided exactly."""
    value, *divisors = (
        Fraction(part.strip().removesuffix("f")) for part in text.split("/")
    )
    for divisor in divisors:
        value /= divisor
    return value


def test_current_mlx_winograd_tables_match_source(current_mlx):
    root, _ = current_mlx
    text = (root / SOURCE).read_text(encoding="utf-8")
    for name, table, live in TABLES.values():
        match = re.search(
            rf"{name}\[SIMD_MATRIX_SIZE\]\[SIMD_MATRIX_SIZE\] = \{{(.*?)\}};",
            text,
            re.S,
        )
        assert match, name
        rows = re.findall(r"\{([^{}]*)\}", match.group(1))
        assert len(rows) == TILE, name
        for row, items in zip(rows, table):
            values = [_literal_quotient(item) for item in row.split(",")]
            assert len(values) == live, name
            assert tuple(values) + (0,) * (TILE - live) == items, name


def _translate(root, work, target, entry):
    """Translate one entry and check it against its pinned artifact."""
    stage = _stage(entry)
    record = _contract()["artifacts"][target][entry]
    config = work / "crosstl.toml"
    config.write_text(_config(f"{work.name}/out", target, entry), encoding="utf-8")
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
    assert artifact["sourceHash"] == {"algorithm": "sha256", "value": SOURCE_SHA256}
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": record["sha256"],
    }
    assert artifact["generatedSizeBytes"] == record["sizeBytes"]
    (execution,) = artifact["execution"]["entryPoints"]
    assert execution["workgroupSize"] == list(WORKGROUPS[stage])
    generated = root / artifact["path"]
    text = generated.read_text(encoding="utf-8")
    assert "unsupported" not in text
    if target == "metal":
        # Metal keeps the hardware SIMD-group matrices.
        assert "simdgroup_matrix<float, 8, 8>" in text
        _metal_library(generated, work / "translated.metallib", root)
    else:
        assert "simdgroup_matrix" not in text
        # Both products run through the software multiply-accumulate.
        helper = {
            "opengl": "_tile_4x4_row_pair_multiply_accumulate(",
            "directx": "__crossgl_software_matrix_mma(",
        }[target]
        assert text.count(helper) >= 3, helper
        _validate_generated_artifact(generated, work, target)
    descriptor, package = _prepare_native_package(translation, work)
    return descriptor, package, record


def _params_value(target, layout, params):
    """Pack MLXConvParams<2> for the binding's reflected physical layout."""
    if target == "metal":
        assert layout["elementType"] == "record", layout
        assert layout["elementStrideBytes"] == PARAMS_SIZES[2]
        words = pack_storage_records(layout, [params])
        # The reflected record reproduces MLX's C struct byte for byte.
        assert struct.pack(f"<{len(words)}I", *words) == _params_bytes(params)
    else:
        # A constant reference binds as a uniform block or constant buffer.
        assert (
            layout["storageLayout"]
            == {"opengl": "std140", "directx": "hlsl-constant-buffer"}[target]
        )
        assert layout["blockSizeBytes"] == PARAMS_BLOCK_SIZES[target]
        words = pack_uniform_block(layout, params)
    # The member preconditions read C and O at the same offsets everywhere.
    for member, offset in PARAMS_MEMBER_OFFSETS.items():
        assert words[offset // 4] == params[member]
    return {"dtype": "uint32", "shape": [len(words)], "values": words}


def _request(descriptor, package, target, entry, data):
    """Bind ``data`` by source parameter name and build one dispatch."""
    stage, dtype = _stage(entry), _dtype(entry)
    values = {
        name: _runtime_value(target, dtype, items)
        for name, items in data["arrays"].items()
    }
    values.update(
        {
            name: {"dtype": element, "shape": [len(items)], "values": list(items)}
            for name, (element, items) in data["scalars"].items()
        }
    )
    inputs, observed, output = {}, [], None
    # DirectX constant members carry the sanitized entry name as a prefix.
    prefix = re.sub(r"_+", "_", entry + "_")
    bound = set()
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        layout = binding["scalarLayout"]
        name = layout.get("memberName", binding["name"]).removeprefix(prefix)
        assert name not in bound, name
        bound.add(name)
        if name == "params":
            value = _params_value(target, layout, data["params"])
        else:
            value = values[name]
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
        if name == OUTPUT_BUFFER[stage]:
            output = binding["name"]
        else:
            observed.append(binding["name"])
    assert bound == set(BUFFER_ORDER[stage]), bound
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output: dict(values[OUTPUT_BUFFER[stage]])},
        {"workgroupCount": data["grid"], "workgroupSize": list(WORKGROUPS[stage])},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, {OUTPUT_BUFFER[stage]: output}, len(observed)


def _payloads(entry, data):
    stage, dtype = _stage(entry), _dtype(entry)
    raw = {name: _metal_bytes(dtype, items) for name, items in data["arrays"].items()}
    for name, (_, items) in data["scalars"].items():
        raw[name] = struct.pack(f"<{len(items)}i", *items)
    if data["params"] is not None:
        raw["params"] = _params_bytes(data["params"])
    return [raw[name] for name in BUFFER_ORDER[stage]]


def _upstream(work, reference, entry, label, payloads, grid):
    """Run the upstream metallib on raw buffers in ``[[buffer(i)]]`` order."""
    stage = _stage(entry)
    runner, library = reference
    files = []
    for index, payload in enumerate(payloads):
        path = work / f"{label}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    request = {
        "buffers": files,
        "workgroupCount": grid,
        "workgroupSize": list(WORKGROUPS[stage]),
        "simdWidth": SIMD_SIZE,
    }
    path = work / f"{label}-upstream-request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    readback = work / f"{label}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, path, readback], work, f"{label}-upstream")
    )
    output_index = BUFFER_ORDER[stage].index(OUTPUT_BUFFER[stage])
    raw = None
    for index, payload in enumerate(payloads):
        data = (readback / f"buffer-{index}.bin").read_bytes()
        if index == output_index:
            raw = data
        else:
            assert data == payload, (label, index)
    return device, raw


@pytest.mark.parametrize("entry", ENTRIES)
def test_current_mlx_winograd_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    stage, dtype = _stage(entry), _dtype(entry)
    with tempfile.TemporaryDirectory(prefix=".current-winograd-", dir=root) as path:
        work = Path(path)
        try:
            descriptor, package, record = _translate(root, work, target, entry)
            executor = _executor(target)
            evidence = []
            try:
                for name in CASES:
                    data = _case_data(entry, name)
                    request, output_names, observed = _request(
                        descriptor, package, target, entry, data
                    )
                    outputs = _execute(
                        executor, work, name, request, output_names, observed
                    )
                    output = outputs[OUTPUT_BUFFER[stage]]
                    expected = _runtime_value(target, dtype, data["expected"])
                    assert output["dtype"] == expected["dtype"]
                    _assert_words(name, output["values"], expected["values"])
                    upstream = None
                    if target == "metal":
                        device, raw = _upstream(
                            work,
                            metal_reference,
                            entry,
                            name,
                            _payloads(entry, data),
                            data["grid"],
                        )
                        assert raw == _metal_bytes(dtype, data["expected"]), name
                        upstream = {"device": device, "matched": True}
                    evidence.append(
                        {
                            "case": name,
                            "grid": data["grid"],
                            "checkedValues": len(data["expected"]) - GUARD_COUNT,
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
                        "stage": stage,
                        "target": target,
                        "artifactSha256": record["sha256"],
                        "cases": evidence,
                        "comparison": "exact words",
                        "matrixProductProfile": (
                            MMA_PROFILE if target in SOFTWARE_MATRIX_TARGETS else None
                        ),
                        "metadataPreconditions": (
                            [METADATA_PRECONDITIONS[stage]]
                            if target in SOFTWARE_MATRIX_TARGETS
                            else []
                        ),
                        "indexRangeAssertions": (
                            list(OPENGL_INDEX_RANGE_ASSERTIONS[stage])
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


def test_ci_requires_native_winograd_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_step(
        workflow, "Prove current MLX Winograd native parity", group="convolution"
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/winograd",
        "--junitxml=mlx-current-results/winograd.xml",
        "demos/integrations/mlx/tests/kernels/test_current_winograd.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_winograd.py"
    )
