"""Pinned MLX Winograd weight transforms on Metal, OpenGL and DirectX.

``conv.cpp``'s ``winograd_conv_2D_gpu`` moves every 3x3 filter into the 8x8
Winograd domain with ``winograd_conv_2d_weight_transform`` before its batched
GEMM. Each SIMD-group forms ``G g G^T`` for one output filter, one channel at a
time, with two ``simdgroup_matrix<float, 8, 8>`` products. This gate translates
the three weight-transform entries to Metal, OpenGL and DirectX and executes
them natively with ``conv.cpp``'s launch geometry.

OpenGL and DirectX lower each product as the software multiply-accumulate from
a +0.0 accumulator with fused, ties-to-even binary32 steps in increasing k,
which Apple's SIMD-group matrices match. ``G`` holds non-dyadic constants such
as -2/9, so nearly every step rounds: the reference words come from an exact
fused multiply-add oracle and every output word is compared exactly.

The input and output transforms stay pending on OpenGL and DirectX. Their
channel loops run to ``params.C`` and ``params.O`` and their matrix loops start
at the SIMD-group index, and translation cannot yet prove either uniform for
the software subgroup barriers.
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
    SOURCE,
    SOURCE_SHA256,
    WINOGRAD_ENTRIES,
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
# instantiate_winograd_conv_2d runs for float, bfloat16_t and half, in order.
DTYPES = ("float32", "bfloat16", "float16")
ENTRIES = tuple(f"winograd_conv_2d_weight_transform_{dtype}_bc32" for dtype in DTYPES)
PENDING_ENTRIES = tuple(entry for entry in WINOGRAD_ENTRIES if entry not in ENTRIES)
ENTRY_PATTERN = re.compile(
    r"winograd_conv_2d_weight_transform_(?P<dtype>float32|float16|bfloat16)_bc32"
)
PENDING_PATTERN = re.compile(
    r"winograd_conv_2d_(?P<stage>input|output)_transform_"
    r"(?P<dtype>float32|float16|bfloat16)_b[co]32"
)
SIMD_SIZE = 32
SIGNED_MAX = 2**31 - 1
# winograd_conv_2D_gpu: bc = 32 channels per block and bo = 4 filters, one per
# SIMD-group, in (32, bo, 1) groups over (O / bo, 1, 1).
BLOCK_CHANNELS = 32
GROUP_FILTERS = 4
WORKGROUP = (SIMD_SIZE, GROUP_FILTERS, 1)
# The input and output transforms use (32, wn, wm) = (32, 2, 2) groups.
PENDING_WORKGROUP = (SIMD_SIZE, 2, 2)
FILTER = 3
TILE = 8
# WinogradTransforms<6, 3, 8>::wt_transform; its other five columns are zero.
WT_TRANSFORM = (
    ((1, 1), (0, 1), (0, 1)),
    ((-2, 9), (-2, 9), (-2, 9)),
    ((-2, 9), (2, 9), (-2, 9)),
    ((1, 90), (1, 45), (2, 45)),
    ((1, 90), (-1, 45), (2, 45)),
    ((32, 45), (16, 45), (8, 45)),
    ((32, 45), (-16, 45), (8, 45)),
    ((0, 1), (0, 1), (1, 1)),
)
# The kernels index lanes the way Apple lays out an 8x8 SIMD-group matrix:
# four 4x4 tiles in row-major order, each lane holding a row pair.
FRAGMENT_MAPPING = "tile_4x4_row_pair"
FRAGMENT_MAPPING_PROVENANCE = "mlx_conv_winograd_lane_coordinates"
SOFTWARE_MATRIX_TARGETS = ("opengl", "directx")
MMA_PROFILE = "fma-rne-gradual"
# `for (int bc = 0; bc < C; bc += 32)` holds barriers, so OpenGL and DirectX
# must prove it terminates without overflow: C <= INT_MAX - 31.
CHANNEL_PRECONDITION = {"parameter": "C", "minimum": 0, "maximum": SIGNED_MAX - 31}
# OpenGL indexes 32-bit storage arrays. These host preconditions bound the
# composed element offsets to signed int range; the cases stay far below it.
OPENGL_INDEX_RANGE_ASSERTIONS = (
    "ko * 3 * 3 * C",
    "ohw_0 * C * O + ko",
    "ohw_1 * C * O + ko",
    "wt_in + kh * 3 * C + kw * C + kc",
    "wt_out_0 + c * O",
    "wt_out_1 + c * O",
)
# The pending transforms' offsets, so that their recorded OpenGL diagnostic is
# the remaining translation gap rather than a missing precondition.
PENDING_OPENGL_INDEX_RANGE_ASSERTIONS = {
    "input": (
        "tid.z * params.in_strides[0] + bh * params.in_strides[1]"
        " + bw * params.in_strides[2]",
        "ohw_0 * N_TILES * params.C + tile_id * params.C",
        "ohw_1 * N_TILES * params.C + tile_id * params.C",
        "in_ptr + c",
    ),
    "output": (
        "tid.z * params.out_strides[0] + bh * params.out_strides[1]"
        " + bw * params.out_strides[2]",
        "ohw_0 * N_TILES * params.O + tile_id * params.O",
        "ohw_1 * N_TILES * params.O + tile_id * params.O",
    ),
}
# Cases: name -> (C, O). conv.cpp only takes Winograd when C and O are
# multiples of 32; three channel blocks exercise the block pointer updates.
CASES = {"one-block": (32, 64), "three-blocks": (96, 32)}
# Filter slices with known transforms: all taps -0.0 must give +0.0 words,
# since every product chain starts from +0.0; a unit impulse at (kh, kw)
# gives the outer product of G's columns kh and kw.
NEGATIVE_ZERO_SLICE = (1, 3)
IMPULSE_SLICE = (2, 5)
IMPULSE_TAP = (1, 2)
# -2^15 is exact in every type and no transform reaches it.
GUARD_VALUE = -32768.0
GUARD_COUNT = 16
INPUT_GUARD_COUNT = 16
BUFFER_ORDER = ("wt_in", "wt_out", "C", "O")
OUTPUT_BUFFER = "wt_out"


def _dtype(entry):
    return ENTRY_PATTERN.fullmatch(entry).group("dtype")


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


def _transform_words():
    """``wt_transform`` as binary32 words: G, an 8x8 table with 3 live columns."""
    return tuple(
        tuple(_round32(Fraction(*item)) for item in row) + (0,) * (TILE - FILTER)
        for row in WT_TRANSFORM
    )


G = _transform_words()
GT = tuple(tuple(G[column][row] for column in range(TILE)) for row in range(TILE))


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
    product = _oracle(left, right, 0) if left & right & 0x7FFFFFFF else None
    if product is None:
        return _fused(left, right, accumulator)
    return _oracle(product, 0x3F800000, accumulator)


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


def _filter_matrix(taps):
    """``g``: a 3x3 slice of binary32 words in the 8x8 matrix's top left."""
    return tuple(
        tuple(
            taps[row][column] if row < FILTER and column < FILTER else 0
            for column in range(TILE)
        )
        for row in range(TILE)
    )


def _transform(taps, *, left=G, right=GT, step=_fused):
    """``(G * g) * Gt`` for one filter slice ``taps[kh][kw]``."""
    return _product(_product(left, _filter_matrix(taps), step), right, step)


def _tap_index(channels, filter_index, kh, kw, channel):
    """conv.cpp's (O, 3, 3, C) weight layout."""
    return ((filter_index * FILTER + kh) * FILTER + kw) * channels + channel


def _output_index(channels, filters, row, column, channel, filter_index):
    """wt_out is stored transposed: (8 * 8, C, O)."""
    return ((row * TILE + column) * channels + channel) * filters + filter_index


@functools.lru_cache(maxsize=None)
def _taps(dtype, name):
    """Filter values exact in ``dtype``: a sign, a binade in [2^-6, 2) and a
    random significand, with the two special slices."""
    channels, filters = CASES[name]
    state = zlib.crc32(f"{dtype}-{name}".encode()) | 1
    values = []
    for _ in range(filters * FILTER * FILTER * channels):
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        sign = -1.0 if state >> 63 else 1.0
        binade = (state >> 56) % 7 - 6
        significand = 1 + ((state >> 24) & 0xFFFFFF) / 2**24
        values.append(_quantize(dtype, sign * significand * 2.0**binade))
    for kh in range(FILTER):
        for kw in range(FILTER):
            values[_tap_index(channels, NEGATIVE_ZERO_SLICE[0], kh, kw, 3)] = -0.0
            values[_tap_index(channels, IMPULSE_SLICE[0], kh, kw, IMPULSE_SLICE[1])] = (
                1.0 if (kh, kw) == IMPULSE_TAP else 0.0
            )
    return tuple(values)


def _slice_words(dtype, name, filter_index, channel):
    channels, _ = CASES[name]
    taps = _taps(dtype, name)
    return tuple(
        tuple(
            _float32_bits(taps[_tap_index(channels, filter_index, kh, kw, channel)])
            for kw in range(FILTER)
        )
        for kh in range(FILTER)
    )


@functools.lru_cache(maxsize=None)
def _reference_words(dtype, name):
    """Binary32 transform words in wt_out's (8 * 8, C, O) order."""
    channels, filters = CASES[name]
    words = [None] * (TILE * TILE * channels * filters)
    for filter_index in range(filters):
        for channel in range(channels):
            result = _transform(_slice_words(dtype, name, filter_index, channel))
            for row in range(TILE):
                for column in range(TILE):
                    index = _output_index(
                        channels, filters, row, column, channel, filter_index
                    )
                    words[index] = result[row][column]
    assert None not in words
    return tuple(words)


@functools.lru_cache(maxsize=None)
def _reference(dtype, name):
    """The stored values: conv.metal's ``static_cast<T>`` rounds to nearest even."""
    return tuple(
        _quantize(dtype, _bits_float32(word)) for word in _reference_words(dtype, name)
    )


def _case_data(entry, name):
    dtype = _dtype(entry)
    channels, filters = CASES[name]
    live = list(_reference(dtype, name))
    return {
        "arrays": {
            "wt_in": list(_taps(dtype, name)) + [math.nan] * INPUT_GUARD_COUNT,
            "wt_out": [GUARD_VALUE] * (len(live) + GUARD_COUNT),
        },
        "scalars": {"C": ("int32", [channels]), "O": ("int32", [filters])},
        "expected": live + [GUARD_VALUE] * GUARD_COUNT,
        "grid": [filters // GROUP_FILTERS, 1, 1],
    }


def _lane_coordinates(lane):
    """conv.metal's ``sm`` and ``sn`` for one SIMD lane."""
    qid = lane // 4
    sm = (qid & 4) + (lane // 2) % 4
    sn = (qid & 2) * 2 + (lane % 2) * 2
    return sm, sn


def _simulate_writes(channels, filters, grid):
    """Replay the kernel's index arithmetic: output index -> (row, column, c, o)."""
    written = {}
    for tid in range(grid[0]):
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
                            written[index] = (sm, column, bc + c, ko)
    return written


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def _opengl_index_range_assertions(entry):
    pending = PENDING_PATTERN.fullmatch(entry)
    if pending is None:
        return OPENGL_INDEX_RANGE_ASSERTIONS
    return PENDING_OPENGL_INDEX_RANGE_ASSERTIONS[pending.group("stage")]


def _config(output, target, entry):
    pending = PENDING_PATTERN.fullmatch(entry)
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
        if pending is None:
            options += f"""
[[project.metadata_preconditions]]
source = "{SOURCE}"
entryPoint = "{entry}"
parameter = "{CHANNEL_PRECONDITION["parameter"]}"
minimum = {CHANNEL_PRECONDITION["minimum"]}
maximum = {CHANNEL_PRECONDITION["maximum"]}
"""
    if target == "opengl":
        for expression in _opengl_index_range_assertions(entry):
            options += f"""
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "{expression}"
minimum = 0
maximum = {SIGNED_MAX}
"""
    size = ", ".join(
        str(item) for item in (PENDING_WORKGROUP if pending else WORKGROUP)
    )
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


def _checked_elements():
    return sum(TILE * TILE * channels * filters for channels, filters in CASES.values())


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
        "workgroupSize": list(WORKGROUP),
        "grid": "[O / 4, 1, 1] groups, C % 32 == 0, O % 32 == 0",
        "cases": {
            name: {
                "C": channels,
                "O": filters,
                "grid": [filters // GROUP_FILTERS, 1, 1],
            }
            for name, (channels, filters) in CASES.items()
        },
        "transformWords": [[f"0x{word:08x}" for word in row[:FILTER]] for row in G],
        "specialSlices": {
            "negativeZero": list(NEGATIVE_ZERO_SLICE),
            "impulse": {"slice": list(IMPULSE_SLICE), "tap": list(IMPULSE_TAP)},
        },
        "metadataPreconditions": {
            target: (
                [{**CHANNEL_PRECONDITION}] if target in SOFTWARE_MATRIX_TARGETS else []
            )
            for target in TARGETS
        },
        "indexRangeAssertions": {
            "opengl": [
                {"expression": expression, "minimum": 0, "maximum": SIGNED_MAX}
                for expression in OPENGL_INDEX_RANGE_ASSERTIONS
            ],
            "metal": [],
            "directx": [],
        },
        "pendingOpenglIndexRangeAssertions": {
            stage: list(expressions)
            for stage, expressions in PENDING_OPENGL_INDEX_RANGE_ASSERTIONS.items()
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
        "pendingEntries": list(PENDING_ENTRIES),
        "pendingReason": (
            "the input and output transforms loop to params.C and params.O, "
            "which metadata preconditions cannot bound yet, and start their "
            "matrix loops at the SIMD-group index, which translation cannot yet "
            "prove uniform for OpenGL and DirectX software subgroup barriers"
        ),
        "artifactCountByTarget": len(ENTRIES),
        "dispatchCountByTarget": len(ENTRIES) * len(CASES),
        "checkedElementCountByEntry": {entry: _checked_elements() for entry in ENTRIES},
        "completeCorpusEntryCount": 17832,
        "partialThreadgroupsExercised": False,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_winograd_contract_is_exact():
    contract = _contract()
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
    pending = contract["pendingEntries"]
    assert tuple(pending) == PENDING_ENTRIES
    for entry, record in pending.items():
        assert set(record) == {"diagnostics"}
        # Metal translates every transform; OpenGL and DirectX fail closed.
        assert record["diagnostics"]["metal"] is None
        assert set(record["diagnostics"]) == set(TARGETS)
        for target in SOFTWARE_MATRIX_TARGETS:
            assert record["diagnostics"][target].startswith("project.translate.")
    assert contract["validationContract"] == _expected_validation_contract()
    assert contract["scope"] == _expected_scope()


def test_winograd_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "winograd.current-tree.translation.json" in readme
    assert len(ENTRIES) == len(set(ENTRIES)) == 3
    assert len(PENDING_ENTRIES) == 6
    assert set(ENTRIES) | set(PENDING_ENTRIES) == set(WINOGRAD_ENTRIES)
    # instantiate_winograd_conv_2d emits weight, input and output per type.
    assert WINOGRAD_ENTRIES[0::3] == ENTRIES
    assert [_dtype(entry) for entry in ENTRIES] == list(DTYPES)
    assert all(PENDING_PATTERN.fullmatch(entry) for entry in PENDING_ENTRIES)


def test_winograd_cases_follow_mlx_dispatch():
    for name, (channels, filters) in CASES.items():
        # conv_2D_gpu's Winograd conditions on the channel counts.
        assert channels % 32 == 0 and filters % 32 == 0, name
        data = _case_data(ENTRIES[0], name)
        assert data["grid"] == [filters // GROUP_FILTERS, 1, 1]
        assert len(data["arrays"]["wt_in"]) == (
            filters * FILTER * FILTER * channels + INPUT_GUARD_COUNT
        )
        assert all(-SIGNED_MAX <= item <= SIGNED_MAX for item in data["grid"])
    assert WORKGROUP == (32, GROUP_FILTERS, 1)
    assert any(channels > BLOCK_CHANNELS for channels, _ in CASES.values())
    assert any(filters > 32 for _, filters in CASES.values())


def test_winograd_tables_round_once():
    for row, items in zip(G, WT_TRANSFORM):
        for word, (numerator, denominator) in zip(row, items):
            # Metal and Mesa divide binary32 literals, which rounds once, and
            # DXC folds literal arithmetic in double first; both give RN32.
            assert word == _float32_bits(numerator / denominator)
        assert row[FILTER:] == (0,) * (TILE - FILTER)
    assert G[1][0] == 0xBE638E39  # -2/9
    assert G[3][0] == 0x3C360B61  # 1/90
    assert G[5][0] == 0x3F360B61  # 32/45
    assert [row[:FILTER] for row in GT[:FILTER]] == [
        tuple(G[column][row] for column in range(FILTER)) for row in range(FILTER)
    ]


def test_winograd_reference_matches_kernel_index_arithmetic():
    for name, (channels, filters) in CASES.items():
        written = _simulate_writes(channels, filters, [filters // GROUP_FILTERS, 1, 1])
        assert len(written) == TILE * TILE * channels * filters
        for index, (row, column, channel, filter_index) in written.items():
            assert index == _output_index(
                channels, filters, row, column, channel, filter_index
            )
    # Every lane owns a distinct row pair of the 8x8 matrix.
    pairs = {
        (sm, sn + offset)
        for lane in range(SIMD_SIZE)
        for sm, sn in [_lane_coordinates(lane)]
        for offset in (0, 1)
    }
    assert len(pairs) == TILE * TILE


@pytest.mark.parametrize("dtype", DTYPES)
def test_winograd_values_are_exact_and_guards_distinct(dtype):
    for name, (channels, filters) in CASES.items():
        taps = _taps(dtype, name)
        assert all(_quantize(dtype, value) == value for value in taps)
        # bfloat16 holds 1,792 values in these binades; the others far more.
        assert len(set(taps)) >= 1024
        live = _reference(dtype, name)
        assert GUARD_VALUE not in live
        assert all(math.isfinite(value) for value in live)
        assert all(_quantize(dtype, value) == value for value in live)
        zero = _reference_words(dtype, name)
        for row in range(TILE):
            for column in range(TILE):
                index = _output_index(
                    channels,
                    filters,
                    row,
                    column,
                    NEGATIVE_ZERO_SLICE[1],
                    NEGATIVE_ZERO_SLICE[0],
                )
                # The chains start from +0.0, so -0.0 taps give +0.0.
                assert zero[index] == 0
                impulse = _output_index(
                    channels,
                    filters,
                    row,
                    column,
                    IMPULSE_SLICE[1],
                    IMPULSE_SLICE[0],
                )
                kh, kw = IMPULSE_TAP
                expected = _oracle(G[row][kh], G[column][kw], 0)
                assert zero[impulse] == expected
    assert _quantize(dtype, GUARD_VALUE) == GUARD_VALUE


def _mutated(dtype, name, mutation, filter_index, channel):
    taps = _slice_words(dtype, name, filter_index, channel)
    if mutation == "unfused":
        return _transform(taps, step=_unfused)
    if mutation == "transposed-filter":
        return _transform(tuple(zip(*taps)))
    if mutation == "untransposed-right":
        return _transform(taps, right=G)
    if mutation == "neighbor-channel":
        channels, _ = CASES[name]
        return _transform(
            _slice_words(dtype, name, filter_index, (channel + 1) % channels)
        )
    if mutation == "neighbor-filter":
        _, filters = CASES[name]
        return _transform(
            _slice_words(dtype, name, (filter_index + 1) % filters, channel)
        )
    raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    (
        "unfused",
        "transposed-filter",
        "untransposed-right",
        "neighbor-channel",
        "neighbor-filter",
    ),
)
def test_winograd_reference_rejects_defects(mutation):
    dtype, name = DTYPES[0], "three-blocks"
    channels, filters = CASES[name]
    words = _reference_words(dtype, name)
    differing = 0
    for filter_index, channel in ((0, 0), (5, 40), (31, 95), (17, 64)):
        result = _mutated(dtype, name, mutation, filter_index, channel)
        expected = [
            words[_output_index(channels, filters, row, column, channel, filter_index)]
            for row in range(TILE)
            for column in range(TILE)
        ]
        actual = [result[row][column] for row in range(TILE) for column in range(TILE)]
        differing += sum(got != want for got, want in zip(actual, expected))
    assert differing >= 4, mutation


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("entry", ENTRIES + PENDING_ENTRIES)
def test_winograd_configs_scope_target_options(tmp_path, target, entry):
    config_path = tmp_path / "crosstl.toml"
    config_path.write_text(_config("out", target, entry), encoding="utf-8")
    config = load_project_config(tmp_path, config_path)
    assert list(config.targets) == [target]
    metal = config.source_options["metal"]
    assert metal["cooperative_matrix_fragment_mapping"] == FRAGMENT_MAPPING
    assert metal["preserve_resource_origins"] is True
    options = metal.get("target_options", {}).get(target, {})
    pending = entry in PENDING_ENTRIES
    if target in SOFTWARE_MATRIX_TARGETS:
        assert options["cooperative_matrix_software_lowering"] is True
        assert options["software_subgroup_width"] == SIMD_SIZE
        assert options["cooperative_matrix_mma_profile"] == MMA_PROFILE
        preconditions = [item.to_json() for item in config.metadata_preconditions]
        assert preconditions == (
            []
            if pending
            else [
                {
                    "source": SOURCE,
                    "entryPoint": entry,
                    **CHANNEL_PRECONDITION,
                }
            ]
        )
    else:
        assert options == {}
        assert not config.metadata_preconditions
    expressions = [item.expression for item in config.index_range_assertions]
    assert expressions == (
        list(_opengl_index_range_assertions(entry)) if target == "opengl" else []
    )
    text = config_path.read_text(encoding="utf-8")
    size = ", ".join(
        str(item) for item in (PENDING_WORKGROUP if pending else WORKGROUP)
    )
    assert f'"{entry}" = [{size}]' in text


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


def test_current_mlx_winograd_table_matches_source(current_mlx):
    root, _ = current_mlx
    text = (root / SOURCE).read_text(encoding="utf-8")
    match = re.search(
        r"wt_transform\[SIMD_MATRIX_SIZE\]\[SIMD_MATRIX_SIZE\] = \{(.*?)\};", text, re.S
    )
    assert match
    rows = re.findall(r"\{([^{}]*)\}", match.group(1))
    assert len(rows) == TILE
    for row, items in zip(rows, WT_TRANSFORM):
        values = [_literal_quotient(item) for item in row.split(",")]
        assert values == [Fraction(*item) for item in items]


def _literal_quotient(text):
    """An entry such as ``-2.0 / 9.00``: decimal literals divided exactly."""
    value, *divisors = (Fraction(part.strip()) for part in text.split("/"))
    for divisor in divisors:
        value /= divisor
    return value


def _translate(root, work, target, entry):
    """Translate one entry and check it against its pinned artifact."""
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
    assert execution["workgroupSize"] == list(WORKGROUP)
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


def _request(descriptor, package, target, entry, data):
    """Bind ``data`` by source parameter name and build one dispatch."""
    dtype = _dtype(entry)
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
        if name == OUTPUT_BUFFER:
            output = binding["name"]
        else:
            observed.append(binding["name"])
    assert bound == set(BUFFER_ORDER), bound
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output: dict(values[OUTPUT_BUFFER])},
        {"workgroupCount": data["grid"], "workgroupSize": list(WORKGROUP)},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, {OUTPUT_BUFFER: output}, len(observed)


def _payloads(entry, data):
    dtype = _dtype(entry)
    raw = {name: _metal_bytes(dtype, items) for name, items in data["arrays"].items()}
    for name, (_, items) in data["scalars"].items():
        raw[name] = struct.pack(f"<{len(items)}i", *items)
    return [raw[name] for name in BUFFER_ORDER]


def _upstream(work, reference, entry, label, payloads, grid):
    """Run the upstream metallib on raw buffers in ``[[buffer(i)]]`` order."""
    runner, library = reference
    files = []
    for index, payload in enumerate(payloads):
        path = work / f"{label}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    request = {
        "buffers": files,
        "workgroupCount": grid,
        "workgroupSize": list(WORKGROUP),
        "simdWidth": SIMD_SIZE,
    }
    path = work / f"{label}-upstream-request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    readback = work / f"{label}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, path, readback], work, f"{label}-upstream")
    )
    output_index = BUFFER_ORDER.index(OUTPUT_BUFFER)
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
    dtype = _dtype(entry)
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
                    output = outputs[OUTPUT_BUFFER]
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
                        "target": target,
                        "artifactSha256": record["sha256"],
                        "cases": evidence,
                        "comparison": "exact words",
                        "matrixProductProfile": (
                            MMA_PROFILE if target in SOFTWARE_MATRIX_TARGETS else None
                        ),
                        "indexRangeAssertions": (
                            list(OPENGL_INDEX_RANGE_ASSERTIONS)
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


@pytest.mark.parametrize("entry", PENDING_ENTRIES)
def test_current_mlx_winograd_transforms_stay_pending(current_mlx, entry):
    # The contract records why the input and output transforms are not
    # covered; the record must stay true, so a translator that starts
    # accepting them fails here until the entries join the gate.
    root, target = current_mlx
    expected = _contract()["pendingEntries"][entry]["diagnostics"][target]
    with tempfile.TemporaryDirectory(prefix=".current-winograd-", dir=root) as path:
        work = Path(path)
        config = work / "crosstl.toml"
        config.write_text(_config(f"{work.name}/out", target, entry), encoding="utf-8")
        report = translate_project(
            load_project_config(root, config),
            format_output=False,
            validate=True,
            run_toolchains=False,
        ).to_json()
    codes = {item["code"] for item in report["diagnostics"]}
    if expected is None:
        assert report["summary"]["translatedCount"] == 1, codes
        assert report["summary"]["failedCount"] == 0, codes
        return
    assert report["summary"]["translatedCount"] == 0
    assert report["summary"]["failedCount"] == 1
    assert expected in codes, codes


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
