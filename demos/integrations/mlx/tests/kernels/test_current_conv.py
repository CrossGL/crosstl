"""Native parity for the pinned MLX ``conv.metal`` unfold and depthwise kernels.

``conv.cpp`` unfolds the input for its explicit GEMM convolution with
``naive_unfold_Nd`` (``groups == 1``) or ``naive_unfold_transpose_Nd``
(grouped), runs small-kernel depthwise 2-D convolutions with
``depthwise_conv_2d`` and fully separable 1-D convolutions with
``depthwise_conv_1d``. This gate translates those 27 entries to Metal, OpenGL
and DirectX and executes them natively with ``conv.cpp``'s launch geometry.
The unfold kernels copy distinct values, and the depthwise inputs are quarters
and weights halves in ``[-1, 1]``, so every product and partial sum is exact in
the storage type and each output word is compared exactly. The nine Winograd
transforms have their own gate, ``test_current_winograd.py``.
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
from demos.integrations.mlx.tests.kernels.test_current_cross_entropy import (
    _from_bits,
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
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/conv.metal"
SOURCE_SHA256 = "d0604f69e144ffb65dcfc99e6383880b6d2d61645fa73147f35230034a1cd874"
HEADERS = {
    "mlx/backend/metal/kernels/steel/conv/params.h": (
        "354a832c5bf37791f4ad42ddc3ca9456a9c1ebbf3216650bb4a10c18110e03a8"
    ),
    "mlx/backend/metal/kernels/utils.h": (
        "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf"
    ),
}
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_CONV"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/conv.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
# conv.metal instantiates every kernel for float, half and bfloat16_t.
DTYPES = ("float32", "float16", "bfloat16")
UNFOLD_ENTRIES = tuple(
    f"{kernel}_{dtype}_{spatial}"
    for dtype in DTYPES
    for spatial in (1, 2, 3)
    for kernel in ("naive_unfold_nd", "naive_unfold_transpose_nd")
)
DW2_ENTRIES = tuple(f"depthwise_conv_2d_{dtype}" for dtype in DTYPES)
DW1_ENTRIES = tuple(
    f"depthwise_conv_1d_{dtype}{suffix}"
    for dtype in DTYPES
    for suffix in ("", "_large")
)
ENTRIES = UNFOLD_ENTRIES + DW2_ENTRIES + DW1_ENTRIES
WINOGRAD_ENTRIES = tuple(
    f"winograd_conv_2d_{stage}_transform_{dtype}_{block}"
    for dtype in ("float32", "bfloat16", "float16")
    for stage, block in (("weight", "bc32"), ("input", "bc32"), ("output", "bo32"))
)
UNFOLD_PATTERN = re.compile(
    r"naive_unfold(?P<transpose>_transpose)?_nd_"
    r"(?P<dtype>float32|float16|bfloat16)_(?P<spatial>[123])"
)
DW2_PATTERN = re.compile(r"depthwise_conv_2d_(?P<dtype>float32|float16|bfloat16)")
DW1_PATTERN = re.compile(
    r"depthwise_conv_1d_(?P<dtype>float32|float16|bfloat16)(?P<large>_large)?"
)
SIMD_SIZE = 32
SIGNED_MAX = 2**31 - 1
# depthwise_conv_2d indexes its threadgroup tile with simd_gid * 32 + simd_lid,
# which is the linear thread index only for 32-lane SIMD groups.
SOFTWARE_SUBGROUP_TARGETS = ("opengl", "directx")
# The depthwise kernels accumulate into `float o = 0.`. GLSL and HLSL compilers
# may fold that +0.0 away: Mesa folds 0.0 + x to x once a 1x1 kernel unrolls,
# even for a `precise` addition, so a negative-zero product would end as -0.0
# where Metal's IEEE addition gives +0.0. The software binary32 addition, with
# Metal's flushed subnormal results, works on the operands' bits, which no
# floating-point rewrite can drop.
PROFILED_ADDITION_TARGETS = ("opengl", "directx")
PROFILED_ADDITION_KINDS = ("dw2", "dw1", "dw1_large")
ADDITIVE_PROFILE = "rne-flush"

# conv.cpp: the unfold kernels launch (C, prod(wS), tile_rows) threads in
# (min(tgp_x, C), min(tgp_y, prod(wS)), 1) groups with tgp_x = 32 for C <= 32
# and tgp_y = 256 / tgp_x. Metal clips the last groups of dispatch_threads;
# OpenGL and DirectX cannot, so the cases use C = 32 and prod(wS) % 8 == 0,
# whose grids those groups tile exactly.
UNFOLD_CHANNELS = 32
UNFOLD_OUT_CHANNELS = 8
# naive_unfold_transpose_nd serves grouped convolutions.
UNFOLD_GROUPS = 4
UNFOLD_WORKGROUP = (32, 8, 1)
# depthwise_conv_2D_gpu: (tc, tw, th) groups over (C / tc, ceil(oW / tw),
# ceil(oH / th) * N), with tgp_h = th and tgp_w = tw as function constants.
DW2_TILE = {"tc": 8, "tw": 8, "th": 4}
DW2_WORKGROUP = (8, 8, 4)
# depthwise_conv_1D_gpu: (D, Tout, B) threads in get_block_dims(D, Tout, B)
# groups, which is (16, 16, 4) for D, Tout >= 16 and 4 <= B < 8. The cases
# use B = 4 and multiples of 16.
DW1_WORKGROUP = (16, 16, 4)
DW1_BATCH = 4
WORKGROUPS = {
    "unfold": UNFOLD_WORKGROUP,
    "unfold_transpose": UNFOLD_WORKGROUP,
    "dw2": DW2_WORKGROUP,
    "dw1": DW1_WORKGROUP,
    "dw1_large": DW1_WORKGROUP,
}
# conv.metal's depthwise function constants, as conv.cpp sets them.
CONSTANT_IDS = {
    "ker_h": 0,
    "ker_w": 1,
    "str_h": 10,
    "str_w": 11,
    "tgp_h": 100,
    "tgp_w": 101,
    "do_flip": 200,
}
# depthwise_conv_2D_gpu takes kernels up to 7x7 with strides up to 2:
# name -> (ker_h, ker_w, str_h, str_w, do_flip).
DW2_VARIANTS = {
    "k3x3": (3, 3, 1, 1, False),
    "k3x3-s2": (3, 3, 2, 2, False),
    "k5x5-flip": (5, 5, 1, 1, True),
    "k7x7-s2-flip": (7, 7, 2, 2, True),
    "k1x1": (1, 1, 1, 1, False),
    "k3x5-s1x2-flip": (3, 5, 1, 2, True),
}
DEFAULT_VARIANT = "default"
# Buffer indices in conv.metal's signatures.
BUFFER_ORDER = {
    "unfold": ("in_", "out_", "params", "row_offset"),
    "unfold_transpose": ("in_", "out_", "params", "row_offset"),
    "dw2": ("in_", "wt", "out_", "params"),
    "dw1": ("in_", "w", "out_", "strides", "kernel_size"),
    "dw1_large": ("in_", "w", "out_", "strides", "kernel_size"),
}
OUTPUT_BUFFER = "out_"
# OpenGL indexes 32-bit storage arrays. These host preconditions bound the
# composed element offsets to signed int range; the cases stay far below it.
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "unfold": (
        "uint64(gid.z) * filter_size + uint64(gid.y) * *params.C",
        "out_ + gid.x",
        "gid.x",
        "in_offset + gid.x",
    ),
    "unfold_transpose": (
        "uint64(gid.z) * filter_size + uint64(gid.x) * filter_size / *params.C",
        "out_ + 0",
        "in_offset + gid.x",
    ),
    "dw2": (
        "n * params.in_strides[0]",
        "in_offset + ih * params.in_strides[1] + iw * params.in_strides[2] + tg_c",
        "in_load + c_per_thr * thr_c + cc",
        "c * params.wt_strides[0]",
        "wt + wt_h * ker_w + wt_w",
        "n * params.out_strides[0] + oh * params.out_strides[1]"
        " + ow * params.out_strides[2]",
        "out_ + c",
    ),
    "dw1": (),
    "dw1_large": (
        "tid.z * int64(grid_dim.y) + tid.y * grid_dim.x + tid.x",
        "tid.z * buffer_load(strides, 0) + tid.y * buffer_load(strides, 1)"
        " + tid.x * buffer_load(strides, 2)",
        "in_ + 0",
        "buffer_load(strides, 1)",
        "in_ + buffer_load(strides, 1)",
        "out_ + 0",
    ),
}
# Reflected MLXConvParams<N> stride on every target: Metal's natural layout,
# which std430 and HLSL structured buffers reproduce with a 4-byte bool.
PARAMS_SIZES = {1: 120, 2: 176, 3: 224}
# depthwise_conv_2d reads its MLXConvParams<2> through a reference, which
# OpenGL and DirectX bind as a uniform block and a constant buffer.
PARAMS_BLOCK_SIZES = {"opengl": 448, "directx": 432}

# Distinct, nonzero, finite and normal values for the copies: word base + k
# for the k-th pair, the odd member negated.
DISTINCT_BASE = {"float32": 0x3F800000, "float16": 0x3000, "bfloat16": 0x3800}
DISTINCT_STEP = {"float32": 0x100, "float16": 1, "bfloat16": 1}
SIGN_BIT = {"float32": 0x80000000, "float16": 0x8000, "bfloat16": 0x8000}
# Quarters times halves in [-1, 1] are eighths, exact in every storage type;
# sums stay below 32, which bfloat16's eight significand bits hold exactly.
INPUT_LEVELS = tuple(index / 4 for index in range(-4, 5))
WEIGHT_LEVELS = tuple(index / 2 for index in range(-2, 3))
EXACT_SUM_LIMIT = 32.0
# -2^15 is exact in every type and differs from every input and result.
GUARD_VALUE = -32768.0
GUARD_COUNT = 16
INPUT_GUARD_COUNT = 16

# Unfold cases: N, spatial input iS, kernel wS, strides, low and high
# padding, kernel and input dilation, flip and, for later row tiles of a
# large unfold, (row_offset, tile_rows).
UNFOLD_CASES = {
    1: {
        "valid": dict(
            batch=2,
            iS=(9,),
            wS=(8,),
            str=(1,),
            padLo=(0,),
            padHi=(0,),
            kdil=(1,),
            idil=(1,),
            flip=False,
        ),
        "padded-strided-flip": dict(
            batch=1,
            iS=(11,),
            wS=(8,),
            str=(2,),
            padLo=(3,),
            padHi=(4,),
            kdil=(1,),
            idil=(1,),
            flip=True,
        ),
        "kernel-dilated": dict(
            batch=1,
            iS=(20,),
            wS=(8,),
            str=(3,),
            padLo=(2,),
            padHi=(2,),
            kdil=(2,),
            idil=(1,),
            flip=False,
        ),
        "input-dilated-last-tile": dict(
            batch=3,
            iS=(7,),
            wS=(16,),
            str=(1,),
            padLo=(8,),
            padHi=(7,),
            kdil=(1,),
            idil=(2,),
            flip=False,
            rows=(32, 7),
        ),
    },
    2: {
        "valid": dict(
            batch=2,
            iS=(5, 6),
            wS=(2, 4),
            str=(1, 1),
            padLo=(0, 0),
            padHi=(0, 0),
            kdil=(1, 1),
            idil=(1, 1),
            flip=False,
        ),
        "padded-strided-flip": dict(
            batch=1,
            iS=(7, 9),
            wS=(4, 2),
            str=(2, 3),
            padLo=(1, 2),
            padHi=(2, 1),
            kdil=(1, 2),
            idil=(1, 1),
            flip=True,
        ),
        "input-dilated-middle-tile": dict(
            batch=2,
            iS=(4, 5),
            wS=(2, 4),
            str=(1, 1),
            padLo=(1, 3),
            padHi=(1, 3),
            kdil=(1, 1),
            idil=(2, 1),
            flip=False,
            rows=(48, 48),
        ),
    },
    3: {
        "valid": dict(
            batch=1,
            iS=(3, 4, 5),
            wS=(2, 2, 2),
            str=(1, 1, 1),
            padLo=(0, 0, 0),
            padHi=(0, 0, 0),
            kdil=(1, 1, 1),
            idil=(1, 1, 1),
            flip=False,
        ),
        "strided-dilated-flip": dict(
            batch=2,
            iS=(4, 3, 5),
            wS=(1, 2, 4),
            str=(2, 1, 2),
            padLo=(0, 1, 2),
            padHi=(1, 0, 1),
            kdil=(1, 2, 1),
            idil=(1, 1, 2),
            flip=True,
        ),
        "padded-last-tile": dict(
            batch=2,
            iS=(3, 3, 3),
            wS=(2, 2, 2),
            str=(1, 1, 1),
            padLo=(1, 1, 1),
            padHi=(1, 1, 1),
            kdil=(1, 1, 1),
            idil=(1, 1, 1),
            flip=False,
            rows=(96, 32),
        ),
    },
}
# Depthwise 2-D cases: N, (H, W), C and padding. "same" pads k // 2 on both
# sides; "skewed" pads k_h - 1 rows above and k_w // 2 columns on the right.
DW2_CASES = {
    "single": (1, (6, 10), 16, "same"),
    "batched": (2, (9, 13), 32, "skewed"),
}
# Depthwise 1-D cases: (D, Tout, K) with B = 4; conv.cpp only takes this
# path for stride 1, no padding or dilation and groups == C == O.
DW1_CASES = {"pointwise": (16, 16, 1), "short": (32, 16, 3), "long": (16, 48, 9)}


def _parse(entry):
    match = UNFOLD_PATTERN.fullmatch(entry)
    if match is not None:
        return {
            "kind": "unfold_transpose" if match["transpose"] else "unfold",
            "dtype": match["dtype"],
            "spatial": int(match["spatial"]),
        }
    match = DW2_PATTERN.fullmatch(entry)
    if match is not None:
        return {"kind": "dw2", "dtype": match["dtype"], "spatial": 2}
    match = DW1_PATTERN.fullmatch(entry)
    assert match is not None, entry
    return {
        "kind": "dw1_large" if match["large"] else "dw1",
        "dtype": match["dtype"],
        "spatial": 1,
    }


def _kind(entry):
    return _parse(entry)["kind"]


def _dtype(entry):
    return _parse(entry)["dtype"]


def _dw2_constants(key):
    ker_h, ker_w, str_h, str_w, flip = DW2_VARIANTS[key]
    return {
        "ker_h": ker_h,
        "ker_w": ker_w,
        "str_h": str_h,
        "str_w": str_w,
        "tgp_h": DW2_TILE["th"],
        "tgp_w": DW2_TILE["tw"],
        "do_flip": flip,
    }


def _variants(entry):
    if _kind(entry) == "dw2":
        return tuple((key, _dw2_constants(key)) for key in DW2_VARIANTS)
    return ((DEFAULT_VARIANT, {}),)


VARIANTS = tuple(
    (entry, key, constants) for entry in ENTRIES for key, constants in _variants(entry)
)


def _cases(entry):
    info = _parse(entry)
    if info["kind"].startswith("unfold"):
        return tuple(UNFOLD_CASES[info["spatial"]])
    if info["kind"] == "dw2":
        return tuple(DW2_CASES)
    return tuple(DW1_CASES)


# ---------------------------------------------------------------------------
# MLX launch geometry
# ---------------------------------------------------------------------------


def _mlx_unfold_group(channels, kernel_points):
    """``explicit_gemm_conv_ND_gpu``'s group size for its unfold grid."""
    tgp_x = min(channels, 64)
    tgp_x = 32 * ((tgp_x + 32 - 1) // 32)
    tgp_y = 256 // tgp_x
    return (min(tgp_x, channels), min(tgp_y, kernel_points), 1)


def _mlx_block_dims(dim0, dim1, dim2, pow2=10):
    """``get_block_dims_common`` from mlx/backend/common/utils.cpp."""
    pows = [0, 0, 0]
    total = 0
    while True:
        presum = total
        if dim0 >= (1 << (pows[0] + 1)):
            pows[0] += 1
            total += 1
        if total == 10:
            break
        if dim1 >= (1 << (pows[1] + 1)):
            pows[1] += 1
            total += 1
        if total == 10:
            break
        if dim2 >= (1 << (pows[2] + 1)):
            pows[2] += 1
            total += 1
        if total == presum or total == pow2:
            break
    return tuple(1 << value for value in pows)


def _out_extent(size, kernel, stride, low, high, kdil, idil):
    """Output extent for MLX's dilated, padded convolution geometry."""
    dilated_in = 1 + idil * (size - 1)
    dilated_kernel = 1 + kdil * (kernel - 1)
    return (dilated_in + low + high - dilated_kernel) // stride + 1


def _contiguous(shape):
    strides, step = [], 1
    for extent in reversed(shape):
        strides.append(step)
        step *= extent
    return strides[::-1]


def _unravel(index, shape):
    coordinates = []
    for extent in reversed(shape):
        index, remainder = divmod(index, extent)
        coordinates.append(remainder)
    return coordinates[::-1]


def _c_div(left, right):
    quotient = abs(left) // abs(right)
    return quotient if (left >= 0) == (right > 0) else -quotient


def _c_mod(left, right):
    return left - _c_div(left, right) * right


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _distinct(dtype, count):
    words = [
        (DISTINCT_BASE[dtype] + (index >> 1) * DISTINCT_STEP[dtype])
        | (SIGN_BIT[dtype] if index & 1 else 0)
        for index in range(count)
    ]
    return [_from_bits(dtype, word) for word in words]


def _draw(salt, count, levels):
    state = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], "little")
    values = []
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        values.append(levels[(state >> 33) % len(levels)])
    return values


def _guarded(values):
    return list(values) + [math.nan] * INPUT_GUARD_COUNT


def _params_bytes(params):
    """``MLXConvParams<N>`` in Metal's natural C layout."""
    spatial = len(params["iS"])
    head = struct.pack(
        f"<3i{7 * spatial}i",
        params["N"],
        params["C"],
        params["O"],
        *params["iS"],
        *params["wS"],
        *params["oS"],
        *params["str"],
        *params["pad"],
        *params["kdil"],
        *params["idil"],
    )
    head += bytes(-len(head) % 8)
    body = struct.pack(
        f"<{3 * (spatial + 2)}q",
        *params["in_strides"],
        *params["wt_strides"],
        *params["out_strides"],
    )
    data = head + body + struct.pack("<i?", params["groups"], params["flip"])
    return data + bytes(-len(data) % 8)


# ---------------------------------------------------------------------------
# Unfold
# ---------------------------------------------------------------------------


def _unfold_geometry(spatial, name):
    case = UNFOLD_CASES[spatial][name]
    out_shape = tuple(
        _out_extent(*values)
        for values in zip(
            case["iS"],
            case["wS"],
            case["str"],
            case["padLo"],
            case["padHi"],
            case["kdil"],
            case["idil"],
        )
    )
    rows_total = case["batch"] * math.prod(out_shape)
    row_offset, rows = case.get("rows") or (0, rows_total)
    points = math.prod(case["wS"])
    return {
        "case": case,
        "oS": out_shape,
        "implicitM": rows_total,
        "rowOffset": row_offset,
        "rows": rows,
        "kernelPoints": points,
        "filterSize": UNFOLD_CHANNELS * points,
        "inShape": (case["batch"], *case["iS"], UNFOLD_CHANNELS),
        "grid": [
            UNFOLD_CHANNELS // UNFOLD_WORKGROUP[0],
            points // UNFOLD_WORKGROUP[1],
            rows,
        ],
    }


def _unfold_params(transpose, geometry):
    case = geometry["case"]
    groups = UNFOLD_GROUPS if transpose else 1
    wt_shape = (UNFOLD_OUT_CHANNELS, *case["wS"], UNFOLD_CHANNELS // groups)
    out_shape = (case["batch"], *geometry["oS"], UNFOLD_OUT_CHANNELS)
    return {
        "N": case["batch"],
        "C": UNFOLD_CHANNELS,
        "O": UNFOLD_OUT_CHANNELS,
        "iS": list(case["iS"]),
        "wS": list(case["wS"]),
        "oS": list(geometry["oS"]),
        "str": list(case["str"]),
        "pad": list(case["padLo"]),
        "kdil": list(case["kdil"]),
        "idil": list(case["idil"]),
        "in_strides": _contiguous(geometry["inShape"]),
        "wt_strides": _contiguous(wt_shape),
        "out_strides": _contiguous(out_shape),
        "groups": groups,
        "flip": case["flip"],
    }


def _unfold_reference(transpose, geometry, mutation=None):
    """Source element of every unfolded slot (None for zero), from the
    convolution's definition rather than the kernel's loop structure."""
    case = dict(geometry["case"])
    spatial = len(case["iS"])
    if mutation == "flip":
        case["flip"] = not case["flip"]
    for name, field in (
        ("padding", "padLo"),
        ("stride", "str"),
        ("kernel-dilation", "kdil"),
        ("input-dilation", "idil"),
    ):
        if mutation == name:
            case[field] = (0 if field == "padLo" else 1,) * spatial
    if mutation == "layout":
        transpose = not transpose
    channels, points = UNFOLD_CHANNELS, geometry["kernelPoints"]
    strides = _contiguous(geometry["inShape"])
    pixels = math.prod(geometry["oS"])
    row_offset = 0 if mutation == "row-offset" else geometry["rowOffset"]
    sources = []
    for row in range(geometry["rows"]):
        batch, pixel = divmod(row_offset + row, pixels)
        if mutation == "batch":
            batch = 0
        position = _unravel(pixel, geometry["oS"])
        slots = [None] * geometry["filterSize"]
        for point in range(points):
            offset = None
            if batch < case["batch"]:
                offset = batch * strides[0]
                tap = _unravel(point, case["wS"])
                for axis in range(spatial):
                    taken = (
                        case["wS"][axis] - 1 - tap[axis] if case["flip"] else tap[axis]
                    )
                    dilated = (
                        position[axis] * case["str"][axis]
                        - case["padLo"][axis]
                        + taken * case["kdil"][axis]
                    )
                    limit = case["idil"][axis] * (case["iS"][axis] - 1)
                    if not 0 <= dilated <= limit or dilated % case["idil"][axis]:
                        offset = None
                        break
                    offset += dilated // case["idil"][axis] * strides[axis + 1]
            for channel in range(channels):
                slot = (
                    channel * points + point
                    if transpose
                    else point * channels + channel
                )
                if offset is None:
                    slots[slot] = "guard" if mutation == "zero-fill" else None
                else:
                    shift = (
                        (channel + 1) % channels if mutation == "channel" else channel
                    )
                    slots[slot] = offset + shift
        sources += slots
    return sources


def _unfold_simulate(transpose, params, row_offset, grid):
    """Run naive_unfold[_transpose]_Nd's own arithmetic for every thread."""
    spatial = len(params["iS"])
    written = {}
    filter_size = params["C"]
    for axis in range(spatial):
        filter_size *= params["wS"][axis]
    out_pixels = math.prod(params["oS"])
    for z in range(grid[2]):
        for y in range(grid[1]):
            for x in range(grid[0]):
                if transpose:
                    out = z * filter_size + x * (filter_size // params["C"])
                else:
                    out = z * filter_size + y * params["C"]
                global_row = row_offset + z
                n = global_row // out_pixels
                pixel = global_row % out_pixels
                tap = y
                valid = n < params["N"]
                coordinates = [0] * spatial
                kernel_stride = 1
                for axis in reversed(range(spatial)):
                    os_ = pixel % params["oS"][axis]
                    ws_ = tap % params["wS"][axis]
                    if transpose:
                        out += ws_ * kernel_stride
                    if params["flip"]:
                        ws_ = params["wS"][axis] - ws_ - 1
                    is_ = (
                        os_ * params["str"][axis]
                        - params["pad"][axis]
                        + ws_ * params["kdil"][axis]
                    )
                    is_max = 1 + params["idil"][axis] * (params["iS"][axis] - 1)
                    valid = valid and (
                        is_ >= 0
                        and is_ < is_max
                        and _c_mod(is_, params["idil"][axis]) == 0
                    )
                    coordinates[axis] = _c_div(is_, params["idil"][axis])
                    pixel //= params["oS"][axis]
                    tap //= params["wS"][axis]
                    if transpose:
                        kernel_stride *= params["wS"][axis]
                slot = out if transpose else out + x
                assert slot not in written
                if valid:
                    offset = n * params["in_strides"][0]
                    for axis in range(spatial):
                        offset += coordinates[axis] * params["in_strides"][axis + 1]
                    written[slot] = offset + x
                else:
                    written[slot] = None
    return [written[slot] for slot in range(len(written))]


@functools.lru_cache(maxsize=None)
def _unfold_sources(transpose, spatial, name, mutation=None):
    return tuple(
        _unfold_reference(transpose, _unfold_geometry(spatial, name), mutation)
    )


def _unfold_data(entry, name):
    info = _parse(entry)
    transpose = info["kind"] == "unfold_transpose"
    geometry = _unfold_geometry(info["spatial"], name)
    values = _distinct(info["dtype"], math.prod(geometry["inShape"]))
    sources = _unfold_sources(transpose, info["spatial"], name)
    live = [0.0 if source is None else values[source] for source in sources]
    return {
        "geometry": geometry,
        "arrays": {
            "in_": _guarded(values),
            "out_": [GUARD_VALUE] * (len(live) + GUARD_COUNT),
        },
        "scalars": {"row_offset": ("int32", [geometry["rowOffset"]])},
        "params": _unfold_params(transpose, geometry),
        "expected": live + [GUARD_VALUE] * GUARD_COUNT,
        "grid": geometry["grid"],
    }


# ---------------------------------------------------------------------------
# Depthwise 2-D
# ---------------------------------------------------------------------------


def _dw2_geometry(key, name):
    ker_h, ker_w, str_h, str_w, flip = DW2_VARIANTS[key]
    batch, (height, width), channels, padding = DW2_CASES[name]
    if padding == "same":
        low = (ker_h // 2, ker_w // 2)
        high = low
    else:
        low, high = (ker_h - 1, 0), (0, ker_w // 2)
    out_shape = tuple(
        _out_extent(size, kernel, stride, lo, hi, 1, 1)
        for size, kernel, stride, lo, hi in zip(
            (height, width), (ker_h, ker_w), (str_h, str_w), low, high
        )
    )
    tile = DW2_TILE
    return {
        "batch": batch,
        "iS": (height, width),
        "C": channels,
        "kernel": (ker_h, ker_w),
        "stride": (str_h, str_w),
        "flip": flip,
        "padLo": low,
        "padHi": high,
        "oS": out_shape,
        "inShape": (batch, height, width, channels),
        "wtShape": (channels, ker_h, ker_w, 1),
        "outShape": (batch, *out_shape, channels),
        "grid": [
            channels // tile["tc"],
            (out_shape[1] + tile["tw"] - 1) // tile["tw"],
            (out_shape[0] + tile["th"] - 1) // tile["th"] * batch,
        ],
    }


def _dw2_params(geometry):
    return {
        "N": geometry["batch"],
        "C": geometry["C"],
        "O": geometry["C"],
        "iS": list(geometry["iS"]),
        "wS": list(geometry["kernel"]),
        "oS": list(geometry["oS"]),
        "str": list(geometry["stride"]),
        "pad": list(geometry["padLo"]),
        "kdil": [1, 1],
        "idil": [1, 1],
        "in_strides": _contiguous(geometry["inShape"]),
        "wt_strides": _contiguous(geometry["wtShape"]),
        "out_strides": _contiguous(geometry["outShape"]),
        "groups": geometry["C"],
        "flip": geometry["flip"],
    }


@functools.lru_cache(maxsize=None)
def _dw2_values(key, name):
    geometry = _dw2_geometry(key, name)
    return (
        tuple(
            _draw(f"dw2:{key}:{name}:in", math.prod(geometry["inShape"]), INPUT_LEVELS)
        ),
        tuple(
            _draw(f"dw2:{key}:{name}:wt", math.prod(geometry["wtShape"]), WEIGHT_LEVELS)
        ),
    )


@functools.lru_cache(maxsize=None)
def _dw2_reference(key, name, mutation=None):
    """out[n, oh, ow, c] = sum in[n, oh*s - p + h, ow*s - p + w, c] * wt[c, h, w]
    over the (flipped) kernel, with zeros outside the input."""
    geometry = _dw2_geometry(key, name)
    inputs, weights = _dw2_values(key, name)
    batch, (height, width), channels = geometry["batch"], geometry["iS"], geometry["C"]
    ker_h, ker_w = geometry["kernel"]
    str_h, str_w = (1, 1) if mutation == "stride" else geometry["stride"]
    pad_h, pad_w = (0, 0) if mutation == "padding" else geometry["padLo"]
    flip = geometry["flip"] != (mutation == "flip")
    out_h, out_w = geometry["oS"]
    expected = []
    for n in range(batch):
        source = 0 if mutation == "batch" else n
        for oh in range(out_h):
            for ow in range(out_w):
                for c in range(channels):
                    read = (c + 1) % channels if mutation == "channel" else c
                    total = 0.0
                    for h in range(ker_h):
                        for w in range(ker_w):
                            ih = oh * str_h - pad_h + h
                            iw = ow * str_w - pad_w + w
                            if not (0 <= ih < height and 0 <= iw < width):
                                continue
                            fh, fw = (ker_h - 1 - h, ker_w - 1 - w) if flip else (h, w)
                            tap = (
                                fw * ker_h + fh
                                if mutation == "kernel-transpose"
                                else fh * ker_w + fw
                            )
                            total += (
                                inputs[
                                    ((source * height + ih) * width + iw) * channels
                                    + read
                                ]
                                * weights[c * ker_h * ker_w + tap]
                            )
                    expected.append(total)
    return tuple(expected)


def _dw2_simulate(constants, params, inputs, weights, grid):
    """Run depthwise_conv_2d's tile load and accumulation for every thread."""
    tc, tw, th = DW2_TILE["tc"], DW2_TILE["tw"], DW2_TILE["th"]
    c_per_thr = 8
    tgh, tgw, tgc = th * 2 + 6, tw * 2 + 6, tc
    ker_h, ker_w = constants["ker_h"], constants["ker_w"]
    str_h, str_w = constants["str_h"], constants["str_w"]
    span_h = constants["tgp_h"] * str_h + ker_h - 1
    span_w = constants["tgp_w"] * str_w + ker_w - 1
    assert span_h <= tgh and span_w <= tgw
    span_hw = span_h * span_w
    n_threads = th * tw * tc
    thr_per_hw = tc // c_per_thr
    hw_per_group = n_threads // thr_per_hw
    n_tgblocks_h = (params["oS"][0] + th - 1) // th
    written = {}
    for tz in range(grid[2]):
        for ty in range(grid[1]):
            for tx in range(grid[0]):
                n, tghid = divmod(tz, n_tgblocks_h)
                base = n * params["in_strides"][0]
                tile = [None] * (tgh * tgw * tgc)
                tg_oh = tghid * th * str_h - params["pad"][0]
                tg_ow = ty * tw * str_w - params["pad"][1]
                tg_c = tx * tc
                # thread_idx = simd_gid * 32 + simd_lid: the linear index.
                for thread_idx in range(n_threads):
                    thr_c = thread_idx % thr_per_hw
                    thr_hw = thread_idx // thr_per_hw
                    for hw in range(thr_hw, span_hw, hw_per_group):
                        h, w = divmod(hw, span_w)
                        ih, iw = tg_oh + h, tg_ow + w
                        offset = h * span_w * tgc + w * tgc
                        inside = 0 <= ih < params["iS"][0] and 0 <= iw < params["iS"][1]
                        load = (
                            base
                            + ih * params["in_strides"][1]
                            + iw * params["in_strides"][2]
                            + tg_c
                        )
                        for cc in range(c_per_thr):
                            slot = offset + c_per_thr * thr_c + cc
                            tile[slot] = (
                                inputs[load + c_per_thr * thr_c + cc] if inside else 0.0
                            )
                for lz in range(th):
                    for ly in range(tw):
                        for lx in range(tc):
                            oh, ow, c = tghid * th + lz, ty * tw + ly, tg_c + lx
                            wt = c * params["wt_strides"][0]
                            pointer = lz * str_h * span_w * tgc + ly * str_w * tgc + lx
                            total = 0.0
                            for h in range(ker_h):
                                for w in range(ker_w):
                                    wt_h, wt_w = h, w
                                    if constants["do_flip"]:
                                        wt_h, wt_w = ker_h - h - 1, ker_w - w - 1
                                    value = tile[pointer + h * span_w * tgc + w * tgc]
                                    assert value is not None
                                    total += value * weights[wt + wt_h * ker_w + wt_w]
                            if oh >= params["oS"][0] or ow >= params["oS"][1]:
                                continue
                            slot = (
                                n * params["out_strides"][0]
                                + oh * params["out_strides"][1]
                                + ow * params["out_strides"][2]
                                + c
                            )
                            assert slot not in written
                            written[slot] = total
    return [written[slot] for slot in range(len(written))]


def _dw2_data(entry, key, name):
    geometry = _dw2_geometry(key, name)
    inputs, weights = _dw2_values(key, name)
    live = list(_dw2_reference(key, name))
    return {
        "geometry": geometry,
        "arrays": {
            "in_": _guarded(inputs),
            "wt": _guarded(weights),
            "out_": [GUARD_VALUE] * (len(live) + GUARD_COUNT),
        },
        "scalars": {},
        "params": _dw2_params(geometry),
        "expected": live + [GUARD_VALUE] * GUARD_COUNT,
        "grid": geometry["grid"],
    }


# ---------------------------------------------------------------------------
# Depthwise 1-D
# ---------------------------------------------------------------------------


def _dw1_geometry(name):
    channels, steps, taps = DW1_CASES[name]
    return {
        "B": DW1_BATCH,
        "D": channels,
        "Tout": steps,
        "K": taps,
        "Tin": steps + taps - 1,
        "strides": [(steps + taps - 1) * channels, channels, 1],
        "grid": [
            channels // DW1_WORKGROUP[0],
            steps // DW1_WORKGROUP[1],
            DW1_BATCH // DW1_WORKGROUP[2],
        ],
    }


@functools.lru_cache(maxsize=None)
def _dw1_values(name):
    geometry = _dw1_geometry(name)
    count = geometry["B"] * geometry["Tin"] * geometry["D"]
    return (
        tuple(_draw(f"dw1:{name}:in", count, INPUT_LEVELS)),
        tuple(_draw(f"dw1:{name}:w", geometry["D"] * geometry["K"], WEIGHT_LEVELS)),
    )


@functools.lru_cache(maxsize=None)
def _dw1_reference(name, mutation=None):
    """out[b, t, d] = sum_i in[b, t + i, d] * w[d, i]."""
    geometry = _dw1_geometry(name)
    inputs, weights = _dw1_values(name)
    batch, channels, steps, taps = (geometry[key] for key in ("B", "D", "Tout", "K"))
    rows = geometry["Tin"]
    used = taps - 1 if mutation == "taps" else taps
    expected = []
    for b in range(batch):
        source = 0 if mutation == "batch" else b
        for t in range(steps):
            for d in range(channels):
                total = 0.0
                for i in range(used):
                    weight = (
                        weights[i * channels + d]
                        if mutation == "weights"
                        else weights[d * taps + i]
                    )
                    # The "time" defect advances by the channel stride.
                    flat = (
                        (source * rows + t) * channels + d + i
                        if mutation == "time"
                        else (source * rows + t + i) * channels + d
                    )
                    total += inputs[flat % len(inputs)] * weight
                expected.append(total)
    return tuple(expected)


def _dw1_simulate(strides, taps, inputs, weights, grid_threads):
    """Run depthwise_conv_1d's pointer arithmetic for every thread."""
    gx, gy, gz = grid_threads
    written = {}
    for z in range(gz):
        for y in range(gy):
            for x in range(gx):
                out = (z * gy + y) * gx + x
                source = z * strides[0] + y * strides[1] + x * strides[2]
                wt = x * taps
                total = 0.0
                for i in range(taps):
                    total += inputs[source] * weights[wt + i]
                    source += strides[1]
                assert out not in written
                written[out] = total
    return [written[slot] for slot in range(len(written))]


def _dw1_data(entry, name):
    geometry = _dw1_geometry(name)
    inputs, weights = _dw1_values(name)
    live = list(_dw1_reference(name))
    index_type = "int64" if _kind(entry) == "dw1_large" else "int32"
    return {
        "geometry": geometry,
        "arrays": {
            "in_": _guarded(inputs),
            "w": _guarded(weights),
            "out_": [GUARD_VALUE] * (len(live) + GUARD_COUNT),
        },
        "scalars": {
            "strides": (index_type, geometry["strides"]),
            "kernel_size": ("int32", [geometry["K"]]),
        },
        "params": None,
        "expected": live + [GUARD_VALUE] * GUARD_COUNT,
        "grid": geometry["grid"],
    }


def _case_data(entry, key, name):
    kind = _kind(entry)
    if kind.startswith("unfold"):
        return _unfold_data(entry, name)
    if kind == "dw2":
        return _dw2_data(entry, key, name)
    return _dw1_data(entry, name)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def _toml(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


def _profiled_additions(target, entry):
    return (
        target in PROFILED_ADDITION_TARGETS and _kind(entry) in PROFILED_ADDITION_KINDS
    )


def _config(output, target, entry, constants):
    kind = _kind(entry)
    options = ""
    if constants:
        options += "\n[project.specialization_constants]\n"
        for name, value in constants.items():
            options += f'"{CONSTANT_IDS[name]}" = {_toml(value)}\n'
    target_options = []
    if kind == "dw2" and target in SOFTWARE_SUBGROUP_TARGETS:
        target_options.append(f"software_subgroup_width = {SIMD_SIZE}")
    if _profiled_additions(target, entry):
        target_options.append(f'binary32_additive_profile = "{ADDITIVE_PROFILE}"')
    if target_options:
        options += f"\n[project.source_options.metal.target_options.{target}]\n"
        options += "".join(f"{line}\n" for line in target_options)
    if target == "opengl":
        for expression in OPENGL_INDEX_RANGE_ASSERTIONS[kind]:
            options += f"""
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "{expression}"
minimum = 0
maximum = {SIGNED_MAX}
"""
    size = ", ".join(str(item) for item in WORKGROUPS[kind])
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
{options}"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _dispatch_count():
    return sum(len(_cases(entry)) for entry, _, _ in VARIANTS)


def _checked_elements(entry):
    total = 0
    for item, key, _ in VARIANTS:
        if item == entry:
            for name in _cases(entry):
                total += len(_case_data(entry, key, name)["expected"]) - GUARD_COUNT
    return total


def _case_contract():
    unfold = {
        str(spatial): {
            name: {
                **{
                    field: list(value) if isinstance(value, tuple) else value
                    for field, value in case.items()
                },
                "oS": list(_unfold_geometry(spatial, name)["oS"]),
                "grid": _unfold_geometry(spatial, name)["grid"],
            }
            for name, case in cases.items()
        }
        for spatial, cases in UNFOLD_CASES.items()
    }
    dw2 = {
        key: {
            name: {
                "batch": geometry["batch"],
                "iS": list(geometry["iS"]),
                "C": geometry["C"],
                "padLo": list(geometry["padLo"]),
                "padHi": list(geometry["padHi"]),
                "oS": list(geometry["oS"]),
                "grid": geometry["grid"],
            }
            for name in DW2_CASES
            for geometry in (_dw2_geometry(key, name),)
        }
        for key in DW2_VARIANTS
    }
    dw1 = {
        name: {
            key: value
            for key, value in _dw1_geometry(name).items()
            if key not in {"strides"}
        }
        for name in DW1_CASES
    }
    return {"naiveUnfold": unfold, "depthwiseConv2d": dw2, "depthwiseConv1d": dw1}


def _expected_validation_contract():
    return {
        "comparison": "exact words",
        "exactArithmetic": (
            "unfold copies distinct values; depthwise inputs are quarters and "
            "weights halves in [-1, 1], so products and sums are exact"
        ),
        "workgroupSize": {
            "naiveUnfold": list(UNFOLD_WORKGROUP),
            "depthwiseConv2d": list(DW2_WORKGROUP),
            "depthwiseConv1d": list(DW1_WORKGROUP),
        },
        "grid": {
            "naiveUnfold": "[C / 32, prod(wS) / 8, tile_rows] groups, C = 32",
            "depthwiseConv2d": "[C / 8, ceil(oW / 8), ceil(oH / 4) * N] groups",
            "depthwiseConv1d": "[D / 16, Tout / 16, B / 4] groups, B = 4",
        },
        "specializationConstants": dict(CONSTANT_IDS),
        "depthwiseConv2dVariants": {
            key: {
                "ker_h": ker_h,
                "ker_w": ker_w,
                "str_h": str_h,
                "str_w": str_w,
                "do_flip": flip,
            }
            for key, (ker_h, ker_w, str_h, str_w, flip) in DW2_VARIANTS.items()
        },
        "cases": _case_contract(),
        "softwareSubgroupWidth": {
            "targets": list(SOFTWARE_SUBGROUP_TARGETS),
            "width": SIMD_SIZE,
            "kernels": ["depthwise_conv_2d"],
        },
        "binary32AdditiveProfile": {
            "profile": ADDITIVE_PROFILE,
            "targets": list(PROFILED_ADDITION_TARGETS),
            "kernels": ["depthwise_conv_2d", "depthwise_conv_1d"],
            "reason": (
                "software addition keeps the +0.0 that compilers may fold out "
                "of the +0.0-initialized accumulator"
            ),
        },
        "indexRangeAssertions": {
            "opengl": {
                kind: [
                    {"expression": expression, "minimum": 0, "maximum": SIGNED_MAX}
                    for expression in expressions
                ]
                for kind, expressions in OPENGL_INDEX_RANGE_ASSERTIONS.items()
                if expressions
            },
            "metal": {},
            "directx": {},
        },
        "paramsBindings": {
            "record": {str(spatial): size for spatial, size in PARAMS_SIZES.items()},
            "depthwiseConv2dBlock": dict(PARAMS_BLOCK_SIZES),
        },
        "outputGuard": GUARD_VALUE,
        "outputGuardCount": GUARD_COUNT,
        "inputGuard": "nan",
        "inputGuardCount": INPUT_GUARD_COUNT,
        "elementTypes": {
            target: {dtype: _element_type(target, dtype) for dtype in DTYPES}
            for target in TARGETS
        },
        "dispatchCountByTarget": _dispatch_count(),
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": len(ENTRIES) + len(WINOGRAD_ENTRIES),
        "coveredEntryCount": len(ENTRIES),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "uncoveredEntries": list(WINOGRAD_ENTRIES),
        "uncoveredReason": (
            "the Winograd transforms have their own gate, test_current_winograd.py"
        ),
        "artifactCountByTarget": len(VARIANTS),
        "dispatchCountByTarget": _dispatch_count(),
        "checkedElementCountByEntry": {
            entry: _checked_elements(entry) for entry in ENTRIES
        },
        "completeCorpusEntryCount": 17832,
        # conv.cpp picks the int64-index kernels above INT32_MAX elements; the
        # gate runs them on the same small cases as their int32 forms.
        "largeIndexKernelsUseSmallCases": True,
        # Metal clips dispatch_threads' last groups; every case tiles exactly.
        "partialThreadgroupsExercised": False,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_conv_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-conv-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "headers": HEADERS,
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
        for entry, records in artifacts.items():
            assert tuple(records) == tuple(key for key, _ in _variants(entry))
            for record in records.values():
                assert set(record) == {"sha256", "sizeBytes"}
                assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
    # Metal keeps function constants and OpenGL specialization constants, so
    # every depthwise 2-D specialization shares one artifact; DirectX
    # materializes each.
    for entry in DW2_ENTRIES:
        for target in ("metal", "opengl"):
            records = contract["artifacts"][target][entry]
            assert len({record["sha256"] for record in records.values()}) == 1
        records = contract["artifacts"]["directx"][entry]
        assert len({record["sha256"] for record in records.values()}) == len(records)
    for target in TARGETS:
        hashes = [
            record["sha256"]
            for entry in ENTRIES
            if entry not in DW2_ENTRIES
            for record in contract["artifacts"][target][entry].values()
        ]
        assert len(set(hashes)) == len(hashes)
    assert contract["validationContract"] == _expected_validation_contract()
    assert contract["scope"] == _expected_scope()


def test_conv_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "conv.current-tree.translation.json" in readme
    assert len(ENTRIES) == len(set(ENTRIES)) == 27
    # instantiate_naive_unfold_nd_dims: per type, per N, unfold then transpose.
    assert UNFOLD_ENTRIES[:4] == (
        "naive_unfold_nd_float32_1",
        "naive_unfold_transpose_nd_float32_1",
        "naive_unfold_nd_float32_2",
        "naive_unfold_transpose_nd_float32_2",
    )
    assert DW1_ENTRIES[:2] == (
        "depthwise_conv_1d_float32",
        "depthwise_conv_1d_float32_large",
    )
    assert len(WINOGRAD_ENTRIES) == len(set(WINOGRAD_ENTRIES)) == 9
    assert not set(WINOGRAD_ENTRIES) & set(ENTRIES)
    assert {_kind(entry) for entry in ENTRIES} == set(WORKGROUPS)
    assert [_dtype(entry) for entry in DW2_ENTRIES] == list(DTYPES)
    assert len(VARIANTS) == 18 + 3 * len(DW2_VARIANTS) + 6
    assert len({(entry, key) for entry, key, _ in VARIANTS}) == len(VARIANTS)


def test_conv_cases_follow_mlx_dispatch():
    for spatial, cases in UNFOLD_CASES.items():
        for name in cases:
            geometry = _unfold_geometry(spatial, name)
            points = geometry["kernelPoints"]
            assert all(extent >= 1 for extent in geometry["oS"]), name
            assert _mlx_unfold_group(UNFOLD_CHANNELS, points) == UNFOLD_WORKGROUP
            assert points % UNFOLD_WORKGROUP[1] == 0
            assert [
                count * size for count, size in zip(geometry["grid"], UNFOLD_WORKGROUP)
            ] == [UNFOLD_CHANNELS, points, geometry["rows"]]
            # A later row tile of a tiled unfold stays within implicit_M.
            assert geometry["rowOffset"] + geometry["rows"] <= geometry["implicitM"]
        group = list(cases.values())
        assert any(case["flip"] for case in group)
        assert any(max(case["padLo"]) for case in group)
        assert any(max(case["str"]) > 1 for case in group)
        assert any(max(case["kdil"]) > 1 or max(case["idil"]) > 1 for case in group)
        assert any(case.get("rows") for case in group)
        assert any(case["batch"] > 1 for case in group)
    for key, (ker_h, ker_w, str_h, str_w, flip) in DW2_VARIANTS.items():
        constants = _dw2_constants(key)
        for name in DW2_CASES:
            geometry = _dw2_geometry(key, name)
            params = _dw2_params(geometry)
            # depthwise_conv_2D_gpu's conditions in dispatch_conv_2D_gpu.
            assert params["C"] % 16 == 0 and params["C"] == params["O"]
            assert params["groups"] == params["C"]
            assert ker_h <= 7 and ker_w <= 7 and str_h <= 2 and str_w <= 2
            assert params["wt_strides"][1] == params["wS"][1]
            assert params["kdil"] == params["idil"] == [1, 1]
            assert (constants["ker_h"], constants["ker_w"]) == tuple(params["wS"])
            assert (constants["str_h"], constants["str_w"]) == tuple(params["str"])
            assert constants["do_flip"] is params["flip"]
            assert all(extent >= 1 for extent in geometry["oS"])
    oddly = [
        _dw2_geometry(key, name)["oS"] for key in DW2_VARIANTS for name in DW2_CASES
    ]
    # Ragged edges exercise the kernel's early return after the barriers.
    assert any(height % DW2_TILE["th"] for height, _ in oddly)
    assert any(width % DW2_TILE["tw"] for _, width in oddly)
    for name in DW1_CASES:
        geometry = _dw1_geometry(name)
        assert _mlx_block_dims(geometry["D"], geometry["Tout"], geometry["B"]) == (
            DW1_WORKGROUP
        )
        assert [
            count * size for count, size in zip(geometry["grid"], DW1_WORKGROUP)
        ] == [geometry["D"], geometry["Tout"], geometry["B"]]
    assert {DW1_CASES[name][2] for name in DW1_CASES} >= {1, 9}


def test_conv_references_match_kernel_index_arithmetic():
    for spatial in UNFOLD_CASES:
        for transpose in (False, True):
            for name in UNFOLD_CASES[spatial]:
                geometry = _unfold_geometry(spatial, name)
                params = _unfold_params(transpose, geometry)
                simulated = _unfold_simulate(
                    transpose,
                    params,
                    geometry["rowOffset"],
                    [
                        count * size
                        for count, size in zip(geometry["grid"], UNFOLD_WORKGROUP)
                    ],
                )
                assert simulated == list(_unfold_sources(transpose, spatial, name))
    for key in DW2_VARIANTS:
        for name in DW2_CASES:
            geometry = _dw2_geometry(key, name)
            inputs, weights = _dw2_values(key, name)
            assert _dw2_simulate(
                _dw2_constants(key),
                _dw2_params(geometry),
                inputs,
                weights,
                geometry["grid"],
            ) == list(_dw2_reference(key, name))
    for name in DW1_CASES:
        geometry = _dw1_geometry(name)
        inputs, weights = _dw1_values(name)
        assert _dw1_simulate(
            geometry["strides"],
            geometry["K"],
            inputs,
            weights,
            [geometry["D"], geometry["Tout"], geometry["B"]],
        ) == list(_dw1_reference(name))


@pytest.mark.parametrize("dtype", DTYPES)
def test_conv_values_are_exact_and_guards_distinct(dtype):
    for entry, key, _ in VARIANTS:
        if _dtype(entry) != dtype:
            continue
        for name in _cases(entry):
            data = _case_data(entry, key, name)
            live = data["expected"][:-GUARD_COUNT]
            assert data["expected"][-GUARD_COUNT:] == [GUARD_VALUE] * GUARD_COUNT
            assert GUARD_VALUE not in live
            assert all(_quantize(dtype, value) == value for value in live)
            for values in data["arrays"].values():
                assert all(
                    math.isnan(value) or _quantize(dtype, value) == value
                    for value in values
                )
            if _kind(entry).startswith("unfold"):
                inputs = data["arrays"]["in_"][:-INPUT_GUARD_COUNT]
                assert len(set(inputs)) == len(inputs) and 0.0 not in inputs
                # Padding and dilation holes leave zeros in some cases.
            else:
                assert max(abs(value) for value in live) < EXACT_SUM_LIMIT
            assert all(-SIGNED_MAX <= item <= SIGNED_MAX for item in data["grid"])
    assert _quantize(dtype, GUARD_VALUE) == GUARD_VALUE


UNFOLD_MUTATIONS = (
    "flip",
    "padding",
    "stride",
    "kernel-dilation",
    "input-dilation",
    "row-offset",
    "layout",
    "batch",
    "channel",
    "zero-fill",
)
DW2_MUTATIONS = ("flip", "padding", "stride", "kernel-transpose", "channel", "batch")
DW1_MUTATIONS = ("taps", "batch", "weights", "time")


def test_conv_references_reject_defects():
    for spatial in UNFOLD_CASES:
        for transpose in (False, True):
            rejected = set()
            for name in UNFOLD_CASES[spatial]:
                exact = _unfold_sources(transpose, spatial, name)
                for mutation in UNFOLD_MUTATIONS:
                    if _unfold_sources(transpose, spatial, name, mutation) != exact:
                        rejected.add(mutation)
            assert rejected == set(UNFOLD_MUTATIONS), (spatial, transpose)
    rejected = set()
    for key in DW2_VARIANTS:
        for name in DW2_CASES:
            exact = _dw2_reference(key, name)
            rejected |= {
                mutation
                for mutation in DW2_MUTATIONS
                if _dw2_reference(key, name, mutation) != exact
            }
    assert rejected == set(DW2_MUTATIONS)
    rejected = {
        mutation
        for name in DW1_CASES
        for mutation in DW1_MUTATIONS
        if _dw1_reference(name, mutation) != _dw1_reference(name)
    }
    assert rejected == set(DW1_MUTATIONS)


def test_conv_params_follow_the_mlx_struct_layout():
    for spatial, size in PARAMS_SIZES.items():
        name = next(iter(UNFOLD_CASES[spatial]))
        params = _unfold_params(False, _unfold_geometry(spatial, name))
        data = _params_bytes(params)
        assert len(data) == size
        # int64 strides start at the first 8-byte boundary after 3 + 7N ints.
        offset = (4 * (3 + 7 * spatial) + 7) // 8 * 8
        assert struct.unpack_from("<q", data, offset)[0] == params["in_strides"][0]
        assert data[offset + 24 * (spatial + 2) + 4] == params["flip"]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "entry,key,constants", VARIANTS, ids=[f"{e}-{k}" for e, k, _ in VARIANTS]
)
def test_conv_configs_scope_target_options(tmp_path, target, entry, key, constants):
    kind = _kind(entry)
    path = tmp_path / "crosstl.toml"
    path.write_text(_config("out", target, entry, constants), encoding="utf-8")
    config = load_project_config(tmp_path, path)
    assert list(config.targets) == [target]
    assert [item.to_json() for item in config.index_range_assertions] == (
        [
            {
                "source": SOURCE,
                "expression": expression,
                "minimum": 0,
                "maximum": SIGNED_MAX,
            }
            for expression in OPENGL_INDEX_RANGE_ASSERTIONS[kind]
        ]
        if target == "opengl"
        else []
    )
    text = path.read_text(encoding="utf-8")
    size = ", ".join(str(item) for item in WORKGROUPS[kind])
    assert f'"{entry}" = [{size}]' in text
    assert ("software_subgroup_width" in text) is (
        kind == "dw2" and target in SOFTWARE_SUBGROUP_TARGETS
    )
    assert ("binary32_additive_profile" in text) is _profiled_additions(target, entry)
    options = config.source_options["metal"].get("target_options", {})
    assert options.get(target, {}).get("binary32_additive_profile") == (
        ADDITIVE_PROFILE if _profiled_additions(target, entry) else None
    )
    assert ("specialization_constants" in text) is (kind == "dw2")
    for name, value in constants.items():
        assert f'"{CONSTANT_IDS[name]}" = {_toml(value)}' in text


# ---------------------------------------------------------------------------
# Native gate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native conv gate")
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
    work = tmp_path_factory.mktemp("conv-metal-reference")
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


def _translate(root, work, target, entry, key, constants):
    """Translate one specialization and check it against its pinned artifact."""
    record = _contract()["artifacts"][target][entry][key]
    stage = work / f"{entry}-{key}"
    stage.mkdir()
    config = stage / "crosstl.toml"
    config.write_text(
        _config(f"{work.name}/{stage.name}/out", target, entry, constants),
        encoding="utf-8",
    )
    translation = translate_project(
        load_project_config(root, config),
        format_output=False,
        validate=True,
        run_toolchains=False,
    )
    report = translation.to_json()
    translation.write_json(stage / "report.json")
    validation = validate_project_report(stage / "report.json")
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
    assert artifact["provenance"].get("binary32AdditiveProfile") == (
        ADDITIVE_PROFILE if _profiled_additions(target, entry) else None
    )
    (execution,) = artifact["execution"]["entryPoints"]
    assert execution["workgroupSize"] == list(WORKGROUPS[_kind(entry)])
    generated = root / artifact["path"]
    text = generated.read_text(encoding="utf-8")
    assert "unsupported" not in text
    if target == "metal":
        _metal_library(generated, stage / "translated.metallib", root)
    else:
        _validate_generated_artifact(generated, stage, target)
    descriptor, package = _prepare_native_package(translation, stage)
    assert {
        item["id"]: item["value"] for item in descriptor["specializationConstants"]
    } == {CONSTANT_IDS[name]: value for name, value in constants.items()}
    return descriptor, package, record


def _params_value(target, entry, layout, params):
    """Pack MLXConvParams<N> for the binding's reflected physical layout."""
    spatial = len(params["iS"])
    if _kind(entry) == "dw2" and target != "metal":
        assert (
            layout["storageLayout"]
            == {
                "opengl": "std140",
                "directx": "hlsl-constant-buffer",
            }[target]
        )
        assert layout["blockSizeBytes"] == PARAMS_BLOCK_SIZES[target]
        words = pack_uniform_block(layout, params)
    else:
        assert layout["elementType"] == "record", layout
        assert layout["elementStrideBytes"] == PARAMS_SIZES[spatial]
        words = pack_storage_records(layout, [params])
        # Every reflected record reproduces MLX's C struct byte for byte.
        assert struct.pack(f"<{len(words)}I", *words) == _params_bytes(params)
    return {"dtype": "uint32", "shape": [len(words)], "values": words}


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
        if name == "params":
            value = _params_value(target, entry, layout, data["params"])
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
        if name == OUTPUT_BUFFER:
            output = binding["name"]
        else:
            observed.append(binding["name"])
    assert bound == set(BUFFER_ORDER[_kind(entry)]), bound
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output: dict(values[OUTPUT_BUFFER])},
        {
            "workgroupCount": data["grid"],
            "workgroupSize": list(WORKGROUPS[_kind(entry)]),
        },
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, {OUTPUT_BUFFER: output}, len(observed)


def _payloads(entry, data):
    dtype = _dtype(entry)
    raw = {name: _metal_bytes(dtype, items) for name, items in data["arrays"].items()}
    for name, (element, items) in data["scalars"].items():
        code = "q" if element == "int64" else "i"
        raw[name] = struct.pack(f"<{len(items)}{code}", *items)
    if data["params"] is not None:
        raw["params"] = _params_bytes(data["params"])
    return [raw[name] for name in BUFFER_ORDER[_kind(entry)]]


def _upstream(work, reference, entry, label, payloads, grid, constants):
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
        "workgroupSize": list(WORKGROUPS[_kind(entry)]),
        "simdWidth": SIMD_SIZE,
    }
    if constants:
        request["intConstants"] = [
            {"index": CONSTANT_IDS[name], "value": value}
            for name, value in constants.items()
            if type(value) is int
        ]
        request["boolConstants"] = [
            {"index": CONSTANT_IDS[name], "value": value}
            for name, value in constants.items()
            if type(value) is bool
        ]
    path = work / f"{label}-upstream-request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    readback = work / f"{label}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, path, readback], work, f"{label}-upstream")
    )
    output_index = BUFFER_ORDER[_kind(entry)].index(OUTPUT_BUFFER)
    raw = None
    for index, payload in enumerate(payloads):
        data = (readback / f"buffer-{index}.bin").read_bytes()
        if index == output_index:
            raw = data
        else:
            assert data == payload, (label, index)
    return device, raw


@pytest.mark.parametrize(
    "entry,key,constants", VARIANTS, ids=[f"{e}-{k}" for e, k, _ in VARIANTS]
)
def test_current_mlx_conv_native_parity(
    current_mlx, metal_reference, tmp_path, entry, key, constants
):
    root, target = current_mlx
    dtype = _dtype(entry)
    with tempfile.TemporaryDirectory(prefix=".current-conv-", dir=root) as directory:
        work = Path(directory)
        try:
            descriptor, package, record = _translate(
                root, work, target, entry, key, constants
            )
            executor = _executor(target)
            evidence = []
            try:
                for name in _cases(entry):
                    data = _case_data(entry, key, name)
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
                            constants,
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
                        "specialization": key,
                        "constants": constants,
                        "artifactSha256": record["sha256"],
                        "cases": evidence,
                        "comparison": "exact words",
                        "indexRangeAssertions": (
                            list(OPENGL_INDEX_RANGE_ASSERTIONS[_kind(entry)])
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


def test_ci_requires_native_conv_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_step(
        workflow, "Prove current MLX convolution native parity", group="convolution"
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/conv",
        "--junitxml=mlx-current-results/conv.xml",
        "demos/integrations/mlx/tests/kernels/test_current_conv.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_conv.py"
    )
