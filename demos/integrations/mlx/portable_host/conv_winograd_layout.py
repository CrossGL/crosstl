"""Entries, batch step and launch contract of conv.cpp's Winograd route.

``winograd_conv_2D_gpu`` computes ungrouped, unflipped, stride-one 3x3
convolutions whose channels are multiples of 32 as F(6x6, 3x3) Winograd
products. ``winograd_conv_2d_weight_transform`` moves the weights once into
an (8 x 8, C, O) array. The input is zero-padded into a scratch of
``n_step`` batch rows, and each batch tile then runs
``winograd_conv_2d_input_transform`` into an (8 x 8, tiles, C) array, 64
batched steel GEMMs into (8 x 8, tiles, O), and
``winograd_conv_2d_output_transform`` into the tile's output rows.

``winograd_batch_step`` sizes the batch tile from the device's recommended
working set, which the host does not have. Unless
``MLX_CONV_WINOGRAD_WORKING_SET`` names one, the host budgets with the bound
every translated launch has instead: at most MAX_ELEMENTS elements in each
scratch array and at most 65535 batch rows, the Z workgroups of one launch,
as ``conv_unfold_rows`` uses that bound in place of maxBufferLength. A named
working set applies conv.cpp's arithmetic, and the bound still caps it.
conv.cpp's ``MLX_CONV_WINOGRAD_TILE_BATCH`` and its minimum of 32 GEMM rows
per tile apply either way, so a budget that admits no tile falls back to the
next route. ``plan`` checks each transform launch against conv.cpp's
parameters and proves every read and write in bounds.
"""

import math
import os
import re

from demos.integrations.mlx.portable_host import gemv_layout
from demos.integrations.mlx.portable_host.quantization_layout import ITEM_SIZES

ROUTE = "winograd_conv_2D_gpu"
SOURCE = "mlx/backend/metal/kernels/conv.metal"
PREFIX = "winograd_conv_2d_"
PATTERN = re.compile(
    r"winograd_conv_2d_(?P<stage>weight|input|output)_transform_"
    r"(?P<dtype>float32|float16|bfloat16)_(?P<block>bc|bo)32"
)
STAGES = ("weight", "input", "output")
DTYPES = ("float32", "float16", "bfloat16")
# conv.metal's [[buffer(i)]] parameters, in index order.
PARAMETERS = {
    "weight": ("wt_in", "wt_out", "C", "O"),
    "input": ("inp_in", "inp_out", "params"),
    "output": ("out_in", "out_out", "params"),
}
ARRAYS = {stage: names[:2] for stage, names in PARAMETERS.items()}
OUTPUTS = {stage: names[1] for stage, names in PARAMETERS.items()}
SCALARS = ("C", "O")
# winograd_conv_2D_gpu: the weight transform runs bo = 4 filters, one per
# SIMD-group, in (32, bo, 1) groups over (O / bo, 1, 1). The input and output
# transforms run (32, wn, wm) = (32, 2, 2) groups over (tiles_w, tiles_h, N).
SIMD_SIZE = 32
GROUP_FILTERS = 4
WORKGROUPS = {
    "weight": (SIMD_SIZE, GROUP_FILTERS, 1),
    "input": (SIMD_SIZE, 2, 2),
    "output": (SIMD_SIZE, 2, 2),
}
# Every transform moves 32 channels per block.
BLOCK = 32
TILE = 8
OUT_TILE = 6
DOMAIN = TILE * TILE
SIGNED_MAX = gemv_layout.SIGNED_MAX
MAX_ELEMENTS = gemv_layout.MAX_ELEMENTS
MAX_GROUPS = gemv_layout.MAX_GROUPS
WORKING_SET_ENV = "MLX_CONV_WINOGRAD_WORKING_SET"
TILE_BATCH_ENV = "MLX_CONV_WINOGRAD_TILE_BATCH"
# winograd_batch_step's min_rows_per_tile.
MIN_ROWS_PER_TILE = 32
# The translation options of the current-tree Winograd gate
# (test_current_winograd): OpenGL and DirectX lower each SIMD-group product as
# the software multiply-accumulate with fused, ties-to-even binary32 steps,
# which Apple's SIMD-group matrices match.
FRAGMENT_MAPPING = "tile_4x4_row_pair"
FRAGMENT_MAPPING_PROVENANCE = "mlx_conv_winograd_lane_coordinates"
MMA_PROFILE = "fma-rne-gradual"
# Every channel loop holds barriers, so OpenGL and DirectX prove it ends
# without overflow from a bound of at most INT_MAX - 31 on its limit.
LOOP_BOUND = {"minimum": 0, "maximum": SIGNED_MAX - (BLOCK - 1)}
METADATA_PRECONDITIONS = {
    "weight": {"parameter": "C", **LOOP_BOUND},
    "input": {"parameter": "params", "member": "C", **LOOP_BOUND},
    "output": {"parameter": "params", "member": "O", **LOOP_BOUND},
}
# OpenGL indexes 32-bit storage arrays. ``plan`` bounds every array at
# MAX_ELEMENTS elements, which keeps each composed offset in signed range.
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


def is_entry(entry):
    return isinstance(entry, str) and entry.startswith(PREFIX)


def entry(stage, dtype):
    """The pipeline conv.cpp names for ``stage``'s transform."""
    block = "bo" if stage == "output" else "bc"
    return f"{PREFIX}{stage}_transform_{dtype}_{block}{BLOCK}"


def signature(name):
    """The kind, stage and element type of a Winograd transform entry."""
    match = PATTERN.fullmatch(name) if isinstance(name, str) else None
    if match is None or entry(match["stage"], match["dtype"]) != name:
        raise ValueError("Unsupported Winograd convolution entry")
    return {
        "kind": f"winograd_{match['stage']}",
        "stage": match["stage"],
        "transpose": False,
        "dtype": match["dtype"],
        "spatial": 2,
    }


def entries(dtype):
    return [entry(stage, dtype) for stage in STAGES]


def static_route(p):
    """conv.cpp's conditions for Winograd before its working-set check."""
    return (
        p["groups"] == 1
        and not p["flip"]
        and list(p["str"]) == [1, 1]
        and list(p["kdil"]) == [1, 1]
        and list(p["idil"]) == [1, 1]
        and list(p["wS"]) == [3, 3]
        and p["C"] % BLOCK == 0
        and p["O"] % BLOCK == 0
        and p["N"] * p["iS"][0] * p["iS"][1] >= 4096
        and p["C"] + p["O"] >= 256
    )


def padded_size(p):
    """conv.cpp winograd_padded_size: whole 6-row tiles plus the 2-row halo."""
    return tuple(
        OUT_TILE * -(-(p["iS"][i] + 2 * p["pad"][i] - 2) // OUT_TILE) + 2
        for i in range(2)
    )


def tile_grid(p):
    """The (tiles_h, tiles_w) output tiles of one batch row."""
    return tuple(-(-p["oS"][i] // OUT_TILE) for i in range(2))


def _env_int(environ, name):
    """env::get_var's atoi: a leading signed decimal, otherwise 0."""
    match = re.match(r"\s*([+-]?\d+)", environ.get(name, ""))
    return int(match.group(1)) if match else 0


def _host_rows(p, padded, tiles_per_n):
    """The most batch rows each scratch array and launch admits."""
    if DOMAIN * p["C"] * p["O"] > MAX_ELEMENTS:
        return 0
    return min(
        MAX_ELEMENTS // (padded[0] * padded[1] * p["C"]),
        MAX_ELEMENTS // (DOMAIN * tiles_per_n * p["C"]),
        MAX_ELEMENTS // (DOMAIN * tiles_per_n * p["O"]),
        MAX_GROUPS,
    )


def batch_step(p, dtype, environ=None):
    """conv.cpp winograd_batch_step with the host's budget; 0 falls back."""
    environ = os.environ if environ is None else environ
    padded = padded_size(p)
    tiles_h, tiles_w = tile_grid(p)
    tiles_per_n = tiles_h * tiles_w
    max_n = _host_rows(p, padded, tiles_per_n)
    working_set = _env_int(environ, WORKING_SET_ENV)
    if working_set > 0:
        if dtype not in ITEM_SIZES:
            raise ValueError("Winograd working set needs the element type")
        itemsize = ITEM_SIZES[dtype]
        limit = working_set // 4 * 3
        filt_bytes = DOMAIN * p["C"] * p["O"] * itemsize
        io_bytes = itemsize * (
            p["N"] * p["iS"][0] * p["iS"][1] * p["C"]
            + p["N"] * p["oS"][0] * p["oS"][1] * p["O"]
        )
        used = io_bytes + filt_bytes
        budget = limit - used if limit > used else 0
        bytes_per_n = (
            padded[0] * padded[1] * p["C"] * itemsize
            + DOMAIN * tiles_per_n * (p["C"] + p["O"]) * itemsize
        )
        max_n = min(max_n, budget // bytes_per_n)
    safe_n = min(max_n, p["N"])
    forced = _env_int(environ, TILE_BATCH_ENV)
    if forced > 0:
        return min(forced, safe_n)
    if safe_n < p["N"] and safe_n * tiles_per_n < MIN_ROWS_PER_TILE:
        return 0
    return safe_n


def batch_tiles(p, n_step):
    """The batch rows of every step conv.cpp's loop takes."""
    return [min(n_step, p["N"] - start) for start in range(0, p["N"], n_step)]


def transform_params(p):
    """conv_params_updated: the padded input, the strides of its buffers."""
    from demos.integrations.mlx.portable_host.conv_layout import contiguous

    padded = padded_size(p)
    return {
        **p,
        "iS": list(padded),
        "str": [1, 1],
        "pad": [0, 0],
        "kdil": [1, 1],
        "idil": [1, 1],
        "in_strides": contiguous([1, *padded, p["C"]]),
        "groups": 1,
        "flip": False,
    }


def _check_transform_params(p):
    """The fields conv_params_updated holds for every batch tile."""
    from demos.integrations.mlx.portable_host.conv_layout import contiguous

    scalars = ("N", "C", "O", "groups")
    vectors = ("iS", "wS", "oS", "str", "pad", "kdil", "idil")
    if (
        any(type(p[name]) is not int for name in scalars)
        or any(len(p[name]) != 2 for name in vectors)
        or min(p["N"], p["C"], p["O"], *p["oS"]) < 1
        or p["C"] % BLOCK
        or p["O"] % BLOCK
        or p["groups"] != 1
        or p["flip"]
        or list(p["wS"]) != [3, 3]
        or list(p["str"]) != [1, 1]
        or list(p["pad"]) != [0, 0]
        or list(p["kdil"]) != [1, 1]
        or list(p["idil"]) != [1, 1]
    ):
        raise ValueError("Winograd convolution parameters are invalid")
    tiles = tile_grid(p)
    padded = [OUT_TILE * count + 2 for count in tiles]
    if (
        list(p["iS"]) != padded
        or list(p["in_strides"]) != contiguous([1, *padded, p["C"]])
        or list(p["wt_strides"]) != contiguous([p["O"], 3, 3, p["C"]])
        or list(p["out_strides"]) != contiguous([1, *p["oS"], p["O"]])
    ):
        raise ValueError("Winograd strides differ from conv.cpp's buffers")
    return tiles, padded


def _check_launch(stage, execution, grid):
    if (
        not isinstance(execution, dict)
        or set(execution) != {"workgroupCount", "workgroupSize"}
        or list(execution["workgroupSize"]) != list(WORKGROUPS[stage])
        or list(execution["workgroupCount"]) != list(grid)
        or not all(1 <= value <= MAX_GROUPS for value in grid)
    ):
        raise ValueError("Winograd launch differs from conv.cpp's threadgroups")


def _plan_weight(counts, parameters, threads, execution):
    c, o = parameters["C"], parameters["O"]
    if (
        type(c) is not int
        or type(o) is not int
        or c < BLOCK
        or o < BLOCK
        or c % BLOCK
        or o % BLOCK
    ):
        raise ValueError("Winograd weight transform channels are invalid")
    _check_launch("weight", execution, [o // GROUP_FILTERS, 1, 1])
    # Each SIMD-group reads its filter's 3 x 3 x C taps and writes its 64
    # domain words of every channel.
    total = DOMAIN * c * o
    if threads != total or counts["wt_out"] != total:
        raise ValueError("Winograd weight transform output differs from conv.cpp's")
    if counts["wt_in"] < o * 9 * c:
        raise ValueError("Winograd weight transform reads past its weights")
    return {"channels": c, "filters": o}


def _plan_tiles(stage, counts, parameters, threads, execution):
    p = parameters["params"]
    (tiles_h, tiles_w), padded = _check_transform_params(p)
    _check_launch(stage, execution, [tiles_w, tiles_h, p["N"]])
    tiles = p["N"] * tiles_h * tiles_w
    if stage == "input":
        # Each tile reads its 8 x 8 window of the padded rows, which ends
        # inside them, and writes its 64 domain words of every channel.
        domain, rows = "inp_out", padded[0] * padded[1] * p["C"]
        if counts["inp_in"] < p["N"] * rows or counts["inp_in"] % rows:
            raise ValueError("Winograd input transform reads past its rows")
        written = DOMAIN * tiles * p["C"]
    else:
        # Each tile reads its 64 domain words of every filter and writes the
        # output positions its 6 x 6 block holds.
        domain = "out_out"
        if counts["out_in"] != DOMAIN * tiles * p["O"]:
            raise ValueError("Winograd output transform reads differ from conv.cpp's")
        written = p["N"] * p["oS"][0] * p["oS"][1] * p["O"]
    if threads != written or counts[domain] != written:
        raise ValueError("Winograd transform output differs from conv.cpp's")
    return {"tiles": tiles, "batch": p["N"], "channels": p["C"], "filters": p["O"]}


def plan(name, info, counts, parameters, threads, execution):
    """Validate one transform launch against conv.cpp and return its metadata."""
    stage = info["stage"]
    if any(count > MAX_ELEMENTS for count in counts.values()):
        raise ValueError("Winograd arrays exceed their bounds")
    if stage == "weight":
        metadata = _plan_weight(counts, parameters, threads, execution)
    else:
        metadata = _plan_tiles(stage, counts, parameters, threads, execution)
    return {
        "kind": info["kind"],
        "route": ROUTE,
        "stage": stage,
        "spatial": 2,
        "groups": 1,
        **metadata,
        "weightSlack": 0,
    }


def products(p, dtype, n_step):
    """The steel GEMM entries of every batch tile's 64 products."""
    from demos.integrations.mlx.portable_host import gemm_layout

    tiles_h, tiles_w = tile_grid(p)
    return [
        gemm_layout.select(
            dtype, False, False, rows * tiles_h * tiles_w, p["O"], p["C"], 1
        )
        for rows in batch_tiles(p, n_step)
    ]


def launches(p, dtype, n_step):
    """The weight transform, then each tile's input transform, GEMM and output."""
    weight, inputs, outputs = entries(dtype)
    sequence = [weight]
    for product in products(p, dtype, n_step):
        sequence += [inputs, product, outputs]
    return sequence


def scratch_elements(p, n_step):
    """The padded input, domain input and domain output scratch arrays."""
    padded = padded_size(p)
    tiles = math.prod(tile_grid(p)) * n_step
    return {
        "padded": n_step * padded[0] * padded[1] * p["C"],
        "input": DOMAIN * tiles * p["C"],
        "output": DOMAIN * tiles * p["O"],
    }
