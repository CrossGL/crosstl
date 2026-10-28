"""Exercise MLX Convolution through the translated unfold, depthwise and GEMM kernels.

Every case runs one ``mx.conv_general`` call, which ``Convolution::eval_gpu``
routes as conv.cpp does: explicit GEMM convolutions unfold the input with
``naive_unfold_nd`` or ``naive_unfold_transpose_nd`` in row tiles and multiply
each tile with the steel GEMM or its split-K kernels, fully separable 1-D
convolutions run ``depthwise_conv_1d``, small-kernel depthwise 2-D
convolutions (and the 1-D ones conv.cpp sends there) run one
``depthwise_conv_2d`` launch of conv.cpp's specialization, implicit GEMM
convolutions run one ``implicit_gemm_conv_2d`` or ``implicit_gemm_conv_3d``
launch, padded-channel convolutions zero-pad the input and weight channels
through ``Pad`` and run one ``implicit_gemm_conv_2d`` launch on the padded
parameters, input-dilated and unaligned-channel 2-D convolutions run one
``implicit_gemm_conv_2d_general`` launch with ``align_C`` frozen, 3-D
convolutions of unaligned channels pad them and run one
``implicit_gemm_conv_3d`` launch into a wider intermediate, small-depth 3-D
convolutions run one 2-D convolution per depth tap and add them, stride-two
transposed 3-D convolutions run one steel GEMM and one strided copy per kernel
phase, Winograd convolutions run the weight transform once and the input
transform, 64 batched steel GEMMs and the output transform per batch tile,
and an empty output launches nothing.

Inputs are nonzero quarters and weights nonzero halves, so every product and
sum of the other routes is exact in float32 and rounds once to the output
type. Every output bit is checked against a float64 reference of the
convolution's definition, and float32 outputs against MLX's CPU values as
well. Winograd rounds by design, so its outputs must lie in the interval
``conv_winograd_reference`` certifies from its stages, a few units in the
last place wide, and a batch-tiled run must reproduce the untiled bits, as
conv.cpp keeps every element's reduction order.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from demos.integrations.mlx.portable_host import conv_layout as layout
from demos.integrations.mlx.portable_host import conv_winograd_layout as winograd
from demos.integrations.mlx.portable_host import (
    conv_winograd_reference as winograd_reference,
)
from demos.integrations.mlx.portable_host import gemm_layout, gemm_splitk_layout
from demos.integrations.mlx.portable_host.prepare import verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_quantized_matmul import rounded

TILE_ENV = "MLX_CONV_UNFOLD_TILE_ROWS"
# The environment variables a case may set; run_case clears the others.
CASE_ENV = (winograd.WORKING_SET_ENV, winograd.TILE_BATCH_ENV)


def _working_set(in_shape, wt_shape, pad, rows, itemsize):
    """A MLX_CONV_WINOGRAD_WORKING_SET that leaves room for ``rows`` and a half.

    winograd_batch_step keeps three quarters of the working set, takes the
    input, output and transformed weights from it and divides the rest by
    each batch row's padded input and transformed tiles.
    """
    n, h, w, c = in_shape
    o = wt_shape[0]
    tiles = -(-h // 6) * -(-w // 6)
    padded = (6 * -(-h // 6) + 2) * (6 * -(-w // 6) + 2)
    per_row = itemsize * (padded * c + 64 * tiles * (c + o))
    used = itemsize * (n * h * w * (c + o) + 64 * c * o)
    assert all(side == 1 for side in pad)
    return str((used + (2 * rows + 1) * per_row // 2) * 4 // 3)


# (name, dtype, input shape, weight shape, options, view, tile rows). Options
# are conv_general's stride, padding (low, high), kernel and input dilation,
# groups and flip; "moved" inputs are channels-last views of channels-first
# arrays, which conv.cpp copies before the unfold, and "expanded" inputs gain
# their size-one width axis through expand_dims, whose stride is not the
# dense one.
CASES = (
    # Strided and dilated, uneven padding: a regular steel GEMM per tile.
    (
        "explicit-1d-strided",
        "float32",
        (2, 19, 5),
        (7, 3, 5),
        dict(stride=2, padding=((1,), (2,)), kernel_dilation=2),
        "dense",
        None,
    ),
    # conv_transpose1d's input dilation and flip over tiles of seven rows.
    (
        "explicit-1d-transpose-tiled",
        "float16",
        (3, 9, 4),
        (6, 3, 4),
        dict(padding=((2,), (2,)), input_dilation=2, flip=True),
        "dense",
        7,
    ),
    # A long contraction: the split-K kernels and their accumulation.
    (
        "explicit-1d-splitk",
        "bfloat16",
        (1, 40, 5),
        (7, 31, 5),
        {},
        "dense",
        None,
    ),
    # Unaligned channels and a small output: the explicit 2-D route.
    (
        "explicit-2d-unaligned",
        "float32",
        (1, 9, 10, 5),
        (7, 3, 3, 5),
        dict(padding=((1, 1), (1, 1))),
        "dense",
        None,
    ),
    # A copied channels-last view with input and kernel dilation and flip.
    (
        "explicit-2d-dilated-copied",
        "float16",
        (2, 4, 5, 3),
        (5, 2, 3, 3),
        dict(stride=(1, 2), kernel_dilation=(1, 2), input_dilation=(2, 1), flip=True),
        "moved",
        None,
    ),
    # conv_transpose3d with a three-wide kernel.
    (
        "explicit-3d-transpose",
        "bfloat16",
        (1, 3, 3, 2, 4),
        (3, 3, 3, 3, 4),
        dict(padding=((2, 2, 2), (2, 2, 2)), input_dilation=2, flip=True),
        "dense",
        None,
    ),
    # Five channels per group: the grouped unfold and batched GEMM.
    (
        "group-1d-unaligned",
        "float32",
        (2, 12, 10),
        (6, 3, 5),
        dict(groups=2, padding=((1,), (0,))),
        "dense",
        None,
    ),
    # Grouped input dilation over tiles of three rows.
    (
        "group-1d-transpose-tiled",
        "float16",
        (2, 7, 8),
        (4, 2, 4),
        dict(groups=2, input_dilation=2, flip=True, padding=((1,), (1,))),
        "dense",
        3,
    ),
    # Grouped 2-D input dilation: dispatch_conv_2D_gpu's first grouped route.
    (
        "group-2d-transpose",
        "bfloat16",
        (1, 4, 4, 6),
        (6, 3, 3, 2),
        dict(groups=3, input_dilation=2, flip=True, padding=((2, 2), (2, 2))),
        "dense",
        None,
    ),
    # Seventeen outputs per group: dispatch_conv_2D_gpu's last grouped route.
    (
        "group-2d-unaligned",
        "float32",
        (1, 6, 6, 15),
        (51, 3, 3, 5),
        dict(groups=3, stride=2),
        "dense",
        None,
    ),
    # Fully separable 1-D convolutions.
    ("depthwise-1d", "float32", (2, 20, 8), (8, 5, 1), dict(groups=8), "dense", None),
    (
        "depthwise-1d-pointwise-copied",
        "float16",
        (2, 16, 4),
        (4, 1, 1),
        dict(groups=4),
        "moved",
        None,
    ),
    (
        "depthwise-1d-long",
        "bfloat16",
        (3, 33, 5),
        (5, 7, 1),
        dict(groups=5),
        "dense",
        None,
    ),
    # Sixteen channels and outputs: the small-filter general-channel loader
    # with eight-column tiles.
    (
        "implicit-2d-small-filter",
        "float32",
        (1, 8, 8, 16),
        (16, 3, 3, 16),
        dict(padding=((1, 1), (1, 1))),
        "dense",
        None,
    ),
    # conv1d as conv2d over expand_dims views, as test_conv_1d_with_2d runs it:
    # the size-one width axis keeps expand_dims' stride, which conv.cpp binds
    # as it is and the port binds densely.
    (
        "implicit-2d-expanded-width",
        "float32",
        (2, 10, 1, 16),
        (16, 3, 1, 16),
        dict(padding=((1, 0), (1, 0))),
        "expanded",
        None,
    ),
    # Three channels, strided with uneven padding, from a copied view.
    (
        "implicit-2d-channels-3",
        "float16",
        (2, 9, 7, 3),
        (8, 3, 3, 3),
        dict(stride=2, padding=((1, 0), (1, 2))),
        "moved",
        None,
    ),
    # One channel with kernel dilation and flip.
    (
        "implicit-2d-channel-1-flipped",
        "bfloat16",
        (1, 10, 10, 1),
        (5, 3, 3, 1),
        dict(kernel_dilation=2, flip=True, padding=((2, 2), (2, 2))),
        "dense",
        None,
    ),
    # 48 outputs per group: the second 32-column tile of the last group reads
    # sixteen weight rows past the weights.
    (
        "implicit-2d-grouped-partial",
        "float32",
        (1, 5, 6, 32),
        (96, 3, 3, 16),
        dict(groups=2, padding=((1, 1), (1, 1))),
        "dense",
        None,
    ),
    # conv_1D_gpu's direct implicit GEMM: one-wide 2-D parameters.
    (
        "implicit-1d-strided",
        "float16",
        (2, 30, 16),
        (16, 5, 16),
        dict(stride=2),
        "dense",
        None,
    ),
    # Seventeen taps: the large-filter loader.
    (
        "implicit-1d-large-filter",
        "bfloat16",
        (1, 40, 16),
        (8, 17, 16),
        {},
        "dense",
        None,
    ),
    # 8192 rows of 64 channels: 64 x 64 tiles.
    (
        "implicit-2d-wide-tiles",
        "float32",
        (2, 64, 64, 64),
        (64, 1, 1, 64),
        {},
        "dense",
        None,
    ),
    (
        "implicit-3d",
        "float16",
        (2, 3, 4, 5, 16),
        (16, 2, 3, 3, 16),
        dict(padding=((0, 1, 1), (1, 1, 1))),
        "dense",
        None,
    ),
    # 48 outputs (conv_general groups only 1-D and 2-D convolutions): the
    # second 32-column tile reads sixteen weight rows past the weights.
    (
        "implicit-3d-partial",
        "bfloat16",
        (1, 3, 4, 5, 32),
        (48, 2, 3, 3, 32),
        dict(stride=(1, 2, 1), padding=((1, 1, 1), (0, 1, 1))),
        "dense",
        None,
    ),
    # depthwise_conv_2D_gpu: partial (8, 8, 4) tiles, strides, flips, every
    # kernel extent up to 7, a pointwise kernel and conv_1D_gpu's 2-D view.
    (
        "depthwise-2d-padded",
        "float32",
        (2, 9, 13, 16),
        (16, 3, 3, 1),
        dict(groups=16, padding=((1, 1), (1, 1))),
        "dense",
        None,
    ),
    (
        "depthwise-2d-strided-flip",
        "float16",
        (1, 15, 18, 32),
        (32, 5, 5, 1),
        dict(groups=32, stride=2, padding=((2, 1), (2, 2)), flip=True),
        "dense",
        None,
    ),
    (
        "depthwise-2d-7x7",
        "bfloat16",
        (1, 16, 20, 16),
        (16, 7, 7, 1),
        dict(groups=16, stride=(2, 1), padding=((3, 3), (3, 3)), flip=True),
        "dense",
        None,
    ),
    (
        "depthwise-2d-3x5-copied",
        "float32",
        (1, 6, 11, 16),
        (16, 3, 5, 1),
        dict(groups=16, stride=(1, 2)),
        "moved",
        None,
    ),
    (
        "depthwise-2d-pointwise",
        "float16",
        (3, 4, 5, 48),
        (48, 1, 1, 1),
        dict(groups=48),
        "dense",
        None,
    ),
    (
        "depthwise-2d-from-1d",
        "float32",
        (1, 12, 16),
        (16, 3, 1),
        dict(groups=16, padding=((1,), (1,))),
        "dense",
        None,
    ),
    # Unaligned channels over a large stride-one output: conv.cpp zero-pads
    # the input and weight channels to 16 and runs the implicit GEMM, here
    # also from a copied view and with a 3x3 kernel over 21 channels.
    (
        "pad-in-channels-2d",
        "float16",
        (1, 20, 20, 8),
        (8, 3, 3, 8),
        dict(padding=((1, 1), (1, 1))),
        "dense",
        None,
    ),
    (
        "pad-in-channels-2d-copied",
        "float32",
        (2, 16, 16, 21),
        (32, 3, 3, 21),
        {},
        "moved",
        None,
    ),
    # Unaligned channels and a large output: align_C false, one residue.
    (
        "general-2d-strided-unaligned",
        "float32",
        (1, 40, 40, 8),
        (8, 3, 3, 8),
        dict(stride=2),
        "dense",
        None,
    ),
    # conv_transpose2d's input dilation and flip: four output residues.
    (
        "general-2d-transpose",
        "float16",
        (2, 6, 7, 16),
        (16, 3, 3, 16),
        dict(padding=((2, 2), (2, 2)), input_dilation=2, flip=True),
        "dense",
        None,
    ),
    # Mixed strides and dilations: six residues, some stepping three taps.
    (
        "general-2d-dilated-copied",
        "bfloat16",
        (1, 5, 6, 32),
        (32, 2, 3, 32),
        dict(
            stride=(1, 2),
            padding=((1, 1), (1, 1)),
            kernel_dilation=(2, 1),
            input_dilation=(2, 3),
        ),
        "moved",
        None,
    ),
    # stride_two_conv_transpose_3D_gpu: a stride-two transposed convolution
    # with a two-wide kernel, one steel GEMM and one strided copy per phase.
    (
        "stride-two-transpose-3d",
        "float32",
        (2, 2, 3, 2, 8),
        (12, 2, 2, 2, 8),
        dict(padding=((1, 1, 1), (1, 1, 1)), input_dilation=2, flip=True),
        "dense",
        None,
    ),
    (
        "stride-two-transpose-3d-half",
        "float16",
        (1, 2, 2, 2, 4),
        (4, 2, 2, 2, 4),
        dict(padding=((1, 1, 1), (1, 1, 1)), input_dilation=2, flip=True),
        "dense",
        None,
    ),
    # pad_and_slice_conv_3D_gpu: unaligned input and output channels, then
    # aligned input channels with unaligned outputs, whose input is not padded.
    (
        "pad-and-slice-3d",
        "float32",
        (1, 3, 4, 4, 4),
        (4, 2, 2, 2, 4),
        {},
        "dense",
        None,
    ),
    (
        "pad-and-slice-3d-outputs",
        "float16",
        (2, 3, 4, 4, 16),
        (20, 2, 2, 2, 16),
        dict(padding=((1, 0, 1), (0, 1, 0))),
        "dense",
        None,
    ),
    # small_kd_conv_3D_gpu: one 2-D convolution per depth tap, summed; the
    # flipped case mirrors the depth taps and pads the frames.
    (
        "small-kd-3d",
        "bfloat16",
        (1, 4, 4, 4, 16),
        (16, 2, 2, 2, 16),
        {},
        "dense",
        None,
    ),
    (
        "small-kd-3d-flipped",
        "float32",
        (1, 5, 4, 3, 32),
        (32, 3, 2, 2, 32),
        dict(padding=((0, 1, 0), (0, 1, 1)), flip=True),
        "dense",
        None,
    ),
    # Winograd: stride-one 3x3 convolutions of 32-aligned channels over a
    # large input. 62 x 62 outputs leave partial 6 x 6 tiles in both
    # directions, and one batch tile holds the whole batch.
    (
        "winograd-2d",
        "float32",
        (1, 64, 64, 32),
        (224, 3, 3, 32),
        {},
        "dense",
        None,
    ),
    # A forced tile of two batch rows leaves a short last tile; the untiled
    # run that follows must give the same bits.
    (
        "winograd-2d-tiled",
        "float16",
        (3, 40, 36, 64),
        (192, 3, 3, 64),
        dict(
            padding=((1, 1), (1, 1)),
            env={winograd.TILE_BATCH_ENV: "2"},
            untiled=True,
        ),
        "dense",
        None,
    ),
    # bfloat16 from a channels-first array, padded.
    (
        "winograd-2d-moved",
        "bfloat16",
        (2, 48, 48, 32),
        (224, 3, 3, 32),
        dict(padding=((1, 1), (1, 1))),
        "moved",
        None,
    ),
    # A working set with room for two batch rows makes winograd_batch_step
    # tile the batch of five as 2, 2 and 1.
    (
        "winograd-2d-working-set",
        "float32",
        (5, 30, 30, 32),
        (224, 3, 3, 32),
        dict(
            padding=((1, 1), (1, 1)),
            env={
                winograd.WORKING_SET_ENV: _working_set(
                    (5, 30, 30, 32), (224, 3, 3, 32), (1, 1), 2, 4
                )
            },
        ),
        "dense",
        None,
    ),
    # A working set too small for one batch row: conv.cpp falls back to the
    # implicit GEMM.
    (
        "winograd-2d-fallback",
        "float32",
        (1, 64, 64, 32),
        (224, 3, 3, 32),
        dict(env={winograd.WORKING_SET_ENV: "1"}),
        "dense",
        None,
    ),
    # No output element.
    ("empty-batch", "float32", (0, 9, 5), (7, 3, 5), {}, "dense", None),
)
IMPLICIT_ROUTES = {"implicit_gemm_conv_2D_gpu", "implicit_gemm_conv_3D_gpu"}


def _vector(value, spatial, default):
    if value is None:
        return [default] * spatial
    if isinstance(value, int):
        return [value] * spatial
    return list(value)


def options(case):
    """conv_general's arguments and the Convolution parameters MLX derives."""
    _, _, in_shape, wt_shape, opts, _, _ = case
    spatial = len(in_shape) - 2
    low, high = opts.get("padding", ((0,) * spatial, (0,) * spatial))
    return {
        "stride": _vector(opts.get("stride"), spatial, 1),
        "padLo": list(low),
        "padHi": list(high),
        "kdil": _vector(opts.get("kernel_dilation"), spatial, 1),
        "idil": _vector(opts.get("input_dilation"), spatial, 1),
        "groups": opts.get("groups", 1),
        "flip": opts.get("flip", False),
    }


def out_shape(in_shape, wt_shape, o):
    spatial = []
    for i in range(len(in_shape) - 2):
        dilated = o["idil"][i] * (in_shape[i + 1] - 1) + 1
        span = dilated + o["padLo"][i] + o["padHi"][i]
        kernel = o["kdil"][i] * (wt_shape[i + 1] - 1) + 1
        spatial.append((span - kernel) // o["stride"][i] + 1)
    return [in_shape[0], *spatial, wt_shape[0]]


def conv_params(in_shape, wt_shape, shape, o):
    """The MLXConvParams conv.cpp fills for row-contiguous operands."""
    return {
        "N": in_shape[0],
        "C": in_shape[-1],
        "O": wt_shape[0],
        "iS": list(in_shape[1:-1]),
        "wS": list(wt_shape[1:-1]),
        "oS": list(shape[1:-1]),
        "str": o["stride"],
        "pad": o["padLo"],
        "kdil": o["kdil"],
        "idil": o["idil"],
        "in_strides": layout.contiguous(in_shape),
        "wt_strides": layout.contiguous(wt_shape),
        "out_strides": layout.contiguous(shape),
        "groups": o["groups"],
        "flip": o["flip"],
    }


def _unfold_rows(row_elements, total_rows, forced):
    """conv.inc conv_unfold_rows."""
    rows = min(layout.MAX_ELEMENTS // row_elements, layout.MAX_GROUPS)
    if forced:
        rows = min(rows, forced)
    return min(rows, total_rows)


def _products(dtype, groups, m, n, k):
    """The steel GEMM entries one unfolded tile's product launches."""
    if groups == 1 and gemm_layout.split_k(m, n, k, 1):
        return list(gemm_splitk_layout.select(dtype, False, True, m, n, k))
    return [gemm_layout.select(dtype, False, True, m, n, k, 1)]


def _small_kd_record(record, params, dtype):
    """small_kd_conv_3D_gpu: one 2-D convolution per depth tap, summed.

    Each tap convolves every output frame's input slab, a dense view, with a
    strided view of the tap's weight plane, which conv_2D_gpu copies dense,
    and every tap after the first adds into the running sum.
    """
    od, h, w = params["oS"][0], *params["iS"][1:]
    frames = [od, h, w, params["C"]]
    taps = [params["O"], *params["wS"][1:], params["C"]]
    planar = {
        "N": od,
        "C": params["C"],
        "O": params["O"],
        "iS": [h, w],
        "wS": params["wS"][1:],
        "oS": params["oS"][1:],
        "str": params["str"][1:],
        "pad": params["pad"][1:],
        "kdil": params["kdil"][1:],
        "idil": params["idil"][1:],
        "in_strides": layout.contiguous(frames),
        "wt_strides": layout.contiguous(taps),
        "out_strides": layout.contiguous([od, *params["oS"][1:], params["O"]]),
        "groups": 1,
        "flip": params["flip"],
    }
    route = layout.route(planar)
    if route == layout.general.ROUTE:
        entry = layout.general.entry(planar, dtype)
    elif route == "implicit_gemm_conv_2D_gpu":
        entry = layout.implicit_entry(planar, dtype)
    else:
        raise ValueError(f"Small-depth case routes its frames to {route}")
    depth = params["wS"][0]
    launches = [entry]
    for _ in range(1, depth):
        launches += [entry, f"vv_Add{dtype}"]
    return {
        **record,
        "convEntries": [entry],
        "launches": launches,
        "launchRoute": route,
        "frameParams": planar,
        "copies": record["copies"] + depth,
    }


def _winograd_record(record, params, dtype):
    """winograd_conv_2D_gpu: one weight transform, then per batch tile the
    zero-padded input copy, the input transform, 64 batched steel GEMMs and
    the output transform. The padded scratch is zero-filled once.

    An ``untiled`` case runs again without its forced batch tile.
    """
    runs = [record["env"]]
    if record["untiled"]:
        runs.append(
            {
                key: value
                for key, value in record["env"].items()
                if key != winograd.TILE_BATCH_ENV
            }
        )
    launches, products, rows, copies = [], [], [], 0
    for environ in runs:
        step = winograd.batch_step(params, dtype, environ)
        tiles = winograd.batch_tiles(params, step)
        launches += winograd.launches(params, dtype, step)
        products += winograd.products(params, dtype, step)
        rows.append(tiles)
        # The input copy of a moved view, the zero fill and each tile's copy.
        copies += int(record["view"] == "moved") + 1 + len(tiles)
    return {
        **record,
        "convEntries": winograd.entries(dtype),
        "productEntries": list(dict.fromkeys(products)),
        "launches": launches,
        "batchRows": rows,
        "copies": copies,
    }


def case_record(case):
    name, dtype, in_shape, wt_shape, opts, view, tile = case
    o = options(case)
    shape = out_shape(in_shape, wt_shape, o)
    params = conv_params(in_shape, wt_shape, shape, o)
    env = dict(opts.get("env", {}))
    if set(env) - set(CASE_ENV):
        raise ValueError(f"Convolution case sets an unknown variable: {name}")
    record = {
        "name": name,
        "dtype": dtype,
        "inShape": list(in_shape),
        "wtShape": list(wt_shape),
        "outShape": shape,
        "options": o,
        "view": view,
        "tileRows": tile,
        "env": env,
        "untiled": bool(opts.get("untiled")),
        "params": params,
        "convEntries": [],
        "productEntries": [],
        "launches": [],
        "copies": int(view == "moved"),
    }
    if math.prod(shape) == 0:
        return {**record, "route": "none"}
    route = layout.route(params, dtype, env)
    record["route"] = route
    if record["untiled"] and route != winograd.ROUTE:
        raise ValueError(f"Only Winograd cases rerun untiled: {name}")
    if route == winograd.ROUTE:
        return _winograd_record(record, params, dtype)
    if route == "depthwise_conv_1D_gpu":
        entry = layout.dw1_entry(dtype)
        return {**record, "convEntries": [entry], "launches": [entry]}
    if route == "depthwise_conv_2D_gpu":
        # conv_1D_gpu dispatches 1-D depthwise convolutions with 2-D parameters.
        planar = layout.as_2d(params) if len(in_shape) == 3 else params
        entry = layout.dw2_entry(layout.dw2_constants(planar), dtype)
        return {**record, "convEntries": [entry], "launches": [entry]}
    if route == "pad_in_channels_conv_2D_gpu":
        # Two Pad launches per operand: the zero fill and the copy into it.
        extra = -(-params["C"] // 16) * 16 - params["C"]
        entry = layout.implicit_entry(layout.with_padded_channels(params, extra), dtype)
        return {
            **record,
            "convEntries": [entry],
            "launches": [entry],
            "copies": record["copies"] + 4,
        }
    if route == layout.general.ROUTE:
        entry = layout.general.entry(params, dtype)
        return {**record, "convEntries": [entry], "launches": [entry]}
    if route == "pad_and_slice_conv_3D_gpu":
        # Pad fills and copies each padded operand: the weight always, the
        # input when its channels grow.
        extra_c, extra_o = layout.padded_channels(params)
        padded = layout.with_padded_channels(params, extra_c, extra_o)
        entry = layout.implicit_entry(padded, dtype)
        return {
            **record,
            "convEntries": [entry],
            "launches": [entry],
            "copies": record["copies"] + 2 + 2 * bool(extra_c),
        }
    if route == "small_kd_conv_3D_gpu":
        return _small_kd_record(record, params, dtype)
    if route == "stride_two_conv_transpose_3D_gpu":
        # Eight phases of one [N*D*H*W, C] by [C, O] product and one strided
        # copy into the phase's output coordinates.
        m = params["N"] * math.prod(params["iS"])
        entries = _products(dtype, 1, m, params["O"], params["C"])
        return {
            **record,
            "productEntries": entries,
            "launches": entries * 8,
            "copies": record["copies"] + 8,
        }
    if route in IMPLICIT_ROUTES:
        # conv_1D_gpu dispatches its implicit GEMM with 2-D parameters.
        entry = layout.implicit_entry(
            layout.as_2d(params) if len(in_shape) == 3 else params, dtype
        )
        return {**record, "convEntries": [entry], "launches": [entry]}
    groups, total = params["groups"], math.prod(shape) // params["O"]
    k = math.prod(wt_shape) // params["O"]
    transpose = route == "explicit_gemm_conv_group_ND_gpu"
    unfold = layout.unfold_entry(transpose, dtype, len(in_shape) - 2)
    rows = _unfold_rows(k * groups, total, tile)
    launches, products = [], []
    for offset in range(0, total, rows):
        tile_rows = min(rows, total - offset)
        entries = _products(dtype, groups, tile_rows, params["O"] // groups, k)
        launches += [unfold, *entries]
        products += entries
    return {
        **record,
        "convEntries": [unfold],
        "productEntries": list(dict.fromkeys(products)),
        "launches": launches,
        "tiles": -(-total // rows),
        # The grouped route copies the transposed weights once.
        "copies": record["copies"] + int(transpose),
    }


def _nonzero(rng, shape, scale):
    """Nonzero multiples of ``scale`` in [-1, 1]: no product is a signed zero."""
    steps = round(1 / scale)
    return (
        rng.integers(1, steps + 1, size=shape) * rng.choice((-1, 1), size=shape) * scale
    )


def reference(np, x, w, record):
    """conv_general's definition in float64: correlate the dilated, padded input.

    Each output position sums input taps ``kdil`` apart over its group's
    channels, the kernel reversed with ``flip``.
    """
    o, params = record["options"], record["params"]
    spatial = x.ndim - 2
    groups = o["groups"]
    c_group, o_group = params["C"] // groups, params["O"] // groups
    dims = [
        o["idil"][i] * (x.shape[i + 1] - 1) + 1 + o["padLo"][i] + o["padHi"][i]
        for i in range(spatial)
    ]
    padded = np.zeros((x.shape[0], *dims, x.shape[-1]))
    place = tuple(
        slice(
            o["padLo"][i],
            o["padLo"][i] + o["idil"][i] * (x.shape[i + 1] - 1) + 1,
            o["idil"][i],
        )
        for i in range(spatial)
    )
    padded[(slice(None), *place)] = x
    out = np.zeros(record["outShape"])
    taps = w.shape[1:-1]
    for tap in np.ndindex(*taps):
        source = tuple(
            taps[i] - 1 - tap[i] if o["flip"] else tap[i] for i in range(spatial)
        )
        window = tuple(
            slice(
                source[i] * o["kdil"][i],
                source[i] * o["kdil"][i]
                + (record["outShape"][i + 1] - 1) * o["stride"][i]
                + 1,
                o["stride"][i],
            )
            for i in range(spatial)
        )
        patch = padded[(slice(None), *window)]
        for g in range(groups):
            weights = w[(slice(g * o_group, (g + 1) * o_group), *tap)]
            out[..., g * o_group : (g + 1) * o_group] += (
                patch[..., g * c_group : (g + 1) * c_group] @ weights.T
            )
    return out + 0.0


def case_data(np, record, seed):
    rng = np.random.default_rng(seed)
    x = _nonzero(rng, record["inShape"], 0.25)
    w = _nonzero(rng, record["wtShape"], 0.5)
    if record["route"] == winograd.ROUTE:
        o = record["options"]
        if o["padLo"] != o["padHi"]:
            raise ValueError("Winograd cases pad symmetrically")
        value, uncertainty = winograd_reference.stages(
            np, x, w, o["padLo"], record["dtype"]
        )
        return {"x": x, "w": w, "interval": (value, uncertainty)}
    total = reference(np, x, w, record)
    if total.size and np.abs(total).max() >= 2**19:
        raise ValueError("Convolution case exceeds the exact float32 budget")
    return {"x": x, "w": w, "expected": rounded(total, record["dtype"])}


def _input(mx, np, values, view, dtype):
    if view == "dense":
        array = mx.array(values.astype(np.float32), dtype=dtype)
        return array, array
    if view == "expanded":
        stored = mx.array(np.squeeze(values, axis=-2).astype(np.float32), dtype=dtype)
        return stored, mx.expand_dims(stored, axis=-2)
    # A row-major channels-first array viewed channels-last. mx.array keeps a
    # NumPy view's strides, so the channels-first copy must be contiguous.
    first = np.ascontiguousarray(np.moveaxis(values, -1, 1), dtype=np.float32)
    stored = mx.array(first, dtype=dtype)
    return stored, mx.moveaxis(stored, 1, -1)


def _arguments(record):
    o = record["options"]
    return {
        "stride": o["stride"],
        "padding": (o["padLo"], o["padHi"]),
        "kernel_dilation": o["kdil"],
        "input_dilation": o["idil"],
        "groups": o["groups"],
        "flip": o["flip"],
    }


def _bits(np, value):
    return np.asarray(value.tolist(), dtype=np.float32).tobytes()


def run_case(mx, np, host, record, seed):
    data = case_data(np, record, seed)
    dtype = getattr(mx, record["dtype"])
    stored, x = _input(mx, np, data["x"], record["view"], dtype)
    w = mx.array(data["w"].astype(np.float32), dtype=dtype)
    mx.eval(stored, w)
    before = {"x": _bits(np, stored), "w": _bits(np, w)}
    keywords = _arguments(record)
    names = (TILE_ENV, *CASE_ENV)
    previous = {name: os.environ.pop(name, None) for name in names}
    if record["tileRows"]:
        os.environ[TILE_ENV] = str(record["tileRows"])
    os.environ.update(record["env"])
    start = host.dispatch_count
    try:
        y = mx.conv_general(x, w, **keywords)
        mx.eval(y)
        untiled = None
        if record["untiled"]:
            os.environ.pop(winograd.TILE_BATCH_ENV)
            untiled = mx.conv_general(x, w, **keywords)
            mx.eval(untiled)
    finally:
        for name in names:
            os.environ.pop(name, None)
            if previous[name] is not None:
                os.environ[name] = previous[name]
    count = host.dispatch_count - start
    if list(y.shape) != record["outShape"] or y.dtype != dtype:
        raise AssertionError(f"Convolution output layout differs: {record['name']}")
    actual = np.asarray(y.tolist(), dtype=np.float32).reshape(record["outShape"])
    if record["route"] == winograd.ROUTE:
        if not winograd_reference.contains(np, actual, *data["interval"]):
            raise AssertionError(
                f"Convolution leaves its certified interval: {record['name']}"
            )
        if untiled is not None and (
            np.asarray(untiled.tolist(), dtype=np.float32).tobytes() != actual.tobytes()
        ):
            raise AssertionError(
                f"Convolution batch tiles change its bits: {record['name']}"
            )
    elif actual.view(np.uint32).tobytes() != data["expected"].view(np.uint32).tobytes():
        raise AssertionError(f"Convolution output differs: {record['name']}")
    if count != len(record["launches"]) + record["copies"]:
        raise AssertionError(f"Convolution dispatch count differs: {record['name']}")
    if before != {"x": _bits(np, stored), "w": _bits(np, w)}:
        raise AssertionError(f"Convolution input changed: {record['name']}")
    cpu = False
    # Winograd rounds where the CPU's direct convolution does not.
    if (
        record["dtype"] == "float32"
        and actual.size
        and record["route"] != winograd.ROUTE
    ):
        reference_value = mx.conv_general(x, w, **keywords, stream=mx.cpu)
        values = np.asarray(reference_value.tolist(), dtype=np.float32)
        if (
            values.reshape(-1).view(np.uint32).tobytes()
            != actual.reshape(-1).view(np.uint32).tobytes()
        ):
            raise AssertionError(f"Convolution differs from MLX CPU: {record['name']}")
        cpu = True
    return {
        **record,
        "seed": seed,
        "dispatchStart": start,
        "dispatchCount": count,
        "outputHash": hashlib.sha256(actual.tobytes()).hexdigest(),
        "cpuEqual": cpu,
        "inputUnchanged": True,
    }


def _covered(records):
    routes = {record["route"] for record in records}
    infos = [
        layout.signature(entry) for record in records for entry in record["convEntries"]
    ]
    depthwise = [info for info in infos if info["kind"] == "dw2"]
    winograd_records = [
        record for record in records if record["route"] == winograd.ROUTE
    ]
    return (
        set(layout.PORTED_ROUTES) <= routes
        and {info["dtype"] for info in depthwise} == set(layout.DTYPES)
        and {info["constants"]["do_flip"] for info in depthwise} == {False, True}
        and any(
            record["route"] == "depthwise_conv_2D_gpu" and len(record["inShape"]) == 3
            for record in records
        )
        and "none" in routes
        and {info["dtype"] for info in infos} == set(layout.DTYPES)
        and {info["spatial"] for info in infos if info["kind"] == "unfold"}
        == set(layout.SPATIAL)
        and any(
            entry.startswith("steel_gemm_splitk_")
            for record in records
            for entry in record["productEntries"]
        )
        and any(record.get("tiles", 1) > 1 for record in records)
        and any(record["view"] == "moved" for record in records)
        and any(record["view"] == "expanded" for record in records)
        and all(
            "outputHash" in record or record["route"] == "none" for record in records
        )
        # Winograd runs every dtype, a short last batch tile, a tiled run
        # against its untiled bits, and a working set that falls back.
        and {record["dtype"] for record in winograd_records} == set(layout.DTYPES)
        and any(
            len(set(rows)) > 1
            for record in winograd_records
            for rows in record["batchRows"]
        )
        and any(record["untiled"] for record in winograd_records)
        and any(
            winograd.WORKING_SET_ENV in record["env"]
            and record["route"] != winograd.ROUTE
            for record in records
        )
    )


def validate_trace(trace, records, target):
    if any(event.get("target") != target for event in trace):
        raise RuntimeError("Convolution trace does not match the routed entries")
    for record in records:
        start, count = record["dispatchStart"], record["dispatchCount"]
        events = trace[start : start + count]
        launches = [event for event in events if "copy" not in event.get("entry", "")]
        if [event["entry"] for event in launches] != record["launches"]:
            raise RuntimeError(
                f"Convolution trace differs from its route: {record['name']}"
            )
        if len(events) - len(launches) != record["copies"]:
            raise RuntimeError(f"Convolution copies differ: {record['name']}")
        for event in launches:
            metadata = event.get("convMetadata")
            route = record.get(
                "launchRoute",
                layout.LAUNCH_ROUTES.get(record["route"], record["route"]),
            )
            if metadata is not None and metadata["route"] != route:
                raise RuntimeError(f"Convolution launch differs: {record['name']}")
            grouped = event.get("gemmMetadata", {}).get("groups")
            if grouped is not None and grouped != record["options"]["groups"]:
                raise RuntimeError(f"Convolution groups differ: {record['name']}")
        if record["route"] == winograd.ROUTE:
            # Each tile's transforms run its batch rows, in conv.cpp's order.
            batches = [
                event["convMetadata"]["batch"]
                for event in launches
                if event.get("convMetadata", {}).get("stage") == "input"
            ]
            if batches != [row for rows in record["batchRows"] for row in rows]:
                raise RuntimeError(f"Convolution batch tiles differ: {record['name']}")
    if not _covered(records) or not any(
        event.get("convMetadata", {}).get("weightSlack", 0) > 0 for event in trace
    ):
        raise RuntimeError("Convolution cases do not cover every route")


def verify(root, packages, output):
    import mlx.core as mx
    import numpy as np

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    preparation = verify_prepared(root)
    (output / "adaptation.json").write_text(json.dumps(preparation, indent=2))
    host = HostRuntime(packages, output / "trace.jsonl", mlx_root=root)
    host.install()
    records = []
    try:
        for seed, case in enumerate(CASES):
            record = run_case(mx, np, host, case_record(case), seed)
            records.append(record)
            print(
                json.dumps({key: record[key] for key in ("name", "route", "launches")}),
                flush=True,
            )
    finally:
        (output / "results.json").write_text(
            json.dumps({"target": host.target, "cases": records}, indent=2)
        )
        close = getattr(host.executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
    trace_path = output / "trace.jsonl"
    trace = (
        [json.loads(line) for line in trace_path.read_text().splitlines()]
        if trace_path.exists()
        else []
    )
    validate_trace(trace, records, host.target)
    if verify_prepared(root) != preparation:
        raise RuntimeError("MLX sources changed during convolution verification")
    print(
        json.dumps({"cases": len(records), "dispatches": host.dispatch_count}),
        flush=True,
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.mlx_root, args.packages, args.output_dir)
