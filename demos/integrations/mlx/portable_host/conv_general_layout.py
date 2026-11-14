"""Entries, parameters and launch contract of conv.cpp's general implicit GEMM.

``implicit_gemm_conv_2D_general_gpu`` runs steel_conv_general's
``implicit_gemm_conv_2d_general`` for ungrouped 2-D convolutions that are
input-dilated, or whose channels the specialized loaders cannot take. It
selects 32- or 64-wide tiles with 2 x 2 SIMD groups and specializes the
function constant ``align_C`` (id 200), which holds when the channels fill
whole 16-channel blocks. conv.cpp names the pipeline by the kernel and that
constant (``..._alC_1`` or ``..._alC_0``); the host names each translated
entry the same way, with the constant frozen into it.

Grid ``z`` enumerates the output row and column residues that share filter
taps: ``f_out_jump_h * f_out_jump_w`` of them. ``base_h`` and ``base_w`` give
each residue's first tap and tap count, and the kernel steps
``f_wgt_jump`` taps at a time. Its weight loader skips the rows from ``O``
on and ends each row at ``C``, so no read leaves the bound weights. ``plan``
checks a launch against conv.cpp's routing and parameters before the
translated kernel runs.
"""

import math
import re

from demos.integrations.mlx.portable_host import gemv_layout
from demos.integrations.mlx.tests.kernels import steel_conv_general_model as model

ROUTE = "implicit_gemm_conv_2D_general_gpu"
# conv.cpp's pipeline hash name: the kernel name and align_C as 0 or 1.
PATTERN = re.compile(
    r"(?P<source>implicit_gemm_conv_2d_general_"
    r"(?P<dtype>float32|float16|bfloat16)_bm(?P<bm>\d+)_bn(?P<bn>\d+)"
    r"_bk16_wm(?P<wm>\d+)_wn(?P<wn>\d+))_alC_(?P<align>[01])"
)
PREFIX = "implicit_gemm_conv_2d_general_"
# steel_conv_general.metal's [[buffer(i)]] parameters, in index order.
PARAMETERS = (
    "A",
    "B",
    "C",
    "params",
    "gemm_params",
    "jump_params",
    "base_h",
    "base_w",
)
ARRAYS = ("A", "B", "C")
OUTPUT = "C"
# Conv2DGeneralJumpParams' and Conv2DGeneralBaseInfo's int fields.
JUMP_FIELDS = (
    "f_wgt_jump_h",
    "f_wgt_jump_w",
    "f_out_jump_h",
    "f_out_jump_w",
    "adj_out_h",
    "adj_out_w",
    "adj_out_hw",
    "adj_implicit_m",
)
BASE_FIELDS = ("weight_base", "weight_size")
# The int records the host binds, by name: their fields and whether the
# kernel reads an array of them.
RECORDS = {
    "jump_params": (JUMP_FIELDS, False),
    "base_h": (BASE_FIELDS, True),
    "base_w": (BASE_FIELDS, True),
}
RECORD_BYTES = {name: 4 * len(fields) for name, (fields, _) in RECORDS.items()}
BK = model.BK
WARPS = (2, 2)
SIGNED_MAX = gemv_layout.SIGNED_MAX
MAX_ELEMENTS = gemv_layout.MAX_ELEMENTS
MAX_GROUPS = gemv_layout.MAX_GROUPS


def is_entry(entry):
    return isinstance(entry, str) and entry.startswith(PREFIX)


def signature(entry):
    """The kind, element type and frozen constant of a general entry."""
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    source = model.BY_NAME.get(match["source"]) if match else None
    if (
        source is None
        or (source.wm, source.wn) != WARPS
        or source.bm not in (32, 64)
        or source.bn not in (32, 64)
    ):
        raise ValueError("Unsupported general implicit GEMM convolution entry")
    return {
        "kind": "general",
        "transpose": False,
        "dtype": match["dtype"],
        "spatial": 2,
        "source": match["source"],
        "model": source,
        "constants": {"align_C": match["align"] == "1"},
    }


def entry(p, dtype):
    """The pipeline implicit_gemm_conv_2D_general_gpu selects for ``p``."""
    jump = jump_params(p)
    bm = 64 if jump["adj_implicit_m"] >= 8192 and p["C"] >= 64 else 32
    bn = 64 if bm == 64 and p["O"] >= 64 else 32
    align = int(p["C"] % BK == 0)
    return f"{PREFIX}{dtype}_bm{bm}_bn{bn}_bk16_wm2_wn2_alC_{align}"


def jump_params(p):
    """Conv2DGeneralJumpParams as conv.cpp fills them."""
    wgt = [math.lcm(i, k) // k for i, k in zip(p["idil"], p["kdil"])]
    out = [math.lcm(i, s) // s for i, s in zip(p["idil"], p["str"])]
    adj = [-(-size // step) for size, step in zip(p["oS"], out)]
    return dict(
        f_wgt_jump_h=wgt[0],
        f_wgt_jump_w=wgt[1],
        f_out_jump_h=out[0],
        f_out_jump_w=out[1],
        adj_out_h=adj[0],
        adj_out_w=adj[1],
        adj_out_hw=adj[0] * adj[1],
        adj_implicit_m=p["N"] * adj[0] * adj[1],
    )


def base_info(p, axis):
    """conv.cpp's Conv2DGeneralBaseInfo table along ``axis``.

    Each output residue starts at the first tap that lands on an input pixel
    and steps ``lcm(idil, kdil) / kdil`` taps at a time.
    """
    idil, kdil, taps = p["idil"][axis], p["kdil"][axis], p["wS"][axis]
    residues = math.lcm(idil, p["str"][axis]) // p["str"][axis]
    step = math.lcm(idil, kdil) // kdil
    jump = -kdil if p["flip"] else kdil
    first = (taps - 1) * kdil if p["flip"] else 0
    table = []
    for residue in range(residues):
        position = residue * p["str"][axis] - p["pad"][axis] + first
        base = 0
        # C++ % keeps the dividend's sign and Python's the divisor's; both
        # are zero exactly when idil divides the position.
        while base < taps and position % idil:
            base += 1
            position += jump
        table.append({"weight_base": base, "weight_size": -(-(taps - base) // step)})
    return table


def gemm_params(p, source):
    """ImplicitGemmConv2DParams as implicit_gemm_conv_2D_general_gpu fills them."""
    sign = -1 if p["flip"] else 1
    step_w = p["in_strides"][2] * p["kdil"][1]
    step_h = p["in_strides"][1] * p["kdil"][0]
    taps_h, taps_w = p["wS"]
    implicit_n = p["O"]
    return dict(
        M=p["N"] * p["oS"][0] * p["oS"][1],
        N=implicit_n,
        K=taps_h * taps_w * p["C"],
        gemm_k_iterations=-(-p["C"] // BK),
        inp_jump_w=sign * step_w,
        inp_jump_h=sign * (step_h - (taps_w - 1) * step_w),
        inp_jump_c=BK - sign * (taps_h - 1) * step_h - sign * (taps_w - 1) * step_w,
        tiles_n=-(-implicit_n // source.bn),
        tiles_m=-(-jump_params(p)["adj_implicit_m"] // source.bm),
        swizzle_log=0,
    )


def decode(name, values):
    """One bound int record, or a table of them, as field maps."""
    fields, table = RECORDS[name]
    if len(values) % len(fields) or (not table and len(values) != len(fields)):
        raise ValueError("General convolution records differ in size")
    records = [
        dict(zip(fields, values[start : start + len(fields)]))
        for start in range(0, len(values), len(fields))
    ]
    return records if table else records[0]


def limit(name):
    """The most int32 words the host binds for ``name``."""
    fields, table = RECORDS[name]
    # f_out_jump = lcm(idil, str) / str is at most the input dilation.
    return len(fields) * (MAX_GROUPS if table else 1)


def _check_params(p):
    scalars = ("N", "C", "O", "groups")
    vectors = ("iS", "wS", "oS", "str", "pad", "kdil", "idil")
    if (
        any(type(p[name]) is not int for name in scalars)
        or any(len(p[name]) != 2 for name in vectors)
        or min(p["N"], p["C"], p["O"]) < 1
        or p["groups"] != 1
        or any(min(p[name]) < 1 for name in ("iS", "wS", "oS", "str", "kdil", "idil"))
        or min(p["pad"]) < 0
    ):
        raise ValueError("Convolution parameters are invalid")
    from demos.integrations.mlx.portable_host.conv_layout import contiguous

    in_shape = [p["N"], *p["iS"], p["C"]]
    wt_shape = [p["O"], *p["wS"], p["C"]]
    out_shape = [p["N"], *p["oS"], p["O"]]
    for shape, name in (
        (in_shape, "in_strides"),
        (wt_shape, "wt_strides"),
        (out_shape, "out_strides"),
    ):
        if p[name] != contiguous(shape):
            raise ValueError(
                "Convolution strides differ from conv.cpp's contiguous arrays"
            )
    # The loaders form dilated input coordinates in int.
    for i in range(2):
        reach = p["oS"][i] * p["str"][i] + p["pad"][i] + p["wS"][i] * p["kdil"][i]
        if reach > SIGNED_MAX or p["iS"][i] * p["idil"][i] > SIGNED_MAX:
            raise ValueError("Convolution coordinates exceed signed range")
    for shape in (in_shape, wt_shape, out_shape):
        if math.prod(shape) > MAX_ELEMENTS:
            raise ValueError("Convolution arrays exceed their bounds")
    return in_shape, wt_shape, out_shape


def plan(name, info, counts, parameters, threads, execution):
    """Validate one general launch against conv.cpp and return its metadata."""
    from demos.integrations.mlx.portable_host.conv_layout import route

    p = parameters["params"]
    in_shape, wt_shape, out_shape = _check_params(p)
    if route(p, info["dtype"]) != ROUTE:
        raise ValueError("Convolution general GEMM differs from conv.cpp's routing")
    if entry(p, info["dtype"]) != name:
        raise ValueError("Convolution general GEMM entry differs from conv.cpp's")
    source = info["model"]
    gemm = gemm_params(p, source)
    jump = jump_params(p)
    tables = {"base_h": base_info(p, 0), "base_w": base_info(p, 1)}
    values = [*gemm.values(), *jump.values()] + [
        value for table in tables.values() for row in table for value in row.values()
    ]
    if any(abs(value) > SIGNED_MAX for value in values):
        raise ValueError("Convolution GEMM parameters exceed signed range")
    if (
        parameters["gemm_params"] != gemm
        or parameters["jump_params"] != jump
        or any(parameters[key] != table for key, table in tables.items())
    ):
        raise ValueError("Convolution general GEMM parameters differ from conv.cpp's")
    grid = [
        gemm["tiles_n"],
        gemm["tiles_m"],
        jump["f_out_jump_h"] * jump["f_out_jump_w"],
    ]
    if (
        not isinstance(execution, dict)
        or set(execution) != {"workgroupCount", "workgroupSize"}
        or list(execution["workgroupSize"]) != list(source.workgroup_size)
        or list(execution["workgroupCount"]) != grid
        or not all(1 <= value <= MAX_GROUPS for value in grid)
    ):
        raise ValueError("Convolution launch differs from conv.cpp's threadgroups")
    total = math.prod(out_shape)
    if threads != total or counts["C"] != total:
        raise ValueError("Convolution output differs from conv.cpp's")
    if counts["A"] < math.prod(in_shape) or counts["B"] < math.prod(wt_shape):
        raise ValueError("Convolution general GEMM operand exceeds its buffer")
    return {
        "kind": "general",
        "route": ROUTE,
        "spatial": 2,
        "groups": 1,
        "alignC": info["constants"]["align_C"],
        "gemm": gemm,
        "jump": jump,
        "weightSlack": 0,
    }
