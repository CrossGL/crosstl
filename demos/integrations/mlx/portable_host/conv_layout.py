"""Routing, storage and launch contracts for MLX's ported convolution kernels.

``Convolution::eval_gpu`` in ``conv.inc`` routes every convolution as
mlx/backend/metal/conv.cpp does. Its explicit GEMM routes unfold the input
with ``naive_unfold_nd`` (``groups == 1``) or ``naive_unfold_transpose_nd``
and multiply the unfolded rows with the steel GEMM, whose launches
``gemm_layout`` checks; fully separable 1-D convolutions run
``depthwise_conv_1d``; and the implicit GEMM routes run steel_conv's
``implicit_gemm_conv_2d`` and ``implicit_gemm_conv_3d``. ``route`` mirrors
conv.cpp's routing, and ``plan`` checks every launch against it and proves
each read and write in bounds before a translated kernel runs.

The unfold and depthwise kernels read only ``thread_position_in_grid`` and
``threads_per_grid``, so the host runs them as one-thread workgroups over
conv.cpp's exact thread grid: Direct3D and OpenGL cannot clip the partial
workgroups conv.cpp's ``dispatch_threads`` leaves.
``depthwise_conv_1d_*_large`` serves inputs of more than 2^31 - 1 elements,
which every host launch's bound excludes. The implicit GEMM kernels run with
conv.cpp's threadgroups; their weight loaders read whole ``BN``-row tiles,
so the last tile of the last group reads rows past the weights, which only
feed output columns the kernel never stores. The host binds those rows as
poisoned storage after the weights, so the translated read stays inside its
binding.
"""

import ctypes
import math
import re
import struct

from demos.integrations.mlx.portable_host import gemv_layout
from demos.integrations.mlx.portable_host.quantization_layout import ITEM_SIZES
from demos.integrations.mlx.tests.kernels import steel_conv_model as steel

SOURCE = "mlx/backend/metal/kernels/conv.metal"
MAX_ELEMENTS = gemv_layout.MAX_ELEMENTS
SIGNED_MAX = gemv_layout.SIGNED_MAX
MAX_GROUPS = gemv_layout.MAX_GROUPS
DTYPES = ("float32", "float16", "bfloat16")
SPATIAL = (1, 2, 3)
UNFOLD_PATTERN = re.compile(
    r"naive_unfold(?P<transpose>_transpose)?_nd_"
    r"(?P<dtype>float32|float16|bfloat16)_(?P<spatial>[123])"
)
DW1_PATTERN = re.compile(r"depthwise_conv_1d_(?P<dtype>float32|float16|bfloat16)")
ENTRY_PREFIXES = (
    "naive_unfold_",
    "depthwise_conv_1d_",
    "implicit_gemm_conv_2d_",
    "implicit_gemm_conv_3d_",
)
WORKGROUP = (1, 1, 1)
# conv.metal's [[buffer(i)]] parameters, in index order.
PARAMETERS = {
    "unfold": ("in", "out", "params", "row_offset"),
    "dw1": ("in", "w", "out", "strides", "kernel_size"),
    # steel_conv's A, B, C, params and gemm_params.
    "implicit": ("A", "B", "C", "params", "gemm_params"),
}
OUTPUTS = {"unfold": "out", "dw1": "out", "implicit": "C"}
ARRAYS = {
    "unfold": ("in", "out"),
    "dw1": ("in", "w", "out"),
    "implicit": ("A", "B", "C"),
}
SCALAR_TYPES = {
    "row_offset": "int32",
    "strides": "int32",
    "kernel_size": "int32",
    "gemm_params": "int32",
}
# ImplicitGemmConv{2,3}DParams' int fields, in declaration order.
GEMM_FIELDS = {
    2: (
        "M",
        "N",
        "K",
        "gemm_k_iterations",
        "inp_jump_w",
        "inp_jump_h",
        "inp_jump_c",
        "tiles_n",
        "tiles_m",
        "swizzle_log",
    ),
    3: (
        "M",
        "N",
        "K",
        "gemm_k_iterations",
        "inp_jump_w",
        "inp_jump_h",
        "inp_jump_d",
        "inp_jump_c",
        "tiles_n",
        "tiles_m",
        "swizzle_log",
    ),
}
# sizeof(MLXConvParams<N>) in Metal's natural C layout.
PARAMS_SIZES = {1: 120, 2: 176, 3: 224}
# test_current_conv's OpenGL index range assertions: host preconditions that
# keep the composed element offsets in signed int range. ``plan`` bounds every
# array below MAX_ELEMENTS elements, which implies each of them.
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
    "dw1": (),
}
# The routes of conv.cpp's eval_gpu the host runs.
PORTED_ROUTES = (
    "explicit_gemm_conv_ND_gpu",
    "explicit_gemm_conv_group_ND_gpu",
    "depthwise_conv_1D_gpu",
    "implicit_gemm_conv_2D_gpu",
    "implicit_gemm_conv_3D_gpu",
)


def signature(entry):
    """The kind, element type and spatial rank of a ported conv.metal entry."""
    if isinstance(entry, str):
        match = UNFOLD_PATTERN.fullmatch(entry)
        if match:
            return {
                "kind": "unfold",
                "transpose": match["transpose"] is not None,
                "dtype": match["dtype"],
                "spatial": int(match["spatial"]),
            }
        match = DW1_PATTERN.fullmatch(entry)
        if match:
            return {"kind": "dw1", "transpose": False, "dtype": match["dtype"]}
        model = steel.BY_NAME.get(entry)
        if model is not None:
            return {
                "kind": "implicit",
                "transpose": False,
                "dtype": model.dtype,
                "spatial": model.rank,
                "model": model,
            }
    raise ValueError("Unsupported convolution entry")


def source(entry):
    """The pinned source that instantiates ``entry``."""
    info = signature(entry)
    return info["model"].source if info["kind"] == "implicit" else SOURCE


def workgroup_size(entry):
    """The workgroup size the host launches ``entry`` with."""
    info = signature(entry)
    return info["model"].workgroup_size if info["kind"] == "implicit" else WORKGROUP


def output(entry):
    return OUTPUTS[signature(entry)["kind"]]


def assertion_kind(entry):
    info = signature(entry)
    if info["kind"] == "dw1":
        return "dw1"
    return "unfold_transpose" if info["transpose"] else "unfold"


def host_parameters(entry):
    return PARAMETERS[signature(entry)["kind"]]


def parameter_type(entry, name):
    if name in ARRAYS[signature(entry)["kind"]]:
        return signature(entry)["dtype"]
    if name == "params":
        return "uint8"
    return SCALAR_TYPES[name]


def unfold_entry(transpose, dtype, spatial):
    kernel = "naive_unfold_transpose_nd" if transpose else "naive_unfold_nd"
    return f"{kernel}_{dtype}_{spatial}"


def dw1_entry(dtype):
    return f"depthwise_conv_1d_{dtype}"


def contiguous(shape):
    strides, step = [], 1
    for extent in reversed(shape):
        strides.append(step)
        step *= extent
    return list(reversed(strides))


def decode_params(data, spatial):
    """``MLXConvParams<spatial>`` from its natural C layout."""
    if len(data) != PARAMS_SIZES[spatial]:
        raise ValueError("Convolution parameters do not match MLXConvParams")
    head = struct.Struct(f"<{3 + 7 * spatial}i")
    offset = head.size + (-head.size % 8)
    values = head.unpack_from(data, 0)
    count = spatial + 2
    strides = struct.unpack_from(f"<{3 * count}q", data, offset)
    groups, flip = struct.unpack_from("<iB", data, offset + 24 * count)
    if flip not in (0, 1):
        raise ValueError("Convolution flip is not a boolean")
    fields = {"N": values[0], "C": values[1], "O": values[2]}
    for index, name in enumerate(("iS", "wS", "oS", "str", "pad", "kdil", "idil")):
        start = 3 + index * spatial
        fields[name] = list(values[start : start + spatial])
    for index, name in enumerate(("in_strides", "wt_strides", "out_strides")):
        fields[name] = list(strides[index * count : (index + 1) * count])
    fields["groups"] = groups
    fields["flip"] = bool(flip)
    return fields


def _route_1d(p):
    """conv_1D_gpu: the separable fast path, the 2-D dispatch or explicit GEMM."""
    idil_one = p["idil"][0] == 1
    c, o, groups = p["C"], p["O"], p["groups"]
    if (
        idil_one
        and groups == c == o
        and p["str"][0] == 1
        and p["kdil"][0] == 1
        and p["pad"][0] == 0
        and not p["flip"]
    ):
        return "depthwise_conv_1D_gpu"
    c_group, o_group = c // groups, o // groups
    if (
        idil_one
        and (c_group <= 4 or c_group % 16 == 0)
        and (o_group <= 16 or o_group % 16 == 0)
    ):
        return _route_2d(as_2d(p))
    return (
        "explicit_gemm_conv_group_ND_gpu" if groups > 1 else "explicit_gemm_conv_ND_gpu"
    )


def as_2d(p):
    """conv_1D_gpu's MLXConvParams<2> for the convolutions it dispatches in 2-D."""
    q = dict(p)
    for name in ("iS", "wS", "oS", "str", "kdil", "idil"):
        q[name] = [p[name][0], 1]
    q["pad"] = [p["pad"][0], 0]
    for name in ("in_strides", "wt_strides", "out_strides"):
        q[name] = [p[name][0], p[name][1], 0, p[name][2]]
    return q


def _route_2d(p):
    """dispatch_conv_2D_gpu's route.

    The host has no working-set size for winograd_batch_step, so it names
    winograd whenever conv.cpp's static conditions hold.
    """
    stride_one = all(value == 1 for value in p["str"])
    kdil_one = all(value == 1 for value in p["kdil"])
    idil_one = all(value == 1 for value in p["idil"])
    c, o, groups = p["C"], p["O"], p["groups"]
    if groups > 1:
        c_group, o_group = c // groups, o // groups
        if not idil_one:
            return "explicit_gemm_conv_group_ND_gpu"
        if (
            c_group == 1
            and o_group == 1
            and kdil_one
            and max(p["wS"]) <= 7
            and max(p["str"]) <= 2
            and p["wt_strides"][1] == p["wS"][1]
            and c % 16 == 0
            and c == o
        ):
            return "depthwise_conv_2D_gpu"
        if (c_group <= 4 or c_group % 16 == 0) and (o_group <= 16 or o_group % 16 == 0):
            return "implicit_gemm_conv_2D_gpu"
        return "explicit_gemm_conv_group_ND_gpu"
    n = p["N"]
    inp_large = n * p["iS"][0] * p["iS"][1] >= 4096
    channels_large = c + o >= 256
    out_large = n * p["oS"][0] * p["oS"][1] >= 256
    if (
        not p["flip"]
        and stride_one
        and kdil_one
        and idil_one
        and p["wS"] == [3, 3]
        and c % 32 == 0
        and o % 32 == 0
        and inp_large
        and channels_large
    ):
        return "winograd_conv_2D_gpu"
    specialized = (c <= 4 or c % 16 == 0) and (o <= 16 or o % 16 == 0)
    if (
        idil_one
        and stride_one
        and out_large
        and not specialized
        and (o <= 16 or o % 16 == 0)
        and p["wS"][0] * p["wS"][1] >= 9
    ):
        return "pad_in_channels_conv_2D_gpu"
    if idil_one and specialized:
        return "implicit_gemm_conv_2D_gpu"
    if (c % 16 == 0 and o % 16 == 0) or out_large:
        return "implicit_gemm_conv_2D_general_gpu"
    return "explicit_gemm_conv_ND_gpu"


def _route_3d(p):
    """dispatch_conv_3D_gpu's route."""
    idil_one = all(value == 1 for value in p["idil"])
    groups = p["groups"]
    c_group, o_group = p["C"] // groups, p["O"] // groups
    mod16 = c_group % 16 == 0 and (o_group <= 16 or o_group % 16 == 0)
    if idil_one and not mod16 and groups == 1:
        return "pad_and_slice_conv_3D_gpu"
    if (
        groups == 1
        and p["flip"]
        and p["str"] == [1, 1, 1]
        and p["idil"] == [2, 2, 2]
        and p["kdil"] == [1, 1, 1]
        and p["wS"] == [2, 2, 2]
        and p["pad"] == [1, 1, 1]
        and p["oS"] == [2 * extent for extent in p["iS"]]
    ):
        return "stride_two_conv_transpose_3D_gpu"
    if (
        idil_one
        and mod16
        and groups == 1
        and p["N"] == 1
        and p["wS"][0] <= 7
        and p["str"][0] == 1
        and p["kdil"][0] == 1
        and p["pad"][0] == 0
    ):
        return "small_kd_conv_3D_gpu"
    if idil_one and mod16:
        return "implicit_gemm_conv_3D_gpu"
    return (
        "explicit_gemm_conv_group_ND_gpu" if groups > 1 else "explicit_gemm_conv_ND_gpu"
    )


def route(p):
    """The conv.cpp route ``MLXConvParams`` field map ``p`` takes."""
    return {1: _route_1d, 2: _route_2d, 3: _route_3d}[len(p["iS"])](p)


def read_storage(entry, buffers):
    """Check native buffer identity and read the scalar parameters.

    Returns ``(counts, parameters)``: element counts of the arrays and the
    values of the other parameters, ``params`` as an MLXConvParams field map.
    """
    info = signature(entry)
    names = PARAMETERS[info["kind"]]
    out_name = OUTPUTS[info["kind"]]
    if set(buffers) != set(names):
        raise ValueError("Convolution buffers do not match the entry")
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    counts, parameters, spans = {}, {}, {}
    for name in names:
        buffer, dtype = buffers[name], parameter_type(entry, name)
        itemsize = 1 if dtype == "uint8" else ITEM_SIZES.get(dtype, 4)
        limit = {
            "params": PARAMS_SIZES.get(info.get("spatial")),
            "gemm_params": len(GEMM_FIELDS.get(info.get("spatial"), ())),
            "row_offset": 1,
            "strides": 3,
            "kernel_size": 1,
        }.get(name, MAX_ELEMENTS)
        count = buffer.count
        if (
            not buffer.data
            or buffer.data % itemsize
            or buffer.dtype != dtype.encode("ascii")
            or buffer.output != int(name == out_name)
            or type(count) is not int
            or not 1 <= count <= limit
            or buffer.data + count * itemsize > address_limit
        ):
            raise ValueError("Convolution buffer storage or direction does not match")
        spans[name] = (buffer.data, buffer.data + count * itemsize)
        if name in ARRAYS[info["kind"]]:
            counts[name] = count
        elif name == "params":
            parameters[name] = decode_params(
                ctypes.string_at(buffer.data, count), info["spatial"]
            )
        else:
            values = list((ctypes.c_int32 * count).from_address(buffer.data))
            if name == "gemm_params":
                if count != limit:
                    raise ValueError("Convolution GEMM parameters differ in size")
                parameters[name] = dict(zip(GEMM_FIELDS[info["spatial"]], values))
            else:
                parameters[name] = values if name == "strides" else values[0]
    if "strides" in parameters and len(parameters["strides"]) != 3:
        raise ValueError("depthwise_conv_1d reads three input strides")
    low, high = spans[out_name]
    if any(
        other != out_name and max(low, start) < min(high, end)
        for other, (start, end) in spans.items()
    ):
        raise ValueError("Convolution output overlaps another buffer")
    return counts, parameters


def _launch(execution, count):
    """The one-thread workgroup launch over ``count`` threads per axis."""
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError("Convolution launch geometry is missing")
    if (
        list(execution["workgroupSize"]) != list(WORKGROUP)
        or list(execution["workgroupCount"]) != list(count)
        or not all(1 <= value <= MAX_GROUPS for value in count)
    ):
        raise ValueError("Convolution launch does not cover conv.cpp's thread grid")


def _check_unfold_params(info, p):
    spatial = info["spatial"]
    scalars = ("N", "C", "O", "groups")
    vectors = ("iS", "wS", "oS", "str", "pad", "kdil", "idil")
    if (
        any(type(p[name]) is not int for name in scalars)
        or any(len(p[name]) != spatial for name in vectors)
        or min(p["N"], p["C"], p["O"], p["groups"]) < 1
        or p["C"] % p["groups"]
        or p["O"] % p["groups"]
        or (p["groups"] > 1) != info["transpose"]
        or any(min(p[name]) < 1 for name in ("iS", "wS", "oS", "str", "kdil", "idil"))
    ):
        raise ValueError("Convolution parameters are invalid")
    in_shape = [p["N"], *p["iS"], p["C"]]
    wt_shape = [p["O"], *p["wS"], p["C"] // p["groups"]]
    out_shape = [p["N"], *p["oS"], p["O"]]
    if (
        p["in_strides"] != contiguous(in_shape)
        or p["wt_strides"] != contiguous(wt_shape)
        or p["out_strides"] != contiguous(out_shape)
    ):
        raise ValueError("Convolution strides differ from conv.cpp's contiguous arrays")
    # naive_unfold forms these in int; keep them in signed range.
    for i in range(spatial):
        reach = p["oS"][i] * p["str"][i] + abs(p["pad"][i]) + p["wS"][i] * p["kdil"][i]
        if reach > SIGNED_MAX or p["idil"][i] * p["iS"][i] > SIGNED_MAX:
            raise ValueError("Convolution coordinates exceed signed range")
    for shape in (in_shape, wt_shape, out_shape):
        if math.prod(shape) > MAX_ELEMENTS:
            raise ValueError("Convolution arrays exceed their bounds")
    return in_shape


def _plan_unfold(info, counts, parameters, threads, execution):
    p = parameters["params"]
    in_shape = _check_unfold_params(info, p)
    expected = (
        "explicit_gemm_conv_group_ND_gpu"
        if info["transpose"]
        else ("explicit_gemm_conv_ND_gpu")
    )
    if route(p) != expected:
        raise ValueError("Convolution unfold differs from conv.cpp's routing")
    points = math.prod(p["wS"])
    rows_total = p["N"] * math.prod(p["oS"])
    row_offset = parameters["row_offset"]
    rows = execution["workgroupCount"][2] if isinstance(execution, dict) else 0
    _launch(execution, (p["C"], points, rows))
    filter_size = p["C"] * points
    if (
        not 0 <= row_offset < rows_total
        or row_offset + rows > rows_total
        or threads != rows * filter_size
        or counts["out"] != threads
    ):
        raise ValueError("Convolution unfold tile differs from conv.cpp's")
    # The largest valid read is the last input element; invalid taps write 0.
    if counts["in"] < math.prod(in_shape):
        raise ValueError("Convolution unfold input exceeds its buffer")
    return {
        "kind": "unfold",
        "route": expected,
        "spatial": info["spatial"],
        "groups": p["groups"],
        "rowOffset": row_offset,
        "rows": rows,
        "implicitM": rows_total,
        "filterSize": filter_size,
    }


def _plan_dw1(counts, parameters, threads, execution):
    strides, taps = parameters["strides"], parameters["kernel_size"]
    count = execution.get("workgroupCount") if isinstance(execution, dict) else None
    if not isinstance(count, list) or len(count) != 3:
        raise ValueError("Convolution launch geometry is missing")
    channels, steps, batches = count
    _launch(execution, (channels, steps, batches))
    if strides[2] != 1 or strides[1] != channels or strides[0] % channels:
        raise ValueError("depthwise_conv_1d input is not row contiguous")
    length = strides[0] // channels
    if (
        taps < 1
        or length != steps + taps - 1
        or threads != channels * steps * batches
        or counts["out"] != threads
    ):
        raise ValueError("depthwise_conv_1d launch differs from conv.cpp's")
    # Thread (x, y, z) reads in[z * s0 + (y + i) * s1 + x] and w[x * K + i].
    if counts["in"] < batches * length * channels or counts["w"] < channels * taps:
        raise ValueError("depthwise_conv_1d operand exceeds its buffer")
    return {
        "kind": "dw1",
        "route": "depthwise_conv_1D_gpu",
        "batches": batches,
        "length": length,
        "channels": channels,
        "taps": taps,
    }


def implicit_entry(p, dtype):
    """The steel_conv entry implicit_gemm_conv_{2,3}D_gpu selects for ``p``."""
    rank = len(p["iS"])
    c_group, o_group = p["C"] // p["groups"], p["O"] // p["groups"]
    implicit_m = p["N"] * math.prod(p["oS"])
    wm, wn = 2, 2
    bm = 64 if implicit_m >= 8192 and c_group >= 64 else 32
    bn = 64 if bm == 64 or o_group >= 64 else 32
    if o_group <= 16:
        bn, wm, wn = 8, 4, 1
    channels = c_group if rank == 2 and c_group <= 4 else 0
    small = not channels and max(p["wS"]) <= steel.SMALL_FILTER_TAPS
    channel = "" if rank == 3 else f"_channel_{channels or 'l'}"
    return (
        f"implicit_gemm_conv_{rank}d_{dtype}_bm{bm}_bn{bn}_bk16_wm{wm}_wn{wn}"
        f"{channel}_filter_{'s' if small else 'l'}"
    )


def implicit_gemm_params(p, model):
    """ImplicitGemmConv{2,3}DParams as implicit_gemm_conv_{2,3}D_gpu fills them.

    Each jump moves the input pointer from the last tap along one axis to the
    first tap along the next, and ``inp_jump_c`` to the next channel block.
    """
    rank = model.rank
    c_group = p["C"] // p["groups"]
    taps = math.prod(p["wS"])
    implicit_m = p["N"] * math.prod(p["oS"])
    implicit_n = p["O"] // p["groups"]
    implicit_k = taps * c_group
    bk = model.bk
    k_iterations = taps * -(-c_group // bk)
    if rank == 2 and c_group <= 2:
        k_iterations = -(-implicit_k // bk)
    elif rank == 2 and c_group <= 4:
        k_iterations = -(-(taps * 4) // bk)
    sign = -1 if p["flip"] else 1
    # The input step of one tap along each spatial axis, outermost first.
    steps = [p["in_strides"][axis + 1] * p["kdil"][axis] for axis in range(rank)]
    jumps = {}
    for depth, name in enumerate(("inp_jump_w", "inp_jump_h", "inp_jump_d")[:rank]):
        axis = rank - 1 - depth
        inner = sum((p["wS"][a] - 1) * steps[a] for a in range(axis + 1, rank))
        jumps[name] = sign * (steps[axis] - inner)
    jumps["inp_jump_c"] = bk - sign * sum(
        (p["wS"][a] - 1) * steps[a] for a in range(rank)
    )
    return dict(
        M=implicit_m,
        N=implicit_n,
        K=implicit_k,
        gemm_k_iterations=k_iterations,
        **jumps,
        tiles_n=-(-implicit_n // model.bn),
        tiles_m=-(-implicit_m // model.bm),
        swizzle_log=0,
    )


def _implicit_strides(p, shape, name):
    """conv.cpp's contiguous strides, or conv_1D_gpu's with a zero width step."""
    strides = contiguous(shape)
    if p[name] == strides:
        return True
    rank = len(p["iS"])
    unit = rank == 2 and p["iS"][1] == p["wS"][1] == p["oS"][1] == 1
    return unit and p[name] == [*strides[:2], 0, strides[3]]


def _check_implicit_params(info, p):
    rank = info["spatial"]
    scalars = ("N", "C", "O", "groups")
    vectors = ("iS", "wS", "oS", "str", "pad", "kdil", "idil")
    if (
        any(type(p[name]) is not int for name in scalars)
        or any(len(p[name]) != rank for name in vectors)
        or min(p["N"], p["C"], p["O"], p["groups"]) < 1
        or p["C"] % p["groups"]
        or p["O"] % p["groups"]
        or any(min(p[name]) < 1 for name in ("iS", "wS", "oS", "str", "kdil"))
        or p["idil"] != [1] * rank
        or min(p["pad"]) < 0
    ):
        raise ValueError("Convolution parameters are invalid")
    in_shape = [p["N"], *p["iS"], p["C"]]
    wt_shape = [p["O"], *p["wS"], p["C"] // p["groups"]]
    out_shape = [p["N"], *p["oS"], p["O"]]
    if not all(
        _implicit_strides(p, shape, name)
        for shape, name in (
            (in_shape, "in_strides"),
            (wt_shape, "wt_strides"),
            (out_shape, "out_strides"),
        )
    ):
        raise ValueError("Convolution strides differ from conv.cpp's contiguous arrays")
    # The loaders form input coordinates in int.
    for i in range(rank):
        reach = p["oS"][i] * p["str"][i] + p["pad"][i] + p["wS"][i] * p["kdil"][i]
        if reach > SIGNED_MAX or p["iS"][i] > SIGNED_MAX:
            raise ValueError("Convolution coordinates exceed signed range")
    for shape in (in_shape, wt_shape, out_shape):
        if math.prod(shape) > MAX_ELEMENTS:
            raise ValueError("Convolution arrays exceed their bounds")
    return in_shape, wt_shape, out_shape


def _plan_implicit(entry, info, counts, parameters, threads, execution):
    p, model = parameters["params"], info["model"]
    in_shape, wt_shape, out_shape = _check_implicit_params(info, p)
    expected = f"implicit_gemm_conv_{model.rank}D_gpu"
    routed = _route_2d(p) if model.rank == 2 else _route_3d(p)
    if routed != expected:
        raise ValueError("Convolution implicit GEMM differs from conv.cpp's routing")
    if implicit_entry(p, info["dtype"]) != entry:
        raise ValueError("Convolution implicit GEMM entry differs from conv.cpp's")
    c_group = p["C"] // p["groups"]
    if (c_group != model.channels) if model.channels else (c_group % model.bk):
        raise ValueError("Convolution channels differ from the entry's loaders")
    gemm = implicit_gemm_params(p, model)
    if any(abs(value) > SIGNED_MAX for value in gemm.values()):
        raise ValueError("Convolution GEMM parameters exceed signed range")
    if parameters["gemm_params"] != gemm:
        raise ValueError("Convolution GEMM parameters differ from conv.cpp's")
    grid = [gemm["tiles_n"], gemm["tiles_m"], p["groups"]]
    if (
        not isinstance(execution, dict)
        or set(execution) != {"workgroupCount", "workgroupSize"}
        or list(execution["workgroupSize"]) != list(model.workgroup_size)
        or list(execution["workgroupCount"]) != grid
        or not all(1 <= value <= MAX_GROUPS for value in grid)
    ):
        raise ValueError("Convolution launch differs from conv.cpp's threadgroups")
    total = math.prod(out_shape)
    if threads != total or counts["C"] != total:
        raise ValueError("Convolution output differs from conv.cpp's")
    # Every input read is bounds checked. The weight loaders read K elements
    # of each row of their group's BN-row tiles; eight-row tiles skip the rows
    # from O on, counted from the group's first row.
    weights = math.prod(wt_shape)
    rows = gemm["tiles_n"] * model.bn
    if model.bn == 8:
        rows = min(rows, p["O"])
    reach = ((p["groups"] - 1) * gemm["N"] + rows) * gemm["K"]
    if (
        counts["A"] < math.prod(in_shape)
        or counts["B"] < weights
        or reach > MAX_ELEMENTS
    ):
        raise ValueError("Convolution implicit GEMM operand exceeds its buffer")
    return {
        "kind": "implicit",
        "route": expected,
        "spatial": model.rank,
        "groups": p["groups"],
        "gemm": gemm,
        "weightSlack": max(0, reach - counts["B"]),
    }


def plan(entry, counts, parameters, threads, execution):
    """Validate one launch against conv.cpp's routing and return its metadata."""
    info = signature(entry)
    if info["kind"] == "unfold":
        return _plan_unfold(info, counts, parameters, threads, execution)
    if info["kind"] == "implicit":
        return _plan_implicit(entry, info, counts, parameters, threads, execution)
    return _plan_dw1(counts, parameters, threads, execution)
