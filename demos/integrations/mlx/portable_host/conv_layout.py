"""Routing, storage and launch contracts for MLX's ported convolution kernels.

``Convolution::eval_gpu`` in ``conv.inc`` routes every convolution as
mlx/backend/metal/conv.cpp does. Its explicit GEMM routes unfold the input
with ``naive_unfold_nd`` (``groups == 1``) or ``naive_unfold_transpose_nd``
and multiply the unfolded rows with the steel GEMM, whose launches
``gemm_layout`` checks; fully separable 1-D convolutions run
``depthwise_conv_1d``. ``route`` mirrors conv.cpp's routing, and ``plan``
checks every unfold and depthwise launch against it and proves each read and
write in bounds before a translated kernel runs.

Both kernels read only ``thread_position_in_grid`` and ``threads_per_grid``,
so the host runs them as one-thread workgroups over conv.cpp's exact thread
grid: Direct3D and OpenGL cannot clip the partial workgroups conv.cpp's
``dispatch_threads`` leaves. ``depthwise_conv_1d_*_large`` serves inputs of
more than 2^31 - 1 elements, which every host launch's bound excludes.
"""

import ctypes
import math
import re
import struct

from demos.integrations.mlx.portable_host import gemv_layout
from demos.integrations.mlx.portable_host.quantization_layout import ITEM_SIZES

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
ENTRY_PREFIXES = ("naive_unfold_", "depthwise_conv_1d_")
WORKGROUP = (1, 1, 1)
# conv.metal's [[buffer(i)]] parameters, in index order.
PARAMETERS = {
    "unfold": ("in", "out", "params", "row_offset"),
    "dw1": ("in", "w", "out", "strides", "kernel_size"),
}
OUTPUT = "out"
ARRAYS = {"unfold": ("in", "out"), "dw1": ("in", "w", "out")}
SCALAR_TYPES = {"row_offset": "int32", "strides": "int32", "kernel_size": "int32"}
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
# The routes conv.cpp's eval_gpu takes; the host runs the first three.
PORTED_ROUTES = (
    "explicit_gemm_conv_ND_gpu",
    "explicit_gemm_conv_group_ND_gpu",
    "depthwise_conv_1D_gpu",
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
    raise ValueError("Unsupported convolution entry")


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
    if set(buffers) != set(names):
        raise ValueError("Convolution buffers do not match the entry")
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    counts, parameters, spans = {}, {}, {}
    for name in names:
        buffer, dtype = buffers[name], parameter_type(entry, name)
        itemsize = 1 if dtype == "uint8" else ITEM_SIZES.get(dtype, 4)
        limit = {
            "params": PARAMS_SIZES.get(info.get("spatial")),
            "row_offset": 1,
            "strides": 3,
            "kernel_size": 1,
        }.get(name, MAX_ELEMENTS)
        count = buffer.count
        if (
            not buffer.data
            or buffer.data % itemsize
            or buffer.dtype != dtype.encode("ascii")
            or buffer.output != int(name == OUTPUT)
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
            parameters[name] = values if name == "strides" else values[0]
    if "strides" in parameters and len(parameters["strides"]) != 3:
        raise ValueError("depthwise_conv_1d reads three input strides")
    low, high = spans[OUTPUT]
    if any(
        other != OUTPUT and max(low, start) < min(high, end)
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
        or counts[OUTPUT] != threads
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
        or counts[OUTPUT] != threads
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


def plan(entry, counts, parameters, threads, execution):
    """Validate one launch against conv.cpp's routing and return its metadata."""
    info = signature(entry)
    if info["kind"] == "unfold":
        return _plan_unfold(info, counts, parameters, threads, execution)
    return _plan_dw1(counts, parameters, threads, execution)
