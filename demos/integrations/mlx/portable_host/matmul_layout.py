"""Storage and launch contracts for the affine quantized matrix-vector kernels.

``QuantizedMatmul::eval_gpu`` in ``backend.cpp`` routes affine products with
few input vectors as quantized.cpp does. Transposed weights go to
``dispatch_qmv``'s kernels: ``affine_qmv_quad`` for ``K`` of 64 or 128 and
power-of-two widths, then ``affine_qmv_fast`` when ``N`` and ``K`` are aligned,
otherwise ``affine_qmv``. The portable host reports no Apple GPU generation, so
it never selects ``affine_qmv_wide`` (MLX uses it from generation 15).
Untransposed weights go to ``affine_qvm`` while ``K`` is below 1024, and longer
reductions to ``affine_qvm_split_k``: 8 partitions of ``K`` (32 above 8192)
whose partial products a column sum then adds.
"""

import ctypes
import math
import re

from demos.integrations.mlx.portable_host.quantization_layout import (
    ITEM_SIZES,
    SOURCE_TYPES,
)

# OpenGL may evaluate adjust_matrix_offsets' batch offsets in 32 bits; the
# translation asserts that each names an element of a bound buffer, and no
# bound buffer holds more elements than this.
MAX_ELEMENTS = 2**28 - 1
MAX_GROUPS = 65535
SIMD_SIZE = 32
QUAD_DIMENSIONS = (64, 128)
PATTERN = re.compile(
    r"affine_(qmv_quad|qmv_fast|qmv|qvm)_(float|float16_t|bfloat16_t)"
    r"_gs_(32|64|128)_b_(2|3|4|5|6|8)(?:_d_(64|128))?_batch_(0|1)"
)
SPLIT_PATTERN = re.compile(
    r"affine_qvm_split_k_(float|float16_t|bfloat16_t)_gs_(32|64|128)"
    r"_b_(2|3|4|5|6|8)_spk_(8|32)"
)
ENTRY_PREFIXES = ("affine_qmv_", "affine_qvm_")
LABEL = "Quantized matrix-vector"
# QuantizedMatmul::eval_gpu sends untransposed products with K at or above
# this through qvm_split_k, in 32 partitions above QVM_WIDE_SPLIT and 8 below.
QVM_SPLIT_K = 1024
QVM_WIDE_SPLIT = 8192
SPLIT_SCALARS = ("final_block_size",)
MATRIX_PARAMETERS = ("w", "scales", "biases", "x", "y")
SCALAR_PARAMETERS = ("in_vec_size", "out_vec_size")
BATCH_SCALARS = ("x_batch_ndims", "w_batch_ndims")
BATCH_VECTORS = {
    "x_shape": "int32",
    "x_strides": "int64",
    "w_shape": "int32",
    "w_strides": "int64",
    "s_strides": "int64",
    "b_strides": "int64",
}
BATCH_PARAMETERS = (
    "x_batch_ndims",
    "x_shape",
    "x_strides",
    "w_batch_ndims",
    "w_shape",
    "w_strides",
    "s_strides",
    "b_strides",
)
INTEGER_TYPES = {"int32": ctypes.c_int32, "int64": ctypes.c_int64}
MAX_BATCH_DIMENSIONS = 16


def signature(entry):
    """The kernel, element type, group size, bits, ``D`` and batching of an entry.

    Split-K entries add their partition count and always bind batch views.
    """
    split = SPLIT_PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if split is not None:
        source, group, bits, partitions = split.groups()
        return {
            "kernel": "qvm_split_k",
            "dtype": next(
                key for key, value in SOURCE_TYPES.items() if value == source
            ),
            "groupSize": int(group),
            "bits": int(bits),
            "D": None,
            "batched": True,
            "splitK": int(partitions),
        }
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError("Unsupported quantized matrix-vector entry")
    kernel, source, group, bits, dimension, batched = match.groups()
    group, bits = int(group), int(bits)
    quad = kernel == "qmv_quad"
    if quad != (dimension is not None) or (
        quad and (bits not in (2, 4, 8) or int(dimension) % group)
    ):
        # MLX instantiates every width and group size, but dispatches qmv_quad
        # only for power-of-two widths over complete groups.
        raise ValueError("Quantized matrix-vector entry is never dispatched")
    dtype = next(key for key, value in SOURCE_TYPES.items() if value == source)
    return {
        "kernel": kernel,
        "dtype": dtype,
        "groupSize": group,
        "bits": bits,
        "D": int(dimension) if quad else None,
        "batched": batched == "1",
    }


def word_pack_factor(bits):
    """quantized.h get_pack_factor<bits, 32>()."""
    return 8 if bits in (3, 5) else 4 if bits == 6 else 32 // bits


def fast_alignment(bits):
    """quantized.cpp qmv_fast_k_alignment(bits)."""
    return word_pack_factor(bits) * (1 if bits == 2 else 2) * SIMD_SIZE


def workgroup_size(entry):
    return (SIMD_SIZE, 1 if signature(entry)["kernel"] == "qmv_quad" else 2, 1)


def transposed(entry):
    """Whether the entry multiplies by transposed weights, ``w`` of ``(N, K)``."""
    return signature(entry)["kernel"] not in ("qvm", "qvm_split_k")


def tile_rows(entry):
    """Outputs per threadgroup.

    qmv_quad's 8 quads x 8 rows, qmv's 2 simdgroups x 4 rows, and qvm's 2
    simdgroups x ``min(group_size, 32)`` columns.
    """
    info = signature(entry)
    if info["kernel"] in ("qvm", "qvm_split_k"):
        return 2 * min(info["groupSize"], SIMD_SIZE)
    return 64 if info["kernel"] == "qmv_quad" else 8


def vector_limit(k, n, transpose):
    """quantized.cpp's matrix-vector bound on ``M``.

    get_qmv_batch_limit for a device without an Apple GPU generation when the
    weights are transposed, and 4 otherwise.
    """
    if not transpose:
        return 4
    return 18 if k <= 2048 and n <= 2048 else 12 if k <= 4096 and n <= 4096 else 10


def select(dtype, group, bits, m, n, k, batches, transpose=True):
    """quantized.cpp QuantizedMatmul::eval_gpu's entry for one product.

    Raises for the products MLX sends to kernels the host does not route: the
    matrix-matrix kernels.
    """
    if m >= vector_limit(k, n, transpose):
        raise ValueError("Quantized matrix-matrix products are not routed")
    stem = f"{SOURCE_TYPES[dtype]}_gs_{group}_b_{bits}"
    batch = "_batch_1" if batches > 1 else "_batch_0"
    if not transpose:
        if k >= QVM_SPLIT_K:
            partitions = 32 if k > QVM_WIDE_SPLIT else 8
            return f"affine_qvm_split_k_{stem}_spk_{partitions}"
        return f"affine_qvm_{stem}{batch}"
    if k in QUAD_DIMENSIONS and bits in (2, 4, 8):
        return f"affine_qmv_quad_{stem}_d_{k}{batch}"
    fast = n % 8 == 0 and k % fast_alignment(bits) == 0
    return f"affine_{'qmv_fast' if fast else 'qmv'}_{stem}{batch}"


def parameter_names(entry):
    info = signature(entry)
    names = (*MATRIX_PARAMETERS, *SCALAR_PARAMETERS)
    if "splitK" in info:
        return (*names, *BATCH_PARAMETERS, *SPLIT_SCALARS)
    return (*names, *BATCH_PARAMETERS) if info["batched"] else names


def parameter_type(entry, name):
    if name == "w":
        return "uint32"
    if name in MATRIX_PARAMETERS:
        return signature(entry)["dtype"]
    if name in SCALAR_PARAMETERS or name in BATCH_SCALARS or name in SPLIT_SCALARS:
        return "int32"
    return BATCH_VECTORS[name]


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters.

    Returns ``(counts, parameters)``: element counts of the matrix buffers and
    the integer values of the scalar and vector parameters.
    """
    names = parameter_names(entry)
    return read_buffers(
        buffers,
        {name: parameter_type(entry, name) for name in names},
        (*SCALAR_PARAMETERS, *BATCH_SCALARS, *SPLIT_SCALARS),
        LABEL,
    )


def read_buffers(buffers, types, scalars, label, matrices=MATRIX_PARAMETERS):
    """``read_storage`` for the parameters ``types`` names, in their order.

    ``scalars`` names the single-value parameters; the matrix buffers are
    ``matrices`` with ``y`` the only output, and the other names are integer
    vectors of at most ``MAX_BATCH_DIMENSIONS + 2`` values.
    """
    if set(buffers) != set(types):
        raise ValueError(f"{label} buffers do not match the entry")
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    counts, parameters, spans = {}, {}, {}
    for name, dtype in types.items():
        buffer = buffers[name]
        itemsize = ITEM_SIZES.get(dtype) or ctypes.sizeof(
            INTEGER_TYPES.get(dtype, ctypes.c_uint32)
        )
        count = buffer.count
        scalar = name in scalars
        if (
            not buffer.data
            or buffer.data % itemsize
            or buffer.dtype != dtype.encode("ascii")
            or buffer.output != int(name == "y")
            or type(count) is not int
            or not 1 <= count <= (1 if scalar else MAX_ELEMENTS)
            or (name in BATCH_VECTORS and count > MAX_BATCH_DIMENSIONS + 2)
            or buffer.data + count * itemsize > address_limit
        ):
            raise ValueError(f"{label} buffer storage or direction does not match")
        spans[name] = (buffer.data, buffer.data + count * itemsize)
        if name in matrices:
            counts[name] = count
            continue
        values = list(
            ctypes.cast(
                buffer.data, ctypes.POINTER(INTEGER_TYPES[dtype] * count)
            ).contents
        )
        parameters[name] = values[0] if scalar else values
    low, high = spans["y"]
    if any(
        other != "y" and max(low, start) < min(high, end)
        for other, (start, end) in spans.items()
    ):
        raise ValueError(f"{label} output overlaps another buffer")
    return counts, parameters


def _location(index, shape, strides):
    """utils.h elem_to_loc over the leading batch dimensions."""
    location = 0
    for extent, stride in zip(reversed(shape), reversed(strides)):
        location += (index % extent) * stride
        index //= extent
    return location


def _matrix_offset(index, shape, strides):
    """quantized.h adjust_matrix_offsets: one batch axis is never wrapped.

    A single batch dimension offsets by ``tid.z * strides[0]`` for every
    ``tid.z`` in the grid, so it addresses past the axis when the grid has
    more batches than the axis; more dimensions go through elem_to_loc.
    """
    if len(shape) == 1:
        return index * strides[0]
    return _location(index, shape, strides)


def batch_offsets(parameters, m, k, rows, words, groups, batches, label=LABEL):
    """Per ``tid.z`` element offsets of x, w, scales and biases.

    ``rows`` is the weight matrices' row count: ``N`` for transposed weights,
    ``K`` otherwise. Without ``b_strides`` (kernels without biases) the
    biases' offsets follow the scales'.
    """
    x_shape, x_strides = parameters["x_shape"], parameters["x_strides"]
    w_shape, w_strides = parameters["w_shape"], parameters["w_strides"]
    s_strides = parameters["s_strides"]
    b_strides = parameters.get("b_strides", s_strides)
    x_batch, w_batch = parameters["x_batch_ndims"], parameters["w_batch_ndims"]
    if (
        x_batch != len(x_shape) - 2
        or len(x_strides) != len(x_shape)
        or not 1 <= x_batch <= MAX_BATCH_DIMENSIONS
        or w_batch != len(w_shape) - 2
        or not 0 <= w_batch <= MAX_BATCH_DIMENSIONS
        or not len(w_strides) == len(s_strides) == len(b_strides) == len(w_shape)
    ):
        raise ValueError(f"{label} batch ranks do not match")
    if x_shape[-2:] != [m, k] or w_shape[-2:] != [rows, words]:
        raise ValueError(f"{label} matrix shapes do not match")
    # ensure_row_contiguous_matrix: unit columns and dense rows.
    for strides, extent, columns in (
        (x_strides, m, k),
        (w_strides, rows, words),
        (s_strides, rows, groups),
        (b_strides, rows, groups),
    ):
        if strides[-1] != 1 or (extent > 1 and strides[-2] != columns):
            raise ValueError(f"{label} matrices are not row contiguous")
    if math.prod(x_shape[:x_batch]) != batches or (
        w_batch and math.prod(w_shape[:w_batch]) != batches
    ):
        raise ValueError(f"{label} batch extents do not match")
    if any(
        extent < 1 or stride < 0
        for shape, strides, ndim in (
            (x_shape, x_strides, x_batch),
            (w_shape, w_strides, w_batch),
            (w_shape, s_strides, w_batch),
            (w_shape, b_strides, w_batch),
        )
        for extent, stride in zip(shape[:ndim], strides[:ndim])
    ):
        raise ValueError(f"{label} batch strides must be nonnegative")
    return [
        (
            _matrix_offset(z, x_shape[:x_batch], x_strides[:x_batch]),
            _matrix_offset(z, w_shape[:w_batch], w_strides[:w_batch]),
            _matrix_offset(z, w_shape[:w_batch], s_strides[:w_batch]),
            _matrix_offset(z, w_shape[:w_batch], b_strides[:w_batch]),
        )
        for z in range(batches)
    ]


def plan(entry, counts, parameters, threads, execution):
    """Validate one launch against MLX's routing and return its metadata.

    ``counts`` holds the element counts of the matrix buffers, ``parameters``
    the integer parameters the host supplied. The returned ``parameters`` add
    the inert unbatched values qmv's batch_0 kernels reflect but never read.
    """
    info = signature(entry)
    if "splitK" in info:
        return plan_split(
            entry,
            counts,
            parameters,
            threads,
            execution,
            info,
            lambda m, n, k, batches: select(
                info["dtype"], info["groupSize"], info["bits"], m, n, k, batches, False
            ),
        )
    k, n = parameters.get("in_vec_size"), parameters.get("out_vec_size")
    group, bits = info["groupSize"], info["bits"]
    transpose = transposed(entry)
    # Groups run along the weights' rows: K when transposed, N otherwise.
    grouped = k if transpose else n
    if (
        type(k) is not int
        or type(n) is not int
        or not 1 <= min(k, n) <= max(k, n) <= MAX_ELEMENTS
        or grouped % group
    ):
        raise ValueError("Quantized matrix-vector dimensions are invalid")
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError("Quantized matrix-vector launch geometry is missing")
    m, tiles, batches = execution["workgroupCount"]
    if (
        list(execution["workgroupSize"]) != list(workgroup_size(entry))
        or any(type(value) is not int for value in (m, tiles, batches))
        or not 1 <= m <= MAX_GROUPS
        or not 1 <= batches <= MAX_GROUPS
        or tiles != -(-n // tile_rows(entry))
        or tiles > MAX_GROUPS
    ):
        raise ValueError("Quantized matrix-vector launch does not match MLX's grid")
    if select(info["dtype"], group, bits, m, n, k, batches, transpose) != entry:
        raise ValueError("Quantized matrix-vector entry differs from MLX's routing")
    if threads != m * n * batches or counts["y"] != threads:
        raise ValueError("Quantized matrix-vector output extent differs")
    rows = n if transpose else k
    words, groups = grouped * bits // 32, grouped // group
    supplied = set(parameters) - set(SCALAR_PARAMETERS)
    if info["batched"]:
        if supplied != set(BATCH_PARAMETERS):
            raise ValueError("Batched quantized matrix-vector parameters are missing")
        offsets = batch_offsets(parameters, m, k, rows, words, groups, batches)
        values = dict(parameters)
    else:
        if supplied or batches != 1:
            raise ValueError("Unbatched quantized matrix-vector launch is batched")
        offsets = [(0, 0, 0, 0)]
        values = {
            **parameters,
            "x_batch_ndims": 0,
            "x_shape": [m, k],
            "x_strides": [k, 1],
            "w_batch_ndims": 0,
            "w_shape": [rows, words],
            "w_strides": [words, 1],
            "s_strides": [groups, 1],
            "b_strides": [groups, 1],
        }
    extents = {
        "x": m * k,
        "w": rows * words,
        "scales": rows * groups,
        "biases": rows * groups,
    }
    for column, name in enumerate(("x", "w", "scales", "biases")):
        if max(offset[column] for offset in offsets) + extents[name] > counts[name]:
            raise ValueError(f"Quantized matrix-vector {name} batch exceeds its buffer")
    # qmv_quad loads one scale and bias per row of its last tile, past N.
    padding = {"scales": 0, "biases": 0}
    if info["kernel"] == "qmv_quad":
        covered = tiles * tile_rows(entry)
        for column, name in ((2, "scales"), (3, "biases")):
            end = max(offset[column] for offset in offsets) + covered * groups
            padding[name] = max(0, end - counts[name])
    if any(counts[name] + padding.get(name, 0) > MAX_ELEMENTS for name in counts):
        raise ValueError("Quantized matrix-vector buffers exceed the index bound")
    return {
        "kernel": info["kernel"],
        "M": m,
        "N": n,
        "K": k,
        "batches": batches,
        "overreadPadding": padding,
        "parameters": values,
    }


def split_views(
    x_shape, x_strides, w_shape, w_strides, s_strides, b_strides, split, k, groups
):
    """quantized.cpp qvm_split_k's views: a ``split`` batch axis before the matrices.

    Takes the operands' shapes and strides as MLX binds them (a 1-D ``x`` is
    not yet widened) and the scales' row width ``groups``, and returns the
    bound batch parameters. Kernels without biases pass ``b_strides`` of
    ``None`` and bind no bias strides.
    """
    split_d = -(-k // split)
    x_shape, x_strides = list(x_shape), list(x_strides)
    if len(x_shape) == 1:
        x_shape.insert(0, 1)
        x_strides.insert(0, 0)
    x_ndim, w_ndim, words = len(x_shape), len(w_shape), w_shape[-1]
    x_shape.insert(x_ndim - 2, split)
    x_shape[-1] //= split
    x_strides.insert(x_ndim - 2, split_d)
    # quantized.cpp overwrites the row stride, which the kernel never reads.
    x_strides[x_ndim - 1] = split_d
    w_shape = list(w_shape)
    w_shape.insert(w_ndim - 2, split)
    w_shape[w_ndim - 1] //= split
    views = {
        "x_batch_ndims": x_ndim - 1,
        "x_shape": x_shape,
        "x_strides": x_strides,
        "w_batch_ndims": w_ndim - 1,
        "w_shape": w_shape,
        "w_strides": _inserted(w_strides, split_d * words),
        "s_strides": _inserted(s_strides, split_d * groups),
    }
    if b_strides is not None:
        views["b_strides"] = _inserted(b_strides, split_d * groups)
    return views


def plan_split(entry, counts, parameters, threads, execution, info, route):
    """Validate one qvm_split_k launch against quantized.cpp's partitions.

    ``info`` is the entry's signature and ``route(m, n, k, batches)`` the
    entry QuantizedMatmul selects for an untransposed product. The kernel
    binds biases and their strides exactly when ``counts`` holds ``biases``.
    """
    split, group, bits = info["splitK"], info["groupSize"], info["bits"]
    biased = "biases" in counts
    batch_parameters = tuple(
        name for name in BATCH_PARAMETERS if biased or name != "b_strides"
    )
    strided = ("x_strides", "w_strides", "s_strides", "b_strides")[: 3 + biased]
    split_d, n = parameters.get("in_vec_size"), parameters.get("out_vec_size")
    final = parameters.get("final_block_size")
    if any(type(value) is not int for value in (split_d, n, final)):
        raise ValueError("Quantized split-K dimensions are invalid")
    k = (split - 1) * split_d + final
    if (
        not 1 <= final <= split_d
        or split_d != -(-k // split)
        or not 1 <= n <= MAX_ELEMENTS
        or n % group
        or k > MAX_ELEMENTS
    ):
        raise ValueError("Quantized split-K dimensions are invalid")
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError("Quantized matrix-vector launch geometry is missing")
    m, tiles, batches = execution["workgroupCount"]
    # quantized.cpp: 2 simdgroups of min(group_size, 32) output columns.
    if (
        list(execution["workgroupSize"]) != [SIMD_SIZE, 2, 1]
        or any(type(value) is not int for value in (m, tiles, batches))
        or not 1 <= m <= MAX_GROUPS
        or not split <= batches <= MAX_GROUPS
        or batches % split
        or tiles != -(-n // (2 * min(group, SIMD_SIZE)))
        or tiles > MAX_GROUPS
    ):
        raise ValueError("Quantized matrix-vector launch does not match MLX's grid")
    if route(m, n, k, batches // split) != entry:
        raise ValueError("Quantized matrix-vector entry differs from MLX's routing")
    if threads != m * n * batches or counts["y"] != threads:
        raise ValueError("Quantized matrix-vector output extent differs")
    if set(parameters) != {*SCALAR_PARAMETERS, *batch_parameters, *SPLIT_SCALARS}:
        raise ValueError("Quantized split-K parameters are missing")
    words, groups = n * bits // 32, n // group
    x_shape, w_shape = parameters["x_shape"], parameters["w_shape"]
    x_batch, w_batch = parameters["x_batch_ndims"], parameters["w_batch_ndims"]
    if (
        x_batch != len(x_shape) - 2
        or not 1 <= x_batch <= MAX_BATCH_DIMENSIONS
        or w_batch != len(w_shape) - 2
        or not 1 <= w_batch <= MAX_BATCH_DIMENSIONS
        or len(parameters["x_strides"]) != len(x_shape)
        or any(len(parameters[name]) != len(w_shape) for name in strided[1:])
        or x_shape[x_batch - 1] != split
        or w_shape[w_batch - 1] != split
    ):
        raise ValueError("Quantized split-K batch ranks do not match")
    # Undo the reshape: drop the partition axis and restore the K extents and
    # the dense row strides ensure_row_contiguous_matrix leaves.
    x_strides = parameters["x_strides"]
    original = dict(
        x_shape=[*x_shape[: x_batch - 1], m, k],
        x_strides=[*x_strides[: x_batch - 1], k, 1],
        w_shape=[*w_shape[: w_batch - 1], k, words],
        w_strides=[*parameters["w_strides"][: w_batch - 1], words, 1],
        s_strides=[*parameters["s_strides"][: w_batch - 1], groups, 1],
        b_strides=(
            [*parameters["b_strides"][: w_batch - 1], groups, 1] if biased else None
        ),
    )
    expected = split_views(**original, split=split, k=k, groups=groups)
    if {name: parameters[name] for name in batch_parameters} != expected:
        raise ValueError("Quantized split-K views differ from quantized.cpp's")
    if math.prod(x_shape[:x_batch]) != batches or (
        math.prod(w_shape[:w_batch]) > batches
    ):
        raise ValueError("Quantized matrix-vector batch extents do not match")
    if any(stride < 0 for name in strided for stride in parameters[name]):
        raise ValueError("Quantized matrix-vector batch strides must be nonnegative")
    extents = {"x": (m - 1) * k, "w": 0, "scales": 0, "biases": 0}
    widths = {"x": None, "w": words, "scales": groups, "biases": groups}
    views = {
        "x": (x_shape, x_strides, x_batch),
        "w": (w_shape, parameters["w_strides"], w_batch),
        "scales": (w_shape, parameters["s_strides"], w_batch),
    }
    if biased:
        views["biases"] = (w_shape, parameters["b_strides"], w_batch)
    for z in range(batches):
        # The last partition reads the shorter final block.
        rows = final if z % split == split - 1 else split_d
        for name, (shape, strides, ndim) in views.items():
            offset = _matrix_offset(z, shape[:ndim], strides[:ndim])
            end = (
                offset + extents[name] + (rows if name == "x" else rows * widths[name])
            )
            if end > counts[name]:
                raise ValueError(
                    f"Quantized matrix-vector {name} batch exceeds its buffer"
                )
    if any(counts[name] > MAX_ELEMENTS for name in counts):
        raise ValueError("Quantized matrix-vector buffers exceed the index bound")
    return {
        "kernel": info["kernel"],
        "M": m,
        "N": n,
        "K": k,
        "batches": batches // split,
        "splitK": split,
        "overreadPadding": {name: 0 for name in ("scales", "biases")[: 1 + biased]},
        "parameters": dict(parameters),
    }


def _inserted(strides, value):
    values = list(strides)
    values.insert(len(values) - 2, value)
    return values
