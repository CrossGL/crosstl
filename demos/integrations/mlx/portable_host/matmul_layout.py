"""Storage and launch contracts for the affine quantized matrix-vector kernels.

``QuantizedMatmul::eval_gpu`` in ``backend.cpp`` routes affine products with
few input vectors as quantized.cpp does. Transposed weights go to
``dispatch_qmv``'s kernels: ``affine_qmv_quad`` for ``K`` of 64 or 128 and
power-of-two widths, then ``affine_qmv_fast`` when ``N`` and ``K`` are aligned,
otherwise ``affine_qmv``. The portable host reports no Apple GPU generation, so
it never selects ``affine_qmv_wide`` (MLX uses it from generation 15).
Untransposed weights go to ``affine_qvm`` while ``K`` is below 1024; MLX splits
longer reductions over ``qvm_split_k`` and a sum, which the host does not route.
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
ENTRY_PREFIXES = ("affine_qmv_", "affine_qvm_")
# QuantizedMatmul::eval_gpu sends untransposed products with K at or above
# this through qvm_split_k.
QVM_SPLIT_K = 1024
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
    """The kernel, element type, group size, bits, ``D`` and batching of an entry."""
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
    return signature(entry)["kernel"] != "qvm"


def tile_rows(entry):
    """Outputs per threadgroup.

    qmv_quad's 8 quads x 8 rows, qmv's 2 simdgroups x 4 rows, and qvm's 2
    simdgroups x ``min(group_size, 32)`` columns.
    """
    info = signature(entry)
    if info["kernel"] == "qvm":
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
    matrix-matrix kernels and qvm_split_k.
    """
    if m >= vector_limit(k, n, transpose):
        raise ValueError("Quantized matrix-matrix products are not routed")
    stem = f"{SOURCE_TYPES[dtype]}_gs_{group}_b_{bits}"
    batch = "_batch_1" if batches > 1 else "_batch_0"
    if not transpose:
        if k >= QVM_SPLIT_K:
            raise ValueError("Split-K quantized vector-matrix products are not routed")
        return f"affine_qvm_{stem}{batch}"
    if k in QUAD_DIMENSIONS and bits in (2, 4, 8):
        return f"affine_qmv_quad_{stem}_d_{k}{batch}"
    fast = n % 8 == 0 and k % fast_alignment(bits) == 0
    return f"affine_{'qmv_fast' if fast else 'qmv'}_{stem}{batch}"


def parameter_names(entry):
    names = (*MATRIX_PARAMETERS, *SCALAR_PARAMETERS)
    return (*names, *BATCH_PARAMETERS) if signature(entry)["batched"] else names


def parameter_type(entry, name):
    if name == "w":
        return "uint32"
    if name in MATRIX_PARAMETERS:
        return signature(entry)["dtype"]
    if name in SCALAR_PARAMETERS or name in BATCH_SCALARS:
        return "int32"
    return BATCH_VECTORS[name]


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters.

    Returns ``(counts, parameters)``: element counts of the matrix buffers and
    the integer values of the scalar and vector parameters.
    """
    names = parameter_names(entry)
    if set(buffers) != set(names):
        raise ValueError("Quantized matrix-vector buffers do not match the entry")
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    counts, parameters, spans = {}, {}, {}
    for name in names:
        buffer, dtype = buffers[name], parameter_type(entry, name)
        itemsize = ITEM_SIZES.get(dtype) or ctypes.sizeof(
            INTEGER_TYPES.get(dtype, ctypes.c_uint32)
        )
        count = buffer.count
        scalar = name in SCALAR_PARAMETERS or name in BATCH_SCALARS
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
            raise ValueError(
                "Quantized matrix-vector buffer storage or direction does not match"
            )
        spans[name] = (buffer.data, buffer.data + count * itemsize)
        if name in MATRIX_PARAMETERS:
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
        raise ValueError("Quantized matrix-vector output overlaps another buffer")
    return counts, parameters


def _location(index, shape, strides):
    """utils.h elem_to_loc over the leading batch dimensions."""
    location = 0
    for extent, stride in zip(reversed(shape), reversed(strides)):
        location += (index % extent) * stride
        index //= extent
    return location


def _batch_offsets(parameters, m, k, rows, words, groups, batches):
    """Per ``tid.z`` element offsets of x, w, scales and biases.

    ``rows`` is the weight matrices' row count: ``N`` for transposed weights,
    ``K`` otherwise.
    """
    x_shape, x_strides = parameters["x_shape"], parameters["x_strides"]
    w_shape, w_strides = parameters["w_shape"], parameters["w_strides"]
    s_strides, b_strides = parameters["s_strides"], parameters["b_strides"]
    x_batch, w_batch = parameters["x_batch_ndims"], parameters["w_batch_ndims"]
    if (
        x_batch != len(x_shape) - 2
        or len(x_strides) != len(x_shape)
        or not 1 <= x_batch <= MAX_BATCH_DIMENSIONS
        or w_batch != len(w_shape) - 2
        or not 0 <= w_batch <= MAX_BATCH_DIMENSIONS
        or not len(w_strides) == len(s_strides) == len(b_strides) == len(w_shape)
    ):
        raise ValueError("Quantized matrix-vector batch ranks do not match")
    if x_shape[-2:] != [m, k] or w_shape[-2:] != [rows, words]:
        raise ValueError("Quantized matrix-vector matrix shapes do not match")
    # ensure_row_contiguous_matrix: unit columns and dense rows.
    for strides, extent, columns in (
        (x_strides, m, k),
        (w_strides, rows, words),
        (s_strides, rows, groups),
        (b_strides, rows, groups),
    ):
        if strides[-1] != 1 or (extent > 1 and strides[-2] != columns):
            raise ValueError("Quantized matrix-vector matrices are not row contiguous")
    if math.prod(x_shape[:x_batch]) != batches or (
        w_batch and math.prod(w_shape[:w_batch]) != batches
    ):
        raise ValueError("Quantized matrix-vector batch extents do not match")
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
        raise ValueError("Quantized matrix-vector batch strides must be nonnegative")
    return [
        (
            _location(z, x_shape[:x_batch], x_strides[:x_batch]),
            _location(z, w_shape[:w_batch], w_strides[:w_batch]),
            _location(z, w_shape[:w_batch], s_strides[:w_batch]),
            _location(z, w_shape[:w_batch], b_strides[:w_batch]),
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
        offsets = _batch_offsets(parameters, m, k, rows, words, groups, batches)
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
