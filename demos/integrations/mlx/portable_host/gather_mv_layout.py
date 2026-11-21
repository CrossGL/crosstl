"""Storage and launch contracts for MLX's gathered matrix-vector kernels.

``GatherMM::eval_gpu`` in ``gather_mm.inc`` sends gathered products with one
output row (and unsorted right indices) or one output column to
``gather_mv`` as matmul.cpp does: ``check_transpose`` on the matrix and the
vector, the flip of the matrix transposition for vector-matrix products, then
``gemv_gather`` or ``gemv_t_gather`` with gather_mv's tile shape. Neither
kernel computes axpby, so ``bias``, ``alpha`` and ``beta`` stay unbound. The
kernels read one uint32 index per output batch from each index buffer and move
the vector and the matrix to the gathered batch with ``elem_to_loc``;
``plan`` reads those indices and proves every gathered operand lies inside
its buffer. ``gather_mm_layout`` describes the matrix-matrix routes.
"""

import ctypes
import math
import re

from demos.integrations.mlx.portable_host import gemv_layout
from demos.integrations.mlx.portable_host.quantization_layout import ITEM_SIZES

SOURCE = gemv_layout.SOURCE
SIMD_SIZE = gemv_layout.SIMD_SIZE
MAX_ELEMENTS = gemv_layout.MAX_ELEMENTS
SIGNED_MAX = gemv_layout.SIGNED_MAX
MAX_GROUPS = gemv_layout.MAX_GROUPS
MAX_BATCH_DIMENSIONS = gemv_layout.MAX_BATCH_DIMENSIONS
DTYPES = gemv_layout.DTYPES
PATTERN = re.compile(
    r"(gemv_gather|gemv_t_gather)_(float32|float16|bfloat16)"
    r"_bm(\d+)_bn(\d+)_sm(\d+)_sn(\d+)_tm(\d+)_tn(\d+)"
)
ENTRY_PREFIXES = ("gemv_gather_", "gemv_t_gather_")
# gemv.metal's instantiate_gemv_bs_blocks and instantiate_gemv_t_bs_blocks.
BLOCKS = {
    "gemv_gather": ((4, 1, 1, 32, 1, 4), (4, 1, 1, 32, 4, 4), (8, 1, 1, 32, 4, 4)),
    "gemv_t_gather": gemv_layout.BLOCKS["gemv_t"],
}
# [[buffer(i)]] parameters of gemv_gather and gemv_t_gather, in index order.
PARAMETERS = (
    "mat",
    "in_vec",
    "bias",
    "out_vec",
    "in_vec_size",
    "out_vec_size",
    "marix_ld",
    "alpha",
    "beta",
    "batch_ndim",
    "batch_shape",
    "index_batch_strides",
    "vector_batch_ndim",
    "vector_batch_shape",
    "vector_batch_stride",
    "matrix_batch_ndim",
    "matrix_batch_shape",
    "matrix_batch_stride",
    "vec_indices",
    "mat_indices",
)
# What gather_mv binds: everything but the axpby epilogue.
HOST_PARAMETERS = tuple(
    name for name in PARAMETERS if name not in ("bias", "alpha", "beta")
)
MATRIX_PARAMETERS = ("mat", "in_vec", "out_vec")
INDEX_PARAMETERS = ("vec_indices", "mat_indices")
SCALAR_PARAMETERS = (
    "in_vec_size",
    "out_vec_size",
    "marix_ld",
    "batch_ndim",
    "vector_batch_ndim",
    "matrix_batch_ndim",
)
VECTOR_TYPES = {
    "batch_shape": "int32",
    "index_batch_strides": "int64",
    "vector_batch_shape": "int32",
    "vector_batch_stride": "int64",
    "matrix_batch_shape": "int32",
    "matrix_batch_stride": "int64",
}
INTEGER_TYPES = {
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "uint32": ctypes.c_uint32,
}
# set_vector_bytes(out.shape()) and the operand shapes and strides carry two
# matrix axes after at most MAX_BATCH_DIMENSIONS batch axes; the index strides
# concatenate two index arrays' strides.
VECTOR_LIMITS = {
    "batch_shape": MAX_BATCH_DIMENSIONS + 2,
    "index_batch_strides": 2 * MAX_BATCH_DIMENSIONS,
    "vector_batch_shape": MAX_BATCH_DIMENSIONS + 2,
    "vector_batch_stride": MAX_BATCH_DIMENSIONS + 2,
    "matrix_batch_shape": MAX_BATCH_DIMENSIONS + 2,
    "matrix_batch_stride": MAX_BATCH_DIMENSIONS + 2,
}
# What gather_mv leaves unbound; GEMVKernel<..., false> never reads them.
INERT = {"alpha": 1.0, "beta": 0.0}
# Expressions OpenGL may evaluate in 32 bits, as the gemv gate asserts them.
INDEX_RANGE_ASSERTIONS = {
    "gemv_gather": gemv_layout.INDEX_RANGE_ASSERTIONS["gemv"],
    "gemv_t_gather": gemv_layout.INDEX_RANGE_ASSERTIONS["gemv_t"],
    "gather": (
        "buffer_load(index_batch_strides, 0) * tid.z",
        "buffer_load(index_batch_strides, batch_ndim) * tid.z",
        "batch_offsets.x",
        "batch_offsets.y",
        "indx_vec * buffer_load(vector_batch_stride, 0)",
        "in_vec + indx_vec * buffer_load(vector_batch_stride, 0)",
        "indx_mat * buffer_load(matrix_batch_stride, 0)",
        "mat + indx_mat * buffer_load(matrix_batch_stride, 0)",
    ),
}


def signature(entry):
    """The kernel, element type and tile of an entry."""
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError("Unsupported gathered matrix-vector entry")
    kind, dtype, *tile = match.groups()
    block = tuple(int(value) for value in tile)
    if block not in BLOCKS[kind]:
        raise ValueError("Gathered matrix-vector entry is not instantiated")
    bm, bn, sm, sn, tm, tn = block
    return {
        "kind": kind,
        "dtype": dtype,
        "bm": bm,
        "bn": bn,
        "sm": sm,
        "sn": sn,
        "tm": tm,
        "tn": tn,
    }


def workgroup_size(entry):
    info = signature(entry)
    return (SIMD_SIZE, info["bn"], info["bm"])


def outputs_per_group(entry):
    """gather_mv's n_out_per_tgp."""
    info = signature(entry)
    if info["kind"] == "gemv_t_gather":
        return info["bn"] * info["sn"] * info["tn"]
    return info["bm"] * info["sm"] * info["tm"]


def index_assertions(entry):
    return INDEX_RANGE_ASSERTIONS[signature(entry)["kind"]] + (
        INDEX_RANGE_ASSERTIONS["gather"]
    )


def select(dtype, transpose_mat, in_size, out_size):
    """gather_mv's kernel name for one product."""
    if dtype not in DTYPES:
        raise ValueError(
            "Gathered matrix-vector products support float32, float16 and bfloat16"
        )
    if transpose_mat:
        sm, sn = (4, 8) if in_size >= 8192 and out_size >= 2048 else (8, 4)
        bn = 16 if out_size >= 2048 else 4 if out_size >= 512 else 2
        bm, tm, tn = 1, 4, 1 if out_size < 4 else 4
        kind = "gemv_t_gather"
    else:
        bm, bn, sm, sn, tn = 8 if out_size >= 4096 else 4, 1, 1, 32, 4
        tm = 1 if out_size < 4 else 4
        kind = "gemv_gather"
    return f"{kind}_{dtype}_bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}"


def parameter_type(entry, name):
    if name in MATRIX_PARAMETERS:
        return signature(entry)["dtype"]
    if name in SCALAR_PARAMETERS:
        return "int32"
    if name in INDEX_PARAMETERS:
        return "uint32"
    return VECTOR_TYPES[name]


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters.

    Returns ``(counts, parameters, indices)``: element counts of the matrix
    and index buffers, the values of the scalar and vector parameters, and
    zero-copy views of the two index buffers.
    """
    if set(buffers) != set(HOST_PARAMETERS):
        raise ValueError("Gathered matrix-vector buffers do not match the entry")
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    counts, parameters, indices, spans = {}, {}, {}, {}
    for name in HOST_PARAMETERS:
        buffer, dtype = buffers[name], parameter_type(entry, name)
        itemsize = ITEM_SIZES.get(dtype) or ctypes.sizeof(INTEGER_TYPES[dtype])
        count = buffer.count
        limit = (
            1 if name in SCALAR_PARAMETERS else VECTOR_LIMITS.get(name, MAX_ELEMENTS)
        )
        if (
            not buffer.data
            or buffer.data % itemsize
            or buffer.dtype != dtype.encode("ascii")
            or buffer.output != int(name == "out_vec")
            or type(count) is not int
            or not 1 <= count <= limit
            or buffer.data + count * itemsize > address_limit
        ):
            raise ValueError(
                "Gathered matrix-vector buffer storage or direction does not match"
            )
        spans[name] = (buffer.data, buffer.data + count * itemsize)
        if name in MATRIX_PARAMETERS:
            counts[name] = count
            continue
        values = (INTEGER_TYPES[dtype] * count).from_address(buffer.data)
        if name in INDEX_PARAMETERS:
            counts[name], indices[name] = count, values
            continue
        parameters[name] = values[0] if name in SCALAR_PARAMETERS else list(values)
    low, high = spans["out_vec"]
    if any(
        other != "out_vec" and max(low, start) < min(high, end)
        for other, (start, end) in spans.items()
    ):
        raise ValueError("Gathered matrix-vector output overlaps another buffer")
    return counts, parameters, indices


def location(index, shape, strides):
    """utils.h elem_to_loc over the last ``len(shape)`` axes."""
    offset = 0
    for extent, stride in zip(reversed(shape), reversed(strides)):
        offset += (index % extent) * stride
        index //= extent
    return offset


def gathered_offset(index, ndim, shape, strides):
    """The kernels' move to batch ``index``: elem_to_loc, or one stride."""
    if ndim > 1:
        return location(index, shape[:ndim], strides[:ndim])
    return index * strides[0]


def batch_index_locations(z, ndim, shape, strides):
    """Where batch ``z`` reads the vector and matrix indices.

    ``elem_to_loc_broadcast`` with more than one batch axis, otherwise the
    first stride of each index array scaled by ``z``.
    """
    if ndim > 1:
        return (
            location(z, shape[:ndim], strides[:ndim]),
            location(z, shape[:ndim], strides[ndim : 2 * ndim]),
        )
    return strides[0] * z, strides[ndim] * z


def _check_dimensions(parameters):
    k = parameters.get("in_vec_size")
    out = parameters.get("out_vec_size")
    ld = parameters.get("marix_ld")
    ndim = parameters.get("batch_ndim")
    vec_ndim = parameters.get("vector_batch_ndim")
    mat_ndim = parameters.get("matrix_batch_ndim")
    shape = parameters.get("batch_shape")
    strides = parameters.get("index_batch_strides")
    vec_shape = parameters.get("vector_batch_shape")
    vec_strides = parameters.get("vector_batch_stride")
    mat_shape = parameters.get("matrix_batch_shape")
    mat_strides = parameters.get("matrix_batch_stride")
    if (
        any(type(value) is not int for value in (k, out, ld, ndim, vec_ndim, mat_ndim))
        or not 1 <= min(k, out) <= max(k, out) <= MAX_ELEMENTS
        or not 0 <= ld <= MAX_ELEMENTS
        or not 0 <= ndim <= MAX_BATCH_DIMENSIONS
        or not 0 <= vec_ndim <= MAX_BATCH_DIMENSIONS
        or not 0 <= mat_ndim <= MAX_BATCH_DIMENSIONS
        or len(shape) != ndim + 2
        # Scalar indices bind two zero strides, which only batch 0 reads.
        or len(strides) != (2 * ndim if ndim else 2)
        or (not ndim and strides != [0, 0])
        or not len(vec_shape) == len(vec_strides) == vec_ndim + 2
        or not len(mat_shape) == len(mat_strides) == mat_ndim + 2
        or any(extent < 1 for extent in (*shape, *vec_shape, *mat_shape))
        or any(stride < 0 for stride in (*strides, *vec_strides, *mat_strides))
    ):
        raise ValueError("Gathered matrix-vector dimensions are invalid")
    return k, out, ld, ndim


def plan(entry, counts, parameters, indices, threads, execution):
    """Validate one launch against gather_mv's routing and return its metadata.

    Every batch's two indices are read from their buffers; the gathered
    vector must hold ``K`` elements and the gathered matrix every row or
    column the kernel reads.
    """
    info = signature(entry)
    k, out, ld, ndim = _check_dimensions(parameters)
    shape = parameters["batch_shape"]
    batches = math.prod(shape[:ndim])
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError("Gathered matrix-vector launch geometry is missing")
    tiles, rows, groups = execution["workgroupCount"]
    if (
        list(execution["workgroupSize"]) != list(workgroup_size(entry))
        or any(type(value) is not int for value in (tiles, rows, groups))
        or rows != 1
        or tiles != -(-out // outputs_per_group(entry))
        or not 1 <= tiles <= MAX_GROUPS
        or not 1 <= groups <= MAX_GROUPS
        or groups != batches
    ):
        raise ValueError("Gathered matrix-vector launch does not match MLX's grid")
    transposed = info["kind"] == "gemv_t_gather"
    if select(info["dtype"], transposed, k, out) != entry:
        raise ValueError("Gathered matrix-vector entry differs from MLX's routing")
    if threads != out * batches or counts["out_vec"] != threads:
        raise ValueError("Gathered matrix-vector output extent differs")
    # gemv reads out rows of K elements, gemv_t K rows of out elements.
    extents = {
        "in_vec": k,
        "mat": (out - 1) * ld + k if not transposed else (k - 1) * ld + out,
    }
    operands = {
        "in_vec": ("vec_indices", "vector_batch_ndim", "vector_batch_shape"),
        "mat": ("mat_indices", "matrix_batch_ndim", "matrix_batch_shape"),
    }
    stride_names = {"in_vec": "vector_batch_stride", "mat": "matrix_batch_stride"}
    strides = parameters["index_batch_strides"]
    gathered = {"vec_indices": [], "mat_indices": []}
    for z in range(batches):
        for name, location_ in zip(
            INDEX_PARAMETERS, batch_index_locations(z, ndim, shape, strides)
        ):
            if location_ >= counts[name]:
                raise ValueError(
                    f"Gathered matrix-vector {name} read exceeds its buffer"
                )
            gathered[name].append(indices[name][location_])
    for name, (index_name, ndim_name, shape_name) in operands.items():
        operand_ndim = parameters[ndim_name]
        operand_shape = parameters[shape_name]
        operand_strides = parameters[stride_names[name]]
        batch_count = math.prod(operand_shape[:operand_ndim])
        for index in set(gathered[index_name]):
            if operand_ndim and index >= batch_count:
                raise ValueError(f"Gathered matrix-vector {index_name} is out of range")
            offset = gathered_offset(
                index, operand_ndim, operand_shape, operand_strides
            )
            if offset + extents[name] > counts[name]:
                raise ValueError(f"Gathered matrix-vector {name} exceeds its buffer")
    return {
        "kind": info["kind"],
        "K": k,
        "outVecSize": out,
        "matrixLd": ld,
        "batches": batches,
        "vectorIndices": gathered["vec_indices"],
        "matrixIndices": gathered["mat_indices"],
        "inert": dict(INERT),
    }
