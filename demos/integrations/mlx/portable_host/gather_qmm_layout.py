"""Storage and launch contracts for MLX's affine gathered quantized products.

``GatherQMM::eval_gpu`` in ``gather_qmm.inc`` routes affine products as
quantized.cpp does on a device without NAX kernels. Products with at least
``matmul_layout.vector_limit`` rows take ``gather_qmm``: ``affine_gather_qmm_t``
for transposed weights and ``affine_gather_qmm_n`` otherwise, both running
``(32, 2, 2)`` threadgroups over 32 by 32 output tiles. Fewer rows take
``gather_qmv``, ``affine_gather_qmv_fast`` when ``N`` and ``K`` are aligned and
``affine_gather_qmv`` otherwise, or, for untransposed weights,
``gather_qvm``'s ``affine_gather_qvm``; these run ``(32, 2, 1)``
threadgroups over a ``(M, tiles, B)`` grid. quantized.cpp sends sorted
single-row batches to ``gather_qmm_rhs``, which the host refuses before any
launch.

Every kernel first runs quantized.h ``adjust_matrix_offsets``: ``tid.z``'s
lhs and rhs indices, read through ``lhs_strides`` and ``rhs_strides`` or, for
several index dimensions, ``elem_to_loc_broadcast`` over ``batch_shape``,
select one activation matrix and one weight, scale and bias matrix, and
``y`` moves to ``tid.z``'s ``M`` by ``N`` slice. ``plan`` replays those reads
from the bound index buffers and proves every offset names elements of its
buffer before a translated kernel runs.
"""

import ctypes
import math
import re

from demos.integrations.mlx.portable_host import matmul_layout, qmm_layout
from demos.integrations.mlx.portable_host.quantization_layout import SOURCE_TYPES

SOURCE = "mlx/backend/metal/kernels/quantized.metal"
LABEL = "Gathered quantized product"
SIMD_SIZE = 32
MAX_ELEMENTS = matmul_layout.MAX_ELEMENTS
MAX_GROUPS = matmul_layout.MAX_GROUPS
MAX_BATCH_DIMENSIONS = matmul_layout.MAX_BATCH_DIMENSIONS
PATTERN = re.compile(
    r"affine_gather_(qmv_fast|qmv|qvm|qmm_t|qmm_n)_(float|float16_t|bfloat16_t)"
    r"_gs_(32|64|128)_b_(2|3|4|5|6|8)(?:_alN_(true|false))?"
)
ENTRY_PREFIXES = (
    "affine_gather_qmv_",
    "affine_gather_qvm_",
    "affine_gather_qmm_t_",
    "affine_gather_qmm_n_",
)
MATRIX_KERNELS = ("qmm_t", "qmm_n")
TRANSPOSED_KERNELS = ("qmv_fast", "qmv", "qmm_t")
INDEX_DTYPE = "uint32"
INDICES = ("lhs_indices", "rhs_indices")
MATRIX_PARAMETERS = ("w", "scales", "biases", "x", *INDICES, "y")
# gather_qmv and gather_qvm bind the vector sizes; gather_qmm K, N and M.
VECTOR_SCALARS = ("in_vec_size", "out_vec_size")
MATRIX_SCALARS = ("K", "N", "M")
GATHER_VECTORS = {
    "batch_shape": "int32",
    "lhs_strides": "int64",
    "rhs_strides": "int64",
}
GATHER_PARAMETERS = ("batch_ndims", *GATHER_VECTORS)
# Expressions OpenGL may evaluate in 32 bits in the matrix-vector kernels:
# adjust_matrix_offsets' index locations and matrix offsets. ``plan`` proves
# each names an element of a buffer of at most MAX_ELEMENTS elements.
GATHER_INDEX_EXPRESSIONS = (
    "tid.z * lhs_strides[0]",
    "tid.z * rhs_strides[0]",
    "idx.x",
    "idx.y",
    "x_idx * x_strides[0]",
    "w_idx * w_strides[0]",
    "w_idx * s_strides[0]",
    "w_idx * b_strides[0]",
    "idx.z",
)
SIGNED_MAX = qmm_layout.SIGNED_MAX
INDEX_RECORD_LIMIT = 256


def signature(entry):
    """The kernel, element type, group size, bits and, for qmm_t, alignment."""
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError("Unsupported gathered quantized entry")
    kernel, source, group, bits, aligned = match.groups()
    # Only affine_gather_qmm_t names whether N is whole tiles.
    if (aligned is None) != (kernel != "qmm_t"):
        raise ValueError("Unsupported gathered quantized entry")
    return {
        "kernel": kernel,
        "dtype": next(key for key, value in SOURCE_TYPES.items() if value == source),
        "groupSize": int(group),
        "bits": int(bits),
        "aligned": None if aligned is None else aligned == "true",
        "transpose": kernel in TRANSPOSED_KERNELS,
    }


def is_matrix(entry):
    return signature(entry)["kernel"] in MATRIX_KERNELS


def workgroup_size(entry):
    return (SIMD_SIZE, 2, 2) if is_matrix(entry) else (SIMD_SIZE, 2, 1)


def tile_columns(entry):
    """Outputs per threadgroup along ``N``.

    qmm's 32-column tiles, qmv's 2 simdgroups of 4 rows, and qvm's 2
    simdgroups of ``min(group_size, 32)`` columns.
    """
    info = signature(entry)
    if info["kernel"] in MATRIX_KERNELS:
        return qmm_layout.BN
    if info["kernel"] == "qvm":
        return 2 * min(info["groupSize"], SIMD_SIZE)
    return 8


def select(dtype, group, bits, m, n, k, transpose=True):
    """quantized.cpp GatherQMM::eval_gpu's entry for an unsorted product.

    The sorted single-row route, ``gather_qmm_rhs``, is decided by the
    primitive's ``right_sorted`` flag; ``gather_qmm.inc`` refuses it.
    """
    stem = f"{SOURCE_TYPES[dtype]}_gs_{group}_b_{bits}"
    if m >= matmul_layout.vector_limit(k, n, transpose):
        if not transpose:
            return f"affine_gather_qmm_n_{stem}"
        aligned = "true" if n % qmm_layout.BN == 0 else "false"
        return f"affine_gather_qmm_t_{stem}_alN_{aligned}"
    if not transpose:
        return f"affine_gather_qvm_{stem}"
    fast = n % 8 == 0 and k % matmul_layout.fast_alignment(bits) == 0
    return f"affine_gather_{'qmv_fast' if fast else 'qmv'}_{stem}"


def scalars(entry):
    return MATRIX_SCALARS if is_matrix(entry) else VECTOR_SCALARS


def parameter_names(entry):
    """quantized.cpp's buffers, in its binding order: every launch binds all."""
    return (
        *MATRIX_PARAMETERS,
        *scalars(entry),
        *matmul_layout.BATCH_PARAMETERS,
        *GATHER_PARAMETERS,
    )


def parameter_type(entry, name):
    if name == "w" or name in INDICES:
        return INDEX_DTYPE
    if name in MATRIX_PARAMETERS:
        return signature(entry)["dtype"]
    if name in GATHER_VECTORS:
        return GATHER_VECTORS[name]
    if name in matmul_layout.BATCH_VECTORS:
        return matmul_layout.BATCH_VECTORS[name]
    if name in (*MATRIX_SCALARS, *VECTOR_SCALARS, "batch_ndims"):
        return "int32"
    if name in matmul_layout.BATCH_SCALARS:
        return "int32"
    raise ValueError(f"{LABEL} parameter is unknown")


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters.

    The returned parameters also hold the index buffers' values, which
    ``plan`` reads and does not return.
    """
    return read_gathered(
        buffers,
        {name: parameter_type(entry, name) for name in parameter_names(entry)},
        scalars(entry),
        MATRIX_PARAMETERS,
    )


def read_gathered(buffers, types, sizes, matrices, label=LABEL):
    """``read_storage`` for gathered kernels binding ``types``.

    ``sizes`` names the kernel's dimension scalars and ``matrices`` its
    matrix buffers, which hold the index buffers; their values are read too.
    """
    counts, parameters = matmul_layout.read_buffers(
        buffers,
        types,
        (*sizes, *matmul_layout.BATCH_SCALARS, "batch_ndims"),
        label,
        matrices,
    )
    if any(len(parameters[name]) > MAX_BATCH_DIMENSIONS for name in GATHER_VECTORS):
        raise ValueError(f"{label} index batch has too many dimensions")
    for name in INDICES:
        buffer = buffers[name]
        parameters[name] = list(
            (ctypes.c_uint32 * buffer.count).from_address(buffer.data)
        )
    return counts, parameters


def _location(index, shape, strides):
    """utils.h elem_to_loc and elem_to_loc_broadcast over ``shape``."""
    location = 0
    for extent, stride in zip(reversed(shape), reversed(strides)):
        location += (index % extent) * stride
        index //= extent
    return location


# The weight views adjust_matrix_offsets moves by ``w_idx``: the affine
# kernels' weights, scales and biases. The block-scaled kernels have no biases.
WEIGHT_VIEWS = ("w_strides", "s_strides", "b_strides")


def _matrix_views(parameters, m, k, rows, words, groups, views, label):
    """Check add_strides_and_shapes' views; return the x and w matrix counts."""
    x_shape, x_strides = parameters["x_shape"], parameters["x_strides"]
    w_shape = parameters["w_shape"]
    x_batch, w_batch = parameters["x_batch_ndims"], parameters["w_batch_ndims"]
    weights = [parameters[name] for name in views]
    if (
        x_batch != len(x_shape) - 2
        or len(x_strides) != len(x_shape)
        or not 0 <= x_batch <= MAX_BATCH_DIMENSIONS
        or w_batch != len(w_shape) - 2
        or not 0 <= w_batch <= MAX_BATCH_DIMENSIONS
        or any(len(strides) != len(w_shape) for strides in weights)
    ):
        raise ValueError(f"{label} batch ranks do not match")
    if x_shape[-2:] != [m, k] or w_shape[-2:] != [rows, words]:
        raise ValueError(f"{label} matrix shapes do not match")
    # ensure_row_contiguous_matrix: unit columns and dense rows. Only the
    # packed weights have ``words`` columns; scales and biases have ``groups``.
    for strides, extent, columns in (
        (x_strides, m, k),
        *(
            (strides, rows, words if name == "w_strides" else groups)
            for name, strides in zip(views, weights)
        ),
    ):
        if strides[-1] != 1 or (extent > 1 and strides[-2] != columns):
            raise ValueError(f"{label} matrices are not row contiguous")
    if any(
        extent < 1 or stride < 0
        for shape, strides, ndim in (
            (x_shape, x_strides, x_batch),
            *((w_shape, strides, w_batch) for strides in weights),
        )
        for extent, stride in zip(shape[:ndim], strides[:ndim])
    ):
        raise ValueError(f"{label} batch strides must be nonnegative")
    return math.prod(x_shape[:x_batch]), math.prod(w_shape[:w_batch])


def gathered_offsets(
    parameters,
    counts,
    m,
    k,
    rows,
    words,
    groups,
    batches,
    views=WEIGHT_VIEWS,
    label=LABEL,
):
    """Per ``tid.z`` matrix indices and x and weight offsets, one per ``views``.

    adjust_matrix_offsets reads ``tid.z``'s indices at ``tid.z * strides[0]``
    for one index dimension and through elem_to_loc_broadcast otherwise,
    then offsets x by ``x_idx * x_strides[0]`` for one batch dimension and
    through elem_to_loc otherwise, and the weights likewise by ``w_idx``.
    Every index must name one of the matrices its operand holds. Also returns
    the weights' matrix count.
    """
    x_matrices, w_matrices = _matrix_views(
        parameters, m, k, rows, words, groups, views, label
    )
    ndims, shape = parameters["batch_ndims"], parameters["batch_shape"]
    lhs_strides, rhs_strides = parameters["lhs_strides"], parameters["rhs_strides"]
    if (
        not 1 <= ndims <= MAX_BATCH_DIMENSIONS
        or not len(shape) == len(lhs_strides) == len(rhs_strides) == ndims
        or any(extent < 1 for extent in shape)
        or any(stride < 0 for stride in (*lhs_strides, *rhs_strides))
        or math.prod(shape) != batches
    ):
        raise ValueError(f"{label} index batch does not match the launch")
    x_shape, x_strides = parameters["x_shape"], parameters["x_strides"]
    w_shape = parameters["w_shape"]
    x_batch, w_batch = parameters["x_batch_ndims"], parameters["w_batch_ndims"]
    w_views = [parameters[name] for name in views]
    lhs, rhs = parameters["lhs_indices"], parameters["rhs_indices"]
    indices, offsets = [], []
    for z in range(batches):
        if ndims == 1:
            lhs_at, rhs_at = z * lhs_strides[0], z * rhs_strides[0]
        else:
            lhs_at = _location(z, shape, lhs_strides)
            rhs_at = _location(z, shape, rhs_strides)
        if lhs_at >= counts["lhs_indices"] or rhs_at >= counts["rhs_indices"]:
            raise ValueError(f"{label} index view exceeds its buffer")
        x_idx, w_idx = lhs[lhs_at], rhs[rhs_at]
        if x_idx >= x_matrices or w_idx >= w_matrices:
            raise ValueError(f"{label} index names no matrix of its operand")
        if x_batch == 1:
            x_offset = x_idx * x_strides[0]
        else:
            x_offset = _location(x_idx, x_shape[:x_batch], x_strides[:x_batch])
        if w_batch == 1:
            w_offsets = [w_idx * view[0] for view in w_views]
        else:
            w_offsets = [
                _location(w_idx, w_shape[:w_batch], view[:w_batch]) for view in w_views
            ]
        indices.append((x_idx, w_idx))
        offsets.append((x_offset, *w_offsets))
    return indices, offsets, w_matrices


def grid_batches(execution, size, matrix, m, n, tile, label=LABEL):
    """The batch count of a launch whose grid and threadgroups match MLX's.

    gather_qmm runs ``(ceil(N / 32), ceil(M / 32), B)`` threadgroups of
    ``size``; gather_qmv and gather_qvm run ``(M, ceil(N / tile), B)``.
    """
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError(f"{label} launch geometry is missing")
    count = list(execution["workgroupCount"])
    tiles = -(-n // tile)
    expected = [tiles, -(-m // qmm_layout.BM)] if matrix else [m, tiles]
    if (
        list(execution["workgroupSize"]) != list(size)
        or len(count) != 3
        or any(type(value) is not int for value in count)
        or count[:2] != expected
        or not 1 <= min(count) <= max(count) <= MAX_GROUPS
    ):
        raise ValueError(f"{label} launch does not match MLX's grid")
    return count[2]


def launch_shape(entry, execution, m, n):
    """The batch count of a launch, after checking MLX's grid for ``entry``."""
    return grid_batches(
        execution, workgroup_size(entry), is_matrix(entry), m, n, tile_columns(entry)
    )


def plan(entry, counts, parameters, threads, execution):
    """Validate one launch against MLX's routing and return its metadata.

    The returned ``parameters`` are the bound integer parameters without the
    index values; ``matrixIndices`` lists each ``tid.z``'s ``(x, w)`` matrices
    for launches of at most ``INDEX_RECORD_LIMIT`` batches.
    """
    info = signature(entry)
    group, bits, transpose = info["groupSize"], info["bits"], info["transpose"]
    if is_matrix(entry):
        k, n, m = (parameters.get(name) for name in MATRIX_SCALARS)
    else:
        k, n = (parameters.get(name) for name in VECTOR_SCALARS)
        count = execution.get("workgroupCount") if isinstance(execution, dict) else None
        m = count[0] if isinstance(count, (list, tuple)) and count else None
    if (
        any(type(value) is not int for value in (k, n, m))
        or not 1 <= min(k, n, m) <= max(k, n, m) <= MAX_ELEMENTS
        or (k if transpose else n) % group
    ):
        raise ValueError(f"{LABEL} dimensions are invalid")
    batches = launch_shape(entry, execution, m, n)
    if select(info["dtype"], group, bits, m, n, k, transpose) != entry:
        raise ValueError(f"{LABEL} entry differs from MLX's routing")
    if threads != m * n * batches or counts["y"] != threads:
        raise ValueError(f"{LABEL} output extent differs")
    # Groups run along the weights' rows: K when transposed, N otherwise.
    rows, columns = (n, k) if transpose else (k, n)
    words, groups = columns * bits // 32, columns // group
    indices, offsets, _ = gathered_offsets(
        parameters, counts, m, k, rows, words, groups, batches
    )
    extents = {
        "x": m * k,
        "w": rows * words,
        "scales": rows * groups,
        "biases": rows * groups,
    }
    for column, name in enumerate(("x", "w", "scales", "biases")):
        if max(offset[column] for offset in offsets) + extents[name] > counts[name]:
            raise ValueError(f"{LABEL} {name} matrix exceeds its buffer")
    values = {name: value for name, value in parameters.items() if name not in INDICES}
    return {
        "kernel": info["kernel"],
        "M": m,
        "N": n,
        "K": k,
        "batches": batches,
        # The trace keeps the gathered matrices of launches of few batches.
        "matrixIndices": (
            [list(pair) for pair in indices] if batches <= INDEX_RECORD_LIMIT else None
        ),
        "overreadPadding": {"scales": 0, "biases": 0},
        "parameters": values,
    }


def sibling(entry):
    """The batched qmm entry whose tiles a gathered qmm entry shares."""
    info = signature(entry)
    stem = f"{SOURCE_TYPES[info['dtype']]}_gs_{info['groupSize']}_b_{info['bits']}"
    if info["kernel"] == "qmm_t":
        return f"affine_qmm_t_{stem}_alN_true_batch_1"
    if info["kernel"] == "qmm_n":
        return f"affine_qmm_n_{stem}_batch_1"
    raise ValueError("Gathered matrix-vector entries share no qmm tiles")


def access_assertions(entry):
    """The qmm x loader's ``dst`` slots; matrix-vector kernels have none."""
    if not is_matrix(entry):
        return ()
    return tuple(
        {**assertion, "entryPoint": entry}
        for assertion in qmm_layout.access_assertions(sibling(entry))
    )


def index_assertions(entry):
    """OpenGL ranges for the 64-bit offsets each kernel forms.

    Matrix-vector kernels bound adjust_matrix_offsets' expressions by
    ``MAX_ELEMENTS``. qmm kernels take their batched siblings' resource and
    tile ranges, and each lhs or rhs index element ``crosstl_resource_load_uint``
    reads must fit a 32-bit index.
    """
    if not is_matrix(entry):
        return tuple(
            {
                "source": SOURCE,
                "expression": expression,
                "minimum": 0,
                "maximum": MAX_ELEMENTS,
            }
            for expression in GATHER_INDEX_EXPRESSIONS
        )
    items = list(qmm_layout.index_assertions(sibling(entry)))
    tiles = [item for item in items if "indexed" in item]
    items = [item for item in items if "indexed" not in item]
    items.append(
        {
            "source": SOURCE,
            "expression": "reference.offset + index",
            "function": "crosstl_resource_load_uint",
            "minimum": 0,
            "maximum": SIGNED_MAX,
        }
    )
    return tuple(items + tiles)


def bounded_parameters(entry):
    """The scalar parameters each qmm translation bounds by ``K_LIMIT``."""
    return ("K",) if is_matrix(entry) else ()
