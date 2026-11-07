"""Storage and launch contracts for MLX's gathered quantized matrix-vector kernels.

``GatherQQMM::eval_gpu`` in ``gather_qmm.inc`` quantizes both operands as
quantized.cpp does and sends every product that is not a globally scaled
bfloat16 matrix product to ``gather_qmv``: fp_quantized.metal's
``fp_gather_qmv_fast`` when ``N`` and ``K`` are aligned and ``fp_gather_qmv``
otherwise, their ``_hgs`` instantiations when nvfp4 weights carry global
scales. Every launch runs ``(32, 2, 1)`` threadgroups over an
``(M, ceil(N / 8), B)`` grid: two simdgroups of four output rows for each
input vector and gathered batch. Its launches carry a ``route`` of
``gather_qqmm``.

The kernels read one uint32 index per batch from each index buffer, move the
input and the weights to the gathered matrices as fp_quantized.h
adjust_matrix_offsets does, and an ``_hgs`` kernel reads the global scale of
the gathered weights. ``plan`` reads those indices and proves every gathered
operand, and every offset OpenGL evaluates in 32 bits, lies inside its buffer.
"""

import ctypes
import math
import re

from demos.integrations.mlx.portable_host import fp_matmul_layout, matmul_layout
from demos.integrations.mlx.portable_host.matmul_layout import (
    MAX_BATCH_DIMENSIONS,
    MAX_ELEMENTS,
    MAX_GROUPS,
    SIMD_SIZE,
    fast_alignment,
)
from demos.integrations.mlx.portable_host.quantization_layout import SOURCE_TYPES

SOURCE = fp_matmul_layout.SOURCE
MODES = fp_matmul_layout.MODES
LABEL = "Gathered quantized matrix-vector"
PATTERN = re.compile(
    r"(mxfp4|mxfp8|nvfp4)_gather_(qmv_fast|qmv)_(float|float16_t|bfloat16_t)"
    r"_gs_(16|32)_b_(4|8)(_hgs)?"
)
ENTRY_PREFIXES = tuple(f"{mode}_gather_qmv_" for mode in MODES)
# The primitives whose routing a launch follows; every launch names one.
ROUTES = ("gather_qqmm",)
# gather_qmv's bn: two simdgroups of four rows.
TILE_ROWS = 8
WORKGROUP_SIZE = (SIMD_SIZE, 2, 1)
INDEX_PARAMETERS = ("lhs_indices", "rhs_indices")
MATRIX_PARAMETERS = ("w", "scales", "x", *INDEX_PARAMETERS, "y")
SCALAR_PARAMETERS = (
    "in_vec_size",
    "out_vec_size",
    "x_batch_ndims",
    "w_batch_ndims",
    "batch_ndims",
)
VECTOR_TYPES = {
    "x_shape": "int32",
    "x_strides": "int64",
    "w_shape": "int32",
    "w_strides": "int64",
    "s_strides": "int64",
    "batch_shape": "int32",
    "lhs_strides": "int64",
    "rhs_strides": "int64",
}
# The value of the global scale the non-_hgs kernels declare and never read.
INACTIVE_GLOBAL_SCALE = fp_matmul_layout.INACTIVE_GLOBAL_SCALE
# Expressions OpenGL may evaluate in 32 bits: adjust_matrix_offsets' index
# locations and batch offsets. ``plan`` proves each names an element of a
# bound buffer, and no bound buffer holds more than MAX_ELEMENTS elements.
INDEX_EXPRESSIONS = (
    "tid.z * lhs_strides[0]",
    "tid.z * rhs_strides[0]",
    "x_idx * x_strides[0]",
    "w_idx * w_strides[0]",
    "w_idx * s_strides[0]",
    "idx.x",
    "idx.y",
)


def signature(entry):
    """The mode, kernel, element type, group size, bits and global scale."""
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError(f"Unsupported {LABEL.lower()} entry")
    mode, kernel, source, group, bits, scaled = match.groups()
    group, bits = int(group), int(bits)
    # Only nvfp4 products carry global scales.
    if MODES[mode] != (group, bits) or (scaled and mode != "nvfp4"):
        raise ValueError(f"{LABEL} entry is never routed")
    return {
        "mode": mode,
        "kernel": kernel,
        "dtype": next(key for key, value in SOURCE_TYPES.items() if value == source),
        "groupSize": group,
        "bits": bits,
        "globalScale": scaled is not None,
    }


def workgroup_size(entry):
    signature(entry)
    return WORKGROUP_SIZE


def select(mode, dtype, n, k, global_scale=False):
    """quantized.cpp gather_qmv's kernel for one gathered product."""
    group, bits = MODES[mode]
    if global_scale and mode != "nvfp4":
        raise ValueError(f"{LABEL} global scales come only with nvfp4 weights")
    fast = n % TILE_ROWS == 0 and k % fast_alignment(bits) == 0
    kernel = "qmv_fast" if fast else "qmv"
    scaled = "_hgs" if global_scale else ""
    return f"{mode}_gather_{kernel}_{SOURCE_TYPES[dtype]}_gs_{group}_b_{bits}{scaled}"


def matrix_parameters(entry):
    """The buffers the host binds by address: the global scales only for _hgs."""
    if signature(entry)["globalScale"]:
        return (*MATRIX_PARAMETERS, "global_scale")
    return MATRIX_PARAMETERS


def parameter_names(entry):
    return (*matrix_parameters(entry), *SCALAR_PARAMETERS, *VECTOR_TYPES)


def parameter_type(entry, name):
    if name in ("w", *INDEX_PARAMETERS):
        return "uint32"
    if name == "scales":
        return "uint8"
    if name == "global_scale":
        return "float32"
    if name in ("x", "y"):
        return signature(entry)["dtype"]
    if name in SCALAR_PARAMETERS:
        return "int32"
    if name in VECTOR_TYPES:
        return VECTOR_TYPES[name]
    raise ValueError(f"{LABEL} parameter is unknown")


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters.

    Returns ``(counts, parameters)``. The parameters hold zero-copy views of
    the two index buffers under their names, which ``plan`` reads and drops.
    """
    names = parameter_names(entry)
    # read_buffers bounds the operands' batch views; the index views have at
    # most MAX_BATCH_DIMENSIONS collapsed axes.
    for name in ("batch_shape", "lhs_strides", "rhs_strides"):
        buffer = buffers.get(name)
        if buffer is not None and not (
            type(buffer.count) is int and buffer.count <= MAX_BATCH_DIMENSIONS
        ):
            raise ValueError(f"{LABEL} index views exceed their bound")
    counts, parameters = matmul_layout.read_buffers(
        buffers,
        {name: parameter_type(entry, name) for name in names},
        SCALAR_PARAMETERS,
        LABEL,
        matrices=matrix_parameters(entry),
    )
    for name in INDEX_PARAMETERS:
        buffer = buffers[name]
        parameters[name] = (ctypes.c_uint32 * buffer.count).from_address(buffer.data)
    return counts, parameters


def _location(index, shape, strides):
    """utils.h elem_to_loc over the leading ``len(shape)`` axes."""
    location = 0
    for extent, stride in zip(reversed(shape), reversed(strides)):
        location += (index % extent) * stride
        index //= extent
    return location


def _operand_offsets(index, ndim, shape, *strides):
    """adjust_matrix_offsets' move to batch ``index`` of one operand.

    One batch axis scales its first strides by the index; more go through
    elem_to_loc and elem_to_loc_broadcast, and none leaves the operand at 0.
    """
    if ndim == 1:
        return tuple(index * values[0] for values in strides)
    return tuple(_location(index, shape[:ndim], values[:ndim]) for values in strides)


def _check_views(parameters, k, n, words, groups):
    """The batch views add_strides_and_shapes binds: rank, shape and contiguity."""
    x_ndim, x_shape, x_strides = (
        parameters["x_batch_ndims"],
        parameters["x_shape"],
        parameters["x_strides"],
    )
    w_ndim, w_shape = parameters["w_batch_ndims"], parameters["w_shape"]
    w_strides, s_strides = parameters["w_strides"], parameters["s_strides"]
    if (
        not 0 <= x_ndim <= MAX_BATCH_DIMENSIONS
        or not 0 <= w_ndim <= MAX_BATCH_DIMENSIONS
        or not len(x_shape) == len(x_strides) == x_ndim + 2
        or not len(w_shape) == len(w_strides) == len(s_strides) == w_ndim + 2
        or any(extent < 1 for extent in (*x_shape, *w_shape))
        or any(stride < 0 for stride in (*x_strides, *w_strides, *s_strides))
    ):
        raise ValueError(f"{LABEL} batch views are invalid")
    if x_shape[-1] != k or w_shape[-2:] != [n, words]:
        raise ValueError(f"{LABEL} matrix shapes do not match")
    # ensure_row_contiguous_matrix: unit columns and dense rows.
    for strides, extent, columns in (
        (x_strides, x_shape[-2], k),
        (w_strides, n, words),
        (s_strides, n, groups),
    ):
        if strides[-1] != 1 or (extent > 1 and strides[-2] != columns):
            raise ValueError(f"{LABEL} matrices are not row contiguous")


def _check_indices(parameters, batches):
    """add_gather_strides_and_shapes' collapsed index views.

    Scalar indices bind a one-element shape and zero strides the kernels
    never read, with ``batch_ndims`` 0.
    """
    ndim = parameters["batch_ndims"]
    shape = parameters["batch_shape"]
    lhs, rhs = parameters["lhs_strides"], parameters["rhs_strides"]
    if not 0 <= ndim <= MAX_BATCH_DIMENSIONS:
        raise ValueError(f"{LABEL} index views are invalid")
    if ndim == 0:
        if shape != [1] or lhs != [0] or rhs != [0] or batches != 1:
            raise ValueError(f"{LABEL} scalar index views are invalid")
        return
    if (
        not len(shape) == len(lhs) == len(rhs) == ndim
        or any(extent < 1 for extent in shape)
        or any(stride < 0 for stride in (*lhs, *rhs))
        or math.prod(shape) != batches
    ):
        raise ValueError(f"{LABEL} index views are invalid")


def _index_locations(z, parameters):
    """Where batch ``z`` reads its two indices."""
    ndim = parameters["batch_ndims"]
    lhs, rhs = parameters["lhs_strides"], parameters["rhs_strides"]
    if ndim == 1:
        return z * lhs[0], z * rhs[0]
    shape = parameters["batch_shape"][:ndim]
    return _location(z, shape, lhs[:ndim]), _location(z, shape, rhs[:ndim])


def _grid(entry, n, execution):
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError(f"{LABEL} launch geometry is missing")
    m, tiles, batches = execution["workgroupCount"]
    if (
        list(execution["workgroupSize"]) != list(workgroup_size(entry))
        or any(type(value) is not int for value in (m, tiles, batches))
        or not 1 <= m <= MAX_GROUPS
        or not 1 <= batches <= MAX_GROUPS
        or tiles != -(-n // TILE_ROWS)
        or tiles > MAX_GROUPS
    ):
        raise ValueError(f"{LABEL} launch does not match MLX's grid")
    return m, batches


def plan(entry, counts, parameters, threads, execution, route=None):
    """Validate one launch against MLX's routing and return its metadata.

    Every batch's two indices are read from their buffers. The gathered input
    must hold ``M`` rows of ``K`` values, the gathered weights and scales
    ``N`` rows, an ``_hgs`` launch one global scale per weight batch, and the
    output every row the kernels write. The returned ``parameters`` drop the
    index views and add the inert global scale of the non-_hgs kernels.
    """
    info = signature(entry)
    if route not in ROUTES:
        raise ValueError(f"{LABEL} route is unknown")
    parameters = dict(parameters)
    indices = {name: parameters.pop(name, None) for name in INDEX_PARAMETERS}
    if set(parameters) != {*SCALAR_PARAMETERS, *VECTOR_TYPES} or any(
        view is None for view in indices.values()
    ):
        raise ValueError(f"{LABEL} parameters are missing")
    k, n = parameters["in_vec_size"], parameters["out_vec_size"]
    group, bits = info["groupSize"], info["bits"]
    if (
        type(k) is not int
        or type(n) is not int
        or not 1 <= min(k, n) <= max(k, n) <= MAX_ELEMENTS
        or k % group
    ):
        raise ValueError(f"{LABEL} dimensions are invalid")
    m, batches = _grid(entry, n, execution)
    if select(info["mode"], info["dtype"], n, k, info["globalScale"]) != entry:
        raise ValueError(f"{LABEL} entry differs from MLX's routing")
    if threads != m * n * batches or counts["y"] != threads:
        raise ValueError(f"{LABEL} output extent differs")
    words, groups = k * bits // 32, k // group
    _check_views(parameters, k, n, words, groups)
    _check_indices(parameters, batches)
    x_ndim, x_shape = parameters["x_batch_ndims"], parameters["x_shape"]
    w_ndim, w_shape = parameters["w_batch_ndims"], parameters["w_shape"]
    # The kernels step the output by out_vec_size times x's rows per batch.
    rows = x_shape[x_ndim]
    if batches > 1 and rows != m:
        raise ValueError(f"{LABEL} input rows differ from the grid")
    x_batches = math.prod(x_shape[:x_ndim])
    w_batches = math.prod(w_shape[:w_ndim])
    gathered = {"lhs_indices": [], "rhs_indices": []}
    for z in range(batches):
        for name, location in zip(INDEX_PARAMETERS, _index_locations(z, parameters)):
            if location >= counts[name]:
                raise ValueError(f"{LABEL} {name} read exceeds its buffer")
            gathered[name].append(indices[name][location])
    for x_index in sorted(set(gathered["lhs_indices"])):
        if x_index >= x_batches:
            raise ValueError(f"{LABEL} input index is out of range")
        (offset,) = _operand_offsets(x_index, x_ndim, x_shape, parameters["x_strides"])
        if offset + m * k > counts["x"]:
            raise ValueError(f"{LABEL} input batch exceeds its buffer")
    for w_index in sorted(set(gathered["rhs_indices"])):
        if w_index >= w_batches:
            raise ValueError(f"{LABEL} weight index is out of range")
        w_offset, s_offset = _operand_offsets(
            w_index,
            w_ndim,
            w_shape,
            parameters["w_strides"],
            parameters["s_strides"],
        )
        if (
            w_offset + n * words > counts["w"]
            or s_offset + n * groups > counts["scales"]
        ):
            raise ValueError(f"{LABEL} weight batch exceeds its buffer")
        if info["globalScale"] and w_index >= counts["global_scale"]:
            raise ValueError(f"{LABEL} global scale read exceeds its buffer")
    if (batches - 1) * rows * n + m * n > counts["y"]:
        raise ValueError(f"{LABEL} output batch exceeds its buffer")
    values = dict(parameters)
    if not info["globalScale"]:
        values["global_scale"] = [INACTIVE_GLOBAL_SCALE]
    return {
        "mode": info["mode"],
        "kernel": info["kernel"],
        "route": route,
        "M": m,
        "N": n,
        "K": k,
        "batches": batches,
        "lhsIndices": gathered["lhs_indices"],
        "rhsIndices": gathered["rhs_indices"],
        "overreadPadding": {},
        "parameters": values,
    }
