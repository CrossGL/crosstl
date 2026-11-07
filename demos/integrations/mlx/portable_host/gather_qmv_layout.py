"""Storage and launch contracts for MLX's gathered quantized matrix-vector kernels.

Two primitives in ``gather_qmm.inc`` reach quantized.cpp's ``gather_qmv`` and
``gather_qvm``. ``GatherQQMM::eval_gpu`` quantizes both operands and sends
every product that is not a globally scaled bfloat16 matrix product to
``gather_qmv``: fp_quantized.metal's ``fp_gather_qmv_fast`` when ``N`` and
``K`` are aligned and ``fp_gather_qmv`` otherwise, their ``_hgs``
instantiations when nvfp4 weights carry global scales. ``GatherQMM::eval_gpu``
sends products of fewer than ``matmul_layout.vector_limit`` rows there too,
for affine weights (quantized.metal's ``affine_gather_qmv_fast`` and
``affine_gather_qmv``, which bind biases) and for mxfp4, mxfp8 and nvfp4 ones,
and sends untransposed weights to ``gather_qvm``. Every launch runs
``(32, 2, 1)`` threadgroups over an ``(M, ceil(N / bn), B)`` grid: ``bn`` is
gather_qmv's two simdgroups of four rows, or gather_qvm's two simdgroups of
``min(group_size, 32)`` columns. A ``route`` buffer names the primitive whose
routing the launch follows.

The kernels read one uint32 index per batch from each index buffer, move the
input and the weights to the gathered matrices as adjust_matrix_offsets
does, and an ``_hgs`` kernel reads the global scale of the gathered weights.
``plan`` reads those indices and proves every gathered operand, and every
offset OpenGL evaluates in 32 bits, lies inside its buffer.
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
AFFINE_SOURCE = "mlx/backend/metal/kernels/quantized.metal"
MODES = fp_matmul_layout.MODES
# quantized.metal's affine group sizes and bit widths.
AFFINE_GROUPS = (32, 64, 128)
AFFINE_BITS = (2, 3, 4, 5, 6, 8)
LABEL = "Gathered quantized matrix-vector"
PATTERN = re.compile(
    r"(affine|mxfp4|mxfp8|nvfp4)_gather_(qmv_fast|qmv|qvm)"
    r"_(float|float16_t|bfloat16_t)_gs_(16|32|64|128)_b_(2|3|4|5|6|8)(_hgs)?"
)
ENTRY_PREFIXES = tuple(
    f"{mode}_gather_{kernel}_"
    for mode in ("affine", *MODES)
    for kernel in ("qmv", "qvm")
)
# The primitives whose routing a launch follows; every launch names one.
ROUTES = ("gather_qqmm", "gather_qmm")
# gather_qmv's bn: two simdgroups of four rows.
TILE_ROWS = 8
WORKGROUP_SIZE = (SIMD_SIZE, 2, 1)
INDEX_PARAMETERS = ("lhs_indices", "rhs_indices")
MATRIX_PARAMETERS = ("w", "scales", "x", *INDEX_PARAMETERS, "y")
AFFINE_MATRIX_PARAMETERS = ("w", "scales", "biases", "x", *INDEX_PARAMETERS, "y")
SCALAR_PARAMETERS = (
    "in_vec_size",
    "out_vec_size",
    "x_batch_ndims",
    "w_batch_ndims",
    "batch_ndims",
)
# The block-scaled kernels' views; the affine kernels add the bias strides.
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
AFFINE_VECTOR_TYPES = {**VECTOR_TYPES, "b_strides": "int64"}
INDEX_VIEWS = ("batch_shape", "lhs_strides", "rhs_strides")
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
# The affine adjust_matrix_offsets also moves the biases.
AFFINE_INDEX_EXPRESSIONS = (*INDEX_EXPRESSIONS, "w_idx * b_strides[0]", "idx.z")


def check_mode(mode, group, bits, scaled, label=LABEL):
    """Refuse a mode, group size and bit width quantized.cpp never routes.

    Only nvfp4 products carry global scales; affine ones never do.
    """
    if mode == "affine":
        if group not in AFFINE_GROUPS or bits not in AFFINE_BITS or scaled:
            raise ValueError(f"{label} entry is never routed")
    elif MODES[mode] != (group, bits) or (scaled and mode != "nvfp4"):
        raise ValueError(f"{label} entry is never routed")


def signature(entry):
    """The mode, kernel, element type, group size, bits and global scale."""
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError(f"Unsupported {LABEL.lower()} entry")
    mode, kernel, source, group, bits, scaled = match.groups()
    group, bits = int(group), int(bits)
    check_mode(mode, group, bits, scaled is not None)
    return {
        "mode": mode,
        "kernel": kernel,
        "dtype": next(key for key, value in SOURCE_TYPES.items() if value == source),
        "groupSize": group,
        "bits": bits,
        "globalScale": scaled is not None,
    }


def source(entry):
    """The pinned source that instantiates ``entry``."""
    return AFFINE_SOURCE if signature(entry)["mode"] == "affine" else SOURCE


def index_expressions_for(mode):
    """The offsets OpenGL asserts for a mode's adjust_matrix_offsets."""
    return AFFINE_INDEX_EXPRESSIONS if mode == "affine" else INDEX_EXPRESSIONS


def index_expressions(entry):
    return index_expressions_for(signature(entry)["mode"])


def workgroup_size(entry):
    signature(entry)
    return WORKGROUP_SIZE


def transposed(entry):
    """Whether the entry multiplies by transposed weights, ``w`` of ``(N, K)``."""
    return signature(entry)["kernel"] != "qvm"


def tile_rows(entry):
    """Outputs per threadgroup: gather_qmv's 8, gather_qvm's 2 x min(gs, 32)."""
    info = signature(entry)
    if info["kernel"] == "qvm":
        return 2 * min(info["groupSize"], SIMD_SIZE)
    return TILE_ROWS


def select(
    mode, dtype, n, k, global_scale=False, transpose=True, group=None, bits=None
):
    """quantized.cpp gather_qmv's or gather_qvm's kernel for one product.

    The block-scaled modes fix their group size and bits; affine products
    name theirs.
    """
    if mode == "affine":
        if group is None or bits is None:
            raise ValueError(f"{LABEL} affine products name their group and bits")
    else:
        group, bits = (group, bits) if group is not None else MODES[mode]
    if global_scale and mode != "nvfp4":
        raise ValueError(f"{LABEL} global scales come only with nvfp4 weights")
    check_mode(mode, group, bits, global_scale)
    if not transpose:
        kernel = "qvm"
    else:
        fast = n % TILE_ROWS == 0 and k % fast_alignment(bits) == 0
        kernel = "qmv_fast" if fast else "qmv"
    scaled = "_hgs" if global_scale else ""
    return f"{mode}_gather_{kernel}_{SOURCE_TYPES[dtype]}_gs_{group}_b_{bits}{scaled}"


def matrix_parameters(entry):
    """The buffers the host binds by address.

    Affine kernels bind biases; the block-scaled ones bind global scales
    only for _hgs.
    """
    info = signature(entry)
    if info["mode"] == "affine":
        return AFFINE_MATRIX_PARAMETERS
    if info["globalScale"]:
        return (*MATRIX_PARAMETERS, "global_scale")
    return MATRIX_PARAMETERS


def vector_types(entry):
    return AFFINE_VECTOR_TYPES if signature(entry)["mode"] == "affine" else VECTOR_TYPES


def parameter_names(entry):
    return (*matrix_parameters(entry), *SCALAR_PARAMETERS, *vector_types(entry))


def parameter_type(entry, name):
    info = signature(entry)
    if name in ("w", *INDEX_PARAMETERS):
        return "uint32"
    if name == "scales":
        return info["dtype"] if info["mode"] == "affine" else "uint8"
    if name == "global_scale":
        return "float32"
    if name in ("x", "y", "biases"):
        return info["dtype"]
    if name in SCALAR_PARAMETERS:
        return "int32"
    if name in vector_types(entry):
        return vector_types(entry)[name]
    raise ValueError(f"{LABEL} parameter is unknown")


def read_gathered(buffers, types, scalars, matrices, label):
    """``matmul_layout.read_buffers`` plus zero-copy views of both index buffers.

    The index views have at most MAX_BATCH_DIMENSIONS collapsed axes. The
    returned parameters hold the index views under their names, which
    ``gathered_operands`` reads.
    """
    for name in INDEX_VIEWS:
        buffer = buffers.get(name)
        if buffer is not None and not (
            type(buffer.count) is int and buffer.count <= MAX_BATCH_DIMENSIONS
        ):
            raise ValueError(f"{label} index views exceed their bound")
    counts, parameters = matmul_layout.read_buffers(
        buffers, types, scalars, label, matrices=matrices
    )
    for name in INDEX_PARAMETERS:
        buffer = buffers[name]
        parameters[name] = (ctypes.c_uint32 * buffer.count).from_address(buffer.data)
    return counts, parameters


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters.

    Returns ``(counts, parameters)``. The parameters hold zero-copy views of
    the two index buffers under their names, which ``plan`` reads and drops.
    """
    return read_gathered(
        buffers,
        {name: parameter_type(entry, name) for name in parameter_names(entry)},
        SCALAR_PARAMETERS,
        matrix_parameters(entry),
        LABEL,
    )


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


def _check_views(parameters, k, rows, words, groups, label):
    """The batch views add_strides_and_shapes binds: rank, shape and contiguity.

    ``rows`` is the weight matrices' row count: ``N`` for transposed weights,
    ``K`` otherwise. Without ``b_strides`` the biases are not bound.
    """
    x_ndim, x_shape, x_strides = (
        parameters["x_batch_ndims"],
        parameters["x_shape"],
        parameters["x_strides"],
    )
    w_ndim, w_shape = parameters["w_batch_ndims"], parameters["w_shape"]
    w_strides, s_strides = parameters["w_strides"], parameters["s_strides"]
    weight_views = [w_strides, s_strides]
    if "b_strides" in parameters:
        weight_views.append(parameters["b_strides"])
    if (
        not 0 <= x_ndim <= MAX_BATCH_DIMENSIONS
        or not 0 <= w_ndim <= MAX_BATCH_DIMENSIONS
        or not len(x_shape) == len(x_strides) == x_ndim + 2
        or len(w_shape) != w_ndim + 2
        or any(len(strides) != w_ndim + 2 for strides in weight_views)
        or any(extent < 1 for extent in (*x_shape, *w_shape))
        or any(stride < 0 for view in (x_strides, *weight_views) for stride in view)
    ):
        raise ValueError(f"{label} batch views are invalid")
    if x_shape[-1] != k or w_shape[-2:] != [rows, words]:
        raise ValueError(f"{label} matrix shapes do not match")
    # ensure_row_contiguous_matrix: unit columns and dense rows.
    views = [(x_strides, x_shape[-2], k), (w_strides, rows, words)]
    views += [(strides, rows, groups) for strides in weight_views[1:]]
    for strides, extent, columns in views:
        if strides[-1] != 1 or (extent > 1 and strides[-2] != columns):
            raise ValueError(f"{label} matrices are not row contiguous")


def _check_indices(parameters, batches, label):
    """add_gather_strides_and_shapes' collapsed index views.

    Scalar indices bind a one-element shape and zero strides the kernels
    never read, with ``batch_ndims`` 0.
    """
    ndim = parameters["batch_ndims"]
    shape = parameters["batch_shape"]
    lhs, rhs = parameters["lhs_strides"], parameters["rhs_strides"]
    if not 0 <= ndim <= MAX_BATCH_DIMENSIONS:
        raise ValueError(f"{label} index views are invalid")
    if ndim == 0:
        if shape != [1] or lhs != [0] or rhs != [0] or batches != 1:
            raise ValueError(f"{label} scalar index views are invalid")
        return
    if (
        not len(shape) == len(lhs) == len(rhs) == ndim
        or any(extent < 1 for extent in shape)
        or any(stride < 0 for stride in (*lhs, *rhs))
        or math.prod(shape) != batches
    ):
        raise ValueError(f"{label} index views are invalid")


def _index_locations(z, parameters):
    """Where batch ``z`` reads its two indices."""
    ndim = parameters["batch_ndims"]
    lhs, rhs = parameters["lhs_strides"], parameters["rhs_strides"]
    if ndim == 1:
        return z * lhs[0], z * rhs[0]
    shape = parameters["batch_shape"][:ndim]
    return _location(z, shape, lhs[:ndim]), _location(z, shape, rhs[:ndim])


def gathered_operands(info, counts, parameters, indices, shapes, label):
    """Read every batch's two indices and bound the operands they gather.

    ``shapes`` holds ``M``, ``K``, the weight rows, words and groups and the
    batch count. The gathered input must hold ``M`` rows of ``K`` values,
    the gathered weights, scales and biases their rows, an ``_hgs`` launch
    one global scale per weight matrix, and the output every batch the
    kernels write. Returns the gathered lhs and rhs indices.
    """
    m, k, rows, words, groups, batches = shapes
    _check_views(parameters, k, rows, words, groups, label)
    _check_indices(parameters, batches, label)
    x_ndim, x_shape = parameters["x_batch_ndims"], parameters["x_shape"]
    w_ndim, w_shape = parameters["w_batch_ndims"], parameters["w_shape"]
    # The kernels step the output by M rows of N outputs per batch.
    if batches > 1 and x_shape[x_ndim] != m:
        raise ValueError(f"{label} input rows differ from the grid")
    x_batches = math.prod(x_shape[:x_ndim])
    w_batches = math.prod(w_shape[:w_ndim])
    gathered = {"lhs_indices": [], "rhs_indices": []}
    for z in range(batches):
        for name, location in zip(INDEX_PARAMETERS, _index_locations(z, parameters)):
            if location >= counts[name]:
                raise ValueError(f"{label} {name} read exceeds its buffer")
            gathered[name].append(indices[name][location])
    for x_index in sorted(set(gathered["lhs_indices"])):
        if x_index >= x_batches:
            raise ValueError(f"{label} input index is out of range")
        (offset,) = _operand_offsets(x_index, x_ndim, x_shape, parameters["x_strides"])
        if offset + m * k > counts["x"]:
            raise ValueError(f"{label} input batch exceeds its buffer")
    weights = [("w", "w_strides", words), ("scales", "s_strides", groups)]
    if "b_strides" in parameters:
        weights.append(("biases", "b_strides", groups))
    for w_index in sorted(set(gathered["rhs_indices"])):
        if w_index >= w_batches:
            raise ValueError(f"{label} weight index is out of range")
        offsets = _operand_offsets(
            w_index, w_ndim, w_shape, *(parameters[view] for _, view, _ in weights)
        )
        for (name, _, columns), offset in zip(weights, offsets):
            if offset + rows * columns > counts[name]:
                raise ValueError(f"{label} weight batch exceeds its buffer")
        if info["globalScale"] and w_index >= counts["global_scale"]:
            raise ValueError(f"{label} global scale read exceeds its buffer")
    return gathered["lhs_indices"], gathered["rhs_indices"]


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
        or tiles != -(-n // tile_rows(entry))
        or tiles > MAX_GROUPS
    ):
        raise ValueError(f"{LABEL} launch does not match MLX's grid")
    return m, batches


def _check_route(info, route, m, n, k):
    """GatherQQMM sends only transposed block-scaled products here, GatherQMM
    only products of fewer rows than its vector limit."""
    if route not in ROUTES:
        raise ValueError(f"{LABEL} route is unknown")
    if route == "gather_qqmm" and (info["mode"] == "affine" or info["kernel"] == "qvm"):
        raise ValueError(f"{LABEL} entry differs from GatherQQMM's routing")
    transpose = info["kernel"] != "qvm"
    if route == "gather_qmm" and m >= matmul_layout.vector_limit(k, n, transpose):
        raise ValueError(f"{LABEL} matrix products take the gather_qmm route")


def plan(entry, counts, parameters, threads, execution, route=None):
    """Validate one launch against MLX's routing and return its metadata.

    Every batch's two indices are read from their buffers and the operands
    they gather bounded (``gathered_operands``). The returned ``parameters``
    drop the index views and add the inert global scale of the block-scaled
    kernels without _hgs.
    """
    info = signature(entry)
    parameters = dict(parameters)
    indices = {name: parameters.pop(name, None) for name in INDEX_PARAMETERS}
    if set(parameters) != {*SCALAR_PARAMETERS, *vector_types(entry)} or any(
        view is None for view in indices.values()
    ):
        raise ValueError(f"{LABEL} parameters are missing")
    k, n = parameters["in_vec_size"], parameters["out_vec_size"]
    group, bits = info["groupSize"], info["bits"]
    transpose = info["kernel"] != "qvm"
    # Groups run along the weights' rows: K when transposed, N otherwise.
    grouped = k if transpose else n
    if (
        type(k) is not int
        or type(n) is not int
        or not 1 <= min(k, n) <= max(k, n) <= MAX_ELEMENTS
        or grouped % group
    ):
        raise ValueError(f"{LABEL} dimensions are invalid")
    m, batches = _grid(entry, n, execution)
    _check_route(info, route, m, n, k)
    expected = select(
        info["mode"], info["dtype"], n, k, info["globalScale"], transpose, group, bits
    )
    if expected != entry:
        raise ValueError(f"{LABEL} entry differs from MLX's routing")
    if threads != m * n * batches or counts["y"] != threads:
        raise ValueError(f"{LABEL} output extent differs")
    rows = n if transpose else k
    words, groups = grouped * bits // 32, grouped // group
    lhs, rhs = gathered_operands(
        info, counts, parameters, indices, (m, k, rows, words, groups, batches), LABEL
    )
    values = dict(parameters)
    if info["mode"] != "affine" and not info["globalScale"]:
        values["global_scale"] = [INACTIVE_GLOBAL_SCALE]
    return {
        "mode": info["mode"],
        "kernel": info["kernel"],
        "route": route,
        "M": m,
        "N": n,
        "K": k,
        "batches": batches,
        "lhsIndices": lhs,
        "rhsIndices": rhs,
        "overreadPadding": {},
        "parameters": values,
    }
