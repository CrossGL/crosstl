"""Storage and launch contracts for the gathered quantized product kernels.

``GatherQMM::eval_gpu`` in ``gather_qmm.inc`` routes ``mx.gather_qmm`` as
quantized.cpp does on a device without NAX:

- a single row per product (``M == 1``) whose right indices are sorted, with
  at least 16 products and 4 per weight matrix, goes to ``gather_qmm_rhs``,
  which the host does not route: it refuses those products;
- ``M`` at or above ``matmul_layout.vector_limit`` goes to ``gather_qmm_t``
  for transposed weights and ``gather_qmm_n`` otherwise, ``(32, 2, 2)``
  threadgroups over 32 by 32 output tiles;
- transposed weights below it go to ``gather_qmv_fast`` when ``N`` is a
  multiple of 8 and ``K`` of ``fast_alignment(bits)``, else ``gather_qmv``;
- untransposed weights below it go to ``gather_qvm``.

Affine kernels come from quantized.metal and bind biases. The mxfp4, mxfp8
and nvfp4 kernels come from fp_quantized.metal: uint8 scales, no biases, and
a float32 global scale per weight matrix that only their ``_hgs``
instantiations read; the others still declare it, and the host binds an
inert value for it. Every kernel binds the batch views of ``x`` and the
weights and the collapsed views of the two index arrays.

The host reads both index arrays: every product's gathered activation rows,
weights, scales, biases and global scale must lie inside their buffers.
"""

import ctypes
import math
import re

from demos.integrations.mlx.portable_host import matmul_layout, qmm_layout
from demos.integrations.mlx.portable_host.fp_matmul_layout import (
    INACTIVE_GLOBAL_SCALE,
    MODES,
)
from demos.integrations.mlx.portable_host.matmul_layout import (
    BATCH_VECTORS,
    MAX_BATCH_DIMENSIONS,
    MAX_ELEMENTS,
    MAX_GROUPS,
    SIMD_SIZE,
    fast_alignment,
    vector_limit,
)
from demos.integrations.mlx.portable_host.quantization_layout import (
    ITEM_SIZES,
    SOURCE_TYPES,
)

SOURCES = {
    "affine": "mlx/backend/metal/kernels/quantized.metal",
    "fp": "mlx/backend/metal/kernels/fp_quantized.metal",
}
LABEL = "Gathered quantized product"
PATTERN = re.compile(
    r"(affine|mxfp4|mxfp8|nvfp4)_gather_(qmv_fast|qmv|qvm|qmm_t|qmm_n)"
    r"_(float|float16_t|bfloat16_t)_gs_(16|32|64|128)_b_(2|3|4|5|6|8)"
    r"(?:_alN_(true|false))?(_hgs)?"
)
ENTRY_PREFIXES = tuple(f"{mode}_gather_q" for mode in ("affine", *MODES))
AFFINE_GROUPS = (32, 64, 128)
VECTOR_KERNELS = ("qmv_fast", "qmv", "qvm")
# qmv's 2 simdgroups x 4 rows per threadgroup.
QMV_ROWS = 8
VECTOR_SCALARS = ("in_vec_size", "out_vec_size")
MATRIX_SCALARS = qmm_layout.SCALARS
INDEX_PARAMETERS = ("lhs_indices", "rhs_indices")
GATHER_SCALARS = ("batch_ndims",)
GATHER_VECTORS = {
    "batch_shape": "int32",
    "lhs_strides": "int64",
    "rhs_strides": "int64",
}
BATCH_SCALARS = ("x_batch_ndims", "w_batch_ndims")
SCALARS = (*VECTOR_SCALARS, *MATRIX_SCALARS, *BATCH_SCALARS, *GATHER_SCALARS)
# OpenGL index ranges of the gathered offsets every kernel forms; the plan
# keeps each inside a buffer of at most MAX_ELEMENTS elements.
GATHER_INDEX_EXPRESSIONS = (
    "tid.z * lhs_strides[0]",
    "tid.z * rhs_strides[0]",
    "idx.x",
    "idx.y",
    "x_idx * x_strides[0]",
    "w_idx * w_strides[0]",
    "w_idx * s_strides[0]",
)
AFFINE_INDEX_EXPRESSIONS = ("w_idx * b_strides[0]", "idx.z")


class GatherLayout:
    """One binding family, in ``matmul_dispatch.dispatch_with``'s protocol.

    The affine kernels, the block-scaled kernels without a global scale, and
    the block-scaled ``_hgs`` kernels bind different matrix buffers.
    """

    LABEL = LABEL

    def __init__(self, family, global_scale):
        self.family, self.global_scale = family, global_scale
        self.SOURCE = SOURCES[family]
        if family == "affine":
            self.MATRIX_PARAMETERS = ("w", "scales", "biases", "x", "y")
        elif global_scale:
            self.MATRIX_PARAMETERS = ("w", "scales", "global_scale", "x", "y")
        else:
            self.MATRIX_PARAMETERS = ("w", "scales", "x", "y")
        self.BATCH_PARAMETERS = tuple(
            name
            for name in matmul_layout.BATCH_PARAMETERS
            if family == "affine" or name != "b_strides"
        )

    def signature(self, entry):
        info = signature(entry)
        if layout_for(entry) is not self:
            raise ValueError(f"{LABEL} entry belongs to another layout")
        return info

    def parameter_names(self, entry):
        return parameter_names(entry)

    def parameter_type(self, entry, name):
        return parameter_type(entry, name)

    def read_storage(self, entry, buffers):
        return read_storage(entry, buffers)

    def plan(self, entry, counts, parameters, threads, execution):
        return plan(entry, counts, parameters, threads, execution)


LAYOUTS = {
    ("affine", False): GatherLayout("affine", False),
    ("fp", False): GatherLayout("fp", False),
    ("fp", True): GatherLayout("fp", True),
}


def signature(entry):
    """The mode, kernel, element type, group size, bits, alignment and global scale."""
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError(f"Unsupported {LABEL.lower()} entry")
    mode, kernel, source, group, bits, aligned, hgs = match.groups()
    group, bits = int(group), int(bits)
    affine = mode == "affine"
    if (
        (aligned is None) != (kernel != "qmm_t")
        or (affine and (hgs or group not in AFFINE_GROUPS))
        or (not affine and MODES[mode] != (group, bits))
    ):
        raise ValueError(f"{LABEL} entry is never instantiated")
    return {
        "mode": mode,
        "family": "affine" if affine else "fp",
        "kernel": kernel,
        "dtype": next(key for key, value in SOURCE_TYPES.items() if value == source),
        "groupSize": group,
        "bits": bits,
        "aligned": None if aligned is None else aligned == "true",
        "globalScale": hgs is not None,
    }


def layout_for(entry):
    info = signature(entry)
    return LAYOUTS[(info["family"], info["globalScale"])]


def source(entry):
    return SOURCES[signature(entry)["family"]]


def workgroup_size(entry):
    if signature(entry)["kernel"] in VECTOR_KERNELS:
        return (SIMD_SIZE, 2, 1)
    return (SIMD_SIZE, qmm_layout.WN, qmm_layout.WM)


def transposed(entry):
    """Whether the entry multiplies by transposed weights, ``w`` of ``(N, K)``."""
    return signature(entry)["kernel"] not in ("qvm", "qmm_n")


def tile_columns(entry):
    """Output columns per threadgroup along the grid's column axis.

    qmv's 8 rows, qvm's 2 simdgroups x ``min(group_size, 32)`` columns and
    qmm's 32-column tiles.
    """
    info = signature(entry)
    if info["kernel"] == "qvm":
        return 2 * min(info["groupSize"], SIMD_SIZE)
    return QMV_ROWS if info["kernel"] in VECTOR_KERNELS else qmm_layout.BN


def select(mode, dtype, group, bits, m, n, k, transpose, global_scale=False):
    """GatherQMM::eval_gpu's entry for one product it does not send to
    ``gather_qmm_rhs``."""
    stem = f"{mode}_gather_"
    tail = f"_{SOURCE_TYPES[dtype]}_gs_{group}_b_{bits}"
    hgs = "_hgs" if global_scale else ""
    if m >= vector_limit(k, n, transpose):
        if not transpose:
            return f"{stem}qmm_n{tail}{hgs}"
        aligned = "true" if n % qmm_layout.BN == 0 else "false"
        return f"{stem}qmm_t{tail}_alN_{aligned}{hgs}"
    if not transpose:
        return f"{stem}qvm{tail}{hgs}"
    fast = n % QMV_ROWS == 0 and k % fast_alignment(bits) == 0
    return f"{stem}{'qmv_fast' if fast else 'qmv'}{tail}{hgs}"


def parameter_names(entry):
    """The parameters gather_qmm.inc binds, as quantized.cpp binds them."""
    info = signature(entry)
    layout = layout_for(entry)
    scalars = VECTOR_SCALARS if info["kernel"] in VECTOR_KERNELS else MATRIX_SCALARS
    matrices = layout.MATRIX_PARAMETERS
    return (
        *matrices[:-1],
        *INDEX_PARAMETERS,
        "y",
        *scalars,
        *layout.BATCH_PARAMETERS,
        *GATHER_SCALARS,
        *GATHER_VECTORS,
    )


def parameter_type(entry, name):
    info = signature(entry)
    if name in ("w", *INDEX_PARAMETERS):
        return "uint32"
    if name == "scales":
        return info["dtype"] if info["family"] == "affine" else "uint8"
    if name == "global_scale":
        return "float32"
    if name in ("biases", "x", "y"):
        return info["dtype"]
    if name in SCALARS:
        return "int32"
    if name in GATHER_VECTORS:
        return GATHER_VECTORS[name]
    if name in BATCH_VECTORS:
        return BATCH_VECTORS[name]
    raise ValueError(f"{LABEL} parameter is unknown")


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters.

    Returns ``(counts, parameters)``: element counts of the matrix and index
    buffers, and the values of the scalars, the views and the two index
    arrays, which bind by value.
    """
    names = parameter_names(entry)
    if set(buffers) != set(names):
        raise ValueError(f"{LABEL} buffers do not match the entry")
    plain = [name for name in names if name not in INDEX_PARAMETERS]
    counts, parameters = matmul_layout.read_buffers(
        {name: buffers[name] for name in plain},
        {name: parameter_type(entry, name) for name in plain},
        SCALARS,
        LABEL,
        matrices=layout_for(entry).MATRIX_PARAMETERS,
    )
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    output = buffers["y"]
    low = output.data
    high = low + output.count * ITEM_SIZES[signature(entry)["dtype"]]
    for name in INDEX_PARAMETERS:
        buffer, count = buffers[name], buffers[name].count
        if (
            not buffer.data
            or buffer.data % 4
            or buffer.dtype != b"uint32"
            or buffer.output != 0
            or type(count) is not int
            or not 1 <= count <= MAX_ELEMENTS
            or buffer.data + count * 4 > address_limit
        ):
            raise ValueError(f"{LABEL} index storage or direction does not match")
        if max(low, buffer.data) < min(high, buffer.data + count * 4):
            raise ValueError(f"{LABEL} output overlaps an index array")
        counts[name] = count
        parameters[name] = list((ctypes.c_uint32 * count).from_address(buffer.data))
    return counts, parameters


def location(index, shape, strides):
    """utils.h elem_to_loc over every axis of ``shape``."""
    offset = 0
    for extent, stride in zip(reversed(shape), reversed(strides)):
        offset += (index % extent) * stride
        index //= extent
    return offset


def gathered_offset(index, ndim, shape, strides):
    """adjust_matrix_offsets' move: one unwrapped stride, or elem_to_loc."""
    if ndim == 1:
        return index * strides[0]
    return location(index, shape[:ndim], strides[:ndim])


def _check_view(shape, strides, ndim, label):
    if (
        type(ndim) is not int
        or not 0 <= ndim <= MAX_BATCH_DIMENSIONS
        or not len(shape) == len(strides) == ndim + 2
        or any(extent < 1 for extent in shape)
        or any(stride < 0 for stride in strides[:ndim])
    ):
        raise ValueError(f"{LABEL} {label} view is invalid")


def _check_gather(parameters):
    """The collapsed index views; scalar indices bind one inert axis."""
    ndim = parameters["batch_ndims"]
    shape = parameters["batch_shape"]
    lhs, rhs = parameters["lhs_strides"], parameters["rhs_strides"]
    if type(ndim) is not int or not 0 <= ndim <= MAX_BATCH_DIMENSIONS:
        raise ValueError(f"{LABEL} index views are invalid")
    if ndim == 0:
        if (shape, lhs, rhs) != ([1], [0], [0]):
            raise ValueError(f"{LABEL} scalar index views are not inert")
        return 1
    if (
        not len(shape) == len(lhs) == len(rhs) == ndim
        or any(extent < 1 for extent in shape)
        or any(stride < 0 for stride in (*lhs, *rhs))
    ):
        raise ValueError(f"{LABEL} index views are invalid")
    return math.prod(shape)


def index_locations(z, parameters):
    """Where product ``z`` reads its left and right indices."""
    ndim = parameters["batch_ndims"]
    lhs, rhs = parameters["lhs_strides"], parameters["rhs_strides"]
    if ndim == 1:
        return z * lhs[0], z * rhs[0]
    if ndim == 0:
        return 0, 0
    shape = parameters["batch_shape"]
    return location(z, shape, lhs), location(z, shape, rhs)


def _dimensions(entry, parameters):
    """``(M, N, K)`` from the scalars, or qmv's and qvm's row count in x."""
    info = signature(entry)
    if info["kernel"] in VECTOR_KERNELS:
        k, n = (parameters.get(name) for name in VECTOR_SCALARS)
        shape, ndim = parameters["x_shape"], parameters["x_batch_ndims"]
        m = shape[ndim] if type(ndim) is int and 0 <= ndim < len(shape) else None
    else:
        k, n, m = (parameters.get(name) for name in MATRIX_SCALARS)
    if (
        any(type(value) is not int for value in (k, n, m))
        or not 1 <= min(k, n, m) <= max(k, n, m) <= MAX_ELEMENTS
        or (k if transposed(entry) else n) % info["groupSize"]
    ):
        raise ValueError(f"{LABEL} dimensions are invalid")
    return m, n, k


def _launch(entry, execution, m, n):
    """The product count of a launch whose grid matches quantized.cpp's."""
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError(f"{LABEL} launch geometry is missing")
    count = list(execution["workgroupCount"])
    columns = -(-n // tile_columns(entry))
    if signature(entry)["kernel"] in VECTOR_KERNELS:
        expected = [m, columns]
    else:
        expected = [columns, -(-m // qmm_layout.BM)]
    if (
        list(execution["workgroupSize"]) != list(workgroup_size(entry))
        or len(count) != 3
        or any(type(value) is not int for value in count)
        or count[:2] != expected
        or not 1 <= min(count) <= max(count) <= MAX_GROUPS
    ):
        raise ValueError(f"{LABEL} launch does not match MLX's grid")
    return count[2]


def plan(entry, counts, parameters, threads, execution):
    """Validate one launch against MLX's routing and return its metadata.

    The returned ``parameters`` add the inert global scale the block-scaled
    kernels without ``_hgs`` declare.
    """
    info = signature(entry)
    layout_for(entry)
    group, bits, transpose = info["groupSize"], info["bits"], transposed(entry)
    m, n, k = _dimensions(entry, parameters)
    batches = _launch(entry, execution, m, n)
    if (
        select(
            info["mode"],
            info["dtype"],
            group,
            bits,
            m,
            n,
            k,
            transpose,
            info["globalScale"],
        )
        != entry
    ):
        raise ValueError(f"{LABEL} entry differs from MLX's routing")
    if threads != m * n * batches or counts["y"] != threads:
        raise ValueError(f"{LABEL} output extent differs")
    if _check_gather(parameters) != batches:
        raise ValueError(f"{LABEL} index extents differ from the launch")
    rows, columns = (n, k) if transpose else (k, n)
    words, groups = columns * bits // 32, columns // group
    x_ndim, w_ndim = parameters["x_batch_ndims"], parameters["w_batch_ndims"]
    x_shape, x_strides = parameters["x_shape"], parameters["x_strides"]
    w_shape, w_strides = parameters["w_shape"], parameters["w_strides"]
    views = {"s_strides": parameters["s_strides"]}
    if info["family"] == "affine":
        views["b_strides"] = parameters["b_strides"]
    _check_view(x_shape, x_strides, x_ndim, "x")
    for name, strides in (("w_strides", w_strides), *views.items()):
        _check_view(w_shape, strides, w_ndim, name)
    if x_shape[-2:] != [m, k] or w_shape[-2:] != [rows, words]:
        raise ValueError(f"{LABEL} matrix shapes do not match")
    # ensure_row_contiguous_matrix: unit columns and dense rows.
    for strides, extent, width in (
        (x_strides, m, k),
        (w_strides, rows, words),
        *((strides, rows, groups) for strides in views.values()),
    ):
        if strides[-1] != 1 or (extent > 1 and strides[-2] != width):
            raise ValueError(f"{LABEL} matrices are not row contiguous")
    extents = {"x": m * k, "w": rows * words, "scales": rows * groups}
    if info["family"] == "affine":
        extents["biases"] = rows * groups
    offsets = {
        "x": (x_ndim, x_shape, x_strides),
        "w": (w_ndim, w_shape, w_strides),
        "scales": (w_ndim, w_shape, views["s_strides"]),
        "biases": (w_ndim, w_shape, views.get("b_strides")),
    }
    gathered = []
    for z in range(batches):
        lhs, rhs = index_locations(z, parameters)
        if lhs >= counts["lhs_indices"] or rhs >= counts["rhs_indices"]:
            raise ValueError(f"{LABEL} index read exceeds its buffer")
        x_index = parameters["lhs_indices"][lhs]
        w_index = parameters["rhs_indices"][rhs]
        gathered.append((x_index, w_index))
        for name, extent in extents.items():
            ndim, shape, strides = offsets[name]
            index = x_index if name == "x" else w_index
            if gathered_offset(index, ndim, shape, strides) + extent > counts[name]:
                raise ValueError(f"{LABEL} gathered {name} exceeds its buffer")
        if info["globalScale"] and w_index >= counts["global_scale"]:
            raise ValueError(f"{LABEL} gathered global scale exceeds its buffer")
    values = dict(parameters)
    if info["family"] == "fp" and not info["globalScale"]:
        values["global_scale"] = [INACTIVE_GLOBAL_SCALE]
    return {
        "kernel": info["kernel"],
        "M": m,
        "N": n,
        "K": k,
        "batches": batches,
        "gathered": gathered if batches <= 4096 else None,
        "overreadPadding": {},
        "parameters": values,
    }


def access_assertions(entry):
    """The matrix kernels' x loader slot bounds, as the qmm gates assert them."""
    info = signature(entry)
    if info["kernel"] in VECTOR_KERNELS:
        return ()
    return qmm_layout.access_assertions(entry, info, source(entry))


def index_assertions(entry):
    """OpenGL ranges for the 32-bit indices each kernel forms.

    The vector kernels assert the gathered offsets, as the qmv and qvm gates
    do for their gather entries; ``plan`` keeps each inside a buffer of at
    most ``MAX_ELEMENTS`` elements. The matrix kernels keep the qmm gates'
    ranges and add the index arrays' element reads ahead of the threadgroup
    tiles, as the gather_qmm gate asserts them.
    """
    info = signature(entry)
    path = source(entry)
    if info["kernel"] in VECTOR_KERNELS:
        expressions = GATHER_INDEX_EXPRESSIONS
        if info["family"] == "affine":
            expressions += AFFINE_INDEX_EXPRESSIONS
        return tuple(
            {
                "source": path,
                "expression": expression,
                "minimum": 0,
                "maximum": MAX_ELEMENTS,
            }
            for expression in expressions
        )
    tiles = qmm_layout.index_assertions(entry, info, path)
    items = [item for item in tiles if "indexed" not in item]
    items.append(
        {
            "source": path,
            "expression": "reference.offset + index",
            "function": "crosstl_resource_load_uint",
            "minimum": 0,
            "maximum": qmm_layout.SIGNED_MAX,
        }
    )
    return (*items, *(item for item in tiles if "indexed" in item))


def bounded_parameters(entry):
    """The scalars the matrix translations bound by ``qmm_layout.K_LIMIT``."""
    if signature(entry)["kernel"] in VECTOR_KERNELS:
        return ()
    return ("K",)
