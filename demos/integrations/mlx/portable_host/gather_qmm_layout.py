"""Storage and launch contracts for MLX's gathered quantized matrix-matrix kernels.

``GatherQMM::eval_gpu`` in ``gather_qmm.inc`` sends products of at least
``matmul_layout.vector_limit`` rows to quantized.cpp's ``gather_qmm``. The
portable host has no NAX unit, so ``gather_qmm`` never takes
``gather_qmm_nax``: transposed weights go to ``gather_qmm_t``, named for
whether ``N`` fills whole 32-column tiles, and untransposed ones to
``gather_qmm_n``. Affine weights run quantized.metal's kernels, which bind
biases; mxfp4, mxfp8 and nvfp4 weights run fp_quantized.metal's, with uint8
scales and, for nvfp4 weights that carry them, the ``_hgs`` instantiations
that read the gathered matrix's global scale. Every launch runs
``(32, 2, 2)`` threadgroups over a ``(ceil(N / 32), ceil(M / 32), B)`` grid,
the tiles and loaders of the ungathered qmm kernels (``qmm_layout``).

The kernels move the input and the weights to the gathered matrices as the
matrix-vector kernels do, so ``plan`` reads both indices of every batch and
bounds the operands they gather through ``gather_qmv_layout``.
"""

import re

from demos.integrations.mlx.portable_host import (
    gather_qmv_layout,
    matmul_layout,
    qmm_layout,
)
from demos.integrations.mlx.portable_host.gather_qmv_layout import (
    AFFINE_SOURCE,
    INACTIVE_GLOBAL_SCALE,
    INDEX_PARAMETERS,
    MAX_ELEMENTS,
    MODES,
    SIMD_SIZE,
    SOURCE,
    check_mode,
)
from demos.integrations.mlx.portable_host.qmm_layout import BN, K_LIMIT, WM, WN
from demos.integrations.mlx.portable_host.quantization_layout import SOURCE_TYPES

LABEL = "Gathered quantized matrix-matrix"
PATTERN = re.compile(
    r"(affine|mxfp4|mxfp8|nvfp4)_gather_(qmm_t|qmm_n)_(float|float16_t|bfloat16_t)"
    r"_gs_(16|32|64|128)_b_(2|3|4|5|6|8)(?:_alN_(true|false))?(_hgs)?"
)
ENTRY_PREFIXES = tuple(
    f"{mode}_gather_{kernel}_"
    for mode in ("affine", *MODES)
    for kernel in ("qmm_t", "qmm_n")
)
# The primitives whose routing a launch follows; every launch names one.
ROUTES = ("gather_qmm",)
SCALAR_PARAMETERS = ("K", "N", "M", "x_batch_ndims", "w_batch_ndims", "batch_ndims")


def signature(entry):
    """The mode, kernel, element type, group size, bits, alignment and scale.

    gather_qmm_t names whether ``N`` is aligned; gather_qmm_n does not.
    """
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError(f"Unsupported {LABEL.lower()} entry")
    mode, kernel, source, group, bits, aligned, scaled = match.groups()
    group, bits = int(group), int(bits)
    check_mode(mode, group, bits, scaled is not None, LABEL)
    if (aligned is None) != (kernel == "qmm_n"):
        raise ValueError(f"Unsupported {LABEL.lower()} entry")
    return {
        "mode": mode,
        "kernel": kernel,
        "dtype": next(key for key, value in SOURCE_TYPES.items() if value == source),
        "groupSize": group,
        "bits": bits,
        "aligned": None if aligned is None else aligned == "true",
        "globalScale": scaled is not None,
    }


def source(entry):
    """The pinned source that instantiates ``entry``."""
    return AFFINE_SOURCE if signature(entry)["mode"] == "affine" else SOURCE


def workgroup_size(entry):
    signature(entry)
    return (SIMD_SIZE, WN, WM)


def transposed(entry):
    """Whether the entry multiplies by transposed weights, ``w`` of ``(N, K)``."""
    return signature(entry)["kernel"] == "qmm_t"


def select(
    mode, dtype, m, n, k, global_scale=False, transpose=True, group=None, bits=None
):
    """quantized.cpp gather_qmm's kernel for one product without NAX.

    The block-scaled modes fix their group size and bits; affine products
    name theirs.
    """
    if mode != "affine" and group is None:
        group, bits = MODES[mode]
    if group is None or bits is None:
        raise ValueError(f"{LABEL} affine products name their group and bits")
    check_mode(mode, group, bits, global_scale, LABEL)
    if m < matmul_layout.vector_limit(k, n, transpose):
        raise ValueError(f"{LABEL} vector products take gather_qmv and gather_qvm")
    stem = f"{SOURCE_TYPES[dtype]}_gs_{group}_b_{bits}"
    scaled = "_hgs" if global_scale else ""
    if not transpose:
        return f"{mode}_gather_qmm_n_{stem}{scaled}"
    aligned = "_alN_true" if n % BN == 0 else "_alN_false"
    return f"{mode}_gather_qmm_t_{stem}{aligned}{scaled}"


def matrix_parameters(entry):
    """The buffers the host binds by address, as for the vector kernels."""
    info = signature(entry)
    if info["mode"] == "affine":
        return gather_qmv_layout.AFFINE_MATRIX_PARAMETERS
    if info["globalScale"]:
        return (*gather_qmv_layout.MATRIX_PARAMETERS, "global_scale")
    return gather_qmv_layout.MATRIX_PARAMETERS


def vector_types(entry):
    if signature(entry)["mode"] == "affine":
        return gather_qmv_layout.AFFINE_VECTOR_TYPES
    return gather_qmv_layout.VECTOR_TYPES


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


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters.

    The parameters hold zero-copy views of the two index buffers under their
    names, which ``plan`` reads and drops.
    """
    return gather_qmv_layout.read_gathered(
        buffers,
        {name: parameter_type(entry, name) for name in parameter_names(entry)},
        SCALAR_PARAMETERS,
        matrix_parameters(entry),
        LABEL,
    )


def plan(entry, counts, parameters, threads, execution, route=None):
    """Validate one launch against MLX's routing and return its metadata.

    The launch must have at least the vector limit's rows, and every batch's
    gathered operands must lie inside their buffers. The returned
    ``parameters`` drop the index views and add the inert global scale of
    the block-scaled kernels without _hgs.
    """
    info = signature(entry)
    if route not in ROUTES:
        raise ValueError(f"{LABEL} route is unknown")
    parameters = dict(parameters)
    indices = {name: parameters.pop(name, None) for name in INDEX_PARAMETERS}
    if set(parameters) != {*SCALAR_PARAMETERS, *vector_types(entry)} or any(
        view is None for view in indices.values()
    ):
        raise ValueError(f"{LABEL} parameters are missing")
    k, n, m = (parameters[name] for name in ("K", "N", "M"))
    group, bits = info["groupSize"], info["bits"]
    transpose = info["kernel"] == "qmm_t"
    # Groups run along the weights' rows: K when transposed, N otherwise.
    grouped = k if transpose else n
    if (
        any(type(value) is not int for value in (k, n, m))
        or not 1 <= min(k, n, m) <= max(k, n, m) <= MAX_ELEMENTS
        or k > K_LIMIT
        or grouped % group
    ):
        raise ValueError(f"{LABEL} dimensions are invalid")
    # (32, 2, 2) threadgroups over 32 by 32 output tiles.
    batches = qmm_layout.launch_depth(execution, m, n, LABEL)
    if m < matmul_layout.vector_limit(k, n, transpose):
        raise ValueError(f"{LABEL} vector products take gather_qmv and gather_qvm")
    expected = select(
        info["mode"],
        info["dtype"],
        m,
        n,
        k,
        info["globalScale"],
        transpose,
        group,
        bits,
    )
    if expected != entry:
        raise ValueError(f"{LABEL} entry differs from MLX's routing")
    if threads != m * n * batches or counts["y"] != threads:
        raise ValueError(f"{LABEL} output extent differs")
    rows = n if transpose else k
    words, groups = grouped * bits // 32, grouped // group
    lhs, rhs = gather_qmv_layout.gathered_operands(
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


def _tile_info(entry):
    """The signature fields qmm_layout's loader and offset helpers read."""
    info = signature(entry)
    return {"kernel": info["kernel"], "dtype": info["dtype"]}


def access_assertions(entry):
    """The ungathered kernels' x loader slot bounds, scoped to this entry."""
    return qmm_layout.access_assertions(entry, _tile_info(entry), source(entry))


def index_assertions(entry):
    """OpenGL ranges: the qmm kernels' offsets and the gathered index reads.

    The gathered kernels run the qmm tiles, whose resource helpers read
    elements, weight bytes and int64 batch strides; ``crosstl_resource_load_uint``
    reads each batch's two uint32 indices. ``plan`` proves every read names
    an element of a buffer of at most MAX_ELEMENTS elements, so each offset
    fits GLSL's 32-bit index.
    """
    path = source(entry)
    items = qmm_layout.index_assertions(entry, _tile_info(entry), path)
    tiles = tuple(item for item in items if "indexed" in item)
    indices = {
        "source": path,
        "expression": "reference.offset + index",
        "function": "crosstl_resource_load_uint",
        "minimum": 0,
        "maximum": qmm_layout.SIGNED_MAX,
    }
    return (*(item for item in items if "indexed" not in item), indices, *tiles)
