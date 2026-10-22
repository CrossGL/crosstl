"""Storage and launch contracts for the block-scaled quantized matrix-matrix kernels.

``QuantizedMatmul::eval_gpu`` in ``backend.cpp`` sends mxfp4, mxfp8 and nvfp4
products with at least ``matmul_layout.vector_limit`` rows to quantized.cpp's
matrix-matrix kernels, through fp_quantized.metal's ``fp_qmm_t``,
``fp_qmm_n`` and ``fp_qmm_t_splitk``: uint32 weights, one uint8 scale per
group and no biases. The routes are the affine ones (``qmm_layout``):
unbatched transposed products go to ``qmm_t_splitk`` when ``K`` admits a
split, else to ``qmm_t``; untransposed products go to ``qmm_n``. The tiles,
loaders and ``(32, 2, 2)`` threadgroups are the affine kernels' too.

QuantizedMatmul never passes a global scale, so no routed entry reads one and
the ``_hgs`` entries are never selected. ``fp_qmm_t`` still declares that
argument; quantized.cpp leaves its slot unbound, and the host binds an inert
value for it.
"""

import re

from demos.integrations.mlx.portable_host import matmul_layout, qmm_layout
from demos.integrations.mlx.portable_host.fp_matmul_layout import (
    INACTIVE_GLOBAL_SCALE,
    MODES,
)
from demos.integrations.mlx.portable_host.qmm_layout import (
    BN,
    MAX_ELEMENTS,
    SCALARS,
    SIMD_SIZE,
    SPLIT_SCALARS,
    WM,
    WN,
    split_k,
)
from demos.integrations.mlx.portable_host.quantization_layout import SOURCE_TYPES

SOURCE = "mlx/backend/metal/kernels/fp_quantized.metal"
LABEL = "Floating-point quantized matrix-matrix"
PATTERN = re.compile(
    r"(mxfp4|mxfp8|nvfp4)_(qmm_t_splitk|qmm_t|qmm_n)_(float|float16_t|bfloat16_t)"
    r"_gs_(16|32)_b_(4|8)(?:_alN_(true|false))?(?:_batch_(0|1))?"
)
ENTRY_PREFIXES = tuple(
    f"{mode}_{kernel}_" for mode in MODES for kernel in ("qmm_t", "qmm_n")
)
MATRIX_PARAMETERS = ("w", "scales", "x", "y")
# fp_quantized.h adjust_matrix_offsets: the affine views without bias strides.
BATCH_PARAMETERS = tuple(
    name for name in matmul_layout.BATCH_PARAMETERS if name != "b_strides"
)


def signature(entry):
    """The mode, kernel, element type, group size, bits, alignment and batching.

    As for the affine kernels, qmm_t names alignment and batching,
    qmm_t_splitk only alignment and qmm_n only batching.
    """
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError(f"Unsupported {LABEL.lower()} entry")
    mode, kernel, source, group, bits, aligned, batched = match.groups()
    if (
        MODES[mode] != (int(group), int(bits))
        or (aligned is None) != (kernel == "qmm_n")
        or (batched is None) != (kernel == "qmm_t_splitk")
    ):
        raise ValueError(f"{LABEL} entry is never instantiated")
    return {
        "mode": mode,
        "kernel": kernel,
        "dtype": next(key for key, value in SOURCE_TYPES.items() if value == source),
        "groupSize": int(group),
        "bits": int(bits),
        "aligned": None if aligned is None else aligned == "true",
        "batched": batched == "1",
    }


def workgroup_size(entry):
    signature(entry)
    return (SIMD_SIZE, WN, WM)


def transposed(entry):
    """Whether the entry multiplies by transposed weights, ``w`` of ``(N, K)``."""
    return signature(entry)["kernel"] != "qmm_n"


def select(mode, dtype, m, n, k, batches, transpose=True):
    """quantized.cpp QuantizedMatmul::eval_gpu's matrix-matrix entry."""
    group, bits = MODES[mode]
    if m < matmul_layout.vector_limit(k, n, transpose):
        raise ValueError(f"{LABEL} vector products take the qmv and qvm routes")
    stem = f"{SOURCE_TYPES[dtype]}_gs_{group}_b_{bits}"
    batch = "_batch_1" if batches > 1 else "_batch_0"
    if not transpose:
        return f"{mode}_qmm_n_{stem}{batch}"
    aligned = "_alN_true" if n % BN == 0 else "_alN_false"
    if batches == 1 and split_k(m, n, k, group) > 1:
        return f"{mode}_qmm_t_splitk_{stem}{aligned}"
    return f"{mode}_qmm_t_{stem}{aligned}{batch}"


def declares_global_scale(entry):
    """Whether the kernel declares the global scale it never reads here."""
    return signature(entry)["kernel"] == "qmm_t"


def parameter_names(entry):
    """The parameters backend.cpp binds, as quantized.cpp binds them.

    add_strides_and_shapes skips the batch views of unbatched launches; the
    host never binds the global scale.
    """
    info = signature(entry)
    names = (*MATRIX_PARAMETERS, *SCALARS)
    if info["kernel"] == "qmm_t_splitk":
        return (*names, *SPLIT_SCALARS)
    return (*names, *BATCH_PARAMETERS) if info["batched"] else names


def parameter_type(entry, name):
    if name == "w":
        return "uint32"
    if name == "scales":
        return "uint8"
    if name == "global_scale":
        return "float32"
    if name in MATRIX_PARAMETERS:
        return signature(entry)["dtype"]
    if name in SCALARS or name in SPLIT_SCALARS or name in matmul_layout.BATCH_SCALARS:
        return "int32"
    if name in BATCH_PARAMETERS:
        return matmul_layout.BATCH_VECTORS[name]
    raise ValueError(f"{LABEL} parameter is unknown")


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters."""
    names = parameter_names(entry)
    return matmul_layout.read_buffers(
        buffers,
        {name: parameter_type(entry, name) for name in names},
        (*SCALARS, *SPLIT_SCALARS, *matmul_layout.BATCH_SCALARS),
        LABEL,
        matrices=MATRIX_PARAMETERS,
    )


def _inert(m, k, rows, words, groups):
    """The batch values an unbatched kernel reflects but never reads."""
    return {
        "x_batch_ndims": 0,
        "x_shape": [m, k],
        "x_strides": [k, 1],
        "w_batch_ndims": 0,
        "w_shape": [rows, words],
        "w_strides": [words, 1],
        "s_strides": [groups, 1],
    }


def _partitions(parameters, supplied, m, n, k, split):
    if supplied != set(SPLIT_SCALARS) or (
        parameters["k_partition_size"],
        parameters["split_k_partition_stride"],
    ) != (k // split, m * n):
        raise ValueError(f"{LABEL} split-K partitions differ from quantized.cpp's")


def plan(entry, counts, parameters, threads, execution):
    """Validate one launch against MLX's routing and return its metadata.

    The returned ``parameters`` add the inert global scale ``fp_qmm_t``
    declares and the inert batch views unbatched kernels reflect.
    """
    info = signature(entry)
    kernel, group, bits = info["kernel"], info["groupSize"], info["bits"]
    k, n, m = (parameters.get(name) for name in SCALARS)
    transpose = kernel != "qmm_n"
    if (
        any(type(value) is not int for value in (k, n, m))
        or not 1 <= min(k, n, m) <= max(k, n, m) <= MAX_ELEMENTS
        or (k if transpose else n) % group
    ):
        raise ValueError(f"{LABEL} dimensions are invalid")
    depth = qmm_layout.launch_depth(execution, m, n, LABEL)
    split = depth if kernel == "qmm_t_splitk" else 1
    batches = depth // split
    if select(info["mode"], info["dtype"], m, n, k, batches, transpose) != entry or (
        split != 1 and split != split_k(m, n, k, group)
    ):
        raise ValueError(f"{LABEL} entry differs from MLX's routing")
    if threads != m * n * depth or counts["y"] != threads:
        raise ValueError(f"{LABEL} output extent differs")
    # Groups run along the weights' rows: K when transposed, N otherwise.
    rows, columns = (n, k) if transpose else (k, n)
    words, groups = columns * bits // 32, columns // group
    supplied = set(parameters) - set(SCALARS)
    offsets, values = [(0, 0, 0, 0)], dict(parameters)
    if kernel == "qmm_t_splitk":
        _partitions(parameters, supplied, m, n, k, split)
    elif info["batched"]:
        if supplied != set(BATCH_PARAMETERS):
            raise ValueError(f"Batched {LABEL.lower()} parameters are missing")
        offsets = matmul_layout.batch_offsets(
            parameters, m, k, rows, words, groups, batches, LABEL
        )
    else:
        if supplied or batches != 1:
            raise ValueError(f"Unbatched {LABEL.lower()} launch is batched")
        values.update(_inert(m, k, rows, words, groups))
    if declares_global_scale(entry):
        values["global_scale"] = [INACTIVE_GLOBAL_SCALE]
    extents = {"x": m * k, "w": rows * words, "scales": rows * groups}
    for column, name in enumerate(("x", "w", "scales")):
        if max(offset[column] for offset in offsets) + extents[name] > counts[name]:
            raise ValueError(f"{LABEL} {name} batch exceeds its buffer")
    return {
        "mode": info["mode"],
        "kernel": kernel,
        "M": m,
        "N": n,
        "K": k,
        "batches": batches,
        "splitK": split,
        "overreadPadding": {"scales": 0},
        "parameters": values,
    }


def access_assertions(entry):
    """The affine x loader's slot bounds, scoped to the block-scaled source."""
    return qmm_layout.access_assertions(entry, signature(entry), SOURCE)


def index_assertions(entry):
    """The affine kernels' OpenGL offset ranges, scoped to this source."""
    return qmm_layout.index_assertions(entry, signature(entry), SOURCE)


def bounded_parameters(entry):
    return qmm_layout.bounded_parameters(entry, signature(entry))
