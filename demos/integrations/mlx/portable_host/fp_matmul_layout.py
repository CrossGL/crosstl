"""Storage and launch contracts for floating-point quantized matrix-vector kernels.

``QuantizedMatmul::eval_gpu`` in ``backend.cpp`` routes mxfp4, mxfp8 and
nvfp4 products with few input vectors as quantized.cpp does, through
fp_quantized.metal's kernels: uint32 weights, one uint8 scale per group and no
biases. Transposed weights go to ``dispatch_qmv``: ``fp_qmv_quad`` for ``K``
of 64 or 128, then ``fp_qmv_wide`` for two or more vectors (``use_qmv_wide``
holds for every floating-point mode), then ``fp_qmv_fast`` when ``N`` and
``K`` are aligned and ``fp_qmv`` otherwise. Untransposed weights go to
``fp_qvm`` while ``K`` is below 1024, and longer reductions to
``fp_qvm_split_k``: 8 partitions of ``K`` (32 above 8192) whose partial
products a column sum then adds. QuantizedMatmul never passes a global scale,
so no routed entry reads one; the pinned qmv and qvm templates still declare
that argument, and the host binds an inert value for it.
"""

import re

from demos.integrations.mlx.portable_host import matmul_layout
from demos.integrations.mlx.portable_host.matmul_layout import (
    BATCH_VECTORS,
    MAX_ELEMENTS,
    MAX_GROUPS,
    QUAD_DIMENSIONS,
    SIMD_SIZE,
    fast_alignment,
    vector_limit,
)
from demos.integrations.mlx.portable_host.quantization_layout import SOURCE_TYPES

SOURCE = "mlx/backend/metal/kernels/fp_quantized.metal"
# fp_quantized.metal's instantiate_quantized_types: each mode's group size and bits.
MODES = {"mxfp4": (32, 4), "mxfp8": (32, 8), "nvfp4": (16, 4)}
PATTERN = re.compile(
    r"(mxfp4|mxfp8|nvfp4)_(qmv_quad|qmv_fast|qmv_wide|qmv|qvm)"
    r"_(float|float16_t|bfloat16_t)_gs_(16|32)_b_(4|8)"
    r"(?:_d_(64|128))?(?:_nv_([2-5])_kl_16)?_batch_(0|1)"
)
SPLIT_PATTERN = re.compile(
    r"(mxfp4|mxfp8|nvfp4)_qvm_split_k_(float|float16_t|bfloat16_t)"
    r"_gs_(16|32)_b_(4|8)_spk_(8|32)"
)
ENTRY_PREFIXES = tuple(
    f"{mode}_{kernel}_" for mode in MODES for kernel in ("qmv", "qvm")
)
LABEL = "Floating-point quantized matrix-vector"
# quantized.cpp qmv_wide: at most 5 vectors per threadgroup, 16 lanes per row.
WIDE_VECTORS = 5
WIDE_LANES = 16
MATRIX_PARAMETERS = ("w", "scales", "x", "y")
SCALAR_PARAMETERS = ("in_vec_size", "out_vec_size")
BATCH_SCALARS = ("x_batch_ndims", "w_batch_ndims")
BATCH_PARAMETERS = (
    "x_batch_ndims",
    "x_shape",
    "x_strides",
    "w_batch_ndims",
    "w_shape",
    "w_strides",
    "s_strides",
)
# The inert value of the global scale qmv, qmv_fast and qvm declare.
INACTIVE_GLOBAL_SCALE = 0
# QuantizedMatmul::eval_gpu sends untransposed products from this K to
# qvm_split_k, in 32 partitions above QVM_WIDE_SPLIT and 8 below.
QVM_SPLIT_K = matmul_layout.QVM_SPLIT_K
QVM_WIDE_SPLIT = matmul_layout.QVM_WIDE_SPLIT
SPLIT_SCALARS = matmul_layout.SPLIT_SCALARS


def signature(entry):
    """The mode, kernel, element type, group size, bits, ``D``, vectors and batching.

    Split-K entries add their partition count and always bind batch views.
    """
    split = SPLIT_PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if split is not None:
        mode, source, group, bits, partitions = split.groups()
        if MODES[mode] != (int(group), int(bits)):
            raise ValueError(f"{LABEL} entry is never instantiated")
        return {
            "mode": mode,
            "kernel": "qvm_split_k",
            "dtype": next(
                key for key, value in SOURCE_TYPES.items() if value == source
            ),
            "groupSize": int(group),
            "bits": int(bits),
            "D": None,
            "vectors": None,
            "batched": True,
            "splitK": int(partitions),
        }
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError(f"Unsupported {LABEL.lower()} entry")
    mode, kernel, source, group, bits, dimension, vectors, batched = match.groups()
    group, bits = int(group), int(bits)
    if (
        MODES[mode] != (group, bits)
        or (kernel == "qmv_quad") != (dimension is not None)
        or (kernel == "qmv_wide") != (vectors is not None)
    ):
        raise ValueError(f"{LABEL} entry is never instantiated")
    return {
        "mode": mode,
        "kernel": kernel,
        "dtype": next(key for key, value in SOURCE_TYPES.items() if value == source),
        "groupSize": group,
        "bits": bits,
        "D": int(dimension) if dimension else None,
        "vectors": int(vectors) if vectors else None,
        "batched": batched == "1",
    }


def workgroup_size(entry):
    return (SIMD_SIZE, 1 if signature(entry)["kernel"] == "qmv_quad" else 2, 1)


def transposed(entry):
    """Whether the entry multiplies by transposed weights, ``w`` of ``(N, K)``."""
    return signature(entry)["kernel"] not in ("qvm", "qvm_split_k")


def tile_rows(entry):
    """Outputs per threadgroup.

    qmv_quad's 8 quads x 8 rows, qmv_wide's 2 simdgroups x 32 / 16 rows,
    qmv's 2 simdgroups x 4 rows, and qvm's 2 simdgroups x
    ``min(group_size, 32)`` columns.
    """
    info = signature(entry)
    return {
        "qmv_quad": 64,
        "qmv_wide": 2 * (SIMD_SIZE // WIDE_LANES),
        "qvm": 2 * min(info["groupSize"], SIMD_SIZE),
        "qvm_split_k": 2 * min(info["groupSize"], SIMD_SIZE),
    }.get(info["kernel"], 8)


def wide_vectors(m):
    """quantized.cpp qmv_wide's vecs_per_tg: the fewest tiles of at most 5."""
    tiles = -(-m // WIDE_VECTORS)
    return -(-m // tiles)


def select(mode, dtype, m, n, k, batches, transpose=True):
    """quantized.cpp QuantizedMatmul::eval_gpu's entry for one product.

    Raises for the products MLX sends to kernels the host does not route:
    the matrix-matrix kernels.
    """
    group, bits = MODES[mode]
    if m >= vector_limit(k, n, transpose):
        raise ValueError(f"{LABEL} matrix-matrix products are not routed")
    stem = f"{SOURCE_TYPES[dtype]}_gs_{group}_b_{bits}"
    batch = "_batch_1" if batches > 1 else "_batch_0"
    if not transpose:
        if k >= QVM_SPLIT_K:
            partitions = 32 if k > QVM_WIDE_SPLIT else 8
            return f"{mode}_qvm_split_k_{stem}_spk_{partitions}"
        return f"{mode}_qvm_{stem}{batch}"
    if k in QUAD_DIMENSIONS:
        # Both 4 and 8 bits are powers of two.
        return f"{mode}_qmv_quad_{stem}_d_{k}{batch}"
    if m >= 2:
        return f"{mode}_qmv_wide_{stem}_nv_{wide_vectors(m)}_kl_{WIDE_LANES}{batch}"
    fast = n % 8 == 0 and k % fast_alignment(bits) == 0
    return f"{mode}_{'qmv_fast' if fast else 'qmv'}_{stem}{batch}"


def declares_global_scale(entry):
    """Whether the kernel declares the global scale argument it never reads here."""
    return signature(entry)["kernel"] not in ("qmv_quad", "qmv_wide", "qvm_split_k")


def scalar_parameters(entry):
    wide = signature(entry)["kernel"] == "qmv_wide"
    return (*SCALAR_PARAMETERS, "M") if wide else SCALAR_PARAMETERS


def parameter_names(entry):
    """The parameters the host binds: never the global scale."""
    names = (*MATRIX_PARAMETERS, *scalar_parameters(entry))
    if "splitK" in signature(entry):
        return (*names, *BATCH_PARAMETERS, *SPLIT_SCALARS)
    return (*names, *BATCH_PARAMETERS) if signature(entry)["batched"] else names


def parameter_type(entry, name):
    if name == "w":
        return "uint32"
    if name == "scales":
        return "uint8"
    if name == "global_scale":
        return "float32"
    if name in MATRIX_PARAMETERS:
        return signature(entry)["dtype"]
    if name in scalar_parameters(entry) or name in (*BATCH_SCALARS, *SPLIT_SCALARS):
        return "int32"
    if name in BATCH_PARAMETERS:
        return BATCH_VECTORS[name]
    raise ValueError(f"{LABEL} parameter is unknown")


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters."""
    names = parameter_names(entry)
    return matmul_layout.read_buffers(
        buffers,
        {name: parameter_type(entry, name) for name in names},
        (*scalar_parameters(entry), *BATCH_SCALARS, *SPLIT_SCALARS),
        LABEL,
        matrices=MATRIX_PARAMETERS,
    )


def _grid(entry, info, n, parameters, execution):
    """The vector count and batches of a launch that matches quantized.cpp's grid."""
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError(f"{LABEL} launch geometry is missing")
    columns, tiles, batches = execution["workgroupCount"]
    m = parameters.get("M") if info["kernel"] == "qmv_wide" else columns
    if (
        list(execution["workgroupSize"]) != list(workgroup_size(entry))
        or any(type(value) is not int for value in (m, columns, tiles, batches))
        or not 1 <= m <= MAX_GROUPS * WIDE_VECTORS
        or not 1 <= columns <= MAX_GROUPS
        or not 1 <= batches <= MAX_GROUPS
        or tiles != -(-n // tile_rows(entry))
        or tiles > MAX_GROUPS
    ):
        raise ValueError(f"{LABEL} launch does not match MLX's grid")
    if info["kernel"] == "qmv_wide" and (
        info["vectors"] != wide_vectors(m) or columns != -(-m // info["vectors"])
    ):
        raise ValueError(f"{LABEL} vector tiles differ from quantized.cpp's")
    return m, batches


def plan(entry, counts, parameters, threads, execution):
    """Validate one launch against MLX's routing and return its metadata.

    The returned ``parameters`` add the inert global scale and the unbatched
    batch views that the translations reflect but never read.
    """
    info = signature(entry)
    if "splitK" in info:
        metadata = matmul_layout.plan_split(
            entry,
            counts,
            parameters,
            threads,
            execution,
            info,
            lambda m, n, k, batches: select(
                info["mode"], info["dtype"], m, n, k, batches, transpose=False
            ),
        )
        return {"mode": info["mode"], **metadata}
    k, n = parameters.get("in_vec_size"), parameters.get("out_vec_size")
    group, bits = info["groupSize"], info["bits"]
    transpose = transposed(entry)
    grouped = k if transpose else n
    if (
        type(k) is not int
        or type(n) is not int
        or not 1 <= min(k, n) <= max(k, n) <= MAX_ELEMENTS
        or grouped % group
    ):
        raise ValueError(f"{LABEL} dimensions are invalid")
    m, batches = _grid(entry, info, n, parameters, execution)
    if select(info["mode"], info["dtype"], m, n, k, batches, transpose) != entry:
        raise ValueError(f"{LABEL} entry differs from MLX's routing")
    if threads != m * n * batches or counts["y"] != threads:
        raise ValueError(f"{LABEL} output extent differs")
    rows = n if transpose else k
    words, groups = grouped * bits // 32, grouped // group
    supplied = set(parameters) - set(scalar_parameters(entry))
    if info["batched"]:
        if supplied != set(BATCH_PARAMETERS):
            raise ValueError(f"Batched {LABEL.lower()} parameters are missing")
        offsets = matmul_layout.batch_offsets(
            parameters, m, k, rows, words, groups, batches, LABEL
        )
        values = dict(parameters)
    else:
        if supplied or batches != 1:
            raise ValueError(f"Unbatched {LABEL.lower()} launch is batched")
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
        }
    if declares_global_scale(entry):
        values["global_scale"] = [INACTIVE_GLOBAL_SCALE]
    extents = {"x": m * k, "w": rows * words, "scales": rows * groups}
    for column, name in enumerate(("x", "w", "scales")):
        if max(offset[column] for offset in offsets) + extents[name] > counts[name]:
            raise ValueError(f"{LABEL} {name} batch exceeds its buffer")
    # qmv_quad loads one scale per row of its last tile, past N.
    padding = {"scales": 0}
    if info["kernel"] == "qmv_quad":
        end = max(offset[2] for offset in offsets) + tiles_covered(entry, n) * groups
        padding["scales"] = max(0, end - counts["scales"])
    if any(counts[name] + padding.get(name, 0) > MAX_ELEMENTS for name in counts):
        raise ValueError(f"{LABEL} buffers exceed the index bound")
    return {
        "mode": info["mode"],
        "kernel": info["kernel"],
        "M": m,
        "N": n,
        "K": k,
        "batches": batches,
        "overreadPadding": padding,
        "parameters": values,
    }


def tiles_covered(entry, n):
    """The output rows the launch's threadgroups cover."""
    rows = tile_rows(entry)
    return -(-n // rows) * rows
