"""Storage and launch contracts for MLX's block-scaled gathered quantized products.

``GatherQMM::eval_gpu`` in ``gather_qmm.inc`` routes mxfp4, mxfp8 and nvfp4
products as it routes affine ones (``gather_qmm_layout``), through
fp_quantized.metal's ``fp_gather_qmv_fast``, ``fp_gather_qmv``,
``fp_gather_qvm``, ``fp_gather_qmm_t`` and ``fp_gather_qmm_n``: uint32 weights
and one uint8 scale per group, with no biases, on the affine kernels' grids
and threadgroups. ops.cpp ``gather_qmm`` lets nvfp4 products pass one float32
global scale per expert, shaped as the weights' batch. quantized.cpp then
selects the ``_hgs`` entries and binds the global scale at the biases' slot,
and fp_quantized.h ``adjust_matrix_offsets`` moves it to the gathered
expert's, ``global_scale + w_idx``. Without a global scale the kernels still
declare the argument, which they never read: quantized.cpp leaves its slot
unbound and the host binds an inert value.
"""

import re

from demos.integrations.mlx.portable_host import (
    fp_qmm_layout,
    gather_qmm_layout,
    matmul_layout,
    qmm_layout,
)
from demos.integrations.mlx.portable_host.fp_matmul_layout import (
    INACTIVE_GLOBAL_SCALE,
    MODES,
)
from demos.integrations.mlx.portable_host.gather_qmm_layout import (
    GATHER_PARAMETERS,
    GATHER_VECTORS,
    INDEX_DTYPE,
    INDEX_RECORD_LIMIT,
    INDICES,
    MATRIX_KERNELS,
    MATRIX_SCALARS,
    MAX_ELEMENTS,
    SIGNED_MAX,
    SIMD_SIZE,
    TRANSPOSED_KERNELS,
    VECTOR_SCALARS,
)
from demos.integrations.mlx.portable_host.quantization_layout import SOURCE_TYPES

SOURCE = "mlx/backend/metal/kernels/fp_quantized.metal"
LABEL = "Block-scaled gathered quantized product"
PATTERN = re.compile(
    r"(mxfp4|mxfp8|nvfp4)_gather_(qmv_fast|qmv|qvm|qmm_t|qmm_n)"
    r"_(float|float16_t|bfloat16_t)_gs_(16|32)_b_(4|8)"
    r"(?:_alN_(true|false))?(_hgs)?"
)
# fp_quantized.metal's gather_qmm_rhs entries share these prefixes; their
# names never match PATTERN, so the host refuses them.
ENTRY_PREFIXES = tuple(f"{mode}_gather_" for mode in MODES)
# ops.cpp gather_qmm accepts a global scale for nvfp4 alone.
GLOBAL_SCALE_MODES = ("nvfp4",)
GLOBAL_SCALE = "global_scale"
MATRIX_PARAMETERS = ("w", "scales", "x", *INDICES, "y")
# fp_quantized.h adjust_matrix_offsets moves the weights and scales by w_idx.
WEIGHT_VIEWS = ("w_strides", "s_strides")
BATCH_PARAMETERS = fp_qmm_layout.BATCH_PARAMETERS
# Expressions OpenGL may evaluate in 32 bits in the matrix-vector kernels:
# adjust_matrix_offsets' index locations, matrix offsets and, with a global
# scale, its expert offset. ``plan`` proves each names an element of a
# buffer of at most MAX_ELEMENTS elements.
GATHER_INDEX_EXPRESSIONS = (
    "tid.z * lhs_strides[0]",
    "tid.z * rhs_strides[0]",
    "idx.x",
    "idx.y",
    "x_idx * x_strides[0]",
    "w_idx * w_strides[0]",
    "w_idx * s_strides[0]",
)


def signature(entry):
    """The mode, kernel, element type, group size, bits, alignment and global scale."""
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError(f"Unsupported {LABEL.lower()} entry")
    mode, kernel, source, group, bits, aligned, scaled = match.groups()
    # Only fp_gather_qmm_t names whether N is whole tiles.
    if (
        MODES[mode] != (int(group), int(bits))
        or (aligned is None) != (kernel != "qmm_t")
        or (scaled is not None and mode not in GLOBAL_SCALE_MODES)
    ):
        raise ValueError(f"{LABEL} entry is never routed")
    return {
        "mode": mode,
        "kernel": kernel,
        "dtype": next(key for key, value in SOURCE_TYPES.items() if value == source),
        "groupSize": int(group),
        "bits": int(bits),
        "aligned": None if aligned is None else aligned == "true",
        "transpose": kernel in TRANSPOSED_KERNELS,
        "globalScale": scaled is not None,
    }


def is_matrix(entry):
    return signature(entry)["kernel"] in MATRIX_KERNELS


def workgroup_size(entry):
    return (SIMD_SIZE, 2, 2) if is_matrix(entry) else (SIMD_SIZE, 2, 1)


def tile_columns(entry):
    """Outputs per threadgroup along ``N``, as for the affine kernels."""
    info = signature(entry)
    if info["kernel"] in MATRIX_KERNELS:
        return qmm_layout.BN
    if info["kernel"] == "qvm":
        return 2 * min(info["groupSize"], SIMD_SIZE)
    return 8


def select(mode, dtype, m, n, k, transpose=True, global_scale=False):
    """quantized.cpp GatherQMM::eval_gpu's entry for an unsorted product.

    ``gather_qmm.inc`` refuses the sorted single-row route, ``gather_qmm_rhs``.
    """
    group, bits = MODES[mode]
    if global_scale and mode not in GLOBAL_SCALE_MODES:
        raise ValueError(f"{LABEL} global scales require nvfp4")
    stem = f"{SOURCE_TYPES[dtype]}_gs_{group}_b_{bits}"
    scaled = "_hgs" if global_scale else ""
    if m >= matmul_layout.vector_limit(k, n, transpose):
        if not transpose:
            return f"{mode}_gather_qmm_n_{stem}{scaled}"
        aligned = "true" if n % qmm_layout.BN == 0 else "false"
        return f"{mode}_gather_qmm_t_{stem}_alN_{aligned}{scaled}"
    if not transpose:
        return f"{mode}_gather_qvm_{stem}{scaled}"
    fast = n % 8 == 0 and k % matmul_layout.fast_alignment(bits) == 0
    return f"{mode}_gather_{'qmv_fast' if fast else 'qmv'}_{stem}{scaled}"


def scalars(entry):
    return MATRIX_SCALARS if is_matrix(entry) else VECTOR_SCALARS


def matrix_parameters(entry):
    """The bound arrays, in quantized.cpp's order: the global scale at slot 2."""
    if signature(entry)["globalScale"]:
        return ("w", "scales", GLOBAL_SCALE, *MATRIX_PARAMETERS[2:])
    return MATRIX_PARAMETERS


def parameter_names(entry):
    """quantized.cpp's buffers, in its binding order: every launch binds all."""
    return (
        *matrix_parameters(entry),
        *scalars(entry),
        *BATCH_PARAMETERS,
        *GATHER_PARAMETERS,
    )


def parameter_type(entry, name):
    if name == "w" or name in INDICES:
        return INDEX_DTYPE
    if name == "scales":
        return "uint8"
    if name == GLOBAL_SCALE:
        return "float32"
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
    """Check native buffer identity and read the integer and index parameters."""
    return gather_qmm_layout.read_gathered(
        buffers,
        {name: parameter_type(entry, name) for name in parameter_names(entry)},
        scalars(entry),
        matrix_parameters(entry),
        LABEL,
    )


def _dimensions(entry, info, parameters, execution):
    if is_matrix(entry):
        return tuple(parameters.get(name) for name in MATRIX_SCALARS)
    k, n = (parameters.get(name) for name in VECTOR_SCALARS)
    count = execution.get("workgroupCount") if isinstance(execution, dict) else None
    m = count[0] if isinstance(count, (list, tuple)) and count else None
    return k, n, m


def plan(entry, counts, parameters, threads, execution):
    """Validate one launch against MLX's routing and return its metadata.

    The returned ``parameters`` are the bound integer parameters without the
    index values, plus the inert global scale of entries that never read one.
    ``matrixIndices`` lists each ``tid.z``'s ``(x, w)`` matrices for launches
    of at most ``INDEX_RECORD_LIMIT`` batches.
    """
    info = signature(entry)
    group, bits, transpose = info["groupSize"], info["bits"], info["transpose"]
    k, n, m = _dimensions(entry, info, parameters, execution)
    if (
        any(type(value) is not int for value in (k, n, m))
        or not 1 <= min(k, n, m) <= max(k, n, m) <= MAX_ELEMENTS
        or (k if transpose else n) % group
    ):
        raise ValueError(f"{LABEL} dimensions are invalid")
    batches = gather_qmm_layout.grid_batches(
        execution,
        workgroup_size(entry),
        is_matrix(entry),
        m,
        n,
        tile_columns(entry),
        LABEL,
    )
    if (
        select(info["mode"], info["dtype"], m, n, k, transpose, info["globalScale"])
        != entry
    ):
        raise ValueError(f"{LABEL} entry differs from MLX's routing")
    if threads != m * n * batches or counts["y"] != threads:
        raise ValueError(f"{LABEL} output extent differs")
    # Groups run along the weights' rows: K when transposed, N otherwise.
    rows, columns = (n, k) if transpose else (k, n)
    words, groups = columns * bits // 32, columns // group
    indices, offsets, experts = gather_qmm_layout.gathered_offsets(
        parameters,
        counts,
        m,
        k,
        rows,
        words,
        groups,
        batches,
        WEIGHT_VIEWS,
        LABEL,
    )
    extents = {"x": m * k, "w": rows * words, "scales": rows * groups}
    for column, name in enumerate(("x", "w", "scales")):
        if max(offset[column] for offset in offsets) + extents[name] > counts[name]:
            raise ValueError(f"{LABEL} {name} matrix exceeds its buffer")
    # ops.cpp shapes the global scale as the weights' batch: one per expert.
    if info["globalScale"] and counts[GLOBAL_SCALE] != experts:
        raise ValueError(f"{LABEL} global scale does not hold one value per expert")
    values = {name: value for name, value in parameters.items() if name not in INDICES}
    if not info["globalScale"]:
        values[GLOBAL_SCALE] = [INACTIVE_GLOBAL_SCALE]
    return {
        "mode": info["mode"],
        "kernel": info["kernel"],
        "globalScale": info["globalScale"],
        "M": m,
        "N": n,
        "K": k,
        "batches": batches,
        "matrixIndices": (
            [list(pair) for pair in indices] if batches <= INDEX_RECORD_LIMIT else None
        ),
        "overreadPadding": {"scales": 0},
        "parameters": values,
    }


def sibling(entry):
    """The batched block-scaled qmm entry whose tiles a gathered qmm entry shares."""
    info = signature(entry)
    stem = (
        f"{info['mode']}_{{}}_{SOURCE_TYPES[info['dtype']]}"
        f"_gs_{info['groupSize']}_b_{info['bits']}"
    )
    if info["kernel"] == "qmm_t":
        return stem.format("qmm_t") + "_alN_true_batch_1"
    if info["kernel"] == "qmm_n":
        return stem.format("qmm_n") + "_batch_1"
    raise ValueError("Gathered matrix-vector entries share no qmm tiles")


def access_assertions(entry):
    """The qmm x loader's ``dst`` slots; matrix-vector kernels have none."""
    if not is_matrix(entry):
        return ()
    return tuple(
        {**assertion, "entryPoint": entry}
        for assertion in fp_qmm_layout.access_assertions(sibling(entry))
    )


def index_assertions(entry):
    """OpenGL ranges for the 64-bit offsets each kernel forms.

    Matrix-vector kernels bound adjust_matrix_offsets' expressions by
    ``MAX_ELEMENTS``. qmm kernels take their batched siblings' resource and
    tile ranges, and each lhs or rhs index element must fit a 32-bit index;
    so must the global scale element ``_hgs`` kernels load as a float.
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
    items = list(fp_qmm_layout.index_assertions(sibling(entry)))
    tiles = [item for item in items if "indexed" in item]
    items = [item for item in items if "indexed" not in item]
    helpers = ["crosstl_resource_load_uint"]
    info = signature(entry)
    if info["globalScale"] and info["dtype"] != "float32":
        helpers.append("crosstl_resource_load_float")
    items += [
        {
            "source": SOURCE,
            "expression": "reference.offset + index",
            "function": helper,
            "minimum": 0,
            "maximum": SIGNED_MAX,
        }
        for helper in helpers
    ]
    return tuple(items + tiles)


def bounded_parameters(entry):
    """The scalar parameters each qmm translation bounds by ``K_LIMIT``."""
    return ("K",) if is_matrix(entry) else ()
