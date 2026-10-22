"""Storage and launch contracts for MLX's affine quantized matrix-matrix kernels.

``QuantizedMatmul::eval_gpu`` in ``backend.cpp`` sends products with at least
``matmul_layout.vector_limit`` rows to quantized.cpp's matrix-matrix kernels.
The portable host has no NAX unit, so ``qmm`` never takes ``qmm_nax``.
Unbatched products with transposed weights go to ``qmm_splitk``:
``affine_qmm_t_splitk`` writes ``split_k`` partial products over partitions of
``K`` to a partition-major intermediate that a column sum then adds, or, when
``K`` admits no split, ``affine_qmm_t`` computes the product. Batched
transposed products go to ``affine_qmm_t`` and untransposed ones to
``affine_qmm_n``. Every kernel runs ``(32, 2, 2)`` threadgroups over 32 by 32
output tiles.
"""

import re
from dataclasses import dataclass

from demos.integrations.mlx.portable_host import matmul_layout
from demos.integrations.mlx.portable_host.quantization_layout import SOURCE_TYPES
from demos.integrations.mlx.tests.kernels.gemm_fused_model import Loader

SOURCE = "mlx/backend/metal/kernels/quantized.metal"
LABEL = "Quantized matrix-matrix"
SIMD_SIZE = 32
# qmm_t_impl's and qmm_n_impl's tile and simdgroup grid.
BM = BN = BK = 32
WM = WN = 2
THREADS = WM * WN * SIMD_SIZE
# qmm_n_impl's loader_x_t reads four adjacent activations per thread.
QMM_N_X_READS = 4
MAX_ELEMENTS = matmul_layout.MAX_ELEMENTS
MAX_GROUPS = matmul_layout.MAX_GROUPS
SIGNED_MAX = 2**31 - 1
# The loop ``k < K_eff; k += BK`` stays in int32 below this K.
K_LIMIT = (1 << 31) - 64
# qmm_splitk chooses split_k to target about this many threadgroups.
SPLIT_TARGET = 512
PATTERN = re.compile(
    r"affine_(qmm_t_splitk|qmm_t|qmm_n)_(float|float16_t|bfloat16_t)"
    r"_gs_(32|64|128)_b_(2|3|4|5|6|8)(?:_alN_(true|false))?(?:_batch_(0|1))?"
)
ENTRY_PREFIXES = ("affine_qmm_t_", "affine_qmm_n_")
MATRIX_PARAMETERS = matmul_layout.MATRIX_PARAMETERS
SCALARS = ("K", "N", "M")
SPLIT_SCALARS = ("k_partition_size", "split_k_partition_stride")
BATCH_PARAMETERS = matmul_layout.BATCH_PARAMETERS
HELPER_ELEMENTS = {"float32": "float", "float16": "float16", "bfloat16": "bfloat16"}


@dataclass(frozen=True)
class VectorLoader(Loader):
    """A ``BlockLoader`` whose ``n_reads`` template argument is explicit."""

    vector: int = QMM_N_X_READS

    @property
    def reads(self):
        return self.vector


def signature(entry):
    """The kernel, element type, group size, bits, alignment and batching."""
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError("Unsupported quantized matrix-matrix entry")
    kernel, source, group, bits, aligned, batched = match.groups()
    # qmm_t names alignment and batching, qmm_t_splitk only alignment (it runs
    # for single matrices) and qmm_n only batching (N is whole groups).
    if (aligned is None) != (kernel == "qmm_n") or (batched is None) != (
        kernel == "qmm_t_splitk"
    ):
        raise ValueError("Unsupported quantized matrix-matrix entry")
    return {
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


def split_k(m, n, k, group):
    """quantized.cpp qmm_splitk's partition count for one unbatched product.

    Each partition is whole BK-wide tiles and whole quantization groups; 1
    sends the product to qmm_t.
    """
    split = max(1, SPLIT_TARGET // (-(-n // BN) * -(-m // BM)))
    align = group if group > BK else BK
    split = min(split, k // align)
    while split > 1 and k % (split * align):
        split -= 1
    return split


def select(dtype, group, bits, m, n, k, batches, transpose=True):
    """quantized.cpp QuantizedMatmul::eval_gpu's matrix-matrix entry."""
    if m < matmul_layout.vector_limit(k, n, transpose):
        raise ValueError("Quantized matrix-vector products take the qmv and qvm routes")
    stem = f"{SOURCE_TYPES[dtype]}_gs_{group}_b_{bits}"
    batch = "_batch_1" if batches > 1 else "_batch_0"
    if not transpose:
        return f"affine_qmm_n_{stem}{batch}"
    aligned = "_alN_true" if n % BN == 0 else "_alN_false"
    if batches == 1 and split_k(m, n, k, group) > 1:
        return f"affine_qmm_t_splitk_{stem}{aligned}"
    return f"affine_qmm_t_{stem}{aligned}{batch}"


def parameter_names(entry):
    """The parameters backend.cpp binds, as quantized.cpp binds them.

    add_strides_and_shapes skips the batch parameters of unbatched launches.
    """
    info = signature(entry)
    names = (*MATRIX_PARAMETERS, *SCALARS)
    if info["kernel"] == "qmm_t_splitk":
        return (*names, *SPLIT_SCALARS)
    return (*names, *BATCH_PARAMETERS) if info["batched"] else names


def parameter_type(entry, name):
    if name == "w":
        return "uint32"
    if name in MATRIX_PARAMETERS:
        return signature(entry)["dtype"]
    if name in SCALARS or name in SPLIT_SCALARS or name in matmul_layout.BATCH_SCALARS:
        return "int32"
    return matmul_layout.BATCH_VECTORS[name]


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters."""
    names = parameter_names(entry)
    return matmul_layout.read_buffers(
        buffers,
        {name: parameter_type(entry, name) for name in names},
        (*SCALARS, *SPLIT_SCALARS, *matmul_layout.BATCH_SCALARS),
        LABEL,
    )


def launch_depth(execution, m, n, label=LABEL):
    """The grid depth of a ``(ceil(N / BN), ceil(M / BM), depth)`` launch.

    Every affine and block-scaled qmm kernel runs ``(32, 2, 2)`` threadgroups.
    """
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError(f"{label} launch geometry is missing")
    count = list(execution["workgroupCount"])
    if (
        list(execution["workgroupSize"]) != [SIMD_SIZE, WN, WM]
        or len(count) != 3
        or any(type(value) is not int for value in count)
        or count[:2] != [-(-n // BN), -(-m // BM)]
        or not 1 <= min(count) <= max(count) <= MAX_GROUPS
    ):
        raise ValueError(f"{label} launch does not match MLX's grid")
    return count[2]


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
        "b_strides": [groups, 1],
    }


def plan(entry, counts, parameters, threads, execution):
    """Validate one launch against MLX's routing and return its metadata.

    The returned ``parameters`` add the inert batch values unbatched qmm_t
    and qmm_n kernels reflect but never read.
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
    depth = launch_depth(execution, m, n)
    split = depth if kernel == "qmm_t_splitk" else 1
    batches = depth // split
    if select(info["dtype"], group, bits, m, n, k, batches, transpose) != entry or (
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
        if supplied != set(SPLIT_SCALARS) or (
            parameters["k_partition_size"],
            parameters["split_k_partition_stride"],
        ) != (k // split, m * n):
            raise ValueError(f"{LABEL} split-K partitions differ from quantized.cpp's")
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
    extents = {
        "x": m * k,
        "w": rows * words,
        "scales": rows * groups,
        "biases": rows * groups,
    }
    for column, name in enumerate(("x", "w", "scales", "biases")):
        if max(offset[column] for offset in offsets) + extents[name] > counts[name]:
            raise ValueError(f"{LABEL} {name} batch exceeds its buffer")
    return {
        "kernel": kernel,
        "M": m,
        "N": n,
        "K": k,
        "batches": batches,
        "splitK": split,
        "overreadPadding": {"scales": 0, "biases": 0},
        "parameters": values,
    }


def padded(dtype):
    """qmm_t_impl's and qmm_n_impl's BK_padded: BK plus 16 bytes of elements."""
    return BK + 16 // (4 if dtype == "float32" else 2)


def x_loader(entry, info=None):
    """The activation ``BlockLoader`` the translation materializes.

    ``info`` is the entry's signature; the block-scaled kernels share these
    tiles, so ``fp_qmm_layout`` passes its own.
    """
    info = signature(entry) if info is None else info
    itype, width = SOURCE_TYPES[info["dtype"]], padded(info["dtype"])
    name = f"BlockLoader_{itype}_{BM}_{BK}_{width}_1_{THREADS}"
    if info["kernel"] == "qmm_n":
        return VectorLoader(
            f"{name}_1_{QMM_N_X_READS}__load_unsafe", BM, BK, width, THREADS
        )
    return Loader(f"{name}__load_unsafe", BM, BK, width, THREADS)


def access_assertions(entry, info=None, source=SOURCE):
    """The x loader's ``dst`` stays inside the slots ``BlockLoader`` writes."""
    loader = x_loader(entry, info)
    slots = [slot for lane in loader.slots() for slot in lane]
    return (
        {
            "source": source,
            "entryPoint": entry,
            "function": loader.function,
            "parameter": "self.dst",
            "minimum": min(slots),
            "maximum": max(slots),
        },
    )


def index_assertions(entry, info=None, source=SOURCE):
    """OpenGL ranges for the 64-bit offsets the kernel forms.

    The x loader reads its whole elements past each source offset; the
    resource helpers index elements, weight bytes and, for kernels that bind
    batch strides, int64 strides of buffers below ``MAX_ELEMENTS`` elements;
    and each threadgroup tile view reaches only its ``BM`` padded rows.
    """
    info = signature(entry) if info is None else info
    element = HELPER_ELEMENTS[info["dtype"]]
    reads = x_loader(entry, info).reads
    helpers = [f"load_{element}", f"store_{element}", "load_uint8"]
    if info["kernel"] != "qmm_t_splitk":
        helpers.append("load_int64_t")
    last = (BM - 1) * padded(info["dtype"]) + BK - 1
    items = [
        ({"expression": f"source.offset + {index}"}, SIGNED_MAX)
        for index in range(reads)
    ]
    items += [
        (
            {
                "expression": "reference.offset + index",
                "function": f"crosstl_resource_{helper}",
            },
            SIGNED_MAX,
        )
        for helper in helpers
    ]
    items += [
        (
            {
                "expression": "reference.offset + index",
                "indexed": f"crosstl_workgroup_{tile}",
            },
            last,
        )
        for tile in ("Xs", "Ws")
    ]
    return tuple(
        {"source": source, **scope, "minimum": 0, "maximum": maximum}
        for scope, maximum in items
    )


def bounded_parameters(entry, info=None):
    """The scalar parameters each translation bounds by ``K_LIMIT``.

    qmm_t_splitk's K loop runs to k_partition_size; its rows still stride by K.
    """
    if (signature(entry) if info is None else info)["kernel"] == "qmm_t_splitk":
        return ("K", "k_partition_size")
    return ("K",)


def partition_sum(m, n, split):
    """The column sum reduce.cpp runs over a split-K product's partitions.

    ``strided_reduce_general_dispatch`` over ``split`` partitions of ``M * N``
    outputs: the small pass below 32 partitions, the looped pass to 256, and
    the two-pass reduction above when the output holds under 32,768 values.
    """
    if split < 32:
        return "small"
    if split <= 256 or m * n // 32 >= 1024:
        return "looped"
    return "two-pass"
