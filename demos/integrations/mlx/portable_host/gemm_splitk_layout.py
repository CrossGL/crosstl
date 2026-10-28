"""Storage and launch contracts for MLX's split-K steel GEMM route.

``steel_matmul_axpby`` sends a single product with a small ``M x N`` tile
count and a long ``K`` to ``steel_gemm_splitk_axpby``: ``gemm_splitk`` writes
one float32 partial product per ``K`` partition into a temporary ``C_split``
and ``gemm_splitk_accum`` sums the partitions into the output. ``Matmul``
never reads ``C``, so the accumulation takes no ``axpby`` epilogue.

matmul.cpp launches the accumulation with ``dispatch_threads`` over exactly
``N x M`` threads; the kernel reads only ``thread_position_in_grid`` and has no
bounds check, so the host covers the same threads with one-thread workgroups.
"""

import ctypes
import re

from demos.integrations.mlx.portable_host import gemm_layout, gemv_layout
from demos.integrations.mlx.portable_host.quantization_layout import ITEM_SIZES
from demos.integrations.mlx.tests.kernels import gemm_splitk_model as model

SOURCE = "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk.metal"
SIMD_SIZE = gemm_layout.SIMD_SIZE
MAX_ELEMENTS = gemm_layout.MAX_ELEMENTS
SIGNED_MAX = gemm_layout.SIGNED_MAX
MAX_GROUPS = gemm_layout.MAX_GROUPS
DTYPES = gemm_layout.DTYPES
ACCUMULATOR = "float32"
BK, WM, WN = 16, 2, 2
MAX_PARTITIONS = 32
ACCUM_WORKGROUP = (1, 1, 1)
PATTERN = re.compile(
    r"steel_gemm_splitk_(nn|nt|tn|tt)_(float32|float16|bfloat16)_float32"
    r"_bm(16|32)_bn(16|32)_bk16_wm2_wn2_MN_([tn])aligned_K_([tn])aligned"
)
ACCUM_PATTERN = re.compile(
    r"steel_gemm_splitk_accum_(float32|float16|bfloat16)_float32"
)
ENTRY_PREFIXES = ("steel_gemm_splitk_",)
# [[buffer(i)]] parameters of each kernel in index order; the host binds all.
PARAMETERS = {
    "splitk": ("A", "B", "C", "params"),
    "accum": ("C_split", "D", "k_partitions", "partition_stride", "ldd"),
}
OUTPUTS = {"splitk": "C", "accum": "D"}
# GEMMSpiltKParams in params.h order; every field is a 32-bit int.
PARAMS_FIELDS = (
    "M",
    "N",
    "K",
    "lda",
    "ldb",
    "ldc",
    "tiles_n",
    "tiles_m",
    "split_k_partitions",
    "split_k_partition_stride",
    "split_k_partition_size",
    "swizzle_log",
    "gemm_k_iterations_aligned",
)
SCALARS = ("k_partitions", "partition_stride", "ldd")
# (indexed storage, GLSL offset) of the accumulation kernels' 64-bit offsets,
# shared with their axpby variants; pointer rebases name no storage.
ACCUM_INDEX_EXPRESSIONS = (
    (None, "gid.x + gid.y * uint64(ldd)"),
    (None, "gid.x * uint64(fdc) + gid.y * uint64(ldc)"),
    ("C_split", "offset"),
    ("C_split", "C_split + offset"),
    ("D", "D + 0"),
    ("C", "C + 0"),
)


def _flag(value):
    return "t" if value else "n"


def signature(entry):
    """The kernel kind, element type and tile of one split-K host entry."""
    if not isinstance(entry, str):
        raise ValueError("Unsupported split-K GEMM entry")
    match = ACCUM_PATTERN.fullmatch(entry)
    if match is not None:
        model_entry = model.ACCUM_BY_NAME.get(entry)
        if model_entry is None or model_entry.axpby:
            raise ValueError("Unsupported split-K accumulation entry")
        return {"kind": "accum", "model": model_entry, "dtype": match.group(1)}
    match = PATTERN.fullmatch(entry)
    model_entry = model.BY_NAME.get(entry) if match is not None else None
    if model_entry is None:
        raise ValueError("Unsupported split-K GEMM entry")
    transpose, dtype, bm, bn, mn, k = match.groups()
    return {
        "kind": "splitk",
        "model": model_entry,
        "dtype": dtype,
        "transposeA": transpose[0] == "t",
        "transposeB": transpose[1] == "t",
        "bm": int(bm),
        "bn": int(bn),
        "bk": BK,
        "wm": WM,
        "wn": WN,
        "mnAligned": mn == "t",
        "kAligned": k == "t",
    }


def workgroup_size(entry):
    info = signature(entry)
    if info["kind"] == "accum":
        return ACCUM_WORKGROUP
    return (SIMD_SIZE, info["wn"], info["wm"])


def tile(m, n):
    """steel_gemm_splitk_axpby's (bm, bn) for an ``M x N`` output."""
    return (16 if m < 40 else 32), (16 if n < 40 else 32)


def partition_plan(m, n, k):
    """matmul.cpp's partition count, aligned iterations and partition size."""
    return model.partition_plan(m, n, k, BK)


def select(dtype, transpose_a, transpose_b, m, n, k):
    """The ``gemm_splitk`` and accumulation names matmul.cpp builds."""
    if dtype not in DTYPES:
        raise ValueError("Split-K products support float32, float16 and bfloat16")
    bm, bn = tile(m, n)
    transpose = _flag(transpose_a) + _flag(transpose_b)
    mn_aligned = m % bm == 0 and n % bn == 0
    entry = (
        f"steel_gemm_splitk_{transpose}_{dtype}_{ACCUMULATOR}"
        f"_bm{bm}_bn{bn}_bk{BK}_wm{WM}_wn{WN}"
        f"_MN_{_flag(mn_aligned)}aligned_K_{_flag(k % BK == 0)}aligned"
    )
    return entry, f"steel_gemm_splitk_accum_{dtype}_{ACCUMULATOR}"


def splitk_params(m, n, k, lda, ldb):
    """GEMMSpiltKParams as steel_gemm_splitk_axpby fills it."""
    bm, bn = tile(m, n)
    partitions, iterations, size = partition_plan(m, n, k)
    return {
        "M": m,
        "N": n,
        "K": k,
        "lda": lda,
        "ldb": ldb,
        "ldc": n,
        "tiles_n": -(-n // bn),
        "tiles_m": -(-m // bm),
        "split_k_partitions": partitions,
        "split_k_partition_stride": m * n,
        "split_k_partition_size": size,
        "swizzle_log": 0,
        "gemm_k_iterations_aligned": iterations,
    }


def route(dtype, a_shape, a_strides, b_shape, b_strides):
    """Matmul::eval_gpu's split-K route; raises for every other route."""
    prepared = gemv_layout.operands(a_shape, a_strides, b_shape, b_strides)
    m, n, k = prepared["M"], prepared["N"], prepared["K"]
    if prepared["dot"] or min(m, n) == 1:
        raise ValueError("Matrix-vector products take the gemv route")
    if not gemm_layout.split_k(m, n, k, prepared["batches"]):
        raise ValueError("The product takes the fused steel GEMM route")
    entry, accum = select(
        dtype, prepared["transposeA"], prepared["transposeB"], m, n, k
    )
    params = splitk_params(m, n, k, prepared["lda"], prepared["ldb"])
    partitions = params["split_k_partitions"]
    return {
        "entry": entry,
        "accum": accum,
        "M": m,
        "N": n,
        "K": k,
        "batches": 1,
        "partitions": partitions,
        "params": params,
        "accumParams": {
            "k_partitions": partitions,
            "partition_stride": m * n,
            "ldd": n,
        },
        "copies": prepared["copies"],
    }


def parameter_type(entry, name):
    info = signature(entry)
    if name in ("C", "C_split"):
        return ACCUMULATOR
    if name in ("A", "B", "D"):
        return info["dtype"]
    return "int32"


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters.

    Returns ``(counts, parameters)``: element counts of the array buffers and
    the integer values of the others, ``params`` as a field map.
    """
    info = signature(entry)
    names = PARAMETERS[info["kind"]]
    output = OUTPUTS[info["kind"]]
    if set(buffers) != set(names):
        raise ValueError("Split-K GEMM buffers do not match the entry")
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    counts, parameters, spans = {}, {}, {}
    for name in names:
        buffer, dtype = buffers[name], parameter_type(entry, name)
        itemsize = ITEM_SIZES.get(dtype) or ctypes.sizeof(ctypes.c_int32)
        count = buffer.count
        limit = {"params": len(PARAMS_FIELDS)}.get(
            name, 1 if name in SCALARS else MAX_ELEMENTS
        )
        if (
            not buffer.data
            or buffer.data % itemsize
            or buffer.dtype != dtype.encode("ascii")
            or buffer.output != int(name == output)
            or type(count) is not int
            or not 1 <= count <= limit
            or buffer.data + count * itemsize > address_limit
        ):
            raise ValueError("Split-K GEMM buffer storage or direction does not match")
        spans[name] = (buffer.data, buffer.data + count * itemsize)
        if dtype != "int32":
            counts[name] = count
            continue
        values = list(
            ctypes.cast(buffer.data, ctypes.POINTER(ctypes.c_int32 * count)).contents
        )
        parameters[name] = values[0] if name in SCALARS else values
    if "params" in parameters:
        if len(parameters["params"]) != len(PARAMS_FIELDS):
            raise ValueError("Split-K GEMM parameters do not match GEMMSpiltKParams")
        parameters["params"] = dict(zip(PARAMS_FIELDS, parameters["params"]))
    low, high = spans[output]
    if any(
        other != output and max(low, start) < min(high, end)
        for other, (start, end) in spans.items()
    ):
        raise ValueError("Split-K GEMM output overlaps another buffer")
    return counts, parameters


def _check_launch(execution, count, size):
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError("Split-K GEMM launch geometry is missing")
    if (
        list(execution["workgroupSize"]) != list(size)
        or list(execution["workgroupCount"]) != list(count)
        or not max(count) <= MAX_GROUPS
    ):
        raise ValueError("Split-K GEMM launch does not match MLX's grid")


def _plan_splitk(entry, info, counts, parameters, threads, execution):
    params = parameters.get("params")
    if not isinstance(params, dict) or set(params) != set(PARAMS_FIELDS):
        raise ValueError("Split-K GEMM parameters are missing")
    if any(type(value) is not int for value in params.values()):
        raise ValueError("Split-K GEMM parameters must be integers")
    m, n, k = params["M"], params["N"], params["K"]
    lda, ldb = params["lda"], params["ldb"]
    if not 1 <= min(m, n, k) <= max(m, n, k) <= MAX_ELEMENTS:
        raise ValueError("Split-K GEMM dimensions are invalid")
    minor_a = m if info["transposeA"] else k
    minor_b = k if info["transposeB"] else n
    if (
        params != splitk_params(m, n, k, lda, ldb)
        or not minor_a <= lda <= MAX_ELEMENTS
        or not minor_b <= ldb <= MAX_ELEMENTS
    ):
        raise ValueError("Split-K GEMM parameters differ from matmul.cpp's")
    if not gemm_layout.split_k(m, n, k, 1):
        raise ValueError("matmul.cpp would not split this product")
    selected, _ = select(info["dtype"], info["transposeA"], info["transposeB"], m, n, k)
    if selected != entry:
        raise ValueError("Split-K GEMM entry differs from MLX's routing")
    partitions = params["split_k_partitions"]
    _check_launch(
        execution,
        (params["tiles_n"], params["tiles_m"], partitions),
        workgroup_size(entry),
    )
    if threads != partitions * m * n or counts["C"] != threads:
        raise ValueError("Split-K GEMM partition extent differs")
    extents = {
        "A": (k - 1) * lda + m if info["transposeA"] else (m - 1) * lda + k,
        "B": (n - 1) * ldb + k if info["transposeB"] else (k - 1) * ldb + n,
    }
    for name, extent in extents.items():
        if extent > counts[name]:
            raise ValueError(f"Split-K GEMM {name} exceeds its buffer")
    return {
        "kind": "splitk",
        "M": m,
        "N": n,
        "K": k,
        "lda": lda,
        "ldb": ldb,
        "partitions": partitions,
    }


def _plan_accum(entry, counts, parameters, threads, execution):
    if set(parameters) != set(SCALARS) or any(
        type(value) is not int for value in parameters.values()
    ):
        raise ValueError("Split-K accumulation parameters are missing")
    count = execution.get("workgroupCount") if isinstance(execution, dict) else None
    if not isinstance(count, (list, tuple)) or len(count) != 3:
        raise ValueError("Split-K GEMM launch geometry is missing")
    n, m = count[0], count[1]
    _check_launch(execution, (n, m, 1), ACCUM_WORKGROUP)
    partitions = parameters["k_partitions"]
    if (
        not 2 <= partitions <= MAX_PARTITIONS
        or parameters["partition_stride"] != m * n
        or parameters["ldd"] != n
        or threads != m * n
        or counts["D"] != threads
        or counts["C_split"] != partitions * m * n
    ):
        raise ValueError("Split-K accumulation differs from matmul.cpp's")
    return {"kind": "accum", "M": m, "N": n, "partitions": partitions}


def plan(entry, counts, parameters, threads, execution):
    """Validate one launch against MLX's routing and return its metadata."""
    info = signature(entry)
    if info["kind"] == "accum":
        return _plan_accum(entry, counts, parameters, threads, execution)
    return _plan_splitk(entry, info, counts, parameters, threads, execution)


def loader_access_assertions(entry):
    """The threadgroup slots each ``BlockLoader::load_unsafe`` writes."""
    info = signature(entry)
    if info["kind"] == "accum":
        return ()
    return gemm_layout.model_loader_assertions(info["model"], SOURCE)


def index_assertions(entry):
    """OpenGL ranges for the 64-bit offsets each kernel forms.

    Every bound buffer stays below MAX_ELEMENTS elements. A loader reads
    ``reads`` adjacent elements past its source offset and each threadgroup
    view reaches only the slots its loader writes; the accumulation's
    ``offset`` is ``i * partition_stride`` for ``i < k_partitions``, inside
    ``C_split``.
    """
    info = signature(entry)
    if info["kind"] == "accum":
        return tuple(
            {
                "source": SOURCE,
                "expression": expression,
                **({"indexed": indexed} if indexed else {}),
                "minimum": 0,
                "maximum": SIGNED_MAX,
            }
            for indexed, expression in ACCUM_INDEX_EXPRESSIONS
        )
    element = {"float32": "float", "float16": "float16", "bfloat16": "bfloat16"}[
        info["dtype"]
    ]
    reads = max(loader.reads for loader in info["model"].loaders)
    device = [
        {"expression": f"source.offset + {offset}", "maximum": SIGNED_MAX}
        for offset in range(reads)
    ] + [
        {
            "expression": "reference.offset + index",
            "function": f"crosstl_resource_{access}",
            "maximum": SIGNED_MAX,
        }
        for access in sorted({f"load_{element}", "store_float"})
    ]
    workgroup = [
        {
            "expression": "reference.offset + index",
            "indexed": f"crosstl_workgroup_{name}",
            "maximum": loader["maximum"],
        }
        for name, loader in zip(("As", "Bs"), loader_access_assertions(entry))
    ]
    return tuple(
        {"source": SOURCE, **item, "minimum": 0} for item in device + workgroup
    )
