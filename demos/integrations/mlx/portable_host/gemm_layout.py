"""Storage and launch contracts for MLX's steel GEMM matrix-matrix kernel.

``Matmul::eval_gpu`` in ``backend.cpp`` sends products whose output has more
than one row and column to ``steel_matmul`` as matmul.cpp does, after the
operand preparation ``gemv_layout.operands`` mirrors. The portable host
reports no Apple GPU architecture and no NAX unit, so ``steel_matmul_axpby``
compares against the split-K threshold of 1,024 tiles and otherwise dispatches
``steel_gemm_fused`` with ``GEMM_TPARAM_MACRO``'s medium-device tile: 64x64x16
with 2x2 simdgroups. ``Matmul`` never reads ``C``, so ``use_out_source`` and
``do_axpby`` stay false; the entry names carry the remaining function
constants as matmul.cpp's hash name does. Split-K products take
``gemm_splitk_layout``'s route.
"""

import ctypes
import math
import re

from demos.integrations.mlx.portable_host import gemv_layout
from demos.integrations.mlx.portable_host.quantization_layout import ITEM_SIZES
from demos.integrations.mlx.tests.kernels import gemm_fused_model as model

SOURCE = "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal"
SIMD_SIZE = 32
MAX_ELEMENTS = gemv_layout.MAX_ELEMENTS
SIGNED_MAX = gemv_layout.SIGNED_MAX
MAX_GROUPS = gemv_layout.MAX_GROUPS
MAX_BATCH_DIMENSIONS = gemv_layout.MAX_BATCH_DIMENSIONS
DTYPES = gemv_layout.DTYPES
# GEMM_TPARAM_MACRO's medium-device (bm, bn, bk, wm, wn).
HOST_TILE = (64, 64, 16, 2, 2)
# steel_matmul_axpby's min_tmn_threshold for devices other than 's' and 'd'.
SPLIT_K_TILES = 1024
CONSTANTS = ("has_batch", "use_out_source", "do_axpby", "align_M", "align_N", "align_K")
CONSTANT_IDS = dict(zip(CONSTANTS, (10, 100, 110, 200, 201, 202)))
PATTERN = re.compile(
    r"(steel_gemm_fused_(nn|nt|tn|tt)_(float32|float16|bfloat16)_\3"
    r"_bm(\d+)_bn(\d+)_bk(\d+)_wm(\d+)_wn(\d+))"
    r"_has_batch_([tn])_use_out_source_([tn])_do_axpby_([tn])"
    r"_align_M_([tn])_align_N_([tn])_align_K_([tn])"
)
ENTRY_PREFIXES = ("steel_gemm_fused_",)
# [[buffer(i)]] parameters of steel_gemm_fused.h's gemm, in index order.
PARAMETERS = (
    "A",
    "B",
    "C",
    "D",
    "params",
    "addmm_params",
    "batch_shape",
    "batch_strides",
)
# What backend.cpp binds. matmul.cpp binds batch_shape and batch_strides only
# when has_batch holds; the host always passes them and the kernel ignores
# them otherwise.
HOST_PARAMETERS = ("A", "B", "D", "params", "batch_shape", "batch_strides")
MATRIX_PARAMETERS = ("A", "B", "D")
# GEMMParams in params.h order; backend.cpp passes each field as an int64.
PARAMS_FIELDS = (
    "M",
    "N",
    "K",
    "lda",
    "ldb",
    "ldd",
    "tiles_n",
    "tiles_m",
    "batch_stride_a",
    "batch_stride_b",
    "batch_stride_d",
    "swizzle_log",
    "gemm_k_iterations_aligned",
    "batch_ndim",
)
VECTOR_TYPES = {"params": "int64", "batch_shape": "int32", "batch_strides": "int64"}
INTEGER_TYPES = {"int32": ctypes.c_int32, "int64": ctypes.c_int64}
# What matmul.cpp would pass in GEMMAddMMParams; the kernel never reads it
# without use_out_source.
INERT_ADDMM = {"ldc": 0, "fdc": 0, "batch_stride_c": 0, "alpha": 1.0, "beta": 0.0}


def _flag(value):
    return "t" if value else "n"


def signature(entry):
    """The source entry, transposition, element type and constants of an entry."""
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError("Unsupported steel GEMM entry")
    base, transpose, dtype, *rest = match.groups()
    tile, flags = tuple(int(value) for value in rest[:5]), rest[5:]
    model_entry = model.BY_NAME.get(base)
    if model_entry is None or tile != HOST_TILE:
        raise ValueError("Steel GEMM entry is not the host's instantiated tile")
    constants = {name: flag == "t" for name, flag in zip(CONSTANTS, flags)}
    if constants["use_out_source"] or constants["do_axpby"]:
        raise ValueError("Matmul never computes a steel GEMM epilogue")
    bm, bn, bk, wm, wn = tile
    return {
        "source": base,
        "model": model_entry,
        "transposeA": transpose[0] == "t",
        "transposeB": transpose[1] == "t",
        "dtype": dtype,
        "bm": bm,
        "bn": bn,
        "bk": bk,
        "wm": wm,
        "wn": wn,
        "constants": constants,
    }


def workgroup_size(entry):
    info = signature(entry)
    return (SIMD_SIZE, info["wn"], info["wm"])


def select(dtype, transpose_a, transpose_b, m, n, k, batch_ndim):
    """steel_matmul_regular_axpby's hash name for one product without C."""
    if dtype not in DTYPES:
        raise ValueError("Steel GEMM products support float32, float16 and bfloat16")
    bm, bn, bk, wm, wn = HOST_TILE
    transpose = ("t" if transpose_a else "n") + ("t" if transpose_b else "n")
    return (
        f"steel_gemm_fused_{transpose}_{dtype}_{dtype}"
        f"_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}"
        f"_has_batch_{_flag(batch_ndim > 1)}_use_out_source_n_do_axpby_n"
        f"_align_M_{_flag(m % bm == 0)}_align_N_{_flag(n % bn == 0)}"
        f"_align_K_{_flag(k % bk == 0)}"
    )


def split_k(m, n, k, batches):
    """steel_matmul_axpby's split-K case for a device without NAX."""
    tiles = math.ceil(m / 16) * math.ceil(n / 16)
    return batches == 1 and tiles <= SPLIT_K_TILES and k // 16 >= 8 and k >= max(m, n)


def route(dtype, a_shape, a_strides, b_shape, b_strides):
    """Matmul::eval_gpu's steel GEMM route; raises for every other route."""
    prepared = gemv_layout.operands(a_shape, a_strides, b_shape, b_strides)
    m, n, k = prepared["M"], prepared["N"], prepared["K"]
    if prepared["dot"] or min(m, n) == 1:
        raise ValueError("Matrix-vector products take the gemv route")
    batches = prepared["batches"]
    if split_k(m, n, k, batches):
        raise ValueError("Split-K matrix products take the split-K route")
    batch_shape = prepared["batchShape"]
    a_batch, b_batch = prepared["aBatchStride"], prepared["bBatchStride"]
    ta, tb = prepared["transposeA"], prepared["transposeB"]
    bm, bn, bk, _, _ = HOST_TILE
    params = {
        "M": m,
        "N": n,
        "K": k,
        "lda": prepared["lda"],
        "ldb": prepared["ldb"],
        "ldd": n,
        "tiles_n": -(-n // bn),
        "tiles_m": -(-m // bm),
        "batch_stride_a": a_batch[-1],
        "batch_stride_b": b_batch[-1],
        "batch_stride_d": m * n,
        "swizzle_log": 0,
        "gemm_k_iterations_aligned": k // bk,
        "batch_ndim": len(batch_shape),
    }
    return {
        "entry": select(dtype, ta, tb, m, n, k, len(batch_shape)),
        "M": m,
        "N": n,
        "K": k,
        "batches": batches,
        "batchShape": batch_shape,
        "batchStrides": list(a_batch) + list(b_batch),
        "params": params,
        "copies": prepared["copies"],
    }


def matmul_route(dtype, a_shape, a_strides, b_shape, b_strides):
    """The gemv, steel GEMM or split-K route Matmul::eval_gpu takes, by family."""
    from demos.integrations.mlx.portable_host import gemm_splitk_layout

    operands = (dtype, a_shape, a_strides, b_shape, b_strides)
    prepared = gemv_layout.operands(a_shape, a_strides, b_shape, b_strides)
    m, n, k = prepared["M"], prepared["N"], prepared["K"]
    if prepared["dot"] or min(m, n) == 1:
        return "gemv", gemv_layout.route(*operands)
    if split_k(m, n, k, prepared["batches"]):
        return "splitk", gemm_splitk_layout.route(*operands)
    return "gemm", route(*operands)


def parameter_type(entry, name):
    if name in MATRIX_PARAMETERS:
        return signature(entry)["dtype"]
    return VECTOR_TYPES[name]


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters.

    Returns ``(counts, parameters)``: element counts of the matrix buffers and
    the integer values of the vector parameters, ``params`` as a field map.
    """
    signature(entry)
    if set(buffers) != set(HOST_PARAMETERS):
        raise ValueError("Steel GEMM buffers do not match the entry")
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    counts, parameters, spans = {}, {}, {}
    for name in HOST_PARAMETERS:
        buffer, dtype = buffers[name], parameter_type(entry, name)
        itemsize = ITEM_SIZES.get(dtype) or ctypes.sizeof(INTEGER_TYPES[dtype])
        count = buffer.count
        limit = {
            "params": len(PARAMS_FIELDS),
            "batch_shape": MAX_BATCH_DIMENSIONS,
            "batch_strides": 2 * MAX_BATCH_DIMENSIONS,
        }.get(name, MAX_ELEMENTS)
        if (
            not buffer.data
            or buffer.data % itemsize
            or buffer.dtype != dtype.encode("ascii")
            or buffer.output != int(name == "D")
            or type(count) is not int
            or not 1 <= count <= limit
            or buffer.data + count * itemsize > address_limit
        ):
            raise ValueError("Steel GEMM buffer storage or direction does not match")
        spans[name] = (buffer.data, buffer.data + count * itemsize)
        if name in MATRIX_PARAMETERS:
            counts[name] = count
            continue
        parameters[name] = list(
            ctypes.cast(
                buffer.data, ctypes.POINTER(INTEGER_TYPES[dtype] * count)
            ).contents
        )
    if len(parameters["params"]) != len(PARAMS_FIELDS):
        raise ValueError("Steel GEMM parameters do not match GEMMParams")
    parameters["params"] = dict(zip(PARAMS_FIELDS, parameters["params"]))
    low, high = spans["D"]
    if any(
        other != "D" and max(low, start) < min(high, end)
        for other, (start, end) in spans.items()
    ):
        raise ValueError("Steel GEMM output overlaps another buffer")
    return counts, parameters


def _batch_end(shape, strides):
    """The largest elem_to_loc offset of nonnegative strides over ``shape``."""
    return sum((extent - 1) * stride for extent, stride in zip(shape, strides))


def _check_params(info, params, shape, strides):
    bm, bn, bk = info["bm"], info["bn"], info["bk"]
    if any(type(value) is not int for value in params.values()):
        raise ValueError("Steel GEMM parameters must be integers")
    m, n, k = params["M"], params["N"], params["K"]
    ndim = params["batch_ndim"]
    if (
        not 1 <= min(m, n, k) <= max(m, n, k) <= MAX_ELEMENTS
        or not 1 <= ndim <= MAX_BATCH_DIMENSIONS
        or len(shape) != ndim
        or len(strides) != 2 * ndim
        or any(extent < 1 for extent in shape)
        or any(stride < 0 for stride in strides)
    ):
        raise ValueError("Steel GEMM dimensions are invalid")
    # The leading dimensions cover the minor axis of each operand's layout.
    minor_a = m if info["transposeA"] else k
    minor_b = k if info["transposeB"] else n
    expected = {
        "ldd": n,
        "tiles_n": -(-n // bn),
        "tiles_m": -(-m // bm),
        "batch_stride_a": strides[ndim - 1],
        "batch_stride_b": strides[2 * ndim - 1],
        "batch_stride_d": m * n,
        "swizzle_log": 0,
        "gemm_k_iterations_aligned": k // bk,
    }
    if (
        any(params[name] != value for name, value in expected.items())
        or not minor_a <= params["lda"] <= MAX_ELEMENTS
        or not minor_b <= params["ldb"] <= MAX_ELEMENTS
    ):
        raise ValueError("Steel GEMM parameters differ from matmul.cpp's")


def plan(entry, counts, parameters, threads, execution):
    """Validate one launch against MLX's routing and return its metadata."""
    info = signature(entry)
    params = parameters.get("params")
    shape = parameters.get("batch_shape")
    strides = parameters.get("batch_strides")
    if not isinstance(params, dict) or set(params) != set(PARAMS_FIELDS):
        raise ValueError("Steel GEMM parameters are missing")
    _check_params(info, params, shape, strides)
    m, n, k, ndim = params["M"], params["N"], params["K"], params["batch_ndim"]
    batches = math.prod(shape)
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError("Steel GEMM launch geometry is missing")
    if (
        list(execution["workgroupSize"]) != list(workgroup_size(entry))
        or list(execution["workgroupCount"])
        != [params["tiles_n"], params["tiles_m"], batches]
        or not max(params["tiles_n"], params["tiles_m"], batches) <= MAX_GROUPS
    ):
        raise ValueError("Steel GEMM launch does not match MLX's grid")
    selected = select(
        info["dtype"], info["transposeA"], info["transposeB"], m, n, k, ndim
    )
    if selected != entry or split_k(m, n, k, batches):
        raise ValueError("Steel GEMM entry differs from MLX's routing")
    if threads != batches * m * n or counts["D"] != threads:
        raise ValueError("Steel GEMM output extent differs")
    # Without has_batch the kernel offsets each batch by GEMMParams' strides,
    # which _check_params ties to the last batch strides.
    a_strides, b_strides = strides[:ndim], strides[ndim:]
    lda, ldb = params["lda"], params["ldb"]
    extents = {
        "A": (k - 1) * lda + m if info["transposeA"] else (m - 1) * lda + k,
        "B": (n - 1) * ldb + k if info["transposeB"] else (k - 1) * ldb + n,
    }
    for name, values in (("A", a_strides), ("B", b_strides)):
        if _batch_end(shape, values) + extents[name] > counts[name]:
            raise ValueError(f"Steel GEMM {name} batch exceeds its buffer")
    return {
        "source": info["source"],
        "M": m,
        "N": n,
        "K": k,
        "lda": lda,
        "ldb": ldb,
        "batches": batches,
        "constants": info["constants"],
        "inert": dict(INERT_ADDMM),
    }


def loader_access_assertions(entry):
    """The threadgroup slots each ``BlockLoader::load_unsafe`` writes.

    ``loader.h`` distributes adjacent elements of a tile row to each thread,
    and ``gemm.h`` pads every threadgroup row by 16 bytes.
    """
    return model_loader_assertions(signature(entry)["model"], SOURCE)


def model_loader_assertions(model_entry, source):
    """``loader_access_assertions`` for one ``GEMMKernel`` model entry."""
    assertions = []
    for loader in model_entry.loaders:
        slots = [slot for lane in loader.slots() for slot in lane]
        if len(set(slots)) != loader.rows * loader.columns:
            raise ValueError("Steel GEMM loader slots overlap")
        assertions.append(
            {
                "source": source,
                "entryPoint": model_entry.name,
                "function": loader.function,
                "parameter": "self.dst",
                "minimum": min(slots),
                "maximum": max(slots),
            }
        )
    return tuple(assertions)


def index_assertions(entry):
    """OpenGL ranges for the 64-bit pointer offsets the kernel forms.

    Device views index MLX buffers below MAX_ELEMENTS elements; a loader reads
    ``reads`` adjacent elements past its source offset, and each threadgroup
    view reaches only the slots its loader writes.
    """
    info = signature(entry)
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
        for access in (f"load_{element}", f"store_{element}", "load_int64_t")
    ]
    workgroup = [
        {
            "expression": "reference.offset + index",
            "indexed": f"crosstl_workgroup_{tile}",
            "maximum": loader["maximum"],
        }
        for tile, loader in zip(("As", "Bs"), loader_access_assertions(entry))
    ]
    return tuple(
        {"source": SOURCE, **item, "minimum": 0} for item in device + workgroup
    )
