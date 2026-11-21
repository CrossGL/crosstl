"""Storage and launch contracts for MLX's segmented steel GEMM kernel.

``SegmentedMM::eval_gpu`` in ``segmented_mm.inc`` routes every nonempty
segmented product as matmul.cpp's ``segmented_mm`` does on a device without
NAX: to ``steel_segmented_mm`` with ``GEMM_TPARAM_MACRO``'s medium-device
64x64x16 tile and 2x2 simdgroups. Output batch ``z`` is ``A[:, s:e] @ B[s:e]``
over the inner range ``segments`` gives for it, either one ``(k_start,
k_end)`` pair per batch or, for views whose two trailing strides are both one,
batch boundaries that ``z`` and ``z + 1`` index.

The entry names carry the function constants as matmul.cpp's hash name does.
Each plan reads the uint32 segment words the kernel will read, replays the
kernel's uint32 K loop and 16-bit tail for each batch, and proves every
operand read lies inside its buffer.
"""

import ctypes
import re

from demos.integrations.mlx.portable_host import gemm_layout, gemv_layout
from demos.integrations.mlx.portable_host.quantization_layout import ITEM_SIZES
from demos.integrations.mlx.tests.kernels import gemm_segmented_model as model

SOURCE = "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_segmented.metal"
SIMD_SIZE = 32
MAX_ELEMENTS = gemv_layout.MAX_ELEMENTS
SIGNED_MAX = gemv_layout.SIGNED_MAX
MAX_GROUPS = gemv_layout.MAX_GROUPS
DTYPES = gemv_layout.DTYPES
# segmented_mm's GEMM_TPARAM_MACRO tile on a medium device.
HOST_TILE = gemm_layout.HOST_TILE
CONSTANTS = ("segments_contiguous", "align_M", "align_N")
CONSTANT_IDS = {"segments_contiguous": 199, "align_M": 200, "align_N": 201}
PATTERN = re.compile(
    r"(steel_segmented_mm_(nn|nt|tn|tt)_(float32|float16|bfloat16)_\3"
    r"_bm(\d+)_bn(\d+)_bk(\d+)_wm(\d+)_wn(\d+))"
    r"_segments_contiguous_([tn])_align_M_([tn])_align_N_([tn])"
)
ENTRY_PREFIXES = ("steel_segmented_mm_",)
# segmented_mm's [[buffer(i)]] parameters in index order; matmul.cpp binds
# every one of them.
PARAMETERS = ("A", "B", "segments", "C", "params")
MATRIX_PARAMETERS = ("A", "B", "C")
OUTPUT = "C"
PARAMS_FIELDS = gemm_layout.PARAMS_FIELDS
TYPES = {"segments": "uint32", "params": "int64"}
INTEGER_TYPES = {"int64": ctypes.c_int64, "uint32": ctypes.c_uint32}


def _flag(value):
    return "t" if value else "n"


def signature(entry):
    """The source entry, transposition, element type and constants of an entry."""
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError("Unsupported segmented steel GEMM entry")
    base, transpose, dtype, *rest = match.groups()
    tile, flags = tuple(int(value) for value in rest[:5]), rest[5:]
    model_entry = model.BY_NAME.get(base)
    if model_entry is None or tile != HOST_TILE:
        raise ValueError("Segmented steel GEMM entry is not the host's tile")
    bm, bn, bk, wm, wn = tile
    return {
        "kind": "segmented",
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
        "constants": {name: flag == "t" for name, flag in zip(CONSTANTS, flags)},
    }


def host_parameters(entry):
    signature(entry)
    return PARAMETERS


def workgroup_size(entry):
    info = signature(entry)
    return (SIMD_SIZE, info["wn"], info["wm"])


def select(dtype, transpose_a, transpose_b, m, n, segments_contiguous):
    """segmented_mm's hash name for one product."""
    if dtype not in DTYPES:
        raise ValueError("Segmented products support float32, float16 and bfloat16")
    bm, bn, bk, wm, wn = HOST_TILE
    transpose = _flag(transpose_a) + _flag(transpose_b)
    return (
        f"steel_segmented_mm_{transpose}_{dtype}_{dtype}"
        f"_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}"
        f"_segments_contiguous_{_flag(segments_contiguous)}"
        f"_align_M_{_flag(m % bm == 0)}_align_N_{_flag(n % bn == 0)}"
    )


def parameter_type(entry, name):
    if name in MATRIX_PARAMETERS:
        return signature(entry)["dtype"]
    return TYPES[name]


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters.

    Returns ``(counts, parameters, views)``: element counts of the matrix and
    segment buffers, ``params`` as a field map, and a zero-copy view of the
    segment words.
    """
    if set(buffers) != set(host_parameters(entry)):
        raise ValueError("Segmented steel GEMM buffers do not match the entry")
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    counts, parameters, views, spans = {}, {}, {}, {}
    for name in PARAMETERS:
        buffer, dtype = buffers[name], parameter_type(entry, name)
        itemsize = ITEM_SIZES.get(dtype) or ctypes.sizeof(INTEGER_TYPES[dtype])
        count = buffer.count
        limit = len(PARAMS_FIELDS) if name == "params" else MAX_ELEMENTS
        if (
            not buffer.data
            or buffer.data % itemsize
            or buffer.dtype != dtype.encode("ascii")
            or buffer.output != int(name == OUTPUT)
            or type(count) is not int
            or not 1 <= count <= limit
            or buffer.data + count * itemsize > address_limit
        ):
            raise ValueError(
                "Segmented steel GEMM buffer storage or direction does not match"
            )
        spans[name] = (buffer.data, buffer.data + count * itemsize)
        if name in MATRIX_PARAMETERS:
            counts[name] = count
        elif name == "segments":
            counts[name] = count
            views[name] = (ctypes.c_uint32 * count).from_address(buffer.data)
        else:
            values = list((ctypes.c_int64 * count).from_address(buffer.data))
            if len(values) != len(PARAMS_FIELDS):
                raise ValueError(
                    "Segmented steel GEMM parameters do not match GEMMParams"
                )
            parameters[name] = dict(zip(PARAMS_FIELDS, values))
    low, high = spans[OUTPUT]
    if any(
        other != OUTPUT and max(low, start) < min(high, end)
        for other, (start, end) in spans.items()
    ):
        raise ValueError("Segmented steel GEMM output overlaps another buffer")
    return counts, parameters, views


def _short(value):
    """``value`` converted to a 16-bit ``short`` as Metal converts it."""
    return (value + 2**15) % 2**16 - 2**15


def k_loop(bk, start, end):
    """``(tiles, tail, reach)`` of segmented_mm's K loop for one batch.

    The kernel counts ``k`` in uint32 from ``start + BK`` while ``k <= end``,
    then loads a tail of ``short(BK - short(k - end))`` columns if positive.
    ``reach`` is one past the last inner index the batch reads; ``load_safe``
    never reads past its BK-column tile.
    """
    tiles, k = 0, (start + bk) % 2**32
    while k <= end:
        tiles += 1
        k = (k + bk) % 2**32
    tail = _short(bk - _short((k - end) % 2**32))
    reach = start + tiles * bk + min(max(tail, 0), bk)
    return tiles, tail, reach


def segment_ranges(words, contiguous, batches):
    """The ``(k_start, k_end)`` words segmented_mm reads for each batch."""
    return model.segments(list(words), contiguous, batches)


def _check_launch(entry, execution, count):
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError("Segmented steel GEMM launch geometry is missing")
    groups = list(execution["workgroupCount"])
    if (
        list(execution["workgroupSize"]) != list(workgroup_size(entry))
        or groups != list(count)
        or not all(type(value) is int and 1 <= value <= MAX_GROUPS for value in groups)
    ):
        raise ValueError("Segmented steel GEMM launch does not match MLX's grid")


def _check_params(info, params):
    if not isinstance(params, dict) or set(params) != set(PARAMS_FIELDS):
        raise ValueError("Segmented steel GEMM parameters are missing")
    if any(type(value) is not int for value in params.values()):
        raise ValueError("Segmented steel GEMM parameters must be integers")
    m, n, k = params["M"], params["N"], params["K"]
    # A leading dimension only steps between rows the loaders read when the
    # operand has more than one of them; check_transpose passes any stride
    # of a single row.
    minor_a, major_a = (m, k) if info["transposeA"] else (k, m)
    minor_b, major_b = (k, n) if info["transposeB"] else (n, k)
    expected = {
        "ldd": n,
        "tiles_n": -(-n // info["bn"]),
        "tiles_m": -(-m // info["bm"]),
        "batch_stride_a": 0,
        "batch_stride_b": 0,
        "batch_stride_d": m * n,
        "swizzle_log": 0,
        "gemm_k_iterations_aligned": 0,
        "batch_ndim": 0,
    }
    if (
        not 1 <= min(m, n, k) <= max(m, n, k) <= MAX_ELEMENTS
        or any(params[name] != value for name, value in expected.items())
        or not gemm_layout.leading_dimension(params["lda"], minor_a, major_a)
        or not gemm_layout.leading_dimension(params["ldb"], minor_b, major_b)
    ):
        raise ValueError("Segmented steel GEMM parameters differ from matmul.cpp's")
    return m, n, k


def plan(entry, counts, parameters, views, threads, execution):
    """Validate one launch against MLX's routing and return its metadata."""
    info = signature(entry)
    params = parameters.get("params")
    m, n, k = _check_params(info, params)
    if threads % (m * n) or counts[OUTPUT] != threads:
        raise ValueError("Segmented steel GEMM output extent differs")
    batches = threads // (m * n)
    _check_launch(entry, execution, (params["tiles_n"], params["tiles_m"], batches))
    contiguous = info["constants"]["segments_contiguous"]
    if select(
        info["dtype"], info["transposeA"], info["transposeB"], m, n, contiguous
    ) != (entry):
        raise ValueError("Segmented steel GEMM entry differs from MLX's routing")
    words = 2 * batches if contiguous else batches + 1
    if counts["segments"] < words:
        raise ValueError("Segmented steel GEMM segments read exceeds its buffer")
    ranges = segment_ranges(views["segments"][:words], contiguous, batches)
    loops = []
    for start, end in ranges:
        tiles, tail, reach = k_loop(info["bk"], start, end)
        # Every word stays inside the contraction, so no pointer the kernel
        # forms leaves the operands and no uint32 K count wraps.
        if max(start, end, reach) > k:
            raise ValueError("Segmented steel GEMM segment exceeds the contraction")
        loops.append([tiles, tail])
    lda, ldb = params["lda"], params["ldb"]
    extents = {
        "A": (k - 1) * lda + m if info["transposeA"] else (m - 1) * lda + k,
        "B": (n - 1) * ldb + k if info["transposeB"] else (k - 1) * ldb + n,
    }
    for name, extent in extents.items():
        if extent > counts[name]:
            raise ValueError(f"Segmented steel GEMM {name} exceeds its buffer")
    return {
        "kind": "segmented",
        "source": info["source"],
        "M": m,
        "N": n,
        "K": k,
        "lda": lda,
        "ldb": ldb,
        "batches": batches,
        "constants": info["constants"],
        "segments": [list(segment) for segment in ranges],
        "kLoops": loops,
    }


def loader_access_assertions(entry):
    """The threadgroup slots each ``BlockLoader::load_unsafe`` writes."""
    return gemm_layout.model_loader_assertions(signature(entry)["model"], SOURCE)


def index_assertions(entry):
    """OpenGL ranges for the 64-bit pointer offsets the kernel forms.

    As ``test_current_gemm_segmented`` asserts them: device views stay below
    2^31 elements, ``segments`` is read as ``uint``, and each threadgroup view
    reaches only the slots its loader writes.
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
        for access in sorted({f"load_{element}", f"store_{element}", "load_uint"})
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


def _check_transpose(shape, strides):
    """segmented_mm.inc's gemv_operand: matmul.cpp check_transpose that also
    copies views with negative strides. Returns ``(transposed, ld, copied)``."""
    if min(strides) >= 0:
        stx, sty = strides[-2], strides[-1]
        if sty == 1:
            return False, stx, False
        if stx == 1:
            return True, sty, False
    return False, shape[-1], True


def segments_layout(shape, strides, row_contiguous):
    """segmented_mm.inc's segmented_segments: ``(contiguous, copied)``."""
    if row_contiguous:
        return True, False
    ndim = len(shape)
    rc = all(
        strides[i + 1] * shape[i] == strides[i] or shape[i] == 1
        for i in range(ndim - 2)
    )
    rc = rc and strides[-1] == 1 and (ndim < 2 or strides[-2] == 1)
    batches = 1
    for i in reversed(range(ndim - 2)):
        batches *= shape[i + 1]
        rc = rc and (shape[i] == 1 or strides[i] == batches)
    return (False, False) if rc else (True, True)


def route(dtype, a_shape, a_strides, b_shape, b_strides, segments):
    """SegmentedMM::eval_gpu's route for nonempty operands.

    ``segments`` is ``(shape, strides, row_contiguous)``. Returns the kind,
    the entries in launch order and the copies the host makes first, each one
    translated copy dispatch.
    """
    (m, k), n = a_shape, b_shape[-1]
    if b_shape[0] != k:
        raise ValueError("Segmented product operand shapes differ")
    contiguous, segments_copy = segments_layout(*segments)
    transpose_a, lda, a_copy = _check_transpose(a_shape, a_strides)
    transpose_b, ldb, b_copy = _check_transpose(b_shape, b_strides)
    return {
        "kind": "segmented",
        "entries": [select(dtype, transpose_a, transpose_b, m, n, contiguous)],
        "M": m,
        "N": n,
        "K": k,
        "lda": lda,
        "ldb": ldb,
        "contiguous": contiguous,
        "copies": {"a": a_copy, "b": b_copy, "segments": segments_copy},
    }
