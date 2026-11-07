"""Storage and launch contracts for MLX's block-masked matrix kernels.

``BlockMaskedMM::eval_gpu`` in ``block_masked_mm.inc`` routes each nonempty
product as matmul.cpp does: products with one output row or column take
``gemv_masked`` or ``gemv_t_masked`` from gemv_masked.metal, every other one
the steel ``block_masked_gemm`` of steel_gemm_masked.metal with square
``block_size`` tiles. The entry names carry the mask types (``bool_``, the
element type, or ``nomask``), the tile and, for steel, the alignment, as
matmul.cpp's kernel names do.

Masks hold one value per ``block_size`` block: the output mask one per output
tile, the operand masks one per block of their operand. A zero output value
writes zeros; a pair of zero operand values skips a block; masks of the
element type also scale. Each plan checks the launch against MLX's routing
and proves that every element and mask value the kernel can read lies inside
its buffer. matmul.cpp binds no buffer for an absent mask; the kernels never
read one, and the dispatcher fills those bindings with inert words.
"""

import ctypes
import math
import re

from demos.integrations.mlx.portable_host import gemm_layout, gemv_layout
from demos.integrations.mlx.portable_host.quantization_layout import ITEM_SIZES
from demos.integrations.mlx.tests.kernels import gemm_masked_model as model

GEMM_SOURCE = "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_masked.metal"
GEMV_SOURCE = "mlx/backend/metal/kernels/gemv_masked.metal"
SIMD_SIZE = 32
MAX_ELEMENTS = gemv_layout.MAX_ELEMENTS
SIGNED_MAX = gemv_layout.SIGNED_MAX
MAX_GROUPS = gemv_layout.MAX_GROUPS
MAX_BATCH_DIMENSIONS = gemv_layout.MAX_BATCH_DIMENSIONS
DTYPES = gemv_layout.DTYPES
BLOCK_SIZES = (32, 64)
BK = 16
GEMV_PREFIXES = ("gemv_outmask_", "gemv_t_outmask_")
GEMM_PREFIXES = ("steel_gemm_block_outmask_",)
ENTRY_PREFIXES = (*GEMV_PREFIXES, *GEMM_PREFIXES)
MASK_NAMES = ("bool_", "nomask", *DTYPES)
GEMV_PATTERN = re.compile(
    r"(gemv|gemv_t)_outmask_(bool_|nomask|float32|float16|bfloat16)"
    r"_opmask_(bool_|nomask|float32|float16|bfloat16)_(float32|float16|bfloat16)"
    r"_bm(\d+)_bn(\d+)_sm(\d+)_sn(\d+)_tm(\d+)_tn(\d+)_nc([01])"
)
# gemv_masked.metal's instantiated tiles (bm, bn, sm, sn, tm, tn).
GEMV_BLOCKS = {
    "gemv": (
        (2, 1, 4, 8, 1, 4),
        (2, 1, 4, 8, 4, 4),
        (2, 1, 2, 16, 1, 4),
        (2, 1, 2, 16, 4, 4),
        (4, 1, 2, 16, 4, 4),
    ),
    "gemv_t": (
        (1, 1, 8, 4, 4, 1),
        (1, 2, 8, 4, 4, 4),
        (1, 1, 8, 4, 8, 1),
        (1, 1, 8, 4, 8, 4),
        (1, 2, 8, 4, 8, 4),
        (1, 4, 8, 4, 8, 4),
    ),
}
# (output mask, operand mask) pairs both sources instantiate; "T" is the
# element type. No kernel takes two absent masks: block_masked_mm then
# calls matmul.
MASK_PAIRS = (
    ("bool_", "bool_"),
    ("T", "T"),
    ("bool_", "T"),
    ("T", "bool_"),
    ("nomask", "T"),
    ("nomask", "bool_"),
    ("bool_", "nomask"),
    ("T", "nomask"),
)
# [[buffer(i)]] parameters in index order.
GEMV_PARAMETERS = (
    "mat",
    "in_vec",
    "out_vec",
    "in_vec_size",
    "out_vec_size",
    "marix_ld",
    "batch_ndim",
    "batch_shape",
    "vector_batch_stride",
    "matrix_batch_stride",
    "out_mask",
    "mat_mask",
    "vec_mask",
    "mask_strides",
    "mask_batch_strides",
)
GEMM_PARAMETERS = (
    "A",
    "B",
    "D",
    "params",
    "batch_shape",
    "batch_strides",
    "out_mask",
    "lhs_mask",
    "rhs_mask",
    "mask_strides",
)
OUTPUTS = {"masked-mv": "out_vec", "masked-mm": "D"}
MATRICES = {"masked-mv": ("mat", "in_vec", "out_vec"), "masked-mm": ("A", "B", "D")}
MASKS = {
    "masked-mv": {"out": "out_mask", "op": ("mat_mask", "vec_mask")},
    "masked-mm": {"out": "out_mask", "op": ("lhs_mask", "rhs_mask")},
}
SCALARS = ("in_vec_size", "out_vec_size", "marix_ld", "batch_ndim")
VECTOR_TYPES = {
    "batch_shape": "int32",
    "vector_batch_stride": "int64",
    "matrix_batch_stride": "int64",
    "mask_strides": "int32",
    "mask_batch_strides": "int64",
    "batch_strides": "int64",
    "params": "int64",
}
INTEGER_TYPES = {"int32": ctypes.c_int32, "int64": ctypes.c_int64}
PARAMS_FIELDS = gemm_layout.PARAMS_FIELDS
# Expressions OpenGL may evaluate in 32 bits in the masked gemv kernels: the
# batch prologue's pointer offsets (test_current_gemv_masked).
GEMV_INDEX_RANGE_ASSERTIONS = (
    "tid.z * buffer_load(vector_batch_stride, 0)",
    "tid.z * buffer_load(matrix_batch_stride, 0)",
    "tid.z * buffer_load(mask_batch_strides, 0)",
    "tid.z * buffer_load(mask_batch_strides, batch_ndim)",
    "batch_offsets.x",
    "batch_offsets.y",
)


def _flag(value):
    return "t" if value else "n"


def _mask_kind(name):
    """``"nomask"``, ``"bool"`` or ``"scale"`` for a kernel-name mask token."""
    return {"nomask": "nomask", "bool_": "bool"}.get(name, "scale")


def _gemv_signature(entry):
    match = GEMV_PATTERN.fullmatch(entry)
    if match is None:
        raise ValueError("Unsupported block masked matrix-vector entry")
    kind, out_mask, op_mask, dtype, *rest = match.groups()
    block, nc = tuple(int(value) for value in rest[:6]), rest[6]
    pair = tuple("T" if name == dtype else name for name in (out_mask, op_mask))
    if block not in GEMV_BLOCKS[kind] or pair not in MASK_PAIRS:
        raise ValueError("Block masked matrix-vector entry is not instantiated")
    if any(name not in ("bool_", "nomask", dtype) for name in (out_mask, op_mask)):
        raise ValueError("Block masked matrix-vector mask type differs")
    bm, bn, sm, sn, tm, tn = block
    # The mask block is the threadgroup's input extent per iteration.
    mask_block = bn * sn * tn if kind == "gemv" else bm * sm * tm
    return {
        "kind": "masked-mv",
        "kernel": kind,
        "source": entry,
        "dtype": dtype,
        "bm": bm,
        "bn": bn,
        "sm": sm,
        "sn": sn,
        "tm": tm,
        "tn": tn,
        "nc": nc == "1",
        "block": mask_block,
        "outMask": _mask_kind(out_mask),
        "opMask": _mask_kind(op_mask),
    }


def _gemm_signature(entry):
    found = model.BY_NAME.get(entry)
    if found is None:
        raise ValueError("Unsupported block masked steel GEMM entry")
    gemm = found.gemm
    return {
        "kind": "masked-mm",
        "source": entry,
        "model": found,
        "dtype": found.dtype,
        "transposeA": found.transpose_a,
        "transposeB": found.transpose_b,
        "bm": gemm.bm,
        "bn": gemm.bn,
        "bk": gemm.bk,
        "wm": gemm.wm,
        "wn": gemm.wn,
        "block": found.block,
        "mnAligned": found.mn_aligned,
        "kAligned": found.k_aligned,
        "outMask": found.mask_kind("out"),
        "opMask": found.mask_kind("op"),
    }


def signature(entry):
    """The kind, element type, tile and mask kinds of an entry."""
    if isinstance(entry, str) and entry.startswith(GEMV_PREFIXES):
        return _gemv_signature(entry)
    if isinstance(entry, str) and entry.startswith(GEMM_PREFIXES):
        return _gemm_signature(entry)
    raise ValueError("Unsupported block masked matrix entry")


def kind_of(entry):
    return signature(entry)["kind"]


def parameters(entry):
    """Every ``[[buffer(i)]]`` parameter of the entry's kernel."""
    return GEMV_PARAMETERS if kind_of(entry) == "masked-mv" else GEMM_PARAMETERS


def absent_masks(entry):
    """The mask parameters matmul.cpp leaves unbound for ``entry``."""
    info = signature(entry)
    masks = MASKS[info["kind"]]
    absent = []
    if info["outMask"] == "nomask":
        absent.append(masks["out"])
    if info["opMask"] == "nomask":
        absent.extend(masks["op"])
    return tuple(absent)


def host_parameters(entry):
    """The parameters BlockMaskedMM::eval_gpu binds for ``entry``."""
    absent = absent_masks(entry)
    return tuple(name for name in parameters(entry) if name not in absent)


def workgroup_size(entry):
    info = signature(entry)
    if info["kind"] == "masked-mv":
        return (SIMD_SIZE, info["bn"], info["bm"])
    return (SIMD_SIZE, info["wn"], info["wm"])


def mask_dtype(entry, which):
    """The storage type of the ``"out"`` or ``"op"`` mask, None if absent."""
    info = signature(entry)
    kind = info["outMask" if which == "out" else "opMask"]
    return {"nomask": None, "bool": "bool_"}.get(kind, info["dtype"])


def parameter_type(entry, name):
    info = signature(entry)
    if name in MATRICES[info["kind"]]:
        return info["dtype"]
    masks = MASKS[info["kind"]]
    if name == masks["out"]:
        return mask_dtype(entry, "out")
    if name in masks["op"]:
        return mask_dtype(entry, "op")
    if name in SCALARS:
        return "int32"
    return VECTOR_TYPES[name]


def outputs_per_group(entry):
    """matmul.cpp's n_out_per_tgp for a masked gemv entry."""
    info = signature(entry)
    if info["kernel"] == "gemv_t":
        return info["bn"] * info["sn"] * info["tn"]
    return info["bm"] * info["sm"] * info["tm"]


def select_gemv(dtype, transpose_mat, out_size, block_size, out_mask, op_mask, nc):
    """matmul.cpp's masked gemv kernel name. Masks are kernel-name tokens."""
    if dtype not in DTYPES or block_size not in BLOCK_SIZES:
        raise ValueError("Block masked products support float32, float16 and bfloat16")
    if transpose_mat:
        sm, sn, bm = 8, 4, 1
        bn = 4 if block_size == 64 and out_size >= 2048 else 2
        tm = 4 if block_size == 32 else 8
        tn = 1 if out_size < 4 else 4
        kind = "gemv_t"
    else:
        tn = 4
        if block_size == 32:
            sm, sn, bm = 4, 8, 2
        else:
            sm, sn, bm = 2, 16, 4 if out_size >= 512 else 2
        bn = 1
        tm = 1 if out_size < 4 else 4
        kind = "gemv"
    return (
        f"{kind}_outmask_{out_mask}_opmask_{op_mask}_{dtype}"
        f"_bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}_nc{int(nc)}"
    )


def select_gemm(
    dtype, transpose_a, transpose_b, m, n, k, block_size, out_mask, op_mask
):
    """matmul.cpp's steel block_masked_gemm kernel name."""
    if dtype not in DTYPES or block_size not in BLOCK_SIZES:
        raise ValueError("Block masked products support float32, float16 and bfloat16")
    aligned = m % block_size == 0 and n % block_size == 0
    return (
        f"steel_gemm_block_outmask_{out_mask}_opmask_{op_mask}"
        f"_{_flag(transpose_a)}{_flag(transpose_b)}_{dtype}_{dtype}"
        f"_bm{block_size}_bn{block_size}_bk{BK}_wm2_wn2"
        f"_MN_{_flag(aligned)}aligned_K_{_flag(k % BK == 0)}aligned"
    )


def _token(entry, which):
    info = signature(entry)
    kind = info["outMask" if which == "out" else "opMask"]
    return {"nomask": "nomask", "bool": "bool_"}.get(kind, info["dtype"])


def block_size_of(entry):
    """The block_masked_mm block size an entry serves."""
    info = signature(entry)
    if info["kind"] == "masked-mm":
        return info["block"]
    if info["kernel"] == "gemv":
        return info["bn"] * info["sn"] * info["tn"]
    return 32 if info["tm"] == 4 else 64


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters.

    Returns ``(counts, parameters, views)``: element counts of the matrix and
    mask buffers, the scalar and vector parameters (``params`` as a field
    map) and no views.
    """
    names = host_parameters(entry)
    if set(buffers) != set(names):
        raise ValueError("Block masked matmul buffers do not match the entry")
    info = signature(entry)
    output = OUTPUTS[info["kind"]]
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    counts, values, spans = {}, {}, {}
    for name in names:
        buffer, dtype = buffers[name], parameter_type(entry, name)
        itemsize = (
            1
            if dtype == "bool_"
            else ITEM_SIZES.get(dtype) or ctypes.sizeof(INTEGER_TYPES[dtype])
        )
        count = buffer.count
        scalar = name in SCALARS
        limit = (
            1
            if scalar
            else (
                len(PARAMS_FIELDS)
                if name == "params"
                else (
                    3 * MAX_BATCH_DIMENSIONS * 5
                    if name in VECTOR_TYPES
                    else MAX_ELEMENTS
                )
            )
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
            raise ValueError(
                "Block masked matmul buffer storage or direction does not match"
            )
        spans[name] = (buffer.data, buffer.data + count * itemsize)
        if name not in SCALARS and name not in VECTOR_TYPES:
            counts[name] = count
            continue
        words = list((INTEGER_TYPES[dtype] * count).from_address(buffer.data))
        if name == "params":
            if count != len(PARAMS_FIELDS):
                raise ValueError("Block masked GEMM parameters do not match GEMMParams")
            values[name] = dict(zip(PARAMS_FIELDS, words))
        else:
            values[name] = words[0] if scalar else words
    low, high = spans[output]
    if any(
        other != output and max(low, start) < min(high, end)
        for other, (start, end) in spans.items()
    ):
        raise ValueError("Block masked matmul output overlaps another buffer")
    return counts, values, {}


def _location(index, shape, strides):
    """utils.h elem_to_loc."""
    return gemv_layout._location(index, shape, strides)


def _check_launch(entry, execution, count):
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError("Block masked matmul launch geometry is missing")
    groups = list(execution["workgroupCount"])
    if (
        list(execution["workgroupSize"]) != list(workgroup_size(entry))
        or groups != list(count)
        or not all(type(value) is int and 1 <= value <= MAX_GROUPS for value in groups)
    ):
        raise ValueError("Block masked matmul launch does not match MLX's grid")


def _check_strides(strides, length, label):
    if (
        not isinstance(strides, list)
        or len(strides) != length
        or any(
            type(value) is not int or not 0 <= value <= MAX_ELEMENTS
            for value in strides
        )
    ):
        raise ValueError(f"Block masked matmul {label} differ from matmul.cpp's")


def _mask_reads(name, counts, shape, batch_strides, batches, extent):
    """Require every batch's mask reads, ``extent`` past its base, in bounds."""
    end = max(_location(z, shape, batch_strides) for z in range(batches))
    if end + extent > counts[name]:
        raise ValueError(f"Block masked matmul {name} read exceeds its buffer")


def _plan_gemv(entry, info, counts, values, threads, execution):
    k, out, ld = values["in_vec_size"], values["out_vec_size"], values["marix_ld"]
    ndim, shape = values["batch_ndim"], values["batch_shape"]
    if (
        any(type(value) is not int for value in (k, out, ld, ndim))
        or not 1 <= min(k, out) <= max(k, out) <= MAX_ELEMENTS
        or not 0 <= ld <= MAX_ELEMENTS
        or not 1 <= ndim <= MAX_BATCH_DIMENSIONS
        or len(shape) != ndim
        or any(type(extent) is not int or extent < 1 for extent in shape)
    ):
        raise ValueError("Block masked matrix-vector dimensions are invalid")
    if info["nc"] != (ndim != 1):
        raise ValueError("Block masked matrix-vector batch variant differs")
    _check_strides(values["vector_batch_stride"], ndim, "vector batch strides")
    _check_strides(values["matrix_batch_stride"], ndim, "matrix batch strides")
    has_out, has_op = info["outMask"] != "nomask", info["opMask"] != "nomask"
    _check_strides(values["mask_strides"], 2 * has_out + 4 * has_op, "mask strides")
    _check_strides(
        values["mask_batch_strides"],
        ndim * (has_out + 2 * has_op),
        "mask batch strides",
    )
    batches = math.prod(shape)
    tiles = -(-out // outputs_per_group(entry))
    _check_launch(entry, execution, (tiles, 1, batches))
    transpose_mat = info["kernel"] == "gemv_t"
    block = block_size_of(entry)
    selected = select_gemv(
        info["dtype"],
        transpose_mat,
        out,
        block,
        _token(entry, "out"),
        _token(entry, "op"),
        info["nc"],
    )
    if selected != entry or info["block"] != block:
        raise ValueError("Block masked matrix-vector entry differs from MLX's routing")
    if threads != out * batches or counts["out_vec"] != threads:
        raise ValueError("Block masked matrix-vector output extent differs")
    extents = {
        "in_vec": k,
        "mat": (out - 1) * ld + k if not transpose_mat else (k - 1) * ld + out,
    }
    for name, strides in (
        ("in_vec", values["vector_batch_stride"]),
        ("mat", values["matrix_batch_stride"]),
    ):
        end = max(_location(z, shape, strides) for z in range(batches))
        if end + extents[name] > counts[name]:
            raise ValueError(f"Block masked matrix-vector {name} exceeds its buffer")
    # Mask blocks the kernel can index: output rows (gemv) or columns
    # (gemv_t), and input blocks along K.
    out_blocks, k_blocks = -(-out // block), -(-k // block)
    strides, batch = values["mask_strides"], values["mask_batch_strides"]
    position = 0
    if has_out:
        os = strides[:2]
        # gemv_t reads its output mask by block index alone.
        extent = (out_blocks - 1) * (1 if transpose_mat else os[1]) + 1
        _mask_reads("out_mask", counts, shape, batch[:ndim], batches, extent)
        strides, position = strides[2:], ndim
    if has_op:
        ms, vs = strides[:2], strides[2:4]
        if transpose_mat:
            mat_extent = (out_blocks - 1) * ms[0] + (k_blocks - 1) * ms[1] + 1
            vec_extent = (k_blocks - 1) * vs[0] + 1
        else:
            mat_extent = (out_blocks - 1) * ms[1] + (k_blocks - 1) * ms[0] + 1
            vec_extent = (k_blocks - 1) * vs[1] + 1
        mat_batch = batch[position : position + ndim]
        vec_batch = batch[position + ndim : position + 2 * ndim]
        _mask_reads("mat_mask", counts, shape, mat_batch, batches, mat_extent)
        _mask_reads("vec_mask", counts, shape, vec_batch, batches, vec_extent)
    return {
        "kind": "masked-mv",
        "kernel": info["kernel"],
        "K": k,
        "outVecSize": out,
        "matrixLd": ld,
        "batches": batches,
        "blockSize": block,
        "outMask": info["outMask"],
        "opMask": info["opMask"],
    }


def _check_params(info, params, ndim, shape):
    if not isinstance(params, dict) or set(params) != set(PARAMS_FIELDS):
        raise ValueError("Block masked GEMM parameters are missing")
    if any(type(value) is not int for value in params.values()):
        raise ValueError("Block masked GEMM parameters must be integers")
    m, n, k = params["M"], params["N"], params["K"]
    minor_a, major_a = (m, k) if info["transposeA"] else (k, m)
    minor_b, major_b = (k, n) if info["transposeB"] else (n, k)
    expected = {
        "ldd": n,
        "tiles_n": -(-n // info["bn"]),
        "tiles_m": -(-m // info["bm"]),
        "batch_stride_d": m * n,
        "swizzle_log": 0,
        "gemm_k_iterations_aligned": k // info["bk"],
        "batch_ndim": ndim,
    }
    if (
        not 2 <= min(m, n)
        or not 1 <= k
        or max(m, n, k) > MAX_ELEMENTS
        or not 1 <= ndim <= MAX_BATCH_DIMENSIONS
        or len(shape) != ndim
        or any(type(extent) is not int or extent < 1 for extent in shape)
        or any(params[name] != value for name, value in expected.items())
        or not gemm_layout.leading_dimension(params["lda"], minor_a, major_a)
        or not gemm_layout.leading_dimension(params["ldb"], minor_b, major_b)
    ):
        raise ValueError("Block masked GEMM parameters differ from matmul.cpp's")
    return m, n, k


def _plan_gemm(entry, info, counts, values, threads, execution):
    params, shape = values["params"], values["batch_shape"]
    ndim = len(shape) if isinstance(shape, list) else 0
    m, n, k = _check_params(info, params, ndim, shape)
    has_out, has_op = info["outMask"] != "nomask", info["opMask"] != "nomask"
    inputs = 2 + has_out + 2 * has_op
    _check_strides(values["batch_strides"], ndim * inputs, "batch strides")
    _check_strides(values["mask_strides"], 2 * has_out + 4 * has_op, "mask strides")
    batch = values["batch_strides"]
    # Without batch_ndim > 1 the kernel offsets A and B by GEMMParams'
    # strides, which must be the last batch strides eval_gpu collected.
    if (
        params["batch_stride_a"] != batch[ndim - 1]
        or params["batch_stride_b"] != batch[2 * ndim - 1]
    ):
        raise ValueError("Block masked GEMM batch strides differ from matmul.cpp's")
    batches = math.prod(shape)
    _check_launch(entry, execution, (params["tiles_n"], params["tiles_m"], batches))
    selected = select_gemm(
        info["dtype"],
        info["transposeA"],
        info["transposeB"],
        m,
        n,
        k,
        info["block"],
        _token(entry, "out"),
        _token(entry, "op"),
    )
    if selected != entry:
        raise ValueError("Block masked GEMM entry differs from MLX's routing")
    if threads != batches * m * n or counts["D"] != threads:
        raise ValueError("Block masked GEMM output extent differs")
    lda, ldb = params["lda"], params["ldb"]
    extents = {
        "A": (k - 1) * lda + m if info["transposeA"] else (m - 1) * lda + k,
        "B": (n - 1) * ldb + k if info["transposeB"] else (k - 1) * ldb + n,
    }
    for index, name in enumerate(("A", "B")):
        strides = batch[index * ndim : (index + 1) * ndim]
        end = max(_location(z, shape, strides) for z in range(batches))
        if end + extents[name] > counts[name]:
            raise ValueError(f"Block masked GEMM {name} batch exceeds its buffer")
    tiles_m, tiles_n = params["tiles_m"], params["tiles_n"]
    # K iterations share one operand mask block per block_size columns.
    k_blocks = (k - 1) // info["block"] + 1
    strides, position = values["mask_strides"], 2 * ndim
    if has_out:
        os = strides[:2]
        extent = (tiles_m - 1) * os[1] + (tiles_n - 1) * os[0] + 1
        _mask_reads(
            "out_mask",
            counts,
            shape,
            batch[position : position + ndim],
            batches,
            extent,
        )
        strides, position = strides[2:], position + ndim
    if has_op:
        ls, rs = strides[:2], strides[2:4]
        lhs_extent = (tiles_m - 1) * ls[1] + (k_blocks - 1) * ls[0] + 1
        rhs_extent = (tiles_n - 1) * rs[0] + (k_blocks - 1) * rs[1] + 1
        lhs_batch = batch[position : position + ndim]
        rhs_batch = batch[position + ndim : position + 2 * ndim]
        _mask_reads("lhs_mask", counts, shape, lhs_batch, batches, lhs_extent)
        _mask_reads("rhs_mask", counts, shape, rhs_batch, batches, rhs_extent)
    return {
        "kind": "masked-mm",
        "source": info["source"],
        "M": m,
        "N": n,
        "K": k,
        "lda": lda,
        "ldb": ldb,
        "batches": batches,
        "blockSize": info["block"],
        "outMask": info["outMask"],
        "opMask": info["opMask"],
    }


def plan(entry, counts, parameters, views, threads, execution):
    """Validate one launch against MLX's routing and return its metadata."""
    info = signature(entry)
    if info["kind"] == "masked-mv":
        return _plan_gemv(entry, info, counts, parameters, threads, execution)
    return _plan_gemm(entry, info, counts, parameters, threads, execution)


def loader_access_assertions(entry):
    """The threadgroup slots each steel ``BlockLoader::load_unsafe`` writes."""
    return gemm_layout.model_loader_assertions(signature(entry)["model"], GEMM_SOURCE)


def gemm_index_assertions(entry):
    """OpenGL ranges for the steel kernel's 64-bit pointer offsets.

    As ``test_current_gemm_masked`` asserts them: device views and the
    loaders' reads stay below 2^31 elements, masks are read through views of
    their own element type, and each threadgroup view reaches only the slots
    its loader writes.
    """
    info = signature(entry)
    found = info["model"]
    element = {"float32": "float", "float16": "float16", "bfloat16": "bfloat16"}[
        info["dtype"]
    ]
    reads = max(loader.reads for loader in found.loaders)
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
    kinds = {found.mask_kind("out"), found.mask_kind("op")}
    masks = [
        {
            "expression": "reference.offset + index",
            "function": f"crosstl_resource_load_{element_type}",
            "maximum": SIGNED_MAX,
        }
        for kind, element_type in (("bool", "bool"), ("nomask", "_NoMask"))
        if kind in kinds
    ]
    return tuple(
        {"source": GEMM_SOURCE, **item, "minimum": 0}
        for item in device + workgroup + masks
    )


def gemv_index_assertions(entry):
    signature(entry)
    return tuple(
        {
            "source": GEMV_SOURCE,
            "expression": expression,
            "minimum": 0,
            "maximum": SIGNED_MAX,
        }
        for expression in GEMV_INDEX_RANGE_ASSERTIONS
    )


def _check_transpose(shape, strides, is_vector):
    """block_masked_mm.inc's gemv_operand: matmul.cpp check_transpose that
    also copies views with negative strides. ``(transposed, ld, copied)``."""
    if min(strides) >= 0:
        transposed, ld, copied = gemv_layout._check_transpose(shape, strides, is_vector)
        if not copied:
            return transposed, ld, False
    return False, shape[-1], True


def route(dtype, a, b, block_size, out_mask, op_mask):
    """BlockMaskedMM::eval_gpu's route for nonempty operands.

    ``a`` and ``b`` are the ``(shape, strides)`` the primitive receives, both
    broadcast to the output's batch shape. ``out_mask`` and ``op_mask`` are
    the masks' kernel-name tokens, ``"nomask"`` when absent. Returns the
    kind, the entries in launch order and the operand copies made first.
    """
    (a_shape, a_strides), (b_shape, b_strides) = a, b
    m, k, n = a_shape[-2], a_shape[-1], b_shape[-1]
    transpose_a, lda, a_copy = _check_transpose(a_shape, a_strides, m == 1)
    transpose_b, ldb, b_copy = _check_transpose(b_shape, b_strides, n == 1)
    batch_shape = list(a_shape[:-2]) or [1]
    record = {"M": m, "N": n, "K": k, "copies": {"a": a_copy, "b": b_copy}}
    if min(m, n) == 1:
        is_b_matrix = n != 1
        transpose_mat = not transpose_b if is_b_matrix else transpose_a
        out_size = n if is_b_matrix else m
        entry = select_gemv(
            dtype,
            transpose_mat,
            out_size,
            block_size,
            out_mask,
            op_mask,
            len(batch_shape) > 1,
        )
        return {**record, "kind": "masked-mv", "entries": [entry]}
    entry = select_gemm(
        dtype, transpose_a, transpose_b, m, n, k, block_size, out_mask, op_mask
    )
    return {
        **record,
        "kind": "masked-mm",
        "entries": [entry],
        "lda": lda,
        "ldb": ldb,
    }
