"""Storage and launch contracts for MLX's floating-point dot-product kernel.

``Matmul::eval_gpu`` in ``backend.cpp`` sends a product with one output
element, no batch and row-contiguous operands to ``dot_product`` as
matmul.cpp does, after the operand preparation ``gemv_layout.operands``
mirrors. Each 512-thread group of the pinned dot.metal reads 32 elements per
thread and writes one float32 partial sum; ``all_reduce_dispatch`` sums the
partials in one pass, or in two once there are more than 4,096 of them, and
``copy_gpu`` converts the float32 total to the output type.
"""

import ctypes
import re

from demos.integrations.mlx.portable_host.quantization_layout import ITEM_SIZES

SOURCE = "mlx/backend/metal/kernels/dot.metal"
SIMD_SIZE = 32
ITEMS_PER_THREAD = 32
THREADGROUP_SIZE = 512
SIMD_GROUPS = THREADGROUP_SIZE // SIMD_SIZE
# Every buffer the host binds holds at most this many elements, so each
# element index the kernel forms in 32 bits stays below 2^31 - 1.
MAX_ELEMENTS = 2**28 - 1
MAX_GROUPS = 65535
# reduce.cpp all_reduce_dispatch: REDUCE_N_READS * 1024 elements take one pass.
ONE_PASS_ELEMENTS = 4 * 1024
DTYPES = ("float32", "float16", "bfloat16")
PATTERN = re.compile(
    r"dot_product_(float32|float16|bfloat16)"
    rf"_it{ITEMS_PER_THREAD}_tg{THREADGROUP_SIZE}_sg{SIMD_GROUPS}"
)
ENTRY_PREFIXES = ("dot_product_",)
# [[buffer(i)]] parameters of dot.h's kernel, in index order; the host binds
# every one of them.
PARAMETERS = ("a", "b", "output", "n")
HOST_PARAMETERS = PARAMETERS
VECTOR_PARAMETERS = ("a", "b")
INTEGER_TYPES = {"int32": ctypes.c_int32}


def signature(entry):
    """The element type of a dot-product entry."""
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError("Unsupported dot-product entry")
    return {"dtype": match.group(1)}


def workgroup_size(entry):
    signature(entry)
    return (THREADGROUP_SIZE, 1, 1)


def select(dtype):
    """matmul.cpp dot_product's kernel name."""
    if dtype not in DTYPES:
        raise ValueError("Dot products support float32, float16 and bfloat16")
    return (
        f"dot_product_{dtype}_it{ITEMS_PER_THREAD}_tg{THREADGROUP_SIZE}"
        f"_sg{SIMD_GROUPS}"
    )


def groups(k):
    """dot_product's threadgroup count: one partial sum per group."""
    threads = -(-k // ITEMS_PER_THREAD)
    return -(-threads // THREADGROUP_SIZE)


def reduction_passes(partials):
    """The all_reduce_dispatch passes that sum ``partials`` float32 values."""
    return 1 if partials <= ONE_PASS_ELEMENTS else 2


def route(dtype, a_shape, a_strides, b_shape, b_strides):
    """Matmul::eval_gpu's dot-product route for operands of equal rank.

    Returns the entry, the partial count and the dispatches the product takes:
    the kernel, its partial sums' reduction passes and the output copy.
    """
    from demos.integrations.mlx.portable_host import gemv_layout

    prepared = gemv_layout.operands(a_shape, a_strides, b_shape, b_strides)
    if not prepared["dot"]:
        raise ValueError("Only single row-contiguous products take the dot route")
    k = prepared["K"]
    if not 1 <= k <= MAX_ELEMENTS:
        raise ValueError("Dot-product length exceeds the index bound")
    partials = groups(k)
    return {
        "entry": select(dtype),
        "M": 1,
        "N": 1,
        "K": k,
        "batches": 1,
        "partials": partials,
        "reductionPasses": reduction_passes(partials),
        "kernels": 2 + reduction_passes(partials),
        "copies": prepared["copies"],
    }


def parameter_type(entry, name):
    if name in VECTOR_PARAMETERS:
        return signature(entry)["dtype"]
    if name == "output":
        return "float32"
    if name == "n":
        return "int32"
    raise ValueError("Unknown dot-product parameter")


def read_storage(entry, buffers):
    """Check native buffer identity and read the length.

    Returns ``(counts, parameters)``: element counts of the vectors and the
    partial sums, and the value of ``n``.
    """
    signature(entry)
    if set(buffers) != set(HOST_PARAMETERS):
        raise ValueError("Dot-product buffers do not match the entry")
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    counts, spans = {}, {}
    for name in HOST_PARAMETERS:
        buffer, dtype = buffers[name], parameter_type(entry, name)
        itemsize = ITEM_SIZES.get(dtype) or ctypes.sizeof(INTEGER_TYPES[dtype])
        count = buffer.count
        limit = 1 if name == "n" else MAX_GROUPS if name == "output" else MAX_ELEMENTS
        if (
            not buffer.data
            or buffer.data % itemsize
            or buffer.dtype != dtype.encode("ascii")
            or buffer.output != int(name == "output")
            or type(count) is not int
            or not 1 <= count <= limit
            or buffer.data + count * itemsize > address_limit
        ):
            raise ValueError("Dot-product buffer storage or direction does not match")
        spans[name] = (buffer.data, buffer.data + count * itemsize)
        counts[name] = count
    low, high = spans["output"]
    if any(
        other != "output" and max(low, start) < min(high, end)
        for other, (start, end) in spans.items()
    ):
        raise ValueError("Dot-product output overlaps another buffer")
    n = ctypes.cast(buffers["n"].data, ctypes.POINTER(ctypes.c_int32)).contents.value
    del counts["n"]
    return counts, {"n": n}


def plan(entry, counts, parameters, threads, execution):
    """Validate one launch against matmul.cpp's dot_product and its grid."""
    signature(entry)
    k = parameters.get("n")
    if type(k) is not int or not 1 <= k <= MAX_ELEMENTS:
        raise ValueError("Dot-product length is invalid")
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError("Dot-product launch geometry is missing")
    partials = groups(k)
    if (
        list(execution["workgroupSize"]) != list(workgroup_size(entry))
        or list(execution["workgroupCount"]) != [partials, 1, 1]
        or partials > MAX_GROUPS
    ):
        raise ValueError("Dot-product launch does not match MLX's grid")
    if threads != partials or counts["output"] != partials:
        raise ValueError("Dot-product partial sums differ from its groups")
    if min(counts["a"], counts["b"]) < k:
        raise ValueError("Dot-product vectors are shorter than their length")
    return {"K": k, "partials": partials}
