"""Validate MLX ``arg_reduce_general`` launches before native dispatch.

The pinned arg_reduce.metal instantiates ``argmin_<type>`` and
``argmax_<type>`` for every real dtype. ``ArgReduce::eval_gpu`` binds the
input view, a ``uint32`` index output and the shape and strides of every
axis but the reduced one, then launches one threadgroup per output with
``min(ceil(axis_size / 4), 1024)`` threads rounded up to a multiple of 32.
Every check below is that plan's; a launch outside it is refused before
any package is translated.

The kernel reads its threadgroup size, so each width is its own package,
keyed ``<entry>@<width>``. OpenGL keeps the kernel's 64-bit source offsets in
signed 32-bit logical indices; ``index_assertions`` names the offsets the
host bounds, and ``validate`` refuses a launch whose furthest offset,
including the pointer advanced past the last padded read, leaves that range.
"""

import ctypes
from types import SimpleNamespace

SOURCE = "mlx/backend/metal/kernels/arg_reduce.metal"
# instantiate_arg_reduce in the pinned arg_reduce.metal, in source order.
INSTANTIATED = (
    "bool_",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "bfloat16",
)
OPERATIONS = ("argmin", "argmax")
ENTRIES = {
    f"{operation}_{dtype}": (operation, dtype)
    for dtype in INSTANTIATED
    for operation in OPERATIONS
}
INPUT, OUTPUT = "in", "out"
OUTPUT_TYPE = "uint32"
# The kernel's metadata buffers and their element types: size_t is 64 bits.
METADATA = {
    "shape": "int32",
    "in_strides": "int64",
    "out_strides": "int64",
    "ndim": "uint64",
    "axis_stride": "int64",
    "axis_size": "uint64",
}
ARRAYS = ("shape", "in_strides", "out_strides")
METADATA_TYPES = {
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,
}
N_READS = 4
SIMD_SIZE = 32
MAX_WIDTH = 1024
MAX_RANK = 64
# The bounds the host's column reductions use: every bound buffer and stride
# stays far below the INT32_MAX OpenGL's signed offsets allow, and one launch
# axis carries at most 65535 workgroups.
MAX_ELEMENTS = 2**28 - 1
MAX_GROUPS = 65535
SIGNED_MAX = 2**31 - 1
SEPARATOR = "@"
# OpenGL runs simdgroups as 32-lane software subgroups. A threadgroup of more
# than one simdgroup returns every simdgroup but the first before the final
# shuffles, which that lowering cannot synchronize yet (CrossGL/crosstl#2207).
TARGET_WIDTH_LIMITS = {"opengl": SIMD_SIZE}


def signature(entry):
    """``(operation, element type)`` of an ``arg_reduce_general`` entry."""
    try:
        return ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unsupported arg reduction entry") from error


def operand_types(entry):
    """Every source parameter's element type."""
    _, dtype = signature(entry)
    return {INPUT: dtype, OUTPUT: OUTPUT_TYPE, **METADATA}


def width(axis_size):
    """ArgReduce::eval_gpu's threadgroup width for an axis of ``axis_size``."""
    if type(axis_size) is not int or axis_size < 1:
        raise ValueError("Arg reduction axis must be non-empty")
    threads = min((axis_size + N_READS - 1) // N_READS, MAX_WIDTH)
    return (threads + SIMD_SIZE - 1) // SIMD_SIZE * SIMD_SIZE


def variant(entry, size):
    """The cache key of one entry at one threadgroup width."""
    signature(entry)
    if type(size) is not int or size % SIMD_SIZE or not 0 < size <= MAX_WIDTH:
        raise ValueError("Arg reduction threadgroup width exceeds its bounds")
    return f"{entry}{SEPARATOR}{size}"


def split_variant(key):
    """``(entry, workgroup)`` of a cache key ``variant`` built."""
    if not isinstance(key, str):
        raise ValueError("Invalid arg reduction variant")
    entry, separator, size = key.partition(SEPARATOR)
    if not separator or not size.isdigit() or variant(entry, int(size)) != key:
        raise ValueError("Invalid arg reduction variant")
    return entry, (int(size), 1, 1)


def index_assertions():
    """The OpenGL source offsets the host bounds to ``[0, INT32_MAX]``.

    ``current_in`` starts at the row's element and advances by ``axis_stride``
    after each of the ``N_READS`` reads, padded reads included.
    """
    return (
        "in_idx + current_index * axis_stride",
        "out_idx",
        "current_in + 0",
        "axis_stride",
        "current_in + axis_stride",
    )


def contiguous_strides(shape):
    """Row-contiguous strides of ``shape``, as MLX allocates outputs."""
    strides, size = [], 1
    for extent in reversed(shape):
        strides.append(size)
        size *= extent
    return strides[::-1]


def _read(buffer, kind):
    return list(
        ctypes.cast(
            buffer.data, ctypes.POINTER(METADATA_TYPES[kind] * buffer.count)
        ).contents
    )


def _check_buffers(entry, buffers):
    types = operand_types(entry)
    if set(buffers) != set(types):
        raise ValueError("Arg reduction buffers do not match their kernel")
    for name, kind in types.items():
        buffer = buffers[name]
        maximum = MAX_ELEMENTS if name in (INPUT, OUTPUT) else MAX_RANK
        if (
            buffer.dtype != kind.encode("ascii")
            or not buffer.data
            or not 1
            <= buffer.count
            <= (maximum if name in (INPUT, OUTPUT, *ARRAYS) else 1)
            or buffer.output != int(name == OUTPUT)
        ):
            raise ValueError("Arg reduction buffer type, extent or direction differs")


def plan(metadata, input_count, output_count, logical_size):
    """Check metadata against ArgReduce::eval_gpu; returns the launch it implies.

    ``input_count`` is the bound source span and ``output_count`` the index
    output's extent. The returned execution is the launch the host must use.
    """
    ndim = metadata["ndim"][0]
    if not 0 <= ndim <= MAX_RANK:
        raise ValueError("Arg reduction rank exceeds its bounds")
    rank = max(ndim, 1)
    if any(len(metadata[name]) != rank for name in ARRAYS):
        raise ValueError("Arg reduction metadata ranks do not match")
    shape = metadata["shape"][:ndim]
    in_strides = metadata["in_strides"][:ndim]
    out_strides = metadata["out_strides"][:ndim]
    if ndim == 0 and any(metadata[name] != [0] for name in ARRAYS):
        raise ValueError("Arg reduction placeholders must be zero")
    axis_size, axis_stride = metadata["axis_size"][0], metadata["axis_stride"][0]
    rows = 1
    for extent in shape:
        if not 1 <= extent <= MAX_GROUPS:
            raise ValueError("Arg reduction shape exceeds its bounds")
        rows *= extent
    if (
        not 1 <= axis_size <= MAX_ELEMENTS
        or not 0 <= axis_stride <= MAX_ELEMENTS
        or any(not 0 <= stride <= MAX_ELEMENTS for stride in in_strides)
    ):
        raise ValueError("Arg reduction axis or strides exceed their bounds")
    if out_strides != contiguous_strides(shape):
        raise ValueError("Arg reduction output is not row contiguous")
    if rows > MAX_GROUPS or output_count != rows:
        raise ValueError("Arg reduction output does not match its rows")
    if logical_size != rows * axis_size:
        raise ValueError("Arg reduction logical size does not match its plan")
    row_span = sum((extent - 1) * stride for extent, stride in zip(shape, in_strides))
    span = row_span + (axis_size - 1) * axis_stride + 1
    if input_count != span or span > MAX_ELEMENTS:
        raise ValueError("Arg reduction source span does not match its view")
    size = width(axis_size)
    # current_in advances past every read of the last bucket, padded or not.
    rounded = -(-axis_size // (N_READS * size)) * N_READS * size
    if row_span + rounded * axis_stride > SIGNED_MAX:
        raise ValueError("Arg reduction source offsets exceed signed 32-bit indices")
    return {"workgroupCount": [1, rows, 1], "workgroupSize": [size, 1, 1]}


def validate(entry, buffers, logical_size, execution):
    """Check one launch against ArgReduce::eval_gpu and return its metadata."""
    signature(entry)
    _check_buffers(entry, buffers)
    if type(logical_size) is not int:
        raise ValueError("Arg reduction logical size does not match its plan")
    metadata = {name: _read(buffers[name], kind) for name, kind in METADATA.items()}
    expected = plan(metadata, buffers[INPUT].count, buffers[OUTPUT].count, logical_size)
    if execution != expected:
        raise ValueError("Native arg reduction launch does not match the upstream plan")
    return metadata


def validate_metadata(
    entry, metadata, logical_size, input_count, output_count, execution
):
    """Validate a detached launch's metadata, as ``validate`` checks a launch.

    The values are range-checked, bound again as the buffers the native host
    passes and run through the same plan.
    """
    signature(entry)
    if (
        not isinstance(metadata, dict)
        or set(metadata) != set(METADATA)
        or any(type(value) is not int for value in (input_count, output_count))
    ):
        raise ValueError("Invalid arg reduction layout")
    storage, buffers = {}, {}
    for name, kind in operand_types(entry).items():
        if name in (INPUT, OUTPUT):
            # Element storage is not read here; one byte stands in for it.
            words = (ctypes.c_uint8 * 1)()
            count = input_count if name == INPUT else output_count
        else:
            values = metadata[name]
            bits = 32 if kind == "int32" else 64
            low = 0 if kind == "uint64" else -(1 << (bits - 1))
            high = (1 << (bits if kind == "uint64" else bits - 1)) - 1
            if (
                not isinstance(values, list)
                or not 1 <= len(values) <= (MAX_RANK if name in ARRAYS else 1)
                or any(
                    type(value) is not int or not low <= value <= high
                    for value in values
                )
            ):
                raise ValueError("Invalid arg reduction metadata values")
            words, count = (METADATA_TYPES[kind] * len(values))(*values), len(values)
        storage[name] = words
        buffers[name] = SimpleNamespace(
            dtype=kind.encode("ascii"),
            count=count,
            data=ctypes.addressof(words),
            output=int(name == OUTPUT),
        )
    return validate(entry, buffers, logical_size, execution)
