"""Validate MLX softmax and logsumexp row launches before native dispatch.

The pinned softmax.metal instantiates ``block_softmax_<type>`` and
``looped_softmax_<type>`` for float32, float16 and bfloat16, plus
``*_softmax_precise_<type>`` with a float32 accumulator for the 16-bit types;
logsumexp.metal instantiates ``block_logsumexp_<type>`` and
``looped_logsumexp_<type>`` for the same three types. ``Softmax::eval_gpu``
and ``LogSumExp::eval_gpu`` bind a last-axis-contiguous input, the output and
``int axis_size``, and launch one threadgroup per row along ``x``: rows of at
most 4096 elements run the block kernel in ``ceil(axis_size / 4)`` threads
rounded up to a simdgroup, longer rows run the looped kernel in the pipeline's
maximum of 1024 threads. Softmax writes one element per input element,
logsumexp one per row. Every check below is that plan's; a launch outside it
is refused before any package is translated.

The kernels size their work by the launch, so each width is its own package,
keyed ``<entry>@<width>``. OpenGL keeps the kernels' 64-bit row offsets in
signed 32-bit logical indices; ``validate`` refuses a launch whose furthest
offset, including the pointer advanced past the last padded read, leaves
that range.
"""

import ctypes
from types import SimpleNamespace

SOURCES = {
    "softmax": "mlx/backend/metal/kernels/softmax.metal",
    "logsumexp": "mlx/backend/metal/kernels/logsumexp.metal",
}
DTYPES = ("float32", "float16", "bfloat16")
MODES = ("block", "looped")
# instantiate_softmax and instantiate_logsumexp in the pinned sources:
# (operation, mode, element type, float32 accumulation of a 16-bit type).
ENTRIES = {
    **{
        f"{mode}_softmax_{dtype}": ("softmax", mode, dtype, False)
        for dtype in DTYPES
        for mode in MODES
    },
    **{
        f"{mode}_softmax_precise_{dtype}": ("softmax", mode, dtype, True)
        for dtype in ("float16", "bfloat16")
        for mode in MODES
    },
    **{
        f"{mode}_logsumexp_{dtype}": ("logsumexp", mode, dtype, False)
        for dtype in DTYPES
        for mode in MODES
    },
}
INPUT, OUTPUT, AXIS_SIZE = "in", "out", "axis_size"
N_READS = 4
SIMD_SIZE = 32
LOOPED_LIMIT = 4096
LOOPED_WIDTH = 1024
# The bounds the host's other row families use: every bound buffer stays far
# below the INT32_MAX OpenGL's signed offsets allow, and one launch axis
# carries at most 65535 workgroups.
MAX_ELEMENTS = 2**28 - 1
MAX_GROUPS = 65535
SIGNED_MAX = 2**31 - 1
SEPARATOR = "@"
# Entry prefixes whose trace records keep their values for the audit.
PREFIXES = tuple(f"{mode}_{operation}_" for operation in SOURCES for mode in MODES)


def signature(entry):
    """``(operation, mode, element type, precise)`` of a row entry."""
    try:
        return ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unsupported softmax or logsumexp entry") from error


def source(entry):
    """The pinned source that instantiates ``entry``."""
    return SOURCES[signature(entry)[0]]


def entry_name(operation, dtype, axis_size, precise=False):
    """The kernel ``eval_gpu`` selects for one row length."""
    if operation not in SOURCES or dtype not in DTYPES:
        raise ValueError("Unsupported softmax or logsumexp operation")
    mode = "looped" if axis_size > LOOPED_LIMIT else "block"
    qualifier = "precise_" if precise and dtype != "float32" else ""
    entry = f"{mode}_{operation}_{qualifier}{dtype}"
    signature(entry)
    return entry


def operand_types(entry):
    """Every source parameter's element type."""
    _, _, dtype, _ = signature(entry)
    return {INPUT: dtype, OUTPUT: dtype, AXIS_SIZE: "int32"}


def width(entry, axis_size):
    """``eval_gpu``'s threadgroup width for ``entry`` over rows of ``axis_size``."""
    _, mode, _, _ = signature(entry)
    if type(axis_size) is not int or not 1 <= axis_size <= MAX_ELEMENTS:
        raise ValueError("Softmax and logsumexp rows must be non-empty and bounded")
    if (mode == "looped") != (axis_size > LOOPED_LIMIT):
        raise ValueError("Softmax or logsumexp kernel does not match its row length")
    if mode == "looped":
        return LOOPED_WIDTH
    threads = (axis_size + N_READS - 1) // N_READS
    return (threads + SIMD_SIZE - 1) // SIMD_SIZE * SIMD_SIZE


def variant(entry, size):
    """The cache key of one entry at one threadgroup width."""
    _, mode, _, _ = signature(entry)
    widths = (
        (LOOPED_WIDTH,)
        if mode == "looped"
        else range(SIMD_SIZE, LOOPED_WIDTH + 1, SIMD_SIZE)
    )
    if type(size) is not int or size not in widths:
        raise ValueError("Softmax or logsumexp threadgroup width exceeds its bounds")
    return f"{entry}{SEPARATOR}{size}"


def split_variant(key):
    """``(entry, workgroup)`` of a cache key ``variant`` built."""
    if not isinstance(key, str):
        raise ValueError("Invalid softmax or logsumexp variant")
    entry, separator, size = key.partition(SEPARATOR)
    if not separator or not size.isdigit() or variant(entry, int(size)) != key:
        raise ValueError("Invalid softmax or logsumexp variant")
    return entry, (int(size), 1, 1)


def index_assertions():
    """The OpenGL row offsets the host bounds to ``[0, INT32_MAX]``.

    Each lane's pointer starts at its row's first element plus four reads per
    lane; ``plan`` bounds the furthest such offset of the launch.
    """
    return (
        "gid * uint64(axis_size) + lid * SOFTMAX_N_READS",
        "gid * uint64(axis_size) + lid * 4",
        "gid * uint64(axis_size)",
        *(
            f"{pointer} + {index}"
            for pointer in ("in_", "out_")
            for index in ("i", "offset + i")
        ),
    )


def plan(entry, axis_size, input_count, output_count, logical_size):
    """Check one launch's extents against ``eval_gpu``; returns its launch.

    ``input_count`` is the bound input extent, ``output_count`` the output's
    and ``logical_size`` the input's data size the host passes.
    """
    operation, _, _, _ = signature(entry)
    size = width(entry, axis_size)
    if (
        any(type(value) is not int for value in (input_count, output_count))
        or logical_size != input_count
        or not 1 <= input_count <= MAX_ELEMENTS
        or input_count % axis_size
    ):
        raise ValueError("Softmax or logsumexp input does not cover whole rows")
    rows = input_count // axis_size
    if rows > MAX_GROUPS:
        raise ValueError("Softmax or logsumexp launch exceeds its group limit")
    if output_count != (input_count if operation == "softmax" else rows):
        raise ValueError("Softmax or logsumexp output does not match its rows")
    # Each lane's pointer starts at its first read and the looped kernels
    # advance a whole threadgroup of reads past the row's end.
    rounded = -(-axis_size // (N_READS * size)) * N_READS * size
    if (rows - 1) * axis_size + rounded > SIGNED_MAX:
        raise ValueError("Softmax or logsumexp offsets exceed signed 32-bit indices")
    return {"workgroupCount": [rows, 1, 1], "workgroupSize": [size, 1, 1]}


def _check_buffers(entry, buffers):
    types = operand_types(entry)
    if set(buffers) != set(types):
        raise ValueError("Softmax or logsumexp buffers do not match their kernel")
    for name, kind in types.items():
        buffer = buffers[name]
        maximum = 1 if name == AXIS_SIZE else MAX_ELEMENTS
        if (
            buffer.dtype != kind.encode("ascii")
            or not buffer.data
            or not 1 <= buffer.count <= maximum
            or buffer.output != int(name == OUTPUT)
        ):
            raise ValueError(
                "Softmax or logsumexp buffer type, extent or direction differs"
            )


def validate(entry, buffers, logical_size, execution):
    """Check one launch against ``eval_gpu`` and return its row length."""
    signature(entry)
    _check_buffers(entry, buffers)
    axis_size = ctypes.cast(
        buffers[AXIS_SIZE].data, ctypes.POINTER(ctypes.c_int32)
    ).contents.value
    expected = plan(
        entry,
        axis_size,
        buffers[INPUT].count,
        buffers[OUTPUT].count,
        logical_size,
    )
    if execution != expected:
        raise ValueError(
            "Native softmax or logsumexp launch does not match the upstream plan"
        )
    return axis_size


def validate_metadata(
    entry, axis_size, logical_size, input_count, output_count, execution
):
    """Validate a detached launch, binding its values as the native host does."""
    if (
        any(type(value) is not int for value in (axis_size, input_count, output_count))
        or not -(2**31) <= axis_size < 2**31
    ):
        raise ValueError("Invalid softmax or logsumexp row length")
    storage = ctypes.c_int32(axis_size)
    placeholder = (ctypes.c_uint8 * 1)()
    buffers = {
        name: SimpleNamespace(
            dtype=kind.encode("ascii"),
            count=(
                1
                if name == AXIS_SIZE
                else input_count if name == INPUT else output_count
            ),
            data=ctypes.addressof(storage if name == AXIS_SIZE else placeholder),
            output=int(name == OUTPUT),
        )
        for name, kind in operand_types(entry).items()
    }
    return validate(entry, buffers, logical_size, execution)
