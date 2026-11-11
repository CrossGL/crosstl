"""Validate MLX fused cross-entropy launches before native dispatch.

The pinned cross_entropy.metal instantiates ``cross_entropy_<type>`` and
``cross_entropy_vjp_<type>`` for float32, float16 and bfloat16.
``CrossEntropy::eval_gpu`` in cross_entropy.cpp binds row-contiguous logits,
their int32 class indices, the float32 losses and ``int axis_size``;
``CrossEntropyVJP::eval_gpu`` binds the logits, the indices, the losses, the
float32 cotangents, the gradient in the logits' type and ``int axis_size``.
Both launch one threadgroup per row along ``x`` in ``ceil(axis_size / 4)``
threads rounded up to a simdgroup and clamped to ``[32, 1024]``, the
pipelines' maximum. Every check below is that plan's; a launch outside it is
refused before any package is translated.

The kernels read the logit a row's class index names. MLX does not bound the
indices, so the host refuses a launch with any index outside
``[-axis_size, axis_size)`` rather than read outside the row.

The kernels size their work by the launch, so each width is its own package,
keyed ``<entry>@<width>``. OpenGL keeps the kernels' 64-bit row offsets in
signed 32-bit logical indices; ``plan`` refuses a launch whose furthest
offset, the padded reads of the last row included, leaves that range.
"""

import ctypes
from types import SimpleNamespace

SOURCE = "mlx/backend/metal/kernels/cross_entropy.metal"
DTYPES = ("float32", "float16", "bfloat16")
# instantiate_cross_entropy: host entry -> (gradient, element type).
ENTRIES = {
    f"cross_entropy{kind}_{dtype}": (gradient, dtype)
    for dtype in DTYPES
    for kind, gradient in (("", False), ("_vjp", True))
}
INPUT, TARGETS, LOSS, COTAN, OUTPUT = "in", "targets", "loss", "cotan", "out"
AXIS_SIZE = "axis_size"
# CROSS_ENTROPY_N_READS in the pinned kernels' defines.
N_READS = 4
SIMD_SIZE = 32
MAX_WIDTH = 1024
# The bounds the host's other row families use: every bound buffer stays far
# below the INT32_MAX OpenGL's signed offsets allow, and one launch axis
# carries at most 65535 workgroups.
MAX_ELEMENTS = 2**28 - 1
MAX_GROUPS = 65535
SIGNED_MAX = 2**31 - 1
SEPARATOR = "@"
# Entry prefix whose trace records keep their values for the audit.
PREFIX = "cross_entropy_"


def signature(entry):
    """``(gradient, element type)`` of a cross-entropy entry."""
    try:
        return ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unsupported cross-entropy entry") from error


def entry_name(gradient, dtype):
    """The kernel ``eval_gpu`` selects for logits of ``dtype``."""
    entry = f"cross_entropy{'_vjp' if gradient else ''}_{dtype}"
    signature(entry)
    return entry


def operand_types(entry):
    """Every source parameter's element type, in source order."""
    gradient, dtype = signature(entry)
    if gradient:
        return {
            INPUT: dtype,
            TARGETS: "int32",
            LOSS: "float32",
            COTAN: "float32",
            OUTPUT: dtype,
            AXIS_SIZE: "int32",
        }
    return {INPUT: dtype, TARGETS: "int32", OUTPUT: "float32", AXIS_SIZE: "int32"}


def inputs_of(entry):
    """The element parameters ``entry`` reads without writing."""
    return tuple(
        name for name in operand_types(entry) if name not in (OUTPUT, AXIS_SIZE)
    )


def width(axis_size):
    """``cross_entropy_dims``' threadgroup width over rows of ``axis_size``."""
    if type(axis_size) is not int or not 1 <= axis_size <= MAX_ELEMENTS:
        raise ValueError("Cross-entropy rows must be non-empty and bounded")
    needed = -(-axis_size // N_READS)
    return min(max(-(-needed // SIMD_SIZE) * SIMD_SIZE, SIMD_SIZE), MAX_WIDTH)


def variant(entry, size):
    """The cache key of one entry at one threadgroup width."""
    signature(entry)
    if type(size) is not int or size not in range(SIMD_SIZE, MAX_WIDTH + 1, SIMD_SIZE):
        raise ValueError("Cross-entropy threadgroup width exceeds its bounds")
    return f"{entry}{SEPARATOR}{size}"


def split_variant(key):
    """``(entry, workgroup)`` of a cache key ``variant`` built."""
    if not isinstance(key, str):
        raise ValueError("Invalid cross-entropy variant")
    entry, separator, size = key.partition(SEPARATOR)
    if not separator or not size.isdigit() or variant(entry, int(size)) != key:
        raise ValueError("Invalid cross-entropy variant")
    return entry, (int(size), 1, 1)


def index_assertions(entry):
    """The OpenGL offsets the host bounds to ``[0, INT32_MAX]``.

    Each row starts at ``gid * axis_size``; the class index and every column
    stay inside the row's padded reads, and ``plan`` bounds the furthest such
    offset of the launch. OpenGL proves a composed pointer's subscript by its
    complete ``<pointer> + <index>`` key.
    """
    if signature(entry)[0]:
        return ("row_offset", "y_n", "in_ + y_n", "col", "in_ + col", "out_ + col")
    return (
        "gid * uint64(axis_size)",
        "y_n",
        "in_ + y_n",
        "offset + i",
        "in_ + offset + i",
    )


def plan(entry, axis_size, counts, logical_size):
    """Check one launch's extents against ``eval_gpu``; returns its launch.

    ``counts`` holds every element buffer's bound extent and ``logical_size``
    the logits' data size the host passes.
    """
    gradient, _ = signature(entry)
    size = width(axis_size)
    logits = counts.get(INPUT)
    if (
        set(counts) != set(operand_types(entry)) - {AXIS_SIZE}
        or any(type(value) is not int for value in counts.values())
        or logical_size != logits
        or not 1 <= logits <= MAX_ELEMENTS
        or logits % axis_size
    ):
        raise ValueError("Cross-entropy logits do not cover whole rows")
    rows = logits // axis_size
    if rows > MAX_GROUPS:
        raise ValueError("Cross-entropy launch exceeds its group limit")
    expected = {TARGETS: rows, OUTPUT: logits if gradient else rows}
    if gradient:
        expected |= {LOSS: rows, COTAN: rows}
    if any(counts[name] != value for name, value in expected.items()):
        raise ValueError("Cross-entropy buffers do not match their rows")
    # The last row's lanes form offsets up to its padded reads.
    span = -(-axis_size // (N_READS * size)) * N_READS * size
    if (rows - 1) * axis_size + span > SIGNED_MAX:
        raise ValueError("Cross-entropy offsets exceed signed 32-bit indices")
    return {"workgroupCount": [rows, 1, 1], "workgroupSize": [size, 1, 1]}


def require_targets(targets, axis_size):
    """Every class index must name a column of its row, counted from either end."""
    if any(not -axis_size <= value < axis_size for value in targets):
        raise ValueError("Cross-entropy class indices must lie inside their rows")


def _check_buffers(entry, buffers):
    types = operand_types(entry)
    if set(buffers) != set(types):
        raise ValueError("Cross-entropy buffers do not match their kernel")
    for name, kind in types.items():
        buffer = buffers[name]
        maximum = 1 if name == AXIS_SIZE else MAX_ELEMENTS
        if (
            buffer.dtype != kind.encode("ascii")
            or not buffer.data
            or not 1 <= buffer.count <= maximum
            or buffer.output != int(name == OUTPUT)
        ):
            raise ValueError("Cross-entropy buffer type, extent or direction differs")


def validate(entry, buffers, logical_size, execution):
    """Check one launch against ``eval_gpu`` and return its row length."""
    signature(entry)
    _check_buffers(entry, buffers)
    axis_size = ctypes.cast(
        buffers[AXIS_SIZE].data, ctypes.POINTER(ctypes.c_int32)
    ).contents.value
    counts = {
        name: buffer.count for name, buffer in buffers.items() if name != AXIS_SIZE
    }
    if plan(entry, axis_size, counts, logical_size) != execution:
        raise ValueError("Native cross-entropy launch does not match the upstream plan")
    return axis_size


def validate_metadata(entry, axis_size, counts, logical_size, execution):
    """Validate a detached launch, binding its values as the native host does."""
    types = operand_types(entry)
    if (
        type(axis_size) is not int
        or not -(2**31) <= axis_size < 2**31
        or not isinstance(counts, dict)
        or set(counts) != set(types) - {AXIS_SIZE}
    ):
        raise ValueError("Invalid cross-entropy layout")
    storage = ctypes.c_int32(axis_size)
    placeholder = (ctypes.c_uint8 * 1)()
    buffers = {
        name: SimpleNamespace(
            dtype=kind.encode("ascii"),
            count=1 if name == AXIS_SIZE else counts[name],
            data=ctypes.addressof(storage if name == AXIS_SIZE else placeholder),
            output=int(name == OUTPUT),
        )
        for name, kind in types.items()
    }
    return validate(entry, buffers, logical_size, execution)
