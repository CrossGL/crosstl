"""Validate MLX RMS and layer normalization launches before native dispatch.

The pinned rms_norm.metal and layer_norm.metal instantiate, for float32,
float16 and bfloat16, a single-row and a looped forward kernel and a
single-row and a looped vector-Jacobian kernel each: ``rms<type>``,
``rms_looped<type>``, ``vjp_rms<type>``, ``vjp_rms_looped<type>`` and the
``layer_norm`` counterparts. ``RMSNorm``, ``LayerNorm``, ``RMSNormVJP`` and
``LayerNormVJP::eval_gpu`` in normalization.cpp bind last-axis-contiguous
rows and launch one threadgroup per row along ``x`` (``vjp_rms`` covers
``ceil(rows / 512)`` rows per threadgroup). A row that fits the single-row
kernel runs in ``ceil(axis_size / reads)`` threads rounded up to a
simdgroup; a longer row runs the looped kernel in the pipeline's 1024
threads. RMS kernels read four values per thread and loop past 4096 values;
layer normalization reads eight and loops past 6656 forward values and 8192
gradient values, where its looped kernels read four. Every check below is
that plan's; a launch outside it is refused before any package is translated.

The gradient kernels specialize ``has_w`` (function constant 20), which
normalization.cpp names in the kernel's hash name: host entries are
``vjp_*<type>_w`` and ``vjp_*<type>_now``. Without a weight, the kernels
bind the weight scalar as the ``gw`` output and never write it.

The kernels size their work by the launch, so each width is its own package,
keyed ``<entry>@<width>``. OpenGL keeps the kernels' 64-bit row offsets in
signed 32-bit logical indices; ``plan`` refuses a launch whose furthest
offset, including pointers advanced past a row's padded reads, leaves that
range.
"""

import ctypes
from types import SimpleNamespace

SOURCES = {
    "rms": "mlx/backend/metal/kernels/rms_norm.metal",
    "layer_norm": "mlx/backend/metal/kernels/layer_norm.metal",
}
DTYPES = ("float32", "float16", "bfloat16")
# normalization.cpp's reads per thread and looped limits: (single-row reads,
# looped reads, forward looped limit, gradient looped limit).
LIMITS = {"rms": (4, 4, 4096, 4096), "layer_norm": (8, 4, 6656, 8192)}
WEIGHT_SUFFIXES = {True: "_w", False: "_now"}
SIMD_SIZE = 32
LOOPED_WIDTH = 1024
# RMSNormVJP::eval_gpu's target of 512 single-row threadgroups.
TARGET_GROUPS = 512
# The bounds the host's other row families use: every bound buffer stays far
# below the INT32_MAX OpenGL's signed offsets allow, and one launch axis
# carries at most 65535 workgroups.
MAX_ELEMENTS = 2**28 - 1
MAX_GROUPS = 65535
SIGNED_MAX = 2**31 - 1
SEPARATOR = "@"
SCALARS = ("eps", "axis_size", "w_stride", "b_stride", "n_rows", "rows_per_group")
# The element type each scalar parameter binds; ``eps`` travels as its bits.
SCALAR_TYPES = {name: "uint32" for name in SCALARS} | {"eps": "float32"}


def _entries():
    entries = {}
    for operation in SOURCES:
        for looped in (False, True):
            mode = "_looped" if looped else ""
            for dtype in DTYPES:
                entries[f"{operation}{mode}{dtype}"] = (
                    operation,
                    False,
                    looped,
                    dtype,
                    None,
                )
                for has_w, suffix in WEIGHT_SUFFIXES.items():
                    entries[f"vjp_{operation}{mode}{dtype}{suffix}"] = (
                        operation,
                        True,
                        looped,
                        dtype,
                        has_w,
                    )
    return entries


# Host entry -> (operation, gradient, looped, element type, has_w).
ENTRIES = _entries()
# Entry prefixes whose trace records the audit reads.
PREFIXES = tuple(f"{kind}{operation}" for operation in SOURCES for kind in ("", "vjp_"))


def signature(entry):
    """``(operation, gradient, looped, element type, has_w)`` of a host entry."""
    try:
        return ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unsupported normalization entry") from error


def source(entry):
    """The pinned source that instantiates ``entry``."""
    return SOURCES[signature(entry)[0]]


def source_entry(entry):
    """The instantiated kernel name; gradient entries drop their hash suffix."""
    _, gradient, _, _, has_w = signature(entry)
    return entry[: -len(WEIGHT_SUFFIXES[has_w])] if gradient else entry


def constants(entry):
    """The function constants a gradient entry specializes."""
    _, gradient, _, _, has_w = signature(entry)
    return {"has_w": has_w} if gradient else {}


def reads(entry):
    """The values each thread reads per pass."""
    operation, _, looped, _, _ = signature(entry)
    single, repeated, _, _ = LIMITS[operation]
    return repeated if looped else single


def entry_name(operation, gradient, dtype, axis_size, has_w=None):
    """The kernel ``eval_gpu`` selects for one row length."""
    if operation not in SOURCES or dtype not in DTYPES:
        raise ValueError("Unsupported normalization operation")
    if gradient != (has_w is not None) or type(axis_size) is not int:
        raise ValueError("Unsupported normalization operation")
    limit = LIMITS[operation][3 if gradient else 2]
    mode = "_looped" if axis_size > limit else ""
    entry = f"{'vjp_' if gradient else ''}{operation}{mode}{dtype}"
    if gradient:
        entry += WEIGHT_SUFFIXES[has_w]
    signature(entry)
    return entry


def operand_types(entry):
    """Every source parameter's element type, in source order."""
    operation, gradient, looped, dtype, _ = signature(entry)
    if gradient:
        names = ("x", "w", "g", "gx", "gw", "eps", "axis_size", "w_stride")
        if operation == "rms" and not looped:
            names += ("n_rows", "rows_per_group")
    elif operation == "rms":
        names = ("x", "w", "out", "eps", "axis_size", "w_stride")
    else:
        names = ("x", "w", "b", "out", "eps", "axis_size", "w_stride", "b_stride")
    return {name: SCALAR_TYPES.get(name, dtype) for name in names}


def outputs(entry):
    """The buffers a launch writes back."""
    return ("gx", "gw") if signature(entry)[1] else ("out",)


def width(entry, axis_size):
    """``eval_gpu``'s threadgroup width for ``entry`` over rows of ``axis_size``."""
    operation, gradient, looped, _, _ = signature(entry)
    if type(axis_size) is not int or not 1 <= axis_size <= MAX_ELEMENTS:
        raise ValueError("Normalization rows must be non-empty and bounded")
    if looped != (axis_size > LIMITS[operation][3 if gradient else 2]):
        raise ValueError("Normalization kernel does not match its row length")
    if looped:
        return LOOPED_WIDTH
    threads = -(-axis_size // reads(entry))
    return -(-threads // SIMD_SIZE) * SIMD_SIZE


def variant(entry, size):
    """The cache key of one entry at one threadgroup width."""
    _, _, looped, _, _ = signature(entry)
    widths = (
        (LOOPED_WIDTH,) if looped else range(SIMD_SIZE, LOOPED_WIDTH + 1, SIMD_SIZE)
    )
    if type(size) is not int or size not in widths:
        raise ValueError("Normalization threadgroup width exceeds its bounds")
    return f"{entry}{SEPARATOR}{size}"


def split_variant(key):
    """``(entry, workgroup)`` of a cache key ``variant`` built."""
    if not isinstance(key, str):
        raise ValueError("Invalid normalization variant")
    entry, separator, size = key.partition(SEPARATOR)
    if not separator or not size.isdigit() or variant(entry, int(size)) != key:
        raise ValueError("Invalid normalization variant")
    return entry, (int(size), 1, 1)


def rows_per_group(entry, rows):
    """Rows each threadgroup covers: ``vjp_rms`` targets 512 threadgroups."""
    operation, gradient, looped, _, _ = signature(entry)
    if operation == "rms" and gradient and not looped:
        return -(-rows // TARGET_GROUPS)
    return 1


def index_assertions(entry):
    """The OpenGL offsets the host bounds to ``[0, INT32_MAX]``.

    Each lane's pointers start at its row's first element plus its reads,
    and the weight and bias pointers at their stride times the same reads;
    every subscript then adds at most one padded row of reads. ``plan``
    bounds the furthest such offset of the launch. OpenGL proves a composed
    pointer's subscript by its complete ``<pointer> + <index>`` key.
    """
    count = reads(entry)
    # The kernels' read count appears as the template parameter, its
    # rms_norm.metal default constant or its value.
    spellings = ("N_READS", "RMS_N_READS", count)
    # Row offsets widen either the row (``size_t(row) * axis_size``) or the
    # axis size (``gid * size_t(axis_size)``) before multiplying.
    row_offsets = (
        "gid * uint64(axis_size)",
        "uint64(row) * axis_size",
        "uint64(gid) * axis_size",
    )
    bases = tuple(
        f"{offset} + lid * {per_lane}"
        for offset in row_offsets
        for per_lane in spellings
    )
    rows = (*row_offsets, "uint64(row)", "uint64(gid)")
    strided = tuple(
        f"{stride} * lid * {per_lane}"
        for stride in ("w_stride", "b_stride")
        for per_lane in spellings
    )
    relative = ("i + r", "r + i")
    weighted = tuple(
        index
        for stride in ("w_stride", "b_stride")
        for index in (
            f"{stride} * i",
            f"{stride} * (i + r)",
            f"i * {stride}",
            f"(i + r) * {stride}",
        )
    )
    composed = (
        *(
            f"{pointer} + {index}"
            # GLSL renames the reserved ``out`` parameter ``out_``.
            for pointer in ("x", "g", "out_", "gx", "gw", "x_in", "g_in", "gx_out")
            for index in ("i", *relative)
        ),
        *(
            f"{pointer} + {index}"
            for pointer in ("w", "b")
            for index in weighted
            if index.count(f"{pointer}_stride")
        ),
    )
    return (*bases, *rows, *strided, *relative, *weighted, *composed)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def plan(entry, metadata, counts, logical_size):
    """Check one launch's scalars and extents against ``eval_gpu``.

    ``metadata`` holds every scalar parameter's value (``eps`` as its float32
    bits), ``counts`` every buffer's bound extent and ``logical_size`` the
    input's data size the host passes. Returns the launch the host must use.
    """
    operation, gradient, _, _, has_w = signature(entry)
    types = operand_types(entry)
    _require(
        set(metadata) == {name for name in types if name in SCALAR_TYPES}
        and set(counts) == set(types)
        and all(type(value) is int for value in [*metadata.values(), *counts.values()]),
        "Normalization launch does not bind its kernel's parameters",
    )
    axis_size = metadata["axis_size"]
    size = width(entry, axis_size)
    _require(
        logical_size == counts["x"]
        and 1 <= counts["x"] <= MAX_ELEMENTS
        and counts["x"] % axis_size == 0,
        "Normalization input does not cover whole rows",
    )
    rows = counts["x"] // axis_size
    per_group = rows_per_group(entry, rows)
    groups = -(-rows // per_group)
    _require(groups <= MAX_GROUPS, "Normalization launch exceeds its group limit")
    if "n_rows" in metadata:
        _require(
            metadata["n_rows"] == rows and metadata["rows_per_group"] == per_group,
            "Normalization gradient rows do not match their launch",
        )
    expected = {name: counts["x"] for name in ("x", "g", "gx", "out") if name in types}
    if gradient:
        expected["gw"] = groups * axis_size if has_w else 1
    _require(
        all(counts[name] == value for name, value in expected.items()),
        "Normalization buffers do not match their rows",
    )
    # Each lane's pointer starts at its first read and the looped kernels
    # advance a whole threadgroup of reads past the row's end.
    span = -(-axis_size // (reads(entry) * size)) * reads(entry) * size
    _require(
        (rows - 1) * axis_size + span <= SIGNED_MAX,
        "Normalization offsets exceed signed 32-bit indices",
    )
    for name in ("w", "b"):
        if name not in types:
            continue
        stride = metadata[f"{name}_stride"]
        # A broadcast or absent parameter binds one element at stride zero.
        _require(
            1 <= counts[name] <= MAX_ELEMENTS
            and (counts[name] >= (axis_size - 1) * stride + 1 if stride else True)
            and stride * span <= SIGNED_MAX,
            "Normalization weight or bias does not cover its row",
        )
    if gradient and not has_w:
        _require(
            metadata["w_stride"] == 0,
            "Normalization gradient without a weight binds a strided weight",
        )
    return {"workgroupCount": [groups, 1, 1], "workgroupSize": [size, 1, 1]}


def _read(buffer):
    # Every scalar is one 32-bit word; ``eps`` keeps its float32 bits.
    return ctypes.cast(buffer.data, ctypes.POINTER(ctypes.c_uint32)).contents.value


def _check_buffers(entry, buffers):
    types = operand_types(entry)
    written = outputs(entry)
    if set(buffers) != set(types):
        raise ValueError("Normalization buffers do not match their kernel")
    for name, kind in types.items():
        buffer = buffers[name]
        maximum = 1 if name in SCALAR_TYPES else MAX_ELEMENTS
        if (
            buffer.dtype != kind.encode("ascii")
            or not buffer.data
            or not 1 <= buffer.count <= maximum
            or buffer.output != int(name in written)
        ):
            raise ValueError("Normalization buffer type, extent or direction differs")


def metadata_of(entry, buffers):
    """Every scalar parameter's value; ``eps`` as its float32 bits."""
    return {
        name: _read(buffers[name])
        for name in operand_types(entry)
        if name in SCALAR_TYPES
    }


def validate(entry, buffers, logical_size, execution):
    """Check one launch against ``eval_gpu`` and return its scalar metadata."""
    signature(entry)
    _check_buffers(entry, buffers)
    metadata = metadata_of(entry, buffers)
    counts = {name: buffer.count for name, buffer in buffers.items()}
    if plan(entry, metadata, counts, logical_size) != execution:
        raise ValueError("Native normalization launch does not match the upstream plan")
    return metadata


def validate_metadata(entry, metadata, counts, logical_size, execution):
    """Validate a detached launch, binding its values as the native host does.

    ``counts`` names every element buffer's extent; scalars are range-checked
    and bound again as the 32-bit words the native host passes.
    """
    types = operand_types(entry)
    if not isinstance(metadata, dict) or not isinstance(counts, dict):
        raise ValueError("Invalid normalization layout")
    storage, buffers = {}, {}
    for name, kind in types.items():
        if name in SCALAR_TYPES:
            value = metadata.get(name)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError("Invalid normalization metadata values")
            words, count = ctypes.c_uint32(value), 1
        else:
            count = counts.get(name)
            if type(count) is not int:
                raise ValueError("Invalid normalization buffer extent")
            # Element storage is not read here; one byte stands in for it.
            words = (ctypes.c_uint8 * 1)()
        storage[name] = words
        buffers[name] = SimpleNamespace(
            dtype=kind.encode("ascii"),
            count=count,
            data=ctypes.addressof(words),
            output=int(name in outputs(entry)),
        )
    if set(metadata) != set(metadata_of(entry, buffers)):
        raise ValueError("Invalid normalization layout")
    return validate(entry, buffers, logical_size, execution)
