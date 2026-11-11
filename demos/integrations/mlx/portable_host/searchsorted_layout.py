"""Validate searchsorted.metal launches and their buffers.

``SearchSorted::eval_gpu`` binds the sorted sequence ``a`` with its element
stride, the row-contiguous values ``v``, the uint32 output and the sequence
length ``n``. Every thread binary-searches one value. The host launches at
most ``MAX_GROUPS`` one-thread groups per pass and advances ``v`` and ``out``
between passes, so a pass's thread grid is exactly its value count; the
kernel derives each thread's element from that grid, as upstream's
``get_2d_grid_dims`` grid does.
"""

import ctypes

from demos.integrations.mlx.portable_host.element_storage import CTYPES

SOURCE = "mlx/backend/metal/kernels/searchsorted.metal"
# instantiate_searchsorted in the pinned searchsorted.metal.
DTYPES = (
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
    "complex64",
)
SIDES = ("left", "right")
ENTRIES = {
    f"searchsorted_{dtype}_{side}": (dtype, side == "right")
    for dtype in DTYPES
    for side in SIDES
}
PREFIX = "searchsorted_"
OUTPUT_TYPE = "uint32"
METADATA = {"n": "uint32", "a_stride": "int64"}
MAX_GROUPS = 65535
# The largest sequence span a pass binds. Every index the kernel forms,
# ``mid * a_stride``, is below it; OpenGL is translated with that host
# precondition.
MAX_SPAN = 2**28 - 1
SEPARATOR = "@"
_METADATA_CTYPES = {"uint32": ctypes.c_uint32, "int64": ctypes.c_int64}


def signature(entry):
    """The ``(element type, right)`` of a searchsorted entry."""
    try:
        return ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown searchsorted entry") from error


def operand_types(entry):
    """Every source parameter's element type, in parameter order."""
    dtype, _ = signature(entry)
    return {"a": dtype, "v": dtype, "out": OUTPUT_TYPE, **METADATA}


def variant(entry, count):
    """The cache key of ``entry`` launched over ``count`` values."""
    signature(entry)
    if type(count) is not int or not 1 <= count <= MAX_GROUPS:
        raise ValueError("Searchsorted pass exceeds its group limit")
    return f"{entry}{SEPARATOR}{count}"


def split_variant(key):
    """The ``(entry, count)`` a cache key names."""
    if not isinstance(key, str) or key.count(SEPARATOR) != 1:
        raise ValueError("Unknown searchsorted variant")
    entry, count = key.split(SEPARATOR)
    if not count.isdigit() or count != str(int(count)):
        raise ValueError("Unknown searchsorted variant")
    if variant(entry, int(count)) != key:
        raise ValueError("Unknown searchsorted variant")
    return entry, int(count)


def execution(count):
    """The launch of one pass over ``count`` values."""
    if type(count) is not int or not 1 <= count <= MAX_GROUPS:
        raise ValueError("Searchsorted pass exceeds its group limit")
    return {"workgroupCount": [count, 1, 1], "workgroupSize": [1, 1, 1]}


def sequence_span(n, stride):
    """The elements of ``a`` a search over ``n`` with ``stride`` can read."""
    return 1 + (n - 1) * stride


def check_metadata(metadata, counts):
    """Check a pass's sequence metadata against its buffer extents."""
    n, stride = metadata.get("n"), metadata.get("aStride")
    if (
        set(metadata) != {"n", "aStride"}
        or type(n) is not int
        or type(stride) is not int
        or not 1 <= n <= 2**32 - 1
        or stride < 0
        or (n > 1 and stride == 0)
        or sequence_span(n, stride) > MAX_SPAN
        or counts["a"] != sequence_span(n, stride)
    ):
        raise ValueError("Searchsorted sequence metadata exceeds its bounds")


def _value(buffer, dtype):
    return ctypes.cast(
        buffer.data, ctypes.POINTER(_METADATA_CTYPES[dtype])
    ).contents.value


def _span(buffer, dtype):
    size = ctypes.sizeof(CTYPES.get(dtype) or _METADATA_CTYPES[dtype])
    return buffer.data, buffer.data + buffer.count * size


def validate(entry, buffers, threads, launch):
    """Check one pass's buffers and launch; returns its sequence metadata."""
    types = operand_types(entry)
    if set(buffers) != set(types):
        raise ValueError("Searchsorted buffers do not match the source contract")
    if type(threads) is not int or launch != execution(threads):
        raise ValueError("Searchsorted launch differs from its value count")
    for name, buffer in buffers.items():
        expected = 1 if name in METADATA else None
        if (
            not buffer.data
            or buffer.dtype != types[name].encode("ascii")
            or buffer.output != int(name == "out")
            or buffer.count < 1
            or (expected is not None and buffer.count != expected)
            or (name in {"v", "out"} and buffer.count != threads)
        ):
            raise ValueError("Searchsorted buffer type, extent or direction differs")
    start, end = _span(buffers["out"], OUTPUT_TYPE)
    for name, buffer in buffers.items():
        other_start, other_end = _span(buffer, types[name])
        if name != "out" and max(start, other_start) < min(end, other_end):
            raise ValueError("Searchsorted output overlaps an input")
    metadata = {
        "n": _value(buffers["n"], "uint32"),
        "aStride": _value(buffers["a_stride"], "int64"),
    }
    check_metadata(metadata, {"a": buffers["a"].count})
    return metadata
