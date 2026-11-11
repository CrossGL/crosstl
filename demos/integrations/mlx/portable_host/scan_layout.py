"""Validate pinned scan storage and preserve source launch geometry."""

import ctypes

MAX_ELEMENTS = 65535
TYPES = {
    "bool_": ctypes.c_uint8,
    "float16": ctypes.c_uint16,
    "bfloat16": ctypes.c_uint16,
    "float32": ctypes.c_uint32,
    "int32": ctypes.c_int32,
    "uint32": ctypes.c_uint32,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,
    # One 64-bit storage word: the real part's float32 bits low, the imaginary
    # part's high (element_storage).
    "complex64": ctypes.c_uint64,
}
ENTRY_PREFIXES = ("contig_scan_", "strided_scan_")
SIGNATURES = {
    f"{layout}_scan_{direction}{mode}_{operation}_{source}_{destination}": (
        layout,
        operation,
        source,
        destination,
    )
    for layout in ("contig", "strided")
    for direction in ("", "reverse_")
    for mode in ("inclusive", "exclusive")
    for operation in ("sum", "prod", "min", "max", "logaddexp")
    for source in TYPES
    for destination in (
        ("int32", "uint32") if source == "bool_" and operation == "sum" else (source,)
    )
    if operation != "logaddexp"
    or source in {"float16", "bfloat16", "float32", "complex64"}
}


def signature(entry):
    try:
        return SIGNATURES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown scalar scan entry") from error


def workgroup_width(axis_size, dtype, *, contiguous):
    """Mirror scan.cpp with the explicit 1024-thread source pipeline limit."""
    if (
        type(axis_size) is not int
        or not 1 <= axis_size <= MAX_ELEMENTS
        or dtype not in TYPES
        or type(contiguous) is not bool
    ):
        raise ValueError("Scan axis or scalar type exceeds the host contract")
    reads = 4 if ctypes.sizeof(TYPES[dtype]) <= 4 else 2
    if not contiguous:
        return 32 * (32 // reads)
    elements_per_subgroup = reads * 32
    if axis_size <= reads * 1024:
        return ((axis_size + elements_per_subgroup - 1) // elements_per_subgroup) * 32
    if axis_size <= reads * 2048:
        return (
            (axis_size // 2 + elements_per_subgroup - 1) // elements_per_subgroup
        ) * 32
    return 1024


def execution(entry, size, axis_size, stride=1):
    layout, _, dtype, _ = signature(entry)
    if (
        type(size) is not int
        or not 1 <= size <= MAX_ELEMENTS
        or type(stride) is not int
        or not 1 <= stride <= MAX_ELEMENTS
    ):
        raise ValueError("Scan storage extent exceeds the host contract")
    width = workgroup_width(axis_size, dtype, contiguous=layout == "contig")
    if (layout == "contig") != (stride == 1) or size % (axis_size * stride):
        raise ValueError("Scan axis and stride do not partition its storage")
    blocks = (stride + 31) // 32
    groups = (size // (axis_size * stride)) * blocks
    if not 1 <= groups <= 65535:
        raise ValueError("Scan launch exceeds the native group limit")
    return {"workgroupCount": [1, groups, 1], "workgroupSize": [width, 1, 1]}


def values(buffer):
    dtype = buffer.dtype.decode("ascii")
    ctype = ctypes.c_uint64 if dtype == "uint64" else TYPES[dtype]
    return list(ctypes.cast(buffer.data, ctypes.POINTER(ctype * buffer.count)).contents)


def validate(entry, buffers, size, launch):
    layout, operation, source, destination = signature(entry)
    types = {"in": source, "out": destination, "axis_size": "uint64"}
    if layout == "strided":
        types.update({"stride": "uint64", "stride_blocks": "uint64"})
    if (
        set(buffers) != set(types)
        or type(size) is not int
        or not 1 <= size <= MAX_ELEMENTS
    ):
        raise ValueError("Scan buffers do not match their entry and storage extent")
    pointer_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    for name, buffer in buffers.items():
        itemsize = ctypes.sizeof(TYPES[types[name]])
        if (
            buffer.dtype != types[name].encode("ascii")
            or buffer.count != (size if name in {"in", "out"} else 1)
            or not buffer.data
            or buffer.data % itemsize
            or buffer.data + buffer.count * itemsize > pointer_limit
            or buffer.output != int(name == "out")
        ):
            raise ValueError("Scan buffer type, extent, alignment or direction differs")
    output = buffers["out"]
    output_end = output.data + size * ctypes.sizeof(TYPES[destination])
    for name, buffer in buffers.items():
        if name == "out":
            continue
        end = buffer.data + buffer.count * ctypes.sizeof(TYPES[types[name]])
        if max(output.data, buffer.data) < min(output_end, end) and not (
            name == "in"
            and source == destination
            and output.data == buffer.data
            and output_end == end
        ):
            raise ValueError("Scan output overlaps metadata or a partial input view")
    axis_size = values(buffers["axis_size"])[0]
    stride = values(buffers["stride"])[0] if layout == "strided" else 1
    expected = execution(entry, size, axis_size, stride)
    if layout == "strided" and values(buffers["stride_blocks"]) != [
        (stride + 31) // 32
    ]:
        raise ValueError("Scan stride block count does not match its source geometry")
    if launch != expected:
        raise ValueError("Scan launch does not match the source geometry")
    return {
        "axisSize": axis_size,
        "stride": stride,
        "strideBlocks": (stride + 31) // 32,
        "storageElements": size,
        "operation": operation,
        "sourceType": source,
        "outputType": destination,
    }
