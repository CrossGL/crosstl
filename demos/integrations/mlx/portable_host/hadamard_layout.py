"""Plan pinned Hadamard specializations and check their storage and launches.

``Hadamard::eval_gpu`` in ``backend.cpp`` mirrors the pinned
``mlx/backend/metal/hadamard.cpp``. The transform of the last axis,
``m * 2^k``, splits into at most three stages, each one JIT entry named as
upstream names it:

- ``n1hadamard_<size>_<dtype>``: a strided power-of-two transform, used only
  when ``2^k`` exceeds 8192;
- ``n2hadamard_<size>_<dtype>``: a row-contiguous power-of-two transform;
- ``mhadamard_<size>_<dtype>``: the O(m^2) codelet for ``m`` in 12, 20, 28.

The entry name fixes every template argument, so ``signature`` derives the
specialization exactly as ``hadamard_mn_contiguous`` does. Each dispatch covers
whole transforms; the host splits larger arrays into launches that stay within
the portable 65,535 workgroups per axis.
"""

import ctypes
import re

# MAX_HADAMARD_THREADS_PER_GROUP in hadamard.cpp.
MAX_THREADS = 256
# Hadamard::eval_gpu splits power-of-two factors above this size.
MAX_ROW = 8192
# decompose_hadamard's largest power-of-two factor, 2^26.
MAX_POWER = 1 << 26
# hadamard_matrices() is a std::map, so decompose_hadamard tries these in order.
FACTORS = (12, 20, 28)
TYPES = {"float32": "float", "float16": "float16_t", "bfloat16": "bfloat16_t"}
ITEM_SIZES = {"float32": 4, "float16": 2, "bfloat16": 2}
STAGES = ("n1", "n2", "m")
ENTRY_PREFIXES = tuple(stage + "hadamard_" for stage in STAGES)
PATTERN = re.compile(r"(n1|n2|m)hadamard_([1-9][0-9]*)_(float32|float16|bfloat16)")
MAX_GROUPS = 65535
# Every bound array holds at most this many elements, so each OpenGL and
# DirectX element offset stays below 2^31 - 1.
MAX_ELEMENTS = 2**28 - 1
PARAMETERS = ("in", "out", "scale")


def _power_of_two(value):
    return value > 0 and not value & (value - 1)


def decompose(size):
    """``decompose_hadamard``: ``(n, m)`` with ``size = m * n`` and ``n = 2^k``.

    Upstream only checks that ``m`` divides ``size``; a remaining factor that is
    not a power of two would make the kernels read outside their transform, so
    it fails here instead.
    """
    if type(size) is not int or size < 2:
        raise ValueError("Hadamard transforms need a last axis of at least two")
    n, m = size, 1
    if not _power_of_two(n):
        for factor in FACTORS:
            if n % factor == 0:
                m, n = factor, n // factor
                break
        if m == 1:
            raise ValueError(
                "[hadamard] Only supports n = m*2^k where m in (1, 12, 20, 28)."
            )
    if n > MAX_POWER:
        raise ValueError("[hadamard] Only supports n = m*2^k where k <= 26")
    if not _power_of_two(n):
        raise ValueError("Hadamard size is not m*2^k for a supported m")
    return n, m


def plan(size):
    """Every stage's template arguments for a last axis of ``size``."""
    n, m = decompose(size)
    n1, n2 = 1, n
    if n > MAX_ROW:
        n2 = 2
        while n2 * n2 < n:
            n2 *= 2
        n1 = n // n2
    read_width_n1 = 2 if n1 == 2 else 4
    read_width_n2 = 2 if n2 == 2 else 4
    # The m stage strides by n / read_width_m, so upstream caps it at n.
    read_width_m = 1 if n == 1 else (2 if n == 2 or m == 28 else 4)
    stages = {}
    if n1 > 1:
        radix = min(n1, 16)
        stages["n1"] = {
            "function": "hadamard_n",
            "arguments": (n1, radix, read_width_n1, n2),
            "workgroupWidth": n1 // radix,
            "transform": n,
        }
    if n2 > 1:
        radix = min(n2, 16)
        stages["n2"] = {
            "function": "hadamard_n",
            "arguments": (n2, radix, read_width_n2),
            "workgroupWidth": n2 // radix,
            "transform": n2,
        }
    if m > 1:
        stages["m"] = {
            "function": "hadamard_m",
            "arguments": (n, m, read_width_m),
            "workgroupWidth": min(n // read_width_m, MAX_THREADS),
            "transform": m * n,
        }
    return {"size": size, "n": n, "m": m, "n1": n1, "n2": n2, "stages": stages}


def entries(size, dtype):
    """The entries ``hadamard_mn_contiguous`` launches, in launch order."""
    if dtype not in TYPES:
        raise ValueError("Hadamard transforms support float32, float16 and bfloat16")
    stages = plan(size)["stages"]
    return [f"{stage}hadamard_{size}_{dtype}" for stage in STAGES if stage in stages]


def signature(entry):
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError("Unknown Hadamard entry")
    stage, size, dtype = match.group(1), int(match.group(2)), match.group(3)
    if str(size) != match.group(2):
        raise ValueError("Unknown Hadamard entry")
    layout = plan(size)
    if stage not in layout["stages"]:
        raise ValueError("Hadamard entry names a stage its size does not launch")
    detail = layout["stages"][stage]
    return {
        "entry": entry,
        "stage": stage,
        "size": size,
        "dtype": dtype,
        "m": layout["m"],
        "n": layout["n"],
        "function": detail["function"],
        "templateArguments": (TYPES[dtype], *detail["arguments"]),
        "workgroupSize": (detail["workgroupWidth"], 1, 1),
        "transform": detail["transform"],
    }


def scale(entry, value):
    """The scale ``hadamard_mn_contiguous`` binds for ``entry``'s stage."""
    info = signature(entry)
    if info["stage"] == "n1" or (info["stage"] == "n2" and info["m"] > 1):
        return 1.0
    return value


def execution(entry, elements):
    """The workgroup count and size of one launch over ``elements``."""
    info = signature(entry)
    size = info["size"]
    if (
        type(elements) is not int
        or not size <= elements <= MAX_ELEMENTS
        or elements % size
    ):
        raise ValueError("Hadamard launch must cover whole transforms")
    width = info["workgroupSize"][0]
    if info["stage"] == "n2":
        count = (1, elements // info["templateArguments"][1], 1)
    elif info["stage"] == "n1":
        count = (1, elements // info["n"], info["templateArguments"][4])
    else:
        read_width = info["templateArguments"][3]
        count = (1, elements // info["m"] // read_width // width, 1)
    if any(not 1 <= value <= MAX_GROUPS for value in count):
        raise ValueError("Hadamard launch exceeds the portable workgroup limits")
    return {"workgroupCount": list(count), "workgroupSize": list(info["workgroupSize"])}


def batch_limit(size):
    """The most whole transforms one launch of every stage can cover."""
    layout = plan(size)
    limits = [MAX_ELEMENTS // size]
    for stage, detail in layout["stages"].items():
        if stage == "n2":
            per_transform = size // detail["arguments"][0]
        elif stage == "n1":
            per_transform = layout["m"]
        else:
            per_transform = layout["n"] // (
                detail["arguments"][2] * detail["workgroupWidth"]
            )
        limits.append(MAX_GROUPS // per_transform)
    return min(limits)


def read_float(buffer):
    return ctypes.cast(buffer.data, ctypes.POINTER(ctypes.c_float)).contents.value


def validate(entry, buffers, elements, launch):
    """Check one native launch's buffers and geometry against ``entry``."""
    info = signature(entry)
    dtype = info["dtype"]
    if set(buffers) != set(PARAMETERS):
        raise ValueError("Hadamard buffers do not match the entry")
    expected = execution(entry, elements)
    if launch != expected:
        raise ValueError("Hadamard launch does not match the pinned geometry")
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    spans = {}
    for name in PARAMETERS:
        buffer = buffers[name]
        kind = "float32" if name == "scale" else dtype
        itemsize = ITEM_SIZES[kind]
        count = 1 if name == "scale" else elements
        if (
            not buffer.data
            or buffer.data % itemsize
            or buffer.dtype != kind.encode("ascii")
            or buffer.count != count
            or buffer.output != int(name == "out")
            or buffer.data + count * itemsize > address_limit
        ):
            raise ValueError(
                "Hadamard buffer type, extent, alignment or direction differs"
            )
        spans[name] = (buffer.data, buffer.data + count * itemsize)
    low, high = spans["out"]
    for name, (start, end) in spans.items():
        if name == "out" or not max(low, start) < min(high, end):
            continue
        # Upstream transforms in place; only an exact alias of the input is valid.
        if name != "in" or (start, end) != (low, high):
            raise ValueError("Hadamard output overlaps metadata or a partial input")
    value = read_float(buffers["scale"])
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("Hadamard scale must be finite")
    if value != scale(entry, value):
        raise ValueError("Only the last Hadamard stage scales its transform")
    return {
        "stage": info["stage"],
        "size": info["size"],
        "m": info["m"],
        "n": info["n"],
        "dtype": dtype,
        "elements": elements,
        "transforms": elements // info["size"],
        "scale": value,
    }
