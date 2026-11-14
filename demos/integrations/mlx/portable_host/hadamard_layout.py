"""Validate MLX Hadamard transform launches before native dispatch.

``Hadamard::eval_gpu`` in the pinned ``mlx/backend/metal/hadamard.cpp``
splits a transform of ``n = m * 2**k`` points (``decompose_hadamard``, with
``m`` in 1, 12, 20 and 28) into at most three kernels that it generates from
``kernels/hadamard.h`` for one size and type:

- ``n1hadamard_<n>_<type>``: ``hadamard_n`` over ``n1`` points at stride
  ``n2``, when the power-of-two part exceeds 8,192 points;
- ``n2hadamard_<n>_<type>``: ``hadamard_n`` over ``n2`` contiguous points;
- ``mhadamard_<n>_<type>``: ``hadamard_m`` with the O(m**2) codelet
  ``gen_hadamard_codelet`` writes from the pinned Hadamard matrix strings.

Every check below is ``hadamard_mn_contiguous``'s plan. The host splits a
launch into slices of whole rows (or whole ``m``-point batches) so that no
axis carries more than 65,535 workgroups; each slice binds its rows from
their first element, which leaves each kernel's indices unchanged.
"""

import ctypes
import re
from types import MappingProxyType

COMMON_HEADER = "mlx/backend/common/hadamard.h"
KERNEL_HEADER = "mlx/backend/metal/kernels/hadamard.h"
UTILS_HEADER = "mlx/backend/metal/kernels/utils.h"
# The pinned files a generated library depends on.
HEADERS = (COMMON_HEADER, KERNEL_HEADER, UTILS_HEADER)
# hadamard_matrices() in std::map order, which decompose_hadamard tries.
FACTORS = (12, 20, 28)
MAX_LOG = 26
MAX_RADIX = 16
# Eval_gpu splits power-of-two parts above this size into two strided passes.
MAX_STAGE = 8192
# MAX_HADAMARD_THREADS_PER_GROUP.
MAX_THREADS = 256
# type_to_name -> get_type_string of the floating types hadamard_transform keeps.
DTYPES = MappingProxyType(
    {"float32": "float", "float16": "float16_t", "bfloat16": "bfloat16_t"}
)
STAGES = ("n1", "n2", "m")
# Entry prefixes whose trace records keep their values for the audit.
PREFIXES = tuple(f"{stage}hadamard_" for stage in STAGES)
INPUT, OUTPUT, SCALE = "in", "out", "scale"
# The float32 storage word of 1.0f.
UNIT_SCALE = 0x3F800000
# The bounds the host's other families use: one launch axis carries at most
# 65535 workgroups and one bound buffer at most 2**28 - 1 elements.
MAX_GROUPS = 65535
MAX_ELEMENTS = 2**28 - 1
_MATRIX = re.compile(r'constexpr std::string_view h(\d+) = R"\((.*?)\)";', re.S)


def _power_of_two(value):
    return value > 0 and value & (value - 1) == 0


def decompose(size):
    """``decompose_hadamard``: ``(n, m)`` with ``size == m * n``.

    MLX does not check that the remaining factor is a power of two; the host
    refuses such sizes rather than generate kernels for them.
    """
    if type(size) is not int or size < 2:
        raise ValueError("Hadamard transforms need at least two points")
    n, m = size, 1
    if not _power_of_two(n):
        for factor in FACTORS:
            if n % factor == 0:
                m, n = factor, n // factor
                break
        else:
            raise ValueError(
                "[hadamard] Only supports n = m*2^k where m in (1, 12, 20, 28)."
            )
    if not _power_of_two(n) or n > 1 << MAX_LOG:
        raise ValueError("CrossTL Hadamard transforms need n = m * 2^k, k <= 26")
    return n, m


def split(n):
    """Eval_gpu's ``(n1, n2)`` with ``n1 * n2 == n`` and both at most 8,192."""
    if n <= MAX_STAGE:
        return 1, n
    n2 = 2
    while n2 * n2 < n:
        n2 *= 2
    return n // n2, n2


def _stage_plan(stage, size, dtype):
    n, m = decompose(size)
    n1, n2 = split(n)
    kind = DTYPES[dtype]
    if stage == "n1" and n1 > 1:
        radix = min(n1, MAX_RADIX)
        width = 2 if n1 == 2 else 4
        return {
            "function": "hadamard_n",
            "arguments": (kind, n1, radix, width, n2),
            "workgroupSize": (n1 // radix, 1, 1),
            # One workgroup per (row, column of the n2 stride).
            "unitElements": n,
            "unitGroups": 1,
            "depth": n2,
            "unitScale": True,
        }
    if stage == "n2" and n2 > 1:
        radix = min(n2, MAX_RADIX)
        width = 2 if n2 == 2 else 4
        return {
            "function": "hadamard_n",
            "arguments": (kind, n2, radix, width),
            "workgroupSize": (n2 // radix, 1, 1),
            "unitElements": n2,
            "unitGroups": 1,
            "depth": 1,
            "unitScale": m > 1,
        }
    if stage == "m" and m > 1:
        # The m stage strides by n / read_width_m, so the width is capped at n.
        width = 1 if n == 1 else 2 if n == 2 or m == 28 else 4
        threads = min(n // width, MAX_THREADS)
        return {
            "function": "hadamard_m",
            "arguments": (kind, n, m, width),
            "workgroupSize": (threads, 1, 1),
            # Thread (x, y) of a launch of G rows takes index x * G + y, so a
            # slice holds whole m * n batches of n / width / threads rows.
            "unitElements": m * n,
            "unitGroups": n // width // threads,
            "depth": 1,
            "unitScale": False,
        }
    raise ValueError("Hadamard size does not launch this stage")


def entry_name(stage, size, dtype):
    """The pipeline name ``hadamard_mn_contiguous`` gives one stage."""
    entry = f"{stage}hadamard_{size}_{dtype}"
    signature(entry)
    return entry


def stages(size, dtype):
    """The entries eval_gpu launches for ``size`` points, in launch order."""
    n, m = decompose(size)
    n1, n2 = split(n)
    present = {"n1": n1 > 1, "n2": n2 > 1, "m": m > 1}
    return tuple(entry_name(stage, size, dtype) for stage in STAGES if present[stage])


def _entries():
    entries = {}
    for m in (1, *FACTORS):
        for k in range(MAX_LOG + 1):
            size = m * 2**k
            if size < 2:
                continue
            for dtype in DTYPES:
                n = size // m
                n1, n2 = split(n)
                for stage, present in (("n1", n1 > 1), ("n2", n2 > 1), ("m", m > 1)):
                    if present:
                        entries[f"{stage}hadamard_{size}_{dtype}"] = (
                            stage,
                            size,
                            dtype,
                        )
    return MappingProxyType(entries)


# Host entry -> (stage, transform size, element type).
ENTRIES = _entries()


def signature(entry):
    """``(stage, size, dtype)`` of a Hadamard entry."""
    try:
        return ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unsupported Hadamard entry") from error


def stage_plan(entry):
    """The template instantiation, threadgroup and slice unit of ``entry``."""
    return _stage_plan(*signature(entry))


def workgroup_size(entry):
    return stage_plan(entry)["workgroupSize"]


def operand_types(entry):
    """Every source parameter's element type, in source order."""
    dtype = signature(entry)[2]
    return {INPUT: dtype, OUTPUT: dtype, SCALE: "float32"}


def matrices(text):
    """The pinned ``hadamard_matrices()`` strings of ``common/hadamard.h``.

    Each is the raw string between its leading and trailing newlines: one
    row of ``+`` and ``-`` per line, ``m`` rows of ``m`` signs.
    """
    found = {int(m): body for m, body in _MATRIX.findall(text)}
    if set(found) != set(FACTORS):
        raise ValueError("The pinned Hadamard matrices are missing")
    for m, body in found.items():
        rows = body[1:].split("\n")
        if (
            not body.startswith("\n")
            or rows[-1] != ""
            or len(rows) != m + 1
            or any(len(row) != m or set(row) - {"+", "-"} for row in rows[:-1])
        ):
            raise ValueError(f"The pinned {m}-point Hadamard matrix is malformed")
    return found


def codelet(m, text):
    """``gen_hadamard_codelet(m)`` from the pinned ``common/hadamard.h`` text."""
    lines = ["METAL_FUNC void hadamard_radix_m(thread float *x) {"]
    if m == 1:
        return "\n".join([*lines, "}", ""])
    lines.append(f"  float tmp[{m}];")
    rows = matrices(text)[m][1:].split("\n")[:-1]
    for index, row in enumerate(rows):
        terms = "".join(f" {sign} x[{i}]" for i, sign in enumerate(row))
        lines.append(f"  tmp[{index}] = {terms};")
    lines.append(f"  for (int i = 0; i < {m}; i++) {{ x[i] = tmp[i]; }}")
    return "\n".join([*lines, "}", ""])


def render(entry, text):
    """The library ``hadamard_mn_contiguous`` builds, with ``entry``'s kernel.

    MLX concatenates ``metal::utils()``, the codelet and ``metal::hadamard()``,
    the embedded pinned headers, then ``get_template_definition`` for each
    stage. The host includes the same pinned headers and instantiates only
    ``entry``'s stage, so each stage is a package of its own.
    """
    stage, size, _ = signature(entry)
    _, m = decompose(size)
    plan = stage_plan(entry)
    function = f"{plan['function']}<{', '.join(map(str, plan['arguments']))}>"
    return (
        f'#include "{UTILS_HEADER}"\n'
        + codelet(m, text)
        + f'#include "{KERNEL_HEADER}"\n'
        + f'\ntemplate [[host_name("{entry}")]] [[kernel]] '
        f"decltype({function}) {function};\n"
    )


def plan(entry, count, scale):
    """The launch of a slice of ``count`` elements; refuses any other.

    ``scale`` is the float32 storage word the stage binds:
    hadamard_mn_contiguous passes 1 to the strided ``n1`` pass and to the
    ``n2`` pass ahead of an ``m`` pass.
    """
    info = stage_plan(entry)
    unit = info["unitElements"]
    if type(count) is not int or not 1 <= count <= MAX_ELEMENTS or count % unit:
        raise ValueError("Hadamard slice does not cover whole transforms")
    rows = count // unit * info["unitGroups"]
    if rows > MAX_GROUPS:
        raise ValueError("Hadamard launch exceeds its group limit")
    if type(scale) is not int or not 0 <= scale <= 0xFFFFFFFF:
        raise ValueError("Hadamard scale is not a float32 word")
    if info["unitScale"] and scale != UNIT_SCALE:
        raise ValueError("This Hadamard pass takes a unit scale")
    return {
        "workgroupCount": [1, rows, info["depth"]],
        "workgroupSize": list(info["workgroupSize"]),
    }


def slices(entry, total):
    """The ``(first element, count)`` slices hadamard.inc launches.

    ``dispatch_hadamard_stage`` binds as many whole units per launch as keep
    its rows within ``MAX_GROUPS`` and its buffers within ``MAX_ELEMENTS``.
    """
    info = stage_plan(entry)
    unit = info["unitElements"]
    if type(total) is not int or total < 1 or total % unit:
        raise ValueError("Hadamard stage does not cover whole transforms")
    per_launch = min(MAX_GROUPS // info["unitGroups"], MAX_ELEMENTS // unit)
    if per_launch == 0:
        raise ValueError("Hadamard transforms exceed the portable group limit")
    units = total // unit
    return [
        (first * unit, min(per_launch, units - first) * unit)
        for first in range(0, units, per_launch)
    ]


def threads(execution):
    """The thread count of ``execution``'s grid, as the host passes it."""
    total = 1
    for count, size in zip(execution["workgroupCount"], execution["workgroupSize"]):
        total *= count * size
    return total


def validate(entry, buffers, threads_, execution):
    """Check one slice against eval_gpu; returns ``(count, scale word)``."""
    types = operand_types(entry)
    if set(buffers) != set(types):
        raise ValueError("Hadamard buffers do not match their kernel")
    for name, kind in types.items():
        buffer = buffers[name]
        maximum = 1 if name == SCALE else MAX_ELEMENTS
        if (
            buffer.dtype != kind.encode("ascii")
            or not buffer.data
            or not 1 <= buffer.count <= maximum
            or buffer.output != int(name == OUTPUT)
        ):
            raise ValueError("Hadamard buffer type, extent or direction differs")
    count = buffers[INPUT].count
    if buffers[OUTPUT].count != count:
        raise ValueError("Hadamard input and output extents differ")
    scale = ctypes.cast(buffers[SCALE].data, ctypes.POINTER(ctypes.c_uint32))
    scale = scale.contents.value
    expected = plan(entry, count, scale)
    if expected != execution or threads(expected) != threads_:
        raise ValueError("Native Hadamard launch does not match the upstream plan")
    return count, scale
