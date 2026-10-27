"""Plan, name and bound MLX's single-threadgroup Stockham, Rader and Bluestein FFT launches.

``mlx/backend/metal/fft.cpp`` plans every 1-D transform, sets the function
constants of ``fft.metal``'s kernels from the plan and names each pipeline
``<kernel>_n<size>_inv_<inverse>``. The host names each launch by that hash
name; its translated entry has the function constants the hash name implies
frozen into it. This module mirrors fft.cpp's planning exactly (radix
preference, Rader and Bluestein selection, the four-step split and the
elements-per-thread heuristics), parses hash names back into their constants,
threadgroups and buffer extents, and checks every launch against them.

The ``fft`` kernel computes ``threadgroup_batch_size`` transforms of one size
in one threadgroup of ``(1, threadgroup_batch_size, threads_per_fft)``
threads: complex-to-complex, real-to-complex (two real rows per transform)
and complex-to-real. The ``rader_fft`` kernel runs plans with one Rader
factor in the same threadgroups, with the tables fft.cpp's
compute_raders_constants builds on the CPU: the FFT of the Rader kernel
``raders_b_q`` and the generator powers ``raders_g_q`` and
``raders_g_minus_q``. The ``bluestein_fft`` kernel runs plans of any other
size up to 2048 as a ``bluestein_n``-point Stockham convolution in the same
threadgroups, with the chirp ``w_k`` and its spectrum ``w_q``
compute_bluestein_constants builds on the CPU; its hash name carries
``bluestein_n``, and the transform length binds as ``length``. Four-step
plans use other kernels.
"""

import cmath
import functools
import math
import re

SOURCE = "mlx/backend/metal/kernels/fft.metal"
# fft.cpp supported_radices(), in decomposition preference.
RADICES = (13, 11, 8, 7, 6, 5, 4, 3, 2)
MAX_STOCKHAM_FFT_SIZE = 4096
MAX_RADER_FFT_SIZE = 2048
MAX_BLUESTEIN_FFT_SIZE = 2048
MIN_THREADGROUP_MEM_SIZE = 256
MIN_COALESCE_WIDTH = 4
# fft.metal instantiate_ffts: the threadgroup memory sizes it instantiates.
MEMORY_SIZES = (256, 512, 1024, 2048, 4096)
# The function constants fft.h declares, by fft.cpp's indices: the Stockham
# radix step counts from 4 and the Rader step counts from 13.
STOCKHAM_CONSTANTS = tuple(f"radix_{radix}_steps_" for radix in RADICES)
RADER_CONSTANTS = tuple(f"rader_{radix}_steps_" for radix in RADICES)
CONSTANT_IDS = {
    "inv_": 0,
    "is_power_of_2_": 1,
    "elems_per_thread_": 2,
    "rader_m_": 3,
    **{name: 4 + index for index, name in enumerate(STOCKHAM_CONSTANTS)},
    **{name: 13 + index for index, name in enumerate(RADER_CONSTANTS)},
}
# The function constant fft.cpp sets for complex Bluestein plans only.
TWIDDLE_TABLE_CONSTANT = "use_bluestein_twiddle_table_"
# The storage each fft.metal type name binds: float2 holds one complex64.
STORAGE = {"float": "float32", "float2": "complex64"}
ITEM_SIZES = {"float32": 4, "complex64": 8, "int16": 2}
# The (input, output) instantiations fft.cpp selects, by transform.
TRANSFORMS = {
    ("float2", "float2"): "c2c",
    ("float", "float2"): "r2c",
    ("float2", "float"): "c2r",
}
# The plan kind each fft.metal kernel runs, by kernel name.
KERNELS = {"fft": "stockham", "rader_fft": "rader", "bluestein_fft": "bluestein"}
ENTRY_PREFIXES = ("fft_mem_", "rader_fft_mem_", "bluestein_fft_mem_")
ENTRY = re.compile(
    r"(?P<source>(?P<kernel>bluestein_fft|rader_fft|fft)_mem_(?P<memory>[0-9]+)_"
    r"(?P<input>float2|float)_(?P<output>float2|float))_n(?P<n>[1-9][0-9]*)"
    r"_inv_(?P<inverse>[01])"
)
# The parameters fft.cpp binds to each kernel, in buffer order.
PARAMETERS = {
    "stockham": ("in", "out", "n", "batch_size"),
    "rader": (
        "in",
        "out",
        "raders_b_q",
        "raders_g_q",
        "raders_g_minus_q",
        "n",
        "batch_size",
        "rader_n",
    ),
    "bluestein": ("in", "out", "w_q", "w_k", "length", "n", "batch_size"),
}
OUTPUT = "out"
SCALARS = {
    "stockham": ("n", "batch_size"),
    "rader": ("n", "batch_size", "rader_n"),
    "bluestein": ("length", "n", "batch_size"),
}
# The tables fft.cpp computes on the CPU for each kernel, and their storage.
TABLES = {
    "stockham": {},
    "rader": {
        "raders_b_q": "complex64",
        "raders_g_q": "int16",
        "raders_g_minus_q": "int16",
    },
    "bluestein": {"w_q": "complex64", "w_k": "complex64"},
}
# Every device index the host binds stays a signed 32-bit element offset.
SIGNED_MAX = (1 << 31) - 1
MAX_WORKGROUPS = 65535
# The device subscripts readwrite.h forms from a transform's batch offset.
DEVICE_INDEX_EXPRESSIONS = (
    "batch_idx + index",
    "batch_idx + index + r",
    "batch_idx + index + next_in",
    "batch_idx + index + next_out",
)
# The subscript the translated reference helpers form, scoped to the storage
# it indexes: the threadgroup buffer or the bound input and output buffers.
REFERENCE_INDEX = "reference.offset + index"
SHARED_STORAGE = "crosstl_workgroup_shared_in"
# The translated names of the bound device buffers, in buffer order.
DEVICE_STORAGES = ("crosstl_resource", "crosstl_resource_2")
TABLE_STORAGES = ("crosstl_resource_3", "crosstl_resource_4", "crosstl_resource_5")


def is_power_of_2(n):
    """mlx/utils.h is_power_of_2."""
    return n != 0 and n & (n - 1) == 0


def next_power_of_2(n):
    """mlx/utils.h next_power_of_2 for the positive sizes fft.cpp passes."""
    if n < 1:
        raise ValueError("FFT sizes must be positive")
    return n if is_power_of_2(n) else 1 << (n - 1).bit_length()


def prime_factors(n):
    """fft.cpp prime_factors: the prime factors of ``n`` with multiplicity."""
    factors, z = [], 2
    while z * z <= n:
        if n % z == 0:
            factors.append(z)
            n //= z
        else:
            z += 1
    if n > 1:
        factors.append(n)
    return factors


def plan_stockham_fft(n):
    """fft.cpp plan_stockham_fft: the step count of each radix of ``n``."""
    plan = [0] * len(RADICES)
    original = n
    if n == 1:
        return plan
    for index, radix in enumerate(RADICES):
        # Manually tuned radices for powers of 2.
        if is_power_of_2(original) and original < 512 and radix > 4:
            continue
        while n % radix == 0:
            plan[index] += 1
            n //= radix
            if n == 1:
                return plan
    raise ValueError("Unplannable")


def plan_fft(n):
    """fft.cpp plan_fft, as a dict of the FFTPlan fields."""
    radices = set(RADICES)
    plan = {
        "n": n,
        "stockham": [],
        "rader": [0] * len(RADICES),
        "rader_n": 1,
        "bluestein_n": -1,
        "four_step": False,
        "n1": 0,
        "n2": 0,
    }

    def bluestein():
        plan["four_step"] = n > MAX_BLUESTEIN_FFT_SIZE
        plan["bluestein_n"] = next_power_of_2(2 * n - 1)
        plan["stockham"] = plan_stockham_fft(plan["bluestein_n"])
        plan["rader"] = [0] * len(RADICES)
        return plan

    if n > MAX_STOCKHAM_FFT_SIZE and is_power_of_2(n):
        plan["four_step"] = True
        plan["n2"] = 1024 if n > 65536 else 64
        plan["n1"] = n // plan["n2"]
        if plan["n1"] > MAX_STOCKHAM_FFT_SIZE:
            plan["n1"] = MAX_STOCKHAM_FFT_SIZE
            plan["n2"] = n // plan["n1"]
        if plan["n2"] > MAX_STOCKHAM_FFT_SIZE:
            raise ValueError(f"[FFT] GPU FFT is not supported for size {n}")
        return plan
    if n > MAX_STOCKHAM_FFT_SIZE:
        plan["four_step"] = True
        plan["bluestein_n"] = next_power_of_2(2 * n - 1)
        return plan
    remaining = n
    for factor in prime_factors(n):
        if factor in radices:
            continue
        if plan["rader_n"] > 1 or n > MAX_RADER_FFT_SIZE:
            return bluestein()
        if any(rf not in radices for rf in prime_factors(factor - 1)):
            return bluestein()
        plan["rader"] = plan_stockham_fft(factor - 1)
        plan["rader_n"] = factor
        remaining //= factor
    plan["stockham"] = plan_stockham_fft(remaining)
    return plan


def mod_exp(x, y, n):
    """fft.cpp mod_exp: ``x ** y % n`` by repeated squaring in int."""
    out = 1
    while y:
        if y & 1:
            out = out * x % n
        y >>= 1
        x = x * x % n
    return out


def primitive_root(n):
    """fft.cpp primitive_root: the least primitive root of the prime ``n``."""
    factors = prime_factors(n - 1)
    for root in range(2, n - 1):
        if all(mod_exp(root, (n - 1) // factor, n) != 1 for factor in factors):
            return root
    return -1


def rader_generator_tables(rader_n):
    """compute_raders_constants' ``g_q`` and ``g_minus_q``: the powers of the
    primitive root and of its inverse modulo ``rader_n``, as fft.cpp's shorts."""
    root = primitive_root(rader_n)
    # Fermat's little theorem.
    inverse = mod_exp(root, rader_n - 2, rader_n)
    return (
        [mod_exp(root, index, rader_n) for index in range(rader_n - 1)],
        [mod_exp(inverse, index, rader_n) for index in range(rader_n - 1)],
    )


@functools.lru_cache(maxsize=16)
def rader_kernel_spectrum(rader_n):
    """compute_raders_constants' ``b_q`` in double precision: the forward DFT
    of ``exp(-2 pi i g_minus_q[i] / rader_n)``, which fft.cpp computes with
    pocketfft in float32."""
    _, g_minus_q = rader_generator_tables(rader_n)
    size = rader_n - 1
    kernel = [cmath.exp(complex(0, g * -2.0 * math.pi / rader_n)) for g in g_minus_q]
    roots = [
        cmath.exp(complex(0, -2.0 * math.pi * index / size)) for index in range(size)
    ]
    return tuple(
        sum(value * roots[index * k % size] for index, value in enumerate(kernel))
        for k in range(size)
    )


def _fft(values):
    """The forward DFT of a power-of-two sequence in double precision."""
    size = len(values)
    if size == 1:
        return list(values)
    even, odd = _fft(values[0::2]), _fft(values[1::2])
    out = [0j] * size
    for k in range(size // 2):
        twiddled = cmath.exp(complex(0, -2.0 * math.pi * k / size)) * odd[k]
        out[k], out[k + size // 2] = even[k] + twiddled, even[k] - twiddled
    return out


@functools.lru_cache(maxsize=16)
def bluestein_tables(length, bluestein_n):
    """compute_bluestein_constants' ``w_k`` and ``w_q`` in double precision:
    the chirp ``exp(-i pi k^2 / length)`` and the forward DFT of its inverse
    over ``1 - length .. length - 1``, zero padded to ``bluestein_n``, which
    fft.cpp computes with pocketfft in float32."""
    chirp = [cmath.exp(complex(0, -(k**2) * math.pi / length)) for k in range(length)]
    padded = [0j] * bluestein_n
    for k in range(1 - length, length):
        padded[k + length - 1] = cmath.exp(complex(0, k**2 * math.pi / length))
    return tuple(chirp), tuple(_fft(padded))


@functools.lru_cache(maxsize=None)
def bluestein_lengths(bluestein_n):
    """The transform lengths fft.cpp runs as a ``bluestein_n``-point
    Bluestein convolution in one threadgroup."""
    lengths = []
    for length in range(bluestein_n // 4 + 1, bluestein_n // 2 + 1):
        plan = plan_fft(length)
        if plan_kind(plan) == "bluestein" and plan["bluestein_n"] == bluestein_n:
            lengths.append(length)
    return tuple(lengths)


def compute_elems_per_thread(plan):
    """fft.cpp compute_elems_per_thread's heuristics."""
    n = plan["n"]
    steps = [*plan["stockham"], *plan["rader"]]
    used = sorted(
        {RADICES[i % len(RADICES)] for i, step in enumerate(steps) if step > 0}
    )
    if 7 in used and (11 in used or 13 in used):
        return 7
    if 11 in used and 13 in used:
        return 11
    tuned = {3159: 13, 3645: 5, 3969: 7, 1982: 5}
    if n in tuned:
        return tuned[n]
    if len(used) == 1:
        return used[0]
    if len(used) == 2 and (11 in used or 13 in used):
        return sum(used) // 2
    return used[1]


def plan_kind(plan):
    """The kernel kind of a plan: ``stockham``, ``rader``, ``bluestein`` or
    ``four_step``, in fft.cpp's order of precedence."""
    if plan["four_step"]:
        return "four_step"
    if plan["bluestein_n"] > 0:
        return "bluestein"
    if plan["rader_n"] > 1:
        return "rader"
    return "stockham"


def plan_constants(n, inverse, kind, transform="c2c"):
    """The function constants fft.cpp sets for a ``kind`` plan of ``n``.

    For Bluestein plans ``n`` is the hash name's ``bluestein_n``: fft.cpp
    plans every length it convolves at that size with the same Stockham
    steps and elements per thread. It also sets ``rader_m_`` to the length,
    which no Bluestein kernel reads, and caches the pipeline by hash name,
    so the first length compiled serves every other; the entry freezes the
    smallest length. ``use_bluestein_twiddle_table_`` is set for
    complex-to-complex plans only, false outside twiddle-table plans.
    """
    if kind == "bluestein":
        lengths = bluestein_lengths(n) if is_power_of_2(n) else ()
        if not lengths:
            raise ValueError(f"FFT size {n} is not a Bluestein convolution size")
        plan = plan_fft(lengths[0])
    else:
        plan = plan_fft(n)
    if plan_kind(plan) != kind or kind not in PARAMETERS:
        raise ValueError(f"FFT size does not have a single-threadgroup {kind} plan")
    constants = {
        "inv_": bool(inverse),
        "is_power_of_2_": is_power_of_2(n),
        "elems_per_thread_": compute_elems_per_thread(plan),
        "rader_m_": plan["n"] // plan["rader_n"],
    }
    constants.update(zip(STOCKHAM_CONSTANTS, plan["stockham"]))
    constants.update(zip(RADER_CONSTANTS, plan["rader"]))
    if kind == "bluestein" and transform == "c2c":
        constants[TWIDDLE_TABLE_CONSTANT] = False
    return constants


def stockham_constants(n, inverse):
    """The function constants fft.cpp sets for a Stockham-only plan of ``n``."""
    return plan_constants(n, inverse, "stockham")


def threadgroup_batch_size(n):
    """fft.cpp's transforms per threadgroup outside four-step plans."""
    return max(MIN_THREADGROUP_MEM_SIZE // n, 1)


def entry_name(n, inverse, input_type, output_type):
    """fft.cpp's pipeline hash name for one single-threadgroup transform of
    length ``n``; a Bluestein plan's name carries its ``bluestein_n``."""
    plan = plan_fft(n)
    kind = plan_kind(plan)
    kernel = {"stockham": "fft", "rader": "rader_fft", "bluestein": "bluestein_fft"}
    if kind not in kernel:
        raise ValueError(
            f"FFT size {n} has no single-threadgroup Stockham, Rader or Bluestein plan"
        )
    size = plan["bluestein_n"] if kind == "bluestein" else n
    memory = next_power_of_2(threadgroup_batch_size(size) * size)
    return (
        f"{kernel[kind]}_mem_{memory}_{input_type}_{output_type}_n{size}"
        f"_inv_{int(bool(inverse))}"
    )


def signature(entry):
    """The launch a ``fft`` or ``rader_fft`` hash name implies, refusing every
    other name."""
    match = ENTRY.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError(f"Not an MLX single-threadgroup FFT entry: {entry!r}")
    n, inverse = int(match["n"]), match["inverse"] == "1"
    transform = TRANSFORMS.get((match["input"], match["output"]))
    kind = KERNELS[match["kernel"]]
    if (
        transform is None
        or (transform == "r2c" and inverse)
        or (transform == "c2r" and not inverse)
        or not 2 <= n <= MAX_STOCKHAM_FFT_SIZE
    ):
        raise ValueError(f"Not an MLX single-threadgroup FFT entry: {entry!r}")
    constants = plan_constants(n, inverse, kind, transform)
    batch = threadgroup_batch_size(n)
    memory = next_power_of_2(batch * n)
    lengths = bluestein_lengths(n) if kind == "bluestein" else (n,)
    if (
        memory not in MEMORY_SIZES
        or int(match["memory"]) != memory
        or entry_name(lengths[0], inverse, match["input"], match["output"]) != entry
    ):
        raise ValueError(f"FFT entry does not name fft.cpp's pipeline: {entry!r}")
    elems = constants["elems_per_thread_"]
    threads = (n + elems - 1) // elems
    plan = plan_fft(lengths[0])
    return {
        "kind": kind,
        "source": match["source"],
        "transform": transform,
        "n": n,
        "inverse": inverse,
        "raderN": plan["rader_n"],
        "lengths": lengths,
        "input": STORAGE[match["input"]],
        "output": STORAGE[match["output"]],
        "memory": memory,
        "threadgroupBatchSize": batch,
        "threadsPerFFT": threads,
        "constants": constants,
        "workgroupSize": (1, batch, threads),
    }


def source_entry(entry):
    return signature(entry)["source"]


def workgroup_size(entry):
    return signature(entry)["workgroupSize"]


def parameters(entry):
    """The parameters fft.cpp binds to the entry's kernel, in buffer order."""
    return PARAMETERS[signature(entry)["kind"]]


def scalars(entry):
    return SCALARS[signature(entry)["kind"]]


def parameter_type(entry, name):
    info = signature(entry)
    if name not in PARAMETERS[info["kind"]]:
        raise KeyError(name)
    if name in TABLES[info["kind"]]:
        return TABLES[info["kind"]][name]
    return {"in": info["input"], "out": info["output"]}.get(name, "int32")


def extents(info, total, length=None):
    """The elements of ``in`` and ``out`` that ``total`` transforms span, and
    of every table the kernel reads. Bluestein transforms span ``length``
    elements each; their ``w_q`` holds ``bluestein_n`` elements and ``w_k``
    one per element of a transform."""
    bluestein = info["kind"] == "bluestein"
    if bluestein and length not in info["lengths"]:
        raise ValueError("Bluestein FFT length differs from its pipeline")
    n = length if bluestein else info["n"]
    half = n // 2 + 1
    spans = {
        "c2c": {"in": total * n, "out": total * n},
        "r2c": {"in": total * n, "out": total * half},
        "c2r": {"in": total * half, "out": total * n},
    }[info["transform"]]
    if bluestein:
        spans.update(w_q=info["n"], w_k=n)
    for name in TABLES[info["kind"]]:
        spans.setdefault(name, info["raderN"] - 1)
    return spans


def _cooperative_io_end(n, rows, threads, elems, batch_rows, width):
    """One past the last element readwrite.h's generic ReadWriter::load and
    write touch from a threadgroup's first transform.

    The ``rows * threads`` in-bounds threads copy the threadgroup's
    transforms cooperatively, ``width`` elements per access, clamped to the
    ``batch_rows`` transforms the threadgroup holds. When ``threads *
    elems`` exceeds ``n``, a partial last threadgroup's copy runs past its
    last transform into elements no transform owns.
    """
    per_group = rows * threads
    full = elems // width
    last_full = batch_rows * n - width
    last = batch_rows * n - 1
    end = 0
    for thread in range(per_group):
        for e in range(full):
            index = min(width * thread + width * per_group * e, last_full)
            end = max(end, index + width)
        for r in range(elems % width):
            index = min(thread + r * per_group + width * per_group * full, last)
            end = max(end, index + 1)
    return end


def footprint(info, total, length=None):
    """The elements of ``in`` and ``out`` the kernel touches for ``total``
    transforms: their extents, and for complex-to-complex transforms the
    tail the generic ReadWriter's cooperative copy reads and writes past a
    partial last threadgroup's transforms."""
    spans = extents(info, total, length)
    footprint = {"in": spans["in"], "out": spans["out"]}
    # Bluestein's padded load and write touch only each transform's elements.
    if info["transform"] != "c2c" or info["kind"] == "bluestein":
        return footprint
    n, batch = info["n"], info["threadgroupBatchSize"]
    groups = (total + batch - 1) // batch
    rows = total - (groups - 1) * batch
    end = _cooperative_io_end(
        n,
        rows,
        info["threadsPerFFT"],
        info["constants"]["elems_per_thread_"],
        batch,
        # 16 bytes per sequential access: two float2 elements.
        16 // ITEM_SIZES["complex64"],
    )
    reach = (groups - 1) * batch * n + max(end, rows * n)
    return {"in": reach, "out": reach}


def launch(info, total):
    """fft.cpp's dispatch_threads grid as whole threadgroups."""
    groups = (total + info["threadgroupBatchSize"] - 1) // info["threadgroupBatchSize"]
    if info["transform"] in {"r2c", "c2r"}:
        # Two real transforms share one complex transform, either way.
        groups = (groups + 1) // 2
    return {
        "workgroupCount": [groups, 1, 1],
        "workgroupSize": list(info["workgroupSize"]),
    }


def plan_launch(entry, counts, values, threads, execution):
    """Check one launch against its hash name and return its metadata.

    ``values`` holds the scalar each of the entry's scalar parameters binds.
    """
    info = signature(entry)
    total = values.get("batch_size")
    length = values.get("length")
    expected_values = {"n": info["n"], "batch_size": total}
    if info["kind"] == "rader":
        expected_values["rader_n"] = info["raderN"]
    if info["kind"] == "bluestein":
        if type(length) is not int or length not in info["lengths"]:
            raise ValueError("Bluestein FFT length differs from its pipeline")
        expected_values["length"] = length
    if (
        set(values) != set(SCALARS[info["kind"]])
        or values != expected_values
        or type(total) is not int
        or total < 1
    ):
        raise ValueError("FFT launch size or batch differs from its pipeline")
    expected = launch(info, total)
    spans = extents(info, total, length)
    if (
        execution != expected
        or not 0 < expected["workgroupCount"][0] <= MAX_WORKGROUPS
        or counts != spans
        or max(spans.values()) > SIGNED_MAX
        or threads != spans["out"]
    ):
        raise ValueError("FFT launch differs from fft.cpp's dispatch")
    return {
        "kind": info["kind"],
        "transform": info["transform"],
        "footprint": footprint(info, total, length),
        "n": info["n"],
        **({"length": length} if info["kind"] == "bluestein" else {}),
        "batchSize": total,
        "inverse": info["inverse"],
        "elements": spans,
    }


def opengl_index_assertions(entry):
    """The subscript ranges OpenGL needs proven, from the host's bounds.

    Device subscripts stay below the signed 32-bit extents ``plan_launch``
    admits; threadgroup subscripts stay inside the entry's ``shared_in``.
    """
    info = signature(entry)
    memory = info["memory"]
    storages = DEVICE_STORAGES + TABLE_STORAGES[: len(TABLES[info["kind"]])]

    def assertion(expression, maximum, indexed=None):
        record = {"source": SOURCE, "expression": expression, "minimum": 0}
        record["maximum"] = maximum
        return {**record, "indexed": indexed} if indexed else record

    return [
        *(assertion(expression, SIGNED_MAX) for expression in DEVICE_INDEX_EXPRESSIONS),
        assertion(REFERENCE_INDEX, memory - 1, SHARED_STORAGE),
        *(
            assertion(expression, SIGNED_MAX, indexed)
            for storage in storages
            # The relative subscript and, through a rebased buffer, the
            # complete offset.
            for expression, indexed in (
                (REFERENCE_INDEX, storage),
                (f"{storage} + {REFERENCE_INDEX}", None),
            )
        ),
    ]


def workgroup_access_assertions(entry):
    """Every threadgroup access of the entry stays inside ``shared_in``."""
    info = signature(entry)
    return [
        {
            "source": SOURCE,
            "entry_point": info["source"],
            "function": "*",
            "parameter": "*",
            "minimum": 0,
            "maximum": info["memory"] - 1,
        }
    ]
