"""Plan, name and bound MLX's single-threadgroup Stockham FFT launches.

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
and complex-to-real. Rader, Bluestein and four-step plans use other kernels.
"""

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
# The storage each fft.metal type name binds: float2 holds one complex64.
STORAGE = {"float": "float32", "float2": "complex64"}
ITEM_SIZES = {"float32": 4, "complex64": 8}
# The (input, output) instantiations fft.cpp selects, by transform.
TRANSFORMS = {
    ("float2", "float2"): "c2c",
    ("float", "float2"): "r2c",
    ("float2", "float"): "c2r",
}
ENTRY_PREFIXES = ("fft_mem_",)
ENTRY = re.compile(
    r"(?P<source>fft_mem_(?P<memory>[0-9]+)_(?P<input>float2|float)_"
    r"(?P<output>float2|float))_n(?P<n>[1-9][0-9]*)_inv_(?P<inverse>[01])"
)
# The parameters fft.cpp binds to the fft kernel, in buffer order.
PARAMETERS = ("in", "out", "n", "batch_size")
OUTPUT = "out"
SCALARS = ("n", "batch_size")
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
DEVICE_STORAGES = ("crosstl_resource", "crosstl_resource_2")


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


def stockham_constants(n, inverse):
    """The function constants fft.cpp sets for a Stockham-only plan of ``n``."""
    plan = plan_fft(n)
    if plan["four_step"] or plan["rader_n"] > 1 or plan["bluestein_n"] > 0:
        raise ValueError("FFT size does not have a single-threadgroup Stockham plan")
    constants = {
        "inv_": bool(inverse),
        "is_power_of_2_": is_power_of_2(n),
        "elems_per_thread_": compute_elems_per_thread(plan),
        "rader_m_": n // plan["rader_n"],
    }
    constants.update(zip(STOCKHAM_CONSTANTS, plan["stockham"]))
    constants.update(zip(RADER_CONSTANTS, plan["rader"]))
    return constants


def threadgroup_batch_size(n):
    """fft.cpp's transforms per threadgroup outside four-step plans."""
    return max(MIN_THREADGROUP_MEM_SIZE // n, 1)


def entry_name(n, inverse, input_type, output_type):
    """fft.cpp's pipeline hash name for one Stockham transform."""
    memory = next_power_of_2(threadgroup_batch_size(n) * n)
    return f"fft_mem_{memory}_{input_type}_{output_type}_n{n}_inv_{int(bool(inverse))}"


def signature(entry):
    """The launch an ``fft`` hash name implies, refusing every other name."""
    match = ENTRY.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError(f"Not an MLX Stockham FFT entry: {entry!r}")
    n, inverse = int(match["n"]), match["inverse"] == "1"
    transform = TRANSFORMS.get((match["input"], match["output"]))
    if (
        transform is None
        or (transform == "r2c" and inverse)
        or (transform == "c2r" and not inverse)
        or not 2 <= n <= MAX_STOCKHAM_FFT_SIZE
    ):
        raise ValueError(f"Not an MLX Stockham FFT entry: {entry!r}")
    constants = stockham_constants(n, inverse)
    batch = threadgroup_batch_size(n)
    memory = next_power_of_2(batch * n)
    if (
        memory not in MEMORY_SIZES
        or int(match["memory"]) != memory
        or entry_name(n, inverse, match["input"], match["output"]) != entry
    ):
        raise ValueError(f"FFT entry does not name fft.cpp's pipeline: {entry!r}")
    elems = constants["elems_per_thread_"]
    threads = (n + elems - 1) // elems
    return {
        "kind": "fft",
        "source": match["source"],
        "transform": transform,
        "n": n,
        "inverse": inverse,
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


def parameter_type(entry, name):
    info = signature(entry)
    return {
        "in": info["input"],
        "out": info["output"],
        "n": "int32",
        "batch_size": "int32",
    }[name]


def extents(info, total):
    """The elements of ``in`` and ``out`` that ``total`` transforms span."""
    n, half = info["n"], info["n"] // 2 + 1
    return {
        "c2c": {"in": total * n, "out": total * n},
        "r2c": {"in": total * n, "out": total * half},
        "c2r": {"in": total * half, "out": total * n},
    }[info["transform"]]


def launch(info, total):
    """fft.cpp's dispatch_threads grid as whole threadgroups."""
    groups = (total + info["threadgroupBatchSize"] - 1) // info["threadgroupBatchSize"]
    if info["transform"] == "r2c":
        # Two real transforms share one complex transform.
        groups = (groups + 1) // 2
    return {
        "workgroupCount": [groups, 1, 1],
        "workgroupSize": list(info["workgroupSize"]),
    }


def plan_launch(entry, counts, n, total, threads, execution):
    """Check one launch against its hash name and return its metadata."""
    info = signature(entry)
    if n != info["n"] or type(total) is not int or total < 1:
        raise ValueError("FFT launch size or batch differs from its pipeline")
    expected = launch(info, total)
    spans = extents(info, total)
    if (
        execution != expected
        or not 0 < expected["workgroupCount"][0] <= MAX_WORKGROUPS
        or counts != spans
        or max(spans.values()) > SIGNED_MAX
        or threads != spans["out"]
    ):
        raise ValueError("FFT launch differs from fft.cpp's dispatch")
    return {
        "transform": info["transform"],
        "n": n,
        "batchSize": total,
        "inverse": info["inverse"],
        "elements": spans,
    }


def opengl_index_assertions(entry):
    """The subscript ranges OpenGL needs proven, from the host's bounds.

    Device subscripts stay below the signed 32-bit extents ``plan_launch``
    admits; threadgroup subscripts stay inside the entry's ``shared_in``.
    """
    memory = signature(entry)["memory"]

    def assertion(expression, maximum, indexed=None):
        record = {"source": SOURCE, "expression": expression, "minimum": 0}
        record["maximum"] = maximum
        return {**record, "indexed": indexed} if indexed else record

    return [
        *(assertion(expression, SIGNED_MAX) for expression in DEVICE_INDEX_EXPRESSIONS),
        assertion(REFERENCE_INDEX, memory - 1, SHARED_STORAGE),
        *(
            assertion(expression, SIGNED_MAX, indexed)
            for storage in DEVICE_STORAGES
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
