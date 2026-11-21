"""Units, launches, cases and references for fft.metal's current-tree gate.

``fft.metal`` instantiates ``fft``, ``rader_fft``, ``bluestein_fft`` and
``four_step_fft`` at the threadgroup memory sizes 256 to 4096, for float2
and complex32_t complex-to-complex transforms and float real-to-complex and
complex-to-real transforms, plus the plain ``generate_bluestein_twiddles``
kernel: 101 entries. ``fft.cpp`` freezes a plan's function constants into
each pipeline and names it by its hash name, so a unit here is one hash
name: an entry with the constants and threadgroup of one plan.

Of the 101 entries, ``fft.cpp`` reaches 61, all float or float2. Their units
are the host's hash names (``portable_host/fft_layout``), and the gate
checks that each unit's launch equals the host's ``signature``. The others
get gate-local units with the same naming and planning rules:

- ``rader_fft`` at 4096 points of threadgroup memory: ``fft.cpp`` sends
  every transform past ``MAX_RADER_FFT_SIZE`` (2048) to Bluestein, so the
  unit plans the Rader transform of a length between 2048 and 4096 that
  ``plan_fft`` would plan by Rader without that limit.
- ``four_step_fft`` passes at sizes ``fft.cpp``'s splits never produce:
  step 0 at 256 points and step 1 at 512, 1024 and 2048. The units split
  ``8192``, ``16384`` and ``32768`` points as ``64 x 128``, ``64 x 256`` and
  ``64 x 512`` and batch each pass as ``fft.cpp`` batches four-step passes.
- every complex32_t entry: MLX has no complex32 array type, so the units
  repeat the float2 units' plans in complex32_t storage.

Each single-threadgroup unit runs ``fft.cpp``'s grid over a few transforms,
with the Rader and Bluestein tables ``compute_raders_constants`` and
``compute_bluestein_constants`` build, here from their double-precision
values. Each four-step unit runs one pass of its split; the step 0
reference is the column DFTs times the four-step twiddles, and the step 1
reference the row DFTs of a step 0 intermediate, both in double precision.
"""

import functools
import hashlib
import re

import numpy as np

from demos.integrations.mlx.portable_host import fft_layout as layout

SOURCE = layout.SOURCE
HEADERS = (
    "mlx/backend/metal/kernels/fft.h",
    "mlx/backend/metal/kernels/fft/radix.h",
    "mlx/backend/metal/kernels/fft/readwrite.h",
)
HOST_SOURCE = "mlx/backend/metal/fft.cpp"
UPSTREAM_TEST = "python/tests/test_fft.py"
MEMORY_SIZES = layout.MEMORY_SIZES
HALF = "complex32_t"
COMPLEX_TYPES = ("float2", HALF)
STORAGE = {**layout.STORAGE, HALF: "complex32"}
KERNEL_NAMES = {"stockham": "fft", "rader": "rader_fft", "bluestein": "bluestein_fft"}
TRANSFORM_TYPES = {
    "c2c": ("float2", "float2"),
    "r2c": ("float", "float2"),
    "c2r": ("float2", "float"),
}
# The directions fft.cpp runs each transform in.
DIRECTIONS = {"c2c": (False, True), "r2c": (False,), "c2r": (True,)}
# fft.metal instantiate_four_step's (input, output, step, real) arguments by
# transform, in instantiate_ffts order.
FOUR_STEP_ARGUMENTS = {
    "c2c": (("float2", "float2", 0, False), ("float2", "float2", 1, False)),
    "r2c": (("float", "float2", 0, True), ("float2", "float2", 1, True)),
    "c2r": (("float2", "float2", 0, True), ("float2", "float", 1, True)),
}
# fft.cpp's four-step splits of the powers of two the units run, and the
# gate-local splits that reach the passes fft.cpp never plans.
HOST_SPLITS = ((128, 64), (256, 64), (512, 64), (1024, 64), (128, 1024))
LOCAL_SPLITS = ((64, 128), (64, 256), (64, 512))
# Transforms per batch of each four-step case, as test_fft's big powers of
# two run them.
FOUR_STEP_BATCH = 3
# test_fft._run_ffts' tolerances: test_fft_shared_mem's by length, and
# test_fft_big_powers_of_two's past 4096 points.
RTOL = 1e-4
# binary16's unit roundoff: complex32_t kernels compute in half precision.
HALF_UNIT_ROUNDOFF = 2.0**-11


def instantiated_entries():
    """fft.metal's entries in instantiate_ffts order, then the twiddle kernel."""
    entries = []
    for memory in MEMORY_SIZES:
        for complex_type in COMPLEX_TYPES:
            pair = f"{complex_type}_{complex_type}"
            entries += [
                f"fft_mem_{memory}_{pair}",
                f"rader_fft_mem_{memory}_{pair}",
                f"bluestein_fft_mem_{memory}_{pair}",
                f"four_step_mem_{memory}_{pair}_0_false",
                f"four_step_mem_{memory}_{pair}_1_false",
            ]
        for kernel in KERNEL_NAMES.values():
            for pair in ("float_float2", "float2_float"):
                entries.append(f"{kernel}_mem_{memory}_{pair}")
        for transform in ("r2c", "c2r"):
            for input_type, output_type, step, _real in FOUR_STEP_ARGUMENTS[transform]:
                entries.append(
                    f"four_step_mem_{memory}_{input_type}_{output_type}_{step}_true"
                )
    entries.append(layout.TWIDDLE_ENTRY)
    return tuple(entries)


ENTRIES = instantiated_entries()

INSTANTIATION_LINES = (
    "  instantiate_fft(tg_mem_size, complex_T, complex_T) \\",
    "  instantiate_rader(tg_mem_size, complex_T, complex_T) \\",
    "  instantiate_bluestein(tg_mem_size, complex_T, complex_T) \\",
    "  instantiate_four_step(tg_mem_size, complex_T, complex_T, 0, /*real=*/false) \\",
    "  instantiate_four_step(tg_mem_size, complex_T, complex_T, 1, /*real=*/false)",
    "  instantiate_c2c_ffts(tg_mem_size, float2) \\",
    "  instantiate_c2c_ffts(tg_mem_size, complex32_t) \\",
    "  instantiate_fft(tg_mem_size, float, float2) \\",
    "  instantiate_fft(tg_mem_size, float2, float) \\",
    "  instantiate_rader(tg_mem_size, float, float2) \\",
    "  instantiate_rader(tg_mem_size, float2, float) \\",
    "  instantiate_bluestein(tg_mem_size, float, float2) \\",
    "  instantiate_bluestein(tg_mem_size, float2, float) \\",
    "  instantiate_four_step(tg_mem_size, float, float2, 0, /*real=*/true) \\",
    "  instantiate_four_step(tg_mem_size, float2, float2, 1, /*real=*/true) \\",
    "  instantiate_four_step(tg_mem_size, float2, float2, 0, /*real=*/true) \\",
    "  instantiate_four_step(tg_mem_size, float2, float, 1, /*real=*/true)",
)


def check_instantiation(text):
    """Refuse a fft.metal whose instantiations differ from ``ENTRIES``."""
    lines = text.splitlines()
    for line in INSTANTIATION_LINES:
        assert lines.count(line) == 1, line
    sizes = [
        int(match)
        for match in re.findall(r"^instantiate_ffts\((\d+)\)", text, re.MULTILINE)
    ]
    assert tuple(sizes) == MEMORY_SIZES, sizes
    names = re.findall(r'"([a-z_]+)_mem_" #tg_mem_size', text)
    assert names == ["fft", "rader_fft", "bluestein_fft", "four_step"], names


def check_kernels(header):
    """Refuse an fft.h whose kernels or function constants moved."""
    for kernel in ("fft", "rader_fft", "bluestein_fft", "four_step_fft"):
        assert f"[[kernel]] void {kernel}(" in header, kernel
    assert "[[kernel]] void generate_bluestein_twiddles(" in header
    for name, index in layout.CONSTANT_IDS.items():
        assert f"{name} [[function_constant({index})]]" in header, name
    assert (
        f"{layout.TWIDDLE_TABLE_CONSTANT} [[function_constant(22)]]" in header
    ), "use_bluestein_twiddle_table_"


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


def _radix_count(plan):
    return sum(1 for steps in (*plan["stockham"], *plan["rader"]) if steps)


def unbounded_rader_plan(n):
    """plan_fft's Rader plan of ``n`` without MAX_RADER_FFT_SIZE, or None."""
    radices = set(layout.RADICES)
    factors = [factor for factor in layout.prime_factors(n) if factor not in radices]
    if len(factors) != 1:
        return None
    (factor,) = factors
    if any(item not in radices for item in layout.prime_factors(factor - 1)):
        return None
    return {
        "n": n,
        "stockham": layout.plan_stockham_fft(n // factor),
        "rader": layout.plan_stockham_fft(factor - 1),
        "rader_n": factor,
        "bluestein_n": -1,
        "four_step": False,
        "n1": 0,
        "n2": 0,
    }


@functools.lru_cache(maxsize=None)
def single_lengths():
    """Every length fft.cpp plans as one Stockham or Rader threadgroup, by
    (kind, threadgroup memory)."""
    found = {}
    for n in range(2, layout.MAX_STOCKHAM_FFT_SIZE + 1):
        plan = layout.plan_fft(n)
        kind = layout.plan_kind(plan)
        if kind in ("stockham", "rader"):
            memory = layout.next_power_of_2(layout.threadgroup_batch_size(n) * n)
            found.setdefault((kind, memory), []).append(n)
    return found


def _richest(lengths, plan):
    """The length whose plan uses the most radices, the longest on ties."""
    return max(lengths, key=lambda n: (_radix_count(plan(n)), n))


@functools.lru_cache(maxsize=None)
def chosen_lengths():
    """The transform length of each Stockham and Rader unit, by (kind,
    memory), and the Bluestein convolution size of each Bluestein unit."""
    lengths = {
        key: _richest(values, layout.plan_fft)
        for key, values in single_lengths().items()
    }
    # fft.cpp never plans a Rader transform past 2048 points.
    assert ("rader", 4096) not in lengths
    local = [
        n
        for n in range(layout.MAX_RADER_FFT_SIZE + 1, layout.MAX_STOCKHAM_FFT_SIZE + 1)
        if unbounded_rader_plan(n) is not None
    ]
    lengths["rader", 4096] = _richest(local, unbounded_rader_plan)
    for memory in MEMORY_SIZES:
        assert layout.bluestein_lengths(memory), memory
        lengths["bluestein", memory] = memory
    return lengths


def _single_constants(plan, n, inverse, kind, transform, precomputed):
    constants = {
        "inv_": bool(inverse),
        "is_power_of_2_": layout.is_power_of_2(n),
        "elems_per_thread_": layout.compute_elems_per_thread(plan),
        "rader_m_": plan["n"] // plan["rader_n"],
    }
    constants.update(zip(layout.STOCKHAM_CONSTANTS, plan["stockham"]))
    constants.update(zip(layout.RADER_CONSTANTS, plan["rader"]))
    if kind == "bluestein" and transform == "c2c":
        constants[layout.TWIDDLE_TABLE_CONSTANT] = bool(precomputed)
    return constants


def single_signature(kind, memory, types, n, inverse, precomputed=False):
    """``fft_layout.signature``'s launch for a Stockham, Rader or Bluestein
    unit of any instantiated entry, planned as fft.cpp plans ``n`` (the
    convolution size for Bluestein); Rader plans past 2048 points ignore
    MAX_RADER_FFT_SIZE."""
    input_type, output_type = types
    transform = (
        "c2c"
        if input_type == output_type
        else layout.TRANSFORMS[input_type, output_type]
    )
    if kind == "bluestein":
        lengths = layout.bluestein_lengths(n)
        plan = layout.plan_fft(lengths[0])
    else:
        lengths = (n,)
        plan = layout.plan_fft(n)
        if kind == "rader" and n > layout.MAX_RADER_FFT_SIZE:
            plan = unbounded_rader_plan(n)
        assert plan is not None and layout.plan_kind(plan) == kind, (kind, n)
    batch = layout.threadgroup_batch_size(n)
    assert layout.next_power_of_2(batch * n) == memory, (kind, memory, n)
    constants = _single_constants(plan, n, inverse, kind, transform, precomputed)
    elems = constants["elems_per_thread_"]
    threads = (n + elems - 1) // elems
    return {
        "kind": kind,
        "source": f"{KERNEL_NAMES[kind]}_mem_{memory}_{input_type}_{output_type}",
        "transform": transform,
        "n": n,
        "inverse": bool(inverse),
        "raderN": plan["rader_n"],
        "lengths": lengths,
        "precomputed": bool(precomputed),
        "input": STORAGE[input_type],
        "output": STORAGE[output_type],
        "memory": memory,
        "threadgroupBatchSize": batch,
        "threadsPerFFT": threads,
        "constants": constants,
        "workgroupSize": (1, batch, threads),
    }


def four_step_signature(arguments, n, inverse, complex_type="float2"):
    """``fft_layout``'s four-step launch for one pass of ``n`` points of the
    instantiation ``arguments`` (input, output, step, real), batched as
    fft.cpp batches four-step passes."""
    input_type, output_type, step, real = arguments
    rename = {"float2": complex_type}
    input_type, output_type = rename.get(input_type, input_type), rename.get(
        output_type, output_type
    )
    batch = layout.four_step_threadgroup_batch_size(n)
    memory = layout.next_power_of_2(batch * n)
    assert memory in MEMORY_SIZES, (n, memory)
    transform = {
        args: name for name, items in FOUR_STEP_ARGUMENTS.items() for args in items
    }[tuple(arguments)]
    constants = layout.plan_constants(n, inverse, "stockham")
    elems = constants["elems_per_thread_"]
    threads = (n + elems - 1) // elems
    return {
        "kind": "four_step",
        "source": (
            f"four_step_mem_{memory}_{input_type}_{output_type}_{step}_"
            f"{'true' if real else 'false'}"
        ),
        "transform": transform,
        "n": n,
        "step": step,
        "real": real,
        "inverse": bool(inverse),
        "raderN": 1,
        "lengths": (n,),
        "input": STORAGE[input_type],
        "output": STORAGE[output_type],
        "memory": memory,
        "threadgroupBatchSize": batch,
        "threadsPerFFT": threads,
        "constants": constants,
        "workgroupSize": (1, batch, threads),
    }


def unit_name(info):
    """fft.cpp's hash name of a launch: ``<source>_n<size>_inv_<inverse>``."""
    if info["kind"] == "twiddles":
        return info["source"]
    suffix = layout.PRECOMPUTED if info.get("precomputed") else ""
    return f"{info['source']}_n{info['n']}_inv_{int(info['inverse'])}{suffix}"


def _host_reaches(name):
    try:
        layout.signature(name)
    except ValueError:
        return False
    return True


@functools.lru_cache(maxsize=None)
def units():
    """Every unit by hash name: its launch, whether fft.cpp reaches it, and
    the four-step split each four-step unit runs."""
    found = {}

    def add(info, split=None):
        name = unit_name(info)
        host = _host_reaches(name)
        if host:
            # The host derives the same launch from the hash name.
            assert layout.signature(name) == info, name
        record = {"info": info, "hostReachable": host}
        if split is not None:
            record["split"] = split
        assert found.setdefault(name, record) == record, name

    lengths = chosen_lengths()
    for complex_type in COMPLEX_TYPES:
        for memory in MEMORY_SIZES:
            for kind in KERNEL_NAMES:
                transforms = ("c2c",) if complex_type == HALF else TRANSFORM_TYPES
                for transform in transforms:
                    types = (
                        (HALF, HALF)
                        if complex_type == HALF
                        else TRANSFORM_TYPES[transform]
                    )
                    for inverse in DIRECTIONS[transform]:
                        add(
                            single_signature(
                                kind, memory, types, lengths[kind, memory], inverse
                            )
                        )
    # fft.cpp's twiddle-table plans: complex 4096-point Bluestein batches.
    for inverse in (False, True):
        add(
            single_signature(
                "bluestein", 4096, ("float2", "float2"), 4096, inverse, True
            )
        )
    for complex_type in COMPLEX_TYPES:
        transforms = ("c2c",) if complex_type == HALF else TRANSFORM_TYPES
        for transform in transforms:
            for inverse in DIRECTIONS[transform]:
                for split in HOST_SPLITS + LOCAL_SPLITS:
                    for arguments in FOUR_STEP_ARGUMENTS[transform]:
                        size = split[arguments[2]]
                        info = four_step_signature(
                            arguments, size, inverse, complex_type
                        )
                        name = unit_name(info)
                        if name not in found:
                            add(info, split)
    add(layout.signature(layout.TWIDDLE_ENTRY))
    return found


UNITS = units()


def unit_entries():
    """The entry each unit translates, in unit order."""
    return {name: record["info"]["source"] for name, record in UNITS.items()}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def tolerance(length):
    """test_fft's (atol, rtol) for a transform of ``length`` points."""
    if length > layout.MAX_STOCKHAM_FFT_SIZE:
        if length >= 1 << 20:
            return 1e-2, 1e-3
        return (1e-3 if length < 1 << 17 else 1e-2), RTOL
    return (1e-4 if length < 1025 else 1e-3), RTOL


def binary16(values):
    """``values`` with each real and imaginary part rounded to binary16."""
    parts = np.ascontiguousarray(values, dtype=np.complex64).view(np.float32)
    return parts.astype(np.float16).astype(np.float32).view(np.complex64)


def _complex32_inputs(values):
    """complex32_t inputs: test_fft's halved, so that every forward transform
    up to 2^17 points stays below binary16's largest finite value, and
    rounded to binary16, so that the reference transforms the exact stored
    words."""
    return binary16(np.asarray(values, dtype=np.complex64) / np.float32(2))


def _log2_ceil(n):
    return int(n - 1).bit_length()


def half_levels(info):
    """The radix-2 levels a complex32_t unit chains: one per halving of each
    transform it runs, plus one per pointwise product with a table: Rader's
    kernel spectrum, Bluestein's two chirps and kernel spectrum, and the
    four-step twiddles at the end of step 0."""
    kind, n = info["kind"], info["n"]
    if kind == "stockham":
        return _log2_ceil(n)
    if kind == "rader":
        return _log2_ceil(n) + _log2_ceil(info["raderN"] - 1) + 1
    if kind == "bluestein":
        return 2 * _log2_ceil(n) + 3
    if kind == "four_step":
        return _log2_ceil(n) + (1 if info["step"] == 0 else 0)
    raise KeyError(kind)


def half_normwise_bound(levels):
    """Higham's bound for a radix-2 FFT of ``2**levels`` points (Accuracy and
    Stability of Numerical Algorithms, 2nd ed., Theorem 24.2) in binary16:
    ``||y_hat - y|| / ||y|| <= t eta / (1 - t eta)``, ``eta = mu +
    gamma_4 (sqrt(2) + mu)``, with ``t`` the unit's levels and twiddles
    rounded to binary16, ``mu = u``."""
    u = HALF_UNIT_ROUNDOFF
    gamma4 = 4 * u / (1 - 4 * u)
    eta = u + gamma4 * (np.sqrt(2.0) + u)
    return float(levels * eta / (1 - levels * eta))


def normwise(actual, reference):
    """The relative 2-norm error of ``actual``; infinite unless it is finite."""
    actual = np.asarray(actual)
    if not np.isfinite(actual).all():
        return float("inf")
    reference = np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def _accuracy(name, length):
    """test_fft's elementwise tolerances for float units; complex32_t units,
    which test_fft cannot run, bound the normwise error instead."""
    if is_half(name):
        return {
            "tolerance": None,
            "normwiseBound": half_normwise_bound(half_levels(UNITS[name]["info"])),
        }
    return {"tolerance": tolerance(length), "normwiseBound": None}


def _rng(name, label):
    digest = hashlib.sha256(f"{name}:{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _complex_inputs(rng, shape):
    """test_fft's inputs: real and imaginary parts uniform in [0, 1) in
    float32."""
    real = rng.random(shape, dtype=np.float32)
    imag = rng.random(shape, dtype=np.float32)
    return (real + 1j * imag).astype(np.complex64)


def single_transforms(info):
    """The transforms one single-threadgroup case runs: a partial last
    threadgroup when several transforms share one, else three."""
    batch = info["threadgroupBatchSize"]
    return 2 * batch + 1 if batch > 1 else 3


def _reference(transform, inverse, values, length):
    data = values.astype(np.complex128 if transform != "r2c" else np.float64)
    if transform == "c2c":
        return (np.fft.ifft if inverse else np.fft.fft)(data, axis=-1)
    if transform == "r2c":
        return np.fft.rfft(data, axis=-1)
    return np.fft.irfft(data, n=length, axis=-1)


def single_case(name, length=None):
    """One case of a single-threadgroup unit: inputs, tables, scalars,
    launch and double-precision reference."""
    record = UNITS[name]
    info = record["info"]
    length = info["n"] if length is None else length
    total = single_transforms(info)
    rng = _rng(name, length)
    transform = info["transform"]
    if transform == "r2c":
        values = rng.random((total, length), dtype=np.float32)
    elif transform == "c2r":
        real = rng.random((total, length), dtype=np.float32)
        values = np.fft.rfft(real.astype(np.float64), axis=-1).astype(np.complex64)
    else:
        values = _complex_inputs(rng, (total, length))
        if is_half(name):
            values = _complex32_inputs(values)
    tables = {}
    scalars = {"n": info["n"], "batch_size": total}
    if info["kind"] == "rader":
        rader_n = info["raderN"]
        g_q, g_minus_q = layout.rader_generator_tables(rader_n)
        tables = {
            "raders_b_q": np.array(
                layout.rader_kernel_spectrum(rader_n), dtype=np.complex64
            ),
            "raders_g_q": np.array(g_q, dtype=np.int16),
            "raders_g_minus_q": np.array(g_minus_q, dtype=np.int16),
        }
        scalars["rader_n"] = rader_n
    if info["kind"] == "bluestein":
        chirp, spectrum = layout.bluestein_tables(length, info["n"])
        if info["precomputed"]:
            spectrum = spectrum + layout.bluestein_twiddles()
        tables = {
            "w_q": np.array(spectrum, dtype=np.complex64),
            "w_k": np.array(chirp, dtype=np.complex64),
        }
        scalars["length"] = length
    spans = layout.extents(info, total, length)
    reach = layout.footprint(info, total, length)
    return {
        "unit": name,
        "length": length,
        "transforms": total,
        "input": values,
        "tables": tables,
        "scalars": scalars,
        "spans": spans,
        "footprint": reach,
        "launch": layout.launch(info, total),
        "reference": _reference(transform, info["inverse"], values, length),
        **_accuracy(name, length),
    }


def _four_step_twiddles(n1, n2):
    overall = n1 * n2
    k1 = np.arange(n1)[:, None]
    column = np.arange(n2)[None, :]
    return np.exp(-2j * np.pi * k1 * column / overall)


def four_step_intermediate(values, n1, n2):
    """fft.cpp's four-step temporary after step 0: each column's ``n1``-point
    DFT times the twiddles ``exp(-2 pi i k1 c / (n1 n2))``, stored at
    ``c + k1 * n2``."""
    columns = values.reshape(values.shape[0], n1, n2).astype(np.complex128)
    return (np.fft.fft(columns, axis=1) * _four_step_twiddles(n1, n2)).reshape(
        values.shape[0], n1 * n2
    )


def _full_spectrum(half, overall):
    """The Hermitian spectrum of ``overall`` points ``half`` holds."""
    tail = np.conj(half[:, 1 : overall - overall // 2][:, ::-1])
    return np.concatenate([half, tail], axis=1)


def _step1_rows(intermediate, n1, n2):
    """Each row's ``n2``-point DFT of the temporary, in frequency order
    ``k1 + k2 * n1``."""
    rows = intermediate.reshape(intermediate.shape[0], n1, n2)
    return (
        np.fft.fft(rows, axis=2)
        .transpose(0, 2, 1)
        .reshape(intermediate.shape[0], n1 * n2)
    )


def four_step_case(name):
    """One case of a four-step unit: one pass over FOUR_STEP_BATCH
    transforms of its split, with its double-precision reference.

    Step 1 reads the float32 rounding of the reference step 0 temporary, or
    its binary16 rounding for complex32_t.
    """
    record = UNITS[name]
    info = record["info"]
    n1, n2 = record["split"]
    overall = n1 * n2
    batch = FOUR_STEP_BATCH
    transform, inverse, step = info["transform"], info["inverse"], info["step"]
    rng = _rng(name, overall)
    real = rng.random((batch, overall), dtype=np.float32)
    signal = _complex_inputs(rng, (batch, overall))
    half = is_half(name)
    if half:
        signal = _complex32_inputs(signal)
    if transform == "r2c":
        source = real
    elif transform == "c2r":
        source = np.fft.rfft(real.astype(np.float64), axis=-1).astype(np.complex64)
    else:
        source = signal
    if transform == "c2r":
        columns = np.conj(_full_spectrum(source.astype(np.complex128), overall))
    elif transform == "c2c" and inverse:
        columns = np.conj(source.astype(np.complex128))
    else:
        columns = source.astype(np.complex128)
    intermediate = four_step_intermediate(columns, n1, n2)
    # complex32_t passes normalize an inverse transform on input, by 1/n per
    # pass (readwrite.h's load_strided), so its temporary carries 1/n1.
    scale = n1 if half and transform == "c2c" and inverse else 1
    if step == 0:
        values, reference = source, intermediate / scale
    else:
        values = intermediate / scale
        values = binary16(values) if half else values.astype(np.complex64)
        rows = _step1_rows(values.astype(np.complex128), n1, n2)
        if transform == "c2c":
            reference = np.conj(rows) / (overall // scale) if inverse else rows
        elif transform == "r2c":
            reference = rows[:, : overall // 2 + 1]
        else:
            reference = rows.real / overall
    total = batch * (n2 if step == 0 else n1)
    spans = split_extents(info, total, n1, n2)
    return {
        "unit": name,
        "split": [n1, n2],
        "transforms": total,
        "input": values,
        "tables": {},
        "scalars": {"n1": n1, "n2": n2, "batch_size": total},
        "spans": spans,
        "footprint": dict(spans),
        "launch": layout.launch(info, total),
        "reference": reference,
        **_accuracy(name, overall),
    }


def split_extents(info, total, n1, n2):
    """``fft_layout.four_step_extents`` for any split, planned or not."""
    overall = n1 * n2
    assert info["n"] == (n1, n2)[info["step"]] and total * info["n"] % overall == 0
    transforms = total * info["n"] // overall
    full, half = transforms * overall, transforms * (overall // 2 + 1)
    return {
        ("c2c", 0): {"in": full, "out": full},
        ("c2c", 1): {"in": full, "out": full},
        ("r2c", 0): {"in": full, "out": full},
        ("r2c", 1): {"in": full, "out": half},
        ("c2r", 0): {"in": half, "out": full},
        ("c2r", 1): {"in": full, "out": full},
    }[info["transform"], info["step"]]


def twiddle_case():
    """generate_bluestein_twiddles' single threadgroup and its table."""
    info = UNITS[layout.TWIDDLE_ENTRY]["info"]
    size = layout.BLUESTEIN_TWIDDLE_TABLE_SIZE
    return {
        "unit": layout.TWIDDLE_ENTRY,
        "transforms": 1,
        "input": None,
        "tables": {},
        "scalars": {},
        "spans": {"twiddles": size},
        "footprint": {"twiddles": size},
        "launch": {
            "workgroupCount": [1, 1, 1],
            "workgroupSize": list(info["workgroupSize"]),
        },
        "reference": np.array(layout.bluestein_twiddles(), dtype=np.complex128),
        # fft_dispatch's bound on the float32 cos and sin of the table.
        "tolerance": (1e-5, 0.0),
    }


def cases(name):
    """Every case of a unit."""
    info = UNITS[name]["info"]
    if info["kind"] == "twiddles":
        return [twiddle_case()]
    if info["kind"] == "four_step":
        return [four_step_case(name)]
    if info["kind"] == "bluestein" and not info["precomputed"]:
        # The first length compiles the pipeline; the last reuses it.
        return [
            single_case(name, info["lengths"][0]),
            single_case(name, info["lengths"][-1]),
        ]
    if info["kind"] == "bluestein":
        return [single_case(name, info["lengths"][0])]
    return [single_case(name)]


def compare(actual, reference, tol):
    """test_fft's ``assert_allclose(out_np, out_mx, atol, rtol)``: every
    element within ``atol + rtol * |out_mx|``; returns the worst ratio of
    error to bound."""
    atol, rtol = tol
    actual = np.asarray(actual)
    error = np.abs(np.asarray(reference) - actual)
    bound = atol + rtol * np.abs(actual)
    finite = np.isfinite(actual).all()
    worst = float(np.max(error / bound)) if error.size else 0.0
    return bool(finite and worst <= 1.0), worst


def expected_unit_counts():
    """Units by (kind, storage, host reachability)."""
    counts = {}
    for record in UNITS.values():
        info = record["info"]
        key = (
            info["kind"],
            "complex32" if HALF in info["source"] else "float",
            record["hostReachable"],
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


def is_half(name):
    return HALF in UNITS[name]["info"]["source"]
