"""Translate and execute the pinned MLX softmax kernels on all backends."""

import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    load_project_config,
    translate_project,
    validate_project_report,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.kernels.test_current_cross_entropy import (
    GUARD_BITS,
    GUARD_COUNT,
    _bits,
    _decode,
    _element_layout,
    _from_bits,
    _guard_values,
    _quantize,
    _runtime_value,
)
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/softmax.metal"
SOURCE_SHA256 = "d19231c66973edc3944f12529d1cc393029e7f7262b914907c710ef9dbcb39e2"
HEADER = "mlx/backend/metal/kernels/softmax.h"
HEADER_SHA256 = "cb233fe42a7358f5f658c9a1ef52cd2ee9dd3786ddfad01793a976d58f4d4051"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_SOFTMAX"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/softmax.current-tree.translation.json"
)
# (storage type, precise) in softmax.metal's instantiation order. Precise
# variants accumulate in float; the others accumulate in the storage type.
TYPES = (
    ("float32", False),
    ("float16", False),
    ("bfloat16", False),
    ("float16", True),
    ("bfloat16", True),
)
KINDS = ("block", "looped")
ENTRIES = tuple(
    f"{kind}_softmax_{'precise_' if precise else ''}{dtype}"
    for dtype, precise in TYPES
    for kind in KINDS
)
TARGETS = ("metal", "opengl", "directx")
N_READS = 4
SIMD_SIZE = 32
# mlx/backend/metal/softmax.cpp: rows longer than 4096 use the looped kernel
# with maxTotalThreadsPerThreadgroup threads (1024 for these pipelines on Apple
# GPUs); shorter rows use 32 * ceil(ceil(axis / 4) / 32) block threads.
LOOPED_LIMIT = 4096
LOOPED_WORKGROUP = 1024
# One SIMD group, a non-power-of-two count of nine, and all 32 SIMD groups.
WORKGROUP_SIZES = {"block": (32, 288, 1024), "looped": (LOOPED_WORKGROUP,)}
# OpenGL indexes 32-bit storage arrays. These host preconditions bound the
# rebased row offsets and the composed element indices to signed int range.
SIGNED_MAX = 2**31 - 1
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "block": (
        "gid * uint64(axis_size) + lid * SOFTMAX_N_READS",
        "in_ + i",
        "out_ + i",
    ),
    "looped": ("gid * uint64(axis_size)", "in_ + offset + i", "out_ + offset + i"),
}
# OpenGL and DirectX lower the SIMD reductions to software 32-lane groups, so
# results do not depend on the device's subgroup width. The block kernel's
# second reductions run under ``simd_group_id == 0``; both targets predicate
# them so every invocation reaches the barriers.
SOFTWARE_SUBGROUP_TARGETS = ("opengl", "directx")
SUBGROUPS = {
    "metal": "native-simdgroup-32",
    "opengl": "software-subgroup-32",
    "directx": "software-subgroup-32",
}
# name -> (rows, axis size, values)
CASES = {
    "one": (3, 1, "uniform"),
    "tail": (5, 7, "uniform"),
    "rows": (64, 32, "uniform"),
    "full": (4, 128, "uniform"),
    "masked": (3, 100, "masked"),
    "large": (3, 77, "large"),
    "infinite": (2, 40, "infinite"),
    "simd-groups": (2, 1152, "uniform"),
    "simd-tail": (3, 1025, "uniform"),
    "group": (2, 4096, "uniform"),
    "group-tail": (2, 3969, "large"),
    "two-passes": (2, 4097, "uniform"),
    "passes": (1, 20000, "uniform"),
    "rising": (1, 12288, "rising"),
    "looped-large": (2, 5000, "large"),
    "looped-masked": (1, 4500, "masked"),
    "looped-infinite": (2, 4200, "infinite"),
}
# Each output is checked against a float64 softmax with a per-element bound on
# the kernel's own rounding: every operation in the accumulation type rounds
# once (relative error UNIT, absolute TINY below the normal range), the
# exponentials may be (3 + 2|x|) binary32 units in the last place from exp(x)
# as Metal's fast::exp allows, plus EXP_SLACK in the accumulation type, and the
# sums are bounded by their depth: 31 sequential binary32 additions per SIMD
# reduction, or the five rounded levels of the binary16 pairwise tree. The
# result must then be a correct rounding of a value in that interval, so
# truncation, a dropped contribution or a stale normalizer fails.
UNIT = {"float32": 2.0**-24, "float16": 2.0**-11, "bfloat16": 2.0**-8}
TINY = {"float32": 2.0**-150, "float16": 2.0**-25, "bfloat16": 2.0**-134}
EXP_SLACK = {"float32": 0.0, "float16": 4 * 2.0**-11, "bfloat16": 2.0**-8}
SIMD_REDUCTION_ERROR = {
    "float32": 31 * 2.0**-24,
    "float16": 5 * 2.0**-11,
    # MLX's bfloat16 simd_sum reduces a float conversion and rounds once.
    "bfloat16": 31 * 2.0**-24 + 2.0**-8,
}
SIGNIFICAND_BITS = {"float32": 24, "float16": 11, "bfloat16": 8}
# frexp exponent of the smallest normal value of each type.
MIN_NORMAL_EXPONENT = {"float32": -125, "float16": -13, "bfloat16": -125}
INPUT_GUARD_COUNT = 16


def _dtype(entry):
    return entry.rsplit("_", 1)[1]


def _kind(entry):
    return entry.split("_", 1)[0]


def _precise(entry):
    return "_precise_" in entry


def _accumulator(entry):
    """Return the type softmax accumulates in for ``entry``."""
    dtype = _dtype(entry)
    return "float32" if dtype == "float32" or _precise(entry) else dtype


def _workgroup_size(axis):
    """Return MLX's threadgroup size for one row of ``axis`` elements."""
    if axis > LOOPED_LIMIT:
        return LOOPED_WORKGROUP
    threads = (axis + N_READS - 1) // N_READS
    return SIMD_SIZE * ((threads + SIMD_SIZE - 1) // SIMD_SIZE)


def _case_kind(axis):
    return "looped" if axis > LOOPED_LIMIT else "block"


VARIANTS = tuple(
    (entry, size) for entry in ENTRIES for size in WORKGROUP_SIZES[_kind(entry)]
)


def _variant_cases(entry, size):
    return tuple(
        name
        for name, (_, axis, _) in CASES.items()
        if _case_kind(axis) == _kind(entry) and _workgroup_size(axis) == size
    )


def _infinite_column(axis):
    # Late in the row, so the looped kernel meets it in its second pass.
    return axis - 7


def _values(name, dtype):
    rows, axis, kind = CASES[name]
    state = (zlib.crc32(name.encode("utf-8")) << 1) | 1
    values = []
    for row in range(rows):
        for column in range(axis):
            state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
            unit = ((state >> 11) / 2**53) * 2.0 - 1.0
            value = 4.0 * unit
            if kind == "masked" and column % 9 == 4:
                value = -math.inf  # masked entries contribute nothing
            elif kind == "large":
                value = 1000.0 + 6.0 * unit  # exp overflows without the max
            elif kind == "rising":
                # The maximum grows in every pass, so the running normalizer
                # must be rescaled.
                value = 6.0 * column / axis - 3.0 + 0.5 * unit
            elif (
                kind == "infinite"
                and row == rows - 1
                and column == _infinite_column(axis)
            ):
                value = math.inf
            values.append(_quantize(dtype, value))
    return values


def _reference(name, dtype):
    """Return the inputs and a float64 softmax of each row.

    A row holding +inf has a NaN normalizer in MLX's kernel (exp(inf - inf)),
    so each of its outputs is NaN.
    """
    rows, axis, _ = CASES[name]
    values = _values(name, dtype)
    outputs = []
    for row in range(rows):
        row_values = values[row * axis : (row + 1) * axis]
        maximum = max(row_values)
        if math.isinf(maximum):
            outputs.extend([math.nan] * axis)
            continue
        exponentials = [math.exp(value - maximum) for value in row_values]
        total = math.fsum(exponentials)
        outputs.extend(item / total for item in exponentials)
    return values, outputs


def _exp_error(accumulator, distance):
    return (3 + 2 * distance) * 2.0**-23 + EXP_SLACK[accumulator]


def _bounds(entry, size, name):
    """Return each output's relative and absolute error bound."""
    rows, axis, _ = CASES[name]
    accumulator = _accumulator(entry)
    unit, tiny = UNIT[accumulator], TINY[accumulator]
    values = _values(name, _dtype(entry))
    looped = _kind(entry) == "looped"
    passes = math.ceil(axis / (N_READS * size)) if looped else 1
    # Per-thread roundings (the looped kernel's first pass starts from an
    # exact zero, then each pass rescales once and adds four terms, and two
    # cross-thread rescales follow), then the SIMD-group and threadgroup
    # reductions.
    depth = (5 * passes if looped else 3) * unit
    depth += 2 * SIMD_REDUCTION_ERROR[accumulator]
    bounds = []
    for row in range(rows):
        row_values = values[row * axis : (row + 1) * axis]
        maximum = max(row_values)
        if math.isinf(maximum):
            bounds.extend([(math.inf, math.inf)] * axis)
            continue
        distances = [maximum - value for value in row_values]
        weights = [math.exp(-distance) for distance in distances]
        total = math.fsum(weights)
        finite = [distance for distance in distances if math.isfinite(distance)]
        spread = max(finite)
        rescale = 0.0
        if looped:
            # Each thread rescales once per pass and twice across threads; the
            # maxima it rescales by rise by at most the row's spread in total.
            count = passes + 2
            rescale = (3 * count + 2 * spread) * 2.0**-23
            rescale += count * EXP_SLACK[accumulator]
        contribution = math.fsum(
            weight * (unit * distance + _exp_error(accumulator, distance))
            for weight, distance in zip(weights, distances)
            if weight
        )
        normalizer = (
            contribution / total
            + rescale
            + depth
            + len(row_values) * tiny / total
            + tiny * total
        )
        for distance in distances:
            if not math.isfinite(distance):
                bounds.append((0.0, 0.0))
                continue
            element = unit * distance + _exp_error(accumulator, distance)
            bounds.append((element + normalizer + 2 * unit, 2 * tiny))
    return bounds


def _round(dtype, value):
    """Round a float64 value to ``dtype`` with ties to even, in one step."""
    if value == 0 or not math.isfinite(value):
        return value
    exponent = max(math.frexp(value)[1], MIN_NORMAL_EXPONENT[dtype])
    quantum = 2.0 ** (exponent - SIGNIFICAND_BITS[dtype])
    return round(value / quantum) * quantum


def _allowed(dtype, expected, bound):
    relative, absolute = bound
    error = relative * abs(expected) + absolute
    return _round(dtype, expected - error), _round(dtype, expected + error)


def _check(dtype, actual, expected, bounds):
    """Check results against the reference.

    Return the largest fraction of an output's allowed deviation that the
    results use, which stays informative where outputs underflow.
    """
    assert len(actual) == len(expected) == len(bounds)
    bad, worst = [], 0.0
    for index, (got, want, bound) in enumerate(zip(actual, expected, bounds)):
        if math.isnan(want):
            if not math.isnan(got):
                bad.append((index, got, want))
            continue
        low, high = _allowed(dtype, want, bound)
        if math.isnan(got) or not low <= got <= high:
            bad.append((index, got, want, low, high))
        elif got != want:
            allowed = high - want if got > want else want - low
            worst = max(worst, abs(got - want) / allowed)
    assert not bad, bad[:8]
    return worst


def _config(output, target, entry, size):
    kind = _kind(entry)
    options = ""
    if target in SOFTWARE_SUBGROUP_TARGETS:
        options += f"""
[project.source_options.metal.target_options.{target}]
software_subgroup_width = {SIMD_SIZE}
"""
    if target == "directx":
        options += 'relative_wave_shuffle_out_of_range = "self"\n'
    if target == "opengl":
        for expression in OPENGL_INDEX_RANGE_ASSERTIONS[kind]:
            options += f"""
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "{expression}"
minimum = 0
maximum = {SIGNED_MAX}
"""
    return f"""
[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
output_dir = "{output}"
[project.entry_points]
"{SOURCE}" = "{entry}"
[project.entry_workgroup_size_rules."{SOURCE}"]
"{entry}" = [{size}, 1, 1]
[project.source_options.metal]
max_template_specializations = 64
max_template_materialization_work = 4096
{options}"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _expected_validation_contract():
    return {
        "workgroupSizes": {
            kind: list(sizes) for kind, sizes in WORKGROUP_SIZES.items()
        },
        "looped": {"axisSizeAbove": LOOPED_LIMIT, "workgroupSize": LOOPED_WORKGROUP},
        "accumulators": {entry: _accumulator(entry) for entry in ENTRIES},
        "subgroups": dict(SUBGROUPS),
        "indexRangeAssertions": {
            "opengl": {
                kind: [
                    {"expression": expression, "minimum": 0, "maximum": SIGNED_MAX}
                    for expression in expressions
                ]
                for kind, expressions in OPENGL_INDEX_RANGE_ASSERTIONS.items()
            },
            "metal": {},
            "directx": {},
        },
        "cases": {
            name: {
                "rows": rows,
                "axisSize": axis,
                "values": values,
                "kernel": _case_kind(axis),
                "workgroupSize": _workgroup_size(axis),
            }
            for name, (rows, axis, values) in CASES.items()
        },
        "errorModel": {
            "unit": dict(UNIT),
            "tiny": dict(TINY),
            "expSlack": dict(EXP_SLACK),
            "expUlpBound": "3 + 2|x| binary32 units in the last place",
            "simdReductionError": dict(SIMD_REDUCTION_ERROR),
        },
        "results": "correct-rounding-of-a-value-within-the-error-bound",
        "infiniteRows": "nan",
        "inputGuardCount": INPUT_GUARD_COUNT,
        "outputGuardCount": GUARD_COUNT,
        "observedInputCount": 2,
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def _expected_scope():
    return {
        "coveredEntryCount": len(ENTRIES),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "artifactCountByTarget": len(VARIANTS),
        "checkedElementCountByEntry": {
            entry: sum(
                CASES[name][0] * CASES[name][1]
                for size in WORKGROUP_SIZES[_kind(entry)]
                for name in _variant_cases(entry, size)
            )
            for entry in ENTRIES
        },
        "completeCorpusEntryCount": 17832,
        "fastMathLowering": "metal-keeps-fast-exp-others-use-default-exp",
        "allMaskedRowsCovered": False,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_softmax_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-softmax-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "header": HEADER,
        "headerSha256": HEADER_SHA256,
    }
    assert contract["corpusCensus"] == {
        "metalUnitCount": 49,
        "entryCount": 17832,
        "discoveryDiagnosticCount": 0,
    }
    assert tuple(contract["entries"]) == ENTRIES
    assert set(contract["artifacts"]) == set(TARGETS)
    for target in TARGETS:
        artifacts = contract["artifacts"][target]
        assert tuple(artifacts) == ENTRIES
        for entry, records in artifacts.items():
            assert tuple(records) == tuple(
                str(size) for size in WORKGROUP_SIZES[_kind(entry)]
            )
            for record in records.values():
                assert set(record) == {"sha256", "sizeBytes"}
                assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
    assert contract["validationContract"] == _expected_validation_contract()
    assert contract["scope"] == _expected_scope()


def test_softmax_entries_follow_mlx_instantiation():
    source = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "softmax.current-tree.translation.json" in source
    assert len(ENTRIES) == len(set(ENTRIES)) == 10
    assert [_accumulator(entry) for entry in ENTRIES] == [
        "float32",
        "float32",
        "float16",
        "float16",
        "bfloat16",
        "bfloat16",
        "float32",
        "float32",
        "float32",
        "float32",
    ]


def test_softmax_cases_follow_mlx_dispatch():
    # Every pinned workgroup size has cases, and every case runs on the kernel
    # and threadgroup size that MLX would choose for its row length.
    for entry, size in VARIANTS:
        assert _variant_cases(entry, size), (entry, size)
    covered = {name for entry, size in VARIANTS for name in _variant_cases(entry, size)}
    assert covered == set(CASES)
    assert _workgroup_size(1) == _workgroup_size(128) == 32
    assert _workgroup_size(129) == 64 and _workgroup_size(4096) == 1024
    assert _case_kind(4096) == "block" and _case_kind(4097) == "looped"
    passes = {
        math.ceil(axis / (N_READS * LOOPED_WORKGROUP))
        for _, axis, _ in CASES.values()
        if axis > LOOPED_LIMIT
    }
    assert passes == {2, 3, 5}
    # Partial reads, idle threads and a last SIMD group with a single active
    # thread all appear.
    axes = [axis for _, axis, _ in CASES.values()]
    assert any(axis % N_READS for axis in axes)
    assert any(-(-axis // N_READS) % SIMD_SIZE == 1 for axis in axes)
    assert any(_workgroup_size(axis) * N_READS - axis >= 64 for axis in axes)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_softmax_references_cover_edges(dtype):
    for name, (rows, axis, kind) in CASES.items():
        values, outputs = _reference(name, dtype)
        assert len(values) == len(outputs) == rows * axis
        assert all(_quantize(dtype, value) == value for value in values)
        if kind == "infinite":
            assert all(math.isnan(item) for item in outputs[-axis:])
            assert all(math.isfinite(item) for item in outputs[:-axis])
            continue
        assert all(0.0 <= item <= 1.0 for item in outputs)
        for row in range(rows):
            assert math.isclose(
                math.fsum(outputs[row * axis : (row + 1) * axis]), 1.0, rel_tol=1e-12
            )
        if kind == "masked":
            assert any(math.isinf(value) and value < 0 for value in values)
            assert outputs.count(0.0) == sum(1 for value in values if math.isinf(value))
        if kind == "large":
            # Without subtracting the maximum, exp overflows binary32.
            assert min(values) > 900.0
    # The rising row's maximum grows in each of the three passes.
    values, _ = _reference("rising", dtype)
    stride = N_READS * LOOPED_WORKGROUP
    maxima = [max(values[start : start + stride]) for start in range(0, 12288, stride)]
    assert maxima == sorted(maxima) and len(set(maxima)) == 3


def _ulp(dtype, value):
    exponent = max(math.frexp(value)[1], MIN_NORMAL_EXPONENT[dtype])
    return 2.0 ** (exponent - SIGNIFICAND_BITS[dtype])


def _largest_bound(entry, size, name):
    return max(relative for relative, _ in _bounds(entry, size, name))


def test_softmax_bounds_stay_meaningful():
    # The bounds reflect each accumulation type: binary32 within 2^-14, the
    # binary16 block kernel within 2%, and the worst-case envelopes of the
    # long binary16 and bfloat16 chains of the looped kernel within 5% and 20%.
    limits = {
        ("float32", "block"): 2.0**-14,
        ("float32", "looped"): 2.0**-14,
        ("float16", "block"): 0.02,
        ("float16", "looped"): 0.05,
        ("bfloat16", "block"): 0.08,
        ("bfloat16", "looped"): 0.2,
    }
    for entry, size in VARIANTS:
        for name in _variant_cases(entry, size):
            if CASES[name][2] == "infinite":
                continue
            limit = limits[(_accumulator(entry), _kind(entry))]
            assert _largest_bound(entry, size, name) < limit


def test_softmax_checks_reject_defects():
    for entry, size in (("block_softmax_float32", 32), *VARIANTS):
        dtype = _dtype(entry)
        names = _variant_cases(entry, size)
        name = next(item for item in names if CASES[item][2] == "uniform")
        values, expected = _reference(name, dtype)
        bounds = _bounds(entry, size, name)
        rounded = [_round(dtype, value) for value in expected]
        _check(dtype, rounded, expected, bounds)
        rows, axis, _ = CASES[name]
        # Normalizing a row by a sum that misses a quarter of the row, or by
        # the neighbouring row's sum, fails.
        row = values[:axis]
        maximum = max(row)
        exponentials = [math.exp(value - maximum) for value in row]
        partial = math.fsum(exponentials[: max(1, 3 * axis // 4)])
        wrong = [_round(dtype, item / partial) for item in exponentials]
        if partial != math.fsum(exponentials):
            with pytest.raises(AssertionError):
                _check(dtype, wrong + rounded[axis:], expected, bounds)
        swapped = rounded[axis:] + rounded[:axis] if rows > 1 else rounded[::-1]
        if swapped != rounded:
            with pytest.raises(AssertionError):
                _check(dtype, swapped, expected, bounds)
        # Unwritten zeros or guards fail.
        with pytest.raises(AssertionError):
            _check(dtype, [0.0] + rounded[1:], expected, bounds)
        guard = _from_bits(dtype, GUARD_BITS[dtype])
        with pytest.raises(AssertionError):
            _check(dtype, rounded[:-1] + [guard], expected, bounds)
    # Binary32 accumulation leaves no room for one 16-bit unit in the last
    # place, and rounding the precise results by truncation fails.
    for dtype in ("float16", "bfloat16"):
        entry = f"block_softmax_precise_{dtype}"
        _, expected = _reference("full", dtype)
        bounds = _bounds(entry, 32, "full")
        rounded = [_round(dtype, value) for value in expected]
        nudged = [value + _ulp(dtype, value) for value in rounded]
        with pytest.raises(AssertionError):
            _check(dtype, nudged, expected, bounds)
        truncated = []
        for value in expected:
            quantum = _ulp(dtype, value)
            truncated.append(math.floor(value / quantum) * quantum)
        with pytest.raises(AssertionError):
            _check(dtype, truncated, expected, bounds)
    # A looped kernel that skips the running rescale fails on the rising row.
    _, expected = _reference("rising", "float32")
    values = _values("rising", "float32")
    stride = N_READS * LOOPED_WORKGROUP
    first = max(values[:stride])
    stale = math.fsum(math.exp(value - first) for value in values[:stride])
    stale += math.fsum(math.exp(value - max(values)) for value in values[stride:])
    wrong = [math.exp(value - max(values)) / stale for value in values]
    with pytest.raises(AssertionError):
        _check(
            "float32",
            wrong,
            expected,
            _bounds("looped_softmax_float32", LOOPED_WORKGROUP, "rising"),
        )


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("entry,size", VARIANTS)
def test_softmax_configs_scope_target_options(tmp_path, target, entry, size):
    path = tmp_path / "crosstl.toml"
    path.write_text(_config("out", target, entry, size), encoding="utf-8")
    config = load_project_config(tmp_path, path)
    assert list(config.targets) == [target]
    assert [item.to_json() for item in config.index_range_assertions] == (
        [
            {
                "source": SOURCE,
                "expression": expression,
                "minimum": 0,
                "maximum": SIGNED_MAX,
            }
            for expression in OPENGL_INDEX_RANGE_ASSERTIONS[_kind(entry)]
        ]
        if target == "opengl"
        else []
    )
    text = path.read_text(encoding="utf-8")
    software = target in SOFTWARE_SUBGROUP_TARGETS
    assert ("software_subgroup_width = 32" in text) is software
    assert f'"{entry}" = [{size}, 1, 1]' in text
    assert "[project.subgroup_width_rules]" not in text


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native softmax gate")
    assert root_value and target in TARGETS
    for name in {
        "directx": ["dxc"],
        "opengl": ["glslangValidator", "spirv-val"],
        "metal": ["xcrun"],
    }[target]:
        assert shutil.which(name), f"Required tool is missing: {name}"
    root = Path(root_value).resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert revision == MLX_COMMIT
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert hashlib.sha256((root / SOURCE).read_bytes()).hexdigest() == SOURCE_SHA256
    assert hashlib.sha256((root / HEADER).read_bytes()).hexdigest() == HEADER_SHA256
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("softmax-metal-reference")
    runner = work / "readback"
    _run(
        [
            "xcrun",
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            runner,
        ],
        work,
        "build-runner",
    )
    return runner, _metal_library(
        root / SOURCE, work / "upstream.metallib", root, upstream=True
    )


def _padded(values):
    return list(values) + [math.nan] * INPUT_GUARD_COUNT


def _metal_payload(dtype, values):
    if dtype == "float32":
        return struct.pack(f"<{len(values)}I", *(_bits(dtype, v) for v in values))
    return struct.pack(f"<{len(values)}H", *(_bits(dtype, v) for v in values))


def _upstream_metal(work, reference, entry, name, values):
    runner, library = reference
    dtype = _dtype(entry)
    rows, axis, _ = CASES[name]
    count = rows * axis
    guards = [_from_bits(dtype, GUARD_BITS[dtype])] * GUARD_COUNT
    payloads = [
        _metal_payload(dtype, _padded(values)),
        _metal_payload(dtype, [0.0] * count + guards),
        struct.pack("<i", axis),
    ]
    files = []
    for index, payload in enumerate(payloads):
        path = work / f"{name}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    request = work / f"{name}-upstream-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": [rows, 1, 1],
                "workgroupSize": [_workgroup_size(axis), 1, 1],
                "simdWidth": SIMD_SIZE,
            }
        ),
        encoding="utf-8",
    )
    output = work / f"{name}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, output], work, f"{name}-upstream")
    )
    for index in (0, 2):
        assert (output / f"buffer-{index}.bin").read_bytes() == payloads[index]
    raw = (output / "buffer-1.bin").read_bytes()
    if dtype == "float32":
        words = struct.unpack(f"<{len(raw) // 4}I", raw)
    else:
        words = struct.unpack(f"<{len(raw) // 2}H", raw)
    assert list(words[count:]) == [GUARD_BITS[dtype]] * GUARD_COUNT
    return device, [_from_bits(dtype, word) for word in words[:count]]


def _dispatch(descriptor, package, target, entry, name, values):
    dtype = _dtype(entry)
    rows, axis, _ = CASES[name]
    count = rows * axis
    inputs, observed, output = {}, [], None
    prefix = re.sub(r"_+", "_", entry) + "_"
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        layout = binding["scalarLayout"]
        # DirectX constant members carry the sanitized entry name as a prefix.
        member = layout.get("memberName", binding["name"]).removeprefix(prefix)
        member = member.removesuffix("Buffer").rstrip("_")
        element_type, encoding = _element_layout(binding)
        if member == "in":
            value = _runtime_value(dtype, element_type, _padded(values), encoding)
        elif member == "axis_size":
            value = {"dtype": "int32", "shape": [1], "values": [axis]}
        else:
            assert member == "out", member
            value = _runtime_value(dtype, element_type, [0.0] * count, encoding)
            value["values"] = value["values"][:count] + _guard_values(
                dtype, element_type
            )
            value["shape"] = [count + GUARD_COUNT]
            output = (binding["name"], value["dtype"])
        if member != "axis_size" and encoding is not None:
            assert encoding["logicalElementType"] == dtype, (member, encoding)
        elif member != "axis_size":
            assert element_type in {dtype, "float32", "uint16"}, (member, element_type)
        inputs[binding["name"]] = value
        if member != "out":
            observed.append(binding["name"])
    assert output is not None and len(observed) == 2, (output, observed)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output[0]: dict(inputs[output[0]])},
        {
            "workgroupCount": [rows, 1, 1],
            "workgroupSize": [_workgroup_size(axis), 1, 1],
        },
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output, inputs[output[0]]["values"][count:]


def _assert_lowering(text, target, entry, size):
    assert "unsupported" not in text
    accumulator = _accumulator(entry)
    block = _kind(entry) == "block"
    if target == "metal":
        # softmax_exp keeps softmax.h's fast::exp.
        assert text.count("metal::fast::exp(") == text.count("fast::") == 1
        assert text.count("simd_max(") == 2 and text.count("simd_sum(") == 2
        return
    # OpenGL and DirectX call their default exp.
    assert "fast::" not in text
    if target == "opengl":
        assert "subgroupAdd" not in text and "subgroupMax" not in text
        assert "gl_Subgroup" not in text
        sum_helper = (
            "crossglSoftwareSubgroupSumHalf("
            if accumulator == "float16"
            else "crossglSoftwareSubgroupSumFloat("
        )
        assert text.count(sum_helper) == 3
        assert text.count("crossglSoftwareSubgroupMaxFloat(") == 3
        predicated = block and size > SIMD_SIZE
        assert ("crossglSoftwareSubgroupActive" in text) is predicated
        # Inactive invocations add the exact identity -0.0 to the normalizer.
        assert (
            "float crossglSoftwareSubgroupInput_2 = uintBitsToFloat(0x80000000u);"
            in text
        ) is predicated
        return
    assert "WaveActiveSum" not in text and "WaveActiveMax" not in text
    assert "[WaveSize" not in text
    payload = "float16_t" if accumulator == "float16" else "float"
    assert text.count(f"__crossgl_software_subgroup_sum_{payload}(") == 3
    assert text.count(f"__crossgl_software_subgroup_max_{payload}(") == 3
    if block:
        assert "bool crossglSoftwareSubgroupActive = simd_group_id == 0;" in text
        identity = (
            "asfloat16(uint16_t(0xfc00u))"
            if payload == "float16_t"
            else "asfloat(0xff800000u)"
        )
        assert f"{payload} crossglSoftwareSubgroupInput = {identity};" in text
    else:
        assert "crossglSoftwareSubgroupActive" not in text


@pytest.mark.parametrize("entry,size", VARIANTS)
def test_current_mlx_softmax_native_parity(
    current_mlx, metal_reference, tmp_path, entry, size
):
    root, target = current_mlx
    dtype = _dtype(entry)
    record = _contract()["artifacts"][target][entry][str(size)]
    with tempfile.TemporaryDirectory(prefix=".current-softmax-", dir=root) as directory:
        work = Path(directory)
        try:
            config = work / "crosstl.toml"
            config.write_text(
                _config(f"{work.name}/out", target, entry, size), encoding="utf-8"
            )
            translation = translate_project(
                load_project_config(root, config),
                format_output=False,
                validate=True,
                run_toolchains=False,
            )
            report = translation.to_json()
            translation.write_json(work / "report.json")
            validation = validate_project_report(work / "report.json")
            assert validation["success"] is True, validation
            assert report["diagnostics"] == []
            assert report["summary"]["translatedCount"] == 1
            assert report["summary"]["failedCount"] == 0
            (artifact,) = report["artifacts"]
            assert artifact["sourceHash"] == {
                "algorithm": "sha256",
                "value": SOURCE_SHA256,
            }
            check_generated_artifact(
                artifact, sha256=record["sha256"], size_bytes=record["sizeBytes"]
            )
            (execution,) = artifact["execution"]["entryPoints"]
            assert execution["workgroupSize"] == [size, 1, 1]
            generated = root / artifact["path"]
            _assert_lowering(generated.read_text(encoding="utf-8"), target, entry, size)
            if target == "metal":
                _metal_library(generated, work / "translated.metallib", root)
            else:
                _validate_generated_artifact(generated, work, target)
            descriptor, package = _prepare_native_package(translation, work)
            executor = _executor(target)
            evidence = []
            try:
                for name in _variant_cases(entry, size):
                    values, expected = _reference(name, dtype)
                    bounds = _bounds(entry, size, name)
                    count = len(expected)
                    request, output, guards = _dispatch(
                        descriptor, package, target, entry, name, values
                    )
                    availability = executor.is_available(request)
                    assert availability.available, availability.reason
                    result = executor.run(request)
                    (work / f"{name}-result.json").write_text(
                        json.dumps(
                            {
                                "status": result.status,
                                "outputs": result.outputs,
                                "details": result.details,
                            },
                            indent=2,
                            default=str,
                        ),
                        encoding="utf-8",
                    )
                    assert result.status == "ok"
                    output_name, element_type = output
                    value = result.outputs[output_name]
                    assert value["dtype"] == element_type
                    assert value["values"][count:] == guards
                    actual = _decode(
                        dtype, {**value, "values": value["values"][:count]}
                    )
                    worst = _check(dtype, actual, expected, bounds)
                    observations = result.details["inputObservations"]
                    assert len(observations) == 2
                    assert all(item["unchanged"] for item in observations.values())
                    upstream = None
                    if target == "metal":
                        device, upstream_values = _upstream_metal(
                            work, metal_reference, entry, name, values
                        )
                        upstream = {
                            "device": device,
                            "boundFractionUsed": _check(
                                dtype, upstream_values, expected, bounds
                            ),
                            "bitIdenticalToTranslated": (
                                [_bits(dtype, item) for item in upstream_values]
                                == [_bits(dtype, item) for item in actual]
                            ),
                        }
                    evidence.append(
                        {
                            "case": name,
                            "rows": CASES[name][0],
                            "axisSize": CASES[name][1],
                            "workgroupSize": size,
                            "boundFractionUsed": worst,
                            "largestRelativeBound": (
                                max(
                                    relative
                                    for relative, _ in bounds
                                    if math.isfinite(relative)
                                )
                                if CASES[name][2] != "infinite"
                                else None
                            ),
                            "outputGuardCount": GUARD_COUNT,
                            "upstream": upstream,
                        }
                    )
            finally:
                close = getattr(executor.runtime_adapter.runtime, "close", None)
                if callable(close):
                    close()
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entry": entry,
                        "target": target,
                        "dtype": dtype,
                        "accumulator": _accumulator(entry),
                        "workgroupSize": size,
                        "subgroups": SUBGROUPS[target],
                        "artifactSha256": record["sha256"],
                        "cases": evidence,
                        "indexRangeAssertions": (
                            list(OPENGL_INDEX_RANGE_ASSERTIONS[_kind(entry)])
                            if target == "opengl"
                            else []
                        ),
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if target == "metal":
                # The round trip keeps fast::exp, so the translated kernel
                # reproduces the unchanged upstream kernel bit for bit.
                assert [
                    case["case"]
                    for case in evidence
                    if not case["upstream"]["bitIdenticalToTranslated"]
                ] == []
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)


def test_ci_requires_native_softmax_parity():
    from tests import workflow_text
    from tests.ci_helpers import (
        assert_native_step_prerequisites,
        assert_workflow_triggers,
    )

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = workflow_text.workflow_job_step_section(
        workflow,
        "mlx-metal-porting",
        "Prove current MLX softmax native parity",
    )
    assert_native_step_prerequisites(step, job="mlx-metal-porting")
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/softmax",
        "--junitxml=mlx-current-results/softmax.xml",
        "demos/integrations/mlx/tests/kernels/test_current_softmax.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_softmax.py"
    )
    upload = workflow_text.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Upload current MLX runtime diagnostics"
    )
    assert "if: always()" in upload
    assert "path: mlx-current-results" in upload
    assert workflow.index("Prove current MLX logsumexp native parity") < (
        workflow.index("Prove current MLX softmax native parity")
    )
    assert workflow.index("Prove current MLX softmax native parity") < (
        workflow.index("Upload current MLX runtime diagnostics")
    )
