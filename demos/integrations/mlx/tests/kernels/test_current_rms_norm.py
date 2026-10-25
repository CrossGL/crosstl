"""Translate and execute the pinned MLX RMS normalization kernels everywhere."""

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

SOURCE = "mlx/backend/metal/kernels/rms_norm.metal"
SOURCE_SHA256 = "b2e04e377fdad1d645581f9beeaf9cbb06d1ad32926161e06cbc15240caf12bf"
HEADER = "mlx/backend/metal/kernels/utils.h"
HEADER_SHA256 = "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_RMS_NORM"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/rms_norm.current-tree.translation.json"
)
DTYPES = ("float32", "float16", "bfloat16")
# instantiate_rms's order in rms_norm.metal.
KERNELS = ("rms", "vjp_rms", "rms_looped", "vjp_rms_looped")
ENTRIES = tuple(f"{kernel}{dtype}" for dtype in DTYPES for kernel in KERNELS)
TARGETS = ("metal", "opengl", "directx")
SIMD_SIZE = 32
# mlx/backend/metal/normalization.cpp with defines.h: rows up to
# RMS_LOOPED_LIMIT use the single-row kernels with 32 * ceil(ceil(axis /
# RMS_N_READS) / 32) threads; longer rows use the looped kernels with
# maxTotalThreadsPerThreadgroup (1024 for these pipelines on Apple GPUs).
LOOPED_LIMIT = 4096
N_READS = 4
LOOPED_WORKGROUP = 1024
# RMSNormVJP spreads at most TARGET_GROUPS threadgroups over the rows of a
# single-row VJP; each group accumulates gw over its rows into gw_temp.
TARGET_GROUPS = 512
# One SIMD group, two, nine, and the largest single-row group.
WORKGROUP_SIZES = {
    "rms": (32, 288, 1024),
    "vjp_rms": (32, 64, 288, 1024),
    "rms_looped": (LOOPED_WORKGROUP,),
    "vjp_rms_looped": (LOOPED_WORKGROUP,),
}
# RMSNormVJP specializes has_w (function constant 20) on whether a weight was
# given; without one MLX passes a scalar one with stride 0 and gw_temp is that
# weight's buffer, which the kernel must not write.
HAS_W_CONSTANT_ID = 20
EPS = struct.unpack("<f", struct.pack("<f", 1e-5))[0]
# OpenGL indexes 32-bit storage arrays. These host preconditions bound the
# rebased row offsets and the composed element indices to signed int range.
SIGNED_MAX = 2**31 - 1
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "rms": ("gid * uint64(axis_size) + lid * RMS_N_READS", "x + i", "out_ + i"),
    "rms_looped": (
        "gid * uint64(axis_size) + lid * RMS_N_READS",
        "i + r",
        "x + i + r",
        "r + i",
        "out_ + r + i",
        "x + r + i",
    ),
    "vjp_rms": (
        "x + uint64(row) * axis_size + lid * RMS_N_READS",
        "g + uint64(row) * axis_size + lid * RMS_N_READS",
        "gx + uint64(row) * axis_size + lid * RMS_N_READS",
        "x_in + i",
        "g_in + i",
        "gx_out + i",
        "uint64(gid) * axis_size + lid * RMS_N_READS",
        "gw + i",
    ),
    "vjp_rms_looped": (
        "gid * uint64(axis_size) + lid * RMS_N_READS",
        "i + r",
        "x + i + r",
        "g + i + r",
        "gx + i + r",
        "gw + i + r",
    ),
}
# OpenGL and DirectX lower the SIMD sums to software 32-lane groups, so results
# do not depend on the device's subgroup width.
SOFTWARE_SUBGROUP_TARGETS = ("opengl", "directx")
SUBGROUPS = {
    "metal": "native-simdgroup-32",
    "opengl": "software-subgroup-32",
    "directx": "software-subgroup-32",
}
# name -> (rows, axis size, values, weight). A missing weight is MLX's scalar
# one with stride 0; a strided weight has stride 2.
FORWARD_CASES = {
    "one": (3, 1, "uniform", "vector"),
    "tail": (5, 7, "uniform", "none"),
    "rows": (64, 32, "uniform", "vector"),
    "full": (4, 128, "offset", "strided"),
    "zero": (2, 100, "zero", "vector"),
    "small": (2, 1100, "small", "strided"),
    "simd-groups": (2, 1152, "uniform", "vector"),
    "simd-tail": (3, 1025, "offset", "none"),
    "group": (2, 4096, "uniform", "vector"),
    "group-tail": (2, 3969, "offset", "strided"),
    "two-passes": (2, 4097, "uniform", "vector"),
    "passes": (1, 20000, "uniform", "strided"),
    "looped-offset": (2, 8193, "offset", "none"),
    "looped-small": (1, 5000, "small", "vector"),
}
# name -> (rows, axis size, values, weight with has_w). Every case also runs
# without a weight. More than TARGET_GROUPS rows make each single-row group
# loop over several rows, and the last group's clamped row range is shorter.
VJP_CASES = {
    "one": (3, 1, "uniform", "vector"),
    "tail": (5, 7, "uniform", "vector"),
    "rows": (64, 32, "uniform", "vector"),
    "many-rows": (1025, 7, "uniform", "strided"),
    "full": (4, 128, "offset", "strided"),
    "zero": (2, 100, "zero", "vector"),
    "group-rows": (513, 129, "offset", "vector"),
    "small": (2, 1100, "small", "strided"),
    "simd-groups": (2, 1152, "uniform", "vector"),
    "simd-tail": (3, 1025, "offset", "vector"),
    "group": (2, 4096, "uniform", "vector"),
    "group-tail": (2, 3969, "offset", "strided"),
    "two-passes": (2, 4097, "uniform", "vector"),
    "passes": (1, 20000, "uniform", "strided"),
    "looped-offset": (2, 8193, "offset", "vector"),
    "looped-small": (1, 5000, "small", "strided"),
}
CASES = {"forward": FORWARD_CASES, "vjp": VJP_CASES}
# Each output is checked against a float64 reference with a per-element bound
# on the kernel's own rounding. Everything up to the output conversion runs in
# binary32 (unit roundoff U32): products round once each, the per-thread sums
# and two software or native 32-lane reductions per threadgroup sum are bounded
# by their depth (at most 31 sequential additions per reduction), divisions
# may be 2.5 units in the last place from the exact quotient and the precise
# reciprocal square root 2 units. The forward kernels of the 16-bit types
# round the normalized value and the weighted product in the storage type
# (UNIT, TINY below the normal range). Results must be a correct rounding of a
# value within the bound.
U32 = 2.0**-24
DIVISION_ERROR = 5 * U32
ROOT_ERROR = 4 * U32
SIMD_REDUCTION_DEPTH = 31
UNIT = {"float32": 2.0**-24, "float16": 2.0**-11, "bfloat16": 2.0**-8}
TINY = {"float32": 2.0**-150, "float16": 2.0**-25, "bfloat16": 2.0**-134}
SIGNIFICAND_BITS = {"float32": 24, "float16": 11, "bfloat16": 8}
# frexp exponent of the smallest normal value of each type.
MIN_NORMAL_EXPONENT = {"float32": -125, "float16": -13, "bfloat16": -125}
# Second-order terms of the first-order bound.
BOUND_SAFETY = 1.0 + 2.0**-10
INPUT_GUARD_COUNT = 16


def _dtype(entry):
    # bfloat16 also ends with float16, so match the longest suffix first.
    return next(
        dtype
        for dtype in sorted(DTYPES, key=len, reverse=True)
        if entry.endswith(dtype)
    )


def _kernel(entry):
    return entry[: -len(_dtype(entry))]


def _family(entry):
    return "vjp" if _kernel(entry).startswith("vjp_") else "forward"


def _looped(entry):
    return _kernel(entry).endswith("_looped")


def _workgroup_size(axis):
    """Return MLX's threadgroup size for one row of ``axis`` elements."""
    if axis > LOOPED_LIMIT:
        return LOOPED_WORKGROUP
    threads = -(-axis // N_READS)
    return SIMD_SIZE * -(-threads // SIMD_SIZE)


def _case_kernel(family, axis):
    looped = "_looped" if axis > LOOPED_LIMIT else ""
    return ("vjp_" if family == "vjp" else "") + "rms" + looped


def _rows_per_group(entry, rows):
    """Return the rows each threadgroup of ``entry`` reduces, as MLX does."""
    if _family(entry) == "vjp" and not _looped(entry):
        return -(-rows // TARGET_GROUPS)
    return 1


def _groups(entry, rows):
    return -(-rows // _rows_per_group(entry, rows))


def _weight_modes(entry):
    return (True, False) if _family(entry) == "vjp" else (None,)


VARIANTS = tuple(
    (entry, size, has_w)
    for entry in ENTRIES
    for size in WORKGROUP_SIZES[_kernel(entry)]
    for has_w in _weight_modes(entry)
)


def _variant_key(size, has_w):
    """Name a variant like MLX's pipeline hash: ``_w`` or ``_now`` for VJPs."""
    return str(size) if has_w is None else f"{size}_{'w' if has_w else 'now'}"


def _variant_cases(entry, size):
    family = _family(entry)
    return tuple(
        name
        for name, (_, axis, *_rest) in CASES[family].items()
        if _case_kernel(family, axis) == _kernel(entry)
        and _workgroup_size(axis) == size
    )


def _passes(entry, size, axis):
    if not _looped(entry):
        return 1
    return math.ceil(axis / (N_READS * size))


def _stream(name, salt):
    state = (zlib.crc32(f"{name}:{salt}".encode()) << 1) | 1
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        yield ((state >> 11) / 2**53) * 2.0 - 1.0


def _row_values(family, name, dtype):
    rows, axis, kind = CASES[family][name][:3]
    stream = _stream(f"{family}:{name}", "x")
    values = []
    for _ in range(rows * axis):
        unit = next(stream)
        if kind == "offset":
            # Large values relative to their spread.
            value = 64.0 + 2.0 * unit
        elif kind == "small":
            # The mean square is comparable to eps, so a dropped eps fails.
            value = 2.0**-6 * unit
        elif kind == "zero":
            value = 0.0  # zero mean square: the normalizer is rsqrt(eps)
        else:
            value = 4.0 * unit
        values.append(_quantize(dtype, value))
    return values


def _weights(family, name, dtype, axis, mode):
    """Return the weight buffer and its stride.

    Strided buffers hold NaN between the used elements, so a wrong stride
    poisons the result.
    """
    if mode == "none":
        return [1.0], 0
    stream = _stream(f"{family}:{name}", "w")
    values = [_quantize(dtype, 0.5 + next(stream)) for _ in range(axis)]
    if mode == "vector":
        return values, 1
    assert mode == "strided"
    buffer = []
    for value in values:
        buffer.extend((value, math.nan))
    return buffer, 2


def _inputs(entry, name, has_w=None):
    """Return the logical input buffers and scalars of one case."""
    family, dtype = _family(entry), _dtype(entry)
    rows, axis, _, weight = CASES[family][name]
    x = _row_values(family, name, dtype)
    if family == "vjp" and not has_w:
        weight = "none"
    w, w_stride = _weights(family, name, dtype, axis, weight)
    inputs = {"x": x, "w": w, "w_stride": w_stride, "axis": axis, "rows": rows}
    if family == "vjp":
        stream = _stream(f"{family}:{name}", "g")
        inputs["g"] = [_quantize(dtype, next(stream)) for _ in range(rows * axis)]
        inputs["rows_per_group"] = _rows_per_group(entry, rows)
        inputs["groups"] = _groups(entry, rows)
    return inputs


def _depth(entry, size, axis):
    """Return the longest chain of binary32 additions in a threadgroup sum."""
    return N_READS * _passes(entry, size, axis) + 2 * SIMD_REDUCTION_DEPTH


def _normalizer(entry, size, row):
    """Return rsqrt(mean(x^2) + eps) for one row and its relative error."""
    n = len(row)
    depth = _depth(entry, size, n)
    squares = math.fsum(value * value for value in row)
    # Every term is nonnegative and rounds once, so the sum's error is
    # relative.
    squares_error = (depth + 1) * U32 * squares
    mean = squares / n
    mean_error = squares_error / n * (1 + DIVISION_ERROR) + DIVISION_ERROR * mean
    # mean + eps rounds once; the reciprocal square root halves the relative
    # error of its operand and adds its own.
    relative = mean_error / (mean + EPS) + U32
    return 1.0 / math.sqrt(mean + EPS), relative / 2 + ROOT_ERROR


def _round(dtype, value):
    """Round a float64 value to ``dtype`` with ties to even, in one step."""
    if value == 0 or not math.isfinite(value):
        return value
    exponent = max(math.frexp(value)[1], MIN_NORMAL_EXPONENT[dtype])
    quantum = 2.0 ** (exponent - SIGNIFICAND_BITS[dtype])
    return round(value / quantum) * quantum


def _forward_reference(entry, size, name):
    """Return the inputs, a float64 RMS norm and each output's error bound."""
    dtype = _dtype(entry)
    unit, tiny = UNIT[dtype], TINY[dtype]
    inputs = _inputs(entry, name)
    rows, axis, w = inputs["rows"], inputs["axis"], inputs["w"]
    expected, bounds = [], []
    for row_index in range(rows):
        row = inputs["x"][row_index * axis : (row_index + 1) * axis]
        normalizer, normalizer_error = _normalizer(entry, size, row)
        for index, item in enumerate(row):
            weight = w[inputs["w_stride"] * index]
            normalized = item * normalizer
            normalized_error = abs(normalized) * (normalizer_error + U32)
            # The 16-bit kernels round the normalized value and the product
            # in the storage type; float32 rounds the product.
            conversion = 0.0 if dtype == "float32" else unit * abs(normalized)
            bound = abs(weight) * (normalized_error + conversion + tiny)
            bound += unit * abs(weight * normalized) + tiny
            expected.append(weight * normalized)
            bounds.append(bound * BOUND_SAFETY)
    return inputs, {"out": expected}, {"out": bounds}


def _vjp_reference(entry, size, name, has_w):
    """Return the inputs, float64 gradients and each gradient's error bound.

    gw is MLX's gw_temp: one row per threadgroup, accumulated over its rows.
    """
    inputs = _inputs(entry, name, has_w)
    rows, axis, w, g = inputs["rows"], inputs["axis"], inputs["w"], inputs["g"]
    weights = [w[inputs["w_stride"] * index] for index in range(axis)]
    per_group = inputs["rows_per_group"]
    gx, gx_bounds = [], []
    gw = [[] for _ in range(axis)]
    for row_index in range(rows):
        start = row_index * axis
        row = inputs["x"][start : start + axis]
        gradients = g[start : start + axis]
        normalizer, normalizer_error = _normalizer(entry, size, row)
        depth = _depth(entry, size, axis)
        # sumgwx adds (x * w) * g, two roundings per term.
        terms = [x * wi * gi for x, wi, gi in zip(row, weights, gradients)]
        mean_gwx = math.fsum(terms) / axis
        mean_gwx_error = (depth + 2) * U32 * math.fsum(abs(t) for t in terms)
        mean_gwx_error = mean_gwx_error / axis * (1 + DIVISION_ERROR)
        mean_gwx_error += DIVISION_ERROR * abs(mean_gwx)
        cube = normalizer**3
        cube_error = 3 * normalizer_error + 2 * U32
        for index in range(axis):
            x, gi, wi = row[index], gradients[index], weights[index]
            first = gi * wi * normalizer
            first_error = abs(first) * (2 * U32 + normalizer_error)
            second = x * mean_gwx * cube
            second_error = abs(x * cube) * mean_gwx_error
            second_error += abs(second) * (cube_error + 2 * U32)
            bound = first_error + second_error + U32 * abs(first - second)
            gx.append(first - second)
            gx_bounds.append(bound * BOUND_SAFETY)
            if has_w:
                # (g * x) * normalizer rounds twice before it is accumulated.
                term = gi * x * normalizer
                gw[index].append((term, abs(term) * (2 * U32 + normalizer_error)))
    expected = {"gx": gx}
    bounds = {"gx": gx_bounds}
    if has_w:
        expected["gw"], bounds["gw"] = [], []
        for group in range(inputs["groups"]):
            for index in range(axis):
                terms = gw[index][group * per_group : (group + 1) * per_group]
                total = math.fsum(term for term, _ in terms)
                # Sequential accumulation from zero: the first addition is
                # exact, every later one rounds the running sum.
                bound = math.fsum(error for _, error in terms)
                bound += (len(terms) - 1) * U32 * math.fsum(abs(t) for t, _ in terms)
                expected["gw"].append(total)
                bounds["gw"].append(bound * BOUND_SAFETY)
    return inputs, expected, bounds


def _reference(entry, size, name, has_w=None):
    if _family(entry) == "vjp":
        return _vjp_reference(entry, size, name, has_w)
    return _forward_reference(entry, size, name)


def _allowed(dtype, expected, bound):
    return _round(dtype, expected - bound), _round(dtype, expected + bound)


def _ulp(dtype, value):
    exponent = max(math.frexp(value)[1], MIN_NORMAL_EXPONENT[dtype])
    return 2.0 ** (exponent - SIGNIFICAND_BITS[dtype])


def _check(dtype, actual, expected, bounds):
    """Check results against the reference.

    Return the largest fraction of an output's bound that the results use
    beyond the final rounding's half unit in the last place, so correctly
    rounded results report zero.
    """
    assert len(actual) == len(expected) == len(bounds)
    bad, worst = [], 0.0
    for index, (got, want, bound) in enumerate(zip(actual, expected, bounds)):
        low, high = _allowed(dtype, want, bound)
        if math.isnan(got) or not low <= got <= high:
            bad.append((index, got, want, low, high))
        elif got != want:
            half = max(_ulp(dtype, want), _ulp(dtype, got)) / 2
            excess = max(0.0, abs(got - want) - half)
            if excess:
                worst = max(worst, excess / bound if bound else math.inf)
    assert not bad, bad[:8]
    return worst


def _config(output, target, entry, size, has_w):
    options = ""
    if has_w is not None:
        options += f"""
[project.specialization_constants]
"{HAS_W_CONSTANT_ID}" = {str(has_w).lower()}
"""
    if target == "opengl":
        for expression in OPENGL_INDEX_RANGE_ASSERTIONS[_kernel(entry)]:
            options += f"""
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "{expression}"
minimum = 0
maximum = {SIGNED_MAX}
"""
    options += """
[project.source_options.metal]
max_template_specializations = 64
max_template_materialization_work = 4096
"""
    if target in SOFTWARE_SUBGROUP_TARGETS:
        options += f"""
[project.source_options.metal.target_options.{target}]
software_subgroup_width = {SIMD_SIZE}
"""
    if target == "directx":
        options += 'relative_wave_shuffle_out_of_range = "self"\n'
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
{options}"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _expected_validation_contract():
    return {
        "workgroupSizes": {
            kernel: list(sizes) for kernel, sizes in WORKGROUP_SIZES.items()
        },
        "looped": {"axisSizeAbove": LOOPED_LIMIT, "workgroupSize": LOOPED_WORKGROUP},
        "reads": N_READS,
        "vjpTargetGroups": TARGET_GROUPS,
        "epsilon": EPS,
        "specializationConstants": {
            "has_w": {"id": HAS_W_CONSTANT_ID, "values": [True, False]},
        },
        "subgroups": dict(SUBGROUPS),
        "indexRangeAssertions": {
            "opengl": {
                kernel: [
                    {"expression": expression, "minimum": 0, "maximum": SIGNED_MAX}
                    for expression in expressions
                ]
                for kernel, expressions in OPENGL_INDEX_RANGE_ASSERTIONS.items()
            },
            "metal": {},
            "directx": {},
        },
        "cases": {
            family: {
                name: {
                    "rows": case[0],
                    "axisSize": case[1],
                    "values": case[2],
                    "weight": case[3],
                    "kernel": _case_kernel(family, case[1]),
                    "workgroupSize": _workgroup_size(case[1]),
                    **(
                        {
                            "rowsPerGroup": _rows_per_group(
                                _case_kernel(family, case[1]) + "float32", case[0]
                            )
                        }
                        if family == "vjp"
                        else {}
                    ),
                }
                for name, case in cases.items()
            }
            for family, cases in CASES.items()
        },
        "errorModel": {
            "binary32Unit": U32,
            "divisionError": DIVISION_ERROR,
            "rootError": ROOT_ERROR,
            "simdReductionDepth": SIMD_REDUCTION_DEPTH,
            "unit": dict(UNIT),
            "tiny": dict(TINY),
        },
        "results": "correct-rounding-of-a-value-within-the-error-bound",
        "inputGuardCount": INPUT_GUARD_COUNT,
        "outputGuardCount": GUARD_COUNT,
        "unusedGradientBufferUnchanged": True,
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def _checked_elements(entry):
    family = _family(entry)
    total = 0
    for size in WORKGROUP_SIZES[_kernel(entry)]:
        for name in _variant_cases(entry, size):
            rows, axis = CASES[family][name][:2]
            if family == "forward":
                total += rows * axis
                continue
            # gx with and without a weight, and gw_temp with one.
            total += 2 * rows * axis + _groups(entry, rows) * axis
    return total


def _expected_scope():
    return {
        "coveredEntryCount": len(ENTRIES),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "artifactCountByTarget": len(VARIANTS),
        "checkedElementCountByEntry": {
            entry: _checked_elements(entry) for entry in ENTRIES
        },
        "completeCorpusEntryCount": 17832,
        "gwTempReductionExecuted": False,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_rms_norm_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-rms-norm-contract"
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
                _variant_key(size, has_w)
                for size in WORKGROUP_SIZES[_kernel(entry)]
                for has_w in _weight_modes(entry)
            )
            for record in records.values():
                assert set(record) == {"sha256", "sizeBytes"}
                assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
    # Only DirectX materializes has_w; Metal and OpenGL keep a function or
    # specialization constant, so both weight modes share one artifact.
    for target in ("metal", "opengl"):
        for entry in ENTRIES:
            if _family(entry) == "vjp":
                records = contract["artifacts"][target][entry]
                for size in WORKGROUP_SIZES[_kernel(entry)]:
                    assert records[f"{size}_w"] == records[f"{size}_now"]
    for entry in ENTRIES:
        if _family(entry) == "vjp":
            records = contract["artifacts"]["directx"][entry]
            for size in WORKGROUP_SIZES[_kernel(entry)]:
                assert records[f"{size}_w"] != records[f"{size}_now"]
    assert contract["validationContract"] == _expected_validation_contract()
    assert contract["scope"] == _expected_scope()


def test_rms_norm_entries_follow_mlx_instantiation():
    source = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "rms_norm.current-tree.translation.json" in source
    assert len(ENTRIES) == len(set(ENTRIES)) == 12
    assert ENTRIES[:4] == (
        "rmsfloat32",
        "vjp_rmsfloat32",
        "rms_loopedfloat32",
        "vjp_rms_loopedfloat32",
    )
    assert [_dtype(entry) for entry in ENTRIES[4::4]] == ["float16", "bfloat16"]
    assert len(VARIANTS) == 3 * (3 + 2 * 4 + 1 + 2 * 1)


def test_rms_norm_cases_follow_mlx_dispatch():
    # Every pinned workgroup size has cases, and every case runs on the kernel
    # and threadgroup size that MLX would choose for its row length.
    for entry, size, _ in VARIANTS:
        assert _variant_cases(entry, size), (entry, size)
    for family, cases in CASES.items():
        covered = {
            name
            for entry, size, _ in VARIANTS
            if _family(entry) == family
            for name in _variant_cases(entry, size)
        }
        assert covered == set(cases)
    assert _workgroup_size(1) == _workgroup_size(128) == 32
    assert _workgroup_size(129) == 64
    assert _workgroup_size(4096) == 1024
    assert _case_kernel("forward", 4096) == "rms"
    assert _case_kernel("forward", 4097) == "rms_looped"
    assert _case_kernel("vjp", 4096) == "vjp_rms"
    assert _case_kernel("vjp", 4097) == "vjp_rms_looped"
    # Rows just over the looped limit take two passes, and one takes five.
    passes = {
        _passes("rms_loopedfloat32", LOOPED_WORKGROUP, axis)
        for cases in CASES.values()
        for _, axis, *_ in cases.values()
        if axis > LOOPED_LIMIT
    }
    assert {2, 3, 5} <= passes
    for cases in CASES.values():
        axes = [case[1] for case in cases.values()]
        # Partial reads, a last thread reading one element, and idle threads.
        assert any(axis % N_READS for axis in axes)
        assert any(axis % N_READS == 1 for axis in axes)
        assert any(
            _workgroup_size(axis) * N_READS - axis >= 64
            for axis in axes
            if axis <= LOOPED_LIMIT
        )
    # Single-row VJP groups loop over three rows with a clamped last group of
    # two, and over two rows with a last group of one, on one and two SIMD
    # groups.
    vjp = "vjp_rmsfloat32"
    assert (_rows_per_group(vjp, 1025), _groups(vjp, 1025)) == (3, 342)
    assert (_rows_per_group(vjp, 513), _groups(vjp, 513)) == (2, 257)
    assert _rows_per_group(vjp, 512) == 1
    assert _rows_per_group("vjp_rms_loopedfloat32", 1025) == 1
    assert _workgroup_size(VJP_CASES["group-rows"][1]) == 64
    # Weights with stride 0, 1 and 2 appear.
    assert {case[3] for case in FORWARD_CASES.values()} == {
        "none",
        "vector",
        "strided",
    }
    assert {case[3] for case in VJP_CASES.values()} == {"vector", "strided"}


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_norm_references_cover_edges(dtype):
    for name, (rows, axis, kind, weight) in FORWARD_CASES.items():
        size = _workgroup_size(axis)
        forward = _case_kernel("forward", axis) + dtype
        inputs, expected, _ = _reference(forward, size, name)
        assert len(inputs["x"]) == len(expected["out"]) == rows * axis
        assert all(_quantize(dtype, value) == value for value in inputs["x"])
        assert inputs["w_stride"] == {"none": 0, "vector": 1, "strided": 2}[weight]
        if weight == "strided":
            assert all(math.isnan(value) for value in inputs["w"][1::2])
        if kind == "zero":
            assert all(value == 0 for value in expected["out"])
        if kind == "small":
            # eps changes the normalizer by more than 5%.
            row = inputs["x"][:axis]
            mean = math.fsum(value * value for value in row) / axis
            assert math.sqrt((mean + EPS) / mean) > 1.05
    for name, (rows, axis, kind, weight) in VJP_CASES.items():
        size = _workgroup_size(axis)
        vjp = _case_kernel("vjp", axis) + dtype
        for has_w in (True, False):
            inputs, expected, _ = _reference(vjp, size, name, has_w)
            assert len(expected["gx"]) == rows * axis
            assert ("gw" in expected) is has_w
            if has_w:
                assert len(expected["gw"]) == inputs["groups"] * axis
            else:
                assert inputs["w"] == [1.0] and inputs["w_stride"] == 0
            if kind == "zero":
                # The gradient keeps only the weighted cotangent term.
                assert all(math.isfinite(item) for item in expected["gx"])
                assert all(item == 0 for item in expected.get("gw", []))
            # x . gx = n * sum(x w g) * eps / (mean(x^2) + eps) on every row:
            # the second term removes the weighted cotangent's projection onto
            # x up to eps.
            weights = [inputs["w"][inputs["w_stride"] * index] for index in range(axis)]
            for row in range(rows):
                values = inputs["x"][row * axis : (row + 1) * axis]
                cotangents = inputs["g"][row * axis : (row + 1) * axis]
                gradients = expected["gx"][row * axis : (row + 1) * axis]
                mean = math.fsum(value * value for value in values) / axis
                weighted = math.fsum(
                    a * b * c for a, b, c in zip(values, weights, cotangents)
                )
                want = weighted * EPS / (mean + EPS) / math.sqrt(mean + EPS)
                dot = math.fsum(a * b for a, b in zip(values, gradients))
                scale = math.fsum(abs(a * b) for a, b in zip(values, gradients))
                assert abs(dot - want) <= 1e-9 * (scale + abs(want)) + 1e-300


def _largest_relative_bound(entry, size, name, has_w):
    _, expected, bounds = _reference(entry, size, name, has_w)
    largest = 0.0
    for key, values in expected.items():
        for want, bound in zip(values, bounds[key]):
            if abs(want) > 0.25:
                largest = max(largest, bound / abs(want))
    return largest


def test_rms_norm_bounds_stay_meaningful():
    # Binary32 rows of mixed sign stay within 2^-14 relative; gradients that
    # cancel widen the worst case to 2^-8, and 16-bit outputs add a few units
    # in the last place.
    limits = {"float32": 2.0**-8, "float16": 2.0**-8, "bfloat16": 2.0**-5}
    for entry, size, has_w in VARIANTS:
        family, dtype = _family(entry), _dtype(entry)
        for name in _variant_cases(entry, size):
            limit = limits[dtype]
            if dtype == "float32" and family == "forward":
                limit = 2.0**-14
            assert _largest_relative_bound(entry, size, name, has_w) < limit, (
                entry,
                size,
                name,
            )


def _last_subgroup_elements(size, axis):
    """Return the row elements that the last SIMD group of a group reads."""
    stride = N_READS * size
    first = (size - SIMD_SIZE) * N_READS
    return {
        index
        for start in range(0, axis, stride)
        for index in range(start + first, min(start + stride, axis))
    }


def _mutated_forward(entry, size, name, mutation):
    """Return forward outputs with one defect, rounded to the storage type."""
    dtype = _dtype(entry)
    inputs = _inputs(entry, name)
    rows, axis = inputs["rows"], inputs["axis"]
    dropped = (
        _last_subgroup_elements(size, axis) if mutation == "missing-subgroup" else ()
    )
    outputs = []
    for row_index in range(rows):
        row = inputs["x"][row_index * axis : (row_index + 1) * axis]
        # The threadgroup sum may drop the last SIMD group's contribution.
        kept = [value for index, value in enumerate(row) if index not in dropped]
        mean = math.fsum(value * value for value in kept) / axis
        eps = 0.0 if mutation == "no-eps" else EPS
        normalizer = 1.0 / math.sqrt(mean + eps) if mean + eps else math.inf
        for index, item in enumerate(row):
            stride = 1 if mutation == "unit-stride" else inputs["w_stride"]
            weight = inputs["w"][min(stride * index, len(inputs["w"]) - 1)]
            outputs.append(_round(dtype, weight * item * normalizer))
    return outputs


def _mutated_vjp(entry, size, name, mutation):
    """Return VJP gradients with one defect, rounded to the storage type."""
    dtype = _dtype(entry)
    inputs, expected, _ = _reference(entry, size, name, True)
    rows, axis = inputs["rows"], inputs["axis"]
    weights = [inputs["w"][inputs["w_stride"] * index] for index in range(axis)]
    if mutation == "last-row-gw":
        # gw_temp keeps only the group's last row instead of accumulating.
        per_group = inputs["rows_per_group"]
        outputs = []
        for group in range(inputs["groups"]):
            last = min((group + 1) * per_group, rows) - 1
            row = inputs["x"][last * axis : (last + 1) * axis]
            normalizer = 1.0 / math.sqrt(
                math.fsum(value * value for value in row) / axis + EPS
            )
            for index in range(axis):
                gradient = inputs["g"][last * axis + index]
                outputs.append(_round(dtype, gradient * row[index] * normalizer))
        return "gw", outputs
    assert mutation == "uncorrected"
    outputs = []
    for row_index in range(rows):
        row = inputs["x"][row_index * axis : (row_index + 1) * axis]
        normalizer = 1.0 / math.sqrt(
            math.fsum(value * value for value in row) / axis + EPS
        )
        for index in range(axis):
            gradient = inputs["g"][row_index * axis + index]
            outputs.append(_round(dtype, gradient * weights[index] * normalizer))
    return "gx", outputs


def test_rms_norm_checks_reject_defects():
    for entry, size, has_w in VARIANTS:
        if has_w is not None:
            continue
        dtype = _dtype(entry)
        for name in _variant_cases(entry, size):
            _, expected, bounds = _reference(entry, size, name)
            rounded = [_round(dtype, value) for value in expected["out"]]
            _check(dtype, rounded, expected["out"], bounds["out"])
            # Unwritten zeros and guards fail.
            guard = _from_bits(dtype, GUARD_BITS[dtype])
            with pytest.raises(AssertionError):
                _check(dtype, rounded[:-1] + [guard], expected["out"], bounds["out"])
            kind, weight = FORWARD_CASES[name][2:]
            mutations = []
            if kind == "small":
                mutations.append("no-eps")
            if weight == "strided":
                mutations.append("unit-stride")
            if FORWARD_CASES[name][1] > SIMD_SIZE * N_READS and kind != "zero":
                # In tail cases the last SIMD group may read one element,
                # which 16-bit outputs cannot reveal.
                axis = FORWARD_CASES[name][1]
                dropped = len(_last_subgroup_elements(size, axis))
                if dtype == "float32" or dropped * 64 >= axis:
                    mutations.append("missing-subgroup")
            for mutation in mutations:
                wrong = _mutated_forward(entry, size, name, mutation)
                with pytest.raises(AssertionError):
                    _check(dtype, wrong, expected["out"], bounds["out"])
    # Binary32 bounds leave no room for one 16-bit unit in the last place.
    _, expected, bounds = _reference("rmsfloat32", 32, "rows")
    nudged = [_round("bfloat16", value) for value in expected["out"]]
    with pytest.raises(AssertionError):
        _check("float32", nudged, expected["out"], bounds["out"])
    # VJPs: a gradient without the projection term, swapped outputs, or gw
    # that does not accumulate a group's rows fail.
    for dtype in DTYPES:
        for name in ("rows", "many-rows", "group-rows"):
            axis = VJP_CASES[name][1]
            entry, size = f"vjp_rms{dtype}", _workgroup_size(axis)
            _, expected, bounds = _reference(entry, size, name, True)
            for key in ("gx", "gw"):
                rounded = [_round(dtype, value) for value in expected[key]]
                _check(dtype, rounded, expected[key], bounds[key])
            mutations = ["uncorrected"]
            if name != "rows":
                mutations.append("last-row-gw")
            else:
                swapped = [_round(dtype, value) for value in expected["gw"]]
                with pytest.raises(AssertionError):
                    _check(dtype, swapped, expected["gx"], bounds["gx"])
            for mutation in mutations:
                key, wrong = _mutated_vjp(entry, size, name, mutation)
                with pytest.raises(AssertionError):
                    _check(dtype, wrong, expected[key], bounds[key])


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("entry,size,has_w", VARIANTS)
def test_rms_norm_configs_scope_target_options(tmp_path, target, entry, size, has_w):
    path = tmp_path / "crosstl.toml"
    path.write_text(_config("out", target, entry, size, has_w), encoding="utf-8")
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
            for expression in OPENGL_INDEX_RANGE_ASSERTIONS[_kernel(entry)]
        ]
        if target == "opengl"
        else []
    )
    text = path.read_text(encoding="utf-8")
    software = target in SOFTWARE_SUBGROUP_TARGETS
    assert ("software_subgroup_width = 32" in text) is software
    assert f'"{entry}" = [{size}, 1, 1]' in text
    assert ("[project.specialization_constants]" in text) is (has_w is not None)
    assert "[project.subgroup_width_rules]" not in text


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native RMS norm gate")
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
    work = tmp_path_factory.mktemp("rms-norm-metal-reference")
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


def _guarded(dtype, count):
    return [0.0] * count + [_from_bits(dtype, GUARD_BITS[dtype])] * GUARD_COUNT


def _gradient_count(inputs, has_w):
    # Without a weight MLX binds the scalar weight's buffer as gw_temp.
    return inputs["groups"] * inputs["axis"] if has_w else 1


def _workgroup_count(entry, inputs):
    return inputs["groups"] if _family(entry) == "vjp" else inputs["rows"]


def _metal_payload(dtype, values):
    if dtype == "float32":
        return struct.pack(f"<{len(values)}I", *(_bits(dtype, v) for v in values))
    return struct.pack(f"<{len(values)}H", *(_bits(dtype, v) for v in values))


def _metal_words(dtype, raw):
    if dtype == "float32":
        return struct.unpack(f"<{len(raw) // 4}I", raw)
    return struct.unpack(f"<{len(raw) // 2}H", raw)


def _upstream_metal(work, reference, entry, name, has_w):
    runner, library = reference
    dtype = _dtype(entry)
    inputs = _inputs(entry, name, has_w)
    rows, axis = inputs["rows"], inputs["axis"]
    count = rows * axis
    scalars = [
        struct.pack("<f", EPS),
        struct.pack("<I", axis),
        struct.pack("<I", inputs["w_stride"]),
    ]
    if _family(entry) == "forward":
        payloads = [
            _metal_payload(dtype, _padded(inputs["x"])),
            _metal_payload(dtype, _padded(inputs["w"])),
            _metal_payload(dtype, _guarded(dtype, count)),
            *scalars,
        ]
        outputs = {"out": (2, count)}
    else:
        payloads = [
            _metal_payload(dtype, _padded(inputs["x"])),
            _metal_payload(dtype, _padded(inputs["w"])),
            _metal_payload(dtype, _padded(inputs["g"])),
            _metal_payload(dtype, _guarded(dtype, count)),
            _metal_payload(dtype, _guarded(dtype, _gradient_count(inputs, has_w))),
            *scalars,
        ]
        if not _looped(entry):
            payloads.append(struct.pack("<I", rows))
            payloads.append(struct.pack("<I", inputs["rows_per_group"]))
        outputs = {
            "gx": (3, count),
            "gw": (4, _gradient_count(inputs, True) if has_w else 0),
        }
    files = []
    for index, payload in enumerate(payloads):
        path = work / f"{name}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    request = {
        "buffers": files,
        "workgroupCount": [_workgroup_count(entry, inputs), 1, 1],
        "workgroupSize": [_workgroup_size(axis), 1, 1],
        "simdWidth": SIMD_SIZE,
    }
    if has_w is not None:
        request["boolConstants"] = [{"index": HAS_W_CONSTANT_ID, "value": has_w}]
    request_path = work / f"{name}-upstream-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output = work / f"{name}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request_path, output], work, f"{name}-upstream")
    )
    written = {index for index, _ in outputs.values()}
    for index, payload in enumerate(payloads):
        if index not in written:
            assert (output / f"buffer-{index}.bin").read_bytes() == payload
    results = {}
    for key, (index, written_count) in outputs.items():
        raw = (output / f"buffer-{index}.bin").read_bytes()
        if written_count == 0:
            assert raw == payloads[index]  # gw stays untouched without a weight
            continue
        words = _metal_words(dtype, raw)
        assert list(words[written_count:]) == [GUARD_BITS[dtype]] * GUARD_COUNT
        results[key] = [_from_bits(dtype, word) for word in words[:written_count]]
    return device, results


def _member(binding, entry):
    layout = binding["scalarLayout"]
    # DirectX constant members carry the sanitized entry name as a prefix.
    prefix = re.sub(r"_+", "_", entry) + "_"
    member = layout.get("memberName", binding["name"]).removeprefix(prefix)
    return member.removesuffix("Buffer").rstrip("_")


def _scalar(element_type, encoding, value):
    if isinstance(value, float):
        return _runtime_value("float32", element_type, [value], encoding)
    assert element_type in {"uint32", "int32"}, element_type
    return {"dtype": element_type, "shape": [1], "values": [value]}


def _observed_input_count(entry):
    if _family(entry) == "forward":
        return 5  # x, w, eps, axis_size, w_stride
    # x, w, g, eps, axis_size, w_stride, and n_rows and rows_per_group for the
    # single-row kernel.
    return 6 if _looped(entry) else 8


def _dispatch(descriptor, package, target, entry, name, has_w):
    dtype = _dtype(entry)
    inputs = _inputs(entry, name, has_w)
    rows, axis = inputs["rows"], inputs["axis"]
    count = rows * axis
    scalars = {
        "eps": EPS,
        "axis_size": axis,
        "w_stride": inputs["w_stride"],
        "n_rows": rows,
        "rows_per_group": inputs.get("rows_per_group"),
    }
    counts = {"out": count, "gx": count}
    if _family(entry) == "vjp":
        counts["gw"] = _gradient_count(inputs, has_w)
    values, observed, outputs = {}, [], {}
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        member = _member(binding, entry)
        element_type, encoding = _element_layout(binding)
        if member in scalars:
            assert scalars[member] is not None, member
            value = _scalar(element_type, encoding, scalars[member])
        elif member in counts:
            written = counts[member]
            value = _runtime_value(dtype, element_type, [0.0] * written, encoding)
            value["values"] = value["values"][:written] + _guard_values(
                dtype, element_type
            )
            value["shape"] = [written + GUARD_COUNT]
            outputs[member] = (binding["name"], value["dtype"], written)
        else:
            assert member in {"x", "w", "g"}, member
            value = _runtime_value(
                dtype, element_type, _padded(inputs[member]), encoding
            )
        if member not in scalars and encoding is not None:
            assert encoding["logicalElementType"] == dtype, (member, encoding)
        elif member not in scalars:
            assert element_type in {dtype, "float32", "uint16"}, (member, element_type)
        values[binding["name"]] = value
        if member not in counts:
            observed.append(binding["name"])
    family = _family(entry)
    expected_outputs = {"out"} if family == "forward" else {"gx", "gw"}
    assert set(outputs) == expected_outputs, outputs
    assert len(observed) == _observed_input_count(entry), observed
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        values,
        {name_: dict(values[name_]) for name_, _, _ in outputs.values()},
        {
            "workgroupCount": [_workgroup_count(entry, inputs), 1, 1],
            "workgroupSize": [_workgroup_size(axis), 1, 1],
        },
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    guards = {
        member: values[binding]["values"][written:]
        for member, (binding, _, written) in outputs.items()
    }
    return request, outputs, guards


def _assert_lowering(text, target, entry, size, has_w):
    assert "unsupported" not in text
    vjp = _family(entry) == "vjp"
    calls = 4 if vjp else 2
    if target == "metal":
        assert text.count("simd_sum(") == calls
        # The round trip keeps the precise root as a correctly rounded helper.
        assert "__crossgl_metal_precise_rsqrt_float(" in text
        if vjp:
            assert "constant bool has_w [[function_constant(20)]];" in text
        return
    if target == "opengl":
        assert "subgroupAdd" not in text and "gl_Subgroup" not in text
        assert text.count("crossglSoftwareSubgroupSumFloat(") == calls + 1
        assert "crossgl_metal_precise_rsqrt_float(" in text
        if vjp:
            # Specialized when the pipeline is created, as on Metal.
            assert "layout(constant_id = 20) const bool has_w = false;" in text
    else:
        assert "WaveActiveSum" not in text and "[WaveSize" not in text
        assert text.count("__crossgl_software_subgroup_sum_float(") == calls + 1
        assert "__crossgl_metal_precise_rsqrt_float(" in text
        if vjp:
            assert f"static const bool has_w = {str(has_w).lower()};" in text
    if vjp and size > SIMD_SIZE:
        # Both guarded SIMD-group sums of the VJP run predicated in every
        # invocation, after the first result is assigned under the guard.
        lines = [line.strip() for line in text.splitlines()]
        first = lines.index("sumx2 = crossglSoftwareSubgroupResult;")
        assert lines[first + 1].startswith("crossglSoftwareSubgroupInput")
        assert any(
            re.fullmatch(r"sumgwx = crossglSoftwareSubgroupResult_\d*;", line)
            for line in lines
        )


@pytest.mark.parametrize("entry,size,has_w", VARIANTS)
def test_current_mlx_rms_norm_native_parity(
    current_mlx, metal_reference, tmp_path, entry, size, has_w
):
    root, target = current_mlx
    dtype = _dtype(entry)
    key = _variant_key(size, has_w)
    record = _contract()["artifacts"][target][entry][key]
    with tempfile.TemporaryDirectory(prefix=".current-rms-norm-", dir=root) as path:
        work = Path(path)
        try:
            config = work / "crosstl.toml"
            config.write_text(
                _config(f"{work.name}/out", target, entry, size, has_w),
                encoding="utf-8",
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
            _assert_lowering(
                generated.read_text(encoding="utf-8"), target, entry, size, has_w
            )
            if target == "metal":
                _metal_library(generated, work / "translated.metallib", root)
            else:
                _validate_generated_artifact(generated, work, target)
            descriptor, package = _prepare_native_package(translation, work)
            if has_w is not None:
                (constant,) = descriptor["specializationConstants"]
                assert constant["id"] == HAS_W_CONSTANT_ID
                assert constant["value"] is has_w
            executor = _executor(target)
            evidence = []
            try:
                for name in _variant_cases(entry, size):
                    _, expected, bounds = _reference(entry, size, name, has_w)
                    request, outputs, guards = _dispatch(
                        descriptor, package, target, entry, name, has_w
                    )
                    availability = executor.is_available(request)
                    assert availability.available, availability.reason
                    result = executor.run(request)
                    assert result.status == "ok", result.status
                    used, actual = {}, {}
                    for member, (binding, element_type, written) in outputs.items():
                        value = result.outputs[binding]
                        assert value["dtype"] == element_type
                        assert value["values"][written:] == guards[member]
                        decoded = _decode(
                            dtype, {**value, "values": value["values"][:written]}
                        )
                        if member in expected:
                            actual[member] = decoded
                            used[member] = _check(
                                dtype, decoded, expected[member], bounds[member]
                            )
                        else:
                            # Without a weight the kernel must not write gw.
                            assert decoded == [0.0] * written
                    observations = result.details["inputObservations"]
                    assert len(observations) == _observed_input_count(entry)
                    assert all(item["unchanged"] for item in observations.values())
                    upstream = None
                    if target == "metal":
                        device, upstream_values = _upstream_metal(
                            work, metal_reference, entry, name, has_w
                        )
                        upstream = {
                            "device": device,
                            "boundFractionUsed": {
                                member: _check(
                                    dtype,
                                    upstream_values[member],
                                    expected[member],
                                    bounds[member],
                                )
                                for member in expected
                            },
                            "bitIdenticalToTranslated": all(
                                [_bits(dtype, item) for item in upstream_values[member]]
                                == [_bits(dtype, item) for item in actual[member]]
                                for member in expected
                            ),
                        }
                    rows, axis = CASES[_family(entry)][name][:2]
                    evidence.append(
                        {
                            "case": name,
                            "rows": rows,
                            "axisSize": axis,
                            "workgroupSize": size,
                            "workgroupCount": _workgroup_count(
                                entry, _inputs(entry, name, has_w)
                            ),
                            "boundFractionUsed": used,
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
                        "workgroupSize": size,
                        "hasWeight": has_w,
                        "subgroups": SUBGROUPS[target],
                        "artifactSha256": record["sha256"],
                        "cases": evidence,
                        "indexRangeAssertions": (
                            list(OPENGL_INDEX_RANGE_ASSERTIONS[_kernel(entry)])
                            if target == "opengl"
                            else []
                        ),
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)


def test_ci_requires_native_rms_norm_parity():
    from tests import workflow_text
    from tests.ci_helpers import (
        assert_native_step_prerequisites,
        assert_workflow_triggers,
    )

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = workflow_text.workflow_job_step_section(
        workflow,
        "mlx-metal-porting",
        "Prove current MLX RMS norm native parity",
    )
    assert_native_step_prerequisites(step, job="mlx-metal-porting")
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/rms-norm",
        "--junitxml=mlx-current-results/rms-norm.xml",
        "demos/integrations/mlx/tests/kernels/test_current_rms_norm.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_rms_norm.py"
    )
    upload = workflow_text.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Upload current MLX runtime diagnostics"
    )
    assert "if: always()" in upload
    assert "path: mlx-current-results" in upload
    assert workflow.index("Prove current MLX layer norm native parity") < (
        workflow.index("Prove current MLX RMS norm native parity")
    )
    assert workflow.index("Prove current MLX RMS norm native parity") < (
        workflow.index("Upload current MLX runtime diagnostics")
    )
