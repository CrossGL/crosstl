"""Translate and execute the pinned MLX layer normalization kernels everywhere."""

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

SOURCE = "mlx/backend/metal/kernels/layer_norm.metal"
SOURCE_SHA256 = "2d243f5abea7353929f9bc838ceb5a98e52a452dfc29609ad4d5974447ea689f"
HEADER = "mlx/backend/metal/kernels/utils.h"
HEADER_SHA256 = "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_LAYER_NORM"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/layer_norm.current-tree.translation.json"
)
DTYPES = ("float32", "float16", "bfloat16")
# instantiate_layer_norm's order in layer_norm.metal.
KERNELS = ("layer_norm", "vjp_layer_norm", "layer_norm_looped", "vjp_layer_norm_looped")
ENTRIES = tuple(f"{kernel}{dtype}" for dtype in DTYPES for kernel in KERNELS)
TARGETS = ("metal", "opengl", "directx")
SIMD_SIZE = 32
# mlx/backend/metal/normalization.cpp: rows up to the looped limit use the
# single-row kernel, eight reads per thread and 32 * ceil(ceil(axis / 8) / 32)
# threads; longer rows use the looped kernel, four reads per thread and
# maxTotalThreadsPerThreadgroup (1024 for these pipelines on Apple GPUs).
LOOPED_LIMIT = {"forward": 6656, "vjp": 8192}
N_READS = {"single": 8, "looped": 4}
LOOPED_WORKGROUP = 1024
# One SIMD group, nine, and the largest single-row group of each family.
WORKGROUP_SIZES = {
    "layer_norm": (32, 288, 832),
    "vjp_layer_norm": (32, 288, 1024),
    "layer_norm_looped": (LOOPED_WORKGROUP,),
    "vjp_layer_norm_looped": (LOOPED_WORKGROUP,),
}
# LayerNormVJP specializes has_w (function constant 20) on whether a weight was
# given; without one MLX passes a scalar one with stride 0 and gw is unused.
HAS_W_CONSTANT_ID = 20
EPS = struct.unpack("<f", struct.pack("<f", 1e-5))[0]
# OpenGL indexes 32-bit storage arrays. These host preconditions bound the
# rebased row offsets and the composed element indices to signed int range.
SIGNED_MAX = 2**31 - 1
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "layer_norm": ("gid * uint64(axis_size) + lid * 8", "x + i", "out_ + i"),
    "layer_norm_looped": (
        "gid * uint64(axis_size) + lid * 4",
        "x + i + r",
        "x + r + i",
        "out_ + r + i",
        "i + r",
        "r + i",
    ),
    "vjp_layer_norm": (
        "gid * uint64(axis_size) + lid * 8",
        "x + i",
        "g + i",
        "gx + i",
        "gw + i",
    ),
    "vjp_layer_norm_looped": (
        "gid * uint64(axis_size) + lid * 4",
        "x + i + r",
        "g + i + r",
        "gx + i + r",
        "gw + i + r",
        "i + r",
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
# name -> (rows, axis size, values, weight, bias). A missing weight or bias is
# MLX's scalar one or zero with stride 0; a strided weight has stride 2.
FORWARD_CASES = {
    "one": (3, 1, "uniform", "vector", "vector"),
    "tail": (5, 7, "uniform", "none", "none"),
    "rows": (64, 32, "uniform", "vector", "none"),
    "full": (4, 256, "offset", "strided", "vector"),
    "constant": (2, 100, "constant", "vector", "vector"),
    "simd-groups": (2, 2304, "uniform", "vector", "vector"),
    "simd-tail": (3, 2049, "offset", "none", "vector"),
    "small": (2, 2100, "small", "strided", "none"),
    "group": (2, 6656, "uniform", "vector", "vector"),
    "group-tail": (2, 6403, "offset", "vector", "none"),
    "two-passes": (2, 6657, "uniform", "vector", "vector"),
    "passes": (1, 20000, "uniform", "strided", "vector"),
    "looped-offset": (2, 8195, "offset", "none", "none"),
    "looped-constant": (1, 7000, "constant", "vector", "vector"),
}
# name -> (rows, axis size, values, weight with has_w). Every case also runs
# without a weight.
VJP_CASES = {
    "one": (3, 1, "uniform", "vector"),
    "tail": (5, 7, "uniform", "vector"),
    "rows": (64, 32, "uniform", "vector"),
    "full": (4, 256, "offset", "strided"),
    "constant": (2, 100, "constant", "vector"),
    "simd-groups": (2, 2304, "uniform", "vector"),
    "simd-tail": (3, 2049, "offset", "vector"),
    "small": (2, 2100, "small", "strided"),
    "group": (2, 8192, "uniform", "vector"),
    "group-tail": (2, 7937, "offset", "strided"),
    "two-passes": (2, 8193, "uniform", "vector"),
    "passes": (1, 20000, "uniform", "strided"),
    "looped-offset": (2, 12289, "offset", "vector"),
    "looped-constant": (1, 9000, "constant", "vector"),
}
CASES = {"forward": FORWARD_CASES, "vjp": VJP_CASES}
# Each output is checked against a float64 reference with a per-element bound
# on the kernel's own rounding. Everything up to the output conversion runs in
# binary32 (unit roundoff U32): the per-thread sums and two software or native
# 32-lane reductions per threadgroup sum are bounded by their depth (at most
# 31 sequential additions per reduction), divisions may be 2.5 units in the
# last place from the exact quotient and the precise square roots 2 units. The
# output arithmetic of the 16-bit kernels rounds in the storage type (UNIT,
# TINY below the normal range). Results must be a correct rounding of a value
# within the bound.
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


def _workgroup_size(family, axis):
    """Return MLX's threadgroup size for one row of ``axis`` elements."""
    if axis > LOOPED_LIMIT[family]:
        return LOOPED_WORKGROUP
    threads = -(-axis // N_READS["single"])
    return SIMD_SIZE * -(-threads // SIMD_SIZE)


def _case_kernel(family, axis):
    looped = "_looped" if axis > LOOPED_LIMIT[family] else ""
    return ("vjp_" if family == "vjp" else "") + "layer_norm" + looped


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
        and _workgroup_size(family, axis) == size
    )


def _passes(entry, size, axis):
    if not _looped(entry):
        return 1
    return math.ceil(axis / (N_READS["looped"] * size))


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
            # A large mean relative to the spread stresses x - mean.
            value = 64.0 + 2.0 * unit
        elif kind == "small":
            # The variance is comparable to eps, so a dropped eps fails.
            value = 2.0**-6 * unit
        elif kind == "constant":
            value = 1.5  # zero variance: the normalizer is rsqrt(eps)
        else:
            value = 4.0 * unit
        values.append(_quantize(dtype, value))
    return values


def _strided(family, name, dtype, salt, axis, mode, scale, offset, missing):
    """Return a per-element parameter buffer and its stride.

    Strided buffers hold NaN between the used elements, so a wrong stride
    poisons the result.
    """
    if mode == "none":
        return [missing], 0
    stream = _stream(f"{family}:{name}", salt)
    values = [_quantize(dtype, offset + scale * next(stream)) for _ in range(axis)]
    if mode == "vector":
        return values, 1
    assert mode == "strided"
    buffer = []
    for value in values:
        buffer.extend((value, math.nan))
    return buffer, 2


def _inputs(entry, name, has_w=None):
    """Return the logical input buffers and strides of one case."""
    family, dtype = _family(entry), _dtype(entry)
    case = CASES[family][name]
    rows, axis = case[:2]
    x = _row_values(family, name, dtype)
    weight = case[3] if family == "forward" or has_w else "none"
    w, w_stride = _strided(family, name, dtype, "w", axis, weight, 1.0, 0.5, 1.0)
    inputs = {"x": x, "w": w, "w_stride": w_stride, "axis": axis, "rows": rows}
    if family == "forward":
        b, b_stride = _strided(family, name, dtype, "b", axis, case[4], 0.25, 0.0, 0.0)
        inputs.update(b=b, b_stride=b_stride)
    else:
        stream = _stream(f"{family}:{name}", "g")
        inputs["g"] = [_quantize(dtype, next(stream)) for _ in range(rows * axis)]
    return inputs


def _row_statistics(row):
    n = len(row)
    mean = math.fsum(row) / n
    centered = [value - mean for value in row]
    variance = math.fsum(item * item for item in centered) / n
    return mean, centered, variance


def _depth(entry, size, axis):
    """Return the longest chain of binary32 additions in a threadgroup sum."""
    reads = N_READS["looped" if _looped(entry) else "single"]
    return reads * _passes(entry, size, axis) + 2 * SIMD_REDUCTION_DEPTH


def _statistics_error(entry, size, row, mean, centered, variance):
    """Bound the binary32 mean, centered values and normalizer of one row.

    Return the absolute error of the mean, the absolute error of each centered
    value, the variance's absolute error and the normalizer's relative error.
    """
    n = len(row)
    depth = _depth(entry, size, n)
    absolute = math.fsum(abs(value) for value in row)
    mean_error = depth * U32 * absolute / n * (1 + DIVISION_ERROR)
    mean_error += DIVISION_ERROR * abs(mean)
    center_errors = [mean_error + U32 * (abs(item) + mean_error) for item in centered]
    squares = math.fsum(item * item for item in centered)
    # The mean's error cancels to first order in the sum of squares because
    # the centered values sum to zero; it remains as a second-order term.
    largest = max(abs(item) for item in centered)
    squares_error = (
        2 * U32 * (squares + mean_error * math.fsum(abs(item) for item in centered))
    )
    squares_error += n * (mean_error + U32 * largest) ** 2
    squares_error += (depth + 1) * U32 * (squares + squares_error)
    variance_error = squares_error / n * (1 + DIVISION_ERROR)
    variance_error += DIVISION_ERROR * variance
    # var + eps rounds once; rsqrt and 1 / x then sqrt are within ROOT_ERROR
    # and DIVISION_ERROR plus half the relative error of their operand.
    relative = variance_error / (variance + EPS) + U32
    return mean_error, center_errors, variance_error, relative


def _round(dtype, value):
    """Round a float64 value to ``dtype`` with ties to even, in one step."""
    if value == 0 or not math.isfinite(value):
        return value
    exponent = max(math.frexp(value)[1], MIN_NORMAL_EXPONENT[dtype])
    quantum = 2.0 ** (exponent - SIGNIFICAND_BITS[dtype])
    return round(value / quantum) * quantum


def _forward_reference(entry, size, name):
    """Return the inputs, a float64 layer norm and each output's error bound."""
    dtype = _dtype(entry)
    unit, tiny = UNIT[dtype], TINY[dtype]
    inputs = _inputs(entry, name)
    rows, axis = inputs["rows"], inputs["axis"]
    w, b = inputs["w"], inputs["b"]
    expected, bounds = [], []
    for row_index in range(rows):
        row = inputs["x"][row_index * axis : (row_index + 1) * axis]
        mean, centered, variance = _row_statistics(row)
        _, center_errors, _, relative = _statistics_error(
            entry, size, row, mean, centered, variance
        )
        normalizer = 1.0 / math.sqrt(variance + EPS)
        normalizer_error = relative / 2 + ROOT_ERROR
        for index, (item, error) in enumerate(zip(centered, center_errors)):
            weight = w[inputs["w_stride"] * index]
            bias = b[inputs["b_stride"] * index]
            normalized = item * normalizer
            normalized_error = normalizer * error + abs(normalized) * (
                normalizer_error + U32
            )
            # The 16-bit kernels round the normalized value, the product and
            # the sum in the storage type; float32 rounds the product.
            conversion = 0.0 if dtype == "float32" else unit * abs(normalized)
            bound = abs(weight) * (normalized_error + conversion + tiny)
            bound += unit * abs(weight * normalized) + tiny
            expected.append(weight * normalized + bias)
            bounds.append(bound * BOUND_SAFETY)
    return inputs, {"out": expected}, {"out": bounds}


def _vjp_reference(entry, size, name, has_w):
    """Return the inputs, float64 gradients and each gradient's error bound."""
    dtype = _dtype(entry)
    narrow = 0.0 if dtype == "float32" else U32
    inputs = _inputs(entry, name, has_w)
    rows, axis = inputs["rows"], inputs["axis"]
    w, g = inputs["w"], inputs["g"]
    gx, gw, gx_bounds, gw_bounds = [], [], [], []
    for row_index in range(rows):
        start = row_index * axis
        row = inputs["x"][start : start + axis]
        gradients = g[start : start + axis]
        weights = [w[inputs["w_stride"] * index] for index in range(axis)]
        mean, centered, variance = _row_statistics(row)
        _, center_errors, _, relative = _statistics_error(
            entry, size, row, mean, centered, variance
        )
        depth = _depth(entry, size, axis)
        products = [weight * gradient for weight, gradient in zip(weights, gradients)]
        mean_wg = math.fsum(products) / axis
        mean_wg_error = (depth + 1) * U32 * math.fsum(abs(p) for p in products) / axis
        mean_wg_error = mean_wg_error * (1 + DIVISION_ERROR)
        mean_wg_error += DIVISION_ERROR * abs(mean_wg)
        mean_wgxc = math.fsum(p * c for p, c in zip(products, centered)) / axis
        mean_wgxc_error = math.fsum(
            abs(p) * error + (depth + 2) * U32 * abs(p * c)
            for p, c, error in zip(products, centered, center_errors)
        )
        mean_wgxc_error = mean_wgxc_error / axis * (1 + DIVISION_ERROR)
        mean_wgxc_error += DIVISION_ERROR * abs(mean_wgxc)
        inverse = 1.0 / (variance + EPS)
        inverse_error = relative + DIVISION_ERROR
        normalizer = math.sqrt(inverse)
        normalizer_error = inverse_error / 2 + ROOT_ERROR
        for index in range(axis):
            product, item = products[index], centered[index]
            normalized = item * normalizer
            normalized_error = normalizer * center_errors[index] + abs(normalized) * (
                normalizer_error + U32
            )
            first = normalizer * (product - mean_wg)
            inner_error = U32 * abs(product) + mean_wg_error
            inner_error += U32 * abs(product - mean_wg)
            first_error = normalizer * inner_error
            first_error += abs(first) * (normalizer_error + U32)
            second = normalized * mean_wgxc * inverse
            second_error = abs(mean_wgxc * inverse) * normalized_error
            second_error += abs(normalized * inverse) * mean_wgxc_error
            second_error += abs(second) * (inverse_error + 2 * U32)
            bound = first_error + second_error + narrow * (abs(first) + abs(second))
            gx.append(first - second)
            gx_bounds.append(bound * BOUND_SAFETY)
            if has_w:
                gradient = gradients[index]
                gw.append(gradient * normalized)
                gw_bound = abs(gradient) * normalized_error
                gw_bound += narrow * abs(gradient * normalized)
                gw_bounds.append(gw_bound * BOUND_SAFETY)
    expected = {"gx": gx}
    bounds = {"gx": gx_bounds}
    if has_w:
        expected["gw"], bounds["gw"] = gw, gw_bounds
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
        "looped": {
            "axisSizeAbove": dict(LOOPED_LIMIT),
            "workgroupSize": LOOPED_WORKGROUP,
        },
        "reads": dict(N_READS),
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
                    **({"bias": case[4]} if family == "forward" else {}),
                    "kernel": _case_kernel(family, case[1]),
                    "workgroupSize": _workgroup_size(family, case[1]),
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
            # gx and, with a weight, gw for VJPs; out for the forward kernels.
            total += rows * axis * (3 if family == "vjp" else 1)
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
        "idleThreadPaddingStores": "translated-as-written",
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_layer_norm_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-layer-norm-contract"
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


def test_layer_norm_entries_follow_mlx_instantiation():
    source = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "layer_norm.current-tree.translation.json" in source
    assert len(ENTRIES) == len(set(ENTRIES)) == 12
    assert ENTRIES[:4] == (
        "layer_normfloat32",
        "vjp_layer_normfloat32",
        "layer_norm_loopedfloat32",
        "vjp_layer_norm_loopedfloat32",
    )
    assert [_dtype(entry) for entry in ENTRIES[4::4]] == ["float16", "bfloat16"]
    assert len(VARIANTS) == 3 * (3 + 1 + 2 * 3 + 2 * 1)


def test_layer_norm_cases_follow_mlx_dispatch():
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
    assert _workgroup_size("forward", 1) == _workgroup_size("forward", 256) == 32
    assert _workgroup_size("forward", 257) == 64
    assert _workgroup_size("forward", 6656) == 832
    assert _workgroup_size("vjp", 8192) == 1024
    assert _case_kernel("forward", 6656) == "layer_norm"
    assert _case_kernel("forward", 6657) == "layer_norm_looped"
    assert _case_kernel("vjp", 8192) == "vjp_layer_norm"
    assert _case_kernel("vjp", 8193) == "vjp_layer_norm_looped"
    # Rows just over each looped limit take the fewest passes (two forward,
    # three backward), and one takes five.
    for family, cases in CASES.items():
        looped = f"{'vjp_' if family == 'vjp' else ''}layer_norm_looped"
        passes = {
            _passes(looped + "float32", LOOPED_WORKGROUP, axis)
            for _, axis, *_ in cases.values()
            if axis > LOOPED_LIMIT[family]
        }
        fewest = math.ceil(
            (LOOPED_LIMIT[family] + 1) / (N_READS["looped"] * LOOPED_WORKGROUP)
        )
        assert min(passes) == fewest and {fewest + 1, 5} <= passes
        axes = [case[1] for case in cases.values()]
        # Partial reads, a last thread reading one element, and idle threads
        # whose padding loop starts below zero all appear.
        assert any(axis % N_READS["single"] for axis in axes)
        assert any(axis % N_READS["single"] == 1 for axis in axes)
        assert any(
            _workgroup_size(family, axis) * N_READS["single"] - axis >= 64
            for axis in axes
            if axis <= LOOPED_LIMIT[family]
        )
    # Weights with stride 0, 1 and 2 and biases with stride 0 and 1 appear.
    assert {case[3] for case in FORWARD_CASES.values()} == {
        "none",
        "vector",
        "strided",
    }
    assert {case[4] for case in FORWARD_CASES.values()} == {"none", "vector"}
    assert {case[3] for case in VJP_CASES.values()} == {"vector", "strided"}


@pytest.mark.parametrize("dtype", DTYPES)
def test_layer_norm_references_cover_edges(dtype):
    for name, (rows, axis, kind, weight, bias) in FORWARD_CASES.items():
        size = _workgroup_size("forward", axis)
        forward = _case_kernel("forward", axis) + dtype
        inputs, expected, _ = _reference(forward, size, name)
        assert len(inputs["x"]) == len(expected["out"]) == rows * axis
        assert all(_quantize(dtype, value) == value for value in inputs["x"])
        assert inputs["w_stride"] == {"none": 0, "vector": 1, "strided": 2}[weight]
        assert inputs["b_stride"] == {"none": 0, "vector": 1}[bias]
        if weight == "strided":
            assert all(math.isnan(value) for value in inputs["w"][1::2])
        if kind == "constant" or axis == 1:
            # Zero variance: every output is the bias.
            assert expected["out"] == [
                inputs["b"][inputs["b_stride"] * (index % axis)]
                for index in range(rows * axis)
            ]
        elif weight == "none" and bias == "none":
            for row in range(rows):
                values = expected["out"][row * axis : (row + 1) * axis]
                assert abs(math.fsum(values)) < 1e-9 * axis
        if kind == "small":
            # eps changes the normalizer by more than 5%.
            _, _, variance = _row_statistics(inputs["x"][:axis])
            assert math.sqrt((variance + EPS) / variance) > 1.05
    for name, (rows, axis, kind, weight) in VJP_CASES.items():
        size = _workgroup_size("vjp", axis)
        vjp = _case_kernel("vjp", axis) + dtype
        for has_w in (True, False):
            inputs, expected, _ = _reference(vjp, size, name, has_w)
            assert len(expected["gx"]) == rows * axis
            assert ("gw" in expected) is has_w
            if not has_w:
                assert inputs["w"] == [1.0] and inputs["w_stride"] == 0
            if kind == "constant" or axis == 1:
                assert all(math.isfinite(item) for item in expected["gx"])
            # The input gradient of each row sums to zero.
            for row in range(rows):
                values = expected["gx"][row * axis : (row + 1) * axis]
                scale = math.fsum(abs(item) for item in values) + 1.0
                assert abs(math.fsum(values)) < 1e-9 * scale


def _largest_relative_bound(entry, size, name, has_w):
    _, expected, bounds = _reference(entry, size, name, has_w)
    largest = 0.0
    for key, values in expected.items():
        for want, bound in zip(values, bounds[key]):
            if abs(want) > 0.25:
                largest = max(largest, bound / abs(want))
    return largest


def test_layer_norm_bounds_stay_meaningful():
    # Binary32 rows of mixed sign stay within 2^-14 relative. Rows with a mean
    # 32 times their spread, and gradients that cancel, widen the worst case to
    # 2^-8; 16-bit outputs add a few units in the last place.
    limits = {"float32": 2.0**-8, "float16": 2.0**-8, "bfloat16": 2.0**-5}
    for entry, size, has_w in VARIANTS:
        family, dtype = _family(entry), _dtype(entry)
        for name in _variant_cases(entry, size):
            limit = limits[dtype]
            if dtype == "float32" and CASES[family][name][2] == "uniform":
                limit = 2.0**-14
            assert _largest_relative_bound(entry, size, name, has_w) < limit, (
                entry,
                size,
                name,
            )


def _mutated_forward(entry, size, name, mutation):
    """Return forward outputs with one defect, rounded to the storage type."""
    dtype = _dtype(entry)
    inputs = _inputs(entry, name)
    rows, axis = inputs["rows"], inputs["axis"]
    outputs = []
    for row_index in range(rows):
        row = inputs["x"][row_index * axis : (row_index + 1) * axis]
        if mutation == "missing-subgroup":
            # The threadgroup sum drops the last SIMD group's contribution.
            kept = row[: max(1, len(row) - SIMD_SIZE * N_READS["single"])]
            mean = math.fsum(kept) / axis
        else:
            mean = math.fsum(row) / axis
        centered = [value - mean for value in row]
        variance = math.fsum(item * item for item in centered) / axis
        eps = 0.0 if mutation == "no-eps" else EPS
        normalizer = 1.0 / math.sqrt(variance + eps)
        for index, item in enumerate(centered):
            stride = 1 if mutation == "unit-stride" else inputs["w_stride"]
            weight = inputs["w"][min(stride * index, len(inputs["w"]) - 1)]
            bias = inputs["b"][inputs["b_stride"] * index]
            if mutation == "no-bias":
                bias = 0.0
            outputs.append(_round(dtype, weight * item * normalizer + bias))
    return outputs


def test_layer_norm_checks_reject_defects():
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
            kind, weight, bias = FORWARD_CASES[name][2:]
            mutations = []
            if kind == "small":
                mutations.append("no-eps")
            if bias == "vector":
                mutations.append("no-bias")
            if weight == "strided":
                mutations.append("unit-stride")
            if FORWARD_CASES[name][1] > SIMD_SIZE * N_READS["single"] and (
                kind != "constant"
            ):
                mutations.append("missing-subgroup")
            for mutation in mutations:
                wrong = _mutated_forward(entry, size, name, mutation)
                with pytest.raises(AssertionError):
                    _check(dtype, wrong, expected["out"], bounds["out"])
    # Binary32 bounds leave no room for one 16-bit unit in the last place.
    _, expected, bounds = _reference("layer_normfloat32", 32, "rows")
    nudged = [_round("bfloat16", value) for value in expected["out"]]
    with pytest.raises(AssertionError):
        _check("float32", nudged, expected["out"], bounds["out"])
    # VJPs: a gradient without the mean-weighted correction, or swapped
    # outputs, fail.
    for dtype in DTYPES:
        entry = f"vjp_layer_norm{dtype}"
        inputs, expected, bounds = _reference(entry, 32, "rows", True)
        for key in ("gx", "gw"):
            rounded = [_round(dtype, value) for value in expected[key]]
            _check(dtype, rounded, expected[key], bounds[key])
        swapped = [_round(dtype, value) for value in expected["gw"]]
        with pytest.raises(AssertionError):
            _check(dtype, swapped, expected["gx"], bounds["gx"])
        axis = inputs["axis"]
        uncorrected = []
        for row in range(inputs["rows"]):
            values = inputs["x"][row * axis : (row + 1) * axis]
            mean, centered, variance = _row_statistics(values)
            normalizer = 1.0 / math.sqrt(variance + EPS)
            for index in range(axis):
                weight = inputs["w"][inputs["w_stride"] * index]
                gradient = inputs["g"][row * axis + index]
                uncorrected.append(_round(dtype, normalizer * weight * gradient))
        with pytest.raises(AssertionError):
            _check(dtype, uncorrected, expected["gx"], bounds["gx"])


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("entry,size,has_w", VARIANTS)
def test_layer_norm_configs_scope_target_options(tmp_path, target, entry, size, has_w):
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
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native layer norm gate")
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
    work = tmp_path_factory.mktemp("layer-norm-metal-reference")
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
    # Without a weight MLX binds the scalar weight's buffer as gw.
    return inputs["rows"] * inputs["axis"] if has_w else 1


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
            _metal_payload(dtype, _padded(inputs["b"])),
            _metal_payload(dtype, _guarded(dtype, count)),
            *scalars,
            struct.pack("<I", inputs["b_stride"]),
        ]
        outputs = {"out": (3, count)}
    else:
        payloads = [
            _metal_payload(dtype, _padded(inputs["x"])),
            _metal_payload(dtype, _padded(inputs["w"])),
            _metal_payload(dtype, _padded(inputs["g"])),
            _metal_payload(dtype, _guarded(dtype, count)),
            _metal_payload(dtype, _guarded(dtype, _gradient_count(inputs, has_w))),
            *scalars,
        ]
        outputs = {"gx": (3, count), "gw": (4, count if has_w else 0)}
    files = []
    for index, payload in enumerate(payloads):
        path = work / f"{name}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    request = {
        "buffers": files,
        "workgroupCount": [rows, 1, 1],
        "workgroupSize": [_workgroup_size(_family(entry), axis), 1, 1],
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


def _dispatch(descriptor, package, target, entry, name, has_w):
    dtype = _dtype(entry)
    inputs = _inputs(entry, name, has_w)
    rows, axis = inputs["rows"], inputs["axis"]
    count = rows * axis
    scalars = {
        "eps": EPS,
        "axis_size": axis,
        "w_stride": inputs["w_stride"],
        "b_stride": inputs.get("b_stride"),
    }
    counts = {"out": count, "gx": count, "gw": _gradient_count(inputs, has_w)}
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
            assert member in {"x", "w", "b", "g"}, member
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
    expected_inputs = 7 if family == "forward" else 6
    expected_outputs = {"out"} if family == "forward" else {"gx", "gw"}
    assert set(outputs) == expected_outputs and len(observed) == expected_inputs, (
        outputs,
        observed,
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        values,
        {name_: dict(values[name_]) for name_, _, _ in outputs.values()},
        {
            "workgroupCount": [rows, 1, 1],
            "workgroupSize": [_workgroup_size(family, axis), 1, 1],
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


def _assert_lowering(text, target, entry, has_w):
    assert "unsupported" not in text
    vjp = _family(entry) == "vjp"
    calls = 4 if vjp else 2
    root = "sqrt" if vjp else "rsqrt"
    if target == "metal":
        assert text.count("simd_sum(") == calls
        # The round trip keeps precise roots as correctly rounded helpers.
        assert f"__crossgl_metal_precise_{root}_float(" in text
        if vjp:
            assert "constant bool has_w [[function_constant(20)]];" in text
        return
    if target == "opengl":
        assert "subgroupAdd" not in text and "gl_Subgroup" not in text
        assert text.count("crossglSoftwareSubgroupSumFloat(") == calls + 1
        assert f"crossgl_metal_precise_{root}_float(" in text
        if vjp:
            # Specialized when the pipeline is created, as on Metal.
            assert "layout(constant_id = 20) const bool has_w = false;" in text
        return
    assert "WaveActiveSum" not in text and "[WaveSize" not in text
    assert text.count("__crossgl_software_subgroup_sum_float(") == calls + 1
    assert f"__crossgl_metal_precise_{root}_float(" in text
    if vjp:
        assert f"static const bool has_w = {str(has_w).lower()};" in text


@pytest.mark.parametrize("entry,size,has_w", VARIANTS)
def test_current_mlx_layer_norm_native_parity(
    current_mlx, metal_reference, tmp_path, entry, size, has_w
):
    root, target = current_mlx
    dtype = _dtype(entry)
    key = _variant_key(size, has_w)
    record = _contract()["artifacts"][target][entry][key]
    with tempfile.TemporaryDirectory(prefix=".current-layer-norm-", dir=root) as path:
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
                generated.read_text(encoding="utf-8"), target, entry, has_w
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
                    assert len(observations) == (
                        7 if _family(entry) == "forward" else 6
                    )
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


def test_ci_requires_native_layer_norm_parity():
    from tests import workflow_text
    from tests.ci_helpers import (
        assert_native_step_prerequisites,
        assert_workflow_triggers,
    )

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = workflow_text.workflow_job_step_section(
        workflow,
        "mlx-metal-porting",
        "Prove current MLX layer norm native parity",
    )
    assert_native_step_prerequisites(step, job="mlx-metal-porting")
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/layer-norm",
        "--junitxml=mlx-current-results/layer-norm.xml",
        "demos/integrations/mlx/tests/kernels/test_current_layer_norm.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_layer_norm.py"
    )
    upload = workflow_text.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Upload current MLX runtime diagnostics"
    )
    assert "if: always()" in upload
    assert "path: mlx-current-results" in upload
    assert workflow.index("Prove current MLX softmax native parity") < (
        workflow.index("Prove current MLX layer norm native parity")
    )
    assert workflow.index("Prove current MLX layer norm native parity") < (
        workflow.index("Upload current MLX runtime diagnostics")
    )
