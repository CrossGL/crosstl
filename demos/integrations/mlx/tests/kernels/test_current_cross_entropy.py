"""Translate and execute the pinned MLX cross-entropy kernels on all backends."""

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
from crosstl.project.runtime_value_encoding import (
    BFLOAT16_BITS,
    FLOAT16_BITS,
    FLOAT32_BITS,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/cross_entropy.metal"
SOURCE_SHA256 = "18e62f5a1ea339a58858b77a4666f9dbcccac70f64c38b7f6bc9b49ff1571646"
HEADER = "mlx/backend/metal/kernels/cross_entropy.h"
HEADER_SHA256 = "1e53903b9a713faafd55932f0c039552f18f6108e223dadba0a3ec2245ecab39"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_CROSS_ENTROPY"
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/cross_entropy.current-tree.translation.json"
)
DTYPES = ("float32", "float16", "bfloat16")
ENTRIES = tuple(
    f"cross_entropy{kind}_{dtype}" for dtype in DTYPES for kind in ("", "_vjp")
)
TARGETS = ("metal", "opengl", "directx")
# Four 32-lane SIMD groups exercise the cross-group threadgroup reduction.
WORKGROUP_SIZE = (128, 1, 1)
N_READS = 4
SIMD_SIZE = 32
SIGNED_MAX = 2**31 - 1
# OpenGL indexes 32-bit storage arrays. These host preconditions bound the row
# offsets and the composed element indices to signed int range.
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "forward": (
        "gid * uint64(axis_size)",
        "y_n",
        "in_ + y_n",
        "offset + i",
        "in_ + offset + i",
    ),
    "vjp": ("row_offset", "y_n", "in_ + y_n", "col", "in_ + col", "out_ + col"),
}
# Only the forward kernel reduces across SIMD groups. OpenGL and DirectX lower
# those reductions to software 32-lane groups so results do not depend on the
# device's native subgroup width.
SOFTWARE_SUBGROUP_TARGETS = ("opengl", "directx")
# name -> (rows, axis size, targets, cotangents, include in VJP cases)
CASES = {
    "tiny": (3, 7, (0, 6, -1), (0.75, -1.25, 2.0), True),
    "exact": (2, 512, (100, -512), (-0.5, 1.5), True),
    "multi": (3, 1300, (1299, -7, 640), (1.0, -2.0, 0.25), True),
    # A non-target +inf logit selects MLX's infinite-maximum branch.
    "infinite": (1, 40, (3,), (1.0,), False),
}
FORWARD_GUARD = 0xC0DEC0DE
GUARD_COUNT = 16
GUARD_BITS = {"float32": 0xC0DEC0DE, "float16": 0xC0DE, "bfloat16": 0xC0DE}
# Same tolerance as the existing MLX softmax and logsumexp native proofs.
FLOAT32_TOLERANCE = {"absolute": 5e-5, "relative": 5e-5}
# 16-bit VJP results may differ by one unit in the last place when the float
# intermediate differs; nearly all values must still round exactly.
NARROW_EXACT_FRACTION = 0.99
ULP_BITS = {"float16": 10, "bfloat16": 7}


def _dtype(entry):
    return entry.rsplit("_", 1)[1]


def _is_vjp(entry):
    return entry.startswith("cross_entropy_vjp_")


def _kind(entry):
    return "vjp" if _is_vjp(entry) else "forward"


def _cases(entry):
    return tuple(name for name, case in CASES.items() if case[4] or not _is_vjp(entry))


def _float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _float32_bits(value):
    if math.isnan(value):
        return 0x7FC00000
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _bits_float32(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _half_bits(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def _bits_half(word):
    return struct.unpack("<e", struct.pack("<H", word))[0]


def _bfloat_bits(value):
    """Round a float32 value to bfloat16 with ties to even, as MLX does."""
    bits = _float32_bits(_float32(value))
    if (bits & 0x7F800000) == 0x7F800000 and bits & 0x007FFFFF:
        return (bits >> 16) | 0x40
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def _bits_bfloat(word):
    return _bits_float32(word << 16)


def _quantize(dtype, value):
    if dtype == "float32":
        return _float32(value)
    if dtype == "float16":
        return _bits_half(_half_bits(_float32(value)))
    return _bits_bfloat(_bfloat_bits(value))


def _bits(dtype, value):
    if dtype == "float32":
        return _float32_bits(value)
    if dtype == "float16":
        return _half_bits(_float32(value))
    return _bfloat_bits(value)


def _from_bits(dtype, word):
    if dtype == "float32":
        return _bits_float32(word)
    if dtype == "float16":
        return _bits_half(word)
    return _bits_bfloat(word)


def _logits(name, dtype):
    rows, axis, _, _, _ = CASES[name]
    state = (zlib.crc32(name.encode("utf-8")) << 1) | 1
    values = []
    for row in range(rows):
        for column in range(axis):
            state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
            unit = ((state >> 11) / 2**53) * 2.0 - 1.0
            value = 4.0 * unit
            if name == "multi" and row == 1 and column % 9 == 4:
                value = -math.inf  # masked classes contribute nothing
            elif name == "multi" and row == 2:
                value = 1000.0 + 6.0 * unit  # needs the running maximum
            elif name == "infinite" and column == 17:
                value = math.inf
            values.append(_quantize(dtype, value))
    return values


def _forward_reference(name, dtype):
    rows, axis, targets, _, _ = CASES[name]
    logits = _logits(name, dtype)
    losses = []
    for row in range(rows):
        values = logits[row * axis : (row + 1) * axis]
        target = targets[row] + axis if targets[row] < 0 else targets[row]
        maximum = max(values)
        if math.isinf(maximum):
            losses.append(maximum - values[target])
            continue
        total = math.fsum(math.exp(value - maximum) for value in values)
        losses.append(math.log(total) + (maximum - values[target]))
    return logits, losses


def _vjp_reference(name, dtype):
    rows, axis, targets, cotangents, _ = CASES[name]
    logits, losses = _forward_reference(name, dtype)
    losses = [_float32(value) for value in losses]
    gradients = []
    for row in range(rows):
        target = targets[row] + axis if targets[row] < 0 else targets[row]
        x_t = logits[row * axis + target]
        for column in range(axis):
            p = math.exp((logits[row * axis + column] - x_t) - losses[row])
            gradients.append(cotangents[row] * (p - (1.0 if column == target else 0.0)))
    return logits, losses, gradients


def _within(actual, expected, tolerance):
    if math.isinf(expected) or math.isnan(expected):
        return actual == expected or (math.isnan(actual) and math.isnan(expected))
    return abs(actual - expected) <= tolerance["absolute"] + tolerance[
        "relative"
    ] * abs(expected)


def _narrow_ulp(dtype, value):
    magnitude = max(abs(value), 2.0**-14 if dtype == "float16" else 2.0**-126)
    return 2.0 ** (math.floor(math.log2(magnitude)) - ULP_BITS[dtype])


def _check_vjp(dtype, actual, gradients):
    assert len(actual) == len(gradients)
    if dtype == "float32":
        bad = [
            (index, got, want)
            for index, (got, want) in enumerate(zip(actual, gradients))
            if not _within(got, want, FLOAT32_TOLERANCE)
        ]
        assert not bad, bad[:8]
        return {"tolerance": FLOAT32_TOLERANCE, "exactFraction": None}
    rounded = [_quantize(dtype, _float32(value)) for value in gradients]
    bad = [
        (index, got, want)
        for index, (got, want) in enumerate(zip(actual, gradients))
        if abs(got - want) > _narrow_ulp(dtype, want)
    ]
    assert not bad, bad[:8]
    exact = sum(got == want for got, want in zip(actual, rounded)) / len(rounded)
    assert exact >= NARROW_EXACT_FRACTION, exact
    return {"ulpTolerance": 1, "exactFraction": exact}


def _element_layout(binding):
    layout = binding["scalarLayout"]
    return layout["elementType"], layout.get("storageEncoding")


def _runtime_value(logical, element_type, values, storage_encoding=None):
    """Encode logical values for the reflected physical storage."""
    if element_type == "float32":
        assert storage_encoding is None
        return {
            "dtype": "float32",
            "shape": [len(values)],
            "encoding": FLOAT32_BITS,
            "values": [_float32_bits(_float32(value)) for value in values],
        }
    words = [_bits(logical, value) for value in values]
    if storage_encoding is not None:
        # DirectX binary16 storage takes logical float16 words in uint16 buffers.
        assert element_type == "uint16"
        assert storage_encoding == {
            "encoding": FLOAT16_BITS,
            "logicalElementType": logical,
        }
        return {
            "dtype": logical,
            "shape": [len(values)],
            "encoding": FLOAT16_BITS,
            "values": words,
        }
    if element_type == "uint16":
        return {"dtype": "uint16", "shape": [len(values)], "values": words}
    assert element_type == logical, (element_type, logical)
    return {
        "dtype": logical,
        "shape": [len(values)],
        "encoding": FLOAT16_BITS if logical == "float16" else BFLOAT16_BITS,
        "values": words,
    }


def _decode(logical, value):
    if value["dtype"] == "float32":
        words = value["values"]
        assert all(isinstance(word, int) for word in words), words[:4]
        floats = [_bits_float32(word) for word in words]
        # Widened 16-bit storage must still hold exactly representable values.
        if logical != "float32":
            assert all(
                math.isnan(item) or _quantize(logical, item) == item for item in floats
            )
        return floats
    assert value["dtype"] in {logical, "uint16"}, (value["dtype"], logical)
    return [_from_bits(logical, word) for word in value["values"]]


def _guard_values(logical, element_type):
    if element_type == "float32":
        if logical == "float32":
            return [GUARD_BITS["float32"]] * GUARD_COUNT
        return [_float32_bits(_from_bits(logical, GUARD_BITS[logical]))] * GUARD_COUNT
    return [GUARD_BITS[logical]] * GUARD_COUNT


def _config(output, target, entry):
    kind = _kind(entry)
    options = ""
    if kind == "forward" and target == "directx":
        options += f"""
[project.subgroup_width_rules]
"{SOURCE}" = {SIMD_SIZE}
[project.source_options.metal.target_options.directx]
relative_wave_shuffle_out_of_range = "self"
software_subgroup_width = {SIMD_SIZE}
"""
    if target == "opengl":
        if kind == "forward":
            options += f"""
[project.source_options.metal.target_options.opengl]
software_subgroup_width = {SIMD_SIZE}
"""
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
"{entry}" = [{", ".join(str(item) for item in WORKGROUP_SIZE)}]
[project.source_options.metal]
max_template_specializations = 128
max_template_materialization_work = 8192
{options}"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _expected_validation_contract():
    return {
        "workgroupSize": list(WORKGROUP_SIZE),
        "softwareSubgroupWidth": {
            target: {
                entry: (
                    SIMD_SIZE
                    if target in SOFTWARE_SUBGROUP_TARGETS and not _is_vjp(entry)
                    else None
                )
                for entry in ENTRIES
            }
            for target in TARGETS
        },
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
                "targets": list(targets),
                "cotangents": list(cotangents),
                "vjp": vjp,
            }
            for name, (rows, axis, targets, cotangents, vjp) in CASES.items()
        },
        "outputGuardCount": GUARD_COUNT,
        "observedInputCount": {"forward": 3, "vjp": 5},
        "float32Tolerance": FLOAT32_TOLERANCE,
        "narrowVjpTolerance": {
            "ulp": 1,
            "minimumExactFraction": NARROW_EXACT_FRACTION,
        },
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def test_current_mlx_cross_entropy_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-cross-entropy-contract"
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
        for record in artifacts.values():
            assert set(record) == {"sha256", "sizeBytes"}
            assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
    assert contract["validationContract"] == _expected_validation_contract()
    value_counts = {
        entry: sum(
            CASES[name][0] * (CASES[name][1] if _is_vjp(entry) else 1)
            for name in _cases(entry)
        )
        for entry in ENTRIES
    }
    assert contract["scope"] == {
        "coveredEntryCount": 6,
        "runtimeCoveredEntryCount": 6,
        "checkedValueCountByEntry": value_counts,
        "completeCorpusEntryCount": 17832,
        "fastMathLowering": "metal-keeps-fast-exp-others-use-default-exp",
        "donatedInputAliasingCovered": False,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


@pytest.mark.parametrize("dtype", DTYPES)
def test_cross_entropy_cases_cover_reduction_edges(dtype):
    for name, (rows, axis, targets, cotangents, vjp) in CASES.items():
        logits, losses = _forward_reference(name, dtype)
        assert len(logits) == rows * axis and len(losses) == rows
        assert all(_quantize(dtype, value) == value for value in logits)
        assert len(targets) == len(cotangents) == rows
        assert all(-axis <= target < axis for target in targets)
        if name == "infinite":
            assert losses == [math.inf]
            continue
        assert all(math.isfinite(loss) and loss > 0 for loss in losses)
    # One full row per pass, a partial tail, several passes and a single pass
    # with idle threads all appear among the cases.
    passes = {
        math.ceil(axis / (N_READS * WORKGROUP_SIZE[0]))
        for _, axis, *_ in CASES.values()
    }
    assert passes == {1, 3}
    assert any(
        axis % (N_READS * WORKGROUP_SIZE[0]) == 0 for _, axis, *_ in CASES.values()
    )
    assert any(target < 0 for case in CASES.values() for target in case[2])
    multi, _ = _forward_reference("multi", dtype)
    assert any(math.isinf(value) and value < 0 for value in multi[1300:2600])
    assert min(multi[2600:]) > 900.0
    for name in _cases("cross_entropy_vjp_float32"):
        _, losses, gradients = _vjp_reference(name, dtype)
        rows, axis = CASES[name][:2]
        for row in range(rows):
            # Probabilities sum to one, so each row's gradient sums to zero.
            row_sum = math.fsum(gradients[row * axis : (row + 1) * axis])
            assert abs(row_sum) < 1e-5 * max(1.0, abs(CASES[name][3][row]))


def test_cross_entropy_narrow_checks_reject_rounding_defects():
    _, _, gradients = _vjp_reference("multi", "bfloat16")
    rounded = [_quantize("bfloat16", _float32(value)) for value in gradients]
    assert _check_vjp("bfloat16", rounded, gradients)["exactFraction"] == 1.0
    truncated = [
        _bits_bfloat(_float32_bits(_float32(value)) >> 16) for value in gradients
    ]
    with pytest.raises(AssertionError):
        _check_vjp("bfloat16", truncated, gradients)
    shifted = list(rounded)
    shifted[5] = shifted[5] * 1.25 + 0.01
    with pytest.raises(AssertionError):
        _check_vjp("bfloat16", shifted, gradients)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_cross_entropy_configs_scope_target_options(tmp_path, target, entry):
    path = tmp_path / "crosstl.toml"
    path.write_text(_config("out", target, entry), encoding="utf-8")
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
    software = target in SOFTWARE_SUBGROUP_TARGETS and not _is_vjp(entry)
    assert ("software_subgroup_width = 32" in text) is software


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native cross-entropy gate")
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
    work = tmp_path_factory.mktemp("cross-entropy-metal-reference")
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


def _metal_payload(dtype, values):
    if dtype == "float32":
        return struct.pack(f"<{len(values)}I", *(_float32_bits(v) for v in values))
    return struct.pack(f"<{len(values)}H", *(_bits(dtype, v) for v in values))


def _upstream_metal(work, reference, entry, data, name):
    runner, library = reference
    dtype = _dtype(entry)
    rows, axis = CASES[name][:2]
    output_type = dtype if _is_vjp(entry) else "float32"
    count = rows * axis if _is_vjp(entry) else rows
    guards = [_from_bits(output_type, GUARD_BITS[output_type])] * GUARD_COUNT
    payloads = [_metal_payload(dtype, data["logits"])]
    payloads.append(struct.pack(f"<{rows}i", *data["targets"]))
    if _is_vjp(entry):
        payloads.append(_metal_payload("float32", data["losses"]))
        payloads.append(_metal_payload("float32", data["cotangents"]))
    payloads.append(_metal_payload(output_type, [0.0] * count + guards))
    payloads.append(struct.pack("<i", axis))
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
                "workgroupSize": list(WORKGROUP_SIZE),
                "simdWidth": SIMD_SIZE,
            }
        ),
        encoding="utf-8",
    )
    output = work / f"{name}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, output], work, f"{name}-upstream")
    )
    output_index = len(payloads) - 2
    for index in range(len(payloads)):
        if index != output_index:
            assert (output / f"buffer-{index}.bin").read_bytes() == payloads[index]
    raw = (output / f"buffer-{output_index}.bin").read_bytes()
    if output_type == "float32":
        words = struct.unpack(f"<{len(raw) // 4}I", raw)
    else:
        words = struct.unpack(f"<{len(raw) // 2}H", raw)
    values = [_from_bits(output_type, word) for word in words]
    assert [_bits(output_type, v) for v in values[count:]] == [
        GUARD_BITS[output_type]
    ] * GUARD_COUNT
    return device, values[:count]


def _case_data(entry, name):
    dtype = _dtype(entry)
    rows, axis, targets, cotangents, _ = CASES[name]
    if _is_vjp(entry):
        logits, losses, gradients = _vjp_reference(name, dtype)
        return {
            "logits": logits,
            "targets": list(targets),
            "losses": losses,
            "cotangents": [_float32(value) for value in cotangents],
            "expected": gradients,
            "count": rows * axis,
        }
    logits, losses = _forward_reference(name, dtype)
    return {
        "logits": logits,
        "targets": list(targets),
        "expected": losses,
        "count": rows,
    }


def _dispatch(descriptor, package, target, entry, name, data):
    dtype = _dtype(entry)
    rows, axis = CASES[name][:2]
    inputs, observed, output = {}, [], None
    prefix = re.sub(r"_+", "_", entry) + "_"
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        layout = binding["scalarLayout"]
        # DirectX constant members carry the sanitized entry name as a prefix.
        member = layout.get("memberName", binding["name"]).removeprefix(prefix)
        element_type, encoding = _element_layout(binding)
        if member == "in_":
            value = _runtime_value(dtype, element_type, data["logits"], encoding)
        elif member == "targets":
            value = {"dtype": "int32", "shape": [rows], "values": data["targets"]}
        elif member in {"loss", "cotan"}:
            key = "losses" if member == "loss" else "cotangents"
            value = _runtime_value("float32", element_type, data[key])
        elif member == "axis_size":
            value = {"dtype": "int32", "shape": [1], "values": [axis]}
        else:
            assert member == "out_", member
            logical = dtype if _is_vjp(entry) else "float32"
            value = _runtime_value(
                logical, element_type, [0.0] * data["count"], encoding
            )
            value["values"] = value["values"][: data["count"]] + _guard_values(
                logical, element_type
            )
            value["shape"] = [data["count"] + GUARD_COUNT]
            output = (binding["name"], logical, value["dtype"])
        if encoding is None:
            assert element_type == value["dtype"], (member, element_type)
        else:
            assert encoding["logicalElementType"] == dtype, (member, encoding)
            assert value["dtype"] == dtype and value["encoding"] == FLOAT16_BITS
        inputs[binding["name"]] = value
        if member != "out_":
            observed.append(binding["name"])
    expected_observed = 5 if _is_vjp(entry) else 3
    assert output is not None and len(observed) == expected_observed
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output[0]: dict(inputs[output[0]])},
        {"workgroupCount": [rows, 1, 1], "workgroupSize": list(WORKGROUP_SIZE)},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output, inputs[output[0]]["values"][data["count"] :]


def _check_output(entry, actual, data):
    if _is_vjp(entry):
        return _check_vjp(_dtype(entry), actual, data["expected"])
    bad = [
        (index, got, want)
        for index, (got, want) in enumerate(zip(actual, data["expected"]))
        if not _within(got, want, FLOAT32_TOLERANCE)
    ]
    assert len(actual) == len(data["expected"]) and not bad, bad
    return {"tolerance": FLOAT32_TOLERANCE}


@pytest.mark.parametrize("entry", ENTRIES)
def test_current_mlx_cross_entropy_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    dtype = _dtype(entry)
    record = _contract()["artifacts"][target][entry]
    with tempfile.TemporaryDirectory(
        prefix=".current-cross-entropy-", dir=root
    ) as directory:
        work = Path(directory)
        try:
            config = work / "crosstl.toml"
            config.write_text(
                _config(f"{work.name}/out", target, entry), encoding="utf-8"
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
            assert execution["workgroupSize"] == list(WORKGROUP_SIZE)
            generated = root / artifact["path"]
            text = generated.read_text(encoding="utf-8")
            assert "unsupported" not in text
            software = target in SOFTWARE_SUBGROUP_TARGETS and not _is_vjp(entry)
            if target == "metal":
                # Every fast::exp of cross_entropy.h keeps its namespace: four
                # in the forward kernel, two in the VJP.
                assert (
                    text.count("metal::fast::exp(")
                    == text.count("fast::")
                    == (2 if _is_vjp(entry) else 4)
                )
            else:
                # OpenGL and DirectX call their default exp.
                assert "fast::" not in text
            if target == "opengl":
                assert ("subgroupAdd" in text) is False
                assert ("crossglSoftwareSubgroupSumFloat" in text) is software
            if target == "directx":
                assert "WaveActiveSum" not in text and "WaveActiveMax" not in text
            if target == "metal":
                _metal_library(generated, work / "translated.metallib", root)
            else:
                _validate_generated_artifact(generated, work, target)
            descriptor, package = _prepare_native_package(translation, work)
            executor = _executor(target)
            evidence = []
            try:
                for name in _cases(entry):
                    data = _case_data(entry, name)
                    request, output, guards = _dispatch(
                        descriptor, package, target, entry, name, data
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
                    output_name, logical, element_type = output
                    value = result.outputs[output_name]
                    assert value["dtype"] == element_type
                    assert value["values"][data["count"] :] == guards
                    actual = _decode(
                        logical,
                        {**value, "values": value["values"][: data["count"]]},
                    )
                    summary = _check_output(entry, actual, data)
                    observations = result.details["inputObservations"]
                    assert len(observations) == (5 if _is_vjp(entry) else 3)
                    assert all(item["unchanged"] for item in observations.values())
                    upstream = None
                    if target == "metal":
                        device, values = _upstream_metal(
                            work, metal_reference, entry, data, name
                        )
                        upstream = {
                            "device": device,
                            "reference": _check_output(entry, values, data),
                            "bitIdenticalToTranslated": (
                                [
                                    _bits("float32" if not _is_vjp(entry) else dtype, a)
                                    for a in actual
                                ]
                                == [
                                    _bits("float32" if not _is_vjp(entry) else dtype, b)
                                    for b in values
                                ]
                            ),
                        }
                    evidence.append(
                        {
                            "case": name,
                            "rows": CASES[name][0],
                            "axisSize": CASES[name][1],
                            "checkedValueCount": data["count"],
                            "comparison": summary,
                            "maxAbsoluteError": max(
                                (
                                    abs(a - b)
                                    for a, b in zip(actual, data["expected"])
                                    if math.isfinite(b)
                                ),
                                default=0.0,
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


def test_ci_requires_native_cross_entropy_parity():
    from tests import workflow_text
    from tests.ci_helpers import (
        assert_native_step_prerequisites,
        assert_workflow_triggers,
    )

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = workflow_text.workflow_job_step_section(
        workflow,
        "mlx-metal-porting",
        "Prove current MLX new-unit native parity",
    )
    assert_native_step_prerequisites(step, job="mlx-metal-porting")
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/new-units",
        "--junitxml=mlx-current-results/new-units.xml",
        "demos/integrations/mlx/tests/kernels/test_current_cross_entropy.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_cross_entropy.py"
    )
