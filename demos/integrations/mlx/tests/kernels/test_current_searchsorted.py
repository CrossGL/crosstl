"""Translate and execute pinned MLX searchsorted kernels on all three backends."""

import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
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

SOURCE = "mlx/backend/metal/kernels/searchsorted.metal"
SOURCE_SHA256 = "40c41784560f48d230879ed82dfb6b9317c9a9ab4603e53549510fa0e88024fb"
HEADER = "mlx/backend/metal/kernels/searchsorted.h"
HEADER_SHA256 = "4e80cde60cf0d79cee2e630a86e517b8d77589ba93eca3312d8e2c607963d093"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_SEARCHSORTED"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/searchsorted.current-tree.translation.json"
)
DTYPES = (
    "bool_",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "bfloat16",
    "complex64",
)
ENTRIES = tuple(
    f"searchsorted_{dtype}_{side}" for dtype in DTYPES for side in ("left", "right")
)
TARGETS = ("metal", "opengl", "directx")
# OpenGL indexes 32-bit storage arrays. These host preconditions bound the
# strided element index and the flattened output offset to signed int range.
OPENGL_INDEX_RANGE_ASSERTIONS = (
    ("int64(mid) * a_stride", 0, 2**31 - 1),
    ("offset", 0, 2**31 - 1),
)
# (sequence length, sequence stride, dispatch grid). Each grid has 48 threads;
# the multi-axis grids exercise MLX's flattened (x, y, z) output offset.
CASES = {
    "single": (1, 1, (48, 1, 1)),
    "odd": (33, 1, (8, 3, 2)),
    "strided": (40, 2, (4, 4, 3)),
}
QUERY_COUNT = 48
GUARD = 0xABCDEF01
GUARD_COUNT = 16
INTEGER_LIMITS = {
    "uint8": (0, 2**8 - 1),
    "uint16": (0, 2**16 - 1),
    "uint32": (0, 2**32 - 1),
    "uint64": (0, 2**64 - 1),
    "int8": (-(2**7), 2**7 - 1),
    "int16": (-(2**15), 2**15 - 1),
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
}
METAL_FORMATS = {
    "bool_": "?",
    "uint8": "B",
    "uint16": "H",
    "uint32": "I",
    "uint64": "Q",
    "int8": "b",
    "int16": "h",
    "int32": "i",
    "int64": "q",
}
# Reflected physical element types: narrow and 16-bit floating storage widens
# on OpenGL, DirectX keeps 16-bit integers and stores 16-bit floats as bits.
ELEMENT_TYPES = {
    "metal": {
        "bool_": "bool",
        "float16": "float16",
        "bfloat16": "bfloat16",
        "complex64": "float32",
    },
    "directx": {
        "bool_": "uint32",
        "uint8": "uint32",
        "int8": "int32",
        "float16": "uint16",
        "bfloat16": "uint16",
        "complex64": "float32",
    },
    "opengl": {
        "bool_": "uint32",
        "uint8": "uint32",
        "uint16": "uint32",
        "int8": "int32",
        "int16": "int32",
        "float16": "float32",
        "bfloat16": "float32",
        "complex64": "float32",
    },
}


def _dtype(entry):
    return entry[len("searchsorted_") :].rsplit("_", 1)[0]


def _element_type(target, dtype):
    return ELEMENT_TYPES[target].get(dtype, dtype)


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _float32_word(value):
    return 0x7FC00000 if math.isnan(value) else _float_bits(value)


def _half_bits(value):
    if math.isnan(value):
        return 0x7E00
    return struct.unpack("<H", struct.pack("<e", value))[0]


def _bfloat_bits(value):
    bits = 0x7FC00000 if math.isnan(value) else _float_bits(value)
    assert bits & 0xFFFF == 0, value
    return bits >> 16


def _levels(dtype):
    """Return strictly increasing logical values ordered by MLX ``LessThan``."""
    if dtype == "bool_":
        return [False, True]
    if dtype in INTEGER_LIMITS:
        low, high = INTEGER_LIMITS[dtype]
        bits = int(dtype.lstrip("uint"))
        candidates = {low, low + 1, high - 1, high, 0, 1, 2, 3, 5, 8, 13, 21, 34}
        candidates |= {55, 89, 100, 127}
        if low < 0:
            candidates |= {-1, -2, -3, -5, -8, -13, -21, -34, -55, -89, -100}
        else:
            # Values around the sign bit catch signed comparison of unsigned data.
            candidates |= {2 ** (bits - 1) - 1, 2 ** (bits - 1), 2 ** (bits - 1) + 1}
        return sorted(value for value in candidates if low <= value <= high)
    if dtype == "float32":
        return [
            -math.inf,
            -3.0e38,
            -1.5e10,
            -2.5,
            -1.0,
            -0.5,
            0.0,
            1.0e-30,
            0.5,
            1.0,
            1.5,
            2.5,
            1000.0,
            1.5e10,
            3.0e38,
            math.inf,
            math.nan,
        ]
    if dtype == "float16":
        return [
            -math.inf,
            -65504.0,
            -1000.0,
            -2.5,
            -1.0,
            -0.5,
            0.0,
            2.0**-14,
            0.5,
            1.0,
            1.5,
            2.5,
            1000.0,
            65504.0,
            math.inf,
            math.nan,
        ]
    if dtype == "bfloat16":
        return [
            -math.inf,
            -(2.0**100),
            -1024.0,
            -2.5,
            -1.0,
            -0.5,
            0.0,
            2.0**-100,
            0.5,
            1.0,
            1.5,
            2.5,
            1024.0,
            2.0**100,
            math.inf,
            math.nan,
        ]
    assert dtype == "complex64"
    return [
        (-math.inf, 0.0),
        (-2.5, 1.0),
        (-2.5, 3.0),
        (-1.0, -1.0),
        (0.0, -2.0),
        (0.0, 0.0),
        (0.0, 0.5),
        (1.5, -1.0),
        (1.5, 2.5),
        (1000.0, 0.0),
        (math.inf, -1.0),
        (math.nan, 0.0),
    ]


def _is_nan(dtype, value):
    if dtype == "complex64":
        return math.isnan(value[0]) or math.isnan(value[1])
    return isinstance(value, float) and math.isnan(value)


def _less_than(dtype, left, right):
    """Mirror MLX ``LessThan``: NaN sorts after every ordered value."""
    if dtype in {"float16", "float32", "bfloat16", "complex64"}:
        left_nan = _is_nan(dtype, left)
        right_nan = _is_nan(dtype, right)
        if left_nan or right_nan:
            return (not left_nan) and right_nan
    if dtype == "complex64":
        return left[0] < right[0] or (left[0] == right[0] and left[1] < right[1])
    return left < right


def _search(dtype, sequence, value, right):
    """Mirror the upstream binary search, including its midpoint arithmetic."""
    low, high = 0, len(sequence)
    while low < high:
        middle = low + (high - low) // 2
        candidate = sequence[middle]
        below = (
            not _less_than(dtype, value, candidate)
            if right
            else _less_than(dtype, candidate, value)
        )
        if below:
            low = middle + 1
        else:
            high = middle
    return low


def _case(dtype, case):
    length, stride, grid = CASES[case]
    levels = _levels(dtype)
    if length == 1:
        sequence = [levels[len(levels) // 2]]
    else:
        # A nondecreasing sequence with repeated values and every level present.
        sequence = [levels[(index * len(levels)) // length] for index in range(length)]
    queries = list(levels)
    if dtype in {"float16", "float32", "bfloat16"}:
        queries.append(-0.0)
    elif dtype == "complex64":
        # An imaginary NaN is unordered in the same way as a real NaN.
        queries.extend([(0.0, math.nan), (-0.0, 0.0)])
    while len(queries) < QUERY_COUNT:
        queries.extend(reversed(levels))
    queries = queries[:QUERY_COUNT]
    # Strided padding must never be read; use the smallest value as a decoy.
    storage = [levels[0]] * (length * stride)
    storage[::stride] = sequence
    assert math.prod(grid) == QUERY_COUNT
    return {
        "length": length,
        "stride": stride,
        "grid": list(grid),
        "sequence": sequence,
        "storage": storage,
        "queries": queries,
    }


def _expected(dtype, case, right):
    return [_search(dtype, case["sequence"], value, right) for value in case["queries"]]


def _runtime_value(target, dtype, values):
    element_type = _element_type(target, dtype)
    if dtype == "complex64":
        return {
            "dtype": element_type,
            "shape": [len(values), 2],
            "encoding": FLOAT32_BITS,
            "values": [_float32_word(part) for value in values for part in value],
        }
    if dtype == "float32":
        return {
            "dtype": "float32",
            "shape": [len(values)],
            "encoding": FLOAT32_BITS,
            "values": [_float32_word(value) for value in values],
        }
    if dtype in {"float16", "bfloat16"}:
        bits = [
            (_half_bits if dtype == "float16" else _bfloat_bits)(value)
            for value in values
        ]
        if element_type == "uint16" and dtype == "float16":
            # DirectX binary16 storage takes logical float16 words.
            return {
                "dtype": "float16",
                "shape": [len(values)],
                "encoding": FLOAT16_BITS,
                "values": bits,
            }
        if element_type == "uint16":
            return {"dtype": "uint16", "shape": [len(values)], "values": bits}
        if element_type == "float32":
            widened = []
            for word in bits:
                if dtype == "bfloat16":
                    widened.append(word << 16)
                else:
                    value = struct.unpack("<e", struct.pack("<H", word))[0]
                    widened.append(_float32_word(value))
            return {
                "dtype": "float32",
                "shape": [len(values)],
                "encoding": FLOAT32_BITS,
                "values": widened,
            }
        return {
            "dtype": element_type,
            "shape": [len(values)],
            "encoding": FLOAT16_BITS if dtype == "float16" else BFLOAT16_BITS,
            "values": bits,
        }
    if dtype == "bool_":
        converted = [bool(value) for value in values]
        if element_type == "uint32":
            converted = [int(value) for value in converted]
        return {"dtype": element_type, "shape": [len(values)], "values": converted}
    return {"dtype": element_type, "shape": [len(values)], "values": list(values)}


def _metal_bytes(dtype, values):
    if dtype == "complex64":
        return b"".join(
            struct.pack("<II", *(_float32_word(part) for part in value))
            for value in values
        )
    if dtype == "float32":
        return b"".join(struct.pack("<I", _float32_word(value)) for value in values)
    if dtype == "float16":
        return b"".join(struct.pack("<H", _half_bits(value)) for value in values)
    if dtype == "bfloat16":
        return b"".join(struct.pack("<H", _bfloat_bits(value)) for value in values)
    return struct.pack("<" + METAL_FORMATS[dtype] * len(values), *values)


def _config(output, target, entry):
    assertions = ""
    if target == "opengl":
        for expression, low, high in OPENGL_INDEX_RANGE_ASSERTIONS:
            assertions += f"""
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "{expression}"
minimum = {low}
maximum = {high}
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
"{entry}" = [1, 1, 1]
[project.source_options.metal]
max_template_specializations = 128
max_template_materialization_work = 8192
{assertions}"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_current_mlx_searchsorted_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-searchsorted-contract"
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
    validation = contract["validationContract"]
    assert validation["workgroupSize"] == [1, 1, 1]
    assert validation["indexRangeAssertions"] == {
        "opengl": [
            {"expression": expression, "minimum": low, "maximum": high}
            for expression, low, high in OPENGL_INDEX_RANGE_ASSERTIONS
        ],
        "metal": [],
        "directx": [],
    }
    assert validation["cases"] == {
        name: {"length": length, "stride": stride, "grid": list(grid)}
        for name, (length, stride, grid) in CASES.items()
    }
    assert validation["queriesPerCase"] == QUERY_COUNT
    assert validation["outputGuardCount"] == GUARD_COUNT
    assert validation["observedInputCount"] == 4
    assert validation["elementTypes"] == {
        target: {dtype: _element_type(target, dtype) for dtype in DTYPES}
        for target in TARGETS
    }
    assert validation["compilerValidationRequiredInCi"] == list(TARGETS)
    assert validation["runtimeRequiredInCi"] == list(TARGETS)
    assert validation["upstreamMetallibParityTargets"] == ["metal"]
    assert contract["scope"] == {
        "coveredEntryCount": 26,
        "runtimeCoveredEntryCount": 26,
        "caseCountPerEntry": len(CASES),
        "queryCountByTarget": {
            target: len(ENTRIES) * len(CASES) * QUERY_COUNT for target in TARGETS
        },
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


@pytest.mark.parametrize("dtype", DTYPES)
def test_searchsorted_cases_cover_ordering_edges(dtype):
    levels = _levels(dtype)
    assert all(
        _less_than(dtype, left, right) for left, right in zip(levels, levels[1:])
    )
    for name in CASES:
        case = _case(dtype, name)
        sequence = case["sequence"]
        assert len(sequence) == case["length"]
        assert not any(
            _less_than(dtype, right, left)
            for left, right in zip(sequence, sequence[1:])
        )
        assert case["storage"][:: case["stride"]] == sequence or dtype in {
            "float16",
            "float32",
            "bfloat16",
            "complex64",
        }
        left = _expected(dtype, case, False)
        right = _expected(dtype, case, True)
        assert len(left) == len(right) == QUERY_COUNT
        assert all(low <= high for low, high in zip(left, right))
        if name != "single":
            # Repeated values and an absent minimum/maximum separate both sides.
            assert any(low < high for low, high in zip(left, right))
            assert 0 in left and case["length"] in right
    if dtype in {"float16", "float32", "bfloat16", "complex64"}:
        case = _case(dtype, "odd")
        nan_query = case["queries"].index(levels[-1])
        first_nan = next(
            index
            for index, value in enumerate(case["sequence"])
            if _is_nan(dtype, value)
        )
        assert _expected(dtype, case, False)[nan_query] == first_nan
        assert _expected(dtype, case, True)[nan_query] == case["length"]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_searchsorted_runtime_values_match_physical_storage(target, dtype):
    case = _case(dtype, "strided")
    value = _runtime_value(target, dtype, case["storage"])
    if target == "directx" and dtype == "float16":
        # Physical uint16 storage carries logical binary16 words.
        assert _element_type(target, dtype) == "uint16"
        assert value["dtype"] == "float16" and value["encoding"] == FLOAT16_BITS
    else:
        assert value["dtype"] == _element_type(target, dtype)
    width = 2 if dtype == "complex64" else 1
    assert value["shape"][0] * width == len(value["values"]) or dtype != "complex64"
    if target == "metal":
        payload = _metal_bytes(dtype, case["storage"])
        item = {"complex64": 8, "float16": 2, "bfloat16": 2, "float32": 4}.get(
            dtype, struct.calcsize(METAL_FORMATS.get(dtype, "I"))
        )
        assert len(payload) == item * len(case["storage"])


@pytest.mark.parametrize("target", TARGETS)
def test_searchsorted_configs_scope_opengl_index_preconditions(tmp_path, target):
    path = tmp_path / "crosstl.toml"
    path.write_text(_config("out", target, ENTRIES[0]), encoding="utf-8")
    config = load_project_config(tmp_path, path)
    assert [item.to_json() for item in config.index_range_assertions] == (
        [
            {
                "source": SOURCE,
                "expression": expression,
                "minimum": low,
                "maximum": high,
            }
            for expression, low, high in OPENGL_INDEX_RANGE_ASSERTIONS
        ]
        if target == "opengl"
        else []
    )
    assert list(config.targets) == [target]


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native searchsorted gate")
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
    work = tmp_path_factory.mktemp("searchsorted-metal-reference")
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


def _upstream_metal(work, reference, entry, dtype, case, name):
    runner, library = reference
    guarded = [GUARD] * (QUERY_COUNT + GUARD_COUNT)
    payloads = [
        _metal_bytes(dtype, case["storage"]),
        _metal_bytes(dtype, case["queries"]),
        struct.pack(f"<{len(guarded)}I", *guarded),
        struct.pack("<I", case["length"]),
        struct.pack("<q", case["stride"]),
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
                "workgroupCount": case["grid"],
                "workgroupSize": [1, 1, 1],
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    output = work / f"{name}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, output], work, f"{name}-upstream")
    )
    raw = (output / "buffer-2.bin").read_bytes()
    for index in (0, 1, 3, 4):
        assert (output / f"buffer-{index}.bin").read_bytes() == payloads[index]
    return device, list(struct.unpack(f"<{len(raw) // 4}I", raw))


def _dispatch(descriptor, package, target, entry, dtype, case):
    data = {
        "a": _runtime_value(target, dtype, case["storage"]),
        "v": _runtime_value(target, dtype, case["queries"]),
        "out_": {
            "dtype": "uint32",
            "shape": [QUERY_COUNT + GUARD_COUNT],
            "values": [GUARD] * (QUERY_COUNT + GUARD_COUNT),
        },
        "n": {"dtype": "uint32", "shape": [1], "values": [case["length"]]},
        "a_stride": {"dtype": "int64", "shape": [1], "values": [case["stride"]]},
    }
    inputs, observed, output_name = {}, [], None
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        layout = binding["scalarLayout"]
        # DirectX constant members carry the sanitized entry name as a prefix.
        name = layout.get("memberName", binding["name"]).removeprefix(
            re.sub(r"_+", "_", entry) + "_"
        )
        value = data[name]
        storage = layout.get("storageEncoding")
        if storage is None:
            assert layout["elementType"] == value["dtype"], (name, layout)
        else:
            assert layout["elementType"] == "uint16", (name, layout)
            assert storage == {
                "encoding": FLOAT16_BITS,
                "logicalElementType": value["dtype"],
            }
            assert value["encoding"] == FLOAT16_BITS
        inputs[binding["name"]] = value
        if name == "out_":
            output_name = binding["name"]
        else:
            observed.append(binding["name"])
    assert output_name is not None and len(inputs) == 5 and len(observed) == 4
    expected_outputs = {output_name: dict(data["out_"])}
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected_outputs,
        {"workgroupCount": case["grid"], "workgroupSize": [1, 1, 1]},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output_name


@pytest.mark.parametrize("entry", ENTRIES)
def test_current_mlx_searchsorted_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    dtype = _dtype(entry)
    right = entry.endswith("_right")
    record = _contract()["artifacts"][target][entry]
    with tempfile.TemporaryDirectory(
        prefix=".current-searchsorted-", dir=root
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
            assert execution["workgroupSize"] == [1, 1, 1]
            generated = root / artifact["path"]
            text = generated.read_text(encoding="utf-8")
            assert "unsupported" not in text
            if target == "directx" and dtype in {"uint8", "int8"}:
                # Widened entry storage reaches the helper as one carrier per byte.
                assert "% 4) * 8" not in text and ") / 4)]" not in text
            if target == "metal":
                _metal_library(generated, work / "translated.metallib", root)
            else:
                _validate_generated_artifact(generated, work, target)
            descriptor, package = _prepare_native_package(translation, work)
            executor = _executor(target)
            evidence = []
            try:
                for name in CASES:
                    case = _case(dtype, name)
                    expected = _expected(dtype, case, right)
                    request, output_name = _dispatch(
                        descriptor, package, target, entry, dtype, case
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
                    output = result.outputs[output_name]
                    assert output["dtype"] == "uint32"
                    assert output["values"] == expected + [GUARD] * GUARD_COUNT
                    observations = result.details["inputObservations"]
                    assert len(observations) == 4
                    assert all(item["unchanged"] for item in observations.values())
                    upstream = None
                    if target == "metal":
                        device, values = _upstream_metal(
                            work, metal_reference, entry, dtype, case, name
                        )
                        assert values == expected + [GUARD] * GUARD_COUNT
                        upstream = {"device": device, "matched": True}
                    evidence.append(
                        {
                            "case": name,
                            "length": case["length"],
                            "stride": case["stride"],
                            "grid": case["grid"],
                            "expected": expected,
                            "outputGuardCount": GUARD_COUNT,
                            "observedInputCount": 4,
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
                        "artifactSha256": record["sha256"],
                        "cases": evidence,
                        "indexRangeAssertions": (
                            [list(item) for item in OPENGL_INDEX_RANGE_ASSERTIONS]
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


def test_ci_requires_native_searchsorted_parity():
    from tests import workflow_text
    from tests.ci_helpers import (
        assert_native_step_prerequisites,
        assert_workflow_triggers,
    )

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Prove current MLX searchsorted native parity"
    )
    assert_native_step_prerequisites(step, job="mlx-metal-porting")
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "--timeout-seconds 900",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/searchsorted",
        "--junitxml=mlx-current-results/searchsorted.xml",
        "demos/integrations/mlx/tests/kernels/test_current_searchsorted.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_searchsorted.py"
    )
    upload = workflow_text.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Upload current MLX runtime diagnostics"
    )
    assert "if: always()" in upload
    assert "path: mlx-current-results" in upload
    assert workflow.index("Prove current MLX searchsorted native parity") < (
        workflow.index("Upload current MLX runtime diagnostics")
    )
