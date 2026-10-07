"""Pinned minimum and maximum preserve selected operands through native packages."""

import hashlib
import itertools
import json
import os
import random
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import load_project_config, translate_project
from demos.integrations.mlx.tests.kernels.test_binary_complete_opengl import (
    BINARY_OPENGL_WORKLOADS,
    MLX_BINARY_SHA256,
    MLX_BINARY_SOURCE,
    _project_config,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import MLX_COMMIT
from demos.integrations.mlx.tests.kernels.test_current_copy import _half_payload
from tests.runtime_helpers import _prepare_native_package, _validate
from tests.test_translator.test_bfloat_buffer_runtime import _storage as _bfloat_payload
from tests.test_translator.test_boolean_buffer_runtime import (
    _bound_values,
)
from tests.test_translator.test_boolean_buffer_runtime import (
    _request as _dispatch_request,
)
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_product import _package
from tools import ci_coverage

ROOT = Path(__file__).resolve().parents[5]
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_EXTREMA"
TYPES = {
    "float16": ("half", 16, 0x7C00),
    "bfloat16": ("bfloat", 16, 0x7F80),
    "float32": ("float", 32, 0x7F800000),
}
OPERATIONS = ("Minimum", "Maximum")


def _entry(dtype, operation):
    return f"vv_{operation}{dtype}"


def _pairs(dtype):
    _, bits, infinity = TYPES[dtype]
    sign = 1 << (bits - 1)
    fraction_bits = 10 if dtype == "float16" else 7 if dtype == "bfloat16" else 23
    one = (
        0x3C00 if dtype == "float16" else 0x3F80 if dtype == "bfloat16" else 0x3F800000
    )
    edges = [
        0,
        1,
        (1 << fraction_bits) - 1,
        1 << fraction_bits,
        one - 1,
        one,
        one + 1,
        infinity - 1,
        infinity,
        infinity + 1,
        infinity | (1 << (fraction_bits - 1)),
        infinity | (1 << (fraction_bits - 1)) | 7,
    ]
    words = edges + [word | sign for word in edges]
    pairs = list(itertools.product(words, repeat=2))
    rng = random.Random(600 + bits)
    pairs.extend((rng.getrandbits(bits), rng.getrandbits(bits)) for _ in range(4096))
    return pairs


def _expected(a, b, dtype, operation):
    _, bits, infinity = TYPES[dtype]
    sign = 1 << (bits - 1)
    magnitude = sign - 1
    if a & magnitude > infinity:
        return a
    if b & magnitude > infinity:
        return b
    x, y = a, b
    if dtype != "float16":
        x, y = (word & sign if word & infinity == 0 else word for word in (x, y))
    x, y = (0 if word & magnitude == 0 else word for word in (x, y))
    mask = (1 << bits) - 1
    x, y = ((~word & mask) if word & sign else (word | sign) for word in (x, y))
    choose_a = x < y if operation == "Minimum" else x > y
    return a if choose_a else b


def _payload(dtype, target, words):
    if dtype == "float16":
        return _half_payload(target, words)
    if dtype == "bfloat16":
        return _bfloat_payload(target, words)
    return {
        "dtype": "float32",
        "encoding": "ieee754-binary32",
        "shape": [len(words)],
        "values": list(words),
    }


def _guard(dtype):
    return 0x35555555 if dtype == "float32" else 0x3555


def _check_words(actual, expected):
    assert len(actual) == len(expected)
    differences = [
        {"index": i, "expected": b, "actual": a}
        for i, (a, b) in enumerate(zip(actual, expected))
        if type(a) is not int or a != b
    ]
    assert not differences, differences[:20]


def _request(descriptor, package, dtype, target, pairs, expected):
    assert descriptor["target"] == target
    inputs = {
        name: _payload(dtype, target, words)
        for name, words in (
            ("a", [a for a, _ in pairs]),
            ("b", [b for _, b in pairs]),
            ("c", [_guard(dtype)] * len(expected)),
        )
    }
    (constant,) = (
        binding
        for binding in descriptor["bindings"]
        if binding["kind"] == "constant-buffer"
    )
    size = constant["scalarLayout"].get("memberName", constant["name"])
    inputs[size] = {"dtype": "uint32", "shape": [1], "values": [len(pairs)]}
    outputs = {"c": _payload(dtype, target, expected)}
    request = _dispatch_request(descriptor, package, inputs, outputs, len(pairs))
    return request, inputs, _bound_values(descriptor, outputs)


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_extrema_reference_preserves_zero_sign_nan_payload_and_comparison_policy(
    dtype, operation
):
    _, bits, infinity = TYPES[dtype]
    sign = 1 << (bits - 1)
    for a, b in (
        (0, sign),
        (sign, 0),
        (infinity + 1, 0),
        (0, infinity + 1),
        (infinity | sign | 1, infinity + 7),
    ):
        expected = a if a & (sign - 1) > infinity else b
        assert _expected(a, b, dtype, operation) == expected
    # Only comparisons flush subnormals; selected storage words stay unchanged.
    expected = 0 if dtype == "float16" and operation == "Minimum" else 1
    assert _expected(0, 1, dtype, operation) == expected
    pairs = _pairs(dtype)
    assert len(pairs) == 4672 and pairs == _pairs(dtype)
    assert (0, 1) in pairs and (1, 0) in pairs


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_extrema_request_retains_operand_bits_and_physical_storage(
    tmp_path, dtype, target
):
    typename = TYPES[dtype][0]
    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void selection(const device {typename}* a [[buffer(0)]],
                      const device {typename}* b [[buffer(1)]],
                      device {typename}* c [[buffer(2)]],
                      constant uint& size [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {{
    if (i < size) c[i] = a[i];
}}
"""
    _, descriptor, package = _package(
        tmp_path, target, typename, (1, 1, 1), source=source, software_subgroups=False
    )
    sign = 1 << (TYPES[dtype][1] - 1)
    pairs = [(sign, TYPES[dtype][2] + 1)]
    expected = [sign] + [_guard(dtype)] * 8
    request, inputs, outputs = _request(
        descriptor, package, dtype, target, pairs, expected
    )
    assert not request.execution_plan.diagnostics
    by_name = {value.name: value for value in request.fixture.inputs}
    for binding in descriptor["bindings"]:
        if binding["kind"] != "buffer":
            continue
        layout = binding["scalarLayout"]
        name = layout.get("memberName", binding["name"])
        value, payload = by_name[binding["name"]], inputs[name]
        assert value.dtype == payload["dtype"] and value.encoding == payload.get(
            "encoding"
        )
        assert list(value.values) == payload["values"]
        assert layout["elementSizeBytes"] == (
            4 if target == "opengl" else TYPES[dtype][1] // 8
        )
    (output,) = outputs.values()
    assert output == _payload(dtype, target, expected)


@pytest.mark.parametrize(
    "actual,expected",
    [
        ([0], [0x80000000]),
        ([0x7FC00001], [0x7FC00002]),
        ([0x7F800001], [0x7FC00001]),
        ([0, 0], [0, 0x3555]),
        ([False], [0]),
        ([], [0]),
    ],
)
def test_extrema_comparison_rejects_payload_value_and_guard_changes(actual, expected):
    with pytest.raises(AssertionError):
        _check_words(actual, expected)


def _original_metal(work, runner, library, dtype, operation, pairs, expected):
    buffers = []
    for index, words in enumerate(
        (
            [a for a, _ in pairs],
            [b for _, b in pairs],
            [_guard(dtype)] * len(expected),
            [len(pairs)],
        )
    ):
        path = work / f"original-input-{index}.bin"
        code = "I" if dtype == "float32" or index == 3 else "H"
        path.write_bytes(struct.pack(f"<{len(words)}{code}", *words))
        buffers.append(str(path))
    request = work / "original-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": buffers,
                "workgroupCount": [len(pairs), 1, 1],
                "workgroupSize": [1, 1, 1],
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    output = work / "original-readback"
    _run(
        [runner, library, _entry(dtype, operation), request, output],
        work,
        "original-execute",
    )
    code = "I" if dtype == "float32" else "H"
    actual = list(
        struct.unpack(f"<{len(expected)}{code}", (output / "buffer-2.bin").read_bytes())
    )
    _check_words(actual, expected)
    for index in (0, 1, 3):
        assert (output / f"buffer-{index}.bin").read_bytes() == Path(
            buffers[index]
        ).read_bytes()


def test_current_extrema_native_parity(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned native minimum/maximum")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    target = os.environ["CROSTL_MLX_CURRENT_TARGET"]
    assert target in {"directx", "opengl", "metal"}
    assert (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
        ).strip()
        == MLX_COMMIT
    )
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
    assert (
        hashlib.sha256((root / MLX_BINARY_SOURCE).read_bytes()).hexdigest()
        == MLX_BINARY_SHA256
    )
    runner = library = None
    if target == "metal":
        original = tmp_path / "source-control"
        original.mkdir()
        runner = original / "readback"
        _run(
            [
                "xcrun",
                "swiftc",
                ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
                "-o",
                runner,
            ],
            original,
            "build-runner",
        )
        library = _metal_library(
            root / MLX_BINARY_SOURCE,
            original / "original.metallib",
            root,
            upstream=True,
        )
    for dtype, operation in itertools.product(TYPES, OPERATIONS):
        entry = _entry(dtype, operation)
        pairs = _pairs(dtype)
        expected = [_expected(a, b, dtype, operation) for a, b in pairs] + [
            _guard(dtype)
        ] * 8
        workload = next(w for w in BINARY_OPENGL_WORKLOADS if w.entry_point == entry)
        with tempfile.TemporaryDirectory(
            prefix=".current-extrema-", dir=root
        ) as directory:
            work = Path(directory)
            try:
                config_path = work / "crosstl.toml"
                config_path.write_text(_project_config(workload), encoding="utf-8")
                report = translate_project(
                    load_project_config(root, config_path),
                    targets=(target,),
                    output_dir=work.name + "/out",
                    format_output=False,
                )
                report.write_json(work / "report.json")
                data = report.to_json()
                assert (
                    data["summary"]["translatedCount"] == 1
                    and data["summary"]["failedCount"] == 0
                ), data["diagnostics"]
                assert not data["diagnostics"]
                (artifact,) = data["artifacts"]
                assert artifact["provenance"].get("binary32ComparisonProfile") == (
                    None if dtype == "float16" else "flush-subnormals"
                )
                descriptor, package = _prepare_native_package(report, work)
                source = package / descriptor["artifact"]["packagePath"]
                assert (
                    hashlib.sha256(source.read_bytes()).hexdigest()
                    == artifact["generatedHash"]["value"]
                )
                _validate(source, work, target)
                request, inputs, outputs = _request(
                    descriptor, package, dtype, target, pairs, expected
                )
                assert not request.execution_plan.diagnostics
                (work / "values.json").write_text(
                    json.dumps(
                        {"pairs": pairs, "inputs": inputs, "outputs": outputs}, indent=2
                    ),
                    encoding="utf-8",
                )
                executor = _executor(target)
                try:
                    availability = executor.is_available(request)
                    assert availability.available, availability.reason
                    result = executor.run(request)
                finally:
                    close = getattr(executor.runtime_adapter.runtime, "close", None)
                    if close:
                        close()
                (work / "result.json").write_text(
                    json.dumps(
                        {
                            "status": result.status,
                            "outputs": result.outputs,
                            "details": result.details,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                assert result.status == "ok"
                (actual,), (wanted,) = result.outputs.values(), outputs.values()
                assert {k: v for k, v in actual.items() if k != "values"} == {
                    k: v for k, v in wanted.items() if k != "values"
                }
                _check_words(actual["values"], wanted["values"])
                if target == "metal":
                    _original_metal(
                        work, runner, library, dtype, operation, pairs, expected
                    )
                (work / "evidence.json").write_text(
                    json.dumps(
                        {
                            "commit": MLX_COMMIT,
                            "sourceSha256": MLX_BINARY_SHA256,
                            "entryPoint": entry,
                            "target": target,
                            "dtype": dtype,
                            "operation": operation,
                            "artifactSha256": artifact["generatedHash"]["value"],
                            "comparisonProfile": (
                                None if dtype == "float16" else "flush-subnormals"
                            ),
                            "pairCount": len(pairs),
                            "guardCount": 8,
                            "comparison": "exact selected operand words",
                            "sourceControlInputsVerified": target == "metal",
                            "wholeFamilyParity": False,
                            "physicalDriverUploadBytes": False,
                            "originalLibrarySha256": (
                                hashlib.sha256(library.read_bytes()).hexdigest()
                                if library
                                else None
                            ),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            finally:
                shutil.copytree(work, tmp_path / entry, dirs_exist_ok=True)


def test_ci_requires_extrema_once_per_native_target():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned native binary math"
    )
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    path = "demos/integrations/mlx/tests/kernels/test_current_extrema.py"
    assert workflow.count(path) == 1
    assert path in step and "pytest -q -n auto" in step
    assert "--timeout-seconds 900" in step
