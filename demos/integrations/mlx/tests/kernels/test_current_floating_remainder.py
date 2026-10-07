"""Pinned floating remainder through generated packages and native source controls."""

import hashlib
import json
import os
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
from tests.runtime_helpers import _prepare_native_package, _validate
from tests.test_translator.test_bfloat_buffer_runtime import _storage as _bfloat_payload
from tests.test_translator.test_boolean_buffer_runtime import (
    _bound_values,
)
from tests.test_translator.test_boolean_buffer_runtime import (
    _request as _dispatch_request,
)
from tests.test_translator.test_floating_remainder_math import _pairs as _float_pairs
from tests.test_translator.test_metal_additive_profile import _oracle as _add
from tests.test_translator.test_metal_comparison_profile import _oracle as _compare
from tests.test_translator.test_metal_division import _bfloat
from tests.test_translator.test_metal_remainder_profile import _oracle as _remainder
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_product import _package
from tools import ci_coverage

ROOT = Path(__file__).resolve().parents[5]
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_FLOATING_REMAINDER"
ENTRIES = {"float32": "vv_Remainderfloat32", "bfloat16": "vv_Remainderbfloat16"}
PROVENANCE = {
    "binary32ComparisonProfile": "flush-subnormals",
    "binary32RemainderProfile": "flush-arithmetic-subnormals",
    "binary32AdditiveProfile": "rne-flush",
}


def _pairs(dtype):
    pairs = _float_pairs(32)
    if dtype == "float32":
        return pairs + [
            (0x00800000, 0x80800001),
            (0x80800000, 0x00800001),
            (0x81E657A8, 0x01F010C6),
        ]
    pairs = [tuple(_bfloat(word) >> 16 for word in pair) for pair in pairs]
    pairs.extend((a, (a + 1) ^ 0x8000) for a in range(0x80, 0x180))
    pairs.extend((a ^ 0x8000, a + 1) for a in range(0x80, 0x180))
    return pairs


def _expected(a, b, dtype):
    if dtype == "bfloat16":
        a, b = a << 16, b << 16
    result = _remainder(a, b, True)
    if dtype == "bfloat16":
        result = _bfloat(result)
    if _compare(result, 0, "ne", True) and _compare(result, 0, "lt", True) != _compare(
        b, 0, "lt", True
    ):
        result = _add(result, b, True)
    return _bfloat(result) >> 16 if dtype == "bfloat16" else result


def _payload(dtype, target, words):
    if dtype == "bfloat16":
        return _bfloat_payload(target, words)
    return {
        "dtype": "float32",
        "encoding": "ieee754-binary32",
        "shape": [len(words)],
        "values": list(words),
    }


def _guard(dtype):
    return 0x3555 if dtype == "bfloat16" else 0x35555555


def _check_words(actual, expected, dtype, target):
    assert len(actual) == len(expected)
    narrow = dtype == "bfloat16" and target != "opengl"
    width = 16 if narrow else 32
    mismatches = []
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert type(got) is int and 0 <= got < 1 << width
        a, b = (got << 16, want << 16) if narrow else (got, want)
        both_nan = (a & 0x7FFFFFFF) > 0x7F800000 and (b & 0x7FFFFFFF) > 0x7F800000
        if got != want and not both_nan:
            mismatches.append({"index": index, "expected": want, "actual": got})
    assert not mismatches, mismatches[:20]
    assert actual[-8:] == expected[-8:]


def _request(descriptor, package, dtype, target, pairs, expected):
    inputs = {
        name: _payload(dtype, target, words)
        for name, words in (
            ("a", [a for a, _ in pairs]),
            ("b", [b for _, b in pairs]),
            ("c", [_guard(dtype)] * len(expected)),
        )
    }
    size = ENTRIES[dtype] + "_size" if target == "directx" else "size"
    inputs[size] = {"dtype": "uint32", "shape": [1], "values": [len(pairs)]}
    outputs = {"c": _payload(dtype, target, expected)}
    request = _dispatch_request(descriptor, package, inputs, outputs, len(pairs))
    return request, inputs, _bound_values(descriptor, outputs)


@pytest.mark.parametrize(
    "dtype,a,b,expected",
    (
        ("float32", 0x00800000, 0x80800001, 0x80000000),
        ("float32", 0x80800000, 0x00800001, 0),
        ("float32", 0x81E657A8, 0x01F010C6, 0),
        ("bfloat16", 0x0080, 0x8081, 0x8000),
        ("bfloat16", 0x8080, 0x0081, 0),
    ),
)
def test_reference_retains_separate_remainder_comparison_and_addition(
    dtype, a, b, expected
):
    assert _expected(a, b, dtype) == expected
    assert (a, b) in _pairs(dtype)


@pytest.mark.parametrize("dtype", ENTRIES)
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_pinned_remainder_payloads_use_reflected_storage(tmp_path, dtype, target):
    typename = "float" if dtype == "float32" else "bfloat"
    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void {ENTRIES[dtype]}(const device {typename}* a [[buffer(0)]],
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
    pairs = [(0x80000000, 0x7FC00001)] if dtype == "float32" else [(0x8000, 0x7FC1)]
    expected = [pairs[0][0]] + [_guard(dtype)] * 8
    request, inputs, outputs = _request(
        descriptor, package, dtype, target, pairs, expected
    )
    assert not request.execution_plan.diagnostics
    by_name = {value.name: value for value in request.fixture.inputs}
    for binding in descriptor["bindings"]:
        layout = binding["scalarLayout"]
        name = layout.get("memberName", binding["name"])
        if name not in {"a", "b", "c"}:
            continue
        value, payload = by_name[binding["name"]], inputs[name]
        assert value.dtype == payload["dtype"]
        assert value.encoding == payload.get("encoding")
        assert list(value.values) == payload["values"]
        assert layout["elementSizeBytes"] == (
            2 if dtype == "bfloat16" and target != "opengl" else 4
        )
    (output,) = outputs.values()
    assert output == _payload(dtype, target, expected)


@pytest.mark.parametrize("dtype", ENTRIES)
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_remainder_comparison_rejects_wrong_zero_finite_values_and_guards(
    dtype, target
):
    narrow = dtype == "bfloat16" and target != "opengl"
    sign = 0x8000 if narrow else 0x80000000
    guard = _payload(dtype, target, [_guard(dtype)])["values"][0]
    for actual in ([sign] + [guard] * 8, [1] + [guard] * 8, [0] + [guard] * 7 + [0]):
        with pytest.raises(AssertionError):
            _check_words(actual, [0] + [guard] * 8, dtype, target)
    # These bfloat values are finite, not binary16 NaNs.
    finite = 0x7F7F if narrow else 0x7F7F0000
    with pytest.raises(AssertionError):
        _check_words([finite] + [guard] * 8, [finite - 1] + [guard] * 8, dtype, target)


def _original_metal(work, runner, library, dtype, pairs, expected):
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
        code = "H" if dtype == "bfloat16" and index != 3 else "I"
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
    _run([runner, library, ENTRIES[dtype], request, output], work, "original-execute")
    code = "H" if dtype == "bfloat16" else "I"
    actual = list(
        struct.unpack(f"<{len(expected)}{code}", (output / "buffer-2.bin").read_bytes())
    )
    _check_words(actual, expected, dtype, "metal")
    for index in (0, 1, 3):
        assert (output / f"buffer-{index}.bin").read_bytes() == Path(
            buffers[index]
        ).read_bytes()


def test_current_floating_remainder_native_parity(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned native floating remainder")
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
    for dtype, entry in ENTRIES.items():
        pairs = _pairs(dtype)
        expected = [_expected(a, b, dtype) for a, b in pairs] + [_guard(dtype)] * 8
        workload = next(w for w in BINARY_OPENGL_WORKLOADS if w.entry_point == entry)
        with tempfile.TemporaryDirectory(
            prefix=".current-floating-remainder-", dir=root
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
                for field, value in PROVENANCE.items():
                    assert artifact["provenance"][field] == value
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
                _check_words(actual["values"], wanted["values"], dtype, target)
                if target == "metal":
                    _original_metal(work, runner, library, dtype, pairs, expected)
                (work / "evidence.json").write_text(
                    json.dumps(
                        {
                            "commit": MLX_COMMIT,
                            "sourceSha256": MLX_BINARY_SHA256,
                            "entryPoint": entry,
                            "target": target,
                            "dtype": dtype,
                            "artifactSha256": artifact["generatedHash"]["value"],
                            "profiles": PROVENANCE,
                            "pairCount": len(pairs),
                            "guardCount": 8,
                            "comparison": "exact words; NaN classification",
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
                shutil.copytree(work, tmp_path / dtype, dirs_exist_ok=True)


def test_ci_requires_floating_remainder_once_per_native_target():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned native binary math"
    )
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    path = "demos/integrations/mlx/tests/kernels/test_current_floating_remainder.py"
    assert workflow.count(path) == 1
    assert path in step and "pytest -q -n auto" in step
    assert "--timeout-seconds 900" in step
