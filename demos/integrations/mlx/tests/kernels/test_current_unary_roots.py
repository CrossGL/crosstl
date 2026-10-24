"""Pinned MLX root functions with explicit operand-underflow contracts."""

import hashlib
import json
import os
import random
import shutil
import struct
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import (
    load_project_config,
    reflect_target_host_interface,
    translate_project,
)
from demos.integrations.mlx.tests.kernels import test_unary_native_loader as unary
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_magnitude import _samples
from demos.integrations.mlx.tests.kernels.test_current_complex_power import MLX_COMMIT
from demos.integrations.mlx.tests.kernels.test_current_unary_fma import (
    _run_current_unary,
)
from tests import workflow_text
from tests.test_translator import test_metal_precise_rsqrt as rsqrt
from tests.test_translator import test_metal_precise_sqrt as sqrt
from tests.test_translator.test_metal_precise_trig import _bits, _float

ROOT = Path(__file__).resolve().parents[5]
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_UNARY_ROOTS"
TYPES = ("float16", "bfloat16", "float32", "complex64")
PROFILES = {
    "binary32_sqrt_profile": "flush-subnormals",
    "binary32_rsqrt_profile": "flush-subnormals",
}
PROVENANCE = {
    "binary32SqrtProfile": "flush-subnormals",
    "binary32RsqrtProfile": "flush-subnormals",
}


def _format(dtype):
    if dtype in {"float16", "bfloat16"}:
        return "H", 0x7FFF, 0x7C00 if dtype == "float16" else 0x7F80
    assert dtype in {"float32", "complex64"}
    return "I", 0x7FFFFFFF, 0x7F800000


def _inputs(dtype):
    if dtype in {"float16", "bfloat16"}:
        return list(range(65536))
    if dtype == "complex64":
        return [word for pair in _samples() for word in pair]
    assert dtype == "float32"
    generator = random.Random(3401)
    return [
        0,
        0x80000000,
        1,
        0x7FFFFF,
        0x800000,
        0x3F800000,
        0xBF800000,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
    ] + [generator.getrandbits(32) for _ in range(65536)]


def _pack(words, dtype):
    return struct.pack(f"<{len(words)}{_format(dtype)[0]}", *words)


def _guards(dtype):
    return (
        (0x3555, 0xB555) * 8
        if _format(dtype)[0] == "H"
        else (0x35555555, 0xB5555555) * 8
    )


def _fills(dtype):
    return (0x3555, 0x4AAA) if _format(dtype)[0] == "H" else (0x4EAA5555, 0x4D55AAAA)


def _assert_words(actual, expected, dtype):
    assert len(actual) == len(expected), "Output length changed"
    _, mask, infinity = _format(dtype)
    mismatches = [
        (index, hex(got), hex(want))
        for index, (got, want) in enumerate(zip(actual, expected, strict=True))
        if got != want and not ((got & mask) > infinity and (want & mask) > infinity)
    ]
    assert not mismatches, f"{len(mismatches)} mismatched words: {mismatches[:12]}"


def _check_readback(payloads, readbacks, dtype):
    assert set(readbacks) == set(payloads), "Missing readback"
    for name, payload in payloads.items():
        assert len(readbacks[name]) == len(payload), f"Truncated readback: {name}"
        if name != "out_" and readbacks[name] != payload:
            # Keep exact byte equality without pytest expanding whole buffers.
            raise AssertionError(f"Read-only buffer changed: {name}")
    code = _format(dtype)[0]
    raw = readbacks["out_"]
    words = struct.unpack(f"<{len(raw) // struct.calcsize(code)}{code}", raw)
    guards = _guards(dtype)
    assert words[-len(guards) :] == guards, "Output guard changed"
    return words[: -len(guards)]


def _execute(runner, library, entry, directory, payloads, count, resources, dtype):
    directory.mkdir(parents=True)
    assert [(row["binding"], row["name"]) for row in resources] == [
        (0, "in_"),
        (1, "out_"),
        (2, "size"),
    ]
    buffers = []
    for row in resources:
        path = directory / f"input-{row['binding']}.bin"
        path.write_bytes(payloads[row["name"]])
        buffers.append(str(path))
    request = directory / "request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": buffers,
                "workgroupCount": [count, 1, 1],
                "workgroupSize": [1, 1, 1],
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    _run([runner, library, entry, request, directory], directory, "execute")
    readbacks = {
        row["name"]: (directory / f"buffer-{row['binding']}.bin").read_bytes()
        for row in resources
    }
    return _check_readback(payloads, readbacks, dtype)


@pytest.mark.parametrize(
    "path",
    [
        unary.UNARY_METAL_CONTRACT_PATH,
        unary.SCALAR_UNARY_METAL_CONTRACT_PATH,
    ],
)
def test_root_contracts_record_explicit_operand_policy(path):
    contract = json.loads(path.read_text())
    assert {
        name: contract["sourceOptionsByOperator"][name] for name in ("Sqrt", "Rsqrt")
    } == {"Sqrt": PROFILES, "Rsqrt": PROFILES}
    assert unary.ROOT_SOURCE_OPTIONS == PROFILES


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("operator", ("Sqrt", "Rsqrt", "Square"))
def test_root_policy_is_scoped_to_metal_root_selections(tmp_path, target, operator):
    workload = next(
        row
        for row in unary.UNARY_METAL_WORKLOADS
        if row.entry_point == f"v_{operator}float32float32"
    )
    path = tmp_path / "crosstl.toml"
    path.write_text(unary._project_config(target, workload), encoding="utf-8")
    options = load_project_config(tmp_path, path).source_options["metal"]
    actual = {key: options[key] for key in PROFILES if key in options}
    assert actual == (PROFILES if target == "metal" and operator != "Square" else {})


@pytest.mark.parametrize("dtype", TYPES)
def test_root_inputs_and_readback_checks(dtype):
    values = _inputs(dtype)
    expected_count = (
        8840 if dtype == "complex64" else 65546 if dtype == "float32" else 65536
    )
    assert len(values) == expected_count and values == _inputs(dtype)
    if dtype in {"float16", "bfloat16"}:
        assert set(values) == set(range(65536))
    guards = list(_guards(dtype))
    payloads = {
        "in_": _pack(values, dtype),
        "out_": _pack([3, 4] + guards, dtype),
        "size": struct.pack("<I", 2),
    }
    assert _check_readback(payloads, payloads, dtype) == (3, 4)
    for mutation in ("input", "constant", "guard", "truncated", "missing"):
        readbacks = dict(payloads)
        if mutation == "input":
            readbacks["in_"] = b"\0" * len(readbacks["in_"])
        elif mutation == "constant":
            readbacks["size"] = struct.pack("<I", 1)
        elif mutation == "guard":
            readbacks["out_"] = _pack([3, 4] + guards[:-1] + [0], dtype)
        elif mutation == "truncated":
            readbacks["out_"] = readbacks["out_"][:-1]
        else:
            del readbacks["in_"]
        with pytest.raises(AssertionError) as error:
            _check_readback(payloads, readbacks, dtype)
        if mutation in {"input", "constant"}:
            name = "in_" if mutation == "input" else "size"
            assert str(error.value) == f"Read-only buffer changed: {name}"


@pytest.mark.parametrize("dtype", TYPES)
def test_root_comparison_rejects_wrong_or_unwritten_results(dtype):
    _, mask, infinity = _format(dtype)
    _assert_words([infinity + 1], [infinity + 2], dtype)
    for left, right in (
        ([1], [2]),
        ([0], [mask + 1]),
        ([infinity], [infinity + 1]),
        ([1], []),
        ([_fills(dtype)[0]], [_fills(dtype)[1]]),
    ):
        with pytest.raises(AssertionError):
            _assert_words(left, right, dtype)
    # Every finite 16-bit fill can also be a legitimate result. Two distinct
    # initializations detect unwritten outputs without banning those values.
    _assert_words(list(_fills(dtype)), list(_fills(dtype)), dtype)


def test_ci_requires_root_execution():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = (
        workflow_text.workflow_job_text(workflow, "mlx-metal-porting")
        .split("      - name: Validate pinned unary roots and logarithms\n", 1)[1]
        .split("      - name:", 1)[0]
    )
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "demos/integrations/mlx/tests/kernels/test_current_unary_roots.py" in step
    assert "--timeout-seconds 300" in step and "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step


@pytest.fixture(scope="module")
def root_source():
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned root execution")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    assert target in {"metal", "directx", "opengl"}, "Missing or invalid target"
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=30
        ).strip()
        == MLX_COMMIT
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "mlx/backend/metal/kernels"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert (
        hashlib.sha256((root / unary.MLX_UNARY_SOURCE).read_bytes()).hexdigest()
        == unary.MLX_UNARY_SHA256
    )
    return root


@pytest.mark.parametrize("target", (None, "unknown"))
def test_root_execution_requires_valid_target(monkeypatch, target):
    monkeypatch.setenv(REQUIRE_ENV, "1")
    if target is None:
        monkeypatch.delenv("CROSTL_MLX_CURRENT_TARGET", raising=False)
    else:
        monkeypatch.setenv("CROSTL_MLX_CURRENT_TARGET", target)
    with pytest.raises(AssertionError, match="Missing or invalid target"):
        root_source.__wrapped__()


@pytest.mark.parametrize("operation, oracle", (("Sqrt", sqrt), ("Rsqrt", rsqrt)))
def test_current_roots_native_loader(root_source, tmp_path, operation, oracle):
    inputs = oracle._inputs()
    expected = [oracle._oracle(word, "flush-subnormals") for word in inputs]
    actual = _run_current_unary(
        tmp_path,
        operation,
        [_float(word) for word in inputs],
        [_float(word) for word in expected],
        0,
        0,
        f"metal_precise_{operation.lower()}_float",
        source_options=PROFILES,
    )
    _assert_words([_bits(value) for value in actual], expected, "float32")
    evidence = tmp_path / "evidence"
    for path in (
        evidence / "report.json",
        evidence / "artifacts.json",
        evidence / "package/runtime-package.json",
    ):
        artifacts = json.loads(path.read_text())["artifacts"]
        assert len(artifacts) == 1
        assert {
            key: artifacts[0]["provenance"][key] for key in PROVENANCE
        } == PROVENANCE
    (evidence / "exact-root-parity.json").write_text(
        json.dumps(
            {
                "commit": MLX_COMMIT,
                "operation": operation,
                "target": os.environ["CROSTL_MLX_CURRENT_TARGET"],
                "sourceOptions": PROFILES,
                "computedValues": len(inputs),
                "nonNaNWordsExact": True,
                "nanClassificationOnly": True,
                "includesSignedZeros": True,
                "fullTranslatedBackend": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def root_reference(root_source, tmp_path_factory):
    if os.environ["CROSTL_MLX_CURRENT_TARGET"] != "metal":
        pytest.skip("Original-source raw-buffer comparison requires native Metal")
    work = tmp_path_factory.mktemp("root-original")
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
    original = _metal_library(
        root_source / unary.MLX_UNARY_SOURCE,
        work / "original.metallib",
        root_source,
        upstream=True,
    )
    return runner, original


@pytest.mark.parametrize("operation", ("Sqrt", "Rsqrt"))
def test_current_roots_original_metal_parity(
    root_source, root_reference, tmp_path, operation
):
    root = root_source
    runner, original = root_reference
    entries = [f"v_{operation}{dtype}{dtype}" for dtype in TYPES]
    workloads = {row.entry_point: row for row in unary.UNARY_METAL_WORKLOADS}
    with tempfile.TemporaryDirectory(prefix=".current-unary-roots-", dir=root) as path:
        work = Path(path)
        try:
            config_path = work / "crosstl.toml"
            config_path.write_text(
                unary._project_config("metal", workloads[entries[0]]), encoding="utf-8"
            )
            config = replace(
                load_project_config(root, config_path),
                entry_points={unary.MLX_UNARY_SOURCE: entries},
                entry_workgroup_size_rules={unary.MLX_UNARY_SOURCE: {"*": (1, 1, 1)}},
            )
            report = translate_project(
                config, output_dir=work.name + "/out", format_output=False
            )
            report.write_json(work / "report.json")
            payload = report.to_json()
            assert (
                payload["summary"]["translatedCount"] == 4
                and payload["summary"]["failedCount"] == 0
            ), payload["diagnostics"]
            assert not [
                row for row in payload["diagnostics"] if row["severity"] == "error"
            ]
            artifacts = {
                row["entryPoint"]["source"]: row for row in payload["artifacts"]
            }
            assert len(payload["artifacts"]) == 4 and set(artifacts) == set(entries)
            for dtype, entry in zip(TYPES, entries, strict=True):
                directory = work / entry
                directory.mkdir()
                artifact = artifacts[entry]
                source = root / artifact["path"]
                assert (
                    hashlib.sha256(source.read_bytes()).hexdigest()
                    == artifact["generatedHash"]["value"]
                )
                assert artifact["sourceHash"]["value"] == unary.MLX_UNARY_SHA256
                assert {
                    key: artifact["provenance"][key] for key in PROVENANCE
                } == PROVENANCE
                assert artifact["templateMaterialization"][
                    "specializations"
                ] == unary._expected_unary_materializations(workloads[entry])
                interface = reflect_target_host_interface(source, target="metal")
                assert interface["status"] == "ready" and not interface["diagnostics"]
                resources = sorted(
                    interface["resources"], key=lambda row: row["binding"]
                )
                generated = _metal_library(
                    source, directory / "generated.metallib", root
                )
                values = _inputs(dtype)
                count = len(values) // (2 if dtype == "complex64" else 1)
                previous = {}
                for index, fill in enumerate(_fills(dtype)):
                    payloads = {
                        "in_": _pack(values + list(_guards(dtype)), dtype),
                        "out_": _pack(
                            [fill] * len(values) + list(_guards(dtype)), dtype
                        ),
                        "size": struct.pack("<I", count),
                    }
                    results = {}
                    for label, library, name in (
                        ("original", original, entry),
                        ("translated", generated, artifact["entryPoint"]["target"]),
                    ):
                        results[label] = _execute(
                            runner,
                            library,
                            name,
                            directory / str(index) / label,
                            payloads,
                            count,
                            resources,
                            dtype,
                        )
                        if label in previous:
                            _assert_words(results[label], previous[label], dtype)
                    _assert_words(results["translated"], results["original"], dtype)
                    previous = results
                (directory / "evidence.json").write_text(
                    json.dumps(
                        {
                            "commit": MLX_COMMIT,
                            "entryPoint": entry,
                            "sourceOptions": PROFILES,
                            "computedValuesPerInitialization": count,
                            "initializations": 2,
                            "guardWordsPerPath": len(_guards(dtype)) * 2,
                            "all16BitPatterns": dtype in {"float16", "bfloat16"},
                            "comparison": "exact-words-except-nan-payloads",
                            "readonlyBuffersUnchanged": True,
                            "generatedSourceSHA256": (
                                hashlib.sha256(source.read_bytes()).hexdigest()
                            ),
                            "generatedLibrarySHA256": (
                                hashlib.sha256(generated.read_bytes()).hexdigest()
                            ),
                            "originalLibrarySHA256": (
                                hashlib.sha256(original.read_bytes()).hexdigest()
                            ),
                            "fullUpstreamSuite": False,
                            "fullTranslatedBackend": False,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)
