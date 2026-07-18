import json
import os
from pathlib import Path

import pytest

from crosstl._crosstl import translate
from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    OpenGLComputeRuntime,
)
from crosstl.project.runtime_verification import (
    DirectXRuntimeParityAdapter,
    OpenGLRuntimeParityAdapter,
    build_runtime_test_manifest,
    plan_runtime_test_manifest,
    verify_runtime_test_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = (
    ROOT / "tests" / "fixtures" / "runtime_verification" / "private_pointer_partition"
)
EXPECTED_VALUES = [100, 101, 102, 103, 200, 201, 202, 203]

GENUINE_UNAVAILABILITY_REASONS = {
    "backend-unavailable",
    "dependency-unavailable",
    "device-selection-failed",
    "device-unavailable",
    "direct3d-runtime-unavailable",
    "opengl-runtime-unavailable",
    "opengl-version-unsupported",
    "platform-unavailable",
    "tool-unavailable",
}


def _prepare_runtime_fixture(tmp_path: Path, target: str):
    source_path = FIXTURE_DIR / "private_pointer_partition.cgl"
    artifact_report = json.loads(
        (FIXTURE_DIR / f"{target}.artifacts.json").read_text(encoding="utf-8")
    )
    generated = translate(
        str(source_path),
        backend=target,
        format_output=False,
    )

    assert "values + chunk" not in generated
    if target == "directx":
        assert (
            "void write_partition(inout uint values[8], int values_base, int chunk)"
            in generated
        )
        assert "write_partition(values, (chunk * span), chunk);" in generated
        assert "[numthreads(1, 1, 1)]" in generated
    else:
        assert target == "opengl"
        assert (
            "void write_partition(inout uint values[8], int values_base, int chunk)"
            in generated
        )
        assert "write_partition(values, int((chunk * span)), chunk);" in generated
        assert (
            "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;"
            in generated
        )

    artifact_report["project"]["root"] = str(tmp_path)
    artifact_path = tmp_path / artifact_report["artifacts"][0]["path"]
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(generated, encoding="utf-8")
    manifest = build_runtime_test_manifest(
        artifact_report,
        FIXTURE_DIR / f"{target}.fixture-metadata.json",
        project_root=tmp_path,
    )
    assert manifest["success"] is True, json.dumps(manifest, indent=2)
    assert manifest["tests"][0]["expectedOutputs"][0]["values"] == EXPECTED_VALUES

    plan = plan_runtime_test_manifest(
        artifact_report,
        manifest,
        project_root=tmp_path,
    )
    assert plan["success"] is True, json.dumps(plan, indent=2)
    runtime_execution = plan["testCases"][0]["runtimeExecution"]
    assert [
        (
            item["binding"]["name"],
            item["binding"].get("set", 0),
            item["binding"]["binding"],
            item["source"],
        )
        for item in runtime_execution["resourceBindings"]
    ] == [("output", 0, 0, "expectedOutput")]
    assert runtime_execution["dispatch"] == {
        "entryPoint": "CSMain" if target == "directx" else "main",
        "workgroupSize": [1, 1, 1],
        "workgroupCount": [1, 1, 1],
        "globalSize": [1, 1, 1],
        "metadata": {
            "stage": "compute",
            "source": "fixture",
            "status": "available",
        },
    }
    return artifact_report, manifest


def _runtime_adapter(target: str):
    if target == "directx":
        return DirectXRuntimeParityAdapter(runtime=DirectXComputeRuntime())
    assert target == "opengl"
    return OpenGLRuntimeParityAdapter(
        runtime=OpenGLComputeRuntime(context_backends=("egl",))
    )


def _runtime_is_required(target: str) -> bool:
    requested = {
        item.strip().lower()
        for item in os.environ.get(
            "CROSTL_REQUIRE_PRIVATE_POINTER_PARTITION_RUNTIME", ""
        ).split(",")
        if item.strip()
    }
    return "all" in requested or target in requested


def _assert_native_readback(report: dict, target: str) -> None:
    result = report["results"][0]
    if result["status"] == "skipped":
        if _runtime_is_required(target):
            pytest.fail(
                f"{target} private-pointer partition runtime was required: "
                + json.dumps(report, indent=2)
            )
        assert result.get("failurePhase") == "platform-requirements", json.dumps(
            report, indent=2
        )
        executor = result.get("executor", {})
        details = executor.get("details", {})
        assert details.get("missingTools") or details.get(
            "missingEnvironment"
        ), json.dumps(report, indent=2)
        pytest.skip(executor.get("message") or f"{target} runtime is unavailable")

    if result["status"] == "unavailable":
        if _runtime_is_required(target):
            pytest.fail(
                f"{target} private-pointer partition runtime was required: "
                + json.dumps(report, indent=2)
            )
        executor = result.get("executor", {})
        reason_kind = executor.get("details", {}).get("reasonKind")
        assert reason_kind in GENUINE_UNAVAILABILITY_REASONS, json.dumps(
            report, indent=2
        )
        pytest.skip(executor.get("message") or f"{target} runtime is unavailable")

    assert report["success"] is True, json.dumps(report, indent=2)
    summary = report["summary"]
    assert summary["fixtureCount"] == 1, json.dumps(report, indent=2)
    assert summary["passedCount"] == 1, json.dumps(report, indent=2)
    assert summary["failedCount"] == 0, json.dumps(report, indent=2)
    assert summary["skippedCount"] == 0, json.dumps(report, indent=2)
    assert summary["unavailableCount"] == 0, json.dumps(report, indent=2)
    assert summary["runtimeFailedCount"] == 0, json.dumps(report, indent=2)
    assert summary["comparisonFailedCount"] == 0, json.dumps(report, indent=2)
    assert result["status"] == "passed", json.dumps(report, indent=2)
    assert result["comparisons"] == [
        {
            "name": "output",
            "kind": "buffer",
            "status": "passed",
            "tolerance": {"absolute": 0.0, "relative": 0.0},
            "expected": {"dtype": "uint32", "shape": [8]},
            "actual": {"dtype": "uint32", "shape": [8]},
            "mismatchCount": 0,
            "maxAbsoluteError": 0.0,
            "maxRelativeError": 0.0,
        }
    ]


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_private_pointer_partition_translation_and_manifest(tmp_path, target):
    _prepare_runtime_fixture(tmp_path, target)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_private_pointer_partition_native_readback(tmp_path, target):
    artifact_report, manifest = _prepare_runtime_fixture(tmp_path, target)
    report = verify_runtime_test_manifest(
        artifact_report,
        manifest,
        executors={target: _runtime_adapter(target)},
    )

    _assert_native_readback(report, target)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("status", ["skipped", "unavailable"])
def test_required_private_pointer_partition_runtime_fails_closed(
    monkeypatch, target, status
):
    monkeypatch.setenv(
        "CROSTL_REQUIRE_PRIVATE_POINTER_PARTITION_RUNTIME",
        target,
    )

    with pytest.raises(pytest.fail.Exception, match="runtime was required"):
        _assert_native_readback({"results": [{"status": status}]}, target)
