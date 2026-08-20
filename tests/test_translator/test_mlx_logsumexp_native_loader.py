from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

from crosstl.project import (
    DirectXComputeRuntime,
    DirectXRuntimeParityAdapter,
    RuntimeParityExecutor,
    RuntimeTestAdapterSpec,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
)

ROOT = Path(__file__).resolve().parents[2]
MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
MLX_LOGSUMEXP_SOURCE = "mlx/backend/metal/kernels/logsumexp.metal"
MLX_LOGSUMEXP_SHA256 = (
    "f9bec5e1e5a23d20bedf9ff8d29a8c03bbb5144bc5d751bbfe906d32ee894817"
)
MLX_LOGSUMEXP_ENTRY = "block_logsumexp_float32"
MLX_LOGSUMEXP_AXIS_32_ARTIFACT = (
    "sha256:f51ab67b8a9ad3240e3fbb52f6de00cdf4a8532e58790758e59aa50fb95e2c52"
)
MLX_LOGSUMEXP_DISPATCH_CONTRACT = (
    ROOT / "demos" / "integrations" / "mlx" / "contracts" / "logsumexp.dispatch.json"
)
REQUIRE_PROOF_ENV = "CROSTL_REQUIRE_MLX_LOGSUMEXP_DIRECTX_NATIVE_LOADER"


def _project_config(*, output_dir: str, dispatch_contract: str) -> str:
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_LOGSUMEXP_SOURCE}"]
        include_dirs = ["."]
        targets = ["directx"]
        output_dir = "{output_dir}"
        dispatch_contracts = ["{dispatch_contract}"]

        [project.sources]
        "**/*.metal" = "metal"
        """).strip()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _skip_or_fail(message: str) -> None:
    if os.environ.get(REQUIRE_PROOF_ENV) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if os.environ.get(REQUIRE_PROOF_ENV) == "1":
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")

    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_LOGSUMEXP_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX LogSumExp source is missing: {source_path}")

    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_LOGSUMEXP_SHA256
    return mlx_root


def _expected_scalar_layouts() -> dict[str, dict]:
    runtime_array = {
        "physicalType": "float",
        "elementType": "float32",
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
        "alignmentBytes": 4,
        "memberOffsetBytes": 0,
        "storageLayout": "hlsl-structured-buffer",
        "runtimeSized": True,
    }
    return {
        "block_logsumexp_float32_axis_size_Constants": {
            "physicalType": "int",
            "elementType": "int32",
            "elementSizeBytes": 4,
            "elementStrideBytes": 4,
            "alignmentBytes": 16,
            "memberOffsetBytes": 0,
            "storageLayout": "hlsl-constant-buffer",
            "runtimeSized": False,
            "memberName": "block_logsumexp_float32_axis_size",
            "blockSizeBytes": 16,
        },
        "in_": dict(runtime_array),
        "out_": dict(runtime_array),
    }


def _build_runtime_package(mlx_root: Path, work_dir: Path) -> tuple[dict, Path]:
    contract_path = work_dir / "logsumexp.dispatch.json"
    shutil.copyfile(MLX_LOGSUMEXP_DISPATCH_CONTRACT, contract_path)
    output_dir = work_dir / "out"
    config_path = work_dir / "crosstl.toml"
    config_path.write_text(
        _project_config(
            output_dir=output_dir.relative_to(mlx_root).as_posix(),
            dispatch_contract=contract_path.relative_to(mlx_root).as_posix(),
        )
        + "\n",
        encoding="utf-8",
    )
    report = translate_project(
        load_project_config(mlx_root, config_path),
        targets=("directx",),
        output_dir=output_dir.relative_to(mlx_root).as_posix(),
        format_output=False,
        validate=True,
    )
    report_payload = report.to_json()

    assert report_payload["summary"]["unitCount"] == 1
    assert report_payload["summary"]["translatedCount"] == 2
    assert report_payload["summary"]["failedCount"] == 0
    axis_32_artifact = next(
        artifact
        for artifact in report_payload["artifacts"]
        if artifact["dispatchArtifact"]["artifactId"] == MLX_LOGSUMEXP_AXIS_32_ARTIFACT
    )
    expected_variant = "dispatch-" + MLX_LOGSUMEXP_AXIS_32_ARTIFACT.removeprefix(
        "sha256:"
    )
    assert axis_32_artifact["source"] == MLX_LOGSUMEXP_SOURCE
    assert axis_32_artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_LOGSUMEXP_SHA256,
    }
    assert axis_32_artifact["variant"] == expected_variant
    assert axis_32_artifact["entryPoint"] == {
        "source": MLX_LOGSUMEXP_ENTRY,
        "target": "CSMain",
        "stage": "compute",
    }
    assert axis_32_artifact["dispatchArtifact"]["workgroupSize"] == [32, 1, 1]
    assert axis_32_artifact["dispatchArtifact"]["subgroupWidth"] == 32
    assert not axis_32_artifact.get("specializationConstants")

    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
    runtime_artifacts = build_runtime_artifact_manifest(report_path)
    assert runtime_artifacts["success"] is True, json.dumps(runtime_artifacts, indent=2)
    assert runtime_artifacts["summary"]["artifactCount"] == 2
    assert runtime_artifacts["summary"]["specializationConstantCount"] == 0

    reflected_artifact = next(
        artifact
        for artifact in runtime_artifacts["artifacts"]
        if artifact["variant"] == expected_variant
    )
    assert reflected_artifact["hostInterface"]["status"] == "ready"
    resources = reflected_artifact["hostInterface"]["resources"]
    assert {resource["name"]: resource["binding"] for resource in resources} == {
        "block_logsumexp_float32_axis_size_Constants": 2,
        "in_": 0,
        "out_": 1,
    }
    assert {resource["name"]: resource["access"] for resource in resources} == {
        "block_logsumexp_float32_axis_size_Constants": "read",
        "in_": "read",
        "out_": "read_write",
    }
    reflected_layouts = {
        resource["name"]: resource["scalarLayout"] for resource in resources
    }
    assert reflected_layouts == _expected_scalar_layouts()

    runtime_artifacts_path = work_dir / "runtime-artifacts.json"
    _write_json(runtime_artifacts_path, runtime_artifacts)
    package_dir = work_dir / "runtime-package"
    package = build_runtime_package(runtime_artifacts_path, package_dir)
    assert package["success"] is True, json.dumps(package, indent=2)

    loader_manifest = build_runtime_loader_manifest(
        package_dir / "runtime-package.json"
    )
    assert loader_manifest["success"] is True, json.dumps(loader_manifest, indent=2)
    assert loader_manifest["summary"]["readyLoadUnitCount"] == 2
    assert loader_manifest["summary"]["blockedLoadUnitCount"] == 0
    load_unit = next(
        unit
        for unit in loader_manifest["loadUnits"]
        if unit["variant"] == expected_variant
    )
    descriptor = build_native_loader_abi_descriptor(
        loader_manifest,
        load_unit_id=load_unit["id"],
    )
    assert descriptor["target"] == "directx"
    assert descriptor["entryPoint"]["name"] == "CSMain"
    assert descriptor["entryPoint"]["executionConfig"] == {
        "numthreads": [32, 1, 1],
        "subgroupWidth": 32,
    }
    assert descriptor["specializationConstants"] == []
    descriptor_layouts = {
        binding["name"]: binding["scalarLayout"] for binding in descriptor["bindings"]
    }
    assert descriptor_layouts == _expected_scalar_layouts()
    return descriptor, package_dir


def test_pinned_mlx_logsumexp_executes_through_directx_native_loader():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-logsumexp-directx-native-loader-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
        )
        input_values = [(index - 16) / 8.0 for index in range(32)]
        maximum = max(input_values)
        expected_value = maximum + math.log(
            math.fsum(math.exp(value - maximum) for value in input_values)
        )
        request = build_native_loader_dispatch_request(
            descriptor,
            package_dir,
            {
                "block_logsumexp_float32_axis_size_Constants": {
                    "dtype": "int32",
                    "shape": [1],
                    "values": [32],
                },
                "in_": {
                    "dtype": "float32",
                    "shape": [32],
                    "values": input_values,
                },
            },
            {
                "out_": {
                    "dtype": "float32",
                    "shape": [1],
                    "values": [expected_value],
                    "tolerance": {"absolute": 5e-5, "relative": 5e-5},
                }
            },
            (1, 1, 1),
            expected_target="directx",
        )
        assert request.execution_plan is not None
        assert request.execution_plan.diagnostics == ()
        assert request.execution_plan.dispatch.workgroup_size == (32, 1, 1)
        assert request.execution_plan.dispatch.workgroup_count == (1, 1, 1)
        assert request.execution_plan.dispatch.global_size == (32, 1, 1)

        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id="mlx-logsumexp-directx-native-loader",
                target="directx",
                executor="directx",
                adapter_kind="directx-native-runtime",
            ),
            runtime_adapter=DirectXRuntimeParityAdapter(
                runtime=DirectXComputeRuntime()
            ),
        )
        availability = executor.is_available(request)
        if not availability.available:
            _skip_or_fail(
                availability.reason or "The native DirectX runtime is unavailable"
            )

        result = executor.run(request)

    assert result.status == "ok"
    assert result.outputs["out_"]["dtype"] == "float32"
    assert result.outputs["out_"]["shape"] == [1]
    assert result.outputs["out_"]["values"] == pytest.approx(
        [expected_value],
        abs=5e-5,
        rel=5e-5,
    )
