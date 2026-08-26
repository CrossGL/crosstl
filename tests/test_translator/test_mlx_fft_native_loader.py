from __future__ import annotations

import cmath
import hashlib
import json
import math
import os
import subprocess
import tempfile
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

MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
CURRENT_MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
MLX_FFT_SOURCE = "mlx/backend/metal/kernels/fft.metal"
MLX_FFT_SHA256 = "3a1fbb38ed64f50a49a20d0c5adb1748d9d06ea20e5931e99aa26be543cb7825"
CURRENT_MLX_FFT_SHA256 = (
    "c478eb84283bbdf585c0cb34b2bfde5b0fc32d1740c6ad76e8559698a57b8d2e"
)
MLX_FFT_ENTRY = "fft_mem_256_float2_float2"
MLX_FFT_GENERATED_SHA256 = (
    "07f9300c2e4860077b344610fbfaa2eadb330e1f9723cb519794f91272bd2289"
)
MLX_FFT_GENERATED_SIZE_BYTES = 116160
MLX_FFT_SIZE = 256
REQUIRE_PROOF_ENV = "CROSTL_REQUIRE_MLX_FFT_DIRECTX_NATIVE_LOADER"
CURRENT_MLX_ROOT_ENV = "CROSTL_MLX_CURRENT_ROOT"

FFT_SPECIALIZATION_CONSTANTS = {
    0: False,
    1: True,
    2: 4,
    3: 256,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 4,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    21: 0,
}
CURRENT_FFT_SPECIALIZATION_CONSTANTS = {
    **FFT_SPECIALIZATION_CONSTANTS,
    22: False,
}


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _project_config(
    output_dir: str,
    *,
    specialization_constants: dict[int, object] = FFT_SPECIALIZATION_CONSTANTS,
) -> str:
    constants = "\n".join(
        f'"{constant_id}" = {_toml_value(value)}'
        for constant_id, value in specialization_constants.items()
    )
    return f"""
[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{MLX_FFT_SOURCE}"]
include_dirs = ["."]
targets = ["directx"]
output_dir = "{output_dir}"

[project.sources]
"**/*.metal" = "metal"

[[project.index_range_assertions]]
source = "{MLX_FFT_SOURCE}"
expression = "batch_idx + index"
minimum = 0
maximum = 4294967295

[[project.index_range_assertions]]
source = "{MLX_FFT_SOURCE}"
expression = "batch_idx + index + 1"
minimum = 0
maximum = 4294967295

[[project.index_range_assertions]]
source = "{MLX_FFT_SOURCE}"
expression = "batch_idx + index + next_in"
minimum = 0
maximum = 4294967295

[[project.index_range_assertions]]
source = "{MLX_FFT_SOURCE}"
expression = "batch_idx + index + next_out"
minimum = 0
maximum = 4294967295

[[project.workgroup_access_assertions]]
source = "{MLX_FFT_SOURCE}"
entry_point = "{MLX_FFT_ENTRY}"
function = "*"
parameter = "*"
minimum = 0
maximum = 255

[project.entry_points]
"{MLX_FFT_SOURCE}" = "{MLX_FFT_ENTRY}"

[project.specialization_constants]
{constants}

[project.entry_workgroup_size_rules."{MLX_FFT_SOURCE}"]
"{MLX_FFT_ENTRY}" = [1, 1, 64]

[project.source_options.metal]
max_template_specializations = 4096
max_template_materialization_work = 2097152
""".strip()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if os.environ.get(REQUIRE_PROOF_ENV) == "1":
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")

    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_FFT_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX FFT source is missing: {source_path}")

    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_FFT_SHA256
    return mlx_root


def _current_mlx_root() -> Path:
    root_value = os.environ.get(CURRENT_MLX_ROOT_ENV)
    if not root_value:
        pytest.skip(f"{CURRENT_MLX_ROOT_ENV} is not configured")

    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_FFT_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Current MLX FFT source is missing: {source_path}")

    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == CURRENT_MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
        CURRENT_MLX_FFT_SHA256
    )
    return mlx_root


def test_current_mlx_fft_reports_directx_workgroup_alias_blocker():
    mlx_root = _current_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-fft-current-frontier-",
        dir=mlx_root,
    ) as temporary_directory:
        work_dir = Path(temporary_directory)
        output_dir = work_dir / "out"
        config_path = work_dir / "crosstl.toml"
        config_path.write_text(
            _project_config(
                output_dir.relative_to(mlx_root).as_posix(),
                specialization_constants=CURRENT_FFT_SPECIALIZATION_CONSTANTS,
            )
            + "\n",
            encoding="utf-8",
        )
        report = translate_project(
            load_project_config(mlx_root, config_path),
            targets=("directx",),
            output_dir=output_dir.relative_to(mlx_root).as_posix(),
            format_output=False,
            validate=False,
        )
        payload = report.to_json()

        assert payload["summary"]["unitCount"] == 1
        assert payload["summary"]["artifactCount"] == 1
        assert payload["summary"]["translatedCount"] == 0
        assert payload["summary"]["failedCount"] == 1
        assert payload["summary"]["diagnosticCounts"] == {
            "error": 1,
            "note": 0,
            "warning": 0,
        }
        artifact = payload["artifacts"][0]
        assert artifact["sourceHash"] == {
            "algorithm": "sha256",
            "value": CURRENT_MLX_FFT_SHA256,
        }
        assert artifact["sourceSizeBytes"] == 3436
        assert artifact["status"] == "failed"
        assert artifact["entryPoint"] == {
            "source": MLX_FFT_ENTRY,
            "target": MLX_FFT_ENTRY,
        }
        assert artifact["execution"]["entryPoints"][0]["workgroupSize"] == [
            1,
            1,
            64,
        ]
        assert artifact["specializationMaterialization"] == {
            "status": "concrete",
            "mode": "concrete-crossgl-variant",
            "source": "shared-crossgl-specialization",
            "constantCount": 21,
            "requiredCount": 21,
            "concreteCount": 21,
            "overriddenCount": 21,
            "targetSupportsDeferredSpecialization": False,
        }
        materialization = artifact["templateMaterialization"]
        assert materialization["status"] == "materialized"
        assert materialization["specializationCount"] == 37
        assert materialization["unsupported"] == []
        assert materialization["accounting"] == {
            "reachableSpecializationCount": 42,
            "dependencyDiscoveryWorkCount": 0,
            "prunedCandidateCount": 2120,
        }
        assert artifact["error"] == (
            "DirectX cannot emit a workgroup pointer as a first-class value: "
            "crosstl_ptr_buf"
        )
        assert not (mlx_root / artifact["path"]).exists()

        assert payload["diagnostics"] == [
            {
                "code": "project.translate.directx-workgroup-pointer-unsupported",
                "message": artifact["error"],
                "severity": "error",
                "sourceBackend": "metal",
                "target": "directx",
                "location": {
                    "file": MLX_FFT_SOURCE,
                    "line": 1,
                    "column": 1,
                    "endLine": 1,
                    "endColumn": 1,
                    "offset": 0,
                    "endOffset": 0,
                    "length": 0,
                },
                "missingCapabilities": ["directx.workgroup-pointer-lowering"],
                "details": {
                    "sourcePath": MLX_FFT_SOURCE,
                    "targetArtifact": artifact["path"],
                    "workgroupPointer": {
                        "function": "ReadWriter_float2_float2__load",
                        "parameter": "crosstl_ptr_buf",
                        "reason": "bare-pointer-expression",
                    },
                },
            }
        ]


def _expected_layouts() -> dict[str, dict]:
    return {
        "fft_mem_256_float2_float2_n_Constants": {
            "physicalType": "int",
            "elementType": "int32",
            "elementSizeBytes": 4,
            "elementStrideBytes": 4,
            "alignmentBytes": 16,
            "memberOffsetBytes": 0,
            "storageLayout": "hlsl-constant-buffer",
            "runtimeSized": False,
            "memberName": "fft_mem_256_float2_float2_n",
            "blockSizeBytes": 16,
        },
        "fft_mem_256_float2_float2_batch_size_Constants": {
            "physicalType": "int",
            "elementType": "int32",
            "elementSizeBytes": 4,
            "elementStrideBytes": 4,
            "alignmentBytes": 16,
            "memberOffsetBytes": 0,
            "storageLayout": "hlsl-constant-buffer",
            "runtimeSized": False,
            "memberName": "fft_mem_256_float2_float2_batch_size",
            "blockSizeBytes": 16,
        },
        "CrossGLDispatchInfo": {
            "physicalType": "uint3",
            "elementType": "uint32",
            "elementSizeBytes": 12,
            "elementStrideBytes": 12,
            "alignmentBytes": 16,
            "memberOffsetBytes": 0,
            "storageLayout": "hlsl-constant-buffer",
            "runtimeSized": False,
            "vectorWidth": 3,
            "memberName": "crossglNumWorkGroups",
            "blockSizeBytes": 16,
        },
        "in_": {
            "physicalType": "float2",
            "elementType": "float32",
            "elementSizeBytes": 8,
            "elementStrideBytes": 8,
            "alignmentBytes": 4,
            "memberOffsetBytes": 0,
            "storageLayout": "hlsl-structured-buffer",
            "runtimeSized": True,
            "vectorWidth": 2,
        },
        "out_": {
            "physicalType": "float2",
            "elementType": "float32",
            "elementSizeBytes": 8,
            "elementStrideBytes": 8,
            "alignmentBytes": 4,
            "memberOffsetBytes": 0,
            "storageLayout": "hlsl-structured-buffer",
            "runtimeSized": True,
            "vectorWidth": 2,
        },
    }


def _build_runtime_package(mlx_root: Path, work_dir: Path) -> tuple[dict, Path]:
    output_dir = work_dir / "out"
    config_path = work_dir / "crosstl.toml"
    config_path.write_text(
        _project_config(output_dir.relative_to(mlx_root).as_posix()) + "\n",
        encoding="utf-8",
    )
    report = translate_project(
        load_project_config(mlx_root, config_path),
        targets=("directx",),
        output_dir=output_dir.relative_to(mlx_root).as_posix(),
        format_output=False,
        validate=False,
    )
    report_payload = report.to_json()

    assert report_payload["summary"]["unitCount"] == 1
    assert report_payload["summary"]["translatedCount"] == 1
    assert report_payload["summary"]["failedCount"] == 0
    assert report_payload["summary"]["diagnosticCounts"] == {
        "error": 0,
        "note": 0,
        "warning": 0,
    }, json.dumps(report_payload["diagnostics"], indent=2)
    artifact = report_payload["artifacts"][0]
    assert artifact["source"] == MLX_FFT_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_FFT_SHA256,
    }
    assert artifact["entryPoint"] == {
        "source": MLX_FFT_ENTRY,
        "target": "CSMain",
        "stage": "compute",
    }
    assert artifact["execution"]["entryPoints"][0]["workgroupSize"] == [1, 1, 64]
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": MLX_FFT_GENERATED_SHA256,
    }
    assert artifact["generatedSizeBytes"] == MLX_FFT_GENERATED_SIZE_BYTES
    assert len(artifact["templateMaterialization"]["specializations"]) == 24
    assert artifact["templateMaterialization"]["unsupported"] == []

    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
    runtime_artifacts = build_runtime_artifact_manifest(report_path)
    assert runtime_artifacts["success"] is True, json.dumps(runtime_artifacts, indent=2)
    reflected = runtime_artifacts["artifacts"][0]
    resources = reflected["hostInterface"]["resources"]
    assert {resource["name"]: resource["scalarLayout"] for resource in resources} == (
        _expected_layouts()
    )
    dispatch_info = next(
        resource for resource in resources if resource["name"] == "CrossGLDispatchInfo"
    )
    assert dispatch_info["provenance"] == {
        "kind": "generated-execution-input",
        "executionInput": {
            "kind": "dispatch-workgroup-count",
            "valueSource": "dispatch.workgroupCount",
            "coordinateSpace": "physical",
            "dimensions": 3,
            "memberName": "crossglNumWorkGroups",
        },
    }

    artifacts_path = work_dir / "runtime-artifacts.json"
    _write_json(artifacts_path, runtime_artifacts)
    package_dir = work_dir / "runtime-package"
    package = build_runtime_package(artifacts_path, package_dir)
    assert package["success"] is True, json.dumps(package, indent=2)
    loader_manifest = build_runtime_loader_manifest(
        package_dir / "runtime-package.json"
    )
    assert loader_manifest["success"] is True, json.dumps(loader_manifest, indent=2)
    assert loader_manifest["summary"]["readyLoadUnitCount"] == 1
    load_unit = loader_manifest["loadUnits"][0]
    descriptor = build_native_loader_abi_descriptor(
        loader_manifest,
        load_unit_id=load_unit["id"],
    )
    assert descriptor["entryPoint"]["executionConfig"] == {"numthreads": [1, 1, 64]}
    assert {
        binding["name"]: binding["scalarLayout"] for binding in descriptor["bindings"]
    } == _expected_layouts()
    descriptor_dispatch_info = next(
        binding
        for binding in descriptor["bindings"]
        if binding["name"] == "CrossGLDispatchInfo"
    )
    assert descriptor_dispatch_info["provenance"] == dispatch_info["provenance"]
    return descriptor, package_dir


def _complex_impulse() -> tuple[list[float], list[float]]:
    input_values = [0.0] * (MLX_FFT_SIZE * 2)
    input_values[2] = 1.0
    expected_values: list[float] = []
    for index in range(MLX_FFT_SIZE):
        value = cmath.exp(-2j * math.pi * index / MLX_FFT_SIZE)
        expected_values.extend((value.real, value.imag))
    return input_values, expected_values


def test_pinned_mlx_fft_executes_through_directx_native_loader():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-fft-directx-native-loader-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
        )
        input_values, expected_values = _complex_impulse()
        request = build_native_loader_dispatch_request(
            descriptor,
            package_dir,
            {
                "fft_mem_256_float2_float2_n_Constants": {
                    "dtype": "int32",
                    "shape": [1],
                    "values": [MLX_FFT_SIZE],
                },
                "fft_mem_256_float2_float2_batch_size_Constants": {
                    "dtype": "int32",
                    "shape": [1],
                    "values": [1],
                },
                "in_": {
                    "dtype": "float32",
                    "shape": [MLX_FFT_SIZE, 2],
                    "values": input_values,
                },
            },
            {
                "out_": {
                    "dtype": "float32",
                    "shape": [MLX_FFT_SIZE, 2],
                    "values": expected_values,
                    "tolerance": {"absolute": 2e-4, "relative": 2e-4},
                }
            },
            (1, 1, 1),
            expected_target="directx",
        )
        assert request.execution_plan is not None
        assert request.execution_plan.diagnostics == ()
        assert request.execution_plan.dispatch.workgroup_size == (1, 1, 64)
        assert request.execution_plan.dispatch.workgroup_count == (1, 1, 1)
        assert request.execution_plan.dispatch.global_size == (1, 1, 64)
        allocations = {
            item.binding.name: item.allocation
            for item in request.execution_plan.resource_bindings
        }
        assert allocations["in_"].byte_length == MLX_FFT_SIZE * 8
        assert allocations["in_"].allocation_byte_length == MLX_FFT_SIZE * 8
        assert allocations["out_"].byte_length == MLX_FFT_SIZE * 8
        assert allocations["out_"].allocation_byte_length == MLX_FFT_SIZE * 8
        assert allocations["CrossGLDispatchInfo"].byte_length == 16
        assert allocations["CrossGLDispatchInfo"].allocation_byte_length == 16
        fixture_inputs = {value.name: value for value in request.fixture.inputs}
        assert fixture_inputs["CrossGLDispatchInfo"].values == [1, 1, 1]
        assert fixture_inputs["CrossGLDispatchInfo"].metadata["source"] == (
            "dispatch.workgroupCount"
        )

        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id="mlx-fft-directx-native-loader",
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
            message = availability.reason or "The native DirectX runtime is unavailable"
            if os.environ.get(REQUIRE_PROOF_ENV) == "1":
                pytest.fail(message)
            pytest.skip(message)

        result = executor.run(request)

    assert result.status == "ok"
    assert result.outputs["out_"]["dtype"] == "float32"
    assert result.outputs["out_"]["shape"] == [MLX_FFT_SIZE, 2]
    assert result.outputs["out_"]["values"] == pytest.approx(
        expected_values,
        abs=2e-4,
        rel=2e-4,
    )
