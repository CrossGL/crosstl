from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    DirectXComputeRuntime,
    DirectXRuntimeParityAdapter,
    OpenGLComputeRuntime,
    OpenGLRuntimeParityAdapter,
    ProjectConfig,
    RuntimeParityExecutor,
    RuntimeTestAdapterSpec,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    translate_project,
)

MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
MLX_COPY_SOURCE = "mlx/backend/metal/kernels/copy.metal"
MLX_COPY_SHA256 = "ed8a579eb6fe6a14c36560d2c8b548baf99e66fa77d300fb4ad7554883820eba"
MLX_COPY_ENTRY = "v_copyfloat32float32"
MLX_COPY_TEMPLATE_ARGUMENTS = {"N": "1", "T": "float", "U": "float"}
MLX_COPY_GENERATED_ARTIFACTS = {
    "directx": {
        "sha256": "023227a86b82cfeff6c32219e2526efbb658d018a4679b47f06c8369186a3495",
        "sizeBytes": 1234,
    },
    "opengl": {
        "sha256": "fac13358ba17271c622634c3e42f9b3cd0863adb75bcfe71aca7ce13aa5628cb",
        "sizeBytes": 1619,
    },
}
REQUIRE_PROOF_ENVS = {
    "directx": "CROSTL_REQUIRE_MLX_COPY_DIRECTX_NATIVE_LOADER",
    "opengl": "CROSTL_REQUIRE_MLX_COPY_OPENGL_NATIVE_LOADER",
}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _skip_or_fail(target: str, message: str) -> None:
    if os.environ.get(REQUIRE_PROOF_ENVS[target]) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if any(os.environ.get(name) == "1" for name in REQUIRE_PROOF_ENVS.values()):
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")

    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_COPY_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX copy source is missing: {source_path}")

    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_COPY_SHA256
    return mlx_root


def _project_config(mlx_root: Path, output_dir: str, target: str) -> ProjectConfig:
    return ProjectConfig(
        root=mlx_root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(MLX_COPY_SOURCE,),
        targets=(target,),
        output_dir=output_dir,
        source_overrides={MLX_COPY_SOURCE: "metal"},
        entry_points={MLX_COPY_SOURCE: MLX_COPY_ENTRY},
        include_dirs=(".",),
        source_options={
            "metal": {
                "max_template_specializations": 16,
                "max_template_materialization_work": 64,
            }
        },
    )


def _expected_scalar_layouts(target: str) -> dict[str, dict]:
    runtime_array = {
        "physicalType": "float",
        "elementType": "float32",
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
        "alignmentBytes": 4,
        "memberOffsetBytes": 0,
        "storageLayout": "hlsl-structured-buffer" if target == "directx" else "std430",
        "runtimeSized": True,
    }
    constant = {
        "physicalType": "uint",
        "elementType": "uint32",
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
        "alignmentBytes": 16,
        "memberOffsetBytes": 0,
        "storageLayout": "hlsl-constant-buffer" if target == "directx" else "std140",
        "runtimeSized": False,
        "blockSizeBytes": 16,
    }
    if target == "directx":
        return {
            "v_copyfloat32float32_size_Constants": {
                **constant,
                "memberName": "v_copyfloat32float32_size",
            },
            "src": dict(runtime_array),
            "dst": dict(runtime_array),
        }
    return {
        "srcBuffer": {**runtime_array, "memberName": "src"},
        "dstBuffer": {**runtime_array, "memberName": "dst"},
        "v_copyfloat32float32_size_Args": {**constant, "memberName": "size"},
    }


def _build_runtime_package(
    mlx_root: Path, work_dir: Path, target: str
) -> tuple[dict, Path]:
    output_dir = work_dir / "out"
    report = translate_project(
        _project_config(
            mlx_root,
            output_dir.relative_to(mlx_root).as_posix(),
            target,
        ),
        targets=(target,),
        output_dir=output_dir.relative_to(mlx_root).as_posix(),
        format_output=False,
        validate=False,
    )
    payload = report.to_json()

    assert payload["summary"]["unitCount"] == 1
    assert payload["summary"]["artifactCount"] == 1
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    assert payload["summary"]["diagnosticCounts"] == {
        "error": 0,
        "note": 0,
        "warning": 0,
    }
    assert payload["diagnostics"] == []

    artifact = payload["artifacts"][0]
    target_entry = "CSMain" if target == "directx" else "main"
    assert artifact["source"] == MLX_COPY_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_COPY_SHA256,
    }
    assert artifact["entryPoint"] == {
        "source": MLX_COPY_ENTRY,
        "target": target_entry,
        "stage": "compute",
    }
    assert artifact["provenance"] == {
        "intermediate": "crossgl",
        "pipeline": "entry-scoped-translate",
    }
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": MLX_COPY_GENERATED_ARTIFACTS[target]["sha256"],
    }
    assert artifact["generatedSizeBytes"] == (
        MLX_COPY_GENERATED_ARTIFACTS[target]["sizeBytes"]
    )
    materialization = artifact["templateMaterialization"]
    assert materialization["status"] == "materialized"
    assert materialization["specializationCount"] == 2
    assert materialization["unsupported"] == []
    assert materialization["accounting"] == {
        "reachableSpecializationCount": 5,
        "dependencyDiscoveryWorkCount": 7,
        "prunedCandidateCount": 62424,
    }
    assert materialization["specializations"] == [
        {
            "name": "copy_v",
            "hostName": MLX_COPY_ENTRY,
            "materializedName": MLX_COPY_ENTRY,
            "parameters": MLX_COPY_TEMPLATE_ARGUMENTS,
            "parameterSources": {
                "N": "source-instantiation",
                "T": "source-instantiation",
                "U": "source-instantiation",
            },
            "source": "source-instantiation",
        },
        {
            "name": "cast_to",
            "materializedName": "cast_to_float_float",
            "parameters": {"T": "float", "U": "float"},
            "parameterSources": {"T": "call-site", "U": "call-site"},
            "source": "call-site",
        },
    ]

    generated = (mlx_root / artifact["path"]).read_text(encoding="utf-8")
    if target == "directx":
        assert "float cast_to_float_float(float val)" in generated
        assert (
            "dst[(index + i)] = cast_to_float_float(src.Load((index + i)));"
            in generated
        )
    else:
        assert "float cast_to_float_float(float val)" in generated
        assert (
            "dst[(index + uint(i))] = "
            "cast_to_float_float(src[(index + uint(i))]);" in generated
        )

    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
    runtime_artifacts = build_runtime_artifact_manifest(report_path)
    assert runtime_artifacts["success"] is True, json.dumps(runtime_artifacts, indent=2)
    assert runtime_artifacts["summary"]["artifactCount"] == 1
    reflected = runtime_artifacts["artifacts"][0]
    assert reflected["hostInterface"]["status"] == "ready"
    reflected_layouts = {
        resource["name"]: resource["scalarLayout"]
        for resource in reflected["hostInterface"]["resources"]
    }
    assert reflected_layouts == _expected_scalar_layouts(target)

    runtime_artifacts_path = work_dir / "runtime-artifacts.json"
    _write_json(runtime_artifacts_path, runtime_artifacts)
    package_dir = work_dir / "runtime-package"
    package = build_runtime_package(runtime_artifacts_path, package_dir)
    assert package["success"] is True, json.dumps(package, indent=2)

    loader_manifest = build_runtime_loader_manifest(
        package_dir / "runtime-package.json"
    )
    assert loader_manifest["success"] is True, json.dumps(loader_manifest, indent=2)
    assert loader_manifest["summary"]["readyLoadUnitCount"] == 1
    assert loader_manifest["summary"]["blockedLoadUnitCount"] == 0
    load_unit = loader_manifest["loadUnits"][0]
    descriptor = build_native_loader_abi_descriptor(
        loader_manifest,
        load_unit_id=load_unit["id"],
    )
    assert descriptor["target"] == target
    assert descriptor["entryPoint"]["name"] == target_entry
    expected_execution = (
        {"numthreads": [1, 1, 1]}
        if target == "directx"
        else {"local_size_x": 1, "local_size_y": 1, "local_size_z": 1}
    )
    assert descriptor["entryPoint"]["executionConfig"] == expected_execution
    assert descriptor["specializationConstants"] == []
    descriptor_layouts = {
        binding["name"]: binding["scalarLayout"] for binding in descriptor["bindings"]
    }
    assert descriptor_layouts == _expected_scalar_layouts(target)
    return descriptor, package_dir


def _execute_pinned_mlx_copy(target: str) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-copy-{target}-native-loader-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
            target,
        )
        values = [-3.5, -1.25, 0.0, 0.125, 1.5, 7.75, 1024.5, -0.03125]
        if target == "directx":
            size_binding = "v_copyfloat32float32_size_Constants"
            input_binding = "src"
            output_binding = "dst"
        else:
            size_binding = "v_copyfloat32float32_size_Args"
            input_binding = "srcBuffer"
            output_binding = "dstBuffer"

        request = build_native_loader_dispatch_request(
            descriptor,
            package_dir,
            {
                size_binding: {
                    "dtype": "uint32",
                    "shape": [1],
                    "values": [len(values)],
                },
                input_binding: {
                    "dtype": "float32",
                    "shape": [len(values)],
                    "values": values,
                },
            },
            {
                output_binding: {
                    "dtype": "float32",
                    "shape": [len(values)],
                    "values": values,
                }
            },
            (len(values), 1, 1),
            expected_target=target,
        )
        assert request.execution_plan is not None
        assert request.execution_plan.diagnostics == ()
        assert request.execution_plan.dispatch.workgroup_size == (1, 1, 1)
        assert request.execution_plan.dispatch.workgroup_count == (len(values), 1, 1)
        assert request.execution_plan.dispatch.global_size == (len(values), 1, 1)

        runtime_adapter = (
            DirectXRuntimeParityAdapter(runtime=DirectXComputeRuntime())
            if target == "directx"
            else OpenGLRuntimeParityAdapter(
                runtime=OpenGLComputeRuntime(context_backends=("egl",))
            )
        )
        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id=f"mlx-copy-{target}-native-loader",
                target=target,
                executor=target,
                adapter_kind=f"{target}-native-runtime",
            ),
            runtime_adapter=runtime_adapter,
        )
        availability = executor.is_available(request)
        if not availability.available:
            _skip_or_fail(
                target,
                availability.reason or f"The native {target} runtime is unavailable",
            )

        result = executor.run(request)

    assert result.status == "ok"
    assert result.outputs == {
        output_binding: {
            "dtype": "float32",
            "shape": [len(values)],
            "values": values,
        }
    }


def test_pinned_mlx_copy_executes_through_directx_native_loader():
    _execute_pinned_mlx_copy("directx")


def test_pinned_mlx_copy_executes_through_opengl_native_loader():
    _execute_pinned_mlx_copy("opengl")
