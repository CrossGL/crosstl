from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

from crosstl.project import (
    DirectXComputeRuntime,
    DirectXRuntimeParityAdapter,
    OpenGLComputeRuntime,
    OpenGLRuntimeParityAdapter,
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
MLX_ARANGE_SOURCE = "mlx/backend/metal/kernels/arange.metal"
REQUIRE_PROOF_ENVS = {
    "directx": "CROSTL_REQUIRE_MLX_ARANGE_DIRECTX_NATIVE_LOADER",
    "opengl": "CROSTL_REQUIRE_MLX_ARANGE_OPENGL_NATIVE_LOADER",
}


def _project_config(target: str) -> str:
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_ARANGE_SOURCE}"]
        include_dirs = ["."]
        targets = ["{target}"]
        output_dir = ".crosstl-mlx-arange-native-loader/out"

        [project.sources]
        "**/*.metal" = "metal"

        [project.entry_points]
        "{MLX_ARANGE_SOURCE}" = "arangeuint32"
        """).strip()


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
    source_path = mlx_root / MLX_ARANGE_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX arange source is missing: {source_path}")

    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == MLX_COMMIT
    return mlx_root


def _expected_scalar_layouts(target: str) -> dict[str, dict]:
    scalar_array = {
        "physicalType": "uint",
        "elementType": "uint32",
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
        "alignmentBytes": 4,
        "memberOffsetBytes": 0,
        "storageLayout": "hlsl-structured-buffer" if target == "directx" else "std430",
        "runtimeSized": True,
    }
    if target == "opengl":
        scalar_array["memberName"] = "out_"
    scalar_block = {
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
            "out_": scalar_array,
            "arangeuint32_start_Constants": {
                **scalar_block,
                "memberName": "arangeuint32_start",
            },
            "arangeuint32_step_Constants": {
                **scalar_block,
                "memberName": "arangeuint32_step",
            },
        }
    return {
        "out_Buffer": scalar_array,
        "arangeuint32_start_Args": {**scalar_block, "memberName": "start"},
        "arangeuint32_step_Args": {**scalar_block, "memberName": "step"},
    }


def _build_runtime_package(
    mlx_root: Path, work_dir: Path, target: str
) -> tuple[dict, Path]:
    config_path = work_dir / "crosstl.toml"
    config_path.write_text(_project_config(target) + "\n", encoding="utf-8")
    output_dir = work_dir / "out"
    report = translate_project(
        load_project_config(mlx_root, config_path),
        targets=(target,),
        output_dir=output_dir.relative_to(mlx_root).as_posix(),
        format_output=False,
        validate=True,
    )
    report_payload = report.to_json()

    assert report_payload["summary"]["unitCount"] == 1
    assert report_payload["summary"]["translatedCount"] == 1
    assert report_payload["summary"]["failedCount"] == 0
    artifact = report_payload["artifacts"][0]
    assert artifact["source"] == MLX_ARANGE_SOURCE
    assert artifact["entryPoint"] == {
        "source": "arangeuint32",
        "target": "CSMain" if target == "directx" else "main",
        "stage": "compute",
    }
    assert artifact["provenance"]["pipeline"] == "entry-scoped-translate"
    assert artifact["includePathProcessing"] == {
        "frontend": "lexer",
        "includePathCount": 1,
        "status": "forwarded",
        "supportsIncludePaths": True,
    }
    assert artifact["path"].endswith(
        f"/{target}/mlx/backend/metal/kernels/arange/arangeuint32."
        f"{'hlsl' if target == 'directx' else 'glsl'}"
    )

    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
    runtime_artifacts = build_runtime_artifact_manifest(report_path)
    assert runtime_artifacts["success"] is True, json.dumps(runtime_artifacts, indent=2)

    reflected_artifact = runtime_artifacts["artifacts"][0]
    assert reflected_artifact["hostInterface"]["status"] == "ready"
    reflected_layouts = {
        resource["name"]: resource["scalarLayout"]
        for resource in reflected_artifact["hostInterface"]["resources"]
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
    descriptor_layouts = {
        binding["name"]: binding["scalarLayout"] for binding in descriptor["bindings"]
    }
    assert descriptor_layouts == _expected_scalar_layouts(target)
    return descriptor, package_dir


def _execute_pinned_mlx_arange(target: str) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-arange-{target}-native-loader-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
            target,
        )
        expected_values = [3, 5, 7, 9]
        if target == "directx":
            start_binding = "arangeuint32_start_Constants"
            step_binding = "arangeuint32_step_Constants"
            output_binding = "out_"
        else:
            start_binding = "arangeuint32_start_Args"
            step_binding = "arangeuint32_step_Args"
            output_binding = "out_Buffer"
        request = build_native_loader_dispatch_request(
            descriptor,
            package_dir,
            {
                start_binding: {
                    "dtype": "uint32",
                    "shape": [1],
                    "values": [3],
                },
                step_binding: {
                    "dtype": "uint32",
                    "shape": [1],
                    "values": [2],
                },
            },
            {
                output_binding: {
                    "dtype": "uint32",
                    "shape": [4],
                    "values": expected_values,
                }
            },
            (4, 1, 1),
            expected_target=target,
        )
        assert request.execution_plan is not None
        assert request.execution_plan.diagnostics == ()
        assert request.execution_plan.dispatch.global_size == (4, 1, 1)

        runtime_adapter = (
            DirectXRuntimeParityAdapter(runtime=DirectXComputeRuntime())
            if target == "directx"
            else OpenGLRuntimeParityAdapter(
                runtime=OpenGLComputeRuntime(context_backends=("egl",))
            )
        )
        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id=f"mlx-arange-{target}-native-loader",
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
            "dtype": "uint32",
            "shape": [4],
            "values": expected_values,
        }
    }


def test_pinned_mlx_arange_executes_through_directx_native_loader():
    _execute_pinned_mlx_arange("directx")


def test_pinned_mlx_arange_executes_through_opengl_native_loader():
    _execute_pinned_mlx_arange("opengl")
