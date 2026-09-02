from __future__ import annotations

import hashlib
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

MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
MLX_BINARY_SOURCE = "mlx/backend/metal/kernels/binary.metal"
MLX_BINARY_SHA256 = "4dadb612a9b768f9d51b3b394b32fc0129d361a55b35d545b3c014c87e00897e"
MLX_BINARY_ENTRY = "ss_Addfloat32"
REQUIRE_PROOF_ENVS = {
    "directx": "CROSTL_REQUIRE_MLX_BINARY_DIRECTX_NATIVE_LOADER",
    "opengl": "CROSTL_REQUIRE_MLX_BINARY_OPENGL_NATIVE_LOADER",
}


def _project_config(target: str) -> str:
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_BINARY_SOURCE}"]
        include_dirs = ["."]
        targets = ["{target}"]
        output_dir = ".crosstl-mlx-binary-native-loader/out"

        [project.sources]
        "**/*.metal" = "metal"

        [project.entry_points]
        "{MLX_BINARY_SOURCE}" = "{MLX_BINARY_ENTRY}"
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
    source_path = mlx_root / MLX_BINARY_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX binary source is missing: {source_path}")

    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_BINARY_SHA256
    return mlx_root


def _expected_scalar_layouts(target: str) -> dict[str, dict]:
    def runtime_array(*, member_name: str | None = None) -> dict:
        layout = {
            "physicalType": "float",
            "elementType": "float32",
            "elementSizeBytes": 4,
            "elementStrideBytes": 4,
            "alignmentBytes": 4,
            "memberOffsetBytes": 0,
            "storageLayout": (
                "hlsl-structured-buffer" if target == "directx" else "std430"
            ),
            "runtimeSized": True,
        }
        if member_name is not None:
            layout["memberName"] = member_name
        return layout

    if target == "directx":
        return {
            "a": runtime_array(),
            "b": runtime_array(),
            "c": runtime_array(),
        }
    return {
        "aBuffer": runtime_array(member_name="a"),
        "bBuffer": runtime_array(member_name="b"),
        "cBuffer": runtime_array(member_name="c"),
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
    assert artifact["source"] == MLX_BINARY_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_BINARY_SHA256,
    }
    assert artifact["entryPoint"] == {
        "source": MLX_BINARY_ENTRY,
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
        f"/{target}/mlx/backend/metal/kernels/binary/{MLX_BINARY_ENTRY}."
        f"{'hlsl' if target == 'directx' else 'glsl'}"
    )

    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
    runtime_artifacts = build_runtime_artifact_manifest(report_path)
    assert runtime_artifacts["success"] is True, json.dumps(runtime_artifacts, indent=2)

    reflected_artifact = runtime_artifacts["artifacts"][0]
    assert reflected_artifact["hostInterface"]["status"] == "ready"
    resources = reflected_artifact["hostInterface"]["resources"]
    assert [resource["access"] for resource in resources] == [
        "read",
        "read",
        "read_write",
    ]
    assert [resource["binding"] for resource in resources] == [0, 1, 2]
    reflected_layouts = {
        resource["name"]: resource["scalarLayout"] for resource in resources
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


def _execute_pinned_mlx_binary_add(target: str) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-binary-{target}-native-loader-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
            target,
        )
        if target == "directx":
            left_binding = "a"
            right_binding = "b"
            output_binding = "c"
        else:
            left_binding = "aBuffer"
            right_binding = "bBuffer"
            output_binding = "cBuffer"
        expected_values = [3.75, 3.75, 3.75, 3.75]
        request = build_native_loader_dispatch_request(
            descriptor,
            package_dir,
            {
                left_binding: {
                    "dtype": "float32",
                    "shape": [1],
                    "values": [1.5],
                },
                right_binding: {
                    "dtype": "float32",
                    "shape": [1],
                    "values": [2.25],
                },
            },
            {
                output_binding: {
                    "dtype": "float32",
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
            else OpenGLRuntimeParityAdapter(runtime=OpenGLComputeRuntime())
        )
        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id=f"mlx-binary-{target}-native-loader",
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
            "shape": [4],
            "values": expected_values,
        }
    }


def test_pinned_mlx_binary_add_executes_through_directx_native_loader():
    _execute_pinned_mlx_binary_add("directx")


def test_pinned_mlx_binary_add_executes_through_opengl_native_loader():
    _execute_pinned_mlx_binary_add("opengl")
