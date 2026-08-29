from __future__ import annotations

import hashlib
import json
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
    validate_project_report,
)

MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
MLX_DOT_SOURCE = "mlx/backend/metal/kernels/dot.metal"
MLX_DOT_SHA256 = "97bcad13d09c3d5fed87482a0bb9719d6eeff9b21d364967cd6aec5b695b3462"
MLX_DOT_ENTRY = "dot_product_float32_it32_tg512_sg16"
MLX_DOT_GENERATED_ARTIFACTS = {
    "directx": {
        "sha256": "f902bae0e7603d302340327c61e5a82ab392b9ce1afffb5557b1760592a1465f",
        "sizeBytes": 5110,
    },
    "opengl": {
        "sha256": "ef69a757339fe09897a38804c27be279a19a7db146e2e02f85f0349c59f3168d",
        "sizeBytes": 5188,
    },
}
MLX_DOT_SOFTWARE_OPENGL_ARTIFACT = {
    "sha256": "a3c1958daa680419ce3f38559de1a6a2319a7abdac556a049632194c88223a32",
    "sizeBytes": 6275,
}
REQUIRE_PROOF_ENVS = {
    "directx": "CROSTL_REQUIRE_MLX_DOT_DIRECTX_NATIVE_LOADER",
    "opengl": "CROSTL_REQUIRE_MLX_DOT_OPENGL_TOOLCHAIN",
}
REQUIRE_OPENGL_RUNTIME_ENV = "CROSTL_REQUIRE_MLX_DOT_OPENGL_NATIVE_LOADER"


def _project_config(
    target: str,
    output_dir: str,
    *,
    software_subgroups: bool = False,
) -> str:
    if software_subgroups and target != "opengl":
        raise ValueError("software_subgroups is only valid for the OpenGL target")
    sections = [textwrap.dedent(f"""
            [project]
            source_roots = ["mlx/backend/metal/kernels"]
            include = ["{MLX_DOT_SOURCE}"]
            include_dirs = ["."]
            targets = ["{target}"]
            output_dir = "{output_dir}"

            [project.sources]
            "**/*.metal" = "metal"

            [project.entry_points]
            "{MLX_DOT_SOURCE}" = "{MLX_DOT_ENTRY}"
            """).strip()]
    if target in {"directx", "opengl"}:
        sections.append(textwrap.dedent(f"""
            [project.entry_workgroup_size_rules."{MLX_DOT_SOURCE}"]
            "{MLX_DOT_ENTRY}" = [512, 1, 1]
            """).strip())
    if target in {"directx", "opengl"} and not software_subgroups:
        sections.append(textwrap.dedent(f"""
            [project.subgroup_width_rules]
            "{MLX_DOT_SOURCE}" = 32
            """).strip())
    sections.append(textwrap.dedent("""
        [project.source_options.metal]
        max_template_specializations = 64
        max_template_materialization_work = 4096
        """).strip())
    if software_subgroups:
        sections.append(textwrap.dedent("""
            [project.source_options.metal.target_options.opengl]
            software_subgroup_width = 32
            """).strip())
    return "\n\n".join(sections)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _skip_or_fail(
    target: str,
    message: str,
    *,
    require_env: str | None = None,
) -> None:
    if os.environ.get(require_env or REQUIRE_PROOF_ENVS[target]) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if any(
            os.environ.get(name) == "1"
            for name in (*REQUIRE_PROOF_ENVS.values(), REQUIRE_OPENGL_RUNTIME_ENV)
        ):
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")

    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_DOT_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX dot-product source is missing: {source_path}")

    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_DOT_SHA256
    return mlx_root


def _translate_dot_artifact(
    mlx_root: Path,
    work_dir: Path,
    target: str,
    *,
    run_toolchains: bool,
    software_subgroups: bool = False,
):
    output_dir = work_dir / "out"
    config_path = work_dir / "crosstl.toml"
    config_path.write_text(
        _project_config(
            target,
            output_dir.relative_to(mlx_root).as_posix(),
            software_subgroups=software_subgroups,
        )
        + "\n",
        encoding="utf-8",
    )
    report = translate_project(
        load_project_config(mlx_root, config_path),
        targets=(target,),
        output_dir=output_dir.relative_to(mlx_root).as_posix(),
        format_output=False,
        validate=True,
        run_toolchains=run_toolchains,
    )
    payload = report.to_json()

    assert payload["summary"]["unitCount"] == 1
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    artifact = payload["artifacts"][0]
    expected_identity = (
        MLX_DOT_SOFTWARE_OPENGL_ARTIFACT
        if software_subgroups
        else MLX_DOT_GENERATED_ARTIFACTS[target]
    )
    assert artifact["source"] == MLX_DOT_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_DOT_SHA256,
    }
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": expected_identity["sha256"],
    }
    assert artifact["generatedSizeBytes"] == expected_identity["sizeBytes"]
    assert artifact["entryPoint"] == {
        "source": MLX_DOT_ENTRY,
        "target": "CSMain" if target == "directx" else "main",
        "stage": "compute",
    }
    assert artifact["provenance"]["pipeline"] == "entry-scoped-translate"
    entry = artifact["execution"]["entryPoints"][0]
    assert entry["sourceEntryPoint"] == MLX_DOT_ENTRY
    assert entry["workgroupSize"] == [512, 1, 1]
    if software_subgroups:
        assert "subgroupWidth" not in entry
    else:
        assert entry["subgroupWidth"] == 32
    assert entry["parameters"] == {
        "ITEMS_PER_THREAD": "32",
        "SIMD_GROUPS": "16",
        "T": "float",
        "TG_SIZE": "512",
    }

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    if target == "directx":
        assert "[numthreads(512, 1, 1)]" in generated
        assert "[WaveSize(32)]" in generated
        assert "groupshared float" in generated
        assert generated.count("asfloat(asuint(a[uint(") >= 4
        assert generated.count("asfloat(asuint(b[uint(") >= 4
        assert generated.count("WaveActiveSum(sum)") == 2
        assert "groupshared uint __crossgl_physical_subgroup_counter;" in generated
        assert "InterlockedAdd(__crossgl_physical_subgroup_counter" in generated
        assert "uint crossglPhysicalSubgroupID = " in generated
    elif software_subgroups:
        assert "layout(local_size_x = 512" in generated
        assert "#define CROSSTL_SOFTWARE_SUBGROUP_WIDTH 32u" in generated
        assert "GL_KHR_shader_subgroup" not in generated
        assert "gl_Subgroup" not in generated
        assert "subgroupAdd" not in generated
        assert "shared float crossglSoftwareSubgroupScratchFloat[512];" in generated
        assert "uint subgroupBase = invocation - lane;" in generated
        assert generated.count("crossglSoftwareSubgroupSumFloat(sum)") == 1
        assert "crossglSoftwareSubgroupSumFloat(" in generated
        assert "float crossglSoftwareSubgroupInput = 0.0;" in generated
        assert "if (crossglSoftwareSubgroupActive)" in generated
    else:
        assert "layout(local_size_x = 512" in generated
        assert "#define CROSSTL_REQUIRED_SUBGROUP_WIDTH 32u" in generated
        assert "if (gl_SubgroupSize != CROSSTL_REQUIRED_SUBGROUP_WIDTH)" in generated
        assert "shared float" in generated
        assert generated.count("uintBitsToFloat(floatBitsToUint(a[int(") >= 4
        assert generated.count("uintBitsToFloat(floatBitsToUint(b[int(") >= 4
        assert generated.count("subgroupAdd(sum)") == 2

    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True
    runtime_artifacts = build_runtime_artifact_manifest(report_path)
    assert runtime_artifacts["success"] is True, json.dumps(
        runtime_artifacts,
        indent=2,
    )
    assert runtime_artifacts["summary"]["artifactCount"] == 1
    assert runtime_artifacts["summary"]["entryPointCount"] == 1
    reflected = runtime_artifacts["artifacts"][0]
    execution_config = reflected["hostInterface"]["entryPoints"][0]["executionConfig"]
    if target == "directx":
        assert execution_config == {
            "numthreads": [512, 1, 1],
            "subgroupWidth": 32,
        }
    else:
        assert execution_config["local_size"] == [512, 1, 1]
        if software_subgroups:
            assert "subgroupWidth" not in execution_config
        else:
            assert execution_config["subgroupWidth"] == 32
    return report, runtime_artifacts


def _assert_software_opengl_spirv(generated_path: Path, work_dir: Path) -> None:
    tools = {
        name: shutil.which(name)
        for name in ("glslangValidator", "spirv-val", "spirv-dis")
    }
    if not all(tools.values()):
        if os.environ.get(REQUIRE_OPENGL_RUNTIME_ENV) == "1":
            missing = [name for name, path in tools.items() if path is None]
            pytest.fail("The dot OpenGL runtime proof requires: " + ", ".join(missing))
        return

    spirv_path = work_dir / "dot-software.spv"
    assembly_path = work_dir / "dot-software.spvasm"
    subprocess.run(
        [
            tools["glslangValidator"],
            "--target-env",
            "opengl",
            "--target-env",
            "spirv1.3",
            "-S",
            "comp",
            str(generated_path),
            "-o",
            str(spirv_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [tools["spirv-val"], "--target-env", "spv1.3", str(spirv_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [tools["spirv-dis"], str(spirv_path), "-o", str(assembly_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assembly = assembly_path.read_text(encoding="utf-8")
    assert assembly.count("OpControlBarrier") == 4
    assert "OpGroupNonUniform" not in assembly
    assert "OpExecutionMode %main LocalSize 512 1 1" in assembly


def test_pinned_mlx_dot_translates_to_guarded_opengl_artifact():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-dot-opengl-toolchain-",
        dir=mlx_root,
    ) as temporary_directory:
        report, runtime_artifacts = _translate_dot_artifact(
            mlx_root,
            Path(temporary_directory),
            "opengl",
            run_toolchains=True,
        )
        toolchain_runs = report.to_json()["validation"]["toolchainRuns"]
        if not toolchain_runs or toolchain_runs[0]["status"] != "ok":
            _skip_or_fail(
                "opengl",
                "The OpenGL validation toolchain did not accept the MLX dot artifact",
            )
        resources = runtime_artifacts["artifacts"][0]["hostInterface"]["resources"]
        assert {resource["name"]: resource["binding"] for resource in resources} == {
            "aBuffer": 0,
            "bBuffer": 1,
            "output_Buffer": 2,
            f"{MLX_DOT_ENTRY}_n_Args": 3,
        }


def test_pinned_mlx_dot_records_metal_roundtrip_boundary():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-dot-metal-roundtrip-",
        dir=mlx_root,
    ) as temporary_directory:
        work_dir = Path(temporary_directory)
        output_dir = work_dir / "out"
        config_path = work_dir / "crosstl.toml"
        config_path.write_text(
            _project_config("metal", output_dir.relative_to(mlx_root).as_posix())
            + "\n",
            encoding="utf-8",
        )
        report = translate_project(
            load_project_config(mlx_root, config_path),
            targets=("metal",),
            output_dir=output_dir.relative_to(mlx_root).as_posix(),
            format_output=False,
            validate=True,
            run_toolchains=True,
        )
        payload = report.to_json()

    assert payload["summary"]["unitCount"] == 1
    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 1
    artifact = payload["artifacts"][0]
    assert artifact["source"] == MLX_DOT_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_DOT_SHA256,
    }
    assert artifact["target"] == "metal"
    assert artifact["status"] == "failed"
    assert artifact["error"] == (
        "metal does not implement storage-backed pointer reinterpretation"
    )
    assert artifact["templateMaterialization"]["status"] == "materialized"
    assert artifact["templateMaterialization"]["specializationCount"] == 1
    assert not (mlx_root / artifact["path"]).exists()

    diagnostic = next(
        item
        for item in payload["diagnostics"]
        if item["code"] == "project.translate.pointer-reinterpret-unsupported"
    )
    assert diagnostic["target"] == "metal"
    assert diagnostic["missingCapabilities"] == ["pointer.reinterpretation"]
    assert diagnostic["details"]["pointerReinterpretation"] == {
        "reason": "target-lowering-unavailable",
        "targetBackend": "metal",
        "targetType": (
            "VectorType(element_type=PrimitiveType(name=float, size_bits=None), "
            "size=4)"
        ),
    }


def _build_directx_runtime_package(
    mlx_root: Path,
    work_dir: Path,
) -> tuple[dict, Path]:
    report, runtime_artifacts = _translate_dot_artifact(
        mlx_root,
        work_dir,
        "directx",
        run_toolchains=True,
    )
    if shutil.which("dxc") is not None:
        assert report.to_json()["validation"]["toolchainRuns"][0]["status"] == "ok"
    elif os.environ.get(REQUIRE_PROOF_ENVS["directx"]) == "1":
        pytest.fail("dxc is required for the MLX dot-product DirectX proof")

    resources = runtime_artifacts["artifacts"][0]["hostInterface"]["resources"]
    assert {resource["name"]: resource["binding"] for resource in resources} == {
        f"{MLX_DOT_ENTRY}_n_Constants": 3,
        "a": 0,
        "b": 1,
        "output": 2,
    }
    assert {resource["name"]: resource["access"] for resource in resources} == {
        f"{MLX_DOT_ENTRY}_n_Constants": "read",
        "a": "read",
        "b": "read",
        "output": "read_write",
    }

    runtime_artifacts_path = work_dir / "runtime-artifacts.json"
    _write_json(runtime_artifacts_path, runtime_artifacts)
    package_dir = work_dir / "runtime-package"
    package = build_runtime_package(runtime_artifacts_path, package_dir)
    assert package["success"] is True, json.dumps(package, indent=2)
    loader_manifest = build_runtime_loader_manifest(
        package_dir / "runtime-package.json"
    )
    assert loader_manifest["success"] is True, json.dumps(
        loader_manifest,
        indent=2,
    )
    assert loader_manifest["summary"]["readyLoadUnitCount"] == 1
    assert loader_manifest["summary"]["blockedLoadUnitCount"] == 0
    descriptor = build_native_loader_abi_descriptor(
        loader_manifest,
        load_unit_id=loader_manifest["loadUnits"][0]["id"],
    )
    assert descriptor["target"] == "directx"
    assert descriptor["entryPoint"]["name"] == "CSMain"
    assert descriptor["entryPoint"]["stage"] == "compute"
    assert descriptor["entryPoint"]["executionConfig"] == {
        "numthreads": [512, 1, 1],
        "subgroupWidth": 32,
    }
    return descriptor, package_dir


def test_pinned_mlx_dot_executes_through_directx_native_loader():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-dot-directx-native-loader-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir = _build_directx_runtime_package(
            mlx_root,
            Path(temporary_directory),
        )
        input_count = 1024
        expected_value = 256.0
        request = build_native_loader_dispatch_request(
            descriptor,
            package_dir,
            {
                "a": {
                    "dtype": "float32",
                    "shape": [input_count],
                    "values": [1.0] * input_count,
                },
                "b": {
                    "dtype": "float32",
                    "shape": [input_count],
                    "values": [0.25] * input_count,
                },
                f"{MLX_DOT_ENTRY}_n_Constants": {
                    "dtype": "int32",
                    "shape": [1],
                    "values": [input_count],
                },
            },
            {
                "output": {
                    "dtype": "float32",
                    "shape": [1],
                    "values": [expected_value],
                    "tolerance": {"absolute": 1e-5, "relative": 1e-5},
                }
            },
            (1, 1, 1),
            expected_target="directx",
        )
        assert request.execution_plan is not None
        assert request.execution_plan.diagnostics == ()
        assert request.execution_plan.dispatch.workgroup_size == (512, 1, 1)
        assert request.execution_plan.dispatch.workgroup_count == (1, 1, 1)
        assert request.execution_plan.dispatch.global_size == (512, 1, 1)

        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id="mlx-dot-directx-native-loader",
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
                "directx",
                availability.reason or "The native DirectX runtime is unavailable",
            )

        result = executor.run(request)

    assert result.status == "ok"
    assert result.outputs["output"]["dtype"] == "float32"
    assert result.outputs["output"]["shape"] == [1]
    assert result.outputs["output"]["values"] == pytest.approx(
        [expected_value],
        abs=1e-5,
        rel=1e-5,
    )


def _build_opengl_software_runtime_package(
    mlx_root: Path,
    work_dir: Path,
) -> tuple[dict, Path]:
    report, runtime_artifacts = _translate_dot_artifact(
        mlx_root,
        work_dir,
        "opengl",
        run_toolchains=True,
        software_subgroups=True,
    )
    toolchain_runs = report.to_json()["validation"]["toolchainRuns"]
    if not toolchain_runs or toolchain_runs[0]["status"] != "ok":
        _skip_or_fail(
            "opengl",
            "The OpenGL toolchain did not accept the software dot artifact",
            require_env=REQUIRE_OPENGL_RUNTIME_ENV,
        )
    generated_path = mlx_root / report.to_json()["artifacts"][0]["path"]
    _assert_software_opengl_spirv(generated_path, work_dir)

    resources = runtime_artifacts["artifacts"][0]["hostInterface"]["resources"]
    assert {resource["name"]: resource["binding"] for resource in resources} == {
        "aBuffer": 0,
        "bBuffer": 1,
        "output_Buffer": 2,
        f"{MLX_DOT_ENTRY}_n_Args": 3,
    }
    assert {resource["name"]: resource["access"] for resource in resources} == {
        "aBuffer": "read",
        "bBuffer": "read",
        "output_Buffer": "read_write",
        f"{MLX_DOT_ENTRY}_n_Args": "read",
    }

    runtime_artifacts_path = work_dir / "runtime-artifacts.json"
    _write_json(runtime_artifacts_path, runtime_artifacts)
    package_dir = work_dir / "runtime-package"
    package = build_runtime_package(runtime_artifacts_path, package_dir)
    assert package["success"] is True, json.dumps(package, indent=2)
    loader_manifest = build_runtime_loader_manifest(
        package_dir / "runtime-package.json"
    )
    assert loader_manifest["success"] is True, json.dumps(
        loader_manifest,
        indent=2,
    )
    assert loader_manifest["summary"]["loadUnitCount"] == 1
    assert loader_manifest["summary"]["readyLoadUnitCount"] == 1
    assert loader_manifest["summary"]["blockedLoadUnitCount"] == 0
    descriptor = build_native_loader_abi_descriptor(
        loader_manifest,
        load_unit_id=loader_manifest["loadUnits"][0]["id"],
    )
    assert descriptor["target"] == "opengl"
    assert descriptor["entryPoint"]["name"] == "main"
    assert descriptor["entryPoint"]["stage"] == "compute"
    execution_config = descriptor["entryPoint"]["executionConfig"]
    assert execution_config["local_size"] == [512, 1, 1]
    assert "subgroupWidth" not in execution_config
    return descriptor, package_dir


def test_pinned_mlx_dot_executes_with_opengl_software_subgroups():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-dot-opengl-software-native-loader-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir = _build_opengl_software_runtime_package(
            mlx_root,
            Path(temporary_directory),
        )
        input_count = 1024
        expected_value = 256.0
        request = build_native_loader_dispatch_request(
            descriptor,
            package_dir,
            {
                "aBuffer": {
                    "dtype": "float32",
                    "shape": [input_count],
                    "values": [1.0] * input_count,
                },
                "bBuffer": {
                    "dtype": "float32",
                    "shape": [input_count],
                    "values": [0.25] * input_count,
                },
                f"{MLX_DOT_ENTRY}_n_Args": {
                    "dtype": "int32",
                    "shape": [1],
                    "values": [input_count],
                },
            },
            {
                "output_Buffer": {
                    "dtype": "float32",
                    "shape": [1],
                    "values": [expected_value],
                    "tolerance": {"absolute": 1e-5, "relative": 1e-5},
                }
            },
            (1, 1, 1),
            expected_target="opengl",
        )
        assert request.execution_plan is not None
        assert request.execution_plan.diagnostics == ()
        assert request.execution_plan.dispatch.workgroup_size == (512, 1, 1)
        assert request.execution_plan.dispatch.workgroup_count == (1, 1, 1)
        assert request.execution_plan.dispatch.global_size == (512, 1, 1)

        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id="mlx-dot-opengl-software-native-loader",
                target="opengl",
                executor="opengl",
                adapter_kind="opengl-native-runtime",
            ),
            runtime_adapter=OpenGLRuntimeParityAdapter(runtime=OpenGLComputeRuntime()),
        )
        availability = executor.is_available(request)
        if not availability.available:
            _skip_or_fail(
                "opengl",
                availability.reason or "The native OpenGL runtime is unavailable",
                require_env=REQUIRE_OPENGL_RUNTIME_ENV,
            )

        result = executor.run(request)

    assert result.status == "ok"
    assert result.outputs["output_Buffer"]["dtype"] == "float32"
    assert result.outputs["output_Buffer"]["shape"] == [1]
    assert result.outputs["output_Buffer"]["values"] == pytest.approx(
        [expected_value],
        abs=1e-5,
        rel=1e-5,
    )
