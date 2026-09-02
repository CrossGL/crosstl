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
MLX_QUANTIZED_SOURCE = "mlx/backend/metal/kernels/quantized.metal"
MLX_QUANTIZED_SHA256 = (
    "292aab5a98e3fc047b8ed91343fc10b66e5a92e12c258cde168929520ab2abfd"
)
MLX_QUANTIZED_ENTRY = "affine_quantize_float_gs_32_b_2"
MLX_QUANTIZED_OPENGL_ARTIFACT = {
    "sha256": "e4d8e5931bfc93f81e2c3686c102a1d676c9a3dcdfd6447e90918aa7581beecb",
    "sizeBytes": 6642,
}
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_QUANTIZED_OPENGL_NATIVE_LOADER"


def _project_config() -> str:
    assertions = "\n\n".join(
        textwrap.dedent(f"""
            [[project.index_range_assertions]]
            source = "{MLX_QUANTIZED_SOURCE}"
            expression = "{expression}"
            minimum = 0
            maximum = 2147483647
            """).strip()
        for expression in (
            "in_index + i",
            "gindex",
            "out_index / writes_per_reduce",
        )
    )
    return textwrap.dedent(f"""
            [project]
            source_roots = ["mlx/backend/metal/kernels"]
            include = ["{MLX_QUANTIZED_SOURCE}"]
            include_dirs = ["."]
            targets = ["opengl"]
            output_dir = ".crosstl-mlx-quantized-native-loader/out"

            [project.sources]
            "**/*.metal" = "metal"

            [project.entry_points]
            "{MLX_QUANTIZED_SOURCE}" = "{MLX_QUANTIZED_ENTRY}"

            [project.entry_workgroup_size_rules."{MLX_QUANTIZED_SOURCE}"]
            "{MLX_QUANTIZED_ENTRY}" = [32, 1, 1]

            [project.source_options.metal]
            max_template_specializations = 128
            max_template_materialization_work = 4096

            [project.source_options.metal.target_options.opengl]
            software_subgroup_width = 32
            """).strip() + "\n\n" + assertions


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _skip_or_fail(message: str) -> None:
    if os.environ.get(REQUIRE_ENV) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")

    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_QUANTIZED_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX quantized source is missing: {source_path}")

    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
        MLX_QUANTIZED_SHA256
    )
    return mlx_root


def _translate_affine_artifact(mlx_root: Path, work_dir: Path) -> Path:
    config_path = work_dir / "crosstl.toml"
    config_path.write_text(_project_config() + "\n", encoding="utf-8")
    output_dir = work_dir / "out"
    report = translate_project(
        load_project_config(mlx_root, config_path),
        targets=("opengl",),
        output_dir=output_dir.relative_to(mlx_root).as_posix(),
        format_output=False,
        validate=True,
        run_toolchains=True,
    )
    payload = report.to_json()

    assert payload["summary"]["unitCount"] == 1
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    assert payload["project"]["subgroupWidthRules"] == {}
    assert payload["project"]["sourceOptions"]["metal"]["target_options"] == {
        "opengl": {"software_subgroup_width": 32}
    }

    artifact = payload["artifacts"][0]
    assert artifact["source"] == MLX_QUANTIZED_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_QUANTIZED_SHA256,
    }
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": MLX_QUANTIZED_OPENGL_ARTIFACT["sha256"],
    }
    assert artifact["generatedSizeBytes"] == (
        MLX_QUANTIZED_OPENGL_ARTIFACT["sizeBytes"]
    )
    assert artifact["entryPoint"] == {
        "source": MLX_QUANTIZED_ENTRY,
        "target": "main",
        "stage": "compute",
    }
    assert artifact["provenance"]["pipeline"] == "entry-scoped-translate"

    materialization = artifact["templateMaterialization"]
    assert materialization["status"] == "materialized"
    assert materialization["specializationCount"] == 3
    assert materialization["accounting"] == {
        "reachableSpecializationCount": 9,
        "dependencyDiscoveryWorkCount": 0,
        "prunedCandidateCount": 104702,
    }
    affine = next(
        item
        for item in materialization["specializations"]
        if item["name"] == "affine_quantize"
    )
    assert affine["parameters"] == {
        "T": "float",
        "bits": "2",
        "group_size": "32",
        "has_global_scale": "false",
    }

    execution = artifact["execution"]
    assert "subgroupWidth" not in execution
    assert "subgroupWidthEnforcement" not in execution
    assert len(execution["entryPoints"]) == 1
    entry = execution["entryPoints"][0]
    assert entry["sourceEntryPoint"] == MLX_QUANTIZED_ENTRY
    assert entry["targetEntryPoint"] == "main"
    assert entry["workgroupSize"] == [32, 1, 1]
    assert "subgroupWidth" not in entry

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert "#define CROSSTL_SOFTWARE_SUBGROUP_WIDTH 32u" in generated
    assert "CROSSTL_REQUIRED_SUBGROUP_WIDTH" not in generated
    assert "GL_KHR_shader_subgroup" not in generated
    assert "gl_Subgroup" not in generated
    assert "subgroupMin" not in generated
    assert "subgroupMax" not in generated
    assert "subgroupShuffleDown" not in generated
    assert "w_min = crossglSoftwareSubgroupMinFloat(w_min);" in generated
    assert "w_max = crossglSoftwareSubgroupMaxFloat(w_max);" in generated
    assert (
        "uint sval = crossglSoftwareSubgroupShuffleDownUint(val, uint(j));" in generated
    )
    assert generated.count("barrier();") == 8
    assert (
        "layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;" in generated
    )

    glslang = shutil.which("glslangValidator")
    spirv_val = shutil.which("spirv-val")
    spirv_dis = shutil.which("spirv-dis")
    if glslang is not None:
        toolchain_runs = payload["validation"]["toolchainRuns"]
        assert len(toolchain_runs) == 1
        assert toolchain_runs[0]["status"] == "ok"
    if glslang is not None and spirv_val is not None and spirv_dis is not None:
        spirv_path = work_dir / "affine-quantize.spv"
        assembly_path = work_dir / "affine-quantize.spvasm"
        subprocess.run(
            [
                glslang,
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
        assert spirv_path.stat().st_size > 0
        subprocess.run(
            [spirv_val, "--target-env", "spv1.3", str(spirv_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [spirv_dis, str(spirv_path), "-o", str(assembly_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        assembly = assembly_path.read_text(encoding="utf-8")
        assert assembly.count("OpControlBarrier") == 8
        assert "OpGroupNonUniform" not in assembly
    elif os.environ.get(REQUIRE_ENV) == "1":
        missing = [
            name
            for name, tool in (
                ("glslangValidator", glslang),
                ("spirv-val", spirv_val),
                ("spirv-dis", spirv_dis),
            )
            if tool is None
        ]
        pytest.fail(
            "The affine OpenGL proof requires these tools: " + ", ".join(missing)
        )

    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True
    return report_path


def test_pinned_mlx_quantized_affine_translates_to_software_subgroup_opengl():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-quantized-affine-opengl-translation-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_affine_artifact(mlx_root, Path(temporary_directory))


def _build_runtime_package(mlx_root: Path, work_dir: Path) -> tuple[dict, Path]:
    report_path = _translate_affine_artifact(mlx_root, work_dir)
    runtime_artifacts = build_runtime_artifact_manifest(report_path)
    assert runtime_artifacts["success"] is True, json.dumps(
        runtime_artifacts,
        indent=2,
    )
    assert runtime_artifacts["summary"]["artifactCount"] == 1
    reflected = runtime_artifacts["artifacts"][0]
    assert reflected["hostInterface"]["status"] == "ready"
    assert {
        resource["name"]: (resource["binding"], resource["access"])
        for resource in reflected["hostInterface"]["resources"]
    } == {
        "wBuffer": (0, "read"),
        "out_Buffer": (1, "read_write"),
        "scalesBuffer": (2, "read_write"),
        "biasesBuffer": (3, "read_write"),
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
    execution_config = descriptor["entryPoint"]["executionConfig"]
    assert execution_config["local_size"] == [32, 1, 1]
    assert execution_config["local_size_x"] == 32
    assert execution_config["local_size_y"] == 1
    assert execution_config["local_size_z"] == 1
    assert "subgroupWidth" not in execution_config
    assert {
        binding["name"]: binding["access"] for binding in descriptor["bindings"]
    } == {
        "wBuffer": "read",
        "out_Buffer": "read_write",
        "scalesBuffer": "read_write",
        "biasesBuffer": "read_write",
    }
    return descriptor, package_dir


def test_pinned_mlx_quantized_affine_executes_through_opengl_native_loader():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-quantized-affine-opengl-native-loader-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
        )
        input_values = [0.0, 1.0, 2.0, 3.0] * 8
        request = build_native_loader_dispatch_request(
            descriptor,
            package_dir,
            {
                "wBuffer": {
                    "dtype": "float32",
                    "shape": [32],
                    "values": input_values,
                }
            },
            {
                "out_Buffer": {
                    "dtype": "uint32",
                    "shape": [8],
                    "values": [27] * 8,
                },
                "scalesBuffer": {
                    "dtype": "float32",
                    "shape": [1],
                    "values": [-1.0],
                    "tolerance": {"absolute": 1e-6, "relative": 1e-6},
                },
                "biasesBuffer": {
                    "dtype": "float32",
                    "shape": [1],
                    "values": [3.0],
                    "tolerance": {"absolute": 1e-6, "relative": 1e-6},
                },
            },
            (1, 1, 1),
            expected_target="opengl",
        )
        assert request.execution_plan is not None
        assert request.execution_plan.diagnostics == ()
        assert request.execution_plan.dispatch.workgroup_size == (32, 1, 1)
        assert request.execution_plan.dispatch.global_size == (32, 1, 1)

        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id="mlx-quantized-affine-opengl-native-loader",
                target="opengl",
                executor="opengl",
                adapter_kind="opengl-native-runtime",
            ),
            runtime_adapter=OpenGLRuntimeParityAdapter(
                runtime=OpenGLComputeRuntime(context_backends=("egl",))
            ),
        )
        availability = executor.is_available(request)
        if not availability.available:
            _skip_or_fail(
                availability.reason or "The native OpenGL runtime is unavailable"
            )
        result = executor.run(request)

    assert result.status == "ok"
    assert result.outputs["out_Buffer"] == {
        "dtype": "uint32",
        "shape": [8],
        "values": [27] * 8,
    }
    assert result.outputs["scalesBuffer"]["values"] == pytest.approx(
        [-1.0], abs=1e-6, rel=1e-6
    )
    assert result.outputs["biasesBuffer"]["values"] == pytest.approx(
        [3.0], abs=1e-6, rel=1e-6
    )
