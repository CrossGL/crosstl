from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
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
    load_dispatch_contract,
    load_project_config,
    translate_project,
    validate_project_report,
)

ROOT = Path(__file__).resolve().parents[2]
MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
MLX_SOFTMAX_SOURCE = "mlx/backend/metal/kernels/softmax.metal"
MLX_SOFTMAX_SHA256 = "d19231c66973edc3944f12529d1cc393029e7f7262b914907c710ef9dbcb39e2"
MLX_SOFTMAX_ENTRY = "block_softmax_float32"
MLX_SOFTMAX_DISPATCH_CONTRACT = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "softmax.native-loader.dispatch.json"
)
MLX_SOFTMAX_DISPATCH_IDENTITY = (
    "1ef1b663b0ff87dbd193bf73dded4dd1c8008e03dcd9c7e8d9ff8aaad832b006"
)
MLX_SOFTMAX_VARIANTS = {
    "block-float32-axis-32-two-rows": {
        "inputs": {"axisSize": 32, "dtype": "float32", "nRows": 2},
        "workgroupSize": [32, 1, 1],
        "dispatchWorkgroupCount": [2, 1, 1],
        "artifactId": (
            "sha256:fa3eb57ad31078c7300855a29c8d67d5f6ac9595e33b651085a9e5fcf3883ea2"
        ),
        "dispatchVariantId": (
            "sha256:59a3029f50afad7ed80bd0533849cf33f1cdd8d2e5586b6b17b82d0e979119e8"
        ),
    },
    "block-float32-axis-2049": {
        "inputs": {"axisSize": 2049, "dtype": "float32", "nRows": 1},
        "workgroupSize": [544, 1, 1],
        "dispatchWorkgroupCount": [1, 1, 1],
        "artifactId": (
            "sha256:956659b706f518421d9f99b08d028ec0a9ceeb359a556a7777c565dfb8e5df37"
        ),
        "dispatchVariantId": (
            "sha256:fa751ebed377b5371cb0a4f91c38eaf8ac95e6430f03ac4a21ef4bddcda91a0d"
        ),
    },
}
MLX_SOFTMAX_GUARDED_ARTIFACTS = {
    "directx": {
        32: {
            "sha256": (
                "8b5540acc90669bc8b4a75985b42ee34c9e45258c63889f703f140b1330337ee"
            ),
            "sizeBytes": 4213,
        },
        544: {
            "sha256": (
                "1c20679115f29d981762165f7c9e1ecd57a641ceff376b2f8d13f33520857f05"
            ),
            "sizeBytes": 4784,
        },
    },
    "opengl": {
        32: {
            "sha256": (
                "5a1da27d4acc858937ddbfc73aed27b2270c9b17e02d6cf9ebb920980dbc5a51"
            ),
            "sizeBytes": 4663,
        },
        544: {
            "sha256": (
                "9d185604faa31f5dd013b2e009683658c063f0105f516fb02891ea3ab2b5485f"
            ),
            "sizeBytes": 4664,
        },
    },
}
MLX_SOFTMAX_SOFTWARE_OPENGL_ARTIFACTS = {
    32: {
        "sha256": "f69dad597cefc34f7908799aaf0ba2eac47a0dcdd91e5f2bf3d7247172fa84b9",
        "sizeBytes": 5585,
    },
    544: {
        "sha256": "eb195e15089f4e7bade380af55e8b7e167c4b89f80b2f25675eb71196a5468ce",
        "sizeBytes": 7204,
    },
}
REQUIRE_DIRECTX_RUNTIME_ENV = "CROSTL_REQUIRE_MLX_SOFTMAX_DIRECTX_NATIVE_LOADER"
REQUIRE_OPENGL_TOOLCHAIN_ENV = "CROSTL_REQUIRE_MLX_SOFTMAX_OPENGL_TOOLCHAIN"
REQUIRE_OPENGL_RUNTIME_ENV = "CROSTL_REQUIRE_MLX_SOFTMAX_OPENGL_NATIVE_LOADER"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _skip_or_fail(message: str, require_env: str) -> None:
    if os.environ.get(require_env) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if any(
            os.environ.get(name) == "1"
            for name in (
                REQUIRE_DIRECTX_RUNTIME_ENV,
                REQUIRE_OPENGL_TOOLCHAIN_ENV,
                REQUIRE_OPENGL_RUNTIME_ENV,
            )
        ):
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")

    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_SOFTMAX_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX Softmax source is missing: {source_path}")
    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_SOFTMAX_SHA256
    return mlx_root


def _evaluated_variants():
    manifest = load_dispatch_contract(MLX_SOFTMAX_DISPATCH_CONTRACT)
    assert manifest.provenance["commit"] == MLX_COMMIT
    assert manifest.content_identity.to_json() == {
        "algorithm": "sha256",
        "value": MLX_SOFTMAX_DISPATCH_IDENTITY,
    }
    variants = {variant.workload_id: variant for variant in manifest.evaluate()}
    assert set(variants) == set(MLX_SOFTMAX_VARIANTS)
    for workload_id, expected in MLX_SOFTMAX_VARIANTS.items():
        variant = variants[workload_id]
        assert variant.inputs == expected["inputs"]
        assert variant.entry_point == MLX_SOFTMAX_ENTRY
        assert list(variant.workgroup_size) == expected["workgroupSize"]
        assert variant.subgroup_width == 32
        assert variant.dispatch_field == "workgroupCount"
        assert list(variant.dispatch_size) == expected["dispatchWorkgroupCount"]
        assert variant.artifact_id == expected["artifactId"]
        assert variant.variant_id == expected["dispatchVariantId"]
    return variants


def test_pinned_mlx_softmax_dispatch_contract_is_exact():
    _evaluated_variants()


def _dispatch_project_config(
    *, output_dir: str, dispatch_contract: str, target: str
) -> str:
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_SOFTMAX_SOURCE}"]
        include_dirs = ["."]
        targets = ["{target}"]
        output_dir = "{output_dir}"
        dispatch_contracts = ["{dispatch_contract}"]

        [project.sources]
        "**/*.metal" = "metal"

        [project.source_options.metal]
        max_template_specializations = 64
        max_template_materialization_work = 4096
        """).strip()


def _selected_project_config(
    *, output_dir: str, target: str, workgroup_size: int
) -> str:
    sections = [textwrap.dedent(f"""
            [project]
            source_roots = ["mlx/backend/metal/kernels"]
            include = ["{MLX_SOFTMAX_SOURCE}"]
            include_dirs = ["."]
            targets = ["{target}"]
            output_dir = "{output_dir}"

            [project.sources]
            "**/*.metal" = "metal"

            [project.entry_points]
            "{MLX_SOFTMAX_SOURCE}" = "{MLX_SOFTMAX_ENTRY}"

            [project.entry_workgroup_size_rules."{MLX_SOFTMAX_SOURCE}"]
            "{MLX_SOFTMAX_ENTRY}" = [{workgroup_size}, 1, 1]
            """).strip()]
    if target == "directx":
        sections.append(textwrap.dedent(f"""
                [project.subgroup_width_rules]
                "{MLX_SOFTMAX_SOURCE}" = 32
                """).strip())
    sections.append(textwrap.dedent("""
            [project.source_options.metal]
            max_template_specializations = 64
            max_template_materialization_work = 4096
            """).strip())
    if target == "opengl":
        sections.append(textwrap.dedent("""
                [project.source_options.metal.target_options.opengl]
                software_subgroup_width = 32
                """).strip())
    return "\n\n".join(sections)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_pinned_mlx_softmax_translates_to_guarded_dispatch_artifacts(target):
    mlx_root = _pinned_mlx_root()
    variants = _evaluated_variants()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-softmax-{target}-dispatch-",
        dir=mlx_root,
    ) as temporary_directory:
        work_dir = Path(temporary_directory)
        contract_path = work_dir / "softmax.dispatch.json"
        shutil.copyfile(MLX_SOFTMAX_DISPATCH_CONTRACT, contract_path)
        output_dir = work_dir / "out"
        config_path = work_dir / "crosstl.toml"
        config_path.write_text(
            _dispatch_project_config(
                output_dir=output_dir.relative_to(mlx_root).as_posix(),
                dispatch_contract=contract_path.relative_to(mlx_root).as_posix(),
                target=target,
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
            run_toolchains=True,
        )
        payload = report.to_json()

        assert payload["summary"]["unitCount"] == 1
        assert payload["summary"]["translatedCount"] == 2
        assert payload["summary"]["failedCount"] == 0
        assert not [
            diagnostic
            for diagnostic in payload["diagnostics"]
            if diagnostic["severity"] == "error"
        ]
        assert len(payload["artifacts"]) == 2
        for artifact in payload["artifacts"]:
            artifact_id = artifact["dispatchArtifact"]["artifactId"]
            variant = next(
                item for item in variants.values() if item.artifact_id == artifact_id
            )
            workgroup_size = variant.workgroup_size[0]
            expected_identity = MLX_SOFTMAX_GUARDED_ARTIFACTS[target][workgroup_size]
            assert artifact["sourceHash"] == {
                "algorithm": "sha256",
                "value": MLX_SOFTMAX_SHA256,
            }
            assert artifact["generatedHash"] == {
                "algorithm": "sha256",
                "value": expected_identity["sha256"],
            }
            assert artifact["generatedSizeBytes"] == expected_identity["sizeBytes"]
            assert artifact["dispatchArtifact"]["workgroupSize"] == list(
                variant.workgroup_size
            )
            assert artifact["dispatchArtifact"]["subgroupWidth"] == 32
            generated = (mlx_root / artifact["path"]).read_text(encoding="utf-8")
            if target == "directx":
                assert f"[numthreads({workgroup_size}, 1, 1)]" in generated
                assert "[WaveSize(32)]" in generated
                assert generated.count("WaveActiveMax(") == 2
                assert generated.count("WaveActiveSum(") == 2
                if workgroup_size == 32:
                    assert "__crossgl_physical_subgroup" not in generated
                else:
                    assert (
                        "groupshared uint __crossgl_physical_subgroup_counter;"
                        in generated
                    )
                    assert (
                        "InterlockedAdd(__crossgl_physical_subgroup_counter"
                        in generated
                    )
                    assert "uint crossglPhysicalSubgroupID = " in generated
            else:
                assert (
                    f"layout(local_size_x = {workgroup_size}, "
                    "local_size_y = 1, local_size_z = 1) in;"
                ) in generated
                assert "#define CROSSTL_REQUIRED_SUBGROUP_WIDTH 32u" in generated
                assert "subgroupMax" in generated
                assert "subgroupAdd" in generated

        if target == "opengl" and shutil.which("glslangValidator") is not None:
            assert len(payload["validation"]["toolchainRuns"]) == 2
            assert {
                run["status"] for run in payload["validation"]["toolchainRuns"]
            } == {"ok"}
        report_path = work_dir / "portability-report.json"
        report.write_json(report_path)
        assert validate_project_report(report_path)["success"] is True
        runtime_artifacts = build_runtime_artifact_manifest(report_path)
        assert runtime_artifacts["success"] is True
        assert runtime_artifacts["summary"]["artifactCount"] == 2


def _assert_software_spirv(
    generated_path: Path,
    work_dir: Path,
    *,
    workgroup_size: int,
) -> None:
    tools = {
        name: shutil.which(name)
        for name in ("glslangValidator", "spirv-val", "spirv-dis")
    }
    if not all(tools.values()):
        if any(
            os.environ.get(name) == "1"
            for name in (REQUIRE_OPENGL_TOOLCHAIN_ENV, REQUIRE_OPENGL_RUNTIME_ENV)
        ):
            missing = [name for name, path in tools.items() if path is None]
            pytest.fail("The Softmax OpenGL proof requires: " + ", ".join(missing))
        return

    spirv_path = work_dir / f"softmax-{workgroup_size}.spv"
    assembly_path = work_dir / f"softmax-{workgroup_size}.spvasm"
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
    assert assembly.count("OpControlBarrier") == 11
    assert "OpGroupNonUniform" not in assembly
    assert f"OpExecutionMode %main LocalSize {workgroup_size} 1 1" in assembly


def _translate_selected_artifact(
    mlx_root: Path,
    work_dir: Path,
    *,
    target: str,
    workload_id: str,
) -> Path:
    variant = _evaluated_variants()[workload_id]
    workgroup_size = variant.workgroup_size[0]
    output_dir = work_dir / "out"
    config_path = work_dir / "crosstl.toml"
    config_path.write_text(
        _selected_project_config(
            output_dir=output_dir.relative_to(mlx_root).as_posix(),
            target=target,
            workgroup_size=workgroup_size,
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
        run_toolchains=True,
    )
    payload = report.to_json()
    assert payload["summary"]["unitCount"] == 1
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    assert not [
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["severity"] == "error"
    ]
    artifact = payload["artifacts"][0]
    expected_identity = (
        MLX_SOFTMAX_GUARDED_ARTIFACTS["directx"][workgroup_size]
        if target == "directx"
        else MLX_SOFTMAX_SOFTWARE_OPENGL_ARTIFACTS[workgroup_size]
    )
    assert artifact["source"] == MLX_SOFTMAX_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_SOFTMAX_SHA256,
    }
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": expected_identity["sha256"],
    }
    assert artifact["generatedSizeBytes"] == expected_identity["sizeBytes"]
    assert artifact["entryPoint"] == {
        "source": MLX_SOFTMAX_ENTRY,
        "target": "CSMain" if target == "directx" else "main",
        "stage": "compute",
    }
    materialization = artifact["templateMaterialization"]
    assert materialization["status"] == "materialized"
    assert materialization["specializationCount"] == 2
    assert materialization["accounting"] == {
        "reachableSpecializationCount": 5,
        "dependencyDiscoveryWorkCount": 11,
        "prunedCandidateCount": 131,
    }
    assert materialization["specializations"][0]["parameters"] == {
        "AccT": "float",
        "N_READS": "SOFTMAX_N_READS",
        "T": "float",
    }
    entry = artifact["execution"]["entryPoints"][0]
    assert entry["sourceEntryPoint"] == variant.entry_point
    assert entry["workgroupSize"] == list(variant.workgroup_size)
    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    if target == "directx":
        assert entry["subgroupWidth"] == 32
        assert "[WaveSize(32)]" in generated
    else:
        assert "subgroupWidth" not in entry
        assert "#define CROSSTL_SOFTWARE_SUBGROUP_WIDTH 32u" in generated
        assert "GL_KHR_shader_subgroup" not in generated
        assert "gl_Subgroup" not in generated
        assert "subgroupMax" not in generated
        assert "subgroupAdd" not in generated
        assert generated.count("crossglSoftwareSubgroupMaxFloat(") == 3
        assert generated.count("crossglSoftwareSubgroupSumFloat(") == 3
        assert generated.count("barrier();") == 11
        if workgroup_size == 32:
            assert "crossglSoftwareSubgroupActive" not in generated
        else:
            assert "uintBitsToFloat(0xff800000u)" in generated
            assert generated.count("crossglSoftwareSubgroupActive") == 6
            assert "uint subgroupBase = invocation - lane;" in generated
        _assert_software_spirv(
            generated_path,
            work_dir,
            workgroup_size=workgroup_size,
        )

    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True
    return report_path


@pytest.mark.parametrize("workload_id", list(MLX_SOFTMAX_VARIANTS))
def test_pinned_mlx_softmax_translates_to_software_subgroup_opengl(workload_id):
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-softmax-software-{workload_id}-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_selected_artifact(
            mlx_root,
            Path(temporary_directory),
            target="opengl",
            workload_id=workload_id,
        )


def _expected_scalar_layouts(target: str) -> dict[str, dict]:
    if target == "directx":
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
            "block_softmax_float32_axis_size_Constants": {
                "physicalType": "int",
                "elementType": "int32",
                "elementSizeBytes": 4,
                "elementStrideBytes": 4,
                "alignmentBytes": 16,
                "memberOffsetBytes": 0,
                "storageLayout": "hlsl-constant-buffer",
                "runtimeSized": False,
                "memberName": "block_softmax_float32_axis_size",
                "blockSizeBytes": 16,
            },
            "in_": dict(runtime_array),
            "out_": dict(runtime_array),
        }

    def runtime_array(member_name: str) -> dict:
        return {
            "physicalType": "float",
            "elementType": "float32",
            "elementSizeBytes": 4,
            "elementStrideBytes": 4,
            "alignmentBytes": 4,
            "memberOffsetBytes": 0,
            "storageLayout": "std430",
            "runtimeSized": True,
            "memberName": member_name,
        }

    return {
        "block_softmax_float32_axis_size_Args": {
            "physicalType": "int",
            "elementType": "int32",
            "elementSizeBytes": 4,
            "elementStrideBytes": 4,
            "alignmentBytes": 16,
            "memberOffsetBytes": 0,
            "storageLayout": "std140",
            "runtimeSized": False,
            "memberName": "axis_size",
            "blockSizeBytes": 16,
        },
        "in_Buffer": runtime_array("in_"),
        "out_Buffer": runtime_array("out_"),
    }


def _build_runtime_package(
    mlx_root: Path,
    work_dir: Path,
    *,
    target: str,
    workload_id: str,
) -> tuple[dict, Path]:
    report_path = _translate_selected_artifact(
        mlx_root,
        work_dir,
        target=target,
        workload_id=workload_id,
    )
    runtime_artifacts = build_runtime_artifact_manifest(report_path)
    assert runtime_artifacts["success"] is True, json.dumps(
        runtime_artifacts,
        indent=2,
    )
    assert runtime_artifacts["summary"]["artifactCount"] == 1
    assert runtime_artifacts["summary"]["specializationConstantCount"] == 0
    reflected = runtime_artifacts["artifacts"][0]
    assert reflected["hostInterface"]["status"] == "ready"
    resources = reflected["hostInterface"]["resources"]
    expected_names = (
        {
            "block_softmax_float32_axis_size_Constants": 2,
            "in_": 0,
            "out_": 1,
        }
        if target == "directx"
        else {
            "block_softmax_float32_axis_size_Args": 2,
            "in_Buffer": 0,
            "out_Buffer": 1,
        }
    )
    assert {resource["name"]: resource["binding"] for resource in resources} == (
        expected_names
    )
    assert {resource["name"]: resource["access"] for resource in resources} == {
        name: "read_write" if name in {"out_", "out_Buffer"} else "read"
        for name in expected_names
    }
    expected_layouts = _expected_scalar_layouts(target)
    assert {
        resource["name"]: resource["scalarLayout"] for resource in resources
    } == expected_layouts

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
    variant = _evaluated_variants()[workload_id]
    assert descriptor["target"] == target
    assert descriptor["entryPoint"]["name"] == (
        "CSMain" if target == "directx" else "main"
    )
    if target == "directx":
        assert descriptor["entryPoint"]["executionConfig"] == {
            "numthreads": list(variant.workgroup_size),
            "subgroupWidth": 32,
        }
    else:
        assert descriptor["entryPoint"]["executionConfig"] == {
            "local_size": list(variant.workgroup_size),
            "local_size_x": variant.workgroup_size[0],
            "local_size_y": 1,
            "local_size_z": 1,
        }
    assert descriptor["specializationConstants"] == []
    assert {
        binding["name"]: binding["scalarLayout"] for binding in descriptor["bindings"]
    } == expected_layouts
    return descriptor, package_dir


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _softmax_workload(workload_id: str) -> tuple[list[float], list[float]]:
    inputs = MLX_SOFTMAX_VARIANTS[workload_id]["inputs"]
    axis_size = inputs["axisSize"]
    n_rows = inputs["nRows"]
    values: list[float] = []
    for row in range(n_rows):
        if axis_size == 32 and row == 0:
            row_values = [_float32((index - 16) / 8.0) for index in range(axis_size)]
        elif axis_size == 32:
            row_values = [
                _float32((((index * 7) % 19) - 9) / 5.0) for index in range(axis_size)
            ]
        else:
            row_values = [
                _float32((((index * 13) % 37) - 18) / 7.0) for index in range(axis_size)
            ]
        values.extend(row_values)

    expected: list[float] = []
    for row in range(n_rows):
        row_values = values[row * axis_size : (row + 1) * axis_size]
        maximum = max(row_values)
        exponentials = [math.exp(value - maximum) for value in row_values]
        normalizer = math.fsum(exponentials)
        expected.extend(_float32(value / normalizer) for value in exponentials)
    return values, expected


def _runtime_request(
    descriptor: dict,
    package_dir: Path,
    *,
    target: str,
    workload_id: str,
):
    variant = _evaluated_variants()[workload_id]
    axis_size = variant.inputs["axisSize"]
    n_rows = variant.inputs["nRows"]
    input_values, expected_values = _softmax_workload(workload_id)
    if target == "directx":
        axis_name = "block_softmax_float32_axis_size_Constants"
        input_name = "in_"
        output_name = "out_"
    else:
        axis_name = "block_softmax_float32_axis_size_Args"
        input_name = "in_Buffer"
        output_name = "out_Buffer"
    request = build_native_loader_dispatch_request(
        descriptor,
        package_dir,
        {
            axis_name: {
                "dtype": "int32",
                "shape": [1],
                "values": [axis_size],
            },
            input_name: {
                "dtype": "float32",
                "shape": [axis_size * n_rows],
                "values": input_values,
            },
        },
        {
            output_name: {
                "dtype": "float32",
                "shape": [axis_size * n_rows],
                "values": expected_values,
                "tolerance": {"absolute": 5e-5, "relative": 5e-5},
            }
        },
        variant.dispatch_size,
        expected_target=target,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    assert request.execution_plan.dispatch.workgroup_size == variant.workgroup_size
    assert request.execution_plan.dispatch.workgroup_count == variant.dispatch_size
    assert request.execution_plan.dispatch.global_size == tuple(
        left * right
        for left, right in zip(variant.workgroup_size, variant.dispatch_size)
    )
    return request, output_name, expected_values


@pytest.mark.parametrize("workload_id", list(MLX_SOFTMAX_VARIANTS))
def test_pinned_mlx_softmax_executes_through_directx_native_loader(workload_id):
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-softmax-directx-{workload_id}-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
            target="directx",
            workload_id=workload_id,
        )
        request, output_name, expected_values = _runtime_request(
            descriptor,
            package_dir,
            target="directx",
            workload_id=workload_id,
        )
        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id=f"mlx-softmax-directx-{workload_id}",
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
                availability.reason or "The native DirectX runtime is unavailable",
                REQUIRE_DIRECTX_RUNTIME_ENV,
            )
        result = executor.run(request)

    assert result.status == "ok"
    assert result.outputs[output_name]["dtype"] == "float32"
    assert result.outputs[output_name]["shape"] == [len(expected_values)]
    assert result.outputs[output_name]["values"] == pytest.approx(
        expected_values,
        abs=5e-5,
        rel=5e-5,
    )


@pytest.mark.parametrize("workload_id", list(MLX_SOFTMAX_VARIANTS))
def test_pinned_mlx_softmax_executes_through_opengl_native_loader(workload_id):
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-softmax-opengl-{workload_id}-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
            target="opengl",
            workload_id=workload_id,
        )
        request, output_name, expected_values = _runtime_request(
            descriptor,
            package_dir,
            target="opengl",
            workload_id=workload_id,
        )
        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id=f"mlx-softmax-opengl-{workload_id}",
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
                availability.reason or "The native OpenGL runtime is unavailable",
                REQUIRE_OPENGL_RUNTIME_ENV,
            )
        result = executor.run(request)

    assert result.status == "ok"
    assert result.outputs[output_name]["dtype"] == "float32"
    assert result.outputs[output_name]["shape"] == [len(expected_values)]
    assert result.outputs[output_name]["values"] == pytest.approx(
        expected_values,
        abs=5e-5,
        rel=5e-5,
    )
