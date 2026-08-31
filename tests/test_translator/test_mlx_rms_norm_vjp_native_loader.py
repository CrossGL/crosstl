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
    NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
    NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH,
    NATIVE_RUNTIME_VARIANT_REGISTRY_PATH,
    DirectXComputeRuntime,
    DirectXRuntimeParityAdapter,
    NativeDeferredCompilationRuntimeError,
    OpenGLComputeRuntime,
    OpenGLRuntimeParityAdapter,
    RuntimeParityExecutor,
    RuntimeTestAdapterSpec,
    build_native_loader_abi_descriptor,
    build_native_loader_abi_package,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    build_runtime_variant_deferred_compilation_request,
    execute_native_deferred_compilation_request,
    load_dispatch_contract,
    load_project_config,
    translate_project,
    validate_project_report,
)
from crosstl.project.directx_toolchain import dxc_compiler_arguments_for_source

ROOT = Path(__file__).resolve().parents[2]
MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
MLX_RMS_NORM_SOURCE = "mlx/backend/metal/kernels/rms_norm.metal"
MLX_RMS_NORM_SHA256 = "b2e04e377fdad1d645581f9beeaf9cbb06d1ad32926161e06cbc15240caf12bf"
MLX_RMS_NORM_VJP_ENTRY = "vjp_rmsfloat32"
MLX_RMS_NORM_VJP_ARTIFACT_ID = (
    "sha256:a9be06b43a6156fb9ee1f9a6955d03d6bda0940c2a8223b58f564c2d12bd0cd0"
)
MLX_RMS_NORM_VJP_VARIANT_ID = (
    "sha256:5f30015f711d2884061ea69c110e54a5ac2e1c03361315e6cac9de2b2c7891a5"
)
MLX_RMS_NORM_VJP_DISPATCH_IDENTITY = (
    "6b80c42a03de10db01881cbf2ca01c119ee4537cb5c221b0be9efcff138edfb3"
)
MLX_RMS_NORM_VJP_DISPATCH_CONTRACT = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "rms_norm_vjp.native-loader.dispatch.json"
)
MLX_RMS_NORM_VJP_GENERATED_ARTIFACTS = {
    "directx": {
        "sha256": "7c1fe2a3c5f6d883b11b3fb17511663ebb3ead2a0931611229930c3f07035c9f",
        "sizeBytes": 6795,
    },
    "opengl": {
        "sha256": "2112adeb6c1693fa42c48fe3013cd57637f34a9393c0d468b547ed06ab42cf73",
        "sizeBytes": 7771,
    },
}
REQUIRE_DIRECTX_PROOF_ENV = "CROSTL_REQUIRE_MLX_RMS_NORM_VJP_DIRECTX_NATIVE_LOADER"
REQUIRE_OPENGL_PROOF_ENV = "CROSTL_REQUIRE_MLX_RMS_NORM_VJP_OPENGL_NATIVE_LOADER"
AXIS_SIZE = 32
ROW_COUNT = 1
TARGET_GROUPS = 512
ROWS_PER_GROUP = 1
EPSILON = 1e-5
HAS_W_CONSTANT_ID = 20
ABSOLUTE_TOLERANCE = 7e-5
RELATIVE_TOLERANCE = 7e-5
INDEX_RANGE_ASSERTIONS = (
    "uint64(row) * axis_size + lid * RMS_N_READS",
    "uint64(gid) * axis_size + lid * RMS_N_READS",
)


def _index_range_assertions() -> str:
    return "\n\n".join(textwrap.dedent(f"""
            [[project.index_range_assertions]]
            source = "{MLX_RMS_NORM_SOURCE}"
            expression = "{expression}"
            minimum = 0
            maximum = 31
            """).strip() for expression in INDEX_RANGE_ASSERTIONS)


def _directx_project_config(*, output_dir: str, dispatch_contract: str) -> str:
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_RMS_NORM_SOURCE}"]
        include_dirs = ["."]
        targets = ["directx"]
        output_dir = "{output_dir}"
        dispatch_contracts = ["{dispatch_contract}"]

        [project.sources]
        "**/*.metal" = "metal"

        {_index_range_assertions()}
        """).strip()


def _opengl_project_config(*, output_dir: str) -> str:
    # This target-specific projection replaces the checked hardware WaveSize
    # contract with an explicit one-workgroup software subgroup. The dispatch
    # fixture remains authoritative for entry, workgroup, and constant values.
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_RMS_NORM_SOURCE}"]
        include_dirs = ["."]
        targets = ["opengl"]
        output_dir = "{output_dir}"

        [project.sources]
        "**/*.metal" = "metal"

        [project.entry_points]
        "{MLX_RMS_NORM_SOURCE}" = "{MLX_RMS_NORM_VJP_ENTRY}"

        [project.entry_workgroup_size_rules."{MLX_RMS_NORM_SOURCE}"]
        "{MLX_RMS_NORM_VJP_ENTRY}" = [32, 1, 1]

        [project.specialization_constants]
        "20" = true

        {_index_range_assertions()}

        [project.source_options.metal.target_options.opengl]
        software_subgroup_width = 32
        """).strip()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _skip_or_fail(message: str, *, require_env: str) -> None:
    if os.environ.get(require_env) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _pinned_mlx_root() -> Path:
    value = os.environ.get("CROSTL_MLX_ROOT")
    if not value:
        if any(
            os.environ.get(name) == "1"
            for name in (REQUIRE_DIRECTX_PROOF_ENV, REQUIRE_OPENGL_PROOF_ENV)
        ):
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")

    mlx_root = Path(value).resolve()
    source_path = mlx_root / MLX_RMS_NORM_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX RMSNorm source is missing: {source_path}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_RMS_NORM_SHA256
    return mlx_root


def _dispatch_variant():
    manifest = load_dispatch_contract(MLX_RMS_NORM_VJP_DISPATCH_CONTRACT)
    assert manifest.provenance["commit"] == MLX_COMMIT
    assert manifest.provenance["sourceReferences"] == {
        "hostDispatch": "mlx/backend/metal/normalization.cpp",
        "kernel": MLX_RMS_NORM_SOURCE,
        "test": "python/tests/test_fast.py::test_rms_norm_grad",
    }
    assert manifest.content_identity.to_json() == {
        "algorithm": "sha256",
        "value": MLX_RMS_NORM_VJP_DISPATCH_IDENTITY,
    }
    variants = manifest.evaluate()
    assert len(variants) == 1
    variant = variants[0]
    assert variant.workload_id == "vjp-float32-axis-32-one-row-has-w"
    assert variant.inputs == {
        "axisSize": AXIS_SIZE,
        "dtype": "float32",
        "hasW": True,
        "isVjp": True,
        "nRows": ROW_COUNT,
        "targetGroups": TARGET_GROUPS,
    }
    assert variant.variant_id == MLX_RMS_NORM_VJP_VARIANT_ID
    assert variant.artifact_id == MLX_RMS_NORM_VJP_ARTIFACT_ID
    assert variant.entry_point == MLX_RMS_NORM_VJP_ENTRY
    assert variant.workgroup_size == (32, 1, 1)
    assert variant.subgroup_width == 32
    assert variant.dispatch_field == "workgroupCount"
    assert variant.dispatch_size == (1, 1, 1)
    assert variant.specialization_constants == {"20": True}
    return variant


def _copy_dispatch_contract(destination: Path) -> None:
    _dispatch_variant()
    shutil.copyfile(MLX_RMS_NORM_VJP_DISPATCH_CONTRACT, destination)


def _assert_directx_compiles(generated_path: Path, work_dir: Path) -> None:
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    source = generated_path.read_text(encoding="utf-8")
    arguments = dxc_compiler_arguments_for_source(source)
    assert arguments == ("-enable-16bit-types",)
    dxil_path = work_dir / "vjp_rmsfloat32.dxil"
    result = subprocess.run(
        [
            dxc,
            *arguments,
            "-WX",
            "-T",
            "cs_6_6",
            "-E",
            "CSMain",
            "-Fo",
            str(dxil_path),
            str(generated_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert dxil_path.stat().st_size > 0


def _assert_opengl_spirv(generated_path: Path, work_dir: Path) -> None:
    tools = {
        name: shutil.which(name)
        for name in ("glslangValidator", "spirv-val", "spirv-dis")
    }
    if not all(tools.values()):
        if os.environ.get(REQUIRE_OPENGL_PROOF_ENV) == "1":
            missing = [name for name, value in tools.items() if value is None]
            pytest.fail("The RMSNorm VJP OpenGL proof requires: " + ", ".join(missing))
        return

    spirv_path = work_dir / "vjp_rmsfloat32.spv"
    assembly_path = work_dir / "vjp_rmsfloat32.spvasm"
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
    assert assembly.count("OpControlBarrier") == 6
    assert "OpGroupNonUniform" not in assembly
    assert "OpSpecConstantTrue" in assembly or "OpSpecConstantFalse" in assembly


def _expected_binding_names(target: str) -> dict[int, str]:
    if target == "directx":
        return {
            0: "x",
            1: "w",
            2: "g",
            3: "gx",
            4: "gw",
            5: "vjp_rmsfloat32_eps_Constants",
            6: "vjp_rmsfloat32_axis_size_Constants",
            7: "vjp_rmsfloat32_w_stride_Constants",
            8: "vjp_rmsfloat32_n_rows_Constants",
            9: "vjp_rmsfloat32_rows_per_group_Constants",
        }
    return {
        0: "xBuffer",
        1: "wBuffer",
        2: "gBuffer",
        3: "gxBuffer",
        4: "gwBuffer",
        5: "vjp_rmsfloat32_eps_Args",
        6: "vjp_rmsfloat32_axis_size_Args",
        7: "vjp_rmsfloat32_w_stride_Args",
        8: "vjp_rmsfloat32_n_rows_Args",
        9: "vjp_rmsfloat32_rows_per_group_Args",
    }


def _expected_layouts(target: str) -> dict[str, dict]:
    runtime_layout = "hlsl-structured-buffer" if target == "directx" else "std430"
    uniform_layout = "hlsl-constant-buffer" if target == "directx" else "std140"
    names = _expected_binding_names(target)

    def runtime_array(member_name: str) -> dict:
        layout = {
            "physicalType": "float",
            "elementType": "float32",
            "elementSizeBytes": 4,
            "elementStrideBytes": 4,
            "alignmentBytes": 4,
            "memberOffsetBytes": 0,
            "storageLayout": runtime_layout,
            "runtimeSized": True,
        }
        if target == "opengl":
            layout["memberName"] = member_name
        return layout

    def uniform(physical: str, element: str, member_name: str) -> dict:
        return {
            "physicalType": physical,
            "elementType": element,
            "elementSizeBytes": 4,
            "elementStrideBytes": 4,
            "alignmentBytes": 16,
            "memberOffsetBytes": 0,
            "storageLayout": uniform_layout,
            "runtimeSized": False,
            "memberName": member_name,
            "blockSizeBytes": 16,
        }

    prefix = "vjp_rmsfloat32_" if target == "directx" else ""
    return {
        names[0]: runtime_array("x"),
        names[1]: runtime_array("w"),
        names[2]: runtime_array("g"),
        names[3]: runtime_array("gx"),
        names[4]: runtime_array("gw"),
        names[5]: uniform("float", "float32", prefix + "eps"),
        names[6]: uniform("uint", "uint32", prefix + "axis_size"),
        names[7]: uniform("uint", "uint32", prefix + "w_stride"),
        names[8]: uniform("uint", "uint32", prefix + "n_rows"),
        names[9]: uniform("uint", "uint32", prefix + "rows_per_group"),
    }


def _translate_artifact(mlx_root: Path, work_dir: Path, target: str) -> Path:
    output_dir = work_dir / "out"
    config_path = work_dir / "crosstl.toml"
    if target == "directx":
        contract_path = work_dir / "rms_norm_vjp.dispatch.json"
        _copy_dispatch_contract(contract_path)
        config_text = _directx_project_config(
            output_dir=output_dir.relative_to(mlx_root).as_posix(),
            dispatch_contract=contract_path.relative_to(mlx_root).as_posix(),
        )
    else:
        _dispatch_variant()
        config_text = _opengl_project_config(
            output_dir=output_dir.relative_to(mlx_root).as_posix(),
        )
    config_path.write_text(config_text + "\n", encoding="utf-8")

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
    assert payload["summary"]["diagnosticCounts"]["error"] == 0
    assert payload["project"]["indexRangeAssertions"] == [
        {
            "source": MLX_RMS_NORM_SOURCE,
            "expression": expression,
            "minimum": 0,
            "maximum": 31,
        }
        for expression in INDEX_RANGE_ASSERTIONS
    ]

    artifact = payload["artifacts"][0]
    expected = MLX_RMS_NORM_VJP_GENERATED_ARTIFACTS[target]
    assert artifact["source"] == MLX_RMS_NORM_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_RMS_NORM_SHA256,
    }
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": expected["sha256"],
    }
    assert artifact["generatedSizeBytes"] == expected["sizeBytes"]
    assert artifact["entryPoint"] == {
        "source": MLX_RMS_NORM_VJP_ENTRY,
        "target": "CSMain" if target == "directx" else "main",
        "stage": "compute",
    }
    constants = artifact["specializationConstants"]
    assert len(constants) == 1
    constant = constants[0]
    assert constant["name"] == "has_w"
    assert constant["id"] == HAS_W_CONSTANT_ID
    assert constant["sourceType"] == "bool"
    assert constant["required"] is True
    assert constant["overridden"] is True
    assert constant["concreteValue"] is True
    assert constant["value"] is True
    assert constant["deferred"] is (target == "opengl")
    assert artifact["specializationMaterialization"]["status"] == (
        "deferred" if target == "opengl" else "concrete"
    )

    materialization = artifact["templateMaterialization"]
    assert materialization["status"] == "materialized"
    assert materialization["specializationCount"] == 1
    assert materialization["accounting"] == {
        "reachableSpecializationCount": 4,
        "dependencyDiscoveryWorkCount": 0,
        "prunedCandidateCount": 168,
    }
    assert materialization["specializations"][0]["materializedName"] == (
        MLX_RMS_NORM_VJP_ENTRY
    )
    assert materialization["specializations"][0]["parameters"] == {
        "N_READS": "RMS_N_READS",
        "T": "float",
    }

    entry = artifact["execution"]["entryPoints"][0]
    assert entry["sourceEntryPoint"] == MLX_RMS_NORM_VJP_ENTRY
    assert entry["workgroupSize"] == [32, 1, 1]
    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    for name in ("local_sumx2", "local_sumgwx", "local_normalizer", "local_meangwx"):
        assert f"vjp_rmsfloat32_{name}" in generated
    if target == "directx":
        assert entry["subgroupWidth"] == 32
        assert artifact["dispatchArtifact"]["artifactId"] == (
            MLX_RMS_NORM_VJP_ARTIFACT_ID
        )
        assert artifact["dispatchArtifact"]["dispatchVariantIds"] == [
            MLX_RMS_NORM_VJP_VARIANT_ID
        ]
        assert "[numthreads(32, 1, 1)]" in generated
        assert "[WaveSize(32)]" in generated
        assert "static const bool has_w = true;" in generated
        assert "constant_id" not in generated
        assert dxc_compiler_arguments_for_source(generated) == ("-enable-16bit-types",)
        assert generated.count("WaveActiveSum(") == 4
        assert generated.count("GroupMemoryBarrierWithGroupSync();") == 3
        _assert_directx_compiles(generated_path, work_dir)
    else:
        assert "subgroupWidth" not in entry
        assert payload["project"]["subgroupWidthRules"] == {}
        assert payload["project"]["specializationConstants"] == {"20": True}
        assert payload["project"]["sourceOptions"]["metal"]["target_options"] == {
            "opengl": {"software_subgroup_width": 32}
        }
        assert "#define CROSSTL_SOFTWARE_SUBGROUP_WIDTH 32u" in generated
        assert "CROSSTL_REQUIRED_SUBGROUP_WIDTH" not in generated
        assert "GL_KHR_shader_subgroup" not in generated
        assert "gl_Subgroup" not in generated
        assert "subgroupAdd" not in generated
        assert "layout(constant_id = 20) const bool has_w = false;" in generated
        assert generated.count("crossglSoftwareSubgroupSumFloat(") == 5
        assert generated.count("barrier();") == 6
        assert "for (uint row = (gid * rows_per_group);" in generated
        _assert_opengl_spirv(generated_path, work_dir)

    toolchain_runs = payload["validation"]["toolchainRuns"]
    if (target == "directx" and shutil.which("dxc")) or (
        target == "opengl" and shutil.which("glslangValidator")
    ):
        assert len(toolchain_runs) == 1
        assert toolchain_runs[0]["status"] == "ok"

    report_path = work_dir / f"{target}-portability-report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True
    return report_path


def _build_runtime_package(
    mlx_root: Path,
    work_dir: Path,
    target: str,
) -> tuple[dict, Path, tuple[Path, dict] | None]:
    report_path = _translate_artifact(mlx_root, work_dir, target)
    runtime_artifacts = build_runtime_artifact_manifest(report_path)
    assert runtime_artifacts["success"] is True
    assert runtime_artifacts["summary"]["artifactCount"] == 1
    assert runtime_artifacts["summary"]["resourceBindingCount"] == 10
    assert runtime_artifacts["summary"]["specializationConstantCount"] == 1
    reflected = runtime_artifacts["artifacts"][0]
    assert reflected["hostInterface"]["status"] == "ready"
    specializations = reflected["hostInterface"]["specializationConstants"]
    assert len(specializations) == 1
    assert specializations[0]["id"] == HAS_W_CONSTANT_ID
    assert specializations[0]["name"] == "has_w"
    assert specializations[0]["value"] is True
    assert specializations[0]["deferred"] is (target == "opengl")
    resources = reflected["hostInterface"]["resources"]
    assert {resource["binding"]: resource["name"] for resource in resources} == (
        _expected_binding_names(target)
    )
    assert {resource["name"]: resource["access"] for resource in resources} == {
        name: "read_write" if binding in (3, 4) else "read"
        for binding, name in _expected_binding_names(target).items()
    }
    assert {
        resource["name"]: resource["scalarLayout"] for resource in resources
    } == _expected_layouts(target)

    runtime_artifacts_path = work_dir / "runtime-artifacts.json"
    _write_json(runtime_artifacts_path, runtime_artifacts)
    package_dir = work_dir / "runtime-package"
    package = build_runtime_package(runtime_artifacts_path, package_dir)
    assert package["success"] is True
    loader = build_runtime_loader_manifest(package_dir / "runtime-package.json")
    assert loader["success"] is True
    assert loader["summary"]["readyLoadUnitCount"] == 1
    assert loader["summary"]["blockedLoadUnitCount"] == 0
    loader_path = package_dir / "runtime-loader-manifest.json"
    _write_json(loader_path, loader)
    descriptor = build_native_loader_abi_descriptor(
        loader,
        load_unit_id=loader["loadUnits"][0]["id"],
    )
    assert descriptor["target"] == target
    assert descriptor["entryPoint"]["name"] == (
        "CSMain" if target == "directx" else "main"
    )
    expected_execution = (
        {"numthreads": [32, 1, 1], "subgroupWidth": 32}
        if target == "directx"
        else {
            "local_size": [32, 1, 1],
            "local_size_x": 32,
            "local_size_y": 1,
            "local_size_z": 1,
        }
    )
    assert descriptor["entryPoint"]["executionConfig"] == expected_execution
    assert len(descriptor["specializationConstants"]) == 1
    assert descriptor["specializationConstants"][0]["id"] == HAS_W_CONSTANT_ID
    assert descriptor["specializationConstants"][0]["value"] is True
    assert descriptor["specializationConstants"][0]["deferred"] is (target == "opengl")
    assert {
        binding["name"]: binding["scalarLayout"] for binding in descriptor["bindings"]
    } == _expected_layouts(target)

    deferred = None
    if target == "opengl":
        abi_root = work_dir / "native-loader-abi-package"
        abi_package = build_native_loader_abi_package(loader_path, abi_root)
        assert abi_package["success"] is True
        assert abi_package["summary"]["runtimeVariantCount"] == 1
        assert abi_package["runtimeVariantRegistry"]["available"] is True
        assert abi_package["runtimeVariantRegistry"]["nativeHeader"] == {
            "available": False,
            "reason": "specialization-requires-deferred-compilation",
        }
        assert not (abi_root / NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH).exists()
        assert (abi_root / NATIVE_LOADER_ABI_PACKAGE_MANIFEST).is_file()
        registry = json.loads(
            (abi_root / NATIVE_RUNTIME_VARIANT_REGISTRY_PATH).read_text(
                encoding="utf-8"
            )
        )
        assert registry["status"] == "ready"
        assert registry["summary"]["readyVariantCount"] == 1
        assert registry["summary"]["blockedVariantCount"] == 0
        key = registry["lookup"]["readyKeys"][0]
        request = build_runtime_variant_deferred_compilation_request(
            registry,
            key,
            abi_root,
        )
        assert request["source"] == {
            "path": descriptor["artifact"]["packagePath"],
            "format": "GLSL source",
            "hash": descriptor["artifact"]["hash"],
            "sizeBytes": descriptor["artifact"]["sizeBytes"],
        }
        assert request["target"] == {
            "backend": "opengl",
            "profile": None,
            "stage": "compute",
            "entryPoint": "main",
            "outputFormat": "SPIR-V binary",
        }
        assert request["variant"]["specializationValues"] == [
            {"id": HAS_W_CONSTANT_ID, "name": "has_w", "value": True}
        ]
        assert request["variant"]["execution"] == {
            "workgroupSize": [32, 1, 1],
            "subgroupWidth": None,
        }
        deferred = (abi_root, request)
    return descriptor, package_dir, deferred


def _workload() -> (
    tuple[list[float], list[float], list[float], list[float], list[float]]
):
    values = [(index - 16) / 8.0 for index in range(AXIS_SIZE)]
    weights = [0.625 + (index % 7) * 0.09375 for index in range(AXIS_SIZE)]
    gradients = [((index % 11) - 5) * 0.1875 + 0.03125 for index in range(AXIS_SIZE)]
    mean_square = math.fsum(value * value for value in values) / AXIS_SIZE
    normalizer = 1.0 / math.sqrt(mean_square + EPSILON)
    mean_gwx = (
        math.fsum(
            value * weight * gradient
            for value, weight, gradient in zip(values, weights, gradients)
        )
        / AXIS_SIZE
    )
    normalizer3 = normalizer * normalizer * normalizer
    expected_gx = [
        gradient * weight * normalizer - value * mean_gwx * normalizer3
        for value, weight, gradient in zip(values, weights, gradients)
    ]
    expected_gw = [
        gradient * value * normalizer for value, gradient in zip(values, gradients)
    ]
    return values, weights, gradients, expected_gx, expected_gw


def _runtime_values(target: str):
    names = _expected_binding_names(target)
    values, weights, gradients, expected_gx, expected_gw = _workload()
    inputs = {
        names[0]: {"dtype": "float32", "shape": [32], "values": values},
        names[1]: {"dtype": "float32", "shape": [32], "values": weights},
        names[2]: {"dtype": "float32", "shape": [32], "values": gradients},
        names[5]: {"dtype": "float32", "shape": [1], "values": [EPSILON]},
        names[6]: {"dtype": "uint32", "shape": [1], "values": [AXIS_SIZE]},
        names[7]: {"dtype": "uint32", "shape": [1], "values": [1]},
        names[8]: {"dtype": "uint32", "shape": [1], "values": [ROW_COUNT]},
        names[9]: {
            "dtype": "uint32",
            "shape": [1],
            "values": [ROWS_PER_GROUP],
        },
    }
    outputs = {
        names[3]: {
            "dtype": "float32",
            "shape": [32],
            "values": expected_gx,
            "tolerance": {
                "absolute": ABSOLUTE_TOLERANCE,
                "relative": RELATIVE_TOLERANCE,
            },
        },
        names[4]: {
            "dtype": "float32",
            "shape": [32],
            "values": expected_gw,
            "tolerance": {
                "absolute": ABSOLUTE_TOLERANCE,
                "relative": RELATIVE_TOLERANCE,
            },
        },
    }
    return inputs, outputs, expected_gx, expected_gw


def _directx_dispatch_request(descriptor: dict, package_dir: Path):
    inputs, outputs, expected_gx, expected_gw = _runtime_values("directx")
    request = build_native_loader_dispatch_request(
        descriptor,
        package_dir,
        inputs,
        outputs,
        (1, 1, 1),
        {HAS_W_CONSTANT_ID: True},
        expected_target="directx",
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    assert request.execution_plan.dispatch.workgroup_size == (32, 1, 1)
    assert request.execution_plan.dispatch.workgroup_count == (1, 1, 1)
    assert request.execution_plan.dispatch.global_size == (32, 1, 1)
    assert len(request.adapter_contract.specialization_constants) == 1
    constant = request.adapter_contract.specialization_constants[0]
    assert constant.constant_id == HAS_W_CONSTANT_ID
    assert constant.value is True
    assert constant.metadata["mechanism"] == "compiled"
    return request, expected_gx, expected_gw


def test_rms_norm_vjp_native_loader_dispatch_contract_is_exact():
    _dispatch_variant()


def test_pinned_mlx_rms_norm_vjp_translates_to_directx_native_loader_artifact():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-rms-norm-vjp-directx-translation-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_artifact(mlx_root, Path(temporary_directory), "directx")


def test_pinned_mlx_rms_norm_vjp_translates_to_deferred_software_opengl():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-rms-norm-vjp-opengl-translation-",
        dir=mlx_root,
    ) as temporary_directory:
        _descriptor, _package_dir, deferred = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
            "opengl",
        )
        assert deferred is not None


def test_pinned_mlx_rms_norm_vjp_executes_through_directx_native_loader():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-rms-norm-vjp-directx-native-loader-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir, deferred = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
            "directx",
        )
        assert deferred is None
        request, expected_gx, expected_gw = _directx_dispatch_request(
            descriptor,
            package_dir,
        )
        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id="mlx-rms-norm-vjp-directx-native-loader",
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
                require_env=REQUIRE_DIRECTX_PROOF_ENV,
            )
        result = executor.run(request)

    assert result.status == "ok"
    names = _expected_binding_names("directx")
    assert result.outputs[names[3]]["dtype"] == "float32"
    assert result.outputs[names[3]]["shape"] == [32]
    assert result.outputs[names[3]]["values"] == pytest.approx(
        expected_gx,
        abs=ABSOLUTE_TOLERANCE,
        rel=RELATIVE_TOLERANCE,
    )
    assert result.outputs[names[4]]["dtype"] == "float32"
    assert result.outputs[names[4]]["shape"] == [32]
    assert result.outputs[names[4]]["values"] == pytest.approx(
        expected_gw,
        abs=ABSOLUTE_TOLERANCE,
        rel=RELATIVE_TOLERANCE,
    )


def test_pinned_mlx_rms_norm_vjp_executes_through_opengl_native_loader():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-rms-norm-vjp-opengl-native-loader-",
        dir=mlx_root,
    ) as temporary_directory:
        work_dir = Path(temporary_directory)
        _descriptor, _package_dir, deferred = _build_runtime_package(
            mlx_root,
            work_dir,
            "opengl",
        )
        assert deferred is not None
        abi_root, compilation_request = deferred
        inputs, outputs, expected_gx, expected_gw = _runtime_values("opengl")
        try:
            result = execute_native_deferred_compilation_request(
                compilation_request,
                abi_root,
                work_dir / "deferred-cache",
                inputs,
                outputs,
                (1, 1, 1),
                runtime_adapter=OpenGLRuntimeParityAdapter(
                    runtime=OpenGLComputeRuntime(context_backends=("egl",))
                ),
            )
        except NativeDeferredCompilationRuntimeError as exc:
            if exc.code.endswith(".runtime-unavailable"):
                _skip_or_fail(
                    str(exc),
                    require_env=REQUIRE_OPENGL_PROOF_ENV,
                )
            raise

    assert result.status == "ok"
    names = _expected_binding_names("opengl")
    assert result.outputs[names[3]]["dtype"] == "float32"
    assert result.outputs[names[3]]["shape"] == [32]
    assert result.outputs[names[3]]["values"] == pytest.approx(
        expected_gx,
        abs=ABSOLUTE_TOLERANCE,
        rel=RELATIVE_TOLERANCE,
    )
    assert result.outputs[names[4]]["dtype"] == "float32"
    assert result.outputs[names[4]]["shape"] == [32]
    assert result.outputs[names[4]]["values"] == pytest.approx(
        expected_gw,
        abs=ABSOLUTE_TOLERANCE,
        rel=RELATIVE_TOLERANCE,
    )
    deferred_report = result.details["nativeDeferredCompilation"]
    assert deferred_report["success"] is True
    assert deferred_report["target"]["backend"] == "opengl"
    assert deferred_report["variant"]["specializationValues"] == [
        {"id": HAS_W_CONSTANT_ID, "name": "has_w", "value": True}
    ]
    assert deferred_report["interface"]["status"] == "verified"
    assert deferred_report["cache"]["status"] == "published"
