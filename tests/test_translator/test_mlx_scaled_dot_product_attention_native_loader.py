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
MLX_ATTENTION_SOURCE = "mlx/backend/metal/kernels/scaled_dot_product_attention.metal"
MLX_ATTENTION_SHA256 = (
    "f6fefad1d91b01f05c12095e69f248e255f880d65b5ae9e9d8bc2714da56fb41"
)
MLX_ATTENTION_ENTRY = "sdpa_vector_float_64_64"
MLX_ATTENTION_ARTIFACT_ID = (
    "sha256:dd0138695bd82e1f8ea49bd667052b484420ee96cb2849c6eed20ba5eae39a89"
)
MLX_ATTENTION_VARIANT_ID = (
    "sha256:8b2abb9f7179e051530697fb8d1956d0ff03a324e7acaa5fcdf4f4dd9f1befbb"
)
MLX_ATTENTION_DISPATCH_IDENTITY = (
    "97e5ebb69af8da3a0082776015787456f23b8bfdb0cff757f5364db2cfef8d2c"
)
MLX_ATTENTION_DISPATCH_CONTRACT = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "scaled_dot_product_attention.native-loader.dispatch.json"
)
MLX_ATTENTION_GENERATED_ARTIFACTS = {
    "directx": {
        "sha256": "003c8b9e85bad7363bae2e3d80380d979cbe0b8988d0d98751131c3acfbff6b6",
        "sizeBytes": 8721,
    },
    "opengl": {
        "sha256": "9b7cb7dc9a76b9fb93c30fd93d13ad639f5493f60fd97b965514db0fe6b4840b",
        "sizeBytes": 12089,
    },
}
REQUIRE_DIRECTX_PROOF_ENV = "CROSTL_REQUIRE_MLX_ATTENTION_DIRECTX_NATIVE_LOADER"
REQUIRE_OPENGL_PROOF_ENV = "CROSTL_REQUIRE_MLX_ATTENTION_OPENGL_NATIVE_LOADER"
DIMENSION = 64
KEY_LENGTH = 4
SCALE = 0.125
ABSOLUTE_TOLERANCE = 2e-4
RELATIVE_TOLERANCE = 2e-4
SPECIALIZATION_VALUES = {constant_id: False for constant_id in range(20, 26)}
SPECIALIZATION_NAMES = {
    20: "has_mask",
    21: "query_transposed",
    22: "do_causal",
    23: "bool_mask",
    24: "float_mask",
    25: "has_sinks",
}


def _project_config(
    target: str,
    *,
    output_dir: str,
    dispatch_contract: str | None = None,
) -> str:
    sections = [textwrap.dedent(f"""
            [project]
            source_roots = ["mlx/backend/metal/kernels"]
            include = ["{MLX_ATTENTION_SOURCE}"]
            include_dirs = ["."]
            targets = ["{target}"]
            output_dir = "{output_dir}"
            {f'dispatch_contracts = ["{dispatch_contract}"]' if dispatch_contract else ''}

            [project.sources]
            "**/*.metal" = "metal"
            """).strip()]
    if dispatch_contract is None:
        sections.append(textwrap.dedent(f"""
                [project.entry_points]
                "{MLX_ATTENTION_SOURCE}" = "{MLX_ATTENTION_ENTRY}"

                [project.entry_workgroup_size_rules."{MLX_ATTENTION_SOURCE}"]
                "{MLX_ATTENTION_ENTRY}" = [1024, 1, 1]

                [project.specialization_constants]
                "20" = false
                "21" = false
                "22" = false
                "23" = false
                "24" = false
                "25" = false
                """).strip())
    sections.append(textwrap.dedent("""
            [project.source_options.metal]
            max_template_specializations = 128
            max_template_materialization_work = 16384
            """).strip())
    if target == "directx":
        if dispatch_contract is None:
            sections.append(textwrap.dedent(f"""
                    [project.subgroup_width_rules]
                    "{MLX_ATTENTION_SOURCE}" = 32
                    """).strip())
    elif target == "opengl":
        sections.append(textwrap.dedent("""
                [project.source_options.metal.target_options.opengl]
                software_subgroup_width = 32
                """).strip())
    else:
        raise ValueError(f"Unsupported attention test target: {target}")
    return "\n\n".join(sections)


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
    source_path = mlx_root / MLX_ATTENTION_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX attention source is missing: {source_path}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
        MLX_ATTENTION_SHA256
    )
    return mlx_root


def _dispatch_variant():
    manifest = load_dispatch_contract(MLX_ATTENTION_DISPATCH_CONTRACT)
    assert manifest.provenance["commit"] == MLX_COMMIT
    assert manifest.provenance["sourceReferences"] == {
        "hostDispatch": "mlx/backend/metal/scaled_dot_product_attention.cpp",
        "implementation": "mlx/backend/metal/kernels/sdpa_vector.h",
        "kernel": MLX_ATTENTION_SOURCE,
    }
    assert manifest.content_identity.to_json() == {
        "algorithm": "sha256",
        "value": MLX_ATTENTION_DISPATCH_IDENTITY,
    }
    variants = manifest.evaluate()
    assert len(variants) == 1
    variant = variants[0]
    assert variant.workload_id == "vector-float32-b1-h1-q1-k4-d64-v64-nomask"
    assert variant.inputs == {
        "batchSize": 1,
        "boolMask": False,
        "doCausal": False,
        "dtype": "float32",
        "floatMask": False,
        "hasMask": False,
        "hasSinks": False,
        "keyLength": KEY_LENGTH,
        "kvHeads": 1,
        "queryDimension": DIMENSION,
        "queryHeads": 1,
        "queryLength": 1,
        "queryTransposed": False,
        "valueDimension": DIMENSION,
    }
    assert variant.variant_id == MLX_ATTENTION_VARIANT_ID
    assert variant.artifact_id == MLX_ATTENTION_ARTIFACT_ID
    assert variant.entry_point == MLX_ATTENTION_ENTRY
    assert variant.workgroup_size == (1024, 1, 1)
    assert variant.subgroup_width == 32
    assert variant.dispatch_field == "workgroupCount"
    assert variant.dispatch_size == (1, 1, 1)
    assert variant.specialization_constants == {
        str(key): value for key, value in SPECIALIZATION_VALUES.items()
    }
    return variant


def _copy_dispatch_contract(destination: Path) -> None:
    _dispatch_variant()
    shutil.copyfile(MLX_ATTENTION_DISPATCH_CONTRACT, destination)


def _assert_directx_compiles(generated_path: Path, work_dir: Path) -> None:
    dxc = shutil.which("dxc")
    if dxc is None:
        if os.environ.get(REQUIRE_DIRECTX_PROOF_ENV) == "1":
            pytest.fail("The attention DirectX proof requires dxc")
        return
    source = generated_path.read_text(encoding="utf-8")
    arguments = dxc_compiler_arguments_for_source(source)
    assert arguments == ("-enable-16bit-types",)
    dxil_path = work_dir / "sdpa_vector_float_64_64.dxil"
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
            pytest.fail("The attention OpenGL proof requires: " + ", ".join(missing))
        return

    spirv_path = work_dir / "sdpa_vector_float_64_64.spv"
    assembly_path = work_dir / "sdpa_vector_float_64_64.spvasm"
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
    assert assembly.count("OpControlBarrier") == 9
    assert "OpGroupNonUniform" not in assembly
    assert assembly.count("OpSpecConstantFalse") == 6
    assert "OpExecutionMode %main LocalSize 1024 1 1" in assembly


def _expected_binding_names(target: str) -> dict[int, str]:
    if target == "directx":
        return {
            0: "queries",
            1: "keys",
            2: "values",
            3: "out_",
            4: f"{MLX_ATTENTION_ENTRY}_gqa_factor_Constants",
            5: f"{MLX_ATTENTION_ENTRY}_N_Constants",
            6: f"{MLX_ATTENTION_ENTRY}_k_head_stride_Constants",
            7: f"{MLX_ATTENTION_ENTRY}_k_seq_stride_Constants",
            8: f"{MLX_ATTENTION_ENTRY}_v_head_stride_Constants",
            9: f"{MLX_ATTENTION_ENTRY}_v_seq_stride_Constants",
            10: f"{MLX_ATTENTION_ENTRY}_scale_Constants",
            11: "bmask",
            12: "fmask",
            13: f"{MLX_ATTENTION_ENTRY}_mask_kv_seq_stride_Constants",
            14: f"{MLX_ATTENTION_ENTRY}_mask_q_seq_stride_Constants",
            15: f"{MLX_ATTENTION_ENTRY}_mask_head_stride_Constants",
            16: "sinks",
            17: f"{MLX_ATTENTION_ENTRY}_num_q_heads_Constants",
            18: "CrossGLDispatchInfo",
        }
    return {
        0: "queriesBuffer",
        1: "keysBuffer",
        2: "valuesBuffer",
        3: "out_Buffer",
        4: f"{MLX_ATTENTION_ENTRY}_gqa_factor_Args",
        5: f"{MLX_ATTENTION_ENTRY}_N_Args",
        6: f"{MLX_ATTENTION_ENTRY}_k_head_stride_Args",
        7: f"{MLX_ATTENTION_ENTRY}_k_seq_stride_Args",
        8: f"{MLX_ATTENTION_ENTRY}_v_head_stride_Args",
        9: f"{MLX_ATTENTION_ENTRY}_v_seq_stride_Args",
        10: f"{MLX_ATTENTION_ENTRY}_scale_Args",
        11: "bmaskBuffer",
        12: "fmaskBuffer",
        13: f"{MLX_ATTENTION_ENTRY}_mask_kv_seq_stride_Args",
        14: f"{MLX_ATTENTION_ENTRY}_mask_q_seq_stride_Args",
        15: f"{MLX_ATTENTION_ENTRY}_mask_head_stride_Args",
        16: "sinksBuffer",
        17: f"{MLX_ATTENTION_ENTRY}_num_q_heads_Args",
    }


def _translate_artifact(mlx_root: Path, work_dir: Path, target: str) -> Path:
    output_dir = work_dir / "out"
    config_path = work_dir / "crosstl.toml"
    dispatch_contract = None
    if target == "directx":
        contract_path = work_dir / "attention.dispatch.json"
        _copy_dispatch_contract(contract_path)
        dispatch_contract = contract_path.relative_to(mlx_root).as_posix()
    else:
        _dispatch_variant()
    config_path.write_text(
        _project_config(
            target,
            output_dir=output_dir.relative_to(mlx_root).as_posix(),
            dispatch_contract=dispatch_contract,
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
    assert payload["summary"]["diagnosticCounts"]["error"] == 0, json.dumps(
        payload["diagnostics"], indent=2
    )

    artifact = payload["artifacts"][0]
    expected = MLX_ATTENTION_GENERATED_ARTIFACTS[target]
    assert artifact["source"] == MLX_ATTENTION_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_ATTENTION_SHA256,
    }
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": expected["sha256"],
    }
    assert artifact["generatedSizeBytes"] == expected["sizeBytes"]
    assert artifact["entryPoint"] == {
        "source": MLX_ATTENTION_ENTRY,
        "target": "CSMain" if target == "directx" else "main",
        "stage": "compute",
    }
    constants = artifact["specializationConstants"]
    assert [(item["id"], item["name"]) for item in constants] == list(
        SPECIALIZATION_NAMES.items()
    )
    for constant in constants:
        assert constant["sourceType"] == "bool"
        assert constant["required"] is True
        assert constant["overridden"] is True
        assert constant["concreteValue"] is False
        assert constant["value"] is False
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
        "prunedCandidateCount": 753,
    }
    assert materialization["specializations"][0]["materializedName"] == (
        MLX_ATTENTION_ENTRY
    )
    assert materialization["specializations"][0]["parameters"] == {
        "D": "64",
        "T": "float",
        "V": "64",
    }

    entry = artifact["execution"]["entryPoints"][0]
    assert entry["sourceEntryPoint"] == MLX_ATTENTION_ENTRY
    assert entry["workgroupSize"] == [1024, 1, 1]
    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    if target == "directx":
        assert entry["subgroupWidth"] == 32
        assert artifact["dispatchArtifact"]["artifactId"] == (MLX_ATTENTION_ARTIFACT_ID)
        assert artifact["dispatchArtifact"]["dispatchVariantIds"] == [
            MLX_ATTENTION_VARIANT_ID
        ]
        assert "[numthreads(1024, 1, 1)]" in generated
        assert "[WaveSize(32)]" in generated
        assert "static const bool has_mask = false;" in generated
        assert generated.count("WaveActiveSum(") == 3
        assert generated.count("WaveActiveMax(") == 1
        assert "groupshared uint __crossgl_physical_subgroup_counter;" in generated
        assert "InterlockedAdd(__crossgl_physical_subgroup_counter" in generated
        assert "uint crossglPhysicalSubgroupID = " in generated
        _assert_directx_compiles(generated_path, work_dir)
    else:
        assert "subgroupWidth" not in entry
        assert payload["project"]["subgroupWidthRules"] == {}
        assert payload["project"]["sourceOptions"]["metal"]["target_options"] == {
            "opengl": {"software_subgroup_width": 32}
        }
        assert "#define CROSSTL_SOFTWARE_SUBGROUP_WIDTH 32u" in generated
        assert "GL_KHR_shader_subgroup" not in generated
        assert "gl_Subgroup" not in generated
        assert "subgroupAdd" not in generated
        assert "subgroupMax" not in generated
        assert "crossglSoftwareSubgroupBase" in generated
        assert "crossglSoftwareSubgroupLoopActive" in generated
        assert generated.count("layout(constant_id = ") == 6
        assert generated.count("barrier();") == 9
        _assert_opengl_spirv(generated_path, work_dir)

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
    assert runtime_artifacts["summary"]["resourceBindingCount"] == (
        19 if target == "directx" else 18
    )
    assert runtime_artifacts["summary"]["specializationConstantCount"] == 6
    reflected = runtime_artifacts["artifacts"][0]
    assert reflected["hostInterface"]["status"] == "ready"
    resources = reflected["hostInterface"]["resources"]
    assert {resource["binding"]: resource["name"] for resource in resources} == (
        _expected_binding_names(target)
    )
    assert all(resource.get("scalarLayout") is not None for resource in resources)
    bool_name = _expected_binding_names(target)[11]
    bool_resource = next(item for item in resources if item["name"] == bool_name)
    assert bool_resource["scalarLayout"] == {
        "physicalType": "uint",
        "elementType": "uint32",
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
        "alignmentBytes": 4,
        "memberOffsetBytes": 0,
        "storageLayout": "hlsl-structured-buffer" if target == "directx" else "std430",
        "runtimeSized": True,
        **({"memberName": "bmask"} if target == "opengl" else {}),
    }

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
        {"numthreads": [1024, 1, 1], "subgroupWidth": 32}
        if target == "directx"
        else {
            "local_size": [1024, 1, 1],
            "local_size_x": 1024,
            "local_size_y": 1,
            "local_size_z": 1,
        }
    )
    assert descriptor["entryPoint"]["executionConfig"] == expected_execution
    assert [
        (item["id"], item["name"], item["value"], item["deferred"])
        for item in descriptor["specializationConstants"]
    ] == [
        (constant_id, name, False, target == "opengl")
        for constant_id, name in SPECIALIZATION_NAMES.items()
    ]

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
        key = registry["lookup"]["readyKeys"][0]
        request = build_runtime_variant_deferred_compilation_request(
            registry,
            key,
            abi_root,
        )
        assert request["variant"]["specializationValues"] == [
            {"id": constant_id, "name": name, "value": False}
            for constant_id, name in SPECIALIZATION_NAMES.items()
        ]
        assert request["variant"]["execution"] == {
            "workgroupSize": [1024, 1, 1],
            "subgroupWidth": None,
        }
        deferred = (abi_root, request)
    return descriptor, package_dir, deferred


def _workload() -> tuple[list[float], list[float], list[float], list[float]]:
    queries = [(((index * 7) % 19) - 9) / 16.0 for index in range(DIMENSION)]
    keys = [
        ((((key + 1) * 11 + index * 5) % 23) - 11) / 16.0
        for key in range(KEY_LENGTH)
        for index in range(DIMENSION)
    ]
    values = [
        ((((key + 2) * 13 + index * 3) % 29) - 14) / 12.0
        for key in range(KEY_LENGTH)
        for index in range(DIMENSION)
    ]
    scores = [
        SCALE
        * math.fsum(
            queries[index] * keys[key * DIMENSION + index] for index in range(DIMENSION)
        )
        for key in range(KEY_LENGTH)
    ]
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    denominator = math.fsum(exponentials)
    expected = [
        math.fsum(
            exponentials[key] * values[key * DIMENSION + index]
            for key in range(KEY_LENGTH)
        )
        / denominator
        for index in range(DIMENSION)
    ]
    return queries, keys, values, expected


def _runtime_values(target: str):
    names = _expected_binding_names(target)
    queries, keys, values, expected = _workload()
    inputs = {
        names[0]: {"dtype": "float32", "shape": [DIMENSION], "values": queries},
        names[1]: {
            "dtype": "float32",
            "shape": [KEY_LENGTH * DIMENSION],
            "values": keys,
        },
        names[2]: {
            "dtype": "float32",
            "shape": [KEY_LENGTH * DIMENSION],
            "values": values,
        },
        names[4]: {"dtype": "int32", "shape": [1], "values": [1]},
        names[5]: {"dtype": "int32", "shape": [1], "values": [KEY_LENGTH]},
        names[6]: {
            "dtype": "uint64",
            "shape": [1],
            "values": [KEY_LENGTH * DIMENSION],
        },
        names[7]: {"dtype": "uint64", "shape": [1], "values": [DIMENSION]},
        names[8]: {
            "dtype": "uint64",
            "shape": [1],
            "values": [KEY_LENGTH * DIMENSION],
        },
        names[9]: {"dtype": "uint64", "shape": [1], "values": [DIMENSION]},
        names[10]: {"dtype": "float32", "shape": [1], "values": [SCALE]},
        names[11]: {"dtype": "uint32", "shape": [1], "values": [0]},
        names[12]: {"dtype": "float32", "shape": [1], "values": [0.0]},
        names[13]: {"dtype": "int32", "shape": [1], "values": [0]},
        names[14]: {"dtype": "int32", "shape": [1], "values": [0]},
        names[15]: {"dtype": "int32", "shape": [1], "values": [0]},
        names[16]: {"dtype": "float32", "shape": [1], "values": [0.0]},
        names[17]: {"dtype": "int32", "shape": [1], "values": [1]},
    }
    outputs = {
        names[3]: {
            "dtype": "float32",
            "shape": [DIMENSION],
            "values": expected,
            "tolerance": {
                "absolute": ABSOLUTE_TOLERANCE,
                "relative": RELATIVE_TOLERANCE,
            },
        }
    }
    return inputs, outputs, expected


def _dispatch_request(descriptor: dict, package_dir: Path, target: str):
    inputs, outputs, expected = _runtime_values(target)
    request = build_native_loader_dispatch_request(
        descriptor,
        package_dir,
        inputs,
        outputs,
        (1, 1, 1),
        SPECIALIZATION_VALUES,
        expected_target=target,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    assert request.execution_plan.dispatch.workgroup_size == (1024, 1, 1)
    assert request.execution_plan.dispatch.workgroup_count == (1, 1, 1)
    assert request.execution_plan.dispatch.global_size == (1024, 1, 1)
    assert len(request.adapter_contract.resource_bindings) == (
        19 if target == "directx" else 18
    )
    assert len(request.adapter_contract.specialization_constants) == 6
    return request, expected


def test_attention_native_loader_dispatch_contract_is_exact():
    _dispatch_variant()


def test_pinned_mlx_attention_translates_to_directx_native_loader_artifact():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-sdpa-dx-translate-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir, deferred = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
            "directx",
        )
        assert deferred is None
        _dispatch_request(descriptor, package_dir, "directx")


def test_pinned_mlx_attention_translates_to_deferred_software_opengl():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-attention-opengl-translation-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir, deferred = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
            "opengl",
        )
        assert deferred is not None
        _dispatch_request(descriptor, package_dir, "opengl")


def test_pinned_mlx_attention_executes_through_directx_native_loader():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        # Keep the deeply nested dispatch artifact below legacy dxc.exe MAX_PATH.
        prefix=".crosstl-sdpa-dx-runtime-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir, deferred = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
            "directx",
        )
        assert deferred is None
        request, expected = _dispatch_request(descriptor, package_dir, "directx")
        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id="mlx-attention-directx-native-loader",
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
    output_name = _expected_binding_names("directx")[3]
    assert result.outputs[output_name]["dtype"] == "float32"
    assert result.outputs[output_name]["shape"] == [DIMENSION]
    assert result.outputs[output_name]["values"] == pytest.approx(
        expected,
        abs=ABSOLUTE_TOLERANCE,
        rel=RELATIVE_TOLERANCE,
    )


def test_pinned_mlx_attention_executes_through_opengl_native_loader():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-attention-opengl-native-loader-",
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
        inputs, outputs, expected = _runtime_values("opengl")
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
                _skip_or_fail(str(exc), require_env=REQUIRE_OPENGL_PROOF_ENV)
            raise

    assert result.status == "ok"
    output_name = _expected_binding_names("opengl")[3]
    assert result.outputs[output_name]["dtype"] == "float32"
    assert result.outputs[output_name]["shape"] == [DIMENSION]
    assert result.outputs[output_name]["values"] == pytest.approx(
        expected,
        abs=ABSOLUTE_TOLERANCE,
        rel=RELATIVE_TOLERANCE,
    )
    deferred_report = result.details["nativeDeferredCompilation"]
    assert deferred_report["success"] is True
    assert deferred_report["target"]["backend"] == "opengl"
    assert deferred_report["variant"]["specializationValues"] == [
        {"id": constant_id, "name": name, "value": False}
        for constant_id, name in SPECIALIZATION_NAMES.items()
    ]
    assert deferred_report["interface"]["status"] == "verified"
    assert deferred_report["cache"]["status"] == "published"
