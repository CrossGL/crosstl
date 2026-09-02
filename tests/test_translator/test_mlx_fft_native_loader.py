from __future__ import annotations

import cmath
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
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
    load_project_config,
    translate_project,
    validate_project_report,
)

MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
CURRENT_MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
MLX_FFT_SOURCE = "mlx/backend/metal/kernels/fft.metal"
MLX_FFT_SHA256 = "3a1fbb38ed64f50a49a20d0c5adb1748d9d06ea20e5931e99aa26be543cb7825"
MLX_FFT_SOURCE_SIZE_BYTES = 3278
CURRENT_MLX_FFT_SHA256 = (
    "c478eb84283bbdf585c0cb34b2bfde5b0fc32d1740c6ad76e8559698a57b8d2e"
)
CURRENT_MLX_FFT_SOURCE_SIZE_BYTES = 3436
MLX_FFT_ENTRY = "fft_mem_256_float2_float2"
MLX_FFT_GENERATED_SHA256 = (
    "d5fa5d6e088743cc00a4a3f4b604c0704d33a8697a570f11a61cf99cb8d8d6fb"
)
MLX_FFT_GENERATED_SIZE_BYTES = 116260
CURRENT_MLX_FFT_GENERATED_SHA256 = (
    "3bc42b2dd3bf128bcbe1fd202763f3434d64df6e60b53da4b6e754ceff0f6e7a"
)
CURRENT_MLX_FFT_GENERATED_SIZE_BYTES = 146763
CURRENT_MLX_FFT_OPENGL_GENERATED_SHA256 = (
    "a1ab0c346d9143e6749e391fb971aeaed71bd84e15fedaf7a7e92808a56449bb"
)
CURRENT_MLX_FFT_OPENGL_GENERATED_SIZE_BYTES = 82045
MLX_FFT_SIZE = 256
REQUIRE_PROOF_ENV = "CROSTL_REQUIRE_MLX_FFT_DIRECTX_NATIVE_LOADER"
REQUIRE_OPENGL_PROOF_ENV = "CROSTL_REQUIRE_MLX_FFT_OPENGL_NATIVE_LOADER"
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
CURRENT_FFT_SPECIALIZATION_NAMES = {
    0: "inv_",
    1: "is_power_of_2_",
    2: "elems_per_thread_",
    4: "radix_13_steps_",
    5: "radix_11_steps_",
    6: "radix_8_steps_",
    7: "radix_7_steps_",
    8: "radix_6_steps_",
    9: "radix_5_steps_",
    10: "radix_4_steps_",
    11: "radix_3_steps_",
    12: "radix_2_steps_",
    13: "rader_13_steps_",
    14: "rader_11_steps_",
    15: "rader_8_steps_",
    16: "rader_7_steps_",
    17: "rader_6_steps_",
    18: "rader_5_steps_",
    19: "rader_4_steps_",
    20: "rader_3_steps_",
    21: "rader_2_steps_",
}
CURRENT_FFT_DEFERRED_SPECIALIZATION_VALUES = [
    {
        "id": constant_id,
        "name": CURRENT_FFT_SPECIALIZATION_NAMES[constant_id],
        "value": CURRENT_FFT_SPECIALIZATION_CONSTANTS[constant_id],
    }
    for constant_id in CURRENT_FFT_SPECIALIZATION_NAMES
]


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _project_config(
    output_dir: str,
    *,
    target: str = "directx",
    specialization_constants: dict[int, object] = FFT_SPECIALIZATION_CONSTANTS,
    include_current_opengl_index_assertion: bool = False,
) -> str:
    constants = "\n".join(
        f'"{constant_id}" = {_toml_value(value)}'
        for constant_id, value in specialization_constants.items()
    )
    additional_index_assertion = ""
    if include_current_opengl_index_assertion:
        additional_index_assertion = f"""\n[[project.index_range_assertions]]
source = "{MLX_FFT_SOURCE}"
expression = "batch_idx + index + r"
minimum = 0
maximum = 4294967295
"""
    return f"""
[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{MLX_FFT_SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
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
{additional_index_assertion}
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


def test_current_mlx_fft_executes_through_directx_native_loader():
    _execute_mlx_fft_through_directx_native_loader(
        _current_mlx_root(),
        temporary_prefix=".crosstl-fft-current-directx-native-loader-",
        source_sha256=CURRENT_MLX_FFT_SHA256,
        source_size_bytes=CURRENT_MLX_FFT_SOURCE_SIZE_BYTES,
        generated_sha256=CURRENT_MLX_FFT_GENERATED_SHA256,
        generated_size_bytes=CURRENT_MLX_FFT_GENERATED_SIZE_BYTES,
        specialization_constants=CURRENT_FFT_SPECIALIZATION_CONSTANTS,
        template_specialization_count=37,
        template_accounting={
            "reachableSpecializationCount": 42,
            "dependencyDiscoveryWorkCount": 0,
            "prunedCandidateCount": 2120,
        },
    )


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


def _build_runtime_package(
    mlx_root: Path,
    work_dir: Path,
    *,
    source_sha256: str,
    source_size_bytes: int,
    generated_sha256: str,
    generated_size_bytes: int,
    specialization_constants: dict[int, object],
    template_specialization_count: int,
    template_accounting: dict[str, int] | None = None,
) -> tuple[dict, Path]:
    output_dir = work_dir / "out"
    config_path = work_dir / "crosstl.toml"
    config_path.write_text(
        _project_config(
            output_dir.relative_to(mlx_root).as_posix(),
            specialization_constants=specialization_constants,
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
    report_payload = report.to_json()

    assert report_payload["summary"]["unitCount"] == 1
    assert report_payload["summary"]["artifactCount"] == 1
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
        "value": source_sha256,
    }
    assert artifact["sourceSizeBytes"] == source_size_bytes
    assert artifact["status"] == "translated"
    assert artifact["entryPoint"] == {
        "source": MLX_FFT_ENTRY,
        "target": "CSMain",
        "stage": "compute",
    }
    assert artifact["execution"]["entryPoints"][0]["workgroupSize"] == [1, 1, 64]
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": generated_sha256,
    }
    assert artifact["generatedSizeBytes"] == generated_size_bytes
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
    assert len(materialization["specializations"]) == template_specialization_count
    assert materialization["specializationCount"] == template_specialization_count
    assert materialization["unsupported"] == []
    if template_accounting is not None:
        assert materialization["accounting"] == template_accounting

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert "groupshared float2 fft_mem_256_float2_float2_shared_in[256];" in generated
    assert re.search(r"\bcrosstl_ptr_buf\s*[,)]", generated) is None
    assert "float2*" not in generated
    assert "decltype(" not in generated
    assert "vec_u3cfloat_u2c2_u3e" not in generated
    assert "return float2(0);" not in generated
    assert generated.count("<< int(power)") == 20
    assert "<< uint16_t(power)" not in generated
    assert (
        re.search(r"\b(?:RW)?StructuredBuffer<float2>\s+twiddles\b", generated) is None
    )

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


def _execute_mlx_fft_through_directx_native_loader(
    mlx_root: Path,
    *,
    temporary_prefix: str,
    source_sha256: str,
    source_size_bytes: int,
    generated_sha256: str,
    generated_size_bytes: int,
    specialization_constants: dict[int, object],
    template_specialization_count: int,
    template_accounting: dict[str, int] | None = None,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=temporary_prefix,
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
            source_sha256=source_sha256,
            source_size_bytes=source_size_bytes,
            generated_sha256=generated_sha256,
            generated_size_bytes=generated_size_bytes,
            specialization_constants=specialization_constants,
            template_specialization_count=template_specialization_count,
            template_accounting=template_accounting,
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


def test_pinned_mlx_fft_executes_through_directx_native_loader():
    _execute_mlx_fft_through_directx_native_loader(
        _pinned_mlx_root(),
        temporary_prefix=".crosstl-fft-directx-native-loader-",
        source_sha256=MLX_FFT_SHA256,
        source_size_bytes=MLX_FFT_SOURCE_SIZE_BYTES,
        generated_sha256=MLX_FFT_GENERATED_SHA256,
        generated_size_bytes=MLX_FFT_GENERATED_SIZE_BYTES,
        specialization_constants=FFT_SPECIALIZATION_CONSTANTS,
        template_specialization_count=24,
    )


def _expected_current_opengl_layouts() -> dict[str, dict]:
    return {
        "in_Buffer": {
            "physicalType": "float2",
            "elementType": "float32",
            "elementSizeBytes": 8,
            "elementStrideBytes": 8,
            "alignmentBytes": 8,
            "memberOffsetBytes": 0,
            "storageLayout": "std430",
            "runtimeSized": True,
            "vectorWidth": 2,
            "memberName": "in_",
        },
        "out_Buffer": {
            "physicalType": "float2",
            "elementType": "float32",
            "elementSizeBytes": 8,
            "elementStrideBytes": 8,
            "alignmentBytes": 8,
            "memberOffsetBytes": 0,
            "storageLayout": "std430",
            "runtimeSized": True,
            "vectorWidth": 2,
            "memberName": "out_",
        },
        "fft_mem_256_float2_float2_n_Args": {
            "physicalType": "int",
            "elementType": "int32",
            "elementSizeBytes": 4,
            "elementStrideBytes": 4,
            "alignmentBytes": 16,
            "memberOffsetBytes": 0,
            "storageLayout": "std140",
            "runtimeSized": False,
            "memberName": "n",
            "blockSizeBytes": 16,
        },
        "fft_mem_256_float2_float2_batch_size_Args": {
            "physicalType": "int",
            "elementType": "int32",
            "elementSizeBytes": 4,
            "elementStrideBytes": 4,
            "alignmentBytes": 16,
            "memberOffsetBytes": 0,
            "storageLayout": "std140",
            "runtimeSized": False,
            "memberName": "batch_size",
            "blockSizeBytes": 16,
        },
    }


def _current_opengl_tool(name: str) -> str:
    tool = shutil.which(name)
    if tool is not None:
        return tool
    message = f"{name} is required for current MLX FFT OpenGL validation"
    if os.environ.get(REQUIRE_OPENGL_PROOF_ENV) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _assert_current_opengl_spirv(generated_path: Path, work_dir: Path) -> None:
    glslang = _current_opengl_tool("glslangValidator")
    spirv_val = _current_opengl_tool("spirv-val")
    spirv_dis = _current_opengl_tool("spirv-dis")
    spirv_path = work_dir / "fft_mem_256_float2_float2.spv"
    compiled = subprocess.run(
        [glslang, "-V", "-S", "comp", "-o", str(spirv_path), str(generated_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    validated = subprocess.run(
        [spirv_val, str(spirv_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validated.returncode == 0, validated.stdout + validated.stderr
    spirv = spirv_path.read_bytes()
    assert len(spirv) > 0
    assert len(spirv) % 4 == 0
    disassembled = subprocess.run(
        [spirv_dis, str(spirv_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert disassembled.count("OpControlBarrier") == 19
    assert "OpGroupNonUniform" not in disassembled


def _build_current_opengl_runtime_package(
    mlx_root: Path, work_dir: Path
) -> tuple[Path, dict]:
    output_dir = work_dir / "out"
    config_path = work_dir / "crosstl.toml"
    config_path.write_text(
        _project_config(
            output_dir.relative_to(mlx_root).as_posix(),
            target="opengl",
            specialization_constants=CURRENT_FFT_SPECIALIZATION_CONSTANTS,
            include_current_opengl_index_assertion=True,
        )
        + "\n",
        encoding="utf-8",
    )
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
    assert payload["summary"]["artifactCount"] == 1
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    assert payload["summary"]["diagnosticCounts"] == {
        "error": 0,
        "note": 0,
        "warning": 0,
    }, json.dumps(payload["diagnostics"], indent=2)
    assert payload["project"]["indexRangeAssertionCount"] == 5
    assert payload["project"]["workgroupAccessAssertionCount"] == 1
    assert payload["summary"]["sourceRemapMappingCount"] == 84

    artifact = payload["artifacts"][0]
    assert artifact["source"] == MLX_FFT_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": CURRENT_MLX_FFT_SHA256,
    }
    assert artifact["sourceSizeBytes"] == CURRENT_MLX_FFT_SOURCE_SIZE_BYTES
    assert artifact["status"] == "translated"
    assert artifact["entryPoint"] == {
        "source": MLX_FFT_ENTRY,
        "target": "main",
        "stage": "compute",
    }
    assert artifact["execution"]["entryPoints"][0]["workgroupSize"] == [1, 1, 64]
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": CURRENT_MLX_FFT_OPENGL_GENERATED_SHA256,
    }
    assert artifact["generatedSizeBytes"] == CURRENT_MLX_FFT_OPENGL_GENERATED_SIZE_BYTES
    assert artifact["specializationMaterialization"] == {
        "status": "deferred",
        "mode": "deferred",
        "source": "shared-crossgl-specialization",
        "constantCount": 21,
        "requiredCount": 21,
        "concreteCount": 21,
        "overriddenCount": 21,
        "targetSupportsDeferredSpecialization": True,
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

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert hashlib.sha256(generated_path.read_bytes()).hexdigest() == (
        CURRENT_MLX_FFT_OPENGL_GENERATED_SHA256
    )
    assert "layout(std430, binding = 0) readonly buffer in_Buffer" in generated
    assert "layout(std430, binding = 1) buffer out_Buffer" in generated
    assert "vec_u3cfloat_u2c2_u3e" not in generated
    assert "nullptr" not in generated
    assert "crosstl_ptr_buf_offset" in generated
    _assert_current_opengl_spirv(generated_path, work_dir)

    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True
    runtime_artifacts = build_runtime_artifact_manifest(report_path)
    assert runtime_artifacts["success"] is True, json.dumps(runtime_artifacts, indent=2)
    assert runtime_artifacts["summary"]["artifactCount"] == 1
    assert runtime_artifacts["summary"]["resourceBindingCount"] == 4
    assert runtime_artifacts["summary"]["specializationConstantCount"] == 21
    reflected = runtime_artifacts["artifacts"][0]
    assert reflected["hostInterface"]["status"] == "ready"
    assert {
        resource["name"]: resource["scalarLayout"]
        for resource in reflected["hostInterface"]["resources"]
    } == _expected_current_opengl_layouts()

    runtime_artifacts_path = work_dir / "runtime-artifacts.json"
    _write_json(runtime_artifacts_path, runtime_artifacts)
    package_dir = work_dir / "runtime-package"
    package = build_runtime_package(runtime_artifacts_path, package_dir)
    assert package["success"] is True, json.dumps(package, indent=2)
    loader = build_runtime_loader_manifest(package_dir / "runtime-package.json")
    assert loader["success"] is True, json.dumps(loader, indent=2)
    assert loader["summary"]["readyLoadUnitCount"] == 1
    assert loader["summary"]["blockedLoadUnitCount"] == 0
    loader_path = package_dir / "runtime-loader-manifest.json"
    _write_json(loader_path, loader)
    descriptor = build_native_loader_abi_descriptor(
        loader,
        load_unit_id=loader["loadUnits"][0]["id"],
    )
    assert descriptor["target"] == "opengl"
    assert descriptor["entryPoint"]["name"] == "main"
    assert descriptor["entryPoint"]["executionConfig"] == {
        "local_size": [1, 1, 64],
        "local_size_x": 1,
        "local_size_y": 1,
        "local_size_z": 64,
    }
    assert {
        binding["name"]: binding["scalarLayout"] for binding in descriptor["bindings"]
    } == _expected_current_opengl_layouts()
    assert [
        (item["id"], item["name"], item["value"], item["deferred"])
        for item in descriptor["specializationConstants"]
    ] == [
        (item["id"], item["name"], item["value"], True)
        for item in CURRENT_FFT_DEFERRED_SPECIALIZATION_VALUES
    ]

    abi_root = work_dir / "native-loader-abi-package"
    abi_package = build_native_loader_abi_package(loader_path, abi_root)
    assert abi_package["success"] is True, json.dumps(abi_package, indent=2)
    assert abi_package["summary"]["runtimeVariantCount"] == 1
    registry = json.loads(
        (abi_root / NATIVE_RUNTIME_VARIANT_REGISTRY_PATH).read_text(encoding="utf-8")
    )
    assert registry["status"] == "ready"
    assert registry["lookup"]["blockedKeys"] == []
    request = build_runtime_variant_deferred_compilation_request(
        registry,
        registry["lookup"]["readyKeys"][0],
        abi_root,
    )
    assert request["variant"]["specializationValues"] == (
        CURRENT_FFT_DEFERRED_SPECIALIZATION_VALUES
    )
    assert request["variant"]["execution"] == {
        "workgroupSize": [1, 1, 64],
        "subgroupWidth": None,
    }
    return abi_root, request


def test_current_mlx_fft_executes_through_opengl_native_loader():
    mlx_root = _current_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-fft-current-opengl-native-loader-",
        dir=mlx_root,
    ) as temporary_directory:
        work_dir = Path(temporary_directory)
        abi_root, compilation_request = _build_current_opengl_runtime_package(
            mlx_root, work_dir
        )
        input_values, expected_values = _complex_impulse()
        try:
            result = execute_native_deferred_compilation_request(
                compilation_request,
                abi_root,
                work_dir / "deferred-cache",
                {
                    "in_Buffer": {
                        "dtype": "float32",
                        "shape": [MLX_FFT_SIZE, 2],
                        "values": input_values,
                    },
                    "fft_mem_256_float2_float2_n_Args": {
                        "dtype": "int32",
                        "shape": [1],
                        "values": [MLX_FFT_SIZE],
                    },
                    "fft_mem_256_float2_float2_batch_size_Args": {
                        "dtype": "int32",
                        "shape": [1],
                        "values": [1],
                    },
                },
                {
                    "out_Buffer": {
                        "dtype": "float32",
                        "shape": [MLX_FFT_SIZE, 2],
                        "values": expected_values,
                        "tolerance": {"absolute": 2e-4, "relative": 2e-4},
                    }
                },
                (1, 1, 1),
                runtime_adapter=OpenGLRuntimeParityAdapter(
                    runtime=OpenGLComputeRuntime(context_backends=("egl",))
                ),
            )
        except NativeDeferredCompilationRuntimeError as exc:
            if exc.code.endswith(".runtime-unavailable"):
                if os.environ.get(REQUIRE_OPENGL_PROOF_ENV) == "1":
                    pytest.fail(str(exc))
                pytest.skip(str(exc))
            raise

    assert result.status == "ok"
    assert result.outputs["out_Buffer"]["dtype"] == "float32"
    assert result.outputs["out_Buffer"]["shape"] == [MLX_FFT_SIZE, 2]
    assert result.outputs["out_Buffer"]["values"] == pytest.approx(
        expected_values,
        abs=2e-4,
        rel=2e-4,
    )
    deferred_report = result.details["nativeDeferredCompilation"]
    assert deferred_report["success"] is True
    assert deferred_report["target"]["backend"] == "opengl"
    assert deferred_report["variant"]["specializationValues"] == (
        CURRENT_FFT_DEFERRED_SPECIALIZATION_VALUES
    )
    assert deferred_report["interface"]["status"] == "verified"
    assert deferred_report["cache"]["status"] == "published"
