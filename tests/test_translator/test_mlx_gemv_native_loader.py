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
    load_dispatch_contract,
    load_project_config,
    translate_project,
    validate_project_report,
)

ROOT = Path(__file__).resolve().parents[2]
MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
MLX_GEMV_SOURCE = "mlx/backend/metal/kernels/gemv.metal"
MLX_GEMV_SHA256 = "0bd8bde0c867a17c345a3651f9f0a6c2909e0c74e76ea2a08f373fe4dcafaeda"
MLX_GEMV_ENTRY = "gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0"
MLX_GEMV_DISPATCH_CONTRACT = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "gemv.native-loader.dispatch.json"
)
MLX_GEMV_DISPATCH_IDENTITY = (
    "6b3bb18d130159f13874f06668b536fe4b9270ffbb2a1f44b6d9aac257aba7e4"
)
MLX_GEMV_ARTIFACT_ID = (
    "sha256:34eab189b10cc699f06f4cbed04faae41a2658a2a3665a6866ed987f5946949a"
)
MLX_GEMV_VARIANT_ID = (
    "sha256:acaba2ec4813a364b06d95a5136bda80351797591a4d9f0b3d195f85da287fe3"
)
MLX_GEMV_GENERATED_ARTIFACTS = {
    "directx": {
        "sha256": "9972997d87bb4c8c5fac0c0f7182bb19648654ca2c30eacd4b304bfaf18f64d2",
        "sizeBytes": 8188,
    },
    "opengl": {
        "sha256": "f5ef8900ee65d63a6df2818ef111f56b4f269c6366c82d82a9d97c967042f562",
        "sizeBytes": 7705,
    },
}
REQUIRE_DIRECTX_RUNTIME_ENV = "CROSTL_REQUIRE_MLX_GEMV_DIRECTX_NATIVE_LOADER"
REQUIRE_OPENGL_RUNTIME_ENV = "CROSTL_REQUIRE_MLX_GEMV_OPENGL_NATIVE_LOADER"
INPUT_SIZE = 32
OUTPUT_SIZE = 32
WORKGROUP_SIZE = (32, 2, 1)
WORKGROUP_COUNT = (1, 1, 1)
ABSOLUTE_TOLERANCE = 1e-5
RELATIVE_TOLERANCE = 1e-5
INDEX_ASSERTION = {
    "source": MLX_GEMV_SOURCE,
    "expression": "uint64(bm + tm) * marix_ld + out_col + tn",
    "minimum": 0,
    "maximum": (1 << 32) - 1,
}


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
            for name in (REQUIRE_DIRECTX_RUNTIME_ENV, REQUIRE_OPENGL_RUNTIME_ENV)
        ):
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")

    mlx_root = Path(value).resolve()
    source_path = mlx_root / MLX_GEMV_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX GEMV source is missing: {source_path}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_GEMV_SHA256
    return mlx_root


def _dispatch_variant():
    manifest = load_dispatch_contract(MLX_GEMV_DISPATCH_CONTRACT)
    assert manifest.provenance["commit"] == MLX_COMMIT
    assert manifest.provenance["sourceReferences"] == {
        "hostDispatch": "mlx/backend/metal/matmul.cpp",
        "implementation": "mlx/backend/metal/kernels/gemv.h",
        "kernel": MLX_GEMV_SOURCE,
    }
    assert manifest.content_identity.to_json() == {
        "algorithm": "sha256",
        "value": MLX_GEMV_DISPATCH_IDENTITY,
    }
    variants = manifest.evaluate()
    assert len(variants) == 1
    variant = variants[0]
    assert variant.workload_id == "vector-matrix-float32-m1-n32-k32-contiguous"
    assert variant.inputs == {
        "K": INPUT_SIZE,
        "M": 1,
        "N": OUTPUT_SIZE,
        "batchSize": 1,
        "contiguousBatch": True,
        "doAxpby": False,
        "dtype": "float32",
        "transposeA": False,
        "transposeB": False,
    }
    assert variant.variant_id == MLX_GEMV_VARIANT_ID
    assert variant.artifact_id == MLX_GEMV_ARTIFACT_ID
    assert variant.entry_point == MLX_GEMV_ENTRY
    assert variant.workgroup_size == WORKGROUP_SIZE
    assert variant.subgroup_width == 32
    assert variant.dispatch_field == "workgroupCount"
    assert variant.dispatch_size == WORKGROUP_COUNT
    assert variant.specialization_constants == {}
    return variant


def test_current_mlx_gemv_dispatch_contract_is_exact():
    _dispatch_variant()


def _project_config(target: str, *, output_dir: str) -> str:
    sections = [textwrap.dedent(f"""
            [project]
            source_roots = ["mlx/backend/metal/kernels"]
            include = ["{MLX_GEMV_SOURCE}"]
            include_dirs = ["."]
            targets = ["{target}"]
            output_dir = "{output_dir}"

            [project.sources]
            "**/*.metal" = "metal"

            [project.entry_points]
            "{MLX_GEMV_SOURCE}" = "{MLX_GEMV_ENTRY}"

            [project.entry_workgroup_size_rules."{MLX_GEMV_SOURCE}"]
            "{MLX_GEMV_ENTRY}" = [32, 2, 1]
            """).strip()]
    if target == "directx":
        sections.append(textwrap.dedent(f"""
                [project.subgroup_width_rules]
                "{MLX_GEMV_SOURCE}" = 32
                """).strip())
    elif target == "opengl":
        sections.append(textwrap.dedent(f"""
                [[project.index_range_assertions]]
                source = "{INDEX_ASSERTION['source']}"
                expression = "{INDEX_ASSERTION['expression']}"
                minimum = {INDEX_ASSERTION['minimum']}
                maximum = {INDEX_ASSERTION['maximum']}
                """).strip())
    else:
        raise ValueError(f"Unsupported GEMV test target: {target}")
    sections.append(textwrap.dedent("""
            [project.source_options.metal]
            max_template_specializations = 64
            max_template_materialization_work = 4096
            """).strip())
    if target == "directx":
        sections.append(textwrap.dedent("""
                [project.source_options.metal.target_options.directx]
                relative_wave_shuffle_out_of_range = "self"
                software_subgroup_width = 32
                """).strip())
    elif target == "opengl":
        sections.append(textwrap.dedent("""
                [project.source_options.metal.target_options.opengl]
                software_subgroup_width = 32
                """).strip())
    return "\n\n".join(sections)


def _assert_opengl_spirv(generated_path: Path, work_dir: Path) -> None:
    tools = {
        name: shutil.which(name)
        for name in ("glslangValidator", "spirv-val", "spirv-dis")
    }
    if not all(tools.values()):
        missing = [name for name, path in tools.items() if path is None]
        _skip_or_fail(
            "The GEMV OpenGL proof requires: " + ", ".join(missing),
            require_env=REQUIRE_OPENGL_RUNTIME_ENV,
        )

    spirv_path = work_dir / "gemv.spv"
    assembly_path = work_dir / "gemv.spvasm"
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
    assert assembly.count("OpControlBarrier") == 3
    assert "OpGroupNonUniform" not in assembly
    assert "OpExecutionMode %main LocalSize 32 2 1" in assembly


def _binding_names(target: str) -> dict[str, str]:
    if target == "directx":
        return {
            "matrix": "mat",
            "vector": "in_vec",
            "bias": "bias",
            "output": "out_vec",
            "batch_shape": "batch_shape",
            "vector_stride": "vector_batch_stride",
            "matrix_stride": "matrix_batch_stride",
            "bias_stride_array": "bias_batch_stride",
        }
    return {
        "matrix": "matBuffer",
        "vector": "in_vecBuffer",
        "bias": "biasBuffer",
        "output": "out_vecBuffer",
        "batch_shape": "batch_shapeBuffer",
        "vector_stride": "vector_batch_strideBuffer",
        "matrix_stride": "matrix_batch_strideBuffer",
        "bias_stride_array": "bias_batch_strideBuffer",
    }


def _scalar_binding(target: str, field: str) -> str:
    suffix = "Constants" if target == "directx" else "Args"
    return f"{MLX_GEMV_ENTRY}_{field}_{suffix}"


def _expected_binding_contract(target: str) -> dict[str, tuple[int, str, str]]:
    names = _binding_names(target)
    contract = {
        names["matrix"]: (0, "read", "float32"),
        names["vector"]: (1, "read", "float32"),
        names["bias"]: (2, "read", "float32"),
        names["output"]: (3, "read_write", "float32"),
        names["batch_shape"]: (10, "read", "int32"),
        names["vector_stride"]: (11, "read", "int64"),
        names["matrix_stride"]: (12, "read", "int64"),
        names["bias_stride_array"]: (13, "read", "int64"),
    }
    for field, binding, dtype in (
        ("in_vec_size", 4, "int32"),
        ("out_vec_size", 5, "int32"),
        ("marix_ld", 6, "int32"),
        ("alpha", 7, "float32"),
        ("beta", 8, "float32"),
        ("batch_ndim", 9, "int32"),
        ("bias_stride", 14, "int32"),
    ):
        contract[_scalar_binding(target, field)] = (binding, "read", dtype)
    return contract


def _build_runtime_package(
    mlx_root: Path,
    work_dir: Path,
    target: str,
) -> tuple[dict, Path]:
    _dispatch_variant()
    output_dir = work_dir / "o"
    config_path = work_dir / "c.toml"
    config_path.write_text(
        _project_config(
            target,
            output_dir=output_dir.relative_to(mlx_root).as_posix(),
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
    if target == "directx":
        assert payload["project"]["sourceOptions"]["metal"]["target_options"] == {
            "directx": {
                "relative_wave_shuffle_out_of_range": "self",
                "software_subgroup_width": 32,
            }
        }
    elif target == "opengl":
        assert payload["project"]["indexRangeAssertions"] == [INDEX_ASSERTION]
        assert payload["project"]["indexRangeAssertionCount"] == 1

    artifact = payload["artifacts"][0]
    expected_identity = MLX_GEMV_GENERATED_ARTIFACTS[target]
    assert artifact["source"] == MLX_GEMV_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_GEMV_SHA256,
    }
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": expected_identity["sha256"],
    }
    assert artifact["generatedSizeBytes"] == expected_identity["sizeBytes"]
    assert artifact["entryPoint"] == {
        "source": MLX_GEMV_ENTRY,
        "target": "CSMain" if target == "directx" else "main",
        "stage": "compute",
    }
    assert artifact["provenance"]["pipeline"] == "entry-scoped-translate"
    execution = artifact["execution"]["entryPoints"]
    assert len(execution) == 1
    assert execution[0]["workgroupSize"] == list(WORKGROUP_SIZE)
    assert execution[0]["parameters"] == {
        "BM": "1",
        "BN": "2",
        "SM": "8",
        "SN": "4",
        "T": "float",
        "TM": "4",
        "TN": "4",
        "kDoAxpby": "0",
        "kDoNCBatch": "0",
    }
    materialization = artifact["templateMaterialization"]
    assert materialization["status"] == "materialized"
    assert materialization["specializationCount"] == 2
    assert materialization["unsupported"] == []
    assert {
        record["materializedName"] for record in materialization["specializations"]
    } == {MLX_GEMV_ENTRY, "elem_to_loc_uint"}

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    if target == "directx":
        assert "[numthreads(32, 2, 1)]" in generated
        assert "[WaveSize(32)]" in generated
        assert (
            "groupshared float __crossgl_software_subgroup_scratch_float[64];"
            in generated
        )
        assert (
            "float __crossgl_software_subgroup_shuffle_down_float("
            "float value, uint delta, uint invocation)" in generated
        )
        assert "uint lane = invocation % 32u;" in generated
        assert "uint subgroupBase = invocation - lane;" in generated
        assert "bool sourceValid = delta < (32u - lane);" in generated
        assert "uint sourceLane = sourceValid ? lane + delta : lane;" in generated
        assert "subgroupBase + lane + delta" not in generated
        assert generated.count("GroupMemoryBarrierWithGroupSync();") == 3
        assert (
            "__crossgl_software_subgroup_shuffle_down_float("
            "result[tn], uint((4 * int(sm)))" in generated
        )
        assert "uint(lid.x) + 32u * (uint(lid.y) + 2u * uint(lid.z))" in generated
        assert "uint simd_gid = (groupIndex / 32u);" in generated
        assert "uint simd_lid = (groupIndex % 32u);" in generated
        assert "WaveReadLaneAt" not in generated
        assert "WaveGetLaneIndex" not in generated
        assert "__crossgl_physical_subgroup" not in generated
        assert "InterlockedAdd" not in generated
    else:
        assert "layout(local_size_x = 32, local_size_y = 2" in generated
        assert "#define CROSSTL_SOFTWARE_SUBGROUP_WIDTH 32u" in generated
        assert "shared float crossglSoftwareSubgroupScratchFloat[64];" in generated
        assert "crossglSoftwareSubgroupShuffleDownFloat(result[tn]" in generated
        assert "(int(sm) >= 1)" in generated
        assert "GL_KHR_shader_subgroup" not in generated
        assert "gl_Subgroup" not in generated
        assert "subgroupShuffle" not in generated
        _assert_opengl_spirv(generated_path, work_dir)

    report_path = work_dir / "r.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True
    runtime_artifacts = build_runtime_artifact_manifest(report_path)
    assert runtime_artifacts["success"] is True, json.dumps(
        runtime_artifacts,
        indent=2,
    )
    assert runtime_artifacts["summary"]["artifactCount"] == 1
    assert runtime_artifacts["summary"]["resourceBindingCount"] == 15
    reflected = runtime_artifacts["artifacts"][0]["hostInterface"]
    assert reflected["status"] == "ready"
    assert reflected["entryPoints"] == [
        {
            "name": "CSMain" if target == "directx" else "main",
            "stage": "compute",
            "executionConfig": (
                {"numthreads": [32, 2, 1], "subgroupWidth": 32}
                if target == "directx"
                else {
                    "local_size_x": 32,
                    "local_size_y": 2,
                    "local_size_z": 1,
                    "local_size": [32, 2, 1],
                }
            ),
        }
    ]
    reflected_contract = {
        resource["name"]: (
            resource["binding"],
            resource["access"],
            resource["scalarLayout"]["elementType"],
        )
        for resource in reflected["resources"]
    }
    assert reflected_contract == _expected_binding_contract(target)

    runtime_artifacts_path = work_dir / "a.json"
    _write_json(runtime_artifacts_path, runtime_artifacts)
    package_dir = work_dir / "p"
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
    descriptor_contract = {
        binding["name"]: (
            binding["coordinates"]["binding"],
            binding["access"],
            binding["scalarLayout"]["elementType"],
        )
        for binding in descriptor["bindings"]
    }
    assert descriptor_contract == _expected_binding_contract(target)
    return descriptor, package_dir


def _gemv_workload() -> tuple[list[float], list[float], list[float]]:
    vector = [(row + 1) / 32.0 for row in range(INPUT_SIZE)]
    matrix = [
        (row + 1) / 64.0 + (column + 1) / 64.0
        for row in range(INPUT_SIZE)
        for column in range(OUTPUT_SIZE)
    ]
    expected = [
        sum(
            vector[row] * matrix[row * OUTPUT_SIZE + column]
            for row in range(INPUT_SIZE)
        )
        for column in range(OUTPUT_SIZE)
    ]
    assert expected == [
        5.5859375 + 0.2578125 * (column + 1) for column in range(OUTPUT_SIZE)
    ]
    return matrix, vector, expected


def _dispatch_request(target: str, descriptor: dict, package_dir: Path):
    names = _binding_names(target)
    matrix, vector, expected = _gemv_workload()
    inputs = {
        names["matrix"]: {
            "dtype": "float32",
            "shape": [INPUT_SIZE * OUTPUT_SIZE],
            "values": matrix,
        },
        names["vector"]: {
            "dtype": "float32",
            "shape": [INPUT_SIZE],
            "values": vector,
        },
        names["bias"]: {
            "dtype": "float32",
            "shape": [1],
            "values": [0.0],
        },
        names["batch_shape"]: {
            "dtype": "int32",
            "shape": [1],
            "values": [1],
        },
        names["vector_stride"]: {
            "dtype": "int64",
            "shape": [1],
            "values": [0],
        },
        names["matrix_stride"]: {
            "dtype": "int64",
            "shape": [1],
            "values": [0],
        },
        names["bias_stride_array"]: {
            "dtype": "int64",
            "shape": [1],
            "values": [0],
        },
    }
    for field, dtype, value in (
        ("in_vec_size", "int32", INPUT_SIZE),
        ("out_vec_size", "int32", OUTPUT_SIZE),
        ("marix_ld", "int32", OUTPUT_SIZE),
        ("alpha", "float32", 1.0),
        ("beta", "float32", 0.0),
        ("batch_ndim", "int32", 1),
        ("bias_stride", "int32", 1),
    ):
        inputs[_scalar_binding(target, field)] = {
            "dtype": dtype,
            "shape": [1],
            "values": [value],
        }

    request = build_native_loader_dispatch_request(
        descriptor,
        package_dir,
        inputs,
        {
            names["output"]: {
                "dtype": "float32",
                "shape": [OUTPUT_SIZE],
                "values": expected,
                "tolerance": {
                    "absolute": ABSOLUTE_TOLERANCE,
                    "relative": RELATIVE_TOLERANCE,
                },
            }
        },
        WORKGROUP_COUNT,
        expected_target=target,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    assert request.execution_plan.dispatch.workgroup_size == WORKGROUP_SIZE
    assert request.execution_plan.dispatch.workgroup_count == WORKGROUP_COUNT
    assert request.execution_plan.dispatch.global_size == WORKGROUP_SIZE
    return request, names["output"], expected


def _execute_current_mlx_gemv(target: str) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(prefix=".gv-", dir=mlx_root) as temporary:
        work_dir = Path(temporary)
        descriptor, package_dir = _build_runtime_package(
            mlx_root,
            work_dir,
            target,
        )
        request, output_name, expected = _dispatch_request(
            target,
            descriptor,
            package_dir,
        )
        runtime_adapter = (
            DirectXRuntimeParityAdapter(runtime=DirectXComputeRuntime())
            if target == "directx"
            else OpenGLRuntimeParityAdapter(runtime=OpenGLComputeRuntime())
        )
        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id=f"mlx-gemv-{target}-native-loader",
                target=target,
                executor=target,
                adapter_kind=f"{target}-native-runtime",
            ),
            runtime_adapter=runtime_adapter,
        )
        availability = executor.is_available(request)
        if not availability.available:
            _skip_or_fail(
                availability.reason or f"The native {target} runtime is unavailable",
                require_env=(
                    REQUIRE_DIRECTX_RUNTIME_ENV
                    if target == "directx"
                    else REQUIRE_OPENGL_RUNTIME_ENV
                ),
            )
        result = executor.run(request)

    assert result.status == "ok"
    assert result.outputs[output_name]["dtype"] == "float32"
    assert result.outputs[output_name]["shape"] == [OUTPUT_SIZE]
    assert result.outputs[output_name]["values"] == pytest.approx(
        expected,
        abs=ABSOLUTE_TOLERANCE,
        rel=RELATIVE_TOLERANCE,
    )


def test_current_mlx_gemv_executes_through_directx_native_loader():
    _execute_current_mlx_gemv("directx")


def test_current_mlx_gemv_executes_with_opengl_software_subgroups():
    _execute_current_mlx_gemv("opengl")
