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
from crosstl.project.directx_toolchain import dxc_compiler_arguments_for_source

ROOT = Path(__file__).resolve().parents[2]
MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
MLX_LAYER_NORM_SOURCE = "mlx/backend/metal/kernels/layer_norm.metal"
MLX_LAYER_NORM_SHA256 = (
    "2d243f5abea7353929f9bc838ceb5a98e52a452dfc29609ad4d5974447ea689f"
)
MLX_LAYER_NORM_ENTRY = "layer_normfloat32"
MLX_LAYER_NORM_ARTIFACT_ID = (
    "sha256:4b1c27c05949e3021e5b28aeb4552e668aad8dd00141a1880fe98e7ebc7b129e"
)
MLX_LAYER_NORM_DISPATCH_IDENTITY = (
    "320929bc503640b12748a28f72aa571f9a79123e498d5f8cda4a9827c2d01add"
)
MLX_LAYER_NORM_DISPATCH_CONTRACT = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "layer_norm.native-loader.dispatch.json"
)
MLX_LAYER_NORM_GENERATED_ARTIFACTS = {
    "directx": {
        "sha256": "7e790d4e665c72025e46c7c038aba2bec57ba6f65e209178eae5160c0c7ea8e9",
        "sizeBytes": 5216,
    },
    "opengl": {
        "sha256": "f86f83b6835b7d4b07ece9f153df883300f7a131bbcec5d084bf29084c1bf51a",
        "sizeBytes": 5914,
    },
}
REQUIRE_DIRECTX_PROOF_ENV = "CROSTL_REQUIRE_MLX_LAYER_NORM_DIRECTX_NATIVE_LOADER"
REQUIRE_OPENGL_PROOF_ENV = "CROSTL_REQUIRE_MLX_LAYER_NORM_OPENGL_NATIVE_LOADER"
AXIS_SIZE = 32
ROW_COUNT = 2
EPSILON = 1e-5
ABSOLUTE_TOLERANCE = 5e-5
RELATIVE_TOLERANCE = 5e-5


def _workgroup_access_assertion() -> str:
    return textwrap.dedent(f"""
        [[project.workgroup_access_assertions]]
        source = "{MLX_LAYER_NORM_SOURCE}"
        entry_point = "{MLX_LAYER_NORM_ENTRY}"
        function = "*"
        parameter = "*"
        minimum = 0
        maximum = 31
        """).strip()


def _directx_project_config(*, output_dir: str, dispatch_contract: str) -> str:
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_LAYER_NORM_SOURCE}"]
        include_dirs = ["."]
        targets = ["directx"]
        output_dir = "{output_dir}"
        dispatch_contracts = ["{dispatch_contract}"]

        [project.sources]
        "**/*.metal" = "metal"

        {_workgroup_access_assertion()}
        """).strip()


def _opengl_project_config(*, output_dir: str) -> str:
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_LAYER_NORM_SOURCE}"]
        include_dirs = ["."]
        targets = ["opengl"]
        output_dir = "{output_dir}"

        [project.sources]
        "**/*.metal" = "metal"

        [project.entry_points]
        "{MLX_LAYER_NORM_SOURCE}" = "{MLX_LAYER_NORM_ENTRY}"

        [project.entry_workgroup_size_rules."{MLX_LAYER_NORM_SOURCE}"]
        "{MLX_LAYER_NORM_ENTRY}" = [32, 1, 1]

        {_workgroup_access_assertion()}

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
    source_path = mlx_root / MLX_LAYER_NORM_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX LayerNorm source is missing: {source_path}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
        MLX_LAYER_NORM_SHA256
    )
    return mlx_root


def _dispatch_variant():
    manifest = load_dispatch_contract(MLX_LAYER_NORM_DISPATCH_CONTRACT)
    assert manifest.provenance["commit"] == MLX_COMMIT
    assert manifest.provenance["sourceReferences"] == {
        "hostDispatch": "mlx/backend/metal/normalization.cpp",
        "kernel": MLX_LAYER_NORM_SOURCE,
        "test": "python/tests/test_fast.py::test_layer_norm",
    }
    assert manifest.content_identity.to_json() == {
        "algorithm": "sha256",
        "value": MLX_LAYER_NORM_DISPATCH_IDENTITY,
    }
    variants = manifest.evaluate()
    assert len(variants) == 1
    variant = variants[0]
    assert variant.workload_id == "forward-float32-axis-32"
    assert variant.inputs == {
        "axisSize": AXIS_SIZE,
        "dtype": "float32",
        "isVjp": False,
        "nRows": ROW_COUNT,
    }
    assert variant.entry_point == MLX_LAYER_NORM_ENTRY
    assert variant.workgroup_size == (32, 1, 1)
    assert variant.subgroup_width == 32
    assert variant.dispatch_field == "workgroupCount"
    assert variant.dispatch_size == (ROW_COUNT, 1, 1)
    assert variant.specialization_constants == {}
    assert variant.artifact_id == MLX_LAYER_NORM_ARTIFACT_ID
    return variant


def _copy_dispatch_contract(destination: Path) -> None:
    _dispatch_variant()
    shutil.copyfile(MLX_LAYER_NORM_DISPATCH_CONTRACT, destination)


def _assert_directx_compiles(generated_path: Path, work_dir: Path) -> None:
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    arguments = dxc_compiler_arguments_for_source(
        generated_path.read_text(encoding="utf-8")
    )
    dxil_path = work_dir / "layer_normfloat32.dxil"
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
            pytest.fail("The LayerNorm OpenGL proof requires: " + ", ".join(missing))
        return

    spirv_path = work_dir / "layer_normfloat32.spv"
    assembly_path = work_dir / "layer_normfloat32.spvasm"
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


def _expected_binding_names(target: str) -> dict[int, str]:
    if target == "directx":
        return {
            0: "x",
            1: "w",
            2: "b",
            3: "out_",
            4: "layer_normfloat32_eps_Constants",
            5: "layer_normfloat32_axis_size_Constants",
            6: "layer_normfloat32_w_stride_Constants",
            7: "layer_normfloat32_b_stride_Constants",
        }
    return {
        0: "xBuffer",
        1: "wBuffer",
        2: "bBuffer",
        3: "out_Buffer",
        4: "layer_normfloat32_eps_Args",
        5: "layer_normfloat32_axis_size_Args",
        6: "layer_normfloat32_w_stride_Args",
        7: "layer_normfloat32_b_stride_Args",
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

    prefix = "layer_normfloat32_" if target == "directx" else ""
    return {
        names[0]: runtime_array("x"),
        names[1]: runtime_array("w"),
        names[2]: runtime_array("b"),
        names[3]: runtime_array("out_"),
        names[4]: uniform("float", "float32", prefix + "eps"),
        names[5]: uniform("uint", "uint32", prefix + "axis_size"),
        names[6]: uniform("uint", "uint32", prefix + "w_stride"),
        names[7]: uniform("uint", "uint32", prefix + "b_stride"),
    }


def _translate_artifact(mlx_root: Path, work_dir: Path, target: str) -> Path:
    output_dir = work_dir / "out"
    config_path = work_dir / "crosstl.toml"
    if target == "directx":
        contract_path = work_dir / "layer_norm.dispatch.json"
        _copy_dispatch_contract(contract_path)
        config_text = _directx_project_config(
            output_dir=output_dir.relative_to(mlx_root).as_posix(),
            dispatch_contract=contract_path.relative_to(mlx_root).as_posix(),
        )
    else:
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
    assert payload["project"]["workgroupAccessAssertions"] == [
        {
            "source": MLX_LAYER_NORM_SOURCE,
            "entryPoint": MLX_LAYER_NORM_ENTRY,
            "function": "*",
            "parameter": "*",
            "minimum": 0,
            "maximum": 31,
        }
    ]

    artifact = payload["artifacts"][0]
    expected = MLX_LAYER_NORM_GENERATED_ARTIFACTS[target]
    assert artifact["source"] == MLX_LAYER_NORM_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_LAYER_NORM_SHA256,
    }
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": expected["sha256"],
    }
    assert artifact["generatedSizeBytes"] == expected["sizeBytes"]
    assert artifact["entryPoint"] == {
        "source": MLX_LAYER_NORM_ENTRY,
        "target": "CSMain" if target == "directx" else "main",
        "stage": "compute",
    }
    assert artifact.get("specializationConstants", []) == []
    assert "specializationMaterialization" not in artifact
    materialization = artifact["templateMaterialization"]
    assert materialization["status"] == "materialized"
    assert materialization["specializationCount"] == 3
    assert materialization["accounting"] == {
        "reachableSpecializationCount": 6,
        "dependencyDiscoveryWorkCount": 8,
        "prunedCandidateCount": 194,
    }
    assert [item["name"] for item in materialization["specializations"]] == [
        "layer_norm_single_row",
        "initialize_buffer",
        "threadgroup_sum",
    ]
    assert materialization["specializations"][0]["parameters"] == {
        "N_READS": "8",
        "T": "float",
    }

    entry = artifact["execution"]["entryPoints"][0]
    assert entry["sourceEntryPoint"] == MLX_LAYER_NORM_ENTRY
    assert entry["workgroupSize"] == [32, 1, 1]
    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    assert "layer_normfloat32_local_buffer" in generated
    assert "initialize_buffer_1" in generated
    assert "threadgroup_sum_1" in generated
    if target == "directx":
        assert entry["subgroupWidth"] == 32
        assert artifact["dispatchArtifact"]["artifactId"] == (
            MLX_LAYER_NORM_ARTIFACT_ID
        )
        assert "[numthreads(32, 1, 1)]" in generated
        assert "[WaveSize(32)]" in generated
        assert dxc_compiler_arguments_for_source(generated) == ("-enable-16bit-types",)
        assert generated.count("WaveActiveSum(") == 2
        assert generated.count("GroupMemoryBarrierWithGroupSync();") == 3
        _assert_directx_compiles(generated_path, work_dir)
    else:
        assert "subgroupWidth" not in entry
        assert payload["project"]["subgroupWidthRules"] == {}
        assert payload["project"]["sourceOptions"]["metal"]["target_options"] == {
            "opengl": {"software_subgroup_width": 32}
        }
        assert "#define CROSSTL_SOFTWARE_SUBGROUP_WIDTH 32u" in generated
        assert "CROSSTL_REQUIRED_SUBGROUP_WIDTH" not in generated
        assert "GL_KHR_shader_subgroup" not in generated
        assert "gl_Subgroup" not in generated
        assert "subgroupAdd" not in generated
        assert generated.count("crossglSoftwareSubgroupSumFloat(") == 3
        assert generated.count("barrier();") == 6
        assert (
            "threadgroup_sum_1_glsl_xs_layer_normfloat32_local_buffer_float_32"
            in generated
        )
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
) -> tuple[dict, Path]:
    report_path = _translate_artifact(mlx_root, work_dir, target)
    runtime_artifacts = build_runtime_artifact_manifest(report_path)
    assert runtime_artifacts["success"] is True
    assert runtime_artifacts["summary"]["artifactCount"] == 1
    assert runtime_artifacts["summary"]["resourceBindingCount"] == 8
    assert runtime_artifacts["summary"]["specializationConstantCount"] == 0
    reflected = runtime_artifacts["artifacts"][0]
    assert reflected["hostInterface"]["status"] == "ready"
    assert reflected["hostInterface"]["specializationConstants"] == []
    resources = reflected["hostInterface"]["resources"]
    assert {resource["binding"]: resource["name"] for resource in resources} == (
        _expected_binding_names(target)
    )
    assert {resource["name"]: resource["access"] for resource in resources} == {
        name: "read_write" if binding == 3 else "read"
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
    assert descriptor["specializationConstants"] == []
    assert {
        binding["name"]: binding["scalarLayout"] for binding in descriptor["bindings"]
    } == _expected_layouts(target)
    return descriptor, package_dir


def _workload() -> tuple[list[float], list[float], list[float], list[float]]:
    values = [(index - 16) / 8.0 for index in range(AXIS_SIZE)]
    values.extend(((index % 9) - 4) * 0.3125 for index in range(AXIS_SIZE))
    weights = [0.5 + (index % 5) * 0.125 for index in range(AXIS_SIZE)]
    biases = [(index % 7 - 3) * 0.0625 for index in range(AXIS_SIZE)]
    expected: list[float] = []
    for row in range(ROW_COUNT):
        row_values = values[row * AXIS_SIZE : (row + 1) * AXIS_SIZE]
        mean = math.fsum(row_values) / AXIS_SIZE
        centered = [value - mean for value in row_values]
        variance = math.fsum(value * value for value in centered) / AXIS_SIZE
        normalizer = 1.0 / math.sqrt(variance + EPSILON)
        expected.extend(
            value * normalizer * weights[index] + biases[index]
            for index, value in enumerate(centered)
        )
    return values, weights, biases, expected


def _dispatch_request(descriptor: dict, package_dir: Path, target: str):
    names = _expected_binding_names(target)
    values, weights, biases, expected = _workload()
    request = build_native_loader_dispatch_request(
        descriptor,
        package_dir,
        {
            names[0]: {"dtype": "float32", "shape": [64], "values": values},
            names[1]: {"dtype": "float32", "shape": [32], "values": weights},
            names[2]: {"dtype": "float32", "shape": [32], "values": biases},
            names[4]: {
                "dtype": "float32",
                "shape": [1],
                "values": [EPSILON],
            },
            names[5]: {
                "dtype": "uint32",
                "shape": [1],
                "values": [AXIS_SIZE],
            },
            names[6]: {"dtype": "uint32", "shape": [1], "values": [1]},
            names[7]: {"dtype": "uint32", "shape": [1], "values": [1]},
        },
        {
            names[3]: {
                "dtype": "float32",
                "shape": [64],
                "values": expected,
                "tolerance": {
                    "absolute": ABSOLUTE_TOLERANCE,
                    "relative": RELATIVE_TOLERANCE,
                },
            }
        },
        (ROW_COUNT, 1, 1),
        expected_target=target,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    assert request.execution_plan.dispatch.workgroup_size == (32, 1, 1)
    assert request.execution_plan.dispatch.workgroup_count == (ROW_COUNT, 1, 1)
    assert request.execution_plan.dispatch.global_size == (64, 1, 1)
    return request, expected


def test_layer_norm_native_loader_dispatch_contract_is_exact():
    _dispatch_variant()


def test_pinned_mlx_layer_norm_translates_to_directx_native_loader_artifact():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-layer-norm-directx-translation-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_artifact(mlx_root, Path(temporary_directory), "directx")


def test_pinned_mlx_layer_norm_translates_to_software_subgroup_opengl():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-layer-norm-opengl-translation-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_artifact(mlx_root, Path(temporary_directory), "opengl")


def test_pinned_mlx_layer_norm_executes_through_directx_native_loader():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-layer-norm-directx-native-loader-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
            "directx",
        )
        request, expected = _dispatch_request(descriptor, package_dir, "directx")
        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id="mlx-layer-norm-directx-native-loader",
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
    assert result.outputs[output_name]["shape"] == [64]
    assert result.outputs[output_name]["values"] == pytest.approx(
        expected,
        abs=ABSOLUTE_TOLERANCE,
        rel=RELATIVE_TOLERANCE,
    )


def test_pinned_mlx_layer_norm_executes_through_opengl_native_loader():
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=".crosstl-layer-norm-opengl-native-loader-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
            "opengl",
        )
        request, expected = _dispatch_request(descriptor, package_dir, "opengl")
        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id="mlx-layer-norm-software-opengl-native-loader",
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
                require_env=REQUIRE_OPENGL_PROOF_ENV,
            )
        result = executor.run(request)

    assert result.status == "ok"
    output_name = _expected_binding_names("opengl")[3]
    assert result.outputs[output_name]["dtype"] == "float32"
    assert result.outputs[output_name]["shape"] == [64]
    assert result.outputs[output_name]["values"] == pytest.approx(
        expected,
        abs=ABSOLUTE_TOLERANCE,
        rel=RELATIVE_TOLERANCE,
    )
