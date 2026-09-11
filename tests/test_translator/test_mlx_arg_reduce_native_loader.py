from __future__ import annotations

import hashlib
import json
import os
import re
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
MLX_ARG_REDUCE_SOURCE = "mlx/backend/metal/kernels/arg_reduce.metal"
MLX_ARG_REDUCE_SHA256 = (
    "3d413c4d7eb5a6c397a52487721f445e08ba206a997a90fddcb7e51bf126d1f2"
)
MLX_ARG_REDUCE_DISPATCH_CONTRACT = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "arg_reduce.native-loader.dispatch.json"
)
MLX_ARG_REDUCE_DISPATCH_IDENTITY = (
    "9c9a196f8bf4c12264422136921d2c6cbf02e1f524e1791ef277406144df1c62"
)
MLX_ARG_REDUCE_VARIANTS = {
    "argmin-float32-axis-32-two-rows": {
        "entry": "argmin_float32",
        "inputs": {
            "argMax": False,
            "axisSize": 32,
            "dtype": "float32",
            "nRows": 2,
        },
        "artifactId": (
            "sha256:3e7aebfb8869a3c9169b5941a5457b582cc72a098ba3bfd1aec8e4f73c4f8e25"
        ),
        "dispatchVariantId": (
            "sha256:c000ab737f30b46a98bd1354ccffe502ed0b1460a77b8ba7940469d266d71ae3"
        ),
        "expected": [5, 7],
    },
    "argmax-float32-axis-32-two-rows": {
        "entry": "argmax_float32",
        "inputs": {
            "argMax": True,
            "axisSize": 32,
            "dtype": "float32",
            "nRows": 2,
        },
        "artifactId": (
            "sha256:1281be909c51f5c74d898e4c1e9992de3572d0135abc42e6bb8ad8308332fa82"
        ),
        "dispatchVariantId": (
            "sha256:5f6dbeea6ab134cc1d25a3914e643600b4c6eb474eae60a1dac869e1c79b1f06"
        ),
        "expected": [3, 2],
    },
}
MLX_ARG_REDUCE_ARTIFACTS = {
    "directx": {
        "argmin_float32": {
            "sha256": (
                "e3f7392023bbb6457eb03398a766bdaa128ed709d66ce814c7209cd13de7e896"
            ),
            "sizeBytes": 6655,
        },
        "argmax_float32": {
            "sha256": (
                "ef67c5d24ae7c7492a6676a35e0604800c1d18e4113c411fffaa2070090a92c3"
            ),
            "sizeBytes": 6657,
        },
    },
    "opengl": {
        "argmin_float32": {
            "sha256": (
                "b74534a5120665ad07755141af2a73702cb5ea504a0526b92306eabedfed4765"
            ),
            "sizeBytes": 7581,
        },
        "argmax_float32": {
            "sha256": (
                "d90e758132832490b7f356c6750d4deb2bdb3341f053a921228a9f24ce8d27d8"
            ),
            "sizeBytes": 7587,
        },
    },
}
REQUIRE_DIRECTX_RUNTIME_ENV = "CROSTL_REQUIRE_MLX_ARG_REDUCE_DIRECTX_NATIVE_LOADER"
REQUIRE_OPENGL_TOOLCHAIN_ENV = "CROSTL_REQUIRE_MLX_ARG_REDUCE_OPENGL_TOOLCHAIN"
REQUIRE_OPENGL_RUNTIME_ENV = "CROSTL_REQUIRE_MLX_ARG_REDUCE_OPENGL_NATIVE_LOADER"


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
    source_path = mlx_root / MLX_ARG_REDUCE_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX arg-reduce source is missing: {source_path}")
    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
        MLX_ARG_REDUCE_SHA256
    )
    return mlx_root


def _evaluated_variants():
    manifest = load_dispatch_contract(MLX_ARG_REDUCE_DISPATCH_CONTRACT)
    assert manifest.provenance["commit"] == MLX_COMMIT
    assert manifest.content_identity.to_json() == {
        "algorithm": "sha256",
        "value": MLX_ARG_REDUCE_DISPATCH_IDENTITY,
    }
    variants = {variant.workload_id: variant for variant in manifest.evaluate()}
    assert set(variants) == set(MLX_ARG_REDUCE_VARIANTS)
    for workload_id, expected in MLX_ARG_REDUCE_VARIANTS.items():
        variant = variants[workload_id]
        assert variant.inputs == expected["inputs"]
        assert variant.entry_point == expected["entry"]
        assert list(variant.workgroup_size) == [32, 1, 1]
        assert variant.subgroup_width == 32
        assert variant.dispatch_field == "workgroupCount"
        assert list(variant.dispatch_size) == [1, 2, 1]
        assert variant.artifact_id == expected["artifactId"]
        assert variant.variant_id == expected["dispatchVariantId"]
    return variants


def test_pinned_mlx_arg_reduce_dispatch_contract_is_exact():
    _evaluated_variants()


def _index_range_assertions() -> str:
    assertions = (
        ("in_idx + current_index * axis_stride", 63),
        ("out_idx", 1),
    )
    return "\n\n".join(textwrap.dedent(f"""
            [[project.index_range_assertions]]
            source = "{MLX_ARG_REDUCE_SOURCE}"
            expression = "{expression}"
            minimum = 0
            maximum = {maximum}
            """).strip() for expression, maximum in assertions)


def _project_config(*, output_dir: str, target: str, entry: str) -> str:
    sections = [textwrap.dedent(f"""
            [project]
            source_roots = ["mlx/backend/metal/kernels"]
            include = ["{MLX_ARG_REDUCE_SOURCE}"]
            include_dirs = ["."]
            targets = ["{target}"]
            output_dir = "{output_dir}"

            [project.sources]
            "**/*.metal" = "metal"

            [project.entry_points]
            "{MLX_ARG_REDUCE_SOURCE}" = "{entry}"

            [project.entry_workgroup_size_rules."{MLX_ARG_REDUCE_SOURCE}"]
            "{entry}" = [32, 1, 1]
            """).strip()]
    if target == "directx":
        sections.append(textwrap.dedent(f"""
                [project.subgroup_width_rules]
                "{MLX_ARG_REDUCE_SOURCE}" = 32
                """).strip())
    else:
        sections.append(_index_range_assertions())
    sections.append(textwrap.dedent("""
            [project.source_options.metal]
            max_template_specializations = 128
            max_template_materialization_work = 8192
            """).strip())
    if target == "directx":
        sections.append(textwrap.dedent("""
                [project.source_options.metal.target_options.directx]
                relative_wave_shuffle_out_of_range = "self"
                """).strip())
    elif target == "opengl":
        sections.append(textwrap.dedent("""
                [project.source_options.metal.target_options.opengl]
                software_subgroup_width = 32
                """).strip())
    return "\n\n".join(sections)


def _assert_software_spirv(generated_path: Path, work_dir: Path) -> None:
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
            pytest.fail("The arg-reduce OpenGL proof requires: " + ", ".join(missing))
        return

    spirv_path = work_dir / "arg-reduce.spv"
    assembly_path = work_dir / "arg-reduce.spvasm"
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
    assert assembly.count("OpControlBarrier") == 5
    assert "OpGroupNonUniform" not in assembly
    assert "OpExecutionMode %main LocalSize 32 1 1" in assembly


def _translate_artifact(
    mlx_root: Path,
    work_dir: Path,
    *,
    target: str,
    workload_id: str,
) -> tuple[Path, Path]:
    variant = _evaluated_variants()[workload_id]
    entry = variant.entry_point
    output_dir = work_dir / "out"
    config_path = work_dir / "crosstl.toml"
    config_path.write_text(
        _project_config(
            output_dir=output_dir.relative_to(mlx_root).as_posix(),
            target=target,
            entry=entry,
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
    expected_identity = MLX_ARG_REDUCE_ARTIFACTS[target][entry]
    assert artifact["source"] == MLX_ARG_REDUCE_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_ARG_REDUCE_SHA256,
    }
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": expected_identity["sha256"],
    }
    assert artifact["generatedSizeBytes"] == expected_identity["sizeBytes"]
    assert artifact["entryPoint"] == {
        "source": entry,
        "target": "CSMain" if target == "directx" else "main",
        "stage": "compute",
    }
    materialization = artifact["templateMaterialization"]
    assert materialization["status"] == "materialized"
    assert materialization["specializationCount"] == 5
    assert materialization["accounting"] == {
        "reachableSpecializationCount": 10,
        "dependencyDiscoveryWorkCount": 75,
        "prunedCandidateCount": 274,
    }
    assert [
        item
        for item in materialization["specializations"]
        if item["name"] == "elem_to_loc"
    ] == [
        {
            "name": "elem_to_loc",
            "materializedName": "elem_to_loc_int64_t",
            "parameters": {"IdxT": "int64_t"},
            "parameterSources": {"IdxT": "call-site"},
            "source": "call-site",
        }
    ]
    entry_record = artifact["execution"]["entryPoints"][0]
    assert entry_record["sourceEntryPoint"] == entry
    assert entry_record["workgroupSize"] == [32, 1, 1]
    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    if target == "directx":
        assert entry_record["subgroupWidth"] == 32
        assert "[numthreads(32, 1, 1)]" in generated
        assert "[WaveSize(32)]" in generated
        assert "int64_t elem_to_loc_int64_t(int64_t elem," in generated
        assert (
            "uint3 elem"
            not in generated.split("int64_t elem_to_loc_int64_t(int64_t elem,", 1)[1]
        )
        assert (
            "float __crossgl_wave_shuffle_down_self_float("
            "float value, uint delta)" in generated
        )
        assert (
            "uint __crossgl_wave_shuffle_down_self_uint("
            "uint value, uint delta)" in generated
        )
        assert "bool valid = delta < (laneCount - lane);" in generated
        assert "uint sourceLane = valid" in generated
        assert "? (lane + delta)\n        : lane;" in generated
        wave_reads = re.findall(r"WaveReadLaneAt\(([^)]*)\)", generated)
        assert wave_reads
        assert all(read.endswith(", sourceLane") for read in wave_reads)
        assert (
            "__crossgl_wave_shuffle_down_self_uint(data.index, uint(delta))"
            in generated
        )
        assert (
            "__crossgl_wave_shuffle_down_self_float(data.val, uint(delta))" in generated
        )
    else:
        assert "subgroupWidth" not in entry_record
        assert "#define CROSSTL_SOFTWARE_SUBGROUP_WIDTH 32u" in generated
        assert "GL_KHR_shader_subgroup" not in generated
        assert "OpGroupNonUniform" not in generated
        assert "crossglSoftwareSubgroupShuffleDownFloat" in generated
        assert "crossglSoftwareSubgroupShuffleDownUint" in generated
        assert generated.count("barrier();") == 5
        _assert_software_spirv(generated_path, work_dir)

    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True
    return report_path, generated_path


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("workload_id", list(MLX_ARG_REDUCE_VARIANTS))
def test_pinned_mlx_arg_reduce_translates_to_bounded_artifact(target, workload_id):
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-arg-reduce-{target}-{workload_id}-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_artifact(
            mlx_root,
            Path(temporary_directory),
            target=target,
            workload_id=workload_id,
        )


def _expected_element_types(target: str, entry: str) -> dict[str, str]:
    if target == "directx":
        return {
            f"{entry}_ndim_Constants": "uint64",
            f"{entry}_axis_stride_Constants": "int64",
            f"{entry}_axis_size_Constants": "uint64",
            "CrossGLDispatchInfo": "uint32",
            "in_": "float32",
            "out_": "uint32",
            "shape": "int32",
            "in_strides": "int64",
            "out_strides": "int64",
        }
    return {
        "in_Buffer": "float32",
        "out_Buffer": "uint32",
        "shapeBuffer": "int32",
        "in_stridesBuffer": "int64",
        "out_stridesBuffer": "int64",
        f"{entry}_ndim_Args": "uint64",
        f"{entry}_axis_stride_Args": "int64",
        f"{entry}_axis_size_Args": "uint64",
    }


def _build_runtime_package(
    mlx_root: Path,
    work_dir: Path,
    *,
    target: str,
    workload_id: str,
) -> tuple[dict, Path]:
    report_path, _ = _translate_artifact(
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

    variant = _evaluated_variants()[workload_id]
    entry = variant.entry_point
    resources = reflected["hostInterface"]["resources"]
    expected_element_types = _expected_element_types(target, entry)
    assert {resource["name"] for resource in resources} == set(expected_element_types)
    assert {
        resource["name"]: resource["scalarLayout"]["elementType"]
        for resource in resources
    } == expected_element_types
    for resource in resources:
        if resource["scalarLayout"]["elementType"] in {"int64", "uint64"}:
            expected_alignment = 16 if resource["kind"] == "constant-buffer" else 8
            assert resource["scalarLayout"]["alignmentBytes"] == expected_alignment

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
    assert descriptor["target"] == target
    assert descriptor["entryPoint"]["name"] == (
        "CSMain" if target == "directx" else "main"
    )
    if target == "directx":
        assert descriptor["entryPoint"]["executionConfig"] == {
            "numthreads": [32, 1, 1],
            "subgroupWidth": 32,
        }
        dispatch_binding = next(
            binding
            for binding in descriptor["bindings"]
            if binding["name"] == "CrossGLDispatchInfo"
        )
        assert dispatch_binding["provenance"]["executionInput"] == {
            "kind": "dispatch-workgroup-count",
            "valueSource": "dispatch.workgroupCount",
            "coordinateSpace": "physical",
            "dimensions": 3,
            "memberName": "crossglNumWorkGroups",
        }
    else:
        assert descriptor["entryPoint"]["executionConfig"] == {
            "local_size": [32, 1, 1],
            "local_size_x": 32,
            "local_size_y": 1,
            "local_size_z": 1,
        }
    assert descriptor["specializationConstants"] == []
    assert {
        binding["name"]: binding["scalarLayout"]["elementType"]
        for binding in descriptor["bindings"]
    } == expected_element_types
    return descriptor, package_dir


def _arg_reduce_values() -> list[float]:
    row0 = [float(((index * 7) % 23) - 11) for index in range(32)]
    row1 = [float(((index * 11) % 29) - 14) for index in range(32)]
    row0[3] = row0[17] = 50.0
    row0[5] = row0[25] = -50.0
    row1[2] = row1[29] = 60.0
    row1[7] = row1[30] = -60.0
    return row0 + row1


def _runtime_request(
    descriptor: dict,
    package_dir: Path,
    *,
    target: str,
    workload_id: str,
):
    variant = _evaluated_variants()[workload_id]
    entry = variant.entry_point
    if target == "directx":
        names = {
            "input": "in_",
            "output": "out_",
            "shape": "shape",
            "inStrides": "in_strides",
            "outStrides": "out_strides",
            "ndim": f"{entry}_ndim_Constants",
            "axisStride": f"{entry}_axis_stride_Constants",
            "axisSize": f"{entry}_axis_size_Constants",
        }
    else:
        names = {
            "input": "in_Buffer",
            "output": "out_Buffer",
            "shape": "shapeBuffer",
            "inStrides": "in_stridesBuffer",
            "outStrides": "out_stridesBuffer",
            "ndim": f"{entry}_ndim_Args",
            "axisStride": f"{entry}_axis_stride_Args",
            "axisSize": f"{entry}_axis_size_Args",
        }
    expected = MLX_ARG_REDUCE_VARIANTS[workload_id]["expected"]
    request = build_native_loader_dispatch_request(
        descriptor,
        package_dir,
        {
            names["input"]: {
                "dtype": "float32",
                "shape": [64],
                "values": _arg_reduce_values(),
            },
            names["shape"]: {
                "dtype": "int32",
                "shape": [1],
                "values": [2],
            },
            names["inStrides"]: {
                "dtype": "int64",
                "shape": [1],
                "values": [32],
            },
            names["outStrides"]: {
                "dtype": "int64",
                "shape": [1],
                "values": [1],
            },
            names["ndim"]: {
                "dtype": "uint64",
                "shape": [1],
                "values": [1],
            },
            names["axisStride"]: {
                "dtype": "int64",
                "shape": [1],
                "values": [1],
            },
            names["axisSize"]: {
                "dtype": "uint64",
                "shape": [1],
                "values": [32],
            },
        },
        {
            names["output"]: {
                "dtype": "uint32",
                "shape": [2],
                "values": expected,
            }
        },
        variant.dispatch_size,
        expected_target=target,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    assert request.execution_plan.dispatch.workgroup_size == (32, 1, 1)
    assert request.execution_plan.dispatch.workgroup_count == (1, 2, 1)
    assert request.execution_plan.dispatch.global_size == (32, 2, 1)
    if target == "directx":
        generated = [
            bound
            for bound in request.execution_plan.resource_bindings
            if bound.value is not None and "executionInput" in bound.value.metadata
        ]
        assert len(generated) == 1
        assert generated[0].binding.name == "CrossGLDispatchInfo"
        assert generated[0].value.metadata["source"] == "dispatch.workgroupCount"
        assert generated[0].value.values == [1, 2, 1]
    return request, names["output"], expected


# CI invokes these selectors with pytest-xdist. Keep both workloads in one test
# item per target so native WARP/EGL dispatches cannot overlap across workers.
def test_pinned_mlx_arg_reduce_executes_through_directx_native_loader():
    mlx_root = _pinned_mlx_root()
    for workload_id in MLX_ARG_REDUCE_VARIANTS:
        with tempfile.TemporaryDirectory(
            prefix=f".crosstl-arg-reduce-directx-{workload_id}-",
            dir=mlx_root,
        ) as temporary_directory:
            descriptor, package_dir = _build_runtime_package(
                mlx_root,
                Path(temporary_directory),
                target="directx",
                workload_id=workload_id,
            )
            request, output_name, expected = _runtime_request(
                descriptor,
                package_dir,
                target="directx",
                workload_id=workload_id,
            )
            executor = RuntimeParityExecutor(
                RuntimeTestAdapterSpec(
                    adapter_id=f"mlx-arg-reduce-directx-{workload_id}",
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
        assert result.outputs[output_name]["dtype"] == "uint32"
        assert result.outputs[output_name]["shape"] == [2]
        assert result.outputs[output_name]["values"] == expected


def test_pinned_mlx_arg_reduce_executes_through_opengl_native_loader():
    mlx_root = _pinned_mlx_root()
    for workload_id in MLX_ARG_REDUCE_VARIANTS:
        with tempfile.TemporaryDirectory(
            prefix=f".crosstl-arg-reduce-opengl-{workload_id}-",
            dir=mlx_root,
        ) as temporary_directory:
            descriptor, package_dir = _build_runtime_package(
                mlx_root,
                Path(temporary_directory),
                target="opengl",
                workload_id=workload_id,
            )
            request, output_name, expected = _runtime_request(
                descriptor,
                package_dir,
                target="opengl",
                workload_id=workload_id,
            )
            executor = RuntimeParityExecutor(
                RuntimeTestAdapterSpec(
                    adapter_id=f"mlx-arg-reduce-opengl-{workload_id}",
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
        assert result.outputs[output_name]["dtype"] == "uint32"
        assert result.outputs[output_name]["shape"] == [2]
        assert result.outputs[output_name]["values"] == expected
