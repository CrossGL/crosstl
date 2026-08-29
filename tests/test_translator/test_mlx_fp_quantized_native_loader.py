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
from crosstl.project.directx_toolchain import dxc_compiler_arguments_for_source

ROOT = Path(__file__).resolve().parents[2]
MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
MLX_MXFP4_SOURCE = "mlx/backend/metal/kernels/fp_quantized.metal"
MLX_MXFP4_SHA256 = "ef4ba099710a63a0b5d27d3e5ce69a8528bee8f1757805aa606c8d8e43de18d4"
MLX_MXFP4_SOURCE_SIZE_BYTES = 9700
MLX_MXFP4_ENTRY = "mxfp4_quantize_dequantize_float_gs_32_b_4_hgs_false"
MLX_MXFP4_DISPATCH_CONTRACT = (
    ROOT
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "fp_quantized.native-loader.dispatch.json"
)
MLX_MXFP4_DISPATCH_IDENTITY = (
    "5256e32b364ac303a6873f28b5ac3e9a1a811ac5c38bc41a977bce9191a025ed"
)
MLX_MXFP4_ARTIFACT_ID = (
    "sha256:bde1bfa31c116a52a1dc3b6e546dfa2ee43dc968719393ba04f629b4e2d95319"
)
MLX_MXFP4_VARIANT_ID = (
    "sha256:ebd6ab3f40f5839764f592943180ba64f11a66f10a5561b9468019032c04df8a"
)
MLX_MXFP4_GENERATED_ARTIFACTS = {
    "directx": {
        "sha256": "4e8044758d65b6b2c189092ce56fff3c5ba7948221883de490c1a4b9c5563352",
        "sizeBytes": 7809,
    },
    "opengl": {
        "sha256": "cbbe989c40317c04ffe915f1f314f55db8896edfd38f04ad4b8882be53b2a4da",
        "sizeBytes": 9571,
    },
}
REQUIRE_DIRECTX_RUNTIME_ENV = "CROSTL_REQUIRE_MLX_MXFP4_DIRECTX_NATIVE_LOADER"
REQUIRE_OPENGL_RUNTIME_ENV = "CROSTL_REQUIRE_MLX_MXFP4_OPENGL_NATIVE_LOADER"
WORKGROUP_SIZE = (32, 1, 1)
WORKGROUP_COUNT = (1, 1, 1)
INDEX_ASSERTION = {
    "source": MLX_MXFP4_SOURCE,
    "expression": "index",
    "minimum": 0,
    "maximum": 31,
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
    source_path = mlx_root / MLX_MXFP4_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX MXFP4 source is missing: {source_path}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == MLX_COMMIT
    assert source_path.stat().st_size == MLX_MXFP4_SOURCE_SIZE_BYTES
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_MXFP4_SHA256
    return mlx_root


def _dispatch_variant():
    manifest = load_dispatch_contract(MLX_MXFP4_DISPATCH_CONTRACT)
    assert manifest.provenance["commit"] == MLX_COMMIT
    assert manifest.provenance["sourceReferences"] == {
        "hostDispatch": "mlx/backend/metal/quantized.cpp",
        "implementation": "mlx/backend/metal/kernels/fp_quantized.h",
        "kernel": MLX_MXFP4_SOURCE,
    }
    assert manifest.content_identity.to_json() == {
        "algorithm": "sha256",
        "value": MLX_MXFP4_DISPATCH_IDENTITY,
    }
    variants = manifest.evaluate()
    assert len(variants) == 1
    variant = variants[0]
    assert variant.workload_id == (
        "mxfp4-quantize-dequantize-float32-32-no-global-scale"
    )
    assert variant.inputs == {
        "bits": 4,
        "dtype": "float32",
        "elementCount": 32,
        "groupSize": 32,
        "hasGlobalScale": False,
        "isMxfp4": True,
        "rowContiguous": True,
    }
    assert variant.variant_id == MLX_MXFP4_VARIANT_ID
    assert variant.artifact_id == MLX_MXFP4_ARTIFACT_ID
    assert variant.entry_point == MLX_MXFP4_ENTRY
    assert variant.workgroup_size == WORKGROUP_SIZE
    assert variant.subgroup_width == 32
    assert variant.dispatch_field == "workgroupCount"
    assert variant.dispatch_size == WORKGROUP_COUNT
    assert variant.specialization_constants == {}
    return variant


def test_current_mlx_mxfp4_dispatch_contract_is_exact():
    _dispatch_variant()


def _project_config(
    target: str,
    *,
    output_dir: str,
    dispatch_contract: str | None = None,
) -> str:
    sections = [textwrap.dedent(f"""
            [project]
            source_roots = ["mlx/backend/metal/kernels"]
            include = ["{MLX_MXFP4_SOURCE}"]
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
                "{MLX_MXFP4_SOURCE}" = "{MLX_MXFP4_ENTRY}"

                [project.entry_workgroup_size_rules."{MLX_MXFP4_SOURCE}"]
                "{MLX_MXFP4_ENTRY}" = [32, 1, 1]
                """).strip())
    if target == "opengl":
        sections.append(textwrap.dedent(f"""
                [[project.index_range_assertions]]
                source = "{INDEX_ASSERTION['source']}"
                expression = "{INDEX_ASSERTION['expression']}"
                minimum = {INDEX_ASSERTION['minimum']}
                maximum = {INDEX_ASSERTION['maximum']}
                """).strip())
    elif target != "directx":
        raise ValueError(f"Unsupported MXFP4 test target: {target}")
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


def _dxc_path() -> str | None:
    discovered = shutil.which("dxc")
    if discovered is not None:
        return discovered
    cached = Path.home() / ".cache" / "crosstl-tools" / "bin" / "dxc"
    return str(cached) if cached.is_file() else None


def _assert_directx_compiles(generated_path: Path) -> None:
    dxc = _dxc_path()
    if dxc is None:
        if os.environ.get(REQUIRE_DIRECTX_RUNTIME_ENV) == "1":
            pytest.fail("The MXFP4 DirectX proof requires dxc")
        return

    source = generated_path.read_text(encoding="utf-8")
    arguments = dxc_compiler_arguments_for_source(source)
    assert arguments == ("-enable-16bit-types",)
    # The cached macOS DXC wrapper only maps paths below this worktree, and the
    # short fixture also avoids legacy dxc.exe MAX_PATH failures on Windows.
    with tempfile.TemporaryDirectory(prefix=".mq-dxc-", dir=ROOT) as temporary:
        work_dir = Path(temporary)
        source_path = work_dir / "m.hlsl"
        dxil_path = work_dir / "m.dxil"
        assembly_path = work_dir / "m.asm"
        source_path.write_text(source, encoding="utf-8")
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
                "-Fc",
                str(assembly_path),
                str(source_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert dxil_path.stat().st_size > 0
        assembly = assembly_path.read_text(encoding="utf-8")
        assert "bitcast i16" in assembly
        assert "uitofp i16" not in assembly
        assert "sitofp i16" not in assembly


def _assert_opengl_spirv(generated_path: Path, work_dir: Path) -> None:
    tools = {
        name: shutil.which(name)
        for name in ("glslangValidator", "spirv-val", "spirv-dis")
    }
    if not all(tools.values()):
        missing = [name for name, path in tools.items() if path is None]
        if os.environ.get(REQUIRE_OPENGL_RUNTIME_ENV) == "1":
            pytest.fail("The MXFP4 OpenGL proof requires: " + ", ".join(missing))
        return

    spirv_path = work_dir / "m.spv"
    assembly_path = work_dir / "m.spvasm"
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
    assert "OpExecutionMode %main LocalSize 32 1 1" in assembly


def _binding_names(target: str) -> dict[str, str]:
    if target == "directx":
        return {"input": "w", "global_scale": "global_scale", "output": "out_"}
    return {
        "input": "wBuffer",
        "global_scale": "global_scaleBuffer",
        "output": "out_Buffer",
    }


def _translate_artifact(mlx_root: Path, work_dir: Path, target: str) -> Path:
    output_dir = work_dir / "o"
    dispatch_contract = None
    if target == "directx":
        contract_path = work_dir / "d.json"
        _dispatch_variant()
        shutil.copyfile(MLX_MXFP4_DISPATCH_CONTRACT, contract_path)
        dispatch_contract = contract_path.relative_to(mlx_root).as_posix()
    else:
        _dispatch_variant()

    config_path = work_dir / "c.toml"
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
    if target == "opengl":
        assert payload["project"]["indexRangeAssertions"] == [INDEX_ASSERTION]
        assert payload["project"]["subgroupWidthRules"] == {}
        assert payload["project"]["sourceOptions"]["metal"]["target_options"] == {
            "opengl": {"software_subgroup_width": 32}
        }

    artifact = payload["artifacts"][0]
    expected = MLX_MXFP4_GENERATED_ARTIFACTS[target]
    assert artifact["source"] == MLX_MXFP4_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_MXFP4_SHA256,
    }
    assert artifact["sourceSizeBytes"] == MLX_MXFP4_SOURCE_SIZE_BYTES
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": expected["sha256"],
    }
    assert artifact["generatedSizeBytes"] == expected["sizeBytes"]
    assert artifact["entryPoint"] == {
        "source": MLX_MXFP4_ENTRY,
        "target": "CSMain" if target == "directx" else "main",
        "stage": "compute",
    }
    assert artifact["provenance"]["pipeline"] == "entry-scoped-translate"

    materialization = artifact["templateMaterialization"]
    assert materialization["status"] == "materialized"
    assert materialization["specializationCount"] == 1
    assert materialization["unsupported"] == []
    assert materialization["accounting"] == {
        "reachableSpecializationCount": 9,
        "dependencyDiscoveryWorkCount": 0,
        "prunedCandidateCount": 25492,
    }
    assert materialization["specializations"][0]["materializedName"] == (
        MLX_MXFP4_ENTRY
    )
    assert materialization["specializations"][0]["parameters"] == {
        "T": "float",
        "bits": "4",
        "group_size": "32",
        "has_global_scale": "false",
    }

    entry = artifact["execution"]["entryPoints"][0]
    assert entry["sourceEntryPoint"] == MLX_MXFP4_ENTRY
    assert entry["workgroupSize"] == list(WORKGROUP_SIZE)
    if target == "directx":
        assert entry["subgroupWidth"] == 32
        assert artifact["dispatchArtifact"]["artifactId"] == MLX_MXFP4_ARTIFACT_ID
        assert artifact["dispatchArtifact"]["dispatchVariantIds"] == [
            MLX_MXFP4_VARIANT_ID
        ]
        assert artifact["dispatchArtifact"]["manifestContentIdentities"] == [
            f"sha256:{MLX_MXFP4_DISPATCH_IDENTITY}"
        ]

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    global_scale_lines = [
        line for line in generated.splitlines() if "global_scale" in line
    ]
    assert len(global_scale_lines) == 1
    assert "crosstl_ctor_fp8_e8m0_1(scale_dec_b)" in generated
    assert "crosstl_ctor_value.bits" in generated
    assert "fp8_e4m3" in generated
    assert "fp8_e4m3" not in next(
        line for line in generated.splitlines() if "float scale =" in line
    )
    if target == "directx":
        assert "[numthreads(32, 1, 1)]" in generated
        assert "[WaveSize(32)]" in generated
        assert "scale_dec_b = WaveActiveMax(abs(w_thread));" in generated
        assert "float scale = fp8_e8m0__operator_float(" in generated
        assert generated.count("asfloat16(") == 2
        assert "float16_t(uint16_t(" not in generated
        assert "int n = int(round(le));" in generated
        assert "metal_u3a_u3a" not in generated
        assert "__crossgl_physical_subgroup" not in generated
        _assert_directx_compiles(generated_path)
    else:
        assert "layout(local_size_x = 32, local_size_y = 1" in generated
        assert "#define CROSSTL_SOFTWARE_SUBGROUP_WIDTH 32u" in generated
        assert "shared float crossglSoftwareSubgroupScratchFloat[32];" in generated
        assert (
            "scale_dec_b = crossglSoftwareSubgroupMaxFloat(abs(w_thread));" in generated
        )
        assert "float scale = fp8_e8m0_operator_float(" in generated
        assert "(floatBitsToUint(x) & 0x7f800000u)" in generated
        assert "isfinite(" not in generated
        assert "GL_KHR_shader_subgroup" not in generated
        assert "gl_Subgroup" not in generated
        assert "subgroupMax" not in generated
        assert generated.count("unpackHalf2x16(") == 2
        assert "float converted = unpackHalf2x16((v & 0xffffu)).x;" in generated
        assert generated.count("barrier();") == 3
        _assert_opengl_spirv(generated_path, work_dir)

    report_path = work_dir / "r.json"
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
    assert runtime_artifacts["success"] is True, json.dumps(runtime_artifacts, indent=2)
    assert runtime_artifacts["summary"]["artifactCount"] == 1
    assert runtime_artifacts["summary"]["resourceBindingCount"] == (
        4 if target == "directx" else 3
    )
    reflected = runtime_artifacts["artifacts"][0]["hostInterface"]
    assert reflected["status"] == "ready"
    assert reflected["diagnostics"] == []
    resources = reflected["resources"]
    names = _binding_names(target)
    reflected_data = {
        resource["name"]: (
            resource["binding"],
            resource["access"],
            resource["scalarLayout"]["elementType"],
        )
        for resource in resources
        if resource["name"] != "CrossGLDispatchInfo"
    }
    assert reflected_data == {
        names["input"]: (0, "read", "float32"),
        names["global_scale"]: (1, "read", "float32"),
        names["output"]: (2, "read_write", "float32"),
    }
    if target == "directx":
        dispatch_info = next(
            resource
            for resource in resources
            if resource["name"] == "CrossGLDispatchInfo"
        )
        assert dispatch_info["binding"] == 0
        assert dispatch_info["kind"] == "constant-buffer"
        assert dispatch_info["provenance"]["executionInput"]["valueSource"] == (
            "dispatch.workgroupCount"
        )

    runtime_artifacts_path = work_dir / "a.json"
    _write_json(runtime_artifacts_path, runtime_artifacts)
    package_dir = work_dir / "p"
    package = build_runtime_package(runtime_artifacts_path, package_dir)
    assert package["success"] is True, json.dumps(package, indent=2)
    loader_manifest = build_runtime_loader_manifest(
        package_dir / "runtime-package.json"
    )
    assert loader_manifest["success"] is True, json.dumps(loader_manifest, indent=2)
    assert loader_manifest["summary"]["readyLoadUnitCount"] == 1
    assert loader_manifest["summary"]["blockedLoadUnitCount"] == 0
    descriptor = build_native_loader_abi_descriptor(
        loader_manifest,
        load_unit_id=loader_manifest["loadUnits"][0]["id"],
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
    descriptor_data = {
        binding["name"]: (
            binding["coordinates"]["binding"],
            binding["access"],
            binding["scalarLayout"]["elementType"],
        )
        for binding in descriptor["bindings"]
        if binding["name"] != "CrossGLDispatchInfo"
    }
    assert descriptor_data == reflected_data
    return descriptor, package_dir


def _mxfp4_workload() -> list[float]:
    values = [
        -6.0,
        -4.0,
        -3.0,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        6.0,
        4.0,
        3.0,
        2.0,
        1.5,
        1.0,
        0.5,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
        0.0,
        0.0,
    ]
    assert len(values) == 32
    assert max(abs(value) for value in values) == 6.0
    return values


def _dispatch_request(target: str, descriptor: dict, package_dir: Path):
    names = _binding_names(target)
    values = _mxfp4_workload()
    request = build_native_loader_dispatch_request(
        descriptor,
        package_dir,
        {
            names["input"]: {
                "dtype": "float32",
                "shape": [32],
                "values": values,
            },
            # The source signature retains buffer(1), but has_global_scale=false
            # statically removes every read. The host leaves it unbound; the
            # generic loader allocates this inert reflected placeholder.
            names["global_scale"]: {
                "dtype": "float32",
                "shape": [1],
                "values": [1.0],
            },
        },
        {
            names["output"]: {
                "dtype": "float32",
                "shape": [32],
                "values": values,
                "tolerance": {"absolute": 0.0, "relative": 0.0},
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
    return request, names["output"], values


def _execute_current_mlx_mxfp4(target: str) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(prefix=".mq-", dir=mlx_root) as temporary:
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
            else OpenGLRuntimeParityAdapter(
                runtime=OpenGLComputeRuntime(context_backends=("egl",))
            )
        )
        executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id=f"mlx-mxfp4-{target}-native-loader",
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
    assert result.outputs[output_name]["shape"] == [32]
    assert result.outputs[output_name]["values"] == expected


def test_current_mlx_mxfp4_executes_through_directx_native_loader():
    _execute_current_mlx_mxfp4("directx")


def test_current_mlx_mxfp4_executes_with_opengl_software_subgroups():
    _execute_current_mlx_mxfp4("opengl")
