from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
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
MLX_UNARY_SOURCE = "mlx/backend/metal/kernels/unary.metal"
MLX_UNARY_SHA256 = "51af04126d68e1f5baee5f467268408650d24a68db66e8c044f7f0be3f15368b"
REQUIRE_PROOF_ENVS = {
    "directx": "CROSTL_REQUIRE_MLX_UNARY_DIRECTX_NATIVE_LOADER",
    "metal": "CROSTL_REQUIRE_MLX_UNARY_METAL_ROUNDTRIP",
    "opengl": "CROSTL_REQUIRE_MLX_UNARY_OPENGL_NATIVE_LOADER",
}


@dataclass(frozen=True)
class UnaryWorkload:
    name: str
    entry_point: str
    operator_type: str
    generated_operation: dict[str, str]
    generated_artifacts: dict[str, dict[str, str | int]]
    input_values: tuple[float, ...]
    expected_values: tuple[float, ...]
    absolute_tolerance: float
    relative_tolerance: float


SQUARE_WORKLOAD = UnaryWorkload(
    name="square",
    entry_point="v_Squarefloat32float32",
    operator_type="Square",
    generated_operation={
        "directx": "return (x * x);",
        "metal": "return x * x;",
        "opengl": "return (x * x);",
    },
    generated_artifacts={
        "directx": {
            "sha256": (
                "64540a89c95e39914a4d616aff9bec98b939a5209fa4caef5cc1425511abb4e5"
            ),
            "sizeBytes": 2314,
        },
        "metal": {
            "sha256": (
                "244e34b7aa58b7abe7c3ff09f3f51f3aa283a42bf7585bf88200590767032495"
            ),
            "sizeBytes": 1015,
        },
        "opengl": {
            "sha256": (
                "2bb46a3bb0858eb849e533bfe46eff1d59b9192436e15b2639c7998698db6a48"
            ),
            "sizeBytes": 3613,
        },
    },
    input_values=(-3.0, -1.5, 0.0, 2.0, 4.25),
    expected_values=(9.0, 2.25, 0.0, 4.0, 18.0625),
    absolute_tolerance=1e-6,
    relative_tolerance=1e-6,
)

ARCCOS_WORKLOAD = UnaryWorkload(
    name="arccos",
    entry_point="v_ArcCosfloat32float32",
    operator_type="ArcCos",
    generated_operation={
        "directx": "return __crossgl_metal_precise_acos_float(x);",
        "metal": "return __crossgl_metal_precise_acos_float(x);",
        "opengl": "return crossgl_metal_precise_acos_float(x);",
    },
    generated_artifacts={
        "directx": {
            "sha256": (
                "4562332ad4fb951478ca419180ccdf3589f74b9e0956226badc6de877d343239"
            ),
            "sizeBytes": 4175,
        },
        "metal": {
            "sha256": (
                "1247739bc0c48d11692aee81953d8a6a4071de488bfe7ea8d7b2083aa48d9b2b"
            ),
            "sizeBytes": 2742,
        },
        "opengl": {
            "sha256": (
                "280864c39e88198cd5e660127db453877349fadb090cb37f022bcc46300660b3"
            ),
            "sizeBytes": 5965,
        },
    },
    input_values=(-1.0, -0.5, 0.0, 0.5, 1.0),
    expected_values=(
        3.141592653589793,
        2.0943951023931957,
        1.5707963267948966,
        1.0471975511965979,
        0.0,
    ),
    absolute_tolerance=1e-6,
    relative_tolerance=1e-5,
)


FLOAT32_UNARY_METAL_ARTIFACT_IDENTITIES = {
    "Abs": ("1c225a5efe013e30a3022926c09fcf951b34e478848b2f38559f81f0912ae967", 989),
    "ArcCos": (
        "1247739bc0c48d11692aee81953d8a6a4071de488bfe7ea8d7b2083aa48d9b2b",
        2742,
    ),
    "ArcCosh": (
        "dfea065f445a672ac49014c81df90799e533a4435e0ca2ad6548f15236e8a86b",
        1027,
    ),
    "ArcSin": (
        "16fb2dc2b32d43e79ef7ef5412a976a643af0983f1919c14fb15cd9771115c67",
        1017,
    ),
    "ArcSinh": (
        "95989af274be41c1b5e8584198dcdc978eb8d9d37b05319f03d1dc79f1e16e4b",
        1027,
    ),
    "ArcTan": (
        "525caa32b75facb1f267acd8bdab88146aaea9408c1099c489d8796f567d1016",
        1017,
    ),
    "ArcTanh": (
        "5915c6c9998dee80281c91f4ce1831fc690963f78b6d34a94643c2bfafc1b2af",
        1027,
    ),
    "Ceil": ("b687d1a08d534d44b6cb7e8f38e18f858239077597e1ef209224e3c5e1428d50", 999),
    "Cos": ("254204b39a4da7277372aa44ed7b0c7ae52f179d108d9312e10319d4eb841bfb", 989),
    "Cosh": ("70c16dd91be3aca1b54e88af72a150de09f34990b17bb99520b2839b613f7c58", 999),
    "Exp": ("ab5207445d620502e66f8a4d215b423c5d817e365c69a3a7569566076e4a96e9", 989),
    "Expm1": ("4cd6bc9a9bd8937eed1dc11a71ec65d00b23ad6f455f5f3e5c27a7cca890dbc8", 2003),
    "Floor": ("409df915fc3869066df9c2b9b127fde6ce16a35ec2bcc7ead07890e40f623188", 1009),
    "Log": ("770b0befc6b8970e08e205ffff7d2f8be3fe7e303cb6882ca86d32627752b42a", 989),
    "Log2": ("fe12519ca0749b9f4c362572faed177c87aa903f6181edf7ad5311730e6e45ad", 999),
    "Log10": ("e6add6cf4944a842df04d993fefa3d3b102cb8f8c05060c57aa3130fa5c8797b", 1009),
    "Log1p": ("23e3bf52d386fe4a78fa7db0d44ce9aee4b884da7afef88a999d0c0850554267", 1279),
    "Negative": (
        "29a647c06f40977101e3c50408e1118a440f322f38e23758f90266603686b05c",
        1030,
    ),
    "Sigmoid": (
        "2300101589b447614c6deff51b9fee39bd6c44e01f22f7a7b53a0da0ded0d27b",
        1073,
    ),
    "Erf": ("30ca7a972cf75191160decadc7c0b02abad90bc3ec9342c5daf914dd90aa4831", 2707),
    "ErfInv": (
        "9eb863c7ebf468b07780d986463d6434d925a5ec7dabf01da18721deb82fe83c",
        1893,
    ),
    "Sign": ("f3804c200fd8f8dce96d5a63f6dbc71ccc36c8c5d39a80923105dfa83ac3ba3c", 1023),
    "Sin": ("556d5d04978a0ad37d3a224b30f46b039cab44d7956aa131a1a200ce2e891f75", 989),
    "Sinh": ("08e011639022b4ab43997f7dffa03a605e3a63df2b11e365f6dcd1aae88f7ebc", 999),
    "Square": (
        "244e34b7aa58b7abe7c3ff09f3f51f3aa283a42bf7585bf88200590767032495",
        1015,
    ),
    "Sqrt": ("88a3324c88ec900652bd4816037b449d32d4abbb151f1e6d9fb6310c85b2aa94", 999),
    "Rsqrt": ("db5434e6844457c3e78e013bfaf139afaf63e29171a0d27ecd826a7b84948a48", 1009),
    "Tan": ("35f7ab6b8e0d4aec3476f0956f261be1b2f300b9c21a846fea3fb56f5968742a", 989),
    "Tanh": ("f82fc0eec4708ee6e8dc6315d173b91e819cfc248ce9bcf4e7a5518bc0fb8e39", 999),
    "Round": ("51a5152996626c057daabb9199d19f81c9a1cfe7d93a2533cee468772eee1206", 1008),
}


def _metal_roundtrip_workload(operator_type: str) -> UnaryWorkload:
    if operator_type == SQUARE_WORKLOAD.operator_type:
        return SQUARE_WORKLOAD
    if operator_type == ARCCOS_WORKLOAD.operator_type:
        return ARCCOS_WORKLOAD
    sha256, size_bytes = FLOAT32_UNARY_METAL_ARTIFACT_IDENTITIES[operator_type]
    return UnaryWorkload(
        name=operator_type.lower(),
        entry_point=f"v_{operator_type}float32float32",
        operator_type=operator_type,
        generated_operation={},
        generated_artifacts={
            "metal": {
                "sha256": sha256,
                "sizeBytes": size_bytes,
            }
        },
        input_values=(),
        expected_values=(),
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )


FLOAT32_UNARY_METAL_WORKLOADS = tuple(
    _metal_roundtrip_workload(operator_type)
    for operator_type in FLOAT32_UNARY_METAL_ARTIFACT_IDENTITIES
)


def _project_config(target: str, workload: UnaryWorkload) -> str:
    return textwrap.dedent(f"""
        [project]
        source_roots = ["mlx/backend/metal/kernels"]
        include = ["{MLX_UNARY_SOURCE}"]
        include_dirs = ["."]
        targets = ["{target}"]
        output_dir = ".crosstl-mlx-unary-native-loader/out"

        [project.sources]
        "**/*.metal" = "metal"

        [project.entry_points]
        "{MLX_UNARY_SOURCE}" = "{workload.entry_point}"

        [project.entry_workgroup_size_rules."{MLX_UNARY_SOURCE}"]
        "{workload.entry_point}" = [1, 1, 1]

        [project.source_options.metal]
        max_template_specializations = 64
        max_template_materialization_work = 4096
        """).strip()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _skip_or_fail(target: str, message: str) -> None:
    if os.environ.get(REQUIRE_PROOF_ENVS[target]) == "1":
        pytest.fail(message)
    pytest.skip(message)


def _pinned_mlx_root() -> Path:
    root_value = os.environ.get("CROSTL_MLX_ROOT")
    if not root_value:
        if any(os.environ.get(name) == "1" for name in REQUIRE_PROOF_ENVS.values()):
            pytest.fail("CROSTL_MLX_ROOT is not configured")
        pytest.skip("CROSTL_MLX_ROOT is not configured")

    mlx_root = Path(root_value).resolve()
    source_path = mlx_root / MLX_UNARY_SOURCE
    if not source_path.is_file():
        pytest.fail(f"Pinned MLX unary source is missing: {source_path}")

    checkout_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mlx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_commit == MLX_COMMIT
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == MLX_UNARY_SHA256
    return mlx_root


def _translate_unary_artifact(
    mlx_root: Path,
    work_dir: Path,
    target: str,
    workload: UnaryWorkload,
):
    config_path = work_dir / "crosstl.toml"
    config_path.write_text(
        _project_config(target, workload) + "\n",
        encoding="utf-8",
    )
    output_dir = work_dir / "out"
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
    artifact = payload["artifacts"][0]
    expected_identity = workload.generated_artifacts[target]
    assert artifact["source"] == MLX_UNARY_SOURCE
    assert artifact["sourceHash"] == {
        "algorithm": "sha256",
        "value": MLX_UNARY_SHA256,
    }
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": expected_identity["sha256"],
    }
    assert artifact["generatedSizeBytes"] == expected_identity["sizeBytes"]
    expected_target_entry = {
        "directx": "CSMain",
        "metal": workload.entry_point,
        "opengl": "main",
    }[target]
    assert artifact["entryPoint"] == {
        "source": workload.entry_point,
        "target": expected_target_entry,
        "stage": "compute",
    }
    assert artifact["provenance"]["pipeline"] == "entry-scoped-translate"
    assert artifact["execution"]["entryPoints"][0]["workgroupSize"] == [1, 1, 1]
    materialization = artifact["templateMaterialization"]
    assert materialization["status"] == "materialized"
    assert materialization["specializationCount"] == 1
    assert materialization["specializations"] == [
        {
            "name": "unary_v",
            "materializedName": workload.entry_point,
            "parameters": {
                "N": "1",
                "Op": workload.operator_type,
                "T": "float",
                "U": "float",
            },
            "parameterSources": {
                "N": "source-instantiation",
                "Op": "source-instantiation",
                "T": "source-instantiation",
                "U": "source-instantiation",
            },
            "source": "source-instantiation",
            "hostName": workload.entry_point,
        }
    ]
    assert materialization["unsupported"] == []

    generated_path = mlx_root / artifact["path"]
    generated = generated_path.read_text(encoding="utf-8")
    expected_operation = workload.generated_operation.get(target)
    if expected_operation is not None:
        assert expected_operation in generated
    if workload is ARCCOS_WORKLOAD:
        assert "return acos(x);" not in generated
        if target == "opengl":
            assert "precise float crossgl_metal_precise_acos" not in generated
            assert "precise float crossglPreciseReturn" in generated
        elif target == "directx":
            assert "precise float __crossgl_metal_precise_acos" in generated
        else:
            assert target == "metal"
            assert "float __crossgl_metal_precise_acos_ratio(float value)" in generated
            assert "float __crossgl_metal_precise_acos_float(float value)" in generated
            assert generated.count("#pragma clang fp contract(off)") == 2
            assert generated.count("#pragma clang fp contract(fast)") == 2
    assert "Log{}(x + i * Sqrt{}(1.0 - x * x))" not in generated
    if target == "directx":
        assert "[numthreads(1, 1, 1)]" in generated
        validator = "dxc"
    elif target == "opengl":
        assert (
            "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1)" in generated
        )
        validator = "glslangValidator"
    else:
        assert target == "metal"
        assert f"kernel void {workload.entry_point}" in generated
        assert "[[static]]" not in generated
        assert f"struct {workload.operator_type} {{" in generated
        assert generated.count(f"struct {workload.operator_type} {{") == 1
        for pruned_operator in set(FLOAT32_UNARY_METAL_ARTIFACT_IDENTITIES) - {
            workload.operator_type
        }:
            assert f"struct {pruned_operator} {{" not in generated
        validator = "xcrun"
    if shutil.which(validator) is not None:
        toolchain_runs = payload["validation"]["toolchainRuns"]
        assert len(toolchain_runs) == 1
        assert toolchain_runs[0]["status"] == "ok"
    elif os.environ.get(REQUIRE_PROOF_ENVS[target]) == "1":
        pytest.fail(f"{validator} is required for the MLX unary {target} proof")

    report_path = work_dir / "portability-report.json"
    report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True
    return report_path


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_pinned_mlx_unary_square_translates_to_selected_target(target):
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-unary-{target}-translation-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_unary_artifact(
            mlx_root,
            Path(temporary_directory),
            target,
            SQUARE_WORKLOAD,
        )


def _roundtrip_pinned_mlx_unary_through_metal(workload: UnaryWorkload) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-unary-{workload.name}-metal-roundtrip-",
        dir=mlx_root,
    ) as temporary_directory:
        work_dir = Path(temporary_directory)
        report_path = _translate_unary_artifact(
            mlx_root,
            work_dir,
            "metal",
            workload,
        )
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        artifact = report_payload["artifacts"][0]
        generated_path = mlx_root / artifact["path"]

        runtime_artifacts = build_runtime_artifact_manifest(report_path)
        assert runtime_artifacts["success"] is True, json.dumps(
            runtime_artifacts,
            indent=2,
        )
        reflected = runtime_artifacts["artifacts"][0]["hostInterface"]
        assert reflected["status"] == "ready"
        assert reflected["entryPoints"] == [
            {
                "name": workload.entry_point,
                "stage": "compute",
                "executionConfig": {},
            }
        ]
        assert {
            resource["name"]: (
                resource["kind"],
                resource["binding"],
                resource["access"],
                resource["metadata"]["entryPoint"],
            )
            for resource in reflected["resources"]
        } == {
            "in_": ("buffer", 0, "read", workload.entry_point),
            "out_": (
                "buffer",
                1,
                "read_write",
                workload.entry_point,
            ),
            "size": (
                "constant-buffer",
                2,
                "read",
                workload.entry_point,
            ),
        }

        xcrun = shutil.which("xcrun")
        if xcrun is None:
            _skip_or_fail("metal", "xcrun is required for the MLX unary Metal proof")
        air_path = work_dir / f"{workload.entry_point}.air"
        compiled = subprocess.run(
            [
                xcrun,
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(generated_path),
                "-o",
                str(air_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert compiled.returncode == 0, compiled.stdout + compiled.stderr
        assert air_path.is_file()
        assert air_path.stat().st_size > 0


def test_pinned_mlx_unary_square_roundtrips_through_metal():
    _roundtrip_pinned_mlx_unary_through_metal(SQUARE_WORKLOAD)


def test_pinned_mlx_unary_arccos_roundtrips_through_metal():
    _roundtrip_pinned_mlx_unary_through_metal(ARCCOS_WORKLOAD)


@pytest.mark.parametrize(
    "workload",
    FLOAT32_UNARY_METAL_WORKLOADS,
    ids=lambda workload: workload.operator_type,
)
def test_current_mlx_float32_unary_family_roundtrips_through_metal(workload):
    _roundtrip_pinned_mlx_unary_through_metal(workload)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_pinned_mlx_unary_arccos_translates_to_selected_target(target):
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-unary-arccos-{target}-translation-",
        dir=mlx_root,
    ) as temporary_directory:
        _translate_unary_artifact(
            mlx_root,
            Path(temporary_directory),
            target,
            ARCCOS_WORKLOAD,
        )


def _build_runtime_package(
    mlx_root: Path,
    work_dir: Path,
    target: str,
    workload: UnaryWorkload,
) -> tuple[dict, Path]:
    report_path = _translate_unary_artifact(
        mlx_root,
        work_dir,
        target,
        workload,
    )
    runtime_artifacts = build_runtime_artifact_manifest(report_path)
    assert runtime_artifacts["success"] is True, json.dumps(
        runtime_artifacts,
        indent=2,
    )
    assert runtime_artifacts["summary"]["artifactCount"] == 1
    reflected = runtime_artifacts["artifacts"][0]
    assert reflected["hostInterface"]["status"] == "ready"

    expected_resources = {
        "directx": {
            "in_": (0, "read"),
            "out_": (1, "read_write"),
            f"{workload.entry_point}_size_Constants": (2, "read"),
        },
        "opengl": {
            "in_Buffer": (0, "read"),
            "out_Buffer": (1, "read_write"),
            f"{workload.entry_point}_size_Args": (2, "read"),
        },
    }[target]
    resources = reflected["hostInterface"]["resources"]
    assert {
        resource["name"]: (resource["binding"], resource["access"])
        for resource in resources
    } == expected_resources

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
    if target == "directx":
        assert execution_config == {"numthreads": [1, 1, 1]}
    else:
        assert execution_config == {
            "local_size": [1, 1, 1],
            "local_size_x": 1,
            "local_size_y": 1,
            "local_size_z": 1,
        }
    assert {
        binding["name"]: binding["access"] for binding in descriptor["bindings"]
    } == {name: access for name, (_binding, access) in expected_resources.items()}
    return descriptor, package_dir


def _execute_pinned_mlx_unary(
    target: str,
    workload: UnaryWorkload,
) -> None:
    mlx_root = _pinned_mlx_root()
    with tempfile.TemporaryDirectory(
        prefix=f".crosstl-unary-{workload.name}-{target}-native-loader-",
        dir=mlx_root,
    ) as temporary_directory:
        descriptor, package_dir = _build_runtime_package(
            mlx_root,
            Path(temporary_directory),
            target,
            workload,
        )
        if target == "directx":
            input_binding = "in_"
            output_binding = "out_"
            size_binding = f"{workload.entry_point}_size_Constants"
        else:
            input_binding = "in_Buffer"
            output_binding = "out_Buffer"
            size_binding = f"{workload.entry_point}_size_Args"
        input_values = list(workload.input_values)
        expected_values = list(workload.expected_values)
        request = build_native_loader_dispatch_request(
            descriptor,
            package_dir,
            {
                input_binding: {
                    "dtype": "float32",
                    "shape": [len(input_values)],
                    "values": input_values,
                },
                size_binding: {
                    "dtype": "uint32",
                    "shape": [1],
                    "values": [len(input_values)],
                },
            },
            {
                output_binding: {
                    "dtype": "float32",
                    "shape": [len(expected_values)],
                    "values": expected_values,
                    "tolerance": {
                        "absolute": workload.absolute_tolerance,
                        "relative": workload.relative_tolerance,
                    },
                }
            },
            (len(input_values), 1, 1),
            expected_target=target,
        )
        assert request.execution_plan is not None
        assert request.execution_plan.diagnostics == ()
        assert request.execution_plan.dispatch.workgroup_size == (1, 1, 1)
        assert request.execution_plan.dispatch.global_size == (
            len(input_values),
            1,
            1,
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
                adapter_id=f"mlx-unary-{workload.name}-{target}-native-loader",
                target=target,
                executor=target,
                adapter_kind=f"{target}-native-runtime",
            ),
            runtime_adapter=runtime_adapter,
        )
        availability = executor.is_available(request)
        if not availability.available:
            _skip_or_fail(
                target,
                availability.reason or f"The native {target} runtime is unavailable",
            )

        result = executor.run(request)

    assert result.status == "ok"
    assert result.outputs[output_binding]["dtype"] == "float32"
    assert result.outputs[output_binding]["shape"] == [len(expected_values)]
    assert result.outputs[output_binding]["values"] == pytest.approx(
        expected_values,
        abs=workload.absolute_tolerance,
        rel=workload.relative_tolerance,
    )


def test_pinned_mlx_unary_square_executes_through_directx_native_loader():
    _execute_pinned_mlx_unary("directx", SQUARE_WORKLOAD)


def test_pinned_mlx_unary_square_executes_through_opengl_native_loader():
    _execute_pinned_mlx_unary("opengl", SQUARE_WORKLOAD)


def test_pinned_mlx_unary_arccos_executes_through_directx_native_loader():
    _execute_pinned_mlx_unary("directx", ARCCOS_WORKLOAD)


def test_pinned_mlx_unary_arccos_executes_through_opengl_native_loader():
    _execute_pinned_mlx_unary("opengl", ARCCOS_WORKLOAD)
