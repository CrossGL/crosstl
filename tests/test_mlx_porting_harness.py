import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "demos" / "integrations" / "mlx" / "run_mlx_porting.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("run_mlx_porting", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _full_corpus_report(module, mlx_root, work_dir, *, include_extra_failure=False):
    per_target = {
        target: {
            "translatedCount": module.EXPECTED_METAL_KERNEL_COUNT - 1,
            "failedCount": 1,
        }
        for target in module.FULL_CORPUS_TARGETS
    }
    diagnostics_by_code = {
        contract["diagnosticCode"]: 1
        for contract in module.MLX_FENCE_TARGET_CONTRACTS.values()
    }
    diagnostics_by_code["project.validate.failed-artifact"] = 3
    missing_capability_counts = {
        contract["missingCapability"]: 1
        for contract in module.MLX_FENCE_TARGET_CONTRACTS.values()
    }
    missing_capability_counts["batch.translation"] = 3
    diagnostics = []
    artifacts = []
    extensions = {"directx": ".hlsl", "opengl": ".glsl", "vulkan": ".spvasm"}
    for target, contract in module.MLX_FENCE_TARGET_CONTRACTS.items():
        message = module._atomic_fence_expected_message(contract)
        artifact_path = (
            work_dir
            / "out-full-corpus"
            / target
            / Path(module.MLX_FENCE_SOURCE).with_suffix(extensions[target])
        ).relative_to(mlx_root)
        diagnostics.append(
            {
                "severity": "error",
                "code": contract["diagnosticCode"],
                "message": message,
                "location": {"file": module.MLX_FENCE_SOURCE},
                "target": target,
                "sourceBackend": "metal",
                "missingCapabilities": [contract["missingCapability"]],
            }
        )
        diagnostics.append(
            {
                "severity": "error",
                "code": "project.validate.failed-artifact",
                "message": f"Artifact translation failed before validation: {message}",
                "location": {"file": module.MLX_FENCE_SOURCE},
                "target": target,
                "sourceBackend": "metal",
                "missingCapabilities": ["batch.translation"],
            }
        )
        artifacts.append(
            {
                "source": module.MLX_FENCE_SOURCE,
                "sourceBackend": "metal",
                "target": target,
                "path": artifact_path.as_posix(),
                "status": "failed",
                "error": message,
            }
        )

    translated_count = module.FULL_CORPUS_EXPECTED_TRANSLATED_ARTIFACT_COUNT
    failed_count = module.FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
    if include_extra_failure:
        per_target["directx"] = {
            "translatedCount": module.EXPECTED_METAL_KERNEL_COUNT - 2,
            "failedCount": 2,
        }
        translated_count -= 1
        failed_count += 1
        diagnostics_by_code["project.translate.failed"] = 1
        diagnostics_by_code["project.validate.failed-artifact"] += 1
        missing_capability_counts["batch.translation"] += 2
        diagnostics.extend(
            [
                {
                    "severity": "error",
                    "code": "project.translate.failed",
                    "message": "unrelated full-corpus translation failure",
                    "location": {"file": "mlx/backend/metal/kernels/other.metal"},
                    "target": "directx",
                    "sourceBackend": "metal",
                    "missingCapabilities": ["batch.translation"],
                },
                {
                    "severity": "error",
                    "code": "project.validate.failed-artifact",
                    "message": "Artifact translation failed before validation",
                    "location": {"file": "mlx/backend/metal/kernels/other.metal"},
                    "target": "directx",
                    "sourceBackend": "metal",
                    "missingCapabilities": ["batch.translation"],
                },
            ]
        )
        artifacts.append(
            {
                "source": "mlx/backend/metal/kernels/other.metal",
                "sourceBackend": "metal",
                "target": "directx",
                "path": (
                    work_dir / "out-full-corpus/directx/other.hlsl"
                ).relative_to(mlx_root).as_posix(),
                "status": "failed",
                "error": "unrelated full-corpus translation failure",
            }
        )

    summary = {
        "unitCount": module.EXPECTED_METAL_KERNEL_COUNT,
        "artifactCount": module.FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
        "translatedCount": translated_count,
        "failedCount": failed_count,
        "diagnosticCounts": {
            "error": len(diagnostics),
            "note": 0,
            "warning": 0,
        },
        "diagnosticsByCode": diagnostics_by_code,
        "missingCapabilityCounts": missing_capability_counts,
        "artifactsByTarget": per_target,
    }
    return {
        "summary": summary,
        "diagnostics": diagnostics,
        "artifacts": artifacts,
        "validation": {"summary": {"failedCount": failed_count}},
    }


def _translated_arange_report(module, target):
    return {
        "kind": "crosstl-project-portability-report",
        "project": {"targets": [target]},
        "artifacts": [
            {
                "source": module.MLX_ARANGE_SOURCE,
                "path": f"out/{target}/arange",
                "target": target,
                "sourceBackend": "metal",
                "status": "translated",
            }
        ],
    }


def _runtime_arange_artifact_manifest(module, target, output_name="out"):
    entry_point = module.RUNTIME_READINESS_ENTRY_POINTS[target]
    return {
        "kind": "crosstl-project-runtime-artifact-manifest",
        "project": {"targets": [target]},
        "summary": {
            "artifactCount": 1,
            "entryPointCount": 1,
            "resourceBindingCount": 3,
            "dispatchMetadataCount": 1,
        },
        "artifacts": [
            {
                "id": (
                    f"{module.MLX_ARANGE_SOURCE}|{target}|default|out/{target}/arange"
                ),
                "source": module.MLX_ARANGE_SOURCE,
                "path": f"out/{target}/arange",
                "target": target,
                "sourceBackend": "metal",
                "status": "translated",
                "entryPoints": [
                    {
                        "name": entry_point,
                        "stage": "compute",
                        "workgroupSize": [1, 1, 1],
                    }
                ],
                "resourceBindings": [
                    {
                        "name": "start",
                        "kind": "constant",
                        "binding": 0,
                    },
                    {
                        "name": "step",
                        "kind": "constant",
                        "binding": 1,
                    },
                    {
                        "name": output_name,
                        "kind": "buffer",
                        "binding": 2,
                    },
                ],
                "dispatch": {
                    "entryPoint": entry_point,
                    "workgroupSize": [1, 1, 1],
                    "workgroupCount": [1, 1, 1],
                },
            }
        ],
        "runtimeDiagnosticCounts": {"note": 0, "warning": 0, "error": 0},
        "runtimeDiagnostics": [],
    }


def _write_metal_roundtrip_report(module, mlx_root, work_dir, report_path):
    source_path = mlx_root / module.MLX_METAL_ROUNDTRIP_SOURCE
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "#include <metal_atomic>\n[[kernel]] void fence_wait() {}\n",
        encoding="utf-8",
    )
    generated_path = (
        work_dir / "out-metal-roundtrip" / "metal" / module.MLX_METAL_ROUNDTRIP_SOURCE
    )
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(
        "\n".join(
            (
                "#include <metal_stdlib>",
                "using namespace metal;",
                "kernel void input_coherent(device uint* input [[buffer(0)]], ",
                "    uint index [[thread_position_in_grid]]) {",
                "  metal::atomic_thread_fence(metal::mem_flags::mem_device,",
                "      metal::memory_order_seq_cst, metal::thread_scope_system);",
                "}",
                "kernel void fence_update(device uint* out [[buffer(0)]]) {",
                "  metal::atomic_thread_fence(metal::mem_flags::mem_device,",
                "      metal::memory_order_seq_cst, metal::thread_scope_system);",
                "}",
                "kernel void fence_wait(device uint* out [[buffer(0)]]) {",
                "  metal::atomic_thread_fence(metal::mem_flags::mem_device,",
                "      metal::memory_order_seq_cst, metal::thread_scope_system);",
                "}",
                "",
            )
        ),
        encoding="utf-8",
    )
    artifact_path = generated_path.relative_to(mlx_root).as_posix()
    artifact = {
        "source": module.MLX_METAL_ROUNDTRIP_SOURCE,
        "sourceBackend": "metal",
        "target": "metal",
        "path": artifact_path,
        "status": "translated",
        "provenance": {
            "pipeline": "single-file-translate",
            "intermediate": "crossgl",
        },
        "sourceHash": {
            "algorithm": "sha256",
            "value": module._sha256(source_path),
        },
        "generatedHash": {
            "algorithm": "sha256",
            "value": module._sha256(generated_path),
        },
        "sourceSizeBytes": source_path.stat().st_size,
        "generatedSizeBytes": generated_path.stat().st_size,
    }
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "unitCount": 1,
                    "artifactCount": 1,
                    "translatedCount": 1,
                    "failedCount": 0,
                    "diagnosticCounts": {"error": 0, "note": 0, "warning": 0},
                },
                "artifacts": [artifact],
                "validation": {
                    "summary": {
                        "artifactCount": 1,
                        "okCount": 1,
                        "failedCount": 0,
                    },
                    "artifacts": [
                        {
                            "path": artifact_path,
                            "status": "ok",
                            "sourceHashStatus": "ok",
                            "generatedHashStatus": "ok",
                            "sourceSizeStatus": "ok",
                            "generatedSizeStatus": "ok",
                            "sourceMapStatus": "ok",
                            "sourceRemapStatus": "ok",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def test_metal_roundtrip_validates_generated_artifact_natively(tmp_path, monkeypatch):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)
    commands = []

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        commands.append((name, list(command)))
        if name == "translate-metal-roundtrip":
            _write_metal_roundtrip_report(
                module,
                mlx_root,
                work_dir,
                report_dir / "metal-roundtrip.json",
            )
        elif name == "validate-metal-roundtrip-native":
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_bytes(b"AIR")
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(
        module,
        "_probe_native_metal_toolchain",
        lambda *args: {
            "status": "available",
            "platform": "darwin",
            "xcrun": "/usr/bin/xcrun",
            "reason": None,
        },
    )

    result = module._check_metal_roundtrip(
        mlx_root,
        work_dir,
        config_dir,
        report_dir,
        log_dir,
        "python",
        require_metal_toolchain=True,
    )

    config = (config_dir / "metal-roundtrip.toml").read_text(encoding="utf-8")
    assert f'include = ["{module.MLX_METAL_ROUNDTRIP_SOURCE}"]' in config
    assert 'targets = ["metal"]' in config
    assert result["roundTripStages"] == ["metal", "crossgl", "metal"]
    assert result["artifactValidationStatus"] == "validated"
    assert result["fenceContract"] == {
        "memoryFlags": ["mem_device"],
        "memoryOrder": "memory_order_seq_cst",
        "threadScope": "thread_scope_system",
        "occurrences": module.MLX_FENCE_EXPECTED_ATOMIC_FENCE_COUNT,
        "preserved": True,
    }
    assert result["semanticReadinessStatus"] == "blocked"
    assert result["semanticTrackedIssues"] == [
        "https://github.com/CrossGL/crosstl/issues/1660"
    ]
    assert result["runtimeParityClaimed"] is False
    assert result["nativeMetalValidation"]["status"] == "validated"
    assert result["nativeMetalValidation"]["required"] is True
    assert result["nativeMetalValidation"]["artifactCompiled"] is True
    assert [name for name, _command in commands] == [
        "translate-metal-roundtrip",
        "validate-metal-roundtrip-native",
    ]
    assert "--validate" in commands[0][1]
    native_command = commands[1][1]
    assert native_command[:5] == [
        "/usr/bin/xcrun",
        "-sdk",
        "macosx",
        "metal",
        "-c",
    ]
    assert Path(native_command[5]) == mlx_root / result["artifact"]
    assert (mlx_root / result["nativeMetalValidation"]["compiledArtifact"]).is_file()


def test_runtime_readiness_uses_runtime_artifact_manifest_metadata(
    tmp_path, monkeypatch
):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    report_dir = mlx_root / ".crosstl-mlx-porting" / "reports"
    report_dir.mkdir(parents=True)
    artifact_report = report_dir / "directx-readiness-artifacts.json"
    artifact_report.write_text(
        json.dumps(_translated_arange_report(module, "directx")),
        encoding="utf-8",
    )

    build_calls = []

    def fake_runtime_artifact_manifest(report_path):
        build_calls.append(Path(report_path))
        return _runtime_arange_artifact_manifest(module, "directx")

    monkeypatch.setattr(
        module,
        "build_runtime_artifact_manifest",
        fake_runtime_artifact_manifest,
    )

    result = module._plan_runtime_readiness_for_report(
        mlx_root=mlx_root,
        report_dir=report_dir,
        name="directx-runtime-readiness",
        artifact_report=artifact_report,
        targets=("directx",),
        require_vulkan_native_runtime=False,
    )

    assert build_calls == [artifact_report]
    assert result["status"] == "planned"
    assert result["trackedRuntimeIssues"] == [
        "https://github.com/CrossGL/crosstl/issues/1388",
        "https://github.com/CrossGL/crosstl/issues/1471",
    ]
    assert result["testCount"] == 1
    assert result["diagnosticCounts"] == {"error": 0, "note": 0, "warning": 0}
    assert result["metadataGapCodes"] == []
    assert result["planBlockerCodes"] == []
    assert result["runtimeArtifactSummary"]["resourceBindingCount"] == 3
    assert (mlx_root / result["fixtureMetadata"]).is_file()
    assert (mlx_root / result["runtimeArtifactManifest"]).is_file()
    assert (mlx_root / result["runtimeTestManifest"]).is_file()
    assert (mlx_root / result["runtimeTestPlan"]).is_file()
    assert result["runtimeFixtureExecutionIncluded"] is True
    execution = result["runtimeFixtureExecution"]
    assert execution["status"] == "passed"
    assert execution["summary"]["fixtureCount"] == 1
    assert execution["summary"]["passedCount"] == 1
    assert execution["summary"]["failedCount"] == 0
    assert execution["projectRunnerSummary"]["skippedCount"] == 1
    assert (mlx_root / execution["fixtureMetadata"]).is_file()
    assert (mlx_root / execution["runtimeTestManifest"]).is_file()
    assert (mlx_root / execution["projectTestRunnerPlan"]).is_file()
    assert (mlx_root / execution["projectTestRunnerReport"]).is_file()
    native_execution = result["nativeRuntimeExecution"]
    assert result["nativeRuntimeExecutionIncluded"] is True
    assert native_execution["status"] == "blocked-by-runtime-driver"
    assert native_execution["summary"]["fixtureCount"] == 1
    assert native_execution["summary"]["unavailableCount"] == 1
    assert (mlx_root / native_execution["fixtureMetadata"]).is_file()
    assert (mlx_root / native_execution["runtimeTestManifest"]).is_file()
    assert (mlx_root / native_execution["projectTestRunnerPlan"]).is_file()
    assert (mlx_root / native_execution["projectTestRunnerReport"]).is_file()

    manifest = json.loads((mlx_root / result["runtimeTestManifest"]).read_text())
    assert manifest["success"] is True
    assert manifest["summary"]["testsByTarget"] == {"directx": 1}
    assert manifest["metadata"]["trackedIssues"] == [
        "https://github.com/CrossGL/crosstl/issues/1388",
        "https://github.com/CrossGL/crosstl/issues/1471",
    ]
    assert manifest["tests"][0]["selector"] == {
        "source": module.MLX_ARANGE_SOURCE,
        "target": "directx",
    }
    assert manifest["tests"][0]["entryPoint"] == "CSMain"

    plan = json.loads((mlx_root / result["runtimeTestPlan"]).read_text())
    assert plan["testCases"][0]["runtimeExecution"]["dispatch"]["entryPoint"] == (
        "CSMain"
    )

    runner_report = json.loads(
        (mlx_root / execution["projectTestRunnerReport"]).read_text()
    )
    runtime_result = runner_report["runtimeTestReport"]["results"][0]
    assert runner_report["success"] is True
    assert runtime_result["status"] == "passed"
    assert (
        runtime_result["executor"]["details"]["runtimeParityAdapter"]["runtimeAdapter"]
        == "mlx-arange-reference-runtime"
    )


def test_runtime_fixture_execution_metadata_uses_toolchain_free_adapters():
    module = _load_harness()

    metadata = module._runtime_fixture_execution_metadata(
        ("directx", "opengl", "vulkan")
    )

    assert metadata["metadata"]["runtimeFixtureExecutionIncluded"] is True
    assert {adapter["id"] for adapter in metadata["adapters"]} == {
        "mlx-arange-reference-directx",
        "mlx-arange-reference-opengl",
        "mlx-arange-reference-vulkan",
    }
    assert all(
        adapter["platformRequirements"]["requiredTools"] == []
        for adapter in metadata["adapters"]
    )
    assert all("target" not in adapter for adapter in metadata["adapters"])
    assert {fixture["adapter"] for fixture in metadata["fixtures"]} == {
        "mlx-arange-reference-directx",
        "mlx-arange-reference-opengl",
        "mlx-arange-reference-vulkan",
    }


def test_native_runtime_execution_metadata_uses_target_executors():
    module = _load_harness()

    metadata = module._native_runtime_execution_metadata(
        ("directx", "opengl", "vulkan")
    )

    assert metadata["metadata"]["nativeRuntimeExecutionIncluded"] is True
    assert {
        (adapter["id"], adapter["executor"], adapter["adapterKind"])
        for adapter in metadata["adapters"]
    } == {
        ("mlx-arange-native-directx", "directx", "directx-native-runtime"),
        ("mlx-arange-native-opengl", "opengl", "opengl-native-runtime"),
        ("mlx-arange-native-vulkan", "vulkan", "vulkan-native-runtime"),
    }
    assert all(
        adapter["platformRequirements"]["requiredTools"] == []
        for adapter in metadata["adapters"]
    )
    assert {fixture["adapter"] for fixture in metadata["fixtures"]} == {
        "mlx-arange-native-directx",
        "mlx-arange-native-opengl",
        "mlx-arange-native-vulkan",
    }
    assert len(metadata["fixtures"]) == 5


def test_vulkan_runtime_readiness_covers_supported_numeric_variants():
    module = _load_harness()

    fixtures = module._runtime_readiness_fixtures(("vulkan",))

    assert [fixture["id"] for fixture in fixtures] == [
        "mlx-arange-vulkan-runtime-readiness",
        "mlx-arange-vulkan-int32-runtime-readiness",
        "mlx-arange-vulkan-float32-runtime-readiness",
    ]
    assert [fixture["entryPoint"] for fixture in fixtures] == [
        "arangeuint32",
        "arangeint32",
        "arangefloat32",
    ]
    assert [fixture["inputs"][0]["dtype"] for fixture in fixtures] == [
        "uint32",
        "int32",
        "float32",
    ]
    assert fixtures[0]["expectedOutputs"][0]["values"] == [300, 317, 334, 351]
    assert fixtures[1]["expectedOutputs"][0]["values"] == [-3, -1, 1, 3]
    assert fixtures[2]["expectedOutputs"][0]["values"] == [
        1.5,
        1.75,
        2.0,
        2.25,
    ]
    assert module._runtime_fixture_scalar(1.5, default=0) == 1.5


def test_expected_gaps_tracks_current_frontier_and_runtime_fixture_counts():
    module = _load_harness()
    expected_gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    frontier = expected_gaps["frontier_status"]
    assert frontier["sources"] == len(module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES)
    assert frontier["artifacts"] == len(
        module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    ) * len(frontier["targets"])
    assert frontier["status"] == "structurally-validated"
    assert frontier["scope"] == "clean-frontier"
    assert frontier["semantic_readiness_status"] == "not-established"
    assert frontier["blocked_by"] == []
    assert frontier["excluded_blocked_sources"] == [module.MLX_FENCE_SOURCE]
    assert frontier["runtime_integration_included"] is False
    assert frontier["runtime_parity_claimed"] is False

    fence = expected_gaps["fence_contract_status"]
    assert fence["status"] == "blocked-as-expected"
    assert fence["source"] == module.MLX_FENCE_SOURCE
    assert fence["targets"] == list(module.MLX_FENCE_TARGET_CONTRACTS)
    assert fence["artifact_records"] == 3
    assert fence["translated_artifacts"] == 0
    assert fence["failed_artifacts"] == 3
    assert fence["emitted_artifacts"] == 0
    assert fence["requested_contract"] == {
        "memory_flags": ["mem_device"],
        "memory_order": "memory_order_seq_cst",
        "thread_scope": "thread_scope_system",
    }
    assert fence["diagnostics"] == {
        target: {
            "code": contract["diagnosticCode"],
            "missing_capability": contract["missingCapability"],
        }
        for target, contract in module.MLX_FENCE_TARGET_CONTRACTS.items()
    }
    assert fence["blocked_by"] == list(module.FENCE_CONTRACT_TRACKED_ISSUES)
    assert fence["runtime_parity_claimed"] is False

    arg_reduce = expected_gaps["arg_reduce_status"]
    assert arg_reduce == {
        "status": "translated",
        "source": module.MLX_ARG_REDUCE_SOURCE,
        "targets": list(module.FULL_CORPUS_TARGETS),
        "artifact_records": len(module.FULL_CORPUS_TARGETS),
        "translated_artifacts": len(module.FULL_CORPUS_TARGETS),
        "failed_artifacts": 0,
        "template_materialization_status": "materialized",
        "specialization_count": 51,
        "unsupported_specialization_count": 0,
        "native_validation": {
            "directx": "required-on-windows-ci",
            "opengl": "validated",
            "vulkan": "validated",
        },
        "entry_packaging_blocked_by": [
            "https://github.com/CrossGL/crosstl/issues/1523"
        ],
    }

    directx = expected_gaps["directx_toolchain_status"]
    assert directx["dxc_validated_sources"] == list(
        module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
    )
    assert set(directx["directx_toolchain_gaps"]) == set(
        module.MLX_REDUCED_FRONTIER_SOURCES
    ) - set(module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES)
    assert "1519" not in directx["directx_toolchain_gaps"][
        "mlx/backend/metal/kernels/layer_norm.metal"
    ]
    assert "1519" not in directx["directx_toolchain_gaps"][
        module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE
    ]

    pointer_reinterpretation = expected_gaps["pointer_reinterpretation_status"]
    assert pointer_reinterpretation["status"] == "partial"
    assert pointer_reinterpretation["targets"] == list(module.FULL_CORPUS_TARGETS)
    assert pointer_reinterpretation["next_kernel_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1574",
        "https://github.com/CrossGL/crosstl/issues/1544",
    ]
    assert set(pointer_reinterpretation["remaining_pointer_cases_blocked_by"]) == {
        "https://github.com/CrossGL/crosstl/issues/1546"
    }

    callback = expected_gaps["captured_callback_status"]
    assert callback["status"] == "partial"
    assert callback["targets"] == list(module.FULL_CORPUS_TARGETS)
    assert callback["remaining_callback_helpers_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1554"
    ]
    assert callback["next_kernel_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1574",
    ]

    compile_time_loop = expected_gaps["compile_time_loop_status"]
    assert compile_time_loop["status"] == "partial"
    assert compile_time_loop["targets"] == list(module.FULL_CORPUS_TARGETS)
    assert "verified integral_constant" in compile_time_loop["supported_contract"]
    assert compile_time_loop["native_validation"] == {
        "directx": "generated-reduced-fixture-dxc-not-run",
        "opengl": "blocked-by-tracked-issue",
        "vulkan": "validated-reduced-fixture",
    }
    assert compile_time_loop["next_kernel_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1574",
    ]

    template_alias = expected_gaps["template_alias_status"]
    assert template_alias["status"] == "materialized"
    assert template_alias["target"] == "vulkan"
    assert template_alias["struct_owned_alias_template_supported_contract"] == (
        "concrete non-variadic struct-owned alias templates with declaring-owner "
        "retention, default argument substitution, dependent owner integral "
        "constants, cross-owner alias chains, and namespace-qualified or nested "
        "owner disambiguation"
    )
    assert template_alias["struct_owned_alias_template_tracked_by"] == (
        "https://github.com/CrossGL/crosstl/issues/1490"
    )
    assert template_alias["plain_helper_supported_contract"] == (
        "call-site deduction through unnamed parameters, empty braced type values, "
        "and proven lexical integral constants"
    )
    assert template_alias["callback_handoff_supported_contract"] == (
        "verified dispatch_bool helpers with only reachable lambda calls defer to "
        "structured frontend callback lowering"
    )
    assert template_alias["high_budget_report"] == {
        "unsupported_before": 111,
        "unsupported_after": 0,
        "residual_int_alias_uses": 0,
        "artifact_status": "blocked-after-materialization",
    }
    assert template_alias["remaining_helpers"] == []
    assert template_alias["resolved_helpers"] == [
        "dispatch_bool<F>",
        "const_for_loop<start, stop, step, F>",
        "tile_matmad_nax<CTile, ATile, BTile, transpose_a, transpose_b>",
    ]
    assert template_alias["resolved_value_arguments"] == [
        "BK_padded",
        "BN_padded",
    ]
    assert template_alias["wide_vector_aggregate_lowering"] == {
        "status": "validated-reduced-fixture",
        "source_types": [
            "vec<float,8>",
            "vec<float16_t,8>",
            "vec<bfloat16_t,8>",
        ],
        "representation": (
            "fixed aggregate wrapper with explicit lane storage and element-wise "
            "helpers"
        ),
        "native_validation": {
            "directx": "validated-if-toolchain-available",
            "opengl": "validated-if-toolchain-available",
            "vulkan": "validated-if-toolchain-available",
        },
        "tracked_by": "https://github.com/CrossGL/crosstl/issues/1569",
    }
    assert template_alias["post_materialization_translation"] == {
        "status": "blocked-by-tracked-issue",
        "diagnostic_code": "project.translate.unsupported-feature",
        "feature": "empty initializer on unresolved dependent static call",
        "first_unsupported_expression": "metal::bool_constant<false>{}",
        "missing_capability": "spirv.empty_initializer_type_inference",
        "issue": "https://github.com/CrossGL/crosstl/issues/1574",
        "artifact_status": "failed",
    }
    assert template_alias["next_kernel_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1574",
    ]
    assert template_alias["semantic_readiness_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1557",
        "https://github.com/CrossGL/crosstl/issues/1574",
    ]

    struct_scoped_cast_alias = expected_gaps["struct_scoped_cast_alias_status"]
    assert struct_scoped_cast_alias["status"] == "partial"
    assert struct_scoped_cast_alias["targets"] == list(module.FULL_CORPUS_TARGETS)
    assert struct_scoped_cast_alias["concrete_specializations"] == ["float", "int"]
    assert struct_scoped_cast_alias["qualifier_transport"] == "retained-in-metal-ast"
    assert struct_scoped_cast_alias["strict_crossgl_function_body_parse"] == (
        "passing-reduced-fixture"
    )
    assert struct_scoped_cast_alias["native_validation"] == {
        "directx": "required-on-windows-ci",
        "opengl": "validated-reduced-fixture",
        "vulkan": "validated-reduced-fixture",
    }
    assert struct_scoped_cast_alias["high_budget_report"] == {
        "specialization_count": 722,
        "unsupported_specialization_count": 0,
        "resolved_function": "NAXTile_float_2_2__elems",
        "prior_diagnostic_code": "project.translate.crossgl-function-body-parse-failed",
        "next_diagnostic_code": "project.translate.unsupported-feature",
        "artifact_status": "failed",
    }
    assert struct_scoped_cast_alias["remaining_contract_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1566",
    ]
    assert struct_scoped_cast_alias["next_kernel_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1574",
    ]

    function_local_alias = expected_gaps["function_local_alias_status"]
    assert function_local_alias["status"] == "partial"
    assert function_local_alias["targets"] == list(module.FULL_CORPUS_TARGETS)
    assert function_local_alias["entry_count"] == 36
    assert function_local_alias["resolved_use_counts"] == {
        "declaration_types": 336,
        "casts": 72,
        "static_members": 36,
    }
    assert function_local_alias["native_validation"] == {
        "directx": "translated-dxc-blocked-by-existing-issues",
        "opengl": "validated",
        "vulkan": "validated",
    }
    assert function_local_alias["vulkan_project_warning_count"] == 0
    assert function_local_alias["single_file_vulkan_unreachable_warning_count"] == 5
    assert function_local_alias["single_file_vulkan_warnings_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1568",
    ]
    assert function_local_alias["remaining_alias_shapes_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1567",
    ]

    generic_member_call = expected_gaps["generic_member_call_status"]
    assert generic_member_call["status"] == "validated-reduced-fixture"
    assert generic_member_call["sources"] == [
        "mlx/backend/metal/kernels/fp_quantized.metal",
        "mlx/backend/metal/kernels/quantized_nax.metal",
    ]
    assert generic_member_call["targets"] == list(module.FULL_CORPUS_TARGETS)
    assert generic_member_call["native_validation"] == {
        "directx": "validated-with-glslang-hlsl",
        "opengl": "validated-with-glslang",
        "vulkan": "validated-with-spirv-tools",
    }
    assert generic_member_call["pinned_vulkan_replay"] == {
        "mlx/backend/metal/kernels/fp_quantized.metal": {
            "status": "blocked-by-tracked-issue",
            "diagnostic_code": "project.translate.metal-struct-method",
            "missing_capability": "struct.template-method",
            "first_unresolved_expression": "frag_at(i, j)",
            "issue": "https://github.com/CrossGL/crosstl/issues/1557",
            "artifact_status": "failed",
        },
        "mlx/backend/metal/kernels/quantized_nax.metal": {
            "status": "blocked-by-tracked-issue",
            "diagnostic_code": "project.translate.unsupported-feature",
            "missing_capability": "spirv.empty_initializer_type_inference",
            "first_unresolved_call": "mma",
            "issue": "https://github.com/CrossGL/crosstl/issues/1574",
            "artifact_status": "failed",
        },
    }
    assert generic_member_call["runtime_integration_included"] is False

    gemv = expected_gaps["vulkan_gemv_toolchain_status"]
    assert gemv == {
        "status": "passed",
        "source": module.MLX_GEMV_SOURCE,
        "target": "vulkan",
        "specialization_count": module.GEMV_EXPECTED_SPECIALIZATION_COUNT,
        "entry_point_count": module.GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "structural_validation_status": "validated",
        "available_validators": ["spirv-as", "spirv-val"],
        "target_environment": "vulkan1.1",
        "semantic_readiness_status": "no-known-codegen-fallbacks",
        "semantic_warning_count": 0,
        "semantic_warnings_by_issue": {},
        "semantic_warning_descriptions_by_issue": {},
        "semantic_blockers": [],
        "report_warning_transport_tracked_by": None,
        "runtime_integration_included": False,
    }

    runtime = expected_gaps["runtime_readiness_status"]
    assert runtime["fixture_count"] == len(
        module._runtime_readiness_fixtures(("directx", "opengl", "vulkan"))
    )

    full_corpus = expected_gaps["full_corpus_scout"]
    assert full_corpus["artifacts"] == module.FULL_CORPUS_EXPECTED_ARTIFACT_COUNT
    assert full_corpus["expected_translated_artifacts"] == (
        module.FULL_CORPUS_EXPECTED_TRANSLATED_ARTIFACT_COUNT
    )
    assert full_corpus["expected_fence_failures"] == (
        module.FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
    )
    assert full_corpus["expected_fence_diagnostics"] == {
        contract["diagnosticCode"]: 1
        for contract in module.MLX_FENCE_TARGET_CONTRACTS.values()
    }
    assert "https://github.com/CrossGL/crosstl/issues/1537" in full_corpus[
        "semantic_blocked_by"
    ]
    assert full_corpus["runtime_integration_included"] is False
    assert full_corpus["runtime_parity_claimed"] is False
    assert full_corpus["last_completed"]["snapshot_scope"] == (
        "pre-canonical-fence-contract"
    )


def test_arange_reference_runtime_resolves_fixture_resource_aliases():
    module = _load_harness()
    runtime = module.MlxArangeReferenceRuntime("vulkan")
    state = SimpleNamespace(
        resource_values={
            "startUniform": [-3],
            "stepUniform": [2],
            "out_": None,
        },
        plan=SimpleNamespace(
            resource_bindings=[
                SimpleNamespace(
                    source="input",
                    value=SimpleNamespace(name="start", values=[-3]),
                ),
                SimpleNamespace(
                    source="input",
                    value=SimpleNamespace(name="step", values=[2]),
                ),
                SimpleNamespace(
                    source="expectedOutput",
                    value=SimpleNamespace(name="out", values=None),
                ),
            ]
        ),
    )

    prepared = runtime.prepare_buffers(state)

    assert prepared["start"] == [-3]
    assert prepared["step"] == [2]
    assert "out" not in prepared


def test_runtime_report_target_selection_returns_every_fixture_result():
    module = _load_harness()
    runtime_report = {
        "results": [
            {"status": "passed", "artifact": {"target": "vulkan"}},
            {
                "status": "passed",
                "fixture": {"selector": {"target": "vulkan"}},
            },
            {"status": "unavailable", "artifact": {"target": "directx"}},
        ]
    }

    assert len(module._runtime_report_results_for_target(runtime_report, "vulkan")) == 2
    assert (
        len(module._runtime_report_results_for_target(runtime_report, "directx")) == 1
    )


def test_required_vulkan_runtime_rejects_any_failed_numeric_variant():
    module = _load_harness()
    runtime_report = {
        "results": [
            {
                "status": status,
                "fixture": {"selector": {"target": "vulkan"}},
            }
            for status in ("passed", "passed", "runtime-failed")
        ]
    }

    with pytest.raises(
        module.PortingCheckError,
        match="every MLX arange fixture",
    ):
        module._require_vulkan_native_runtime_results(runtime_report)


@pytest.mark.parametrize(
    ("target", "entry_point", "dtype"),
    (
        ("directx", "CSMain", "uint8"),
        ("opengl", "main", "uint8"),
        ("vulkan", "arangeuint32", "uint32"),
    ),
)
def test_runtime_readiness_selects_entry_point_independently(
    target, entry_point, dtype
):
    module = _load_harness()

    fixture = module._runtime_readiness_fixture(target)

    assert fixture["selector"] == {
        "source": module.MLX_ARANGE_SOURCE,
        "target": target,
    }
    assert fixture["entryPoint"] == entry_point
    assert fixture["inputs"][0]["dtype"] == dtype
    assert fixture["runtimeAdapter"]["dispatch"] == {
        "globalSize": [4, 1, 1],
    }
    assert "https://github.com/CrossGL/crosstl/issues/1394" not in (
        module.RUNTIME_READINESS_TRACKED_ISSUES
    )
    assert "https://github.com/CrossGL/crosstl/issues/1394" in (
        module.RESOLVED_FRONTIER_ISSUES
    )


def test_static_constant_materialization_issue_is_active():
    module = _load_harness()
    issue = "https://github.com/CrossGL/crosstl/issues/1491"

    assert issue not in module.RESOLVED_FRONTIER_ISSUES
    assert issue in module.FULL_CORPUS_TRACKED_ISSUES


def test_arg_reduce_advances_into_clean_toolchain_frontiers():
    module = _load_harness()

    assert module.MLX_ARG_REDUCE_SOURCE in (module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES)
    assert module.MLX_ARG_REDUCE_SOURCE in (
        module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES
    )
    assert module.MLX_ARG_REDUCE_SOURCE in (
        module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
    )
    assert module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES == (
        module.MLX_ARG_REDUCE_SOURCE,
        "mlx/backend/metal/kernels/logsumexp.metal",
        "mlx/backend/metal/kernels/softmax.metal",
    )
    assert module.MLX_REDUCED_FRONTIER_SOURCES == tuple(
        sorted(
            (
                *module.MLX_CLEAN_REDUCED_FRONTIER_SOURCES,
                *module.MLX_BLOCKED_REDUCED_FRONTIER_SOURCES,
            )
        )
    )
    assert "https://github.com/CrossGL/crosstl/issues/1551" in (
        module.RESOLVED_FRONTIER_ISSUES
    )


def test_fence_is_blocked_outside_clean_and_directx_toolchain_frontiers():
    module = _load_harness()

    assert module.MLX_FENCE_SOURCE not in module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    assert module.MLX_FENCE_SOURCE not in module.MLX_CLEAN_REDUCED_FRONTIER_SOURCES
    assert module.MLX_FENCE_SOURCE not in module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
    assert module.MLX_BLOCKED_REDUCED_FRONTIER_SOURCES == (module.MLX_FENCE_SOURCE,)
    assert module.MLX_FENCE_SOURCE in module.MLX_REDUCED_FRONTIER_SOURCES
    assert module.FENCE_CONTRACT_TRACKED_ISSUES == (
        "https://github.com/CrossGL/crosstl/issues/1537",
    )


def test_float_subgroup_xor_issue_is_resolved_for_mlx_gemv():
    module = _load_harness()
    issue = "https://github.com/CrossGL/crosstl/issues/1498"

    assert module.VULKAN_GEMV_SEMANTIC_TRACKED_ISSUES == ()
    assert issue in module.RESOLVED_FRONTIER_ISSUES
    assert issue not in module.FULL_CORPUS_TRACKED_ISSUES


def test_nested_return_inlining_issue_is_resolved_for_mlx_gemv():
    module = _load_harness()
    issue = "https://github.com/CrossGL/crosstl/issues/1561"

    assert issue in module.RESOLVED_FRONTIER_ISSUES
    assert issue not in module.FULL_CORPUS_TRANSLATION_TRACKED_ISSUES
    assert issue not in module.FULL_CORPUS_TRACKED_ISSUES


def test_generic_member_call_issue_is_resolved_for_pinned_quantized_kernels():
    module = _load_harness()
    issue = "https://github.com/CrossGL/crosstl/issues/1555"
    expected_gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    assert issue in module.RESOLVED_FRONTIER_ISSUES
    assert issue not in module.FULL_CORPUS_TRANSLATION_TRACKED_ISSUES
    assert issue not in module.FULL_CORPUS_TRACKED_ISSUES
    assert issue in expected_gaps["resolved_issues"]
    assert issue not in expected_gaps["tracked_issues"]
    assert issue not in expected_gaps["full_corpus_scout"]["translation_blocked_by"]


def test_empty_initializer_issue_is_resolved_for_pinned_quantized_nax():
    module = _load_harness()
    issue = "https://github.com/CrossGL/crosstl/issues/1573"
    next_issue = "https://github.com/CrossGL/crosstl/issues/1574"
    expected_gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    assert issue in module.RESOLVED_FRONTIER_ISSUES
    assert issue not in module.FULL_CORPUS_TRACKED_ISSUES
    assert issue in expected_gaps["resolved_issues"]
    assert issue not in expected_gaps["tracked_issues"]
    replay = expected_gaps["generic_member_call_status"]["pinned_vulkan_replay"]
    assert replay["mlx/backend/metal/kernels/quantized_nax.metal"]["issue"] == (
        next_issue
    )


def test_scaled_dot_product_attention_tracks_opengl_function_constant_blocker():
    module = _load_harness()
    source = module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE
    resolved_issue = "https://github.com/CrossGL/crosstl/issues/1535"
    static_constant_issue = "https://github.com/CrossGL/crosstl/issues/1491"
    function_constant_issue = "https://github.com/CrossGL/crosstl/issues/1538"

    assert source in module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    assert source not in (
        module.MLX_ARANGE_SOURCE,
        module.MLX_ARG_REDUCE_SOURCE,
        module.MLX_GEMV_SOURCE,
    )
    assert resolved_issue in module.RESOLVED_FRONTIER_ISSUES
    assert resolved_issue not in module.FULL_CORPUS_TRACKED_ISSUES
    assert module.OPENGL_SCALED_DOT_PRODUCT_ATTENTION_TRACKED_ISSUES == (
        function_constant_issue,
    )
    assert static_constant_issue in module.FULL_CORPUS_TRACKED_ISSUES
    assert function_constant_issue in module.FULL_CORPUS_TRACKED_ISSUES


def test_scaled_attention_local_alias_evidence_requires_complete_entries(tmp_path):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    directx_path = Path("out/directx/scaled_dot_product_attention.hlsl")
    vulkan_path = Path("out/vulkan/scaled_dot_product_attention.spvasm")
    for path in (directx_path, vulkan_path):
        (mlx_root / path).parent.mkdir(parents=True, exist_ok=True)

    directx = "\n".join(
        f"[numthreads(1, 1, 1)]\nvoid CSMain_{index}() {{}}" for index in range(36)
    )
    vulkan = "\n".join(
        f'  OpEntryPoint GLCompute %{index + 1} "sdpa_{index}"' for index in range(36)
    )
    (mlx_root / directx_path).write_text(directx, encoding="utf-8")
    (mlx_root / vulkan_path).write_text(vulkan, encoding="utf-8")
    payload = {
        "artifacts": [
            {
                "source": module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE,
                "target": target,
                "path": path.as_posix(),
                "status": "translated",
            }
            for target, path in (
                ("directx", directx_path),
                ("vulkan", vulkan_path),
            )
        ]
    }

    evidence = module._scaled_attention_local_alias_evidence(mlx_root, payload)

    assert evidence["entryCountByTarget"] == {"directx": 36, "vulkan": 36}
    assert evidence["resolvedDeclarationTypeCount"] == 336
    assert evidence["resolvedCastCount"] == 72
    assert evidence["resolvedStaticMemberCount"] == 36
    assert evidence["vulkanProjectWarningCount"] == 0


def _fence_contract_report(module, mlx_root, work_dir):
    extensions = {"directx": ".hlsl", "opengl": ".glsl", "vulkan": ".spvasm"}
    diagnostics = []
    artifacts = []
    for target, contract in module.MLX_FENCE_TARGET_CONTRACTS.items():
        message = module._atomic_fence_expected_message(contract)
        artifact_path = (
            work_dir
            / "out-fence-contract"
            / target
            / Path(module.MLX_FENCE_SOURCE).with_suffix(extensions[target])
        ).relative_to(mlx_root)
        diagnostics.append(
            {
                "severity": "error",
                "code": contract["diagnosticCode"],
                "message": message,
                "location": {"file": module.MLX_FENCE_SOURCE},
                "target": target,
                "sourceBackend": "metal",
                "missingCapabilities": [contract["missingCapability"]],
            }
        )
        artifacts.append(
            {
                "source": module.MLX_FENCE_SOURCE,
                "sourceBackend": "metal",
                "target": target,
                "path": artifact_path.as_posix(),
                "status": "failed",
                "error": message,
            }
        )
    target_count = len(module.MLX_FENCE_TARGET_CONTRACTS)
    return {
        "summary": {
            "unitCount": 1,
            "artifactCount": target_count,
            "translatedCount": 0,
            "failedCount": target_count,
            "diagnosticCounts": {"error": target_count, "note": 0, "warning": 0},
            "diagnosticsByCode": {
                contract["diagnosticCode"]: 1
                for contract in module.MLX_FENCE_TARGET_CONTRACTS.values()
            },
            "missingCapabilityCounts": {
                contract["missingCapability"]: 1
                for contract in module.MLX_FENCE_TARGET_CONTRACTS.values()
            },
            "artifactsByTarget": {
                target: {
                    "artifactCount": 1,
                    "translatedCount": 0,
                    "failedCount": 1,
                }
                for target in module.MLX_FENCE_TARGET_CONTRACTS
            },
        },
        "diagnostics": diagnostics,
        "artifacts": artifacts,
    }


def _prepare_fence_contract_check(module, tmp_path, monkeypatch, mutate=None):
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)
    commands = []

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        payload = _fence_contract_report(module, mlx_root, work_dir)
        if mutate is not None:
            mutate(payload, mlx_root)
        (report_dir / "fence-contract.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        commands.append((name, list(command), check))
        return module.CommandResult(name, list(command), 1, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    return mlx_root, work_dir, config_dir, report_dir, log_dir, commands


def test_atomic_fence_contract_records_exact_target_failures(tmp_path, monkeypatch):
    module = _load_harness()
    (
        mlx_root,
        work_dir,
        config_dir,
        report_dir,
        log_dir,
        commands,
    ) = _prepare_fence_contract_check(module, tmp_path, monkeypatch)

    result = module._check_atomic_fence_contract(
        mlx_root,
        work_dir,
        config_dir,
        report_dir,
        log_dir,
        "python",
    )

    assert result["status"] == "blocked-as-expected"
    assert result["source"] == module.MLX_FENCE_SOURCE
    assert result["targets"] == ["directx", "opengl", "vulkan"]
    assert result["artifactRecordCount"] == 3
    assert result["failedArtifactCount"] == 3
    assert result["emittedArtifactCount"] == 0
    assert result["requestedContract"] == {
        "memoryFlags": ["mem_device"],
        "memoryOrder": "memory_order_seq_cst",
        "threadScope": "thread_scope_system",
    }
    assert result["semanticTrackedIssues"] == [
        "https://github.com/CrossGL/crosstl/issues/1537"
    ]
    assert result["runtimeParityClaimed"] is False
    assert set(result["targetContracts"]) == {"directx", "opengl", "vulkan"}
    for target, contract in module.MLX_FENCE_TARGET_CONTRACTS.items():
        target_result = result["targetContracts"][target]
        assert target_result["diagnosticCode"] == contract["diagnosticCode"]
        assert target_result["missingCapability"] == contract["missingCapability"]
        assert target_result["artifactStatus"] == "failed"
        assert target_result["artifactEmitted"] is False

    config = (config_dir / "fence-contract.toml").read_text(encoding="utf-8")
    assert f'include = ["{module.MLX_FENCE_SOURCE}"]' in config
    assert 'targets = ["directx", "opengl", "vulkan"]' in config
    assert commands == [
        (
            "translate-fence-contract",
            [
                "python",
                "-m",
                "crosstl",
                "translate-project",
                str(mlx_root),
                "--config",
                str(config_dir / "fence-contract.toml"),
                "--report",
                str(report_dir / "fence-contract.json"),
            ],
            False,
        )
    ]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda payload, _root: payload["diagnostics"][0].__setitem__(
                "code", "project.translate.failed"
            ),
            "structured diagnostic changed",
        ),
        (
            lambda payload, _root: payload["diagnostics"][1].__setitem__(
                "missingCapabilities", ["batch.translation"]
            ),
            "structured diagnostic changed",
        ),
        (
            lambda payload, _root: payload["diagnostics"][2].__setitem__(
                "message",
                payload["diagnostics"][2]["message"].replace(
                    "memory_order_seq_cst", "memory_order_acq_rel"
                ),
            ),
            "structured diagnostic changed",
        ),
    ],
)
def test_atomic_fence_contract_rejects_diagnostic_drift(
    tmp_path, monkeypatch, mutation, error
):
    module = _load_harness()
    check = _prepare_fence_contract_check(module, tmp_path, monkeypatch, mutation)

    with pytest.raises(module.PortingCheckError, match=error):
        module._check_atomic_fence_contract(*check[:5], "python")


def test_atomic_fence_contract_rejects_emitted_target_artifact(
    tmp_path, monkeypatch
):
    module = _load_harness()

    def emit_directx_artifact(payload, mlx_root):
        artifact_path = mlx_root / payload["artifacts"][0]["path"]
        artifact_path.parent.mkdir(parents=True)
        artifact_path.write_text("unexpected", encoding="utf-8")

    check = _prepare_fence_contract_check(
        module,
        tmp_path,
        monkeypatch,
        emit_directx_artifact,
    )

    with pytest.raises(module.PortingCheckError, match="unexpectedly emitted"):
        module._check_atomic_fence_contract(*check[:5], "python")


def test_opengl_codegen_fixes_advance_native_validation_frontier():
    module = _load_harness()
    resolved_issues = {
        "https://github.com/CrossGL/crosstl/issues/1535",
        "https://github.com/CrossGL/crosstl/issues/1500",
        "https://github.com/CrossGL/crosstl/issues/1502",
        "https://github.com/CrossGL/crosstl/issues/1503",
        "https://github.com/CrossGL/crosstl/issues/1504",
        "https://github.com/CrossGL/crosstl/issues/1489",
    }

    assert resolved_issues.issubset(module.RESOLVED_FRONTIER_ISSUES)
    assert resolved_issues.isdisjoint(module.FULL_CORPUS_TRACKED_ISSUES)
    assert resolved_issues.isdisjoint(module.OPENGL_ARANGE_VALIDATION_TRACKED_ISSUES)
    assert module.OPENGL_ARANGE_VALIDATION_TRACKED_ISSUES == ()


def _prepare_arange_opengl_check(module, tmp_path, generated):
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    generated_path = work_dir / "out" / "opengl" / "arange.glsl"
    for path in (config_dir, report_dir, log_dir, generated_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(generated, encoding="utf-8")
    report = {
        "summary": {"translatedCount": 1, "failedCount": 0},
        "artifacts": [
            {
                "source": module.MLX_ARANGE_SOURCE,
                "target": "opengl",
                "path": generated_path.relative_to(mlx_root).as_posix(),
                "status": "translated",
            }
        ],
    }
    (report_dir / "arange-opengl.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return mlx_root, work_dir, config_dir, report_dir, log_dir


def _arange_opengl_frontier_source():
    return """
    float log1p_float(float value) { return value; }
    float log1p_bfloat16(float value) { return value; }
    uint crossglWaveShuffleAndFillUp(uint value, uint fill, uint delta) {
        uint shuffled = subgroupShuffleUp(value, delta);
        return gl_SubgroupInvocationID >= delta ? shuffled : fill;
    }
    struct complex64_t { float real; float imag; };
    bool simd_shuffle_down(bool data, uint delta) {
        return (subgroupShuffleDown(uint(data), delta) != 0u);
    }
    bool simd_shuffle_up(bool data, uint delta) {
        return (subgroupShuffleUp(uint(data), delta) != 0u);
    }
    bool simd_shuffle_and_fill_up(bool data, bool filling, uint delta) {
        return (
            crossglWaveShuffleAndFillUp(uint(data), uint(filling), delta) != 0u
        );
    }
    bool simd_shuffle(bool data, uint lane) {
        return (subgroupShuffle(uint(data), lane) != 0u);
    }
    void signed_arange_coercions(uint index) {
        arangeint8_out[index] = bitfieldExtract(
            int((uint(arangeint8_start_Args_start) +
                 (index * uint(arangeint8_step_Args_step)))),
            0,
            8
        );
        arangeint16_out[index] = bitfieldExtract(
            int((uint(arangeint16_start_Args_start) +
                 (index * uint(arangeint16_step_Args_step)))),
            0,
            16
        );
        arangeint32_out[index] = int(
            (uint(arangeint32_start_Args_start) +
             (index * uint(arangeint32_step_Args_step)))
        );
        arangeint64_out[index] = (
            arangeint64_start_Args_start +
            (int64_t(index) * arangeint64_step_Args_step)
        );
    }
    complex64_t probe(float x, float theta, float r, float z0) {
        log1p_float(r);
        if (x > 0.0) { return complex64_t(x, theta); }
        if (r > 0.0) {
            return complex64_t((0.5 * log1p_float(r)), theta);
        }
        return complex64_t(log(z0), theta);
    }
    """


def _prepare_opengl_frontier_check(module, tmp_path):
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for path in (config_dir, report_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for source in module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES:
        generated_path = work_dir / "out" / "opengl" / f"{Path(source).stem}.glsl"
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        generated_path.write_text(
            "#version 450 core\nvoid main() {}\n",
            encoding="utf-8",
        )
        artifacts.append(
            {
                "source": source,
                "target": "opengl",
                "path": generated_path.relative_to(mlx_root).as_posix(),
                "status": "translated",
            }
        )
    frontier_count = len(module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES)
    report = {
        "summary": {
            "unitCount": frontier_count,
            "artifactCount": frontier_count,
            "translatedCount": frontier_count,
            "failedCount": 0,
        },
        "artifacts": artifacts,
    }
    (report_dir / "opengl-frontier.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return mlx_root, work_dir, config_dir, report_dir, log_dir


def test_opengl_frontier_required_toolchain_compiles_and_validates_artifacts(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_opengl_frontier_check(module, tmp_path)
    commands = []

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        commands.append((name, list(command)))
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        if name.startswith("validate-") and name.endswith("-opengl"):
            output_path = Path(command[command.index("-o") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x03\x02\x23\x07")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    tools = {
        "glslangValidator": "/tools/glslangValidator",
        "spirv-val": "/tools/spirv-val",
    }
    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", tools.get)

    result = module._check_opengl_frontier(
        *paths,
        "python",
        require_toolchain=True,
    )

    assert result["status"] == "passed"
    assert result["sources"] == list(module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES)
    assert result["artifactCount"] == len(module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES)
    assert result["toolchainRequired"] is True
    assert result["nativeValidationStatus"] == "validated"
    assert result["spirvValidator"] == "spirv-val"
    assert result["runtimeIntegrationIncluded"] is False
    assert [name for name, _command in commands] == [
        "translate-opengl-frontier",
        "validate-arg-reduce-opengl",
        "validate-arg-reduce-opengl-spirv",
        "validate-logsumexp-opengl",
        "validate-logsumexp-opengl-spirv",
        "validate-softmax-opengl",
        "validate-softmax-opengl-spirv",
    ]
    assert commands[1][1][:5] == [
        "/tools/glslangValidator",
        "--target-env",
        "opengl",
        "--target-env",
        "spirv1.3",
    ]
    assert commands[2][1][:3] == [
        "/tools/spirv-val",
        "--target-env",
        "spv1.3",
    ]
    assert set(result["nativeValidationOutputs"]) == set(
        module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES
    )
    config = (paths[2] / "opengl-frontier.toml").read_text(encoding="utf-8")
    for source in module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES:
        assert source in config
    assert module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE not in config
    assert "mlx/backend/metal/kernels/rms_norm.metal" not in config
    assert 'targets = ["opengl"]' in config


def test_opengl_frontier_skips_toolchain_when_not_required(tmp_path, monkeypatch):
    module = _load_harness()
    paths = _prepare_opengl_frontier_check(module, tmp_path)
    commands = []

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        commands.append(name)
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: pytest.fail(f"tool lookup should not run for {name}"),
    )

    result = module._check_opengl_frontier(
        *paths,
        "python",
        require_toolchain=False,
    )

    assert commands == ["translate-opengl-frontier"]
    assert result["toolchainRequired"] is False
    assert result["nativeValidationStatus"] == "not-required"
    assert result["nativeValidationOutputs"] == {}


def test_opengl_frontier_requires_every_clean_artifact(tmp_path, monkeypatch):
    module = _load_harness()
    paths = _prepare_opengl_frontier_check(module, tmp_path)
    report_path = paths[3] / "opengl-frontier.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["summary"]["artifactCount"] -= 1
    report["artifacts"].pop()
    report_path.write_text(json.dumps(report), encoding="utf-8")

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    with pytest.raises(
        module.PortingCheckError,
        match="every clean translated artifact",
    ):
        module._check_opengl_frontier(
            *paths,
            "python",
            require_toolchain=False,
        )


def _prepare_gemv_opengl_check(module, tmp_path, generated):
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    generated_path = work_dir / "out" / "opengl" / "gemv.glsl"
    for path in (config_dir, report_dir, log_dir, generated_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(generated, encoding="utf-8")
    report = {
        "summary": {"translatedCount": 1, "failedCount": 0},
        "artifacts": [
            {
                "source": module.MLX_GEMV_SOURCE,
                "target": "opengl",
                "path": generated_path.relative_to(mlx_root).as_posix(),
                "status": "translated",
                "templateMaterialization": {
                    "specializationCount": module.GEMV_EXPECTED_SPECIALIZATION_COUNT
                },
            }
        ],
    }
    (report_dir / "gemv-opengl.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return mlx_root, work_dir, config_dir, report_dir, log_dir


def _gemv_opengl_frontier_source():
    helpers = "\n".join(
        f"void compute_main_{index}() {{}}" for index in range(2, 1 + 224)
    )
    return f"#version 450 core\nvoid main() {{}}\n{helpers}\n"


def test_gemv_opengl_toolchain_check_compiles_and_validates_full_artifact(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_gemv_opengl_check(
        module,
        tmp_path,
        _gemv_opengl_frontier_source(),
    )
    commands = []

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        commands.append((name, list(command)))
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text(
            (
                'WARNING: identifiers containing consecutive underscores ("__") '
                "are reserved\n"
                if name == "validate-gemv-opengl"
                else ""
            ),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        if name == "validate-gemv-opengl":
            output_path = Path(command[command.index("-o") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x03\x02\x23\x07")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    tools = {
        "glslangValidator": "/tools/glslangValidator",
        "spirv-val": "/tools/spirv-val",
    }
    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", tools.get)

    result = module._check_gemv_opengl_toolchain(*paths, "python")

    assert result["status"] == "passed"
    assert result["specializationCount"] == 225
    assert result["entryPointCount"] == 224
    assert result["nativeValidationStatus"] == "validated"
    assert result["nativeWarningCount"] == 1
    assert result["nativeWarningsTrackedBy"].endswith("/1513")
    assert result["runtimeIntegrationIncluded"] is False
    assert [name for name, _command in commands] == [
        "translate-gemv-opengl",
        "validate-gemv-opengl",
        "validate-gemv-opengl-spirv",
    ]
    assert commands[1][1][:5] == [
        "/tools/glslangValidator",
        "--target-env",
        "opengl",
        "--target-env",
        "spirv1.3",
    ]
    assert commands[2][1][:3] == [
        "/tools/spirv-val",
        "--target-env",
        "spv1.3",
    ]
    config = (paths[2] / "gemv-opengl.toml").read_text(encoding="utf-8")
    assert "max_template_specializations = 4096" in config
    assert "max_template_materialization_work = 2097152" in config


def test_gemv_opengl_toolchain_check_rejects_materialization_residue(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_gemv_opengl_check(
        module,
        tmp_path,
        _gemv_opengl_frontier_source() + "OffsetT unresolved_value;\n",
    )
    commands = []

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        commands.append(name)
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/tools/{name}")

    with pytest.raises(
        module.PortingCheckError,
        match="retained unresolved materialization text",
    ):
        module._check_gemv_opengl_toolchain(*paths, "python")

    assert commands == ["translate-gemv-opengl"]


def test_gemv_opengl_toolchain_check_rejects_untracked_native_warning(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_gemv_opengl_check(
        module,
        tmp_path,
        _gemv_opengl_frontier_source(),
    )

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text(
            (
                "WARNING: subgroup behavior changed\n"
                if name == "validate-gemv-opengl"
                else ""
            ),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        if name == "validate-gemv-opengl":
            output_path = Path(command[command.index("-o") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x03\x02\x23\x07")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/tools/{name}")

    with pytest.raises(
        module.PortingCheckError,
        match="emitted an untracked warning",
    ):
        module._check_gemv_opengl_toolchain(*paths, "python")


def _prepare_gemv_vulkan_check(module, tmp_path, generated):
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    generated_path = work_dir / "out" / "vulkan" / "gemv.spvasm"
    for path in (config_dir, report_dir, log_dir, generated_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(generated, encoding="utf-8")
    report = {
        "summary": {
            "translatedCount": 1,
            "failedCount": 0,
            "diagnosticCounts": {"warning": 0},
        },
        "artifacts": [
            {
                "source": module.MLX_GEMV_SOURCE,
                "target": "vulkan",
                "path": generated_path.relative_to(mlx_root).as_posix(),
                "status": "translated",
                "templateMaterialization": {
                    "specializationCount": module.GEMV_EXPECTED_SPECIALIZATION_COUNT
                },
            }
        ],
    }
    (report_dir / "gemv-vulkan.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return mlx_root, work_dir, config_dir, report_dir, log_dir


def _gemv_vulkan_frontier_source():
    entry_points = "\n".join(
        f'  OpEntryPoint GLCompute %main_{index} "compute_main_{index}"'
        for index in range(1, 225)
    )
    return entry_points + "\n"


def _stub_gemv_vulkan_toolchain(module, monkeypatch, commands):
    def fake_run_command(name, command, *, log_dir, **_kwargs):
        commands.append((name, list(command)))
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        if name == "assemble-gemv-vulkan":
            output_path = Path(command[command.index("-o") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x03\x02\x23\x07")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/tools/{name}")


def test_gemv_vulkan_toolchain_check_structurally_validates_full_artifact(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_gemv_vulkan_check(
        module,
        tmp_path,
        _gemv_vulkan_frontier_source(),
    )
    commands = []
    _stub_gemv_vulkan_toolchain(module, monkeypatch, commands)

    result = module._check_gemv_vulkan_toolchain(*paths, "python")

    assert result["status"] == "passed"
    assert result["specializationCount"] == 225
    assert result["entryPointCount"] == 224
    assert result["structuralValidationStatus"] == "validated"
    assert result["semanticReadinessStatus"] == "no-known-codegen-fallbacks"
    assert result["semanticWarningCount"] == 0
    assert result["semanticWarningsByIssue"] == {}
    assert result["semanticBlockers"] == []
    assert result["reportWarningCount"] == 0
    assert result["reportWarningTransportTrackedBy"] is None
    assert result["runtimeIntegrationIncluded"] is False
    assert [name for name, _command in commands] == [
        "translate-gemv-vulkan",
        "assemble-gemv-vulkan",
        "validate-gemv-vulkan-spirv",
    ]
    assert commands[1][1][:3] == [
        "/tools/spirv-as",
        "--target-env",
        "vulkan1.1",
    ]
    assert commands[2][1][:3] == [
        "/tools/spirv-val",
        "--target-env",
        "vulkan1.1",
    ]
    config = (paths[2] / "gemv-vulkan.toml").read_text(encoding="utf-8")
    assert "max_template_specializations = 4096" in config
    assert "max_template_materialization_work = 2097152" in config


def test_gemv_vulkan_toolchain_check_rejects_translation_failure(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_gemv_vulkan_check(module, tmp_path, "")
    _mlx_root, _work_dir, _config_dir, report_dir, log_dir = paths
    report = {
        "summary": {
            "translatedCount": 0,
            "failedCount": 1,
            "diagnosticCounts": {"error": 1},
        },
        "artifacts": [
            {
                "source": module.MLX_GEMV_SOURCE,
                "sourceBackend": "metal",
                "target": "vulkan",
                "status": "failed",
            }
        ],
        "diagnostics": [
            {
                "code": "project.translate.unsupported-feature",
                "sourceBackend": "metal",
                "target": "vulkan",
                "missingCapabilities": ["spirv.nested_return_storage_buffer_function"],
                "message": (
                    "SPIR-V pointer-preserving function inlining requires returns "
                    "to be top-level statements; helper 'run' contains a nested "
                    "return"
                ),
            }
        ],
    }
    (report_dir / "gemv-vulkan.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    commands = []

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        commands.append((name, list(command)))
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 1, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/tools/{name}")

    with pytest.raises(
        module.PortingCheckError,
        match="Vulkan GEMV translation failed: .*contains a nested return",
    ):
        module._check_gemv_vulkan_toolchain(*paths, "python")

    assert [name for name, _command in commands] == ["translate-gemv-vulkan"]
    assert not (paths[1] / "validation" / "gemv-vulkan.spv").exists()


def test_gemv_vulkan_toolchain_check_rejects_untracked_warning(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_gemv_vulkan_check(
        module,
        tmp_path,
        _gemv_vulkan_frontier_source() + "; WARNING: subgroup behavior changed\n",
    )
    commands = []
    _stub_gemv_vulkan_toolchain(module, monkeypatch, commands)

    with pytest.raises(
        module.PortingCheckError,
        match="emitted a semantic warning",
    ):
        module._check_gemv_vulkan_toolchain(*paths, "python")

    assert [name for name, _command in commands] == ["translate-gemv-vulkan"]


def test_gemv_vulkan_toolchain_check_rejects_resolved_float_fallback_warning(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    generated = (
        "; WARNING: WaveActiveBitXor requires a compatible arithmetic or "
        "bitwise operand; got float\n" + _gemv_vulkan_frontier_source()
    )
    paths = _prepare_gemv_vulkan_check(module, tmp_path, generated)
    commands = []
    _stub_gemv_vulkan_toolchain(module, monkeypatch, commands)

    with pytest.raises(
        module.PortingCheckError,
        match="emitted a semantic warning",
    ):
        module._check_gemv_vulkan_toolchain(*paths, "python")

    assert [name for name, _command in commands] == ["translate-gemv-vulkan"]


def test_gemv_vulkan_toolchain_check_rejects_materialization_residue(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_gemv_vulkan_check(
        module,
        tmp_path,
        _gemv_vulkan_frontier_source() + 'OpName %residue "PrimitiveType"\n',
    )
    commands = []
    _stub_gemv_vulkan_toolchain(module, monkeypatch, commands)

    with pytest.raises(
        module.PortingCheckError,
        match="retained unresolved materialization text",
    ):
        module._check_gemv_vulkan_toolchain(*paths, "python")

    assert [name for name, _command in commands] == ["translate-gemv-vulkan"]


def test_arange_opengl_check_confirms_native_validation(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_arange_opengl_check(
        module,
        tmp_path,
        _arange_opengl_frontier_source(),
    )
    commands = []

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        commands.append((name, list(command)))
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(
            name,
            list(command),
            0,
            stdout_path,
            stderr_path,
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/tools/glslangValidator")

    result = module._check_arange_opengl(*paths, "python")

    assert result["aggregateInitializerLowered"] is True
    assert result["scalarCoercionLowered"] is True
    assert result["shuffleAndFillLowered"] is True
    assert result["nativeValidationAttempted"] is True
    assert result["nativeValidationBlockerConfirmed"] is False
    assert result["nativeValidationStatus"] == "validated"
    assert result["nativeValidationExitCode"] == 0
    assert result["trackedIssues"] == list(
        module.OPENGL_ARANGE_VALIDATION_TRACKED_ISSUES
    )
    validator_command = commands[1][1]
    assert validator_command[:5] == [
        "/tools/glslangValidator",
        "--target-env",
        "opengl",
        "--target-env",
        "spirv1.3",
    ]


def test_arange_opengl_check_reports_unavailable_validator(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_arange_opengl_check(
        module,
        tmp_path,
        _arange_opengl_frontier_source(),
    )

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    result = module._check_arange_opengl(*paths, "python")

    assert result["nativeValidationAttempted"] is False
    assert result["nativeValidationBlockerConfirmed"] is False
    assert result["nativeValidationStatus"] == "not-run-tool-unavailable"
    assert result["nativeValidatorStatus"] == "unavailable"
    assert result["trackedIssues"] == []


@pytest.mark.parametrize(
    ("resolved_marker", "invalid_marker"),
    (
        (
            "subgroupShuffleDown(uint(data), delta) != 0u",
            "subgroupShuffleDown(uint(data), delta)",
        ),
        (
            "subgroupShuffleUp(uint(data), delta) != 0u",
            "subgroupShuffleUp(uint(data), delta)",
        ),
        (
            "crossglWaveShuffleAndFillUp(uint(data), uint(filling), delta) != 0u",
            "crossglWaveShuffleAndFillUp(uint(data), uint(filling), delta)",
        ),
        (
            "subgroupShuffle(uint(data), lane) != 0u",
            "subgroupShuffle(uint(data), lane)",
        ),
        (
            "uint(arangeint8_start_Args_start)",
            "arangeint8_start_Args_start",
        ),
        (
            "uint(arangeint16_start_Args_start)",
            "arangeint16_start_Args_start",
        ),
        (
            "uint(arangeint32_start_Args_start)",
            "arangeint32_start_Args_start",
        ),
        ("int64_t(index)", "index"),
    ),
)
def test_arange_opengl_check_requires_scalar_coercions_before_validation(
    tmp_path,
    monkeypatch,
    resolved_marker,
    invalid_marker,
):
    module = _load_harness()
    generated = _arange_opengl_frontier_source()
    assert resolved_marker in generated
    paths = _prepare_arange_opengl_check(
        module,
        tmp_path,
        generated.replace(resolved_marker, invalid_marker, 1),
    )
    commands = []

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        commands.append(name)
        if name == "validate-arange-opengl":
            pytest.fail("native validation ran before scalar coercion checks passed")
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/tools/glslangValidator")

    with pytest.raises(module.PortingCheckError, match="scalar coercion"):
        module._check_arange_opengl(*paths, "python")

    assert commands == ["translate-arange-opengl"]


@pytest.mark.parametrize(
    "diagnostic",
    (
        "ERROR: return type does not match the function return type",
        "ERROR: cannot convert from uint to bool",
        (
            "ERROR: arange.glsl:301: 'simd_shuffle_down' : "
            "no matching overloaded function found\n"
            "ERROR: return type does not match the function return type"
        ),
    ),
)
def test_arange_opengl_check_rejects_native_failure_after_frontier_resolved(
    tmp_path,
    monkeypatch,
    diagnostic,
):
    module = _load_harness()
    paths = _prepare_arange_opengl_check(
        module,
        tmp_path,
        _arange_opengl_frontier_source(),
    )

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text(
            diagnostic if name == "validate-arange-opengl" else "",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(
            name,
            list(command),
            2 if name == "validate-arange-opengl" else 0,
            stdout_path,
            stderr_path,
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/tools/glslangValidator")

    with pytest.raises(
        module.PortingCheckError,
        match="failed without a tracked validation issue",
    ):
        module._check_arange_opengl(*paths, "python")


def test_arange_opengl_check_rejects_unrelated_overload_diagnostic(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_arange_opengl_check(
        module,
        tmp_path,
        _arange_opengl_frontier_source(),
    )

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text(
            (
                "ERROR: arange.glsl:301: 'unrelated_helper' : "
                "no matching overloaded function found"
                if name == "validate-arange-opengl"
                else ""
            ),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(
            name,
            list(command),
            2 if name == "validate-arange-opengl" else 0,
            stdout_path,
            stderr_path,
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/tools/glslangValidator")

    with pytest.raises(
        module.PortingCheckError,
        match="failed without a tracked validation issue",
    ):
        module._check_arange_opengl(*paths, "python")


def test_arange_opengl_check_rejects_untyped_aggregate_return(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    generated = _arange_opengl_frontier_source().replace(
        "return complex64_t(x, theta);",
        "return { x, theta };",
    )
    paths = _prepare_arange_opengl_check(module, tmp_path, generated)

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(
            name,
            list(command),
            0,
            stdout_path,
            stderr_path,
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    with pytest.raises(
        module.PortingCheckError,
        match="retained an untyped aggregate return",
    ):
        module._check_arange_opengl(*paths, "python")


def test_runtime_readiness_reports_tracked_plan_resource_blockers(
    tmp_path, monkeypatch
):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    report_dir = mlx_root / ".crosstl-mlx-porting" / "reports"
    report_dir.mkdir(parents=True)
    artifact_report = report_dir / "opengl-readiness-artifacts.json"
    artifact_report.write_text(
        json.dumps(_translated_arange_report(module, "opengl")),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "build_runtime_artifact_manifest",
        lambda report_path: _runtime_arange_artifact_manifest(
            module, "opengl", output_name="unrelatedResource"
        ),
    )

    result = module._plan_runtime_readiness_for_report(
        mlx_root=mlx_root,
        report_dir=report_dir,
        name="opengl-runtime-readiness",
        artifact_report=artifact_report,
        targets=("opengl",),
        require_vulkan_native_runtime=False,
    )

    assert result["status"] == "blocked-by-tracked-issues"
    assert result["diagnosticCounts"] == {"error": 0, "note": 0, "warning": 0}
    assert result["metadataGapCodes"] == []
    assert result["planBlockerCodes"] == [
        "project.runtime-verification.resource-unbound"
    ]
    assert (
        result["runtimePlanDiagnosticsByCode"][
            "project.runtime-verification.resource-unbound"
        ]
        == 1
    )
    assert (
        "https://github.com/CrossGL/crosstl/issues/1392"
        not in result["trackedRuntimeIssues"]
    )
    assert result["runtimeFixtureExecution"]["status"] == "blocked-by-tracked-issues"
    assert result["nativeRuntimeExecution"]["status"] in {
        "blocked-by-runtime-driver",
        "blocked-by-tracked-issues",
    }


def test_reduced_runtime_readiness_aggregates_fixture_execution(monkeypatch):
    module = _load_harness()
    reports = [
        {
            "name": "directx-vulkan-runtime-readiness",
            "status": "planned",
            "testCount": 2,
            "diagnosticsByCode": {},
            "runtimeArtifactDiagnosticsByCode": {},
            "runtimePlanDiagnosticsByCode": {},
            "runtimeFixtureExecution": {
                "status": "passed",
                "summary": {
                    "fixtureCount": 2,
                    "passedCount": 2,
                    "skippedCount": 0,
                    "unavailableCount": 0,
                    "translationFailedCount": 0,
                    "runtimeFailedCount": 0,
                    "comparisonFailedCount": 0,
                    "failedCount": 0,
                },
            },
            "nativeRuntimeExecution": {
                "status": "blocked-by-runtime-driver",
                "summary": {
                    "fixtureCount": 2,
                    "passedCount": 0,
                    "skippedCount": 0,
                    "unavailableCount": 2,
                    "translationFailedCount": 0,
                    "runtimeFailedCount": 0,
                    "comparisonFailedCount": 0,
                    "failedCount": 0,
                },
            },
        },
        {
            "name": "opengl-runtime-readiness",
            "status": "blocked-by-tracked-issues",
            "testCount": 1,
            "diagnosticsByCode": {},
            "runtimeArtifactDiagnosticsByCode": {},
            "runtimePlanDiagnosticsByCode": {
                "project.runtime-verification.resource-unbound": 1
            },
            "runtimeFixtureExecution": {
                "status": "blocked-by-tracked-issues",
                "summary": {
                    "fixtureCount": 1,
                    "passedCount": 0,
                    "skippedCount": 1,
                    "unavailableCount": 0,
                    "translationFailedCount": 0,
                    "runtimeFailedCount": 0,
                    "comparisonFailedCount": 0,
                    "failedCount": 0,
                },
            },
            "nativeRuntimeExecution": {
                "status": "blocked-by-runtime-driver",
                "summary": {
                    "fixtureCount": 1,
                    "passedCount": 0,
                    "skippedCount": 0,
                    "unavailableCount": 1,
                    "translationFailedCount": 0,
                    "runtimeFailedCount": 0,
                    "comparisonFailedCount": 0,
                    "failedCount": 0,
                },
            },
        },
    ]

    monkeypatch.setattr(
        module,
        "_plan_runtime_readiness_for_report",
        lambda **kwargs: reports.pop(0),
    )

    result = module._plan_reduced_runtime_readiness(
        Path("/tmp/mlx"),
        Path("/tmp/reports"),
        require_vulkan_native_runtime=False,
    )

    assert result["status"] == "blocked-by-tracked-issues"
    assert result["runtimeFixtureExecutionIncluded"] is True
    assert result["runtimeFixtureExecutionByStatus"] == {
        "blocked-by-tracked-issues": 1,
        "passed": 1,
    }
    assert result["runtimeFixtureExecutionSummary"] == {
        "comparisonFailedCount": 0,
        "failedCount": 0,
        "fixtureCount": 3,
        "passedCount": 2,
        "runtimeFailedCount": 0,
        "skippedCount": 1,
        "translationFailedCount": 0,
        "unavailableCount": 0,
    }
    assert result["nativeRuntimeExecutionIncluded"] is True
    assert result["nativeRuntimeExecutionByStatus"] == {
        "blocked-by-runtime-driver": 2,
    }
    assert result["nativeRuntimeExecutionSummary"] == {
        "comparisonFailedCount": 0,
        "failedCount": 0,
        "fixtureCount": 3,
        "passedCount": 0,
        "runtimeFailedCount": 0,
        "skippedCount": 0,
        "translationFailedCount": 0,
        "unavailableCount": 3,
    }


def test_full_corpus_mode_writes_bounded_config_and_checks_counts(
    tmp_path, monkeypatch
):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)
    commands = []

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        commands.append(list(command))
        (report_dir / "full-corpus.json").write_text(
            json.dumps(_full_corpus_report(module, mlx_root, work_dir)),
            encoding="utf-8",
        )
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 1, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    result = module._translate_full_corpus(
        mlx_root, work_dir, config_dir, report_dir, log_dir, "python"
    )

    config = (config_dir / "full-corpus.toml").read_text(encoding="utf-8")
    assert 'include = ["mlx/backend/metal/kernels/**/*.metal"]' in config
    assert 'targets = ["directx", "opengl", "vulkan"]' in config
    assert "max_template_specializations = 4096" in config
    assert "max_template_materialization_work = 131072" in config
    assert commands == [
        [
            "python",
            "-m",
            "crosstl",
            "translate-project",
            str(mlx_root),
            "--config",
            str(config_dir / "full-corpus.toml"),
            "--report",
            str(report_dir / "full-corpus.json"),
            "--validate",
        ]
    ]
    assert "--run-toolchains" not in commands[0]
    assert result["unitCount"] == 40
    assert result["artifactCount"] == 120
    assert result["translatedCount"] == 117
    assert result["failedCount"] == 3
    assert result["status"] == "passed-with-expected-fence-blockers"
    assert result["targetCounts"] == {
        "directx": {"translatedCount": 39, "failedCount": 1},
        "opengl": {"translatedCount": 39, "failedCount": 1},
        "vulkan": {"translatedCount": 39, "failedCount": 1},
    }
    assert result["fenceContract"]["status"] == "blocked-as-expected"
    assert set(result["fenceContract"]["targetContracts"]) == {
        "directx",
        "opengl",
        "vulkan",
    }
    assert result["shaderArtifactsOnly"] is True
    assert result["runtimeIntegrationIncluded"] is False
    assert result["runtimeParityClaimed"] is False


def test_full_corpus_mode_rejects_untracked_translation_errors(tmp_path, monkeypatch):
    module = _load_harness()
    monkeypatch.setattr(module, "FULL_CORPUS_TRANSLATION_TRACKED_ISSUES", ())
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        report = _full_corpus_report(
            module,
            mlx_root,
            work_dir,
            include_extra_failure=True,
        )
        (report_dir / "full-corpus.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    with pytest.raises(module.PortingCheckError, match="tracked issue references"):
        module._translate_full_corpus(
            mlx_root, work_dir, config_dir, report_dir, log_dir, "python"
        )


def test_full_corpus_mode_reports_tracked_translation_errors(tmp_path, monkeypatch):
    module = _load_harness()
    monkeypatch.setattr(
        module,
        "FULL_CORPUS_TRANSLATION_TRACKED_ISSUES",
        ("https://github.com/CrossGL/crosstl/issues/1354",),
    )
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        report = _full_corpus_report(
            module,
            mlx_root,
            work_dir,
            include_extra_failure=True,
        )
        (report_dir / "full-corpus.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 1, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    result = module._translate_full_corpus(
        mlx_root, work_dir, config_dir, report_dir, log_dir, "python"
    )

    assert result["status"] == "blocked-by-tracked-issues"
    assert result["translatedCount"] == 116
    assert result["failedCount"] == 4
    assert result["expectedFenceFailureCount"] == 3
    assert result["unexpectedFailedCount"] == 1
    assert result["unexpectedErrorDiagnosticsByCode"] == {
        "project.translate.failed": 1,
        "project.validate.failed-artifact": 1,
    }
    assert result["trackedTranslationIssues"] == [
        "https://github.com/CrossGL/crosstl/issues/1354"
    ]


def test_full_corpus_mode_reports_tracked_timeout_without_report(tmp_path, monkeypatch):
    module = _load_harness()
    monkeypatch.setattr(
        module,
        "FULL_CORPUS_TRANSLATION_TRACKED_ISSUES",
        ("https://github.com/CrossGL/crosstl/issues/1376",),
    )
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("timed out", encoding="utf-8")
        return module.CommandResult(name, list(command), 124, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    result = module._translate_full_corpus(
        mlx_root, work_dir, config_dir, report_dir, log_dir, "python"
    )

    assert result["status"] == "blocked-by-tracked-issues"
    assert result["reportProduced"] is False
    assert result["returncode"] == 124
    assert result["trackedTranslationIssues"] == [
        "https://github.com/CrossGL/crosstl/issues/1376"
    ]


def test_reduced_frontier_accepts_multiple_vulkan_toolchain_runs_per_artifact(
    tmp_path, monkeypatch
):
    module = _load_harness()
    monkeypatch.setattr(module, "FRONTIER_VALIDATION_TRACKED_ISSUES", ())
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)

    frontier_count = len(module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES)
    base_vulkan_paths = [
        f".crosstl/out/vulkan/{Path(source).with_suffix('.spvasm')}"
        for source in module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    ]
    toolchain_vulkan_paths = [
        f".crosstl/toolchain/vulkan/{Path(source).with_suffix('.spvasm')}"
        for source in module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    ]
    commands = []
    alias_evidence = {"source": module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE}
    monkeypatch.setattr(
        module,
        "_scaled_attention_local_alias_evidence",
        lambda *_args: alias_evidence,
    )

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        commands.append((name, list(command)))
        report = {
            "summary": {
                "unitCount": frontier_count,
                "artifactCount": frontier_count * 2,
                "translatedCount": frontier_count * 2,
                "failedCount": 0,
                "diagnosticCounts": {"error": 0},
                "artifactsByTarget": {
                    "directx": {
                        "translatedCount": frontier_count,
                        "failedCount": 0,
                    },
                    "vulkan": {
                        "translatedCount": frontier_count,
                        "failedCount": 0,
                    },
                },
            },
            "artifacts": [
                {"target": "vulkan", "path": path, "status": "translated"}
                for path in (
                    base_vulkan_paths
                    if name == "translate-directx-vulkan-frontier"
                    else toolchain_vulkan_paths
                )
            ],
            "validation": {
                "summary": {"failedCount": 0},
                "toolchainRuns": (
                    []
                    if name == "translate-directx-vulkan-frontier"
                    else [
                        {
                            "target": "vulkan",
                            "path": path,
                            "status": "ok",
                        }
                        for path in toolchain_vulkan_paths
                        for _ in range(2)
                    ]
                ),
            },
        }
        report_name = (
            "directx-vulkan-frontier.json"
            if name == "translate-directx-vulkan-frontier"
            else "vulkan-frontier-toolchain.json"
        )
        (report_dir / report_name).write_text(json.dumps(report), encoding="utf-8")
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    result = module._translate_directx_vulkan_frontier(
        mlx_root,
        work_dir,
        config_dir,
        report_dir,
        log_dir,
        "python",
        require_directx_toolchain=False,
        require_vulkan_toolchain=True,
    )

    assert result["toolchainRuns"] == frontier_count * 2
    assert result["status"] == "passed"
    assert result["scope"] == "clean-frontier"
    assert result["vulkanValidationStatus"] == "validated"
    assert result["semanticReadinessStatus"] == "not-established"
    assert result["regressionEvidence"] == [alias_evidence]
    assert result["runtimeParityClaimed"] is False
    assert commands[0][0] == "translate-directx-vulkan-frontier"
    assert "--validate" in commands[0][1]
    assert "--run-toolchains" not in commands[0][1]
    assert commands[1][0] == "validate-vulkan-frontier-toolchain"
    assert "--run-toolchains" in commands[1][1]
    toolchain_config = (config_dir / "vulkan-frontier-toolchain.toml").read_text(
        encoding="utf-8"
    )
    assert 'targets = ["vulkan"]' in toolchain_config
    assert module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE in toolchain_config


def test_reduced_frontier_requires_directx_toolchain_runs_per_artifact(
    tmp_path, monkeypatch
):
    module = _load_harness()
    monkeypatch.setattr(module, "FRONTIER_VALIDATION_TRACKED_ISSUES", ())
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)

    frontier_count = len(module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES)
    directx_subset_count = len(module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES)
    directx_paths = [
        f".crosstl/out/directx/{Path(source).with_suffix('.hlsl')}"
        for source in module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    ]
    directx_subset_paths = [
        f".crosstl/out/directx/{Path(source).with_suffix('.hlsl')}"
        for source in module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
    ]
    commands = []
    alias_evidence = {"source": module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE}
    monkeypatch.setattr(
        module,
        "_scaled_attention_local_alias_evidence",
        lambda *_args: alias_evidence,
    )

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        commands.append((name, list(command)))
        is_toolchain = name != "translate-directx-vulkan-frontier"
        # The DXC compile gate runs only on the verified subset; the translation
        # frontier still emits every frontier artifact.
        report_directx_paths = directx_subset_paths if is_toolchain else directx_paths
        report = {
            "summary": {
                "unitCount": frontier_count,
                "artifactCount": frontier_count * 2,
                "translatedCount": frontier_count * 2,
                "failedCount": 0,
                "diagnosticCounts": {"error": 0},
                "artifactsByTarget": {
                    "directx": {
                        "translatedCount": frontier_count,
                        "failedCount": 0,
                    },
                    "vulkan": {
                        "translatedCount": frontier_count,
                        "failedCount": 0,
                    },
                },
            },
            "artifacts": [
                {"target": "directx", "path": path, "status": "translated"}
                for path in report_directx_paths
            ],
            "validation": {
                "summary": {"failedCount": 0},
                "toolchainRuns": (
                    []
                    if not is_toolchain
                    else [
                        {
                            "target": "directx",
                            "path": path,
                            "status": "ok",
                        }
                        for path in directx_subset_paths
                    ]
                ),
            },
        }
        report_name = (
            "directx-vulkan-frontier.json"
            if name == "translate-directx-vulkan-frontier"
            else "directx-frontier-toolchain.json"
        )
        (report_dir / report_name).write_text(json.dumps(report), encoding="utf-8")
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    result = module._translate_directx_vulkan_frontier(
        mlx_root,
        work_dir,
        config_dir,
        report_dir,
        log_dir,
        "python",
        require_directx_toolchain=True,
        require_vulkan_toolchain=False,
    )

    assert result["toolchainRuns"] == directx_subset_count
    assert result["status"] == "passed"
    assert result["directxToolchainRequired"] is True
    assert result["semanticReadinessStatus"] == "not-established"
    assert result["directxValidationStatus"] == "validated"
    assert result["vulkanValidationStatus"] == "not-run"
    assert commands[1][0] == "validate-directx-frontier-toolchain"
    assert "--run-toolchains" in commands[1][1]
    toolchain_config = (config_dir / "directx-frontier-toolchain.toml").read_text(
        encoding="utf-8"
    )
    assert 'targets = ["directx"]' in toolchain_config
    # The DXC compile gate is scoped to the verified subset, not the whole
    # frontier, so only those sources appear in the toolchain config.
    for source in module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES:
        assert source in toolchain_config


def test_directx_toolchain_frontier_includes_rope_and_gap_ledger_matches():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    assert module.MLX_ROPE_SOURCE in module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
    assert module.MLX_ROPE_SOURCE in module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    directx_status = gaps["directx_toolchain_status"]
    assert module.MLX_ROPE_SOURCE in directx_status["dxc_validated_sources"]
    assert module.MLX_ROPE_SOURCE not in directx_status["directx_toolchain_gaps"]


def test_binary_resource_relocation_issue_is_full_corpus_only():
    module = _load_harness()
    issue = "https://github.com/CrossGL/crosstl/issues/1659"

    assert issue in module.FULL_CORPUS_TRANSLATION_TRACKED_ISSUES
    assert issue not in module.FRONTIER_VALIDATION_TRACKED_ISSUES
    assert issue not in module.RUNTIME_READINESS_TRACKED_ISSUES


def test_run_checks_full_corpus_mode_skips_reduced_frontier(tmp_path, monkeypatch):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()

    monkeypatch.setattr(
        module,
        "_verify_mlx_checkout",
        lambda *args: {"name": "mlx-checkout", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_scan_metal_kernels",
        lambda *args: {"name": "metal-kernel-scan", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_check_metal_roundtrip",
        lambda *args, **kwargs: {"name": "metal-roundtrip", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_translate_full_corpus",
        lambda *args: {"name": "full-corpus", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_check_atomic_fence_contract",
        lambda *args: pytest.fail("dedicated fence check should not run twice"),
    )
    monkeypatch.setattr(
        module,
        "_translate_directx_vulkan_frontier",
        lambda *args, **kwargs: pytest.fail("reduced frontier should not run"),
    )
    monkeypatch.setattr(
        module,
        "_check_arange_opengl",
        lambda *args: pytest.fail("OpenGL smoke check should not run"),
    )
    monkeypatch.setattr(
        module,
        "_check_opengl_frontier",
        lambda *args, **kwargs: pytest.fail("OpenGL frontier should not run"),
    )

    result = module.run_checks(
        SimpleNamespace(
            mlx_root=str(mlx_root),
            work_dir=None,
            no_clean=False,
            python="python",
            require_directx_toolchain=False,
            require_vulkan_toolchain=False,
            mode=module.FULL_CORPUS_MODE,
        )
    )

    assert [check["name"] for check in result["checks"]] == [
        "mlx-checkout",
        "metal-kernel-scan",
        "metal-roundtrip",
        "full-corpus",
    ]
    assert result["scope"]["mode"] == module.FULL_CORPUS_MODE
    assert result["scope"]["metalRoundTripIncluded"] is True
    assert result["scope"]["fullCorpusExpectedUnitCount"] == 40
    assert result["scope"]["fullCorpusExpectedArtifactCount"] == 120
    assert result["scope"]["fullCorpusExpectedTranslatedArtifactCount"] == 117
    assert result["scope"]["fullCorpusExpectedFenceFailureCount"] == 3


def test_run_checks_reduced_frontier_includes_runtime_readiness(tmp_path, monkeypatch):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()

    monkeypatch.setattr(
        module,
        "_verify_mlx_checkout",
        lambda *args: {"name": "mlx-checkout", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_scan_metal_kernels",
        lambda *args: {"name": "metal-kernel-scan", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_check_metal_roundtrip",
        lambda *args, **kwargs: {"name": "metal-roundtrip", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_check_atomic_fence_contract",
        lambda *args: {
            "name": "atomic-fence-contract",
            "status": "blocked-as-expected",
        },
    )
    monkeypatch.setattr(
        module,
        "_translate_directx_vulkan_frontier",
        lambda *args, **kwargs: {
            "name": "directx-vulkan-frontier",
            "status": "passed",
        },
    )
    monkeypatch.setattr(
        module,
        "_check_arange_opengl",
        lambda *args: {"name": "arange-opengl", "status": "passed"},
    )
    opengl_frontier_requirements = []

    def fake_opengl_frontier_check(*args, require_toolchain):
        opengl_frontier_requirements.append(require_toolchain)
        return {"name": "opengl-frontier", "status": "passed"}

    monkeypatch.setattr(
        module,
        "_check_opengl_frontier",
        fake_opengl_frontier_check,
    )
    monkeypatch.setattr(
        module,
        "_check_gemv_opengl_toolchain",
        lambda *args: {"name": "gemv-opengl-toolchain", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_check_gemv_vulkan_toolchain",
        lambda *args: {
            "name": "gemv-vulkan-toolchain",
            "status": "passed",
        },
    )
    monkeypatch.setattr(
        module,
        "_plan_reduced_runtime_readiness",
        lambda *args, **kwargs: {
            "name": "runtime-readiness",
            "status": "blocked-by-tracked-issues",
        },
    )
    monkeypatch.setattr(
        module,
        "_translate_full_corpus",
        lambda *args: pytest.fail("full-corpus check should not run"),
    )

    result = module.run_checks(
        SimpleNamespace(
            mlx_root=str(mlx_root),
            work_dir=None,
            no_clean=False,
            python="python",
            require_directx_toolchain=False,
            require_vulkan_toolchain=False,
            require_vulkan_native_runtime=False,
            require_opengl_frontier_toolchain=True,
            require_opengl_gemv_toolchain=True,
            require_vulkan_gemv_toolchain=True,
            mode=module.REDUCED_FRONTIER_MODE,
        )
    )

    assert [check["name"] for check in result["checks"]] == [
        "mlx-checkout",
        "metal-kernel-scan",
        "metal-roundtrip",
        "atomic-fence-contract",
        "directx-vulkan-frontier",
        "arange-opengl",
        "opengl-frontier",
        "gemv-opengl-toolchain",
        "gemv-vulkan-toolchain",
        "runtime-readiness",
    ]
    assert opengl_frontier_requirements == [True]
    assert result["scope"]["openglFrontierToolchainRequired"] is True
    assert result["scope"]["openglGemvToolchainRequired"] is True
    assert result["scope"]["vulkanGemvToolchainRequired"] is True
    assert result["scope"]["runtimeReadinessIncluded"] is True
    assert result["scope"]["runtimeFixtureExecutionIncluded"] is True
    assert result["scope"]["nativeRuntimeExecutionIncluded"] is True
    assert result["scope"]["cleanFrontierSources"] == list(
        module.MLX_CLEAN_REDUCED_FRONTIER_SOURCES
    )
    assert result["scope"]["blockedFrontierSources"] == [module.MLX_FENCE_SOURCE]
    assert result["scope"]["blockedFrontierIssues"] == [
        "https://github.com/CrossGL/crosstl/issues/1537"
    ]
