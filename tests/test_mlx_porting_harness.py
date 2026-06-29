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


def _full_corpus_report(module, **summary_overrides):
    per_target = {
        target: {
            "translatedCount": module.EXPECTED_METAL_KERNEL_COUNT,
            "failedCount": 0,
        }
        for target in module.FULL_CORPUS_TARGETS
    }
    summary = {
        "unitCount": module.EXPECTED_METAL_KERNEL_COUNT,
        "artifactCount": module.FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
        "translatedCount": module.FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
        "failedCount": 0,
        "diagnosticCounts": {"error": 0},
        "artifactsByTarget": per_target,
    }
    summary.update(summary_overrides)
    return {"summary": summary, "validation": {"summary": {"failedCount": 0}}}


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
    )

    assert build_calls == [artifact_report]
    assert result["status"] == "planned"
    assert result["trackedRuntimeIssues"] == [
        "https://github.com/CrossGL/crosstl/issues/1388",
        "https://github.com/CrossGL/crosstl/issues/1392",
        "https://github.com/CrossGL/crosstl/issues/1396",
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
        "https://github.com/CrossGL/crosstl/issues/1392",
        "https://github.com/CrossGL/crosstl/issues/1396",
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


@pytest.mark.parametrize(
    ("target", "entry_point"),
    (("directx", "CSMain"), ("opengl", "main"), ("vulkan", "arangeuint8")),
)
def test_runtime_readiness_selects_entry_point_independently(target, entry_point):
    module = _load_harness()

    fixture = module._runtime_readiness_fixture(target)

    assert fixture["selector"] == {
        "source": module.MLX_ARANGE_SOURCE,
        "target": target,
    }
    assert fixture["entryPoint"] == entry_point
    assert fixture["runtimeAdapter"]["dispatch"] == {
        "globalSize": [4, 1, 1],
    }
    assert "https://github.com/CrossGL/crosstl/issues/1394" not in (
        module.RUNTIME_READINESS_TRACKED_ISSUES
    )
    assert "https://github.com/CrossGL/crosstl/issues/1394" in (
        module.RESOLVED_FRONTIER_ISSUES
    )


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
        in result["trackedRuntimeIssues"]
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
        Path("/tmp/mlx"), Path("/tmp/reports")
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
            json.dumps(_full_corpus_report(module)), encoding="utf-8"
        )
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

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
    assert result["targetCounts"] == {
        "directx": {"translatedCount": 40, "failedCount": 0},
        "opengl": {"translatedCount": 40, "failedCount": 0},
        "vulkan": {"translatedCount": 40, "failedCount": 0},
    }
    assert result["shaderArtifactsOnly"] is True
    assert result["runtimeIntegrationIncluded"] is False


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
            translatedCount=119,
            failedCount=1,
            diagnosticCounts={"error": 1},
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
            translatedCount=119,
            failedCount=1,
            diagnosticCounts={"error": 1},
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
    assert result["translatedCount"] == 119
    assert result["failedCount"] == 1
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
    vulkan_paths = [
        f".crosstl/out/vulkan/{Path(source).with_suffix('.spvasm')}"
        for source in module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    ]
    commands = []

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
                for path in vulkan_paths
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
                        for path in vulkan_paths
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
        require_vulkan_toolchain=True,
    )

    assert result["toolchainRuns"] == frontier_count * 2
    assert result["vulkanValidationStatus"] == "validated"
    assert commands[0][0] == "translate-directx-vulkan-frontier"
    assert "--validate" in commands[0][1]
    assert "--run-toolchains" not in commands[0][1]
    assert commands[1][0] == "validate-vulkan-frontier-toolchain"
    assert "--run-toolchains" in commands[1][1]
    toolchain_config = (config_dir / "vulkan-frontier-toolchain.toml").read_text(
        encoding="utf-8"
    )
    assert 'targets = ["vulkan"]' in toolchain_config


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
        "_translate_full_corpus",
        lambda *args: {"name": "full-corpus", "status": "passed"},
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

    result = module.run_checks(
        SimpleNamespace(
            mlx_root=str(mlx_root),
            work_dir=None,
            no_clean=False,
            python="python",
            require_vulkan_toolchain=False,
            mode=module.FULL_CORPUS_MODE,
        )
    )

    assert [check["name"] for check in result["checks"]] == [
        "mlx-checkout",
        "metal-kernel-scan",
        "full-corpus",
    ]
    assert result["scope"]["mode"] == module.FULL_CORPUS_MODE
    assert result["scope"]["fullCorpusExpectedUnitCount"] == 40
    assert result["scope"]["fullCorpusExpectedArtifactCount"] == 120


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
    monkeypatch.setattr(
        module,
        "_plan_reduced_runtime_readiness",
        lambda *args: {
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
            require_vulkan_toolchain=False,
            mode=module.REDUCED_FRONTIER_MODE,
        )
    )

    assert [check["name"] for check in result["checks"]] == [
        "mlx-checkout",
        "metal-kernel-scan",
        "directx-vulkan-frontier",
        "arange-opengl",
        "runtime-readiness",
    ]
    assert result["scope"]["runtimeReadinessIncluded"] is True
    assert result["scope"]["runtimeFixtureExecutionIncluded"] is True
    assert result["scope"]["nativeRuntimeExecutionIncluded"] is True
