import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "integrations" / "mlx" / "run_mlx_porting.py"


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

    def fake_run_command(name, command, *, log_dir, check=True):
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
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)

    def fake_run_command(name, command, *, log_dir, check=True):
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
