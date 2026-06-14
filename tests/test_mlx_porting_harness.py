import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "integrations" / "mlx" / "run_mlx_porting.py"


def _load_harness_module():
    spec = importlib.util.spec_from_file_location("run_mlx_porting", HARNESS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_mlx_checkout(tmp_path, module):
    mlx_root = tmp_path / "mlx"
    for source in module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES:
        path = mlx_root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// metal\n", encoding="utf-8")
    return mlx_root


def _report_path(command, flag):
    return Path(command[command.index(flag) + 1])


def _install_fake_runner(monkeypatch, module, mlx_root, *, full_corpus_payload=None):
    calls = []

    def fake_run_command(name, command, *, log_dir, check=True):
        calls.append((name, list(command)))
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        returncode = 0

        if name == "mlx-revision":
            stdout_path.write_text(module.MLX_COMMIT + "\n", encoding="utf-8")
        elif name == "scan-metal-kernels":
            report_path = _report_path(command, "--output")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(
                    {
                        "summary": {
                            "unitCount": module.EXPECTED_METAL_KERNEL_COUNT,
                            "includeDependencyCount": 3,
                            "diagnosticCounts": {"error": 0},
                        },
                        "units": [
                            {"path": source}
                            for source in module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
                        ],
                    }
                ),
                encoding="utf-8",
            )
        elif name == "translate-directx-vulkan-frontier":
            report_path = _report_path(command, "--report")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(
                    {
                        "summary": {
                            "unitCount": len(
                                module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
                            ),
                            "artifactCount": 10,
                            "translatedCount": 10,
                            "failedCount": 0,
                            "diagnosticCounts": {"error": 0},
                            "artifactsByTarget": {
                                "directx": {
                                    "translatedCount": 5,
                                    "failedCount": 0,
                                },
                                "vulkan": {
                                    "translatedCount": 5,
                                    "failedCount": 0,
                                },
                            },
                        },
                        "validation": {
                            "summary": {"failedCount": 0},
                            "toolchainRuns": [
                                {
                                    "target": "vulkan",
                                    "source": source,
                                    "status": "ok",
                                }
                                for source in module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
        elif name == "translate-arange-opengl":
            artifact_path = ".crosstl-mlx-porting/out-arange-opengl/arange.glsl"
            generated_path = mlx_root / artifact_path
            generated_path.parent.mkdir(parents=True, exist_ok=True)
            generated_path.write_text("// glsl\n", encoding="utf-8")
            report_path = _report_path(command, "--report")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(
                    {
                        "summary": {"translatedCount": 1, "failedCount": 0},
                        "artifacts": [
                            {
                                "source": module.MLX_ARANGE_SOURCE,
                                "target": "opengl",
                                "path": artifact_path,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        elif name == "translate-full-corpus":
            report_path = _report_path(command, "--report")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            assert full_corpus_payload is not None
            report_path.write_text(json.dumps(full_corpus_payload), encoding="utf-8")
            returncode = 1 if full_corpus_payload["summary"]["failedCount"] else 0
        else:
            raise AssertionError(f"unexpected command: {name}")

        if check and returncode != 0:
            raise module.PortingCheckError(f"{name} failed")
        return module.CommandResult(
            name=name,
            command=list(command),
            returncode=returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    return calls


def _passing_full_corpus_payload(module):
    return {
        "summary": {
            "unitCount": module.EXPECTED_METAL_KERNEL_COUNT,
            "artifactCount": module.EXPECTED_METAL_KERNEL_COUNT * 3,
            "translatedCount": module.EXPECTED_METAL_KERNEL_COUNT * 3,
            "failedCount": 0,
            "diagnosticCounts": {"error": 0},
            "artifactsByTarget": {
                target: {
                    "translatedCount": module.EXPECTED_METAL_KERNEL_COUNT,
                    "failedCount": 0,
                }
                for target in ("directx", "opengl", "vulkan")
            },
        },
        "validation": {
            "toolchainRuns": [
                {"target": "vulkan", "source": f"kernel-{index}.metal", "status": "ok"}
                for index in range(module.EXPECTED_METAL_KERNEL_COUNT)
            ]
        },
    }


def test_default_mlx_harness_marks_full_corpus_not_run(monkeypatch, tmp_path):
    module = _load_harness_module()
    mlx_root = _write_mlx_checkout(tmp_path, module)
    _install_fake_runner(monkeypatch, module, mlx_root)

    args = module.parse_args(["--mlx-root", str(mlx_root)])
    summary = module.run_checks(args)

    assert summary["status"] == "passed"
    assert summary["scope"]["fullCorpusIncluded"] is False
    assert summary["fullCorpus"]["status"] == "not-run"
    assert "full-corpus" not in {check["name"] for check in summary["checks"]}
    assert (
        "https://github.com/CrossGL/crosstl/issues/1317" not in summary["trackedIssues"]
    )
    assert (
        "https://github.com/CrossGL/crosstl/issues/1317"
        in summary["resolvedFrontierIssues"]
    )
    assert (
        "https://github.com/CrossGL/crosstl/issues/1362" not in summary["trackedIssues"]
    )
    assert (
        "https://github.com/CrossGL/crosstl/issues/1362"
        in summary["resolvedFrontierIssues"]
    )
    assert "https://github.com/CrossGL/crosstl/issues/1354" in summary["trackedIssues"]
    assert (
        "https://github.com/CrossGL/crosstl/issues/1355" not in summary["trackedIssues"]
    )
    assert (
        "https://github.com/CrossGL/crosstl/issues/1355"
        in summary["resolvedFrontierIssues"]
    )


def test_full_corpus_flag_enforces_clean_artifact_counts(monkeypatch, tmp_path):
    module = _load_harness_module()
    mlx_root = _write_mlx_checkout(tmp_path, module)
    calls = _install_fake_runner(
        monkeypatch,
        module,
        mlx_root,
        full_corpus_payload=_passing_full_corpus_payload(module),
    )

    args = module.parse_args(
        ["--mlx-root", str(mlx_root), "--full-corpus", "--require-vulkan-toolchain"]
    )
    summary = module.run_checks(args)

    full_corpus = summary["fullCorpus"]
    assert full_corpus["status"] == "passed"
    assert full_corpus["artifactCount"] == module.EXPECTED_METAL_KERNEL_COUNT * 3
    assert full_corpus["vulkanValidationStatus"] == "validated"
    full_corpus_command = next(
        command for name, command in calls if name == "translate-full-corpus"
    )
    assert "--run-toolchains" in full_corpus_command


def test_full_corpus_failure_is_issue_linked(monkeypatch, tmp_path):
    module = _load_harness_module()
    mlx_root = _write_mlx_checkout(tmp_path, module)
    failing_payload = {
        "summary": {
            "unitCount": module.EXPECTED_METAL_KERNEL_COUNT,
            "artifactCount": module.EXPECTED_METAL_KERNEL_COUNT * 3,
            "translatedCount": 57,
            "failedCount": 63,
            "diagnosticCounts": {"error": 0},
        },
        "validation": {"toolchainRuns": []},
    }
    _install_fake_runner(
        monkeypatch,
        module,
        mlx_root,
        full_corpus_payload=failing_payload,
    )
    args = module.parse_args(["--mlx-root", str(mlx_root), "--full-corpus"])

    with pytest.raises(module.PortingCheckError) as excinfo:
        module.run_checks(args)

    message = str(excinfo.value)
    assert "full-corpus translation is not clean" in message
    assert "https://github.com/CrossGL/crosstl/issues/1354" in message
    assert "https://github.com/CrossGL/crosstl/issues/1355" not in message
    assert "https://github.com/CrossGL/crosstl/issues/1362" not in message
