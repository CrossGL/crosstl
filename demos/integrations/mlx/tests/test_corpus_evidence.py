import importlib
import json
import subprocess
import sys
from contextlib import nullcontext

import pytest

from demos.integrations.mlx.tests.corpus_evidence import (
    EVIDENCE_DIRECTORY,
    KEEP_EVIDENCE_ENV,
    corpus_workspace,
    run_compiler,
)


@pytest.mark.parametrize("keep", (False, True))
@pytest.mark.parametrize("fail", (False, True))
def test_workspace_retention_does_not_change_failure(keep, fail, tmp_path, monkeypatch):
    monkeypatch.setenv(KEEP_EVIDENCE_ENV, "1" if keep else "0")
    error = AssertionError("artifact mismatch")
    with pytest.raises(AssertionError) if fail else nullcontext():
        with corpus_workspace(
            tmp_path, family="copy", target="directx", entry_point="copy_float"
        ) as work_dir:
            assert work_dir.relative_to(tmp_path).parts
            (work_dir / "shader.hlsl").write_text("void CSMain() {}", encoding="utf-8")
            if fail:
                raise error
    assert work_dir.exists() is keep
    if keep:
        assert work_dir.parent == tmp_path / EVIDENCE_DIRECTORY
        assert (work_dir / "shader.hlsl").is_file()
        assert json.loads((work_dir / "case.json").read_text(encoding="utf-8")) == {
            "family": "copy",
            "target": "directx",
            "entryPoint": "copy_float",
        }


def test_retained_workspaces_do_not_overwrite_repeated_cases(tmp_path, monkeypatch):
    monkeypatch.setenv(KEEP_EVIDENCE_ENV, "1")
    paths = []
    for _ in range(2):
        with corpus_workspace(
            tmp_path, family="copy", target="directx", entry_point="copy_float"
        ) as work_dir:
            paths.append(work_dir)
    assert paths[0] != paths[1]
    assert all(path.is_dir() for path in paths)


@pytest.mark.parametrize("returncode", (0, 7))
def test_compiler_preserves_command_output_and_failure(returncode, tmp_path):
    command = [
        sys.executable,
        "-c",
        f"import sys; print('output'); print('diagnostic', file=sys.stderr); sys.exit({returncode})",
    ]
    result = run_compiler(command, work_dir=tmp_path, timeout=30)
    assert result.returncode == returncode
    assert json.loads((tmp_path / "compiler.json").read_text(encoding="utf-8")) == {
        "command": command,
        "timeoutSeconds": 30,
        "status": "completed",
        "returncode": returncode,
        "stdout": "output\n",
        "stderr": "diagnostic\n",
    }


@pytest.mark.parametrize("byte_output", (False, True))
def test_compiler_retains_timeout_output_and_reraises(
    byte_output, tmp_path, monkeypatch
):
    stdout = b"partial\xff" if byte_output else "partial"
    stderr = b"diagnostic" if byte_output else "diagnostic"
    error = subprocess.TimeoutExpired(["dxc"], 120, output=stdout, stderr=stderr)

    def timeout(*args, **kwargs):
        record = json.loads((tmp_path / "compiler.json").read_text(encoding="utf-8"))
        assert record["status"] == "running"
        assert kwargs == {
            "check": False,
            "capture_output": True,
            "text": True,
            "timeout": 120,
        }
        raise error

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_compiler(["dxc"], work_dir=tmp_path, timeout=120)
    assert caught.value is error
    record = json.loads((tmp_path / "compiler.json").read_text(encoding="utf-8"))
    assert record["status"] == "timed-out"
    assert record["stdout"] == ("partial\ufffd" if byte_output else "partial")
    assert record["stderr"] == "diagnostic"
    assert "returncode" not in record


def test_compiler_retains_launch_error_and_reraises(tmp_path, monkeypatch):
    error = OSError("compiler unavailable")

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(OSError) as caught:
        run_compiler(["dxc"], work_dir=tmp_path, timeout=120)
    assert caught.value is error
    record = json.loads((tmp_path / "compiler.json").read_text(encoding="utf-8"))
    assert record["status"] == "launch-failed"
    assert record["error"] == "compiler unavailable"
    assert "returncode" not in record


@pytest.mark.parametrize("family", ("unary", "binary", "copy", "reduce"))
def test_corpus_retains_report_before_translation_assertions(
    family, tmp_path, monkeypatch
):
    module = importlib.import_module(
        f"demos.integrations.mlx.tests.kernels.test_{family}_complete_directx"
    )
    payload = {
        "summary": {"unitCount": 0},
        "diagnostics": [{"message": "test failure"}],
    }

    class Report:
        def to_json(self):
            return payload

        def write_json(self, path):
            path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(module, "load_project_config", lambda *args: None)
    monkeypatch.setattr(module, "translate_project", lambda *args, **kwargs: Report())
    workload = getattr(module, f"CURRENT_{family.upper()}_DIRECTX_WORKLOADS")[0]
    with pytest.raises(AssertionError):
        module._translate_and_validate(tmp_path, tmp_path, workload)
    assert (
        json.loads((tmp_path / "portability-report.json").read_text(encoding="utf-8"))
        == payload
    )
    assert (tmp_path / "crosstl.toml").is_file()
    assert not (tmp_path / "compiler.json").exists()
