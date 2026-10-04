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


@pytest.mark.parametrize(
    "family,target",
    [(name, "directx") for name in ("unary", "binary", "copy", "reduce")]
    + [("binary", "metal")],
)
def test_corpus_retains_report_before_translation_assertions(
    family, target, tmp_path, monkeypatch
):
    suffix = "metal_roundtrip" if target == "metal" else target
    module = importlib.import_module(
        f"demos.integrations.mlx.tests.kernels.test_{family}_complete_{suffix}"
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
    workload = getattr(module, f"CURRENT_{family.upper()}_{target.upper()}_WORKLOADS")[
        0
    ]
    translate = (
        module._translate_binary_metal_artifact
        if target == "metal"
        else module._translate_and_validate
    )
    with pytest.raises(AssertionError):
        translate(tmp_path, tmp_path, workload)
    assert (
        json.loads((tmp_path / "portability-report.json").read_text(encoding="utf-8"))
        == payload
    )
    assert (tmp_path / "crosstl.toml").is_file()
    assert not (tmp_path / "compiler.json").exists()


@pytest.mark.parametrize(
    "damage", (None, "translation", "manifest", "entry_point", "resources")
)
def test_binary_bundle_export_requires_translation_and_host_interface(
    damage, tmp_path, monkeypatch
):
    from demos.integrations.mlx.tests.kernels import (
        test_binary_complete_metal_roundtrip as binary,
    )

    workload = binary.CURRENT_BINARY_METAL_WORKLOADS[0]
    monkeypatch.setattr(binary, "_pinned_mlx_root", lambda: tmp_path)
    manifest = {
        "success": damage != "manifest",
        "artifacts": [
            {
                "hostInterface": {
                    "status": "ready",
                    "entryPoints": [
                        {
                            "name": workload.entry_point,
                            "stage": "compute",
                            "executionConfig": {},
                        }
                    ],
                    "resources": [
                        dict(resource, metadata={"entryPoint": workload.entry_point})
                        for resource in binary._resources(
                            binary.BINARY_SHAPE_SPECS[workload.shape].resource_kind
                        )
                    ],
                }
            }
        ],
    }
    if damage == "entry_point":
        manifest["artifacts"][0]["hostInterface"]["entryPoints"][0][
            "name"
        ] = "incorrect"
    if damage == "resources":
        manifest["artifacts"][0]["hostInterface"]["resources"] = []
    calls = []

    def translate(*args):
        calls.append("translation")
        assert damage != "translation"
        return tmp_path / "report.json", tmp_path / "source.metal"

    def reflect(*args):
        calls.append("reflection")
        return manifest

    def export(source, contract, entry, root):
        assert calls == ["translation", "reflection"]
        assert source == tmp_path / "source.metal"
        assert contract == binary.BINARY_METAL_CONTRACT_PATH
        assert entry == workload.entry_point
        assert root == tmp_path / "bundle"
        calls.append("export")

    monkeypatch.setattr(binary, "_translate_binary_metal_artifact", translate)
    monkeypatch.setattr(binary, "build_runtime_artifact_manifest", reflect)
    monkeypatch.setattr(binary, "write_bundle_entry", export)
    monkeypatch.setattr(
        binary.shutil,
        "which",
        lambda *args: pytest.fail("Export attempted native compilation"),
    )
    with pytest.raises(AssertionError) if damage else nullcontext():
        binary._roundtrip_pinned_mlx_binary_through_metal(
            workload, bundle_root=tmp_path / "bundle"
        )
    assert ("export" in calls) is (damage is None)


@pytest.mark.parametrize("mode", ("native", "source"))
def test_binary_required_source_cannot_skip(mode, monkeypatch):
    from demos.integrations.mlx.tests.kernels import (
        test_binary_complete_metal_roundtrip as binary,
    )

    monkeypatch.delenv("CROSTL_MLX_ROOT", raising=False)
    monkeypatch.delenv(binary.REQUIRE_BINARY_METAL_ENV, raising=False)
    monkeypatch.delenv(binary.REQUIRE_BINARY_METAL_SOURCE_ENV, raising=False)
    flag = (
        binary.REQUIRE_BINARY_METAL_ENV
        if mode == "native"
        else binary.REQUIRE_BINARY_METAL_SOURCE_ENV
    )
    monkeypatch.setenv(flag, "1")
    with pytest.raises(
        pytest.fail.Exception, match="CROSTL_MLX_ROOT is not configured"
    ):
        binary._pinned_mlx_root()
