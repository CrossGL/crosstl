import json
import sys
from pathlib import Path

import pytest

from demos.integrations.mlx import run_mlx_metal_host as host


@pytest.mark.parametrize("count,skipped", [(1, 0), (287, 0), (287, 3)])
def test_upstream_test_accounting(tmp_path, count, skipped):
    path = tmp_path / "result"
    suffix = f" (skipped={skipped})" if skipped else ""
    path.write_text(f"Ran {count} tests in 12.3s\n\nOK{suffix}\n")
    assert host.unittest_counts(path) == {"total": count, "skipped": skipped}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Ran 0 tests in 0s\nOK\n",
        "Ran 1 test in 1s\nFAILED\n",
        "Ran 1 test in 1s\nOK\nRan 1 test in 1s\nOK\n",
    ],
)
def test_incomplete_upstream_accounting_is_rejected(tmp_path, text):
    path = tmp_path / "result"
    path.write_text(text)
    with pytest.raises(ValueError):
        host.unittest_counts(path)


def test_dispatch_trace_requires_loaded_entry_and_dimensions(tmp_path):
    path = tmp_path / "trace"
    path.write_text(
        "library\tss_Powercomplex64\n"
        "dispatch\tss_Powercomplex64\tthreads\t2,3,1\t2,1,1\n"
    )
    assert host.parse_trace(path) == [
        {
            "entry": "ss_Powercomplex64",
            "kind": "threads",
            "grid": [2, 3, 1],
            "group": [2, 1, 1],
        }
    ]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "library\tss_Powercomplex64\n",
        "library\tunknown\n",
        "dispatch\tss_Powercomplex64\tthreads\t1,1,1\t1,1,1\n",
        "library\tss_Powercomplex64\ndispatch\tss_Powercomplex64\tthreads\t0,1,1\t1,1,1\n",
        "library\tss_Powercomplex64\ndispatch\tss_Powercomplex64\tthreads\t1,1\t1,1,1\n",
        "library\tss_Powercomplex64\ndispatch\tss_Powercomplex64\tunknown\t1,1,1\t1,1,1\n",
    ],
)
def test_unproven_dispatch_is_rejected(tmp_path, text):
    path = tmp_path / "trace"
    path.write_text(text)
    with pytest.raises(ValueError):
        host.parse_trace(path)


def test_native_command_preserves_failure_evidence(tmp_path):
    result = host.run(
        [sys.executable, "-c", "raise SystemExit(3)"], tmp_path, "failure", check=False
    )
    assert result["returncode"] == 3
    assert json.loads((tmp_path / "failure.json").read_text()) == result


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX native Metal worker")
def test_native_command_is_bounded(tmp_path):
    with pytest.raises(RuntimeError, match="deadline"):
        host.run(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            tmp_path,
            "deadline",
            timeout=0.1,
        )
    assert json.loads((tmp_path / "deadline.json").read_text())["timedOut"] is True


@pytest.mark.skipif(
    sys.platform == "win32", reason="Metal uses POSIX virtual environments"
)
def test_verify_preserves_virtual_environment_executable(tmp_path, monkeypatch):
    executable = tmp_path / "python"
    executable.symlink_to(sys.executable)
    calls = []
    monkeypatch.setattr(host, "verify", lambda *args: calls.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "host",
            "verify",
            "--mlx-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--python",
            str(executable),
        ],
    )
    host.main()
    assert calls[0][1] == executable
    assert calls[0][1] != executable.resolve()


def test_prepare_refuses_existing_header(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "verify_checkout", lambda *args, **kwargs: None)
    header = tmp_path / host.HEADER
    header.parent.mkdir(parents=True)
    header.write_text("local work")
    with pytest.raises(ValueError, match="existing header"):
        host.prepare(tmp_path, tmp_path / "output")
    assert header.read_text() == "local work"


def test_ci_requires_pinned_native_host_execution():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/mlx-metal-host.yml").read_text()
    assert host.MLX_COMMIT in workflow
    assert "runs-on: macos-26" in workflow
    assert 'python-version: "3.13"' in workflow
    assert "-DMLX_METAL_JIT=ON" in workflow
    assert "run_mlx_metal_host.py prepare" in workflow
    assert "run_mlx_metal_host.py verify" in workflow
    assert "--timeout-seconds 3300" in workflow
    assert '--label "MLX translated host verification"' in workflow
    assert "pytest -q -n auto tests/test_mlx_metal_host.py" in workflow
    assert "if: always()" in workflow
    assert "continue-on-error" not in workflow


@pytest.mark.parametrize("patched", [False, True])
def test_checkout_identity(tmp_path, monkeypatch, patched):
    path = tmp_path / "device.cpp"
    path.write_text("pinned adapter" if patched else "pinned original")
    expected = host.digest(path)
    monkeypatch.setattr(host, "SOURCE_HASHES", {"device.cpp": (expected, expected)})
    header = tmp_path / host.HEADER
    if patched:
        header.parent.mkdir(parents=True)
        header.write_bytes((host.HERE / "metal_library_overrides.h").read_bytes())
    monkeypatch.setattr(
        host.subprocess,
        "check_output",
        lambda command, **kwargs: (
            host.MLX_COMMIT
            if "rev-parse" in command
            else ("device.cpp\n" if patched else "")
        ),
    )
    host.verify_checkout(tmp_path, patched=patched)
    path.write_text("unexpected changes")
    with pytest.raises(ValueError, match="runtime source"):
        host.verify_checkout(tmp_path, patched=patched)


@pytest.mark.parametrize(
    "revision,modified", [("bad", ""), (host.MLX_COMMIT, "test_ops.py\n")]
)
def test_checkout_rejects_wrong_pin_or_unrelated_edits(
    tmp_path, monkeypatch, revision, modified
):
    monkeypatch.setattr(
        host.subprocess,
        "check_output",
        lambda command, **kwargs: revision if "rev-parse" in command else modified,
    )
    with pytest.raises(ValueError):
        host.verify_checkout(tmp_path, patched=False)


def test_complete_host_evidence_keeps_command_and_numerical_records_separate(
    tmp_path, monkeypatch
):
    root, output = tmp_path / "mlx", tmp_path / "proof"
    test_source = root / "python/tests/test_ops.py"
    test_source.parent.mkdir(parents=True)
    test_source.write_text("unchanged upstream test")
    monkeypatch.setattr(host, "verify_checkout", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        host, "compile_libraries", lambda *args: (output / "libraries", [])
    )
    shapes = ("ss", "sv", "vs", "vv", "g1", "g2", "g3", "gn2")

    def fake_run(command, directory, name, **kwargs):
        result = {"returncode": 0, "command": list(map(str, command))}
        env = kwargs["env"]
        if name == "runtime-identity":
            host.save_json(
                directory / f"{name}.stdout",
                {
                    "module": str(root / "core.so"),
                    "device": "gpu",
                    "metalAvailable": True,
                },
            )
        if name.startswith("upstream-"):
            (directory / f"{name}.stderr").write_text("Ran 163 tests in 1.0s\nOK\n")
        if name == "upstream-translated" or name == "execute-host-workloads":
            selected = ("ss",) if name == "upstream-translated" else shapes
            Path(env["CROSTL_METAL_LIBRARY_TRACE"]).write_text(
                "".join(
                    f"library\t{s}_Powercomplex64\ndispatch\t{s}_Powercomplex64\tthreads\t1,1,1\t1,1,1\n"
                    for s in selected
                )
            )
        if name == "execute-host-workloads":
            assert command[-1] != directory / (name + ".json")
            host.save_json(command[-1], [{"shape": s, "matched": True} for s in shapes])
        if name == "missing-required-library":
            result["returncode"] = 1
            (directory / f"{name}.stderr").write_text(
                "Cannot load required translated Metal library"
            )
        host.save_json(directory / f"{name}.json", result)
        return result

    monkeypatch.setattr(host, "run", fake_run)
    host.verify(root, Path(sys.executable), output)
    evidence = json.loads((output / "evidence.json").read_text())
    assert len(evidence["hostWorkloads"]) == 8
    assert evidence["missingRequiredLibraryRejected"] is True
    assert evidence["fullTranslatedBackend"] is False
