from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from demos.integrations.mlx.tests import pins

ROOT = Path(__file__).resolve().parents[4]

DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64


def _artifact(value=DIGEST, size=128):
    return {
        "path": "out/directx/kernel.hlsl",
        "generatedHash": {"algorithm": "sha256", "value": value},
        "generatedSizeBytes": size,
    }


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    monkeypatch.delenv(pins.PIN_MODE_ENV, raising=False)
    monkeypatch.delenv(pins.PIN_DRIFT_DIR_ENV, raising=False)


def test_pins_are_strict_by_default():
    assert pins.pin_mode() == pins.STRICT_PIN_MODE
    assert pins.check_generated_artifact(_artifact(), sha256=DIGEST, size_bytes=128)
    with pytest.raises(AssertionError, match="pinned generatedHash changed"):
        pins.check_generated_artifact(_artifact(), sha256=OTHER_DIGEST)
    with pytest.raises(AssertionError, match="pinned generatedSizeBytes changed"):
        pins.check_generated_artifact(_artifact(), sha256=DIGEST, size_bytes=129)


def test_unknown_pin_mode_is_a_configuration_error(monkeypatch):
    monkeypatch.setenv(pins.PIN_MODE_ENV, "ignore")
    with pytest.raises(ValueError, match="must be one of strict, report"):
        pins.pin_mode()


def test_report_mode_records_drift_and_continues(monkeypatch, tmp_path):
    monkeypatch.setenv(pins.PIN_MODE_ENV, pins.REPORT_PIN_MODE)
    monkeypatch.setenv(pins.PIN_DRIFT_DIR_ENV, str(tmp_path / "drift"))

    with pytest.warns(pins.PinDriftWarning) as recorded:
        matches = pins.check_generated_artifact(
            _artifact(size=130), sha256=OTHER_DIGEST, size_bytes=128
        )

    assert matches is False
    assert len(recorded) == 2
    (log,) = (tmp_path / "drift").glob("drift-*.jsonl")
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [
        (record["field"], record["expected"], record["actual"]) for record in records
    ] == [
        ("generatedHash", OTHER_DIGEST, DIGEST),
        ("generatedSizeBytes", 128, 130),
    ]
    assert {record["subject"] for record in records} == {"out/directx/kernel.hlsl"}
    assert all("test_report_mode_records_drift" in record["test"] for record in records)


def test_report_mode_keeps_matching_pins_silent(monkeypatch, recwarn):
    monkeypatch.setenv(pins.PIN_MODE_ENV, pins.REPORT_PIN_MODE)
    assert pins.check_generated_artifact(_artifact(), sha256=DIGEST, size_bytes=128)
    assert not [w for w in recwarn if issubclass(w.category, pins.PinDriftWarning)]


@pytest.mark.parametrize(
    "artifact",
    [
        {"path": "a.hlsl"},
        {"path": "a.hlsl", "generatedHash": {"algorithm": "sha1", "value": DIGEST}},
        {"path": "a.hlsl", "generatedHash": {"algorithm": "sha256", "value": "abc"}},
        {
            "path": "a.hlsl",
            "generatedHash": {"algorithm": "sha256", "value": DIGEST, "extra": 1},
        },
        {
            "path": "a.hlsl",
            "generatedHash": {"algorithm": "sha256", "value": DIGEST},
            "generatedSizeBytes": 0,
        },
    ],
)
def test_report_mode_still_requires_a_well_formed_artifact_identity(
    monkeypatch, artifact
):
    monkeypatch.setenv(pins.PIN_MODE_ENV, pins.REPORT_PIN_MODE)
    with pytest.raises(AssertionError):
        pins.check_generated_artifact(artifact, sha256=DIGEST, size_bytes=128)


def test_drift_records_merge_worker_files_once_in_a_stable_order(tmp_path):
    first = {"subject": "b", "field": "generatedHash", "expected": "1", "actual": "2"}
    second = {"subject": "a", "field": "generatedHash", "expected": "3", "actual": "4"}
    (tmp_path / "drift-1.jsonl").write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8"
    )
    (tmp_path / "drift-2.jsonl").write_text(json.dumps(first) + "\n\n")
    (tmp_path / "other.jsonl").write_text(json.dumps({"ignored": True}) + "\n")

    assert pins.drift_records(tmp_path) == [second, first]


_DRIFTING_TESTS = """
import pytest

from demos.integrations.mlx.tests.pins import check_generated_artifact


@pytest.mark.parametrize("index", range(4))
def test_drifts(index):
    artifact = {
        "path": f"out/kernel{index}.hlsl",
        "generatedHash": {"algorithm": "sha256", "value": "a" * 64},
        "generatedSizeBytes": 10,
    }
    assert check_generated_artifact(artifact, sha256="b" * 64, size_bytes=10) is False
"""


@pytest.mark.parametrize("workers", [None, "2"])
def test_report_mode_lists_every_drifted_pin_after_the_session(tmp_path, workers):
    if workers:
        pytest.importorskip("xdist")
    (tmp_path / "test_drifting.py").write_text(_DRIFTING_TESTS, encoding="utf-8")
    summary = tmp_path / "step-summary.md"
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
        ),
        pins.PIN_MODE_ENV: pins.REPORT_PIN_MODE,
        "GITHUB_STEP_SUMMARY": str(summary),
    }
    environment.pop(pins.PIN_DRIFT_DIR_ENV, None)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "demos.integrations.mlx.tests.conftest",
        *(["-n", workers] if workers else []),
        "test_drifting.py",
    ]
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = [
        json.loads(line.split(" ", 1)[1])
        for line in completed.stdout.splitlines()
        if line.startswith("MLX-PIN-DRIFT ")
    ]
    assert sorted((record["subject"], record["field"]) for record in lines) == [
        (f"out/kernel{index}.hlsl", "generatedHash") for index in range(4)
    ]
    assert {record["actual"] for record in lines} == {"a" * 64}
    assert "4 passed" in completed.stdout
    assert "MLX generated-artifact pin drift: 4 pins" in summary.read_text()
