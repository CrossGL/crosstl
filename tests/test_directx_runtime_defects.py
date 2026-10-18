"""Contracts for the pinned WARP runtime-defect registry."""

import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests import directx_runtime_defects as defects

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/demo-project-testing.yml"


def _item(nodeid):
    item = SimpleNamespace(nodeid=nodeid, markers=[])
    item.add_marker = item.markers.append
    return item


def _portable_host_job():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["portable-host"]


def test_registry_names_both_pinned_libraries_of_the_native_host_job():
    job = _portable_host_job()
    identity = next(
        step for step in job["steps"] if step.get("id") == "host-warp-identity"
    )
    expected = identity["env"]["EXPECTED_WARP_SHA256"]
    legs = [
        row
        for row in job["strategy"]["matrix"]["include"]
        if row["os"].startswith("windows")
    ]
    assert sorted(row.get("warp_qualification", "1.0.21") for row in legs) == [
        "1.0.13",
        "1.0.21",
    ]
    for digest, version in defects.PINNED_WARP_LIBRARIES.items():
        assert digest in expected, version
    for defect in defects.DEFECTS:
        assert defect.defective in defects.PINNED_WARP_LIBRARIES
        assert defect.conformant in defects.PINNED_WARP_LIBRARIES
        assert defect.defective != defect.conformant
        assert defect.issue > 0 and defect.summary and defect.tests


def test_registered_tests_run_on_both_native_host_legs():
    # Every leg of the native host job runs the same steps, so the conformant
    # library executes each registered test without the expected-failure marker.
    steps = [
        step
        for step in _portable_host_job()["steps"]
        if "warp_qualification" not in str(step.get("if", ""))
    ]
    commands = "\n".join(str(step.get("run", "")) for step in steps)
    for defect in defects.DEFECTS:
        for node in defect.tests:
            path, function = node.split("[", 1)[0].split("::")
            assert path in commands, node
            assert f"def {function}(" in (ROOT / path).read_text(encoding="utf-8")


def test_registered_test_nodes_exist():
    nodes = [node for defect in defects.DEFECTS for node in defect.tests]
    assert len(nodes) == len(set(nodes))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *nodes,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    collected = {line.strip() for line in result.stdout.splitlines()}
    assert set(nodes) <= collected


@pytest.mark.parametrize("defect", defects.DEFECTS, ids=lambda d: f"issue{d.issue}")
def test_only_the_defective_library_marks_its_tests(defect):
    other = _item("tests/test_example.py::test_unrelated")
    items = [_item(node) for node in defect.tests] + [other]
    defects.mark_known_runtime_defects(items, digest=defect.conformant)
    assert all(not item.markers for item in items)

    defects.mark_known_runtime_defects(items, digest=defect.defective)
    assert not other.markers
    for item in items[:-1]:
        (marker,) = item.markers
        assert marker.name == "xfail"
        assert marker.kwargs["strict"] is True
        assert marker.kwargs["raises"] is AssertionError
        assert f"#{defect.issue}" in marker.kwargs["reason"]


def test_unknown_or_missing_library_marks_nothing():
    items = [_item(node) for defect in defects.DEFECTS for node in defect.tests]
    defects.mark_known_runtime_defects(items, digest="0" * 64)
    assert all(not item.markers for item in items)
    assert defects.expected_failures(None) == {}


def test_loaded_library_is_hashed_beside_the_interpreter(tmp_path, monkeypatch):
    interpreter = tmp_path / "python.exe"
    interpreter.write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(interpreter))
    defects.loaded_warp_sha256.cache_clear()
    try:
        monkeypatch.setattr(sys, "platform", "linux")
        assert defects.loaded_warp_sha256() is None
        defects.loaded_warp_sha256.cache_clear()

        monkeypatch.setattr(sys, "platform", "win32")
        assert defects.loaded_warp_sha256() is None
        defects.loaded_warp_sha256.cache_clear()

        payload = b"warp library"
        (tmp_path / "d3d10warp.dll").write_bytes(payload)
        assert defects.loaded_warp_sha256() == hashlib.sha256(payload).hexdigest()
    finally:
        defects.loaded_warp_sha256.cache_clear()
