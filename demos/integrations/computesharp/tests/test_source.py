"""Check immutable upstream input receipts independently of shader execution."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "computesharp_source", Path(__file__).resolve().parents[1] / "source.py"
)
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)


def _git(root, *arguments):
    subprocess.run(["git", "-C", str(root), *arguments], check=True)


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "input.bin").write_bytes(b"\0immutable input\xff")
    _git(root, "add", "input.bin")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "Record test input",
    )
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setattr(SOURCE, "PIN", commit)
    return root


def test_snapshot_covers_tracked_bytes_and_ignores_untracked_build_output(repository):
    first = SOURCE.snapshot(repository)
    assert set(first["files"]) == {"input.bin"}
    assert first["commit"] == SOURCE.PIN
    (repository / "compiler-output.bin").write_bytes(b"build output")
    assert SOURCE.snapshot(repository) == first


def test_rejects_the_wrong_upstream_revision(repository, monkeypatch):
    monkeypatch.setattr(SOURCE, "PIN", "0" * 40)
    with pytest.raises(ValueError, match="revision"):
        SOURCE.snapshot(repository)


@pytest.mark.parametrize("change", ["modify", "delete", "add"])
def test_rejects_changes_to_tracked_inputs(repository, change):
    if change == "modify":
        (repository / "input.bin").write_bytes(b"changed input")
    elif change == "delete":
        (repository / "input.bin").unlink()
    else:
        (repository / "additional.bin").write_bytes(b"new input")
        _git(repository, "add", "additional.bin")
    with pytest.raises(ValueError, match="unchanged tracked"):
        SOURCE.snapshot(repository)


def test_rejects_a_subdirectory_in_place_of_the_source_root(repository):
    nested = repository / "nested"
    nested.mkdir()
    with pytest.raises(ValueError, match="repository root"):
        SOURCE.snapshot(nested)
