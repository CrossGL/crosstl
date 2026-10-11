"""Checkpoint reuse requires the same installed translator implementation."""

import os
import shutil
from pathlib import Path

import pytest

import crosstl.project.translation_checkpoint as checkpoint


def _package(tmp_path, monkeypatch):
    root = tmp_path / "crosstl"
    for name in (
        "__init__.py",
        "_crosstl.py",
        "backend/Metal/preprocessor.py",
        "translator/codegen/GLSL_codegen.py",
        "project/pipeline.py",
        "project/translation_checkpoint.py",
        "project/directx_runtime_worker.cpp",
        "project/metal_runtime_worker.swift",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"first line\nsecond line\n")
    monkeypatch.setattr(
        checkpoint, "__file__", str(root / "project/translation_checkpoint.py")
    )
    return root


@pytest.mark.parametrize(
    "relative",
    (
        "_crosstl.py",
        "backend/Metal/preprocessor.py",
        "translator/codegen/GLSL_codegen.py",
        "project/pipeline.py",
        "project/directx_runtime_worker.cpp",
        "project/metal_runtime_worker.swift",
    ),
)
def test_implementation_identity_detects_same_size_same_timestamp_edits(
    tmp_path, monkeypatch, relative
):
    root = _package(tmp_path, monkeypatch)
    before = checkpoint.project_translation_implementation_identity()
    path = root / relative
    stat = path.stat()
    path.write_bytes(path.read_bytes().replace(b"first", b"other"))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = checkpoint.project_translation_implementation_identity()
    assert before != after
    assert before["fileCount"] == after["fileCount"] == 8
    assert before["algorithm"] == after["algorithm"] == "sha256"


@pytest.mark.parametrize("change", ("add", "remove", "rename"))
def test_implementation_identity_covers_source_inventory(tmp_path, monkeypatch, change):
    root = _package(tmp_path, monkeypatch)
    before = checkpoint.project_translation_implementation_identity()
    source = root / "_crosstl.py"
    if change == "add":
        (root / "extra.py").write_bytes(source.read_bytes())
    elif change == "remove":
        source.unlink()
    else:
        source.rename(root / "renamed.py")
    assert checkpoint.project_translation_implementation_identity() != before


def test_implementation_identity_is_portable_across_paths_and_text_newlines(
    tmp_path, monkeypatch
):
    root = _package(tmp_path, monkeypatch)
    before = checkpoint.project_translation_implementation_identity()
    relocated = tmp_path / "site-packages" / "crosstl"
    shutil.copytree(root, relocated)
    for path in relocated.rglob("*"):
        if path.is_file():
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    monkeypatch.setattr(
        checkpoint, "__file__", str(relocated / "project/translation_checkpoint.py")
    )
    assert checkpoint.project_translation_implementation_identity() == before
    for name in (
        "__pycache__/cached.pyc",
        "__pycache__/cached.py",
        ".local/scratch.py",
        "notes.txt",
    ):
        path = relocated / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not installed implementation source")
    assert checkpoint.project_translation_implementation_identity() == before


def test_implementation_identity_rejects_missing_source_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(
        checkpoint, "__file__", str(tmp_path / "missing/project/checkpoint.py")
    )
    with pytest.raises(checkpoint.ProjectTranslationCheckpointError) as caught:
        checkpoint.project_translation_implementation_identity()
    assert caught.value.reason == "implementation-unavailable"


def test_implementation_identity_does_not_ignore_unreadable_source(
    tmp_path, monkeypatch
):
    root = _package(tmp_path, monkeypatch)
    original = Path.read_bytes

    def unreadable(path):
        if path == root / "_crosstl.py":
            raise PermissionError("source is unreadable")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    with pytest.raises(checkpoint.ProjectTranslationCheckpointError) as caught:
        checkpoint.project_translation_implementation_identity()
    assert caught.value.reason == "implementation-unavailable"
