"""The translator-source identity reuses settled digests and sees every edit."""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from crosstl.translator import dispatch_region_identity as identity

SETTLED = time.time_ns() - 3600 * 1_000_000_000


def _reference(root):
    """The identity as a full rglob reread computes it."""
    files = {
        path.relative_to(root).as_posix(): (
            hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        )
        for path in sorted(root.rglob("*.py"))
    }
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write(path, text, mtime_ns=SETTLED):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mtime_ns is not None:
        os.utime(path, ns=(mtime_ns, mtime_ns))


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "package"
    _write(root / "a.py", "A = 1\n")
    _write(root / "nested" / "b.py", "B = 2\r\n")
    _write(root / "nested" / "deeper" / "c.py", "C = 3\n")
    _write(root / "notes.txt", "not source\n")
    _write(root / "nested" / "d.pyc", "not source\n")
    return root


def test_matches_a_full_reread(tree):
    assert identity.source_tree_hash(tree) == _reference(tree)


def test_the_installed_translator_matches_a_full_reread():
    root = Path(identity.__file__).resolve().parents[1]
    assert identity.translation_implementation_hash() == _reference(root)


def test_settled_files_are_not_reread(tree, monkeypatch):
    first = identity.source_tree_hash(tree)

    def refuse(*args, **kwargs):
        raise AssertionError("A settled, unchanged source was reread")

    monkeypatch.setattr(identity, "open", refuse, raising=False)
    assert identity.source_tree_hash(tree) == first


@pytest.mark.parametrize("change", ["content", "same-size", "added", "removed"])
def test_every_edit_changes_the_identity(tree, change):
    before = identity.source_tree_hash(tree)
    if change == "content":
        _write(tree / "a.py", "A = 10\n")
    elif change == "same-size":
        # Same length; only the modification time distinguishes the read.
        _write(tree / "a.py", "A = 9\n", mtime_ns=SETTLED + 1_000_000_000)
    elif change == "added":
        _write(tree / "nested" / "e.py", "E = 5\n")
    else:
        (tree / "nested" / "deeper" / "c.py").unlink()
    after = identity.source_tree_hash(tree)
    assert after != before
    assert after == _reference(tree)


def test_recent_files_are_reread_even_with_an_unchanged_signature(tree):
    recent = time.time_ns()
    path = tree / "a.py"
    _write(path, "A = 1\n", mtime_ns=recent)
    before = identity.source_tree_hash(tree)
    # An edit within the timestamp's granularity keeps size and time.
    _write(path, "A = 7\n", mtime_ns=recent)
    after = identity.source_tree_hash(tree)
    assert after != before
    assert after == _reference(tree)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symbolic links")
def test_linked_directories_are_not_walked(tree, tmp_path):
    outside = tmp_path / "outside"
    _write(outside / "x.py", "X = 1\n")
    try:
        (tree / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are not permitted here")
    assert identity.source_tree_hash(tree) == _reference(tree)
