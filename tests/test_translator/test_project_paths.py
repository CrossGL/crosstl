"""Resolved project paths preserve containment across Windows path namespaces."""

from pathlib import PurePosixPath, PureWindowsPath
from unittest.mock import Mock

import pytest

from crosstl.project.pipeline import _is_relative_to, _relpath


@pytest.mark.parametrize(
    "path,root,expected",
    [
        (r"\\?\C:\project\out\kernel.glsl", r"C:\project", "out/kernel.glsl"),
        (r"C:\project\out\kernel.glsl", r"\\?\C:\project", "out/kernel.glsl"),
        (r"\\?\C:\Project\out", r"c:\project", "out"),
        (r"\\?\C:\project", r"C:\project", "."),
        (r"\\?\UNC\server\share\project\out", r"\\server\share\project", "out"),
        (r"\\server\share\project\out", r"\\?\UNC\server\share\project", "out"),
        (r"\\?\unc\SERVER\SHARE\project\out", r"\\server\share\project", "out"),
        (r"\\?\C:\project\out", r"\\?\C:\project", "out"),
        (r"\\?\UNC\server\share\out", r"\\?\UNC\server\share", "out"),
        (r"C:\project\out", r"C:\project", "out"),
        (r"\\?\Volume{123}\project\out", r"\\?\Volume{123}\project", "out"),
    ],
)
def test_resolved_windows_namespace_paths_are_relative(path, root, expected):
    candidate = Mock(resolve=Mock(return_value=PureWindowsPath(path)))
    project = Mock(resolve=Mock(return_value=PureWindowsPath(root)))

    assert _relpath(candidate, project) == expected
    assert _is_relative_to(candidate, project)
    assert candidate.resolve.call_count == 2
    assert project.resolve.call_count == 2


@pytest.mark.parametrize(
    "path,root",
    [
        (r"\\?\D:\project\out", r"C:\project"),
        (r"\\?\C:\outside\out", r"C:\project"),
        (r"\\?\C:\project-other\out", r"C:\project"),
        (r"\\?\UNC\server\other\out", r"\\server\share"),
        (r"\\?\UNC\other\share\out", r"\\server\share"),
        (r"\\?\UNC\server\share\outside", r"\\server\share\project"),
        (r"\\?\Volume{123}\project\out", r"C:\project"),
        (r"\\.\C:\project\out", r"C:\project"),
    ],
)
def test_resolved_windows_paths_outside_project_are_rejected(path, root):
    candidate = Mock(resolve=Mock(return_value=PureWindowsPath(path)))
    project = Mock(resolve=Mock(return_value=PureWindowsPath(root)))

    assert not _is_relative_to(candidate, project)
    with pytest.raises(ValueError):
        _relpath(candidate, project)


def test_resolved_posix_path_preserves_literal_windows_namespace():
    path = PurePosixPath(r"/project/\\?\C:\out")
    candidate = Mock(resolve=Mock(return_value=path))
    project = Mock(resolve=Mock(return_value=PurePosixPath("/project")))

    assert _relpath(candidate, project) == r"\\?\C:\out"
    assert _is_relative_to(candidate, project)


@pytest.mark.parametrize("outside", [False, True])
def test_project_path_containment_resolves_symlinks(tmp_path, outside):
    root = tmp_path / "project"
    root.mkdir()
    target = (tmp_path if outside else root) / "target"
    target.mkdir()
    link = root / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlink creation unavailable: {exc}")

    candidate = link / "kernel.glsl"
    assert _is_relative_to(candidate, root) is not outside
    if outside:
        with pytest.raises(ValueError):
            _relpath(candidate, root)
    else:
        assert _relpath(candidate, root) == "target/kernel.glsl"


@pytest.mark.parametrize(
    "error", [OSError("inaccessible"), RuntimeError("symlink loop")]
)
def test_project_path_resolution_errors_are_not_treated_as_relative(error):
    candidate = Mock(resolve=Mock(side_effect=error))
    project = Mock()

    with pytest.raises(type(error), match=str(error)):
        _relpath(candidate, project)
    with pytest.raises(type(error), match=str(error)):
        _is_relative_to(candidate, project)
