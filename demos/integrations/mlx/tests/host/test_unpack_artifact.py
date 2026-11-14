import zipfile

import pytest

from demos.integrations.mlx.portable_host.unpack_artifact import (
    ArtifactArchiveError,
    main,
    unpack,
)


def _archive(path, members):
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return path


def test_unpack_restores_the_artifact_tree_and_removes_the_archive(tmp_path, capsys):
    base = tmp_path / "base"
    base.mkdir()
    _archive(
        base / "mlx-portable-host-directx-1.zip",
        {
            "adaptation.json": "{}",
            ".hidden/marker": "x",
            "wheel/mlx.whl": b"\x00\x01",
        },
    )

    assert main([str(base)]) == 0

    assert sorted(p.relative_to(base).as_posix() for p in base.rglob("*")) == [
        ".hidden",
        ".hidden/marker",
        "adaptation.json",
        "wheel",
        "wheel/mlx.whl",
    ]
    assert (base / "wheel/mlx.whl").read_bytes() == b"\x00\x01"
    assert "Unpacked 3 files" in capsys.readouterr().out


@pytest.mark.parametrize(
    "setup",
    ["missing", "empty", "two-files", "directory", "not-zip", "empty-zip"],
)
def test_unpack_requires_exactly_one_zip_archive(tmp_path, setup):
    base = tmp_path / "base"
    if setup != "missing":
        base.mkdir()
    if setup == "two-files":
        _archive(base / "a.zip", {"a": "1"})
        (base / "b.txt").write_text("b")
    elif setup == "directory":
        (base / "nested").mkdir()
    elif setup == "not-zip":
        (base / "artifact").write_text("plain")
    elif setup == "empty-zip":
        _archive(base / "artifact.zip", {})

    with pytest.raises(ArtifactArchiveError):
        unpack(base)


@pytest.mark.parametrize("member", ["../escape", "/abs/escape", "a/../../escape"])
def test_unpack_refuses_members_outside_the_directory(tmp_path, member):
    base = tmp_path / "base"
    base.mkdir()
    archive = _archive(base / "artifact.zip", {"ok": "1", member: "2"})

    with pytest.raises(ArtifactArchiveError, match="escapes"):
        unpack(base)

    assert archive.exists()
    assert not (base / "ok").exists()
    assert not (tmp_path / "escape").exists()


def test_unpack_refuses_a_member_that_overwrites_the_archive(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    archive = _archive(base / "artifact.zip", {"artifact.zip": "x"})

    with pytest.raises(ArtifactArchiveError, match="overwrites"):
        unpack(base)

    assert zipfile.is_zipfile(archive)
