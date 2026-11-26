import subprocess

import pytest

from demos.integrations.mlx import fetch_upstream
from demos.integrations.mlx.fetch_upstream import FetchError, fetch, main, verify

CRLF_BYTES = b"first line\r\nsecond line\r\n"
LF_BYTES = b"kernel void k() {}\n"


def _run(*arguments, cwd):
    return subprocess.run(
        ["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def git_home(tmp_path, monkeypatch):
    """Isolate Git from the user's configuration and turn autocrlf on globally.

    Git for Windows installs ``core.autocrlf=true``; the checkout must still
    write the stored bytes.
    """
    home = tmp_path / "home"
    home.mkdir()
    config = home / ".gitconfig"
    config.write_text(
        "[core]\n\tautocrlf = true\n"
        "[user]\n\tname = Test\n\temail = test@example.invalid\n"
        "[init]\n\tdefaultBranch = main\n"
        '[protocol "file"]\n\tallow = always\n'
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    return home


@pytest.fixture
def upstream(tmp_path, git_home):
    """A local repository with two commits; the first one is the pin."""
    root = tmp_path / "upstream"
    (root / "mlx/backend/metal/kernels").mkdir(parents=True)
    (root / "python/tests").mkdir(parents=True)
    (root / "mlx/backend/metal/kernels/crlf.h").write_bytes(CRLF_BYTES)
    (root / "mlx/backend/metal/kernels/unary.metal").write_bytes(LF_BYTES)
    (root / "python/tests/test_ops.py").write_bytes(LF_BYTES)
    (root / "CMakeLists.txt").write_bytes(LF_BYTES)
    _run("init", "--quiet", cwd=root)
    _run("config", "core.autocrlf", "false", cwd=root)
    _run("config", "uploadpack.allowAnySHA1InWant", "true", cwd=root)
    _run("config", "uploadpack.allowFilter", "true", cwd=root)
    _run("add", ".", cwd=root)
    _run("commit", "--quiet", "-m", "pin", cwd=root)
    pin = _run("rev-parse", "HEAD", cwd=root)
    (root / "python/tests/test_ops.py").write_bytes(b"changed later\n")
    _run("commit", "--quiet", "-am", "later", cwd=root)
    return root.as_uri(), pin


def _quiet(_message):
    pass


def test_fetch_writes_stored_bytes_at_the_pinned_commit(tmp_path, upstream):
    remote, pin = upstream
    destination = tmp_path / "mlx-upstream"

    fetch(destination, pin, remote=remote, log=_quiet)

    kernels = destination / "mlx/backend/metal/kernels"
    assert (kernels / "crlf.h").read_bytes() == CRLF_BYTES
    assert (kernels / "unary.metal").read_bytes() == LF_BYTES
    assert (destination / "python/tests/test_ops.py").read_bytes() == LF_BYTES
    assert _run("rev-parse", "HEAD", cwd=destination) == pin
    assert _run("rev-parse", "--is-shallow-repository", cwd=destination) == "true"


def test_sparse_fetch_checks_out_only_the_named_directories(tmp_path, upstream):
    remote, pin = upstream
    destination = tmp_path / "mlx-current-upstream"

    fetch(destination, pin, ("mlx/backend/metal/kernels",), remote=remote, log=_quiet)

    assert (destination / "mlx/backend/metal/kernels/crlf.h").read_bytes() == (
        CRLF_BYTES
    )
    assert not (destination / "python").exists()
    assert _run("rev-parse", "HEAD", cwd=destination) == pin
    written = sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(destination).parts
    )
    assert written == [
        "CMakeLists.txt",
        "mlx/backend/metal/kernels/crlf.h",
        "mlx/backend/metal/kernels/unary.metal",
    ]


def test_fetch_replaces_an_existing_directory(tmp_path, upstream):
    remote, pin = upstream
    destination = tmp_path / "mlx-upstream"
    destination.mkdir()
    (destination / "stale.txt").write_text("left by an earlier step\n")

    fetch(destination, pin, remote=remote, log=_quiet)

    assert not (destination / "stale.txt").exists()


def _flaky_git(failures):
    """Wrap the real runner so the first ``failures`` fetches fail."""
    calls = {"fetch": 0}

    def git(*arguments, cwd=None):
        if arguments[0] == "fetch":
            calls["fetch"] += 1
            if calls["fetch"] <= failures:
                # A failed attempt may leave files behind; the next must not
                # see them.
                (cwd / "partial.pack").write_text("half written\n")
                raise FetchError("fatal: unable to access: Could not resolve host")
        return fetch_upstream._git(*arguments, cwd=cwd)

    return git, calls


def test_fetch_retries_from_an_empty_directory_with_growing_delays(tmp_path, upstream):
    remote, pin = upstream
    destination = tmp_path / "mlx-upstream"
    git, calls = _flaky_git(failures=2)
    sleeps = []

    fetch(destination, pin, remote=remote, git=git, sleep=sleeps.append, log=_quiet)

    assert calls["fetch"] == 3
    assert sleeps == [15, 30]
    assert _run("rev-parse", "HEAD", cwd=destination) == pin
    assert not (destination / "partial.pack").exists()


def test_fetch_gives_up_after_the_last_attempt(tmp_path, upstream):
    remote, pin = upstream
    git, calls = _flaky_git(failures=99)
    sleeps = []

    with pytest.raises(FetchError, match="4 attempts failed; last: .*resolve host"):
        fetch(
            tmp_path / "mlx-upstream",
            pin,
            remote=remote,
            git=git,
            sleep=sleeps.append,
            log=_quiet,
        )

    assert calls["fetch"] == 4
    assert sleeps == [15, 30, 60]


def test_verify_rejects_a_changed_or_untracked_file(tmp_path, upstream):
    remote, pin = upstream
    destination = tmp_path / "mlx-upstream"
    fetch(destination, pin, remote=remote, log=_quiet)

    (destination / "extra.metal").write_bytes(LF_BYTES)
    with pytest.raises(FetchError, match="not clean"):
        verify(destination, pin)

    (destination / "extra.metal").unlink()
    (destination / "CMakeLists.txt").write_bytes(b"edited\n")
    with pytest.raises(FetchError, match="not clean"):
        verify(destination, pin)


def test_verify_rejects_another_commit_and_autocrlf(tmp_path, upstream):
    remote, pin = upstream
    destination = tmp_path / "mlx-upstream"
    fetch(destination, pin, remote=remote, log=_quiet)

    with pytest.raises(FetchError, match="expected " + "0" * 40):
        verify(destination, "0" * 40)

    _run("config", "core.autocrlf", "true", cwd=destination)
    with pytest.raises(FetchError, match="autocrlf"):
        verify(destination, pin)


@pytest.mark.parametrize(
    "commit", ["9c3d355", "9C3D35571AC450A8ECF5C17B4D0E3FAC52C08BC8", "main"]
)
def test_fetch_requires_a_full_object_name(tmp_path, commit):
    with pytest.raises(FetchError, match="full 40-digit"):
        fetch(tmp_path / "mlx-upstream", commit, log=_quiet)
    assert not (tmp_path / "mlx-upstream").exists()


def test_main_reports_failure(tmp_path, git_home, capsys):
    missing = (tmp_path / "missing").as_uri()

    status = main(
        [
            str(tmp_path / "mlx-upstream"),
            "1" * 40,
            "--remote",
            missing,
            "--attempts",
            "1",
        ]
    )

    assert status == 1
    assert "1 attempts failed" in capsys.readouterr().err
