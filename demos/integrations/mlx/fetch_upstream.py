"""Check out one pinned MLX commit, retrying transient network failures.

Every MLX workflow job reads upstream MLX sources at an exact commit. A clone
that fails on a dropped connection or a name-resolution error used to fail the
job, and in the merge queue it removed the whole group. This script makes one
shallow fetch of the commit, retries it from an empty directory a few times
with growing delays, and then checks that the result is exactly that commit:

- ``core.autocrlf`` is off, so every file has the bytes stored in Git on every
  operating system;
- ``HEAD`` is the requested commit, given as a full 40-digit object name;
- the working tree has no changes and no untracked files.

``--sparse`` limits the working tree to the named directories (Git's cone
mode); the fetch then downloads only the file contents those directories need.
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

UPSTREAM = "https://github.com/ml-explore/mlx.git"
ATTEMPTS = 4
FIRST_DELAY = 15
COMMAND_TIMEOUT = 900
_COMMIT = re.compile(r"[0-9a-f]{40}")


class FetchError(RuntimeError):
    """A checkout that failed or does not match the requested commit."""


def _git(*arguments, cwd=None):
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        raise FetchError(
            f"git {' '.join(arguments)} exited {completed.returncode}: "
            + " | ".join(detail[-3:])
        )
    return completed.stdout


def _remove(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def checkout_once(destination, commit, sparse=(), remote=UPSTREAM, git=_git):
    """Make one attempt: an empty repository, one fetch, one checkout."""
    _remove(destination)
    destination.mkdir(parents=True)
    run = lambda *arguments: git(*arguments, cwd=destination)  # noqa: E731
    run("init", "--quiet")
    run("config", "core.autocrlf", "false")
    run("config", "advice.detachedHead", "false")
    run("remote", "add", "origin", remote)
    fetch = ["fetch", "--quiet", "--depth=1"]
    if sparse:
        run("sparse-checkout", "set", "--cone", *sparse)
        fetch.append("--filter=blob:none")
    run(*fetch, "origin", commit)
    run("checkout", "--quiet", "--detach", "FETCH_HEAD")


def verify(destination, commit, git=_git):
    """Fail unless the checkout is exactly ``commit`` with a clean tree."""
    run = lambda *arguments: git(*arguments, cwd=destination)  # noqa: E731
    if run("config", "--get", "core.autocrlf").strip() != "false":
        raise FetchError(f"{destination}: core.autocrlf is not false")
    head = run("rev-parse", "HEAD").strip()
    if head != commit:
        raise FetchError(f"{destination}: HEAD is {head}, expected {commit}")
    changes = run("status", "--porcelain=v1", "--untracked-files=all")
    if changes.strip():
        first = changes.strip().splitlines()[0]
        raise FetchError(f"{destination}: working tree is not clean ({first})")


def fetch(
    destination,
    commit,
    sparse=(),
    *,
    remote=UPSTREAM,
    attempts=ATTEMPTS,
    first_delay=FIRST_DELAY,
    git=_git,
    sleep=time.sleep,
    log=print,
):
    """Check out ``commit`` into ``destination``, retrying failed attempts.

    A failed attempt starts again from an empty directory, so a half-written
    checkout never survives into the next attempt. The delay doubles after
    each failure. The final checkout is verified once; a mismatch is not
    retried, because fetching the same commit again cannot change it.
    """
    if not _COMMIT.fullmatch(commit):
        raise FetchError(f"commit must be a full 40-digit object name: {commit!r}")
    if attempts < 1:
        raise FetchError("attempts must be at least 1")
    destination = Path(destination)
    delay = first_delay
    for attempt in range(1, attempts + 1):
        try:
            checkout_once(destination, commit, sparse, remote, git)
            break
        except (FetchError, OSError, subprocess.TimeoutExpired) as error:
            if attempt == attempts:
                raise FetchError(
                    f"{destination}: {attempts} attempts failed; last: {error}"
                ) from error
            log(f"attempt {attempt} of {attempts} failed: {error}; retry in {delay} s")
            sleep(delay)
            delay *= 2
    verify(destination, commit, git)
    log(f"{destination}: {commit}" + (f" ({' '.join(sparse)})" if sparse else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("destination", type=Path)
    parser.add_argument("commit")
    parser.add_argument(
        "--sparse",
        nargs="+",
        default=(),
        metavar="DIRECTORY",
        help="check out only these directories (cone mode)",
    )
    parser.add_argument("--remote", default=UPSTREAM)
    parser.add_argument("--attempts", type=int, default=ATTEMPTS)
    args = parser.parse_args(argv)
    try:
        fetch(
            args.destination,
            args.commit,
            tuple(args.sparse),
            remote=args.remote,
            attempts=args.attempts,
        )
    except FetchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
