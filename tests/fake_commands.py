"""Fake commands for tests that run workflow steps in a shell."""

import shlex
import sys
from pathlib import Path


def write_python_command(path: Path, source: str) -> None:
    """Make ``path`` a command that runs the Python ``source`` with the interpreter running the tests.

    A ``#!`` line cannot quote its interpreter, so it breaks when the path to Python has a space, as it does for a
    virtual environment under ``My Projects``. ``path`` is a shell script instead, which runs ``source`` from
    ``<path>.py``; the source sees that file as ``sys.argv[0]``.
    """
    script = path.with_name(f"{path.name}.py")
    script.write_text(source, encoding="utf-8")
    path.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(script))} "$@"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)
