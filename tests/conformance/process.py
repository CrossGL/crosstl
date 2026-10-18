"""Bounded subprocesses; missing tools and timeouts are distinct outcomes."""

import os
import signal
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    command: tuple
    status: str
    returncode: object
    log: str

    def to_json(self):
        return asdict(self)


def run(command, *, timeout=60, env=None):
    command = tuple(str(value) for value in command)
    try:
        child = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=os.name != "nt",
        )
    except FileNotFoundError as error:
        return CommandResult(command, "unavailable", None, str(error))
    try:
        output, _ = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            child.kill()
        else:
            os.killpg(child.pid, signal.SIGKILL)
        output, _ = child.communicate()
        return CommandResult(
            command,
            "timeout",
            child.returncode,
            output.decode(errors="replace")[-6000:],
        )
    return CommandResult(
        command,
        "accepted" if child.returncode == 0 else "rejected",
        child.returncode,
        output.decode(errors="replace")[-6000:],
    )


def require_artifact(result, path):
    if result.status == "accepted" and not (
        Path(path).is_file() and Path(path).stat().st_size
    ):
        return CommandResult(
            result.command,
            "harness-error",
            result.returncode,
            result.log + "\nCompiler returned success without an output artifact.",
        )
    return result
