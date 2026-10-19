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


def run(command, *, timeout=60, env=None, input=None, separate_stderr=False):
    """Run ``command``; ``input`` is fed to stdin, and ``separate_stderr`` keeps stderr out of the log."""
    command = tuple(str(value) for value in command)
    try:
        child = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if input is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL if separate_stderr else subprocess.STDOUT,
            env=env,
            start_new_session=os.name != "nt",
        )
    except FileNotFoundError as error:
        return CommandResult(command, "unavailable", None, str(error))
    data = input.encode() if isinstance(input, str) else input
    try:
        output, _ = child.communicate(input=data, timeout=timeout)
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
    text = output.decode(errors="replace")
    return CommandResult(
        command,
        "accepted" if child.returncode == 0 else "rejected",
        child.returncode,
        text if separate_stderr else text[-6000:],
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
