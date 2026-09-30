#!/usr/bin/env python3
"""Run one CI command with live output and a hard process-tree timeout."""

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from typing import Sequence

TIMEOUT_EXIT_CODE = 124
TERMINATION_GRACE_SECONDS = 10.0


def _elapsed_seconds(started_at):
    return time.monotonic() - started_at


def _terminate_process_tree(process):
    """Terminate *process* and descendants without trusting the child to respond."""

    if process.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                timeout=TERMINATION_GRACE_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def run_bounded_command(command, *, timeout_seconds, label):
    """Run *command*, forwarding output live and returning a shell exit code."""

    started_at = time.monotonic()
    rendered_command = shlex.join(command)
    print(
        "[bounded-command] START "
        f"label={label!r} timeout={timeout_seconds:g}s command={rendered_command}",
        flush=True,
    )

    popen_options = {}
    if os.name == "nt":
        popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_options["start_new_session"] = True

    try:
        process = subprocess.Popen(command, **popen_options)
    except OSError as exc:
        print(
            f"[bounded-command] ERROR label={label!r} failed to start: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 127

    previous_handlers = {}

    def terminate_for_signal(signum, _frame):
        print(
            f"[bounded-command] SIGNAL label={label!r} signal={signum}; "
            "terminating child process tree",
            file=sys.stderr,
            flush=True,
        )
        _terminate_process_tree(process)
        raise SystemExit(128 + signum)

    for signal_name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signal_name, None)
        if signum is None:
            continue
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, terminate_for_signal)

    try:
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            elapsed = _elapsed_seconds(started_at)
            print(
                "[bounded-command] TIMEOUT "
                f"label={label!r} elapsed={elapsed:.1f}s "
                f"limit={timeout_seconds:g}s; terminating child process tree",
                file=sys.stderr,
                flush=True,
            )
            _terminate_process_tree(process)
            print(
                "[bounded-command] END "
                f"label={label!r} status=timeout exit={TIMEOUT_EXIT_CODE} "
                f"elapsed={_elapsed_seconds(started_at):.1f}s",
                flush=True,
            )
            return TIMEOUT_EXIT_CODE
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    status = "passed" if returncode == 0 else "failed"
    print(
        "[bounded-command] END "
        f"label={label!r} status={status} exit={returncode} "
        f"elapsed={_elapsed_seconds(started_at):.1f}s",
        flush=True,
    )
    if 0 <= returncode <= 255:
        return returncode
    if returncode < 0:
        return min(255, 128 - returncode)
    return 1


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run one command with inherited live output and terminate its complete "
            "process tree if the deadline expires."
        )
    )
    parser.add_argument(
        "--timeout-seconds",
        required=True,
        type=float,
        help="Positive wall-clock deadline in seconds.",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Human-readable case identity included in every diagnostic.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run, conventionally preceded by --.",
    )
    return parser


def main(argv: Sequence[str] = None):
    parser = _parser()
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be greater than zero")
    if not command:
        parser.error("a command is required after --")
    return run_bounded_command(
        command,
        timeout_seconds=args.timeout_seconds,
        label=args.label,
    )


if __name__ == "__main__":
    sys.exit(main())
