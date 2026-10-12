#!/usr/bin/env python3
"""Wait for a downloaded Metal compiler to become executable."""

import argparse
import subprocess
import sys
import time

PROBE_TIMEOUT_SECONDS = 10
DOWNLOAD_TIMEOUT_SECONDS = 480
ACTIVATION_TIMEOUT_SECONDS = 60
ACTIVATION_ATTEMPTS = 10
RETRY_DELAY_SECONDS = 3


def _run(command, *, timeout, capture_output=False):
    try:
        return subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=capture_output,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"Metal toolchain command timed out: {command[0]}", file=sys.stderr)
        return subprocess.CompletedProcess(command, 124, "", "")
    except OSError as error:
        print(f"Cannot run Metal toolchain command: {error}", file=sys.stderr)
        return subprocess.CompletedProcess(command, 127, "", "")


def _probe(timeout=PROBE_TIMEOUT_SECONDS):
    result = _run(
        ["xcrun", "--sdk", "macosx", "metal", "--version"],
        timeout=timeout,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    return result


def _missing_component(result):
    return (
        "missing metal toolchain" in (
            (result.stdout or "") + "\n" + (result.stderr or "")
        ).lower()
    )


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    result = _probe()
    if result.returncode == 0 or not _missing_component(result):
        return result.returncode
    download = _run(
        ["xcodebuild", "-downloadComponent", "MetalToolchain"],
        timeout=DOWNLOAD_TIMEOUT_SECONDS,
    )
    if download.returncode:
        return download.returncode

    # Download completion can precede cryptex activation on a fresh runner.
    deadline = time.monotonic() + ACTIVATION_TIMEOUT_SECONDS
    for attempt in range(ACTIVATION_ATTEMPTS):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        result = _probe(min(PROBE_TIMEOUT_SECONDS, remaining))
        if result.returncode == 0 or not _missing_component(result):
            return result.returncode
        remaining = deadline - time.monotonic()
        if attempt + 1 == ACTIVATION_ATTEMPTS or remaining <= 0:
            break
        print(
            "Waiting for downloaded Metal toolchain activation "
            f"({attempt + 1}/{ACTIVATION_ATTEMPTS}).",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(min(RETRY_DELAY_SECONDS, remaining))
    print("Downloaded Metal toolchain is still unavailable.", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
