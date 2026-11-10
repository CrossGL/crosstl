"""Compile a translated Metal compute kernel and dispatch one thread of it on a single 32-bit word buffer.

The kernel is compiled with ``xcrun metal -Werror`` and run through
``tests/fixtures/runtime_verification/metal_raw_buffers.swift``. Without a Metal toolchain the calling test is
skipped, or fails when ``require_env`` is set to ``1``, as the macOS runtime-parity job sets it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift"

__all__ = ["compile_metal_library", "dispatch_word_buffer"]


def _unavailable(require_env, reason):
    if os.environ.get(require_env) == "1":
        pytest.fail(f"{require_env}=1 needs {reason}")
    pytest.skip(f"the Metal dispatch needs {reason}")


def _run(command):
    completed = subprocess.run(
        [str(item) for item in command], capture_output=True, text=True, timeout=300
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def compile_metal_library(source, tmp_path, require_env):
    """Compile ``source`` with warnings as errors into a metallib and return its path."""
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        _unavailable(require_env, "the Metal toolchain")
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        _unavailable(require_env, "the Metal toolchain")
    shader = tmp_path / "kernel.metal"
    shader.write_text(source, encoding="utf-8")
    air, library = tmp_path / "kernel.air", tmp_path / "kernel.metallib"
    _run([xcrun, "-sdk", "macosx", "metal", "-Werror", "-c", shader, "-o", air])
    _run([xcrun, "-sdk", "macosx", "metallib", air, "-o", library])
    return library


def dispatch_word_buffer(source, tmp_path, words, require_env):
    """Run ``source``'s kernel on one thread with buffer 0 holding ``words``; return the buffer's signed words."""
    if sys.platform != "darwin" or shutil.which("swiftc") is None:
        _unavailable(require_env, "macOS with swiftc")
    library = compile_metal_library(source, tmp_path, require_env)
    kernel = re.search(r"kernel void (\w+)\((.*?)\) \{", source, re.S)
    assert kernel is not None, source
    assert re.findall(r"\[\[buffer\((\d+)\)\]\]", kernel.group(2)) == ["0"], source
    payload = tmp_path / "buffer-0.in"
    payload.write_bytes(struct.pack(f"<{len(words)}i", *words))
    runner = tmp_path / "runner"
    _run(["swiftc", RUNNER, "-o", runner])
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": [str(payload)],
                "workgroupCount": [1, 1, 1],
                "workgroupSize": [1, 1, 1],
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    _run([runner, library, kernel.group(1), request, output])
    data = (output / "buffer-0.bin").read_bytes()[: 4 * len(words)]
    return list(struct.unpack(f"<{len(words)}i", data))
