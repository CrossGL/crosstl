"""Tint, the WGSL compiler of Dawn (the WebGPU implementation in Chromium), which checks WGSL with naga.

Tint runs through Dawn's Node.js bindings, the npm package ``webgpu``, on Dawn's null backend, so it needs Node.js but
no GPU. ``CROSSTL_COVERAGE_WEBGPU`` names the directory of the installed package; ``tools/coverage/webgpu`` pins it,
with every dependency's integrity hash in its lockfile::

    npm ci --prefix <dir> --ignore-scripts    # after copying package.json and package-lock.json to <dir>
    export CROSSTL_COVERAGE_WEBGPU=<dir>/node_modules/webgpu

``webgpu/tint.mjs`` creates the device and prints Tint's errors for the files it is given.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

SCRIPT = Path(__file__).with_name("webgpu") / "tint.mjs"
VARIABLE = "CROSSTL_COVERAGE_WEBGPU"


def package() -> Path | None:
    """The installed webgpu package, or ``None`` when it is not configured."""
    configured = os.environ.get(VARIABLE)
    if not configured:
        return None
    found = Path(configured)
    return found if (found / "index.js").is_file() else None


def command() -> tuple[str, ...]:
    """The command that checks the WGSL files appended to it, or ``()`` when Node.js or the package is missing."""
    found, node = package(), shutil.which("node")
    if not (found and node):
        return ()
    return (node, str(SCRIPT), str(found))


def version() -> str:
    """The package's version, which pins the Dawn build and so Tint's, or an empty string."""
    found = package()
    if not found:
        return ""
    try:
        release = json.loads((found / "package.json").read_text())["version"]
    except (OSError, ValueError, KeyError):
        return ""
    return f"tint (npm webgpu {release})"
