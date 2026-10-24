"""``crosstl doctor``: the validation tools CrossTL looks for, and which of them are on ``PATH``."""

from __future__ import annotations

import subprocess
from typing import Any

from .pipeline import TOOLCHAIN_BY_BACKEND, _tool_status

VERSION_COMMAND_TIMEOUT_SECONDS = 5

# The pipeline's TOOLCHAIN_AVAILABILITY_COMMANDS only check that a tool runs
# (``dxc -help`` prints usage), so the doctor keeps its own commands, each of
# which prints the tool's version.
DOCTOR_VERSION_COMMANDS: dict[str, tuple[str, ...]] = {
    "cuda": ("nvcc", "--version"),
    "directx": ("dxc", "--version"),
    "hip": ("hipcc", "--version"),
    "metal": ("xcrun", "metal", "-v"),
    "mojo": ("mojo", "--version"),
    "opengl": ("glslangValidator", "--version"),
    "rust": ("rustc", "--version"),
    "slang": ("slangc", "-v"),
    "vulkan": ("spirv-val", "--version"),
    "webgl": ("glslangValidator", "--version"),
    "wgsl": ("naga", "--version"),
}


def toolchain_doctor_report() -> dict[str, Any]:
    """One row per backend: tools looked for, resolved path, and version line."""
    rows: list[dict[str, Any]] = []
    for target in sorted(TOOLCHAIN_BY_BACKEND):
        status = _tool_status(target)
        version_command = DOCTOR_VERSION_COMMANDS.get(target)
        version = None
        if version_command and status["status"] == "available":
            version = _first_output_line(version_command)
        rows.append(
            {
                "target": target,
                "status": status["status"],
                "tools": status["tools"],
                "versionCommand": list(version_command) if version_command else None,
                "version": version,
            }
        )
    return {"tools": rows}


def _first_output_line(command: tuple[str, ...]) -> str | None:
    """The first line a version command prints.

    ``None`` if the command cannot run or exits non-zero.
    """
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=VERSION_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    lines = (completed.stdout or completed.stderr or "").splitlines()
    return lines[0].strip() if lines else None
