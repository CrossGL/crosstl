"""``crosstl doctor``: the validation tools CrossTL looks for, and which of them are on ``PATH``."""

from __future__ import annotations

import subprocess
from typing import Any

from .pipeline import (
    TOOLCHAIN_AVAILABILITY_COMMANDS,
    TOOLCHAIN_BY_BACKEND,
    _tool_status,
)

VERSION_COMMAND_TIMEOUT_SECONDS = 5


def toolchain_doctor_report() -> dict[str, Any]:
    """One row per backend: tools looked for, resolved path, and version line."""
    rows: list[dict[str, Any]] = []
    for target in sorted(TOOLCHAIN_BY_BACKEND):
        status = _tool_status(target)
        version_command = TOOLCHAIN_AVAILABILITY_COMMANDS.get(target)
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
    """The first line a version command prints, or ``None`` if it cannot run."""
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
    lines = (completed.stdout or completed.stderr or "").splitlines()
    return lines[0].strip() if lines else None
