"""Helpers shared by the current-tree MLX native gates.

Gates pin each translated artifact as ``<sha256>:<sizeBytes>`` and keep one
line of per-target pins per entry, so a contract stays compact and a re-pin
diff names the entries it changes.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

GATE_TOOLS = {
    "directx": ("dxc",),
    "opengl": ("glslangValidator", "spirv-val"),
    "metal": ("xcrun",),
}


def parse_pin(pin):
    """Split a ``<sha256>:<sizeBytes>`` pin into an artifact record."""
    digest, size = pin.split(":")
    return {"sha256": digest, "sizeBytes": int(size)}


def compact_contract_text(contract):
    """Serialize a contract with one line of target pins per artifact entry."""
    placeholder = "@artifacts@"
    text = json.dumps({**contract, "artifacts": placeholder}, indent=2)
    lines = [
        f"    {json.dumps(entry)}: {json.dumps(pins)}"
        for entry, pins in contract["artifacts"].items()
    ]
    block = "{\n" + ",\n".join(lines) + "\n  }"
    return text.replace(json.dumps(placeholder), block, 1) + "\n"


def require_pinned_checkout(require_env, description, targets, commit):
    """Return ``(root, target)`` for an unmodified pinned MLX checkout.

    Skips unless ``require_env`` is ``1``; then the target's toolchain must be
    installed and the checkout must be at ``commit`` with no change under
    ``mlx/backend/metal``.
    """
    if os.environ.get(require_env) != "1":
        pytest.skip(f"set {require_env}=1 for the pinned native {description} gate")
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    assert root_value and target in targets
    for name in GATE_TOOLS[target]:
        assert shutil.which(name), f"Required tool is missing: {name}"
    root = Path(root_value).resolve()
    git = ["git", "-C", str(root)]
    revision = subprocess.run(
        [*git, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert revision == commit
    subprocess.run(
        [*git, "diff", "--exit-code", "HEAD", "--", "mlx/backend/metal"],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return root, target


def declared_buffers(header, kernel):
    """A kernel's parameters without attributes, in declaration order."""
    match = re.search(rf"\]\]\s*void\s+{kernel}\s*\((.*?)\)\s*\{{", header, re.DOTALL)
    assert match, kernel
    names = []
    for parameter in match.group(1).split(","):
        if "[[" in parameter:
            continue
        names.append(re.search(r"(\w+)\s*$", parameter.strip()).group(1))
    return tuple((name, index) for index, name in enumerate(names))


def row_major(shape):
    """Element strides of a contiguous row-major array of ``shape``."""
    strides = [1]
    for size in reversed(shape[1:]):
        strides.insert(0, strides[0] * size)
    return strides


def assert_identical(label, actual, expected):
    """Compare two word sequences exactly, reporting the first mismatches."""
    assert len(actual) == len(expected), label
    mismatches = [
        (index, want, got)
        for index, (got, want) in enumerate(zip(actual, expected))
        if got != want
    ]
    assert not mismatches, (label, len(mismatches), mismatches[:8])
