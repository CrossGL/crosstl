"""Reference validators, one per target language.

A validator is looked up on ``PATH``; ``CROSSTL_COVERAGE_<TOOL>`` (for example ``CROSSTL_COVERAGE_DXC``) points at a
specific binary instead.
"""

from __future__ import annotations

import functools
import os
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

TIMEOUT = 60
# The tool each language is validated with, and the suffix its compiler expects.
TOOLS = {
    "glsl": "glslangValidator",
    "hlsl": "dxc",
    "metal": "metal",
    "spirv": "spirv-val",
    "wgsl": "naga",
}
SUFFIX = {
    "glsl": ".comp",
    "hlsl": ".hlsl",
    "metal": ".metal",
    "spirv": ".spvasm",
    "wgsl": ".wgsl",
}
GLSL_STAGES = {"compute": "comp", "fragment": "frag", "vertex": "vert"}
HLSL_PROFILES = {"compute": "cs_6_6", "fragment": "ps_6_6", "vertex": "vs_6_6"}
SPIRV_ENVIRONMENT = "vulkan1.1"


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    log: str = ""


def _executable(name: str) -> str | None:
    override = os.environ.get("CROSSTL_COVERAGE_" + re.sub(r"\W", "_", name).upper())
    return override or shutil.which(name)


@functools.lru_cache(maxsize=None)
def command(name: str) -> tuple[str, ...]:
    """The command that runs ``name``, or ``()`` when it is not installed."""
    if name == "metal":
        # `xcrun -f metal` succeeds even when the Metal toolchain component is not installed; --version does not.
        metal = ("xcrun", "-sdk", "macosx", "metal")
        if platform.system() != "Darwin" or not shutil.which("xcrun"):
            return ()
        try:
            probe = subprocess.run(
                [*metal, "--version"], capture_output=True, text=True, timeout=TIMEOUT
            )
        except (OSError, subprocess.TimeoutExpired):
            return ()
        return metal if probe.returncode == 0 else ()
    found = _executable(name)
    return (found,) if found else ()


@functools.lru_cache(maxsize=None)
def version(name: str) -> str:
    """The first line the tool prints for ``--version``, or an empty string when it is not installed."""
    base = command(name)
    if not base:
        return ""
    try:
        result = subprocess.run(
            [*base, "--version"], capture_output=True, text=True, timeout=TIMEOUT
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    lines = [
        line.strip()
        for line in (result.stdout + result.stderr).splitlines()
        if line.strip()
    ]
    return lines[0] if lines else ""


def available(language: str) -> bool:
    if language == "spirv":
        return bool(command("spirv-as") and command("spirv-val"))
    return bool(command(TOOLS[language]))


def _run(arguments: list[str], cwd: str) -> Verdict:
    try:
        result = subprocess.run(
            arguments, capture_output=True, text=True, timeout=TIMEOUT, cwd=cwd
        )
    except subprocess.TimeoutExpired:
        return Verdict(False, f"{arguments[0]} timed out after {TIMEOUT} s")
    log = (result.stdout + result.stderr).replace(cwd + os.sep, "").strip()
    return Verdict(result.returncode == 0, log[-2000:])


def hlsl_entry(source: str, stage: str) -> str:
    """The entry point of an HLSL compute shader: the function that follows ``[numthreads(...)]``."""
    if stage == "compute":
        match = re.search(
            r"\[numthreads\([^)]*\)\]\s*(?:\[[^\]]*\]\s*)*void\s+(\w+)", source
        )
        if match:
            return match.group(1)
    return "main"


def validate(
    language: str, source: str, stage: str = "compute", entry: str = ""
) -> Verdict:
    """Check ``source`` with the language's reference validator. Call only when :func:`available` is true."""
    with tempfile.TemporaryDirectory(prefix="crosstl-coverage-") as scratch:
        name = "probe" + (
            f".{GLSL_STAGES[stage]}" if language == "glsl" else SUFFIX[language]
        )
        Path(scratch, name).write_text(source)
        if language == "glsl":
            return _run([*command("glslangValidator"), name], scratch)
        if language == "hlsl":
            entry = entry or hlsl_entry(source, stage)
            profile = HLSL_PROFILES[stage]
            return _run(
                [
                    *command("dxc"),
                    "-T",
                    profile,
                    "-E",
                    entry,
                    "-HV",
                    "2021",
                    name,
                    "-Fo",
                    "probe.bin",
                ],
                scratch,
            )
        if language == "metal":
            return _run([*command("metal"), "-c", name, "-o", "probe.air"], scratch)
        if language == "wgsl":
            return _run([*command("naga"), name], scratch)
        environment = ["--target-env", SPIRV_ENVIRONMENT]
        assembled = _run(
            [*command("spirv-as"), *environment, name, "-o", "probe.spv"], scratch
        )
        if not assembled.accepted:
            return assembled
        return _run([*command("spirv-val"), *environment, "probe.spv"], scratch)
