import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest


@lru_cache(maxsize=1)
def _xcrun_resolves_missing_metal_toolchain() -> bool:
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return False

    lookup = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if lookup.returncode != 0:
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = Path(temp_dir) / "probe.metal"
        output_path = Path(temp_dir) / "probe.air"
        source_path.write_text(
            """
#include <metal_stdlib>
using namespace metal;
kernel void probe() {}
""".lstrip(),
            encoding="utf-8",
        )
        probe = subprocess.run(
            [
                xcrun,
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(source_path),
                "-o",
                str(output_path),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )

    diagnostics = "\n".join(
        part for part in (probe.stdout, probe.stderr) if part.strip()
    )
    return (
        probe.returncode != 0
        and "missing Metal Toolchain" in diagnostics
        and "xcodebuild -downloadComponent MetalToolchain" in diagnostics
    )


@pytest.fixture(autouse=True)
def hide_xcrun_when_metal_toolchain_is_incomplete(monkeypatch):
    if not _xcrun_resolves_missing_metal_toolchain():
        return

    real_which = shutil.which

    def which(name, *args, **kwargs):
        if name == "xcrun":
            return None
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", which)
