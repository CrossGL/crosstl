#!/usr/bin/env python3
"""Collect the images written by the upstream image tests, with content hashes.

``ShadersTests`` save every rendered shader as ``Shaders/<name>.png`` next to the
test assembly and compare it with ``Assets/<name>.png``; ``ImageProcessingTests``
and ``ImagingTests`` write ``Imaging/*_gpu.jpg`` and ``Imaging/*_saved.jpg``. The
files are rewritten by every data row, so the collected file is the output of the
last row that ran. The collector copies the outputs, the reference image of every
shader output and the CPU references of the processing tests into one directory
and writes ``outputs.json``. It uses only the standard library so the original
Windows baseline can run it unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

OUTPUT_PATTERNS = ("Shaders/*.png", "Imaging/*_gpu.jpg", "Imaging/*_saved.jpg")
# Assets/<shader>.png is collected for every Shaders/<shader>.png output; the
# other assets are inputs of tests that compare in process.
REFERENCE_PATTERNS = ("Imaging/*_cpu.jpg",)
TEST_BINARIES = "tests/ComputeSharp.Tests/bin"


def find_test_output(checkout: Path) -> Path:
    """Return the single directory that holds the built upstream test assembly."""

    candidates = sorted(
        path.parent
        for path in (checkout / TEST_BINARIES).rglob("ComputeSharp.Tests.dll")
        if path.parent.name.startswith("net")
    )
    if len(candidates) != 1:
        raise SystemExit(
            f"Expected one built ComputeSharp.Tests output under {checkout}, found {candidates}"
        )
    return candidates[0]


def _entry(path: Path, root: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def collect(test_output: Path, destination: Path) -> dict[str, object]:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    report: dict[str, object] = {
        "schemaVersion": 1,
        "kind": "computesharp-test-outputs",
        "testOutput": str(test_output),
    }
    outputs = [
        source
        for pattern in OUTPUT_PATTERNS
        for source in sorted(test_output.glob(pattern))
        if source.is_file()
    ]
    goldens = [
        test_output / "Assets" / source.name
        for source in outputs
        if source.parent.name == "Shaders"
    ]
    references = [
        source
        for pattern in REFERENCE_PATTERNS
        for source in sorted(test_output.glob(pattern))
        if source.is_file()
    ]
    missing = []
    for key, sources in (("outputs", outputs), ("references", goldens + references)):
        entries = []
        for source in sources:
            if not source.is_file():
                missing.append(source.relative_to(test_output).as_posix())
                continue
            target = destination / key / source.relative_to(test_output)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            entries.append(_entry(target, destination / key))
        report[key] = entries
    report["missingReferences"] = missing
    (destination / "outputs.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkout", type=Path, help="upstream or adapted checkout")
    source.add_argument(
        "--test-output", type=Path, help="directory of the test assembly"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    test_output = args.test_output or find_test_output(args.checkout)
    report = collect(test_output, args.output)
    print(
        f"Collected {len(report['outputs'])} outputs and {len(report['references'])} references"
    )
    return 0 if report["outputs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
