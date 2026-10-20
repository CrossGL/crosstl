"""Create a patched copy of the pinned ComputeSharp tree for non-Windows builds.

The original checkout is verified against the pinned revision and left unchanged.
Each patch in ``patches/`` is applied in order to a separate clone, and a receipt
records the patch hashes and the resulting diff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from source import PIN, snapshot  # noqa: E402

PATCH_DIR = HERE / "patches"


def patch_series() -> list[Path]:
    patches = sorted(PATCH_DIR.glob("*.patch"))
    if not patches:
        raise ValueError("No upstream patches found")
    return patches


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def prepare(source: Path, output: Path) -> dict:
    receipt = snapshot(source)
    output = output.resolve()
    if output.exists():
        raise ValueError("Output already exists: " + str(output))
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--shared",
            "--no-checkout",
            str(source),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(output, "checkout", "--quiet", "--detach", PIN)
    applied = []
    for patch in patch_series():
        try:
            _git(output, "apply", "--check", str(patch))
            _git(output, "apply", str(patch))
        except subprocess.CalledProcessError as error:
            raise ValueError(f"{patch.name} does not apply: {error.stderr.strip()}")
        applied.append(
            {
                "name": patch.name,
                "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            }
        )
    diff = _git(output, "diff", "--no-color", PIN)
    return {
        "schemaVersion": 1,
        "kind": "computesharp.adapted-source",
        "commit": PIN,
        "tree": receipt["tree"],
        "patches": applied,
        "changedFiles": sorted(
            line.split("\t", 1)[1]
            for line in _git(output, "diff", "--name-status", PIN).splitlines()
        ),
        "diffSHA256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        receipt = prepare(args.source, args.output)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, str(error) + "\n")
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
