"""Record and recheck every tracked file in the pinned upstream test corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PIN = "40dbe40de0b1eafb570e2af43f78bcdbffdcc00a"


def snapshot(root: Path) -> dict:
    root = root.resolve()

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments])

    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise ValueError("Source must be the upstream repository root")
    if git("rev-parse", "HEAD").decode().strip() != PIN:
        raise ValueError("Unexpected upstream revision")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("The baseline requires unchanged tracked upstream files")
    files = {}
    for raw in git("ls-files", "-z").split(b"\0"):
        if not raw:
            continue
        name = raw.decode("utf-8")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Unsupported tracked source entry: " + name)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:
        raise ValueError("Empty source corpus")
    return {
        "schemaVersion": 1,
        "kind": "computesharp.source",
        "commit": PIN,
        "tree": git("rev-parse", "HEAD^{tree}").decode().strip(),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if bool(args.output) == bool(args.check):
        parser.error("Specify exactly one of --output or --check")
    try:
        current = snapshot(args.source)
        if args.check:
            if current != json.loads(args.check.read_text(encoding="utf-8")):
                raise ValueError("Upstream source changed during the baseline")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                output.write(json.dumps(current, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
