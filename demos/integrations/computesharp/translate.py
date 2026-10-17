#!/usr/bin/env python3
"""Translate exported ComputeSharp HLSL into Metal and OpenGL artifacts with CrossTL.

The input is a discovery export (``discovery.json`` plus ``hlsl/``) produced by
``discovery/``. Each generated HLSL source is translated with
``crosstl.translate_with_metadata`` and written with a manifest that the portable
runtime loads at dispatch time. The manifest records, per shader, the HLSL hash
the runtime looks up, the artifact hash, the target binding of every HLSL
register, the thread-group size and the CrossTL revision that produced it.

Translation failures are recorded rather than skipped. ``--validate`` also
compiles Metal artifacts with the native offline compiler when it is available;
OpenGL artifacts are compiled by the actual driver when the runtime loads them.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 1
TARGETS = {
    "metal": {"extension": ".metal", "entry_point": "Execute"},
    "opengl": {"extension": ".comp", "entry_point": "main"},
}
MAX_ERROR_CHARS = 4000
# ComputeSharp compiles its HLSL with DXC -Zpr (row-major matrix packing); the
# HLSL frontend needs the same packing to order matrix products correctly.
SOURCE_OPTIONS = {"matrix_packing": "row_major"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def crosstl_revision() -> dict[str, Any]:
    revision: dict[str, Any] = {"package": "crosstl"}
    try:
        revision["commit"] = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "crosstl"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        revision["translatorTreeClean"] = not status.strip()
        if status.strip():
            diff = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "diff", "HEAD", "--", "crosstl"],
                check=True,
                capture_output=True,
            ).stdout
            revision["translatorDiffSha256"] = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        revision["commit"] = None
    return revision


def translate_one(job: tuple[str, str, str, str]) -> dict[str, Any]:
    shader_id, source_path, target, output_dir = job
    sys.path.insert(0, str(REPO_ROOT))
    import crosstl  # noqa: PLC0415 - imported in the worker process

    source = Path(source_path).read_text(encoding="utf-8")
    record: dict[str, Any] = {
        "id": shader_id,
        "target": target,
        "hlslSha256": sha256_text(source),
    }
    try:
        result = crosstl.translate_with_metadata(
            source_path,
            backend=target,
            format_output=False,
            source_options=SOURCE_OPTIONS,
        )
    except Exception as exc:  # noqa: BLE001 - each failure is recorded per shader
        record["status"] = "translation-failed"
        record["error"] = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
        return record
    artifact_name = shader_id + TARGETS[target]["extension"]
    artifact_path = Path(output_dir) / target / artifact_name
    artifact_path.write_text(result.code, encoding="utf-8")
    record.update(
        {
            "status": "translated",
            "artifact": f"{target}/{artifact_name}",
            "artifactSha256": sha256_text(result.code),
            "entryPoint": TARGETS[target]["entry_point"],
            "bindings": [dict(binding) for binding in result.resource_bindings],
        }
    )
    return record


def validate_metal(record: dict[str, Any], output_dir: Path, metal: str) -> None:
    """Compile and link one artifact into a Metal library.

    Linking matters: the runtime compiles source into a library, and errors
    such as program-scope global constructors only surface at link time. The
    math flags match the runtime compile options (see csbackend_metal.m).
    """
    artifact = output_dir / record["artifact"]
    with tempfile.TemporaryDirectory() as scratch:
        library = Path(scratch) / "shader.metallib"
        completed = subprocess.run(
            [
                metal,
                "-fmetal-math-mode=safe",
                "-fmetal-math-fp32-functions=fast",
                str(artifact),
                "-o",
                str(library),
            ],
            capture_output=True,
            text=True,
        )
    if completed.returncode == 0:
        record["validation"] = {"tool": "metal", "status": "linked"}
    else:
        record["status"] = "compile-failed"
        record["validation"] = {
            "tool": "metal",
            "status": "failed",
            "log": (completed.stderr or completed.stdout)[:MAX_ERROR_CHARS],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--target", action="append", choices=sorted(TARGETS), dest="targets"
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = parser.parse_args(argv)

    discovery = json.loads(
        (args.discovery / "discovery.json").read_text(encoding="utf-8")
    )
    targets = args.targets or sorted(TARGETS)
    if args.output.exists():
        shutil.rmtree(args.output)
    for target in targets:
        (args.output / target).mkdir(parents=True)

    shaders = {shader["id"]: shader for shader in discovery["shaders"]}
    jobs = [
        (shader["id"], str(args.discovery / shader["path"]), target, str(args.output))
        for shader in discovery["shaders"]
        for target in targets
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(translate_one, jobs))

    metal = shutil.which("xcrun")
    metal_compiler = None
    if args.validate and "metal" in targets and metal:
        located = subprocess.run(
            ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
        )
        if located.returncode == 0:
            metal_compiler = located.stdout.strip()
    if metal_compiler:
        translated = [
            r for r in records if r["target"] == "metal" and r["status"] == "translated"
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(
                pool.map(
                    lambda record: validate_metal(record, args.output, metal_compiler),
                    translated,
                )
            )

    revision = crosstl_revision()
    summary: dict[str, Any] = {}
    for target in targets:
        entries = {}
        for record in records:
            if record["target"] != target:
                continue
            shader = shaders[record["id"]]
            if record["hlslSha256"] != shader["sha256"]:
                raise SystemExit(f"HLSL hash mismatch for {record['id']}")
            entry = {
                "type": shader["type"],
                "threads": shader["threads"],
                "constantBufferSize": shader["constantBufferSize"],
                "staticSamplerRequired": shader["staticSamplerRequired"],
                "resourceDescriptorRanges": shader["resourceDescriptorRanges"],
            }
            entry.update(
                {key: value for key, value in record.items() if key != "target"}
            )
            entries[record["hlslSha256"]] = entry
        counts: dict[str, int] = {}
        for entry in entries.values():
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        manifest = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "computesharp-translated-shaders",
            "target": target,
            "translator": revision,
            "validated": bool(metal_compiler) if target == "metal" else False,
            "counts": counts,
            "shaders": dict(sorted(entries.items())),
        }
        (args.output / target / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary[target] = counts
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
