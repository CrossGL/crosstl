#!/usr/bin/env python3
"""Compile a complete, pinned source bundle without repeating translation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.refresh_artifact_contract import _compile, _sha256, _write_json

ENTRY_FIELDS = {
    "schemaVersion",
    "contractSha256",
    "entryPoint",
    "artifact",
    "sha256",
    "sizeBytes",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
    )


def _contract(path: Path) -> tuple[dict, dict[str, dict], str]:
    data = path.read_bytes()
    contract = json.loads(data, object_pairs_hook=_unique_object)
    _require(isinstance(contract, dict), "Contract must be an object")
    entries = contract.get("entries")
    _require(isinstance(entries, list) and entries, "Contract entries are missing")
    expected = {}
    for entry in entries:
        _require(isinstance(entry, dict), "Contract entry must be an object")
        name = entry.get("entryPoint")
        _require(isinstance(name, str) and name, "Contract entry point is invalid")
        _require(name not in expected, f"Duplicate contract entry: {name}")
        digest = entry.get("sha256")
        _require(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
            f"Invalid artifact hash: {name}",
        )
        size = entry.get("sizeBytes")
        _require(type(size) is int and size > 0, f"Invalid artifact size: {name}")
        expected[name] = entry
    requirements = contract.get("artifactContract")
    _require(isinstance(requirements, dict), "Artifact contract is missing")
    count = requirements.get("artifactCount")
    _require(type(count) is int and count == len(expected), "Contract count differs")
    return contract, expected, hashlib.sha256(data).hexdigest()


def write_bundle_entry(
    source: Path, contract_path: Path, entry_point: str, bundle_root: Path
) -> Path:
    """Export one verified artifact; duplicate writers fail instead of overwriting."""

    _, expected, contract_hash = _contract(contract_path)
    _require(entry_point in expected, f"Unexpected artifact: {entry_point}")
    _require(
        source.is_file() and not source.is_symlink(), "Source is not a regular file"
    )
    _require(re.fullmatch(r"\.[A-Za-z0-9]+", source.suffix), "Source suffix is invalid")
    data = source.read_bytes()
    entry = expected[entry_point]
    _require(len(data) == entry["sizeBytes"], f"Artifact size differs: {entry_point}")
    _require(
        hashlib.sha256(data).hexdigest() == entry["sha256"],
        f"Artifact hash differs: {entry_point}",
    )
    _require(not bundle_root.is_symlink(), "Bundle root must not be a symlink")
    directory = bundle_root / hashlib.sha256(entry_point.encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=False)
    artifact = directory / ("artifact" + source.suffix)
    artifact.write_bytes(data)
    _write_json(
        directory / "entry.json",
        {
            "schemaVersion": 1,
            "contractSha256": contract_hash,
            "entryPoint": entry_point,
            "artifact": artifact.name,
            "sha256": entry["sha256"],
            "sizeBytes": entry["sizeBytes"],
        },
    )
    return artifact


def _bundle_records(
    root: Path, expected: dict[str, dict], contract_hash: str
) -> list[dict]:
    _require(root.is_dir() and not root.is_symlink(), "Bundle root is not a directory")
    root = root.resolve()
    paths = list(root.rglob("*"))
    _require(not any(path.is_symlink() for path in paths), "Bundle contains a symlink")
    _require(
        all(path.is_dir() or path.is_file() for path in paths),
        "Bundle contains a non-regular file",
    )
    seen = {}
    verified_files = set()
    for receipt in sorted(path for path in paths if path.name == "entry.json"):
        record = _read_json(receipt)
        _require(
            isinstance(record, dict) and set(record) == ENTRY_FIELDS,
            "Invalid bundle entry fields",
        )
        _require(
            type(record["schemaVersion"]) is int and record["schemaVersion"] == 1,
            "Unsupported bundle schema",
        )
        _require(
            record["contractSha256"] == contract_hash, "Bundle contract hash differs"
        )
        name = record["entryPoint"]
        _require(isinstance(name, str) and name in expected, "Unexpected bundle entry")
        _require(name not in seen, f"Duplicate bundle entry: {name}")
        entry = expected[name]
        _require(
            record["sha256"] == entry["sha256"]
            and type(record["sizeBytes"]) is int
            and record["sizeBytes"] == entry["sizeBytes"],
            f"Bundle identity differs: {name}",
        )
        filename = record["artifact"]
        _require(
            isinstance(filename, str)
            and re.fullmatch(r"artifact\.[A-Za-z0-9]+", filename),
            "Bundle artifact filename is invalid",
        )
        artifact = receipt.parent / filename
        _require(artifact.is_file(), f"Bundle artifact is missing: {name}")
        _require(
            artifact.stat().st_size == entry["sizeBytes"],
            f"Artifact size differs: {name}",
        )
        _require(_sha256(artifact) == entry["sha256"], f"Artifact hash differs: {name}")
        seen[name] = {
            "entryPoint": name,
            "targetEntryPoint": entry.get("targetEntryPoint", name),
            "path": str(artifact),
            "sha256": entry["sha256"],
            "sizeBytes": entry["sizeBytes"],
        }
        verified_files.update((receipt, artifact))
    _require(set(seen) == set(expected), "Bundle coverage is incomplete")
    _require(
        {path for path in paths if path.is_file()} == verified_files,
        "Bundle contains unrecognized files",
    )
    return [seen[name] for name in sorted(seen)]


def compile_bundle(
    bundle_root: Path,
    contract_path: Path,
    output_dir: Path,
    compiler_command: list[str],
    *,
    jobs: int = 2,
    timeout: float = 120,
) -> dict:
    _require(type(jobs) is int and jobs > 0, "Worker count must be positive")
    _require(
        math.isfinite(timeout) and timeout > 0, "Timeout must be positive and finite"
    )
    _require(
        isinstance(compiler_command, list)
        and all(isinstance(arg, str) and arg for arg in compiler_command)
        and "{artifact}" in compiler_command
        and "{output}" in compiler_command,
        "Compiler command requires artifact and output placeholders",
    )
    bundle_root, contract_path = bundle_root.absolute(), contract_path.absolute()
    output_dir = output_dir.resolve()
    _require(
        output_dir != bundle_root.resolve()
        and bundle_root.resolve() not in output_dir.parents
        and output_dir not in bundle_root.resolve().parents
        and contract_path.resolve() != output_dir
        and output_dir not in contract_path.resolve().parents,
        "Compiler output must be separate from bundle and contract inputs",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "crosstl-artifact-bundle-compilation",
        "schemaVersion": 1,
        "status": "running",
        "contract": str(contract_path),
        "compilerCommand": compiler_command,
        "records": [],
        "failures": [],
        "numericalExecution": False,
        "fullUpstreamSuite": False,
    }
    report_path = output_dir / "report.json"
    _write_json(report_path, report)
    try:
        contract, expected, contract_hash = _contract(contract_path)
        report.update(
            contractSha256=contract_hash,
            expectedCount=len(expected),
            target=contract.get("target"),
        )
        records = _bundle_records(bundle_root, expected, contract_hash)
        # Use the same warnings, timeout and nonempty-output checks as contract refreshes.
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            compiled = executor.map(
                lambda record: _compile(record, compiler_command, output_dir, timeout),
                records,
            )
            for result in compiled:
                report["records"].append(result)
                if result["status"] != "passed":
                    report["failures"].append(result["entryPoint"])
                if len(report["records"]) % 50 == 0:
                    _write_json(report_path, report)
        report["status"] = "failed" if report["failures"] else "passed"
    except (OSError, ValueError, TypeError, KeyError) as error:
        report.update(status="failed", error=str(error))
    _write_json(report_path, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compiler-command",
        type=json.loads,
        required=True,
        help="JSON argv with {artifact} and {output} placeholders",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    try:
        report = compile_bundle(
            args.bundle_root,
            args.contract,
            args.output_dir,
            args.compiler_command,
            jobs=args.jobs,
            timeout=args.timeout,
        )
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"{error}\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"}, indent=2
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
