#!/usr/bin/env python3
"""Regenerate a pinned artifact contract without changing its coverage or ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

RESOURCE_FIELDS = ("name", "kind", "set", "binding", "access", "type")
MATERIALIZATION_FIELDS = (
    "status",
    "specializationCount",
    "specializations",
    "unsupported",
    "configuredParameterCount",
    "configuredParameters",
    "configuredParameterSources",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _inside(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve()
    _require(
        path != root and root in path.parents, f"Path escapes project root: {value}"
    )
    return path


def _matches_json_hash(value: Any, expected: str) -> bool:
    # Existing contracts use canonical JSON both with and without a final newline.
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    return expected in {
        hashlib.sha256(encoded).hexdigest(),
        hashlib.sha256(encoded + b"\n").hexdigest(),
    }


def _validate_source(root: Path, contract: dict) -> None:
    source = _inside(root, contract["source"])
    _require(_sha256(source) == contract["sourceSha256"], "Pinned source hash differs")
    if "commit" in contract:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        _require(revision == contract["commit"], "Pinned repository revision differs")
        subprocess.run(
            ["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--", "."],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )


def _check_artifact(root: Path, contract: dict, expected: dict, artifact: dict) -> dict:
    from crosstl.project import reflect_target_host_interface

    _require(artifact.get("status") == "translated", "Artifact was not translated")
    _require(artifact.get("source") == contract["source"], "Artifact source differs")
    _require(artifact.get("target") == contract["target"], "Artifact target differs")
    _require(
        artifact.get("sourceHash")
        == {"algorithm": "sha256", "value": contract["sourceSha256"]},
        "Artifact source hash differs",
    )
    _require(
        artifact.get("entryPoint", {}).get("source") == expected["entryPoint"],
        "Artifact entry point differs",
    )
    path = _inside(root, artifact["path"])
    identity = _sha256(path)
    size = path.stat().st_size
    _require(
        artifact.get("generatedHash") == {"algorithm": "sha256", "value": identity},
        "Generated artifact hash differs",
    )
    _require(
        artifact.get("generatedSizeBytes") == size and size > 0,
        "Generated artifact size differs",
    )
    requirements = contract.get("artifactContract", {})
    provenance = artifact.get("provenance", {})
    for key, field in (("provenance", "pipeline"), ("intermediate", "intermediate")):
        if key in requirements:
            _require(
                provenance.get(field) == requirements[key], f"Artifact {key} differs"
            )
    if "hostDispatchWorkgroupSize" in requirements:
        execution = artifact.get("execution", {})
        entries = execution.get("entryPoints")
        if entries is None and len(execution.get("sourceEntryPoints", [])) == 1:
            entries = [execution]
        _require(
            isinstance(entries, list)
            and len(entries) == 1
            and entries[0].get("workgroupSize")
            == requirements["hostDispatchWorkgroupSize"],
            "Workgroup size differs",
        )
    materialization = artifact.get("templateMaterialization", {})
    normalized = {field: materialization.get(field) for field in MATERIALIZATION_FIELDS}
    if materialization.get("specializationCount", 0):
        _require(
            materialization.get("status") == "materialized",
            "Template materialization failed",
        )
        _require(
            materialization.get("unsupported") == [],
            "Unsupported template specializations remain",
        )
    if "specializationCount" in expected:
        _require(
            normalized["specializationCount"] == expected["specializationCount"],
            "Specialization count differs",
        )
    if "materializationSha256" in expected:
        _require(
            _matches_json_hash(normalized, expected["materializationSha256"]),
            "Materialization contract differs",
        )
    interface = reflect_target_host_interface(path, target=contract["target"])
    _require(
        interface is not None and interface.get("status") == "ready",
        "Host interface is not ready",
    )
    fields = requirements.get("resourceAbiFields", RESOURCE_FIELDS)
    resources = [
        {field: resource[field] for field in fields}
        for resource in interface["resources"]
    ]
    if "resources" in expected:
        _require(resources == expected["resources"], "Resource ABI differs")
    if "resourceCount" in expected:
        _require(len(resources) == expected["resourceCount"], "Resource count differs")
    if "resourcesSha256" in expected:
        _require(
            _matches_json_hash(resources, expected["resourcesSha256"]),
            "Resource ABI digest differs",
        )
    return {
        "entryPoint": expected["entryPoint"],
        "path": str(path),
        "targetEntryPoint": artifact["entryPoint"]["target"],
        "sha256": identity,
        "sizeBytes": size,
        "resources": resources,
        "specializationCount": materialization.get("specializationCount", 0),
        "previousSha256": expected["sha256"],
        "previousSizeBytes": expected["sizeBytes"],
    }


def _compile(
    record: dict, command_template: list[str], work: Path, timeout: float
) -> dict:
    result = dict(record)
    name = hashlib.sha256(record["entryPoint"].encode()).hexdigest()
    directory = work / "compiled" / name
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "artifact.bin"
    output.unlink(missing_ok=True)
    substitutions = {
        "{artifact}": record["path"],
        "{output}": str(output),
        "{entry_point}": record["targetEntryPoint"],
    }
    command = [substitutions.get(argument, argument) for argument in command_template]
    result["command"] = command
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
        result.update(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        _require(completed.returncode == 0, "Native compiler failed")
        _require(
            output.is_file() and not output.is_symlink() and output.stat().st_size > 0,
            "Native compiler produced no nonempty output",
        )
        _require(
            _sha256(Path(record["path"])) == record["sha256"],
            "Artifact changed during compilation",
        )
        result.update(
            status="passed", compiledPath=str(output), compiledSha256=_sha256(output)
        )
    except subprocess.TimeoutExpired as error:
        result.update(
            status="failed",
            error=str(error),
        )
        for stream in ("stdout", "stderr"):
            captured = getattr(error, stream)
            result[stream] = (
                captured.decode("utf-8", errors="replace")
                if isinstance(captured, bytes)
                else captured or ""
            )
    except (OSError, ValueError) as error:
        result.update(status="failed", error=str(error))
    _write_json(directory / "evidence.json", result)
    return result


def _candidate(contract: dict, records: list[dict]) -> dict:
    by_entry = {record["entryPoint"]: record for record in records}
    expected_names = [entry["entryPoint"] for entry in contract["entries"]]
    _require(
        len(records) == len(by_entry) == len(expected_names)
        and set(by_entry) == set(expected_names),
        "Incomplete or duplicate artifact coverage",
    )
    _require(
        all(record["status"] == "passed" for record in records),
        "Native validation is incomplete",
    )
    requirements = contract.get("artifactContract", {})
    aggregates = {
        "artifactCount": len(records),
        "specializationCount": sum(record["specializationCount"] for record in records),
        "reflectedResourceCount": sum(len(record["resources"]) for record in records),
        "reflectedResourceTypeCounts": dict(
            Counter(
                resource["type"]
                for record in records
                for resource in record["resources"]
            )
        ),
    }
    for key in ("specializationCountsByShape", "reflectedResourceCountsByShape"):
        counts = Counter()
        for entry in contract["entries"]:
            record = by_entry[entry["entryPoint"]]
            if "shape" in entry:
                counts[entry["shape"]] += (
                    record["specializationCount"]
                    if key == "specializationCountsByShape"
                    else len(record["resources"])
                )
        aggregates[key] = dict(counts)
    for key, actual in aggregates.items():
        if key in requirements:
            _require(actual == requirements[key], f"Aggregate {key} differs")
    candidate = deepcopy(contract)
    for entry in candidate["entries"]:
        entry.update(
            {key: by_entry[entry["entryPoint"]][key] for key in ("sha256", "sizeBytes")}
        )
    artifact_contract = candidate.get("artifactContract", {})
    if "generatedSizeBytesTotal" in artifact_contract:
        artifact_contract["generatedSizeBytesTotal"] = sum(
            record["sizeBytes"] for record in records
        )
    if "generatedSizeRange" in artifact_contract:
        ordered = sorted(candidate["entries"], key=lambda entry: entry["sizeBytes"])
        artifact_contract["generatedSizeRange"] = {
            label: {key: entry[key] for key in ("entryPoint", "sizeBytes")}
            for label, entry in (("minimum", ordered[0]), ("maximum", ordered[-1]))
        }
    return candidate


def refresh(
    root: Path,
    config_path: Path,
    contract_path: Path,
    work: Path,
    compiler_command: list[str],
    *,
    jobs: int = 1,
    timeout: float = 120,
    entries: list[str] | None = None,
    resume: bool = False,
) -> dict:
    from crosstl.project import load_project_config, translate_project

    root = root.resolve()
    work = _inside(root, work)
    for path in (config_path.resolve(), contract_path.resolve()):
        _require(
            path != work and work not in path.parents,
            "Input contract and configuration must be outside the work directory",
        )
    _require(
        type(jobs) is int and jobs > 0 and math.isfinite(timeout) and timeout > 0,
        "Worker count and timeout must be positive and finite",
    )
    _require(
        isinstance(compiler_command, list)
        and all(isinstance(arg, str) and arg for arg in compiler_command),
        "Compiler command must be a nonempty JSON argv array",
    )
    _require(
        "{artifact}" in compiler_command and "{output}" in compiler_command,
        "Compiler command requires {artifact} and {output} arguments",
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    source = _inside(root, contract["source"])
    _require(
        source != work and work not in source.parents,
        "Pinned source must be outside the work directory",
    )
    expected = {entry["entryPoint"]: entry for entry in contract["entries"]}
    _require(
        expected and len(expected) == len(contract["entries"]),
        "Contract entry points must be nonempty and unique",
    )
    selected = list(expected) if entries is None else entries
    _require(
        selected
        and len(selected) == len(set(selected))
        and set(selected) <= set(expected),
        "Selected entry points must be nonempty, unique and present in the contract",
    )
    config = load_project_config(root, config_path)
    _require(
        not config.variants and not config.translate_discovered_entry_points,
        "Variant/discovery configurations need separate contracts",
    )
    requirements = contract.get("artifactContract", {})
    workgroup_size = requirements.get("hostDispatchWorkgroupSize")
    rules = dict(config.entry_workgroup_size_rules)
    configured_size = config.workgroup_size
    if workgroup_size is not None:
        # Metal template roundtrips retain launch metadata through entry rules.
        if contract["target"] == "metal" and requirements.get("specializationCount", 0):
            rules[contract["source"]] = {"*": workgroup_size}
        else:
            configured_size = workgroup_size
    config = replace(
        config,
        include_patterns=(contract["source"],),
        entry_points={contract["source"]: selected},
        workgroup_size=configured_size,
        entry_workgroup_size_rules=rules,
        targets=(contract["target"],),
    )
    work.mkdir(parents=True, exist_ok=True)
    candidate_path = work / "candidate.json"
    _require(
        candidate_path.resolve() != contract_path.resolve(),
        "Candidate must not overwrite the input contract",
    )
    candidate_path.unlink(missing_ok=True)
    audit = {
        "kind": "crosstl-artifact-contract-refresh",
        "schemaVersion": 1,
        "status": "running",
        "contract": str(contract_path.resolve()),
        "contractSha256": _sha256(contract_path),
        "selectedCount": len(selected),
        "expectedCount": len(expected),
        "records": [],
        "failures": [],
        "root": str(root),
        "source": contract["source"],
        "target": contract["target"],
        "configSha256": _sha256(config_path),
        "compilerCommand": compiler_command,
        "defaultToolchainDiagnostics": [],
        "portabilityReport": str(work / "portability-report.json"),
        "numericalExecution": False,
        "fullUpstreamSuite": False,
    }
    _write_json(work / "audit.json", audit)
    try:
        _validate_source(root, contract)
        report = translate_project(
            config,
            output_dir=(work / "out").relative_to(root).as_posix(),
            format_output=False,
            validate=True,
            run_toolchains=False,
            checkpoint_path=work / "checkpoint.json",
            resume=resume,
            checkpoint_interval_jobs=50,
            max_workers=jobs,
            job_timeout_seconds=timeout,
        )
        report.write_json(work / "portability-report.json")
        payload = report.to_json()
        # Default discovery does not describe the explicit compiler command below.
        # Keep its availability warnings; the command must still compile each entry.
        audit["defaultToolchainDiagnostics"] = [
            diagnostic
            for diagnostic in payload["diagnostics"]
            if diagnostic.get("code") == "project.validate.toolchain-unavailable"
            and diagnostic.get("severity") == "warning"
            and diagnostic.get("target") == contract["target"]
            and diagnostic.get("missingCapabilities") == ["toolchain.validation"]
        ]
        counts = payload["summary"]["diagnosticCounts"]
        _require(
            payload["summary"]["failedCount"] == 0
            and counts.get("error", 0) == 0
            and counts.get("warning", 0) == len(audit["defaultToolchainDiagnostics"]),
            "Translation reported warnings or failures",
        )
        artifacts = payload["artifacts"]
        _require(len(artifacts) == len(selected), "Translation artifact count differs")
        records = []
        seen = set()
        for artifact in artifacts:
            name = artifact.get("entryPoint", {}).get("source")
            _require(
                name in selected and name not in seen,
                "Unexpected or duplicate translated entry",
            )
            seen.add(name)
            try:
                records.append(
                    _check_artifact(root, contract, expected[name], artifact)
                )
            except (ValueError, KeyError, OSError) as error:
                audit["failures"].append({"entryPoint": name, "error": str(error)})
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            audit["records"] = list(
                pool.map(
                    lambda record: _compile(record, compiler_command, work, timeout),
                    records,
                )
            )
        _validate_source(root, contract)
        _require(
            _sha256(contract_path) == audit["contractSha256"]
            and _sha256(config_path) == audit["configSha256"],
            "Input contract or configuration changed during the audit",
        )
        _require(
            not audit["failures"],
            "Artifact contracts changed beyond generated identities",
        )
        candidate = _candidate(contract, audit["records"])
        _write_json(candidate_path, candidate)
        audit.update(
            status="passed",
            candidate=str(candidate_path),
            candidateSha256=_sha256(candidate_path),
            changedCount=sum(
                record["sha256"] != record["previousSha256"]
                or record["sizeBytes"] != record["previousSizeBytes"]
                for record in audit["records"]
            ),
        )
    except KeyboardInterrupt:
        audit["status"] = "interrupted"
        audit["failures"].append({"error": "Audit interrupted"})
        _write_json(work / "audit.json", audit)
        raise
    except (
        ValueError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        audit["status"] = "failed"
        audit["failures"].append({"error": str(error)})
    _write_json(work / "audit.json", audit)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--compiler-command",
        type=json.loads,
        required=True,
        help="JSON argv array with {artifact}, {output}, and optional {entry_point}",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument(
        "--entry",
        action="append",
        help="Audit a subset; never writes an incomplete candidate",
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        result = refresh(
            arguments.root,
            arguments.config,
            arguments.contract,
            arguments.work_dir,
            arguments.compiler_command,
            jobs=arguments.jobs,
            timeout=arguments.timeout_seconds,
            entries=arguments.entry,
            resume=arguments.resume,
        )
    except KeyboardInterrupt:
        parser.exit(130, "Artifact contract refresh interrupted.\n")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Artifact contract refresh: {error}\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "records"}, indent=2
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
