"""Per-entry contracts for MLX frontier reports of one-entry-per-stage targets.

A GLSL compilation unit holds one entry point per stage (GLSL 4.60 section 6.1), so project translation gives each
kernel of a multi-kernel MLX unit its own OpenGL artifact (``crosstl.project.glsl_entry_split``). These checks join
every artifact of such a report to the entry discovery of its unit: each discovered entry has exactly one artifact,
in discovery order, under the unit's output directory, with its own outcome and diagnostics.
"""

from __future__ import annotations

import fnmatch
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "ENTRY_OUTCOME_TRANSLATED",
    "WORKGROUP_SIZE_REQUIRED_CODE",
    "WORKGROUP_SIZE_REQUIRED_MESSAGE",
    "PortingCheckError",
    "join_entry_artifacts",
    "require",
    "require_clean_entry_frontier",
    "require_entry_outcome_frontier",
    "require_entry_rule_sizes",
]

ENTRY_OUTCOME_TRANSLATED = "translated"
WORKGROUP_SIZE_REQUIRED_CODE = "project.translate.workgroup-size-required"
WORKGROUP_SIZE_REQUIRED_MESSAGE = (
    "A consumed threads_per_threadgroup value requires a concrete project "
    "workgroup_size or source execution size for this target."
)


class PortingCheckError(RuntimeError):
    """Raised when an MLX project-porting check fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PortingCheckError(message)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _discovered_entries(
    unit: Mapping[str, Any], label: str, source: str
) -> tuple[str, ...]:
    discovery = unit.get("entryDiscovery")
    entries = discovery.get("entries") if isinstance(discovery, Mapping) else None
    require(
        isinstance(discovery, Mapping)
        and discovery.get("diagnosticCount") == 0
        and isinstance(entries, list)
        and all(
            isinstance(entry, Mapping)
            and entry.get("stage") == "compute"
            and isinstance(entry.get("name"), str)
            and bool(entry["name"])
            for entry in entries
        ),
        f"{label} entry discovery changed for {source}",
    )
    names = tuple(str(entry["name"]) for entry in entries)
    require(
        len(set(names)) == len(names),
        f"{label} entry discovery repeated an entry of {source}",
    )
    return names


def _artifact_entry(artifact: Mapping[str, Any]) -> str | None:
    """The source entry an artifact holds; a refused artifact records no translated stage."""
    entry = artifact.get("entryPoint")
    stage_recorded = artifact.get("status") == ENTRY_OUTCOME_TRANSLATED
    if (
        isinstance(entry, Mapping)
        and (
            entry.get("stage") == "compute"
            if stage_recorded
            else entry.get("stage", "compute") == "compute"
        )
        and entry.get("target") == "main"
        and isinstance(entry.get("source"), str)
        and entry["source"]
    ):
        return str(entry["source"])
    return None


def _host_names(artifact: Mapping[str, Any]) -> list[Any] | None:
    materialization = artifact.get("templateMaterialization")
    specializations = (
        materialization.get("specializations")
        if isinstance(materialization, Mapping)
        else None
    )
    if not (
        isinstance(materialization, Mapping)
        and materialization.get("status") == "materialized"
        and materialization.get("unsupported") == []
        and isinstance(specializations, list)
        and materialization.get("specializationCount") == len(specializations)
        and all(isinstance(record, Mapping) for record in specializations)
    ):
        return None
    return [record.get("hostName") for record in specializations]


def join_entry_artifacts(
    mlx_root: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    target: str,
    units_by_source: Mapping[str, Mapping[str, Any]],
    entry_counts: Mapping[str, int],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Artifacts by source and entry; every discovered entry has exactly one, in discovery order."""
    label = f"{target.title()} per-entry frontier"
    artifacts = payload.get("artifacts")
    require(isinstance(artifacts, list), f"{label} artifact records are missing")
    output_root = output_dir.resolve()
    by_source: dict[str, dict[str, Mapping[str, Any]]] = {
        source: {} for source in units_by_source
    }
    for artifact in artifacts:
        source = artifact.get("source") if isinstance(artifact, Mapping) else None
        require(
            isinstance(source, str)
            and source in units_by_source
            and artifact.get("target") == target
            and artifact.get("sourceBackend") == "metal",
            f"{label} artifact/source join changed",
        )
        unit = units_by_source[source]
        entry = _artifact_entry(artifact)
        require(
            entry is not None and entry not in by_source[source],
            f"{label} artifact entry is missing or repeated for {source}",
        )
        require(
            artifact.get("sourceHash") == unit.get("sourceHash")
            and artifact.get("sourceSizeBytes") == unit.get("sourceSizeBytes"),
            f"{label} artifact source provenance changed for {source}:{entry}",
        )
        host_names = _host_names(artifact)
        require(
            host_names is not None and entry in host_names,
            f"{label} materialization does not name {source}:{entry}",
        )
        artifact_path = artifact.get("path")
        require(
            isinstance(artifact_path, str) and bool(artifact_path),
            f"{label} artifact path is missing for {source}:{entry}",
        )
        unit_dir = (output_root / target / Path(source).with_suffix("")).resolve()
        require(
            (mlx_root / artifact_path).resolve().parent == unit_dir,
            f"{label} artifact is outside its unit directory: {artifact_path}",
        )
        by_source[source][entry] = artifact
    for source, unit in units_by_source.items():
        discovered = _discovered_entries(unit, label, source)
        require(
            len(discovered) == entry_counts[source],
            f"{label} discovered {len(discovered)} entries of {source}, "
            f"expected {entry_counts[source]}",
        )
        require(
            tuple(by_source[source]) == discovered,
            f"{label} artifacts do not match the discovered entries of {source}",
        )
    return by_source


def _require_translated_artifact(
    mlx_root: Path, artifact: Mapping[str, Any], label: str
) -> None:
    artifact_path = str(artifact["path"])
    generated_path = mlx_root / artifact_path
    require(
        artifact.get("status") == ENTRY_OUTCOME_TRANSLATED
        and "error" not in artifact
        and generated_path.is_file(),
        f"{label} translated artifact is missing: {artifact_path}",
    )
    generated = generated_path.read_bytes()
    generated_hash = artifact.get("generatedHash")
    require(
        isinstance(generated_hash, Mapping)
        and generated_hash.get("algorithm") == "sha256"
        and generated_hash.get("value") == hashlib.sha256(generated).hexdigest()
        and artifact.get("generatedSizeBytes") == len(generated),
        f"{label} artifact identity does not match {artifact_path}",
    )


def _require_summary(
    payload: Mapping[str, Any],
    *,
    target: str,
    unit_count: int,
    artifact_count: int,
    translated_count: int,
    diagnostics_by_code: Mapping[str, int],
    label: str,
) -> None:
    failed_count = artifact_count - translated_count
    summary = payload.get("summary")
    observed_by_code = (
        summary.get("diagnosticsByCode", {}) if isinstance(summary, Mapping) else None
    )
    require(
        isinstance(summary, Mapping)
        and summary.get("unitCount") == unit_count
        and summary.get("artifactCount") == artifact_count
        and summary.get("translatedCount") == translated_count
        and summary.get("failedCount") == failed_count
        and summary.get("diagnosticCounts") == {
            "error": sum(diagnostics_by_code.values()),
            "note": 0,
            "warning": 0,
        }
        and dict(observed_by_code or {}) == dict(diagnostics_by_code)
        and summary.get("artifactsByTarget") == {
            target: {
                "artifactCount": artifact_count,
                "translatedCount": translated_count,
                "failedCount": failed_count,
            }
        },
        f"{label} accounting changed",
    )


def require_clean_entry_frontier(
    mlx_root: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    target: str,
    units_by_source: Mapping[str, Mapping[str, Any]],
    entry_counts: Mapping[str, int],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """A frontier whose every discovered entry translates, with no diagnostics."""
    label = f"{target.title()} clean per-entry frontier"
    artifact_count = sum(entry_counts[source] for source in units_by_source)
    _require_summary(
        payload,
        target=target,
        unit_count=len(units_by_source),
        artifact_count=artifact_count,
        translated_count=artifact_count,
        diagnostics_by_code={},
        label=label,
    )
    require(payload.get("diagnostics") == [], f"{label} reported diagnostics")
    by_source = join_entry_artifacts(
        mlx_root,
        output_dir,
        payload,
        target=target,
        units_by_source=units_by_source,
        entry_counts=entry_counts,
    )
    for artifacts in by_source.values():
        for artifact in artifacts.values():
            _require_translated_artifact(mlx_root, artifact, label)
    return by_source


def _require_workgroup_diagnostic(diagnostic: Mapping[str, Any], entry: str) -> bool:
    details = diagnostic.get("details")
    execution = (
        details.get("executionSpecialization") if isinstance(details, Mapping) else None
    )
    return (
        diagnostic.get("checkKind") == "execution-specialization"
        and diagnostic.get("message") == WORKGROUP_SIZE_REQUIRED_MESSAGE
        and diagnostic.get("missingCapabilities")
        == ["execution.workgroup-size-specialization"]
        and execution == {
            "reason": "consumed-source-value-unconfigured",
            "sourceEntryPoints": [entry],
        }
    )


_DIAGNOSTIC_CONTRACTS = {
    WORKGROUP_SIZE_REQUIRED_CODE: _require_workgroup_diagnostic,
}


def _error_diagnostics_by_artifact(
    payload: Mapping[str, Any], target: str, label: str
) -> dict[str, Mapping[str, Any]]:
    diagnostics = payload.get("diagnostics")
    require(isinstance(diagnostics, list), f"{label} diagnostics are missing")
    by_artifact: dict[str, Mapping[str, Any]] = {}
    for diagnostic in diagnostics:
        details = diagnostic.get("details") if isinstance(diagnostic, Mapping) else None
        artifact_path = (
            details.get("targetArtifact") if isinstance(details, Mapping) else None
        )
        require(
            isinstance(artifact_path, str)
            and artifact_path not in by_artifact
            and diagnostic.get("severity") == "error"
            and diagnostic.get("target") == target
            and diagnostic.get("sourceBackend") == "metal"
            and diagnostic.get("code") in _DIAGNOSTIC_CONTRACTS,
            f"{label} diagnostic/artifact join changed",
        )
        by_artifact[artifact_path] = diagnostic
    return by_artifact


def require_entry_outcome_frontier(
    mlx_root: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    target: str,
    units_by_source: Mapping[str, Mapping[str, Any]],
    entry_counts: Mapping[str, int],
    expected_outcomes: Mapping[str, Mapping[str, int]],
    translated_entries: Mapping[str, Sequence[str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Mapping[str, Any]]]]:
    """A frontier whose entries each translate or fail with one pinned, typed diagnostic.

    ``expected_outcomes`` counts, per source, the entries that translate (``ENTRY_OUTCOME_TRANSLATED``) and the entries
    refused with each diagnostic code; ``translated_entries`` names the translated ones. A refused entry emits no
    file, records its diagnostic message as its error, and has exactly one diagnostic naming its artifact path.
    Returns the evidence by source and the artifacts by source and entry.
    """
    label = f"{target.title()} per-entry outcome frontier"
    require(
        set(expected_outcomes) == set(units_by_source)
        and all(
            sum(outcomes.values()) == entry_counts[source]
            for source, outcomes in expected_outcomes.items()
        ),
        f"{label} expected outcomes do not cover every entry",
    )
    totals: Counter[str] = Counter()
    for outcomes in expected_outcomes.values():
        totals.update(outcomes)
    translated_count = totals.pop(ENTRY_OUTCOME_TRANSLATED, 0)
    _require_summary(
        payload,
        target=target,
        unit_count=len(units_by_source),
        artifact_count=sum(entry_counts[source] for source in units_by_source),
        translated_count=translated_count,
        diagnostics_by_code=dict(totals),
        label=label,
    )
    diagnostics = _error_diagnostics_by_artifact(payload, target, label)
    by_source = join_entry_artifacts(
        mlx_root,
        output_dir,
        payload,
        target=target,
        units_by_source=units_by_source,
        entry_counts=entry_counts,
    )
    output_root = output_dir.resolve()
    evidence: dict[str, dict[str, Any]] = {}
    joined_diagnostics = 0
    for source, artifacts in by_source.items():
        observed: Counter[str] = Counter()
        translated: list[str] = []
        for entry, artifact in artifacts.items():
            artifact_path = str(artifact["path"])
            diagnostic = diagnostics.get(artifact_path)
            if artifact.get("status") == ENTRY_OUTCOME_TRANSLATED:
                require(
                    diagnostic is None,
                    f"{label} translated entry has a diagnostic: {source}:{entry}",
                )
                _require_translated_artifact(mlx_root, artifact, label)
                observed[ENTRY_OUTCOME_TRANSLATED] += 1
                translated.append(entry)
                continue
            details = diagnostic.get("details") if diagnostic is not None else None
            generated_path = (mlx_root / artifact_path).resolve()
            require(
                artifact.get("status") == "failed"
                and diagnostic is not None
                and isinstance(details, Mapping)
                and details.get("sourcePath") == source
                and artifact.get("error") == diagnostic.get("message")
                and "generatedHash" not in artifact
                and "generatedSizeBytes" not in artifact
                and _is_relative_to(generated_path, output_root)
                and not generated_path.exists(),
                f"{label} refused entry contract changed for {source}:{entry}",
            )
            code = str(diagnostic.get("code"))
            require(
                _DIAGNOSTIC_CONTRACTS[code](diagnostic, entry),
                f"{label} {code} diagnostic changed for {source}:{entry}",
            )
            observed[code] += 1
            joined_diagnostics += 1
        require(
            dict(observed) == dict(expected_outcomes[source])
            and translated == list(translated_entries.get(source, ())),
            f"{label} entry outcomes changed for {source}: {dict(sorted(observed.items()))}",
        )
        evidence[source] = {
            "entryPointCount": entry_counts[source],
            "entryOutcomes": dict(sorted(observed.items())),
            "translatedEntryPoints": translated,
        }
    require(
        joined_diagnostics == len(diagnostics),
        f"{label} reported diagnostics that name no refused entry",
    )
    return evidence, by_source


def require_entry_rule_sizes(
    artifacts_by_source: Mapping[str, Mapping[str, Mapping[str, Any]]],
    entry_rules: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    """Every translated entry an entry workgroup-size rule names reports that rule and its size; no other entry does.

    ``entry_rules`` maps a source to its ``project.entry_workgroup_size_rules`` table. An entry the table names must
    report one execution entry whose ``workgroupSize`` is the rule's constant components and whose ``rule`` records
    the matching pattern. An entry it does not name must report no project workgroup size, because the host keeps
    choosing it. Returns the sized and host-sized entries by source.
    """
    evidence: dict[str, dict[str, Any]] = {}
    for source, rules in sorted(entry_rules.items()):
        artifacts = artifacts_by_source.get(source)
        require(artifacts is not None, f"{label} has no artifacts for {source}")
        sized: dict[str, list[int]] = {}
        host_sized: list[str] = []
        for entry, artifact in artifacts.items():
            if artifact.get("status") != ENTRY_OUTCOME_TRANSLATED:
                continue
            patterns = [
                pattern for pattern in rules if fnmatch.fnmatchcase(entry, pattern)
            ]
            execution = artifact.get("execution")
            entries = (
                execution.get("entryPoints") if isinstance(execution, Mapping) else None
            )
            if not patterns:
                require(
                    not entries,
                    f"{label} reports a project workgroup size for host-sized {source}:{entry}",
                )
                host_sized.append(entry)
                continue
            require(
                len(patterns) == 1, f"{label} entry rules overlap for {source}:{entry}"
            )
            expected = [int(component) for component in rules[patterns[0]]]
            rule = (
                entries[0].get("rule")
                if isinstance(entries, list) and len(entries) == 1
                else None
            )
            require(
                isinstance(rule, Mapping)
                and entries[0].get("sourceEntryPoint") == entry
                and entries[0].get("workgroupSize") == expected
                and rule.get("entryPattern") == patterns[0]
                and rule.get("components")
                == [str(component) for component in rules[patterns[0]]],
                f"{label} entry rule size changed for {source}:{entry}",
            )
            sized[entry] = expected
        for pattern in rules:
            require(
                any(fnmatch.fnmatchcase(entry, pattern) for entry in sized),
                f"{label} entry rule {pattern!r} sized no translated entry of {source}",
            )
        evidence[source] = {
            "entryRules": {
                pattern: list(components)
                for pattern, components in sorted(rules.items())
            },
            "ruleSizedEntryPoints": sized,
            "hostSizedEntryPoints": host_sized,
        }
    return evidence
