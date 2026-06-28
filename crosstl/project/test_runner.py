"""Repository-agnostic project test-runner orchestration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from crosstl.project.runtime_verification import (
    RUNTIME_TEST_MANIFEST_KIND,
    RUNTIME_TEST_PLAN_KIND,
    RuntimeVerificationError,
    load_runtime_test_manifest,
    parse_runtime_test_manifest,
    plan_runtime_test_manifest,
    verify_runtime_test_manifest,
)
from crosstl.translator.codegen import normalize_backend_name

PROJECT_TEST_RUNNER_PLAN_KIND = "crosstl-project-test-runner-plan"
PROJECT_TEST_RUNNER_INSPECTION_KIND = "crosstl-project-test-runner-inspection"
PROJECT_TEST_RUNNER_REPORT_KIND = "crosstl-project-test-runner-report"
PROJECT_TEST_RUNNER_SCHEMA_VERSION = 1

PASSED = "passed"
PLANNED = "planned"
SKIPPED = "skipped"
RUNTIME_FAILED = "runtime-failed"


def build_project_test_runner_plan(
    artifact_report: str | os.PathLike[str] | Mapping[str, Any],
    manifest: str | os.PathLike[str] | Mapping[str, Any] | None = None,
    *,
    handoff_packages: Sequence[str | os.PathLike[str]] | None = None,
    selected_targets: Sequence[str] | None = None,
    test_commands: Sequence[Mapping[str, Any]] | None = None,
    expected_artifacts: (
        Sequence[Mapping[str, Any] | str | os.PathLike[str]] | None
    ) = None,
    environment: Mapping[str, Any] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build a read-only test-runner plan for translated runtime handoffs."""

    artifact_payload, artifact_source = _load_json_payload(
        artifact_report, label="artifact report"
    )
    runtime_plan = (
        plan_runtime_test_manifest(
            artifact_payload,
            manifest,
            project_root=project_root,
        )
        if manifest is not None
        else _empty_runtime_plan(artifact_payload, artifact_source, project_root)
    )
    runtime_manifest_payload = _runtime_manifest_payload(manifest)
    if runtime_manifest_payload is not None:
        runtime_plan = _runtime_plan_with_manifest_metadata(
            runtime_plan, runtime_manifest_payload
        )
    target_filter = {
        _normalize_target(target)
        for target in selected_targets or []
        if isinstance(target, str) and target.strip()
    }
    plan_targets = _selected_targets(artifact_payload, runtime_plan, target_filter)
    adapters = _adapter_availability(runtime_plan.get("adapters", []))
    commands = _plan_commands(
        test_commands or _commands_from_runtime_tests(runtime_plan),
        targets=plan_targets,
        adapter_by_id={adapter["id"]: adapter for adapter in adapters},
    )
    expected = _expected_artifact_records(expected_artifacts or (), artifact_payload)
    runtime_cases = _runtime_cases_with_provenance(runtime_plan.get("testCases", []))
    diagnostics = _plan_diagnostics(commands, runtime_cases)
    summary = _plan_summary(commands, runtime_cases, diagnostics)
    return {
        "schemaVersion": PROJECT_TEST_RUNNER_SCHEMA_VERSION,
        "kind": PROJECT_TEST_RUNNER_PLAN_KIND,
        "generatedAt": int(time.time()),
        "success": summary["failedCount"] == 0,
        "sourceArtifacts": {
            "path": str(artifact_source) if artifact_source is not None else None,
            "kind": artifact_payload.get("kind"),
        },
        "runtimeTestPlan": runtime_plan,
        "runtimeTestManifest": runtime_manifest_payload,
        "runtimeHandoffPackages": [
            _handoff_package_record(path) for path in handoff_packages or ()
        ],
        "selectedTargets": plan_targets,
        "environment": _environment_setup(environment or {}),
        "adapters": adapters,
        "testCommands": commands,
        "runtimeTests": runtime_cases,
        "expectedArtifacts": expected,
        "summary": summary,
        "diagnostics": diagnostics,
    }


def inspect_project_test_runner_plan(
    plan: str | os.PathLike[str] | Mapping[str, Any],
) -> dict[str, Any]:
    """Inspect a project test-runner plan without running host code."""

    plan_payload, plan_source = _load_json_payload(plan, label="test-runner plan")
    if plan_payload.get("kind") != PROJECT_TEST_RUNNER_PLAN_KIND:
        raise RuntimeVerificationError(
            f"Test-runner plan kind must be {PROJECT_TEST_RUNNER_PLAN_KIND}."
        )
    command_statuses = Counter(
        command.get("status")
        for command in _record_sequence(plan_payload, "testCommands")
    )
    runtime_statuses = Counter(
        test.get("status") for test in _record_sequence(plan_payload, "runtimeTests")
    )
    adapters = _record_sequence(plan_payload, "adapters")
    unavailable = [
        adapter
        for adapter in adapters
        if not adapter.get("availability", {}).get("available", False)
    ]
    diagnostics = list(_record_sequence(plan_payload, "diagnostics"))
    return {
        "schemaVersion": PROJECT_TEST_RUNNER_SCHEMA_VERSION,
        "kind": PROJECT_TEST_RUNNER_INSPECTION_KIND,
        "generatedAt": int(time.time()),
        "success": not any(
            diagnostic.get("severity") == "error" for diagnostic in diagnostics
        ),
        "sourcePlan": str(plan_source) if plan_source is not None else None,
        "summary": {
            "targetCount": len(plan_payload.get("selectedTargets", [])),
            "adapterCount": len(adapters),
            "unavailableAdapterCount": len(unavailable),
            "commandCount": sum(command_statuses.values()),
            "plannedCommandCount": command_statuses[PLANNED],
            "skippedCommandCount": command_statuses[SKIPPED],
            "runtimeTestCount": sum(runtime_statuses.values()),
            "plannedRuntimeTestCount": runtime_statuses[PLANNED],
            "skippedRuntimeTestCount": runtime_statuses[SKIPPED],
            "expectedArtifactCount": len(
                _record_sequence(plan_payload, "expectedArtifacts")
            ),
            "diagnosticCount": len(diagnostics),
        },
        "unavailableAdapters": unavailable,
        "diagnostics": diagnostics,
    }


def execute_project_test_runner_plan(
    plan: str | os.PathLike[str] | Mapping[str, Any],
    *,
    project_root: str | os.PathLike[str] | None = None,
    output_path: str | os.PathLike[str] | None = None,
    run_runtime_tests: bool = True,
    runtime_executors: Mapping[str, Any] | Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Run available project commands and runtime tests from a plan."""

    plan_payload, plan_source = _load_json_payload(plan, label="test-runner plan")
    if plan_payload.get("kind") != PROJECT_TEST_RUNNER_PLAN_KIND:
        raise RuntimeVerificationError(
            f"Test-runner plan kind must be {PROJECT_TEST_RUNNER_PLAN_KIND}."
        )
    root = Path(project_root or plan_payload.get("environment", {}).get("cwd") or ".")
    adapter_by_id = {
        adapter["id"]: adapter for adapter in _record_sequence(plan_payload, "adapters")
    }
    command_results = [
        _execute_command(command, root_path=root, adapter_by_id=adapter_by_id)
        for command in _record_sequence(plan_payload, "testCommands")
    ]
    runtime_report = None
    runtime_plan = plan_payload.get("runtimeTestPlan")
    if (
        run_runtime_tests
        and isinstance(runtime_plan, Mapping)
        and runtime_plan.get("kind") == RUNTIME_TEST_PLAN_KIND
    ):
        runtime_report = _execute_runtime_plan(
            runtime_plan,
            plan_payload.get("runtimeTestManifest"),
            root,
            artifact_source=plan_payload.get("sourceArtifacts"),
            executors=runtime_executors,
        )
    results = list(command_results)
    if isinstance(runtime_report, Mapping):
        results.extend(_runtime_report_results(runtime_report))
    summary = _execution_summary(results)
    report = {
        "schemaVersion": PROJECT_TEST_RUNNER_SCHEMA_VERSION,
        "kind": PROJECT_TEST_RUNNER_REPORT_KIND,
        "generatedAt": int(time.time()),
        "success": summary["failedCount"] == 0,
        "sourcePlan": str(plan_source) if plan_source is not None else None,
        "sourceArtifacts": plan_payload.get("sourceArtifacts"),
        "runtimeHandoffPackages": plan_payload.get("runtimeHandoffPackages", []),
        "selectedTargets": plan_payload.get("selectedTargets", []),
        "summary": summary,
        "results": results,
    }
    if runtime_report is not None:
        report["runtimeTestReport"] = runtime_report
    if output_path is not None:
        write_project_test_runner_report(report, output_path)
    return report


def write_project_test_runner_report(
    report: Mapping[str, Any], output_path: str | os.PathLike[str]
) -> None:
    """Write a project test-runner report as deterministic JSON."""

    path = Path(os.fspath(output_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _empty_runtime_plan(
    artifact_payload: Mapping[str, Any],
    artifact_source: Path | None,
    project_root: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    _ = artifact_payload
    return {
        "schemaVersion": PROJECT_TEST_RUNNER_SCHEMA_VERSION,
        "kind": RUNTIME_TEST_PLAN_KIND,
        "generatedAt": int(time.time()),
        "success": True,
        "sourceArtifacts": {
            "path": str(artifact_source) if artifact_source is not None else None,
            "kind": artifact_payload.get("kind"),
        },
        "sourceManifest": None,
        "projectRoot": str(project_root) if project_root is not None else None,
        "summary": {
            "testCount": 0,
            "plannedCount": 0,
            "skippedCount": 0,
            "unavailableCount": 0,
            "translationFailedCount": 0,
            "runtimeFailedCount": 0,
            "comparisonFailedCount": 0,
            "failedCount": 0,
        },
        "adapters": [],
        "testCases": [],
    }


def _runtime_manifest_payload(
    manifest: str | os.PathLike[str] | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if manifest is None:
        return None
    if isinstance(manifest, Mapping):
        return parse_runtime_test_manifest(manifest).to_json()
    return load_runtime_test_manifest(manifest).to_json()


def _runtime_plan_with_manifest_metadata(
    runtime_plan: Mapping[str, Any], manifest_payload: Mapping[str, Any]
) -> dict[str, Any]:
    metadata_by_id = {}
    for test in _record_sequence(manifest_payload, "tests"):
        fixture_id = test.get("id")
        metadata = test.get("metadata")
        if isinstance(fixture_id, str) and isinstance(metadata, Mapping):
            metadata_by_id[fixture_id] = dict(metadata)
    if not metadata_by_id:
        return dict(runtime_plan)
    payload = dict(runtime_plan)
    cases = []
    for test_case in _record_sequence(runtime_plan, "testCases"):
        case_payload = dict(test_case)
        fixture_id = case_payload.get("fixture")
        if isinstance(fixture_id, str) and fixture_id in metadata_by_id:
            case_payload["metadata"] = metadata_by_id[fixture_id]
        cases.append(case_payload)
    payload["testCases"] = cases
    return payload


def _load_json_payload(
    value: str | os.PathLike[str] | Mapping[str, Any], *, label: str
) -> tuple[Mapping[str, Any], Path | None]:
    if isinstance(value, Mapping):
        return value, None
    path = Path(os.fspath(value))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeVerificationError(f"{label} could not be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeVerificationError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeVerificationError(f"{label} must be a JSON object.")
    return payload, path


def _normalize_target(value: str) -> str:
    return normalize_backend_name(value) or value.strip().lower()


def _selected_targets(
    artifact_payload: Mapping[str, Any],
    runtime_plan: Mapping[str, Any],
    target_filter: set[str],
) -> list[str]:
    targets: set[str] = set()
    for artifact in _record_sequence(artifact_payload, "artifacts"):
        target = artifact.get("target")
        if isinstance(target, str) and target.strip():
            targets.add(_normalize_target(target))
    for test_case in _record_sequence(runtime_plan, "testCases"):
        selector = test_case.get("selector")
        target = selector.get("target") if isinstance(selector, Mapping) else None
        if isinstance(target, str) and target.strip():
            targets.add(_normalize_target(target))
    if target_filter:
        targets &= target_filter
    return sorted(targets)


def _adapter_availability(adapters: Sequence[Any]) -> list[dict[str, Any]]:
    records = []
    for adapter in adapters:
        if not isinstance(adapter, Mapping):
            continue
        requirements = adapter.get("platformRequirements", {})
        if not isinstance(requirements, Mapping):
            requirements = {}
        tools = [
            str(tool)
            for tool in requirements.get("requiredTools", [])
            if isinstance(tool, str) and tool.strip()
        ]
        environment = [
            str(name)
            for name in requirements.get("requiredEnvironment", [])
            if isinstance(name, str) and name.strip()
        ]
        missing_tools = [tool for tool in tools if shutil.which(tool) is None]
        missing_environment = [name for name in environment if not os.environ.get(name)]
        record = dict(adapter)
        record["availability"] = {
            "available": not missing_tools and not missing_environment,
            "requiredTools": tools,
            "missingTools": missing_tools,
            "requiredEnvironment": environment,
            "missingEnvironment": missing_environment,
        }
        records.append(record)
    return records


def _commands_from_runtime_tests(
    runtime_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    commands = []
    for test_case in _record_sequence(runtime_plan, "testCases"):
        fixture = test_case.get("fixture")
        if not isinstance(fixture, str) or not fixture:
            continue
        metadata = test_case.get("metadata") if isinstance(test_case, Mapping) else {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        upstream_name = metadata.get("upstreamTestName", fixture)
        command = metadata.get("testCommand")
        adapter_id = _adapter_reference_id(test_case.get("adapter"))
        platform_requirements = test_case.get("platformRequirements")
        if not isinstance(platform_requirements, Mapping):
            platform_requirements = {}
        commands.append(
            {
                "name": str(upstream_name),
                "command": command if isinstance(command, Sequence) else [],
                "adapter": adapter_id,
                "fixture": fixture,
                "targets": _case_targets(test_case),
                "platformRequirements": dict(platform_requirements),
                "provenance": _case_provenance(test_case),
            }
        )
    return commands


def _plan_commands(
    commands: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[str],
    adapter_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    target_set = set(targets)
    planned = []
    for index, command in enumerate(commands):
        if not isinstance(command, Mapping):
            continue
        command_targets = [
            _normalize_target(target)
            for target in command.get("targets", [])
            if isinstance(target, str) and target.strip()
        ]
        if (
            target_set
            and command_targets
            and not target_set.intersection(command_targets)
        ):
            continue
        adapter_ref = command.get("adapter")
        adapter_id = _adapter_reference_id(adapter_ref)
        adapter = adapter_by_id.get(adapter_id) if adapter_id is not None else None
        if adapter is None and isinstance(adapter_ref, Mapping):
            adapter = adapter_ref
        availability = adapter.get("availability", {}) if adapter is not None else {}
        required = _command_requirements(command, adapter)
        missing_tools = [
            tool for tool in required["requiredTools"] if shutil.which(tool) is None
        ]
        missing_environment = [
            name for name in required["requiredEnvironment"] if not os.environ.get(name)
        ]
        missing_tools = sorted(
            set(missing_tools) | set(availability.get("missingTools", []))
        )
        missing_environment = sorted(
            set(missing_environment) | set(availability.get("missingEnvironment", []))
        )
        status = SKIPPED if missing_tools or missing_environment else PLANNED
        record = {
            "id": str(command.get("id") or f"command-{index + 1}"),
            "name": str(
                command.get("name") or command.get("fixture") or f"command-{index + 1}"
            ),
            "command": _command_parts(command.get("command")),
            "cwd": str(command.get("cwd", ".")),
            "targets": command_targets,
            "adapter": adapter_id,
            "fixture": command.get("fixture"),
            "status": status,
            "requiredAdapters": [adapter_id] if isinstance(adapter_id, str) else [],
            "requiredTools": required["requiredTools"],
            "requiredEnvironment": required["requiredEnvironment"],
            "expectedArtifacts": list(command.get("expectedArtifacts", [])),
            "provenance": (
                dict(command.get("provenance", {}))
                if isinstance(command.get("provenance"), Mapping)
                else {}
            ),
            "diagnostics": [],
        }
        if missing_tools or missing_environment:
            record["failurePhase"] = "adapter-availability"
            record["diagnostics"].append(
                {
                    "severity": "note",
                    "code": "project.test-runner.adapter-unavailable",
                    "message": (
                        "Project test command skipped because required adapters or tools are unavailable."
                    ),
                    "missingTools": missing_tools,
                    "missingEnvironment": missing_environment,
                    "adapter": adapter_id,
                    "fixture": command.get("fixture"),
                }
            )
        planned.append(record)
    return planned


def _adapter_reference_id(adapter_ref: Any) -> str | None:
    if isinstance(adapter_ref, str) and adapter_ref.strip():
        return adapter_ref
    if isinstance(adapter_ref, Mapping):
        adapter_id = adapter_ref.get("id")
        if isinstance(adapter_id, str) and adapter_id.strip():
            return adapter_id
    return None


def _command_requirements(
    command: Mapping[str, Any], adapter: Mapping[str, Any] | None
) -> dict[str, list[str]]:
    requirements = command.get("platformRequirements", {})
    if not isinstance(requirements, Mapping):
        requirements = {}
    adapter_requirements = (
        adapter.get("platformRequirements", {}) if adapter is not None else {}
    )
    if not isinstance(adapter_requirements, Mapping):
        adapter_requirements = {}
    return {
        "requiredTools": _unique_strings(
            adapter_requirements.get("requiredTools", ()),
            requirements.get("requiredTools", command.get("requiredTools", ())),
        ),
        "requiredEnvironment": _unique_strings(
            adapter_requirements.get("requiredEnvironment", ()),
            requirements.get(
                "requiredEnvironment", command.get("requiredEnvironment", ())
            ),
        ),
    }


def _command_parts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(part) for part in value if str(part)]
    return []


def _unique_strings(*values: Any) -> list[str]:
    result = []
    for value in values:
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, Sequence):
            candidates = value
        else:
            candidates = []
        for candidate in candidates:
            if (
                isinstance(candidate, str)
                and candidate.strip()
                and candidate not in result
            ):
                result.append(candidate)
    return result


def _expected_artifact_records(
    expected: Sequence[Mapping[str, Any] | str | os.PathLike[str]],
    artifact_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for index, item in enumerate(expected):
        if isinstance(item, Mapping):
            record = dict(item)
        else:
            record = {"path": os.fspath(item)}
        record.setdefault("id", f"expected-artifact-{index + 1}")
        record.setdefault("required", True)
        records.append(record)
    if records:
        return records
    for artifact in _record_sequence(artifact_payload, "artifacts"):
        path = artifact.get("path")
        if isinstance(path, str) and path:
            records.append(
                {
                    "id": artifact.get("id") or path,
                    "path": path,
                    "backend": artifact.get("target"),
                    "sourceFile": artifact.get("source"),
                    "required": artifact.get("status") == "translated",
                }
            )
    return records


def _runtime_cases_with_provenance(test_cases: Sequence[Any]) -> list[dict[str, Any]]:
    cases = []
    for test_case in test_cases:
        if not isinstance(test_case, Mapping):
            continue
        record = dict(test_case)
        record["provenance"] = _case_provenance(test_case)
        cases.append(record)
    return cases


def _case_targets(test_case: Mapping[str, Any]) -> list[str]:
    selector = test_case.get("selector")
    target = selector.get("target") if isinstance(selector, Mapping) else None
    return (
        [_normalize_target(target)]
        if isinstance(target, str) and target.strip()
        else []
    )


def _case_provenance(test_case: Mapping[str, Any]) -> dict[str, Any]:
    selector = test_case.get("selector")
    artifact = test_case.get("artifact")
    metadata = test_case.get("metadata")
    if not isinstance(selector, Mapping):
        selector = {}
    if not isinstance(artifact, Mapping):
        artifact = {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "sourceFile": artifact.get("source") or selector.get("source"),
        "generatedArtifact": artifact.get("path") or selector.get("path"),
        "backend": artifact.get("target") or selector.get("target"),
        "fixture": test_case.get("fixture"),
        "upstreamTestName": metadata.get("upstreamTestName", test_case.get("fixture")),
        "logs": artifact.get("toolchainRuns", []),
        "adapterAvailability": test_case.get("adapter"),
    }


def _handoff_package_record(path_value: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(os.fspath(path_value))
    exists = path.exists()
    return {
        "path": str(path),
        "available": exists,
        "kind": "runtime-handoff-package",
    }


def _environment_setup(environment: Mapping[str, Any]) -> dict[str, Any]:
    cwd = environment.get("cwd", ".")
    variables = environment.get("variables", {})
    if not isinstance(variables, Mapping):
        variables = {}
    setup_commands = environment.get("setupCommands", [])
    return {
        "cwd": str(cwd),
        "variables": {str(key): str(value) for key, value in variables.items()},
        "setupCommands": [
            _command_parts(command) for command in setup_commands if command
        ],
    }


def _plan_diagnostics(
    commands: Sequence[Mapping[str, Any]], runtime_cases: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    if not commands and not runtime_cases:
        diagnostics.append(
            {
                "severity": "note",
                "code": "project.test-runner.no-tests-selected",
                "message": (
                    "No project test commands or runtime fixtures were selected."
                ),
            }
        )
    for command in commands:
        diagnostics.extend(command.get("diagnostics", []))
    for runtime_case in runtime_cases:
        diagnostics.extend(runtime_case.get("diagnostics", []))
    return diagnostics


def _plan_summary(
    commands: Sequence[Mapping[str, Any]],
    runtime_cases: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    command_statuses = Counter(command.get("status") for command in commands)
    runtime_statuses = Counter(test.get("status") for test in runtime_cases)
    failed = command_statuses[RUNTIME_FAILED] + runtime_statuses[RUNTIME_FAILED]
    return {
        "commandCount": len(commands),
        "plannedCommandCount": command_statuses[PLANNED],
        "skippedCommandCount": command_statuses[SKIPPED],
        "runtimeTestCount": len(runtime_cases),
        "plannedRuntimeTestCount": runtime_statuses[PLANNED],
        "skippedRuntimeTestCount": runtime_statuses[SKIPPED],
        "diagnosticCount": len(diagnostics),
        "failedCount": failed,
    }


def _execute_command(
    command: Mapping[str, Any],
    *,
    root_path: Path,
    adapter_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    adapter_id = command.get("adapter")
    adapter = adapter_by_id.get(adapter_id) if isinstance(adapter_id, str) else None
    availability = adapter.get("availability", {}) if adapter is not None else {}
    diagnostics = list(command.get("diagnostics", []))
    if command.get("status") == SKIPPED or not availability.get("available", True):
        return {
            "kind": "project-test-command",
            "name": command.get("name"),
            "fixture": command.get("fixture"),
            "status": SKIPPED,
            "failurePhase": command.get("failurePhase", "adapter-availability"),
            "provenance": command.get("provenance", {}),
            "logs": [],
            "adapterAvailability": availability,
            "diagnostics": diagnostics,
        }
    parts = _command_parts(command.get("command"))
    if not parts:
        return {
            "kind": "project-test-command",
            "name": command.get("name"),
            "fixture": command.get("fixture"),
            "status": SKIPPED,
            "failurePhase": "test-command",
            "provenance": command.get("provenance", {}),
            "logs": [],
            "adapterAvailability": availability,
            "diagnostics": [
                *diagnostics,
                {
                    "severity": "note",
                    "code": "project.test-runner.command-empty",
                    "message": (
                        "Project test command skipped because no command was provided."
                    ),
                },
            ],
        }
    cwd = root_path / str(command.get("cwd", "."))
    result = subprocess.run(
        parts,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    status = PASSED if result.returncode == 0 else RUNTIME_FAILED
    return {
        "kind": "project-test-command",
        "name": command.get("name"),
        "fixture": command.get("fixture"),
        "status": status,
        "failurePhase": None if status == PASSED else "test-command",
        "provenance": command.get("provenance", {}),
        "logs": [
            {
                "command": parts,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        ],
        "adapterAvailability": availability,
        "diagnostics": diagnostics,
    }


def _execute_runtime_plan(
    runtime_plan: Mapping[str, Any],
    manifest_payload: Any,
    root: Path,
    *,
    artifact_source: Any = None,
    executors: Mapping[str, Any] | Sequence[Any] | None = None,
) -> Mapping[str, Any] | None:
    manifest = (
        dict(manifest_payload)
        if isinstance(manifest_payload, Mapping)
        else _manifest_from_runtime_plan(runtime_plan)
    )
    if not manifest.get("tests"):
        return None
    runtime_artifact_source = runtime_plan.get("sourceArtifacts", {})
    artifact_path = (
        runtime_artifact_source.get("path")
        if isinstance(runtime_artifact_source, Mapping)
        else None
    )
    if not artifact_path:
        artifact_path = (
            artifact_source.get("path")
            if isinstance(artifact_source, Mapping)
            else None
        )
    if not artifact_path:
        return None
    return verify_runtime_test_manifest(
        artifact_path,
        manifest,
        project_root=root,
        executors=executors,
    )


def _manifest_from_runtime_plan(runtime_plan: Mapping[str, Any]) -> dict[str, Any]:
    adapters = [
        _strip_availability(adapter)
        for adapter in _record_sequence(runtime_plan, "adapters")
    ]
    tests = []
    for test_case in _record_sequence(runtime_plan, "testCases"):
        fixture = {
            "id": test_case.get("fixture"),
            "selector": test_case.get("selector", {}),
            "adapter": test_case.get("adapter"),
        }
        metadata = test_case.get("metadata")
        if isinstance(metadata, Mapping):
            fixture["metadata"] = dict(metadata)
        tests.append(fixture)
    manifest = {
        "kind": RUNTIME_TEST_MANIFEST_KIND,
        "adapters": adapters,
        "tests": tests,
    }
    parse_runtime_test_manifest(manifest)
    return manifest


def _strip_availability(adapter: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(adapter)
    payload.pop("availability", None)
    return payload


def _runtime_report_results(runtime_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = []
    for result in _record_sequence(runtime_report, "results"):
        record = dict(result)
        record.setdefault("kind", "runtime-test-fixture")
        logs = []
        artifact = result.get("artifact")
        if isinstance(artifact, Mapping):
            logs.extend(artifact.get("toolchainRuns", []))
        record["logs"] = logs
        record["provenance"] = {
            "sourceFile": (
                artifact.get("source") if isinstance(artifact, Mapping) else None
            ),
            "generatedArtifact": (
                artifact.get("path") if isinstance(artifact, Mapping) else None
            ),
            "backend": (
                artifact.get("target") if isinstance(artifact, Mapping) else None
            ),
            "fixture": result.get("fixture"),
        }
        results.append(record)
    return results


def _execution_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    statuses = Counter(result.get("status") for result in results)
    failed = statuses[RUNTIME_FAILED] + statuses["comparison-failed"]
    return {
        "resultCount": len(results),
        "passedCount": statuses[PASSED],
        "skippedCount": statuses[SKIPPED],
        "runtimeFailedCount": statuses[RUNTIME_FAILED],
        "comparisonFailedCount": statuses["comparison-failed"],
        "failedCount": failed,
    }


def _record_sequence(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    values = payload.get(key, [])
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [value for value in values if isinstance(value, Mapping)]
