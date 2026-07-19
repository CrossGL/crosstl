"""Runtime execution graph verification against packaged artifacts."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

from .pipeline import inspect_runtime_package
from .runtime_graph import (
    RuntimeExecutionGraph,
    RuntimeGraphArtifactSelector,
    RuntimeGraphError,
    RuntimeGraphNode,
    parse_runtime_execution_graph,
    validate_runtime_execution_graph,
)

RUNTIME_GRAPH_PACKAGE_INSPECTION_KIND = (
    "crosstl-runtime-execution-graph-package-inspection"
)
RUNTIME_GRAPH_PACKAGE_INSPECTION_SCHEMA_VERSION = 1
RUNTIME_GRAPH_PACKAGE_INSPECTION_SCOPE = "runtime-graph-package-verification"
RUNTIME_GRAPH_PACKAGE_INSPECTION_NON_GOALS = (
    "device-execution",
    "runtime-scheduling",
    "resource-allocation",
)

_DIAGNOSTIC_PREFIX = "project.runtime-graph-package"
_CHECK_KIND = "runtime-graph-package-inspection"


def inspect_runtime_graph_package(
    graph: RuntimeExecutionGraph | Mapping[str, Any],
    package_manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify graph dispatch references against a runtime package manifest.

    The inspection validates the graph before reading package metadata. It verifies
    package and host-interface metadata only; it does not execute graph nodes.
    """

    source_package = os.fspath(package_manifest_path)
    try:
        parsed_graph = (
            graph
            if isinstance(graph, RuntimeExecutionGraph)
            else parse_runtime_execution_graph(graph)
        )
    except RuntimeGraphError as error:
        diagnostic = error.to_json()
        return _inspection_payload(
            graph_id=_graph_id_hint(graph),
            source_package=source_package,
            package=None,
            references=(),
            graph_diagnostics=(diagnostic,),
            package_diagnostics=(),
            diagnostics=(diagnostic,),
        )

    graph_validation = validate_runtime_execution_graph(parsed_graph)
    graph_diagnostics = tuple(
        diagnostic.to_json() for diagnostic in graph_validation.diagnostics
    )
    if not graph_validation.valid:
        return _inspection_payload(
            graph_id=parsed_graph.graph_id,
            source_package=source_package,
            package=None,
            references=(),
            graph_diagnostics=graph_diagnostics,
            package_diagnostics=(),
            diagnostics=graph_diagnostics,
        )

    package = inspect_runtime_package(package_manifest_path)
    package_diagnostics = tuple(
        _package_diagnostic(diagnostic, graph_id=parsed_graph.graph_id)
        for diagnostic in _mapping_sequence(package.get("diagnostics"))
    )
    diagnostics: list[dict[str, Any]] = [
        *graph_diagnostics,
        *package_diagnostics,
    ]
    bindings = sorted(
        _mapping_sequence(package.get("bindings")), key=_artifact_sort_key
    )
    references: list[dict[str, Any]] = []
    for node_index, node in enumerate(parsed_graph.nodes):
        if node.kind != "dispatch" or node.dispatch is None:
            continue
        reference, reference_diagnostics = _inspect_dispatch_reference(
            node,
            node_index=node_index,
            bindings=bindings,
            graph_id=parsed_graph.graph_id,
        )
        references.append(reference)
        diagnostics.extend(reference_diagnostics)

    return _inspection_payload(
        graph_id=parsed_graph.graph_id,
        source_package=source_package,
        package=package,
        references=references,
        graph_diagnostics=graph_diagnostics,
        package_diagnostics=package_diagnostics,
        diagnostics=diagnostics,
    )


def runtime_graph_package_inspection_json(
    inspection: Mapping[str, Any], *, indent: int | None = 2
) -> str:
    """Serialize an inspection using stable key ordering and JSON formatting."""

    return json.dumps(
        _stable_json_value(inspection),
        indent=indent,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
    ) + ("" if indent is None else "\n")


def _inspect_dispatch_reference(
    node: RuntimeGraphNode,
    *,
    node_index: int,
    bindings: Sequence[Mapping[str, Any]],
    graph_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dispatch = node.dispatch
    assert dispatch is not None
    selector = dispatch.artifact_selector
    selector_path = f"$.nodes[{node_index}].dispatch.artifactSelector"
    matches = [binding for binding in bindings if _selector_matches(selector, binding)]
    candidates = [_artifact_reference(binding) for binding in matches]
    diagnostics: list[dict[str, Any]] = []
    base_reference: dict[str, Any] = {
        "nodeId": node.node_id,
        "selector": selector.to_json(),
        "matchCount": len(matches),
        "candidates": candidates,
        "artifact": None,
        "interface": None,
    }

    if not matches:
        diagnostics.append(
            _diagnostic(
                "artifact-missing",
                "No packaged artifact matches the dispatch artifact selector.",
                path=selector_path,
                graph_id=graph_id,
                node_id=node.node_id,
                details={"selector": selector.to_json()},
            )
        )
        base_reference["status"] = "missing"
        return base_reference, diagnostics

    if len(matches) > 1:
        diagnostics.append(
            _diagnostic(
                "artifact-ambiguous",
                "The dispatch artifact selector matches multiple packaged artifacts.",
                path=selector_path,
                graph_id=graph_id,
                node_id=node.node_id,
                details={
                    "selector": selector.to_json(),
                    "matches": [candidate["bindingId"] for candidate in candidates],
                },
            )
        )
        base_reference["status"] = "ambiguous"
        return base_reference, diagnostics

    selected = matches[0]
    artifact = _artifact_reference(selected)
    base_reference["artifact"] = artifact
    if not _artifact_ready(selected):
        diagnostics.append(
            _diagnostic(
                "artifact-not-ready",
                "The selected packaged artifact did not pass package inspection.",
                path=selector_path,
                graph_id=graph_id,
                node_id=node.node_id,
                artifact_id=_optional_string(selected.get("artifact")),
                details={
                    "artifactStatus": selected.get("artifactStatus"),
                    "bindingStatus": selected.get("status"),
                },
            )
        )
        base_reference["status"] = "not-ready"
        return base_reference, diagnostics

    interface, interface_diagnostics = _inspect_host_interface(
        node,
        node_index=node_index,
        selected=selected,
        graph_id=graph_id,
    )
    diagnostics.extend(interface_diagnostics)
    base_reference["interface"] = interface
    base_reference["status"] = (
        "failed"
        if any(item["severity"] == "error" for item in interface_diagnostics)
        else "ready"
    )
    return base_reference, diagnostics


def _inspect_host_interface(
    node: RuntimeGraphNode,
    *,
    node_index: int,
    selected: Mapping[str, Any],
    graph_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dispatch = node.dispatch
    assert dispatch is not None
    host_interface = selected.get("hostInterface")
    host_interface = host_interface if isinstance(host_interface, Mapping) else {}
    host_status = _optional_string(host_interface.get("status")) or "not-inspected"
    artifact_id = _optional_string(selected.get("artifact"))
    interface: dict[str, Any] = {
        "hostInterfaceStatus": host_status,
        "verificationStatus": "not-available",
        "entryPoint": None,
        "bindings": [],
    }
    if host_status != "ready":
        return interface, [
            _diagnostic(
                "host-interface-unavailable",
                "Host-interface metadata is unavailable; entry point and binding "
                "verification was not performed.",
                path=f"$.nodes[{node_index}].dispatch",
                severity="note",
                graph_id=graph_id,
                node_id=node.node_id,
                artifact_id=artifact_id,
                details={"hostInterfaceStatus": host_status},
            )
        ]

    diagnostics: list[dict[str, Any]] = []
    entry_points = sorted(
        _mapping_sequence(host_interface.get("entryPoints")),
        key=_interface_entry_sort_key,
    )
    matching_entry_points = [
        entry
        for entry in entry_points
        if entry.get("name") == dispatch.entry_point
        and (
            dispatch.artifact_selector.stage is None
            or not _optional_string(entry.get("stage"))
            or entry.get("stage") == dispatch.artifact_selector.stage
        )
    ]
    if matching_entry_points:
        entry = matching_entry_points[0]
        interface["entryPoint"] = {
            "name": entry.get("name"),
            "stage": entry.get("stage"),
            "status": "present",
        }
    else:
        interface["entryPoint"] = {
            "name": dispatch.entry_point,
            "stage": dispatch.artifact_selector.stage,
            "status": "missing",
        }
        diagnostics.append(
            _diagnostic(
                "entry-point-missing",
                f"Entry point {dispatch.entry_point!r} is not present in the "
                "selected artifact host interface.",
                path=f"$.nodes[{node_index}].dispatch.entryPoint",
                graph_id=graph_id,
                node_id=node.node_id,
                artifact_id=artifact_id,
                details={
                    "availableEntryPoints": [
                        entry.get("name")
                        for entry in entry_points
                        if _optional_string(entry.get("name"))
                    ]
                },
            )
        )

    reflected_resources: dict[str, list[Mapping[str, Any]]] = {}
    for resource in _mapping_sequence(host_interface.get("resources")):
        name = _optional_string(resource.get("name"))
        if name is not None:
            reflected_resources.setdefault(name, []).append(resource)
    reflected_names = set(reflected_resources)
    binding_results = []
    for binding_name in sorted(dispatch.bindings):
        graph_binding = dispatch.bindings[binding_name]
        matches = reflected_resources.get(binding_name, [])
        reflected_access = (
            _optional_string(matches[0].get("access")) if len(matches) == 1 else None
        )
        status = "present"
        if not matches:
            status = "missing"
        elif len(matches) > 1:
            status = "ambiguous"
        elif not _access_contains(reflected_access, graph_binding.access):
            status = "access-mismatch"
        binding_results.append(
            {
                "name": binding_name,
                "resource": graph_binding.resource_id,
                "access": graph_binding.access,
                "reflectedAccess": reflected_access,
                "status": status,
            }
        )
        binding_path = f"$.nodes[{node_index}].dispatch.bindings.{binding_name}"
        if not matches:
            diagnostics.append(
                _diagnostic(
                    "resource-binding-missing",
                    f"Resource binding {binding_name!r} is not present in the "
                    "selected artifact host interface.",
                    path=binding_path,
                    graph_id=graph_id,
                    node_id=node.node_id,
                    artifact_id=artifact_id,
                    details={
                        "availableBindings": sorted(reflected_names),
                        "resource": graph_binding.resource_id,
                    },
                )
            )
        elif len(matches) > 1:
            diagnostics.append(
                _diagnostic(
                    "resource-binding-ambiguous",
                    f"Resource binding {binding_name!r} appears more than once in "
                    "the selected artifact host interface.",
                    path=binding_path,
                    graph_id=graph_id,
                    node_id=node.node_id,
                    artifact_id=artifact_id,
                    details={"matchCount": len(matches)},
                )
            )
        elif status == "access-mismatch":
            diagnostics.append(
                _diagnostic(
                    "resource-binding-access-mismatch",
                    f"Resource binding {binding_name!r} does not provide the "
                    "access declared by the graph.",
                    path=f"{binding_path}.access",
                    graph_id=graph_id,
                    node_id=node.node_id,
                    artifact_id=artifact_id,
                    details={
                        "requiredAccess": graph_binding.access,
                        "reflectedAccess": reflected_access,
                    },
                )
            )
    interface["bindings"] = binding_results
    interface["verificationStatus"] = "failed" if diagnostics else "verified"
    return interface, diagnostics


def _selector_matches(
    selector: RuntimeGraphArtifactSelector, binding: Mapping[str, Any]
) -> bool:
    comparisons = (
        (
            selector.artifact_id,
            (binding.get("artifact"), binding.get("id")),
        ),
        (selector.source, (binding.get("source"),)),
        (selector.target, (binding.get("target"),)),
        (selector.variant, (binding.get("variant"),)),
        (selector.stage, (binding.get("stage"),)),
        (
            selector.path,
            (binding.get("sourcePath"), binding.get("packagePath")),
        ),
    )
    return all(
        expected is None or expected in actual_values
        for expected, actual_values in comparisons
    )


def _artifact_ready(binding: Mapping[str, Any]) -> bool:
    return binding.get("status") == "ready" and binding.get("artifactStatus") == "ready"


def _artifact_reference(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "bindingId": binding.get("id"),
        "artifactId": binding.get("artifact"),
        "source": binding.get("source"),
        "sourcePath": binding.get("sourcePath"),
        "packagePath": binding.get("packagePath"),
        "target": binding.get("target"),
        "variant": binding.get("variant"),
        "stage": binding.get("stage"),
        "status": binding.get("status"),
        "artifactStatus": binding.get("artifactStatus"),
    }


def _inspection_payload(
    *,
    graph_id: str | None,
    source_package: str,
    package: Mapping[str, Any] | None,
    references: Sequence[Mapping[str, Any]],
    graph_diagnostics: Sequence[Mapping[str, Any]],
    package_diagnostics: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stable_diagnostics = [_stable_json_value(item) for item in diagnostics]
    resolved_artifact_references = [
        reference
        for reference in references
        if isinstance(reference.get("artifact"), Mapping)
        and reference["artifact"].get("status") == "ready"
        and reference["artifact"].get("artifactStatus") == "ready"
    ]
    unique_artifacts = {
        artifact.get("bindingId")
        for artifact in (
            reference.get("artifact") for reference in resolved_artifact_references
        )
        if isinstance(artifact, Mapping) and artifact.get("bindingId") is not None
    }
    verified_interfaces = sum(
        1
        for reference in references
        if isinstance(reference.get("interface"), Mapping)
        and reference["interface"].get("verificationStatus") == "verified"
    )
    failed_interfaces = sum(
        1
        for reference in references
        if isinstance(reference.get("interface"), Mapping)
        and reference["interface"].get("verificationStatus") == "failed"
    )
    unavailable_interfaces = sum(
        1
        for reference in references
        if isinstance(reference.get("interface"), Mapping)
        and reference["interface"].get("verificationStatus") == "not-available"
    )
    severity_counts = {
        severity: sum(
            1
            for diagnostic in stable_diagnostics
            if diagnostic.get("severity") == severity
        )
        for severity in ("error", "warning", "note")
    }
    package_payload = {
        "inspected": package is not None,
        "kind": package.get("kind") if package is not None else None,
        "success": bool(package.get("success")) if package is not None else False,
        "sourcePackage": (
            package.get("sourcePackage") if package is not None else source_package
        ),
        "sourcePackageHash": (
            package.get("sourcePackageHash") if package is not None else None
        ),
    }
    return {
        "schemaVersion": RUNTIME_GRAPH_PACKAGE_INSPECTION_SCHEMA_VERSION,
        "kind": RUNTIME_GRAPH_PACKAGE_INSPECTION_KIND,
        "graphId": graph_id,
        "sourcePackage": source_package,
        "success": severity_counts["error"] == 0 and package_payload["success"],
        "scope": RUNTIME_GRAPH_PACKAGE_INSPECTION_SCOPE,
        "nonGoals": list(RUNTIME_GRAPH_PACKAGE_INSPECTION_NON_GOALS),
        "executionStatus": "not-performed",
        "package": package_payload,
        "summary": {
            "dispatchReferenceCount": len(references),
            "resolvedDispatchReferenceCount": sum(
                1 for reference in references if reference.get("status") == "ready"
            ),
            "failedDispatchReferenceCount": sum(
                1 for reference in references if reference.get("status") != "ready"
            ),
            "resolvedArtifactReferenceCount": len(resolved_artifact_references),
            "failedArtifactReferenceCount": (
                len(references) - len(resolved_artifact_references)
            ),
            "referencedArtifactCount": len(unique_artifacts),
            "verifiedInterfaceCount": verified_interfaces,
            "failedInterfaceCount": failed_interfaces,
            "unavailableInterfaceCount": unavailable_interfaces,
            "graphDiagnosticCount": len(graph_diagnostics),
            "packageDiagnosticCount": len(package_diagnostics),
            "diagnosticCount": len(stable_diagnostics),
            "errorCount": severity_counts["error"],
            "warningCount": severity_counts["warning"],
            "noteCount": severity_counts["note"],
        },
        "references": [_stable_json_value(reference) for reference in references],
        "graphDiagnostics": [
            _stable_json_value(diagnostic) for diagnostic in graph_diagnostics
        ],
        "packageDiagnostics": [
            _stable_json_value(diagnostic) for diagnostic in package_diagnostics
        ],
        "diagnostics": stable_diagnostics,
    }


def _package_diagnostic(
    diagnostic: Mapping[str, Any], *, graph_id: str
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    if isinstance(diagnostic.get("location"), Mapping):
        details["location"] = diagnostic["location"]
    if _optional_string(diagnostic.get("checkKind")):
        details["packageCheckKind"] = diagnostic["checkKind"]
    if isinstance(diagnostic.get("details"), Mapping):
        details["packageDetails"] = diagnostic["details"]
    return _diagnostic(
        str(diagnostic.get("code") or f"{_DIAGNOSTIC_PREFIX}.package-invalid"),
        str(diagnostic.get("message") or "Runtime package inspection failed."),
        path="$.package",
        severity=str(diagnostic.get("severity") or "error"),
        graph_id=graph_id,
        details=details,
    )


def _diagnostic(
    code: str,
    message: str,
    *,
    path: str,
    severity: str = "error",
    graph_id: str | None,
    node_id: str | None = None,
    artifact_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "severity": severity,
        "code": code if code.startswith("project.") else f"{_DIAGNOSTIC_PREFIX}.{code}",
        "message": message,
        "path": path,
        "checkKind": _CHECK_KIND,
    }
    if graph_id is not None:
        payload["graphId"] = graph_id
    if node_id is not None:
        payload["nodeId"] = node_id
    if artifact_id is not None:
        payload["artifactId"] = artifact_id
    if details:
        payload["details"] = _stable_json_value(details)
    return payload


def _graph_id_hint(graph: Any) -> str | None:
    if isinstance(graph, Mapping):
        return _optional_string(graph.get("id"))
    return None


def _mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _artifact_sort_key(binding: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(binding.get(field) or "")
        for field in (
            "artifact",
            "target",
            "variant",
            "stage",
            "source",
            "sourcePath",
            "packagePath",
            "id",
        )
    )


def _interface_entry_sort_key(entry: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("name") or ""),
        str(entry.get("stage") or ""),
        json.dumps(_stable_json_value(entry), sort_keys=True, separators=(",", ":")),
    )


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _access_contains(actual: Any, required: Any) -> bool:
    def components(value: Any) -> set[str]:
        if not isinstance(value, str):
            return set()
        normalized = value.strip().lower().replace("_", "-")
        if normalized in {"read-write", "readwrite", "write-read"}:
            return {"read", "write"}
        if normalized in {"read", "write"}:
            return {normalized}
        return set()

    required_components = components(required)
    return bool(required_components) and required_components <= components(actual)


def _stable_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_json_value(item) for item in value]
    return value
