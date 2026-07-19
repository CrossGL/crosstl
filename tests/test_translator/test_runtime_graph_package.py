import hashlib
import json
import textwrap

import crosstl.project as project_api
from crosstl.project.pipeline import RUNTIME_PACKAGE_KIND
from crosstl.project.runtime_graph import (
    RUNTIME_EXECUTION_GRAPH_KIND,
    RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION,
    parse_runtime_execution_graph,
)
from crosstl.project.runtime_graph_package import (
    RUNTIME_GRAPH_PACKAGE_INSPECTION_KIND,
    inspect_runtime_graph_package,
    runtime_graph_package_inspection_json,
)


def _layout():
    return {
        "storageLayout": "contiguous",
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
        "byteOffset": 0,
        "byteLength": 16,
    }


def _resource(resource_id, role, *, lifetime=None):
    payload = {
        "id": resource_id,
        "role": role,
        "kind": "buffer",
        "dtype": "float32",
        "shape": [4],
        "physicalLayout": _layout(),
    }
    if lifetime is not None:
        payload["lifetime"] = {
            "firstNode": lifetime[0],
            "lastNode": lifetime[1],
        }
    return payload


def _dispatch(
    node_id,
    selector,
    entry_point,
    bindings,
    *,
    depends_on=(),
):
    return {
        "id": node_id,
        "kind": "dispatch",
        "dependsOn": list(depends_on),
        "dispatch": {
            "artifactSelector": selector,
            "entryPoint": entry_point,
            "bindings": {
                name: {"resource": resource, "access": access}
                for name, (resource, access) in bindings.items()
            },
            "constants": {},
            "geometry": {"workgroupCount": [1, 1, 1]},
        },
    }


def _single_dispatch_graph(
    *, selector=None, entry_point="main", binding_name="result", access="write"
):
    return {
        "kind": RUNTIME_EXECUTION_GRAPH_KIND,
        "schemaVersion": RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION,
        "id": "package.fixture",
        "resources": [_resource("output", "external-output")],
        "nodes": [
            _dispatch(
                "launch",
                selector or {"id": "directx|kernel.hlsl"},
                entry_point,
                {binding_name: ("output", access)},
            )
        ],
    }


def _two_dispatch_graph():
    return {
        "kind": RUNTIME_EXECUTION_GRAPH_KIND,
        "schemaVersion": RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION,
        "id": "package.two-stage",
        "resources": [
            _resource("input", "external-input"),
            _resource("partial", "temporary", lifetime=("first", "second")),
            _resource("output", "external-output"),
        ],
        "nodes": [
            _dispatch(
                "first",
                {"id": "directx|first.hlsl"},
                "first_stage",
                {
                    "source": ("input", "read"),
                    "partial": ("partial", "write"),
                },
            ),
            {
                "id": "partial-visible",
                "kind": "barrier",
                "dependsOn": ["first"],
                "barrier": {
                    "resources": ["partial"],
                    "beforeAccess": "write",
                    "afterAccess": "read",
                },
            },
            _dispatch(
                "second",
                {"id": "directx|second.hlsl"},
                "second_stage",
                {
                    "partial": ("partial", "read"),
                    "result": ("output", "write"),
                },
                depends_on=("partial-visible",),
            ),
        ],
    }


def _hlsl(entry_point="main", resources=None):
    declarations = resources or ["RWStructuredBuffer<float> result : register(u0);"]
    return textwrap.dedent(f"""
        {chr(10).join(declarations)}

        [numthreads(4, 1, 1)]
        void {entry_point}(uint3 tid : SV_DispatchThreadID) {{
        }}
        """).strip()


def _write_package(tmp_path, artifacts):
    package_dir = tmp_path / "runtime-package"
    package_artifacts = []
    for artifact in artifacts:
        artifact_path = package_dir / artifact["packagePath"]
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(artifact["sourceCode"], encoding="utf-8")
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        package_artifacts.append(
            {
                "id": artifact["id"],
                "status": "packaged",
                "source": artifact.get("source", "mlx/backend/metal/kernel.metal"),
                "sourcePath": artifact.get(
                    "sourcePath", f"out/directx/{artifact_path.name}"
                ),
                "packagePath": artifact["packagePath"],
                "target": artifact.get("target", "directx"),
                "sourceBackend": "metal",
                "stage": artifact.get("stage", "compute"),
                "variant": artifact.get("variant", "default"),
                "defines": {},
                "hash": {"algorithm": "sha256", "value": digest},
                "sizeBytes": artifact_path.stat().st_size,
                "sourceRemap": None,
                "hostInterface": None,
            }
        )
    package = {
        "schemaVersion": 1,
        "kind": RUNTIME_PACKAGE_KIND,
        "success": True,
        "packageRoot": str(package_dir),
        "project": {"targets": ["directx"]},
        "runtimePlan": {"runtimeReferenceCount": 0},
        "artifacts": package_artifacts,
    }
    manifest_path = package_dir / "runtime-package.json"
    manifest_path.write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _single_artifact(entry_point="main", resources=None):
    return {
        "id": "directx|kernel.hlsl",
        "packagePath": "artifacts/directx/kernel.hlsl",
        "sourceCode": _hlsl(entry_point, resources),
    }


def _codes(inspection):
    return [diagnostic["code"] for diagnostic in inspection["diagnostics"]]


def test_project_api_exports_runtime_graph_package_inspection():
    assert project_api.inspect_runtime_graph_package is inspect_runtime_graph_package
    assert (
        project_api.runtime_graph_package_inspection_json
        is runtime_graph_package_inspection_json
    )


def test_inspection_resolves_two_dispatches_and_verifies_host_interfaces(tmp_path):
    manifest_path = _write_package(
        tmp_path,
        [
            {
                "id": "directx|first.hlsl",
                "packagePath": "artifacts/directx/first.hlsl",
                "sourceCode": _hlsl(
                    "first_stage",
                    [
                        "StructuredBuffer<float> source : register(t0);",
                        "RWStructuredBuffer<float> partial : register(u0);",
                    ],
                ),
            },
            {
                "id": "directx|second.hlsl",
                "packagePath": "artifacts/directx/second.hlsl",
                "sourceCode": _hlsl(
                    "second_stage",
                    [
                        "StructuredBuffer<float> partial : register(t0);",
                        "RWStructuredBuffer<float> result : register(u0);",
                    ],
                ),
            },
        ],
    )

    inspection = inspect_runtime_graph_package(_two_dispatch_graph(), manifest_path)

    assert inspection["kind"] == RUNTIME_GRAPH_PACKAGE_INSPECTION_KIND
    assert inspection["success"] is True
    assert inspection["executionStatus"] == "not-performed"
    assert inspection["nonGoals"] == [
        "device-execution",
        "runtime-scheduling",
        "resource-allocation",
    ]
    assert inspection["summary"] == {
        "dispatchReferenceCount": 2,
        "resolvedDispatchReferenceCount": 2,
        "failedDispatchReferenceCount": 0,
        "resolvedArtifactReferenceCount": 2,
        "failedArtifactReferenceCount": 0,
        "referencedArtifactCount": 2,
        "verifiedInterfaceCount": 2,
        "failedInterfaceCount": 0,
        "unavailableInterfaceCount": 0,
        "graphDiagnosticCount": 0,
        "packageDiagnosticCount": 0,
        "diagnosticCount": 0,
        "errorCount": 0,
        "warningCount": 0,
        "noteCount": 0,
    }
    assert [reference["nodeId"] for reference in inspection["references"]] == [
        "first",
        "second",
    ]
    assert [
        reference["artifact"]["artifactId"] for reference in inspection["references"]
    ] == ["directx|first.hlsl", "directx|second.hlsl"]
    assert all(
        reference["interface"]["verificationStatus"] == "verified"
        for reference in inspection["references"]
    )


def test_inspection_reports_missing_artifact_with_selector_path(tmp_path):
    manifest_path = _write_package(tmp_path, [_single_artifact()])
    graph = _single_dispatch_graph(selector={"id": "directx|missing.hlsl"})

    inspection = inspect_runtime_graph_package(graph, manifest_path)

    assert inspection["success"] is False
    assert inspection["references"][0]["status"] == "missing"
    diagnostic = inspection["diagnostics"][0]
    assert diagnostic["code"] == "project.runtime-graph-package.artifact-missing"
    assert diagnostic["path"] == "$.nodes[0].dispatch.artifactSelector"


def test_inspection_rejects_ambiguous_artifact_selectors_deterministically(tmp_path):
    manifest_path = _write_package(
        tmp_path,
        [
            _single_artifact(),
            {
                "id": "directx|other.hlsl",
                "packagePath": "artifacts/directx/other.hlsl",
                "sourceCode": _hlsl(),
            },
        ],
    )
    graph = _single_dispatch_graph(selector={"target": "directx"})

    inspection = inspect_runtime_graph_package(graph, manifest_path)

    reference = inspection["references"][0]
    assert reference["status"] == "ambiguous"
    assert [candidate["artifactId"] for candidate in reference["candidates"]] == [
        "directx|kernel.hlsl",
        "directx|other.hlsl",
    ]
    assert "project.runtime-graph-package.artifact-ambiguous" in _codes(inspection)


def test_inspection_rejects_selected_artifact_that_fails_integrity_check(tmp_path):
    manifest_path = _write_package(tmp_path, [_single_artifact()])
    artifact_path = manifest_path.parent / "artifacts/directx/kernel.hlsl"
    artifact_path.write_text(_hlsl() + "\n// changed\n", encoding="utf-8")

    inspection = inspect_runtime_graph_package(_single_dispatch_graph(), manifest_path)

    assert inspection["success"] is False
    assert inspection["references"][0]["status"] == "not-ready"
    assert inspection["references"][0]["artifact"]["artifactStatus"] == "failed"
    assert "project.runtime-package-inspection.artifact-hash-mismatch" in _codes(
        inspection
    )
    assert "project.runtime-graph-package.artifact-not-ready" in _codes(inspection)


def test_inspection_reports_missing_entry_point_from_ready_interface(tmp_path):
    manifest_path = _write_package(
        tmp_path, [_single_artifact(entry_point="actual_entry")]
    )
    graph = _single_dispatch_graph(entry_point="expected_entry")

    inspection = inspect_runtime_graph_package(graph, manifest_path)

    assert inspection["references"][0]["interface"]["verificationStatus"] == ("failed")
    diagnostic = next(
        diagnostic
        for diagnostic in inspection["diagnostics"]
        if diagnostic["code"] == "project.runtime-graph-package.entry-point-missing"
    )
    assert diagnostic["path"] == "$.nodes[0].dispatch.entryPoint"
    assert diagnostic["details"]["availableEntryPoints"] == ["actual_entry"]


def test_inspection_reports_missing_named_interface_binding(tmp_path):
    manifest_path = _write_package(tmp_path, [_single_artifact()])
    graph = _single_dispatch_graph(binding_name="missingOutput")

    inspection = inspect_runtime_graph_package(graph, manifest_path)

    diagnostic = next(
        diagnostic
        for diagnostic in inspection["diagnostics"]
        if diagnostic["code"]
        == "project.runtime-graph-package.resource-binding-missing"
    )
    assert diagnostic["path"] == ("$.nodes[0].dispatch.bindings.missingOutput")
    assert diagnostic["details"]["availableBindings"] == ["result"]
    assert inspection["references"][0]["interface"]["bindings"] == [
        {
            "name": "missingOutput",
            "resource": "output",
            "access": "write",
            "reflectedAccess": None,
            "status": "missing",
        }
    ]


def test_inspection_rejects_reflected_binding_access_mismatch(tmp_path):
    artifact = _single_artifact(
        resources=["StructuredBuffer<float> result : register(t0);"]
    )
    manifest_path = _write_package(tmp_path, [artifact])

    inspection = inspect_runtime_graph_package(_single_dispatch_graph(), manifest_path)

    reference = inspection["references"][0]
    assert reference["interface"]["bindings"] == [
        {
            "name": "result",
            "resource": "output",
            "access": "write",
            "reflectedAccess": "read",
            "status": "access-mismatch",
        }
    ]
    diagnostic = next(
        item
        for item in inspection["diagnostics"]
        if item["code"]
        == "project.runtime-graph-package.resource-binding-access-mismatch"
    )
    assert diagnostic["path"] == "$.nodes[0].dispatch.bindings.result.access"
    assert diagnostic["details"] == {
        "requiredAccess": "write",
        "reflectedAccess": "read",
    }


def test_invalid_graph_preserves_diagnostics_and_stops_before_package_inspection(
    tmp_path,
):
    graph = _single_dispatch_graph()
    graph["nodes"] = []
    missing_manifest = tmp_path / "not-read.json"

    inspection = inspect_runtime_graph_package(graph, missing_manifest)

    expected = parse_runtime_execution_graph(graph).validate().to_json()["diagnostics"]
    assert inspection["success"] is False
    assert inspection["package"]["inspected"] is False
    assert inspection["references"] == []
    assert inspection["graphDiagnostics"] == expected
    assert inspection["diagnostics"] == expected
    assert "project.runtime-graph.nodes-empty" in _codes(inspection)


def test_inspection_json_is_deterministic_for_mapping_and_parsed_graph(tmp_path):
    manifest_path = _write_package(tmp_path, [_single_artifact()])
    graph_payload = _single_dispatch_graph()

    first = inspect_runtime_graph_package(graph_payload, manifest_path)
    second = inspect_runtime_graph_package(
        parse_runtime_execution_graph(graph_payload), manifest_path
    )
    first_json = runtime_graph_package_inspection_json(first)
    second_json = runtime_graph_package_inspection_json(second)

    assert first == second
    assert first_json == second_json
    assert first_json.endswith("\n")
    assert json.loads(first_json) == first
