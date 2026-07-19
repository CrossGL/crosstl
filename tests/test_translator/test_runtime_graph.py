import copy
import json

import pytest

import crosstl.project as project_api
from crosstl.project.runtime_graph import (
    RUNTIME_EXECUTION_GRAPH_KIND,
    RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION,
    RuntimeExecutionGraph,
    RuntimeGraphError,
    parse_runtime_execution_graph,
    runtime_execution_request_to_graph,
    validate_runtime_execution_graph,
)
from crosstl.project.runtime_verification import (
    RuntimeAdapterContract,
    RuntimeAllocationView,
    RuntimeArtifactSelector,
    RuntimeDispatchGeometry,
    RuntimeEntryPoint,
    RuntimeExecutionRequest,
    RuntimeFixture,
    RuntimeResourceBinding,
    RuntimeSpecializationConstant,
    RuntimeValue,
)


def _layout(*, byte_length=16, element_size=4, storage="contiguous", **extra):
    payload = {
        "storageLayout": storage,
        "elementSizeBytes": element_size,
        "elementStrideBytes": element_size,
        "byteOffset": 0,
        "byteLength": byte_length,
    }
    payload.update(extra)
    return payload


def _resource(
    resource_id,
    role,
    *,
    dtype="float32",
    shape=(4,),
    layout=None,
    allocation_id=None,
    lifetime=None,
):
    payload = {
        "id": resource_id,
        "role": role,
        "kind": "buffer",
        "dtype": dtype,
        "shape": list(shape),
        "physicalLayout": layout or _layout(),
    }
    if allocation_id is not None:
        payload["allocationId"] = allocation_id
    if lifetime is not None:
        payload["lifetime"] = {
            "firstNode": lifetime[0],
            "lastNode": lifetime[1],
        }
    return payload


def _dispatch(
    node_id,
    bindings,
    *,
    depends_on=(),
    selector=None,
    entry_point="main",
    constants=None,
    repeat=None,
    condition=None,
):
    payload = {
        "id": node_id,
        "kind": "dispatch",
        "dependsOn": list(depends_on),
        "dispatch": {
            "artifactSelector": selector or {"id": f"artifact.{node_id}"},
            "entryPoint": entry_point,
            "bindings": {
                name: {"resource": resource_id, "access": access}
                for name, (resource_id, access) in bindings.items()
            },
            "constants": constants or {},
            "geometry": {"workgroupCount": [1, 1, 1]},
        },
    }
    if repeat is not None:
        payload["repeat"] = repeat
    if condition is not None:
        payload["condition"] = condition
    return payload


def _graph(resources, nodes, *, graph_id="fixture.graph", provenance=None):
    payload = {
        "kind": RUNTIME_EXECUTION_GRAPH_KIND,
        "schemaVersion": RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION,
        "id": graph_id,
        "resources": resources,
        "nodes": nodes,
    }
    if provenance is not None:
        payload["provenance"] = provenance
    return payload


def _codes(result):
    return {diagnostic.code for diagnostic in result}


def test_two_dispatch_graph_round_trips_and_validates_deterministically():
    payload = _graph(
        [
            _resource("input", "external-input"),
            _resource(
                "partial",
                "temporary",
                lifetime=("initialize", "reduce"),
            ),
            _resource("output", "external-output"),
        ],
        [
            _dispatch(
                "initialize",
                {"source": ("input", "read"), "partial": ("partial", "write")},
                constants={"zValue": 3, "aValue": 1},
            ),
            {
                "id": "partial-visible",
                "kind": "barrier",
                "dependsOn": ["initialize"],
                "barrier": {
                    "resources": ["partial"],
                    "beforeAccess": "write",
                    "afterAccess": "read",
                },
            },
            _dispatch(
                "reduce",
                {"partial": ("partial", "read"), "result": ("output", "write")},
                depends_on=("partial-visible",),
            ),
        ],
        provenance={"revision": "4367c73", "repository": "ml-explore/mlx"},
    )

    graph = parse_runtime_execution_graph(payload)
    result = validate_runtime_execution_graph(graph)
    first_json = graph.to_json()
    reparsed = parse_runtime_execution_graph(first_json)

    assert isinstance(graph, RuntimeExecutionGraph)
    assert result.valid
    assert result.to_json()["diagnostics"] == []
    assert reparsed.to_json() == first_json
    assert list(first_json["nodes"][0]["dispatch"]["constants"]) == [
        "aValue",
        "zValue",
    ]
    assert list(first_json["provenance"]) == ["repository", "revision"]
    assert json.dumps(reparsed.to_json(), separators=(",", ":")) == json.dumps(
        first_json, separators=(",", ":")
    )


def test_fill_then_copy_provides_a_valid_temporary_lifetime():
    payload = _graph(
        [
            _resource("scratch", "temporary", lifetime=("clear", "copy-out")),
            _resource("output", "external-output"),
        ],
        [
            {
                "id": "clear",
                "kind": "fill",
                "fill": {
                    "resource": "scratch",
                    "range": {"byteOffset": 0, "byteLength": 16},
                    "value": 0,
                },
            },
            {
                "id": "copy-out",
                "kind": "copy",
                "dependsOn": ["clear"],
                "copy": {
                    "source": "scratch",
                    "destination": "output",
                    "sourceRange": {"byteOffset": 0, "byteLength": 16},
                    "destinationRange": {"byteOffset": 0, "byteLength": 16},
                },
            },
        ],
    )

    assert parse_runtime_execution_graph(payload).validate().valid


def test_validation_reports_duplicate_resource_and_node_ids():
    payload = _graph(
        [
            _resource("output", "external-output"),
            _resource("output", "external-output"),
        ],
        [
            {
                "id": "clear",
                "kind": "fill",
                "fill": {
                    "resource": "output",
                    "range": {"byteOffset": 0, "byteLength": 16},
                    "value": 0,
                },
            },
            {
                "id": "clear",
                "kind": "fill",
                "fill": {
                    "resource": "output",
                    "range": {"byteOffset": 0, "byteLength": 16},
                    "value": 1,
                },
            },
        ],
    )

    codes = _codes(parse_runtime_execution_graph(payload).validate())

    assert "project.runtime-graph.resource-id-duplicate" in codes
    assert "project.runtime-graph.node-id-duplicate" in codes


def test_dispatch_validation_reports_missing_references_selector_and_entry_point():
    payload = _graph(
        [_resource("input", "external-input")],
        [
            {
                "id": "launch",
                "kind": "dispatch",
                "dependsOn": ["not-declared"],
                "dispatch": {
                    "artifactSelector": {},
                    "bindings": {"value": {"resource": "missing"}},
                    "constants": {},
                    "geometry": {},
                },
            }
        ],
    )

    codes = _codes(parse_runtime_execution_graph(payload).validate())

    assert "project.runtime-graph.node-reference-missing" in codes
    assert "project.runtime-graph.resource-reference-missing" in codes
    assert "project.runtime-graph.resource-access-invalid" in codes
    assert "project.runtime-graph.dispatch-artifact-selector-missing" in codes
    assert "project.runtime-graph.dispatch-entry-point-missing" in codes
    assert "project.runtime-graph.dispatch-geometry-missing" in codes


def test_dependency_cycles_are_reported_before_execution():
    payload = _graph(
        [
            _resource("first-output", "external-output"),
            _resource("second-output", "external-output"),
        ],
        [
            _dispatch(
                "first",
                {"output": ("first-output", "write")},
                depends_on=("second",),
            ),
            _dispatch(
                "second",
                {"output": ("second-output", "write")},
                depends_on=("first",),
            ),
        ],
    )

    result = parse_runtime_execution_graph(payload).validate()

    diagnostic = next(
        item for item in result if item.code == "project.runtime-graph.dependency-cycle"
    )
    assert diagnostic.details["nodes"] == ["first", "second"]


def test_conflicting_alias_accesses_require_an_explicit_dependency():
    resources = [
        _resource("source", "external-input", allocation_id="working-set"),
        _resource("destination", "external-output", allocation_id="working-set"),
    ]
    nodes = [
        _dispatch("read", {"source": ("source", "read")}),
        _dispatch("write", {"destination": ("destination", "write")}),
    ]

    unordered = parse_runtime_execution_graph(_graph(resources, nodes)).validate()
    nodes[1]["dependsOn"] = ["read"]
    ordered = parse_runtime_execution_graph(_graph(resources, nodes)).validate()

    assert "project.runtime-graph.resource-access-unordered" in _codes(unordered)
    assert ordered.valid


def test_independent_non_conflicting_nodes_may_remain_unordered():
    payload = _graph(
        [
            _resource("left", "external-output"),
            _resource("right", "external-output"),
        ],
        [
            _dispatch("left-node", {"left": ("left", "write")}),
            _dispatch("right-node", {"right": ("right", "write")}),
        ],
    )

    assert parse_runtime_execution_graph(payload).validate().valid


def test_copy_validation_catches_range_and_layout_incompatibility():
    payload = _graph(
        [
            _resource("source", "external-input"),
            _resource(
                "destination",
                "external-output",
                dtype="uint32",
                shape=(2,),
                layout=_layout(byte_length=8),
            ),
        ],
        [
            {
                "id": "copy",
                "kind": "copy",
                "copy": {
                    "source": "source",
                    "destination": "destination",
                    "sourceRange": {"byteOffset": 12, "byteLength": 8},
                    "destinationRange": {"byteOffset": 0, "byteLength": 4},
                },
            }
        ],
    )

    codes = _codes(parse_runtime_execution_graph(payload).validate())

    assert "project.runtime-graph.resource-range-out-of-bounds" in codes
    assert "project.runtime-graph.copy-range-incompatible" in codes
    assert "project.runtime-graph.copy-layout-incompatible" in codes


def test_overlapping_shared_views_require_compatible_physical_layouts():
    payload = _graph(
        [
            _resource("float-view", "external-input", allocation_id="shared"),
            _resource(
                "half-view",
                "external-output",
                dtype="float16",
                shape=(8,),
                layout=_layout(element_size=2),
                allocation_id="shared",
            ),
        ],
        [_dispatch("write", {"output": ("half-view", "write")})],
    )

    assert "project.runtime-graph.allocation-layout-incompatible" in _codes(
        parse_runtime_execution_graph(payload).validate()
    )


def test_physical_layout_validation_reports_typed_extent_and_alignment_errors():
    payload = _graph(
        [
            _resource(
                "output",
                "external-output",
                layout={
                    "elementSizeBytes": 4,
                    "elementStrideBytes": 2,
                    "byteOffset": 3,
                    "byteLength": 2,
                    "allocationByteLength": 4,
                    "alignmentBytes": 4,
                },
            )
        ],
        [_dispatch("write", {"output": ("output", "write")})],
    )

    codes = _codes(parse_runtime_execution_graph(payload).validate())

    assert "project.runtime-graph.resource-layout-invalid" in codes
    assert "project.runtime-graph.resource-layout-incompatible" in codes


def test_explicit_zero_element_size_is_not_replaced_by_dtype_inference():
    payload = _graph(
        [
            _resource(
                "output",
                "external-output",
                layout=_layout(element_size=0),
            )
        ],
        [_dispatch("write", {"output": ("output", "write")})],
    )

    assert "project.runtime-graph.resource-layout-invalid" in _codes(
        parse_runtime_execution_graph(payload).validate()
    )


def test_barrier_requires_a_non_empty_safe_visibility_transition():
    empty_payload = _graph(
        [_resource("value", "external-input-output")],
        [
            {
                "id": "barrier",
                "kind": "barrier",
                "barrier": {
                    "resources": [],
                    "beforeAccess": "write",
                    "afterAccess": "read",
                },
            }
        ],
    )
    unsafe_payload = _graph(
        [_resource("value", "external-input-output")],
        [
            _dispatch("writer", {"value": ("value", "write")}),
            {
                "id": "barrier",
                "kind": "barrier",
                "barrier": {
                    "resources": ["value"],
                    "beforeAccess": "read",
                    "afterAccess": "read",
                },
            },
            _dispatch(
                "reader",
                {"value": ("value", "read")},
                depends_on=("barrier",),
            ),
        ],
    )

    assert "project.runtime-graph.barrier-empty" in _codes(
        parse_runtime_execution_graph(empty_payload).validate()
    )
    assert "project.runtime-graph.barrier-unsafe" in _codes(
        parse_runtime_execution_graph(unsafe_payload).validate()
    )


def test_bounded_repeat_and_condition_accept_explicit_scalar_control_inputs():
    payload = _graph(
        [
            _resource(
                "iterations",
                "external-input",
                dtype="uint32",
                shape=(),
                layout=_layout(byte_length=4),
            ),
            _resource(
                "enabled",
                "external-input",
                dtype="bool",
                shape=(),
                layout=_layout(byte_length=1, element_size=1),
            ),
            _resource("output", "external-output"),
        ],
        [
            _dispatch(
                "bounded-launch",
                {"output": ("output", "write")},
                repeat={"controlInput": "iterations", "maxIterations": 8},
                condition={
                    "controlInput": "enabled",
                    "equals": True,
                    "maxEvaluations": 1,
                },
            )
        ],
    )

    graph = parse_runtime_execution_graph(payload)

    assert graph.validate().valid
    assert parse_runtime_execution_graph(graph.to_json()).to_json() == graph.to_json()


def test_bounded_control_rejects_missing_limits_and_non_input_controls():
    payload = _graph(
        [_resource("output", "external-output")],
        [
            _dispatch(
                "invalid-control",
                {"output": ("output", "write")},
                repeat={"count": 2, "controlInput": "missing"},
                condition={"controlInput": "output", "maxEvaluations": 0},
            )
        ],
    )

    codes = _codes(parse_runtime_execution_graph(payload).validate())

    assert "project.runtime-graph.bounded-control-invalid" in codes
    assert "project.runtime-graph.bounded-control-input-invalid" in codes


def test_barrier_nodes_reject_bounded_control_as_unsupported_capability():
    payload = _graph(
        [_resource("value", "external-input-output")],
        [
            {
                "id": "barrier",
                "kind": "barrier",
                "barrier": {
                    "resources": ["value"],
                    "beforeAccess": "write",
                    "afterAccess": "read",
                },
                "repeat": {"count": 1, "maxIterations": 1},
            }
        ],
    )

    result = parse_runtime_execution_graph(payload).validate()
    diagnostic = next(
        item
        for item in result
        if item.code == "project.runtime-graph.node-control-unsupported"
    )

    assert diagnostic.missing_capabilities == ("runtime.graph.controlled-barrier",)


def test_temporary_resources_require_producer_consumer_and_lifetime():
    payload = _graph(
        [
            _resource("input", "external-input"),
            _resource("scratch", "temporary"),
            _resource("output", "external-output"),
        ],
        [
            _dispatch(
                "launch",
                {"input": ("input", "read"), "output": ("output", "write")},
            )
        ],
    )

    codes = _codes(parse_runtime_execution_graph(payload).validate())

    assert "project.runtime-graph.temporary-producer-missing" in codes
    assert "project.runtime-graph.temporary-consumer-missing" in codes
    assert "project.runtime-graph.temporary-lifetime-missing" in codes


def test_temporary_lifetime_must_be_dependency_ordered_and_cover_accesses():
    payload = _graph(
        [
            _resource(
                "scratch",
                "temporary",
                lifetime=("copy-out", "clear"),
            ),
            _resource("output", "external-output"),
        ],
        [
            {
                "id": "clear",
                "kind": "fill",
                "fill": {
                    "resource": "scratch",
                    "range": {"byteOffset": 0, "byteLength": 16},
                    "value": 0,
                },
            },
            {
                "id": "copy-out",
                "kind": "copy",
                "dependsOn": ["clear"],
                "copy": {
                    "source": "scratch",
                    "destination": "output",
                    "sourceRange": {"byteOffset": 0, "byteLength": 16},
                    "destinationRange": {"byteOffset": 0, "byteLength": 16},
                },
            },
        ],
    )

    assert "project.runtime-graph.temporary-lifetime-invalid" in _codes(
        parse_runtime_execution_graph(payload).validate()
    )


def test_parser_rejects_arbitrary_host_callbacks_with_structured_diagnostic():
    payload = _graph(
        [_resource("output", "external-output")],
        [_dispatch("launch", {"output": ("output", "write")})],
    )
    payload["nodes"][0]["callback"] = "run_host_code"

    with pytest.raises(RuntimeGraphError) as exc_info:
        parse_runtime_execution_graph(payload)

    diagnostic = exc_info.value.to_json()
    assert diagnostic["code"] == "project.runtime-graph.field-unsupported"
    assert diagnostic["path"] == "$.nodes[0].callback"
    assert diagnostic["checkKind"] == "runtime-execution-graph"


def test_parser_reports_unsupported_schema_versions_as_structured_errors():
    payload = _graph(
        [_resource("output", "external-output")],
        [_dispatch("launch", {"output": ("output", "write")})],
    )
    payload["schemaVersion"] = 2

    with pytest.raises(RuntimeGraphError) as exc_info:
        parse_runtime_execution_graph(payload)

    assert exc_info.value.code == "project.runtime-graph.schema-version-unsupported"
    assert exc_info.value.path == "$.schemaVersion"


def test_parser_reports_non_string_constant_keys_as_structured_errors():
    payload = _graph(
        [_resource("output", "external-output")],
        [_dispatch("launch", {"output": ("output", "write")})],
    )
    payload["nodes"][0]["dispatch"]["constants"] = {1: "invalid-key"}

    with pytest.raises(RuntimeGraphError) as exc_info:
        parse_runtime_execution_graph(payload)

    assert exc_info.value.code == "project.runtime-graph.json-key-invalid"
    assert exc_info.value.path == "$.nodes[0].dispatch.constants"


def test_parser_reports_non_string_binding_names_as_structured_errors():
    payload = _graph(
        [_resource("output", "external-output")],
        [_dispatch("launch", {"output": ("output", "write")})],
    )
    payload["nodes"][0]["dispatch"]["bindings"] = {
        0: {"resource": "output", "access": "write"}
    }

    with pytest.raises(RuntimeGraphError) as exc_info:
        parse_runtime_execution_graph(payload)

    assert exc_info.value.code == "project.runtime-graph.json-key-invalid"
    assert exc_info.value.path == "$.nodes[0].dispatch.bindings"


def test_validation_rejects_resource_kinds_without_defined_execution_semantics():
    payload = _graph(
        [_resource("output", "external-output")],
        [_dispatch("launch", {"output": ("output", "write")})],
    )
    payload["resources"][0]["kind"] = "texture"

    result = parse_runtime_execution_graph(payload).validate()
    diagnostic = next(
        item
        for item in result
        if item.code == "project.runtime-graph.resource-kind-unsupported"
    )

    assert diagnostic.resource_id == "output"
    assert diagnostic.missing_capabilities == ("runtime.graph.resource.texture",)


def test_project_api_exports_runtime_execution_graph_contract():
    assert project_api.RuntimeExecutionGraph is RuntimeExecutionGraph
    assert project_api.parse_runtime_execution_graph is parse_runtime_execution_graph


def test_single_runtime_request_wraps_as_a_valid_one_dispatch_graph():
    fixture = RuntimeFixture(
        id="vector add",
        selector=RuntimeArtifactSelector(
            artifact_id="directx.vector-add",
            target="directx",
            variant="float32",
        ),
        entry_point="vector_add",
        inputs=(
            RuntimeValue(
                name="input",
                dtype="float32",
                shape=(4,),
                values=[1.0, 2.0, 3.0, 4.0],
                allocation=RuntimeAllocationView(
                    allocation_id="input-allocation",
                    byte_length=16,
                    allocation_byte_length=16,
                ),
            ),
        ),
        expected_outputs=(
            RuntimeValue(
                name="output",
                dtype="float32",
                shape=(4,),
                values=[2.0, 4.0, 6.0, 8.0],
            ),
        ),
        metadata={"suite": "runtime-compatibility"},
    )
    contract = RuntimeAdapterContract(
        entry_points=(RuntimeEntryPoint(name="vector_add", stage="compute"),),
        resource_bindings=(
            RuntimeResourceBinding(
                name="input", binding=0, access="read", value="input"
            ),
            RuntimeResourceBinding(
                name="output", binding=1, access="write", value="output"
            ),
        ),
        specialization_constants=(
            RuntimeSpecializationConstant(name="scale", value=2.0),
        ),
        dispatch=RuntimeDispatchGeometry(
            entry_point="vector_add",
            workgroup_size=(4, 1, 1),
            workgroup_count=(1, 1, 1),
        ),
    )
    request = RuntimeExecutionRequest(
        fixture=fixture,
        artifact={
            "id": "directx.vector-add",
            "target": "directx",
            "targetProfile": "cs_6_0",
            "status": "validated",
        },
        artifact_path=None,
        project_root=None,
        adapter_contract=contract,
    )

    graph = runtime_execution_request_to_graph(request)
    payload = graph.to_json()

    assert graph.validate().valid
    assert payload["id"] == "vector-add.graph"
    assert payload["resources"][0]["allocationId"] == "input-allocation"
    assert payload["nodes"][0]["id"] == "vector-add.dispatch"
    assert payload["nodes"][0]["dispatch"]["targetProfile"] == "cs_6_0"
    assert payload["nodes"][0]["dispatch"]["constants"] == {"scale": 2.0}
    assert payload["nodes"][0]["dispatch"]["bindings"] == {
        "input": {"resource": "input", "access": "read"},
        "output": {"resource": "output", "access": "write"},
    }
    assert parse_runtime_execution_graph(copy.deepcopy(payload)).to_json() == payload


def test_single_runtime_request_preserves_initialized_read_write_resources():
    allocation = RuntimeAllocationView(
        allocation_id="working-set",
        byte_length=16,
        allocation_byte_length=16,
    )
    fixture = RuntimeFixture(
        id="in-place",
        selector=RuntimeArtifactSelector(artifact_id="in-place-artifact"),
        entry_point="main",
        inputs=(
            RuntimeValue(
                name="values",
                dtype="float32",
                shape=(4,),
                allocation=allocation,
            ),
        ),
        expected_outputs=(
            RuntimeValue(
                name="values",
                dtype="float32",
                shape=(4,),
                allocation=allocation,
            ),
        ),
    )
    request = RuntimeExecutionRequest(
        fixture=fixture,
        artifact={"id": "in-place-artifact"},
        artifact_path=None,
        project_root=None,
        adapter_contract=RuntimeAdapterContract(
            resource_bindings=(
                RuntimeResourceBinding(
                    name="values", value="values", access="read_write"
                ),
            ),
            dispatch=RuntimeDispatchGeometry(workgroup_count=(1, 1, 1)),
        ),
    )

    graph = runtime_execution_request_to_graph(request)

    assert graph.validate().valid
    assert graph.to_json()["resources"][0]["role"] == "external-input-output"
    assert graph.to_json()["nodes"][0]["dispatch"]["bindings"]["values"] == {
        "resource": "values",
        "access": "read-write",
    }
