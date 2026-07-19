from dataclasses import replace

import pytest

import crosstl.project as project_api
from crosstl.project.runtime_graph import (
    RUNTIME_EXECUTION_GRAPH_KIND,
    RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION,
    parse_runtime_execution_graph,
)
from crosstl.project.runtime_graph_execution import (
    RUNTIME_GRAPH_EXECUTION_RESULT_KIND,
    RuntimeGraphExecutionError,
    execute_runtime_graph,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeAllocationView,
    RuntimeDispatchGeometry,
    RuntimeExecutionError,
    RuntimeResourceBinding,
    RuntimeValue,
)


def _layout():
    return {
        "storageLayout": "contiguous",
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
        "byteOffset": 0,
        "byteLength": 16,
        "allocationByteLength": 16,
        "alignmentBytes": 4,
    }


def _resource(resource_id, role, *, lifetime=None):
    payload = {
        "id": resource_id,
        "role": role,
        "kind": "buffer",
        "dtype": "uint32",
        "shape": [4],
        "physicalLayout": _layout(),
        "allocationId": resource_id,
    }
    if lifetime is not None:
        payload["lifetime"] = {
            "firstNode": lifetime[0],
            "lastNode": lifetime[1],
        }
    return payload


def _dispatch(node_id, bindings, *, depends_on=()):
    return {
        "id": node_id,
        "kind": "dispatch",
        "dependsOn": list(depends_on),
        "dispatch": {
            "artifactSelector": {"id": f"artifact.{node_id}", "target": "directx"},
            "entryPoint": node_id,
            "bindings": {
                name: {"resource": resource_id, "access": access}
                for name, (resource_id, access) in bindings.items()
            },
            "constants": {},
            "geometry": {"workgroupCount": [1, 1, 1]},
        },
    }


def _graph_payload():
    return {
        "kind": RUNTIME_EXECUTION_GRAPH_KIND,
        "schemaVersion": RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION,
        "id": "two-stage.graph",
        "resources": [
            _resource("input", "external-input"),
            _resource("temporary", "temporary", lifetime=("stage-one", "stage-two")),
            _resource("output", "external-output"),
        ],
        "nodes": [
            _dispatch(
                "stage-one",
                {"source": ("input", "read"), "temporary": ("temporary", "write")},
            ),
            {
                "id": "temporary-visible",
                "kind": "barrier",
                "dependsOn": ["stage-one"],
                "barrier": {
                    "resources": ["temporary"],
                    "beforeAccess": "write",
                    "afterAccess": "read",
                },
            },
            _dispatch(
                "stage-two",
                {"temporary": ("temporary", "read"), "result": ("output", "write")},
                depends_on=("temporary-visible",),
            ),
        ],
    }


def _native_resource(name, binding, access):
    return RuntimeResourceBinding(
        name=name,
        kind="storage-buffer",
        type_name=(
            "RWStructuredBuffer<uint>"
            if access == "write"
            else "StructuredBuffer<uint>"
        ),
        set=0,
        binding=binding,
        access=access,
    )


def _requests(tmp_path):
    temporary = RuntimeAllocationView(
        allocation_id="temporary",
        byte_length=16,
        allocation_byte_length=16,
    )
    producer = NativeRuntimeDispatchRequest(
        target="directx",
        artifact={"id": "artifact.stage-one", "target": "directx"},
        artifact_path=tmp_path / "stage-one.hlsl",
        module_path=tmp_path / "stage-one.dxil",
        loaded_artifact=b"stage-one",
        buffers={
            "source": NativeRuntimeBufferBinding(
                name="source",
                binding=_native_resource("source", 0, "read"),
                value=[1, 2, 3, 4],
                source="input",
                dtype="uint32",
                shape=(4,),
            ),
            "temporary": NativeRuntimeBufferBinding(
                name="temporary",
                binding=_native_resource("temporary", 0, "write"),
                dtype="uint32",
                shape=(4,),
                allocation=temporary,
            ),
        },
        constants={},
        dispatch=RuntimeDispatchGeometry(
            entry_point="stage-one", workgroup_count=(1, 1, 1)
        ),
        entry_point="stage-one",
    )
    consumer = NativeRuntimeDispatchRequest(
        target="directx",
        artifact={"id": "artifact.stage-two", "target": "directx"},
        artifact_path=tmp_path / "stage-two.hlsl",
        module_path=tmp_path / "stage-two.dxil",
        loaded_artifact=b"stage-two",
        buffers={
            "temporary": NativeRuntimeBufferBinding(
                name="temporary",
                binding=_native_resource("temporary", 0, "read"),
                dtype="uint32",
                shape=(4,),
                allocation=temporary,
            ),
            "result": NativeRuntimeBufferBinding(
                name="result",
                binding=_native_resource("result", 0, "write"),
                dtype="uint32",
                shape=(4,),
                expected_output=RuntimeValue(
                    name="output",
                    dtype="uint32",
                    shape=(4,),
                    values=[3, 9, 45, 255],
                ),
            ),
        },
        constants={},
        dispatch=RuntimeDispatchGeometry(
            entry_point="stage-two", workgroup_count=(1, 1, 1)
        ),
        entry_point="stage-two",
    )
    return {"stage-one": producer, "stage-two": consumer}


class _SequenceRuntime:
    def __init__(self, outputs=None):
        self.calls = []
        self.outputs = outputs or {
            "output": {
                "dtype": "uint32",
                "shape": [4],
                "values": [3, 9, 45, 255],
            }
        }

    def dispatch_sequence(self, adapter, state, requests):
        self.calls.append((adapter, state, requests))
        assert requests[0].buffers["temporary"].allocation is (
            requests[1].buffers["temporary"].allocation
        )
        return self.outputs


def _codes(error):
    return {diagnostic.code for diagnostic in error.diagnostics}


def test_project_api_exports_runtime_graph_execution():
    assert project_api.execute_runtime_graph is execute_runtime_graph
    assert project_api.RuntimeGraphExecutionError is RuntimeGraphExecutionError


def test_execute_runtime_graph_uses_dependency_order_and_preserves_temporary(tmp_path):
    graph = parse_runtime_execution_graph(_graph_payload())
    requests = _requests(tmp_path)
    runtime = _SequenceRuntime()
    marker = object()

    result = execute_runtime_graph(
        graph,
        runtime=runtime,
        requests=requests,
        adapter="adapter",
        state=marker,
    )

    assert result.outputs["output"]["values"] == [3, 9, 45, 255]
    assert result.executed_nodes == ("stage-one", "stage-two")
    assert result.barrier_nodes == ("temporary-visible",)
    assert [request.entry_point for request in runtime.calls[0][2]] == [
        "stage-one",
        "stage-two",
    ]
    assert runtime.calls[0][:2] == ("adapter", marker)
    assert result.to_json() == {
        "kind": RUNTIME_GRAPH_EXECUTION_RESULT_KIND,
        "graphId": "two-stage.graph",
        "success": True,
        "target": "directx",
        "executedNodes": ["stage-one", "stage-two"],
        "barrierNodes": ["temporary-visible"],
        "outputs": {
            "output": {
                "dtype": "uint32",
                "shape": [4],
                "values": [3, 9, 45, 255],
            }
        },
        "diagnostics": [],
    }


def test_execute_runtime_graph_rejects_invalid_graph_before_runtime(tmp_path):
    payload = _graph_payload()
    payload["nodes"][2]["dependsOn"] = []
    runtime = _SequenceRuntime()

    with pytest.raises(RuntimeGraphExecutionError) as exc_info:
        execute_runtime_graph(payload, runtime=runtime, requests=_requests(tmp_path))

    assert "project.runtime-graph.resource-access-unordered" in _codes(exc_info.value)
    assert runtime.calls == []
    assert exc_info.value.to_json()["success"] is False


def test_execute_runtime_graph_reports_missing_and_unreferenced_requests(tmp_path):
    requests = _requests(tmp_path)
    requests.pop("stage-two")
    requests["unused"] = requests["stage-one"]

    with pytest.raises(RuntimeGraphExecutionError) as exc_info:
        execute_runtime_graph(
            _graph_payload(), runtime=_SequenceRuntime(), requests=requests
        )

    assert _codes(exc_info.value) >= {
        "project.runtime-graph.execution.request-missing",
        "project.runtime-graph.execution.request-unreferenced",
    }


def test_execute_runtime_graph_reports_artifact_entry_geometry_and_binding_mismatch(
    tmp_path,
):
    requests = _requests(tmp_path)
    producer = requests["stage-one"]
    requests["stage-one"] = replace(
        producer,
        artifact={"id": "wrong", "target": "directx"},
        entry_point="wrong",
        dispatch=RuntimeDispatchGeometry(
            entry_point="wrong", workgroup_count=(2, 1, 1)
        ),
        buffers={"source": producer.buffers["source"]},
    )

    with pytest.raises(RuntimeGraphExecutionError) as exc_info:
        execute_runtime_graph(
            _graph_payload(), runtime=_SequenceRuntime(), requests=requests
        )

    assert _codes(exc_info.value) >= {
        "project.runtime-graph.execution.artifact-selector-mismatch",
        "project.runtime-graph.execution.entry-point-mismatch",
        "project.runtime-graph.execution.geometry-mismatch",
        "project.runtime-graph.execution.binding-missing",
    }


@pytest.mark.parametrize("failure", ["payload", "readback", "allocation"])
def test_execute_runtime_graph_rejects_invalid_temporary_host_contract(
    tmp_path, failure
):
    requests = _requests(tmp_path)
    producer = requests["stage-one"]
    temporary = producer.buffers["temporary"]
    if failure == "payload":
        replacement = replace(temporary, value=[0, 0, 0, 0])
    elif failure == "readback":
        replacement = replace(temporary, source="expectedOutput")
    else:
        replacement = replace(temporary, allocation=None)
    requests["stage-one"] = replace(
        producer, buffers={**producer.buffers, "temporary": replacement}
    )

    with pytest.raises(RuntimeGraphExecutionError) as exc_info:
        execute_runtime_graph(
            _graph_payload(), runtime=_SequenceRuntime(), requests=requests
        )

    expected = (
        "project.runtime-graph.execution.temporary-host-transfer-invalid"
        if failure in {"payload", "readback"}
        else "project.runtime-graph.execution.temporary-allocation-missing"
    )
    assert expected in _codes(exc_info.value)


def test_execute_runtime_graph_rejects_controlled_nodes_with_capability_diagnostic(
    tmp_path,
):
    payload = _graph_payload()
    payload["nodes"][0]["repeat"] = {"count": 1, "maxIterations": 1}

    with pytest.raises(RuntimeGraphExecutionError) as exc_info:
        execute_runtime_graph(
            payload, runtime=_SequenceRuntime(), requests=_requests(tmp_path)
        )

    diagnostic = next(
        item
        for item in exc_info.value.diagnostics
        if item.code == "project.runtime-graph.execution.control-unsupported"
    )
    assert diagnostic.missing_capabilities == ("runtime.graph.execute.control",)


def test_execute_runtime_graph_requires_sequence_runtime_capability(tmp_path):
    with pytest.raises(RuntimeGraphExecutionError) as exc_info:
        execute_runtime_graph(
            _graph_payload(), runtime=object(), requests=_requests(tmp_path)
        )

    assert "project.runtime-graph.execution.runtime-capability-missing" in _codes(
        exc_info.value
    )


def test_execute_runtime_graph_rejects_graph_without_dispatch_nodes():
    payload = {
        "kind": RUNTIME_EXECUTION_GRAPH_KIND,
        "schemaVersion": RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION,
        "id": "fill-only.graph",
        "resources": [_resource("output", "external-output")],
        "nodes": [
            {
                "id": "clear",
                "kind": "fill",
                "fill": {
                    "resource": "output",
                    "range": {"byteOffset": 0, "byteLength": 16},
                    "value": 0,
                },
            }
        ],
    }
    runtime = _SequenceRuntime()

    with pytest.raises(RuntimeGraphExecutionError) as exc_info:
        execute_runtime_graph(payload, runtime=runtime, requests={})

    assert _codes(exc_info.value) >= {
        "project.runtime-graph.execution.dispatch-missing",
        "project.runtime-graph.execution.node-unsupported",
    }
    assert runtime.calls == []


def test_execute_runtime_graph_rejects_missing_target_and_invalid_artifact_metadata(
    tmp_path,
):
    requests = _requests(tmp_path)
    producer = requests["stage-one"]
    requests["stage-one"] = replace(producer, target="", artifact=[])
    runtime = _SequenceRuntime()

    with pytest.raises(RuntimeGraphExecutionError) as exc_info:
        execute_runtime_graph(_graph_payload(), runtime=runtime, requests=requests)

    assert _codes(exc_info.value) >= {
        "project.runtime-graph.execution.target-missing",
        "project.runtime-graph.execution.artifact-metadata-invalid",
    }
    assert runtime.calls == []


def test_execute_runtime_graph_rejects_non_mapping_runtime_outputs(tmp_path):
    runtime = _SequenceRuntime(outputs=[])
    runtime.outputs = []

    with pytest.raises(RuntimeExecutionError) as exc_info:
        execute_runtime_graph(
            _graph_payload(), runtime=runtime, requests=_requests(tmp_path)
        )

    assert exc_info.value.failure_phase == "runtime-graph-collect"
    assert exc_info.value.diagnostic_code.endswith(".output-invalid")
