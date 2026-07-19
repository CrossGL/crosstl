"""Execution bridge for validated runtime graphs and native dispatch requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .runtime_graph import (
    RuntimeExecutionGraph,
    RuntimeGraphDiagnostic,
    RuntimeGraphNode,
    parse_runtime_execution_graph,
)
from .runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeExecutionError,
)

RUNTIME_GRAPH_EXECUTION_RESULT_KIND = "crosstl-runtime-graph-execution-result"

_DIAGNOSTIC_PREFIX = "project.runtime-graph.execution"
_EXECUTABLE_TARGETS = frozenset(("directx", "opengl"))


@dataclass(frozen=True)
class RuntimeGraphExecutionResult:
    """Successful result of one dependency-ordered native graph execution."""

    graph_id: str
    target: str
    outputs: Mapping[str, Any]
    executed_nodes: tuple[str, ...]
    barrier_nodes: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": RUNTIME_GRAPH_EXECUTION_RESULT_KIND,
            "graphId": self.graph_id,
            "success": True,
            "target": self.target,
            "executedNodes": list(self.executed_nodes),
            "barrierNodes": list(self.barrier_nodes),
            "outputs": _json_value(self.outputs),
            "diagnostics": [],
        }


class RuntimeGraphExecutionError(RuntimeExecutionError):
    """A graph cannot be executed without violating its portable contract."""

    failure_phase = "runtime-graph-preflight"
    diagnostic_code = f"{_DIAGNOSTIC_PREFIX}.invalid"

    def __init__(
        self,
        message: str,
        diagnostics: Sequence[RuntimeGraphDiagnostic],
    ) -> None:
        ordered = tuple(
            sorted(
                diagnostics,
                key=lambda item: (
                    item.path,
                    item.code,
                    item.node_id or "",
                    item.resource_id or "",
                    item.message,
                ),
            )
        )
        self.diagnostics = ordered
        self.graph_id = next(
            (item.graph_id for item in ordered if item.graph_id is not None), None
        )
        super().__init__(
            message,
            details={
                "diagnosticCount": len(ordered),
                "diagnostics": [item.to_json() for item in ordered],
            },
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": RUNTIME_GRAPH_EXECUTION_RESULT_KIND,
            "success": False,
            "graphId": self.graph_id,
            "failurePhase": self.failure_phase,
            "diagnosticCount": len(self.diagnostics),
            "diagnostics": [item.to_json() for item in self.diagnostics],
        }


def execute_runtime_graph(
    graph: RuntimeExecutionGraph | Mapping[str, Any],
    *,
    runtime: Any,
    requests: Mapping[str, NativeRuntimeDispatchRequest],
    adapter: Any = None,
    state: Any = None,
) -> RuntimeGraphExecutionResult:
    """Validate and execute native dispatch nodes in stable dependency order."""

    parsed = (
        graph
        if isinstance(graph, RuntimeExecutionGraph)
        else parse_runtime_execution_graph(graph)
    )
    validation = parsed.validate()
    if not validation.valid:
        raise RuntimeGraphExecutionError(
            "Runtime graph validation failed before native execution.",
            validation.diagnostics,
        )
    if not isinstance(requests, Mapping):
        raise TypeError("requests must map graph node IDs to native dispatch requests")

    ordered_nodes = _dependency_order(parsed)
    diagnostics = _execution_diagnostics(parsed, ordered_nodes, requests, runtime)
    if diagnostics:
        raise RuntimeGraphExecutionError(
            "Runtime graph native execution preflight failed.", diagnostics
        )

    dispatch_nodes = tuple(node for node in ordered_nodes if node.kind == "dispatch")
    sequence = tuple(requests[node.node_id] for node in dispatch_nodes)
    target = sequence[0].target.strip().lower()
    outputs = runtime.dispatch_sequence(adapter, state, sequence)
    if not isinstance(outputs, Mapping):
        raise RuntimeExecutionError(
            "Native runtime graph execution returned an invalid output payload.",
            failure_phase="runtime-graph-collect",
            diagnostic_code=f"{_DIAGNOSTIC_PREFIX}.output-invalid",
            details={"graphId": parsed.graph_id, "target": target},
        )
    return RuntimeGraphExecutionResult(
        graph_id=parsed.graph_id,
        target=target,
        outputs=dict(outputs),
        executed_nodes=tuple(node.node_id for node in dispatch_nodes),
        barrier_nodes=tuple(
            node.node_id for node in ordered_nodes if node.kind == "barrier"
        ),
    )


def _dependency_order(graph: RuntimeExecutionGraph) -> tuple[RuntimeGraphNode, ...]:
    by_id = {node.node_id: node for node in graph.nodes}
    remaining = set(by_id)
    completed: set[str] = set()
    ordered: list[RuntimeGraphNode] = []
    while remaining:
        ready = [
            node
            for node in graph.nodes
            if node.node_id in remaining and set(node.depends_on) <= completed
        ]
        if not ready:
            raise RuntimeGraphExecutionError(
                "Runtime graph dependencies cannot be ordered.",
                (
                    _diagnostic(
                        graph,
                        "dependency-order-invalid",
                        "Runtime graph dependencies cannot be ordered for execution.",
                        path="$.nodes",
                    ),
                ),
            )
        for node in ready:
            ordered.append(node)
            completed.add(node.node_id)
            remaining.remove(node.node_id)
    return tuple(ordered)


def _execution_diagnostics(graph, ordered_nodes, requests, runtime):
    diagnostics: list[RuntimeGraphDiagnostic] = []
    dispatch_nodes = tuple(node for node in ordered_nodes if node.kind == "dispatch")
    dispatch_ids = {node.node_id for node in dispatch_nodes}
    if not dispatch_nodes:
        diagnostics.append(
            _diagnostic(
                graph,
                "dispatch-missing",
                "Native graph execution requires at least one dispatch node.",
                path="$.nodes",
                missing_capabilities=("runtime.graph.execute.dispatch",),
            )
        )
    for index, node in enumerate(graph.nodes):
        path = f"$.nodes[{index}]"
        if node.kind in {"copy", "fill"}:
            diagnostics.append(
                _diagnostic(
                    graph,
                    "node-unsupported",
                    f"Native graph execution does not yet implement {node.kind} nodes.",
                    path=path,
                    node=node,
                    missing_capabilities=(f"runtime.graph.execute.{node.kind}",),
                )
            )
        if node.repeat is not None or node.condition is not None:
            diagnostics.append(
                _diagnostic(
                    graph,
                    "control-unsupported",
                    "Native graph execution does not yet implement controlled nodes.",
                    path=path,
                    node=node,
                    missing_capabilities=("runtime.graph.execute.control",),
                )
            )

    for node in dispatch_nodes:
        request = requests.get(node.node_id)
        if not isinstance(request, NativeRuntimeDispatchRequest):
            diagnostics.append(
                _diagnostic(
                    graph,
                    "request-missing",
                    f"Dispatch node {node.node_id!r} has no native runtime request.",
                    path=f"$.requests.{node.node_id}",
                    node=node,
                )
            )
            continue
        diagnostics.extend(_request_diagnostics(graph, node, request))
        if not isinstance(request.target, str) or not request.target.strip():
            diagnostics.append(
                _diagnostic(
                    graph,
                    "target-missing",
                    "Native runtime request target must be explicit.",
                    path=f"$.requests.{node.node_id}.target",
                    node=node,
                )
            )

    for request_id in sorted(str(key) for key in requests if key not in dispatch_ids):
        diagnostics.append(
            _diagnostic(
                graph,
                "request-unreferenced",
                f"Native runtime request {request_id!r} is not referenced by the graph.",
                path=f"$.requests.{request_id}",
            )
        )

    valid_requests = [
        requests[node.node_id]
        for node in dispatch_nodes
        if isinstance(requests.get(node.node_id), NativeRuntimeDispatchRequest)
    ]
    targets = {
        request.target.strip().lower()
        for request in valid_requests
        if isinstance(request.target, str) and request.target.strip()
    }
    if len(targets) > 1:
        diagnostics.append(
            _diagnostic(
                graph,
                "target-mismatch",
                "All dispatch nodes in one native execution must use the same target.",
                path="$.nodes",
                details={"targets": sorted(targets)},
            )
        )
    elif targets and next(iter(targets)) not in _EXECUTABLE_TARGETS:
        target = next(iter(targets))
        diagnostics.append(
            _diagnostic(
                graph,
                "target-unsupported",
                f"Native graph execution does not support target {target!r}.",
                path="$.nodes",
                missing_capabilities=(f"runtime.graph.execute.{target}",),
            )
        )
    if not callable(getattr(runtime, "dispatch_sequence", None)):
        diagnostics.append(
            _diagnostic(
                graph,
                "runtime-capability-missing",
                "Native runtime does not provide ordered dispatch sequence execution.",
                path="$.runtime",
                missing_capabilities=("runtime.dispatch-sequence",),
            )
        )
    return diagnostics


def _request_diagnostics(graph, node, request):
    diagnostics: list[RuntimeGraphDiagnostic] = []
    dispatch = node.dispatch
    if dispatch is None:
        return diagnostics
    node_index = next(
        index for index, candidate in enumerate(graph.nodes) if candidate is node
    )
    path = f"$.nodes[{node_index}].dispatch"
    if not isinstance(request.artifact, Mapping):
        diagnostics.append(
            _diagnostic(
                graph,
                "artifact-metadata-invalid",
                "Native request artifact metadata must be an object.",
                path=f"$.requests.{node.node_id}.artifact",
                node=node,
            )
        )
    else:
        diagnostics.extend(_selector_diagnostics(graph, node, request, path=path))

    request_entry = request.entry_point
    if request_entry is None and request.dispatch is not None:
        request_entry = request.dispatch.entry_point
    if dispatch.entry_point != request_entry:
        diagnostics.append(
            _diagnostic(
                graph,
                "entry-point-mismatch",
                "Graph and native request entry points do not match.",
                path=f"{path}.entryPoint",
                node=node,
                details={
                    "graphEntryPoint": dispatch.entry_point,
                    "requestEntryPoint": request_entry,
                },
            )
        )
    diagnostics.extend(_geometry_diagnostics(graph, node, request, path=path))
    diagnostics.extend(_constant_diagnostics(graph, node, request, path=path))
    diagnostics.extend(_binding_diagnostics(graph, node, request, path=path))
    return diagnostics


def _selector_diagnostics(graph, node, request, *, path):
    dispatch = node.dispatch
    if dispatch is None:
        return []
    selector = dispatch.artifact_selector
    artifact = request.artifact
    expected = {
        "id": selector.artifact_id,
        "source": selector.source,
        "target": selector.target,
        "variant": selector.variant,
        "stage": selector.stage,
    }
    aliases = {"id": ("id", "artifactId")}
    diagnostics = []
    for field_name, selector_value in expected.items():
        if selector_value is None:
            continue
        keys = aliases.get(field_name, (field_name,))
        artifact_value = next(
            (artifact.get(key) for key in keys if key in artifact), None
        )
        if str(artifact_value or "") == selector_value:
            continue
        diagnostics.append(
            _diagnostic(
                graph,
                "artifact-selector-mismatch",
                f"Native request artifact does not match selector field {field_name!r}.",
                path=f"{path}.artifactSelector.{field_name}",
                node=node,
                details={"expected": selector_value, "actual": artifact_value},
            )
        )
    if selector.path is not None:
        request_paths = {
            str(value)
            for value in (
                request.artifact_path,
                artifact.get("path"),
                artifact.get("sourcePath"),
                artifact.get("packagePath"),
            )
            if value is not None
        }
        if selector.path not in request_paths:
            diagnostics.append(
                _diagnostic(
                    graph,
                    "artifact-selector-mismatch",
                    "Native request artifact path does not match the graph selector.",
                    path=f"{path}.artifactSelector.path",
                    node=node,
                    details={
                        "expected": selector.path,
                        "actual": sorted(request_paths),
                    },
                )
            )
    return diagnostics


def _geometry_diagnostics(graph, node, request, *, path):
    dispatch = node.dispatch
    if dispatch is None:
        return []
    request_geometry = request.dispatch
    if request_geometry is None:
        return [
            _diagnostic(
                graph,
                "geometry-missing",
                "Native request has no dispatch geometry.",
                path=f"{path}.geometry",
                node=node,
            )
        ]
    diagnostics = []
    fields = (
        (
            "workgroupSize",
            dispatch.geometry.workgroup_size,
            request_geometry.workgroup_size,
        ),
        (
            "workgroupCount",
            dispatch.geometry.workgroup_count,
            request_geometry.workgroup_count,
        ),
        ("globalSize", dispatch.geometry.global_size, request_geometry.global_size),
        ("gridSize", dispatch.geometry.grid_size, request_geometry.grid_size),
    )
    for field_name, graph_value, request_value in fields:
        if graph_value and tuple(request_value) != tuple(graph_value):
            diagnostics.append(
                _diagnostic(
                    graph,
                    "geometry-mismatch",
                    f"Native request {field_name} does not match the graph.",
                    path=f"{path}.geometry.{field_name}",
                    node=node,
                    details={
                        "expected": list(graph_value),
                        "actual": list(request_value),
                    },
                )
            )
    return diagnostics


def _constant_diagnostics(graph, node, request, *, path):
    dispatch = node.dispatch
    if dispatch is None:
        return []
    diagnostics = []
    if not isinstance(request.constants, Mapping):
        return [
            _diagnostic(
                graph,
                "constants-invalid",
                "Native request constants must be an object.",
                path=f"$.requests.{node.node_id}.constants",
                node=node,
            )
        ]
    graph_names = set(dispatch.constants)
    request_names = set(request.constants)
    for name in sorted(graph_names - request_names):
        diagnostics.append(
            _diagnostic(
                graph,
                "constant-missing",
                f"Native request is missing graph constant {name!r}.",
                path=f"{path}.constants.{name}",
                node=node,
            )
        )
    for name in sorted(request_names - graph_names):
        diagnostics.append(
            _diagnostic(
                graph,
                "constant-unreferenced",
                f"Native request constant {name!r} is not declared by the graph.",
                path=f"{path}.constants.{name}",
                node=node,
            )
        )
    for name in sorted(graph_names & request_names):
        constant = request.constants[name]
        if not hasattr(constant, "value"):
            diagnostics.append(
                _diagnostic(
                    graph,
                    "constant-invalid",
                    f"Native request constant {name!r} has invalid metadata.",
                    path=f"{path}.constants.{name}",
                    node=node,
                )
            )
            continue
        actual = constant.value
        if actual != dispatch.constants[name]:
            diagnostics.append(
                _diagnostic(
                    graph,
                    "constant-value-mismatch",
                    f"Native request constant {name!r} does not match the graph.",
                    path=f"{path}.constants.{name}",
                    node=node,
                    details={"expected": dispatch.constants[name], "actual": actual},
                )
            )
    return diagnostics


def _binding_diagnostics(graph, node, request, *, path):
    dispatch = node.dispatch
    if dispatch is None:
        return []
    diagnostics = []
    if not isinstance(request.buffers, Mapping):
        return [
            _diagnostic(
                graph,
                "bindings-invalid",
                "Native request buffer bindings must be an object.",
                path=f"$.requests.{node.node_id}.buffers",
                node=node,
            )
        ]
    graph_names = set(dispatch.bindings)
    request_names = set(request.buffers)
    for name in sorted(graph_names - request_names):
        diagnostics.append(
            _diagnostic(
                graph,
                "binding-missing",
                f"Native request is missing graph binding {name!r}.",
                path=f"{path}.bindings.{name}",
                node=node,
            )
        )
    for name in sorted(request_names - graph_names):
        diagnostics.append(
            _diagnostic(
                graph,
                "binding-unreferenced",
                f"Native request binding {name!r} is not declared by the graph.",
                path=f"{path}.bindings.{name}",
                node=node,
            )
        )
    resources = {resource.resource_id: resource for resource in graph.resources}
    for name in sorted(graph_names & request_names):
        graph_binding = dispatch.bindings[name]
        native_binding = request.buffers[name]
        resource = resources[graph_binding.resource_id]
        binding_path = f"{path}.bindings.{name}"
        if not isinstance(native_binding, NativeRuntimeBufferBinding):
            diagnostics.append(
                _diagnostic(
                    graph,
                    "binding-invalid",
                    f"Native request binding {name!r} has invalid metadata.",
                    path=binding_path,
                    node=node,
                    resource=resource,
                )
            )
            continue
        diagnostics.extend(
            _resource_binding_diagnostics(
                graph,
                node,
                resource,
                graph_binding.access,
                native_binding,
                path=binding_path,
            )
        )
    return diagnostics


def _resource_binding_diagnostics(
    graph,
    node,
    resource,
    graph_access,
    native_binding,
    *,
    path,
):
    diagnostics = []
    native_access = _normalize_access(native_binding.binding.access)
    if not _access_contains(native_access, _normalize_access(graph_access)):
        diagnostics.append(
            _diagnostic(
                graph,
                "binding-access-mismatch",
                "Native resource access does not satisfy the graph binding access.",
                path=f"{path}.access",
                node=node,
                resource=resource,
                details={
                    "expected": graph_access,
                    "actual": native_binding.binding.access,
                },
            )
        )
    if _normalize_dtype(native_binding.dtype) != _normalize_dtype(resource.dtype):
        diagnostics.append(
            _diagnostic(
                graph,
                "resource-dtype-mismatch",
                "Native resource dtype does not match the graph resource.",
                path=path,
                node=node,
                resource=resource,
                details={"expected": resource.dtype, "actual": native_binding.dtype},
            )
        )
    if tuple(native_binding.shape) != tuple(resource.shape):
        diagnostics.append(
            _diagnostic(
                graph,
                "resource-shape-mismatch",
                "Native resource shape does not match the graph resource.",
                path=path,
                node=node,
                resource=resource,
                details={
                    "expected": list(resource.shape),
                    "actual": list(native_binding.shape),
                },
            )
        )
    if resource.role in {"external-input", "external-input-output"} and (
        "read" in _access_components(graph_access) and native_binding.value is None
    ):
        diagnostics.append(
            _diagnostic(
                graph,
                "external-input-payload-missing",
                "Readable external graph resource has no native input payload.",
                path=path,
                node=node,
                resource=resource,
            )
        )
    if resource.role in {"external-output", "external-input-output"} and (
        "write" in _access_components(graph_access)
        and native_binding.expected_output is None
        and native_binding.source != "expectedOutput"
    ):
        diagnostics.append(
            _diagnostic(
                graph,
                "external-output-readback-missing",
                "Writable external graph resource has no native output readback binding.",
                path=path,
                node=node,
                resource=resource,
            )
        )
    allocation = native_binding.allocation
    if resource.role == "temporary":
        if (
            native_binding.value is not None
            or native_binding.expected_output is not None
            or native_binding.source == "expectedOutput"
        ):
            diagnostics.append(
                _diagnostic(
                    graph,
                    "temporary-host-transfer-invalid",
                    "Temporary graph resources cannot carry host upload or readback payloads.",
                    path=path,
                    node=node,
                    resource=resource,
                )
            )
        if allocation is None:
            diagnostics.append(
                _diagnostic(
                    graph,
                    "temporary-allocation-missing",
                    "Temporary graph resources require an explicit native allocation view.",
                    path=path,
                    node=node,
                    resource=resource,
                )
            )
    if allocation is not None:
        diagnostics.extend(
            _allocation_diagnostics(graph, node, resource, native_binding, path=path)
        )
    return diagnostics


def _allocation_diagnostics(graph, node, resource, binding, *, path):
    allocation = binding.allocation
    if allocation is None:
        return []
    layout = resource.physical_layout
    expected_id = resource.allocation_id or resource.resource_id
    comparisons = (
        ("allocationId", expected_id, allocation.allocation_id),
        ("byteOffset", layout.byte_offset, allocation.byte_offset),
        ("byteLength", layout.byte_length, allocation.byte_length),
        (
            "allocationByteLength",
            layout.allocation_byte_length,
            allocation.allocation_byte_length,
        ),
    )
    diagnostics = []
    for field_name, expected, actual in comparisons:
        if expected is None or expected == actual:
            continue
        diagnostics.append(
            _diagnostic(
                graph,
                "resource-allocation-mismatch",
                f"Native allocation {field_name} does not match the graph layout.",
                path=f"{path}.allocation.{field_name}",
                node=node,
                resource=resource,
                details={"expected": expected, "actual": actual},
            )
        )
    return diagnostics


def _diagnostic(
    graph,
    code,
    message,
    *,
    path,
    node=None,
    resource=None,
    missing_capabilities=(),
    details=None,
):
    return RuntimeGraphDiagnostic(
        code=f"{_DIAGNOSTIC_PREFIX}.{code}",
        message=message,
        path=path,
        graph_id=graph.graph_id,
        node_id=node.node_id if node is not None else None,
        resource_id=resource.resource_id if resource is not None else None,
        missing_capabilities=tuple(missing_capabilities),
        details=details or {},
    )


def _normalize_access(value):
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"readwrite", "write-read"}:
        return "read-write"
    return normalized


def _access_components(value):
    normalized = _normalize_access(value)
    if normalized == "read-write":
        return {"read", "write"}
    if normalized in {"read", "write"}:
        return {normalized}
    return set()


def _access_contains(actual, expected):
    expected_components = _access_components(expected)
    return bool(expected_components) and expected_components <= _access_components(
        actual
    )


def _normalize_dtype(value):
    if not isinstance(value, str):
        return ""
    aliases = {
        "float": "float32",
        "half": "float16",
        "int": "int32",
        "uint": "uint32",
    }
    normalized = value.strip().lower().replace("_t", "")
    return aliases.get(normalized, normalized)


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(key): _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")
