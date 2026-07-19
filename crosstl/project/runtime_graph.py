"""Backend-neutral runtime execution graph contracts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

RUNTIME_EXECUTION_GRAPH_KIND = "crosstl-runtime-execution-graph"
RUNTIME_EXECUTION_GRAPH_VALIDATION_KIND = "crosstl-runtime-execution-graph-validation"
RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION = 1
MAX_RUNTIME_GRAPH_NODE_EXECUTIONS = 1_000_000

_DIAGNOSTIC_PREFIX = "project.runtime-graph"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
_RESOURCE_ROLES = frozenset(
    ("external-input", "external-output", "external-input-output", "temporary")
)
_RESOURCE_KINDS = frozenset(("buffer",))
_NODE_KINDS = frozenset(("dispatch", "copy", "fill", "barrier"))
_ACCESS_MODES = frozenset(("read", "write", "read-write"))
_CONTROL_DTYPES = frozenset(
    (
        "bool",
        "int",
        "uint",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    )
)
_DTYPE_BYTE_SIZES = {
    "bool": 1,
    "char": 1,
    "uchar": 1,
    "int8": 1,
    "int8_t": 1,
    "uint8": 1,
    "uint8_t": 1,
    "half": 2,
    "float16": 2,
    "bfloat16": 2,
    "int16": 2,
    "int16_t": 2,
    "uint16": 2,
    "uint16_t": 2,
    "float": 4,
    "float32": 4,
    "int": 4,
    "int32": 4,
    "int32_t": 4,
    "uint": 4,
    "uint32": 4,
    "uint32_t": 4,
    "double": 8,
    "float64": 8,
    "int64": 8,
    "int64_t": 8,
    "uint64": 8,
    "uint64_t": 8,
}


@dataclass(frozen=True)
class RuntimeGraphDiagnostic:
    """Structured validation or parsing diagnostic for a runtime graph."""

    code: str
    message: str
    path: str = "$"
    severity: str = "error"
    graph_id: str | None = None
    node_id: str | None = None
    resource_id: str | None = None
    missing_capabilities: tuple[str, ...] = field(default_factory=tuple)
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "checkKind": "runtime-execution-graph",
        }
        if self.graph_id is not None:
            payload["graphId"] = self.graph_id
        if self.node_id is not None:
            payload["nodeId"] = self.node_id
        if self.resource_id is not None:
            payload["resourceId"] = self.resource_id
        if self.missing_capabilities:
            payload["missingCapabilities"] = list(self.missing_capabilities)
        if self.details:
            payload["details"] = _stable_json_value(self.details)
        return payload


class RuntimeGraphError(ValueError):
    """A runtime graph document cannot be parsed safely."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str = "$",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_code = (
            code
            if code.startswith(f"{_DIAGNOSTIC_PREFIX}.")
            else f"{_DIAGNOSTIC_PREFIX}.{code}"
        )
        self.diagnostic = RuntimeGraphDiagnostic(
            code=normalized_code,
            message=message,
            path=path,
            details=details or {},
        )
        super().__init__(f"{path}: {message} ({normalized_code})")

    @property
    def code(self) -> str:
        return self.diagnostic.code

    @property
    def path(self) -> str:
        return self.diagnostic.path

    def to_json(self) -> dict[str, Any]:
        return self.diagnostic.to_json()


@dataclass(frozen=True)
class RuntimeGraphPhysicalLayout:
    """Physical byte layout of a graph resource or allocation view."""

    storage_layout: str | None = None
    element_size_bytes: int | None = None
    element_stride_bytes: int | None = None
    byte_offset: int = 0
    byte_length: int | None = None
    allocation_byte_length: int | None = None
    alignment_bytes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"byteOffset": self.byte_offset}
        for key, value in (
            ("storageLayout", self.storage_layout),
            ("elementSizeBytes", self.element_size_bytes),
            ("elementStrideBytes", self.element_stride_bytes),
            ("byteLength", self.byte_length),
            ("allocationByteLength", self.allocation_byte_length),
            ("alignmentBytes", self.alignment_bytes),
        ):
            if value is not None:
                payload[key] = value
        if self.metadata:
            payload["metadata"] = _stable_json_value(self.metadata)
        return payload


@dataclass(frozen=True)
class RuntimeGraphResourceLifetime:
    """Inclusive node bounds for one temporary resource."""

    first_node: str
    last_node: str

    def to_json(self) -> dict[str, str]:
        return {"firstNode": self.first_node, "lastNode": self.last_node}


@dataclass(frozen=True)
class RuntimeGraphResource:
    """A graph-scoped external or temporary resource."""

    resource_id: str
    role: str
    dtype: str | None
    shape: tuple[int, ...]
    physical_layout: RuntimeGraphPhysicalLayout
    resource_kind: str = "buffer"
    allocation_id: str | None = None
    lifetime: RuntimeGraphResourceLifetime | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.resource_id,
            "role": self.role,
            "kind": self.resource_kind,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "physicalLayout": self.physical_layout.to_json(),
        }
        if self.allocation_id is not None:
            payload["allocationId"] = self.allocation_id
        if self.lifetime is not None:
            payload["lifetime"] = self.lifetime.to_json()
        if self.provenance:
            payload["provenance"] = _stable_json_value(self.provenance)
        return payload


@dataclass(frozen=True)
class RuntimeGraphArtifactSelector:
    """Stable selector for one translated artifact."""

    artifact_id: str | None = None
    source: str | None = None
    target: str | None = None
    variant: str | None = None
    stage: str | None = None
    path: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in (
            ("id", self.artifact_id),
            ("source", self.source),
            ("target", self.target),
            ("variant", self.variant),
            ("stage", self.stage),
            ("path", self.path),
        ):
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class RuntimeGraphBinding:
    """Dispatch binding mapped to a graph resource with explicit access."""

    resource_id: str
    access: str | None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"resource": self.resource_id}
        if self.access is not None:
            payload["access"] = self.access
        return payload


@dataclass(frozen=True)
class RuntimeGraphDispatchGeometry:
    """Concrete backend-neutral launch geometry."""

    workgroup_size: tuple[int, ...] = field(default_factory=tuple)
    workgroup_count: tuple[int, ...] = field(default_factory=tuple)
    global_size: tuple[int, ...] = field(default_factory=tuple)
    grid_size: tuple[int, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in (
            ("workgroupSize", self.workgroup_size),
            ("workgroupCount", self.workgroup_count),
            ("globalSize", self.global_size),
            ("gridSize", self.grid_size),
        ):
            if value:
                payload[key] = list(value)
        return payload


@dataclass(frozen=True)
class RuntimeGraphDispatch:
    """One translated artifact dispatch."""

    artifact_selector: RuntimeGraphArtifactSelector
    entry_point: str | None
    bindings: Mapping[str, RuntimeGraphBinding] = field(default_factory=dict)
    constants: Mapping[str, Any] = field(default_factory=dict)
    geometry: RuntimeGraphDispatchGeometry = field(
        default_factory=RuntimeGraphDispatchGeometry
    )
    target_profile: str | None = None
    validation_status: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifactSelector": self.artifact_selector.to_json(),
            "bindings": {
                key: self.bindings[key].to_json() for key in sorted(self.bindings)
            },
            "constants": _stable_json_value(self.constants),
            "geometry": self.geometry.to_json(),
        }
        if self.entry_point is not None:
            payload["entryPoint"] = self.entry_point
        if self.target_profile is not None:
            payload["targetProfile"] = self.target_profile
        if self.validation_status is not None:
            payload["validationStatus"] = self.validation_status
        if self.provenance:
            payload["provenance"] = _stable_json_value(self.provenance)
        return payload


@dataclass(frozen=True)
class RuntimeGraphByteRange:
    """A bounded byte range relative to a resource view."""

    byte_offset: int
    byte_length: int

    def to_json(self) -> dict[str, int]:
        return {"byteOffset": self.byte_offset, "byteLength": self.byte_length}


@dataclass(frozen=True)
class RuntimeGraphCopy:
    """A byte-preserving copy between compatible resource ranges."""

    source: str
    destination: str
    source_range: RuntimeGraphByteRange
    destination_range: RuntimeGraphByteRange

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "destination": self.destination,
            "sourceRange": self.source_range.to_json(),
            "destinationRange": self.destination_range.to_json(),
        }


@dataclass(frozen=True)
class RuntimeGraphFill:
    """A scalar fill over one bounded resource range."""

    resource: str
    byte_range: RuntimeGraphByteRange
    value: Any

    def to_json(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "range": self.byte_range.to_json(),
            "value": _stable_json_value(self.value),
        }


@dataclass(frozen=True)
class RuntimeGraphBarrier:
    """A backend-neutral visibility transition over named resources."""

    resources: tuple[str, ...]
    before_access: str | None
    after_access: str | None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"resources": list(self.resources)}
        if self.before_access is not None:
            payload["beforeAccess"] = self.before_access
        if self.after_access is not None:
            payload["afterAccess"] = self.after_access
        return payload


@dataclass(frozen=True)
class RuntimeGraphRepeat:
    """A statically bounded fixed or control-input repeat."""

    max_iterations: int | None
    count: int | None = None
    control_input: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.count is not None:
            payload["count"] = self.count
        if self.control_input is not None:
            payload["controlInput"] = self.control_input
        if self.max_iterations is not None:
            payload["maxIterations"] = self.max_iterations
        return payload


@dataclass(frozen=True)
class RuntimeGraphCondition:
    """A bounded condition controlled by a graph input."""

    control_input: str | None
    max_evaluations: int | None
    equals: Any = True

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"equals": _stable_json_value(self.equals)}
        if self.control_input is not None:
            payload["controlInput"] = self.control_input
        if self.max_evaluations is not None:
            payload["maxEvaluations"] = self.max_evaluations
        return payload


@dataclass(frozen=True)
class RuntimeGraphNode:
    """One stable operation in a dependency-ordered execution graph."""

    node_id: str
    kind: str
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    dispatch: RuntimeGraphDispatch | None = None
    copy: RuntimeGraphCopy | None = None
    fill: RuntimeGraphFill | None = None
    barrier: RuntimeGraphBarrier | None = None
    repeat: RuntimeGraphRepeat | None = None
    condition: RuntimeGraphCondition | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.node_id,
            "kind": self.kind,
            "dependsOn": list(self.depends_on),
        }
        operation = getattr(self, self.kind, None)
        if operation is not None:
            payload[self.kind] = operation.to_json()
        if self.repeat is not None:
            payload["repeat"] = self.repeat.to_json()
        if self.condition is not None:
            payload["condition"] = self.condition.to_json()
        if self.provenance:
            payload["provenance"] = _stable_json_value(self.provenance)
        return payload


@dataclass(frozen=True)
class RuntimeExecutionGraph:
    """A versioned, backend-neutral runtime execution graph."""

    graph_id: str
    resources: tuple[RuntimeGraphResource, ...]
    nodes: tuple[RuntimeGraphNode, ...]
    schema_version: int = RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION
    provenance: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": RUNTIME_EXECUTION_GRAPH_KIND,
            "schemaVersion": self.schema_version,
            "id": self.graph_id,
            "resources": [resource.to_json() for resource in self.resources],
            "nodes": [node.to_json() for node in self.nodes],
        }
        if self.provenance:
            payload["provenance"] = _stable_json_value(self.provenance)
        return payload

    def validate(self) -> RuntimeGraphValidationResult:
        return validate_runtime_execution_graph(self)


@dataclass(frozen=True)
class RuntimeGraphValidationResult(Sequence[RuntimeGraphDiagnostic]):
    """Deterministically ordered diagnostics produced before graph execution."""

    graph_id: str
    diagnostics: tuple[RuntimeGraphDiagnostic, ...]

    @property
    def valid(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)

    def __getitem__(self, index: int) -> RuntimeGraphDiagnostic:
        return self.diagnostics[index]

    def __len__(self) -> int:
        return len(self.diagnostics)

    def __iter__(self) -> Iterator[RuntimeGraphDiagnostic]:
        return iter(self.diagnostics)

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": RUNTIME_EXECUTION_GRAPH_VALIDATION_KIND,
            "schemaVersion": RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION,
            "graphId": self.graph_id,
            "valid": self.valid,
            "diagnosticCount": len(self.diagnostics),
            "diagnostics": [item.to_json() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class _ResourceAccess:
    node_id: str
    resource_id: str
    mode: str
    byte_offset: int
    byte_length: int | None
    path: str


def parse_runtime_execution_graph(
    payload: Mapping[str, Any], *, source: str | Path | None = None
) -> RuntimeExecutionGraph:
    """Parse a JSON-shaped runtime execution graph without hiding semantic errors."""

    if isinstance(payload, RuntimeExecutionGraph):
        return payload
    root = _mapping(payload, "$", "runtime execution graph")
    _check_fields(
        root,
        required=("kind", "schemaVersion", "id", "resources", "nodes"),
        optional=("provenance",),
        path="$",
    )
    if root["kind"] != RUNTIME_EXECUTION_GRAPH_KIND:
        raise RuntimeGraphError(
            "kind-invalid",
            f"kind must be {RUNTIME_EXECUTION_GRAPH_KIND!r}.",
            path="$.kind",
        )
    schema_version = root["schemaVersion"]
    if type(schema_version) is not int:
        raise RuntimeGraphError(
            "schema-invalid",
            "schemaVersion must be an integer.",
            path="$.schemaVersion",
        )
    if schema_version != RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION:
        raise RuntimeGraphError(
            "schema-version-unsupported",
            (
                f"Unsupported schemaVersion {schema_version!r}; expected "
                f"{RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION}."
            ),
            path="$.schemaVersion",
        )
    resources = tuple(
        _parse_resource(value, f"$.resources[{index}]")
        for index, value in enumerate(_sequence(root["resources"], "$.resources"))
    )
    nodes = tuple(
        _parse_node(value, f"$.nodes[{index}]")
        for index, value in enumerate(_sequence(root["nodes"], "$.nodes"))
    )
    return RuntimeExecutionGraph(
        graph_id=_string(root["id"], "$.id", allow_empty=True),
        resources=resources,
        nodes=nodes,
        provenance=_json_mapping(root.get("provenance", {}), "$.provenance"),
        source=str(source) if source is not None else None,
    )


def validate_runtime_execution_graph(
    graph: RuntimeExecutionGraph,
) -> RuntimeGraphValidationResult:
    """Validate graph safety, dependencies, layouts, lifetimes, and bounded control."""

    if not isinstance(graph, RuntimeExecutionGraph):
        raise TypeError("graph must be a RuntimeExecutionGraph")
    diagnostics: list[RuntimeGraphDiagnostic] = []

    def emit(
        code: str,
        message: str,
        *,
        path: str = "$",
        node_id: str | None = None,
        resource_id: str | None = None,
        missing_capabilities: Sequence[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        diagnostics.append(
            RuntimeGraphDiagnostic(
                code=(
                    code
                    if code.startswith(f"{_DIAGNOSTIC_PREFIX}.")
                    else f"{_DIAGNOSTIC_PREFIX}.{code}"
                ),
                message=message,
                path=path,
                graph_id=graph.graph_id,
                node_id=node_id,
                resource_id=resource_id,
                missing_capabilities=tuple(missing_capabilities),
                details=details or {},
            )
        )

    if not _valid_identifier(graph.graph_id):
        emit(
            "graph-id-invalid",
            "Graph id must be a non-empty stable identifier.",
            path="$.id",
        )
    if graph.schema_version != RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION:
        emit(
            "schema-version-unsupported",
            "Runtime graph schema version is unsupported.",
            path="$.schemaVersion",
            details={
                "actual": graph.schema_version,
                "expected": RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION,
            },
        )
    if not graph.resources:
        emit("resources-empty", "Runtime graph must declare at least one resource.")
    if not graph.nodes:
        emit("nodes-empty", "Runtime graph must declare at least one node.")

    resource_by_id: dict[str, RuntimeGraphResource] = {}
    resource_index: dict[str, int] = {}
    extents: dict[str, int | None] = {}
    for index, resource in enumerate(graph.resources):
        path = f"$.resources[{index}]"
        if not _valid_identifier(resource.resource_id):
            emit(
                "resource-id-invalid",
                "Resource id must be a non-empty stable identifier.",
                path=f"{path}.id",
                resource_id=resource.resource_id,
            )
        if resource.resource_id in resource_by_id:
            emit(
                "resource-id-duplicate",
                f"Resource id {resource.resource_id!r} is declared more than once.",
                path=f"{path}.id",
                resource_id=resource.resource_id,
                details={"firstIndex": resource_index[resource.resource_id]},
            )
        else:
            resource_by_id[resource.resource_id] = resource
            resource_index[resource.resource_id] = index
        if resource.role not in _RESOURCE_ROLES:
            emit(
                "resource-role-invalid",
                f"Resource role {resource.role!r} is not supported.",
                path=f"{path}.role",
                resource_id=resource.resource_id,
            )
        if resource.resource_kind not in _RESOURCE_KINDS:
            emit(
                "resource-kind-unsupported",
                f"Resource kind {resource.resource_kind!r} is not supported.",
                path=f"{path}.kind",
                resource_id=resource.resource_id,
                missing_capabilities=(
                    f"runtime.graph.resource.{resource.resource_kind}",
                ),
            )
        if not isinstance(resource.dtype, str) or not resource.dtype.strip():
            emit(
                "resource-dtype-missing",
                "Resource dtype must be explicit.",
                path=f"{path}.dtype",
                resource_id=resource.resource_id,
            )
        for dimension_index, dimension in enumerate(resource.shape):
            if type(dimension) is not int or dimension < 0:
                emit(
                    "resource-shape-invalid",
                    "Resource shape dimensions must be non-negative integers.",
                    path=f"{path}.shape[{dimension_index}]",
                    resource_id=resource.resource_id,
                )
        extents[resource.resource_id] = _validate_physical_layout(
            resource, path=path, emit=emit
        )

    _validate_shared_allocation_layouts(graph.resources, extents, emit)

    node_by_id: dict[str, RuntimeGraphNode] = {}
    node_index: dict[str, int] = {}
    for index, node in enumerate(graph.nodes):
        path = f"$.nodes[{index}]"
        if not _valid_identifier(node.node_id):
            emit(
                "node-id-invalid",
                "Node id must be a non-empty stable identifier.",
                path=f"{path}.id",
                node_id=node.node_id,
            )
        if node.node_id in node_by_id:
            emit(
                "node-id-duplicate",
                f"Node id {node.node_id!r} is declared more than once.",
                path=f"{path}.id",
                node_id=node.node_id,
                details={"firstIndex": node_index[node.node_id]},
            )
        else:
            node_by_id[node.node_id] = node
            node_index[node.node_id] = index
        if node.kind not in _NODE_KINDS:
            emit(
                "node-kind-unsupported",
                f"Node kind {node.kind!r} is not supported by the portable graph.",
                path=f"{path}.kind",
                node_id=node.node_id,
                missing_capabilities=(f"runtime.graph.node.{node.kind}",),
            )
        operations = {
            "dispatch": node.dispatch,
            "copy": node.copy,
            "fill": node.fill,
            "barrier": node.barrier,
        }
        if operations.get(node.kind) is None:
            emit(
                "node-operation-missing",
                f"Node kind {node.kind!r} requires a matching operation payload.",
                path=f"{path}.{node.kind}",
                node_id=node.node_id,
            )
        for operation_kind, operation in operations.items():
            if operation is not None and operation_kind != node.kind:
                emit(
                    "node-operation-incompatible",
                    f"{operation_kind} payload does not match node kind {node.kind!r}.",
                    path=f"{path}.{operation_kind}",
                    node_id=node.node_id,
                )
        seen_dependencies: set[str] = set()
        for dependency_index, dependency in enumerate(node.depends_on):
            dependency_path = f"{path}.dependsOn[{dependency_index}]"
            if dependency in seen_dependencies:
                emit(
                    "dependency-duplicate",
                    f"Dependency {dependency!r} is listed more than once.",
                    path=dependency_path,
                    node_id=node.node_id,
                )
            seen_dependencies.add(dependency)
            if dependency not in node_by_id and dependency not in {
                item.node_id for item in graph.nodes
            }:
                emit(
                    "node-reference-missing",
                    f"Dependency node {dependency!r} is not declared.",
                    path=dependency_path,
                    node_id=node.node_id,
                    details={"reference": dependency},
                )

        _validate_node_operation(
            node,
            index=index,
            resource_by_id=resource_by_id,
            extents=extents,
            emit=emit,
        )
        _validate_node_control(
            node,
            index=index,
            resource_by_id=resource_by_id,
            emit=emit,
        )

    dependencies = {
        node_id: {item for item in node.depends_on if item in node_by_id}
        for node_id, node in node_by_id.items()
    }
    ancestors = _dependency_ancestors(dependencies)
    cycle_nodes = sorted(
        node_id
        for node_id, node_ancestors in ancestors.items()
        if node_id in node_ancestors
    )
    if cycle_nodes:
        emit(
            "dependency-cycle",
            "Runtime graph dependencies contain a cycle.",
            path="$.nodes",
            details={"nodes": cycle_nodes},
        )

    accesses = _collect_resource_accesses(graph, resource_by_id, extents)
    _validate_unordered_accesses(
        graph,
        accesses=accesses,
        resources=resource_by_id,
        ancestors=ancestors,
        emit=emit,
    )
    _validate_barriers(
        graph,
        accesses=accesses,
        resources=resource_by_id,
        ancestors=ancestors,
        emit=emit,
    )
    _validate_temporary_lifetimes(
        graph,
        accesses=accesses,
        resources=resource_by_id,
        node_index=node_index,
        ancestors=ancestors,
        emit=emit,
    )

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
    return RuntimeGraphValidationResult(graph_id=graph.graph_id, diagnostics=ordered)


def runtime_execution_request_to_graph(
    request: Any, *, graph_id: str | None = None
) -> RuntimeExecutionGraph:
    """Wrap one existing runtime request as a one-dispatch compatibility graph."""

    # The local import keeps runtime_verification free to consume this module later.
    from crosstl.project.runtime_verification import RuntimeExecutionRequest

    if not isinstance(request, RuntimeExecutionRequest):
        raise TypeError("request must be a RuntimeExecutionRequest")

    fixture = request.fixture
    resources: list[RuntimeGraphResource] = []
    resource_positions: dict[str, int] = {}
    for value, role in (
        *((item, "external-input") for item in fixture.inputs),
        *((item, "external-output") for item in fixture.expected_outputs),
    ):
        resource = _runtime_value_resource(value, role=role)
        existing_index = resource_positions.get(resource.resource_id)
        if existing_index is None:
            resource_positions[resource.resource_id] = len(resources)
            resources.append(resource)
            continue
        existing = resources[existing_index]
        resources[existing_index] = RuntimeGraphResource(
            resource_id=existing.resource_id,
            role="external-input-output",
            dtype=existing.dtype or resource.dtype,
            shape=existing.shape or resource.shape,
            physical_layout=_coalesce_layout(
                existing.physical_layout, resource.physical_layout
            ),
            resource_kind=existing.resource_kind,
            allocation_id=existing.allocation_id or resource.allocation_id,
            provenance=existing.provenance,
        )

    resource_by_id = {resource.resource_id: resource for resource in resources}
    bindings: dict[str, RuntimeGraphBinding] = {}
    for index, binding in enumerate(request.adapter_contract.resource_bindings):
        resource_id = binding.value or binding.name or binding.binding_id or ""
        key = _runtime_binding_key(binding, index=index)
        if key in bindings:
            key = f"{key}@{index}"
        resource = resource_by_id.get(resource_id)
        bindings[key] = RuntimeGraphBinding(
            resource_id=resource_id,
            access=_normalize_access(binding.access)
            or _default_resource_access(resource.role if resource else None),
        )
    if not bindings:
        for resource in resources:
            bindings[resource.resource_id] = RuntimeGraphBinding(
                resource_id=resource.resource_id,
                access=_default_resource_access(resource.role),
            )

    constants: dict[str, Any] = {}
    for index, constant in enumerate(request.adapter_contract.specialization_constants):
        key = constant.name or (
            str(constant.constant_id)
            if constant.constant_id is not None
            else f"constant-{index}"
        )
        constants[key] = (
            constant.value if constant.value is not None else constant.default
        )

    selector = fixture.selector
    artifact = request.artifact
    artifact_selector = RuntimeGraphArtifactSelector(
        artifact_id=selector.artifact_id
        or _optional_mapping_string(artifact, "id", "artifactId"),
        source=selector.source or _optional_mapping_string(artifact, "source"),
        target=selector.target or _optional_mapping_string(artifact, "target"),
        variant=selector.variant or _optional_mapping_string(artifact, "variant"),
        stage=selector.stage or _optional_mapping_string(artifact, "stage"),
        path=selector.path or _optional_mapping_string(artifact, "path"),
    )
    contract = request.adapter_contract
    contract_dispatch = contract.dispatch
    entry_point = fixture.entry_point
    if entry_point is None and contract_dispatch is not None:
        entry_point = contract_dispatch.entry_point
    if entry_point is None and contract.entry_points:
        entry_point = contract.entry_points[0].name
    if entry_point is None:
        entry_point = _optional_mapping_string(artifact, "entryPoint", "entry_point")
    geometry = RuntimeGraphDispatchGeometry(
        workgroup_size=(
            tuple(contract_dispatch.workgroup_size)
            if contract_dispatch is not None
            else ()
        ),
        workgroup_count=(
            tuple(contract_dispatch.workgroup_count)
            if contract_dispatch is not None
            else ()
        ),
        global_size=(
            tuple(contract_dispatch.global_size)
            if contract_dispatch is not None
            else ()
        ),
        grid_size=(
            tuple(contract_dispatch.grid_size) if contract_dispatch is not None else ()
        ),
    )

    resolved_graph_id = graph_id or f"{_stable_identifier_component(fixture.id)}.graph"
    node_id = f"{_stable_identifier_component(fixture.id)}.dispatch"
    node = RuntimeGraphNode(
        node_id=node_id,
        kind="dispatch",
        dispatch=RuntimeGraphDispatch(
            artifact_selector=artifact_selector,
            entry_point=entry_point,
            bindings=bindings,
            constants=constants,
            geometry=geometry,
            target_profile=_optional_mapping_string(
                artifact, "targetProfile", "profile", "shaderModel"
            ),
            validation_status=_optional_mapping_string(
                artifact, "validationStatus", "status"
            ),
            provenance={
                key: artifact[key]
                for key in ("source", "path", "target", "variant")
                if key in artifact and artifact[key] is not None
            },
        ),
    )
    return RuntimeExecutionGraph(
        graph_id=resolved_graph_id,
        resources=tuple(resources),
        nodes=(node,),
        provenance=_stable_json_value(fixture.metadata),
    )


def _parse_resource(value: Any, path: str) -> RuntimeGraphResource:
    record = _mapping(value, path, "resource")
    _check_fields(
        record,
        required=("id", "role", "dtype", "shape", "physicalLayout"),
        optional=("kind", "allocationId", "lifetime", "provenance"),
        path=path,
    )
    lifetime_value = record.get("lifetime")
    lifetime = None
    if lifetime_value is not None:
        lifetime_record = _mapping(lifetime_value, f"{path}.lifetime", "lifetime")
        _check_fields(
            lifetime_record,
            required=("firstNode", "lastNode"),
            optional=(),
            path=f"{path}.lifetime",
        )
        lifetime = RuntimeGraphResourceLifetime(
            first_node=_string(
                lifetime_record["firstNode"],
                f"{path}.lifetime.firstNode",
                allow_empty=True,
            ),
            last_node=_string(
                lifetime_record["lastNode"],
                f"{path}.lifetime.lastNode",
                allow_empty=True,
            ),
        )
    return RuntimeGraphResource(
        resource_id=_string(record["id"], f"{path}.id", allow_empty=True),
        role=_string(record["role"], f"{path}.role", allow_empty=True),
        resource_kind=_string(
            record.get("kind", "buffer"), f"{path}.kind", allow_empty=True
        ),
        dtype=_optional_string(record["dtype"], f"{path}.dtype", allow_empty=True),
        shape=_integer_tuple(record["shape"], f"{path}.shape"),
        physical_layout=_parse_physical_layout(
            record["physicalLayout"], f"{path}.physicalLayout"
        ),
        allocation_id=_optional_string(
            record.get("allocationId"), f"{path}.allocationId", allow_empty=True
        ),
        lifetime=lifetime,
        provenance=_json_mapping(record.get("provenance", {}), f"{path}.provenance"),
    )


def _parse_physical_layout(value: Any, path: str) -> RuntimeGraphPhysicalLayout:
    record = _mapping(value, path, "physical layout")
    _check_fields(
        record,
        required=(),
        optional=(
            "storageLayout",
            "elementSizeBytes",
            "elementStrideBytes",
            "byteOffset",
            "byteLength",
            "allocationByteLength",
            "alignmentBytes",
            "metadata",
        ),
        path=path,
    )
    return RuntimeGraphPhysicalLayout(
        storage_layout=_optional_string(
            record.get("storageLayout"), f"{path}.storageLayout", allow_empty=True
        ),
        element_size_bytes=_optional_integer(
            record.get("elementSizeBytes"), f"{path}.elementSizeBytes"
        ),
        element_stride_bytes=_optional_integer(
            record.get("elementStrideBytes"), f"{path}.elementStrideBytes"
        ),
        byte_offset=_optional_integer(record.get("byteOffset"), f"{path}.byteOffset")
        or 0,
        byte_length=_optional_integer(record.get("byteLength"), f"{path}.byteLength"),
        allocation_byte_length=_optional_integer(
            record.get("allocationByteLength"), f"{path}.allocationByteLength"
        ),
        alignment_bytes=_optional_integer(
            record.get("alignmentBytes"), f"{path}.alignmentBytes"
        ),
        metadata=_json_mapping(record.get("metadata", {}), f"{path}.metadata"),
    )


def _parse_node(value: Any, path: str) -> RuntimeGraphNode:
    record = _mapping(value, path, "node")
    _check_fields(
        record,
        required=("id", "kind"),
        optional=(
            "dependsOn",
            "dispatch",
            "copy",
            "fill",
            "barrier",
            "repeat",
            "condition",
            "provenance",
        ),
        path=path,
    )
    return RuntimeGraphNode(
        node_id=_string(record["id"], f"{path}.id", allow_empty=True),
        kind=_string(record["kind"], f"{path}.kind", allow_empty=True),
        depends_on=_string_tuple(record.get("dependsOn", []), f"{path}.dependsOn"),
        dispatch=(
            _parse_dispatch(record["dispatch"], f"{path}.dispatch")
            if "dispatch" in record
            else None
        ),
        copy=_parse_copy(record["copy"], f"{path}.copy") if "copy" in record else None,
        fill=_parse_fill(record["fill"], f"{path}.fill") if "fill" in record else None,
        barrier=(
            _parse_barrier(record["barrier"], f"{path}.barrier")
            if "barrier" in record
            else None
        ),
        repeat=(
            _parse_repeat(record["repeat"], f"{path}.repeat")
            if "repeat" in record
            else None
        ),
        condition=(
            _parse_condition(record["condition"], f"{path}.condition")
            if "condition" in record
            else None
        ),
        provenance=_json_mapping(record.get("provenance", {}), f"{path}.provenance"),
    )


def _parse_dispatch(value: Any, path: str) -> RuntimeGraphDispatch:
    record = _mapping(value, path, "dispatch")
    _check_fields(
        record,
        required=(),
        optional=(
            "artifactSelector",
            "entryPoint",
            "bindings",
            "constants",
            "geometry",
            "targetProfile",
            "validationStatus",
            "provenance",
        ),
        path=path,
    )
    selector_record = _mapping(
        record.get("artifactSelector", {}),
        f"{path}.artifactSelector",
        "artifact selector",
    )
    _check_fields(
        selector_record,
        required=(),
        optional=("id", "source", "target", "variant", "stage", "path"),
        path=f"{path}.artifactSelector",
    )
    bindings_record = _mapping(
        record.get("bindings", {}), f"{path}.bindings", "bindings"
    )
    if not all(isinstance(name, str) for name in bindings_record):
        raise RuntimeGraphError(
            "json-key-invalid",
            "Dispatch binding names must be strings.",
            path=f"{path}.bindings",
        )
    bindings: dict[str, RuntimeGraphBinding] = {}
    for name in sorted(bindings_record):
        binding_path = f"{path}.bindings.{name}"
        binding_record = _mapping(bindings_record[name], binding_path, "binding")
        _check_fields(
            binding_record,
            required=("resource",),
            optional=("access",),
            path=binding_path,
        )
        bindings[name] = RuntimeGraphBinding(
            resource_id=_string(
                binding_record["resource"], f"{binding_path}.resource", allow_empty=True
            ),
            access=_optional_string(
                binding_record.get("access"), f"{binding_path}.access", allow_empty=True
            ),
        )
    return RuntimeGraphDispatch(
        artifact_selector=RuntimeGraphArtifactSelector(
            artifact_id=_optional_string(
                selector_record.get("id"),
                f"{path}.artifactSelector.id",
                allow_empty=True,
            ),
            source=_optional_string(
                selector_record.get("source"),
                f"{path}.artifactSelector.source",
                allow_empty=True,
            ),
            target=_optional_string(
                selector_record.get("target"),
                f"{path}.artifactSelector.target",
                allow_empty=True,
            ),
            variant=_optional_string(
                selector_record.get("variant"),
                f"{path}.artifactSelector.variant",
                allow_empty=True,
            ),
            stage=_optional_string(
                selector_record.get("stage"),
                f"{path}.artifactSelector.stage",
                allow_empty=True,
            ),
            path=_optional_string(
                selector_record.get("path"),
                f"{path}.artifactSelector.path",
                allow_empty=True,
            ),
        ),
        entry_point=_optional_string(
            record.get("entryPoint"), f"{path}.entryPoint", allow_empty=True
        ),
        bindings=bindings,
        constants=_json_mapping(record.get("constants", {}), f"{path}.constants"),
        geometry=_parse_geometry(record.get("geometry", {}), f"{path}.geometry"),
        target_profile=_optional_string(
            record.get("targetProfile"), f"{path}.targetProfile", allow_empty=True
        ),
        validation_status=_optional_string(
            record.get("validationStatus"), f"{path}.validationStatus", allow_empty=True
        ),
        provenance=_json_mapping(record.get("provenance", {}), f"{path}.provenance"),
    )


def _parse_geometry(value: Any, path: str) -> RuntimeGraphDispatchGeometry:
    record = _mapping(value, path, "dispatch geometry")
    _check_fields(
        record,
        required=(),
        optional=("workgroupSize", "workgroupCount", "globalSize", "gridSize"),
        path=path,
    )
    return RuntimeGraphDispatchGeometry(
        workgroup_size=_integer_tuple(
            record.get("workgroupSize", []), f"{path}.workgroupSize"
        ),
        workgroup_count=_integer_tuple(
            record.get("workgroupCount", []), f"{path}.workgroupCount"
        ),
        global_size=_integer_tuple(record.get("globalSize", []), f"{path}.globalSize"),
        grid_size=_integer_tuple(record.get("gridSize", []), f"{path}.gridSize"),
    )


def _parse_byte_range(value: Any, path: str) -> RuntimeGraphByteRange:
    record = _mapping(value, path, "byte range")
    _check_fields(
        record,
        required=("byteOffset", "byteLength"),
        optional=(),
        path=path,
    )
    return RuntimeGraphByteRange(
        byte_offset=_integer(record["byteOffset"], f"{path}.byteOffset"),
        byte_length=_integer(record["byteLength"], f"{path}.byteLength"),
    )


def _parse_copy(value: Any, path: str) -> RuntimeGraphCopy:
    record = _mapping(value, path, "copy")
    _check_fields(
        record,
        required=("source", "destination", "sourceRange", "destinationRange"),
        optional=(),
        path=path,
    )
    return RuntimeGraphCopy(
        source=_string(record["source"], f"{path}.source", allow_empty=True),
        destination=_string(
            record["destination"], f"{path}.destination", allow_empty=True
        ),
        source_range=_parse_byte_range(record["sourceRange"], f"{path}.sourceRange"),
        destination_range=_parse_byte_range(
            record["destinationRange"], f"{path}.destinationRange"
        ),
    )


def _parse_fill(value: Any, path: str) -> RuntimeGraphFill:
    record = _mapping(value, path, "fill")
    _check_fields(
        record,
        required=("resource", "range", "value"),
        optional=(),
        path=path,
    )
    return RuntimeGraphFill(
        resource=_string(record["resource"], f"{path}.resource", allow_empty=True),
        byte_range=_parse_byte_range(record["range"], f"{path}.range"),
        value=_json_value(record["value"], f"{path}.value"),
    )


def _parse_barrier(value: Any, path: str) -> RuntimeGraphBarrier:
    record = _mapping(value, path, "barrier")
    _check_fields(
        record,
        required=(),
        optional=("resources", "beforeAccess", "afterAccess"),
        path=path,
    )
    return RuntimeGraphBarrier(
        resources=_string_tuple(record.get("resources", []), f"{path}.resources"),
        before_access=_optional_string(
            record.get("beforeAccess"), f"{path}.beforeAccess", allow_empty=True
        ),
        after_access=_optional_string(
            record.get("afterAccess"), f"{path}.afterAccess", allow_empty=True
        ),
    )


def _parse_repeat(value: Any, path: str) -> RuntimeGraphRepeat:
    record = _mapping(value, path, "repeat")
    _check_fields(
        record,
        required=(),
        optional=("count", "controlInput", "maxIterations"),
        path=path,
    )
    return RuntimeGraphRepeat(
        count=_optional_integer(record.get("count"), f"{path}.count"),
        control_input=_optional_string(
            record.get("controlInput"), f"{path}.controlInput", allow_empty=True
        ),
        max_iterations=_optional_integer(
            record.get("maxIterations"), f"{path}.maxIterations"
        ),
    )


def _parse_condition(value: Any, path: str) -> RuntimeGraphCondition:
    record = _mapping(value, path, "condition")
    _check_fields(
        record,
        required=(),
        optional=("controlInput", "equals", "maxEvaluations"),
        path=path,
    )
    return RuntimeGraphCondition(
        control_input=_optional_string(
            record.get("controlInput"), f"{path}.controlInput", allow_empty=True
        ),
        max_evaluations=_optional_integer(
            record.get("maxEvaluations"), f"{path}.maxEvaluations"
        ),
        equals=_json_value(record.get("equals", True), f"{path}.equals"),
    )


def _validate_physical_layout(resource, *, path, emit):
    layout = resource.physical_layout
    layout_path = f"{path}.physicalLayout"
    if not isinstance(layout.storage_layout, str) or not layout.storage_layout.strip():
        emit(
            "resource-layout-invalid",
            "Physical layout must declare storageLayout.",
            path=f"{layout_path}.storageLayout",
            resource_id=resource.resource_id,
        )
    element_size = layout.element_size_bytes
    if element_size is None:
        element_size = _dtype_byte_size(resource.dtype)
    if element_size is None or type(element_size) is not int or element_size <= 0:
        emit(
            "resource-layout-invalid",
            "Physical layout must provide a positive elementSizeBytes for this dtype.",
            path=f"{layout_path}.elementSizeBytes",
            resource_id=resource.resource_id,
        )
        element_size = None
    stride = layout.element_stride_bytes
    if stride is None:
        stride = element_size
    if stride is None or type(stride) is not int or stride <= 0:
        emit(
            "resource-layout-invalid",
            "Physical layout elementStrideBytes must be positive.",
            path=f"{layout_path}.elementStrideBytes",
            resource_id=resource.resource_id,
        )
        stride = None
    elif element_size is not None and stride < element_size:
        emit(
            "resource-layout-invalid",
            "Physical layout stride cannot be smaller than its element size.",
            path=f"{layout_path}.elementStrideBytes",
            resource_id=resource.resource_id,
        )
    for field_name, value in (
        ("byteOffset", layout.byte_offset),
        ("byteLength", layout.byte_length),
        ("allocationByteLength", layout.allocation_byte_length),
        ("alignmentBytes", layout.alignment_bytes),
    ):
        if value is not None and (type(value) is not int or value < 0):
            emit(
                "resource-layout-invalid",
                f"Physical layout {field_name} must be a non-negative integer.",
                path=f"{layout_path}.{field_name}",
                resource_id=resource.resource_id,
            )
    if layout.alignment_bytes is not None:
        if layout.alignment_bytes <= 0:
            emit(
                "resource-layout-invalid",
                "Physical layout alignmentBytes must be positive.",
                path=f"{layout_path}.alignmentBytes",
                resource_id=resource.resource_id,
            )
        elif layout.byte_offset % layout.alignment_bytes != 0:
            emit(
                "resource-layout-invalid",
                "Physical layout byteOffset does not satisfy alignmentBytes.",
                path=f"{layout_path}.byteOffset",
                resource_id=resource.resource_id,
            )
    count = reduce(mul, resource.shape, 1)
    inferred_length = count * stride if stride is not None else None
    extent = layout.byte_length if layout.byte_length is not None else inferred_length
    if extent is None:
        emit(
            "resource-layout-invalid",
            "Physical layout byte length cannot be inferred.",
            path=f"{layout_path}.byteLength",
            resource_id=resource.resource_id,
        )
    elif inferred_length is not None and extent < inferred_length:
        emit(
            "resource-layout-incompatible",
            "Physical layout byteLength is smaller than the declared typed shape.",
            path=f"{layout_path}.byteLength",
            resource_id=resource.resource_id,
            details={"byteLength": extent, "requiredByteLength": inferred_length},
        )
    if (
        extent is not None
        and layout.allocation_byte_length is not None
        and layout.byte_offset + extent > layout.allocation_byte_length
    ):
        emit(
            "resource-layout-incompatible",
            "Resource view exceeds allocationByteLength.",
            path=f"{layout_path}.allocationByteLength",
            resource_id=resource.resource_id,
            details={
                "viewEnd": layout.byte_offset + extent,
                "allocationByteLength": layout.allocation_byte_length,
            },
        )
    return extent


def _validate_shared_allocation_layouts(resources, extents, emit):
    for left_index, left in enumerate(resources):
        if left.allocation_id is None:
            continue
        for right_index in range(left_index + 1, len(resources)):
            right = resources[right_index]
            if right.allocation_id != left.allocation_id:
                continue
            left_span = (
                left.physical_layout.byte_offset,
                extents.get(left.resource_id),
            )
            right_span = (
                right.physical_layout.byte_offset,
                extents.get(right.resource_id),
            )
            if not _spans_overlap(left_span, right_span):
                continue
            if _layout_signature(left) == _layout_signature(right):
                continue
            emit(
                "allocation-layout-incompatible",
                "Overlapping views of one allocation have incompatible physical layouts.",
                path=f"$.resources[{right_index}].physicalLayout",
                resource_id=right.resource_id,
                details={
                    "allocationId": left.allocation_id,
                    "firstResource": left.resource_id,
                    "secondResource": right.resource_id,
                },
            )


def _validate_node_operation(node, *, index, resource_by_id, extents, emit):
    path = f"$.nodes[{index}]"
    if node.kind == "dispatch" and node.dispatch is not None:
        dispatch = node.dispatch
        if not _artifact_selector_present(dispatch.artifact_selector):
            emit(
                "dispatch-artifact-selector-missing",
                "Dispatch must select a translated artifact.",
                path=f"{path}.dispatch.artifactSelector",
                node_id=node.node_id,
            )
        if (
            not isinstance(dispatch.entry_point, str)
            or not dispatch.entry_point.strip()
        ):
            emit(
                "dispatch-entry-point-missing",
                "Dispatch entryPoint must be explicit.",
                path=f"{path}.dispatch.entryPoint",
                node_id=node.node_id,
            )
        for binding_name, binding in dispatch.bindings.items():
            binding_path = f"{path}.dispatch.bindings.{binding_name}"
            if binding.resource_id not in resource_by_id:
                emit(
                    "resource-reference-missing",
                    f"Dispatch binding references undeclared resource {binding.resource_id!r}.",
                    path=f"{binding_path}.resource",
                    node_id=node.node_id,
                    resource_id=binding.resource_id,
                )
            if _normalize_access(binding.access) not in _ACCESS_MODES:
                emit(
                    "resource-access-invalid",
                    "Dispatch binding access must be read, write, or read-write.",
                    path=f"{binding_path}.access",
                    node_id=node.node_id,
                    resource_id=binding.resource_id,
                )
        _validate_geometry(
            dispatch.geometry, path=f"{path}.dispatch.geometry", node=node, emit=emit
        )
    elif node.kind == "copy" and node.copy is not None:
        copy = node.copy
        for role, resource_id, byte_range in (
            ("source", copy.source, copy.source_range),
            ("destination", copy.destination, copy.destination_range),
        ):
            if resource_id not in resource_by_id:
                emit(
                    "resource-reference-missing",
                    f"Copy {role} references undeclared resource {resource_id!r}.",
                    path=f"{path}.copy.{role}",
                    node_id=node.node_id,
                    resource_id=resource_id,
                )
            else:
                _validate_range(
                    byte_range,
                    extent=extents.get(resource_id),
                    path=f"{path}.copy.{role}Range",
                    node_id=node.node_id,
                    resource_id=resource_id,
                    emit=emit,
                )
        if copy.source_range.byte_length != copy.destination_range.byte_length:
            emit(
                "copy-range-incompatible",
                "Copy source and destination ranges must have equal byte lengths.",
                path=f"{path}.copy",
                node_id=node.node_id,
            )
        source = resource_by_id.get(copy.source)
        destination = resource_by_id.get(copy.destination)
        if source is not None and destination is not None:
            if _layout_signature(source) != _layout_signature(destination):
                emit(
                    "copy-layout-incompatible",
                    "Copy source and destination have incompatible typed physical layouts.",
                    path=f"{path}.copy",
                    node_id=node.node_id,
                    details={"source": copy.source, "destination": copy.destination},
                )
            if _copy_ranges_overlap(source, destination, copy):
                emit(
                    "copy-range-overlap-unsafe",
                    "Copy source and destination overlap in the same physical allocation.",
                    path=f"{path}.copy",
                    node_id=node.node_id,
                )
    elif node.kind == "fill" and node.fill is not None:
        fill = node.fill
        if fill.resource not in resource_by_id:
            emit(
                "resource-reference-missing",
                f"Fill references undeclared resource {fill.resource!r}.",
                path=f"{path}.fill.resource",
                node_id=node.node_id,
                resource_id=fill.resource,
            )
        else:
            _validate_range(
                fill.byte_range,
                extent=extents.get(fill.resource),
                path=f"{path}.fill.range",
                node_id=node.node_id,
                resource_id=fill.resource,
                emit=emit,
            )
        if not isinstance(fill.value, (bool, int, float)) or (
            isinstance(fill.value, float) and not math.isfinite(fill.value)
        ):
            emit(
                "fill-value-invalid",
                "Fill value must be a finite scalar.",
                path=f"{path}.fill.value",
                node_id=node.node_id,
            )
    elif node.kind == "barrier" and node.barrier is not None:
        barrier = node.barrier
        if not barrier.resources:
            emit(
                "barrier-empty",
                "Barrier must scope at least one resource.",
                path=f"{path}.barrier.resources",
                node_id=node.node_id,
            )
        seen: set[str] = set()
        for resource_index, resource_id in enumerate(barrier.resources):
            if resource_id in seen:
                emit(
                    "barrier-resource-duplicate",
                    f"Barrier resource {resource_id!r} is listed more than once.",
                    path=f"{path}.barrier.resources[{resource_index}]",
                    node_id=node.node_id,
                    resource_id=resource_id,
                )
            seen.add(resource_id)
            if resource_id not in resource_by_id:
                emit(
                    "resource-reference-missing",
                    f"Barrier references undeclared resource {resource_id!r}.",
                    path=f"{path}.barrier.resources[{resource_index}]",
                    node_id=node.node_id,
                    resource_id=resource_id,
                )
        for field_name, access in (
            ("beforeAccess", barrier.before_access),
            ("afterAccess", barrier.after_access),
        ):
            if _normalize_access(access) not in _ACCESS_MODES:
                emit(
                    "barrier-access-invalid",
                    f"Barrier {field_name} must be read, write, or read-write.",
                    path=f"{path}.barrier.{field_name}",
                    node_id=node.node_id,
                )


def _validate_geometry(geometry, *, path, node, emit):
    fields = (
        ("workgroupSize", geometry.workgroup_size),
        ("workgroupCount", geometry.workgroup_count),
        ("globalSize", geometry.global_size),
        ("gridSize", geometry.grid_size),
    )
    if not any(value for _, value in fields[1:]):
        emit(
            "dispatch-geometry-missing",
            "Dispatch geometry must include workgroupCount, globalSize, or gridSize.",
            path=path,
            node_id=node.node_id,
        )
    for field_name, value in fields:
        if not value:
            continue
        if len(value) != 3 or any(type(item) is not int or item <= 0 for item in value):
            emit(
                "dispatch-geometry-invalid",
                f"{field_name} must contain exactly three positive integers.",
                path=f"{path}.{field_name}",
                node_id=node.node_id,
            )


def _validate_range(byte_range, *, extent, path, node_id, resource_id, emit):
    if type(byte_range.byte_offset) is not int or byte_range.byte_offset < 0:
        emit(
            "resource-range-invalid",
            "Range byteOffset must be a non-negative integer.",
            path=f"{path}.byteOffset",
            node_id=node_id,
            resource_id=resource_id,
        )
    if type(byte_range.byte_length) is not int or byte_range.byte_length <= 0:
        emit(
            "resource-range-invalid",
            "Range byteLength must be a positive integer.",
            path=f"{path}.byteLength",
            node_id=node_id,
            resource_id=resource_id,
        )
    if (
        extent is not None
        and byte_range.byte_offset >= 0
        and byte_range.byte_length > 0
        and byte_range.byte_offset + byte_range.byte_length > extent
    ):
        emit(
            "resource-range-out-of-bounds",
            "Resource range exceeds its physical byte length.",
            path=path,
            node_id=node_id,
            resource_id=resource_id,
            details={
                "rangeEnd": byte_range.byte_offset + byte_range.byte_length,
                "byteLength": extent,
            },
        )


def _validate_node_control(node, *, index, resource_by_id, emit):
    path = f"$.nodes[{index}]"
    if node.kind == "barrier" and (
        node.repeat is not None or node.condition is not None
    ):
        emit(
            "node-control-unsupported",
            "Barrier nodes cannot be conditional or repeated.",
            path=path,
            node_id=node.node_id,
            missing_capabilities=("runtime.graph.controlled-barrier",),
        )
    repeat_bound = 1
    if node.repeat is not None:
        repeat = node.repeat
        if repeat.max_iterations is None or repeat.max_iterations <= 0:
            emit(
                "bounded-control-invalid",
                "Repeat maxIterations must be an explicit positive bound.",
                path=f"{path}.repeat.maxIterations",
                node_id=node.node_id,
            )
        else:
            repeat_bound = repeat.max_iterations
        fixed = repeat.count is not None
        controlled = bool(repeat.control_input)
        if fixed == controlled:
            emit(
                "bounded-control-invalid",
                "Repeat must provide exactly one of count or controlInput.",
                path=f"{path}.repeat",
                node_id=node.node_id,
            )
        if fixed and (repeat.count is None or repeat.count <= 0):
            emit(
                "bounded-control-invalid",
                "Repeat count must be positive.",
                path=f"{path}.repeat.count",
                node_id=node.node_id,
            )
        if (
            fixed
            and repeat.count is not None
            and repeat.max_iterations is not None
            and repeat.count > repeat.max_iterations
        ):
            emit(
                "bounded-control-invalid",
                "Repeat count exceeds maxIterations.",
                path=f"{path}.repeat.count",
                node_id=node.node_id,
            )
        if controlled:
            _validate_control_resource(
                repeat.control_input,
                path=f"{path}.repeat.controlInput",
                node_id=node.node_id,
                resources=resource_by_id,
                emit=emit,
            )
    condition_bound = 1
    if node.condition is not None:
        condition = node.condition
        if not condition.control_input:
            emit(
                "bounded-control-invalid",
                "Condition controlInput must be explicit.",
                path=f"{path}.condition.controlInput",
                node_id=node.node_id,
            )
        else:
            _validate_control_resource(
                condition.control_input,
                path=f"{path}.condition.controlInput",
                node_id=node.node_id,
                resources=resource_by_id,
                emit=emit,
            )
        if condition.max_evaluations is None or condition.max_evaluations <= 0:
            emit(
                "bounded-control-invalid",
                "Condition maxEvaluations must be an explicit positive bound.",
                path=f"{path}.condition.maxEvaluations",
                node_id=node.node_id,
            )
        else:
            condition_bound = condition.max_evaluations
    if repeat_bound * condition_bound > MAX_RUNTIME_GRAPH_NODE_EXECUTIONS:
        emit(
            "bounded-control-limit-exceeded",
            "Node execution bound exceeds the portable graph safety limit.",
            path=path,
            node_id=node.node_id,
            details={
                "bound": repeat_bound * condition_bound,
                "limit": MAX_RUNTIME_GRAPH_NODE_EXECUTIONS,
            },
        )


def _validate_control_resource(resource_id, *, path, node_id, resources, emit):
    resource = resources.get(resource_id)
    if resource is None:
        emit(
            "resource-reference-missing",
            f"Control input references undeclared resource {resource_id!r}.",
            path=path,
            node_id=node_id,
            resource_id=resource_id,
        )
        return
    normalized_dtype = str(resource.dtype or "").strip().lower().replace("_t", "")
    if resource.role not in {"external-input", "external-input-output"}:
        emit(
            "bounded-control-input-invalid",
            "Control input must be an external graph input.",
            path=path,
            node_id=node_id,
            resource_id=resource_id,
        )
    if resource.shape not in ((), (1,)) or normalized_dtype not in _CONTROL_DTYPES:
        emit(
            "bounded-control-input-invalid",
            "Control input must be a scalar boolean or integer resource.",
            path=path,
            node_id=node_id,
            resource_id=resource_id,
        )


def _dependency_ancestors(dependencies):
    ancestors = {node_id: set(items) for node_id, items in dependencies.items()}
    for _ in range(len(ancestors)):
        changed = False
        for node_id, values in ancestors.items():
            expanded = set(values)
            for dependency in tuple(values):
                expanded.update(ancestors.get(dependency, ()))
            if expanded != values:
                ancestors[node_id] = expanded
                changed = True
        if not changed:
            break
    return ancestors


def _collect_resource_accesses(graph, resources, extents):
    accesses: list[_ResourceAccess] = []
    for index, node in enumerate(graph.nodes):
        path = f"$.nodes[{index}]"
        if node.kind == "dispatch" and node.dispatch is not None:
            for binding_name, binding in node.dispatch.bindings.items():
                access = _normalize_access(binding.access)
                if binding.resource_id in resources and access in _ACCESS_MODES:
                    accesses.append(
                        _ResourceAccess(
                            node_id=node.node_id,
                            resource_id=binding.resource_id,
                            mode=access,
                            byte_offset=0,
                            byte_length=extents.get(binding.resource_id),
                            path=f"{path}.dispatch.bindings.{binding_name}",
                        )
                    )
        elif node.kind == "copy" and node.copy is not None:
            for resource_id, mode, byte_range, range_name in (
                (node.copy.source, "read", node.copy.source_range, "sourceRange"),
                (
                    node.copy.destination,
                    "write",
                    node.copy.destination_range,
                    "destinationRange",
                ),
            ):
                if resource_id in resources:
                    accesses.append(
                        _ResourceAccess(
                            node_id=node.node_id,
                            resource_id=resource_id,
                            mode=mode,
                            byte_offset=byte_range.byte_offset,
                            byte_length=byte_range.byte_length,
                            path=f"{path}.copy.{range_name}",
                        )
                    )
        elif node.kind == "fill" and node.fill is not None:
            if node.fill.resource in resources:
                accesses.append(
                    _ResourceAccess(
                        node_id=node.node_id,
                        resource_id=node.fill.resource,
                        mode="write",
                        byte_offset=node.fill.byte_range.byte_offset,
                        byte_length=node.fill.byte_range.byte_length,
                        path=f"{path}.fill.range",
                    )
                )
        for control in (
            node.repeat.control_input if node.repeat is not None else None,
            node.condition.control_input if node.condition is not None else None,
        ):
            if control in resources:
                accesses.append(
                    _ResourceAccess(
                        node_id=node.node_id,
                        resource_id=control,
                        mode="read",
                        byte_offset=0,
                        byte_length=extents.get(control),
                        path=path,
                    )
                )
    return accesses


def _validate_unordered_accesses(graph, *, accesses, resources, ancestors, emit):
    accesses_by_node: dict[str, list[_ResourceAccess]] = {}
    for access in accesses:
        accesses_by_node.setdefault(access.node_id, []).append(access)
    for left_index, left_node in enumerate(graph.nodes):
        if left_node.node_id not in accesses_by_node:
            continue
        for right_node in graph.nodes[left_index + 1 :]:
            if right_node.node_id not in accesses_by_node:
                continue
            if _nodes_ordered(left_node.node_id, right_node.node_id, ancestors):
                continue
            conflict = None
            for left_access in accesses_by_node[left_node.node_id]:
                for right_access in accesses_by_node[right_node.node_id]:
                    if _accesses_conflict(left_access, right_access, resources):
                        conflict = (left_access, right_access)
                        break
                if conflict is not None:
                    break
            if conflict is None:
                continue
            left_access, right_access = conflict
            emit(
                "resource-access-unordered",
                "Conflicting resource accesses are not ordered by explicit dependencies.",
                path=right_access.path,
                node_id=right_node.node_id,
                resource_id=right_access.resource_id,
                details={
                    "firstNode": left_node.node_id,
                    "secondNode": right_node.node_id,
                    "firstResource": left_access.resource_id,
                    "secondResource": right_access.resource_id,
                },
            )


def _validate_barriers(graph, *, accesses, resources, ancestors, emit):
    accesses_by_resource: dict[str, list[_ResourceAccess]] = {}
    for access in accesses:
        accesses_by_resource.setdefault(access.resource_id, []).append(access)
    for index, node in enumerate(graph.nodes):
        if node.kind != "barrier" or node.barrier is None:
            continue
        barrier = node.barrier
        before = _normalize_access(barrier.before_access)
        after = _normalize_access(barrier.after_access)
        path = f"$.nodes[{index}].barrier"
        if before not in _ACCESS_MODES or after not in _ACCESS_MODES:
            continue
        if "write" not in _access_components(before):
            emit(
                "barrier-unsafe",
                "Barrier beforeAccess must include a write requiring visibility.",
                path=f"{path}.beforeAccess",
                node_id=node.node_id,
            )
        for resource_id in barrier.resources:
            if resource_id not in resources:
                continue
            candidates = accesses_by_resource.get(resource_id, [])
            prior = [
                access
                for access in candidates
                if access.node_id in ancestors.get(node.node_id, set())
                and _declared_access_matches(before, access.mode)
                and "write" in _access_components(access.mode)
            ]
            following = [
                access
                for access in candidates
                if node.node_id in ancestors.get(access.node_id, set())
                and _declared_access_matches(after, access.mode)
            ]
            if not prior:
                emit(
                    "barrier-unsafe",
                    "Barrier has no dependency-ordered producer matching beforeAccess.",
                    path=path,
                    node_id=node.node_id,
                    resource_id=resource_id,
                )
            if not following:
                emit(
                    "barrier-unsafe",
                    "Barrier has no dependent consumer matching afterAccess.",
                    path=path,
                    node_id=node.node_id,
                    resource_id=resource_id,
                )


def _validate_temporary_lifetimes(
    graph, *, accesses, resources, node_index, ancestors, emit
):
    accesses_by_resource: dict[str, list[_ResourceAccess]] = {}
    for access in accesses:
        accesses_by_resource.setdefault(access.resource_id, []).append(access)
    for resource_index, resource in enumerate(graph.resources):
        if resource.role != "temporary":
            continue
        path = f"$.resources[{resource_index}]"
        resource_accesses = accesses_by_resource.get(resource.resource_id, [])
        producers = [
            access
            for access in resource_accesses
            if "write" in _access_components(access.mode)
        ]
        consumers = [
            access
            for access in resource_accesses
            if "read" in _access_components(access.mode)
        ]
        ordered_pairs = [
            (producer, consumer)
            for producer in producers
            for consumer in consumers
            if producer.node_id != consumer.node_id
            and producer.node_id in ancestors.get(consumer.node_id, set())
        ]
        if not producers:
            emit(
                "temporary-producer-missing",
                "Temporary resource has no producer.",
                path=path,
                resource_id=resource.resource_id,
            )
        if not consumers or not ordered_pairs:
            emit(
                "temporary-consumer-missing",
                "Temporary resource has no dependency-ordered consumer after a producer.",
                path=path,
                resource_id=resource.resource_id,
            )
        lifetime = resource.lifetime
        if lifetime is None:
            emit(
                "temporary-lifetime-missing",
                "Temporary resource must declare an explicit node lifetime.",
                path=f"{path}.lifetime",
                resource_id=resource.resource_id,
            )
            continue
        first_index = node_index.get(lifetime.first_node)
        last_index = node_index.get(lifetime.last_node)
        if first_index is None or last_index is None:
            emit(
                "temporary-lifetime-invalid",
                "Temporary lifetime references undeclared nodes.",
                path=f"{path}.lifetime",
                resource_id=resource.resource_id,
                details={
                    "firstNode": lifetime.first_node,
                    "lastNode": lifetime.last_node,
                },
            )
            continue
        if first_index > last_index or (
            lifetime.first_node != lifetime.last_node
            and lifetime.first_node not in ancestors.get(lifetime.last_node, set())
        ):
            emit(
                "temporary-lifetime-invalid",
                "Temporary lifetime bounds are not dependency ordered.",
                path=f"{path}.lifetime",
                resource_id=resource.resource_id,
            )
        outside = []
        for access in resource_accesses:
            access_index = node_index.get(access.node_id)
            if access_index is None:
                continue
            after_first = (
                access.node_id == lifetime.first_node
                or lifetime.first_node in ancestors.get(access.node_id, set())
            )
            before_last = (
                access.node_id == lifetime.last_node
                or access.node_id in ancestors.get(lifetime.last_node, set())
            )
            if (
                not first_index <= access_index <= last_index
                or not after_first
                or not before_last
            ):
                outside.append(access.node_id)
        if outside:
            emit(
                "temporary-lifetime-invalid",
                "Temporary resource is accessed outside its declared lifetime.",
                path=f"{path}.lifetime",
                resource_id=resource.resource_id,
                details={"outsideNodes": sorted(set(outside))},
            )


def _accesses_conflict(left, right, resources):
    if "write" not in (_access_components(left.mode) | _access_components(right.mode)):
        return False
    left_resource = resources[left.resource_id]
    right_resource = resources[right.resource_id]
    left_allocation = left_resource.allocation_id or f"resource:{left.resource_id}"
    right_allocation = right_resource.allocation_id or f"resource:{right.resource_id}"
    if left_allocation != right_allocation:
        return False
    left_span = (
        left_resource.physical_layout.byte_offset + left.byte_offset,
        left.byte_length,
    )
    right_span = (
        right_resource.physical_layout.byte_offset + right.byte_offset,
        right.byte_length,
    )
    return _spans_overlap(left_span, right_span)


def _copy_ranges_overlap(source, destination, copy):
    source_allocation = source.allocation_id or f"resource:{source.resource_id}"
    destination_allocation = (
        destination.allocation_id or f"resource:{destination.resource_id}"
    )
    if source_allocation != destination_allocation:
        return False
    return _spans_overlap(
        (
            source.physical_layout.byte_offset + copy.source_range.byte_offset,
            copy.source_range.byte_length,
        ),
        (
            destination.physical_layout.byte_offset
            + copy.destination_range.byte_offset,
            copy.destination_range.byte_length,
        ),
    )


def _spans_overlap(left, right):
    left_offset, left_length = left
    right_offset, right_length = right
    if left_length is None or right_length is None:
        return True
    return (
        left_offset < right_offset + right_length
        and right_offset < left_offset + left_length
    )


def _layout_signature(resource):
    layout = resource.physical_layout
    element_size = layout.element_size_bytes
    if element_size is None:
        element_size = _dtype_byte_size(resource.dtype)
    stride = layout.element_stride_bytes
    if stride is None:
        stride = element_size
    return (
        str(resource.dtype or "").strip().lower(),
        str(layout.storage_layout or "").strip().lower(),
        element_size,
        stride,
    )


def _artifact_selector_present(selector):
    return any(
        isinstance(value, str) and bool(value.strip())
        for value in (
            selector.artifact_id,
            selector.source,
            selector.target,
            selector.variant,
            selector.stage,
            selector.path,
        )
    )


def _nodes_ordered(left, right, ancestors):
    return left in ancestors.get(right, set()) or right in ancestors.get(left, set())


def _declared_access_matches(declared, actual):
    return bool(_access_components(declared) & _access_components(actual))


def _access_components(access):
    normalized = _normalize_access(access)
    if normalized == "read-write":
        return {"read", "write"}
    if normalized in {"read", "write"}:
        return {normalized}
    return set()


def _normalize_access(access):
    if not isinstance(access, str):
        return None
    normalized = access.strip().lower().replace("_", "-")
    if normalized in {"readwrite", "read-write", "write-read"}:
        return "read-write"
    return normalized


def _dtype_byte_size(dtype):
    if not isinstance(dtype, str):
        return None
    return _DTYPE_BYTE_SIZES.get(dtype.strip().lower())


def _valid_identifier(value):
    return isinstance(value, str) and bool(_IDENTIFIER.fullmatch(value.strip()))


def _runtime_value_resource(value, *, role):
    metadata = value.metadata if isinstance(value.metadata, Mapping) else {}
    raw_layout = metadata.get("physicalLayout", metadata.get("scalarLayout", {}))
    raw_layout = raw_layout if isinstance(raw_layout, Mapping) else {}
    element_size = _positive_mapping_integer(raw_layout, "elementSizeBytes")
    if element_size is None:
        element_size = _dtype_byte_size(value.dtype)
    stride = _positive_mapping_integer(raw_layout, "elementStrideBytes") or element_size
    inferred_length = (
        reduce(mul, value.shape, 1) * stride if stride is not None else None
    )
    allocation = value.allocation
    byte_length = (
        allocation.byte_length
        if allocation is not None and allocation.byte_length is not None
        else _positive_mapping_integer(raw_layout, "byteLength", "blockSizeBytes")
        or inferred_length
    )
    known_layout_fields = {
        "storageLayout",
        "elementSizeBytes",
        "elementStrideBytes",
        "byteOffset",
        "byteLength",
        "blockSizeBytes",
        "allocationByteLength",
        "alignmentBytes",
    }
    return RuntimeGraphResource(
        resource_id=value.name,
        role=role,
        dtype=value.dtype,
        shape=tuple(value.shape),
        resource_kind=value.kind,
        allocation_id=allocation.allocation_id if allocation is not None else None,
        physical_layout=RuntimeGraphPhysicalLayout(
            storage_layout=str(raw_layout.get("storageLayout") or "contiguous"),
            element_size_bytes=element_size,
            element_stride_bytes=stride,
            byte_offset=(
                allocation.byte_offset
                if allocation is not None
                else _mapping_integer(raw_layout, "byteOffset") or 0
            ),
            byte_length=byte_length,
            allocation_byte_length=(
                allocation.allocation_byte_length
                if allocation is not None
                else _positive_mapping_integer(raw_layout, "allocationByteLength")
            ),
            alignment_bytes=_positive_mapping_integer(raw_layout, "alignmentBytes"),
            metadata={
                key: raw_layout[key]
                for key in sorted(raw_layout)
                if key not in known_layout_fields
            },
        ),
    )


def _coalesce_layout(first, second):
    return RuntimeGraphPhysicalLayout(
        storage_layout=first.storage_layout or second.storage_layout,
        element_size_bytes=first.element_size_bytes or second.element_size_bytes,
        element_stride_bytes=first.element_stride_bytes or second.element_stride_bytes,
        byte_offset=first.byte_offset or second.byte_offset,
        byte_length=first.byte_length or second.byte_length,
        allocation_byte_length=first.allocation_byte_length
        or second.allocation_byte_length,
        alignment_bytes=first.alignment_bytes or second.alignment_bytes,
        metadata=first.metadata or second.metadata,
    )


def _runtime_binding_key(binding, *, index):
    if binding.binding_id:
        return str(binding.binding_id)
    if binding.name:
        return str(binding.name)
    if binding.binding is not None:
        if binding.set is not None:
            return f"set-{binding.set}-binding-{binding.binding}"
        return f"binding-{binding.binding}"
    if binding.index is not None:
        return f"index-{binding.index}"
    return f"binding-{index}"


def _default_resource_access(role):
    return {
        "external-input": "read",
        "external-output": "write",
        "external-input-output": "read-write",
        "temporary": "read-write",
    }.get(role, "read")


def _optional_mapping_string(value, *keys):
    if not isinstance(value, Mapping):
        return None
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item
    return None


def _stable_identifier_component(value):
    normalized = re.sub(r"[^A-Za-z0-9_.:/-]+", "-", str(value).strip()).strip("-")
    return normalized or "fixture"


def _positive_mapping_integer(value, *keys):
    result = _mapping_integer(value, *keys)
    return result if result is not None and result > 0 else None


def _mapping_integer(value, *keys):
    if not isinstance(value, Mapping):
        return None
    for key in keys:
        item = value.get(key)
        if type(item) is int:
            return item
    return None


def _mapping(value, path, label):
    if not isinstance(value, Mapping):
        raise RuntimeGraphError(
            "type-invalid", f"{label} must be an object.", path=path
        )
    return value


def _sequence(value, path):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RuntimeGraphError("type-invalid", "Value must be a list.", path=path)
    return value


def _check_fields(record, *, required, optional, path):
    allowed = set(required) | set(optional)
    missing = [name for name in required if name not in record]
    if missing:
        raise RuntimeGraphError(
            "field-missing",
            f"Required field {missing[0]!r} is missing.",
            path=path,
            details={"field": missing[0]},
        )
    unknown = sorted(str(name) for name in record if name not in allowed)
    if unknown:
        raise RuntimeGraphError(
            "field-unsupported",
            f"Field {unknown[0]!r} is not part of the portable runtime graph contract.",
            path=f"{path}.{unknown[0]}",
            details={"fields": unknown},
        )


def _string(value, path, *, allow_empty=False):
    if not isinstance(value, str):
        raise RuntimeGraphError("type-invalid", "Value must be a string.", path=path)
    if not allow_empty and not value.strip():
        raise RuntimeGraphError("value-invalid", "Value must be non-empty.", path=path)
    return value


def _optional_string(value, path, *, allow_empty=False):
    if value is None:
        return None
    return _string(value, path, allow_empty=allow_empty)


def _integer(value, path):
    if type(value) is not int:
        raise RuntimeGraphError("type-invalid", "Value must be an integer.", path=path)
    return value


def _optional_integer(value, path):
    if value is None:
        return None
    return _integer(value, path)


def _integer_tuple(value, path):
    return tuple(
        _integer(item, f"{path}[{index}]")
        for index, item in enumerate(_sequence(value, path))
    )


def _string_tuple(value, path):
    return tuple(
        _string(item, f"{path}[{index}]", allow_empty=True)
        for index, item in enumerate(_sequence(value, path))
    )


def _json_mapping(value, path):
    record = _mapping(value, path, "value")
    return _json_value(record, path)


def _json_value(value, path):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeGraphError(
                "json-value-invalid", "JSON numbers must be finite.", path=path
            )
        return value
    if isinstance(value, Mapping):
        for key in value:
            if not isinstance(key, str):
                raise RuntimeGraphError(
                    "json-key-invalid", "JSON object keys must be strings.", path=path
                )
        result = {}
        for key in sorted(value):
            result[key] = _json_value(value[key], f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_value(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    raise RuntimeGraphError(
        "json-value-invalid",
        f"Value of type {type(value).__name__} is not JSON serializable.",
        path=path,
    )


def _stable_json_value(value):
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return {key: _stable_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")
