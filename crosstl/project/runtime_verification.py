"""Runtime verification harness for translated project artifacts."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from crosstl.translator.codegen import normalize_backend_name

RUNTIME_VERIFICATION_FIXTURES_KIND = "crosstl-runtime-verification-fixtures"
RUNTIME_VERIFICATION_REPORT_KIND = "crosstl-runtime-verification-report"
RUNTIME_TEST_MANIFEST_KIND = "crosstl-project-runtime-test-manifest"
RUNTIME_TEST_PLAN_KIND = "crosstl-project-runtime-test-plan"
RUNTIME_TEST_REPORT_KIND = "crosstl-project-runtime-test-report"
RUNTIME_VERIFICATION_SCHEMA_VERSION = 1

PASSED = "passed"
SKIPPED = "skipped"
UNAVAILABLE = "unavailable"
TRANSLATION_FAILED = "translation-failed"
RUNTIME_FAILED = "runtime-failed"
COMPARISON_FAILED = "comparison-failed"
EXECUTOR_OK_STATUSES = frozenset(("ok", PASSED, "success"))
EXECUTOR_SKIPPED_STATUSES = frozenset((SKIPPED, "skip"))
EXECUTOR_UNAVAILABLE_STATUSES = frozenset((UNAVAILABLE, "not-available"))
EXECUTOR_FAILED_STATUSES = frozenset(("failed", "error", RUNTIME_FAILED))
RUNTIME_TEST_DEFAULT_ADAPTERS = {
    "cuda": {
        "adapterKind": "cuda-runtime-probe",
        "platformClass": "native-compute",
        "requiredTools": ("nvcc",),
    },
    "directx": {
        "adapterKind": "directx-runtime-probe",
        "platformClass": "native-graphics",
        "requiredTools": ("dxc",),
    },
    "hip": {
        "adapterKind": "hip-runtime-probe",
        "platformClass": "native-compute",
        "requiredTools": ("hipcc",),
    },
    "metal": {
        "adapterKind": "metal-runtime-probe",
        "platformClass": "native-graphics",
        "requiredTools": ("xcrun",),
    },
    "mojo": {
        "adapterKind": "mojo-runtime-probe",
        "platformClass": "native-compute",
        "requiredTools": ("mojo",),
    },
    "opengl": {
        "adapterKind": "opengl-runtime-probe",
        "platformClass": "native-graphics",
        "requiredTools": ("glslangValidator",),
    },
    "rust": {
        "adapterKind": "rust-gpu-runtime-probe",
        "platformClass": "native-compute",
        "requiredTools": ("rustc",),
    },
    "slang": {
        "adapterKind": "slang-runtime-probe",
        "platformClass": "native-compute",
        "requiredTools": ("slangc",),
    },
    "vulkan": {
        "adapterKind": "vulkan-runtime-probe",
        "platformClass": "native-graphics",
        "requiredTools": ("spirv-val", "spirv-as"),
    },
    "webgl": {
        "adapterKind": "webgl-runtime-probe",
        "platformClass": "web-graphics",
        "requiredTools": ("glslangValidator",),
    },
    "wgsl": {
        "adapterKind": "webgpu-runtime-probe",
        "platformClass": "web-compute",
        "requiredTools": ("naga",),
    },
}


class RuntimeVerificationError(ValueError):
    """Base runtime verification error."""


class RuntimeExecutorUnavailable(RuntimeVerificationError):
    """Raised by executors when their backend runtime is not available."""


class RuntimeExecutorSkipped(RuntimeVerificationError):
    """Raised by executors when a fixture should be skipped."""


class RuntimeExecutionError(RuntimeVerificationError):
    """Raised by executors for backend runtime integration failures."""


@dataclass(frozen=True)
class RuntimeTolerance:
    """Absolute and relative tolerance for output comparison."""

    absolute: float = 0.0
    relative: float = 0.0

    def to_json(self) -> dict[str, float]:
        return {"absolute": self.absolute, "relative": self.relative}


@dataclass(frozen=True)
class RuntimeValue:
    """Repository-neutral tensor, buffer, image, or scalar fixture value."""

    name: str
    kind: str = "buffer"
    dtype: str | None = None
    shape: tuple[int, ...] = field(default_factory=tuple)
    values: Any = None
    tolerance: RuntimeTolerance | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.dtype is not None:
            payload["dtype"] = self.dtype
        if self.shape:
            payload["shape"] = list(self.shape)
        if self.values is not None:
            payload["values"] = self.values
        if self.tolerance is not None:
            payload["tolerance"] = self.tolerance.to_json()
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RuntimeArtifactSelector:
    """Select a translated artifact by stable source, target, and variant metadata."""

    source: str | None = None
    target: str | None = None
    variant: str | None = None
    stage: str | None = None
    path: str | None = None
    artifact_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for field_name, value in (
            ("source", self.source),
            ("target", self.target),
            ("variant", self.variant),
            ("stage", self.stage),
            ("path", self.path),
            ("id", self.artifact_id),
        ):
            if value is not None:
                payload[field_name] = value
        return payload


@dataclass(frozen=True)
class RuntimeEntryPoint:
    """Backend-neutral entry-point metadata required by runtime adapters."""

    name: str
    stage: str | None = None
    execution_config: Mapping[str, Any] = field(default_factory=dict)
    workgroup_size: tuple[Any, ...] = field(default_factory=tuple)
    parameters: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name}
        if self.stage is not None:
            payload["stage"] = self.stage
        if self.execution_config:
            payload["executionConfig"] = dict(self.execution_config)
        if self.workgroup_size:
            payload["workgroupSize"] = list(self.workgroup_size)
        if self.parameters:
            payload["parameters"] = [dict(parameter) for parameter in self.parameters]
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RuntimeResourceBinding:
    """Backend-neutral resource binding consumed by a runtime adapter."""

    binding_id: str | None = None
    name: str | None = None
    kind: str | None = None
    type_name: str | None = None
    set: Any = None
    binding: Any = None
    index: int | None = None
    access: str | None = None
    value: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for field_name, value in (
            ("id", self.binding_id),
            ("name", self.name),
            ("kind", self.kind),
            ("type", self.type_name),
            ("set", self.set),
            ("binding", self.binding),
            ("index", self.index),
            ("access", self.access),
            ("value", self.value),
        ):
            if value is not None:
                payload[field_name] = value
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RuntimeSpecializationConstant:
    """Specialization or function constant value for adapter execution."""

    name: str | None = None
    constant_id: Any = None
    kind: str = "specialization-constant"
    dtype: str | None = None
    value: Any = None
    default: Any = None
    required: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind, "required": self.required}
        if self.name is not None:
            payload["name"] = self.name
        if self.constant_id is not None:
            payload["id"] = self.constant_id
        if self.dtype is not None:
            payload["dtype"] = self.dtype
        if self.value is not None:
            payload["value"] = self.value
        if self.default is not None:
            payload["default"] = self.default
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RuntimeDispatchGeometry:
    """Backend-neutral dispatch geometry for kernel execution."""

    entry_point: str | None = None
    workgroup_size: tuple[Any, ...] = field(default_factory=tuple)
    workgroup_count: tuple[int, ...] = field(default_factory=tuple)
    global_size: tuple[int, ...] = field(default_factory=tuple)
    grid_size: tuple[int, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.entry_point is not None:
            payload["entryPoint"] = self.entry_point
        if self.workgroup_size:
            payload["workgroupSize"] = list(self.workgroup_size)
        if self.workgroup_count:
            payload["workgroupCount"] = list(self.workgroup_count)
        if self.global_size:
            payload["globalSize"] = list(self.global_size)
        if self.grid_size:
            payload["gridSize"] = list(self.grid_size)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RuntimeValidationHook:
    """Validation hook expected before, during, or after adapter execution."""

    name: str
    phase: str = "pre-run"
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "phase": self.phase,
            "required": self.required,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RuntimeAdapterContract:
    """Backend-neutral contract runtime adapters use to execute an artifact."""

    contract_id: str | None = None
    entry_points: tuple[RuntimeEntryPoint, ...] = field(default_factory=tuple)
    resource_bindings: tuple[RuntimeResourceBinding, ...] = field(default_factory=tuple)
    specialization_constants: tuple[RuntimeSpecializationConstant, ...] = field(
        default_factory=tuple
    )
    dispatch: RuntimeDispatchGeometry | None = None
    validation_hooks: tuple[RuntimeValidationHook, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.contract_id is not None:
            payload["id"] = self.contract_id
        if self.entry_points:
            payload["entryPoints"] = [
                entry_point.to_json() for entry_point in self.entry_points
            ]
        if self.resource_bindings:
            payload["resourceBindings"] = [
                binding.to_json() for binding in self.resource_bindings
            ]
        if self.specialization_constants:
            payload["specializationConstants"] = [
                constant.to_json() for constant in self.specialization_constants
            ]
        if self.dispatch is not None:
            dispatch = self.dispatch.to_json()
            if dispatch:
                payload["dispatch"] = dispatch
        if self.validation_hooks:
            payload["validationHooks"] = [
                hook.to_json() for hook in self.validation_hooks
            ]
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RuntimeExecutionStep:
    """A planned compile, load, or adapter action for a translated artifact."""

    phase: str
    action: str
    status: str = "planned"
    command: tuple[str, ...] = field(default_factory=tuple)
    artifact_path: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "phase": self.phase,
            "action": self.action,
            "status": self.status,
        }
        if self.command:
            payload["command"] = list(self.command)
        if self.artifact_path is not None:
            payload["artifactPath"] = self.artifact_path
        if self.details:
            payload["details"] = dict(self.details)
        if self.diagnostics:
            payload["diagnostics"] = [
                dict(diagnostic) for diagnostic in self.diagnostics
            ]
        return payload


@dataclass(frozen=True)
class RuntimeBoundResource:
    """Fixture value selected for a resource binding before adapter execution."""

    binding: RuntimeResourceBinding
    value: RuntimeValue | None = None
    source: str | None = None
    status: str = "bound"
    diagnostics: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "binding": self.binding.to_json(),
            "status": self.status,
        }
        if self.value is not None:
            payload["value"] = _runtime_value_reference(self.value)
        if self.source is not None:
            payload["source"] = self.source
        if self.diagnostics:
            payload["diagnostics"] = [
                dict(diagnostic) for diagnostic in self.diagnostics
            ]
        return payload


@dataclass(frozen=True)
class RuntimeBoundConstant:
    """Specialization or function constant value selected for execution."""

    constant: RuntimeSpecializationConstant
    value: Any = None
    source: str | None = None
    status: str = "bound"
    diagnostics: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "constant": self.constant.to_json(),
            "status": self.status,
        }
        if self.value is not None:
            payload["value"] = self.value
        if self.source is not None:
            payload["source"] = self.source
        if self.diagnostics:
            payload["diagnostics"] = [
                dict(diagnostic) for diagnostic in self.diagnostics
            ]
        return payload


@dataclass(frozen=True)
class RuntimeExecutionPlan:
    """Prepared backend-neutral execution data derived from artifact metadata."""

    artifact: Mapping[str, Any]
    artifact_path: Path | None
    compile_steps: tuple[RuntimeExecutionStep, ...] = field(default_factory=tuple)
    load_steps: tuple[RuntimeExecutionStep, ...] = field(default_factory=tuple)
    resource_bindings: tuple[RuntimeBoundResource, ...] = field(default_factory=tuple)
    constant_bindings: tuple[RuntimeBoundConstant, ...] = field(default_factory=tuple)
    dispatch: RuntimeDispatchGeometry | None = None
    diagnostics: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact": _artifact_result_payload(self.artifact),
            "compileSteps": [step.to_json() for step in self.compile_steps],
            "loadSteps": [step.to_json() for step in self.load_steps],
            "resourceBindings": [
                binding.to_json() for binding in self.resource_bindings
            ],
            "constantBindings": [
                binding.to_json() for binding in self.constant_bindings
            ],
            "diagnostics": [dict(diagnostic) for diagnostic in self.diagnostics],
        }
        if self.artifact_path is not None:
            payload["artifactPath"] = str(self.artifact_path)
        if self.dispatch is not None:
            payload["dispatch"] = self.dispatch.to_json()
        return payload


@dataclass(frozen=True)
class RuntimeFixture:
    """A runtime verification fixture bound to one translated artifact."""

    id: str
    selector: RuntimeArtifactSelector
    inputs: tuple[RuntimeValue, ...] = field(default_factory=tuple)
    expected_outputs: tuple[RuntimeValue, ...] = field(default_factory=tuple)
    default_tolerance: RuntimeTolerance = field(default_factory=RuntimeTolerance)
    adapter_contract: RuntimeAdapterContract | None = None
    executor: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "selector": self.selector.to_json(),
            "inputs": [value.to_json() for value in self.inputs],
            "expectedOutputs": [value.to_json() for value in self.expected_outputs],
            "defaultTolerance": self.default_tolerance.to_json(),
        }
        if self.executor is not None:
            payload["executor"] = self.executor
        if self.adapter_contract is not None:
            adapter_contract = self.adapter_contract.to_json()
            if adapter_contract:
                payload["runtimeAdapter"] = adapter_contract
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RuntimePlatformRequirements:
    """Platform and dependency requirements for one runtime test adapter."""

    platform_class: str | None = None
    required_tools: tuple[str, ...] = field(default_factory=tuple)
    required_environment: tuple[str, ...] = field(default_factory=tuple)
    runtimes: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.platform_class is not None:
            payload["platformClass"] = self.platform_class
        if self.required_tools:
            payload["requiredTools"] = list(self.required_tools)
        if self.required_environment:
            payload["requiredEnvironment"] = list(self.required_environment)
        if self.runtimes:
            payload["runtimes"] = list(self.runtimes)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RuntimeTestAdapterSpec:
    """Runtime adapter entry referenced by project runtime tests."""

    adapter_id: str
    target: str | None = None
    executor: str | None = None
    adapter_kind: str | None = None
    platform_requirements: RuntimePlatformRequirements = field(
        default_factory=RuntimePlatformRequirements
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.adapter_id}
        if self.target is not None:
            payload["target"] = self.target
        if self.executor is not None:
            payload["executor"] = self.executor
        if self.adapter_kind is not None:
            payload["adapterKind"] = self.adapter_kind
        platform_requirements = self.platform_requirements.to_json()
        if platform_requirements:
            payload["platformRequirements"] = platform_requirements
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RuntimeTestCase:
    """One project-level runtime test bound to an artifact selector and adapter."""

    fixture: RuntimeFixture
    adapter: str | None = None
    platform_requirements: RuntimePlatformRequirements = field(
        default_factory=RuntimePlatformRequirements
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = self.fixture.to_json()
        if self.adapter is not None:
            payload["adapter"] = self.adapter
        platform_requirements = self.platform_requirements.to_json()
        if platform_requirements:
            payload["platformRequirements"] = platform_requirements
        if self.metadata:
            payload["metadata"] = {
                **dict(payload.get("metadata", {})),
                **dict(self.metadata),
            }
        return payload


@dataclass(frozen=True)
class RuntimeTestManifest:
    """Parsed project-level runtime test manifest."""

    adapters: tuple[RuntimeTestAdapterSpec, ...] = field(default_factory=tuple)
    test_cases: tuple[RuntimeTestCase, ...] = field(default_factory=tuple)
    artifact_manifest: str | None = None
    project_root: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": RUNTIME_TEST_MANIFEST_KIND,
            "adapters": [adapter.to_json() for adapter in self.adapters],
            "tests": [test_case.to_json() for test_case in self.test_cases],
        }
        if self.artifact_manifest is not None:
            payload["artifactManifest"] = self.artifact_manifest
        if self.project_root is not None:
            payload["projectRoot"] = self.project_root
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RuntimeExecutionRequest:
    """Input passed to backend-specific runtime executors."""

    fixture: RuntimeFixture
    artifact: Mapping[str, Any]
    artifact_path: Path | None
    project_root: Path | None
    adapter_contract: RuntimeAdapterContract = field(
        default_factory=RuntimeAdapterContract
    )
    execution_plan: RuntimeExecutionPlan | None = None


@dataclass(frozen=True)
class RuntimeExecutorAvailability:
    """Backend executor availability probe result."""

    available: bool
    reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeExecutorResult:
    """Backend executor output before tolerance comparison."""

    status: str = "ok"
    outputs: Any = field(default_factory=dict)
    message: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeExecutionState:
    """Mutable state used by RuntimeExecutionAdapter phase hooks."""

    request: RuntimeExecutionRequest
    plan: RuntimeExecutionPlan
    compiled_artifact: Any = None
    loaded_artifact: Any = None
    resource_values: dict[str, Any] = field(default_factory=dict)
    constant_values: dict[str, Any] = field(default_factory=dict)
    outputs: Any = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[Mapping[str, Any]] = field(default_factory=list)
    adapter_steps: list[RuntimeExecutionStep] = field(default_factory=list)

    def record_step(
        self,
        phase: str,
        action: str,
        *,
        status: str = PASSED,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.adapter_steps.append(
            RuntimeExecutionStep(
                phase=phase,
                action=action,
                status=status,
                artifact_path=(
                    str(self.plan.artifact_path)
                    if self.plan.artifact_path is not None
                    else None
                ),
                details=dict(details or {}),
            )
        )


@runtime_checkable
class RuntimeAdapter(Protocol):
    """Protocol implemented by backend or downstream runtime adapters."""

    name: str

    def is_available(
        self, request: RuntimeExecutionRequest
    ) -> RuntimeExecutorAvailability: ...

    def run(self, request: RuntimeExecutionRequest) -> RuntimeExecutorResult: ...


class RuntimeExecutor:
    """Base class for backend-specific runtime executors."""

    name = "runtime-executor"

    def is_available(
        self, request: RuntimeExecutionRequest
    ) -> RuntimeExecutorAvailability:
        return RuntimeExecutorAvailability(True)

    def run(self, request: RuntimeExecutionRequest) -> RuntimeExecutorResult:
        raise RuntimeExecutorUnavailable(
            "No runtime execution implementation is configured for "
            f"{request.artifact.get('target')}"
        )


class RuntimeDependencyProbeExecutor(RuntimeExecutor):
    """Metadata-only executor hook that reports dependency skip diagnostics."""

    def __init__(self, adapter: RuntimeTestAdapterSpec):
        self.adapter = adapter
        self.name = adapter.adapter_kind or "runtime-dependency-probe"

    @property
    def target(self) -> str | None:
        return self.adapter.target

    def run(self, request: RuntimeExecutionRequest) -> RuntimeExecutorResult:
        requirements = self.adapter.platform_requirements
        missing = _missing_platform_requirements(requirements)
        details = {
            "adapter": self.adapter.adapter_id,
            "target": self.adapter.target or request.artifact.get("target"),
            "platformRequirements": requirements.to_json(),
            "missingTools": missing["tools"],
            "missingEnvironment": missing["environment"],
        }
        if missing["tools"] or missing["environment"]:
            missing_labels = missing["tools"] + missing["environment"]
            return RuntimeExecutorResult(
                status=SKIPPED,
                message=(
                    "Runtime dependencies are unavailable: "
                    + ", ".join(missing_labels)
                ),
                details=details,
            )
        return RuntimeExecutorResult(
            status=SKIPPED,
            message=(
                "Runtime adapter hook is metadata-only; provide a native "
                "executor to run this fixture."
            ),
            details=details,
        )


class RuntimeExecutionAdapter(RuntimeExecutor):
    """Reusable adapter pipeline for executing prepared runtime fixtures."""

    name = "runtime-execution-adapter"

    def run(self, request: RuntimeExecutionRequest) -> RuntimeExecutorResult:
        plan = request.execution_plan or prepare_runtime_execution(request)
        if _runtime_diagnostics_have_errors(plan.diagnostics):
            raise RuntimeExecutionError("Runtime execution plan contains setup errors.")

        state = RuntimeExecutionState(request=request, plan=plan)
        state.compiled_artifact = self.compile_artifact(state)
        state.loaded_artifact = self.load_artifact(state)
        state.resource_values = self.bind_resources(state)
        state.constant_values = self.bind_constants(state)
        dispatched_outputs = self.dispatch_fixture(state)
        if dispatched_outputs is not None:
            state.outputs = dispatched_outputs
        outputs = self.collect_outputs(state)
        return RuntimeExecutorResult(
            outputs=outputs,
            details={
                "runtimeExecution": plan.to_json(),
                "adapterSteps": [step.to_json() for step in state.adapter_steps],
                "adapterDiagnostics": [
                    dict(diagnostic) for diagnostic in state.diagnostics
                ],
                **state.details,
            },
        )

    def compile_artifact(self, state: RuntimeExecutionState) -> Any:
        state.record_step(
            "compile",
            "prepare-translated-artifact",
            details={"stepCount": len(state.plan.compile_steps)},
        )
        return state.plan.artifact_path

    def load_artifact(self, state: RuntimeExecutionState) -> Any:
        state.record_step(
            "load",
            "load-translated-artifact",
            details={"stepCount": len(state.plan.load_steps)},
        )
        return state.compiled_artifact

    def bind_resources(self, state: RuntimeExecutionState) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for resource in state.plan.resource_bindings:
            if resource.value is None:
                continue
            binding = resource.binding
            key = binding.name or binding.binding_id or str(binding.binding)
            values[key] = (
                None if resource.source == "expectedOutput" else resource.value.values
            )
        state.record_step(
            "bind",
            "bind-fixture-resources",
            details={"resourceCount": len(values)},
        )
        return values

    def bind_constants(self, state: RuntimeExecutionState) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for constant in state.plan.constant_bindings:
            key = constant.constant.name
            if key is None and constant.constant.constant_id is not None:
                key = str(constant.constant.constant_id)
            if key is not None and constant.value is not None:
                values[key] = constant.value
        state.record_step(
            "bind",
            "bind-runtime-constants",
            details={"constantCount": len(values)},
        )
        return values

    def dispatch_fixture(self, state: RuntimeExecutionState) -> Any:
        state.record_step("dispatch", "dispatch-fixture", status=UNAVAILABLE)
        raise RuntimeExecutorUnavailable(
            "Runtime execution adapter must implement dispatch_fixture."
        )

    def collect_outputs(self, state: RuntimeExecutionState) -> Any:
        state.record_step("collect", "collect-fixture-outputs")
        return state.outputs


class RuntimeParityAdapter:
    """Interface for backend adapters that execute translated runtime artifacts."""

    name = "runtime-parity-adapter"
    target: str | None = None
    targets: tuple[str, ...] = ()

    def is_available(
        self, request: RuntimeExecutionRequest
    ) -> RuntimeExecutorAvailability:
        return RuntimeExecutorAvailability(True)

    def prepare_buffers(self, state: RuntimeExecutionState) -> Any:
        return dict(state.resource_values)

    def dispatch(self, state: RuntimeExecutionState, prepared_buffers: Any) -> Any:
        raise RuntimeExecutorUnavailable(
            "Runtime parity adapter must implement dispatch."
        )

    def collect_outputs(
        self, state: RuntimeExecutionState, dispatch_result: Any
    ) -> Any:
        return dispatch_result


class RuntimeParityExecutor(RuntimeExecutionAdapter):
    """Executor wrapper that runs translated artifacts through a target adapter."""

    name = "runtime-parity-executor"

    def __init__(
        self,
        adapter: RuntimeTestAdapterSpec,
        runtime_adapter: Any | None = None,
    ):
        self.adapter = adapter
        self.runtime_adapter = runtime_adapter
        self.name = adapter.adapter_kind or self.name

    @property
    def target(self) -> str | None:
        return self.adapter.target

    @property
    def targets(self) -> tuple[str, ...]:
        values = []
        for value in (
            self.adapter.target,
            self.adapter.executor,
            self.adapter.adapter_id,
        ):
            if isinstance(value, str) and value not in values:
                values.append(value)
        return tuple(values)

    def is_available(
        self, request: RuntimeExecutionRequest
    ) -> RuntimeExecutorAvailability:
        missing = _missing_platform_requirements(self.adapter.platform_requirements)
        if missing["tools"] or missing["environment"]:
            missing_labels = missing["tools"] + missing["environment"]
            return RuntimeExecutorAvailability(
                False,
                reason=(
                    "Runtime dependencies are unavailable: "
                    + ", ".join(missing_labels)
                ),
                details={
                    **self._adapter_details(request),
                    "missingTools": missing["tools"],
                    "missingEnvironment": missing["environment"],
                },
            )
        if self.runtime_adapter is None:
            target = self.adapter.target or request.artifact.get("target")
            return RuntimeExecutorAvailability(
                False,
                reason=(
                    "Runtime parity adapter is not implemented for "
                    f"{target or self.adapter.adapter_id}."
                ),
                details=self._adapter_details(request),
            )
        availability = _executor_availability(self.runtime_adapter, request)
        if availability.available:
            return availability
        return RuntimeExecutorAvailability(
            False,
            reason=availability.reason,
            details={
                **self._adapter_details(request),
                **dict(availability.details),
            },
        )

    def dispatch_fixture(self, state: RuntimeExecutionState) -> Any:
        if self.runtime_adapter is None:
            details = self._adapter_details(state.request)
            state.record_step(
                "dispatch",
                "runtime-parity-adapter-gap",
                status=UNAVAILABLE,
                details=details,
            )
            raise RuntimeExecutorUnavailable(
                "Runtime parity adapter is not implemented for "
                f"{details.get('target') or details.get('adapter')}."
            )

        details = self._adapter_details(state.request)
        state.details["runtimeParityAdapter"] = details
        prepared_buffers = self._prepare_buffers(state)
        prepare_details = {
            **details,
            "resourceCount": len(state.resource_values),
        }
        if isinstance(prepared_buffers, Mapping):
            prepare_details["preparedBufferCount"] = len(prepared_buffers)
        state.record_step(
            "prepare",
            "prepare-runtime-buffers",
            details=prepare_details,
        )

        dispatch = getattr(self.runtime_adapter, "dispatch", None)
        if not callable(dispatch):
            raise RuntimeExecutorUnavailable(
                "Runtime parity adapter must expose dispatch(state, buffers)."
            )
        state.record_step(
            "dispatch",
            "dispatch-translated-artifact",
            details=details,
        )
        return dispatch(state, prepared_buffers)

    def collect_outputs(self, state: RuntimeExecutionState) -> Any:
        collect = (
            getattr(self.runtime_adapter, "collect_outputs", None)
            if self.runtime_adapter is not None
            else None
        )
        if not callable(collect):
            return super().collect_outputs(state)
        state.record_step(
            "collect",
            "collect-runtime-outputs",
            details=self._adapter_details(state.request),
        )
        outputs = collect(state, state.outputs)
        return state.outputs if outputs is None else outputs

    def _prepare_buffers(self, state: RuntimeExecutionState) -> Any:
        prepare = (
            getattr(self.runtime_adapter, "prepare_buffers", None)
            if self.runtime_adapter is not None
            else None
        )
        if callable(prepare):
            prepared = prepare(state)
            return state.resource_values if prepared is None else prepared
        return dict(state.resource_values)

    def _adapter_details(self, request: RuntimeExecutionRequest) -> dict[str, Any]:
        runtime_adapter = self.runtime_adapter
        adapter_name = (
            getattr(runtime_adapter, "name", runtime_adapter.__class__.__name__)
            if runtime_adapter is not None
            else None
        )
        return {
            "adapter": self.adapter.adapter_id,
            "adapterKind": self.adapter.adapter_kind,
            "executor": self.adapter.executor,
            "target": self.adapter.target or request.artifact.get("target"),
            "runtimeAdapter": adapter_name,
            "platformRequirements": self.adapter.platform_requirements.to_json(),
        }


def load_runtime_verification_fixtures(
    fixture_path: str | os.PathLike[str],
) -> list[RuntimeFixture]:
    """Load runtime verification fixtures from a JSON or TOML file."""

    path = _filesystem_path_arg(fixture_path, field_name="Runtime fixture path")
    try:
        if path.suffix.lower() == ".toml":
            payload = _load_toml(path)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeVerificationError(
            f"Runtime fixture file could not be read: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeVerificationError(
            f"Runtime fixture file is not valid JSON: {exc}"
        ) from exc
    return parse_runtime_verification_fixtures(payload)


def parse_runtime_verification_fixtures(payload: Any) -> list[RuntimeFixture]:
    """Parse runtime fixtures from a mapping or fixture sequence."""

    if isinstance(payload, Mapping):
        if payload.get("kind") not in (None, RUNTIME_VERIFICATION_FIXTURES_KIND):
            raise RuntimeVerificationError(
                "Runtime fixture document kind must be "
                f"{RUNTIME_VERIFICATION_FIXTURES_KIND}."
            )
        fixtures = payload.get("fixtures")
    else:
        fixtures = payload
    if not isinstance(fixtures, Sequence) or isinstance(
        fixtures, (str, bytes, bytearray)
    ):
        raise RuntimeVerificationError("Runtime fixtures must be a list.")
    return [
        _parse_runtime_fixture(fixture, index=index)
        for index, fixture in enumerate(fixtures)
    ]


def compare_runtime_outputs(
    expected_outputs: Sequence[RuntimeValue | Mapping[str, Any]],
    actual_outputs: Any,
    *,
    default_tolerance: RuntimeTolerance | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compare executor outputs with fixture expectations."""

    tolerance = _parse_tolerance(default_tolerance, field_name="defaultTolerance")
    expected = [
        (
            value
            if isinstance(value, RuntimeValue)
            else _parse_runtime_value(value, field_name=f"expectedOutputs[{index}]")
        )
        for index, value in enumerate(expected_outputs)
    ]
    actual_by_name = _runtime_values_by_name(actual_outputs)
    return [
        _compare_runtime_value(
            expected_value,
            actual_by_name.get(expected_value.name),
            default_tolerance=tolerance,
        )
        for expected_value in expected
    ]


def prepare_runtime_execution(
    request: RuntimeExecutionRequest,
) -> RuntimeExecutionPlan:
    """Prepare backend-neutral runtime execution data for one fixture."""

    artifact = request.artifact
    artifact_path = request.artifact_path
    contract = request.adapter_contract
    dispatch = _complete_runtime_dispatch_geometry(contract)
    resource_bindings, resource_diagnostics = _runtime_bound_resources(
        request.fixture, contract, artifact
    )
    constant_bindings, constant_diagnostics = _runtime_bound_constants(
        contract, artifact
    )
    diagnostics = tuple(resource_diagnostics + constant_diagnostics)
    return RuntimeExecutionPlan(
        artifact=artifact,
        artifact_path=artifact_path,
        compile_steps=_runtime_execution_steps_from_artifact(
            artifact,
            artifact_path=artifact_path,
            phase="compile",
            keys=("compileSteps", "compilerRequests", "buildSteps"),
            fallback_action="use-translated-artifact",
        ),
        load_steps=_runtime_execution_steps_from_artifact(
            artifact,
            artifact_path=artifact_path,
            phase="load",
            keys=("loadSteps", "loaderSteps", "runtimeLoadSteps"),
            fallback_action="load-translated-artifact",
        ),
        resource_bindings=resource_bindings,
        constant_bindings=constant_bindings,
        dispatch=dispatch,
        diagnostics=diagnostics,
    )


def verify_runtime_fixtures(
    artifact_report: str | os.PathLike[str] | Mapping[str, Any],
    fixtures: (
        str
        | os.PathLike[str]
        | Mapping[str, Any]
        | Sequence[RuntimeFixture | Mapping[str, Any]]
    ),
    *,
    executors: Mapping[str, Any] | Sequence[Any] | None = None,
    project_root: str | os.PathLike[str] | None = None,
    output_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run runtime fixtures against translated artifacts and return a JSON report."""

    artifact_payload, artifact_source = _load_artifact_payload(artifact_report)
    runtime_fixtures, fixture_source = _load_fixture_argument(fixtures)
    root_path = _runtime_project_root(
        artifact_payload,
        artifact_source=artifact_source,
        override=project_root,
    )
    executor_registry = _executor_registry(executors)

    results = [
        _verify_runtime_fixture(
            fixture,
            artifact_payload,
            root_path=root_path,
            executors=executor_registry,
        )
        for fixture in runtime_fixtures
    ]
    summary = _runtime_verification_summary(results)
    report = {
        "schemaVersion": RUNTIME_VERIFICATION_SCHEMA_VERSION,
        "kind": RUNTIME_VERIFICATION_REPORT_KIND,
        "generatedAt": int(time.time()),
        "success": summary["failedCount"] == 0,
        "sourceArtifacts": {
            "path": str(artifact_source) if artifact_source is not None else None,
            "kind": artifact_payload.get("kind"),
        },
        "fixtureSource": str(fixture_source) if fixture_source is not None else None,
        "projectRoot": str(root_path) if root_path is not None else None,
        "summary": summary,
        "results": results,
    }
    if output_path is not None:
        write_runtime_verification_report(report, output_path)
    return report


def load_runtime_test_manifest(
    manifest_path: str | os.PathLike[str],
) -> RuntimeTestManifest:
    """Load a project runtime test manifest from JSON or TOML."""

    path = _filesystem_path_arg(
        manifest_path, field_name="Runtime test manifest path"
    )
    try:
        if path.suffix.lower() == ".toml":
            payload = _load_toml(path)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeVerificationError(
            f"Runtime test manifest could not be read: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeVerificationError(
            f"Runtime test manifest is not valid JSON: {exc}"
        ) from exc
    return parse_runtime_test_manifest(payload)


def parse_runtime_test_manifest(payload: Any) -> RuntimeTestManifest:
    """Parse a project-level manifest that maps fixtures to runtime adapters."""

    if isinstance(payload, RuntimeTestManifest):
        return payload
    if not isinstance(payload, Mapping):
        raise RuntimeVerificationError("Runtime test manifest must be an object.")
    if payload.get("kind") not in (None, RUNTIME_TEST_MANIFEST_KIND):
        raise RuntimeVerificationError(
            "Runtime test manifest kind must be "
            f"{RUNTIME_TEST_MANIFEST_KIND}."
        )

    adapters = tuple(
        _parse_runtime_test_adapter(adapter, index=index)
        for index, adapter in enumerate(
            _required_sequence(payload.get("adapters", []), field_name="adapters")
        )
    )
    adapter_by_id = {adapter.adapter_id: adapter for adapter in adapters}
    if len(adapter_by_id) != len(adapters):
        raise RuntimeVerificationError("Runtime test adapter ids must be unique.")

    test_payloads = payload.get("tests", payload.get("fixtures", []))
    test_cases = tuple(
        _parse_runtime_test_case(
            test_case,
            index=index,
            adapters=adapter_by_id,
        )
        for index, test_case in enumerate(
            _required_sequence(test_payloads, field_name="tests")
        )
    )
    fixture_ids = [test_case.fixture.id for test_case in test_cases]
    if len(set(fixture_ids)) != len(fixture_ids):
        raise RuntimeVerificationError("Runtime test fixture ids must be unique.")

    metadata = payload.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise RuntimeVerificationError("metadata must be an object.")
    return RuntimeTestManifest(
        adapters=adapters,
        test_cases=test_cases,
        artifact_manifest=_optional_string(
            payload.get("artifactManifest"),
            field_name="artifactManifest",
        ),
        project_root=_optional_string(
            payload.get("projectRoot"),
            field_name="projectRoot",
        ),
        metadata=dict(metadata),
    )


def default_runtime_test_adapters(
    targets: Sequence[str] | None = None,
) -> tuple[RuntimeTestAdapterSpec, ...]:
    """Return metadata-only runtime probe adapters for known target classes."""

    target_names = (
        sorted(RUNTIME_TEST_DEFAULT_ADAPTERS)
        if targets is None
        else [
            _normalize_target(target)
            for target in targets
            if isinstance(target, str) and target.strip()
        ]
    )
    adapters = [
        _runtime_test_default_adapter(target)
        for target in target_names
        if target in RUNTIME_TEST_DEFAULT_ADAPTERS
    ]
    return tuple(adapters)


def plan_runtime_test_manifest(
    artifact_report: str | os.PathLike[str] | Mapping[str, Any],
    manifest: (
        str | os.PathLike[str] | Mapping[str, Any] | RuntimeTestManifest | None
    ) = None,
    *,
    project_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Plan project runtime tests without requiring native runtime execution."""

    runtime_manifest, manifest_source = _load_runtime_test_manifest_argument(
        artifact_report if manifest is None else manifest
    )
    artifact_input = (
        _runtime_test_manifest_artifact_input(runtime_manifest, manifest_source)
        if manifest is None
        else artifact_report
    )
    artifact_payload, artifact_source = _load_artifact_payload(artifact_input)
    root_path = _runtime_project_root(
        artifact_payload,
        artifact_source=artifact_source,
        override=project_root or runtime_manifest.project_root,
    )
    adapters = _runtime_test_adapters(runtime_manifest, artifact_payload)
    adapter_by_id = {adapter.adapter_id: adapter for adapter in adapters}
    test_cases = [
        _runtime_test_case_with_adapter(test_case, adapter_by_id)
        for test_case in runtime_manifest.test_cases
    ]
    cases = [
        _runtime_test_plan_case(
            test_case,
            artifact_payload,
            root_path=root_path,
            adapter_by_id=adapter_by_id,
        )
        for test_case in test_cases
    ]
    summary = _runtime_test_summary(cases)
    return {
        "schemaVersion": RUNTIME_VERIFICATION_SCHEMA_VERSION,
        "kind": RUNTIME_TEST_PLAN_KIND,
        "generatedAt": int(time.time()),
        "success": summary["failedCount"] == 0,
        "sourceArtifacts": {
            "path": str(artifact_source) if artifact_source is not None else None,
            "kind": artifact_payload.get("kind"),
        },
        "sourceManifest": str(manifest_source) if manifest_source is not None else None,
        "projectRoot": str(root_path) if root_path is not None else None,
        "summary": summary,
        "adapters": [adapter.to_json() for adapter in adapters],
        "testCases": cases,
    }


def verify_runtime_test_manifest(
    artifact_report: str | os.PathLike[str] | Mapping[str, Any],
    manifest: (
        str | os.PathLike[str] | Mapping[str, Any] | RuntimeTestManifest | None
    ) = None,
    *,
    executors: Mapping[str, Any] | Sequence[Any] | None = None,
    project_root: str | os.PathLike[str] | None = None,
    output_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run or skip project runtime tests from a backend-agnostic manifest."""

    runtime_manifest, manifest_source = _load_runtime_test_manifest_argument(
        artifact_report if manifest is None else manifest
    )
    artifact_input = (
        _runtime_test_manifest_artifact_input(runtime_manifest, manifest_source)
        if manifest is None
        else artifact_report
    )
    artifact_payload, artifact_source = _load_artifact_payload(artifact_input)
    root_path = _runtime_project_root(
        artifact_payload,
        artifact_source=artifact_source,
        override=project_root or runtime_manifest.project_root,
    )
    adapters = _runtime_test_adapters(runtime_manifest, artifact_payload)
    adapter_by_id = {adapter.adapter_id: adapter for adapter in adapters}
    test_cases = [
        _runtime_test_case_with_adapter(test_case, adapter_by_id)
        for test_case in runtime_manifest.test_cases
    ]
    plan_cases = [
        _runtime_test_plan_case(
            test_case,
            artifact_payload,
            root_path=root_path,
            adapter_by_id=adapter_by_id,
        )
        for test_case in test_cases
    ]
    preplanned_results = {
        plan_case["fixture"]: _runtime_test_planned_result(test_case, plan_case)
        for test_case, plan_case in zip(test_cases, plan_cases)
        if plan_case["status"] in {SKIPPED, TRANSLATION_FAILED, RUNTIME_FAILED}
    }
    runnable = [
        test_case.fixture
        for test_case, plan_case in zip(test_cases, plan_cases)
        if plan_case["fixture"] not in preplanned_results
    ]
    executor_registry = _runtime_test_executor_registry(adapters, executors)
    run_results: dict[str, dict[str, Any]] = {}
    if runnable:
        run_report = verify_runtime_fixtures(
            artifact_payload,
            runnable,
            executors=executor_registry,
            project_root=root_path,
        )
        run_results = {
            result["fixture"]: result
            for result in run_report.get("results", [])
            if isinstance(result, Mapping)
        }
    results = [
        preplanned_results.get(test_case.fixture.id)
        or run_results.get(test_case.fixture.id)
        or _runtime_test_missing_result(test_case)
        for test_case in test_cases
    ]
    summary = _runtime_verification_summary(results)
    report = {
        "schemaVersion": RUNTIME_VERIFICATION_SCHEMA_VERSION,
        "kind": RUNTIME_TEST_REPORT_KIND,
        "generatedAt": int(time.time()),
        "success": summary["failedCount"] == 0,
        "sourceArtifacts": {
            "path": str(artifact_source) if artifact_source is not None else None,
            "kind": artifact_payload.get("kind"),
        },
        "sourceManifest": str(manifest_source) if manifest_source is not None else None,
        "projectRoot": str(root_path) if root_path is not None else None,
        "summary": summary,
        "runtimeTestPlan": {
            "kind": RUNTIME_TEST_PLAN_KIND,
            "summary": _runtime_test_summary(plan_cases),
            "adapters": [adapter.to_json() for adapter in adapters],
            "testCases": plan_cases,
        },
        "results": results,
    }
    if output_path is not None:
        write_runtime_test_report(report, output_path)
    return report


def write_runtime_verification_report(
    report: Mapping[str, Any], output_path: str | os.PathLike[str]
) -> None:
    """Write a runtime verification report as deterministic JSON."""

    path = _filesystem_path_arg(
        output_path, field_name="Runtime verification output path"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_runtime_test_report(
    report: Mapping[str, Any], output_path: str | os.PathLike[str]
) -> None:
    """Write a project runtime test report as deterministic JSON."""

    path = _filesystem_path_arg(
        output_path, field_name="Runtime test report output path"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_runtime_test_manifest_argument(
    manifest: str | os.PathLike[str] | Mapping[str, Any] | RuntimeTestManifest,
) -> tuple[RuntimeTestManifest, Path | None]:
    if isinstance(manifest, RuntimeTestManifest):
        return manifest, None
    if isinstance(manifest, (str, os.PathLike)):
        path = _filesystem_path_arg(
            manifest, field_name="Runtime test manifest path"
        )
        return load_runtime_test_manifest(path), path
    if isinstance(manifest, Mapping):
        return parse_runtime_test_manifest(manifest), None
    raise RuntimeVerificationError(
        "Runtime test manifest must be a manifest path or object."
    )


def _runtime_test_manifest_artifact_input(
    manifest: RuntimeTestManifest, manifest_source: Path | None
) -> str | os.PathLike[str]:
    if manifest.artifact_manifest is None:
        raise RuntimeVerificationError(
            "Runtime test manifest must include artifactManifest when no "
            "artifact report argument is provided."
        )
    artifact_path = Path(manifest.artifact_manifest)
    if not artifact_path.is_absolute() and manifest_source is not None:
        artifact_path = manifest_source.parent / artifact_path
    return artifact_path


def _parse_runtime_test_adapter(
    value: Any, *, index: int
) -> RuntimeTestAdapterSpec:
    if isinstance(value, RuntimeTestAdapterSpec):
        return value
    if not isinstance(value, Mapping):
        raise RuntimeVerificationError(f"adapters[{index}] must be an object.")
    adapter_id = _required_string(value.get("id"), field_name=f"adapters[{index}].id")
    target = _optional_string(
        value.get("target"), field_name=f"adapters[{index}].target"
    )
    normalized_target = _normalize_target(target) if target is not None else None
    default_adapter = (
        _runtime_test_default_adapter(normalized_target)
        if normalized_target in RUNTIME_TEST_DEFAULT_ADAPTERS
        else None
    )
    platform_requirements = _parse_runtime_platform_requirements(
        value.get("platformRequirements", value),
        field_name=f"adapters[{index}].platformRequirements",
        defaults=(
            default_adapter.platform_requirements
            if default_adapter is not None
            else RuntimePlatformRequirements()
        ),
    )
    metadata = value.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise RuntimeVerificationError(f"adapters[{index}].metadata must be an object.")
    return RuntimeTestAdapterSpec(
        adapter_id=adapter_id,
        target=normalized_target,
        executor=_optional_string(
            value.get("executor"), field_name=f"adapters[{index}].executor"
        )
        or normalized_target
        or adapter_id,
        adapter_kind=(
            _optional_string(
                value.get("adapterKind"), field_name=f"adapters[{index}].adapterKind"
            )
            or (default_adapter.adapter_kind if default_adapter is not None else None)
        ),
        platform_requirements=platform_requirements,
        metadata=dict(metadata),
    )


def _parse_runtime_test_case(
    value: Any,
    *,
    index: int,
    adapters: Mapping[str, RuntimeTestAdapterSpec],
) -> RuntimeTestCase:
    if isinstance(value, RuntimeTestCase):
        return value
    if not isinstance(value, Mapping):
        raise RuntimeVerificationError(f"tests[{index}] must be an object.")
    adapter_id = _optional_string(
        value.get("adapter", value.get("adapterId")),
        field_name=f"tests[{index}].adapter",
    )
    adapter = adapters.get(adapter_id) if adapter_id is not None else None
    if adapter_id is not None and adapter is None:
        raise RuntimeVerificationError(
            f"tests[{index}].adapter references unknown adapter {adapter_id}."
        )
    fixture_payload = _runtime_test_fixture_payload(value)
    if adapter is not None and "executor" not in fixture_payload:
        fixture_payload["executor"] = adapter.executor or adapter.adapter_id
    fixture = _parse_runtime_fixture(fixture_payload, index=index)
    platform_requirements = _parse_runtime_platform_requirements(
        value.get("platformRequirements", {}),
        field_name=f"tests[{index}].platformRequirements",
        defaults=(
            adapter.platform_requirements
            if adapter is not None
            else RuntimePlatformRequirements()
        ),
    )
    metadata = value.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise RuntimeVerificationError(f"tests[{index}].metadata must be an object.")
    return RuntimeTestCase(
        fixture=fixture,
        adapter=adapter_id,
        platform_requirements=platform_requirements,
        metadata=dict(metadata),
    )


def _runtime_test_fixture_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    fixture_record = value.get("fixture")
    if fixture_record is not None:
        if not isinstance(fixture_record, Mapping):
            raise RuntimeVerificationError("Runtime test fixture must be an object.")
        payload = dict(fixture_record)
    else:
        payload = dict(value)
    if "artifact" in value and not any(
        key in payload for key in ("selector", "artifactSelector", "artifact")
    ):
        payload["selector"] = value["artifact"]
    return payload


def _parse_runtime_platform_requirements(
    value: Any,
    *,
    field_name: str,
    defaults: RuntimePlatformRequirements | None = None,
) -> RuntimePlatformRequirements:
    defaults = defaults or RuntimePlatformRequirements()
    if value is None:
        value = {}
    if isinstance(value, RuntimePlatformRequirements):
        return _merge_runtime_platform_requirements(defaults, value)
    if not isinstance(value, Mapping):
        raise RuntimeVerificationError(f"{field_name} must be an object.")
    platform_class = _optional_string(
        value.get("platformClass", value.get("platform")),
        field_name=f"{field_name}.platformClass",
    )
    requirements = RuntimePlatformRequirements(
        platform_class=platform_class or defaults.platform_class,
        required_tools=_merge_string_tuples(
            defaults.required_tools,
            _string_tuple(
                value.get("requiredTools", value.get("tools", ())),
                field_name=f"{field_name}.requiredTools",
            ),
        ),
        required_environment=_merge_string_tuples(
            defaults.required_environment,
            _string_tuple(
                value.get(
                    "requiredEnvironment",
                    value.get("environment", value.get("env", ())),
                ),
                field_name=f"{field_name}.requiredEnvironment",
            ),
        ),
        runtimes=_merge_string_tuples(
            defaults.runtimes,
            _string_tuple(
                value.get("runtimes", value.get("runtime", ())),
                field_name=f"{field_name}.runtimes",
            ),
        ),
        metadata={
            **dict(defaults.metadata),
            **(
                dict(value.get("metadata"))
                if isinstance(value.get("metadata"), Mapping)
                else {}
            ),
        },
    )
    return requirements


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_required_string(value, field_name=field_name),)
    items = _required_sequence(value, field_name=field_name)
    result = []
    for index, item in enumerate(items):
        result.append(_required_string(item, field_name=f"{field_name}[{index}]"))
    return tuple(result)


def _merge_string_tuples(
    base: Sequence[str], override: Sequence[str]
) -> tuple[str, ...]:
    result: list[str] = []
    for value in tuple(base) + tuple(override):
        if value not in result:
            result.append(value)
    return tuple(result)


def _merge_runtime_platform_requirements(
    base: RuntimePlatformRequirements, override: RuntimePlatformRequirements
) -> RuntimePlatformRequirements:
    return RuntimePlatformRequirements(
        platform_class=override.platform_class or base.platform_class,
        required_tools=_merge_string_tuples(
            base.required_tools, override.required_tools
        ),
        required_environment=_merge_string_tuples(
            base.required_environment, override.required_environment
        ),
        runtimes=_merge_string_tuples(base.runtimes, override.runtimes),
        metadata={**dict(base.metadata), **dict(override.metadata)},
    )


def _runtime_test_default_adapter(target: str) -> RuntimeTestAdapterSpec:
    entry = RUNTIME_TEST_DEFAULT_ADAPTERS[target]
    return RuntimeTestAdapterSpec(
        adapter_id=f"{target}-runtime-probe",
        target=target,
        executor=target,
        adapter_kind=str(entry["adapterKind"]),
        platform_requirements=RuntimePlatformRequirements(
            platform_class=str(entry["platformClass"]),
            required_tools=tuple(str(tool) for tool in entry["requiredTools"]),
            metadata={"source": "default-runtime-test-adapter"},
        ),
    )


def _runtime_test_adapters(
    manifest: RuntimeTestManifest, artifact_payload: Mapping[str, Any]
) -> tuple[RuntimeTestAdapterSpec, ...]:
    _ = artifact_payload
    adapters = {adapter.adapter_id: adapter for adapter in manifest.adapters}
    targets = {
        test_case.fixture.selector.target
        for test_case in manifest.test_cases
        if test_case.adapter is None
        if test_case.fixture.selector.target is not None
    }
    for target in sorted(targets):
        if target in RUNTIME_TEST_DEFAULT_ADAPTERS:
            default_adapter = _runtime_test_default_adapter(target)
            adapters.setdefault(default_adapter.adapter_id, default_adapter)
    return tuple(adapters.values())


def _runtime_test_case_with_adapter(
    test_case: RuntimeTestCase,
    adapter_by_id: Mapping[str, RuntimeTestAdapterSpec],
) -> RuntimeTestCase:
    if test_case.adapter is not None:
        return test_case
    target = test_case.fixture.selector.target
    if target is None:
        return test_case
    default_id = f"{target}-runtime-probe"
    adapter = adapter_by_id.get(default_id)
    if adapter is None:
        return test_case
    fixture = test_case.fixture
    if fixture.executor is None:
        fixture = replace(fixture, executor=adapter.executor or adapter.adapter_id)
    return RuntimeTestCase(
        fixture=fixture,
        adapter=adapter.adapter_id,
        platform_requirements=_merge_runtime_platform_requirements(
            adapter.platform_requirements,
            test_case.platform_requirements,
        ),
        metadata=test_case.metadata,
    )


def _runtime_test_plan_case(
    test_case: RuntimeTestCase,
    artifact_payload: Mapping[str, Any],
    *,
    root_path: Path | None,
    adapter_by_id: Mapping[str, RuntimeTestAdapterSpec],
) -> dict[str, Any]:
    selected = _select_runtime_artifact(artifact_payload, test_case.fixture.selector)
    adapter = (
        adapter_by_id.get(test_case.adapter) if test_case.adapter is not None else None
    )
    case_payload = {
        "fixture": test_case.fixture.id,
        "selector": test_case.fixture.selector.to_json(),
        "adapter": adapter.to_json() if adapter is not None else None,
        "platformRequirements": test_case.platform_requirements.to_json(),
        "status": "planned",
        "diagnostics": [],
    }
    if selected["status"] != PASSED:
        case_payload.update(
            {
                "status": selected["status"],
                "failurePhase": selected.get("failurePhase"),
                "artifact": _artifact_result_payload(selected.get("artifact")),
                "diagnostics": [
                    dict(diagnostic)
                    for diagnostic in selected.get("diagnostics", [])
                    if isinstance(diagnostic, Mapping)
                ],
            }
        )
        return case_payload

    artifact = selected["artifact"]
    artifact_path = _runtime_artifact_path(artifact, root_path=root_path)
    adapter_contract = _runtime_adapter_contract(test_case.fixture, artifact)
    request = RuntimeExecutionRequest(
        fixture=test_case.fixture,
        artifact=artifact,
        artifact_path=artifact_path,
        project_root=root_path,
        adapter_contract=adapter_contract,
    )
    execution_plan = prepare_runtime_execution(request)
    diagnostics = list(execution_plan.diagnostics)
    missing = _missing_platform_requirements(test_case.platform_requirements)
    if missing["tools"] or missing["environment"]:
        diagnostics.append(
            _runtime_test_platform_skip_diagnostic(
                test_case,
                artifact,
                missing=missing,
            )
        )
        status = SKIPPED
        failure_phase = "platform-requirements"
    elif _runtime_diagnostics_have_errors(diagnostics):
        status = RUNTIME_FAILED
        failure_phase = "runtime-setup"
    else:
        status = "planned"
        failure_phase = None
    case_payload.update(
        {
            "status": status,
            "artifact": _artifact_result_payload(artifact),
            "runtimeAdapter": adapter_contract.to_json(),
            "runtimeExecution": execution_plan.to_json(),
            "diagnostics": diagnostics,
        }
    )
    if failure_phase is not None:
        case_payload["failurePhase"] = failure_phase
    return case_payload


def _runtime_test_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    statuses = Counter(case.get("status") for case in cases)
    failed_count = statuses[TRANSLATION_FAILED] + statuses[RUNTIME_FAILED]
    failed_count += statuses[COMPARISON_FAILED]
    return {
        "testCount": len(cases),
        "plannedCount": statuses["planned"],
        "skippedCount": statuses[SKIPPED],
        "unavailableCount": statuses[UNAVAILABLE],
        "translationFailedCount": statuses[TRANSLATION_FAILED],
        "runtimeFailedCount": statuses[RUNTIME_FAILED],
        "comparisonFailedCount": statuses[COMPARISON_FAILED],
        "failedCount": failed_count,
    }


def _runtime_test_planned_result(
    test_case: RuntimeTestCase, plan_case: Mapping[str, Any]
) -> dict[str, Any]:
    executor_key = test_case.fixture.executor
    status = str(plan_case.get("status"))
    executor_payload = None
    if status == SKIPPED:
        diagnostics = [
            diagnostic
            for diagnostic in _runtime_record_sequence(plan_case.get("diagnostics"))
            if isinstance(diagnostic, Mapping)
        ]
        skip_diagnostic = diagnostics[-1] if diagnostics else {}
        executor_payload = {
            "key": executor_key,
            "status": SKIPPED,
            "message": skip_diagnostic.get("message"),
            "details": {
                "platformRequirements": plan_case.get("platformRequirements", {}),
                "missingTools": skip_diagnostic.get("missingTools", []),
                "missingEnvironment": skip_diagnostic.get("missingEnvironment", []),
            },
        }
    result = {
        "fixture": test_case.fixture.id,
        "selector": test_case.fixture.selector.to_json(),
        "status": status,
        "artifact": plan_case.get("artifact"),
        "executor": executor_payload,
        "comparisons": [],
        "diagnostics": [
            dict(diagnostic)
            for diagnostic in _runtime_record_sequence(plan_case.get("diagnostics"))
            if isinstance(diagnostic, Mapping)
        ],
    }
    for key in ("runtimeAdapter", "runtimeExecution", "failurePhase"):
        if key in plan_case:
            result[key] = plan_case[key]
    return result


def _runtime_test_missing_result(test_case: RuntimeTestCase) -> dict[str, Any]:
    return _fixture_result(
        test_case.fixture,
        status=RUNTIME_FAILED,
        failure_phase="runtime",
        diagnostics=[
            _diagnostic(
                "error",
                "project.runtime-test.result-missing",
                "Runtime test did not produce a result.",
            )
        ],
    )


def _runtime_test_executor_registry(
    adapters: Sequence[RuntimeTestAdapterSpec],
    executors: Mapping[str, Any] | Sequence[Any] | None,
) -> dict[str, Any]:
    registry = _executor_registry(executors)
    for adapter in adapters:
        executor_key = _normalize_target(adapter.executor or adapter.adapter_id)
        executor = registry.get(executor_key)
        if executor is None:
            registry[executor_key] = RuntimeParityExecutor(adapter)
        elif _is_runtime_parity_adapter(executor):
            registry[executor_key] = RuntimeParityExecutor(adapter, executor)
    return registry


def _is_runtime_parity_adapter(value: Any) -> bool:
    if isinstance(value, RuntimeParityExecutor):
        return False
    required_methods = ("prepare_buffers", "dispatch", "collect_outputs")
    return all(callable(getattr(value, method, None)) for method in required_methods)


def _runtime_test_platform_skip_diagnostic(
    test_case: RuntimeTestCase,
    artifact: Mapping[str, Any],
    *,
    missing: Mapping[str, list[str]],
) -> dict[str, Any]:
    missing_labels = list(missing.get("tools", [])) + list(
        missing.get("environment", [])
    )
    message = (
        "Runtime test skipped because platform requirements are unavailable: "
        + ", ".join(missing_labels)
    )
    return _runtime_execution_diagnostic(
        "note",
        "project.runtime-test.platform-requirements-unavailable",
        message,
        artifact,
        fixture=test_case.fixture.id,
        adapter=test_case.adapter,
        platformRequirements=test_case.platform_requirements.to_json(),
        missingTools=list(missing.get("tools", [])),
        missingEnvironment=list(missing.get("environment", [])),
    )


def _missing_platform_requirements(
    requirements: RuntimePlatformRequirements,
) -> dict[str, list[str]]:
    return {
        "tools": [
            tool for tool in requirements.required_tools if shutil.which(tool) is None
        ],
        "environment": [
            name
            for name in requirements.required_environment
            if not os.environ.get(name)
        ],
    }


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11 only
        try:
            import tomli as tomllib
        except ImportError as exc:  # pragma: no cover
            raise RuntimeVerificationError(
                "Reading runtime fixture TOML on Python < 3.11 requires tomli."
            ) from exc
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _filesystem_path_arg(value: Any, *, field_name: str) -> Path:
    try:
        path_value = os.fspath(value)
    except TypeError as exc:
        raise RuntimeVerificationError(
            f"{field_name} must be a string or path-like object returning str."
        ) from exc
    if not isinstance(path_value, str):
        raise RuntimeVerificationError(
            f"{field_name} must be a string or path-like object returning str."
        )
    if not path_value.strip():
        raise RuntimeVerificationError(f"{field_name} must be non-empty.")
    return Path(path_value)


def _parse_runtime_fixture(value: Any, *, index: int) -> RuntimeFixture:
    if not isinstance(value, Mapping):
        raise RuntimeVerificationError(f"fixtures[{index}] must be an object.")
    fixture_id = _required_string(value.get("id"), field_name=f"fixtures[{index}].id")
    selector = _parse_artifact_selector(value, field_name=f"fixtures[{index}]")
    default_tolerance = _parse_tolerance(
        value.get("defaultTolerance", value.get("tolerance")),
        field_name=f"fixtures[{index}].defaultTolerance",
    )
    inputs = _parse_runtime_value_list(
        value.get("inputs", []), field_name=f"fixtures[{index}].inputs"
    )
    outputs_payload = value.get("expectedOutputs", value.get("outputs", []))
    expected_outputs = tuple(
        _parse_runtime_value(
            output,
            field_name=f"fixtures[{index}].expectedOutputs[{output_index}]",
            default_tolerance=default_tolerance,
        )
        for output_index, output in enumerate(
            _required_sequence(
                outputs_payload, field_name=f"fixtures[{index}].expectedOutputs"
            )
        )
    )
    adapter_contract = _parse_runtime_adapter_contract(
        value.get("runtimeAdapter", value.get("adapterContract")),
        field_name=f"fixtures[{index}].runtimeAdapter",
    )
    executor = _optional_string(
        value.get("executor"), field_name=f"fixtures[{index}].executor"
    )
    metadata = value.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise RuntimeVerificationError(f"fixtures[{index}].metadata must be an object.")
    return RuntimeFixture(
        id=fixture_id,
        selector=selector,
        inputs=inputs,
        expected_outputs=expected_outputs,
        default_tolerance=default_tolerance,
        adapter_contract=adapter_contract,
        executor=executor,
        metadata=dict(metadata),
    )


def _parse_artifact_selector(value: Mapping[str, Any], *, field_name: str):
    selector_payload = (
        value.get("artifactSelector")
        if "artifactSelector" in value
        else value.get("selector", value.get("artifact", None))
    )
    if selector_payload is None:
        selector_payload = {
            key: value[key]
            for key in ("source", "target", "variant", "stage", "path", "artifactId")
            if key in value
        }
    if not isinstance(selector_payload, Mapping):
        raise RuntimeVerificationError(f"{field_name}.selector must be an object.")
    artifact_id = selector_payload.get("id", selector_payload.get("artifactId"))
    target = _optional_string(
        selector_payload.get("target"), field_name=f"{field_name}.selector.target"
    )
    selector = RuntimeArtifactSelector(
        source=_optional_string(
            selector_payload.get("source"), field_name=f"{field_name}.selector.source"
        ),
        target=_normalize_target(target) if target is not None else None,
        variant=_optional_string(
            selector_payload.get("variant"), field_name=f"{field_name}.selector.variant"
        ),
        stage=_optional_string(
            selector_payload.get("stage"), field_name=f"{field_name}.selector.stage"
        ),
        path=_optional_string(
            selector_payload.get("path"), field_name=f"{field_name}.selector.path"
        ),
        artifact_id=_optional_string(
            artifact_id, field_name=f"{field_name}.selector.id"
        ),
    )
    if not selector.to_json():
        raise RuntimeVerificationError(
            f"{field_name}.selector must identify an artifact."
        )
    return selector


def _parse_runtime_adapter_contract(
    value: Any, *, field_name: str
) -> RuntimeAdapterContract | None:
    if value is None:
        return None
    if isinstance(value, RuntimeAdapterContract):
        return value
    if not isinstance(value, Mapping):
        raise RuntimeVerificationError(f"{field_name} must be an object.")
    metadata = _parse_runtime_metadata(
        value.get("metadata", {}), field_name=f"{field_name}.metadata"
    )
    dispatch = (
        _parse_runtime_dispatch_geometry(
            value.get("dispatch", value.get("dispatchGeometry")),
            field_name=f"{field_name}.dispatch",
        )
        if "dispatch" in value or "dispatchGeometry" in value
        else None
    )
    return RuntimeAdapterContract(
        contract_id=_optional_string(value.get("id"), field_name=f"{field_name}.id"),
        entry_points=tuple(
            _parse_runtime_entry_point(
                entry_point, field_name=f"{field_name}.entryPoints[{index}]"
            )
            for index, entry_point in enumerate(
                _required_sequence(
                    value.get("entryPoints", []),
                    field_name=f"{field_name}.entryPoints",
                )
            )
        ),
        resource_bindings=tuple(
            _parse_runtime_resource_binding(
                binding, field_name=f"{field_name}.resourceBindings[{index}]"
            )
            for index, binding in enumerate(
                _required_sequence(
                    value.get("resourceBindings", value.get("resources", [])),
                    field_name=f"{field_name}.resourceBindings",
                )
            )
        ),
        specialization_constants=_parse_runtime_specialization_constants(
            value, field_name=field_name
        ),
        dispatch=dispatch,
        validation_hooks=tuple(
            _parse_runtime_validation_hook(
                hook, field_name=f"{field_name}.validationHooks[{index}]"
            )
            for index, hook in enumerate(
                _required_sequence(
                    value.get("validationHooks", []),
                    field_name=f"{field_name}.validationHooks",
                )
            )
        ),
        metadata=metadata,
    )


def _parse_runtime_entry_point(value: Any, *, field_name: str) -> RuntimeEntryPoint:
    if isinstance(value, RuntimeEntryPoint):
        return value
    if not isinstance(value, Mapping):
        raise RuntimeVerificationError(f"{field_name} must be an object.")
    execution_config = value.get("executionConfig", {})
    if execution_config is None:
        execution_config = {}
    if not isinstance(execution_config, Mapping):
        raise RuntimeVerificationError(
            f"{field_name}.executionConfig must be an object."
        )
    return RuntimeEntryPoint(
        name=_required_string(value.get("name"), field_name=f"{field_name}.name"),
        stage=_optional_string(value.get("stage"), field_name=f"{field_name}.stage"),
        execution_config=dict(execution_config),
        workgroup_size=_parse_runtime_vector(
            value.get("workgroupSize"),
            field_name=f"{field_name}.workgroupSize",
            integer_only=False,
        ),
        parameters=tuple(
            _parse_runtime_metadata(
                parameter, field_name=f"{field_name}.parameters[{index}]"
            )
            for index, parameter in enumerate(
                _required_sequence(
                    value.get("parameters", []), field_name=f"{field_name}.parameters"
                )
            )
        ),
        metadata=_parse_runtime_metadata(
            value.get("metadata", {}), field_name=f"{field_name}.metadata"
        ),
    )


def _parse_runtime_resource_binding(
    value: Any, *, field_name: str
) -> RuntimeResourceBinding:
    if isinstance(value, RuntimeResourceBinding):
        return value
    if not isinstance(value, Mapping):
        raise RuntimeVerificationError(f"{field_name} must be an object.")
    binding_id = _optional_string(value.get("id"), field_name=f"{field_name}.id")
    name = _optional_string(value.get("name"), field_name=f"{field_name}.name")
    kind = _optional_string(value.get("kind"), field_name=f"{field_name}.kind")
    index = value.get("index")
    if index is not None and (
        not isinstance(index, int) or isinstance(index, bool) or index < 0
    ):
        raise RuntimeVerificationError(
            f"{field_name}.index must be a non-negative integer."
        )
    if binding_id is None and name is None and value.get("binding") is None:
        raise RuntimeVerificationError(
            f"{field_name} must identify a resource by id, name, or binding."
        )
    metadata = _parse_runtime_metadata(
        value.get("metadata", {}), field_name=f"{field_name}.metadata"
    )
    if "required" in value:
        metadata["required"] = _optional_bool(
            value.get("required"), field_name=f"{field_name}.required"
        )
    return RuntimeResourceBinding(
        binding_id=binding_id,
        name=name,
        kind=kind,
        type_name=_optional_string(value.get("type"), field_name=f"{field_name}.type"),
        set=value.get("set"),
        binding=value.get("binding"),
        index=index,
        access=_optional_string(value.get("access"), field_name=f"{field_name}.access"),
        value=_optional_string(value.get("value"), field_name=f"{field_name}.value"),
        metadata=metadata,
    )


def _parse_runtime_specialization_constants(
    value: Mapping[str, Any], *, field_name: str
) -> tuple[RuntimeSpecializationConstant, ...]:
    constants: list[RuntimeSpecializationConstant] = []
    for key, default_kind in (
        ("specializationConstants", "specialization-constant"),
        ("functionConstants", "function-constant"),
    ):
        for index, constant in enumerate(
            _required_sequence(value.get(key, []), field_name=f"{field_name}.{key}")
        ):
            constants.append(
                _parse_runtime_specialization_constant(
                    constant,
                    field_name=f"{field_name}.{key}[{index}]",
                    default_kind=default_kind,
                )
            )
    return tuple(constants)


def _parse_runtime_specialization_constant(
    value: Any, *, field_name: str, default_kind: str
) -> RuntimeSpecializationConstant:
    if isinstance(value, RuntimeSpecializationConstant):
        return value
    if not isinstance(value, Mapping):
        raise RuntimeVerificationError(f"{field_name} must be an object.")
    name = _optional_string(value.get("name"), field_name=f"{field_name}.name")
    constant_id = value.get("id", value.get("constantId"))
    if name is None and constant_id is None:
        raise RuntimeVerificationError(
            f"{field_name} must identify a constant by name or id."
        )
    return RuntimeSpecializationConstant(
        name=name,
        constant_id=constant_id,
        kind=(
            _optional_string(value.get("kind"), field_name=f"{field_name}.kind")
            or default_kind
        ),
        dtype=_optional_string(value.get("dtype"), field_name=f"{field_name}.dtype"),
        value=value.get("value"),
        default=value.get("default"),
        required=_optional_bool(
            value.get("required", False), field_name=f"{field_name}.required"
        ),
        metadata=_parse_runtime_metadata(
            value.get("metadata", {}), field_name=f"{field_name}.metadata"
        ),
    )


def _parse_runtime_dispatch_geometry(
    value: Any, *, field_name: str
) -> RuntimeDispatchGeometry:
    if isinstance(value, RuntimeDispatchGeometry):
        return value
    if not isinstance(value, Mapping):
        raise RuntimeVerificationError(f"{field_name} must be an object.")
    return RuntimeDispatchGeometry(
        entry_point=_optional_string(
            value.get("entryPoint", value.get("entry_point")),
            field_name=f"{field_name}.entryPoint",
        ),
        workgroup_size=_parse_runtime_vector(
            value.get("workgroupSize"),
            field_name=f"{field_name}.workgroupSize",
            integer_only=False,
        ),
        workgroup_count=_parse_runtime_vector(
            value.get("workgroupCount"),
            field_name=f"{field_name}.workgroupCount",
            integer_only=True,
        ),
        global_size=_parse_runtime_vector(
            value.get("globalSize"),
            field_name=f"{field_name}.globalSize",
            integer_only=True,
        ),
        grid_size=_parse_runtime_vector(
            value.get("gridSize"),
            field_name=f"{field_name}.gridSize",
            integer_only=True,
        ),
        metadata=_parse_runtime_metadata(
            value.get("metadata", {}), field_name=f"{field_name}.metadata"
        ),
    )


def _parse_runtime_validation_hook(
    value: Any, *, field_name: str
) -> RuntimeValidationHook:
    if isinstance(value, RuntimeValidationHook):
        return value
    if not isinstance(value, Mapping):
        raise RuntimeVerificationError(f"{field_name} must be an object.")
    return RuntimeValidationHook(
        name=_required_string(value.get("name"), field_name=f"{field_name}.name"),
        phase=(
            _optional_string(value.get("phase"), field_name=f"{field_name}.phase")
            or "pre-run"
        ),
        required=_optional_bool(
            value.get("required", True), field_name=f"{field_name}.required"
        ),
        metadata=_parse_runtime_metadata(
            value.get("metadata", {}), field_name=f"{field_name}.metadata"
        ),
    )


def _parse_runtime_value_list(value: Any, *, field_name: str):
    return tuple(
        _parse_runtime_value(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(_required_sequence(value, field_name=field_name))
    )


def _parse_runtime_value(
    value: Any,
    *,
    field_name: str,
    default_tolerance: RuntimeTolerance | None = None,
) -> RuntimeValue:
    if not isinstance(value, Mapping):
        raise RuntimeVerificationError(f"{field_name} must be an object.")
    name = _required_string(value.get("name"), field_name=f"{field_name}.name")
    kind = (
        _optional_string(value.get("kind"), field_name=f"{field_name}.kind") or "buffer"
    )
    dtype = _optional_string(value.get("dtype"), field_name=f"{field_name}.dtype")
    shape = _parse_shape(value.get("shape"), field_name=f"{field_name}.shape")
    tolerance = (
        _parse_tolerance(value["tolerance"], field_name=f"{field_name}.tolerance")
        if "tolerance" in value
        else default_tolerance
    )
    metadata = value.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise RuntimeVerificationError(f"{field_name}.metadata must be an object.")
    return RuntimeValue(
        name=name,
        kind=kind,
        dtype=dtype,
        shape=shape,
        values=value.get("values"),
        tolerance=tolerance,
        metadata=dict(metadata),
    )


def _parse_shape(value: Any, *, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RuntimeVerificationError(f"{field_name} must be a list of integers.")
    shape: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise RuntimeVerificationError(
                f"{field_name}[{index}] must be a non-negative integer."
            )
        shape.append(item)
    return tuple(shape)


def _parse_runtime_vector(
    value: Any, *, field_name: str, integer_only: bool
) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        expected = "integers" if integer_only else "scalars"
        raise RuntimeVerificationError(f"{field_name} must be a list of {expected}.")
    vector: list[Any] = []
    for index, item in enumerate(value):
        if integer_only:
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                raise RuntimeVerificationError(
                    f"{field_name}[{index}] must be a non-negative integer."
                )
            vector.append(item)
            continue
        if isinstance(item, (str, int, float)) and not isinstance(item, bool):
            vector.append(item)
            continue
        raise RuntimeVerificationError(f"{field_name}[{index}] must be a scalar.")
    return tuple(vector)


def _parse_runtime_metadata(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeVerificationError(f"{field_name} must be an object.")
    return dict(value)


def _parse_tolerance(
    value: RuntimeTolerance | Mapping[str, Any] | None, *, field_name: str
) -> RuntimeTolerance:
    if value is None:
        return RuntimeTolerance()
    if isinstance(value, RuntimeTolerance):
        return value
    if not isinstance(value, Mapping):
        raise RuntimeVerificationError(f"{field_name} must be an object.")
    absolute = value.get("absolute", value.get("atol", 0.0))
    relative = value.get("relative", value.get("rtol", 0.0))
    return RuntimeTolerance(
        absolute=_non_negative_float(absolute, field_name=f"{field_name}.absolute"),
        relative=_non_negative_float(relative, field_name=f"{field_name}.relative"),
    )


def _non_negative_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise RuntimeVerificationError(f"{field_name} must be a non-negative number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeVerificationError(
            f"{field_name} must be a non-negative number."
        ) from exc
    if math.isnan(parsed) or parsed < 0:
        raise RuntimeVerificationError(f"{field_name} must be a non-negative number.")
    return parsed


def _required_sequence(value: Any, *, field_name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RuntimeVerificationError(f"{field_name} must be a list.")
    return value


def _required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeVerificationError(f"{field_name} must be a non-empty string.")
    return value


def _optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeVerificationError(f"{field_name} must be a non-empty string.")
    return value


def _optional_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeVerificationError(f"{field_name} must be a boolean.")
    return value


def _normalize_target(target: str) -> str:
    return normalize_backend_name(target) or target.strip().lower()


def _load_artifact_payload(
    artifact_report: str | os.PathLike[str] | Mapping[str, Any],
) -> tuple[Mapping[str, Any], Path | None]:
    if isinstance(artifact_report, Mapping):
        return artifact_report, None
    path = _filesystem_path_arg(artifact_report, field_name="Artifact report path")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeVerificationError(
            f"Artifact report could not be read: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeVerificationError(
            f"Artifact report is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeVerificationError("Artifact report must be a JSON object.")
    return payload, path


def _load_fixture_argument(
    fixtures: (
        str
        | os.PathLike[str]
        | Mapping[str, Any]
        | Sequence[RuntimeFixture | Mapping[str, Any]]
    ),
) -> tuple[list[RuntimeFixture], Path | None]:
    if isinstance(fixtures, (str, os.PathLike)):
        path = _filesystem_path_arg(fixtures, field_name="Runtime fixture path")
        return load_runtime_verification_fixtures(path), path
    if isinstance(fixtures, Mapping):
        return parse_runtime_verification_fixtures(fixtures), None
    if not isinstance(fixtures, Sequence):
        raise RuntimeVerificationError(
            "Runtime fixtures must be a fixture path or list."
        )
    parsed = [
        (
            fixture
            if isinstance(fixture, RuntimeFixture)
            else _parse_runtime_fixture(fixture, index=index)
        )
        for index, fixture in enumerate(fixtures)
    ]
    return parsed, None


def _runtime_project_root(
    payload: Mapping[str, Any],
    *,
    artifact_source: Path | None,
    override: str | os.PathLike[str] | None,
) -> Path | None:
    if override is not None:
        return _filesystem_path_arg(override, field_name="Project root").resolve()
    package_root = payload.get("packageRoot")
    if isinstance(package_root, str) and package_root.strip():
        return Path(package_root).resolve()
    project = payload.get("project")
    if isinstance(project, Mapping):
        root = project.get("root")
        if isinstance(root, str) and root.strip():
            return Path(root).resolve()
    if artifact_source is not None:
        return artifact_source.parent.resolve()
    return None


def _executor_registry(executors: Mapping[str, Any] | Sequence[Any] | None):
    registry: dict[str, Any] = {}
    if executors is None:
        return registry
    if isinstance(executors, Mapping):
        for key, executor in executors.items():
            if not isinstance(key, str) or not key.strip():
                raise RuntimeVerificationError(
                    "Executor registry keys must be strings."
                )
            registry[_normalize_target(key)] = executor
        return registry
    if not isinstance(executors, Sequence) or isinstance(
        executors, (str, bytes, bytearray)
    ):
        raise RuntimeVerificationError("Executors must be a mapping or list.")
    for executor in executors:
        targets = getattr(executor, "targets", None)
        if isinstance(targets, str):
            targets = [targets]
        if not targets:
            target = getattr(executor, "target", None)
            targets = [target] if isinstance(target, str) else []
        if not targets:
            raise RuntimeVerificationError(
                "Executor list entries must expose target or targets."
            )
        for target in targets:
            registry[_normalize_target(target)] = executor
    return registry


def _verify_runtime_fixture(
    fixture: RuntimeFixture,
    artifact_payload: Mapping[str, Any],
    *,
    root_path: Path | None,
    executors: Mapping[str, Any],
) -> dict[str, Any]:
    selected = _select_runtime_artifact(artifact_payload, fixture.selector)
    if selected["status"] != PASSED:
        return _fixture_result(
            fixture,
            status=selected["status"],
            failure_phase=selected["failurePhase"],
            artifact=selected.get("artifact"),
            adapter_contract=fixture.adapter_contract,
            diagnostics=selected.get("diagnostics", []),
        )

    artifact = selected["artifact"]
    artifact_path = _runtime_artifact_path(artifact, root_path=root_path)
    adapter_contract = _runtime_adapter_contract(fixture, artifact)
    initial_request = RuntimeExecutionRequest(
        fixture=fixture,
        artifact=artifact,
        artifact_path=artifact_path,
        project_root=root_path,
        adapter_contract=adapter_contract,
    )
    execution_plan = prepare_runtime_execution(initial_request)
    setup_diagnostics = list(execution_plan.diagnostics)
    if _runtime_diagnostics_have_errors(setup_diagnostics):
        return _fixture_result(
            fixture,
            status=RUNTIME_FAILED,
            failure_phase="runtime-setup",
            artifact=artifact,
            adapter_contract=adapter_contract,
            runtime_execution=execution_plan,
            diagnostics=setup_diagnostics,
        )

    target = _normalize_target(
        str(artifact.get("target", fixture.selector.target or ""))
    )
    executor_key = _normalize_target(fixture.executor) if fixture.executor else target
    executor = executors.get(executor_key)
    if executor is None:
        return _fixture_result(
            fixture,
            status=UNAVAILABLE,
            failure_phase="runtime",
            artifact=artifact,
            adapter_contract=adapter_contract,
            runtime_execution=execution_plan,
            executor={"key": executor_key, "status": UNAVAILABLE},
            diagnostics=setup_diagnostics
            + [
                _diagnostic(
                    "note",
                    "project.runtime-verification.executor-unavailable",
                    f"No runtime executor is registered for {executor_key}.",
                    target=target,
                )
            ],
        )

    request = RuntimeExecutionRequest(
        fixture=fixture,
        artifact=artifact,
        artifact_path=artifact_path,
        project_root=root_path,
        adapter_contract=adapter_contract,
        execution_plan=execution_plan,
    )
    executor_name = getattr(executor, "name", executor.__class__.__name__)
    availability = _executor_availability(executor, request)
    if not availability.available:
        return _fixture_result(
            fixture,
            status=UNAVAILABLE,
            failure_phase="runtime",
            artifact=artifact,
            adapter_contract=adapter_contract,
            runtime_execution=execution_plan,
            executor={
                "key": executor_key,
                "name": executor_name,
                "status": UNAVAILABLE,
                "message": availability.reason,
                "details": dict(availability.details),
            },
            diagnostics=setup_diagnostics
            + [
                _diagnostic(
                    "note",
                    "project.runtime-verification.executor-unavailable",
                    availability.reason
                    or f"Runtime executor is unavailable for {target}.",
                    target=target,
                )
            ],
        )

    try:
        execution = _normalize_executor_result(executor.run(request))
    except RuntimeExecutorUnavailable as exc:
        return _fixture_result(
            fixture,
            status=UNAVAILABLE,
            failure_phase="runtime",
            artifact=artifact,
            adapter_contract=adapter_contract,
            runtime_execution=execution_plan,
            executor={
                "key": executor_key,
                "name": executor_name,
                "status": UNAVAILABLE,
            },
            diagnostics=setup_diagnostics
            + [
                _diagnostic(
                    "note",
                    "project.runtime-verification.executor-unavailable",
                    str(exc),
                    target=target,
                )
            ],
        )
    except RuntimeExecutorSkipped as exc:
        return _fixture_result(
            fixture,
            status=SKIPPED,
            artifact=artifact,
            adapter_contract=adapter_contract,
            runtime_execution=execution_plan,
            executor={"key": executor_key, "name": executor_name, "status": SKIPPED},
            diagnostics=setup_diagnostics
            + [
                _diagnostic(
                    "note",
                    "project.runtime-verification.fixture-skipped",
                    str(exc),
                    target=target,
                )
            ],
        )
    except RuntimeExecutionError as exc:
        return _fixture_result(
            fixture,
            status=RUNTIME_FAILED,
            failure_phase="runtime",
            artifact=artifact,
            adapter_contract=adapter_contract,
            runtime_execution=execution_plan,
            executor={
                "key": executor_key,
                "name": executor_name,
                "status": RUNTIME_FAILED,
            },
            diagnostics=setup_diagnostics
            + [
                _diagnostic(
                    "error",
                    "project.runtime-verification.executor-failed",
                    str(exc),
                    target=target,
                )
            ],
        )
    except Exception as exc:  # pragma: no cover - defensive boundary
        return _fixture_result(
            fixture,
            status=RUNTIME_FAILED,
            failure_phase="runtime",
            artifact=artifact,
            adapter_contract=adapter_contract,
            runtime_execution=execution_plan,
            executor={
                "key": executor_key,
                "name": executor_name,
                "status": RUNTIME_FAILED,
            },
            diagnostics=setup_diagnostics
            + [
                _diagnostic(
                    "error",
                    "project.runtime-verification.executor-exception",
                    f"Runtime executor raised {exc.__class__.__name__}: {exc}",
                    target=target,
                )
            ],
        )

    normalized_status = execution.status.strip().lower()
    executor_payload = {
        "key": executor_key,
        "name": executor_name,
        "status": normalized_status,
    }
    if execution.message:
        executor_payload["message"] = execution.message
    if execution.details:
        executor_payload["details"] = dict(execution.details)

    if normalized_status in EXECUTOR_SKIPPED_STATUSES:
        return _fixture_result(
            fixture,
            status=SKIPPED,
            artifact=artifact,
            adapter_contract=adapter_contract,
            runtime_execution=execution_plan,
            executor=executor_payload,
            diagnostics=setup_diagnostics
            + _runtime_executor_status_diagnostics(
                SKIPPED, target=target, execution=execution
            ),
        )
    if normalized_status in EXECUTOR_UNAVAILABLE_STATUSES:
        return _fixture_result(
            fixture,
            status=UNAVAILABLE,
            failure_phase="runtime",
            artifact=artifact,
            adapter_contract=adapter_contract,
            runtime_execution=execution_plan,
            executor=executor_payload,
            diagnostics=setup_diagnostics
            + _runtime_executor_status_diagnostics(
                UNAVAILABLE, target=target, execution=execution
            ),
        )
    if normalized_status in EXECUTOR_FAILED_STATUSES:
        return _fixture_result(
            fixture,
            status=RUNTIME_FAILED,
            failure_phase="runtime",
            artifact=artifact,
            adapter_contract=adapter_contract,
            runtime_execution=execution_plan,
            executor=executor_payload,
            diagnostics=setup_diagnostics,
        )
    if normalized_status not in EXECUTOR_OK_STATUSES:
        return _fixture_result(
            fixture,
            status=RUNTIME_FAILED,
            failure_phase="runtime",
            artifact=artifact,
            adapter_contract=adapter_contract,
            runtime_execution=execution_plan,
            executor=executor_payload,
            diagnostics=setup_diagnostics
            + [
                _diagnostic(
                    "error",
                    "project.runtime-verification.executor-status-invalid",
                    f"Runtime executor returned unsupported status: {execution.status}",
                    target=target,
                )
            ],
        )

    comparisons = compare_runtime_outputs(
        fixture.expected_outputs,
        execution.outputs,
        default_tolerance=fixture.default_tolerance,
    )
    comparison_failed = any(
        comparison["status"] != PASSED for comparison in comparisons
    )
    comparison_diagnostics = (
        _runtime_comparison_diagnostics(comparisons, artifact)
        if comparison_failed
        else []
    )
    return _fixture_result(
        fixture,
        status=COMPARISON_FAILED if comparison_failed else PASSED,
        failure_phase="comparison" if comparison_failed else None,
        artifact=artifact,
        adapter_contract=adapter_contract,
        runtime_execution=execution_plan,
        executor=executor_payload,
        comparisons=comparisons,
        diagnostics=setup_diagnostics + comparison_diagnostics,
    )


def _select_runtime_artifact(
    payload: Mapping[str, Any], selector: RuntimeArtifactSelector
) -> dict[str, Any]:
    artifacts = [
        artifact
        for artifact in payload.get("artifacts", [])
        if isinstance(artifact, Mapping)
        and _selector_matches_artifact(selector, artifact)
    ]
    ready = [artifact for artifact in artifacts if not _artifact_failed(artifact)]
    if len(ready) == 1:
        return {"status": PASSED, "artifact": ready[0]}
    if len(ready) > 1:
        return {
            "status": TRANSLATION_FAILED,
            "failurePhase": "artifact-selection",
            "artifact": ready[0],
            "diagnostics": [
                _diagnostic(
                    "error",
                    "project.runtime-verification.artifact-ambiguous",
                    "Runtime fixture selector matched multiple translated artifacts.",
                )
            ],
        }
    if artifacts:
        failed = artifacts[0]
        return {
            "status": TRANSLATION_FAILED,
            "failurePhase": "translation",
            "artifact": failed,
            "diagnostics": [
                _diagnostic(
                    "error",
                    "project.runtime-verification.translation-failed",
                    str(
                        failed.get("error")
                        or "Selected artifact was not translated successfully."
                    ),
                    target=failed.get("target"),
                )
            ],
        }
    return {
        "status": TRANSLATION_FAILED,
        "failurePhase": "translation",
        "diagnostics": [
            _diagnostic(
                "error",
                "project.runtime-verification.artifact-missing",
                "Runtime fixture selector did not match a translated artifact.",
            )
        ],
    }


def _selector_matches_artifact(
    selector: RuntimeArtifactSelector, artifact: Mapping[str, Any]
) -> bool:
    if selector.artifact_id is not None and artifact.get("id") != selector.artifact_id:
        return False
    if selector.source is not None and artifact.get("source") != selector.source:
        return False
    if selector.target is not None:
        artifact_target = artifact.get("target")
        if not isinstance(artifact_target, str):
            return False
        if _normalize_target(artifact_target) != selector.target:
            return False
    if selector.variant is not None and artifact.get("variant") != selector.variant:
        return False
    if selector.stage is not None and artifact.get("stage") != selector.stage:
        return False
    if selector.path is not None:
        artifact_path = artifact.get("path", artifact.get("packagePath"))
        if artifact_path != selector.path:
            return False
    return True


def _artifact_failed(artifact: Mapping[str, Any]) -> bool:
    status = artifact.get("status")
    return isinstance(status, str) and status.strip().lower() in {
        "failed",
        TRANSLATION_FAILED,
    }


def _runtime_adapter_contract(
    fixture: RuntimeFixture, artifact: Mapping[str, Any]
) -> RuntimeAdapterContract:
    artifact_contract = _runtime_adapter_contract_from_artifact(artifact)
    if fixture.adapter_contract is None:
        return artifact_contract
    return _merge_runtime_adapter_contract(artifact_contract, fixture.adapter_contract)


def _runtime_adapter_contract_from_artifact(
    artifact: Mapping[str, Any],
) -> RuntimeAdapterContract:
    explicit_contract = _parse_runtime_adapter_contract(
        artifact.get("runtimeAdapter", artifact.get("adapterContract")),
        field_name="artifact.runtimeAdapter",
    )
    entry_points = tuple(
        _parse_runtime_entry_point(
            entry_point, field_name=f"artifact.entryPoints[{index}]"
        )
        for index, entry_point in enumerate(
            _runtime_record_sequence(artifact.get("entryPoints"))
        )
        if isinstance(entry_point, Mapping)
    )
    resource_bindings = tuple(
        _parse_runtime_resource_binding(
            binding, field_name=f"artifact.resourceBindings[{index}]"
        )
        for index, binding in enumerate(
            _runtime_record_sequence(artifact.get("resourceBindings"))
        )
        if isinstance(binding, Mapping)
    )
    artifact_payload = {
        "specializationConstants": artifact.get("specializationConstants", []),
        "functionConstants": artifact.get("functionConstants", []),
    }
    manifest_contract = RuntimeAdapterContract(
        entry_points=entry_points,
        resource_bindings=resource_bindings,
        specialization_constants=_parse_runtime_specialization_constants(
            artifact_payload, field_name="artifact"
        ),
        dispatch=_runtime_dispatch_geometry_from_artifact(artifact),
        validation_hooks=tuple(
            _parse_runtime_validation_hook(
                hook, field_name=f"artifact.validationHooks[{index}]"
            )
            for index, hook in enumerate(
                _runtime_record_sequence(artifact.get("validationHooks"))
            )
            if isinstance(hook, Mapping)
        ),
    )
    if explicit_contract is None:
        return manifest_contract
    return _merge_runtime_adapter_contract(manifest_contract, explicit_contract)


def _runtime_dispatch_geometry_from_artifact(
    artifact: Mapping[str, Any],
) -> RuntimeDispatchGeometry | None:
    dispatch = artifact.get("dispatch")
    if not isinstance(dispatch, Mapping):
        return None
    for index, workgroup in enumerate(
        _runtime_record_sequence(dispatch.get("workgroups"))
    ):
        if not isinstance(workgroup, Mapping):
            continue
        metadata = {}
        for field_name in ("stage", "source", "executionConfig"):
            if field_name in workgroup:
                metadata[field_name] = workgroup[field_name]
        if "status" in dispatch:
            metadata["status"] = dispatch["status"]
        return _parse_runtime_dispatch_geometry(
            {
                "entryPoint": workgroup.get("entryPoint"),
                "workgroupSize": workgroup.get("workgroupSize"),
                "metadata": metadata,
            },
            field_name=f"artifact.dispatch.workgroups[{index}]",
        )
    if any(
        key in dispatch
        for key in (
            "entryPoint",
            "workgroupSize",
            "workgroupCount",
            "globalSize",
            "gridSize",
        )
    ):
        return _parse_runtime_dispatch_geometry(
            dispatch, field_name="artifact.dispatch"
        )
    return None


def _merge_runtime_adapter_contract(
    base: RuntimeAdapterContract, override: RuntimeAdapterContract
) -> RuntimeAdapterContract:
    return RuntimeAdapterContract(
        contract_id=override.contract_id or base.contract_id,
        entry_points=_merge_runtime_contract_items(
            base.entry_points,
            override.entry_points,
            key=lambda item: (item.name, item.stage),
        ),
        resource_bindings=_merge_runtime_contract_items(
            base.resource_bindings,
            override.resource_bindings,
            key=_runtime_resource_binding_key,
        ),
        specialization_constants=_merge_runtime_contract_items(
            base.specialization_constants,
            override.specialization_constants,
            key=lambda item: item.name if item.name is not None else item.constant_id,
        ),
        dispatch=_merge_runtime_dispatch(base.dispatch, override.dispatch),
        validation_hooks=_merge_runtime_contract_items(
            base.validation_hooks,
            override.validation_hooks,
            key=lambda item: (item.phase, item.name),
        ),
        metadata={**dict(base.metadata), **dict(override.metadata)},
    )


def _merge_runtime_contract_items(
    base: tuple[Any, ...], override: tuple[Any, ...], *, key: Any
) -> tuple[Any, ...]:
    if not base:
        return override
    if not override:
        return base
    result = list(base)
    positions = {key(item): index for index, item in enumerate(result)}
    for item in override:
        item_key = key(item)
        if item_key in positions:
            result[positions[item_key]] = item
        else:
            positions[item_key] = len(result)
            result.append(item)
    return tuple(result)


def _runtime_resource_binding_key(binding: RuntimeResourceBinding) -> Any:
    if binding.binding_id is not None:
        return ("id", binding.binding_id)
    if binding.name is not None:
        return ("name", binding.name)
    return ("binding", binding.set, binding.binding, binding.index)


def _merge_runtime_dispatch(
    base: RuntimeDispatchGeometry | None, override: RuntimeDispatchGeometry | None
) -> RuntimeDispatchGeometry | None:
    if base is None:
        return override
    if override is None:
        return base
    return RuntimeDispatchGeometry(
        entry_point=override.entry_point or base.entry_point,
        workgroup_size=override.workgroup_size or base.workgroup_size,
        workgroup_count=override.workgroup_count or base.workgroup_count,
        global_size=override.global_size or base.global_size,
        grid_size=override.grid_size or base.grid_size,
        metadata={**dict(base.metadata), **dict(override.metadata)},
    )


def _runtime_record_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _runtime_artifact_path(
    artifact: Mapping[str, Any], *, root_path: Path | None
) -> Path | None:
    path_value = artifact.get("packagePath", artifact.get("path"))
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
    if not path.is_absolute() and root_path is not None:
        path = root_path / path
    return path.resolve()


def _runtime_execution_steps_from_artifact(
    artifact: Mapping[str, Any],
    *,
    artifact_path: Path | None,
    phase: str,
    keys: Sequence[str],
    fallback_action: str,
) -> tuple[RuntimeExecutionStep, ...]:
    steps: list[RuntimeExecutionStep] = []
    for key in keys:
        for index, record in enumerate(_runtime_record_sequence(artifact.get(key))):
            if not isinstance(record, Mapping):
                continue
            steps.append(
                _runtime_execution_step_from_record(
                    record,
                    phase=phase,
                    fallback_action=key,
                    artifact_path=artifact_path,
                    index=index,
                )
            )
    if steps:
        return tuple(steps)
    return (
        RuntimeExecutionStep(
            phase=phase,
            action=fallback_action,
            artifact_path=str(artifact_path) if artifact_path is not None else None,
            details=_runtime_step_fallback_details(artifact),
        ),
    )


def _runtime_execution_step_from_record(
    record: Mapping[str, Any],
    *,
    phase: str,
    fallback_action: str,
    artifact_path: Path | None,
    index: int,
) -> RuntimeExecutionStep:
    action = (
        _optional_runtime_step_text(record.get("action"))
        or _optional_runtime_step_text(record.get("kind"))
        or _optional_runtime_step_text(record.get("tool"))
        or fallback_action
    )
    command = record.get("command")
    command_items: tuple[str, ...] = ()
    if isinstance(command, Sequence) and not isinstance(
        command, (str, bytes, bytearray)
    ):
        command_items = tuple(str(item) for item in command)
    elif command is not None:
        command_items = (str(command),)
    details = {
        key: value
        for key, value in record.items()
        if key not in {"action", "kind", "tool", "command", "status", "diagnostics"}
    }
    details.setdefault("index", index)
    status = _optional_runtime_step_text(record.get("status")) or "planned"
    diagnostics = tuple(
        dict(diagnostic)
        for diagnostic in _runtime_record_sequence(record.get("diagnostics"))
        if isinstance(diagnostic, Mapping)
    )
    return RuntimeExecutionStep(
        phase=phase,
        action=action,
        status=status,
        command=command_items,
        artifact_path=str(artifact_path) if artifact_path is not None else None,
        details=details,
        diagnostics=diagnostics,
    )


def _optional_runtime_step_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _runtime_step_fallback_details(artifact: Mapping[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for key in ("id", "source", "target", "stage", "variant", "path", "packagePath"):
        if key in artifact:
            details[key] = artifact[key]
    return details


def _runtime_bound_resources(
    fixture: RuntimeFixture,
    contract: RuntimeAdapterContract,
    artifact: Mapping[str, Any],
) -> tuple[tuple[RuntimeBoundResource, ...], list[dict[str, Any]]]:
    values_by_name = _runtime_fixture_values_by_name(fixture)
    bindings: list[RuntimeBoundResource] = []
    diagnostics: list[dict[str, Any]] = []
    for binding in contract.resource_bindings:
        value, source = _runtime_resource_value(binding, values_by_name)
        if value is not None:
            bindings.append(
                RuntimeBoundResource(binding=binding, value=value, source=source)
            )
            continue
        diagnostic = _runtime_execution_diagnostic(
            "error" if _runtime_resource_required(binding) else "warning",
            "project.runtime-verification.resource-unbound",
            "Runtime fixture does not provide a value for a required resource binding.",
            artifact,
            resource=_runtime_resource_binding_name(binding),
            binding=binding.to_json(),
        )
        diagnostics.append(diagnostic)
        bindings.append(
            RuntimeBoundResource(
                binding=binding,
                status="missing",
                diagnostics=(diagnostic,),
            )
        )
    return tuple(bindings), diagnostics


def _runtime_fixture_values_by_name(
    fixture: RuntimeFixture,
) -> dict[str, tuple[RuntimeValue, str]]:
    values: dict[str, tuple[RuntimeValue, str]] = {}
    for value in fixture.inputs:
        values[value.name] = (value, "input")
    for value in fixture.expected_outputs:
        values.setdefault(value.name, (value, "expectedOutput"))
    return values


def _runtime_resource_value(
    binding: RuntimeResourceBinding,
    values_by_name: Mapping[str, tuple[RuntimeValue, str]],
) -> tuple[RuntimeValue | None, str | None]:
    candidate_names = []
    for value in (binding.value, binding.name, binding.binding_id):
        if isinstance(value, str) and value not in candidate_names:
            candidate_names.append(value)
    for candidate in candidate_names:
        match = values_by_name.get(candidate)
        if match is not None:
            return match
    return None, None


def _runtime_resource_required(binding: RuntimeResourceBinding) -> bool:
    required = binding.metadata.get("required")
    return required if isinstance(required, bool) else True


def _runtime_resource_binding_name(binding: RuntimeResourceBinding) -> str | None:
    if binding.name is not None:
        return binding.name
    if binding.binding_id is not None:
        return binding.binding_id
    if binding.binding is not None:
        return str(binding.binding)
    return None


def _runtime_bound_constants(
    contract: RuntimeAdapterContract,
    artifact: Mapping[str, Any],
) -> tuple[tuple[RuntimeBoundConstant, ...], list[dict[str, Any]]]:
    bindings: list[RuntimeBoundConstant] = []
    diagnostics: list[dict[str, Any]] = []
    for constant in contract.specialization_constants:
        if constant.value is not None:
            bindings.append(
                RuntimeBoundConstant(
                    constant=constant,
                    value=constant.value,
                    source="value",
                )
            )
            continue
        if constant.default is not None:
            bindings.append(
                RuntimeBoundConstant(
                    constant=constant,
                    value=constant.default,
                    source="default",
                )
            )
            continue
        if constant.required:
            diagnostic = _runtime_execution_diagnostic(
                "error",
                "project.runtime-verification.constant-unbound",
                "Runtime fixture does not provide a value for a required constant.",
                artifact,
                constant=_runtime_constant_name(constant),
            )
            diagnostics.append(diagnostic)
            bindings.append(
                RuntimeBoundConstant(
                    constant=constant,
                    status="missing",
                    diagnostics=(diagnostic,),
                )
            )
            continue
        bindings.append(
            RuntimeBoundConstant(
                constant=constant,
                source="optional",
                status="unbound",
            )
        )
    return tuple(bindings), diagnostics


def _runtime_constant_name(constant: RuntimeSpecializationConstant) -> Any:
    return constant.name if constant.name is not None else constant.constant_id


def _complete_runtime_dispatch_geometry(
    contract: RuntimeAdapterContract,
) -> RuntimeDispatchGeometry | None:
    dispatch = contract.dispatch
    entry_point = _runtime_dispatch_entry_point(contract, dispatch)
    if dispatch is None:
        if entry_point is None:
            return None
        dispatch = RuntimeDispatchGeometry(
            entry_point=entry_point.name,
            workgroup_size=entry_point.workgroup_size,
        )
    workgroup_size = dispatch.workgroup_size
    if not workgroup_size and entry_point is not None:
        workgroup_size = entry_point.workgroup_size
    global_size = dispatch.global_size or dispatch.grid_size
    workgroup_count = dispatch.workgroup_count
    if not workgroup_count and global_size and workgroup_size:
        workgroup_count = _runtime_workgroup_count(global_size, workgroup_size)
    if not global_size and workgroup_count and workgroup_size:
        global_size = _runtime_global_size(workgroup_count, workgroup_size)
    return RuntimeDispatchGeometry(
        entry_point=dispatch.entry_point or (entry_point.name if entry_point else None),
        workgroup_size=workgroup_size,
        workgroup_count=workgroup_count,
        global_size=global_size,
        grid_size=dispatch.grid_size,
        metadata=dispatch.metadata,
    )


def _runtime_dispatch_entry_point(
    contract: RuntimeAdapterContract, dispatch: RuntimeDispatchGeometry | None
) -> RuntimeEntryPoint | None:
    if not contract.entry_points:
        return None
    if dispatch is not None and dispatch.entry_point is not None:
        for entry_point in contract.entry_points:
            if entry_point.name == dispatch.entry_point:
                return entry_point
    return contract.entry_points[0]


def _runtime_workgroup_count(
    global_size: Sequence[Any], workgroup_size: Sequence[Any]
) -> tuple[int, ...]:
    counts: list[int] = []
    for global_item, workgroup_item in zip(global_size, workgroup_size):
        if not _is_positive_int_like(global_item) or not _is_positive_int_like(
            workgroup_item
        ):
            return ()
        counts.append(math.ceil(int(global_item) / int(workgroup_item)))
    return tuple(counts)


def _runtime_global_size(
    workgroup_count: Sequence[Any], workgroup_size: Sequence[Any]
) -> tuple[int, ...]:
    sizes: list[int] = []
    for count_item, workgroup_item in zip(workgroup_count, workgroup_size):
        if not _is_positive_int_like(count_item) or not _is_positive_int_like(
            workgroup_item
        ):
            return ()
        sizes.append(int(count_item) * int(workgroup_item))
    return tuple(sizes)


def _is_positive_int_like(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _runtime_value_reference(value: RuntimeValue) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": value.name, "kind": value.kind}
    if value.dtype is not None:
        payload["dtype"] = value.dtype
    shape = value.shape or _infer_shape(value.values)
    if shape:
        payload["shape"] = list(shape)
    if value.metadata:
        payload["metadata"] = dict(value.metadata)
    return payload


def _runtime_diagnostics_have_errors(
    diagnostics: Sequence[Mapping[str, Any]],
) -> bool:
    return any(diagnostic.get("severity") == "error" for diagnostic in diagnostics)


def _runtime_comparison_diagnostics(
    comparisons: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for comparison in comparisons:
        if comparison.get("status") == PASSED:
            continue
        diagnostics.append(
            _runtime_execution_diagnostic(
                "error",
                "project.runtime-verification.output-mismatch",
                str(comparison.get("message") or "Runtime output comparison failed."),
                artifact,
                output=comparison.get("name"),
                comparison={
                    key: comparison[key]
                    for key in (
                        "name",
                        "kind",
                        "mismatchCount",
                        "maxAbsoluteError",
                        "maxRelativeError",
                    )
                    if key in comparison
                },
            )
        )
    return diagnostics


def _runtime_execution_diagnostic(
    severity: str,
    code: str,
    message: str,
    artifact: Mapping[str, Any],
    **context: Any,
) -> dict[str, Any]:
    return _diagnostic(
        severity,
        code,
        message,
        artifact=_artifact_result_payload(artifact),
        **_runtime_artifact_span_context(artifact),
        **context,
    )


def _runtime_artifact_span_context(artifact: Mapping[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    source_span = artifact.get("sourceSpan")
    generated_span = artifact.get("generatedSpan")
    source_map = artifact.get("sourceMap")
    if isinstance(source_map, Mapping):
        source_span = source_span or source_map.get("source")
        generated_span = generated_span or source_map.get("generated")
    if isinstance(source_span, Mapping):
        context["sourceSpan"] = dict(source_span)
        context["location"] = dict(source_span)
    if isinstance(generated_span, Mapping):
        context["generatedSpan"] = dict(generated_span)
    return context


def _executor_availability(
    executor: Any, request: RuntimeExecutionRequest
) -> RuntimeExecutorAvailability:
    probe = getattr(executor, "is_available", None)
    if probe is None:
        return RuntimeExecutorAvailability(True)
    value = probe(request)
    if isinstance(value, RuntimeExecutorAvailability):
        return value
    if isinstance(value, bool):
        return RuntimeExecutorAvailability(value)
    if isinstance(value, Mapping):
        return RuntimeExecutorAvailability(
            bool(value.get("available")),
            reason=(
                value.get("reason") if isinstance(value.get("reason"), str) else None
            ),
            details={
                key: item
                for key, item in value.items()
                if key not in {"available", "reason"}
            },
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
        if values:
            reason = (
                values[1] if len(values) > 1 and isinstance(values[1], str) else None
            )
            return RuntimeExecutorAvailability(bool(values[0]), reason=reason)
    raise RuntimeVerificationError("Executor availability probe returned invalid data.")


def _runtime_executor_status_diagnostics(
    status: str, *, target: str, execution: RuntimeExecutorResult
) -> list[dict[str, Any]]:
    if not execution.message and not execution.details:
        return []
    code = (
        "project.runtime-verification.fixture-skipped"
        if status == SKIPPED
        else "project.runtime-verification.executor-unavailable"
    )
    message = execution.message or (
        "Runtime fixture was skipped."
        if status == SKIPPED
        else "Runtime executor is unavailable."
    )
    return [
        _diagnostic(
            "note",
            code,
            message,
            target=target,
            executorStatus=status,
            executorDetails=dict(execution.details) if execution.details else None,
        )
    ]


def _normalize_executor_result(value: Any) -> RuntimeExecutorResult:
    if isinstance(value, RuntimeExecutorResult):
        return value
    if isinstance(value, Mapping):
        return RuntimeExecutorResult(
            status=str(value.get("status", "ok")),
            outputs=value.get("outputs", {}),
            message=(
                value.get("message") if isinstance(value.get("message"), str) else None
            ),
            details=(
                dict(value.get("details"))
                if isinstance(value.get("details"), Mapping)
                else {}
            ),
        )
    return RuntimeExecutorResult(outputs=value)


def _runtime_values_by_name(outputs: Any) -> dict[str, RuntimeValue]:
    if outputs is None:
        return {}
    if isinstance(outputs, RuntimeValue):
        return {outputs.name: outputs}
    if isinstance(outputs, Mapping):
        if "name" in outputs and "values" in outputs:
            value = _parse_runtime_value(outputs, field_name="actualOutputs")
            return {value.name: value}
        result: dict[str, RuntimeValue] = {}
        for name, value in outputs.items():
            if isinstance(value, RuntimeValue):
                result[value.name] = value
                continue
            if isinstance(value, Mapping):
                payload = dict(value)
                payload.setdefault("name", str(name))
                parsed = _parse_runtime_value(
                    payload, field_name=f"actualOutputs.{name}"
                )
            else:
                parsed = RuntimeValue(name=str(name), values=value)
            result[parsed.name] = parsed
        return result
    if isinstance(outputs, Sequence) and not isinstance(
        outputs, (str, bytes, bytearray)
    ):
        result = {}
        for index, output in enumerate(outputs):
            value = (
                output
                if isinstance(output, RuntimeValue)
                else _parse_runtime_value(output, field_name=f"actualOutputs[{index}]")
            )
            result[value.name] = value
        return result
    raise RuntimeVerificationError("Executor outputs must be a mapping or list.")


def _compare_runtime_value(
    expected: RuntimeValue,
    actual: RuntimeValue | None,
    *,
    default_tolerance: RuntimeTolerance,
) -> dict[str, Any]:
    tolerance = expected.tolerance or default_tolerance
    comparison = {
        "name": expected.name,
        "kind": expected.kind,
        "status": PASSED,
        "tolerance": tolerance.to_json(),
        "expected": _value_metadata(expected),
        "actual": _value_metadata(actual) if actual is not None else None,
        "mismatchCount": 0,
        "maxAbsoluteError": 0.0,
        "maxRelativeError": 0.0,
    }
    if actual is None:
        comparison.update(
            {
                "status": COMPARISON_FAILED,
                "message": "Expected output was not produced.",
                "mismatchCount": 1,
            }
        )
        return comparison
    if (
        expected.dtype is not None
        and actual.dtype is not None
        and expected.dtype != actual.dtype
    ):
        comparison.update(
            {
                "status": COMPARISON_FAILED,
                "message": "Output dtype mismatch.",
                "mismatchCount": 1,
            }
        )
        return comparison

    expected_shape = expected.shape or _infer_shape(expected.values)
    actual_shape = actual.shape or _infer_shape(actual.values)
    if expected_shape and actual_shape and expected_shape != actual_shape:
        comparison.update(
            {
                "status": COMPARISON_FAILED,
                "message": "Output shape mismatch.",
                "expected": {**comparison["expected"], "shape": list(expected_shape)},
                "actual": {**comparison["actual"], "shape": list(actual_shape)},
                "mismatchCount": 1,
            }
        )
        return comparison

    expected_flat = _flatten_values(expected.values)
    actual_flat = _flatten_values(actual.values)
    if len(expected_flat) != len(actual_flat):
        comparison.update(
            {
                "status": COMPARISON_FAILED,
                "message": "Output value count mismatch.",
                "mismatchCount": abs(len(expected_flat) - len(actual_flat)) or 1,
            }
        )
        return comparison

    mismatch_count = 0
    first_mismatch = None
    max_absolute = 0.0
    max_relative = 0.0
    for index, (expected_item, actual_item) in enumerate(
        zip(expected_flat, actual_flat)
    ):
        matches, absolute_error, relative_error = _values_match(
            expected_item, actual_item, tolerance
        )
        max_absolute = max(max_absolute, absolute_error)
        if not math.isinf(relative_error):
            max_relative = max(max_relative, relative_error)
        elif not math.isinf(max_relative):
            max_relative = math.inf
        if not matches:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = {
                    "index": index,
                    "expected": expected_item,
                    "actual": actual_item,
                    "absoluteError": _json_metric(absolute_error),
                    "relativeError": _json_metric(relative_error),
                }
    comparison["mismatchCount"] = mismatch_count
    comparison["maxAbsoluteError"] = _json_metric(max_absolute)
    comparison["maxRelativeError"] = _json_metric(max_relative)
    if mismatch_count:
        comparison["status"] = COMPARISON_FAILED
        comparison["message"] = "Output values differ beyond tolerance."
        comparison["firstMismatch"] = first_mismatch
    return comparison


def _value_metadata(value: RuntimeValue | None) -> dict[str, Any]:
    if value is None:
        return {}
    payload: dict[str, Any] = {}
    if value.dtype is not None:
        payload["dtype"] = value.dtype
    shape = value.shape or _infer_shape(value.values)
    if shape:
        payload["shape"] = list(shape)
    return payload


def _infer_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        length = len(value)
        if length == 0:
            return (0,)
        child_shapes = [_infer_shape(item) for item in value]
        first = child_shapes[0]
        if all(shape == first for shape in child_shapes):
            return (length,) + first
        return (length,)
    return ()


def _flatten_values(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_values(item))
        return flattened
    return [value]


def _values_match(
    expected: Any, actual: Any, tolerance: RuntimeTolerance
) -> tuple[bool, float, float]:
    if _is_number(expected) and _is_number(actual):
        expected_number = float(expected)
        actual_number = float(actual)
        if math.isnan(expected_number) or math.isnan(actual_number):
            matches = math.isnan(expected_number) and math.isnan(actual_number)
            return matches, 0.0 if matches else math.inf, 0.0 if matches else math.inf
        if math.isinf(expected_number) or math.isinf(actual_number):
            matches = expected_number == actual_number
            return matches, 0.0 if matches else math.inf, 0.0 if matches else math.inf
        absolute_error = abs(actual_number - expected_number)
        relative_error = (
            absolute_error / abs(expected_number)
            if expected_number != 0
            else (0.0 if absolute_error == 0 else math.inf)
        )
        allowed = tolerance.absolute + tolerance.relative * abs(expected_number)
        return absolute_error <= allowed, absolute_error, relative_error
    return expected == actual, 0.0 if expected == actual else 1.0, 0.0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _json_metric(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _runtime_verification_summary(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    statuses = Counter(result.get("status") for result in results)
    runtime_failed = statuses[RUNTIME_FAILED]
    translation_failed = statuses[TRANSLATION_FAILED]
    comparison_failed = statuses[COMPARISON_FAILED]
    return {
        "fixtureCount": len(results),
        "passedCount": statuses[PASSED],
        "skippedCount": statuses[SKIPPED],
        "unavailableCount": statuses[UNAVAILABLE],
        "translationFailedCount": translation_failed,
        "runtimeFailedCount": runtime_failed,
        "comparisonFailedCount": comparison_failed,
        "failedCount": runtime_failed + translation_failed + comparison_failed,
    }


def _fixture_result(
    fixture: RuntimeFixture,
    *,
    status: str,
    failure_phase: str | None = None,
    artifact: Mapping[str, Any] | None = None,
    adapter_contract: RuntimeAdapterContract | None = None,
    runtime_execution: RuntimeExecutionPlan | None = None,
    executor: Mapping[str, Any] | None = None,
    comparisons: Sequence[Mapping[str, Any]] = (),
    diagnostics: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    result = {
        "fixture": fixture.id,
        "selector": fixture.selector.to_json(),
        "status": status,
        "artifact": _artifact_result_payload(artifact),
        "executor": dict(executor) if isinstance(executor, Mapping) else None,
        "comparisons": [dict(comparison) for comparison in comparisons],
        "diagnostics": [dict(diagnostic) for diagnostic in diagnostics],
    }
    if adapter_contract is not None:
        adapter_payload = adapter_contract.to_json()
        if adapter_payload:
            result["runtimeAdapter"] = adapter_payload
    if runtime_execution is not None:
        result["runtimeExecution"] = runtime_execution.to_json()
    if failure_phase is not None:
        result["failurePhase"] = failure_phase
    return result


def _artifact_result_payload(
    artifact: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if artifact is None:
        return None
    payload = {}
    for field_name in (
        "id",
        "source",
        "path",
        "packagePath",
        "target",
        "sourceBackend",
        "stage",
        "variant",
        "status",
        "error",
        "toolchain",
        "toolchainRuns",
    ):
        if field_name in artifact:
            payload[field_name] = artifact[field_name]
    return payload


def _diagnostic(
    severity: str, code: str, message: str, **context: Any
) -> dict[str, Any]:
    diagnostic = {"severity": severity, "code": code, "message": message}
    for key, value in context.items():
        if value is not None:
            diagnostic[key] = value
    return diagnostic
