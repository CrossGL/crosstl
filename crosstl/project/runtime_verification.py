"""Runtime verification harness for translated project artifacts."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from crosstl.translator.codegen import normalize_backend_name


RUNTIME_VERIFICATION_FIXTURES_KIND = "crosstl-runtime-verification-fixtures"
RUNTIME_VERIFICATION_REPORT_KIND = "crosstl-runtime-verification-report"
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
class RuntimeFixture:
    """A runtime verification fixture bound to one translated artifact."""

    id: str
    selector: RuntimeArtifactSelector
    inputs: tuple[RuntimeValue, ...] = field(default_factory=tuple)
    expected_outputs: tuple[RuntimeValue, ...] = field(default_factory=tuple)
    default_tolerance: RuntimeTolerance = field(default_factory=RuntimeTolerance)
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

    tolerance = _parse_tolerance(
        default_tolerance, field_name="defaultTolerance"
    )
    expected = [
        value
        if isinstance(value, RuntimeValue)
        else _parse_runtime_value(value, field_name=f"expectedOutputs[{index}]")
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
    kind = _optional_string(value.get("kind"), field_name=f"{field_name}.kind") or "buffer"
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


def _normalize_target(target: str) -> str:
    return normalize_backend_name(target) or target.strip().lower()


def _load_artifact_payload(
    artifact_report: str | os.PathLike[str] | Mapping[str, Any]
) -> tuple[Mapping[str, Any], Path | None]:
    if isinstance(artifact_report, Mapping):
        return artifact_report, None
    path = _filesystem_path_arg(artifact_report, field_name="Artifact report path")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeVerificationError(f"Artifact report could not be read: {exc}") from exc
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
        raise RuntimeVerificationError("Runtime fixtures must be a fixture path or list.")
    parsed = [
        fixture
        if isinstance(fixture, RuntimeFixture)
        else _parse_runtime_fixture(fixture, index=index)
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
                raise RuntimeVerificationError("Executor registry keys must be strings.")
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
            diagnostics=selected.get("diagnostics", []),
        )

    artifact = selected["artifact"]
    artifact_path = _runtime_artifact_path(artifact, root_path=root_path)
    target = _normalize_target(str(artifact.get("target", fixture.selector.target or "")))
    executor_key = _normalize_target(fixture.executor) if fixture.executor else target
    executor = executors.get(executor_key)
    if executor is None:
        return _fixture_result(
            fixture,
            status=UNAVAILABLE,
            failure_phase="runtime",
            artifact=artifact,
            executor={"key": executor_key, "status": UNAVAILABLE},
            diagnostics=[
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
    )
    executor_name = getattr(executor, "name", executor.__class__.__name__)
    availability = _executor_availability(executor, request)
    if not availability.available:
        return _fixture_result(
            fixture,
            status=UNAVAILABLE,
            failure_phase="runtime",
            artifact=artifact,
            executor={
                "key": executor_key,
                "name": executor_name,
                "status": UNAVAILABLE,
                "message": availability.reason,
                "details": dict(availability.details),
            },
            diagnostics=[
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
            executor={"key": executor_key, "name": executor_name, "status": UNAVAILABLE},
            diagnostics=[
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
            executor={"key": executor_key, "name": executor_name, "status": SKIPPED},
            diagnostics=[
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
            executor={
                "key": executor_key,
                "name": executor_name,
                "status": RUNTIME_FAILED,
            },
            diagnostics=[
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
            executor={
                "key": executor_key,
                "name": executor_name,
                "status": RUNTIME_FAILED,
            },
            diagnostics=[
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
            executor=executor_payload,
        )
    if normalized_status in EXECUTOR_UNAVAILABLE_STATUSES:
        return _fixture_result(
            fixture,
            status=UNAVAILABLE,
            failure_phase="runtime",
            artifact=artifact,
            executor=executor_payload,
        )
    if normalized_status in EXECUTOR_FAILED_STATUSES:
        return _fixture_result(
            fixture,
            status=RUNTIME_FAILED,
            failure_phase="runtime",
            artifact=artifact,
            executor=executor_payload,
        )
    if normalized_status not in EXECUTOR_OK_STATUSES:
        return _fixture_result(
            fixture,
            status=RUNTIME_FAILED,
            failure_phase="runtime",
            artifact=artifact,
            executor=executor_payload,
            diagnostics=[
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
    comparison_failed = any(comparison["status"] != PASSED for comparison in comparisons)
    return _fixture_result(
        fixture,
        status=COMPARISON_FAILED if comparison_failed else PASSED,
        failure_phase="comparison" if comparison_failed else None,
        artifact=artifact,
        executor=executor_payload,
        comparisons=comparisons,
    )


def _select_runtime_artifact(
    payload: Mapping[str, Any], selector: RuntimeArtifactSelector
) -> dict[str, Any]:
    artifacts = [
        artifact
        for artifact in payload.get("artifacts", [])
        if isinstance(artifact, Mapping) and _selector_matches_artifact(selector, artifact)
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
            reason=value.get("reason") if isinstance(value.get("reason"), str) else None,
            details={
                key: item
                for key, item in value.items()
                if key not in {"available", "reason"}
            },
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
        if values:
            reason = values[1] if len(values) > 1 and isinstance(values[1], str) else None
            return RuntimeExecutorAvailability(bool(values[0]), reason=reason)
    raise RuntimeVerificationError("Executor availability probe returned invalid data.")


def _normalize_executor_result(value: Any) -> RuntimeExecutorResult:
    if isinstance(value, RuntimeExecutorResult):
        return value
    if isinstance(value, Mapping):
        return RuntimeExecutorResult(
            status=str(value.get("status", "ok")),
            outputs=value.get("outputs", {}),
            message=value.get("message") if isinstance(value.get("message"), str) else None,
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
                parsed = _parse_runtime_value(payload, field_name=f"actualOutputs.{name}")
            else:
                parsed = RuntimeValue(name=str(name), values=value)
            result[parsed.name] = parsed
        return result
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes, bytearray)):
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
    for index, (expected_item, actual_item) in enumerate(zip(expected_flat, actual_flat)):
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


def _runtime_verification_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
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
    if failure_phase is not None:
        result["failurePhase"] = failure_phase
    return result


def _artifact_result_payload(artifact: Mapping[str, Any] | None) -> dict[str, Any] | None:
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
