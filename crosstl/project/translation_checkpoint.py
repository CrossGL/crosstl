"""Durable progress records for repository translation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_TRANSLATION_CHECKPOINT_KIND = "crosstl-project-translation-checkpoint"
PROJECT_TRANSLATION_CHECKPOINT_VERSION = 1

_ERROR_PREFIX = "project.translation-checkpoint"
_STATES = frozenset(("running", "interrupted", "complete"))
_CHECKPOINT_FIELDS = frozenset(
    (
        "schemaVersion",
        "kind",
        "state",
        "startedAt",
        "updatedAt",
        "completedAt",
        "projectIdentity",
        "plan",
        "interruption",
        "finalReport",
        "checkpointHash",
    )
)
_PLAN_FIELDS = frozenset(
    (
        "jobCount",
        "completedCount",
        "pendingCount",
        "jobs",
        "completed",
        "active",
        "pending",
    )
)
_COORDINATE_FIELDS = frozenset(("source", "target", "path", "variant", "entryPoint"))
_COMPLETION_FIELDS = frozenset(("coordinate", "artifacts", "diagnostics"))
_INTERRUPTION_FIELDS = frozenset(("type", "message"))
_HASH_FIELDS = frozenset(("algorithm", "value"))


class ProjectTranslationCheckpointError(ValueError):
    """Raised when project translation checkpoint data is invalid."""

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        path: str = "$",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason = reason
        self.code = f"{_ERROR_PREFIX}.{reason}"
        self.path = path
        self.message = message
        self.details = copy.deepcopy(dict(details or {}))
        super().__init__(f"{path}: {message} ({self.code})")

    def to_json(self) -> dict[str, Any]:
        payload = {
            "severity": "error",
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }
        if self.details:
            payload["details"] = copy.deepcopy(self.details)
        return payload


class ProjectTranslationCheckpointRecorder:
    """Atomically persist translation progress for one deterministic job plan."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        project_identity: Mapping[str, Any],
        jobs: Sequence[Mapping[str, Any]],
        *,
        started_at: int | None = None,
        completed: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.path = _checkpoint_path(path)
        self.project_identity = _json_mapping(
            project_identity,
            path="$.projectIdentity",
        )
        self.jobs = tuple(
            _coordinate(job, path=f"$.plan.jobs[{index}]")
            for index, job in enumerate(jobs)
        )
        _require_unique_coordinates(self.jobs, path="$.plan.jobs")
        self.started_at = _timestamp(
            int(time.time()) if started_at is None else started_at,
            path="$.startedAt",
        )
        self._completed: dict[str, dict[str, Any]] = {}
        for index, record in enumerate(completed):
            normalized = _completion_record(
                record,
                path=f"$.plan.completed[{index}]",
            )
            key = _coordinate_key(normalized["coordinate"])
            if key in self._completed:
                raise ProjectTranslationCheckpointError(
                    "completed-duplicate",
                    "Completed translation coordinates must be unique.",
                    path=f"$.plan.completed[{index}].coordinate",
                )
            self._completed[key] = normalized
        _require_completed_coordinates_in_plan(self.jobs, self._completed)

    @classmethod
    def resume(
        cls,
        path: str | os.PathLike[str],
        project_identity: Mapping[str, Any],
        jobs: Sequence[Mapping[str, Any]],
    ) -> ProjectTranslationCheckpointRecorder:
        checkpoint = load_project_translation_checkpoint(path)
        normalized_identity = _json_mapping(
            project_identity,
            path="$.projectIdentity",
        )
        normalized_jobs = [
            _coordinate(job, path=f"$.plan.jobs[{index}]")
            for index, job in enumerate(jobs)
        ]
        if checkpoint["projectIdentity"] != normalized_identity:
            raise ProjectTranslationCheckpointError(
                "project-identity-mismatch",
                "Checkpoint project identity does not match this translation.",
                path="$.projectIdentity",
                details={
                    "expected": normalized_identity,
                    "actual": checkpoint["projectIdentity"],
                },
            )
        if checkpoint["plan"]["jobs"] != normalized_jobs:
            raise ProjectTranslationCheckpointError(
                "job-plan-mismatch",
                "Checkpoint translation jobs do not match this translation.",
                path="$.plan.jobs",
                details={
                    "expected": normalized_jobs,
                    "actual": checkpoint["plan"]["jobs"],
                },
            )
        if checkpoint["state"] == "complete":
            raise ProjectTranslationCheckpointError(
                "already-complete",
                "Checkpoint already records a complete translation.",
                path="$.state",
            )
        return cls(
            path,
            normalized_identity,
            normalized_jobs,
            started_at=checkpoint["startedAt"],
            completed=checkpoint["plan"]["completed"],
        )

    @property
    def completed(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            copy.deepcopy(self._completed[key])
            for key in self._ordered_completed_keys()
        )

    def completion_for(self, coordinate: Mapping[str, Any]) -> dict[str, Any] | None:
        key = _coordinate_key(
            _coordinate(coordinate, path="$.plan.completed[].coordinate")
        )
        record = self._completed.get(key)
        return copy.deepcopy(record) if record is not None else None

    def write_running(
        self,
        active: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._write(state="running", active=active)

    def record_completion(
        self,
        coordinate: Mapping[str, Any],
        artifacts: Sequence[Mapping[str, Any]],
        diagnostics: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        normalized = _completion_record(
            {
                "coordinate": coordinate,
                "artifacts": artifacts,
                "diagnostics": diagnostics,
            },
            path="$.plan.completed[]",
        )
        key = _coordinate_key(normalized["coordinate"])
        planned_keys = {_coordinate_key(job) for job in self.jobs}
        if key not in planned_keys:
            raise ProjectTranslationCheckpointError(
                "completed-not-planned",
                "Completed translation coordinate is not present in the job plan.",
                path="$.plan.completed[].coordinate",
            )
        if key in self._completed:
            raise ProjectTranslationCheckpointError(
                "completed-duplicate",
                "Translation coordinate is already recorded as complete.",
                path="$.plan.completed[].coordinate",
            )
        self._completed[key] = normalized
        return self._write(state="running")

    def write_interrupted(
        self,
        active: Mapping[str, Any] | None,
        error: BaseException,
    ) -> dict[str, Any]:
        return self._write(
            state="interrupted",
            active=active,
            interruption={
                "type": type(error).__name__,
                "message": str(error),
            },
        )

    def write_complete(self, final_report: Mapping[str, Any]) -> dict[str, Any]:
        if len(self._completed) != len(self.jobs):
            raise ProjectTranslationCheckpointError(
                "completion-incomplete",
                "A complete checkpoint requires every planned job to be complete.",
                path="$.plan.completed",
                details={
                    "jobCount": len(self.jobs),
                    "completedCount": len(self._completed),
                },
            )
        return self._write(
            state="complete",
            final_report=_json_mapping(final_report, path="$.finalReport"),
        )

    def _write(
        self,
        *,
        state: str,
        active: Mapping[str, Any] | None = None,
        interruption: Mapping[str, Any] | None = None,
        final_report: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_active = (
            _coordinate(active, path="$.plan.active") if active is not None else None
        )
        completed_keys = set(self._completed)
        if normalized_active is not None:
            active_key = _coordinate_key(normalized_active)
            planned_keys = {_coordinate_key(job) for job in self.jobs}
            if active_key not in planned_keys:
                raise ProjectTranslationCheckpointError(
                    "active-not-planned",
                    "Active translation coordinate is not present in the job plan.",
                    path="$.plan.active",
                )
            if active_key in completed_keys:
                raise ProjectTranslationCheckpointError(
                    "active-completed",
                    "Active translation coordinate is already complete.",
                    path="$.plan.active",
                )
        else:
            active_key = None
        pending = [
            copy.deepcopy(job)
            for job in self.jobs
            if _coordinate_key(job) not in completed_keys
            and _coordinate_key(job) != active_key
        ]
        updated_at = int(time.time())
        payload: dict[str, Any] = {
            "schemaVersion": PROJECT_TRANSLATION_CHECKPOINT_VERSION,
            "kind": PROJECT_TRANSLATION_CHECKPOINT_KIND,
            "state": state,
            "startedAt": self.started_at,
            "updatedAt": updated_at,
            "completedAt": updated_at if state == "complete" else None,
            "projectIdentity": copy.deepcopy(self.project_identity),
            "plan": {
                "jobCount": len(self.jobs),
                "completedCount": len(self._completed),
                "pendingCount": len(pending),
                "jobs": [copy.deepcopy(job) for job in self.jobs],
                "completed": [
                    copy.deepcopy(self._completed[key])
                    for key in self._ordered_completed_keys()
                ],
                "active": normalized_active,
                "pending": pending,
            },
            "interruption": (
                _json_mapping(interruption, path="$.interruption")
                if interruption is not None
                else None
            ),
            "finalReport": (
                _json_mapping(final_report, path="$.finalReport")
                if final_report is not None
                else None
            ),
        }
        payload["checkpointHash"] = _payload_hash(payload)
        normalized = validate_project_translation_checkpoint(payload)
        _atomic_write_json(self.path, normalized)
        return normalized

    def _ordered_completed_keys(self) -> tuple[str, ...]:
        return tuple(
            key
            for key in (_coordinate_key(job) for job in self.jobs)
            if key in self._completed
        )


def load_project_translation_checkpoint(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    checkpoint_path = _checkpoint_path(path)
    try:
        value = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProjectTranslationCheckpointError(
            "read-failed",
            f"Project translation checkpoint could not be read: {exc}",
            path="$",
            details={"checkpointPath": str(checkpoint_path)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise ProjectTranslationCheckpointError(
            "json-invalid",
            f"Project translation checkpoint is not valid JSON: {exc}",
            path="$",
            details={"checkpointPath": str(checkpoint_path)},
        ) from exc
    return validate_project_translation_checkpoint(value)


def validate_project_translation_checkpoint(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectTranslationCheckpointError(
            "invalid",
            "Project translation checkpoint must be a JSON object.",
        )
    checkpoint = copy.deepcopy(dict(value))
    _require_fields(checkpoint, _CHECKPOINT_FIELDS, path="$")
    if checkpoint.get("schemaVersion") != PROJECT_TRANSLATION_CHECKPOINT_VERSION:
        raise ProjectTranslationCheckpointError(
            "version-unsupported",
            "Project translation checkpoint schema version is unsupported.",
            path="$.schemaVersion",
            details={
                "expected": PROJECT_TRANSLATION_CHECKPOINT_VERSION,
                "actual": checkpoint.get("schemaVersion"),
            },
        )
    if checkpoint.get("kind") != PROJECT_TRANSLATION_CHECKPOINT_KIND:
        raise ProjectTranslationCheckpointError(
            "kind-invalid",
            "Project translation checkpoint kind is invalid.",
            path="$.kind",
        )
    state = checkpoint.get("state")
    if state not in _STATES:
        raise ProjectTranslationCheckpointError(
            "state-invalid",
            "Project translation checkpoint state is invalid.",
            path="$.state",
            details={"state": state},
        )
    started_at = _timestamp(checkpoint.get("startedAt"), path="$.startedAt")
    updated_at = _timestamp(checkpoint.get("updatedAt"), path="$.updatedAt")
    if updated_at < started_at:
        raise ProjectTranslationCheckpointError(
            "timestamp-invalid",
            "Checkpoint update time cannot precede its start time.",
            path="$.updatedAt",
        )
    completed_at = checkpoint.get("completedAt")
    if completed_at is not None:
        completed_at = _timestamp(completed_at, path="$.completedAt")
        if completed_at < updated_at:
            raise ProjectTranslationCheckpointError(
                "timestamp-invalid",
                "Checkpoint completion time cannot precede its update time.",
                path="$.completedAt",
            )
    checkpoint["projectIdentity"] = _json_mapping(
        checkpoint.get("projectIdentity"),
        path="$.projectIdentity",
    )
    checkpoint["plan"] = _validate_plan(checkpoint.get("plan"))
    interruption = checkpoint.get("interruption")
    final_report = checkpoint.get("finalReport")
    if state == "running":
        if (
            interruption is not None
            or final_report is not None
            or completed_at is not None
        ):
            raise ProjectTranslationCheckpointError(
                "state-payload-invalid",
                "Running checkpoints cannot contain interruption or final data.",
                path="$.state",
            )
    elif state == "interrupted":
        _validate_interruption(interruption)
        if final_report is not None or completed_at is not None:
            raise ProjectTranslationCheckpointError(
                "state-payload-invalid",
                "Interrupted checkpoints cannot contain final data.",
                path="$.state",
            )
    else:
        if interruption is not None:
            raise ProjectTranslationCheckpointError(
                "state-payload-invalid",
                "Complete checkpoints cannot contain interruption data.",
                path="$.state",
            )
        _json_mapping(final_report, path="$.finalReport")
        if completed_at is None:
            raise ProjectTranslationCheckpointError(
                "state-payload-invalid",
                "Complete checkpoints require a completion time.",
                path="$.completedAt",
            )
        plan = checkpoint["plan"]
        if (
            plan["active"] is not None
            or plan["pending"]
            or plan["completedCount"] != plan["jobCount"]
        ):
            raise ProjectTranslationCheckpointError(
                "completion-incomplete",
                "Complete checkpoints require every planned job to be complete.",
                path="$.plan",
            )
    expected_hash = _payload_hash(
        {key: item for key, item in checkpoint.items() if key != "checkpointHash"}
    )
    observed_hash = checkpoint.get("checkpointHash")
    _validate_hash(observed_hash, path="$.checkpointHash")
    if observed_hash != expected_hash:
        raise ProjectTranslationCheckpointError(
            "hash-mismatch",
            "Project translation checkpoint hash does not match its payload.",
            path="$.checkpointHash",
            details={"expected": expected_hash, "actual": observed_hash},
        )
    return checkpoint


def _validate_plan(value: Any) -> dict[str, Any]:
    plan = _json_mapping(value, path="$.plan")
    _require_fields(plan, _PLAN_FIELDS, path="$.plan")
    jobs_value = plan.get("jobs")
    completed_value = plan.get("completed")
    pending_value = plan.get("pending")
    if not isinstance(jobs_value, list):
        _invalid("Plan jobs must be a list.", path="$.plan.jobs")
    if not isinstance(completed_value, list):
        _invalid("Completed jobs must be a list.", path="$.plan.completed")
    if not isinstance(pending_value, list):
        _invalid("Pending jobs must be a list.", path="$.plan.pending")
    jobs = [
        _coordinate(job, path=f"$.plan.jobs[{index}]")
        for index, job in enumerate(jobs_value)
    ]
    _require_unique_coordinates(jobs, path="$.plan.jobs")
    completed = [
        _completion_record(record, path=f"$.plan.completed[{index}]")
        for index, record in enumerate(completed_value)
    ]
    completed_by_key: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(completed):
        key = _coordinate_key(record["coordinate"])
        if key in completed_by_key:
            raise ProjectTranslationCheckpointError(
                "completed-duplicate",
                "Completed translation coordinates must be unique.",
                path=f"$.plan.completed[{index}].coordinate",
            )
        completed_by_key[key] = record
    _require_completed_coordinates_in_plan(jobs, completed_by_key)
    active_value = plan.get("active")
    active = (
        _coordinate(active_value, path="$.plan.active")
        if active_value is not None
        else None
    )
    job_keys = {_coordinate_key(job) for job in jobs}
    completed_keys = set(completed_by_key)
    if active is not None:
        active_key = _coordinate_key(active)
        if active_key not in job_keys:
            raise ProjectTranslationCheckpointError(
                "active-not-planned",
                "Active translation coordinate is not present in the job plan.",
                path="$.plan.active",
            )
        if active_key in completed_keys:
            raise ProjectTranslationCheckpointError(
                "active-completed",
                "Active translation coordinate is already complete.",
                path="$.plan.active",
            )
    else:
        active_key = None
    pending = [
        _coordinate(job, path=f"$.plan.pending[{index}]")
        for index, job in enumerate(pending_value)
    ]
    expected_pending = [
        job
        for job in jobs
        if _coordinate_key(job) not in completed_keys
        and _coordinate_key(job) != active_key
    ]
    if pending != expected_pending:
        raise ProjectTranslationCheckpointError(
            "pending-mismatch",
            "Pending translation coordinates do not match the job plan state.",
            path="$.plan.pending",
            details={"expected": expected_pending, "actual": pending},
        )
    expected_counts = {
        "jobCount": len(jobs),
        "completedCount": len(completed),
        "pendingCount": len(pending),
    }
    for field_name, expected in expected_counts.items():
        if plan.get(field_name) != expected:
            raise ProjectTranslationCheckpointError(
                "count-mismatch",
                f"Checkpoint {field_name} does not match its records.",
                path=f"$.plan.{field_name}",
                details={"expected": expected, "actual": plan.get(field_name)},
            )
    return {
        **plan,
        "jobs": jobs,
        "completed": completed,
        "active": active,
        "pending": pending,
    }


def _coordinate(value: Any, *, path: str) -> dict[str, Any]:
    coordinate = _json_mapping(value, path=path)
    _require_fields(coordinate, _COORDINATE_FIELDS, path=path, required=False)
    for field_name in ("source", "target", "path"):
        field_value = coordinate.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            _invalid(
                f"Translation coordinate {field_name} must be a non-empty string.",
                path=f"{path}.{field_name}",
            )
    for field_name in ("variant", "entryPoint"):
        field_value = coordinate.get(field_name)
        if field_value is not None and (
            not isinstance(field_value, str) or not field_value.strip()
        ):
            _invalid(
                f"Translation coordinate {field_name} must be null or a non-empty string.",
                path=f"{path}.{field_name}",
            )
    return coordinate


def _completion_record(value: Any, *, path: str) -> dict[str, Any]:
    record = _json_mapping(value, path=path)
    _require_fields(record, _COMPLETION_FIELDS, path=path)
    coordinate = _coordinate(record.get("coordinate"), path=f"{path}.coordinate")
    artifacts = record.get("artifacts")
    diagnostics = record.get("diagnostics")
    if not isinstance(artifacts, list) or not all(
        isinstance(item, Mapping) for item in artifacts
    ):
        _invalid(
            "Completed translation artifacts must be a list of objects.",
            path=f"{path}.artifacts",
        )
    if not isinstance(diagnostics, list) or not all(
        isinstance(item, Mapping) for item in diagnostics
    ):
        _invalid(
            "Completed translation diagnostics must be a list of objects.",
            path=f"{path}.diagnostics",
        )
    return {
        "coordinate": coordinate,
        "artifacts": [copy.deepcopy(dict(item)) for item in artifacts],
        "diagnostics": [copy.deepcopy(dict(item)) for item in diagnostics],
    }


def _validate_interruption(value: Any) -> None:
    interruption = _json_mapping(value, path="$.interruption")
    _require_fields(interruption, _INTERRUPTION_FIELDS, path="$.interruption")
    for field_name in _INTERRUPTION_FIELDS:
        field_value = interruption.get(field_name)
        if not isinstance(field_value, str) or not field_value:
            _invalid(
                f"Interruption {field_name} must be a non-empty string.",
                path=f"$.interruption.{field_name}",
            )


def _require_completed_coordinates_in_plan(
    jobs: Sequence[Mapping[str, Any]],
    completed: Mapping[str, Mapping[str, Any]],
) -> None:
    job_keys = {_coordinate_key(job) for job in jobs}
    for index, key in enumerate(completed):
        if key not in job_keys:
            raise ProjectTranslationCheckpointError(
                "completed-not-planned",
                "Completed translation coordinate is not present in the job plan.",
                path=f"$.plan.completed[{index}].coordinate",
            )


def _require_unique_coordinates(
    coordinates: Sequence[Mapping[str, Any]],
    *,
    path: str,
) -> None:
    seen: dict[str, int] = {}
    for index, coordinate in enumerate(coordinates):
        key = _coordinate_key(coordinate)
        if key in seen:
            raise ProjectTranslationCheckpointError(
                "job-duplicate",
                "Translation job coordinates must be unique.",
                path=f"{path}[{index}]",
                details={"firstIndex": seen[key], "duplicateIndex": index},
            )
        seen[key] = index


def _coordinate_key(coordinate: Mapping[str, Any]) -> str:
    return _canonical_json(coordinate)


def _payload_hash(payload: Mapping[str, Any]) -> dict[str, str]:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return {"algorithm": "sha256", "value": digest}


def _validate_hash(value: Any, *, path: str) -> None:
    hash_record = _json_mapping(value, path=path)
    _require_fields(hash_record, _HASH_FIELDS, path=path)
    if hash_record.get("algorithm") != "sha256":
        _invalid("Checkpoint hash algorithm must be sha256.", path=f"{path}.algorithm")
    digest = hash_record.get("value")
    if not isinstance(digest, str) or len(digest) != 64:
        _invalid(
            "Checkpoint hash value must contain 64 hexadecimal characters.",
            path=f"{path}.value",
        )
    try:
        int(digest, 16)
    except ValueError:
        _invalid(
            "Checkpoint hash value must contain 64 hexadecimal characters.",
            path=f"{path}.value",
        )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            json.dumps(
                payload,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ProjectTranslationCheckpointError(
            "write-failed",
            f"Project translation checkpoint could not be written: {exc}",
            path="$",
            details={"checkpointPath": str(path)},
        ) from exc


def _checkpoint_path(value: str | os.PathLike[str]) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise ProjectTranslationCheckpointError(
            "path-invalid",
            "Project translation checkpoint path must be string or path-like.",
            path="$",
        ) from exc
    if not isinstance(raw, str) or not raw.strip():
        raise ProjectTranslationCheckpointError(
            "path-invalid",
            "Project translation checkpoint path must be non-empty.",
            path="$",
        )
    return Path(raw)


def _timestamp(value: Any, *, path: str) -> int:
    if type(value) is not int or value < 0:
        _invalid("Checkpoint timestamp must be a non-negative integer.", path=path)
    return value


def _json_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _invalid("Expected a JSON object.", path=path)
    try:
        normalized = json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=True,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ProjectTranslationCheckpointError(
            "json-invalid",
            f"Value must contain JSON-compatible data: {exc}",
            path=path,
        ) from exc
    if not isinstance(normalized, dict):
        _invalid("Expected a JSON object.", path=path)
    return normalized


def _require_fields(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    path: str,
    required: bool = True,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ProjectTranslationCheckpointError(
            "field-unknown",
            "Project translation checkpoint contains unknown fields.",
            path=path,
            details={"fields": unknown},
        )
    if not required:
        return
    missing = sorted(allowed - set(value))
    if missing:
        raise ProjectTranslationCheckpointError(
            "field-missing",
            "Project translation checkpoint is missing required fields.",
            path=path,
            details={"fields": missing},
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _invalid(message: str, *, path: str) -> None:
    raise ProjectTranslationCheckpointError("invalid", message, path=path)
