import copy
import json

import pytest

import crosstl.project.translation_checkpoint as checkpoint_module
from crosstl.project.translation_checkpoint import (
    PROJECT_TRANSLATION_CHECKPOINT_KIND,
    PROJECT_TRANSLATION_CHECKPOINT_VERSION,
    ProjectTranslationCheckpointError,
    ProjectTranslationCheckpointRecorder,
    load_project_translation_checkpoint,
    validate_project_translation_checkpoint,
)


def _identity():
    return {
        "root": "/project",
        "configHash": {
            "algorithm": "sha256",
            "value": "1" * 64,
        },
        "invocationHash": {
            "algorithm": "sha256",
            "value": "2" * 64,
        },
    }


def _jobs():
    return [
        {
            "source": "kernels/first.metal",
            "target": "directx",
            "path": "out/directx/kernels/first.hlsl",
            "variant": "float32",
        },
        {
            "source": "kernels/second.metal",
            "target": "opengl",
            "path": "out/opengl/kernels/second.glsl",
            "entryPoint": "second_float",
        },
    ]


def _artifacts(job):
    return [
        {
            "source": job["source"],
            "target": job["target"],
            "path": job["path"],
            "status": "translated",
        }
    ]


def _diagnostics():
    return [
        {
            "severity": "warning",
            "code": "project.translate.example",
            "message": "Example diagnostic.",
            "location": {"file": "kernels/first.metal", "line": 1, "column": 1},
        }
    ]


def test_checkpoint_records_active_completed_pending_and_final_report(tmp_path):
    path = tmp_path / "progress.json"
    jobs = _jobs()
    recorder = ProjectTranslationCheckpointRecorder(
        path,
        _identity(),
        jobs,
        started_at=100,
    )

    initial = recorder.write_running()
    assert initial["schemaVersion"] == PROJECT_TRANSLATION_CHECKPOINT_VERSION
    assert initial["kind"] == PROJECT_TRANSLATION_CHECKPOINT_KIND
    assert initial["state"] == "running"
    assert initial["plan"]["completed"] == []
    assert initial["plan"]["active"] is None
    assert initial["plan"]["pending"] == jobs

    active = recorder.write_running(jobs[0])
    assert active["plan"]["active"] == jobs[0]
    assert active["plan"]["pending"] == [jobs[1]]

    completed = recorder.record_completion(
        jobs[0],
        _artifacts(jobs[0]),
        _diagnostics(),
    )
    assert completed["plan"]["completedCount"] == 1
    assert completed["plan"]["completed"][0]["coordinate"] == jobs[0]
    assert completed["plan"]["active"] is None
    assert completed["plan"]["pending"] == [jobs[1]]

    recorder.write_running(jobs[1])
    recorder.record_completion(jobs[1], _artifacts(jobs[1]), [])
    final_report = {"kind": "crosstl-project-portability-report", "artifacts": []}
    final = recorder.write_complete(final_report)

    assert final["state"] == "complete"
    assert final["completedAt"] is not None
    assert final["plan"]["completedCount"] == 2
    assert final["plan"]["pending"] == []
    assert final["finalReport"] == final_report
    assert load_project_translation_checkpoint(path) == final


def test_checkpoint_records_interruption_and_resumes_completed_jobs(tmp_path):
    path = tmp_path / "progress.json"
    jobs = _jobs()
    recorder = ProjectTranslationCheckpointRecorder(path, _identity(), jobs)
    recorder.record_completion(jobs[0], _artifacts(jobs[0]), _diagnostics())

    interrupted = recorder.write_interrupted(jobs[1], RuntimeError("stopped"))

    assert interrupted["state"] == "interrupted"
    assert interrupted["plan"]["active"] == jobs[1]
    assert interrupted["plan"]["pending"] == []
    assert interrupted["interruption"] == {
        "type": "RuntimeError",
        "message": "stopped",
    }

    resumed = ProjectTranslationCheckpointRecorder.resume(
        path,
        _identity(),
        jobs,
    )
    assert resumed.completed == tuple(interrupted["plan"]["completed"])
    assert resumed.completion_for(jobs[0]) == interrupted["plan"]["completed"][0]
    assert resumed.completion_for(jobs[1]) is None

    running = resumed.write_running(jobs[1])
    assert running["state"] == "running"
    assert running["interruption"] is None
    assert running["plan"]["active"] == jobs[1]


@pytest.mark.parametrize("mismatch", ["identity", "jobs"])
def test_checkpoint_resume_rejects_stale_translation_identity(tmp_path, mismatch):
    path = tmp_path / "progress.json"
    jobs = _jobs()
    recorder = ProjectTranslationCheckpointRecorder(path, _identity(), jobs)
    recorder.write_running()

    identity = _identity()
    resumed_jobs = copy.deepcopy(jobs)
    expected_reason = "project-identity-mismatch"
    if mismatch == "identity":
        identity["root"] = "/different"
    else:
        resumed_jobs[0]["target"] = "opengl"
        expected_reason = "job-plan-mismatch"

    with pytest.raises(ProjectTranslationCheckpointError) as caught:
        ProjectTranslationCheckpointRecorder.resume(path, identity, resumed_jobs)

    assert caught.value.reason == expected_reason


def test_checkpoint_resume_rejects_complete_translation(tmp_path):
    path = tmp_path / "progress.json"
    jobs = _jobs()
    recorder = ProjectTranslationCheckpointRecorder(path, _identity(), jobs)
    for job in jobs:
        recorder.record_completion(job, _artifacts(job), [])
    recorder.write_complete({"kind": "report"})

    with pytest.raises(ProjectTranslationCheckpointError) as caught:
        ProjectTranslationCheckpointRecorder.resume(path, _identity(), jobs)

    assert caught.value.reason == "already-complete"


def test_checkpoint_validation_rejects_inconsistent_plan_and_payload_hash(tmp_path):
    path = tmp_path / "progress.json"
    recorder = ProjectTranslationCheckpointRecorder(path, _identity(), _jobs())
    payload = recorder.write_running(_jobs()[0])

    invalid_count = copy.deepcopy(payload)
    invalid_count["plan"]["pendingCount"] = 2
    with pytest.raises(ProjectTranslationCheckpointError) as caught:
        validate_project_translation_checkpoint(invalid_count)
    assert caught.value.reason == "count-mismatch"

    invalid_hash = copy.deepcopy(payload)
    invalid_hash["checkpointHash"]["value"] = "0" * 64
    with pytest.raises(ProjectTranslationCheckpointError) as caught:
        validate_project_translation_checkpoint(invalid_hash)
    assert caught.value.reason == "hash-mismatch"


def test_checkpoint_atomic_write_preserves_previous_file_on_serialization_failure(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "progress.json"
    path.write_text('{"previous": true}\n', encoding="utf-8")

    def fail_serialization(*args, **kwargs):
        raise TypeError("cannot serialize")

    monkeypatch.setattr(checkpoint_module.json, "dumps", fail_serialization)

    with pytest.raises(ProjectTranslationCheckpointError) as caught:
        checkpoint_module._atomic_write_json(path, {"replacement": True})

    assert caught.value.reason == "write-failed"
    assert json.loads(path.read_text(encoding="utf-8")) == {"previous": True}
    assert list(tmp_path.glob(".progress.json.*.tmp")) == []
