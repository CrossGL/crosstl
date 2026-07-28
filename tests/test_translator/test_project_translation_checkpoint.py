import copy
import json
import shutil
import textwrap
from pathlib import Path

import pytest

import crosstl._crosstl as crosstl_cli
import crosstl.project as project_api
import crosstl.project.pipeline as project_pipeline
import crosstl.project.translation_checkpoint as checkpoint_module
from crosstl.project.translation_checkpoint import (
    PROJECT_TRANSLATION_CHECKPOINT_KIND,
    PROJECT_TRANSLATION_CHECKPOINT_VERSION,
    ProjectTranslationCheckpointError,
    ProjectTranslationCheckpointRecorder,
    load_project_translation_checkpoint,
    validate_project_translation_checkpoint,
)

SIMPLE_CROSSL = textwrap.dedent("""
    shader CheckpointShader {
        struct VertexInput {
            vec3 position;
        }

        struct VertexOutput {
            vec4 position;
        }

        vertex {
            VertexOutput main(VertexInput input) {
                VertexOutput output;
                output.position = vec4(input.position, 1.0);
                return output;
            }
        }
    }
    """).strip()

MULTI_ENTRY_COMPUTE_CROSSL = textwrap.dedent("""
    shader CheckpointCompute {
        RWStructuredBuffer<uint> output @set(0) @binding(0);

        compute first {
            @numthreads(1, 1, 1)
            void main(uint index @gl_GlobalInvocationID) {
                output[index] = 1u;
            }
        }

        compute second {
            @numthreads(1, 1, 1)
            void main(uint index @gl_GlobalInvocationID) {
                output[index] = 2u;
            }
        }
    }
    """).strip()


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


def _write_project(root):
    root.mkdir()
    for name in ("first.cgl", "second.cgl"):
        (root / name).write_text(SIMPLE_CROSSL, encoding="utf-8")


def test_checkpoint_records_active_completed_pending_and_final_report(tmp_path):
    path = tmp_path / "progress.json"
    jobs = _jobs()
    recorder = ProjectTranslationCheckpointRecorder(
        path,
        _identity(),
        jobs,
        started_at=100,
        initial_diagnostics=_diagnostics(),
    )

    initial = recorder.write_running()
    assert initial["schemaVersion"] == PROJECT_TRANSLATION_CHECKPOINT_VERSION
    assert initial["kind"] == PROJECT_TRANSLATION_CHECKPOINT_KIND
    assert initial["state"] == "running"
    assert initial["plan"]["completed"] == []
    assert initial["plan"]["active"] is None
    assert initial["plan"]["pending"] == jobs
    assert initial["initialDiagnosticCount"] == 1
    assert initial["diagnostics"] == _diagnostics()
    assert initial["artifactMatrix"]["expectedJobCount"] == 2
    assert initial["artifactMatrix"]["pendingJobCount"] == 2

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
    assert completed["diagnostics"] == [*_diagnostics(), *_diagnostics()]
    assert completed["artifactMatrix"]["completedJobCount"] == 1
    assert completed["artifactMatrix"]["translatedArtifactCount"] == 1

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


def test_checkpoint_write_interval_bounds_progress_writes(tmp_path):
    path = tmp_path / "progress.json"
    jobs = _jobs()
    recorder = ProjectTranslationCheckpointRecorder(
        path,
        _identity(),
        jobs,
        write_interval_jobs=2,
    )
    recorder.write_running()
    recorder.activate(jobs[0])
    assert recorder.record_completion(jobs[0], _artifacts(jobs[0]), []) is None
    assert recorder.activate(jobs[1]) is None

    persisted = load_project_translation_checkpoint(path)
    assert persisted["plan"]["active"] == jobs[0]
    assert persisted["plan"]["completedCount"] == 0

    updated = recorder.record_completion(jobs[1], _artifacts(jobs[1]), [])
    assert updated is not None
    assert updated["plan"]["active"] is None
    assert updated["plan"]["completedCount"] == 2


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


def test_checkpoint_records_message_for_bare_keyboard_interrupt(tmp_path):
    path = tmp_path / "progress.json"
    jobs = _jobs()
    recorder = ProjectTranslationCheckpointRecorder(path, _identity(), jobs)

    interrupted = recorder.write_interrupted(jobs[0], KeyboardInterrupt())

    assert interrupted["state"] == "interrupted"
    assert interrupted["interruption"] == {
        "type": "KeyboardInterrupt",
        "message": "KeyboardInterrupt",
    }


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


def test_translate_project_checkpoint_resumes_only_verified_pending_jobs(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    _write_project(repo)
    checkpoint_path = repo / "progress.json"
    config = project_pipeline.ProjectConfig(
        root=repo,
        include_dirs=("missing-includes",),
    )
    monkeypatch.setattr(project_pipeline.time, "time", lambda: 1000)
    monkeypatch.setattr(checkpoint_module.time, "time", lambda: 1000)

    baseline = project_api.translate_project(
        config,
        targets=["cgl"],
        output_dir="out",
        format_output=False,
    ).to_json()
    shutil.rmtree(repo / "out")

    original_translate = project_pipeline.translate
    interrupted_calls = []

    def interrupt_second(path, *args, **kwargs):
        interrupted_calls.append(Path(path).name)
        if Path(path).name == "second.cgl":
            raise KeyboardInterrupt("interrupted")
        return original_translate(path, *args, **kwargs)

    monkeypatch.setattr(project_pipeline, "translate", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        project_api.translate_project(
            config,
            targets=["cgl"],
            output_dir="out",
            format_output=False,
            checkpoint_path=checkpoint_path,
        )

    interrupted = load_project_translation_checkpoint(checkpoint_path)
    assert interrupted["state"] == "interrupted"
    assert interrupted["plan"]["completedCount"] == 1
    assert interrupted["plan"]["active"]["source"] == "second.cgl"
    assert interrupted["plan"]["pending"] == []
    assert interrupted["initialDiagnosticCount"] == 1
    assert interrupted["diagnostics"][0]["code"] == (
        "project.config.missing-include-dir"
    )
    assert interrupted_calls == ["first.cgl", "second.cgl"]

    resumed_calls = []

    def record_resumed_translation(path, *args, **kwargs):
        resumed_calls.append(Path(path).name)
        return original_translate(path, *args, **kwargs)

    monkeypatch.setattr(
        project_pipeline,
        "translate",
        record_resumed_translation,
    )
    resumed = project_api.translate_project(
        config,
        targets=["cgl"],
        output_dir="out",
        format_output=False,
        checkpoint_path=checkpoint_path,
        resume=True,
    ).to_json()

    assert resumed_calls == ["second.cgl"]
    assert resumed == baseline
    completed = load_project_translation_checkpoint(checkpoint_path)
    assert completed["state"] == "complete"
    assert completed["plan"]["completedCount"] == 2
    assert completed["plan"]["pending"] == []
    assert completed["finalReport"] == resumed


def test_translate_project_checkpoint_resumes_pending_entry_artifact(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "multi.cgl").write_text(
        MULTI_ENTRY_COMPUTE_CROSSL,
        encoding="utf-8",
    )
    checkpoint_path = repo / "progress.json"
    config = project_pipeline.ProjectConfig(
        root=repo,
        include_patterns=("multi.cgl",),
        targets=("opengl",),
        output_dir="out",
        entry_points={"multi.cgl": ("first", "second")},
    )
    original_generate = project_pipeline._generate_project_target_from_crossgl_ast
    interrupted_entries = []

    def interrupt_second(*args, **kwargs):
        entry_point = kwargs.get("entry_point")
        interrupted_entries.append(entry_point)
        if entry_point == "second":
            raise KeyboardInterrupt("interrupted")
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(
        project_pipeline,
        "_generate_project_target_from_crossgl_ast",
        interrupt_second,
    )
    with pytest.raises(KeyboardInterrupt):
        project_api.translate_project(
            config,
            format_output=False,
            checkpoint_path=checkpoint_path,
        )

    interrupted = load_project_translation_checkpoint(checkpoint_path)
    assert interrupted["state"] == "interrupted"
    assert interrupted["plan"]["completedCount"] == 1
    assert interrupted["plan"]["completed"][0]["coordinate"]["entryPoint"] == "first"
    assert interrupted["plan"]["active"]["entryPoint"] == "second"
    assert interrupted["plan"]["pending"] == []
    assert interrupted_entries == ["first", "second"]

    resumed_entries = []

    def record_resumed_translation(*args, **kwargs):
        resumed_entries.append(kwargs.get("entry_point"))
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(
        project_pipeline,
        "_generate_project_target_from_crossgl_ast",
        record_resumed_translation,
    )
    resumed = project_api.translate_project(
        config,
        format_output=False,
        checkpoint_path=checkpoint_path,
        resume=True,
    ).to_json()

    assert resumed_entries == ["second"]
    assert resumed["summary"]["translatedCount"] == 2
    assert [artifact["entryPoint"]["source"] for artifact in resumed["artifacts"]] == [
        "first",
        "second",
    ]
    completed = load_project_translation_checkpoint(checkpoint_path)
    assert completed["state"] == "complete"
    assert completed["plan"]["completedCount"] == 2
    assert completed["finalReport"] == resumed


@pytest.mark.parametrize(
    ("modified_output", "expected_reason"),
    (
        ("artifact", "artifact-hash-mismatch"),
        ("source-remap", "source-remap-hash-mismatch"),
    ),
)
def test_translate_project_resume_rejects_modified_completed_output(
    tmp_path,
    monkeypatch,
    modified_output,
    expected_reason,
):
    repo = tmp_path / "repo"
    _write_project(repo)
    checkpoint_path = tmp_path / "progress.json"
    original_translate = project_pipeline.translate

    def interrupt_second(path, *args, **kwargs):
        if Path(path).name == "second.cgl":
            raise KeyboardInterrupt("interrupted")
        return original_translate(path, *args, **kwargs)

    monkeypatch.setattr(project_pipeline, "translate", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        project_api.translate_project(
            repo,
            targets=["cgl"],
            output_dir="out",
            format_output=False,
            checkpoint_path=checkpoint_path,
        )

    interrupted = load_project_translation_checkpoint(checkpoint_path)
    first_artifact = interrupted["plan"]["completed"][0]["artifacts"][0]
    modified_path = (
        first_artifact["path"]
        if modified_output == "artifact"
        else first_artifact["sourceRemap"]["path"]
    )
    (repo / modified_path).write_text("modified\n", encoding="utf-8")
    monkeypatch.setattr(project_pipeline, "translate", original_translate)

    with pytest.raises(ProjectTranslationCheckpointError) as caught:
        project_api.translate_project(
            repo,
            targets=["cgl"],
            output_dir="out",
            format_output=False,
            checkpoint_path=checkpoint_path,
            resume=True,
        )

    assert caught.value.reason == expected_reason


def test_translate_project_records_final_checkpoint_write_failure(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    _write_project(repo)
    checkpoint_path = tmp_path / "progress.json"

    def fail_final_write(self, final_report):
        raise RuntimeError("final checkpoint unavailable")

    monkeypatch.setattr(
        ProjectTranslationCheckpointRecorder,
        "write_complete",
        fail_final_write,
    )

    with pytest.raises(RuntimeError, match="final checkpoint unavailable"):
        project_api.translate_project(
            repo,
            targets=["cgl"],
            output_dir="out",
            checkpoint_path=checkpoint_path,
        )

    checkpoint = load_project_translation_checkpoint(checkpoint_path)
    assert checkpoint["state"] == "interrupted"
    assert checkpoint["plan"]["completedCount"] == 2
    assert checkpoint["plan"]["active"] is None
    assert checkpoint["interruption"]["type"] == "RuntimeError"


def test_translate_project_requires_checkpoint_path_for_resume(tmp_path):
    repo = tmp_path / "repo"
    _write_project(repo)

    with pytest.raises(ValueError, match="resume requires checkpoint_path"):
        project_api.translate_project(repo, targets=["cgl"], resume=True)


@pytest.mark.parametrize(
    "checkpoint_name",
    (
        "first.cgl",
        "out/progress.json",
    ),
)
def test_translate_project_rejects_checkpoint_path_collisions(
    tmp_path,
    checkpoint_name,
):
    repo = tmp_path / "repo"
    _write_project(repo)

    with pytest.raises(ProjectTranslationCheckpointError) as caught:
        project_api.translate_project(
            repo,
            targets=["cgl"],
            output_dir="out",
            checkpoint_path=repo / checkpoint_name,
        )

    assert caught.value.reason == "path-conflict"


def test_project_checkpoint_api_is_public():
    for name in (
        "PROJECT_TRANSLATION_CHECKPOINT_KIND",
        "PROJECT_TRANSLATION_CHECKPOINT_VERSION",
        "ProjectTranslationCheckpointError",
        "ProjectTranslationCheckpointRecorder",
        "load_project_translation_checkpoint",
        "validate_project_translation_checkpoint",
    ):
        assert name in project_api.__all__


def test_translate_project_cli_forwards_checkpoint_resume(
    tmp_path,
    monkeypatch,
    capsys,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    checkpoint_path = tmp_path / "progress.json"
    calls = []
    payload = {
        "kind": project_pipeline.REPORT_KIND,
        "summary": {"failedCount": 0, "diagnosticCounts": {"error": 0}},
    }

    def fake_translate_project(config, **kwargs):
        calls.append({"root": config.root, **kwargs})
        return type("Report", (), {"to_json": lambda self: payload})()

    monkeypatch.setattr(project_api, "translate_project", fake_translate_project)

    exit_code = crosstl_cli.main(
        [
            "translate-project",
            str(repo),
            "--target",
            "opengl",
            "--checkpoint",
            str(checkpoint_path),
            "--resume",
            "--checkpoint-interval-jobs",
            "4",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == payload
    assert calls == [
        {
            "root": repo.resolve(),
            "targets": ("opengl",),
            "output_dir": None,
            "variants": None,
            "format_output": True,
            "validate": False,
            "run_toolchains": False,
            "checkpoint_path": str(checkpoint_path),
            "resume": True,
            "checkpoint_interval_jobs": 4,
        }
    ]


def test_translate_project_cli_rejects_resume_without_checkpoint(
    tmp_path,
    capsys,
):
    repo = tmp_path / "repo"
    repo.mkdir()

    exit_code = crosstl_cli.main(["translate-project", str(repo), "--resume"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "Error: --resume requires --checkpoint\n"


def test_translate_project_cli_rejects_checkpoint_interval_without_checkpoint(
    tmp_path,
    capsys,
):
    repo = tmp_path / "repo"
    repo.mkdir()

    exit_code = crosstl_cli.main(
        [
            "translate-project",
            str(repo),
            "--checkpoint-interval-jobs",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "Error: --checkpoint-interval-jobs requires --checkpoint\n"
