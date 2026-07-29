import shutil
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import crosstl._crosstl as crosstl_cli
import crosstl.project as project_api
import crosstl.project.pipeline as project_pipeline
from crosstl.project import ProjectConfig, translate_project
from crosstl.project.translation_checkpoint import (
    ProjectTranslationCheckpointRecorder,
    load_project_translation_checkpoint,
)

SIMPLE_CROSSL = textwrap.dedent("""
    shader ConcurrentShader {
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

MULTI_ENTRY_METAL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    kernel void first(
        device float* output [[buffer(0)]],
        uint index [[thread_position_in_grid]]
    ) {
        output[index] = float(index);
    }

    kernel void second(
        device float* output [[buffer(0)]],
        uint index [[thread_position_in_grid]]
    ) {
        output[index] = float(index) + 1.0f;
    }
    """).strip()


def _write_project(root, *, unit_count=2):
    root.mkdir()
    for index in range(unit_count):
        (root / f"shader-{index}.cgl").write_text(
            SIMPLE_CROSSL,
            encoding="utf-8",
        )


def _artifact_sources(root, payload):
    return {
        artifact["path"]: (root / artifact["path"]).read_text(encoding="utf-8")
        for artifact in payload["artifacts"]
        if artifact["status"] == "translated"
    }


def test_parallel_translation_matches_sequential_artifacts_and_order(tmp_path):
    repo = tmp_path / "repo"
    _write_project(repo)
    config = ProjectConfig(
        root=repo,
        targets=("directx", "opengl"),
        output_dir="translated",
        variants={
            "debug": {"MODE": "1"},
            "release": {"MODE": "0"},
        },
    )

    sequential = translate_project(config, format_output=False)
    sequential_payload = sequential.to_json()
    sequential_sources = _artifact_sources(repo, sequential_payload)
    shutil.rmtree(repo / "translated")

    parallel = translate_project(config, format_output=False, max_workers=2)
    parallel_payload = parallel.to_json()
    parallel_sources = _artifact_sources(repo, parallel_payload)

    assert parallel_payload["artifacts"] == sequential_payload["artifacts"]
    assert parallel_payload["diagnostics"] == sequential_payload["diagnostics"]
    assert parallel_payload["artifactMatrix"] == sequential_payload["artifactMatrix"]
    assert parallel_sources == sequential_sources
    assert [
        (artifact["source"], artifact["target"], artifact["variant"])
        for artifact in parallel_payload["artifacts"]
    ] == [
        ("shader-0.cgl", "directx", "debug"),
        ("shader-0.cgl", "directx", "release"),
        ("shader-0.cgl", "opengl", "debug"),
        ("shader-0.cgl", "opengl", "release"),
        ("shader-1.cgl", "directx", "debug"),
        ("shader-1.cgl", "directx", "release"),
        ("shader-1.cgl", "opengl", "debug"),
        ("shader-1.cgl", "opengl", "release"),
    ]


def test_artifact_report_path_does_not_resolve_unpublished_output(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = ProjectConfig(root=repo, output_dir="translated")
    output_path = config.output_path / "directx" / "kernel.hlsl"

    def fail_resolve(*_args, **_kwargs):
        raise AssertionError("artifact report paths must not resolve the filesystem")

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    assert project_pipeline._artifact_report_path(output_path, config) == (
        "translated/directx/kernel.hlsl"
    )


def test_parallel_translation_preserves_failure_order(tmp_path):
    repo = tmp_path / "repo"
    _write_project(repo)
    (repo / "shader-0.cgl").write_text(
        "shader Broken { vertex { void main( { } }",
        encoding="utf-8",
    )
    config = ProjectConfig(
        root=repo,
        targets=("directx", "opengl"),
        output_dir="translated",
    )

    sequential = translate_project(config, format_output=False).to_json()
    shutil.rmtree(repo / "translated")
    parallel = translate_project(
        config,
        format_output=False,
        max_workers=2,
    ).to_json()

    assert parallel["artifacts"] == sequential["artifacts"]
    assert parallel["diagnostics"] == sequential["diagnostics"]
    assert parallel["summary"]["failedCount"] == 2
    assert parallel["summary"]["translatedCount"] == 2


def test_parallel_translation_does_not_duplicate_target_diagnostics(tmp_path):
    repo = tmp_path / "repo"
    _write_project(repo)
    config = ProjectConfig(
        root=repo,
        targets=("unsupported-target",),
        output_dir="translated",
    )

    sequential = translate_project(config, format_output=False).to_json()
    parallel = translate_project(
        config,
        format_output=False,
        max_workers=2,
    ).to_json()

    assert parallel["artifacts"] == sequential["artifacts"]
    assert parallel["diagnostics"] == sequential["diagnostics"]
    assert (
        parallel["summary"]["diagnosticsByCode"]["project.config.unsupported-target"]
        == 1
    )


def test_parallel_translation_preserves_entry_scoped_metal_artifacts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "kernels.metal").write_text(MULTI_ENTRY_METAL, encoding="utf-8")
    config = ProjectConfig(
        root=repo,
        targets=("directx", "opengl"),
        output_dir="translated",
        entry_points={"kernels.metal": ("first", "second")},
        include_dirs=(".",),
    )

    sequential = translate_project(config, format_output=False).to_json()
    sequential_sources = _artifact_sources(repo, sequential)
    shutil.rmtree(repo / "translated")
    parallel = translate_project(
        config,
        format_output=False,
        max_workers=2,
    ).to_json()

    assert parallel["artifacts"] == sequential["artifacts"]
    assert parallel["diagnostics"] == sequential["diagnostics"]
    assert _artifact_sources(repo, parallel) == sequential_sources
    assert [
        (
            artifact["target"],
            artifact["entryPoint"]["source"],
            artifact["entryPoint"]["target"],
        )
        for artifact in parallel["artifacts"]
    ] == [
        ("directx", "first", "CSMain"),
        ("directx", "second", "CSMain"),
        ("opengl", "first", "main"),
        ("opengl", "second", "main"),
    ]


def test_parallel_translation_checkpoint_is_coordinated_in_plan_order(tmp_path):
    repo = tmp_path / "repo"
    _write_project(repo)
    checkpoint_path = tmp_path / "translation-checkpoint.json"

    translate_project(
        ProjectConfig(
            root=repo,
            targets=("directx", "opengl"),
            output_dir="translated",
        ),
        format_output=False,
        max_workers=2,
        checkpoint_path=checkpoint_path,
    )

    checkpoint = load_project_translation_checkpoint(checkpoint_path)

    assert checkpoint["state"] == "complete"
    assert checkpoint["plan"]["active"] is None
    assert checkpoint["plan"]["completedCount"] == 4
    assert [
        completion["coordinate"] for completion in checkpoint["plan"]["completed"]
    ] == checkpoint["plan"]["jobs"]


def test_parallel_translation_resumes_verified_checkpoint_completions(tmp_path):
    repo = tmp_path / "repo"
    _write_project(repo)
    checkpoint_path = tmp_path / "translation-checkpoint.json"
    config = ProjectConfig(
        root=repo,
        targets=("directx", "opengl"),
        output_dir="translated",
    )

    baseline = translate_project(
        config,
        format_output=False,
        max_workers=2,
        checkpoint_path=checkpoint_path,
    ).to_json()
    complete_checkpoint = load_project_translation_checkpoint(checkpoint_path)
    initial_count = complete_checkpoint["initialDiagnosticCount"]
    recorder = ProjectTranslationCheckpointRecorder(
        checkpoint_path,
        complete_checkpoint["projectIdentity"],
        complete_checkpoint["plan"]["jobs"],
        started_at=complete_checkpoint["startedAt"],
        completed=complete_checkpoint["plan"]["completed"][:2],
        initial_diagnostics=complete_checkpoint["diagnostics"][:initial_count],
    )
    recorder.write_interrupted(None, RuntimeError("translation interrupted"))

    resumed = translate_project(
        config,
        format_output=False,
        max_workers=2,
        checkpoint_path=checkpoint_path,
        resume=True,
    ).to_json()
    resumed_checkpoint = load_project_translation_checkpoint(checkpoint_path)

    assert resumed["artifacts"] == baseline["artifacts"]
    assert resumed["diagnostics"] == baseline["diagnostics"]
    assert resumed_checkpoint["state"] == "complete"
    assert resumed_checkpoint["plan"]["completedCount"] == 4


def test_parallel_translation_bounds_submitted_work(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _write_project(repo, unit_count=5)
    executor_state = {"outstanding": 0, "maximum": 0, "submitted": 0}

    class TrackingFuture:
        def __init__(self, result):
            self._result = result
            self._resolved = False

        def result(self):
            if not self._resolved:
                executor_state["outstanding"] -= 1
                self._resolved = True
            return self._result

        def cancel(self):
            return False

    class TrackingExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            initializer(*initargs)

        def submit(self, function, *args):
            executor_state["outstanding"] += 1
            executor_state["submitted"] += 1
            executor_state["maximum"] = max(
                executor_state["maximum"],
                executor_state["outstanding"],
            )
            return TrackingFuture(function(*args))

        def shutdown(self, *, wait):
            assert wait is True

    monkeypatch.setattr(
        project_pipeline,
        "ProcessPoolExecutor",
        TrackingExecutor,
    )

    report = translate_project(
        ProjectConfig(
            root=repo,
            targets=("directx",),
            output_dir="translated",
        ),
        format_output=False,
        max_workers=2,
    ).to_json()

    assert report["summary"]["translatedCount"] == 5
    assert executor_state == {
        "outstanding": 0,
        "maximum": 2,
        "submitted": 5,
    }


def test_parallel_translation_publishes_only_when_results_are_consumed(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    _write_project(repo, unit_count=2)
    consumed_paths = []

    class StagedFuture:
        def __init__(self, request, result):
            self._request = request
            self._result = result

        def result(self):
            output_path = repo / self._request.coordinate["path"]
            assert not output_path.exists()
            assert self._result.publications
            assert all(
                publication.staging_directory.is_dir()
                for publication in self._result.publications
            )
            consumed_paths.append(self._request.coordinate["path"])
            return self._result

        def cancel(self):
            return False

    class StagedExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            initializer(*initargs)

        def submit(self, function, request):
            return StagedFuture(request, function(request))

        def shutdown(self, *, wait):
            assert wait is True

    monkeypatch.setattr(
        project_pipeline,
        "ProcessPoolExecutor",
        StagedExecutor,
    )

    payload = translate_project(
        ProjectConfig(
            root=repo,
            targets=("directx",),
            output_dir="translated",
        ),
        format_output=False,
        max_workers=2,
    ).to_json()

    assert consumed_paths == [
        "translated/directx/shader-0.hlsl",
        "translated/directx/shader-1.hlsl",
    ]
    assert [artifact["path"] for artifact in payload["artifacts"]] == consumed_paths
    assert all((repo / path).is_file() for path in consumed_paths)
    assert list((repo / "translated" / "directx").glob(".*.tmp")) == []


def test_parallel_worker_failure_removes_deferred_staging(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    staging_directory = repo / "translated" / ".shader.hlsl.test.tmp"
    staging_directory.mkdir(parents=True)
    publication = project_pipeline._DeferredProjectArtifactPublication(
        staging_directory=staging_directory,
        staged_output_path=staging_directory / "shader.hlsl",
        staged_artifact_paths=(),
        artifact_paths=(),
        failure_artifact={},
    )
    context = project_pipeline._ProjectTranslationWorkerContext(
        config=ProjectConfig(root=repo),
        dispatch_artifact_plan=None,
        preserve_source_suffix=frozenset(),
        format_output=False,
        output_dir_blocked=False,
    )
    monkeypatch.setattr(
        project_pipeline,
        "_PROJECT_TRANSLATION_WORKER_CONTEXT",
        context,
    )

    def fail_after_staging(*_args, deferred_publications, **_kwargs):
        deferred_publications.append(publication)
        raise RuntimeError("report assembly failed")

    monkeypatch.setattr(
        project_pipeline,
        "_translate_project_impl",
        fail_after_staging,
    )

    request = SimpleNamespace(unit=SimpleNamespace(), target="directx")
    with pytest.raises(RuntimeError, match="report assembly failed"):
        project_pipeline._run_project_translation_worker(request)

    assert not staging_directory.exists()


def test_parallel_translation_interrupt_discards_unconsumed_staging(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    _write_project(repo, unit_count=2)
    output_dir = repo / "translated" / "directx"
    output_dir.mkdir(parents=True)
    previous_outputs = {}
    for index in range(2):
        artifact_path = output_dir / f"shader-{index}.hlsl"
        remap_path = output_dir / f"shader-{index}.source-remap.json"
        artifact_path.write_bytes(f"previous artifact {index}\n".encode())
        remap_path.write_bytes(f'{{"previous": {index}}}\n'.encode())
        previous_outputs[artifact_path] = artifact_path.read_bytes()
        previous_outputs[remap_path] = remap_path.read_bytes()

    checkpoint_path = tmp_path / "translation-checkpoint.json"
    executor_state = {"submitted": 0, "shutdown": False}

    class InterruptingFuture:
        def __init__(self, result, *, interrupt):
            self._result = result
            self._interrupt = interrupt
            self._result_calls = 0

        def result(self):
            self._result_calls += 1
            if self._interrupt and self._result_calls == 1:
                raise KeyboardInterrupt("translation interrupted")
            assert executor_state["shutdown"] is True
            return self._result

        def cancel(self):
            return False

    class InterruptingExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            initializer(*initargs)

        def submit(self, function, request):
            executor_state["submitted"] += 1
            return InterruptingFuture(
                function(request),
                interrupt=executor_state["submitted"] == 1,
            )

        def shutdown(self, *, wait):
            assert wait is True
            executor_state["shutdown"] = True

    monkeypatch.setattr(
        project_pipeline,
        "ProcessPoolExecutor",
        InterruptingExecutor,
    )

    with pytest.raises(KeyboardInterrupt, match="translation interrupted"):
        translate_project(
            ProjectConfig(
                root=repo,
                targets=("directx",),
                output_dir="translated",
            ),
            format_output=False,
            max_workers=2,
            checkpoint_path=checkpoint_path,
        )

    assert executor_state == {"submitted": 2, "shutdown": True}
    assert {path: path.read_bytes() for path in previous_outputs} == previous_outputs
    assert list(output_dir.glob(".*.tmp")) == []
    checkpoint = load_project_translation_checkpoint(checkpoint_path)
    assert checkpoint["state"] == "interrupted"
    assert checkpoint["plan"]["completedCount"] == 0
    assert checkpoint["plan"]["active"]["path"] == ("translated/directx/shader-0.hlsl")


def test_parallel_translation_streams_plan_with_bounded_lookahead(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    _write_project(repo, unit_count=6)
    original_plan = project_pipeline._project_artifact_translation_plan
    plan_state = {
        "passes": 0,
        "requested": 0,
        "resolved": 0,
        "maximum_ahead": 0,
    }

    def tracking_plan(*args, **kwargs):
        plan_state["passes"] += 1
        current_pass = plan_state["passes"]
        for item in original_plan(*args, **kwargs):
            if current_pass == 2:
                plan_state["requested"] += 1
                ahead = plan_state["requested"] - plan_state["resolved"]
                plan_state["maximum_ahead"] = max(
                    plan_state["maximum_ahead"],
                    ahead,
                )
                assert ahead <= 2
            yield item

    class TrackingFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            plan_state["resolved"] += 1
            return self._result

        def cancel(self):
            return False

    class TrackingExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            initializer(*initargs)

        def submit(self, function, *args):
            return TrackingFuture(function(*args))

        def shutdown(self, *, wait):
            assert wait is True

    monkeypatch.setattr(
        project_pipeline,
        "_project_artifact_translation_plan",
        tracking_plan,
    )
    monkeypatch.setattr(
        project_pipeline,
        "ProcessPoolExecutor",
        TrackingExecutor,
    )

    report = translate_project(
        ProjectConfig(
            root=repo,
            targets=("directx",),
            output_dir="translated",
        ),
        format_output=False,
        max_workers=2,
    ).to_json()

    assert report["summary"]["translatedCount"] == 6
    assert plan_state == {
        "passes": 2,
        "requested": 6,
        "resolved": 6,
        "maximum_ahead": 2,
    }


def test_parallel_translation_rejects_output_path_collisions(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    _write_project(repo, unit_count=1)
    job = project_pipeline._ProjectArtifactTranslationJob(
        variant=None,
        defines={},
        entry_point=None,
    )
    monkeypatch.setattr(
        project_pipeline,
        "_project_translation_jobs_for_target",
        lambda *_args, **_kwargs: [job, job],
    )

    class UnexpectedExecutor:
        def __init__(self, **_kwargs):
            raise AssertionError("workers must not start before collision preflight")

    monkeypatch.setattr(
        project_pipeline,
        "ProcessPoolExecutor",
        UnexpectedExecutor,
    )

    with pytest.raises(
        ValueError,
        match="Project translation jobs resolve to the same artifact path",
    ):
        translate_project(
            ProjectConfig(
                root=repo,
                targets=("directx",),
                output_dir="translated",
            ),
            format_output=False,
            max_workers=2,
        )

    assert not (repo / "translated").exists()


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "2"))
def test_translate_project_rejects_invalid_max_workers(tmp_path, value):
    repo = tmp_path / "repo"
    _write_project(repo, unit_count=1)

    with pytest.raises(
        ValueError,
        match="max_workers must be a positive integer",
    ):
        translate_project(repo, max_workers=value)


def test_translate_project_cli_forwards_workers(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    observed = {}
    payload = {
        "kind": project_pipeline.REPORT_KIND,
        "summary": {"failedCount": 0, "diagnosticCounts": {"error": 0}},
    }

    def fake_translate_project(config, **kwargs):
        observed["root"] = config.root
        observed.update(kwargs)
        return SimpleNamespace(to_json=lambda: payload)

    monkeypatch.setattr(project_api, "translate_project", fake_translate_project)

    exit_code = crosstl_cli.main(
        [
            "translate-project",
            str(repo),
            "--workers",
            "3",
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    assert observed["root"] == repo.resolve()
    assert observed["max_workers"] == 3
