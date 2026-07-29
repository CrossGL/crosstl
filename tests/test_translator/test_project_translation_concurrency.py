import json
import shutil
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import crosstl._crosstl as crosstl_cli
import crosstl.project as project_api
import crosstl.project.pipeline as project_pipeline
from crosstl.project import (
    ProjectConfig,
    ProjectTranslationWorkerError,
    translate_project,
)
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


def test_worker_limit_one_uses_sequential_execution(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _write_project(repo, unit_count=2)

    class UnexpectedExecutor:
        def __init__(self, **_kwargs):
            raise AssertionError("max_workers=1 must not construct a worker pool")

    monkeypatch.setattr(
        project_pipeline,
        "ProcessPoolExecutor",
        UnexpectedExecutor,
    )

    report = translate_project(
        ProjectConfig(
            root=repo,
            targets=("directx",),
            output_dir="translated",
        ),
        format_output=False,
        max_workers=1,
    ).to_json()

    assert report["summary"]["translatedCount"] == 2


def test_parallel_validation_starts_after_worker_pool_shutdown(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    _write_project(repo, unit_count=2)
    executor_state = {"shutdown": False}
    validation_state = {"artifacts": False, "toolchains": False}

    class ImmediateFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

        def cancel(self):
            return False

    class ImmediateExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            initializer(*initargs)

        def submit(self, function, request):
            return ImmediateFuture(function(request))

        def shutdown(self, *, wait):
            assert wait is True
            executor_state["shutdown"] = True

    def validate_artifacts(_artifacts, _targets, _config):
        assert executor_state["shutdown"] is True
        validation_state["artifacts"] = True
        return {
            "toolchains": [],
            "artifacts": [],
            "_diagnostics": [],
        }

    def run_toolchains(_artifacts, _root):
        assert executor_state["shutdown"] is True
        validation_state["toolchains"] = True
        return []

    monkeypatch.setattr(
        project_pipeline,
        "ProcessPoolExecutor",
        ImmediateExecutor,
    )
    monkeypatch.setattr(
        project_pipeline,
        "_validate_artifacts",
        validate_artifacts,
    )
    monkeypatch.setattr(
        project_pipeline,
        "_run_toolchain_smoke",
        run_toolchains,
    )

    report = translate_project(
        ProjectConfig(
            root=repo,
            targets=("directx",),
            output_dir="translated",
        ),
        format_output=False,
        validate=True,
        run_toolchains=True,
        max_workers=2,
    ).to_json()

    assert report["summary"]["translatedCount"] == 2
    assert executor_state["shutdown"] is True
    assert validation_state == {
        "artifacts": True,
        "toolchains": True,
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
        publication_staging_token="test",
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


def test_worker_failure_error_reports_entry_point():
    original_error = ValueError("worker result could not be decoded")
    error = ProjectTranslationWorkerError(
        {
            "source": "kernels.metal",
            "target": "directx",
            "path": "translated/directx/kernels/copy.hlsl",
            "entryPoint": "copy",
        },
        original_error,
    )

    assert error.original_error is original_error
    assert "entry point 'copy'" in str(error)


def test_executor_termination_uses_supported_pool_api():
    state = {"terminated": False, "shutdown": False}

    class Executor:
        def terminate_workers(self):
            state["terminated"] = True

        def shutdown(self, *, wait):
            assert wait is True
            assert state["terminated"] is True
            state["shutdown"] = True

    project_pipeline._terminate_project_translation_executor(Executor())

    assert state == {
        "terminated": True,
        "shutdown": True,
    }


def test_executor_termination_supports_legacy_pool_processes():
    state = {"terminated": [], "shutdown": False}

    class Process:
        def __init__(self, name, *, alive):
            self.name = name
            self.alive = alive

        def is_alive(self):
            return self.alive

        def terminate(self):
            state["terminated"].append(self.name)

    class Executor:
        def __init__(self):
            self._processes = {
                1: Process("active", alive=True),
                2: Process("complete", alive=False),
            }

        def shutdown(self, *, wait):
            assert wait is True
            state["shutdown"] = True

    project_pipeline._terminate_project_translation_executor(Executor())

    assert state == {
        "terminated": ["active"],
        "shutdown": True,
    }


def test_parallel_worker_failure_reports_coordinate_and_checkpoint(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    _write_project(repo, unit_count=2)
    checkpoint_path = tmp_path / "translation-checkpoint.json"
    executor_state = {
        "submitted": 0,
        "shutdown": False,
        "pending_cancelled": False,
    }

    class FailingFuture:
        def result(self):
            raise RuntimeError("worker transport failed")

        def cancel(self):
            raise AssertionError("the active future must not be cancelled")

    class PendingFuture:
        def result(self):
            assert executor_state["shutdown"] is True
            return project_pipeline._ProjectArtifactTranslationResult((), ())

        def cancel(self):
            executor_state["pending_cancelled"] = True
            return True

    class FailingExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            initializer(*initargs)

        def submit(self, _function, _request):
            executor_state["submitted"] += 1
            if executor_state["submitted"] == 1:
                return FailingFuture()
            return PendingFuture()

        def shutdown(self, *, wait):
            assert wait is True
            executor_state["shutdown"] = True

    monkeypatch.setattr(
        project_pipeline,
        "ProcessPoolExecutor",
        FailingExecutor,
    )

    with pytest.raises(ProjectTranslationWorkerError) as caught:
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

    error = caught.value
    expected_coordinate = {
        "source": "shader-0.cgl",
        "target": "directx",
        "path": "translated/directx/shader-0.hlsl",
    }
    assert error.coordinate == expected_coordinate
    assert error.error_type == "RuntimeError"
    assert isinstance(error.original_error, RuntimeError)
    assert str(error.original_error) == "worker transport failed"
    assert error.__cause__ is error.original_error
    assert str(error) == (
        "Project translation worker failed for source 'shader-0.cgl', "
        "target 'directx', artifact 'translated/directx/shader-0.hlsl': "
        "RuntimeError: worker transport failed"
    )
    assert executor_state == {
        "submitted": 2,
        "shutdown": True,
        "pending_cancelled": True,
    }

    checkpoint = load_project_translation_checkpoint(checkpoint_path)
    assert checkpoint["state"] == "interrupted"
    assert checkpoint["plan"]["completedCount"] == 0
    assert checkpoint["plan"]["active"] == expected_coordinate
    assert checkpoint["interruption"] == {
        "type": "ProjectTranslationWorkerError",
        "message": str(error),
    }


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
    unrelated_staging = output_dir / ".shader-0.hlsl.unrelated.tmp"
    unrelated_staging.mkdir()

    checkpoint_path = tmp_path / "translation-checkpoint.json"
    executor_state = {
        "submitted": 0,
        "terminated": False,
        "shutdown": False,
    }

    class InterruptingFuture:
        def __init__(self, result, *, interrupt):
            self._result = result
            self._interrupt = interrupt

        def result(self):
            if self._interrupt and not executor_state["terminated"]:
                raise KeyboardInterrupt("translation interrupted")
            if executor_state["terminated"]:
                raise RuntimeError("worker terminated")
            return self._result

        def cancel(self):
            return False

    class InterruptingExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            self._context = initargs[0]
            initializer(*initargs)

        def submit(self, function, request):
            executor_state["submitted"] += 1
            result = function(request)
            assert result.publications
            assert all(
                f".{self._context.publication_staging_token}."
                in publication.staging_directory.name
                for publication in result.publications
            )
            return InterruptingFuture(
                result,
                interrupt=executor_state["submitted"] == 1,
            )

        def terminate_workers(self):
            executor_state["terminated"] = True

        def shutdown(self, *, wait):
            assert wait is True
            assert executor_state["terminated"] is True
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

    assert executor_state == {
        "submitted": 2,
        "terminated": True,
        "shutdown": True,
    }
    assert {path: path.read_bytes() for path in previous_outputs} == previous_outputs
    assert list(output_dir.glob(".*.tmp")) == [unrelated_staging]
    assert unrelated_staging.is_dir()
    checkpoint = load_project_translation_checkpoint(checkpoint_path)
    assert checkpoint["state"] == "interrupted"
    assert checkpoint["plan"]["completedCount"] == 0
    assert checkpoint["plan"]["active"]["path"] == ("translated/directx/shader-0.hlsl")


def test_job_timeout_records_failure_and_restarts_pending_work(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    _write_project(repo, unit_count=2)
    output_dir = repo / "translated" / "directx"
    output_dir.mkdir(parents=True)
    previous_artifact = output_dir / "shader-0.hlsl"
    previous_remap = output_dir / "shader-0.source-remap.json"
    previous_artifact.write_text("previous artifact\n", encoding="utf-8")
    previous_remap.write_text('{"previous": true}\n', encoding="utf-8")
    unrelated_staging = output_dir / ".shader-0.hlsl.unrelated.tmp"
    unrelated_staging.mkdir()
    checkpoint_path = tmp_path / "translation-checkpoint.json"
    state = {
        "generation": 0,
        "terminated": set(),
        "shutdown": [],
        "cancelled": [],
        "submitted": [],
    }

    class RestartFuture:
        def __init__(self, request, result, *, generation, times_out):
            self.request = request
            self._result = result
            self.generation = generation
            self.times_out = times_out

        def result(self, timeout=None):
            if (
                timeout is not None
                and self.times_out
                and self.generation not in state["terminated"]
            ):
                raise project_pipeline.FutureTimeoutError()
            return self._result

        def done(self):
            return not self.times_out or self.generation in state["terminated"]

        def cancel(self):
            state["cancelled"].append(
                (self.generation, self.request.coordinate["source"])
            )
            return True

    class RestartExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            state["generation"] += 1
            self.generation = state["generation"]
            self.context = initargs[0]
            initializer(*initargs)

        def submit(self, function, request):
            result = function(request)
            assert result.publications
            assert all(
                f".{self.context.publication_staging_token}."
                in publication.staging_directory.name
                for publication in result.publications
            )
            state["submitted"].append((self.generation, request.coordinate["source"]))
            return RestartFuture(
                request,
                result,
                generation=self.generation,
                times_out=(
                    self.generation == 1
                    and request.coordinate["source"] == "shader-0.cgl"
                ),
            )

        def terminate_workers(self):
            state["terminated"].add(self.generation)

        def shutdown(self, *, wait):
            assert wait is True
            state["shutdown"].append(self.generation)

    monkeypatch.setattr(
        project_pipeline,
        "ProcessPoolExecutor",
        RestartExecutor,
    )

    report = translate_project(
        ProjectConfig(
            root=repo,
            targets=("directx",),
            output_dir="translated",
        ),
        format_output=False,
        max_workers=2,
        job_timeout_seconds=5,
        checkpoint_path=checkpoint_path,
    )
    payload = report.to_json()

    assert [
        (artifact["source"], artifact["status"]) for artifact in payload["artifacts"]
    ] == [
        ("shader-0.cgl", "failed"),
        ("shader-1.cgl", "translated"),
    ]
    timeout_artifact = payload["artifacts"][0]
    assert timeout_artifact["path"] == "translated/directx/shader-0.hlsl"
    assert timeout_artifact["sourceHash"]
    assert timeout_artifact["sourceSizeBytes"] > 0
    timeout_diagnostics = [
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["code"] == "project.translate.timeout"
    ]
    assert timeout_diagnostics == [
        {
            "severity": "error",
            "code": "project.translate.timeout",
            "message": timeout_artifact["error"],
            "location": {
                "file": "shader-0.cgl",
                "line": 1,
                "column": 1,
                "offset": 0,
                "length": 0,
                "endLine": 1,
                "endColumn": 1,
                "endOffset": 0,
            },
            "target": "directx",
            "sourceBackend": "cgl",
            "checkKind": "artifact",
            "details": {
                "timeoutSeconds": 5.0,
                "coordinate": {
                    "source": "shader-0.cgl",
                    "target": "directx",
                    "path": "translated/directx/shader-0.hlsl",
                },
            },
        }
    ]
    assert state == {
        "generation": 2,
        "terminated": {1},
        "shutdown": [1, 2],
        "cancelled": [(1, "shader-1.cgl")],
        "submitted": [
            (1, "shader-0.cgl"),
            (1, "shader-1.cgl"),
            (2, "shader-1.cgl"),
        ],
    }
    assert previous_artifact.read_text(encoding="utf-8") == "previous artifact\n"
    assert previous_remap.read_text(encoding="utf-8") == '{"previous": true}\n'
    assert unrelated_staging.is_dir()
    assert list(output_dir.glob(".*.tmp")) == [unrelated_staging]

    checkpoint = load_project_translation_checkpoint(checkpoint_path)
    assert checkpoint["state"] == "complete"
    assert checkpoint["plan"]["completedCount"] == 2
    assert [
        completion["coordinate"]["source"]
        for completion in checkpoint["plan"]["completed"]
    ] == ["shader-0.cgl", "shader-1.cgl"]

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    validation = project_api.validate_project_report(report_path)
    assert validation["success"] is False
    assert not any(
        diagnostic["code"] == "project.validate.invalid-report"
        for diagnostic in validation["diagnostics"]
    )


def test_job_timeout_isolates_worker_limit_one(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _write_project(repo, unit_count=1)
    state = {"constructed": 0, "result_timeout": None}
    monotonic_values = iter((10.0, 50.0))
    real_time = project_pipeline.time.time
    monkeypatch.setattr(
        project_pipeline,
        "time",
        SimpleNamespace(
            monotonic=lambda: next(monotonic_values),
            time=real_time,
        ),
    )

    class ImmediateFuture:
        def __init__(self, result):
            self._result = result

        def result(self, timeout=None):
            state["result_timeout"] = timeout
            return self._result

        def cancel(self):
            return False

    class ImmediateExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            state["constructed"] += 1
            initializer(*initargs)

        def submit(self, function, request):
            return ImmediateFuture(function(request))

        def shutdown(self, *, wait):
            assert wait is True

    monkeypatch.setattr(
        project_pipeline,
        "ProcessPoolExecutor",
        ImmediateExecutor,
    )

    report = translate_project(
        ProjectConfig(
            root=repo,
            targets=("directx",),
            output_dir="translated",
        ),
        format_output=False,
        max_workers=1,
        job_timeout_seconds=30,
    ).to_json()

    assert report["summary"]["translatedCount"] == 1
    assert state["constructed"] == 1
    assert state["result_timeout"] == 0.0


def test_real_process_job_timeout_returns_structured_report(tmp_path):
    repo = tmp_path / "repo"
    _write_project(repo, unit_count=1)

    payload = translate_project(
        ProjectConfig(
            root=repo,
            targets=("directx",),
            output_dir="translated",
        ),
        format_output=False,
        max_workers=1,
        job_timeout_seconds=1e-9,
    ).to_json()

    assert payload["summary"]["failedCount"] == 1
    assert payload["summary"]["translatedCount"] == 0
    assert payload["artifacts"][0]["status"] == "failed"
    assert payload["summary"]["diagnosticsByCode"] == {
        "project.translate.timeout": 1,
    }
    assert not (repo / payload["artifacts"][0]["path"]).exists()


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


@pytest.mark.parametrize(
    "value",
    (0, -1, True, float("inf"), float("-inf"), float("nan"), "30"),
)
def test_translate_project_rejects_invalid_job_timeout(tmp_path, value):
    repo = tmp_path / "repo"
    _write_project(repo, unit_count=1)

    with pytest.raises(
        ValueError,
        match="job_timeout_seconds must be a positive finite number",
    ):
        translate_project(repo, job_timeout_seconds=value)


def test_job_timeout_changes_checkpoint_invocation_identity(tmp_path):
    repo = tmp_path / "repo"
    _write_project(repo, unit_count=1)
    scan = project_pipeline.scan_project(
        ProjectConfig(
            root=repo,
            targets=("directx",),
            output_dir="translated",
        )
    )

    identity = project_pipeline._project_translation_checkpoint_identity(
        scan,
        ("directx",),
        format_output=False,
        validate=False,
        run_toolchains=False,
        job_timeout_seconds=30,
    )
    changed_identity = project_pipeline._project_translation_checkpoint_identity(
        scan,
        ("directx",),
        format_output=False,
        validate=False,
        run_toolchains=False,
        job_timeout_seconds=60,
    )

    assert identity["invocationHash"] != changed_identity["invocationHash"]


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
            "--job-timeout-seconds",
            "12.5",
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    assert observed["root"] == repo.resolve()
    assert observed["max_workers"] == 3
    assert observed["job_timeout_seconds"] == 12.5
