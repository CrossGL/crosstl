import textwrap

import pytest

import crosstl.project.pipeline as project_pipeline
from crosstl.project import translate_project

SIMPLE_CROSSL = textwrap.dedent("""
    shader PublicationShader {
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


def _write_project(repo):
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")


def _output_paths(repo):
    output_dir = repo / "out" / "opengl"
    return (
        output_dir / "simple.glsl",
        output_dir / "simple.source-remap.json",
    )


def _publication_directories(repo):
    output_dir = repo / "out" / "opengl"
    return list(output_dir.glob(".simple.glsl.*.tmp"))


def test_source_remap_failure_preserves_previous_artifact_pair(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    _write_project(repo)
    artifact_path, remap_path = _output_paths(repo)
    artifact_path.parent.mkdir(parents=True)
    previous_artifact = b"previous artifact\n"
    previous_remap = b'{"previous": true}\n'
    artifact_path.write_bytes(previous_artifact)
    remap_path.write_bytes(previous_remap)
    observed_staged_paths = []

    def fail_source_remap(path, _payload):
        observed_staged_paths.append(path)
        assert path != remap_path
        assert path.parent.parent == artifact_path.parent
        assert artifact_path.read_bytes() == previous_artifact
        assert remap_path.read_bytes() == previous_remap
        raise OSError("simulated source-remap write failure")

    monkeypatch.setattr(
        project_pipeline,
        "_write_source_remap_sidecar",
        fail_source_remap,
    )

    payload = translate_project(
        repo,
        targets=["opengl"],
        output_dir="out",
        format_output=False,
    ).to_json()

    assert payload["summary"]["failedCount"] == 1
    assert payload["artifacts"][0]["status"] == "failed"
    assert "simulated source-remap write failure" in payload["artifacts"][0]["error"]
    assert observed_staged_paths
    assert artifact_path.read_bytes() == previous_artifact
    assert remap_path.read_bytes() == previous_remap
    assert _publication_directories(repo) == []


def test_pre_generation_failure_preserves_previous_artifact_pair(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    _write_project(repo)
    baseline = translate_project(
        repo,
        targets=["opengl"],
        output_dir="out",
        format_output=False,
    ).to_json()
    assert baseline["summary"]["failedCount"] == 0
    artifact_path, remap_path = _output_paths(repo)
    previous_artifact = artifact_path.read_bytes()
    previous_remap = remap_path.read_bytes()

    def fail_pre_generation(*_args, **_kwargs):
        raise ValueError("simulated pre-generation failure")

    monkeypatch.setattr(
        project_pipeline,
        "_project_template_materialization_for_artifact",
        fail_pre_generation,
    )

    payload = translate_project(
        repo,
        targets=["opengl"],
        output_dir="out",
        format_output=False,
    ).to_json()

    assert payload["summary"]["failedCount"] == 1
    assert payload["artifacts"][0]["status"] == "failed"
    assert "simulated pre-generation failure" in payload["artifacts"][0]["error"]
    assert artifact_path.read_bytes() == previous_artifact
    assert remap_path.read_bytes() == previous_remap
    assert _publication_directories(repo) == []


@pytest.mark.parametrize("max_workers", (1, 2))
def test_publication_failure_restores_previous_artifact_pair(
    tmp_path,
    monkeypatch,
    max_workers,
):
    repo = tmp_path / "repo"
    _write_project(repo)
    baseline = translate_project(
        repo,
        targets=["opengl"],
        output_dir="out",
        format_output=False,
    ).to_json()
    assert baseline["summary"]["failedCount"] == 0
    artifact_path, remap_path = _output_paths(repo)
    previous_artifact = artifact_path.read_bytes()
    previous_remap = remap_path.read_bytes()
    (repo / "simple.cgl").write_text(
        SIMPLE_CROSSL.replace("vec4(input.position, 1.0)", "vec4(0.5, 0.5, 0.5, 1.0)"),
        encoding="utf-8",
    )

    original_replace = project_pipeline._replace_project_output_file
    failed = False
    publication_order = []

    def fail_artifact_publication(source, destination):
        nonlocal failed
        if (
            not failed
            and destination in {artifact_path, remap_path}
            and source.parent != artifact_path.parent
        ):
            publication_order.append(destination)
        if (
            not failed
            and destination == artifact_path
            and source.parent != artifact_path.parent
        ):
            assert publication_order == [remap_path, artifact_path]
            failed = True
            raise OSError("simulated artifact publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(
        project_pipeline,
        "_replace_project_output_file",
        fail_artifact_publication,
    )

    payload = translate_project(
        repo,
        targets=["opengl"],
        output_dir="out",
        format_output=False,
        max_workers=max_workers,
    ).to_json()

    assert failed is True
    assert publication_order == [remap_path, artifact_path]
    assert payload["summary"]["failedCount"] == 1
    assert payload["artifacts"][0]["status"] == "failed"
    assert "simulated artifact publication failure" in payload["artifacts"][0]["error"]
    assert artifact_path.read_bytes() == previous_artifact
    assert remap_path.read_bytes() == previous_remap
    assert _publication_directories(repo) == []


def test_initial_publication_failure_removes_partial_artifact_pair(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    _write_project(repo)
    artifact_path, remap_path = _output_paths(repo)
    original_replace = project_pipeline._replace_project_output_file
    failed = False

    def fail_artifact_publication(source, destination):
        nonlocal failed
        if not failed and destination == artifact_path:
            failed = True
            raise OSError("simulated initial artifact publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(
        project_pipeline,
        "_replace_project_output_file",
        fail_artifact_publication,
    )

    payload = translate_project(
        repo,
        targets=["opengl"],
        output_dir="out",
        format_output=False,
    ).to_json()

    assert failed is True
    assert payload["summary"]["failedCount"] == 1
    assert payload["artifacts"][0]["status"] == "failed"
    assert "simulated initial artifact publication failure" in (
        payload["artifacts"][0]["error"]
    )
    assert not artifact_path.exists()
    assert not remap_path.exists()
    assert _publication_directories(repo) == []
