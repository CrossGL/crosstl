import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import crosstl.project.pipeline as project_pipeline
from crosstl.project import (
    load_project_config,
    scan_project,
    translate_project,
    validate_project_report,
)

ROOT = Path(__file__).resolve().parents[2]


SIMPLE_CROSSL = textwrap.dedent("""
    shader RepoShader {
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


def test_support_external_corpus_manifest_documents_pinned_reductions():
    manifest = json.loads(
        (ROOT / "support" / "external-corpus.json").read_text(encoding="utf-8")
    )

    assert manifest["schemaVersion"] == 1
    assert manifest["entries"]
    source_backends = {entry["sourceBackend"] for entry in manifest["entries"]}
    assert {"opengl", "directx", "metal", "cuda", "hip"}.issubset(source_backends)
    for entry in manifest["entries"]:
        assert entry["id"]
        assert entry["path"]
        assert entry["repository"].startswith("https://github.com/")
        assert len(entry["commit"]) == 40
        assert entry["sourceUrl"].startswith(entry["repository"])
        assert entry["targets"] == ["cgl"]


def test_scan_project_discovers_supported_sources_and_ignores_default_unsupported(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (shader_dir / "main.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (shader_dir / "post.frag").write_text(
        "#version 450\nvoid main() { gl_FragColor = vec4(1.0); }\n",
        encoding="utf-8",
    )
    (shader_dir / "README.txt").write_text("not shader code", encoding="utf-8")

    scan = scan_project(repo)

    assert [unit.relative_path for unit in scan.units] == [
        "shaders/main.cgl",
        "shaders/post.frag",
    ]
    assert [unit.source_backend for unit in scan.units] == ["cgl", "opengl"]
    assert scan.skipped == []


def test_scan_project_reports_explicitly_included_unsupported_sources(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            include = ["**/*"]
            exclude = []
            """).strip(),
        encoding="utf-8",
    )
    (repo / "kernel.txt").write_text("not shader code", encoding="utf-8")

    config = load_project_config(repo)
    scan = scan_project(config)

    assert scan.units == []
    assert scan.skipped == [{"path": "kernel.txt", "reason": "unsupported-extension"}]
    assert {diagnostic.code for diagnostic in scan.diagnostics} == {
        "project.scan.empty",
        "project.scan.unsupported-source",
    }


def test_scan_project_reports_invalid_source_roots_without_hiding_valid_units(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    outside_dir = tmp_path / "outside"
    shader_dir.mkdir(parents=True)
    outside_dir.mkdir()
    (shader_dir / "main.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (outside_dir / "external.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders", "missing", "../outside"]
            include = ["**/*"]
            """).strip(),
        encoding="utf-8",
    )

    scan = scan_project(load_project_config(repo))
    payload = scan.to_report().to_json()

    assert [unit.relative_path for unit in scan.units] == ["shaders/main.cgl"]
    assert [
        unit.relative_path for unit in scan.units if "external" in unit.path.name
    ] == []
    diagnostics = {
        diagnostic["code"]: diagnostic for diagnostic in payload["diagnostics"]
    }
    assert set(diagnostics) == {
        "project.config.source-root-outside-project",
        "project.scan.missing-source-root",
    }
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 1, "error": 1}
    assert diagnostics["project.scan.missing-source-root"]["location"]["file"] == (
        "crosstl.toml"
    )
    assert diagnostics["project.config.source-root-outside-project"][
        "missingCapabilities"
    ] == ["repo.scan"]


def test_scan_project_accepts_repository_relative_include_patterns(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    other_dir = repo / "other"
    shader_dir.mkdir(parents=True)
    other_dir.mkdir()
    (shader_dir / "main.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (other_dir / "ignored.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]
            include = ["shaders/**/*.cgl", "other/**/*.cgl"]
            exclude = []
            """).strip(),
        encoding="utf-8",
    )

    scan = scan_project(load_project_config(repo))

    assert [unit.relative_path for unit in scan.units] == ["shaders/main.cgl"]
    assert scan.skipped == []
    assert {diagnostic.code for diagnostic in scan.diagnostics} == set()


def test_scan_project_reports_include_patterns_outside_project(tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (outside / "external.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    absolute_pattern = (outside / "*.cgl").as_posix()
    (repo / "crosstl.toml").write_text(
        textwrap.dedent(f"""
            [project]
            include = ["{absolute_pattern}", "../outside/*.cgl"]
            exclude = []
            """).strip(),
        encoding="utf-8",
    )

    scan = scan_project(load_project_config(repo))
    payload = scan.to_report().to_json()

    assert scan.units == []
    assert payload["summary"]["diagnosticCounts"] == {
        "note": 0,
        "warning": 1,
        "error": 2,
    }
    diagnostics = {diagnostic["message"] for diagnostic in payload["diagnostics"]}
    assert any(absolute_pattern in message for message in diagnostics)
    assert any("../outside/*.cgl" in message for message in diagnostics)
    assert [
        diagnostic["code"]
        for diagnostic in payload["diagnostics"]
        if diagnostic["severity"] == "error"
    ] == [
        "project.config.include-pattern-outside-project",
        "project.config.include-pattern-outside-project",
    ]
    assert all(
        diagnostic["missingCapabilities"] == ["repo.scan"]
        for diagnostic in payload["diagnostics"]
        if diagnostic["severity"] == "error"
    )


def test_scan_report_normalizes_and_deduplicates_targets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    payload = scan_project(repo).to_report(targets=["OpenGL", "opengl"]).to_json()

    assert payload["project"]["targets"] == ["opengl"]
    assert payload["summary"]["targetCount"] == 1
    assert payload["migration"]["actions"][0]["targets"] == ["opengl"]


def test_scan_report_records_documented_migration_actions(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    payload = scan_project(repo).to_report(targets=["opengl"]).to_json()

    assert payload["migration"]["scope"] == "shader-kernel-translation"
    assert payload["migration"]["actions"] == [
        {
            "kind": "manual-runtime-integration",
            "severity": "note",
            "message": (
                "CrossTL translated shader/kernel source artifacts only; review "
                "host runtime API calls, resource binding setup, build scripts, "
                "and backend framework integration separately."
            ),
            "targets": ["opengl"],
        }
    ]


def test_scan_report_records_unsupported_targets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    payload = scan_project(repo).to_report(targets=["not-a-backend"]).to_json()

    assert payload["project"]["targets"] == ["not-a-backend"]
    assert payload["summary"]["diagnosticCounts"]["error"] == 1
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.config.unsupported-target"
    assert diagnostic["target"] == "not-a-backend"
    assert diagnostic["missingCapabilities"] == ["target.backend"]
    assert "Supported targets:" in diagnostic["message"]


def test_project_config_loads_overrides_and_variant_metadata(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "gpu"
    shader_dir.mkdir(parents=True)
    (shader_dir / "include").mkdir()
    (shader_dir / "kernel.shader").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["gpu"]
            include = ["**/*"]
            exclude = []
            targets = ["metal"]
            output_dir = "generated"
            include_dirs = ["gpu/include"]

            [project.sources]
            "gpu/*.shader" = "cgl"

            [project.defines]
            USE_FAST_PATH = "1"

            [project.variants.debug]
            USE_FAST_PATH = "0"
            """).strip(),
        encoding="utf-8",
    )

    config = load_project_config(repo)
    scan = scan_project(config)

    assert config.targets == ["metal"]
    assert config.output_dir == "generated"
    assert config.include_dirs == ["gpu/include"]
    assert config.defines == {"USE_FAST_PATH": "1"}
    assert config.variants == {"debug": {"USE_FAST_PATH": "0"}}
    assert [(unit.relative_path, unit.source_backend) for unit in scan.units] == [
        ("gpu/kernel.shader", "cgl")
    ]
    assert scan.diagnostics == []
    payload = scan.to_report(targets=config.targets).to_json()
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 0}
    assert payload["project"]["defines"] == {"USE_FAST_PATH": "1"}
    assert payload["project"]["defineCount"] == 1
    assert payload["project"]["variants"] == {"debug": {"USE_FAST_PATH": "0"}}
    assert payload["project"]["variantCount"] == 1


def test_scan_project_reports_missing_include_dirs_without_hiding_units(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (shader_dir / "main.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]
            include_dirs = ["missing-includes"]
            """).strip(),
        encoding="utf-8",
    )

    scan = scan_project(load_project_config(repo))
    payload = scan.to_report().to_json()

    assert [unit.relative_path for unit in scan.units] == ["shaders/main.cgl"]
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 1, "error": 0}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.config.missing-include-dir"
    assert diagnostic["location"]["file"] == "crosstl.toml"
    assert diagnostic["missingCapabilities"] == ["include.resolution"]
    assert "missing-includes" in diagnostic["message"]


def test_scan_project_reports_include_dirs_outside_project_without_hiding_units(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    outside_dir = tmp_path / "outside-includes"
    shader_dir.mkdir(parents=True)
    outside_dir.mkdir()
    (shader_dir / "main.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]
            include_dirs = ["../outside-includes"]
            """).strip(),
        encoding="utf-8",
    )

    scan = scan_project(load_project_config(repo))
    payload = scan.to_report().to_json()

    assert [unit.relative_path for unit in scan.units] == ["shaders/main.cgl"]
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 1, "error": 0}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.config.include-dir-outside-project"
    assert diagnostic["location"]["file"] == "crosstl.toml"
    assert diagnostic["missingCapabilities"] == ["include.resolution"]
    assert "../outside-includes" in diagnostic["message"]


def test_project_config_rejects_malformed_variant_entries(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project.variants]
            debug = "USE_FAST_PATH=0"
            """).strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"crosstl\.toml \[project\.variants\.debug\] must be a table",
    ):
        load_project_config(repo)


def test_translate_project_honors_source_backend_overrides(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "gpu"
    shader_dir.mkdir(parents=True)
    (shader_dir / "kernel.shader").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["gpu"]
            targets = ["opengl"]
            output_dir = "translated"

            [project.sources]
            "gpu/*.shader" = "cgl"
            """).strip(),
        encoding="utf-8",
    )

    config = load_project_config(repo)
    report = translate_project(config)
    payload = report.to_json()

    output = repo / "translated" / "opengl" / "gpu" / "kernel.glsl"
    assert output.exists()
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    assert payload["units"][0]["sourceOverride"] == "cgl"
    assert payload["artifacts"][0]["sourceBackend"] == "cgl"
    assert payload["artifacts"][0]["path"] == "translated/opengl/gpu/kernel.glsl"


def test_scan_project_reports_unsupported_source_overrides(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "gpu"
    shader_dir.mkdir(parents=True)
    (shader_dir / "kernel.shader").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["gpu"]

            [project.sources]
            "gpu/*.shader" = "unknown-backend"
            """).strip(),
        encoding="utf-8",
    )

    scan = scan_project(load_project_config(repo))
    payload = scan.to_report().to_json()

    assert scan.units == []
    assert scan.skipped == [
        {
            "path": "gpu/kernel.shader",
            "reason": "unsupported-source-override",
            "sourceOverride": "unknown-backend",
        }
    ]
    assert [diagnostic["code"] for diagnostic in payload["diagnostics"]] == [
        "project.config.unsupported-source-override",
        "project.scan.empty",
    ]
    assert payload["diagnostics"][0]["severity"] == "error"
    assert payload["diagnostics"][0]["location"]["file"] == "crosstl.toml"
    assert payload["diagnostics"][0]["missingCapabilities"] == ["source.override"]
    assert "unknown-backend" in payload["diagnostics"][0]["message"]


def test_scan_project_reports_source_override_patterns_outside_project(tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (outside / "kernel.shader").write_text(SIMPLE_CROSSL, encoding="utf-8")
    absolute_pattern = (outside / "*.shader").as_posix()
    (repo / "crosstl.toml").write_text(
        textwrap.dedent(f"""
            [project.sources]
            "{absolute_pattern}" = "cgl"
            "../outside/*.shader" = "cgl"
            """).strip(),
        encoding="utf-8",
    )

    scan = scan_project(load_project_config(repo))
    payload = scan.to_report().to_json()

    assert [unit.relative_path for unit in scan.units] == ["simple.cgl"]
    assert payload["summary"]["diagnosticCounts"] == {
        "note": 0,
        "warning": 0,
        "error": 2,
    }
    diagnostics = {diagnostic["message"] for diagnostic in payload["diagnostics"]}
    assert any(absolute_pattern in message for message in diagnostics)
    assert any("../outside/*.shader" in message for message in diagnostics)
    assert [diagnostic["code"] for diagnostic in payload["diagnostics"]] == [
        "project.config.source-override-pattern-outside-project",
        "project.config.source-override-pattern-outside-project",
    ]
    assert all(
        diagnostic["missingCapabilities"] == ["source.override"]
        for diagnostic in payload["diagnostics"]
    )


def test_translate_project_applies_include_dirs_and_defines(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    include_dir = repo / "includes"
    shader_dir.mkdir(parents=True)
    include_dir.mkdir(parents=True)
    (include_dir / "shared.glsl").write_text(
        "vec4 project_color() { return vec4(1.0); }\n",
        encoding="utf-8",
    )
    (shader_dir / "main.frag").write_text(
        textwrap.dedent("""
            #version 450
            #ifdef USE_PROJECT_SHARED
            #include <shared.glsl>
            #else
            #error "missing project define"
            #endif

            layout(location = 0) out vec4 outColor;

            void main()
            {
                outColor = project_color();
            }
            """).strip(),
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]
            targets = ["cgl"]
            output_dir = "translated"
            include_dirs = ["includes"]

            [project.defines]
            USE_PROJECT_SHARED = "1"
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    output = repo / "translated" / "cgl" / "shaders" / "main.cgl"

    assert output.exists()
    assert payload["summary"]["translatedCount"] == 1
    assert payload["diagnosticCounts"]["error"] == 0
    assert payload["diagnosticCounts"]["warning"] == 0
    assert "project_color" in output.read_text(encoding="utf-8")


def test_translate_project_expands_named_variants_with_merged_defines(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["opengl"]
            output_dir = "translated"

            [project.defines]
            MODE = "base"
            USE_FAST_PATH = "1"

            [project.variants.debug]
            MODE = "debug"

            [project.variants.release]
            """).strip(),
        encoding="utf-8",
    )

    def write_defines(
        file_path,
        backend="cgl",
        save_shader=None,
        format_output=True,
        source_backend=None,
        *,
        include_paths=None,
        defines=None,
    ):
        del file_path, backend, format_output, source_backend, include_paths
        text = json.dumps({"defines": dict(defines or {})}, sort_keys=True)
        Path(save_shader).write_text(text, encoding="utf-8")
        return text

    monkeypatch.setattr(project_pipeline, "translate", write_defines)

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "translated" / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)

    assert validation["success"] is True
    assert payload["summary"]["artifactCount"] == 2
    assert payload["summary"]["translatedCount"] == 2
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 0}
    assert [artifact["variant"] for artifact in payload["artifacts"]] == [
        "debug",
        "release",
    ]
    assert [artifact["path"] for artifact in payload["artifacts"]] == [
        "translated/opengl/debug/simple.glsl",
        "translated/opengl/release/simple.glsl",
    ]
    assert validation["validation"]["artifacts"] == [
        {
            "source": "simple.cgl",
            "target": "opengl",
            "path": "translated/opengl/debug/simple.glsl",
            "exists": True,
            "status": "ok",
            "variant": "debug",
        },
        {
            "source": "simple.cgl",
            "target": "opengl",
            "path": "translated/opengl/release/simple.glsl",
            "exists": True,
            "status": "ok",
            "variant": "release",
        },
    ]
    assert json.loads(
        (repo / "translated" / "opengl" / "debug" / "simple.glsl").read_text(
            encoding="utf-8"
        )
    )["defines"] == {"MODE": "debug", "USE_FAST_PATH": "1"}
    assert json.loads(
        (repo / "translated" / "opengl" / "release" / "simple.glsl").read_text(
            encoding="utf-8"
        )
    )["defines"] == {"MODE": "base", "USE_FAST_PATH": "1"}


def test_validate_project_report_rejects_artifacts_with_undeclared_variants(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            output_dir = "out"

            [project.variants.debug]
            MODE = "debug"
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    payload["artifacts"][0]["variant"] = "profile"
    report_path = repo / "out" / "portability-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].variant must be listed in project.variants" in (
        diagnostic["message"]
    )


def test_translate_project_preserves_relative_paths_and_reports_artifacts(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders" / "graphics"
    shader_dir.mkdir(parents=True)
    (shader_dir / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["opengl"], output_dir="translated")
    payload = report.to_json()

    output = repo / "translated" / "opengl" / "shaders" / "graphics" / "simple.glsl"
    assert output.exists()
    assert payload["kind"] == "crosstl-project-portability-report"
    assert payload["schemaVersion"] == 1
    assert payload["summary"]["unitCount"] == 1
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    assert payload["summary"]["unitsBySourceBackend"] == {"cgl": 1}
    assert payload["summary"]["artifactsByTarget"] == {
        "opengl": {
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
        }
    }
    assert payload["artifacts"][0]["source"] == "shaders/graphics/simple.cgl"
    assert payload["artifacts"][0]["target"] == "opengl"
    assert payload["artifacts"][0]["path"] == (
        "translated/opengl/shaders/graphics/simple.glsl"
    )
    assert payload["artifacts"][0]["sourceHash"]["algorithm"] == "sha256"
    assert payload["migration"]["nonGoals"] == [
        "automatic runtime API migration",
        "application build-system rewrites",
        "backend framework integration",
    ]


def test_translate_project_records_file_granularity_source_maps(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()

    artifact = payload["artifacts"][0]
    source_map = artifact["sourceMap"]

    assert payload["summary"]["sourceMapCount"] == 1
    assert payload["summary"]["fineGrainedSourceMapCount"] == 0
    assert source_map["schemaVersion"] == 1
    assert source_map["kind"] == "crosstl-artifact-source-map"
    assert source_map["mappingGranularity"] == "file"
    assert source_map["target"] == "cgl"
    assert source_map["source"]["file"] == "simple.cgl"
    assert source_map["generated"]["file"] == "out/cgl/simple.cgl"
    assert source_map["source"]["length"] == len(SIMPLE_CROSSL)
    assert source_map["generated"]["length"] == len(SIMPLE_CROSSL)
    assert source_map["mappings"] == [
        {
            "source": source_map["source"],
            "generated": source_map["generated"],
        }
    ]


def test_translate_project_records_external_corpus_manifest_summary(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "corpus.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "name": "Reduced project corpus",
                "entries": [
                    {
                        "id": "repo/simple",
                        "path": "simple.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                        "repository": "https://github.com/example/project",
                        "commit": "0" * 40,
                    },
                    {
                        "id": "repo/missing-hlsl",
                        "path": "missing.hlsl",
                        "sourceBackend": "directx",
                        "targets": ["cgl", "opengl"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            external_corpus_manifest = "corpus.json"
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)

    assert validation["success"] is True
    assert payload["project"]["externalCorpusManifest"] == "corpus.json"
    external_corpus = payload["externalCorpus"]
    assert external_corpus["schemaVersion"] == 1
    assert external_corpus["manifest"] == "corpus.json"
    assert external_corpus["status"] == "ok"
    assert external_corpus["name"] == "Reduced project corpus"
    assert external_corpus["summary"] == {
        "entryCount": 2,
        "presentCount": 1,
        "missingCount": 1,
        "discoveredUnitCount": 1,
        "undiscoveredPresentCount": 0,
        "entriesBySourceBackend": {"cgl": 1, "directx": 1},
        "entriesByTarget": {"cgl": 2, "opengl": 1},
        "artifactsByTarget": {
            "cgl": {
                "artifactCount": 1,
                "translatedCount": 1,
                "failedCount": 0,
            }
        },
    }
    assert external_corpus["entries"][0] == {
        "id": "repo/simple",
        "path": "simple.cgl",
        "sourceBackend": "cgl",
        "targets": ["cgl"],
        "present": True,
        "discovered": True,
        "artifactCount": 1,
        "translatedCount": 1,
        "failedCount": 0,
        "repository": "https://github.com/example/project",
        "commit": "0" * 40,
    }
    assert external_corpus["entries"][1] == {
        "id": "repo/missing-hlsl",
        "path": "missing.hlsl",
        "sourceBackend": "directx",
        "targets": ["cgl", "opengl"],
        "present": False,
        "discovered": False,
        "artifactCount": 0,
        "translatedCount": 0,
        "failedCount": 0,
    }


def test_validate_project_report_accepts_generated_source_maps(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report_path = repo / "portability-report.json"

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    report.write_json(report_path)

    payload = validate_project_report(report_path)

    assert payload["success"] is True
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 0}
    assert payload["validation"]["artifacts"] == [
        {
            "source": "simple.cgl",
            "target": "cgl",
            "path": "out/cgl/simple.cgl",
            "exists": True,
            "status": "ok",
        }
    ]


def test_translate_project_normalizes_and_deduplicates_targets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["OpenGL", "opengl"], output_dir="out")
    payload = report.to_json()

    assert payload["project"]["targets"] == ["opengl"]
    assert payload["summary"]["targetCount"] == 1
    assert payload["summary"]["artifactCount"] == 1
    assert payload["artifacts"][0]["target"] == "opengl"
    assert payload["artifacts"][0]["path"] == "out/opengl/simple.glsl"
    assert (repo / "out" / "opengl" / "simple.glsl").exists()


def test_scan_project_ignores_configured_output_dir(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (shader_dir / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["opengl"]
            output_dir = "translated"
            """).strip(),
        encoding="utf-8",
    )

    config = load_project_config(repo)
    report = translate_project(config)
    output = repo / "translated" / "opengl" / "shaders" / "simple.glsl"
    assert output.exists()
    assert report.to_json()["summary"]["translatedCount"] == 1

    scan = scan_project(config)

    assert [unit.relative_path for unit in scan.units] == ["shaders/simple.cgl"]


def test_scan_project_reports_output_dir_outside_project(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            output_dir = "../outside"
            """).strip(),
        encoding="utf-8",
    )

    scan = scan_project(load_project_config(repo))
    payload = scan.to_report().to_json()

    assert [unit.relative_path for unit in scan.units] == ["simple.cgl"]
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 1}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.config.output-dir-outside-project"
    assert diagnostic["location"]["file"] == "crosstl.toml"
    assert diagnostic["missingCapabilities"] == ["artifact.manifest"]
    assert "../outside" in diagnostic["message"]


def test_translate_project_rejects_output_dir_outside_project_without_writing(
    tmp_path,
):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["opengl"]
            output_dir = "../outside"
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()

    assert outside.exists() is False
    assert payload["summary"]["artifactCount"] == 1
    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 1
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 1}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.config.output-dir-outside-project"
    artifact = payload["artifacts"][0]
    assert artifact["status"] == "failed"
    assert artifact["source"] == "simple.cgl"
    assert artifact["target"] == "opengl"
    assert Path(artifact["path"]).resolve() == (outside / "opengl" / "simple.glsl")
    assert artifact["error"] == (
        "Configured output directory resolves outside the repository; "
        "artifact was not written."
    )


def test_translate_project_validation_records_artifacts_and_toolchains(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(
        repo,
        targets=["opengl"],
        output_dir="out",
        validate=True,
    )
    payload = report.to_json()

    assert payload["validation"]["artifacts"] == [
        {
            "source": "simple.cgl",
            "target": "opengl",
            "path": "out/opengl/simple.glsl",
            "exists": True,
            "status": "ok",
        }
    ]
    assert payload["validation"]["toolchains"][0]["target"] == "opengl"
    assert payload["validation"]["toolchains"][0]["status"] in {
        "available",
        "unavailable",
    }
    assert payload["diagnosticCounts"]["error"] == 0


def test_validate_project_report_preserves_source_diagnostics(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "portability-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": [],
                    "outputDir": "out",
                },
                "artifacts": [],
                "diagnostics": [
                    {
                        "severity": "error",
                        "code": "project.config.source-root-outside-project",
                        "message": (
                            "Configured source root resolves outside the repository."
                        ),
                        "location": {
                            "file": "crosstl.toml",
                            "line": 1,
                            "column": 1,
                            "offset": 0,
                            "length": 0,
                            "endLine": 1,
                            "endColumn": 1,
                            "endOffset": 0,
                        },
                        "missingCapabilities": ["repo.scan"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 1}
    assert payload["diagnostics"][0]["code"] == (
        "project.config.source-root-outside-project"
    )
    assert payload["diagnostics"][0]["missingCapabilities"] == ["repo.scan"]


def test_validate_project_report_returns_structured_invalid_report_diagnostics(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "invalid-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 99,
                "kind": "not-a-project-report",
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 1}
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert diagnostic["location"]["file"] == str(report_path)
    assert diagnostic["missingCapabilities"] == ["artifact.manifest"]
    assert "expected schemaVersion 1" in diagnostic["message"]
    assert "expected kind crosstl-project-portability-report" in diagnostic["message"]
    assert "missing project object" in diagnostic["message"]


def test_validate_project_report_rejects_invalid_project_metadata(tmp_path):
    report_path = tmp_path / "invalid-project-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": "relative/repo",
                    "targets": "opengl",
                    "outputDir": [],
                },
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 1}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "project.root must be an absolute path" in diagnostic["message"]
    assert "project.targets must be a list of backend names" in diagnostic["message"]
    assert "project.outputDir must be a string" in diagnostic["message"]


def test_validate_project_report_rejects_malformed_project_config_metadata(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "invalid-project-config-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "config": [],
                    "sourceRoots": "shaders",
                    "includePatterns": ["*.cgl", 1],
                    "excludePatterns": [False],
                    "targets": ["opengl"],
                    "outputDir": "out",
                    "includeDirs": "include",
                    "defines": {"USE_FAST_PATH": 1},
                    "defineCount": 2,
                    "variants": {"debug": "not a define map", "": {"MODE": 1}},
                    "variantCount": "1",
                },
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "project.config must be a string or null" in diagnostic["message"]
    assert "project.sourceRoots must be a list of strings" in diagnostic["message"]
    assert "project.includePatterns must be a list of strings" in (
        diagnostic["message"]
    )
    assert "project.excludePatterns must be a list of strings" in (
        diagnostic["message"]
    )
    assert "project.includeDirs must be a list of strings" in diagnostic["message"]
    assert "project.defines values must be strings" in diagnostic["message"]
    assert "project.defineCount must match project.defines" in diagnostic["message"]
    assert "project.variants keys must be non-empty strings" in diagnostic["message"]
    assert "project.variants.debug must be an object" in diagnostic["message"]
    assert "project.variants. values must be strings" in diagnostic["message"]
    assert "project.variantCount must be a non-negative integer" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_malformed_unit_and_skipped_records(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "invalid-unit-record-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                },
                "units": [
                    {
                        "id": "other.cgl",
                        "path": "../simple.cgl",
                        "sourceBackend": "",
                        "extension": [],
                        "sourceOverride": "",
                    },
                    "not a unit",
                ],
                "skipped": [
                    {
                        "path": "C:/tmp/outside.txt",
                        "reason": "",
                        "sourceOverride": [],
                    },
                    "not a skipped record",
                ],
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "units[0].path must be repository-relative" in diagnostic["message"]
    assert "units[0].id must match units[0].path" in diagnostic["message"]
    assert "units[0].sourceBackend must be a string" in diagnostic["message"]
    assert "units[0].extension must be a string" in diagnostic["message"]
    assert "units[0].sourceOverride must be a string" in diagnostic["message"]
    assert "units[1] must be an object" in diagnostic["message"]
    assert "skipped[0].path must be repository-relative" in diagnostic["message"]
    assert "skipped[0].reason must be a string" in diagnostic["message"]
    assert "skipped[0].sourceOverride must be a string" in diagnostic["message"]
    assert "skipped[1] must be an object" in diagnostic["message"]


def test_validate_project_report_rejects_malformed_generator_and_validation_records(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "invalid-validation-record-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "generator": {
                    "name": "",
                    "pipeline": "single-file-translate",
                },
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                },
                "artifacts": [],
                "validation": {
                    "toolchains": [
                        {
                            "target": "",
                            "status": "ready",
                            "tools": [
                                {
                                    "name": "",
                                    "path": [],
                                    "available": "yes",
                                },
                                "not a tool",
                            ],
                            "message": "",
                        },
                        "not a toolchain",
                    ],
                    "artifacts": [
                        {
                            "source": "",
                            "target": "",
                            "path": "",
                            "exists": "yes",
                            "status": "missing",
                            "variant": "",
                        },
                        "not an artifact check",
                    ],
                    "toolchainRuns": [
                        {
                            "source": "",
                            "target": "",
                            "path": "",
                            "variant": "",
                            "command": ["glslangValidator", ""],
                            "returncode": True,
                            "status": "ok",
                            "stdout": [],
                            "stderr": None,
                        },
                        {
                            "source": "simple.cgl",
                            "target": "opengl",
                            "path": "out/opengl/simple.glsl",
                            "command": ["glslangValidator"],
                            "returncode": 2,
                            "status": "ok",
                            "stdout": "",
                            "stderr": "",
                        },
                        "not a toolchain run",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "generator.name must be a string" in diagnostic["message"]
    assert "generator.pipeline must be project-porting" in diagnostic["message"]
    assert "validation.toolchains[0].target must be a string" in (diagnostic["message"])
    assert (
        "validation.toolchains[0].status must be available, unavailable, or "
        "not-configured"
    ) in diagnostic["message"]
    assert "validation.toolchains[0].tools[0].name must be a string" in (
        diagnostic["message"]
    )
    assert "validation.toolchains[0].tools[0].path must be a string or null" in (
        diagnostic["message"]
    )
    assert "validation.toolchains[0].tools[0].available must be a boolean" in (
        diagnostic["message"]
    )
    assert "validation.toolchains[0].tools[1] must be an object" in (
        diagnostic["message"]
    )
    assert "validation.toolchains[0].message must be a string" in (
        diagnostic["message"]
    )
    assert "validation.toolchains[1] must be an object" in diagnostic["message"]
    assert "validation.artifacts[0].source must be a string" in diagnostic["message"]
    assert "validation.artifacts[0].target must be a string" in diagnostic["message"]
    assert "validation.artifacts[0].path must be a string" in diagnostic["message"]
    assert "validation.artifacts[0].exists must be a boolean" in (diagnostic["message"])
    assert "validation.artifacts[0].status must be ok or failed" in (
        diagnostic["message"]
    )
    assert "validation.artifacts[0].variant must be a string" in (diagnostic["message"])
    assert "validation.artifacts[1] must be an object" in diagnostic["message"]
    assert "validation.toolchainRuns[0].source must be a string" in (
        diagnostic["message"]
    )
    assert "validation.toolchainRuns[0].target must be a string" in (
        diagnostic["message"]
    )
    assert "validation.toolchainRuns[0].path must be a string" in (
        diagnostic["message"]
    )
    assert "validation.toolchainRuns[0].variant must be a string" in (
        diagnostic["message"]
    )
    assert "validation.toolchainRuns[0].command must be a list of strings" in (
        diagnostic["message"]
    )
    assert "validation.toolchainRuns[0].returncode must be an integer" in (
        diagnostic["message"]
    )
    assert "validation.toolchainRuns[0].stdout must be a string" in (
        diagnostic["message"]
    )
    assert "validation.toolchainRuns[0].stderr must be a string" in (
        diagnostic["message"]
    )
    assert "validation.toolchainRuns[1].status must match returncode" in (
        diagnostic["message"]
    )
    assert "validation.toolchainRuns[2] must be an object" in diagnostic["message"]


def test_validate_project_report_rejects_malformed_external_corpus_records(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "invalid-external-corpus-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                    "externalCorpusManifest": "corpus.json",
                },
                "artifacts": [],
                "externalCorpus": {
                    "schemaVersion": 2,
                    "manifest": "",
                    "status": "ready",
                    "name": "",
                    "entries": [
                        {
                            "id": "",
                            "path": "../outside.cgl",
                            "sourceBackend": "",
                            "targets": ["opengl", 1],
                            "present": "yes",
                            "discovered": "no",
                            "artifactCount": -1,
                            "translatedCount": "0",
                            "failedCount": False,
                            "repository": "",
                        },
                        "not an entry",
                    ],
                    "summary": {
                        "entryCount": "2",
                        "presentCount": 2,
                        "missingCount": 0,
                        "discoveredUnitCount": 2,
                        "undiscoveredPresentCount": 1,
                        "entriesBySourceBackend": {},
                        "entriesByTarget": {},
                        "artifactsByTarget": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "project.externalCorpusManifest must be a string or null" not in (
        diagnostic["message"]
    )
    assert "externalCorpus.schemaVersion must be 1" in diagnostic["message"]
    assert "externalCorpus.manifest must be a string" in diagnostic["message"]
    assert (
        "externalCorpus.status must be ok, missing, invalid, or outside-project"
    ) in diagnostic["message"]
    assert "externalCorpus.name must be a string" in diagnostic["message"]
    assert "externalCorpus.entries[0].id must be a string" in diagnostic["message"]
    assert "externalCorpus.entries[0].path must be repository-relative" in (
        diagnostic["message"]
    )
    assert "externalCorpus.entries[0].sourceBackend must be a string" in (
        diagnostic["message"]
    )
    assert "externalCorpus.entries[0].targets must be a list of strings" in (
        diagnostic["message"]
    )
    assert "externalCorpus.entries[0].present must be a boolean" in (
        diagnostic["message"]
    )
    assert "externalCorpus.entries[0].discovered must be a boolean" in (
        diagnostic["message"]
    )
    assert (
        "externalCorpus.entries[0].artifactCount must be a non-negative integer"
    ) in diagnostic["message"]
    assert (
        "externalCorpus.entries[0].translatedCount must be a non-negative integer"
    ) in diagnostic["message"]
    assert (
        "externalCorpus.entries[0].failedCount must be a non-negative integer"
    ) in diagnostic["message"]
    assert "externalCorpus.entries[0].repository must be a string" in (
        diagnostic["message"]
    )
    assert "externalCorpus.entries[1] must be an object" in diagnostic["message"]
    assert "externalCorpus.summary.entryCount must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "externalCorpus.summary.presentCount must match externalCorpus.entries" in (
        diagnostic["message"]
    )
    assert (
        "externalCorpus.summary.entriesBySourceBackend must match "
        "externalCorpus.entries"
    ) in diagnostic["message"]
    assert "externalCorpus.summary.artifactsByTarget must be an object" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_malformed_artifact_records(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "invalid-artifact-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                },
                "artifacts": [
                    {
                        "source": "simple.cgl",
                        "target": "opengl",
                        "status": "unknown",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].path must be a string" in diagnostic["message"]
    assert "artifacts[0].status must be translated or failed" in (diagnostic["message"])


def test_validate_project_report_rejects_malformed_source_maps(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "invalid-source-map-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                },
                "artifacts": [
                    {
                        "source": "simple.cgl",
                        "target": "opengl",
                        "path": "out/opengl/simple.glsl",
                        "status": "translated",
                        "sourceMap": {
                            "schemaVersion": 2,
                            "kind": "wrong-kind",
                            "mappingGranularity": "line",
                            "target": "metal",
                            "source": {"file": "simple.cgl", "line": 1},
                            "generated": "out/opengl/simple.glsl",
                            "mappings": "invalid",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].sourceMap.schemaVersion must be 1" in diagnostic["message"]
    assert "artifacts[0].sourceMap.kind must be crosstl-artifact-source-map" in (
        diagnostic["message"]
    )
    assert "artifacts[0].sourceMap.mappingGranularity must be file" in (
        diagnostic["message"]
    )
    assert "artifacts[0].sourceMap.target must match artifacts[0].target" in (
        diagnostic["message"]
    )
    assert "artifacts[0].sourceMap.generated must be an object" in (
        diagnostic["message"]
    )
    assert "artifacts[0].sourceMap.mappings must be a list" in diagnostic["message"]


def test_validate_project_report_rejects_inconsistent_source_map_anchors(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "inconsistent-source-map-report.json"
    source_span = {
        "file": "other.cgl",
        "line": 1,
        "column": 1,
        "offset": 0,
        "length": 12,
        "endLine": 1,
        "endColumn": 13,
        "endOffset": 12,
    }
    generated_span = {
        "file": "out/opengl/other.glsl",
        "line": 1,
        "column": 1,
        "offset": 0,
        "length": 24,
        "endLine": 1,
        "endColumn": 25,
        "endOffset": 24,
    }
    mapping_source_span = dict(source_span, file="simple.cgl")
    mapping_generated_span = dict(generated_span, file="out/opengl/simple.glsl")
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                },
                "artifacts": [
                    {
                        "source": "simple.cgl",
                        "target": "opengl",
                        "path": "out/opengl/simple.glsl",
                        "status": "translated",
                        "sourceMap": {
                            "schemaVersion": 1,
                            "kind": "crosstl-artifact-source-map",
                            "mappingGranularity": "file",
                            "target": "opengl",
                            "source": source_span,
                            "generated": generated_span,
                            "mappings": [
                                {
                                    "source": mapping_source_span,
                                    "generated": mapping_generated_span,
                                }
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].sourceMap.source.file must match artifacts[0].source" in (
        diagnostic["message"]
    )
    assert "artifacts[0].sourceMap.generated.file must match artifacts[0].path" in (
        diagnostic["message"]
    )
    assert (
        "artifacts[0].sourceMap.mappings[0].source must match "
        "artifacts[0].sourceMap.source"
    ) in diagnostic["message"]
    assert (
        "artifacts[0].sourceMap.mappings[0].generated must match "
        "artifacts[0].sourceMap.generated"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_malformed_artifact_metadata(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "invalid-artifact-metadata-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                },
                "artifacts": [
                    {
                        "source": "simple.cgl",
                        "sourceBackend": "",
                        "target": "opengl",
                        "path": "out/opengl/simple.glsl",
                        "status": "translated",
                        "variant": "",
                        "sourceHash": {
                            "algorithm": "md5",
                            "value": "A" * 64,
                        },
                        "provenance": {
                            "pipeline": "",
                            "intermediate": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].sourceBackend must be a string" in diagnostic["message"]
    assert "artifacts[0].variant must be a string" in diagnostic["message"]
    assert "artifacts[0].sourceHash.algorithm must be sha256" in (diagnostic["message"])
    assert (
        "artifacts[0].sourceHash.value must be a lowercase 64-character hex digest"
        in diagnostic["message"]
    )
    assert "artifacts[0].provenance.pipeline must be a string" in (
        diagnostic["message"]
    )
    assert "artifacts[0].provenance.intermediate must be a string or null" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_inconsistent_summary_counts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "invalid-summary-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                },
                "summary": {
                    "unitCount": 2,
                    "skippedCount": 1,
                    "targetCount": 2,
                    "artifactCount": 2,
                    "translatedCount": 0,
                    "failedCount": 1,
                    "diagnosticCounts": {"note": 0, "warning": 0, "error": 0},
                    "unitsBySourceBackend": {"metal": 1},
                    "artifactsByTarget": {
                        "metal": {
                            "artifactCount": 1,
                            "translatedCount": 1,
                            "failedCount": 0,
                        }
                    },
                    "sourceMapCount": 1,
                    "fineGrainedSourceMapCount": 1,
                },
                "units": [
                    {
                        "id": "simple.cgl",
                        "path": "simple.cgl",
                        "sourceBackend": "cgl",
                        "extension": ".cgl",
                    }
                ],
                "skipped": [],
                "artifacts": [
                    {
                        "source": "simple.cgl",
                        "target": "opengl",
                        "path": "out/opengl/simple.glsl",
                        "status": "translated",
                    }
                ],
                "diagnosticCounts": {"note": 1, "warning": 0, "error": 0},
                "diagnostics": [
                    {
                        "severity": "warning",
                        "code": "project.scan.missing-source-root",
                        "message": "Configured source root does not exist.",
                        "location": {
                            "file": "crosstl.toml",
                            "line": 1,
                            "column": 1,
                            "offset": 0,
                            "length": 0,
                            "endLine": 1,
                            "endColumn": 1,
                            "endOffset": 0,
                        },
                        "missingCapabilities": ["repo.scan"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "summary.unitCount must match units length" in diagnostic["message"]
    assert "summary.skippedCount must match skipped length" in diagnostic["message"]
    assert "summary.targetCount must match project.targets length" in (
        diagnostic["message"]
    )
    assert "summary.artifactCount must match artifacts length" in (
        diagnostic["message"]
    )
    assert "summary.translatedCount must match translated artifacts" in (
        diagnostic["message"]
    )
    assert "summary.diagnosticCounts must match diagnostics" in diagnostic["message"]
    assert "summary.unitsBySourceBackend must match units" in diagnostic["message"]
    assert "summary.artifactsByTarget must match artifacts" in diagnostic["message"]
    assert "summary.sourceMapCount must match artifact source maps" in (
        diagnostic["message"]
    )
    assert "summary.fineGrainedSourceMapCount must match artifact source maps" in (
        diagnostic["message"]
    )
    assert "diagnosticCounts must match diagnostics" in diagnostic["message"]


def test_validate_project_report_rejects_malformed_migration_actions(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "invalid-migration-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                },
                "artifacts": [],
                "migration": {
                    "scope": "runtime-porting",
                    "nonGoals": "runtime migration",
                    "actions": [
                        {
                            "kind": "",
                            "severity": "fatal",
                            "message": "",
                            "targets": "opengl",
                        },
                        {
                            "kind": "runtime-porting",
                            "severity": "note",
                            "message": "Review host runtime integration.",
                            "targets": ["opengl"],
                        },
                        "not an action",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "migration.scope must be shader-kernel-translation" in (
        diagnostic["message"]
    )
    assert "migration.nonGoals must be a list of strings" in diagnostic["message"]
    assert "migration.actions[0].kind must be a string" in diagnostic["message"]
    assert "migration.actions[0].message must be a string" in diagnostic["message"]
    assert "migration.actions[0].severity must be note, warning, or error" in (
        diagnostic["message"]
    )
    assert "migration.actions[0].targets must be a list of strings" in (
        diagnostic["message"]
    )
    assert "migration.actions[1].kind must be one of manual-runtime-integration" in (
        diagnostic["message"]
    )
    assert "migration.actions[2] must be an object" in diagnostic["message"]


def test_validate_project_report_rejects_artifacts_with_undeclared_targets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "undeclared-artifact-target-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                },
                "artifacts": [
                    {
                        "source": "simple.cgl",
                        "target": "metal",
                        "path": "out/metal/simple.metal",
                        "status": "translated",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].target must be listed in project.targets" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_malformed_diagnostics(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "invalid-diagnostics-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": [],
                    "outputDir": "out",
                },
                "artifacts": [],
                "diagnostics": [
                    {
                        "severity": "fatal",
                        "code": "",
                        "message": "bad diagnostic",
                        "location": {},
                        "target": "",
                        "missingCapabilities": "repo.scan",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "diagnostics[0].severity must be note, warning, or error" in (
        diagnostic["message"]
    )
    assert "diagnostics[0].code must be a string" in diagnostic["message"]
    assert "diagnostics[0].location.file must be a string" in diagnostic["message"]
    assert "diagnostics[0].target must be a string" in diagnostic["message"]
    assert "diagnostics[0].missingCapabilities must be a list of strings" in (
        diagnostic["message"]
    )


def test_validate_project_report_records_toolchain_failures(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    artifact = repo / "out" / "opengl" / "simple.glsl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("#version 450\nvoid main() {}\n", encoding="utf-8")
    report_path = repo / "portability-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                },
                "artifacts": [
                    {
                        "source": "simple.cgl",
                        "target": "opengl",
                        "path": "out/opengl/simple.glsl",
                        "status": "translated",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        project_pipeline.shutil,
        "which",
        lambda tool: (
            "/usr/bin/glslangValidator" if tool == "glslangValidator" else None
        ),
    )
    monkeypatch.setattr(
        project_pipeline.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="shader validation failed",
        ),
    )

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is False
    assert payload["diagnosticCounts"]["error"] == 1
    assert payload["diagnostics"][0]["code"] == "project.validate.toolchain-failed"
    assert payload["diagnostics"][0]["target"] == "opengl"
    assert payload["diagnostics"][0]["location"]["file"] == ("out/opengl/simple.glsl")
    assert payload["validation"]["toolchainRuns"][0]["status"] == "failed"
    assert payload["validation"]["toolchainRuns"][0]["returncode"] == 2
    assert payload["validation"]["toolchainRuns"][0]["stderr"] == (
        "shader validation failed"
    )


def test_validate_project_report_skips_toolchain_smoke_for_missing_artifacts(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "portability-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                },
                "artifacts": [
                    {
                        "source": "simple.cgl",
                        "target": "opengl",
                        "path": "out/opengl/simple.glsl",
                        "status": "translated",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        project_pipeline.shutil,
        "which",
        lambda tool: (
            "/usr/bin/glslangValidator" if tool == "glslangValidator" else None
        ),
    )
    monkeypatch.setattr(
        project_pipeline.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("toolchain should not run"),
    )

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is False
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 1}
    assert payload["validation"]["artifacts"] == [
        {
            "source": "simple.cgl",
            "target": "opengl",
            "path": "out/opengl/simple.glsl",
            "exists": False,
            "status": "failed",
        }
    ]
    assert payload["validation"]["toolchainRuns"] == []
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.missing-artifact"
    assert diagnostic["target"] == "opengl"


def test_validate_project_report_records_failed_artifacts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "portability-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["not-a-backend"],
                    "outputDir": "out",
                },
                "artifacts": [
                    {
                        "source": "simple.cgl",
                        "target": "not-a-backend",
                        "path": "out/not-a-backend/simple.out",
                        "status": "failed",
                        "error": "unsupported target backend",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["diagnosticCounts"]["error"] == 1
    assert payload["validation"]["artifacts"] == [
        {
            "source": "simple.cgl",
            "target": "not-a-backend",
            "path": "out/not-a-backend/simple.out",
            "exists": False,
            "status": "failed",
        }
    ]
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.failed-artifact"
    assert diagnostic["location"]["file"] == "simple.cgl"
    assert diagnostic["target"] == "not-a-backend"
    assert "unsupported target backend" in diagnostic["message"]


def test_validate_project_report_rejects_artifacts_outside_project(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (outside / "simple.glsl").write_text(
        "#version 450\nvoid main() {}\n", encoding="utf-8"
    )
    report_path = repo / "portability-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                },
                "artifacts": [
                    {
                        "source": "simple.cgl",
                        "target": "opengl",
                        "path": "../outside/simple.glsl",
                        "status": "translated",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        project_pipeline.shutil,
        "which",
        lambda tool: (
            "/usr/bin/glslangValidator" if tool == "glslangValidator" else None
        ),
    )
    monkeypatch.setattr(
        project_pipeline.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("toolchain should not run"),
    )

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is False
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 1}
    assert payload["validation"]["artifacts"] == [
        {
            "source": "simple.cgl",
            "target": "opengl",
            "path": "../outside/simple.glsl",
            "exists": False,
            "status": "failed",
        }
    ]
    assert payload["validation"]["toolchainRuns"] == []
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.artifact-outside-project"
    assert diagnostic["location"]["file"] == "simple.cgl"
    assert diagnostic["target"] == "opengl"
    assert diagnostic["missingCapabilities"] == ["artifact.manifest"]


def test_translate_project_records_structured_diagnostics_for_failures(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(
        repo,
        targets=["opengl", "not-a-backend"],
        output_dir="out",
    )
    payload = report.to_json()

    assert payload["summary"]["artifactCount"] == 2
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 1
    assert payload["summary"]["artifactsByTarget"] == {
        "not-a-backend": {
            "artifactCount": 1,
            "translatedCount": 0,
            "failedCount": 1,
        },
        "opengl": {
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
        },
    }
    assert payload["diagnosticCounts"]["error"] == 2
    target_diagnostic = next(
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["code"] == "project.config.unsupported-target"
    )
    assert target_diagnostic["severity"] == "error"
    assert target_diagnostic["target"] == "not-a-backend"
    assert target_diagnostic["missingCapabilities"] == ["target.backend"]
    translate_diagnostic = next(
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["code"] == "project.translate.failed"
    )
    assert translate_diagnostic["severity"] == "error"
    assert translate_diagnostic["target"] == "not-a-backend"
    assert translate_diagnostic["location"]["file"] == "simple.cgl"
    failed_artifact = next(
        artifact for artifact in payload["artifacts"] if artifact["status"] == "failed"
    )
    assert failed_artifact["target"] == "not-a-backend"
    assert failed_artifact["error"]


def test_project_cli_translate_project_writes_report(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "translate-project",
            str(repo),
            "--target",
            "opengl",
            "--output-dir",
            "out",
            "--report",
            str(report_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "Wrote" in result.stdout
    assert payload["summary"]["translatedCount"] == 1
    assert (repo / "out" / "opengl" / "simple.glsl").exists()


def test_project_cli_translate_project_fails_on_error_diagnostics(tmp_path):
    repo = tmp_path / "repo"
    outside_dir = tmp_path / "outside"
    repo.mkdir()
    outside_dir.mkdir()
    (outside_dir / "external.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["../outside"]
            include = ["**/*"]
            """).strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "translate-project",
            str(repo),
            "--target",
            "opengl",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["summary"]["failedCount"] == 0
    assert payload["summary"]["diagnosticCounts"]["error"] == 1
    assert payload["diagnostics"][0]["code"] == (
        "project.config.source-root-outside-project"
    )


def test_project_cli_scan_fails_on_error_diagnostics(tmp_path):
    repo = tmp_path / "repo"
    outside_dir = tmp_path / "outside"
    repo.mkdir()
    outside_dir.mkdir()
    (outside_dir / "external.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["../outside"]
            include = ["**/*"]
            """).strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "scan",
            str(repo),
            "--target",
            "opengl",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["summary"]["diagnosticCounts"]["error"] == 1
    assert payload["diagnostics"][0]["code"] == (
        "project.config.source-root-outside-project"
    )


def test_project_cli_scan_reports_unsupported_targets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "scan",
            str(repo),
            "--target",
            "not-a-backend",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["summary"]["diagnosticCounts"]["error"] == 1
    assert payload["diagnostics"][0]["code"] == "project.config.unsupported-target"
    assert payload["diagnostics"][0]["target"] == "not-a-backend"
    assert payload["diagnostics"][0]["missingCapabilities"] == ["target.backend"]


def test_project_cli_report_writes_output_and_fails_on_error_diagnostics(tmp_path):
    repo = tmp_path / "repo"
    outside_dir = tmp_path / "outside"
    output = tmp_path / "report.json"
    repo.mkdir()
    outside_dir.mkdir()
    (outside_dir / "external.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["../outside"]
            include = ["**/*"]
            """).strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "report",
            str(repo),
            "--target",
            "opengl",
            "--output",
            str(output),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert "Wrote" in result.stdout
    assert payload["summary"]["diagnosticCounts"]["error"] == 1
    assert payload["diagnostics"][0]["code"] == (
        "project.config.source-root-outside-project"
    )


def test_project_cli_translate_project_rejects_output_dir_outside_project(tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "translate-project",
            str(repo),
            "--target",
            "opengl",
            "--output-dir",
            "../outside",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert outside.exists() is False
    assert payload["summary"]["failedCount"] == 1
    assert payload["diagnostics"][0]["code"] == (
        "project.config.output-dir-outside-project"
    )


def test_project_cli_validate_project_reports_failed_artifacts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "portability-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["not-a-backend"],
                    "outputDir": "out",
                },
                "artifacts": [
                    {
                        "source": "simple.cgl",
                        "target": "not-a-backend",
                        "path": "out/not-a-backend/simple.out",
                        "status": "failed",
                        "error": "unsupported target backend",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "validate-project",
            str(report_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["success"] is False
    assert payload["diagnostics"][0]["code"] == "project.validate.failed-artifact"


def test_project_cli_validate_project_reports_invalid_report_shape(tmp_path):
    report_path = tmp_path / "invalid-report.json"
    report_path.write_text(json.dumps({"kind": "not-a-report"}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "validate-project",
            str(report_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["success"] is False
    assert payload["diagnostics"][0]["code"] == "project.validate.invalid-report"


def test_legacy_single_file_cli_still_works(tmp_path):
    shader = tmp_path / "simple.cgl"
    output = tmp_path / "simple.glsl"
    shader.write_text(SIMPLE_CROSSL, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            str(shader),
            "--backend",
            "opengl",
            "--output",
            str(output),
            "--no-format",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Successfully translated" in result.stdout
    assert output.exists()
