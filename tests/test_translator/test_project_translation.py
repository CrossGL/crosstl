import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import crosstl._crosstl as crosstl_cli
import crosstl.project.pipeline as project_pipeline
from crosstl.project import (
    inspect_project_report,
    load_project_config,
    scan_project,
    translate_project,
    validate_project_report,
)
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

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


def _source_hash_status_counts(**overrides):
    counts = {
        "missing": 0,
        "mismatch": 0,
        "not-recorded": 0,
        "ok": 0,
        "outside-project": 0,
    }
    counts.update(overrides)
    return counts


def _generated_hash_status_counts(**overrides):
    counts = {
        "missing": 0,
        "mismatch": 0,
        "not-applicable": 0,
        "not-recorded": 0,
        "ok": 0,
        "outside-project": 0,
    }
    counts.update(overrides)
    return counts


def _source_map_status_counts(**overrides):
    counts = {
        "mismatch": 0,
        "not-applicable": 0,
        "not-checked": 0,
        "not-recorded": 0,
        "ok": 0,
    }
    counts.update(overrides)
    return counts


def _source_remap_status_counts(**overrides):
    counts = {
        "hash-mismatch": 0,
        "invalid": 0,
        "mismatch": 0,
        "missing": 0,
        "not-applicable": 0,
        "not-recorded": 0,
        "ok": 0,
        "outside-project": 0,
    }
    counts.update(overrides)
    return counts


def _refresh_artifact_summary(payload):
    artifacts = payload["artifacts"]
    summary = payload["summary"]
    summary["artifactCount"] = len(artifacts)
    summary["translatedCount"] = sum(
        1 for artifact in artifacts if artifact.get("status") == "translated"
    )
    summary["failedCount"] = sum(
        1 for artifact in artifacts if artifact.get("status") == "failed"
    )
    summary["artifactsBySourceBackend"] = (
        project_pipeline._artifact_counts_by_source_backend(artifacts)
    )
    summary["artifactsByVariant"] = project_pipeline._artifact_counts_by_variant(
        artifacts
    )
    summary["artifactsByTarget"] = project_pipeline._artifact_counts_by_target(
        artifacts
    )
    summary.update(project_pipeline._source_map_rollups(artifacts))


def _diagnostic_location(file):
    return {
        "file": file,
        "line": 1,
        "column": 1,
        "offset": 0,
        "length": 0,
        "endLine": 1,
        "endColumn": 1,
        "endOffset": 0,
    }


def _write_count_balanced_artifact_gap_report(repo, *, omit_artifact_matrix=False):
    repo.mkdir()
    (repo / "first.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "second.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    payload = translate_project(repo, targets=["cgl"], output_dir="out").to_json()
    second_artifact = next(
        artifact
        for artifact in payload["artifacts"]
        if artifact["source"] == "second.cgl"
    )
    second_artifact["path"] = "out/cgl/wrong.cgl"
    if omit_artifact_matrix:
        payload.pop("artifactMatrix")
    report_path = repo / "out" / "count-balanced-artifact-gap-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return report_path


def _write_large_migration_report(repo, *, action_count=21):
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    payload = translate_project(repo, targets=["cgl"], output_dir="out").to_json()
    actions = [
        {
            "kind": "manual-runtime-integration",
            "severity": "note",
            "message": f"Review host integration task {index}.",
            "targets": ["cgl"],
        }
        for index in range(action_count)
    ]
    payload["migration"].update(project_pipeline._migration_action_rollups(actions))
    payload["migration"]["actions"] = actions
    report_path = repo / "out" / "large-migration-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return report_path


def _write_failed_artifact_report(repo):
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
    return report_path


def test_support_external_corpus_manifest_documents_pinned_reductions():
    manifest = json.loads(
        (ROOT / "support" / "external-corpus.json").read_text(encoding="utf-8")
    )
    backend_catalog = json.loads(
        (ROOT / "support" / "backends.json").read_text(encoding="utf-8")
    )

    assert manifest["schemaVersion"] == 1
    assert manifest["entries"]
    source_backends = {entry["sourceBackend"] for entry in manifest["entries"]}
    backend_ids = {backend["id"] for backend in backend_catalog["backends"]}
    assert backend_ids == source_backends
    for entry in manifest["entries"]:
        assert entry["id"]
        assert entry["path"]
        assert entry["repository"].startswith("https://github.com/")
        assert len(entry["commit"]) == 40
        assert entry["sourceUrl"].startswith(entry["repository"])
        assert entry["targets"] == ["cgl"]


def test_support_project_feature_evidence_references_existing_tests():
    catalog = json.loads(
        (ROOT / "support" / "features.json").read_text(encoding="utf-8")
    )
    test_file = ROOT / "tests" / "test_translator" / "test_project_translation.py"
    declared_tests = {
        line.strip().split("(", 1)[0][len("def ") :]
        for line in test_file.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("def test_")
    }
    evidence_prefix = "tests/test_translator/test_project_translation.py::def "

    missing_evidence = []
    missing_tests = []
    for feature in catalog["features"]:
        if feature.get("category") != "project":
            continue
        for backend, support in feature.get("support", {}).items():
            evidence = support.get("evidence")
            if not evidence:
                missing_evidence.append(f"{feature.get('id')}:{backend}")
                continue
            project_evidence = [
                item
                for item in evidence
                if isinstance(item, str) and item.startswith(evidence_prefix)
            ]
            if not project_evidence:
                missing_evidence.append(f"{feature.get('id')}:{backend}")
                continue
            for item in project_evidence:
                test_name = item[len(evidence_prefix) :]
                if test_name not in declared_tests:
                    missing_tests.append(item)

    assert missing_evidence == []
    assert missing_tests == []


def test_support_project_source_map_notes_document_granularity_contract():
    catalog = json.loads(
        (ROOT / "support" / "features.json").read_text(encoding="utf-8")
    )
    checked_features = {"project.validation_hooks", "project.source_provenance"}
    stale_phrase = "single file-level source-map mapping"
    required_phrases = (
        "file-level source-map mapping cardinality",
        "fine-grained positive-length source-map mappings",
    )

    stale_notes = []
    missing_notes = []
    for feature in catalog["features"]:
        if feature.get("id") not in checked_features:
            continue
        for backend, support in feature.get("support", {}).items():
            notes = support.get("notes", "")
            if stale_phrase in notes:
                stale_notes.append(f"{feature.get('id')}:{backend}")
            if not all(phrase in notes for phrase in required_phrases):
                missing_notes.append(f"{feature.get('id')}:{backend}")

    assert stale_notes == []
    assert missing_notes == []


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
    payload = scan.to_report(targets=["cgl"]).to_json()

    assert [unit.relative_path for unit in scan.units] == [
        "shaders/main.cgl",
        "shaders/post.frag",
    ]
    assert [unit.source_backend for unit in scan.units] == ["cgl", "opengl"]
    assert scan.skipped == []
    assert payload["units"][0]["sourceHash"] == project_pipeline._source_hash(
        shader_dir / "main.cgl"
    )
    assert payload["units"][1]["sourceHash"] == project_pipeline._source_hash(
        shader_dir / "post.frag"
    )


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
    payload = scan.to_report(targets=["cgl"]).to_json()

    assert scan.units == []
    assert scan.skipped == [{"path": "kernel.txt", "reason": "unsupported-extension"}]
    assert payload["summary"]["skippedByReason"] == {"unsupported-extension": 1}
    assert payload["summary"]["skippedByExtension"] == {".txt": 1}
    assert {diagnostic.code for diagnostic in scan.diagnostics} == {
        "project.scan.empty",
        "project.scan.unsupported-source",
    }


def test_scan_project_reports_extensionless_unsupported_sources(tmp_path):
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
    (repo / "kernel").write_text("not shader code", encoding="utf-8")

    scan = scan_project(load_project_config(repo))
    payload = scan.to_report(targets=["cgl"]).to_json()

    assert scan.units == []
    assert scan.skipped == [{"path": "kernel", "reason": "unsupported-extension"}]
    assert payload["summary"]["skippedByReason"] == {"unsupported-extension": 1}
    assert payload["summary"]["skippedByExtension"] == {"extensionless": 1}
    diagnostics_by_code = {
        diagnostic["code"]: diagnostic for diagnostic in payload["diagnostics"]
    }
    assert set(diagnostics_by_code) == {
        "project.scan.empty",
        "project.scan.unsupported-source",
    }
    assert diagnostics_by_code["project.scan.unsupported-source"][
        "missingCapabilities"
    ] == ["source.discovery"]
    assert (
        "No registered source backend for kernel"
        in diagnostics_by_code["project.scan.unsupported-source"]["message"]
    )


def test_scan_project_skips_known_unsupported_source_artifacts(tmp_path):
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
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "compiled.spv").write_bytes(b"\x03\x02#shader-binary")
    (repo / "future.wgsl").write_text("@compute fn main() {}\n", encoding="utf-8")

    config = load_project_config(repo)
    scan = scan_project(config)
    payload = scan.to_report(targets=["cgl"]).to_json()

    assert [unit.relative_path for unit in scan.units] == ["simple.cgl"]
    assert scan.skipped == [
        {"path": "compiled.spv", "reason": "unsupported-extension"},
        {"path": "future.wgsl", "reason": "unsupported-extension"},
    ]
    assert payload["summary"]["skippedByReason"] == {"unsupported-extension": 2}
    assert payload["summary"]["skippedByExtension"] == {".spv": 1, ".wgsl": 1}
    diagnostics_by_file = {
        diagnostic.location.file: diagnostic for diagnostic in scan.diagnostics
    }
    assert diagnostics_by_file["compiled.spv"].code == "project.scan.unsupported-source"
    assert "Binary SPIR-V input files" in diagnostics_by_file["compiled.spv"].message
    assert diagnostics_by_file["future.wgsl"].code == "project.scan.unsupported-source"
    assert "WGSL/WebGPU source files" in diagnostics_by_file["future.wgsl"].message


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
    assert payload["summary"]["diagnosticsByCode"] == {
        "project.config.source-root-outside-project": 1,
        "project.scan.missing-source-root": 1,
    }
    assert payload["summary"]["missingCapabilityCounts"] == {"repo.scan": 2}
    assert payload["project"]["sourceRootStatusCounts"] == {
        "active": 1,
        "missing": 1,
        "outside-project": 1,
    }
    assert [record["status"] for record in payload["project"]["sourceRootStatus"]] == [
        "active",
        "missing",
        "outside-project",
    ]
    assert [
        record["scanVisible"] for record in payload["project"]["sourceRootStatus"]
    ] == [True, False, False]
    assert diagnostics["project.scan.missing-source-root"]["location"]["file"] == (
        "crosstl.toml"
    )
    assert diagnostics["project.config.source-root-outside-project"][
        "missingCapabilities"
    ] == ["repo.scan"]


def test_scan_project_reports_source_roots_that_are_not_directories(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["simple.cgl"]
            include = ["**/*"]
            """).strip(),
        encoding="utf-8",
    )

    scan = scan_project(load_project_config(repo))
    payload = scan.to_report().to_json()

    assert scan.units == []
    assert payload["project"]["sourceRootStatusCounts"] == {"not-directory": 1}
    assert payload["project"]["sourceRootStatus"] == [
        {
            "path": "simple.cgl",
            "resolvedPath": str((repo / "simple.cgl").resolve()),
            "status": "not-directory",
            "scanVisible": False,
        }
    ]
    diagnostics = {diagnostic.code for diagnostic in scan.diagnostics}
    assert diagnostics == {
        "project.config.source-root-not-directory",
        "project.scan.empty",
    }


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


def test_scan_project_reports_drive_relative_include_patterns(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            include = ["C:tmp/*.cgl"]
            exclude = []
            """).strip(),
        encoding="utf-8",
    )

    payload = scan_project(load_project_config(repo)).to_report().to_json()

    assert payload["summary"]["diagnosticCounts"] == {
        "note": 0,
        "warning": 1,
        "error": 1,
    }
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.config.include-pattern-outside-project"
    assert "C:tmp/*.cgl" in diagnostic["message"]
    assert diagnostic["missingCapabilities"] == ["repo.scan"]


def test_scan_project_reports_exclude_patterns_outside_project(tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (outside / "external.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    absolute_pattern = (outside / "*.cgl").as_posix()
    (repo / "crosstl.toml").write_text(
        textwrap.dedent(f"""
            [project]
            exclude = ["{absolute_pattern}", "../outside/*.cgl"]
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
    assert any("../outside/*.cgl" in message for message in diagnostics)
    assert [diagnostic["code"] for diagnostic in payload["diagnostics"]] == [
        "project.config.exclude-pattern-outside-project",
        "project.config.exclude-pattern-outside-project",
    ]
    assert all(
        diagnostic["missingCapabilities"] == ["repo.scan"]
        for diagnostic in payload["diagnostics"]
    )


def test_scan_project_reports_drive_relative_exclude_patterns(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            exclude = ["C:tmp/*.cgl"]
            """).strip(),
        encoding="utf-8",
    )

    payload = scan_project(load_project_config(repo)).to_report().to_json()

    assert [unit["path"] for unit in payload["units"]] == ["simple.cgl"]
    assert payload["summary"]["diagnosticCounts"] == {
        "note": 0,
        "warning": 0,
        "error": 1,
    }
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.config.exclude-pattern-outside-project"
    assert "C:tmp/*.cgl" in diagnostic["message"]
    assert diagnostic["missingCapabilities"] == ["repo.scan"]


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
    assert payload["migration"]["actionCount"] == 1
    assert payload["migration"]["actionsByKind"] == {"manual-runtime-integration": 1}
    assert payload["migration"]["actionsBySeverity"] == {"note": 1}
    assert payload["migration"]["actionsByTarget"] == {"opengl": 1}
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


def test_validate_project_report_rejects_missing_migration_rollups(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    payload = scan_project(repo).to_report(targets=["opengl"]).to_json()
    payload["migration"].pop("actionCount")
    payload["migration"].pop("actionsByKind")
    payload["migration"].pop("actionsBySeverity")
    payload["migration"].pop("actionsByTarget")
    report_path = repo / "missing-migration-rollups-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "migration.actionCount must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "migration.actionsByKind must be an object" in diagnostic["message"]
    assert "migration.actionsBySeverity must be an object" in diagnostic["message"]
    assert "migration.actionsByTarget must be an object" in diagnostic["message"]


def test_scan_report_records_unsupported_targets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = scan_project(repo).to_report(targets=["not-a-backend"])
    payload = report.to_json()
    report_path = repo / "unsupported-target-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)

    assert payload["project"]["targets"] == ["not-a-backend"]
    assert payload["summary"]["diagnosticCounts"]["error"] == 1
    assert payload["summary"]["diagnosticsByCode"] == {
        "project.config.unsupported-target": 1
    }
    assert payload["summary"]["missingCapabilityCounts"] == {"target.backend": 1}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.config.unsupported-target"
    assert diagnostic["location"]["file"] == "."
    assert diagnostic["target"] == "not-a-backend"
    assert diagnostic["missingCapabilities"] == ["target.backend"]
    assert "Supported targets:" in diagnostic["message"]
    assert payload["migration"]["actions"] == []
    assert "project.validate.invalid-report" not in {
        diagnostic["code"] for diagnostic in validation["diagnostics"]
    }


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
    assert payload["project"]["sourceOverrides"] == {"gpu/*.shader": "cgl"}
    assert payload["project"]["sourceOverrideCount"] == 1
    assert payload["project"]["defines"] == {"USE_FAST_PATH": "1"}
    assert payload["project"]["defineCount"] == 1
    assert payload["project"]["variants"] == {"debug": {"USE_FAST_PATH": "0"}}
    assert payload["project"]["variantCount"] == 1
    assert payload["project"]["variantDefineCounts"] == {"debug": 1}


def test_project_config_resolves_relative_config_path_from_root(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "custom.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["opengl"]
            output_dir = "generated"
            """).strip(),
        encoding="utf-8",
    )

    config = load_project_config(repo, "custom.toml")

    assert config.config_path == repo.resolve() / "custom.toml"
    assert config.targets == ["opengl"]
    assert config.output_dir == "generated"


def test_project_config_rejects_missing_explicit_config_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(ValueError) as excinfo:
        load_project_config(repo, "missing.toml")

    assert str(repo.resolve() / "missing.toml") in str(excinfo.value)


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


def test_scan_project_reports_include_dir_files_without_hiding_units(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (repo / "include-file").write_text("// not a directory\n", encoding="utf-8")
    (shader_dir / "main.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]
            include_dirs = ["include-file"]
            """).strip(),
        encoding="utf-8",
    )

    scan = scan_project(load_project_config(repo))
    payload = scan.to_report().to_json()

    assert [unit.relative_path for unit in scan.units] == ["shaders/main.cgl"]
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 1, "error": 0}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.config.include-dir-not-directory"
    assert diagnostic["location"]["file"] == "crosstl.toml"
    assert diagnostic["missingCapabilities"] == ["include.resolution"]
    assert "include-file" in diagnostic["message"]
    assert payload["project"]["includeDirStatus"] == [
        {
            "path": "include-file",
            "resolvedPath": str((repo / "include-file").resolve()),
            "status": "not-directory",
            "frontendVisible": False,
        }
    ]
    assert payload["project"]["includeDirStatusCounts"] == {"not-directory": 1}


def test_scan_project_records_include_dependency_resolution(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    include_dir = repo / "includes"
    shader_dir.mkdir(parents=True)
    include_dir.mkdir()
    (shader_dir / "local.inc").write_text("vec4 local_color();\n", encoding="utf-8")
    (include_dir / "shared.inc").write_text("vec4 shared_color();\n", encoding="utf-8")
    (tmp_path / "outside.inc").write_text("vec4 outside_color();\n", encoding="utf-8")
    (shader_dir / "main.frag").write_text(
        textwrap.dedent("""
            #version 450
            #include "local.inc"
            #include <shared.inc>
            #include <cuda_runtime.h>
            #include "missing.inc"
            #include PROJECT_HEADER
            #include UNKNOWN_HEADER
            #include "../../outside.inc"
            void main() {}
            """).strip(),
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]
            include_dirs = ["includes"]

            [project.defines]
            PROJECT_HEADER = "<shared.inc>"
            """).strip(),
        encoding="utf-8",
    )

    payload = (
        scan_project(load_project_config(repo)).to_report(targets=["cgl"]).to_json()
    )

    assert [unit["path"] for unit in payload["units"]] == ["shaders/main.frag"]
    dependencies = payload["units"][0]["includeDependencies"]
    assert dependencies == [
        {
            "include": "local.inc",
            "kind": "local",
            "status": "resolved",
            "line": 2,
            "column": 1,
            "resolvedPath": "shaders/local.inc",
            "resolvedHash": project_pipeline._source_hash(shader_dir / "local.inc"),
            "resolvedFrom": "source",
        },
        {
            "include": "shared.inc",
            "kind": "system",
            "status": "resolved",
            "line": 3,
            "column": 1,
            "resolvedPath": "includes/shared.inc",
            "resolvedHash": project_pipeline._source_hash(include_dir / "shared.inc"),
            "resolvedFrom": "include-dir",
        },
        {
            "include": "cuda_runtime.h",
            "kind": "system",
            "status": "system",
            "line": 4,
            "column": 1,
        },
        {
            "include": "missing.inc",
            "kind": "local",
            "status": "missing",
            "line": 5,
            "column": 1,
        },
        {
            "include": "shared.inc",
            "kind": "system",
            "status": "resolved",
            "line": 6,
            "column": 1,
            "resolvedFromDefine": "PROJECT_HEADER",
            "resolvedPath": "includes/shared.inc",
            "resolvedHash": project_pipeline._source_hash(include_dir / "shared.inc"),
            "resolvedFrom": "include-dir",
        },
        {
            "include": "UNKNOWN_HEADER",
            "kind": "dynamic",
            "status": "dynamic",
            "line": 7,
            "column": 1,
        },
        {
            "include": "../../outside.inc",
            "kind": "local",
            "status": "outside-project",
            "line": 8,
            "column": 1,
        },
    ]
    assert payload["summary"]["includeDependencyCount"] == 7
    assert payload["summary"]["includeDependenciesByKind"] == {
        "dynamic": 1,
        "local": 3,
        "system": 3,
    }
    assert payload["summary"]["includeDependenciesByStatus"] == {
        "dynamic": 1,
        "missing": 1,
        "outside-project": 1,
        "resolved": 3,
        "system": 1,
    }
    assert payload["summary"]["includeDependenciesByResolvedFrom"] == {
        "include-dir": 2,
        "source": 1,
    }
    assert payload["summary"]["includeDependenciesBySourceBackend"] == {"opengl": 7}
    assert payload["summary"]["includeDependenciesBySourceBackendStatus"] == {
        "opengl": {
            "dynamic": 1,
            "missing": 1,
            "outside-project": 1,
            "resolved": 3,
            "system": 1,
        }
    }
    assert payload["summary"]["diagnosticsByCode"] == {
        "project.scan.dynamic-include": 1,
        "project.scan.include-outside-project": 1,
        "project.scan.missing-include": 1,
    }
    assert payload["summary"]["missingCapabilityCounts"] == {"include.resolution": 3}


def test_scan_project_skips_inactive_ifdef_include_dependencies(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (shader_dir / "fast.inc").write_text("vec4 fast_color();\n", encoding="utf-8")
    (shader_dir / "main.frag").write_text(
        textwrap.dedent("""
            #version 450
            #ifdef USE_FAST_PATH
            #include "fast.inc"
            #else
            #include "missing.inc"
            #endif
            void main() {}
            """).strip(),
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]

            [project.defines]
            USE_FAST_PATH = "1"
            """).strip(),
        encoding="utf-8",
    )

    payload = (
        scan_project(load_project_config(repo)).to_report(targets=["cgl"]).to_json()
    )

    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 0}
    assert payload["units"][0]["includeDependencies"] == [
        {
            "include": "fast.inc",
            "kind": "local",
            "status": "resolved",
            "line": 3,
            "column": 1,
            "resolvedPath": "shaders/fast.inc",
            "resolvedHash": project_pipeline._source_hash(shader_dir / "fast.inc"),
            "resolvedFrom": "source",
        }
    ]


def test_scan_project_records_variant_conditional_include_dependencies(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    include_dir = repo / "includes"
    shader_dir.mkdir(parents=True)
    include_dir.mkdir()
    (shader_dir / "debug.inc").write_text("vec4 debug_color();\n", encoding="utf-8")
    (include_dir / "release.inc").write_text(
        "vec4 release_color();\n",
        encoding="utf-8",
    )
    (shader_dir / "main.frag").write_text(
        textwrap.dedent("""
            #version 450
            #if defined(DEBUG_HEADER)
            #include DEBUG_HEADER
            #elif defined(RELEASE_HEADER)
            #include RELEASE_HEADER
            #else
            #include "missing.inc"
            #endif
            void main() {}
            """).strip(),
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]
            include_dirs = ["includes"]

            [project.variants.debug]
            DEBUG_HEADER = '"debug.inc"'

            [project.variants.release]
            RELEASE_HEADER = "<release.inc>"
            """).strip(),
        encoding="utf-8",
    )

    payload = (
        scan_project(load_project_config(repo)).to_report(targets=["cgl"]).to_json()
    )

    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 0}
    assert payload["units"][0]["includeDependencies"] == [
        {
            "include": "debug.inc",
            "kind": "local",
            "status": "resolved",
            "line": 3,
            "column": 1,
            "resolvedFromDefine": "DEBUG_HEADER",
            "variant": "debug",
            "resolvedPath": "shaders/debug.inc",
            "resolvedHash": project_pipeline._source_hash(shader_dir / "debug.inc"),
            "resolvedFrom": "source",
        },
        {
            "include": "release.inc",
            "kind": "system",
            "status": "resolved",
            "line": 5,
            "column": 1,
            "resolvedFromDefine": "RELEASE_HEADER",
            "variant": "release",
            "resolvedPath": "includes/release.inc",
            "resolvedHash": project_pipeline._source_hash(include_dir / "release.inc"),
            "resolvedFrom": "include-dir",
        },
    ]
    assert payload["summary"]["includeDependenciesByVariant"] == {
        "debug": 1,
        "release": 1,
    }


def test_scan_project_honors_ifndef_else_in_nested_include_dependencies(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    include_dir = repo / "includes"
    shader_dir.mkdir(parents=True)
    include_dir.mkdir()
    (include_dir / "shared.inc").write_text("vec4 shared_color();\n", encoding="utf-8")
    (shader_dir / "material.inc").write_text(
        textwrap.dedent("""
            #ifndef USE_SHARED
            #include "missing-local.inc"
            #else
            #include <shared.inc>
            #endif
            vec4 material_color();
            """).strip(),
        encoding="utf-8",
    )
    (shader_dir / "main.frag").write_text(
        '#version 450\n#include "material.inc"\nvoid main() {}\n',
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]
            include_dirs = ["includes"]

            [project.defines]
            USE_SHARED = "1"
            """).strip(),
        encoding="utf-8",
    )

    payload = (
        scan_project(load_project_config(repo)).to_report(targets=["cgl"]).to_json()
    )

    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 0}
    assert payload["units"][0]["includeDependencies"] == [
        {
            "include": "material.inc",
            "kind": "local",
            "status": "resolved",
            "line": 2,
            "column": 1,
            "resolvedPath": "shaders/material.inc",
            "resolvedHash": project_pipeline._source_hash(shader_dir / "material.inc"),
            "resolvedFrom": "source",
        },
        {
            "source": "shaders/material.inc",
            "include": "shared.inc",
            "kind": "system",
            "status": "resolved",
            "line": 4,
            "column": 1,
            "resolvedPath": "includes/shared.inc",
            "resolvedHash": project_pipeline._source_hash(include_dir / "shared.inc"),
            "resolvedFrom": "include-dir",
        },
    ]


def test_scan_project_keeps_includes_for_unsupported_conditional_expressions(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (shader_dir / "fast.inc").write_text("vec4 fast_color();\n", encoding="utf-8")
    (shader_dir / "fallback.inc").write_text(
        "vec4 fallback_color();\n",
        encoding="utf-8",
    )
    (shader_dir / "main.frag").write_text(
        textwrap.dedent("""
            #version 450
            #if USE_FAST_PATH == 1
            #include "fast.inc"
            #else
            #include "fallback.inc"
            #endif
            void main() {}
            """).strip(),
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]

            [project.defines]
            USE_FAST_PATH = "1"
            """).strip(),
        encoding="utf-8",
    )

    payload = (
        scan_project(load_project_config(repo)).to_report(targets=["cgl"]).to_json()
    )

    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 0}
    assert [
        (dependency["include"], dependency["status"])
        for dependency in payload["units"][0]["includeDependencies"]
    ] == [("fast.inc", "resolved"), ("fallback.inc", "resolved")]
    assert payload["summary"]["includeDependencyCount"] == 2


def test_scan_project_records_nested_include_dependencies(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    nested_dir = shader_dir / "include"
    include_dir = repo / "includes"
    nested_dir.mkdir(parents=True)
    include_dir.mkdir()
    (nested_dir / "constants.inc").write_text(
        "vec4 nested_color();\n",
        encoding="utf-8",
    )
    (include_dir / "shared.inc").write_text(
        "vec4 shared_color();\n",
        encoding="utf-8",
    )
    (nested_dir / "material.inc").write_text(
        textwrap.dedent("""
            #include "constants.inc"
            #include <shared.inc>
            vec4 material_color();
            """).strip(),
        encoding="utf-8",
    )
    (shader_dir / "main.frag").write_text(
        '#version 450\n#include "include/material.inc"\nvoid main() {}\n',
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]
            include_dirs = ["includes"]
            """).strip(),
        encoding="utf-8",
    )

    report = scan_project(load_project_config(repo)).to_report(targets=["cgl"])
    payload = report.to_json()
    report_path = repo / "scan-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    inspection = inspect_project_report(report_path)
    unit_hash = project_pipeline._source_hash(shader_dir / "main.frag")
    material_hash = project_pipeline._source_hash(nested_dir / "material.inc")
    constants_hash = project_pipeline._source_hash(nested_dir / "constants.inc")
    shared_hash = project_pipeline._source_hash(include_dir / "shared.inc")

    assert validation["success"] is True
    assert payload["units"][0]["includeDependencies"] == [
        {
            "include": "include/material.inc",
            "kind": "local",
            "status": "resolved",
            "line": 2,
            "column": 1,
            "resolvedPath": "shaders/include/material.inc",
            "resolvedHash": material_hash,
            "resolvedFrom": "source",
        },
        {
            "source": "shaders/include/material.inc",
            "include": "constants.inc",
            "kind": "local",
            "status": "resolved",
            "line": 1,
            "column": 1,
            "resolvedPath": "shaders/include/constants.inc",
            "resolvedHash": constants_hash,
            "resolvedFrom": "source",
        },
        {
            "source": "shaders/include/material.inc",
            "include": "shared.inc",
            "kind": "system",
            "status": "resolved",
            "line": 2,
            "column": 1,
            "resolvedPath": "includes/shared.inc",
            "resolvedHash": shared_hash,
            "resolvedFrom": "include-dir",
        },
    ]
    assert payload["summary"]["includeDependencyCount"] == 3
    assert payload["summary"]["includeDependenciesByKind"] == {
        "local": 2,
        "system": 1,
    }
    assert payload["summary"]["includeDependenciesByStatus"] == {"resolved": 3}
    assert payload["summary"]["includeDependenciesByResolvedFrom"] == {
        "include-dir": 1,
        "source": 2,
    }
    assert inspection["includeDependencies"]["resolvedDependencies"] == [
        {
            "source": "shaders/main.frag",
            "sourceBackend": "opengl",
            "include": "include/material.inc",
            "status": "resolved",
            "kind": "local",
            "line": 2,
            "column": 1,
            "resolvedPath": "shaders/include/material.inc",
            "resolvedFrom": "source",
            "unitSourceHashAlgorithm": unit_hash["algorithm"],
            "unitSourceHash": unit_hash["value"],
            "resolvedHashAlgorithm": material_hash["algorithm"],
            "resolvedHash": material_hash["value"],
        },
        {
            "source": "shaders/include/material.inc",
            "sourceBackend": "opengl",
            "include": "constants.inc",
            "status": "resolved",
            "kind": "local",
            "line": 1,
            "column": 1,
            "resolvedPath": "shaders/include/constants.inc",
            "resolvedFrom": "source",
            "unitSourceHashAlgorithm": unit_hash["algorithm"],
            "unitSourceHash": unit_hash["value"],
            "resolvedHashAlgorithm": constants_hash["algorithm"],
            "resolvedHash": constants_hash["value"],
        },
        {
            "source": "shaders/include/material.inc",
            "sourceBackend": "opengl",
            "include": "shared.inc",
            "status": "resolved",
            "kind": "system",
            "line": 2,
            "column": 1,
            "resolvedPath": "includes/shared.inc",
            "resolvedFrom": "include-dir",
            "unitSourceHashAlgorithm": unit_hash["algorithm"],
            "unitSourceHash": unit_hash["value"],
            "resolvedHashAlgorithm": shared_hash["algorithm"],
            "resolvedHash": shared_hash["value"],
        },
    ]


def test_scan_project_reports_nested_include_read_failures(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    nested_dir = shader_dir / "include"
    nested_dir.mkdir(parents=True)
    material_path = nested_dir / "material.inc"
    material_path.write_text(
        '#include "constants.inc"\nvec4 material_color();\n',
        encoding="utf-8",
    )
    (shader_dir / "main.frag").write_text(
        '#version 450\n#include "include/material.inc"\nvoid main() {}\n',
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]
            """).strip(),
        encoding="utf-8",
    )
    config = load_project_config(repo)
    original_read_text = Path.read_text
    unreadable_path = material_path.resolve()

    def read_text_or_fail(path, *args, **kwargs):
        if path.resolve() == unreadable_path:
            raise OSError("permission denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text_or_fail)

    report = scan_project(config).to_report(targets=["cgl"])
    payload = report.to_json()
    report_path = repo / "scan-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)

    assert validation["success"] is True
    assert payload["units"][0]["includeDependencies"] == [
        {
            "include": "include/material.inc",
            "kind": "local",
            "status": "resolved",
            "line": 2,
            "column": 1,
            "resolvedPath": "shaders/include/material.inc",
            "resolvedHash": project_pipeline._source_hash(material_path),
            "resolvedFrom": "source",
        }
    ]
    assert payload["summary"]["includeDependencyCount"] == 1
    assert payload["summary"]["includeDependenciesByKind"] == {"local": 1}
    assert payload["summary"]["includeDependenciesByStatus"] == {"resolved": 1}
    assert payload["summary"]["includeDependenciesByResolvedFrom"] == {"source": 1}
    assert payload["summary"]["diagnosticsByCode"] == {
        "project.scan.include-read-failed": 1
    }
    assert payload["summary"]["missingCapabilityCounts"] == {"include.resolution": 1}
    assert validation["diagnosticsByCode"] == {"project.scan.include-read-failed": 1}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic == {
        "severity": "warning",
        "code": "project.scan.include-read-failed",
        "message": (
            "Could not scan include directives in shaders/include/material.inc: "
            "permission denied"
        ),
        "location": {
            "file": "shaders/include/material.inc",
            "line": 1,
            "column": 1,
            "offset": 0,
            "length": 0,
            "endLine": 1,
            "endColumn": 1,
            "endOffset": 0,
        },
        "missingCapabilities": ["include.resolution"],
    }


def test_scan_project_reports_nested_include_cycles(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    nested_dir = shader_dir / "include"
    nested_dir.mkdir(parents=True)
    (nested_dir / "a.inc").write_text('#include "b.inc"\n', encoding="utf-8")
    (nested_dir / "b.inc").write_text('#include "a.inc"\n', encoding="utf-8")
    (shader_dir / "main.frag").write_text(
        '#version 450\n#include "include/a.inc"\nvoid main() {}\n',
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]
            """).strip(),
        encoding="utf-8",
    )

    report = scan_project(load_project_config(repo)).to_report(targets=["cgl"])
    payload = report.to_json()
    report_path = repo / "scan-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)

    assert validation["success"] is True
    assert payload["units"][0]["includeDependencies"] == [
        {
            "include": "include/a.inc",
            "kind": "local",
            "status": "resolved",
            "line": 2,
            "column": 1,
            "resolvedPath": "shaders/include/a.inc",
            "resolvedHash": project_pipeline._source_hash(nested_dir / "a.inc"),
            "resolvedFrom": "source",
        },
        {
            "source": "shaders/include/a.inc",
            "include": "b.inc",
            "kind": "local",
            "status": "resolved",
            "line": 1,
            "column": 1,
            "resolvedPath": "shaders/include/b.inc",
            "resolvedHash": project_pipeline._source_hash(nested_dir / "b.inc"),
            "resolvedFrom": "source",
        },
        {
            "source": "shaders/include/b.inc",
            "include": "a.inc",
            "kind": "local",
            "status": "resolved",
            "line": 1,
            "column": 1,
            "resolvedPath": "shaders/include/a.inc",
            "resolvedHash": project_pipeline._source_hash(nested_dir / "a.inc"),
            "resolvedFrom": "source",
        },
    ]
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 1, "error": 0}
    assert payload["diagnostics"] == [
        {
            "severity": "warning",
            "code": "project.scan.include-cycle",
            "message": (
                "Include directive in shaders/include/b.inc:1 creates a cycle "
                "and was not scanned recursively: a.inc"
            ),
            "location": {
                "file": "shaders/include/b.inc",
                "line": 1,
                "column": 1,
                "offset": 0,
                "length": 0,
                "endLine": 1,
                "endColumn": 1,
                "endOffset": 0,
            },
            "missingCapabilities": ["include.resolution"],
        }
    ]
    assert payload["summary"]["includeDependencyCount"] == 3
    assert payload["summary"]["includeDependenciesByKind"] == {"local": 3}
    assert payload["summary"]["includeDependenciesByStatus"] == {"resolved": 3}
    assert payload["summary"]["includeDependenciesByResolvedFrom"] == {"source": 3}
    assert payload["summary"]["diagnosticsByCode"] == {"project.scan.include-cycle": 1}
    assert payload["summary"]["missingCapabilityCounts"] == {"include.resolution": 1}
    assert validation["diagnosticsByCode"] == {"project.scan.include-cycle": 1}
    assert validation["missingCapabilityCounts"] == {"include.resolution": 1}


def test_scan_project_reports_define_backed_include_resolution_diagnostics(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.frag").write_text("#include PROJECT_HEADER\n", encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]

            [project.defines]
            PROJECT_HEADER = "\\"missing.inc\\""
            """).strip(),
        encoding="utf-8",
    )

    report = scan_project(load_project_config(repo)).to_report(targets=["cgl"])
    payload = report.to_json()

    assert payload["units"][0]["includeDependencies"] == [
        {
            "include": "missing.inc",
            "kind": "local",
            "status": "missing",
            "line": 1,
            "column": 1,
            "resolvedFromDefine": "PROJECT_HEADER",
        }
    ]
    assert payload["diagnostics"][0]["code"] == "project.scan.missing-include"
    assert payload["diagnostics"][0]["message"] == (
        "Include directive in main.frag:1 could not be resolved "
        "(from project define PROJECT_HEADER): missing.inc"
    )

    report_path = repo / "scan-report.json"
    report.write_json(report_path)
    inspection = inspect_project_report(report_path)
    unit_hash = project_pipeline._source_hash(repo / "main.frag")
    assert inspection["includeDependencies"]["unresolvedDependencies"] == [
        {
            "source": "main.frag",
            "sourceBackend": "opengl",
            "include": "missing.inc",
            "status": "missing",
            "kind": "local",
            "line": 1,
            "column": 1,
            "resolvedFromDefine": "PROJECT_HEADER",
            "unitSourceHashAlgorithm": unit_hash["algorithm"],
            "unitSourceHash": unit_hash["value"],
        }
    ]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    unit_hash_preview = f"{unit_hash['algorithm']}:{unit_hash['value'][:12]}..."
    assert (
        "- main.frag:1:1 [opengl]: missing local include missing.inc "
        f"(define PROJECT_HEADER, unitHash={unit_hash_preview})"
    ) in result.stdout


def test_scan_project_records_variant_define_backed_include_resolution(tmp_path):
    repo = tmp_path / "repo"
    include_dir = repo / "includes"
    repo.mkdir()
    include_dir.mkdir()
    (repo / "debug.inc").write_text("vec4 debug_color();\n", encoding="utf-8")
    (include_dir / "release.inc").write_text(
        "vec4 release_color();\n",
        encoding="utf-8",
    )
    (repo / "main.frag").write_text("#include PROJECT_HEADER\n", encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            include_dirs = ["includes"]

            [project.defines]
            PROJECT_HEADER = "\\"base.inc\\""

            [project.variants.debug]
            PROJECT_HEADER = "\\"debug.inc\\""

            [project.variants.release]
            PROJECT_HEADER = "<release.inc>"
            """).strip(),
        encoding="utf-8",
    )

    report = scan_project(load_project_config(repo)).to_report(targets=["cgl"])
    payload = report.to_json()
    report_path = repo / "scan-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    inspection = inspect_project_report(report_path)
    unit_hash = project_pipeline._source_hash(repo / "main.frag")
    debug_hash = project_pipeline._source_hash(repo / "debug.inc")
    release_hash = project_pipeline._source_hash(include_dir / "release.inc")
    debug_hash_preview = f"{debug_hash['algorithm']}:{debug_hash['value'][:12]}..."
    release_hash_preview = (
        f"{release_hash['algorithm']}:{release_hash['value'][:12]}..."
    )

    assert validation["success"] is True
    assert payload["units"][0]["includeDependencies"] == [
        {
            "include": "debug.inc",
            "kind": "local",
            "status": "resolved",
            "line": 1,
            "column": 1,
            "resolvedFromDefine": "PROJECT_HEADER",
            "variant": "debug",
            "resolvedPath": "debug.inc",
            "resolvedHash": debug_hash,
            "resolvedFrom": "source",
        },
        {
            "include": "release.inc",
            "kind": "system",
            "status": "resolved",
            "line": 1,
            "column": 1,
            "resolvedFromDefine": "PROJECT_HEADER",
            "variant": "release",
            "resolvedPath": "includes/release.inc",
            "resolvedHash": release_hash,
            "resolvedFrom": "include-dir",
        },
    ]
    assert payload["summary"]["includeDependencyCount"] == 2
    assert payload["summary"]["includeDependenciesByVariant"] == {
        "debug": 1,
        "release": 1,
    }
    assert payload["summary"]["includeDependenciesByKind"] == {
        "local": 1,
        "system": 1,
    }
    assert payload["summary"]["includeDependenciesByResolvedFrom"] == {
        "include-dir": 1,
        "source": 1,
    }
    assert inspection["includeDependencies"]["byVariant"] == {
        "debug": 1,
        "release": 1,
    }
    assert inspection["includeDependencies"]["resolvedDependencies"] == [
        {
            "source": "main.frag",
            "sourceBackend": "opengl",
            "include": "debug.inc",
            "status": "resolved",
            "kind": "local",
            "line": 1,
            "column": 1,
            "resolvedPath": "debug.inc",
            "resolvedFrom": "source",
            "resolvedFromDefine": "PROJECT_HEADER",
            "variant": "debug",
            "unitSourceHashAlgorithm": unit_hash["algorithm"],
            "unitSourceHash": unit_hash["value"],
            "resolvedHashAlgorithm": debug_hash["algorithm"],
            "resolvedHash": debug_hash["value"],
        },
        {
            "source": "main.frag",
            "sourceBackend": "opengl",
            "include": "release.inc",
            "status": "resolved",
            "kind": "system",
            "line": 1,
            "column": 1,
            "resolvedPath": "includes/release.inc",
            "resolvedFrom": "include-dir",
            "resolvedFromDefine": "PROJECT_HEADER",
            "variant": "release",
            "unitSourceHashAlgorithm": unit_hash["algorithm"],
            "unitSourceHash": unit_hash["value"],
            "resolvedHashAlgorithm": release_hash["algorithm"],
            "resolvedHash": release_hash["value"],
        },
    ]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Include dependencies by variant: debug=1, release=1" in result.stdout
    unit_hash_preview = f"{unit_hash['algorithm']}:{unit_hash['value'][:12]}..."
    assert (
        "- main.frag:1:1 [opengl]: resolved local include debug.inc -> debug.inc "
        f"(variant debug, source, define PROJECT_HEADER, "
        f"unitHash={unit_hash_preview}, hash={debug_hash_preview})"
    ) in result.stdout
    assert (
        "- main.frag:1:1 [opengl]: resolved system include release.inc -> "
        "includes/release.inc "
        f"(variant release, include-dir, define PROJECT_HEADER, "
        f"unitHash={unit_hash_preview}, hash={release_hash_preview})"
    ) in result.stdout


def test_validate_project_report_rejects_malformed_include_dependency_records(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (shader_dir / "main.frag").write_text(
        '#version 450\n#include "local.inc"\nvoid main() {}\n',
        encoding="utf-8",
    )

    payload = scan_project(repo).to_report(targets=["cgl"]).to_json()
    dependency = payload["units"][0]["includeDependencies"][0]
    dependency["kind"] = "module"
    dependency["status"] = "resolved"
    dependency["line"] = 0
    dependency["resolvedPath"] = "../outside.inc"
    dependency["resolvedHash"] = {"algorithm": "md5", "value": "not-a-sha"}
    dependency["resolvedFrom"] = "workspace"
    dependency["resolvedFromDefine"] = ""
    payload["summary"]["includeDependencyCount"] = 2
    payload["summary"]["includeDependenciesByKind"] = {"module": 1}
    payload["summary"]["includeDependenciesByResolvedFrom"] = {"workspace": 1}
    payload["summary"]["includeDependenciesBySourceBackend"] = {"cgl": 1}
    payload["summary"]["includeDependenciesBySourceBackendStatus"] = {
        "cgl": {"missing": 1}
    }
    report_path = repo / "bad-include-dependencies-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "units[0].includeDependencies[0].kind must be one of" in (
        diagnostic["message"]
    )
    assert "units[0].includeDependencies[0].line must be a positive integer" in (
        diagnostic["message"]
    )
    assert (
        "units[0].includeDependencies[0].resolvedPath must be repository-relative"
        in (diagnostic["message"])
    )
    assert "units[0].includeDependencies[0].resolvedHash.algorithm must be sha256" in (
        diagnostic["message"]
    )
    assert (
        "units[0].includeDependencies[0].resolvedHash.value must be a lowercase "
        "64-character hex digest"
    ) in diagnostic["message"]
    assert (
        "units[0].includeDependencies[0].resolvedFrom must be source or include-dir"
        in (diagnostic["message"])
    )
    assert "units[0].includeDependencies[0].resolvedFromDefine must be a string" in (
        diagnostic["message"]
    )
    assert "summary.includeDependencyCount must match unit include dependencies" in (
        diagnostic["message"]
    )
    assert "summary.includeDependenciesByKind must match unit include dependencies" in (
        diagnostic["message"]
    )
    assert (
        "summary.includeDependenciesByResolvedFrom must match unit include "
        "dependencies"
    ) in diagnostic["message"]
    assert (
        "summary.includeDependenciesBySourceBackend must match unit include "
        "dependencies"
    ) in diagnostic["message"]
    assert (
        "summary.includeDependenciesBySourceBackendStatus must match unit include "
        "dependencies"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_missing_current_include_dependencies(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (shader_dir / "local.inc").write_text("vec4 local_color();\n", encoding="utf-8")
    (shader_dir / "main.frag").write_text(
        '#version 450\n#include "local.inc"\nvoid main() {}\n',
        encoding="utf-8",
    )

    payload = scan_project(repo).to_report(targets=["cgl"]).to_json()
    payload["units"][0].pop("includeDependencies")
    payload["summary"]["includeDependencyCount"] = 0
    payload["summary"]["includeDependenciesByKind"] = {}
    payload["summary"]["includeDependenciesByStatus"] = {}
    payload["summary"]["includeDependenciesByResolvedFrom"] = {}
    payload["summary"]["includeDependenciesBySourceBackend"] = {}
    payload["summary"]["includeDependenciesBySourceBackendStatus"] = {}
    payload["summary"]["includeDependenciesByVariant"] = {}
    report_path = repo / "missing-current-include-dependencies-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "units[0].includeDependencies must include current include dependency "
        "unit source:2:1 resolved local include local.inc -> "
        "shaders/local.inc (source)"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_extra_current_include_dependencies(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (shader_dir / "local.inc").write_text("vec4 local_color();\n", encoding="utf-8")
    (shader_dir / "main.frag").write_text(
        '#version 450\n#include "local.inc"\nvoid main() {}\n',
        encoding="utf-8",
    )

    payload = scan_project(repo).to_report(targets=["cgl"]).to_json()
    dependency = dict(payload["units"][0]["includeDependencies"][0])
    dependency["line"] = 99
    payload["units"][0]["includeDependencies"].append(dependency)
    payload["summary"]["includeDependencyCount"] = 2
    payload["summary"]["includeDependenciesByKind"] = {"local": 2}
    payload["summary"]["includeDependenciesByStatus"] = {"resolved": 2}
    payload["summary"]["includeDependenciesByResolvedFrom"] = {"source": 2}
    payload["summary"]["includeDependenciesBySourceBackend"] = {"opengl": 2}
    payload["summary"]["includeDependenciesBySourceBackendStatus"] = {
        "opengl": {"resolved": 2}
    }
    report_path = repo / "extra-current-include-dependencies-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "units[0].includeDependencies contains include dependency not found "
        "in current source: unit source:99:1 resolved local include local.inc -> "
        "shaders/local.inc (source)"
    ) in diagnostic["message"]


def test_validate_project_report_labels_forged_define_include_provenance(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "local.inc").write_text("vec4 local_color();\n", encoding="utf-8")
    (repo / "main.frag").write_text('#include "local.inc"\n', encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]

            [project.defines]
            PROJECT_HEADER = "\\"local.inc\\""
            """).strip(),
        encoding="utf-8",
    )
    payload = (
        scan_project(load_project_config(repo)).to_report(targets=["cgl"]).to_json()
    )
    payload["units"][0]["includeDependencies"][0][
        "resolvedFromDefine"
    ] = "PROJECT_HEADER"
    report_path = repo / "forged-define-include-provenance-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "units[0].includeDependencies must include current include dependency "
        "unit source:1:1 resolved local include local.inc -> local.inc (source)"
    ) in diagnostic["message"]
    assert (
        "units[0].includeDependencies contains include dependency not found "
        "in current source: unit source:1:1 resolved local include local.inc -> "
        "local.inc (source, define PROJECT_HEADER)"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_undeclared_include_dependency_variants(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "debug.inc").write_text("vec4 debug_color();\n", encoding="utf-8")
    (repo / "main.frag").write_text("#include PROJECT_HEADER\n", encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]

            [project.variants.debug]
            PROJECT_HEADER = "\\"debug.inc\\""
            """).strip(),
        encoding="utf-8",
    )
    payload = (
        scan_project(load_project_config(repo)).to_report(targets=["cgl"]).to_json()
    )
    payload["units"][0]["includeDependencies"][0]["variant"] = "release"
    payload["summary"]["includeDependenciesByVariant"] = {"release": 1}
    report_path = repo / "forged-include-variant-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["diagnostics"][0]["code"] == "project.validate.invalid-report"
    assert (
        "units[0].includeDependencies[0].variant must match a declared project "
        "variant"
    ) in validation["diagnostics"][0]["message"]


def test_validate_project_report_rejects_stale_include_dependency_resolution(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (shader_dir / "local.inc").write_text("vec4 local_color();\n", encoding="utf-8")
    (shader_dir / "main.frag").write_text(
        '#version 450\n#include "local.inc"\nvoid main() {}\n',
        encoding="utf-8",
    )

    report = scan_project(repo).to_report(targets=["cgl"])
    report_path = repo / "include-dependencies-report.json"
    report.write_json(report_path)
    (shader_dir / "local.inc").unlink()

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "units[0].includeDependencies[0].status must match current include resolution"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_stale_include_dependency_hashes(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (shader_dir / "local.inc").write_text("vec4 local_color();\n", encoding="utf-8")
    (shader_dir / "main.frag").write_text(
        '#version 450\n#include "local.inc"\nvoid main() {}\n',
        encoding="utf-8",
    )

    report = scan_project(repo).to_report(targets=["cgl"])
    report_path = repo / "include-dependencies-report.json"
    report.write_json(report_path)
    (shader_dir / "local.inc").write_text("vec4 changed_color();\n", encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "units[0].includeDependencies[0].resolvedHash must match current "
        "include file"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_include_dependency_resolution_mismatches(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    include_dir = repo / "includes"
    shader_dir.mkdir(parents=True)
    include_dir.mkdir()
    (shader_dir / "local.inc").write_text("vec4 local_color();\n", encoding="utf-8")
    (include_dir / "shared.inc").write_text("vec4 shared_color();\n", encoding="utf-8")
    (shader_dir / "main.frag").write_text(
        '#version 450\n#include "local.inc"\nvoid main() {}\n',
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]
            include_dirs = ["includes"]
            """).strip(),
        encoding="utf-8",
    )

    payload = (
        scan_project(load_project_config(repo)).to_report(targets=["cgl"]).to_json()
    )
    dependency = payload["units"][0]["includeDependencies"][0]
    dependency["resolvedPath"] = "includes/shared.inc"
    dependency["resolvedFrom"] = "include-dir"
    report_path = repo / "bad-include-dependencies-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "units[0].includeDependencies[0].resolvedPath must match current "
        "include resolution"
    ) in diagnostic["message"]
    assert (
        "units[0].includeDependencies[0].resolvedFrom must match current "
        "include resolution"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_stale_define_include_dependencies(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    include_dir = repo / "includes"
    shader_dir.mkdir(parents=True)
    include_dir.mkdir()
    (include_dir / "shared.inc").write_text("vec4 shared_color();\n", encoding="utf-8")
    (include_dir / "other.inc").write_text("vec4 other_color();\n", encoding="utf-8")
    (shader_dir / "main.frag").write_text(
        "#version 450\n#include PROJECT_HEADER\nvoid main() {}\n",
        encoding="utf-8",
    )
    config_path = repo / "crosstl.toml"
    config_path.write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]
            include_dirs = ["includes"]

            [project.defines]
            PROJECT_HEADER = "<shared.inc>"
            """).strip(),
        encoding="utf-8",
    )

    report = scan_project(load_project_config(repo)).to_report(targets=["cgl"])
    report_path = repo / "include-dependencies-report.json"
    payload = report.to_json()
    payload["project"]["defines"]["PROJECT_HEADER"] = "<other.inc>"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "units[0].includeDependencies[0].include must match current project "
        "define include"
    ) in diagnostic["message"]


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


@pytest.mark.parametrize(
    ("toml_text", "message"),
    [
        (
            """
            [project]
            output_dir = 1
            """,
            "crosstl.toml project.output_dir must be a string",
        ),
        (
            """
            [project]
            output_dir = ""
            """,
            "crosstl.toml project.output_dir must be a non-empty string",
        ),
        (
            """
            [project]
            external_corpus_manifest = ["corpus.json"]
            """,
            "crosstl.toml project.external_corpus_manifest must be a string",
        ),
        (
            """
            [project]
            external_corpus_manifest = ""
            """,
            "crosstl.toml project.external_corpus_manifest must be a non-empty string",
        ),
        (
            """
            [project]
            source_roots = [" "]
            """,
            "project.source_roots entries must be non-empty strings",
        ),
        (
            """
            [project]
            include = [" "]
            """,
            "project.include entries must be non-empty strings",
        ),
        (
            """
            [project]
            exclude = [" "]
            """,
            "project.exclude entries must be non-empty strings",
        ),
        (
            """
            [project]
            targets = [" "]
            """,
            "project.targets entries must be non-empty strings",
        ),
        (
            """
            [project]
            include_dirs = [" "]
            """,
            "project.include_dirs entries must be non-empty strings",
        ),
        (
            """
            [project.sources]
            "gpu/*.shader" = 1
            """,
            (
                "crosstl.toml [project.sources] entries must map non-empty "
                "strings to strings"
            ),
        ),
        (
            """
            [project.sources]
            "" = "cgl"
            """,
            (
                "crosstl.toml [project.sources] entries must map non-empty "
                "strings to strings"
            ),
        ),
        (
            """
            [project.defines]
            USE_FAST_PATH = 1
            """,
            (
                "crosstl.toml [project.defines] entries must map non-empty "
                "strings to strings"
            ),
        ),
        (
            """
            [project.defines]
            "" = "1"
            """,
            (
                "crosstl.toml [project.defines] entries must map non-empty "
                "strings to strings"
            ),
        ),
        (
            """
            [project.variants.debug]
            MODE = 1
            """,
            (
                "crosstl.toml [project.variants.debug] entries must map "
                "non-empty strings to strings"
            ),
        ),
        (
            """
            [project.variants.debug]
            "" = "1"
            """,
            (
                "crosstl.toml [project.variants.debug] entries must map "
                "non-empty strings to strings"
            ),
        ),
        (
            """
            [project.variants."qa/profile"]
            MODE = 1
            """,
            (
                'crosstl.toml [project.variants["qa/profile"]] entries must map '
                "non-empty strings to strings"
            ),
        ),
        (
            """
            [project.variants.""]
            MODE = "debug"
            """,
            "crosstl.toml [project.variants] keys must be non-empty strings",
        ),
    ],
)
def test_project_config_rejects_malformed_string_metadata(tmp_path, toml_text, message):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "crosstl.toml").write_text(
        textwrap.dedent(toml_text).strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        load_project_config(repo)

    assert message in str(excinfo.value)


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
    assert payload["summary"]["unitsBySourceOverride"] == {"cgl": 1}
    assert payload["summary"]["skippedBySourceOverride"] == {}
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
    assert payload["summary"]["unitsBySourceOverride"] == {}
    assert payload["summary"]["skippedBySourceOverride"] == {"unknown-backend": 1}
    assert [diagnostic["code"] for diagnostic in payload["diagnostics"]] == [
        "project.config.unsupported-source-override",
        "project.scan.empty",
    ]
    assert payload["diagnostics"][0]["severity"] == "error"
    assert payload["diagnostics"][0]["location"]["file"] == "crosstl.toml"
    assert payload["diagnostics"][0]["missingCapabilities"] == ["source.override"]
    assert "unknown-backend" in payload["diagnostics"][0]["message"]


@pytest.mark.parametrize(
    ("skipped_override", "message"),
    [
        (
            None,
            (
                "skipped[0].sourceOverride must be recorded for "
                "unsupported-source-override records"
            ),
        ),
        (
            "other-backend",
            (
                "skipped[0].sourceOverride must match project.sourceOverrides "
                "for skipped[0].path"
            ),
        ),
    ],
)
def test_validate_project_report_rejects_inconsistent_skipped_source_overrides(
    tmp_path,
    skipped_override,
    message,
):
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

    payload = scan_project(load_project_config(repo)).to_report().to_json()
    if skipped_override is None:
        payload["skipped"][0].pop("sourceOverride")
    else:
        payload["skipped"][0]["sourceOverride"] = skipped_override
    report_path = repo / "scan-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert message in diagnostic["message"]


def test_scan_project_reports_unsupported_source_override_without_matches(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project.sources]
            "gpu/*.shader" = "unknown-backend"
            """).strip(),
        encoding="utf-8",
    )

    payload = scan_project(load_project_config(repo)).to_report().to_json()

    assert payload["skipped"] == []
    assert payload["summary"]["diagnosticsByCode"] == {
        "project.config.unsupported-source-override": 1,
        "project.scan.empty": 1,
    }
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.config.unsupported-source-override"
    assert "unknown-backend" in diagnostic["message"]
    assert diagnostic["location"]["file"] == "crosstl.toml"


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


def test_scan_project_reports_drive_relative_source_override_patterns(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project.sources]
            "C:tmp/*.shader" = "cgl"
            """).strip(),
        encoding="utf-8",
    )

    payload = scan_project(load_project_config(repo)).to_report().to_json()

    assert [unit["path"] for unit in payload["units"]] == ["simple.cgl"]
    assert payload["summary"]["diagnosticCounts"] == {
        "note": 0,
        "warning": 0,
        "error": 1,
    }
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == (
        "project.config.source-override-pattern-outside-project"
    )
    assert "C:tmp/*.shader" in diagnostic["message"]
    assert diagnostic["missingCapabilities"] == ["source.override"]


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
    assert payload["artifacts"][0]["includePathProcessing"] == {
        "status": "forwarded",
        "frontend": "lexer",
        "supportsIncludePaths": True,
        "includePathCount": 1,
    }
    assert payload["summary"]["includePathProcessingByStatus"] == {"forwarded": 1}
    assert payload["summary"]["includePathProcessingBySourceBackend"] == {
        "opengl": {"forwarded": 1}
    }
    assert "project_color" in output.read_text(encoding="utf-8")


def test_translate_project_filters_invalid_include_dirs_before_frontend(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    include_dir = repo / "includes"
    include_file = repo / "include-file"
    outside_dir = tmp_path / "outside-includes"
    shader_dir.mkdir(parents=True)
    include_dir.mkdir(parents=True)
    include_file.write_text("// not a directory\n", encoding="utf-8")
    outside_dir.mkdir()
    (shader_dir / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["shaders"]
            targets = ["opengl"]
            output_dir = "translated"
            include_dirs = [
                "includes",
                "missing-includes",
                "include-file",
                "../outside-includes",
            ]
            """).strip(),
        encoding="utf-8",
    )
    captured_include_paths = []

    def write_shader(
        file_path,
        backend="cgl",
        save_shader=None,
        format_output=True,
        source_backend=None,
        *,
        include_paths=None,
        defines=None,
    ):
        del file_path, backend, format_output, source_backend, defines
        captured_include_paths.append(list(include_paths or ()))
        Path(save_shader).write_text("// translated\n", encoding="utf-8")
        return "// translated\n"

    monkeypatch.setattr(project_pipeline, "translate", write_shader)

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "translated" / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    inspection = inspect_project_report(report_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    artifact_path = payload["artifacts"][0]["path"]

    assert captured_include_paths == [[str(include_dir.resolve())]]
    assert validation["success"] is True
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 3, "error": 0}
    assert payload["artifacts"][0]["includePathProcessing"] == {
        "status": "forwarded",
        "frontend": "lexer",
        "supportsIncludePaths": True,
        "includePathCount": 1,
    }
    assert payload["summary"]["includePathProcessingByStatus"] == {"forwarded": 1}
    assert payload["summary"]["includePathProcessingBySourceBackend"] == {
        "cgl": {"forwarded": 1}
    }
    assert inspection["includePathProcessing"] == {
        "available": True,
        "byStatus": {"forwarded": 1},
        "bySourceBackend": {"cgl": {"forwarded": 1}},
        "byVariant": {},
        "artifactCount": 1,
        "truncatedArtifactCount": 0,
        "artifacts": [
            {
                "source": "shaders/simple.cgl",
                "sourceBackend": "cgl",
                "target": "opengl",
                "path": artifact_path,
                "status": "forwarded",
                "frontend": "lexer",
                "supportsIncludePaths": True,
                "includePathCount": 1,
            }
        ],
        "notSupportedArtifactCount": 0,
        "truncatedNotSupportedArtifactCount": 0,
        "notSupportedArtifacts": [],
    }
    assert result.returncode == 0
    assert "Include path processing: forwarded=1" in result.stdout
    assert (
        "Include path processing by source backend: cgl=(forwarded=1)" in result.stdout
    )
    assert "Include path processing artifacts:" in result.stdout
    assert (
        f"- shaders/simple.cgl -> {artifact_path} "
        "(sourceBackend=cgl, target=opengl, status=forwarded, "
        "frontend=lexer, supportsIncludePaths=true, includePaths=1)"
    ) in result.stdout
    assert "Include path processing issues:" not in result.stdout
    assert payload["project"]["includeDirs"] == [
        "includes",
        "missing-includes",
        "include-file",
        "../outside-includes",
    ]
    assert payload["project"]["includeDirStatus"] == [
        {
            "path": "includes",
            "resolvedPath": str(include_dir.resolve()),
            "status": "active",
            "frontendVisible": True,
        },
        {
            "path": "missing-includes",
            "resolvedPath": str((repo / "missing-includes").resolve()),
            "status": "missing",
            "frontendVisible": False,
        },
        {
            "path": "include-file",
            "resolvedPath": str(include_file.resolve()),
            "status": "not-directory",
            "frontendVisible": False,
        },
        {
            "path": "../outside-includes",
            "resolvedPath": str(outside_dir.resolve()),
            "status": "outside-project",
            "frontendVisible": False,
        },
    ]
    assert payload["project"]["includeDirStatusCounts"] == {
        "active": 1,
        "missing": 1,
        "not-directory": 1,
        "outside-project": 1,
    }
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["diagnosticsByCode"] == {
        "project.config.include-dir-outside-project": 1,
        "project.config.include-dir-not-directory": 1,
        "project.config.missing-include-dir": 1,
    }
    assert payload["summary"]["missingCapabilityCounts"] == {
        "include.resolution": 3,
    }
    assert validation["missingCapabilityCounts"]["include.resolution"] == 3


def test_translate_project_records_include_forwarding_for_all_source_frontends(
    tmp_path, monkeypatch
):
    register_default_sources()
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    include_dir = repo / "includes"
    shader_dir.mkdir(parents=True)
    include_dir.mkdir()
    include_dir.joinpath("shared.inc").write_text("// shared\n", encoding="utf-8")
    source_names = sorted(SOURCE_REGISTRY.names())
    assert source_names
    source_overrides = []
    for source_name in source_names:
        shader_path = shader_dir / f"{source_name}.shader"
        shader_path.write_text(SIMPLE_CROSSL, encoding="utf-8")
        source_overrides.append(f'"shaders/{source_name}.shader" = "{source_name}"')
    source_override_text = "\n".join(source_overrides)
    (repo / "crosstl.toml").write_text(
        textwrap.dedent(f"""
            [project]
            source_roots = ["shaders"]
            targets = ["cgl"]
            output_dir = "translated"
            include_dirs = ["includes"]

            [project.sources]
            {source_override_text}
            """).strip(),
        encoding="utf-8",
    )
    calls = []

    def write_shader(
        file_path,
        backend="cgl",
        save_shader=None,
        format_output=True,
        source_backend=None,
        *,
        include_paths=None,
        defines=None,
    ):
        del file_path, backend, format_output, defines
        calls.append(
            {
                "sourceBackend": source_backend,
                "includePaths": list(include_paths or ()),
            }
        )
        Path(save_shader).write_text(
            f"// translated from {source_backend}\n",
            encoding="utf-8",
        )
        return f"// translated from {source_backend}\n"

    monkeypatch.setattr(project_pipeline, "translate", write_shader)

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "translated" / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    include_path = str(include_dir.resolve())
    expected_by_source = {}
    expected_by_status = {}
    unsupported_sources = []
    for source_name in source_names:
        supports_include_paths = SOURCE_REGISTRY.get(
            source_name
        ).supports_lexer_keyword("include_paths")
        status = "forwarded" if supports_include_paths else "not-supported"
        expected_by_source[source_name] = {status: 1}
        expected_by_status[status] = expected_by_status.get(status, 0) + 1
        if not supports_include_paths:
            unsupported_sources.append(source_name)

    artifacts_by_source = {
        artifact["sourceBackend"]: artifact for artifact in payload["artifacts"]
    }

    assert validation["success"] is True
    assert payload["summary"]["translatedCount"] == len(source_names)
    assert payload["summary"]["includePathProcessingByStatus"] == expected_by_status
    assert (
        payload["summary"]["includePathProcessingBySourceBackend"] == expected_by_source
    )
    assert payload["diagnosticCounts"] == {
        "note": 0,
        "warning": len(unsupported_sources),
        "error": 0,
    }
    assert payload["summary"]["diagnosticsByCode"] == (
        {"project.translate.include-paths-not-forwarded": len(unsupported_sources)}
        if unsupported_sources
        else {}
    )
    assert payload["summary"]["missingCapabilityCounts"] == (
        {"include.forwarding": len(unsupported_sources)} if unsupported_sources else {}
    )
    assert [
        {
            "sourceBackend": call["sourceBackend"],
            "includePaths": call["includePaths"],
        }
        for call in calls
    ] == [
        {"sourceBackend": source_name, "includePaths": [include_path]}
        for source_name in source_names
    ]
    assert {
        diagnostic["location"]["file"]
        for diagnostic in payload["diagnostics"]
        if diagnostic["code"] == "project.translate.include-paths-not-forwarded"
    } == {f"shaders/{source_name}.shader" for source_name in unsupported_sources}
    for source_name, artifact in artifacts_by_source.items():
        include_path_processing = artifact["includePathProcessing"]
        supports_include_paths = SOURCE_REGISTRY.get(
            source_name
        ).supports_lexer_keyword("include_paths")
        assert include_path_processing == {
            "status": "forwarded" if supports_include_paths else "not-supported",
            "frontend": "lexer",
            "supportsIncludePaths": supports_include_paths,
            "includePathCount": 1,
        }


def test_translate_project_records_define_forwarding_for_all_source_frontends(
    tmp_path, monkeypatch
):
    register_default_sources()
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    source_names = sorted(SOURCE_REGISTRY.names())
    assert source_names
    source_overrides = []
    for source_name in source_names:
        shader_path = shader_dir / f"{source_name}.shader"
        shader_path.write_text(SIMPLE_CROSSL, encoding="utf-8")
        source_overrides.append(f'"shaders/{source_name}.shader" = "{source_name}"')
    source_override_text = "\n".join(source_overrides)
    (repo / "crosstl.toml").write_text(
        textwrap.dedent(f"""
            [project]
            source_roots = ["shaders"]
            targets = ["cgl"]
            output_dir = "translated"

            [project.defines]
            MODE = "debug"
            USE_FAST_PATH = "1"

            [project.sources]
            {source_override_text}
            """).strip(),
        encoding="utf-8",
    )
    calls = []

    def write_shader(
        file_path,
        backend="cgl",
        save_shader=None,
        format_output=True,
        source_backend=None,
        *,
        include_paths=None,
        defines=None,
    ):
        del file_path, backend, format_output, include_paths
        calls.append(
            {
                "sourceBackend": source_backend,
                "defines": dict(defines or {}),
            }
        )
        Path(save_shader).write_text(
            f"// translated from {source_backend}\n",
            encoding="utf-8",
        )
        return f"// translated from {source_backend}\n"

    monkeypatch.setattr(project_pipeline, "translate", write_shader)

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "translated" / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    defines = {"MODE": "debug", "USE_FAST_PATH": "1"}
    expected_by_source = {}
    expected_by_status = {}
    unsupported_sources = []
    for source_name in source_names:
        supports_defines = SOURCE_REGISTRY.get(source_name).supports_lexer_keyword(
            "defines"
        )
        status = "forwarded" if supports_defines else "not-supported"
        expected_by_source[source_name] = {status: 1}
        expected_by_status[status] = expected_by_status.get(status, 0) + 1
        if not supports_defines:
            unsupported_sources.append(source_name)

    artifacts_by_source = {
        artifact["sourceBackend"]: artifact for artifact in payload["artifacts"]
    }

    assert validation["success"] is True
    assert payload["summary"]["translatedCount"] == len(source_names)
    assert payload["summary"]["defineProcessingByStatus"] == expected_by_status
    assert payload["summary"]["defineProcessingBySourceBackend"] == expected_by_source
    assert payload["summary"]["defineProcessingByVariant"] == {}
    assert payload["diagnosticCounts"] == {
        "note": 0,
        "warning": len(unsupported_sources),
        "error": 0,
    }
    assert payload["summary"]["diagnosticsByCode"] == (
        {"project.translate.defines-not-forwarded": len(unsupported_sources)}
        if unsupported_sources
        else {}
    )
    assert payload["summary"]["missingCapabilityCounts"] == (
        {"macro.defines": len(unsupported_sources)} if unsupported_sources else {}
    )
    assert validation["diagnosticsByCode"] == (
        {"project.translate.defines-not-forwarded": len(unsupported_sources)}
        if unsupported_sources
        else {}
    )
    assert validation["missingCapabilityCounts"] == (
        {"macro.defines": len(unsupported_sources)} if unsupported_sources else {}
    )
    assert [
        {
            "sourceBackend": call["sourceBackend"],
            "defines": call["defines"],
        }
        for call in calls
    ] == [
        {"sourceBackend": source_name, "defines": defines}
        for source_name in source_names
    ]
    assert {
        diagnostic["location"]["file"]
        for diagnostic in payload["diagnostics"]
        if diagnostic["code"] == "project.translate.defines-not-forwarded"
    } == {f"shaders/{source_name}.shader" for source_name in unsupported_sources}
    for source_name, artifact in artifacts_by_source.items():
        define_processing = artifact["defineProcessing"]
        supports_defines = SOURCE_REGISTRY.get(source_name).supports_lexer_keyword(
            "defines"
        )
        assert artifact["defines"] == defines
        assert define_processing == {
            "status": "forwarded" if supports_defines else "not-supported",
            "frontend": "lexer",
            "supportsDefines": supports_defines,
            "defineCount": 2,
        }


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
    inspection = inspect_project_report(report_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert validation["success"] is True
    assert payload["summary"]["artifactCount"] == 2
    assert payload["summary"]["translatedCount"] == 2
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 0}
    assert payload["summary"]["artifactsByVariant"] == {
        "debug": {
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
        },
        "release": {
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
        },
    }
    assert validation["artifactStatusByVariant"] == {
        "debug": {"artifactCount": 1, "okCount": 1, "failedCount": 0},
        "release": {"artifactCount": 1, "okCount": 1, "failedCount": 0},
    }
    assert [artifact["variant"] for artifact in payload["artifacts"]] == [
        "debug",
        "release",
    ]
    assert [artifact["defines"] for artifact in payload["artifacts"]] == [
        {"MODE": "debug", "USE_FAST_PATH": "1"},
        {"MODE": "base", "USE_FAST_PATH": "1"},
    ]
    assert [artifact["defineProcessing"] for artifact in payload["artifacts"]] == [
        {
            "status": "forwarded",
            "frontend": "lexer",
            "supportsDefines": True,
            "defineCount": 2,
        },
        {
            "status": "forwarded",
            "frontend": "lexer",
            "supportsDefines": True,
            "defineCount": 2,
        },
    ]
    assert payload["summary"]["defineProcessingByStatus"] == {"forwarded": 2}
    assert payload["summary"]["defineProcessingBySourceBackend"] == {
        "cgl": {"forwarded": 2}
    }
    assert payload["summary"]["defineProcessingByVariant"] == {
        "debug": {"forwarded": 1},
        "release": {"forwarded": 1},
    }
    assert payload["summary"]["includePathProcessingByVariant"] == {
        "debug": {"not-requested": 1},
        "release": {"not-requested": 1},
    }
    assert inspection["defineProcessing"]["byVariant"] == {
        "debug": {"forwarded": 1},
        "release": {"forwarded": 1},
    }
    assert inspection["defineProcessing"]["artifactCount"] == 2
    assert inspection["defineProcessing"]["truncatedArtifactCount"] == 0
    assert inspection["defineProcessing"]["artifacts"] == [
        {
            "source": "simple.cgl",
            "sourceBackend": "cgl",
            "target": "opengl",
            "path": "translated/opengl/debug/simple.glsl",
            "status": "forwarded",
            "frontend": "lexer",
            "supportsDefines": True,
            "defineCount": 2,
            "defineNames": ["MODE", "USE_FAST_PATH"],
            "variant": "debug",
        },
        {
            "source": "simple.cgl",
            "sourceBackend": "cgl",
            "target": "opengl",
            "path": "translated/opengl/release/simple.glsl",
            "status": "forwarded",
            "frontend": "lexer",
            "supportsDefines": True,
            "defineCount": 2,
            "defineNames": ["MODE", "USE_FAST_PATH"],
            "variant": "release",
        },
    ]
    assert all(
        "defines" not in artifact
        for artifact in inspection["defineProcessing"]["artifacts"]
    )
    assert '"base"' not in json.dumps(inspection["defineProcessing"]["artifacts"])
    assert inspection["includePathProcessing"]["byVariant"] == {
        "debug": {"not-requested": 1},
        "release": {"not-requested": 1},
    }
    assert result.returncode == 0
    assert "Define processing by source backend: cgl=(forwarded=2)" in result.stdout
    assert (
        "Define processing by variant: debug=(forwarded=1), release=(forwarded=1)"
    ) in result.stdout
    assert "Define processing artifacts:" in result.stdout
    assert (
        "- simple.cgl -> translated/opengl/debug/simple.glsl "
        "(sourceBackend=cgl, target=opengl, variant=debug, status=forwarded, "
        "frontend=lexer, supportsDefines=true, defines=2, "
        "defineNames=MODE,USE_FAST_PATH)"
    ) in result.stdout
    assert (
        "- simple.cgl -> translated/opengl/release/simple.glsl "
        "(sourceBackend=cgl, target=opengl, variant=release, status=forwarded, "
        "frontend=lexer, supportsDefines=true, defines=2, "
        "defineNames=MODE,USE_FAST_PATH)"
    ) in result.stdout
    assert (
        "Include path processing by variant: "
        "debug=(not-requested=1), release=(not-requested=1)"
    ) in result.stdout
    assert (
        "Include path processing by source backend: cgl=(not-requested=2)"
        in result.stdout
    )
    assert [artifact["path"] for artifact in payload["artifacts"]] == [
        "translated/opengl/debug/simple.glsl",
        "translated/opengl/release/simple.glsl",
    ]
    assert validation["validation"]["artifacts"] == [
        {
            "source": "simple.cgl",
            "sourceBackend": "cgl",
            "target": "opengl",
            "path": "translated/opengl/debug/simple.glsl",
            "exists": True,
            "status": "ok",
            "sourceHashStatus": "ok",
            "generatedHashStatus": "ok",
            "sourceMapStatus": "ok",
            "sourceRemapStatus": "not-recorded",
            "variant": "debug",
        },
        {
            "source": "simple.cgl",
            "sourceBackend": "cgl",
            "target": "opengl",
            "path": "translated/opengl/release/simple.glsl",
            "exists": True,
            "status": "ok",
            "sourceHashStatus": "ok",
            "generatedHashStatus": "ok",
            "sourceMapStatus": "ok",
            "sourceRemapStatus": "not-recorded",
            "variant": "release",
        },
    ]
    assert validation["artifactStatusBySourceBackend"] == {
        "cgl": {"artifactCount": 2, "okCount": 2, "failedCount": 0}
    }
    assert validation["validation"]["summary"] == {
        "artifactCount": 2,
        "okCount": 2,
        "failedCount": 0,
        "sourceHashStatusCounts": _source_hash_status_counts(ok=2),
        "generatedHashStatusCounts": _generated_hash_status_counts(ok=2),
        "sourceMapStatusCounts": _source_map_status_counts(ok=2),
        "sourceRemapStatusCounts": _source_remap_status_counts(**{"not-recorded": 2}),
    }
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


def test_translate_project_limits_named_variants_to_selected(tmp_path, monkeypatch):
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

    report = translate_project(load_project_config(repo), variants=["debug", "debug"])
    payload = report.to_json()
    report_path = repo / "translated" / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    inspection = inspect_project_report(report_path)
    inspection_text = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert validation["success"] is True
    assert inspection_text.returncode == 0
    assert payload["project"]["variants"] == {"debug": {"MODE": "debug"}}
    assert payload["project"]["variantCount"] == 1
    assert payload["project"]["variantDefineCounts"] == {"debug": 1}
    assert payload["project"]["selectedVariants"] == ["debug"]
    assert inspection["report"]["project"]["selectedVariants"] == ["debug"]
    assert "Selected variants: debug" in inspection_text.stdout
    assert payload["summary"]["artifactCount"] == 1
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["artifactsByVariant"] == {
        "debug": {
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
        }
    }
    assert payload["summary"]["defineProcessingByVariant"] == {
        "debug": {"forwarded": 1}
    }
    assert payload["summary"]["includePathProcessingByVariant"] == {
        "debug": {"not-requested": 1}
    }
    assert payload["artifactMatrix"]["variantCount"] == 1
    assert payload["artifactMatrix"]["statusBySourceBackend"] == {
        "cgl": {
            "expectedArtifactCount": 1,
            "emittedArtifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
            "missingArtifactCount": 0,
            "extraArtifactCount": 0,
            "complete": True,
        }
    }
    assert payload["artifactMatrix"]["statusByVariant"] == {
        "debug": {
            "expectedArtifactCount": 1,
            "emittedArtifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
            "missingArtifactCount": 0,
            "extraArtifactCount": 0,
            "complete": True,
        }
    }
    assert [artifact["variant"] for artifact in payload["artifacts"]] == ["debug"]
    assert [artifact["path"] for artifact in payload["artifacts"]] == [
        "translated/opengl/debug/simple.glsl"
    ]
    assert payload["artifacts"][0]["defines"] == {
        "MODE": "debug",
        "USE_FAST_PATH": "1",
    }
    assert (repo / "translated" / "opengl" / "debug" / "simple.glsl").exists()
    assert not (repo / "translated" / "opengl" / "release" / "simple.glsl").exists()
    assert "release" not in json.dumps(payload)


def test_translate_project_sanitizes_variant_output_segments(tmp_path, monkeypatch):
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

            [project.variants."qa/profile"]
            MODE = "qa"
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
    variant_segment = project_pipeline._variant_output_segment("qa/profile")
    expected_path = f"translated/opengl/{variant_segment}/simple.glsl"

    assert validation["success"] is True
    assert variant_segment != "qa/profile"
    assert "/" not in variant_segment
    assert payload["project"]["variants"] == {"qa/profile": {"MODE": "qa"}}
    assert payload["summary"]["artifactsByVariant"] == {
        "qa/profile": {
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
        }
    }
    assert payload["artifactMatrix"]["statusByVariant"] == {
        "qa/profile": {
            "expectedArtifactCount": 1,
            "emittedArtifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
            "missingArtifactCount": 0,
            "extraArtifactCount": 0,
            "complete": True,
        }
    }
    assert payload["artifacts"][0]["variant"] == "qa/profile"
    assert payload["artifacts"][0]["path"] == expected_path
    assert payload["artifacts"][0]["defines"] == {"MODE": "qa"}
    assert validation["artifactStatusByVariant"] == {
        "qa/profile": {"artifactCount": 1, "okCount": 1, "failedCount": 0}
    }
    assert (repo / expected_path).exists()
    assert not (
        repo / "translated" / "opengl" / "qa" / "profile" / "simple.glsl"
    ).exists()


def test_translate_project_rejects_unknown_selected_variant(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]

            [project.variants.debug]
            MODE = "debug"
            """).strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "selected project variant is not declared in project config: "
            "profile \\(available: debug\\)"
        ),
    ):
        translate_project(load_project_config(repo), variants=["profile"])


def test_translate_project_named_variants_apply_crossgl_defines(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "macro.cgl").write_text(
        textwrap.dedent("""
            shader MacroShader {
                fragment {
                    vec4 main() @ gl_FragColor {
            #if USE_RED
                        return vec4(1.0, 0.0, 0.0, 1.0);
            #else
                        return vec4(0.0, 0.0, 1.0, 1.0);
            #endif
                    }
                }
            }
            """).strip(),
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["opengl"]
            output_dir = "translated"

            [project.defines]
            USE_RED = "0"

            [project.variants.debug]
            USE_RED = "1"

            [project.variants.release]
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "translated" / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    debug_output = (repo / "translated" / "opengl" / "debug" / "macro.glsl").read_text(
        encoding="utf-8"
    )
    release_output = (
        repo / "translated" / "opengl" / "release" / "macro.glsl"
    ).read_text(encoding="utf-8")

    assert validation["success"] is True
    assert payload["summary"]["artifactCount"] == 2
    assert payload["summary"]["translatedCount"] == 2
    assert "fragColor = vec4(1.0, 0.0, 0.0, 1.0);" in debug_output
    assert "fragColor = vec4(0.0, 0.0, 1.0, 1.0);" in release_output
    assert "#if" not in debug_output
    assert "#if" not in release_output


def test_translate_project_named_variants_apply_native_opengl_preprocessor(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    include_dir = repo / "includes"
    shader_dir.mkdir(parents=True)
    include_dir.mkdir()
    (include_dir / "palette.glsl").write_text(
        textwrap.dedent("""
            vec4 debug_color() { return vec4(1.0, 0.0, 0.0, 1.0); }
            vec4 release_color() { return vec4(0.0, 0.0, 1.0, 1.0); }
            """).strip() + "\n",
        encoding="utf-8",
    )
    (shader_dir / "main.frag").write_text(
        textwrap.dedent("""
            #version 450
            #include <palette.glsl>
            layout(location = 0) out vec4 outColor;

            void main()
            {
            #if USE_DEBUG_COLOR
                outColor = debug_color();
            #else
                outColor = release_color();
            #endif
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
            USE_DEBUG_COLOR = "0"

            [project.variants.debug]
            USE_DEBUG_COLOR = "1"

            [project.variants.release]
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "translated" / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    debug_output = (
        repo / "translated" / "cgl" / "debug" / "shaders" / "main.cgl"
    ).read_text(encoding="utf-8")
    release_output = (
        repo / "translated" / "cgl" / "release" / "shaders" / "main.cgl"
    ).read_text(encoding="utf-8")

    assert validation["success"] is True
    assert payload["summary"]["artifactCount"] == 2
    assert payload["summary"]["translatedCount"] == 2
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 0}
    assert payload["summary"]["includeDependencyCount"] == 2
    assert payload["summary"]["includeDependenciesByKind"] == {"system": 2}
    assert payload["summary"]["includeDependenciesByStatus"] == {"resolved": 2}
    assert payload["summary"]["includeDependenciesByResolvedFrom"] == {"include-dir": 2}
    assert payload["summary"]["includeDependenciesByVariant"] == {
        "debug": 1,
        "release": 1,
    }
    assert payload["summary"]["defineProcessingBySourceBackend"] == {
        "opengl": {"forwarded": 2}
    }
    assert payload["summary"]["defineProcessingByVariant"] == {
        "debug": {"forwarded": 1},
        "release": {"forwarded": 1},
    }
    assert payload["summary"]["includePathProcessingBySourceBackend"] == {
        "opengl": {"forwarded": 2}
    }
    assert payload["summary"]["includePathProcessingByVariant"] == {
        "debug": {"forwarded": 1},
        "release": {"forwarded": 1},
    }
    assert payload["units"][0]["includeDependencies"] == [
        {
            "include": "palette.glsl",
            "kind": "system",
            "status": "resolved",
            "line": 2,
            "column": 1,
            "variant": "debug",
            "resolvedPath": "includes/palette.glsl",
            "resolvedHash": project_pipeline._source_hash(include_dir / "palette.glsl"),
            "resolvedFrom": "include-dir",
        },
        {
            "include": "palette.glsl",
            "kind": "system",
            "status": "resolved",
            "line": 2,
            "column": 1,
            "variant": "release",
            "resolvedPath": "includes/palette.glsl",
            "resolvedHash": project_pipeline._source_hash(include_dir / "palette.glsl"),
            "resolvedFrom": "include-dir",
        },
    ]
    assert [artifact["defines"] for artifact in payload["artifacts"]] == [
        {"USE_DEBUG_COLOR": "1"},
        {"USE_DEBUG_COLOR": "0"},
    ]
    assert "outColor = debug_color();" in debug_output
    assert "outColor = release_color();" not in debug_output
    assert "outColor = release_color();" in release_output
    assert "outColor = debug_color();" not in release_output
    assert "#include" not in debug_output
    assert "#include" not in release_output
    assert "#if" not in debug_output
    assert "#if" not in release_output


def test_translate_project_named_variants_apply_native_directx_preprocessor(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    include_dir = repo / "includes"
    shader_dir.mkdir(parents=True)
    include_dir.mkdir()
    include_path = include_dir / "palette.hlsli"
    include_path.write_text(
        textwrap.dedent("""
            #if USE_DEBUG_COLOR
            float4 selected_color() { return float4(1.0, 0.0, 0.0, 1.0); }
            #else
            float4 selected_color() { return float4(0.0, 0.0, 1.0, 1.0); }
            #endif
            """).strip() + "\n",
        encoding="utf-8",
    )
    (shader_dir / "main.hlsl").write_text(
        textwrap.dedent("""
            #include "palette.hlsli"

            float4 main() : SV_Target
            {
                return selected_color();
            }
            """).strip() + "\n",
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
            USE_DEBUG_COLOR = "0"

            [project.variants.debug]
            USE_DEBUG_COLOR = "1"

            [project.variants.release]
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "translated" / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    debug_output = (
        repo / "translated" / "cgl" / "debug" / "shaders" / "main.cgl"
    ).read_text(encoding="utf-8")
    release_output = (
        repo / "translated" / "cgl" / "release" / "shaders" / "main.cgl"
    ).read_text(encoding="utf-8")

    assert validation["success"] is True
    assert payload["summary"]["artifactCount"] == 2
    assert payload["summary"]["translatedCount"] == 2
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 0}
    assert payload["summary"]["includeDependencyCount"] == 2
    assert payload["summary"]["includeDependenciesByKind"] == {"local": 2}
    assert payload["summary"]["includeDependenciesByStatus"] == {"resolved": 2}
    assert payload["summary"]["includeDependenciesByResolvedFrom"] == {"include-dir": 2}
    assert payload["summary"]["includeDependenciesBySourceBackend"] == {"directx": 2}
    assert payload["summary"]["includeDependenciesBySourceBackendStatus"] == {
        "directx": {"resolved": 2}
    }
    assert payload["summary"]["includeDependenciesByVariant"] == {
        "debug": 1,
        "release": 1,
    }
    assert payload["summary"]["defineProcessingBySourceBackend"] == {
        "directx": {"forwarded": 2}
    }
    assert payload["summary"]["defineProcessingByVariant"] == {
        "debug": {"forwarded": 1},
        "release": {"forwarded": 1},
    }
    assert payload["summary"]["includePathProcessingBySourceBackend"] == {
        "directx": {"forwarded": 2}
    }
    assert payload["summary"]["includePathProcessingByVariant"] == {
        "debug": {"forwarded": 1},
        "release": {"forwarded": 1},
    }
    assert payload["units"][0]["includeDependencies"] == [
        {
            "include": "palette.hlsli",
            "kind": "local",
            "status": "resolved",
            "line": 1,
            "column": 1,
            "variant": "debug",
            "resolvedPath": "includes/palette.hlsli",
            "resolvedHash": project_pipeline._source_hash(include_path),
            "resolvedFrom": "include-dir",
        },
        {
            "include": "palette.hlsli",
            "kind": "local",
            "status": "resolved",
            "line": 1,
            "column": 1,
            "variant": "release",
            "resolvedPath": "includes/palette.hlsli",
            "resolvedHash": project_pipeline._source_hash(include_path),
            "resolvedFrom": "include-dir",
        },
    ]
    assert [artifact["defines"] for artifact in payload["artifacts"]] == [
        {"USE_DEBUG_COLOR": "1"},
        {"USE_DEBUG_COLOR": "0"},
    ]
    assert "vec4(1.0, 0.0, 0.0, 1.0)" in debug_output
    assert "vec4(0.0, 0.0, 1.0, 1.0)" not in debug_output
    assert "vec4(0.0, 0.0, 1.0, 1.0)" in release_output
    assert "vec4(1.0, 0.0, 0.0, 1.0)" not in release_output
    assert "#include" not in debug_output
    assert "#include" not in release_output
    assert "#if" not in debug_output
    assert "#if" not in release_output


def test_translate_project_named_variants_apply_native_metal_preprocessor(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    include_dir = repo / "includes"
    shader_dir.mkdir(parents=True)
    include_dir.mkdir()
    include_path = include_dir / "palette.metal"
    include_path.write_text(
        textwrap.dedent("""
            #if USE_DEBUG_COLOR
            float4 selected_color() { return float4(1.0, 0.0, 0.0, 1.0); }
            #else
            float4 selected_color() { return float4(0.0, 0.0, 1.0, 1.0); }
            #endif
            """).strip() + "\n",
        encoding="utf-8",
    )
    (shader_dir / "main.metal").write_text(
        textwrap.dedent("""
            #include "palette.metal"

            fragment float4 fragment_main() {
                return selected_color();
            }
            """).strip() + "\n",
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
            USE_DEBUG_COLOR = "0"

            [project.variants.debug]
            USE_DEBUG_COLOR = "1"

            [project.variants.release]
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "translated" / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    debug_output = (
        repo / "translated" / "cgl" / "debug" / "shaders" / "main.cgl"
    ).read_text(encoding="utf-8")
    release_output = (
        repo / "translated" / "cgl" / "release" / "shaders" / "main.cgl"
    ).read_text(encoding="utf-8")

    assert validation["success"] is True
    assert payload["summary"]["artifactCount"] == 2
    assert payload["summary"]["translatedCount"] == 2
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 0}
    assert payload["summary"]["includeDependencyCount"] == 2
    assert payload["summary"]["includeDependenciesByKind"] == {"local": 2}
    assert payload["summary"]["includeDependenciesByStatus"] == {"resolved": 2}
    assert payload["summary"]["includeDependenciesByResolvedFrom"] == {"include-dir": 2}
    assert payload["summary"]["includeDependenciesBySourceBackend"] == {"metal": 2}
    assert payload["summary"]["includeDependenciesBySourceBackendStatus"] == {
        "metal": {"resolved": 2}
    }
    assert payload["summary"]["includeDependenciesByVariant"] == {
        "debug": 1,
        "release": 1,
    }
    assert payload["summary"]["defineProcessingBySourceBackend"] == {
        "metal": {"forwarded": 2}
    }
    assert payload["summary"]["defineProcessingByVariant"] == {
        "debug": {"forwarded": 1},
        "release": {"forwarded": 1},
    }
    assert payload["summary"]["includePathProcessingBySourceBackend"] == {
        "metal": {"forwarded": 2}
    }
    assert payload["summary"]["includePathProcessingByVariant"] == {
        "debug": {"forwarded": 1},
        "release": {"forwarded": 1},
    }
    assert payload["units"][0]["includeDependencies"] == [
        {
            "include": "palette.metal",
            "kind": "local",
            "status": "resolved",
            "line": 1,
            "column": 1,
            "variant": "debug",
            "resolvedPath": "includes/palette.metal",
            "resolvedHash": project_pipeline._source_hash(include_path),
            "resolvedFrom": "include-dir",
        },
        {
            "include": "palette.metal",
            "kind": "local",
            "status": "resolved",
            "line": 1,
            "column": 1,
            "variant": "release",
            "resolvedPath": "includes/palette.metal",
            "resolvedHash": project_pipeline._source_hash(include_path),
            "resolvedFrom": "include-dir",
        },
    ]
    assert [artifact["defines"] for artifact in payload["artifacts"]] == [
        {"USE_DEBUG_COLOR": "1"},
        {"USE_DEBUG_COLOR": "0"},
    ]
    assert "vec4(1.0, 0.0, 0.0, 1.0)" in debug_output
    assert "vec4(0.0, 0.0, 1.0, 1.0)" not in debug_output
    assert "vec4(0.0, 0.0, 1.0, 1.0)" in release_output
    assert "vec4(1.0, 0.0, 0.0, 1.0)" not in release_output
    assert "#include" not in debug_output
    assert "#include" not in release_output
    assert "#if" not in debug_output
    assert "#if" not in release_output


@pytest.mark.parametrize(
    (
        "source_backend",
        "extension",
        "runtime_header",
        "runtime_include",
        "local_header",
    ),
    (
        (
            "cuda",
            ".cu",
            "<cuda_runtime.h>",
            "cuda_runtime.h",
            "constants.cuh",
        ),
        (
            "hip",
            ".hip",
            "<hip/hip_runtime.h>",
            "hip/hip_runtime.h",
            "constants.hip",
        ),
    ),
)
def test_translate_project_named_variants_apply_cuda_hip_preprocessor(
    tmp_path,
    source_backend,
    extension,
    runtime_header,
    runtime_include,
    local_header,
):
    repo = tmp_path / "repo"
    source_dir = repo / "src"
    include_dir = repo / "include"
    source_dir.mkdir(parents=True)
    include_dir.mkdir()
    include_path = include_dir / local_header
    include_path.write_text(
        textwrap.dedent("""
            #define SCALE_VALUE 4
            __device__ float scale(float value) { return value * SCALE_VALUE; }
            """).strip() + "\n",
        encoding="utf-8",
    )
    (source_dir / f"kernel{extension}").write_text(
        textwrap.dedent(f"""
            #include {runtime_header}
            #include "{local_header}"
            #if USE_FAST_PATH
            __global__ void selected_kernel(float* out) {{ out[0] = scale(1.0f); }}
            #else
            __global__ void rejected_kernel(float* out) {{ out[0] = scale(0.0f); }}
            #endif
            """).strip() + "\n",
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["src"]
            targets = ["cgl"]
            output_dir = "translated"
            include_dirs = ["include"]

            [project.defines]
            USE_FAST_PATH = "0"

            [project.variants.fast]
            USE_FAST_PATH = "1"

            [project.variants.safe]
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "translated" / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    fast_output = (
        repo / "translated" / "cgl" / "fast" / "src" / "kernel.cgl"
    ).read_text(encoding="utf-8")
    safe_output = (
        repo / "translated" / "cgl" / "safe" / "src" / "kernel.cgl"
    ).read_text(encoding="utf-8")

    assert validation["success"] is True
    assert payload["summary"]["artifactCount"] == 2
    assert payload["summary"]["translatedCount"] == 2
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 0}
    assert payload["summary"]["includeDependencyCount"] == 4
    assert payload["summary"]["includeDependenciesByKind"] == {
        "local": 2,
        "system": 2,
    }
    assert payload["summary"]["includeDependenciesByStatus"] == {
        "resolved": 2,
        "system": 2,
    }
    assert payload["summary"]["includeDependenciesByResolvedFrom"] == {"include-dir": 2}
    assert payload["summary"]["includeDependenciesBySourceBackend"] == {
        source_backend: 4
    }
    assert payload["summary"]["includeDependenciesBySourceBackendStatus"] == {
        source_backend: {"resolved": 2, "system": 2}
    }
    assert payload["summary"]["includeDependenciesByVariant"] == {
        "fast": 2,
        "safe": 2,
    }
    assert payload["summary"]["defineProcessingBySourceBackend"] == {
        source_backend: {"forwarded": 2}
    }
    assert payload["summary"]["defineProcessingByVariant"] == {
        "fast": {"forwarded": 1},
        "safe": {"forwarded": 1},
    }
    assert payload["summary"]["includePathProcessingBySourceBackend"] == {
        source_backend: {"forwarded": 2}
    }
    assert payload["summary"]["includePathProcessingByVariant"] == {
        "fast": {"forwarded": 1},
        "safe": {"forwarded": 1},
    }
    assert payload["units"][0]["includeDependencies"] == [
        {
            "include": runtime_include,
            "kind": "system",
            "status": "system",
            "line": 1,
            "column": 1,
            "variant": "fast",
        },
        {
            "include": local_header,
            "kind": "local",
            "status": "resolved",
            "line": 2,
            "column": 1,
            "variant": "fast",
            "resolvedPath": f"include/{local_header}",
            "resolvedHash": project_pipeline._source_hash(include_path),
            "resolvedFrom": "include-dir",
        },
        {
            "include": runtime_include,
            "kind": "system",
            "status": "system",
            "line": 1,
            "column": 1,
            "variant": "safe",
        },
        {
            "include": local_header,
            "kind": "local",
            "status": "resolved",
            "line": 2,
            "column": 1,
            "variant": "safe",
            "resolvedPath": f"include/{local_header}",
            "resolvedHash": project_pipeline._source_hash(include_path),
            "resolvedFrom": "include-dir",
        },
    ]
    assert [artifact["defines"] for artifact in payload["artifacts"]] == [
        {"USE_FAST_PATH": "1"},
        {"USE_FAST_PATH": "0"},
    ]
    assert "f32 scale(f32 value)" in fast_output
    assert "selected_kernel" in fast_output
    assert "rejected_kernel" not in fast_output
    assert "rejected_kernel" in safe_output
    assert "selected_kernel" not in safe_output
    assert "SCALE_VALUE" not in fast_output
    assert "SCALE_VALUE" not in safe_output


def test_translate_project_records_define_processing_without_frontend_support(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "shader.rs").write_text("fn main() {}\n", encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            output_dir = "translated"

            [project.defines]
            ENABLE_PATH = "1"
            """).strip(),
        encoding="utf-8",
    )

    def write_artifact(
        file_path,
        backend="cgl",
        save_shader=None,
        format_output=True,
        source_backend=None,
        *,
        include_paths=None,
        defines=None,
    ):
        del file_path, backend, format_output, source_backend, include_paths, defines
        Path(save_shader).write_text("// translated\n", encoding="utf-8")
        return "// translated\n"

    monkeypatch.setattr(project_pipeline, "translate", write_artifact)

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "translated" / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    inspection = inspect_project_report(report_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert validation["success"] is True
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 1, "error": 0}
    assert payload["summary"]["diagnosticsByCode"] == {
        "project.translate.defines-not-forwarded": 1
    }
    assert payload["summary"]["missingCapabilityCounts"] == {"macro.defines": 1}
    assert (
        payload["diagnostics"][0]["code"] == "project.translate.defines-not-forwarded"
    )
    assert payload["diagnostics"][0]["missingCapabilities"] == ["macro.defines"]
    assert payload["diagnostics"][0]["location"]["file"] == "shader.rs"
    assert "not forwarded to the rust lexer frontend" in (
        payload["diagnostics"][0]["message"]
    )
    assert validation["diagnosticsByCode"] == {
        "project.translate.defines-not-forwarded": 1
    }
    assert validation["missingCapabilityCounts"] == {"macro.defines": 1}
    assert payload["artifacts"][0]["sourceBackend"] == "rust"
    assert payload["artifacts"][0]["defines"] == {"ENABLE_PATH": "1"}
    assert payload["artifacts"][0]["defineProcessing"] == {
        "status": "not-supported",
        "frontend": "lexer",
        "supportsDefines": False,
        "defineCount": 1,
    }
    assert payload["summary"]["defineProcessingByStatus"] == {"not-supported": 1}
    assert payload["summary"]["defineProcessingBySourceBackend"] == {
        "rust": {"not-supported": 1}
    }
    assert payload["summary"]["defineProcessingByVariant"] == {}
    assert inspection["defineProcessing"] == {
        "available": True,
        "byStatus": {"not-supported": 1},
        "bySourceBackend": {"rust": {"not-supported": 1}},
        "byVariant": {},
        "artifactCount": 1,
        "truncatedArtifactCount": 0,
        "artifacts": [
            {
                "source": "shader.rs",
                "sourceBackend": "rust",
                "target": "cgl",
                "path": "translated/cgl/shader.cgl",
                "status": "not-supported",
                "frontend": "lexer",
                "supportsDefines": False,
                "defineCount": 1,
                "defineNames": ["ENABLE_PATH"],
            }
        ],
        "notSupportedArtifactCount": 1,
        "truncatedNotSupportedArtifactCount": 0,
        "notSupportedArtifacts": [
            {
                "source": "shader.rs",
                "sourceBackend": "rust",
                "target": "cgl",
                "path": "translated/cgl/shader.cgl",
                "status": "not-supported",
                "frontend": "lexer",
                "supportsDefines": False,
                "defineCount": 1,
                "defineNames": ["ENABLE_PATH"],
            }
        ],
    }
    assert result.returncode == 0
    assert "Define processing: not-supported=1" in result.stdout
    assert (
        "Define processing by source backend: rust=(not-supported=1)" in result.stdout
    )
    assert "Define processing issues:" in result.stdout
    assert (
        "- shader.rs -> cgl at translated/cgl/shader.cgl: "
        "1 define not forwarded by rust lexer frontend"
    ) in result.stdout


def test_translate_project_records_artifact_matrix_metadata(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "first.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "second.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl", "opengl"]
            output_dir = "translated"

            [project.variants.debug]
            MODE = "debug"

            [project.variants.release]
            MODE = "release"
            """).strip(),
        encoding="utf-8",
    )

    def write_artifact(
        file_path,
        backend="cgl",
        save_shader=None,
        format_output=True,
        source_backend=None,
        *,
        include_paths=None,
        defines=None,
    ):
        del format_output, source_backend, include_paths
        text = json.dumps(
            {
                "source": Path(file_path).name,
                "target": backend,
                "defines": dict(defines or {}),
            },
            sort_keys=True,
        )
        Path(save_shader).write_text(text, encoding="utf-8")
        return text

    monkeypatch.setattr(project_pipeline, "translate", write_artifact)

    payload = translate_project(load_project_config(repo)).to_json()
    complete_cgl_row = {
        "expectedArtifactCount": 4,
        "emittedArtifactCount": 4,
        "translatedCount": 4,
        "failedCount": 0,
        "missingArtifactCount": 0,
        "extraArtifactCount": 0,
        "complete": True,
    }
    complete_opengl_row = dict(complete_cgl_row)
    complete_debug_row = dict(complete_cgl_row)
    complete_release_row = dict(complete_cgl_row)

    assert payload["artifactMatrix"] == {
        "unitCount": 2,
        "targetCount": 2,
        "variantCount": 2,
        "variantMode": "named",
        "expectedArtifactCount": 8,
        "emittedArtifactCount": 8,
        "translatedCount": 8,
        "failedCount": 0,
        "identityCoverageAvailable": True,
        "missingArtifactCount": 0,
        "extraArtifactCount": 0,
        "complete": True,
        "statusByTarget": {
            "cgl": complete_cgl_row,
            "opengl": complete_opengl_row,
        },
        "statusBySourceBackend": {
            "cgl": {
                "expectedArtifactCount": 8,
                "emittedArtifactCount": 8,
                "translatedCount": 8,
                "failedCount": 0,
                "missingArtifactCount": 0,
                "extraArtifactCount": 0,
                "complete": True,
            }
        },
        "statusByVariant": {
            "debug": complete_debug_row,
            "release": complete_release_row,
        },
    }
    assert payload["summary"]["artifactCount"] == 8
    assert {
        (artifact["source"], artifact["target"], artifact.get("variant"))
        for artifact in payload["artifacts"]
    } == {
        ("first.cgl", "cgl", "debug"),
        ("first.cgl", "cgl", "release"),
        ("first.cgl", "opengl", "debug"),
        ("first.cgl", "opengl", "release"),
        ("second.cgl", "cgl", "debug"),
        ("second.cgl", "cgl", "release"),
        ("second.cgl", "opengl", "debug"),
        ("second.cgl", "opengl", "release"),
    }


def test_translate_project_batches_real_units_targets_and_variants(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    targets = (
        "cuda",
        "directx",
        "hip",
        "metal",
        "mojo",
        "opengl",
        "rust",
        "slang",
        "vulkan",
    )
    variants = ("debug", "release")
    units = ("shaders/first.cgl", "shaders/second.cgl")
    shader_dir.mkdir(parents=True)
    (shader_dir / "first.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (shader_dir / "second.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent(f"""
            [project]
            source_roots = ["shaders"]
            targets = [{", ".join(json.dumps(target) for target in targets)}]
            output_dir = "translated"

            [project.variants.debug]
            MODE = "debug"

            [project.variants.release]
            MODE = "release"
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "translated" / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)

    def complete_row(expected_count):
        return {
            "expectedArtifactCount": expected_count,
            "emittedArtifactCount": expected_count,
            "translatedCount": expected_count,
            "failedCount": 0,
            "missingArtifactCount": 0,
            "extraArtifactCount": 0,
            "complete": True,
        }

    def expected_path(source, target, variant):
        extension = project_pipeline._artifact_target_extension(target)
        return (
            Path("translated")
            .joinpath(target, variant, Path(source).with_suffix(extension))
            .as_posix()
        )

    expected_artifacts = {
        (source, target, variant, expected_path(source, target, variant))
        for source in units
        for target in targets
        for variant in variants
    }

    assert validation["success"] is True
    assert payload["summary"]["artifactCount"] == 36
    assert payload["summary"]["translatedCount"] == 36
    assert payload["summary"]["failedCount"] == 0
    assert payload["artifactMatrix"] == {
        "unitCount": 2,
        "targetCount": 9,
        "variantCount": 2,
        "variantMode": "named",
        "expectedArtifactCount": 36,
        "emittedArtifactCount": 36,
        "translatedCount": 36,
        "failedCount": 0,
        "identityCoverageAvailable": True,
        "missingArtifactCount": 0,
        "extraArtifactCount": 0,
        "complete": True,
        "statusByTarget": {
            target: complete_row(len(units) * len(variants)) for target in targets
        },
        "statusBySourceBackend": {
            "cgl": complete_row(len(units) * len(targets) * len(variants))
        },
        "statusByVariant": {
            variant: complete_row(len(units) * len(targets)) for variant in variants
        },
    }
    assert {
        (
            artifact["source"],
            artifact["target"],
            artifact["variant"],
            artifact["path"],
        )
        for artifact in payload["artifacts"]
    } == expected_artifacts
    assert all(
        (repo / artifact_path).is_file() for *_, artifact_path in expected_artifacts
    )


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


def test_validate_project_report_rejects_missing_target_artifact_matrix_entries(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl", "opengl"], output_dir="out")
    payload = report.to_json()
    payload["artifacts"] = [
        artifact for artifact in payload["artifacts"] if artifact["target"] == "cgl"
    ]
    _refresh_artifact_summary(payload)
    report_path = repo / "out" / "missing-target-artifact-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts must include units[0].path simple.cgl target opengl" in (
        diagnostic["message"]
    )
    assert "summary.artifactCount must match" not in diagnostic["message"]


def test_validate_project_report_rejects_empty_translated_artifact_matrix(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["artifacts"] = []
    _refresh_artifact_summary(payload)
    report_path = repo / "out" / "empty-artifact-matrix-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts must include units[0].path simple.cgl target cgl" in (
        diagnostic["message"]
    )
    assert "summary.artifactCount must match" not in diagnostic["message"]


def test_validate_project_report_rejects_artifact_matrix_count_mismatches(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    payload = translate_project(repo, targets=["cgl"], output_dir="out").to_json()
    payload["artifactMatrix"]["expectedArtifactCount"] = 2
    report_path = repo / "out" / "invalid-artifact-matrix-count-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifactMatrix.expectedArtifactCount must match expected artifact matrix"
        in (diagnostic["message"])
    )


def test_validate_project_report_rejects_artifact_matrix_variant_mode_mismatches(
    tmp_path,
):
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

    payload = translate_project(load_project_config(repo)).to_json()
    payload["artifactMatrix"]["variantMode"] = "none"
    report_path = repo / "out" / "invalid-artifact-matrix-variant-mode-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifactMatrix.variantMode must match project.variants"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_unexpected_generated_artifact_matrix_fields(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    payload = translate_project(repo, targets=["cgl"], output_dir="out").to_json()
    payload["artifactMatrix"]["unexpected"] = "metadata"
    report_path = repo / "out" / "unexpected-artifact-matrix-fields-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifactMatrix.unexpected is not allowed" in diagnostic["message"]


def test_translate_project_artifact_matrix_rolls_up_source_backends(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "kernel.metal").write_text("kernel void k() {}", encoding="utf-8")

    def write_artifact(
        file_path,
        backend="cgl",
        save_shader=None,
        format_output=True,
        source_backend=None,
        *,
        include_paths=None,
        defines=None,
    ):
        del file_path, backend, format_output, include_paths, defines
        Path(save_shader).write_text(
            f"// source backend: {source_backend}\n",
            encoding="utf-8",
        )
        return ""

    monkeypatch.setattr(project_pipeline, "translate", write_artifact)

    payload = translate_project(repo, targets=["cgl"], output_dir="out").to_json()

    complete_row = {
        "expectedArtifactCount": 1,
        "emittedArtifactCount": 1,
        "translatedCount": 1,
        "failedCount": 0,
        "missingArtifactCount": 0,
        "extraArtifactCount": 0,
        "complete": True,
    }
    assert payload["summary"]["unitsBySourceBackend"] == {"cgl": 1, "metal": 1}
    assert payload["artifactMatrix"]["statusBySourceBackend"] == {
        "cgl": complete_row,
        "metal": complete_row,
    }


def test_validate_project_report_rejects_artifact_matrix_rollup_mismatches(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    payload = translate_project(repo, targets=["cgl"], output_dir="out").to_json()
    payload["artifactMatrix"]["emittedArtifactCount"] = 2
    payload["artifactMatrix"]["complete"] = False
    payload["artifactMatrix"]["statusByTarget"] = {}
    payload["artifactMatrix"]["statusBySourceBackend"] = {}
    report_path = repo / "out" / "invalid-artifact-matrix-rollup-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifactMatrix.emittedArtifactCount must match artifact matrix artifacts"
        in diagnostic["message"]
    )
    assert "artifactMatrix.complete must match artifact matrix" in diagnostic["message"]
    assert (
        "artifactMatrix.statusByTarget must match artifact matrix artifacts"
        in diagnostic["message"]
    )
    assert (
        "artifactMatrix.statusBySourceBackend must match artifact matrix artifacts"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_artifact_matrix_variant_rollup_mismatches(
    tmp_path,
):
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

    payload = translate_project(load_project_config(repo)).to_json()
    payload["artifactMatrix"]["statusByVariant"] = {}
    report_path = repo / "out" / "invalid-artifact-matrix-variant-rollup-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifactMatrix.statusByVariant must match artifact matrix artifacts"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_missing_variant_artifact_matrix_entries(
    tmp_path,
):
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

            [project.variants.release]
            MODE = "release"
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    payload["artifacts"] = [
        artifact
        for artifact in payload["artifacts"]
        if artifact.get("variant") == "debug"
    ]
    _refresh_artifact_summary(payload)
    report_path = repo / "out" / "missing-variant-artifact-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts must include units[0].path simple.cgl target cgl variant release"
        in diagnostic["message"]
    )
    assert "summary.artifactCount must match" not in diagnostic["message"]


def test_validate_project_report_allows_scan_reports_without_artifacts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = scan_project(repo).to_report(targets=["cgl", "opengl"])
    report_path = repo / "scan-report.json"
    report.write_json(report_path)

    validation = validate_project_report(report_path)

    assert validation["success"] is True
    assert validation["validation"]["artifacts"] == []
    assert validation["validation"]["summary"] == {
        "artifactCount": 0,
        "okCount": 0,
        "failedCount": 0,
        "sourceHashStatusCounts": _source_hash_status_counts(),
        "generatedHashStatusCounts": _generated_hash_status_counts(),
        "sourceMapStatusCounts": _source_map_status_counts(),
        "sourceRemapStatusCounts": _source_remap_status_counts(),
    }


def test_inspect_project_report_does_not_derive_scan_only_artifact_matrix(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = scan_project(repo).to_report(targets=["cgl", "opengl"])
    report_path = repo / "scan-report.json"
    report.write_json(report_path)

    payload = inspect_project_report(report_path)

    assert payload["success"] is True
    assert payload["artifactMatrix"] == {"available": False}


def test_validate_project_report_rejects_missing_artifact_defines(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["artifacts"][0].pop("defines")
    report_path = repo / "out" / "portability-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].defines must be an object" in diagnostic["message"]


def test_validate_project_report_rejects_artifact_define_mismatches(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            output_dir = "out"

            [project.defines]
            MODE = "base"

            [project.variants.debug]
            MODE = "debug"
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    payload["artifacts"][0]["defines"] = {"MODE": "base"}
    report_path = repo / "out" / "portability-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].defines must match project defines and artifact variant"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_artifact_define_processing_mismatches(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            output_dir = "out"

            [project.defines]
            MODE = "base"
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    payload["artifacts"][0]["defineProcessing"]["supportsDefines"] = False
    report_path = repo / "out" / "portability-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].defineProcessing.status must match define count "
        "and source frontend support"
    ) in diagnostic["message"]
    assert (
        "artifacts[0].defineProcessing.supportsDefines must match "
        "artifacts[0].sourceBackend"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_unexpected_generated_processing_fields(
    tmp_path,
):
    repo = tmp_path / "repo"
    include_dir = repo / "includes"
    repo.mkdir()
    include_dir.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            output_dir = "out"
            include_dirs = ["includes"]

            [project.defines]
            MODE = "base"
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    payload["artifacts"][0]["defineProcessing"]["unexpected"] = "metadata"
    payload["artifacts"][0]["includePathProcessing"]["unexpected"] = "metadata"
    report_path = repo / "out" / "portability-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].defineProcessing.unexpected is not allowed"
        in diagnostic["message"]
    )
    assert (
        "artifacts[0].includePathProcessing.unexpected is not allowed"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_define_processing_summary_mismatches(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["summary"]["defineProcessingByStatus"] = {"forwarded": 1}
    payload["summary"]["defineProcessingBySourceBackend"] = {"cgl": {"forwarded": 1}}
    payload["summary"]["defineProcessingByVariant"] = {"debug": {"forwarded": 1}}
    report_path = repo / "out" / "portability-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "summary.defineProcessingByStatus must match artifact define processing"
        in diagnostic["message"]
    )
    assert (
        "summary.defineProcessingBySourceBackend must match "
        "artifact define processing"
    ) in diagnostic["message"]
    assert (
        "summary.defineProcessingByVariant must match artifact define processing"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_artifact_include_path_processing_mismatches(
    tmp_path,
):
    repo = tmp_path / "repo"
    include_dir = repo / "includes"
    repo.mkdir()
    include_dir.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            output_dir = "out"
            include_dirs = ["includes"]
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    payload["artifacts"][0]["includePathProcessing"]["supportsIncludePaths"] = False
    report_path = repo / "out" / "portability-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].includePathProcessing.status must match include path count "
        "and source frontend support"
    ) in diagnostic["message"]
    assert (
        "artifacts[0].includePathProcessing.supportsIncludePaths must match "
        "artifacts[0].sourceBackend"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_include_path_processing_summary_mismatches(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["summary"]["includePathProcessingByStatus"] = {"forwarded": 1}
    payload["summary"]["includePathProcessingBySourceBackend"] = {
        "cgl": {"forwarded": 1}
    }
    payload["summary"]["includePathProcessingByVariant"] = {"debug": {"forwarded": 1}}
    report_path = repo / "out" / "portability-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "summary.includePathProcessingByStatus must match "
        "artifact include path processing"
    ) in diagnostic["message"]
    assert (
        "summary.includePathProcessingBySourceBackend must match "
        "artifact include path processing"
    ) in diagnostic["message"]
    assert (
        "summary.includePathProcessingByVariant must match "
        "artifact include path processing"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_artifacts_with_undeclared_sources(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["artifacts"][0]["source"] = "other.cgl"
    report_path = repo / "out" / "portability-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path, run_toolchains=True)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].source must be listed in units" in diagnostic["message"]


def test_validate_project_report_rejects_artifacts_with_mismatched_source_backend(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["artifacts"][0]["sourceBackend"] = "directx"
    report_path = repo / "out" / "portability-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path, run_toolchains=True)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].sourceBackend must match units[0].sourceBackend"
        in diagnostic["message"]
    )


@pytest.mark.parametrize(
    ("source_backend", "message"),
    [
        (
            "crossgl",
            "units[0].sourceBackend must use canonical source backend name cgl",
        ),
        (
            "not-a-source",
            "units[0].sourceBackend must be a registered source backend",
        ),
    ],
)
def test_validate_project_report_rejects_invalid_unit_source_backends(
    tmp_path,
    source_backend,
    message,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["units"][0]["sourceBackend"] = source_backend
    payload["artifacts"][0]["sourceBackend"] = source_backend
    payload["summary"]["unitsBySourceBackend"] = {source_backend: 1}
    payload["summary"]["artifactsBySourceBackend"] = {
        source_backend: {
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
        }
    }
    payload["summary"]["sourceMapsBySourceBackend"] = {source_backend: 1}
    report_path = repo / "out" / "portability-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path, run_toolchains=True)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert message in diagnostic["message"]


@pytest.mark.parametrize(
    ("source_overrides", "source_override", "message"),
    [
        (
            {"gpu/*.shader": "cgl"},
            "directx",
            (
                "units[0].sourceOverride must match project.sourceOverrides "
                "for units[0].path"
            ),
        ),
        (
            {"gpu/*.shader": "directx"},
            "directx",
            "units[0].sourceOverride must resolve to units[0].sourceBackend",
        ),
    ],
)
def test_validate_project_report_rejects_inconsistent_unit_source_overrides(
    tmp_path,
    source_overrides,
    source_override,
    message,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "gpu"
    shader_dir.mkdir(parents=True)
    (shader_dir / "kernel.shader").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["gpu"]
            targets = ["cgl"]
            output_dir = "out"

            [project.sources]
            "gpu/*.shader" = "cgl"
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    payload["project"]["sourceOverrides"] = source_overrides
    payload["project"]["sourceOverrideCount"] = len(source_overrides)
    payload["units"][0]["sourceOverride"] = source_override
    report_path = repo / "out" / "portability-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path, run_toolchains=True)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert message in diagnostic["message"]


def test_validate_project_report_rejects_noncanonical_project_targets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["opengl"], output_dir="out")
    payload = report.to_json()
    payload["project"]["targets"] = ["OpenGL", "opengl"]
    report_path = repo / "out" / "portability-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "project.targets must use normalized backend names without duplicates"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_noncanonical_full_report_targets(tmp_path):
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
    artifact = payload["artifacts"][0]
    artifact["target"] = "OpenGL"
    artifact["sourceMap"]["target"] = "OpenGL"
    validation_artifact = payload["validation"]["artifacts"][0]
    validation_artifact["target"] = "OpenGL"
    payload["validation"]["toolchains"][0]["target"] = "OpenGL"
    payload["validation"]["toolchainRuns"] = [
        {
            "source": validation_artifact["source"],
            "sourceBackend": validation_artifact["sourceBackend"],
            "target": "OpenGL",
            "path": validation_artifact["path"],
            "command": ["glslangValidator", "--stdin"],
            "returncode": 0,
            "status": "ok",
            "stdout": "",
            "stderr": "",
        }
    ]
    report_path = repo / "out" / "noncanonical-record-targets-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].target must use normalized backend name opengl"
        in diagnostic["message"]
    )
    assert (
        "artifacts[0].sourceMap.target must use normalized backend name opengl"
        in diagnostic["message"]
    )
    assert (
        "validation.toolchains[0].target must use normalized backend name opengl"
        in diagnostic["message"]
    )
    assert (
        "validation.artifacts[0].target must use normalized backend name opengl"
        in diagnostic["message"]
    )
    assert (
        "validation.toolchainRuns[0].target must use normalized backend name opengl"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_noncanonical_source_remap_targets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    artifact = payload["artifacts"][0]
    artifact["target"] = "CGL"
    artifact["sourceMap"]["target"] = "CGL"
    artifact["sourceRemap"]["target"] = "CGL"
    report_path = repo / "out" / "noncanonical-source-remap-targets-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].target must use normalized backend name cgl"
        in diagnostic["message"]
    )
    assert (
        "artifacts[0].sourceMap.target must use normalized backend name cgl"
        in diagnostic["message"]
    )
    assert (
        "artifacts[0].sourceRemap.target must use normalized backend name cgl"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_noncanonical_diagnostic_targets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    payload = scan_project(repo).to_report(targets=["not-a-backend"]).to_json()
    payload["diagnostics"][0]["target"] = "Not-A-Backend"
    report_path = repo / "noncanonical-diagnostic-target-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "diagnostics[0].target must use normalized backend name not-a-backend"
        in diagnostic["message"]
    )


def test_translate_project_preserves_relative_paths_and_reports_artifacts(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders" / "graphics"
    shader_dir.mkdir(parents=True)
    (shader_dir / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["opengl"], output_dir="translated")
    payload = report.to_json()
    repeated_payload = report.to_json()

    output = repo / "translated" / "opengl" / "shaders" / "graphics" / "simple.glsl"
    assert output.exists()
    assert payload["kind"] == "crosstl-project-portability-report"
    assert payload["schemaVersion"] == 1
    assert isinstance(payload["generatedAt"], int)
    assert payload["generatedAt"] >= 0
    assert repeated_payload["generatedAt"] == payload["generatedAt"]
    assert payload["generator"]["name"] == "CrossTL"
    assert payload["generator"]["pipeline"] == "project-porting"
    assert isinstance(payload["generator"]["packageVersion"], str)
    assert payload["generator"]["packageVersion"]
    assert payload["project"]["outputDir"] == str((repo / "translated").resolve())
    assert payload["summary"]["unitCount"] == 1
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    assert payload["summary"]["diagnosticsByCode"] == {}
    assert payload["summary"]["missingCapabilityCounts"] == {}
    assert payload["summary"]["unitsBySourceBackend"] == {"cgl": 1}
    assert payload["summary"]["unitsByExtension"] == {".cgl": 1}
    assert payload["summary"]["skippedByReason"] == {}
    assert payload["summary"]["skippedByExtension"] == {}
    assert payload["summary"]["artifactsBySourceBackend"] == {
        "cgl": {
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
        }
    }
    assert payload["summary"]["artifactsByVariant"] == {}
    assert payload["summary"]["artifactsByTarget"] == {
        "opengl": {
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
        }
    }
    assert payload["summary"]["artifactProvenanceByPipeline"] == {
        "single-file-translate": 1
    }
    assert payload["summary"]["artifactProvenanceByIntermediate"] == {"none": 1}
    assert payload["summary"]["artifactProvenanceIntermediateBySourceBackend"] == {
        "cgl": {"none": 1}
    }
    assert payload["project"]["sourceRootCount"] == 1
    assert payload["project"]["includePatternCount"] == 0
    assert payload["project"]["excludePatternCount"] == len(
        project_pipeline.DEFAULT_EXCLUDE_PATTERNS
    )
    assert payload["project"]["includeDirCount"] == 0
    assert payload["units"][0]["sourceHash"] == project_pipeline._source_hash(
        shader_dir / "simple.cgl"
    )
    assert payload["artifacts"][0]["source"] == "shaders/graphics/simple.cgl"
    assert payload["artifacts"][0]["target"] == "opengl"
    assert payload["artifacts"][0]["path"] == (
        "translated/opengl/shaders/graphics/simple.glsl"
    )
    assert payload["artifacts"][0]["provenance"] == {
        "pipeline": "single-file-translate",
        "intermediate": None,
    }
    assert payload["artifacts"][0]["sourceHash"]["algorithm"] == "sha256"
    assert payload["artifacts"][0]["generatedHash"] == project_pipeline._source_hash(
        output
    )
    assert "sourceRemap" not in payload["artifacts"][0]
    assert payload["summary"]["sourceRemapCount"] == 0
    assert payload["summary"]["sourceRemapsByTarget"] == {}
    assert payload["summary"]["sourceRemapsBySourceBackend"] == {}
    assert payload["summary"]["sourceRemapsByVariant"] == {}
    assert payload["migration"]["nonGoals"] == [
        "automatic runtime API migration",
        "application build-system rewrites",
        "backend framework integration",
    ]


def test_translate_project_emits_closed_portability_report_schema(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    payload = translate_project(repo, targets=["opengl"], output_dir="out").to_json()

    assert set(payload) == project_pipeline.REPORT_FIELDS - {"externalCorpus"}
    assert set(payload["generator"]) == project_pipeline.REPORT_GENERATOR_FIELDS
    assert set(payload["project"]) == project_pipeline.REPORT_PROJECT_FIELDS
    assert set(payload["summary"]) == project_pipeline.REPORT_SUMMARY_FIELDS
    assert (
        set(payload["artifactMatrix"]) == project_pipeline.REPORT_ARTIFACT_MATRIX_FIELDS
    )
    assert set(payload["migration"]) == project_pipeline.REPORT_MIGRATION_FIELDS
    assert set(payload["migration"]["actions"][0]) == (
        project_pipeline.REPORT_MIGRATION_ACTION_FIELDS
    )

    unit = payload["units"][0]
    assert set(unit) == project_pipeline.REPORT_UNIT_FIELDS - {
        "sourceOverride",
        "includeDependencies",
    }
    assert set(unit["sourceHash"]) == project_pipeline.REPORT_HASH_FIELDS

    artifact = payload["artifacts"][0]
    assert set(artifact) == project_pipeline.REPORT_ARTIFACT_FIELDS - {
        "variant",
        "error",
        "sourceRemap",
    }
    assert set(artifact["sourceHash"]) == project_pipeline.REPORT_HASH_FIELDS
    assert set(artifact["generatedHash"]) == project_pipeline.REPORT_HASH_FIELDS
    assert set(artifact["defineProcessing"]) == (
        project_pipeline.REPORT_ARTIFACT_DEFINE_PROCESSING_FIELDS
    )
    assert set(artifact["includePathProcessing"]) == (
        project_pipeline.REPORT_ARTIFACT_INCLUDE_PATH_PROCESSING_FIELDS
    )
    assert set(artifact["provenance"]) == (
        project_pipeline.REPORT_ARTIFACT_PROVENANCE_FIELDS
    )

    source_map = artifact["sourceMap"]
    assert set(source_map) == project_pipeline.SOURCE_MAP_PAYLOAD_FIELDS
    assert set(source_map["source"]) == project_pipeline.SOURCE_MAP_SPAN_FIELD_SET
    assert set(source_map["generated"]) == project_pipeline.SOURCE_MAP_SPAN_FIELD_SET
    assert set(source_map["mappings"][0]) == project_pipeline.SOURCE_MAP_MAPPING_FIELDS


def test_translate_project_records_bridge_artifact_provenance_rollups(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "shader.rs").write_text("fn main() {}\n", encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["opengl"]
            output_dir = "translated"

            [project.sources]
            "*.rs" = "rust"
            """).strip(),
        encoding="utf-8",
    )

    def write_artifact(
        file_path,
        backend="cgl",
        save_shader=None,
        format_output=True,
        source_backend=None,
        *,
        include_paths=None,
        defines=None,
    ):
        del file_path, backend, format_output, source_backend, include_paths, defines
        Path(save_shader).write_text("// translated\n", encoding="utf-8")
        return "// translated\n"

    monkeypatch.setattr(project_pipeline, "translate", write_artifact)

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "translated" / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)

    assert validation["success"] is True
    assert payload["artifacts"][0]["provenance"] == {
        "pipeline": "single-file-translate",
        "intermediate": "crossgl",
    }
    assert payload["summary"]["artifactProvenanceByPipeline"] == {
        "single-file-translate": 1
    }
    assert payload["summary"]["artifactProvenanceByIntermediate"] == {"crossgl": 1}
    assert payload["summary"]["artifactProvenanceIntermediateBySourceBackend"] == {
        "rust": {"crossgl": 1}
    }


def test_inspect_project_report_groups_direct_and_bridged_artifact_provenance(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "direct.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "shader.rs").write_text("fn main() {}\n", encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["opengl"]
            output_dir = "out"

            [project.sources]
            "*.rs" = "rust"
            """).strip(),
        encoding="utf-8",
    )

    def write_artifact(
        file_path,
        backend="cgl",
        save_shader=None,
        format_output=True,
        source_backend=None,
        *,
        include_paths=None,
        defines=None,
    ):
        del file_path, backend, format_output, source_backend, include_paths, defines
        Path(save_shader).write_text("// translated\n", encoding="utf-8")
        return "// translated\n"

    monkeypatch.setattr(project_pipeline, "translate", write_artifact)

    report = translate_project(load_project_config(repo))
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)

    payload = inspect_project_report(report_path)
    provenance = payload["artifactProvenance"]
    direct_artifact = provenance["directArtifacts"][0]
    bridged_artifact = provenance["bridgedArtifacts"][0]

    assert payload["success"] is True
    assert provenance["artifactCount"] == 2
    assert provenance["directArtifactCount"] == 1
    assert provenance["truncatedDirectArtifactCount"] == 0
    assert provenance["bridgedArtifactCount"] == 1
    assert provenance["truncatedBridgedArtifactCount"] == 0
    assert direct_artifact["source"] == "direct.cgl"
    assert direct_artifact["intermediate"] == "none"
    assert bridged_artifact["source"] == "shader.rs"
    assert bridged_artifact["intermediate"] == "crossgl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Artifact provenance samples:" in result.stdout
    assert "Direct artifact provenance samples:" in result.stdout
    assert (
        "- direct.cgl -> out/opengl/direct.glsl "
        "(sourceBackend=cgl, target=opengl, "
        "pipeline=single-file-translate, intermediate=none)"
    ) in result.stdout
    assert "Bridged artifact provenance samples:" in result.stdout
    assert (
        "- shader.rs -> out/opengl/shader.glsl "
        "(sourceBackend=rust, target=opengl, "
        "pipeline=single-file-translate, intermediate=crossgl)"
    ) in result.stdout


def test_translate_project_records_line_preserving_source_maps(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()

    artifact = payload["artifacts"][0]
    source_map = artifact["sourceMap"]

    assert payload["summary"]["sourceMapCount"] == 1
    assert payload["summary"]["fineGrainedSourceMapCount"] == 1
    assert payload["summary"]["sourceMapsByGranularity"] == {"line": 1}
    assert payload["summary"]["sourceMapsByTarget"] == {"cgl": 1}
    assert payload["summary"]["sourceMapsBySourceBackend"] == {"cgl": 1}
    assert payload["summary"]["sourceMapsByVariant"] == {}
    assert payload["summary"]["sourceRemapCount"] == 1
    assert payload["summary"]["sourceRemapsByTarget"] == {"cgl": 1}
    assert payload["summary"]["sourceRemapsBySourceBackend"] == {"cgl": 1}
    assert payload["summary"]["sourceRemapsByVariant"] == {}
    assert source_map["schemaVersion"] == 1
    assert source_map["kind"] == "crosstl-artifact-source-map"
    assert source_map["mappingGranularity"] == "line"
    assert source_map["target"] == "cgl"
    assert source_map["source"]["file"] == "simple.cgl"
    assert source_map["generated"]["file"] == "out/cgl/simple.cgl"
    assert source_map["source"]["length"] == len((repo / "simple.cgl").read_bytes())
    assert source_map["generated"]["length"] == len(
        (repo / artifact["path"]).read_bytes()
    )
    assert source_map["mappings"] == [
        {
            "source": source_span.to_json(),
            "generated": generated_span.to_json(),
        }
        for source_span, generated_span in zip(
            project_pipeline._line_spans(repo / "simple.cgl", "simple.cgl"),
            project_pipeline._line_spans(repo / artifact["path"], "out/cgl/simple.cgl"),
        )
    ]
    assert all(mapping["source"]["length"] > 0 for mapping in source_map["mappings"])
    assert all(mapping["generated"]["length"] > 0 for mapping in source_map["mappings"])
    source_remap = artifact["sourceRemap"]
    source_remap_path = repo / source_remap["path"]
    source_remap_payload = json.loads(source_remap_path.read_text(encoding="utf-8"))
    assert source_remap == {
        "schemaVersion": 1,
        "path": "out/cgl/simple.source-remap.json",
        "target": "cgl",
        "generatedFile": "out/cgl/simple.cgl",
        "mappingGranularity": "file",
        "hash": project_pipeline._source_hash(source_remap_path),
    }
    assert source_remap_payload == {
        "schemaVersion": 1,
        "generatedFile": source_map["generated"]["file"],
        "mappings": [
            {
                "generated": source_map["generated"],
                "original": source_map["source"],
            }
        ],
    }


def test_translate_project_uses_file_source_maps_for_generated_artifacts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["opengl"], output_dir="out")
    payload = report.to_json()

    artifact = payload["artifacts"][0]
    source_map = artifact["sourceMap"]

    assert payload["summary"]["sourceMapCount"] == 1
    assert payload["summary"]["fineGrainedSourceMapCount"] == 0
    assert payload["summary"]["sourceMapsByGranularity"] == {"file": 1}
    assert source_map["mappingGranularity"] == "file"
    assert source_map["mappings"] == [
        {
            "source": source_map["source"],
            "generated": source_map["generated"],
        }
    ]
    assert "sourceRemap" not in artifact


def test_source_map_rollups_count_fine_grained_artifact_maps():
    rollups = project_pipeline._source_map_rollups(
        [
            {
                "sourceMap": {"mappingGranularity": "file"},
                "target": "cgl",
                "sourceBackend": "cgl",
            },
            {
                "sourceMap": {"mappingGranularity": "line"},
                "target": "metal",
                "sourceBackend": "metal",
                "variant": "debug",
            },
            {
                "sourceMap": {"mappingGranularity": "statement"},
                "target": "metal",
                "sourceBackend": "metal",
            },
        ]
    )

    assert rollups == {
        "sourceMapCount": 3,
        "fineGrainedSourceMapCount": 2,
        "sourceMapsByGranularity": {
            "file": 1,
            "line": 1,
            "statement": 1,
        },
        "sourceMapsByTarget": {"cgl": 1, "metal": 2},
        "sourceMapsBySourceBackend": {"cgl": 1, "metal": 2},
        "sourceMapsByVariant": {"debug": 1},
        "sourceRemapCount": 0,
        "sourceRemapsByTarget": {},
        "sourceRemapsBySourceBackend": {},
        "sourceRemapsByVariant": {},
    }


def test_file_level_source_map_spans_use_utf8_byte_offsets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source_text = "// unicode marker: \u2603\n" + SIMPLE_CROSSL
    source_path = repo / "simple.cgl"
    source_path.write_text(source_text, encoding="utf-8")

    span = project_pipeline._file_span(source_path, "simple.cgl").to_json()

    assert span["file"] == "simple.cgl"
    source_bytes = source_path.read_bytes()
    assert span["length"] == len(source_bytes)
    assert span["endOffset"] == len(source_bytes)
    assert span["endOffset"] > len(source_text)


def test_line_source_map_spans_use_utf8_byte_offsets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    first_line = "// unicode marker: \u2603\n"
    source_path = repo / "simple.cgl"
    source_path.write_bytes((first_line + SIMPLE_CROSSL).encode("utf-8"))

    spans = [
        span.to_json()
        for span in project_pipeline._line_spans(source_path, "simple.cgl")
    ]

    assert spans[0]["file"] == "simple.cgl"
    assert spans[0]["line"] == 1
    assert spans[0]["column"] == 1
    assert spans[0]["offset"] == 0
    assert spans[0]["length"] == len(first_line.encode("utf-8"))
    assert spans[0]["endOffset"] == len(first_line.encode("utf-8"))
    assert spans[0]["endOffset"] > len(first_line)
    assert spans[1]["offset"] == spans[0]["endOffset"]


def test_translate_project_reports_source_maps_and_remaps_by_variant(
    tmp_path, monkeypatch
):
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

            [project.variants.release]
            MODE = "release"
            """).strip(),
        encoding="utf-8",
    )

    def write_shader(
        file_path,
        backend="cgl",
        save_shader=None,
        format_output=True,
        source_backend=None,
        *,
        include_paths=None,
        defines=None,
    ):
        del file_path, backend, format_output, source_backend, include_paths, defines
        Path(save_shader).write_text("// translated\n", encoding="utf-8")
        return "// translated\n"

    monkeypatch.setattr(project_pipeline, "translate", write_shader)

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    inspection = inspect_project_report(report_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert validation["success"] is True
    assert payload["summary"]["sourceMapCount"] == 2
    assert payload["summary"]["sourceMapsByVariant"] == {
        "debug": 1,
        "release": 1,
    }
    assert payload["summary"]["sourceRemapCount"] == 2
    assert payload["summary"]["sourceRemapsByVariant"] == {
        "debug": 1,
        "release": 1,
    }
    assert inspection["sourceMaps"]["sourceMapsByVariant"] == {
        "debug": 1,
        "release": 1,
    }
    assert inspection["sourceMaps"]["sourceRemapsByVariant"] == {
        "debug": 1,
        "release": 1,
    }
    assert inspection["sourceMaps"]["sourceMapArtifactCount"] == 2
    assert {
        artifact["variant"]
        for artifact in inspection["sourceMaps"]["sourceMapArtifacts"]
    } == {"debug", "release"}
    assert all(
        artifact["sourceHashAlgorithm"] == "sha256"
        and isinstance(artifact["sourceHash"], str)
        and artifact["sourceHash"]
        for artifact in inspection["sourceMaps"]["sourceMapArtifacts"]
    )
    assert inspection["sourceMaps"]["sourceRemapArtifactCount"] == 2
    assert {
        artifact["variant"]
        for artifact in inspection["sourceMaps"]["sourceRemapArtifacts"]
    } == {"debug", "release"}
    assert all(
        artifact["sourceHashAlgorithm"] == "sha256"
        and isinstance(artifact["sourceHash"], str)
        and artifact["sourceHash"]
        for artifact in inspection["sourceMaps"]["sourceRemapArtifacts"]
    )
    assert result.returncode == 0
    assert "Source maps by variant: debug=1, release=1" in result.stdout
    assert "Source remaps by variant: debug=1, release=1" in result.stdout


def test_translate_project_sanitizes_variant_source_map_and_remap_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            output_dir = "out"

            [project.variants."qa/profile"]
            MODE = "qa"
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)
    inspection = inspect_project_report(report_path)

    variant_segment = project_pipeline._variant_output_segment("qa/profile")
    expected_artifact_path = f"out/cgl/{variant_segment}/simple.cgl"
    expected_remap_path = f"out/cgl/{variant_segment}/simple.source-remap.json"
    artifact = payload["artifacts"][0]
    source_map = artifact["sourceMap"]
    source_remap = artifact["sourceRemap"]
    source_remap_payload = json.loads(
        (repo / expected_remap_path).read_text(encoding="utf-8")
    )

    assert validation["success"] is True
    assert variant_segment != "qa/profile"
    assert "/" not in variant_segment
    assert artifact["variant"] == "qa/profile"
    assert artifact["path"] == expected_artifact_path
    assert source_map["generated"]["file"] == expected_artifact_path
    assert source_map["mappings"][0]["generated"]["file"] == expected_artifact_path
    assert source_remap["path"] == expected_remap_path
    assert source_remap["generatedFile"] == expected_artifact_path
    assert source_remap_payload["generatedFile"] == expected_artifact_path
    assert (
        source_remap_payload["mappings"][0]["generated"]["file"]
        == expected_artifact_path
    )
    assert payload["summary"]["sourceMapsByVariant"] == {"qa/profile": 1}
    assert payload["summary"]["sourceRemapsByVariant"] == {"qa/profile": 1}
    assert validation["artifactStatusByVariant"] == {
        "qa/profile": {"artifactCount": 1, "okCount": 1, "failedCount": 0}
    }
    assert inspection["sourceMaps"]["sourceMapsByVariant"] == {"qa/profile": 1}
    assert inspection["sourceMaps"]["sourceRemapsByVariant"] == {"qa/profile": 1}
    assert (repo / expected_artifact_path).exists()
    assert (repo / expected_remap_path).exists()
    assert not (repo / "out" / "cgl" / "qa" / "profile" / "simple.cgl").exists()
    assert not (
        repo / "out" / "cgl" / "qa" / "profile" / "simple.source-remap.json"
    ).exists()


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
        "manifestEntryCount": 2,
        "validEntryCount": 2,
        "invalidEntryCount": 0,
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


def test_translate_project_records_missing_external_corpus_manifest(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    config_path = repo / "crosstl.toml"
    config_path.write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            external_corpus_manifest = "missing-corpus.json"
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo))
    payload = report.to_json()
    report_path = repo / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)

    assert validation["success"] is True
    assert payload["project"]["config"] == str(config_path.resolve())
    assert payload["project"]["externalCorpusManifest"] == "missing-corpus.json"
    assert payload["externalCorpus"] == {
        "schemaVersion": 1,
        "manifest": "missing-corpus.json",
        "status": "missing",
        "entries": [],
        "summary": project_pipeline._external_corpus_empty_summary(),
    }
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 1, "error": 0}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.config.external-corpus-missing"
    assert diagnostic["severity"] == "warning"
    assert "missing-corpus.json" in diagnostic["message"]


def test_translate_project_skips_invalid_external_corpus_entries(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "corpus.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "entries": [
                    {
                        "id": "repo/outside",
                        "path": "../outside.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                    },
                    {
                        "id": "repo/simple",
                        "path": "simple.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
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
    external_corpus = payload["externalCorpus"]
    assert external_corpus["status"] == "ok"
    assert external_corpus["summary"]["manifestEntryCount"] == 2
    assert external_corpus["summary"]["validEntryCount"] == 1
    assert external_corpus["summary"]["invalidEntryCount"] == 1
    assert external_corpus["summary"]["entryCount"] == 1
    assert [entry["path"] for entry in external_corpus["entries"]] == ["simple.cgl"]
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 1, "error": 0}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.config.external-corpus-entry-invalid"
    assert diagnostic["severity"] == "warning"
    assert "entry 1" in diagnostic["message"]
    assert "path must be repository-relative" in diagnostic["message"]


def test_translate_project_skips_invalid_external_corpus_provenance(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "corpus.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "entries": [
                    {
                        "id": "repo/simple",
                        "path": "simple.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                        "repository": "https://github.com/example/project",
                        "commit": "1" * 40,
                        "sourceUrl": (
                            "https://github.com/example/project/blob/"
                            f"{'1' * 40}/simple.cgl"
                        ),
                    },
                    {
                        "id": "repo/bad-commit",
                        "path": "bad-commit.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                        "repository": "https://github.com/example/project",
                        "commit": "ABC123",
                        "sourceUrl": (
                            "https://github.com/example/project/blob/ABC123/"
                            "bad-commit.cgl"
                        ),
                    },
                    {
                        "id": "repo/bad-source-url",
                        "path": "bad-source-url.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                        "repository": "https://github.com/example/project",
                        "commit": "2" * 40,
                        "sourceUrl": (
                            "https://github.com/other/project/blob/"
                            f"{'2' * 40}/bad-source-url.cgl"
                        ),
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
    external_corpus = payload["externalCorpus"]
    assert external_corpus["summary"]["manifestEntryCount"] == 3
    assert external_corpus["summary"]["validEntryCount"] == 1
    assert external_corpus["summary"]["invalidEntryCount"] == 2
    assert [entry["id"] for entry in external_corpus["entries"]] == ["repo/simple"]
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 2, "error": 0}
    diagnostic_messages = [
        diagnostic["message"] for diagnostic in payload["diagnostics"]
    ]
    assert any(
        "entry 2" in message
        and "commit must be a lowercase 40-character hex digest" in message
        for message in diagnostic_messages
    )
    assert any(
        "entry 3" in message and "sourceUrl must start with repository" in message
        for message in diagnostic_messages
    )


def test_translate_project_skips_duplicate_external_corpus_entries(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "other.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "corpus.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "entries": [
                    {
                        "id": "repo/simple",
                        "path": "simple.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                    },
                    {
                        "id": "repo/simple-path-duplicate",
                        "path": "simple.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                    },
                    {
                        "id": "repo/simple",
                        "path": "other.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
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
    external_corpus = payload["externalCorpus"]
    assert external_corpus["summary"]["manifestEntryCount"] == 3
    assert external_corpus["summary"]["validEntryCount"] == 1
    assert external_corpus["summary"]["invalidEntryCount"] == 2
    assert [entry["path"] for entry in external_corpus["entries"]] == ["simple.cgl"]
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 2, "error": 0}
    diagnostics = payload["diagnostics"]
    assert all(
        diagnostic["code"] == "project.config.external-corpus-entry-invalid"
        for diagnostic in diagnostics
    )
    assert "entry 2" in diagnostics[0]["message"]
    assert "path duplicates entry 1" in diagnostics[0]["message"]
    assert "entry 3" in diagnostics[1]["message"]
    assert "id duplicates entry 1" in diagnostics[1]["message"]


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
            "sourceBackend": "cgl",
            "target": "cgl",
            "path": "out/cgl/simple.cgl",
            "exists": True,
            "status": "ok",
            "sourceHashStatus": "ok",
            "generatedHashStatus": "ok",
            "sourceMapStatus": "ok",
            "sourceRemapStatus": "ok",
        }
    ]
    assert payload["validation"]["summary"] == {
        "artifactCount": 1,
        "okCount": 1,
        "failedCount": 0,
        "sourceHashStatusCounts": _source_hash_status_counts(ok=1),
        "generatedHashStatusCounts": _generated_hash_status_counts(ok=1),
        "sourceMapStatusCounts": _source_map_status_counts(ok=1),
        "sourceRemapStatusCounts": _source_remap_status_counts(ok=1),
    }
    assert payload["artifactStatusByTarget"] == {
        "cgl": {"artifactCount": 1, "okCount": 1, "failedCount": 0}
    }


def test_translate_project_can_embed_toolchain_smoke_runs(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    commands = []

    monkeypatch.setattr(
        project_pipeline.shutil,
        "which",
        lambda tool: (
            "/usr/bin/glslangValidator" if tool == "glslangValidator" else None
        ),
    )

    def run_toolchain(command, **kwargs):
        commands.append((command, kwargs))
        assert command == ["glslangValidator", "--stdin"]
        assert kwargs["cwd"] == str(repo)
        assert kwargs["timeout"] == project_pipeline.TOOLCHAIN_SMOKE_TIMEOUT_SECONDS
        assert kwargs["input"]
        return SimpleNamespace(returncode=0, stdout="validation ok", stderr="")

    monkeypatch.setattr(project_pipeline.subprocess, "run", run_toolchain)

    report = translate_project(
        repo,
        targets=["opengl"],
        output_dir="out",
        run_toolchains=True,
    )
    payload = report.to_json()

    assert len(commands) == 1
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 0}
    assert payload["validation"]["toolchains"] == [
        {
            "target": "opengl",
            "status": "available",
            "tools": [
                {
                    "name": "glslangValidator",
                    "path": "/usr/bin/glslangValidator",
                    "available": True,
                }
            ],
        }
    ]
    assert payload["validation"]["artifacts"][0]["status"] == "ok"
    assert payload["validation"]["summary"] == {
        "artifactCount": 1,
        "okCount": 1,
        "failedCount": 0,
        "sourceHashStatusCounts": _source_hash_status_counts(ok=1),
        "generatedHashStatusCounts": _generated_hash_status_counts(ok=1),
        "sourceMapStatusCounts": _source_map_status_counts(ok=1),
        "sourceRemapStatusCounts": _source_remap_status_counts(**{"not-recorded": 1}),
    }
    assert payload["validation"]["toolchainRuns"] == [
        {
            "source": "simple.cgl",
            "sourceBackend": "cgl",
            "target": "opengl",
            "path": "out/opengl/simple.glsl",
            "command": ["glslangValidator", "--stdin"],
            "checkKind": "artifact",
            "returncode": 0,
            "status": "ok",
            "stdout": "validation ok",
            "stderr": "",
        }
    ]


def test_validate_project_report_emits_closed_validation_report_schema(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    report_path = repo / "portability-report.json"
    report.write_json(report_path)

    payload = validate_project_report(report_path)

    assert set(payload) == project_pipeline.VALIDATION_REPORT_FIELDS
    assert payload["schemaVersion"] == project_pipeline.REPORT_SCHEMA_VERSION
    assert payload["kind"] == project_pipeline.VALIDATION_REPORT_KIND
    assert payload["sourceReport"] == str(report_path)
    assert isinstance(payload["generatedAt"], int)
    assert payload["validation"]["summary"] == {
        "artifactCount": 1,
        "okCount": 1,
        "failedCount": 0,
        "sourceHashStatusCounts": _source_hash_status_counts(ok=1),
        "generatedHashStatusCounts": _generated_hash_status_counts(ok=1),
        "sourceMapStatusCounts": _source_map_status_counts(ok=1),
        "sourceRemapStatusCounts": _source_remap_status_counts(ok=1),
    }
    assert payload["artifactStatusBySourceBackend"] == {
        "cgl": {"artifactCount": 1, "okCount": 1, "failedCount": 0}
    }
    assert payload["artifactStatusByVariant"] == {}
    assert payload["sourceHashStatusCounts"] == _source_hash_status_counts(ok=1)
    assert payload["generatedHashStatusCounts"] == _generated_hash_status_counts(ok=1)
    assert payload["sourceMapStatusCounts"] == _source_map_status_counts(ok=1)
    assert payload["sourceRemapStatusCounts"] == _source_remap_status_counts(ok=1)
    assert payload["toolchainStatusCounts"] == {
        "available": 0,
        "not-configured": 1,
        "unavailable": 0,
    }
    assert payload["toolchainRunStatusCounts"] == {"failed": 0, "ok": 0}
    assert payload["toolchainRunStatusByTarget"] == {}
    assert payload["toolchainRunStatusBySourceBackend"] == {}
    assert payload["toolchainRunStatusByVariant"] == {}


def test_validate_project_report_detects_modified_generated_artifacts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report_path = repo / "portability-report.json"

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    report.write_json(report_path)
    (repo / "out" / "cgl" / "simple.cgl").write_text(
        SIMPLE_CROSSL + "\n// edited after report\n", encoding="utf-8"
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 1}
    assert payload["validation"]["artifacts"][0]["status"] == "failed"
    assert payload["validation"]["artifacts"][0]["sourceHashStatus"] == "ok"
    assert payload["validation"]["artifacts"][0]["generatedHashStatus"] == "mismatch"
    assert payload["validation"]["artifacts"][0]["sourceMapStatus"] == "not-checked"
    assert payload["validation"]["artifacts"][0]["sourceRemapStatus"] == "ok"
    assert payload["validation"]["summary"] == {
        "artifactCount": 1,
        "okCount": 0,
        "failedCount": 1,
        "sourceHashStatusCounts": _source_hash_status_counts(ok=1),
        "generatedHashStatusCounts": _generated_hash_status_counts(mismatch=1),
        "sourceMapStatusCounts": _source_map_status_counts(**{"not-checked": 1}),
        "sourceRemapStatusCounts": _source_remap_status_counts(ok=1),
    }
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.generated-hash-mismatch"
    assert diagnostic["location"]["file"] == "simple.cgl"
    assert diagnostic["target"] == "cgl"
    assert diagnostic["missingCapabilities"] == ["artifact.manifest"]


def test_validate_project_report_groups_artifact_status_by_source_backend(tmp_path):
    repo = tmp_path / "repo"
    output_dir = repo / "out" / "cgl"
    output_dir.mkdir(parents=True)
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "shader.frag").write_text(
        "#version 450\nvoid main() { gl_FragColor = vec4(1.0); }\n",
        encoding="utf-8",
    )
    (output_dir / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report_path = repo / "portability-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["cgl"],
                    "outputDir": "out",
                },
                "artifacts": [
                    {
                        "source": "simple.cgl",
                        "sourceBackend": "cgl",
                        "target": "cgl",
                        "path": "out/cgl/simple.cgl",
                        "status": "translated",
                    },
                    {
                        "source": "shader.frag",
                        "sourceBackend": "opengl",
                        "target": "cgl",
                        "path": "out/cgl/shader.cgl",
                        "status": "translated",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)
    text_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "validate-project",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert payload["success"] is False
    assert payload["artifactStatusBySourceBackend"] == {
        "cgl": {"artifactCount": 1, "okCount": 1, "failedCount": 0},
        "opengl": {"artifactCount": 1, "okCount": 0, "failedCount": 1},
    }
    assert payload["validation"]["artifacts"] == [
        {
            "source": "simple.cgl",
            "sourceBackend": "cgl",
            "target": "cgl",
            "path": "out/cgl/simple.cgl",
            "exists": True,
            "status": "ok",
            "sourceHashStatus": "not-recorded",
            "generatedHashStatus": "not-recorded",
            "sourceMapStatus": "not-recorded",
            "sourceRemapStatus": "not-recorded",
        },
        {
            "source": "shader.frag",
            "sourceBackend": "opengl",
            "target": "cgl",
            "path": "out/cgl/shader.cgl",
            "exists": False,
            "status": "failed",
            "sourceHashStatus": "not-recorded",
            "generatedHashStatus": "missing",
            "sourceMapStatus": "not-recorded",
            "sourceRemapStatus": "not-recorded",
        },
    ]
    assert text_result.returncode == 1
    assert (
        "Validation artifacts by source backend: "
        "cgl=1 artifact (1 ok, 0 failed), "
        "opengl=1 artifact (0 ok, 1 failed)"
    ) in text_result.stdout


def test_validate_project_report_detects_modified_source_artifacts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "simple.cgl"
    source.write_text(SIMPLE_CROSSL, encoding="utf-8")
    report_path = repo / "portability-report.json"

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    report.write_json(report_path)
    source.write_text(SIMPLE_CROSSL + "\n// edited after report\n", encoding="utf-8")

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 1}
    assert payload["validation"]["artifacts"][0]["status"] == "failed"
    assert payload["validation"]["artifacts"][0]["sourceHashStatus"] == "mismatch"
    assert payload["validation"]["artifacts"][0]["generatedHashStatus"] == "ok"
    assert payload["validation"]["artifacts"][0]["sourceMapStatus"] == "not-checked"
    assert payload["validation"]["artifacts"][0]["sourceRemapStatus"] == "ok"
    assert payload["validation"]["summary"] == {
        "artifactCount": 1,
        "okCount": 0,
        "failedCount": 1,
        "sourceHashStatusCounts": _source_hash_status_counts(mismatch=1),
        "generatedHashStatusCounts": _generated_hash_status_counts(ok=1),
        "sourceMapStatusCounts": _source_map_status_counts(**{"not-checked": 1}),
        "sourceRemapStatusCounts": _source_remap_status_counts(ok=1),
    }
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.source-hash-mismatch"
    assert diagnostic["location"]["file"] == "simple.cgl"
    assert diagnostic["target"] == "cgl"
    assert diagnostic["missingCapabilities"] == ["source.provenance"]


def test_validate_project_report_detects_missing_source_artifacts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "simple.cgl"
    source.write_text(SIMPLE_CROSSL, encoding="utf-8")
    report_path = repo / "portability-report.json"

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    report.write_json(report_path)
    source.unlink()

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 1}
    assert payload["validation"]["artifacts"][0]["status"] == "failed"
    assert payload["validation"]["artifacts"][0]["sourceHashStatus"] == "missing"
    assert payload["validation"]["artifacts"][0]["generatedHashStatus"] == "ok"
    assert payload["validation"]["artifacts"][0]["sourceMapStatus"] == "not-checked"
    assert payload["validation"]["artifacts"][0]["sourceRemapStatus"] == "ok"
    assert payload["validation"]["summary"] == {
        "artifactCount": 1,
        "okCount": 0,
        "failedCount": 1,
        "sourceHashStatusCounts": _source_hash_status_counts(missing=1),
        "generatedHashStatusCounts": _generated_hash_status_counts(ok=1),
        "sourceMapStatusCounts": _source_map_status_counts(**{"not-checked": 1}),
        "sourceRemapStatusCounts": _source_remap_status_counts(ok=1),
    }
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.missing-source"
    assert diagnostic["location"]["file"] == "simple.cgl"
    assert diagnostic["target"] == "cgl"
    assert diagnostic["missingCapabilities"] == ["source.provenance"]


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
            "sourceBackend": "cgl",
            "target": "opengl",
            "path": "out/opengl/simple.glsl",
            "exists": True,
            "status": "ok",
            "sourceHashStatus": "ok",
            "generatedHashStatus": "ok",
            "sourceMapStatus": "ok",
            "sourceRemapStatus": "not-recorded",
        }
    ]
    assert payload["validation"]["summary"] == {
        "artifactCount": 1,
        "okCount": 1,
        "failedCount": 0,
        "sourceHashStatusCounts": _source_hash_status_counts(ok=1),
        "generatedHashStatusCounts": _generated_hash_status_counts(ok=1),
        "sourceMapStatusCounts": _source_map_status_counts(ok=1),
        "sourceRemapStatusCounts": _source_remap_status_counts(**{"not-recorded": 1}),
    }
    assert payload["validation"]["toolchains"][0]["target"] == "opengl"
    assert payload["validation"]["toolchains"][0]["status"] in {
        "available",
        "unavailable",
    }
    assert payload["diagnosticCounts"]["error"] == 0


def test_validate_project_report_records_unavailable_toolchains_deterministically(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    monkeypatch.setattr(project_pipeline.shutil, "which", lambda tool: None)

    report = translate_project(repo, targets=["opengl"], output_dir="out")
    report_path = repo / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)

    assert validation["success"] is True
    assert validation["diagnosticCounts"] == {"note": 0, "warning": 1, "error": 0}
    assert validation["diagnosticsByCode"] == {
        "project.validate.toolchain-unavailable": 1
    }
    assert validation["missingCapabilityCounts"] == {"toolchain.validation": 1}
    assert validation["toolchainStatusCounts"] == {
        "available": 0,
        "not-configured": 0,
        "unavailable": 1,
    }
    assert validation["validation"]["toolchains"] == [
        {
            "target": "opengl",
            "status": "unavailable",
            "tools": [
                {
                    "name": "glslangValidator",
                    "path": None,
                    "available": False,
                }
            ],
        }
    ]
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.toolchain-unavailable"
    assert diagnostic["severity"] == "warning"
    assert diagnostic["target"] == "opengl"
    assert diagnostic["missingCapabilities"] == ["toolchain.validation"]


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
    assert payload["diagnosticsByCode"] == {"project.validate.invalid-report": 1}
    assert payload["missingCapabilityCounts"] == {"artifact.manifest": 1}
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


@pytest.mark.parametrize("output_dir_kind", ("relative", "absolute"))
def test_validate_project_report_rejects_output_dir_outside_project(
    tmp_path, output_dir_kind
):
    repo = tmp_path / "repo"
    repo.mkdir()
    output_dir = (
        "../outside"
        if output_dir_kind == "relative"
        else str((tmp_path / "outside").resolve())
    )
    report_path = repo / "escaped-output-dir-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": output_dir,
                },
                "artifacts": [],
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
    assert "project.outputDir must resolve inside project.root" in (
        diagnostic["message"]
    )


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
                    "sourceRootCount": "1",
                    "includePatterns": ["*.cgl", 1],
                    "includePatternCount": [],
                    "excludePatterns": [False],
                    "excludePatternCount": False,
                    "targets": ["opengl"],
                    "outputDir": "out",
                    "includeDirs": "include",
                    "includeDirCount": "1",
                    "defines": {"USE_FAST_PATH": 1},
                    "defineCount": 2,
                    "sourceOverrides": {"gpu/*.shader": 1},
                    "sourceOverrideCount": 2,
                    "variants": {"debug": "not a define map", "": {"MODE": 1}},
                    "variantCount": "1",
                    "variantDefineCounts": {"debug": 1},
                    "selectedVariants": ["profile"],
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
    assert "project.sourceRootCount must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "project.includePatternCount must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "project.excludePatternCount must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "project.includeDirs must be a list of strings" in diagnostic["message"]
    assert "project.includeDirCount must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "project.defines values must be strings" in diagnostic["message"]
    assert "project.defineCount must match project.defines" in diagnostic["message"]
    assert "project.sourceOverrides values must be strings" in (diagnostic["message"])
    assert (
        "project.sourceOverrideCount must match project.sourceOverrides"
        in diagnostic["message"]
    )
    assert "project.variants keys must be non-empty strings" in diagnostic["message"]
    assert "project.variants.debug must be an object" in diagnostic["message"]
    assert "project.variants values must be strings" in diagnostic["message"]
    assert "project.variantCount must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "project.variantDefineCounts must match project.variants" in (
        diagnostic["message"]
    )
    assert "project.selectedVariants must be listed in project.variants" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_blank_project_config_list_entries(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "blank-project-config-list-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "sourceRoots": [" "],
                    "includePatterns": [""],
                    "excludePatterns": ["\t"],
                    "targets": ["opengl"],
                    "outputDir": "out",
                    "includeDirs": ["  "],
                },
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "project.sourceRoots entries must be non-empty strings" in (
        diagnostic["message"]
    )
    assert "project.includePatterns entries must be non-empty strings" in (
        diagnostic["message"]
    )
    assert "project.excludePatterns entries must be non-empty strings" in (
        diagnostic["message"]
    )
    assert "project.includeDirs entries must be non-empty strings" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_invalid_project_config_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            """).strip(),
        encoding="utf-8",
    )
    base_payload = translate_project(load_project_config(repo)).to_json()

    cases = (
        ("relative", "crosstl.toml", "project.config must be an absolute path"),
        ("missing", str(repo / "missing.toml"), "project.config must exist"),
        ("directory", str(repo), "project.config must be a file"),
    )
    for name, config_value, expected_message in cases:
        payload = json.loads(json.dumps(base_payload))
        payload["project"]["config"] = config_value
        report_path = repo / f"invalid-project-config-{name}.json"
        report_path.write_text(json.dumps(payload), encoding="utf-8")

        validation = validate_project_report(report_path)

        assert validation["success"] is False
        assert validation["validation"] == {"toolchains": [], "artifacts": []}
        diagnostic = validation["diagnostics"][0]
        assert diagnostic["code"] == "project.validate.invalid-report"
        assert expected_message in diagnostic["message"]


def test_validate_project_report_rejects_unexpected_generated_metadata_fields(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            """).strip(),
        encoding="utf-8",
    )
    payload = translate_project(load_project_config(repo)).to_json()
    payload["generator"]["unexpected"] = "metadata"
    payload["project"]["unexpected"] = "metadata"
    report_path = repo / "unexpected-generated-metadata-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "generator.unexpected is not allowed" in diagnostic["message"]
    assert "project.unexpected is not allowed" in diagnostic["message"]


def test_validate_project_report_rejects_unexpected_generated_report_fields(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    payload = translate_project(repo, targets=["cgl"], output_dir="out").to_json()
    payload["unexpected"] = "metadata"
    payload["summary"]["unexpected"] = "metadata"
    payload["migration"]["unexpected"] = "metadata"
    payload["migration"]["actions"][0]["unexpected"] = "metadata"
    report_path = repo / "unexpected-generated-report-fields.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "report.unexpected is not allowed" in diagnostic["message"]
    assert "summary.unexpected is not allowed" in diagnostic["message"]
    assert "migration.unexpected is not allowed" in diagnostic["message"]
    assert "migration.actions[0].unexpected is not allowed" in diagnostic["message"]


def test_validate_project_report_rejects_unexpected_generated_validation_fields(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    payload = translate_project(
        repo, targets=["opengl"], output_dir="out", validate=True
    ).to_json()
    validation_payload = payload["validation"]
    validation_artifact = validation_payload["artifacts"][0]
    validation_payload["unexpected"] = "metadata"
    validation_payload["summary"]["unexpected"] = "metadata"
    validation_payload["toolchains"][0]["unexpected"] = "metadata"
    validation_payload["toolchains"][0]["tools"][0]["unexpected"] = "metadata"
    validation_artifact["unexpected"] = "metadata"
    validation_payload["toolchainRuns"] = [
        {
            "source": validation_artifact["source"],
            "sourceBackend": validation_artifact["sourceBackend"],
            "target": validation_artifact["target"],
            "path": validation_artifact["path"],
            "command": ["glslangValidator"],
            "returncode": 0,
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "unexpected": "metadata",
        }
    ]
    report_path = repo / "out" / "unexpected-generated-validation-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "validation.unexpected is not allowed" in diagnostic["message"]
    assert "validation.summary.unexpected is not allowed" in diagnostic["message"]
    assert "validation.toolchains[0].unexpected is not allowed" in (
        diagnostic["message"]
    )
    assert "validation.toolchains[0].tools[0].unexpected is not allowed" in (
        diagnostic["message"]
    )
    assert "validation.artifacts[0].unexpected is not allowed" in diagnostic["message"]
    assert "validation.toolchainRuns[0].unexpected is not allowed" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_empty_project_mapping_keys(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "empty-project-mapping-key-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                    "defines": {"": "1"},
                    "sourceOverrides": {"": "cgl"},
                    "variants": {"debug": {"": "1"}},
                    "variantDefineCounts": {"debug": 1},
                },
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "project.defines keys must be non-empty strings" in diagnostic["message"]
    assert "project.sourceOverrides keys must be non-empty strings" in (
        diagnostic["message"]
    )
    assert "project.variants.debug keys must be non-empty strings" in (
        diagnostic["message"]
    )


def test_validate_project_report_quotes_variant_keys_with_punctuation(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "punctuated-variant-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                    "variants": {"qa/profile": {"": "1"}},
                    "variantDefineCounts": {"qa/profile": 1},
                },
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert 'project.variants["qa/profile"] keys must be non-empty strings' in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_project_config_count_mismatches(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "invalid-include-dir-count-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "sourceRoots": ["."],
                    "sourceRootCount": 2,
                    "includePatterns": [],
                    "includePatternCount": 1,
                    "excludePatterns": [],
                    "excludePatternCount": 1,
                    "targets": ["opengl"],
                    "outputDir": "out",
                    "sourceOverrides": {},
                    "sourceOverrideCount": 0,
                    "includeDirs": ["includes", "generated/includes"],
                    "includeDirCount": 1,
                    "defines": {},
                    "defineCount": 0,
                    "variants": {},
                    "variantCount": 0,
                    "variantDefineCounts": {},
                },
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "project.sourceRootCount must match project.sourceRoots" in (
        diagnostic["message"]
    )
    assert "project.includePatternCount must match project.includePatterns" in (
        diagnostic["message"]
    )
    assert "project.excludePatternCount must match project.excludePatterns" in (
        diagnostic["message"]
    )
    assert "project.includeDirCount must match project.includeDirs" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_malformed_include_dir_status_records(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "invalid-include-dir-status-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                    "includeDirs": ["includes"],
                    "includeDirCount": 1,
                    "includeDirStatus": [
                        {
                            "path": 1,
                            "resolvedPath": "includes",
                            "status": "ready",
                            "frontendVisible": "yes",
                        },
                        "not a record",
                    ],
                    "includeDirStatusCounts": [],
                },
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "project.includeDirStatus must match project.includeDirs" in (
        diagnostic["message"]
    )
    assert "project.includeDirStatus[0].path must be a string" in (
        diagnostic["message"]
    )
    assert "project.includeDirStatus[0].resolvedPath must be an absolute path" in (
        diagnostic["message"]
    )
    assert (
        "project.includeDirStatus[0].status must be a known include directory status"
        in diagnostic["message"]
    )
    assert "project.includeDirStatus[0].frontendVisible must be a boolean" in (
        diagnostic["message"]
    )
    assert "project.includeDirStatus[1] must be an object" in diagnostic["message"]
    assert "project.includeDirStatusCounts must be an object" in (diagnostic["message"])


def test_validate_project_report_rejects_malformed_source_root_status_records(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "invalid-source-root-status-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                    "sourceRoots": ["shaders"],
                    "sourceRootCount": 1,
                    "sourceRootStatus": [
                        {
                            "path": 1,
                            "resolvedPath": "shaders",
                            "status": "ready",
                            "scanVisible": "yes",
                        },
                        "not a record",
                    ],
                    "sourceRootStatusCounts": [],
                },
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "project.sourceRootStatus must match project.sourceRoots" in (
        diagnostic["message"]
    )
    assert "project.sourceRootStatus[0].path must be a string" in (
        diagnostic["message"]
    )
    assert "project.sourceRootStatus[0].resolvedPath must be an absolute path" in (
        diagnostic["message"]
    )
    assert (
        "project.sourceRootStatus[0].status must be a known source root status"
        in diagnostic["message"]
    )
    assert "project.sourceRootStatus[0].scanVisible must be a boolean" in (
        diagnostic["message"]
    )
    assert "project.sourceRootStatus[1] must be an object" in diagnostic["message"]
    assert "project.sourceRootStatusCounts must be an object" in (diagnostic["message"])


def test_validate_project_report_rejects_include_dir_status_count_mismatches(
    tmp_path,
):
    repo = tmp_path / "repo"
    include_dir = repo / "includes"
    repo.mkdir()
    include_dir.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            include_dirs = ["includes"]
            """).strip(),
        encoding="utf-8",
    )
    report = translate_project(load_project_config(repo), output_dir="out").to_json()
    report["project"]["includeDirStatusCounts"] = {"missing": 1}
    report_path = repo / "out" / "invalid-include-dir-status-count-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "project.includeDirStatusCounts must match project.includeDirStatus" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_source_root_status_count_mismatches(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    repo.mkdir()
    shader_dir.mkdir()
    (shader_dir / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            source_roots = ["shaders"]
            """).strip(),
        encoding="utf-8",
    )
    report = translate_project(load_project_config(repo), output_dir="out").to_json()
    report["project"]["sourceRootStatusCounts"] = {"missing": 1}
    report_path = repo / "out" / "invalid-source-root-status-count-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "project.sourceRootStatusCounts must match project.sourceRootStatus" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_stale_include_dir_status(tmp_path):
    repo = tmp_path / "repo"
    include_dir = repo / "includes"
    repo.mkdir()
    include_dir.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            include_dirs = ["includes"]
            """).strip(),
        encoding="utf-8",
    )
    report = translate_project(load_project_config(repo), output_dir="out").to_json()
    report["project"]["includeDirStatus"][0]["status"] = "missing"
    report_path = repo / "out" / "stale-include-dir-status-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "project.includeDirStatus[0].status must match the resolved include directory"
        in diagnostic["message"]
    )
    assert "project.includeDirStatusCounts must match project.includeDirStatus" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_stale_include_dir_resolved_path(tmp_path):
    repo = tmp_path / "repo"
    include_dir = repo / "includes"
    other_include_dir = repo / "other-includes"
    repo.mkdir()
    include_dir.mkdir()
    other_include_dir.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            include_dirs = ["includes"]
            """).strip(),
        encoding="utf-8",
    )
    report = translate_project(load_project_config(repo), output_dir="out").to_json()
    report["project"]["includeDirStatus"][0]["resolvedPath"] = str(
        other_include_dir.resolve()
    )
    report_path = repo / "out" / "stale-include-dir-resolved-path-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "project.includeDirStatus[0].resolvedPath must match the resolved "
        "include directory"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_stale_source_root_status(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    repo.mkdir()
    shader_dir.mkdir()
    (shader_dir / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            source_roots = ["shaders"]
            """).strip(),
        encoding="utf-8",
    )
    report = translate_project(load_project_config(repo), output_dir="out").to_json()
    report["project"]["sourceRootStatus"][0]["status"] = "missing"
    report_path = repo / "out" / "stale-source-root-status-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "project.sourceRootStatus[0].status must match the resolved source root" in (
        diagnostic["message"]
    )
    assert "project.sourceRootStatusCounts must match project.sourceRootStatus" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_unexpected_generated_project_status_fields(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    include_dir = repo / "includes"
    repo.mkdir()
    shader_dir.mkdir()
    include_dir.mkdir()
    (shader_dir / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            source_roots = ["shaders"]
            include_dirs = ["includes"]
            output_dir = "out"
            """).strip(),
        encoding="utf-8",
    )
    payload = translate_project(load_project_config(repo)).to_json()
    payload["project"]["sourceRootStatus"][0]["unexpected"] = "metadata"
    payload["project"]["includeDirStatus"][0]["unexpected"] = "metadata"
    report_path = repo / "out" / "unexpected-generated-project-status-fields.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "project.sourceRootStatus[0].unexpected is not allowed" in diagnostic["message"]
    )
    assert (
        "project.includeDirStatus[0].unexpected is not allowed" in diagnostic["message"]
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
                "generatedAt": 1,
                "generator": {
                    "name": "CrossTL",
                    "pipeline": "project-porting",
                    "packageVersion": "test",
                },
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


def test_validate_project_report_rejects_unexpected_generated_unit_and_skipped_fields(
    tmp_path,
):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    shader_dir.mkdir(parents=True)
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            include = ["**/*"]
            exclude = []
            """).strip(),
        encoding="utf-8",
    )
    (shader_dir / "main.frag").write_text(
        '#version 450\n#include "local.inc"\nvoid main() {}\n',
        encoding="utf-8",
    )
    (shader_dir / "local.inc").write_text("// local include\n", encoding="utf-8")
    (repo / "notes.txt").write_text("not shader code\n", encoding="utf-8")

    payload = (
        scan_project(load_project_config(repo)).to_report(targets=["cgl"]).to_json()
    )
    unit = payload["units"][0]
    dependency = unit["includeDependencies"][0]
    skipped = payload["skipped"][0]
    unit["unexpected"] = "metadata"
    dependency["unexpected"] = "metadata"
    dependency["resolvedHash"]["unexpected"] = "metadata"
    skipped["unexpected"] = "metadata"
    report_path = repo / "unexpected-generated-unit-and-skipped-fields-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "units[0].unexpected is not allowed" in diagnostic["message"]
    assert (
        "units[0].includeDependencies[0].unexpected is not allowed"
        in diagnostic["message"]
    )
    assert (
        "units[0].includeDependencies[0].resolvedHash.unexpected is not allowed"
        in diagnostic["message"]
    )
    assert "skipped[0].unexpected is not allowed" in diagnostic["message"]


def test_validate_project_report_rejects_missing_unit_source_hashes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["units"][0].pop("sourceHash")
    report_path = repo / "out" / "missing-unit-source-hash-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "units[0].sourceHash must be an object" in diagnostic["message"]


def test_validate_project_report_rejects_artifact_source_hash_mismatches_unit_source_hash(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    payload = translate_project(repo, targets=["cgl"], output_dir="out").to_json()
    payload["units"][0]["sourceHash"]["value"] = "0" * 64
    report_path = repo / "out" / "artifact-unit-source-hash-mismatch-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].sourceHash must match units[0].sourceHash"
        in diagnostic["message"]
    )


def test_translate_project_preserves_discovered_unit_source_hash(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "simple.cgl"
    source.write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    captured_hash = report.to_json()["units"][0]["sourceHash"]
    source.write_text(
        SIMPLE_CROSSL + "\n// edited before serialization\n", encoding="utf-8"
    )

    payload = report.to_json()

    assert payload["units"][0]["sourceHash"] == captured_hash
    assert payload["artifacts"][0]["sourceHash"] == captured_hash
    assert project_pipeline._source_hash(source) != captured_hash


def test_validate_project_report_detects_modified_unit_sources(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "simple.cgl"
    source.write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = scan_project(repo).to_report(targets=["cgl"])
    report_path = repo / "scan-report.json"
    report.write_json(report_path)
    source.write_text(SIMPLE_CROSSL + "\n// edited\n", encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "units[0].sourceHash must match the current source file" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_unit_extension_mismatches(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "unit-extension-mismatch-report.json"
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
                        "id": "simple.cgl",
                        "path": "simple.cgl",
                        "sourceBackend": "cgl",
                        "extension": ".hlsl",
                    }
                ],
                "skipped": [],
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "units[0].extension must match units[0].path suffix" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_duplicate_unit_and_skipped_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "duplicate-scan-record-report.json"
    unit = {
        "id": "simple.cgl",
        "path": "simple.cgl",
        "sourceBackend": "cgl",
        "extension": ".cgl",
    }
    skipped = {
        "path": "ignored.shader",
        "reason": "unsupported-extension",
    }
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
                "units": [unit, dict(unit)],
                "skipped": [
                    {"path": "simple.cgl", "reason": "unsupported-extension"},
                    skipped,
                    dict(skipped),
                ],
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "units[1].path duplicates units[0].path" in diagnostic["message"]
    assert "skipped[2].path duplicates skipped[1].path" in diagnostic["message"]
    assert "skipped[0].path duplicates units[0].path" in diagnostic["message"]


def test_validate_project_report_rejects_drive_relative_unit_and_skipped_paths(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "drive-relative-scan-record-report.json"
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
                        "id": "C:tmp/simple.cgl",
                        "path": "C:tmp/simple.cgl",
                        "sourceBackend": "cgl",
                        "extension": ".cgl",
                    }
                ],
                "skipped": [
                    {
                        "path": "C:tmp/ignored.shader",
                        "reason": "unsupported-extension",
                    }
                ],
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "units[0].path must be repository-relative" in diagnostic["message"]
    assert "skipped[0].path must be repository-relative" in diagnostic["message"]


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
                "generatedAt": False,
                "generator": {
                    "name": "",
                    "pipeline": "single-file-translate",
                    "packageVersion": [],
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
                            "sourceBackend": "",
                            "sourceHashStatus": "ready",
                            "generatedHashStatus": "ready",
                            "sourceMapStatus": "ready",
                            "sourceRemapStatus": "ready",
                        },
                        "not an artifact check",
                    ],
                    "summary": {
                        "artifactCount": "2",
                        "okCount": 2,
                        "failedCount": 0,
                        "sourceHashStatusCounts": {"ok": 2},
                        "generatedHashStatusCounts": [],
                    },
                    "toolchainRuns": [
                        {
                            "source": "",
                            "sourceBackend": "",
                            "target": "",
                            "path": "",
                            "variant": "",
                            "command": ["glslangValidator", ""],
                            "checkKind": "maybe",
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
    assert "generatedAt must be a non-negative integer" in diagnostic["message"]
    assert "generator.name must be a string" in diagnostic["message"]
    assert "generator.pipeline must be project-porting" in diagnostic["message"]
    assert "generator.packageVersion must be a string" in diagnostic["message"]
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
    assert "validation.artifacts[0].sourceBackend must be a string" in (
        diagnostic["message"]
    )
    assert "validation.artifacts[0].sourceHashStatus must be one of" in (
        diagnostic["message"]
    )
    assert "validation.artifacts[0].generatedHashStatus must be one of" in (
        diagnostic["message"]
    )
    assert "validation.artifacts[0].sourceMapStatus must be one of" in (
        diagnostic["message"]
    )
    assert "validation.artifacts[0].sourceRemapStatus must be one of" in (
        diagnostic["message"]
    )
    assert "validation.artifacts[1] must be an object" in diagnostic["message"]
    assert "validation.summary.artifactCount must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "validation.summary.okCount must match validation.artifacts" in (
        diagnostic["message"]
    )
    assert (
        "validation.summary.sourceHashStatusCounts must match validation.artifacts"
        in diagnostic["message"]
    )
    assert "validation.summary.generatedHashStatusCounts must be an object" in (
        diagnostic["message"]
    )
    assert "validation.toolchainRuns[0].source must be a string" in (
        diagnostic["message"]
    )
    assert "validation.toolchainRuns[0].sourceBackend must be a string" in (
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
    assert (
        "validation.toolchainRuns[0].command must be a non-empty list of strings"
        in diagnostic["message"]
    )
    assert "validation.toolchainRuns[0].checkKind must be one of" in (
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


def test_validate_project_report_rejects_empty_toolchain_run_command(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "empty-toolchain-run-command-report.json"
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
                        "sourceBackend": "cgl",
                        "target": "opengl",
                        "path": "out/opengl/simple.glsl",
                        "status": "translated",
                    }
                ],
                "validation": {
                    "toolchains": [],
                    "artifacts": [],
                    "toolchainRuns": [
                        {
                            "source": "simple.cgl",
                            "sourceBackend": "cgl",
                            "target": "opengl",
                            "path": "out/opengl/simple.glsl",
                            "command": [],
                            "returncode": 0,
                            "status": "ok",
                            "stdout": "",
                            "stderr": "",
                        }
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
    assert (
        "validation.toolchainRuns[0].command must be a non-empty list of strings"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_toolchain_run_command_target_mismatch(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "mismatched-toolchain-run-command-report.json"
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
                        "sourceBackend": "cgl",
                        "target": "opengl",
                        "path": "out/opengl/simple.glsl",
                        "status": "translated",
                    }
                ],
                "validation": {
                    "toolchains": [],
                    "artifacts": [],
                    "toolchainRuns": [
                        {
                            "source": "simple.cgl",
                            "sourceBackend": "cgl",
                            "target": "opengl",
                            "path": "out/opengl/simple.glsl",
                            "command": ["dxc", "-help"],
                            "returncode": 0,
                            "status": "ok",
                            "stdout": "",
                            "stderr": "",
                        }
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
    assert (
        "validation.toolchainRuns[0].command[0] must match a configured "
        "validation tool for target opengl"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_inconsistent_toolchain_status(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "inconsistent-toolchain-status-report.json"
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
                "validation": {
                    "toolchains": [
                        {
                            "target": "opengl",
                            "status": "available",
                            "tools": [
                                {
                                    "name": "glslangValidator",
                                    "path": None,
                                    "available": False,
                                }
                            ],
                        }
                    ],
                    "artifacts": [],
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "validation.toolchains[0].status must match tools availability"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_duplicate_toolchain_targets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "duplicate-toolchain-target-report.json"
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
                "validation": {
                    "toolchains": [
                        {
                            "target": "opengl",
                            "status": "not-configured",
                            "tools": [],
                        },
                        {
                            "target": "opengl",
                            "status": "not-configured",
                            "tools": [],
                        },
                    ],
                    "artifacts": [],
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "validation.toolchains[1] duplicates validation.toolchains[0].target"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_duplicate_toolchain_tool_names(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "duplicate-toolchain-tool-report.json"
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
                "validation": {
                    "toolchains": [
                        {
                            "target": "opengl",
                            "status": "available",
                            "tools": [
                                {
                                    "name": "glslangValidator",
                                    "path": "/usr/bin/glslangValidator",
                                    "available": True,
                                },
                                {
                                    "name": "glslangValidator",
                                    "path": "/opt/bin/glslangValidator",
                                    "available": True,
                                },
                            ],
                        },
                    ],
                    "artifacts": [],
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "validation.toolchains[0].tools[1].name duplicates "
        "validation.toolchains[0].tools[0].name"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_inconsistent_validation_artifact_status(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "inconsistent-validation-artifact-status-report.json"
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
                    },
                    {
                        "source": "clean.cgl",
                        "target": "opengl",
                        "path": "out/opengl/clean.glsl",
                        "status": "translated",
                    },
                ],
                "validation": {
                    "toolchains": [],
                    "artifacts": [
                        {
                            "source": "simple.cgl",
                            "target": "opengl",
                            "path": "out/opengl/simple.glsl",
                            "exists": False,
                            "status": "ok",
                            "sourceHashStatus": "ok",
                            "generatedHashStatus": "ok",
                        },
                        {
                            "source": "clean.cgl",
                            "target": "opengl",
                            "path": "out/opengl/clean.glsl",
                            "exists": True,
                            "status": "failed",
                            "sourceHashStatus": "ok",
                            "generatedHashStatus": "ok",
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "validation.artifacts[0].status must match exists, hash statuses, "
        "and provenance statuses" in diagnostic["message"]
    )
    assert (
        "validation.artifacts[1].status must match exists, hash statuses, "
        "and provenance statuses" in diagnostic["message"]
    )


def test_validate_project_report_rejects_summarized_validation_without_status_fields(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out", validate=True)
    payload = report.to_json()
    validation_artifact = payload["validation"]["artifacts"][0]
    validation_artifact.pop("sourceHashStatus")
    validation_artifact.pop("generatedHashStatus")
    validation_artifact.pop("sourceMapStatus")
    validation_artifact.pop("sourceRemapStatus")
    payload["validation"]["summary"][
        "sourceHashStatusCounts"
    ] = _source_hash_status_counts()
    payload["validation"]["summary"][
        "generatedHashStatusCounts"
    ] = _generated_hash_status_counts()
    payload["validation"]["summary"][
        "sourceMapStatusCounts"
    ] = _source_map_status_counts()
    payload["validation"]["summary"][
        "sourceRemapStatusCounts"
    ] = _source_remap_status_counts()
    report_path = repo / "out" / "validation-artifact-missing-status-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "validation.artifacts[0].sourceHashStatus must be recorded "
        "when validation.summary is present"
    ) in diagnostic["message"]
    assert (
        "validation.artifacts[0].generatedHashStatus must be recorded "
        "when validation.summary is present"
    ) in diagnostic["message"]
    assert (
        "validation.artifacts[0].sourceMapStatus must be recorded "
        "when validation.summary is present"
    ) in diagnostic["message"]
    assert (
        "validation.artifacts[0].sourceRemapStatus must be recorded "
        "when validation.summary is present"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_validation_ok_for_failed_report_artifact(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "validation-ok-for-failed-artifact-report.json"
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
                        "status": "failed",
                        "error": "translation failed",
                    }
                ],
                "validation": {
                    "toolchains": [],
                    "artifacts": [
                        {
                            "source": "simple.cgl",
                            "target": "opengl",
                            "path": "out/opengl/simple.glsl",
                            "exists": True,
                            "status": "ok",
                            "sourceHashStatus": "ok",
                            "generatedHashStatus": "ok",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "validation.artifacts[0].status must match report.artifacts[0].status"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_toolchain_runs_for_failed_artifacts(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "toolchain-run-for-failed-artifact-report.json"
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
                        "status": "failed",
                        "error": "translation failed",
                    }
                ],
                "validation": {
                    "toolchains": [],
                    "artifacts": [],
                    "toolchainRuns": [
                        {
                            "source": "simple.cgl",
                            "target": "opengl",
                            "path": "out/opengl/simple.glsl",
                            "command": ["glslangValidator"],
                            "returncode": 0,
                            "status": "ok",
                            "stdout": "",
                            "stderr": "",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "validation.toolchainRuns[0] must reference a translated "
        "report.artifacts[0] record"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_validation_records_with_undeclared_targets(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "undeclared-validation-target-report.json"
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
                "validation": {
                    "toolchains": [
                        {
                            "target": "metal",
                            "status": "not-configured",
                            "tools": [],
                        }
                    ],
                    "artifacts": [
                        {
                            "source": "simple.cgl",
                            "target": "metal",
                            "path": "out/metal/simple.metal",
                            "exists": True,
                            "status": "ok",
                        }
                    ],
                    "toolchainRuns": [
                        {
                            "source": "simple.cgl",
                            "target": "metal",
                            "path": "out/metal/simple.metal",
                            "command": ["xcrun", "metal"],
                            "returncode": 0,
                            "status": "ok",
                            "stdout": "",
                            "stderr": "",
                        }
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
    assert "validation.toolchains[0].target must be listed in project.targets" in (
        diagnostic["message"]
    )
    assert "validation.artifacts[0].target must be listed in project.targets" in (
        diagnostic["message"]
    )
    assert "validation.toolchainRuns[0].target must be listed in project.targets" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_validation_records_with_undeclared_variants(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "undeclared-validation-variant-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                    "variants": {"debug": {}},
                    "variantDefineCounts": {"debug": 0},
                },
                "artifacts": [],
                "validation": {
                    "toolchains": [],
                    "artifacts": [
                        {
                            "source": "simple.cgl",
                            "target": "opengl",
                            "path": "out/opengl/simple.glsl",
                            "exists": True,
                            "status": "ok",
                            "variant": "profile",
                        }
                    ],
                    "toolchainRuns": [
                        {
                            "source": "simple.cgl",
                            "target": "opengl",
                            "path": "out/opengl/simple.glsl",
                            "variant": "profile",
                            "command": ["glslangValidator"],
                            "returncode": 0,
                            "status": "ok",
                            "stdout": "",
                            "stderr": "",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "validation.artifacts[0].variant must be listed in project.variants" in (
        diagnostic["message"]
    )
    assert "validation.toolchainRuns[0].variant must be listed in project.variants" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_validation_records_with_undeclared_artifacts(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "undeclared-validation-artifact-report.json"
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
                "validation": {
                    "toolchains": [],
                    "artifacts": [
                        {
                            "source": "simple.cgl",
                            "target": "opengl",
                            "path": "out/opengl/other.glsl",
                            "exists": True,
                            "status": "ok",
                        }
                    ],
                    "toolchainRuns": [
                        {
                            "source": "simple.cgl",
                            "target": "opengl",
                            "path": "out/opengl/other.glsl",
                            "command": ["glslangValidator"],
                            "returncode": 0,
                            "status": "ok",
                            "stdout": "",
                            "stderr": "",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "validation.artifacts[0] must reference an artifact in report.artifacts" in (
        diagnostic["message"]
    )
    assert (
        "validation.toolchainRuns[0] must reference an artifact in report.artifacts"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_validation_source_backend_mismatches(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out", validate=True)
    payload = report.to_json()
    validation_artifact = payload["validation"]["artifacts"][0]
    validation_artifact["sourceBackend"] = "opengl"
    payload["validation"]["toolchainRuns"] = [
        {
            "source": validation_artifact["source"],
            "target": validation_artifact["target"],
            "path": validation_artifact["path"],
            "sourceBackend": "opengl",
            "command": ["crosstl-validate"],
            "returncode": 0,
            "status": "ok",
            "stdout": "",
            "stderr": "",
        }
    ]
    report_path = repo / "out" / "validation-source-backend-mismatch-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "validation.artifacts[0].sourceBackend must match "
        "report.artifacts[0].sourceBackend"
    ) in diagnostic["message"]
    assert (
        "validation.toolchainRuns[0].sourceBackend must match "
        "report.artifacts[0].sourceBackend"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_validation_source_backend_omissions(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out", validate=True)
    payload = report.to_json()
    validation_artifact = payload["validation"]["artifacts"][0]
    validation_artifact.pop("sourceBackend")
    payload["validation"]["toolchainRuns"] = [
        {
            "source": validation_artifact["source"],
            "target": validation_artifact["target"],
            "path": validation_artifact["path"],
            "command": ["crosstl-validate"],
            "returncode": 0,
            "status": "ok",
            "stdout": "",
            "stderr": "",
        }
    ]
    report_path = repo / "out" / "validation-source-backend-omission-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "validation.artifacts[0].sourceBackend must be recorded when "
        "report.artifacts[0].sourceBackend is recorded"
    ) in diagnostic["message"]
    assert (
        "validation.toolchainRuns[0].sourceBackend must be recorded when "
        "report.artifacts[0].sourceBackend is recorded"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_validation_summary_missing_artifact_checks(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "incomplete-validation-summary-report.json"
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
                    },
                    {
                        "source": "other.cgl",
                        "target": "opengl",
                        "path": "out/opengl/other.glsl",
                        "status": "translated",
                    },
                ],
                "validation": {
                    "toolchains": [],
                    "artifacts": [
                        {
                            "source": "simple.cgl",
                            "target": "opengl",
                            "path": "out/opengl/simple.glsl",
                            "exists": True,
                            "status": "ok",
                        }
                    ],
                    "summary": {
                        "artifactCount": 1,
                        "okCount": 1,
                        "failedCount": 0,
                        "sourceHashStatusCounts": _source_hash_status_counts(),
                        "generatedHashStatusCounts": _generated_hash_status_counts(),
                        "sourceMapStatusCounts": _source_map_status_counts(),
                        "sourceRemapStatusCounts": _source_remap_status_counts(),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "validation.artifacts must include report.artifacts[1]" in diagnostic["message"]
    )


def test_validate_project_report_rejects_validation_summary_missing_toolchain_targets(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(
        repo,
        targets=["cgl", "opengl"],
        output_dir="out",
        validate=True,
    )
    payload = report.to_json()
    payload["validation"]["toolchains"] = payload["validation"]["toolchains"][:1]
    report_path = repo / "out" / "validation-missing-toolchain-target-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "validation.toolchains must include project.targets[1]" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_validation_records_with_escaped_paths(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "escaped-validation-path-report.json"
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
                "validation": {
                    "toolchains": [],
                    "artifacts": [
                        {
                            "source": "../outside/simple.cgl",
                            "target": "opengl",
                            "path": "C:tmp/simple.glsl",
                            "exists": True,
                            "status": "ok",
                        }
                    ],
                    "toolchainRuns": [
                        {
                            "source": "C:tmp/simple.cgl",
                            "target": "opengl",
                            "path": "../outside/simple.glsl",
                            "command": ["glslangValidator"],
                            "returncode": 0,
                            "status": "ok",
                            "stdout": "",
                            "stderr": "",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "validation.artifacts[0].source must be repository-relative" in (
        diagnostic["message"]
    )
    assert "validation.artifacts[0].path must be repository-relative" in (
        diagnostic["message"]
    )
    assert "validation.toolchainRuns[0].source must be repository-relative" in (
        diagnostic["message"]
    )
    assert "validation.toolchainRuns[0].path must be repository-relative" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_duplicate_validation_record_identities(
    tmp_path,
):
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
    validation_artifact = dict(payload["validation"]["artifacts"][0])
    payload["validation"]["artifacts"].append(validation_artifact)
    payload["validation"]["summary"] = {
        "artifactCount": 2,
        "okCount": 2,
        "failedCount": 0,
        "sourceHashStatusCounts": {"ok": 2},
        "generatedHashStatusCounts": {"ok": 2},
        "sourceMapStatusCounts": {"ok": 2},
        "sourceRemapStatusCounts": {"not-recorded": 2},
    }
    toolchain_run = {
        "source": "simple.cgl",
        "target": "opengl",
        "path": "out/opengl/simple.glsl",
        "command": ["glslangValidator", "--stdin"],
        "returncode": 0,
        "status": "ok",
        "stdout": "",
        "stderr": "",
    }
    payload["validation"]["toolchainRuns"] = [toolchain_run, dict(toolchain_run)]
    report_path = repo / "out" / "portability-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "validation.artifacts[1] duplicates validation.artifacts[0] identity" in (
        diagnostic["message"]
    )
    assert (
        "validation.toolchainRuns[1] duplicates validation.toolchainRuns[0] identity"
        in diagnostic["message"]
    )


def test_validate_project_report_accepts_legacy_validation_without_summary(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "legacy-validation-report.json"
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
                "validation": {
                    "toolchains": [],
                    "artifacts": [],
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is True
    assert payload["validation"]["summary"] == {
        "artifactCount": 0,
        "okCount": 0,
        "failedCount": 0,
        "sourceHashStatusCounts": _source_hash_status_counts(),
        "generatedHashStatusCounts": _generated_hash_status_counts(),
        "sourceMapStatusCounts": _source_map_status_counts(),
        "sourceRemapStatusCounts": _source_remap_status_counts(),
    }


def test_validate_project_report_rejects_missing_diagnostic_summary_rollups(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    payload = translate_project(repo, targets=["opengl"], output_dir="out").to_json()
    payload["summary"].pop("diagnosticsByCode")
    payload["summary"].pop("missingCapabilityCounts")
    report_path = repo / "report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "summary.diagnosticsByCode must be an object" in diagnostic["message"]
    assert "summary.missingCapabilityCounts must be an object" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_malformed_validation_summary(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "invalid-validation-summary-report.json"
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
                "validation": {
                    "toolchains": [],
                    "artifacts": [],
                    "summary": "not a summary",
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "validation.summary must be an object" in diagnostic["message"]


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


def test_validate_project_report_rejects_unexpected_generated_external_corpus_fields(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "corpus.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "entries": [
                    {
                        "id": "repo/simple",
                        "path": "simple.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            output_dir = "out"
            external_corpus_manifest = "corpus.json"
            """).strip(),
        encoding="utf-8",
    )

    payload = translate_project(load_project_config(repo)).to_json()
    payload["externalCorpus"]["unexpected"] = "metadata"
    payload["externalCorpus"]["entries"][0]["unexpected"] = "metadata"
    payload["externalCorpus"]["summary"]["unexpected"] = "metadata"
    report_path = repo / "out" / "unexpected-external-corpus-fields-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "externalCorpus.unexpected is not allowed" in diagnostic["message"]
    assert (
        "externalCorpus.entries[0].unexpected is not allowed" in diagnostic["message"]
    )
    assert "externalCorpus.summary.unexpected is not allowed" in diagnostic["message"]


def test_validate_project_report_rejects_missing_external_corpus_accounting(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "corpus.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "entries": [
                    {
                        "id": "repo/simple",
                        "path": "simple.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                    }
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
    payload = translate_project(load_project_config(repo)).to_json()
    payload["externalCorpus"]["summary"].pop("manifestEntryCount")
    payload["externalCorpus"]["summary"].pop("validEntryCount")
    payload["externalCorpus"]["summary"].pop("invalidEntryCount")
    report_path = repo / "missing-external-corpus-accounting-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "externalCorpus.summary.manifestEntryCount must be a non-negative integer"
        in diagnostic["message"]
    )
    assert (
        "externalCorpus.summary.validEntryCount must be a non-negative integer"
        in diagnostic["message"]
    )
    assert (
        "externalCorpus.summary.invalidEntryCount must be a non-negative integer"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_noncanonical_external_corpus_targets(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "corpus.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "entries": [
                    {
                        "id": "repo/simple",
                        "path": "simple.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                    }
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
    payload = translate_project(load_project_config(repo)).to_json()
    payload["externalCorpus"]["entries"][0]["targets"] = ["crossgl", "cgl", "OpenGL"]
    payload["externalCorpus"]["summary"]["entriesByTarget"] = {
        "OpenGL": 1,
        "cgl": 1,
        "crossgl": 1,
    }
    report_path = repo / "noncanonical-external-corpus-targets.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "externalCorpus.entries[0].targets must use normalized backend names "
        "without duplicates"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_malformed_external_corpus_provenance(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "corpus.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "entries": [
                    {
                        "id": "repo/simple",
                        "path": "simple.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                        "repository": "https://github.com/example/project",
                        "commit": "1" * 40,
                        "sourceUrl": (
                            "https://github.com/example/project/blob/"
                            f"{'1' * 40}/simple.cgl"
                        ),
                    }
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
    payload = translate_project(load_project_config(repo)).to_json()
    payload["externalCorpus"]["entries"][0]["commit"] = "ABC123"
    payload["externalCorpus"]["entries"][0][
        "sourceUrl"
    ] = "https://github.com/other/project/blob/ABC123/simple.cgl"
    report_path = repo / "malformed-external-corpus-provenance.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "externalCorpus.entries[0].commit must be a lowercase "
        "40-character hex digest"
    ) in diagnostic["message"]
    assert (
        "externalCorpus.entries[0].sourceUrl must start with repository"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_drive_relative_external_corpus_paths(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "drive-relative-external-corpus-report.json"
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
                    "schemaVersion": 1,
                    "manifest": "corpus.json",
                    "status": "ok",
                    "entries": [
                        {
                            "id": "repo/simple",
                            "path": "C:tmp/simple.glsl",
                            "sourceBackend": "opengl",
                            "targets": ["opengl"],
                            "present": False,
                            "discovered": False,
                            "artifactCount": 0,
                            "translatedCount": 0,
                            "failedCount": 0,
                        }
                    ],
                    "summary": {
                        "entryCount": 1,
                        "presentCount": 0,
                        "missingCount": 1,
                        "discoveredUnitCount": 0,
                        "undiscoveredPresentCount": 0,
                        "entriesBySourceBackend": {"opengl": 1},
                        "entriesByTarget": {"opengl": 1},
                        "artifactsByTarget": {},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "externalCorpus.entries[0].path must be repository-relative" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_external_corpus_entry_state_mismatches(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "corpus.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "entries": [
                    {
                        "id": "repo/simple",
                        "path": "simple.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                    }
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
    entry = payload["externalCorpus"]["entries"][0]
    entry["present"] = False
    entry["discovered"] = False
    entry["sourceBackend"] = "directx"
    payload["externalCorpus"]["summary"].update(
        {
            "presentCount": 0,
            "missingCount": 1,
            "discoveredUnitCount": 0,
            "entriesBySourceBackend": {"directx": 1},
        }
    )
    report_path = repo / "portability-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "externalCorpus.entries[0].present must match project.root" in (
        diagnostic["message"]
    )
    assert "externalCorpus.entries[0].discovered must match units" in (
        diagnostic["message"]
    )
    assert (
        "externalCorpus.entries[0].sourceBackend must match units[0].sourceBackend"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_duplicate_external_corpus_entries(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "duplicate-external-corpus-entry-report.json"
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
                    "schemaVersion": 1,
                    "manifest": "corpus.json",
                    "status": "ok",
                    "entries": [
                        {
                            "id": "repo/simple",
                            "path": "shaders/simple.glsl",
                            "sourceBackend": "opengl",
                            "targets": ["opengl"],
                            "present": False,
                            "discovered": False,
                            "artifactCount": 0,
                            "translatedCount": 0,
                            "failedCount": 0,
                        },
                        {
                            "id": "repo/simple",
                            "path": "shaders/simple.glsl",
                            "sourceBackend": "opengl",
                            "targets": ["opengl"],
                            "present": False,
                            "discovered": False,
                            "artifactCount": 0,
                            "translatedCount": 0,
                            "failedCount": 0,
                        },
                    ],
                    "summary": {
                        "entryCount": 2,
                        "presentCount": 0,
                        "missingCount": 2,
                        "discoveredUnitCount": 0,
                        "undiscoveredPresentCount": 0,
                        "entriesBySourceBackend": {"opengl": 2},
                        "entriesByTarget": {"opengl": 2},
                        "artifactsByTarget": {},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "externalCorpus.entries[1].id duplicates externalCorpus.entries[0].id"
        in diagnostic["message"]
    )
    assert (
        "externalCorpus.entries[1].path duplicates externalCorpus.entries[0].path"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_mismatched_external_corpus_manifest(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "mismatched-external-corpus-report.json"
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
                    "schemaVersion": 1,
                    "manifest": "other-corpus.json",
                    "status": "missing",
                    "entries": [],
                    "summary": {
                        "entryCount": 0,
                        "presentCount": 0,
                        "missingCount": 0,
                        "discoveredUnitCount": 0,
                        "undiscoveredPresentCount": 0,
                        "entriesBySourceBackend": {},
                        "entriesByTarget": {},
                        "artifactsByTarget": {},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "externalCorpus.manifest must match project.externalCorpusManifest"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_empty_external_corpus_manifest(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "empty-external-corpus-manifest-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl"],
                    "outputDir": "out",
                    "externalCorpusManifest": "",
                },
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "project.externalCorpusManifest must be a non-empty string or null"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_non_ok_external_corpus_with_entries(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "non-ok-external-corpus-with-entries-report.json"
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
                    "schemaVersion": 1,
                    "manifest": "corpus.json",
                    "status": "missing",
                    "entries": [
                        {
                            "id": "repo/missing",
                            "path": "missing.glsl",
                            "sourceBackend": "opengl",
                            "targets": ["opengl"],
                            "present": False,
                            "discovered": False,
                            "artifactCount": 0,
                            "translatedCount": 0,
                            "failedCount": 0,
                        }
                    ],
                    "summary": {
                        "entryCount": 1,
                        "presentCount": 0,
                        "missingCount": 1,
                        "discoveredUnitCount": 0,
                        "undiscoveredPresentCount": 0,
                        "entriesBySourceBackend": {"opengl": 1},
                        "entriesByTarget": {"opengl": 1},
                        "artifactsByTarget": {},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "externalCorpus.entries must be empty when status is not ok" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_external_corpus_entry_count_mismatches(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "corpus.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "entries": [
                    {
                        "id": "repo/simple",
                        "path": "simple.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                    }
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
    payload = translate_project(load_project_config(repo)).to_json()
    payload["externalCorpus"]["entries"][0]["artifactCount"] = 2
    payload["externalCorpus"]["entries"][0]["translatedCount"] = 0
    payload["externalCorpus"]["entries"][0]["failedCount"] = 1
    payload["externalCorpus"]["summary"]["manifestEntryCount"] = 3
    payload["externalCorpus"]["summary"]["validEntryCount"] = 2
    payload["externalCorpus"]["summary"]["invalidEntryCount"] = 2
    report_path = repo / "invalid-external-corpus-counts.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "externalCorpus.entries[0].artifactCount must match report.artifacts"
    ) in diagnostic["message"]
    assert (
        "externalCorpus.entries[0].translatedCount must match report.artifacts"
    ) in diagnostic["message"]
    assert (
        "externalCorpus.entries[0].failedCount must match report.artifacts"
    ) in diagnostic["message"]
    assert (
        "externalCorpus.summary.validEntryCount must match externalCorpus.entries"
        in diagnostic["message"]
    )
    assert (
        "externalCorpus.summary.manifestEntryCount must equal "
        "externalCorpus.summary.validEntryCount plus "
        "externalCorpus.summary.invalidEntryCount"
    ) in diagnostic["message"]


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


def test_validate_project_report_rejects_unexpected_generated_artifact_fields(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    payload = translate_project(repo, targets=["cgl"], output_dir="out").to_json()
    payload["artifacts"][0]["unexpected"] = "metadata"
    report_path = repo / "out" / "unexpected-generated-artifact-fields-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].unexpected is not allowed" in diagnostic["message"]


def test_validate_project_report_rejects_unexpected_generated_artifact_metadata_fields(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    payload = translate_project(repo, targets=["cgl"], output_dir="out").to_json()
    payload["units"][0]["sourceHash"]["unexpected"] = "metadata"
    artifact = payload["artifacts"][0]
    artifact["sourceHash"]["unexpected"] = "metadata"
    artifact["generatedHash"]["unexpected"] = "metadata"
    artifact["provenance"]["unexpected"] = "metadata"
    artifact["sourceRemap"]["unexpected"] = "metadata"
    artifact["sourceRemap"]["hash"]["unexpected"] = "metadata"
    report_path = (
        repo / "out" / "unexpected-generated-artifact-metadata-fields-report.json"
    )
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "units[0].sourceHash.unexpected is not allowed" in diagnostic["message"]
    assert "artifacts[0].sourceHash.unexpected is not allowed" in (
        diagnostic["message"]
    )
    assert "artifacts[0].generatedHash.unexpected is not allowed" in (
        diagnostic["message"]
    )
    assert "artifacts[0].provenance.unexpected is not allowed" in (
        diagnostic["message"]
    )
    assert "artifacts[0].sourceRemap.unexpected is not allowed" in (
        diagnostic["message"]
    )
    assert "artifacts[0].sourceRemap.hash.unexpected is not allowed" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_failed_artifacts_without_error(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    artifact = payload["artifacts"][0]
    artifact["status"] = "failed"
    artifact.pop("error", None)
    payload["summary"]["translatedCount"] = 0
    payload["summary"]["failedCount"] = 1
    payload["summary"]["artifactsByTarget"]["cgl"]["translatedCount"] = 0
    payload["summary"]["artifactsByTarget"]["cgl"]["failedCount"] = 1
    report_path = repo / "out" / "missing-failed-artifact-error-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].error must be a string" in diagnostic["message"]


def test_validate_project_report_rejects_translated_artifacts_with_error_metadata(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["artifacts"][0]["error"] = "translation failed"
    report_path = repo / "out" / "translated-artifact-error-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].error must be omitted for translated artifacts"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_failed_artifacts_without_source_hash(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    artifact = payload["artifacts"][0]
    artifact["status"] = "failed"
    artifact["error"] = "translation failed"
    artifact.pop("sourceHash")
    artifact.pop("generatedHash")
    artifact.pop("sourceMap")
    artifact.pop("sourceRemap")
    payload["summary"]["translatedCount"] = 0
    payload["summary"]["failedCount"] = 1
    payload["summary"]["artifactsByTarget"]["cgl"]["translatedCount"] = 0
    payload["summary"]["artifactsByTarget"]["cgl"]["failedCount"] = 1
    payload["summary"]["sourceMapCount"] = 0
    payload["summary"]["sourceRemapCount"] = 0
    payload["summary"]["sourceMapsByGranularity"] = {}
    payload["summary"]["sourceMapsByTarget"] = {}
    payload["summary"]["sourceMapsBySourceBackend"] = {}
    payload["summary"]["sourceRemapsByTarget"] = {}
    payload["summary"]["sourceRemapsBySourceBackend"] = {}
    report_path = repo / "out" / "failed-artifact-missing-source-hash-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].sourceHash must be an object" in diagnostic["message"]


def test_validate_project_report_rejects_failed_artifacts_with_generated_metadata(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    artifact = payload["artifacts"][0]
    artifact["status"] = "failed"
    artifact["error"] = "translation failed"
    payload["summary"]["translatedCount"] = 0
    payload["summary"]["failedCount"] = 1
    payload["summary"]["artifactsByTarget"]["cgl"]["translatedCount"] = 0
    payload["summary"]["artifactsByTarget"]["cgl"]["failedCount"] = 1
    report_path = repo / "out" / "failed-artifact-generated-metadata-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].generatedHash must be omitted for failed artifacts"
        in diagnostic["message"]
    )
    assert (
        "artifacts[0].sourceMap must be omitted for failed artifacts"
        in diagnostic["message"]
    )
    assert (
        "artifacts[0].sourceRemap must be omitted for failed artifacts"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_artifacts_with_escaped_source_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = repo / "out" / "opengl" / "simple.glsl"
    output.parent.mkdir(parents=True)
    output.write_text("#version 450\nvoid main() {}\n", encoding="utf-8")
    report_path = repo / "escaped-artifact-source-report.json"
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
                        "source": "../outside/simple.cgl",
                        "target": "opengl",
                        "path": "out/opengl/simple.glsl",
                        "status": "translated",
                    }
                ],
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
    assert "artifacts[0].source must be repository-relative" in (diagnostic["message"])
    assert diagnostic["missingCapabilities"] == ["artifact.manifest"]


def test_validate_project_report_rejects_artifact_paths_outside_output_dir(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    misplaced_output = repo / "misplaced.cgl"
    misplaced_output.write_text(
        (repo / "out" / "cgl" / "simple.cgl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    payload["artifacts"][0]["path"] = "misplaced.cgl"
    payload["artifacts"][0]["generatedHash"] = project_pipeline._source_hash(
        misplaced_output
    )
    source_map = payload["artifacts"][0]["sourceMap"]
    source_map["generated"]["file"] = "misplaced.cgl"
    source_map["mappings"][0]["generated"]["file"] = "misplaced.cgl"
    report_path = repo / "out" / "misplaced-artifact-output-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].path must be under project.outputDir" in diagnostic["message"]


def test_validate_project_report_rejects_artifact_paths_outside_target_dir(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    misplaced_output = repo / "out" / "other-target" / "simple.cgl"
    misplaced_output.parent.mkdir(parents=True)
    misplaced_output.write_text(
        (repo / "out" / "cgl" / "simple.cgl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    payload["artifacts"][0]["path"] = "out/other-target/simple.cgl"
    payload["artifacts"][0]["generatedHash"] = project_pipeline._source_hash(
        misplaced_output
    )
    source_map = payload["artifacts"][0]["sourceMap"]
    source_map["generated"]["file"] = "out/other-target/simple.cgl"
    source_map["mappings"][0]["generated"]["file"] = "out/other-target/simple.cgl"
    report_path = repo / "out" / "misplaced-artifact-target-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    expected = (
        "artifacts[0].path must be under " "project.outputDir target/variant directory"
    )
    assert expected in diagnostic["message"]


def test_validate_project_report_rejects_artifact_path_suffix_mismatches(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["opengl"], output_dir="out")
    payload = report.to_json()
    misplaced_output = repo / "out" / "opengl" / "simple.hlsl"
    misplaced_output.write_text(
        (repo / "out" / "opengl" / "simple.glsl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    payload["artifacts"][0]["path"] = "out/opengl/simple.hlsl"
    payload["artifacts"][0]["generatedHash"] = project_pipeline._source_hash(
        misplaced_output
    )
    source_map = payload["artifacts"][0]["sourceMap"]
    source_map["generated"]["file"] = "out/opengl/simple.hlsl"
    source_map["mappings"][0]["generated"]["file"] = "out/opengl/simple.hlsl"
    report_path = repo / "out" / "artifact-suffix-mismatch-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].path suffix must match artifacts[0].target" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_artifact_path_source_layout_mismatches(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "shaders" / "simple.cgl").parent.mkdir()
    (repo / "shaders" / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    misplaced_output = repo / "out" / "cgl" / "renamed.cgl"
    misplaced_output.write_text(
        (repo / "out" / "cgl" / "shaders" / "simple.cgl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    payload["artifacts"][0]["path"] = "out/cgl/renamed.cgl"
    payload["artifacts"][0]["generatedHash"] = project_pipeline._source_hash(
        misplaced_output
    )
    source_map = payload["artifacts"][0]["sourceMap"]
    source_map["generated"]["file"] = "out/cgl/renamed.cgl"
    source_map["mappings"][0]["generated"]["file"] = "out/cgl/renamed.cgl"
    report_path = repo / "out" / "artifact-source-layout-mismatch-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].path must match project.outputDir target/variant "
        "directory plus artifacts[0].source"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_artifacts_with_escaped_output_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = repo / "out" / "opengl" / "simple.glsl"
    output.parent.mkdir(parents=True)
    output.write_text("#version 450\nvoid main() {}\n", encoding="utf-8")
    report_path = repo / "escaped-artifact-output-report.json"
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

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 1}
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].path must be repository-relative" in diagnostic["message"]
    assert diagnostic["missingCapabilities"] == ["artifact.manifest"]


def test_validate_project_report_rejects_artifacts_with_drive_relative_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "drive-relative-artifact-report.json"
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
                        "source": "C:tmp/simple.cgl",
                        "target": "opengl",
                        "path": "C:tmp/simple.glsl",
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
    assert "artifacts[0].source must be repository-relative" in diagnostic["message"]
    assert "artifacts[0].path must be repository-relative" in diagnostic["message"]


def test_validate_project_report_rejects_backslash_report_identity_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "backslash-report-identity-paths.json"
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
                        "id": "src\\simple.cgl",
                        "path": "src\\simple.cgl",
                        "sourceBackend": "cgl",
                        "extension": ".cgl",
                    }
                ],
                "skipped": [
                    {
                        "path": "ignored\\shader.bin",
                        "reason": "unsupported-extension",
                    }
                ],
                "artifacts": [
                    {
                        "source": "src\\simple.cgl",
                        "target": "opengl",
                        "path": "out\\opengl\\src\\simple.glsl",
                        "status": "translated",
                    }
                ],
                "diagnostics": [
                    {
                        "severity": "warning",
                        "code": "project.test",
                        "message": "diagnostic with non-canonical location",
                        "location": _diagnostic_location("src\\simple.cgl"),
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
    assert "units[0].path must be repository-relative" in diagnostic["message"]
    assert "skipped[0].path must be repository-relative" in diagnostic["message"]
    assert "artifacts[0].source must be repository-relative" in diagnostic["message"]
    assert "artifacts[0].path must be repository-relative" in diagnostic["message"]
    assert (
        "diagnostics[0].location.file must be repository-relative"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_duplicate_artifact_identities(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "duplicate-artifact-report.json"
    artifact = {
        "source": "simple.cgl",
        "target": "opengl",
        "path": "out/opengl/simple.glsl",
        "status": "failed",
        "error": "translation failed",
    }
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
                "artifacts": [artifact, dict(artifact)],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[1] duplicates artifacts[0] identity" in diagnostic["message"]


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
                            "mappingGranularity": "block",
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
    assert (
        "artifacts[0].sourceMap.mappingGranularity must be one of "
        "file, line, statement, token"
    ) in diagnostic["message"]
    assert "artifacts[0].sourceMap.target must match artifacts[0].target" in (
        diagnostic["message"]
    )
    assert "artifacts[0].sourceMap.generated must be an object" in (
        diagnostic["message"]
    )
    assert "artifacts[0].sourceMap.mappings must be a list" in diagnostic["message"]


def test_validate_project_report_rejects_source_map_extra_fields(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    source_map = payload["artifacts"][0]["sourceMap"]
    source_map["debug"] = "unexpected"
    source_map["source"]["debug"] = "unexpected"
    source_map["generated"]["debug"] = "unexpected"
    source_map["mappings"][0]["debug"] = "unexpected"
    source_map["mappings"][0]["source"] = dict(source_map["source"])
    source_map["mappings"][0]["generated"] = dict(source_map["generated"])
    report_path = repo / "out" / "source-map-extra-fields-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].sourceMap.debug is not allowed" in diagnostic["message"]
    assert "artifacts[0].sourceMap.source.debug is not allowed" in (
        diagnostic["message"]
    )
    assert "artifacts[0].sourceMap.generated.debug is not allowed" in (
        diagnostic["message"]
    )
    assert "artifacts[0].sourceMap.mappings[0].debug is not allowed" in (
        diagnostic["message"]
    )
    assert "artifacts[0].sourceMap.mappings[0].source.debug is not allowed" in (
        diagnostic["message"]
    )
    assert "artifacts[0].sourceMap.mappings[0].generated.debug is not allowed" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_empty_source_map_mappings(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["artifacts"][0]["sourceMap"]["mappings"] = []
    report_path = repo / "out" / "empty-source-map-mappings-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].sourceMap.mappings must not be empty" in diagnostic["message"]


def test_validate_project_report_rejects_multiple_file_level_source_map_mappings(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    source_map = payload["artifacts"][0]["sourceMap"]
    source_map["mappingGranularity"] = "file"
    source_map["mappings"] = [
        {
            "source": dict(source_map["source"]),
            "generated": dict(source_map["generated"]),
        }
    ]
    mappings = source_map["mappings"]
    mappings.append(dict(mappings[0]))
    report_path = repo / "out" / "multiple-source-map-mappings-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].sourceMap.mappings must contain one file-level mapping"
        in diagnostic["message"]
    )


def test_validate_project_report_accepts_fine_grained_source_map_contract(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["opengl"], output_dir="out")
    payload = report.to_json()
    source_map = payload["artifacts"][0]["sourceMap"]
    source_map["mappingGranularity"] = "line"
    source_map["mappings"] = [
        {
            "source": dict(source_map["source"]),
            "generated": dict(source_map["generated"]),
        },
        {
            "source": dict(source_map["source"]),
            "generated": dict(source_map["generated"]),
        },
    ]
    payload["summary"]["fineGrainedSourceMapCount"] = 1
    payload["summary"]["sourceMapsByGranularity"] = {"line": 1}
    report_path = repo / "out" / "fine-grained-source-map-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is True
    assert validation["validation"]["summary"]["sourceMapStatusCounts"] == (
        _source_map_status_counts(ok=1)
    )
    assert validation["validation"]["summary"]["sourceRemapStatusCounts"] == (
        _source_remap_status_counts(**{"not-recorded": 1})
    )


def test_validate_project_report_rejects_fine_grained_source_map_file_mismatches(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    source_map = payload["artifacts"][0]["sourceMap"]
    source_map["mappingGranularity"] = "statement"
    source_map["mappings"] = [
        {
            "source": dict(source_map["source"], file="other.cgl"),
            "generated": dict(source_map["generated"], file="out/cgl/other.cgl"),
        }
    ]
    payload["summary"]["fineGrainedSourceMapCount"] = 1
    payload["summary"]["sourceMapsByGranularity"] = {"statement": 1}
    report_path = repo / "out" / "fine-grained-source-map-file-mismatch-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].sourceMap.mappings[0].source.file must match "
        "artifacts[0].sourceMap.source.file"
    ) in diagnostic["message"]
    assert (
        "artifacts[0].sourceMap.mappings[0].generated.file must match "
        "artifacts[0].sourceMap.generated.file"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_zero_length_fine_grained_source_map_spans(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    source_map = payload["artifacts"][0]["sourceMap"]
    source_map["mappingGranularity"] = "token"
    source_span = dict(source_map["source"])
    generated_span = dict(source_map["generated"])
    for span in (source_span, generated_span):
        span["length"] = 0
        span["endOffset"] = span["offset"]
        span["endLine"] = span["line"]
        span["endColumn"] = span["column"]
    source_map["mappings"] = [{"source": source_span, "generated": generated_span}]
    payload["summary"]["fineGrainedSourceMapCount"] = 1
    payload["summary"]["sourceMapsByGranularity"] = {"token": 1}
    report_path = repo / "out" / "zero-length-fine-grained-source-map-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].sourceMap.mappings[0].source.length must be greater than zero"
        in diagnostic["message"]
    )
    assert (
        "artifacts[0].sourceMap.mappings[0].generated.length must be greater than zero"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_out_of_anchor_fine_grained_source_map_spans(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    source_map = payload["artifacts"][0]["sourceMap"]
    source_map["mappingGranularity"] = "line"
    source_span = dict(source_map["source"])
    generated_span = dict(source_map["generated"])
    for span in (source_span, generated_span):
        span["offset"] = span["endOffset"] + 1
        span["length"] = 1
        span["endOffset"] = span["offset"] + 1
        span["endLine"] = span["line"]
        span["endColumn"] = span["column"] + 1
    source_map["mappings"] = [{"source": source_span, "generated": generated_span}]
    payload["summary"]["fineGrainedSourceMapCount"] = 1
    payload["summary"]["sourceMapsByGranularity"] = {"line": 1}
    report_path = repo / "out" / "out-of-anchor-source-map-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].sourceMap.mappings[0].source must be within "
        "artifacts[0].sourceMap.source"
    ) in diagnostic["message"]
    assert (
        "artifacts[0].sourceMap.mappings[0].generated must be within "
        "artifacts[0].sourceMap.generated"
    ) in diagnostic["message"]


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


def test_validate_project_report_rejects_inconsistent_source_map_spans(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    source_map = payload["artifacts"][0]["sourceMap"]
    source_map["source"]["endLine"] = 0
    source_map["source"]["endOffset"] -= 1
    source_map["generated"]["endLine"] = source_map["generated"]["line"]
    source_map["generated"]["endColumn"] = 0
    source_map["mappings"][0]["source"] = dict(source_map["source"])
    source_map["mappings"][0]["generated"] = dict(source_map["generated"])
    report_path = repo / "out" / "portability-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].sourceMap.source.endOffset must equal "
        "artifacts[0].sourceMap.source.offset plus length"
    ) in diagnostic["message"]
    assert (
        "artifacts[0].sourceMap.source.endLine must be after or equal to "
        "artifacts[0].sourceMap.source.line"
    ) in diagnostic["message"]
    assert (
        "artifacts[0].sourceMap.generated.endColumn must be greater than or equal to "
        "artifacts[0].sourceMap.generated.column when endLine equals line"
    ) in diagnostic["message"]
    assert (
        "artifacts[0].sourceMap.mappings[0].source.endOffset must equal "
        "artifacts[0].sourceMap.mappings[0].source.offset plus length"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_incomplete_file_level_source_map_spans(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    artifact = payload["artifacts"][0]
    source_map = artifact["sourceMap"]
    source_map["mappingGranularity"] = "file"
    for span in (source_map["source"], source_map["generated"]):
        span["length"] -= 1
        span["endOffset"] -= 1
    source_map["mappings"] = [
        {
            "source": dict(source_map["source"]),
            "generated": dict(source_map["generated"]),
        }
    ]
    source_remap_path = repo / artifact["sourceRemap"]["path"]
    source_remap_path.write_text(
        json.dumps(
            project_pipeline._source_remap_payload(source_map),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact["sourceRemap"]["hash"] = project_pipeline._source_hash(source_remap_path)
    _refresh_artifact_summary(payload)
    report_path = repo / "out" / "incomplete-source-map-span-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 1}
    assert validation["validation"]["summary"]["failedCount"] == 1
    assert validation["validation"]["artifacts"][0]["sourceHashStatus"] == "ok"
    assert validation["validation"]["artifacts"][0]["generatedHashStatus"] == "ok"
    assert validation["validation"]["artifacts"][0]["sourceMapStatus"] == "mismatch"
    assert validation["validation"]["artifacts"][0]["sourceRemapStatus"] == "ok"
    assert validation["validation"]["summary"]["sourceMapStatusCounts"] == (
        _source_map_status_counts(mismatch=1)
    )
    assert validation["validation"]["summary"]["sourceRemapStatusCounts"] == (
        _source_remap_status_counts(ok=1)
    )
    assert validation["diagnosticsByCode"] == {
        "project.validate.source-map-file-span-mismatch": 1,
    }
    assert validation["missingCapabilityCounts"] == {"source.provenance": 1}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["missingCapabilities"] == ["source.provenance"]
    assert "sourceMap.source.length must match source file length" in (
        diagnostic["message"]
    )
    assert "sourceMap.generated.length must match generated artifact length" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_stale_line_preserving_source_map_mappings(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    artifact = payload["artifacts"][0]
    source_map = artifact["sourceMap"]
    assert source_map["mappingGranularity"] == "line"
    assert len(source_map["mappings"]) > 1
    source_map["mappings"].pop()
    report_path = repo / "out" / "stale-line-source-map-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["diagnosticCounts"] == {"note": 0, "warning": 0, "error": 1}
    assert validation["validation"]["summary"]["failedCount"] == 1
    assert validation["validation"]["artifacts"][0]["sourceHashStatus"] == "ok"
    assert validation["validation"]["artifacts"][0]["generatedHashStatus"] == "ok"
    assert validation["validation"]["artifacts"][0]["sourceMapStatus"] == "mismatch"
    assert validation["validation"]["artifacts"][0]["sourceRemapStatus"] == "ok"
    assert validation["diagnosticsByCode"] == {
        "project.validate.source-map-line-span-mismatch": 1,
    }
    assert validation["missingCapabilityCounts"] == {"source.provenance": 1}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["missingCapabilities"] == ["source.provenance"]
    assert (
        "sourceMap.mappings count must match current line-preserving line count"
        in diagnostic["message"]
    )


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
                        "generatedHash": {
                            "algorithm": "sha1",
                            "value": "not-a-hash",
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
    assert "artifacts[0].generatedHash.algorithm must be sha256" in (
        diagnostic["message"]
    )
    assert (
        "artifacts[0].generatedHash.value must be a lowercase 64-character hex digest"
        in diagnostic["message"]
    )
    assert "artifacts[0].provenance.pipeline must be a string" in (
        diagnostic["message"]
    )
    assert "artifacts[0].provenance.intermediate must be a string or null" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_missing_or_forged_artifact_provenance(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["cgl", "opengl"], output_dir="out")
    payload = report.to_json()
    payload["artifacts"][0].pop("provenance")
    payload["artifacts"][1]["provenance"] = {
        "pipeline": "manual-copy",
        "intermediate": "crossgl",
    }
    report_path = repo / "out" / "forged-provenance-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].provenance must be an object" in diagnostic["message"]
    assert (
        "artifacts[1].provenance.pipeline must be one of single-file-translate"
        in diagnostic["message"]
    )
    assert (
        "artifacts[1].provenance.intermediate must match "
        "artifacts[1].sourceBackend and artifacts[1].target"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_current_translated_artifacts_without_hashes(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["artifacts"][0].pop("sourceHash")
    payload["artifacts"][0].pop("generatedHash")
    report_path = repo / "out" / "portability-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].sourceHash must be an object" in diagnostic["message"]
    assert "artifacts[0].generatedHash must be an object" in diagnostic["message"]


def test_validate_project_report_rejects_current_translated_artifacts_without_source_maps(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["artifacts"][0].pop("sourceMap")
    payload["summary"]["sourceMapCount"] = 0
    payload["summary"]["fineGrainedSourceMapCount"] = 0
    payload["summary"]["sourceMapsByGranularity"] = {}
    payload["summary"]["sourceMapsByTarget"] = {}
    payload["summary"]["sourceMapsBySourceBackend"] = {}
    report_path = repo / "out" / "missing-source-map-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].sourceMap must be an object" in diagnostic["message"]


def test_validate_project_report_rejects_current_crossgl_artifacts_without_source_remaps(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["artifacts"][0].pop("sourceRemap")
    payload["summary"]["sourceRemapCount"] = 0
    payload["summary"]["sourceRemapsByTarget"] = {}
    payload["summary"]["sourceRemapsBySourceBackend"] = {}
    report_path = repo / "out" / "missing-source-remap-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].sourceRemap must be an object" in diagnostic["message"]


def test_validate_project_report_rejects_malformed_source_remap_metadata(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["artifacts"][0]["sourceRemap"] = {
        "schemaVersion": 2,
        "path": "out/cgl/wrong.source-remap.json",
        "target": "opengl",
        "generatedFile": "out/cgl/wrong.cgl",
        "mappingGranularity": "line",
        "hash": {"algorithm": "md5", "value": "not-a-sha"},
    }
    report_path = repo / "out" / "malformed-source-remap-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "artifacts[0].sourceRemap.schemaVersion must be 1" in (diagnostic["message"])
    assert "artifacts[0].sourceRemap.path must match artifacts[0].path" in (
        diagnostic["message"]
    )
    assert "artifacts[0].sourceRemap.target must match artifacts[0].target" in (
        diagnostic["message"]
    )
    assert (
        "artifacts[0].sourceRemap.generatedFile must match artifacts[0].path"
        in diagnostic["message"]
    )
    assert "artifacts[0].sourceRemap.mappingGranularity must be file" in (
        diagnostic["message"]
    )
    assert "artifacts[0].sourceRemap.hash.algorithm must be sha256" in (
        diagnostic["message"]
    )
    assert (
        "artifacts[0].sourceRemap.hash.value must be a lowercase "
        "64-character hex digest"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_non_crossgl_source_remap_metadata(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    payload = translate_project(repo, targets=["opengl"], output_dir="out").to_json()
    artifact = payload["artifacts"][0]
    artifact["sourceRemap"] = {
        "schemaVersion": 1,
        "path": "out/opengl/simple.source-remap.json",
        "target": "opengl",
        "generatedFile": artifact["path"],
        "mappingGranularity": "file",
        "hash": {"algorithm": "sha256", "value": "0" * 64},
    }
    _refresh_artifact_summary(payload)
    report_path = repo / "out" / "non-crossgl-source-remap-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].sourceRemap must be omitted unless "
        "artifacts[0].target is CrossGL"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_backslash_source_remap_metadata(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    payload = translate_project(repo, targets=["cgl"], output_dir="out").to_json()
    payload["artifacts"][0]["sourceRemap"][
        "path"
    ] = "out\\cgl\\simple.source-remap.json"
    report_path = repo / "out" / "backslash-source-remap-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "artifacts[0].sourceRemap.path must be repository-relative"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_stale_source_remap_sidecar(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)
    payload = report.to_json()
    source_remap_path = repo / payload["artifacts"][0]["sourceRemap"]["path"]
    source_remap_path.write_text("{}\n", encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"]["summary"]["failedCount"] == 1
    assert validation["validation"]["artifacts"][0]["sourceMapStatus"] == "ok"
    assert validation["validation"]["artifacts"][0]["sourceRemapStatus"] == "invalid"
    assert validation["validation"]["summary"]["sourceMapStatusCounts"] == (
        _source_map_status_counts(ok=1)
    )
    assert validation["validation"]["summary"]["sourceRemapStatusCounts"] == (
        _source_remap_status_counts(invalid=1)
    )
    assert validation["sourceMapStatusCounts"] == _source_map_status_counts(ok=1)
    assert validation["sourceRemapStatusCounts"] == _source_remap_status_counts(
        invalid=1
    )
    assert validation["diagnosticsByCode"] == {
        "project.validate.source-remap-hash-mismatch": 1,
        "project.validate.source-remap-invalid": 1,
        "project.validate.source-remap-mismatch": 1,
    }
    diagnostic = next(
        diagnostic
        for diagnostic in validation["diagnostics"]
        if diagnostic["code"] == "project.validate.source-remap-invalid"
    )
    assert "$.schemaVersion must be 1" in diagnostic["message"]
    assert "$.generatedFile must be a string" in diagnostic["message"]
    assert "$.mappings must be a list" in diagnostic["message"]


def test_validate_project_report_rejects_source_remap_content_mismatches(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    artifact = payload["artifacts"][0]
    source_remap_path = repo / artifact["sourceRemap"]["path"]
    source_remap_payload = json.loads(source_remap_path.read_text(encoding="utf-8"))
    source_remap_payload["mappings"][0]["original"]["file"] = "other.cgl"
    source_remap_path.write_text(
        json.dumps(source_remap_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact["sourceRemap"]["hash"] = project_pipeline._source_hash(source_remap_path)
    report_path = repo / "out" / "source-remap-content-mismatch-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"]["summary"]["failedCount"] == 1
    assert validation["validation"]["artifacts"][0]["sourceMapStatus"] == "ok"
    assert validation["validation"]["artifacts"][0]["sourceRemapStatus"] == "mismatch"
    assert validation["diagnosticsByCode"] == {
        "project.validate.source-remap-mismatch": 1,
    }


def test_validate_project_report_rejects_source_remap_sidecar_extra_fields(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    artifact = payload["artifacts"][0]
    source_remap_path = repo / artifact["sourceRemap"]["path"]
    source_remap_payload = json.loads(source_remap_path.read_text(encoding="utf-8"))
    source_remap_payload["extra"] = True
    source_remap_payload["mappings"][0]["extra"] = True
    source_remap_payload["mappings"][0]["generated"]["extra"] = True
    source_remap_payload["mappings"][0]["original"]["extra"] = True
    source_remap_path.write_text(
        json.dumps(source_remap_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact["sourceRemap"]["hash"] = project_pipeline._source_hash(source_remap_path)
    report_path = repo / "out" / "source-remap-extra-field-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"]["summary"]["failedCount"] == 1
    assert validation["diagnosticsByCode"] == {
        "project.validate.source-remap-invalid": 1,
        "project.validate.source-remap-mismatch": 1,
    }
    diagnostic = next(
        diagnostic
        for diagnostic in validation["diagnostics"]
        if diagnostic["code"] == "project.validate.source-remap-invalid"
    )
    assert "$.extra is not allowed" in diagnostic["message"]
    assert "$.mappings[0].extra is not allowed" in diagnostic["message"]
    assert "$.mappings[0].generated.extra is not allowed" in diagnostic["message"]
    assert "$.mappings[0].original.extra is not allowed" in diagnostic["message"]


def test_validate_project_report_rejects_compiler_incompatible_source_remap_sidecar(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    artifact = payload["artifacts"][0]
    source_map = artifact["sourceMap"]
    source_map["mappingGranularity"] = "file"
    for span in (source_map["source"], source_map["generated"]):
        span["length"] = 0
        span["endOffset"] = span["offset"]
        span["endLine"] = span["line"]
        span["endColumn"] = span["column"]
    source_map["mappings"] = [
        {
            "source": dict(source_map["source"]),
            "generated": dict(source_map["generated"]),
        }
    ]
    source_remap_path = repo / artifact["sourceRemap"]["path"]
    source_remap_path.write_text(
        json.dumps(
            project_pipeline._source_remap_payload(source_map),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact["sourceRemap"]["hash"] = project_pipeline._source_hash(source_remap_path)
    _refresh_artifact_summary(payload)
    report_path = repo / "out" / "compiler-incompatible-source-remap-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"]["summary"]["failedCount"] == 1
    assert validation["diagnosticsByCode"] == {
        "project.validate.source-map-file-span-mismatch": 1,
        "project.validate.source-remap-invalid": 1,
    }
    diagnostic = next(
        diagnostic
        for diagnostic in validation["diagnostics"]
        if diagnostic["code"] == "project.validate.source-remap-invalid"
    )
    assert "Source remap sidecar is not compiler-compatible" in diagnostic["message"]
    assert "$.mappings[0].generated.length must be greater than zero" in (
        diagnostic["message"]
    )
    assert "$.mappings[0].original.length must be greater than zero" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_missing_artifact_provenance_source_backend_rollup(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["summary"].pop("artifactProvenanceIntermediateBySourceBackend")
    report_path = repo / "out" / "missing-provenance-rollup-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert (
        "summary.artifactProvenanceIntermediateBySourceBackend must be an object"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_missing_source_map_variant_rollups(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["summary"].pop("sourceMapsByVariant")
    payload["summary"].pop("sourceRemapsByVariant")
    report_path = repo / "out" / "missing-source-map-variant-rollups-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "summary.sourceMapsByVariant must be an object" in diagnostic["message"]
    assert "summary.sourceRemapsByVariant must be an object" in diagnostic["message"]


def test_validate_project_report_rejects_missing_processing_variant_rollups(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    payload["summary"].pop("defineProcessingByVariant")
    payload["summary"].pop("includePathProcessingByVariant")
    report_path = repo / "out" / "missing-processing-variant-rollups-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "summary.defineProcessingByVariant must be an object" in (
        diagnostic["message"]
    )
    assert "summary.includePathProcessingByVariant must be an object" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_missing_scan_summary_rollups(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    for field_name in (
        "unitsByExtension",
        "unitsBySourceOverride",
        "includeDependencyCount",
        "includeDependenciesByKind",
        "includeDependenciesByStatus",
        "includeDependenciesByResolvedFrom",
        "includeDependenciesBySourceBackend",
        "includeDependenciesBySourceBackendStatus",
        "includeDependenciesByVariant",
        "skippedByExtension",
        "skippedBySourceOverride",
    ):
        payload["summary"].pop(field_name)
    report_path = repo / "out" / "missing-scan-rollups-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "summary.unitsByExtension must be an object" in diagnostic["message"]
    assert "summary.unitsBySourceOverride must be an object" in diagnostic["message"]
    assert "summary.includeDependencyCount must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "summary.includeDependenciesByKind must be an object" in (
        diagnostic["message"]
    )
    assert "summary.includeDependenciesBySourceBackendStatus must be an object" in (
        diagnostic["message"]
    )
    assert "summary.includeDependenciesByVariant must be an object" in (
        diagnostic["message"]
    )
    assert "summary.skippedByExtension must be an object" in diagnostic["message"]
    assert "summary.skippedBySourceOverride must be an object" in (
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
                    "skippedByReason": {"unsupported-extension": 1},
                    "targetCount": 2,
                    "artifactCount": 2,
                    "translatedCount": 0,
                    "failedCount": 1,
                    "diagnosticCounts": {"note": 0, "warning": 0, "error": 0},
                    "diagnosticsByCode": {},
                    "missingCapabilityCounts": {},
                    "unitsBySourceBackend": {"metal": 1},
                    "unitsByExtension": {".metal": 1},
                    "unitsBySourceOverride": {"cgl": 2},
                    "skippedByExtension": {".txt": 1},
                    "skippedBySourceOverride": {"cgl": 1},
                    "artifactsBySourceBackend": {
                        "unknown": {
                            "artifactCount": 1,
                            "translatedCount": 0,
                            "failedCount": 1,
                        }
                    },
                    "artifactsByVariant": {
                        "debug": {
                            "artifactCount": 1,
                            "translatedCount": 1,
                            "failedCount": 0,
                        }
                    },
                    "artifactsByTarget": {
                        "metal": {
                            "artifactCount": 1,
                            "translatedCount": 1,
                            "failedCount": 0,
                        }
                    },
                    "sourceMapCount": 1,
                    "fineGrainedSourceMapCount": 1,
                    "sourceMapsByGranularity": {"line": 1},
                    "sourceMapsByTarget": {"metal": 1},
                    "sourceMapsBySourceBackend": {"unknown": 1},
                    "sourceMapsByVariant": {"debug": 1},
                    "sourceRemapCount": 1,
                    "sourceRemapsByTarget": {"cgl": 1},
                    "sourceRemapsBySourceBackend": {"cgl": 1},
                    "sourceRemapsByVariant": {"debug": 1},
                    "artifactProvenanceByPipeline": {"manual-copy": 1},
                    "artifactProvenanceByIntermediate": {"crossgl": 1},
                    "artifactProvenanceIntermediateBySourceBackend": {
                        "cgl": {"crossgl": 1}
                    },
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
    assert "summary.skippedByReason must match skipped" in diagnostic["message"]
    assert "summary.unitsByExtension must match units" in diagnostic["message"]
    assert "summary.unitsBySourceOverride must match units" in diagnostic["message"]
    assert "summary.skippedByExtension must match skipped" in diagnostic["message"]
    assert "summary.skippedBySourceOverride must match skipped" in (
        diagnostic["message"]
    )
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
    assert "summary.diagnosticsByCode must match diagnostics" in diagnostic["message"]
    assert "summary.missingCapabilityCounts must match diagnostics" in (
        diagnostic["message"]
    )
    assert "summary.unitsBySourceBackend must match units" in diagnostic["message"]
    assert "summary.artifactsBySourceBackend must match artifacts" in (
        diagnostic["message"]
    )
    assert "summary.artifactsByVariant must match artifacts" in diagnostic["message"]
    assert "summary.artifactsByTarget must match artifacts" in diagnostic["message"]
    assert "summary.artifactProvenanceByPipeline must match artifact provenance" in (
        diagnostic["message"]
    )
    assert (
        "summary.artifactProvenanceByIntermediate must match artifact provenance"
        in diagnostic["message"]
    )
    assert (
        "summary.artifactProvenanceIntermediateBySourceBackend must match "
        "artifact provenance"
    ) in diagnostic["message"]
    assert "summary.sourceMapCount must match artifact source maps" in (
        diagnostic["message"]
    )
    assert "summary.fineGrainedSourceMapCount must match artifact source maps" in (
        diagnostic["message"]
    )
    assert "summary.sourceMapsByGranularity must match artifact source maps" in (
        diagnostic["message"]
    )
    assert "summary.sourceMapsByTarget must match artifact source maps" in (
        diagnostic["message"]
    )
    assert "summary.sourceMapsBySourceBackend must match artifact source maps" in (
        diagnostic["message"]
    )
    assert "summary.sourceMapsByVariant must match artifact source maps" in (
        diagnostic["message"]
    )
    assert "summary.sourceRemapCount must match artifact source remaps" in (
        diagnostic["message"]
    )
    assert "summary.sourceRemapsByTarget must match artifact source remaps" in (
        diagnostic["message"]
    )
    assert "summary.sourceRemapsBySourceBackend must match artifact source remaps" in (
        diagnostic["message"]
    )
    assert "summary.sourceRemapsByVariant must match artifact source remaps" in (
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
                    "actionCount": "3",
                    "actionsByKind": [],
                    "actionsBySeverity": {"note": 2},
                    "actionsByTarget": {"opengl": 2},
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
    assert "migration.actionCount must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "migration.actionsByKind must be an object" in diagnostic["message"]
    assert "migration.actionsBySeverity must match migration.actions" in (
        diagnostic["message"]
    )
    assert "migration.actionsByTarget must match migration.actions" in (
        diagnostic["message"]
    )
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


def test_validate_project_report_rejects_migration_actions_with_undeclared_targets(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "undeclared-migration-target-report.json"
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
                    "scope": "shader-kernel-translation",
                    "nonGoals": [
                        "automatic runtime API migration",
                        "application build-system rewrites",
                        "backend framework integration",
                    ],
                    "actions": [
                        {
                            "kind": "manual-runtime-integration",
                            "severity": "note",
                            "message": "Review host runtime integration.",
                            "targets": ["opengl", "metal"],
                        }
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
    assert "migration.actions[0].targets must be listed in project.targets" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_noncanonical_migration_action_targets(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "noncanonical-migration-target-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": ["opengl", "metal"],
                    "outputDir": "out",
                },
                "artifacts": [],
                "migration": {
                    "scope": "shader-kernel-translation",
                    "nonGoals": [
                        "automatic runtime API migration",
                        "application build-system rewrites",
                        "backend framework integration",
                    ],
                    "actions": [
                        {
                            "kind": "manual-runtime-integration",
                            "severity": "note",
                            "message": "Review host runtime integration.",
                            "targets": ["OpenGL", "opengl"],
                        }
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
    assert (
        "migration.actions[0].targets must use normalized backend names "
        "without duplicates"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_empty_migration_action_targets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    payload = translate_project(repo, targets=["cgl"], output_dir="out").to_json()
    actions = [
        {
            "kind": "manual-runtime-integration",
            "severity": "note",
            "message": "Review host runtime integration.",
            "targets": [],
        }
    ]
    payload["migration"].update(project_pipeline._migration_action_rollups(actions))
    payload["migration"]["actions"] = actions
    report_path = repo / "out" / "empty-migration-target-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "migration.actions[0].targets must not be empty" in (diagnostic["message"])


def test_validate_project_report_rejects_migration_actions_without_translated_targets(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    payload = translate_project(
        repo,
        targets=["cgl", "not-a-backend"],
        output_dir="out",
    ).to_json()
    actions = [
        {
            "kind": "manual-runtime-integration",
            "severity": "note",
            "message": "Review host runtime integration.",
            "targets": ["cgl", "not-a-backend"],
        }
    ]
    payload["migration"].update(project_pipeline._migration_action_rollups(actions))
    payload["migration"]["actions"] = actions
    report_path = repo / "out" / "stale-migration-target-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    payload = validate_project_report(report_path, run_toolchains=True)

    diagnostic = next(
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["code"] == "project.validate.invalid-report"
    )
    assert (
        "migration.actions[0].targets must reference translated artifact targets"
        in diagnostic["message"]
    )


def test_validate_project_report_rejects_altered_migration_non_goals(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "altered-migration-non-goals-report.json"
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
                    "scope": "shader-kernel-translation",
                    "nonGoals": ["automatic runtime API migration"],
                    "actions": [],
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "migration.nonGoals must match documented report non-goals" in (
        diagnostic["message"]
    )


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
                        "location": {
                            "file": "",
                            "line": "1",
                            "column": False,
                            "offset": -1,
                            "length": [],
                            "endLine": None,
                            "endColumn": {},
                            "endOffset": "0",
                        },
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
    assert "diagnostics[0].location.line must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "diagnostics[0].location.column must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "diagnostics[0].location.offset must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "diagnostics[0].location.length must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "diagnostics[0].location.endLine must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "diagnostics[0].location.endColumn must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "diagnostics[0].location.endOffset must be a non-negative integer" in (
        diagnostic["message"]
    )
    assert "diagnostics[0].target must be a string" in diagnostic["message"]
    assert "diagnostics[0].missingCapabilities must be a list of strings" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_unexpected_generated_diagnostic_fields(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    payload = scan_project(repo).to_report(targets=["not-a-backend"]).to_json()
    payload["diagnostics"][0]["unexpected"] = "metadata"
    payload["diagnostics"][0]["location"]["unexpected"] = "metadata"
    report_path = repo / "unexpected-generated-diagnostic-fields-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "diagnostics[0].unexpected is not allowed" in diagnostic["message"]
    assert "diagnostics[0].location.unexpected is not allowed" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_inconsistent_diagnostic_location_spans(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "inconsistent-diagnostic-span-report.json"
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
                        "severity": "warning",
                        "code": "project.scan.missing-source-root",
                        "message": "Configured source root does not exist.",
                        "location": {
                            "file": "crosstl.toml",
                            "line": 2,
                            "column": 1,
                            "offset": 10,
                            "length": 3,
                            "endLine": 1,
                            "endColumn": 4,
                            "endOffset": 12,
                        },
                        "missingCapabilities": ["repo.scan"],
                    },
                    {
                        "severity": "note",
                        "code": "project.scan.empty",
                        "message": "No shader sources were discovered.",
                        "location": {
                            "file": "crosstl.toml",
                            "line": 1,
                            "column": 4,
                            "offset": 0,
                            "length": 2,
                            "endLine": 1,
                            "endColumn": 3,
                            "endOffset": 2,
                        },
                        "missingCapabilities": ["repo.scan"],
                    },
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
    assert (
        "diagnostics[0].location.endOffset must equal "
        "diagnostics[0].location.offset plus length"
    ) in diagnostic["message"]
    assert (
        "diagnostics[0].location.endLine must be after or equal to "
        "diagnostics[0].location.line"
    ) in diagnostic["message"]
    assert (
        "diagnostics[1].location.endColumn must be greater than or equal to "
        "diagnostics[1].location.column when endLine equals line"
    ) in diagnostic["message"]


def test_validate_project_report_rejects_diagnostic_locations_outside_project(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["opengl"], output_dir="out")
    payload = report.to_json()
    payload["diagnostics"] = [
        {
            "severity": "warning",
            "code": "project.test.absolute-location",
            "message": "Diagnostic location must be report-relative.",
            "location": _diagnostic_location(str(repo / "simple.cgl")),
            "missingCapabilities": ["repo.scan"],
        },
        {
            "severity": "note",
            "code": "project.test.parent-location",
            "message": "Diagnostic location must not escape the project.",
            "location": _diagnostic_location("../outside.cgl"),
            "missingCapabilities": ["repo.scan"],
        },
    ]
    diagnostic_counts = {"note": 1, "warning": 1, "error": 0}
    payload["diagnosticCounts"] = diagnostic_counts
    payload["summary"]["diagnosticCounts"] = diagnostic_counts
    payload["summary"]["diagnosticsByCode"] = {
        "project.test.absolute-location": 1,
        "project.test.parent-location": 1,
    }
    payload["summary"]["missingCapabilityCounts"] = {"repo.scan": 2}
    report_path = repo / "out" / "invalid-diagnostic-locations-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    assert validation["validation"] == {"toolchains": [], "artifacts": []}
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "diagnostics[0].location.file must be repository-relative" in (
        diagnostic["message"]
    )
    assert "diagnostics[1].location.file must be repository-relative" in (
        diagnostic["message"]
    )


def test_validate_project_report_rejects_diagnostics_with_undeclared_targets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    report = translate_project(repo, targets=["opengl"], output_dir="out")
    payload = report.to_json()
    payload["diagnostics"] = [
        {
            "severity": "error",
            "code": "project.config.unsupported-target",
            "message": "Target is not declared by the report.",
            "location": _diagnostic_location("crosstl.toml"),
            "target": "metal",
            "missingCapabilities": ["target.backend"],
        }
    ]
    report_path = repo / "out" / "portability-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_project_report(report_path)

    assert validation["success"] is False
    diagnostic = validation["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.invalid-report"
    assert "diagnostics[0].target must be listed in project.targets" in (
        diagnostic["message"]
    )


def _write_opengl_toolchain_report(repo, *, variant=None):
    repo.mkdir()
    artifact_segments = ["out", "opengl"]
    if variant is not None:
        artifact_segments.append(variant)
    artifact = repo.joinpath(*artifact_segments, "simple.glsl")
    artifact.parent.mkdir(parents=True)
    artifact.write_text("#version 450\nvoid main() {}\n", encoding="utf-8")
    project = {
        "root": str(repo),
        "targets": ["opengl"],
        "outputDir": "out",
    }
    if variant is not None:
        project["variants"] = {variant: {}}
        project["variantDefineCounts"] = {variant: 0}
    artifact_record = {
        "source": "simple.cgl",
        "sourceBackend": "cgl",
        "target": "opengl",
        "path": artifact.relative_to(repo).as_posix(),
        "status": "translated",
    }
    if variant is not None:
        artifact_record["variant"] = variant
    report_path = repo / "portability-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": project,
                "artifacts": [artifact_record],
            }
        ),
        encoding="utf-8",
    )
    return report_path


def _write_target_toolchain_report(repo, *, target, extension):
    repo.mkdir()
    artifact = repo / "out" / target / f"simple.{extension}"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("void main() {}\n", encoding="utf-8")
    report_path = repo / "portability-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "crosstl-project-portability-report",
                "project": {
                    "root": str(repo),
                    "targets": [target],
                    "outputDir": "out",
                },
                "artifacts": [
                    {
                        "source": "simple.cgl",
                        "sourceBackend": "cgl",
                        "target": target,
                        "path": artifact.relative_to(repo).as_posix(),
                        "status": "translated",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return report_path


def test_validate_project_report_records_toolchain_failures(
    tmp_path, monkeypatch, capsys
):
    report_path = _write_opengl_toolchain_report(tmp_path / "repo")
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
    assert payload["validation"]["toolchainRuns"][0]["sourceBackend"] == "cgl"
    assert payload["validation"]["toolchainRuns"][0]["checkKind"] == "artifact"
    assert payload["validation"]["toolchainRuns"][0]["returncode"] == 2
    assert payload["validation"]["toolchainRuns"][0]["stderr"] == (
        "shader validation failed"
    )
    assert payload["toolchainStatusCounts"] == {
        "available": 1,
        "not-configured": 0,
        "unavailable": 0,
    }
    assert payload["toolchainRunStatusCounts"] == {"failed": 1, "ok": 0}
    assert payload["toolchainRunStatusByTarget"] == {
        "opengl": {"runCount": 1, "okCount": 0, "failedCount": 1}
    }
    assert payload["toolchainRunStatusBySourceBackend"] == {
        "cgl": {"runCount": 1, "okCount": 0, "failedCount": 1}
    }
    assert payload["toolchainRunStatusByVariant"] == {}
    inspection = inspect_project_report(report_path, run_toolchains=True)
    assert inspection["validation"]["artifactCount"] == 1
    assert inspection["validation"]["artifacts"] == [
        {
            "source": "simple.cgl",
            "sourceBackend": "cgl",
            "target": "opengl",
            "path": "out/opengl/simple.glsl",
            "status": "ok",
            "exists": True,
            "sourceHashStatus": "not-recorded",
            "generatedHashStatus": "not-recorded",
            "sourceMapStatus": "not-recorded",
            "sourceRemapStatus": "not-recorded",
        }
    ]
    assert inspection["validation"]["toolchainRunCount"] == 1
    assert inspection["validation"]["truncatedToolchainRunCount"] == 0
    assert inspection["validation"]["toolchainRuns"] == [
        {
            "source": "simple.cgl",
            "sourceBackend": "cgl",
            "target": "opengl",
            "path": "out/opengl/simple.glsl",
            "checkKind": "artifact",
            "status": "failed",
            "returncode": 2,
            "command": ["glslangValidator", "--stdin"],
            "stdoutLength": 0,
            "stderrLength": len("shader validation failed"),
        }
    ]
    assert "stdout" not in inspection["validation"]["toolchainRuns"][0]
    assert "stderr" not in inspection["validation"]["toolchainRuns"][0]

    exit_code = crosstl_cli.main(
        [
            "validate-project",
            str(report_path),
            "--run-toolchains",
            "--format",
            "text",
        ]
    )
    assert exit_code == 1
    stdout = capsys.readouterr().out
    assert (
        "Validation toolchain runs by target: opengl=1 run (0 ok, 1 failed)" in stdout
    )
    assert (
        "Validation toolchain runs by source backend: cgl=1 run (0 ok, 1 failed)"
        in stdout
    )
    exit_code = crosstl_cli.main(
        [
            "inspect-report",
            str(report_path),
            "--run-toolchains",
            "--format",
            "text",
        ]
    )
    assert exit_code == 1
    stdout = capsys.readouterr().out
    assert "Validation artifact samples:" in stdout
    assert (
        "- simple.cgl -> opengl at out/opengl/simple.glsl "
        "(status=ok, exists=true, sourceHash=not-recorded, "
        "generatedHash=not-recorded, sourceMap=not-recorded, "
        "sourceRemap=not-recorded)"
    ) in stdout
    assert "Validation toolchain run samples:" in stdout
    assert (
        "- simple.cgl -> opengl at out/opengl/simple.glsl "
        "(sourceBackend=cgl, status=failed, checkKind=artifact, returncode=2, "
        "command=glslangValidator --stdin, stdout=0 chars, stderr=24 chars)"
    ) in stdout


def test_inspect_project_report_omits_invalid_toolchain_run_commands(
    tmp_path, monkeypatch
):
    report_path = _write_opengl_toolchain_report(tmp_path / "repo")
    monkeypatch.setattr(
        project_pipeline,
        "_run_toolchain_smoke",
        lambda *args, **kwargs: [
            {
                "source": "simple.cgl",
                "sourceBackend": "cgl",
                "target": "opengl",
                "path": "out/opengl/simple.glsl",
                "checkKind": "artifact",
                "command": ["glslangValidator", ""],
                "returncode": 0,
                "status": "ok",
                "stdout": "",
                "stderr": "",
            }
        ],
    )

    inspection = inspect_project_report(report_path, run_toolchains=True)

    assert inspection["validation"]["toolchainRunCount"] == 1
    assert inspection["validation"]["toolchainRuns"] == [
        {
            "source": "simple.cgl",
            "sourceBackend": "cgl",
            "target": "opengl",
            "path": "out/opengl/simple.glsl",
            "checkKind": "artifact",
            "status": "ok",
            "returncode": 0,
            "stdoutLength": 0,
            "stderrLength": 0,
        }
    ]


@pytest.mark.parametrize(
    ("target", "extension", "tool", "command"),
    (
        ("directx", "hlsl", "dxc", ["dxc", "-help"]),
        ("metal", "metal", "xcrun", ["xcrun", "metal", "-v"]),
    ),
)
def test_validate_project_report_marks_availability_only_toolchain_runs(
    tmp_path, monkeypatch, target, extension, tool, command
):
    report_path = _write_target_toolchain_report(
        tmp_path / "repo", target=target, extension=extension
    )
    calls = []
    monkeypatch.setattr(
        project_pipeline.shutil,
        "which",
        lambda requested_tool: f"/usr/bin/{tool}" if requested_tool == tool else None,
    )

    def run_toolchain(*args, **kwargs):
        calls.append((args, kwargs))
        assert args[0] == command
        assert kwargs["input"] is None
        return SimpleNamespace(returncode=0, stdout="tool available", stderr="")

    monkeypatch.setattr(project_pipeline.subprocess, "run", run_toolchain)

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is True
    assert calls
    run = payload["validation"]["toolchainRuns"][0]
    assert run["source"] == "simple.cgl"
    assert run["sourceBackend"] == "cgl"
    assert run["target"] == target
    assert run["path"] == f"out/{target}/simple.{extension}"
    assert run["command"] == command
    assert run["checkKind"] == "tool-availability"
    assert run["status"] == "ok"
    assert payload["toolchainRunStatusByTarget"] == {
        target: {"runCount": 1, "okCount": 1, "failedCount": 0}
    }

    inspection = inspect_project_report(report_path, run_toolchains=True)
    assert inspection["validation"]["toolchainRuns"][0]["checkKind"] == (
        "tool-availability"
    )


def test_validate_project_report_records_toolchain_run_variant_rollups(
    tmp_path, monkeypatch
):
    report_path = _write_opengl_toolchain_report(tmp_path / "repo", variant="debug")
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
            returncode=0,
            stdout="validation ok",
            stderr="",
        ),
    )

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is True
    assert payload["validation"]["toolchainRuns"][0]["variant"] == "debug"
    assert payload["validation"]["toolchainRuns"][0]["sourceBackend"] == "cgl"
    assert payload["validation"]["toolchainRuns"][0]["checkKind"] == "artifact"
    assert payload["toolchainRunStatusCounts"] == {"failed": 0, "ok": 1}
    assert payload["toolchainRunStatusByTarget"] == {
        "opengl": {"runCount": 1, "okCount": 1, "failedCount": 0}
    }
    assert payload["toolchainRunStatusBySourceBackend"] == {
        "cgl": {"runCount": 1, "okCount": 1, "failedCount": 0}
    }
    assert payload["toolchainRunStatusByVariant"] == {
        "debug": {"runCount": 1, "okCount": 1, "failedCount": 0}
    }


def test_validate_project_report_records_toolchain_timeouts(tmp_path, monkeypatch):
    report_path = _write_opengl_toolchain_report(tmp_path / "repo")
    monkeypatch.setattr(
        project_pipeline.shutil,
        "which",
        lambda tool: (
            "/usr/bin/glslangValidator" if tool == "glslangValidator" else None
        ),
    )

    def run_with_timeout(*args, **kwargs):
        assert kwargs["timeout"] == project_pipeline.TOOLCHAIN_SMOKE_TIMEOUT_SECONDS
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs["timeout"],
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(project_pipeline.subprocess, "run", run_with_timeout)

    payload = validate_project_report(report_path, run_toolchains=True)

    assert payload["success"] is False
    assert payload["diagnosticCounts"]["error"] == 1
    assert payload["diagnostics"][0]["code"] == "project.validate.toolchain-failed"
    assert "timed out after" in payload["diagnostics"][0]["message"]
    run = payload["validation"]["toolchainRuns"][0]
    assert run["status"] == "failed"
    assert run["sourceBackend"] == "cgl"
    assert run["checkKind"] == "artifact"
    assert run["returncode"] == project_pipeline.TOOLCHAIN_TIMEOUT_RETURNCODE
    assert run["stdout"] == "partial stdout"
    assert run["stderr"] == (
        "partial stderr\n"
        f"Validation toolchain timed out after "
        f"{project_pipeline.TOOLCHAIN_SMOKE_TIMEOUT_SECONDS} seconds."
    )
    assert payload["toolchainRunStatusCounts"] == {"failed": 1, "ok": 0}
    assert payload["toolchainRunStatusByTarget"] == {
        "opengl": {"runCount": 1, "okCount": 0, "failedCount": 1}
    }
    assert payload["toolchainRunStatusBySourceBackend"] == {
        "cgl": {"runCount": 1, "okCount": 0, "failedCount": 1}
    }
    assert payload["toolchainRunStatusByVariant"] == {}


def test_inspect_project_report_summarizes_toolchain_run_failures(
    tmp_path, monkeypatch, capsys
):
    report_path = _write_opengl_toolchain_report(tmp_path / "repo")
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

    inspection = inspect_project_report(report_path, run_toolchains=True)
    assert inspection["success"] is False
    assert inspection["validation"]["toolchainRunStatusCounts"] == {
        "failed": 1,
        "ok": 0,
    }
    assert inspection["validation"]["toolchainRunStatusByTarget"] == {
        "opengl": {"runCount": 1, "okCount": 0, "failedCount": 1}
    }
    assert inspection["validation"]["toolchainRunStatusBySourceBackend"] == {
        "cgl": {"runCount": 1, "okCount": 0, "failedCount": 1}
    }
    assert inspection["validation"]["toolchainRunStatusByVariant"] == {}

    exit_code = crosstl_cli.main(
        [
            "inspect-report",
            str(report_path),
            "--format",
            "text",
            "--run-toolchains",
        ]
    )
    assert exit_code == 1
    stdout = capsys.readouterr().out
    assert "Validation toolchain runs: failed=1" in stdout
    assert (
        "Validation toolchain runs by target: opengl=1 run (0 ok, 1 failed)" in stdout
    )
    assert (
        "Validation toolchain runs by source backend: cgl=1 run (0 ok, 1 failed)"
        in stdout
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
            "sourceHashStatus": "not-recorded",
            "generatedHashStatus": "missing",
            "sourceMapStatus": "not-recorded",
            "sourceRemapStatus": "not-recorded",
        }
    ]
    assert payload["validation"]["summary"] == {
        "artifactCount": 1,
        "okCount": 0,
        "failedCount": 1,
        "sourceHashStatusCounts": _source_hash_status_counts(**{"not-recorded": 1}),
        "generatedHashStatusCounts": _generated_hash_status_counts(missing=1),
        "sourceMapStatusCounts": _source_map_status_counts(**{"not-recorded": 1}),
        "sourceRemapStatusCounts": _source_remap_status_counts(**{"not-recorded": 1}),
    }
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
            "sourceHashStatus": "not-recorded",
            "generatedHashStatus": "not-applicable",
            "sourceMapStatus": "not-applicable",
            "sourceRemapStatus": "not-applicable",
        }
    ]
    assert payload["validation"]["summary"] == {
        "artifactCount": 1,
        "okCount": 0,
        "failedCount": 1,
        "sourceHashStatusCounts": _source_hash_status_counts(**{"not-recorded": 1}),
        "generatedHashStatusCounts": _generated_hash_status_counts(
            **{"not-applicable": 1}
        ),
        "sourceMapStatusCounts": _source_map_status_counts(**{"not-applicable": 1}),
        "sourceRemapStatusCounts": _source_remap_status_counts(**{"not-applicable": 1}),
    }
    assert payload["artifactStatusByTarget"] == {
        "not-a-backend": {"artifactCount": 1, "okCount": 0, "failedCount": 1}
    }
    assert payload["sourceHashStatusCounts"] == _source_hash_status_counts(
        **{"not-recorded": 1}
    )
    assert payload["generatedHashStatusCounts"] == _generated_hash_status_counts(
        **{"not-applicable": 1}
    )
    assert payload["sourceMapStatusCounts"] == _source_map_status_counts(
        **{"not-applicable": 1}
    )
    assert payload["sourceRemapStatusCounts"] == _source_remap_status_counts(
        **{"not-applicable": 1}
    )
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["code"] == "project.validate.failed-artifact"
    assert diagnostic["location"]["file"] == "simple.cgl"
    assert diagnostic["target"] == "not-a-backend"
    assert "unsupported target backend" in diagnostic["message"]


def test_validate_project_report_deduplicates_regenerated_diagnostics(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    report_path = repo / "portability-report.json"
    generated_message = (
        "Artifact translation failed before validation: "
        "out/not-a-backend/simple.out: unsupported target backend"
    )
    distinct_message = (
        "Artifact translation failed before validation: "
        "out/not-a-backend/other.out: unsupported target backend"
    )
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
                "diagnostics": [
                    {
                        "severity": "error",
                        "code": "project.validate.failed-artifact",
                        "message": generated_message,
                        "location": _diagnostic_location("simple.cgl"),
                        "target": "not-a-backend",
                        "missingCapabilities": ["batch.translation"],
                    },
                    {
                        "severity": "error",
                        "code": "project.validate.failed-artifact",
                        "message": distinct_message,
                        "location": _diagnostic_location("other.cgl"),
                        "target": "not-a-backend",
                        "missingCapabilities": ["batch.translation"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_project_report(report_path)

    assert payload["success"] is False
    assert payload["diagnosticsByCode"] == {"project.validate.failed-artifact": 2}
    assert payload["missingCapabilityCounts"] == {"batch.translation": 2}
    assert [diagnostic["message"] for diagnostic in payload["diagnostics"]] == [
        generated_message,
        distinct_message,
    ]


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
                        "path": "out/opengl/simple.glsl",
                        "status": "translated",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    resolve_report_path = project_pipeline._resolve_report_path
    monkeypatch.setattr(
        project_pipeline,
        "_resolve_report_path",
        lambda config, path: (
            outside / "simple.glsl"
            if path == "out/opengl/simple.glsl"
            else resolve_report_path(config, path)
        ),
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
            "sourceHashStatus": "not-recorded",
            "generatedHashStatus": "outside-project",
            "sourceMapStatus": "not-recorded",
            "sourceRemapStatus": "not-recorded",
        }
    ]
    assert payload["validation"]["summary"] == {
        "artifactCount": 1,
        "okCount": 0,
        "failedCount": 1,
        "sourceHashStatusCounts": _source_hash_status_counts(**{"not-recorded": 1}),
        "generatedHashStatusCounts": _generated_hash_status_counts(
            **{"outside-project": 1}
        ),
        "sourceMapStatusCounts": _source_map_status_counts(**{"not-recorded": 1}),
        "sourceRemapStatusCounts": _source_remap_status_counts(**{"not-recorded": 1}),
    }
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
    assert payload["migration"]["actions"][0]["targets"] == ["opengl"]


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


def test_project_cli_translate_project_validate_records_artifact_checks(tmp_path):
    repo = tmp_path / "repo"
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
            "cgl",
            "--output-dir",
            "out",
            "--validate",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)

    assert payload["summary"]["translatedCount"] == 1
    assert payload["validation"]["toolchains"] == [
        {
            "target": "cgl",
            "status": "not-configured",
            "tools": [],
            "message": "No validation toolchain hook is configured for this target.",
        }
    ]
    assert payload["validation"]["artifacts"][0]["status"] == "ok"
    assert payload["validation"]["artifacts"][0]["sourceMapStatus"] == "ok"
    assert payload["validation"]["artifacts"][0]["sourceRemapStatus"] == "ok"
    assert payload["validation"]["summary"]["artifactCount"] == 1
    assert payload["validation"]["summary"]["okCount"] == 1
    assert "toolchainRuns" not in payload["validation"]


def test_project_cli_translate_project_run_toolchains_records_validation(tmp_path):
    repo = tmp_path / "repo"
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
            "cgl",
            "--output-dir",
            "out",
            "--run-toolchains",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)

    assert payload["summary"]["translatedCount"] == 1
    assert payload["validation"]["toolchains"] == [
        {
            "target": "cgl",
            "status": "not-configured",
            "tools": [],
            "message": "No validation toolchain hook is configured for this target.",
        }
    ]
    assert payload["validation"]["artifacts"][0]["status"] == "ok"
    assert payload["validation"]["summary"]["artifactCount"] == 1
    assert payload["validation"]["summary"]["okCount"] == 1
    assert payload["validation"]["toolchainRuns"] == []


def test_project_cli_report_help_has_clean_stderr():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "report",
            "--help",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stderr == ""
    assert "usage:" in result.stdout
    assert "report" in result.stdout


def test_project_cli_scan_resolves_relative_config_path_from_root(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "custom.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["opengl"]
            output_dir = "generated"
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
            "--config",
            "custom.toml",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["project"]["config"] == str(repo / "custom.toml")
    assert payload["project"]["targets"] == ["opengl"]
    assert payload["project"]["outputDir"] == str(repo / "generated")


def test_project_cli_scan_rejects_missing_explicit_config_path(tmp_path):
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
            "--config",
            "missing.toml",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == ""
    assert "Project config not found:" in result.stdout
    assert str(repo.resolve() / "missing.toml") in result.stdout


def test_project_cli_scan_applies_source_backend_overrides(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "gpu"
    shader_dir.mkdir(parents=True)
    (shader_dir / "kernel.shader").write_text(SIMPLE_CROSSL, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "scan",
            str(repo),
            "--source-override",
            "gpu/*.shader=cgl",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)

    assert payload["project"]["sourceOverrides"] == {"gpu/*.shader": "cgl"}
    assert payload["project"]["sourceOverrideCount"] == 1
    assert payload["summary"]["unitCount"] == 1
    assert payload["summary"]["unitsBySourceOverride"] == {"cgl": 1}
    assert payload["units"][0]["path"] == "gpu/kernel.shader"
    assert payload["units"][0]["sourceBackend"] == "cgl"
    assert payload["units"][0]["sourceOverride"] == "cgl"


def test_project_cli_report_records_include_dir_and_define_overrides(tmp_path):
    repo = tmp_path / "repo"
    output = tmp_path / "portability-report.json"
    include_dir = repo / "includes"
    repo.mkdir()
    include_dir.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "report",
            str(repo),
            "--include-dir",
            "includes",
            "--define",
            "USE_FAST_PATH=1",
            "--output",
            str(output),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result.stdout == f"Wrote {output}\n"
    assert payload["project"]["includeDirs"] == ["includes"]
    assert payload["project"]["includeDirCount"] == 1
    assert payload["project"]["defines"] == {"USE_FAST_PATH": "1"}
    assert payload["project"]["defineCount"] == 1
    assert payload["project"]["includeDirStatus"] == [
        {
            "path": "includes",
            "resolvedPath": str(include_dir.resolve()),
            "status": "active",
            "frontendVisible": True,
        }
    ]
    assert payload["summary"]["unitCount"] == 1


def test_project_cli_report_records_variant_metadata(tmp_path):
    repo = tmp_path / "repo"
    output = tmp_path / "portability-report.json"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]

            [project.defines]
            MODE = "base"

            [project.variants.debug]
            MODE = "debug"

            [project.variants.release]
            MODE = "release"
            ENABLE_FAST_PATH = "1"
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
            "--output",
            str(output),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result.stdout == f"Wrote {output}\n"
    assert payload["project"]["defines"] == {"MODE": "base"}
    assert payload["project"]["defineCount"] == 1
    assert payload["project"]["variants"] == {
        "debug": {"MODE": "debug"},
        "release": {"ENABLE_FAST_PATH": "1", "MODE": "release"},
    }
    assert payload["project"]["variantCount"] == 2
    assert payload["project"]["variantDefineCounts"] == {
        "debug": 1,
        "release": 2,
    }
    assert payload["project"]["selectedVariants"] == []
    assert payload["summary"]["unitCount"] == 1
    assert payload["summary"]["artifactCount"] == 0


def test_project_cli_report_applies_source_backend_overrides(tmp_path):
    repo = tmp_path / "repo"
    output = tmp_path / "portability-report.json"
    shader_dir = repo / "gpu"
    shader_dir.mkdir(parents=True)
    (shader_dir / "kernel.shader").write_text(SIMPLE_CROSSL, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "report",
            str(repo),
            "--source-override",
            "gpu/*.shader=cgl",
            "--output",
            str(output),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result.stdout == f"Wrote {output}\n"
    assert payload["project"]["sourceOverrides"] == {"gpu/*.shader": "cgl"}
    assert payload["project"]["sourceOverrideCount"] == 1
    assert payload["summary"]["unitCount"] == 1
    assert payload["summary"]["unitsBySourceOverride"] == {"cgl": 1}
    assert payload["units"][0]["path"] == "gpu/kernel.shader"
    assert payload["units"][0]["sourceBackend"] == "cgl"
    assert payload["units"][0]["sourceOverride"] == "cgl"


def test_project_cli_scan_applies_source_root_overrides(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    configured_dir = repo / "configured"
    shader_dir.mkdir(parents=True)
    configured_dir.mkdir()
    (shader_dir / "scoped.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (configured_dir / "ignored.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["configured"]
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
            "--source-root",
            "shaders",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)

    assert payload["project"]["sourceRoots"] == ["shaders"]
    assert payload["project"]["sourceRootCount"] == 1
    assert payload["project"]["sourceRootStatus"] == [
        {
            "path": "shaders",
            "resolvedPath": str(shader_dir.resolve()),
            "status": "active",
            "scanVisible": True,
        }
    ]
    assert payload["summary"]["unitCount"] == 1
    assert payload["units"][0]["path"] == "shaders/scoped.cgl"


def test_project_cli_report_records_source_root_overrides(tmp_path):
    repo = tmp_path / "repo"
    output = tmp_path / "portability-report.json"
    shader_dir = repo / "shaders"
    configured_dir = repo / "configured"
    shader_dir.mkdir(parents=True)
    configured_dir.mkdir()
    (shader_dir / "scoped.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (configured_dir / "ignored.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["configured"]
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
            "--source-root",
            "shaders",
            "--output",
            str(output),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result.stdout == f"Wrote {output}\n"
    assert payload["project"]["sourceRoots"] == ["shaders"]
    assert payload["project"]["sourceRootCount"] == 1
    assert payload["summary"]["unitCount"] == 1
    assert payload["units"][0]["path"] == "shaders/scoped.cgl"


def test_project_cli_translate_project_applies_include_dir_and_define_overrides(
    tmp_path,
):
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
            "cgl",
            "--output-dir",
            "translated",
            "--include-dir",
            "includes",
            "--define",
            "USE_PROJECT_SHARED=1",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    output = repo / "translated" / "cgl" / "shaders" / "main.cgl"

    assert payload["project"]["includeDirs"] == ["includes"]
    assert payload["project"]["defines"] == {"USE_PROJECT_SHARED": "1"}
    assert payload["summary"]["translatedCount"] == 1
    assert "project_color" in output.read_text(encoding="utf-8")


def test_project_cli_translate_project_applies_source_backend_overrides(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "gpu"
    shader_dir.mkdir(parents=True)
    (shader_dir / "kernel.shader").write_text(SIMPLE_CROSSL, encoding="utf-8")

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
            "translated",
            "--source-override",
            "gpu/*.shader=cgl",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    output = repo / "translated" / "opengl" / "gpu" / "kernel.glsl"

    assert output.exists()
    assert payload["project"]["sourceOverrides"] == {"gpu/*.shader": "cgl"}
    assert payload["project"]["sourceOverrideCount"] == 1
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["unitsBySourceOverride"] == {"cgl": 1}
    assert payload["units"][0]["sourceOverride"] == "cgl"
    assert payload["artifacts"][0]["sourceBackend"] == "cgl"


def test_project_cli_translate_project_applies_source_root_overrides(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    configured_dir = repo / "configured"
    shader_dir.mkdir(parents=True)
    configured_dir.mkdir()
    (shader_dir / "scoped.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (configured_dir / "ignored.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["configured"]
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
            "cgl",
            "--output-dir",
            "translated",
            "--source-root",
            "shaders",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)

    assert payload["project"]["sourceRoots"] == ["shaders"]
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["unitCount"] == 1
    assert payload["units"][0]["path"] == "shaders/scoped.cgl"
    assert payload["artifacts"][0]["path"] == "translated/cgl/shaders/scoped.cgl"
    assert (repo / "translated" / "cgl" / "shaders" / "scoped.cgl").exists()
    assert not (repo / "translated" / "cgl" / "configured" / "ignored.cgl").exists()


def test_project_cli_translate_project_limits_named_variants_to_selected(tmp_path):
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

            [project.variants.debug]
            MODE = "debug"

            [project.variants.release]
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
            "--variant",
            "debug",
            "--variant",
            "debug",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)

    assert payload["project"]["variants"] == {"debug": {"MODE": "debug"}}
    assert payload["project"]["selectedVariants"] == ["debug"]
    assert payload["summary"]["artifactCount"] == 1
    assert payload["summary"]["artifactsByVariant"] == {
        "debug": {
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
        }
    }
    assert payload["artifacts"][0]["variant"] == "debug"
    assert payload["artifacts"][0]["path"] == "translated/opengl/debug/simple.glsl"
    assert (repo / "translated" / "opengl" / "debug" / "simple.glsl").exists()
    assert not (repo / "translated" / "opengl" / "release" / "simple.glsl").exists()


def test_project_cli_translate_project_rejects_unknown_selected_variant(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]

            [project.variants.debug]
            MODE = "debug"
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
            "--variant",
            "profile",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert (
        "Error: selected project variant is not declared in project config: "
        "profile (available: debug)"
    ) in result.stdout


def test_project_cli_scan_rejects_empty_define_override(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "scan",
            str(repo),
            "--define",
            "=1",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Error: --define entries must use NAME or NAME=VALUE" in result.stdout


def test_project_cli_scan_rejects_empty_source_root_override(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "scan",
            str(repo),
            "--source-root",
            " ",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Error: --source-root entries must be non-empty" in result.stdout


def test_project_cli_scan_rejects_empty_include_dir_override(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "scan",
            str(repo),
            "--include-dir",
            " ",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Error: --include-dir entries must be non-empty" in result.stdout


def test_project_cli_scan_rejects_empty_target_override(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "scan",
            str(repo),
            "--target",
            " ",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Error: --target entries must be non-empty" in result.stdout


def test_project_cli_scan_rejects_malformed_source_backend_override(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "scan",
            str(repo),
            "--source-override",
            "gpu/*.shader=",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Error: --source-override entries must use PATTERN=BACKEND" in result.stdout


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
    report_path = _write_failed_artifact_report(repo)

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
    assert payload["diagnosticsByCode"] == {"project.validate.failed-artifact": 1}
    assert payload["missingCapabilityCounts"] == {"batch.translation": 1}
    assert payload["artifactStatusByTarget"] == {
        "not-a-backend": {"artifactCount": 1, "okCount": 0, "failedCount": 1}
    }
    assert payload["sourceHashStatusCounts"] == _source_hash_status_counts(
        **{"not-recorded": 1}
    )
    assert payload["generatedHashStatusCounts"] == _generated_hash_status_counts(
        **{"not-applicable": 1}
    )
    assert payload["diagnostics"][0]["code"] == "project.validate.failed-artifact"

    text_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "validate-project",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert text_result.returncode == 1
    assert f"Project validation report: {report_path}" in text_result.stdout
    assert "Status: failed" in text_result.stdout
    assert "Diagnostics: 1 errors, 0 warnings, 0 notes" in text_result.stdout
    assert "Diagnostic codes: project.validate.failed-artifact=1" in (
        text_result.stdout
    )
    assert "Missing capabilities: batch.translation=1" in text_result.stdout
    expected_artifact_rollup = (
        "Validation artifacts by target: not-a-backend=1 artifact " "(0 ok, 1 failed)"
    )
    assert expected_artifact_rollup in text_result.stdout
    assert "Validation source hashes: not-recorded=1" in text_result.stdout
    assert "Validation generated hashes: not-applicable=1" in text_result.stdout
    assert "Validation source maps: not-applicable=1" in text_result.stdout
    assert "Validation source remaps: not-applicable=1" in text_result.stdout
    assert "Validation diagnostics:" in text_result.stdout

    sarif_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "validate-project",
            str(report_path),
            "--format",
            "sarif",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    sarif_payload = json.loads(sarif_result.stdout)
    assert sarif_result.returncode == 1
    assert sarif_payload["version"] == "2.1.0"
    run = sarif_payload["runs"][0]
    assert run["invocations"][0]["executionSuccessful"] is False
    assert run["invocations"][0]["properties"]["sourceReport"] == str(report_path)
    assert run["tool"]["driver"]["name"] == "CrossTL project validation"
    assert run["tool"]["driver"]["rules"] == [
        {
            "id": "project.validate.failed-artifact",
            "name": "project.validate.failed-artifact",
        }
    ]
    assert len(run["results"]) == 1
    result = run["results"][0]
    assert result["ruleId"] == "project.validate.failed-artifact"
    assert result["level"] == "error"
    assert result["message"]["text"] == (
        "Artifact translation failed before validation: "
        "out/not-a-backend/simple.out: unsupported target backend"
    )
    assert result["locations"][0]["physicalLocation"] == {
        "artifactLocation": {"uri": "simple.cgl"},
        "region": {
            "endColumn": 1,
            "endLine": 1,
            "startColumn": 1,
            "startLine": 1,
        },
    }
    assert result["properties"] == {
        "target": "not-a-backend",
        "missingCapabilities": ["batch.translation"],
    }


@pytest.mark.parametrize(
    ("format_args", "output_name", "expected_text"),
    (
        ([], "validation.json", '"kind": "crosstl-project-validation-report"'),
        (["--format", "text"], "validation.txt", "Project validation report:"),
        (["--format", "sarif"], "validation.sarif", '"version": "2.1.0"'),
    ),
)
def test_project_cli_validate_project_writes_selected_format_to_output(
    tmp_path, format_args, output_name, expected_text
):
    report_path = _write_failed_artifact_report(tmp_path / "repo")
    output_path = tmp_path / output_name

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "validate-project",
            str(report_path),
            *format_args,
            "--output",
            str(output_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    output_text = output_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert result.stdout == f"Wrote {output_path}\n"
    assert expected_text in output_text
    assert expected_text not in result.stdout


def test_project_cli_validate_project_sarif_reports_generated_diagnostics(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["opengl"], output_dir="out")
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)
    (repo / "out" / "opengl" / "simple.glsl").write_text(
        "void main() {}\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "validate-project",
            str(report_path),
            "--format",
            "sarif",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    run = payload["runs"][0]
    assert run["tool"]["driver"]["name"] == "CrossTL project validation"
    assert run["invocations"][0]["executionSuccessful"] is False
    assert run["invocations"][0]["properties"]["sourceReport"] == str(report_path)
    rules_by_id = {rule["id"]: rule for rule in run["tool"]["driver"]["rules"]}
    assert rules_by_id["project.validate.generated-hash-mismatch"] == {
        "id": "project.validate.generated-hash-mismatch",
        "name": "project.validate.generated-hash-mismatch",
    }
    sarif_result = next(
        result
        for result in run["results"]
        if result.get("ruleId") == "project.validate.generated-hash-mismatch"
    )
    assert sarif_result["ruleId"] == "project.validate.generated-hash-mismatch"
    assert sarif_result["level"] == "error"
    assert sarif_result["message"]["text"] == (
        "Generated artifact hash does not match report: out/opengl/simple.glsl"
    )
    assert sarif_result["locations"][0]["physicalLocation"] == {
        "artifactLocation": {"uri": "simple.cgl"},
        "region": {
            "endColumn": 1,
            "endLine": 1,
            "startColumn": 1,
            "startLine": 1,
        },
    }
    assert sarif_result["properties"] == {
        "target": "opengl",
        "missingCapabilities": ["artifact.manifest"],
    }


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


def test_project_cli_inspect_report_text_marks_invalid_reports(tmp_path):
    report_path = tmp_path / "invalid-report.json"
    report_path.write_text(json.dumps({"kind": "not-a-report"}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Status: failed" in result.stdout
    assert "Report: invalid" in result.stdout
    assert "Validation diagnostic codes: project.validate.invalid-report=1" in (
        result.stdout
    )
    assert "project.validate.invalid-report" in result.stdout

    payload = inspect_project_report(report_path)
    assert payload["report"]["available"] is True
    assert payload["report"]["valid"] is False


def test_inspect_project_report_summarizes_generated_report(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            """).strip(),
        encoding="utf-8",
    )
    report = translate_project(load_project_config(repo), output_dir="out")
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)

    payload = inspect_project_report(report_path)
    source_hash = project_pipeline._source_hash(repo / "simple.cgl")
    source_remap_hash = project_pipeline._source_hash(
        repo / "out" / "cgl" / "simple.source-remap.json"
    )

    assert payload["kind"] == "crosstl-project-report-inspection"
    assert payload["sourceReport"] == str(report_path)
    assert payload["success"] is True
    assert payload["report"]["available"] is True
    assert payload["report"]["valid"] is True
    assert payload["report"]["summary"]["unitCount"] == 1
    assert payload["report"]["summary"]["translatedCount"] == 1
    assert payload["report"]["project"]["config"] == str(repo / "crosstl.toml")
    assert payload["report"]["project"]["targets"] == ["cgl"]
    assert payload["sourceMaps"] == {
        "available": True,
        "sourceMapCount": 1,
        "fileLevelSourceMapCount": 0,
        "fineGrainedSourceMapCount": 1,
        "sourceMapArtifactCount": 1,
        "truncatedSourceMapArtifactCount": 0,
        "sourceMapArtifacts": [
            {
                "source": "simple.cgl",
                "sourceBackend": "cgl",
                "target": "cgl",
                "path": "out/cgl/simple.cgl",
                "mappingGranularity": "line",
                "sourceFile": "simple.cgl",
                "sourceMapTarget": "cgl",
                "generatedFile": "out/cgl/simple.cgl",
                "mappingCount": len(
                    project_pipeline._line_spans(repo / "simple.cgl", "simple.cgl")
                ),
                "sourceHashAlgorithm": source_hash["algorithm"],
                "sourceHash": source_hash["value"],
            }
        ],
        "sourceRemapCount": 1,
        "sourceRemapArtifactCount": 1,
        "truncatedSourceRemapArtifactCount": 0,
        "sourceRemapArtifacts": [
            {
                "source": "simple.cgl",
                "sourceBackend": "cgl",
                "target": "cgl",
                "path": "out/cgl/simple.cgl",
                "sourceRemapPath": "out/cgl/simple.source-remap.json",
                "sourceRemapTarget": "cgl",
                "generatedFile": "out/cgl/simple.cgl",
                "mappingGranularity": "file",
                "sourceHashAlgorithm": source_hash["algorithm"],
                "sourceHash": source_hash["value"],
                "sourceRemapHashAlgorithm": source_remap_hash["algorithm"],
                "sourceRemapHash": source_remap_hash["value"],
            }
        ],
        "sourceMapsByGranularity": {"line": 1},
        "sourceMapsByTarget": {"cgl": 1},
        "sourceMapsBySourceBackend": {"cgl": 1},
        "sourceMapsByVariant": {},
        "sourceRemapsByTarget": {"cgl": 1},
        "sourceRemapsBySourceBackend": {"cgl": 1},
        "sourceRemapsByVariant": {},
    }
    assert payload["artifactProvenance"] == {
        "available": True,
        "byPipeline": {"single-file-translate": 1},
        "byIntermediate": {"none": 1},
        "intermediateBySourceBackend": {"cgl": {"none": 1}},
        "artifactCount": 1,
        "truncatedArtifactCount": 0,
        "artifacts": [
            {
                "source": "simple.cgl",
                "sourceBackend": "cgl",
                "target": "cgl",
                "path": "out/cgl/simple.cgl",
                "pipeline": "single-file-translate",
                "intermediate": "none",
            }
        ],
        "directArtifactCount": 1,
        "truncatedDirectArtifactCount": 0,
        "directArtifacts": [
            {
                "source": "simple.cgl",
                "sourceBackend": "cgl",
                "target": "cgl",
                "path": "out/cgl/simple.cgl",
                "pipeline": "single-file-translate",
                "intermediate": "none",
            }
        ],
        "bridgedArtifactCount": 0,
        "truncatedBridgedArtifactCount": 0,
        "bridgedArtifacts": [],
    }
    assert payload["defineProcessing"] == {
        "available": True,
        "byStatus": {"not-requested": 1},
        "bySourceBackend": {"cgl": {"not-requested": 1}},
        "byVariant": {},
        "artifactCount": 1,
        "truncatedArtifactCount": 0,
        "artifacts": [
            {
                "source": "simple.cgl",
                "sourceBackend": "cgl",
                "target": "cgl",
                "path": "out/cgl/simple.cgl",
                "status": "not-requested",
                "frontend": "lexer",
                "supportsDefines": True,
                "defineCount": 0,
            }
        ],
        "notSupportedArtifactCount": 0,
        "truncatedNotSupportedArtifactCount": 0,
        "notSupportedArtifacts": [],
    }
    assert payload["includeDependencies"] == {
        "available": True,
        "dependencyCount": 0,
        "byStatus": {},
        "byKind": {},
        "byResolvedFrom": {},
        "byVariant": {},
        "bySourceBackend": {},
        "bySourceBackendStatus": {},
        "resolvedDependencyCount": 0,
        "truncatedResolvedDependencyCount": 0,
        "resolvedDependencies": [],
        "unresolvedDependencyCount": 0,
        "truncatedUnresolvedDependencyCount": 0,
        "unresolvedDependencies": [],
    }
    assert payload["includePathProcessing"] == {
        "available": True,
        "byStatus": {"not-requested": 1},
        "bySourceBackend": {"cgl": {"not-requested": 1}},
        "byVariant": {},
        "artifactCount": 1,
        "truncatedArtifactCount": 0,
        "artifacts": [
            {
                "source": "simple.cgl",
                "sourceBackend": "cgl",
                "target": "cgl",
                "path": "out/cgl/simple.cgl",
                "status": "not-requested",
                "frontend": "lexer",
                "supportsIncludePaths": True,
                "includePathCount": 0,
            }
        ],
        "notSupportedArtifactCount": 0,
        "truncatedNotSupportedArtifactCount": 0,
        "notSupportedArtifacts": [],
    }
    assert payload["artifactMatrix"] == {
        "available": True,
        "source": "report",
        "unitCount": 1,
        "targetCount": 1,
        "variantCount": 0,
        "expectedArtifactCount": 1,
        "variantMode": "none",
        "emittedArtifactCount": 1,
        "translatedCount": 1,
        "failedCount": 0,
        "identityCoverageAvailable": True,
        "missingArtifactCount": 0,
        "extraArtifactCount": 0,
        "missingArtifacts": [],
        "extraArtifacts": [],
        "truncatedMissingArtifactCount": 0,
        "truncatedExtraArtifactCount": 0,
        "complete": True,
        "statusByTarget": {
            "cgl": {
                "expectedArtifactCount": 1,
                "emittedArtifactCount": 1,
                "translatedCount": 1,
                "failedCount": 0,
                "missingArtifactCount": 0,
                "extraArtifactCount": 0,
                "complete": True,
            }
        },
        "statusBySourceBackend": {
            "cgl": {
                "expectedArtifactCount": 1,
                "emittedArtifactCount": 1,
                "translatedCount": 1,
                "failedCount": 0,
                "missingArtifactCount": 0,
                "extraArtifactCount": 0,
                "complete": True,
            }
        },
        "statusByVariant": {},
    }
    assert payload["validation"]["success"] is True
    assert payload["validation"]["toolchainStatusCounts"] == {
        "available": 0,
        "not-configured": 1,
        "unavailable": 0,
    }
    assert payload["validation"]["toolchainRunStatusCounts"] == {
        "failed": 0,
        "ok": 0,
    }
    assert payload["validation"]["toolchainRunStatusByTarget"] == {}
    assert payload["validation"]["toolchainRunStatusBySourceBackend"] == {}
    assert payload["validation"]["toolchainRunStatusByVariant"] == {}
    assert payload["validation"][
        "sourceHashStatusCounts"
    ] == _source_hash_status_counts(ok=1)
    assert payload["validation"]["generatedHashStatusCounts"] == (
        _generated_hash_status_counts(ok=1)
    )
    assert payload["validation"]["sourceMapStatusCounts"] == (
        _source_map_status_counts(ok=1)
    )
    assert payload["validation"]["sourceRemapStatusCounts"] == (
        _source_remap_status_counts(ok=1)
    )
    assert payload["validation"]["artifactStatusByTarget"] == {
        "cgl": {
            "artifactCount": 1,
            "okCount": 1,
            "failedCount": 0,
        }
    }
    assert payload["validation"]["artifactStatusBySourceBackend"] == {
        "cgl": {
            "artifactCount": 1,
            "okCount": 1,
            "failedCount": 0,
        }
    }
    assert payload["validation"]["artifactCount"] == 1
    assert payload["validation"]["truncatedArtifactCount"] == 0
    assert payload["validation"]["artifacts"] == [
        {
            "source": "simple.cgl",
            "sourceBackend": "cgl",
            "target": "cgl",
            "path": "out/cgl/simple.cgl",
            "status": "ok",
            "exists": True,
            "sourceHashStatus": "ok",
            "generatedHashStatus": "ok",
            "sourceMapStatus": "ok",
            "sourceRemapStatus": "ok",
        }
    ]
    assert payload["validation"]["toolchainRunCount"] == 0
    assert payload["validation"]["truncatedToolchainRunCount"] == 0
    assert payload["validation"]["toolchainRuns"] == []
    assert payload["validation"]["result"]["summary"] == {
        "artifactCount": 1,
        "okCount": 1,
        "failedCount": 0,
        "sourceHashStatusCounts": _source_hash_status_counts(ok=1),
        "generatedHashStatusCounts": _generated_hash_status_counts(ok=1),
        "sourceMapStatusCounts": _source_map_status_counts(ok=1),
        "sourceRemapStatusCounts": _source_remap_status_counts(ok=1),
    }
    assert payload["failedArtifacts"] == []
    assert payload["migration"]["scope"] == "shader-kernel-translation"
    assert payload["migration"]["nonGoals"] == [
        "automatic runtime API migration",
        "application build-system rewrites",
        "backend framework integration",
    ]
    assert payload["migration"]["actionCount"] == 1
    assert payload["migration"]["actionsByKind"] == {"manual-runtime-integration": 1}
    assert payload["migration"]["actionsBySeverity"] == {"note": 1}
    assert payload["migration"]["actionsByTarget"] == {"cgl": 1}
    assert payload["migration"]["truncatedActionCount"] == 0
    assert payload["migration"]["actions"] == [
        {
            "kind": "manual-runtime-integration",
            "severity": "note",
            "message": (
                "CrossTL translated shader/kernel source artifacts only; review "
                "host runtime API calls, resource binding setup, build scripts, "
                "and backend framework integration separately."
            ),
            "targets": ["cgl"],
        }
    ]


def test_inspect_project_report_emits_closed_inspection_report_schema(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            """).strip(),
        encoding="utf-8",
    )
    report = translate_project(load_project_config(repo), output_dir="out")
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)

    payload = inspect_project_report(report_path)

    assert set(payload) <= project_pipeline.REPORT_INSPECTION_FIELDS
    assert (project_pipeline.REPORT_INSPECTION_FIELDS - {"externalCorpus"}) == set(
        payload
    )
    assert payload["schemaVersion"] == project_pipeline.REPORT_SCHEMA_VERSION
    assert payload["kind"] == project_pipeline.REPORT_INSPECTION_KIND
    assert payload["sourceReport"] == str(report_path)
    assert isinstance(payload["generatedAt"], int)


def test_inspect_project_report_samples_migration_actions(tmp_path):
    report_path = _write_large_migration_report(tmp_path / "repo")

    payload = inspect_project_report(report_path)

    assert payload["success"] is True
    assert payload["migration"]["actionCount"] == 21
    assert payload["migration"]["actionsByKind"] == {"manual-runtime-integration": 21}
    assert payload["migration"]["actionsBySeverity"] == {"note": 21}
    assert payload["migration"]["actionsByTarget"] == {"cgl": 21}
    assert payload["migration"]["truncatedActionCount"] == 1
    assert len(payload["migration"]["actions"]) == 20
    assert payload["migration"]["actions"][0] == {
        "kind": "manual-runtime-integration",
        "severity": "note",
        "message": "Review host integration task 0.",
        "targets": ["cgl"],
    }
    assert payload["migration"]["actions"][-1]["message"] == (
        "Review host integration task 19."
    )


def test_inspect_project_report_detects_count_balanced_artifact_matrix_gaps(tmp_path):
    report_path = _write_count_balanced_artifact_gap_report(tmp_path / "repo")

    payload = inspect_project_report(report_path)

    artifact_matrix = payload["artifactMatrix"]
    assert payload["success"] is False
    assert artifact_matrix["identityCoverageAvailable"] is True
    assert artifact_matrix["source"] == "report"
    assert artifact_matrix["expectedArtifactCount"] == 2
    assert artifact_matrix["emittedArtifactCount"] == 2
    assert artifact_matrix["complete"] is False
    assert artifact_matrix["missingArtifactCount"] == 1
    assert artifact_matrix["extraArtifactCount"] == 1
    assert artifact_matrix["statusBySourceBackend"] == {
        "cgl": {
            "expectedArtifactCount": 2,
            "emittedArtifactCount": 2,
            "translatedCount": 2,
            "failedCount": 0,
            "missingArtifactCount": 1,
            "extraArtifactCount": 1,
            "complete": False,
        }
    }
    assert artifact_matrix["missingArtifacts"] == [
        {
            "source": "second.cgl",
            "target": "cgl",
            "path": "out/cgl/second.cgl",
        }
    ]
    assert artifact_matrix["extraArtifacts"] == [
        {
            "source": "second.cgl",
            "target": "cgl",
            "path": "out/cgl/wrong.cgl",
        }
    ]
    assert artifact_matrix["truncatedMissingArtifactCount"] == 0
    assert artifact_matrix["truncatedExtraArtifactCount"] == 0


def test_inspect_project_report_derives_missing_artifact_matrix_gaps(tmp_path):
    report_path = _write_count_balanced_artifact_gap_report(
        tmp_path / "repo",
        omit_artifact_matrix=True,
    )

    payload = inspect_project_report(report_path)

    artifact_matrix = payload["artifactMatrix"]
    assert payload["success"] is False
    assert artifact_matrix["available"] is True
    assert artifact_matrix["source"] == "derived"
    assert artifact_matrix["identityCoverageAvailable"] is True
    assert artifact_matrix["expectedArtifactCount"] == 2
    assert artifact_matrix["emittedArtifactCount"] == 2
    assert artifact_matrix["complete"] is False
    assert artifact_matrix["missingArtifactCount"] == 1
    assert artifact_matrix["extraArtifactCount"] == 1
    assert artifact_matrix["statusBySourceBackend"] == {
        "cgl": {
            "expectedArtifactCount": 2,
            "emittedArtifactCount": 2,
            "translatedCount": 2,
            "failedCount": 0,
            "missingArtifactCount": 1,
            "extraArtifactCount": 1,
            "complete": False,
        }
    }
    assert artifact_matrix["missingArtifacts"] == [
        {
            "source": "second.cgl",
            "target": "cgl",
            "path": "out/cgl/second.cgl",
        }
    ]
    assert artifact_matrix["extraArtifacts"] == [
        {
            "source": "second.cgl",
            "target": "cgl",
            "path": "out/cgl/wrong.cgl",
        }
    ]


def test_inspect_project_report_derives_malformed_artifact_matrix_gaps(tmp_path):
    report_path = _write_count_balanced_artifact_gap_report(tmp_path / "repo")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["artifactMatrix"] = {
        "unitCount": "2",
        "targetCount": 1,
        "variantCount": 0,
        "variantMode": "invalid",
        "expectedArtifactCount": 2,
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    payload = inspect_project_report(report_path)

    artifact_matrix = payload["artifactMatrix"]
    assert payload["success"] is False
    assert artifact_matrix["available"] is True
    assert artifact_matrix["source"] == "derived"
    assert artifact_matrix["identityCoverageAvailable"] is True
    assert artifact_matrix["missingArtifactCount"] == 1
    assert artifact_matrix["extraArtifactCount"] == 1
    assert artifact_matrix["statusBySourceBackend"] == {
        "cgl": {
            "expectedArtifactCount": 2,
            "emittedArtifactCount": 2,
            "translatedCount": 2,
            "failedCount": 0,
            "missingArtifactCount": 1,
            "extraArtifactCount": 1,
            "complete": False,
        }
    }


def test_inspect_project_report_records_truncation_metadata(tmp_path):
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
                        "source": f"shader-{index}.cgl",
                        "target": "opengl",
                        "path": f"out/opengl/shader-{index}.glsl",
                        "status": "failed",
                        "error": "translation failed",
                    }
                    for index in range(3)
                ],
                "diagnostics": [
                    {
                        "severity": "error",
                        "code": f"project.test.diagnostic-{index}",
                        "message": "Synthetic diagnostic",
                        "location": _diagnostic_location(f"shader-{index}.cgl"),
                    }
                    for index in range(4)
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = inspect_project_report(
        report_path, max_diagnostics=2, max_failed_artifacts=1
    )

    assert payload["diagnosticCount"] >= 7
    assert payload["truncatedDiagnosticCount"] == payload["diagnosticCount"] - 2
    assert len(payload["diagnostics"]) == 2
    assert payload["failedArtifactCount"] == 3
    assert payload["truncatedFailedArtifactCount"] == 2
    assert len(payload["failedArtifacts"]) == 1


def test_inspect_project_report_applies_custom_sample_limits(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    for index in range(4):
        (repo / f"shader-{index}.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    payload = translate_project(repo, targets=["cgl"], output_dir="out").to_json()
    actions = [
        {
            "kind": "manual-runtime-integration",
            "severity": "note",
            "message": f"Review host integration task {index}.",
            "targets": ["cgl"],
        }
        for index in range(5)
    ]
    payload["migration"].update(project_pipeline._migration_action_rollups(actions))
    payload["migration"]["actions"] = actions
    report_path = repo / "out" / "sample-limit-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    toolchain_runs = [
        {
            "source": artifact["source"],
            "sourceBackend": artifact["sourceBackend"],
            "target": artifact["target"],
            "path": artifact["path"],
            "command": ["crosstl-test-validator", str(index)],
            "returncode": 0,
            "status": "ok",
            "stdout": "",
            "stderr": "",
        }
        for index, artifact in enumerate(payload["artifacts"])
    ]
    monkeypatch.setattr(
        project_pipeline,
        "_run_toolchain_smoke",
        lambda *args, **kwargs: toolchain_runs,
    )

    payload = inspect_project_report(
        report_path,
        run_toolchains=True,
        max_source_map_artifacts=2,
        max_artifact_provenance_artifacts=2,
        max_define_processing_artifacts=2,
        max_include_path_processing_artifacts=2,
        max_validation_artifacts=2,
        max_toolchain_runs=3,
        max_migration_actions=4,
    )

    assert payload["sourceMaps"]["sourceMapArtifactCount"] == 4
    assert payload["sourceMaps"]["truncatedSourceMapArtifactCount"] == 2
    assert len(payload["sourceMaps"]["sourceMapArtifacts"]) == 2
    assert payload["sourceMaps"]["sourceRemapArtifactCount"] == 4
    assert payload["sourceMaps"]["truncatedSourceRemapArtifactCount"] == 2
    assert len(payload["sourceMaps"]["sourceRemapArtifacts"]) == 2
    assert payload["artifactProvenance"]["artifactCount"] == 4
    assert payload["artifactProvenance"]["truncatedArtifactCount"] == 2
    assert len(payload["artifactProvenance"]["artifacts"]) == 2
    assert payload["artifactProvenance"]["directArtifactCount"] == 4
    assert payload["artifactProvenance"]["truncatedDirectArtifactCount"] == 2
    assert len(payload["artifactProvenance"]["directArtifacts"]) == 2
    assert payload["artifactProvenance"]["bridgedArtifactCount"] == 0
    assert payload["artifactProvenance"]["truncatedBridgedArtifactCount"] == 0
    assert payload["artifactProvenance"]["bridgedArtifacts"] == []
    assert payload["defineProcessing"]["artifactCount"] == 4
    assert payload["defineProcessing"]["truncatedArtifactCount"] == 2
    assert len(payload["defineProcessing"]["artifacts"]) == 2
    assert payload["includePathProcessing"]["artifactCount"] == 4
    assert payload["includePathProcessing"]["truncatedArtifactCount"] == 2
    assert len(payload["includePathProcessing"]["artifacts"]) == 2
    assert payload["validation"]["artifactCount"] == 4
    assert payload["validation"]["truncatedArtifactCount"] == 2
    assert len(payload["validation"]["artifacts"]) == 2
    assert payload["validation"]["toolchainRunCount"] == 4
    assert payload["validation"]["truncatedToolchainRunCount"] == 1
    assert len(payload["validation"]["toolchainRuns"]) == 3
    assert payload["migration"]["actionCount"] == 5
    assert payload["migration"]["truncatedActionCount"] == 1
    assert len(payload["migration"]["actions"]) == 4


def test_inspect_project_report_applies_artifact_matrix_sample_limit(tmp_path):
    report_path = _write_count_balanced_artifact_gap_report(tmp_path / "repo")

    payload = inspect_project_report(report_path, max_artifact_matrix_artifacts=0)

    artifact_matrix = payload["artifactMatrix"]
    assert artifact_matrix["missingArtifactCount"] == 1
    assert artifact_matrix["truncatedMissingArtifactCount"] == 1
    assert artifact_matrix["missingArtifacts"] == []
    assert artifact_matrix["extraArtifactCount"] == 1
    assert artifact_matrix["truncatedExtraArtifactCount"] == 1
    assert artifact_matrix["extraArtifacts"] == []


def test_inspect_project_report_applies_include_dependency_sample_limit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    includes = "\n".join(f'#include "missing-{index}.inc"' for index in range(4))
    (repo / "main.frag").write_text(
        f"#version 450\n{includes}\nvoid main() {{}}\n",
        encoding="utf-8",
    )
    report = scan_project(repo).to_report(targets=["cgl"])
    report_path = repo / "portability-report.json"
    report.write_json(report_path)

    payload = inspect_project_report(report_path, max_include_dependencies=2)

    include_dependencies = payload["includeDependencies"]
    assert include_dependencies["dependencyCount"] == 4
    assert include_dependencies["unresolvedDependencyCount"] == 4
    assert include_dependencies["truncatedUnresolvedDependencyCount"] == 2
    assert len(include_dependencies["unresolvedDependencies"]) == 2


def test_project_cli_inspect_report_writes_json_summary(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    report_path = repo / "out" / "portability-report.json"
    output = tmp_path / "inspection.json"
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--output",
            str(output),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    source_hash = project_pipeline._source_hash(repo / "simple.cgl")
    source_remap_hash = project_pipeline._source_hash(
        repo / "out" / "cgl" / "simple.source-remap.json"
    )

    assert result.returncode == 0
    assert "Wrote" in result.stdout
    assert payload["success"] is True
    assert payload["report"]["available"] is True
    assert payload["report"]["valid"] is True
    assert payload["report"]["summary"]["artifactCount"] == 1
    assert payload["report"]["summary"]["diagnosticsByCode"] == {}
    assert payload["report"]["summary"]["missingCapabilityCounts"] == {}
    assert payload["report"]["generator"]["pipeline"] == "project-porting"
    assert payload["report"]["project"]["config"] is None
    assert payload["sourceMaps"] == {
        "available": True,
        "sourceMapCount": 1,
        "fileLevelSourceMapCount": 0,
        "fineGrainedSourceMapCount": 1,
        "sourceMapArtifactCount": 1,
        "truncatedSourceMapArtifactCount": 0,
        "sourceMapArtifacts": [
            {
                "source": "simple.cgl",
                "sourceBackend": "cgl",
                "target": "cgl",
                "path": "out/cgl/simple.cgl",
                "mappingGranularity": "line",
                "sourceFile": "simple.cgl",
                "sourceMapTarget": "cgl",
                "generatedFile": "out/cgl/simple.cgl",
                "mappingCount": len(
                    project_pipeline._line_spans(repo / "simple.cgl", "simple.cgl")
                ),
                "sourceHashAlgorithm": source_hash["algorithm"],
                "sourceHash": source_hash["value"],
            }
        ],
        "sourceRemapCount": 1,
        "sourceRemapArtifactCount": 1,
        "truncatedSourceRemapArtifactCount": 0,
        "sourceRemapArtifacts": [
            {
                "source": "simple.cgl",
                "sourceBackend": "cgl",
                "target": "cgl",
                "path": "out/cgl/simple.cgl",
                "sourceRemapPath": "out/cgl/simple.source-remap.json",
                "sourceRemapTarget": "cgl",
                "generatedFile": "out/cgl/simple.cgl",
                "mappingGranularity": "file",
                "sourceHashAlgorithm": source_hash["algorithm"],
                "sourceHash": source_hash["value"],
                "sourceRemapHashAlgorithm": source_remap_hash["algorithm"],
                "sourceRemapHash": source_remap_hash["value"],
            }
        ],
        "sourceMapsByGranularity": {"line": 1},
        "sourceMapsByTarget": {"cgl": 1},
        "sourceMapsBySourceBackend": {"cgl": 1},
        "sourceMapsByVariant": {},
        "sourceRemapsByTarget": {"cgl": 1},
        "sourceRemapsBySourceBackend": {"cgl": 1},
        "sourceRemapsByVariant": {},
    }
    assert payload["artifactProvenance"] == {
        "available": True,
        "byPipeline": {"single-file-translate": 1},
        "byIntermediate": {"none": 1},
        "intermediateBySourceBackend": {"cgl": {"none": 1}},
        "artifactCount": 1,
        "truncatedArtifactCount": 0,
        "artifacts": [
            {
                "source": "simple.cgl",
                "sourceBackend": "cgl",
                "target": "cgl",
                "path": "out/cgl/simple.cgl",
                "pipeline": "single-file-translate",
                "intermediate": "none",
            }
        ],
        "directArtifactCount": 1,
        "truncatedDirectArtifactCount": 0,
        "directArtifacts": [
            {
                "source": "simple.cgl",
                "sourceBackend": "cgl",
                "target": "cgl",
                "path": "out/cgl/simple.cgl",
                "pipeline": "single-file-translate",
                "intermediate": "none",
            }
        ],
        "bridgedArtifactCount": 0,
        "truncatedBridgedArtifactCount": 0,
        "bridgedArtifacts": [],
    }
    assert payload["defineProcessing"] == {
        "available": True,
        "byStatus": {"not-requested": 1},
        "bySourceBackend": {"cgl": {"not-requested": 1}},
        "byVariant": {},
        "artifactCount": 1,
        "truncatedArtifactCount": 0,
        "artifacts": [
            {
                "source": "simple.cgl",
                "sourceBackend": "cgl",
                "target": "cgl",
                "path": "out/cgl/simple.cgl",
                "status": "not-requested",
                "frontend": "lexer",
                "supportsDefines": True,
                "defineCount": 0,
            }
        ],
        "notSupportedArtifactCount": 0,
        "truncatedNotSupportedArtifactCount": 0,
        "notSupportedArtifacts": [],
    }
    assert payload["includePathProcessing"] == {
        "available": True,
        "byStatus": {"not-requested": 1},
        "bySourceBackend": {"cgl": {"not-requested": 1}},
        "byVariant": {},
        "artifactCount": 1,
        "truncatedArtifactCount": 0,
        "artifacts": [
            {
                "source": "simple.cgl",
                "sourceBackend": "cgl",
                "target": "cgl",
                "path": "out/cgl/simple.cgl",
                "status": "not-requested",
                "frontend": "lexer",
                "supportsIncludePaths": True,
                "includePathCount": 0,
            }
        ],
        "notSupportedArtifactCount": 0,
        "truncatedNotSupportedArtifactCount": 0,
        "notSupportedArtifacts": [],
    }
    assert payload["artifactMatrix"]["expectedArtifactCount"] == 1
    assert payload["artifactMatrix"]["emittedArtifactCount"] == 1
    assert payload["artifactMatrix"]["complete"] is True


@pytest.mark.parametrize(
    ("format_name", "output_name", "expected_text"),
    (
        ("text", "inspection.txt", "Project report:"),
        ("sarif", "inspection.sarif", '"version": "2.1.0"'),
    ),
)
def test_project_cli_inspect_report_writes_selected_format_to_output(
    tmp_path, format_name, output_name, expected_text
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = scan_project(repo).to_report(targets=["not-a-backend"])
    report_path = repo / "portability-report.json"
    output_path = tmp_path / output_name
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            format_name,
            "--output",
            str(output_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    output_text = output_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert result.stdout == f"Wrote {output_path}\n"
    assert expected_text in output_text
    assert expected_text not in result.stdout


def test_project_cli_inspect_report_text_includes_migration_actions(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Migration actions by kind: manual-runtime-integration=1" in result.stdout
    assert "Migration actions by severity: note=1" in result.stdout
    assert "Migration actions by target: cgl=1" in result.stdout
    assert "Migration actions:" in result.stdout
    assert "- manual-runtime-integration [targets: cgl]:" in result.stdout
    assert "CrossTL translated shader/kernel source artifacts only" in result.stdout
    assert (
        "review host runtime API calls, resource binding setup, build scripts, "
        "and backend framework integration separately"
    ) in result.stdout


def test_project_cli_inspect_report_text_reports_truncated_migration_actions(tmp_path):
    report_path = _write_large_migration_report(tmp_path / "repo")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Migration actions by kind: manual-runtime-integration=21" in result.stdout
    assert (
        "- manual-runtime-integration [targets: cgl]: "
        "Review host integration task 0."
    ) in (result.stdout)
    assert (
        "- manual-runtime-integration [targets: cgl]: "
        "Review host integration task 19."
    ) in (result.stdout)
    assert "Review host integration task 20." not in result.stdout
    assert "- +1 more" in result.stdout


def test_project_cli_inspect_report_text_includes_project_config_counts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]

            [project.sources]
            "*.cgl" = "cgl"

            [project.defines]
            MODE = "base"

            [project.variants.debug]
            MODE = "debug"

            [project.variants.release]
            MODE = "release"
            """).strip(),
        encoding="utf-8",
    )
    report = translate_project(load_project_config(repo), output_dir="out")
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)
    payload = inspect_project_report(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert f"Config file: {repo / 'crosstl.toml'}" in result.stdout
    assert (
        "Project config: sourceRoots=1, includePatterns=0, excludePatterns="
        f"{len(project_pipeline.DEFAULT_EXCLUDE_PATTERNS)}, sourceOverrides=1, "
        "includeDirs=0, defines=1, variants=2" in result.stdout
    )
    assert payload["report"]["project"]["variantNames"] == ["debug", "release"]
    assert payload["report"]["project"]["variantDefineCounts"] == {
        "debug": 1,
        "release": 1,
    }
    assert "Project variants: debug, release" in result.stdout
    assert "Project variant define counts: debug=1, release=1" in result.stdout
    assert (
        "Artifacts by variant: debug=1 artifact (1 translated, 0 failed), "
        "release=1 artifact (1 translated, 0 failed)"
    ) in result.stdout
    assert "defineNames=MODE" in result.stdout
    assert "MODE=base" not in result.stdout
    assert "MODE=debug" not in result.stdout
    assert "MODE=release" not in result.stdout


def test_project_cli_inspect_report_text_includes_validation_variant_rollups(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            output_dir = "out"

            [project.variants.debug]

            [project.variants.release]
            """).strip(),
        encoding="utf-8",
    )

    def write_variant_output(
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

    monkeypatch.setattr(project_pipeline, "translate", write_variant_output)

    report = translate_project(load_project_config(repo))
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)
    (repo / "out" / "cgl" / "release" / "simple.cgl").write_text(
        "modified release artifact",
        encoding="utf-8",
    )

    validation = validate_project_report(report_path)
    payload = inspect_project_report(report_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    validation_text = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "validate-project",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    expected_rollup = (
        "Validation artifacts by variant: "
        "debug=1 artifact (1 ok, 0 failed), "
        "release=1 artifact (0 ok, 1 failed)"
    )
    expected_payload = {
        "debug": {"artifactCount": 1, "okCount": 1, "failedCount": 0},
        "release": {"artifactCount": 1, "okCount": 0, "failedCount": 1},
    }
    assert validation["artifactStatusByVariant"] == expected_payload
    assert validation["artifactStatusBySourceBackend"] == {
        "cgl": {"artifactCount": 2, "okCount": 1, "failedCount": 1}
    }
    assert payload["validation"]["artifactStatusBySourceBackend"] == {
        "cgl": {"artifactCount": 2, "okCount": 1, "failedCount": 1}
    }
    assert payload["validation"]["artifactStatusByVariant"] == expected_payload
    assert payload["failedArtifacts"] == [
        {
            "source": "simple.cgl",
            "sourceBackend": "cgl",
            "target": "cgl",
            "path": "out/cgl/release/simple.cgl",
            "variant": "release",
            "exists": True,
            "sourceHashStatus": "ok",
            "generatedHashStatus": "mismatch",
            "sourceMapStatus": "not-checked",
            "sourceRemapStatus": "ok",
            "validationStatus": "failed",
        }
    ]
    assert result.returncode == 1
    assert expected_rollup in result.stdout
    assert (
        "Validation artifacts by source backend: " "cgl=2 artifacts (1 ok, 1 failed)"
    ) in result.stdout
    assert (
        "- simple.cgl -> cgl (variant: release) at "
        "out/cgl/release/simple.cgl: validation failed "
        "(generated hash: mismatch; source map: not-checked)"
    ) in result.stdout
    assert validation_text.returncode == 1
    assert (
        "Validation artifacts by source backend: " "cgl=2 artifacts (1 ok, 1 failed)"
    ) in validation_text.stdout
    assert expected_rollup in validation_text.stdout


def test_project_cli_inspect_report_text_includes_include_dir_status(tmp_path):
    repo = tmp_path / "repo"
    include_dir = repo / "includes"
    repo.mkdir()
    include_dir.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            include_dirs = ["includes", "missing-includes"]
            """).strip(),
        encoding="utf-8",
    )
    report = translate_project(load_project_config(repo), output_dir="out")
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Include dirs by status: active=1, missing=1" in result.stdout
    assert (
        "Include dir issues: missing-includes "
        f"(missing; resolved={(repo / 'missing-includes').resolve()}; "
        "frontendVisible=false)"
    ) in result.stdout
    assert "includes (active)" not in result.stdout

    payload = inspect_project_report(report_path)
    assert payload["report"]["project"]["includeDirStatusCounts"] == {
        "active": 1,
        "missing": 1,
    }
    assert payload["report"]["project"]["includeDirStatus"][0]["status"] == "active"
    assert payload["report"]["project"]["includeDirStatus"][1]["status"] == "missing"


def test_project_cli_inspect_report_text_includes_include_dependency_rollups(
    tmp_path,
):
    repo = tmp_path / "repo"
    include_dir = repo / "includes"
    repo.mkdir()
    include_dir.mkdir()
    (include_dir / "shared.inc").write_text("vec4 shared_color();\n", encoding="utf-8")
    (repo / "generated.inc").write_text("vec4 generated_color();\n", encoding="utf-8")
    (repo / "main.frag").write_text(
        textwrap.dedent("""
            #version 450
            #include <shared.inc>
            #include PROJECT_HEADER
            #include "missing.inc"
            void main() {}
            """).strip(),
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            include_dirs = ["includes"]

            [project.defines]
            PROJECT_HEADER = "\\"generated.inc\\""
            """).strip(),
        encoding="utf-8",
    )
    report = scan_project(load_project_config(repo)).to_report(targets=["cgl"])
    report_path = repo / "scan-report.json"
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    unit_hash = project_pipeline._source_hash(repo / "main.frag")
    shared_hash = project_pipeline._source_hash(include_dir / "shared.inc")
    generated_hash = project_pipeline._source_hash(repo / "generated.inc")
    unit_hash_preview = f"{unit_hash['algorithm']}:{unit_hash['value'][:12]}..."
    shared_hash_preview = f"{shared_hash['algorithm']}:{shared_hash['value'][:12]}..."
    generated_hash_preview = (
        f"{generated_hash['algorithm']}:{generated_hash['value'][:12]}..."
    )
    assert "Include dependencies by status: resolved=2, missing=1" in result.stdout
    assert "Include dependencies by kind: local=2, system=1" in result.stdout
    assert "Include dependencies by source backend: opengl=3" in result.stdout
    assert (
        "Include dependencies by source backend status: "
        "opengl=(resolved=2, missing=1)"
    ) in result.stdout
    assert (
        "Include dependencies by resolution source: include-dir=1, source=1"
        in result.stdout
    )
    assert "Resolved include dependencies:" in result.stdout
    assert (
        "- main.frag:2:1 [opengl]: resolved system include shared.inc -> "
        f"includes/shared.inc (include-dir, unitHash={unit_hash_preview}, "
        f"hash={shared_hash_preview})"
    ) in result.stdout
    assert (
        "- main.frag:3:1 [opengl]: resolved local include generated.inc -> "
        f"generated.inc (source, define PROJECT_HEADER, "
        f"unitHash={unit_hash_preview}, hash={generated_hash_preview})"
    ) in result.stdout
    assert "Include dependency issues:" in result.stdout
    assert (
        "- main.frag:4:1 [opengl]: missing local include missing.inc "
        f"(unitHash={unit_hash_preview})"
    ) in result.stdout

    payload = inspect_project_report(report_path)
    assert payload["includeDependencies"] == {
        "available": True,
        "dependencyCount": 3,
        "byStatus": {"missing": 1, "resolved": 2},
        "byKind": {"local": 2, "system": 1},
        "byResolvedFrom": {"include-dir": 1, "source": 1},
        "byVariant": {},
        "bySourceBackend": {"opengl": 3},
        "bySourceBackendStatus": {"opengl": {"missing": 1, "resolved": 2}},
        "resolvedDependencyCount": 2,
        "truncatedResolvedDependencyCount": 0,
        "resolvedDependencies": [
            {
                "source": "main.frag",
                "sourceBackend": "opengl",
                "include": "shared.inc",
                "status": "resolved",
                "kind": "system",
                "line": 2,
                "column": 1,
                "resolvedPath": "includes/shared.inc",
                "resolvedFrom": "include-dir",
                "unitSourceHashAlgorithm": unit_hash["algorithm"],
                "unitSourceHash": unit_hash["value"],
                "resolvedHashAlgorithm": shared_hash["algorithm"],
                "resolvedHash": shared_hash["value"],
            },
            {
                "source": "main.frag",
                "sourceBackend": "opengl",
                "include": "generated.inc",
                "status": "resolved",
                "kind": "local",
                "line": 3,
                "column": 1,
                "resolvedPath": "generated.inc",
                "resolvedFrom": "source",
                "resolvedFromDefine": "PROJECT_HEADER",
                "unitSourceHashAlgorithm": unit_hash["algorithm"],
                "unitSourceHash": unit_hash["value"],
                "resolvedHashAlgorithm": generated_hash["algorithm"],
                "resolvedHash": generated_hash["value"],
            },
        ],
        "unresolvedDependencyCount": 1,
        "truncatedUnresolvedDependencyCount": 0,
        "unresolvedDependencies": [
            {
                "source": "main.frag",
                "sourceBackend": "opengl",
                "include": "missing.inc",
                "status": "missing",
                "kind": "local",
                "line": 4,
                "column": 1,
                "unitSourceHashAlgorithm": unit_hash["algorithm"],
                "unitSourceHash": unit_hash["value"],
            }
        ],
    }


def test_project_cli_inspect_report_text_includes_source_root_status(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "shaders"
    repo.mkdir()
    shader_dir.mkdir()
    (shader_dir / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            targets = ["cgl"]
            source_roots = ["shaders", "missing-shaders"]
            """).strip(),
        encoding="utf-8",
    )
    report = translate_project(load_project_config(repo), output_dir="out")
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Source roots by status: active=1, missing=1" in result.stdout
    assert (
        "Source root issues: missing-shaders "
        f"(missing; resolved={(repo / 'missing-shaders').resolve()}; "
        "scanVisible=false)"
    ) in result.stdout
    assert "shaders (active)" not in result.stdout

    payload = inspect_project_report(report_path)
    assert payload["report"]["project"]["sourceRootStatusCounts"] == {
        "active": 1,
        "missing": 1,
    }
    assert payload["report"]["project"]["sourceRootStatus"][0]["status"] == "active"
    assert payload["report"]["project"]["sourceRootStatus"][1]["status"] == "missing"


def test_project_cli_inspect_report_text_includes_skipped_reason_rollups(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "kernel").write_text("not shader code", encoding="utf-8")
    (repo / "notes.txt").write_text("not shader code", encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            include = ["**/*"]
            exclude = []
            targets = ["cgl"]
            """).strip(),
        encoding="utf-8",
    )
    report = translate_project(load_project_config(repo), output_dir="out")
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Skipped by reason: unsupported-extension=2" in result.stdout
    assert "Skipped by extension: .txt=1, extensionless=1" in result.stdout


def test_project_cli_inspect_report_text_includes_source_override_rollups(tmp_path):
    repo = tmp_path / "repo"
    shader_dir = repo / "gpu"
    shader_dir.mkdir(parents=True)
    (shader_dir / "kernel.shader").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (shader_dir / "unsupported.shader").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["gpu"]
            include = ["**/*"]
            targets = ["cgl"]

            [project.sources]
            "gpu/kernel.shader" = "cgl"
            "gpu/unsupported.shader" = "unknown-backend"
            """).strip(),
        encoding="utf-8",
    )
    report = translate_project(load_project_config(repo), output_dir="out")
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Units by source override: cgl=1" in result.stdout
    assert "Skipped by source override: unknown-backend=1" in result.stdout


def test_project_cli_inspect_report_text_includes_source_map_counts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Source maps: 0 file-level, 1 fine-grained" in result.stdout
    assert "Source remaps: 1" in result.stdout
    assert "Define processing: not-requested=1" in result.stdout
    assert "Define processing by source backend: cgl=(not-requested=1)" in result.stdout
    assert "Define processing artifacts:" in result.stdout
    assert (
        "- simple.cgl -> out/cgl/simple.cgl "
        "(sourceBackend=cgl, target=cgl, status=not-requested, "
        "frontend=lexer, supportsDefines=true, defines=0)"
    ) in result.stdout
    assert "Include path processing: not-requested=1" in result.stdout
    assert (
        "Include path processing by source backend: cgl=(not-requested=1)"
        in result.stdout
    )
    assert "Include path processing artifacts:" in result.stdout
    assert (
        "- simple.cgl -> out/cgl/simple.cgl "
        "(sourceBackend=cgl, target=cgl, status=not-requested, "
        "frontend=lexer, supportsIncludePaths=true, includePaths=0)"
    ) in result.stdout
    assert "Source maps by granularity: line=1" in result.stdout
    assert "Source maps by target: cgl=1" in result.stdout
    assert "Source maps by source backend: cgl=1" in result.stdout
    assert "Source remaps by target: cgl=1" in result.stdout
    assert "Source remaps by source backend: cgl=1" in result.stdout
    assert "Source map artifacts:" in result.stdout
    line_mapping_count = len(
        project_pipeline._line_spans(repo / "simple.cgl", "simple.cgl")
    )
    source_hash = project_pipeline._source_hash(repo / "simple.cgl")
    source_hash_preview = f"{source_hash['algorithm']}:{source_hash['value'][:12]}..."
    assert (
        "- simple.cgl -> out/cgl/simple.cgl "
        f"(sourceBackend=cgl, target=cgl, granularity=line, "
        f"mappings={line_mapping_count}, sourceHash={source_hash_preview})"
    ) in result.stdout
    assert "Source remap artifacts:" in result.stdout
    source_remap_hash = project_pipeline._source_hash(
        repo / "out" / "cgl" / "simple.source-remap.json"
    )
    source_remap_hash_preview = (
        f"{source_remap_hash['algorithm']}:{source_remap_hash['value'][:12]}..."
    )
    assert (
        "- out/cgl/simple.source-remap.json -> out/cgl/simple.cgl "
        f"(sourceBackend=cgl, target=cgl, granularity=file, "
        f"sourceHash={source_hash_preview}, hash={source_remap_hash_preview})"
    ) in result.stdout
    assert "Artifact provenance by pipeline: single-file-translate=1" in result.stdout
    assert "Artifact provenance by intermediate: none=1" in result.stdout
    assert (
        "Artifact provenance by source backend and intermediate: cgl=(none=1)"
        in result.stdout
    )
    assert "Artifact provenance samples:" in result.stdout
    assert (
        "- simple.cgl -> out/cgl/simple.cgl "
        "(sourceBackend=cgl, target=cgl, "
        "pipeline=single-file-translate, intermediate=none)"
    ) in result.stdout
    assert "Validation artifact samples:" in result.stdout
    assert (
        "- simple.cgl -> cgl at out/cgl/simple.cgl "
        "(status=ok, exists=true, sourceHash=ok, generatedHash=ok, "
        "sourceMap=ok, sourceRemap=ok)"
    ) in result.stdout


def test_project_cli_inspect_report_text_includes_source_map_target_mismatches(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    payload = translate_project(repo, targets=["cgl"], output_dir="out").to_json()
    artifact = payload["artifacts"][0]
    artifact["sourceMap"]["target"] = "opengl"
    artifact["sourceRemap"]["target"] = "opengl"
    report_path = repo / "out" / "source-map-target-mismatch-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    line_mapping_count = len(
        project_pipeline._line_spans(repo / "simple.cgl", "simple.cgl")
    )
    assert result.returncode == 1
    assert "Report: invalid" in result.stdout
    source_remap_hash = project_pipeline._source_hash(
        repo / "out" / "cgl" / "simple.source-remap.json"
    )
    source_remap_hash_preview = (
        f"{source_remap_hash['algorithm']}:{source_remap_hash['value'][:12]}..."
    )
    source_hash = project_pipeline._source_hash(repo / "simple.cgl")
    source_hash_preview = f"{source_hash['algorithm']}:{source_hash['value'][:12]}..."
    assert (
        "- simple.cgl -> out/cgl/simple.cgl "
        f"(sourceBackend=cgl, target=cgl, sourceMapTarget=opengl, "
        f"granularity=line, mappings={line_mapping_count}, "
        f"sourceHash={source_hash_preview})"
    ) in result.stdout
    assert (
        "- out/cgl/simple.source-remap.json -> out/cgl/simple.cgl "
        f"(sourceBackend=cgl, target=cgl, sourceRemapTarget=opengl, "
        f"granularity=file, sourceHash={source_hash_preview}, "
        f"hash={source_remap_hash_preview})"
    ) in result.stdout


def test_project_cli_source_map_counts_split_file_and_fine_grained_totals():
    assert (
        crosstl_cli._format_source_map_counts(
            {
                "sourceMapCount": 3,
                "fineGrainedSourceMapCount": 1,
            }
        )
        == "Source maps: 2 file-level, 1 fine-grained"
    )
    assert (
        crosstl_cli._format_source_map_counts(
            {
                "sourceMapCount": 1,
                "fineGrainedSourceMapCount": 2,
            }
        )
        is None
    )


def test_project_cli_inspect_report_text_includes_artifact_matrix(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert (
        "Artifact matrix: 1 emitted of 1 expected "
        "(1 translated, 0 failed, 0 missing, 0 extra; variants=none)"
    ) in result.stdout
    assert (
        "Artifact matrix by target: cgl=1/1 emitted "
        "(1 translated, 0 failed, 0 missing, 0 extra, complete)"
    ) in result.stdout
    assert (
        "Artifact matrix by source backend: cgl=1/1 emitted "
        "(1 translated, 0 failed, 0 missing, 0 extra, complete)"
    ) in result.stdout


def test_project_cli_inspect_report_text_omits_scan_only_artifact_matrix(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = scan_project(repo).to_report(targets=["cgl", "opengl"])
    report_path = repo / "scan-report.json"
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Artifact matrix:" not in result.stdout
    assert "Artifact matrix by target:" not in result.stdout


def test_project_cli_inspect_report_text_reports_artifact_matrix_gaps(tmp_path):
    report_path = _write_count_balanced_artifact_gap_report(tmp_path / "repo")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert (
        "Artifact matrix: 2 emitted of 2 expected "
        "(2 translated, 0 failed, 1 missing, 1 extra; variants=none)"
    ) in result.stdout
    assert (
        "Artifact matrix by target: cgl=2/2 emitted "
        "(2 translated, 0 failed, 1 missing, 1 extra, incomplete)"
    ) in result.stdout
    assert (
        "Artifact matrix by source backend: cgl=2/2 emitted "
        "(2 translated, 0 failed, 1 missing, 1 extra, incomplete)"
    ) in result.stdout
    assert "Artifact matrix missing artifacts:" in result.stdout
    assert "- second.cgl -> cgl at out/cgl/second.cgl" in result.stdout
    assert "Artifact matrix extra artifacts:" in result.stdout
    assert "- second.cgl -> cgl at out/cgl/wrong.cgl" in result.stdout


def test_project_cli_inspect_report_text_derives_missing_artifact_matrix_gaps(
    tmp_path,
):
    report_path = _write_count_balanced_artifact_gap_report(
        tmp_path / "repo",
        omit_artifact_matrix=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert (
        "Artifact matrix: 2 emitted of 2 expected "
        "(2 translated, 0 failed, 1 missing, 1 extra; variants=none)"
    ) in result.stdout
    assert "Artifact matrix missing artifacts:" in result.stdout
    assert "- second.cgl -> cgl at out/cgl/second.cgl" in result.stdout
    assert "Artifact matrix extra artifacts:" in result.stdout
    assert "- second.cgl -> cgl at out/cgl/wrong.cgl" in result.stdout


def test_project_cli_inspect_report_text_includes_report_rollups(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl", "opengl"], output_dir="out")
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Units by source backend: cgl=1" in result.stdout
    assert "Units by extension: .cgl=1" in result.stdout
    assert (
        "Artifacts by source backend: cgl=2 artifacts (2 translated, 0 failed)"
        in result.stdout
    )
    assert (
        "Artifacts by target: cgl=1 artifact (1 translated, 0 failed), "
        "opengl=1 artifact (1 translated, 0 failed)"
    ) in result.stdout


def test_project_cli_inspect_report_text_includes_external_corpus_rollups(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "fixtures").mkdir()
    (repo / "src" / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    (repo / "fixtures" / "undiscovered.cgl").write_text(
        SIMPLE_CROSSL,
        encoding="utf-8",
    )
    (repo / "corpus.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "entries": [
                    {
                        "id": "repo/simple",
                        "path": "src/simple.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                    },
                    {
                        "id": "repo/missing",
                        "path": "missing.hlsl",
                        "sourceBackend": "directx",
                        "targets": ["cgl", "opengl"],
                    },
                    {
                        "id": "repo/undiscovered",
                        "path": "fixtures/undiscovered.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                    },
                    {
                        "id": "repo/outside",
                        "path": "../outside.cgl",
                        "sourceBackend": "cgl",
                        "targets": ["cgl"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["src"]
            targets = ["cgl"]
            external_corpus_manifest = "corpus.json"
            """).strip(),
        encoding="utf-8",
    )
    report = translate_project(load_project_config(repo))
    report_path = repo / "portability-report.json"
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    payload = inspect_project_report(report_path)

    assert payload["externalCorpus"]["missingEntries"] == [
        {
            "id": "repo/missing",
            "path": "missing.hlsl",
            "sourceBackend": "directx",
            "targets": ["cgl", "opengl"],
        }
    ]
    assert payload["externalCorpus"]["undiscoveredPresentEntries"] == [
        {
            "id": "repo/undiscovered",
            "path": "fixtures/undiscovered.cgl",
            "sourceBackend": "cgl",
            "targets": ["cgl"],
        }
    ]
    assert payload["externalCorpus"]["truncatedMissingEntryCount"] == 0
    assert payload["externalCorpus"]["truncatedUndiscoveredPresentEntryCount"] == 0
    assert result.returncode == 0
    assert (
        "External corpus: ok; 3 entries, 2 present, 1 missing, 1 invalid"
        in result.stdout
    )
    assert (
        "External corpus coverage: 1 discovered, 1 present but undiscovered; "
        "4 manifest entries, 3 valid"
    ) in result.stdout
    assert (
        "External corpus missing entries: "
        "missing.hlsl (repo/missing; directx; targets=cgl,opengl)"
    ) in result.stdout
    assert (
        "External corpus undiscovered present entries: "
        "fixtures/undiscovered.cgl (repo/undiscovered; cgl; targets=cgl)"
    ) in result.stdout
    assert "External corpus sources: cgl=2, directx=1" in result.stdout
    assert "External corpus targets: cgl=3, opengl=1" in result.stdout
    assert (
        "External corpus artifacts: cgl=1 artifact (1 translated, 0 failed)"
    ) in result.stdout


def test_project_cli_inspect_report_applies_external_corpus_sample_limit(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "fixtures").mkdir()
    (repo / "src" / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    for index in range(3):
        (repo / "fixtures" / f"undiscovered-{index}.cgl").write_text(
            SIMPLE_CROSSL,
            encoding="utf-8",
        )
    corpus_entries = [
        {
            "id": "repo/simple",
            "path": "src/simple.cgl",
            "sourceBackend": "cgl",
            "targets": ["cgl"],
        },
        *[
            {
                "id": f"repo/missing-{index}",
                "path": f"missing-{index}.hlsl",
                "sourceBackend": "directx",
                "targets": ["cgl"],
            }
            for index in range(3)
        ],
        *[
            {
                "id": f"repo/undiscovered-{index}",
                "path": f"fixtures/undiscovered-{index}.cgl",
                "sourceBackend": "cgl",
                "targets": ["cgl"],
            }
            for index in range(3)
        ],
    ]
    (repo / "corpus.json").write_text(
        json.dumps({"schemaVersion": 1, "entries": corpus_entries}),
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["src"]
            targets = ["cgl"]
            external_corpus_manifest = "corpus.json"
            """).strip(),
        encoding="utf-8",
    )
    report = translate_project(load_project_config(repo))
    report_path = repo / "portability-report.json"
    report.write_json(report_path)

    payload = inspect_project_report(report_path, max_external_corpus_entries=2)

    assert payload["externalCorpus"]["missingEntryCount"] == 3
    assert payload["externalCorpus"]["truncatedMissingEntryCount"] == 1
    assert [entry["id"] for entry in payload["externalCorpus"]["missingEntries"]] == [
        "repo/missing-0",
        "repo/missing-1",
    ]
    assert payload["externalCorpus"]["undiscoveredPresentEntryCount"] == 3
    assert payload["externalCorpus"]["truncatedUndiscoveredPresentEntryCount"] == 1
    assert [
        entry["id"] for entry in payload["externalCorpus"]["undiscoveredPresentEntries"]
    ] == [
        "repo/undiscovered-0",
        "repo/undiscovered-1",
    ]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--max-external-corpus-entries",
            "1",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    cli_payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert len(cli_payload["externalCorpus"]["missingEntries"]) == 1
    assert cli_payload["externalCorpus"]["truncatedMissingEntryCount"] == 2
    assert len(cli_payload["externalCorpus"]["undiscoveredPresentEntries"]) == 1
    assert cli_payload["externalCorpus"]["truncatedUndiscoveredPresentEntryCount"] == 2


def test_project_cli_inspect_report_text_includes_validation_hash_rollups(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = translate_project(repo, targets=["cgl"], output_dir="out")
    report_path = repo / "out" / "portability-report.json"
    report.write_json(report_path)
    (repo / "out" / "cgl" / "simple.cgl").write_text(
        "shader main() {}\n",
        encoding="utf-8",
    )

    payload = inspect_project_report(report_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert payload["failedArtifacts"] == [
        {
            "source": "simple.cgl",
            "sourceBackend": "cgl",
            "target": "cgl",
            "path": "out/cgl/simple.cgl",
            "exists": True,
            "sourceHashStatus": "ok",
            "generatedHashStatus": "mismatch",
            "sourceMapStatus": "not-checked",
            "sourceRemapStatus": "ok",
            "validationStatus": "failed",
        }
    ]
    assert payload["validation"]["artifactStatusByTarget"] == {
        "cgl": {
            "artifactCount": 1,
            "okCount": 0,
            "failedCount": 1,
        }
    }
    assert payload["validation"]["artifactStatusBySourceBackend"] == {
        "cgl": {
            "artifactCount": 1,
            "okCount": 0,
            "failedCount": 1,
        }
    }
    assert payload["validation"]["diagnosticsByCode"] == {
        "project.validate.generated-hash-mismatch": 1
    }
    assert payload["validation"]["missingCapabilityCounts"] == {"artifact.manifest": 1}
    assert result.returncode == 1
    assert "Validation toolchains: not-configured=1" in result.stdout
    assert "Validation artifacts: 0 ok, 1 failed" in result.stdout
    assert (
        "Validation artifacts by target: cgl=1 artifact (0 ok, 1 failed)"
        in result.stdout
    )
    assert (
        "Validation artifacts by source backend: cgl=1 artifact (0 ok, 1 failed)"
        in result.stdout
    )
    assert (
        "Validation diagnostic codes: project.validate.generated-hash-mismatch=1"
        in result.stdout
    )
    assert "Validation missing capabilities: artifact.manifest=1" in result.stdout
    assert "Validation source hashes: ok=1" in result.stdout
    assert "Validation generated hashes: mismatch=1" in result.stdout
    assert "Validation source maps: not-checked=1" in result.stdout
    assert "Validation source remaps: ok=1" in result.stdout
    assert (
        "- simple.cgl -> cgl at out/cgl/simple.cgl: "
        "validation failed (generated hash: mismatch; source map: not-checked)"
    ) in result.stdout
    assert "project.validate.generated-hash-mismatch" in result.stdout


def test_project_cli_inspect_report_text_fails_on_error_diagnostics(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = scan_project(repo).to_report(targets=["not-a-backend"])
    report_path = repo / "portability-report.json"
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Status: failed" in result.stdout
    assert "Targets: not-a-backend" in result.stdout
    assert "Diagnostic codes: project.config.unsupported-target=1" in result.stdout
    assert "Missing capabilities: target.backend=1" in result.stdout
    assert "Validation diagnostics: 1 errors" in result.stdout
    assert (
        "Validation diagnostic codes: project.config.unsupported-target=1"
        in result.stdout
    )
    assert "Validation missing capabilities: target.backend=1" in result.stdout
    assert "Validation artifacts: 0 ok, 0 failed" in result.stdout
    assert "project.config.unsupported-target" in result.stdout
    assert "location=.:1:1" in result.stdout
    assert "target=not-a-backend" in result.stdout
    assert "missingCapabilities=target.backend" in result.stdout


def test_project_cli_inspect_report_sarif_reports_diagnostics(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = scan_project(repo).to_report(targets=["not-a-backend"])
    report_path = repo / "portability-report.json"
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "sarif",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["version"] == "2.1.0"
    run = payload["runs"][0]
    assert run["tool"]["driver"]["name"] == "CrossTL project report inspection"
    assert run["invocations"][0]["executionSuccessful"] is False
    assert run["invocations"][0]["properties"]["sourceReport"] == str(report_path)
    assert run["tool"]["driver"]["rules"] == [
        {
            "id": "project.config.unsupported-target",
            "name": "project.config.unsupported-target",
        }
    ]
    assert len(run["results"]) == 1
    sarif_result = run["results"][0]
    assert sarif_result["ruleId"] == "project.config.unsupported-target"
    assert sarif_result["level"] == "error"
    assert "Target backend 'not-a-backend' is not supported" in (
        sarif_result["message"]["text"]
    )
    assert sarif_result["locations"][0]["physicalLocation"] == {
        "artifactLocation": {"uri": "."},
        "region": {
            "endColumn": 1,
            "endLine": 1,
            "startColumn": 1,
            "startLine": 1,
        },
    }
    assert sarif_result["properties"] == {
        "target": "not-a-backend",
        "missingCapabilities": ["target.backend"],
    }


def test_project_cli_inspect_report_text_reports_truncated_sections(tmp_path):
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
                        "source": f"shader-{index}.cgl",
                        "target": "opengl",
                        "path": f"out/opengl/shader-{index}.glsl",
                        "status": "failed",
                        "error": "translation failed",
                    }
                    for index in range(21)
                ],
                "diagnostics": [
                    {
                        "severity": "error",
                        "code": f"project.test.diagnostic-{index}",
                        "message": "Synthetic diagnostic",
                        "location": _diagnostic_location(f"shader-{index}.cgl"),
                    }
                    for index in range(21)
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
            "inspect-report",
            str(report_path),
            "--format",
            "text",
            "--max-failed-artifacts",
            "3",
            "--max-diagnostics",
            "4",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Failed artifacts truncated: showing 3 of 21" in result.stdout
    assert "Diagnostics truncated: showing 4 of " in result.stdout


@pytest.mark.parametrize(
    "limit_flag",
    (
        "--max-diagnostics",
        "--max-failed-artifacts",
        "--max-source-map-artifacts",
        "--max-artifact-matrix-artifacts",
        "--max-artifact-provenance-artifacts",
        "--max-define-processing-artifacts",
        "--max-include-path-processing-artifacts",
        "--max-include-dependencies",
        "--max-validation-artifacts",
        "--max-toolchain-runs",
        "--max-migration-actions",
        "--max-external-corpus-entries",
    ),
)
def test_project_cli_inspect_report_rejects_negative_sample_limits(
    tmp_path,
    limit_flag,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "simple.cgl").write_text(SIMPLE_CROSSL, encoding="utf-8")
    report = scan_project(repo).to_report(targets=["cgl"])
    report_path = repo / "portability-report.json"
    report.write_json(report_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "inspect-report",
            str(report_path),
            "--format",
            "text",
            limit_flag,
            "-1",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "must be a non-negative integer" in result.stderr
    assert limit_flag in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("command_prefix", ([], ["translate"]))
def test_single_file_cli_forwards_frontend_options(
    tmp_path, monkeypatch, capsys, command_prefix
):
    shader = tmp_path / "kernel.shader"
    output = tmp_path / "kernel.glsl"
    include_dir = tmp_path / "include"
    shader.write_text(SIMPLE_CROSSL, encoding="utf-8")
    include_dir.mkdir()
    calls = []

    def fake_translate(
        file_path,
        backend="cgl",
        save_shader=None,
        format_output=True,
        source_backend=None,
        *,
        include_paths=None,
        defines=None,
    ):
        calls.append(
            {
                "file_path": file_path,
                "backend": backend,
                "save_shader": save_shader,
                "format_output": format_output,
                "source_backend": source_backend,
                "include_paths": list(include_paths or []),
                "defines": dict(defines or {}),
            }
        )
        Path(save_shader).write_text("// translated\n", encoding="utf-8")
        return "// translated\n"

    monkeypatch.setattr(crosstl_cli, "translate", fake_translate)

    exit_code = crosstl_cli.main(
        [
            *command_prefix,
            str(shader),
            "--backend",
            "opengl",
            "--output",
            str(output),
            "--no-format",
            "--source-backend",
            "cgl",
            "--include-dir",
            str(include_dir),
            "--define",
            "USE_FAST_PATH",
            "--define",
            "MODE=debug",
        ]
    )

    assert exit_code == 0
    assert "Successfully translated" in capsys.readouterr().out
    assert calls == [
        {
            "file_path": str(shader),
            "backend": "opengl",
            "save_shader": str(output),
            "format_output": False,
            "source_backend": "cgl",
            "include_paths": [str(include_dir)],
            "defines": {"MODE": "debug", "USE_FAST_PATH": "1"},
        }
    ]
    assert output.read_text(encoding="utf-8") == "// translated\n"


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
