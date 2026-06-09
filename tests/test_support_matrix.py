import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

import crosstl.translator.codegen as codegen
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "support_matrix.py"


def load_support_matrix_module():
    spec = importlib.util.spec_from_file_location("support_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _status_descriptions(module):
    return {status: status for status in module.STATUS_ORDER}


def _backend(backend_id, aliases, extension):
    native_backend = {
        "directx": "crosstl/backend/DirectX",
        "opengl": "crosstl/backend/GLSL",
        "metal": "crosstl/backend/Metal",
    }[backend_id]
    codegen_path = {
        "directx": "crosstl/translator/codegen/directx_codegen.py",
        "opengl": "crosstl/translator/codegen/GLSL_codegen.py",
        "metal": "crosstl/translator/codegen/metal_codegen.py",
    }[backend_id]
    test_path = {
        "directx": "tests/test_translator/test_codegen/test_directx_codegen.py",
        "opengl": "tests/test_translator/test_codegen/test_GLSL_codegen.py",
        "metal": "tests/test_translator/test_codegen/test_metal_codegen.py",
    }[backend_id]
    return {
        "id": backend_id,
        "name": backend_id,
        "aliases": aliases,
        "target_extension": extension,
        "translator_codegen": codegen_path,
        "native_backend": native_backend,
        "tests": [test_path],
        "docs": [{"name": "Docs", "url": f"https://example.com/{backend_id}"}],
    }


def _minimal_catalogs(module):
    backends = {
        "backends": [
            _backend("directx", ["hlsl"], ".hlsl"),
            _backend("opengl", ["glsl"], ".glsl"),
            _backend("metal", ["msl"], ".metal"),
        ]
    }
    features = {
        "statuses": _status_descriptions(module),
        "features": [
            {
                "id": "target.codegen",
                "category": "target",
                "name": "Code generation",
                "description": "Emit target code.",
                "support_plan": {
                    "current_gap": "OpenGL code generation still needs coverage.",
                    "next_scope": "Add parser and codegen fixtures for OpenGL.",
                    "completion_criteria": (
                        "Mark supported when OpenGL has matching evidence."
                    ),
                },
                "support": {
                    "directx": {"status": "supported"},
                    "opengl": {"status": "partial", "notes": "Needs audit."},
                },
            }
        ],
    }
    return backends, features


def test_docs_report_writes_support_signals_compatible_schema(tmp_path, monkeypatch):
    module = load_support_matrix_module()
    support_signals = module.load_support_signals_module()
    monkeypatch.setattr(module, "load_support_signals_module", lambda: support_signals)

    def fetch_url(url, timeout):
        text = "Code generation texture gather support reference."
        return {
            "ok": True,
            "status": 200,
            "url": url,
            "final_url": url,
            "content_type": "text/html",
            "content_length": len(text),
            "sha256": "0" * 64,
            "elapsed_ms": 1,
            "text": text,
            "text_extraction": {
                "kind": "html",
                "parser": "html.parser",
                "links": [],
                "text_length": len(text),
            },
        }

    monkeypatch.setattr(support_signals, "fetch_url", fetch_url)
    backends, features = _minimal_catalogs(module)
    output = tmp_path / "backend-docs-report.json"

    result = module.docs_report(
        backends,
        features,
        output,
        timeout=0.1,
        strict=True,
        max_linked_pages=0,
    )

    assert result == 0
    loaded = support_signals.load_docs_report(output)
    assert loaded["generator"] == support_signals.DOCS_REPORT_GENERATOR
    assert loaded["summary"]["total"] == 3
    assert loaded["summary"]["failed"] == 0


def test_support_matrix_generated_artifacts_are_current():
    result = subprocess.run(
        [sys.executable, "tools/support_matrix.py", "check"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "generated artifacts are current" in result.stdout


def test_current_supported_rows_have_evidence():
    module = load_support_matrix_module()
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )

    rows = module.filtered_support_rows(
        matrix,
        statuses=["supported"],
        evidence="missing",
    )

    assert rows == []


def test_project_report_inspection_is_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.report_inspection"]

    assert feature["category"] == "project"
    assert feature["name"] == "Project report inspection"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_writes_json_summary"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_sarif_reports_diagnostics"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_project_report_detects_count_balanced_artifact_matrix_gaps"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_project_report_derives_missing_artifact_matrix_gaps"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_project_report_derives_malformed_artifact_matrix_gaps"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_project_report_emits_closed_inspection_report_schema"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_derives_missing_artifact_matrix_gaps"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_reports_artifact_matrix_gaps"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_project_report_summarizes_scan_only_artifact_matrix"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_scan_only_artifact_matrix"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_marks_source_map_validation_"
            "failures"
        ) in backend_support["evidence"]
        assert "JSON, text, and SARIF summaries" in backend_support["notes"]
        assert (
            "source-map count, source-map artifact samples, source-remap artifact "
            "samples with sidecar hash metadata, failed validation metadata in "
            "JSON and text samples, and provenance rollups"
        ) in backend_support["notes"]
        assert "failed validation metadata in JSON and text samples" in (
            backend_support["notes"]
        )
        assert (
            "report-or-translation-artifact-derived metadata source, does not "
            "derive artifact-matrix gaps for scan-only reports, plus sampled "
            "missing and extra artifact identities"
        ) in backend_support["notes"]
        assert (
            "validation status, diagnostic-code, missing-capability, "
            "toolchain-status, toolchain-run target, source-backend, check-kind, "
            "tool, and variant rollups plus artifact target, source-backend, and "
            "variant rollups" in (backend_support["notes"])
        )
        assert "configurable diagnostic and failed-artifact truncation" in (
            backend_support["notes"]
        )
        assert "failed-artifact variant labels" in backend_support["notes"]
        assert "skipped-reason" in backend_support["notes"]
        assert "source-extension" in backend_support["notes"]
        assert "skipped-extension" in backend_support["notes"]
        assert "sampled skipped-source paths" in backend_support["notes"]
        assert "skipped-source samples with custom limits" in backend_support["notes"]
        assert "invalid-report markers" in backend_support["notes"]
        assert "diagnostic location, target, and missing-capability context" in (
            backend_support["notes"]
        )
        assert "project-config count" in backend_support["notes"]
        assert "variant-name summaries" in backend_support["notes"]
        assert "source-root status" in backend_support["notes"]
        assert "include-directory status rollups" in backend_support["notes"]
        assert (
            "include dependency status, kind, and resolution-source rollups plus "
            "resolved include hash and byte-size metadata"
        ) in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_source_root_status"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_include_dir_status"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_include_dependency_rollups"
        ) in backend_support["evidence"]
        assert "sampled missing or undiscovered entries" in (backend_support["notes"])
        assert "inactive source-root and include-directory record details" in (
            backend_support["notes"]
        )


def test_project_repo_scan_documents_source_root_status():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.repo_scan"]

    for backend_support in feature["support"].values():
        assert "source-root status counts" in backend_support["notes"]
        assert (
            "expected/actual source-root status and resolved-path validation context"
            in backend_support["notes"]
        )
        assert "roots that resolve to non-directory paths" in (backend_support["notes"])
        assert "validates skipped source override provenance" in (
            backend_support["notes"]
        )
        assert "scan/report CLI source-root overrides" in (backend_support["notes"])
        assert "source override rollups" in (backend_support["notes"])
        assert "omit currently scanned units or skipped sources" in (
            backend_support["notes"]
        )
        assert (
            "repository-relative explicit include and source override patterns "
            "from the repository root before source-root visibility filtering"
        ) in backend_support["notes"]
        assert "validation rejects missing scan summary rollups" in (
            backend_support["notes"]
        )
        assert "extensionless unsupported explicit includes" in (
            backend_support["notes"]
        )
        assert (
            "text inspection identifies inactive source roots by path, "
            "resolved path, and scan visibility"
        ) in backend_support["notes"]
        assert "exclude patterns that leave valid units visible" in (
            backend_support["notes"]
        )
        assert "rejects missing explicit config paths" in backend_support["notes"]
        assert "CLI source-root, include-directory, and source-override separators" in (
            backend_support["notes"]
        )
        assert "project configuration scalar, list, and source override metadata" in (
            backend_support["notes"]
        )
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_source_roots_that_are_not_directories"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_accepts_repository_relative_include_patterns"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_applies_source_override_patterns_from_repository_root"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_extensionless_unsupported_sources"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_inconsistent_skipped_source_"
            "overrides"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_stale_source_root_resolved_path"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_scan_summary_rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_source_root_status"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_scan_applies_source_backend_overrides"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_scan_applies_source_root_overrides"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_report_applies_source_backend_overrides"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_report_records_source_root_overrides"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_exclude_patterns_outside_project"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_drive_relative_exclude_patterns"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_config_rejects_missing_explicit_config_path"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_scan_rejects_missing_explicit_config_path"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_scan_normalizes_path_override_separators"
        ) in backend_support["evidence"]


def test_project_report_inspection_documents_rollups():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.report_inspection"]

    for backend_support in feature["support"].values():
        assert (
            "inspection identity, SARIF invocation metadata, and source report "
            "schema/kind metadata" in backend_support["notes"]
        )
        assert "source report generation metadata" in backend_support["notes"]
        assert "variant-name summaries" in backend_support["notes"]
        assert "source-override mappings and rollups" in backend_support["notes"]
        assert "project root, output directory" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_source_map_counts"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_include_dir_status"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_skipped_reason_rollups"
        ) in backend_support["evidence"]
        assert "extensionless skipped-extension rollups" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_source_override_rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_validation_variant_"
            "rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_project_report_summarizes_toolchain_run_failures"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_marks_invalid_reports"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_reports_artifact_matrix_gaps"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_project_report_summarizes_scan_only_artifact_matrix"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_scan_only_artifact_matrix"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_reports_truncated_sections"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_rejects_negative_sample_limits"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_project_report_applies_custom_sample_limits"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_project_report_applies_artifact_matrix_sample_limit"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_project_report_applies_include_dependency_sample_limit"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_project_report_omits_invalid_toolchain_run_commands"
        ) in backend_support["evidence"]
        assert "source-override" in backend_support["notes"]
        assert "skipped source-override" in backend_support["notes"]
        assert "negative sample-limit rejection" in backend_support["notes"]
        assert "closed inspection-report field set" in backend_support["notes"]
        assert "invalid toolchain-run command omission" in backend_support["notes"]
        assert "failure-reason samples for failed toolchain runs" in (
            backend_support["notes"]
        )
        assert (
            "source-map, source-remap, provenance, define-processing, "
            "include-path-processing, include-dependency, validation artifact, "
            "validation toolchain-run, and skipped-source samples with custom limits"
        ) in backend_support["notes"]
        assert (
            "migration scope and non-goal text output, action count and kind, "
            "severity, and target rollups, bounded migration action samples with "
            "custom limits, target lists, and truncation metadata"
        ) in backend_support["notes"]
        assert "does not derive artifact-matrix gaps for scan-only reports" in (
            backend_support["notes"]
        )


def test_project_migration_actions_are_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.migration_actions"]

    assert feature["category"] == "project"
    assert feature["name"] == "Migration action report"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert "noncanonical or duplicate action targets" in backend_support["notes"]
        assert "empty action targets" in backend_support["notes"]
        assert "action targets without translated artifacts" in (
            backend_support["notes"]
        )
        assert "action count and kind, severity, and target rollups" in (
            backend_support["notes"]
        )
        assert (
            "bounded inspection samples with target lists and truncation metadata"
            in (backend_support["notes"])
        )
        assert "missing or altered action rollups" in backend_support["notes"]
        assert "shader/kernel source translation from host runtime APIs" in (
            backend_support["notes"]
        )
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_report_records_documented_migration_actions"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_reports_truncated_migration_"
            "actions"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_noncanonical_migration_action_"
            "targets"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_empty_migration_action_targets"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_migration_actions_without_"
            "translated_targets"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_migration_rollups"
        ) in backend_support["evidence"]


def test_project_include_resolution_documents_status_reporting():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.include_resolution"]

    for backend_support in feature["support"].values():
        assert backend_support["status"] == "partial"
        assert "per-include directory status records and status counts" in (
            backend_support["notes"]
        )
        assert "current include-directory resolved paths" in (backend_support["notes"])
        assert (
            "expected/actual include-directory status and resolved-path "
            "validation context" in backend_support["notes"]
        )
        assert "non-directory paths" in backend_support["notes"]
        assert "only active existing repository-contained resolved paths" in (
            backend_support["notes"]
        )
        assert (
            "applies CLI include-directory overrides with normalized "
            "repository-relative separators and rejects empty CLI "
            "include-directory overrides"
        ) in backend_support["notes"]
        assert "include-path processing status" in backend_support["notes"]
        assert "source frontend support metadata" in backend_support["notes"]
        assert "warning diagnostics and missing-capability rollups" in (
            backend_support["notes"]
        )
        assert (
            "include-path processing rollups by status, target, source backend, "
            "and variant" in backend_support["notes"]
        )
        assert "forged artifact include-path processing metadata" in (
            backend_support["notes"]
        )
        assert (
            "include-path processing summary rollup mismatches including target "
            "and variant rollups" in backend_support["notes"]
        )
        assert "missing include-path processing target and variant rollups" in (
            backend_support["notes"]
        )
        assert (
            "text inspection identifies inactive include directories by path, "
            "resolved path, and frontend visibility"
        ) in backend_support["notes"]
        assert (
            "include dependency status, kind, source-backend, "
            "source-backend status, resolution-source"
        ) in backend_support["notes"]
        assert "resolved include dependency samples" in backend_support["notes"]
        assert "unresolved include dependency samples" in backend_support["notes"]
        assert "system include dependency samples" in backend_support["notes"]
        assert (
            "source-backend labels, project-define provenance, variant names"
            in backend_support["notes"]
        )
        assert "unit source hash metadata" in backend_support["notes"]
        assert (
            "resolved include hash and byte-size metadata" in backend_support["notes"]
        )
        assert (
            "Resolved include dependency reports and inspection samples include "
            "byte-size metadata"
        ) in backend_support["notes"]
        assert (
            "OpenGL/GLSL translation with a resolved angle include and "
            "variant-specific conditional output"
        ) in backend_support["notes"]
        assert (
            "DirectX/HLSL and Metal/MSL project translation with resolved local headers "
            "and variant-specific conditional output"
        ) in backend_support["notes"]
        assert (
            "Slang project translation with resolved local headers and "
            "variant-specific conditional output"
        ) in backend_support["notes"]
        assert (
            "Vulkan project translation with a resolved angle include and "
            "variant-specific conditional output"
        ) in backend_support["notes"]
        assert (
            "CUDA/HIP project translation with unresolved runtime system includes, "
            "resolved local headers, and variant-specific conditional output"
        ) in backend_support["notes"]
        assert "project-define plus variant include provenance" in (
            backend_support["notes"]
        )
        assert "structured variant fields and diagnostic variant rollups" in (
            backend_support["notes"]
        )
        assert (
            "missing, dynamic, and cyclic include diagnostics"
            in backend_support["notes"]
        )
        assert "sampled include-path processing artifact metadata" in (
            backend_support["notes"]
        )
        assert "includeDependencies records" in backend_support["notes"]
        assert "ignores block-commented preprocessor directives" in (
            backend_support["notes"]
        )
        assert "directives after same-line block comments" in (backend_support["notes"])
        assert (
            "simple #if, #ifdef, #ifndef, #elif, #else, and #endif branches"
            in backend_support["notes"]
        )
        assert "simple integer comparisons" in backend_support["notes"]
        assert "integer and boolean define values" in backend_support["notes"]
        assert (
            "summary rollups including source-backend, source-backend status, "
            "resolution-source, and variant counts"
        ) in backend_support["notes"]
        assert "missing current include-scan diagnostics" in (backend_support["notes"])
        assert "missing include dependency summary rollups" in (
            backend_support["notes"]
        )
        assert "resolved include read failures" in backend_support["notes"]
        assert "project.scan.include-read-failed" in backend_support["notes"]
        assert "current status, resolved path, resolved include hash" in (
            backend_support["notes"]
        )
        assert "resolved include size" in backend_support["notes"]
        assert (
            "expected/actual include status, path, source, hash, and byte-size "
            "mismatch context" in backend_support["notes"]
        )
        assert (
            "current-scan mismatch labels include resolved path, resolution source, "
            "and project-define provenance"
        ) in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_include_dir_files_without_hiding_units"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_records_include_dependency_resolution"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_skips_inactive_ifdef_include_dependencies"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_records_variant_conditional_include_dependencies"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_variant_dynamic_include_diagnostics"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_variant_nested_include_cycle_diagnostics"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_limits_named_variants_to_selected"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_honors_ifndef_else_in_nested_include_dependencies"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_keeps_includes_for_unsupported_conditional_"
            "expressions"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_evaluates_integer_comparison_include_conditions"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_ignores_block_commented_preprocessor_directives"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_nested_include_read_failures"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_current_include_scan_"
            "diagnostics"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_filters_invalid_include_dirs_before_frontend"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_stale_include_dir_resolved_path"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_stale_include_dependency_hashes"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_malformed_include_dependency_records"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_stale_include_dependency_resolution"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_include_dependency_resolution_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_labels_forged_define_include_provenance"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_include_path_processing_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_include_path_processing_summary_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_processing_variant_rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_scan_summary_rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_translation_pipeline.py::def "
            "test_source_registry_reports_lexer_option_support"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_report_records_include_dir_and_define_overrides"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_scan_rejects_empty_include_dir_override"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_scan_normalizes_path_override_separators"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_named_variants_apply_native_opengl_preprocessor"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_named_variants_apply_cuda_hip_preprocessor"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_named_variants_apply_native_directx_preprocessor"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_named_variants_apply_native_metal_preprocessor"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_named_variants_apply_native_slang_preprocessor"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_named_variants_apply_native_vulkan_preprocessor"
        ) in backend_support["evidence"]


def test_project_macro_variants_document_artifact_define_maps():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.macro_variants"]

    for backend_support in feature["support"].values():
        assert backend_support["status"] == "partial"
        assert "records each artifact applied define map" in backend_support["notes"]
        assert "define-processing status" in backend_support["notes"]
        assert "applies CLI define overrides" in backend_support["notes"]
        assert "source frontend support metadata" in backend_support["notes"]
        assert "warning diagnostics and missing-capability rollups" in (
            backend_support["notes"]
        )
        assert (
            "active #define or #undef directives in translation units or resolved "
            "include files shadow configured project or selected variant define names"
        ) in backend_support["notes"]
        assert (
            "active #error and #warning directives scoped by project and "
            "selected variant conditionals"
        ) in backend_support["notes"]
        assert "ignoring block-commented directives" in backend_support["notes"]
        assert (
            "inspection summaries and text output expose variant names, "
            "selected-variant summaries for scoped runs, and per-variant "
            "define counts" in backend_support["notes"]
        )
        assert (
            "Project inspection exposes project-level define names and "
            "deterministic define fingerprints, variant define names, and "
            "deterministic per-variant define fingerprints without define values"
        ) in backend_support["notes"]
        assert "report CLI variant metadata" in backend_support["notes"]
        assert (
            "define-processing status, target, source-backend, and variant rollups"
            in backend_support["notes"]
        )
        assert "sampled define-processing artifact metadata" in (
            backend_support["notes"]
        )
        assert "with define names and without define values" in backend_support["notes"]
        assert "scoped runs for selected declared variants" in backend_support["notes"]
        assert "configured selected_variants defaults" in backend_support["notes"]
        assert "de-duplicates repeated selections before scan or artifact planning" in (
            backend_support["notes"]
        )
        assert "sanitizes unsafe variant names into stable output path segments" in (
            backend_support["notes"]
        )
        assert (
            "malformed define/variant metadata including empty mapping keys, "
            "punctuation-bearing variant keys in config/report diagnostics, "
            "and forged variant define counts"
        ) in backend_support["notes"]
        assert "artifact define maps that do not match base defines merged" in (
            backend_support["notes"]
        )
        assert "forged artifact define-processing metadata" in (
            backend_support["notes"]
        )
        assert (
            "define-processing summary rollup mismatches including target and "
            "variant rollups" in backend_support["notes"]
        )
        assert "missing define-processing target and variant rollups" in (
            backend_support["notes"]
        )
        assert "selected-variant TOML values" in backend_support["notes"]
        assert "object-like define expansion" in backend_support["notes"]
        assert "#if/#ifdef/#ifndef/#elif/#else/#endif branch selection" in (
            backend_support["notes"]
        )
        assert (
            "OpenGL/GLSL native preprocessing during project translation"
            in backend_support["notes"]
        )
        assert (
            "DirectX/HLSL native preprocessing during project translation"
            in backend_support["notes"]
        )
        assert (
            "Metal/MSL native preprocessing during project translation"
            in backend_support["notes"]
        )
        assert (
            "CUDA/HIP native preprocessing during project translation"
            in backend_support["notes"]
        )
        assert (
            "Slang native preprocessing during project translation"
            in backend_support["notes"]
        )
        assert (
            "Vulkan native preprocessing during project translation"
            in backend_support["notes"]
        )
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_define_mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_configured_define_shadowing"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_scopes_define_shadowing_to_selected_variants"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_include_define_shadowing"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_ignores_block_commented_preprocessor_directives"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_define_processing_mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_define_processing_summary_mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_processing_variant_rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_translation_pipeline.py::def "
            "test_source_registry_reports_lexer_option_support"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_project_config_counts"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_report_records_include_dir_and_define_overrides"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_report_records_variant_metadata"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_report_limits_named_variants_to_selected"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_lexer.py::def "
            "test_define_preprocessing_selects_active_branch_and_expands_macros"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_named_variants_apply_crossgl_defines"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_named_variants_apply_native_opengl_preprocessor"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_named_variants_apply_cuda_hip_preprocessor"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_named_variants_apply_native_directx_preprocessor"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_named_variants_apply_native_metal_preprocessor"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_named_variants_apply_native_slang_preprocessor"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_named_variants_apply_native_vulkan_preprocessor"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_records_define_processing_without_frontend_"
            "support"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_limits_named_variants_to_selected"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_limits_named_variants_to_selected"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_uses_configured_selected_variants"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_sanitizes_variant_output_segments"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_translate_project_limits_named_variants_to_selected"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_empty_project_mapping_keys"
        ) in backend_support["evidence"]

        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_quotes_variant_keys_with_punctuation"
        ) in backend_support["evidence"]


def test_project_diagnostics_document_location_path_checks():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.diagnostics"]

    for backend_support in feature["support"].values():
        assert "translation-time warnings surfaced in report summaries" in (
            backend_support["notes"]
        )
        assert "comment-aware scan-time translation-unit and include-file" in (
            backend_support["notes"]
        )
        assert "non-repository-relative diagnostic locations" in (
            backend_support["notes"]
        )
        assert "optional originalLocation fields" in backend_support["notes"]
        assert "original source locations as SARIF related locations" in (
            backend_support["notes"]
        )
        assert "include, source override, and exclude scan patterns" in (
            backend_support["notes"]
        )
        assert "contract-checked rollups by severity, diagnostic code" in (
            backend_support["notes"]
        )
        assert "validation rejects missing diagnostic severity, code" in (
            backend_support["notes"]
        )
        assert "text output carries diagnostic location, original source location" in (
            backend_support["notes"]
        )
        assert (
            "variant-scoped include resolution and active preprocessor diagnostics"
            in backend_support["notes"]
        )
        assert "diagnostics whose targets are not declared by the report" in (
            backend_support["notes"]
        )
        assert "or noncanonical in full reports" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_filters_invalid_include_dirs_before_frontend"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_records_define_processing_without_frontend_"
            "support"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_validate_project_sarif_reports_generated_diagnostics"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_diagnostics_preserve_original_locations"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_variant_define_backed_include_resolution_"
            "diagnostics"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_variant_dynamic_include_diagnostics"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_variant_nested_include_cycle_diagnostics"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_active_preprocessor_diagnostics_by_variant"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_fails_on_error_diagnostics"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_diagnostic_locations_outside_project"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_noncanonical_diagnostic_targets"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_diagnostic_summary_rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_top_level_"
            "diagnostic_counts"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_exclude_patterns_outside_project"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_drive_relative_exclude_patterns"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_configured_define_shadowing"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_scopes_define_shadowing_to_selected_variants"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_include_define_shadowing"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_ignores_block_commented_preprocessor_directives"
        ) in backend_support["evidence"]


def test_project_batch_translation_documents_artifact_matrix_rollups():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.batch_translation"]

    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert "artifact matrix emitted, translated, failed, missing, extra" in (
            backend_support["notes"]
        )
        assert "target and variant rollups" in backend_support["notes"]
        assert (
            "real translator coverage for multiple units, all supported target "
            "backends, and variants"
        ) in backend_support["notes"]
        assert "command-scoped source-root overrides" in backend_support["notes"]
        assert "rejects empty CLI target overrides" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_records_artifact_matrix_metadata"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_batches_real_units_targets_and_variants"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_matrix_rollup_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_reports_artifact_matrix_gaps"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_translate_project_applies_source_root_overrides"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_scan_rejects_empty_target_override"
        ) in backend_support["evidence"]


def test_project_artifact_manifest_documents_source_map_requirement():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.artifact_manifest"]

    for backend_support in feature["support"].values():
        assert "current translated artifacts without source-map records" in (
            backend_support["notes"]
        )
        assert "full reports with units or artifacts missing source hashes" in (
            backend_support["notes"]
        )
        assert "artifacts missing source byte sizes" in backend_support["notes"]
        assert "failed artifacts missing error metadata or generated metadata" in (
            backend_support["notes"]
        )
        assert "translated artifacts with error metadata" in backend_support["notes"]
        assert "source-relative target/variant layout" in backend_support["notes"]
        assert "stable repository-relative POSIX paths" in backend_support["notes"]
        assert "unit-target-variant artifact matrix entries" in (
            backend_support["notes"]
        )
        assert "applied define map" in backend_support["notes"]
        assert "missing or mismatched artifact define maps" in (
            backend_support["notes"]
        )
        assert "discovered units with source hashes" in backend_support["notes"]
        assert "full reports with units or artifacts missing source hashes" in (
            backend_support["notes"]
        )
        assert "artifactMatrix metadata" in backend_support["notes"]
        assert "scan-only artifactMatrix plans" in backend_support["notes"]
        assert "before artifact generation" in backend_support["notes"]
        assert "closed top-level report field set" in backend_support["notes"]
        assert "artifact records whose targets are not declared by the report" in (
            backend_support["notes"]
        )
        assert "or noncanonical in full reports" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_current_translated_artifacts_"
            "without_source_maps"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_failed_artifacts_with_"
            "generated_metadata"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_failed_artifacts_without_"
            "source_hash"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_target_artifact_"
            "matrix_entries"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_variant_artifact_"
            "matrix_entries"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_records_artifact_matrix_metadata"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_allows_scan_reports_without_artifacts"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_project_report_summarizes_scan_only_artifact_matrix"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_emits_closed_portability_report_schema"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_empty_translated_artifact_"
            "matrix"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_matrix_count_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_matrix_variant_"
            "mode_mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_noncanonical_full_report_targets"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_translated_artifacts_with_"
            "error_metadata"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_path_source_layout_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_backslash_report_identity_paths"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_artifact_defines"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_unit_source_hashes"
        ) in backend_support["evidence"]


def test_project_source_provenance_documents_source_map_mapping_checks():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.source_provenance"]

    for backend_support in feature["support"].values():
        assert "unit source hashes" in backend_support["notes"]
        assert "unit source hash records that are missing" in backend_support["notes"]
        assert "unregistered or non-canonical unit source backend names" in (
            backend_support["notes"]
        )
        assert "inconsistent unit and skipped source override provenance" in (
            backend_support["notes"]
        )
        assert "artifact source hashes that do not match declared" in (
            backend_support["notes"]
        )
        assert "line-preserving source-map mappings" in (backend_support["notes"])
        assert "line-preserving source-map validation" in backend_support["notes"]
        assert "non-empty source-map mappings" in backend_support["notes"]
        assert "file-level source-map mapping cardinality" in backend_support["notes"]
        assert "fine-grained positive-length source-map mappings" in (
            backend_support["notes"]
        )
        assert (
            "fine-grained source-map byte and line/column span containment"
            in backend_support["notes"]
        )
        assert "source-map summary rollups" in backend_support["notes"]
        assert "source-map summary rollups including variant rollups" in (
            backend_support["notes"]
        )
        assert (
            "source-remap summary totals for artifact count and mapping count "
            "plus rollups by granularity, target, source backend, and variant"
        ) in (backend_support["notes"])
        assert (
            "sanitized variant output segments in source-map and source-remap paths"
            in (backend_support["notes"])
        )
        assert "stable repository-relative POSIX report identity paths" in (
            backend_support["notes"]
        )
        assert "canonical source-map and source-remap target metadata" in (
            backend_support["notes"]
        )
        assert (
            "source-map, source-remap target, source-remap mapping-count, "
            "and source-remap sidecar hash and byte-size"
        ) in backend_support["notes"]
        assert "missing source-map and source-remap granularity or variant rollups" in (
            backend_support["notes"]
        )
        assert "source-remap mapping-count summary mismatches" in (
            backend_support["notes"]
        )
        assert "source-remap mapping-count mismatches" in backend_support["notes"]
        assert "source-remap sidecar size mismatches" in backend_support["notes"]
        assert "current artifact-level source-map span coverage" in (
            backend_support["notes"]
        )
        assert "closed compiler source-remap sidecar field sets" in (
            backend_support["notes"]
        )
        assert (
            "bounded source-map and source-remap artifact inspection samples "
            "with source/generated span, mapping-count, source hash and byte-size "
            "metadata, sidecar hash plus byte-size metadata, failed validation "
            "status metadata, and custom limits"
        ) in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_marks_source_map_validation_"
            "failures"
        ) in backend_support["evidence"]
        assert (
            "text inspection derives file-level source-map counts from total "
            "and fine-grained source-map totals and reports aggregate "
            "source-remap mapping counts"
        ) in backend_support["notes"]
        assert (
            "artifact provenance summary rollups by pipeline, intermediate "
            "marker, source backend with intermediate marker, target with "
            "intermediate marker, and variant with intermediate marker"
        ) in backend_support["notes"]
        assert (
            "bounded direct, bridged, and combined artifact provenance "
            "inspection samples with source hash and byte-size metadata plus "
            "generated hash and byte-size metadata, failed validation status metadata"
        ) in backend_support["notes"]
        assert "compiler-compatible source-remap sidecar semantics" in (
            backend_support["notes"]
        )
        assert "source-map-derived generated/original mappings" in (
            backend_support["notes"]
        )
        assert "unit and skipped source override rollups" in (backend_support["notes"])
        assert "generated artifact hashes and byte sizes" in backend_support["notes"]
        assert "source-relative target/variant layout" in backend_support["notes"]
        assert (
            "source-size validation records that no longer match the current "
            "source artifact, artifact generated byte-size records that are "
            "missing or malformed, "
            "generated-size validation records that no longer match the current "
            "generated artifact"
        ) in backend_support["notes"]
        assert (
            "records per-artifact source hash, source-size, generated hash, "
            "generated-size, source-map, and source-remap statuses"
        ) in backend_support["notes"]
        assert (
            "records aggregate validation status counts including source-size "
            "and generated-size counts"
        ) in backend_support["notes"]
        assert "missing artifact provenance summary rollups" in (
            backend_support["notes"]
        )
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_detects_modified_unit_sources"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_source_hash_"
            "mismatches_unit_source_hash"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_out_of_anchor_fine_grained_"
            "source_map_spans"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_invalid_unit_source_backends"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_inconsistent_unit_source_overrides"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_inconsistent_skipped_source_"
            "overrides"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_source_override_rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_source_map_counts"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_source_map_target_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_project_report_applies_custom_sample_limits"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_project_report_groups_direct_and_bridged_artifact_"
            "provenance"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_validation_variant_"
            "rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_source_map_counts_split_file_and_fine_grained_totals"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_sanitizes_variant_source_map_and_remap_paths"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_source_remap_mapping_count_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_backslash_source_remap_metadata"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_artifact_provenance_"
            "source_backend_rollup"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_artifact_provenance_"
            "variant_rollup"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_noncanonical_source_remap_targets"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_source_map_variant_rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_inconsistent_summary_counts"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_empty_source_map_mappings"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_multiple_file_level_"
            "source_map_mappings"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_path_source_layout_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_compiler_incompatible_source_"
            "remap_sidecar"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_incomplete_file_level_source_"
            "map_spans"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_source_remap_sidecar_extra_fields"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_source_remap_size_mismatches"
        ) in backend_support["evidence"]


def test_project_validation_hooks_document_migration_contract_checks():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.validation_hooks"]

    for backend_support in feature["support"].values():
        assert (
            "migration scope, non-goals, required action count and kind, "
            "severity, and target rollups, non-empty action targets, "
            "translated artifact target references, and canonical target "
            "declarations"
        ) in backend_support["notes"]
        assert "unit source hash checks" in backend_support["notes"]
        assert "canonical source backend declarations" in backend_support["notes"]
        assert "unit and skipped source override provenance" in (
            backend_support["notes"]
        )
        assert "check-kind metadata" in backend_support["notes"]
        assert "summary rollups including source override counts" in (
            backend_support["notes"]
        )
        assert "non-empty scan-scope list entries" in backend_support["notes"]
        assert "unit extension/path consistency" in backend_support["notes"]
        assert "artifactMatrix metadata" in backend_support["notes"]
        assert "scan-only artifactMatrix plans" in backend_support["notes"]
        assert "artifact matrix coverage" in backend_support["notes"]
        assert (
            "direct validation report artifact target, artifact source backend, "
            "artifact variant"
        ) in backend_support["notes"]
        assert "closed standalone validation-report field set" in (
            backend_support["notes"]
        )
        assert "applied define map consistency" in backend_support["notes"]
        assert "target/variant directory containment" in backend_support["notes"]
        assert "source-root status record and count consistency" in (
            backend_support["notes"]
        )
        assert "include-directory status record and count consistency" in (
            backend_support["notes"]
        )
        assert (
            "expected/actual project status record mismatch context"
            in backend_support["notes"]
        )
        assert "source-relative layout" in backend_support["notes"]
        assert "artifact target suffix consistency" in backend_support["notes"]
        assert "canonical artifact target records" in backend_support["notes"]
        assert "canonical validation target records" in backend_support["notes"]
        assert "noncanonical in full reports" in backend_support["notes"]
        assert "required full-report artifact source hashes" in (
            backend_support["notes"]
        )
        assert (
            "expected/actual hash and byte-size mismatch context"
            in backend_support["notes"]
        )
        assert "failed artifact error metadata" in backend_support["notes"]
        assert "translated artifact error metadata rejection" in (
            backend_support["notes"]
        )
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_malformed_include_dir_status_"
            "records"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_malformed_source_root_status_"
            "records"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_blank_project_config_list_entries"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_source_root_status_count_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_stale_source_root_status"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_stale_source_root_resolved_path"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_define_mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_validation_variant_"
            "rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_groups_artifact_status_by_source_"
            "backend"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_emits_closed_validation_report_schema"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_noncanonical_full_report_targets"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_noncanonical_diagnostic_targets"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_target_artifact_"
            "matrix_entries"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_variant_artifact_"
            "matrix_entries"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_empty_translated_artifact_"
            "matrix"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_matrix_count_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_matrix_variant_"
            "mode_mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_allows_scan_reports_without_artifacts"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_detects_modified_unit_sources"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_detects_modified_source_artifacts"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_generated_size_"
            "mismatches_current_artifact"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_detects_modified_generated_artifacts"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_artifact_provenance_"
            "variant_rollup"
        ) in backend_support["evidence"]
        assert "failed artifact generated metadata rejection" in (
            backend_support["notes"]
        )
        assert "required and canonical artifact provenance" in backend_support["notes"]
        assert "required translated artifact source maps" in backend_support["notes"]
        assert "source-map summary rollups" in backend_support["notes"]
        assert "artifact provenance source-backend, target, and variant rollups" in (
            backend_support["notes"]
        )
        assert "failed artifact-provenance validation metadata" in (
            backend_support["notes"]
        )
        assert "including source-size and generated-size status" in (
            backend_support["notes"]
        )
        assert "non-empty source-map mappings" in backend_support["notes"]
        assert "file-level source-map mapping cardinality" in backend_support["notes"]
        assert "fine-grained positive-length source-map mappings" in (
            backend_support["notes"]
        )
        assert (
            "fine-grained source-map byte and line/column span containment"
            in backend_support["notes"]
        )
        assert "repository-relative file paths" in backend_support["notes"]
        assert (
            "required summarized validation artifact hash, source-size, "
            "generated-size, source-map, and source-remap status fields"
        ) in backend_support["notes"]
        assert "source-size and generated-size status counts" in (
            backend_support["notes"]
        )
        assert (
            "source-size status, generated-size status, and validation status rollups"
        ) in backend_support["notes"]
        assert "validation diagnostic-code and missing-capability rollups" in (
            backend_support["notes"]
        )
        assert (
            "validate-project JSON, text, and SARIF summaries"
            in backend_support["notes"]
        )
        assert "validation report schema/kind/generated-at metadata" in (
            backend_support["notes"]
        )
        assert "direct validation report artifact target" in backend_support["notes"]
        assert "artifact source backend" in backend_support["notes"]
        assert "toolchain run status" in backend_support["notes"]
        assert "toolchain run target" in backend_support["notes"]
        assert "toolchain run source backend" in backend_support["notes"]
        assert "toolchain run check-kind" in backend_support["notes"]
        assert "toolchain run tool" in backend_support["notes"]
        assert "toolchain run variant" in backend_support["notes"]
        assert "translate-project can embed artifact validation" in (
            backend_support["notes"]
        )
        assert "target-tool command consistency" in backend_support["notes"]
        assert "target check-kind consistency" in backend_support["notes"]
        assert "artifact command argv consistency" in backend_support["notes"]
        assert "availability command argv consistency" in backend_support["notes"]
        assert "required summarized validation toolchain target coverage" in (
            backend_support["notes"]
        )
        assert "scan-scope count consistency" in backend_support["notes"]
        assert "config hash freshness" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_altered_migration_non_goals"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_stale_project_config_hash"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_migration_rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_external_corpus_"
            "accounting"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_unit_extension_mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_paths_outside_target_dir"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_path_suffix_mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_path_source_layout_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_failed_artifacts_without_error"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_translated_artifacts_with_"
            "error_metadata"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_failed_artifacts_with_"
            "generated_metadata"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_failed_artifacts_without_"
            "source_hash"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_diagnostic_locations_outside_project"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_summarized_validation_without_"
            "status_fields"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_validation_summary_missing_"
            "toolchain_targets"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_records_toolchain_run_variant_rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_validation_variant_"
            "rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_marks_availability_only_toolchain_runs"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_can_embed_toolchain_smoke_runs"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_translate_project_validate_records_artifact_checks"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_translate_project_run_toolchains_records_validation"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_toolchain_run_command_target_"
            "mismatch"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_toolchain_run_check_kind_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_toolchain_run_"
            "argv_mismatch"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_availability_toolchain_run_"
            "argv_mismatch"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_runs_vulkan_assembly_when_only_"
            "assembler_available"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_records_unavailable_toolchains_"
            "deterministically"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_or_forged_artifact_provenance"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_current_translated_artifacts_"
            "without_source_maps"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_empty_source_map_mappings"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_multiple_file_level_"
            "source_map_mappings"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_migration_actions_with_"
            "undeclared_targets"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_project_config_count_mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_invalid_unit_source_backends"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_inconsistent_unit_source_overrides"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_inconsistent_skipped_source_"
            "overrides"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_source_override_rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_inconsistent_summary_counts"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_empty_project_mapping_keys"
        ) in backend_support["evidence"]

    opengl_support = feature["support"]["opengl"]
    assert (
        "OpenGL glslangValidator stdin smoke checks with an explicit inferred stage"
        in (opengl_support["notes"])
    )
    assert (
        "tests/test_translator/test_project_translation.py::def "
        "test_opengl_toolchain_smoke_command_selects_glslang_stage"
    ) in opengl_support["evidence"]

    vulkan_support = feature["support"]["vulkan"]
    assert (
        "Vulkan SPIR-V assembly smoke checks use spirv-as while binary "
        "SPIR-V artifacts use spirv-val"
    ) in vulkan_support["notes"]
    assert (
        "tests/test_translator/test_project_translation.py::def "
        "test_vulkan_toolchain_smoke_command_selects_spirv_tool"
    ) in vulkan_support["evidence"]
    assert (
        "tests/test_translator/test_project_translation.py::def "
        "test_validate_project_report_assembles_vulkan_spirv_assembly"
    ) in vulkan_support["evidence"]


def test_project_external_corpus_coverage_documents_entry_consistency_checks():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.external_corpus_coverage"]

    for backend_support in feature["support"].values():
        assert "entry presence/discovery/source-backend consistency" in (
            backend_support["notes"]
        )
        assert "commit hash and source URL provenance consistency" in (
            backend_support["notes"]
        )
        assert "required manifest accounting" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_external_corpus_entry_"
            "state_mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_records_missing_external_corpus_manifest"
        ) in backend_support["evidence"]
        assert "skip malformed or duplicate manifest entries" in (
            backend_support["notes"]
        )
        assert "sampled missing and present-but-undiscovered entries" in (
            backend_support["notes"]
        )
        assert "retained repository, commit, and source URL provenance metadata" in (
            backend_support["notes"]
        )
        assert (
            "pinned Slang reduced fixture participates in a manifest-backed "
            "project translation run"
        ) in backend_support["notes"]
        assert "configurable inspection sample limits" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_external_corpus_rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_applies_external_corpus_sample_limit"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_skips_duplicate_external_corpus_entries"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_skips_invalid_external_corpus_provenance"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_runs_pinned_slang_external_corpus_fixture"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_missing_external_corpus_"
            "accounting"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_malformed_external_corpus_"
            "provenance"
        ) in backend_support["evidence"]


def test_support_matrix_check_writes_machine_readable_report(tmp_path):
    report_path = tmp_path / "support-matrix-check.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/support_matrix.py",
            "check",
            "--output",
            str(report_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "Wrote" in result.stdout
    assert report["schema_version"] == 1
    assert report["ok"] is True
    assert report["summary"] == {
        "artifact_count": 3,
        "stale_count": 0,
        "stale_artifacts": [],
        "total_diff_line_count": 0,
    }
    assert {artifact["path"] for artifact in report["artifacts"]} == {
        "support/generated/support-matrix.json",
        "support/generated/graphics-backend-roadmap.json",
        "docs/source/support-matrix.rst",
    }
    assert all(artifact["diff"] == [] for artifact in report["artifacts"])


def test_generated_check_report_includes_stale_artifact_diff(tmp_path, monkeypatch):
    module = load_support_matrix_module()
    backends, features = _minimal_catalogs(module)
    matrix = module.build_matrix(backends, features)
    matrix_path = tmp_path / "support-matrix.json"
    roadmap_path = tmp_path / "graphics-backend-roadmap.json"
    docs_path = tmp_path / "support-matrix.rst"
    monkeypatch.setattr(module, "MATRIX_JSON_PATH", matrix_path)
    monkeypatch.setattr(module, "GRAPHICS_ROADMAP_JSON_PATH", roadmap_path)
    monkeypatch.setattr(module, "DOCS_RST_PATH", docs_path)
    module.write_generated(matrix)
    matrix_path.write_text("{}\n", encoding="utf-8")

    report = module.build_generated_check_report(matrix)
    stale_artifacts = [
        artifact for artifact in report["artifacts"] if artifact["stale"]
    ]

    assert report["ok"] is False
    assert report["summary"] == {
        "artifact_count": 3,
        "stale_count": 1,
        "stale_artifacts": [stale_artifacts[0]["path"]],
        "total_diff_line_count": stale_artifacts[0]["diff_line_count"],
    }
    assert len(stale_artifacts) == 1
    assert stale_artifacts[0]["path"].endswith("support-matrix.json")
    assert stale_artifacts[0]["exists"] is True
    assert stale_artifacts[0]["actual_sha256"] != stale_artifacts[0]["expected_sha256"]
    assert stale_artifacts[0]["diff_line_count"] == len(stale_artifacts[0]["diff"])
    assert any(line.startswith("-{}") for line in stale_artifacts[0]["diff"])


def test_print_generated_failures_reports_stale_artifact_summary(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = load_support_matrix_module()
    backends, features = _minimal_catalogs(module)
    matrix = module.build_matrix(backends, features)
    matrix_path = tmp_path / "support-matrix.json"
    roadmap_path = tmp_path / "graphics-backend-roadmap.json"
    docs_path = tmp_path / "support-matrix.rst"
    monkeypatch.setattr(module, "MATRIX_JSON_PATH", matrix_path)
    monkeypatch.setattr(module, "GRAPHICS_ROADMAP_JSON_PATH", roadmap_path)
    monkeypatch.setattr(module, "DOCS_RST_PATH", docs_path)
    module.write_generated(matrix)
    matrix_path.write_text("{}\n", encoding="utf-8")
    report = module.build_generated_check_report(matrix)

    module.print_generated_failures(report)

    captured = capsys.readouterr()
    assert "Generated support matrix artifacts are stale." in captured.err
    assert "Run: python tools/support_matrix.py update" in captured.err
    assert "Stale artifact summary:" in captured.err
    assert "support-matrix.json:" in captured.err
    assert "diff lines (exists); actual=" in captured.err
    assert "expected=" in captured.err
    assert "Diff for" in captured.err


def test_validate_backend_catalog_rejects_duplicate_aliases():
    module = load_support_matrix_module()
    backends = {
        "backends": [
            _backend("directx", ["shared"], ".hlsl"),
            _backend("opengl", ["shared"], ".glsl"),
        ]
    }

    with pytest.raises(module.SupportMatrixError, match="Duplicate backend alias"):
        module.validate_backend_catalog(backends)


def test_validate_feature_catalog_requires_all_status_definitions():
    module = load_support_matrix_module()
    statuses = _status_descriptions(module)
    statuses.pop("unknown")
    features = {
        "statuses": statuses,
        "features": [
            {
                "id": "target.codegen",
                "category": "target",
                "name": "Code generation",
                "description": "Emit target code.",
                "support": {},
            }
        ],
    }

    with pytest.raises(
        module.SupportMatrixError,
        match="missing status definition",
    ):
        module.validate_feature_catalog(features, {"directx"})


def test_validate_feature_catalog_rejects_typoed_support_keys():
    module = load_support_matrix_module()
    features = {
        "statuses": _status_descriptions(module),
        "features": [
            {
                "id": "target.codegen",
                "category": "target",
                "name": "Code generation",
                "description": "Emit target code.",
                "support": {
                    "directx": {
                        "status": "supported",
                        "evidnece": [],
                    }
                },
            }
        ],
    }

    with pytest.raises(
        module.SupportMatrixError,
        match="unsupported support key",
    ):
        module.validate_feature_catalog(features, {"directx"})


def test_validate_feature_catalog_rejects_typoed_support_plan_keys():
    module = load_support_matrix_module()
    features = {
        "statuses": _status_descriptions(module),
        "features": [
            {
                "id": "target.codegen",
                "category": "target",
                "name": "Code generation",
                "description": "Emit target code.",
                "support_plan": {
                    "completion_rule": "Wrong key name.",
                },
                "support": {
                    "directx": {
                        "status": "partial",
                    }
                },
            }
        ],
    }

    with pytest.raises(
        module.SupportMatrixError,
        match="unsupported support plan key",
    ):
        module.validate_feature_catalog(features, {"directx"})


def test_support_plan_fields_flow_to_generated_backlog_rows():
    module = load_support_matrix_module()
    backends, features = _minimal_catalogs(module)

    matrix = module.build_matrix(backends, features)

    support = matrix["features"][0]["support"]
    assert "current_gap" not in support["directx"]
    assert support["opengl"]["current_gap"] == (
        "OpenGL code generation still needs coverage."
    )
    assert support["opengl"]["next_scope"] == (
        "Add parser and codegen fixtures for OpenGL."
    )
    assert support["opengl"]["completion_criteria"] == (
        "Mark supported when OpenGL has matching evidence."
    )
    row = next(item for item in matrix["backlog"] if item["backend_id"] == "opengl")
    assert row == {
        "feature_id": "target.codegen",
        "feature": "Code generation",
        "category": "target",
        "backend_id": "opengl",
        "backend": "opengl",
        "status": "partial",
        "notes": "Needs audit.",
        "current_gap": "OpenGL code generation still needs coverage.",
        "next_scope": "Add parser and codegen fixtures for OpenGL.",
        "completion_criteria": "Mark supported when OpenGL has matching evidence.",
    }


def test_rendered_docs_label_actionable_backlog_policy():
    module = load_support_matrix_module()
    backends, features = _minimal_catalogs(module)
    features["features"].append(
        {
            "id": "target.fallback",
            "category": "target",
            "name": "Fallback diagnostics",
            "description": "Emit deterministic diagnostics.",
            "support": {
                "directx": {"status": "diagnostic", "notes": "Reports fallback."},
                "opengl": {
                    "status": "validated_rejection",
                    "notes": "Rejected with validation.",
                },
                "metal": {"status": "diagnostic", "notes": "Reports fallback."},
            },
        }
    )
    matrix = module.build_matrix(backends, features)

    docs = module.render_docs(matrix)

    assert "Actionable Backlog" in docs
    assert "Actionable backlog rows" in docs
    assert "DirectX/OpenGL/Metal actionable backlog" in docs
    assert "Non-supported or unaudited feature rows" not in docs
    assert (
        "Evidence-backed\n``diagnostic`` and ``validated_rejection`` rows "
        "remain visible in\nthe matrix counts"
    ) in docs
    assert all(item["feature_id"] != "target.fallback" for item in matrix["backlog"])


def test_validate_matrix_catches_inconsistent_generated_counts():
    module = load_support_matrix_module()
    backends, features = _minimal_catalogs(module)
    module.validate_catalogs(backends, features)
    matrix = module.build_matrix(backends, features)

    module.validate_matrix(matrix)
    matrix["summary"]["feature_count"] += 1

    with pytest.raises(
        module.SupportMatrixError,
        match="feature_count is inconsistent",
    ):
        module.validate_matrix(matrix)


def test_validate_matrix_catches_inconsistent_generated_backlog():
    module = load_support_matrix_module()
    backends, features = _minimal_catalogs(module)
    matrix = module.build_matrix(backends, features)
    matrix["backlog"][0]["notes"] = "stale generated note"

    with pytest.raises(
        module.SupportMatrixError,
        match="backlog rows are inconsistent",
    ):
        module.validate_matrix(matrix)


def test_audit_accepts_comma_separated_status_filters(capsys):
    module = load_support_matrix_module()
    backends, features = _minimal_catalogs(module)
    matrix = module.build_matrix(backends, features)

    result = module.audit(matrix, [], statuses=["partial,unknown"])

    captured = capsys.readouterr()
    assert result == 0
    assert "statuses=partial,unknown" in captured.out
    assert "Backlog rows: 2" in captured.out


def test_audit_rejects_unknown_status_filters():
    module = load_support_matrix_module()
    backends, features = _minimal_catalogs(module)
    matrix = module.build_matrix(backends, features)

    with pytest.raises(module.SupportMatrixError, match="Unknown status filter"):
        module.audit(matrix, [], statuses=["partial,nope"])


def test_evidence_audit_reports_supported_rows_missing_evidence(tmp_path, capsys):
    module = load_support_matrix_module()
    backends, features = _minimal_catalogs(module)
    matrix = module.build_matrix(backends, features)
    output_path = tmp_path / "evidence.json"

    result = module.evidence_audit(
        matrix,
        statuses=["supported"],
        evidence="missing",
        output=output_path,
    )

    captured = capsys.readouterr()
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == 0
    assert "Evidence rows: 1" in captured.out
    assert "directx: present=0, missing=1" in captured.out
    assert report["generator"] == "tools/support_matrix.py evidence"
    assert report["filters"] == {
        "backend_ids": [],
        "categories": [],
        "statuses": ["supported"],
        "evidence": "missing",
    }
    assert report["summary"]["row_count"] == 1
    assert report["summary"]["missing_evidence_count"] == 1
    assert report["summary"]["present_evidence_count"] == 0
    assert report["summary"]["by_backend"]["directx"] == {
        "rows": 1,
        "present": 0,
        "missing": 1,
    }
    assert report["rows"][0]["backend_id"] == "directx"
    assert report["rows"][0]["feature_id"] == "target.codegen"
    assert report["rows"][0]["evidence_count"] == 0


def test_evidence_audit_can_fail_on_missing_evidence():
    module = load_support_matrix_module()
    backends, features = _minimal_catalogs(module)
    matrix = module.build_matrix(backends, features)

    result = module.evidence_audit(
        matrix,
        statuses=["supported"],
        evidence="missing",
        fail_on_missing=True,
    )

    assert result == 1


def test_evidence_audit_rejects_unknown_evidence_filter():
    module = load_support_matrix_module()
    backends, features = _minimal_catalogs(module)
    matrix = module.build_matrix(backends, features)

    with pytest.raises(module.SupportMatrixError, match="Unknown evidence filter"):
        module.evidence_audit(matrix, evidence="stale")


def test_support_matrix_covers_all_cataloged_backends():
    matrix_path = ROOT / "support" / "generated" / "support-matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))

    backend_ids = {backend["id"] for backend in matrix["backends"]}
    assert backend_ids == {
        "directx",
        "opengl",
        "metal",
        "vulkan",
        "cuda",
        "hip",
        "mojo",
        "rust",
        "slang",
    }

    assert matrix["summary"]["feature_count"] == len(matrix["features"])
    assert matrix["summary"]["backend_count"] == len(backend_ids)

    statuses = set(matrix["status_codes"])
    for backend_id in backend_ids:
        counts = matrix["summary"]["status_counts"][backend_id]
        assert set(counts) == statuses
        assert sum(counts.values()) == matrix["summary"]["feature_count"]

    for backend in matrix["backends"]:
        for sample in backend["signals"]["unsupported_marker_samples"]:
            assert set(sample) == {"path", "text"}
            assert sample["path"].endswith(".py")
            assert not sample["path"].startswith("tests/")
            assert sample["text"]


def test_external_corpus_notes_match_manifest_source_backends():
    features_path = ROOT / "support" / "features.json"
    backends_path = ROOT / "support" / "backends.json"
    manifest_path = ROOT / "support" / "external-corpus.json"
    docs_path = ROOT / "docs" / "source" / "project-porting.rst"
    features = json.loads(features_path.read_text(encoding="utf-8"))
    backends = json.loads(backends_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs = docs_path.read_text(encoding="utf-8")

    assert "fixture-backed coverage manifest" in docs
    assert "one pinned entry per registered source backend" in docs

    backend_names = {
        backend["id"]: backend["name"].partition("/")[0].strip()
        for backend in backends["backends"]
    }
    manifest_backend_ids = {
        entry["sourceBackend"]
        for entry in manifest["entries"]
        if "sourceBackend" in entry
    }
    manifest_backend_names = {
        backend_names[backend_id] for backend_id in manifest_backend_ids
    }
    feature = next(
        feature
        for feature in features["features"]
        if feature["id"] == "project.external_corpus_coverage"
    )

    for backend_id in manifest_backend_ids:
        notes = feature["support"][backend_id]["notes"]
        assert f"focused {backend_names[backend_id]} reduction" in notes
        evidence = feature["support"][backend_id]["evidence"]
        assert any(
            item.startswith("tests/test_backend/")
            and "test_external_fixtures.py" in item
            for item in evidence
        )

    for backend_id, support in feature["support"].items():
        if backend_id in manifest_backend_ids:
            continue
        assert not any(
            item.startswith("tests/test_backend/")
            and "test_external_fixtures.py" in item
            for item in support["evidence"]
        )
        if "current pinned reduced-corpus manifest covers" not in support["notes"]:
            continue
        for backend_name in manifest_backend_names:
            assert backend_name in support["notes"]


def test_support_backend_catalog_matches_codegen_registry():
    backends_path = ROOT / "support" / "backends.json"
    catalog = json.loads(backends_path.read_text(encoding="utf-8"))
    register_default_sources()

    backend_ids = {backend["id"] for backend in catalog["backends"]}
    assert backend_ids == set(codegen.backend_names())
    assert backend_ids.issubset(set(SOURCE_REGISTRY.names()))

    for backend in catalog["backends"]:
        backend_id = backend["id"]
        spec = codegen.get_backend(backend_id)
        assert spec is not None, f"{backend_id} is not registered"
        assert codegen.get_backend_extension(backend_id) == backend["target_extension"]
        assert (
            SOURCE_REGISTRY.get_by_extension(backend["target_extension"]).name
            == backend_id
        )

        for alias in backend.get("aliases", []):
            assert (
                codegen.normalize_backend_name(alias) == backend_id
            ), f"{backend_id} support alias is not accepted by codegen: {alias}"
            assert (
                SOURCE_REGISTRY.get(alias).name == backend_id
            ), f"{backend_id} support alias is not accepted by source registry: {alias}"


def test_graphics_backend_roadmap_is_focused_on_primary_graphics_targets():
    roadmap_path = ROOT / "support" / "generated" / "graphics-backend-roadmap.json"
    roadmap = json.loads(roadmap_path.read_text(encoding="utf-8"))

    assert roadmap["view"]["backend_ids"] == ["directx", "opengl", "metal"]
    assert roadmap["summary"]["feature_count"] == len(roadmap["features"])
    assert roadmap["summary"]["backend_count"] == 3
    assert roadmap["summary"]["backlog"]["backlog_count"] == len(roadmap["backlog"])

    for item in roadmap["backlog"]:
        assert item["backend_id"] in {"directx", "opengl", "metal"}


def test_graphics_texture_query_row_is_supported_with_evidence():
    roadmap_path = ROOT / "support" / "generated" / "graphics-backend-roadmap.json"
    roadmap = json.loads(roadmap_path.read_text(encoding="utf-8"))

    texture_query = next(
        feature for feature in roadmap["features"] if feature["id"] == "texture.query"
    )

    for backend_id in ("directx", "opengl", "metal"):
        support = texture_query["support"][backend_id]
        assert support["status"] == "supported"
        assert support["evidence"]

    assert all(item["feature_id"] != "texture.query" for item in roadmap["backlog"])


def test_graphics_texel_fetch_row_is_supported_with_evidence():
    roadmap_path = ROOT / "support" / "generated" / "graphics-backend-roadmap.json"
    roadmap = json.loads(roadmap_path.read_text(encoding="utf-8"))

    texel_fetch = next(
        feature
        for feature in roadmap["features"]
        if feature["id"] == "texture.texel_fetch"
    )

    for backend_id in ("directx", "opengl", "metal"):
        support = texel_fetch["support"][backend_id]
        assert support["status"] == "supported"
        assert support["evidence"]

    assert all(
        item["feature_id"] != "texture.texel_fetch" for item in roadmap["backlog"]
    )


def test_graphics_match_row_is_supported_with_evidence():
    roadmap_path = ROOT / "support" / "generated" / "graphics-backend-roadmap.json"
    roadmap = json.loads(roadmap_path.read_text(encoding="utf-8"))

    match_lowering = next(
        feature for feature in roadmap["features"] if feature["id"] == "language.match"
    )

    for backend_id in ("directx", "opengl", "metal"):
        support = match_lowering["support"][backend_id]
        assert support["status"] == "supported"
        assert support["evidence"]

    assert all(item["feature_id"] != "language.match" for item in roadmap["backlog"])


def test_graphics_metal_wave_intrinsics_row_is_supported_with_evidence():
    roadmap_path = ROOT / "support" / "generated" / "graphics-backend-roadmap.json"
    roadmap = json.loads(roadmap_path.read_text(encoding="utf-8"))

    wave_intrinsics = next(
        feature
        for feature in roadmap["features"]
        if feature["id"] == "language.wave_intrinsics"
    )
    support = wave_intrinsics["support"]["metal"]

    assert support["status"] == "supported"
    assert len(support["evidence"]) >= 13
    assert all(
        not (
            item["backend_id"] == "metal"
            and item["feature_id"] == "language.wave_intrinsics"
        )
        for item in roadmap["backlog"]
    )


def test_support_matrix_audit_writes_filtered_json(tmp_path):
    output_path = tmp_path / "graphics-partial.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/support_matrix.py",
            "audit",
            "--backend",
            "directx,opengl,metal",
            "--status",
            "partial",
            "--output",
            str(output_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Wrote" in result.stdout
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["filters"] == {
        "backend_ids": ["directx", "opengl", "metal"],
        "categories": [],
        "statuses": ["partial"],
    }
    assert report["summary"]["backlog_count"] == len(report["backlog"])
    for item in report["backlog"]:
        assert item["backend_id"] in {"directx", "opengl", "metal"}
        assert item["status"] == "partial"
