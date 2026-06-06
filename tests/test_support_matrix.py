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
                "support": {
                    "directx": {"status": "supported"},
                    "opengl": {"status": "partial", "notes": "Needs audit."},
                },
            }
        ],
    }
    return backends, features


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
        assert "JSON, text, and SARIF summaries" in backend_support["notes"]
        assert "source-map count and provenance rollups" in backend_support["notes"]
        assert "sampled missing and extra artifact identities" in (
            backend_support["notes"]
        )
        assert (
            "validation hash-status, diagnostic-code, missing-capability, "
            "toolchain-status, toolchain-run, and artifact target and variant "
            "rollups" in (backend_support["notes"])
        )
        assert "configurable diagnostic and failed-artifact truncation" in (
            backend_support["notes"]
        )
        assert "failed-artifact variant labels" in backend_support["notes"]
        assert "skipped-reason" in backend_support["notes"]
        assert "source-extension" in backend_support["notes"]
        assert "skipped-extension" in backend_support["notes"]
        assert "invalid-report markers" in backend_support["notes"]
        assert "project-config count" in backend_support["notes"]
        assert "variant-name summaries" in backend_support["notes"]
        assert "source-root status" in backend_support["notes"]
        assert "include-directory status rollups" in backend_support["notes"]
        assert "include dependency status, kind, and resolution-source rollups" in (
            backend_support["notes"]
        )
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
        assert "roots that resolve to non-directory paths" in (backend_support["notes"])
        assert "validates skipped source override provenance" in (
            backend_support["notes"]
        )
        assert "text inspection identifies inactive source roots by path" in (
            backend_support["notes"]
        )
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_project_reports_source_roots_that_are_not_directories"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_inconsistent_skipped_source_"
            "overrides"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_source_root_status"
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
        assert "variant-name summaries" in backend_support["notes"]
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
            "test_project_cli_inspect_report_text_marks_invalid_reports"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_reports_artifact_matrix_gaps"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_reports_truncated_sections"
        ) in backend_support["evidence"]
        assert "source-override" in backend_support["notes"]
        assert "skipped source-override" in backend_support["notes"]
        assert "migration scope, non-goals, action counts, actions" in (
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
        assert "shader/kernel source translation from host runtime APIs" in (
            backend_support["notes"]
        )
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_report_records_documented_migration_actions"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_noncanonical_migration_action_"
            "targets"
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
        assert "non-directory paths" in backend_support["notes"]
        assert "only active existing repository-contained resolved paths" in (
            backend_support["notes"]
        )
        assert "include-path processing status" in backend_support["notes"]
        assert "source frontend support metadata" in backend_support["notes"]
        assert "status, source-backend, and variant rollups" in backend_support["notes"]
        assert "forged artifact include-path processing metadata" in (
            backend_support["notes"]
        )
        assert (
            "include-path processing summary rollup mismatches including variant rollups"
            in backend_support["notes"]
        )
        assert "text inspection identifies inactive include directories by path" in (
            backend_support["notes"]
        )
        assert "include dependency status, kind, source-backend, resolution-source" in (
            backend_support["notes"]
        )
        assert "resolved and unresolved include dependency samples" in (
            backend_support["notes"]
        )
        assert "unresolved include dependency samples" in backend_support["notes"]
        assert "not-supported include-path processing artifact samples" in (
            backend_support["notes"]
        )
        assert "includeDependencies records" in backend_support["notes"]
        assert "summary rollups including source-backend" in (backend_support["notes"])
        assert "current status, resolved path, resolved include hash" in (
            backend_support["notes"]
        )
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
            "test_validate_project_report_rejects_artifact_include_path_processing_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_include_path_processing_summary_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_translation_pipeline.py::def "
            "test_source_registry_reports_lexer_option_support"
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
        assert "source frontend support metadata" in backend_support["notes"]
        assert "inspection summaries expose variant names without define values" in (
            backend_support["notes"]
        )
        assert (
            "define-processing status, source-backend, and variant rollups"
            in backend_support["notes"]
        )
        assert "not-supported define-processing artifact samples" in (
            backend_support["notes"]
        )
        assert "malformed define/variant metadata including empty mapping keys" in (
            backend_support["notes"]
        )
        assert "artifact define maps that do not match base defines merged" in (
            backend_support["notes"]
        )
        assert "forged artifact define-processing metadata" in (
            backend_support["notes"]
        )
        assert (
            "define-processing summary rollup mismatches including variant rollups"
            in backend_support["notes"]
        )
        assert "object-like define expansion" in backend_support["notes"]
        assert "#if/#ifdef/#ifndef/#elif/#else/#endif branch selection" in (
            backend_support["notes"]
        )
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_artifact_define_mismatches"
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
            "tests/test_translator/test_translation_pipeline.py::def "
            "test_source_registry_reports_lexer_option_support"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_project_config_counts"
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
            "test_validate_project_report_rejects_empty_project_mapping_keys"
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
        assert "non-repository-relative diagnostic locations" in (
            backend_support["notes"]
        )
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_diagnostic_locations_outside_project"
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
        assert "full report units or artifacts without source hashes" in (
            backend_support["notes"]
        )
        assert "failed artifacts missing error metadata or generated metadata" in (
            backend_support["notes"]
        )
        assert "translated artifacts with error metadata" in backend_support["notes"]
        assert "source-relative target/variant layout" in backend_support["notes"]
        assert "unit-target-variant artifact matrix entries" in (
            backend_support["notes"]
        )
        assert "applied define map" in backend_support["notes"]
        assert "missing or mismatched artifact define maps" in (
            backend_support["notes"]
        )
        assert "discovered units with source hashes" in backend_support["notes"]
        assert "full report units or artifacts without source hashes" in (
            backend_support["notes"]
        )
        assert "artifactMatrix metadata" in backend_support["notes"]
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
        assert "non-empty source-map mappings" in backend_support["notes"]
        assert "single file-level source-map mapping" in backend_support["notes"]
        assert "source-map summary rollups" in backend_support["notes"]
        assert "source-map summary rollups including variant rollups" in (
            backend_support["notes"]
        )
        assert "source-remap summary rollups including variant rollups" in (
            backend_support["notes"]
        )
        assert "artifact provenance summary rollups" in backend_support["notes"]
        assert "compiler-compatible source-remap sidecar semantics" in (
            backend_support["notes"]
        )
        assert "unit and skipped source override rollups" in (backend_support["notes"])
        assert "source-relative target/variant layout" in backend_support["notes"]
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
            "migration scope, non-goals, action kinds, and canonical target "
            "declarations"
        ) in backend_support["notes"]
        assert "unit source hash checks" in backend_support["notes"]
        assert "canonical source backend declarations" in backend_support["notes"]
        assert "unit and skipped source override provenance" in (
            backend_support["notes"]
        )
        assert "summary rollups including source override counts" in (
            backend_support["notes"]
        )
        assert "unit extension/path consistency" in backend_support["notes"]
        assert "artifactMatrix metadata" in backend_support["notes"]
        assert "artifact matrix coverage" in backend_support["notes"]
        assert "direct validation report artifact target, artifact variant" in (
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
        assert "source-relative layout" in backend_support["notes"]
        assert "artifact target suffix consistency" in backend_support["notes"]
        assert "required full-report artifact source hashes" in (
            backend_support["notes"]
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
            "test_validate_project_report_rejects_source_root_status_count_"
            "mismatches"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_stale_source_root_status"
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
            "test_validate_project_report_allows_scan_reports_without_artifacts"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_detects_modified_unit_sources"
        ) in backend_support["evidence"]
        assert "failed artifact generated metadata rejection" in (
            backend_support["notes"]
        )
        assert "required and canonical artifact provenance" in backend_support["notes"]
        assert "required translated artifact source maps" in backend_support["notes"]
        assert "source-map summary rollups" in backend_support["notes"]
        assert "non-empty source-map mappings" in backend_support["notes"]
        assert "single file-level source-map mapping" in backend_support["notes"]
        assert "repository-relative file paths" in backend_support["notes"]
        assert "required summarized validation artifact hash status fields" in (
            backend_support["notes"]
        )
        assert "validation diagnostic-code and missing-capability rollups" in (
            backend_support["notes"]
        )
        assert (
            "validate-project JSON, text, and SARIF summaries"
            in backend_support["notes"]
        )
        assert "direct validation report artifact target" in backend_support["notes"]
        assert "toolchain run status" in backend_support["notes"]
        assert "toolchain run target" in backend_support["notes"]
        assert "toolchain run variant" in backend_support["notes"]
        assert "required summarized validation toolchain target coverage" in (
            backend_support["notes"]
        )
        assert "scan-scope count consistency" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_altered_migration_non_goals"
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
            "hash_statuses"
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
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_external_corpus_entry_"
            "state_mismatches"
        ) in backend_support["evidence"]
        assert "skip malformed or duplicate manifest entries" in (
            backend_support["notes"]
        )
        assert "sampled missing and present-but-undiscovered entries" in (
            backend_support["notes"]
        )
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_includes_external_corpus_rollups"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_translate_project_skips_duplicate_external_corpus_entries"
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
    features = json.loads(features_path.read_text(encoding="utf-8"))
    backends = json.loads(backends_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

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
