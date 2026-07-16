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
        "source_kind": "native",
        "aliases": aliases,
        "target_extension": extension,
        "translator_codegen": codegen_path,
        "native_backend": native_backend,
        "tests": [test_path],
        "docs": [{"name": "Docs", "url": f"https://example.com/{backend_id}"}],
    }


def _target_only_backend(backend_id, aliases, extension):
    return {
        "id": backend_id,
        "name": backend_id,
        "source_kind": "target-only",
        "aliases": aliases,
        "target_extension": extension,
        "translator_codegen": "crosstl/translator/codegen/GLSL_codegen.py",
        "tests": ["tests/test_translator/test_codegen/test_GLSL_codegen.py"],
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


def test_generated_target_only_backend_aliases_are_visible_as_target_aliases():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    backends = {backend["id"]: backend for backend in matrix["backends"]}

    assert backends["webgl"]["target_aliases"] == ["webgl2", "essl", "glsl-es"]
    assert backends["wgsl"]["target_aliases"] == ["webgpu"]


def test_project_workgroup_size_specialization_support_is_target_scoped():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.workgroup_size_specialization"]
    backend_ids = {backend["id"] for backend in matrix["backends"]}
    supported_targets = {"directx", "opengl"}
    rejection_evidence = (
        "tests/test_translator/test_project_workgroup_rules.py::def "
        "test_project_workgroup_rule_rejects_unsupported_target"
    )

    assert feature["category"] == "project"
    assert feature["name"] == "Per-entry workgroup size specialization"
    assert set(feature["support"]) == backend_ids
    assert {
        backend_id
        for backend_id, support in feature["support"].items()
        if support["status"] == "supported"
    } == supported_targets

    directx = feature["support"]["directx"]
    assert "[project.workgroup_size_rules]" in directx["notes"]
    assert "independent numthreads values per entry" in directx["notes"]
    assert (
        "tests/test_translator/test_codegen/test_directx_codegen.py::def "
        "test_hlsl_compute_workgroup_size_projects_declared_source_shape"
    ) in directx["evidence"]

    opengl = feature["support"]["opengl"]
    assert "standalone GLSL artifacts with runnable main entry points" in (
        opengl["notes"]
    )
    assert (
        "tests/test_translator/test_project_workgroup_rules.py::def "
        "test_project_workgroup_rule_report_rejects_missing_opengl_split"
    ) in opengl["evidence"]
    assert (
        "tests/test_translator/test_codegen/test_GLSL_codegen.py::def "
        "test_glsl_compute_workgroup_size_projects_declared_source_shape"
    ) in opengl["evidence"]

    for backend_id, support in feature["support"].items():
        assert "shader/kernel artifact" in support["notes"]
        assert "runtime execution and numerical parity are not claimed" in (
            support["notes"]
        )
        if backend_id in supported_targets:
            assert support["status"] == "supported"
            assert (
                "tests/test_translator/test_project_workgroup_rules.py::def "
                "test_project_workgroup_rules_emit_directx_library_and_opengl_entries"
            ) in support["evidence"]
        else:
            assert support["status"] == "validated_rejection"
            assert rejection_evidence in support["evidence"]
            assert (
                "project.translate.workgroup-size-rule-unsupported-target"
                in support["notes"]
            )


def test_project_execution_contract_documentation_has_focused_evidence():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    workgroup_evidence = {
        (
            "tests/test_translator/test_project_workgroup_rules.py::def "
            "test_project_workgroup_size_materializes_each_host_named_entry"
        ),
        (
            "tests/test_translator/test_project_workgroup_rules.py::def "
            "test_project_workgroup_size_variants_keep_host_entry_identities_"
            "and_replay"
        ),
        (
            "tests/test_translator/test_project_workgroup_rules.py::def "
            "test_project_workgroup_size_keeps_ordinary_multi_entry_aggregate_closed"
        ),
    }
    runtime_evidence = {
        (
            "tests/test_translator/test_runtime_verification.py::def "
            "test_plan_runtime_test_manifest_rejects_compiled_workgroup_size_"
            "mismatch"
        ),
        (
            "tests/test_translator/test_runtime_verification.py::def "
            "test_plan_runtime_test_manifest_completes_missing_workgroup_size_side"
        ),
    }
    for backend in ("directx", "opengl"):
        assert workgroup_evidence <= set(
            features["project.workgroup_size_specialization"]["support"][backend][
                "evidence"
            ]
        )
        assert runtime_evidence <= set(
            features["project.runtime_test_manifest"]["support"][backend]["evidence"]
        )

    docs = (ROOT / "docs" / "source" / "project-porting.rst").read_text(
        encoding="utf-8"
    )
    normalized_docs = " ".join(docs.split())
    assert "host-named template materialization" in normalized_docs
    assert (
        "sources without that deterministic materialization identity" in normalized_docs
    )
    assert "project.runtime-verification.workgroup-size-mismatch" in normalized_docs
    assert "either missing side of the runtime contract can be" in normalized_docs


def test_project_subgroup_width_specialization_support_is_target_scoped():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.subgroup_width_specialization"]
    backend_ids = {backend["id"] for backend in matrix["backends"]}
    rejection_evidence = (
        "tests/test_translator/test_project_subgroup_width_rules.py::def "
        "test_subgroup_width_rules_fail_closed_without_target_enforcement"
    )

    assert feature["category"] == "project"
    assert feature["name"] == "Per-entry subgroup width specialization"
    assert set(feature["support"]) == backend_ids
    assert {
        backend_id
        for backend_id, support in feature["support"].items()
        if support["status"] == "supported"
    } == {"directx"}

    directx = feature["support"]["directx"]
    assert "[project.subgroup_width_rules]" in directx["notes"]
    assert "single-value WaveSize(width)" in directx["notes"]
    assert "cs_6_6 profile requirement" in directx["notes"]
    assert "exact widths 4, 8, 16, 32, 64, and 128" in directx["notes"]
    assert (
        "tests/test_translator/test_project_subgroup_width_rules.py::def "
        "test_subgroup_width_rules_emit_exact_directx_contract"
    ) in directx["evidence"]
    assert (
        "tests/test_translator/test_project_subgroup_width_rules.py::def "
        "test_directx_exact_wave_size_profile_boundary"
    ) in directx["evidence"]

    opengl = feature["support"]["opengl"]
    assert opengl["status"] == "validated_rejection"
    assert "opengl-enforcement-unavailable" in opengl["notes"]
    assert "no GLSL artifact is emitted" in opengl["notes"]

    for backend_id, support in feature["support"].items():
        assert "shader/kernel artifact" in support["notes"]
        assert "runtime execution and numerical parity are not established" in (
            support["notes"]
        )
        if backend_id == "directx":
            assert support["status"] == "supported"
        else:
            assert support["status"] == "validated_rejection"
            assert rejection_evidence in support["evidence"]
            assert (
                "project.translate.subgroup-width-enforcement-unsupported"
                in support["notes"]
            )
            assert "execution.subgroup-width-specialization" in support["notes"]


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
            "report-or-translation-artifact-derived metadata source, text "
            "inspection of artifact-matrix source provenance"
        ) in backend_support["notes"]
        assert "text inspection of artifact-matrix source provenance" in (
            backend_support["notes"]
        )
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
            "unit source hash and byte-size metadata plus resolved include hash "
            "and byte-size metadata"
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
            "expected/actual source-root status, resolved-path, and scan-visibility "
            "validation context" in backend_support["notes"]
        )
        assert "roots that resolve to non-directory paths" in (backend_support["notes"])
        assert "validates skipped source override provenance" in (
            backend_support["notes"]
        )
        assert "expected/actual source override provenance mismatch context" in (
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
            "test_validate_project_report_rejects_stale_source_root_scan_visibility"
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
            "schema/kind and hash metadata" in backend_support["notes"]
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
            "validation toolchain-run, runtime-reference, and skipped-source "
            "samples with custom limits"
        ) in backend_support["notes"]
        assert (
            "migration scope and non-goal text output, action count and kind, "
            "severity, target, and runtime-reference rollups, bounded migration "
            "action and runtime-reference samples with custom limits, target "
            "lists, and truncation metadata"
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
        assert (
            "action count and kind, severity, target, and runtime-reference rollups"
            in (backend_support["notes"])
        )
        assert (
            "bounded inspection samples with target lists and truncation metadata"
            in (backend_support["notes"])
        )
        assert "runtime-reference count, backend, kind, and path rollups" in (
            backend_support["notes"]
        )
        assert "bounded runtime-reference samples" in backend_support["notes"]
        assert "runtime-reference rollups and samples in CLI inspection text" in (
            backend_support["notes"]
        )
        assert "missing or altered action rollups" in backend_support["notes"]
        assert "shader/kernel source translation from host runtime APIs" in (
            backend_support["notes"]
        )
        assert "malformed runtime-reference records" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_report_records_documented_migration_actions"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_report_records_runtime_reference_evidence"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_scan_report_records_build_system_runtime_reference"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_malformed_runtime_references"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_report_text_reports_truncated_migration_"
            "actions"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_project_report_applies_custom_sample_limits"
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


def test_project_runtime_integration_plan_is_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.runtime_integration_plan"]

    assert feature["category"] == "project"
    assert feature["name"] == "Runtime integration planning"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert "metadata-only crosstl-runtime-integration-plan" in (
            backend_support["notes"]
        )
        assert "explicit scope and non-goals" in backend_support["notes"]
        assert "runtime-loader-plan-v1" in backend_support["notes"]
        assert "diagnostic-only failed plans" in backend_support["notes"]
        assert "does not import compiler internals" in backend_support["notes"]
        assert "execute device code" in backend_support["notes"]
        assert "rewrite host application code" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_integration_builds_metadata_plan"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_integration_applies_runtime_reference_sample_limit"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_integration_invalid_report_is_diagnostic_only"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_plan_runtime_text_outputs_requests_and_actions"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_plan_runtime_rejects_negative_sample_limit"
        ) in backend_support["evidence"]


def test_project_runtime_artifact_manifest_is_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.runtime_artifact_manifest"]

    assert feature["category"] == "project"
    assert feature["name"] == "Runtime artifact manifest"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert "crosstl-runtime-artifact-manifest" in backend_support["notes"]
        assert "translated artifact consumption metadata" in (backend_support["notes"])
        assert "generated artifact hash and byte-size metadata" in (
            backend_support["notes"]
        )
        assert "source-map anchors" in backend_support["notes"]
        assert "source-remap sidecars" in backend_support["notes"]
        assert "runtime-loader-plan-v1" in backend_support["notes"]
        assert "diagnostic-only failed manifests" in backend_support["notes"]
        assert "does not generate runtime framework code" in backend_support["notes"]
        assert "execute device code" in backend_support["notes"]
        assert "rewrite host application code" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_build_runtime_artifact_manifest_from_project_report"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_runtime_artifact_manifest_invalid_report_is_diagnostic_only"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_runtime_manifest_text_outputs_artifacts_and_plan"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_runtime_manifest_json_writes_output"
        ) in backend_support["evidence"]


def test_project_runtime_package_handoff_is_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.runtime_package_handoff"]

    assert feature["category"] == "project"
    assert feature["name"] == "Runtime package handoff"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert "crosstl-runtime-package" in backend_support["notes"]
        assert "runtime artifact manifest" in backend_support["notes"]
        assert "validates artifact hash and byte-size metadata" in (
            backend_support["notes"]
        )
        assert "source-remap sidecars" in backend_support["notes"]
        assert "integration guide" in backend_support["notes"]
        assert "runtime-loader-plan-v1" in backend_support["notes"]
        assert "does not rewrite host application code" in backend_support["notes"]
        assert "execute device code" in backend_support["notes"]
        assert "install target SDKs" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_build_runtime_package_from_runtime_artifact_manifest"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_runtime_package_rejects_stale_artifact_hash"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_package_runtime_text_outputs_package_summary"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_package_runtime_json_writes_output"
        ) in backend_support["evidence"]


def test_project_runtime_package_inspection_is_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.runtime_package_inspection"]

    assert feature["category"] == "project"
    assert feature["name"] == "Runtime package inspection"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert "crosstl-runtime-package-inspection" in backend_support["notes"]
        assert "verifies copied packaged artifact" in backend_support["notes"]
        assert "source-remap sidecar" in backend_support["notes"]
        assert "ready and failed host binding records" in backend_support["notes"]
        assert "runtime-loader-plan-v1" in backend_support["notes"]
        assert "Inspection is read-only" in backend_support["notes"]
        assert "does not rewrite host application code" in backend_support["notes"]
        assert "execute device code" in backend_support["notes"]
        assert "install target SDKs" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_runtime_package_reports_ready_bindings"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_runtime_package_detects_missing_packaged_artifact"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_runtime_package_detects_packaged_artifact_hash_mismatch"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_runtime_package_detects_missing_source_remap"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_runtime_package_rejects_failed_package"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_runtime_package_text_outputs_readiness"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_runtime_package_json_writes_output"
        ) in backend_support["evidence"]


def test_project_runtime_host_binding_plan_is_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.runtime_host_binding_plan"]

    assert feature["category"] == "project"
    assert feature["name"] == "Runtime host binding plan"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert "crosstl-runtime-host-binding-plan" in backend_support["notes"]
        assert "runtime package manifest" in backend_support["notes"]
        assert "runtime package inspection readiness" in backend_support["notes"]
        assert "packageInspection" in backend_support["notes"]
        assert "bind-runtime-artifact" in backend_support["notes"]
        assert "does not emit host bind actions for failed package records" in (
            backend_support["notes"]
        )
        assert "review-runtime-references" in backend_support["notes"]
        assert "runtime-loader-plan-v1" in backend_support["notes"]
        assert "does not rewrite host application code" in backend_support["notes"]
        assert "execute device code" in backend_support["notes"]
        assert "install target SDKs" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_host_bindings_from_runtime_package"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_host_bindings_rejects_failed_package"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_host_bindings_rejects_missing_packaged_artifact"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_host_bindings_rejects_missing_source_remap"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_plan_host_bindings_text_outputs_actions"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_plan_host_bindings_text_rejects_stale_package"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_plan_host_bindings_json_writes_output"
        ) in backend_support["evidence"]


def test_project_runtime_adapter_plan_is_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.runtime_adapter_plan"]

    assert feature["category"] == "project"
    assert feature["name"] == "Runtime adapter plan"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert "crosstl-runtime-adapter-plan" in backend_support["notes"]
        assert "runtime package manifest" in backend_support["notes"]
        assert "package inspection diagnostics" in backend_support["notes"]
        assert "adapterKind" in backend_support["notes"]
        assert "artifactFormat" in backend_support["notes"]
        assert "requiredTools" in backend_support["notes"]
        assert "hostResponsibilities" in backend_support["notes"]
        assert "wire-runtime-adapter" in backend_support["notes"]
        assert "review-runtime-references" in backend_support["notes"]
        assert "does not rewrite host application code" in backend_support["notes"]
        assert "execute device code" in backend_support["notes"]
        assert "install target SDKs" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_adapters_from_runtime_package"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_adapters_rejects_failed_package"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_plan_runtime_adapters_text_outputs_adapters"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_plan_runtime_adapters_json_writes_output"
        ) in backend_support["evidence"]


def test_project_runtime_loader_manifest_is_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.runtime_loader_manifest"]

    assert feature["category"] == "project"
    assert feature["name"] == "Runtime loader manifest"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert "crosstl-runtime-loader-manifest" in backend_support["notes"]
        assert "runtime package manifest" in backend_support["notes"]
        assert "load units" in backend_support["notes"]
        assert "adapterKind" in backend_support["notes"]
        assert "artifactFormat" in backend_support["notes"]
        assert "source-remap handoff paths" in backend_support["notes"]
        assert "hostInterface" in backend_support["notes"]
        assert "requiredTools" in backend_support["notes"]
        assert "hostResponsibilities" in backend_support["notes"]
        assert "loadSteps" in backend_support["notes"]
        assert "resolve-host-interface-metadata" in backend_support["notes"]
        assert "does not rewrite host application code" in backend_support["notes"]
        assert "execute device code" in backend_support["notes"]
        assert "install target SDKs" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_build_runtime_loader_manifest_from_runtime_package"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_runtime_loader_manifest_uses_reflected_vulkan_host_interface"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_runtime_loader_manifest_text_outputs_load_units"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_runtime_loader_manifest_json_writes_output"
        ) in backend_support["evidence"]


def test_project_runtime_variant_registry_documents_execution_key_schema_v2():
    catalog = json.loads(
        (ROOT / "support" / "features.json").read_text(encoding="utf-8")
    )
    features = {feature["id"]: feature for feature in catalog["features"]}
    feature = features["project.runtime_variant_registry"]
    execution_evidence = {
        (
            "tests/test_translator/test_runtime_variant_registry.py::def "
            "test_runtime_variant_execution_keys_are_distinct_reorderable_and_exact"
        ),
        (
            "tests/test_translator/test_runtime_variant_registry.py::def "
            "test_runtime_variant_execution_uses_only_the_selected_entry"
        ),
        (
            "tests/test_translator/test_runtime_variant_registry.py::def "
            "test_runtime_variant_subgroup_width_distinguishes_exact_variants"
        ),
        (
            "tests/test_translator/test_runtime_variant_registry.py::def "
            "test_runtime_variant_key_round_trips_absent_execution_and_reordered_inputs"
        ),
        (
            "tests/test_translator/test_runtime_variant_registry.py::def "
            "test_runtime_variant_key_rejects_legacy_schema_with_migration_error"
        ),
    }

    assert feature["category"] == "project"
    assert feature["name"] == "Runtime variant registry"
    for backend_support in feature["support"].values():
        notes = backend_support["notes"]
        assert backend_support["status"] == "supported"
        assert "canonical crosstl-rvk2 exact keys" in notes
        assert "selected binding-interface entry point" in notes
        assert "workgroupSize and subgroupWidth" in notes
        assert "unavailable values remain null" in notes
        assert "exact available execution alternatives" in notes
        assert "legacy crosstl-rvk1 keys" in notes
        assert "regenerate both the key and registry" in notes
        assert "selection and packaging metadata" in notes
        assert "runtime execution and numerical parity are not established" in notes
        assert execution_evidence <= set(backend_support["evidence"])

    docs = (ROOT / "docs" / "source" / "project-porting.rst").read_text(
        encoding="utf-8"
    )
    normalized_docs = " ".join(docs.split())
    assert "runtime variant key schema is version 2" in normalized_docs
    assert "selected binding-interface entry point's execution identity" in (
        normalized_docs
    )
    assert "availableExecutionAlternatives" in normalized_docs
    assert "regenerate both the key and registry" in normalized_docs
    assert "numerical parity are not established by the registry" in normalized_docs


def test_project_runtime_test_manifest_is_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.runtime_test_manifest"]

    assert feature["category"] == "project"
    assert feature["name"] == "Runtime test manifest"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert "crosstl-project-runtime-test-manifest" in backend_support["notes"]
        assert "crosstl-project-runtime-fixture-metadata" in (backend_support["notes"])
        assert "crosstl-project-runtime-test-plan" in backend_support["notes"]
        assert "crosstl-project-runtime-test-report" in backend_support["notes"]
        assert "generate manifests" in backend_support["notes"]
        assert "maps artifact selectors to runtime adapters" in (
            backend_support["notes"]
        )
        assert "inputs" in backend_support["notes"]
        assert "expected outputs" in backend_support["notes"]
        assert "tolerances" in backend_support["notes"]
        assert "resource bindings" in backend_support["notes"]
        assert "function or specialization constants" in backend_support["notes"]
        assert "dispatch geometry" in backend_support["notes"]
        assert "platform requirements" in backend_support["notes"]
        assert "structured skip diagnostics" in backend_support["notes"]
        assert "incomplete fixture data" in backend_support["notes"]
        assert "ambiguous selectors" in backend_support["notes"]
        assert "toolchain/toolchainRuns logs" in backend_support["notes"]
        assert "native graphics and native compute" in backend_support["notes"]
        assert "does not rewrite host application code" in backend_support["notes"]
        assert "install target SDKs" in backend_support["notes"]
        assert (
            "tests/test_translator/test_runtime_verification.py::def "
            "test_parse_runtime_test_manifest_maps_adapters_and_platform_requirements"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_runtime_verification.py::def "
            "test_build_runtime_test_manifest_from_mlx_fixture_metadata"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_runtime_verification.py::def "
            "test_build_runtime_test_manifest_reports_ambiguous_fixture_selector"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_runtime_verification.py::def "
            "test_build_runtime_test_manifest_reports_incomplete_fixture_data"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_runtime_verification.py::def "
            "test_project_cli_runtime_test_manifest_text_outputs_generated_tests"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_runtime_verification.py::def "
            "test_plan_runtime_test_manifest_records_structured_skip_and_toolchain_logs"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_runtime_verification.py::def "
            "test_verify_runtime_test_manifest_reports_skipped_dependency_record"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_runtime_verification.py::def "
            "test_verify_runtime_test_manifest_runs_executor_and_links_failed_check"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_runtime_verification.py::def "
            "test_default_runtime_test_adapters_cover_native_platform_classes"
        ) in backend_support["evidence"]


def test_project_runtime_host_loader_scaffolds_is_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.runtime_host_loader_scaffolds"]

    assert feature["category"] == "project"
    assert feature["name"] == "Runtime host loader scaffolds"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert "crosstl-runtime-host-loader-scaffolds" in backend_support["notes"]
        assert "runtime loader manifest" in backend_support["notes"]
        assert "host-loader-scaffolds.json" in backend_support["notes"]
        assert "HOST_LOADERS.md" in backend_support["notes"]
        assert "target-scoped loader metadata files" in backend_support["notes"]
        assert "source-remap handoff paths" in backend_support["notes"]
        assert "hostInterface" in backend_support["notes"]
        assert "requiredTools" in backend_support["notes"]
        assert "hostResponsibilities" in backend_support["notes"]
        assert "loadSteps" in backend_support["notes"]
        assert "resolve-host-interface-metadata" in backend_support["notes"]
        assert "sanitizes target and load-unit derived output paths" in (
            backend_support["notes"]
        )
        assert "do not rewrite host application code" in backend_support["notes"]
        assert "execute device code" in backend_support["notes"]
        assert "install target SDKs" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_build_runtime_host_loader_scaffolds_from_loader_manifest"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_runtime_host_loader_scaffolds_report_blocked_units_without_unit_files"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_runtime_host_loader_scaffolds_reject_wrong_manifest_kind"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_runtime_host_loader_scaffolds_sanitize_target_and_unit_paths"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_scaffold_host_loaders_text_outputs_scaffolds"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_scaffold_host_loaders_json_writes_output"
        ) in backend_support["evidence"]


def test_project_runtime_host_loader_scaffold_inspection_is_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.runtime_host_loader_scaffold_inspection"]

    assert feature["category"] == "project"
    assert feature["name"] == "Runtime host loader scaffold inspection"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert "crosstl-runtime-host-loader-scaffolds-inspection" in (
            backend_support["notes"]
        )
        assert "inspect-host-loader-scaffolds" in backend_support["notes"]
        assert "host-loader-scaffolds.json" in backend_support["notes"]
        assert "HOST_LOADERS.md" in backend_support["notes"]
        assert "target-scoped loader metadata files" in backend_support["notes"]
        assert "checks scaffold identity" in backend_support["notes"]
        assert "adapterKind" in backend_support["notes"]
        assert "packagePath" in backend_support["notes"]
        assert "blocked load units without requiring target loader files" in (
            backend_support["notes"]
        )
        assert "structured diagnostics" in backend_support["notes"]
        assert "keeps inspection read-only" in backend_support["notes"]
        assert "does not rewrite host application code" in backend_support["notes"]
        assert "execute device code" in backend_support["notes"]
        assert "install target SDKs" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_runtime_host_loader_scaffolds_reports_ready_files"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_runtime_host_loader_scaffolds_detects_missing_unit_file"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_runtime_host_loader_scaffolds_reports_blocked_units"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_runtime_host_loader_scaffolds_rejects_wrong_manifest_kind"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_host_loader_scaffolds_text_outputs_readiness"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_host_loader_scaffolds_json_writes_output"
        ) in backend_support["evidence"]


def test_project_runtime_host_loader_consumption_plan_is_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.runtime_host_loader_consumption_plan"]

    assert feature["category"] == "project"
    assert feature["name"] == "Runtime host loader consumption plan"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert "crosstl-runtime-host-loader-consumption-plan" in (
            backend_support["notes"]
        )
        assert "plan-host-loader-consumption" in backend_support["notes"]
        assert "host loader scaffold manifest" in backend_support["notes"]
        assert "runs scaffold inspection before consumption planning" in (
            backend_support["notes"]
        )
        assert "reads ready target-scoped host loader unit JSON files" in (
            backend_support["notes"]
        )
        assert "promotes loadSteps into actionable consumption records" in (
            backend_support["notes"]
        )
        assert "requiredTools" in backend_support["notes"]
        assert "hostResponsibilities" in backend_support["notes"]
        assert "resolve-loader-scaffold-blockers" in backend_support["notes"]
        assert "failed scaffold inspection diagnostics" in backend_support["notes"]
        assert "keeps planning read-only" in backend_support["notes"]
        assert "does not rewrite host application code" in backend_support["notes"]
        assert "execute device code" in backend_support["notes"]
        assert "install target SDKs" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_host_loader_consumption_reports_ready_units"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_host_loader_consumption_carries_blocked_units"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_host_loader_consumption_rejects_failed_inspection"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_plan_host_loader_consumption_text_outputs_actions"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_plan_host_loader_consumption_json_writes_output"
        ) in backend_support["evidence"]


def test_project_runtime_host_integration_handoff_is_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.runtime_host_integration_handoff"]

    assert feature["category"] == "project"
    assert feature["name"] == "Runtime host integration handoff"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert "crosstl-runtime-host-integration-handoff" in backend_support["notes"]
        assert "host-integration-handoff" in backend_support["notes"]
        assert "host-integration.json" in backend_support["notes"]
        assert "HOST_INTEGRATION.md" in backend_support["notes"]
        assert "targets/*.integration.json" in backend_support["notes"]
        assert "loader units" in backend_support["notes"]
        assert "promoted consumption actions" in backend_support["notes"]
        assert "requiredTools" in backend_support["notes"]
        assert "hostResponsibilities" in backend_support["notes"]
        assert "blocked-unit records" in backend_support["notes"]
        assert "validates the input plan kind before writing files" in (
            backend_support["notes"]
        )
        assert "keeps the bundle metadata-only" in backend_support["notes"]
        assert "does not rewrite host application code" in backend_support["notes"]
        assert "execute device code" in backend_support["notes"]
        assert "install target SDKs" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_build_runtime_host_integration_handoff_writes_ready_bundle"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_build_runtime_host_integration_handoff_writes_blocked_bundle"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_build_runtime_host_integration_handoff_rejects_wrong_plan_kind"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_host_integration_handoff_text_outputs_bundle"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_host_integration_handoff_json_writes_output"
        ) in backend_support["evidence"]


def test_project_runtime_host_integration_handoff_inspection_is_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.runtime_host_integration_handoff_inspection"]

    assert feature["category"] == "project"
    assert feature["name"] == "Runtime host integration handoff inspection"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert (
            "crosstl-runtime-host-integration-handoff-inspection"
            in backend_support["notes"]
        )
        assert "inspect-host-integration-handoff" in backend_support["notes"]
        assert "host-integration.json" in backend_support["notes"]
        assert "HOST_INTEGRATION.md" in backend_support["notes"]
        assert "targets/*.integration.json" in backend_support["notes"]
        assert "parses target handoff files" in backend_support["notes"]
        assert "handoff status is blocked" in backend_support["notes"]
        assert "target, status, loader-unit count, and action-count consistency" in (
            backend_support["notes"]
        )
        assert "rejects absolute or outside-root generated paths" in (
            backend_support["notes"]
        )
        assert "structured diagnostics" in backend_support["notes"]
        assert "read-only and bundle-local" in backend_support["notes"]
        assert "does not rewrite host application code" in backend_support["notes"]
        assert "execute device code" in backend_support["notes"]
        assert "install target SDKs" in backend_support["notes"]
        assert "re-run host integration" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_runtime_host_integration_handoff_reports_ready_bundle"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_runtime_host_integration_handoff_detects_missing_target_file"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_runtime_host_integration_handoff_rejects_unsafe_generated_path"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_runtime_host_integration_handoff_detects_target_count_mismatch"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_runtime_host_integration_handoff_reports_blocked_bundle"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_inspect_runtime_host_integration_handoff_rejects_wrong_manifest_kind"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_host_integration_handoff_text_outputs_readiness"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_inspect_host_integration_handoff_json_writes_output"
        ) in backend_support["evidence"]


def test_project_runtime_host_integration_execution_plan_is_first_class_support_feature():
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    features = {feature["id"]: feature for feature in matrix["features"]}
    feature = features["project.runtime_host_integration_execution_plan"]

    assert feature["category"] == "project"
    assert feature["name"] == "Runtime host integration execution plan"
    assert set(feature["support"]) == {backend["id"] for backend in matrix["backends"]}
    for backend_support in feature["support"].values():
        assert backend_support["status"] == "supported"
        assert "crosstl-runtime-host-integration-execution-plan" in (
            backend_support["notes"]
        )
        assert "plan-host-integration-execution" in backend_support["notes"]
        assert "host integration handoff manifest" in backend_support["notes"]
        assert "runs handoff inspection before planning" in backend_support["notes"]
        assert "host-root readiness" in backend_support["notes"]
        assert "stable phase-ordered execution steps" in backend_support["notes"]
        assert "prepare-tools" in backend_support["notes"]
        assert "consume-loader" in backend_support["notes"]
        assert "load-artifact" in backend_support["notes"]
        assert "satisfy-host-responsibility" in backend_support["notes"]
        assert "resolve-blockers" in backend_support["notes"]
        assert "requiredTools" in backend_support["notes"]
        assert "hostResponsibilities" in backend_support["notes"]
        assert "blocked-step records" in backend_support["notes"]
        assert "failed handoff inspection" in backend_support["notes"]
        assert "host-root diagnostics" in backend_support["notes"]
        assert "does not rewrite host application code" in backend_support["notes"]
        assert "execute device code" in backend_support["notes"]
        assert "install target SDKs" in backend_support["notes"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_host_integration_execution_reports_ready_steps"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_host_integration_execution_carries_blocked_steps"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_host_integration_execution_rejects_failed_"
            "handoff_inspection"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_plan_runtime_host_integration_execution_rejects_missing_host_root"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_plan_host_integration_execution_text_outputs_steps"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_project_cli_plan_host_integration_execution_json_writes_output"
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
        assert backend_support["status"] == "supported"
        assert "per-include directory status records and status counts" in (
            backend_support["notes"]
        )
        assert "current include-directory resolved paths" in (backend_support["notes"])
        assert (
            "expected/actual include-directory status, resolved-path, and "
            "frontend-visibility validation context" in backend_support["notes"]
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
        assert "unit source hash and byte-size metadata" in backend_support["notes"]
        assert (
            "resolved include hash and byte-size metadata" in backend_support["notes"]
        )
        assert (
            "Resolved include dependency reports and inspection samples include "
            "byte-size metadata"
        ) in backend_support["notes"]
        assert "per-artifact includeDependencyProcessing metadata" in (
            backend_support["notes"]
        )
        assert (
            "backend-native status rollups by status, source backend, target, "
            "and variant"
        ) in backend_support["notes"]
        assert "forwarded for source frontends with include-path support" in (
            backend_support["notes"]
        )
        assert (
            "preserved for source frontends that delegate them to platform compilers"
            in backend_support["notes"]
        )
        assert (
            "GLSL/Vulkan and unresolved local/dynamic/outside-project forms "
            "are reported as rejected"
        ) in backend_support["notes"]
        assert (
            "source frontends without include-path support are explicitly reported"
            in (backend_support["notes"])
        )
        assert (
            "HLSL-style preprocessing preserves unresolved angle system includes"
            in (backend_support["notes"])
        )
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
        assert (
            "include-path processing inspection summaries include configured "
            "include-directory status records plus frontend-visible and "
            "inactive directory counts"
        ) in backend_support["notes"]
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
            "test_validate_project_report_rejects_stale_include_dir_frontend_"
            "visibility"
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

    source_backend_ids = {
        "cuda",
        "directx",
        "hip",
        "metal",
        "mojo",
        "opengl",
        "rust",
        "slang",
        "vulkan",
    }
    for backend_id, backend_support in feature["support"].items():
        if backend_id in source_backend_ids:
            assert backend_support["status"] == "supported"
        else:
            assert backend_support["status"] in {"partial", "supported"}
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
        assert (
            "Define-processing inspection summaries expose redacted project "
            "define names, deterministic define fingerprints, selected variant "
            "names, per-variant define records, and artifact effective define "
            "fingerprints without define values"
        ) in backend_support["notes"]
        assert "report CLI variant metadata" in backend_support["notes"]
        assert (
            "define-processing status, target, source-backend, and variant rollups"
            in backend_support["notes"]
        )
        assert "sampled define-processing artifact metadata" in (
            backend_support["notes"]
        )
        assert (
            "with define names, deterministic define fingerprints, and without "
            "define values"
        ) in backend_support["notes"]
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
        if backend_id in source_backend_ids:
            assert (
                "Project scanning classifies backend-native macro semantics"
                in backend_support["notes"]
            )
            assert "project.scan.unsupported-macro-form diagnostics" in (
                backend_support["notes"]
            )
            assert "macro.native missing-capability metadata" in (
                backend_support["notes"]
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
        if backend_id in source_backend_ids:
            assert (
                "tests/test_translator/test_project_translation.py::def "
                "test_scan_project_accepts_supported_native_macro_forms_across_"
                "source_frontends"
            ) in backend_support["evidence"]
            assert (
                "tests/test_translator/test_project_translation.py::def "
                "test_scan_project_reports_unsupported_macro_forms_across_source_"
                "frontends"
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
        assert "expected/actual artifact identity mismatch context" in (
            backend_support["notes"]
        )
        assert "stable repository-relative POSIX paths" in backend_support["notes"]
        assert "unit-target-variant artifact matrix entries" in (
            backend_support["notes"]
        )
        assert (
            "expected/actual boolean rollup mismatch context"
            in backend_support["notes"]
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
        assert "expected/actual unit source hash and byte-size mismatch context" in (
            backend_support["notes"]
        )
        assert "unregistered or non-canonical unit source backend names" in (
            backend_support["notes"]
        )
        assert "inconsistent unit and skipped source override provenance" in (
            backend_support["notes"]
        )
        assert "expected/actual source override provenance mismatch context" in (
            backend_support["notes"]
        )
        assert "artifact source hashes that do not match declared" in (
            backend_support["notes"]
        )
        assert (
            "expected/actual artifact source hash and byte-size mismatch context"
            in (backend_support["notes"])
        )
        assert "line-preserving source-map mappings" in (backend_support["notes"])
        assert "line-preserving source-map validation" in backend_support["notes"]
        assert "expected/actual line-preserving mapping mismatch context" in (
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
        assert "expected/actual source-map and source-remap mismatch context" in (
            backend_support["notes"]
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
            "with source/generated span, mapping-count, source and generated hash "
            "and byte-size metadata, sidecar hash plus byte-size metadata, "
            "failed validation status metadata, and custom limits"
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
        assert "expected/actual artifact identity mismatch context" in (
            backend_support["notes"]
        )
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
            "SARIF remapped diagnostics with sanitized diagnosticLocation "
            "and originalLocation result properties"
        ) in backend_support["notes"]
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
            "severity, target, and runtime-reference rollups, non-empty action "
            "targets, translated artifact target references, and canonical "
            "target declarations"
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
        assert "expected/actual project-config count mismatch context" in (
            backend_support["notes"]
        )
        assert "expected/actual report count and rollup mismatch context" in (
            backend_support["notes"]
        )
        assert "unit extension/path consistency" in backend_support["notes"]
        assert "artifactMatrix metadata" in backend_support["notes"]
        assert "scan-only artifactMatrix plans" in backend_support["notes"]
        assert "artifact matrix coverage" in backend_support["notes"]
        assert (
            "direct validation report artifact target, artifact source backend, "
            "artifact variant"
        ) in backend_support["notes"]
        assert "source-report-hash metadata" in backend_support["notes"]
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
        assert (
            "expected/actual validation status mismatch context"
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
            "test_validate_project_report_rejects_stale_source_root_scan_visibility"
        ) in backend_support["evidence"]
        assert (
            "tests/test_translator/test_project_translation.py::def "
            "test_validate_project_report_rejects_stale_include_dir_frontend_"
            "visibility"
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
        assert (
            "validation report schema/kind/generated-at and source-report-hash "
            "metadata"
        ) in backend_support["notes"]
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
            "test_validate_project_report_rejects_directx_artifact_toolchain_run_"
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
            "test_translate_project_preserves_external_corpus_entries_with_shared_path"
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
        "artifact_count": 4,
        "stale_count": 0,
        "stale_artifacts": [],
        "total_diff_line_count": 0,
    }
    assert {artifact["path"] for artifact in report["artifacts"]} == {
        "support/generated/support-matrix.json",
        "support/generated/graphics-backend-roadmap.json",
        "support/generated/project-porting-roadmap.json",
        "docs/source/support-matrix.rst",
    }
    assert all(artifact["diff"] == [] for artifact in report["artifacts"])


def test_generated_check_report_includes_stale_artifact_diff(tmp_path, monkeypatch):
    module = load_support_matrix_module()
    backends, features = _minimal_catalogs(module)
    matrix = module.build_matrix(backends, features)
    matrix_path = tmp_path / "support-matrix.json"
    roadmap_path = tmp_path / "graphics-backend-roadmap.json"
    project_roadmap_path = tmp_path / "project-porting-roadmap.json"
    docs_path = tmp_path / "support-matrix.rst"
    monkeypatch.setattr(module, "MATRIX_JSON_PATH", matrix_path)
    monkeypatch.setattr(module, "GRAPHICS_ROADMAP_JSON_PATH", roadmap_path)
    monkeypatch.setattr(
        module, "PROJECT_PORTING_ROADMAP_JSON_PATH", project_roadmap_path
    )
    monkeypatch.setattr(module, "DOCS_RST_PATH", docs_path)
    module.write_generated(matrix)
    matrix_path.write_text("{}\n", encoding="utf-8")

    report = module.build_generated_check_report(matrix)
    stale_artifacts = [
        artifact for artifact in report["artifacts"] if artifact["stale"]
    ]

    assert report["ok"] is False
    assert report["summary"] == {
        "artifact_count": 4,
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
    project_roadmap_path = tmp_path / "project-porting-roadmap.json"
    docs_path = tmp_path / "support-matrix.rst"
    monkeypatch.setattr(module, "MATRIX_JSON_PATH", matrix_path)
    monkeypatch.setattr(module, "GRAPHICS_ROADMAP_JSON_PATH", roadmap_path)
    monkeypatch.setattr(
        module, "PROJECT_PORTING_ROADMAP_JSON_PATH", project_roadmap_path
    )
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


def test_validate_backend_catalog_rejects_duplicate_target_aliases():
    module = load_support_matrix_module()
    directx = _backend("directx", ["hlsl"], ".hlsl")
    opengl = _backend("opengl", ["glsl"], ".glsl")
    directx["target_aliases"] = ["shared"]
    opengl["target_aliases"] = ["shared"]

    with pytest.raises(module.SupportMatrixError, match="Duplicate backend alias"):
        module.validate_backend_catalog({"backends": [directx, opengl]})


def test_backend_inventory_preserves_target_profiles():
    module = load_support_matrix_module()
    directx = _backend("directx", ["hlsl"], ".hlsl")
    directx["target_aliases"] = ["dx12"]
    directx["target_profiles"] = ["directx-12", "shader-model-6"]

    matrix = module.build_matrix(
        {"backends": [directx]},
        {
            "statuses": _status_descriptions(module),
            "features": [
                {
                    "id": "target.codegen",
                    "category": "target",
                    "name": "Code generation",
                    "description": "Emit target code.",
                    "support": {"directx": {"status": "partial"}},
                }
            ],
        },
    )

    assert matrix["backends"][0]["target_aliases"] == ["dx12"]
    assert matrix["backends"][0]["target_profiles"] == [
        "directx-12",
        "shader-model-6",
    ]


def test_validate_backend_catalog_accepts_target_only_backends():
    module = load_support_matrix_module()
    backends = {"backends": [_target_only_backend("webgl", ["webgl2"], ".webgl.glsl")]}

    backend_ids = module.validate_backend_catalog(backends)
    matrix = module.build_matrix(
        backends,
        {
            "statuses": _status_descriptions(module),
            "features": [
                {
                    "id": "target.codegen",
                    "category": "target",
                    "name": "Code generation",
                    "description": "Emit target code.",
                    "support": {"webgl": {"status": "partial"}},
                }
            ],
        },
    )

    assert backend_ids == {"webgl"}
    assert matrix["backends"][0]["source_kind"] == "target-only"
    assert matrix["backends"][0]["target_aliases"] == ["webgl2"]
    assert matrix["backends"][0]["native_backend"] == {
        "path": None,
        "exists": False,
    }


def test_target_only_backend_missing_source_feature_support_stays_auditable():
    module = load_support_matrix_module()
    backends = {"backends": [_target_only_backend("webgl", ["webgl2"], ".webgl.glsl")]}

    matrix = module.build_matrix(
        backends,
        {
            "statuses": _status_descriptions(module),
            "features": [
                {
                    "id": "target.codegen",
                    "category": "target",
                    "name": "Code generation",
                    "description": "Emit target code.",
                    "support": {"webgl": {"status": "partial"}},
                },
                {
                    "id": "source.parse",
                    "category": "source",
                    "name": "Source parsing",
                    "description": "Parse native source.",
                    "support": {},
                },
            ],
        },
    )

    source_feature = next(
        feature for feature in matrix["features"] if feature["id"] == "source.parse"
    )
    assert source_feature["support"]["webgl"]["status"] == "unknown"
    assert any(
        row["feature_id"] == "source.parse"
        and row["backend_id"] == "webgl"
        and row["status"] == "unknown"
        for row in matrix["backlog"]
    )


def test_validate_backend_catalog_requires_source_kind():
    module = load_support_matrix_module()
    backend = _backend("directx", ["hlsl"], ".hlsl")
    backend.pop("source_kind")

    with pytest.raises(module.SupportMatrixError, match="missing 'source_kind'"):
        module.validate_backend_catalog({"backends": [backend]})


def test_validate_backend_catalog_requires_native_backend_for_native_sources():
    module = load_support_matrix_module()
    backend = _backend("directx", ["hlsl"], ".hlsl")
    backend.pop("native_backend")

    with pytest.raises(module.SupportMatrixError, match="missing 'native_backend'"):
        module.validate_backend_catalog({"backends": [backend]})


def test_validate_backend_catalog_rejects_unknown_source_kind():
    module = load_support_matrix_module()
    backend = _target_only_backend("webgl", ["webgl2"], ".webgl.glsl")
    backend["source_kind"] = "frontendish"

    with pytest.raises(module.SupportMatrixError, match="source_kind"):
        module.validate_backend_catalog({"backends": [backend]})


def test_validate_backend_catalog_rejects_native_backend_for_target_only():
    module = load_support_matrix_module()
    backend = _target_only_backend("webgl", ["webgl2"], ".webgl.glsl")
    backend["native_backend"] = "crosstl/backend/GLSL"

    with pytest.raises(module.SupportMatrixError, match="must not define"):
        module.validate_backend_catalog({"backends": [backend]})


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
        "webgl",
        "wgsl",
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
    native_source_backend_ids = {
        backend["id"]
        for backend in catalog["backends"]
        if backend.get("source_kind", "native") == "native"
    }
    assert backend_ids == set(codegen.backend_names())
    assert native_source_backend_ids == set(codegen.source_backend_names())
    assert native_source_backend_ids.issubset(set(SOURCE_REGISTRY.names()))

    for backend in catalog["backends"]:
        backend_id = backend["id"]
        spec = codegen.get_backend(backend_id)
        assert spec is not None, f"{backend_id} is not registered"
        assert codegen.get_backend_extension(backend_id) == backend["target_extension"]
        assert tuple(backend.get("target_profiles", [])) == tuple(
            codegen.target_profiles(backend_id)
        )
        if backend_id in native_source_backend_ids:
            assert spec.source_registry_name == backend_id
            assert (
                SOURCE_REGISTRY.get_by_extension(backend["target_extension"]).name
                == backend_id
            )
        else:
            assert spec.source_registry_name is None

        for alias in backend.get("aliases", []):
            assert (
                codegen.normalize_backend_name(alias) == backend_id
            ), f"{backend_id} support alias is not accepted by codegen: {alias}"
            if backend_id in native_source_backend_ids:
                assert SOURCE_REGISTRY.get(alias).name == backend_id, (
                    f"{backend_id} support alias is not accepted by source registry: "
                    f"{alias}"
                )
        for alias in backend.get("target_aliases", []):
            assert (
                codegen.normalize_backend_name(alias) == backend_id
            ), f"{backend_id} target alias is not accepted by codegen: {alias}"


def test_project_porting_roadmap_is_focused_on_project_category():
    module = load_support_matrix_module()
    roadmap_path = ROOT / "support" / "generated" / "project-porting-roadmap.json"
    roadmap = json.loads(roadmap_path.read_text(encoding="utf-8"))
    matrix = json.loads(
        (ROOT / "support" / "generated" / "support-matrix.json").read_text(
            encoding="utf-8"
        )
    )

    project_features = [
        feature for feature in matrix["features"] if feature["category"] == "project"
    ]
    project_backlog = [
        item for item in matrix["backlog"] if item["category"] == "project"
    ]

    assert roadmap["view"]["id"] == "project_porting"
    assert roadmap["view"]["categories"] == ["project"]
    assert roadmap["view"]["backend_ids"] == [
        backend["id"] for backend in matrix["backends"]
    ]
    assert roadmap["summary"]["feature_count"] == len(roadmap["features"])
    assert roadmap["summary"]["feature_count"] == len(project_features)
    assert roadmap["summary"]["backend_count"] == len(matrix["backends"])
    assert roadmap["summary"]["backlog"]["backlog_count"] == len(roadmap["backlog"])
    assert roadmap["summary"]["backlog"]["backlog_count"] == len(project_backlog)
    assert all(feature["category"] == "project" for feature in roadmap["features"])
    assert all(item["category"] == "project" for item in roadmap["backlog"])

    expected_counts = {
        backend["id"]: {status: 0 for status in module.STATUS_ORDER}
        for backend in matrix["backends"]
    }
    for feature in project_features:
        for backend_id, support in feature["support"].items():
            expected_counts[backend_id][support["status"]] += 1
    assert roadmap["summary"]["status_counts"] == expected_counts


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
