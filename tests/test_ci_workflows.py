import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
CI_COVERAGE_SCRIPT = ROOT / "tools" / "ci_coverage.py"
PYTHON_VERSIONS = {"3.8", "3.9", "3.10", "3.11", "3.12", "3.13"}
RUNNER_OSES = {"ubuntu-latest", "windows-latest", "macOS-latest"}
BACKEND_TEST_MATRIX_NAMES = {
    "cuda": "CUDA",
    "directx": "directx",
    "hip": "HIP",
    "metal": "metal",
    "mojo": "mojo",
    "opengl": "GLSL",
    "rust": "rust",
    "slang": "slang",
    "vulkan": "SPIRV",
}
TRANSLATOR_TEST_MATRIX_NAMES = {
    **BACKEND_TEST_MATRIX_NAMES,
    "hip": "hip",
}


def _workflow_texts():
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(WORKFLOW_DIR.glob("*.yml"))
    }


def _load_ci_coverage_module():
    spec = importlib.util.spec_from_file_location("ci_coverage", CI_COVERAGE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _catalog_backend_ids():
    catalog = json.loads((ROOT / "support" / "backends.json").read_text())
    return {backend["id"] for backend in catalog["backends"]}


def _parse_matrix_values(raw):
    raw = raw.strip().strip("[]")
    return {
        item.strip().strip("\"'")
        for item in raw.split(",")
        if item.strip().strip("\"'")
    }


def _matrix_values(workflow_text, key):
    inline = re.search(
        rf"^\s*{re.escape(key)}:\s*(\[[^\n]+\])\s*$",
        workflow_text,
        flags=re.MULTILINE,
    )
    if inline:
        return _parse_matrix_values(inline.group(1))

    block = re.search(
        rf"^\s*{re.escape(key)}:\s*\n\s*\[\s*(.*?)\s*\]",
        workflow_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if block:
        return _parse_matrix_values(block.group(1))
    raise AssertionError(f"matrix key not found: {key}")


def test_ci_runs_the_complete_pytest_suite_on_pull_requests_and_pushes():
    workflows = _workflow_texts()
    full_suite = workflows.get("full-tests.yml", "")

    assert full_suite, "full-tests.yml must exist"
    assert re.search(r"\bpull_request\s*:", full_suite)
    assert re.search(r"\bpush\s*:", full_suite)
    assert "glslang-tools" in full_suite
    assert "brew install glslang" in full_suite
    assert "choco install vulkan-sdk --version=1.4.341" in full_suite
    assert "sdk.lunarg.com/sdk/download/$vulkanSdkVersion/windows" in full_suite
    assert (
        "--accept-licenses --default-answer --confirm-command install copy_only=1"
        in full_suite
    )
    assert "DirectXShaderCompiler/releases/download/v1.9.2602" in full_suite
    assert "linux_dxc_2026_02_20.x86_64.tar.gz" in full_suite
    assert "dxc_2026_02_20.zip" in full_suite
    assert "shader-validators:" in full_suite
    assert "macOS-latest" in full_suite
    assert "windows-latest" in full_suite
    assert re.search(r"python\s+-m\s+pytest\s+tests\b", full_suite)
    assert "test_external_shader_validators.py" in full_suite


def test_full_suite_keeps_required_compiler_smoke_coverage():
    workflows = _workflow_texts()
    full_suite = workflows.get("full-tests.yml", "")

    assert "compiler-smoke-linux:" in full_suite
    assert "Compiler Smoke (Linux CUDA/DXC/SPIR-V/Slang)" in full_suite
    assert "runs-on: ubuntu-24.04" in full_suite
    for tool in ("glslangValidator", "spirv-as", "spirv-val", "dxc", "slangc", "nvcc"):
        assert tool in full_suite
    assert "Jimver/cuda-toolkit@v0.2.35" in full_suite
    assert 'CUDA_VERSION: "13.2.0"' in full_suite
    assert "SLANG_VERSION: v2026.9.1" in full_suite
    assert "test_external_shader_validators.py" in full_suite

    assert "compiler-smoke-macos:" in full_suite
    assert "Compiler Smoke (macOS Metal)" in full_suite
    assert "runs-on: macOS-latest" in full_suite
    assert "xcrun -sdk macosx -f metal" in full_suite
    assert "test_shader_validation.py" in full_suite
    assert "test_metal_codegen.py" in full_suite
    assert "--junitxml support/generated/full-tests-pytest.xml" in full_suite
    assert "python tools/pytest_failure_summary.py" in full_suite
    assert "full-tests-failure-summary.json" in full_suite
    assert "shader-validators-failure-summary.json" in full_suite
    assert "compiler-smoke-linux-failure-summary.json" in full_suite
    assert "compiler-smoke-macos-failure-summary.json" in full_suite
    assert "--retry 5 --retry-all-errors --retry-delay 10" in full_suite
    assert 'Write-Warning "DXC download failed on attempt $attempt; retrying."' in (
        full_suite
    )
    shader_job = full_suite[
        full_suite.index("  shader-validators:") : full_suite.index(
            "  compiler-smoke-linux:"
        )
    ]
    assert "shell: bash" in shader_job


def test_ci_coverage_report_summarizes_required_workflow_dimensions():
    module = _load_ci_coverage_module()

    report = module.build_report()

    assert report["summary"] == {"ok": True, "errors": 0}
    assert report["catalog"]["backend_count"] == len(_catalog_backend_ids())
    assert module.validation_errors(report) == []
    assert report["workflows"]["runtime"]["workflow_count"] == len(_workflow_texts())
    assert report["workflows"]["runtime"]["job_count"] > 0
    assert (
        report["workflows"]["runtime"]["jobs_with_timeouts"]
        == report["workflows"]["runtime"]["job_count"]
    )
    assert report["workflows"]["runtime"]["missing_job_timeouts"] == {}
    assert report["workflows"]["runtime"]["invalid_job_timeouts"] == {}
    assert report["workflows"]["permissions"]["workflow_count"] == len(
        _workflow_texts()
    )
    assert all(report["workflows"]["permissions"]["explicit_permissions"].values())
    assert report["workflows"]["permissions"]["unexpected_write_permissions"] == {}
    assert (
        report["workflows"]["permissions"]["missing_required_write_permissions"] == {}
    )
    assert report["workflows"]["actions"]["workflow_count"] == len(_workflow_texts())
    assert report["workflows"]["actions"]["missing_node24_opt_in"] == []
    assert report["workflows"]["actions"]["mutable_refs"] == {}
    assert report["workflows"]["pull_request_target"]["workflows"] == [
        "pr-issue-links.yml"
    ]
    assert report["workflows"]["pull_request_target"]["unexpected_workflows"] == []
    assert all(
        report["workflows"]["pull_request_target"]["trusted_base_checkout"].values()
    )
    assert not any(
        report["workflows"]["pull_request_target"][
            "checkout_credentials_persist"
        ].values()
    )
    assert report["workflows"]["pull_request_target"]["head_context_markers"] == {}
    assert all(
        report["workflows"]["pull_request_target"]["support_traceability"].values()
    )
    assert all(
        report["workflows"]["pull_request_target"][
            "support_traceability_enforcement"
        ].values()
    )
    assert all(
        report["workflows"]["pull_request_target"]["support_closure_sync"].values()
    )
    assert all(
        report["workflows"]["pull_request_target"]["support_reference_sync"].values()
    )
    assert all(
        report["workflows"]["pull_request_target"][
            "github_token_scoped_to_sync"
        ].values()
    )
    assert report["workflows"]["backend_tests"]["components"]["missing"] == []
    assert all(report["workflows"]["backend_tests"]["failure_summary"].values())
    assert report["workflows"]["translator_tests"]["components"]["missing"] == []
    assert all(report["workflows"]["translator_tests"]["failure_summary"].values())
    assert report["workflows"]["translator_tests"]["general_frontend_suite"] is True
    assert all(report["workflows"]["docs"]["required_policies"].values())
    assert report["workflows"]["examples"]["python_versions"]["missing"] == []
    assert report["workflows"]["examples"]["oses"]["missing"] == []
    assert report["workflows"]["examples"]["backend_coverage"]["missing"] == []
    assert all(report["workflows"]["examples"]["required_policies"].values())
    assert report["workflows"]["examples"]["backend_specific_strict"] is True
    assert report["workflows"]["examples"]["stability_fails_on_regression"] is True
    assert all(report["workflows"]["full_tests"]["required_tools"].values())
    assert all(
        all(fields.values())
        for fields in report["workflows"]["full_tests"]["failure_summaries"].values()
    )
    assert all(report["workflows"]["support_matrix"]["required_policies"].values())
    assert (
        report["workflows"]["support_matrix"]["uploads_check_report_artifact"] is True
    )
    assert report["workflows"]["support_matrix"]["evidence_audit_on_failure"] is True
    assert (
        report["workflows"]["support_matrix"]["evidence_audit_after_validate"] is True
    )
    assert (
        report["workflows"]["support_matrix"]["evidence_audit_fails_on_missing"] is True
    )
    assert (
        report["workflows"]["support_matrix"]["uploads_evidence_report_artifact"]
        is True
    )
    assert (
        report["workflows"]["support_matrix"][
            "uploads_check_report_artifact_on_failure"
        ]
        is True
    )
    assert (
        report["workflows"]["support_matrix"]["check_report_artifact_retention"] is True
    )
    assert (
        report["workflows"]["support_matrix"]["check_report_upload_after_validate"]
        is True
    )
    assert (
        report["workflows"]["support_matrix"][
            "check_report_upload_after_evidence_audit"
        ]
        is True
    )
    assert report["workflows"]["support_matrix"]["uploads_docs_probe_artifact"] is True
    assert (
        report["workflows"]["support_matrix"]["uploads_docs_probe_artifact_on_failure"]
        is True
    )
    assert (
        report["workflows"]["support_matrix"]["docs_probe_artifact_retention"] is True
    )
    assert all(report["workflows"]["pr_issue_links"]["required_policies"].values())
    assert all(report["workflows"]["support_issue_sync"]["required_tests"].values())
    assert all(
        report["workflows"]["support_issue_sync"]["required_path_filters"].values()
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "uploads_ci_coverage_artifact_on_failure"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["ci_coverage_artifact_retention"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["writes_support_matrix_check_report"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["writes_support_evidence_report"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "support_evidence_report_fails_on_missing"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "support_evidence_report_after_validate"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["support_evidence_report_on_failure"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["uploads_support_matrix_check_report"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["uploads_support_evidence_report"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "uploads_support_matrix_check_report_on_failure"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "support_matrix_check_report_ignores_missing_files"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "support_matrix_check_report_retention"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "support_matrix_check_upload_after_validate"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "issue_sync_uses_support_matrix_check_report"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["writes_support_automation_summary"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "support_automation_summary_on_failure"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "appends_support_automation_summary_to_step_summary"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "support_automation_summary_emits_annotations"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "support_automation_summary_fails_on_attention"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "support_automation_summary_after_issue_sync"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["dry_run_writes_issue_plan"] is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["plans_issue_sync_before_mutation"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["checks_planned_action_budget"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["sync_replans_before_mutation"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["sync_checks_planned_action_budget"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["sync_writes_issue_summary"] is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["uploads_support_signal_artifact"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "uploads_support_signal_artifact_on_failure"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "support_signal_artifact_ignores_missing_files"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["support_signal_artifact_retention"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["support_signal_upload_after_extract"]
        is True
    )
    assert report["workflows"]["support_issue_sync"]["workflow_run_full_tests"] is True
    assert (
        report["workflows"]["support_issue_sync"]["workflow_run_backend_tests"] is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["workflow_run_translator_tests"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["downloads_test_failure_summaries"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "downloads_test_failure_summaries_on_workflow_run"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "test_failure_summary_download_non_blocking"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "writes_clean_complete_test_pytest_summary"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "clean_complete_test_pytest_summary_on_success"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "clean_complete_test_pytest_summary_after_download"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "support_signals_uses_pytest_failure_summaries"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["chooses_pytest_failure_closure_mode"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "issue_sync_uses_pytest_failure_closure_mode"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "uploads_pytest_failure_summary_inputs"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["uploads_issue_sync_report_artifact"]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "uploads_issue_sync_report_artifact_on_failure"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "issue_sync_report_artifact_ignores_missing_files"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"][
            "issue_sync_report_artifact_retention"
        ]
        is True
    )
    assert (
        report["workflows"]["support_issue_sync"]["issue_sync_report_upload_after_sync"]
        is True
    )


def test_workflows_opt_into_node24_action_runtime():
    workflows = _workflow_texts()

    for workflow_name, workflow in workflows.items():
        has_action_refs = re.search(
            r"^\s*(?:-\s*)?uses:\s+[^@\s]+@",
            workflow,
            re.MULTILINE,
        )
        if has_action_refs:
            assert (
                'FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: "true"' in workflow
            ), f"{workflow_name} must opt JS actions into Node 24"


def test_ci_coverage_reports_missing_node24_action_runtime_opt_in():
    module = _load_ci_coverage_module()
    workflows = {
        "enabled.yml": (
            """
name: Enabled

env:
  FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: "true"

jobs:
  check:
    steps:
      - uses: actions/checkout@v4
"""
        ),
        "missing.yml": (
            """
name: Missing

jobs:
  check:
    steps:
      - uses: actions/checkout@v4
"""
        ),
        "no-actions.yml": (
            """
name: No Actions

jobs:
  check:
    steps:
      - run: python -m pytest
"""
        ),
    }

    report = module.workflow_actions_report(workflows)

    assert report["node24_opt_in"] == {
        "enabled.yml": True,
        "missing.yml": False,
    }
    assert report["missing_node24_opt_in"] == ["missing.yml"]


def test_ci_coverage_check_command_passes():
    result = subprocess.run(
        [sys.executable, "tools/ci_coverage.py", "check"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "CI coverage check passed." in result.stdout


def test_ci_coverage_check_command_accepts_explicit_root():
    result = subprocess.run(
        [sys.executable, "tools/ci_coverage.py", "--root", str(ROOT), "check"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "CI coverage check passed." in result.stdout


def test_ci_coverage_summary_command_writes_markdown(tmp_path):
    output = tmp_path / "ci-coverage-report.md"

    result = subprocess.run(
        [
            sys.executable,
            "tools/ci_coverage.py",
            "summary",
            "--output",
            str(output),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    text = output.read_text(encoding="utf-8")
    assert "Wrote" in result.stdout
    assert "# CI Coverage Report" in text
    assert "Status: **pass**" in text
    assert "## Workflow Runtime" in text
    assert "## Workflow Permissions" in text
    assert "## Workflow Actions" in text
    assert "## Pull Request Target" in text
    assert (
        "| Workflow | Components | Python | OS | Fail-fast disabled | Failure summaries | Frontend suite |"
        in text
    )
    assert "backend-tests.yml" in text
    assert "translator-tests.yml" in text
    assert "## Documentation" in text
    assert "## Examples" in text
    assert "Backend-specific failures are fatal" in text
    assert "## Support Matrix" in text
    assert "Support matrix check artifact" in text
    assert "Support evidence artifact" in text
    assert "Check artifact upload on failure" in text
    assert "Documentation probe artifact" in text
    assert "Documentation probe upload on failure" in text
    assert "CI coverage artifact upload on failure" in text
    assert "Support matrix check report" in text
    assert "Support evidence report" in text
    assert "Issue sync uses support matrix check" in text
    assert "Support automation summary" in text
    assert "Support automation summary in step summary" in text
    assert "Planned action budget guard" in text
    assert "Sync replans before mutation" in text
    assert "Sync planned action budget guard" in text
    assert "Support signal artifact" in text
    assert "Clean Complete Test Suite pytest summary" in text
    assert "Pytest failure closure mode" in text
    assert "Issue sync report artifact" in text
    assert "PR path filters" in text


def test_ci_coverage_reports_missing_matrix_entries():
    module = _load_ci_coverage_module()
    report = module.build_report()
    report["workflows"]["backend_tests"]["components"]["actual"].remove("metal")
    report["workflows"]["backend_tests"]["components"]["missing"] = ["metal"]

    errors = module.validation_errors(report)

    assert any("backend-tests.yml components mismatch" in error for error in errors)


def test_ci_coverage_reports_missing_python_versions_and_oses():
    module = _load_ci_coverage_module()
    report = module.build_report()
    report["workflows"]["translator_tests"]["python_versions"]["actual"].remove("3.13")
    report["workflows"]["translator_tests"]["python_versions"]["missing"] = ["3.13"]
    report["workflows"]["translator_tests"]["oses"]["actual"].remove("windows-latest")
    report["workflows"]["translator_tests"]["oses"]["missing"] = ["windows-latest"]

    errors = module.validation_errors(report)

    assert any(
        "translator-tests.yml python_versions mismatch" in error for error in errors
    )
    assert any("translator-tests.yml oses mismatch" in error for error in errors)


def test_ci_coverage_reports_missing_translator_frontend_suite():
    module = _load_ci_coverage_module()
    report = module.build_report()
    report["workflows"]["translator_tests"]["general_frontend_suite"] = False

    errors = module.validation_errors(report)

    assert "translator-tests.yml must run the frontend general suite" in errors


def test_ci_coverage_reports_missing_matrix_failure_summary_policy():
    module = _load_ci_coverage_module()
    report = module.build_report()
    report["workflows"]["backend_tests"]["failure_summary"]["writes_junit"] = False
    report["workflows"]["translator_tests"]["failure_summary"][
        "uploads_failure_summary"
    ] = False

    errors = module.validation_errors(report)

    assert (
        "backend-tests.yml missing pytest failure summary policy: writes_junit"
        in errors
    )
    assert (
        "translator-tests.yml missing pytest failure summary policy: "
        "uploads_failure_summary"
    ) in errors


def test_ci_coverage_reports_missing_docs_policy():
    module = _load_ci_coverage_module()
    report = module.build_report()
    report["workflows"]["docs"]["required_policies"]["build_sphinx_html"] = False

    errors = module.validation_errors(report)

    assert "docs.yml missing policy: build_sphinx_html" in errors


def test_ci_coverage_reports_missing_examples_coverage_and_strictness():
    module = _load_ci_coverage_module()
    report = module.build_report()
    report["workflows"]["examples"]["python_versions"]["actual"].remove("3.13")
    report["workflows"]["examples"]["python_versions"]["missing"] = ["3.13"]
    report["workflows"]["examples"]["backend_coverage"]["actual"].remove("metal")
    report["workflows"]["examples"]["backend_coverage"]["missing"] = ["metal"]
    report["workflows"]["examples"]["required_policies"][
        "comprehensive_test_script"
    ] = False
    report["workflows"]["examples"]["backend_specific_strict"] = False
    report["workflows"]["examples"]["stability_fails_on_regression"] = False
    report["workflows"]["examples"]["diagnostic_continue_on_error_count"] = 2

    errors = module.validation_errors(report)

    assert any(
        "examples-test.yml python_versions mismatch" in error for error in errors
    )
    assert any(
        "examples-test.yml backend_coverage mismatch" in error for error in errors
    )
    assert "examples-test.yml missing policy: comprehensive_test_script" in errors
    assert "examples-test.yml backend-specific job must fail on errors" in errors
    assert "examples-test.yml stability job must fail on regression" in errors
    assert "examples-test.yml has too many continue-on-error steps: 2" in errors


def test_ci_coverage_reports_missing_compiler_smoke_tooling():
    module = _load_ci_coverage_module()
    report = module.build_report()
    report["workflows"]["full_tests"]["required_tools"]["slangc"] = False
    report["workflows"]["full_tests"]["required_markers"][
        "Compiler Smoke (Linux CUDA/DXC/SPIR-V/Slang)"
    ] = False
    report["workflows"]["full_tests"]["failure_summaries"]["compiler_smoke_linux"][
        "writes_failure_summary"
    ] = False
    report["workflows"]["full_tests"]["download_retries"]["dxc_linux"] = False
    report["workflows"]["full_tests"]["failure_summaries"]["shader_validators"][
        "run_shell_bash"
    ] = False

    errors = module.validation_errors(report)

    assert "full-tests.yml missing compiler tool coverage: slangc" in errors
    assert (
        "full-tests.yml missing marker: Compiler Smoke (Linux CUDA/DXC/SPIR-V/Slang)"
        in errors
    )
    assert (
        "full-tests.yml missing pytest failure summary for compiler_smoke_linux: "
        "writes_failure_summary"
    ) in errors
    assert "full-tests.yml missing external download retry: dxc_linux" in errors
    assert (
        "full-tests.yml missing pytest failure summary for shader_validators: "
        "run_shell_bash"
    ) in errors


def test_ci_coverage_reports_missing_support_planner_tests():
    module = _load_ci_coverage_module()
    report = module.build_report()
    report["workflows"]["support_issue_sync"]["required_tests"][
        "tests/test_support_matrix.py"
    ] = False
    report["workflows"]["support_issue_sync"]["min_desired_issues"] = False
    report["workflows"]["support_issue_sync"][
        "uploads_ci_coverage_artifact_on_failure"
    ] = False
    report["workflows"]["support_issue_sync"]["ci_coverage_artifact_retention"] = False
    report["workflows"]["support_issue_sync"][
        "writes_support_matrix_check_report"
    ] = False
    report["workflows"]["support_issue_sync"]["writes_support_evidence_report"] = False
    report["workflows"]["support_issue_sync"][
        "support_evidence_report_fails_on_missing"
    ] = False
    report["workflows"]["support_issue_sync"][
        "support_evidence_report_after_validate"
    ] = False
    report["workflows"]["support_issue_sync"][
        "support_evidence_report_on_failure"
    ] = False
    report["workflows"]["support_issue_sync"][
        "uploads_support_matrix_check_report"
    ] = False
    report["workflows"]["support_issue_sync"]["uploads_support_evidence_report"] = False
    report["workflows"]["support_issue_sync"][
        "uploads_support_matrix_check_report_on_failure"
    ] = False
    report["workflows"]["support_issue_sync"][
        "support_matrix_check_report_ignores_missing_files"
    ] = False
    report["workflows"]["support_issue_sync"][
        "support_matrix_check_report_retention"
    ] = False
    report["workflows"]["support_issue_sync"][
        "support_matrix_check_upload_after_validate"
    ] = False
    report["workflows"]["support_issue_sync"][
        "issue_sync_uses_support_matrix_check_report"
    ] = False
    report["workflows"]["support_issue_sync"][
        "writes_support_automation_summary"
    ] = False
    report["workflows"]["support_issue_sync"][
        "support_automation_summary_on_failure"
    ] = False
    report["workflows"]["support_issue_sync"][
        "appends_support_automation_summary_to_step_summary"
    ] = False
    report["workflows"]["support_issue_sync"][
        "support_automation_summary_emits_annotations"
    ] = False
    report["workflows"]["support_issue_sync"][
        "support_automation_summary_fails_on_attention"
    ] = False
    report["workflows"]["support_issue_sync"][
        "support_automation_summary_after_issue_sync"
    ] = False
    report["workflows"]["support_issue_sync"]["dry_run_writes_issue_plan"] = False
    report["workflows"]["support_issue_sync"][
        "plans_issue_sync_before_mutation"
    ] = False
    report["workflows"]["support_issue_sync"]["checks_planned_action_budget"] = False
    report["workflows"]["support_issue_sync"]["sync_replans_before_mutation"] = False
    report["workflows"]["support_issue_sync"][
        "sync_checks_planned_action_budget"
    ] = False
    report["workflows"]["support_issue_sync"]["sync_writes_issue_summary"] = False
    report["workflows"]["support_issue_sync"]["uploads_support_signal_artifact"] = False
    report["workflows"]["support_issue_sync"][
        "uploads_support_signal_artifact_on_failure"
    ] = False
    report["workflows"]["support_issue_sync"][
        "support_signal_artifact_ignores_missing_files"
    ] = False
    report["workflows"]["support_issue_sync"][
        "support_signal_artifact_retention"
    ] = False
    report["workflows"]["support_issue_sync"][
        "support_signal_upload_after_extract"
    ] = False
    report["workflows"]["support_issue_sync"]["workflow_run_full_tests"] = False
    report["workflows"]["support_issue_sync"]["workflow_run_backend_tests"] = False
    report["workflows"]["support_issue_sync"]["workflow_run_translator_tests"] = False
    report["workflows"]["support_issue_sync"][
        "downloads_test_failure_summaries"
    ] = False
    report["workflows"]["support_issue_sync"][
        "downloads_test_failure_summaries_on_workflow_run"
    ] = False
    report["workflows"]["support_issue_sync"][
        "test_failure_summary_download_non_blocking"
    ] = False
    report["workflows"]["support_issue_sync"][
        "writes_clean_complete_test_pytest_summary"
    ] = False
    report["workflows"]["support_issue_sync"][
        "clean_complete_test_pytest_summary_on_success"
    ] = False
    report["workflows"]["support_issue_sync"][
        "clean_complete_test_pytest_summary_after_download"
    ] = False
    report["workflows"]["support_issue_sync"][
        "support_signals_uses_pytest_failure_summaries"
    ] = False
    report["workflows"]["support_issue_sync"][
        "chooses_pytest_failure_closure_mode"
    ] = False
    report["workflows"]["support_issue_sync"][
        "issue_sync_uses_pytest_failure_closure_mode"
    ] = False
    report["workflows"]["support_issue_sync"][
        "uploads_pytest_failure_summary_inputs"
    ] = False
    report["workflows"]["support_issue_sync"][
        "uploads_issue_sync_report_artifact"
    ] = False
    report["workflows"]["support_issue_sync"][
        "uploads_issue_sync_report_artifact_on_failure"
    ] = False
    report["workflows"]["support_issue_sync"][
        "issue_sync_report_artifact_ignores_missing_files"
    ] = False
    report["workflows"]["support_issue_sync"][
        "issue_sync_report_artifact_retention"
    ] = False
    report["workflows"]["support_issue_sync"][
        "issue_sync_report_upload_after_sync"
    ] = False
    report["workflows"]["support_issue_sync"]["required_path_filters"][
        "crosstl/backend/**"
    ] = False

    errors = module.validation_errors(report)

    assert (
        "support-issue-sync.yml missing planner test: tests/test_support_matrix.py"
        in errors
    )
    assert "support-issue-sync.yml missing min_desired_issues" in errors
    assert (
        "support-issue-sync.yml missing uploads_ci_coverage_artifact_on_failure"
        in errors
    )
    assert "support-issue-sync.yml missing ci_coverage_artifact_retention" in errors
    assert "support-issue-sync.yml missing writes_support_matrix_check_report" in errors
    assert "support-issue-sync.yml missing writes_support_evidence_report" in errors
    assert (
        "support-issue-sync.yml missing support_evidence_report_fails_on_missing"
        in errors
    )
    assert (
        "support-issue-sync.yml missing support_evidence_report_after_validate"
        in errors
    )
    assert "support-issue-sync.yml missing support_evidence_report_on_failure" in errors
    assert (
        "support-issue-sync.yml missing uploads_support_matrix_check_report" in errors
    )
    assert "support-issue-sync.yml missing uploads_support_evidence_report" in errors
    assert (
        "support-issue-sync.yml missing uploads_support_matrix_check_report_on_failure"
        in errors
    )
    assert (
        "support-issue-sync.yml missing support_matrix_check_report_ignores_missing_files"
        in errors
    )
    assert (
        "support-issue-sync.yml missing support_matrix_check_report_retention" in errors
    )
    assert (
        "support-issue-sync.yml missing support_matrix_check_upload_after_validate"
        in errors
    )
    assert (
        "support-issue-sync.yml missing issue_sync_uses_support_matrix_check_report"
        in errors
    )
    assert "support-issue-sync.yml missing writes_support_automation_summary" in errors
    assert (
        "support-issue-sync.yml missing support_automation_summary_on_failure" in errors
    )
    assert (
        "support-issue-sync.yml missing appends_support_automation_summary_to_step_summary"
        in errors
    )
    assert (
        "support-issue-sync.yml missing support_automation_summary_emits_annotations"
        in errors
    )
    assert (
        "support-issue-sync.yml missing support_automation_summary_fails_on_attention"
        in errors
    )
    assert (
        "support-issue-sync.yml missing support_automation_summary_after_issue_sync"
        in errors
    )
    assert "support-issue-sync.yml missing dry_run_writes_issue_plan" in errors
    assert "support-issue-sync.yml missing plans_issue_sync_before_mutation" in errors
    assert "support-issue-sync.yml missing checks_planned_action_budget" in errors
    assert "support-issue-sync.yml missing sync_replans_before_mutation" in errors
    assert "support-issue-sync.yml missing sync_checks_planned_action_budget" in errors
    assert "support-issue-sync.yml missing sync_writes_issue_summary" in errors
    assert "support-issue-sync.yml missing uploads_support_signal_artifact" in errors
    assert (
        "support-issue-sync.yml missing uploads_support_signal_artifact_on_failure"
        in errors
    )
    assert (
        "support-issue-sync.yml missing support_signal_artifact_ignores_missing_files"
        in errors
    )
    assert "support-issue-sync.yml missing support_signal_artifact_retention" in errors
    assert (
        "support-issue-sync.yml missing support_signal_upload_after_extract" in errors
    )
    assert "support-issue-sync.yml missing workflow_run_full_tests" in errors
    assert "support-issue-sync.yml missing workflow_run_backend_tests" in errors
    assert "support-issue-sync.yml missing workflow_run_translator_tests" in errors
    assert "support-issue-sync.yml missing downloads_test_failure_summaries" in errors
    assert (
        "support-issue-sync.yml missing downloads_test_failure_summaries_on_workflow_run"
        in errors
    )
    assert (
        "support-issue-sync.yml missing test_failure_summary_download_non_blocking"
        in errors
    )
    assert (
        "support-issue-sync.yml missing writes_clean_complete_test_pytest_summary"
        in errors
    )
    assert (
        "support-issue-sync.yml missing clean_complete_test_pytest_summary_on_success"
        in errors
    )
    assert (
        "support-issue-sync.yml missing clean_complete_test_pytest_summary_after_download"
        in errors
    )
    assert (
        "support-issue-sync.yml missing support_signals_uses_pytest_failure_summaries"
        in errors
    )
    assert (
        "support-issue-sync.yml missing chooses_pytest_failure_closure_mode" in errors
    )
    assert (
        "support-issue-sync.yml missing issue_sync_uses_pytest_failure_closure_mode"
        in errors
    )
    assert (
        "support-issue-sync.yml missing uploads_pytest_failure_summary_inputs" in errors
    )
    assert "support-issue-sync.yml missing uploads_issue_sync_report_artifact" in errors
    assert (
        "support-issue-sync.yml missing uploads_issue_sync_report_artifact_on_failure"
        in errors
    )
    assert (
        "support-issue-sync.yml missing issue_sync_report_artifact_ignores_missing_files"
        in errors
    )
    assert (
        "support-issue-sync.yml missing issue_sync_report_artifact_retention" in errors
    )
    assert (
        "support-issue-sync.yml missing issue_sync_report_upload_after_sync" in errors
    )
    assert "support-issue-sync.yml missing path filter: crosstl/backend/**" in errors


def test_ci_coverage_reports_missing_support_matrix_policy():
    module = _load_ci_coverage_module()
    report = module.build_report()
    report["workflows"]["support_matrix"]["required_policies"][
        "docs_probe_command"
    ] = False
    report["workflows"]["support_matrix"]["uploads_check_report_artifact"] = False
    report["workflows"]["support_matrix"]["evidence_audit_on_failure"] = False
    report["workflows"]["support_matrix"]["evidence_audit_after_validate"] = False
    report["workflows"]["support_matrix"]["evidence_audit_fails_on_missing"] = False
    report["workflows"]["support_matrix"]["uploads_evidence_report_artifact"] = False
    report["workflows"]["support_matrix"][
        "uploads_check_report_artifact_on_failure"
    ] = False
    report["workflows"]["support_matrix"]["check_report_artifact_retention"] = False
    report["workflows"]["support_matrix"]["check_report_upload_after_validate"] = False
    report["workflows"]["support_matrix"][
        "check_report_upload_after_evidence_audit"
    ] = False
    report["workflows"]["support_matrix"]["uploads_docs_probe_artifact"] = False
    report["workflows"]["support_matrix"][
        "uploads_docs_probe_artifact_on_failure"
    ] = False
    report["workflows"]["support_matrix"]["docs_probe_artifact_retention"] = False

    errors = module.validation_errors(report)

    assert "support-matrix.yml missing policy: docs_probe_command" in errors
    assert "support-matrix.yml missing check report artifact upload" in errors
    assert "support-matrix.yml evidence audit must run on failure" in errors
    assert "support-matrix.yml evidence audit must run after validation" in errors
    assert "support-matrix.yml evidence audit must fail on missing evidence" in errors
    assert "support-matrix.yml missing evidence report artifact upload" in errors
    assert "support-matrix.yml check report artifact must upload on failure" in errors
    assert "support-matrix.yml check report artifact must set retention-days" in errors
    assert "support-matrix.yml check report upload must run after validation" in errors
    assert (
        "support-matrix.yml check report upload must run after evidence audit" in errors
    )
    assert "support-matrix.yml missing docs probe artifact upload" in errors
    assert "support-matrix.yml docs probe artifact must upload on failure" in errors
    assert "support-matrix.yml docs probe artifact must set retention-days" in errors


def test_ci_coverage_reports_missing_workflow_runtime_permission_and_action_policy():
    module = _load_ci_coverage_module()
    report = module.build_report()
    report["workflows"]["runtime"]["missing_job_timeouts"] = {
        "support-matrix.yml": ["check"]
    }
    report["workflows"]["runtime"]["invalid_job_timeouts"] = {
        "full-tests.yml": ["pytest"]
    }
    report["workflows"]["permissions"]["missing_explicit_permissions"] = ["docs.yml"]
    report["workflows"]["permissions"]["unexpected_write_permissions"] = {
        "backend-tests.yml": ["issues"]
    }
    report["workflows"]["permissions"]["missing_required_write_permissions"] = {
        "stale-prs.yml": ["pull-requests"]
    }
    report["workflows"]["actions"]["missing_node24_opt_in"] = ["docs.yml"]
    report["workflows"]["actions"]["mutable_refs"] = {
        "issue_assign.yml": ["bdougie/take-action@main"]
    }

    errors = module.validation_errors(report)

    assert "support-matrix.yml missing timeout-minutes for jobs: check" in errors
    assert "full-tests.yml has invalid timeout-minutes for jobs: pytest" in errors
    assert "docs.yml missing explicit permissions" in errors
    assert "backend-tests.yml has unexpected write permissions: issues" in errors
    assert "stale-prs.yml missing required write permissions: pull-requests" in errors
    assert (
        "docs.yml must set FORCE_JAVASCRIPT_ACTIONS_TO_NODE24 for JavaScript actions"
        in errors
    )
    assert (
        "issue_assign.yml has mutable action refs: bdougie/take-action@main" in errors
    )


def test_ci_coverage_reports_pull_request_target_trust_boundary_regressions():
    module = _load_ci_coverage_module()
    report = module.build_report()
    target = report["workflows"]["pull_request_target"]
    target["unexpected_workflows"] = ["new-unsafe.yml"]
    target["trusted_base_checkout"]["pr-issue-links.yml"] = False
    target["checkout_credentials_persist"]["pr-issue-links.yml"] = True
    target["head_context_markers"]["pr-issue-links.yml"] = ["github.head_ref"]
    target["support_traceability"]["pr-issue-links.yml"] = False
    target["support_traceability_enforcement"]["pr-issue-links.yml"] = False
    target["support_closure_sync"]["pr-issue-links.yml"] = False
    target["support_reference_sync"]["pr-issue-links.yml"] = False
    target["github_token_scoped_to_sync"]["pr-issue-links.yml"] = False

    errors = module.validation_errors(report)

    assert "new-unsafe.yml uses pull_request_target but is not allowlisted" in errors
    assert "pr-issue-links.yml pull_request_target must checkout trusted base" in errors
    assert (
        "pr-issue-links.yml pull_request_target checkout must not persist credentials"
        in errors
    )
    assert (
        "pr-issue-links.yml pull_request_target references PR head context: github.head_ref"
        in errors
    )
    assert (
        "pr-issue-links.yml pull_request_target must check support traceability"
        in errors
    )
    assert (
        "pr-issue-links.yml pull_request_target must enforce support traceability"
        in errors
    )
    assert (
        "pr-issue-links.yml pull_request_target must sync support issue closures"
        in errors
    )
    assert (
        "pr-issue-links.yml pull_request_target must sync support issue references"
        in errors
    )
    assert (
        "pr-issue-links.yml pull_request_target must scope GITHUB_TOKEN to sync step"
        in errors
    )


def test_ci_coverage_reports_missing_pr_issue_link_policy():
    module = _load_ci_coverage_module()
    report = module.build_report()
    report["workflows"]["pr_issue_links"]["required_policies"][
        "support_closure_sync"
    ] = False

    errors = module.validation_errors(report)

    assert "pr-issue-links.yml missing policy: support_closure_sync" in errors


def test_ci_coverage_reads_support_path_filters_only_from_pull_request_paths():
    module = _load_ci_coverage_module()
    workflow = (WORKFLOW_DIR / "support-issue-sync.yml").read_text(encoding="utf-8")
    workflow = workflow.replace('      - "crosstl/backend/**"\n', "")
    workflow += "\n# crosstl/backend/**\n"

    report = module.support_issue_sync_report(workflow)

    assert report["required_path_filters"]["crosstl/backend/**"] is False


def test_ci_coverage_reads_support_issue_sync_guards_from_their_steps():
    module = _load_ci_coverage_module()
    workflow = (WORKFLOW_DIR / "support-issue-sync.yml").read_text(encoding="utf-8")
    workflow = workflow.replace(
        "        if: github.event_name == 'pull_request'\n"
        "        run: >\n"
        "          python tools/sync_support_issues.py",
        "        if: github.event_name != 'pull_request'\n"
        "        run: >\n"
        "          python tools/sync_support_issues.py",
        1,
    )
    workflow += "\n# if: github.event_name == 'pull_request'\n# --dry-run\n"

    report = module.support_issue_sync_report(workflow)

    assert report["dry_run_on_pull_request"] is False


def test_ci_coverage_requires_support_signal_upload_after_extract_step():
    module = _load_ci_coverage_module()
    workflow = (WORKFLOW_DIR / "support-issue-sync.yml").read_text(encoding="utf-8")
    upload_step = module.workflow_step_section(
        workflow, "Upload support signal reports"
    )
    workflow = workflow.replace(upload_step, "")
    workflow = workflow.replace(
        "      - name: Extract generated support signals\n",
        upload_step + "\n      - name: Extract generated support signals\n",
    )

    report = module.support_issue_sync_report(workflow)

    assert report["uploads_support_signal_artifact"] is True
    assert report["support_signal_upload_after_extract"] is False


def test_ci_coverage_requires_issue_sync_report_upload_after_sync_step():
    module = _load_ci_coverage_module()
    workflow = (WORKFLOW_DIR / "support-issue-sync.yml").read_text(encoding="utf-8")
    upload_step = module.workflow_step_section(
        workflow, "Upload support issue sync reports"
    )
    workflow = workflow.replace(upload_step, "")
    workflow = workflow.replace(
        "      - name: Dry-run issue sync\n",
        upload_step + "\n      - name: Dry-run issue sync\n",
    )

    report = module.support_issue_sync_report(workflow)

    assert report["uploads_issue_sync_report_artifact"] is True
    assert report["issue_sync_report_upload_after_sync"] is False


def test_ci_coverage_requires_support_summary_on_failure():
    module = _load_ci_coverage_module()
    workflow = (WORKFLOW_DIR / "support-issue-sync.yml").read_text(encoding="utf-8")
    summary_step = module.workflow_step_section(
        workflow,
        "Write support automation summary",
    )
    workflow = workflow.replace(
        summary_step, summary_step.replace("        if: always()\n", "")
    )
    workflow += "\n# if: always()\n"

    report = module.support_issue_sync_report(workflow)

    assert report["writes_support_automation_summary"] is True
    assert report["support_automation_summary_on_failure"] is False


def test_ci_coverage_reads_pull_request_target_guards_from_trusted_checkout_step():
    module = _load_ci_coverage_module()
    workflow = (WORKFLOW_DIR / "pr-issue-links.yml").read_text(encoding="utf-8")
    workflow = workflow.replace("          persist-credentials: false\n", "")
    workflow += "\n# persist-credentials: false\n"

    report = module.pull_request_target_report({"pr-issue-links.yml": workflow})

    assert report["checkout_credentials_persist"]["pr-issue-links.yml"] is True


def test_ci_coverage_report_summary_reflects_validation_failures():
    module = _load_ci_coverage_module()
    report = module.build_report()
    broken = copy.deepcopy(report)
    broken["workflows"]["backend_tests"]["components"]["actual"].remove("metal")
    broken["workflows"]["backend_tests"]["components"]["missing"] = ["metal"]
    errors = module.validation_errors(broken)

    broken["summary"] = {
        "ok": not errors,
        "errors": len(errors),
    }

    assert broken["summary"] == {"ok": False, "errors": 1}


def test_ci_coverage_comparison_reports_removed_coverage():
    module = _load_ci_coverage_module()
    baseline = module.build_report()
    current = copy.deepcopy(baseline)
    current["workflows"]["backend_tests"]["components"]["actual"].remove("metal")

    comparison = module.build_ci_coverage_comparison(baseline, current)

    assert comparison["summary"] == {
        "ok": False,
        "shrink_count": 1,
        "growth_count": 0,
    }
    assert comparison["shrinks"] == [
        {
            "scope": "backend-tests.yml",
            "dimension": "components",
            "removed": ["metal"],
            "added": [],
        }
    ]


def test_ci_coverage_comparison_reports_workflow_policy_shrink():
    module = _load_ci_coverage_module()
    baseline = module.build_report()
    current = copy.deepcopy(baseline)
    current["workflows"]["runtime"]["job_timeouts"]["support-matrix.yml"][
        "check"
    ] = None
    current["workflows"]["permissions"]["explicit_permissions"]["docs.yml"] = False
    current["workflows"]["actions"]["mutable_refs"] = {
        "issue_assign.yml": ["bdougie/take-action@main"]
    }
    current["workflows"]["pull_request_target"]["checkout_credentials_persist"][
        "pr-issue-links.yml"
    ] = True

    comparison = module.build_ci_coverage_comparison(baseline, current)

    assert comparison["summary"] == {
        "ok": False,
        "shrink_count": 4,
        "growth_count": 0,
    }
    assert {
        "scope": "workflows",
        "dimension": "job_timeouts",
        "removed": ["support-matrix.yml:check"],
        "added": [],
    } in comparison["shrinks"]
    assert {
        "scope": "workflows",
        "dimension": "explicit_permissions",
        "removed": ["docs.yml"],
        "added": [],
    } in comparison["shrinks"]
    assert {
        "scope": "workflows",
        "dimension": "action_ref_policy",
        "removed": ["issue_assign.yml"],
        "added": [],
    } in comparison["shrinks"]
    assert {
        "scope": "workflows",
        "dimension": "pull_request_target_policy",
        "removed": ["pr-issue-links.yml:no_persisted_checkout_credentials"],
        "added": [],
    } in comparison["shrinks"]


def test_ci_coverage_comparison_reports_full_suite_download_retry_shrink():
    module = _load_ci_coverage_module()
    baseline = module.build_report()
    current = copy.deepcopy(baseline)
    current["workflows"]["full_tests"]["download_retries"]["dxc_linux"] = False

    comparison = module.build_ci_coverage_comparison(baseline, current)

    assert comparison["summary"] == {
        "ok": False,
        "shrink_count": 1,
        "growth_count": 0,
    }
    assert {
        "scope": "full-tests.yml",
        "dimension": "download_retries",
        "removed": ["dxc_linux"],
        "added": [],
    } in comparison["shrinks"]


def test_ci_coverage_comparison_reports_added_coverage_without_shrink():
    module = _load_ci_coverage_module()
    baseline = module.build_report()
    current = copy.deepcopy(baseline)
    current["workflows"]["backend_tests"]["components"]["actual"].append("new-backend")

    comparison = module.build_ci_coverage_comparison(baseline, current)

    assert comparison["summary"] == {
        "ok": True,
        "shrink_count": 0,
        "growth_count": 1,
    }
    assert comparison["growth"][0]["added"] == ["new-backend"]


def test_ci_coverage_comparison_reports_support_matrix_policy_shrink():
    module = _load_ci_coverage_module()
    baseline = module.build_report()
    current = copy.deepcopy(baseline)
    current["workflows"]["support_matrix"]["required_policies"][
        "daily_schedule"
    ] = False

    comparison = module.build_ci_coverage_comparison(baseline, current)

    assert comparison["summary"]["ok"] is False
    assert comparison["shrinks"] == [
        {
            "scope": "support-matrix.yml",
            "dimension": "required_policies",
            "removed": ["daily_schedule"],
            "added": [],
        }
    ]


def test_ci_coverage_comparison_reports_support_summary_policy_shrink():
    module = _load_ci_coverage_module()
    baseline = module.build_report()
    current = copy.deepcopy(baseline)
    current["workflows"]["support_issue_sync"][
        "support_automation_summary_emits_annotations"
    ] = False
    current["workflows"]["support_issue_sync"][
        "support_automation_summary_fails_on_attention"
    ] = False

    comparison = module.build_ci_coverage_comparison(baseline, current)

    assert comparison["summary"] == {
        "ok": False,
        "shrink_count": 2,
        "growth_count": 0,
    }
    assert {
        "scope": "support-issue-sync.yml",
        "dimension": "support_automation_summary_emits_annotations",
        "removed": ["support_automation_summary_emits_annotations"],
        "added": [],
    } in comparison["shrinks"]
    assert {
        "scope": "support-issue-sync.yml",
        "dimension": "support_automation_summary_fails_on_attention",
        "removed": ["support_automation_summary_fails_on_attention"],
        "added": [],
    } in comparison["shrinks"]


def test_ci_coverage_comparison_reports_translator_frontend_suite_shrink():
    module = _load_ci_coverage_module()
    baseline = module.build_report()
    current = copy.deepcopy(baseline)
    current["workflows"]["translator_tests"]["general_frontend_suite"] = False

    comparison = module.build_ci_coverage_comparison(baseline, current)

    assert comparison["summary"]["ok"] is False
    assert {
        "scope": "translator-tests.yml",
        "dimension": "general_frontend_suite",
        "removed": ["general_frontend_suite"],
        "added": [],
    } in comparison["shrinks"]


def test_ci_coverage_comparison_reports_matrix_failure_summary_shrink():
    module = _load_ci_coverage_module()
    baseline = module.build_report()
    current = copy.deepcopy(baseline)
    current["workflows"]["backend_tests"]["failure_summary"][
        "uploads_failure_summary"
    ] = False

    comparison = module.build_ci_coverage_comparison(baseline, current)

    assert comparison["summary"]["ok"] is False
    assert {
        "scope": "backend-tests.yml",
        "dimension": "failure_summary.uploads_failure_summary",
        "removed": ["failure_summary.uploads_failure_summary"],
        "added": [],
    } in comparison["shrinks"]


def test_ci_coverage_comparison_reports_examples_backend_shrink():
    module = _load_ci_coverage_module()
    baseline = module.build_report()
    current = copy.deepcopy(baseline)
    current["workflows"]["examples"]["backend_coverage"]["actual"].remove("metal")

    comparison = module.build_ci_coverage_comparison(baseline, current)

    assert {
        "scope": "examples-test.yml",
        "dimension": "backend_coverage",
        "removed": ["metal"],
        "added": [],
    } in comparison["shrinks"]


def test_ci_coverage_compare_command_fails_on_shrink(tmp_path):
    module = _load_ci_coverage_module()
    baseline = module.build_report()
    current = copy.deepcopy(baseline)
    current["workflows"]["full_tests"]["required_tools"]["dxc"] = False
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    baseline_path.write_text(module.stable_json(baseline), encoding="utf-8")
    current_path.write_text(module.stable_json(current), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "tools/ci_coverage.py",
            "compare",
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--fail-on-shrink",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    comparison = json.loads(result.stdout)
    assert result.returncode == 1
    assert comparison["summary"]["shrink_count"] == 1
    assert comparison["shrinks"][0]["removed"] == ["dxc"]
    assert "removed coverage" in result.stderr


def test_backend_and_translator_compatibility_matrices_remain_enabled():
    workflows = _workflow_texts()

    backend_tests = workflows.get("backend-tests.yml", "")
    translator_tests = workflows.get("translator-tests.yml", "")

    assert "python-version" in backend_tests
    assert "OS:" in backend_tests
    assert "pytest tests/test_backend/test_${{ matrix.backend }}" in backend_tests

    assert "python-version" in translator_tests
    assert "OS:" in translator_tests
    assert "pytest tests/test_translator" in translator_tests


def test_backend_test_matrix_matches_support_catalog_and_platform_policy():
    workflows = _workflow_texts()
    backend_tests = workflows.get("backend-tests.yml", "")
    assert backend_tests, "backend-tests.yml must exist"

    assert set(BACKEND_TEST_MATRIX_NAMES) == _catalog_backend_ids()
    assert _matrix_values(backend_tests, "backend") == set(
        BACKEND_TEST_MATRIX_NAMES.values()
    )
    assert _matrix_values(backend_tests, "python-version") == PYTHON_VERSIONS
    assert _matrix_values(backend_tests, "OS") == RUNNER_OSES
    assert "fail-fast: false" in backend_tests
    assert "max-parallel: 24" in backend_tests
    assert "id: setup_python" in backend_tests
    assert "continue-on-error: true" in backend_tests
    assert "name: Classify Python setup failure" in backend_tests
    assert "steps.setup_python.outcome == 'failure'" in backend_tests
    assert "workflow=Backend Tests" in backend_tests
    assert "Classification: setup infrastructure before project tests" in backend_tests
    assert "id: run_backend_tests" in backend_tests
    assert "python -m pytest tests/test_backend/test_${{ matrix.backend }}" in (
        backend_tests
    )
    assert (
        "--junitxml support/generated/backend-tests-${{ matrix.backend }}-${{ matrix.python-version }}-${{ matrix.OS }}.xml"
        in backend_tests
    )
    assert "name: Write backend test failure summary" in backend_tests
    assert "failure() && steps.run_backend_tests.outcome == 'failure'" in backend_tests
    assert "python tools/pytest_failure_summary.py" in backend_tests
    assert (
        "--json-output support/generated/backend-tests-${{ matrix.backend }}-${{ matrix.python-version }}-${{ matrix.OS }}-failure-summary.json"
        in backend_tests
    )
    assert (
        'cat support/generated/backend-tests-${{ matrix.backend }}-${{ matrix.python-version }}-${{ matrix.OS }}-failure-summary.md >> "$GITHUB_STEP_SUMMARY"'
        in backend_tests
    )
    assert "name: Upload backend test failure summary" in backend_tests
    assert (
        "name: backend-tests-failure-summary-${{ matrix.backend }}-${{ matrix.python-version }}-${{ matrix.OS }}"
        in backend_tests
    )
    assert "if-no-files-found: ignore" in backend_tests
    assert "retention-days: 30" in backend_tests


def test_translator_test_matrix_matches_support_catalog_and_frontend_policy():
    workflows = _workflow_texts()
    translator_tests = workflows.get("translator-tests.yml", "")
    assert translator_tests, "translator-tests.yml must exist"

    expected_components = set(TRANSLATOR_TEST_MATRIX_NAMES.values()) | {"general"}
    assert set(TRANSLATOR_TEST_MATRIX_NAMES) == _catalog_backend_ids()
    assert _matrix_values(translator_tests, "component") == expected_components
    assert _matrix_values(translator_tests, "python-version") == PYTHON_VERSIONS
    assert _matrix_values(translator_tests, "OS") == RUNNER_OSES
    assert "fail-fast: false" in translator_tests
    assert "max-parallel: 24" in translator_tests
    assert "id: setup_python" in translator_tests
    assert "continue-on-error: true" in translator_tests
    assert "name: Classify Python setup failure" in translator_tests
    assert "steps.setup_python.outcome == 'failure'" in translator_tests
    assert "workflow=Translator Tests" in translator_tests
    assert "Classification: setup infrastructure before project tests" in (
        translator_tests
    )
    assert "id: run_translator_tests" in translator_tests
    assert 'if [ "${{ matrix.component }}" == "general" ]; then' in translator_tests
    assert (
        "python -m pytest tests/test_translator --ignore=tests/test_translator/test_codegen"
        in translator_tests
    )
    assert "pytest tests/test_translator/test_lexer.py" not in translator_tests
    assert (
        "python -m pytest tests/test_translator/test_codegen/test_${{ matrix.component }}_codegen.py"
        in translator_tests
    )
    assert (
        "--junitxml support/generated/translator-tests-${{ matrix.component }}-${{ matrix.python-version }}-${{ matrix.OS }}.xml"
        in translator_tests
    )
    assert "name: Write translator test failure summary" in translator_tests
    assert (
        "failure() && steps.run_translator_tests.outcome == 'failure'"
        in translator_tests
    )
    assert "python tools/pytest_failure_summary.py" in translator_tests
    assert (
        "--json-output support/generated/translator-tests-${{ matrix.component }}-${{ matrix.python-version }}-${{ matrix.OS }}-failure-summary.json"
        in translator_tests
    )
    assert (
        'cat support/generated/translator-tests-${{ matrix.component }}-${{ matrix.python-version }}-${{ matrix.OS }}-failure-summary.md >> "$GITHUB_STEP_SUMMARY"'
        in translator_tests
    )
    assert "name: Upload translator test failure summary" in translator_tests
    assert (
        "name: translator-tests-failure-summary-${{ matrix.component }}-${{ matrix.python-version }}-${{ matrix.OS }}"
        in translator_tests
    )
    assert "if-no-files-found: ignore" in translator_tests
    assert "retention-days: 30" in translator_tests


def test_support_matrix_workflow_runs_daily_checks_and_docs_probe():
    workflows = _workflow_texts()
    support_matrix = workflows.get("support-matrix.yml", "")

    assert support_matrix, "support-matrix.yml must exist"
    assert re.search(r"\bpush\s*:", support_matrix)
    assert re.search(r"\bpull_request\s*:", support_matrix)
    assert re.search(r"\bschedule\s*:", support_matrix)
    assert 'cron: "17 3 * * *"' in support_matrix
    assert "workflow_dispatch:" in support_matrix
    assert "python tools/support_matrix.py check" in support_matrix
    assert "--output support/generated/support-matrix-check.json" in support_matrix
    assert "python tools/support_matrix.py evidence" in support_matrix
    assert "--status supported" in support_matrix
    assert "--evidence missing" in support_matrix
    assert "--output support/generated/support-evidence-check.json" in support_matrix
    assert "name: Upload support matrix check reports" in support_matrix
    assert "name: support-matrix-check-report" in support_matrix
    assert "support/generated/support-matrix-check.json" in support_matrix
    assert "support/generated/support-evidence-check.json" in support_matrix
    assert "docs-probe:" in support_matrix
    assert "github.event_name == 'schedule'" in support_matrix
    assert "github.event_name == 'workflow_dispatch'" in support_matrix
    assert (
        "python tools/support_matrix.py docs --output "
        "support/generated/backend-docs-report.json"
    ) in support_matrix
    assert "actions/upload-artifact@v4" in support_matrix
    assert "if: always()" in support_matrix
    assert "retention-days: 30" in support_matrix


def test_docs_workflow_builds_doxygen_and_sphinx():
    workflows = _workflow_texts()
    docs = workflows.get("docs.yml", "")

    assert docs, "docs.yml must exist"
    assert re.search(r"\bpush\s*:", docs)
    assert re.search(r"\bpull_request\s*:", docs)
    assert "workflow_dispatch:" in docs
    assert 'python-version: "3.12"' in docs
    assert "sudo apt-get update && sudo apt-get install -y doxygen" in docs
    assert "pip install -r docs/requirements.txt" in docs
    assert "make -C docs doxygen" in docs
    assert "make -C docs html" in docs


def test_examples_workflow_enforces_backend_outputs_and_platform_matrix():
    workflows = _workflow_texts()
    examples = workflows.get("examples-test.yml", "")

    assert examples, "examples-test.yml must exist"
    assert re.search(r"\bpush\s*:", examples)
    assert re.search(r"\bpull_request\s*:", examples)
    assert "workflow_dispatch:" in examples
    assert _matrix_values(examples, "python-version") == PYTHON_VERSIONS
    assert _matrix_values(examples, "os") == RUNNER_OSES
    for backend in BACKEND_TEST_MATRIX_NAMES:
        assert f'backend: "{backend}"' in examples
    assert "python test.py" in examples
    assert "backend-specific:" in examples
    assert "stability-test:" in examples
    assert 'echo "[ERROR] Output file not created: $OUTPUT_FILE"' in examples
    assert 'echo "[ERROR] Output file is too small ($FILE_SIZE bytes)"' in examples
    assert examples.count("continue-on-error: true") == 1
    assert examples.count("raise SystemExit(1)") >= 2
    assert "--summary-json" in examples
    assert 'summary["within_regression_budget"]' in examples
    assert "Example regression detected" in examples
    assert re.search(
        r"- name: Create output directory\n"
        r"\s+run:\s+\|\n"
        r"\s+mkdir -p output/\$\{\{ matrix\.combination\.category \}\}\n"
        r"\s+shell: bash\n"
        r"\s+working-directory: examples",
        examples,
    )


def test_support_issue_sync_workflow_validates_and_creates_managed_issues():
    workflows = _workflow_texts()
    issue_sync = workflows.get("support-issue-sync.yml", "")

    assert issue_sync, "support-issue-sync.yml must exist"
    assert re.search(r"\bschedule\s*:", issue_sync)
    assert 'cron: "17 * * * *"' in issue_sync
    assert "workflow_dispatch:" in issue_sync
    assert re.search(r"\bpull_request\s*:", issue_sync)
    assert "issues: write" in issue_sync
    assert '".github/workflows/backend-tests.yml"' in issue_sync
    assert '".github/workflows/docs.yml"' in issue_sync
    assert '".github/workflows/examples-test.yml"' in issue_sync
    assert '".github/workflows/full-tests.yml"' in issue_sync
    assert '".github/workflows/issue_assign.yml"' in issue_sync
    assert '".github/workflows/support-matrix.yml"' in issue_sync
    assert '".github/workflows/stale-prs.yml"' in issue_sync
    assert '".github/workflows/translator-tests.yml"' in issue_sync
    required_path_filters = [
        '"crosstl/backend/**"',
        '"crosstl/translator/ast.py"',
        '"crosstl/translator/codegen/**"',
        '"crosstl/translator/lexer.py"',
        '"crosstl/translator/parser.py"',
        '"crosstl/translator/validation.py"',
        '"docs/source/support-matrix.rst"',
        '"examples/test.py"',
        '"tests/test_backend/**"',
        '"tests/test_examples_test_script.py"',
        '"tests/test_translator/test_ast_ir_contracts.py"',
        '"tests/test_translator/test_backend_contract.py"',
        '"tests/test_translator/test_codegen/**"',
        '"tests/test_translator/test_frontend_*.py"',
        '"tests/test_translator/test_ir_legacy_alias_contracts.py"',
        '"tests/test_translator/test_lexer.py"',
        '"tests/test_translator/test_parser.py"',
        '"tests/test_translator/test_shader_validation.py"',
        '"tests/test_translator/test_translation_pipeline.py"',
    ]
    for path_filter in required_path_filters:
        assert path_filter in issue_sync
    assert (
        "python tools/support_matrix.py check --output "
        "support/generated/support-matrix-check.json"
    ) in issue_sync
    assert "python tools/support_matrix.py evidence" in issue_sync
    assert "--status supported" in issue_sync
    assert "--evidence missing" in issue_sync
    assert "--output support/generated/support-evidence-check.json" in issue_sync
    assert "name: Upload support matrix check report" in issue_sync
    assert "name: support-matrix-check-report" in issue_sync
    assert "support/generated/support-matrix-check.json" in issue_sync
    assert "support/generated/support-evidence-check.json" in issue_sync
    assert "--support-evidence support/generated/support-evidence-check.json" in (
        issue_sync
    )
    assert (
        "python tools/ci_coverage.py report --output "
        "support/generated/ci-coverage-report.json"
    ) in issue_sync
    assert (
        "python tools/ci_coverage.py summary --output "
        "support/generated/ci-coverage-report.md"
    ) in issue_sync
    assert (
        'cat support/generated/ci-coverage-report.md >> "$GITHUB_STEP_SUMMARY"'
        in issue_sync
    )
    assert "name: Write base CI coverage report" in issue_sync
    assert "git fetch --no-tags --depth=1 origin" in issue_sync
    assert 'git worktree add --detach "$RUNNER_TEMP/ci-coverage-base" FETCH_HEAD' in (
        issue_sync
    )
    assert (
        'python tools/ci_coverage.py --root "$RUNNER_TEMP/ci-coverage-base" report'
        in issue_sync
    )
    assert "support/generated/ci-coverage-base-report.json" in issue_sync
    assert "name: Compare CI coverage with base" in issue_sync
    assert "python tools/ci_coverage.py compare" in issue_sync
    assert "--baseline support/generated/ci-coverage-base-report.json" in issue_sync
    assert "--current support/generated/ci-coverage-report.json" in issue_sync
    assert "--output support/generated/ci-coverage-comparison.json" in issue_sync
    assert "--fail-on-shrink" in issue_sync
    assert "actions/upload-artifact@v4" in issue_sync
    assert "if: always()" in issue_sync
    assert "name: ci-coverage-report" in issue_sync
    assert "path: |" in issue_sync
    assert "support/generated/ci-coverage-report.json" in issue_sync
    assert "support/generated/ci-coverage-report.md" in issue_sync
    assert "support/generated/ci-coverage-comparison.json" in issue_sync
    assert "retention-days: 30" in issue_sync
    assert "python tools/ci_coverage.py check" in issue_sync
    assert '"tools/ci_coverage.py"' in issue_sync
    assert "python tools/support_signals.py docs" in issue_sync
    assert "python tools/support_signals.py extract" in issue_sync
    assert "workflow_run:" in issue_sync
    assert "Complete Test Suite" in issue_sync
    assert "Backend Tests" in issue_sync
    assert "Translator Tests" in issue_sync
    assert "name: Download test failure summaries" in issue_sync
    assert "actions/download-artifact@v4" in issue_sync
    assert "run-id: ${{ github.event.workflow_run.id }}" in issue_sync
    assert 'pattern: "*failure-summary*"' in issue_sync
    assert "support/generated/pytest-failures" in issue_sync
    assert "name: Write clean Complete Test Suite failure summary" in issue_sync
    assert "github.event.workflow_run.name == 'Complete Test Suite'" in issue_sync
    assert "github.event.workflow_run.conclusion == 'success'" in issue_sync
    assert "--clean-workflow" in issue_sync
    assert "--clean-run-id" in issue_sync
    assert "--clean-conclusion" in issue_sync
    assert "--clean-head-sha" in issue_sync
    assert (
        "support/generated/pytest-failures/clean-workflow/clean-workflow-failure-summary.json"
        in issue_sync
    )
    assert "--pytest-failure-summary" in issue_sync
    assert "name: Choose pytest failure closure mode" in issue_sync
    assert "id: pytest_failure_closure" in issue_sync
    assert "--preserve-pytest-failure-issues" in issue_sync
    assert "${{ steps.pytest_failure_closure.outputs.args }}" in issue_sync
    assert "name: Upload support signal reports" in issue_sync
    assert "name: support-signal-reports" in issue_sync
    assert "support/generated/backend-docs-report.json" in issue_sync
    assert "support/generated/pytest-failures/**" in issue_sync
    assert "support/generated/support-signals.json" in issue_sync
    assert "if-no-files-found: ignore" in issue_sync
    assert '"tests/test_ci_workflows.py"' in issue_sync
    assert '"tests/test_examples_test_script.py"' in issue_sync
    assert '"tests/test_support_ci_summary.py"' in issue_sync
    assert '"tests/test_support_matrix.py"' in issue_sync
    assert '"tools/support_ci_summary.py"' in issue_sync
    assert '"tests/test_tool_cli.py"' in issue_sync
    assert "python -m pytest -q" in issue_sync
    assert "tests/test_support_matrix.py" in issue_sync
    assert "tests/test_support_signals.py" in issue_sync
    assert "tests/test_support_ci_summary.py" in issue_sync
    assert "tests/test_support_issue_sync.py" in issue_sync
    assert "tests/test_pr_issue_links.py" in issue_sync
    assert "tests/test_ci_workflows.py" in issue_sync
    assert "tests/test_examples_test_script.py" in issue_sync
    assert "tests/test_tool_cli.py" in issue_sync
    assert "github.event_name == 'pull_request'" in issue_sync
    assert "--dry-run" in issue_sync
    assert "--plan-output support/generated/support-issue-plan.json" in issue_sync
    assert "name: Plan GitHub issue sync" in issue_sync
    assert "--inspect-existing" in issue_sync
    assert "github.event_name != 'pull_request'" in issue_sync
    assert "python tools/sync_support_issues.py" in issue_sync
    assert "--signals support/generated/support-signals.json" in issue_sync
    assert (
        "--matrix-check-report support/generated/support-matrix-check.json"
        in issue_sync
    )
    assert "--max-retries 6" in issue_sync
    assert "--min-desired-issues 10" in issue_sync
    assert "--planned-action-budget-mode fail" in issue_sync
    assert "--max-planned-created 300" in issue_sync
    assert "--max-planned-updated 300" in issue_sync
    assert "--max-planned-closed 500" in issue_sync
    assert "--max-planned-attached 300" in issue_sync
    assert "--max-planned-total 600" in issue_sync
    assert "--max-planned-stale-parent-closures 0" in issue_sync
    assert "--max-planned-stale-backlog-closures 250" in issue_sync
    assert "--max-planned-stale-extracted-closures 250" in issue_sync
    assert "--max-planned-duplicate-marker-closures 25" in issue_sync
    assert (
        "--sync-summary-output support/generated/support-issue-sync-summary.json"
        in issue_sync
    )
    assert "name: Upload support issue sync reports" in issue_sync
    assert "name: support-issue-sync-reports" in issue_sync
    assert "name: Write support automation summary" in issue_sync
    assert "python tools/support_ci_summary.py" in issue_sync
    assert "--matrix-check support/generated/support-matrix-check.json" in issue_sync
    assert "--issue-plan support/generated/support-issue-plan.json" in issue_sync
    assert (
        "--sync-summary support/generated/support-issue-sync-summary.json" in issue_sync
    )
    assert "--output support/generated/support-issue-ci-summary.md" in issue_sync
    assert (
        "--metrics-output support/generated/support-issue-sync-metrics.json"
        in issue_sync
    )
    assert '--step-summary "$GITHUB_STEP_SUMMARY"' in issue_sync
    assert "--github-annotations" in issue_sync
    assert "--fail-on-attention" in issue_sync
    assert "support/generated/support-issue-plan.json" in issue_sync
    assert "support/generated/support-issue-sync-summary.json" in issue_sync
    assert "support/generated/support-issue-sync-metrics.json" in issue_sync
    assert "support/generated/support-issue-ci-summary.md" in issue_sync


def test_pr_issue_link_workflow_assigns_closing_keywords_and_gates_traceability():
    workflows = _workflow_texts()
    pr_issue_links = workflows.get("pr-issue-links.yml", "")

    assert pr_issue_links, "pr-issue-links.yml must exist"
    assert "pull_request_target:" in pr_issue_links
    assert "issues: write" in pr_issue_links
    assert "pull-requests: write" in pr_issue_links
    assert "persist-credentials: false" in pr_issue_links
    assert "python tools/sync_pr_issue_links.py" in pr_issue_links
    assert "--sync-support-closures" in pr_issue_links
    assert "--sync-support-references" in pr_issue_links
    assert "--check-support-traceability" not in pr_issue_links
    assert "--enforce-support-traceability" in pr_issue_links
    assert (
        "--summary-output support/generated/pr-issue-link-summary.json"
        in pr_issue_links
    )
    assert "name: Validate PR issue link summary" in pr_issue_links
    assert "--validate-summary support/generated/pr-issue-link-summary.json" in (
        pr_issue_links
    )
    assert "name: Upload PR issue link summary" in pr_issue_links
    assert "if: always()" in pr_issue_links
    assert "name: pr-issue-link-summary" in pr_issue_links
    assert "path: support/generated/pr-issue-link-summary.json" in pr_issue_links


def test_windows_validator_install_retries_and_uses_direct_lunarg_fallback():
    workflows = _workflow_texts()
    full_suite = workflows.get("full-tests.yml", "")

    assert '$vulkanSdkVersion = "1.4.341.1"' in full_suite
    assert "$maxAttempts = 3" in full_suite
    assert "for ($attempt = 1; $attempt -le $maxAttempts; $attempt++)" in full_suite
    assert "choco install vulkan-sdk --version=1.4.341" in full_suite
    assert "Chocolatey Vulkan SDK install failed" in full_suite
    assert "sdk.lunarg.com/sdk/download/$vulkanSdkVersion/windows" in full_suite
    assert (
        "--accept-licenses --default-answer --confirm-command install copy_only=1"
        in full_suite
    )
    assert "Direct Vulkan SDK install failed" in full_suite
    assert 'throw "Vulkan SDK install directory was not found"' in full_suite
    assert "$global:LASTEXITCODE = 0" not in full_suite
