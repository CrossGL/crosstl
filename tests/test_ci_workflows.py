from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"


def _workflow_texts():
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(WORKFLOW_DIR.glob("*.yml"))
    }


def test_ci_runs_the_complete_pytest_suite_on_pull_requests_and_pushes():
    workflows = _workflow_texts()
    full_suite = workflows.get("full-tests.yml", "")

    assert full_suite, "full-tests.yml must exist"
    assert re.search(r"\bpull_request\s*:", full_suite)
    assert re.search(r"\bpush\s*:", full_suite)
    assert "glslang-tools" in full_suite
    assert "brew install glslang" in full_suite
    assert "choco install vulkan-sdk --version=1.4.341" in full_suite
    assert "DirectXShaderCompiler/releases/download/v1.9.2602" in full_suite
    assert "linux_dxc_2026_02_20.x86_64.tar.gz" in full_suite
    assert "dxc_2026_02_20.zip" in full_suite
    assert "shader-validators:" in full_suite
    assert "macOS-latest" in full_suite
    assert "windows-latest" in full_suite
    assert re.search(r"python\s+-m\s+pytest\s+tests\b", full_suite)
    assert "test_external_shader_validators.py" in full_suite


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
    assert "docs-probe:" in support_matrix
    assert "github.event_name == 'schedule'" in support_matrix
    assert "github.event_name == 'workflow_dispatch'" in support_matrix
    assert (
        "python tools/support_matrix.py docs --output "
        "support/generated/backend-docs-report.json"
    ) in support_matrix
    assert "actions/upload-artifact@v4" in support_matrix
