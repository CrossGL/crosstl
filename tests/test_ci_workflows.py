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
    assert re.search(r"python\s+-m\s+pytest\s+tests\b", full_suite)


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
