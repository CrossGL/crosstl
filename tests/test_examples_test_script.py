import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "test.py"


def load_examples_test_module():
    spec = importlib.util.spec_from_file_location("examples_test_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_summary_records_regression_budget_and_failures():
    module = load_examples_test_module()
    failures = [
        ("ComplexShader", "mojo", "unsupported sample"),
        ("GenericPatternMatching", "cuda", "unsupported match arm"),
    ]

    summary = module.build_summary(
        total_tests=10,
        successful_tests=8,
        failed_tests=failures,
        consistency_summary={"successful": 4, "total": 4},
    )

    assert summary["schema_version"] == 1
    assert summary["failed"] == 2
    assert summary["success_rate"] == 80.0
    assert summary["known_failure_budget"] == module.KNOWN_FAILURE_BUDGET
    assert summary["minimum_success_rate"] == module.MIN_SUCCESS_RATE
    assert summary["within_regression_budget"] is False
    assert summary["failures"][0] == {
        "example": "ComplexShader",
        "backend": "mojo",
        "error": "unsupported sample",
    }
    assert summary["consistency"] == {"successful": 4, "total": 4}


def test_build_summary_accepts_current_known_failure_budget():
    module = load_examples_test_module()
    failures = [
        ("KnownFailure{}".format(index), "backend", "expected")
        for index in range(module.KNOWN_FAILURE_BUDGET)
    ]

    summary = module.build_summary(
        total_tests=60,
        successful_tests=54,
        failed_tests=failures,
    )

    assert summary["failed"] == module.KNOWN_FAILURE_BUDGET
    assert summary["success_rate"] == 90.0
    assert summary["within_regression_budget"] is True


def test_write_summary_json_writes_stable_machine_readable_report(tmp_path):
    module = load_examples_test_module()
    output = tmp_path / "nested" / "examples-summary.json"
    summary = module.build_summary(
        total_tests=1,
        successful_tests=1,
        failed_tests=[],
    )

    module.write_summary_json(summary, output)

    text = output.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert '"schema_version": 1' in text
    assert '"within_regression_budget": true' in text
