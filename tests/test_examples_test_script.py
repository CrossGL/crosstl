import importlib.util
import json
import subprocess
import sys
from pathlib import Path

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
    assert summary["skipped"] == 0
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


def test_build_summary_records_documented_skips():
    module = load_examples_test_module()
    skips = [("GenericPatternMatching", "mojo", "generic functions unsupported")]

    summary = module.build_summary(
        total_tests=59,
        successful_tests=59,
        failed_tests=[],
        skipped_tests=skips,
    )

    assert summary["failed"] == 0
    assert summary["skipped"] == 1
    assert summary["success_rate"] == 100.0
    assert summary["within_regression_budget"] is True
    assert summary["skips"] == [
        {
            "example": "GenericPatternMatching",
            "backend": "mojo",
            "reason": "generic functions unsupported",
        }
    ]


def test_build_summary_accepts_current_known_failure_budget():
    module = load_examples_test_module()
    failures = [
        (f"KnownFailure{index}", "backend", "expected")
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


def test_generic_pattern_matching_cli_reports_documented_generic_gaps():
    source = ROOT / "examples" / "advanced" / "GenericPatternMatching.cgl"
    expected_diagnostics = {
        "cuda": "CUDA codegen cannot emit unresolved generic parameter 'T'",
        "mojo": "generic payload enum specializations must be concrete",
        "slang": "Slang codegen cannot emit unresolved generic parameter 'T'",
    }

    for backend, diagnostic in expected_diagnostics.items():
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "crosstl._crosstl",
                "translate",
                str(source),
                "--backend",
                backend,
                "--no-format",
                "--output",
                "-",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        combined_output = result.stdout + result.stderr

        assert result.returncode != 0
        assert diagnostic in combined_output

    for backend in ("hip", "vulkan"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "crosstl._crosstl",
                "translate",
                str(source),
                "--backend",
                backend,
                "--no-format",
                "--output",
                "-",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]


def test_main_is_independent_of_current_working_directory(monkeypatch, tmp_path):
    module = load_examples_test_module()
    examples_root = tmp_path / "examples"
    source = examples_root / "graphics" / "SimpleShader.cgl"
    source.parent.mkdir(parents=True)
    source.write_text("shader SimpleShader {}", encoding="utf-8")
    summary_path = tmp_path / "summary.json"

    monkeypatch.setattr(module, "EXAMPLES_ROOT", examples_root)
    monkeypatch.setattr(
        module, "EXAMPLES_BY_CATEGORY", {"graphics": ["SimpleShader.cgl"]}
    )
    monkeypatch.setattr(module, "BACKENDS", {"metal": ".metal"})
    monkeypatch.setattr(module, "BACKEND_COMPATIBILITY", {"graphics": ["metal"]})
    monkeypatch.setattr(module, "EXAMPLE_BACKEND_SKIPS", {})
    monkeypatch.chdir(tmp_path)

    def translate(input_path, backend, save_shader=None):
        assert Path(input_path).is_absolute()
        if save_shader:
            Path(save_shader).write_text("//" + ("x" * 120), encoding="utf-8")
        return "x" * 120

    monkeypatch.setattr(module.crosstl, "translate", translate)

    assert module.main(["--summary-json", str(summary_path)]) == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["within_regression_budget"] is True
    assert (
        examples_root / "output" / "metal" / "graphics" / "SimpleShader.metal"
    ).exists()


def test_main_returns_failure_when_regression_budget_is_exceeded(
    monkeypatch, tmp_path, capsys
):
    module = load_examples_test_module()
    examples_root = tmp_path / "examples"
    source = examples_root / "graphics" / "SimpleShader.cgl"
    source.parent.mkdir(parents=True)
    source.write_text("shader SimpleShader {}", encoding="utf-8")

    monkeypatch.setattr(module, "EXAMPLES_ROOT", examples_root)
    monkeypatch.setattr(
        module, "EXAMPLES_BY_CATEGORY", {"graphics": ["SimpleShader.cgl"]}
    )
    monkeypatch.setattr(module, "BACKENDS", {"metal": ".metal"})
    monkeypatch.setattr(module, "BACKEND_COMPATIBILITY", {"graphics": ["metal"]})
    monkeypatch.setattr(module, "EXAMPLE_BACKEND_SKIPS", {})
    monkeypatch.setattr(
        module,
        "test_cross_backend_consistency",
        lambda: {"successful": 0, "total": 0},
    )

    def translate(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(module.crosstl, "translate", translate)

    assert module.main([]) == 1
    captured = capsys.readouterr()
    assert "Example regression budget exceeded" in captured.err
