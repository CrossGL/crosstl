import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "pytest_failure_summary.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("pytest_failure_summary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_junit(path):
    path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="6" failures="3" errors="1" skipped="1">
    <testcase classname="tests.test_translator.test_codegen.test_external_shader_validators" name="test_generated_glsl_validates" file="tests/test_translator/test_codegen/test_external_shader_validators.py">
      <failure message="glslangValidator rejected generated GLSL">shader validation failed</failure>
    </testcase>
    <testcase classname="tests.test_translator.test_codegen.test_mojo_codegen" name="test_generic_user_struct_compile" file="tests/test_translator/test_codegen/test_mojo_codegen.py">
      <error message="mojo compiler rejected generated source">compiler failed</error>
    </testcase>
    <testcase classname="tests.test_support_matrix" name="test_matrix_check" file="tests/test_support_matrix.py">
      <failure message="support matrix check failed">generated support matrix is stale</failure>
    </testcase>
    <testcase classname="tests.test_translator.test_codegen.test_metal_codegen" name="test_codegen_text" file="tests/test_translator/test_codegen/test_metal_codegen.py">
      <failure message="generated source assertion failed">expected texture call</failure>
    </testcase>
    <testcase classname="tests.test_translator.test_parser" name="test_parser_ok" file="tests/test_translator/test_parser.py" />
    <testcase classname="tests.test_translator.test_codegen.test_directx_codegen" name="test_directx_skipped" file="tests/test_translator/test_codegen/test_directx_codegen.py">
      <skipped />
    </testcase>
  </testsuite>
</testsuites>
""",
        encoding="utf-8",
    )


def test_build_summary_categorizes_backend_compiler_and_support_failures(tmp_path):
    module = _load_module()
    junit = tmp_path / "pytest.xml"
    _write_junit(junit)

    summary = module.build_summary([junit])

    assert summary["schema_version"] == 1
    assert summary["generator"] == "tools/pytest_failure_summary.py"
    assert summary["summary"]["report_count"] == 1
    assert summary["summary"]["testcase_count"] == 6
    assert summary["summary"]["failure_count"] == 3
    assert summary["summary"]["error_count"] == 1
    assert summary["summary"]["skipped_count"] == 1
    assert summary["summary"]["failed_testcase_count"] == 4
    assert summary["summary"]["categories"] == {
        "backend_codegen": 1,
        "backend_compiler_validation": 2,
        "support_automation": 1,
    }
    assert summary["summary"]["backends"] == {
        "metal": 1,
        "mojo": 1,
        "opengl": 1,
        "unknown": 1,
    }


def test_missing_report_is_recorded_as_load_error(tmp_path):
    module = _load_module()
    missing = tmp_path / "missing.xml"

    summary = module.build_summary([missing])

    assert summary["summary"]["load_error_count"] == 1
    assert summary["reports"][0]["load_error"]["type"] == "FileNotFoundError"


def test_clean_workflow_summary_records_authoritative_clean_run():
    module = _load_module()

    summary = module.build_summary(
        [],
        clean_workflow_runs=[
            {
                "workflow": "Complete Test Suite",
                "run_id": "123",
                "conclusion": "success",
                "head_sha": "abc",
            }
        ],
    )

    assert summary["summary"]["report_count"] == 0
    assert summary["summary"]["failed_testcase_count"] == 0
    assert summary["clean_workflow_runs"] == [
        {
            "workflow": "Complete Test Suite",
            "run_id": "123",
            "conclusion": "success",
            "head_sha": "abc",
        }
    ]
    assert summary["failures"] == []


def test_cli_writes_json_and_markdown_outputs(tmp_path):
    junit = tmp_path / "pytest.xml"
    json_output = tmp_path / "summary.json"
    markdown_output = tmp_path / "summary.md"
    _write_junit(junit)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(junit),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")
    assert payload["summary"]["failed_testcase_count"] == 4
    assert "# Pytest Failure Summary" in markdown
    assert "backend_compiler_validation" in markdown
    assert "support_automation" in markdown


def test_cli_writes_clean_workflow_summary_without_junit_input(tmp_path):
    json_output = tmp_path / "clean-summary.json"
    markdown_output = tmp_path / "clean-summary.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--clean-workflow",
            "Complete Test Suite",
            "--clean-run-id",
            "123",
            "--clean-conclusion",
            "success",
            "--clean-head-sha",
            "abc",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")
    assert payload["summary"]["report_count"] == 0
    assert payload["summary"]["failed_testcase_count"] == 0
    assert payload["clean_workflow_runs"][0]["workflow"] == "Complete Test Suite"
    assert "Clean workflow runs" in markdown
