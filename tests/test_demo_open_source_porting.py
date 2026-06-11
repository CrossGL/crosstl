import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_CI_METADATA_PATH = ROOT / "support" / "demo-ci-metadata.json"
DEMO_CI_TOOL_PATH = ROOT / "tools" / "demo_ci_metadata.py"
DEMO_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "demo.yml"


def _load_demo_ci_tool():
    spec = importlib.util.spec_from_file_location("demo_ci_metadata", DEMO_CI_TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _test_function_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def test_demo_ci_metadata_matches_checked_in_pytest_cases():
    tool = _load_demo_ci_tool()
    metadata = tool.load_metadata(DEMO_CI_METADATA_PATH)
    test_file = ROOT / metadata["pytest"]["test_file"]
    test_names = _test_function_names(test_file)

    assert tool.pytest_files(metadata) == [metadata["pytest"]["test_file"]]
    assert metadata["pytest"]["cases"]

    for case in metadata["pytest"]["cases"]:
        selector = case["selector"]
        matched_tests = sorted(name for name in test_names if selector in name)

        assert matched_tests, selector
        assert sorted(case["tests"]) == matched_tests
        assert case["targets"]
        assert all(target == target.lower() for target in case["targets"])


def test_demo_ci_metadata_generates_workflow_pytest_inputs():
    tool = _load_demo_ci_tool()
    metadata = tool.load_metadata(DEMO_CI_METADATA_PATH)
    selectors = [case["selector"] for case in metadata["pytest"]["cases"]]

    assert tool.pytest_selectors(metadata) == selectors
    assert tool.pytest_selector_expression(metadata) == " or ".join(selectors)


def test_demo_workflow_consumes_generated_demo_ci_inputs():
    metadata = json.loads(DEMO_CI_METADATA_PATH.read_text(encoding="utf-8"))
    workflow = DEMO_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "tools/demo_ci_metadata.py emit-pytest-files" in workflow
    assert "tools/demo_ci_metadata.py emit-pytest-selector" in workflow
    assert '"${demo_test_files[@]}"' in workflow
    assert '-k "$demo_selector"' in workflow

    for case in metadata["pytest"]["cases"]:
        assert case["selector"] not in workflow


def test_demo_ci_metadata_cli_check_and_emitters():
    check = subprocess.run(
        [sys.executable, "tools/demo_ci_metadata.py", "check"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    assert "demo CI metadata is valid" in check.stdout

    files = subprocess.run(
        [sys.executable, "tools/demo_ci_metadata.py", "emit-pytest-files"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    selector = subprocess.run(
        [sys.executable, "tools/demo_ci_metadata.py", "emit-pytest-selector"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = json.loads(DEMO_CI_METADATA_PATH.read_text(encoding="utf-8"))

    assert files.stdout.splitlines() == [metadata["pytest"]["test_file"]]
    assert selector.stdout.strip() == " or ".join(
        case["selector"] for case in metadata["pytest"]["cases"]
    )
