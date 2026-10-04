"""Project demos remain discoverable without becoming package dependencies."""

import ast
import configparser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_default_pytest_collection_includes_project_demos():
    config = configparser.ConfigParser()
    config.read(ROOT / "setup.cfg")
    assert config["tool:pytest"]["testpaths"].split() == [
        "tests",
        "demos/integrations",
    ]


def test_complete_ci_and_manual_hook_include_project_demos():
    workflow = (ROOT / ".github/workflows/full-tests.yml").read_text()
    hooks = (ROOT / ".pre-commit-config.yaml").read_text()
    command = "python -m pytest tests demos/integrations"
    assert command in workflow
    assert command in hooks


def test_distribution_excludes_demo_packages():
    setup = ast.parse((ROOT / "setup.py").read_text())
    package_calls = [
        node
        for node in ast.walk(setup)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "find_namespace_packages"
    ]
    assert len(package_calls) == 1
    include = next(
        ast.literal_eval(item.value)
        for item in package_calls[0].keywords
        if item.arg == "include"
    )
    assert include == ["crosstl*"]
