"""Project demos remain discoverable without becoming package dependencies."""

import tomllib
from pathlib import Path

from tools.check_release import package_metadata

ROOT = Path(__file__).resolve().parents[1]


def test_default_pytest_collection_includes_project_demos():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["pytest"]["ini_options"]["testpaths"] == [
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
    assert package_metadata(ROOT)["name"] == "crosstl"
    assert (ROOT / "crosstl/__init__.py").is_file()
    assert not (ROOT / "crosstl/demos").exists()
