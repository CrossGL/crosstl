import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.pytest_scope import EVERYTHING, SCOPES, unaffected  # noqa: E402

MLX = "demos/integrations/mlx/"


def test_a_change_outside_the_inputs_leaves_their_tests_unaffected():
    assert unaffected(["crosstl/translator/parser.py", "docs/index.md"]) == [
        "mlx_model"
    ]


@pytest.mark.parametrize(
    "changed",
    [
        [MLX + "models/qmm.py"],
        ["crosstl/translator/parser.py", MLX + "data/cases.json"],
        # Paths that can change any test's outcome or selection.
        *([path] for path in EVERYTHING),
        # An empty list is more likely a broken diff than an empty change.
        [],
    ],
)
def test_changes_that_reach_the_inputs_run_the_scoped_tests(changed):
    assert unaffected(changed) == []


def test_a_sibling_directory_is_not_an_input():
    assert unaffected(["demos/integrations/mlx-other/x.py"]) == ["mlx_model"]


def test_every_scope_names_tracked_inputs():
    for marker, inputs in SCOPES.items():
        for entry in inputs:
            assert (ROOT / entry).exists(), (marker, entry)


def _project(tmp_path):
    # The plugin reads changed paths relative to the rootdir, which an ini file pins.
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text(
        "def pytest_configure(config):\n"
        "    config.addinivalue_line('markers', 'mlx_model: scoped test')\n",
        encoding="utf-8",
    )
    scoped = tmp_path / "demos" / "integrations" / "mlx" / "tests"
    scoped.mkdir(parents=True)
    (scoped / "test_models.py").write_text(
        "import pytest\n\n"
        "pytestmark = pytest.mark.mlx_model\n\n"
        "@pytest.mark.parametrize('n', range(3))\n"
        "def test_model(n):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (scoped / "test_translates.py").write_text(
        "def test_translation():\n    pass\n", encoding="utf-8"
    )
    elsewhere = tmp_path / "tests"
    elsewhere.mkdir()
    (elsewhere / "test_misplaced.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.mlx_model\n"
        "def test_reads_more_than_the_demo():\n"
        "    pass\n",
        encoding="utf-8",
    )
    return tmp_path


def _collect(project, *args):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tools.pytest_scope",
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
            *args,
        ],
        cwd=project,
        env=dict(os.environ, PYTHONPATH=str(ROOT)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 5), result.stdout + result.stderr
    return {line for line in result.stdout.splitlines() if "::" in line}


def test_the_plugin_deselects_only_marked_tests_under_their_inputs(tmp_path):
    project = _project(tmp_path)
    everything = _collect(project)
    assert len(everything) == 5

    changed = project / "changed.txt"
    changed.write_text("crosstl/translator/parser.py\n", encoding="utf-8")
    kept = _collect(project, "--changed-files", str(changed))
    assert kept == {
        "demos/integrations/mlx/tests/test_translates.py::test_translation",
        "tests/test_misplaced.py::test_reads_more_than_the_demo",
    }

    changed.write_text(MLX + "tests/test_models.py\n", encoding="utf-8")
    assert _collect(project, "--changed-files", str(changed)) == everything


def test_an_unreadable_list_is_a_usage_error(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tools.pytest_scope",
            "--changed-files",
            str(tmp_path / "missing.txt"),
            "--collect-only",
        ],
        cwd=_project(tmp_path),
        env=dict(os.environ, PYTHONPATH=str(ROOT)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 4
    assert "cannot read" in result.stderr


def test_the_complete_suite_scopes_only_pull_requests_and_merge_groups():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "full-tests.yml").read_text(encoding="utf-8")
    )
    # The nightly run tests every scoped test against current dependencies.
    assert "schedule" in workflow[True]
    steps = {step.get("name"): step for step in workflow["jobs"]["pytest"]["steps"]}
    assert steps["Checkout repository"]["with"]["fetch-depth"] == 2
    listing = steps["List the files the change touches"]
    assert (
        "git diff --name-only HEAD^1 HEAD > build/changed-files.txt" in listing["run"]
    )
    condition = " ".join(listing["if"].split())
    assert "github.event_name == 'merge_group'" in condition
    assert "github.event_name == 'pull_request'" in condition
    assert "'ci:full'" in condition
    assert "push" not in condition and "schedule" not in condition
    command = steps["Run complete test suite"]["run"]
    assert "if [ -f build/changed-files.txt ]" in command
    assert "-p tools.pytest_scope --changed-files build/changed-files.txt" in command
    assert '"${scope[@]}"' in command
