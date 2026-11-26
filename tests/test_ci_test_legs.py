"""The Ubuntu translator and backend legs and the Complete Test Suite test every supported Python version once."""

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"


def _supported_python_versions():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return set(re.findall(r'"Programming Language :: Python :: (3\.\d+)"', text))


@pytest.mark.parametrize(
    "filename,suite",
    [
        ("translator-tests.yml", "tests/test_translator"),
        ("backend-tests.yml", "tests/test_backend"),
    ],
)
def test_suite_legs_and_the_complete_suite_test_every_python_version_once(
    filename, suite
):
    # The Complete Test Suite runs every test under tests/ on one Python version on
    # Ubuntu. The translator and backend workflows cover the other versions, so the
    # merge queue tests each supported version once, with every test module.
    full = yaml.safe_load((WORKFLOW_DIR / "full-tests.yml").read_text("utf-8"))
    pytest_job = full["jobs"]["pytest"]
    assert pytest_job["runs-on"] == "ubuntu-latest"
    (setup,) = (
        step
        for step in pytest_job["steps"]
        if step.get("uses", "").startswith("actions/setup-python@")
    )
    complete_version = setup["with"]["python-version"]
    (complete_run,) = (
        step["run"]
        for step in pytest_job["steps"]
        if step.get("name") == "Run complete test suite"
    )
    assert "python -m pytest tests demos/integrations" in complete_run
    assert "--ignore" not in complete_run

    workflow = yaml.safe_load((WORKFLOW_DIR / filename).read_text("utf-8"))
    test = workflow["jobs"]["test"]
    matrix = test["strategy"]["matrix"]
    # The matrix picks the versions itself, so no job has to run before the legs to choose them.
    choice = re.fullmatch(
        r"\$\{\{ fromJSON\(\(github\.event_name != 'pull_request'"
        r" \|\| contains\(github\.event\.pull_request\.labels\.\*\.name, 'ci:full'\)\)"
        r" && '(\[[^']*\])' \|\| '(\[[^']*\])'\) \}\}",
        matrix["python-version"],
    )
    assert choice, matrix["python-version"]
    queue_versions = json.loads(choice[1])
    pull_request_versions = json.loads(choice[2])
    supported = _supported_python_versions()
    assert complete_version in supported
    assert complete_version not in queue_versions
    assert len(queue_versions) == len(set(queue_versions))
    assert set(queue_versions) | {complete_version} == supported
    assert set(pull_request_versions) <= set(queue_versions)
    ordered = sorted(supported, key=lambda version: tuple(map(int, version.split("."))))
    # A pull request tests the oldest and the newest version on Ubuntu, each once.
    # The Complete Test Suite runs on every event, so it takes one of them and the
    # suite's own legs add the other: a third version, or a leg that repeated the
    # complete suite's, would hold more runners on every push.
    assert complete_version not in pull_request_versions
    assert set(pull_request_versions) | {complete_version} == {ordered[0], ordered[-1]}

    assert matrix["OS"] == ["ubuntu-latest"]
    (run,) = (
        step["run"] for step in test["steps"] if step.get("id", "").startswith("run_")
    )
    # Each Ubuntu leg runs the whole suite or one shard of it, never a subset of
    # its directories, which would leave the other modules untested on that version.
    assert f"python -m pytest {suite} " in run
    assert "--ignore" not in run
    assert "matrix.component }}_codegen" not in run
    assert "test_${{ matrix.backend }}" not in run
