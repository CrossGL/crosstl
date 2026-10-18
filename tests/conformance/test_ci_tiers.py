"""CI tiering of the conformance and corpus workflows.

macOS and Windows runners are scarce and shared, so a pull request runs only the Linux legs of these workflows;
the macOS and Windows legs run on pushes to main, on the nightly schedule, on manual dispatch, and on a pull
request labelled ``ci:full``. Every job carries a timeout.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
TIERED = ("conformance.yml", "corpus.yml")
SCARCE = ("macos", "xcode", "windows")


def load(name):
    workflow = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    # YAML 1.1 reads the bare key ``on`` as True.
    workflow["on"] = workflow.pop(True, workflow.get("on"))
    return workflow


def selected_legs(workflow, full):
    """Run the leg-selection step as the runner would, with ``FULL`` set as its expression would evaluate."""
    step = workflow["jobs"]["legs"]["steps"][0]
    with tempfile.TemporaryDirectory() as scratch:
        output = Path(scratch) / "output"
        output.write_text("")
        environment = {key: value for key, value in step["env"].items() if key != "FULL"}
        environment.update(
            PATH=os.environ.get("PATH", ""), FULL="true" if full else "false", GITHUB_OUTPUT=str(output)
        )
        subprocess.run(["bash", "-c", step["run"]], env=environment, check=True, capture_output=True, timeout=30)
        key, _, value = output.read_text().strip().partition("=")
    assert key == "legs"
    return json.loads(value)


@pytest.mark.parametrize("name", TIERED)
def test_every_job_has_a_timeout(name):
    for job, definition in load(name)["jobs"].items():
        assert isinstance(definition.get("timeout-minutes"), int) and definition["timeout-minutes"] > 0, job


@pytest.mark.parametrize("name", TIERED)
def test_full_selection_follows_the_event_and_the_label(name):
    expression = load(name)["jobs"]["legs"]["steps"][0]["env"]["FULL"]
    assert "github.event_name != 'pull_request'" in expression
    assert "contains(github.event.pull_request.labels.*.name, 'ci:full')" in expression


@pytest.mark.parametrize("name", TIERED)
def test_matrix_is_the_selected_legs(name):
    jobs = load(name)["jobs"]
    hosted = [job for job, definition in jobs.items() if "matrix" in definition.get("runs-on", "")]
    assert hosted
    for job in hosted:
        assert jobs[job]["needs"] == "legs"
        assert jobs[job]["strategy"]["matrix"] == {"include": "${{ fromJSON(needs.legs.outputs.legs) }}"}
    for job, definition in jobs.items():
        runner = str(definition.get("runs-on", "")).lower()
        assert job in hosted or not any(word in runner for word in SCARCE), job


@pytest.mark.skipif(not Path("/bin/bash").exists() or not any(
    (Path(directory) / "jq").exists() for directory in os.environ.get("PATH", "").split(os.pathsep)
), reason="the leg selection step needs bash and jq")
@pytest.mark.parametrize("name", TIERED)
def test_pull_requests_run_only_linux_legs(name):
    workflow = load(name)
    lean = selected_legs(workflow, full=False)
    full = selected_legs(workflow, full=True)
    assert lean and all(leg["os"].startswith("ubuntu-") for leg in lean)
    assert lean == [leg for leg in full if leg["os"].startswith("ubuntu-")]
    assert any(word in leg["os"] for leg in full for word in SCARCE)
    assert len({leg["name"] for leg in full}) == len(full)
    keys = {frozenset(leg) for leg in full}
    assert len(keys) == 1, "every leg defines the same matrix keys"


def test_conformance_runs_on_pull_requests_for_its_paths():
    trigger = load("conformance.yml")["on"]
    assert "types" not in trigger["pull_request"], "the label takes effect on the next push"
    assert "crosstl/**" in trigger["pull_request"]["paths"]
    assert "tests/conformance/**" in trigger["pull_request"]["paths"]
    assert "schedule" in trigger and "workflow_dispatch" in trigger


def test_corpus_pull_requests_are_path_filtered():
    trigger = load("corpus.yml")["on"]
    paths = trigger["pull_request"]["paths"]
    assert "types" not in trigger["pull_request"]
    for required in ("crosstl/**", "tests/conformance/corpus/**", ".github/workflows/corpus.yml"):
        assert required in paths
    assert "schedule" in trigger and "workflow_dispatch" in trigger
