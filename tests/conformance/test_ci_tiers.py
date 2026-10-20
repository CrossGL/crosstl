"""CI tiering of the conformance, corpus and fuzz workflows.

The conformance and corpus workflows run on every pull request and in the merge queue, so that their single
``result`` check can be required; a planning job (``tests/conformance/ci_plan.py``) skips their work for changes
outside their paths instead of a ``paths`` filter. macOS and Windows runners are scarce and shared, so a pull
request runs only the Linux legs unless it is labelled ``ci:full``; the merge queue runs every conformance leg and
the corpus Linux leg. Every job carries a timeout, and every workflow ends in one ``result`` job.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests.conformance import ci_plan

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
TIERED = ("conformance.yml", "corpus.yml")
RESULTS = {
    "conformance.yml": "Conformance result",
    "corpus.yml": "Corpus result",
    "fuzz.yml": "Fuzz result",
}
SCARCE = ("macos", "xcode", "windows")
GIT = shutil.which("git")


def load(name):
    workflow = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    # YAML 1.1 reads the bare key ``on`` as True.
    workflow["on"] = workflow.pop(True, workflow.get("on"))
    return workflow


def plan_step(workflow):
    return next(
        step for step in workflow["jobs"]["legs"]["steps"] if step.get("id") == "select"
    )


def leg_lists(workflow):
    environment = plan_step(workflow)["env"]
    return json.loads(environment["LINUX_LEGS"]), json.loads(environment["HOSTED_LEGS"])


def run_plan_step(workflow, event, full=False, base="", head=""):
    """Run the plan step as the runner would, with its expressions evaluated as given."""
    step = plan_step(workflow)
    with tempfile.TemporaryDirectory() as scratch:
        output = Path(scratch) / "output"
        output.write_text("")
        environment = {
            key: value
            for key, value in step["env"].items()
            if key not in ("FULL", "BASE", "HEAD")
        }
        environment.update(
            PATH=os.environ.get("PATH", ""),
            GITHUB_EVENT_NAME=event,
            GITHUB_OUTPUT=str(output),
            FULL="true" if full else "false",
            BASE=base,
            HEAD=head,
            PYTHONPATH=str(ROOT),
        )
        completed = subprocess.run(
            ["bash", "-c", step["run"].replace("python3 ", f'"{sys.executable}" ')],
            env=environment,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        values = dict(
            line.partition("=")[::2] for line in output.read_text().splitlines()
        )
    assert set(values) == {"run", "legs"}, completed.stdout
    legs = json.loads(values["legs"])
    assert values["run"] == ("true" if legs else "false")
    return legs


# Workflow structure.


@pytest.mark.parametrize("name", TIERED + ("fuzz.yml",))
def test_every_job_has_a_timeout(name):
    for job, definition in load(name)["jobs"].items():
        assert (
            isinstance(definition.get("timeout-minutes"), int)
            and definition["timeout-minutes"] > 0
        ), job


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_one_result_job_requires_every_other_job(name):
    jobs = load(name)["jobs"]
    result = jobs["result"]
    assert result["name"] == RESULTS[name]
    assert result["if"] == "always()"
    assert sorted(result["needs"]) == sorted(job for job in jobs if job != "result")
    script = result["steps"][0]["run"]
    assert "success|skipped) ;;" in script and "exit 1" in script
    assert result["runs-on"].startswith("ubuntu-")


@pytest.mark.parametrize("name", TIERED)
def test_pull_requests_and_the_merge_queue_always_start_the_workflow(name):
    trigger = load(name)["on"]
    assert trigger["pull_request"] is None, "no paths filter: the plan job skips work"
    assert "merge_group" in trigger
    assert "schedule" in trigger and "workflow_dispatch" in trigger


@pytest.mark.parametrize("name", TIERED)
def test_every_job_waits_for_the_plan(name):
    jobs = load(name)["jobs"]
    for job, definition in jobs.items():
        if job in ("legs", "result"):
            continue
        needs = definition["needs"]
        assert "legs" in ([needs] if isinstance(needs, str) else needs), job
        assert definition["if"] == "needs.legs.outputs.run == 'true'", job


@pytest.mark.parametrize("name", TIERED)
def test_plan_inputs_cover_pull_requests_and_merge_groups(name):
    workflow = load(name)
    environment = plan_step(workflow)["env"]
    assert environment["FULL"] == (
        "${{ contains(github.event.pull_request.labels.*.name, 'ci:full') }}"
    )
    assert "github.event.pull_request.base.sha" in environment["BASE"]
    assert "github.event.merge_group.base_sha" in environment["BASE"]
    assert "github.event.pull_request.head.sha" in environment["HEAD"]
    assert "github.event.merge_group.head_sha" in environment["HEAD"]
    checkout = workflow["jobs"]["legs"]["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["fetch-depth"] == 0, "the plan diffs against the base"
    assert f"--workflow {name[:-4]}" in plan_step(workflow)["run"]


@pytest.mark.parametrize("name", TIERED)
def test_matrix_is_the_planned_legs(name):
    jobs = load(name)["jobs"]
    hosted = [
        job
        for job, definition in jobs.items()
        if "matrix" in definition.get("runs-on", "")
    ]
    assert hosted
    for job in hosted:
        assert jobs[job]["needs"] == "legs"
        assert jobs[job]["strategy"]["matrix"] == {
            "include": "${{ fromJSON(needs.legs.outputs.legs) }}"
        }
    for job, definition in jobs.items():
        runner = str(definition.get("runs-on", "")).lower()
        assert job in hosted or not any(word in runner for word in SCARCE), job


@pytest.mark.parametrize("name", TIERED + ("fuzz.yml",))
def test_python_setup_caches_pip(name):
    for job, definition in load(name)["jobs"].items():
        for step in definition.get("steps", []):
            if str(step.get("uses", "")).startswith("actions/setup-python@"):
                assert step["with"].get("cache") == "pip", job


def test_conformance_push_filter_matches_the_plan_paths():
    trigger = load("conformance.yml")["on"]
    assert trigger["push"]["branches"] == ["main"]
    assert sorted(trigger["push"]["paths"]) == sorted(ci_plan.PATHS["conformance"])


@pytest.mark.parametrize("name", TIERED)
def test_legs_are_well_formed(name):
    linux, hosted = leg_lists(load(name))
    legs = linux + hosted
    assert linux and all(leg["os"].startswith("ubuntu-") for leg in linux)
    assert hosted and all(any(word in leg["os"] for word in SCARCE) for leg in hosted)
    assert len({leg["name"] for leg in legs}) == len(legs)
    assert (
        len({frozenset(leg) for leg in legs}) == 1
    ), "every leg defines the same matrix keys"


def test_conformance_merge_queue_runs_the_held_out_kernels():
    workflow = load("conformance.yml")
    tier = next(
        step
        for step in workflow["jobs"]["native"]["steps"]
        if step.get("name", "").startswith("Select tier")
    )
    assert '"$GITHUB_EVENT_NAME" = merge_group' in tier["run"]


def test_windows_leg_runs_the_reference_warp_library():
    """The WARP leg installs one checksum-verified library and refuses any other build at run time.

    WARP 1.0.19 is the reference library of the WARP conformance matrix in demo-project-testing.yml; the leg
    verifies both the package and the DLL it installs, and exports the version the adapter checks.
    """
    install = next(
        step
        for step in load("conformance.yml")["jobs"]["native"]["steps"]
        if step.get("name") == "Install pinned WARP"
    )
    environment = install["env"]
    assert install["if"] == "runner.os == 'Windows'"
    assert environment["WARP_VERSION"] == "1.0.19"
    assert all(
        len(environment[name]) == 64
        and set(environment[name]) <= set("0123456789abcdef")
        for name in ("WARP_ARCHIVE_SHA256", "WARP_DLL_SHA256")
    )
    script = install["run"]
    assert "$env:WARP_ARCHIVE_SHA256" in script and "$env:WARP_DLL_SHA256" in script
    assert "microsoft.direct3d.warp/$env:WARP_VERSION/" in script
    assert "CROSSTL_CONFORMANCE_WARP_VERSION=$env:WARP_VERSION.0" in script


def test_fuzz_runs_weekly_on_linux_only():
    workflow = load("fuzz.yml")
    assert set(workflow["on"]) == {"schedule", "workflow_dispatch"}
    minute, hour, day, month, weekday = workflow["on"]["schedule"][0]["cron"].split()
    assert day == month == "*" and weekday != "*", "weekly"
    for job, definition in workflow["jobs"].items():
        assert definition["runs-on"].startswith("ubuntu-"), job


def test_harness_job_installs_the_fuzzing_dependency():
    install = next(
        step["run"]
        for step in load("conformance.yml")["jobs"]["harness"]["steps"]
        if "pip install" in step.get("run", "")
    )
    assert "hypothesis==" in install


# The plan.


@pytest.mark.parametrize(
    ("glob", "path", "expected"),
    [
        ("crosstl/**", "crosstl/translator/codegen/glsl.py", True),
        ("crosstl/**", "crosstl", False),
        ("crosstl/**", "tests/crosstl/x.py", False),
        ("tests/conformance/*.py", "tests/conformance/runner.py", True),
        ("tests/conformance/*.py", "tests/conformance/corpus/run.py", False),
        ("pyproject.toml", "pyproject.toml", True),
        ("pyproject.toml", "sub/pyproject.toml", False),
        ("a?c", "abc", True),
        ("a?c", "a/c", False),
        ("docs/v1.0/**", "docs/v1x0/a", False),
    ],
)
def test_globs_follow_github_paths_syntax(glob, path, expected):
    assert bool(ci_plan._pattern(glob).match(path)) is expected


@pytest.mark.parametrize(
    ("workflow", "path", "expected"),
    [
        ("conformance", "crosstl/backend/Metal/MetalLexer.py", True),
        ("conformance", "tests/conformance/corpus/run.py", True),
        ("conformance", ".github/workflows/conformance.yml", True),
        ("conformance", ".github/workflows/corpus.yml", False),
        ("conformance", "README.md", False),
        ("conformance", "demos/integrations/mlx/ci_plan.py", False),
        ("corpus", "tests/conformance/runner.py", True),
        ("corpus", "tests/conformance/corpus/manifest.json", True),
        ("corpus", "tests/conformance/fuzz/grammar.py", False),
        ("corpus", "tools/ensure_metal_toolchain.py", True),
        ("corpus", "docs/source/index.rst", False),
    ],
)
def test_relevance_by_path(workflow, path, expected):
    assert ci_plan.relevant(workflow, [path]) is expected


LINUX = [{"name": "Linux", "os": "ubuntu-24.04"}]
HOSTED = [
    {"name": "macOS", "os": "xcode-27"},
    {"name": "Windows", "os": "windows-2025"},
]


@pytest.mark.parametrize(
    ("workflow", "event", "full", "paths", "expected"),
    [
        # Unrelated changes skip everything, even under ci:full.
        ("conformance", "pull_request", False, ["README.md"], []),
        ("conformance", "pull_request", True, ["README.md"], []),
        ("conformance", "merge_group", False, ["docs/a.rst"], []),
        ("corpus", "merge_group", False, ["README.md"], []),
        # Relevant pull requests run Linux only unless labelled.
        ("conformance", "pull_request", False, ["crosstl/x.py"], LINUX),
        ("conformance", "pull_request", True, ["crosstl/x.py"], LINUX + HOSTED),
        ("corpus", "pull_request", False, ["crosstl/x.py"], LINUX),
        ("corpus", "pull_request", True, ["crosstl/x.py"], LINUX + HOSTED),
        # The merge queue runs every conformance leg and the corpus Linux leg.
        ("conformance", "merge_group", False, ["crosstl/x.py"], LINUX + HOSTED),
        ("corpus", "merge_group", False, ["crosstl/x.py"], LINUX),
        # Unknown changes count as relevant.
        ("conformance", "pull_request", False, None, LINUX),
        ("conformance", "merge_group", False, None, LINUX + HOSTED),
        # Unfiltered events run every leg.
        ("conformance", "push", False, None, LINUX + HOSTED),
        ("conformance", "schedule", False, None, LINUX + HOSTED),
        ("corpus", "schedule", False, None, LINUX + HOSTED),
        ("corpus", "workflow_dispatch", False, None, LINUX + HOSTED),
    ],
)
def test_plan(workflow, event, full, paths, expected):
    legs, reason = ci_plan.plan(workflow, event, full, LINUX, HOSTED, paths)
    assert legs == expected
    assert reason


@pytest.mark.skipif(GIT is None, reason="needs git")
def test_changed_paths_from_the_merge_base(tmp_path):
    def git(*arguments):
        return subprocess.run(
            [GIT, "-C", str(tmp_path), *arguments],
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "t",
                "GIT_AUTHOR_EMAIL": "t@example.com",
                "GIT_COMMITTER_NAME": "t",
                "GIT_COMMITTER_EMAIL": "t@example.com",
            },
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    (tmp_path / "README.md").write_text("a\n")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    base = git("rev-parse", "HEAD")
    git("checkout", "-q", "-b", "topic")
    (tmp_path / "crosstl").mkdir()
    (tmp_path / "crosstl" / "x.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "topic")
    head = git("rev-parse", "HEAD")
    # main moves on after the branch point; its change is not the pull request's.
    git("checkout", "-q", "main")
    (tmp_path / "docs.md").write_text("b\n")
    git("add", ".")
    git("commit", "-q", "-m", "main")
    moved = git("rev-parse", "HEAD")

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert ci_plan.changed_paths(moved, head, git=GIT) == ["crosstl/x.py"]
        assert ci_plan.changed_paths(base, head, git=GIT) == ["crosstl/x.py"]
        assert ci_plan.changed_paths("", head, git=GIT) is None
        assert ci_plan.changed_paths("0" * 40, head, git=GIT) is None
        assert ci_plan.changed_paths("f" * 40, head, git=GIT) is None
    finally:
        os.chdir(cwd)


@pytest.mark.skipif(not Path("/bin/bash").exists(), reason="needs bash")
@pytest.mark.parametrize("name", TIERED)
def test_plan_step_runs_as_written(name):
    workflow = load(name)
    linux, hosted = leg_lists(workflow)
    # Without a base the changed paths are unknown, so a pull request runs its Linux legs.
    assert run_plan_step(workflow, "pull_request") == linux
    assert run_plan_step(workflow, "pull_request", full=True) == linux + hosted
    assert run_plan_step(workflow, "schedule") == linux + hosted
    assert run_plan_step(workflow, "workflow_dispatch") == linux + hosted
    if GIT is not None and (ROOT / ".git").exists():
        head = subprocess.run(
            [GIT, "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # A pull request with no changes at all runs nothing.
        assert run_plan_step(workflow, "pull_request", base=head, head=head) == []
