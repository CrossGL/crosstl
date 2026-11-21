"""Exercise MLX workflow selection and its required numerical smoke coverage."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from demos.integrations.mlx import ci_conditions, ci_legs, smoke_canary
from demos.integrations.mlx.ci_plan import (
    PLAN_JOB,
    _without_plan_job,
    changed_legs,
    changed_paths,
    classify_workflow_change,
    main,
    validation_plan,
)
from demos.integrations.mlx.tests.test_ci_legs import DEMO_WORKFLOW
from demos.integrations.mlx.tests.test_ci_legs import TREE as DEMO_TREE

ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github/workflows/demo-project-testing.yml"
REAL_WORKFLOW = yaml.safe_load(WORKFLOW.read_text())
EVERY_LEG = sorted(ci_legs.workflow_legs(REAL_WORKFLOW))


@pytest.fixture(autouse=True)
def _planned_workflow(monkeypatch):
    # The temporary repositories hold no workflow, so plan the real one's legs.
    monkeypatch.setenv("MLX_CI_WORKFLOW", str(WORKFLOW))


def _outputs(path):
    """Return the step outputs a plan wrote, with ``legs`` and ``jobs`` decoded."""
    outputs = dict(line.split("=", 1) for line in path.read_text().splitlines())
    outputs["legs"] = json.loads(outputs["legs"])
    outputs["jobs"] = json.loads(outputs["jobs"])
    return outputs


def _tier(heavy, smoke, legs=None):
    if legs is None:
        legs = EVERY_LEG if heavy == "true" else []
    jobs = ci_legs.selected_jobs(REAL_WORKFLOW, legs) if legs else []
    return {"heavy": heavy, "smoke": smoke, "legs": legs, "jobs": jobs}


@pytest.mark.parametrize("event", ["pull_request", "push"])
@pytest.mark.parametrize(
    "path,heavy,smoke",
    [
        ("crosstl/translator/codegen/directx_codegen.py", False, True),
        ("crosstl/backend/Metal/MetalParser.py", False, True),
        ("tests/test_translator/test_codegen.py", False, True),
        ("tools/ensure_metal_toolchain.py", False, True),
        (".github/actions/setup-dxc/action.yml", False, True),
        ("pyproject.toml", False, True),
        ("requirements-dev.txt", False, False),
        ("requirements.txt", False, False),
        ("setup.cfg", False, False),
        ("support/features.json", False, False),
        ("demos/integrations/mlx/tests/kernels/test_current_arange.py", True, True),
        ("demos/integrations/mlx/portable_host/runtime.py", True, True),
        (
            "demos/integrations/mlx/contracts/arange.current-tree.translation.json",
            True,
            True,
        ),
        (".github/workflows/demo-project-testing.yml", True, True),
        ("demos/integrations/computesharp/project.toml", False, False),
        ("docs/source/index.rst", False, False),
    ],
)
def test_changed_paths_select_required_tier(event, path, heavy, smoke):
    # A pull request never queues the full tier on its own; only a push to main does.
    expected_heavy = heavy if event == "push" else False
    assert validation_plan(event, [path]) == {"heavy": expected_heavy, "smoke": smoke}


@pytest.mark.parametrize(
    "path",
    [
        "docs/source/index.rst",
        "crosstl/translator/codegen/directx_codegen.py",
        "demos/integrations/mlx/portable_host/runtime.py",
    ],
)
def test_pull_request_runs_full_tier_only_when_requested(path):
    assert validation_plan("pull_request", [path], full_requested=True) == {
        "heavy": True,
        "smoke": True,
    }
    assert validation_plan("pull_request", [path])["heavy"] is False


@pytest.mark.parametrize("full,heavy", [("true", "true"), ("false", "false")])
def test_pull_request_label_reaches_the_plan(tmp_path, monkeypatch, full, heavy):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / "README.md").write_text("start\n")
    base = _commit("Start")
    source = tmp_path / "demos/integrations/mlx/source.metal"
    source.parent.mkdir(parents=True)
    source.write_text("kernel void example() {}\n")
    head = _commit("Corpus change")
    output = tmp_path.parent / f"{tmp_path.name}-output"
    monkeypatch.setenv("MLX_CI_FULL", full)
    assert (
        main(
            [
                "--event",
                "pull_request",
                "--base",
                base,
                "--head",
                head,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert _outputs(output) == _tier(heavy, "true")


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
def test_full_events_do_not_require_changed_paths(tmp_path, event, capsys):
    output = tmp_path / "output"
    assert main(["--event", event, "--output", str(output)]) == 0
    assert _outputs(output) == _tier("true", "false")
    assert json.loads(capsys.readouterr().out) == {
        "heavy": True,
        "smoke": False,
        "legs": len(EVERY_LEG),
        "jobs": sorted({ci_legs.job_of(leg) for leg in EVERY_LEG}),
    }


def test_diff_includes_both_paths_of_renamed_mlx_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    source = tmp_path / "demos/integrations/mlx/source.metal"
    source.parent.mkdir(parents=True)
    source.write_text("kernel void example() {}\n")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "Add source",
        ],
        check=True,
    )
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source.rename(tmp_path / "archived source.metal")
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "Move source",
        ],
        check=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    paths = changed_paths(base, head)
    assert set(paths) == {
        "demos/integrations/mlx/source.metal",
        "archived source.metal",
    }
    assert validation_plan("push", paths) == {"heavy": True, "smoke": True}
    assert validation_plan("pull_request", paths) == {"heavy": False, "smoke": True}


def _commit(message):
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            message,
        ],
        check=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def test_base_branch_changes_after_branching_do_not_select_a_tier(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q", "-b", "main"], check=True)
    (tmp_path / "README.md").write_text("start\n")
    _commit("Start")
    subprocess.run(["git", "checkout", "-qb", "topic"], check=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/index.rst").write_text("topic\n")
    head = _commit("Topic documentation")
    subprocess.run(["git", "checkout", "-q", "main"], check=True)
    source = tmp_path / "demos/integrations/mlx/source.metal"
    source.parent.mkdir(parents=True)
    source.write_text("kernel void example() {}\n")
    base = _commit("Base corpus change")
    assert changed_paths(base, head) == ("docs/index.rst",)
    assert validation_plan("pull_request", changed_paths(base, head)) == {
        "heavy": False,
        "smoke": False,
    }


def test_push_whose_old_tip_is_absent_gets_full_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / "README.md").write_text("start\n")
    head = _commit("Start")
    output = tmp_path.parent / f"{tmp_path.name}-output"
    assert (
        main(
            [
                "--event",
                "push",
                "--base",
                "b" * 40,
                "--head",
                head,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert _outputs(output) == _tier("true", "true")


def test_new_branch_gets_full_validation(tmp_path):
    output = tmp_path / "output"
    assert main(["--event", "push", "--base", "0" * 40, "--output", str(output)]) == 0
    assert _outputs(output) == _tier("true", "true")


@pytest.mark.parametrize("base,head", [("", "a" * 40), ("a" * 40, "--help")])
def test_invalid_diff_revisions_fail_instead_of_skipping(base, head):
    with pytest.raises(ValueError, match="full Git commit hashes"):
        changed_paths(base, head)


def test_documentation_only_pull_request_skips_every_mlx_job():
    assert validation_plan("pull_request", ["docs/source/index.rst", "README.md"]) == {
        "heavy": False,
        "smoke": False,
    }


@pytest.mark.parametrize(
    "path,heavy",
    [
        ("docs/source/index.rst", False),
        ("README.md", False),
        (".github/workflows/conformance.yml", False),
        ("crosstl/translator/codegen/directx_codegen.py", False),
        ("tests/conformance/test_ci_tiers.py", False),
        ("demos/integrations/mlx/portable_host/runtime.py", True),
        (".github/workflows/demo-project-testing.yml", True),
    ],
)
def test_merge_queue_runs_full_tier_only_for_mlx_changes(path, heavy):
    # The queue builds one group at a time, so a group that cannot affect MLX
    # must not wait for the multi-hour native tier; smoke still gates it.
    assert validation_plan("merge_group", [path]) == {"heavy": heavy, "smoke": True}
    assert validation_plan("merge_group", [path]) == {
        "heavy": validation_plan("push", [path])["heavy"],
        "smoke": True,
    }


@pytest.mark.parametrize(
    "path,heavy",
    [
        ("docs/index.rst", "false"),
        ("crosstl/translator/parser.py", "false"),
        ("demos/integrations/mlx/source.metal", "true"),
    ],
)
def test_merge_group_reads_the_diff_from_its_base(tmp_path, monkeypatch, path, heavy):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / "README.md").write_text("start\n")
    base = _commit("Start")
    changed = tmp_path / path
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text("change\n")
    head = _commit("Queued change")
    output = tmp_path.parent / f"{tmp_path.name}-output"
    arguments = ["--event", "merge_group", "--base", base, "--head", head]
    assert main([*arguments, "--output", str(output)]) == 0
    assert _outputs(output) == _tier(heavy, "true")


def test_merge_group_without_its_base_gets_full_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / "README.md").write_text("start\n")
    head = _commit("Start")
    output = tmp_path.parent / f"{tmp_path.name}-output"
    arguments = ["--event", "merge_group", "--base", "c" * 40, "--head", head]
    assert main([*arguments, "--output", str(output)]) == 0
    assert _outputs(output) == _tier("true", "true")


def test_merge_group_without_revisions_fails(tmp_path):
    with pytest.raises(ValueError, match="full Git commit hashes"):
        main(["--event", "merge_group", "--output", str(tmp_path / "output")])


def test_plan_inputs_cover_pull_requests_merge_groups_and_pushes():
    plan = yaml.safe_load(WORKFLOW.read_text())["jobs"]["ci-plan"]
    tier = next(step for step in plan["steps"] if step.get("id") == "plan")
    assert tier["env"]["MLX_CI_BASE"] == (
        "${{ github.event.pull_request.base.sha || "
        "github.event.merge_group.base_sha || github.event.before }}"
    )
    assert tier["env"]["MLX_CI_HEAD"] == (
        "${{ github.event.pull_request.head.sha || "
        "github.event.merge_group.head_sha || github.sha }}"
    )


def _run_result_gate(results):
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["result"]
    step = next(step for step in job["steps"] if "run" in step)
    return subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env={"RESULTS": " ".join(results), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )


def test_result_gate_needs_every_job_and_passes_when_the_plan_skips_them():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    job = workflow["jobs"]["result"]
    assert job["if"] == "always()"
    assert set(job["needs"]) == set(workflow["jobs"]) - {"result"}
    skipped = ["skipped"] * (len(job["needs"]) - 2)
    assert _run_result_gate(["success", "success", *skipped]).returncode == 0


@pytest.mark.parametrize("outcome", ["failure", "cancelled"])
def test_result_gate_fails_when_a_job_that_ran_fails(outcome):
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["result"]
    results = ["success"] * (len(job["needs"]) - 1)
    results.insert(3, outcome)
    completed = _run_result_gate(results)
    assert completed.returncode == 1
    assert f"ended with '{outcome}'" in completed.stdout


@pytest.mark.parametrize("event", ["push", "merge_group"])
@pytest.mark.parametrize(
    "path",
    [
        "demos/integrations/mlx/ci_plan.py",
        "demos/integrations/mlx/smoke_canary.py",
        "demos/integrations/mlx/tests/test_ci_plan.py",
        "demos/integrations/mlx/tests/test_smoke_canary.py",
        "demos/integrations/mlx/README.md",
        "demos/integrations/mlx/portable_host/README.md",
        PLAN_JOB,
    ],
)
def test_planner_changes_need_only_the_smoke_tier(event, path):
    assert validation_plan(event, [path]) == {"heavy": False, "smoke": True}
    kernel = "demos/integrations/mlx/tests/kernels/test_current_arange.py"
    assert validation_plan(event, [path, kernel]) == {"heavy": True, "smoke": True}


def test_plan_job_split_removes_exactly_the_plan_job():
    text = WORKFLOW.read_text()
    workflow = yaml.safe_load(text)
    remaining = _without_plan_job(text)
    del workflow["jobs"]["ci-plan"]
    assert remaining == workflow


_PLANNED_WORKFLOW = """name: Demo
on:
  push:
    branches: [main]
jobs:
  ci-plan:
    runs-on: ubuntu-24.04
    steps:
      - run: echo {plan}
  native:
    needs: ci-plan
    if: needs.ci-plan.outputs.heavy == 'true'
    runs-on: windows-2025
    steps:
      - run: echo {native}
  host:
    needs: ci-plan
    if: needs.ci-plan.outputs.heavy == 'true'
    runs-on: ubuntu-24.04
    steps:
      - run: echo host
"""


@pytest.mark.parametrize(
    "plan,native,trigger,heavy,legs",
    [
        ("planner-two", "native", "main", "false", []),
        # A changed job runs alone; a changed trigger can alter every job.
        ("planner", "native-two", "main", "true", ["native"]),
        ("planner", "native", "release", "true", ["host", "native"]),
    ],
)
def test_workflow_change_confined_to_plan_job_skips_full_tier(
    tmp_path, monkeypatch, plan, native, trigger, heavy, legs
):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    workflow = tmp_path / ".github/workflows/demo-project-testing.yml"
    monkeypatch.setenv("MLX_CI_WORKFLOW", str(workflow))
    workflow.parent.mkdir(parents=True)
    workflow.write_text(_PLANNED_WORKFLOW.format(plan="planner", native="native"))
    base = _commit("Start")
    changed = _PLANNED_WORKFLOW.format(plan=plan, native=native)
    workflow.write_text(changed.replace("[main]", f"[{trigger}]"))
    head = _commit("Edit workflow")
    paths = classify_workflow_change(base, head, changed_paths(base, head))
    assert paths == (
        (PLAN_JOB,) if heavy == "false" else (str(WORKFLOW.relative_to(ROOT)),)
    )
    output = tmp_path.parent / f"{tmp_path.name}-output"
    arguments = ["--event", "merge_group", "--base", base, "--head", head]
    assert main([*arguments, "--output", str(output)]) == 0
    assert _outputs(output) == _tier(heavy, "true", legs)


_ANCHORED_WORKFLOW = """name: Demo
on:
  push:
    branches: [main]
jobs:
  ci-plan:
    runs-on: ubuntu-24.04
    steps:
      - &step-shared
        run: echo {shared}
  native:
    needs: ci-plan
    if: needs.ci-plan.outputs.heavy == 'true'
    runs-on: windows-2025
    steps:
      - *step-shared
"""


@pytest.mark.parametrize(
    "plan,shared,heavy,legs",
    [
        # A comment in the plan job changes nothing outside it.
        ("# planner", "shared", "false", []),
        # The plan job anchors a step that another job reuses: editing the
        # anchor edits that job too.
        ("", "shared-two", "true", ["native"]),
    ],
)
def test_anchored_plan_job_step_reaches_the_jobs_that_alias_it(
    tmp_path, monkeypatch, plan, shared, heavy, legs
):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    workflow = tmp_path / ".github/workflows/demo-project-testing.yml"
    monkeypatch.setenv("MLX_CI_WORKFLOW", str(workflow))
    workflow.parent.mkdir(parents=True)
    workflow.write_text(_ANCHORED_WORKFLOW.format(shared="shared"))
    base = _commit("Start")
    changed = _ANCHORED_WORKFLOW.format(shared=shared)
    workflow.write_text(
        changed.replace("    steps:\n", f"    steps:\n      {plan}\n", 1)
    )
    head = _commit("Edit workflow")
    paths = classify_workflow_change(base, head, changed_paths(base, head))
    assert paths == (
        (PLAN_JOB,) if heavy == "false" else (str(WORKFLOW.relative_to(ROOT)),)
    )
    output = tmp_path.parent / f"{tmp_path.name}-output"
    arguments = ["--event", "merge_group", "--base", base, "--head", head]
    assert main([*arguments, "--output", str(output)]) == 0
    assert _outputs(output) == _tier(heavy, "true", legs)


def test_unparsable_workflow_keeps_the_full_tier(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    workflow = tmp_path / ".github/workflows/demo-project-testing.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(_PLANNED_WORKFLOW.format(plan="planner", native="native"))
    base = _commit("Start")
    workflow.write_text(workflow.read_text() + "  broken: [\n")
    head = _commit("Break workflow")
    paths = changed_paths(base, head)
    assert classify_workflow_change(base, head, paths) == paths


def test_added_workflow_keeps_the_full_tier(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / "README.md").write_text("start\n")
    base = _commit("Start")
    workflow = tmp_path / ".github/workflows/demo-project-testing.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(_PLANNED_WORKFLOW.format(plan="planner", native="native"))
    head = _commit("Add workflow")
    paths = changed_paths(base, head)
    assert classify_workflow_change(base, head, paths) == paths
    assert validation_plan("push", paths)["heavy"] is True


def test_unsupported_event_fails():
    with pytest.raises(ValueError, match="Unsupported workflow event"):
        validation_plan("unknown", [])


def test_every_heavy_job_obeys_the_full_tier():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    jobs = workflow["jobs"]
    for name, job in jobs.items():
        if name in {"ci-plan", "mlx-smoke", "result"}:
            continue
        if name == "warp-conformance":
            # Fixed HLSL controls and the translated reduction on every pinned
            # WARP library: a short Windows job that runs with either tier.
            assert job["needs"] == "ci-plan"
            assert job["if"] == (
                "needs.ci-plan.outputs.smoke == 'true' || "
                "needs.ci-plan.outputs.heavy == 'true'"
            )
            continue
        needs = job["needs"]
        assert "ci-plan" in (needs if isinstance(needs, list) else [needs]), name
        if name != "mlx-full-corpus-scout":
            assert job["if"] == ci_conditions.job_condition(name), name
    assert workflow[True]["schedule"] == [{"cron": "31 4 * * *"}]
    plan = jobs["ci-plan"]
    tier = next(step for step in plan["steps"] if step.get("id") == "plan")
    assert tier["env"]["MLX_CI_FULL"] == (
        "${{ contains(github.event.pull_request.labels.*.name, 'ci:full') }}"
    )
    assert plan["runs-on"] == "ubuntu-24.04"
    assert plan["outputs"].keys() == {"heavy", "smoke", "legs", "jobs"}
    assert plan["outputs"]["jobs"] == "${{ steps.plan.outputs.jobs }}"
    assert plan["outputs"]["legs"] == "${{ steps.plan.outputs.legs }}"
    checkout = next(
        step
        for step in plan["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["fetch-depth"] == 0
    assert checkout["with"]["persist-credentials"] is False


def test_smoke_runs_pinned_integer_and_float_numerics():
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["mlx-smoke"]
    assert job["needs"] == "ci-plan"
    assert job["if"] == "needs.ci-plan.outputs.smoke == 'true'"
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] <= 15
    step = next(
        step for step in job["steps"] if step.get("name") == "Run pinned MLX smoke"
    )
    assert step["env"]["CROSTL_REQUIRE_MLX_CURRENT_ARANGE"] == "1"
    assert step["env"]["CROSTL_REQUIRE_MLX_SMOKE"] == "1"
    assert "demos/integrations/mlx/tests/test_smoke.py" in step["run"]
    dxc = next(
        step
        for step in job["steps"]
        if step.get("uses") == "./.github/actions/install-linux-dxc"
    )
    assert dxc["with"]["pinned"] == "true"
    assert step["env"]["CROSTL_MLX_CURRENT_TARGET"] == "opengl"
    assert "--timeout-seconds 300" in step["run"]
    assert "-n auto" in step["run"]
    for entry in ("arangeint32", "arangefloat32"):
        assert f"test_current_mlx_arange_native_parity[{entry}]" in step["run"]
    assert "continue-on-error" not in step


def test_smoke_runs_the_complete_family_canary():
    """Translator changes reach main through the smoke tier, so it samples every family."""
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["mlx-smoke"]
    assert job["env"]["MLX_CORPUS_COMMIT"] == smoke_canary.CORPUS_COMMIT
    names = [step.get("name") for step in job["steps"]]
    checkout = job["steps"][names.index("Checkout complete-family MLX kernels")]
    canary = job["steps"][names.index("Run complete-family canary")]
    assert names.index("Install pinned Linux DXC") < names.index(canary["name"])
    assert names.index(canary["name"]) < names.index("Upload smoke diagnostics")
    assert 'fetch --depth=1 origin "$MLX_CORPUS_COMMIT"' in checkout["run"]
    assert "sparse-checkout set mlx/backend/metal/kernels" in checkout["run"]
    directory = "mlx-canary-upstream"
    assert f"git init {directory}" in checkout["run"]
    assert canary["env"] == {
        smoke_canary.ROOT_ENV: f"${{{{ github.workspace }}}}/{directory}"
    }
    run = canary["run"]
    assert run.startswith("set -euo pipefail\n")
    assert "--timeout-seconds 600" in run
    assert "python -m demos.integrations.mlx.smoke_canary" in run
    assert "--junitxml=mlx-smoke-results/canary.xml" in run
    assert "continue-on-error" not in canary


def _demo_repository(tmp_path, monkeypatch):
    """Commit the leg tests' small MLX demo and its workflow; return the base."""
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    workflow = tmp_path / ".github/workflows/demo-project-testing.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(DEMO_WORKFLOW)
    monkeypatch.setenv("MLX_CI_WORKFLOW", str(workflow))
    for name, text in DEMO_TREE.items():
        path = tmp_path / ci_legs.MLX / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return _commit("Start")


def _plan_change(tmp_path, base, event="merge_group"):
    head = _commit("Change")
    output = tmp_path.parent / f"{tmp_path.name}-output"
    arguments = ["--event", event, "--base", base, "--head", head]
    assert main([*arguments, "--output", str(output)]) == 0
    return _outputs(output)


@pytest.mark.parametrize("event", ["merge_group", "push"])
@pytest.mark.parametrize(
    "path,legs",
    [
        ("sort_helper.py", ["parity[sort]"]),
        ("portable_host/runtime.py", ["host"]),
        ("tests/kernels/conftest.py", ["parity[scan]", "parity[sort]"]),
        ("orphan.py", ["host", "parity[scan]", "parity[sort]", "translator"]),
        ("tests/kernels/test_unused.py", []),
    ],
)
def test_queue_groups_and_pushes_select_the_legs_that_read_a_change(
    tmp_path, monkeypatch, event, path, legs
):
    base = _demo_repository(tmp_path, monkeypatch)
    changed = tmp_path / ci_legs.MLX / path
    changed.write_text(changed.read_text() + "\n# changed\n")
    assert _plan_change(tmp_path, base, event) == _tier("true", "true", legs)


@pytest.mark.parametrize("path", ["ci_legs.py", "README.md"])
def test_planner_and_prose_changes_select_no_leg(tmp_path, monkeypatch, path):
    base = _demo_repository(tmp_path, monkeypatch)
    changed = tmp_path / ci_legs.MLX / path
    changed.write_text("change\n")
    assert _plan_change(tmp_path, base) == _tier("false", "true", [])


def test_new_mlx_file_that_no_leg_reaches_selects_every_leg(tmp_path, monkeypatch):
    base = _demo_repository(tmp_path, monkeypatch)
    (tmp_path / ci_legs.MLX / "new_module.py").write_text("")
    every = ["host", "parity[scan]", "parity[sort]", "translator"]
    assert _plan_change(tmp_path, base) == _tier("true", "true", every)


def test_workflow_and_file_changes_select_the_union_of_their_legs(
    tmp_path, monkeypatch
):
    base = _demo_repository(tmp_path, monkeypatch)
    workflow = Path(os.environ["MLX_CI_WORKFLOW"])
    workflow.write_text(DEMO_WORKFLOW.replace("tests/test_translator", "tests/x"))
    (tmp_path / ci_legs.MLX / "sort_helper.py").write_text("# changed\n")
    legs = ["parity[sort]", "translator"]
    assert _plan_change(tmp_path, base) == _tier("true", "true", legs)


def test_pull_request_with_full_label_runs_every_leg(tmp_path, monkeypatch):
    base = _demo_repository(tmp_path, monkeypatch)
    (tmp_path / ci_legs.MLX / "sort_helper.py").write_text("# changed\n")
    monkeypatch.setenv("MLX_CI_FULL", "true")
    every = ["host", "parity[scan]", "parity[sort]", "translator"]
    assert _plan_change(tmp_path, base, "pull_request") == _tier("true", "true", every)


def test_a_failed_leg_selection_runs_every_leg_with_a_warning(
    tmp_path, monkeypatch, capsys
):
    base = _demo_repository(tmp_path, monkeypatch)
    (tmp_path / ci_legs.MLX / "sort_helper.py").write_text("# changed\n")

    def fail(*_):
        raise RuntimeError("scan failed")

    monkeypatch.setattr(ci_legs, "affected_legs", fail)
    every = ["host", "parity[scan]", "parity[sort]", "translator"]
    assert _plan_change(tmp_path, base) == _tier("true", "true", every)
    assert (
        "::warning title=MLX leg plan::RuntimeError: scan failed; "
        "running every heavy leg" in capsys.readouterr().err
    )


def test_real_tree_harness_change_selects_only_the_job_that_runs_it(monkeypatch):
    # The planner reads the checkout it runs in, as the ci-plan job does.
    monkeypatch.chdir(ROOT)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    workflow = yaml.safe_load(WORKFLOW.read_text())
    legs = ci_legs.workflow_legs(workflow)
    path = f"{ci_legs.MLX}/run_porting.py"
    assert changed_legs(workflow, legs, head, head, [path]) == {"mlx-metal-porting"}
