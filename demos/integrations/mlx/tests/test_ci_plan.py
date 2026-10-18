"""Exercise MLX workflow selection and its required numerical smoke coverage."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from demos.integrations.mlx.ci_plan import changed_paths, main, validation_plan

ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github/workflows/demo-project-testing.yml"


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
        ("requirements-dev.txt", False, True),
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
    assert output.read_text() == f"heavy={heavy}\nsmoke=true\n"


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
def test_full_events_do_not_require_changed_paths(tmp_path, event, capsys):
    output = tmp_path / "output"
    assert main(["--event", event, "--output", str(output)]) == 0
    assert output.read_text() == "heavy=true\nsmoke=false\n"
    assert json.loads(capsys.readouterr().out) == {"heavy": True, "smoke": False}


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
    assert output.read_text() == "heavy=true\nsmoke=true\n"


def test_new_branch_gets_full_validation(tmp_path):
    output = tmp_path / "output"
    assert main(["--event", "push", "--base", "0" * 40, "--output", str(output)]) == 0
    assert output.read_text() == "heavy=true\nsmoke=true\n"


@pytest.mark.parametrize("base,head", [("", "a" * 40), ("a" * 40, "--help")])
def test_invalid_diff_revisions_fail_instead_of_skipping(base, head):
    with pytest.raises(ValueError, match="full Git commit hashes"):
        changed_paths(base, head)


def test_merge_queue_runs_every_tier(tmp_path, capsys):
    assert validation_plan("merge_group", ["docs/source/index.rst"]) == {
        "heavy": True,
        "smoke": True,
    }
    output = tmp_path / "output"
    assert main(["--event", "merge_group", "--output", str(output)]) == 0
    assert output.read_text() == "heavy=true\nsmoke=true\n"
    assert json.loads(capsys.readouterr().out) == {"heavy": True, "smoke": True}


def test_unsupported_event_fails():
    with pytest.raises(ValueError, match="Unsupported workflow event"):
        validation_plan("unknown", [])


def test_every_heavy_job_obeys_the_full_tier():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    jobs = workflow["jobs"]
    for name, job in jobs.items():
        if name in {"ci-plan", "mlx-smoke", "result"}:
            continue
        needs = job["needs"]
        assert "ci-plan" in (needs if isinstance(needs, list) else [needs]), name
        if name != "mlx-full-corpus-scout":
            assert job["if"] == "needs.ci-plan.outputs.heavy == 'true'", name
    assert workflow[True]["schedule"] == [{"cron": "31 4 * * *"}]
    plan = jobs["ci-plan"]
    tier = next(step for step in plan["steps"] if step.get("id") == "plan")
    assert tier["env"]["MLX_CI_FULL"] == (
        "${{ contains(github.event.pull_request.labels.*.name, 'ci:full') }}"
    )
    assert plan["runs-on"] == "ubuntu-24.04"
    assert plan["outputs"].keys() == {"heavy", "smoke"}
    checkout = next(
        step for step in plan["steps"] if step.get("uses") == "actions/checkout@v4"
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
