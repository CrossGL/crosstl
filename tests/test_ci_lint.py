"""Guards for the Lint workflow, which runs the pinned pre-commit hooks on the files a change touches.

pre-commit.ci pushes the fixes of the same hooks to every open pull request, so a file on main that a hook would
rewrite costs each of them an extra commit and a rerun of its checks.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _hooks():
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    for repo in config["repos"]:
        for hook in repo["hooks"]:
            yield repo["repo"], hook


def _hook_step():
    workflow = yaml.safe_load((ROOT / ".github/workflows/lint.yml").read_text())
    steps = workflow["jobs"]["hooks"]["steps"]
    return next(step for step in steps if step.get("name") == "Run the pinned hooks")


def test_lint_runs_every_hook_but_the_local_test_hooks():
    # The local hooks run test files that the test workflows run; every other hook runs, black included.
    local = {
        hook["id"]
        for repo, hook in _hooks()
        if repo == "local" and hook.get("stages") != ["manual"]
    }
    assert local
    assert set(_hook_step()["env"]["SKIP"].split(",")) == local


def test_lint_runs_the_hooks_on_every_file_when_they_change():
    script = _hook_step()["run"]
    assert "--files" in script
    assert "--all-files" in script
    assert ".pre-commit-config" in script


def test_one_pass_of_the_hooks_leaves_nothing_for_a_second_pass():
    # pyupgrade can leave a typing import unused; autoflake removes it before isort and black lay out the rest.
    order = [hook["id"] for _, hook in _hooks()]
    assert (
        order.index("pyupgrade")
        < order.index("autoflake")
        < order.index("isort")
        < order.index("black")
    )
