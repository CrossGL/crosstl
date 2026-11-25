"""The MLX workflow retains its evidence through one retrying upload action."""

import itertools
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github" / "workflows" / "demo-project-testing.yml"
ACTION_REF = "./demos/integrations/mlx/actions/upload-evidence"
ACTION = ROOT / ACTION_REF / "action.yml"
UPSTREAM = "actions/upload-artifact"
PASSED_THROUGH = (
    "name",
    "path",
    "if-no-files-found",
    "retention-days",
    "include-hidden-files",
)


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _action():
    return yaml.safe_load(ACTION.read_text(encoding="utf-8"))


def _upload_steps(workflow):
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            if step.get("uses") == ACTION_REF:
                yield job_name, job, step


def test_every_mlx_upload_goes_through_the_retrying_action():
    workflow = _workflow()
    direct = [
        (job_name, step.get("name"))
        for job_name, job in workflow["jobs"].items()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith(f"{UPSTREAM}@")
    ]
    assert direct == []
    uploads = list(_upload_steps(workflow))
    assert len(uploads) > 70
    declared = _action()["inputs"]
    required = {key for key, spec in declared.items() if spec.get("required")}
    for job_name, job, step in uploads:
        assert set(step["with"]) <= set(declared), (job_name, step["name"])
        assert required <= set(step["with"]), (job_name, step["name"])
        assert "continue-on-error" not in step, (job_name, step["name"])
        # A local action exists only after the repository is checked out at the workspace root.
        steps = job["steps"]
        checkouts = [
            other
            for other in steps[: steps.index(step)]
            if str(other.get("uses", "")).startswith("actions/checkout@")
            and "path" not in (other.get("with") or {})
        ]
        assert checkouts, (job_name, step["name"])


def test_the_upload_action_makes_three_attempts_and_the_last_one_decides():
    action = _action()
    assert action["runs"]["using"] == "composite"
    steps = action["runs"]["steps"]
    uploads = [step for step in steps if str(step.get("uses", "")).startswith(UPSTREAM)]
    assert len(uploads) == 3
    first, second, last = uploads
    assert first["id"] == "first" and second["id"] == "second"
    # The caller usually runs after a failure; inside a composite the default condition reads the job status.
    for step in steps:
        assert step["if"].startswith("always()"), step["name"]
    assert first["if"] == "always()"
    assert second["if"] == "always() && steps.first.outcome == 'failure'"
    assert last["if"] == "always() && steps.second.outcome == 'failure'"
    assert first["continue-on-error"] is True and second["continue-on-error"] is True
    assert "continue-on-error" not in last
    for step in uploads:
        for key in PASSED_THROUGH:
            assert step["with"][key] == f"${{{{ inputs.{key} }}}}", (step["name"], key)
    # Only a retry replaces what a failed attempt may have created under the same name.
    assert "overwrite" not in first["with"]
    assert second["with"]["overwrite"] is True and last["with"]["overwrite"] is True
    waits = [step for step in steps if "run" in step]
    assert [step["run"] for step in waits] == ["sleep 20", "sleep 60"]
    assert [step["if"] for step in waits] == [second["if"], last["if"]]
    assert steps.index(waits[0]) == steps.index(second) - 1
    assert steps.index(waits[1]) == steps.index(last) - 1


def test_dependabot_updates_the_upload_action():
    refs = {
        step["uses"]
        for step in _action()["runs"]["steps"]
        if str(step.get("uses", "")).startswith(UPSTREAM)
    }
    assert len(refs) == 1 and re.fullmatch(rf"{UPSTREAM}@v\d+(\.\d+){{0,2}}", *refs)
    config = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text("utf-8"))
    (entry,) = (
        update
        for update in config["updates"]
        if update["package-ecosystem"] == "github-actions"
    )
    assert "/" in entry["directories"]
    assert "/" + ACTION_REF.removeprefix("./") in entry["directories"]


def _matrix_legs(job):
    matrix = (job.get("strategy") or {}).get("matrix")
    if not matrix:
        return [{}]
    assert isinstance(matrix, dict), "MLX upload jobs use static matrices"
    original = {
        key: values
        for key, values in matrix.items()
        if key not in {"include", "exclude"}
    }
    for values in original.values():
        assert isinstance(values, list), "MLX upload jobs use static matrices"
    legs = [
        dict(zip(original, combination))
        for combination in itertools.product(*original.values())
    ]
    if legs == [{}]:
        legs = []
    for extra in matrix.get("include", []):
        matched = False
        for leg in legs:
            if all(
                leg[key] == value for key, value in extra.items() if key in original
            ):
                leg.update(extra)
                matched = True
        if not matched:
            legs.append(dict(extra))
    for removed in matrix.get("exclude", []):
        legs = [
            leg
            for leg in legs
            if not all(leg.get(key) == value for key, value in removed.items())
        ]
    return legs or [{}]


def _runner_os(label):
    label = label.lower()
    if label.startswith("ubuntu"):
        return "Linux"
    if label.startswith("windows"):
        return "Windows"
    if label.startswith(("macos", "xcode")):
        return "macOS"
    raise AssertionError(f"unknown runner label {label}")


def _render(template, leg, runner_os):
    def value(match):
        expression = match.group(1)
        if expression == "runner.os":
            return runner_os
        key = re.fullmatch(r"matrix\.([\w-]+)", expression)
        if key:
            return str(leg[key.group(1)])
        if not re.search(r"\bmatrix\.", expression):
            return match.group(0)
        # Any other expression over matrix values: legs that differ in those values may differ in the name.
        keys = sorted(set(re.findall(r"\bmatrix\.([\w-]+)", expression)))
        return "{" + ",".join(f"{key}={leg.get(key)!r}" for key in keys) + "}"

    return re.sub(r"\$\{\{\s*(.*?)\s*\}\}", value, template)


def test_mlx_artifact_names_are_unique_per_leg():
    # Retries replace an artifact of the same name, so two legs must never share one.
    owners = {}
    for job_name, job, step in _upload_steps(_workflow()):
        allowed = set(re.findall(r"runner\.os == '(\w+)'", str(step.get("if", ""))))
        for leg in _matrix_legs(job):
            runs_on = _render(str(job["runs-on"]), leg, "")
            runner_os = _runner_os(runs_on)
            if allowed and runner_os not in allowed:
                continue
            name = _render(step["with"]["name"], leg, runner_os)
            owner = (job_name, step["name"], tuple(sorted(leg.items(), key=str)))
            assert name not in owners, (name, owners.get(name), owner)
            owners[name] = owner
    assert len(owners) > 300
