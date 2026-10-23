"""Guards for the GitHub Actions workflows.

Each gating workflow runs on every pull request and in the merge queue and reports one ``result`` check. Every job
has a timeout, every workflow declares its token permissions, and actions are referenced by release tag.

MLX's ``test_ci.py`` imports the workflow-text helpers below; keep their names and behaviour.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
ACTIONS_DIR = ROOT / ".github" / "actions"
RUNNER_OSES = {"ubuntu-latest", "windows-latest", "macOS-latest"}
GATING_WORKFLOWS = {
    "translator-tests.yml": "Translator Tests result",
    "backend-tests.yml": "Backend Tests result",
    "full-tests.yml": "Complete Test Suite result",
    "examples-test.yml": "Examples Test result",
    "coverage.yml": "Coverage result",
    "lint.yml": "Lint result",
    "demo-project-testing.yml": "MLX validation result",
}


def _workflows():
    return {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(WORKFLOW_DIR.glob("*.yml"))
    }


def _workflow_texts():
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(WORKFLOW_DIR.glob("*.yml"))
    }


@pytest.mark.parametrize("filename,check", sorted(GATING_WORKFLOWS.items()))
def test_gating_workflows_report_one_result_for_pull_requests_and_the_merge_queue(
    filename, check
):
    workflow = yaml.safe_load((WORKFLOW_DIR / filename).read_text(encoding="utf-8"))
    triggers = workflow[True]
    assert "merge_group" in triggers
    # A required check must report on every pull request, so skipping work is the
    # job of a planning step, never of a paths filter.
    pull_request = triggers["pull_request"] or {}
    assert not {"paths", "paths-ignore"} & set(pull_request), filename
    jobs = workflow["jobs"]
    results = [
        job for job in jobs.values() if str(job.get("name", "")).endswith(" result")
    ]
    assert results == [jobs["result"]]
    result = jobs["result"]
    assert result["name"] == check
    assert result["if"] == "always()"
    assert set(result["needs"]) == set(jobs) - {"result"}
    assert result["timeout-minutes"] <= 5
    step = result["steps"][0]
    assert step["env"]["RESULTS"] == "${{ join(needs.*.result, ' ') }}"
    assert "success|skipped) ;;" in step["run"]
    assert "exit 1" in step["run"]


def test_every_job_has_a_timeout():
    for filename, workflow in _workflows().items():
        for name, job in workflow["jobs"].items():
            if "uses" in job:
                continue
            assert isinstance(job.get("timeout-minutes"), int), (filename, name)
            assert 0 < job["timeout-minutes"] <= 360, (filename, name)


def test_workflows_declare_least_privilege_permissions():
    for filename, workflow in _workflows().items():
        scopes = [("workflow", workflow.get("permissions"))]
        scopes += [
            (name, job["permissions"])
            for name, job in workflow["jobs"].items()
            if "permissions" in job
        ]
        assert (
            scopes[0][1] is not None
        ), f"{filename} must declare top-level permissions"
        for where, permissions in scopes:
            assert isinstance(permissions, dict), (filename, where, permissions)
            assert set(permissions.values()) <= {"read", "write", "none"}, (
                filename,
                where,
            )
        # Writing to the repository is granted per job, never to the whole workflow.
        assert workflow["permissions"].get("contents", "read") != "write", filename


def test_no_workflow_runs_pull_request_code_with_a_privileged_token():
    for filename, workflow in _workflows().items():
        assert "pull_request_target" not in workflow[True], filename
        assert "workflow_run" not in workflow[True], filename


def test_the_coverage_comment_job_runs_no_pull_request_code():
    workflow = _workflows()["coverage.yml"]
    job = workflow["jobs"]["comment"]
    # It writes pull-request comments only, and only where the token belongs to this repository.
    assert job["permissions"] == {"pull-requests": "write"}
    assert "github.event.pull_request.head.repo.full_name == github.repository" in (
        job["if"]
    )
    # It reads the report job's comparison as text and never checks out or runs the pull request's code.
    uses = [step["uses"] for step in job["steps"] if "uses" in step]
    assert uses == ["actions/download-artifact@v8"]
    script = job["steps"][-1]["run"]
    assert not re.search(r"\b(?:python|pip)\b", script)


@pytest.mark.parametrize(
    "filename",
    [
        "backend-tests.yml",
        "translator-tests.yml",
        "full-tests.yml",
        "examples-test.yml",
        "demo.yml",
        "coverage.yml",
    ],
)
def test_test_workflows_drop_only_superseded_runs(filename):
    workflow = yaml.safe_load((WORKFLOW_DIR / filename).read_text(encoding="utf-8"))
    policy = workflow["concurrency"]
    # A new pull-request push cancels the older run. Pushes to main share one group,
    # so a burst of merges runs only the newest pending commit: the merge queue has
    # already tested each one. Merge-group, scheduled and dispatched runs never wait.
    assert policy == {
        "group": (
            "ci-${{ github.workflow }}-${{ github.event_name }}-"
            "${{ github.event.pull_request.number"
            " || (github.event_name == 'push' && github.ref)"
            " || github.run_id }}"
        ),
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }
    for job in workflow["jobs"].values():
        assert job.get("concurrency", {}).get("group") != policy["group"]


def test_pip_cache_is_only_enabled_for_jobs_that_install_with_pip():
    # actions/setup-python fails its post step when the pip cache folder was never
    # created, so a job that caches pip downloads must also install with pip.
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for name, job in (workflow.get("jobs") or {}).items():
            steps = job.get("steps") or []
            cached = any(
                str(step.get("uses", "")).startswith("actions/setup-python")
                and (step.get("with") or {}).get("cache") == "pip"
                for step in steps
            )
            if not cached:
                continue
            commands = "\n".join(str(step.get("run", "")) for step in steps)
            assert "pip install" in commands, (path.name, name)
            assert "--no-cache-dir" not in commands, (path.name, name)


def _steps():
    """``(file name, step)`` for every step of every workflow and local action."""
    paths = sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(
        ACTIONS_DIR.glob("*/action.yml")
    )
    for path in paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job in (document.get("jobs") or {}).values():
            for step in job.get("steps") or []:
                yield path.name, step
        for step in (document.get("runs") or {}).get("steps") or []:
            yield path.name, step


def test_actions_are_referenced_by_release_tag():
    # Release tags keep workflows readable and let Dependabot propose upgrades.
    # pypa/gh-action-pypi-publish publishes only its release/v1 branch.
    release_tag = re.compile(r"^[\w.-]+/[\w.-]+@(?:v\d+(?:\.\d+){0,2}|release/v\d+)$")
    for name, step in _steps():
        ref = str(step.get("uses", ""))
        if ref and not ref.startswith("./"):
            assert release_tag.match(ref), (name, ref)


def test_cache_paths_have_no_parent_directory_segments():
    # actions/cache restores a path that contains "..", but refuses to save it and
    # only warns, so such a cache silently never fills.
    for name, step in _steps():
        if not str(step.get("uses", "")).startswith("actions/cache"):
            continue
        for entry in str((step.get("with") or {}).get("path", "")).split():
            assert ".." not in entry.split("/"), (name, entry)


def _local_action_text(name):
    path = ACTIONS_DIR / name / "action.yml"
    return path.read_text(encoding="utf-8")


@pytest.mark.skipif(os.name != "posix", reason="Linux installer requires POSIX tools")
@pytest.mark.parametrize(
    "pinned,failure,success,downloads,checksum",
    [
        ("true", "", True, 1, True),
        ("true", "checksum", False, 1, True),
        ("true", "download", False, 1, False),
        ("false", "first-download", True, 2, False),
        ("invalid", "", False, 0, False),
    ],
)
def test_linux_dxc_installer_pinning(
    tmp_path, pinned, failure, success, downloads, checksum
):
    bash = shutil.which("bash")
    assert bash is not None
    mock_bin = tmp_path / "tools"
    mock_bin.mkdir()
    script = f"#!{sys.executable}\n" + textwrap.dedent("""\
        import json
        import os
        import sys
        from pathlib import Path

        tool = Path(sys.argv[0]).name
        args = sys.argv[1:]
        failure = os.environ["TEST_FAILURE"]
        record = {"tool": tool, "args": args}
        if tool == "sha256sum":
            record["stdin"] = sys.stdin.read()
        with Path(os.environ["TEST_LOG"]).open("a") as output:
            output.write(json.dumps(record) + "\\n")
        if tool == "curl":
            if failure == "download" or (
                failure == "first-download" and "/v1.9.2602.24/" in args[-1]
            ):
                sys.exit(1)
            Path(args[args.index("-fsSLo") + 1]).write_text("archive")
        elif tool == "sha256sum" and failure == "checksum":
            sys.exit(1)
        elif tool == "tar" and args[0] == "-xzf":
            binary = Path(args[args.index("-C") + 1]) / "bin/dxc"
            binary.parent.mkdir(parents=True)
            binary.write_text("#!/bin/sh\\nprintf 'pinned compiler\\n'\\n")
            binary.chmod(0o755)
        """)
    for name in ("curl", "tar", "sha256sum"):
        tool = mock_bin / name
        tool.write_text(script, encoding="utf-8")
        tool.chmod(0o755)
    output = tmp_path / "runner temp"
    output.mkdir()
    log = tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "PATH": str(mock_bin) + os.pathsep + os.environ["PATH"],
        "DXC_PINNED": pinned,
        "RUNNER_TEMP": str(output),
        "GITHUB_PATH": str(tmp_path / "github-path"),
        "GITHUB_ENV": str(tmp_path / "github-env"),
        "TEST_FAILURE": failure,
        "TEST_LOG": str(log),
    }
    body = textwrap.dedent(
        _local_action_text("install-linux-dxc").split("run: |\n", 1)[1]
    )
    result = subprocess.run(
        [bash, "-c", body], env=env, capture_output=True, text=True, timeout=30
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    calls = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    assert sum(call["tool"] == "curl" for call in calls) == downloads
    checksums = [call for call in calls if call["tool"] == "sha256sum"]
    assert bool(checksums) is checksum
    if checksum:
        assert checksums == [
            {
                "tool": "sha256sum",
                "args": ["--check", "--strict"],
                "stdin": (
                    "928b3e9986d11dc4279050e02340950c29bcbd1e5efb9d3ded9669dade37639d"
                    f"  {output / 'dxc.tar.gz'}\n"
                ),
            }
        ]
    assert (tmp_path / "github-path").exists() is success
    if not success:
        assert not any(
            call["tool"] == "tar" and "-xzf" in call["args"] for call in calls
        )


def _workflow_job_section(workflow, job_id):
    match = re.search(
        rf"^  {re.escape(job_id)}:\n(?P<body>.*?)(?=^  [a-z0-9-]+:\n|\Z)",
        workflow,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"workflow job {job_id!r} is missing"
    return match.group(0)


def _parse_matrix_values(raw):
    raw = raw.strip().strip("[]")
    return {
        item.strip().strip("\"'")
        for item in raw.split(",")
        if item.strip().strip("\"'")
    }


def _workflow_event_paths(workflow_text, event_name):
    event_header = f"  {event_name}:"
    lines = workflow_text.splitlines()
    for index, line in enumerate(lines):
        if line != event_header:
            continue
        section = []
        for section_line in lines[index + 1 :]:
            if (
                section_line.strip()
                and section_line.startswith("  ")
                and not section_line.startswith("    ")
            ):
                break
            section.append(section_line)
        break
    else:
        raise AssertionError(f"workflow event not found: {event_name}")

    for index, line in enumerate(section):
        if line.strip() != "paths:":
            continue
        indent = len(line) - len(line.lstrip())
        paths = []
        for item_line in section[index + 1 :]:
            item_indent = len(item_line) - len(item_line.lstrip())
            if item_line.strip() and item_indent <= indent:
                break
            item = re.match(r"^\s*-\s+(.+?)\s*$", item_line)
            if item:
                paths.append(item.group(1).strip("\"'"))
        return paths
    # An event without a paths filter runs the workflow for every change.
    return ["**"]


def _matrix_values(workflow_text, key):
    inline = re.search(
        rf"^\s*{re.escape(key)}:\s*(\[[^\n]+\])\s*$",
        workflow_text,
        flags=re.MULTILINE,
    )
    if inline:
        return _parse_matrix_values(inline.group(1))

    block = re.search(
        rf"^\s*{re.escape(key)}:\s*\n\s*\[\s*(.*?)\s*\]",
        workflow_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if block:
        return _parse_matrix_values(block.group(1))
    dynamic = re.search(
        rf"^\s*{re.escape(key)}:\s*"
        r"\$\{\{\s*fromJSON\(needs\.[\w-]+\.outputs\.[\w-]+\)\s*\}\}\s*$",
        workflow_text,
        flags=re.MULTILINE,
    )
    if dynamic:
        name = "FULL_" + key.upper().replace("-", "_") + "S"
        declared = re.search(
            rf"^\s*{name}:\s*'(\[[^\n]+\])'\s*$", workflow_text, flags=re.MULTILINE
        )
        assert declared, f"run-time matrix key {key} must declare {name}"
        return _parse_matrix_values(declared.group(1))
    raise AssertionError(f"matrix key not found: {key}")


def _load_ci_coverage_module():
    """Return the workflow-text helpers under the name the MLX tests still import."""
    from tools import ci_coverage

    return ci_coverage
