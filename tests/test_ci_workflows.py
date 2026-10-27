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
# The job that decides whether a push to main repeats a commit the merge queue
# already tested, per workflow.
QUEUE_TESTED_WORKFLOWS = {
    "computesharp.yml": "plan",
    "backend-tests.yml": "python-versions",
    "translator-tests.yml": "python-versions",
    "full-tests.yml": "plan",
    "examples-test.yml": "python-versions",
}
RUN_GROUP = "${{ github.event.pull_request.number || github.run_id }}"
MAIN_PUSH_GROUP = (
    "${{ github.event.pull_request.number"
    " || (github.event_name == 'push' && github.ref)"
    " || github.run_id }}"
)


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
        if "workflow_run" not in workflow[True]:
            continue
        # A workflow_run workflow holds this repository's token even after a fork's run. It checks out main only,
        # keeps the token out of git, unpacks artifacts outside the workspace and passes the triggering run's values
        # to scripts through `env`, so none of them is ever expanded into a script.
        for name, job in workflow["jobs"].items():
            for step in job["steps"]:
                action = step.get("uses", "").split("@")[0]
                if action == "actions/checkout":
                    assert step["with"] == {"persist-credentials": False}, (
                        filename,
                        name,
                    )
                if action == "actions/download-artifact":
                    assert step["with"]["path"].startswith("${{ runner.temp }}/")
                assert "${{" not in step.get("run", ""), (filename, name, step)


def test_the_coverage_comment_runs_only_main_code_on_the_report():
    workflow = _workflows()["coverage-comment.yml"]
    assert workflow[True] == {
        "workflow_run": {"workflows": ["Coverage"], "types": ["completed"]}
    }
    job = workflow["jobs"]["comment"]
    assert "github.event.workflow_run.event == 'pull_request'" in job["if"]
    assert job["permissions"] == {
        "actions": "read",
        "contents": "read",
        "pull-requests": "write",
    }
    uses = [step["uses"] for step in job["steps"] if "uses" in step]
    assert uses == [
        "actions/checkout@v7",
        "actions/setup-python@v7",
        "actions/download-artifact@v8",
    ]
    # main's tools/coverage reads the run's coverage.json as data; nothing is installed or run from the artifact.
    # The pull request's head reaches it through env.
    scripts = "\n".join(step.get("run", "") for step in job["steps"])
    scripts = re.sub(r"\s*\\\n\s*", " ", scripts)
    assert not re.search(r"\bpip\b", scripts)
    assert re.findall(r"\bpython\b.*", scripts) == [
        'python -m tools.coverage compare published.json "$REPORT" --repository "$GITHUB_REPOSITORY"'
        ' --head "$HEAD_SHA" > compare.md'
    ]


STATUS_FUNCTION = re.compile(r"\b(?:always|cancelled|failure|success)\(\)")


def _needs(job):
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else needs


def _ancestors(jobs, name):
    found, pending = set(), list(_needs(jobs[name]))
    while pending:
        need = pending.pop()
        if need not in found:
            found.add(need)
            pending.extend(_needs(jobs[need]))
    return found


def test_jobs_after_a_skippable_leg_check_their_own_status():
    # A condition without a status function implies `success()`, which is false when any job up the `needs` chain
    # was skipped, even if the direct dependency ran because it calls `always()`. Such a job is then skipped too,
    # as the coverage comment was on every pull request that planned out a leg.
    for filename, workflow in _workflows().items():
        jobs = workflow["jobs"]
        for name, job in jobs.items():
            if STATUS_FUNCTION.search(str(job.get("if", ""))):
                continue
            for need in _needs(job):
                runs_anyway = STATUS_FUNCTION.search(str(jobs[need].get("if", "")))
                skippable = [a for a in _ancestors(jobs, need) if "if" in jobs[a]]
                assert not (runs_anyway and skippable), (filename, name, need)


@pytest.mark.parametrize(
    "filename,group",
    [
        *((name, RUN_GROUP) for name in sorted(QUEUE_TESTED_WORKFLOWS)),
        # The demo has no merge-group run, and coverage publishes from its own push
        # run, so pushes to main share one group: a burst of merges runs them only
        # for the newest commit.
        ("demo.yml", MAIN_PUSH_GROUP),
        ("coverage.yml", MAIN_PUSH_GROUP),
    ],
)
def test_test_workflows_drop_only_superseded_runs(filename, group):
    # A cancelled run turns the workflow's status badge red, so only pull-request
    # runs and pushes that have a newer push waiting behind them are ever dropped.
    workflow = yaml.safe_load((WORKFLOW_DIR / filename).read_text(encoding="utf-8"))
    policy = workflow["concurrency"]
    assert policy == {
        "group": "ci-${{ github.workflow }}-${{ github.event_name }}-" + group,
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }
    for job in workflow["jobs"].values():
        assert job.get("concurrency", {}).get("group") != policy["group"]


@pytest.mark.parametrize("filename,plan", sorted(QUEUE_TESTED_WORKFLOWS.items()))
def test_pushes_to_main_skip_a_commit_the_merge_queue_already_passed(filename, plan):
    workflow = yaml.safe_load((WORKFLOW_DIR / filename).read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    planner = jobs[plan]
    assert planner["permissions"] == (
        {"actions": "read", "contents": "read"}
        if filename == "computesharp.yml"
        else {"actions": "read"}
    )
    assert planner["outputs"]["tested"] == "${{ steps.queue.outputs.tested }}"
    (step,) = (step for step in planner["steps"] if step.get("id") == "queue")
    assert step["if"] == "github.event_name == 'push'"
    # A failed lookup must run the tests rather than skip them.
    assert "|| echo 0)" in step["run"]
    if filename == "computesharp.yml":
        # Each merge group selects the port's jobs from its own change, so a push
        # that lands several groups checks every pushed commit (after checkout).
        assert "head_sha=$sha&event=merge_group&status=success" in step["run"]
        assert 'git rev-list "$BEFORE..$GITHUB_SHA"' in step["run"]
        assert planner["steps"][0]["with"]["fetch-depth"] == 0
    else:
        assert "head_sha=$GITHUB_SHA&event=merge_group&status=success" in step["run"]
        reference = yaml.safe_load(
            (WORKFLOW_DIR / "translator-tests.yml").read_text(encoding="utf-8")
        )["jobs"]["python-versions"]["steps"][0]
        assert step == reference
    for name, job in jobs.items():
        if name in {plan, "result"}:
            continue
        needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
        assert plan in needs, (filename, name)
        guard = f"needs.{plan}.outputs.tested != 'true'"
        if filename == "computesharp.yml":
            # Keep the port's independent run/execute filters and cancellation rule.
            assert job["if"].startswith("${{ " + guard + " && "), (filename, name)
        else:
            assert job["if"] == guard, (filename, name)


@pytest.mark.skipif(
    sys.platform == "win32" or not (shutil.which("bash") and shutil.which("git")),
    reason="Bash and Git required",
)
@pytest.mark.parametrize(
    "old_tip,passed,api_error,expected",
    [
        # One merge group landed, and its run passed.
        (2, {3}, False, "true"),
        # One push landed three merge groups, and every run passed.
        (0, {1, 2, 3}, False, "true"),
        # Only the newest group passed; an earlier one failed, was cancelled or is
        # still running, and the newest run may have skipped the port.
        (0, {3}, False, "false"),
        # No group passed this commit: an admin merge that bypassed the queue, a
        # failed or cancelled run, or a group that tested a different commit.
        (2, set(), False, "false"),
        # The API fails: gh prints the error body and exits non-zero.
        (2, {3}, True, "false"),
        # The old tip is not in the clone.
        (None, {1, 2, 3}, False, "false"),
    ],
)
def test_computesharp_queue_lookup_requires_every_pushed_commit_to_pass(
    tmp_path, old_tip, passed, api_error, expected
):
    workflow = yaml.safe_load(
        (WORKFLOW_DIR / "computesharp.yml").read_text(encoding="utf-8")
    )
    step = next(
        step for step in workflow["jobs"]["plan"]["steps"] if step.get("id") == "queue"
    )
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }

    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *arguments],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        ).stdout.strip()

    git("init", "-q")
    commits = []
    for index in range(4):
        git("commit", "-q", "--allow-empty", "-m", f"merge group {index}")
        commits.append(git("rev-parse", "HEAD"))
    output = tmp_path / "output"
    args = tmp_path / "args"
    # Run the actual workflow shell with a local API substitute, without network access.
    stub = textwrap.dedent("""\
        gh() {
          printf '%s\\n' "$*" >> "$LOOKUP_ARGS"
          if [ -n "$API_ERROR" ]; then
            printf '%s' '{"message":"Not Found","status":"404"}'
            return 1
          fi
          local sha="${2#*head_sha=}"
          case " $PASSED " in *" ${sha%%&*} "*) printf 1 ;; *) printf 0 ;; esac
        }
        """)
    subprocess.run(
        ["bash", "-e", "-c", stub + step["run"]],
        check=True,
        cwd=tmp_path,
        env={
            **env,
            "API_ERROR": "1" if api_error else "",
            "PASSED": " ".join(commits[index] for index in passed),
            "BEFORE": "0" * 40 if old_tip is None else commits[old_tip],
            "GITHUB_OUTPUT": str(output),
            "LOOKUP_ARGS": str(args),
            "GITHUB_REPOSITORY": "CrossGL/crosstl",
            "GITHUB_SHA": commits[-1],
            "GITHUB_WORKFLOW_REF": (
                "CrossGL/crosstl/.github/workflows/computesharp.yml@refs/heads/main"
            ),
        },
    )
    assert output.read_text().strip() == f"tested={expected}"
    if expected == "true":
        assert args.read_text().splitlines() == [
            "api repos/CrossGL/crosstl/actions/workflows/computesharp.yml/runs"
            f"?head_sha={sha}&event=merge_group&status=success --jq .total_count"
            for sha in reversed(commits[old_tip + 1 :])
        ]


def test_conformance_pushes_skip_a_commit_the_merge_queue_already_passed():
    # Every conformance job already waits for the leg selection, so the planner
    # skips that step instead and leaves `run` empty.
    workflow = yaml.safe_load(
        (WORKFLOW_DIR / "conformance.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    planner = jobs["legs"]
    assert planner["permissions"] == {"actions": "read", "contents": "read"}
    assert planner["outputs"]["tested"] == "${{ steps.queue.outputs.tested }}"
    assert planner["outputs"]["run"] == "${{ steps.select.outputs.run }}"
    steps = {step.get("id"): index for index, step in enumerate(planner["steps"])}
    reference = yaml.safe_load(
        (WORKFLOW_DIR / "translator-tests.yml").read_text(encoding="utf-8")
    )["jobs"]["python-versions"]["steps"][0]
    assert planner["steps"][steps["queue"]] == reference
    select = planner["steps"][steps["select"]]
    assert steps["queue"] < steps["select"]
    assert select["if"] == "steps.queue.outputs.tested != 'true'"
    for name, job in jobs.items():
        if name in {"legs", "result"}:
            continue
        assert job["needs"] == "legs", name
        assert job["if"] == "needs.legs.outputs.run == 'true'", name


@pytest.mark.skipif(
    sys.platform == "win32" or not shutil.which("bash"), reason="Bash required"
)
@pytest.mark.parametrize(
    "response,status,expected",
    [
        # One merge-group run of this workflow passed on the pushed commit.
        ("1", 0, "true"),
        # A requeued group can pass the same commit twice.
        ("2", 0, "true"),
        # No run passed: an admin merge that bypassed the queue, or a failed,
        # cancelled or still running group.
        ("0", 0, "false"),
        # The API fails: gh prints the error body and exits non-zero.
        ('{"message":"Not Found","status":"404"}', 1, "false"),
        # The API fails without a body.
        ("", 1, "false"),
    ],
)
def test_queue_lookup_skips_a_push_only_when_a_merge_group_run_passed(
    tmp_path, response, status, expected
):
    # translator-tests.yml holds the step every other queue-tested workflow copies.
    step = yaml.safe_load(
        (WORKFLOW_DIR / "translator-tests.yml").read_text(encoding="utf-8")
    )["jobs"]["python-versions"]["steps"][0]
    output = tmp_path / "output"
    args = tmp_path / "args"
    # Run the actual workflow shell with a local API substitute, without network access.
    stub = textwrap.dedent("""\
        gh() {
          printf '%s\\n' "$*" >> "$LOOKUP_ARGS"
          printf '%s' "$RESPONSE"
          return "$STATUS"
        }
        """)
    subprocess.run(
        ["bash", "-e", "-c", stub + step["run"]],
        check=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "RESPONSE": response,
            "STATUS": str(status),
            "GITHUB_OUTPUT": str(output),
            "LOOKUP_ARGS": str(args),
            "GITHUB_REPOSITORY": "CrossGL/crosstl",
            "GITHUB_SHA": "0123abcd",
            "GITHUB_WORKFLOW_REF": (
                "CrossGL/crosstl/.github/workflows/backend-tests.yml@refs/heads/main"
            ),
        },
        capture_output=True,
    )
    assert output.read_text().strip() == f"tested={expected}"
    # The lookup asks about this workflow's runs only, not every workflow's.
    assert args.read_text().splitlines() == [
        "api repos/CrossGL/crosstl/actions/workflows/backend-tests.yml/runs"
        "?head_sha=0123abcd&event=merge_group&status=success --jq .total_count"
    ]


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
    # The optional third segment admits sub-actions such as actions/cache/save.
    release_tag = re.compile(r"^[\w.-]+/[\w.-]+(?:/[\w.-]+)?@v\d+(?:\.\d+){0,2}$")
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


_APT_STUB = """\
    import json
    import os
    import sys
    from pathlib import Path

    tool = Path(sys.argv[0]).name
    args = sys.argv[1:]
    if tool == "timeout":
        os.execvp(args[1], args[1:])
    if tool == "sleep":
        sys.exit(0)
    if args[0] != "apt-get":
        os.execvp(args[0], args)
    log = Path(os.environ["TEST_LOG"])
    earlier = log.read_text().splitlines() if log.exists() else []
    if "update" in args:
        operation = "update"
    elif "--download-only" in args:
        operation = "download"
    elif any(arg.endswith(".deb") for arg in args):
        operation = "cached"
    else:
        operation = "install"
    with log.open("a") as output:
        output.write(json.dumps({"operation": operation, "args": args}) + "\\n")
    failure = os.environ["TEST_FAILURE"]
    # "first-<operation>" fails only the first call of that operation.
    first = failure == "first-" + operation and not any(
        json.loads(line)["operation"] == operation for line in earlier
    )
    if failure == operation or first:
        sys.exit(100)
    if operation == "download":
        prefix = "Dir::Cache::Archives="
        archives = next(arg[len(prefix):] for arg in args if arg.startswith(prefix))
        Path(archives, "glslang-tools.deb").write_text("")
    """


def _apt_install_step(step_id):
    document = yaml.safe_load(_local_action_text("apt-install"))
    (step,) = (s for s in document["runs"]["steps"] if s.get("id") == step_id)
    return step


@pytest.mark.skipif(os.name != "posix", reason="apt installer requires POSIX tools")
@pytest.mark.parametrize(
    "recommends,cached,failure,success,operations",
    [
        ("true", [], "", True, ["update", "download", "install"]),
        # An update that fails on one source still leaves the lists the download needs.
        ("false", [], "first-update", True, ["update", "download", "install"]),
        ("true", [], "update", True, ["update", "download", "install"]),
        ("true", [], "first-download", True, ["update", "download"] * 2 + ["install"]),
        ("true", [], "download", False, ["update", "download"] * 3),
        ("true", [], "install", False, ["update", "download", "install"]),
        # A warm cache installs the archives it holds and never contacts a mirror.
        ("false", ["b.deb", "a.deb"], "", True, ["cached"]),
        # Archives that no longer install on the image are downloaded again.
        (
            "true",
            ["a.deb"],
            "cached",
            True,
            ["cached", "update", "download", "install"],
        ),
    ],
)
def test_apt_installer_retries_only_the_network_steps(
    tmp_path, recommends, cached, failure, success, operations
):
    bash = shutil.which("bash")
    assert bash is not None
    mock_bin = tmp_path / "tools"
    mock_bin.mkdir()
    script = f"#!{sys.executable}\n" + textwrap.dedent(_APT_STUB)
    for name in ("sudo", "timeout", "sleep"):
        tool = mock_bin / name
        tool.write_text(script, encoding="utf-8")
        tool.chmod(0o755)
    archives = tmp_path / "apt-archives"
    archives.mkdir()
    for name in cached:
        (archives / name).write_text("")
    log = tmp_path / "calls.jsonl"
    outputs = tmp_path / "github-output"
    outputs.write_text("")
    env = {
        **os.environ,
        "PATH": str(mock_bin) + os.pathsep + os.environ["PATH"],
        "PACKAGES": "glslang-tools spirv-tools=2024.1-1",
        "RECOMMENDS": recommends,
        "CACHED": "true" if cached else "",
        "ARCHIVES": str(archives),
        "GITHUB_OUTPUT": str(outputs),
        "TEST_FAILURE": failure,
        "TEST_LOG": str(log),
    }
    body = _apt_install_step("install")["run"]
    result = subprocess.run(
        [bash, "-c", body], env=env, capture_output=True, text=True, timeout=30
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call["operation"] for call in calls] == operations
    for call in calls:
        assert call["args"][0] == "apt-get"
        assert f"Dir::Cache::Archives={archives}/" in call["args"]
        assert ("--no-install-recommends" in call["args"]) is (recommends == "false")
        if call["operation"] == "cached":
            assert call["args"][-len(cached) :] == [
                str(archives / name) for name in sorted(cached)
            ]
        elif call["operation"] != "update":
            assert call["args"][-2:] == ["glslang-tools", "spirv-tools=2024.1-1"]
    downloaded = success and operations[-1] == "install"
    assert ("downloaded=true" in outputs.read_text()) is downloaded
    assert not (archives / "partial").exists() or not downloaded


def test_apt_installer_saves_only_new_downloads():
    steps = yaml.safe_load(_local_action_text("apt-install"))["runs"]["steps"]
    restore = next(
        s for s in steps if s.get("uses", "").startswith("actions/cache/restore")
    )
    save = next(s for s in steps if s.get("uses", "").startswith("actions/cache/save"))
    assert "steps.restore.outputs.cache-hit != 'true'" in save["if"]
    assert "steps.install.outputs.downloaded == 'true'" in save["if"]
    archives = _apt_install_step("install")["env"]["ARCHIVES"]
    assert restore["with"]["path"] == save["with"]["path"] == archives
    assert restore["with"]["key"] == save["with"]["key"]
    # A new runner image ships other packages, so its archives must not come from an older image's cache.
    key = _apt_install_step("key")["run"]
    assert "${ImageOS" in key and "${ImageVersion" in key


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


def _static_matrix_legs(matrix):
    """Count the legs GitHub expands from a matrix, or None for run-time axes."""
    if not isinstance(matrix, dict):
        return None
    axes = {
        key: value for key, value in matrix.items() if key not in ("include", "exclude")
    }
    if not all(isinstance(value, list) for value in axes.values()):
        return None
    combinations = [{}]
    for key, values in axes.items():
        combinations = [
            {**base, key: value} for base in combinations for value in values
        ]
    if not axes:
        combinations = []
    for excluded in matrix.get("exclude") or []:
        combinations = [
            combination
            for combination in combinations
            if any(combination.get(key) != value for key, value in excluded.items())
        ]
    added = 0
    for entry in matrix.get("include") or []:
        # An entry extends every combination whose original values it does not
        # overwrite; when it extends none, it becomes a leg of its own.
        if not any(
            all(
                combination[key] == value for key, value in entry.items() if key in axes
            )
            for combination in combinations
        ):
            added += 1
    return len(combinations) + added


def test_static_matrix_legs_follow_github_expansion():
    assert _static_matrix_legs({"os": ["a", "b"], "group": [1, 2, 3]}) == 6
    assert (
        _static_matrix_legs(
            {"os": ["a", "b"], "group": [1, 2], "exclude": [{"os": "a", "group": 1}]}
        )
        == 3
    )
    assert (
        _static_matrix_legs(
            {
                "os": ["a"],
                "group": [1, 2],
                "include": [{"group": 1, "shard": 0}, {"group": 9}],
            }
        )
        == 3
    )
    assert _static_matrix_legs({"include": [{"os": "a"}, {"os": "b"}]}) == 2
    assert _static_matrix_legs({"os": "${{ fromJSON(needs.plan.outputs.os) }}"}) is None


@pytest.mark.parametrize(
    "filename", sorted(path.name for path in WORKFLOW_DIR.glob("*.yml"))
)
def test_every_static_matrix_fits_the_github_leg_limit(filename):
    # GitHub refuses to expand a matrix of more than 256 legs, and the job then
    # fails without running a single leg.
    workflow = yaml.safe_load((WORKFLOW_DIR / filename).read_text(encoding="utf-8"))
    for name, job in (workflow.get("jobs") or {}).items():
        legs = _static_matrix_legs((job.get("strategy") or {}).get("matrix"))
        assert legs is None or legs <= 256, (name, legs)
