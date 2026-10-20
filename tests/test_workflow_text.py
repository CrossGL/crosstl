import re
import textwrap

import pytest

from tests import workflow_text as wt

WORKFLOW = textwrap.dedent("""\
    name: Example

    on:
      push:
        branches: [main]
        paths:
          - "crosstl/**"
          - 'tests/**'
      pull_request:
      workflow_dispatch:

    jobs:
      build:
        runs-on: ${{ matrix.os }}
        timeout-minutes: 30
        strategy:
          matrix:
            os: [ubuntu-latest, "macOS-latest"]
            python-version:
              [
                "3.11",
                "3.15",
              ]
        steps:
          - name: Check out
            uses: actions/checkout@v6
          - name: Test
            run: pytest
      result:
        needs: [build]
        runs-on: ubuntu-latest
        steps:
          - name: Report
            run: echo done
    """)


def test_event_path_filters():
    assert wt.workflow_event_path_filters(WORKFLOW, "push") == [
        "crosstl/**",
        "tests/**",
    ]
    assert wt.workflow_event_path_filters(WORKFLOW, "pull_request") == ["**"]
    assert wt.workflow_event_path_filters(WORKFLOW, "merge_group") == []


def test_jobs_and_steps():
    assert wt.workflow_job_names(WORKFLOW) == ["build", "result"]
    assert wt.workflow_job_timeout_minutes(WORKFLOW, "build") == 30
    assert wt.workflow_job_timeout_minutes(WORKFLOW, "result") is None
    step = wt.workflow_job_step_section(WORKFLOW, "build", "Check out")
    assert "actions/checkout@v6" in step and "pytest" not in step
    assert wt.workflow_job_step_section(WORKFLOW, "build", "Report") == ""
    assert wt.workflow_job_step_after(WORKFLOW, "build", "Test", "Check out")
    assert not wt.workflow_job_step_after(WORKFLOW, "build", "Check out", "Test")
    assert not wt.workflow_step_after(WORKFLOW, "Test", "Missing")


def test_matrix_values():
    assert wt.matrix_values(WORKFLOW, "os") == {"ubuntu-latest", "macOS-latest"}
    assert wt.matrix_values(WORKFLOW, "python-version") == {"3.11", "3.15"}
    with pytest.raises(AssertionError, match="not found"):
        wt.matrix_values(WORKFLOW, "shard")


def test_run_time_matrix_must_declare_its_full_list():
    dynamic = "env:\n  FULL_SHARDS: '[0, 1]'\nmatrix:\n  shard: ${{ fromJSON(needs.plan.outputs.shards) }}\n"
    assert wt.matrix_values(dynamic, "shard") == {"0", "1"}
    with pytest.raises(AssertionError, match="FULL_SHARDS"):
        wt.matrix_values(dynamic.split("\n", 2)[2], "shard")


def test_reads_every_repository_workflow():
    texts = wt.workflow_texts()
    assert "coverage.yml" in texts
    assert all(wt.workflow_job_names(text) for text in texts.values())


# Until tools/ci_coverage.py is removed, its copies of these helpers must agree with this module.
def test_matches_the_ci_coverage_helpers():
    ci_coverage = pytest.importorskip("tools.ci_coverage")
    for name, text in wt.workflow_texts().items():
        for event in ("push", "pull_request", "merge_group", "schedule"):
            assert wt.workflow_event_path_filters(
                text, event
            ) == ci_coverage.workflow_event_path_filters(text, event), (name, event)
        jobs = wt.workflow_job_names(text)
        assert jobs == ci_coverage.workflow_job_names(text), name
        for job in jobs:
            body = wt.workflow_job_text(text, job)
            assert body == ci_coverage.workflow_job_text(text, job), (name, job)
            assert wt.workflow_job_timeout_minutes(
                text, job
            ) == ci_coverage.workflow_job_timeout_minutes(text, job)
            for step in re.findall(r"^\s*-\s+name:\s*(.+?)\s*$", body, re.MULTILINE):
                assert wt.workflow_step_section(
                    body, step
                ) == ci_coverage.workflow_step_section(body, step), (name, job, step)
