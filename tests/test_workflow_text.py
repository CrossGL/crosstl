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
