import json
from pathlib import Path

from tools.warp_conformance_matrix import (
    build_matrix,
    main,
    markdown,
    required_failures,
)

WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github/workflows/demo-project-testing.yml"
)

PASSING = """<testsuites><testsuite>
<testcase classname="tests.controls" name="test_sum[uint]"/>
<testcase classname="tests.controls" name="test_sum[float]"/>
<testcase classname="tests.controls" name="test_compile"><skipped/></testcase>
</testsuite></testsuites>"""

FAILING = """<testsuites><testsuite>
<testcase classname="tests.controls" name="test_sum[uint]"><failure/></testcase>
<testcase classname="tests.controls" name="test_sum[float]"/>
<testcase classname="tests.controls" name="test_compile"><skipped/></testcase>
</testsuite></testsuites>"""


def _evidence(tmp_path, runs, results):
    (tmp_path / "runs.json").write_text(json.dumps(runs), encoding="utf-8")
    for version, text in results.items():
        (tmp_path / version).mkdir()
        (tmp_path / version / "results.xml").write_text(text, encoding="utf-8")
    return tmp_path


def _run(version, exit_code):
    return {"version": version, "dllSHA256": version * 2, "exitCode": exit_code}


def test_matrix_records_failures_as_findings(tmp_path):
    evidence = _evidence(
        tmp_path,
        [_run("1.0.13", 0), _run("1.0.21", 1)],
        {"1.0.13": PASSING, "1.0.21": FAILING},
    )
    matrix = build_matrix(evidence)
    assert matrix["complete"] is True
    outcomes = {
        library["version"]: library["outcomes"] for library in matrix["libraries"]
    }
    assert outcomes["1.0.21"]["tests.controls::test_sum[uint]"] == "failed"
    assert outcomes["1.0.13"]["tests.controls::test_sum[uint]"] == "passed"
    table = markdown(matrix)
    assert "| `test_sum[uint]` | pass | FAIL |" in table
    # A test skipped on every library is not part of the table.
    assert "test_compile" not in table
    assert main([str(evidence)]) == 0
    assert json.loads((evidence / "matrix.json").read_text())["complete"] is True


def test_timed_out_or_missing_runs_are_incomplete(tmp_path):
    evidence = _evidence(
        tmp_path,
        [_run("1.0.13", 124), _run("1.0.19", 0), _run("1.0.21", 0)],
        {"1.0.13": PASSING, "1.0.21": PASSING},
    )
    matrix = build_matrix(evidence)
    reasons = {library["version"]: library["reason"] for library in matrix["libraries"]}
    assert matrix["complete"] is False
    assert reasons["1.0.13"] == "the run exited with 124"
    assert reasons["1.0.19"].endswith("results.xml is missing")
    assert reasons["1.0.21"] is None
    assert main([str(evidence)]) == 1
    assert "WARP 1.0.13 is incomplete" in markdown(matrix)


def test_a_library_missing_a_test_is_incomplete(tmp_path):
    partial = PASSING.replace(
        '<testcase classname="tests.controls" name="test_sum[float]"/>', ""
    )
    evidence = _evidence(
        tmp_path,
        [_run("1.0.13", 0), _run("1.0.21", 0)],
        {"1.0.13": PASSING, "1.0.21": partial},
    )
    matrix = build_matrix(evidence)
    (library,) = (item for item in matrix["libraries"] if not item["complete"])
    assert library["version"] == "1.0.21"
    assert library["reason"] == "no result for tests.controls::test_sum[float]"


def test_missing_run_list_fails(tmp_path):
    assert main([str(tmp_path)]) == 1


def test_a_required_library_must_pass_every_test(tmp_path):
    evidence = _evidence(
        tmp_path,
        [_run("1.0.19", 0), _run("1.0.21", 1)],
        {"1.0.19": PASSING, "1.0.21": FAILING},
    )
    matrix = build_matrix(evidence)
    assert required_failures(matrix, ["1.0.19"]) == []
    assert main([str(evidence), "--require", "1.0.19"]) == 0
    (reason,) = required_failures(matrix, ["1.0.21"])
    assert reason == (
        "required WARP 1.0.21 failed 1 test(s), first tests.controls::test_sum[uint]"
    )
    summary = tmp_path / "summary.md"
    assert main([str(evidence), "--require", "1.0.21", "--summary", str(summary)]) == 1
    assert reason in summary.read_text(encoding="utf-8")


def test_a_required_library_must_run_and_execute_what_others_execute(tmp_path):
    skipped = PASSING.replace(
        '<testcase classname="tests.controls" name="test_sum[uint]"/>',
        '<testcase classname="tests.controls" name="test_sum[uint]"><skipped/></testcase>',
    )
    evidence = _evidence(
        tmp_path,
        [_run("1.0.13", 0), _run("1.0.19", 0)],
        {"1.0.13": PASSING, "1.0.19": skipped},
    )
    matrix = build_matrix(evidence)
    assert required_failures(matrix, ["1.0.13"]) == []
    assert required_failures(matrix, ["1.0.19", "1.0.21"]) == [
        "required WARP 1.0.19 skipped 1 test(s) that other libraries executed, "
        "first tests.controls::test_sum[uint]",
        "required WARP 1.0.21 did not run",
    ]
    assert main([str(evidence), "--require", "1.0.21"]) == 1


def test_a_required_library_that_exits_abnormally_fails(tmp_path):
    evidence = _evidence(tmp_path, [_run("1.0.19", 1)], {"1.0.19": PASSING})
    matrix = build_matrix(evidence)
    assert required_failures(matrix, ["1.0.19"]) == [
        "required WARP 1.0.19 exited with 1"
    ]


def test_the_mlx_gate_requires_the_reference_library_to_pass_the_reproducer():
    import yaml

    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["warp-conformance"]
    assert "warp-conformance" in workflow["jobs"]["result"]["needs"]
    assert "smoke" in job["if"] and "heavy" in job["if"]
    steps = {step.get("name"): step for step in job["steps"]}
    execute = steps["Execute the controls on every WARP library"]["run"]
    assert '@{ version = "1.0.19"; archiveSHA256 = "' in execute
    for path in (
        "tests/test_translator/test_directx_partial_wave_controls.py",
        "tests/test_translator/test_directx_multiwave_reduction.py",
    ):
        assert path in execute
    summary = steps["Summarize the conformance matrix"]["run"]
    assert "--require 1.0.19" in summary
    # The step records every exit code, so only the summary decides the outcome.
    assert execute.rstrip().endswith("exit 0")
