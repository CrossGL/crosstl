import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from demos.integrations.mlx.portable_host import (
    verify,
    verify_random,
    verify_selection,
    verify_upstream_ops,
)

ops = verify_upstream_ops


def records(*, native):
    return {
        "tests": [
            {
                "test": name,
                "testsRun": 1,
                "failures": 0,
                "errors": 0,
                "skips": 0,
                "dispatchCount": 3 if native else 0,
            }
            for name in ops.UPSTREAM_TESTS
        ]
    }


def test_upstream_op_inventory_is_unique_and_not_run_elsewhere():
    assert len(ops.UPSTREAM_TESTS) == len(set(ops.UPSTREAM_TESTS)) == 22
    assert all(name.startswith("test_ops.TestOps.test_") for name in ops.UPSTREAM_TESTS)
    elsewhere = (
        set(verify.UPSTREAM_TESTS)
        | set(verify_random.UPSTREAM_TESTS)
        | {verify_selection.UPSTREAM_TEST}
    )
    assert not elsewhere & set(ops.UPSTREAM_TESTS)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "fault", ["none", "missing", "reordered", "failure", "skip", "dispatch", "run"]
)
def test_upstream_op_results_cannot_skip_or_replace_tests(native, fault):
    record = records(native=native)
    tests = record["tests"]
    if fault == "missing":
        tests.pop()
    elif fault == "reordered":
        tests[0], tests[1] = tests[1], tests[0]
    elif fault == "failure":
        tests[3]["failures"] = 1
    elif fault == "skip":
        tests[-1]["skips"] = 1
    elif fault == "dispatch":
        # Native tests must reach a translated kernel; CPU tests must not.
        tests[2]["dispatchCount"] = 0 if native else 1
    elif fault == "run":
        tests[4]["testsRun"] = 0
    if fault == "none":
        ops.validate_upstream(record, native=native)
    else:
        with pytest.raises(ValueError):
            ops.validate_upstream(record, native=native)


def event(entry, size=1, target="metal"):
    return {"entry": entry, "workgroupSize": [size, 1, 1], "target": target}


@pytest.fixture
def audit(monkeypatch, tmp_path):
    seen = {}

    def artifacts(events, directory, index, *, variants, mlx_root=None):
        seen.setdefault(directory.name, []).extend(e["entry"] for e in events)
        assert variants == directory.name.startswith("red")
        assert mlx_root is (tmp_path if variants else None)

    monkeypatch.setattr(ops, "verify_artifacts", artifacts)
    monkeypatch.setattr(ops, "verify_native_identity", lambda trace, target: None)
    monkeypatch.setattr(ops.random_evidence, "audit_event", lambda np, e: None)
    monkeypatch.setattr(
        ops.random_evidence, "verify_packages", lambda e, root, d, target: None
    )
    monkeypatch.setattr(ops, "variant_keys", lambda index: set(index["descriptors"]))
    monkeypatch.setattr(
        ops.divmod_evidence,
        "audit_event",
        lambda e: seen.setdefault("divmod_evidence", []).append(e["entry"]),
    )
    for name in ("gather_evidence", "scatter_axis_evidence", "scatter_evidence"):
        monkeypatch.setattr(
            getattr(ops, name),
            "audit_event",
            lambda np, e, name=name: seen.setdefault(name, []).append(e["entry"]),
        )
    args = SimpleNamespace(
        mlx_root=tmp_path,
        random=tmp_path / "random",
        reductions=[tmp_path / "red-init", tmp_path / "red-checked"],
        **{
            name: tmp_path / name
            for name in ("packages", "bitwise", "selection", "absolute")
        },
    )
    indexes = {
        "packages": {"descriptors": {"vv_Equalint32": {}}},
        "bitwise": {"descriptors": {"vv_BitwiseAndbool_": {}}},
        "selection": {"descriptors": {"v_Selectfloat32": {}}},
        "absolute": {"descriptors": {"v_Absfloat32float32": {}}},
    }
    reductions = [
        {"descriptors": {"w32/init_reduce_andbool_": {}}},
        {"descriptors": {"w32/all_reduce_andbool_": {}}},
    ]
    return SimpleNamespace(args=args, indexes=indexes, reductions=reductions, seen=seen)


def test_upstream_op_trace_ties_every_dispatch_to_one_package(audit):
    trace = [
        event("vv_Equalint32"),
        event(ops.ENTRIES[0]),
        event("vv_BitwiseAndbool_"),
        event("v_Selectfloat32"),
        event("v_Absfloat32float32"),
        event("init_reduce_andbool_", 32),
        event("all_reduce_andbool_", 32),
        event("gatherint32int32_2_1_int"),
        event("scatterfloat32int32_none_2_updc_true_nwork1_int"),
        event("g3_DivModuint16"),
    ]
    count = ops.audit_trace(
        None, audit.args, trace, "metal", audit.indexes, audit.reductions
    )
    assert count == 2
    assert audit.seen["packages"] == ["vv_Equalint32"]
    assert audit.seen["red-checked"] == ["all_reduce_andbool_"]
    assert audit.seen["gather_evidence"] == ["gatherint32int32_2_1_int"]
    assert audit.seen["scatter_evidence"] == [
        "scatterfloat32int32_none_2_updc_true_nwork1_int"
    ]
    assert audit.seen["divmod_evidence"] == ["g3_DivModuint16"]


def test_upstream_op_divmod_dispatch_must_use_the_package_target(audit):
    with pytest.raises(ValueError, match="DivMod dispatch targets another backend"):
        ops.audit_trace(
            None,
            audit.args,
            [event("gatherint32int32_2_1_int"), event("vv_DivModint32", 1, "opengl")],
            "metal",
            audit.indexes,
            audit.reductions,
        )


@pytest.mark.parametrize(
    ("entry", "auditor"),
    [
        ("gather_axisfloat32int32_true_true_true_int", "gather_evidence"),
        ("scatter_axisfloat32int32_none_true_true", "scatter_axis_evidence"),
        ("scatteruint32int64_sum_1_updc_false_nwork4_int", "scatter_evidence"),
    ],
)
def test_upstream_op_indexing_entries_reach_their_evidence_check(audit, entry, auditor):
    ops.audit_trace(
        None,
        audit.args,
        [event("vv_Equalint32"), event(entry)],
        "metal",
        audit.indexes,
        audit.reductions,
    )
    assert audit.seen[auditor] == [entry]


def test_upstream_op_trace_requires_an_indexing_dispatch(audit):
    with pytest.raises(ValueError, match="did not reach gather or scatter"):
        ops.audit_trace(
            None,
            audit.args,
            [event("vv_Equalint32")],
            "metal",
            audit.indexes,
            audit.reductions,
        )


def test_upstream_op_indexing_dispatch_must_use_the_package_target(audit):
    with pytest.raises(ValueError, match="targets another backend"):
        ops.audit_trace(
            None,
            audit.args,
            [event("vv_Equalint32"), event("gatherint32int32_2_1_int", 1, "opengl")],
            "metal",
            audit.indexes,
            audit.reductions,
        )


def test_upstream_op_indexing_audit_failures_fail_the_trace(audit, monkeypatch):
    def reject(np, event):
        raise ValueError("Gather readback and MLX result disagree")

    monkeypatch.setattr(ops.gather_evidence, "audit_event", reject)
    with pytest.raises(ValueError, match="readback"):
        ops.audit_trace(
            None,
            audit.args,
            [event("gatherint32int32_2_1_int")],
            "metal",
            audit.indexes,
            audit.reductions,
        )


@pytest.mark.parametrize(
    "unknown",
    [
        event("contig_scan_inclusive_sum_int32_int32"),
        event("all_reduce_andbool_", 256),
    ],
)
def test_upstream_op_trace_rejects_unindexed_families(audit, unknown):
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        ops.audit_trace(
            None,
            audit.args,
            [event("vv_Equalint32"), event("gatherint32int32_2_1_int"), unknown],
            "metal",
            audit.indexes,
            audit.reductions,
        )


def test_upstream_op_trace_rejects_a_package_in_two_families(audit):
    audit.indexes["bitwise"]["descriptors"]["vv_Equalint32"] = {}
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        ops.audit_trace(
            None,
            audit.args,
            [event("vv_Equalint32"), event("gatherint32int32_2_1_int")],
            "metal",
            audit.indexes,
            audit.reductions,
        )


def test_upstream_op_ci_runs_on_all_three_platforms_with_the_job_families():
    workflow = yaml.safe_load(
        Path(".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["small-row-reductions"]
    assert {row["target"] for row in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "directx",
        "opengl",
    }
    names = [step.get("name") for step in job["steps"]]
    step = job["steps"][
        names.index("Execute unchanged upstream op tests on the small-row families")
    ]
    assert not step.get("continue-on-error") and not step.get("if")
    assert "set -euo pipefail" in step["run"]
    assert "portable_host.verify_upstream_ops" in step["run"]
    deadline = int(re.search(r"--timeout-seconds (\d+)", step["run"]).group(1))
    # The workers' own bounds leave the audit time inside the step's deadline.
    assert sum(ops.TIMEOUTS.values()) + 60 <= deadline
    root = ".mlx-portable-small-rows/"
    for flag, directory in (
        ("--packages", "base/packages"),
        ("--random", "random-packages"),
        ("--bitwise", "bitwise-packages"),
        ("--selection", "selection-packages"),
        ("--absolute", "absolute-packages"),
        ("--reductions", "init-packages"),
        ("--reductions", "random-reductions"),
    ):
        assert f"{flag} {root}{directory}" in step["run"]
        # Every package directory is translated earlier in the same job.
        producer = next(
            i
            for i, item in enumerate(job["steps"])
            if f"--output-dir {root}{directory}" in item.get("run", "")
            or item.get("with", {}).get("path") == root + "base"
        )
        assert producer < names.index(step["name"])
    retain = names.index("Retain small-row execution evidence")
    assert names.index(step["name"]) < retain


@pytest.mark.parametrize("fault", ["none", "w256", "target", "selection-target"])
def test_upstream_op_indexes_require_checked_reductions_and_one_target(
    tmp_path, monkeypatch, fault
):
    args = SimpleNamespace(
        random=tmp_path / "random",
        reductions=[tmp_path / "init", tmp_path / "checked"],
        **{
            name: tmp_path / name
            for name in ("packages", "bitwise", "selection", "absolute")
        },
    )
    for name in ("packages", "bitwise", "selection", "absolute"):
        (tmp_path / name).mkdir()
        target = (
            "opengl" if fault == "selection-target" and name == "selection" else "metal"
        )
        (tmp_path / name / "index.json").write_text(
            json.dumps({"target": target, "descriptors": {}})
        )
    keys = {
        "init": {"w32/init_reduce_andbool_"},
        "checked": set(ops.REQUIRED_REDUCTIONS),
    }
    if fault == "w256":
        keys["checked"].discard("w256/all_reduce_andbool_")
    loaded = []

    def reductions(path, target):
        loaded.append(target)
        if fault == "target":
            raise ValueError("reduction packages target another backend")
        return {"keys": keys[path.name]}

    monkeypatch.setattr(ops, "load_reductions", reductions)
    monkeypatch.setattr(ops, "variant_keys", lambda index: index["keys"])
    monkeypatch.setattr(ops, "load_random", lambda path, target: {ops.ENTRIES[0]: {}})
    if fault == "none":
        target, indexes, _ = ops.load_indexes(args)
        assert target == "metal" and set(indexes) == set(ops.FAMILIES)
        assert loaded == ["metal", "metal"]
    else:
        with pytest.raises(ValueError):
            ops.load_indexes(args)
