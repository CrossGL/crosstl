import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from demos.integrations.mlx.portable_host import (
    upstream_module_groups,
    verify,
    verify_random,
    verify_scatter,
    verify_selection,
    verify_slice_updates,
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
                "skips": int(not native and name in ops.GPU_ONLY_TESTS),
                "dispatchCount": (
                    3 if native and name not in ops.NO_DISPATCH_TESTS else 0
                ),
            }
            for name in ops.UPSTREAM_TESTS
        ]
    }


@pytest.mark.parametrize("native", [False, True])
def test_upstream_gpu_only_tests_skip_only_on_the_cpu(native):
    assert ops.GPU_ONLY_TESTS <= set(ops.UPSTREAM_TESTS)
    assert not ops.GPU_ONLY_TESTS & ops.NO_DISPATCH_TESTS
    record = records(native=native)
    ops.validate_upstream(record, native=native)
    for item in record["tests"]:
        if item["test"] in ops.GPU_ONLY_TESTS:
            # A CPU run must skip it; a native run must not.
            item["skips"] = int(native)
            with pytest.raises(ValueError, match="Unchanged upstream test failed"):
                ops.validate_upstream(record, native=native)
            item["skips"] = int(not native)


def test_upstream_op_tests_without_dispatches_are_exact():
    assert ops.NO_DISPATCH_TESTS <= set(ops.UPSTREAM_TESTS)
    for native in (True, False):
        ops.validate_upstream(records(native=native), native=native)
    for name in sorted(ops.NO_DISPATCH_TESTS):
        record = records(native=True)
        for item in record["tests"]:
            if item["test"] == name:
                # A rejected argument check or an empty result must not reach
                # a kernel.
                item["dispatchCount"] = 1
        with pytest.raises(ValueError, match=name):
            ops.validate_upstream(record, native=True)


def test_upstream_op_inventory_is_unique_and_not_run_elsewhere():
    assert len(ops.UPSTREAM_TESTS) == len(set(ops.UPSTREAM_TESTS)) == 507
    assert all(
        name.startswith(
            (
                "test_ops.TestOps.test_",
                "test_reduce.TestReduce.test_",
                "test_bf16.TestBF16.test_",
                "test_fast.TestFast.test_",
                "test_blas.TestBlas.test_",
                "test_losses.TestLosses.test_",
                "test_einsum.TestEinsum.test_",
                "test_init.TestInit.test_",
                "test_linalg.TestLinalg.test_",
                "test_tree.TestTreeUtils.test_",
                "test_fft.TestFFT.test_",
                "test_conv.TestConv.test_",
                "test_array.TestArray.test_",
                "test_array.TestEquality.test_",
                "test_array.TestInequality.test_",
                "test_autograd.TestAutograd.test_",
                "test_compile.TestCompile.test_",
                "test_nn.TestBase.test_",
                "test_nn.TestLayers.test_",
                "test_vmap.TestVmap.test_",
                "test_eval.TestEval.test_",
                "test_export_import.TestExportImport.test_",
                "test_load.TestLoad.test_",
                "test_optimizers.TestOptimizers.test_",
                "test_optimizers.TestSchedulers.test_",
                "test_random.TestRandom.test_",
                "test_device.TestDevice.test_",
                "test_device.TestStream.test_",
                "test_memory.TestMemory.test_",
                "test_constants.TestConstants.test_",
                "test_upsample.TestUpsample.test_",
            )
        )
        for name in ops.UPSTREAM_TESTS
    )
    tail = (
        ops.MODULE_TESTS
        + ops.CONV_TESTS
        + ops.ARRAY_TESTS
        + ops.AUTOGRAD_TESTS
        + ops.COMPILE_TESTS
        + upstream_module_groups.NN_TESTS
        + upstream_module_groups.TRANSFORM_TESTS
        + upstream_module_groups.SERIALIZATION_TESTS
        + upstream_module_groups.TRAINING_TESTS
        + upstream_module_groups.ROPE_TESTS
        + upstream_module_groups.FFT_TESTS
    )
    assert ops.UPSTREAM_TESTS[-len(tail) :] == tail
    elsewhere = (
        set(verify.UPSTREAM_TESTS)
        | set(verify_random.UPSTREAM_TESTS)
        | {verify_selection.UPSTREAM_TEST}
        | set(verify_scatter.UPSTREAM_TESTS)
        | set(verify_slice_updates.UPSTREAM_TESTS)
    )
    assert not elsewhere & set(ops.UPSTREAM_TESTS)


def test_upstream_op_exclusions_name_their_issue_and_keep_the_order():
    # Only OpenGL excludes tests: its software subgroups cannot run an arg
    # reduction wider than one simdgroup yet, and its reverse scans cannot
    # prove their wrapped pointer offsets yet.
    assert set(ops.TARGET_EXCLUSIONS) == {"opengl"}
    excluded = ops.TARGET_EXCLUSIONS["opengl"]
    assert set(excluded) <= set(ops.UPSTREAM_TESTS)
    assert set(excluded.values()) == {"CrossGL/crosstl#2207", "CrossGL/crosstl#2905"}
    assert [name for name, issue in excluded.items() if issue.endswith("2905")] == [
        "test_ops.TestOps.test_scans",
        "test_ops.TestOps.test_scans_complex_exclusive",
    ]
    assert ops.upstream_tests("metal") == ops.upstream_tests("directx")
    assert ops.upstream_tests("metal") == ops.UPSTREAM_TESTS
    opengl = ops.upstream_tests("opengl")
    assert len(opengl) == 502 and not set(opengl) & set(excluded)
    assert list(opengl) == [name for name in ops.UPSTREAM_TESTS if name in opengl]
    record = {"tests": records(native=True)["tests"]}
    with pytest.raises(ValueError, match="inventory"):
        ops.validate_upstream(record, native=True, inventory=opengl)
    record["tests"] = [item for item in record["tests"] if item["test"] in opengl]
    ops.validate_upstream(record, native=True, inventory=opengl)


def test_upstream_op_groups_partition_the_inventory():
    assert list(ops.GROUPS) == [
        "operations",
        "normalization",
        "reductions",
        "blas",
        "reduction-dtypes",
        "reduction-permutations",
        "blas-wide",
        "blas-gather",
        "blas-gather-blocks",
        "blas-masked",
        "modules",
        "conv",
        "array",
        "autograd",
        "compile",
        "nn",
        "transforms",
        "serialization",
        "training",
        "rope",
        "fft",
    ]
    assert all(
        len(ops.GROUP_TESTS[group]) == 1
        for group in (
            "reduction-dtypes",
            "reduction-permutations",
            "blas-gather-blocks",
            "blas-masked",
        )
    )
    for group, tests in ops.GROUP_TESTS.items():
        assert ops.GROUPS[group] == tests
    combined = [name for group in ops.GROUPS.values() for name in group]
    assert sorted(combined) == sorted(ops.UPSTREAM_TESTS)
    assert len(combined) == len(set(combined))
    assert all(name.startswith("test_fast.") for name in ops.GROUPS["normalization"])
    assert len(ops.GROUPS["normalization"]) == 9
    assert len(ops.GROUPS["reductions"]) == 9
    assert not set(ops.GROUPS["reductions"]) & {
        name for tests in ops.GROUP_TESTS.values() for name in tests
    }
    assert all(name.startswith("test_reduce.") for name in ops.GROUPS["reductions"])
    blas = {
        "blas": 18,
        "blas-wide": 2,
        "blas-gather": 6,
        "blas-gather-blocks": 1,
        "blas-masked": 1,
    }
    for group, count in blas.items():
        assert len(ops.GROUPS[group]) == count
        assert all(name.startswith("test_blas.") for name in ops.GROUPS[group])
    assert not any(
        name.startswith(
            (
                "test_fast.",
                "test_reduce.",
                "test_blas.",
                *{name.split(".")[0] + "." for name in ops.MODULE_TESTS},
            )
        )
        for name in ops.GROUPS["operations"]
    )
    # The module tests run together, and only they run in their group.
    assert ops.GROUPS["modules"] == ops.MODULE_TESTS
    assert len(ops.MODULE_TESTS) == 50
    assert [name for name in ops.MODULE_TESTS if "cross_entropy" in name] == [
        "test_losses.TestLosses.test_binary_cross_entropy",
        "test_losses.TestLosses.test_cross_entropy",
        "test_fast.TestFast.test_cross_entropy",
        "test_fast.TestFast.test_cross_entropy_grad",
    ]
    assert ops.upstream_tests("opengl", "modules") == ops.MODULE_TESTS
    assert ops.GROUPS["conv"] == ops.CONV_TESTS and len(ops.CONV_TESTS) == 7
    assert ops.upstream_tests("directx", "conv") == ops.CONV_TESTS
    # The array, autograd and compile modules each run alone in their group.
    for group, tests, module, count in (
        ("array", ops.ARRAY_TESTS, "test_array.", 41),
        ("autograd", ops.AUTOGRAD_TESTS, "test_autograd.", 45),
        ("compile", ops.COMPILE_TESTS, "test_compile.", 51),
    ):
        assert ops.GROUPS[group] == tests and len(tests) == count
        assert all(name.startswith(module) for name in tests)
        assert ops.upstream_tests("opengl", group) == tests
    # The GPU methods of whole modules run by module family, each family in
    # a group of its own.
    for group, modules, count in (
        ("nn", ("test_nn.",), 39),
        ("transforms", ("test_vmap.", "test_eval."), 35),
        ("serialization", ("test_export_import.", "test_load."), 30),
        (
            "training",
            (
                "test_optimizers.",
                "test_random.",
                "test_bf16.",
                "test_device.",
                "test_memory.",
                "test_constants.",
            ),
            35,
        ),
        ("rope", ("test_fast.", "test_upsample."), 9),
        ("fft", ("test_fft.",), 6),
    ):
        tests = upstream_module_groups.GROUPS[group]
        assert ops.GROUPS[group] == tests and len(tests) == count
        assert {name.split(".")[0] + "." for name in tests} == set(modules)
        assert ops.upstream_tests("opengl", group) == tests
    # The rotary embedding tests are test_fast methods, but not normalization.
    assert not set(ops.GROUPS["rope"]) & set(ops.GROUPS["normalization"])
    # Every slice update reduction's gradient and both shapeless tests that
    # count reduced elements with NumberOfElements run here.
    assert {
        "test_autograd.TestAutograd.test_slice_update_" + name + "_vjp"
        for name in ("add", "max", "min", "multiply")
    } <= set(ops.AUTOGRAD_TESTS)
    assert {
        "test_compile.TestCompile.test_shapeless_mean",
        "test_compile.TestCompile.test_shapeless_compile_reduce_after_gather",
    } <= set(ops.COMPILE_TESTS)
    # Both tests that slice at array start indices reach the translated
    # dynamic offset kernel.
    assert "test_ops.TestOps.test_dynamic_slicing" in ops.GROUPS["operations"]
    assert (
        "test_autograd.TestAutograd.test_jvp_with_partly_traced_inputs"
        in ops.AUTOGRAD_TESTS
    )
    # Boolean-mask assignment, its vmap and a jvp through it reach the
    # translated masked_assign kernel.
    assert {
        "test_array.TestArray.test_setitem_with_boolean_mask",
        "test_autograd.TestAutograd.test_jvp_with_constant_inputs",
        "test_vmap.TestVmap.test_vmap_masked_scatter",
    } <= set(ops.UPSTREAM_TESTS)
    for target in ("metal", "directx", "opengl"):
        grouped = [
            name for group in ops.GROUPS for name in ops.upstream_tests(target, group)
        ]
        assert sorted(grouped) == sorted(ops.upstream_tests(target))
        for group in ops.GROUPS:
            selected = ops.upstream_tests(target, group)
            assert list(selected) == [
                name for name in ops.upstream_tests(target) if name in selected
            ]
    with pytest.raises(ValueError, match="Unknown upstream op group"):
        ops.upstream_tests("metal", "fast")
    # The operation tests own the rejection check and the indexing requirement.
    assert ops.negative_checks(None) == ops.negative_checks("operations")
    assert ops.negative_checks("operations") == ops.NEGATIVE_CHECKS
    for group in set(ops.GROUPS) - {"operations"}:
        assert ops.negative_checks(group) == {}
        assert not ops.requires_indexing(group)
    assert ops.requires_indexing(None) and ops.requires_indexing("operations")


def test_upstream_op_workers_receive_their_group(tmp_path, monkeypatch):
    commands = []

    def run(command, **_):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ops.subprocess, "run", run)
    args = ops.parse(
        [
            *(
                item
                for name in (
                    "mlx-root",
                    "packages",
                    "random",
                    "bitwise",
                    "selection",
                    "absolute",
                    "integer64",
                    "slice-updates",
                    "output-dir",
                )
                for item in ("--" + name, str(tmp_path / name))
            ),
            "--reductions",
            str(tmp_path / "reductions"),
            "--group",
            "normalization",
        ]
    )
    assert args.group == "normalization"
    ops.run_worker(args, tmp_path, "native")
    (command,) = commands
    assert command[command.index("--group") + 1] == "normalization"
    assert command[command.index("--worker") + 1] == "native"
    args.group = None
    ops.run_worker(args, tmp_path, "cpu")
    assert "--group" not in commands[-1]


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
    monkeypatch.setattr(
        ops,
        "verify_native_identity",
        lambda trace, target: seen.setdefault("native", []).extend(
            e["entry"] for e in trace
        ),
    )

    def lazy(event, packages, modules):
        assert packages.target == "metal"
        assert modules == tmp_path / "evidence/native/native-modules"
        seen.setdefault("lazy", []).append(event["entry"])
        # Scans cover their native execution; small reductions do not.
        return event["entry"].startswith("contig_scan_")

    monkeypatch.setattr(ops.lazy_package_evidence, "audit_event", lazy)

    def vectors(event, cache):
        assert cache.target == "metal"
        seen.setdefault("vectors", []).append(event["entry"])

    monkeypatch.setattr(ops.elementwise_evidence, "audit_event", vectors)
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
        output_dir=tmp_path / "evidence",
        random=tmp_path / "random",
        reductions=[tmp_path / "red-init", tmp_path / "red-checked"],
        **{name: tmp_path / name for name in ("packages", *ops.INDEXED_FAMILIES)},
    )
    indexes = {
        "packages": {"descriptors": {"vv_Equalint32": {}}},
        "bitwise": {"descriptors": {"vv_BitwiseAndbool_": {}}},
        "selection": {"descriptors": {"v_Selectfloat32": {}}},
        "absolute": {"descriptors": {"v_Absfloat32float32": {}}},
        "integer64": {"descriptors": {"vv_Addint64": {}}},
        "slice_updates": {"descriptors": {"slice_update_maxfloat32": {}}},
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
        event("vv_Addint64"),
        event("slice_update_maxfloat32"),
        event("init_reduce_andbool_", 32),
        event("all_reduce_andbool_", 32),
        event("gatherint32int32_2_1_int"),
        event("scatterfloat32int32_none_2_updc_true_nwork1_int"),
        event("g3_DivModuint16"),
        event("contig_scan_inclusive_sum_int32_int32"),
        event("row_reduce_small_1_reduce_sumint32", 32),
        event("col_reduce_small_1_reduce_sumint32", 32),
        event("vv_Addfloat16"),
        event("v_Absint8int8"),
    ]
    count = ops.audit_trace(
        None, audit.args, trace, "metal", audit.indexes, audit.reductions
    )
    assert count == 2
    assert audit.seen["packages"] == ["vv_Equalint32"]
    assert audit.seen["integer64"] == ["vv_Addint64"]
    assert audit.seen["slice_updates"] == ["slice_update_maxfloat32"]
    assert audit.seen["red-checked"] == ["all_reduce_andbool_"]
    assert audit.seen["gather_evidence"] == ["gatherint32int32_2_1_int"]
    assert audit.seen["scatter_evidence"] == [
        "scatterfloat32int32_none_2_updc_true_nwork1_int"
    ]
    assert audit.seen["divmod_evidence"] == ["g3_DivModuint16"]
    assert audit.seen["lazy"] == [
        "contig_scan_inclusive_sum_int32_int32",
        "row_reduce_small_1_reduce_sumint32",
        "col_reduce_small_1_reduce_sumint32",
    ]
    # Lazy vectors are audited once, including their native modules.
    assert audit.seen["vectors"] == ["vv_Addfloat16", "v_Absint8int8"]
    assert not {"vv_Addfloat16", "v_Absint8int8"} & set(audit.seen["native"])
    # Scans' own audit covers native execution; small reductions keep it.
    assert "contig_scan_inclusive_sum_int32_int32" not in audit.seen["native"]
    assert {
        "row_reduce_small_1_reduce_sumint32",
        "col_reduce_small_1_reduce_sumint32",
        "vv_Addint64",
    } <= set(audit.seen["native"])


GATHERED_MATRIX_ENTRIES = (
    # GatherMM's group offsets share the general gather prefix.
    "gather_mm_offsets",
    "gemv_gather_float32_bm4_bn1_sm1_sn32_tm4_tn4",
    "gemv_t_gather_float32_bm1_bn8_sm1_sn32_tm4_tn4",
    "steel_gather_mm_nn_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n"
    "_align_M_n_align_N_n_align_K_t",
    "steel_segmented_mm_tn_float32_float32_bm64_bn64_bk16_wm2_wn2"
    "_segments_contiguous_n_align_M_n_align_N_n",
)


def test_upstream_op_gathered_matrix_kernels_reach_only_the_lazy_audit(audit):
    trace = [event(entry) for entry in GATHERED_MATRIX_ENTRIES]
    for entry in GATHERED_MATRIX_ENTRIES:
        assert ops.indexing_auditor(entry) is None
    assert ops.indexing_auditor("gatherint32int32_2_1_int") is not None
    count = ops.audit_trace(
        None,
        audit.args,
        trace,
        "metal",
        audit.indexes,
        audit.reductions,
        require_indexing=False,
    )
    assert count == 0
    assert audit.seen["lazy"] == list(GATHERED_MATRIX_ENTRIES)
    assert "gather_evidence" not in audit.seen


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
    # The normalization group reaches no indexing kernel; its dispatches must
    # still each belong to one package.
    assert (
        ops.audit_trace(
            None,
            audit.args,
            [event("vv_Equalint32")],
            "metal",
            audit.indexes,
            audit.reductions,
            require_indexing=ops.requires_indexing("normalization"),
        )
        == 0
    )
    with pytest.raises(ValueError, match="unknown or ambiguous package"):
        ops.audit_trace(
            None,
            audit.args,
            [event("vv_Equalint32"), event("vv_Unknownint32")],
            "metal",
            audit.indexes,
            audit.reductions,
            require_indexing=False,
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
        event("row_reduce_looped_1_32_32_reduce_sumint32"),
        event("all_reduce_andbool_", 256),
        # The 64-bit vectors have no work-per-thread instantiation.
        event("vvn_Addint64"),
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
    job = workflow["jobs"]["upstream-ops-host"]
    assert job["if"] == "needs.ci-plan.outputs.heavy == 'true'"
    assert "upstream-ops-host" in workflow["jobs"]["result"]["needs"]
    matrix = job["strategy"]["matrix"]
    assert {row["target"] for row in matrix["include"]} == {
        "metal",
        "directx",
        "opengl",
    }
    assert set(matrix["target"]) == {"metal", "directx", "opengl"}
    # Every group runs in its own job on every platform.
    assert matrix["group"] == list(ops.GROUPS)
    assert "${{ matrix.group }}" in job["name"]
    names = [step.get("name") for step in job["steps"]]
    step = job["steps"][names.index("Execute unchanged upstream op tests")]
    assert not step.get("continue-on-error") and not step.get("if")
    assert "set -euo pipefail" in step["run"]
    assert "portable_host.verify_upstream_ops" in step["run"]
    assert '--group "${{ matrix.group }}"' in step["run"]
    retained = job["steps"][names.index("Retain upstream op execution evidence")]
    assert "${{ matrix.group }}" in retained["with"]["name"]
    deadline = int(re.search(r"--timeout-seconds (\d+)", step["run"]).group(1))
    # The workers' own bounds leave the audit time inside the step's deadline.
    assert sum(ops.TIMEOUTS.values()) + 60 <= deadline
    deadlines = [
        int(seconds)
        for item in job["steps"]
        for seconds in re.findall(r"--timeout-seconds (\d+)", item.get("run", ""))
    ]
    assert sum(deadlines) + 1800 < job["timeout-minutes"] * 60
    root = ".mlx-portable-upstream-ops/"
    flags = [("--packages", "base/packages"), ("--random", "random-packages")]
    flags += [
        ("--" + name.replace("_", "-"), name.replace("_", "-") + "-packages")
        for name in ops.INDEXED_FAMILIES
    ]
    flags += [
        ("--reductions", "init-packages"),
        ("--reductions", "all-reductions"),
        ("--reductions", "row-reductions"),
        ("--reductions", "column-reductions"),
    ]
    for flag, directory in flags:
        assert f"{flag} {root}{directory}" in step["run"]
        # Every package directory is translated earlier in the same job.
        producer = next(
            i
            for i, item in enumerate(job["steps"])
            if f"--output-dir {root}{directory}" in item.get("run", "")
            or item.get("with", {}).get("path") == root + "base"
        )
        assert producer < names.index(step["name"])
    configure = job["steps"][names.index("Configure the upstream op reductions")]["run"]
    assert "--family all --on-demand" in configure
    assert "--family row --on-demand" in configure
    # Softmax over a leading axis composes column maxima and sums.
    assert "--family column" in configure
    retain = names.index("Retain upstream op execution evidence")
    assert names.index(step["name"]) < retain


@pytest.mark.parametrize("fault", ["none", "w256", "target", "selection-target"])
def test_upstream_op_indexes_require_checked_reductions_and_one_target(
    tmp_path, monkeypatch, fault
):
    args = SimpleNamespace(
        random=tmp_path / "random",
        reductions=[tmp_path / "init", tmp_path / "checked"],
        **{name: tmp_path / name for name in ("packages", *ops.INDEXED_FAMILIES)},
    )
    for name in ("packages", *ops.INDEXED_FAMILIES):
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


def test_upstream_op_traces_keep_only_the_recomputed_values():
    retained = ops.RETAINED_VALUES
    for entry in (
        ops.ENTRIES[0],
        "gatherint32int32_2_1_int",
        "scatter_axisint32int64_none_intcc",
        "vv_DivModint32",
        "all_reduce_andbool_",
        "init_reduce_sumint32",
        "argmin_bfloat16",
        "argmax_uint64",
        "row_reduce_small_1_reduce_sumfloat16",
        "row_reduce_small_2_reduce_prodint64",
        "col_reduce_small_1_reduce_maxfloat16",
        "col_reduce_small_1_reduce_sumcomplex64",
        "rms_loopedfloat32",
        "vjp_layer_normbfloat16_now",
    ):
        assert entry.startswith(retained), entry
    for entry in (
        "ggn2_dynamic_copyint64int64",
        "vv_Addint64",
        "contig_scan_inclusive_sum_int32_int32",
        "row_reduce_small_1_reduce_sumint32",
        "row_reduce_looped_1_reduce_sumfloat16",
        "col_reduce_looped_1_32_32_reduce_sumfloat16",
    ):
        assert not entry.startswith(retained), entry
