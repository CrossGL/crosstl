"""The shared upstream matrix-product runner and its CI job, for every runner."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    upstream_runner,
    verify_upstream_fp_qmv,
    verify_upstream_fp_qmv_large_output,
    verify_upstream_fp_qvm,
    verify_upstream_qmm,
    verify_upstream_qmm_shapes,
    verify_upstream_qmm_vjp,
    verify_upstream_qmv,
    verify_upstream_qmv_small_non_multiples,
)
from demos.integrations.mlx.portable_host import (
    verify_upstream_quantization as quantization,
)
from demos.integrations.mlx.portable_host import (
    verify_upstream_qvm,
    verify_upstream_qvm_splitk,
)

WORKFLOW = (
    Path(__file__).resolve().parents[5] / ".github/workflows/demo-project-testing.yml"
)
RUNNERS = {
    "qvm": verify_upstream_qvm,
    "qmv": verify_upstream_qmv,
    "qvm_splitk": verify_upstream_qvm_splitk,
    "qmm_shapes": verify_upstream_qmm_shapes,
    "qmm": verify_upstream_qmm,
    "fp_qmv": verify_upstream_fp_qmv,
    "fp_qmv_large_output": verify_upstream_fp_qmv_large_output,
    "fp_qvm": verify_upstream_fp_qvm,
    "qmv_small_non_multiples": verify_upstream_qmv_small_non_multiples,
    "qmm_vjp": verify_upstream_qmm_vjp,
}
RUN_STEP = "Run the unchanged upstream test through translated kernels"


def record(spec, mode, dispatches):
    return {
        "mode": mode,
        "tests": list(spec.UPSTREAM_TESTS),
        "testsRun": len(spec.UPSTREAM_TESTS),
        "success": True,
        "skipped": [],
        "errors": [],
        "failures": [],
        "nativeDispatches": dispatches if mode == "native" else 0,
    }


def trace(spec, target):
    items = [
        {"entry": entry, "target": target}
        for entry, count in sorted(spec.expected_dispatches().items())
        for _ in range(count)
    ]
    return items + [
        {"entry": spec.REDUCTIONS[0], "target": target, "threads": 64},
        {"entry": "rbitsc", "target": target},
    ]


def test_row_strides():
    assert upstream_runner.row_strides((3, 1, 64)) == [64, 64, 1]
    assert upstream_runner.row_strides((7,)) == [1]


@pytest.mark.parametrize(
    "fault", [None, "count", "target", "unexpected", "missing", "maximum"]
)
def test_check_trace_requires_exact_routed_dispatches(fault):
    expected = {"affine_a": 2, "gemv_b": 1}
    items = [
        {"entry": "affine_a", "target": "metal"},
        {"entry": "affine_a", "target": "metal"},
        {"entry": "gemv_b", "target": "metal"},
        {"entry": "copy_c", "target": "metal"},
        {"entry": "max", "target": "metal"},
    ]
    count = len(items)
    if fault == "count":
        count += 1
    elif fault == "target":
        items[3]["target"] = "opengl"
    elif fault == "unexpected":
        items[3]["entry"] = "gemv_c"
    elif fault == "missing":
        items.pop(0)
        count -= 1
    elif fault == "maximum":
        items.pop()
        count -= 1
    check = dict(label="t", expected=expected, prefixes=("affine_", "gemv_"))
    if fault:
        with pytest.raises(RuntimeError, match="Upstream t"):
            upstream_runner.check_trace(
                items, "metal", count, reductions=("max",), **check
            )
    else:
        upstream_runner.check_trace(items, "metal", count, reductions=("max",), **check)


def test_translate_ahead_places_column_sums_beside_the_maxima(tmp_path, monkeypatch):
    from demos.integrations.mlx.portable_host import packages as operator_packages
    from demos.integrations.mlx.portable_host import reduction_packages

    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "opengl"}))
    builds, reductions, operators = [], [], []

    class Builder:
        @staticmethod
        def build_entries(root, directory, target, entries, jobs):
            builds.append((target, list(entries), jobs))
            return [{"entry": entry} for entry in entries]

    monkeypatch.setattr(
        reduction_packages,
        "build_packages",
        lambda root, output, target, **options: reductions.append(
            (output, target, options)
        ),
    )
    monkeypatch.setattr(
        operator_packages,
        "build_packages",
        lambda root, output, target, **options: operators.append(
            (output, target, options)
        ),
    )
    output = tmp_path / "reductions"
    maximum = "all_reduce_maxfloat32"
    upstream_runner.translate_ahead(
        tmp_path, packages, output, 2, [(Builder, ["a"])], (maximum,)
    )
    assert reductions == [
        (output, "opengl", {"family": "all", "entries": (maximum,), "on_demand": True})
    ]
    upstream_runner.translate_ahead(
        tmp_path, packages, output, 2, [(Builder, ["a"])], (maximum,), columns=["col"]
    )
    assert reductions[2:] == [
        (
            tmp_path / "reductions-column",
            "opengl",
            {"family": "column", "entries": ("col",), "jobs": 2},
        )
    ]
    assert builds == [("opengl", ["a"], 2)] * 2
    assert operators == []
    # A float16 maximum has its own family beside the float32 one, and each
    # element-wise family its own directory.
    del reductions[:]
    upstream_runner.translate_ahead(
        tmp_path,
        packages,
        output,
        2,
        [],
        ("all_reduce_maxfloat16", maximum, "all_reduce_minfloat16"),
        operators=("half", "half-arithmetic"),
    )
    assert reductions == [
        (output, "opengl", {"family": "all", "entries": (maximum,), "on_demand": True}),
        (
            tmp_path / "reductions-half",
            "opengl",
            {
                "family": "half",
                "entries": ("all_reduce_maxfloat16", "all_reduce_minfloat16"),
                "on_demand": True,
            },
        ),
    ]
    assert operators == [
        (tmp_path / "half-packages", "opengl", {"family": "half"}),
        (
            tmp_path / "half-arithmetic-packages",
            "opengl",
            {"family": "half-arithmetic"},
        ),
    ]


@pytest.mark.parametrize(
    "maxima, message",
    [(("all_reduce_maxint8",), "reduction family"), (("row_reduce_x",), "family")],
)
def test_maxima_need_a_whole_array_family(maxima, message):
    with pytest.raises(ValueError, match=message):
        upstream_runner.maxima_directories(Path("r"), maxima)


def test_operator_families_are_the_ones_the_worker_loads():
    assert upstream_runner.operator_directory(Path("o/r"), "half") == Path(
        "o/half-packages"
    )
    assert upstream_runner.operator_directory(Path("o/r"), "bfloat") == Path(
        "o/bfloat-packages"
    )
    with pytest.raises(ValueError, match="operator package family"):
        upstream_runner.operator_directory(Path("o/r"), "integer64")


def test_cli_worker_loads_every_reduction_directory(monkeypatch):
    seen = []
    monkeypatch.setattr(
        upstream_runner,
        "worker",
        lambda args, tests, trace_values: seen.append(
            (args.reductions, args.half, args.half_arithmetic, args.bfloat)
        ),
    )
    common = ["--mlx-root", "m", "--packages", "p", "--output-dir", "o"]
    upstream_runner.main(
        verify_upstream_qvm_splitk,
        [*common, "--worker", "native", "--reductions", "r", "--reductions", "c"],
    )
    upstream_runner.main(verify_upstream_qvm, [*common, "--worker", "cpu"])
    upstream_runner.main(
        verify_upstream_qmm,
        [*common, "--worker", "native", "--reductions", "h"]
        + ["--half", "a", "--half-arithmetic", "b"],
    )
    upstream_runner.main(
        verify_upstream_fp_qmv,
        [*common, "--worker", "native", "--reductions", "r", "--bfloat", "f"],
    )
    assert seen == [
        ([Path("r"), Path("c")], None, None, None),
        (None, None, None, None),
        ([Path("h")], Path("a"), Path("b"), None),
        ([Path("r")], None, None, Path("f")),
    ]


@pytest.mark.parametrize("name", sorted(RUNNERS))
@pytest.mark.parametrize(
    "fault", [None, "random", "packages", "cpu", "native", "source"]
)
def test_runner_prepares_packages_before_both_workers(
    tmp_path, monkeypatch, name, fault
):
    spec = RUNNERS[name]
    items = trace(spec, "directx")
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "directx"}))
    preparations = []

    def prepared(root):
        preparations.append(root)
        changed = fault == "source" and len(preparations) > 1
        return {"source": "changed" if changed else "pinned"}

    monkeypatch.setattr(upstream_runner, "verify_prepared", prepared)
    monkeypatch.setattr(
        upstream_runner,
        "upstream_test_sources",
        lambda *args: {"test_quantized.py": "pin"},
    )
    commands = []

    def execute(command, **kwargs):
        if "--worker" in command:
            mode = command[command.index("--worker") + 1]
        else:
            mode = "packages" if "--prepare" in command else "random"
        commands.append((mode, command))
        destination = Path(command[command.index("--output-dir") + 1])
        destination.mkdir()
        if mode in ("cpu", "native"):
            quantization.write_json(
                destination / "results.json", record(spec, mode, len(items))
            )
            if mode == "native":
                (destination / "trace.jsonl").write_text(
                    "\n".join(json.dumps(item) for item in items)
                )
        return SimpleNamespace(returncode=int(mode == fault))

    monkeypatch.setattr(quantization.subprocess, "run", execute)
    output = tmp_path / "evidence"
    if fault:
        with pytest.raises(RuntimeError, match="workers failed|sources changed"):
            spec.run(tmp_path, packages, output)
        assert not (output / "evidence.json").exists()
    else:
        evidence = spec.run(tmp_path, packages, output)
        assert evidence["fullUpstreamSuite"] is False
        assert evidence["target"] == "directx"
        assert sum(evidence["entries"].values()) == len(items)
        assert json.loads((output / "evidence.json").read_text()) == evidence
    expected = ["random", "packages", "cpu", "native"]
    if fault in ("random", "packages"):
        expected = expected[: expected.index(fault) + 1]
    assert [mode for mode, _ in commands] == expected
    for mode, command in commands:
        assert (output / f"{mode}.command.json").exists()
        timeout = int(command[command.index("--timeout-seconds") + 1])
        assert timeout == spec.TIMEOUTS[mode]
        if mode != "random":
            assert spec.MODULE in command
        if mode == "native":
            loaded = [
                command[index + 1]
                for index, item in enumerate(command)
                if item == "--reductions"
            ]
            expected_reductions = [
                str(output / ("reductions-half" if name == "qmm" else "reductions"))
            ]
            if getattr(spec, "COLUMN_SUMS", ()):
                expected_reductions.append(str(output / "reductions-column"))
            assert loaded == expected_reductions
            operators = {
                option: command[command.index(option) + 1]
                for option in upstream_runner.OPERATOR_OPTIONS.values()
                if option in command
            }
            assert operators == {
                upstream_runner.OPERATOR_OPTIONS[family]: str(
                    output / f"{family}-packages"
                )
                for family in getattr(spec, "OPERATORS", ())
            }
            assert command[command.index("--random") + 1] == str(
                output / "random-packages"
            )
    assert len(preparations) == 2


@pytest.mark.parametrize("name", sorted(RUNNERS))
def test_runner_module_names_itself(name):
    spec = RUNNERS[name]
    assert spec.MODULE == spec.__name__
    assert spec.LABEL == "test_" + name
    first, *others = spec.UPSTREAM_TESTS
    assert first == f"test_quantized.TestQuantized.test_{name}"
    # Further tests are distinct tests of the same pinned suite, such as the
    # named test's multi-row form.
    assert len(set(spec.UPSTREAM_TESTS)) == len(spec.UPSTREAM_TESTS)
    assert all(test.startswith("test_quantized.TestQuantized.test_") for test in others)
    assert set(spec.TIMEOUTS) == {"random", "packages", "cpu", "native"}


@pytest.mark.parametrize("name", sorted(RUNNERS))
def test_cli_dispatches_each_mode(tmp_path, monkeypatch, name):
    spec = RUNNERS[name]
    calls = []
    monkeypatch.setattr(
        spec, "prepare_packages", lambda *args: calls.append(("prepare", args))
    )
    monkeypatch.setattr(
        upstream_runner,
        "worker",
        lambda args, tests, trace_values: calls.append(
            ("worker", args.worker, tests, trace_values)
        ),
    )
    monkeypatch.setattr(
        upstream_runner, "run", lambda *args: calls.append(("run", args[0]))
    )
    common = ["--mlx-root", "m", "--packages", "p", "--output-dir", "o"]
    upstream_runner.main(spec, [*common, "--prepare", "--jobs", "2"])
    upstream_runner.main(spec, [*common, "--worker", "native"])
    upstream_runner.main(spec, common)
    assert calls == [
        ("prepare", (Path("m"), Path("p"), Path("o"), 2)),
        ("worker", "native", spec.UPSTREAM_TESTS, False),
        ("run", spec),
    ]


def test_ci_runs_every_upstream_test_on_every_quantized_host_platform():
    import yaml

    workflow = yaml.safe_load(WORKFLOW.read_text())
    job = workflow["jobs"]["quantized-upstream-host"]
    assert job["if"] == "needs.ci-plan.outputs.heavy == 'true'"
    assert job["needs"] == ["ci-plan", "portable-host"]
    assert "continue-on-error" not in job
    assert "quantized-upstream-host" in workflow["jobs"]["result"]["needs"]
    matrix = job["strategy"]["matrix"]
    assert sorted(matrix["test"]) == sorted(RUNNERS)
    assert {item["target"]: item["os"] for item in matrix["include"]} == {
        "metal": "macos-26",
        "directx": "windows-2025",
        "opengl": "ubuntu-24.04",
    }
    assert matrix.get("exclude") is None
    assert "${{ matrix.test }}" in job["name"] and "${{ matrix.target }}" in job["name"]
    names = [item.get("name") for item in job["steps"]]
    steps = dict(zip(names, job["steps"]))
    step = steps[RUN_STEP]
    assert "if" not in step and "continue-on-error" not in step
    assert "set -euo pipefail" in step["run"]
    assert "portable_host.verify_upstream_${{ matrix.test }}" in step["run"]
    assert "--packages .mlx-portable-quantized/base/packages" in step["run"]
    assert "tee .mlx-portable-quantized/upstream-${{ matrix.test }}.log" in step["run"]
    assert step["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    (deadline,) = re.findall(r"--timeout-seconds (\d+)", step["run"])
    for spec in RUNNERS.values():
        # The outer deadline covers every bounded command each runner starts.
        assert int(deadline) >= sum(spec.TIMEOUTS.values()) + 300
    deadlines = [
        int(value)
        for item in job["steps"]
        for value in re.findall(r"--timeout-seconds (\d+)", item.get("run", ""))
    ]
    # Every bounded command, plus setup, fits inside the job limit.
    assert sum(deadlines) + 15 * 60 <= job["timeout-minutes"] * 60
    order = [
        "Build adapted upstream MLX",
        RUN_STEP,
        "Retain quantized execution evidence",
    ]
    assert [names.index(item) for item in order] == sorted(
        names.index(item) for item in order
    )
    contracts = steps["Validate quantized host contracts"]["run"]
    for name in (*RUNNERS, "runner"):
        assert f"test_portable_upstream_{name}.py" in contracts
    retain = steps["Retain quantized execution evidence"]
    assert retain["if"] == "always()"
    assert retain["with"]["name"].startswith("mlx-portable-upstream-quantized-")
    assert "${{ matrix.test }}" in retain["with"]["name"]
