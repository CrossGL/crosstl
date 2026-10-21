"""The unchanged upstream test_qvm runner: routing, trace and worker checks."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    verify_upstream_quantization as quantization,
)
from demos.integrations.mlx.portable_host import verify_upstream_qvm as upstream

GEMV = "gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0"
WORKFLOW = (
    Path(__file__).resolve().parents[5] / ".github/workflows/demo-project-testing.yml"
)


def record(mode, dispatches=None):
    return {
        "mode": mode,
        "tests": list(upstream.UPSTREAM_TESTS),
        "testsRun": 1,
        "success": True,
        "skipped": [],
        "errors": [],
        "failures": [],
        "nativeDispatches": (dispatches or len(trace())) if mode == "native" else 0,
    }


def trace(target="opengl"):
    items = [
        {"entry": entry, "target": target}
        for entry, count in sorted(upstream.expected_dispatches().items())
        for _ in range(count)
    ]
    return items + [
        {"entry": "all_reduce_maxfloat32", "target": target, "threads": 64},
        {"entry": "rbitsc", "target": target},
    ]


def test_expected_dispatches_follow_every_upstream_product():
    counts = upstream.expected_dispatches()
    qvm, gemv = upstream.package_entries()
    # Seven (group, M) pairs keep M >= group; each meets 6 bits, 3 N and 4 B.
    assert sum(1 for _ in upstream.products()) == 7 * 6 * 3 * 4 == 504
    assert gemv == [GEMV] and counts[GEMV] == 504
    assert qvm == sorted(
        f"affine_qvm_float_gs_{group}_b_{bits}_batch_{batch}"
        for group in (32, 64, 128)
        for bits in (2, 3, 4, 5, 6, 8)
        for batch in (0, 1)
    )
    for group, m_count in ((32, 3), (64, 2), (128, 2)):
        for bits in (2, 3, 4, 5, 6, 8):
            # B of 0 and 1 run unbatched, 3 and 8 batched.
            for batch in (0, 1):
                entry = f"affine_qvm_float_gs_{group}_b_{bits}_batch_{batch}"
                assert counts[entry] == m_count * 3 * 2
            for operation in ("quantize", "dequantize"):
                entry = f"affine_{operation}_float_gs_{group}_b_{bits}"
                assert counts[entry] == m_count * 3 * 4
    assert sum(counts.values()) == 504 * 4


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [None, "missing", "extra", "target", "specialization", "reference", "maximum"],
)
def test_trace_requires_every_translated_product(target, fault):
    items = trace(target)
    count = len(items)
    if fault == "missing":
        items.pop()
    elif fault == "extra":
        items.append(items[0])
    elif fault == "target":
        items[0]["target"] = "other"
    elif fault == "specialization":
        items[0]["entry"] = "affine_qmv_float_gs_32_b_4_batch_0"
    elif fault == "reference":
        index = next(i for i, item in enumerate(items) if item["entry"] == GEMV)
        items[index]["entry"] = "gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0"
    elif fault == "maximum":
        items = [item for item in items if item["entry"] != "all_reduce_maxfloat32"]
        count = len(items)
    if fault:
        with pytest.raises(RuntimeError):
            upstream.validate_trace(items, target, count)
    else:
        upstream.validate_trace(items, target, count)


def test_results_name_test_qvm():
    quantization.validate_result(record("cpu"), "cpu", upstream.UPSTREAM_TESTS)
    with pytest.raises(RuntimeError, match="test_qvm did not pass"):
        quantization.validate_result(
            {**record("native"), "failures": [["subtest", "1e-3"]]},
            "native",
            upstream.UPSTREAM_TESTS,
        )
    with pytest.raises(RuntimeError, match="did not pass"):
        quantization.validate_result(record("cpu"), "cpu")


@pytest.mark.parametrize(
    "fault", [None, "random", "packages", "cpu", "native", "source"]
)
def test_runner_prepares_packages_before_both_workers(tmp_path, monkeypatch, fault):
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "directx"}))
    preparations = []

    def prepared(root):
        preparations.append(root)
        changed = fault == "source" and len(preparations) > 1
        return {"source": "changed" if changed else "pinned"}

    monkeypatch.setattr(upstream, "verify_prepared", prepared)
    monkeypatch.setattr(
        upstream, "upstream_test_sources", lambda *args: {"test_quantized.py": "pin"}
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
            quantization.write_json(destination / "results.json", record(mode))
            if mode == "native":
                (destination / "trace.jsonl").write_text(
                    "\n".join(json.dumps(item) for item in trace("directx"))
                )
        return SimpleNamespace(returncode=int(mode == fault))

    monkeypatch.setattr(quantization.subprocess, "run", execute)
    output = tmp_path / "evidence"
    if fault:
        with pytest.raises(RuntimeError, match="workers failed|sources changed"):
            upstream.run(tmp_path, packages, output)
        assert not (output / "evidence.json").exists()
    else:
        evidence = upstream.run(tmp_path, packages, output)
        assert evidence["fullUpstreamSuite"] is False
        assert evidence["entries"][GEMV] == 504
    expected = ["random", "packages", "cpu", "native"]
    if fault in ("random", "packages"):
        expected = expected[: expected.index(fault) + 1]
    assert [mode for mode, _ in commands] == expected
    for mode, command in commands:
        assert (output / f"{mode}.command.json").exists()
        timeout = int(command[command.index("--timeout-seconds") + 1])
        assert timeout == upstream.TIMEOUTS[mode]
        if mode != "random":
            assert "demos.integrations.mlx.portable_host.verify_upstream_qvm" in command
        if mode == "native":
            assert command[command.index("--reductions") + 1] == str(
                output / "reductions"
            )
            assert command[command.index("--random") + 1] == str(
                output / "random-packages"
            )
    assert len(preparations) == 2


def test_prepare_translates_routes_and_indexes_the_maximum(tmp_path, monkeypatch):
    from demos.integrations.mlx.portable_host import (
        gemv_packages,
        matmul_packages,
        reduction_packages,
    )

    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "metal"}))
    calls = {}

    def build(name):
        def run(root, directory, target, entries, jobs):
            calls[name] = (target, list(entries), jobs)
            return [{"entry": entry} for entry in entries]

        return run

    monkeypatch.setattr(matmul_packages, "build_entries", build("qvm"))
    monkeypatch.setattr(gemv_packages, "build_entries", build("gemv"))
    monkeypatch.setattr(
        reduction_packages,
        "build_packages",
        lambda root, output, target, **options: calls.update(
            reductions=(target, options)
        ),
    )
    upstream.prepare_packages(tmp_path, packages, tmp_path / "reductions", 3)
    qvm, gemv = upstream.package_entries()
    assert calls["qvm"] == ("metal", qvm, 3) and len(qvm) == 36
    assert calls["gemv"] == ("metal", [GEMV], 3)
    assert calls["reductions"] == (
        "metal",
        {"family": "all", "entries": ("all_reduce_maxfloat32",), "on_demand": True},
    )


def test_ci_runs_test_qvm_on_every_quantized_host_platform():
    import yaml

    workflow = yaml.safe_load(WORKFLOW.read_text())
    job = workflow["jobs"]["quantized-upstream-host"]
    assert job["if"] == "needs.ci-plan.outputs.heavy == 'true'"
    assert job["needs"] == ["ci-plan", "portable-host"]
    assert "continue-on-error" not in job
    assert "quantized-upstream-host" in workflow["jobs"]["result"]["needs"]
    assert {
        item["target"]: item["os"] for item in job["strategy"]["matrix"]["include"]
    } == {"metal": "macos-26", "directx": "windows-2025", "opengl": "ubuntu-24.04"}
    names = [item.get("name") for item in job["steps"]]
    steps = dict(zip(names, job["steps"]))
    step = steps["Run the unchanged upstream test_qvm through translated kernels"]
    assert "if" not in step and "continue-on-error" not in step
    assert "set -euo pipefail" in step["run"]
    assert "portable_host.verify_upstream_qvm" in step["run"]
    assert "--packages .mlx-portable-quantized/base/packages" in step["run"]
    assert "tee .mlx-portable-quantized/upstream-qvm.log" in step["run"]
    assert step["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    (deadline,) = re.findall(r"--timeout-seconds (\d+)", step["run"])
    # The outer deadline covers every bounded command the runner starts.
    assert int(deadline) >= sum(upstream.TIMEOUTS.values()) + 300
    deadlines = [
        int(value)
        for item in job["steps"]
        for value in re.findall(r"--timeout-seconds (\d+)", item.get("run", ""))
    ]
    # Every bounded command, plus setup, fits inside the job limit.
    assert sum(deadlines) + 15 * 60 <= job["timeout-minutes"] * 60
    order = [
        "Build adapted upstream MLX",
        "Run the unchanged upstream test_qvm through translated kernels",
        "Retain quantized execution evidence",
    ]
    assert [names.index(item) for item in order] == sorted(
        names.index(item) for item in order
    )
    assert (
        "test_portable_upstream_qvm.py" in steps["Validate quantized host contracts"][
            "run"
        ]
    )
    retain = steps["Retain quantized execution evidence"]
    assert retain["if"] == "always()"
    assert retain["with"]["name"].startswith("mlx-portable-upstream-quantized-")
