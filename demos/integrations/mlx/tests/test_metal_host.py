import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx import run_metal_host as host
from tests.ci_helpers import assert_paths_covered


def workload_records(dataset):
    base = complex(2, 1)
    if dataset != "ordinary":
        base = complex(-4, -0.0 if dataset == "negative-zero" else 0.0)
    exponent = complex(0.5, 0)
    expected = base**exponent
    return [
        {
            "dataset": dataset,
            "shape": shape,
            "arrayShape": layout,
            "dtype": "complex64",
            "outputs": math.prod(layout),
            "matched": True,
            **{
                key: [[value.real, value.imag] for _ in range(math.prod(layout))]
                for key, value in (
                    ("base", base),
                    ("exponent", exponent),
                    ("expected", expected),
                    ("actual", expected),
                )
            },
        }
        for shape, layout in host.HOST_LAYOUTS.items()
    ]


def workload_dispatches():
    return [{"entry": f"{shape}_Powercomplex64"} for shape in host.HOST_LAYOUTS]


@pytest.mark.parametrize("dataset", host.HOST_DATASETS)
def test_host_numerical_evidence_checks_every_readback(dataset):
    records = workload_records(dataset)
    host.verify_host_workloads(records, workload_dispatches(), dataset)
    assert sum(item["outputs"] for item in records) == 134
    records[-1]["actual"][-1][1] += 1
    with pytest.raises(ValueError, match="readback does not match"):
        host.verify_host_workloads(records, workload_dispatches(), dataset)


@pytest.mark.parametrize("field", ["base", "exponent", "expected", "actual"])
@pytest.mark.parametrize("bad_value", [[], [1], [0, float("nan")], [True, 0]])
def test_host_numerical_evidence_rejects_invalid_complex_values(field, bad_value):
    records = workload_records("ordinary")
    records[0][field][0] = bad_value
    with pytest.raises(ValueError, match="finite complex readbacks"):
        host.verify_host_workloads(records, workload_dispatches(), "ordinary")


@pytest.mark.parametrize("field", ["base", "exponent", "expected", "actual"])
def test_host_numerical_evidence_requires_complete_buffers(field):
    records = workload_records("ordinary")
    records[-1][field].pop()
    with pytest.raises(ValueError, match="readback count"):
        host.verify_host_workloads(records, workload_dispatches(), "ordinary")


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset", "missing"),
        ("arrayShape", [1]),
        ("dtype", "float32"),
        ("outputs", True),
        ("outputs", 2),
        ("matched", False),
    ],
)
def test_host_numerical_evidence_requires_exact_layouts(field, value):
    records = workload_records("ordinary")
    records[0][field] = value
    with pytest.raises(ValueError, match="layout or numerical"):
        host.verify_host_workloads(records, workload_dispatches(), "ordinary")


@pytest.mark.parametrize("dataset", host.HOST_DATASETS[1:])
def test_host_numerical_evidence_requires_signed_zero_inputs(dataset):
    records = workload_records(dataset)
    records[0]["base"][0][1] *= -1
    with pytest.raises(ValueError, match="signed-zero inputs"):
        host.verify_host_workloads(records, workload_dispatches(), dataset)


@pytest.mark.parametrize("duplicate", [False, True])
def test_host_numerical_evidence_requires_exact_case_accounting(duplicate):
    records = workload_records("ordinary")
    if duplicate:
        records.append(records[0])
    else:
        records.pop()
    with pytest.raises(ValueError, match="evidence is incomplete"):
        host.verify_host_workloads(records, workload_dispatches(), "ordinary")


@pytest.mark.parametrize("duplicate", [False, True])
def test_host_numerical_evidence_requires_one_dispatch_per_case(duplicate):
    dispatches = workload_dispatches()
    if duplicate:
        dispatches.append(dispatches[0])
    else:
        dispatches.pop()
    with pytest.raises(ValueError, match="dispatch each expected entry once"):
        host.verify_host_workloads(workload_records("ordinary"), dispatches, "ordinary")


def test_host_numerical_evidence_recomputes_expected_values():
    records = workload_records("ordinary")
    records[0]["actual"][0] = records[0]["expected"][0] = [0, 0]
    with pytest.raises(ValueError, match="readback does not match"):
        host.verify_host_workloads(records, workload_dispatches(), "ordinary")


@pytest.mark.parametrize("count,skipped", [(1, 0), (287, 0), (287, 3)])
def test_upstream_test_accounting(tmp_path, count, skipped):
    path = tmp_path / "result"
    suffix = f" (skipped={skipped})" if skipped else ""
    path.write_text(f"Ran {count} tests in 12.3s\n\nOK{suffix}\n")
    assert host.unittest_counts(path) == {"total": count, "skipped": skipped}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Ran 0 tests in 0s\nOK\n",
        "Ran 1 test in 1s\nFAILED\n",
        "Ran 1 test in 1s\nOK\nRan 1 test in 1s\nOK\n",
    ],
)
def test_incomplete_upstream_accounting_is_rejected(tmp_path, text):
    path = tmp_path / "result"
    path.write_text(text)
    with pytest.raises(ValueError):
        host.unittest_counts(path)


def test_dispatch_trace_requires_loaded_entry_and_dimensions(tmp_path):
    path = tmp_path / "trace"
    path.write_text(
        "library\tss_Powercomplex64\n"
        "dispatch\tss_Powercomplex64\tthreads\t2,3,1\t2,1,1\n"
    )
    assert host.parse_trace(path) == [
        {
            "entry": "ss_Powercomplex64",
            "kind": "threads",
            "grid": [2, 3, 1],
            "group": [2, 1, 1],
        }
    ]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "library\tss_Powercomplex64\n",
        "library\tunknown\n",
        "dispatch\tss_Powercomplex64\tthreads\t1,1,1\t1,1,1\n",
        "library\tss_Powercomplex64\ndispatch\tss_Powercomplex64\tthreads\t0,1,1\t1,1,1\n",
        "library\tss_Powercomplex64\ndispatch\tss_Powercomplex64\tthreads\t1,1\t1,1,1\n",
        "library\tss_Powercomplex64\ndispatch\tss_Powercomplex64\tunknown\t1,1,1\t1,1,1\n",
    ],
)
def test_unproven_dispatch_is_rejected(tmp_path, text):
    path = tmp_path / "trace"
    path.write_text(text)
    with pytest.raises(ValueError):
        host.parse_trace(path)


def test_native_command_preserves_failure_evidence(tmp_path):
    result = host.run(
        [sys.executable, "-c", "raise SystemExit(3)"], tmp_path, "failure", check=False
    )
    assert result["returncode"] == 3
    assert json.loads((tmp_path / "failure.json").read_text()) == result


def test_native_command_records_only_explicit_runtime_controls(tmp_path):
    controls = {
        "DEVICE": "gpu",
        "CROSTL_METAL_LIBRARY_OVERRIDES": str(tmp_path / "combined-libraries"),
        "CROSTL_METAL_LIBRARY_TRACE": str(tmp_path / "trace.tsv"),
    }
    result = host.run(
        [sys.executable, "-c", "pass"],
        tmp_path,
        "controls",
        env={**os.environ, **controls, "UNRELATED_SECRET": "must-not-be-recorded"},
    )
    assert result["runtimeEnvironment"] == controls
    assert "must-not-be-recorded" not in (tmp_path / "controls.json").read_text()
    assert json.loads((tmp_path / "controls.json").read_text()) == result


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX native Metal worker")
def test_native_command_is_bounded(tmp_path):
    with pytest.raises(RuntimeError, match="deadline"):
        host.run(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            tmp_path,
            "deadline",
            timeout=0.1,
        )
    assert json.loads((tmp_path / "deadline.json").read_text())["timedOut"] is True


@pytest.mark.skipif(
    sys.platform == "win32", reason="Metal uses POSIX virtual environments"
)
def test_verify_preserves_virtual_environment_executable(tmp_path, monkeypatch):
    executable = tmp_path / "python"
    executable.symlink_to(sys.executable)
    calls = []
    monkeypatch.setattr(host, "verify", lambda *args: calls.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "host",
            "verify",
            "--mlx-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--python",
            str(executable),
        ],
    )
    host.main()
    assert calls[0][1] == executable
    assert calls[0][1] != executable.resolve()


def test_prepare_refuses_existing_header(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "verify_checkout", lambda *args, **kwargs: None)
    header = tmp_path / host.HEADER
    header.parent.mkdir(parents=True)
    header.write_text("local work")
    with pytest.raises(ValueError, match="existing header"):
        host.prepare(tmp_path, tmp_path / "output")
    assert header.read_text() == "local work"


def test_ci_requires_pinned_native_host_execution():
    from tests.ci_helpers import assert_workflow_triggers

    root = Path(__file__).resolve().parents[4]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    assert host.MLX_COMMIT in workflow
    assert "runs-on: macos-26" in workflow
    assert 'python-version: "3.13"' in workflow
    assert "-DMLX_METAL_JIT=ON" in workflow
    assert "run_metal_host.py prepare" in workflow
    assert "run_metal_host.py verify" in workflow
    assert "--timeout-seconds 3300" in workflow
    assert '--label "MLX translated host verification"' in workflow
    assert (
        "pytest -q -n auto demos/integrations/mlx/tests/test_metal_host.py" in workflow
    )
    assert 'CROSTL_REQUIRE_METAL_FLOAT_ATOMICS: "1"' in workflow
    assert '--label "Native float atomics"' in workflow
    assert "--basetemp=.mlx-metal-host/float-atomics/pytest" in workflow
    assert "--junitxml=.mlx-metal-host/float-atomics/results.xml" in workflow
    assert_workflow_triggers(
        workflow,
        "tests/test_translator/test_metal_float_atomics.py",
        "tests/fixtures/runtime_verification/metal_uint32_buffers.swift",
    )
    assert "tests/test_translator/test_metal_float_atomics.py \\" in workflow
    assert 'CROSTL_REQUIRE_MLX_GATED_DELTA_METAL: "1"' in workflow
    assert "CROSTL_MLX_CURRENT_ROOT: mlx-upstream" in workflow
    assert "--basetemp=.mlx-metal-host/gated-delta/pytest" in workflow
    assert "--junitxml=.mlx-metal-host/gated-delta/results.xml" in workflow
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_gated_delta_metal.py"
    )
    assert (
        "demos/integrations/mlx/tests/kernels/test_gated_delta_metal.py \\" in workflow
    )
    assert "if: always()" in workflow
    assert "continue-on-error" not in workflow


def test_ci_prepares_metal_before_required_native_tests():
    import yaml

    from tools import ci_coverage

    root = Path(__file__).resolve().parents[4]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "metal-host", "Prepare Metal host toolchain"
    )
    assert "if:" not in step and "continue-on-error" not in step
    assert "timeout-minutes: 10" in step
    assert "set -euo pipefail" in step
    assert "if ! xcrun --sdk macosx metal --version; then" in step
    assert "xcodebuild -downloadComponent MetalToolchain" in step
    assert "metal --version | tee .mlx-metal-host/toolchain/metal-version.txt" in step
    assert "swiftc --version | tee .mlx-metal-host/toolchain/swift-version.txt" in step
    assert "metal -std=metal3.1 -Werror -c" in step
    assert "tests/fixtures/runtime_verification/vector_add.metal" in step
    assert "metallib .mlx-metal-host/toolchain/probe.air" in step
    for suffix in ("air", "metallib"):
        assert f"test -s .mlx-metal-host/toolchain/probe.{suffix}" in step
    assert ci_coverage.workflow_job_step_after(
        workflow,
        "metal-host",
        "Validate Metal argument inference",
        "Prepare Metal host toolchain",
    )
    upload = ci_coverage.workflow_job_step_section(
        workflow, "metal-host", "Retain native host execution evidence"
    )
    assert "if: always()" in upload
    upload_step = yaml.safe_load(upload)[0]
    assert ".mlx-metal-host" in upload_step["with"]["path"].splitlines()
    assert upload_step["with"]["include-hidden-files"] is True
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/fixtures/runtime_verification/vector_add.metal",
        )


def test_ci_requires_pinned_attention_numerical_execution():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[4]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "metal-host", "Execute pinned MLX attention row dots"
    )
    assert "if:" not in step and "continue-on-error" not in step
    assert 'CROSTL_REQUIRE_MLX_ATTENTION_RUNTIME: "1"' in step
    assert "CROSTL_MLX_ATTENTION_TARGET: metal" in step
    assert "CROSTL_MLX_CURRENT_ROOT: mlx-upstream" in step
    assert "set -euo pipefail" in step
    assert "--timeout-seconds 900" in step
    assert "pytest -q -n auto --dist loadscope" in step
    assert "--basetemp=.mlx-metal-host/attention-odo/pytest" in step
    assert "--junitxml=.mlx-metal-host/attention-odo/results.xml" in step
    assert "tee .mlx-metal-host/attention-odo.log" in step
    assert "demos/integrations/mlx/tests/kernels/test_attention_odo_runtime.py" in step
    assert ci_coverage.workflow_job_step_after(
        workflow,
        "metal-host",
        "Execute pinned MLX attention row dots",
        "Checkout pinned upstream MLX",
    )
    for event in ("pull_request", "push"):
        paths = ci_coverage.workflow_event_path_filters(workflow, event)
        for path in (
            "demos/integrations/mlx/tests/kernels/test_attention_odo_runtime.py",
            "demos/integrations/mlx/tests/kernels/test_gated_delta_runtime.py",
            "demos/integrations/mlx/tests/kernels/test_gated_delta_metal.py",
            "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
        ):
            assert_paths_covered(paths, path)
    assert ci_coverage.workflow_job_timeout_minutes(workflow, "metal-host") * 60 > (
        120 + 180 + 300 + 600 + 900 + 1800 + 3300 + 600
    )


def test_ci_requires_half_conversion_reference():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[4]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "metal-host", "Validate half conversion reference"
    )
    assert "if:" not in step and "continue-on-error" not in step
    assert 'CROSTL_REQUIRE_HALF_CONVERSION_RUNTIME: "1"' in step
    assert "CROSTL_HALF_CONVERSION_TARGET: metal" in step
    assert "--timeout-seconds 180" in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=.mlx-metal-host/half-conversions/pytest" in step
    assert "--junitxml=.mlx-metal-host/half-conversions/results.xml" in step
    assert "tee .mlx-metal-host/half-conversions.log" in step
    assert "test_opengl_half_conversion.py -k half_rounding_native" in step
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_opengl_half_conversion.py",
        )


def test_ci_requires_resident_attention_reductions():
    import re

    from tools import ci_coverage

    root = Path(__file__).resolve().parents[4]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "metal-host", "Execute pinned resident attention reductions"
    )
    assert "if:" not in step and "continue-on-error" not in step
    assert 'CROSTL_REQUIRE_MLX_ATTENTION_REDUCE_RUNTIME: "1"' in step
    assert "CROSTL_MLX_ATTENTION_REDUCE_TARGET: metal" in step
    assert "CROSTL_MLX_CURRENT_ROOT: mlx-upstream" in step
    assert 'PYTEST_XDIST_AUTO_NUM_WORKERS: "2"' in step
    assert "CROSTL_REQUIRE_MLX_ATTENTION_REDUCE_COMPILE" not in step
    assert "set -euo pipefail" in step
    assert "--timeout-seconds 300" in step
    assert "pytest -q -n auto --dist loadscope" in step
    assert "--basetemp=.mlx-metal-host/attention-reduce/pytest" in step
    assert "--junitxml=.mlx-metal-host/attention-reduce/results.xml" in step
    assert "tee .mlx-metal-host/attention-reduce.log" in step
    assert (
        "demos/integrations/mlx/tests/kernels/test_attention_reduce_runtime.py" in step
    )
    assert ci_coverage.workflow_job_step_after(
        workflow,
        "metal-host",
        "Execute pinned resident attention reductions",
        "Checkout pinned upstream MLX",
    )
    for event in ("push", "pull_request"):
        for path in (
            "demos/integrations/mlx/tests/kernels/test_attention_reduce_runtime.py",
            "tests/fixtures/runtime_verification/metal_dispatch_sequence.swift",
        ):
            assert_paths_covered(
                ci_coverage.workflow_event_path_filters(workflow, event), path
            )
    assert ci_coverage.workflow_job_timeout_minutes(workflow, "metal-host") * 60 > sum(
        int(value)
        for value in re.findall(
            r"--timeout-seconds (\d+)",
            ci_coverage.workflow_job_text(workflow, "metal-host"),
        )
    )


def test_ci_requires_pinned_attention_derivative_execution():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[4]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "metal-host", "Execute pinned MLX attention derivatives"
    )
    assert "if:" not in step and "continue-on-error" not in step
    assert 'CROSTL_REQUIRE_MLX_ATTENTION_DS_RUNTIME: "1"' in step
    assert "CROSTL_MLX_ATTENTION_DS_TARGET: metal" in step
    assert "CROSTL_MLX_CURRENT_ROOT: mlx-upstream" in step
    assert 'PYTEST_XDIST_AUTO_NUM_WORKERS: "2"' in step
    assert "set -euo pipefail" in step
    assert "--timeout-seconds 600" in step
    assert "pytest -q -n auto --dist loadscope" in step
    assert "--basetemp=.mlx-metal-host/attention-ds/pytest" in step
    assert "--junitxml=.mlx-metal-host/attention-ds/results.xml" in step
    assert "tee .mlx-metal-host/attention-ds.log" in step
    assert "demos/integrations/mlx/tests/kernels/test_attention_ds_runtime.py" in step
    for event in ("pull_request", "push"):
        paths = ci_coverage.workflow_event_path_filters(workflow, event)
        for path in (
            "demos/integrations/mlx/tests/kernels/test_attention_ds_runtime.py",
            "demos/integrations/mlx/tests/kernels/test_attention_odo_runtime.py",
            "demos/integrations/mlx/tests/kernels/test_gated_delta_runtime.py",
            "demos/integrations/mlx/tests/kernels/test_gated_delta_metal.py",
            "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
        ):
            assert_paths_covered(paths, path)
    assert ci_coverage.workflow_job_timeout_minutes(workflow, "metal-host") * 60 > (
        120 + 180 + 300 + 600 + 900 + 600 + 1800 + 3300 + 600
    )


def test_ci_requires_pinned_backward_numerical_execution():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[4]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "metal-host", "Execute pinned MLX gated-delta gradients"
    )
    assert "if:" not in step and "continue-on-error" not in step
    assert 'CROSTL_REQUIRE_MLX_GATED_DELTA_RUNTIME: "1"' in step
    assert "CROSTL_MLX_CURRENT_ROOT: mlx-upstream" in step
    assert 'PYTEST_XDIST_AUTO_NUM_WORKERS: "2"' in step
    assert "set -euo pipefail" in step
    assert "--timeout-seconds 600" in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=.mlx-metal-host/gated-delta-runtime/pytest" in step
    assert "--junitxml=.mlx-metal-host/gated-delta-runtime/results.xml" in step
    assert "tee .mlx-metal-host/gated-delta-runtime.log" in step
    assert "demos/integrations/mlx/tests/kernels/test_gated_delta_runtime.py" in step
    dependencies = ci_coverage.workflow_job_step_section(
        workflow, "metal-host", "Install CrossTL and test dependencies"
    )
    assert "numpy" in dependencies
    assert ci_coverage.workflow_job_step_after(
        workflow,
        "metal-host",
        "Execute pinned MLX gated-delta gradients",
        "Checkout pinned upstream MLX",
    )
    for event in ("pull_request", "push"):
        paths = ci_coverage.workflow_event_path_filters(workflow, event)
        for path in (
            "demos/integrations/mlx/tests/kernels/test_gated_delta_runtime.py",
            "demos/integrations/mlx/tests/kernels/test_gated_delta_metal.py",
            "demos/integrations/mlx/tests/fixtures/gated_delta_reference.py",
            "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
        ):
            assert_paths_covered(paths, path)


@pytest.mark.parametrize("patched", [False, True])
def test_checkout_identity(tmp_path, monkeypatch, patched):
    path = tmp_path / "device.cpp"
    path.write_text("pinned adapter" if patched else "pinned original")
    expected = host.digest(path)
    monkeypatch.setattr(host, "SOURCE_HASHES", {"device.cpp": (expected, expected)})
    header = tmp_path / host.HEADER
    if patched:
        header.parent.mkdir(parents=True)
        header.write_bytes((host.HERE / "metal_library_overrides.h").read_bytes())
    monkeypatch.setattr(
        host.subprocess,
        "check_output",
        lambda command, **kwargs: (
            host.MLX_COMMIT
            if "rev-parse" in command
            else ("device.cpp\n" if patched else "")
        ),
    )
    host.verify_checkout(tmp_path, patched=patched)
    path.write_text("unexpected changes")
    with pytest.raises(ValueError, match="runtime source"):
        host.verify_checkout(tmp_path, patched=patched)


@pytest.mark.parametrize(
    "revision,modified", [("bad", ""), (host.MLX_COMMIT, "test_ops.py\n")]
)
def test_checkout_rejects_wrong_pin_or_unrelated_edits(
    tmp_path, monkeypatch, revision, modified
):
    monkeypatch.setattr(
        host.subprocess,
        "check_output",
        lambda command, **kwargs: revision if "rev-parse" in command else modified,
    )
    with pytest.raises(ValueError):
        host.verify_checkout(tmp_path, patched=False)


@pytest.mark.parametrize("corruption", [None, "readback", "trace"])
def test_host_evidence_requires_complete_command_and_numerical_records(
    tmp_path, monkeypatch, corruption
):
    root, output = tmp_path / "mlx", tmp_path / "proof"
    test_source = root / "python/tests/test_ops.py"
    test_source.parent.mkdir(parents=True)
    test_source.write_text("unchanged upstream test")
    monkeypatch.setattr(host, "verify_checkout", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        host, "compile_libraries", lambda *args: (output / "combined-libraries", [])
    )
    shapes = tuple(host.HOST_LAYOUTS)

    def fake_run(command, directory, name, **kwargs):
        result = {"returncode": 0, "command": list(map(str, command))}
        env = kwargs["env"]
        if name in {"runtime-identity", "upstream-original"}:
            assert "CROSTL_METAL_LIBRARY_OVERRIDES" not in env
            assert "CROSTL_METAL_LIBRARY_TRACE" not in env
        if name == "runtime-identity":
            host.save_json(
                directory / f"{name}.stdout",
                {
                    "module": str(root / "core.so"),
                    "device": "gpu",
                    "metalAvailable": True,
                },
            )
        if name.startswith("upstream-"):
            (directory / f"{name}.stderr").write_text("Ran 163 tests in 1.0s\nOK\n")
        if name == "upstream-translated" or name.startswith("execute-host-workloads-"):
            assert env["CROSTL_METAL_LIBRARY_OVERRIDES"] == str(
                output / "combined-libraries"
            )
            selected = ("ss",) if name == "upstream-translated" else shapes
            if corruption == "trace" and name.endswith("negative-zero"):
                selected = (*selected, "ss")
            Path(env["CROSTL_METAL_LIBRARY_TRACE"]).write_text(
                "".join(
                    f"library\t{s}_Powercomplex64\ndispatch\t{s}_Powercomplex64\tthreads\t1,1,1\t1,1,1\n"
                    for s in selected
                )
            )
        if name.startswith("execute-host-workloads-"):
            assert command[-1] != directory / (name + ".json")
            assert command[-3] == "--dataset"
            records = workload_records(command[-2])
            if corruption == "readback" and name.endswith("negative-zero"):
                records[-1]["actual"][-1][1] *= -1
            host.save_json(command[-1], records)
        if name == "missing-required-library":
            assert env["CROSTL_METAL_LIBRARY_OVERRIDES"] == str(
                output / "missing-library"
            )
            result["returncode"] = 1
            (directory / f"{name}.stderr").write_text(
                "Cannot load required translated Metal library"
            )
        host.save_json(directory / f"{name}.json", result)
        return result

    monkeypatch.setattr(host, "run", fake_run)
    if corruption:
        with pytest.raises(ValueError):
            host.verify(root, Path(sys.executable), output)
        assert not (output / "evidence.json").exists()
        assert (output / "host-workloads-negative-zero.json").exists()
        return
    host.verify(root, Path(sys.executable), output)
    evidence = json.loads((output / "evidence.json").read_text())
    assert evidence["schemaVersion"] == 2
    assert len(evidence["hostWorkloads"]) == 24
    assert sum(item["outputs"] for item in evidence["hostWorkloads"]) == 402
    assert len(evidence["hostWorkloadDispatches"]) == 24
    assert {item["dataset"] for item in evidence["hostWorkloadDispatches"]} == set(
        host.HOST_DATASETS
    )
    assert evidence["missingRequiredLibraryRejected"] is True
    assert evidence["fullTranslatedBackend"] is False


@pytest.mark.parametrize("link_failure", [False, True])
def test_host_uses_combined_library_and_rejects_link_failure(
    tmp_path, monkeypatch, link_failure
):
    import crosstl.project

    entries = {"first", "second"}
    monkeypatch.setattr(host, "ENTRIES", entries)
    monkeypatch.setattr(
        crosstl.project, "load_project_config", lambda root, path: path.parent
    )

    def translate_project(directory, **kwargs):
        artifacts = []
        for entry in sorted(entries):
            source = directory / f"{entry}.metal"
            source.write_text(f"kernel void {entry}() {{}}")
            artifacts.append(
                {
                    "entryPoint": {"source": entry},
                    "path": str(source.relative_to(tmp_path)),
                }
            )
        payload = {"summary": {"failedCount": 0}, "artifacts": artifacts}
        return SimpleNamespace(
            to_json=lambda: payload,
            write_json=lambda path: host.save_json(path, payload),
        )

    monkeypatch.setattr(crosstl.project, "translate_project", translate_project)
    combined_inputs = []

    def run(command, output, name, **kwargs):
        if name == "link-combined":
            combined_inputs.extend(Path(item).stem for item in command[4:-2])
            if link_failure:
                raise RuntimeError("duplicate helper")
        Path(command[-1]).write_bytes(name.encode())

    monkeypatch.setattr(host, "run", run)
    output = tmp_path / "proof"
    if link_failure:
        with pytest.raises(RuntimeError, match="duplicate helper"):
            host.compile_libraries(tmp_path, output)
        assert not (output / "combined-libraries").exists()
    else:
        directory, records = host.compile_libraries(tmp_path, output)
        assert directory.name == "combined-libraries"
        combined = output / "combined.metallib"
        assert {record["librarySha256"] for record in records} == {
            host.digest(combined)
        }
        assert {record["entry"] for record in records} == entries
        for entry in entries:
            assert (
                directory / f"{entry}.metallib"
            ).read_bytes() == combined.read_bytes()
            assert (
                output / "libraries" / f"{entry}.metallib"
            ).read_bytes() != combined.read_bytes()
    assert set(combined_inputs) == entries
