"""Empty MLX reductions require translated initialization and checked readbacks."""

import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import (
    prepare,
    reduction_packages,
    runtime,
)
from demos.integrations.mlx.portable_host import verify_empty_reductions as proof
from tests.ci_helpers import assert_paths_covered


@pytest.fixture(scope="module", params=["metal", "opengl", "directx"])
def packages(tmp_path_factory, request):
    root = tmp_path_factory.mktemp(f"empty-{request.param}")
    source = root / reduction_packages.SOURCE
    source.parent.mkdir(parents=True)
    source.write_text("""
template<typename T> struct Sum { static constexpr T init = 0; };
template<typename T> struct Prod { static constexpr T init = 1; };
template<typename T> struct And { static constexpr T init = true; };
template<typename T> struct Or { static constexpr T init = false; };
template<typename T, typename Op> kernel void init_reduce(
device T* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  out[tid] = Op::init;
}
#define instantiate_init_reduce(name, tname, type, op) \\
template [[host_name("init_reduce_" #name #tname)]] [[kernel]] \\
decltype(init_reduce<type, op<type>>) init_reduce<type, op<type>>;
instantiate_init_reduce(sum, float32, float, Sum)
instantiate_init_reduce(prod, float32, float, Prod)
instantiate_init_reduce(sum, int32, int, Sum)
instantiate_init_reduce(prod, int32, int, Prod)
instantiate_init_reduce(and, bool_, bool, And)
instantiate_init_reduce(or, bool_, bool, Or)
""")
    original = source.read_bytes()
    output = root / "packages"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            reduction_packages.subprocess,
            "check_output",
            lambda command, **kwargs: prepare.COMMIT if "rev-parse" in command else "",
        )
        index = reduction_packages.build_packages(
            root, output, request.param, family="init"
        )
    assert source.read_bytes() == original
    assert reduction_packages.load_index(output, request.param) == index
    assert set(index["descriptors"]) == {
        f"w1/{entry}" for entry in reduction_packages.INIT_ENTRIES
    }
    assert (
        output / "translation/init_reduce.metal"
    ).read_text() == reduction_packages.INIT_SOURCE
    assert (
        "software_subgroup_width"
        not in (output / "translation/crosstl.toml").read_text()
    )
    return output, index


@pytest.mark.parametrize("entry", reduction_packages.INIT_ENTRIES)
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "launch",
        "geometry",
        "exact-grid",
        "count",
        "buffer-count",
        "dtype",
        "null",
        "direction",
        "name",
        "missing",
        "guard",
        "readback",
    ],
)
def test_init_dispatch_contract(packages, tmp_path, entry, fault):
    directory, index = packages
    host = runtime.HostRuntime.__new__(runtime.HostRuntime)
    host.target = index["target"]
    host.descriptors = {}
    host.reduction_directories = {key: directory for key in index["descriptors"]}
    host.reduction_descriptors = dict(index["descriptors"])
    host.trace = tmp_path / "trace.jsonl"
    dtype = reduction_packages.INIT_ENTRIES[entry]
    memory = (runtime.TYPES[dtype] * 3)()
    buffers = (runtime.Buffer * 1)(
        runtime.Buffer(b"out", dtype.encode(), ctypes.addressof(memory), 3, 1)
    )
    launch = runtime.Launch((3, 1, 1), (1, 1, 1))
    if fault == "geometry":
        launch.workgroup_size[0] = 32
    elif fault == "exact-grid":
        launch.thread_grid_size[0] = 3
        launch.thread_grid_size[1] = launch.thread_grid_size[2] = 1
    elif fault == "buffer-count":
        buffers[0].count = 2
    elif fault == "dtype":
        buffers[0].dtype = b"int64"
    elif fault == "null":
        buffers[0].data = None
    elif fault == "direction":
        buffers[0].output = 0
    elif fault == "name":
        buffers[0].name = b"in"
    elif fault == "missing":
        host.reduction_descriptors.clear()
    descriptor = index["descriptors"][f"w1/{entry}"]
    name = descriptor["bindings"][0]["name"]
    guard = proof.guard_values(dtype, index["target"])
    storage = runtime.physical_dtype(dtype, index["target"])
    value = int("prod" in entry or "and" in entry)
    if storage == "bool":
        value = bool(value)
    returned = [value] * 3 + guard
    if fault == "guard":
        returned[-1] = not returned[-1] if storage == "bool" else int(not returned[-1])
    elif fault == "readback":
        returned.pop()
    calls = []

    def execute(request):
        calls.append(request)
        initial = next(item for item in request.fixture.inputs if item.name == name)
        assert initial.values[:3] == [1 - int(value)] * 3
        return SimpleNamespace(
            status="ok",
            details={},
            outputs={
                name: {"dtype": storage, "shape": [3 + len(guard)], "values": returned}
            },
        )

    host.executor = SimpleNamespace(run=execute)
    if fault:
        with pytest.raises(
            RuntimeError if fault in {"guard", "readback"} else ValueError
        ):
            host.dispatch(
                entry,
                buffers,
                0 if fault == "count" else 1,
                3,
                launch=None if fault == "launch" else launch,
            )
        assert bool(calls) == (fault in {"guard", "readback"})
        assert not host.trace.exists()
        assert list(memory) == [0, 0, 0]
    else:
        host.dispatch(entry, buffers, 1, 3, launch=launch)
        assert list(memory) == [value] * 3
        event = json.loads(host.trace.read_text())
        assert event["reductionMetadata"] == {"outputSize": 3}
        assert event["reductionValues"] == [value] * 3
        assert event["reductionGuardValues"] == guard
        assert event["initializationValue"] == 1 - int(value)
        assert calls[0].execution_plan.dispatch.workgroup_size == (1, 1, 1)


def evidence(native, target):
    records, trace = [], []
    for case in proof.cases():
        data, expected, dtype = proof.reference(np, case)
        dispatch = bool(native and expected.size)
        records.append(
            {
                **case,
                "actual": expected.tolist(),
                "expected": expected.tolist(),
                "resultShape": list(expected.shape),
                "resultDtype": "mlx.core." + dtype.removesuffix("_"),
                "inputHash": hashlib.sha256(data.tobytes()).hexdigest(),
                "inputUnchanged": True,
                "inputConstruction": "numpy" if native else "contiguous-empty",
                "dispatchCount": int(dispatch),
            }
        )
        if dispatch:
            operation = {"all": "and", "any": "or"}.get(
                case["operation"], case["operation"]
            )
            trace.append(
                {
                    "entry": f"init_reduce_{operation}{dtype}",
                    "target": target,
                    "dispatchVersion": runtime.DISPATCH_VERSION,
                    "reductionValues": expected.reshape(-1).tolist(),
                    "reductionGuardValues": proof.guard_values(dtype, target),
                    "reductionMetadata": {"outputSize": expected.size},
                    "threads": expected.size,
                    "initializationValue": int(case["operation"] in {"sum", "any"}),
                    "workgroupCount": [expected.size, 1, 1],
                    "workgroupSize": [1, 1, 1],
                }
            )
    return records, trace


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
def test_empty_evidence_covers_all_entries_shapes_and_promotions(native, target):
    records, trace = evidence(native, target)
    proof.validate(records, trace, native=native)
    assert len(records) == 68
    if native:
        assert len(trace) == 50
        assert {row["entry"] for row in trace} == set(reduction_packages.INIT_ENTRIES)
    assert any(
        row["resultDtype"] == "mlx.core.int32" and row["dtype"] == "bool_"
        for row in records
    )
    assert any(row["resultShape"] == [65535] for row in records)


@pytest.mark.parametrize(
    "field",
    [
        "actual",
        "expected",
        "resultShape",
        "resultDtype",
        "inputHash",
        "inputUnchanged",
        "inputConstruction",
        "dispatchCount",
    ],
)
def test_incomplete_result_evidence_is_rejected(field):
    records, trace = evidence(True, "metal")
    records[0].pop(field)
    with pytest.raises(ValueError):
        proof.validate(records, trace, native=True)


@pytest.mark.parametrize(
    "field",
    [
        "entry",
        "target",
        "dispatchVersion",
        "reductionValues",
        "reductionGuardValues",
        "reductionMetadata",
        "initializationValue",
        "threads",
        "workgroupCount",
        "workgroupSize",
    ],
)
def test_incomplete_native_evidence_is_rejected(field):
    records, trace = evidence(True, "opengl")
    trace[0].pop(field)
    with pytest.raises(ValueError):
        proof.validate(records, trace, native=True)


@pytest.mark.parametrize(
    "fault",
    [
        "missing-case",
        "missing-trace",
        "extra-trace",
        "zero-output-dispatch",
        "cpu-dispatch",
        "padded",
        "identity",
    ],
)
def test_misleading_completion_evidence_is_rejected(fault):
    records, trace = evidence(True, "directx")
    if fault == "missing-case":
        records.pop()
    elif fault == "missing-trace":
        trace.pop()
    elif fault == "extra-trace":
        trace.append(trace[0])
    elif fault == "zero-output-dispatch":
        records[3]["dispatchCount"] = 1
    elif fault == "cpu-dispatch":
        records, _ = evidence(False, "directx")
    elif fault == "padded":
        trace[0]["threadGridSize"] = [1, 1, 1]
    elif fault == "identity":
        records[0]["actual"] = records[0]["expected"] = 1.0
    with pytest.raises(ValueError):
        proof.validate(records, trace, native=fault != "cpu-dispatch")


@pytest.mark.parametrize("widths", [[], [32], [1, 1], [True], [1.0]])
def test_init_width_is_exact_before_checkout(tmp_path, widths):
    with pytest.raises(ValueError, match="widths"):
        reduction_packages.build_packages(
            tmp_path, tmp_path / "out", "metal", family="init", widths=widths
        )


def test_ci_requires_initialization_on_every_native_target():
    import yaml

    workflow = yaml.safe_load(
        Path(".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["small-row-reductions"]
    assert {item["target"] for item in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "opengl",
        "directx",
    }
    build = next(
        step
        for step in job["steps"]
        if step.get("name") == "Translate reduction initialization"
    )
    execute = next(
        step
        for step in job["steps"]
        if step.get("name") == "Execute empty host reductions"
    )
    for step in (build, execute):
        assert "if" not in step and "continue-on-error" not in step
        assert "run_bounded_command.py" in step["run"]
    assert "--family init" in build["run"]
    assert "verify_empty_reductions" in execute["run"]
    assert "--reductions .mlx-portable-small-rows/init-packages" in execute["run"]
    for event in ("pull_request", "push"):
        assert_paths_covered(
            workflow.get("on", workflow.get(True))[event]["paths"],
            "demos/integrations/mlx/tests/host/test_portable_empty_reductions.py",
        )


@pytest.mark.parametrize("failure_mode", ["cpu", "native", *proof.NEGATIVE_CHECKS])
def test_worker_timeouts_preserve_failure_without_completion(
    tmp_path, monkeypatch, failure_mode
):
    args = SimpleNamespace(
        output_dir=tmp_path / "evidence",
        mlx_root=tmp_path,
        packages=tmp_path / "packages",
        reductions=tmp_path / "reductions",
    )
    args.packages.mkdir()
    (args.packages / "index.json").write_text(json.dumps({"target": "metal"}))
    monkeypatch.setattr(proof, "verify_prepared", lambda root: {})
    monkeypatch.setattr(
        proof,
        "load_index",
        lambda *args: {
            "family": "init",
            "entries": list(reduction_packages.INIT_ENTRIES),
        },
    )
    modes = ["cpu", "native", *proof.NEGATIVE_CHECKS]
    calls = []

    def execute(command, **kwargs):
        mode = command[command.index("--worker") + 1]
        calls.append(mode)
        assert Path(command[1]).name == "run_bounded_command.py"
        assert command[command.index("--timeout-seconds") + 1] == (
            "600" if mode == "native" else "90"
        )
        destination = args.output_dir / mode
        destination.mkdir()
        (destination / "results.json").write_text("[]")
        (destination / "upstream.json").write_text(
            json.dumps(
                {
                    "test": proof.UPSTREAM_TEST,
                    "testsRun": 1,
                    "failures": 0,
                    "errors": 0,
                    "skips": 0,
                    "dispatchCount": 0,
                }
            )
        )
        if mode in proof.NEGATIVE_CHECKS:
            (destination / "rejection.json").write_text(
                json.dumps(
                    {
                        "check": mode,
                        "error": proof.NEGATIVE_CHECKS[mode],
                        "dispatchCount": 0,
                    }
                )
            )
        return SimpleNamespace(returncode=124 if mode == failure_mode else 0)

    monkeypatch.setattr(proof.subprocess, "run", execute)
    with pytest.raises(RuntimeError, match=f"{failure_mode} worker failed"):
        proof.verify(args)
    assert calls == modes[: modes.index(failure_mode) + 1]
    assert (
        json.loads((args.output_dir / f"{failure_mode}.command.json").read_text())[
            "returncode"
        ]
        == 124
    )
    assert not (args.output_dir / "evidence.json").exists()
