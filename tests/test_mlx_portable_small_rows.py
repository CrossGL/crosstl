"""Exact launch and cache contracts for unchanged MLX small-row kernels."""

import ctypes
import hashlib
import json
import math
import re
import struct
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    row_reduction_layout,
    runtime,
    small_row_packages,
    verify_small_rows,
)

ENTRY = "row_reduce_small_1_reduce_sumfloat32"


def buffers(rows=1031, width=7, nonrows=1):
    data = {
        "in": ("float32", [1.0] * (rows * width * nonrows)),
        "out": ("float32", [-1.0] * rows),
        "row_size": ("int64", [width]),
        "non_row_reductions": ("int64", [nonrows]),
        "shape": ("int32", [rows]),
        "strides": ("int64", [width]),
        "ndim": ("int32", [1]),
        "reduce_shape": ("int32", [nonrows] if nonrows > 1 else [0]),
        "reduce_strides": ("int64", [rows * width] if nonrows > 1 else [0]),
        "reduce_ndim": ("int32", [int(nonrows > 1)]),
    }
    memory = {
        name: (runtime.TYPES[dtype] * len(values))(*values)
        for name, (dtype, values) in data.items()
    }
    supplied = {
        name: runtime.Buffer(
            name.encode(),
            dtype.encode(),
            ctypes.addressof(memory[name]),
            len(values),
            int(name == "out"),
        )
        for name, (dtype, values) in data.items()
    }
    return supplied, memory, rows * width * nonrows


@pytest.mark.parametrize(
    "width,nonrows,scalar",
    [
        (8, 31, True),
        (8, 32, False),
        (9, 8, True),
        (9, 9, False),
        (64, 8, True),
        (64, 9, False),
    ],
)
@pytest.mark.parametrize("rows", [1, 33, 1031])
def test_small_row_source_branch_and_exact_geometry(width, nonrows, scalar, rows):
    if rows * width * nonrows > 65535:
        rows = 33
    supplied, memory, logical = buffers(rows, width, nonrows)
    size = min(rows, 1024) if scalar else 32
    grid = [(rows + size - 1) // size, 1, 1] if scalar else [1, rows, 1]
    exact = [rows, 1, 1] if scalar else [32, rows, 1]
    execution = runtime.Launch(tuple(grid), (size, 1, 1), tuple(exact)).execution()
    assert row_reduction_layout.validate(ENTRY, supplied, logical, execution)
    execution.pop("threadGridSize")
    with pytest.raises(ValueError, match="launch"):
        row_reduction_layout.validate(ENTRY, supplied, logical, execution)


@pytest.mark.parametrize(
    "fault", ["tail", "rounded", "span", "rank", "size", "logical"]
)
def test_small_row_invalid_metadata_is_rejected(fault):
    supplied, memory, logical = buffers()
    execution = {
        "workgroupCount": [2, 1, 1],
        "workgroupSize": [1024, 1, 1],
        "threadGridSize": [1031, 1, 1],
    }
    if fault == "tail":
        execution["workgroupSize"] = [7, 1, 1]
    elif fault == "rounded":
        execution["threadGridSize"] = [2048, 1, 1]
    elif fault == "span":
        supplied["in"].count -= 1
    elif fault == "rank":
        memory["reduce_ndim"][0] = 2
    elif fault == "size":
        memory["row_size"][0] = 65
    else:
        logical -= 1
    with pytest.raises(ValueError):
        row_reduction_layout.validate(ENTRY, supplied, logical, execution)


@pytest.mark.parametrize(
    "exact", [(0, 1, 1), (1031, 0, 1), (0, 0, 1), (1024, 1, 1), (2049, 1, 1)]
)
def test_callback_rejects_inconsistent_exact_grids(exact):
    with pytest.raises(ValueError, match="exact grid"):
        runtime.Launch((2, 1, 1), (1024, 1, 1), exact).execution()


def test_callback_version_and_zero_grid_compatibility():
    header = Path(runtime.__file__).with_name("dispatch.h").read_text()
    assert f"CROSTL_MLX_DISPATCH_VERSION = {runtime.DISPATCH_VERSION}" in header
    assert runtime.DISPATCH_VERSION == 3
    assert ctypes.sizeof(runtime.Launch) == 36
    assert runtime.Launch((2, 1, 1), (32, 1, 1)).execution() == {
        "workgroupCount": [2, 1, 1],
        "workgroupSize": [32, 1, 1],
    }


@pytest.mark.parametrize("failure_mode", ["cpu", "native"])
def test_small_row_workers_use_process_tree_deadlines(
    tmp_path, monkeypatch, failure_mode
):
    args = SimpleNamespace(
        output_dir=tmp_path / "evidence",
        packages=tmp_path / "packages",
        mlx_root=tmp_path / "source",
    )
    monkeypatch.setattr(verify_small_rows, "verify_prepared", lambda root: {})
    calls = []

    def execute(command, **kwargs):
        mode = command[command.index("--worker") + 1]
        calls.append(mode)
        assert Path(command[1]).name == "run_bounded_command.py"
        assert command[command.index("--timeout-seconds") + 1] == (
            "180" if mode == "cpu" else "3600"
        )
        assert kwargs["check"] is False
        destination = args.output_dir / mode
        destination.mkdir()
        (destination / "results.json").write_text("[]")
        return SimpleNamespace(returncode=124 if mode == failure_mode else 0)

    monkeypatch.setattr(verify_small_rows.subprocess, "run", execute)
    with pytest.raises(RuntimeError, match=f"Small-row {failure_mode} worker failed"):
        verify_small_rows.verify(args)
    assert calls == (["cpu"] if failure_mode == "cpu" else ["cpu", "native"])
    command = json.loads((args.output_dir / f"{failure_mode}.command.json").read_text())
    assert command["returncode"] == 124
    assert not (args.output_dir / "evidence.json").exists()


@pytest.fixture
def cache_fixture(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        small_row_packages.SmallRowPackageCache, "_require_source", lambda self: None
    )
    monkeypatch.setattr(
        small_row_packages, "translation_implementation_hash", lambda: "a" * 64
    )
    monkeypatch.setattr(
        small_row_packages,
        "select_native_loader_dispatch_regions",
        lambda packages, **kwargs: tuple(packages),
    )

    def build(self, entry, output, identity, region):
        calls.append(identity)
        (output / "index.json").write_text(json.dumps(identity))

    def load(self, directory, identity):
        assert json.loads((directory / "index.json").read_text()) == identity
        return identity, directory

    monkeypatch.setattr(small_row_packages.SmallRowPackageCache, "_build", build)
    monkeypatch.setattr(small_row_packages.SmallRowPackageCache, "_load", load)
    return (
        small_row_packages.SmallRowPackageCache(tmp_path, tmp_path / "cache", "opengl"),
        calls,
    )


def test_cache_distinguishes_complete_grid_and_translator(cache_fixture, monkeypatch):
    cache, calls = cache_fixture
    first = cache.get(ENTRY, thread_grid_size=[1031], workgroup_size=[1024])
    assert len(first) == len(calls) == 2
    assert cache.get(ENTRY, thread_grid_size=[1031], workgroup_size=[1024]) == first
    assert len(calls) == 2
    second = cache.get(ENTRY, thread_grid_size=[2055], workgroup_size=[1024])
    assert {path for _, path in first}.isdisjoint(path for _, path in second)
    assert len(calls) == 4
    monkeypatch.setattr(
        small_row_packages, "translation_implementation_hash", lambda: "b" * 64
    )
    cache.get(ENTRY, thread_grid_size=[1031], workgroup_size=[1024])
    assert len(calls) == 6


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_cache_rechecks_source_on_hit(cache_fixture, monkeypatch, target):
    cache, calls = cache_fixture
    cache.target = target
    cache.get(ENTRY, thread_grid_size=[1031], workgroup_size=[1024])
    assert len(calls) == (1 if target == "metal" else 2)

    def reject():
        raise ValueError("Changed source")

    monkeypatch.setattr(cache, "_require_source", reject)
    with pytest.raises(ValueError, match="Changed source"):
        cache.get(ENTRY, thread_grid_size=[1031], workgroup_size=[1024])


def test_cache_does_not_publish_mixed_translator_build(cache_fixture, monkeypatch):
    cache, calls = cache_fixture
    hashes = iter(["a" * 64, "b" * 64])
    monkeypatch.setattr(
        small_row_packages, "translation_implementation_hash", lambda: next(hashes)
    )
    with pytest.raises(ValueError, match="Translator changed"):
        cache.get(ENTRY, thread_grid_size=[1031], workgroup_size=[1024])
    assert not list(cache.directory.iterdir())


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault", [None, "metadata", "guard", "missing-source", "compile"]
)
def test_small_row_host_dispatch_checks_before_writeback(
    tmp_path, monkeypatch, target, fault
):
    supplied, memory, logical = buffers()
    host = runtime.HostRuntime.__new__(runtime.HostRuntime)
    host.target, host.descriptors, host.trace = target, {}, tmp_path / "trace.jsonl"
    bindings = []
    for name, buffer in supplied.items():
        member = {"in": "in_", "out": "out_"}.get(name, name)
        if target == "directx":
            member = ENTRY + "_" + member
        dtype = buffer.dtype.decode()
        bindings.append(
            {
                "name": name,
                "scalarLayout": {
                    "memberName": member,
                    "elementType": dtype,
                    "elementStrideBytes": ctypes.sizeof(runtime.TYPES[dtype]),
                },
            }
        )
    descriptor = {"bindings": bindings, "artifact": {}, "provenance": {}}
    packages = [(descriptor, tmp_path)] * (1 if target == "metal" else 2)
    calls = []

    def get(entry, **kwargs):
        calls.append("packages")
        assert entry == ENTRY and kwargs == {
            "thread_grid_size": [1031, 1, 1],
            "workgroup_size": [1024, 1, 1],
        }
        return packages

    host.small_rows = None if fault == "missing-source" else SimpleNamespace(get=get)

    def execute(outputs):
        calls.append("execute")
        outputs["out"]["values"][:1031] = [7.0] * 1031
        if fault == "guard":
            outputs["out"]["values"][-1] = 0
        return outputs

    def prepare_request(*args, **kwargs):
        if fault == "compile":
            raise ValueError("compile failed")
        return args[3]

    @contextmanager
    def prepare_regions(packages, inputs, outputs, **kwargs):
        if fault == "compile":
            raise ValueError("compile failed")
        module = tmp_path / "native-module"
        module.write_bytes(b"test")
        native.outputs = outputs
        yield [
            SimpleNamespace(
                module_path=module,
                dispatch=SimpleNamespace(
                    workgroup_count=(1, 1, 1), workgroup_size=(width, 1, 1)
                ),
            )
            for width in (1024, 7)
        ]

    native = SimpleNamespace(
        name="test", dispatch_sequence=lambda a, b, r: execute(native.outputs)
    )
    host.executor = SimpleNamespace(
        runtime_adapter=SimpleNamespace(runtime=native),
        run=lambda outputs: SimpleNamespace(
            status="ok", outputs=execute(outputs), details={}
        ),
    )
    monkeypatch.setattr(
        runtime, "build_native_loader_dispatch_request", prepare_request
    )
    monkeypatch.setattr(
        runtime, "prepare_native_loader_dispatch_regions", prepare_regions
    )
    if fault == "metadata":
        memory["row_size"][0] = 65
    array = (runtime.Buffer * len(supplied))(*supplied.values())
    launch = runtime.Launch((2, 1, 1), (1024, 1, 1), (1031, 1, 1))
    if fault:
        with pytest.raises(RuntimeError if fault == "guard" else ValueError):
            host.dispatch(ENTRY, array, len(array), logical, launch=launch)
        assert list(memory["out"]) == [-1.0] * 1031
        assert not host.trace.exists()
        assert ("execute" in calls) == (fault == "guard")
        if fault in {"metadata", "missing-source"}:
            assert not calls
    else:
        host.dispatch(ENTRY, array, len(array), logical, launch=launch)
        assert list(memory["out"]) == [7.0] * 1031
        record = json.loads(host.trace.read_text())
        assert record["threadGridSize"] == [1031, 1, 1]
        if target != "metal":
            assert len(record["details"]["regions"]) == 2


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "count",
        "result",
        "input",
        "dtype",
        "shape",
        "modified-input",
        "input-flag",
        "version",
        "guard",
        "rounded",
        "groups",
        "entry",
        "template-rank",
    ],
)
def test_small_row_evidence_rejects_incomplete_or_changed_results(fault):
    import numpy as np

    records, trace = [], []
    for case in verify_small_rows.cases():
        data, expected = verify_small_rows.reference(np, case)
        record = {
            **case,
            "actual": expected.tolist(),
            "expected": expected.tolist(),
            "resultDtype": "mlx.core." + case["dtype"].removesuffix("_"),
            "resultShape": list(expected.shape),
            "inputUnchanged": True,
            "dispatchCount": 1,
            "inputHash": hashlib.sha256(data.tobytes()).hexdigest(),
        }
        records.append(record)
        row = case["shape"][-1]
        rank = len(case["axes"]) - 1
        dimension = 1 if rank <= 1 else 2 if rank == 2 else 5
        nonrows = math.prod(case["shape"][axis] for axis in case["axes"][:-1])
        scalar = (nonrows < 32 and row <= 8) or nonrows <= 8
        count = expected.size
        group = min(count, 1024) if scalar else 32
        guard = (
            runtime.BOOLEAN_GUARD
            if case["dtype"] == "bool_"
            else (
                [
                    struct.unpack("<f", struct.pack("<I", v))[0]
                    for v in runtime.COPY_GUARD
                ]
                if case["dtype"] == "float32"
                else runtime.COPY_GUARD
            )
        )
        trace.append(
            {
                "target": "metal",
                "entry": (
                    f"row_reduce_small_{dimension}_reduce_"
                    + case["operation"]
                    + case["dtype"]
                ),
                "threads": int(data.size),
                "dispatchVersion": runtime.DISPATCH_VERSION,
                "reductionValues": expected.reshape(-1).tolist(),
                "reductionGuardValues": guard,
                "threadGridSize": [int(count), 1, 1] if scalar else [32, int(count), 1],
                "workgroupCount": (
                    [int((count + group - 1) // group), 1, 1]
                    if scalar
                    else [1, int(count), 1]
                ),
                "workgroupSize": [int(group), 1, 1],
            }
        )
    assert len(records) == 34
    if fault == "missing":
        records.pop()
    elif fault == "count":
        records[0]["dispatchCount"] = 0
    elif fault == "result":
        records[0]["actual"] = []
    elif fault == "input":
        records[0]["inputHash"] = "a" * 64
    elif fault == "dtype":
        records[0]["resultDtype"] = "mlx.core.int64"
    elif fault == "shape":
        records[0]["resultShape"] = [1, 1031]
    elif fault == "modified-input":
        records[0]["inputUnchanged"] = False
    elif fault == "input-flag":
        records[0]["inputUnchanged"] = 1
    elif fault == "version":
        trace[0]["dispatchVersion"] = float(runtime.DISPATCH_VERSION)
    elif fault == "guard":
        trace[0]["reductionGuardValues"] = [0] * 32
    elif fault == "rounded":
        trace[0]["threadGridSize"][0] = 2048
    elif fault == "groups":
        trace[0]["workgroupCount"][0] = 1
    elif fault == "entry":
        trace[0]["entry"] = trace[1]["entry"].replace("sum", "prod")
    elif fault == "template-rank":
        trace[-1]["entry"] = trace[-1]["entry"].replace("_5_", "_2_")
    if fault:
        with pytest.raises(ValueError):
            verify_small_rows.validate(records, trace, native=True)
    else:
        verify_small_rows.validate(records, trace, native=True)


def test_small_row_ci_requires_native_execution_on_all_targets():
    import yaml

    workflow = yaml.safe_load(
        Path(".github/workflows/mlx-portable-host.yml").read_text()
    )
    jobs = [
        job
        for job in workflow["jobs"].values()
        if any(
            step.get("name") == "Execute small-row host reductions"
            for step in job.get("steps", [])
        )
    ]
    assert len(jobs) == 1
    job = jobs[0]
    assert job is workflow["jobs"]["small-row-reductions"]
    assert job["needs"] == "portable-host"
    assert "if" not in job and "continue-on-error" not in job
    assert {
        item["target"]: item["os"] for item in job["strategy"]["matrix"]["include"]
    } == {
        "directx": "windows-2025",
        "opengl": "ubuntu-24.04",
        "metal": "macos-26",
    }
    deadlines = [
        int(seconds)
        for step in job["steps"]
        for seconds in re.findall(r"--timeout-seconds (\d+)", step.get("run", ""))
    ]
    assert deadlines == [1800, 4000, 900, 1200, 600, 1500, 600, 1500, 600, 600, 1800]
    assert sum(deadlines) + 1800 < job["timeout-minutes"] * 60 <= 360 * 60
    triggers = workflow.get("on", workflow.get(True))
    for event in ("pull_request", "push"):
        assert "tests/test_mlx_portable_small_rows.py" in triggers[event]["paths"]
    step = next(
        step
        for step in job["steps"]
        if step.get("name") == "Execute small-row host reductions"
    )
    assert "continue-on-error" not in step and "if" not in step
    assert "portable_host.verify_small_rows" in step["run"]
    assert "--mlx-root mlx-upstream" in step["run"]
    retained = next(
        step
        for step in job["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert retained["if"] == "always()"
    assert retained["with"]["include-hidden-files"] is True
    assert retained["with"]["if-no-files-found"] == "error"


@pytest.mark.parametrize("fault", ["value", "dtype", "shape", "input"])
def test_small_row_worker_retains_failed_readbacks(tmp_path, monkeypatch, fault):
    import numpy as np

    case = next(verify_small_rows.cases())
    data, expected = verify_small_rows.reference(np, case)

    def reduce(source, *, axis):
        result = np.sum(source, axis=tuple(axis), dtype=source.dtype)
        if fault == "value":
            result[0] += 1
        elif fault == "dtype":
            result = result.astype("int64")
        elif fault == "shape":
            result = result.reshape(1, -1)
        elif fault == "input":
            source.flat[0] += 1
        return result

    core = SimpleNamespace(
        cpu="cpu",
        gpu="gpu",
        is_available=lambda device: False,
        set_default_device=lambda device: None,
        array=np.copy,
        float32=np.dtype("float32"),
        sum=reduce,
    )
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(verify_small_rows, "cases", lambda: iter([case]))
    args = SimpleNamespace(output_dir=tmp_path / "evidence", worker="cpu")
    with pytest.raises(
        RuntimeError, match="Small-row (numerical mismatch|input changed)"
    ):
        verify_small_rows.worker(args)
    records = json.loads((args.output_dir / "results.json").read_text())
    assert len(records) == 1
    record = records[0]
    assert record["id"] == case["id"]
    assert record["expected"] == expected.tolist()
    assert record["inputHash"] == hashlib.sha256(data.tobytes()).hexdigest()
    assert record["inputUnchanged"] is (fault != "input")
    if fault == "value":
        assert record["actual"][0] == expected[0] + 1
    elif fault == "dtype":
        assert record["resultDtype"] == "int64"
    elif fault == "shape":
        assert record["resultShape"] == [1, expected.size]


def test_native_host_ci_preserves_active_execution():
    import yaml

    workflow = yaml.safe_load(
        Path(".github/workflows/mlx-portable-host.yml").read_text()
    )
    concurrency = workflow["concurrency"]
    assert concurrency["group"] == "mlx-portable-host-${{ github.ref }}"
    assert concurrency["cancel-in-progress"] is False
    assert concurrency.get("queue", "single") == "single"


def test_small_row_oracle_distinguishes_outputs_for_every_operation():
    import numpy as np

    for case in verify_small_rows.cases():
        _, expected = verify_small_rows.reference(np, case)
        assert len(np.unique(expected)) > 1, case["id"]


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "identity",
        "package",
        "descriptor",
        "target",
        "region",
        "entry",
        "implementation",
    ],
)
def test_cache_load_revalidates_package_and_program(tmp_path, monkeypatch, fault):
    from copy import deepcopy

    from crosstl.translator.dispatch_regions import plan_dispatch_regions

    region = plan_dispatch_regions([1031], [1024])[0].to_json()
    identity = {"entry": ENTRY, "region": region, "implementationHash": "a" * 64}
    descriptor = {
        "target": "opengl",
        "provenance": {
            "dispatchRegion": region,
            "dispatchRegionProgram": {
                "sourceEntryPoint": ENTRY,
                "implementationHash": "a" * 64,
            },
        },
    }
    saved_identity, saved_descriptor = deepcopy(identity), deepcopy(descriptor)
    if fault == "identity":
        saved_identity["entry"] = "different"
    if fault == "descriptor":
        saved_descriptor["provenance"] = {}
    if fault == "target":
        descriptor["target"] = saved_descriptor["target"] = "metal"
    if fault in {"region", "entry", "implementation"}:
        if fault == "region":
            descriptor["provenance"]["dispatchRegion"] = None
        else:
            key = "sourceEntryPoint" if fault == "entry" else "implementationHash"
            descriptor["provenance"]["dispatchRegionProgram"][key] = "changed"
        saved_descriptor = deepcopy(descriptor)
    (tmp_path / "index.json").write_text(
        json.dumps({"identity": saved_identity, "descriptor": saved_descriptor})
    )
    monkeypatch.setattr(
        small_row_packages,
        "build_runtime_loader_manifest",
        lambda path: {
            "success": fault != "package",
            "loadUnits": [{}],
            "diagnostics": [],
        },
    )
    monkeypatch.setattr(
        small_row_packages,
        "build_native_loader_abi_descriptor",
        lambda loader: descriptor,
    )
    cache = small_row_packages.SmallRowPackageCache(tmp_path, tmp_path, "opengl")
    if fault:
        with pytest.raises(ValueError):
            cache._load(tmp_path, identity)
    else:
        assert cache._load(tmp_path, identity) == (descriptor, tmp_path / "package")


@pytest.mark.parametrize("dirty", [False, True])
def test_cache_requires_pinned_revision_and_clean_kernel_tree(
    tmp_path, monkeypatch, dirty
):
    calls = []
    monkeypatch.setattr(
        small_row_packages, "require_revision", lambda root: calls.append(root)
    )
    monkeypatch.setattr(
        small_row_packages.subprocess,
        "check_output",
        lambda command, **kwargs: (
            " M mlx/backend/metal/kernels/reduction/reduce_row.h" if dirty else ""
        ),
    )
    cache = small_row_packages.SmallRowPackageCache(
        tmp_path, tmp_path / "cache", "metal"
    )
    if dirty:
        with pytest.raises(ValueError, match="unchanged pinned kernels"):
            cache._require_source()
    else:
        cache._require_source()
    assert calls == [tmp_path]


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("fault", [None, "artifact", "index"])
def test_cache_real_package_roundtrip_and_tampering(
    tmp_path, monkeypatch, target, fault
):
    source = tmp_path / small_row_packages.SOURCE
    source.parent.mkdir(parents=True)
    source.write_text(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "template <typename T>\n[[kernel]] void small_row(device T* out [[buffer(0)]], "
        "uint tid [[thread_position_in_grid]]) { out[tid] = T(simd_sum(tid)); }\n"
        f'instantiate_kernel("{ENTRY}", small_row, float)\n'
    )
    # This checks the package boundary with a synthetic kernel, not MLX coverage.
    monkeypatch.setattr(
        small_row_packages.SmallRowPackageCache, "_require_source", lambda self: None
    )
    cache = small_row_packages.SmallRowPackageCache(
        tmp_path, tmp_path / "cache", target
    )
    packages = cache.get(ENTRY, thread_grid_size=[37], workgroup_size=[32])
    assert len(packages) == (1 if target == "metal" else 2)
    assert cache.get(ENTRY, thread_grid_size=[37], workgroup_size=[32]) == packages
    descriptor, directory = packages[-1]
    if fault == "artifact":
        path = directory / descriptor["artifact"]["packagePath"]
        path.write_text(path.read_text() + "\n// changed artifact\n")
    elif fault == "index":
        path = directory.parent / "index.json"
        record = json.loads(path.read_text())
        record["identity"]["entry"] = "different"
        path.write_text(json.dumps(record))
    if fault:
        with pytest.raises(ValueError):
            cache.get(ENTRY, thread_grid_size=[37], workgroup_size=[32])
