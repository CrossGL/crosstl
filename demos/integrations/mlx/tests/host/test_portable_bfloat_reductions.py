"""Bfloat reduction plans, inventory and numerical evidence must remain exact."""

import ctypes
import json
from pathlib import Path

import pytest
import yaml

from demos.integrations.mlx.portable_host import bfloat_storage, reduction_layout
from demos.integrations.mlx.portable_host import reduction_packages as packages
from demos.integrations.mlx.portable_host import runtime
from demos.integrations.mlx.portable_host import verify_bfloat_reductions as proof
from demos.integrations.mlx.portable_host.runtime import physical_dtype
from demos.integrations.mlx.portable_host.small_row_packages import SMALL_ROW_ENTRIES


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("subnormal_input", [False, True])
def test_reduction_subnormal_guard_reads_only_inputs(
    tmp_path, monkeypatch, target, subnormal_input
):
    entry = "all_reduce_maxbfloat16"
    host = runtime.HostRuntime.__new__(runtime.HostRuntime)
    host.target, host.descriptors, host.retain_native_modules = target, {}, False
    memory = [
        (ctypes.c_uint16 * 2)(1 if subnormal_input else 0x3F80, 0x4000),
        (ctypes.c_uint16 * 1)(1),
        ctypes.c_uint64(2),
        ctypes.c_uint64(2),
    ]
    buffers = (runtime.Buffer * 4)(
        *[
            runtime.Buffer(
                name.encode(),
                dtype.encode(),
                ctypes.addressof(value),
                count,
                int(name == "out"),
            )
            for name, dtype, count, value in zip(
                ("in", "out", "in_size", "row_size"),
                ("bfloat16", "bfloat16", "uint64", "uint64"),
                (2, 1, 1, 1),
                memory,
            )
        ]
    )
    bindings = []
    for buffer in buffers:
        name = buffer.name.decode()
        member = {"in": "in_", "out": "out_"}.get(name, name)
        physical = runtime.physical_dtype(buffer.dtype.decode(), target)
        bindings.append(
            {
                "name": name,
                "kind": "buffer" if name in {"in", "out"} else "constant-buffer",
                "scalarLayout": {
                    "memberName": (entry + "_" if target == "directx" else "") + member,
                    "elementType": physical,
                    "elementStrideBytes": ctypes.sizeof(runtime.TYPES[physical]),
                    "elementSizeBytes": ctypes.sizeof(runtime.TYPES[physical]),
                },
            }
        )
    key = "w32/" + entry
    host.reduction_descriptors = {key: {"bindings": bindings}}
    host.reduction_directories = {key: tmp_path}

    class ReachedDispatch(Exception):
        pass

    def request(*args, **kwargs):
        raise ReachedDispatch

    monkeypatch.setattr(runtime, "build_native_loader_dispatch_request", request)
    expected = ValueError if subnormal_input and target != "metal" else ReachedDispatch
    with pytest.raises(expected):
        host.dispatch(
            entry, buffers, 4, 2, launch=runtime.Launch((1, 1, 1), (32, 1, 1))
        )
    assert memory[1][0] == 1


def test_bfloat_reductions_are_opt_in_and_keep_unsupported_arithmetic_out():
    assert set(packages.BFLOAT_ENTRIES) == {
        "all_reduce_minbfloat16",
        "all_reduce_maxbfloat16",
    }
    assert packages.ENTRY_GROUPS["bfloat"] == packages.BFLOAT_ENTRIES
    assert not packages.BFLOAT_ENTRIES.keys() & packages.ENTRIES.keys()
    assert set(packages.ALL_ENTRIES) == set(packages.ENTRIES) | set(
        packages.BFLOAT_ENTRIES
    )
    assert all(
        "bfloat" not in entry
        for entry in (
            packages.ROW_ENTRIES | packages.COLUMN_ENTRIES | packages.INIT_ENTRIES
        )
    )
    assert {entry for entry in SMALL_ROW_ENTRIES if "bfloat" in entry} == {
        f"row_reduce_small_{rank}_reduce_{operation}bfloat16"
        for rank in (1, 2, 5)
        for operation in ("min", "max")
    }


@pytest.mark.parametrize("fault", [None, "entry", "family", "width"])
def test_bfloat_package_index_rejects_mixed_families(tmp_path, fault):
    entries = list(packages.BFLOAT_ENTRIES)
    index = {
        "family": "bfloat",
        "target": "opengl",
        "entries": entries,
        "widths": [32],
        "descriptors": {f"w32/{entry}": {"target": "opengl"} for entry in entries},
    }
    if fault == "entry":
        index["entries"][0] = "all_reduce_sumbfloat16"
    elif fault == "family":
        index["family"] = "all"
    elif fault == "width":
        index["widths"] = [33]
    (tmp_path / "index.json").write_text(json.dumps(index))
    if fault:
        with pytest.raises(ValueError):
            packages.load_index(tmp_path, "opengl")
    else:
        assert packages.load_index(tmp_path, "opengl") == index


def evidence(target, native):
    records, trace = [], []
    for case in proof.cases():
        source, expected = proof.reference(case)
        whole = len(case["shape"]) == 1
        dispatches = (2 if whole and len(source) > 4096 else 1) if native else 0
        records.append(
            {
                **case,
                "actualWords": expected,
                "inputWords": source,
                "dtype": "mlx.core.bfloat16",
                "shapeOut": [] if whole else [len(expected)],
                "dispatchCount": dispatches,
            }
        )
        values = source
        for _ in range(dispatches):
            packet = {
                "dtype": physical_dtype("bfloat16", target),
                "shape": [len(values)],
                "values": bfloat_storage.pack(values, target),
            }
            if bfloat_storage.encoding(target):
                packet["encoding"] = bfloat_storage.encoding(target)
            if whole:
                plan = reduction_layout.stage(len(values), "bfloat16")
                row = plan["rowSize"]
                event = {
                    "entry": f'all_reduce_{case["operation"]}bfloat16',
                    "threads": len(values),
                    "workgroupCount": plan["workgroupCount"],
                    "workgroupSize": plan["workgroupSize"],
                    "reductionMetadata": {"in_size": len(values), "row_size": row},
                }
                values = [
                    proof.reduce_words(
                        values[i * row : (i + 1) * row], case["operation"]
                    )
                    for i in range(plan["workgroupCount"][1])
                ]
            else:
                event = {
                    "entry": f'row_reduce_small_1_reduce_{case["operation"]}bfloat16',
                    "threads": len(source),
                }
                rows, width = len(expected), case["shape"][-1]
                nonrows = case["shape"][0] if len(case["shape"]) == 3 else 1
                scalar = nonrows <= 8 or (nonrows < 32 and width <= 8)
                group = min(rows, 1024) if scalar else 32
                event.update(
                    workgroupSize=[group, 1, 1],
                    workgroupCount=(
                        [(rows + group - 1) // group, 1, 1] if scalar else [1, rows, 1]
                    ),
                    threadGridSize=[rows, 1, 1] if scalar else [32, rows, 1],
                    reductionMetadata={
                        "row_size": [width],
                        "non_row_reductions": [nonrows],
                        "shape": [rows],
                        "strides": [width],
                        "ndim": [1],
                        "reduce_shape": [nonrows] if nonrows > 1 else [0],
                        "reduce_strides": [rows * width] if nonrows > 1 else [0],
                        "reduce_ndim": [int(nonrows > 1)],
                    },
                )
                values = expected
            trace.append(
                {
                    **event,
                    "target": target,
                    "dispatchVersion": 3,
                    "inputs": {"in_Buffer" if target == "opengl" else "in_": packet},
                    "reductionValues": bfloat_storage.pack(values, target),
                    "reductionGuardValues": bfloat_storage.pack(
                        bfloat_storage.GUARD, target
                    ),
                }
            )
    return records, trace


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "partial",
        "guard",
        "missing",
        "repeat",
        "geometry",
        "metadata",
        "type",
        "input",
        "count",
        "dtype",
    ],
)
def test_reduction_evidence_checks_every_pass(target, fault):
    records, trace = evidence(target, True)
    if fault == "partial":
        trace[7]["reductionValues"][0] = 0
    elif fault == "guard":
        trace[0]["reductionGuardValues"][-1] = 0
    elif fault == "missing":
        trace.pop()
    elif fault == "repeat":
        trace.append(trace[-1])
    elif fault == "geometry":
        trace[0]["workgroupSize"] = [64, 1, 1]
    elif fault == "metadata":
        trace[0]["reductionMetadata"]["row_size"] += 1
    elif fault == "type":
        records[0]["actualWords"][0] = float(records[0]["actualWords"][0])
    elif fault == "input":
        records[0]["inputWords"][0] ^= 1
    elif fault == "count":
        records[0]["dispatchCount"] = True
    elif fault == "dtype":
        records[0]["dtype"] = "mlx.core.float32"
    if fault:
        with pytest.raises(ValueError):
            proof.validate(records, trace, native=True)
    else:
        proof.validate(records, trace, native=True)


def test_cpu_reference_inventory_and_large_launch_widths():
    proof.validate(*evidence("metal", False), native=False)
    assert len(list(proof.cases())) == 36
    assert {
        reduction_layout.stage(case["shape"][0], "bfloat16")["workgroupSize"][0]
        for case in proof.cases()
        if len(case["shape"]) == 1
    } == set(proof.WIDTHS)
    assert proof.reduce_words([], "max") == 0xFF80
    assert proof.reduce_words([], "min") == 0x7F80
    assert proof.reduce_words([0xC000, 0xBF80], "max") == 0xBF80
    assert proof.reduce_words([0x8000, 0x8000], "min") == 0x8000
    assert proof.reduce_words([0x3F80, 0x7FC0], "max") == 0x7FC0


def test_ci_requires_native_bfloat_reductions_on_existing_platform_jobs():
    root = Path(__file__).resolve().parents[5]
    job = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )["jobs"]["half-host"]
    steps = {step.get("name"): step for step in job["steps"]}
    for name in ("Translate bfloat reduction packages", "Execute bfloat reductions"):
        assert "if" not in steps[name] and "continue-on-error" not in steps[name]
    assert (
        "--family bfloat --width 32 --width 128 --width 256"
        in steps["Translate bfloat reduction packages"]["run"]
    )
    assert (
        "portable_host.verify_bfloat_reductions" in steps["Execute bfloat reductions"][
            "run"
        ]
    )
    assert "--timeout-seconds 1800" in steps["Execute bfloat reductions"]["run"]
    assert (
        "test_portable_bfloat_reductions.py" in steps["Validate half host contracts"][
            "run"
        ]
    )
