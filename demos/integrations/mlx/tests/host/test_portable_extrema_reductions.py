"""16-bit float extrema plans, inventory and numerical evidence must remain exact."""

import ctypes
import json
from pathlib import Path

import pytest
import yaml

from demos.integrations.mlx.portable_host import reduction_layout
from demos.integrations.mlx.portable_host import reduction_packages as packages
from demos.integrations.mlx.portable_host import runtime
from demos.integrations.mlx.portable_host import verify_extrema_reductions as proof
from demos.integrations.mlx.portable_host.runtime import physical_dtype
from demos.integrations.mlx.portable_host.small_row_packages import SMALL_ROW_ENTRIES


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("subnormal_input", [False, True])
def test_reduction_subnormal_guard_reads_only_inputs(
    tmp_path, monkeypatch, dtype, target, subnormal_input
):
    """Only bfloat subnormal extrema are refused off Metal (CrossGL/crosstl#2000).

    The half arithmetic proof establishes binary16 subnormal comparisons on
    every target, so float16 extrema reach dispatch with subnormal inputs.
    """
    entry = "all_reduce_max" + dtype
    one = 0x3F80 if dtype == "bfloat16" else 0x3C00
    host = runtime.HostRuntime.__new__(runtime.HostRuntime)
    host.target, host.descriptors, host.retain_native_modules = target, {}, False
    memory = [
        (ctypes.c_uint16 * 2)(1 if subnormal_input else one, 0x4000),
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
                (dtype, dtype, "uint64", "uint64"),
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
    refused = dtype == "bfloat16" and subnormal_input and target != "metal"
    expected = ValueError if refused else ReachedDispatch
    with pytest.raises(expected):
        host.dispatch(
            entry, buffers, 4, 2, launch=runtime.Launch((1, 1, 1), (32, 1, 1))
        )
    assert memory[1][0] == 1


FORMATS = list(proof.FORMATS.values())


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.dtype)
def test_16_bit_reductions_are_extrema_only(fmt):
    assert set(fmt.entries) == {
        f"all_reduce_min{fmt.dtype}",
        f"all_reduce_max{fmt.dtype}",
    }
    assert packages.ENTRY_GROUPS[fmt.family] == fmt.entries
    assert fmt.family in packages.ON_DEMAND_FAMILIES
    assert packages.FAMILY_WIDTHS[fmt.family] == packages.WIDTHS
    assert not fmt.entries.keys() & packages.ENTRIES.keys()
    assert reduction_layout.ITEM_SIZES[fmt.dtype] == 2
    # Compare dtypes, not name suffixes: "float16" ends "bfloat16" too.
    for group in (packages.ROW_ENTRIES, packages.COLUMN_ENTRIES, packages.INIT_ENTRIES):
        assert fmt.dtype not in group.values()
    assert {
        entry for entry, dtype in SMALL_ROW_ENTRIES.items() if dtype == fmt.dtype
    } == {
        f"row_reduce_small_{rank}_reduce_{operation}{fmt.dtype}"
        for rank in (1, 2, 5)
        for operation in ("min", "max")
    }


def test_reduction_inventory_holds_exactly_the_16_bit_extrema():
    assert set(packages.ALL_ENTRIES) == set(packages.ENTRIES) | set(
        packages.BFLOAT_ENTRIES
    ) | set(packages.HALF_ENTRIES)


def test_host_adapter_routes_only_16_bit_extrema():
    source = (
        Path(packages.__file__).with_name("backend.cpp").read_text(encoding="utf-8")
    )
    assert (
        "const bool half_extrema = (in.dtype() == bfloat16 || in.dtype() == float16) &&\n"
        "      (reduce_type_ == Min || reduce_type_ == Max);"
    ) in source
    assert (
        "CrossTL 16-bit float extrema require whole-array or small-row reduction packages."
        in source
    )


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.dtype)
@pytest.mark.parametrize("fault", [None, "entry", "family", "width"])
def test_16_bit_package_index_rejects_mixed_families(tmp_path, fmt, fault):
    entries = list(fmt.entries)
    index = {
        "family": fmt.family,
        "target": "opengl",
        "entries": entries,
        "widths": [32],
        "descriptors": {f"w32/{entry}": {"target": "opengl"} for entry in entries},
    }
    if fault == "entry":
        index["entries"][0] = "all_reduce_sum" + fmt.dtype
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


def evidence(target, native, fmt=proof.BFLOAT):
    dtype, storage = fmt.dtype, fmt.storage
    records, trace = [], []
    for case in proof.cases(fmt):
        source, expected = proof.reference(case, fmt)
        whole = len(case["shape"]) == 1
        dispatches = (2 if whole and len(source) > 4096 else 1) if native else 0
        records.append(
            {
                **case,
                "actualWords": expected,
                "inputWords": source,
                "dtype": f"mlx.core.{dtype}",
                "shapeOut": [] if whole else [len(expected)],
                "dispatchCount": dispatches,
            }
        )
        values = source
        for _ in range(dispatches):
            packet = {
                "dtype": physical_dtype(dtype, target),
                "shape": [len(values)],
                "values": storage.pack(values, target),
            }
            if storage.encoding(target):
                packet["encoding"] = storage.encoding(target)
            if whole:
                plan = reduction_layout.stage(len(values), dtype)
                row = plan["rowSize"]
                event = {
                    "entry": f'all_reduce_{case["operation"]}{dtype}',
                    "threads": len(values),
                    "workgroupCount": plan["workgroupCount"],
                    "workgroupSize": plan["workgroupSize"],
                    "reductionMetadata": {"in_size": len(values), "row_size": row},
                }
                values = [
                    proof.reduce_words(
                        values[i * row : (i + 1) * row], case["operation"], fmt
                    )
                    for i in range(plan["workgroupCount"][1])
                ]
            else:
                event = {
                    "entry": f'row_reduce_small_1_reduce_{case["operation"]}{dtype}',
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
                    "reductionValues": storage.pack(values, target),
                    "reductionGuardValues": storage.pack(storage.GUARD, target),
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
@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.dtype)
def test_reduction_evidence_checks_every_pass(target, fault, fmt):
    records, trace = evidence(target, True, fmt)
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
            proof.validate(records, trace, native=True, fmt=fmt)
    else:
        proof.validate(records, trace, native=True, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.dtype)
def test_cpu_reference_inventory_and_large_launch_widths(fmt):
    proof.validate(*evidence("metal", False, fmt), native=False, fmt=fmt)
    assert len(list(proof.cases(fmt))) == 36 + 2 * len(fmt.profiles)
    assert {
        reduction_layout.stage(case["shape"][0], fmt.dtype)["workgroupSize"][0]
        for case in proof.cases(fmt)
        if len(case["shape"]) == 1
    } == set(proof.WIDTHS)


def test_bfloat_reference_words():
    fmt = proof.BFLOAT
    assert fmt.profiles == ()
    assert proof.reduce_words([], "max") == 0xFF80
    assert proof.reduce_words([], "min") == 0x7F80
    assert proof.reduce_words([0xC000, 0xBF80], "max") == 0xBF80
    assert proof.reduce_words([0x8000, 0x8000], "min") == 0x8000
    assert proof.reduce_words([0x3F80, 0x7FC0], "max") == 0x7FC0
    assert proof.canonical([0xFFC1, 0x7F81, 0x7F80], fmt) == [0x7FC0, 0x7FC0, 0x7F80]


def test_half_reference_words_cover_subnormals():
    fmt = proof.FORMATS["float16"]
    assert proof.reduce_words([], "max", fmt) == 0xFC00
    assert proof.reduce_words([], "min", fmt) == 0x7C00
    assert proof.reduce_words([0xC000, 0xBC00], "max", fmt) == 0xBC00
    assert proof.reduce_words([0x8000, 0x8000], "min", fmt) == 0x8000
    assert proof.reduce_words([0x3C00, 0x7E00], "max", fmt) == 0x7E00
    assert proof.reduce_words([0x0001, 0x03FF, 0x8001], "max", fmt) == 0x03FF
    assert proof.reduce_words([0x0001, 0x03FF, 0x8001], "min", fmt) == 0x8001
    assert proof.canonical([0xFE01, 0x7C01, 0x7C00], fmt) == [0x7E00, 0x7E00, 0x7C00]
    source, expected = proof.reference(
        {"operation": "max", "shape": [509], "profile": "subnormal"}, fmt
    )
    assert all(word & 0x7C00 == 0 for word in source)
    assert {word & 0x8000 for word in source if word & 0x7FFF} == {0, 0x8000}
    assert expected == [26]


def test_ci_requires_native_16_bit_reductions_on_existing_platform_jobs():
    root = Path(__file__).resolve().parents[5]
    job = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )["jobs"]["half-host"]
    steps = {step.get("name"): step for step in job["steps"]}
    for kind, family, dtype, elementwise in (
        ("bfloat", "bfloat", "bfloat16", "bfloat-packages"),
        ("float16", "half", "float16", "half-packages"),
    ):
        translate = steps[f"Translate {kind} reduction packages"]
        execute = steps[f"Execute {kind} reductions"]
        for step in (translate, execute):
            assert "if" not in step and "continue-on-error" not in step
        assert (
            f"--family {family} --width 32 --width 128 --width 256" in translate["run"]
        )
        assert "portable_host.verify_extrema_reductions" in execute["run"]
        assert f"--dtype {dtype}" in execute["run"]
        assert f"--elementwise .mlx-portable-half/{elementwise}" in execute["run"]
        assert "--timeout-seconds 1800" in execute["run"]
    assert (
        "test_portable_extrema_reductions.py" in steps["Validate half host contracts"][
            "run"
        ]
    )


@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_inputs_hold_the_reference_words(dtype):
    """MLX's software float16 conversion stores NaN as 0x7D00; the workers
    build float16 inputs from the reference bits instead."""
    import numpy as np

    fmt = proof.FORMATS[dtype]
    calls = []

    class Array:
        def __init__(self, values, dtype=None):
            calls.append((values, dtype))

        def reshape(self, shape):
            calls.append(shape)
            return self

    mx = type("mx", (), {"array": Array, "bfloat16": "mx.bfloat16"})
    case = {"operation": "min", "shape": [509], "profile": "nan-first"}
    source, _ = proof.reference(case, fmt)
    assert source[0] == fmt.nan
    proof.source_array(mx, source, [509], fmt)
    (values, kind), shape = calls
    assert shape == [509]
    if dtype == "float16":
        assert kind is None and values.dtype == np.float16
        assert values.view(np.uint16).tolist() == source
    else:
        assert kind == "mx.bfloat16"
        assert list(map(fmt.word, values)) == proof.canonical(source, fmt)
