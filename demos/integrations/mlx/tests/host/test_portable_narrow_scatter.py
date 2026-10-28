"""General scatters into lanes of packed atomic words: 8- and 16-bit types."""

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import (
    gather_dispatch,
    runtime,
    scatter_arithmetic,
    scatter_evidence,
)
from demos.integrations.mlx.portable_host import scatter_layout as layout
from demos.integrations.mlx.portable_host import scatter_storage
from demos.integrations.mlx.portable_host import scatter_workloads as workloads
from demos.integrations.mlx.portable_host import verify_scatter
from demos.integrations.mlx.portable_host.element_storage import PHYSICAL_SIZES
from demos.integrations.mlx.tests.host.test_portable_scatter import launch_buffers

CASES = [*workloads.narrow_cases()]
TARGETS = ("metal", "directx", "opengl")


def test_narrow_profile_covers_every_lane_type_and_operation():
    assert {(case["dtype"], case["operation"]) for case in CASES} == {
        (dtype, operation)
        for dtype in workloads.NARROW_TYPES
        for operation in workloads.OPERATIONS
    }
    assert len({case["id"] for case in CASES}) == len(CASES)
    assert {case["layout"] for case in CASES} == {
        *workloads.NARROW_LAYOUTS,
        "wrap",
        "rounding",
    }
    assert list(workloads.profile_cases("narrow")) == CASES
    with pytest.raises(ValueError):
        list(workloads.profile_cases("float64"))


@pytest.mark.parametrize("dtype", workloads.NARROW_TYPES)
def test_narrow_entries_carry_packed_atomic_words(dtype):
    entry = f"scatter{dtype}int64_sum_2_updc_false_nwork4_int"
    assert layout.signature(entry) == (dtype, "int64", 2, "sum", False, 4)
    for target in TARGETS:
        assert layout.atomic_storage_dtype(dtype, target) == "uint32"


@pytest.mark.parametrize("dtype", workloads.NARROW_TYPES)
@pytest.mark.parametrize("count", (1, 2, 3, 5))
def test_packed_words_round_trip_with_padding_and_guards(dtype, count):
    ctype = scatter_storage.CTYPES[dtype]
    bits = 8 * ctypes.sizeof(ctype)
    signed = ctype(-1).value == -1
    low = -(1 << bits - 1) if signed else 0
    logical = [low + (index * 37) % (1 << bits) for index in range(count)]
    words = scatter_storage.pack(dtype, logical)
    lanes = 32 // bits
    assert len(words) == -(-count // lanes) + scatter_storage.GUARD_COUNT
    decoded, trailing = scatter_storage.split(dtype, words, count)
    assert decoded == logical
    assert trailing == scatter_storage.tail(dtype, count)
    assert len(trailing) == -count % lanes + scatter_storage.GUARD_COUNT
    first = sum(
        (word & (1 << bits) - 1) << bits * lane
        for lane, word in enumerate(
            (logical + [scatter_storage.lane_guard(dtype)] * lanes)[:lanes]
        )
    )
    assert words[0] == first
    with pytest.raises(ValueError):
        scatter_storage.pack(dtype, logical[:-1] + [1 << bits])
    with pytest.raises(ValueError):
        scatter_storage.split(dtype, words[:-1], count)
    with pytest.raises(ValueError):
        scatter_storage.split(dtype, words[:-1] + [1 << 32], count)


def test_bfloat16_rounding_matches_mlx_conversion():
    assert scatter_arithmetic.bfloat16_bits(1.0) == 0x3F80
    # Ties round to even; above a tie rounds up.
    assert scatter_arithmetic.bfloat16_bits(1.0 + 2.0**-8) == 0x3F80
    assert scatter_arithmetic.bfloat16_bits(1.0 + 3 * 2.0**-8) == 0x3F82
    assert scatter_arithmetic.bfloat16_bits(1.0 + 2.0**-8 + 2.0**-20) == 0x3F81
    assert scatter_arithmetic.bfloat16_bits(float("nan")) == 0x7FC0
    assert scatter_arithmetic.bfloat16_bits(-float("inf")) == 0xFF80
    assert scatter_arithmetic.bfloat16_bits(3.4e38) == 0x7F80


def test_lane_arithmetic_follows_mlx_atomic_updates():
    combine = scatter_arithmetic.combine
    assert combine(np, "uint8", "sum", np.uint8(250), np.uint8(10)) == 4
    assert combine(np, "int8", "prod", np.int8(-100), np.int8(3)) == -44
    assert combine(np, "int16", "sum", np.int16(32767), np.int16(1)) == -32768
    nan = np.float16("nan")
    # A NaN update never replaces a lane; a NaN lane is never replaced.
    assert combine(np, "float16", "min", np.float16(1), nan) == 1
    assert np.isnan(combine(np, "float16", "max", nan, np.float16(1)))
    assert combine(np, "float16", "none", np.float16(1), nan) is nan
    one = np.float16(1)
    assert combine(np, "float16", "sum", one, np.float16(2.0**-11)) == one
    tie = np.float32(1 + 2.0**-8)
    assert combine(np, "bfloat16", "sum", np.float32(1), tie - 1) == 1
    assert scatter_arithmetic.bfloat16_bits(
        combine(np, "bfloat16", "prod", np.float32(3), np.float32(1 / 3))
    ) == scatter_arithmetic.bfloat16_bits(np.float32(3) * np.float32(1 / 3))


@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_rounding_workload_depends_on_update_order(dtype):
    case = next(case for case in CASES if case["id"] == f"{dtype}-sum-rounding")
    source, (index,), updates, expected = workloads.reference(np, case)
    reverse = source.copy()
    for row in reversed(range(index.size)):
        for lane in range(source.shape[1]):
            reverse[index[row], lane] = scatter_arithmetic.combine(
                np, dtype, "sum", reverse[index[row], lane], updates[row, lane]
            )
    assert workloads.storage_words(np, dtype, reverse) != workloads.storage_words(
        np, dtype, expected
    )


@pytest.mark.parametrize("dtype", workloads.NARROW_TYPES[:4])
@pytest.mark.parametrize("operation", ("sum", "prod"))
def test_wrap_workload_overflows_only_its_own_lanes(dtype, operation):
    case = next(case for case in CASES if case["id"] == f"{dtype}-{operation}-wrap")
    source, (index,), updates, expected = workloads.reference(np, case)
    exact = source.astype(np.int64)
    for row in range(index.size):
        if operation == "sum":
            exact[index[row]] += updates[row].astype(np.int64)
        else:
            exact[index[row]] *= updates[row].astype(np.int64)
    info = np.iinfo(dtype)
    assert ((exact < info.min) | (exact > info.max)).any()
    assert np.array_equal(expected, exact.astype(dtype))
    # Even lanes take only identity updates and keep their source values.
    assert np.array_equal(expected[:, ::2], source[:, ::2])


def storage_array(dtype, array):
    if dtype != "bfloat16":
        return np.asarray(array)
    words = workloads.storage_words(np, dtype, array)
    return np.array(words, dtype=np.uint16).reshape(np.shape(array))


def narrow_buffers(case):
    source, indices, updates, expected = workloads.reference(np, case)
    dtype = case["dtype"]
    entry, supplied, arrays, _, execution = launch_buffers(
        case,
        storage_array(dtype, source),
        indices,
        storage_array(dtype, updates),
        expected,
    )
    for name in ("updates", "out"):
        supplied[name].dtype = dtype.encode()
    return entry, supplied, arrays, expected, execution


def reflected(dtype, target, name, value):
    if name == "out" and scatter_storage.packed(dtype):
        return {
            "memberName": "out_",
            "elementType": "uint32",
            "elementStrideBytes": 4,
            "componentCount": 1,
            "structMembers": scatter_storage.ATOMIC_MEMBERS,
        }
    return {
        "memberName": name,
        "elementType": value["dtype"],
        "elementStrideBytes": PHYSICAL_SIZES.get(value["dtype"], 1),
    }


def descriptor_for(dtype, target, supplied):
    bindings = []
    for name, buffer in supplied.items():
        kind = buffer.dtype.decode()
        if name in ("updates", "out"):
            value = (
                scatter_storage.output_value(dtype, buffer)
                if name == "out"
                else gather_dispatch.gather_storage.value_packet(
                    dtype, target, buffer, False
                )
            )
        else:
            storage = runtime.physical_dtype(kind, target)
            value = {"dtype": storage}
        bindings.append(
            {"name": name, "scalarLayout": reflected(dtype, target, name, value)}
        )
    return {"artifact": {}, "bindings": bindings}


def dispatch(tmp_path, monkeypatch, case, target, fault=None):
    """Run the host dispatch against a native stand-in that applies the reference."""
    entry, supplied, arrays, expected, execution = narrow_buffers(case)
    dtype = case["dtype"]
    descriptor = descriptor_for(dtype, target, supplied)
    logical = scatter_arithmetic.storage_words(np, dtype, expected)
    if dtype in scatter_arithmetic.INTEGERS:
        logical = [int(value) for value in expected.reshape(-1)]
    seen = {}

    def request(_descriptor, _directory, inputs, outputs, launch, **kwargs):
        seen["inputs"] = inputs
        return SimpleNamespace(outputs=outputs)

    def execute(_host, request):
        result = {
            **request.outputs["out"],
            "values": scatter_storage.pack(dtype, logical),
        }
        if fault == "padding":
            result["values"][
                expected.size // (4 // scatter_storage.element_size(dtype))
            ] ^= (0xFF << 24)
        elif fault == "guard":
            result["values"][-1] ^= 1
        elif fault == "dtype":
            result["dtype"] = "int32"
        return SimpleNamespace(status="ok", outputs={"out": result}, details={})

    records = []
    monkeypatch.setattr(
        gather_dispatch, "build_native_loader_dispatch_request", request
    )
    monkeypatch.setattr(gather_dispatch, "execute", execute)
    monkeypatch.setattr(
        gather_dispatch, "write_trace", lambda _host, record: records.append(record)
    )
    host = SimpleNamespace(
        target=target,
        trace=tmp_path / "trace.jsonl",
        dispatch_count=0,
        gathers=SimpleNamespace(get=lambda *_: (descriptor, tmp_path)),
    )
    table = (runtime.Buffer * len(supplied))(*supplied.values())
    launch = runtime.Launch(tuple(execution["workgroupCount"]), (1, 1, 1))
    initial = arrays["out"].copy()
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            gather_dispatch.dispatch(
                host, entry, table, len(table), expected.size, launch
            )
        assert np.array_equal(arrays["out"], initial)
        assert host.dispatch_count == 0 and not records
        return None
    gather_dispatch.dispatch(host, entry, table, len(table), expected.size, launch)
    assert host.dispatch_count == 1
    assert workloads.storage_words(np, None, arrays["out"]) == (
        workloads.storage_words(np, None, storage_array(dtype, expected))
    )
    return records[0], seen["inputs"], descriptor, expected


SAMPLES = [
    case
    for case in CASES
    if case["layout"] in ("dense", "update-broadcast", "multiple", "wrap", "rounding")
    or case["layout"] == "scalar"
    and case["operation"] == "max"
]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", SAMPLES, ids=lambda case: case["id"])
def test_narrow_dispatch_packs_lanes_and_audit_recomputes_them(
    tmp_path, monkeypatch, target, case
):
    from demos.integrations.mlx.tests.host.test_portable_scatter import scatter_event

    record, inputs, descriptor, expected = dispatch(tmp_path, monkeypatch, case, target)
    out = inputs["out"]
    assert out["dtype"] == "uint32" and out["shape"] == [len(out["values"]), 1]
    template = next(case for case in workloads.cases() if case["layout"] == "dense")
    event = scatter_event(tmp_path, template, target)
    layouts = {
        binding["name"]: binding["scalarLayout"] for binding in descriptor["bindings"]
    }
    event.update(
        {
            key: record[key]
            for key in (
                "entry",
                "threads",
                "workgroupCount",
                "workgroupSize",
                "scatterMetadata",
                "scatterStorageType",
                "scatterValues",
                "scatterGuardValues",
                "inputs",
                "outputHash",
            )
        }
    )
    request = event["details"]["request"]
    native_entry = {"metal": record["entry"], "directx": "CSMain", "opengl": "main"}
    request.update(
        entryPoint=native_entry[target],
        buffers={
            name: {
                **{key: item for key, item in value.items() if key != "values"},
                "binding": {"metadata": {"scalarLayout": layouts[name]}},
            }
            for name, value in inputs.items()
        },
    )
    request["dispatch"].update(
        entryPoint=native_entry[target],
        workgroupCount=record["workgroupCount"],
        globalSize=record["workgroupCount"],
        gridSize=record["workgroupCount"],
    )
    initial, indices, updates, actual = scatter_evidence.audit_event(np, event)
    source, _, expected_updates, expected = workloads.reference(np, case)
    dtype = case["dtype"]
    assert workloads.storage_words(np, dtype, initial) == workloads.storage_words(
        np, dtype, source
    )
    assert workloads.storage_words(np, dtype, updates) == workloads.storage_words(
        np, dtype, expected_updates
    )
    assert actual == workloads.storage_words(np, dtype, expected)
    corrupt = dict(event, scatterValues=list(event["scatterValues"]))
    corrupt["scatterValues"][0] ^= 1
    with pytest.raises(ValueError):
        scatter_evidence.audit_event(np, corrupt)
    padded = dict(event, scatterGuardValues=list(event["scatterGuardValues"]))
    padded["scatterGuardValues"][0] ^= 1
    with pytest.raises(ValueError):
        scatter_evidence.audit_event(np, padded)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("fault", ("padding", "guard", "dtype"))
@pytest.mark.parametrize("dtype", ("uint8", "int16", "bfloat16"))
def test_narrow_dispatch_refuses_changed_padding_guards_and_storage(
    tmp_path, monkeypatch, target, fault, dtype
):
    # These 3x5 destinations end inside a word, so they keep padding lanes.
    special = "sum-rounding" if dtype == "bfloat16" else "sum-wrap"
    case = next(case for case in CASES if case["id"] == f"{dtype}-{special}")
    _, _, _, expected, _ = narrow_buffers(case)
    assert expected.size % (4 // scatter_storage.element_size(dtype))
    dispatch(tmp_path, monkeypatch, case, target, fault)


@pytest.mark.parametrize("native", (False, True))
def test_narrow_records_require_the_complete_profile(native):
    records = []
    for index, case in enumerate(CASES):
        *_, expected = workloads.reference(np, case)
        records.append(
            {
                **case,
                "actual": workloads.storage_words(np, case["dtype"], expected),
                "shape": list(expected.shape),
                "resultDtype": "mlx.core." + case["dtype"],
                "inputUnchanged": True,
                "dispatchStart": index if native else 0,
                "dispatchCount": 1 if native else 0,
            }
        )
    verify_scatter.validate_records(np, records, native=native, profile="narrow")
    with pytest.raises(ValueError):
        verify_scatter.validate_records(
            np, records[:-1], native=native, profile="narrow"
        )
    records[0]["actual"] = records[0]["actual"][::-1]
    with pytest.raises(ValueError):
        verify_scatter.validate_records(np, records, native=native, profile="narrow")


def test_rounding_sums_run_as_maximal_disjoint_index_order_runs():
    # Mirrors scatter.inc's scatter_runs: a run ends at the first row whose
    # destination it already holds.
    rounding = [case for case in CASES if case["layout"] == "rounding"]
    assert {case["dtype"] for case in rounding} == {"float16", "bfloat16"}
    for case in rounding:
        runs = workloads.ordered_runs(case)
        assert runs == [(0, 2), (2, 1), (3, 2), (5, 1)]
        assert [first for first, _ in runs] == sorted(first for first, _ in runs)
        assert sum(rows for _, rows in runs) == len(workloads.SPECIAL_INDEX)
        for position, (first, rows) in enumerate(runs):
            destinations = workloads.SPECIAL_INDEX[first : first + rows]
            assert len(set(destinations)) == rows
            if position + 1 < len(runs):
                assert workloads.SPECIAL_INDEX[first + rows] in destinations
    assert all(
        workloads.ordered_runs(case) is None
        for case in CASES
        if case["layout"] != "rounding"
    )


def _rounding_trace(monkeypatch, fault):
    """One float16 rounding workload whose launches ``audit_event`` returns."""
    from demos.integrations.mlx.portable_host import verify_bitwise, verify_gather

    case = next(case for case in CASES if case["id"] == "float16-sum-rounding")
    monkeypatch.setattr(workloads, "profile_cases", lambda profile: iter([case]))
    monkeypatch.setattr(verify_scatter, "validate_records", lambda *a, **k: None)
    monkeypatch.setattr(verify_gather, "validate_upstream", lambda *a, **k: None)
    monkeypatch.setattr(
        verify_bitwise, "verify_native_identity", lambda events, target: None
    )
    source, (index,), updates, expected = workloads.reference(np, case)
    runs = workloads.ordered_runs(case)
    # Distinct intermediate destinations; the last launch leaves the result.
    states = [source, *(source + np.float16(k) for k in range(1, len(runs))), expected]
    tiles = {}
    events = []
    for position, (first, rows) in enumerate(runs):
        event = {"entry": "scatterfloat16int64_sum", "target": "metal"}
        tile_index = index if fault == "rows" else index[first : first + rows]
        initial = source if fault == "chain" else states[position]
        actual = workloads.storage_words(np, "float16", states[position + 1])
        tiles[id(event)] = (
            initial,
            [tile_index],
            updates[first : first + rows],
            actual,
        )
        events.append(event)
    if fault == "count":
        events.pop()
    upstream_event = {"entry": "scatterfloat16int64_sum", "target": "metal"}
    tiles[id(upstream_event)] = tiles[id(events[0])]
    monkeypatch.setattr(
        scatter_evidence, "audit_event", lambda np, event: tiles[id(event)]
    )
    record = {
        **case,
        "actual": workloads.storage_words(np, "float16", expected),
        "dispatchCount": len(events),
    }
    upstream = {"dispatchStart": len(events), "dispatchCount": 1}
    return [record], [*events, upstream_event], upstream


@pytest.mark.parametrize("fault", (None, "count", "chain", "rows"))
def test_rounding_launches_chain_their_destinations_in_index_order(monkeypatch, fault):
    records, trace, upstream = _rounding_trace(monkeypatch, fault)
    if fault:
        with pytest.raises(ValueError):
            scatter_evidence.validate(np, records, trace, upstream, profile="narrow")
    else:
        assert scatter_evidence.validate(
            np, records, trace, upstream, profile="narrow"
        ) == {
            "target": "metal",
            "scatterDispatchCount": 4,
            "upstreamScatterDispatchCount": 1,
        }


@pytest.mark.parametrize("installed", (False, True))
def test_lazy_copies_are_audited_from_their_retained_values(monkeypatch, installed):
    from demos.integrations.mlx.portable_host import copy_evidence, verify_bitwise

    records, trace, upstream = _rounding_trace(monkeypatch, None)
    copy = {"entry": "ggn2_dynamic_copyfloat16float16", "target": "metal"}
    assert copy["entry"] in runtime.LAZY_COPY_ENTRIES
    trace.insert(0, copy)
    records[0]["dispatchCount"] += 1
    upstream["dispatchStart"] += 1
    identities, copies, cache = [], [], object()
    monkeypatch.setattr(
        verify_bitwise,
        "verify_native_identity",
        lambda events, target: identities.append(events),
    )
    monkeypatch.setattr(
        copy_evidence, "audit_event", lambda event, store: copies.append((event, store))
    )
    scatter_evidence.validate(
        np,
        records,
        trace,
        upstream,
        profile="narrow",
        lazy_copies=({copy["entry"]} if installed else set(), cache),
    )
    if installed:
        assert identities == [[copy]] and copies == []
    else:
        assert identities == [[]] and copies == [(copy, cache)]
