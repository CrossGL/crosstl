"""8- and 16-bit integer general scatter through packed atomic words."""

import ctypes
import hashlib
import itertools
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import (
    element_storage,
    gather_dispatch,
    runtime,
    scatter_evidence,
)
from demos.integrations.mlx.portable_host import scatter_layout as layout
from demos.integrations.mlx.portable_host import scatter_workloads as workloads
from demos.integrations.mlx.portable_host import verify_scatter
from demos.integrations.mlx.portable_host.gather_workloads import words
from demos.integrations.mlx.tests.host.test_portable_scatter import (
    buffers,
    scatter_event,
)
from tests.ci_helpers import assert_paths_covered

CASES = list(workloads.small_int_cases())
ACTIVE = [case for case in CASES if case["layout"] != "empty"]
TARGETS = ("metal", "directx", "opengl")
WRAP = {
    "int8": ctypes.c_int8,
    "uint8": ctypes.c_uint8,
    "int16": ctypes.c_int16,
    "uint16": ctypes.c_uint16,
}


def workload_digest(cases):
    digest = hashlib.sha256()
    for case in cases:
        digest.update(case["id"].encode())
        source, indices, updates, expected = workloads.reference(np, case)
        for array in (source, *indices, updates, expected):
            digest.update(str((str(array.dtype), array.shape, array.strides)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def independent_update(dtype, operation, current, update):
    """atomic.h's operators on Python integers, wrapped by ctypes conversion."""
    if operation == "none":
        return update
    if operation == "sum":
        return WRAP[dtype](current + update).value
    if operation == "prod":
        return current if current == 0 else WRAP[dtype](current * update).value
    if operation == "min":
        return update if update < current else current
    return update if update > current else current


def destinations(source, indices, updates):
    for coordinate in np.ndindex(indices[0].shape):
        # Negative indices address the same element as their positive peers.
        prefix = tuple(
            int(index[coordinate]) % size for index, size in zip(indices, source.shape)
        )
        for offset in np.ndindex(source.shape[len(prefix) :]):
            yield prefix + offset, int(updates[coordinate + offset])


def test_small_int_scatter_case_inventory():
    assert len(CASES) == len({case["id"] for case in CASES}) == 264
    assert len(ACTIVE) == 254
    assert {case["dtype"] for case in CASES} == set(workloads.SMALL_INT_TYPES)
    assert {case["operation"] for case in CASES} == {
        "none",
        "sum",
        "prod",
        "min",
        "max",
    }
    assert {case["index_dtype"] for case in CASES} == {
        "int32",
        "uint32",
        "int64",
        "uint64",
    }
    for dtype in workloads.SMALL_INT_TYPES:
        layouts = {case["layout"] for case in CASES if case["dtype"] == dtype}
        assert {"odd", "tail", "extremes", "alias", "work32", "capacity"} <= layouts
    assert {case["layout"] for case in CASES if case["dtype"] == "int8"} == {
        *workloads.SMALL_INT_LAYOUTS,
        "capacity",
    }


def test_small_int_scatter_preserves_the_other_profiles_workload_bytes():
    assert (
        workload_digest(workloads.cases())
        == "d40ae267b8d01ee8445dafc6eb54f8d38374e8de83efdf2be201446ce4f043fa"
    )
    assert (
        workload_digest(workloads.half_cases())
        == "aea11f99f1817f8870a8faace1c9f852444df57db09cf9d86fbaa8587127a3c1"
    )


def test_small_int_scatter_profiles_are_exclusive():
    for flags in (
        {"half": True, "small_int": True},
        {"float32": True, "small_int": True},
    ):
        with pytest.raises(ValueError):
            workloads.profile_cases(**flags)
    assert [case["id"] for case in workloads.profile_cases(small_int=True)] == [
        case["id"] for case in CASES
    ]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_small_int_scatter_reference_matches_wrapping_model(case):
    source, indices, updates, expected = workloads.reference(np, case)
    assert source.dtype == updates.dtype == expected.dtype == np.dtype(case["dtype"])
    actual = source.copy()
    for address, update in destinations(source, indices, updates):
        actual[address] = independent_update(
            case["dtype"], case["operation"], int(actual[address]), update
        )
    assert words(np, actual) == words(np, expected)


@pytest.mark.parametrize(
    "case",
    [case for case in ACTIVE if case["operation"] != "none"],
    ids=lambda case: case["id"],
)
def test_small_int_scatter_accumulations_do_not_depend_on_order(case):
    """Native atomics may apply contending updates in any order."""
    source, indices, updates, expected = workloads.reference(np, case)
    groups = {}
    for address, update in destinations(source, indices, updates):
        groups.setdefault(address, []).append(update)
    shuffle = random.Random(case["id"])
    for address, group in groups.items():
        for _ in range(4):
            order = group.copy()
            shuffle.shuffle(order)
            value = int(source[address])
            for update in order:
                value = independent_update(
                    case["dtype"], case["operation"], value, update
                )
            assert value == int(expected[address])


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case["operation"] == "none"],
    ids=lambda case: case["id"],
)
def test_small_int_scatter_replacements_do_not_depend_on_order(case):
    source, indices, updates, _ = workloads.reference(np, case)
    groups = {}
    for address, update in destinations(source, indices, updates):
        groups.setdefault(address, set()).add(update)
    assert all(len(group) == 1 for group in groups.values())


def extremes(dtype, operation):
    case = next(case for case in CASES if case["id"] == f"{dtype}-{operation}-extremes")
    return case, workloads.reference(np, case)


@pytest.mark.parametrize("dtype", workloads.SMALL_INT_TYPES)
def test_small_int_scatter_extremes_cross_sign_and_wrap_boundaries(dtype):
    info = np.iinfo(dtype)
    raw = f"uint{info.bits}"
    for operation in ("sum", "prod"):
        _, (source, indices, updates, expected) = extremes(dtype, operation)
        exact = source.astype(object)
        for address, update in destinations(source, indices, updates):
            current = exact[address]
            exact[address] = (
                current + update if operation == "sum" else current * update
            )
        # Results wrap past the top of the range, and signed sums past the bottom.
        assert (exact > info.max).any()
        if operation == "sum" and info.min < 0:
            assert (exact < info.min).any()
        assert (exact != expected.astype(object)).any()
        lanes = expected.view(raw)
        if operation == "sum" and info.bits == 16:
            # A carry out of the low byte must stay inside its 16-bit lane.
            assert ((lanes & 0xFF == 0) & (lanes != 0)).any()
        if operation == "prod":
            assert (lanes == 0).any()
    peer = {"int8": "uint8", "uint8": "int8", "int16": "uint16", "uint16": "int16"}
    for operation in ("min", "max"):
        case, (*_, mine) = extremes(dtype, operation)
        theirs = workloads.reference(np, dict(case, dtype=peer[dtype]))[3]
        # Signed and unsigned lanes holding the same bits order differently.
        assert (mine.view(raw) != theirs.view(raw)).any()


def test_small_int_scatter_packs_four_bytes_low_lane_first():
    assert layout.packing("int8") == layout.packing("uint8") == 4
    assert layout.packing("int16") == layout.packing("uint16") == 2
    assert layout.packing("int32") == 1
    assert layout.pack_lanes([1, 2, 3, 4, 5], 0xA5, "uint8") == [
        0x04030201,
        0xA5A5A505,
    ]
    assert layout.unpack_lanes([0x04030201, 0xA5A5A505], 5, "int8") == (
        [1, 2, 3, 4, 5],
        [0xA5, 0xA5, 0xA5],
    )
    assert layout.pack_lanes([0x1234, 0xFFFF, 0x0001], 0x3555, "int16") == [
        0xFFFF1234,
        0x35550001,
    ]
    assert layout.storage_count("int8", 9) == 3
    assert layout.storage_count("uint16", 9) == 5
    assert layout.storage_count("uint32", 9) == 9
    for dtype, lanes in (("uint8", [0x100]), ("int8", [-1]), ("int16", [0x10000])):
        with pytest.raises(ValueError):
            layout.pack_lanes(lanes, 0, dtype)
    with pytest.raises(ValueError):
        layout.pack_lanes([0], 0x100, "uint8")
    with pytest.raises(ValueError):
        layout.unpack_lanes([0, 0], 4, "uint8")
    with pytest.raises(ValueError):
        layout.pack_lanes([0], 0, "int32")
    assert layout.raw_lanes("int8", [-1, -128, 127]) == [0xFF, 0x80, 0x7F]
    assert layout.logical_lanes("int8", [0xFF, 0x80, 0x7F]) == [-1, -128, 127]
    assert layout.logical_lanes("uint16", [0xFFFF]) == [0xFFFF]
    with pytest.raises(ValueError):
        layout.raw_lanes("int8", [128])
    for dtype in workloads.SMALL_INT_TYPES:
        guard = element_storage.GUARDS[dtype]
        assert layout.logical_lanes(dtype, [layout.PADDING[dtype]]) == [guard]


@pytest.mark.parametrize("dtype", workloads.SMALL_INT_TYPES)
@pytest.mark.parametrize("operation", ("none", "sum", "prod", "min", "max"))
def test_small_int_scatter_entries_parse(dtype, operation):
    entry = f"scatter{dtype}int64_{operation}_2_updc_false_nwork8_int"
    assert layout.signature(entry) == (dtype, "int64", 2, operation, False, 8)
    host = SimpleNamespace(target="metal", descriptors={}, gathers=object())
    assert runtime.HostRuntime._entry_available(host, entry.encode()) == 1


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", ACTIVE, ids=lambda case: case["id"])
def test_small_int_scatter_audit_reconstructs_uploads(tmp_path, target, case):
    event = scatter_event(tmp_path, case, target)
    source, indices, updates, expected = workloads.reference(np, case)
    initial, actual_indices, actual_updates, actual = scatter_evidence.audit_event(
        np, event
    )
    assert initial.dtype == actual_updates.dtype == np.dtype(case["dtype"])
    assert words(np, initial) == words(np, source)
    assert all(np.array_equal(a, b) for a, b in zip(indices, actual_indices))
    assert words(np, actual_updates) == words(np, updates)
    assert actual == words(np, expected)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", workloads.SMALL_INT_TYPES)
def test_small_int_scatter_events_carry_widened_updates(tmp_path, target, dtype):
    case = next(case for case in CASES if case["id"] == f"{dtype}-sum-tail")
    event = scatter_event(tmp_path, case, target)
    out, updates = event["inputs"]["out"], event["inputs"]["updates"]
    words_count = layout.storage_count(dtype, 15)
    assert out["dtype"] == "uint32" and "encoding" not in out
    assert out["shape"] == [words_count + len(runtime.COPY_GUARD), 1]
    bits = layout.LANE_BITS[dtype]
    unused = 32 // bits - 15 % (32 // bits)
    assert out["values"][words_count - 1] >> (32 - unused * bits) == int(
        "".join([f"{layout.PADDING[dtype]:0{bits // 4}x}"] * unused), 16
    )
    assert updates["dtype"] == element_storage.WIDENED[target].get(dtype, dtype)
    assert "encoding" not in updates


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", workloads.SMALL_INT_TYPES)
@pytest.mark.parametrize(
    "fault",
    (
        "upload-padding",
        "readback-padding",
        "readback-length",
        "readback-large",
        "readback-value",
        "readback-lane",
        "upload-word",
        "update-carrier",
        "update-dtype",
        "out-dtype",
        "guard",
        "hash",
    ),
)
def test_small_int_scatter_rejects_corrupt_evidence(tmp_path, target, dtype, fault):
    case = next(case for case in CASES if case["id"] == f"{dtype}-max-odd")
    event = scatter_event(tmp_path, case, target)
    request = event["details"]["request"]["buffers"]
    out, updates = event["inputs"]["out"], event["inputs"]["updates"]
    words_count = layout.storage_count(dtype, 9)
    if fault == "upload-padding":
        out["values"][words_count - 1] ^= 1 << 31
    elif fault == "readback-padding":
        event["scatterValues"][-1] ^= 1 << 31
    elif fault == "readback-length":
        event["scatterValues"].pop()
    elif fault == "readback-large":
        event["scatterValues"][0] = 2**32
    elif fault == "readback-value":
        event["scatterValues"][0] ^= 1
    elif fault == "readback-lane":
        # A change in the second lane of the first word.
        event["scatterValues"][0] ^= 1 << layout.LANE_BITS[dtype]
    elif fault == "upload-word":
        out["values"][0] = -1
    elif fault == "update-carrier":
        # Widened carriers must still hold a value of the element type.
        updates["values"][0] = np.iinfo(dtype).max + 1
    elif fault == "update-dtype":
        updates["dtype"] = request["updates"]["dtype"] = "int64"
    elif fault == "out-dtype":
        out["dtype"] = request["out"]["dtype"] = dtype
    elif fault == "guard":
        event["scatterGuardValues"][0] = 0
    else:
        event["outputHash"] = "0" * 64
    with pytest.raises(ValueError):
        scatter_evidence.audit_event(np, event)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", workloads.SMALL_INT_TYPES)
@pytest.mark.parametrize(
    "fault",
    (None, "padding", "guard", "dtype", "large", "member", "carrier"),
)
def test_small_int_scatter_dispatch_round_trips_packed_words(
    tmp_path, monkeypatch, target, dtype, fault
):
    case = next(case for case in CASES if case["id"] == f"{dtype}-sum-tail")
    entry, supplied, arrays, expected, execution = buffers(case)
    event = scatter_event(tmp_path, case, target)
    descriptor = {
        "artifact": {},
        "bindings": [
            {
                "name": name,
                "kind": "buffer",
                "scalarLayout": value["binding"]["metadata"]["scalarLayout"],
            }
            for name, value in event["details"]["request"]["buffers"].items()
        ],
    }
    bindings = {binding["name"]: binding for binding in descriptor["bindings"]}
    if fault == "member":
        bindings["out"]["scalarLayout"]["structMembers"][0]["physicalType"] = "int"
    elif fault == "carrier":
        bindings["updates"]["scalarLayout"]["elementType"] = "uint64"
    lanes = layout.raw_lanes(dtype, arrays["out"].reshape(-1).tolist())
    packed = layout.pack_lanes(lanes, layout.PADDING[dtype], dtype)

    def request(_descriptor, _directory, inputs, outputs, launch, **kwargs):
        assert inputs["out"]["values"] == packed + runtime.COPY_GUARD
        assert inputs["out"]["dtype"] == "uint32" and "encoding" not in inputs["out"]
        assert inputs["updates"] == event["inputs"]["updates"]
        return SimpleNamespace(outputs=outputs)

    def execute(_host, request):
        values = layout.pack_lanes(
            layout.raw_lanes(dtype, expected.reshape(-1).tolist()),
            layout.PADDING[dtype],
            dtype,
        )
        output = {**request.outputs["out"], "values": values + runtime.COPY_GUARD}
        if fault == "padding":
            output["values"][len(values) - 1] ^= 1 << 31
        elif fault == "guard":
            output["values"][-1] = 0
        elif fault == "dtype":
            output["dtype"] = dtype
        elif fault == "large":
            output["values"][0] = 2**32
        return SimpleNamespace(status="ok", outputs={"out": output}, details={})

    monkeypatch.setattr(
        gather_dispatch, "build_native_loader_dispatch_request", request
    )
    monkeypatch.setattr(gather_dispatch, "execute", execute)
    host = SimpleNamespace(
        target=target,
        trace=tmp_path / "trace.jsonl",
        dispatch_count=0,
        gathers=SimpleNamespace(get=lambda *_: (descriptor, tmp_path)),
    )
    table = (runtime.Buffer * len(supplied))(*supplied.values())
    launch = runtime.Launch(tuple(execution["workgroupCount"]), (1, 1, 1))
    initial = arrays["out"].tobytes()
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            gather_dispatch.dispatch(
                host, entry, table, len(table), expected.size, launch
            )
        assert arrays["out"].tobytes() == initial and host.dispatch_count == 0
    else:
        gather_dispatch.dispatch(host, entry, table, len(table), expected.size, launch)
        assert words(np, arrays["out"]) == words(np, expected)
        assert host.dispatch_count == 1
        assert (
            hashlib.sha256(arrays["out"].tobytes()).hexdigest() == event["outputHash"]
        )


@pytest.mark.parametrize("native", (False, True))
def test_small_int_scatter_records_require_the_complete_profile(native):
    records = []
    for index, case in enumerate(CASES):
        *_, expected = workloads.reference(np, case)
        records.append(
            {
                **case,
                "actual": words(np, expected),
                "shape": list(expected.shape),
                "resultDtype": "mlx.core." + case["dtype"],
                "inputUnchanged": True,
                "dispatchStart": index if native else 0,
                "dispatchCount": int(native),
            }
        )
    verify_scatter.validate_records(np, records, native=native, small_int=True)
    with pytest.raises(ValueError):
        verify_scatter.validate_records(np, records, native=native, half=True)
    with pytest.raises(ValueError):
        verify_scatter.validate_records(np, records[:-1], native=native, small_int=True)
    records[0]["actual"][0] ^= 1
    with pytest.raises(ValueError):
        verify_scatter.validate_records(np, records, native=native, small_int=True)


def test_small_int_scatter_profile_names_its_evidence():
    args = SimpleNamespace(half=False, float32=False, small_int=True)
    assert verify_scatter.profile(args) == "small-int"


def test_small_int_scatter_backend_accepts_every_packed_integer():
    source = (
        Path(verify_scatter.__file__).resolve().parent / "scatter.inc"
    ).read_text()
    start = source.index("void Scatter::eval_gpu")
    check = source[start : source.index("throw std::invalid_argument", start)]
    for dtype in (*workloads.SMALL_INT_TYPES, *workloads.HALF_TYPES):
        assert f"out.dtype() != {dtype}" in check


def test_small_int_scatter_runs_natively_on_every_ci_target():
    import yaml

    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[5]
            / ".github/workflows/demo-project-testing.yml"
        ).read_text()
    )
    job = workflow["jobs"]["scatter-host"]
    rows = job["strategy"]["matrix"]["include"]
    assert {
        (row["target"], row["os"]) for row in rows if row["profile"] == "small-int"
    } == {
        ("metal", "macos-26"),
        ("directx", "windows-2025"),
        ("opengl", "ubuntu-24.04"),
    }
    steps = {step.get("name"): step for step in job["steps"]}
    run = steps["Execute general scatter host operations"]["run"]
    assert "matrix.profile == 'small-int' && '--small-int'" in run
    for trigger in ("pull_request", "push"):
        assert_paths_covered(
            workflow.get("on", workflow.get(True))[trigger].get("paths", ["**"]),
            "demos/integrations/mlx/tests/host/test_portable_small_int_scatter.py",
        )


def test_small_int_scatter_lane_types_cover_each_width():
    assert {layout.lane_ctype(dtype) for dtype in workloads.SMALL_INT_TYPES} == {
        ctypes.c_uint8,
        ctypes.c_uint16,
    }
    assert set(itertools.chain(layout.HALF_TYPES, layout.SMALL_INTEGER_TYPES)) == set(
        layout.PACKED_TYPES
    )


@pytest.mark.parametrize(
    "fault", (None, "entry", "target", "root", "unrouted", "copy", "identity")
)
def test_small_int_scatter_audits_its_lazy_word_copies(tmp_path, monkeypatch, fault):
    """Lazy word copies are recomputed; every other event keeps its identity."""
    case = next(case for case in CASES if case["id"] == "int8-sum-odd")
    event = scatter_event(tmp_path, case, "metal")
    packages = tmp_path / "packages"
    copy_root = packages / "copies" / "key" / "package"
    copy = {
        "entry": "ggn2_dynamic_copyuint8uint8",
        "target": "metal",
        "packageRoot": str(copy_root.resolve()),
        "copyMetadata": {"kernel": "ggn2_dynamic_copy"},
    }
    installed = {"entry": "v_copyint32int64", "target": "metal", "packageRoot": "x"}
    if fault == "entry":
        # Only the copy families translate into the lazy copy cache.
        copy["entry"] = "arangeint32"
    elif fault == "target":
        copy["target"] = "opengl"
    elif fault == "unrouted":
        copy["packageRoot"] = str(tmp_path / "elsewhere" / "key" / "package")
    audited, identified = [], []

    def audit(event, cache):
        assert cache.directory == packages / "copies"
        if fault == "copy":
            raise ValueError("Copy result 0 differs")
        audited.append(event["entry"])

    def identity(events, target):
        if fault == "identity" or any("copyMetadata" in item for item in events):
            raise ValueError("Bitwise native compiler identity is incomplete")
        identified.extend(item["entry"] for item in events)

    monkeypatch.setattr(scatter_evidence.copy_evidence, "audit_event", audit)
    monkeypatch.setattr(
        scatter_evidence,
        "CopyPackageCache",
        lambda root, directory, target: SimpleNamespace(directory=directory),
    )
    monkeypatch.setattr(
        "demos.integrations.mlx.portable_host.verify_bitwise.verify_native_identity",
        identity,
    )
    monkeypatch.setattr(workloads, "small_int_cases", lambda: iter([case]))
    *_, expected = workloads.reference(np, case)
    records = [
        {
            **case,
            "actual": words(np, expected),
            "shape": list(expected.shape),
            "resultDtype": "mlx.core." + case["dtype"],
            "inputUnchanged": True,
            "dispatchCount": 3,
            "dispatchStart": 0,
        }
    ]
    upstream = {
        "testsRun": 1,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "dispatchStart": 3,
        "dispatchCount": 1,
    }
    trace = [copy, installed, event, event]
    arguments = {
        "small_int": True,
        "mlx_root": None if fault == "root" else tmp_path,
        "packages": packages,
    }
    if fault:
        with pytest.raises(ValueError):
            scatter_evidence.validate(np, records, trace, upstream, **arguments)
    else:
        assert scatter_evidence.validate(np, records, trace, upstream, **arguments) == {
            "target": "metal",
            "scatterDispatchCount": 1,
            "upstreamScatterDispatchCount": 1,
        }
        assert audited == ["ggn2_dynamic_copyuint8uint8"]
        assert identified == ["v_copyint32int64"]
