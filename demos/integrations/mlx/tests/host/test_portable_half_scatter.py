"""Binary16 and bfloat16 scatters pack atomic words and keep exact evidence."""

import hashlib
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from crosstl.project.runtime_value_encoding import (
    BFLOAT16_BITS,
    FLOAT16_BITS,
    FLOAT32_BITS,
)
from demos.integrations.mlx.portable_host import (
    bfloat_storage,
    gather_dispatch,
    half_storage,
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

CASES = list(workloads.half_cases())
ACTIVE = [case for case in CASES if case["layout"] != "empty"]
TARGETS = ("metal", "directx", "opengl")
# (precision, minimum normal exponent, maximum exponent) of each format.
FORMATS = {"float16": (11, -14, 15), "bfloat16": (8, -126, 127)}


def exact(word, dtype):
    """Decode a raw 16-bit word into an exact rational, or None for NaN/inf."""
    value = float(workloads.half_values(np, [word], dtype)[0])
    return Fraction(value) if np.isfinite(value) else None


def round_exact(value, dtype):
    """Round an exact rational to the nearest 16-bit word, ties to even."""
    precision, minimum, maximum = FORMATS[dtype]
    sign = 0x8000 if value < 0 else 0
    magnitude = abs(value)
    if magnitude == 0:
        return sign
    exponent = 0
    while Fraction(2) ** exponent > magnitude:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= magnitude:
        exponent += 1
    # Subnormals share the quantum of the smallest normal binade.
    quantum = Fraction(2) ** (max(minimum, exponent) - precision + 1)
    units, remainder = divmod(magnitude, quantum)
    units = int(units)
    if remainder * 2 > quantum or (remainder * 2 == quantum and units % 2):
        units += 1
    result = units * quantum
    largest = (2 - Fraction(2) ** (1 - precision)) * Fraction(2) ** maximum
    if result > largest:
        return sign | (0x7C00 if dtype == "float16" else 0x7F80)
    if dtype == "float16":
        word = int(np.array([float(result)], dtype=np.float16).view(np.uint16)[0])
    else:
        bits = np.array([float(result)], dtype=np.float32).view(np.uint32)[0]
        word = int(bits) >> 16
    assert exact(word, dtype) == result
    return word | sign


def independent_update(dtype, operation, current, update):
    if operation == "none":
        return update
    a, b = exact(update, dtype), exact(current, dtype)
    assert a is not None and b is not None
    if operation == "sum":
        return round_exact(a + b, dtype)
    if operation == "prod":
        return current if b == 0 else round_exact(a * b, dtype)
    if operation == "min":
        return update if a < b else current
    return update if a > b else current


def test_half_scatter_case_inventory():
    assert len(CASES) == len({case["id"] for case in CASES}) == 174
    assert len(ACTIVE) == 164
    assert {case["dtype"] for case in CASES} == {"float16", "bfloat16"}
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
    assert {"odd", "payloads", "rounding", "work32"} <= {
        case["layout"] for case in CASES
    }


def test_half_scatter_preserves_float32_workload_bytes():
    digest = hashlib.sha256()
    for case in workloads.float_cases():
        digest.update(case["id"].encode())
        source, indices, updates, expected = workloads.reference(np, case)
        for array in (source, *indices, updates, expected):
            digest.update(str((str(array.dtype), array.shape, array.strides)).encode())
            digest.update(array.tobytes())
    assert (
        digest.hexdigest()
        == "7db33322d58874a8e551a51cef1c8a0f500b9649e74c0e4600c2cfa8e8da7706"
    )


def test_half_scatter_profiles_are_exclusive():
    with pytest.raises(ValueError):
        workloads.profile_cases(float32=True, half=True)
    assert [case["id"] for case in workloads.profile_cases(half=True)] == [
        case["id"] for case in CASES
    ]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_half_scatter_reference_matches_exact_rounding_model(case):
    source, indices, updates, expected = workloads.reference(np, case)
    assert source.dtype == updates.dtype == expected.dtype == np.uint16
    actual = source.copy()
    for coordinate in np.ndindex(indices[0].shape):
        prefix = tuple(int(index[coordinate]) for index in indices)
        for offset in np.ndindex(source.shape[len(prefix) :]):
            address = prefix + offset
            actual[address] = independent_update(
                case["dtype"],
                case["operation"],
                int(actual[address]),
                int(updates[coordinate + offset]),
            )
    assert words(np, actual) == words(np, expected)
    if case["layout"] in ("payloads", "rounding"):
        assert words(np, source) != words(np, expected)


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case["operation"] in ("sum", "prod")],
    ids=lambda case: case["id"],
)
def test_half_scatter_accumulations_do_not_depend_on_order(case):
    """Native atomics may apply updates in any order; only rounding cases round."""
    source, indices, updates, _ = workloads.reference(np, case)
    groups = {}
    for coordinate in np.ndindex(indices[0].shape):
        prefix = tuple(int(index[coordinate]) for index in indices)
        for offset in np.ndindex(source.shape[len(prefix) :]):
            groups.setdefault(prefix + offset, []).append(updates[coordinate + offset])
    precision = FORMATS[case["dtype"]][0]
    for address, group in groups.items():
        initial = exact(int(source[address]), case["dtype"])
        values = [exact(int(value), case["dtype"]) for value in group]
        if case["layout"] == "rounding":
            total = initial + values[0]
            assert len(values) == 1
            assert exact(round_exact(total, case["dtype"]), case["dtype"]) != total
        elif case["operation"] == "sum":
            assert all((value * 4).denominator == 1 for value in [initial, *values])
            assert 4 * (abs(initial) + sum(map(abs, values))) < 2**precision
        elif case["layout"] == "alias":
            assert len(values) == 1
            product = initial * values[0]
            assert exact(round_exact(product, case["dtype"]), case["dtype"]) == product
        else:
            assert set(map(abs, values)) <= {Fraction(1, 2), Fraction(1), Fraction(2)}
            _, minimum, maximum = FORMATS[case["dtype"]]
            lower = abs(initial) * np.prod(
                [min(Fraction(1), abs(value)) for value in values], dtype=object
            )
            upper = abs(initial) * np.prod(
                [max(Fraction(1), abs(value)) for value in values], dtype=object
            )
            # Power-of-two factors stay exact while every partial stays normal.
            assert Fraction(2) ** minimum <= lower <= upper < Fraction(2) ** maximum


@pytest.mark.parametrize("dtype", workloads.HALF_TYPES)
def test_half_scatter_rounding_profile_covers_ties_and_binades(dtype):
    def binade(value):
        exponent = 0
        while Fraction(2) ** exponent > abs(value):
            exponent -= 1
        while Fraction(2) ** (exponent + 1) <= abs(value):
            exponent += 1
        return exponent

    case = next(case for case in CASES if case["id"] == f"{dtype}-sum-rounding")
    source, _, updates, _ = workloads.reference(np, case)
    precision = FORMATS[dtype][0]
    ties = crossings = negatives = 0
    for initial, update in zip(source.reshape(-1), updates.reshape(-1)):
        total = exact(int(initial), dtype) + exact(int(update), dtype)
        rounded = exact(round_exact(total, dtype), dtype)
        quantum = Fraction(2) ** (binade(total) - precision + 1)
        ties += abs(rounded - total) * 2 == quantum
        crossings += binade(rounded) != binade(total)
        negatives += total < 0
    assert ties >= 4 and crossings >= 1 and negatives >= 2


@pytest.mark.parametrize("dtype", workloads.HALF_TYPES)
def test_half_scatter_payloads_store_raw_words(dtype):
    case = next(case for case in CASES if case["id"] == f"{dtype}-none-payloads")
    _, _, updates, expected = workloads.reference(np, case)
    lanes = sorted(words(np, updates))
    assert lanes == sorted(workloads.HALF_PAYLOADS[dtype])
    assert sorted(words(np, expected)) == lanes
    values = workloads.half_values(np, lanes, dtype)
    assert np.isnan(values).sum() == 2 and np.isinf(values).sum() == 2
    assert {0x0000, 0x8000} <= set(lanes)
    if dtype == "float16":
        assert any(0 < lane & 0x7FFF < 0x0400 for lane in lanes)


@pytest.mark.parametrize(
    ("dtype", "value", "word"),
    (
        ("bfloat16", 0x3F808000, 0x3F80),
        ("bfloat16", 0x3F818000, 0x3F82),
        ("bfloat16", 0x3F808001, 0x3F81),
        ("bfloat16", 0xBF817FFF, 0xBF81),
        ("bfloat16", 0x7F7FFFFF, 0x7F80),
        ("bfloat16", 0x7F800001, 0x7FC0),
        ("bfloat16", 0xFF812345, 0xFFC1),
        ("float16", 0x3F801000, 0x3C00),
        ("float16", 0x3F803000, 0x3C02),
        ("float16", 0x477FF000, 0x7C00),
    ),
)
def test_half_scatter_rounds_to_nearest_even(dtype, value, word):
    values = np.array([value], dtype=np.uint32).view(np.float32)
    assert int(workloads.round_half(np, values, dtype)[0]) == word


@pytest.mark.parametrize("dtype", workloads.HALF_TYPES)
def test_half_scatter_combine_stores_only_while_the_condition_holds(dtype):
    one, two, zero, negative_zero = (
        workloads.half_words(np, [value], dtype)[0] for value in (1, 2, 0, -0.0)
    )
    nan = 0x7E00 if dtype == "float16" else 0x7FC0
    assert workloads.combine(np, dtype, "prod", [zero], [two]).tolist() == [zero]
    assert workloads.combine(np, dtype, "prod", [negative_zero], [nan]).tolist() == [
        negative_zero
    ]
    assert workloads.combine(np, dtype, "min", [one], [one]).tolist() == [one]
    assert workloads.combine(np, dtype, "max", [zero], [negative_zero]).tolist() == [
        zero
    ]
    assert workloads.combine(np, dtype, "min", [two], [nan]).tolist() == [two]
    assert workloads.combine(np, dtype, "max", [one], [two]).tolist() == [two]
    assert workloads.combine(np, dtype, "none", [one], [nan]).tolist() == [nan]
    total = workloads.combine(np, dtype, "sum", [one], [nan])
    assert np.isnan(workloads.half_values(np, total, dtype)).all()


def test_half_scatter_packs_low_lane_first_and_pads_odd_tails():
    assert layout.pack_lanes([0x1111, 0x2222, 0x3333], 0xABCD) == [
        0x22221111,
        0xABCD3333,
    ]
    assert layout.unpack_lanes([0x22221111, 0xABCD3333], 3) == (
        [0x1111, 0x2222, 0x3333],
        [0xABCD],
    )
    assert layout.storage_count("float16", 3) == 2
    assert layout.storage_count("bfloat16", 4) == 2
    assert layout.storage_count("float32", 3) == 3
    for lanes in ([-1], [0x10000], [True], [0.5]):
        with pytest.raises(ValueError):
            layout.pack_lanes(lanes, 0)
    for words_, count in (([2**32], 1), ([-1], 1), ([True], 1), ([0, 0], 1)):
        with pytest.raises(ValueError):
            layout.unpack_lanes(words_, count)
    assert layout.PADDING["float16"] != layout.PADDING["bfloat16"]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", ACTIVE, ids=lambda case: case["id"])
def test_half_scatter_audit_reconstructs_raw_uploads(tmp_path, target, case):
    event = scatter_event(tmp_path, case, target)
    source, indices, updates, expected = workloads.reference(np, case)
    initial, actual_indices, actual_updates, actual = scatter_evidence.audit_event(
        np, event
    )
    assert words(np, initial) == words(np, source)
    assert all(np.array_equal(a, b) for a, b in zip(indices, actual_indices))
    assert words(np, actual_updates) == words(np, updates)
    assert actual == words(np, expected)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", workloads.HALF_TYPES)
def test_half_scatter_events_carry_reflected_storage(tmp_path, target, dtype):
    case = next(case for case in CASES if case["id"] == f"{dtype}-sum-odd")
    event = scatter_event(tmp_path, case, target)
    out, updates = event["inputs"]["out"], event["inputs"]["updates"]
    assert out["dtype"] == "uint32" and "encoding" not in out
    assert out["shape"] == [5 + len(runtime.COPY_GUARD), 1]
    assert out["values"][4] >> 16 == layout.PADDING[dtype]
    physical = runtime.physical_dtype(dtype, target)
    assert updates["dtype"] == physical
    assert (
        updates.get("encoding")
        == {
            ("float16", "metal"): FLOAT16_BITS,
            ("float16", "directx"): FLOAT16_BITS,
            ("float16", "opengl"): FLOAT32_BITS,
            ("bfloat16", "metal"): BFLOAT16_BITS,
            ("bfloat16", "directx"): None,
            ("bfloat16", "opengl"): FLOAT32_BITS,
        }[dtype, target]
    )


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", workloads.HALF_TYPES)
@pytest.mark.parametrize(
    "fault",
    (
        "upload-padding",
        "readback-padding",
        "readback-length",
        "readback-large",
        "readback-value",
        "upload-word",
        "update-carrier",
        "update-encoding",
        "out-encoding",
        "out-dtype",
        "storage-encoding",
        "guard",
        "hash",
    ),
)
def test_half_scatter_rejects_corrupt_evidence(tmp_path, target, dtype, fault):
    case = next(case for case in CASES if case["id"] == f"{dtype}-max-odd")
    event = scatter_event(tmp_path, case, target)
    request = event["details"]["request"]["buffers"]
    out, updates = event["inputs"]["out"], event["inputs"]["updates"]
    if fault == "upload-padding":
        out["values"][4] ^= 0x10000
    elif fault == "readback-padding":
        event["scatterValues"][-1] ^= 0x10000
    elif fault == "readback-length":
        event["scatterValues"].pop()
    elif fault == "readback-large":
        event["scatterValues"][0] = 2**32
    elif fault == "readback-value":
        event["scatterValues"][0] ^= 1
    elif fault == "upload-word":
        out["values"][0] = -1
    elif fault == "update-carrier":
        # OpenGL float carriers must widen a 16-bit word exactly.
        updates["values"][0] = 0x3F800001 if target == "opengl" else 0x10000
    elif fault == "update-encoding":
        updates["encoding"] = request["updates"]["encoding"] = "unknown"
    elif fault == "out-encoding":
        out["encoding"] = request["out"]["encoding"] = FLOAT32_BITS
    elif fault == "out-dtype":
        out["dtype"] = request["out"]["dtype"] = "float32"
    elif fault == "storage-encoding":
        scalar = request["updates"]["binding"]["metadata"]["scalarLayout"]
        if "storageEncoding" in scalar:
            scalar.pop("storageEncoding")
        else:
            scalar["storageEncoding"] = {
                "encoding": "ieee754-binary16",
                "logicalElementType": "float16",
            }
    elif fault == "guard":
        event["scatterGuardValues"][0] = 0
    else:
        event["outputHash"] = "0" * 64
    with pytest.raises(ValueError):
        scatter_evidence.audit_event(np, event)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", workloads.HALF_TYPES)
@pytest.mark.parametrize(
    "fault",
    (None, "missing", "logical", "guard", "encoding", "physical", "count", "module"),
)
def test_half_scatter_audits_its_16_bit_copies(tmp_path, target, dtype, fault):
    storage = half_storage if dtype == "float16" else bfloat_storage
    entry = runtime.HALF_COPY_ENTRY if dtype == "float16" else runtime.BFLOAT_COPY_ENTRY
    key = "halfStorage" if dtype == "float16" else "bfloatStorage"
    event = scatter_event(tmp_path, CASES[0], target)
    lanes = [0x0000, 0x8000, 0x3C00, 0x7BFF]
    event.update(entry=entry, threads=len(lanes))
    event[key] = {
        "logicalType": dtype,
        "physicalType": runtime.physical_dtype(dtype, target),
        "encoding": storage.encoding(target),
        "values": storage.pack(lanes, target),
        "guardValues": storage.pack(storage.GUARD, target),
        "logicalWords": lanes,
    }
    record = event[key]
    if fault == "missing":
        del event[key]
    elif fault == "logical":
        record["logicalWords"][0] ^= 1
    elif fault == "guard":
        record["guardValues"][0] = storage.pack([0], target)[0]
    elif fault == "encoding":
        record["encoding"] = "unknown"
    elif fault == "physical":
        record["physicalType"] = "uint32"
    elif fault == "count":
        event["threads"] += 1
    elif fault == "module":
        Path(event["details"]["module"]["file"]).write_bytes(b"changed")
    if fault:
        with pytest.raises(ValueError):
            scatter_evidence.audit_word_copy(event)
    else:
        scatter_evidence.audit_word_copy(event)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", workloads.HALF_TYPES)
@pytest.mark.parametrize("operation", ("none", "sum", "prod", "min", "max"))
def test_half_scatter_advertises_native_targets(target, dtype, operation):
    host = SimpleNamespace(target=target, descriptors={}, gathers=object())
    entry = f"scatter{dtype}uint32_{operation}_1_updc_false_nwork4_int"
    assert runtime.HostRuntime._entry_available(host, entry.encode()) == 1


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", workloads.HALF_TYPES)
@pytest.mark.parametrize(
    "fault", (None, "padding", "guard", "dtype", "encoding", "large", "member")
)
def test_half_scatter_dispatch_round_trips_packed_words(
    tmp_path, monkeypatch, target, dtype, fault
):
    case = next(case for case in CASES if case["id"] == f"{dtype}-sum-odd")
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
    if fault == "member":
        next(value for value in descriptor["bindings"] if value["name"] == "out")[
            "scalarLayout"
        ]["structMembers"][0]["physicalType"] = "float"
    packed = layout.pack_lanes(words(np, arrays["out"]), layout.PADDING[dtype])

    def request(_descriptor, _directory, inputs, outputs, launch, **kwargs):
        assert inputs["out"]["values"] == packed + runtime.COPY_GUARD
        assert inputs["out"]["dtype"] == "uint32" and "encoding" not in inputs["out"]
        assert inputs["updates"] == event["inputs"]["updates"]
        return SimpleNamespace(outputs=outputs)

    def execute(_host, request):
        values = layout.pack_lanes(words(np, expected), layout.PADDING[dtype])
        output = {**request.outputs["out"], "values": values + runtime.COPY_GUARD}
        if fault == "padding":
            output["values"][len(values) - 1] ^= 0x10000
        elif fault == "guard":
            output["values"][-1] = 0
        elif fault == "dtype":
            output["dtype"] = "float32"
        elif fault == "encoding":
            output["encoding"] = FLOAT32_BITS
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
def test_half_scatter_records_require_the_complete_profile(native):
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
    verify_scatter.validate_records(np, records, native=native, half=True)
    with pytest.raises(ValueError):
        verify_scatter.validate_records(np, records, native=native, float32=True)
    with pytest.raises(ValueError):
        verify_scatter.validate_records(np, records[:-1], native=native, half=True)


def test_half_scatter_profile_names_its_evidence():
    assert verify_scatter.profile(SimpleNamespace(half=True, float32=False)) == "half"
    assert verify_scatter.UPSTREAM_TESTS == (
        "test_array.TestArray.test_setitem_with_list",
    )


def test_half_scatter_runs_natively_on_every_ci_target():
    import yaml

    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[5]
            / ".github/workflows/demo-project-testing.yml"
        ).read_text()
    )
    job = workflow["jobs"]["scatter-host"]
    rows = job["strategy"]["matrix"]["include"]
    assert {(row["target"], row["os"]) for row in rows if row["profile"] == "half"} == {
        ("metal", "macos-26"),
        ("directx", "windows-2025"),
        ("opengl", "ubuntu-24.04"),
    }
    steps = {step.get("name"): step for step in job["steps"]}
    copies = steps["Translate 16-bit copy prerequisites for half scatter"]
    assert copies["if"] == "matrix.profile == 'half'"
    assert "for family in half bfloat" in copies["run"]
    assert "matrix.target" in copies["run"]
    run = steps["Execute general scatter host operations"]["run"]
    assert "matrix.profile == 'half' && '--half --half-packages" in run
    for name in ("half", "bfloat"):
        assert f".mlx-portable-scatter/{name}-packages" in run
    for trigger in ("pull_request", "push"):
        assert_paths_covered(
            workflow.get("on", workflow.get(True))[trigger].get("paths", ["**"]),
            "demos/integrations/mlx/tests/host/test_portable_half_scatter.py",
        )
