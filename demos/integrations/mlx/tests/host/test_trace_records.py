"""Compact host dispatch records keep structure and digest long value arrays."""

import hashlib
import json
import struct
from types import SimpleNamespace

import pytest

from crosstl.project import runtime_values
from demos.integrations.mlx.portable_host import runtime, trace_records


def _digest(values):
    encoded = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _int64_digest(values):
    encoded = b"int64\0" + struct.pack(f"<{len(values)}q", *values)
    return hashlib.sha256(encoded).hexdigest()


class _Float(float):
    def __repr__(self):
        return "not json"


class _Int(int):
    def __int__(self):
        return 0


def test_compaction_replaces_only_long_scalar_arrays():
    long_ints = list(range(trace_records.COMPACT_TRACE_LIMIT + 1))
    long_floats = [0.5, float("nan"), None, True] * 100
    nested = [[1, 2]] * (trace_records.COMPACT_TRACE_LIMIT + 1)
    event = {
        "entry": "e",
        "workgroupCount": [1, 2, 1],
        "inputs": {"a": {"dtype": "int32", "values": long_ints}},
        "values": tuple(long_floats),
        "nested": nested,
        "short": list(range(trace_records.COMPACT_TRACE_LIMIT)),
    }
    compact = trace_records.compact_trace_values(event)
    assert compact["entry"] == "e" and compact["workgroupCount"] == [1, 2, 1]
    assert compact["inputs"]["a"] == {
        "dtype": "int32",
        "values": {
            "compactedValues": {
                "length": len(long_ints),
                "sha256": _int64_digest(long_ints),
            }
        },
    }
    # Mixed arrays keep their JSON digest.
    assert compact["values"]["compactedValues"] == {
        "length": 400,
        "sha256": _digest(long_floats),
    }
    assert compact["nested"] == nested
    assert compact["short"] == event["short"]
    assert event["inputs"]["a"]["values"] == long_ints


N = trace_records.COMPACT_TRACE_LIMIT + 1


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([0.25, -3.5] * N, tuple([0.25, -3.5] * N)),
        ([0.25, -3.5] * N, [_Float(0.25), -3.5] * N),
        ([7, -2] * N, [_Int(7), -2] * N),
        ([7, -2] * N, [7.0, -2.0] * N),
        ([True] * N, [1] * N),
        ([0.0] * N, [-0.0] * N),
        ([float("nan")] * N, [float("nan")] * N),
        ([float("nan")] * N, [float("inf")] * N),
        ([1.0] * N + [float("inf")], [1.0] * N + [float("inf")]),
        ([2**63] * N, [2**63] * N),
        ([2**63] * N, [2**63 - 1] * N),
        ([1, 2.5] * N, [1, 2.5] * N),
        (["a"] * N, ["a"] * N),
        ([None] * N, [None] * N),
    ],
)
def test_value_digests_agree_exactly_when_the_json_does(first, second):
    # json writes subclasses through their base type, whatever they override.
    same = json.dumps(first) == json.dumps(second)
    assert (
        trace_records.value_digest(first) == trace_records.value_digest(second)
    ) is same


def test_value_digests_of_one_exact_type_hash_machine_words():
    floats = [0.5, -2.0] * N
    assert (
        trace_records.value_digest(floats)
        == hashlib.sha256(
            b"float64\0" + struct.pack(f"<{len(floats)}d", *floats)
        ).hexdigest()
    )
    assert (
        trace_records.value_digest([True, False] * N)
        == hashlib.sha256(b"bool\0" + bytes([1, 0] * N)).hexdigest()
    )
    # Out of the word range, an array keeps its JSON digest.
    wide = [2**64 - 1] * N
    assert trace_records.value_digest(wide) == _digest(wide)


def test_trace_stream_reads_one_record_at_a_time(tmp_path):
    path = tmp_path / "trace.jsonl"
    records = [{"entry": "a", "values": [1, 2]}, {"entry": "b"}]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    stream = trace_records.TraceStream(path)
    iterator = iter(stream)
    assert next(iterator) == records[0] and stream.records == 1
    assert list(iterator) == records[1:]
    assert stream.summary() == {
        "records": 2,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("retain", (True, False, None))
def test_write_trace_follows_the_host_retention_flag(tmp_path, retain):
    values = list(range(1000))
    host = SimpleNamespace(trace=tmp_path / "trace.jsonl")
    if retain is not None:
        host.trace_values = retain
    trace_records.write_trace(host, {"entry": "e", "values": values})
    trace_records.write_trace(host, {"entry": "f"})
    first, second = (json.loads(line) for line in host.trace.read_text().splitlines())
    if retain is False:
        assert first["values"] == {
            "compactedValues": {"length": 1000, "sha256": _int64_digest(values)}
        }
    else:
        assert first["values"] == values
    assert second == {"entry": "f"}


def test_prepared_trace_records_are_written_as_given(tmp_path):
    host = SimpleNamespace(trace=tmp_path / "trace.jsonl", trace_values=False)
    event = {"entry": "e", "keys": list(range(300))}
    trace_records.write_trace(host, event, prepared=True)
    assert json.loads(host.trace.read_text()) == event


@pytest.mark.parametrize(
    "dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")
)
def test_integer_array_digests_equal_their_list_digests(dtype):
    np = pytest.importorskip("numpy")
    info = np.iinfo(dtype)
    high = min(int(info.max), 2**63 - 1)
    values = np.array(
        [info.min, 0, 1, high, *(n % 100 for n in range(400))], dtype=dtype
    )
    expected = trace_records.value_digest(values.tolist())
    assert trace_records.integer_digest(np, values) == expected
    assert trace_records.integer_digest(np, values, chunk=7) == expected
    assert trace_records.integer_digest(np, values.reshape(2, -1)) == expected
    compact = trace_records.compact_trace_values(values.tolist())
    assert trace_records.matches_integers(np, compact, values)
    assert trace_records.matches_integers(np, values.tolist(), values)
    assert trace_records.recorded_length(compact) == values.size


def test_integer_matches_reject_other_values_and_shapes():
    np = pytest.importorskip("numpy")
    values = np.arange(300, dtype=np.int32)
    compact = trace_records.compact_trace_values(values.tolist())
    changed = values.copy()
    changed[5] += 1
    assert not trace_records.matches_integers(np, compact, changed)
    assert not trace_records.matches_integers(np, compact, values[:-1])
    assert not trace_records.matches_integers(np, {**compact, "x": 1}, values)
    # Equal JSON is not enough: Booleans and floats are not integers.
    assert not trace_records.matches_integers(np, [True, False], np.array([1, 0]))
    assert not trace_records.matches_integers(np, [1.0, 0], np.array([1, 0]))
    assert not trace_records.matches_integers(np, None, np.array([1]))
    short = {"compactedValues": {"length": 2, "sha256": "0" * 64}}
    short["compactedValues"]["sha256"] = trace_records.integer_digest(
        np, np.array([1, 0])
    )
    # Lists no longer than the compact limit are never digested.
    assert not trace_records.matches_integers(np, short, np.array([1, 0]))
    with pytest.raises(ValueError):
        trace_records.integer_digest(np, np.array([2**63], dtype=np.uint64))
    with pytest.raises(ValueError):
        trace_records.integer_digest(np, np.array([1.0]))
    with pytest.raises(ValueError):
        trace_records.recorded_length({"values": [1]})


@pytest.mark.parametrize("value", (0, 1, "false", None))
def test_host_runtime_requires_a_boolean_trace_retention_flag(tmp_path, value):
    with pytest.raises(ValueError, match="Trace value retention"):
        runtime.HostRuntime(tmp_path, tmp_path / "trace.jsonl", trace_values=value)


def test_compaction_replaces_only_long_strings():
    limit = trace_records.COMPACT_STRING_LIMIT
    observed = "ab" * limit
    event = {
        "entry": "e" * limit,
        "details": {"inputObservations": {"a": {"observedBytes": observed}}},
        "names": ["x" * (limit + 1), "y"],
    }
    compact = trace_records.compact_trace_values(event)
    assert compact["entry"] == event["entry"]
    assert compact["details"]["inputObservations"]["a"]["observedBytes"] == {
        "compactedString": {
            "length": 2 * limit,
            "sha256": hashlib.sha256(observed.encode()).hexdigest(),
        }
    }
    assert compact["names"][0]["compactedString"]["length"] == limit + 1
    assert compact["names"][1] == "y"
    assert event["details"]["inputObservations"]["a"]["observedBytes"] == observed


def test_trace_summary_streams_only_the_checked_fields(tmp_path):
    path = tmp_path / "trace.jsonl"
    records = [
        {"entry": "a", "target": "metal", "threads": 3, "details": {"x": [1] * 9}},
        {"entry": "b", "target": "opengl", "inputs": {}},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    assert trace_records.read_trace_summary(path) == [
        {"entry": "a", "target": "metal", "threads": 3},
        {"entry": "b", "target": "opengl"},
    ]
    assert trace_records.read_trace_summary(path, keys=("details",)) == [
        {"details": {"x": [1] * 9}},
        {},
    ]


def test_digests_from_known_item_types_equal_the_scanned_digests():
    class Wide(int):
        pass

    lists = (
        [True, False] * 200,
        list(range(-300, 300)),
        [1 << 70, *range(400)],
        [0.5 * i for i in range(400)],
        [float("nan"), *map(float, range(400))],
        [1, 2.0, None, "x", True] * 80,
        [Wide(i) for i in range(400)],
        [*range(399), Wide(7)],
    )
    for values in lists:
        types = runtime_values.scalar_types(values)
        assert trace_records.value_digest(values, types=types) == (
            trace_records.value_digest(values)
        )
        compact = trace_records.compact_trace_values({"v": values})["v"]
        assert compact == {
            "compactedValues": {
                "length": len(values),
                "sha256": trace_records.value_digest(tuple(values)),
            }
        }
