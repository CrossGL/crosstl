"""Compact host dispatch records keep structure and digest long value arrays."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import runtime, trace_records


def _digest(values):
    encoded = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
            "compactedValues": {"length": len(long_ints), "sha256": _digest(long_ints)}
        },
    }
    assert compact["values"]["compactedValues"] == {
        "length": 400,
        "sha256": _digest(long_floats),
    }
    assert compact["nested"] == nested
    assert compact["short"] == event["short"]
    assert event["inputs"]["a"]["values"] == long_ints


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
            "compactedValues": {"length": 1000, "sha256": _digest(values)}
        }
    else:
        assert first["values"] == values
    assert second == {"entry": "f"}


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
