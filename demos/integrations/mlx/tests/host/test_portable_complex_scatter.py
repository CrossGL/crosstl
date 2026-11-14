"""Complex64 scatter sums bind two float32 atomic lanes per output element."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from demos.integrations.mlx.portable_host import (
    gather_dispatch,
    gather_packages,
    runtime,
    scatter_evidence,
)
from demos.integrations.mlx.portable_host import scatter_layout as layout
from demos.integrations.mlx.portable_host.gather_workloads import words
from demos.integrations.mlx.tests.host.test_portable_scatter import (
    launch_buffers,
    scatter_event,
)

TARGETS = ("metal", "directx", "opengl")
CASE = {"dtype": "complex64", "index_dtype": "int32", "operation": "sum"}


def complex_launch():
    """Repeated destinations along one index axis, as upstream's at[].add uses."""
    source = np.array([1 + 10j, 2 + 20j, 3 + 30j, 4 + 40j], dtype=np.complex64)
    indices = [np.array([3, 1, 3, 3, 0], dtype=np.int32)]
    updates = np.array([1 + 3j, 0.5 - 2j, 1 + 3j, -2 + 0.25j, 8j], dtype=np.complex64)
    expected = source.copy()
    for index, update in zip(indices[0], updates):
        expected[index] += update
    return launch_buffers(CASE, source, indices, updates, expected)


@pytest.mark.parametrize("index_dtype", ("int32", "uint32", "int64", "uint64"))
@pytest.mark.parametrize("contiguous", ("true", "false"))
def test_complex_scatter_signature_accepts_only_sums(index_dtype, contiguous):
    entry = f"scattercomplex64{index_dtype}_sum_2_updc_{contiguous}_nwork4_int"
    assert layout.signature(entry) == (
        "complex64",
        index_dtype,
        2,
        "sum",
        contiguous == "true",
        4,
    )
    assert gather_packages.signature(entry) == layout.signature(entry)
    for operation in ("none", "prod", "min", "max"):
        with pytest.raises(ValueError, match="Unsupported native general-scatter"):
            layout.signature(entry.replace("_sum_", f"_{operation}_"))


def test_complex_scatter_wraps_mlx_complex_sum():
    source = gather_packages.SOURCE_TYPES["complex64"]
    assert source == "complex64_t"
    assert layout.ATOMIC_LANES == {
        "int32": 1,
        "uint32": 1,
        "float32": 1,
        "complex64": 2,
    }
    for target in TARGETS:
        # mlx_atomic<complex64_t> is the generic atomic<uint> wrapper.
        assert layout.atomic_storage_dtype("complex64", target) == "uint32"


@pytest.mark.parametrize("target", TARGETS)
def test_complex_scatter_advertises_native_targets(target):
    host = SimpleNamespace(target=target, descriptors={}, gathers=object())
    entry = "scattercomplex64int64_sum_1_updc_true_nwork1_int"
    assert runtime.HostRuntime._entry_available(host, entry.encode()) == 1


def test_complex_scatter_bounds_output_words():
    entry, supplied, arrays, expected, execution = complex_launch()
    result = layout.validate(entry, supplied, expected.size, execution)
    # Four complex elements are eight atomic words.
    assert result["maximumIndex"] == 2 * expected.size - 1 == 7
    assert result["outputCount"] == 4 and result["updateCount"] == 5
    limit = layout.MAX_STORAGE_ELEMENTS // 2 + 1
    with pytest.raises(ValueError, match="output size"):
        layout.validate(entry, supplied, limit, execution)


@pytest.mark.parametrize("target", TARGETS)
def test_complex_scatter_audit_reconstructs_lane_sums(tmp_path, target):
    launch = complex_launch()
    expected = launch[3]
    event = scatter_event(tmp_path, CASE, target, launch=launch)
    assert event["inputs"]["out"]["dtype"] == "uint32"
    assert len(event["inputs"]["out"]["values"]) == 8 + len(runtime.COPY_GUARD)
    assert event["inputs"]["updates"]["dtype"] == "float32"
    assert event["inputs"]["updates"]["encoding"] == FLOAT32_BITS
    initial, indices, updates, actual = scatter_evidence.audit_event(np, event)
    assert initial.dtype == np.complex64 and updates.dtype == np.complex64
    assert words(np, initial) == words(np, launch[2]["out"])
    assert actual == words(np, expected)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "fault",
    ("lane", "readback-length", "guard", "hash", "update-layout", "update-word"),
)
def test_complex_scatter_rejects_corrupt_evidence(tmp_path, target, fault):
    event = scatter_event(tmp_path, CASE, target, launch=complex_launch())
    if fault == "lane":
        # One imaginary lane disagrees with the index-order sum.
        event["scatterValues"][7] ^= 1
    elif fault == "readback-length":
        event["scatterValues"] = event["scatterValues"][:4]
    elif fault == "guard":
        event["scatterGuardValues"][0] = 0
    elif fault == "hash":
        event["outputHash"] = "0" * 64
    elif fault == "update-layout":
        layout_record = event["details"]["request"]["buffers"]["updates"]["binding"][
            "metadata"
        ]["scalarLayout"]
        layout_record["elementStrideBytes"] = 4
    else:
        event["inputs"]["updates"]["values"][0] = -1
    with pytest.raises(ValueError):
        scatter_evidence.audit_event(np, event)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("fault", (None, "guard", "length", "negative"))
def test_complex_scatter_dispatch_binds_atomic_words(
    tmp_path, monkeypatch, target, fault
):
    entry, supplied, arrays, expected, execution = launch = complex_launch()
    event = scatter_event(tmp_path, CASE, target, launch=launch)
    descriptor = {
        "artifact": {},
        "bindings": [
            {"name": name, "scalarLayout": value["binding"]["metadata"]["scalarLayout"]}
            for name, value in event["details"]["request"]["buffers"].items()
        ],
    }

    def request(_descriptor, _directory, inputs, outputs, launch, **kwargs):
        assert inputs["out"] == event["inputs"]["out"]
        assert inputs["updates"] == event["inputs"]["updates"]
        return SimpleNamespace(outputs=outputs)

    def execute(_host, request):
        values = expected.view(np.uint32).tolist() + runtime.COPY_GUARD
        if fault == "guard":
            values[-1] = 0
        elif fault == "length":
            values = values[:4] + runtime.COPY_GUARD
        elif fault == "negative":
            values[0] = -1
        output = {**request.outputs["out"], "values": values}
        return SimpleNamespace(status="ok", outputs={"out": output}, details={})

    monkeypatch.setattr(
        gather_dispatch, "build_native_loader_dispatch_request", request
    )
    monkeypatch.setattr(gather_dispatch, "execute", execute)
    records = []
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
    launch_record = runtime.Launch(tuple(execution["workgroupCount"]), (1, 1, 1))
    initial = arrays["out"].tobytes()
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            gather_dispatch.dispatch(
                host, entry, table, len(table), expected.size, launch_record
            )
        assert arrays["out"].tobytes() == initial and host.dispatch_count == 0
        return
    gather_dispatch.dispatch(
        host, entry, table, len(table), expected.size, launch_record
    )
    assert words(np, arrays["out"]) == words(np, expected)
    assert host.dispatch_count == 1
    (record,) = records
    assert record["scatterValues"] == event["scatterValues"]
    assert record["scatterGuardValues"] == runtime.COPY_GUARD
    assert record["scatterMetadata"] == event["scatterMetadata"]
    assert record["scatterStorageType"] == "complex64"
    assert record["outputHash"] == hashlib.sha256(expected.tobytes()).hexdigest()
