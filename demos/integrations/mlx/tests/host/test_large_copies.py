"""Large copy layout bounds, exact readbacks and required native verification."""

import ctypes
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from demos.integrations.mlx.portable_host import copy_layout, runtime
from demos.integrations.mlx.portable_host import verify_large_copies as verifier


def layout_buffers(shape, strides, region=None, preserve=False):
    padding = max(0, 2 - len(shape))
    destination_strides = [0] * padding + [
        math.prod(shape[i + 1 :]) for i in range(len(shape))
    ]
    shape = [1] * padding + list(shape)
    strides = [0] * padding + list(strides)
    strides = [0 if size == 1 else step for size, step in zip(shape, strides)]
    count = math.prod(shape)
    part, origin, destination_offset = region or (shape, 0, 0)
    extents = [(size - 1) * step for size, step in zip(part, strides)]
    low = origin + sum(min(value, 0) for value in extents)
    high = origin + sum(max(value, 0) for value in extents)
    values = {
        "src_shape": part,
        "src_strides": strides,
        "dst_strides": destination_strides,
        "ndim": [len(part)],
        "src_offset": [origin - low],
        "dst_offset": [destination_offset],
    }
    memory = {
        name: (runtime.TYPES[copy_layout.DTYPES[name]] * len(data))(*data)
        for name, data in values.items()
    }
    buffers = {
        name: runtime.Buffer(
            name.encode(),
            copy_layout.DTYPES[name].encode(),
            ctypes.addressof(value),
            len(value),
            0,
        )
        for name, value in memory.items()
    }
    # Layout validation reads metadata only, never these synthetic storage addresses.
    buffers["src"] = runtime.Buffer(b"src", b"uint32", 2**40, high - low + 1, 0)
    buffers["dst"] = runtime.Buffer(
        b"dst", b"uint32", 2**42, count, copy_layout.INOUT if preserve else 1
    )
    return buffers, memory


def test_copy_inventory_preserves_exact_source_storage_and_axis_limits():
    cases = verifier.cases()
    assert len(cases) == len({case["id"] for case in cases}) == 39
    assert {case["dtype"] for case in cases} == {"float32", "int32", "uint32", "bool"}
    layouts = {}
    for case in cases:
        source, offset, low, high = verifier.storage(np, case)
        expected = verifier.reference(np, case)
        assert expected.shape == tuple(case["shape"])
        assert offset + low == 5 and source.size - (offset + high + 1) == 7
        assert expected.dtype == source.dtype
        assert np.shares_memory(expected, expected.base)
        assert expected.base.tobytes() == source.tobytes()
        regions, written = [], []
        for index, region in enumerate(verifier.copy_regions(case)):
            buffers, memory = layout_buffers(
                case["shape"], case["strides"], region, preserve=index > 0
            )
            metadata = copy_layout.validate(buffers, math.prod(region[0]))
            assert len(memory) == 6
            assert metadata["preserveDestination"] is (index > 0)
            written.extend(copy_layout.destination_indices(metadata))
            regions.append(metadata)
        # Regions run in ascending destination order and write each element once.
        assert written == list(range(expected.size))
        layouts[case["id"]] = regions
    assert sum(map(len, layouts.values())) == verifier.dispatch_total() == 45
    assert [len(regions) for regions in layouts.values()].count(1) == 35
    assert layouts["uint32-grid-x"][0]["workgroupCount"] == [65535, 1, 1]
    assert layouts["uint32-grid-y"][0]["workgroupCount"] == [1, 65535, 1]
    assert layouts["uint32-grid-z"][0]["workgroupCount"] == [1, 1, 65535]
    assert layouts["uint32-wide-stride"][0]["sourceCount"] > 65535
    assert math.prod(layouts["uint32-wide-stride"][0]["shape"]) == 6
    assert layouts["uint32-scalar-broadcast"][0]["sourceCount"] == 1
    assert math.prod(layouts["uint32-scalar-broadcast"][0]["shape"]) == 131072
    assert {
        name.removeprefix("uint32-split-"): [
            (region["workgroupCount"], region["sourceOffset"], region["sourceCount"])
            for region in regions
        ]
        for name, regions in layouts.items()
        if name.startswith("uint32-split-")
    } == {
        "x": [([65535, 1, 1], 131069, 131070), ([2, 1, 1], 2, 3)],
        "y": [([1, 65535, 1], 0, 2), ([1, 2, 1], 0, 2)],
        "z": [([1, 1, 65535], 0, 2), ([1, 1, 2], 0, 2)],
        "z-inner": [
            ([1, 1, 65535], 0, 131069),
            ([1, 1, 1], 0, 1),
            ([1, 1, 65535], 0, 131069),
            ([1, 1, 1], 0, 1),
        ],
    }


@pytest.mark.parametrize(
    "shape,regions",
    [
        ([1, 131070], [[1, 131070]]),
        ([1, 131071], [[1, 131070], [1, 1]]),
        ([2, 131073], [[2, 131070], [2, 3]]),
        ([65536, 4], [[65535, 4], [1, 4]]),
        ([70000, 1, 1], [[65535, 1, 1], [4465, 1, 1]]),
        ([300, 300, 1, 2], [[218, 300, 1, 2], [82, 300, 1, 2]]),
        ([2, 3, 70000, 1, 1], [[1, 1, 65535, 1, 1], [1, 1, 4465, 1, 1]] * 6),
        ([1, 1, 2, 2], [[1, 1, 2, 2]]),
    ],
)
def test_copy_launch_regions_split_every_axis_into_exact_portable_launches(
    shape, regions
):
    strides = [3 * math.prod(shape[i + 1 :]) for i in range(len(shape))]
    strides[-1] = -strides[-1]
    destination = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    planned = copy_layout.launch_regions(shape, strides, destination, 7)
    assert [part for part, _, _ in planned] == regions
    written = []
    for part, origin, offset in planned:
        assert (part[-1] + 1) // 2 <= 65535 and part[-2] <= 65535
        assert math.prod(part[:-2]) <= 65535
        written.extend(
            (
                offset + sum(c * d for c, d in zip(coordinates, destination)),
                origin + sum(c * s for c, s in zip(coordinates, strides)),
            )
            for coordinates in np.ndindex(*part)
        )
    # Every destination element is written once, from its own source element.
    expected = [
        (
            7 + sum(c * d for c, d in zip(coordinates, destination)),
            sum(c * s for c, s in zip(coordinates, strides)),
        )
        for coordinates in np.ndindex(*shape)
    ]
    assert sorted(written) == expected


@pytest.mark.parametrize("shape", [(131071,), (65536, 2), (65536, 1, 2)])
def test_copy_grid_overflow_is_rejected_before_destination_enumeration(
    monkeypatch, shape
):
    buffers, memory = layout_buffers(shape, [0] * len(shape))
    monkeypatch.setattr(
        copy_layout,
        "destination_indices",
        lambda metadata: pytest.fail("Invalid launch reached enumeration"),
    )
    with pytest.raises(ValueError, match="65535 groups per axis"):
        copy_layout.validate(buffers, math.prod(shape))
    assert memory


@pytest.mark.parametrize("size", [0, -1, True, 1.0, 2**31])
def test_copy_logical_size_requires_signed_integer(size):
    buffers, memory = layout_buffers([2, 3], [7, -2])
    with pytest.raises(ValueError, match="logical size"):
        copy_layout.validate(buffers, size)
    assert memory


@pytest.mark.parametrize(
    "fault",
    [
        "source-count",
        "destination-count",
        "source-stride",
        "source-span",
        "destination-stride",
        "destination-span",
        "source-offset",
        "destination-offset",
        "alias",
        "collision",
    ],
)
def test_copy_rejects_out_of_bounds_metadata(fault):
    buffers, memory = layout_buffers([2, 3], [7, -2])
    if fault == "source-count":
        buffers["src"].count = 2**31
    elif fault == "destination-count":
        buffers["dst"].count = 2**31
    elif fault == "source-stride":
        memory["src_strides"][0] = 2**31
    elif fault == "source-span":
        memory["src_strides"][0] = 2**31 - 1
    elif fault == "destination-stride":
        memory["dst_strides"][0] = 2**31
    elif fault == "destination-span":
        memory["dst_strides"][0] = 2**31 - 1
    elif fault == "source-offset":
        memory["src_offset"][0] -= 1
    elif fault == "destination-offset":
        memory["dst_offset"][0] = 1
    elif fault == "alias":
        buffers["dst"].data = buffers["src"].data
    elif fault == "collision":
        memory["dst_strides"][:] = [1, 1]
    with pytest.raises(ValueError):
        copy_layout.validate(buffers, 6)


@pytest.mark.parametrize(
    "fault", [None, "source", "changed-source", "output", "shape", "dtype"]
)
def test_copy_readbacks_require_exact_storage(fault):
    case = verifier.cases()[0]
    source, _, _, _ = verifier.storage(np, case)
    before, after, actual = (
        source.copy(),
        source.copy(),
        verifier.reference(np, case).copy(),
    )
    if fault == "source":
        before.view(np.uint32)[0] ^= 1
    elif fault == "changed-source":
        after.view(np.uint32)[-1] ^= 1
    elif fault == "output":
        actual.view(np.uint32)[-1] ^= 1
    elif fault == "shape":
        actual = actual.reshape(1, -1)
    elif fault == "dtype":
        actual = actual.view(np.uint32)
    if fault:
        with pytest.raises(ValueError):
            verifier.record(np, case, before, after, actual)
    else:
        item = verifier.record(np, case, before, after, actual)
        assert item["resultHash"] == hashlib.sha256(actual.tobytes()).hexdigest()


def synthetic_events(tmp_path, target, dtype, case):
    """Model the host's native events for every launch region of one copy."""
    source, offset, low, high = verifier.storage(np, case)
    result = verifier.reference(np, case).copy()
    np.savez(
        tmp_path / (case["id"] + ".npz"), before=source, after=source, actual=result
    )
    item = verifier.record(np, case, source, source, result)
    regions = verifier.copy_regions(case)
    item.update(dispatchStart=0, dispatchCount=len(regions))
    logical = "bool_" if dtype == "bool" else "uint32"
    physical = "bool" if logical == "bool_" and target == "metal" else "uint32"
    entry = "ggn2_dynamic_copy" + logical + logical
    native_entry = {"opengl": "main", "directx": "CSMain"}.get(target, entry)
    source = source[offset + low : offset + high + 1]
    result = result.reshape(-1)
    if dtype == "bool":
        source, result = source.astype(physical), result.astype(physical)
        guard = np.asarray(verifier.BOOLEAN_GUARD, dtype=physical).tolist()
    else:
        source, result = source.view(np.uint32), result.view(np.uint32)
        guard = verifier.COPY_GUARD
    source, result = source.tolist(), result.tolist()
    state, events = [0] * len(result), []
    for index, region in enumerate(regions):
        buffers, memory = layout_buffers(
            case["shape"], case["strides"], region, preserve=index > 0
        )
        threads = math.prod(region[0])
        metadata = copy_layout.validate(buffers, threads)
        first = region[1] - memory["src_offset"][0] - low
        values = {
            "src": source[first : first + metadata["sourceCount"]],
            "dst": state + guard,
            **{name: list(value) for name, value in memory.items()},
        }
        state = list(state)
        for position in copy_layout.destination_indices(metadata):
            state[position] = result[position]
        dtypes = {**copy_layout.DTYPES, "src": physical, "dst": physical}
        inputs = {
            name: {
                "dtype": dtypes[name],
                "shape": [len(value)],
                "values": [bool(v) for v in value] if dtypes[name] == "bool" else value,
            }
            for name, value in values.items()
        }
        bindings = {
            name: {
                "dtype": value["dtype"],
                "shape": value["shape"],
                "binding": {
                    "metadata": {
                        "scalarLayout": {
                            "memberName": name,
                            "elementType": value["dtype"],
                            "elementStrideBytes": np.dtype(value["dtype"]).itemsize,
                        }
                    }
                },
            }
            for name, value in inputs.items()
        }
        events.append(
            {
                "entry": entry,
                "target": target,
                "threads": threads,
                "dispatchVersion": 3,
                "copyMetadata": metadata,
                "workgroupCount": list(metadata["workgroupCount"]),
                "workgroupSize": [1, 1, 1],
                "inputs": inputs,
                "copyValues": (
                    [bool(v) for v in state] if physical == "bool" else list(state)
                ),
                "copyGuardWords": list(guard),
                "details": {
                    "request": {
                        "target": target,
                        "entryPoint": native_entry,
                        "buffers": bindings,
                        "dispatch": {
                            "entryPoint": native_entry,
                            "workgroupCount": list(metadata["workgroupCount"]),
                            "workgroupSize": [1, 1, 1],
                        },
                    }
                },
            }
        )
    return case, item, events


def synthetic_case(tmp_path, target, dtype):
    case = {"id": "synthetic", "dtype": dtype, "shape": [2, 3], "strides": [7, -2]}
    case, item, (event,) = synthetic_events(tmp_path, target, dtype, case)
    assert event["workgroupCount"] == [2, 2, 1] and event["threads"] == 6
    return case, item, event


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("dtype", ["uint32", "bool"])
@pytest.mark.parametrize(
    "fault", [None, "overwrite", "unpreserved", "missing", "order", "coverage"]
)
def test_copy_audit_follows_preserved_launch_regions(
    tmp_path, monkeypatch, target, dtype, fault
):
    # A two-group axis limit splits this small copy into the host's regions.
    monkeypatch.setattr(copy_layout, "MAX_GROUPS", 2)
    case = {"id": "split", "dtype": dtype, "shape": [5, 3], "strides": [7, -2]}
    case, item, events = synthetic_events(tmp_path, target, dtype, case)
    assert [event["copyMetadata"]["shape"] for event in events] == [[2, 3]] * 2 + [
        [1, 3]
    ]
    audited = []
    monkeypatch.setattr(verifier, "audit_native_execution", audited.append)
    if fault == "overwrite":
        # A later region that discarded earlier results uploads zeros.
        events[1]["inputs"]["dst"]["values"] = list(
            events[0]["inputs"]["dst"]["values"]
        )
    elif fault == "unpreserved":
        events[2]["copyMetadata"]["preserveDestination"] = False
    elif fault == "missing":
        events.pop()
    elif fault == "order":
        events[0], events[1] = events[1], events[0]
    elif fault == "coverage":
        events[-1]["copyValues"] = list(events[-2]["copyValues"])
    if fault:
        with pytest.raises(ValueError):
            verifier.audit_case(np, case, item, events, tmp_path)
    else:
        verifier.audit_case(np, case, item, events, tmp_path)
        assert audited == events


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("dtype", ["uint32", "bool"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "metadata",
        "launch",
        "width",
        "version",
        "upload",
        "missing",
        "output",
        "guard",
        "receipt",
        "request",
        "dtype",
        "shape",
        "coercion",
        "artifact",
    ],
)
def test_copy_audit_rejects_changed_native_evidence(
    tmp_path, monkeypatch, target, dtype, fault
):
    case, item, event = synthetic_case(tmp_path, target, dtype)
    audited = []

    def audit(event):
        if fault == "artifact":
            raise ValueError("Native artifact differs")
        audited.append(event)

    monkeypatch.setattr(verifier, "audit_native_execution", audit)
    if fault == "metadata":
        event["copyMetadata"]["sourceOffset"] -= 1
    elif fault == "launch":
        event["workgroupCount"] = [1, 1, 1]
    elif fault == "width":
        event["workgroupSize"] = [2, 1, 1]
    elif fault == "version":
        event["dispatchVersion"] = 2
    elif fault == "upload":
        event["inputs"]["src"]["values"][0] = not event["inputs"]["src"]["values"][0]
    elif fault == "missing":
        event["inputs"].pop("src_offset")
    elif fault == "output":
        event["copyValues"][0] = not event["copyValues"][0]
    elif fault == "guard":
        event["copyGuardWords"][0] = not event["copyGuardWords"][0]
    elif fault == "receipt":
        item["resultHash"] = "0" * 64
    elif fault == "request":
        event["details"]["request"]["dispatch"]["entryPoint"] = "other"
    elif fault == "dtype":
        event["inputs"]["src"]["dtype"] = "float32"
    elif fault == "shape":
        event["inputs"]["src"]["shape"] = [1]
    elif fault == "coercion":
        event["copyValues"][0] = str(event["copyValues"][0])
    if fault:
        with pytest.raises(ValueError):
            verifier.audit_case(np, case, item, [event], tmp_path)
    else:
        verifier.audit_case(np, case, item, [event], tmp_path)
        assert audited == [event]


@pytest.mark.parametrize(
    "dtype,values",
    [
        ("bool", [0]),
        ("bool", [1]),
        ("bool", ["false"]),
        ("uint32", [True]),
        ("uint32", [-1]),
        ("uint32", [2**32]),
        ("int32", [2**31]),
        ("int64", [2**63]),
        ("uint32", [1.0]),
        ("uint32", ["1"]),
        ("float32", [1.0]),
        ("uint32", (1,)),
    ],
)
def test_copy_physical_values_reject_coercion(dtype, values):
    with pytest.raises(ValueError):
        verifier.require_values(np, values, [1], dtype)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "failure",
        "timeout",
        "missing",
        "passed",
        "commit",
        "target",
        "casesPerPath",
        "dispatchCount",
        "fullUpstreamSuite",
    ],
)
def test_bounded_copy_requires_complete_evidence(tmp_path, monkeypatch, fault):
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "opengl"}))
    output = tmp_path / "evidence"
    summary = {
        "passed": True,
        "commit": verifier.COMMIT,
        "target": "opengl",
        "casesPerPath": 39,
        "dispatchCount": 45,
        "fullUpstreamSuite": False,
    }
    if fault in summary:
        summary[fault] = None
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[command.index("--timeout-seconds") + 1] == "600"
        assert command[command.index("-m") + 1] == verifier.__name__
        assert command[-1] == str(output) and not kwargs["check"]
        kwargs["stdout"].write("retained output")
        kwargs["stderr"].write("retained error")
        output.mkdir()
        if fault != "missing":
            (output / "evidence.json").write_text(json.dumps(summary))
        return SimpleNamespace(returncode={"failure": 1, "timeout": 124}.get(fault, 0))

    monkeypatch.setattr(verifier.subprocess, "run", run)
    if fault:
        with pytest.raises((ValueError, FileNotFoundError)):
            verifier.run_bounded(tmp_path / "mlx", packages, output)
    else:
        assert verifier.run_bounded(tmp_path / "mlx", packages, output) == summary
    assert len(calls) == 1
    assert output.with_suffix(".stdout").read_text() == "retained output"
    assert output.with_suffix(".stderr").read_text() == "retained error"
    assert json.loads(output.with_suffix(".command.json").read_text())[
        "returncode"
    ] == {
        "failure": 1,
        "timeout": 124,
    }.get(
        fault, 0
    )


def test_large_copy_ci_is_required_on_each_native_platform():
    workflow = yaml.safe_load(
        Path(".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["portable-host"]
    assert {row["target"] for row in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "directx",
        "opengl",
    }
    step = next(
        step
        for step in job["steps"]
        if step.get("name") == "Execute upstream tests on the translated backend"
    )
    assert " ".join(step["if"].split()) == (
        "${{ !cancelled() && steps.host-build.outcome == 'success' && "
        "steps.host-packages.outcome == 'success' }}"
    )
    assert not step.get("continue-on-error")
    assert "set -euo pipefail" in step["run"]
    assert "--timeout-seconds 4000" in step["run"]
    assert "--cast-batches --large-copies" in step["run"]
