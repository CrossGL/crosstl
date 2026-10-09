"""Binary16 transport, packaging and MLX callback validation."""

import copy
import ctypes
import json
import struct
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    half_storage,
    packages,
    prepare,
    runtime,
    verify_half,
)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_half_transport_is_lossless_for_every_storage_word(target):
    words = list(range(65536))
    physical = half_storage.pack(words, target)
    assert len(set(physical)) == 65536
    assert half_storage.unpack(physical, target) == words
    if target == "opengl":
        for word, actual in zip(words, physical):
            expected = (
                ((word & 0x8000) << 16) | 0x7F800000 | ((word & 1023) << 13)
                if word & 0x7C00 == 0x7C00
                else struct.unpack(
                    "<I",
                    struct.pack("<f", struct.unpack("<e", struct.pack("<H", word))[0]),
                )[0]
            )
            assert actual == expected


@pytest.mark.parametrize("value", (-1, 65536, True, 1.0, None))
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_half_transport_rejects_invalid_logical_words(target, value):
    with pytest.raises(ValueError):
        half_storage.pack([value], target)


@pytest.mark.parametrize(
    "value", (1, 0x3F800001, 0x80000001, 0x7F800001, 0x47800000, 2**32, -1, True)
)
def test_half_transport_rejects_inexact_carriers(value):
    with pytest.raises(ValueError):
        half_storage.unpack([value], "opengl")


@pytest.fixture(scope="module", params=("metal", "directx", "opengl"))
def translated(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("half-" + request.param)
    source = root / packages.COPY_SOURCE
    source.parent.mkdir(parents=True)
    source.write_text("""template<typename T> kernel void copy_signature(
device const T* src [[buffer(0)]], device T* dst [[buffer(1)]],
constant int* src_shape [[buffer(2)]], constant long* src_strides [[buffer(3)]],
constant long* dst_strides [[buffer(4)]], constant int& ndim [[buffer(5)]],
constant long& src_offset [[buffer(6)]], constant long& dst_offset [[buffer(7)]],
uint index [[thread_position_in_grid]]) { dst[index] = src[index]; }
template<typename T, typename U> kernel void cast_signature(
device const T* src [[buffer(0)]], device U* dst [[buffer(1)]],
constant uint& size [[buffer(2)]], uint index [[thread_position_in_grid]]) {
if (index < size) dst[index] = U(src[index]); }
template [[host_name("ggn2_dynamic_copyfloat16float16")]] [[kernel]]
decltype(copy_signature<half>) copy_signature<half>;
template [[host_name("v_copyfloat16float32")]] [[kernel]]
decltype(cast_signature<half, float>) cast_signature<half, float>;
template [[host_name("v_copyfloat32float16")]] [[kernel]]
decltype(cast_signature<float, half>) cast_signature<float, half>;
""")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            packages.subprocess,
            "check_output",
            lambda command, **kwargs: prepare.COMMIT if "rev-parse" in command else "",
        )
        index = packages.build_packages(
            root, root / "half", request.param, family="half"
        )
    assert set(index["descriptors"]) == set(packages.HALF_ENTRIES)
    return root / "half", index


def upload_evidence(resource, target):
    layout = resource["scalarLayout"]
    dtype = layout.get("storageEncoding", {}).get(
        "logicalElementType", layout["elementType"]
    )
    value = {"dtype": dtype, "shape": [3], "values": [0, 0, 0]}
    if dtype in {"float16", "float32"}:
        value["encoding"] = (
            half_storage.encoding(target) if dtype == "float16" else "ieee754-binary32"
        )
    binding = {
        **copy.deepcopy(value),
        "binding": {
            "kind": resource["kind"],
            "metadata": {"scalarLayout": copy.deepcopy(layout)},
        },
    }
    return value, binding


def test_half_evidence_accepts_reflected_storage_for_all_entries(translated):
    _, index = translated
    for descriptor in index["descriptors"].values():
        for resource in descriptor["bindings"]:
            if "executionInput" not in resource.get("provenance", {}):
                value, binding = upload_evidence(resource, index["target"])
                verify_half.audit_upload_layout(index["target"], value, binding)


@pytest.mark.parametrize(
    "fault",
    (
        "dtype",
        "shape",
        "encoding",
        "values",
        "elementType",
        "elementSizeBytes",
        "elementStrideBytes",
    ),
)
def test_half_evidence_rejects_upload_layout_changes(translated, fault):
    _, index = translated
    resource = next(
        resource
        for resource in index["descriptors"][packages.HALF_COPY_ENTRY]["bindings"]
        if resource["scalarLayout"]["elementType"] in {"float16", "float32", "uint16"}
    )
    value, binding = upload_evidence(resource, index["target"])
    if fault == "values":
        value["values"].pop()
    elif fault in {"dtype", "shape", "encoding"}:
        binding[fault] = [2] if fault == "shape" else "invalid"
    else:
        binding["binding"]["metadata"]["scalarLayout"][fault] = (
            "int32" if fault == "elementType" else 8
        )
    with pytest.raises(ValueError):
        verify_half.audit_upload_layout(index["target"], value, binding)


@pytest.mark.parametrize("translated", ["directx"], indirect=True)
@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "encoding",
        "logical",
        "physical",
        "alignment",
        "offset",
        "kind",
        "layout",
        "runtime",
        "target",
        "transport",
    ),
)
def test_half_evidence_requires_explicit_directx_binary16_contract(translated, fault):
    _, index = translated
    resource = next(
        resource
        for resource in index["descriptors"][packages.HALF_COPY_ENTRY]["bindings"]
        if "storageEncoding" in resource["scalarLayout"]
    )
    value, binding = upload_evidence(resource, "directx")
    layout = binding["binding"]["metadata"]["scalarLayout"]
    if fault == "missing":
        layout.pop("storageEncoding")
    elif fault in {"encoding", "logical"}:
        key = "encoding" if fault == "encoding" else "logicalElementType"
        layout["storageEncoding"][key] = "invalid"
    elif fault == "kind":
        binding["binding"]["kind"] = "texture"
    elif fault == "transport":
        binding["encoding"] = value["encoding"] = "ieee754-binary32"
    elif fault != "target":
        key, replacement = {
            "physical": ("physicalType", "half"),
            "alignment": ("alignmentBytes", 4),
            "offset": ("memberOffsetBytes", 2),
            "layout": ("storageLayout", "hlsl-constant-buffer"),
            "runtime": ("runtimeSized", False),
        }[fault]
        layout[key] = replacement
    with pytest.raises(ValueError):
        verify_half.audit_upload_layout(
            "metal" if fault == "target" else "directx", value, binding
        )


@pytest.fixture
def host(translated, tmp_path):
    directory, index = translated
    base = tmp_path / "base"
    base.mkdir()
    (base / "index.json").write_text(
        json.dumps(
            {
                "target": index["target"],
                "descriptors": {
                    entry: {"target": index["target"]} for entry in packages.ENTRIES
                },
            }
        )
    )
    return runtime.HostRuntime(base, tmp_path / "trace", half=directory)


def buffers_for(entry):
    if entry == packages.HALF_COPY_ENTRY:
        spec = {
            "src": ("float16", [0x8000, 0x7D01, 0x3C00]),
            "dst": ("float16", [0, 0, 0]),
            "src_shape": ("int32", [1, 3]),
            "src_strides": ("int64", [3, 1]),
            "dst_strides": ("int64", [3, 1]),
            "ndim": ("int32", [2]),
            "src_offset": ("int64", [0]),
            "dst_offset": ("int64", [0]),
        }
        expected, groups = [0x8000, 0x7D01, 0x3C00], [2, 1, 1]
    else:
        source, destination = packages.HALF_CAST_ENTRIES[entry]
        spec = {
            "src": (
                source,
                [0x3C00, 0, 0xC000] if source == "float16" else [1.0, 0.0, -2.0],
            ),
            "dst": (destination, [0, 0, 0]),
            "size": ("uint32", [3]),
        }
        expected = (
            [0x3C00, 0, 0xC000]
            if destination == "float16"
            else [0x3F800000, 0, 0xC0000000]
        )
        groups = [3, 1, 1]
    memory = {
        name: (runtime.TYPES[kind] * len(data))(*data)
        for name, (kind, data) in spec.items()
    }
    buffers = (runtime.Buffer * len(spec))(
        *(
            runtime.Buffer(
                name.encode(),
                kind.encode(),
                ctypes.addressof(memory[name]),
                len(data),
                int(name == "dst"),
            )
            for name, (kind, data) in spec.items()
        )
    )
    launch = runtime.Launch(
        (ctypes.c_uint32 * 3)(*groups), (ctypes.c_uint32 * 3)(1, 1, 1)
    )
    return buffers, memory, spec["dst"][0], expected, launch


@pytest.mark.parametrize("entry", packages.HALF_ENTRIES)
@pytest.mark.parametrize(
    "fault", (None, "dtype", "null", "guard", "encoding", "range", "carrier")
)
def test_half_callback_preserves_transport_and_rejects_invalid_readback(
    host, monkeypatch, entry, fault
):
    buffers, memory, dtype, expected, launch = buffers_for(entry)
    storage = runtime.physical_dtype(dtype, host.target)
    encoding = (
        half_storage.encoding(host.target) if dtype == "float16" else "ieee754-binary32"
    )
    physical = (
        half_storage.pack(expected, host.target) if dtype == "float16" else expected
    )
    guard = (
        half_storage.pack(half_storage.GUARD, host.target)
        if dtype == "float16"
        else runtime.COPY_GUARD
    )
    output = physical + guard
    if fault == "dtype":
        buffers[0].dtype = b"int32"
    elif fault == "null":
        buffers[0].data = None
    elif fault == "guard":
        output[-1] = 0
    elif fault == "range":
        output[0] = 2**32
    elif fault == "carrier":
        output[0] = (
            0x3F800001 if dtype == "float16" and host.target == "opengl" else 2**32
        )
    calls = []

    def execute(request):
        calls.append(request)
        name = next(
            binding["name"]
            for binding in host.descriptors[entry]["bindings"]
            if binding["access"] == "read_write"
        )
        return SimpleNamespace(
            status="ok",
            outputs={
                name: {
                    "dtype": storage,
                    "encoding": "invalid" if fault == "encoding" else encoding,
                    "shape": [35],
                    "values": output,
                }
            },
            details={},
        )

    monkeypatch.setattr(
        runtime.gather_dispatch, "execute", lambda host, request: execute(request)
    )
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            host.dispatch(entry, buffers, len(buffers), 3, launch=launch)
        assert list(memory["dst"]) == [0, 0, 0]
        assert not host.trace.exists()
    else:
        host.dispatch(entry, buffers, len(buffers), 3, launch=launch)
        raw_type = ctypes.c_uint16 if dtype == "float16" else ctypes.c_uint32
        actual = list(ctypes.cast(memory["dst"], ctypes.POINTER(raw_type * 3)).contents)
        assert actual == expected
        record = json.loads(host.trace.read_text())
        assert record["halfStorage"]["logicalWords"] == expected
        assert record["halfStorage"]["guardValues"] == guard
    assert len(calls) == int(fault not in {"dtype", "null"})


@pytest.mark.parametrize(
    "fault", ["encoding", "logical", "physical", "missing", "kind"]
)
def test_half_callback_rejects_inconsistent_storage_codecs(host, monkeypatch, fault):
    entry = packages.HALF_COPY_ENTRY
    buffers, memory, _, _, launch = buffers_for(entry)
    binding = next(
        binding
        for binding in host.descriptors[entry]["bindings"]
        if binding["scalarLayout"].get("memberName", binding["name"]) == "src"
    )
    layout = binding["scalarLayout"]
    if fault == "missing":
        layout.pop("storageEncoding", None)
        layout["elementType"] = "uint16"
    else:
        layout["storageEncoding"] = {
            "logicalElementType": "float16",
            "encoding": "ieee754-binary16",
        }
        if fault == "encoding":
            layout["storageEncoding"]["encoding"] = "unknown"
        elif fault == "logical":
            layout["storageEncoding"]["logicalElementType"] = "uint16"
        elif fault == "kind":
            binding.pop("kind", None)
        else:
            layout["elementType"] = "uint32"
    calls = []
    monkeypatch.setattr(
        runtime.gather_dispatch, "execute", lambda *args: calls.append(args)
    )
    with pytest.raises(ValueError):
        host.dispatch(entry, buffers, len(buffers), 3, launch=launch)
    assert calls == [] and list(memory["dst"]) == [0] * 3


@pytest.mark.parametrize("fault", ("family", "target", "missing", "extra"))
def test_half_packages_require_matching_target_and_complete_inventory(
    host, tmp_path, fault
):
    index = json.loads((host.half_directory / "index.json").read_text())
    if fault in {"family", "target"}:
        index[fault] = "invalid"
    elif fault == "missing":
        index["descriptors"].pop(packages.HALF_COPY_ENTRY)
    else:
        index["descriptors"]["extra"] = copy.deepcopy(
            next(iter(index["descriptors"].values()))
        )
    directory = tmp_path / "invalid-half"
    directory.mkdir()
    (directory / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="exact entry set"):
        runtime.HostRuntime(host.directory, tmp_path / "unused-trace", half=directory)


def test_half_copy_overlap_checks_use_two_byte_elements():
    from demos.integrations.mlx.portable_host import copy_layout

    buffers, memory, _, _, _ = buffers_for(packages.HALF_COPY_ENTRY)
    allocation = (ctypes.c_uint16 * 6)()
    buffers[0].data = ctypes.addressof(allocation)
    buffers[1].data = ctypes.addressof(allocation) + 6
    supplied = {buffer.name.decode(): buffer for buffer in buffers}
    copy_layout.validate(supplied, 3, dtype="float16")
    buffers[1].data -= 2
    with pytest.raises(ValueError, match="overlap"):
        copy_layout.validate(supplied, 3, dtype="float16")


@pytest.mark.parametrize("fault", (False, True))
def test_half_strided_copy_preserves_untouched_destination(host, monkeypatch, fault):
    buffers, memory, _, source, launch = buffers_for(packages.HALF_COPY_ENTRY)
    initial = [0x7E01, 0x3555, 0x8000, 0, 1, 0, 0xFC00]
    destination = (ctypes.c_uint16 * 7)(*initial)
    buffers[1].data = ctypes.addressof(destination)
    buffers[1].count, buffers[1].output = 7, 2
    memory["dst_strides"][1] = 2
    memory["dst_offset"][0] = 1
    expected = list(initial)
    expected[1::2] = source
    wrong = list(expected)
    if fault:
        wrong[2] = 0
    values = half_storage.pack(wrong + half_storage.GUARD, host.target)

    def execute(host, request):
        name = next(
            binding["name"]
            for binding in host.descriptors[packages.HALF_COPY_ENTRY]["bindings"]
            if binding["access"] == "read_write"
        )
        return SimpleNamespace(
            status="ok",
            details={},
            outputs={
                name: {
                    "dtype": runtime.physical_dtype("float16", host.target),
                    "encoding": half_storage.encoding(host.target),
                    "shape": [39],
                    "values": values,
                }
            },
        )

    monkeypatch.setattr(runtime.gather_dispatch, "execute", execute)
    if fault:
        with pytest.raises(RuntimeError, match="untouched destination"):
            host.dispatch(
                packages.HALF_COPY_ENTRY, buffers, len(buffers), 3, launch=launch
            )
        assert list(destination) == initial
    else:
        host.dispatch(packages.HALF_COPY_ENTRY, buffers, len(buffers), 3, launch=launch)
        assert list(destination) == expected


@pytest.fixture
def workload_evidence():
    import numpy as np

    from demos.integrations.mlx.portable_host import half_workloads

    records, trace = [], []
    for case in half_workloads.cases():
        source = half_workloads.inputs(np, case)
        expected = half_workloads.reference(np, case, source)
        entries = half_workloads.entries(case, source)
        bits = half_workloads.words(np, expected)
        records.append(
            {
                **case,
                "inputWords": half_workloads.words(np, source),
                "resultWords": bits,
                "shape": list(expected.shape),
                "dtype": expected.dtype.name,
                "dispatchCount": len(entries),
            }
        )
        for entry in entries:
            trace.append(
                {
                    "entry": entry,
                    "dispatchVersion": 3,
                    "workgroupSize": [1, 1, 1],
                    "threads": expected.size,
                    "target": "metal",
                }
            )
        if entries:
            dtype = expected.dtype.name
            trace[-1]["halfStorage"] = {
                "logicalType": dtype,
                "physicalType": dtype,
                "encoding": (
                    "ieee754-binary16" if dtype == "float16" else "ieee754-binary32"
                ),
                "values": bits,
                "guardValues": [0x3555 if dtype == "float16" else 0x6A15BEEF] * 32,
                "logicalWords": bits,
            }
    return records, trace


@pytest.mark.parametrize(
    "fault",
    (
        None,
        "missing",
        "input",
        "result",
        "guard",
        "dispatch",
        "logical",
        "encoding",
        "extra",
    ),
)
def test_half_workload_evidence_requires_complete_exact_results(
    workload_evidence, fault
):
    from demos.integrations.mlx.portable_host import half_workloads

    records, trace = workload_evidence
    if fault == "missing":
        records.pop()
    elif fault == "input":
        records[-1]["inputWords"][0] ^= 1
    elif fault == "result":
        records[-1]["resultWords"][0] ^= 1
    elif fault == "guard":
        trace[-1]["halfStorage"]["guardValues"][0] ^= 1
    elif fault == "dispatch":
        trace.pop()
    elif fault == "logical":
        trace[-1]["halfStorage"]["logicalWords"] = []
    elif fault == "encoding":
        trace[-1]["halfStorage"]["encoding"] = "invalid"
    elif fault == "extra":
        trace.append(copy.deepcopy(trace[-1]))
    if fault:
        with pytest.raises(ValueError):
            half_workloads.validate(records, trace, native=True)
    else:
        half_workloads.validate(records, trace, native=True)
        assert len(records) == 28
        assert (
            sum(
                len(record["resultWords"])
                for record in records
                if record["layout"] == "exhaustive"
            )
            == 65536
        )


def test_half_ci_requires_every_target_and_retained_evidence():
    import re
    from pathlib import Path

    import yaml

    workflow = yaml.safe_load(
        Path(".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["half-host"]
    assert job["needs"] == "portable-host"
    assert (
        job["if"] == "github.event_name != 'schedule'"
        and "continue-on-error" not in job
    )
    assert {
        item["target"]: item["os"] for item in job["strategy"]["matrix"]["include"]
    } == {"metal": "macos-26", "directx": "windows-2025", "opengl": "ubuntu-24.04"}
    steps = {step.get("name"): step for step in job["steps"]}
    for name in ("Translate half host packages", "Execute half host operations"):
        assert "if" not in steps[name] and "continue-on-error" not in steps[name]
        assert "set -euo pipefail" in steps[name]["run"]
    assert "--family half" in steps["Translate half host packages"]["run"]
    assert "portable_host.verify_half" in steps["Execute half host operations"]["run"]
    assert "-n auto" in steps["Validate half host contracts"]["run"]
    assert (
        '"PyYAML>=6,<7"' in steps["Install CrossTL and MLX build dependencies"]["run"]
    )
    assert steps["Retain half execution evidence"]["if"] == "always()"
    assert (
        steps["Retain half execution evidence"]["with"]["if-no-files-found"] == "error"
    )
    deadlines = [
        int(value)
        for step in job["steps"]
        for value in re.findall(r"--timeout-seconds (\d+)", step.get("run", ""))
    ]
    assert deadlines == [
        1800,
        1800,
        1800,
        1200,
        1200,
        1800,
        600,
        2100,
        900,
        1200,
        900,
        1800,
        900,
        2100,
    ]
    assert sum(deadlines) + 900 < job["timeout-minutes"] * 60
