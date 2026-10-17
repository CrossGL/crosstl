"""Bfloat16 package inventory, exact transport and guarded host callbacks."""

import copy
import ctypes
import json
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    bfloat_storage,
    packages,
    prepare,
    runtime,
)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_bfloat_transport_preserves_every_word(target):
    words = list(range(65536))
    physical = bfloat_storage.pack(words, target)
    assert physical == ([word << 16 for word in words] if target == "opengl" else words)
    assert bfloat_storage.unpack(physical, target) == words


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("word", (-1, 65536, True, 1.0, None))
def test_bfloat_transport_rejects_invalid_words(target, word):
    with pytest.raises(ValueError):
        bfloat_storage.pack([word], target)


@pytest.mark.parametrize("word", (1, 0x3F800001, 0x7F800001, 2**32, -1, True, 1.0))
def test_bfloat_transport_rejects_inexact_gl_carriers(word):
    with pytest.raises(ValueError):
        bfloat_storage.unpack([word], "opengl")


@pytest.fixture(scope="module", params=("metal", "directx", "opengl"))
def translated(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("bfloat-" + request.param)
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
template [[host_name("ggn2_dynamic_copybfloat16bfloat16")]] [[kernel]]
decltype(copy_signature<bfloat>) copy_signature<bfloat>;
template [[host_name("v_copybfloat16float32")]] [[kernel]]
decltype(cast_signature<bfloat, float>) cast_signature<bfloat, float>;
template [[host_name("v_copyfloat32bfloat16")]] [[kernel]]
decltype(cast_signature<float, bfloat>) cast_signature<float, bfloat>;
""")
    (root / packages.BINARY_SOURCE).write_text(
        "\n".join(
            f"""template<typename T> kernel void {entry}_impl(
device const T* a [[buffer(0)]], device const T* b [[buffer(1)]],
device {kind}* c [[buffer(2)]], constant uint& size [[buffer(3)]],
uint index [[thread_position_in_grid]]) {{
if (index < size) c[index] = {expression}; }}
template [[host_name("{entry}")]] [[kernel]]
decltype({entry}_impl<bfloat>) {entry}_impl<bfloat>;"""
            for entry in (
                *packages.BFLOAT_BINARY_ENTRIES,
                *packages.BFLOAT_COMPARISON_ENTRIES,
            )
            for kind, expression in [
                (
                    ("bool", "a[index] < b[index]")
                    if entry in packages.BFLOAT_COMPARISON_ENTRIES
                    else ("T", "a[index] + b[index]")
                )
            ]
        )
    )
    (root / packages.UNARY_SOURCE).write_text(
        """template<typename T> kernel void absolute_impl(
device const T* in [[buffer(0)]], device T* out [[buffer(1)]],
constant uint& size [[buffer(2)]], uint index [[thread_position_in_grid]]) {
if (index < size) out[index] = abs(in[index]); }
template [[host_name("v_Absbfloat16bfloat16")]] [[kernel]]
decltype(absolute_impl<bfloat>) absolute_impl<bfloat>;
"""
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            packages.subprocess,
            "check_output",
            lambda command, **kw: prepare.COMMIT if "rev-parse" in command else "",
        )
        index = packages.build_packages(
            root, root / "bfloat", request.param, family="bfloat"
        )
    assert set(index["descriptors"]) == set(packages.BFLOAT_ENTRIES)
    return root / "bfloat", index


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
    return runtime.HostRuntime(base, tmp_path / "trace", bfloat=directory)


def buffers_for(entry):
    dtype, output_name = "bfloat16", "dst"
    expected = [0x3F80, 0, 0xC000]
    if entry == packages.BFLOAT_COPY_ENTRY:
        spec = {
            "src": ("bfloat16", expected),
            "dst": ("bfloat16", [0] * 3),
            "src_shape": ("int32", [1, 3]),
            "src_strides": ("int64", [3, 1]),
            "dst_strides": ("int64", [3, 1]),
            "ndim": ("int32", [2]),
            "src_offset": ("int64", [0]),
            "dst_offset": ("int64", [0]),
        }
        groups = [2, 1, 1]
    elif entry in packages.BFLOAT_CAST_ENTRIES:
        source, dtype = packages.BFLOAT_CAST_ENTRIES[entry]
        spec = {
            "src": (source, expected if source == "bfloat16" else [1.0, 0.0, -2.0]),
            "dst": (dtype, [0] * 3),
            "size": ("uint32", [3]),
        }
        if dtype == "float32":
            expected = [word << 16 for word in expected]
        groups = [3, 1, 1]
    else:
        absolute = entry in packages.BFLOAT_ABSOLUTE_ENTRIES
        boolean = entry in packages.BFLOAT_COMPARISON_ENTRIES
        dtype = "bool_" if boolean else "bfloat16"
        output_name = "out" if absolute else "c"
        spec = {
            "in" if absolute else "a": ("bfloat16", [0x3F80, 0xC000, 0x4040]),
            **({} if absolute else {"b": ("bfloat16", [0x4000] * 3)}),
            output_name: (dtype, [0] * 3),
            "size": ("uint32", [3]),
        }
        expected = [1, 0, 1] if boolean else [0x4040, 0, 0x40A0]
        groups = [3, 1, 1]
    memory = {
        name: (runtime.TYPES[kind] * len(values))(*values)
        for name, (kind, values) in spec.items()
    }
    buffers = (runtime.Buffer * len(spec))(
        *(
            runtime.Buffer(
                name.encode(),
                kind.encode(),
                ctypes.addressof(memory[name]),
                len(values),
                int(name == output_name),
            )
            for name, (kind, values) in spec.items()
        )
    )
    launch = runtime.Launch(
        (ctypes.c_uint32 * 3)(*groups), (ctypes.c_uint32 * 3)(1, 1, 1)
    )
    return buffers, memory, output_name, dtype, expected, launch


@pytest.mark.parametrize("entry", packages.BFLOAT_ENTRIES)
@pytest.mark.parametrize(
    "fault", (None, "dtype", "null", "guard", "encoding", "range", "shape")
)
def test_bfloat_callback_checks_transport_before_committing(
    host, monkeypatch, entry, fault
):
    buffers, memory, name, dtype, expected, launch = buffers_for(entry)
    storage = runtime.physical_dtype(dtype, host.target)
    if dtype == "bfloat16":
        physical = bfloat_storage.pack(expected, host.target)
        guard = bfloat_storage.pack(bfloat_storage.GUARD, host.target)
        encoding = bfloat_storage.encoding(host.target)
    elif dtype == "bool_":
        physical = list(map(bool, expected)) if host.target == "metal" else expected
        guard = (
            runtime.BOOLEAN_GUARD
            if host.target == "metal"
            else list(map(int, runtime.BOOLEAN_GUARD))
        )
        encoding = None
    else:
        physical, guard, encoding = expected, runtime.COPY_GUARD, "ieee754-binary32"
    output = {"dtype": storage, "shape": [35], "values": physical + guard}
    if encoding:
        output["encoding"] = encoding
    if fault == "dtype":
        buffers[0].dtype = b"int32"
    elif fault == "null":
        buffers[0].data = None
    elif fault == "guard":
        output["values"][-1] = not guard[-1] if dtype == "bool_" else 0
    elif fault == "encoding":
        output["encoding"] = "invalid"
    elif fault == "range":
        output["values"][0] = 2**32
    elif fault == "shape":
        output["shape"] = [34]
    calls = []

    def execute(_host, request):
        calls.append(request)
        binding = next(
            binding["name"]
            for binding in host.descriptors[entry]["bindings"]
            if binding["access"] == "read_write"
        )
        return SimpleNamespace(status="ok", outputs={binding: output}, details={})

    monkeypatch.setattr(runtime.gather_dispatch, "execute", execute)
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            host.dispatch(entry, buffers, len(buffers), 3, launch=launch)
        assert list(memory[name]) == [0] * 3
        assert not host.trace.exists()
    else:
        host.dispatch(entry, buffers, len(buffers), 3, launch=launch)
        raw = ctypes.c_uint32 if dtype == "float32" else runtime.TYPES[dtype]
        assert (
            list(ctypes.cast(memory[name], ctypes.POINTER(raw * 3)).contents)
            == expected
        )
        event = json.loads(host.trace.read_text())
        assert event["bfloatStorage"]["logicalWords"] == expected
        assert event["bfloatStorage"]["guardValues"] == guard
    assert len(calls) == int(fault not in {"dtype", "null"})


@pytest.mark.parametrize(
    "fault", ("elementType", "elementStrideBytes", "elementSizeBytes")
)
def test_bfloat_callback_rejects_inconsistent_reflection(host, monkeypatch, fault):
    entry = packages.BFLOAT_COPY_ENTRY
    host.descriptors = copy.deepcopy(host.descriptors)
    layout = next(
        binding["scalarLayout"]
        for binding in host.descriptors[entry]["bindings"]
        if binding["scalarLayout"]["elementType"]
        == runtime.physical_dtype("bfloat16", host.target)
    )
    layout[fault] = "int32" if fault == "elementType" else 8
    buffers, memory, _, _, _, launch = buffers_for(entry)
    calls = []
    monkeypatch.setattr(
        runtime.gather_dispatch, "execute", lambda *args: calls.append(args)
    )
    with pytest.raises(ValueError):
        host.dispatch(entry, buffers, len(buffers), 3, launch=launch)
    assert not calls
    assert list(memory["dst"]) == [0] * 3


@pytest.mark.parametrize("layout", ("transpose", "reverse", "broadcast", "scalar"))
def test_bfloat_operand_preserves_physical_layout(layout):
    import numpy as np

    from demos.integrations.mlx.portable_host import bfloat_workloads as workloads

    class Arrays:
        float32 = np.float32
        array = staticmethod(np.array)

        @staticmethod
        def as_strided(source, shape, strides, offset):
            return np.lib.stride_tricks.as_strided(
                source[offset:],
                shape=shape,
                strides=tuple(stride * source.itemsize for stride in strides),
            )

    source, _ = workloads.inputs(
        np, {"entry": "v_copyfloat32bfloat16", "layout": layout}
    )
    actual = workloads.operand(Arrays, np, source, "float32")
    np.testing.assert_array_equal(actual.view(np.uint32), source.view(np.uint32))


def test_bfloat_cast_reference_rounds_midpoint_ties_to_even():
    import numpy as np

    from demos.integrations.mlx.portable_host.bfloat_workloads import round_words

    values = np.array(
        [0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000], dtype=np.uint32
    ).view(np.float32)
    assert round_words(np, values).tolist() == [0x3F80, 0x3F82, 0xBF80, 0xBF82]


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "fault",
    (
        None,
        "missing",
        "reordered",
        "value",
        "guard",
        "word-type",
        "record-type",
        "count",
        "extra",
    ),
)
def test_bfloat_batch_evidence_requires_complete_exact_outputs(
    monkeypatch, target, fault
):
    import numpy as np

    from demos.integrations.mlx.portable_host import bfloat_workloads as workloads

    case = {"entry": "v_copyfloat32bfloat16", "layout": "batched"}
    monkeypatch.setattr(workloads, "cases", lambda: [case])
    a, b = workloads.inputs(np, case)
    expected, dtype = workloads.result(np, case, a, b)
    bits = workloads.words(np, expected, dtype)
    record = {
        **case,
        "shape": list(expected.shape),
        "dtype": "mlx.core.bfloat16",
        "inputWords": workloads.words(np, a, "float32"),
        "resultWords": bits,
        "dispatchCount": 2,
    }
    trace = []
    for start in (0, 65535):
        logical = bits[start : start + 65535]
        trace.append(
            {
                "entry": case["entry"],
                "target": target,
                "threads": len(logical),
                "dispatchVersion": 3,
                "workgroupSize": [1, 1, 1],
                "bfloatStorage": {
                    "logicalType": dtype,
                    "physicalType": runtime.physical_dtype(dtype, target),
                    "encoding": bfloat_storage.encoding(target),
                    "values": bfloat_storage.pack(logical, target),
                    "guardValues": bfloat_storage.pack(bfloat_storage.GUARD, target),
                    "logicalWords": logical,
                },
            }
        )
    if fault == "missing":
        trace.pop()
    elif fault == "reordered":
        trace.reverse()
    elif fault == "value":
        trace[0]["bfloatStorage"]["values"][0] = 0
    elif fault == "guard":
        trace[0]["bfloatStorage"]["guardValues"][-1] = 0
    elif fault == "word-type":
        trace[0]["bfloatStorage"]["values"][0] = float(
            trace[0]["bfloatStorage"]["values"][0]
        )
    elif fault == "record-type":
        record["resultWords"] = [float(value) for value in bits]
    elif fault == "count":
        trace[0]["threads"] -= 1
    elif fault == "extra":
        trace.append(copy.deepcopy(trace[0]))
    if fault:
        with pytest.raises(ValueError):
            workloads.validate([record], trace, native=True)
    else:
        workloads.validate([record], trace, native=True)


@pytest.mark.parametrize("translated", ["directx", "opengl"], indirect=True)
@pytest.mark.parametrize("word", (1, 0x7F, 0x8001, 0x807F))
def test_bfloat_comparison_rejects_unproven_subnormals(host, monkeypatch, word):
    entry = "vv_Lessbfloat16"
    buffers, memory, _, _, _, launch = buffers_for(entry)
    memory["a"][0] = word
    calls = []
    monkeypatch.setattr(
        runtime.gather_dispatch, "execute", lambda *args: calls.append(args)
    )
    with pytest.raises(ValueError, match="Subnormal float comparison parity"):
        host.dispatch(entry, buffers, len(buffers), 3, launch=launch)
    assert not calls and list(memory["c"]) == [0] * 3


@pytest.mark.parametrize("fault", ("target", "family", "missing", "extra"))
def test_bfloat_package_inventory_is_required(host, translated, tmp_path, fault):
    _, original = translated
    index = copy.deepcopy(original)
    if fault in {"target", "family"}:
        index[fault] = "invalid"
    elif fault == "missing":
        index["descriptors"].pop(packages.BFLOAT_COPY_ENTRY)
    else:
        index["descriptors"]["unknown"] = {"target": host.target}
    directory = tmp_path / "bad"
    directory.mkdir()
    (directory / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="target and exact entry set"):
        runtime.HostRuntime(host.directory, tmp_path / "bad.trace", bfloat=directory)


def test_bfloat_ci_requires_native_proofs_without_additional_runners():
    from pathlib import Path

    import yaml

    workflow = (
        Path(__file__).resolve().parents[5]
        / ".github/workflows/demo-project-testing.yml"
    )
    job = yaml.safe_load(workflow.read_text())["jobs"]["half-host"]
    assert {
        item["target"]: item["os"] for item in job["strategy"]["matrix"]["include"]
    } == {
        "metal": "macos-26",
        "opengl": "ubuntu-24.04",
        "directx": "windows-2025",
    }
    steps = {step.get("name"): step for step in job["steps"]}
    for name in ("Translate bfloat host packages", "Execute bfloat host operations"):
        assert "if" not in steps[name] and "continue-on-error" not in steps[name]
        assert "set -euo pipefail" in steps[name]["run"]
    assert "--family bfloat" in steps["Translate bfloat host packages"]["run"]
    native = steps["Execute bfloat host operations"]
    assert "portable_host.verify_bfloat" in native["run"]
    assert native["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    assert steps["Validate half host contracts"]["if"] == "runner.os == 'Linux'"
    assert "test_portable_bfloat.py" in steps["Validate half host contracts"]["run"]
    assert steps["Retain half execution evidence"]["if"] == "always()"
