import ctypes
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from demos.integrations.mlx.portable_host import (
    binary_workloads,
    boolean_workloads,
    cast_workloads,
    copy_workloads,
    full_workloads,
    packages,
    prepare,
    runtime,
    unary_workloads,
    verify,
    view_workloads,
)


def checkout(root, monkeypatch, newline=b"\n"):
    backend = root / "mlx/backend/no_gpu"
    backend.mkdir(parents=True)
    originals = {
        "CMakeLists.txt": b"target_sources(mlx PRIVATE primitives.cpp)\n",
        "primitives.cpp": (
            "\n".join(
                f"{'NO_GPU_MULTI' if name in prepare.MULTI_OUTPUT_VIEWS else 'NO_GPU'}({name})"
                for name in (
                    "Arange",
                    "Reduce",
                    "BitwiseBinary",
                    "BitwiseInvert",
                    "Concatenate",
                    "Select",
                    "Gather",
                    "GatherAxis",
                    "ScatterAxis",
                    "Power",
                    *packages.UNARY_OPERATIONS,
                    *prepare.VIEW_PRIMITIVES,
                    *prepare.COPY_PRIMITIVES,
                    *packages.BINARY_OPERATIONS,
                    *prepare.CAST_PRIMITIVES,
                    *packages.COMPARISON_OPERATIONS,
                    *packages.LOGICAL_OPERATIONS,
                    "LogicalNot",
                )
                if name not in {"Log2", "Log10", "Rsqrt"}
            ).encode()
            + b"\n"
        ),
        "event.cpp": (
            b"void Event::wait(Stream stream) {\n  cpu_wait();\n}\n"
            b"void Event::signal(Stream stream) {\n  cpu_signal();\n}\n"
        ),
    }
    # Git blobs retain LF even when the checkout uses converted line endings.
    for name, data in originals.items():
        (backend / name).write_bytes(data.replace(b"\n", newline))

    def git(command, **kwargs):
        if "rev-parse" in command:
            return prepare.COMMIT
        if "show" in command:
            return originals[command[-1].rsplit("/", 1)[-1]]
        if "diff" in command:
            return b"mlx/backend/no_gpu/CMakeLists.txt\0"
        return ""

    monkeypatch.setattr(prepare.subprocess, "check_output", git)
    return backend


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["lf", "crlf"])
def test_prepare_preserves_unimplemented_primitives_and_cpu_events(
    tmp_path, monkeypatch, newline
):
    backend = checkout(tmp_path, monkeypatch, newline)
    record = prepare.prepare(tmp_path, tmp_path / "adaptation.json")
    assert record["commit"] == prepare.COMMIT and len(record["files"]) == 5
    assert "NO_GPU(Arange)" not in (backend / "crosstl_primitives.cpp").read_text()
    assert "NO_GPU(Abs)" not in (backend / "crosstl_primitives.cpp").read_text()
    assert "NO_GPU(AsType)" not in (backend / "crosstl_primitives.cpp").read_text()
    assert "NO_GPU(AsType)" in (backend / "primitives.cpp").read_text()
    assert "NO_GPU(Full)" not in (backend / "crosstl_primitives.cpp").read_text()
    assert "NO_GPU(Full)" in (backend / "primitives.cpp").read_text()
    assert (
        "NO_GPU(BitwiseBinary)" not in (backend / "crosstl_primitives.cpp").read_text()
    )
    assert "NO_GPU(BitwiseBinary)" in (backend / "primitives.cpp").read_text()
    for name in prepare.VIEW_PRIMITIVES:
        macro = "NO_GPU_MULTI" if name in prepare.MULTI_OUTPUT_VIEWS else "NO_GPU"
        assert (
            f"{macro}({name})" not in (backend / "crosstl_primitives.cpp").read_text()
        )
        assert f"{macro}({name})" in (backend / "primitives.cpp").read_text()
    assert "NO_GPU(Power)" in (backend / "crosstl_primitives.cpp").read_text()
    assert "NO_GPU(Arange)" in (backend / "primitives.cpp").read_text()
    events = (backend / "crosstl_event.cpp").read_text()
    assert events.count("stream.device == Device::gpu") == 2
    assert "cpu_wait()" in events and "cpu_signal()" in events
    cmake = (backend / "CMakeLists.txt").read_text()
    assert 'host backend" OFF)' in cmake
    assert "else()\ntarget_sources(mlx PRIVATE primitives.cpp)" in cmake
    for name, digest in record["files"].items():
        assert (
            prepare.hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
        )
    assert prepare.verify_prepared(tmp_path) == record


@pytest.mark.parametrize(
    "fault", ["revision", "unrelated", "missing", "modified", "symlink"]
)
def test_prepared_verifier_rejects_source_changes(tmp_path, monkeypatch, fault):
    backend = checkout(tmp_path, monkeypatch)
    prepare.prepare(tmp_path, tmp_path / "adaptation.json")
    original_git = prepare.subprocess.check_output

    def git(command, **kwargs):
        if fault == "revision" and "rev-parse" in command:
            return "wrong"
        if fault == "unrelated" and "diff" in command:
            return (
                b"mlx/backend/no_gpu/CMakeLists.txt\0mlx/backend/cpu/primitives.cpp\0"
            )
        return original_git(command, **kwargs)

    monkeypatch.setattr(prepare.subprocess, "check_output", git)
    source = backend / "crosstl_backend.cpp"
    if fault == "missing":
        source.unlink()
    elif fault == "modified":
        source.write_bytes(source.read_bytes() + b"\n// changed\n")
    elif fault == "symlink":
        is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path, "is_symlink", lambda path: path == source or is_symlink(path)
        )
    with pytest.raises(ValueError):
        prepare.verify_prepared(tmp_path)


@pytest.mark.parametrize(
    "fault", ["revision", "dirty", "existing", "source", "evidence"]
)
def test_prepare_rejects_invalid_checkout_before_edits(tmp_path, monkeypatch, fault):
    backend = checkout(tmp_path, monkeypatch)
    if fault in {"revision", "dirty"}:
        monkeypatch.setattr(
            prepare.subprocess,
            "check_output",
            lambda command, **kwargs: (
                ("wrong" if fault == "revision" else prepare.COMMIT)
                if "rev-parse" in command
                else "modified" if fault == "dirty" else ""
            ),
        )
    elif fault == "existing":
        (backend / "crosstl_dispatch.h").write_text("local changes")
    elif fault == "source":
        (backend / "primitives.cpp").write_text("NO_GPU(Abs)")
    elif fault == "evidence":
        (tmp_path / "adaptation.json").write_text("prior evidence")
    before = (backend / "CMakeLists.txt").read_bytes()
    with pytest.raises(ValueError):
        prepare.prepare(tmp_path, tmp_path / "adaptation.json")
    assert (backend / "CMakeLists.txt").read_bytes() == before
    assert not (backend / "crosstl_backend.cpp").exists()


@pytest.fixture(scope="module", params=["opengl", "directx", "metal"])
def package_templates(tmp_path_factory, request):
    tmp_path = tmp_path_factory.mktemp(request.param)
    root = tmp_path / "mlx"
    source = root / packages.SOURCE
    source.parent.mkdir(parents=True)
    types = {
        "bool_": "bool",
        "float32": "float",
        "int32": "int",
        "uint32": "uint",
        "int64": "long",
        "uint64": "ulong",
    }
    source.write_text(
        """template <typename T>
[[kernel]] void arange(constant T& start [[buffer(0)]],
    constant T& step [[buffer(1)]], device T* out [[buffer(2)]],
    uint index [[thread_position_in_grid]]) { out[index] = start + index * step; }
"""
        + "\n".join(
            f'template [[host_name("arange{dtype}")]] [[kernel]] '
            f"decltype(arange<{metal}>) arange<{metal}>;"
            for dtype, metal in types.items()
            if dtype != "bool_"
        )
    )
    (root / packages.UNARY_SOURCE).write_text(
        "template <int op> kernel void unary(device const float* in [[buffer(0)]], "
        "device float* out [[buffer(1)]], constant uint& size [[buffer(2)]], "
        "uint index [[thread_position_in_grid]]) { if (index < size) out[index] = in[index]; }\n"
        + "\n".join(
            f'template [[host_name("{entry}")]] [[kernel]] '
            f"decltype(unary<{index}>) unary<{index}>;"
            for index, entry in enumerate(packages.UNARY_ENTRIES)
        )
        + "\ntemplate <typename T> kernel void unary_bool(device const T* in [[buffer(0)]], device T* out [[buffer(1)]], constant uint& size [[buffer(2)]], uint index [[thread_position_in_grid]]) { if (index < size) out[index] = !in[index]; }\n"
        + f'template [[host_name("{packages.LOGICAL_NOT_ENTRY}")]] [[kernel]] decltype(unary_bool<bool>) unary_bool<bool>;\n'
    )
    (root / packages.COPY_SOURCE).write_text(
        """template <typename T, int n> kernel void copy_words(
device const T* src [[buffer(0)]], device T* dst [[buffer(1)]],
constant int* src_shape [[buffer(2)]], constant long* src_strides [[buffer(3)]],
constant long* dst_strides [[buffer(4)]], constant int& ndim [[buffer(5)]],
constant long& src_offset [[buffer(6)]], constant long& dst_offset [[buffer(7)]],
uint index [[thread_position_in_grid]]) { dst[index] = src[index]; }
"""
        + f'template [[host_name("{packages.COPY_ENTRY}")]] [[kernel]] '
        + "decltype(copy_words<uint, 2>) copy_words<uint, 2>;\n"
        + f'template [[host_name("{packages.BOOLEAN_COPY_ENTRY}")]] [[kernel]] decltype(copy_words<bool, 2>) copy_words<bool, 2>;\n'
        + """template <typename T, typename U> kernel void cast_values(
device const T* src [[buffer(0)]], device U* dst [[buffer(1)]],
constant uint& size [[buffer(2)]], uint index [[thread_position_in_grid]]) {
  if (index < size) dst[index] = U(src[index]);
}
"""
        + "\n".join(
            f'template [[host_name("{entry}")]] [[kernel]] '
            f"decltype(cast_values<{types[src]}, {types[dst]}>) cast_values<{types[src]}, {types[dst]}>;"
            for entry, (src, dst) in {
                **packages.CAST_ENTRIES,
                **packages.BOOLEAN_CAST_ENTRIES,
            }.items()
        )
    )
    (root / packages.BINARY_SOURCE).write_text(
        """template <typename T, typename U, int op> kernel void binary(
device const T* a [[buffer(0)]], device const T* b [[buffer(1)]],
device U* c [[buffer(2)]], constant uint& size [[buffer(3)]],
uint index [[thread_position_in_grid]]) { if (index < size) c[index] = U(a[index] + b[index]); }
"""
        + "\n".join(
            f'template [[host_name("{entry}")]] [[kernel]] '
            f"decltype(binary<{types[dtype]}, {types[output]}, {index}>) binary<{types[dtype]}, {types[output]}, {index}>;"
            for index, (entry, dtype, output) in enumerate(
                [
                    (entry, dtype, dtype)
                    for entry, dtype in packages.BINARY_ENTRIES.items()
                ]
                + [
                    (entry, dtype, "bool_")
                    for entry, dtype in packages.COMPARISON_ENTRIES.items()
                ]
            )
        )
    )
    output = tmp_path / "packages"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            packages.subprocess,
            "check_output",
            lambda command, **kwargs: prepare.COMMIT if "rev-parse" in command else "",
        )
        index = packages.build_packages(root, output, request.param)
    assert set(index["descriptors"]) == set(packages.ENTRIES)
    assert (output / "translation/report.json").is_file()
    assert (
        'binary32_fma_profile = "rne-flush"'
        in (output / "translation/crosstl.toml").read_text()
    )
    return output


@pytest.fixture
def translated_packages(tmp_path, package_templates):
    return Path(shutil.copytree(package_templates, tmp_path / "packages"))


def test_runtime_selects_the_requested_native_adapter(translated_packages, tmp_path):
    host = runtime.HostRuntime(translated_packages, tmp_path / "trace")
    adapters = {
        "directx": runtime.DirectXRuntimeParityAdapter,
        "opengl": runtime.OpenGLRuntimeParityAdapter,
        "metal": runtime.MetalRuntimeParityAdapter,
    }
    assert type(host.executor.runtime_adapter) is adapters[host.target]


def native_buffers(dtype="float32", count=3):
    ctype = runtime.TYPES[dtype]
    memory = [ctype(2), ctype(3), (ctype * count)()]
    buffers = (runtime.Buffer * 3)(
        *(
            runtime.Buffer(
                name.encode(),
                dtype.encode(),
                ctypes.addressof(value),
                count if name == "out" else 1,
                int(name == "out"),
            )
            for name, value in zip(("start", "step", "out"), memory)
        )
    )
    return buffers, memory


def unary_buffers(values=(-2.0, 0.0, 3.0)):
    memory = [
        (ctypes.c_float * len(values))(*values),
        (ctypes.c_float * len(values))(),
        ctypes.c_uint32(len(values)),
    ]
    buffers = (runtime.Buffer * 3)(
        *[
            runtime.Buffer(
                name.encode(),
                b"uint32" if name == "size" else b"float32",
                ctypes.addressof(value),
                1 if name == "size" else len(values),
                int(name == "out"),
            )
            for name, value in zip(("in", "out", "size"), memory)
        ]
    )
    return buffers, memory


def copy_buffers(dtype="uint32"):
    values = {
        "src": list(copy_workloads.source_words()[:12]),
        "dst": [0] * 6,
        "src_shape": [2, 3],
        "src_strides": [7, -2],
        "dst_strides": [3, 1],
        "ndim": [2],
        "src_offset": [4],
        "dst_offset": [0],
    }
    if dtype == "bool_":
        values["src"] = [value % 2 for value in values["src"]]
    dtypes = {**runtime.copy_layout.DTYPES, "src": dtype, "dst": dtype}
    memory = {
        name: (runtime.TYPES[dtypes[name]] * len(data))(*data)
        for name, data in values.items()
    }
    buffers = (runtime.Buffer * 8)(
        *[
            runtime.Buffer(
                name.encode(),
                dtypes[name].encode(),
                ctypes.addressof(data),
                len(data),
                int(name == "dst"),
            )
            for name, data in memory.items()
        ]
    )
    return buffers, memory


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "count",
        "dtype",
        "direction",
        "rank",
        "metadata-count",
        "shape",
        "zero-shape",
        "source-short",
        "lower-bound",
        "upper-bound",
        "stride",
        "destination-stride",
        "destination-offset",
        "duplicate",
        "null",
        "guard",
    ],
)
def test_copy_dispatch_checks_metadata_before_submission(
    translated_packages, tmp_path, monkeypatch, fault
):
    host = runtime.HostRuntime(translated_packages, tmp_path / "trace")
    buffers, memory = copy_buffers()
    if fault == "dtype":
        buffers[0].dtype = b"float32"
    elif fault == "direction":
        buffers[0].output = 1
    elif fault == "rank":
        memory["ndim"][0] = 3
    elif fault == "metadata-count":
        buffers[3].count = 1
    elif fault == "shape":
        memory["src_shape"][0] = 3
    elif fault == "zero-shape":
        memory["src_shape"][0] = 0
    elif fault == "source-short":
        buffers[0].count = 11
    elif fault == "lower-bound":
        memory["src_offset"][0] = 3
    elif fault == "upper-bound":
        memory["src_offset"][0] = 5
    elif fault == "stride":
        memory["src_strides"][0] = 2**63 - 1
    elif fault == "destination-stride":
        memory["dst_strides"][0] = 1
    elif fault == "destination-offset":
        memory["dst_offset"][0] = 1
    elif fault == "duplicate":
        buffers[1].name = b"src"
    elif fault == "null":
        buffers[0].data = None
    calls = []
    expected = [memory["src"][index] for index in (4, 2, 0, 11, 9, 7)]
    original_build = runtime.build_native_loader_dispatch_request

    def build(descriptor, package, inputs, outputs, *args, **kwargs):
        output_name = next(iter(outputs))
        assert inputs[output_name]["values"] == [0] * 6 + runtime.COPY_GUARD
        return original_build(descriptor, package, inputs, outputs, *args, **kwargs)

    def execute(request):
        calls.append(request)
        output = next(
            binding["name"]
            for binding in host.descriptors[packages.COPY_ENTRY]["bindings"]
            if binding["access"] == "read_write"
        )
        return SimpleNamespace(
            status="ok",
            outputs={
                output: {
                    "dtype": "uint32",
                    "shape": [38],
                    "values": (
                        expected
                        + ([0] * 32 if fault == "guard" else runtime.COPY_GUARD)
                    ),
                }
            },
            details={},
        )

    monkeypatch.setattr(host.executor, "run", execute)
    monkeypatch.setattr(runtime, "build_native_loader_dispatch_request", build)
    if fault == "guard":
        with pytest.raises(RuntimeError, match="buffer guard"):
            host.dispatch(packages.COPY_ENTRY, buffers, 8, 6)
        assert len(calls) == 1 and list(memory["dst"]) == [0] * 6
        assert not host.trace.exists()
    elif fault:
        with pytest.raises(ValueError):
            host.dispatch(packages.COPY_ENTRY, buffers, 7 if fault == "count" else 8, 6)
        assert not calls and not host.trace.exists()
    else:
        host.dispatch(packages.COPY_ENTRY, buffers, 8, 6)
        assert list(memory["dst"]) == expected
        assert calls[0].execution_plan.dispatch.workgroup_count == (2, 2, 1)
        assert calls[0].execution_plan.dispatch.workgroup_size == (1, 1, 1)
        trace = json.loads(host.trace.read_text())
        assert trace["threads"] == 6 and trace["workgroupCount"] == [2, 2, 1]
        assert trace["copyGuardWords"] == runtime.COPY_GUARD


@pytest.mark.parametrize("dtype", ["uint32", "bool_"])
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "fault",
    [None, "direction", "overlap", "collision", "bounds", "span", "untouched", "guard"],
)
def test_strided_copy_preserves_other_destination_elements(
    translated_packages, tmp_path, monkeypatch, dtype, preserve, reverse, fault
):
    host = runtime.HostRuntime(translated_packages, tmp_path / "trace")
    buffers, memory = copy_buffers(dtype)
    initial = [bool(i % 2) if dtype == "bool_" else 1000 + i for i in range(14)]
    memory["dst"] = (runtime.TYPES[dtype] * 14)(*initial)
    buffers[1].data = ctypes.addressof(memory["dst"])
    buffers[1].count = 14
    buffers[1].output = runtime.copy_layout.INOUT if preserve else 1
    memory["dst_strides"][:] = [-7 if reverse else 7, 2]
    memory["dst_offset"][0] = 8 if reverse else 1
    indices = [8, 10, 12, 1, 3, 5] if reverse else [1, 3, 5, 8, 10, 12]
    expected = list(initial) if preserve else [False if dtype == "bool_" else 0] * 14
    for destination, source in zip(indices, [4, 2, 0, 11, 9, 7]):
        expected[destination] = memory["src"][source]
    if fault == "direction":
        buffers[1].output = 3
    elif fault == "overlap":
        buffers[1].data = buffers[0].data
    elif fault == "collision":
        memory["dst_strides"][:] = [2, 2]
        memory["dst_offset"][0] = 0
    elif fault == "bounds":
        memory["dst_offset"][0] = 14
    elif fault == "span":
        buffers[1].count = 2**31
    calls = []
    entry = packages.BOOLEAN_COPY_ENTRY if dtype == "bool_" else packages.COPY_ENTRY
    physical = runtime.physical_dtype(dtype, host.target)
    guard = (
        runtime.COPY_GUARD
        if dtype != "bool_"
        else (
            runtime.BOOLEAN_GUARD
            if host.target == "metal"
            else [int(v) for v in runtime.BOOLEAN_GUARD]
        )
    )

    def execute(request):
        calls.append(request)
        binding = next(
            item["name"]
            for item in host.descriptors[entry]["bindings"]
            if item["access"] == "read_write"
        )
        uploaded = next(item for item in request.fixture.inputs if item.name == binding)
        assert list(uploaded.values[:14]) == (initial if preserve else [0] * 14)
        values = list(expected) + guard
        if physical == "uint32":
            values = [int(v) for v in values]
        else:
            values = [bool(v) for v in values]
        if fault == "untouched":
            values[0] = not values[0] if physical == "bool" else int(not values[0])
        elif fault == "guard":
            values[-1] = not values[-1] if physical == "bool" else int(not values[-1])
        return SimpleNamespace(
            status="ok",
            outputs={binding: {"dtype": physical, "shape": [46], "values": values}},
            details={},
        )

    monkeypatch.setattr(host.executor, "run", execute)
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            host.dispatch(entry, buffers, 8, 6)
        assert list(memory["dst"]) == initial
        assert len(calls) == int(fault in {"untouched", "guard"})
        assert not host.trace.exists()
    else:
        host.dispatch(entry, buffers, 8, 6)
        assert list(memory["dst"]) == expected
        event = json.loads(host.trace.read_text())
        assert event["copyMetadata"]["preserveDestination"] is preserve
        assert event["copyValues"] == expected


def test_copy_destination_span_is_independent_of_dispatch_size():
    buffers, memory = copy_buffers()
    memory["dst"] = (ctypes.c_uint32 * 65536)()
    buffers[1].data = ctypes.addressof(memory["dst"])
    buffers[1].count = 65536
    memory["dst_offset"][0] = 65530
    metadata = runtime.copy_layout.validate({b.name.decode(): b for b in buffers}, 6)
    assert list(runtime.copy_layout.destination_indices(metadata)) == list(
        range(65530, 65536)
    )
    assert metadata["workgroupCount"] == [2, 2, 1]


@pytest.mark.parametrize(
    "fault", [None, "missing", "shape", "dtype", "payload", "zero", "source"]
)
def test_copy_workload_references_reject_incomplete_or_changed_words(fault):
    records = copy_workloads.expected_records()
    assert len(records) == 33 and len(copy_workloads.dispatches()) == 27
    if fault == "missing":
        records.pop()
    elif fault == "shape":
        records[0]["shape"] = [15]
    elif fault == "dtype":
        records[0]["dtype"] = "float64"
    elif fault in {"payload", "zero"}:
        index = next(
            i
            for i, word in enumerate(records[0]["words"])
            if word == (0x7FC12345 if fault == "payload" else 0x80000000)
        )
        records[0]["words"][index] ^= 1 if fault == "payload" else 0x80000000
    elif fault == "source":
        records[-1]["words"][3] ^= 1
    if fault:
        with pytest.raises(RuntimeError, match="layout-copy"):
            copy_workloads.validate(records)
    else:
        copy_workloads.validate(records)


@pytest.mark.parametrize("entry", list(packages.CAST_ENTRIES))
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "count",
        "size",
        "source-type",
        "destination-type",
        "direction",
        "input-count",
        "guard",
    ],
)
def test_cast_dispatch_contract(
    translated_packages, tmp_path, monkeypatch, entry, fault
):
    host = runtime.HostRuntime(translated_packages, tmp_path / "trace")
    source, destination = packages.CAST_ENTRIES[entry]
    memory = [
        (runtime.TYPES[source] * 3)(1, 2, 3),
        (runtime.TYPES[destination] * 3)(),
        ctypes.c_uint32(3),
    ]
    buffers = (runtime.Buffer * 3)(
        *[
            runtime.Buffer(
                name.encode(),
                dtype.encode(),
                ctypes.addressof(data),
                1 if name == "size" else 3,
                int(name == "dst"),
            )
            for name, dtype, data in zip(
                ("src", "dst", "size"), (source, destination, "uint32"), memory
            )
        ]
    )
    if fault == "size":
        memory[2].value = 2
    elif fault == "source-type":
        buffers[0].dtype = destination.encode()
    elif fault == "destination-type":
        buffers[1].dtype = source.encode()
    elif fault == "direction":
        buffers[0].output = 1
    elif fault == "input-count":
        buffers[0].count = 2
    guard = (
        runtime.COPY_GUARD
        if destination != "float32"
        else [
            ctypes.c_float.from_buffer_copy(ctypes.c_uint32(word)).value
            for word in runtime.COPY_GUARD
        ]
    )
    calls = []
    original_build = runtime.build_native_loader_dispatch_request

    def build(descriptor, package, inputs, outputs, *args, **kwargs):
        name = next(iter(outputs))
        assert inputs[name]["values"] == [0] * 3 + guard
        return original_build(descriptor, package, inputs, outputs, *args, **kwargs)

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
                    "dtype": destination,
                    "shape": [35],
                    "values": [1, 2, 3] + ([0] * 32 if fault == "guard" else guard),
                }
            },
            details={},
        )

    monkeypatch.setattr(runtime, "build_native_loader_dispatch_request", build)
    monkeypatch.setattr(host.executor, "run", execute)
    if fault == "guard":
        with pytest.raises(RuntimeError, match="buffer guard"):
            host.dispatch(entry, buffers, 3, 3)
        assert list(memory[1]) == [0] * 3 and not host.trace.exists()
    elif fault:
        with pytest.raises(ValueError):
            host.dispatch(entry, buffers, 2 if fault == "count" else 3, 3)
        assert not calls
    else:
        host.dispatch(entry, buffers, 3, 3)
        assert list(memory[1]) == [1, 2, 3]
        assert calls[0].execution_plan.dispatch.workgroup_count == (3, 1, 1)
        assert json.loads(host.trace.read_text())["castGuardValues"] == guard


@pytest.mark.parametrize(
    "fault",
    [None, "missing", "identity", "shape", "type", "source", "rounding", "boolean"],
)
def test_cast_references_require_exact_results(fault):
    records = cast_workloads.expected_records()
    assert len(records) == 50 and len(cast_workloads.dispatches()) == 64
    assert sum(len(record["words"]) for record in records) == 1976
    if fault == "missing":
        records.pop()
    elif fault == "identity":
        records[1]["entry"] = "other"
    elif fault == "shape":
        records[1]["shape"] = [1]
    elif fault == "type":
        records[1]["dtype"] = "int64"
    elif fault == "source":
        records[-1]["sourceWords"][0] ^= 1
    elif fault == "rounding":
        record = next(
            record
            for record in records
            if record["entry"] == "v_copyint32float32" and record["layout"] == "tail"
        )
        record["words"][5] ^= 1
    elif fault == "boolean":
        records[1]["words"][0] = False
    if fault:
        with pytest.raises(RuntimeError, match="cast readbacks"):
            cast_workloads.validate(records)
    else:
        cast_workloads.validate(records)


@pytest.mark.parametrize("dtype", ["float32", "int32", "uint32"])
@pytest.mark.parametrize(
    "fault", [None, "count", "size", "dtype", "direction", "input-count", "guard"]
)
def test_binary_dispatch_contract(
    translated_packages, tmp_path, monkeypatch, dtype, fault
):
    host = runtime.HostRuntime(translated_packages, tmp_path / "trace")
    ctype = runtime.TYPES[dtype]
    memory = [
        (ctype * 3)(1, 2, 3),
        (ctype * 3)(4, 5, 6),
        (ctype * 3)(),
        ctypes.c_uint32(3),
    ]
    buffers = (runtime.Buffer * 4)(
        *[
            runtime.Buffer(
                name.encode(),
                ("uint32" if name == "size" else dtype).encode(),
                ctypes.addressof(data),
                1 if name == "size" else 3,
                int(name == "c"),
            )
            for name, data in zip(("a", "b", "c", "size"), memory)
        ]
    )
    if fault == "size":
        memory[-1].value = 2
    elif fault == "dtype":
        buffers[1].dtype = b"int64"
    elif fault == "direction":
        buffers[0].output = 1
    elif fault == "input-count":
        buffers[0].count = 2
    calls = []
    entry = "vv_Add" + dtype
    guard = (
        runtime.COPY_GUARD
        if dtype != "float32"
        else [
            ctypes.c_float.from_buffer_copy(ctypes.c_uint32(word)).value
            for word in runtime.COPY_GUARD
        ]
    )
    original_build = runtime.build_native_loader_dispatch_request

    def build(descriptor, package, inputs, outputs, *args, **kwargs):
        name = next(iter(outputs))
        assert inputs[name]["values"] == [0] * 3 + guard
        return original_build(descriptor, package, inputs, outputs, *args, **kwargs)

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
                    "dtype": dtype,
                    "shape": [35],
                    "values": [5, 7, 9] + ([0] * 32 if fault == "guard" else guard),
                }
            },
            details={},
        )

    monkeypatch.setattr(runtime, "build_native_loader_dispatch_request", build)
    monkeypatch.setattr(host.executor, "run", execute)
    if fault == "guard":
        with pytest.raises(RuntimeError, match="buffer guard"):
            host.dispatch(entry, buffers, 4, 3)
        assert list(memory[2]) == [0] * 3 and not host.trace.exists()
    elif fault:
        with pytest.raises(ValueError):
            host.dispatch(entry, buffers, 3 if fault == "count" else 4, 3)
        assert not calls
    else:
        host.dispatch(entry, buffers, 4, 3)
        assert list(memory[2]) == [5, 7, 9]
        assert calls[0].execution_plan.dispatch.workgroup_count == (3, 1, 1)
        assert json.loads(host.trace.read_text())["binaryGuardValues"] == guard


@pytest.mark.parametrize(
    "fault",
    [None, "missing", "identity", "value", "integer", "source", "nonfinite", "zero"],
)
def test_binary_reference_checks_reject_changed_results(fault):
    records = binary_workloads.expected_records()
    assert len(records) == 134 and len(binary_workloads.dispatches()) == 198
    if fault == "missing":
        records.pop()
    elif fault == "identity":
        records[0]["layout"] = "other"
    elif fault == "value":
        records[1]["values"][0] += 1
    elif fault == "integer":
        record = next(
            row for row in records if row["dtype"] == "uint32" and row["values"]
        )
        record["values"][0] = float(record["values"][0])
    elif fault == "source":
        records[1]["aWords"][0] ^= 1
    elif fault == "nonfinite":
        record = next(row for row in records if row["layout"] == "nonfinite")
        record["values"][0] = 0
    elif fault == "zero":
        record = next(
            row
            for row in records
            if row["entry"] == "vv_Multiplyfloat32" and row["layout"] == "nonfinite"
        )
        assert math.copysign(1, record["values"][3]) == -1
        record["values"][3] = 0.0
    if fault:
        with pytest.raises(RuntimeError):
            binary_workloads.validate(records)
    else:
        binary_workloads.validate(records)


@pytest.mark.parametrize("operation", ["Minimum", "Maximum"])
@pytest.mark.parametrize("index", [3, 4])
def test_binary_extrema_preserve_second_operand_zero_sign(operation, index):
    records = binary_workloads.expected_records()
    record = next(
        row
        for row in records
        if row["entry"] == f"vv_{operation}float32" and row["layout"] == "nonfinite"
    )
    assert record["values"][2] == "nan"
    assert record["values"][index] == 0
    assert math.copysign(1, record["values"][index]) == (-1 if index == 3 else 1)
    binary_workloads.validate(records)
    record["values"][index] = -record["values"][index]
    with pytest.raises(RuntimeError, match="Incorrect float32 binary result"):
        binary_workloads.validate(records)


@pytest.mark.parametrize("fault", [None, "size", "input-count", "dtype", "direction"])
def test_unary_dispatch_contract(translated_packages, tmp_path, monkeypatch, fault):
    host = runtime.HostRuntime(translated_packages, tmp_path / "trace")
    buffers, memory = unary_buffers()
    if fault == "size":
        memory[2].value = 2
    elif fault == "input-count":
        buffers[0].count = 2
    elif fault == "dtype":
        buffers[0].dtype = b"int32"
    elif fault == "direction":
        buffers[0].output = 1
    calls = []

    def execute(request):
        calls.append(request)
        output = next(
            binding["name"]
            for binding in host.descriptors["v_Absfloat32float32"]["bindings"]
            if binding["access"] == "read_write"
        )
        return SimpleNamespace(
            status="ok",
            outputs={output: {"dtype": "float32", "shape": [3], "values": [2, 0, 3]}},
            details={},
        )

    monkeypatch.setattr(host.executor, "run", execute)
    if fault:
        with pytest.raises(ValueError):
            host.dispatch("v_Absfloat32float32", buffers, 3, 3)
        assert calls == [] and not host.trace.exists()
    else:
        host.dispatch("v_Absfloat32float32", buffers, 3, 3)
        assert list(memory[1]) == [2, 0, 3] and len(calls) == 1


@pytest.mark.parametrize("donated", [False, True])
def test_unary_nonfinite_transport(translated_packages, tmp_path, monkeypatch, donated):
    host = runtime.HostRuntime(translated_packages, tmp_path / "trace")
    buffers, memory = unary_buffers((math.nan, math.inf, -math.inf))
    if donated:
        buffers[1].data = buffers[0].data
    original_request = runtime.build_native_loader_dispatch_request
    requests = []

    def build(descriptor, package, inputs, outputs, *args, **kwargs):
        assert any(
            value["values"] == ["nan", "+infinity", "-infinity"]
            for value in inputs.values()
        )
        requests.append(inputs)
        return original_request(descriptor, package, inputs, outputs, *args, **kwargs)

    def execute(request):
        output = next(
            binding["name"]
            for binding in host.descriptors["v_Absfloat32float32"]["bindings"]
            if binding["access"] == "read_write"
        )
        return SimpleNamespace(
            status="ok",
            outputs={
                output: {
                    "dtype": "float32",
                    "shape": [3],
                    "values": ["nan", "+infinity", "+infinity"],
                }
            },
            details={},
        )

    monkeypatch.setattr(runtime, "build_native_loader_dispatch_request", build)
    monkeypatch.setattr(host.executor, "run", execute)
    host.dispatch("v_Absfloat32float32", buffers, 3, 3)
    output = memory[0 if donated else 1]
    assert len(requests) == 1 and math.isnan(output[0])
    assert list(output)[1:] == [math.inf, math.inf]


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "identity",
        "value",
        "same-wrong",
        "zero-sign",
        "zero-value",
        "nonfinite",
        "raw-nan",
        "boolean",
        "nested",
        "acosh-near-one",
        "atan-precision",
        "negative-sign",
    ],
)
def test_unary_verifier_requires_complete_independent_results(fault):
    original = unary_workloads.expected_records(cpu=True)
    translated = unary_workloads.expected_records()
    assert len(translated) == 130
    assert sum(record["count"] for record in translated) == 16202
    assert len(unary_workloads.dispatches()) == 103
    if fault == "missing":
        original.clear()
        translated.clear()
    elif fault == "identity":
        translated[1]["inputs"] = [-3.0]
    elif fault in {"value", "same-wrong"}:
        translated[1]["values"] = [17.0]
        if fault == "same-wrong":
            original[1]["values"] = [17.0]
    elif fault in {"zero-sign", "zero-value"}:
        next(
            record
            for record in translated
            if record["operation"] == "Negative" and record["count"] == 7
        )["values"][3] = (0.0 if fault == "zero-sign" else -1e-8)
    elif fault == "nonfinite":
        next(record for record in translated if record.get("case") == "nonfinite")[
            "values"
        ][-1] = 0.0
    elif fault in {"raw-nan", "boolean", "nested"}:
        translated[1]["values"] = [
            {"raw-nan": math.nan, "boolean": True, "nested": []}[fault]
        ]
    elif fault == "acosh-near-one":
        record = next(
            record for record in translated if record.get("case") == "near-one"
        )
        assert record["count"] == 8193
        assert record["inputs"][0x805] == 1.0002447366714478
        record["values"][0x805] = 0.022124959155917168
    elif fault in {"atan-precision", "negative-sign"}:
        operation = "ArcTan" if fault == "atan-precision" else "Sign"
        record = next(
            record
            for record in translated
            if record["operation"] == operation and record["count"] == 1
        )
        record["values"][0] = 0.12434358894824982 if fault == "atan-precision" else 1.0
    if fault:
        with pytest.raises((RuntimeError, AssertionError)):
            unary_workloads.compare(original, translated)
    else:
        unary_workloads.compare(original, translated)


def test_unary_adapter_definitions_match_packages():
    source = (prepare.HERE / "backend.cpp").read_text()
    for operation in packages.UNARY_OPERATIONS:
        if operation not in {"Log2", "Log10", "Rsqrt"}:
            assert f"CROSSTL_UNARY_GPU({operation})" in source
    assert "set_unary_output_data(in, out)" in source
    assert "in.data_size() > 65535" in source
    assert "!in.flags().contiguous" in source


@pytest.mark.parametrize("profile", ["generic", "apple-accelerate-arm64"])
def test_cpu_erf_zero_profile_does_not_relax_generated_checks(monkeypatch, profile):
    monkeypatch.setattr(unary_workloads, "cpu_reference_profile", lambda: profile)
    cpu = unary_workloads.expected_records(cpu=True)
    for record in cpu:
        if record["operation"] != "Erf":
            continue
        for index, value in enumerate(record["inputs"]):
            if value == 0:
                negative = profile == "generic" or index < record["count"] // 8 * 8
                assert math.copysign(1.0, record["values"][index]) == (
                    -1 if negative else 1
                )
    unary_workloads.validate(cpu, cpu=True)
    with pytest.raises(AssertionError, match="zero sign"):
        unary_workloads.validate(cpu)
    wrong_cpu = unary_workloads.expected_records(cpu=True)
    record = next(
        item for item in wrong_cpu if item["operation"] == "Erf" and item["count"] == 7
    )
    record["values"][2] = -record["values"][2]
    with pytest.raises(AssertionError, match="zero sign"):
        unary_workloads.validate(wrong_cpu, cpu=True)


@pytest.mark.parametrize(
    "system,machine,expected",
    [
        ("Darwin", "arm64", "apple-accelerate-arm64"),
        ("Linux", "aarch64", "generic"),
        ("Windows", "AMD64", "generic"),
    ],
)
def test_cpu_reference_profile_matches_pinned_build(
    monkeypatch, system, machine, expected
):
    monkeypatch.setattr(unary_workloads.platform, "system", lambda: system)
    monkeypatch.setattr(unary_workloads.platform, "machine", lambda: machine)
    assert unary_workloads.cpu_reference_profile() == expected


def test_view_adapter_preserves_upstream_shared_buffer_operations():
    source = (prepare.HERE / "backend.cpp").read_text()
    for name in prepare.VIEW_PRIMITIVES:
        if name in {"Reshape", "Unflatten", "Slice"}:
            assert f"void {name}::eval_gpu(" in source
        else:
            macro = (
                "CROSSTL_SHARED_OUTPUTS_GPU"
                if name in prepare.MULTI_OUTPUT_VIEWS
                else "CROSSTL_SHARED_VIEW_GPU"
            )
            assert f"{macro}({name})" in source
    assert "prepare_reshape(in, out)" in source
    assert "shared_buffer_reshape(in, strides, out)" in source
    assert "slice(inputs[0], out, start_indices_, strides_)" in source
    assert "dispatch_copy(in, out)" in source
    assert "eval_cpu" not in source


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "value",
        "shape",
        "dtype",
        "identity",
        "duplicate",
        "source",
        "int64",
        "nonfinite",
        "slice-order",
        "strided-square",
        "empty-square",
    ],
)
def test_view_verifier_requires_complete_independent_results(fault):
    records = view_workloads.expected_records()
    assert len(records) == 59
    assert sum(len(record["values"]) for record in records) == 461
    assert len(view_workloads.dispatches()) == 34
    assert view_workloads.dispatches()[8:11] == [
        ("v_Squarefloat32float32", 4),
        ("v_Squarefloat32float32", 1),
        ("v_Squarefloat32float32", 6),
    ]
    if fault == "missing":
        records.pop()
    elif fault == "value":
        records[1]["values"][0] += 1
    elif fault == "shape":
        records[0]["shape"] = [12]
    elif fault == "dtype":
        records[0]["dtype"] = "float64"
    elif fault == "identity":
        records[0]["case"] = "other"
    elif fault == "duplicate":
        records[1] = records[0]
    elif fault == "source":
        records[-1]["values"][0] += 1
    elif fault == "int64":
        record = next(
            record for record in records if record["case"] == "int64-transpose"
        )
        record["values"] = [float(value) for value in record["values"]]
    elif fault == "nonfinite":
        records[0]["values"][0] = math.nan
    elif fault == "slice-order":
        record = next(record for record in records if record["case"] == "slice-matrix")
        record["values"].reverse()
    elif fault == "strided-square":
        record = next(record for record in records if record["case"] == "gapped/square")
        record["values"][2] = 9.0
    elif fault == "empty-square":
        record = next(
            record for record in records if record["case"] == "slice-empty/square"
        )
        record["shape"] = [0, 1]
    if fault:
        with pytest.raises((RuntimeError, ValueError)):
            view_workloads.validate(records)
    else:
        view_workloads.validate(records)


@pytest.mark.parametrize("via_callback", [False, True])
@pytest.mark.parametrize("dtype", verify.DTYPES)
def test_translated_binding_contract_and_typed_readback(
    translated_packages, tmp_path, monkeypatch, dtype, via_callback
):
    host = runtime.HostRuntime(translated_packages, tmp_path / "trace")
    buffers, memory = native_buffers(dtype)
    calls = []

    def execute(request):
        calls.append(request)
        output = next(
            binding["name"]
            for binding in host.descriptors["arange" + dtype]["bindings"]
            if binding["access"] == "read_write"
        )
        return SimpleNamespace(
            status="ok",
            outputs={output: {"dtype": dtype, "shape": [3], "values": [2, 5, 8]}},
            details={"testExecutor": True},
        )

    monkeypatch.setattr(host.executor, "run", execute)
    if via_callback:
        launch = runtime.Launch((3, 1, 1), (1, 1, 1))
        error = ctypes.create_string_buffer(256)
        assert (
            host.callback(
                ("arange" + dtype).encode(),
                buffers,
                3,
                3,
                ctypes.byref(launch),
                error,
                len(error),
            )
            == 0
        ), error.value
    else:
        host.dispatch("arange" + dtype, buffers, 3, 3)
    assert list(memory[-1]) == [2, 5, 8] and len(calls) == 1
    assert calls[0].execution_plan.dispatch.workgroup_size == (1, 1, 1)
    assert calls[0].execution_plan.dispatch.workgroup_count == (3, 1, 1)
    trace = json.loads(host.trace.read_text())
    assert trace["target"] == host.target and trace["entry"] == "arange" + dtype
    assert trace["dispatchVersion"] == 3 and trace["workgroupSize"] == [1, 1, 1]


@pytest.mark.parametrize(
    "fault",
    [
        "count",
        "threads",
        "dtype",
        "direction",
        "shape",
        "duplicate",
        "null",
        "layout",
        "binding",
        "missing-binding",
        "missing-artifact",
        "workgroup",
    ],
)
def test_invalid_dispatch_never_reaches_executor(
    translated_packages, tmp_path, monkeypatch, fault
):
    host = runtime.HostRuntime(translated_packages, tmp_path / "trace")
    buffers, memory = native_buffers()
    descriptor = host.descriptors["arangefloat32"]
    count, threads = 3, 3
    if fault == "count":
        count = 2
    elif fault == "threads":
        threads = 65536
    elif fault == "dtype":
        buffers[0].dtype = b"float16"
    elif fault == "direction":
        buffers[0].output = 1
    elif fault == "shape":
        buffers[2].count = 2
    elif fault == "duplicate":
        buffers[1].name = b"start"
    elif fault == "null":
        buffers[0].data = None
    elif fault == "layout":
        descriptor["bindings"][0]["scalarLayout"]["elementStrideBytes"] = 16
    elif fault == "binding":
        descriptor["bindings"][1] = descriptor["bindings"][0]
    elif fault == "missing-binding":
        descriptor["bindings"].pop()
    elif fault == "missing-artifact":
        descriptor["artifact"]["packagePath"] = "missing.glsl"
    elif fault == "workgroup":
        descriptor["entryPoint"]["workgroupSize"] = [2, 1, 1]
    monkeypatch.setattr(
        host.executor, "run", lambda *args: pytest.fail("Invalid request executed")
    )
    with pytest.raises(ValueError):
        host.dispatch("arangefloat32", buffers, count, threads)
    assert list(memory[-1]) == [0, 0, 0]
    assert not host.trace.exists()


@pytest.mark.parametrize("fault", ["status", "output", "dtype", "shape", "length"])
def test_invalid_readback_is_not_copied(
    translated_packages, tmp_path, monkeypatch, fault
):
    host = runtime.HostRuntime(translated_packages, tmp_path / "trace")
    buffers, memory = native_buffers()
    name = next(
        binding["name"]
        for binding in host.descriptors["arangefloat32"]["bindings"]
        if binding["access"] == "read_write"
    )
    value = {"dtype": "float32", "shape": [3], "values": [2, 5, 8]}
    result = SimpleNamespace(status="ok", outputs={name: value}, details={})
    if fault == "status":
        result.status = "failed"
    elif fault == "output":
        result.outputs = {}
    elif fault == "dtype":
        value["dtype"] = "int32"
    elif fault == "shape":
        value["shape"] = [1, 3]
    elif fault == "length":
        value["values"].pop()
    monkeypatch.setattr(host.executor, "run", lambda *args: result)
    with pytest.raises(RuntimeError):
        host.dispatch("arangefloat32", buffers, 3, 3)
    assert list(memory[-1]) == [0, 0, 0] and not host.trace.exists()


def test_callback_reports_bounded_error_without_unwinding(tmp_path, monkeypatch):
    host = runtime.HostRuntime.__new__(runtime.HostRuntime)

    def fail(*args, **kwargs):
        raise ValueError("long error message")

    monkeypatch.setattr(host, "dispatch", fail)
    error = ctypes.create_string_buffer(8)
    launch = ctypes.pointer(runtime.Launch((1, 1, 1), (1, 1, 1)))
    assert host._dispatch(b"entry", None, 0, 0, launch, ctypes.addressof(error), 8) == 1
    assert error.raw == b"long er\0"
    assert host._dispatch(b"entry", None, 0, 0, launch, None, 0) == 1


@pytest.mark.parametrize(
    "count,size", [((7, 3, 2), (32, 4, 1)), ((1, 5, 1), (128, 1, 1))]
)
def test_native_callback_preserves_multidimensional_launch(monkeypatch, count, size):
    host = runtime.HostRuntime.__new__(runtime.HostRuntime)
    calls = []

    def dispatch(entry, buffers, buffer_count, threads, *, launch):
        calls.append((entry, buffer_count, threads, launch.execution()))

    monkeypatch.setattr(host, "dispatch", dispatch)
    callback = runtime.CALLBACK(host._dispatch)
    launch = runtime.Launch(count, size)
    error = ctypes.create_string_buffer(128)
    assert (
        callback(b"reduce", None, 0, 513, ctypes.byref(launch), error, len(error)) == 0
    )
    assert calls == [
        ("reduce", 0, 513, {"workgroupCount": list(count), "workgroupSize": list(size)})
    ]
    assert error.value == b""


def test_native_callback_requires_launch_pointer(monkeypatch):
    host = runtime.HostRuntime.__new__(runtime.HostRuntime)
    monkeypatch.setattr(
        host, "dispatch", lambda *args, **kwargs: pytest.fail("Null launch executed")
    )
    error = ctypes.create_string_buffer(128)
    callback = runtime.CALLBACK(host._dispatch)
    assert callback(b"reduce", None, 0, 513, None, error, len(error)) == 1
    assert b"geometry is missing" in error.value


@pytest.mark.parametrize(
    "count,size",
    [
        ((0, 1, 1), (32, 1, 1)),
        ((1, 65536, 1), (32, 1, 1)),
        ((1, 1, 1), (32, 0, 1)),
        ((1, 1, 1), (32, 64, 1)),
        ((1, 1, 1), (1025, 1, 1)),
    ],
)
def test_invalid_launch_dimensions_fail_closed(count, size):
    with pytest.raises(ValueError, match="geometry exceeds"):
        runtime.Launch(count, size).execution()


@pytest.mark.parametrize(
    "count,size", [((4, 1, 1), (1, 1, 1)), ((3, 1, 1), (32, 1, 1))]
)
def test_native_launch_must_match_integrated_operation(
    translated_packages, tmp_path, monkeypatch, count, size
):
    host = runtime.HostRuntime(translated_packages, tmp_path / "trace")
    buffers, memory = native_buffers()
    monkeypatch.setattr(
        host.executor, "run", lambda *args: pytest.fail("Mismatched launch executed")
    )
    with pytest.raises(ValueError, match="geometry does not match"):
        host.dispatch(
            "arangefloat32", buffers, 3, 3, launch=runtime.Launch(count, size)
        )
    assert list(memory[-1]) == [0, 0, 0] and not host.trace.exists()


@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
def test_registration_retains_callback_and_uses_platform_library(
    tmp_path, monkeypatch, platform
):
    host = runtime.HostRuntime.__new__(runtime.HostRuntime)
    host.callback = runtime.CALLBACK(lambda *args: 0)
    host.entry_available = runtime.ENTRY_AVAILABLE(lambda entry: 0)
    module = SimpleNamespace(__file__=str(tmp_path / "core.pyd"), gpu="gpu")
    selected, loaded, registered = [], [], []
    module.set_default_device = selected.append
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=module))
    monkeypatch.setitem(sys.modules, "mlx.core", module)
    monkeypatch.setattr(runtime.sys, "platform", platform)
    monkeypatch.setattr(runtime, "_installed_runtime", None)

    def register(version, callback, available):
        registered.append((version, callback, available))
        return 0

    def load(path):
        loaded.append(path)
        return SimpleNamespace(crosstl_mlx_register_runtime=register)

    monkeypatch.setattr(runtime.ctypes, "CDLL", load)
    host.install()
    assert runtime._installed_runtime is host
    assert registered == [
        (runtime.DISPATCH_VERSION, host.callback, host.entry_available)
    ] and selected == ["gpu"]
    assert loaded == [
        str(tmp_path / ("mlx.dll" if platform == "win32" else "core.pyd"))
    ]
    with pytest.raises(RuntimeError, match="already installed"):
        host.install()


def test_unary_dtype_rejection_uses_unsupported_width(tmp_path, monkeypatch):
    installed = []
    arrays = []
    module = SimpleNamespace(
        gpu="gpu",
        int16="int16",
        metal=SimpleNamespace(is_available=lambda: False),
        is_available=lambda device: bool(installed),
        default_device=lambda: "gpu",
        array=lambda values, *, dtype: arrays.append((values, dtype)) or values,
        abs=lambda values, *, stream: values,
    )

    def evaluate(value):
        assert arrays == [([-3, 2], "int16")]
        raise ValueError("CrossTL unary dispatch requires float32 arrays")

    module.eval = evaluate
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=module))
    monkeypatch.setitem(sys.modules, "mlx.core", module)
    monkeypatch.setattr(
        verify,
        "HostRuntime",
        lambda *args, **kwargs: SimpleNamespace(install=lambda: installed.append(True)),
    )
    monkeypatch.setenv("DEVICE", "cpu")
    args = SimpleNamespace(
        worker="unary-dtype", output_dir=tmp_path / "proof", packages=tmp_path
    )
    verify.worker(args)
    assert json.loads((args.output_dir / "result.json").read_text())["rejected"] is True


@pytest.mark.parametrize(
    "fault",
    [None, "missing", "payload", "broadcast", "shape", "dtype", "source", "promotion"],
)
def test_full_references_require_exact_storage_and_broadcasts(fault):
    records = full_workloads.expected_records()
    assert len(records) == 62
    assert sum(len(record["words"]) for record in records) == 2421
    assert len(full_workloads.dispatches()) == 52
    if fault == "missing":
        records.pop()
    elif fault == "payload":
        records[1]["words"][0] ^= 1
    elif fault == "broadcast":
        records[3]["words"][5] = records[3]["words"][1]
    elif fault == "shape":
        records[0]["shape"] = [1]
    elif fault == "dtype":
        records[0]["dtype"] = "uint32"
    elif fault == "source":
        records[19]["words"][0] ^= 1
    elif fault == "promotion":
        records[-2]["words"][0] = 1
    if fault:
        with pytest.raises(RuntimeError, match="Full readbacks"):
            full_workloads.validate(records)
    else:
        full_workloads.validate(records)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "command",
        "values",
        "trace",
        "target",
        "arrays",
        "same-wrong-values",
        "truncated-readback",
        "wrong-dtype",
        "boolean-count",
        "tests",
        "skipped",
        "negative",
        "negative-message",
        "threads",
        "duplicate",
        "launch-version",
        "launch-size",
        "launch-count",
        "launch-missing",
        "launch-version-float",
        "launch-size-boolean",
        "source-before",
        "source-after",
        "test-source-before",
        "test-source-after",
        "unary-missing",
        "unary-values",
        "unary-identity",
        "views-missing",
        "views-values",
        "views-shape",
        "copies-missing",
        "copies-values",
        "binary-missing",
        "binary-values",
        "casts-missing",
        "casts-values",
        "full-missing",
        "full-values",
        "upstream-failure",
    ],
)
def test_verifier_keeps_selected_scope_and_rejects_incomplete_evidence(
    tmp_path, monkeypatch, fault
):
    source = tmp_path / "mlx/python/tests/test_ops.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"unchanged tests")
    cast_test_source = source.with_name("test_array.py")
    cast_test_source.write_bytes(
        b"changed tests" if fault == "test-source-before" else b"unchanged tests"
    )
    package_root = tmp_path / "packages"
    package_root.mkdir()
    verify.save(package_root / "index.json", {"target": "opengl"})
    monkeypatch.setattr(
        verify.subprocess,
        "check_output",
        lambda command, **kwargs: (
            prepare.COMMIT if "rev-parse" in command else b"unchanged tests"
        ),
    )
    calls = []
    identities = []

    def prepared(root):
        identities.append(root)
        if fault == "source-before" or (
            fault == "source-after" and len(identities) == 2
        ):
            raise ValueError("Prepared MLX source does not match")
        return {"commit": prepare.COMMIT, "files": {"adapter": "unchanged"}}

    monkeypatch.setattr(verify, "verify_prepared", prepared)

    def run(command, **kwargs):
        calls.append(command)
        if fault == "test-source-after":
            cast_test_source.write_bytes(b"changed tests")
        mode = command[command.index("--worker") + 1]
        assert command[command.index("--timeout-seconds") + 1] == (
            "300" if mode == "native" else "180"
        )
        output = Path(command[-1])
        output.mkdir(parents=True)
        arrays = [
            {
                "dtype": dtype,
                "count": count,
                "values": [2 + 3 * i for i in range(count)],
            }
            for dtype in verify.DTYPES
            for count in verify.COUNTS
        ]
        result = {
            "tests": len(verify.UPSTREAM_TESTS),
            "skipped": 0,
            "failures": 0,
            "errors": 0,
            "arrays": arrays,
            "unary": unary_workloads.expected_records(cpu=mode == "cpu"),
            "views": view_workloads.expected_records(),
            "copies": copy_workloads.expected_records(),
            "binary": binary_workloads.expected_records(),
            "casts": cast_workloads.expected_records(),
            "full": full_workloads.expected_records(),
            "booleans": boolean_workloads.expected_records(),
        }
        if fault == "unary-missing":
            result["unary"] = []
        if fault == "unary-values":
            result["unary"][1]["values"] = [123.0]
        if fault == "unary-identity":
            result["unary"][1]["inputs"] = [123.0]
        if fault == "views-missing":
            result["views"] = []
        if fault == "views-values":
            result["views"][1]["values"][0] += 1
        if fault == "views-shape":
            result["views"][0]["shape"] = [12]
        if fault == "copies-missing":
            result["copies"] = []
        if fault == "copies-values":
            result["copies"][0]["words"][0] ^= 1
        if fault == "binary-missing":
            result["binary"] = []
        if fault == "binary-values":
            result["binary"][1]["values"][0] += 1
        if fault == "casts-missing":
            result["casts"] = []
        if fault == "casts-values":
            result["casts"][1]["words"][0] ^= 1
        if fault == "full-missing":
            result["full"] = []
        if fault == "full-values":
            result["full"][0]["words"][0] ^= 1
        if fault == "upstream-failure":
            result["failures"] = 1
        if fault == "arrays":
            result["arrays"] = arrays[:-1]
        if fault == "same-wrong-values":
            arrays[1]["values"] = [4]
        if fault == "truncated-readback":
            arrays[2]["values"].pop()
        if fault == "wrong-dtype":
            arrays[1]["dtype"] = "uint8"
        if fault == "boolean-count":
            arrays[1]["count"] = True
        if fault == "tests":
            result["tests"] = 0
        if fault == "skipped":
            result["skipped"] = 1
        if mode == "native":
            if fault == "values":
                result["arrays"][1]["values"] = [4]
            dispatches = (
                [
                    (entry, count)
                    for entry in packages.ARANGE_ENTRIES
                    for count in (1, 7, 257)
                ]
                + unary_workloads.dispatches()
                + view_workloads.dispatches()
                + copy_workloads.dispatches()
                + binary_workloads.dispatches()
                + cast_workloads.dispatches()
                + full_workloads.dispatches()
                + boolean_workloads.dispatches()
            )
            trace = [
                {
                    "entry": entry,
                    "target": "directx" if fault == "target" else "opengl",
                    "threads": count,
                    "dispatchVersion": 3,
                    "workgroupCount": [count, 1, 1],
                    "workgroupSize": [1, 1, 1],
                }
                for entry, count in dispatches
                if fault != "trace" or entry != packages.UNARY_ENTRIES[-1]
            ]
            if fault == "threads":
                trace[0]["threads"] = 0
            if fault == "duplicate":
                trace.insert(1, trace[0])
            if fault == "launch-version":
                trace[0]["dispatchVersion"] = 1
            if fault == "launch-size":
                trace[0]["workgroupSize"] = [32, 1, 1]
            if fault == "launch-count":
                trace[0]["workgroupCount"] = [0, 1, 1]
            if fault == "launch-missing":
                trace[0].pop("workgroupCount")
            if fault == "launch-version-float":
                trace[0]["dispatchVersion"] = 2.0
            if fault == "launch-size-boolean":
                trace[0]["workgroupSize"] = [True, 1, 1]
            (output / "dispatch.jsonl").write_text("\n".join(map(json.dumps, trace)))
        verify.save(
            output / "result.json",
            (
                result
                if mode in {"cpu", "native"}
                else {
                    "rejected": fault != "negative",
                    "message": (
                        "unexpected"
                        if fault == "negative-message"
                        else verify.NEGATIVE_CHECKS[mode]
                    ),
                }
            ),
        )
        return SimpleNamespace(returncode=124 if fault == "command" else 0)

    monkeypatch.setattr(verify.subprocess, "run", run)
    args = SimpleNamespace(
        mlx_root=tmp_path / "mlx",
        packages=package_root,
        output_dir=tmp_path / "evidence",
    )
    if fault:
        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            verify.verify(args)
        assert not (args.output_dir / "evidence.json").exists()
        if fault in {"source-before", "test-source-before"}:
            assert calls == []
        if fault == "command":
            assert len(calls) == 20
            assert (
                json.loads((args.output_dir / "cpu.command.json").read_text())[
                    "returncode"
                ]
                == 124
            )
    else:
        evidence = verify.verify(args)
        assert len(calls) == 20 and evidence["dispatchCount"] == 493 + len(
            boolean_workloads.dispatches()
        )
        assert len(identities) == 2
        assert evidence["schemaVersion"] == 2
        assert evidence["adaptation"]["files"] == {"adapter": "unchanged"}
        assert evidence["fullUpstreamSuite"] is False
        assert evidence["fullTranslatedBackend"] is False
        assert evidence["original"] == evidence["translated"]
        assert evidence["upstreamTestSources"] == {
            "python/tests/test_ops.py": (
                verify.hashlib.sha256(b"unchanged tests").hexdigest()
            ),
            "python/tests/test_array.py": (
                verify.hashlib.sha256(b"unchanged tests").hexdigest()
            ),
        }


@pytest.mark.parametrize(
    "job_name", ["portable-host", "small-row-reductions", "reductions"]
)
def test_ci_preserves_upstream_checkout_bytes_in_every_job(job_name):
    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / ".github/workflows/mlx-portable-host.yml"
        ).read_text()
    )
    job = workflow["jobs"][job_name]
    checkouts = [
        step for step in job["steps"] if "git init mlx-upstream" in step.get("run", "")
    ]
    assert len(checkouts) == 1
    step = checkouts[0]
    commands = [line.strip() for line in step["run"].splitlines()]
    assert (
        commands.index("git init mlx-upstream")
        < commands.index("git -C mlx-upstream config core.autocrlf false")
        < commands.index("git -C mlx-upstream checkout --detach FETCH_HEAD")
    )
    assert "if" not in step and "continue-on-error" not in step


@pytest.mark.parametrize("content", [b"unchanged\n", b"unchanged\r\n", b"modified\n"])
def test_upstream_test_sources_requires_exact_git_bytes(tmp_path, monkeypatch, content):
    monkeypatch.setattr(
        verify, "UPSTREAM_TESTS", ("test_ops.TestOps.test_concatenate",)
    )
    name = "python/tests/test_ops.py"
    source = tmp_path / name
    source.parent.mkdir(parents=True)
    source.write_bytes(content)

    def git_blob(command, *, timeout):
        assert command == ["git", "-C", str(tmp_path), "show", f"HEAD:{name}"]
        assert timeout == 30
        return b"unchanged\n"

    monkeypatch.setattr(verify.subprocess, "check_output", git_blob)
    if content == b"unchanged\n":
        assert verify.upstream_test_sources(tmp_path) == {
            name: verify.hashlib.sha256(content).hexdigest()
        }
    else:
        with pytest.raises(ValueError, match="Upstream test source was modified"):
            verify.upstream_test_sources(tmp_path)


def test_ci_requires_all_native_platforms_and_retains_evidence():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/mlx-portable-host.yml").read_text()
    for required in (
        prepare.COMMIT,
        "ubuntu-24.04",
        "windows-2025",
        "macos-26",
        'python-version: "3.12"',
        "Initialize execution evidence",
        "tee .mlx-portable-host/dependencies.log",
        "python -m pip install moderngl PyOpenGL",
        'pip install -e ".[directx-runtime]"',
        "target: opengl",
        "target: directx",
        "target: metal",
        "Verify macOS Metal toolchain",
        "tee .mlx-portable-host/metal-version.txt",
        "tee .mlx-portable-host/swift-version.txt",
        "-DMLX_BUILD_METAL=OFF",
        "-DMLX_BUILD_CUDA=OFF",
        "-DMLX_CROSTL_HOST=ON",
        "portable_host.prepare",
        "portable_host.packages",
        "portable_host.verify",
        "pytest -q -n auto tests/test_mlx_portable_host.py",
        "liblapacke-dev",
        "--timeout-seconds 4000",
        "--timeout-seconds 2400",
        "if: always()",
        "include-hidden-files: true",
        "Get-FileHash",
        "core.autocrlf false",
    ):
        assert required in workflow
    assert "continue-on-error" not in workflow
    assert "opengl-runtime" not in workflow
    assert len(verify.UPSTREAM_TESTS) == 34
    assert "test_ops.TestOps.test_diff" in verify.UPSTREAM_TESTS
    assert "test_ops.TestOps.test_flip" in verify.UPSTREAM_TESTS
    assert "test_array.TestArray.test_array_type_cast" in verify.UPSTREAM_TESTS
    assert "test_ops.TestOps.test_hamming_general" in verify.UPSTREAM_TESTS
    assert "test_ops.TestOps.test_comparisons" in verify.UPSTREAM_TESTS
    assert "test_ops.TestOps.test_logical_not" in verify.UPSTREAM_TESTS
    assert "test_ops.TestOps.test_logical_xor" in verify.UPSTREAM_TESTS
    assert "test_ops.TestOps.test_isclose" in verify.UPSTREAM_TESTS
    assert "test_ops.TestOps.test_allclose" in verify.UPSTREAM_TESTS


@pytest.mark.parametrize(
    "entry",
    [
        *packages.COMPARISON_ENTRIES,
        *packages.BOOLEAN_CAST_ENTRIES,
        packages.BOOLEAN_COPY_ENTRY,
        packages.LOGICAL_NOT_ENTRY,
    ],
)
def test_boolean_host_physical_dispatch(
    translated_packages, tmp_path, monkeypatch, entry
):
    host = runtime.HostRuntime(translated_packages, tmp_path / "trace")
    if entry == packages.BOOLEAN_COPY_ENTRY:
        buffers, memory = copy_buffers("bool_")
        destination, count = memory["dst"], 6
    else:
        if entry in packages.COMPARISON_ENTRIES:
            names = ("a", "b", "c", "size")
            dtypes = (
                packages.COMPARISON_ENTRIES[entry],
                packages.COMPARISON_ENTRIES[entry],
                "bool_",
                "uint32",
            )
        elif entry in packages.BOOLEAN_CAST_ENTRIES:
            names = ("src", "dst", "size")
            dtypes = (*packages.BOOLEAN_CAST_ENTRIES[entry], "uint32")
        else:
            names, dtypes = ("in", "out", "size"), ("bool_", "bool_", "uint32")
        memory = [
            (runtime.TYPES[dtype] * (1 if name == "size" else 3))(
                *([3] if name == "size" else [0, 1, 1])
            )
            for name, dtype in zip(names, dtypes)
        ]
        destination, count = memory[-2], 3
        buffers = (runtime.Buffer * len(names))(
            *[
                runtime.Buffer(
                    name.encode(),
                    dtype.encode(),
                    ctypes.addressof(value),
                    len(value),
                    int(index == len(names) - 2),
                )
                for index, (name, dtype, value) in enumerate(zip(names, dtypes, memory))
            ]
        )
    dtype = next(buffer.dtype.decode() for buffer in buffers if buffer.output)
    physical = runtime.physical_dtype(dtype, host.target)
    expected = [index % 2 for index in range(count)]
    body = [bool(value) for value in expected] if physical == "bool" else expected
    guard = (
        runtime.BOOLEAN_GUARD
        if physical == "bool"
        else (
            [int(value) for value in runtime.BOOLEAN_GUARD]
            if dtype == "bool_"
            else runtime.COPY_GUARD
        )
    )
    if dtype == "float32":
        guard = [
            ctypes.c_float.from_buffer_copy(ctypes.c_uint32(word)).value
            for word in runtime.COPY_GUARD
        ]
    calls = []

    def execute(request):
        calls.append(request)
        name = next(value.name for value in request.fixture.expected_outputs)
        input_value = next(
            value for value in request.fixture.inputs if value.name == name
        )
        assert (
            input_value.values
            == ([False] * count if physical == "bool" else [0] * count) + guard
        )
        return SimpleNamespace(
            status="ok",
            outputs={
                name: {"dtype": physical, "shape": [count + 32], "values": body + guard}
            },
            details={},
        )

    monkeypatch.setattr(host.executor, "run", execute)
    host.dispatch(entry, buffers, len(buffers), count)
    assert list(destination) == expected and len(calls) == 1
    if dtype == "bool_":
        trace = json.loads(host.trace.read_text())
        assert trace["physicalBooleanType"] == physical
        assert trace["booleanGuardValues"] == guard


@pytest.mark.parametrize(
    "fault", ["input-byte", "input-type", "output-byte", "output-type", "guard"]
)
def test_boolean_host_rejects_invalid_storage(
    translated_packages, tmp_path, monkeypatch, fault
):
    host = runtime.HostRuntime(translated_packages, tmp_path / "trace")
    entry = "vv_LogicalAndbool_"
    memory = [
        (ctypes.c_uint8 * 3)(0, 1, 0),
        (ctypes.c_uint8 * 3)(1, 0, 1),
        (ctypes.c_uint8 * 3)(7, 7, 7),
        ctypes.c_uint32(3),
    ]
    buffers = (runtime.Buffer * 4)(
        *[
            runtime.Buffer(
                name.encode(),
                b"uint32" if name == "size" else b"bool_",
                ctypes.addressof(value),
                1 if name == "size" else 3,
                int(name == "c"),
            )
            for name, value in zip(("a", "b", "c", "size"), memory)
        ]
    )
    if fault == "input-byte":
        memory[0][1] = 2
    if fault == "input-type":
        buffers[0].dtype = b"uint32"
    calls = []

    def execute(request):
        calls.append(request)
        value = request.fixture.expected_outputs[0]
        values = [False, True, False] + runtime.BOOLEAN_GUARD
        if host.target != "metal":
            values = [int(item) for item in values]
        if fault == "output-byte":
            values[1] = 2
        if fault == "output-type":
            values[1] = 1 if host.target == "metal" else True
        if fault == "guard":
            values[-1] = not values[-1] if host.target == "metal" else 1 - values[-1]
        return SimpleNamespace(
            status="ok",
            outputs={
                value.name: {"dtype": value.dtype, "shape": [35], "values": values}
            },
            details={},
        )

    monkeypatch.setattr(host.executor, "run", execute)
    with pytest.raises((ValueError, RuntimeError)):
        host.dispatch(entry, buffers, 4, 3)
    assert list(memory[2]) == [7, 7, 7] and not host.trace.exists()
    assert len(calls) == (0 if fault.startswith("input") else 1)


@pytest.mark.parametrize(
    "fault", [None, "missing", "dtype", "shape", "input", "output", "byte-type"]
)
def test_boolean_workload_records_are_exact(fault):
    records = boolean_workloads.expected_records()
    assert len(records) > 250 and len(boolean_workloads.dispatches()) > 400
    if fault == "missing":
        records.pop()
    elif fault == "dtype":
        records[1]["dtype"] = "uint8"
    elif fault == "shape":
        records[1]["shape"] = [1]
    elif fault in {"input", "output", "byte-type"}:
        key = "aBytes" if fault == "input" else "bytes"
        records[1][key][0] = True if fault == "byte-type" else records[1][key][0] ^ 1
    if fault:
        with pytest.raises(RuntimeError, match="Boolean readbacks"):
            boolean_workloads.validate(records)
    else:
        boolean_workloads.validate(records)


@pytest.mark.parametrize(
    "entry",
    [
        entry
        for entry, dtype in packages.COMPARISON_ENTRIES.items()
        if dtype == "float32"
    ]
    + ["v_copyfloat32bool_"],
)
@pytest.mark.parametrize("subnormal", [False, True])
def test_unproven_subnormal_comparisons_fail_before_dispatch(
    translated_packages, tmp_path, monkeypatch, entry, subnormal
):
    host = runtime.HostRuntime(translated_packages, tmp_path / "trace")
    words = (
        [1, 0x80000001, 0x007FFFFF, 0x807FFFFF]
        if subnormal
        else [0, 0x80000000, 0x00800000, 0x80800000]
    )
    source = (ctypes.c_uint32 * 4)(*words)
    other = (ctypes.c_float * 4)(0, 0, 0, 0)
    destination = (ctypes.c_uint8 * 4)(7, 7, 7, 7)
    size = ctypes.c_uint32(4)
    cast = entry in packages.BOOLEAN_CAST_ENTRIES
    arguments = (
        [("src", "float32", source), ("dst", "bool_", destination)]
        if cast
        else [
            ("a", "float32", source),
            ("b", "float32", other),
            ("c", "bool_", destination),
        ]
    ) + [("size", "uint32", size)]
    buffers = (runtime.Buffer * len(arguments))(
        *[
            runtime.Buffer(
                name.encode(),
                dtype.encode(),
                ctypes.addressof(value),
                1 if name == "size" else 4,
                int(value is destination),
            )
            for name, dtype, value in arguments
        ]
    )
    calls = []

    def execute(request):
        calls.append(request)
        raise RuntimeError("executor reached")

    monkeypatch.setattr(host.executor, "run", execute)
    rejected = subnormal and not cast and host.target != "metal"
    with pytest.raises(
        ValueError if rejected else RuntimeError,
        match="Subnormal float comparison parity" if rejected else "executor reached",
    ):
        host.dispatch(entry, buffers, len(buffers), 4)
    assert len(calls) == int(not rejected)
    assert list(destination) == [7] * 4
    assert list(source) == words
    assert not host.trace.exists()


def test_boolean_unary_dispatch_preserves_stored_contiguous_layouts(monkeypatch):
    monkeypatch.setattr(
        boolean_workloads,
        "definitions",
        lambda: [
            (packages.LOGICAL_NOT_ENTRY, "not", "bool_", "bool_", layout)
            for layout in ("transpose", "broadcast", "reverse")
        ],
    )
    assert boolean_workloads.dispatches() == [
        (packages.LOGICAL_NOT_ENTRY, 15),
        (packages.LOGICAL_NOT_ENTRY, 5),
        (packages.BOOLEAN_COPY_ENTRY, 17),
        (packages.LOGICAL_NOT_ENTRY, 17),
    ]


def test_ci_requires_native_math_before_building_mlx():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/mlx-portable-host.yml").read_text()
    order = (
        "Validate native scalar math",
        "Checkout and prepare pinned upstream MLX",
        "Validate pinned native binary math",
        "Build adapted upstream MLX",
    )
    for earlier, later in zip(order, order[1:]):
        assert ci_coverage.workflow_job_step_after(
            workflow, "portable-host", later, earlier
        )
    for name, seconds, evidence_directory, modules, flags in (
        (
            "Validate native scalar math",
            120,
            "scalar-math",
            ("test_directx_atan2.py", "test_metal_precise_asin.py"),
            ("CROSTL_REQUIRE_METAL_PRECISE_ASIN",),
        ),
        (
            "Validate pinned native binary math",
            900,
            "binary-math",
            (
                "test_struct_buffer_layouts.py",
                "test_buffer_requirements.py",
                "test_native_dispatch_limits.py",
                "test_native_loader_dispatch_integration.py",
                "test_mlx_current_complex_power.py",
                "test_mlx_current_binary_shapes.py",
            ),
            (
                "CROSTL_REQUIRE_MLX_CURRENT_COMPLEX_POWER",
                "CROSTL_REQUIRE_MLX_CURRENT_BINARY_SHAPES",
                "CROSTL_REQUIRE_STRUCT_BUFFER_RUNTIME",
            ),
        ),
    ):
        step = ci_coverage.workflow_job_step_section(workflow, "portable-host", name)
        assert "if:" not in step and "continue-on-error" not in step
        assert f"--timeout-seconds {seconds}" in step
        assert "pytest -q -n auto" in step
        assert "set -euo pipefail" in step
        assert "--junitxml=.mlx-portable-host/" in step
        assert "--basetemp=.mlx-portable-host/" in step
        assert step.index(f"mkdir -p .mlx-portable-host/{evidence_directory}") < (
            step.index("python tools/run_bounded_command.py")
        )
        for module in modules:
            path = f"tests/test_translator/{module}"
            assert path in step
            for event in ("push", "pull_request"):
                assert path in ci_coverage.workflow_event_path_filters(workflow, event)
        for flag in flags:
            assert f'{flag}: "1"' in step
    scalar = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate native scalar math"
    )
    assert (
        "CROSTL_REQUIRE_DIRECTX_ATAN2: ${{ runner.os == 'Windows' && '1' || '0' }}"
        in scalar
    )
    half = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate OpenGL half conversions"
    )
    assert "if: runner.os == 'Linux'" in half
    assert "continue-on-error" not in half
    assert 'CROSTL_REQUIRE_HALF_CONVERSION_RUNTIME: "1"' in half
    assert "CROSTL_HALF_CONVERSION_TARGET: opengl" in half
    assert "--timeout-seconds 180" in half
    assert "pytest -q -n auto" in half
    assert "--basetemp=.mlx-portable-host/half-conversions/pytest" in half
    assert "--junitxml=.mlx-portable-host/half-conversions/results.xml" in half
    assert "tee .mlx-portable-host/half-conversions.log" in half
    for event in ("pull_request", "push"):
        assert "tests/test_translator/test_opengl_half_conversion.py" in (
            ci_coverage.workflow_event_path_filters(workflow, event)
        )
    directx_half = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate DirectX half conversions"
    )
    for name, directory, flag, module, seconds in (
        (
            "Validate OpenGL uniform blocks",
            "uniform-blocks",
            "CROSTL_REQUIRE_UNIFORM_BLOCK_RUNTIME",
            "test_uniform_block_layouts.py",
            180,
        ),
        (
            "Execute pinned OpenGL attention derivatives",
            "attention-ds",
            "CROSTL_REQUIRE_MLX_ATTENTION_DS_OPENGL",
            "test_mlx_attention_ds_opengl.py",
            600,
        ),
    ):
        step = ci_coverage.workflow_job_step_section(workflow, "portable-host", name)
        assert "if: runner.os == 'Linux'" in step
        assert "continue-on-error" not in step
        assert f'{flag}: "1"' in step
        assert "EGL_PLATFORM: surfaceless" in step
        assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in step
        assert "set -euo pipefail" in step
        assert "pytest -q -n auto" in step
        assert f"--timeout-seconds {seconds}" in step
        assert f"--basetemp=.mlx-portable-host/{directory}/pytest" in step
        assert f"--junitxml=.mlx-portable-host/{directory}/results.xml" in step
        assert f"tee .mlx-portable-host/{directory}.log" in step
        assert f"tests/test_translator/{module}" in step
        for event in ("pull_request", "push"):
            assert (
                f"tests/test_translator/{module}"
                in ci_coverage.workflow_event_path_filters(workflow, event)
            )
    derivatives = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Execute pinned OpenGL attention derivatives"
    )
    assert (
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-upstream" in derivatives
    )
    assert "if: runner.os == 'Windows'" in directx_half
    assert "continue-on-error" not in directx_half
    assert 'CROSTL_REQUIRE_HALF_CONVERSION_RUNTIME: "1"' in directx_half
    assert "CROSTL_HALF_CONVERSION_TARGET: directx" in directx_half
    assert "--timeout-seconds 180" in directx_half
    assert "pytest -q -n auto" in directx_half
    assert "--basetemp=.mlx-portable-host/half-conversions/pytest" in directx_half
    assert "--junitxml=.mlx-portable-host/half-conversions/results.xml" in directx_half
    assert "tee .mlx-portable-host/half-conversions.log" in directx_half
    assert "tests/test_translator/test_directx_half_conversion.py" in directx_half
    for event in ("pull_request", "push"):
        assert "tests/test_translator/test_directx_half_conversion.py" in (
            ci_coverage.workflow_event_path_filters(workflow, event)
        )
    binary = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned native binary math"
    )
    assert "CROSTL_MLX_CURRENT_TARGET: ${{ matrix.target }}" in binary
    assert (
        "CROSTL_RUN_NATIVE_LOADER_DIRECTX_DEVICE_TEST: "
        "${{ runner.os == 'Windows' && '1' || '0' }}" in binary
    )
    assert (
        "CROSTL_RUN_NATIVE_LOADER_OPENGL_DEVICE_TEST: "
        "${{ runner.os == 'Linux' && '1' || '0' }}" in binary
    )
    assert "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-upstream" in binary
    timeout = ci_coverage.workflow_job_timeout_minutes(workflow, "portable-host")
    assert timeout * 60 > 120 + 900 + 1800 + 900 + 1800


def test_ci_requires_resident_attention_reductions():
    import re

    from tools import ci_coverage

    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/mlx-portable-host.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Execute pinned resident attention reductions"
    )
    assert "if:" not in step and "continue-on-error" not in step
    assert 'CROSTL_REQUIRE_MLX_ATTENTION_REDUCE_RUNTIME: "1"' in step
    assert "CROSTL_MLX_ATTENTION_REDUCE_TARGET: ${{ matrix.target }}" in step
    assert "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-upstream" in step
    assert "CROSTL_REQUIRE_MLX_ATTENTION_REDUCE_COMPILE" not in step
    assert "EGL_PLATFORM: surfaceless" in step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in step
    assert "set -euo pipefail" in step
    assert "--timeout-seconds 300" in step
    assert "pytest -q -n auto --dist loadscope" in step
    assert "--basetemp=.mlx-portable-host/attention-reduce/pytest" in step
    assert "--junitxml=.mlx-portable-host/attention-reduce/results.xml" in step
    assert "tee .mlx-portable-host/attention-reduce.log" in step
    assert "tests/test_translator/test_mlx_attention_reduce_runtime.py" in step
    assert ci_coverage.workflow_job_step_after(
        workflow,
        "portable-host",
        "Execute pinned resident attention reductions",
        "Checkout and prepare pinned upstream MLX",
    )
    for event in ("push", "pull_request"):
        assert (
            "tests/test_translator/test_mlx_attention_reduce_runtime.py"
            in ci_coverage.workflow_event_path_filters(workflow, event)
        )
    assert ci_coverage.workflow_job_timeout_minutes(
        workflow, "portable-host"
    ) * 60 > sum(
        int(value)
        for value in re.findall(
            r"--timeout-seconds (\d+)",
            ci_coverage.workflow_job_text(workflow, "portable-host"),
        )
    )


def test_ci_requires_directx_atomic_execution_and_pinned_compilation():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/mlx-portable-host.yml").read_text()
    for name, directory, seconds, flag, module in (
        (
            "Validate DirectX software reductions",
            "software-reductions",
            180,
            "CROSTL_REQUIRE_DIRECTX_SOFTWARE_REDUCTIONS",
            "test_directx_software_reductions.py",
        ),
        (
            "Validate DirectX float atomics",
            "float-atomics",
            180,
            "CROSTL_REQUIRE_DIRECTX_FLOAT_ATOMICS",
            "test_directx_float_atomics.py",
        ),
        (
            "Compile pinned DirectX gated-delta backward entry",
            "gated-delta",
            300,
            "CROSTL_REQUIRE_MLX_GATED_DELTA_DIRECTX",
            "test_mlx_gated_delta_directx.py",
        ),
        (
            "Execute pinned DirectX gated-delta gradients",
            "gated-delta-runtime",
            900,
            "CROSTL_REQUIRE_MLX_GATED_DELTA_DIRECTX_RUNTIME",
            "test_mlx_gated_delta_directx_runtime.py",
        ),
        (
            "Execute pinned DirectX attention derivatives",
            "attention-ds",
            600,
            "CROSTL_REQUIRE_MLX_ATTENTION_DS_RUNTIME",
            "test_mlx_attention_ds_runtime.py",
        ),
    ):
        step = ci_coverage.workflow_job_step_section(workflow, "portable-host", name)
        assert "if: runner.os == 'Windows'" in step
        assert "continue-on-error" not in step
        assert f'{flag}: "1"' in step
        assert "pytest -q -n auto" in step
        assert "set -euo pipefail" in step
        assert f"--timeout-seconds {seconds}" in step
        assert f"--basetemp=.mlx-portable-host/{directory}/pytest" in step
        assert f"--junitxml=.mlx-portable-host/{directory}/results.xml" in step
        assert f"tee .mlx-portable-host/{directory}.log" in step
        assert f"tests/test_translator/{module}" in step
        assert ci_coverage.workflow_job_step_after(
            workflow, "portable-host", "Build adapted upstream MLX", name
        )
        for event in ("push", "pull_request"):
            paths = ci_coverage.workflow_event_path_filters(workflow, event)
            assert f"tests/test_translator/{module}" in paths
            assert "tests/test_translator/test_metal_float_atomics.py" in paths
            assert "tests/test_translator/test_mlx_gated_delta_metal.py" in paths
            assert "tests/test_translator/test_mlx_gated_delta_runtime.py" in paths
            assert (
                "tests/fixtures/runtime_verification/mlx_gated_delta_reference.py"
                in paths
            )
    metal = (root / ".github/workflows/mlx-metal-host.yml").read_text()
    attention = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Execute pinned portable attention row dots"
    )
    assert "if:" not in attention and "continue-on-error" not in attention
    assert 'CROSTL_REQUIRE_MLX_ATTENTION_RUNTIME: "1"' in attention
    assert "CROSTL_MLX_ATTENTION_TARGET: ${{ matrix.target }}" in attention
    assert "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-upstream" in attention
    assert "EGL_PLATFORM: surfaceless" in attention
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in attention
    assert "set -euo pipefail" in attention
    assert "pytest -q -n auto --dist loadscope" in attention
    assert "--timeout-seconds 900" in attention
    assert "--basetemp=.mlx-portable-host/attention-odo/pytest" in attention
    assert "--junitxml=.mlx-portable-host/attention-odo/results.xml" in attention
    assert "tests/test_translator/test_mlx_attention_odo_runtime.py" in attention
    assert "tee .mlx-portable-host/attention-odo.log" in attention
    for event in ("push", "pull_request"):
        assert "tests/test_translator/test_mlx_attention_odo_runtime.py" in (
            ci_coverage.workflow_event_path_filters(workflow, event)
        )
    assert ci_coverage.workflow_job_timeout_minutes(workflow, "portable-host") * 60 > (
        120
        + 120
        + 180
        + 180
        + 120
        + 300
        + 900
        + 900
        + 300
        + 900
        + 1800
        + 300
        + 1000
        + 600
        + 600
    )
    reductions = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate DirectX software reductions"
    )
    assert "tests/test_translator/test_directx_subgroup_identity.py" in reductions
    assert "tests/test_translator/test_directx_subgroup_uniformity.py" in reductions
    assert "tests/test_translator/test_directx_subgroup_components.py" in reductions
    for event in ("push", "pull_request"):
        assert "tests/test_translator/test_directx_subgroup_identity.py" in (
            ci_coverage.workflow_event_path_filters(workflow, event)
        )
        assert "tests/test_translator/test_directx_subgroup_uniformity.py" in (
            ci_coverage.workflow_event_path_filters(workflow, event)
        )
        assert "tests/test_translator/test_directx_subgroup_components.py" in (
            ci_coverage.workflow_event_path_filters(workflow, event)
        )
    control = ci_coverage.workflow_job_step_section(
        metal, "metal-host", "Validate native float atomics"
    )
    assert (
        "test_directx_float_atomics.py::test_original_metal_matches_atomic_oracle"
        in control
    )
    assert 'CROSTL_REQUIRE_METAL_FLOAT_ATOMICS: "1"' in control
    for event in ("push", "pull_request"):
        assert (
            "tests/test_translator/test_directx_float_atomics.py"
            in ci_coverage.workflow_event_path_filters(metal, event)
        )
