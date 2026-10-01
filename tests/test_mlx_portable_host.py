import ctypes
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    packages,
    prepare,
    runtime,
    unary_workloads,
    verify,
)


def checkout(root, monkeypatch, newline=b"\n"):
    backend = root / "mlx/backend/no_gpu"
    backend.mkdir(parents=True)
    originals = {
        "CMakeLists.txt": b"target_sources(mlx PRIVATE primitives.cpp)\n",
        "primitives.cpp": (
            "\n".join(
                f"NO_GPU({name})"
                for name in ("Arange", "Add", *packages.UNARY_OPERATIONS)
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
    assert "NO_GPU(Add)" in (backend / "crosstl_primitives.cpp").read_text()
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


@pytest.fixture(scope="module", params=["opengl", "directx"])
def package_templates(tmp_path_factory, request):
    tmp_path = tmp_path_factory.mktemp(request.param)
    root = tmp_path / "mlx"
    source = root / packages.SOURCE
    source.parent.mkdir(parents=True)
    types = {
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
    ],
)
def test_unary_verifier_requires_complete_independent_results(fault):
    original = unary_workloads.expected_records(cpu=True)
    translated = unary_workloads.expected_records()
    assert len(translated) == 129
    assert sum(record["count"] for record in translated) == 8009
    assert len(unary_workloads.dispatches()) == 102
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


@pytest.mark.parametrize("dtype", list(runtime.TYPES))
def test_translated_binding_contract_and_typed_readback(
    translated_packages, tmp_path, monkeypatch, dtype
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
    host.dispatch("arange" + dtype, buffers, 3, 3)
    assert list(memory[-1]) == [2, 5, 8] and len(calls) == 1
    trace = json.loads(host.trace.read_text())
    assert trace["target"] == host.target and trace["entry"] == "arange" + dtype


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

    def fail(*args):
        raise ValueError("long error message")

    monkeypatch.setattr(host, "dispatch", fail)
    error = ctypes.create_string_buffer(8)
    assert host._dispatch(b"entry", None, 0, 0, ctypes.addressof(error), 8) == 1
    assert error.raw == b"long er\0"
    assert host._dispatch(b"entry", None, 0, 0, None, 0) == 1


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_registration_retains_callback_and_uses_platform_library(
    tmp_path, monkeypatch, platform
):
    host = runtime.HostRuntime.__new__(runtime.HostRuntime)
    host.callback = runtime.CALLBACK(lambda *args: 0)
    module = SimpleNamespace(__file__=str(tmp_path / "core.pyd"), gpu="gpu")
    selected, loaded, registered = [], [], []
    module.set_default_device = selected.append
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=module))
    monkeypatch.setitem(sys.modules, "mlx.core", module)
    monkeypatch.setattr(runtime.sys, "platform", platform)
    monkeypatch.setattr(runtime, "_installed_runtime", None)

    def register(version, callback):
        registered.append((version, callback))
        return 0

    def load(path):
        loaded.append(path)
        return SimpleNamespace(crosstl_mlx_register_dispatch=register)

    monkeypatch.setattr(runtime.ctypes, "CDLL", load)
    host.install()
    assert runtime._installed_runtime is host
    assert registered == [(1, host.callback)] and selected == ["gpu"]
    assert loaded == [
        str(tmp_path / ("mlx.dll" if platform == "win32" else "core.pyd"))
    ]
    with pytest.raises(RuntimeError, match="already installed"):
        host.install()


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
        "source-before",
        "source-after",
        "unary-missing",
        "unary-values",
        "unary-identity",
        "upstream-failure",
    ],
)
def test_verifier_keeps_selected_scope_and_rejects_incomplete_evidence(
    tmp_path, monkeypatch, fault
):
    source = tmp_path / "mlx/python/tests/test_ops.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"unchanged tests")
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
        assert "--timeout-seconds" in command and "180" in command
        mode = command[command.index("--worker") + 1]
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
        }
        if fault == "unary-missing":
            result["unary"] = []
        if fault == "unary-values":
            result["unary"][1]["values"] = [123.0]
        if fault == "unary-identity":
            result["unary"][1]["inputs"] = [123.0]
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
            dispatches = [
                (entry, count)
                for entry in packages.ARANGE_ENTRIES
                for count in (1, 7, 257)
            ] + unary_workloads.dispatches()
            trace = [
                {
                    "entry": entry,
                    "target": "directx" if fault == "target" else "opengl",
                    "threads": count,
                }
                for entry, count in dispatches
                if fault != "trace" or entry != packages.UNARY_ENTRIES[-1]
            ]
            if fault == "threads":
                trace[0]["threads"] = 0
            if fault == "duplicate":
                trace.insert(1, trace[0])
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
        if fault == "source-before":
            assert calls == []
        if fault == "command":
            assert len(calls) == 8
            assert (
                json.loads((args.output_dir / "cpu.command.json").read_text())[
                    "returncode"
                ]
                == 124
            )
    else:
        evidence = verify.verify(args)
        assert len(calls) == 8 and evidence["dispatchCount"] == 117
        assert len(identities) == 2
        assert evidence["schemaVersion"] == 2
        assert evidence["adaptation"]["files"] == {"adapter": "unchanged"}
        assert evidence["fullUpstreamSuite"] is False
        assert evidence["fullTranslatedBackend"] is False
        assert evidence["original"] == evidence["translated"]


def test_ci_requires_both_native_platforms_and_retains_evidence():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/mlx-portable-host.yml").read_text()
    for required in (
        prepare.COMMIT,
        "ubuntu-24.04",
        "windows-2025",
        'python-version: "3.12"',
        "Initialize execution evidence",
        "tee .mlx-portable-host/dependencies.log",
        "python -m pip install moderngl PyOpenGL",
        'pip install -e ".[directx-runtime]"',
        "target: opengl",
        "target: directx",
        "-DMLX_BUILD_METAL=OFF",
        "-DMLX_BUILD_CUDA=OFF",
        "-DMLX_CROSTL_HOST=ON",
        "portable_host.prepare",
        "portable_host.packages",
        "portable_host.verify",
        "pytest -q -n auto tests/test_mlx_portable_host.py",
        "liblapacke-dev",
        "--timeout-seconds 1600",
        "if: always()",
        "include-hidden-files: true",
        "Get-FileHash",
        "core.autocrlf false",
    ):
        assert required in workflow
    assert "continue-on-error" not in workflow
    assert "opengl-runtime" not in workflow
    assert len(verify.UPSTREAM_TESTS) == 15


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
    assert timeout * 60 > 120 + 900 + 1800 + 900 + 1600


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
        int(value) for value in re.findall(r"--timeout-seconds (\d+)", workflow)
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
