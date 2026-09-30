import ctypes
import json
import sys
import threading
import uuid
from types import SimpleNamespace

import pytest

from tools import run_directx_diagnostics as diagnostics


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("failure", [None, "missing", "truncated", "unreadable"])
def test_loaded_runtime_library_identity(tmp_path, monkeypatch, failure):
    module = tmp_path / "d3d10warp.dll"
    if failure != "unreadable":
        module.write_bytes(b"runtime binary")

    def get_handle(name):
        return 0x100000000 if name == "d3d10warp.dll" else None

    def get_filename(handle, buffer, capacity):
        assert handle == 0x100000000
        if failure == "missing":
            return 0
        if failure == "truncated":
            return capacity
        buffer.value = str(module)
        return len(buffer.value)

    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda name, **kwargs: SimpleNamespace(
            GetModuleHandleW=get_handle, GetModuleFileNameW=get_filename
        ),
        raising=False,
    )
    records = diagnostics._loaded_runtime_libraries()
    assert records[:2] == [
        {"name": "d3d12.dll", "loaded": False},
        {"name": "D3D12Core.dll", "loaded": False},
    ]
    warp = records[2]
    assert warp["loaded"] is True
    assert get_handle.restype is ctypes.c_void_p
    if failure:
        assert "error" in warp
        assert "sha256" not in warp
    else:
        assert warp["path"] == str(module)
        assert (
            warp["sha256"] == diagnostics.hashlib.sha256(b"runtime binary").hexdigest()
        )


def test_debug_layer_uses_native_interface_and_releases_it(monkeypatch):
    calls = []
    release = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
        lambda _self: calls.append("release") or 0
    )
    enable = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
        lambda _self: calls.append("enable")
    )
    vtable = (ctypes.c_void_p * 4)()
    vtable[2] = ctypes.cast(release, ctypes.c_void_p).value
    vtable[3] = ctypes.cast(enable, ctypes.c_void_p).value
    interface = ctypes.c_void_p(ctypes.addressof(vtable))

    def get_interface(identifier, output):
        assert (
            ctypes.string_at(identifier, 16)
            == uuid.UUID("344488b7-6846-474b-b989-f027448245e0").bytes_le
        )
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p)).contents.value = (
            ctypes.addressof(interface)
        )
        return 0

    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE, raising=False)
    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda _name: SimpleNamespace(D3D12GetDebugInterface=get_interface),
        raising=False,
    )
    diagnostics._enable_debug_layer()
    assert calls == ["enable", "release"]


def test_missing_debug_interface_is_checked_before_using_pointer(monkeypatch):
    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda _name: SimpleNamespace(
            D3D12GetDebugInterface=lambda *_args: -2147467262
        ),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="0x80004002"):
        diagnostics._enable_debug_layer()


@pytest.mark.parametrize("library_error", [False, True])
def test_collector_flushes_live_and_final_device_messages(
    tmp_path, monkeypatch, library_error
):
    polled = threading.Event()
    pending = ["first diagnostic"]
    initialized = []

    def messages():
        assert initialized == [True]
        current = pending[:]
        pending.clear()
        polled.set()
        return current

    device = SimpleNamespace(
        name="Test device", is_hardware=False, get_debug_messages=messages
    )
    module = SimpleNamespace(
        get_backend=lambda: SimpleNamespace(__name__="compushady.backends.d3d12"),
        get_current_device=lambda: device,
        HEAP_UPLOAD=1,
        Buffer=lambda size, heap, **kwargs: initialized.append(
            size == 4 and heap == 1 and kwargs == {"device": device}
        ),
    )
    monkeypatch.setattr(diagnostics, "_enable_debug_layer", lambda: None)
    monkeypatch.setattr(diagnostics.importlib, "import_module", lambda _name: module)

    def libraries():
        assert initialized == [True]
        if library_error:
            raise OSError("Module query failed")
        return [{"name": "d3d10warp.dll", "loaded": True, "sha256": "test-digest"}]

    monkeypatch.setattr(diagnostics, "_loaded_runtime_libraries", libraries)
    path = tmp_path / "nested" / "diagnostics.jsonl"
    with diagnostics.collect_diagnostics(path, interval=1000):
        assert polled.wait(timeout=5)
        pending.append("final diagnostic")
    records = _records(path)
    assert [
        record["message"] for record in records if record["event"] == "message"
    ] == ["first diagnostic", "final diagnostic"]
    assert records[-1]["event"] == "finished"
    assert records[-1]["collectorStopped"] is True
    if library_error:
        assert any(r["event"] == "library-inspection-unavailable" for r in records)
    else:
        identity = next(r for r in records if r["event"] == "runtime-libraries")
        assert identity["libraries"][0]["sha256"] == "test-digest"


def test_unavailable_diagnostics_preserve_module_arguments_and_exit(
    tmp_path, monkeypatch
):
    def unavailable():
        raise RuntimeError("debug layer missing")

    def run_module(module, **options):
        assert module == "pytest"
        assert options == {"run_name": "__main__", "alter_sys": True}
        assert sys.argv == ["pytest", "-vv", "test_case.py"]
        assert sys.path[0] == str(diagnostics.Path.cwd())
        raise SystemExit(7)

    monkeypatch.setattr(diagnostics, "_enable_debug_layer", unavailable)
    monkeypatch.setattr(diagnostics.runpy, "run_module", run_module)
    previous_argv = sys.argv
    previous_path = sys.path[:]
    path = tmp_path / "diagnostics.jsonl"
    with pytest.raises(SystemExit) as raised:
        diagnostics.main(
            ["--output", str(path), "--module", "pytest", "--", "-vv", "test_case.py"]
        )
    assert raised.value.code == 7
    assert sys.argv is previous_argv
    assert sys.path == previous_path
    assert [record["event"] for record in _records(path)] == [
        "starting",
        "unavailable",
        "finished",
    ]


def test_collector_failure_does_not_hide_module_exception(tmp_path, monkeypatch):
    def messages():
        raise RuntimeError("device query failed")

    device = SimpleNamespace(
        name="Test device", is_hardware=False, get_debug_messages=messages
    )
    module = SimpleNamespace(
        get_backend=lambda: SimpleNamespace(__name__="compushady.backends.d3d12"),
        get_current_device=lambda: device,
        HEAP_UPLOAD=1,
        Buffer=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(diagnostics, "_enable_debug_layer", lambda: None)
    monkeypatch.setattr(diagnostics.importlib, "import_module", lambda _name: module)
    path = tmp_path / "diagnostics.jsonl"
    with pytest.raises(ValueError, match="original failure"):
        with diagnostics.collect_diagnostics(path):
            raise ValueError("original failure")
    assert any(record["event"] == "collector-error" for record in _records(path))


@pytest.mark.parametrize("failure", ["thread-start", "backend"])
def test_setup_failure_keeps_module_running(tmp_path, monkeypatch, failure):
    backend = "vulkan" if failure == "backend" else "d3d12"
    module = SimpleNamespace(
        get_backend=lambda: SimpleNamespace(__name__=f"compushady.backends.{backend}"),
        get_current_device=lambda: SimpleNamespace(name="Test", is_hardware=False),
        HEAP_UPLOAD=1,
        Buffer=lambda *args, **kwargs: None,
    )

    def start():
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(diagnostics, "_enable_debug_layer", lambda: None)
    monkeypatch.setattr(diagnostics.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(
        diagnostics,
        "threading",
        SimpleNamespace(
            Event=threading.Event,
            Lock=threading.Lock,
            Thread=lambda **kwargs: SimpleNamespace(start=start),
        ),
    )
    path = tmp_path / "diagnostics.jsonl"
    with diagnostics.collect_diagnostics(path):
        assert any(record["event"] == "unavailable" for record in _records(path))
    assert _records(path)[-1]["collectorStopped"] is True
