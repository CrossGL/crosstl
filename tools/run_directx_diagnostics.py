#!/usr/bin/env python3
"""Run a Python module while retaining live Direct3D 12 debug messages."""

import argparse
import ctypes
import importlib
import json
import runpy
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def _enable_debug_layer():
    if sys.platform != "win32":
        raise RuntimeError("Direct3D 12 diagnostics require Windows")
    library = ctypes.WinDLL("d3d12.dll")
    get_interface = library.D3D12GetDebugInterface
    get_interface.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    get_interface.restype = ctypes.c_int32
    identifier = ctypes.create_string_buffer(
        uuid.UUID("344488b7-6846-474b-b989-f027448245e0").bytes_le
    )
    interface = ctypes.c_void_p()
    result = get_interface(identifier, ctypes.byref(interface))
    if result < 0 or not interface.value:
        raise RuntimeError(
            f"D3D12GetDebugInterface failed: 0x{result & 0xFFFFFFFF:08x}"
        )
    table = ctypes.cast(
        interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    ).contents
    release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(table[2])
    enable = ctypes.WINFUNCTYPE(None, ctypes.c_void_p)(table[3])
    try:
        enable(interface)
    finally:
        release(interface)


@contextmanager
def collect_diagnostics(output, *, interval=2.0):
    """Keep evidence outside the module's temporary files without altering its exit."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stopped = threading.Event()
    lock = threading.Lock()
    thread = None

    with output.open("w", encoding="utf-8", buffering=1) as stream:

        def record(event, **details):
            payload = {
                "schemaVersion": 1,
                "event": event,
                "elapsedSeconds": round(time.monotonic() - started, 3),
                **details,
            }
            line = json.dumps(payload, sort_keys=True)
            with lock:
                if not stream.closed:
                    stream.write(line + "\n")
                    stream.flush()
                print("[directx-diagnostics] " + line, flush=True)

        def drain(devices):
            for device in devices:
                try:
                    for message in device.get_debug_messages():
                        record("message", device=device.name, message=message)
                except Exception as error:
                    record("collector-error", error=str(error))
                    return False
            return True

        def poll(devices):
            while drain(devices):
                if stopped.wait(interval):
                    return

        record("starting")
        try:
            _enable_debug_layer()
            compushady = importlib.import_module("compushady")
            if compushady.get_backend().__name__.rsplit(".", 1)[-1] != "d3d12":
                raise RuntimeError("The selected compushady backend is not Direct3D 12")
            device = compushady.get_current_device()
            # Discovery is lazy; create a resource before the debug-queue query
            # so the selected native device exists.
            compushady.Buffer(4, compushady.HEAP_UPLOAD, device=device)
            devices = [device]
            record(
                "enabled",
                devices=[
                    {"name": device.name, "isHardware": device.is_hardware}
                    for device in devices
                ],
            )
            worker = threading.Thread(target=poll, args=(devices,), daemon=True)
            worker.start()
            thread = worker
        except Exception as error:
            # Missing optional debug tooling must not suppress the actual test.
            record("unavailable", error=str(error))
        try:
            yield
        finally:
            stopped.set()
            if thread is not None:
                thread.join(timeout=5.0)
                if not thread.is_alive():
                    drain(devices)
            record("finished", collectorStopped=thread is None or not thread.is_alive())


def run_module(module, arguments, output):
    previous_argv = sys.argv
    previous_path = sys.path[:]
    try:
        sys.argv = [module, *arguments]
        # Match python -m's import path rather than the wrapper's tools directory.
        sys.path.insert(0, str(Path.cwd()))
        with collect_diagnostics(output):
            runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    run_module(args.module, arguments, args.output)


if __name__ == "__main__":
    main()
