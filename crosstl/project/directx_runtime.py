"""Process-isolated Direct3D execution with allocation-backed buffer views."""

from __future__ import annotations

import atexit
import hashlib
import io
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from .runtime_verification import RuntimeAdapterDispatchError, RuntimeAdapterSetupError

_BUILD_LOCK = threading.Lock()
_WORKERS = {}
_MAX_BUFFER_BYTES = 256 * 1024 * 1024


def _error(message, reason, **details):
    return RuntimeAdapterSetupError(
        message, details={"target": "directx", "reasonKind": reason, **details}
    )


def _run(command, *, timeout, env=None):
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            if sys.platform == "win32":
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True,
                        check=False,
                        timeout=15,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if process.poll() is None:
                process.kill()
            stdout, stderr = process.communicate()
            raise _error(
                "DirectX native operation exceeded its deadline.",
                "operation-timeout",
                timeoutSeconds=timeout,
                command=command if isinstance(command, str) else list(command),
            ) from exc
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _compiler_environment():
    environment = dict(os.environ)
    compiler = shutil.which("cl.exe")
    if compiler:
        return compiler, environment
    installer = Path(environment.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
    vswhere = installer / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.is_file():
        raise _error(
            "DirectX buffer views require MSVC and the Windows SDK.",
            "compiler-unavailable",
        )
    result = _run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        timeout=30,
    )
    root = result.stdout.decode("utf-8", errors="replace").strip()
    setup = Path(root) / "VC/Auxiliary/Build/vcvars64.bat"
    if result.returncode or not root or not setup.is_file():
        raise _error(
            "No MSVC x64 development environment was found.", "compiler-unavailable"
        )
    # cmd.exe parses quotes differently from the C runtime used by list2cmdline.
    result = _run(f'cmd.exe /d /s /c ""{setup}" >nul && set"', timeout=60)
    if result.returncode:
        raise _error("MSVC environment setup failed.", "compiler-environment-failed")
    for line in result.stdout.decode("mbcs").splitlines():
        name, separator, value = line.partition("=")
        if separator and name:
            environment[name.upper()] = value
    compiler = shutil.which("cl.exe", path=environment.get("PATH", ""))
    if not compiler:
        raise _error("MSVC setup did not expose cl.exe.", "compiler-unavailable")
    return compiler, environment


def _worker():
    if sys.platform != "win32":
        raise _error("Direct3D buffer views require Windows.", "platform-unavailable")
    source = Path(__file__).with_name("directx_runtime_worker.cpp")
    compiler, environment = _compiler_environment()
    warp = Path(sys.executable).with_name("d3d10warp.dll")
    key = (
        hashlib.sha256(source.read_bytes()).hexdigest(),
        compiler,
        Path(compiler).stat().st_mtime_ns,
        environment.get("INCLUDE"),
        environment.get("LIB"),
        hashlib.sha256(warp.read_bytes()).hexdigest() if warp.is_file() else None,
    )
    with _BUILD_LOCK:
        if key in _WORKERS:
            return _WORKERS[key][1]
        directory = tempfile.TemporaryDirectory(prefix="crosstl-directx-worker-")
        executable = Path(directory.name) / "directx-runtime.exe"
        try:
            result = _run(
                [
                    compiler,
                    "/nologo",
                    "/std:c++17",
                    "/EHsc",
                    "/W4",
                    "/WX",
                    "/O2",
                    str(source),
                    f"/Fe{executable}",
                    f"/Fo{Path(directory.name) / 'worker.obj'}",
                    "/link",
                    "d3d12.lib",
                    "dxgi.lib",
                ],
                timeout=120,
                env=environment,
            )
            if result.returncode or not executable.is_file():
                raise _error(
                    "DirectX worker compilation failed.",
                    "worker-compile-failed",
                    stdout=result.stdout.decode(errors="replace"),
                    stderr=result.stderr.decode(errors="replace"),
                )
            if warp.is_file():
                shutil.copy2(warp, executable.with_name(warp.name))
        except BaseException:
            directory.cleanup()
            raise
        _WORKERS[key] = (directory, executable)
        atexit.register(directory.cleanup)
        return executable


def encode_dispatches(dispatches, allocations, view_keys):
    """Encode validated physical allocations once, then reference them by index."""
    from .native_runtime_drivers import _prepared_allocation_view_payload

    if not 0 < len(allocations) <= 65536 or not 0 < len(dispatches) <= 65536:
        raise _error(
            "DirectX execution exceeds its request count limit.", "request-size-limit"
        )
    size = 12 + sum(8 + item.views[0].allocation_size for item in allocations)
    size += sum(24 + len(node.shader) + 32 * len(node.buffers) for node in dispatches)
    if size > 512 * 1024 * 1024:
        raise _error(
            "DirectX execution exceeds its request size limit.", "request-size-limit"
        )
    for node in dispatches:
        access = {}
        for view in node.buffers:
            key = view_keys[id(view)]
            access.setdefault(key, set()).add(view.namespace == "uav")
        for key, kinds in access.items():
            if len(kinds) > 1:
                raise _error(
                    "DirectX ranged allocations cannot mix SRV/CBV reads with UAV writes in one dispatch.",
                    "allocation-state-incompatible",
                    allocationId=next(
                        item.allocation_id for item in allocations if item.key == key
                    ),
                    targetConstraint="classic-resource-state-access",
                )
    stream = io.BytesIO()
    stream.write(struct.pack("<II", 0x31565844, len(allocations)))
    indices = {allocation.key: index for index, allocation in enumerate(allocations)}
    descriptions = []
    for allocation in allocations:
        size = allocation.views[0].allocation_size
        if not 0 < size <= _MAX_BUFFER_BYTES:
            raise _error(
                "DirectX allocation exceeds its bounded execution limit.",
                "allocation-size-limit",
                allocationId=allocation.allocation_id,
                allocationByteLength=size,
                maxBufferBytes=_MAX_BUFFER_BYTES,
            )
        payload = allocation.upload_payload or b""
        if len(payload) > size:
            raise _error(
                "DirectX upload exceeds its allocation.",
                "allocation-upload-out-of-bounds",
            )
        stream.write(struct.pack("<Q", size))
        stream.write(payload.ljust(size, b"\x00"))
        descriptions.append(
            {
                "allocationId": allocation.allocation_id,
                "allocationByteLength": size,
                "views": [
                    _prepared_allocation_view_payload(view) for view in allocation.views
                ],
            }
        )
    stream.write(struct.pack("<I", len(dispatches)))
    for node in dispatches:
        stream.write(struct.pack("<Q", len(node.shader)))
        stream.write(node.shader)
        stream.write(struct.pack("<4I", *node.workgroup_count, len(node.buffers)))
        for view in node.buffers:
            stream.write(
                struct.pack(
                    "<4I2Q",
                    {"cbv": 0, "srv": 1, "uav": 2}[view.namespace],
                    view.binding_index,
                    indices[view_keys[id(view)]],
                    view.stride,
                    view.byte_offset,
                    view.size,
                )
            )
    return stream.getvalue(), descriptions


def decode_readbacks(payload, allocations):
    stream = io.BytesIO(payload)

    def read(size):
        value = stream.read(size)
        if len(value) != size:
            raise _error(
                "DirectX worker returned a truncated result.", "worker-output-invalid"
            )
        return value

    version, count = struct.unpack("<II", read(8))
    if version != 0x31525844 or count != len(allocations):
        raise _error(
            "DirectX worker returned an incompatible result.", "worker-output-invalid"
        )
    results, addresses = [], []
    for allocation in allocations:
        address, size = struct.unpack("<QQ", read(16))
        if not address or size != allocation.views[0].allocation_size:
            raise _error(
                "DirectX worker returned an invalid allocation.",
                "worker-output-invalid",
            )
        addresses.append(address)
        results.append(read(size))
    if stream.read(1) or len(set(addresses)) != len(addresses):
        raise _error(
            "DirectX worker returned conflicting allocations.", "worker-output-invalid"
        )
    return results, addresses


def _device_identity(device):
    name = getattr(device, "name", None)
    hardware = getattr(device, "is_hardware", None)
    if (
        not isinstance(name, str)
        or not name
        or "\0" in name
        or type(hardware) is not bool
    ):
        raise _error(
            "DirectX buffer views require a named adapter and its hardware flag.",
            "device-selection-failed",
        )
    identity = {"name": name, "isHardware": hardware}
    for field, key, maximum in (
        ("vendor_id", "vendorId", (1 << 32) - 1),
        ("dedicated_video_memory", "dedicatedVideoMemory", (1 << 64) - 1),
        ("dedicated_system_memory", "dedicatedSystemMemory", (1 << 64) - 1),
        ("shared_system_memory", "sharedSystemMemory", (1 << 64) - 1),
    ):
        value = getattr(device, field, None)
        if type(value) is not int or not 0 <= value <= maximum:
            raise _error(
                "DirectX adapter identity is incomplete or invalid.",
                "device-selection-failed",
                field=field,
            )
        identity[key] = value
    return identity


def execute_buffer_views(dispatches, allocations, view_keys, state, *, device):
    """Run unchanged DXIL using native CBV/SRV/UAV descriptors and shared buffers."""
    from .native_runtime_drivers import _buffer_readback

    payload, descriptions = encode_dispatches(dispatches, allocations, view_keys)
    identity = _device_identity(device)
    executable = _worker()
    with tempfile.TemporaryDirectory(prefix="crosstl-directx-dispatch-") as directory:
        request = Path(directory) / "request.bin"
        output = Path(directory) / "readbacks.bin"
        request.write_bytes(payload)
        result = _run(
            [
                str(executable),
                str(request),
                str(output),
                identity["name"],
                str(int(identity["isHardware"])),
                str(identity["vendorId"]),
                str(identity["dedicatedVideoMemory"]),
                str(identity["dedicatedSystemMemory"]),
                str(identity["sharedSystemMemory"]),
            ],
            timeout=120,
        )
        if result.returncode or not output.is_file():
            raise RuntimeAdapterDispatchError(
                "DirectX buffer-view execution failed.",
                details={
                    "target": "directx",
                    "reasonKind": "native-buffer-view-failed",
                    "stderr": result.stderr.decode(errors="replace"),
                    "returnCode": result.returncode,
                    "adapterIdentity": identity,
                },
            )
        readbacks, addresses = decode_readbacks(output.read_bytes(), allocations)
    indices = {allocation.key: index for index, allocation in enumerate(allocations)}
    results = {}
    for node in dispatches:
        for view in node.buffers:
            if view.readback:
                index = indices[view_keys[id(view)]]
                data = readbacks[index][view.byte_offset : view.byte_offset + view.size]
                results[view.output_name or view.name] = _buffer_readback(
                    data,
                    view.dtype,
                    view.shape,
                    target="DirectX",
                    encoding=view.readback_encoding,
                )
    if isinstance(getattr(state, "details", None), dict):
        state.details["directxRuntime"] = {
            "runtime": "native-buffer-views",
            "timeoutSeconds": 120,
            "device": identity["name"],
            "adapterIdentity": identity,
            "requestSHA256": hashlib.sha256(payload).hexdigest(),
            "allocations": [
                {
                    **description,
                    "gpuVirtualAddress": address,
                    "readbackSHA256": hashlib.sha256(data).hexdigest(),
                }
                for description, address, data in zip(
                    descriptions, addresses, readbacks
                )
            ],
        }
    return results
