"""Native compute adapters for the harness's ordered storage-buffer ABI.

Each adapter binds the input files at slots 0 through n-1 and ``output.bin`` at slot n of the job directory,
dispatches ``ceil(threads / group)`` one-dimensional workgroups, and writes every buffer back so the runner can
check the output and confirm that every input buffer and the output guard words were not overwritten. When the job
has a constants block, ``constants.bin`` is bound read-only as that API's uniform or constant buffer at the slot
discovered in the artifact, or pushed as push constants. Every adapter reports the identity of the device that ran
the job; the runner rejects a result from an unexpected driver.
"""

import os
import re
import struct
from pathlib import Path

from .process import run

FILES = ("input.bin", "output.bin")
CONSTANTS = "constants.bin"
VULKAN_TIMEOUT = 15


class DriverRejected(RuntimeError):
    """The driver's own compiler or pipeline creation refused an artifact the offline checker accepted."""


class ExecutionTimeout(RuntimeError):
    """The dispatch did not complete in time, for example because a translated loop no longer terminates."""


def buffer_files(input_count):
    """Storage files in binding order, retaining the existing single-input filenames."""
    if input_count < 1:
        raise ValueError("At least one input buffer is required")
    return (
        "input.bin",
        *(f"input{slot}.bin" for slot in range(1, input_count)),
        "output.bin",
    )


def _groups(job):
    return (job["threads"] + job["group"][0] - 1) // job["group"][0]


def _expect(job, identity):
    if job.get("expect") and job["expect"] not in identity:
        raise RuntimeError(f"Expected a {job['expect']} device, ran on {identity}")
    return identity


def _constants(job, *kinds):
    """The job's constants block and its bytes, or ``(None, None)``; a block kind the API lacks is a harness error."""
    block = job.get("constants")
    if block is None:
        return None, None
    if block["kind"] not in kinds:
        raise RuntimeError(
            f"{job['runtime']} cannot bind a {block['kind']} constants block"
        )
    return block, (Path(job["directory"]) / CONSTANTS).read_bytes()


def execute_metal(job):
    import Metal

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("No Metal device")
    identity = _expect(job, str(device.name()))
    options = Metal.MTLCompileOptions.new()
    strict = job["config"] == "strict"
    options.setMathMode_(Metal.MTLMathModeSafe if strict else Metal.MTLMathModeFast)
    options.setMathFloatingPointFunctions_(
        Metal.MTLMathFloatingPointFunctionsPrecise
        if strict
        else Metal.MTLMathFloatingPointFunctionsFast
    )
    library, error = device.newLibraryWithSource_options_error_(
        Path(job["artifact"]).read_text(), options, None
    )
    if library is None:
        raise DriverRejected(f"Metal driver compile: {error}")
    function = library.newFunctionWithName_(job["entry"])
    if function is None:
        raise RuntimeError(f"Missing Metal entry {job['entry']}")
    pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
    if pipeline is None:
        raise DriverRejected(f"Metal pipeline: {error}")
    if pipeline.maxTotalThreadsPerThreadgroup() < job["group"][0]:
        raise RuntimeError(
            f"Metal pipeline allows {pipeline.maxTotalThreadsPerThreadgroup()} threads per group"
        )
    buffers = []
    for binding, filename in zip(
        job["bindings"], buffer_files(len(job["bindings"]) - 1)
    ):
        data = (Path(job["directory"]) / filename).read_bytes()
        buffer = device.newBufferWithBytes_length_options_(
            data, len(data), Metal.MTLResourceStorageModeShared
        )
        if buffer is None:
            raise RuntimeError("Metal buffer allocation failed")
        buffers.append((binding, buffer, filename))
    block, data = _constants(job, "uniform")
    constants = None
    if block is not None:
        constants = device.newBufferWithBytes_length_options_(
            data, len(data), Metal.MTLResourceStorageModeShared
        )
        if constants is None:
            raise RuntimeError("Metal constants allocation failed")
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for binding, buffer, _ in buffers:
        encoder.setBuffer_offset_atIndex_(buffer, 0, binding)
    if constants is not None:
        encoder.setBuffer_offset_atIndex_(constants, 0, block["slot"])
    # Metal hosts dispatch exactly `threads` threads (non-uniform threadgroups); the other APIs can only launch
    # whole workgroups, which is the hazard the partial-threadgroup cases measure.
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(job["threads"], 1, 1), Metal.MTLSizeMake(*job["group"])
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    if command.error() is not None:
        raise RuntimeError(f"Metal execution: {command.error()}")
    for _, buffer, filename in buffers:
        (Path(job["directory"]) / filename).write_bytes(
            bytes(buffer.contents().as_buffer(buffer.length()))
        )
    return {"device": identity, "runtime": job["runtime"]}


def execute_opengl(job):
    import moderngl

    context = moderngl.create_standalone_context(require=430, backend="egl")
    buffers, shader = [], None
    try:
        identity = _expect(
            job, f"{context.info['GL_RENDERER']}; {context.info['GL_VERSION']}"
        )
        # The driver compile is the "real driver" check for GLSL output.
        try:
            shader = context.compute_shader(Path(job["artifact"]).read_text())
        except moderngl.Error as error:
            raise DriverRejected(str(error)) from error
        for binding, filename in zip(
            job["bindings"], buffer_files(len(job["bindings"]) - 1)
        ):
            buffer = context.buffer((Path(job["directory"]) / filename).read_bytes())
            buffers.append((buffer, filename))
            buffer.bind_to_storage_buffer(binding)
        block, data = _constants(job, "uniform")
        if block is not None:
            constants = context.buffer(data)
            buffers.append((constants, None))
            constants.bind_to_uniform_block(block["slot"])
        shader.run(group_x=_groups(job))
        context.finish()
        for buffer, filename in buffers:
            if filename is not None:
                (Path(job["directory"]) / filename).write_bytes(buffer.read())
        return {"device": identity, "runtime": job["runtime"]}
    finally:
        for buffer, _ in buffers:
            buffer.release()
        if shader is not None:
            shader.release()
        context.release()


def execute_vulkan(job):
    executable = os.environ.get("CROSSTL_CONFORMANCE_VKRUN", "vkrun")
    directory = Path(job["directory"])
    args = [executable, job["binary"], job["entry"], _groups(job), 1, 1]
    for binding, filename in zip(
        job["bindings"], buffer_files(len(job["bindings"]) - 1)
    ):
        args.append(f"{binding}:inout:{directory / filename}")
    block, _ = _constants(job, "uniform", "push")
    if block is not None and block["kind"] == "push":
        args.append(f"push:{directory / CONSTANTS}")
    elif block is not None:
        args.append(f"{block['slot']}:uniform:{directory / CONSTANTS}")
    result = run(args, timeout=job.get("timeout", VULKAN_TIMEOUT))
    if result.status == "timeout":
        raise ExecutionTimeout(f"vkrun exceeded {job.get('timeout', VULKAN_TIMEOUT)} s")
    if re.match(r"error (vkCreateShaderModule|vkCreateComputePipelines)", result.log):
        raise DriverRejected(result.log[-1500:])
    if result.status != "accepted" or not result.log.startswith("ok"):
        raise RuntimeError(f"Vulkan execution {result.status}: {result.log[-1500:]}")
    fields = dict(re.findall(r"^(\w+)=(.*)$", result.log, flags=re.M))
    if "device" not in fields or "driver" not in fields:
        raise RuntimeError("The Vulkan runner did not report its driver and device")
    identity = _expect(
        job, f"{fields['driver']} {fields.get('driver_info', '')} on {fields['device']}"
    )
    return {"runtime": job["runtime"], "device": identity, "api": fields.get("api")}


def file_version(path):
    """Fixed file version of a Windows DLL on disk, or ``None``."""
    import ctypes
    from ctypes import wintypes

    version = ctypes.WinDLL("version", use_last_error=True)
    version.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, wintypes.LPDWORD]
    version.GetFileVersionInfoSizeW.restype = wintypes.DWORD
    version.GetFileVersionInfoW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
    ]
    version.GetFileVersionInfoW.restype = wintypes.BOOL
    version.VerQueryValueW.argtypes = [
        wintypes.LPCVOID,
        wintypes.LPCWSTR,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.UINT),
    ]
    version.VerQueryValueW.restype = wintypes.BOOL
    size = version.GetFileVersionInfoSizeW(path, None)
    data = ctypes.create_string_buffer(size) if size else None
    fixed = wintypes.LPVOID()
    length = wintypes.UINT()
    if (
        data is None
        or not version.GetFileVersionInfoW(path, 0, size, data)
        or not version.VerQueryValueW(
            data, "\\", ctypes.byref(fixed), ctypes.byref(length)
        )
        or length.value < 16
    ):
        return None
    # VS_FIXEDFILEINFO starts with dwSignature, dwStrucVersion, dwFileVersionMS, dwFileVersionLS.
    words = ctypes.cast(fixed, ctypes.POINTER(wintypes.DWORD * 4)).contents
    if words[0] != 0xFEEF04BD:
        return None
    high, low = words[2], words[3]
    return f"{high >> 16}.{high & 0xFFFF}.{low >> 16}.{low & 0xFFFF}"


def _warp_identity(job, name, path):
    """The adapter name plus the d3d10warp.dll the worker loaded; CROSSTL_CONFORMANCE_WARP_VERSION pins it."""
    version = file_version(path) if path else None
    wanted = os.environ.get("CROSSTL_CONFORMANCE_WARP_VERSION")
    if wanted and version != wanted:
        raise RuntimeError(
            f"Expected d3d10warp.dll {wanted}, loaded {version or 'no version'} from {path or 'nowhere'}"
        )
    return _expect(job, f"{name}; d3d10warp {version or 'unknown'}")


def _d3d12_software_device():
    import compushady
    from compushady import config

    config.set_backend("d3d12")
    devices = [
        device
        for device in compushady.get_discovered_devices()
        if not device.is_hardware
    ]
    if len(devices) != 1:
        raise RuntimeError(f"Expected one D3D12 software device, found {len(devices)}")
    return compushady, devices[0]


PROBE_HLSL = """RWStructuredBuffer<uint> o : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { o[id.x] = 7u; }
"""


def d3d12_pipeline_models(dxc, directory, models):
    """Map each ``cs_6_x`` profile to ``"ok"`` or the reason DXC or pipeline creation rejected a trivial kernel.

    A device whose runtime or driver lacks a shader model refuses the pipeline with E_INVALIDARG, which is
    indistinguishable from a binding error unless the same one-UAV kernel is tried at each model in turn.
    """
    compushady, device = _d3d12_software_device()
    source = Path(directory) / "probe.hlsl"
    source.write_text(PROBE_HLSL)
    results = {}
    for model in models:
        binary = Path(directory) / f"probe_{model}.dxil"
        compiled = run([*dxc, source, "-T", model, "-E", "main", "-Fo", binary])
        if compiled.status != "accepted" or not binary.is_file():
            results[model] = f"dxc {compiled.status}: {compiled.log.strip()[-200:]}"
            continue
        buffer = compushady.Buffer(16, stride=4, device=device)
        try:
            compushady.Compute(binary.read_bytes(), uav=[buffer], device=device)
        except Exception as error:  # noqa: BLE001 - the driver's message is the result
            results[model] = f"{type(error).__name__}: {str(error).strip()}"
        else:
            results[model] = "ok"
    return device.name, results


# The request, readback and runtime-identity formats of crosstl/project/directx_runtime_worker.cpp.
D3D12_REQUEST, D3D12_READBACKS, D3D12_IDENTITY = 0x31565844, 0x31525844, 0x314C5844
D3D12_CBV, D3D12_SRV, D3D12_UAV, D3D12_RAW = 0, 1, 2, 4
D3D12_LIBRARIES = ("d3d12.dll", "D3D12Core.dll", "d3d10warp.dll")
# Below the 60 s limit of the job process, so a dispatch that never finishes is reported as a timeout.
D3D12_TIMEOUT = 45


def d3d12_request(job, shader):
    """Encode one dispatch for the buffer-view worker: an allocation per buffer, each view at its own register.

    A ``ByteAddressBuffer`` gets a raw view (``R32_TYPELESS`` with the raw flag) and a ``StructuredBuffer`` a
    structured view of its reflected stride, as HLSL requires; the constants block becomes a constant-buffer view
    padded to 256 bytes. Returns the request bytes and the buffer files in allocation order.
    """
    directory = Path(job["directory"])
    allocations, views, files = [], [], []
    for resource, filename in zip(job["resources"], FILES):
        data = (directory / filename).read_bytes()
        register = resource["register"]
        kind = {"t": D3D12_SRV, "u": D3D12_UAV}[register[0]]
        raw = resource["kind"] == "raw"
        stride = 0 if raw else resource["stride"]
        length = len(data) if raw else len(data) - len(data) % stride
        views.append(
            (
                kind | (D3D12_RAW if raw else 0),
                int(register[1:]),
                len(allocations),
                stride,
                0,
                length,
            )
        )
        allocations.append(data)
        files.append(filename)
    block, data = _constants(job, "uniform")
    if block is not None:
        padded = data.ljust(max(256, -(-len(data) // 256) * 256), b"\0")
        views.append((D3D12_CBV, block["slot"], len(allocations), 0, 0, len(padded)))
        allocations.append(padded)
    parts = [struct.pack("<II", D3D12_REQUEST, len(allocations))]
    for allocation in allocations:
        parts += [struct.pack("<Q", len(allocation)), allocation]
    parts += [
        struct.pack("<IQ", 1, len(shader)),
        shader,
        struct.pack("<4I", _groups(job), 1, 1, len(views)),
        *(struct.pack("<4I2Q", *view) for view in views),
    ]
    return b"".join(parts), files, [len(allocation) for allocation in allocations]


def d3d12_readbacks(payload, sizes):
    """Decode the worker's readbacks: every allocation's final contents, in request order."""
    version, count = struct.unpack_from("<II", payload)
    if version != D3D12_READBACKS or count != len(sizes):
        raise RuntimeError("The D3D12 worker returned an incompatible readback")
    offset, results = 8, []
    for expected in sizes:
        _, size = struct.unpack_from("<QQ", payload, offset)
        offset += 16
        if size != expected or offset + size > len(payload):
            raise RuntimeError("The D3D12 worker returned a truncated readback")
        results.append(payload[offset : offset + size])
        offset += size
    if offset != len(payload):
        raise RuntimeError("The D3D12 worker returned trailing readback data")
    return results


def d3d12_loaded_libraries(payload):
    """Map each library the worker observed after creating its device to the path it was loaded from."""
    version, _, count = struct.unpack_from("<III", payload)
    if version != D3D12_IDENTITY or count != len(D3D12_LIBRARIES):
        raise RuntimeError("The D3D12 worker reported an incompatible runtime identity")
    offset, paths = 12, {}
    for name in D3D12_LIBRARIES:
        loaded, _, length = struct.unpack_from("<III", payload, offset)
        offset += 12
        path = payload[offset : offset + 2 * length].decode("utf-16-le")
        offset += 2 * length
        paths[name] = path if loaded and path else None
    return paths


def execute_d3d12(job):
    """Run the DXIL through CrossTL's Direct3D 12 buffer-view worker on the software adapter.

    The worker builds a root signature with one descriptor table per view at the register the artifact declares,
    so raw, structured and constant-buffer views each get the descriptor HLSL requires. The runtime provisions the
    worker once per run and names it in ``CROSSTL_CONFORMANCE_D3D12_WORKER``.
    """
    _, device = _d3d12_software_device()
    _expect(job, device.name)
    worker = os.environ.get("CROSSTL_CONFORMANCE_D3D12_WORKER")
    if not worker:
        raise RuntimeError("CROSSTL_CONFORMANCE_D3D12_WORKER names no D3D12 worker")
    directory = Path(job["directory"])
    payload, files, sizes = d3d12_request(job, Path(job["binary"]).read_bytes())
    request, output = directory / "d3d12-request.bin", directory / "d3d12-readbacks.bin"
    request.write_bytes(payload)
    identity = (
        device.name,
        int(device.is_hardware),
        device.vendor_id,
        device.dedicated_video_memory,
        device.dedicated_system_memory,
        device.shared_system_memory,
    )
    timeout = job.get("timeout", D3D12_TIMEOUT)
    result = run([worker, request, output, *identity], timeout=timeout)
    log = "\n".join(
        line
        for line in result.log.splitlines()
        if not line.startswith("CROSSTL_DIRECTX_PROGRESS")
    )
    if result.status == "timeout" or "GPU fence timeout" in log:
        raise ExecutionTimeout(f"The D3D12 dispatch exceeded {timeout} s")
    if "CreateComputePipelineState failed" in log:
        raise DriverRejected(log[-1500:])
    if result.status != "accepted" or not output.is_file():
        raise RuntimeError(f"D3D12 execution {result.status}: {log[-1500:]}")
    for filename, data in zip(files, d3d12_readbacks(output.read_bytes(), sizes)):
        (directory / filename).write_bytes(data)
    runtime = Path(f"{output}.runtime")
    libraries = (
        d3d12_loaded_libraries(runtime.read_bytes()) if runtime.is_file() else {}
    )
    return {
        "runtime": job["runtime"],
        "device": _warp_identity(job, device.name, libraries.get("d3d10warp.dll")),
    }
