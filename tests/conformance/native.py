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


def loaded_module_version(name):
    """Path and fixed file version of a DLL loaded into this Windows process, or ``(None, None)``."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleFileNameW.argtypes = [
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    kernel32.GetModuleFileNameW.restype = wintypes.DWORD
    handle = kernel32.GetModuleHandleW(name)
    if not handle:
        return None, None
    buffer = ctypes.create_unicode_buffer(32768)
    if not kernel32.GetModuleFileNameW(handle, buffer, len(buffer)):
        return None, None
    path = buffer.value
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
        return path, None
    # VS_FIXEDFILEINFO starts with dwSignature, dwStrucVersion, dwFileVersionMS, dwFileVersionLS.
    words = ctypes.cast(fixed, ctypes.POINTER(wintypes.DWORD * 4)).contents
    if words[0] != 0xFEEF04BD:
        return path, None
    high, low = words[2], words[3]
    return path, f"{high >> 16}.{high & 0xFFFF}.{low >> 16}.{low & 0xFFFF}"


def _warp_identity(job, name):
    """The adapter name plus the d3d10warp.dll actually loaded; CROSSTL_CONFORMANCE_WARP_VERSION pins it."""
    path, version = loaded_module_version("d3d10warp.dll")
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


def execute_d3d12(job):
    compushady, device = _d3d12_software_device()
    _expect(job, device.name)
    views = []
    for resource, filename in zip(job["resources"], FILES):
        data = (Path(job["directory"]) / filename).read_bytes()
        upload = compushady.Buffer(len(data), compushady.HEAP_UPLOAD, device=device)
        upload.upload(data)
        stride = resource["stride"] if resource["kind"] == "structured" else 0
        buffer = compushady.Buffer(len(data), stride=stride, device=device)
        upload.copy_to(buffer)
        views.append((resource["register"], buffer, filename))
    cbv = []
    block, data = _constants(job, "uniform")
    if block is not None:
        upload = compushady.Buffer(len(data), compushady.HEAP_UPLOAD, device=device)
        upload.upload(data)
        constants = compushady.Buffer(len(data), device=device)
        upload.copy_to(constants)
        # Constant-buffer views are positional too; lower b registers get placeholders.
        cbv = [compushady.Buffer(256, device=device) for _ in range(block["slot"])]
        cbv.append(constants)
    srv = [buffer for register, buffer, _ in views if register.startswith("t")]
    uav = {
        int(register[1:]): buffer
        for register, buffer, _ in views
        if register.startswith("u")
    }
    # UAV slots are positional; an unused u0 slot (input bound as t0) gets a placeholder.
    placeholder = compushady.Buffer(16, stride=4, device=device)
    shader = compushady.Compute(
        Path(job["binary"]).read_bytes(),
        cbv=cbv,
        srv=srv,
        uav=[uav.get(slot, placeholder) for slot in range(max(uav) + 1)],
        device=device,
    )
    shader.dispatch(_groups(job), 1, 1)
    for _, buffer, filename in views:
        readback = compushady.Buffer(
            buffer.size, compushady.HEAP_READBACK, device=device
        )
        buffer.copy_to(readback)
        (Path(job["directory"]) / filename).write_bytes(readback.readback())
    # d3d10warp.dll is loaded once the device exists, so its version is read after the dispatch.
    return {"runtime": job["runtime"], "device": _warp_identity(job, device.name)}
