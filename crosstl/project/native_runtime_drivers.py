"""Optional native runtime drivers for project runtime verification."""

from __future__ import annotations

import ctypes
import importlib
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeAdapterDispatchError,
    RuntimeAdapterSetupError,
    RuntimeExecutionRequest,
    RuntimeExecutorAvailability,
    RuntimeExecutorUnavailable,
)


@dataclass(frozen=True)
class _PreparedDirectXBuffer:
    name: str
    namespace: str
    binding_index: int
    dtype: str
    shape: tuple[int, ...]
    source: str | None
    output_name: str | None
    payload: bytes
    allocation_size: int
    stride: int = 0

    @property
    def size(self) -> int:
        return len(self.payload)


@dataclass
class _DirectXBufferResource:
    prepared: _PreparedDirectXBuffer
    upload_buffer: Any
    device_buffer: Any


@dataclass(frozen=True)
class _DirectXConstantValue:
    name: str
    binding_index: int
    byte_offset: int
    dtype: str
    payload: bytes


@dataclass(frozen=True)
class _PreparedVulkanBuffer:
    name: str
    set_index: int
    binding_index: int
    resource_kind: str | None
    dtype: str
    shape: tuple[int, ...]
    source: str | None
    output_name: str | None
    payload: bytes

    @property
    def size(self) -> int:
        return len(self.payload)


@dataclass
class _VulkanBufferResource:
    prepared: _PreparedVulkanBuffer
    buffer: Any
    memory: Any


@dataclass(frozen=True)
class _PreparedOpenGLBuffer:
    name: str
    binding_index: int
    resource_kind: str | None
    dtype: str
    shape: tuple[int, ...]
    source: str | None
    output_name: str | None
    payload: bytes
    allocation_size: int

    @property
    def size(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class _PreparedOpenGLSpecialization:
    name: str
    constant_id: int
    dtype: str
    value: Any
    encoded_word: int
    source: str | None
    value_provenance: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "id": self.constant_id,
            "dtype": self.dtype,
            "value": self.value,
            "valueProvenance": dict(self.value_provenance),
        }
        if self.source is not None:
            payload["source"] = self.source
        return payload


@dataclass(frozen=True)
class _OpenGLSPIRVAPI:
    gl: Any
    specialize_shader: Any
    specialize_entry_point: str
    binary_format: int


class _OpenGLSPIRVUnavailable(RuntimeError):
    def __init__(self, message: str, *, details: Mapping[str, Any]):
        super().__init__(message)
        self.details = dict(details)


_OPENGL_SPIRV_HEADER_BYTE_LENGTH = 5 * 4
_DIRECTX_MAX_PADDED_DESCRIPTOR_COUNT = 4096


class _OpenGLSPIRVComputeShader:
    def __init__(self, api: _OpenGLSPIRVAPI, *, shader: int, program: int):
        self.api = api
        self.shader = shader
        self.program = program
        self.released = False

    def set_uniform(self, name: str, binding: Any) -> None:
        gl = self.api.gl
        try:
            location = int(gl.glGetUniformLocation(self.program, name))
        except Exception as exc:
            raise _opengl_setup_error(
                f"OpenGL runtime uniform {name!r} could not be queried: {exc}",
                "uniform-query-failed",
                constant=name,
            ) from exc
        if location < 0:
            raise _opengl_setup_error(
                f"OpenGL runtime constant {name!r} is not an active uniform.",
                "active-uniform-missing",
                constant=name,
            )

        values, suffix = _prepare_opengl_uniform_values(name, binding)
        function_name = f"glUniform{len(values)}{suffix}"
        function = getattr(gl, function_name, None)
        if not _opengl_entry_point_available(function):
            raise _opengl_setup_error(
                f"OpenGL runtime uniform entry point {function_name} is unavailable.",
                "uniform-entry-point-unavailable",
                constant=name,
                missingEntryPoints=[function_name],
            )
        try:
            gl.glUseProgram(self.program)
            function(location, *values)
        except RuntimeAdapterSetupError:
            raise
        except Exception as exc:
            raise _opengl_setup_error(
                f"OpenGL runtime constant {name!r} could not be bound: {exc}",
                "constant-binding-failed",
                constant=name,
            ) from exc
        finally:
            try:
                gl.glUseProgram(0)
            except Exception:
                pass

    def run(self, *, group_x: int, group_y: int, group_z: int) -> None:
        gl = self.api.gl
        gl.glUseProgram(self.program)
        try:
            gl.glDispatchCompute(group_x, group_y, group_z)
        finally:
            gl.glUseProgram(0)

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        gl = self.api.gl
        try:
            gl.glDeleteProgram(self.program)
        finally:
            gl.glDeleteShader(self.shader)


class DirectXComputeRuntime:
    """Optional Direct3D 12 compute runtime for buffer fixtures.

    ``compushady`` is imported lazily and receives the DXIL emitted by the
    native DirectX parity adapter. Its descriptor-list API represents
    contiguous CBV, SRV, and UAV registers in space zero. Sparse reflected
    layouts are padded with internal zeroed descriptors before resource
    creation; duplicate bindings and nonzero register spaces are rejected.
    """

    name = "directx-compute-runtime"
    supported_platforms = ("win32",)

    def __init__(
        self,
        *,
        module_loader: Any | None = None,
        platform_name: str | None = None,
    ):
        self._module_loader = module_loader or importlib.import_module
        self.platform_name = platform_name or sys.platform

    def is_available(
        self,
        adapter: Any,
        request: RuntimeExecutionRequest,
    ) -> RuntimeExecutorAvailability:
        _ = adapter, request
        if not self._platform_supported():
            return RuntimeExecutorAvailability(
                False,
                reason=(
                    "Direct3D 12 compute runtime is available on Windows only; "
                    f"current platform is {self.platform_name}."
                ),
                details={
                    "reasonKind": "platform-unavailable",
                    "target": "directx",
                    "platform": self.platform_name,
                    "requiredPlatforms": list(self.supported_platforms),
                },
            )
        try:
            compushady = self._load_compushady()
        except RuntimeExecutorUnavailable as exc:
            return RuntimeExecutorAvailability(
                False,
                reason=str(exc),
                details={
                    "reasonKind": "dependency-unavailable",
                    "target": "directx",
                    "missingPythonModules": ["compushady"],
                },
            )

        try:
            backend = compushady.get_backend()
        except Exception as exc:  # pragma: no cover - depends on local D3D loader
            return RuntimeExecutorAvailability(
                False,
                reason=f"Direct3D 12 backend initialization failed: {exc}",
                details={
                    "reasonKind": "direct3d-runtime-unavailable",
                    "target": "directx",
                    "error": str(exc),
                },
            )
        backend_name = _compushady_backend_name(backend)
        if backend_name != "d3d12":
            return RuntimeExecutorAvailability(
                False,
                reason=(
                    "compushady is not using its Direct3D 12 backend; "
                    f"active backend is {backend_name or 'unknown'}."
                ),
                details={
                    "reasonKind": "backend-unavailable",
                    "target": "directx",
                    "requiredBackend": "d3d12",
                    "activeBackend": backend_name,
                },
            )
        try:
            device = self._select_device(compushady)
        except Exception as exc:  # pragma: no cover - depends on local adapters
            return RuntimeExecutorAvailability(
                False,
                reason=f"Direct3D 12 device selection failed: {exc}",
                details={
                    "reasonKind": "device-selection-failed",
                    "target": "directx",
                    "runtime": self.name,
                    "error": str(exc),
                },
            )
        if device is None:
            return RuntimeExecutorAvailability(
                False,
                reason="No Direct3D 12 compute device is available.",
                details={
                    "reasonKind": "device-unavailable",
                    "target": "directx",
                    "runtime": self.name,
                },
            )

        details: dict[str, Any] = {
            "reasonKind": "available",
            "target": "directx",
            "runtime": self.name,
            "backend": backend_name,
        }
        device_name = getattr(device, "name", None)
        if isinstance(device_name, str) and device_name:
            details["device"] = device_name
        for field_name, detail_name in (
            ("is_hardware", "isHardware"),
            ("is_discrete", "isDiscrete"),
        ):
            value = getattr(device, field_name, None)
            if isinstance(value, bool):
                details[detail_name] = value
        return RuntimeExecutorAvailability(True, details=details)

    def load_artifact(self, adapter: Any, state: Any, module_path: Path) -> bytes:
        _ = adapter, state
        try:
            shader = Path(module_path).read_bytes()
        except OSError as exc:
            raise RuntimeAdapterSetupError(
                f"DirectX runtime artifact could not be read: {exc}",
                details={
                    "target": "directx",
                    "reasonKind": "artifact-read-failed",
                    "modulePath": str(module_path),
                },
            ) from exc
        if not shader:
            raise RuntimeAdapterSetupError(
                "DirectX runtime artifact is empty.",
                details={
                    "target": "directx",
                    "reasonKind": "artifact-empty",
                    "modulePath": str(module_path),
                },
            )
        return shader

    def dispatch(
        self,
        adapter: Any,
        state: Any,
        request: NativeRuntimeDispatchRequest,
    ) -> dict[str, Mapping[str, Any]]:
        _ = adapter, state
        if not self._platform_supported():
            raise RuntimeExecutorUnavailable(
                "Direct3D 12 compute runtime is available on Windows only."
            )
        if (
            request.dispatch is not None
            and request.dispatch.entry_point is not None
            and request.entry_point is not None
            and request.dispatch.entry_point != request.entry_point
        ):
            raise RuntimeAdapterSetupError(
                "DirectX runtime entry-point metadata is inconsistent.",
                details={
                    "target": "directx",
                    "reasonKind": "entry-point-mismatch",
                    "entryPoint": request.entry_point,
                    "dispatchEntryPoint": request.dispatch.entry_point,
                },
            )

        compushady = self._load_compushady()
        shader = self._shader_code(request)
        prepared = (
            *_prepare_directx_buffers(request.buffers),
            *_prepare_directx_constants(request.constants),
        )
        prepared = _complete_directx_register_layout(prepared)
        prepared = _validate_directx_register_layout(prepared)
        workgroup_count = _workgroup_count(request, target="DirectX")

        owned_objects: list[Any] = []
        resources: list[_DirectXBufferResource] = []
        compute = None
        try:
            try:
                backend = compushady.get_backend()
                backend_name = _compushady_backend_name(backend)
                if backend_name != "d3d12":
                    raise RuntimeExecutorUnavailable(
                        "compushady must use its Direct3D 12 backend for DXIL dispatch."
                    )
                device = self._select_device(compushady)
                if device is None:
                    raise RuntimeExecutorUnavailable(
                        "No Direct3D 12 compute device is available."
                    )
            except RuntimeExecutorUnavailable:
                raise
            except Exception as exc:
                raise RuntimeAdapterSetupError(
                    f"Direct3D 12 device selection failed: {exc}",
                    details={
                        "target": "directx",
                        "runtime": self.name,
                        "reasonKind": "device-selection-failed",
                    },
                ) from exc

            for item in prepared:
                try:
                    resource = self._create_buffer_resource(
                        compushady,
                        device,
                        item,
                        owned_objects,
                    )
                except (RuntimeAdapterSetupError, RuntimeExecutorUnavailable):
                    raise
                except Exception as exc:
                    raise RuntimeAdapterSetupError(
                        f"DirectX resource creation failed for {item.name!r}: {exc}",
                        details={
                            "target": "directx",
                            "runtime": self.name,
                            "reasonKind": "resource-creation-failed",
                            "resource": item.name,
                            "namespace": item.namespace,
                            "binding": item.binding_index,
                        },
                    ) from exc
                resources.append(resource)

            bound_resources = {
                namespace: [
                    resource.device_buffer
                    for resource in resources
                    if resource.prepared.namespace == namespace
                ]
                for namespace in ("cbv", "srv", "uav")
            }
            try:
                compute = compushady.Compute(
                    shader,
                    cbv=bound_resources["cbv"],
                    srv=bound_resources["srv"],
                    uav=bound_resources["uav"],
                    device=device,
                )
                owned_objects.append(compute)
            except Exception as exc:
                raise RuntimeAdapterSetupError(
                    f"DirectX compute pipeline creation failed: {exc}",
                    details={
                        "target": "directx",
                        "runtime": self.name,
                        "reasonKind": "compute-pipeline-creation-failed",
                        "cbvCount": len(bound_resources["cbv"]),
                        "srvCount": len(bound_resources["srv"]),
                        "uavCount": len(bound_resources["uav"]),
                    },
                ) from exc
            try:
                compute.dispatch(*workgroup_count)
            except Exception as exc:
                raise RuntimeAdapterDispatchError(
                    f"DirectX compute dispatch failed: {exc}",
                    details={
                        "target": "directx",
                        "runtime": self.name,
                        "reasonKind": "dispatch-failed",
                        "workgroupCount": list(workgroup_count),
                    },
                ) from exc
            return self._read_outputs(
                compushady,
                device,
                resources,
                owned_objects,
            )
        except (
            RuntimeAdapterDispatchError,
            RuntimeAdapterSetupError,
            RuntimeExecutorUnavailable,
        ):
            raise
        except Exception as exc:
            raise RuntimeAdapterDispatchError(
                f"DirectX compute dispatch failed: {exc}",
                details={"target": "directx", "runtime": self.name},
            ) from exc
        finally:
            for value in reversed(owned_objects):
                _release_directx_object(value)
            resources.clear()
            compute = None

    def _platform_supported(self) -> bool:
        return any(
            self.platform_name.startswith(platform)
            for platform in self.supported_platforms
        )

    def _load_compushady(self) -> Any:
        try:
            return self._module_loader("compushady")
        except Exception as exc:
            raise RuntimeExecutorUnavailable(
                "compushady is unavailable; install the optional 'compushady' "
                "package to run DirectX runtime fixtures."
            ) from exc

    def _select_device(self, compushady: Any) -> Any:
        get_discovered_devices = getattr(compushady, "get_discovered_devices", None)
        if not callable(get_discovered_devices):
            raise RuntimeExecutorUnavailable(
                "compushady does not expose Direct3D device discovery."
            )
        devices = get_discovered_devices()
        if not devices:
            return None

        get_current_device = getattr(compushady, "get_current_device", None)
        if callable(get_current_device):
            return get_current_device()
        get_best_device = getattr(compushady, "get_best_device", None)
        if callable(get_best_device):
            return get_best_device()
        raise RuntimeExecutorUnavailable(
            "compushady does not expose a Direct3D device selector."
        )

    def _shader_code(self, request: NativeRuntimeDispatchRequest) -> bytes:
        if isinstance(request.loaded_artifact, (bytes, bytearray, memoryview)):
            shader = bytes(request.loaded_artifact)
            if shader:
                return shader
        return self.load_artifact(None, None, request.module_path)

    def _create_buffer_resource(
        self,
        compushady: Any,
        device: Any,
        prepared: _PreparedDirectXBuffer,
        owned_objects: list[Any],
    ) -> _DirectXBufferResource:
        upload_buffer = compushady.Buffer(
            prepared.allocation_size,
            compushady.HEAP_UPLOAD,
            device=device,
        )
        owned_objects.append(upload_buffer)
        payload = prepared.payload.ljust(prepared.allocation_size, b"\x00")
        upload_buffer.upload(payload)
        device_buffer = compushady.Buffer(
            prepared.allocation_size,
            compushady.HEAP_DEFAULT,
            stride=prepared.stride,
            device=device,
        )
        owned_objects.append(device_buffer)
        upload_buffer.copy_to(device_buffer)
        return _DirectXBufferResource(
            prepared=prepared,
            upload_buffer=upload_buffer,
            device_buffer=device_buffer,
        )

    def _read_outputs(
        self,
        compushady: Any,
        device: Any,
        resources: Sequence[_DirectXBufferResource],
        owned_objects: list[Any],
    ) -> dict[str, Mapping[str, Any]]:
        outputs: dict[str, Mapping[str, Any]] = {}
        for resource in resources:
            prepared = resource.prepared
            if prepared.source != "expectedOutput":
                continue
            try:
                readback = compushady.Buffer(
                    prepared.allocation_size,
                    compushady.HEAP_READBACK,
                    device=device,
                )
                owned_objects.append(readback)
                resource.device_buffer.copy_to(readback)
                payload = bytes(readback.readback(prepared.size))
            except Exception as exc:
                raise RuntimeAdapterDispatchError(
                    f"DirectX output readback failed for {prepared.name!r}: {exc}",
                    details={
                        "target": "directx",
                        "runtime": self.name,
                        "reasonKind": "readback-failed",
                        "resource": prepared.name,
                        "binding": prepared.binding_index,
                    },
                ) from exc
            outputs[prepared.output_name or prepared.name] = {
                "dtype": prepared.dtype,
                "shape": list(prepared.shape),
                "values": _unpack_values(payload, prepared.dtype, target="DirectX"),
            }
        return outputs


class OpenGLComputeRuntime:
    """Optional headless OpenGL compute runtime for buffer fixtures.

    ModernGL is imported lazily so the runtime remains optional. Context
    creation prefers EGL before allowing ModernGL to choose its platform
    default.
    """

    name = "opengl-compute-runtime"

    def __init__(
        self,
        *,
        module_loader: Any | None = None,
        context_factory: Any | None = None,
        context_backends: Sequence[str | None] = ("egl", None),
        require_version: int = 430,
    ):
        self._module_loader = module_loader or importlib.import_module
        self._context_factory = context_factory
        self.context_backends = tuple(context_backends)
        self.require_version = int(require_version)

    def is_available(
        self,
        adapter: Any,
        request: RuntimeExecutionRequest,
    ) -> RuntimeExecutorAvailability:
        _ = adapter
        requires_specialization = _opengl_request_requires_specialization(request)
        try:
            moderngl = self._load_moderngl()
        except RuntimeExecutorUnavailable as exc:
            return RuntimeExecutorAvailability(
                False,
                reason=str(exc),
                details={
                    "reasonKind": "dependency-unavailable",
                    "target": "opengl",
                    "missingPythonModules": ["moderngl"],
                },
            )

        context = None
        try:
            context, backend = self._create_context(moderngl)
            version_code = _opengl_context_version_code(context)
            if version_code < self.require_version:
                return RuntimeExecutorAvailability(
                    False,
                    reason=(
                        "OpenGL compute requires context version "
                        f"{self.require_version}, got {version_code}."
                    ),
                    details={
                        "reasonKind": "opengl-version-unsupported",
                        "target": "opengl",
                        "runtime": self.name,
                        "contextBackend": backend or "default",
                        "requiredVersionCode": self.require_version,
                        "versionCode": version_code,
                    },
                )
            specialization_details: dict[str, Any] = {}
            if requires_specialization:
                api = self._load_opengl_spirv_api(context)
                specialization_details = {
                    "specializationMode": "spirv",
                    "specializationEntryPoint": api.specialize_entry_point,
                    "requiredExtension": "GL_ARB_gl_spirv",
                }
        except _OpenGLSPIRVUnavailable as exc:
            return RuntimeExecutorAvailability(
                False,
                reason=str(exc),
                details={
                    "target": "opengl",
                    **exc.details,
                },
            )
        except Exception as exc:  # pragma: no cover - depends on local GL loader
            return RuntimeExecutorAvailability(
                False,
                reason=f"OpenGL compute runtime is unavailable: {exc}",
                details={
                    "reasonKind": "opengl-runtime-unavailable",
                    "target": "opengl",
                    "error": str(exc),
                },
            )
        finally:
            _release_opengl_object(context)

        return RuntimeExecutorAvailability(
            True,
            details={
                "reasonKind": "available",
                "target": "opengl",
                "runtime": self.name,
                "contextBackend": backend or "default",
                "versionCode": version_code,
                **specialization_details,
            },
        )

    def load_artifact(self, adapter: Any, state: Any, module_path: Path) -> str | bytes:
        _ = adapter, state
        if Path(module_path).suffix.lower() == ".spv":
            try:
                binary = Path(module_path).read_bytes()
            except OSError as exc:
                raise RuntimeAdapterSetupError(
                    f"OpenGL SPIR-V runtime artifact could not be read: {exc}",
                    details={
                        "target": "opengl",
                        "modulePath": str(module_path),
                        "reasonKind": "spirv-artifact-read-failed",
                    },
                ) from exc
            if len(binary) < _OPENGL_SPIRV_HEADER_BYTE_LENGTH or len(binary) % 4:
                raise _opengl_setup_error(
                    "OpenGL SPIR-V runtime artifact must contain a complete, "
                    "word-aligned header.",
                    "spirv-artifact-layout-invalid",
                    modulePath=str(module_path),
                    byteLength=len(binary),
                    minimumByteLength=_OPENGL_SPIRV_HEADER_BYTE_LENGTH,
                )
            if binary[:4] != b"\x03\x02#\x07":
                raise _opengl_setup_error(
                    "OpenGL SPIR-V runtime artifact has an invalid magic word.",
                    "spirv-artifact-magic-invalid",
                    modulePath=str(module_path),
                )
            return binary
        try:
            source = Path(module_path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise RuntimeAdapterSetupError(
                f"OpenGL runtime artifact could not be read: {exc}",
                details={"target": "opengl", "modulePath": str(module_path)},
            ) from exc
        if not source.strip():
            raise RuntimeAdapterSetupError(
                "OpenGL runtime artifact is empty.",
                details={"target": "opengl", "modulePath": str(module_path)},
            )
        return source

    def dispatch(
        self,
        adapter: Any,
        state: Any,
        request: NativeRuntimeDispatchRequest,
    ) -> dict[str, Mapping[str, Any]]:
        _ = adapter, state
        if request.entry_point not in (None, "main"):
            raise RuntimeExecutorUnavailable(
                "OpenGL compute artifacts expose the selected entry point as main; "
                f"got {request.entry_point!r}."
            )

        moderngl = self._load_moderngl()
        specialization_bindings, uniform_bindings = _partition_opengl_constants(
            request.constants
        )
        prepared_specializations = _prepare_opengl_specializations(
            specialization_bindings
        )
        shader_artifact = self._shader_artifact(request)
        prepared_buffers = _prepare_opengl_buffers(request.buffers)
        workgroup_count = _workgroup_count(request, target="OpenGL")
        if not prepared_buffers:
            raise RuntimeExecutorUnavailable(
                "OpenGL compute runtime requires at least one buffer resource."
            )

        context = None
        shader = None
        resources: list[tuple[_PreparedOpenGLBuffer, Any]] = []
        try:
            try:
                context, _backend = self._create_context(moderngl)
            except Exception as exc:
                raise RuntimeAdapterSetupError(
                    f"OpenGL compute context creation failed: {exc}",
                    details={
                        "target": "opengl",
                        "runtime": self.name,
                        "reasonKind": "context-creation-failed",
                    },
                ) from exc
            version_code = _opengl_context_version_code(context)
            if version_code < self.require_version:
                raise _opengl_setup_error(
                    "OpenGL compute requires context version "
                    f"{self.require_version}, got {version_code}.",
                    "opengl-version-unsupported",
                    contextBackend=_backend or "default",
                    requiredVersionCode=self.require_version,
                    versionCode=version_code,
                )
            try:
                if specialization_bindings:
                    if not isinstance(shader_artifact, bytes):
                        raise _opengl_setup_error(
                            "OpenGL specialization constants require a compiled SPIR-V runtime artifact.",
                            "spirv-artifact-required",
                            modulePath=str(request.module_path),
                        )
                    api = self._load_opengl_spirv_api(context)
                    shader = self._create_spirv_shader(
                        api,
                        shader_artifact,
                        prepared_specializations,
                        entry_point=request.entry_point or "main",
                    )
                    self._record_specialization_application(
                        state,
                        prepared_specializations,
                        uniform_count=len(uniform_bindings),
                        entry_point=request.entry_point or "main",
                        api=api,
                    )
                else:
                    if not isinstance(shader_artifact, str):
                        raise _opengl_setup_error(
                            "OpenGL source runtime dispatch requires a GLSL text artifact.",
                            "glsl-source-artifact-required",
                            modulePath=str(request.module_path),
                        )
                    shader = context.compute_shader(shader_artifact)
            except (_OpenGLSPIRVUnavailable, RuntimeAdapterSetupError) as exc:
                if isinstance(exc, _OpenGLSPIRVUnavailable):
                    raise RuntimeAdapterSetupError(
                        str(exc),
                        details={"target": "opengl", **exc.details},
                    ) from exc
                raise
            except Exception as exc:
                raise RuntimeAdapterSetupError(
                    f"OpenGL compute shader compilation or linking failed: {exc}",
                    details={
                        "target": "opengl",
                        "runtime": self.name,
                        "reasonKind": "shader-compilation-or-linking-failed",
                    },
                ) from exc

            for prepared in prepared_buffers:
                try:
                    buffer = self._create_buffer(context, prepared)
                except Exception as exc:
                    raise RuntimeAdapterSetupError(
                        f"OpenGL resource binding failed for {prepared.name!r}: {exc}",
                        details={
                            "target": "opengl",
                            "runtime": self.name,
                            "reasonKind": "resource-binding-failed",
                            "resource": prepared.name,
                            "binding": prepared.binding_index,
                        },
                    ) from exc
                resources.append((prepared, buffer))
            self._bind_constants(shader, uniform_bindings)
            try:
                shader.run(
                    group_x=workgroup_count[0],
                    group_y=workgroup_count[1],
                    group_z=workgroup_count[2],
                )
            except Exception as exc:
                raise RuntimeAdapterDispatchError(
                    f"OpenGL compute dispatch failed: {exc}",
                    details={
                        "target": "opengl",
                        "runtime": self.name,
                        "reasonKind": "dispatch-failed",
                    },
                ) from exc
            try:
                context.memory_barrier()
                finish = getattr(context, "finish", None)
                if callable(finish):
                    finish()
            except Exception as exc:
                raise RuntimeAdapterDispatchError(
                    f"OpenGL compute synchronization failed: {exc}",
                    details={
                        "target": "opengl",
                        "runtime": self.name,
                        "reasonKind": "synchronization-failed",
                    },
                ) from exc
            try:
                return self._read_outputs(resources)
            except RuntimeAdapterDispatchError:
                raise
            except Exception as exc:
                raise RuntimeAdapterDispatchError(
                    f"OpenGL compute output readback failed: {exc}",
                    details={
                        "target": "opengl",
                        "runtime": self.name,
                        "reasonKind": "readback-failed",
                    },
                ) from exc
        except (RuntimeAdapterSetupError, RuntimeExecutorUnavailable):
            raise
        except RuntimeAdapterDispatchError:
            raise
        except Exception as exc:
            raise RuntimeAdapterDispatchError(
                f"OpenGL compute dispatch failed: {exc}",
                details={"target": "opengl", "runtime": self.name},
            ) from exc
        finally:
            for _prepared, buffer in reversed(resources):
                _release_opengl_object(buffer)
            _release_opengl_object(shader)
            _release_opengl_object(context)

    def _load_moderngl(self) -> Any:
        try:
            return self._module_loader("moderngl")
        except Exception as exc:
            raise RuntimeExecutorUnavailable(
                "ModernGL is unavailable; install the optional 'moderngl' package "
                "to run OpenGL runtime fixtures."
            ) from exc

    def _load_opengl_spirv_api(self, context: Any) -> _OpenGLSPIRVAPI:
        version_code = int(getattr(context, "version_code", 0) or 0)
        extensions = {
            str(extension) for extension in getattr(context, "extensions", ()) or ()
        }
        extension_available = "GL_ARB_gl_spirv" in extensions
        if version_code < 460 and not extension_available:
            raise _OpenGLSPIRVUnavailable(
                "OpenGL SPIR-V specialization requires OpenGL 4.6 or the "
                "GL_ARB_gl_spirv extension.",
                details={
                    "reasonKind": "opengl-spirv-capability-unavailable",
                    "requiredExtension": "GL_ARB_gl_spirv",
                    "versionCode": version_code,
                    "extensionAvailable": False,
                },
            )

        try:
            gl = self._module_loader("OpenGL.GL")
        except Exception as exc:
            raise _OpenGLSPIRVUnavailable(
                "PyOpenGL is required to load and specialize OpenGL SPIR-V modules.",
                details={
                    "reasonKind": "dependency-unavailable",
                    "missingPythonModules": ["PyOpenGL"],
                    "requiredExtension": "GL_ARB_gl_spirv",
                },
            ) from exc
        try:
            gl_spirv = self._module_loader("OpenGL.GL.ARB.gl_spirv")
        except Exception:
            gl_spirv = None

        specialize_shader = None
        specialize_entry_point = None
        arb_specialize = getattr(gl_spirv, "glSpecializeShaderARB", None)
        if extension_available and _opengl_entry_point_available(arb_specialize):
            specialize_shader = arb_specialize
            specialize_entry_point = "glSpecializeShaderARB"
        core_specialize = getattr(gl, "glSpecializeShader", None)
        if (
            specialize_shader is None
            and version_code >= 460
            and _opengl_entry_point_available(core_specialize)
        ):
            specialize_shader = core_specialize
            specialize_entry_point = "glSpecializeShader"

        required_entry_points = (
            "glShaderBinary",
            "glCreateShader",
            "glGetShaderiv",
            "glGetShaderInfoLog",
            "glCreateProgram",
            "glAttachShader",
            "glLinkProgram",
            "glGetProgramiv",
            "glGetProgramInfoLog",
            "glUseProgram",
            "glDispatchCompute",
            "glDeleteProgram",
            "glDeleteShader",
        )
        missing_entry_points = [
            name
            for name in required_entry_points
            if not _opengl_entry_point_available(getattr(gl, name, None))
        ]
        if specialize_shader is None:
            missing_entry_points.append(
                "glSpecializeShaderARB"
                if extension_available and version_code < 460
                else "glSpecializeShader"
            )
        if missing_entry_points:
            raise _OpenGLSPIRVUnavailable(
                "OpenGL SPIR-V specialization entry points are unavailable: "
                + ", ".join(missing_entry_points)
                + ".",
                details={
                    "reasonKind": "opengl-spirv-entry-points-unavailable",
                    "requiredExtension": "GL_ARB_gl_spirv",
                    "versionCode": version_code,
                    "extensionAvailable": extension_available,
                    "missingEntryPoints": missing_entry_points,
                },
            )

        binary_format = getattr(gl, "GL_SHADER_BINARY_FORMAT_SPIR_V", None)
        if binary_format is None and gl_spirv is not None:
            binary_format = getattr(
                gl_spirv, "GL_SHADER_BINARY_FORMAT_SPIR_V_ARB", None
            )
        if binary_format is None:
            raise _OpenGLSPIRVUnavailable(
                "OpenGL SPIR-V shader binary format token is unavailable.",
                details={
                    "reasonKind": "opengl-spirv-token-unavailable",
                    "requiredExtension": "GL_ARB_gl_spirv",
                    "missingTokens": ["GL_SHADER_BINARY_FORMAT_SPIR_V"],
                },
            )
        return _OpenGLSPIRVAPI(
            gl=gl,
            specialize_shader=specialize_shader,
            specialize_entry_point=str(specialize_entry_point),
            binary_format=int(binary_format),
        )

    def _create_spirv_shader(
        self,
        api: _OpenGLSPIRVAPI,
        binary: bytes,
        specializations: Sequence[_PreparedOpenGLSpecialization],
        *,
        entry_point: str,
    ) -> _OpenGLSPIRVComputeShader:
        gl = api.gl
        shader = 0
        program = 0
        try:
            shader = int(gl.glCreateShader(gl.GL_COMPUTE_SHADER))
            if not shader:
                raise _opengl_setup_error(
                    "OpenGL could not create a compute shader for SPIR-V.",
                    "spirv-shader-creation-failed",
                )
            shader_handles = (ctypes.c_uint * 1)(shader)
            binary_buffer = (ctypes.c_ubyte * len(binary)).from_buffer_copy(binary)
            try:
                gl.glShaderBinary(
                    1,
                    shader_handles,
                    api.binary_format,
                    binary_buffer,
                    len(binary),
                )
            except Exception as exc:
                raise _opengl_setup_error(
                    f"OpenGL SPIR-V shader binary loading failed: {exc}",
                    "spirv-shader-binary-load-failed",
                ) from exc

            constant_ids = (ctypes.c_uint * len(specializations))(
                *(item.constant_id for item in specializations)
            )
            constant_values = (ctypes.c_uint * len(specializations))(
                *(item.encoded_word for item in specializations)
            )
            try:
                api.specialize_shader(
                    shader,
                    entry_point.encode("utf-8"),
                    len(specializations),
                    constant_ids if specializations else None,
                    constant_values if specializations else None,
                )
            except Exception as exc:
                raise _opengl_setup_error(
                    f"OpenGL SPIR-V shader specialization failed: {exc}",
                    "shader-specialization-failed",
                    entryPoint=entry_point,
                    specializationConstants=[
                        item.to_json() for item in specializations
                    ],
                ) from exc
            if not _opengl_status(gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)):
                raise _opengl_setup_error(
                    "OpenGL SPIR-V shader specialization failed: "
                    + _opengl_info_log(gl.glGetShaderInfoLog(shader)),
                    "shader-specialization-failed",
                    entryPoint=entry_point,
                    specializationConstants=[
                        item.to_json() for item in specializations
                    ],
                )

            program = int(gl.glCreateProgram())
            if not program:
                raise _opengl_setup_error(
                    "OpenGL could not create a program for specialized SPIR-V.",
                    "spirv-program-creation-failed",
                )
            gl.glAttachShader(program, shader)
            gl.glLinkProgram(program)
            if not _opengl_status(gl.glGetProgramiv(program, gl.GL_LINK_STATUS)):
                raise _opengl_setup_error(
                    "OpenGL specialized SPIR-V program linking failed: "
                    + _opengl_info_log(gl.glGetProgramInfoLog(program)),
                    "spirv-program-linking-failed",
                    entryPoint=entry_point,
                )
            return _OpenGLSPIRVComputeShader(
                api,
                shader=shader,
                program=program,
            )
        except Exception:
            if program:
                try:
                    gl.glDeleteProgram(program)
                except Exception:
                    pass
            if shader:
                try:
                    gl.glDeleteShader(shader)
                except Exception:
                    pass
            raise

    def _record_specialization_application(
        self,
        state: Any,
        specializations: Sequence[_PreparedOpenGLSpecialization],
        *,
        uniform_count: int,
        entry_point: str,
        api: _OpenGLSPIRVAPI,
    ) -> None:
        if state is None:
            return
        details = {
            "mode": "spirv-specialization",
            "entryPoint": entry_point,
            "specializationEntryPoint": api.specialize_entry_point,
            "appliedConstantCount": len(specializations),
            "uniformConstantCount": uniform_count,
            "appliedConstants": [item.to_json() for item in specializations],
        }
        state_details = getattr(state, "details", None)
        if isinstance(state_details, dict):
            state_details["openglSpecialization"] = details
        record_step = getattr(state, "record_step", None)
        if callable(record_step):
            record_step(
                "specialize",
                "specialize-opengl-spirv",
                details={
                    "target": "opengl",
                    "entryPoint": entry_point,
                    "specializationEntryPoint": api.specialize_entry_point,
                    "appliedConstantIds": [
                        item.constant_id for item in specializations
                    ],
                    "appliedConstantCount": len(specializations),
                    "uniformConstantCount": uniform_count,
                },
            )

    def _create_context(self, moderngl: Any) -> tuple[Any, str | None]:
        if self._context_factory is not None:
            return self._context_factory(moderngl), None

        failures = []
        for backend in self.context_backends:
            kwargs = {"require": self.require_version}
            if backend is not None:
                kwargs["backend"] = backend
            try:
                return moderngl.create_standalone_context(**kwargs), backend
            except Exception as exc:
                failures.append(f"{backend or 'default'}: {exc}")
        raise RuntimeExecutorUnavailable(
            "No headless OpenGL compute context could be created ("
            + "; ".join(failures)
            + ")."
        )

    def _shader_artifact(self, request: NativeRuntimeDispatchRequest) -> str | bytes:
        if isinstance(request.loaded_artifact, (str, bytes)):
            return request.loaded_artifact
        return self.load_artifact(None, None, request.module_path)

    def _create_buffer(self, context: Any, prepared: _PreparedOpenGLBuffer) -> Any:
        if prepared.payload:
            payload = prepared.payload.ljust(prepared.allocation_size, b"\x00")
            buffer = context.buffer(payload)
        else:
            buffer = context.buffer(reserve=max(prepared.allocation_size, 1))
        try:
            if prepared.resource_kind in {"constant-buffer", "uniform"}:
                buffer.bind_to_uniform_block(prepared.binding_index)
            else:
                buffer.bind_to_storage_buffer(prepared.binding_index)
        except Exception:
            _release_opengl_object(buffer)
            raise
        return buffer

    def _bind_constants(self, shader: Any, constants: Mapping[str, Any]) -> None:
        for name, binding in constants.items():
            if binding.value is None:
                raise RuntimeAdapterSetupError(
                    f"OpenGL runtime constant {name!r} has no bound value.",
                    details={
                        "target": "opengl",
                        "runtime": self.name,
                        "reasonKind": "constant-value-missing",
                        "constant": name,
                    },
                )
            set_uniform = getattr(shader, "set_uniform", None)
            if callable(set_uniform):
                set_uniform(name, binding)
                continue
            try:
                uniform = shader[name]
            except (KeyError, TypeError) as exc:
                raise RuntimeAdapterSetupError(
                    f"OpenGL runtime constant {name!r} is not an active uniform.",
                    details={
                        "target": "opengl",
                        "runtime": self.name,
                        "reasonKind": "active-uniform-missing",
                        "constant": name,
                    },
                ) from exc
            value = binding.value
            if isinstance(value, list):
                value = tuple(value)
            try:
                uniform.value = value
            except Exception as exc:
                raise RuntimeAdapterSetupError(
                    f"OpenGL runtime constant {name!r} could not be bound: {exc}",
                    details={
                        "target": "opengl",
                        "runtime": self.name,
                        "reasonKind": "constant-binding-failed",
                        "constant": name,
                    },
                ) from exc

    def _read_outputs(
        self,
        resources: Sequence[tuple[_PreparedOpenGLBuffer, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        outputs: dict[str, Mapping[str, Any]] = {}
        for prepared, buffer in resources:
            if prepared.source != "expectedOutput":
                continue
            payload = bytes(buffer.read(size=prepared.size))
            if len(payload) != prepared.size:
                raise RuntimeAdapterDispatchError(
                    f"OpenGL output readback for {prepared.name!r} returned "
                    f"{len(payload)} bytes; expected {prepared.size}.",
                    details={
                        "target": "opengl",
                        "runtime": self.name,
                        "reasonKind": "readback-size-mismatch",
                        "resource": prepared.name,
                        "expectedByteLength": prepared.size,
                        "actualByteLength": len(payload),
                    },
                )
            outputs[prepared.output_name or prepared.name] = {
                "dtype": prepared.dtype,
                "shape": list(prepared.shape),
                "values": _unpack_values(payload, prepared.dtype, target="OpenGL"),
            }
        return outputs


class VulkanComputeRuntime:
    """Reference Vulkan compute runtime for simple storage-buffer fixtures.

    The driver is optional. It imports the Python ``vulkan`` binding lazily and
    reports structured unavailability when the binding, Vulkan loader, or a
    compute-capable device is unavailable.
    """

    name = "vulkan-compute-runtime"

    def __init__(
        self,
        *,
        module_loader: Any | None = None,
        application_name: str = "CrossTL Runtime Verification",
        timeout_ns: int = 10_000_000_000,
    ):
        self._module_loader = module_loader or importlib.import_module
        self.application_name = application_name
        self.timeout_ns = int(timeout_ns)

    def is_available(
        self,
        adapter: Any,
        request: RuntimeExecutionRequest,
    ) -> RuntimeExecutorAvailability:
        _ = adapter, request
        try:
            vk = self._load_vulkan()
        except RuntimeExecutorUnavailable as exc:
            return RuntimeExecutorAvailability(
                False,
                reason=str(exc),
                details={
                    "reasonKind": "dependency-unavailable",
                    "target": "vulkan",
                    "missingPythonModules": ["vulkan"],
                },
            )

        instance = None
        try:
            instance = self._create_instance(vk)
            physical_device, queue_family = self._select_compute_device(vk, instance)
        except Exception as exc:  # pragma: no cover - depends on local Vulkan loader
            return RuntimeExecutorAvailability(
                False,
                reason=f"Vulkan runtime is unavailable: {exc}",
                details={
                    "reasonKind": "vulkan-runtime-unavailable",
                    "target": "vulkan",
                    "error": str(exc),
                },
            )
        finally:
            if instance is not None:
                self._destroy_instance(vk, instance)

        return RuntimeExecutorAvailability(
            True,
            details={
                "reasonKind": "available",
                "target": "vulkan",
                "runtime": self.name,
                "physicalDevice": str(physical_device),
                "queueFamilyIndex": queue_family,
            },
        )

    def load_artifact(self, adapter: Any, state: Any, module_path: Path) -> bytes:
        _ = adapter, state
        try:
            code = Path(module_path).read_bytes()
        except OSError as exc:
            raise RuntimeAdapterSetupError(
                f"Vulkan runtime artifact could not be read: {exc}",
                details={"target": "vulkan", "modulePath": str(module_path)},
            ) from exc
        if len(code) % 4:
            raise RuntimeAdapterSetupError(
                "Vulkan runtime artifacts must contain word-aligned SPIR-V bytes.",
                details={
                    "target": "vulkan",
                    "modulePath": str(module_path),
                    "byteLength": len(code),
                },
            )
        return code

    def dispatch(
        self,
        adapter: Any,
        state: Any,
        request: NativeRuntimeDispatchRequest,
    ) -> dict[str, Mapping[str, Any]]:
        _ = adapter, state
        vk = self._load_vulkan()
        shader_code = self._shader_code(request)
        prepared_buffers = _prepare_vulkan_buffers(request.buffers)
        workgroup_count = _workgroup_count(request)
        if not prepared_buffers:
            raise RuntimeExecutorUnavailable(
                "Vulkan compute runtime requires at least one storage buffer."
            )

        context = _VulkanDispatchContext(
            vk=vk,
            runtime=self,
            shader_code=shader_code,
            entry_point=request.entry_point or "main",
            buffers=prepared_buffers,
            workgroup_count=workgroup_count,
        )
        return context.run()

    def _load_vulkan(self) -> Any:
        try:
            return self._module_loader("vulkan")
        except Exception as exc:
            raise RuntimeExecutorUnavailable(
                "Python Vulkan bindings are unavailable; install the optional "
                "'vulkan' package to run Vulkan runtime fixtures."
            ) from exc

    def _shader_code(self, request: NativeRuntimeDispatchRequest) -> bytes:
        if isinstance(request.loaded_artifact, bytes):
            return request.loaded_artifact
        if isinstance(request.loaded_artifact, bytearray):
            return bytes(request.loaded_artifact)
        return self.load_artifact(None, None, request.module_path)

    def _create_instance(self, vk: Any) -> Any:
        application_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=self.application_name,
            applicationVersion=1,
            pEngineName="CrossTL",
            engineVersion=1,
            apiVersion=getattr(vk, "VK_API_VERSION_1_1", vk.VK_API_VERSION_1_0),
        )
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=application_info,
        )
        return vk.vkCreateInstance(create_info, None)

    def _destroy_instance(self, vk: Any, instance: Any) -> None:
        destroy = getattr(vk, "vkDestroyInstance", None)
        if callable(destroy):
            destroy(instance, None)

    def _select_compute_device(self, vk: Any, instance: Any) -> tuple[Any, int]:
        devices = vk.vkEnumeratePhysicalDevices(instance)
        for physical_device in devices:
            families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
            for index, family in enumerate(families):
                if int(family.queueFlags) & int(vk.VK_QUEUE_COMPUTE_BIT):
                    return physical_device, index
        raise RuntimeExecutorUnavailable(
            "No Vulkan physical device exposes a compute queue family."
        )


class _VulkanDispatchContext:
    def __init__(
        self,
        *,
        vk: Any,
        runtime: VulkanComputeRuntime,
        shader_code: bytes,
        entry_point: str,
        buffers: Sequence[_PreparedVulkanBuffer],
        workgroup_count: tuple[int, int, int],
    ):
        self.vk = vk
        self.runtime = runtime
        self.shader_code = shader_code
        self.entry_point = entry_point
        self.buffers = tuple(buffers)
        self.workgroup_count = workgroup_count
        self.instance = None
        self.physical_device = None
        self.queue_family = None
        self.device = None
        self.queue = None
        self.shader_module = None
        self.descriptor_set_layout = None
        self.pipeline_layout = None
        self.pipeline = None
        self.descriptor_pool = None
        self.descriptor_set = None
        self.command_pool = None
        self.command_buffer = None
        self.fence = None
        self.resources: list[_VulkanBufferResource] = []

    def run(self) -> dict[str, Mapping[str, Any]]:
        try:
            self._create_device_context()
            self._create_shader_module()
            self._create_buffers()
            self._create_descriptors()
            self._create_pipeline()
            self._record_and_submit()
            return self._read_outputs()
        except RuntimeExecutorUnavailable:
            raise
        except RuntimeAdapterDispatchError:
            raise
        except Exception as exc:  # pragma: no cover - depends on local Vulkan driver
            raise RuntimeAdapterDispatchError(
                f"Vulkan compute dispatch failed: {exc}",
                details={"target": "vulkan", "runtime": self.runtime.name},
            ) from exc
        finally:
            self._cleanup()

    def _create_device_context(self) -> None:
        vk = self.vk
        self.instance = self.runtime._create_instance(vk)
        self.physical_device, self.queue_family = self.runtime._select_compute_device(
            vk, self.instance
        )
        priority = [1.0]
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.queue_family,
            queueCount=1,
            pQueuePriorities=priority,
        )
        device_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
        )
        self.device = vk.vkCreateDevice(self.physical_device, device_info, None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.queue_family, 0)

    def _create_shader_module(self) -> None:
        vk = self.vk
        create_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(self.shader_code),
            pCode=self.shader_code,
        )
        self.shader_module = vk.vkCreateShaderModule(self.device, create_info, None)

    def _create_buffers(self) -> None:
        for prepared in self.buffers:
            self.resources.append(self._create_buffer(prepared))

    def _create_buffer(self, prepared: _PreparedVulkanBuffer) -> _VulkanBufferResource:
        vk = self.vk
        create_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=max(prepared.size, 1),
            usage=_vulkan_buffer_usage(vk, prepared.resource_kind),
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buffer_handle = vk.vkCreateBuffer(self.device, create_info, None)
        requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer_handle)
        memory_type = self._find_memory_type(
            int(requirements.memoryTypeBits),
            int(vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            | int(vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        )
        allocation = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=requirements.size,
            memoryTypeIndex=memory_type,
        )
        memory = vk.vkAllocateMemory(self.device, allocation, None)
        vk.vkBindBufferMemory(self.device, buffer_handle, memory, 0)
        self._write_memory(memory, prepared.payload)
        return _VulkanBufferResource(
            prepared=prepared, buffer=buffer_handle, memory=memory
        )

    def _find_memory_type(self, type_bits: int, required_flags: int) -> int:
        properties = self.vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        for index in range(int(properties.memoryTypeCount)):
            if not (type_bits & (1 << index)):
                continue
            flags = int(properties.memoryTypes[index].propertyFlags)
            if flags & required_flags == required_flags:
                return index
        raise RuntimeExecutorUnavailable(
            "No host-visible Vulkan memory type is available for runtime buffers."
        )

    def _write_memory(self, memory: Any, payload: bytes) -> None:
        if not payload:
            return
        pointer = self.vk.vkMapMemory(self.device, memory, 0, len(payload), 0)
        try:
            _write_mapped_memory(pointer, payload)
        finally:
            self.vk.vkUnmapMemory(self.device, memory)

    def _read_memory(self, memory: Any, size: int) -> bytes:
        pointer = self.vk.vkMapMemory(self.device, memory, 0, size, 0)
        try:
            return _read_mapped_memory(pointer, size)
        finally:
            self.vk.vkUnmapMemory(self.device, memory)

    def _create_descriptors(self) -> None:
        set_indices = {buffer.set_index for buffer in self.buffers}
        if set_indices != {0}:
            raise RuntimeExecutorUnavailable(
                "Vulkan compute runtime currently supports a single descriptor set 0."
            )
        vk = self.vk
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=buffer.binding_index,
                descriptorType=_vulkan_descriptor_type(vk, buffer.resource_kind),
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for buffer in self.buffers
        ]
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        self.descriptor_set_layout = vk.vkCreateDescriptorSetLayout(
            self.device, layout_info, None
        )
        descriptor_counts: dict[Any, int] = {}
        for buffer in self.buffers:
            descriptor_type = _vulkan_descriptor_type(vk, buffer.resource_kind)
            descriptor_counts[descriptor_type] = (
                descriptor_counts.get(descriptor_type, 0) + 1
            )
        pool_sizes = [
            vk.VkDescriptorPoolSize(type=descriptor_type, descriptorCount=count)
            for descriptor_type, count in descriptor_counts.items()
        ]
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=1,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        self.descriptor_pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)
        allocate_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.descriptor_set_layout],
        )
        self.descriptor_set = vk.vkAllocateDescriptorSets(self.device, allocate_info)[0]
        writes = []
        for resource in self.resources:
            buffer_info = vk.VkDescriptorBufferInfo(
                buffer=resource.buffer,
                offset=0,
                range=max(resource.prepared.size, 1),
            )
            writes.append(
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=self.descriptor_set,
                    dstBinding=resource.prepared.binding_index,
                    descriptorCount=1,
                    descriptorType=_vulkan_descriptor_type(
                        vk, resource.prepared.resource_kind
                    ),
                    pBufferInfo=[buffer_info],
                )
            )
        vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)

    def _create_pipeline(self) -> None:
        vk = self.vk
        layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.descriptor_set_layout],
        )
        self.pipeline_layout = vk.vkCreatePipelineLayout(self.device, layout_info, None)
        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.shader_module,
            pName=self.entry_point,
        )
        pipeline_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=self.pipeline_layout,
        )
        pipelines = vk.vkCreateComputePipelines(
            self.device,
            getattr(vk, "VK_NULL_HANDLE", None),
            1,
            [pipeline_info],
            None,
        )
        self.pipeline = _first_vulkan_handle(pipelines)

    def _record_and_submit(self) -> None:
        vk = self.vk
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.queue_family,
        )
        self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)
        allocate_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self.command_buffer = vk.vkAllocateCommandBuffers(self.device, allocate_info)[0]
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        )
        vk.vkBeginCommandBuffer(self.command_buffer, begin_info)
        vk.vkCmdBindPipeline(
            self.command_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline,
        )
        vk.vkCmdBindDescriptorSets(
            self.command_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_layout,
            0,
            1,
            [self.descriptor_set],
            0,
            None,
        )
        vk.vkCmdDispatch(self.command_buffer, *self.workgroup_count)
        vk.vkEndCommandBuffer(self.command_buffer)
        fence_info = vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        self.fence = vk.vkCreateFence(self.device, fence_info, None)
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffer],
        )
        vk.vkQueueSubmit(self.queue, 1, [submit_info], self.fence)
        vk.vkWaitForFences(self.device, 1, [self.fence], True, self.runtime.timeout_ns)

    def _read_outputs(self) -> dict[str, Mapping[str, Any]]:
        outputs: dict[str, Mapping[str, Any]] = {}
        for resource in self.resources:
            prepared = resource.prepared
            if prepared.source != "expectedOutput":
                continue
            payload = self._read_memory(resource.memory, prepared.size)
            outputs[prepared.output_name or prepared.name] = {
                "dtype": prepared.dtype,
                "shape": list(prepared.shape),
                "values": _unpack_values(payload, prepared.dtype),
            }
        return outputs

    def _cleanup(self) -> None:
        vk = self.vk
        device = self.device
        if device is not None:
            wait_idle = getattr(vk, "vkDeviceWaitIdle", None)
            if callable(wait_idle):
                try:
                    wait_idle(device)
                except Exception:
                    pass
        _vk_destroy(vk, "vkDestroyFence", device, self.fence)
        _vk_destroy(vk, "vkDestroyCommandPool", device, self.command_pool)
        _vk_destroy(vk, "vkDestroyPipeline", device, self.pipeline)
        _vk_destroy(vk, "vkDestroyPipelineLayout", device, self.pipeline_layout)
        _vk_destroy(vk, "vkDestroyDescriptorPool", device, self.descriptor_pool)
        _vk_destroy(
            vk,
            "vkDestroyDescriptorSetLayout",
            device,
            self.descriptor_set_layout,
        )
        _vk_destroy(vk, "vkDestroyShaderModule", device, self.shader_module)
        for resource in reversed(self.resources):
            _vk_destroy(vk, "vkDestroyBuffer", device, resource.buffer)
            free_memory = getattr(vk, "vkFreeMemory", None)
            if (
                callable(free_memory)
                and device is not None
                and resource.memory is not None
            ):
                free_memory(device, resource.memory, None)
        _vk_destroy(vk, "vkDestroyDevice", None, device)
        if self.instance is not None:
            self.runtime._destroy_instance(vk, self.instance)


def _vk_destroy(vk: Any, name: str, owner: Any, handle: Any) -> None:
    if handle is None:
        return
    destroy = getattr(vk, name, None)
    if not callable(destroy):
        return
    if owner is None:
        destroy(handle, None)
    else:
        destroy(owner, handle, None)


def _first_vulkan_handle(handles: Any) -> Any:
    if isinstance(handles, (str, bytes, bytearray)):
        return handles
    try:
        return handles[0]
    except (IndexError, TypeError, KeyError):
        return handles


def _write_mapped_memory(mapped: Any, payload: bytes) -> None:
    try:
        view = memoryview(mapped)
    except TypeError:
        ctypes.memmove(mapped, payload, len(payload))
        return

    try:
        byte_view = view.cast("B") if view.format != "B" else view
        byte_view[: len(payload)] = payload
    finally:
        view.release()


def _read_mapped_memory(mapped: Any, size: int) -> bytes:
    try:
        view = memoryview(mapped)
    except TypeError:
        return ctypes.string_at(mapped, size)

    try:
        byte_view = view.cast("B") if view.format != "B" else view
        return byte_view[:size].tobytes()
    finally:
        view.release()


def _compushady_backend_name(backend: Any) -> str:
    value = getattr(backend, "name", None) or getattr(backend, "__name__", None)
    if value is None and isinstance(backend, str):
        value = backend
    return str(value or "").strip().lower().rsplit(".", 1)[-1]


def _release_directx_object(value: Any) -> None:
    if value is None:
        return
    for method_name in ("release", "close"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
            return


def _prepare_directx_buffers(
    bindings: Mapping[str, NativeRuntimeBufferBinding],
) -> tuple[_PreparedDirectXBuffer, ...]:
    prepared = []
    for name, binding in bindings.items():
        resource = binding.binding
        namespace = _directx_resource_namespace(name, binding)
        binding_value = resource.binding
        if binding_value is None:
            binding_value = resource.index
        if binding_value is None:
            raise _directx_setup_error(
                f"DirectX compute runtime requires an explicit register for {name}.",
                "register-missing",
                resource=name,
                namespace=namespace,
            )
        binding_index = _directx_int_field(
            binding_value,
            field_name="binding",
            resource=name,
            register_prefix=namespace[0],
        )
        set_index = _directx_int_field(
            resource.set,
            field_name="set",
            resource=name,
            default=0,
        )
        if set_index != 0:
            raise _directx_setup_error(
                "DirectX compute runtime supports register space zero only.",
                "unsupported-register-space",
                resource=name,
                namespace=namespace,
                binding=binding_index,
                registerSpace=set_index,
            )
        if namespace == "cbv" and binding.source == "expectedOutput":
            raise _directx_setup_error(
                "DirectX constant buffers cannot be runtime output resources.",
                "unsupported-output-resource",
                resource=name,
                namespace=namespace,
                binding=binding_index,
            )

        dtype = _normalize_directx_dtype(binding.dtype, resource=name)
        try:
            shape = tuple(int(value) for value in binding.shape)
        except (TypeError, ValueError) as exc:
            raise _directx_setup_error(
                f"DirectX runtime buffer {name!r} has an invalid shape.",
                "invalid-buffer-shape",
                resource=name,
            ) from exc
        if any(value < 0 for value in shape):
            raise _directx_setup_error(
                f"DirectX runtime buffer {name!r} has a negative shape dimension.",
                "invalid-buffer-shape",
                resource=name,
                shape=list(shape),
            )
        element_count = (
            math.prod(shape) if shape else len(_flatten_values(binding.value))
        )
        if element_count <= 0:
            raise _directx_setup_error(
                f"DirectX runtime buffer {name!r} has no addressable elements.",
                "buffer-size-missing",
                resource=name,
                shape=list(shape),
            )
        if binding.source == "expectedOutput":
            payload = b"\x00" * (element_count * _dtype_size(dtype))
        else:
            try:
                payload = _pack_values(
                    binding.value,
                    dtype,
                    expected_count=element_count,
                    target="DirectX",
                )
            except (RuntimeExecutorUnavailable, struct.error) as exc:
                raise _directx_setup_error(
                    f"DirectX runtime buffer {name!r} could not be packed: {exc}",
                    "buffer-packing-failed",
                    resource=name,
                    dtype=dtype,
                    shape=list(shape),
                ) from exc

        stride = _directx_buffer_stride(binding, namespace, dtype, len(payload))
        if namespace == "cbv":
            block_size = _scalar_block_size(
                binding,
                target="directx",
                dtype=dtype,
                payload_size=len(payload),
            )
            allocation_size = _align_to(max(len(payload), block_size), 256)
        else:
            allocation_size = len(payload)
        prepared.append(
            _PreparedDirectXBuffer(
                name=name,
                namespace=namespace,
                binding_index=binding_index,
                dtype=dtype,
                shape=shape,
                source=binding.source,
                output_name=_runtime_value_name(binding),
                payload=payload,
                allocation_size=allocation_size,
                stride=stride,
            )
        )
    return tuple(
        sorted(
            prepared,
            key=lambda item: (
                {"cbv": 0, "srv": 1, "uav": 2}[item.namespace],
                item.binding_index,
            ),
        )
    )


def _prepare_directx_constants(
    bindings: Mapping[str, Any],
) -> tuple[_PreparedDirectXBuffer, ...]:
    grouped: dict[int, list[_DirectXConstantValue]] = {}
    for name, binding in bindings.items():
        metadata = _directx_constant_metadata(binding)
        mechanism = str(metadata.get("mechanism") or "").strip().lower()
        constant_kind = str(binding.constant.kind or "").strip().lower()
        if mechanism in {"compiled", "compiled-literal", "static"} or (
            not mechanism
            and constant_kind
            in {"scalar-constant", "compile-time-constant", "static-constant"}
        ):
            _validate_directx_compiled_constant(name, binding)
            continue
        if not mechanism and constant_kind in {"constant-buffer", "cbv", "uniform"}:
            mechanism = "constant-buffer"
        if mechanism not in {"constant-buffer", "cbv"}:
            raise _directx_setup_error(
                f"DirectX runtime constant {name!r} has no supported binding mechanism.",
                "unsupported-constant-binding",
                constant=name,
                constantKind=binding.constant.kind,
            )

        binding_value = metadata.get("binding", metadata.get("register"))
        if binding_value is None:
            raise _directx_setup_error(
                f"DirectX runtime constant {name!r} requires a CBV register.",
                "constant-register-missing",
                constant=name,
            )
        binding_index = _directx_int_field(
            binding_value,
            field_name="binding",
            resource=name,
            register_prefix="b",
        )
        set_index = _directx_int_field(
            metadata.get("set", metadata.get("space")),
            field_name="set",
            resource=name,
            default=0,
        )
        if set_index != 0:
            raise _directx_setup_error(
                "DirectX runtime constants support register space zero only.",
                "unsupported-register-space",
                constant=name,
                namespace="cbv",
                binding=binding_index,
                registerSpace=set_index,
            )
        byte_offset = _directx_int_field(
            metadata.get("byteOffset", metadata.get("offset")),
            field_name="byteOffset",
            resource=name,
            default=0,
        )
        if byte_offset % 4:
            raise _directx_setup_error(
                f"DirectX runtime constant {name!r} must be four-byte aligned.",
                "constant-offset-invalid",
                constant=name,
                byteOffset=byte_offset,
            )
        dtype = _normalize_directx_dtype(binding.constant.dtype, resource=name)
        values = _flatten_values(binding.value)
        if not values:
            raise _directx_setup_error(
                f"DirectX runtime constant {name!r} has no bound value.",
                "constant-value-missing",
                constant=name,
            )
        try:
            payload = _pack_values(
                binding.value,
                dtype,
                expected_count=len(values),
                target="DirectX",
            )
        except (RuntimeExecutorUnavailable, struct.error) as exc:
            raise _directx_setup_error(
                f"DirectX runtime constant {name!r} could not be packed: {exc}",
                "constant-packing-failed",
                constant=name,
                dtype=dtype,
            ) from exc
        grouped.setdefault(binding_index, []).append(
            _DirectXConstantValue(
                name=name,
                binding_index=binding_index,
                byte_offset=byte_offset,
                dtype=dtype,
                payload=payload,
            )
        )

    prepared = []
    for binding_index, constants in sorted(grouped.items()):
        ranges: list[tuple[int, int, str]] = []
        payload_size = max(
            value.byte_offset + len(value.payload) for value in constants
        )
        payload = bytearray(payload_size)
        for value in sorted(constants, key=lambda item: (item.byte_offset, item.name)):
            start = value.byte_offset
            end = start + len(value.payload)
            overlap = next(
                (
                    existing_name
                    for existing_start, existing_end, existing_name in ranges
                    if start < existing_end and end > existing_start
                ),
                None,
            )
            if overlap is not None:
                raise _directx_setup_error(
                    "DirectX runtime constants have overlapping CBV byte ranges.",
                    "constant-layout-overlap",
                    namespace="cbv",
                    binding=binding_index,
                    constant=value.name,
                    overlaps=overlap,
                )
            payload[start:end] = value.payload
            ranges.append((start, end, value.name))
        names = ", ".join(value.name for value in constants)
        prepared.append(
            _PreparedDirectXBuffer(
                name=f"constants[{names}]",
                namespace="cbv",
                binding_index=binding_index,
                dtype="uint32",
                shape=(),
                source="constant",
                output_name=None,
                payload=bytes(payload),
                allocation_size=_align_to(payload_size, 256),
            )
        )
    return tuple(prepared)


def _validate_directx_register_layout(
    prepared: Sequence[_PreparedDirectXBuffer],
) -> tuple[_PreparedDirectXBuffer, ...]:
    result = []
    for namespace in ("cbv", "srv", "uav"):
        resources = sorted(
            (item for item in prepared if item.namespace == namespace),
            key=lambda item: item.binding_index,
        )
        indices = [item.binding_index for item in resources]
        if len(indices) != len(set(indices)):
            duplicates = sorted(
                {index for index in indices if indices.count(index) > 1}
            )
            raise _directx_setup_error(
                f"DirectX runtime has duplicate {namespace.upper()} registers.",
                "duplicate-register-binding",
                namespace=namespace,
                bindings=indices,
                duplicateBindings=duplicates,
            )
        expected = list(range(len(indices)))
        if indices != expected:
            raise _directx_setup_error(
                "compushady requires contiguous DirectX registers starting at zero.",
                "sparse-register-layout",
                namespace=namespace,
                bindings=indices,
                expectedBindings=expected,
            )
        result.extend(resources)
    return tuple(result)


def _complete_directx_register_layout(
    prepared: Sequence[_PreparedDirectXBuffer],
) -> tuple[_PreparedDirectXBuffer, ...]:
    completed = list(prepared)
    for namespace in ("cbv", "srv", "uav"):
        resources = [item for item in prepared if item.namespace == namespace]
        indices = [item.binding_index for item in resources]
        if len(indices) != len(set(indices)):
            continue
        max_binding = max(indices, default=-1)
        descriptor_count = max_binding + 1
        if descriptor_count > _DIRECTX_MAX_PADDED_DESCRIPTOR_COUNT:
            raise _directx_setup_error(
                "DirectX sparse register layout exceeds the descriptor padding limit.",
                "descriptor-padding-limit-exceeded",
                namespace=namespace,
                bindings=sorted(indices),
                requiredDescriptorCount=descriptor_count,
                maxDescriptorCount=_DIRECTX_MAX_PADDED_DESCRIPTOR_COUNT,
            )
        bound_indices = set(indices)
        for binding_index in range(max_binding):
            if binding_index in bound_indices:
                continue
            allocation_size = 256 if namespace == "cbv" else 4
            completed.append(
                _PreparedDirectXBuffer(
                    name=f"__crosstl_descriptor_gap_{namespace}{binding_index}",
                    namespace=namespace,
                    binding_index=binding_index,
                    dtype="uint32",
                    shape=(1,),
                    source="descriptor-gap",
                    output_name=None,
                    payload=b"\x00" * 4,
                    allocation_size=allocation_size,
                    stride=0 if namespace == "cbv" else 4,
                )
            )
    return tuple(completed)


def _directx_resource_namespace(
    name: str,
    binding: NativeRuntimeBufferBinding,
) -> str:
    resource = binding.binding
    kind = str(resource.kind or "buffer").strip().lower().replace("_", "-")
    if kind in {"constant-buffer", "constantbuffer", "uniform", "cbv"}:
        return "cbv"
    if kind in {"srv", "shader-resource", "read-only-buffer"}:
        namespace = "srv"
    elif kind in {"uav", "unordered-access", "read-write-buffer"}:
        namespace = "uav"
    elif kind in {"buffer", "storage-buffer"}:
        namespace = ""
    else:
        raise _directx_setup_error(
            f"DirectX compute runtime supports buffer resources only: {name}.",
            "unsupported-resource-kind",
            resource=name,
            resourceKind=resource.kind,
        )

    type_name = str(resource.type_name or "").strip()
    normalized_type = re.sub(r"\s+", "", type_name).lower()
    type_namespace = None
    if normalized_type:
        if any(
            marker in normalized_type
            for marker in (
                "rwstructuredbuffer",
                "rwbyteaddressbuffer",
                "rwbuffer<",
                "appendstructuredbuffer",
                "consumestructuredbuffer",
                "rasterizerordered",
            )
        ):
            type_namespace = "uav"
        elif any(
            marker in normalized_type
            for marker in ("structuredbuffer", "byteaddressbuffer", "buffer<")
        ):
            type_namespace = "srv"

    access = str(resource.access or "").strip().lower().replace("-", "_")
    access_namespace = None
    if access in {"write", "read_write", "readwrite"}:
        access_namespace = "uav"
    elif access in {"read", "read_only", "readonly"}:
        access_namespace = "srv"

    inferred = namespace or type_namespace or access_namespace
    if binding.source == "expectedOutput":
        if inferred == "srv":
            raise _directx_setup_error(
                f"DirectX output resource {name!r} is reflected as read-only.",
                "resource-access-mismatch",
                resource=name,
                resourceKind=resource.kind,
                access=resource.access,
                type=resource.type_name,
            )
        return "uav"
    return inferred or "srv"


def _directx_buffer_stride(
    binding: NativeRuntimeBufferBinding,
    namespace: str,
    dtype: str,
    payload_size: int,
) -> int:
    resource = binding.binding
    metadata = {**dict(resource.metadata), **dict(binding.metadata)}
    type_name = str(resource.type_name or "").strip()
    normalized_type = re.sub(r"\s+", "", type_name).lower()
    if namespace != "cbv" and (
        "byteaddressbuffer" in normalized_type
        or ("buffer<" in normalized_type and "structuredbuffer<" not in normalized_type)
    ):
        raise _directx_setup_error(
            f"DirectX buffer view is not supported for {binding.name!r}.",
            "unsupported-buffer-view",
            resource=binding.name,
            type=resource.type_name,
        )
    explicit_stride = next(
        (
            metadata[key]
            for key in ("byteStride", "elementStride", "stride")
            if key in metadata
        ),
        None,
    )
    if explicit_stride is not None:
        stride = _directx_int_field(
            explicit_stride,
            field_name="byteStride",
            resource=binding.name,
        )
    else:
        if namespace == "cbv":
            stride = 0
        elif "structuredbuffer" in normalized_type:
            match = re.search(r"structuredbuffer<([^>]+)>", normalized_type)
            element_type = match.group(1) if match else ""
            stride = _directx_hlsl_element_stride(element_type)
            if stride is None:
                raise _directx_setup_error(
                    f"DirectX runtime cannot infer the stride for {binding.name!r}.",
                    "buffer-stride-missing",
                    resource=binding.name,
                    type=resource.type_name,
                )
        elif type_name:
            raise _directx_setup_error(
                f"DirectX runtime cannot infer the buffer view for {binding.name!r}.",
                "unsupported-buffer-view",
                resource=binding.name,
                type=resource.type_name,
            )
        else:
            stride = _dtype_size(dtype)
    if (
        stride < 0
        or (stride == 0 and namespace != "cbv")
        or (stride and payload_size % stride)
    ):
        raise _directx_setup_error(
            f"DirectX runtime buffer {binding.name!r} has an invalid byte stride.",
            "buffer-stride-invalid",
            resource=binding.name,
            byteStride=stride,
            byteLength=payload_size,
        )
    return stride


def _directx_hlsl_element_stride(type_name: str) -> int | None:
    scalar_sizes = {
        "float": 4,
        "float32_t": 4,
        "int": 4,
        "int32_t": 4,
        "uint": 4,
        "uint32_t": 4,
    }
    if type_name in scalar_sizes:
        return scalar_sizes[type_name]
    match = re.fullmatch(r"(float|int|uint)([1-4])", type_name)
    if match:
        return 4 * int(match.group(2))
    return None


def _directx_constant_metadata(binding: Any) -> dict[str, Any]:
    metadata = {
        **dict(binding.constant.metadata),
        **dict(binding.metadata),
    }
    nested = metadata.get("directx")
    if isinstance(nested, Mapping):
        metadata.update(nested)
    directx_binding = metadata.get("directxBinding")
    if isinstance(directx_binding, Mapping):
        metadata.update(directx_binding)
    elif isinstance(directx_binding, str) and "mechanism" not in metadata:
        metadata["mechanism"] = directx_binding
    if "kind" in metadata and "mechanism" not in metadata:
        metadata["mechanism"] = metadata["kind"]
    if "registerSpace" in metadata and "space" not in metadata:
        metadata["space"] = metadata["registerSpace"]
    return metadata


def _validate_directx_compiled_constant(name: str, binding: Any) -> None:
    reflected_value = binding.constant.value
    if reflected_value is None:
        reflected_value = binding.constant.default
    if reflected_value is None or binding.value != reflected_value:
        raise _directx_setup_error(
            f"DirectX compiled constant {name!r} cannot be overridden at dispatch.",
            "compiled-constant-mismatch",
            constant=name,
            reflectedValue=reflected_value,
            boundValue=binding.value,
        )


def _normalize_directx_dtype(dtype: str | None, *, resource: str) -> str:
    try:
        return _normalize_dtype(dtype, target="DirectX")
    except RuntimeExecutorUnavailable as exc:
        raise _directx_setup_error(
            str(exc),
            "unsupported-dtype",
            resource=resource,
            dtype=dtype,
        ) from exc


def _directx_int_field(
    value: Any,
    *,
    field_name: str,
    resource: str,
    default: int | None = None,
    register_prefix: str | None = None,
) -> int:
    if value is None:
        if default is not None:
            return default
        raise _directx_setup_error(
            f"DirectX runtime {field_name} is required for {resource!r}.",
            "integer-field-missing",
            resource=resource,
            field=field_name,
        )
    normalized = value
    if isinstance(value, str) and register_prefix:
        stripped = value.strip().lower()
        if stripped.startswith(register_prefix):
            normalized = stripped[len(register_prefix) :]
    try:
        result = int(normalized)
    except (TypeError, ValueError) as exc:
        raise _directx_setup_error(
            f"DirectX runtime {field_name} must be an integer for {resource!r}.",
            "integer-field-invalid",
            resource=resource,
            field=field_name,
            value=value,
        ) from exc
    if result < 0:
        raise _directx_setup_error(
            f"DirectX runtime {field_name} cannot be negative for {resource!r}.",
            "integer-field-invalid",
            resource=resource,
            field=field_name,
            value=result,
        )
    return result


def _directx_setup_error(
    message: str,
    reason_kind: str,
    **details: Any,
) -> RuntimeAdapterSetupError:
    return RuntimeAdapterSetupError(
        message,
        details={
            "target": "directx",
            "runtime": DirectXComputeRuntime.name,
            "reasonKind": reason_kind,
            **details,
        },
    )


def _align_to(value: int, alignment: int) -> int:
    value = max(value, 1)
    return max(alignment, ((value + alignment - 1) // alignment) * alignment)


_OPENGL_SPECIALIZATION_KINDS = frozenset(
    ("specialization-constant", "function-constant")
)
_OPENGL_SPECIALIZATION_TYPE_ALIASES = {
    "bool": "bool",
    "boolean": "bool",
    "int": "int32",
    "i32": "int32",
    "int32": "int32",
    "int32_t": "int32",
    "uint": "uint32",
    "u32": "uint32",
    "uint32": "uint32",
    "uint32_t": "uint32",
    "float": "float32",
    "f32": "float32",
    "float32": "float32",
    "float32_t": "float32",
}


def _opengl_constant_is_specialization(binding: Any) -> bool:
    kind = str(binding.constant.kind or "").strip().lower().replace("_", "-")
    return kind in _OPENGL_SPECIALIZATION_KINDS


def _opengl_request_requires_specialization(request: RuntimeExecutionRequest) -> bool:
    return any(
        str(constant.kind or "").strip().lower().replace("_", "-")
        in _OPENGL_SPECIALIZATION_KINDS
        for constant in request.adapter_contract.specialization_constants
    )


def _partition_opengl_constants(
    bindings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    specializations: dict[str, Any] = {}
    uniforms: dict[str, Any] = {}
    for name, binding in bindings.items():
        target = (
            specializations if _opengl_constant_is_specialization(binding) else uniforms
        )
        target[name] = binding
    return specializations, uniforms


def _prepare_opengl_specializations(
    bindings: Mapping[str, Any],
) -> tuple[_PreparedOpenGLSpecialization, ...]:
    prepared: list[_PreparedOpenGLSpecialization] = []
    seen_ids: dict[int, str] = {}
    for name, binding in bindings.items():
        constant_id = binding.constant.constant_id
        if not isinstance(constant_id, int) or isinstance(constant_id, bool):
            raise _opengl_setup_error(
                f"OpenGL specialization constant {name!r} requires a numeric id.",
                "specialization-id-invalid",
                constant=name,
                constantId=constant_id,
            )
        if constant_id < 0 or constant_id > 0xFFFFFFFF:
            raise _opengl_setup_error(
                f"OpenGL specialization constant {name!r} id is outside uint32 range.",
                "specialization-id-out-of-range",
                constant=name,
                constantId=constant_id,
            )
        previous = seen_ids.get(constant_id)
        if previous is not None:
            raise _opengl_setup_error(
                f"OpenGL specialization constant id {constant_id} is duplicated by "
                f"{previous!r} and {name!r}.",
                "specialization-id-duplicate",
                constantId=constant_id,
                constants=[previous, name],
            )
        seen_ids[constant_id] = name

        if binding.value is None:
            if binding.constant.required:
                raise _opengl_setup_error(
                    f"OpenGL specialization constant {name!r} has no bound value.",
                    "specialization-value-missing",
                    constant=name,
                    constantId=constant_id,
                    dtype=binding.constant.dtype,
                )
            continue
        dtype = _normalize_opengl_specialization_dtype(name, binding.constant.dtype)
        value, encoded_word = _encode_opengl_specialization_value(
            name,
            constant_id,
            dtype,
            binding.value,
        )
        provenance = dict(binding.constant.value_provenance)
        if binding.source is not None:
            provenance.setdefault("bindingSource", binding.source)
        prepared.append(
            _PreparedOpenGLSpecialization(
                name=name,
                constant_id=constant_id,
                dtype=dtype,
                value=value,
                encoded_word=encoded_word,
                source=binding.source,
                value_provenance=provenance,
            )
        )
    return tuple(sorted(prepared, key=lambda item: (item.constant_id, item.name)))


def _normalize_opengl_specialization_dtype(name: str, dtype: Any) -> str:
    normalized = re.sub(r"\s+", "", str(dtype or "")).lower()
    resolved = _OPENGL_SPECIALIZATION_TYPE_ALIASES.get(normalized)
    if resolved is not None:
        return resolved
    if re.search(r"(?:8|16|64)|double|half", normalized):
        reason_kind = "specialization-width-unsupported"
        message = (
            f"OpenGL specialization constant {name!r} uses unsupported scalar "
            f"width {dtype!r}; only bool and 32-bit scalar values are supported."
        )
    else:
        reason_kind = "specialization-type-unsupported"
        message = (
            f"OpenGL specialization constant {name!r} requires a bool, int32, "
            f"uint32, or float32 scalar type, not {dtype!r}."
        )
    raise _opengl_setup_error(
        message,
        reason_kind,
        constant=name,
        dtype=dtype,
        supportedTypes=["bool", "int32", "uint32", "float32"],
    )


def _encode_opengl_specialization_value(
    name: str,
    constant_id: int,
    dtype: str,
    value: Any,
) -> tuple[Any, int]:
    details = {
        "constant": name,
        "constantId": constant_id,
        "dtype": dtype,
        "valueType": type(value).__name__,
    }
    if isinstance(value, (str, int, float, bool)) or value is None:
        details["value"] = value
    if dtype == "bool":
        if not isinstance(value, bool):
            raise _opengl_setup_error(
                f"OpenGL bool specialization constant {name!r} requires a bool value.",
                "specialization-value-type-invalid",
                **details,
            )
        return value, int(value)
    if dtype in {"int32", "uint32"}:
        if not isinstance(value, int) or isinstance(value, bool):
            raise _opengl_setup_error(
                f"OpenGL {dtype} specialization constant {name!r} requires an integer value.",
                "specialization-value-type-invalid",
                **details,
            )
        minimum, maximum = (
            (-0x80000000, 0x7FFFFFFF) if dtype == "int32" else (0, 0xFFFFFFFF)
        )
        if value < minimum or value > maximum:
            raise _opengl_setup_error(
                f"OpenGL {dtype} specialization constant {name!r} is out of range.",
                "specialization-value-out-of-range",
                minimum=minimum,
                maximum=maximum,
                **details,
            )
        if dtype == "int32":
            encoded = struct.unpack("<I", struct.pack("<i", value))[0]
        else:
            encoded = value
        return value, encoded
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _opengl_setup_error(
            f"OpenGL float32 specialization constant {name!r} requires a numeric value.",
            "specialization-value-type-invalid",
            **details,
        )
    try:
        normalized_value = float(value)
    except (OverflowError, ValueError) as exc:
        raise _opengl_setup_error(
            f"OpenGL float32 specialization constant {name!r} is out of range.",
            "specialization-value-out-of-range",
            **details,
        ) from exc
    if not math.isfinite(normalized_value):
        raise _opengl_setup_error(
            f"OpenGL float32 specialization constant {name!r} must be finite.",
            "specialization-value-invalid",
            **details,
        )
    try:
        encoded = struct.unpack("<I", struct.pack("<f", normalized_value))[0]
    except (OverflowError, struct.error) as exc:
        raise _opengl_setup_error(
            f"OpenGL float32 specialization constant {name!r} is out of range.",
            "specialization-value-out-of-range",
            **details,
        ) from exc
    return normalized_value, encoded


def _prepare_opengl_uniform_values(name: str, binding: Any) -> tuple[list[Any], str]:
    values = _flatten_values(binding.value)
    if not 1 <= len(values) <= 4:
        raise _opengl_setup_error(
            f"OpenGL runtime uniform {name!r} requires one to four scalar values.",
            "uniform-value-shape-unsupported",
            constant=name,
            valueCount=len(values),
        )
    normalized = re.sub(r"\s+", "", str(binding.constant.dtype or "")).lower()
    dtype = _OPENGL_SPECIALIZATION_TYPE_ALIASES.get(normalized)
    if dtype is None:
        raise _opengl_setup_error(
            f"OpenGL runtime uniform {name!r} has unsupported dtype "
            f"{binding.constant.dtype!r}.",
            "uniform-type-unsupported",
            constant=name,
            dtype=binding.constant.dtype,
        )
    if dtype == "bool":
        if any(not isinstance(value, bool) for value in values):
            raise _opengl_setup_error(
                f"OpenGL bool uniform {name!r} requires bool values.",
                "uniform-value-type-invalid",
                constant=name,
                dtype=dtype,
            )
        return [int(value) for value in values], "i"
    if dtype in {"int32", "uint32"}:
        if any(
            not isinstance(value, int) or isinstance(value, bool) for value in values
        ):
            raise _opengl_setup_error(
                f"OpenGL {dtype} uniform {name!r} requires integer values.",
                "uniform-value-type-invalid",
                constant=name,
                dtype=dtype,
            )
        minimum, maximum = (
            (-0x80000000, 0x7FFFFFFF) if dtype == "int32" else (0, 0xFFFFFFFF)
        )
        if any(value < minimum or value > maximum for value in values):
            raise _opengl_setup_error(
                f"OpenGL {dtype} uniform {name!r} is out of range.",
                "uniform-value-out-of-range",
                constant=name,
                dtype=dtype,
                minimum=minimum,
                maximum=maximum,
            )
        return values, "i" if dtype == "int32" else "ui"
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in values
    ):
        raise _opengl_setup_error(
            f"OpenGL float32 uniform {name!r} requires finite numeric values.",
            "uniform-value-type-invalid",
            constant=name,
            dtype=dtype,
        )
    return [float(value) for value in values], "f"


def _opengl_entry_point_available(function: Any) -> bool:
    if not callable(function):
        return False
    try:
        return bool(function)
    except Exception:
        return False


def _opengl_context_version_code(context: Any) -> int:
    try:
        return int(getattr(context, "version_code", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _opengl_status(value: Any) -> bool:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        value = value[0] if value else 0
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def _opengl_info_log(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    text = str(value or "").strip()
    return text or "no driver log was provided"


def _opengl_setup_error(
    message: str,
    reason_kind: str,
    **details: Any,
) -> RuntimeAdapterSetupError:
    return RuntimeAdapterSetupError(
        message,
        details={
            "target": "opengl",
            "runtime": OpenGLComputeRuntime.name,
            "reasonKind": reason_kind,
            **details,
        },
    )


def _release_opengl_object(value: Any) -> None:
    if value is None:
        return
    release = getattr(value, "release", None)
    if callable(release):
        try:
            release()
        except Exception:
            pass


def _vulkan_descriptor_type(vk: Any, resource_kind: str | None) -> Any:
    if resource_kind in {"constant-buffer", "uniform"}:
        return vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    return vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER


def _vulkan_buffer_usage(vk: Any, resource_kind: str | None) -> Any:
    if resource_kind in {"constant-buffer", "uniform"}:
        return vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    return vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT


def _prepare_vulkan_buffers(
    bindings: Mapping[str, NativeRuntimeBufferBinding],
) -> tuple[_PreparedVulkanBuffer, ...]:
    prepared = []
    seen_bindings: set[tuple[int, int]] = set()
    for name, binding in bindings.items():
        resource = binding.binding
        if resource.kind not in (
            None,
            "buffer",
            "storage-buffer",
            "constant-buffer",
            "uniform",
        ):
            raise RuntimeExecutorUnavailable(
                f"Vulkan compute runtime supports buffer resources only: {name}."
            )
        if resource.binding is None:
            raise RuntimeExecutorUnavailable(
                f"Vulkan compute runtime requires an explicit binding for {name}."
            )
        set_index = _int_field(resource.set, default=0)
        binding_index = _int_field(resource.binding)
        descriptor = (set_index, binding_index)
        if descriptor in seen_bindings:
            raise RuntimeExecutorUnavailable(
                "Vulkan compute runtime requires unique set/binding pairs."
            )
        seen_bindings.add(descriptor)
        dtype = _normalize_dtype(binding.dtype, target="Vulkan")
        shape = tuple(int(value) for value in binding.shape)
        element_count = (
            math.prod(shape) if shape else len(_flatten_values(binding.value))
        )
        if binding.source == "expectedOutput":
            payload = b"\x00" * (element_count * _dtype_size(dtype))
        else:
            payload = _pack_values(
                binding.value,
                dtype,
                expected_count=element_count,
                target="Vulkan",
            )
        prepared.append(
            _PreparedVulkanBuffer(
                name=name,
                set_index=set_index,
                binding_index=binding_index,
                resource_kind=resource.kind,
                dtype=dtype,
                shape=shape,
                source=binding.source,
                output_name=_runtime_value_name(binding),
                payload=payload,
            )
        )
    return tuple(
        sorted(prepared, key=lambda item: (item.set_index, item.binding_index))
    )


def _prepare_opengl_buffers(
    bindings: Mapping[str, NativeRuntimeBufferBinding],
) -> tuple[_PreparedOpenGLBuffer, ...]:
    prepared = []
    seen_bindings: set[tuple[str, int]] = set()
    for name, binding in bindings.items():
        resource = binding.binding
        if resource.kind not in (
            None,
            "buffer",
            "storage-buffer",
            "constant-buffer",
            "uniform",
        ):
            raise RuntimeExecutorUnavailable(
                f"OpenGL compute runtime supports buffer resources only: {name}."
            )
        if resource.binding is None:
            raise RuntimeExecutorUnavailable(
                f"OpenGL compute runtime requires an explicit binding for {name}."
            )
        set_index = _int_field(resource.set, default=0)
        if set_index != 0:
            raise RuntimeExecutorUnavailable(
                "OpenGL compute runtime does not support nonzero descriptor sets."
            )
        binding_index = _int_field(resource.binding)
        namespace = (
            "uniform" if resource.kind in {"constant-buffer", "uniform"} else "storage"
        )
        if binding.source == "expectedOutput" and namespace == "uniform":
            raise _opengl_setup_error(
                f"OpenGL uniform buffer {name!r} cannot be a runtime output resource.",
                "unsupported-output-resource",
                resource=name,
                resourceKind=resource.kind,
                binding=binding_index,
            )
        access = str(resource.access or "").strip().lower().replace("-", "_")
        if binding.source == "expectedOutput" and access in {
            "read",
            "read_only",
            "readonly",
        }:
            raise _opengl_setup_error(
                f"OpenGL output resource {name!r} is reflected as read-only.",
                "resource-access-mismatch",
                resource=name,
                resourceKind=resource.kind,
                access=resource.access,
                binding=binding_index,
            )
        descriptor = (namespace, binding_index)
        if descriptor in seen_bindings:
            raise RuntimeExecutorUnavailable(
                f"OpenGL compute runtime requires unique {namespace} buffer bindings."
            )
        seen_bindings.add(descriptor)
        dtype = _normalize_dtype(binding.dtype, target="OpenGL")
        shape = tuple(int(value) for value in binding.shape)
        element_count = (
            math.prod(shape) if shape else len(_flatten_values(binding.value))
        )
        if binding.source == "expectedOutput":
            payload = b"\x00" * (element_count * _dtype_size(dtype))
        else:
            payload = _pack_values(
                binding.value,
                dtype,
                expected_count=element_count,
                target="OpenGL",
            )
        allocation_size = len(payload)
        if namespace == "uniform":
            allocation_size = _scalar_block_size(
                binding,
                target="opengl",
                dtype=dtype,
                payload_size=len(payload),
            )
        prepared.append(
            _PreparedOpenGLBuffer(
                name=name,
                binding_index=binding_index,
                resource_kind=resource.kind,
                dtype=dtype,
                shape=shape,
                source=binding.source,
                output_name=_runtime_value_name(binding),
                payload=payload,
                allocation_size=allocation_size,
            )
        )
    return tuple(
        sorted(
            prepared,
            key=lambda item: (
                item.resource_kind in {"constant-buffer", "uniform"},
                item.binding_index,
            ),
        )
    )


def _scalar_block_size(
    binding: NativeRuntimeBufferBinding,
    *,
    target: str,
    dtype: str,
    payload_size: int,
) -> int:
    resource = binding.binding
    target_name = {"directx": "DirectX", "opengl": "OpenGL"}[target]
    raw_layout = resource.metadata.get("scalarLayout")
    if raw_layout is None:
        raise _scalar_block_error(
            target,
            f"{target_name} scalar block {binding.name!r} has no physical layout metadata.",
            "scalar-block-layout-missing",
            resource=binding.name,
        )
    if not isinstance(raw_layout, Mapping):
        raise _scalar_block_error(
            target,
            f"{target_name} scalar block {binding.name!r} has malformed physical layout metadata.",
            "scalar-block-layout-invalid",
            resource=binding.name,
            scalarLayout=raw_layout,
        )

    required_fields = {
        "physicalType",
        "elementType",
        "elementSizeBytes",
        "elementStrideBytes",
        "storageLayout",
        "alignmentBytes",
        "blockSizeBytes",
        "memberOffsetBytes",
        "runtimeSized",
    }
    missing_fields = sorted(required_fields.difference(raw_layout))
    if missing_fields:
        raise _scalar_block_error(
            target,
            f"{target_name} scalar block {binding.name!r} has incomplete physical layout metadata.",
            "scalar-block-layout-invalid",
            resource=binding.name,
            missingFields=missing_fields,
        )

    storage_layout = raw_layout["storageLayout"]
    expected_storage_layout = {
        "directx": "hlsl-constant-buffer",
        "opengl": "std140",
    }[target]
    if storage_layout != expected_storage_layout:
        raise _scalar_block_error(
            target,
            f"{target_name} scalar block {binding.name!r} has an incompatible storage layout.",
            "scalar-block-storage-layout-unsupported",
            resource=binding.name,
            storageLayout=storage_layout,
            expectedStorageLayout=expected_storage_layout,
        )

    integer_fields = {}
    for field_name in (
        "elementSizeBytes",
        "elementStrideBytes",
        "alignmentBytes",
        "blockSizeBytes",
        "memberOffsetBytes",
    ):
        value = raw_layout[field_name]
        if not isinstance(value, int) or isinstance(value, bool):
            raise _scalar_block_error(
                target,
                f"{target_name} scalar block {binding.name!r} has a non-integer {field_name}.",
                "scalar-block-layout-invalid",
                resource=binding.name,
                field=field_name,
                value=value,
            )
        integer_fields[field_name] = value

    element_type = str(raw_layout["elementType"] or "").strip().lower()
    physical_type = str(raw_layout["physicalType"] or "").strip().lower()
    expected_physical_type = {
        "float32": "float",
        "int32": "int",
        "uint32": "uint",
    }[dtype]
    element_size = integer_fields["elementSizeBytes"]
    element_stride = integer_fields["elementStrideBytes"]
    if (
        element_type != dtype
        or physical_type != expected_physical_type
        or element_size != _dtype_size(dtype)
        or element_stride != element_size
        or payload_size != element_size
    ):
        raise _scalar_block_error(
            target,
            f"{target_name} constant and uniform buffers support one reflected scalar value.",
            "scalar-block-element-layout-unsupported",
            resource=binding.name,
            dtype=dtype,
            payloadSizeBytes=payload_size,
            elementType=raw_layout["elementType"],
            physicalType=raw_layout["physicalType"],
            expectedPhysicalType=expected_physical_type,
            elementSizeBytes=element_size,
            elementStrideBytes=element_stride,
        )

    member_offset = integer_fields["memberOffsetBytes"]
    if member_offset != 0:
        raise _scalar_block_error(
            target,
            f"{target_name} scalar block {binding.name!r} requires member offset zero.",
            "scalar-block-member-offset-unsupported",
            resource=binding.name,
            memberOffsetBytes=member_offset,
        )
    if raw_layout["runtimeSized"] is not False:
        raise _scalar_block_error(
            target,
            f"{target_name} scalar block {binding.name!r} cannot be runtime-sized.",
            "scalar-block-runtime-size-invalid",
            resource=binding.name,
            runtimeSized=raw_layout["runtimeSized"],
        )

    alignment = integer_fields["alignmentBytes"]
    if alignment < 16 or alignment & (alignment - 1):
        raise _scalar_block_error(
            target,
            f"{target_name} scalar block {binding.name!r} has an invalid alignment.",
            "scalar-block-alignment-invalid",
            resource=binding.name,
            alignmentBytes=alignment,
        )
    block_size = integer_fields["blockSizeBytes"]
    if block_size < payload_size or block_size % alignment:
        raise _scalar_block_error(
            target,
            f"{target_name} scalar block {binding.name!r} has an invalid block size.",
            "scalar-block-size-invalid",
            resource=binding.name,
            payloadSizeBytes=payload_size,
            blockSizeBytes=block_size,
            alignmentBytes=alignment,
        )
    return block_size


def _scalar_block_error(
    target: str,
    message: str,
    reason_kind: str,
    **details: Any,
) -> RuntimeAdapterSetupError:
    if target == "directx":
        return _directx_setup_error(message, reason_kind, **details)
    return _opengl_setup_error(message, reason_kind, **details)


def _runtime_value_name(binding: NativeRuntimeBufferBinding) -> str | None:
    value_name = binding.metadata.get("runtimeValueName")
    if isinstance(value_name, str) and value_name.strip():
        return value_name.strip()
    return None


def _workgroup_count(
    request: NativeRuntimeDispatchRequest,
    *,
    target: str = "Vulkan",
) -> tuple[int, int, int]:
    dispatch = request.dispatch
    if dispatch is None:
        raise RuntimeExecutorUnavailable(
            f"{target} compute runtime requires dispatch geometry."
        )
    if dispatch.workgroup_count:
        values = tuple(int(value) for value in dispatch.workgroup_count)
    elif dispatch.global_size and dispatch.workgroup_size:
        values = tuple(
            max(1, math.ceil(int(global_value) / int(local_value)))
            for global_value, local_value in zip(
                dispatch.global_size,
                dispatch.workgroup_size,
            )
        )
    else:
        raise RuntimeExecutorUnavailable(
            f"{target} compute runtime requires workgroupCount or "
            "globalSize/workgroupSize."
        )
    return _pad3(values, field_name="workgroupCount", target=target)


def _pad3(
    values: Sequence[int],
    *,
    field_name: str,
    target: str = "Vulkan",
) -> tuple[int, int, int]:
    if len(values) > 3:
        raise RuntimeExecutorUnavailable(
            f"{target} compute runtime {field_name} must have at most three dimensions."
        )
    padded = tuple(max(1, int(value)) for value in values) + (1,) * (3 - len(values))
    return padded[:3]


def _int_field(value: Any, *, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise RuntimeExecutorUnavailable("Expected an integer field.")
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeExecutorUnavailable(
            f"Expected an integer field, got {value!r}."
        ) from exc


def _normalize_dtype(dtype: str | None, *, target: str = "Vulkan") -> str:
    aliases = {
        "float": "float32",
        "f32": "float32",
        "float32_t": "float32",
        "uint": "uint32",
        "u32": "uint32",
        "int": "int32",
        "i32": "int32",
    }
    normalized = str(dtype or "").strip().lower()
    value = aliases.get(normalized, normalized)
    if value not in {"float32", "uint32", "int32"}:
        raise RuntimeExecutorUnavailable(
            f"{target} compute runtime supports float32, uint32, and int32 buffers, "
            f"not {dtype!r}."
        )
    return value


def _dtype_format(dtype: str) -> str:
    return {"float32": "f", "uint32": "I", "int32": "i"}[dtype]


def _dtype_size(dtype: str) -> int:
    return struct.calcsize("<" + _dtype_format(dtype))


def _flatten_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    if isinstance(value, Sequence):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_values(item))
        return flattened
    return [value]


def _pack_values(
    value: Any,
    dtype: str,
    *,
    expected_count: int,
    target: str = "Vulkan",
) -> bytes:
    values = _flatten_values(value)
    if len(values) != expected_count:
        raise RuntimeExecutorUnavailable(
            f"{target} compute runtime buffer value count does not match shape."
        )
    return struct.pack("<" + _dtype_format(dtype) * expected_count, *values)


def _unpack_values(
    payload: bytes,
    dtype: str,
    *,
    target: str = "Vulkan",
) -> list[Any]:
    size = _dtype_size(dtype)
    if len(payload) % size:
        raise RuntimeAdapterDispatchError(
            f"{target} runtime output byte length is not aligned to the dtype size.",
            details={
                "target": target.lower(),
                "reasonKind": "output-layout-invalid",
                "dtype": dtype,
                "byteLength": len(payload),
            },
        )
    count = len(payload) // size
    if count == 0:
        return []
    return list(struct.unpack("<" + _dtype_format(dtype) * count, payload))
