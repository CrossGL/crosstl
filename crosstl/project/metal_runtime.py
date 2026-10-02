"""Process-isolated native Metal execution for packaged compute artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .native_loader_dispatch import NativeLoaderDispatchError, _validated_scalar_layout
from .native_runtime_drivers import (
    _binding_requires_readback,
    _buffer_readback,
    _buffer_readback_encoding,
    _dtype_size,
    _normalize_dtype,
    _pack_values,
    _runtime_value_name,
)
from .runtime_verification import (
    RuntimeAdapterDispatchError,
    RuntimeAdapterSetupError,
    RuntimeExecutorAvailability,
    RuntimeValue,
)


def run_metal_command(command, *, input_text=None, timeout_seconds=120):
    """Run one operation with a deadline that also terminates compiler children."""
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("Metal operation timeout must be positive and finite.")
    with subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(input_text, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            raise RuntimeAdapterSetupError(
                "Metal native operation exceeded its deadline.",
                details={
                    "target": "metal",
                    "reasonKind": "operation-timeout",
                    "timeoutSeconds": timeout_seconds,
                    "command": list(command),
                    "stdout": stdout,
                    "stderr": stderr,
                },
            ) from exc
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


class MetalComputeRuntime:
    """Execute Metal libraries in a bounded worker without a Python GPU dependency.

    The worker uses Metal shared buffers, explicit argument indices and 3D
    threadgroup dispatch. No host implementation of shader computation is used.
    """

    name = "metal-compute-runtime"
    supported_platforms = ("darwin",)

    def __init__(
        self,
        *,
        timeout_seconds=120,
        max_buffer_bytes=256 * 1024 * 1024,
        command_runner=None,
        tool_resolver=None,
        platform_name=None,
    ):
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("Metal operation timeout must be positive and finite.")
        if (
            type(max_buffer_bytes) is not int
            or not 0 < max_buffer_bytes <= (1 << 63) - 1
        ):
            raise ValueError(
                "Metal allocation limit must be a positive int64 byte count."
            )
        self.timeout_seconds = timeout_seconds
        self.max_buffer_bytes = max_buffer_bytes
        self.command_runner = command_runner or run_metal_command
        self.tool_resolver = tool_resolver or shutil.which
        self.platform_name = platform_name or sys.platform
        self._worker_directory = None
        self._worker_path = None

    def close(self):
        if self._worker_directory is not None:
            self._worker_directory.cleanup()
            self._worker_directory = None
            self._worker_path = None

    def _run(self, command, *, input_text=None):
        return self.command_runner(
            command, input_text=input_text, timeout_seconds=self.timeout_seconds
        )

    def _worker(self):
        if self.platform_name != "darwin":
            raise _setup_error("Metal runtime requires macOS.", "platform-unavailable")
        if self._worker_path is not None:
            return self._worker_path
        xcrun = self.tool_resolver("xcrun")
        if xcrun is None:
            raise _setup_error(
                "Metal runtime requires Xcode tools.", "tool-unavailable"
            )
        directory = tempfile.TemporaryDirectory(prefix="crosstl-metal-worker-")
        path = Path(directory.name) / "metal-runtime"
        try:
            result = self._run(
                [
                    xcrun,
                    "--sdk",
                    "macosx",
                    "swiftc",
                    "-O",
                    "-warnings-as-errors",
                    "-module-cache-path",
                    str(Path(directory.name) / "modules"),
                    str(Path(__file__).with_name("metal_runtime_worker.swift")),
                    "-o",
                    str(path),
                ]
            )
            if result.returncode or not path.is_file():
                raise _setup_error(
                    "Metal worker compilation failed.",
                    "worker-compile-failed",
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
        except BaseException:
            directory.cleanup()
            raise
        self._worker_directory = directory
        self._worker_path = path
        return path

    def is_available(self, adapter, request):
        _ = adapter, request
        try:
            result = self._run([str(self._worker()), "--probe"])
            payload = json.loads(result.stdout)
            if not isinstance(payload, dict):
                raise ValueError("Invalid Metal device probe response.")
            available = result.returncode == 0 and payload.get("available") is True
            return RuntimeExecutorAvailability(
                available,
                reason=None if available else "No Metal compute device is available.",
                details={
                    "target": "metal",
                    "runtime": self.name,
                    "reasonKind": "available" if available else "device-unavailable",
                    "device": payload.get("device"),
                },
            )
        except (OSError, ValueError, TypeError, RuntimeAdapterSetupError) as exc:
            return RuntimeExecutorAvailability(
                False,
                reason=str(exc),
                details={
                    "target": "metal",
                    "runtime": self.name,
                    **getattr(exc, "details", {}),
                },
            )

    def load_artifact(self, adapter, state, module_path):
        _ = adapter, state
        path = Path(module_path)
        if path.suffix != ".metallib" or not path.is_file() or not path.stat().st_size:
            raise _setup_error(
                "Metal runtime requires a compiled nonempty library.",
                "library-unavailable",
            )
        return path

    def dispatch(self, adapter, state, request):
        _ = adapter
        payload, outputs = self._prepare_request(request)
        worker = self._worker()
        try:
            result = self._run([str(worker)], input_text=json.dumps(payload))
        except RuntimeAdapterSetupError as exc:
            raise RuntimeAdapterDispatchError(str(exc), details=exc.details) from exc
        if result.returncode:
            details = {
                "target": "metal",
                "reasonKind": "worker-execution-failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            try:
                native_error = json.loads(result.stdout).get("error")
            except (ValueError, AttributeError):
                native_error = None
            if (
                isinstance(native_error, dict)
                and native_error.get("reasonKind") == "dispatch-limit-exceeded"
            ):
                details["reasonKind"] = "dispatch-limit-exceeded"
                details["dispatchValidation"] = native_error
            raise RuntimeAdapterDispatchError(
                "Metal worker execution failed.",
                details=details,
            )
        try:
            response = json.loads(result.stdout)
            raw_outputs = response["outputs"]
            if not isinstance(raw_outputs, dict) or set(raw_outputs) != set(outputs):
                raise ValueError("Metal readback names do not match the request.")
            values = {}
            for name, (dtype, shape, size, encoding) in outputs.items():
                data = base64.b64decode(raw_outputs[name], validate=True)
                if len(data) != size:
                    raise ValueError(f"Metal readback size does not match {name}.")
                values[name] = _buffer_readback(
                    data, dtype, shape, target="Metal", encoding=encoding
                )
        except (ValueError, TypeError, KeyError) as exc:
            raise RuntimeAdapterDispatchError(
                "Metal worker returned invalid readback data.",
                details={
                    "target": "metal",
                    "reasonKind": "readback-invalid",
                    "error": str(exc),
                },
            ) from exc
        state.details["metalRuntime"] = {
            "device": response.get("device"),
            "timeoutSeconds": self.timeout_seconds,
            "threadExecutionWidth": response.get("threadExecutionWidth"),
            "librarySHA256": (
                hashlib.sha256(Path(request.module_path).read_bytes()).hexdigest()
            ),
            "workgroupCount": payload["workgroupCount"],
            "workgroupSize": payload["workgroupSize"],
            "threadGridSize": payload["threadGridSize"],
            "bufferCount": len(payload["buffers"]),
        }
        return values

    def _prepare_request(self, request):
        if request.target != "metal" or not request.entry_point:
            raise _setup_error(
                "Metal dispatch requires its target and entry point.",
                "entry-point-invalid",
            )
        if request.dispatch is None:
            raise _setup_error("Metal dispatch requires geometry.", "dispatch-missing")
        counts = _dimensions(request.dispatch.workgroup_count, "workgroupCount")
        size = _dimensions(request.dispatch.workgroup_size, "workgroupSize")
        grid = [count * width for count, width in zip(counts, size)]
        thread_grid = None
        if request.dispatch.thread_grid_size != ():
            thread_grid = _dimensions(
                request.dispatch.thread_grid_size, "threadGridSize"
            )
            if [
                (extent + width - 1) // width
                for extent, width in zip(thread_grid, size)
            ] != counts:
                raise _setup_error(
                    "Metal thread grid does not match its covering threadgroups.",
                    "dispatch-size-mismatch",
                )
            grid = thread_grid
        if any(value > (1 << 32) - 1 for value in grid):
            raise _setup_error(
                "Metal grid exceeds uint32 invocation coordinates.",
                "dispatch-grid-overflow",
            )
        for supplied in (request.dispatch.global_size, request.dispatch.grid_size):
            if supplied and _dimensions(supplied, "globalSize") != grid:
                raise _setup_error(
                    "Metal grid conflicts with its dispatch dimensions.",
                    "dispatch-size-mismatch",
                )
        allocations = {}
        upload_ranges = {}
        total_view_bytes = 0
        buffers = []
        outputs = {}
        indices = set()
        for name, binding in request.buffers.items():
            resource = binding.binding
            index = resource.binding
            if (
                resource.kind not in {"buffer", "constant-buffer"}
                or resource.set != 0
                or type(index) is not int
                or not 0 <= index < 31
                or index in indices
            ):
                raise _setup_error(
                    "Metal buffer coordinates must be unique indices 0-30 in set zero.",
                    "buffer-binding-invalid",
                    resource=name,
                )
            indices.add(index)
            dtype = _normalize_dtype(binding.dtype, target="Metal")
            readback_encoding = _buffer_readback_encoding(binding, dtype)
            shape = tuple(binding.shape)
            if not shape or any(type(d) is not int or d <= 0 for d in shape):
                raise _setup_error(
                    "Metal buffer shape must contain positive dimensions.",
                    "buffer-shape-invalid",
                    resource=name,
                )
            value = RuntimeValue(
                name=name, dtype=dtype, shape=shape, values=binding.value
            )
            try:
                layout = _validated_scalar_layout(
                    resource.metadata.get("scalarLayout"),
                    runtime_value=value,
                    target="metal",
                    resource_kind=resource.kind,
                    path=f"$.buffers.{name}.layout",
                )
            except NativeLoaderDispatchError as exc:
                raise _setup_error(
                    str(exc), "buffer-layout-invalid", resource=name
                ) from exc
            length = math.prod(shape) * _dtype_size(dtype)
            total_view_bytes += length
            if total_view_bytes > self.max_buffer_bytes:
                raise _setup_error(
                    "Metal buffer payloads exceed the configured limit.",
                    "allocation-limit",
                )
            view = binding.allocation
            offset = view.byte_offset if view else 0
            view_length = (
                view.byte_length if view and view.byte_length is not None else length
            )
            allocation_length = (
                view.allocation_byte_length
                if view and view.allocation_byte_length is not None
                else offset + view_length
            )
            if (
                any(
                    type(n) is not int for n in (offset, view_length, allocation_length)
                )
                or offset < 0
                or view_length != length
                or offset % layout["alignmentBytes"]
                or offset + view_length > allocation_length
                or allocation_length > self.max_buffer_bytes
            ):
                raise _setup_error(
                    "Metal buffer allocation view is invalid or exceeds its limit.",
                    "allocation-view-invalid",
                    resource=name,
                )
            allocation_id = view.allocation_id if view else name
            if not isinstance(allocation_id, str) or not allocation_id:
                raise _setup_error(
                    "Metal allocation identity is required.",
                    "allocation-id-invalid",
                    resource=name,
                )
            allocation = allocations.setdefault(
                allocation_id,
                {"id": allocation_id, "length": allocation_length, "uploads": []},
            )
            if allocation["length"] != allocation_length:
                raise _setup_error(
                    "Metal allocation views disagree on backing size.",
                    "allocation-size-conflict",
                )
            if binding.value is not None:
                data = _pack_values(
                    binding.value,
                    dtype,
                    expected_count=math.prod(shape),
                    target="Metal",
                    encoding=binding.encoding,
                )
                for previous_offset, previous in upload_ranges.get(allocation_id, ()):
                    start = max(offset, previous_offset)
                    end = min(offset + len(data), previous_offset + len(previous))
                    if (
                        start < end
                        and data[start - offset : end - offset]
                        != previous[start - previous_offset : end - previous_offset]
                    ):
                        raise _setup_error(
                            "Metal aliased inputs contain conflicting bytes.",
                            "allocation-initialization-conflict",
                        )
                upload_ranges.setdefault(allocation_id, []).append((offset, data))
                allocation["uploads"].append(
                    {"offset": offset, "data": base64.b64encode(data).decode("ascii")}
                )
            readback = _binding_requires_readback(binding)
            if resource.access not in {"read", "write", "read_write"} or (
                readback and resource.access == "read"
            ):
                raise _setup_error(
                    "Metal output access is incompatible with its resource.",
                    "resource-access-mismatch",
                    resource=name,
                )
            output_name = _runtime_value_name(binding) or name
            if readback:
                if output_name in outputs:
                    raise _setup_error(
                        "Metal readback names must be unique.", "output-name-ambiguous"
                    )
                outputs[output_name] = (dtype, shape, length, readback_encoding)
            buffers.append(
                {
                    "index": index,
                    "allocation": allocation_id,
                    "offset": offset,
                    "length": length,
                    "alignment": layout["alignmentBytes"],
                    "output": output_name if readback else None,
                }
            )
        if sum(a["length"] for a in allocations.values()) > self.max_buffer_bytes:
            raise _setup_error(
                "Metal total allocation exceeds its configured limit.",
                "allocation-limit",
            )
        constants = []
        seen_constants = set()
        for binding in request.constants.values():
            constant = binding.constant
            formats = {"bool": "?", "float32": "f", "int32": "i", "uint32": "I"}
            if (
                type(constant.constant_id) is not int
                or not 0 <= constant.constant_id <= (1 << 32) - 1
                or constant.constant_id in seen_constants
                or constant.dtype not in formats
                or binding.value is None
            ):
                raise _setup_error(
                    "Metal function constants require unique ids and concrete scalar values.",
                    "function-constant-invalid",
                )
            seen_constants.add(constant.constant_id)
            value = binding.value
            if (
                (constant.dtype == "bool" and type(value) is not bool)
                or (constant.dtype in {"int32", "uint32"} and type(value) is not int)
                or (
                    constant.dtype == "float32"
                    and (type(value) not in {int, float} or not math.isfinite(value))
                )
            ):
                raise _setup_error(
                    "Metal function constant value does not match its scalar type.",
                    "function-constant-invalid",
                )
            try:
                data = struct.pack("<" + formats[constant.dtype], value)
            except (struct.error, OverflowError) as exc:
                raise _setup_error(
                    "Metal function constant is outside its scalar range.",
                    "function-constant-invalid",
                ) from exc
            constants.append(
                {
                    "id": constant.constant_id,
                    "dtype": constant.dtype,
                    "data": base64.b64encode(data).decode("ascii"),
                }
            )
        return {
            "library": str(request.module_path),
            "entryPoint": request.entry_point,
            "workgroupCount": counts,
            "workgroupSize": size,
            "threadGridSize": thread_grid,
            "allocations": list(allocations.values()),
            "buffers": buffers,
            "constants": constants,
        }, outputs


def _dimensions(values, name):
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray))
        or not 1 <= len(values) <= 3
        or any(type(n) is not int or not 0 < n <= (1 << 32) - 1 for n in values)
    ):
        raise _setup_error(
            "Metal dispatch dimensions must be positive uint32 values.",
            "dispatch-dimensions-invalid",
            field=name,
        )
    return list(values) + [1] * (3 - len(values))


def _setup_error(message, reason, **details):
    return RuntimeAdapterSetupError(
        message, details={"target": "metal", "reasonKind": reason, **details}
    )
