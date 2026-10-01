"""Bind the explicit MLX callback ABI to CrossTL native package execution."""

from __future__ import annotations

import ctypes
import json
import math
import sys
from pathlib import Path

from crosstl.project import build_native_loader_dispatch_request
from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    OpenGLComputeRuntime,
)
from crosstl.project.runtime_verification import (
    DirectXRuntimeParityAdapter,
    OpenGLRuntimeParityAdapter,
    RuntimeParityExecutor,
    RuntimeTestAdapterSpec,
)
from demos.integrations.mlx.portable_host.packages import ENTRIES, UNARY_ENTRIES


class Buffer(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("dtype", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("count", ctypes.c_uint64),
        ("output", ctypes.c_uint32),
    ]


CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.POINTER(Buffer),
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_void_p,
    ctypes.c_size_t,
)
TYPES = {
    "float32": ctypes.c_float,
    "int32": ctypes.c_int32,
    "uint32": ctypes.c_uint32,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,
}
_installed_runtime = None


def wire_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else "+infinity" if value > 0 else "-infinity"
    return value


class HostRuntime:
    def __init__(self, directory, trace):
        self.directory = Path(directory).resolve()
        self.trace = Path(trace).resolve()
        index = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        self.target = index["target"]
        self.descriptors = index["descriptors"]
        if set(self.descriptors) != set(ENTRIES):
            raise ValueError("Packages must contain the exact supported entry set")
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        if self.target == "opengl":
            adapter = OpenGLRuntimeParityAdapter(
                runtime=OpenGLComputeRuntime(context_backends=("egl",))
            )
        elif self.target == "directx":
            adapter = DirectXRuntimeParityAdapter(runtime=DirectXComputeRuntime())
        else:
            raise ValueError(f"Unsupported host target: {self.target}")
        self.executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id=f"mlx-host-{self.target}",
                target=self.target,
                executor=self.target,
                adapter_kind=f"{self.target}-native-runtime",
            ),
            runtime_adapter=adapter,
        )
        self.callback = CALLBACK(self._dispatch)
        self.library = None

    def install(self):
        global _installed_runtime
        import mlx.core as mx

        if _installed_runtime is not None:
            raise RuntimeError("A CrossTL host runtime is already installed")
        library = (
            Path(mx.__file__).with_name("mlx.dll")
            if sys.platform == "win32"
            else Path(mx.__file__)
        )
        self.library = ctypes.CDLL(str(library))
        register = self.library.crosstl_mlx_register_dispatch
        register.argtypes = [ctypes.c_uint32, CALLBACK]
        register.restype = ctypes.c_int
        result = register(1, self.callback)
        if result:
            raise RuntimeError(
                f"MLX rejected the native callback registration: {result}"
            )
        _installed_runtime = self
        mx.set_default_device(mx.gpu)

    def _dispatch(self, entry, buffers, count, threads, error, capacity):
        try:
            self.dispatch(entry.decode("ascii"), buffers, count, threads)
            return 0
        except Exception as exception:
            message = str(exception).encode("utf-8")
            if error and capacity:
                payload = message[: capacity - 1] + b"\0"
                ctypes.memmove(error, payload, len(payload))
            return 1

    def dispatch(self, entry, buffers, count, threads):
        if entry not in self.descriptors:
            raise ValueError(f"No translated package for {entry}")
        if count != 3 or not buffers or not 0 < threads <= 65535:
            raise ValueError("Invalid or unsupported native dispatch dimensions")
        descriptor = self.descriptors[entry]
        unary = entry in UNARY_ENTRIES
        names = {"in", "size", "out"} if unary else {"start", "step", "out"}
        supplied = {}
        for index in range(count):
            buffer = buffers[index]
            if not buffer.name or not buffer.dtype:
                raise ValueError("Native buffer identity is missing")
            name = buffer.name.decode("ascii")
            dtype = buffer.dtype.decode("ascii")
            if name in supplied or dtype not in TYPES or not buffer.data:
                raise ValueError("Invalid native buffer")
            expected = threads if name == "out" or (unary and name == "in") else 1
            if buffer.count != expected or buffer.output != int(name == "out"):
                raise ValueError("Native buffer shape or direction does not match")
            if unary and dtype != ("uint32" if name == "size" else "float32"):
                raise ValueError("Native unary buffer dtype does not match")
            supplied[name] = buffer
        if set(supplied) != names:
            raise ValueError("Native buffer names do not match the operation")
        if (
            unary
            and ctypes.cast(supplied["size"].data, ctypes.POINTER(ctypes.c_uint32))[0]
            != threads
        ):
            raise ValueError("Native unary size does not match the launch")
        inputs, outputs, destinations = {}, {}, {}
        matched = set()
        binding_names = set()
        for binding in descriptor["bindings"]:
            layout = binding["scalarLayout"]
            member = layout.get("memberName", binding["name"])
            if self.target == "directx":
                member = member.removeprefix(entry + "_")
            name = {"out_": "out", "in_": "in"}.get(member, member)
            if (
                name not in supplied
                or name in matched
                or binding["name"] in binding_names
            ):
                raise ValueError(
                    "Reflected binding identity does not match the operation"
                )
            matched.add(name)
            binding_names.add(binding["name"])
            buffer = supplied[name]
            dtype = buffer.dtype.decode("ascii")
            if layout["elementType"] != dtype or layout[
                "elementStrideBytes"
            ] != ctypes.sizeof(TYPES[dtype]):
                raise ValueError("Native and reflected buffer layouts disagree")
            ctype = TYPES[dtype]
            view = ctypes.cast(
                buffer.data, ctypes.POINTER(ctype * buffer.count)
            ).contents
            value = {
                "dtype": dtype,
                "shape": [buffer.count],
                "values": (
                    [0] * buffer.count
                    if buffer.output
                    else [wire_value(value) for value in view]
                ),
            }
            if buffer.output:
                outputs[binding["name"]] = value
                destinations[binding["name"]] = (buffer, ctype)
            else:
                inputs[binding["name"]] = value
        if matched != set(supplied):
            raise ValueError("Reflected bindings do not cover the operation")
        request = build_native_loader_dispatch_request(
            descriptor,
            self.directory / "package",
            inputs,
            outputs,
            (threads, 1, 1),
            expected_target=self.target,
        )
        result = self.executor.run(request)
        if result.status != "ok" or set(result.outputs) != set(destinations):
            raise RuntimeError("Native executor did not return the required outputs")
        for name, (buffer, ctype) in destinations.items():
            output = result.outputs[name]
            if output["dtype"] != buffer.dtype.decode("ascii") or output["shape"] != [
                buffer.count
            ]:
                raise RuntimeError("Native readback layout does not match the output")
            if len(output["values"]) != buffer.count:
                raise RuntimeError("Native readback size does not match the output")
            values = (ctype * buffer.count)(
                *(
                    float(value) if ctype is ctypes.c_float else value
                    for value in output["values"]
                )
            )
            ctypes.memmove(buffer.data, values, ctypes.sizeof(values))
        with self.trace.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "entry": entry,
                        "target": self.target,
                        "threads": threads,
                        "artifact": descriptor["artifact"],
                        "details": result.details,
                    }
                )
                + "\n"
            )
