"""Bind the explicit MLX callback ABI to CrossTL native package execution."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import sys
from pathlib import Path

from crosstl.project import (
    build_native_loader_dispatch_request,
    prepare_native_loader_dispatch_regions,
)
from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    OpenGLComputeRuntime,
)
from crosstl.project.runtime_verification import (
    DirectXRuntimeParityAdapter,
    MetalRuntimeParityAdapter,
    OpenGLRuntimeParityAdapter,
    RuntimeExecutorResult,
    RuntimeParityExecutor,
    RuntimeTestAdapterSpec,
)
from demos.integrations.mlx.portable_host import (
    column_reduction_layout,
    copy_layout,
    reduction_layout,
    row_reduction_layout,
)
from demos.integrations.mlx.portable_host.packages import (
    BINARY_ENTRIES,
    BOOLEAN_CAST_ENTRIES,
    BOOLEAN_COPY_ENTRY,
    CAST_ENTRIES,
    COMPARISON_ENTRIES,
    COPY_ENTRY,
    ENTRIES,
    LOGICAL_NOT_ENTRY,
    UNARY_ENTRIES,
)
from demos.integrations.mlx.portable_host.reduction_packages import (
    COLUMN_ENTRIES,
)
from demos.integrations.mlx.portable_host.reduction_packages import (
    ENTRIES as REDUCTION_ENTRIES,
)
from demos.integrations.mlx.portable_host.reduction_packages import ROW_ENTRIES
from demos.integrations.mlx.portable_host.reduction_packages import (
    load_index as load_reduction_index,
)
from demos.integrations.mlx.portable_host.small_row_packages import (
    SMALL_ROW_ENTRIES,
    SmallRowPackageCache,
)


class Buffer(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("dtype", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("count", ctypes.c_uint64),
        ("output", ctypes.c_uint32),
    ]


DISPATCH_VERSION = 3


class Launch(ctypes.Structure):
    _fields_ = [
        ("workgroup_count", ctypes.c_uint32 * 3),
        ("workgroup_size", ctypes.c_uint32 * 3),
        ("thread_grid_size", ctypes.c_uint32 * 3),
    ]

    def execution(self):
        count, size = list(self.workgroup_count), list(self.workgroup_size)
        if (
            any(not 1 <= value <= 65535 for value in count)
            or any(not 1 <= value <= 1024 for value in size)
            or math.prod(size) > 1024
        ):
            raise ValueError("Native launch geometry exceeds its bounds")
        execution = {"workgroupCount": count, "workgroupSize": size}
        exact = list(self.thread_grid_size)
        if any(exact):
            if any(value < 1 for value in exact) or any(
                (extent + width - 1) // width != groups
                for extent, width, groups in zip(exact, size, count)
            ):
                raise ValueError("Native exact grid does not match its covering groups")
            execution["threadGridSize"] = exact
        return execution


CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.POINTER(Buffer),
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.POINTER(Launch),
    ctypes.c_void_p,
    ctypes.c_size_t,
)
TYPES = {
    "bool_": ctypes.c_uint8,
    "float32": ctypes.c_float,
    "int32": ctypes.c_int32,
    "uint32": ctypes.c_uint32,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,
}
_installed_runtime = None
COPY_GUARD = [0x6A15BEEF] * 32
BOOLEAN_GUARD = [index % 2 == 0 for index in range(32)]
ALL_CAST_ENTRIES = {**CAST_ENTRIES, **BOOLEAN_CAST_ENTRIES}


def physical_dtype(dtype, target):
    if dtype == "bool_":
        return "bool" if target == "metal" else "uint32"
    return dtype


def boolean_values(values, dtype):
    expected_type = bool if dtype == "bool" else int
    if any(type(value) is not expected_type or value not in (0, 1) for value in values):
        raise ValueError("Boolean storage must contain canonical zero or one values")
    return [bool(value) if dtype == "bool" else int(value) for value in values]


def wire_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else "+infinity" if value > 0 else "-infinity"
    return value


class HostRuntime:
    def __init__(self, directory, trace, *, reductions=None, mlx_root=None):
        self.directory = Path(directory).resolve()
        self.trace = Path(trace).resolve()
        index = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        self.target = index["target"]
        self.small_rows = (
            SmallRowPackageCache(mlx_root, self.directory / "small-rows", self.target)
            if mlx_root is not None
            else None
        )
        self.descriptors = index["descriptors"]
        if set(self.descriptors) != set(ENTRIES):
            raise ValueError("Packages must contain the exact supported entry set")
        directories = (
            [self.directory / "reductions"]
            if reductions is None
            else reductions if isinstance(reductions, (list, tuple)) else [reductions]
        )
        if not directories:
            raise ValueError("At least one reduction package directory is required")
        self.reduction_descriptors = {}
        self.reduction_directories = {}
        self.dispatch_count = 0
        for directory in directories:
            directory = Path(directory).resolve()
            if not (directory / "index.json").is_file():
                if reductions is None:
                    continue
                raise ValueError("Reduction package index is missing")
            reduction_index = load_reduction_index(directory, self.target)
            descriptors = reduction_index["descriptors"]
            if self.reduction_descriptors.keys() & descriptors.keys():
                raise ValueError("Reduction packages contain duplicate variants")
            self.reduction_descriptors.update(descriptors)
            self.reduction_directories.update({key: directory for key in descriptors})
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        if self.target == "opengl":
            adapter = OpenGLRuntimeParityAdapter(
                runtime=OpenGLComputeRuntime(context_backends=("egl",))
            )
        elif self.target == "directx":
            adapter = DirectXRuntimeParityAdapter(runtime=DirectXComputeRuntime())
        elif self.target == "metal":
            adapter = MetalRuntimeParityAdapter()
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
        result = register(DISPATCH_VERSION, self.callback)
        if result:
            raise RuntimeError(
                f"MLX rejected the native callback registration: {result}"
            )
        _installed_runtime = self
        mx.set_default_device(mx.gpu)

    def _dispatch(self, entry, buffers, count, threads, launch, error, capacity):
        try:
            if not launch:
                raise ValueError("Native launch geometry is missing")
            self.dispatch(
                entry.decode("ascii"), buffers, count, threads, launch=launch.contents
            )
            return 0
        except Exception as exception:
            message = str(exception).encode("utf-8")
            if error and capacity:
                payload = message[: capacity - 1] + b"\0"
                ctypes.memmove(error, payload, len(payload))
            return 1

    def dispatch(self, entry, buffers, count, threads, *, launch=None):
        small_row = entry in SMALL_ROW_ENTRIES
        row_reduction = entry in ROW_ENTRIES or small_row
        column_reduction = entry in COLUMN_ENTRIES
        shaped_reduction = row_reduction or column_reduction
        layout_module = (
            column_reduction_layout if column_reduction else row_reduction_layout
        )
        reduction = entry in REDUCTION_ENTRIES or shaped_reduction
        if entry not in self.descriptors and not reduction:
            raise ValueError(f"No translated package for {entry}")
        if reduction and launch is None:
            raise ValueError("Native reductions require explicit launch geometry")
        copy = entry in {COPY_ENTRY, BOOLEAN_COPY_ENTRY}
        binary = entry in BINARY_ENTRIES
        comparison = entry in COMPARISON_ENTRIES
        binary_operation = binary or comparison
        cast = entry in ALL_CAST_ENTRIES
        if (
            count
            != (
                len(layout_module.signature(entry))
                if shaped_reduction
                else 8 if copy else 4 if binary_operation or reduction else 3
            )
            or not buffers
            or not 0 < threads <= 65535
        ):
            raise ValueError("Invalid or unsupported native dispatch dimensions")
        execution = launch.execution() if launch is not None else None
        if reduction and not small_row:
            key = f'w{execution["workgroupSize"][0]}/{entry}'
            if key not in self.reduction_descriptors:
                raise ValueError(f"No translated reduction variant for {key}")
            descriptor = self.reduction_descriptors[key]
            package_directory = self.reduction_directories[key] / "package"
        elif not small_row:
            descriptor = self.descriptors[entry]
            package_directory = self.directory / "package"
        logical_not = entry == LOGICAL_NOT_ENTRY
        unary = entry in UNARY_ENTRIES or logical_not
        if shaped_reduction:
            names = set(layout_module.signature(entry))
        elif reduction:
            names = {"in", "out", "in_size", "row_size"}
        elif copy:
            names = set(copy_layout.DTYPES)
        elif binary_operation:
            names = {"a", "b", "c", "size"}
        elif cast:
            names = {"src", "dst", "size"}
        else:
            names = {"in", "size", "out"} if unary else {"start", "step", "out"}
        output_name = "dst" if copy or cast else "c" if binary_operation else "out"
        supplied = {}
        for index in range(count):
            buffer = buffers[index]
            if not buffer.name or not buffer.dtype:
                raise ValueError("Native buffer identity is missing")
            name = buffer.name.decode("ascii")
            dtype = buffer.dtype.decode("ascii")
            if (
                name not in names
                or name in supplied
                or dtype not in TYPES
                or not buffer.data
            ):
                raise ValueError("Invalid native buffer")
            if reduction:
                expected = {"in": threads, "out": execution["workgroupCount"][1]}.get(
                    name, 1
                )
            else:
                expected = (
                    threads
                    if name == output_name
                    or (unary and name == "in")
                    or (binary_operation and name in {"a", "b"})
                    or (cast and name == "src")
                    else 1
                )
            if (
                not copy and not shaped_reduction and buffer.count != expected
            ) or buffer.output != int(name == output_name):
                raise ValueError("Native buffer shape or direction does not match")
            if unary and dtype != (
                "uint32" if name == "size" else "bool_" if logical_not else "float32"
            ):
                raise ValueError("Native unary buffer dtype does not match")
            if binary and dtype != (
                "uint32" if name == "size" else BINARY_ENTRIES[entry]
            ):
                raise ValueError("Native binary buffer dtype does not match")
            if comparison and dtype != (
                "uint32"
                if name == "size"
                else "bool_" if name == "c" else COMPARISON_ENTRIES[entry]
            ):
                raise ValueError("Native comparison buffer dtype does not match")
            if cast and dtype != (
                "uint32"
                if name == "size"
                else ALL_CAST_ENTRIES[entry][0 if name == "src" else 1]
            ):
                raise ValueError("Native cast buffer dtype does not match")
            if (
                reduction
                and not shaped_reduction
                and dtype
                != (
                    "uint64"
                    if name in {"in_size", "row_size"}
                    else REDUCTION_ENTRIES[entry]
                )
            ):
                raise ValueError("Native reduction buffer dtype does not match")
            supplied[name] = buffer
        if set(supplied) != names:
            raise ValueError("Native buffer names do not match the operation")
        shaped_metadata = None
        if shaped_reduction:
            shaped_metadata = layout_module.validate(
                entry, supplied, threads, execution
            )
        elif reduction:
            reduction_layout.validate(supplied, threads, execution)
        grid = (
            execution["workgroupCount"]
            if reduction
            else (
                copy_layout.geometry(
                    supplied,
                    threads,
                    dtype="bool_" if entry == BOOLEAN_COPY_ENTRY else "uint32",
                )
                if copy
                else [threads, 1, 1]
            )
        )
        execution = (
            launch.execution()
            if launch is not None
            else {"workgroupCount": grid, "workgroupSize": [1, 1, 1]}
        )
        if not reduction and execution != {
            "workgroupCount": grid,
            "workgroupSize": [1, 1, 1],
        }:
            raise ValueError("Native launch geometry does not match the operation")
        if small_row:
            if self.small_rows is None:
                raise ValueError(
                    "Small-row dispatch requires the pinned MLX source root"
                )
            region_packages = self.small_rows.get(
                entry,
                thread_grid_size=execution["threadGridSize"],
                workgroup_size=execution["workgroupSize"],
            )
            descriptor, package_directory = region_packages[0]
        if (unary or binary_operation or cast) and ctypes.cast(
            supplied["size"].data, ctypes.POINTER(ctypes.c_uint32)
        )[0] != threads:
            raise ValueError("Native operation size does not match the launch")
        guard = COPY_GUARD
        guarded = copy or binary_operation or cast or logical_not or reduction
        output_dtype = supplied[output_name].dtype.decode("ascii")
        if output_dtype == "bool_":
            guard = (
                BOOLEAN_GUARD
                if self.target == "metal"
                else [int(value) for value in BOOLEAN_GUARD]
            )
        if (
            (reduction and output_dtype == "float32")
            or (binary and BINARY_ENTRIES[entry] == "float32")
            or (cast and ALL_CAST_ENTRIES[entry][1] == "float32")
        ):
            guard = [
                ctypes.c_float.from_buffer_copy(ctypes.c_uint32(word)).value
                for word in COPY_GUARD
            ]
        inputs, outputs, destinations = {}, {}, {}
        matched = set()
        binding_names = set()
        for binding in descriptor["bindings"]:
            if "executionInput" in binding.get("provenance", {}):
                continue
            layout = binding["scalarLayout"]
            member = layout.get("memberName", binding["name"])
            if self.target == "directx":
                member = member.removeprefix(entry.rstrip("_") + "_")
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
            storage = physical_dtype(dtype, self.target)
            if layout["elementType"] != storage or layout["elementStrideBytes"] != (
                1 if storage == "bool" else ctypes.sizeof(TYPES[storage])
            ):
                raise ValueError("Native and reflected buffer layouts disagree")
            ctype = TYPES[dtype]
            view = ctypes.cast(
                buffer.data, ctypes.POINTER(ctype * buffer.count)
            ).contents
            if comparison and dtype == "float32" and self.target != "metal":
                words = ctypes.cast(
                    buffer.data, ctypes.POINTER(ctypes.c_uint32 * buffer.count)
                ).contents
                if any(0 < (word & 0x7FFFFFFF) < 0x00800000 for word in words):
                    raise ValueError(
                        "Subnormal float comparison parity is not established for "
                        f"{self.target}; see CrossGL/crosstl#2000"
                    )
            values = (
                [0] * buffer.count
                if buffer.output
                else [wire_value(value) for value in view]
            )
            if dtype == "bool_":
                values = boolean_values(values, "uint32")
                if storage == "bool":
                    values = [bool(value) for value in values]
            value = {
                "dtype": storage,
                "shape": [buffer.count],
                "values": values,
            }
            if guarded and buffer.output:
                value["shape"] = [buffer.count + len(guard)]
                value["values"].extend(guard)
                inputs[binding["name"]] = value
            if buffer.output:
                outputs[binding["name"]] = value
                destinations[binding["name"]] = (buffer, ctype)
            else:
                inputs[binding["name"]] = value
        if matched != set(supplied):
            raise ValueError("Reflected bindings do not cover the operation")
        if small_row and self.target != "metal":
            adapter = self.executor.runtime_adapter
            with prepare_native_loader_dispatch_regions(
                region_packages,
                inputs,
                outputs,
                thread_grid_size=execution["threadGridSize"],
                source_workgroup_size=execution["workgroupSize"],
                adapter=adapter,
            ) as requests:
                records = []
                for (region_descriptor, root), request in zip(
                    region_packages, requests
                ):
                    module_bytes = request.module_path.read_bytes()
                    module_hash = hashlib.sha256(module_bytes).hexdigest()
                    module = (
                        self.trace.parent
                        / "native-modules"
                        / (module_hash + request.module_path.suffix)
                    )
                    module.parent.mkdir(parents=True, exist_ok=True)
                    module.write_bytes(module_bytes)
                    records.append(
                        {
                            "artifact": region_descriptor["artifact"],
                            "packageRoot": str(root),
                            "provenance": region_descriptor["provenance"],
                            "moduleHash": module_hash,
                            "moduleFile": str(module),
                            "workgroupCount": list(request.dispatch.workgroup_count),
                            "workgroupSize": list(request.dispatch.workgroup_size),
                        }
                    )
                native_outputs = adapter.runtime.dispatch_sequence(None, None, requests)
                result = RuntimeExecutorResult(
                    outputs=native_outputs,
                    details={
                        "runtime": adapter.runtime.name,
                        "regions": records,
                    },
                )
        else:
            request = build_native_loader_dispatch_request(
                descriptor,
                package_directory,
                inputs,
                outputs,
                execution,
                expected_target=self.target,
            )
            result = self.executor.run(request)
        if result.status != "ok" or set(result.outputs) != set(destinations):
            raise RuntimeError("Native executor did not return the required outputs")
        for name, (buffer, ctype) in destinations.items():
            output = result.outputs[name]
            size = buffer.count + (len(guard) if guarded else 0)
            dtype = buffer.dtype.decode("ascii")
            storage = physical_dtype(dtype, self.target)
            if output["dtype"] != storage or output["shape"] != [size]:
                raise RuntimeError("Native readback layout does not match the output")
            if len(output["values"]) != size:
                raise RuntimeError("Native readback size does not match the output")
            if dtype == "bool_":
                boolean_values(output["values"], storage)
            if guarded and output["values"][buffer.count :] != guard:
                raise RuntimeError("Native operation changed the output buffer guard")
            values = (ctype * buffer.count)(
                *(
                    float(value) if ctype is ctypes.c_float else value
                    for value in output["values"][: buffer.count]
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
                        **execution,
                        "dispatchVersion": DISPATCH_VERSION,
                        "artifact": descriptor["artifact"],
                        "details": result.details,
                        **(
                            {
                                "reductionGuardValues": output["values"][
                                    buffer.count :
                                ],
                                "reductionValues": output["values"][: buffer.count],
                                "reductionMetadata": (
                                    shaped_metadata
                                    if shaped_reduction
                                    else {
                                        name: ctypes.cast(
                                            supplied[name].data,
                                            ctypes.POINTER(ctypes.c_uint64),
                                        )[0]
                                        for name in ("in_size", "row_size")
                                    }
                                ),
                            }
                            if reduction
                            else {}
                        ),
                        **(
                            {"copyGuardWords": output["values"][buffer.count :]}
                            if copy
                            else {}
                        ),
                        **(
                            {"binaryGuardValues": output["values"][buffer.count :]}
                            if binary
                            else {}
                        ),
                        **(
                            {"castGuardValues": output["values"][buffer.count :]}
                            if cast
                            else {}
                        ),
                        **(
                            {
                                "booleanGuardValues": output["values"][buffer.count :],
                                "physicalBooleanType": physical_dtype(
                                    "bool_", self.target
                                ),
                            }
                            if output_dtype == "bool_"
                            else {}
                        ),
                    }
                )
                + "\n"
            )
        self.dispatch_count = getattr(self, "dispatch_count", 0) + 1
