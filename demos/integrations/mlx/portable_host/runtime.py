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
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from crosstl.project.runtime_verification import (
    DirectXRuntimeParityAdapter,
    MetalRuntimeParityAdapter,
    OpenGLRuntimeParityAdapter,
    RuntimeExecutorResult,
    RuntimeParityExecutor,
    RuntimeTestAdapterSpec,
)
from crosstl.translator.resource_storage import encoded_storage_dtype
from demos.integrations.mlx.portable_host import (
    bfloat_storage,
    column_reduction_layout,
    copy_layout,
    gather_dispatch,
    half_storage,
    quantization_dispatch,
    quantization_layout,
    random_dispatch,
    reduction_layout,
    row_reduction_layout,
    slice_update_layout,
)
from demos.integrations.mlx.portable_host.gather_packages import (
    GatherPackageCache,
)
from demos.integrations.mlx.portable_host.gather_packages import (
    signature as gather_signature,
)
from demos.integrations.mlx.portable_host.packages import (
    ABSOLUTE_ENTRIES,
    ARANGE_ENTRIES,
    BFLOAT_ABSOLUTE_ENTRIES,
    BFLOAT_BINARY_ENTRIES,
    BFLOAT_CAST_ENTRIES,
    BFLOAT_COMPARISON_ENTRIES,
    BFLOAT_COPY_ENTRY,
    BFLOAT_ENTRIES,
    BINARY_ENTRIES,
    BITWISE_ENTRIES,
    BITWISE_INVERT_ENTRIES,
    BITWISE_PACKAGE_ENTRIES,
    BOOLEAN_CAST_ENTRIES,
    BOOLEAN_COPY_ENTRY,
    CAST_ENTRIES,
    COMPARISON_ENTRIES,
    COPY_ENTRY,
    ENTRIES,
    HALF_ABSOLUTE_ENTRIES,
    HALF_ARITHMETIC_ENTRIES,
    HALF_BINARY_ENTRIES,
    HALF_CAST_ENTRIES,
    HALF_COMPARISON_ENTRIES,
    HALF_COPY_ENTRY,
    HALF_ENTRIES,
    INTEGER64_ABSOLUTE_ENTRIES,
    INTEGER64_BINARY_ENTRIES,
    INTEGER64_CAST_ENTRIES,
    INTEGER64_COMPARISON_ENTRIES,
    INTEGER64_COPY_ENTRIES,
    INTEGER64_ENTRIES,
    LOGICAL_NOT_ENTRY,
    SELECTION_ENTRIES,
    SLICE_UPDATE_ENTRIES,
    UNARY_ENTRIES,
)
from demos.integrations.mlx.portable_host.quantization_packages import (
    QuantizationPackageCache,
)
from demos.integrations.mlx.portable_host.random_packages import (
    ENTRIES as RANDOM_ENTRIES,
)
from demos.integrations.mlx.portable_host.random_packages import (
    load_index as load_random_index,
)
from demos.integrations.mlx.portable_host.reduction_packages import (
    COLUMN_ENTRIES,
)
from demos.integrations.mlx.portable_host.reduction_packages import (
    ENTRIES as REDUCTION_ENTRIES,
)
from demos.integrations.mlx.portable_host.reduction_packages import (
    INIT_ENTRIES,
    ROW_ENTRIES,
)
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
ENTRY_AVAILABLE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p)
TYPES = {
    "bool_": ctypes.c_uint8,
    "float32": ctypes.c_float,
    "float16": ctypes.c_uint16,
    "bfloat16": ctypes.c_uint16,
    "uint16": ctypes.c_uint16,
    "int32": ctypes.c_int32,
    "uint32": ctypes.c_uint32,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,
}
_installed_runtime = None
COPY_GUARD = [0x6A15BEEF] * 32
BOOLEAN_GUARD = [index % 2 == 0 for index in range(32)]
ALL_CAST_ENTRIES = {
    **CAST_ENTRIES,
    **BOOLEAN_CAST_ENTRIES,
    **INTEGER64_CAST_ENTRIES,
    **HALF_CAST_ENTRIES,
    **BFLOAT_CAST_ENTRIES,
}
ALL_BINARY_ENTRIES = {
    **BINARY_ENTRIES,
    **BITWISE_ENTRIES,
    **INTEGER64_BINARY_ENTRIES,
    **HALF_BINARY_ENTRIES,
    **BFLOAT_BINARY_ENTRIES,
}
ALL_COMPARISON_ENTRIES = {
    **COMPARISON_ENTRIES,
    **INTEGER64_COMPARISON_ENTRIES,
    **HALF_COMPARISON_ENTRIES,
    **BFLOAT_COMPARISON_ENTRIES,
}
ALL_ABSOLUTE_ENTRIES = {
    **ABSOLUTE_ENTRIES,
    **INTEGER64_ABSOLUTE_ENTRIES,
    **HALF_ABSOLUTE_ENTRIES,
    **BFLOAT_ABSOLUTE_ENTRIES,
}
ALL_HALF_ENTRIES = (*HALF_ENTRIES, *HALF_ARITHMETIC_ENTRIES)
ALL_COPY_ENTRIES = {
    COPY_ENTRY: "uint32",
    BOOLEAN_COPY_ENTRY: "bool_",
    HALF_COPY_ENTRY: "float16",
    BFLOAT_COPY_ENTRY: "bfloat16",
    **INTEGER64_COPY_ENTRIES,
}


def physical_dtype(dtype, target):
    if dtype == "bfloat16":
        return {"metal": "bfloat16", "directx": "uint16", "opengl": "float32"}[target]
    if dtype == "float16" and target == "opengl":
        return "float32"
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
    def __init__(
        self,
        directory,
        trace,
        *,
        reductions=None,
        mlx_root=None,
        bitwise=None,
        selection=None,
        absolute=None,
        integer64=None,
        slice_updates=None,
        random=None,
        half=None,
        half_arithmetic=None,
        bfloat=None,
        retain_native_modules=False,
    ):
        if type(retain_native_modules) is not bool:
            raise ValueError("Native module retention must be a Boolean")
        self.retain_native_modules = retain_native_modules
        self.directory = Path(directory).resolve()
        self.trace = Path(trace).resolve()
        index = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        self.target = index["target"]
        self.small_rows = (
            SmallRowPackageCache(mlx_root, self.directory / "small-rows", self.target)
            if mlx_root is not None
            else None
        )
        self.gathers = (
            GatherPackageCache(mlx_root, self.directory / "gather", self.target)
            if mlx_root is not None
            else None
        )
        self.quantization = (
            QuantizationPackageCache(
                mlx_root, self.directory / "quantization", self.target
            )
            if mlx_root is not None
            else None
        )
        self.descriptors = index["descriptors"]
        if set(self.descriptors) != set(ENTRIES):
            raise ValueError("Packages must contain the exact supported entry set")
        self.random_directory = Path(random).resolve() if random is not None else None
        if self.random_directory is not None:
            self.descriptors.update(
                load_random_index(self.random_directory, self.target)
            )
        self.bitwise_directory = (
            Path(bitwise).resolve() if bitwise is not None else None
        )
        self.selection_directory = (
            Path(selection).resolve() if selection is not None else None
        )
        self.absolute_directory = (
            Path(absolute).resolve() if absolute is not None else None
        )
        self.integer64_directory = (
            Path(integer64).resolve() if integer64 is not None else None
        )
        self.slice_update_directory = (
            Path(slice_updates).resolve() if slice_updates is not None else None
        )
        self.half_directory = Path(half).resolve() if half is not None else None
        self.bfloat_directory = Path(bfloat).resolve() if bfloat is not None else None
        self.half_arithmetic_directory = (
            Path(half_arithmetic).resolve() if half_arithmetic is not None else None
        )
        for family, directory, entries in (
            ("bitwise", self.bitwise_directory, BITWISE_PACKAGE_ENTRIES),
            ("selection", self.selection_directory, SELECTION_ENTRIES),
            ("absolute", self.absolute_directory, ABSOLUTE_ENTRIES),
            ("integer64", self.integer64_directory, INTEGER64_ENTRIES),
            ("slice-update", self.slice_update_directory, SLICE_UPDATE_ENTRIES),
            ("half", self.half_directory, HALF_ENTRIES),
            ("bfloat", self.bfloat_directory, BFLOAT_ENTRIES),
            (
                "half-arithmetic",
                self.half_arithmetic_directory,
                HALF_ARITHMETIC_ENTRIES,
            ),
        ):
            if directory is None:
                continue
            operators = json.loads(
                (directory / "index.json").read_text(encoding="utf-8")
            )
            if (
                not isinstance(operators, dict)
                or operators.get("family") != family
                or operators.get("target") != self.target
                or not isinstance(operators.get("descriptors"), dict)
                or set(operators.get("descriptors", {})) != set(entries)
                or any(
                    not isinstance(value, dict) or value.get("target") != self.target
                    for value in operators["descriptors"].values()
                )
            ):
                raise ValueError(
                    f"{family.title()} packages must match the target and exact entry set"
                )
            self.descriptors.update(operators["descriptors"])
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
        self.entry_available = ENTRY_AVAILABLE(self._entry_available)
        self.library = None

    def _entry_available(self, entry):
        try:
            name = entry.decode("ascii")
            if name in self.descriptors:
                return 1
            if name.startswith(("gather", "scatter")) and self.gathers is not None:
                gather_signature(name)
                return 1
            if (
                name.startswith(quantization_layout.ENTRY_PREFIXES)
                and self.quantization is not None
            ):
                quantization_layout.signature(name)
                return 1
            return 0
        except (AttributeError, UnicodeDecodeError, ValueError):
            return 0

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
        register = self.library.crosstl_mlx_register_runtime
        register.argtypes = [ctypes.c_uint32, CALLBACK, ENTRY_AVAILABLE]
        register.restype = ctypes.c_int
        result = register(DISPATCH_VERSION, self.callback, self.entry_available)
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
        if entry.startswith(quantization_layout.ENTRY_PREFIXES):
            return quantization_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry in RANDOM_ENTRIES:
            return random_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry.startswith(("gather", "scatter")):
            return gather_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        initialization = entry in INIT_ENTRIES
        small_row = entry in SMALL_ROW_ENTRIES
        row_reduction = entry in ROW_ENTRIES or small_row
        column_reduction = entry in COLUMN_ENTRIES
        shaped_reduction = row_reduction or column_reduction
        layout_module = (
            column_reduction_layout if column_reduction else row_reduction_layout
        )
        reduction = entry in REDUCTION_ENTRIES or shaped_reduction or initialization
        if entry not in self.descriptors and not reduction:
            raise ValueError(f"No translated package for {entry}")
        if reduction and launch is None:
            raise ValueError("Native reductions require explicit launch geometry")
        copy = entry in ALL_COPY_ENTRIES
        slice_update = entry in SLICE_UPDATE_ENTRIES
        invert = entry in BITWISE_INVERT_ENTRIES
        absolute = entry in ALL_ABSOLUTE_ENTRIES
        bitwise = entry in BITWISE_PACKAGE_ENTRIES
        binary = entry in ALL_BINARY_ENTRIES
        comparison = entry in ALL_COMPARISON_ENTRIES
        binary_operation = binary or comparison
        selection = entry in SELECTION_ENTRIES
        cast = entry in ALL_CAST_ENTRIES
        if (
            count != (
                1
                if initialization
                else (
                    len(layout_module.signature(entry))
                    if shaped_reduction
                    else (
                        8
                        if copy or slice_update
                        else (
                            5
                            if selection
                            else 4 if binary_operation or reduction else 3
                        )
                    )
                )
            )
            or not buffers
            or not 0
            < threads
            <= (
                copy_layout.MAX_DESTINATION_ELEMENTS
                if copy
                else (
                    reduction_layout.MAX_ELEMENTS
                    if entry in REDUCTION_ENTRIES
                    else 65535
                )
            )
        ):
            raise ValueError("Invalid or unsupported native dispatch dimensions")
        execution = launch.execution() if launch is not None else None
        if initialization and execution != {
            "workgroupCount": [threads, 1, 1],
            "workgroupSize": [1, 1, 1],
        }:
            raise ValueError(
                "Reduction initialization requires one invocation per output"
            )
        if reduction and not small_row:
            key = f'w{execution["workgroupSize"][0]}/{entry}'
            if key not in self.reduction_descriptors:
                raise ValueError(f"No translated reduction variant for {key}")
            descriptor = self.reduction_descriptors[key]
            package_directory = self.reduction_directories[key] / "package"
        elif not small_row:
            descriptor = self.descriptors[entry]
            package_directory = (
                self.bfloat_directory
                if entry in BFLOAT_ENTRIES
                else (
                    (
                        self.half_directory
                        if entry in HALF_ENTRIES
                        else self.half_arithmetic_directory
                    )
                    if entry in ALL_HALF_ENTRIES
                    else (
                        self.slice_update_directory
                        if slice_update
                        else (
                            self.integer64_directory
                            if entry in INTEGER64_ENTRIES
                            else (
                                self.bitwise_directory
                                if bitwise
                                else (
                                    self.selection_directory
                                    if selection
                                    else (
                                        self.absolute_directory
                                        if absolute
                                        else self.directory
                                    )
                                )
                            )
                        )
                    )
                )
            ) / "package"
        logical_not = entry == LOGICAL_NOT_ENTRY
        unary = entry in UNARY_ENTRIES or logical_not or invert or absolute
        if initialization:
            names = {"out"}
        elif shaped_reduction:
            names = set(layout_module.signature(entry))
        elif reduction:
            names = {"in", "out", "in_size", "row_size"}
        elif copy:
            names = set(copy_layout.DTYPES)
        elif slice_update:
            names = set(slice_update_layout.DTYPES)
        elif binary_operation:
            names = {"a", "b", "c", "size"}
        elif selection:
            names = {"a", "b", "c", "d", "size"}
        elif cast:
            names = {"src", "dst", "size"}
        else:
            names = {"in", "size", "out"} if unary else {"start", "step", "out"}
        output_name = (
            "dst"
            if copy or cast
            else "c" if binary_operation else "d" if selection else "out"
        )
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
            if entry in ARANGE_ENTRIES and dtype != entry.removeprefix("arange"):
                raise ValueError("Native arange buffer dtype does not match its entry")
            if initialization:
                expected = threads
                if dtype != INIT_ENTRIES[entry]:
                    raise ValueError(
                        "Reduction initialization dtype does not match its entry"
                    )
            elif reduction:
                expected = {"in": threads, "out": execution["workgroupCount"][1]}.get(
                    name, 1
                )
            else:
                expected = (
                    threads
                    if name == output_name
                    or (unary and name == "in")
                    or (binary_operation and name in {"a", "b"})
                    or (selection and name in {"a", "b", "c"})
                    or (cast and name == "src")
                    else 1
                )
            directions = (
                {copy_layout.INOUT}
                if slice_update and name == output_name
                else (
                    {1, copy_layout.INOUT}
                    if copy and name == output_name
                    else {int(name == output_name)}
                )
            )
            if (
                not copy
                and not slice_update
                and not shaped_reduction
                and buffer.count != expected
            ) or buffer.output not in directions:
                raise ValueError("Native buffer shape or direction does not match")
            if unary and dtype != (
                "uint32"
                if name == "size"
                else (
                    BITWISE_INVERT_ENTRIES[entry]
                    if invert
                    else (
                        ALL_ABSOLUTE_ENTRIES[entry]
                        if absolute
                        else "bool_" if logical_not else "float32"
                    )
                )
            ):
                raise ValueError("Native unary buffer dtype does not match")
            if binary and dtype != (
                "uint32" if name == "size" else ALL_BINARY_ENTRIES[entry]
            ):
                raise ValueError("Native binary buffer dtype does not match")
            if comparison and dtype != (
                "uint32"
                if name == "size"
                else "bool_" if name == "c" else ALL_COMPARISON_ENTRIES[entry]
            ):
                raise ValueError("Native comparison buffer dtype does not match")
            if selection and dtype != (
                "uint32"
                if name == "size"
                else "bool_" if name == "a" else SELECTION_ENTRIES[entry]
            ):
                raise ValueError("Native selection buffer dtype does not match")
            if cast and dtype != (
                "uint32"
                if name == "size"
                else ALL_CAST_ENTRIES[entry][0 if name == "src" else 1]
            ):
                raise ValueError("Native cast buffer dtype does not match")
            if (
                reduction
                and not shaped_reduction
                and not initialization
                and dtype != (
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
        elif reduction and not initialization:
            reduction_layout.validate(
                supplied, threads, execution, REDUCTION_ENTRIES[entry]
            )
        copy_metadata = (
            copy_layout.validate(
                supplied,
                threads,
                dtype=ALL_COPY_ENTRIES[entry],
            )
            if copy
            else None
        )
        slice_metadata = (
            slice_update_layout.validate(
                supplied, threads, dtype=SLICE_UPDATE_ENTRIES[entry]
            )
            if slice_update
            else None
        )
        grid = (
            execution["workgroupCount"]
            if reduction
            else (copy_metadata["workgroupCount"] if copy else [threads, 1, 1])
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
        if (unary or binary_operation or cast or selection) and ctypes.cast(
            supplied["size"].data, ctypes.POINTER(ctypes.c_uint32)
        )[0] != threads:
            raise ValueError("Native operation size does not match the launch")
        if bitwise and entry.startswith(("vv_LeftShift", "vv_RightShift")):
            buffer = supplied["b"]
            counts = ctypes.cast(
                buffer.data, ctypes.POINTER(TYPES[BITWISE_ENTRIES[entry]])
            )
            if any(not 0 <= counts[index] < 32 for index in range(threads)):
                raise ValueError("32-bit shifts require counts in [0, 31]")
        guard = COPY_GUARD
        guarded = (
            copy
            or (unary and self.retain_native_modules)
            or binary_operation
            or cast
            or logical_not
            or reduction
            or invert
            or selection
            or absolute
            or slice_update
        )
        output_dtype = supplied[output_name].dtype.decode("ascii")
        bit_storage = (slice_update and output_dtype == "float32") or entry in (
            *HALF_CAST_ENTRIES,
            *BFLOAT_CAST_ENTRIES,
        )
        if output_dtype == "float16":
            guard = half_storage.pack(half_storage.GUARD, self.target)
        if output_dtype == "bfloat16":
            guard = bfloat_storage.pack(bfloat_storage.GUARD, self.target)
        if output_dtype == "bool_":
            guard = (
                BOOLEAN_GUARD
                if self.target == "metal"
                else [int(value) for value in BOOLEAN_GUARD]
            )
        if not bit_storage and (
            (reduction and output_dtype == "float32")
            or (binary and ALL_BINARY_ENTRIES[entry] == "float32")
            or (cast and ALL_CAST_ENTRIES[entry][1] == "float32")
            or (selection and output_dtype == "float32")
            or (unary and self.retain_native_modules and output_dtype == "float32")
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
            reflected_storage = encoded_storage_dtype(
                layout,
                target=self.target,
                resource_kind=binding.get("kind", ""),
                logical_dtype=storage,
            )
            if layout["elementType"] != reflected_storage or layout[
                "elementStrideBytes"
            ] != (1 if storage == "bool" else ctypes.sizeof(TYPES[storage])):
                raise ValueError("Native and reflected buffer layouts disagree")
            if layout["elementSizeBytes"] != (
                1 if storage == "bool" else ctypes.sizeof(TYPES[storage])
            ):
                raise ValueError("Native and reflected element sizes disagree")
            ctype = TYPES[dtype]
            view = ctypes.cast(
                buffer.data, ctypes.POINTER(ctype * buffer.count)
            ).contents
            if (
                comparison
                and dtype in {"float32", "bfloat16"}
                and self.target != "metal"
            ):
                word_type = ctypes.c_uint16 if dtype == "bfloat16" else ctypes.c_uint32
                words = ctypes.cast(
                    buffer.data, ctypes.POINTER(word_type * buffer.count)
                ).contents
                mask, normal = (
                    (0x7FFF, 0x80) if dtype == "bfloat16" else (0x7FFFFFFF, 0x00800000)
                )
                if any(0 < (word & mask) < normal for word in words):
                    raise ValueError(
                        "Subnormal float comparison parity is not established for "
                        f"{self.target}; see CrossGL/crosstl#2000"
                    )
            if bit_storage and dtype == "float32":
                values = list(
                    ctypes.cast(
                        buffer.data, ctypes.POINTER(ctypes.c_uint32 * buffer.count)
                    ).contents
                )
            else:
                values = (
                    [0] * buffer.count
                    if buffer.output == 1
                    else [wire_value(value) for value in view]
                )
            if initialization:
                # Unwritten outputs must differ from the reduction identity.
                initial_value = int("sum" in entry or entry == "init_reduce_orbool_")
                values = [initial_value] * buffer.count
            if (bitwise or absolute) and buffer.output:
                values = [
                    (
                        half_storage.GUARD[0]
                        if dtype == "float16"
                        else (
                            bfloat_storage.GUARD[0]
                            if dtype == "bfloat16"
                            else 1 if dtype == "bool_" else COPY_GUARD[0]
                        )
                    )
                ] * buffer.count
            if selection and buffer.output:
                values = [
                    int(guard[0]) if dtype == "bool_" else guard[0]
                ] * buffer.count
            if dtype == "bool_":
                values = boolean_values(values, "uint32")
                if storage == "bool":
                    values = [bool(value) for value in values]
            if dtype == "float16":
                values = half_storage.pack(values, self.target)
            if dtype == "bfloat16":
                values = bfloat_storage.pack(values, self.target)
            value = {
                "dtype": storage,
                "shape": [buffer.count],
                "values": values,
            }
            if bit_storage and dtype == "float32":
                value["encoding"] = FLOAT32_BITS
            elif dtype == "float16":
                value["encoding"] = half_storage.encoding(self.target)
            elif dtype == "bfloat16" and bfloat_storage.encoding(self.target):
                value["encoding"] = bfloat_storage.encoding(self.target)
            if guarded and buffer.output:
                value["shape"] = [buffer.count + len(guard)]
                value["values"].extend(guard)
                inputs[binding["name"]] = value
            if buffer.output:
                outputs[binding["name"]] = (
                    {key: item for key, item in value.items() if key != "values"}
                    if slice_update
                    else value
                )
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
            result = (
                gather_dispatch.execute(self, request)
                if entry in (*ALL_HALF_ENTRIES, *BFLOAT_ENTRIES)
                or self.retain_native_modules
                else self.executor.run(request)
            )
        if result.status != "ok" or set(result.outputs) != set(destinations):
            raise RuntimeError("Native executor did not return the required outputs")
        for name, (buffer, ctype) in destinations.items():
            output = result.outputs[name]
            size = buffer.count + (len(guard) if guarded else 0)
            dtype = buffer.dtype.decode("ascii")
            storage = physical_dtype(dtype, self.target)
            if output["dtype"] != storage or output["shape"] != [size]:
                raise RuntimeError("Native readback layout does not match the output")
            expected_encoding = (
                half_storage.encoding(self.target)
                if dtype == "float16"
                else (
                    bfloat_storage.encoding(self.target)
                    if dtype == "bfloat16"
                    else FLOAT32_BITS if bit_storage else None
                )
            )
            if output.get("encoding") != expected_encoding:
                raise RuntimeError(
                    "Native readback storage encoding does not match the output"
                )
            if len(output["values"]) != size:
                raise RuntimeError("Native readback size does not match the output")
            if dtype == "float16":
                half_storage.unpack(output["values"], self.target)
            elif dtype == "bfloat16":
                bfloat_storage.unpack(output["values"], self.target)
            elif dtype == "bool_":
                boolean_values(output["values"], storage)
            elif (
                bitwise
                or copy
                or absolute
                or slice_update
                or (selection and dtype != "float32")
                or dtype in {"int64", "uint64"}
                or bit_storage
            ):
                bits = ctypes.sizeof(ctype) * 8
                low, high = (
                    (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
                    if dtype.startswith("int")
                    else (0, 2**bits - 1)
                )
                if any(
                    type(value) is not int or not low <= value <= high
                    for value in output["values"]
                ):
                    raise RuntimeError(
                        "Native integer readback is outside its integer type"
                    )
            if guarded and output["values"][buffer.count :] != guard:
                raise RuntimeError("Native operation changed the output buffer guard")
            if copy or slice_update:
                written = set(
                    copy_layout.destination_indices(
                        copy_metadata if copy else slice_metadata
                    )
                )
                initial = inputs[name]["values"]

                def storage_word(value):
                    if dtype == "float32" and not bit_storage:
                        return ctypes.c_uint32.from_buffer_copy(
                            ctypes.c_float(float(value))
                        ).value
                    return value

                if any(
                    storage_word(value) != storage_word(initial[index])
                    for index, value in enumerate(output["values"][: buffer.count])
                    if index not in written
                ):
                    raise RuntimeError(
                        "Native operation changed untouched destination storage"
                    )
            storage_type = (
                ctypes.c_uint32 if bit_storage and dtype == "float32" else ctype
            )
            result_values = output["values"][: buffer.count]
            if dtype == "float16":
                result_values = half_storage.unpack(result_values, self.target)
            if dtype == "bfloat16":
                result_values = bfloat_storage.unpack(result_values, self.target)
            values = (storage_type * buffer.count)(
                *(
                    float(value) if storage_type is ctypes.c_float else value
                    for value in result_values
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
                                (
                                    "bfloatStorage"
                                    if entry in BFLOAT_ENTRIES
                                    else "halfStorage"
                                ): {
                                    "logicalType": output_dtype,
                                    "physicalType": storage,
                                    "encoding": expected_encoding,
                                    "values": output["values"][: buffer.count],
                                    "guardValues": output["values"][buffer.count :],
                                    "logicalWords": list(values),
                                },
                                "inputs": inputs,
                                "packageRoot": str(package_directory),
                            }
                            if entry in (*ALL_HALF_ENTRIES, *BFLOAT_ENTRIES)
                            else {}
                        ),
                        **(
                            {
                                "sliceUpdateValues": output["values"][: buffer.count],
                                "sliceUpdateGuardValues": output["values"][
                                    buffer.count :
                                ],
                                "sliceUpdateMetadata": slice_metadata,
                                **(
                                    {
                                        "inputs": inputs,
                                        "packageRoot": str(package_directory),
                                    }
                                    if self.retain_native_modules
                                    else {}
                                ),
                                **(
                                    {
                                        "sliceUpdateStorageWords": output["values"][
                                            : buffer.count
                                        ],
                                        "sliceUpdateGuardWords": output["values"][
                                            buffer.count :
                                        ],
                                        "sliceUpdateValues": [
                                            wire_value(
                                                ctypes.c_float.from_buffer_copy(
                                                    ctypes.c_uint32(word)
                                                ).value
                                            )
                                            for word in output["values"][: buffer.count]
                                        ],
                                        "sliceUpdateGuardValues": [
                                            ctypes.c_float.from_buffer_copy(
                                                ctypes.c_uint32(word)
                                            ).value
                                            for word in output["values"][buffer.count :]
                                        ],
                                    }
                                    if bit_storage
                                    else {}
                                ),
                            }
                            if slice_update
                            else {}
                        ),
                        **(
                            {
                                "integer64Values": output["values"][: buffer.count],
                                "integer64GuardValues": output["values"][
                                    buffer.count :
                                ],
                            }
                            if entry in INTEGER64_ENTRIES
                            else {}
                        ),
                        **(
                            {
                                "selectionValues": output["values"][: buffer.count],
                                "selectionGuardValues": output["values"][
                                    buffer.count :
                                ],
                            }
                            if selection
                            else {}
                        ),
                        **(
                            {"bitwiseValues": output["values"][: buffer.count]}
                            if bitwise
                            else {}
                        ),
                        **(
                            {"absoluteValues": output["values"][: buffer.count]}
                            if absolute
                            else {}
                        ),
                        **(
                            {"unaryGuardValues": output["values"][buffer.count :]}
                            if unary and guarded
                            else {}
                        ),
                        **(
                            {
                                "unaryValues": output["values"][: buffer.count],
                                "inputs": inputs,
                                "packageRoot": str(package_directory),
                            }
                            if unary and self.retain_native_modules
                            else {}
                        ),
                        **(
                            {"initializationValue": initial_value}
                            if initialization
                            else {}
                        ),
                        **(
                            {
                                "reductionGuardValues": output["values"][
                                    buffer.count :
                                ],
                                "reductionValues": output["values"][: buffer.count],
                                "reductionMetadata": (
                                    {"outputSize": threads}
                                    if initialization
                                    else (
                                        shaped_metadata
                                        if shaped_reduction
                                        else {
                                            name: ctypes.cast(
                                                supplied[name].data,
                                                ctypes.POINTER(ctypes.c_uint64),
                                            )[0]
                                            for name in ("in_size", "row_size")
                                        }
                                    )
                                ),
                            }
                            if reduction
                            else {}
                        ),
                        **(
                            {
                                "copyGuardWords": output["values"][buffer.count :],
                                "copyMetadata": copy_metadata,
                                "copyValues": output["values"][: buffer.count],
                                **(
                                    {
                                        "inputs": inputs,
                                        "packageRoot": str(package_directory),
                                    }
                                    if self.retain_native_modules
                                    else {}
                                ),
                            }
                            if copy
                            else {}
                        ),
                        **(
                            {"binaryGuardValues": output["values"][buffer.count :]}
                            if binary_operation
                            else {}
                        ),
                        **(
                            {
                                "binaryValues": output["values"][: buffer.count],
                                "inputs": inputs,
                                "packageRoot": str(package_directory),
                            }
                            if binary_operation and self.retain_native_modules
                            else {}
                        ),
                        **(
                            {
                                "castGuardValues": output["values"][buffer.count :],
                                "castValues": output["values"][: buffer.count],
                                "inputs": inputs,
                                "packageRoot": str(package_directory),
                            }
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
