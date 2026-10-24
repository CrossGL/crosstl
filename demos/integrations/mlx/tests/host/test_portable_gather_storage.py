"""Gathers of the MLX storage types carried as exact element-storage words."""

import ctypes
import hashlib
import json
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    gather_axis_layout,
    gather_dispatch,
    gather_evidence,
    gather_layout,
    gather_packages,
    gather_storage,
    runtime,
)
from demos.integrations.mlx.portable_host.element_storage import (
    COMPLEX_LAYOUT,
    GUARD_COUNT,
    PHYSICAL_SIZES,
    guard_words,
)

TARGETS = ("metal", "directx", "opengl")
# Each dtype's source words: signed zeros, infinities, NaN payloads and extremes.
WORDS = {
    "float16": [0x0000, 0x8000, 0x3C00, 0x7C00, 0xFC00, 0x7E00, 0x0001, 0xBC00],
    "bfloat16": [0x0000, 0x8000, 0x3F80, 0x7F80, 0xFF80, 0x7FC0, 0x4049, 0xBF80],
    "complex64": [
        0x3F800000_00000000,
        0x7FC00000_3F800000,
        0x80000000_7F800000,
        0x40000000_BF800000,
        0x00000000_7FC00000,
        0xFF800000_00000001,
        0x3F000000_C0000000,
        0x41200000_41200000,
    ],
    "int8": [-128, -1, 0, 1, 127, 64, -64, 5],
    "uint8": [0, 1, 127, 128, 255, 64, 200, 5],
    "int16": [-32768, -1, 0, 1, 32767, 1024, -1024, 7],
    "uint16": [0, 1, 32767, 32768, 65535, 1024, 50000, 7],
}
EXTENDED = tuple(WORDS)
# Source shape (4, 2), gather rows [3, 0, 2] of slice (1, 2).
INDICES = [3, 0, -2]
ROWS = [3, 0, 2]


def expected_words(dtype):
    return [WORDS[dtype][2 * row + column] for row in ROWS for column in range(2)]


def allocate(name, kind, values):
    size = values if isinstance(values, int) else len(values)
    memory = (gather_layout.TYPES[kind] * size)()
    if not isinstance(values, int):
        memory[:] = values
    buffer = runtime.Buffer(
        name.encode(), kind.encode(), ctypes.addressof(memory), size, int(name == "out")
    )
    return buffer, memory


def gather_buffers(dtype):
    data = {
        "src": (dtype, WORDS[dtype]),
        "out": (dtype, 6),
        "src_shape": ("int32", [4, 2]),
        "src_strides": ("int64", [2, 1]),
        "src_ndim": ("uint64", [2]),
        "slice_sizes": ("int32", [1, 2]),
        "axes": ("int32", [0]),
        "idx_shapes": ("int32", [3]),
        "idx_strides": ("int64", [1]),
        "idx_contigs": ("bool_", [1]),
        "idx_ndim": ("int32", [1]),
        "idx0": ("int32", INDICES),
    }
    supplied, memory = {}, {}
    for name, (kind, values) in data.items():
        supplied[name], memory[name] = allocate(name, kind, values)
    return supplied, memory


def reflection(entry, name, kind, target):
    member = (entry + "_" if target == "directx" else "") + (
        "out_" if name == "out" else name
    )
    if name in gather_storage.VALUES and kind == "complex64":
        return {"memberName": member, **COMPLEX_LAYOUT}
    if name in gather_storage.VALUES:
        storage = gather_storage.packet(kind, target, [])["dtype"]
        size = PHYSICAL_SIZES[storage]
    else:
        storage = runtime.physical_dtype(kind, target)
        size = 1 if storage == "bool" else ctypes.sizeof(gather_layout.TYPES[storage])
    return {"memberName": member, "elementType": storage, "elementStrideBytes": size}


class Kernel:
    """A correct native gather over carrier values, with optional corruption."""

    def __init__(self, tmp_path, target, fault=None):
        self.tmp_path, self.target, self.fault = tmp_path, target, fault

    def request(self, descriptor, directory, inputs, outputs, execution, **kwargs):
        assert kwargs == {"expected_target": self.target}
        return SimpleNamespace(
            inputs=inputs, outputs=outputs, execution=execution, descriptor=descriptor
        )

    def __call__(self, _host, request):
        names = {
            binding["scalarLayout"]["memberName"]
            .removeprefix(ENTRY_PREFIX[0])
            .removeprefix("_"): binding["name"]
            for binding in request.descriptor["bindings"]
        }
        source = request.inputs[names["src"]]["values"]
        initial = request.inputs[names["out_"]]
        width = len(source) // 8
        values = [
            source[(2 * row + column) * width + part]
            for row in ROWS
            for column in range(2)
            for part in range(width)
        ] + initial["values"][6 * width :]
        if self.fault == "guard":
            values[-1] ^= 1
        readback = dict(request.outputs[names["out_"]], values=values)
        if self.fault == "dtype":
            readback["dtype"] = "int64"
        return SimpleNamespace(
            status="ok", outputs={names["out_"]: readback}, details=self.details(request)
        )

    def details(self, request):
        suffix = {"metal": ".air", "directx": ".dxil", "opengl": ".glsl"}[self.target]
        modules = []
        for name in ("module" + suffix, "validation" + suffix, "module.metallib"):
            path = self.tmp_path / name
            path.write_bytes(b"retained-" + name.encode())
            modules.append(
                {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
        buffers = {
            binding["name"]: {
                **{
                    key: item
                    for key, item in request.inputs[binding["name"]].items()
                    if key != "values"
                },
                "binding": {"metadata": {"scalarLayout": binding["scalarLayout"]}},
            }
            for binding in request.descriptor["bindings"]
        }
        dispatch = {
            "entryPoint": {"opengl": "main", "directx": "CSMain"}.get(
                self.target, ENTRY_PREFIX[1]
            ),
            "target": self.target,
            "buffers": buffers,
            "dispatch": request.execution,
        }
        artifact = request.descriptor["artifact"]
        identity = {key: artifact[key] for key in ("hash", "sizeBytes")}
        action = {
            "metal": ("compile-metal-for-native-runtime", "link-metal-for-native-runtime"),
            "directx": ("compile-hlsl-for-directx-runtime",),
            "opengl": ("validate-glsl-for-opengl-runtime",),
        }[self.target]
        return {
            "request": dispatch,
            "module": modules[0],
            "validationModules": modules[1:],
            "artifactIdentityVerification": {
                "verificationStatus": "verified",
                "target": self.target,
                "expectedIdentity": identity,
                "observedIdentity": identity,
            },
            "nativeRuntimeDispatch": dispatch,
            "adapterSteps": [{"action": step, "status": "passed"} for step in action],
        }


ENTRY_PREFIX = ["", ""]


def run(tmp_path, dtype, target, fault=None):
    entry = f"gather{dtype}int32_1_1_int"
    ENTRY_PREFIX[:] = [entry, entry]
    supplied, memory = gather_buffers(dtype)
    artifact = tmp_path / "source.txt"
    artifact.write_bytes(b"retained-source")
    descriptor = {
        "artifact": {
            "packagePath": artifact.name,
            "sizeBytes": artifact.stat().st_size,
            "hash": {"value": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        },
        "bindings": [
            {
                "name": f"binding{i}",
                "scalarLayout": reflection(
                    entry, name, buffer.dtype.decode(), target
                ),
            }
            for i, (name, buffer) in enumerate(supplied.items())
        ],
    }
    kernel = Kernel(tmp_path, target, fault)
    host = SimpleNamespace(
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        dispatch_count=0,
        gathers=SimpleNamespace(get=lambda entry, maximum: (descriptor, tmp_path)),
    )
    table = (runtime.Buffer * len(supplied))(*supplied.values())
    patch = pytest.MonkeyPatch()
    patch.setattr(gather_dispatch, "build_native_loader_dispatch_request", kernel.request)
    patch.setattr(gather_dispatch, "execute", kernel)
    try:
        gather_dispatch.dispatch(
            host, entry, table, len(table), 6, runtime.Launch((3, 1, 2), (1, 1, 1))
        )
    finally:
        patch.undo()
    return host, memory


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", EXTENDED)
def test_extended_gather_round_trips_exact_storage_and_audits(tmp_path, dtype, target):
    import numpy as np

    host, memory = run(tmp_path, dtype, target)
    assert list(memory["out"]) == expected_words(dtype) and host.dispatch_count == 1
    event = json.loads(host.trace.read_text())
    assert event["gatherValues"] == expected_words(dtype)
    assert event["gatherGuardValues"] == guard_words(dtype)
    assert event["gatherStorageType"] == dtype
    gather_evidence.audit_event(np, event)
    corrupted = json.loads(host.trace.read_text())
    corrupted["gatherValues"][0] = corrupted["gatherValues"][2]
    corrupted["outputHash"] = gather_storage.output_hash(
        dtype, corrupted["gatherValues"]
    )
    with pytest.raises(ValueError, match="disagrees"):
        gather_evidence.audit_event(np, corrupted)
    guarded = json.loads(host.trace.read_text())
    guarded["gatherGuardValues"] = guard_words(dtype, GUARD_COUNT - 1)
    with pytest.raises(ValueError, match="guard"):
        gather_evidence.audit_event(np, guarded)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("fault", ("guard", "dtype"))
@pytest.mark.parametrize("dtype", ("float16", "complex64", "int8"))
def test_extended_gather_refuses_invalid_readback(tmp_path, dtype, target, fault):
    with pytest.raises(RuntimeError):
        run(tmp_path, dtype, target, fault)


@pytest.mark.parametrize("dtype", EXTENDED)
def test_extended_entries_keep_upstream_type_spellings(tmp_path, dtype):
    root = tmp_path
    header = root / gather_packages.JIT_HEADER
    header.parent.mkdir(parents=True)
    header.write_text('gather_kernels = R"({0}|{1}|{2})";\n')
    source = gather_packages.source(root, f"gather{dtype}uint32_1_1_int")
    assert f"{dtype}uint32|{gather_packages.SOURCE_TYPES[dtype]}|uint" in source
    axis = gather_packages.source(root, f"gather_axis{dtype}int64_intcnc")
    assert f"gather_axis<{gather_packages.SOURCE_TYPES[dtype]}, int64_t, int" in axis


def test_axis_gather_accepts_wide_storage_within_per_axis_limits():
    entry = "gather_axisfloat32uint32_intcc"
    size = 3 * 32769
    data = {
        "src": ("float32", size),
        "indices": ("uint32", size),
        "out": ("float32", size),
        "shape": ("int32", [3]),
        "src_strides": ("int64", [32769]),
        "idx_strides": ("int64", [32769]),
        "ndim": ("uint64", [1]),
        "axis": ("int32", [1]),
        "axis_size": ("int32", [32769]),
        "src_ax_stride": ("uint64", [1]),
        "idx_ax_stride": ("uint64", [1]),
    }
    supplied, memory = {}, {}
    for name, (kind, values) in data.items():
        supplied[name], memory[name] = allocate(name, kind, values)
    execution = runtime.Launch((1, 32769, 3), (1, 1, 1)).execution()
    metadata = gather_axis_layout.validate(entry, supplied, size, execution)
    assert metadata["outputCount"] == size and metadata["maximumIndex"] == size - 1
    memory["indices"][size - 1] = 32769
    with pytest.raises(ValueError, match="index"):
        gather_axis_layout.validate(entry, supplied, size, execution)
    memory["indices"][size - 1] = 0
    # The same storage launched along one axis exceeds the portable limit.
    for name, values in (
        ("shape", [1]),
        ("src_strides", [size]),
        ("idx_strides", [size]),
        ("axis_size", [size]),
    ):
        memory[name][:] = values
    with pytest.raises(ValueError, match="per-axis"):
        gather_axis_layout.validate(
            entry,
            supplied,
            size,
            {"workgroupCount": [1, size, 1], "workgroupSize": [1, 1, 1]},
        )


@pytest.mark.parametrize("dtype", ("float64", "complex128", "float8"))
def test_types_without_metal_storage_stay_unsupported(dtype):
    with pytest.raises(ValueError):
        gather_layout.signature(f"gather{dtype}int32_1_1_int")
    with pytest.raises(ValueError):
        gather_axis_layout.signature(f"gather_axis{dtype}int32_intcc")
