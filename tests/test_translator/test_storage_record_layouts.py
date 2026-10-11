"""Mixed scalar field layouts and physical record transport validation."""

import struct
from dataclasses import replace

import pytest

from crosstl.project import NativeLoaderDispatchError, pack_storage_records
from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.project.native_loader_dispatch import _validated_scalar_layout
from crosstl.project.native_runtime_drivers import (
    _prepare_directx_buffers,
    _prepare_opengl_buffers,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    RuntimeAdapterSetupError,
    RuntimeResourceBinding,
    RuntimeValue,
    _runtime_value_physical_byte_length,
)
from crosstl.project.storage_record_layout import (
    storage_record_layout,
    validate_storage_record_layout,
)

STORAGE = {
    "metal": "metal-buffer",
    "directx": "hlsl-structured-buffer",
    "opengl": "std430",
}
FIELDS = [("int", "rank"), ("int64_t", "offset"), ("float", "weight")]


def _layout(target="opengl", fields=FIELDS):
    return storage_record_layout("Metadata", fields, storage_layout=STORAGE[target])


def _reflect(root, target, body):
    declaration = f"struct Metadata {{ {body} }};"
    source = {
        "metal": (
            f"#include <metal_stdlib>\nusing namespace metal;\n{declaration}\nkernel void run(constant Metadata* params [[buffer(0)]]) {{}}"
        ),
        "directx": (
            f"{declaration}\nStructuredBuffer<Metadata> params : register(t0);\n[numthreads(1,1,1)] void CSMain() {{}}"
        ),
        "opengl": (
            f"#version 450 core\n#extension GL_ARB_gpu_shader_int64 : require\n{declaration}\nlayout(std430,binding=0) readonly buffer Params {{ Metadata params[]; }};\nlayout(local_size_x=1) in;\nvoid main() {{}}"
        ),
    }[target]
    path = (
        root
        / {"metal": "record.metal", "directx": "record.hlsl", "opengl": "record.comp"}[
            target
        ]
    )
    path.write_text(source)
    return reflect_target_host_interface(path, target=target, stage="compute")[
        "resources"
    ][0]


@pytest.mark.parametrize("target", STORAGE)
@pytest.mark.parametrize(
    "fields,offsets,stride",
    [
        ([("int", "rank"), ("int64_t", "offset")], [0, 8], 16),
        ([("int64_t", "offset"), ("int", "rank")], [0, 8], 16),
        (FIELDS, [0, 8, 16], 24),
        ([("uint64_t", "count"), ("float", "weight"), ("int", "rank")], [0, 8, 12], 16),
        ([("float", "weight"), ("int", "rank")], [0, 4], 8),
    ],
)
def test_mixed_scalar_reflection(tmp_path, target, fields, offsets, stride):
    result = _reflect(
        tmp_path, target, " ".join(f"{dtype} {name};" for dtype, name in fields)
    )
    layout = result["scalarLayout"]
    assert layout["elementType"] == "record"
    assert layout["payloadEncoding"] == "uint32-le-words"
    assert [field["offsetBytes"] for field in layout["structMembers"]] == offsets
    assert [field["physicalType"] for field in layout["structMembers"]] == [
        dtype for dtype, _ in fields
    ]
    assert validate_storage_record_layout(layout) == stride
    value = RuntimeValue(
        name="params",
        dtype="uint32",
        shape=(2, stride // 4),
        values=[0] * (stride // 2),
    )
    assert (
        _validated_scalar_layout(
            layout,
            runtime_value=value,
            target=target,
            resource_kind="constant-buffer" if target == "metal" else "buffer",
            path="$.layout",
        )
        == layout
    )
    binding = RuntimeResourceBinding(name="params", metadata={"scalarLayout": layout})
    assert _runtime_value_physical_byte_length(value, binding) == stride * 2


@pytest.mark.parametrize("target", STORAGE)
@pytest.mark.parametrize(
    "body",
    [
        "int rank; half weight;",
        "int rank; float2 vector;",
        "int rank; float weights[2];",
        "int rank; Metadata nested;",
        "int rank; int* pointer;",
        "int rank; bool enabled;",
    ],
)
def test_unproven_record_fields_stay_unavailable(tmp_path, target, body):
    assert "scalarLayout" not in _reflect(tmp_path, target, body)


def test_named_record_packing_preserves_types_and_padding():
    words = pack_storage_records(
        _layout(),
        [
            {"rank": -2147483648, "offset": -9223372036854775808, "weight": -0.0},
            {"rank": 2147483647, "offset": 9223372036854775807, "weight": 1.5},
        ],
    )
    assert words == [
        0x80000000,
        0,
        0,
        0x80000000,
        0x80000000,
        0,
        0x7FFFFFFF,
        0,
        0xFFFFFFFF,
        0x7FFFFFFF,
        0x3FC00000,
        0,
    ]
    unsigned = _layout(fields=[("int", "rank"), ("uint64_t", "count")])
    assert pack_storage_records(unsigned, [{"rank": -1, "count": 2**64 - 1}]) == [
        2**32 - 1,
        0,
        2**32 - 1,
        2**32 - 1,
    ]


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"rank": 0, "offset": 0},
        {"rank": 0, "offset": 0, "weight": 0, "extra": 1},
        {"rank": True, "offset": 0, "weight": 0},
        {"rank": 2**31, "offset": 0, "weight": 0},
        {"rank": 0, "offset": 2**63, "weight": 0},
        {"rank": 0, "offset": 0.5, "weight": 0},
        {"rank": 0, "offset": 0, "weight": 1e100},
        {"rank": 0, "offset": 0, "weight": True},
    ],
)
def test_record_packing_rejects_invalid_fields(record):
    with pytest.raises(ValueError):
        pack_storage_records(_layout(), [record])


@pytest.mark.parametrize(
    "mutation",
    [
        {"elementStrideBytes": 20},
        {"elementSizeBytes": 20},
        {"alignmentBytes": 4},
        {"alignmentBytes": True},
        {"memberOffsetBytes": 4},
        {"componentCount": 3},
        {"elementType": "uint32"},
        {"runtimeSized": False},
        {"vectorWidth": 3},
        {"payloadEncoding": "native"},
        {"storageLayout": "std140"},
        {"blockSizeBytes": 24},
        {"minimumBindingSizeBytes": False},
    ],
)
def test_record_dispatch_rejects_forged_metadata(mutation):
    layout = _layout()
    layout.update(mutation)
    code = (
        "resource-minimum-binding-size-invalid"
        if "minimumBindingSizeBytes" in mutation
        else "resource-layout-invalid"
    )
    with pytest.raises(NativeLoaderDispatchError, match=code):
        _validated_scalar_layout(
            layout,
            runtime_value=RuntimeValue(name="p", dtype="uint32", shape=(6,)),
            target="opengl",
            resource_kind="buffer",
            path="$.layout",
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("offsetBytes", 4),
        ("offsetBytes", False),
        ("sizeBytes", 4),
        ("alignmentBytes", 4),
        ("elementType", "uint64"),
        ("name", "rank"),
    ],
)
def test_record_validator_rejects_overlapping_or_retyped_fields(key, value):
    layout = _layout()
    layout["structMembers"][1][key] = value
    with pytest.raises(ValueError):
        validate_storage_record_layout(layout)


@pytest.mark.parametrize(
    "dtype,shape", [("float32", (6,)), ("uint64", (3,)), ("uint32", (5,))]
)
def test_record_dispatch_requires_complete_physical_words(dtype, shape):
    with pytest.raises(NativeLoaderDispatchError, match="resource-layout-mismatch"):
        _validated_scalar_layout(
            _layout(),
            runtime_value=RuntimeValue(name="p", dtype=dtype, shape=shape),
            target="opengl",
            resource_kind="buffer",
            path="$.layout",
        )


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("fault", [None, "stride", "offset", "dtype", "length"])
def test_native_preparation_revalidates_mixed_records(target, fault):
    layout = _layout(target)
    metadata = {"scalarLayout": layout, "byteStride": 24}
    binding = NativeRuntimeBufferBinding(
        name="params",
        binding=RuntimeResourceBinding(
            name="params",
            kind="buffer",
            type_name=(
                "StructuredBuffer<Metadata>" if target == "directx" else "Metadata[]"
            ),
            binding=0,
            set=0,
            access="read",
            metadata=metadata,
        ),
        dtype="uint32",
        shape=(6,),
        value=[0] * 6,
    )
    if fault == "stride":
        layout["elementStrideBytes"] = 12
    elif fault == "offset":
        layout["structMembers"][1]["offsetBytes"] = 4
    elif fault == "dtype":
        binding = replace(binding, dtype="int32")
    elif fault == "length":
        binding = replace(binding, shape=(5,), value=[0] * 5)
    prepare = (
        _prepare_directx_buffers if target == "directx" else _prepare_opengl_buffers
    )
    if fault is not None:
        with pytest.raises(RuntimeAdapterSetupError):
            prepare({"params": binding})
    else:
        prepared = prepare({"params": binding})[0]
        assert prepared.payload == struct.pack("<6I", *([0] * 6))
        assert prepared.byte_length == 24
        if target == "directx":
            assert prepared.stride == 24
