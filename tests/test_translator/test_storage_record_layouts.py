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
        "int rank; float weights[];",
        "int rank; float weights[0];",
        "int rank; float weights[COUNT];",
        "int rank; float weights[-2];",
        "int rank; float weights[2 + ];",
        "int rank; float weights[4097];",
        "int rank; float weights[2][2];",
        "int rank; half weights[2];",
        "int rank; bool flags[2];",
        "int rank; Metadata nested;",
        "int rank; int* pointer;",
    ],
)
def test_unproven_record_fields_stay_unavailable(tmp_path, target, body):
    assert "scalarLayout" not in _reflect(tmp_path, target, body)


# MLXConvParams<2> from MLX's steel/conv/params.h, as CrossTL emits it. The
# offsets match DXC's structured-buffer layout report for the same struct.
CONV_PARAMS = (
    "int N; int C; int O; int iS[2]; int wS[2]; int oS[2]; int str[2]; "
    "int pad[2]; int kdil[2]; int idil[2]; int64_t in_strides[2 + 2]; "
    "int64_t wt_strides[(2 + 2)]; int64_t out_strides[(2 + 2)]; int groups; "
    "bool flip;"
)


@pytest.mark.parametrize("target", STORAGE)
def test_array_and_boolean_record_reflection(tmp_path, target):
    layout = _reflect(tmp_path, target, CONV_PARAMS)["scalarLayout"]
    assert validate_storage_record_layout(layout) == 176
    members = {member["name"]: member for member in layout["structMembers"]}
    assert [member["offsetBytes"] for member in layout["structMembers"]] == [
        0,
        4,
        8,
        12,
        20,
        28,
        36,
        44,
        52,
        60,
        72,
        104,
        136,
        168,
        172,
    ]
    assert members["iS"] == {
        "name": "iS",
        "physicalType": "int",
        "elementType": "int32",
        "offsetBytes": 12,
        "sizeBytes": 8,
        "alignmentBytes": 4,
        "arrayLength": 2,
        "arrayStrideBytes": 4,
    }
    assert members["out_strides"]["arrayLength"] == 4
    assert members["out_strides"]["sizeBytes"] == 32
    boolean = 1 if target == "metal" else 4
    assert members["flip"] == {
        "name": "flip",
        "physicalType": "bool",
        "elementType": "bool",
        "offsetBytes": 172,
        "sizeBytes": boolean,
        "alignmentBytes": boolean,
    }
    assert layout["alignmentBytes"] == 8
    value = RuntimeValue(name="params", dtype="uint32", shape=(44,), values=[0] * 44)
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


@pytest.mark.parametrize("target", STORAGE)
def test_boolean_width_changes_record_stride(tmp_path, target):
    body = "int rank; bool a; bool b; bool c; bool d; bool e;"
    layout = _reflect(tmp_path, target, body)["scalarLayout"]
    offsets = [member["offsetBytes"] for member in layout["structMembers"]]
    if target == "metal":
        assert offsets == [0, 4, 5, 6, 7, 8]
        assert validate_storage_record_layout(layout) == 12
    else:
        assert offsets == [0, 4, 8, 12, 16, 20]
        assert validate_storage_record_layout(layout) == 24


@pytest.mark.parametrize("target", STORAGE)
@pytest.mark.parametrize(
    "body,stride",
    [("float weights[3];", 12), ("bool first; bool second;", None)],
)
def test_homogeneous_array_and_boolean_structs_are_records(
    tmp_path, target, body, stride
):
    layout = _reflect(tmp_path, target, body)["scalarLayout"]
    assert layout["elementType"] == "record" and "componentCount" not in layout
    if stride is None:
        stride = 2 if target == "metal" else 8
    assert validate_storage_record_layout(layout) == stride


def _metal_reference(root, body, declaration):
    path = root / "reference.metal"
    path.write_text(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        f"struct Metadata {{ {body} }};\n"
        f"kernel void run({declaration} params [[buffer(0)]]) {{}}\n"
    )
    return reflect_target_host_interface(path, target="metal", stage="compute")[
        "resources"
    ][0]


def test_metal_constant_record_reference_requires_one_record(tmp_path):
    resource = _metal_reference(tmp_path, CONV_PARAMS, "constant Metadata&")
    layout = resource["scalarLayout"]
    assert resource["kind"] == "constant-buffer"
    assert layout["runtimeSized"] is True
    assert layout["minimumBindingSizeBytes"] == 176
    assert validate_storage_record_layout(layout) == 176
    value = RuntimeValue(name="params", dtype="uint32", shape=(44,), values=[0] * 44)
    assert (
        _validated_scalar_layout(
            layout,
            runtime_value=value,
            target="metal",
            resource_kind="constant-buffer",
            path="$.layout",
        )
        == layout
    )
    binding = RuntimeResourceBinding(name="params", metadata={"scalarLayout": layout})
    assert _runtime_value_physical_byte_length(value, binding) == 176


@pytest.mark.parametrize(
    "body,declaration",
    [
        # Homogeneous structs keep their component layout only through pointers.
        ("float real; float imag;", "constant Metadata&"),
        (CONV_PARAMS, "device Metadata&"),
        (CONV_PARAMS, "thread Metadata&"),
        ("int rank; half weight;", "constant Metadata&"),
    ],
)
def test_unproven_metal_record_references_stay_unavailable(tmp_path, body, declaration):
    assert "scalarLayout" not in _metal_reference(tmp_path, body, declaration)


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


ARRAY_FIELDS = [("int", "rank"), ("int64_t", "strides", 3), ("bool", "flip")]


@pytest.mark.parametrize("target", STORAGE)
def test_array_and_boolean_record_packing(target):
    layout = _layout(target, ARRAY_FIELDS)
    assert validate_storage_record_layout(layout) == 40
    words = pack_storage_records(
        layout,
        [
            {"rank": -1, "strides": [1, -2, 2**40], "flip": True},
            {"rank": 7, "strides": (0, 0, 0), "flip": False},
        ],
    )
    # rank, padding, three int64 strides, then the boolean and its padding.
    assert words == [
        0xFFFFFFFF,
        0,
        1,
        0,
        0xFFFFFFFE,
        0xFFFFFFFF,
        0,
        0x100,
        1,
        0,
        7,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    flags = _layout(target, [("bool", "a"), ("bool", "b"), ("int", "c")])
    packed = pack_storage_records(flags, [{"a": True, "b": True, "c": -3}])
    # Metal packs adjacent booleans into bytes; other targets use whole words.
    assert packed == ([0x0101, 0xFFFFFFFD] if target == "metal" else [1, 1, 0xFFFFFFFD])


@pytest.mark.parametrize(
    "record",
    [
        {"rank": 0, "strides": [0, 0], "flip": False},
        {"rank": 0, "strides": [0, 0, 0, 0], "flip": False},
        {"rank": 0, "strides": 0, "flip": False},
        {"rank": 0, "strides": [0, 0, 0.5], "flip": False},
        {"rank": 0, "strides": [0, 0, True], "flip": False},
        {"rank": 0, "strides": [0, 0, 2**63], "flip": False},
        {"rank": 0, "strides": [0, [0], 0], "flip": False},
        {"rank": 0, "strides": [0, 0, 0], "flip": 1},
        {"rank": 0, "strides": [0, 0, 0], "flip": [True]},
    ],
)
def test_array_and_boolean_packing_rejects_invalid_values(record):
    with pytest.raises(ValueError):
        pack_storage_records(_layout("metal", ARRAY_FIELDS), [record])


@pytest.mark.parametrize(
    "fields",
    [
        [("int", "rank", 0)],
        [("int", "rank", 4097)],
        [("int", "rank", True)],
        [("int", "rank", 2.0)],
        [("bool", "flags", 2)],
        [("half", "weights", 2)],
        [("int", "rank", 2, 3)],
        [("int",)],
        ["int rank"],
    ],
)
def test_record_layout_rejects_unproven_array_fields(fields):
    with pytest.raises(ValueError):
        storage_record_layout("Metadata", fields, storage_layout="std430")


@pytest.mark.parametrize(
    "index,key,value",
    [
        (1, "arrayLength", 2),
        (1, "arrayLength", None),
        (1, "arrayStrideBytes", 16),
        (1, "sizeBytes", 16),
        (2, "sizeBytes", 1),
        (2, "alignmentBytes", 1),
        (2, "elementType", "uint32"),
        (2, "arrayLength", 1),
        (0, "arrayLength", 1),
    ],
)
def test_record_validator_rejects_forged_array_and_boolean_fields(index, key, value):
    layout = _layout("opengl", ARRAY_FIELDS)
    layout["structMembers"][index][key] = value
    with pytest.raises(ValueError):
        validate_storage_record_layout(layout)


def test_record_validator_accepts_only_target_boolean_width():
    metal = _layout("metal", ARRAY_FIELDS)
    metal["storageLayout"] = "std430"
    with pytest.raises(ValueError):
        validate_storage_record_layout(metal)


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
