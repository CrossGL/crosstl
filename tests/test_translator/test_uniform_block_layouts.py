"""Mixed parameter blocks through target reflection, packaging and native dispatch."""

import json
import os
import struct
from dataclasses import replace

import pytest

from crosstl.project import (
    NativeLoaderDispatchError,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
)
from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.project.native_loader_dispatch import _validated_scalar_layout
from crosstl.project.native_runtime_drivers import (
    _prepare_directx_buffers,
    _prepare_opengl_buffers,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    RuntimeAdapterSetupError,
    RuntimeAllocationView,
    RuntimeResourceBinding,
    RuntimeValue,
    _runtime_scalar_layout_signature,
)
from crosstl.project.uniform_layout import (
    pack_uniform_block,
    std140_block_layout,
    validate_std140_block_layout,
    validate_uniform_block_layout,
)
from tests.test_translator.test_native_loader_dispatch_integration import _executor


def _reflect(tmp_path, body, *, qualifiers="std140, binding=0", instance="p"):
    artifact = tmp_path / "params.comp"
    artifact.write_text(
        f"""#version 450 core
        layout({qualifiers}) uniform Parameters {{ {body} }} {instance};
        layout(local_size_x=1) in;
        void main() {{}}
        """,
        encoding="utf-8",
    )
    return reflect_target_host_interface(artifact, target="opengl", stage="compute")


@pytest.mark.parametrize(
    "body,offsets,sizes,alignment,total",
    [
        (
            "int a; int b; int c; int d; int e; float f; int g; int h;",
            list(range(0, 32, 4)),
            [4] * 8,
            16,
            32,
        ),
        (
            "int a; vec2 b; vec3 c; float d; uint e;",
            [0, 8, 16, 28, 32],
            [4, 8, 12, 4, 4],
            16,
            48,
        ),
        (
            "float a; i64vec3 b; int64_t c; uint d;",
            [0, 32, 56, 64],
            [4, 24, 8, 4],
            32,
            96,
        ),
        (
            "bool a; bvec2 b; bvec3 c; uint64_t d;",
            [0, 8, 16, 32],
            [4, 8, 12, 8],
            16,
            48,
        ),
    ],
)
def test_uniform_reflection_retains_physical_layout(
    tmp_path, body, offsets, sizes, alignment, total
):
    result = _reflect(tmp_path, body)
    assert result["status"] == "ready"
    layout = result["resources"][0]["scalarLayout"]
    assert layout["physicalType"] == "Parameters"
    assert layout["payloadEncoding"] == "uint32-le-words"
    assert layout["blockSizeBytes"] == total
    assert layout["alignmentBytes"] == alignment
    assert [member["offsetBytes"] for member in layout["blockMembers"]] == offsets
    assert [member["sizeBytes"] for member in layout["blockMembers"]] == sizes
    signature = _runtime_scalar_layout_signature({"scalarLayout": layout})
    assert signature["blockMembers"] == layout["blockMembers"]
    assert signature["payloadEncoding"] == "uint32-le-words"
    value = RuntimeValue(
        name="p", dtype="uint32", shape=(total // 4,), values=[0] * (total // 4)
    )
    assert (
        _validated_scalar_layout(
            layout,
            runtime_value=value,
            target="opengl",
            resource_kind="constant-buffer",
            path="$.layout",
        )
        == layout
    )


@pytest.mark.parametrize(
    "body",
    [
        "float a; mat2 b;",
        "int a; float b[COUNT];",
        "int a; float b[0];",
        "int a; float b[2][2];",
        "int a; mat2 b[2];",
        "float a; Sample b;",
        "float a; float a;",
        "float a; float b",
        "float a, b;",
        "float a; layout(offset=16) float b;",
        "float a; double b;",
        "float a; uint b[];",
        "float a; highp float b;",
        "float a;; int b;",
    ],
)
def test_uniform_reflection_diagnoses_unsupported_members(tmp_path, body):
    result = _reflect(tmp_path, body)
    assert result["status"] == "incomplete"
    assert "scalarLayout" not in result["resources"][0]
    assert (
        result["diagnosticRecords"][0]["details"]["reasonKind"]
        == "uniform-block-layout-unsupported"
    )


@pytest.mark.parametrize(
    "qualifiers",
    [
        "binding=0",
        "shared, binding=0",
        "std140, packed, binding=0",
        "std140, offset=16, binding=0",
        "std140, row_major, binding=0",
        "align=32) layout(std140, binding=0",
    ],
)
def test_uniform_reflection_rejects_unproven_qualifiers(tmp_path, qualifiers):
    result = _reflect(tmp_path, "float a; int b;", qualifiers=qualifiers)
    assert result["status"] == "incomplete"
    assert "scalarLayout" not in result["resources"][0]


def test_uniform_reflection_diagnoses_block_arrays(tmp_path):
    result = _reflect(tmp_path, "float a; int b;", instance="p[2]")
    assert result["status"] == "incomplete"
    assert result["diagnosticRecords"][0]["details"]["resource"] == "Parameters"


@pytest.mark.parametrize(
    "target,kind", [("directx", "constant-buffer"), ("opengl", "buffer")]
)
def test_uniform_block_rejects_wrong_storage_class(tmp_path, target, kind):
    layout = _reflect(tmp_path, "int a; vec3 b; uint c; float d;")["resources"][0][
        "scalarLayout"
    ]
    value = RuntimeValue(name="p", dtype="uint32", shape=(12,), values=[0] * 12)
    with pytest.raises(NativeLoaderDispatchError, match="resource-layout-mismatch"):
        _validated_scalar_layout(
            layout,
            runtime_value=value,
            target=target,
            resource_kind=kind,
            path="$.layout",
        )


def _binding(layout, *, dtype="uint32", count=12):
    return NativeRuntimeBufferBinding(
        name="p",
        binding=RuntimeResourceBinding(
            name="p",
            kind="constant-buffer",
            binding=0,
            set=0,
            access="read",
            metadata={"scalarLayout": layout},
        ),
        dtype=dtype,
        shape=(count,),
        value=[0] * count,
        source="input",
    )


@pytest.mark.parametrize("view_size,allocation_size", [(44, 304), (48, 300)])
def test_uniform_block_rejects_incomplete_allocation_view(
    tmp_path, view_size, allocation_size
):
    layout = _reflect(tmp_path, "int a; vec3 b; uint c; float d;")["resources"][0][
        "scalarLayout"
    ]
    binding = replace(
        _binding(layout),
        allocation=RuntimeAllocationView(
            allocation_id="parameters",
            byte_offset=256,
            byte_length=view_size,
            allocation_byte_length=allocation_size,
        ),
    )
    with pytest.raises(RuntimeAdapterSetupError):
        _prepare_opengl_buffers({"p": binding})


def test_uniform_block_preserves_bounded_allocation_view(tmp_path):
    layout = _reflect(tmp_path, "int a; vec3 b; uint c; float d;")["resources"][0][
        "scalarLayout"
    ]
    binding = replace(
        _binding(layout),
        allocation=RuntimeAllocationView(
            allocation_id="parameters",
            byte_offset=256,
            byte_length=48,
            allocation_byte_length=512,
        ),
    )
    (prepared,) = _prepare_opengl_buffers({"p": binding})
    assert prepared.byte_offset == 256 and prepared.byte_length == 48
    assert prepared.allocation_size == 512
    assert prepared.payload == b"\x00" * 48


@pytest.mark.parametrize(
    "field,value",
    [
        ("blockSizeBytes", 32),
        ("alignmentBytes", 4),
        ("memberOffsetBytes", 4),
        ("elementSizeBytes", 48),
        ("elementStrideBytes", 48),
        ("elementType", "float32"),
        ("runtimeSized", True),
        ("storageLayout", "std430"),
        ("blockMembers", None),
        ("payloadEncoding", "float32"),
        ("vectorWidth", 4),
        ("physicalType", "P[]"),
        ("blockSizeBytes", 48.0),
        ("memberOffsetBytes", False),
        ("extra", 1),
    ],
)
def test_uniform_dispatch_rejects_malformed_block(tmp_path, field, value):
    layout = _reflect(tmp_path, "int a; vec3 b; uint c; float d;")["resources"][0][
        "scalarLayout"
    ]
    layout[field] = value
    _assert_rejected(layout)


@pytest.mark.parametrize(
    "field,value",
    [
        ("offsetBytes", 4),
        ("sizeBytes", 16),
        ("alignmentBytes", 4),
        ("vectorWidth", 4),
        ("physicalType", "mat3"),
        ("elementType", "int32"),
        ("offsetBytes", True),
        ("name", "a"),
        ("extra", 1),
    ],
)
def test_uniform_dispatch_rejects_malformed_member(tmp_path, field, value):
    layout = _reflect(tmp_path, "int a; vec3 b; uint c; float d;")["resources"][0][
        "scalarLayout"
    ]
    layout["blockMembers"][1][field] = value
    _assert_rejected(layout)


def _assert_rejected(layout):
    value = RuntimeValue(name="p", dtype="uint32", shape=(12,), values=[0] * 12)
    with pytest.raises(NativeLoaderDispatchError, match="resource-layout-invalid"):
        _validated_scalar_layout(
            layout,
            runtime_value=value,
            target="opengl",
            resource_kind="constant-buffer",
            path="$.layout",
        )
    with pytest.raises(RuntimeAdapterSetupError) as exc:
        _prepare_opengl_buffers({"p": _binding(layout)})
    assert exc.value.details["reasonKind"] == "uniform-block-layout-invalid"


@pytest.mark.parametrize(
    "dtype,count", [("float32", 12), ("uint32", 11), ("uint32", 13)]
)
def test_uniform_dispatch_requires_complete_word_payload(tmp_path, dtype, count):
    layout = _reflect(tmp_path, "int a; vec3 b; uint c; float d;")["resources"][0][
        "scalarLayout"
    ]
    value = RuntimeValue(name="p", dtype=dtype, shape=(count,), values=[0] * count)
    with pytest.raises(NativeLoaderDispatchError, match="resource-layout-mismatch"):
        _validated_scalar_layout(
            layout,
            runtime_value=value,
            target="opengl",
            resource_kind="constant-buffer",
            path="$.layout",
        )
    with pytest.raises(RuntimeAdapterSetupError) as exc:
        _prepare_opengl_buffers({"p": _binding(layout, dtype=dtype, count=count)})
    assert exc.value.details["reasonKind"] == "uniform-block-payload-mismatch"


# Fixed arrays start every element on a 16-byte boundary in both std140 and
# HLSL constant buffers; std140 also rounds the array's end up to 16 bytes,
# while HLSL lets the next member share the last element's register.
ARRAY_RECORD_OFFSETS = {
    "opengl": ([0, 16, 48, 96, 100], 112),
    "directx": ([0, 16, 48, 88, 92], 96),
}


def _package(tmp_path, vectors, target="opengl"):
    if vectors == "arrays":
        body = "int count; int scale[2]; long shift[3]; bool flip; float bias;"
        expression = (
            "float(p.count + 2*p.scale[0] + 3*p.scale[1]) "
            "+ float(p.shift[0] + 5*p.shift[2] + (p.shift[1] >> 36)) "
            "+ (p.flip ? 7.0f : 0.0f) + p.bias"
        )
        offsets, size = ARRAY_RECORD_OFFSETS[target]
        payload = bytearray(size)
        count_at, scale_at, shift_at, flip_at, bias_at = offsets
        for offset, fmt, values in [
            (count_at, "i", (2,)),
            (scale_at, "i", (3,)),
            (scale_at + 16, "i", (5,)),
            (shift_at, "q", (7,)),
            (shift_at + 16, "q", (2**40,)),
            (shift_at + 32, "q", (13,)),
            (flip_at, "I", (1,)),
            (bias_at, "f", (0.25,)),
        ]:
            struct.pack_into("<" + fmt, payload, offset, *values)
        base = 118.25
    elif vectors:
        body = "int first; float2 second; float3 third; float last; uint tail;"
        expression = "float(p.first) + 2*p.second.x + 3*p.second.y + 5*p.third.x + 7*p.third.y + 11*p.third.z + 13*p.last + 17*float(p.tail)"
        payload = bytearray(b"\xa5" * 48)
        for offset, fmt, values in [
            (0, "i", (1,)),
            (8, "2f", (2, 3)),
            (16, "3f", (4, 5, 6)),
            (28, "f", (7,)),
            (32, "I", (8,)),
        ]:
            struct.pack_into("<" + fmt, payload, offset, *values)
        base, offsets = 362.0, [0, 8, 16, 28, 32]
    else:
        body = "int width; int height; int count; int row; int column; float scale; int bias; int enabled;"
        expression = "p.scale * float(p.width + 2*p.height + 3*p.count + 5*p.row + 7*p.column + 11*p.bias + 13*p.enabled)"
        payload = struct.pack("<5if2i", 2, 3, 5, 7, 11, 0.25, -2, 1)
        base, offsets = 31.5, list(range(0, 32, 4))
    (tmp_path / "parameters.metal").write_text(
        f"""#include <metal_stdlib>
using namespace metal;
struct Parameters {{ {body} }};
kernel void apply_parameters(constant Parameters& p [[buffer(0)]],
    device float* output [[buffer(1)]], uint index [[thread_position_in_grid]]) {{
    if (index < 4) output[index] = {expression} + float(index);
}}
""",
        encoding="utf-8",
    )
    config = tmp_path / "crosstl.toml"
    config.write_text(
        f"""[project]
source_roots = ["."]
include = ["parameters.metal"]
targets = ["{target}"]
output_dir = "out"
workgroup_size = [1, 1, 1]
[project.entry_points]
"parameters.metal" = "apply_parameters"
""",
        encoding="utf-8",
    )
    report = translate_project(
        load_project_config(tmp_path, config), format_output=False, validate=True
    )
    report.write_json(tmp_path / "report.json")
    assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()[
        "diagnostics"
    ]
    manifest = build_runtime_artifact_manifest(tmp_path / "report.json")
    assert manifest["success"], manifest
    (tmp_path / "artifacts.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    package = tmp_path / "package"
    assert build_runtime_package(tmp_path / "artifacts.json", package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"], loader
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    (tmp_path / "descriptor.json").write_text(
        json.dumps(descriptor, indent=2), encoding="utf-8"
    )
    resources = sorted(
        descriptor["bindings"], key=lambda binding: binding["coordinates"]["binding"]
    )
    layout = resources[0]["scalarLayout"]
    assert [member["offsetBytes"] for member in layout["blockMembers"]] == offsets
    assert layout["blockSizeBytes"] == len(payload)
    (tmp_path / "parameters.bin").write_bytes(payload)
    words = list(struct.unpack(f"<{len(payload)//4}I", payload))
    if vectors == "arrays":
        assert words == pack_uniform_block(
            layout,
            {
                "count": 2,
                "scale": [3, 5],
                "shift": [7, 2**40, 13],
                "flip": True,
                "bias": 0.25,
            },
        )
    inputs = {
        resources[0]["name"]: {
            "dtype": "uint32",
            "shape": [len(words)],
            "values": words,
        }
    }
    outputs = {
        resources[1]["name"]: {
            "dtype": "float32",
            "shape": [8],
            "values": [base + i for i in range(4)] + [0] * 4,
        }
    }
    request = build_native_loader_dispatch_request(
        descriptor, package, inputs, outputs, [8, 1, 1], expected_target=target
    )
    assert request.execution_plan.diagnostics == ()
    binding = request.adapter_contract.resource_bindings[0]
    assert binding.metadata["scalarLayout"] == layout
    return request, outputs


@pytest.mark.parametrize("vectors", [False, True, "arrays"])
def test_uniform_layout_survives_public_package_dispatch(tmp_path, vectors):
    _package(tmp_path, vectors)


def _readback(tmp_path, request, expected, target):
    executor = _executor(target)
    availability = executor.is_available(request)
    assert availability.available, availability.reason
    try:
        result = executor.run(request)
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if callable(close):
            close()
    (tmp_path / "outputs.json").write_text(
        json.dumps(result.outputs, indent=2), encoding="utf-8"
    )
    assert result.status == "ok"
    for name, value in expected.items():
        assert result.outputs[name]["values"] == value["values"]


@pytest.mark.parametrize("vectors", [False, True, "arrays"])
def test_uniform_block_native_readback(tmp_path, vectors):
    if os.environ.get("CROSTL_REQUIRE_UNIFORM_BLOCK_RUNTIME") != "1":
        pytest.skip("requires the native OpenGL uniform-block gate")
    request, expected = _package(tmp_path, vectors)
    _readback(tmp_path, request, expected, "opengl")


def test_directx_constant_buffer_record_survives_public_package_dispatch(tmp_path):
    request, _ = _package(tmp_path, "arrays", target="directx")
    layout = request.adapter_contract.resource_bindings[0].metadata["scalarLayout"]
    assert layout["storageLayout"] == "hlsl-constant-buffer"
    assert layout["physicalType"] == "Parameters"
    generated = next((tmp_path / "out").rglob("*.hlsl")).read_text(encoding="utf-8")
    assert "ConstantBuffer<Parameters> p : register(b0);" in generated


def test_directx_constant_buffer_record_native_readback(tmp_path):
    if os.environ.get("CROSTL_REQUIRE_DIRECTX_CONSTANT_BUFFER_RECORDS") != "1":
        pytest.skip("requires the native DirectX constant-buffer record gate")
    request, expected = _package(tmp_path, "arrays", target="directx")
    _readback(tmp_path, request, expected, "directx")


# Offsets glslangValidator -q reports for this block (arrayStride 16, 16, 16, 32).
STD140_ARRAY_BODY = (
    "int a; int b[2]; int64_t c[(1 + 2)]; bool d; float e; vec3 f[2]; "
    "i64vec3 g[2]; uint h;"
)
STD140_ARRAY_MEMBERS = [
    ("a", 0, 4, None, None),
    ("b", 16, 32, 2, 16),
    ("c", 48, 48, 3, 16),
    ("d", 96, 4, None, None),
    ("e", 100, 4, None, None),
    ("f", 112, 32, 2, 16),
    ("g", 160, 64, 2, 32),
    ("h", 224, 4, None, None),
]


def test_uniform_reflection_lays_out_fixed_arrays_like_glslang(tmp_path):
    result = _reflect(tmp_path, STD140_ARRAY_BODY)
    assert result["status"] == "ready", result
    layout = result["resources"][0]["scalarLayout"]
    assert [
        (
            member["name"],
            member["offsetBytes"],
            member["sizeBytes"],
            member.get("arrayLength"),
            member.get("arrayStrideBytes"),
        )
        for member in layout["blockMembers"]
    ] == STD140_ARRAY_MEMBERS
    # i64vec3 arrays align to 32 bytes, so the block rounds up to 256.
    assert layout["alignmentBytes"] == 32 and layout["blockSizeBytes"] == 256
    value = RuntimeValue(name="p", dtype="uint32", shape=(64,), values=[0] * 64)
    assert (
        _validated_scalar_layout(
            layout,
            runtime_value=value,
            target="opengl",
            resource_kind="constant-buffer",
            path="$.layout",
        )
        == layout
    )


def _reflect_hlsl(tmp_path, body):
    artifact = tmp_path / "params.hlsl"
    artifact.write_text(
        f"""struct Parameters {{ {body} }};
ConstantBuffer<Parameters> p : register(b0);
RWStructuredBuffer<float> o : register(u1);
[numthreads(1, 1, 1)]
void CSMain() {{ o[0] = 0.0; }}
""",
        encoding="utf-8",
    )
    result = reflect_target_host_interface(artifact, target="directx", stage="compute")
    (resource,) = (item for item in result["resources"] if item["name"] == "p")
    return resource


# Offsets DXC 1.9 prints for ConstantBuffer<Parameters> (-Fc, cs_6_2).
HLSL_CONSTANT_BUFFER_BODY = (
    "int a; int b[2]; int64_t c[(1 + 2)]; bool d; float e; uint h; int64_t k; "
    "uint m[1]; float n;"
)
HLSL_CONSTANT_BUFFER_MEMBERS = [
    ("a", 0, 4, None, None),
    ("b", 16, 20, 2, 16),
    ("c", 48, 40, 3, 16),
    ("d", 88, 4, None, None),
    ("e", 92, 4, None, None),
    ("h", 96, 4, None, None),
    ("k", 104, 8, None, None),
    ("m", 112, 4, 1, 16),
    ("n", 116, 4, None, None),
]


def test_directx_constant_buffer_records_follow_legacy_packing(tmp_path):
    resource = _reflect_hlsl(tmp_path, HLSL_CONSTANT_BUFFER_BODY)
    assert resource["kind"] == "constant-buffer"
    layout = resource["scalarLayout"]
    assert layout["storageLayout"] == "hlsl-constant-buffer"
    assert layout["payloadEncoding"] == "uint32-le-words"
    assert [
        (
            member["name"],
            member["offsetBytes"],
            member["sizeBytes"],
            member.get("arrayLength"),
            member.get("arrayStrideBytes"),
        )
        for member in layout["blockMembers"]
    ] == HLSL_CONSTANT_BUFFER_MEMBERS
    # A member after an array shares the last element's register; the block
    # covers whole 16-byte registers.
    assert layout["blockSizeBytes"] == 128
    assert validate_uniform_block_layout(layout) == 128
    value = RuntimeValue(name="p", dtype="uint32", shape=(32,), values=[0] * 32)
    assert (
        _validated_scalar_layout(
            layout,
            runtime_value=value,
            target="directx",
            resource_kind="constant-buffer",
            path="$.layout",
        )
        == layout
    )
    for target, count in (("opengl", 32), ("directx", 31), ("directx", 33)):
        value = RuntimeValue(
            name="p", dtype="uint32", shape=(count,), values=[0] * count
        )
        with pytest.raises(NativeLoaderDispatchError, match="resource-layout-mismatch"):
            _validated_scalar_layout(
                layout,
                runtime_value=value,
                target=target,
                resource_kind="constant-buffer",
                path="$.layout",
            )


def test_directx_constant_buffer_matches_mlx_conv_params(tmp_path):
    # MLXConvParams<2> as DirectX emits it; DXC reports groups at 424, flip at
    # 428 and a 432-byte buffer (std140 would put them at 432 and 436).
    body = (
        "int N; int C; int O; int iS[2]; int wS[2]; int oS[2]; int str[2]; "
        "int pad[2]; int kdil[2]; int idil[2]; int64_t in_strides[(2 + 2)]; "
        "int64_t wt_strides[(2 + 2)]; int64_t out_strides[(2 + 2)]; int groups; "
        "bool flip;"
    )
    layout = _reflect_hlsl(tmp_path, body)["scalarLayout"]
    offsets = {
        member["name"]: member["offsetBytes"] for member in layout["blockMembers"]
    }
    assert offsets == {
        "N": 0,
        "C": 4,
        "O": 8,
        "iS": 16,
        "wS": 48,
        "oS": 80,
        "str": 112,
        "pad": 144,
        "kdil": 176,
        "idil": 208,
        "in_strides": 240,
        "wt_strides": 304,
        "out_strides": 368,
        "groups": 424,
        "flip": 428,
    }
    assert layout["blockSizeBytes"] == 432
    std140 = std140_block_layout(
        "MLXConvParams_2",
        [
            (
                (member["physicalType"], member["name"], member["arrayLength"])
                if "arrayLength" in member
                else (member["physicalType"], member["name"])
            )
            for member in layout["blockMembers"]
        ],
    )
    assert [member["offsetBytes"] for member in std140["blockMembers"][-2:]] == [
        432,
        436,
    ]
    assert std140["blockSizeBytes"] == 448


@pytest.mark.parametrize(
    "body",
    [
        "half a; int b;",
        "float2 a; int b;",
        "float a; float b[COUNT];",
        "float a; float b[0];",
        "float a; Sample b;",
        "float a; float a;",
    ],
)
def test_directx_constant_buffer_rejects_unproven_members(tmp_path, body):
    assert "scalarLayout" not in _reflect_hlsl(tmp_path, body)


def test_uniform_block_packing_places_array_elements_at_their_strides(tmp_path):
    layout = _reflect(tmp_path, STD140_ARRAY_BODY)["resources"][0]["scalarLayout"]
    values = {
        "a": -3,
        "b": [5, 7],
        "c": [-(2**40), 11, 2**62],
        "d": True,
        "e": 0.5,
        "f": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "g": [[1, -2, 3], [2**33, 5, -6]],
        "h": 2**32 - 1,
    }
    words = pack_uniform_block(layout, values)
    payload = struct.pack(f"<{len(words)}I", *words)
    assert len(payload) == 256
    expected = bytearray(256)
    struct.pack_into("<i", expected, 0, -3)
    struct.pack_into("<i", expected, 16, 5)
    struct.pack_into("<i", expected, 32, 7)
    for index, item in enumerate(values["c"]):
        struct.pack_into("<q", expected, 48 + 16 * index, item)
    struct.pack_into("<I", expected, 96, 1)
    struct.pack_into("<f", expected, 100, 0.5)
    struct.pack_into("<3f", expected, 112, 1.0, 2.0, 3.0)
    struct.pack_into("<3f", expected, 128, 4.0, 5.0, 6.0)
    struct.pack_into("<3q", expected, 160, 1, -2, 3)
    struct.pack_into("<3q", expected, 192, 2**33, 5, -6)
    struct.pack_into("<I", expected, 224, 2**32 - 1)
    assert payload == bytes(expected)

    hlsl = _reflect_hlsl(tmp_path, HLSL_CONSTANT_BUFFER_BODY)["scalarLayout"]
    words = pack_uniform_block(
        hlsl,
        {
            "a": 1,
            "b": [2, 3],
            "c": [4, 5, 6],
            "d": False,
            "e": 7.0,
            "h": 8,
            "k": -9,
            "m": [10],
            "n": 11.5,
        },
    )
    expected = bytearray(128)
    for offset, code, item in (
        (0, "i", 1),
        (16, "i", 2),
        (32, "i", 3),
        (48, "q", 4),
        (64, "q", 5),
        (80, "q", 6),
        (88, "I", 0),
        (92, "f", 7.0),
        (96, "I", 8),
        (104, "q", -9),
        (112, "I", 10),
        (116, "f", 11.5),
    ):
        struct.pack_into("<" + code, expected, offset, item)
    assert struct.pack(f"<{len(words)}I", *words) == bytes(expected)


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda values: values.pop("h"), "exactly the reflected members"),
        (lambda values: values.update(extra=1), "exactly the reflected members"),
        (lambda values: values.update(b=[5]), "requires 2 elements"),
        (lambda values: values.update(b=5), "requires 2 elements"),
        (lambda values: values.update(d=1), "requires bool"),
        (lambda values: values.update(a=1.5), "requires int32"),
        (lambda values: values.update(a=True), "requires int32"),
        (lambda values: values.update(h=-1), "outside uint32"),
        (lambda values: values.update(f=[[1.0, 2.0], [3.0, 4.0]]), "3 components"),
    ],
)
def test_uniform_block_packing_rejects_mismatched_values(tmp_path, change, message):
    layout = _reflect(tmp_path, STD140_ARRAY_BODY)["resources"][0]["scalarLayout"]
    values = {
        "a": 0,
        "b": [0, 0],
        "c": [0, 0, 0],
        "d": False,
        "e": 0.0,
        "f": [[0.0] * 3] * 2,
        "g": [[0] * 3] * 2,
        "h": 0,
    }
    change(values)
    with pytest.raises(ValueError, match=message):
        pack_uniform_block(layout, values)


@pytest.mark.parametrize(
    "storage,member,field,value",
    [
        ("std140", 1, "arrayStrideBytes", 4),
        ("std140", 1, "arrayLength", 3),
        ("std140", 1, "arrayLength", True),
        ("std140", 0, "arrayLength", 1),
        ("std140", 6, "arrayStrideBytes", 16),
        ("hlsl", 1, "sizeBytes", 32),
        ("hlsl", 3, "offsetBytes", 96),
        ("hlsl", 1, "arrayStrideBytes", 4),
        ("hlsl", 7, "arrayLength", None),
    ],
)
def test_uniform_block_validation_recomputes_array_packing(
    tmp_path, storage, member, field, value
):
    if storage == "std140":
        layout = _reflect(tmp_path, STD140_ARRAY_BODY)["resources"][0]["scalarLayout"]
    else:
        layout = _reflect_hlsl(tmp_path, HLSL_CONSTANT_BUFFER_BODY)["scalarLayout"]
    layout["blockMembers"][member][field] = value
    with pytest.raises(ValueError):
        validate_uniform_block_layout(layout)


def test_uniform_block_validation_keeps_layouts_apart(tmp_path):
    hlsl = _reflect_hlsl(tmp_path, HLSL_CONSTANT_BUFFER_BODY)["scalarLayout"]
    with pytest.raises(ValueError):
        validate_std140_block_layout(hlsl)
    relabeled = dict(hlsl, storageLayout="std140")
    with pytest.raises(ValueError):
        validate_uniform_block_layout(relabeled)
    with pytest.raises(ValueError):
        validate_uniform_block_layout(dict(hlsl, storageLayout="std430"))


def _directx_binding(layout, count):
    return NativeRuntimeBufferBinding(
        name="p",
        binding=RuntimeResourceBinding(
            name="p",
            kind="constant-buffer",
            binding=0,
            set=0,
            access="read",
            metadata={"scalarLayout": layout},
        ),
        dtype="uint32",
        shape=(count,),
        value=[0] * count,
        source="input",
    )


def test_directx_runtime_uploads_complete_constant_buffer_records(tmp_path):
    layout = _reflect_hlsl(tmp_path, HLSL_CONSTANT_BUFFER_BODY)["scalarLayout"]
    (prepared,) = _prepare_directx_buffers({"p": _directx_binding(layout, 32)})
    assert prepared.namespace == "cbv"
    assert prepared.payload == b"\x00" * 128
    # Constant-buffer views cover whole 256-byte placements.
    assert prepared.allocation_size == 256
    for count in (31, 33):
        with pytest.raises(RuntimeAdapterSetupError) as exc:
            _prepare_directx_buffers({"p": _directx_binding(layout, count)})
        assert exc.value.details["reasonKind"] == "uniform-block-payload-mismatch"
    std140 = _reflect(tmp_path, STD140_ARRAY_BODY)["resources"][0]["scalarLayout"]
    with pytest.raises(RuntimeAdapterSetupError) as exc:
        _prepare_directx_buffers({"p": _directx_binding(std140, 64)})
    assert exc.value.details["reasonKind"] == "uniform-block-payload-mismatch"
    with pytest.raises(RuntimeAdapterSetupError) as exc:
        _prepare_opengl_buffers({"p": _binding(layout, count=32)})
    assert exc.value.details["reasonKind"] == "uniform-block-payload-mismatch"
