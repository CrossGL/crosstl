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
from crosstl.project.native_runtime_drivers import _prepare_opengl_buffers
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    RuntimeAdapterSetupError,
    RuntimeAllocationView,
    RuntimeResourceBinding,
    RuntimeValue,
    _runtime_scalar_layout_signature,
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
        "int a; float b[2];",
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


def _package(tmp_path, vectors):
    if vectors:
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
        """[project]
source_roots = ["."]
include = ["parameters.metal"]
targets = ["opengl"]
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
        descriptor, package, inputs, outputs, [8, 1, 1], expected_target="opengl"
    )
    assert request.execution_plan.diagnostics == ()
    binding = request.adapter_contract.resource_bindings[0]
    assert binding.metadata["scalarLayout"] == layout
    return request, outputs


@pytest.mark.parametrize("vectors", [False, True])
def test_uniform_layout_survives_public_package_dispatch(tmp_path, vectors):
    _package(tmp_path, vectors)


@pytest.mark.parametrize("vectors", [False, True])
def test_uniform_block_native_readback(tmp_path, vectors):
    if os.environ.get("CROSTL_REQUIRE_UNIFORM_BLOCK_RUNTIME") != "1":
        pytest.skip("requires the native OpenGL uniform-block gate")
    request, expected = _package(tmp_path, vectors)
    executor = _executor("opengl")
    availability = executor.is_available(request)
    assert availability.available, availability.reason
    result = executor.run(request)
    (tmp_path / "outputs.json").write_text(
        json.dumps(result.outputs, indent=2), encoding="utf-8"
    )
    assert result.status == "ok"
    for name, value in expected.items():
        assert result.outputs[name]["values"] == value["values"]
