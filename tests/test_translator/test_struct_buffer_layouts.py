"""Exact homogeneous struct storage through reflection and native dispatch."""

import copy
import json
import os
import sys

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
from crosstl.project.runtime_verification import (
    RuntimeValue,
    _runtime_scalar_layout_signature,
)
from tests.test_translator.test_native_loader_dispatch_integration import _executor


def _reflect(tmp_path, target, body, *, declaration=None):
    declaration = declaration or f"struct Sample {{ {body} }};"
    if target == "directx":
        source = f"""{declaration}
        StructuredBuffer<Sample> values : register(t0);
        [numthreads(1, 1, 1)] void CSMain() {{}}
        """
    else:
        source = f"""#version 450 core
        #extension GL_ARB_gpu_shader_int64 : require
        {declaration}
        layout(std430, binding=0) readonly buffer Values {{ Sample values[]; }};
        layout(local_size_x=1) in;
        void main() {{}}
        """
    artifact = tmp_path / ("sample.hlsl" if target == "directx" else "sample.comp")
    artifact.write_text(source, encoding="utf-8")
    result = reflect_target_host_interface(artifact, target=target, stage="compute")
    assert result["status"] == "ready"
    assert len(result["resources"]) == 1
    return result["resources"][0]


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "scalar,dtype,size",
    [
        ("float", "float32", 4),
        ("int", "int32", 4),
        ("uint", "uint32", 4),
        ("int64_t", "int64", 8),
        ("uint64_t", "uint64", 8),
    ],
)
@pytest.mark.parametrize("count", [1, 2, 3, 5, 64])
def test_struct_reflection_retains_exact_members(
    tmp_path, target, scalar, dtype, size, count
):
    resource = _reflect(
        tmp_path, target, " ".join(f"{scalar} field{i};" for i in range(count))
    )
    layout = resource["scalarLayout"]
    expected = {
        "physicalType": "Sample",
        "elementType": dtype,
        "elementSizeBytes": size * count,
        "elementStrideBytes": size * count,
        "alignmentBytes": size,
        "memberOffsetBytes": 0,
        "storageLayout": "hlsl-structured-buffer" if target == "directx" else "std430",
        "runtimeSized": True,
        "componentCount": count,
        "structMembers": [
            {"name": f"field{i}", "physicalType": scalar, "offsetBytes": i * size}
            for i in range(count)
        ],
    }
    if target == "opengl":
        expected["memberName"] = "values"
    assert layout == expected
    value = RuntimeValue(
        name="values", dtype=dtype, shape=(2, count), values=[0] * (2 * count)
    )
    assert (
        _validated_scalar_layout(
            layout,
            runtime_value=value,
            target=target,
            resource_kind="buffer",
            path="$.layout",
        )
        == layout
    )


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "body",
    [
        "",
        "float first; int second;",
        "float first[2];",
        "float2 first;",
        "Sample nested;",
        "float first; float first;",
        "float first; float second : TEXCOORD0;",
        "float first; float value() { return first; }",
        "bool first; bool second;",
        "half first; half second;",
        " ".join(f"float f{i};" for i in range(65)),
    ],
)
def test_struct_reflection_leaves_unproven_layout_unavailable(tmp_path, target, body):
    assert "scalarLayout" not in _reflect(tmp_path, target, body)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_struct_reflection_rejects_duplicate_declarations(tmp_path, target):
    resource = _reflect(
        tmp_path,
        target,
        "",
        declaration="struct Sample { float first; }; struct Sample { float second; };",
    )
    assert "scalarLayout" not in resource


@pytest.mark.parametrize(
    "layout",
    [
        "std430, std140",
        "packed",
        "shared",
        "std430, align=16",
        "std430, offset=4",
        "std430, row_major",
        "align=16) layout(std430",
    ],
)
def test_struct_reflection_rejects_unproven_glsl_storage_layout(tmp_path, layout):
    artifact = tmp_path / "ambiguous.comp"
    artifact.write_text(
        f"""#version 450 core
        struct Sample {{ float first; float second; }};
        layout({layout}, binding=0) buffer Values {{ Sample values[]; }};
        layout(local_size_x=1) in;
        void main() {{}}
    """,
        encoding="utf-8",
    )
    result = reflect_target_host_interface(artifact, target="opengl", stage="compute")
    assert "scalarLayout" not in result["resources"][0]


@pytest.mark.parametrize(
    "mutation",
    [
        {"componentCount": True},
        {"componentCount": 3},
        {"componentCount": 0},
        {"elementSizeBytes": 4},
        {"elementStrideBytes": 16},
        {"alignmentBytes": 8},
        {"memberOffsetBytes": True},
        {"runtimeSized": False},
        {"vectorWidth": 2},
        {"physicalType": "float"},
        {"physicalType": "Sample[]"},
        {"elementType": "uint32"},
        {"structMembers": None},
        {
            "structMembers": [
                {"name": "first", "physicalType": "float", "offsetBytes": 0},
                {"name": "first", "physicalType": "float", "offsetBytes": 4},
            ]
        },
        {
            "structMembers": [
                {"name": "first", "physicalType": "float", "offsetBytes": 0},
                {"name": "second", "physicalType": "int", "offsetBytes": 4},
            ]
        },
        {
            "structMembers": [
                {"name": "first", "physicalType": "float", "offsetBytes": False},
                {"name": "second", "physicalType": "float", "offsetBytes": 4},
            ]
        },
        {
            "structMembers": [
                {"name": "first", "physicalType": "float", "offsetBytes": 0},
                {"name": "second", "physicalType": "float", "offsetBytes": 8},
            ]
        },
    ],
)
def test_struct_dispatch_rejects_forged_layout(tmp_path, mutation):
    layout = _reflect(tmp_path, "opengl", "float first; float second;")["scalarLayout"]
    layout.update(copy.deepcopy(mutation))
    value = RuntimeValue(
        name="values", dtype="float32", shape=(2, 2), values=[1, 2, 3, 4]
    )
    with pytest.raises(NativeLoaderDispatchError, match="resource-layout-unsupported"):
        _validated_scalar_layout(
            layout,
            runtime_value=value,
            target="opengl",
            resource_kind="buffer",
            path="$.layout",
        )


@pytest.mark.parametrize("kind,shape", [("buffer", (3,)), ("constant-buffer", (2,))])
def test_struct_dispatch_rejects_incomplete_elements_and_uniforms(
    tmp_path, kind, shape
):
    layout = _reflect(tmp_path, "opengl", "float first; float second;")["scalarLayout"]
    value = RuntimeValue(
        name="values", dtype="float32", shape=shape, values=[1] * shape[0]
    )
    with pytest.raises(NativeLoaderDispatchError):
        _validated_scalar_layout(
            layout,
            runtime_value=value,
            target="opengl",
            resource_kind=kind,
            path="$.layout",
        )


def _package(tmp_path, target):
    (tmp_path / "pair.metal").write_text(
        """#include <metal_stdlib>
    using namespace metal;
    struct Sample { float first; float second; };
    kernel void transform(device const Sample* input [[buffer(0)]],
                          device Sample* result [[buffer(1)]],
                          uint index [[thread_position_in_grid]]) {
        result[index].first = input[index].first * 2.0f + 1.0f;
        result[index].second = input[index].second * 3.0f - 2.0f;
    }
    """,
        encoding="utf-8",
    )
    config = tmp_path / "crosstl.toml"
    config.write_text(
        f"""[project]
source_roots = ["."]
include = ["pair.metal"]
targets = ["{target}"]
output_dir = "out"
[project.entry_points]
"pair.metal" = "transform"
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
    package_path = tmp_path / "package"
    assert build_runtime_package(tmp_path / "artifacts.json", package_path)["success"]
    loader = build_runtime_loader_manifest(package_path / "runtime-package.json")
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
    inputs = {
        resources[0]["name"]: {
            "dtype": "float32",
            "shape": [3, 2],
            "values": [2, 4, -3, 0.5, 0, -2],
        }
    }
    outputs = {
        resources[1]["name"]: {
            "dtype": "float32",
            "shape": [3, 2],
            "values": [5, 10, -5, -0.5, 1, -8],
        }
    }
    request = build_native_loader_dispatch_request(
        descriptor, package_path, inputs, outputs, [3, 1, 1], expected_target=target
    )
    return request, outputs


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_struct_layout_survives_public_package_dispatch(tmp_path, target):
    request, _ = _package(tmp_path, target)
    assert request.execution_plan.diagnostics == ()
    for binding in request.adapter_contract.resource_bindings:
        assert binding.metadata["byteStride"] == 8
        assert binding.metadata["scalarLayout"]["componentCount"] == 2
        signature = _runtime_scalar_layout_signature(binding.metadata)
        assert (
            signature["structMembers"]
            == binding.metadata["scalarLayout"]["structMembers"]
        )
    for resource in request.execution_plan.resource_bindings:
        assert resource.allocation.byte_length == 24


def test_struct_buffer_native_readback(tmp_path):
    target = {"win32": "directx", "linux": "opengl"}.get(sys.platform)
    if os.environ.get("CROSTL_REQUIRE_STRUCT_BUFFER_RUNTIME") != "1" or target is None:
        pytest.skip("requires the native Windows or Linux struct-buffer gate")
    request, expected = _package(tmp_path, target)
    executor = _executor(target)
    availability = executor.is_available(request)
    assert availability.available, availability.reason
    result = executor.run(request)
    (tmp_path / "outputs.json").write_text(
        json.dumps(result.outputs, indent=2), encoding="utf-8"
    )
    assert result.status == "ok"
    for name, value in expected.items():
        assert result.outputs[name]["values"] == value["values"]
