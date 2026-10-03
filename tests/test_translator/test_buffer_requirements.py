"""Proven buffer footprints through reflection, packaging and native preflight."""

import copy
import json
import os
import sys
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
from crosstl.project.runtime_verification import (
    RuntimeAllocationView,
    prepare_runtime_execution,
)
from tests.test_translator.test_native_loader_dispatch_integration import _executor


def _reflect(tmp_path, target, body, helper="", prefix=""):
    if target == "directx":
        source = f"""{prefix}
        StructuredBuffer<int> data : register(t0);
        RWStructuredBuffer<int> result : register(u0);
        {helper}
        [numthreads(1,1,1)] void CSMain(uint3 tid : SV_DispatchThreadID) {{ {body} }}
        """
    else:
        source = f"""#version 450 core
        {prefix}
        layout(std430,binding=0) readonly buffer Input {{ int data[]; }};
        layout(std430,binding=1) buffer Output {{ int result[]; }};
        layout(local_size_x=1) in;
        {helper}
        void main() {{ uvec3 tid = gl_GlobalInvocationID; {body} }}
        """
    path = tmp_path / ("kernel.hlsl" if target == "directx" else "kernel.comp")
    path.write_text(source, encoding="utf-8")
    reflected = reflect_target_host_interface(path, target=target, stage="compute")
    return {
        resource["scalarLayout"].get("memberName", resource["name"]): (
            resource["scalarLayout"].get("minimumBindingSizeBytes")
        )
        for resource in reflected["resources"]
        if "scalarLayout" in resource
    }


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "body,minimum",
    [
        ("result[0] = data[1];", 8),
        ("int offset = 2; result[0] = data[offset + 1];", 16),
        ("result[0] = data[int(3)];", 16),
        ("result[0] = data[0x2u];", 12),
        ("result[0] = data[2u];", 12),
        ("result[0] = data[2ul];", 12),
        ("result[0] = data[2ull];", 12),
        ("result[0] = data[2147483648u];", None),
        ("result[0] = data[0xffffffffffffffffull];", None),
        ("result[0] = data[tid.x];", None),
        ("int data[4]; result[0] = data[3];", None),
        ("if (tid.x > 0) { result[0] = data[99]; }", None),
        ("if (tid.x > 0) return; result[0] = data[99];", None),
        ("for (int i=0; i<int(tid.x); ++i) { result[0] = data[99]; }", None),
        ("int i=99; i=0; result[0] = data[i];", None),
        ("result[0] = data[2147483647 + 1];", None),
        ("result[0] = data[uint(-1) + 2];", None),
        ("return; result[0] = data[99];", None),
        ("/* data[99] */ result[0] = 1;", None),
        ("result[0] = data[1]; if (tid.x > 0) result[0] = data[99];", 8),
    ],
)
def test_reflects_only_proven_prefix_footprints(tmp_path, target, body, minimum):
    assert _reflect(tmp_path, target, body)["data"] == minimum


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_follows_unique_helper_and_constant_offsets(tmp_path, target):
    if target == "directx":
        helper = """int read_at(StructuredBuffer<int> values, int offset) {
            return values[offset + 1];
        }
        int read_outer(StructuredBuffer<int> values) { return read_at(values, 2); }
        """
        call = "read_outer(data)"
    else:
        helper = """int read_at(int offset) { return data[offset + 1]; }
        int read_outer() { return read_at(2); }
        """
        call = "read_outer()"
    assert _reflect(tmp_path, target, f"result[0] = {call};", helper)["data"] == 16


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "helper,body",
    [
        ("int unused() { return data[99]; }", "result[0] = 1;"),
        (
            "int read_at(int i) { return data[99]; } "
            "int read_at(float i) { return data[1]; }",
            "result[0] = read_at(1);",
        ),
        ("int recurse() { return recurse(); }", "recurse(); result[0] = data[99];"),
        (
            "int mutate(inout int i) { i=0; return 1; }",
            "int i=99; result[0] = data[i] + mutate(i);",
        ),
        (
            "float index_value() { return 16777217; }",
            "result[0] = data[int(index_value())];",
        ),
    ],
)
def test_unproven_helper_calls_do_not_add_requirements(tmp_path, target, helper, body):
    assert _reflect(tmp_path, target, body, helper)["data"] is None


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_does_not_analyze_unresolved_preprocessing(tmp_path, target):
    result = _reflect(
        tmp_path,
        target,
        "result[0] = data[INDEX];",
        prefix="#define INDEX 99",
    )
    assert result["data"] is None


def test_structured_load_requires_indexed_element(tmp_path):
    assert _reflect(tmp_path, "directx", "result[0] = data.Load(3);")["data"] == 16


def test_named_glsl_block_preserves_resource_identity(tmp_path):
    artifact = tmp_path / "instance.comp"
    artifact.write_text(
        """#version 450 core
        layout(std430,binding=0) readonly buffer Input { int values[]; } source;
        layout(std430,binding=1) buffer Output { int values[]; } result;
        layout(local_size_x=1) in;
        void main() { result.values[0] = source.values[2]; }
        """,
        encoding="utf-8",
    )
    reflected = reflect_target_host_interface(artifact, target="opengl")
    assert {
        resource["name"]: resource["scalarLayout"]["minimumBindingSizeBytes"]
        for resource in reflected["resources"]
    } == {"source": 12, "result": 4}


def test_multiple_entry_points_do_not_share_footprints(tmp_path):
    result = _reflect(
        tmp_path,
        "directx",
        "result[0] = data[1];",
        helper="[numthreads(1,1,1)] void other() { result[0] = data[99]; }",
    )
    assert result["data"] is None


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_array_parameter_declaration_is_not_a_minimum_contract(tmp_path, target):
    result = _reflect(
        tmp_path,
        target,
        "int local[128]; result[0] = unused(local);",
        helper="int unused(int data[128]) { return 7; }",
    )
    assert result["data"] is None


def _package(tmp_path, target):
    (tmp_path / "stride.metal").write_text(
        """#include <metal_stdlib>
        using namespace metal;
        int lookup(device const int* offsets) { return offsets[1]; }
        kernel void select_value(device const float* input [[buffer(0)]],
                                 device const int* offsets [[buffer(1)]],
                                 device float* result [[buffer(2)]]) {
            result[0] = input[lookup(offsets)];
        }
        """,
        encoding="utf-8",
    )
    config = tmp_path / "crosstl.toml"
    config.write_text(
        f"""[project]
source_roots = ["."]
include = ["stride.metal"]
targets = ["{target}"]
output_dir = "out"
[project.entry_points]
"stride.metal" = "select_value"
""",
        encoding="utf-8",
    )
    report = translate_project(
        load_project_config(tmp_path, config), format_output=False
    )
    report.write_json(tmp_path / "report.json")
    assert report.to_json()["summary"]["failedCount"] == 0
    manifest = build_runtime_artifact_manifest(tmp_path / "report.json")
    assert manifest["success"], manifest
    manifest_path = tmp_path / "artifacts.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    package = tmp_path / "package"
    assert build_runtime_package(manifest_path, package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"], loader
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    bindings = sorted(descriptor["bindings"], key=lambda b: b["coordinates"]["binding"])
    input_name, offsets_name, result_name = (b["name"] for b in bindings)
    assert bindings[1]["scalarLayout"]["minimumBindingSizeBytes"] == 8
    inputs = {
        input_name: {"dtype": "float32", "shape": [3], "values": [7, 11, 13]},
        offsets_name: {"dtype": "int32", "shape": [2], "values": [0, 2]},
        result_name: {"dtype": "float32", "shape": [1], "values": [-1234]},
    }
    outputs = {result_name: {"dtype": "float32", "shape": [1], "values": [13]}}
    return descriptor, package, inputs, outputs, offsets_name


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_package_dispatch_enforces_minimum_not_exact_extent(tmp_path, target, count):
    descriptor, package, inputs, outputs, offsets = _package(tmp_path, target)
    inputs[offsets].update(shape=[count], values=[0, 2, 0][:count])
    if count == 1:
        with pytest.raises(NativeLoaderDispatchError) as caught:
            build_native_loader_dispatch_request(
                descriptor, package, inputs, outputs, [1, 1, 1], expected_target=target
            )
        diagnostics = caught.value.details["diagnostics"]
        assert any(d["code"].endswith("resource-view-too-small") for d in diagnostics)
    else:
        request = build_native_loader_dispatch_request(
            descriptor, package, inputs, outputs, [1, 1, 1], expected_target=target
        )
        assert request.execution_plan.diagnostics == ()


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_minimum_applies_to_view_not_larger_allocation(tmp_path, target):
    descriptor, package, inputs, outputs, offsets = _package(tmp_path, target)
    request = build_native_loader_dispatch_request(
        descriptor, package, inputs, outputs, [1, 1, 1], expected_target=target
    )
    fixture = replace(
        request.fixture,
        inputs=tuple(
            (
                replace(
                    value,
                    shape=(1,),
                    values=[0],
                    allocation=RuntimeAllocationView(
                        allocation_id="shared",
                        byte_offset=16,
                        byte_length=4,
                        allocation_byte_length=1024,
                    ),
                )
                if value.name == offsets
                else value
            )
            for value in request.fixture.inputs
        ),
    )
    plan = prepare_runtime_execution(
        replace(request, fixture=fixture, execution_plan=None)
    )
    resource = next(r for r in plan.resource_bindings if r.binding.name == offsets)
    assert resource.allocation.byte_offset == 16
    assert resource.allocation.allocation_byte_length == 1024
    assert any(d["code"].endswith("resource-view-too-small") for d in plan.diagnostics)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_value_layout_cannot_weaken_binding_minimum(tmp_path, target):
    descriptor, package, inputs, outputs, offsets = _package(tmp_path, target)
    layout = copy.deepcopy(
        next(b["scalarLayout"] for b in descriptor["bindings"] if b["name"] == offsets)
    )
    layout.update(elementStrideBytes=4096, minimumBindingSizeBytes=1)
    inputs[offsets].update(shape=[1], values=[0], metadata={"scalarLayout": layout})
    with pytest.raises(NativeLoaderDispatchError) as caught:
        build_native_loader_dispatch_request(
            descriptor, package, inputs, outputs, [1, 1, 1], expected_target=target
        )
    assert any(
        d["code"].endswith("resource-view-too-small")
        for d in caught.value.details["diagnostics"]
    )


@pytest.mark.parametrize("minimum", [True, 0, -1, 1.5, "8", 1 << 63])
def test_rejects_malformed_minimum_contract(tmp_path, minimum):
    descriptor, package, inputs, outputs, offsets = _package(tmp_path, "directx")
    for binding in descriptor["bindings"]:
        if binding["name"] == offsets:
            binding["scalarLayout"]["minimumBindingSizeBytes"] = minimum
    for binding in descriptor["scalarLayout"]["bindings"]:
        if binding["binding"] == offsets:
            binding["layout"]["minimumBindingSizeBytes"] = minimum
    with pytest.raises(NativeLoaderDispatchError, match="minimum-binding-size-invalid"):
        build_native_loader_dispatch_request(
            descriptor, package, inputs, outputs, [1, 1, 1]
        )


def test_minimum_footprint_native_readback(tmp_path):
    target = {"win32": "directx", "linux": "opengl"}.get(sys.platform)
    if os.environ.get("CROSTL_REQUIRE_STRUCT_BUFFER_RUNTIME") != "1" or target is None:
        pytest.skip("requires native Windows or Linux buffer validation")
    descriptor, package, inputs, outputs, _ = _package(tmp_path, target)
    request = build_native_loader_dispatch_request(
        descriptor, package, inputs, outputs, [1, 1, 1], expected_target=target
    )
    assert request.execution_plan.diagnostics == ()
    executor = _executor(target)
    availability = executor.is_available(request)
    assert availability.available, availability.reason
    result = executor.run(request)
    (tmp_path / "readback.json").write_text(
        json.dumps(result.outputs), encoding="utf-8"
    )
    assert result.status == "ok"
    for name, expected in outputs.items():
        assert result.outputs[name]["values"] == expected["values"]
