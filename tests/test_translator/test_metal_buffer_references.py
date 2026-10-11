"""Fixed device references retain their physical layout through native dispatch."""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.project import (
    NativeLoaderDispatchError,
    build_native_loader_dispatch_request,
    load_project_config,
    translate_project,
)
from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.project.native_loader_abi import _binding_descriptors
from crosstl.project.native_loader_dispatch import _validated_scalar_layout
from crosstl.project.runtime_verification import RuntimeAllocationView, RuntimeValue
from tests.runtime_helpers import _prepare_native_package, _validate
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_native_loader_dispatch import _write_descriptor
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REFERENCE_TARGETS = [
    ("metal", 1),
    ("metal", 2),
    ("metal", 4),
    ("directx", 1),
    ("directx", 2),
    ("directx", 4),
    ("opengl", 1),
    ("opengl", 2),
    ("opengl", 4),
]


def _reflect(tmp_path, declaration):
    path = tmp_path / "reference.metal"
    path.write_text(
        f"kernel void read_value({declaration} value [[buffer(0)]]) {{}}\n",
        encoding="utf-8",
    )
    reflected = reflect_target_host_interface(path, target="metal", stage="compute")
    assert reflected["status"] == "ready"
    return reflected["resources"][0]


@pytest.mark.parametrize("qualifiers", ["device", "device const", "const device"])
@pytest.mark.parametrize(
    "source_type,physical,dtype,size,width",
    [
        ("bool", "bool", "bool", 1, 1),
        ("char", "char", "int8", 1, 1),
        ("int8_t", "char", "int8", 1, 1),
        ("uchar", "uchar", "uint8", 1, 1),
        ("uint8_t", "uchar", "uint8", 1, 1),
        ("short", "short", "int16", 2, 1),
        ("int16_t", "short", "int16", 2, 1),
        ("ushort", "ushort", "uint16", 2, 1),
        ("uint16_t", "ushort", "uint16", 2, 1),
        ("half", "half", "float16", 2, 1),
        ("bfloat", "bfloat", "bfloat16", 2, 1),
        ("int", "int", "int32", 4, 1),
        ("uint", "uint", "uint32", 4, 1),
        ("float", "float", "float32", 4, 1),
        ("long", "int64_t", "int64", 8, 1),
        ("int64_t", "int64_t", "int64", 8, 1),
        ("ulong", "uint64_t", "uint64", 8, 1),
        ("uint64_t", "uint64_t", "uint64", 8, 1),
        ("half2", "half2", "float16", 4, 2),
        ("int2", "int2", "int32", 8, 2),
        ("float4", "float4", "float32", 16, 4),
    ],
)
def test_reflects_natural_device_reference_layout(
    tmp_path, qualifiers, source_type, physical, dtype, size, width
):
    resource = _reflect(tmp_path, f"{qualifiers} {source_type}&")
    layout = resource["scalarLayout"]
    expected = {
        "physicalType": physical,
        "elementType": dtype,
        "elementSizeBytes": size,
        "elementStrideBytes": size,
        "alignmentBytes": size,
        "memberOffsetBytes": 0,
        "storageLayout": "metal-buffer",
        "runtimeSized": False,
        "blockSizeBytes": size,
        "minimumBindingSizeBytes": size,
    }
    if width != 1:
        expected["vectorWidth"] = width
    assert layout == expected
    assert resource["kind"] == "buffer"
    assert resource["access"] == ("read" if "const" in qualifiers else "read_write")
    assert (
        _validated_scalar_layout(
            layout,
            runtime_value=RuntimeValue(name="value", dtype=dtype, shape=(width,)),
            target="metal",
            resource_kind="buffer",
            path="$.layout",
        )
        == expected
    )


@pytest.mark.parametrize("qualifiers", ["device", "constant"])
@pytest.mark.parametrize("indirection", ["*", "&"])
def test_preserves_pointer_and_constant_reference_layouts(
    tmp_path, qualifiers, indirection
):
    resource = _reflect(tmp_path, f"{qualifiers} const int{indirection}")
    layout = resource["scalarLayout"]
    pointer = indirection == "*"
    assert layout["runtimeSized"] is pointer
    assert layout["storageLayout"] == (
        "metal-constant" if qualifiers == "constant" and not pointer else "metal-buffer"
    )
    assert layout.get("blockSizeBytes") == (None if pointer else 4)
    assert layout.get("minimumBindingSizeBytes") == (
        4 if qualifiers == "device" and not pointer else None
    )


@pytest.mark.parametrize(
    "declaration",
    [
        "device float3&",
        "device long2&",
        "device bool4&",
        "device Unknown&",
        "device const device int&",
        "device constant int&",
        "thread int&",
        "device int**",
    ],
)
def test_unproven_reference_layout_remains_unavailable(tmp_path, declaration):
    assert "scalarLayout" not in _reflect(tmp_path, declaration)


@pytest.mark.parametrize(
    "mutation",
    [
        {"blockSizeBytes": None},
        {"blockSizeBytes": True},
        {"blockSizeBytes": 8},
        {"minimumBindingSizeBytes": None},
        {"minimumBindingSizeBytes": True},
        {"minimumBindingSizeBytes": 1},
        {"alignmentBytes": 2},
        {"memberOffsetBytes": 4},
        {"elementStrideBytes": 8},
        {"runtimeSized": 0},
        {"physicalType": "uint"},
        {"storageLayout": "metal-constant"},
    ],
)
def test_rejects_inconsistent_device_reference_layout(tmp_path, mutation):
    layout = _reflect(tmp_path, "device const int&")["scalarLayout"]
    layout.update(mutation)
    with pytest.raises(NativeLoaderDispatchError):
        _validated_scalar_layout(
            layout,
            runtime_value=RuntimeValue(name="value", dtype="int32", shape=(1,)),
            target="metal",
            resource_kind="buffer",
            path="$.layout",
        )


def _package(tmp_path, target, width=1):
    scalar = "int" if width == 1 else f"int{width}"
    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void references(device const {scalar}& samples [[buffer(0)]],
                       constant {scalar}& bias [[buffer(1)]],
                       device {scalar}* result [[buffer(2)]]) {{
    result[0] = samples + bias;
    result[1] = samples - bias;
}}
"""
    (tmp_path / "reference.metal").write_text(source, encoding="utf-8")
    (tmp_path / "crosstl.toml").write_text(
        f"""[project]
include = ["reference.metal"]
targets = ["{target}"]
workgroup_size = [1, 1, 1]
[project.entry_points]
"reference.metal" = "references"
""",
        encoding="utf-8",
    )
    report = translate_project(load_project_config(tmp_path), format_output=False)
    assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()
    descriptor, package = _prepare_native_package(report, tmp_path)
    values = {
        "samples": list(range(-3, -3 + width)),
        "bias": list(range(4, 4 + width)),
        "result": [0] * (2 * width),
    }
    guard = [1037] * 8
    expected = {
        "result": (
            [a + b for a, b in zip(values["samples"], values["bias"])]
            + [a - b for a, b in zip(values["samples"], values["bias"])]
            + guard
        ),
    }
    inputs, outputs, names = {}, {}, {}
    for binding in descriptor["bindings"]:
        layout = binding["scalarLayout"] or {}
        member = layout.get("memberName", binding["name"])
        member = {"references_bias": "bias"}.get(member, member)
        names[member] = binding["name"]
        data = values[member] + (guard if member in expected else [])
        allocation = (
            RuntimeAllocationView(member, 256, len(data) * 4, len(data) * 4 + 512)
            if binding["kind"] == "buffer"
            else None
        )
        inputs[binding["name"]] = RuntimeValue(
            name=binding["name"],
            dtype="int32",
            shape=(len(data),),
            values=data,
            allocation=allocation,
        )
        if member in expected:
            outputs[binding["name"]] = replace(
                inputs[binding["name"]], values=expected[member]
            )
    return descriptor, package, inputs, outputs, names


def _request(descriptor, package, inputs, outputs, target):
    return build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )


def _reserved_reference_source(qualifier, name, collision):
    additional = f", {qualifier} const int& {name}_ [[buffer(1)]]" if collision else ""
    operand = f"{name}_" if collision else "17"
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void references({qualifier} const int& {name} [[buffer(0)]]{additional},
                       device int* result [[buffer(2)]]) {{
    result[1] = {name} + {operand};
    result[2] = {name} - {operand};
}}
"""


@pytest.mark.parametrize("qualifier", ["device", "constant"])
@pytest.mark.parametrize("name", ["sample", "patch", "subroutine", "common"])
@pytest.mark.parametrize("collision", [False, True])
def test_glsl_reference_keywords_compile(tmp_path, qualifier, name, collision):
    source = tmp_path / "reference.metal"
    source.write_text(
        _reserved_reference_source(qualifier, name, collision), encoding="utf-8"
    )
    generated = translate(str(source), backend="opengl", format_output=False)
    assert generated == translate(str(source), backend="opengl", format_output=False)
    if qualifier == "device":
        escaped = f"{name}_2" if collision else f"{name}_"
        assert f"int {escaped}[];" in generated
        assert f"result[1] = ({escaped}[0] + " in generated
        assert f"result[2] = ({escaped}[0] - " in generated
        if collision:
            assert f"int {name}_[];" in generated
    _compile(generated, "opengl", tmp_path)
    intermediate = tmp_path / "reference.cgl"
    intermediate.write_text(
        translate(str(source), backend="crossgl", format_output=False),
        encoding="utf-8",
    )
    assert (
        translate(str(intermediate), backend="opengl", format_output=False) == generated
    )


def _reserved_reference_package(tmp_path, target, qualifier, collision):
    (tmp_path / "reference.metal").write_text(
        _reserved_reference_source(qualifier, "sample", collision), encoding="utf-8"
    )
    (tmp_path / "crosstl.toml").write_text(
        f"""[project]
include = ["reference.metal"]
targets = ["{target}"]
workgroup_size = [1, 1, 1]
[project.entry_points]
"reference.metal" = "references"
""",
        encoding="utf-8",
    )
    report = translate_project(load_project_config(tmp_path), format_output=False)
    assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()
    descriptor, package = _prepare_native_package(report, tmp_path)
    bindings = {item["coordinates"]["binding"]: item for item in descriptor["bindings"]}
    assert set(bindings) == ({0, 1, 2} if collision else {0, 2})
    assert len({item["name"] for item in bindings.values()}) == len(bindings)
    guard = 1037
    inputs, outputs = {}, {}
    for index, binding in bindings.items():
        assert binding["coordinates"]["set"] == 0
        assert binding["access"] == ("read_write" if index == 2 else "read")
        layout = binding["scalarLayout"]
        assert layout["elementType"] == "int32"
        assert layout["elementSizeBytes"] == 4
        values = [guard] * 11 if index == 2 else ([-7] if index == 0 else [17])
        inputs[binding["name"]] = RuntimeValue(
            name=binding["name"], dtype="int32", shape=(len(values),), values=values
        )
        if index == 2:
            outputs[binding["name"]] = replace(
                inputs[binding["name"]], values=[guard, 10, -24] + [guard] * 8
            )
    return descriptor, package, inputs, outputs


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("qualifier", ["device", "constant"])
@pytest.mark.parametrize("collision", [False, True])
def test_reserved_reference_package_bindings(tmp_path, target, qualifier, collision):
    descriptor, package, inputs, outputs = _reserved_reference_package(
        tmp_path, target, qualifier, collision
    )
    request = _request(descriptor, package, inputs, outputs, target)
    assert request.execution_plan.diagnostics == ()


def test_reserved_reference_native_readback(tmp_path):
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}.get(
        sys.platform
    )
    if os.environ.get("CROSTL_REQUIRE_DEVICE_REFERENCES") != "1" or target is None:
        pytest.skip("requires native device-reference validation")
    executor = _executor(target)
    try:
        for qualifier in ("device", "constant"):
            for collision in (False, True):
                work = tmp_path / f"{qualifier}-{collision}"
                work.mkdir()
                descriptor, package, inputs, outputs = _reserved_reference_package(
                    work, target, qualifier, collision
                )
                artifact = package / descriptor["artifact"]["packagePath"]
                _validate(artifact, work, target)
                request = _request(descriptor, package, inputs, outputs, target)
                _record_reference_request(work, request)
                availability = executor.is_available(request)
                assert availability.available, availability.reason
                result = executor.run(request)
                (work / "readback.json").write_text(
                    json.dumps(
                        {
                            "status": result.status,
                            "outputs": result.outputs,
                            "details": result.details,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                assert result.status == "ok", result.details
                for name, expected in outputs.items():
                    assert result.outputs[name]["values"] == expected.values
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()


def _writable_source(width):
    scalar = "int" if width == 1 else f"int{width}"
    return f"""#include <metal_stdlib>
using namespace metal;
void increment(device {scalar}& value) {{ value += {scalar}(3); }}
kernel void mutate(device {scalar}& value [[buffer(0)]],
                   device {scalar}* result [[buffer(1)]]) {{
    value = value * {scalar}(2);
    increment(value);
    result[0] = value;
    value -= {scalar}(1);
    result[1] = value;
}}
"""


def _writable_package(tmp_path, target, width, *, shared=False):
    (tmp_path / "reference.metal").write_text(_writable_source(width), encoding="utf-8")
    (tmp_path / "crosstl.toml").write_text(
        f"""[project]
include = ["reference.metal"]
targets = ["{target}"]
workgroup_size = [1, 1, 1]
[project.entry_points]
"reference.metal" = "mutate"
""",
        encoding="utf-8",
    )
    report = translate_project(load_project_config(tmp_path), format_output=False)
    assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()
    descriptor, package = _prepare_native_package(report, tmp_path)
    values = list(range(-7, -7 + width))
    guard = [1037] * 8
    before = {"value": values + guard, "result": [0] * (2 * width) + guard}
    after = {
        "value": [2 * item + 2 for item in values] + guard,
        "result": (
            [2 * item + 3 for item in values]
            + [2 * item + 2 for item in values]
            + guard
        ),
    }
    inputs, outputs, names = {}, {}, {}
    for binding in descriptor["bindings"]:
        layout = binding["scalarLayout"] or {}
        member = layout.get("memberName", binding["name"])
        assert member in before
        assert binding["kind"] == "buffer" and binding["access"] == "read_write"
        names[member] = binding["name"]
        value = RuntimeValue(
            name=binding["name"],
            dtype="int32",
            shape=(len(before[member]),),
            values=before[member],
            allocation=RuntimeAllocationView(
                "shared" if shared else member,
                256 if member == "value" else 512,
                len(before[member]) * 4,
                1024,
            ),
        )
        inputs[value.name] = value
        outputs[value.name] = replace(value, values=after[member])
    assert set(names) == {"value", "result"}
    return descriptor, package, inputs, outputs, names


@pytest.mark.parametrize("target,width", REFERENCE_TARGETS)
def test_writable_entry_reference_single_file_retains_resources(
    tmp_path, target, width
):
    source = tmp_path / "source.metal"
    source.write_text(_writable_source(width), encoding="utf-8")
    generated = translate(str(source), backend=target, format_output=False)
    suffix = {"directx": "hlsl", "opengl": "glsl", "metal": "metal"}[target]
    artifact = tmp_path / f"translated.{suffix}"
    artifact.write_text(generated, encoding="utf-8")
    interface = reflect_target_host_interface(artifact, target=target, stage="compute")
    assert interface["status"] == "ready"
    assert len(interface["resources"]) == 2
    assert all(
        resource["access"] == "read_write" for resource in interface["resources"]
    )
    assert "value_offset" not in generated
    if target == "directx":
        assert "value += " in generated


@pytest.mark.parametrize("target,width", REFERENCE_TARGETS)
@pytest.mark.parametrize("shared", [False, True])
def test_writable_entry_reference_survives_public_package(
    tmp_path, target, width, shared
):
    descriptor, package, inputs, outputs, names = _writable_package(
        tmp_path, target, width, shared=shared
    )
    request = _request(descriptor, package, inputs, outputs, target)
    assert request.execution_plan.diagnostics == ()
    allocations = {
        resource.binding.name: resource.allocation
        for resource in request.execution_plan.resource_bindings
    }
    assert allocations[names["value"]].byte_offset == 256
    assert allocations[names["result"]].byte_offset == 512
    assert (
        allocations[names["value"]].allocation_id
        == allocations[names["result"]].allocation_id
    ) is shared


@pytest.mark.parametrize("target,width", REFERENCE_TARGETS)
def test_reference_layout_survives_public_package(tmp_path, target, width):
    descriptor, package, inputs, outputs, names = _package(tmp_path, target, width)
    assert set(names) == {"samples", "bias", "result"}
    request = _request(descriptor, package, inputs, outputs, target)
    assert request.execution_plan.diagnostics == ()
    bindings = {binding["name"]: binding for binding in descriptor["bindings"]}
    assert bindings[names["samples"]]["access"] == "read"
    assert bindings[names["bias"]]["access"] == "read"
    assert bindings[names["result"]]["access"] == "read_write"
    for name in ("samples", "bias", "result"):
        layout = bindings[names[name]]["scalarLayout"]
        assert layout["elementType"] == "int32"
        assert layout.get("vectorWidth", 1) == width
        assert layout["elementSizeBytes"] == width * 4
    if target == "directx":
        layout = bindings[names["bias"]]["scalarLayout"]
        assert layout["storageLayout"] == "hlsl-constant-buffer"
        assert layout["blockSizeBytes"] == 16
    if target == "metal":
        for binding in descriptor["bindings"]:
            if binding["name"] == names["samples"]:
                assert binding["scalarLayout"]["minimumBindingSizeBytes"] == width * 4
                assert binding["scalarLayout"]["runtimeSized"] is False
    for resource in request.execution_plan.resource_bindings:
        if resource.binding.kind == "buffer":
            assert resource.allocation.byte_offset == 256


@pytest.mark.parametrize("qualifiers", ["device const", "const device"])
@pytest.mark.parametrize("base_type", ["int", "uint", "float"])
@pytest.mark.parametrize("width", [2, 4])
def test_directx_vector_reference_component_access(
    tmp_path, qualifiers, base_type, width
):
    vector = f"{base_type}{width}"
    source = tmp_path / "reference.metal"
    source.write_text(
        f"""#include <metal_stdlib>
using namespace metal;
void adjust(device {vector}& value) {{
    value.x += {base_type}(3);
    value[1] -= {base_type}(2);
}}
kernel void references({qualifiers} {vector}& samples [[buffer(0)]],
                       constant {vector}& bias [[buffer(1)]],
                       device {vector}& value [[buffer(2)]]) {{
    value = samples + bias;
    adjust(value);
    value[1] += samples.x + bias[1];
}}
""",
        encoding="utf-8",
    )
    generated = translate(str(source), backend="directx", format_output=False)
    assert f"StructuredBuffer<{vector}> samples : register(t0);" in generated
    assert f"RWStructuredBuffer<{vector}> value : register(u2);" in generated
    assert f"{vector} references_bias;" in generated
    assert "void CSMain()" in generated
    assert f"void adjust(inout {vector} value)" in generated
    assert "adjust(value[0]);" in generated
    ir_vector = {"int": "ivec", "uint": "uvec", "float": "vec"}[base_type]
    helper = f"CrossGLMetalVectorIndex_{ir_vector}{width}"
    assert (
        f"{helper}_add_assign(value[0], 1, "
        f"(samples[0].x + {helper}_get(references_bias, 1)));"
    ) in generated
    assert "_offset" not in generated


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("dtype", "resource-layout-mismatch"),
        ("view", "execution-plan-invalid"),
        ("alignment", "execution-plan-invalid"),
        ("read-only", "value-duplicate"),
    ],
)
@pytest.mark.parametrize("target,width", REFERENCE_TARGETS)
def test_reference_dispatch_rejects_incompatible_values(
    tmp_path, mutation, code, target, width
):
    descriptor, package, inputs, outputs, names = _package(tmp_path, target, width)
    name = names["samples"]
    value = inputs[name]
    if mutation == "dtype":
        inputs[name] = replace(value, dtype="uint32", values=[3] * width)
    elif mutation == "view":
        inputs[name] = replace(
            value, allocation=replace(value.allocation, byte_length=1)
        )
    elif mutation == "alignment":
        inputs[name] = replace(
            value, allocation=replace(value.allocation, byte_offset=1)
        )
    else:
        outputs[name] = value
    with pytest.raises(NativeLoaderDispatchError) as caught:
        _request(descriptor, package, inputs, outputs, target)
    assert caught.value.code.endswith(code)
    if mutation in {"view", "alignment"}:
        assert any(
            diagnostic["code"].endswith(
                "resource-view-too-small"
                if mutation == "view"
                else "resource-allocation-view-misaligned"
            )
            for diagnostic in caught.value.details["diagnostics"]
        )


def test_device_reference_native_readback(tmp_path):
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}.get(
        sys.platform
    )
    if os.environ.get("CROSTL_REQUIRE_DEVICE_REFERENCES") != "1" or target is None:
        pytest.skip("requires native device-reference validation")
    for width in [1, 2, 4]:
        work = tmp_path / str(width)
        work.mkdir()
        descriptor, package, inputs, outputs, names = _package(work, target, width)
        assert set(names) == {"samples", "bias", "result"}
        artifact = package / descriptor["artifact"]["packagePath"]
        _validate(artifact, work, target)
        request = _request(descriptor, package, inputs, outputs, target)
        _record_reference_request(work, request)
        executor = _executor(target)
        availability = executor.is_available(request)
        assert availability.available, availability.reason
        result = executor.run(request)
        (work / "readback.json").write_text(
            json.dumps(
                {
                    "status": result.status,
                    "outputs": result.outputs,
                    "details": result.details,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        assert result.status == "ok"
        for name, expected in outputs.items():
            assert result.outputs[name]["values"] == expected.values


def _record_reference_request(work, request):
    (work / "request.json").write_text(
        json.dumps(
            {
                "fixture": request.fixture.to_json(),
                "executionPlan": request.execution_plan.to_json(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_writable_entry_reference_native_readback(tmp_path):
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}.get(
        sys.platform
    )
    if os.environ.get("CROSTL_REQUIRE_DEVICE_REFERENCES") != "1" or target is None:
        pytest.skip("requires native device-reference validation")
    for case_target, width in REFERENCE_TARGETS:
        if case_target != target:
            continue
        for shared in (False, True):
            work = tmp_path / f"{width}-{shared}"
            work.mkdir()
            descriptor, package, inputs, outputs, _ = _writable_package(
                work, target, width, shared=shared
            )
            artifact = package / descriptor["artifact"]["packagePath"]
            _validate(artifact, work, target)
            request = _request(descriptor, package, inputs, outputs, target)
            _record_reference_request(work, request)
            executor = _executor(target)
            availability = executor.is_available(request)
            assert availability.available, availability.reason
            result = executor.run(request)
            (work / "readback.json").write_text(
                json.dumps(
                    {
                        "status": result.status,
                        "outputs": result.outputs,
                        "details": result.details,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            assert result.status == "ok"
            for name, expected in outputs.items():
                assert result.outputs[name]["values"] == expected.values
            if target == "metal":
                original = work / "original"
                original.mkdir()
                source_descriptor = _original_metal_descriptor(
                    original, _writable_source(width), "mutate"
                )
                _validate(
                    original / source_descriptor["artifact"]["packagePath"],
                    original,
                    "metal",
                )
                source_request = _request(
                    source_descriptor, original, inputs, outputs, "metal"
                )
                _record_reference_request(original, source_request)
                control = executor.run(source_request)
                (original / "readback.json").write_text(
                    json.dumps(
                        {
                            "status": control.status,
                            "outputs": control.outputs,
                            "details": control.details,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                assert control.status == "ok"
                assert control.outputs == result.outputs


def _original_metal_descriptor(tmp_path, source, entry):
    descriptor = _write_descriptor(
        tmp_path,
        "metal",
        constants=[],
        artifact_format="Metal source",
        artifact_bytes=source.encode(),
    )
    artifact = tmp_path / descriptor["artifact"]["packagePath"]
    artifact = artifact.rename(artifact.with_suffix(".metal"))
    descriptor["artifact"]["packagePath"] = artifact.relative_to(tmp_path).as_posix()
    interface = reflect_target_host_interface(artifact, target="metal", stage="compute")
    descriptor["entryPoint"].update(
        name=entry, executionConfig={"workgroupSize": [1, 1, 1]}
    )
    descriptor["bindings"] = _binding_descriptors("metal", interface["resources"])
    descriptor["scalarLayout"]["bindings"] = [
        {"binding": binding["name"], "layout": binding["scalarLayout"]}
        for binding in descriptor["bindings"]
    ]
    return descriptor


@pytest.mark.parametrize("width", [1, 2, 4])
def test_writable_metal_reference_native_readback(tmp_path, width):
    if (
        os.environ.get("CROSTL_REQUIRE_DEVICE_REFERENCES") != "1"
        or sys.platform != "darwin"
    ):
        pytest.skip("requires native Metal device-reference validation")
    scalar = "int" if width == 1 else f"int{width}"
    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void mutate(device {scalar}& value [[buffer(0)]]) {{ value += {scalar}(3); }}
"""
    # This control tests the reflected source ABI independently of entry lowering.
    descriptor = _original_metal_descriptor(tmp_path, source, "mutate")
    artifact = tmp_path / descriptor["artifact"]["packagePath"]
    values = list(range(-3, -3 + width)) + [1037] * 8
    value = RuntimeValue(
        name="value",
        dtype="int32",
        shape=(len(values),),
        values=values,
        allocation=RuntimeAllocationView(
            "value", 256, len(values) * 4, len(values) * 4 + 512
        ),
    )
    expected = replace(
        value, values=[item + 3 for item in values[:width]] + values[width:]
    )
    _validate(artifact, tmp_path, "metal")
    request = _request(
        descriptor, tmp_path, {"value": value}, {"value": expected}, "metal"
    )
    executor = _executor("metal")
    availability = executor.is_available(request)
    assert availability.available, availability.reason
    result = executor.run(request)
    (tmp_path / "readback.json").write_text(
        json.dumps(result.outputs, indent=2), encoding="utf-8"
    )
    assert result.status == "ok"
    assert result.outputs["value"]["values"] == expected.values


def test_ci_requires_reference_buffers_without_an_additional_runner():
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow.split("      - name: Validate scalar alias resolution\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert 'CROSTL_REQUIRE_DEVICE_REFERENCES: "1"' in step
    assert "tests/test_translator/test_metal_buffer_references.py" in step
    assert "--timeout-seconds 120" in step
    assert "pytest -q -n auto" in step
    assert "if:" not in step and "continue-on-error" not in step
