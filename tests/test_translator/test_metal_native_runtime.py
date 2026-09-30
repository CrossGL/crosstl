"""Metal resource contracts and production package-to-device execution."""

import base64
import json
import os
import struct
import subprocess
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from crosstl.project import (
    MetalComputeRuntime,
    MetalRuntimeParityAdapter,
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
from crosstl.project.metal_runtime import run_metal_command
from crosstl.project.runtime_verification import (
    RuntimeAdapterDispatchError,
    RuntimeAdapterSetupError,
    RuntimeAllocationView,
    RuntimeExecutionError,
    RuntimeExecutionState,
    RuntimeValue,
)
from tests.test_translator.test_native_loader_dispatch_integration import _executor


def _reflection(tmp_path, declaration):
    source = tmp_path / "sample.metal"
    source.write_text(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "struct Pair { float real; float imag; };\n"
        f"kernel void sample({declaration} [[buffer(3)]]) {{}}",
        encoding="utf-8",
    )
    return reflect_target_host_interface(source, target="metal")["resources"][0]


@pytest.mark.parametrize(
    "scalar,dtype,size",
    [
        ("float", "float32", 4),
        ("int", "int32", 4),
        ("uint", "uint32", 4),
        ("int64_t", "int64", 8),
        ("uint64_t", "uint64", 8),
        ("long", "int64", 8),
        ("ulong", "uint64", 8),
    ],
)
@pytest.mark.parametrize(
    "qualifier,suffix,runtime_sized",
    [
        ("const device", "*", True),
        ("device", "*", True),
        ("constant", "*", True),
        ("constant", "&", False),
    ],
)
def test_metal_scalar_layouts(
    tmp_path, scalar, dtype, size, qualifier, suffix, runtime_sized
):
    resource = _reflection(tmp_path, f"{qualifier} {scalar}{suffix} values")
    layout = resource["scalarLayout"]
    assert resource["binding"] == 3
    assert resource["kind"] == ("buffer" if runtime_sized else "constant-buffer")
    assert resource["access"] == ("read_write" if qualifier == "device" else "read")
    assert layout["elementType"] == dtype
    assert layout["elementSizeBytes"] == layout["elementStrideBytes"] == size
    assert layout["alignmentBytes"] == size
    assert layout["runtimeSized"] is runtime_sized


@pytest.mark.parametrize("scalar", ["float", "int", "uint"])
@pytest.mark.parametrize("width", [2, 4])
def test_metal_vector_alignment(tmp_path, scalar, width):
    layout = _reflection(tmp_path, f"device {scalar}{width}* values")["scalarLayout"]
    assert layout["elementSizeBytes"] == layout["alignmentBytes"] == 4 * width
    assert layout["vectorWidth"] == width


def test_metal_homogeneous_struct_is_not_a_vector(tmp_path):
    layout = _reflection(tmp_path, "device Pair* values")["scalarLayout"]
    assert layout["alignmentBytes"] == 4
    assert layout["elementStrideBytes"] == 8
    assert layout["componentCount"] == 2
    assert "vectorWidth" not in layout


@pytest.mark.parametrize(
    "declaration",
    [
        "device float3* values",
        "device packed_float3* values",
        "device half* values",
        "device bool* values",
        "device long2* values",
        "device float& values",
        "constant Pair& values",
        "device float** values",
        "volatile device int* values",
        "device atomic_uint* values",
        "device float values[4]",
    ],
)
def test_metal_unproven_layouts_remain_unavailable(tmp_path, declaration):
    assert "scalarLayout" not in _reflection(tmp_path, declaration)


SOURCE = """#include <metal_stdlib>
using namespace metal;
struct Pair { float real; float imag; };
kernel void transform(const device Pair* input [[buffer(0)]],
                      device Pair* result [[buffer(3)]],
                      constant uint& offset [[buffer(5)]],
                      uint3 index [[thread_position_in_grid]],
                      uint3 size [[threads_per_grid]]) {
    uint i = index.x + size.x * (index.y + size.y * index.z);
    result[i].real = input[i].real * 2.0f + float(offset);
    result[i].imag = input[i].imag - 3.0f;
}
"""


def _package(tmp_path, source=SOURCE):
    (tmp_path / "sample.metal").write_text(source, encoding="utf-8")
    config = tmp_path / "crosstl.toml"
    config.write_text(
        """[project]
source_roots = ["."]
include = ["sample.metal"]
targets = ["metal"]
output_dir = "out"
[project.entry_points]
"sample.metal" = "transform"
""",
        encoding="utf-8",
    )
    report = translate_project(
        load_project_config(tmp_path, config), format_output=False
    )
    report.write_json(tmp_path / "report.json")
    assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()[
        "diagnostics"
    ]
    manifest = build_runtime_artifact_manifest(tmp_path / "report.json")
    assert manifest["success"], manifest
    (tmp_path / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
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
    return descriptor, package


def _request(tmp_path):
    descriptor, package = _package(tmp_path)
    inputs = {
        "input": {"dtype": "float32", "shape": [8, 2], "values": list(range(16))},
        "result": {"dtype": "float32", "shape": [8, 2], "values": [-1234] * 16},
        "offset": {"dtype": "uint32", "shape": [1], "values": [7]},
    }
    outputs = {
        "result": {
            "dtype": "float32",
            "shape": [8, 2],
            "values": [value for i in range(8) for value in (i * 4 + 7, i * 2 - 2)],
        }
    }
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [2, 2, 1], "workgroupSize": [2, 1, 1]},
        expected_target="metal",
    )
    assert not request.execution_plan.diagnostics
    return request, descriptor, package, inputs, outputs


def _native_request(request):
    adapter = MetalRuntimeParityAdapter(runtime=SimpleNamespace())
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    native = adapter._prepare_dispatch_request(
        state, request.artifact_path, request.artifact_path.with_suffix(".metallib")
    )
    return state, native


def test_metal_package_preserves_layouts_and_single_buffer_namespace(tmp_path):
    request, descriptor, _, _, _ = _request(tmp_path)
    assert {b["namespace"] for b in descriptor["bindings"]} == {"buffer"}
    assert {b["coordinates"]["binding"] for b in descriptor["bindings"]} == {0, 3, 5}
    state, native = _native_request(request)
    payload, outputs = MetalComputeRuntime()._prepare_request(native)
    assert payload["workgroupCount"] == [2, 2, 1]
    assert payload["workgroupSize"] == [2, 1, 1]
    assert outputs == {"result": ("float32", (8, 2), 64)}
    assert len(payload["allocations"]) == 3
    assert state.adapter_steps


def test_metal_loader_requires_workgroup_size(tmp_path):
    _, descriptor, package, inputs, outputs = _request(tmp_path)
    with pytest.raises(NativeLoaderDispatchError, match="workgroup-size-missing"):
        build_native_loader_dispatch_request(
            descriptor, package, inputs, outputs, [1, 1, 1]
        )


@pytest.mark.parametrize("size", [0, -1, True, float("nan"), float("inf")])
def test_metal_deadline_is_explicit_and_finite(size):
    with pytest.raises(ValueError):
        MetalComputeRuntime(timeout_seconds=size)
    with pytest.raises(ValueError):
        MetalRuntimeParityAdapter(timeout_seconds=size)


def test_metal_runtime_rejects_allocation_limit_before_execution(tmp_path):
    request, _, _, _, _ = _request(tmp_path)
    _, native = _native_request(request)
    with pytest.raises(RuntimeAdapterSetupError, match="limit"):
        MetalComputeRuntime(max_buffer_bytes=32)._prepare_request(native)


@pytest.mark.parametrize("conflicting", [False, True])
def test_metal_alias_views_share_storage_and_reject_conflicting_uploads(
    tmp_path, conflicting
):
    request, _, _, _, _ = _request(tmp_path)
    _, native = _native_request(request)
    buffers = dict(native.buffers)
    view = RuntimeAllocationView(
        "shared", byte_offset=16, byte_length=64, allocation_byte_length=80
    )
    buffers["input"] = replace(buffers["input"], allocation=view)
    buffers["result"] = replace(
        buffers["result"],
        allocation=view,
        value=buffers["result"].value if conflicting else buffers["input"].value,
    )
    native = replace(native, buffers=buffers)
    if conflicting:
        with pytest.raises(RuntimeAdapterSetupError, match="conflicting bytes"):
            MetalComputeRuntime()._prepare_request(native)
    else:
        payload, _ = MetalComputeRuntime()._prepare_request(native)
        assert len(payload["allocations"]) == 2
        assert payload["allocations"][0]["length"] == 80
        assert (
            payload["buffers"][0]["allocation"] == payload["buffers"][1]["allocation"]
        )


@pytest.mark.parametrize(
    "offset,length,backing", [(2, 64, 128), (16, 60, 128), (16, 64, 64), (-4, 64, 128)]
)
def test_metal_runtime_rejects_invalid_allocation_views(
    tmp_path, offset, length, backing
):
    request, _, _, _, _ = _request(tmp_path)
    _, native = _native_request(request)
    binding = native.buffers["input"]
    native = replace(
        native,
        buffers={
            **native.buffers,
            "input": replace(
                binding,
                allocation=RuntimeAllocationView("input", offset, length, backing),
            ),
        },
    )
    with pytest.raises(RuntimeAdapterSetupError, match="allocation view"):
        MetalComputeRuntime()._prepare_request(native)


@pytest.mark.parametrize("index", [-1, 31, True, 3])
def test_metal_runtime_rejects_invalid_or_duplicate_indices(tmp_path, index):
    request, _, _, _, _ = _request(tmp_path)
    _, native = _native_request(request)
    binding = native.buffers["input"]
    native = replace(
        native,
        buffers={
            **native.buffers,
            "input": replace(binding, binding=replace(binding.binding, binding=index)),
        },
    )
    with pytest.raises(RuntimeAdapterSetupError, match="indices 0-30"):
        MetalComputeRuntime()._prepare_request(native)


@pytest.mark.parametrize(
    "groups,size,reason",
    [
        ((0, 1, 1), (1, 1, 1), "positive uint32"),
        ((1, 1, 1), (), "positive uint32"),
        (((1 << 32) - 1, 1, 1), (2, 1, 1), "exceeds uint32"),
        ((1, 1, 1), (1, 1, 1), "conflicts"),
    ],
)
def test_metal_runtime_rejects_invalid_grid(tmp_path, groups, size, reason):
    request, _, _, _, _ = _request(tmp_path)
    _, native = _native_request(request)
    native = replace(
        native,
        dispatch=replace(native.dispatch, workgroup_count=groups, workgroup_size=size),
    )
    with pytest.raises(RuntimeAdapterSetupError, match=reason):
        MetalComputeRuntime()._prepare_request(native)


def test_metal_runtime_does_not_guess_platform_availability():
    runtime = MetalComputeRuntime(platform_name="linux")
    availability = runtime.is_available(None, None)
    assert not availability.available
    assert availability.details["reasonKind"] == "platform-unavailable"


@pytest.mark.skipif(os.name != "posix", reason="Metal process groups require POSIX")
def test_metal_operation_deadline_terminates_process():
    started = time.monotonic()
    with pytest.raises(RuntimeAdapterSetupError) as caught:
        run_metal_command(
            [sys.executable, "-c", "import time; time.sleep(10)"], timeout_seconds=0.2
        )
    assert caught.value.details["reasonKind"] == "operation-timeout"
    assert time.monotonic() - started < 5


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "null",
        "[]",
        '{"outputs": {}}',
        '{"outputs": {"result": "bad"}}',
        '{"outputs": {"result": "AAAAAA=="}}',
    ],
)
def test_metal_runtime_rejects_invalid_readback(tmp_path, response):
    request, _, _, _, _ = _request(tmp_path)
    state, native = _native_request(request)
    runtime = MetalComputeRuntime(
        command_runner=lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, response, ""
        )
    )
    runtime._worker_path = tmp_path / "stub-worker"
    runtime.platform_name = "darwin"
    with pytest.raises(RuntimeAdapterDispatchError, match="readback"):
        runtime.dispatch(None, state, native)


CONSTANT_SOURCE = """#include <metal_stdlib>
using namespace metal;
constant bool enabled [[function_constant(2)]];
constant float scale [[function_constant(4)]];
constant int signed_offset [[function_constant(6)]];
constant uint extra [[function_constant(8)]];
kernel void transform(device float* result [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    result[index] = enabled ? float(index) * scale + float(signed_offset) + float(extra) : -1.0f;
}
"""


def _constant_request(tmp_path, enabled=True):
    descriptor, package = _package(tmp_path, CONSTANT_SOURCE)
    return build_native_loader_dispatch_request(
        descriptor,
        package,
        {"result": {"dtype": "float32", "shape": [4], "values": [-1234] * 4}},
        {
            "result": {
                "dtype": "float32",
                "shape": [4],
                "values": [2, 4.5, 7, 9.5] if enabled else [-1] * 4,
            }
        },
        {"workgroupCount": [2, 1, 1], "workgroupSize": [2, 1, 1]},
        specialization_values={2: enabled, 4: 2.5, 6: -7, 8: 9},
        expected_target="metal",
    )


def test_metal_package_function_constant_types_and_ids(tmp_path):
    request = _constant_request(tmp_path)
    _, native = _native_request(request)
    payload, _ = MetalComputeRuntime()._prepare_request(native)
    assert [
        (c["id"], c["dtype"], base64.b64decode(c["data"])) for c in payload["constants"]
    ] == [
        (2, "bool", struct.pack("<?", True)),
        (4, "float32", struct.pack("<f", 2.5)),
        (6, "int32", struct.pack("<i", -7)),
        (8, "uint32", struct.pack("<I", 9)),
    ]


@pytest.mark.parametrize(
    "name,value",
    [
        ("enabled", 1),
        ("scale", float("nan")),
        ("signed_offset", True),
        ("signed_offset", 1 << 40),
        ("extra", -1),
    ],
)
def test_metal_runtime_rejects_invalid_function_constant_values(tmp_path, name, value):
    _, native = _native_request(_constant_request(tmp_path))
    native = replace(
        native,
        constants={
            **native.constants,
            name: replace(native.constants[name], value=value),
        },
    )
    with pytest.raises(RuntimeAdapterSetupError, match="constant"):
        MetalComputeRuntime()._prepare_request(native)


@pytest.mark.parametrize("response", ["[]", "null", "not json"])
def test_metal_probe_rejects_malformed_response(tmp_path, response):
    runtime = MetalComputeRuntime(
        platform_name="darwin",
        command_runner=lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, response, ""
        ),
    )
    runtime._worker_path = tmp_path / "stub"
    assert not runtime.is_available(None, None).available


def _execute_native(request, tmp_path):
    if os.environ.get("CROSTL_REQUIRE_METAL_PACKAGE_RUNTIME") != "1":
        pytest.skip("requires the macOS native package runtime gate")
    assert sys.platform == "darwin"
    executor = _executor("metal")
    try:
        availability = executor.is_available(request)
        assert availability.available, availability
        result = executor.run(request)
        (tmp_path / "result.json").write_text(
            json.dumps(
                {"outputs": result.outputs, "details": result.details}, indent=2
            ),
            encoding="utf-8",
        )
        assert result.status == "ok", result
        return result
    finally:
        executor.runtime_adapter.runtime.close()


def test_metal_package_native_readback(tmp_path):
    request, _, _, _, outputs = _request(tmp_path)
    result = _execute_native(request, tmp_path)
    assert result.outputs["result"]["values"] == outputs["result"]["values"]


@pytest.mark.parametrize("enabled", [False, True])
def test_metal_package_native_function_constants(tmp_path, enabled):
    result = _execute_native(_constant_request(tmp_path, enabled), tmp_path)
    assert result.outputs["result"]["values"] == (
        [2, 4.5, 7, 9.5] if enabled else [-1] * 4
    )


@pytest.mark.parametrize(
    "physical,dtype,width,values",
    [
        ("float", "float32", 1, [-1.25, 0.5]),
        ("int", "int32", 1, [-(1 << 31), (1 << 31) - 1]),
        ("uint", "uint32", 1, [0, (1 << 32) - 1]),
        ("long", "int64", 1, [-(1 << 63), (1 << 63) - 1]),
        ("ulong", "uint64", 1, [0, (1 << 64) - 1]),
        ("float2", "float32", 2, [-1.25, 0.5] * 2),
        ("float4", "float32", 4, [-1.25, 0.5] * 4),
        ("int2", "int32", 2, [-(1 << 31), (1 << 31) - 1] * 2),
        ("int4", "int32", 4, [-(1 << 31), (1 << 31) - 1] * 4),
        ("uint2", "uint32", 2, [0, (1 << 32) - 1] * 2),
        ("uint4", "uint32", 4, [0, (1 << 32) - 1] * 4),
    ],
)
def test_metal_native_physical_value_roundtrip(
    tmp_path, physical, dtype, width, values
):
    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void transform(const device {physical}* input [[buffer(0)]],
                      device {physical}* result [[buffer(1)]],
                      uint index [[thread_position_in_grid]]) {{
    result[index] = input[index];
}}
"""
    descriptor, package = _package(tmp_path, source)
    shape = [2] if width == 1 else [2, width]
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        {
            "input": {"dtype": dtype, "shape": shape, "values": values},
            "result": {"dtype": dtype, "shape": shape, "values": [1234] * len(values)},
        },
        {"result": {"dtype": dtype, "shape": shape, "values": values}},
        {"workgroupCount": [1, 1, 1], "workgroupSize": [2, 1, 1]},
        expected_target="metal",
    )
    result = _execute_native(request, tmp_path)
    assert result.outputs["result"]["dtype"] == dtype
    assert result.outputs["result"]["shape"] == shape
    assert result.outputs["result"]["values"] == values


def test_metal_package_native_alias_offset(tmp_path):
    _, descriptor, package, inputs, outputs = _request(tmp_path)
    view = RuntimeAllocationView("shared", 16, 64, 80)
    for name in ("input", "result"):
        inputs[name] = RuntimeValue(
            name=name,
            dtype="float32",
            shape=(8, 2),
            values=list(range(16)),
            allocation=view,
        )
    outputs["result"] = RuntimeValue(
        name="result",
        dtype="float32",
        shape=(8, 2),
        values=outputs["result"]["values"],
        allocation=view,
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [2, 2, 1], "workgroupSize": [2, 1, 1]},
        expected_target="metal",
    )
    result = _execute_native(request, tmp_path)
    assert result.outputs["result"]["values"] == outputs["result"].values


@pytest.mark.parametrize(
    "defect,message",
    [("missing-buffer", "Required Metal buffer"), ("group-limit", "group dimension")],
)
def test_metal_compiled_interface_rejects_invalid_native_dispatch(
    tmp_path, defect, message
):
    _, descriptor, package, inputs, outputs = _request(tmp_path)
    geometry = {"workgroupCount": [2, 2, 1], "workgroupSize": [2, 1, 1]}
    if defect == "missing-buffer":
        descriptor["bindings"] = [
            b for b in descriptor["bindings"] if b["name"] != "offset"
        ]
        descriptor["scalarLayout"]["bindings"] = [
            b
            for b in descriptor["scalarLayout"]["bindings"]
            if b["binding"] != "offset"
        ]
        del inputs["offset"]
    else:
        geometry["workgroupSize"] = [65536, 1, 1]
    request = build_native_loader_dispatch_request(
        descriptor, package, inputs, outputs, geometry, expected_target="metal"
    )
    if os.environ.get("CROSTL_REQUIRE_METAL_PACKAGE_RUNTIME") != "1":
        pytest.skip("requires the macOS native package runtime gate")
    with pytest.raises(RuntimeExecutionError) as caught:
        _execute_native(request, tmp_path)
    assert message in caught.value.details["stderr"]
    (tmp_path / "rejection.json").write_text(
        json.dumps(caught.value.details, indent=2), encoding="utf-8"
    )
