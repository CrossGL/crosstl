from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from crosstl.project import (
    NativeLoaderDispatchError,
    build_native_loader_dispatch_request,
)
from crosstl.project.native_loader_abi import (
    NATIVE_LOADER_ABI_KIND,
    NATIVE_LOADER_ABI_VERSION,
)
from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    OpenGLComputeRuntime,
)
from crosstl.project.runtime_verification import (
    DirectXRuntimeParityAdapter,
    OpenGLRuntimeParityAdapter,
    RuntimeExecutorAvailability,
    RuntimeParityExecutor,
    RuntimeTestAdapterSpec,
)

_INPUT_VALUES = [0, 1, 4, 9]
_EXPECTED_VALUES = [7, 10, 19, 34]
_SOURCES = {
    "directx": (
        b"""\
StructuredBuffer<uint> input_values : register(t0);
RWStructuredBuffer<uint> output_values : register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint3 thread_id : SV_DispatchThreadID) {
    output_values[thread_id.x] = input_values[thread_id.x] * 3u + 7u;
}
"""
    ),
    "opengl": (
        b"""\
#version 430
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer InputValues {
    uint input_values[];
};
layout(std430, binding = 1) writeonly buffer OutputValues {
    uint output_values[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    output_values[index] = input_values[index] * 3u + 7u;
}
"""
    ),
}


def _write_package(tmp_path: Path, target: str) -> tuple[dict, Path]:
    source = _SOURCES[target]
    extension = "hlsl" if target == "directx" else "comp"
    entry_point = "CSMain" if target == "directx" else "main"
    artifact_format = "HLSL source" if target == "directx" else "GLSL source"
    input_namespace = "srv" if target == "directx" else "storage-buffer"
    output_namespace = "uav" if target == "directx" else "storage-buffer"
    output_binding = 0 if target == "directx" else 1
    package_path = f"artifacts/{target}/affine_uint32.{extension}"
    artifact_path = tmp_path / package_path
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(source)

    scalar_layout = {
        "physicalType": "uint",
        "elementType": "uint32",
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
        "alignmentBytes": 4,
        "memberOffsetBytes": 0,
        "storageLayout": "hlsl-structured-buffer" if target == "directx" else "std430",
        "runtimeSized": True,
    }
    bindings = [
        {
            "name": "input_values",
            "kind": "buffer",
            "type": "uint32[]",
            "namespace": input_namespace,
            "coordinates": {"set": 0, "binding": 0},
            "access": "read",
            "scalarLayout": dict(scalar_layout),
            "provenance": {"parameter": 0},
        },
        {
            "name": "output_values",
            "kind": "buffer",
            "type": "uint32[]",
            "namespace": output_namespace,
            "coordinates": {"set": 0, "binding": output_binding},
            "access": "write",
            "scalarLayout": dict(scalar_layout),
            "provenance": {"parameter": 1},
        },
    ]
    descriptor = {
        "schemaVersion": NATIVE_LOADER_ABI_VERSION,
        "kind": NATIVE_LOADER_ABI_KIND,
        "abiVersion": NATIVE_LOADER_ABI_VERSION,
        "unitId": f"affine-uint32:{target}",
        "target": target,
        "stage": "compute",
        "entryPoint": {
            "name": entry_point,
            "stage": "compute",
            "executionConfig": {"workgroupSize": [1, 1, 1]},
            "provenance": {"sourceName": "affine_uint32"},
        },
        "artifact": {
            "packagePath": package_path,
            "format": artifact_format,
            "hash": {
                "algorithm": "sha256",
                "value": hashlib.sha256(source).hexdigest(),
            },
            "sizeBytes": len(source),
        },
        "source": {
            "path": "kernels/affine_uint32.metal",
            "artifactPath": f"out/{target}/affine_uint32.{extension}",
            "backend": "metal",
            "hash": None,
            "remap": None,
        },
        "bindings": bindings,
        "scalarLayout": {
            "constants": [],
            "bindings": [
                {
                    "binding": binding["name"],
                    "layout": dict(scalar_layout),
                }
                for binding in bindings
            ],
        },
        "specializationConstants": [],
        "provenance": {
            "pipeline": "metal-to-crossgl",
            "target": target,
        },
    }
    return descriptor, artifact_path


def _build_request(tmp_path: Path, target: str):
    descriptor, artifact_path = _write_package(tmp_path, target)
    request = build_native_loader_dispatch_request(
        descriptor,
        tmp_path,
        {
            "input_values": {
                "dtype": "uint32",
                "shape": [len(_INPUT_VALUES)],
                "values": _INPUT_VALUES,
            }
        },
        {
            "output_values": {
                "dtype": "uint32",
                "shape": [len(_EXPECTED_VALUES)],
                "values": _EXPECTED_VALUES,
            }
        },
        (len(_INPUT_VALUES), 1, 1),
        expected_target=target,
    )
    return request, descriptor, artifact_path


def _executor(target: str) -> RuntimeParityExecutor:
    if target == "directx":
        runtime_adapter = DirectXRuntimeParityAdapter(runtime=DirectXComputeRuntime())
    else:
        runtime_adapter = OpenGLRuntimeParityAdapter(
            runtime=OpenGLComputeRuntime(context_backends=("egl",))
        )
    return RuntimeParityExecutor(
        RuntimeTestAdapterSpec(
            adapter_id=f"{target}-native-loader",
            target=target,
            executor=target,
            adapter_kind=f"{target}-native-runtime",
        ),
        runtime_adapter=runtime_adapter,
    )


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_complete_package_descriptor_builds_preflighted_request(tmp_path, target):
    request, descriptor, artifact_path = _build_request(tmp_path, target)

    assert request.artifact_path == artifact_path.resolve()
    assert request.artifact["hash"] == descriptor["artifact"]["hash"]
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    assert request.execution_plan.dispatch.global_size == (4, 1, 1)
    assert [
        binding.metadata["scalarLayout"]
        for binding in request.adapter_contract.resource_bindings
    ] == [
        {
            "physicalType": "uint",
            "elementType": "uint32",
            "elementSizeBytes": 4,
            "elementStrideBytes": 4,
            "alignmentBytes": 4,
            "memberOffsetBytes": 0,
            "storageLayout": (
                "hlsl-structured-buffer" if target == "directx" else "std430"
            ),
            "runtimeSized": True,
        },
        {
            "physicalType": "uint",
            "elementType": "uint32",
            "elementSizeBytes": 4,
            "elementStrideBytes": 4,
            "alignmentBytes": 4,
            "memberOffsetBytes": 0,
            "storageLayout": (
                "hlsl-structured-buffer" if target == "directx" else "std430"
            ),
            "runtimeSized": True,
        },
    ]


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_tampered_package_artifact_fails_before_runtime_consultation(tmp_path, target):
    descriptor, artifact_path = _write_package(tmp_path, target)
    artifact_path.write_bytes(artifact_path.read_bytes().replace(b"* 3u", b"* 4u"))

    class RuntimeGuard:
        def __init__(self):
            self.calls = []

        def is_available(self, request):
            self.calls.append(("is_available", request))
            return RuntimeExecutorAvailability(True)

        def run(self, request):
            self.calls.append(("run", request))
            raise AssertionError("tampered artifacts must not reach runtime dispatch")

    runtime = RuntimeGuard()
    with pytest.raises(NativeLoaderDispatchError) as caught:
        request = build_native_loader_dispatch_request(
            descriptor,
            tmp_path,
            {
                "input_values": {
                    "dtype": "uint32",
                    "shape": [len(_INPUT_VALUES)],
                    "values": _INPUT_VALUES,
                }
            },
            {
                "output_values": {
                    "dtype": "uint32",
                    "shape": [len(_EXPECTED_VALUES)],
                    "values": _EXPECTED_VALUES,
                }
            },
            (len(_INPUT_VALUES), 1, 1),
            expected_target=target,
        )
        if runtime.is_available(request).available:
            runtime.run(request)

    assert caught.value.code == "project.native-loader-dispatch.artifact-hash-mismatch"
    assert runtime.calls == []


def test_native_loader_descriptor_executes_directx_on_device(tmp_path):
    if os.environ.get("CROSTL_RUN_NATIVE_LOADER_DIRECTX_DEVICE_TEST") != "1":
        pytest.skip("Direct3D native loader device test is not enabled")

    request, _descriptor, _artifact_path = _build_request(tmp_path, "directx")
    executor = _executor("directx")
    availability = executor.is_available(request)
    assert availability.available, availability.reason or availability.details

    result = executor.run(request)

    assert result.status == "ok"
    assert result.outputs == {
        "output_values": {
            "dtype": "uint32",
            "shape": [4],
            "values": _EXPECTED_VALUES,
        }
    }


def test_native_loader_descriptor_executes_opengl_on_device(tmp_path):
    if os.environ.get("CROSTL_RUN_NATIVE_LOADER_OPENGL_DEVICE_TEST") != "1":
        pytest.skip("OpenGL native loader device test is not enabled")

    request, _descriptor, _artifact_path = _build_request(tmp_path, "opengl")
    executor = _executor("opengl")
    availability = executor.is_available(request)
    assert availability.available, availability.reason or availability.details

    result = executor.run(request)

    assert result.status == "ok"
    assert result.outputs == {
        "output_values": {
            "dtype": "uint32",
            "shape": [4],
            "values": _EXPECTED_VALUES,
        }
    }
