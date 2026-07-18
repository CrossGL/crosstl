from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    OpenGLComputeRuntime,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeAllocationView,
    RuntimeDispatchGeometry,
    RuntimeResourceBinding,
    RuntimeValue,
)

_INITIAL_VALUES = [0, 1, 7, 42]
_EXPECTED_VALUES = [7, 10, 28, 133]

_DIRECTX_SOURCE = """\
StructuredBuffer<uint> source_values : register(t0);
RWStructuredBuffer<uint> destination_values : register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint3 thread_id : SV_DispatchThreadID) {
    uint index = thread_id.x;
    destination_values[index] = source_values[index] * 3u + 7u;
}
"""

_OPENGL_SOURCE = """\
#version 430
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer SourceValues {
    uint source_values[];
};
layout(std430, binding = 1) writeonly buffer DestinationValues {
    uint destination_values[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    destination_values[index] = source_values[index] * 3u + 7u;
}
"""


class _DirectXAllocationProbe(DirectXComputeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.working_resources = []
        self.bound_views = []

    def _create_buffer_resource(
        self,
        compushady,
        device,
        prepared,
        owned_objects,
    ):
        resource = super()._create_buffer_resource(
            compushady,
            device,
            prepared,
            owned_objects,
        )
        self.working_resources.append(resource.device_buffer)
        return resource

    def _create_buffer_resources(
        self,
        compushady,
        device,
        prepared_buffers,
        owned_objects,
    ):
        resources = super()._create_buffer_resources(
            compushady,
            device,
            prepared_buffers,
            owned_objects,
        )
        self.bound_views = [
            (
                resource.prepared.namespace,
                resource.prepared.binding_index,
                resource.device_buffer,
            )
            for resource in resources
        ]
        return resources


class _OpenGLAllocationProbe(OpenGLComputeRuntime):
    def __init__(self) -> None:
        super().__init__(context_backends=("egl",))
        self.working_resources = []
        self.bound_views = []

    def _create_allocation_buffer(self, context, payload):
        buffer = super()._create_allocation_buffer(context, payload)
        self.working_resources.append(buffer)
        return buffer

    def _bind_buffer_view(self, buffer, prepared) -> None:
        super()._bind_buffer_view(buffer, prepared)
        self.bound_views.append((prepared.binding_index, buffer))


def _shared_bindings(target: str) -> dict[str, NativeRuntimeBufferBinding]:
    allocation = RuntimeAllocationView(
        allocation_id="working-set",
        byte_length=len(_INITIAL_VALUES) * 4,
        allocation_byte_length=len(_INITIAL_VALUES) * 4,
    )
    directx = target == "directx"
    return {
        "source_values": NativeRuntimeBufferBinding(
            name="source_values",
            binding=RuntimeResourceBinding(
                name="source_values",
                kind="storage-buffer",
                type_name="StructuredBuffer<uint>" if directx else "uint[]",
                set=0,
                binding=0,
                access="read",
            ),
            value=_INITIAL_VALUES,
            source="input",
            dtype="uint32",
            shape=(len(_INITIAL_VALUES),),
            allocation=allocation,
        ),
        "destination_values": NativeRuntimeBufferBinding(
            name="destination_values",
            binding=RuntimeResourceBinding(
                name="destination_values",
                kind="storage-buffer",
                type_name="RWStructuredBuffer<uint>" if directx else "uint[]",
                set=0,
                binding=0 if directx else 1,
                access="write",
            ),
            source="expectedOutput",
            dtype="uint32",
            shape=(len(_EXPECTED_VALUES),),
            expected_output=RuntimeValue(
                name="destination_values",
                dtype="uint32",
                shape=(len(_EXPECTED_VALUES),),
                values=_EXPECTED_VALUES,
            ),
            allocation=allocation,
        ),
    }


def _dispatch_request(
    tmp_path: Path,
    *,
    target: str,
    source: str,
    loaded_artifact: str | bytes,
) -> NativeRuntimeDispatchRequest:
    extension = "hlsl" if target == "directx" else "comp"
    entry_point = "CSMain" if target == "directx" else "main"
    artifact_path = tmp_path / f"shared_allocation.{extension}"
    module_path = (
        tmp_path / "shared_allocation.dxil" if target == "directx" else artifact_path
    )
    artifact_path.write_text(source, encoding="utf-8")
    return NativeRuntimeDispatchRequest(
        target=target,
        artifact={"target": target},
        artifact_path=artifact_path,
        module_path=module_path,
        loaded_artifact=loaded_artifact,
        buffers=_shared_bindings(target),
        constants={},
        dispatch=RuntimeDispatchGeometry(
            entry_point=entry_point,
            workgroup_count=(len(_INITIAL_VALUES), 1, 1),
        ),
        entry_point=entry_point,
    )


def _assert_output(outputs) -> None:
    assert outputs == {
        "destination_values": {
            "dtype": "uint32",
            "shape": [len(_EXPECTED_VALUES)],
            "values": _EXPECTED_VALUES,
        }
    }


def test_directx_shared_native_allocation_executes_on_device(tmp_path):
    if os.environ.get("CROSTL_RUN_SHARED_NATIVE_ALLOCATION_DIRECTX_DEVICE_TEST") != "1":
        pytest.skip("Direct3D shared-allocation device test is not enabled")
    if not sys.platform.startswith("win32"):
        pytest.fail("Direct3D shared-allocation runtime proof requires Windows")
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.fail("DXC is required for the Direct3D shared-allocation proof")
    try:
        __import__("compushady")
    except ImportError as exc:
        pytest.fail(f"Direct3D shared-allocation dependency is unavailable: {exc}")

    source_path = tmp_path / "shared_allocation.hlsl"
    module_path = tmp_path / "shared_allocation.dxil"
    source_path.write_text(_DIRECTX_SOURCE, encoding="utf-8")
    result = subprocess.run(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            "-Fo",
            str(module_path),
            str(source_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    request = _dispatch_request(
        tmp_path,
        target="directx",
        source=_DIRECTX_SOURCE,
        loaded_artifact=module_path.read_bytes(),
    )
    runtime = _DirectXAllocationProbe()

    outputs = runtime.dispatch(None, None, request)

    _assert_output(outputs)
    assert len(runtime.working_resources) == 1
    assert [
        (namespace, binding) for namespace, binding, _resource in runtime.bound_views
    ] == [("srv", 0), ("uav", 0)]
    assert all(
        resource is runtime.working_resources[0]
        for _namespace, _binding, resource in runtime.bound_views
    )


def test_opengl_shared_native_allocation_executes_on_device(tmp_path):
    if os.environ.get("CROSTL_RUN_SHARED_NATIVE_ALLOCATION_OPENGL_DEVICE_TEST") != "1":
        pytest.skip("OpenGL shared-allocation device test is not enabled")
    if not sys.platform.startswith("linux"):
        pytest.fail("OpenGL shared-allocation runtime proof requires Linux")
    try:
        __import__("moderngl")
    except ImportError as exc:
        pytest.fail(f"OpenGL shared-allocation dependency is unavailable: {exc}")

    request = _dispatch_request(
        tmp_path,
        target="opengl",
        source=_OPENGL_SOURCE,
        loaded_artifact=_OPENGL_SOURCE,
    )
    runtime = _OpenGLAllocationProbe()

    outputs = runtime.dispatch(None, None, request)

    _assert_output(outputs)
    assert len(runtime.working_resources) == 1
    assert [binding for binding, _resource in runtime.bound_views] == [0, 1]
    assert all(
        resource is runtime.working_resources[0]
        for _binding, resource in runtime.bound_views
    )
