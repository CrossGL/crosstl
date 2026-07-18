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
    RuntimeDispatchGeometry,
    RuntimeResourceBinding,
    RuntimeValue,
)

_INITIAL_VALUES = [1, 2, 3, 4]
_EXPECTED_VALUES = [3, 5, 7, 9]

_DIRECTX_SOURCE = """\
RWStructuredBuffer<uint> values : register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint3 thread_id : SV_DispatchThreadID) {
    uint index = thread_id.x;
    values[index] = values[index] * 2u + 1u;
}
"""

_OPENGL_SOURCE = """\
#version 430
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer Values {
    uint values[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    values[index] = values[index] * 2u + 1u;
}
"""


def _initialized_read_write_binding(target: str) -> NativeRuntimeBufferBinding:
    type_name = "RWStructuredBuffer<uint>" if target == "directx" else "uint[]"
    return NativeRuntimeBufferBinding(
        name="values",
        binding=RuntimeResourceBinding(
            name="values",
            kind="storage-buffer",
            type_name=type_name,
            set=0,
            binding=0,
            access="read_write",
        ),
        value=_INITIAL_VALUES,
        source="input",
        dtype="uint32",
        shape=(len(_INITIAL_VALUES),),
        expected_output=RuntimeValue(
            name="values",
            dtype="uint32",
            shape=(len(_EXPECTED_VALUES),),
            values=_EXPECTED_VALUES,
        ),
    )


def _dispatch_request(
    tmp_path: Path,
    *,
    target: str,
    source: str,
    loaded_artifact: str | bytes,
) -> NativeRuntimeDispatchRequest:
    extension = "hlsl" if target == "directx" else "comp"
    entry_point = "CSMain" if target == "directx" else "main"
    artifact_path = tmp_path / f"initialized_read_write.{extension}"
    module_path = (
        tmp_path / "initialized_read_write.dxil"
        if target == "directx"
        else artifact_path
    )
    artifact_path.write_text(source, encoding="utf-8")
    binding = _initialized_read_write_binding(target)
    request = NativeRuntimeDispatchRequest(
        target=target,
        artifact={"target": target},
        artifact_path=artifact_path,
        module_path=module_path,
        loaded_artifact=loaded_artifact,
        buffers={"values": binding},
        constants={},
        dispatch=RuntimeDispatchGeometry(
            entry_point=entry_point,
            workgroup_count=(len(_INITIAL_VALUES), 1, 1),
        ),
        entry_point=entry_point,
    )
    assert len(request.buffers) == 1
    assert request.buffers["values"] is binding
    assert binding.binding.binding == 0
    assert binding.expected_output is not None
    return request


def _assert_output(outputs) -> None:
    assert outputs == {
        "values": {
            "dtype": "uint32",
            "shape": [len(_EXPECTED_VALUES)],
            "values": _EXPECTED_VALUES,
        }
    }


def test_directx_initialized_read_write_resource_executes_on_device(tmp_path):
    if os.environ.get("CROSTL_RUN_INITIALIZED_READ_WRITE_DIRECTX_DEVICE_TEST") != "1":
        pytest.skip("Direct3D initialized read-write device test is not enabled")
    if not sys.platform.startswith("win32"):
        pytest.fail("Direct3D initialized read-write runtime proof requires Windows")
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.fail("DXC is required for the Direct3D initialized read-write proof")
    try:
        __import__("compushady")
    except ImportError as exc:
        pytest.fail(
            f"Direct3D initialized read-write runtime dependency is unavailable: {exc}"
        )

    source_path = tmp_path / "initialized_read_write.hlsl"
    module_path = tmp_path / "initialized_read_write.dxil"
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

    outputs = DirectXComputeRuntime().dispatch(None, None, request)

    _assert_output(outputs)


def test_opengl_initialized_read_write_resource_executes_on_device(tmp_path):
    if os.environ.get("CROSTL_RUN_INITIALIZED_READ_WRITE_OPENGL_DEVICE_TEST") != "1":
        pytest.skip("OpenGL initialized read-write device test is not enabled")
    if not sys.platform.startswith("linux"):
        pytest.fail("OpenGL initialized read-write runtime proof requires Linux")
    try:
        __import__("moderngl")
    except ImportError as exc:
        pytest.fail(
            f"OpenGL initialized read-write runtime dependency is unavailable: {exc}"
        )

    request = _dispatch_request(
        tmp_path,
        target="opengl",
        source=_OPENGL_SOURCE,
        loaded_artifact=_OPENGL_SOURCE,
    )

    outputs = OpenGLComputeRuntime(context_backends=("egl",)).dispatch(
        None, None, request
    )

    _assert_output(outputs)
