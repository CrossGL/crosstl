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
from crosstl.project.runtime_graph import (
    RUNTIME_EXECUTION_GRAPH_KIND,
    RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION,
)
from crosstl.project.runtime_graph_execution import execute_runtime_graph
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeAllocationView,
    RuntimeDispatchGeometry,
    RuntimeResourceBinding,
    RuntimeValue,
)

_DIRECTX_DEVICE_TEST_ENV = "CROSTL_RUN_DIRECTX_DISPATCH_SEQUENCE_DEVICE_TEST"
_OPENGL_DEVICE_TEST_ENV = "CROSTL_RUN_OPENGL_DISPATCH_SEQUENCE_DEVICE_TEST"
_MLX_SOURCE_ROOT_ENV = "CROSTL_MLX_SOURCE_ROOT"
_MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
_INPUT_VALUES = [0, 1, 7, 42]
_TEMPORARY_ELEMENT_COUNT = 2
_EXPECTED_VALUES = [50]

_DIRECTX_STAGE_ONE = """\
StructuredBuffer<uint> source_values : register(t0);
RWStructuredBuffer<uint> temporary_values : register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint3 thread_id : SV_DispatchThreadID) {
    uint index = thread_id.x;
    temporary_values[index] = source_values[index * 2u] + source_values[index * 2u + 1u];
}
"""

_DIRECTX_STAGE_TWO = """\
StructuredBuffer<uint> temporary_values : register(t0);
RWStructuredBuffer<uint> result_values : register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint3 thread_id : SV_DispatchThreadID) {
    result_values[thread_id.x] = temporary_values[0] + temporary_values[1];
}
"""

_OPENGL_STAGE_ONE = """\
#version 430
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer SourceValues {
    uint source_values[];
};
layout(std430, binding = 1) writeonly buffer TemporaryValues {
    uint temporary_values[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    temporary_values[index] = source_values[index * 2u] + source_values[index * 2u + 1u];
}
"""

_OPENGL_STAGE_TWO = """\
#version 430
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer TemporaryValues {
    uint temporary_values[];
};
layout(std430, binding = 1) writeonly buffer ResultValues {
    uint result_values[];
};

void main() {
    result_values[gl_GlobalInvocationID.x] = temporary_values[0] + temporary_values[1];
}
"""


class _DirectXSequenceProbe(DirectXComputeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.physical_allocations = []

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
        self.physical_allocations.append((prepared.name, resource.device_buffer))
        return resource


class _OpenGLSequenceProbe(OpenGLComputeRuntime):
    def __init__(self) -> None:
        super().__init__(context_backends=("egl",))
        self.temporary_bindings = []

    def _bind_buffer_view(self, buffer, prepared) -> None:
        super()._bind_buffer_view(buffer, prepared)
        if prepared.name == "temporary":
            self.temporary_bindings.append(buffer)


def _require_device_test(environment: str, description: str) -> None:
    value = os.environ.get(environment)
    if value is None:
        pytest.skip(f"set {environment}=1 to run the {description}")
    if value != "1":
        pytest.fail(f"{environment} must be set to 1, got {value!r}")


def _require_pinned_mlx_checkout() -> Path:
    source_root = os.environ.get(_MLX_SOURCE_ROOT_ENV)
    if not source_root:
        pytest.fail(f"{_MLX_SOURCE_ROOT_ENV} must identify the pinned MLX checkout")
    root = Path(source_root).resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert revision.returncode == 0, revision.stdout + revision.stderr
    assert revision.stdout.strip() == _MLX_COMMIT
    kernels = root / "mlx" / "backend" / "metal" / "kernels"
    assert kernels.is_dir()
    assert any(kernels.glob("*.metal"))
    return root


def _resource_binding(
    name: str,
    *,
    target: str,
    binding: int,
    access: str,
) -> RuntimeResourceBinding:
    type_name = None
    if target == "directx":
        type_name = (
            "RWStructuredBuffer<uint>"
            if access == "write"
            else "StructuredBuffer<uint>"
        )
    return RuntimeResourceBinding(
        name=name,
        kind="storage-buffer",
        type_name=type_name,
        set=0,
        binding=binding,
        access=access,
    )


def _dispatch_sequence_requests(
    tmp_path: Path,
    *,
    target: str,
    stage_one_artifact: str | bytes,
    stage_two_artifact: str | bytes,
) -> tuple[NativeRuntimeDispatchRequest, NativeRuntimeDispatchRequest]:
    directx = target == "directx"
    extension = "hlsl" if directx else "comp"
    entry_point = "CSMain" if directx else "main"
    input_allocation = RuntimeAllocationView(
        allocation_id="input",
        byte_length=len(_INPUT_VALUES) * 4,
        allocation_byte_length=len(_INPUT_VALUES) * 4,
    )
    temporary = RuntimeAllocationView(
        allocation_id="temporary",
        byte_length=_TEMPORARY_ELEMENT_COUNT * 4,
        allocation_byte_length=_TEMPORARY_ELEMENT_COUNT * 4,
    )
    output_allocation = RuntimeAllocationView(
        allocation_id="output",
        byte_length=len(_EXPECTED_VALUES) * 4,
        allocation_byte_length=len(_EXPECTED_VALUES) * 4,
    )

    producer = NativeRuntimeDispatchRequest(
        target=target,
        artifact={"id": "stage-one", "target": target},
        artifact_path=tmp_path / f"stage_one.{extension}",
        module_path=tmp_path / ("stage_one.dxil" if directx else "stage_one.comp"),
        loaded_artifact=stage_one_artifact,
        buffers={
            "source": NativeRuntimeBufferBinding(
                name="source",
                binding=_resource_binding(
                    "source", target=target, binding=0, access="read"
                ),
                value=_INPUT_VALUES,
                source="input",
                dtype="uint32",
                shape=(len(_INPUT_VALUES),),
                allocation=input_allocation,
            ),
            "temporary": NativeRuntimeBufferBinding(
                name="temporary",
                binding=_resource_binding(
                    "temporary",
                    target=target,
                    binding=0 if directx else 1,
                    access="write",
                ),
                dtype="uint32",
                shape=(_TEMPORARY_ELEMENT_COUNT,),
                allocation=temporary,
            ),
        },
        constants={},
        dispatch=RuntimeDispatchGeometry(
            entry_point=entry_point,
            workgroup_count=(_TEMPORARY_ELEMENT_COUNT, 1, 1),
        ),
        entry_point=entry_point,
    )
    consumer = NativeRuntimeDispatchRequest(
        target=target,
        artifact={"id": "stage-two", "target": target},
        artifact_path=tmp_path / f"stage_two.{extension}",
        module_path=tmp_path / ("stage_two.dxil" if directx else "stage_two.comp"),
        loaded_artifact=stage_two_artifact,
        buffers={
            "temporary": NativeRuntimeBufferBinding(
                name="temporary",
                binding=_resource_binding(
                    "temporary", target=target, binding=0, access="read"
                ),
                dtype="uint32",
                shape=(_TEMPORARY_ELEMENT_COUNT,),
                allocation=temporary,
            ),
            "result": NativeRuntimeBufferBinding(
                name="result",
                binding=_resource_binding(
                    "result",
                    target=target,
                    binding=0 if directx else 1,
                    access="write",
                ),
                dtype="uint32",
                shape=(len(_EXPECTED_VALUES),),
                expected_output=RuntimeValue(
                    name="result",
                    dtype="uint32",
                    shape=(len(_EXPECTED_VALUES),),
                    values=_EXPECTED_VALUES,
                ),
                allocation=output_allocation,
            ),
        },
        constants={},
        dispatch=RuntimeDispatchGeometry(
            entry_point=entry_point,
            workgroup_count=(len(_EXPECTED_VALUES), 1, 1),
        ),
        entry_point=entry_point,
    )
    return producer, consumer


def _execution_graph(target: str) -> dict:
    entry_point = "CSMain" if target == "directx" else "main"

    def resource(resource_id, role, element_count, *, lifetime=None):
        payload = {
            "id": resource_id,
            "role": role,
            "kind": "buffer",
            "dtype": "uint32",
            "shape": [element_count],
            "physicalLayout": {
                "storageLayout": "contiguous",
                "elementSizeBytes": 4,
                "elementStrideBytes": 4,
                "byteOffset": 0,
                "byteLength": element_count * 4,
                "allocationByteLength": element_count * 4,
                "alignmentBytes": 4,
            },
            "allocationId": resource_id,
        }
        if lifetime is not None:
            payload["lifetime"] = {
                "firstNode": lifetime[0],
                "lastNode": lifetime[1],
            }
        return payload

    def dispatch(node_id, bindings, workgroup_count, *, depends_on=()):
        return {
            "id": node_id,
            "kind": "dispatch",
            "dependsOn": list(depends_on),
            "dispatch": {
                "artifactSelector": {"id": node_id, "target": target},
                "entryPoint": entry_point,
                "bindings": {
                    name: {"resource": resource_id, "access": access}
                    for name, (resource_id, access) in bindings.items()
                },
                "constants": {},
                "geometry": {"workgroupCount": [workgroup_count, 1, 1]},
            },
        }

    return {
        "kind": RUNTIME_EXECUTION_GRAPH_KIND,
        "schemaVersion": RUNTIME_EXECUTION_GRAPH_SCHEMA_VERSION,
        "id": f"two-stage-{target}",
        "resources": [
            resource("input", "external-input", len(_INPUT_VALUES)),
            resource(
                "temporary",
                "temporary",
                _TEMPORARY_ELEMENT_COUNT,
                lifetime=("stage-one", "stage-two"),
            ),
            resource("output", "external-output", len(_EXPECTED_VALUES)),
        ],
        "nodes": [
            dispatch(
                "stage-one",
                {
                    "source": ("input", "read"),
                    "temporary": ("temporary", "write"),
                },
                _TEMPORARY_ELEMENT_COUNT,
            ),
            {
                "id": "temporary-visible",
                "kind": "barrier",
                "dependsOn": ["stage-one"],
                "barrier": {
                    "resources": ["temporary"],
                    "beforeAccess": "write",
                    "afterAccess": "read",
                },
            },
            dispatch(
                "stage-two",
                {
                    "temporary": ("temporary", "read"),
                    "result": ("output", "write"),
                },
                len(_EXPECTED_VALUES),
                depends_on=("temporary-visible",),
            ),
        ],
    }


def _compile_directx_shader(
    dxc: str,
    tmp_path: Path,
    name: str,
    source: str,
) -> bytes:
    source_path = tmp_path / f"{name}.hlsl"
    module_path = tmp_path / f"{name}.dxil"
    source_path.write_text(source, encoding="utf-8")
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
    artifact = module_path.read_bytes()
    assert artifact
    return artifact


def _validate_opengl_shader(validator: str, source_path: Path) -> None:
    result = subprocess.run(
        [validator, "-S", "comp", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_sequence_output(outputs) -> None:
    assert outputs == {
        "result": {
            "dtype": "uint32",
            "shape": [len(_EXPECTED_VALUES)],
            "values": _EXPECTED_VALUES,
        }
    }


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_reduction_device_fixture_matches_runtime_graph_contract(tmp_path, target):
    requests = _dispatch_sequence_requests(
        tmp_path,
        target=target,
        stage_one_artifact=b"stage-one" if target == "directx" else _OPENGL_STAGE_ONE,
        stage_two_artifact=b"stage-two" if target == "directx" else _OPENGL_STAGE_TWO,
    )

    class ContractRuntime:
        def dispatch_sequence(self, adapter, state, sequence):
            assert adapter is None
            assert state is None
            assert tuple(sequence) == requests
            return {
                "result": {
                    "dtype": "uint32",
                    "shape": [len(_EXPECTED_VALUES)],
                    "values": _EXPECTED_VALUES,
                }
            }

    result = execute_runtime_graph(
        _execution_graph(target),
        runtime=ContractRuntime(),
        requests={"stage-one": requests[0], "stage-two": requests[1]},
    )

    assert requests[0].dispatch.workgroup_count == (_TEMPORARY_ELEMENT_COUNT, 1, 1)
    assert requests[1].dispatch.workgroup_count == (len(_EXPECTED_VALUES), 1, 1)
    assert requests[0].buffers["temporary"].shape == (_TEMPORARY_ELEMENT_COUNT,)
    assert requests[1].buffers["temporary"].allocation is (
        requests[0].buffers["temporary"].allocation
    )
    _assert_sequence_output(result.outputs)


def test_directx_dispatch_sequence_executes_shared_temporary_on_device(tmp_path):
    _require_device_test(
        _DIRECTX_DEVICE_TEST_ENV,
        "Direct3D ordered-dispatch device proof",
    )
    if not sys.platform.startswith("win32"):
        pytest.fail("Direct3D ordered-dispatch runtime proof requires Windows")
    _require_pinned_mlx_checkout()
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.fail("DXC is required for the Direct3D ordered-dispatch proof")
    try:
        __import__("compushady")
    except Exception as exc:
        pytest.fail(f"Direct3D ordered-dispatch dependency is unavailable: {exc}")

    requests = _dispatch_sequence_requests(
        tmp_path,
        target="directx",
        stage_one_artifact=_compile_directx_shader(
            dxc, tmp_path, "stage_one", _DIRECTX_STAGE_ONE
        ),
        stage_two_artifact=_compile_directx_shader(
            dxc, tmp_path, "stage_two", _DIRECTX_STAGE_TWO
        ),
    )
    assert all(
        request.buffers["temporary"].value is None
        and request.buffers["temporary"].expected_output is None
        for request in requests
    )
    runtime = _DirectXSequenceProbe()

    result = execute_runtime_graph(
        _execution_graph("directx"),
        runtime=runtime,
        requests={"stage-one": requests[0], "stage-two": requests[1]},
    )

    _assert_sequence_output(result.outputs)
    assert result.graph_id == "two-stage-directx"
    assert result.target == "directx"
    assert result.executed_nodes == ("stage-one", "stage-two")
    assert result.barrier_nodes == ("temporary-visible",)
    allocation_names = [name for name, _resource in runtime.physical_allocations]
    assert allocation_names.count("input") == 1
    assert allocation_names.count("temporary") == 1
    assert allocation_names.count("output") == 1
    assert len({id(resource) for _name, resource in runtime.physical_allocations}) == 3
    temporary_allocations = [
        resource
        for allocation_id, resource in runtime.physical_allocations
        if allocation_id == "temporary"
    ]
    assert len(temporary_allocations) == 1


def test_opengl_dispatch_sequence_executes_shared_temporary_on_device(tmp_path):
    _require_device_test(
        _OPENGL_DEVICE_TEST_ENV,
        "OpenGL ordered-dispatch device proof",
    )
    if not sys.platform.startswith("linux"):
        pytest.fail("OpenGL ordered-dispatch runtime proof requires Linux")
    _require_pinned_mlx_checkout()
    try:
        __import__("moderngl")
    except Exception as exc:
        pytest.fail(f"OpenGL ordered-dispatch dependency is unavailable: {exc}")
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.fail("glslangValidator is required for the OpenGL shader proof")

    stage_one_path = tmp_path / "stage_one.comp"
    stage_two_path = tmp_path / "stage_two.comp"
    stage_one_path.write_text(_OPENGL_STAGE_ONE, encoding="utf-8")
    stage_two_path.write_text(_OPENGL_STAGE_TWO, encoding="utf-8")
    _validate_opengl_shader(validator, stage_one_path)
    _validate_opengl_shader(validator, stage_two_path)
    requests = _dispatch_sequence_requests(
        tmp_path,
        target="opengl",
        stage_one_artifact=_OPENGL_STAGE_ONE,
        stage_two_artifact=_OPENGL_STAGE_TWO,
    )
    assert all(
        request.buffers["temporary"].value is None
        and request.buffers["temporary"].expected_output is None
        for request in requests
    )
    runtime = _OpenGLSequenceProbe()

    result = execute_runtime_graph(
        _execution_graph("opengl"),
        runtime=runtime,
        requests={"stage-one": requests[0], "stage-two": requests[1]},
    )

    _assert_sequence_output(result.outputs)
    assert result.graph_id == "two-stage-opengl"
    assert result.target == "opengl"
    assert result.executed_nodes == ("stage-one", "stage-two")
    assert result.barrier_nodes == ("temporary-visible",)
    assert len(runtime.temporary_bindings) >= 2
    assert len({id(buffer) for buffer in runtime.temporary_bindings}) == 1
