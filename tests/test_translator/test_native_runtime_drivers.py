import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import crosstl.project as project_api
from crosstl._crosstl import translate
from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    OpenGLComputeRuntime,
    VulkanComputeRuntime,
    _compushady_backend_name,
    _first_vulkan_handle,
    _prepare_directx_buffers,
    _prepare_directx_constants,
    _prepare_opengl_buffers,
    _prepare_opengl_specializations,
    _prepare_vulkan_buffers,
    _read_mapped_memory,
    _validate_directx_register_layout,
    _write_mapped_memory,
)
from crosstl.project.runtime_verification import (
    UNAVAILABLE,
    DirectXRuntimeParityAdapter,
    NativeRuntimeBufferBinding,
    NativeRuntimeConstantBinding,
    NativeRuntimeDispatchRequest,
    OpenGLRuntimeParityAdapter,
    RuntimeAdapterContract,
    RuntimeAdapterDispatchError,
    RuntimeAdapterSetupError,
    RuntimeArtifactSelector,
    RuntimeDispatchGeometry,
    RuntimeExecutionRequest,
    RuntimeExecutorUnavailable,
    RuntimeFixture,
    RuntimeResourceBinding,
    RuntimeSpecializationConstant,
    VulkanRuntimeParityAdapter,
    build_runtime_test_manifest,
    verify_runtime_test_manifest,
)
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

ROOT = Path(__file__).resolve().parents[2]


def _runtime_request(tmp_path: Path) -> RuntimeExecutionRequest:
    return RuntimeExecutionRequest(
        fixture=RuntimeFixture(
            id="vulkan-vector-add",
            selector=RuntimeArtifactSelector(target="vulkan"),
        ),
        artifact={"target": "vulkan", "path": "out/vulkan/add.spv"},
        artifact_path=tmp_path / "out" / "vulkan" / "add.spv",
        project_root=tmp_path,
    )


def _opengl_dispatch_request(tmp_path: Path) -> NativeRuntimeDispatchRequest:
    return NativeRuntimeDispatchRequest(
        target="opengl",
        artifact={"target": "opengl"},
        artifact_path=tmp_path / "runtime.comp",
        module_path=tmp_path / "runtime.comp",
        loaded_artifact="#version 430\nvoid main() {}\n",
        buffers={
            "output_values": NativeRuntimeBufferBinding(
                name="output_values",
                binding=RuntimeResourceBinding(
                    name="output_values", kind="storage-buffer", set=0, binding=0
                ),
                source="expectedOutput",
                dtype="float32",
                shape=(1,),
            )
        },
        constants={},
        dispatch=RuntimeDispatchGeometry(entry_point="main", workgroup_count=(1, 1, 1)),
        entry_point="main",
    )


def _directx_dispatch_request(tmp_path: Path) -> NativeRuntimeDispatchRequest:
    return NativeRuntimeDispatchRequest(
        target="directx",
        artifact={"target": "directx"},
        artifact_path=tmp_path / "runtime.hlsl",
        module_path=tmp_path / "runtime.dxil",
        loaded_artifact=b"DXBC\x00\x01",
        buffers={
            "params": NativeRuntimeBufferBinding(
                name="params",
                binding=RuntimeResourceBinding(
                    name="params",
                    kind="constant-buffer",
                    type_name="Params",
                    set=0,
                    binding=0,
                    access="read",
                ),
                value=[3],
                source="input",
                dtype="uint32",
                shape=(1,),
            ),
            "lhs": NativeRuntimeBufferBinding(
                name="lhs",
                binding=RuntimeResourceBinding(
                    name="lhs",
                    kind="buffer",
                    type_name="StructuredBuffer<float>",
                    set=0,
                    binding=0,
                    access="read",
                ),
                value=[1.0, 2.0],
                source="input",
                dtype="float32",
                shape=(2,),
            ),
            "out": NativeRuntimeBufferBinding(
                name="out",
                binding=RuntimeResourceBinding(
                    name="out",
                    kind="buffer",
                    type_name="RWStructuredBuffer<float>",
                    set=0,
                    binding=0,
                    access="read_write",
                ),
                source="expectedOutput",
                dtype="float32",
                shape=(2,),
                metadata={"runtimeValueName": "result"},
            ),
        },
        constants={},
        dispatch=RuntimeDispatchGeometry(entry_point="main", workgroup_count=(2, 3, 1)),
        entry_point="main",
    )


class _FakeDirectXDevice:
    name = "Fake Direct3D Adapter"
    is_hardware = True
    is_discrete = False


class _FakeDirectXBuffer:
    def __init__(self, runtime, size, heap_type, stride, device):
        self.runtime = runtime
        self.size = size
        self.heap_type = heap_type
        self.stride = stride
        self.device = device
        self.payload = bytearray(size)
        self.released = False

    def upload(self, payload):
        self.payload[: len(payload)] = payload

    def copy_to(self, destination):
        if (
            destination.heap_type == self.runtime.HEAP_READBACK
            and self.runtime.fail_readback_copy
        ):
            raise RuntimeError("readback copy failed")
        destination.payload[: len(self.payload)] = self.payload

    def readback(self, size=0, offset=0):
        if self.runtime.fail_readback:
            raise RuntimeError("readback failed")
        size = size or len(self.payload) - offset
        return bytes(self.payload[offset : offset + size])

    def release(self):
        self.released = True


class _FakeDirectXCompute:
    def __init__(self, runtime, shader, cbv, srv, uav, device):
        self.runtime = runtime
        self.shader = shader
        self.cbv = cbv
        self.srv = srv
        self.uav = uav
        self.device = device
        self.dispatch_args = None
        self.released = False

    def dispatch(self, x, y, z):
        self.dispatch_args = (x, y, z)
        if self.runtime.fail_dispatch:
            raise RuntimeError("dispatch rejected")
        scale = struct.unpack("<I", self.cbv[0].payload[:4])[0]
        values = struct.unpack("<2f", self.srv[0].payload[:8])
        self.uav[0].payload[:8] = struct.pack(
            "<2f", *(value * scale for value in values)
        )

    def release(self):
        self.released = True


class _FakeCompushady:
    HEAP_DEFAULT = 0
    HEAP_UPLOAD = 1
    HEAP_READBACK = 2

    def __init__(self, *, backend="d3d12"):
        self.backend = type("Backend", (), {"name": backend})()
        self.device = _FakeDirectXDevice()
        self.devices = [self.device]
        self.buffers = []
        self.computes = []
        self.fail_device = False
        self.fail_compute = False
        self.fail_dispatch = False
        self.fail_default_buffer = False
        self.fail_readback_copy = False
        self.fail_readback = False

    def get_backend(self):
        return self.backend

    def get_discovered_devices(self):
        return self.devices

    def get_current_device(self):
        if self.fail_device:
            raise RuntimeError("device creation failed")
        return self.device

    def Buffer(self, size, heap_type=0, stride=0, device=None):
        if heap_type == self.HEAP_DEFAULT and self.fail_default_buffer:
            raise RuntimeError("default buffer creation failed")
        buffer = _FakeDirectXBuffer(self, size, heap_type, stride, device)
        self.buffers.append(buffer)
        return buffer

    def Compute(self, shader, cbv=None, srv=None, uav=None, device=None):
        if self.fail_compute:
            raise RuntimeError("pipeline creation failed")
        compute = _FakeDirectXCompute(
            self,
            shader,
            list(cbv or []),
            list(srv or []),
            list(uav or []),
            device,
        )
        self.computes.append(compute)
        return compute


def test_directx_compute_runtime_reports_unsupported_platform_without_import(tmp_path):
    def unexpected_loader(name):
        raise AssertionError(f"unexpected import: {name}")

    runtime = DirectXComputeRuntime(
        module_loader=unexpected_loader,
        platform_name="darwin",
    )

    availability = runtime.is_available(None, _runtime_request(tmp_path))

    assert availability.available is False
    assert availability.details == {
        "reasonKind": "platform-unavailable",
        "target": "directx",
        "platform": "darwin",
        "requiredPlatforms": ["win32"],
    }


def test_directx_compute_runtime_reports_missing_python_binding(tmp_path):
    def missing_loader(name):
        assert name == "compushady"
        raise ModuleNotFoundError(name)

    runtime = DirectXComputeRuntime(
        module_loader=missing_loader,
        platform_name="win32",
    )

    availability = runtime.is_available(None, _runtime_request(tmp_path))

    assert availability.available is False
    assert availability.details["reasonKind"] == "dependency-unavailable"
    assert availability.details["missingPythonModules"] == ["compushady"]


def test_directx_compute_runtime_reports_wrong_compushady_backend(tmp_path):
    module = _FakeCompushady(backend="vulkan")
    runtime = DirectXComputeRuntime(
        module_loader=lambda name: module,
        platform_name="win32",
    )

    availability = runtime.is_available(None, _runtime_request(tmp_path))

    assert availability.available is False
    assert availability.details["reasonKind"] == "backend-unavailable"
    assert availability.details["requiredBackend"] == "d3d12"
    assert availability.details["activeBackend"] == "vulkan"


def test_compushady_backend_name_accepts_imported_backend_module():
    backend = ModuleType("compushady.backends.d3d12")

    assert _compushady_backend_name(backend) == "d3d12"


def test_directx_compute_runtime_fails_closed_for_device_selection_error(tmp_path):
    module = _FakeCompushady()
    module.fail_device = True
    runtime = DirectXComputeRuntime(
        module_loader=lambda name: module,
        platform_name="win32",
    )

    availability = runtime.is_available(None, _runtime_request(tmp_path))

    assert availability.available is False
    assert availability.details["reasonKind"] == "device-selection-failed"
    assert availability.details["error"] == "device creation failed"


def test_directx_compute_runtime_reports_empty_device_list(tmp_path):
    module = _FakeCompushady()
    module.devices = []
    runtime = DirectXComputeRuntime(
        module_loader=lambda name: module,
        platform_name="win32",
    )

    availability = runtime.is_available(None, _runtime_request(tmp_path))

    assert availability.available is False
    assert availability.details["reasonKind"] == "device-unavailable"
    assert "error" not in availability.details


def test_directx_compute_runtime_reports_selected_device(tmp_path):
    module = _FakeCompushady()
    runtime = DirectXComputeRuntime(
        module_loader=lambda name: module,
        platform_name="win32",
    )

    availability = runtime.is_available(None, _runtime_request(tmp_path))

    assert availability.available is True
    assert availability.details == {
        "reasonKind": "available",
        "target": "directx",
        "runtime": "directx-compute-runtime",
        "backend": "d3d12",
        "device": "Fake Direct3D Adapter",
        "isHardware": True,
        "isDiscrete": False,
    }


def test_directx_compute_runtime_loads_nonempty_dxil(tmp_path):
    artifact_path = tmp_path / "add.dxil"
    artifact_path.write_bytes(b"DXBC\x00\x01")
    runtime = DirectXComputeRuntime(
        module_loader=lambda name: object(),
        platform_name="win32",
    )

    assert runtime.load_artifact(None, None, artifact_path) == b"DXBC\x00\x01"


def test_directx_compute_runtime_rejects_empty_dxil(tmp_path):
    artifact_path = tmp_path / "empty.dxil"
    artifact_path.write_bytes(b"")
    runtime = DirectXComputeRuntime(platform_name="win32")

    with pytest.raises(RuntimeAdapterSetupError) as excinfo:
        runtime.load_artifact(None, None, artifact_path)

    assert excinfo.value.details["reasonKind"] == "artifact-empty"


def test_prepare_directx_buffers_maps_and_packs_descriptor_namespaces(tmp_path):
    prepared = _validate_directx_register_layout(
        _prepare_directx_buffers(_directx_dispatch_request(tmp_path).buffers)
    )

    assert [(item.name, item.namespace, item.binding_index) for item in prepared] == [
        ("params", "cbv", 0),
        ("lhs", "srv", 0),
        ("out", "uav", 0),
    ]
    assert prepared[0].payload == struct.pack("<I", 3)
    assert prepared[0].allocation_size == 256
    assert prepared[0].stride == 0
    assert prepared[1].payload == struct.pack("<2f", 1.0, 2.0)
    assert prepared[1].stride == 4
    assert prepared[2].payload == b"\x00" * 8
    assert prepared[2].output_name == "result"


def test_prepare_directx_buffers_rejects_nonzero_register_space(tmp_path):
    request = _directx_dispatch_request(tmp_path)
    lhs = request.buffers["lhs"]
    binding = NativeRuntimeBufferBinding(
        **{
            **lhs.__dict__,
            "binding": RuntimeResourceBinding(**{**lhs.binding.__dict__, "set": 2}),
        }
    )

    with pytest.raises(RuntimeAdapterSetupError) as excinfo:
        _prepare_directx_buffers({"lhs": binding})

    assert excinfo.value.details["reasonKind"] == "unsupported-register-space"
    assert excinfo.value.details["registerSpace"] == 2


def test_prepare_directx_buffers_rejects_unsupported_resource_kind(tmp_path):
    request = _directx_dispatch_request(tmp_path)
    lhs = request.buffers["lhs"]
    binding = NativeRuntimeBufferBinding(
        **{
            **lhs.__dict__,
            "binding": RuntimeResourceBinding(
                **{**lhs.binding.__dict__, "kind": "texture"}
            ),
        }
    )

    with pytest.raises(RuntimeAdapterSetupError) as excinfo:
        _prepare_directx_buffers({"lhs": binding})

    assert excinfo.value.details["reasonKind"] == "unsupported-resource-kind"


@pytest.mark.parametrize(
    ("type_name", "metadata"),
    [
        ("ByteAddressBuffer", {}),
        ("RWBuffer<float>", {"byteStride": 4}),
    ],
)
def test_prepare_directx_buffers_rejects_views_compushady_cannot_describe(
    tmp_path,
    type_name,
    metadata,
):
    request = _directx_dispatch_request(tmp_path)
    lhs = request.buffers["lhs"]
    binding = NativeRuntimeBufferBinding(
        **{
            **lhs.__dict__,
            "binding": RuntimeResourceBinding(
                **{
                    **lhs.binding.__dict__,
                    "type_name": type_name,
                    "metadata": metadata,
                }
            ),
        }
    )

    with pytest.raises(RuntimeAdapterSetupError) as excinfo:
        _prepare_directx_buffers({"lhs": binding})

    assert excinfo.value.details["reasonKind"] == "unsupported-buffer-view"
    assert excinfo.value.details["type"] == type_name


def test_prepare_directx_buffers_rejects_sparse_registers(tmp_path):
    request = _directx_dispatch_request(tmp_path)
    lhs = request.buffers["lhs"]
    binding = NativeRuntimeBufferBinding(
        **{
            **lhs.__dict__,
            "binding": RuntimeResourceBinding(**{**lhs.binding.__dict__, "binding": 1}),
        }
    )

    with pytest.raises(RuntimeAdapterSetupError) as excinfo:
        _validate_directx_register_layout(_prepare_directx_buffers({"lhs": binding}))

    assert excinfo.value.details["reasonKind"] == "sparse-register-layout"
    assert excinfo.value.details["namespace"] == "srv"
    assert excinfo.value.details["bindings"] == [1]
    assert excinfo.value.details["expectedBindings"] == [0]


def test_prepare_directx_buffers_rejects_duplicate_registers(tmp_path):
    request = _directx_dispatch_request(tmp_path)

    with pytest.raises(RuntimeAdapterSetupError) as excinfo:
        _validate_directx_register_layout(
            _prepare_directx_buffers(
                {
                    "lhs": request.buffers["lhs"],
                    "rhs": NativeRuntimeBufferBinding(
                        **{
                            **request.buffers["lhs"].__dict__,
                            "name": "rhs",
                        }
                    ),
                }
            )
        )

    assert excinfo.value.details["reasonKind"] == "duplicate-register-binding"
    assert excinfo.value.details["duplicateBindings"] == [0]


def test_prepare_directx_constants_packs_explicit_constant_buffer():
    constants = _prepare_directx_constants(
        {
            "scale": NativeRuntimeConstantBinding(
                name="scale",
                constant=RuntimeSpecializationConstant(
                    name="scale",
                    kind="constant-buffer",
                    dtype="float32",
                    metadata={
                        "directx": {
                            "binding": "b0",
                            "byteOffset": 0,
                        }
                    },
                ),
                value=2.5,
            ),
            "count": NativeRuntimeConstantBinding(
                name="count",
                constant=RuntimeSpecializationConstant(
                    name="count",
                    kind="constant-buffer",
                    dtype="uint32",
                    metadata={
                        "directx": {
                            "binding": "b0",
                            "byteOffset": 4,
                        }
                    },
                ),
                value=3,
            ),
        }
    )

    assert len(constants) == 1
    assert constants[0].namespace == "cbv"
    assert constants[0].binding_index == 0
    assert constants[0].payload == struct.pack("<fI", 2.5, 3)
    assert constants[0].allocation_size == 256


def test_prepare_directx_constants_accepts_matching_compiled_literal():
    constants = _prepare_directx_constants(
        {
            "tile_size": NativeRuntimeConstantBinding(
                name="tile_size",
                constant=RuntimeSpecializationConstant(
                    name="tile_size",
                    kind="scalar-constant",
                    dtype="uint32",
                    value=4,
                ),
                value=4,
            )
        }
    )

    assert constants == ()


def test_prepare_directx_constants_rejects_ambiguous_specialization_constant():
    with pytest.raises(RuntimeAdapterSetupError) as excinfo:
        _prepare_directx_constants(
            {
                "tile_size": NativeRuntimeConstantBinding(
                    name="tile_size",
                    constant=RuntimeSpecializationConstant(
                        name="tile_size",
                        dtype="uint32",
                        value=4,
                    ),
                    value=4,
                )
            }
        )

    assert excinfo.value.details["reasonKind"] == "unsupported-constant-binding"


def test_directx_compute_runtime_dispatches_and_reads_typed_output(tmp_path):
    module = _FakeCompushady()
    runtime = DirectXComputeRuntime(
        module_loader=lambda name: module,
        platform_name="win32",
    )

    outputs = runtime.dispatch(None, None, _directx_dispatch_request(tmp_path))

    assert outputs == {
        "result": {
            "dtype": "float32",
            "shape": [2],
            "values": [3.0, 6.0],
        }
    }
    compute = module.computes[0]
    assert compute.shader == b"DXBC\x00\x01"
    assert compute.dispatch_args == (2, 3, 1)
    assert len(compute.cbv) == len(compute.srv) == len(compute.uav) == 1
    assert compute.cbv[0].size == 256
    assert compute.srv[0].stride == compute.uav[0].stride == 4
    assert all(buffer.device is module.device for buffer in module.buffers)
    assert all(buffer.released for buffer in module.buffers)
    assert compute.released is True


def test_directx_runtime_adapter_compiles_dxil_and_dispatches_fixture(tmp_path):
    artifact_path = tmp_path / "out" / "directx" / "scale.hlsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        "[numthreads(1,1,1)] void main() {}\n",
        encoding="utf-8",
    )
    module = _FakeCompushady()

    def compile_dxil(command, *, input_text=None):
        assert input_text is None
        output_path = Path(command[command.index("-Fo") + 1])
        output_path.write_bytes(b"DXBC\x00\x01")
        return {"returncode": 0}

    adapter = DirectXRuntimeParityAdapter(
        runtime=DirectXComputeRuntime(
            module_loader=lambda name: module,
            platform_name="win32",
        ),
        required_tools=(),
        supported_platforms=(),
        tool_resolver=lambda tool: f"/fake/{tool}",
        command_runner=compile_dxil,
    )
    report = verify_runtime_test_manifest(
        {
            "kind": "crosstl-project-portability-report",
            "project": {"root": str(tmp_path), "targets": ["directx"]},
            "artifacts": [
                {
                    "source": "kernels/scale.cgl",
                    "path": "out/directx/scale.hlsl",
                    "target": "directx",
                    "status": "translated",
                    "entryPoints": [{"name": "main", "stage": "compute"}],
                    "resourceBindings": [
                        {
                            "name": "params",
                            "kind": "constant-buffer",
                            "type": "Params",
                            "set": 0,
                            "binding": 0,
                            "access": "read",
                        },
                        {
                            "name": "lhs",
                            "kind": "buffer",
                            "type": "StructuredBuffer<float>",
                            "set": 0,
                            "binding": 0,
                            "access": "read",
                        },
                        {
                            "name": "out",
                            "kind": "buffer",
                            "type": "RWStructuredBuffer<float>",
                            "set": 0,
                            "binding": 0,
                            "access": "read_write",
                        },
                    ],
                    "dispatch": {
                        "entryPoint": "main",
                        "workgroupCount": [2, 3, 1],
                    },
                }
            ],
        },
        {
            "kind": "crosstl-project-runtime-test-manifest",
            "adapters": [
                {
                    "id": "native-directx",
                    "executor": "directx",
                    "adapterKind": "directx-native-runtime",
                    "platformRequirements": {"requiredTools": []},
                }
            ],
            "tests": [
                {
                    "id": "directx-scale",
                    "selector": {
                        "source": "kernels/scale.cgl",
                        "target": "directx",
                    },
                    "adapter": "native-directx",
                    "inputs": [
                        {
                            "name": "params",
                            "dtype": "uint32",
                            "shape": [1],
                            "values": [3],
                        },
                        {
                            "name": "lhs",
                            "dtype": "float32",
                            "shape": [2],
                            "values": [1.0, 2.0],
                        },
                    ],
                    "expectedOutputs": [
                        {
                            "name": "out",
                            "dtype": "float32",
                            "shape": [2],
                            "values": [3.0, 6.0],
                        }
                    ],
                }
            ],
        },
        executors={"directx": adapter},
    )

    result = report["results"][0]
    assert report["success"] is True
    assert result["status"] == "passed"
    assert module.computes[0].shader == b"DXBC\x00\x01"
    assert module.computes[0].dispatch_args == (2, 3, 1)


def test_directx_compute_runtime_executes_mlx_file_scope_lookup_on_device(tmp_path):
    if os.environ.get("CROSTL_RUN_DIRECTX_LOOKUP_DEVICE_TEST") != "1":
        pytest.skip(
            "set CROSTL_RUN_DIRECTX_LOOKUP_DEVICE_TEST=1 to run Direct3D lookup test"
        )
    if not sys.platform.startswith("win32"):
        pytest.fail("Direct3D lookup runtime proof requires Windows")
    if shutil.which("dxc") is None:
        pytest.fail("DXC is required for the Direct3D lookup runtime proof")
    try:
        __import__("compushady")
    except ImportError as exc:
        pytest.fail(f"Direct3D lookup runtime dependency is unavailable: {exc}")

    fixture_dir = ROOT / "tests" / "fixtures" / "runtime_verification" / "mlx"
    source_path = fixture_dir / "file_scope_immutable_lookup.metal"
    artifact_report = json.loads(
        (fixture_dir / "file_scope_immutable_lookup.artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    generated = translate(
        str(source_path),
        backend="directx",
        source_backend="metal",
        format_output=False,
    )
    assert (
        "static const uint lookup_table[2][4] = "
        "{{3u, 5u, 7u, 11u}, {13u, 17u, 19u, 23u}};"
    ) in generated

    artifact_report["project"]["root"] = str(tmp_path)
    artifact_path = tmp_path / artifact_report["artifacts"][0]["path"]
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(generated, encoding="utf-8")
    manifest = build_runtime_test_manifest(
        artifact_report,
        fixture_dir / "file_scope_immutable_lookup.fixture-metadata.json",
        project_root=tmp_path,
    )
    assert manifest["success"] is True, json.dumps(manifest, indent=2)

    report = verify_runtime_test_manifest(
        artifact_report,
        manifest,
        executors={
            "directx": DirectXRuntimeParityAdapter(runtime=DirectXComputeRuntime())
        },
    )

    assert report["success"] is True, json.dumps(report, indent=2)
    assert report["summary"]["passedCount"] == 1, json.dumps(report, indent=2)
    assert report["summary"]["skippedCount"] == 0, json.dumps(report, indent=2)
    assert report["summary"]["unavailableCount"] == 0, json.dumps(report, indent=2)
    assert report["summary"]["failedCount"] == 0, json.dumps(report, indent=2)
    result = report["results"][0]
    assert result["status"] == "passed", json.dumps(report, indent=2)
    assert result["comparisons"] == [
        {
            "name": "output",
            "kind": "buffer",
            "status": "passed",
            "tolerance": {"absolute": 0.0, "relative": 0.0},
            "expected": {"dtype": "uint32", "shape": [4]},
            "actual": {"dtype": "uint32", "shape": [4]},
            "mismatchCount": 0,
            "maxAbsoluteError": 0.0,
            "maxRelativeError": 0.0,
        }
    ]


@pytest.mark.parametrize(
    ("failure", "reason_kind"),
    [
        ("fail_default_buffer", "resource-creation-failed"),
        ("fail_compute", "compute-pipeline-creation-failed"),
        ("fail_dispatch", "dispatch-failed"),
        ("fail_readback_copy", "readback-failed"),
    ],
)
def test_directx_compute_runtime_reports_phase_failure_and_releases_resources(
    tmp_path,
    failure,
    reason_kind,
):
    module = _FakeCompushady()
    setattr(module, failure, True)
    runtime = DirectXComputeRuntime(
        module_loader=lambda name: module,
        platform_name="win32",
    )
    error_type = (
        RuntimeAdapterSetupError
        if reason_kind
        in {"resource-creation-failed", "compute-pipeline-creation-failed"}
        else RuntimeAdapterDispatchError
    )

    with pytest.raises(error_type) as excinfo:
        runtime.dispatch(None, None, _directx_dispatch_request(tmp_path))

    assert excinfo.value.details["reasonKind"] == reason_kind
    assert all(buffer.released for buffer in module.buffers)
    assert all(compute.released for compute in module.computes)


def test_directx_compute_runtime_rejects_dispatch_off_windows(tmp_path):
    runtime = DirectXComputeRuntime(platform_name="linux")

    with pytest.raises(RuntimeExecutorUnavailable, match="Windows only"):
        runtime.dispatch(None, None, _directx_dispatch_request(tmp_path))


def test_directx_compute_runtime_is_exported_from_project_api():
    assert project_api.DirectXComputeRuntime is DirectXComputeRuntime


_OPENGL_SPIRV_HEADER = struct.pack("<5I", 0x07230203, 0x00010000, 0, 1, 0)


class _FakeOpenGLSPIRVBuffer:
    def __init__(self, payload):
        self.payload = bytearray(payload)
        self.storage_binding = None
        self.uniform_binding = None
        self.released = False

    def bind_to_storage_buffer(self, binding):
        self.storage_binding = binding

    def bind_to_uniform_block(self, binding):
        self.uniform_binding = binding

    def read(self, size):
        return bytes(self.payload[:size])

    def release(self):
        self.released = True


class _FakeOpenGLSPIRVContext:
    version_code = 450

    def __init__(self, *, extensions=("GL_ARB_gl_spirv",)):
        self.extensions = set(extensions)
        self.buffers = []
        self.compute_shader_calls = []
        self.barrier_called = False
        self.finish_called = False
        self.released = False

    def compute_shader(self, source):
        self.compute_shader_calls.append(source)
        raise AssertionError("specialized artifacts must not use the GLSL source path")

    def buffer(self, data=None, reserve=None):
        payload = data if data is not None else b"\x00" * reserve
        buffer = _FakeOpenGLSPIRVBuffer(payload)
        self.buffers.append(buffer)
        return buffer

    def memory_barrier(self):
        self.barrier_called = True

    def finish(self):
        self.finish_called = True

    def release(self):
        self.released = True


class _FakeOpenGLSPIRVDriver:
    GL_SHADER_BINARY_FORMAT_SPIR_V = 0x9551
    GL_COMPUTE_SHADER = 0x91B9
    GL_COMPILE_STATUS = 0x8B81
    GL_LINK_STATUS = 0x8B82

    def __init__(self):
        self.shader_binary = None
        self.specialization = None
        self.uniform_queries = []
        self.uniform_values = []
        self.use_program_calls = []
        self.dispatch_calls = []
        self.deleted_programs = []
        self.deleted_shaders = []

    def glCreateShader(self, shader_type):
        assert shader_type == self.GL_COMPUTE_SHADER
        return 11

    def glShaderBinary(self, count, shaders, binary_format, binary, length):
        assert count == 1
        assert list(shaders) == [11]
        assert binary_format == self.GL_SHADER_BINARY_FORMAT_SPIR_V
        self.shader_binary = bytes(binary[:length])

    def glSpecializeShaderARB(
        self,
        shader,
        entry_point,
        constant_count,
        constant_ids,
        constant_values,
    ):
        self.specialization = {
            "shader": shader,
            "entryPoint": entry_point.decode("utf-8"),
            "ids": list(constant_ids[:constant_count]),
            "values": list(constant_values[:constant_count]),
        }

    def glGetShaderiv(self, shader, field):
        assert shader == 11
        assert field == self.GL_COMPILE_STATUS
        return 1

    def glGetShaderInfoLog(self, shader):
        assert shader == 11
        return b""

    def glCreateProgram(self):
        return 17

    def glAttachShader(self, program, shader):
        assert (program, shader) == (17, 11)

    def glLinkProgram(self, program):
        assert program == 17

    def glGetProgramiv(self, program, field):
        assert program == 17
        assert field == self.GL_LINK_STATUS
        return 1

    def glGetProgramInfoLog(self, program):
        assert program == 17
        return b""

    def glGetUniformLocation(self, program, name):
        assert program == 17
        self.uniform_queries.append(name)
        return 3

    def glUniform1f(self, location, value):
        self.uniform_values.append((location, value))

    def glUseProgram(self, program):
        self.use_program_calls.append(program)

    def glDispatchCompute(self, group_x, group_y, group_z):
        self.dispatch_calls.append((group_x, group_y, group_z))

    def glDeleteProgram(self, program):
        self.deleted_programs.append(program)

    def glDeleteShader(self, shader):
        self.deleted_shaders.append(shader)


def _opengl_specialization_binding(
    name,
    constant_id,
    dtype,
    value,
    *,
    required=False,
):
    return NativeRuntimeConstantBinding(
        name=name,
        constant=RuntimeSpecializationConstant(
            name=name,
            constant_id=constant_id,
            kind="specialization-constant",
            dtype=dtype,
            required=required,
            value_provenance={"fixture": "typed-specialization"},
        ),
        value=value,
        source="value",
    )


def _opengl_spirv_dispatch_request(tmp_path, constants):
    return NativeRuntimeDispatchRequest(
        target="opengl",
        artifact={"target": "opengl"},
        artifact_path=tmp_path / "runtime.comp",
        module_path=tmp_path / "runtime.spv",
        loaded_artifact=_OPENGL_SPIRV_HEADER,
        buffers={
            "output_values": NativeRuntimeBufferBinding(
                name="output_values",
                binding=RuntimeResourceBinding(
                    name="output_values", kind="storage-buffer", set=0, binding=0
                ),
                source="expectedOutput",
                dtype="uint32",
                shape=(1,),
            )
        },
        constants=constants,
        dispatch=RuntimeDispatchGeometry(entry_point="main", workgroup_count=(2, 1, 1)),
        entry_point="main",
    )


def test_opengl_compute_runtime_reports_missing_python_binding(tmp_path):
    def missing_loader(name):
        assert name == "moderngl"
        raise ModuleNotFoundError(name)

    runtime = OpenGLComputeRuntime(module_loader=missing_loader)

    availability = runtime.is_available(None, _runtime_request(tmp_path))

    assert availability.available is False
    assert availability.details["reasonKind"] == "dependency-unavailable"
    assert availability.details["missingPythonModules"] == ["moderngl"]


def test_opengl_compute_runtime_probes_and_releases_headless_context(tmp_path):
    class FakeContext:
        version_code = 460

        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    context = FakeContext()
    runtime = OpenGLComputeRuntime(
        module_loader=lambda name: object(),
        context_factory=lambda module: context,
    )

    availability = runtime.is_available(None, _runtime_request(tmp_path))

    assert availability.available is True
    assert availability.details["runtime"] == "opengl-compute-runtime"
    assert availability.details["versionCode"] == 460
    assert context.released is True


@pytest.mark.parametrize(
    ("version_code", "reported_version"),
    [(0, 0), (420, 420), ("unknown", 0)],
)
def test_opengl_compute_runtime_rejects_unknown_or_old_context_version(
    tmp_path, version_code, reported_version
):
    class FakeContext:
        def __init__(self):
            self.version_code = version_code
            self.released = False

        def release(self):
            self.released = True

    context = FakeContext()
    runtime = OpenGLComputeRuntime(
        module_loader=lambda name: object(),
        context_factory=lambda module: context,
    )

    availability = runtime.is_available(None, _runtime_request(tmp_path))

    assert availability.available is False
    assert availability.details["reasonKind"] == "opengl-version-unsupported"
    assert availability.details["requiredVersionCode"] == 430
    assert availability.details["versionCode"] == reported_version
    assert context.released is True


def test_opengl_compute_runtime_rechecks_context_version_at_dispatch(tmp_path):
    class FakeContext:
        version_code = 420

        def __init__(self):
            self.released = False

        def compute_shader(self, source):
            raise AssertionError(f"unsupported context compiled shader: {source}")

        def release(self):
            self.released = True

    context = FakeContext()
    runtime = OpenGLComputeRuntime(
        module_loader=lambda name: object(),
        context_factory=lambda module: context,
    )

    with pytest.raises(RuntimeAdapterSetupError) as excinfo:
        runtime.dispatch(None, None, _opengl_dispatch_request(tmp_path))

    assert excinfo.value.details["reasonKind"] == "opengl-version-unsupported"
    assert excinfo.value.details["requiredVersionCode"] == 430
    assert excinfo.value.details["versionCode"] == 420
    assert context.released is True


def test_opengl_compute_runtime_reports_missing_spirv_extension(tmp_path):
    context = _FakeOpenGLSPIRVContext(extensions=())
    runtime = OpenGLComputeRuntime(
        module_loader=lambda name: object(),
        context_factory=lambda module: context,
    )
    request = RuntimeExecutionRequest(
        fixture=RuntimeFixture(
            id="opengl-specialized",
            selector=RuntimeArtifactSelector(target="opengl"),
        ),
        artifact={"target": "opengl"},
        artifact_path=tmp_path / "runtime.comp",
        project_root=tmp_path,
        adapter_contract=RuntimeAdapterContract(
            specialization_constants=(
                RuntimeSpecializationConstant(
                    name="mode", constant_id=7, dtype="uint32", value=2
                ),
            )
        ),
    )

    availability = runtime.is_available(None, request)

    assert availability.available is False
    assert availability.details == {
        "target": "opengl",
        "reasonKind": "opengl-spirv-capability-unavailable",
        "requiredExtension": "GL_ARB_gl_spirv",
        "versionCode": 450,
        "extensionAvailable": False,
    }
    assert context.released is True


def test_opengl_compute_runtime_reports_missing_spirv_entry_points(tmp_path):
    context = _FakeOpenGLSPIRVContext()
    driver = _FakeOpenGLSPIRVDriver()
    driver.glShaderBinary = None
    modules = {
        "moderngl": object(),
        "OpenGL.GL": driver,
        "OpenGL.GL.ARB.gl_spirv": SimpleNamespace(
            glSpecializeShaderARB=driver.glSpecializeShaderARB
        ),
    }
    runtime = OpenGLComputeRuntime(
        module_loader=modules.__getitem__,
        context_factory=lambda module: context,
    )
    request = RuntimeExecutionRequest(
        fixture=RuntimeFixture(
            id="opengl-specialized",
            selector=RuntimeArtifactSelector(target="opengl"),
        ),
        artifact={"target": "opengl"},
        artifact_path=tmp_path / "runtime.comp",
        project_root=tmp_path,
        adapter_contract=RuntimeAdapterContract(
            specialization_constants=(
                RuntimeSpecializationConstant(
                    name="mode", constant_id=7, dtype="uint32", value=2
                ),
            )
        ),
    )

    availability = runtime.is_available(None, request)

    assert availability.available is False
    assert availability.details["reasonKind"] == (
        "opengl-spirv-entry-points-unavailable"
    )
    assert availability.details["missingEntryPoints"] == ["glShaderBinary"]
    assert availability.details["requiredExtension"] == "GL_ARB_gl_spirv"
    assert context.released is True


def test_opengl_compute_runtime_loads_utf8_glsl(tmp_path):
    artifact_path = tmp_path / "add.comp"
    artifact_path.write_text("#version 430\nvoid main() {}\n", encoding="utf-8")
    runtime = OpenGLComputeRuntime(module_loader=lambda name: object())

    assert runtime.load_artifact(None, None, artifact_path).startswith("#version 430")


def test_opengl_compute_runtime_loads_word_aligned_spirv(tmp_path):
    artifact_path = tmp_path / "add.spv"
    artifact_path.write_bytes(_OPENGL_SPIRV_HEADER)
    runtime = OpenGLComputeRuntime(module_loader=lambda name: object())

    assert runtime.load_artifact(None, None, artifact_path) == _OPENGL_SPIRV_HEADER


def test_opengl_compute_runtime_rejects_truncated_spirv_header(tmp_path):
    artifact_path = tmp_path / "truncated.spv"
    artifact_path.write_bytes(_OPENGL_SPIRV_HEADER[:8])
    runtime = OpenGLComputeRuntime(module_loader=lambda name: object())

    with pytest.raises(RuntimeAdapterSetupError) as excinfo:
        runtime.load_artifact(None, None, artifact_path)

    assert excinfo.value.details["reasonKind"] == "spirv-artifact-layout-invalid"
    assert excinfo.value.details["byteLength"] == 8
    assert excinfo.value.details["minimumByteLength"] == 20


def test_opengl_compute_runtime_specializes_typed_values_and_binds_uniforms(
    tmp_path,
):
    context = _FakeOpenGLSPIRVContext()
    driver = _FakeOpenGLSPIRVDriver()
    modules = {
        "moderngl": object(),
        "OpenGL.GL": driver,
        "OpenGL.GL.ARB.gl_spirv": SimpleNamespace(
            glSpecializeShaderARB=driver.glSpecializeShaderARB
        ),
    }
    runtime = OpenGLComputeRuntime(
        module_loader=modules.__getitem__,
        context_factory=lambda module: context,
    )
    constants = {
        "floating": _opengl_specialization_binding("floating", 40, "float32", 1.5),
        "signed": _opengl_specialization_binding("signed", 20, "int32", -2),
        "unsigned": _opengl_specialization_binding(
            "unsigned", 30, "uint32", 0x80000000
        ),
        "enabled": _opengl_specialization_binding("enabled", 10, "bool", True),
        "gain": NativeRuntimeConstantBinding(
            name="gain",
            constant=RuntimeSpecializationConstant(
                name="gain", kind="uniform", dtype="float32"
            ),
            value=3.0,
            source="input",
        ),
    }

    class FakeState:
        def __init__(self):
            self.details = {}
            self.steps = []

        def record_step(self, phase, action, **kwargs):
            self.steps.append((phase, action, kwargs))

    state = FakeState()
    outputs = runtime.dispatch(
        None,
        state,
        _opengl_spirv_dispatch_request(tmp_path, constants),
    )

    assert outputs == {
        "output_values": {"dtype": "uint32", "shape": [1], "values": [0]}
    }
    assert driver.shader_binary == _OPENGL_SPIRV_HEADER
    assert driver.specialization == {
        "shader": 11,
        "entryPoint": "main",
        "ids": [10, 20, 30, 40],
        "values": [
            1,
            0xFFFFFFFE,
            0x80000000,
            struct.unpack("<I", struct.pack("<f", 1.5))[0],
        ],
    }
    assert driver.uniform_queries == ["gain"]
    assert driver.uniform_values == [(3, 3.0)]
    assert driver.dispatch_calls == [(2, 1, 1)]
    assert context.compute_shader_calls == []
    assert context.barrier_called is True
    assert context.finish_called is True
    assert context.released is True
    assert all(buffer.released for buffer in context.buffers)
    assert driver.deleted_programs == [17]
    assert driver.deleted_shaders == [11]
    report = state.details["openglSpecialization"]
    assert report["specializationEntryPoint"] == "glSpecializeShaderARB"
    assert report["uniformConstantCount"] == 1
    assert [item["id"] for item in report["appliedConstants"]] == [10, 20, 30, 40]
    assert report["appliedConstants"][0]["valueProvenance"] == {
        "fixture": "typed-specialization",
        "bindingSource": "value",
    }
    assert state.steps[0][0:2] == (
        "specialize",
        "specialize-opengl-spirv",
    )


@pytest.mark.parametrize(
    ("constant_id", "dtype", "value", "reason_kind"),
    [
        ("7", "uint32", 1, "specialization-id-invalid"),
        (7, "int64", 1, "specialization-width-unsupported"),
        (7, "vec2", [1.0, 2.0], "specialization-type-unsupported"),
        (7, "bool", 1, "specialization-value-type-invalid"),
        (7, "uint32", -1, "specialization-value-out-of-range"),
        (7, "float32", "1.0", "specialization-value-type-invalid"),
        pytest.param(
            7,
            "float32",
            1 << 4096,
            "specialization-value-out-of-range",
            id="float32-overflowing-integer",
        ),
    ],
)
def test_prepare_opengl_specializations_rejects_invalid_ids_types_and_values(
    constant_id,
    dtype,
    value,
    reason_kind,
):
    binding = _opengl_specialization_binding(
        "invalid",
        constant_id,
        dtype,
        value,
    )

    with pytest.raises(RuntimeAdapterSetupError) as excinfo:
        _prepare_opengl_specializations({"invalid": binding})

    assert excinfo.value.details["reasonKind"] == reason_kind
    assert excinfo.value.details["constant"] == "invalid"


def test_prepare_opengl_specializations_rejects_duplicate_ids():
    bindings = {
        "first": _opengl_specialization_binding("first", 7, "int32", 1),
        "second": _opengl_specialization_binding("second", 7, "uint32", 2),
    }

    with pytest.raises(RuntimeAdapterSetupError) as excinfo:
        _prepare_opengl_specializations(bindings)

    assert excinfo.value.details["reasonKind"] == "specialization-id-duplicate"
    assert excinfo.value.details["constantId"] == 7
    assert excinfo.value.details["constants"] == ["first", "second"]


def test_prepare_opengl_specializations_rejects_missing_required_value():
    binding = _opengl_specialization_binding(
        "required_mode",
        7,
        "uint32",
        None,
        required=True,
    )

    with pytest.raises(RuntimeAdapterSetupError) as excinfo:
        _prepare_opengl_specializations({"required_mode": binding})

    assert excinfo.value.details["reasonKind"] == "specialization-value-missing"
    assert excinfo.value.details["constantId"] == 7


def test_prepare_opengl_buffers_packs_storage_and_uniform_buffers():
    buffers = _prepare_opengl_buffers(
        {
            "lhs": NativeRuntimeBufferBinding(
                name="lhs",
                binding=RuntimeResourceBinding(
                    name="lhs",
                    kind="storage-buffer",
                    set=0,
                    binding=0,
                ),
                value=[1.0, 2.0],
                source="input",
                dtype="float32",
                shape=(2,),
            ),
            "params": NativeRuntimeBufferBinding(
                name="params",
                binding=RuntimeResourceBinding(
                    name="params",
                    kind="constant-buffer",
                    set=0,
                    binding=0,
                ),
                value=[3],
                source="input",
                dtype="uint32",
                shape=(1,),
            ),
            "out": NativeRuntimeBufferBinding(
                name="out",
                binding=RuntimeResourceBinding(
                    name="out",
                    kind="storage-buffer",
                    set=0,
                    binding=1,
                ),
                source="expectedOutput",
                dtype="float32",
                shape=(2,),
                metadata={"runtimeValueName": "result"},
            ),
        }
    )

    assert [(buffer.resource_kind, buffer.binding_index) for buffer in buffers] == [
        ("storage-buffer", 0),
        ("storage-buffer", 1),
        ("constant-buffer", 0),
    ]
    assert buffers[0].payload == b"\x00\x00\x80?\x00\x00\x00@"
    assert buffers[1].payload == b"\x00" * 8
    assert buffers[1].output_name == "result"
    assert buffers[2].payload == b"\x03\x00\x00\x00"


@pytest.mark.parametrize(
    ("kind", "access", "reason_kind"),
    [
        ("constant-buffer", "write", "unsupported-output-resource"),
        ("storage-buffer", "read", "resource-access-mismatch"),
    ],
)
def test_prepare_opengl_buffers_rejects_unwritable_outputs(kind, access, reason_kind):
    binding = NativeRuntimeBufferBinding(
        name="out",
        binding=RuntimeResourceBinding(
            name="out",
            kind=kind,
            set=0,
            binding=0,
            access=access,
        ),
        source="expectedOutput",
        dtype="uint32",
        shape=(1,),
    )

    with pytest.raises(RuntimeAdapterSetupError) as excinfo:
        _prepare_opengl_buffers({"out": binding})

    assert excinfo.value.details["reasonKind"] == reason_kind
    assert excinfo.value.details["resource"] == "out"
    assert excinfo.value.details["binding"] == 0


def test_opengl_compute_runtime_dispatches_and_reads_storage_buffer(tmp_path):
    class FakeUniform:
        value = None

    class FakeBuffer:
        def __init__(self, payload):
            self.payload = bytearray(payload)
            self.storage_binding = None
            self.uniform_binding = None
            self.released = False

        def bind_to_storage_buffer(self, binding):
            self.storage_binding = binding

        def bind_to_uniform_block(self, binding):
            self.uniform_binding = binding

        def read(self, size):
            return bytes(self.payload[:size])

        def release(self):
            self.released = True

    class FakeShader:
        def __init__(self, context):
            self.context = context
            self.uniforms = {"scale": FakeUniform()}
            self.run_args = None
            self.released = False

        def __getitem__(self, name):
            return self.uniforms[name]

        def run(self, **kwargs):
            self.run_args = kwargs
            lhs, out = self.context.buffers
            values = struct.unpack("<2f", lhs.payload)
            scale = self.uniforms["scale"].value
            out.payload[:] = struct.pack("<2f", *(value * scale for value in values))

        def release(self):
            self.released = True

    class FakeContext:
        version_code = 460

        def __init__(self):
            self.buffers = []
            self.shader = None
            self.barrier_called = False
            self.finish_called = False
            self.released = False

        def compute_shader(self, source):
            assert source.startswith("#version 430")
            self.shader = FakeShader(self)
            return self.shader

        def buffer(self, data=None, reserve=None):
            payload = data if data is not None else b"\x00" * reserve
            buffer = FakeBuffer(payload)
            self.buffers.append(buffer)
            return buffer

        def memory_barrier(self):
            self.barrier_called = True

        def finish(self):
            self.finish_called = True

        def release(self):
            self.released = True

    context = FakeContext()
    loaded_modules = []

    def load_module(name):
        loaded_modules.append(name)
        return object()

    runtime = OpenGLComputeRuntime(
        module_loader=load_module,
        context_factory=lambda module: context,
    )
    request = NativeRuntimeDispatchRequest(
        target="opengl",
        artifact={"target": "opengl"},
        artifact_path=tmp_path / "add.comp",
        module_path=tmp_path / "add.comp",
        loaded_artifact="#version 430\nvoid main() {}\n",
        buffers={
            "lhs": NativeRuntimeBufferBinding(
                name="lhs",
                binding=RuntimeResourceBinding(
                    name="lhs", kind="storage-buffer", set=0, binding=0
                ),
                value=[1.0, 2.0],
                source="input",
                dtype="float32",
                shape=(2,),
            ),
            "out": NativeRuntimeBufferBinding(
                name="out",
                binding=RuntimeResourceBinding(
                    name="out", kind="storage-buffer", set=0, binding=1
                ),
                source="expectedOutput",
                dtype="float32",
                shape=(2,),
                metadata={"runtimeValueName": "result"},
            ),
        },
        constants={
            "scale": NativeRuntimeConstantBinding(
                name="scale",
                constant=RuntimeSpecializationConstant(
                    name="scale", kind="uniform", dtype="float32"
                ),
                value=3.0,
                source="input",
            )
        },
        dispatch=RuntimeDispatchGeometry(entry_point="main", workgroup_count=(2, 1, 1)),
        entry_point="main",
    )

    outputs = runtime.dispatch(None, None, request)

    assert outputs == {
        "result": {"dtype": "float32", "shape": [2], "values": [3.0, 6.0]}
    }
    assert context.shader.run_args == {"group_x": 2, "group_y": 1, "group_z": 1}
    assert context.shader.uniforms["scale"].value == 3.0
    assert context.barrier_called is True
    assert context.finish_called is True
    assert all(buffer.released for buffer in context.buffers)
    assert context.shader.released is True
    assert context.released is True
    assert loaded_modules == ["moderngl"]


def test_opengl_compute_runtime_releases_buffer_when_binding_fails(tmp_path):
    class FailingBuffer:
        def __init__(self):
            self.released = False

        def bind_to_storage_buffer(self, binding):
            _ = binding
            raise RuntimeError("storage binding unavailable")

        def release(self):
            self.released = True

    class FakeShader:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    class FakeContext:
        version_code = 460

        def __init__(self):
            self.buffer_resource = FailingBuffer()
            self.shader = FakeShader()
            self.released = False

        def compute_shader(self, source):
            _ = source
            return self.shader

        def buffer(self, data=None, reserve=None):
            _ = data, reserve
            return self.buffer_resource

        def release(self):
            self.released = True

    context = FakeContext()
    runtime = OpenGLComputeRuntime(
        module_loader=lambda name: object(),
        context_factory=lambda module: context,
    )

    with pytest.raises(RuntimeAdapterSetupError) as excinfo:
        runtime.dispatch(None, None, _opengl_dispatch_request(tmp_path))

    assert excinfo.value.details["reasonKind"] == "resource-binding-failed"
    assert excinfo.value.details["resource"] == "output_values"
    assert context.buffer_resource.released is True
    assert context.shader.released is True
    assert context.released is True


def test_opengl_compute_runtime_reports_synchronization_failure(tmp_path):
    class FakeBuffer:
        def __init__(self):
            self.released = False

        def bind_to_storage_buffer(self, binding):
            _ = binding

        def release(self):
            self.released = True

    class FakeShader:
        def __init__(self):
            self.released = False

        def run(self, **kwargs):
            _ = kwargs

        def release(self):
            self.released = True

    class FakeContext:
        version_code = 460

        def __init__(self):
            self.buffer_resource = FakeBuffer()
            self.shader = FakeShader()
            self.released = False

        def compute_shader(self, source):
            _ = source
            return self.shader

        def buffer(self, data=None, reserve=None):
            _ = data, reserve
            return self.buffer_resource

        def memory_barrier(self):
            raise RuntimeError("barrier failed")

        def release(self):
            self.released = True

    context = FakeContext()
    runtime = OpenGLComputeRuntime(
        module_loader=lambda name: object(),
        context_factory=lambda module: context,
    )

    with pytest.raises(RuntimeAdapterDispatchError) as excinfo:
        runtime.dispatch(None, None, _opengl_dispatch_request(tmp_path))

    assert excinfo.value.details["reasonKind"] == "synchronization-failed"
    assert context.buffer_resource.released is True
    assert context.shader.released is True
    assert context.released is True


def test_opengl_compute_runtime_rejects_short_output_readback(tmp_path):
    class FakeBuffer:
        def __init__(self):
            self.released = False

        def bind_to_storage_buffer(self, binding):
            _ = binding

        def read(self, size):
            assert size == 4
            return b"\x00\x00"

        def release(self):
            self.released = True

    class FakeShader:
        def __init__(self):
            self.released = False

        def run(self, **kwargs):
            _ = kwargs

        def release(self):
            self.released = True

    class FakeContext:
        version_code = 460

        def __init__(self):
            self.buffer_resource = FakeBuffer()
            self.shader = FakeShader()
            self.released = False

        def compute_shader(self, source):
            _ = source
            return self.shader

        def buffer(self, data=None, reserve=None):
            _ = data, reserve
            return self.buffer_resource

        def memory_barrier(self):
            pass

        def release(self):
            self.released = True

    context = FakeContext()
    runtime = OpenGLComputeRuntime(
        module_loader=lambda name: object(),
        context_factory=lambda module: context,
    )

    with pytest.raises(RuntimeAdapterDispatchError) as excinfo:
        runtime.dispatch(None, None, _opengl_dispatch_request(tmp_path))

    assert excinfo.value.details == {
        "target": "opengl",
        "runtime": "opengl-compute-runtime",
        "reasonKind": "readback-size-mismatch",
        "resource": "output_values",
        "expectedByteLength": 4,
        "actualByteLength": 2,
    }
    assert context.buffer_resource.released is True
    assert context.shader.released is True
    assert context.released is True


def test_opengl_compute_runtime_is_exported_from_project_api():
    assert project_api.OpenGLComputeRuntime is OpenGLComputeRuntime


def test_opengl_compute_runtime_executes_vector_scale_on_device(tmp_path):
    if os.environ.get("CROSTL_RUN_OPENGL_DEVICE_TEST") != "1":
        pytest.skip("set CROSTL_RUN_OPENGL_DEVICE_TEST=1 to run OpenGL device test")
    pytest.importorskip("moderngl")

    source_path = tmp_path / "scale.cgl"
    source_path.write_text(
        """
shader OpenGLRuntimeScale {
    StructuredBuffer<float> input_values @ binding(0);
    RWStructuredBuffer<float> output_values @ binding(1);

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        void main(uint3 tid @ gl_GlobalInvocationID) {
            float value = buffer_load(input_values, tid.x);
            buffer_store(output_values, tid.x, value * 2.0);
        }
    }
}
""".lstrip(),
        encoding="utf-8",
    )
    translated = translate(
        str(source_path),
        backend="opengl",
        format_output=False,
    )

    runtime = OpenGLComputeRuntime(context_backends=("egl",))
    request = NativeRuntimeDispatchRequest(
        target="opengl",
        artifact={"target": "opengl"},
        artifact_path=tmp_path / "scale.comp",
        module_path=tmp_path / "scale.comp",
        loaded_artifact=translated,
        buffers={
            "input_values": NativeRuntimeBufferBinding(
                name="input_values",
                binding=RuntimeResourceBinding(
                    name="input_values", kind="storage-buffer", set=0, binding=0
                ),
                value=[1.5, -2.0, 4.0],
                source="input",
                dtype="float32",
                shape=(3,),
            ),
            "output_values": NativeRuntimeBufferBinding(
                name="output_values",
                binding=RuntimeResourceBinding(
                    name="output_values", kind="storage-buffer", set=0, binding=1
                ),
                source="expectedOutput",
                dtype="float32",
                shape=(3,),
                metadata={"runtimeValueName": "scaled"},
            ),
        },
        constants={},
        dispatch=RuntimeDispatchGeometry(entry_point="main", workgroup_count=(3, 1, 1)),
        entry_point="main",
    )

    outputs = runtime.dispatch(None, None, request)

    assert outputs == {
        "scaled": {
            "dtype": "float32",
            "shape": [3],
            "values": [3.0, -4.0, 8.0],
        }
    }


def test_opengl_compute_runtime_specializes_generated_spirv_on_device(tmp_path):
    if os.environ.get("CROSTL_RUN_OPENGL_SPIRV_DEVICE_TEST") != "1":
        pytest.skip(
            "set CROSTL_RUN_OPENGL_SPIRV_DEVICE_TEST=1 to run OpenGL SPIR-V test"
        )
    try:
        __import__("moderngl")
        __import__("OpenGL.GL")
    except ImportError as exc:
        pytest.fail(f"OpenGL SPIR-V device test dependency is unavailable: {exc}")
    if shutil.which("glslangValidator") is None:
        pytest.fail("glslangValidator is required for the OpenGL SPIR-V device test")

    source_code = """
    shader OpenGLRuntimeSpecialization {
        constant uint selected_value @function_constant(7) = 1u;
        RWStructuredBuffer<uint> output_values @ binding(0);

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main(uint3 tid @ gl_GlobalInvocationID) {
                buffer_store(output_values, tid.x, selected_value + tid.x);
            }
        }
    }
    """
    generated = GLSLCodeGen().generate(Parser(Lexer(source_code).tokens).parse())
    artifact_path = tmp_path / "out" / "opengl" / "specialized.glsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(generated, encoding="utf-8")

    artifact_report = {
        "kind": "crosstl-project-portability-report",
        "project": {"root": str(tmp_path), "targets": ["opengl"]},
        "artifacts": [
            {
                "source": "kernels/specialized.cgl",
                "path": "out/opengl/specialized.glsl",
                "target": "opengl",
                "status": "translated",
                "entryPoints": [
                    {
                        "name": "main",
                        "stage": "compute",
                        "workgroupSize": [1, 1, 1],
                    }
                ],
                "resourceBindings": [
                    {
                        "name": "output_values",
                        "kind": "storage-buffer",
                        "set": 0,
                        "binding": 0,
                        "access": "read_write",
                    }
                ],
                "specializationConstants": [
                    {
                        "name": "selected_value",
                        "id": 7,
                        "kind": "specialization-constant",
                        "dtype": "uint32",
                        "default": 1,
                    }
                ],
                "dispatch": {"entryPoint": "main", "globalSize": [3, 1, 1]},
            }
        ],
    }
    manifest = {
        "kind": "crosstl-project-runtime-test-manifest",
        "adapters": [
            {
                "id": "opengl-native",
                "executor": "opengl",
                "adapterKind": "opengl-native-runtime",
                "platformRequirements": {"requiredTools": []},
            }
        ],
        "tests": [
            {
                "id": f"opengl-specialized-{value}",
                "selector": {
                    "source": "kernels/specialized.cgl",
                    "target": "opengl",
                },
                "adapter": "opengl-native",
                "entryPoint": "main",
                "expectedOutputs": [
                    {
                        "name": "output_values",
                        "kind": "buffer",
                        "dtype": "uint32",
                        "shape": [3],
                        "values": [value, value + 1, value + 2],
                    }
                ],
                "runtimeAdapter": {
                    "specializationConstants": [
                        {
                            "name": "selected_value",
                            "id": 7,
                            "kind": "specialization-constant",
                            "dtype": "uint32",
                            "value": value,
                            "valueProvenance": {
                                "fixture": f"opengl-specialized-{value}",
                                "selection": "explicit",
                            },
                        }
                    ],
                    "dispatch": {"globalSize": [3, 1, 1]},
                },
            }
            for value in (2, 9)
        ],
    }
    adapter = OpenGLRuntimeParityAdapter(
        runtime=OpenGLComputeRuntime(context_backends=("egl",))
    )

    report = verify_runtime_test_manifest(
        artifact_report,
        manifest,
        executors={"opengl": adapter},
    )

    assert report["success"] is True, json.dumps(report, indent=2)
    assert report["summary"]["passedCount"] == 2, json.dumps(report, indent=2)
    assert report["summary"]["skippedCount"] == 0, json.dumps(report, indent=2)
    assert report["summary"]["unavailableCount"] == 0, json.dumps(report, indent=2)
    assert report["summary"]["failedCount"] == 0, json.dumps(report, indent=2)
    assert [result["status"] for result in report["results"]] == [
        "passed",
        "passed",
    ], json.dumps(report, indent=2)
    for result, value in zip(report["results"], (2, 9)):
        details = result["executor"]["details"]
        specialization = details["openglSpecialization"]
        assert specialization["mode"] == "spirv-specialization"
        assert specialization["appliedConstantCount"] == 1
        assert specialization["appliedConstants"] == [
            {
                "name": "selected_value",
                "id": 7,
                "dtype": "uint32",
                "value": value,
                "valueProvenance": {
                    "fixture": f"opengl-specialized-{value}",
                    "selection": "explicit",
                    "bindingSource": "value",
                },
                "source": "value",
            }
        ]


def test_vulkan_compute_runtime_reports_missing_python_binding(tmp_path):
    def missing_loader(name):
        assert name == "vulkan"
        raise ModuleNotFoundError(name)

    runtime = VulkanComputeRuntime(module_loader=missing_loader)

    availability = runtime.is_available(None, _runtime_request(tmp_path))

    assert availability.available is False
    assert availability.details["reasonKind"] == "dependency-unavailable"
    assert availability.details["missingPythonModules"] == ["vulkan"]


def test_vulkan_compute_runtime_reports_unavailable_through_native_adapter(tmp_path):
    artifact_path = tmp_path / "out" / "vulkan" / "add.spv"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"\x03\x02\x23\x07")

    def missing_loader(name):
        assert name == "vulkan"
        raise ModuleNotFoundError(name)

    adapter = VulkanRuntimeParityAdapter(
        runtime=VulkanComputeRuntime(module_loader=missing_loader),
        required_tools=(),
        validate=False,
    )
    report = verify_runtime_test_manifest(
        {
            "kind": "crosstl-project-portability-report",
            "project": {"root": str(tmp_path), "targets": ["vulkan"]},
            "artifacts": [
                {
                    "source": "kernels/add.cgl",
                    "path": "out/vulkan/add.spv",
                    "target": "vulkan",
                    "status": "translated",
                    "entryPoints": [{"name": "main", "stage": "compute"}],
                    "resourceBindings": [
                        {"name": "lhs", "kind": "buffer", "set": 0, "binding": 0},
                        {"name": "out", "kind": "buffer", "set": 0, "binding": 1},
                    ],
                    "dispatch": {"entryPoint": "main", "workgroupCount": [1, 1, 1]},
                }
            ],
        },
        {
            "kind": "crosstl-project-runtime-test-manifest",
            "adapters": [
                {
                    "id": "native-vulkan",
                    "executor": "vulkan",
                    "adapterKind": "vulkan-native-runtime",
                    "platformRequirements": {"requiredTools": []},
                }
            ],
            "tests": [
                {
                    "id": "vulkan-add",
                    "selector": {"source": "kernels/add.cgl", "target": "vulkan"},
                    "adapter": "native-vulkan",
                    "inputs": [
                        {
                            "name": "lhs",
                            "dtype": "float32",
                            "shape": [1],
                            "values": [1.0],
                        }
                    ],
                    "expectedOutputs": [
                        {
                            "name": "out",
                            "dtype": "float32",
                            "shape": [1],
                            "values": [2.0],
                        }
                    ],
                }
            ],
        },
        executors={"vulkan": adapter},
    )

    result = report["results"][0]
    assert report["success"] is True
    assert result["status"] == UNAVAILABLE
    assert result["executor"]["details"]["reasonKind"] == "dependency-unavailable"
    assert result["executor"]["details"]["missingPythonModules"] == ["vulkan"]


def test_vulkan_compute_runtime_is_exported_from_project_api():
    assert project_api.VulkanComputeRuntime is VulkanComputeRuntime


def test_vulkan_compute_runtime_loads_word_aligned_spirv(tmp_path):
    artifact_path = tmp_path / "add.spv"
    artifact_path.write_bytes(b"\x03\x02\x23\x07")
    runtime = VulkanComputeRuntime(module_loader=lambda _name: object())

    assert runtime.load_artifact(None, None, artifact_path) == b"\x03\x02\x23\x07"


def test_vulkan_compute_runtime_rejects_unaligned_spirv(tmp_path):
    artifact_path = tmp_path / "add.spv"
    artifact_path.write_bytes(b"\x03\x02\x23")
    runtime = VulkanComputeRuntime(module_loader=lambda _name: object())

    with pytest.raises(RuntimeAdapterSetupError):
        runtime.load_artifact(None, None, artifact_path)


def test_prepare_vulkan_buffers_packs_inputs_and_zeroes_outputs():
    buffers = _prepare_vulkan_buffers(
        {
            "lhs": NativeRuntimeBufferBinding(
                name="lhs",
                binding=RuntimeResourceBinding(
                    name="lhs",
                    kind="buffer",
                    set=0,
                    binding=0,
                ),
                value=[1.0, 2.0],
                source="input",
                dtype="float32",
                shape=(2,),
            ),
            "scale": NativeRuntimeBufferBinding(
                name="scaleUniform",
                binding=RuntimeResourceBinding(
                    name="scaleUniform",
                    kind="constant-buffer",
                    set=0,
                    binding=2,
                ),
                value=3,
                source="input",
                dtype="uint32",
                shape=(),
            ),
            "out": NativeRuntimeBufferBinding(
                name="out_",
                binding=RuntimeResourceBinding(
                    name="out_",
                    kind="buffer",
                    set=0,
                    binding=3,
                ),
                source="expectedOutput",
                dtype="float32",
                shape=(2,),
                metadata={"runtimeValueName": "out"},
            ),
        }
    )

    assert [buffer.name for buffer in buffers] == ["lhs", "scale", "out"]
    assert buffers[0].payload == b"\x00\x00\x80?\x00\x00\x00@"
    assert buffers[1].resource_kind == "constant-buffer"
    assert buffers[1].payload == b"\x03\x00\x00\x00"
    assert buffers[2].payload == b"\x00" * 8
    assert buffers[2].output_name == "out"


def test_vulkan_handle_array_unwraps_first_handle():
    class HandleArray:
        def __getitem__(self, index):
            if index == 0:
                return "pipeline-handle"
            raise IndexError(index)

    assert _first_vulkan_handle(HandleArray()) == "pipeline-handle"
    assert _first_vulkan_handle("plain-handle") == "plain-handle"


def test_mapped_memory_helpers_use_buffer_protocol():
    mapped = bytearray(8)

    _write_mapped_memory(mapped, b"abcd")

    assert mapped == bytearray(b"abcd\x00\x00\x00\x00")
    assert _read_mapped_memory(mapped, 6) == b"abcd\x00\x00"


def test_runtime_parity_vulkan_compute_runtime_executes_vector_add_on_device(
    tmp_path,
):
    if os.environ.get("CROSTL_RUN_VULKAN_DEVICE_TEST") != "1":
        pytest.skip("set CROSTL_RUN_VULKAN_DEVICE_TEST=1 to run Vulkan device test")
    pytest.importorskip("vulkan")
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is required to build the Vulkan fixture")

    shader_path = tmp_path / "add.comp"
    artifact_path = tmp_path / "out" / "vulkan" / "add.spv"
    artifact_path.parent.mkdir(parents=True)
    shader_path.write_text(
        """
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer Lhs {
    float values[];
} lhs_buffer;
layout(set = 0, binding = 1) readonly buffer Rhs {
    float values[];
} rhs_buffer;
layout(set = 0, binding = 2) writeonly buffer Out {
    float values[];
} out_buffer;
void main() {
    uint index = gl_GlobalInvocationID.x;
    out_buffer.values[index] = lhs_buffer.values[index] + rhs_buffer.values[index];
}
""".lstrip(),
        encoding="utf-8",
    )
    subprocess.run(
        [glslang, "-V", str(shader_path), "-o", str(artifact_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    report = verify_runtime_test_manifest(
        {
            "kind": "crosstl-project-portability-report",
            "project": {"root": str(tmp_path), "targets": ["vulkan"]},
            "artifacts": [
                {
                    "source": "kernels/add.comp",
                    "path": "out/vulkan/add.spv",
                    "target": "vulkan",
                    "status": "translated",
                    "entryPoints": [{"name": "main", "stage": "compute"}],
                    "resourceBindings": [
                        {"name": "lhs", "kind": "buffer", "set": 0, "binding": 0},
                        {"name": "rhs", "kind": "buffer", "set": 0, "binding": 1},
                        {"name": "out", "kind": "buffer", "set": 0, "binding": 2},
                    ],
                    "dispatch": {"entryPoint": "main", "workgroupCount": [2, 1, 1]},
                }
            ],
        },
        {
            "kind": "crosstl-project-runtime-test-manifest",
            "adapters": [
                {
                    "id": "native-vulkan",
                    "executor": "vulkan",
                    "adapterKind": "vulkan-native-runtime",
                    "platformRequirements": {"requiredTools": []},
                }
            ],
            "tests": [
                {
                    "id": "vulkan-add-device",
                    "selector": {"source": "kernels/add.comp", "target": "vulkan"},
                    "adapter": "native-vulkan",
                    "inputs": [
                        {
                            "name": "lhs",
                            "dtype": "float32",
                            "shape": [2],
                            "values": [1.0, 2.0],
                        },
                        {
                            "name": "rhs",
                            "dtype": "float32",
                            "shape": [2],
                            "values": [10.0, 20.0],
                        },
                    ],
                    "expectedOutputs": [
                        {
                            "name": "out",
                            "dtype": "float32",
                            "shape": [2],
                            "values": [11.0, 22.0],
                        }
                    ],
                }
            ],
        },
        executors={
            "vulkan": VulkanRuntimeParityAdapter(
                runtime=VulkanComputeRuntime(),
                required_tools=("spirv-val",),
            )
        },
    )

    result = report["results"][0]
    failure_context = json.dumps(report, indent=2, sort_keys=True)
    assert report["success"] is True, failure_context
    assert result["status"] == "passed"
    assert result["comparisons"][0]["status"] == "passed", failure_context


def test_runtime_parity_glsl_workgroup_alias_offset_executes_on_vulkan(tmp_path):
    if os.environ.get("CROSTL_RUN_VULKAN_DEVICE_TEST") != "1":
        pytest.skip("set CROSTL_RUN_VULKAN_DEVICE_TEST=1 to run Vulkan device test")
    pytest.importorskip("vulkan")
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is required to build the Vulkan fixture")

    source_code = """
    shader WorkgroupAliasRuntime {
        RWStructuredBuffer<float> outValues @binding(0);

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                threadgroup float storage[8];
                storage[3] = -11.0;
                threadgroup float* tile = storage + 3;
                tile[0] = 7.0;
                buffer_store(outValues, 0, storage[3]);
            }
        }
    }
    """
    generated = GLSLCodeGen().generate(Parser(Lexer(source_code).tokens).parse())
    shader_path = tmp_path / "workgroup-alias.comp"
    artifact_path = tmp_path / "out" / "vulkan" / "workgroup-alias.spv"
    artifact_path.parent.mkdir(parents=True)
    shader_path.write_text(generated, encoding="utf-8")
    subprocess.run(
        [glslang, "-V", "-S", "comp", str(shader_path), "-o", str(artifact_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    report = verify_runtime_test_manifest(
        {
            "kind": "crosstl-project-portability-report",
            "project": {"root": str(tmp_path), "targets": ["vulkan"]},
            "artifacts": [
                {
                    "source": "kernels/workgroup-alias.cgl",
                    "path": "out/vulkan/workgroup-alias.spv",
                    "target": "vulkan",
                    "status": "translated",
                    "entryPoints": [{"name": "main", "stage": "compute"}],
                    "resourceBindings": [
                        {
                            "name": "outValues",
                            "kind": "buffer",
                            "set": 0,
                            "binding": 0,
                        }
                    ],
                    "dispatch": {
                        "entryPoint": "main",
                        "workgroupCount": [1, 1, 1],
                    },
                }
            ],
        },
        {
            "kind": "crosstl-project-runtime-test-manifest",
            "adapters": [
                {
                    "id": "native-vulkan",
                    "executor": "vulkan",
                    "adapterKind": "vulkan-native-runtime",
                    "platformRequirements": {"requiredTools": []},
                }
            ],
            "tests": [
                {
                    "id": "opengl-workgroup-alias-device",
                    "selector": {
                        "source": "kernels/workgroup-alias.cgl",
                        "target": "vulkan",
                    },
                    "adapter": "native-vulkan",
                    "expectedOutputs": [
                        {
                            "name": "outValues",
                            "dtype": "float32",
                            "shape": [1],
                            "values": [7.0],
                        }
                    ],
                }
            ],
        },
        executors={
            "vulkan": VulkanRuntimeParityAdapter(
                runtime=VulkanComputeRuntime(),
                required_tools=("spirv-val",),
            )
        },
    )

    result = report["results"][0]
    failure_context = json.dumps(report, indent=2, sort_keys=True)
    assert report["success"] is True, failure_context
    assert result["status"] == "passed"
    assert result["comparisons"][0]["status"] == "passed", failure_context


def test_runtime_parity_glsl_partial_aggregate_executes_on_vulkan(tmp_path):
    if os.environ.get("CROSTL_RUN_VULKAN_DEVICE_TEST") != "1":
        pytest.skip("set CROSTL_RUN_VULKAN_DEVICE_TEST=1 to run Vulkan device test")
    pytest.importorskip("vulkan")
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is required to build the Vulkan fixture")

    source_code = """
    shader PartialAggregateRuntime {
        struct Pair {
            float x;
            float y;
        };

        struct Bundle {
            Pair pair;
            float values[2];
        };

        RWStructuredBuffer<float> outValues @binding(0);

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                Bundle value = {{5.0}};
                buffer_store(outValues, 0, value.pair.x);
                buffer_store(outValues, 1, value.pair.y + 10.0);
                buffer_store(outValues, 2, value.values[0] + 20.0);
                buffer_store(outValues, 3, value.values[1] + 30.0);
            }
        }
    }
    """
    generated = GLSLCodeGen().generate(Parser(Lexer(source_code).tokens).parse())
    shader_path = tmp_path / "partial-aggregate.comp"
    artifact_path = tmp_path / "out" / "vulkan" / "partial-aggregate.spv"
    artifact_path.parent.mkdir(parents=True)
    shader_path.write_text(generated, encoding="utf-8")
    subprocess.run(
        [glslang, "-V", "-S", "comp", str(shader_path), "-o", str(artifact_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    report = verify_runtime_test_manifest(
        {
            "kind": "crosstl-project-portability-report",
            "project": {"root": str(tmp_path), "targets": ["vulkan"]},
            "artifacts": [
                {
                    "source": "kernels/partial-aggregate.cgl",
                    "path": "out/vulkan/partial-aggregate.spv",
                    "target": "vulkan",
                    "status": "translated",
                    "entryPoints": [{"name": "main", "stage": "compute"}],
                    "resourceBindings": [
                        {
                            "name": "outValues",
                            "kind": "buffer",
                            "set": 0,
                            "binding": 0,
                        }
                    ],
                    "dispatch": {
                        "entryPoint": "main",
                        "workgroupCount": [1, 1, 1],
                    },
                }
            ],
        },
        {
            "kind": "crosstl-project-runtime-test-manifest",
            "adapters": [
                {
                    "id": "native-vulkan",
                    "executor": "vulkan",
                    "adapterKind": "vulkan-native-runtime",
                    "platformRequirements": {"requiredTools": []},
                }
            ],
            "tests": [
                {
                    "id": "opengl-partial-aggregate-device",
                    "selector": {
                        "source": "kernels/partial-aggregate.cgl",
                        "target": "vulkan",
                    },
                    "adapter": "native-vulkan",
                    "expectedOutputs": [
                        {
                            "name": "outValues",
                            "dtype": "float32",
                            "shape": [4],
                            "values": [5.0, 10.0, 20.0, 30.0],
                        }
                    ],
                }
            ],
        },
        executors={
            "vulkan": VulkanRuntimeParityAdapter(
                runtime=VulkanComputeRuntime(),
                required_tools=("spirv-val",),
            )
        },
    )

    result = report["results"][0]
    failure_context = json.dumps(report, indent=2, sort_keys=True)
    assert report["success"] is True, failure_context
    assert result["status"] == "passed"
    assert result["comparisons"][0]["status"] == "passed", failure_context


def test_runtime_parity_vulkan_array_aliases_and_offsets_execute_on_device(tmp_path):
    if os.environ.get("CROSTL_RUN_VULKAN_DEVICE_TEST") != "1":
        pytest.skip("set CROSTL_RUN_VULKAN_DEVICE_TEST=1 to run Vulkan device test")
    pytest.importorskip("vulkan")
    spirv_as = shutil.which("spirv-as")
    if spirv_as is None:
        pytest.skip("spirv-as is required to build the Vulkan fixture")

    source_code = """
    shader WritableArrayRuntime {
        struct Payload {
            float[4] values;
        }

        StructuredBuffer<float> input @binding(0);
        RWStructuredBuffer<Payload> output @binding(1);

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void fill(inout float[4] values) {
                values[0] = 1.0;
                values[1] = 2.0;
                values[2] = 3.0;
                values[3] = 4.0;
            }

            float readValue(device float* src, int index) {
                return src[index];
            }

            float relayRead(device float* src, int index) {
                return readValue(src + 1, index);
            }

            void main() {
                fill(output[0].values);
                output[0].values[3] = relayRead(&input[3], 2);
            }
        }
    }
    """
    spv_assembly = VulkanSPIRVCodeGen().generate(
        Parser(Lexer(source_code).tokens).parse()
    )
    assembly_path = tmp_path / "inout-array.spvasm"
    artifact_path = tmp_path / "out" / "vulkan" / "inout-array.spv"
    artifact_path.parent.mkdir(parents=True)
    assembly_path.write_text(spv_assembly, encoding="utf-8")
    subprocess.run(
        [spirv_as, str(assembly_path), "-o", str(artifact_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    report = verify_runtime_test_manifest(
        {
            "kind": "crosstl-project-portability-report",
            "project": {"root": str(tmp_path), "targets": ["vulkan"]},
            "artifacts": [
                {
                    "source": "kernels/inout-array.cgl",
                    "path": "out/vulkan/inout-array.spv",
                    "target": "vulkan",
                    "status": "translated",
                    "entryPoints": [{"name": "main", "stage": "compute"}],
                    "resourceBindings": [
                        {
                            "name": "input",
                            "kind": "buffer",
                            "set": 0,
                            "binding": 0,
                        },
                        {
                            "name": "output",
                            "kind": "buffer",
                            "set": 0,
                            "binding": 1,
                        },
                    ],
                    "dispatch": {
                        "entryPoint": "main",
                        "workgroupCount": [1, 1, 1],
                    },
                }
            ],
        },
        {
            "kind": "crosstl-project-runtime-test-manifest",
            "adapters": [
                {
                    "id": "native-vulkan",
                    "executor": "vulkan",
                    "adapterKind": "vulkan-native-runtime",
                    "platformRequirements": {"requiredTools": []},
                }
            ],
            "tests": [
                {
                    "id": "vulkan-inout-array-device",
                    "selector": {
                        "source": "kernels/inout-array.cgl",
                        "target": "vulkan",
                    },
                    "adapter": "native-vulkan",
                    "inputs": [
                        {
                            "name": "input",
                            "dtype": "float32",
                            "shape": [8],
                            "values": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                        }
                    ],
                    "expectedOutputs": [
                        {
                            "name": "output",
                            "dtype": "float32",
                            "shape": [4],
                            "values": [1.0, 2.0, 3.0, 6.0],
                        }
                    ],
                }
            ],
        },
        executors={
            "vulkan": VulkanRuntimeParityAdapter(
                runtime=VulkanComputeRuntime(),
                required_tools=("spirv-val",),
            )
        },
    )

    result = report["results"][0]
    failure_context = json.dumps(report, indent=2, sort_keys=True)
    assert report["success"] is True, failure_context
    assert result["status"] == "passed"
    assert result["comparisons"][0]["status"] == "passed", failure_context


def test_runtime_parity_metal_mutating_struct_method_executes_on_vulkan(tmp_path):
    if os.environ.get("CROSTL_RUN_VULKAN_DEVICE_TEST") != "1":
        pytest.skip("set CROSTL_RUN_VULKAN_DEVICE_TEST=1 to run Vulkan device test")
    pytest.importorskip("vulkan")
    spirv_as = shutil.which("spirv-as")
    if spirv_as is None:
        pytest.skip("spirv-as is required to build the Vulkan fixture")
    spirv_val = shutil.which("spirv-val")
    if spirv_val is None:
        pytest.skip("spirv-val is required to validate the Vulkan fixture")

    source_code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Counter {
        float value;

        void add(float delta) {
            value += delta;
        }
    };

    kernel void main(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]]) {
        Counter counter;
        counter.value = input[0];
        counter.add(4.0f);
        output[0] = counter.value;
    }
    """
    source_path = tmp_path / "metal-mutating-receiver.metal"
    source_path.write_text(source_code, encoding="utf-8")
    spv_assembly = translate(
        source_path,
        backend="vulkan",
        format_output=False,
    )
    assembly_path = tmp_path / "metal-mutating-receiver.spvasm"
    artifact_path = tmp_path / "out" / "vulkan" / "metal-mutating-receiver.spv"
    artifact_path.parent.mkdir(parents=True)
    assembly_path.write_text(spv_assembly, encoding="utf-8")
    subprocess.run(
        [
            spirv_as,
            "--target-env",
            "vulkan1.1",
            str(assembly_path),
            "-o",
            str(artifact_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [spirv_val, "--target-env", "vulkan1.1", str(artifact_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    report = verify_runtime_test_manifest(
        {
            "kind": "crosstl-project-portability-report",
            "project": {"root": str(tmp_path), "targets": ["vulkan"]},
            "artifacts": [
                {
                    "source": "kernels/metal-mutating-receiver.metal",
                    "path": "out/vulkan/metal-mutating-receiver.spv",
                    "target": "vulkan",
                    "status": "translated",
                    "entryPoints": [{"name": "main", "stage": "compute"}],
                    "resourceBindings": [
                        {
                            "name": "input",
                            "kind": "buffer",
                            "set": 0,
                            "binding": 0,
                        },
                        {
                            "name": "output",
                            "kind": "buffer",
                            "set": 0,
                            "binding": 1,
                        },
                    ],
                    "dispatch": {
                        "entryPoint": "main",
                        "workgroupCount": [1, 1, 1],
                    },
                }
            ],
        },
        {
            "kind": "crosstl-project-runtime-test-manifest",
            "adapters": [
                {
                    "id": "native-vulkan",
                    "executor": "vulkan",
                    "adapterKind": "vulkan-native-runtime",
                    "platformRequirements": {"requiredTools": []},
                }
            ],
            "tests": [
                {
                    "id": "metal-mutating-receiver-device",
                    "selector": {
                        "source": "kernels/metal-mutating-receiver.metal",
                        "target": "vulkan",
                    },
                    "adapter": "native-vulkan",
                    "inputs": [
                        {
                            "name": "input",
                            "dtype": "float32",
                            "shape": [1],
                            "values": [1.0],
                        }
                    ],
                    "expectedOutputs": [
                        {
                            "name": "output",
                            "dtype": "float32",
                            "shape": [1],
                            "values": [5.0],
                        }
                    ],
                }
            ],
        },
        executors={
            "vulkan": VulkanRuntimeParityAdapter(
                runtime=VulkanComputeRuntime(),
                required_tools=("spirv-val",),
            )
        },
    )

    result = report["results"][0]
    failure_context = json.dumps(report, indent=2, sort_keys=True)
    assert report["success"] is True, failure_context
    assert result["status"] == "passed"
    assert result["comparisons"][0]["status"] == "passed", failure_context
