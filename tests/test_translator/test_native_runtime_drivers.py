from pathlib import Path

import pytest

import crosstl.project as project_api
from crosstl.project.native_runtime_drivers import (
    VulkanComputeRuntime,
    _prepare_vulkan_buffers,
)
from crosstl.project.runtime_verification import (
    UNAVAILABLE,
    NativeRuntimeBufferBinding,
    RuntimeAdapterSetupError,
    RuntimeArtifactSelector,
    RuntimeExecutionRequest,
    RuntimeFixture,
    RuntimeResourceBinding,
    VulkanRuntimeParityAdapter,
    verify_runtime_test_manifest,
)


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
            "out": NativeRuntimeBufferBinding(
                name="out",
                binding=RuntimeResourceBinding(
                    name="out",
                    kind="buffer",
                    set=0,
                    binding=1,
                ),
                source="expectedOutput",
                dtype="float32",
                shape=(2,),
            ),
        }
    )

    assert [buffer.name for buffer in buffers] == ["lhs", "out"]
    assert buffers[0].payload == b"\x00\x00\x80?\x00\x00\x00@"
    assert buffers[1].payload == b"\x00" * 8
