import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import crosstl.project as project_api
from crosstl.project.native_runtime_drivers import (
    VulkanComputeRuntime,
    _first_vulkan_handle,
    _prepare_vulkan_buffers,
    _read_mapped_memory,
    _write_mapped_memory,
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
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


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
