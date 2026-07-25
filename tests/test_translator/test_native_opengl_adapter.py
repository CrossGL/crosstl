import hashlib
import os
import re
import shlex
import shutil
import struct
import subprocess
import textwrap
from pathlib import Path

import pytest

from crosstl.project.native_loader_abi import generate_native_loader_execution_abi
from crosstl.project.native_opengl_adapter import (
    generate_opengl_native_loader_adapter,
)

_SHADER_SOURCE = (
    b"#version 430 core\n"
    b"layout(local_size_x=1, local_size_y=1, local_size_z=1) in;\n"
    b"void main() {}\n"
)
_SPIRV_BINARY = struct.pack(
    "<5I",
    0x07230203,
    0x00010000,
    0,
    1,
    0,
)


def _descriptor():
    return {
        "schemaVersion": 1,
        "kind": "crosstl-native-loader-abi-descriptor",
        "abiVersion": 1,
        "unitId": "native-opengl-test",
        "target": "opengl",
        "stage": "compute",
        "entryPoint": {
            "name": "main",
            "stage": "compute",
            "executionConfig": {"workgroupSize": [1, 1, 1]},
            "provenance": {},
        },
        "artifact": {
            "packagePath": "kernel.comp",
            "format": "GLSL source",
            "hash": {
                "algorithm": "sha256",
                "value": hashlib.sha256(_SHADER_SOURCE).hexdigest(),
            },
            "sizeBytes": len(_SHADER_SOURCE),
        },
        "source": {
            "path": "kernel.metal",
            "artifactPath": None,
            "backend": "metal",
            "hash": None,
            "remap": None,
        },
        "bindings": [
            {
                "name": "input_values",
                "kind": "buffer",
                "type": "float[]",
                "namespace": "storage-buffer",
                "coordinates": {"set": 0, "binding": 0},
                "access": "read",
                "scalarLayout": None,
                "provenance": {},
            },
            {
                "name": "output_values",
                "kind": "buffer",
                "type": "float[]",
                "namespace": "storage-buffer",
                "coordinates": {"set": 0, "binding": 1},
                "access": "write",
                "scalarLayout": None,
                "provenance": {},
            },
        ],
        "scalarLayout": {"constants": [], "bindings": []},
        "specializationConstants": [],
        "provenance": {},
    }


def _spirv_descriptor():
    return {
        "schemaVersion": 1,
        "kind": "crosstl-native-loader-abi-descriptor",
        "abiVersion": 1,
        "unitId": "native-opengl-spirv-test",
        "target": "opengl",
        "stage": "compute",
        "entryPoint": {
            "name": "main",
            "stage": "compute",
            "executionConfig": {"workgroupSize": [1, 1, 1]},
            "provenance": {},
        },
        "artifact": {
            "packagePath": "kernel.spv",
            "format": "SPIR-V binary",
            "hash": {
                "algorithm": "sha256",
                "value": hashlib.sha256(_SPIRV_BINARY).hexdigest(),
            },
            "sizeBytes": len(_SPIRV_BINARY),
        },
        "source": {
            "path": "kernel.metal",
            "artifactPath": None,
            "backend": "metal",
            "hash": None,
            "remap": None,
        },
        "bindings": [],
        "scalarLayout": {"constants": [], "bindings": []},
        "specializationConstants": [
            {"id": 10, "name": "enabled", "dtype": "bool"},
            {"id": 20, "name": "signed_value", "dtype": "int32"},
            {"id": 30, "name": "unsigned_value", "dtype": "uint32"},
            {"id": 40, "name": "floating_value", "dtype": "float32"},
        ],
        "provenance": {},
    }


def _native_compiler():
    candidates = []
    configured = os.environ.get("CXX")
    if configured:
        candidates.append(shlex.split(configured, posix=os.name != "nt"))
    candidates.extend([[name] for name in ("clang++", "g++", "c++", "cl", "clang-cl")])
    for command in candidates:
        if command and shutil.which(command[0]):
            command[0] = shutil.which(command[0])
            return command
    return None


def _compile_command(compiler, source_path, output_path):
    executable_names = {
        Path(part).name.lower() for part in compiler if not part.startswith("-")
    }
    msvc_style = bool(executable_names & {"cl", "cl.exe", "clang-cl", "clang-cl.exe"})
    if msvc_style:
        return compiler + [
            "/nologo",
            "/std:c++17",
            "/EHsc",
            "/W4",
            "/WX",
            str(source_path),
            f"/Fe{output_path}",
        ]
    return compiler + [
        "-std=c++17",
        "-pedantic-errors",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(source_path),
        "-o",
        str(output_path),
    ]


def test_generated_opengl_adapter_is_deterministic_and_self_contained():
    first = generate_opengl_native_loader_adapter()
    second = generate_opengl_native_loader_adapter()

    assert first == second
    assert first.startswith("#ifndef CROSSTL_NATIVE_LOADER_OPENGL43_ADAPTER_H\n")
    assert first.endswith("#endif /* CROSSTL_NATIVE_LOADER_OPENGL43_ADAPTER_H */\n")
    assert '#include "GL/' not in first
    assert '#include "EGL/' not in first
    assert "malloc(" not in first
    assert "calloc(" not in first
    assert "reinterpret_cast<const void *>" not in first


def test_generated_opengl_adapter_encodes_platform_and_execution_contracts():
    header = generate_opengl_native_loader_adapter()

    for required in (
        "caller-owned current desktop OpenGL 4.3+ context",
        "CrossTLOpenGL43Functions",
        "CrossTLOpenGL43ContextCurrentFunction",
        "glCompileShader",
        "glShaderBinary",
        "glSpecializeShader",
        "glSpecializeShaderARB",
        "glGetShaderInfoLog",
        "glLinkProgram",
        "glGetProgramInfoLog",
        "CROSSTL_OPENGL43_SHADER_STORAGE_BUFFER",
        "CROSSTL_OPENGL43_UNIFORM_BUFFER",
        "glDispatchCompute",
        "glMemoryBarrier",
        "glFinish",
        "glGetBufferSubData",
        "CROSSTL_OPENGL43_STATUS_CONTEXT_NOT_CURRENT",
        "CROSSTL_OPENGL43_STATUS_HASH_ALGORITHM_UNSUPPORTED",
        "CROSSTL_OPENGL43_STATUS_HASH_FAILED",
        "CROSSTL_OPENGL43_STATUS_HASH_MISMATCH",
        "CROSSTL_OPENGL43_STATUS_ARTIFACT_PATH_UNSAFE",
        "CROSSTL_OPENGL43_STATUS_SPIRV_CAPABILITY_UNSUPPORTED",
        "CROSSTL_OPENGL43_STATUS_SPECIALIZATION_UNSUPPORTED",
        "CROSSTL_OPENGL43_STATUS_RESOURCE_SET_UNSUPPORTED",
        "CROSSTL_OPENGL43_STATUS_RESOURCE_KIND_UNSUPPORTED",
        "CROSSTL_OPENGL43_STATUS_DISPATCH_REQUIRED",
        "CROSSTL_OPENGL43_STATUS_SYNCHRONIZATION_REQUIRED",
        "crosstl_opengl43_make_adapter",
    ):
        assert required in header

    dispatch_body = header.split("static inline int32_t crosstl_opengl43_dispatch(", 1)[
        1
    ].split("static inline int32_t crosstl_opengl43_synchronize(", 1)[0]
    assert "context->functions.dispatch_compute(" in dispatch_body
    assert "context->functions.memory_barrier(" in dispatch_body
    assert dispatch_body.index("context->functions.dispatch_compute(") < (
        dispatch_body.index("return CROSSTL_OPENGL43_STATUS_OK;")
    )


def test_generated_opengl_adapter_runs_execution_abi_and_enforces_lifecycle(
    tmp_path,
):
    compiler = _native_compiler()
    if compiler is None:
        pytest.skip("No native C++ compiler is available")

    execution_header = generate_native_loader_execution_abi(_descriptor())
    symbol_match = re.search(
        r"static inline CrossTLNativeLoaderExecutionResult\s+"
        r"([A-Za-z_]\w*_execute)\(",
        execution_header,
    )
    assert symbol_match is not None
    execute_symbol = symbol_match.group(1)
    unit_match = re.search(
        r"static const CrossTLNativeLoaderUnitDescriptor " r"([A-Za-z_]\w*) = \{",
        execution_header,
    )
    assert unit_match is not None
    unit_symbol = unit_match.group(1)

    (tmp_path / "native_loader_execution.h").write_text(
        execution_header, encoding="utf-8"
    )
    (tmp_path / "native_opengl_adapter.h").write_text(
        generate_opengl_native_loader_adapter(), encoding="utf-8"
    )
    (tmp_path / "kernel.comp").write_bytes(_SHADER_SOURCE)
    source_path = tmp_path / "native_opengl_adapter_test.cpp"
    source_path.write_text(
        textwrap.dedent(r"""
            #include "native_loader_execution.h"
            #include "native_opengl_adapter.h"

            #include <cstring>
            #include <fstream>
            #include <string>
            #include <vector>

            #if defined(_WIN32)
            #define TEST_GL_API __stdcall
            #else
            #define TEST_GL_API
            #endif

            struct FakeState {
                int current;
                int diagnostic_count;
                int last_status;
                std::string last_phase;
                std::string last_message;
            };

            static std::vector<unsigned char> fake_buffers[8];
            static CrossTLOpenGL43UInt fake_next_buffer = 1u;
            static CrossTLOpenGL43UInt fake_storage_buffer = 5u;
            static CrossTLOpenGL43UInt fake_uniform_buffer = 4u;
            static CrossTLOpenGL43UInt fake_base_bindings[8] = {6u, 7u};
            static CrossTLOpenGL43UInt fake_current_program = 44u;
            static int fake_compile_success = 1;
            static int fake_link_success = 1;
            static int fake_create_shader_count = 0;
            static int fake_dispatch_count = 0;
            static int fake_barrier_count = 0;
            static int fake_finish_count = 0;
            static int fake_deleted_shader_count = 0;
            static int fake_deleted_program_count = 0;
            static int fake_deleted_buffer_count = 0;

            static int32_t fake_context_current(void *opaque) {
                return static_cast<FakeState *>(opaque)->current;
            }

            static void fake_diagnostic(
                void *opaque,
                int32_t status,
                const char *phase,
                const char *message) {
                FakeState *state = static_cast<FakeState *>(opaque);
                ++state->diagnostic_count;
                state->last_status = status;
                state->last_phase = phase != NULL ? phase : "";
                state->last_message = message != NULL ? message : "";
            }

            static CrossTLOpenGL43Enum TEST_GL_API fake_get_error(void) {
                return CROSSTL_OPENGL43_NO_ERROR;
            }

            static void TEST_GL_API fake_get_integerv(
                CrossTLOpenGL43Enum token,
                CrossTLOpenGL43Int *value) {
                if (token == CROSSTL_OPENGL43_MAJOR_VERSION) {
                    *value = 4;
                } else if (token == CROSSTL_OPENGL43_MINOR_VERSION) {
                    *value = 5;
                } else if (token == CROSSTL_OPENGL43_CURRENT_PROGRAM) {
                    *value = static_cast<CrossTLOpenGL43Int>(
                        fake_current_program);
                } else if (
                    token ==
                    CROSSTL_OPENGL43_SHADER_STORAGE_BUFFER_BINDING) {
                    *value = static_cast<CrossTLOpenGL43Int>(
                        fake_storage_buffer);
                } else {
                    *value = static_cast<CrossTLOpenGL43Int>(
                        fake_uniform_buffer);
                }
            }

            static void TEST_GL_API fake_get_integer_indexed(
                CrossTLOpenGL43Enum token,
                CrossTLOpenGL43UInt index,
                CrossTLOpenGL43Int *value) {
                *value =
                    token ==
                            CROSSTL_OPENGL43_MAX_COMPUTE_WORK_GROUP_COUNT
                        ? 65535
                        : static_cast<CrossTLOpenGL43Int>(
                              fake_base_bindings[index]);
            }

            static CrossTLOpenGL43UInt TEST_GL_API fake_create_shader(
                CrossTLOpenGL43Enum type) {
                ++fake_create_shader_count;
                return type == CROSSTL_OPENGL43_COMPUTE_SHADER ? 11u : 0u;
            }

            static void TEST_GL_API fake_shader_source(
                CrossTLOpenGL43UInt,
                CrossTLOpenGL43Sizei count,
                const CrossTLOpenGL43Char *const *source,
                const CrossTLOpenGL43Int *length) {
                if (count != 1 || source == NULL || source[0] == NULL ||
                    length == NULL || length[0] <= 0) {
                    fake_compile_success = 0;
                }
            }

            static void TEST_GL_API fake_compile_shader(CrossTLOpenGL43UInt) {}

            static void TEST_GL_API fake_get_shader_iv(
                CrossTLOpenGL43UInt,
                CrossTLOpenGL43Enum token,
                CrossTLOpenGL43Int *value) {
                if (token == CROSSTL_OPENGL43_COMPILE_STATUS) {
                    *value = fake_compile_success;
                } else {
                    *value = fake_compile_success ? 1 : 24;
                }
            }

            static void TEST_GL_API fake_get_shader_info_log(
                CrossTLOpenGL43UInt,
                CrossTLOpenGL43Sizei capacity,
                CrossTLOpenGL43Sizei *written,
                CrossTLOpenGL43Char *message) {
                const char *log = "synthetic compile error";
                std::snprintf(
                    message, static_cast<size_t>(capacity), "%s", log);
                if (written != NULL) {
                    *written = static_cast<CrossTLOpenGL43Sizei>(
                        std::strlen(message));
                }
            }

            static void TEST_GL_API fake_delete_shader(CrossTLOpenGL43UInt) {
                ++fake_deleted_shader_count;
            }

            static CrossTLOpenGL43UInt TEST_GL_API fake_create_program(void) {
                return 22u;
            }

            static void TEST_GL_API fake_attach_shader(
                CrossTLOpenGL43UInt, CrossTLOpenGL43UInt) {}

            static void TEST_GL_API fake_link_program(CrossTLOpenGL43UInt) {}

            static void TEST_GL_API fake_get_program_iv(
                CrossTLOpenGL43UInt,
                CrossTLOpenGL43Enum token,
                CrossTLOpenGL43Int *value) {
                if (token == CROSSTL_OPENGL43_LINK_STATUS) {
                    *value = fake_link_success;
                } else {
                    *value = fake_link_success ? 1 : 21;
                }
            }

            static void TEST_GL_API fake_get_program_info_log(
                CrossTLOpenGL43UInt,
                CrossTLOpenGL43Sizei capacity,
                CrossTLOpenGL43Sizei *written,
                CrossTLOpenGL43Char *message) {
                const char *log = "synthetic link error";
                std::snprintf(
                    message, static_cast<size_t>(capacity), "%s", log);
                if (written != NULL) {
                    *written = static_cast<CrossTLOpenGL43Sizei>(
                        std::strlen(message));
                }
            }

            static void TEST_GL_API fake_delete_program(CrossTLOpenGL43UInt) {
                ++fake_deleted_program_count;
            }

            static void TEST_GL_API fake_use_program(
                CrossTLOpenGL43UInt program) {
                fake_current_program = program;
            }

            static void TEST_GL_API fake_gen_buffers(
                CrossTLOpenGL43Sizei count, CrossTLOpenGL43UInt *buffers) {
                for (CrossTLOpenGL43Sizei index = 0; index < count; ++index) {
                    buffers[index] = fake_next_buffer++;
                }
            }

            static CrossTLOpenGL43UInt *fake_bound_buffer(
                CrossTLOpenGL43Enum target) {
                return target == CROSSTL_OPENGL43_SHADER_STORAGE_BUFFER
                    ? &fake_storage_buffer
                    : &fake_uniform_buffer;
            }

            static void TEST_GL_API fake_bind_buffer(
                CrossTLOpenGL43Enum target, CrossTLOpenGL43UInt buffer) {
                *fake_bound_buffer(target) = buffer;
            }

            static void TEST_GL_API fake_buffer_data(
                CrossTLOpenGL43Enum target,
                CrossTLOpenGL43SizePtr size,
                const void *payload,
                CrossTLOpenGL43Enum) {
                const CrossTLOpenGL43UInt buffer = *fake_bound_buffer(target);
                fake_buffers[buffer].assign(
                    static_cast<const unsigned char *>(payload),
                    static_cast<const unsigned char *>(payload) + size);
            }

            static void TEST_GL_API fake_bind_buffer_base(
                CrossTLOpenGL43Enum,
                CrossTLOpenGL43UInt binding,
                CrossTLOpenGL43UInt buffer) {
                fake_base_bindings[binding] = buffer;
            }

            static void TEST_GL_API fake_delete_buffers(
                CrossTLOpenGL43Sizei count,
                const CrossTLOpenGL43UInt *buffers) {
                for (CrossTLOpenGL43Sizei index = 0; index < count; ++index) {
                    fake_buffers[buffers[index]].clear();
                    ++fake_deleted_buffer_count;
                }
            }

            static void TEST_GL_API fake_dispatch_compute(
                CrossTLOpenGL43UInt x,
                CrossTLOpenGL43UInt y,
                CrossTLOpenGL43UInt z) {
                ++fake_dispatch_count;
                if (x != 1u || y != 1u || z != 1u) {
                    return;
                }
                std::vector<unsigned char> &input =
                    fake_buffers[fake_base_bindings[0]];
                std::vector<unsigned char> &output =
                    fake_buffers[fake_base_bindings[1]];
                const size_t count =
                    output.size() / sizeof(float);
                for (size_t index = 0u; index < count; ++index) {
                    float input_value = 0.0f;
                    std::memcpy(
                        &input_value,
                        input.data() + index * sizeof(float),
                        sizeof(float));
                    const float output_value = input_value + 1.0f;
                    std::memcpy(
                        output.data() + index * sizeof(float),
                        &output_value,
                        sizeof(float));
                }
            }

            static void TEST_GL_API fake_memory_barrier(
                CrossTLOpenGL43Bitfield barriers) {
                const CrossTLOpenGL43Bitfield required =
                    CROSSTL_OPENGL43_SHADER_STORAGE_BARRIER_BIT |
                    CROSSTL_OPENGL43_BUFFER_UPDATE_BARRIER_BIT |
                    CROSSTL_OPENGL43_UNIFORM_BARRIER_BIT;
                if ((barriers & required) == required) {
                    ++fake_barrier_count;
                }
            }

            static void TEST_GL_API fake_finish(void) {
                ++fake_finish_count;
            }

            static void TEST_GL_API fake_get_buffer_sub_data(
                CrossTLOpenGL43Enum target,
                CrossTLOpenGL43SizePtr offset,
                CrossTLOpenGL43SizePtr size,
                void *destination) {
                const std::vector<unsigned char> &source =
                    fake_buffers[*fake_bound_buffer(target)];
                std::memcpy(
                    destination,
                    source.data() + offset,
                    static_cast<size_t>(size));
            }

            static CrossTLOpenGL43Functions fake_functions(void) {
                CrossTLOpenGL43Functions functions = {};
                functions.get_error = fake_get_error;
                functions.get_integerv = fake_get_integerv;
                functions.get_integer_indexed = fake_get_integer_indexed;
                functions.create_shader = fake_create_shader;
                functions.shader_source = fake_shader_source;
                functions.compile_shader = fake_compile_shader;
                functions.get_shader_iv = fake_get_shader_iv;
                functions.get_shader_info_log = fake_get_shader_info_log;
                functions.delete_shader = fake_delete_shader;
                functions.create_program = fake_create_program;
                functions.attach_shader = fake_attach_shader;
                functions.link_program = fake_link_program;
                functions.get_program_iv = fake_get_program_iv;
                functions.get_program_info_log = fake_get_program_info_log;
                functions.delete_program = fake_delete_program;
                functions.use_program = fake_use_program;
                functions.gen_buffers = fake_gen_buffers;
                functions.bind_buffer = fake_bind_buffer;
                functions.buffer_data = fake_buffer_data;
                functions.bind_buffer_base = fake_bind_buffer_base;
                functions.delete_buffers = fake_delete_buffers;
                functions.dispatch_compute = fake_dispatch_compute;
                functions.memory_barrier = fake_memory_barrier;
                functions.finish = fake_finish;
                functions.get_buffer_sub_data = fake_get_buffer_sub_data;
                return functions;
            }

            int main(int argc, char **argv) {
                if (argc != 2) {
                    return 1;
                }
                FakeState state = {0, 0, 0, "", ""};
                CrossTLOpenGL43Functions functions = fake_functions();
                CrossTLOpenGL43Context context = {};
                if (crosstl_opengl43_initialize_context(
                        &context,
                        &functions,
                        &state,
                        fake_context_current) !=
                    CROSSTL_OPENGL43_STATUS_OK) {
                    return 2;
                }
                char diagnostic[256] = {};
                context.artifact_root = argv[1];
                context.report_diagnostic = fake_diagnostic;
                context.diagnostic_buffer = diagnostic;
                context.diagnostic_buffer_capacity = sizeof(diagnostic);
                CrossTLNativeLoaderAdapter adapter =
                    crosstl_opengl43_make_adapter(&context);

                float input_values[4] = {1.0f, 2.0f, 4.0f, 8.0f};
                float output_values[4] = {};
                CrossTLNativeLoaderBindingRequest bindings[2] = {
                    {
                        "input_values",
                        "buffer",
                        "float[]",
                        "storage-buffer",
                        0u,
                        0u,
                        CROSSTL_NATIVE_LOADER_ACCESS_READ,
                        input_values,
                        sizeof(input_values)},
                    {
                        "output_values",
                        "buffer",
                        "float[]",
                        "storage-buffer",
                        0u,
                        1u,
                        CROSSTL_NATIVE_LOADER_ACCESS_WRITE,
                        output_values,
                        sizeof(output_values)}};
                CrossTLNativeLoaderExecutionRequest request = {
                    CROSSTL_NATIVE_LOADER_ABI_VERSION,
                    "opengl",
                    2u,
                    bindings,
                    0u,
                    NULL,
                    {{1u, 1u, 1u}, {1u, 1u, 1u}}};

                CrossTLNativeLoaderExecutionResult result =
                    EXECUTE_SYMBOL(&request, &adapter);
                if (result.succeeded ||
                    result.error.phase !=
                        CROSSTL_NATIVE_LOADER_PHASE_LOAD_ARTIFACT ||
                    result.error.adapter_status !=
                        CROSSTL_OPENGL43_STATUS_CONTEXT_NOT_CURRENT ||
                    fake_dispatch_count != 0 ||
                    state.last_status !=
                        CROSSTL_OPENGL43_STATUS_CONTEXT_NOT_CURRENT) {
                    return 3;
                }

                state.current = 1;
                CrossTLOpenGL43Pipeline unused_pipeline = {11u, 22u, 0u, 0};
                if (adapter.synchronize(
                        adapter.context, &unused_pipeline) !=
                    CROSSTL_OPENGL43_STATUS_DISPATCH_REQUIRED) {
                    return 4;
                }
                CrossTLOpenGL43Resource unused_resource = {
                    &context,
                    CROSSTL_OPENGL43_SHADER_STORAGE_BUFFER,
                    1u,
                    1u,
                    0u,
                    sizeof(output_values)};
                context.dispatch_serial = 1u;
                context.synchronized_serial = 0u;
                if (adapter.readback(
                        adapter.context,
                        &unused_resource,
                        NULL,
                        &bindings[1]) !=
                    CROSSTL_OPENGL43_STATUS_SYNCHRONIZATION_REQUIRED) {
                    return 5;
                }
                context.dispatch_serial = 0u;
                CrossTLOpenGL43Artifact source_artifact = {
                    CROSSTL_OPENGL43_ARTIFACT_GLSL_SOURCE, {}, {}};
                uint32_t source_specialization_value = 1u;
                CrossTLNativeLoaderSpecializationDescriptor
                    source_specialization_descriptor = {
                        1u, 7u, "mode", "uint32"};
                CrossTLNativeLoaderSpecializationRequest
                    source_specialization_request = {
                        1u,
                        7u,
                        "mode",
                        "uint32",
                        &source_specialization_value,
                        sizeof(source_specialization_value)};
                if (adapter.apply_specialization(
                        adapter.context,
                        &source_artifact,
                        &source_specialization_descriptor,
                        &source_specialization_request) !=
                    CROSSTL_OPENGL43_STATUS_SPECIALIZATION_UNSUPPORTED) {
                    return 6;
                }
                CrossTLNativeLoaderBindingDescriptor uniform_descriptor = {
                    "constants",
                    "constant-buffer",
                    "Constants",
                    "uniform-buffer",
                    0u,
                    2u,
                    CROSSTL_NATIVE_LOADER_ACCESS_READ,
                    NULL,
                    NULL};
                CrossTLOpenGL43Enum uniform_target = 0u;
                CrossTLOpenGL43Enum uniform_binding_token = 0u;
                if (crosstl_opengl43_binding_target(
                        &context,
                        &uniform_descriptor,
                        &uniform_target,
                        &uniform_binding_token) !=
                        CROSSTL_OPENGL43_STATUS_OK ||
                    uniform_target != CROSSTL_OPENGL43_UNIFORM_BUFFER ||
                    uniform_binding_token !=
                        CROSSTL_OPENGL43_UNIFORM_BUFFER_BINDING) {
                    return 7;
                }
                uniform_descriptor.access =
                    CROSSTL_NATIVE_LOADER_ACCESS_WRITE;
                if (crosstl_opengl43_binding_target(
                        &context,
                        &uniform_descriptor,
                        &uniform_target,
                        &uniform_binding_token) !=
                    CROSSTL_OPENGL43_STATUS_RESOURCE_ACCESS_UNSUPPORTED) {
                    return 8;
                }

                void *loaded_artifact = NULL;
                CrossTLNativeLoaderUnitDescriptor unsafe_unit = UNIT_SYMBOL;
                unsafe_unit.artifact_path = "../outside.comp";
                if (adapter.load_artifact(
                        adapter.context,
                        &unsafe_unit,
                        &loaded_artifact) !=
                        CROSSTL_OPENGL43_STATUS_ARTIFACT_PATH_UNSAFE ||
                    loaded_artifact != NULL ||
                    fake_create_shader_count != 0) {
                    return 9;
                }
                CrossTLNativeLoaderUnitDescriptor unsupported_hash_unit =
                    UNIT_SYMBOL;
                unsupported_hash_unit.artifact_hash_algorithm = "sha1";
                if (adapter.load_artifact(
                        adapter.context,
                        &unsupported_hash_unit,
                        &loaded_artifact) !=
                        CROSSTL_OPENGL43_STATUS_HASH_ALGORITHM_UNSUPPORTED ||
                    loaded_artifact != NULL ||
                    fake_create_shader_count != 0) {
                    return 10;
                }
                CrossTLNativeLoaderUnitDescriptor malformed_hash_unit =
                    UNIT_SYMBOL;
                malformed_hash_unit.artifact_hash_value = "not-a-digest";
                if (adapter.load_artifact(
                        adapter.context,
                        &malformed_hash_unit,
                        &loaded_artifact) !=
                        CROSSTL_OPENGL43_STATUS_HASH_FAILED ||
                    loaded_artifact != NULL ||
                    fake_create_shader_count != 0) {
                    return 11;
                }
                const std::string artifact_path =
                    std::string(argv[1]) + "/kernel.comp";
                {
                    std::fstream artifact_file(
                        artifact_path,
                        std::ios::binary | std::ios::in | std::ios::out);
                    const char replacement = '!';
                    artifact_file.write(&replacement, 1);
                }
                result = EXECUTE_SYMBOL(&request, &adapter);
                if (result.succeeded ||
                    result.error.phase !=
                        CROSSTL_NATIVE_LOADER_PHASE_LOAD_ARTIFACT ||
                    result.error.adapter_status !=
                        CROSSTL_OPENGL43_STATUS_HASH_MISMATCH ||
                    fake_create_shader_count != 0 ||
                    fake_dispatch_count != 0) {
                    return 12;
                }
                {
                    std::fstream artifact_file(
                        artifact_path,
                        std::ios::binary | std::ios::in | std::ios::out);
                    const char replacement = '#';
                    artifact_file.write(&replacement, 1);
                }

                fake_compile_success = 0;
                state.last_message.clear();
                result = EXECUTE_SYMBOL(&request, &adapter);
                if (result.succeeded ||
                    result.error.phase !=
                        CROSSTL_NATIVE_LOADER_PHASE_CREATE_PIPELINE ||
                    result.error.adapter_status !=
                        CROSSTL_OPENGL43_STATUS_SHADER_COMPILE_FAILED ||
                    state.last_message != "synthetic compile error" ||
                    std::strstr(diagnostic, "synthetic compile error") == NULL ||
                    fake_dispatch_count != 0 ||
                    fake_deleted_shader_count != 1) {
                    return 13;
                }

                fake_compile_success = 1;
                fake_link_success = 0;
                state.last_message.clear();
                result = EXECUTE_SYMBOL(&request, &adapter);
                if (result.succeeded ||
                    result.error.phase !=
                        CROSSTL_NATIVE_LOADER_PHASE_CREATE_PIPELINE ||
                    result.error.adapter_status !=
                        CROSSTL_OPENGL43_STATUS_PROGRAM_LINK_FAILED ||
                    state.last_message != "synthetic link error" ||
                    std::strstr(diagnostic, "synthetic link error") == NULL ||
                    fake_dispatch_count != 0 ||
                    fake_deleted_program_count != 1 ||
                    fake_deleted_shader_count != 2) {
                    return 14;
                }

                fake_link_success = 1;
                CrossTLOpenGL43Functions incomplete = functions;
                incomplete.dispatch_compute = NULL;
                CrossTLOpenGL43Context incomplete_context = {};
                crosstl_opengl43_initialize_context(
                    &incomplete_context,
                    &incomplete,
                    &state,
                    fake_context_current);
                incomplete_context.artifact_root = argv[1];
                CrossTLNativeLoaderAdapter incomplete_adapter =
                    crosstl_opengl43_make_adapter(&incomplete_context);
                result = EXECUTE_SYMBOL(&request, &incomplete_adapter);
                if (result.succeeded ||
                    result.error.adapter_status !=
                        CROSSTL_OPENGL43_STATUS_ENTRY_POINT_MISSING ||
                    fake_dispatch_count != 0) {
                    return 15;
                }

                result = EXECUTE_SYMBOL(&request, &adapter);
                if (!result.succeeded ||
                    result.error.code != CROSSTL_NATIVE_LOADER_CODE_OK ||
                    result.cleanup_error.code !=
                        CROSSTL_NATIVE_LOADER_CODE_OK ||
                    fake_dispatch_count != 1 ||
                    fake_barrier_count != 1 ||
                    fake_finish_count != 1 ||
                    fake_deleted_buffer_count != 2 ||
                    fake_deleted_program_count != 2 ||
                    fake_deleted_shader_count != 3 ||
                    fake_storage_buffer != 5u ||
                    fake_uniform_buffer != 4u ||
                    fake_base_bindings[0] != 6u ||
                    fake_base_bindings[1] != 7u ||
                    fake_current_program != 44u) {
                    return 16;
                }
                const float expected[4] = {2.0f, 3.0f, 5.0f, 9.0f};
                for (size_t index = 0u; index < 4u; ++index) {
                    if (output_values[index] != expected[index]) {
                            return 17;
                    }
                }
                return 0;
            }
            """)
        .replace("EXECUTE_SYMBOL", execute_symbol)
        .replace("UNIT_SYMBOL", unit_symbol)
        .lstrip(),
        encoding="utf-8",
    )

    executable_path = tmp_path / (
        "native_opengl_adapter_test.exe"
        if os.name == "nt"
        else "native_opengl_adapter_test"
    )
    compile_result = subprocess.run(
        _compile_command(compiler, source_path, executable_path),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr

    run_result = subprocess.run(
        [str(executable_path), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, (
        f"exit={run_result.returncode}\n" f"{run_result.stdout}{run_result.stderr}"
    )


def test_generated_opengl_adapter_specializes_spirv_with_core_and_arb_paths(
    tmp_path,
):
    compiler = _native_compiler()
    if compiler is None:
        pytest.skip("No native C++ compiler is available")

    execution_header = generate_native_loader_execution_abi(_spirv_descriptor())
    symbol_match = re.search(
        r"static inline CrossTLNativeLoaderExecutionResult\s+"
        r"([A-Za-z_]\w*_execute)\(",
        execution_header,
    )
    assert symbol_match is not None
    execute_symbol = symbol_match.group(1)
    unit_match = re.search(
        r"static const CrossTLNativeLoaderUnitDescriptor " r"([A-Za-z_]\w*) = \{",
        execution_header,
    )
    assert unit_match is not None
    unit_symbol = unit_match.group(1)

    misaligned_binary = _SPIRV_BINARY + b"\x00"
    (tmp_path / "native_loader_spirv_execution.h").write_text(
        execution_header, encoding="utf-8"
    )
    (tmp_path / "native_opengl_adapter.h").write_text(
        generate_opengl_native_loader_adapter(), encoding="utf-8"
    )
    (tmp_path / "kernel.spv").write_bytes(_SPIRV_BINARY)
    (tmp_path / "misaligned.spv").write_bytes(misaligned_binary)
    source_path = tmp_path / "native_opengl_spirv_test.cpp"
    source_path.write_text(
        textwrap.dedent(r"""
            #include "native_loader_spirv_execution.h"
            #include "native_opengl_adapter.h"

            #include <cstring>
            #include <string>

            #if defined(_WIN32)
            #define TEST_GL_API __stdcall
            #else
            #define TEST_GL_API
            #endif

            struct FakeState {
                std::string diagnostic;
            };

            static int fake_minor_version = 5;
            static int fake_contract_valid = 1;
            static int fake_shader_binary_calls = 0;
            static int fake_arb_specialize_calls = 0;
            static int fake_core_specialize_calls = 0;
            static int fake_dispatch_calls = 0;
            static int fake_events[128] = {};
            static size_t fake_event_count = 0u;
            static CrossTLOpenGL43UInt fake_ids[8] = {};
            static CrossTLOpenGL43UInt fake_values[8] = {};
            static CrossTLOpenGL43UInt fake_specialization_count = 0u;
            static std::string fake_entry_point;

            static void record_event(int event) {
                fake_events[fake_event_count++] = event;
            }

            static int event_position(
                size_t begin, int event, size_t *position_out) {
                for (size_t index = begin;
                     index < fake_event_count;
                     ++index) {
                    if (fake_events[index] == event) {
                        *position_out = index;
                        return 1;
                    }
                }
                return 0;
            }

            static int32_t fake_context_current(void *) {
                return 1;
            }

            static void fake_diagnostic(
                void *opaque,
                int32_t,
                const char *,
                const char *message) {
                static_cast<FakeState *>(opaque)->diagnostic =
                    message != NULL ? message : "";
            }

            static CrossTLOpenGL43Enum TEST_GL_API fake_get_error(void) {
                return CROSSTL_OPENGL43_NO_ERROR;
            }

            static void TEST_GL_API fake_get_integerv(
                CrossTLOpenGL43Enum token,
                CrossTLOpenGL43Int *value) {
                if (token == CROSSTL_OPENGL43_MAJOR_VERSION) {
                    *value = 4;
                } else if (token == CROSSTL_OPENGL43_MINOR_VERSION) {
                    *value = fake_minor_version;
                } else {
                    *value = 0;
                }
            }

            static void TEST_GL_API fake_get_integer_indexed(
                CrossTLOpenGL43Enum token,
                CrossTLOpenGL43UInt,
                CrossTLOpenGL43Int *value) {
                *value =
                    token ==
                            CROSSTL_OPENGL43_MAX_COMPUTE_WORK_GROUP_COUNT
                        ? 65535
                        : 0;
            }

            static CrossTLOpenGL43UInt TEST_GL_API fake_create_shader(
                CrossTLOpenGL43Enum type) {
                record_event(10);
                if (type != CROSSTL_OPENGL43_COMPUTE_SHADER) {
                    fake_contract_valid = 0;
                }
                return 11u;
            }

            static void TEST_GL_API fake_shader_binary(
                CrossTLOpenGL43Sizei count,
                const CrossTLOpenGL43UInt *shaders,
                CrossTLOpenGL43Enum format,
                const void *binary,
                CrossTLOpenGL43Sizei length) {
                record_event(20);
                ++fake_shader_binary_calls;
                const unsigned char *bytes =
                    static_cast<const unsigned char *>(binary);
                if (count != 1 || shaders == NULL ||
                    shaders[0] != 11u ||
                    format !=
                        CROSSTL_OPENGL43_SHADER_BINARY_FORMAT_SPIR_V ||
                    length != 20 || bytes == NULL ||
                    bytes[0] != 0x03u || bytes[1] != 0x02u ||
                    bytes[2] != 0x23u || bytes[3] != 0x07u) {
                    fake_contract_valid = 0;
                }
            }

            static void copy_specializations(
                CrossTLOpenGL43UInt shader,
                const CrossTLOpenGL43Char *entry_point,
                CrossTLOpenGL43UInt count,
                const CrossTLOpenGL43UInt *ids,
                const CrossTLOpenGL43UInt *values) {
                if (shader != 11u || entry_point == NULL ||
                    count > 8u || (count != 0u &&
                                   (ids == NULL || values == NULL))) {
                    fake_contract_valid = 0;
                    return;
                }
                fake_entry_point = entry_point;
                fake_specialization_count = count;
                for (CrossTLOpenGL43UInt index = 0u;
                     index < count;
                     ++index) {
                    fake_ids[index] = ids[index];
                    fake_values[index] = values[index];
                }
            }

            static void TEST_GL_API fake_specialize_shader_arb(
                CrossTLOpenGL43UInt shader,
                const CrossTLOpenGL43Char *entry_point,
                CrossTLOpenGL43UInt count,
                const CrossTLOpenGL43UInt *ids,
                const CrossTLOpenGL43UInt *values) {
                record_event(30);
                ++fake_arb_specialize_calls;
                copy_specializations(
                    shader, entry_point, count, ids, values);
            }

            static void TEST_GL_API fake_specialize_shader_core(
                CrossTLOpenGL43UInt shader,
                const CrossTLOpenGL43Char *entry_point,
                CrossTLOpenGL43UInt count,
                const CrossTLOpenGL43UInt *ids,
                const CrossTLOpenGL43UInt *values) {
                record_event(31);
                ++fake_core_specialize_calls;
                copy_specializations(
                    shader, entry_point, count, ids, values);
            }

            static void TEST_GL_API fake_get_shader_iv(
                CrossTLOpenGL43UInt,
                CrossTLOpenGL43Enum token,
                CrossTLOpenGL43Int *value) {
                record_event(35);
                *value =
                    token == CROSSTL_OPENGL43_COMPILE_STATUS ? 1 : 1;
            }

            static void TEST_GL_API fake_get_shader_info_log(
                CrossTLOpenGL43UInt,
                CrossTLOpenGL43Sizei,
                CrossTLOpenGL43Sizei *written,
                CrossTLOpenGL43Char *message) {
                if (written != NULL) {
                    *written = 0;
                }
                if (message != NULL) {
                    message[0] = '\0';
                }
            }

            static void TEST_GL_API fake_delete_shader(
                CrossTLOpenGL43UInt) {
                record_event(95);
            }

            static CrossTLOpenGL43UInt TEST_GL_API
            fake_create_program(void) {
                record_event(40);
                return 17u;
            }

            static void TEST_GL_API fake_attach_shader(
                CrossTLOpenGL43UInt program,
                CrossTLOpenGL43UInt shader) {
                record_event(45);
                if (program != 17u || shader != 11u) {
                    fake_contract_valid = 0;
                }
            }

            static void TEST_GL_API fake_link_program(
                CrossTLOpenGL43UInt program) {
                record_event(50);
                if (program != 17u) {
                    fake_contract_valid = 0;
                }
            }

            static void TEST_GL_API fake_get_program_iv(
                CrossTLOpenGL43UInt,
                CrossTLOpenGL43Enum token,
                CrossTLOpenGL43Int *value) {
                record_event(55);
                *value =
                    token == CROSSTL_OPENGL43_LINK_STATUS ? 1 : 1;
            }

            static void TEST_GL_API fake_get_program_info_log(
                CrossTLOpenGL43UInt,
                CrossTLOpenGL43Sizei,
                CrossTLOpenGL43Sizei *written,
                CrossTLOpenGL43Char *message) {
                if (written != NULL) {
                    *written = 0;
                }
                if (message != NULL) {
                    message[0] = '\0';
                }
            }

            static void TEST_GL_API fake_delete_program(
                CrossTLOpenGL43UInt) {
                record_event(90);
            }

            static void TEST_GL_API fake_use_program(
                CrossTLOpenGL43UInt) {
                record_event(60);
            }

            static void TEST_GL_API fake_gen_buffers(
                CrossTLOpenGL43Sizei count,
                CrossTLOpenGL43UInt *buffers) {
                for (CrossTLOpenGL43Sizei index = 0;
                     index < count;
                     ++index) {
                    buffers[index] =
                        static_cast<CrossTLOpenGL43UInt>(index + 1);
                }
            }

            static void TEST_GL_API fake_bind_buffer(
                CrossTLOpenGL43Enum, CrossTLOpenGL43UInt) {}

            static void TEST_GL_API fake_buffer_data(
                CrossTLOpenGL43Enum,
                CrossTLOpenGL43SizePtr,
                const void *,
                CrossTLOpenGL43Enum) {}

            static void TEST_GL_API fake_bind_buffer_base(
                CrossTLOpenGL43Enum,
                CrossTLOpenGL43UInt,
                CrossTLOpenGL43UInt) {}

            static void TEST_GL_API fake_delete_buffers(
                CrossTLOpenGL43Sizei,
                const CrossTLOpenGL43UInt *) {}

            static void TEST_GL_API fake_dispatch_compute(
                CrossTLOpenGL43UInt,
                CrossTLOpenGL43UInt,
                CrossTLOpenGL43UInt) {
                record_event(70);
                ++fake_dispatch_calls;
            }

            static void TEST_GL_API fake_memory_barrier(
                CrossTLOpenGL43Bitfield) {
                record_event(75);
            }

            static void TEST_GL_API fake_finish(void) {
                record_event(80);
            }

            static void TEST_GL_API fake_get_buffer_sub_data(
                CrossTLOpenGL43Enum,
                CrossTLOpenGL43SizePtr,
                CrossTLOpenGL43SizePtr,
                void *) {}

            static CrossTLOpenGL43Functions fake_functions(void) {
                CrossTLOpenGL43Functions functions = {};
                functions.get_error = fake_get_error;
                functions.get_integerv = fake_get_integerv;
                functions.get_integer_indexed = fake_get_integer_indexed;
                functions.create_shader = fake_create_shader;
                functions.shader_binary = fake_shader_binary;
                functions.specialize_shader =
                    fake_specialize_shader_core;
                functions.specialize_shader_arb =
                    fake_specialize_shader_arb;
                functions.get_shader_iv = fake_get_shader_iv;
                functions.get_shader_info_log =
                    fake_get_shader_info_log;
                functions.delete_shader = fake_delete_shader;
                functions.create_program = fake_create_program;
                functions.attach_shader = fake_attach_shader;
                functions.link_program = fake_link_program;
                functions.get_program_iv = fake_get_program_iv;
                functions.get_program_info_log =
                    fake_get_program_info_log;
                functions.delete_program = fake_delete_program;
                functions.use_program = fake_use_program;
                functions.gen_buffers = fake_gen_buffers;
                functions.bind_buffer = fake_bind_buffer;
                functions.buffer_data = fake_buffer_data;
                functions.bind_buffer_base = fake_bind_buffer_base;
                functions.delete_buffers = fake_delete_buffers;
                functions.dispatch_compute = fake_dispatch_compute;
                functions.memory_barrier = fake_memory_barrier;
                functions.finish = fake_finish;
                functions.get_buffer_sub_data =
                    fake_get_buffer_sub_data;
                return functions;
            }

            static CrossTLNativeLoaderExecutionRequest request(void) {
                static uint8_t enabled = 1u;
                static int32_t signed_value = -2;
                static uint32_t unsigned_value = 0x80000000u;
                static float floating_value = 1.5f;
                static CrossTLNativeLoaderSpecializationRequest
                    specializations[4] = {
                        {
                            1u,
                            10u,
                            "enabled",
                            "bool",
                            &enabled,
                            sizeof(enabled)},
                        {
                            1u,
                            20u,
                            "signed_value",
                            "int32",
                            &signed_value,
                            sizeof(signed_value)},
                        {
                            1u,
                            30u,
                            "unsigned_value",
                            "uint32",
                            &unsigned_value,
                            sizeof(unsigned_value)},
                        {
                            1u,
                            40u,
                            "floating_value",
                            "float32",
                            &floating_value,
                            sizeof(floating_value)}};
                CrossTLNativeLoaderExecutionRequest value = {
                    CROSSTL_NATIVE_LOADER_ABI_VERSION,
                    "opengl",
                    0u,
                    NULL,
                    4u,
                    specializations,
                    {{1u, 1u, 1u}, {1u, 1u, 1u}}};
                return value;
            }

            int main(int argc, char **argv) {
                if (argc != 2) {
                    return 1;
                }
                FakeState state = {""};
                CrossTLOpenGL43Functions functions = fake_functions();
                CrossTLOpenGL43Context context = {};
                if (crosstl_opengl43_initialize_context(
                        &context,
                        &functions,
                        &state,
                        fake_context_current) !=
                    CROSSTL_OPENGL43_STATUS_OK) {
                    return 2;
                }
                context.artifact_root = argv[1];
                context.report_diagnostic = fake_diagnostic;
                CrossTLNativeLoaderAdapter adapter =
                    crosstl_opengl43_make_adapter(&context);
                CrossTLNativeLoaderExecutionRequest execution_request =
                    request();

                void *artifact = NULL;
                CrossTLNativeLoaderUnitDescriptor misaligned_unit =
                    UNIT_SYMBOL;
                misaligned_unit.artifact_path = "misaligned.spv";
                misaligned_unit.artifact_size_bytes = 21u;
                misaligned_unit.artifact_hash_value =
                    "MISALIGNED_HASH";
                if (adapter.load_artifact(
                        adapter.context,
                        &misaligned_unit,
                        &artifact) !=
                        CROSSTL_OPENGL43_STATUS_SPIRV_LAYOUT_INVALID ||
                    artifact != NULL ||
                    fake_shader_binary_calls != 0) {
                    return 3;
                }

                CrossTLNativeLoaderExecutionResult result =
                    EXECUTE_SYMBOL(&execution_request, &adapter);
                if (result.succeeded ||
                    result.error.phase !=
                        CROSSTL_NATIVE_LOADER_PHASE_CREATE_PIPELINE ||
                    result.error.adapter_status !=
                        CROSSTL_OPENGL43_STATUS_SPIRV_CAPABILITY_UNSUPPORTED ||
                    state.diagnostic.find("OpenGL 4.6") ==
                        std::string::npos ||
                    state.diagnostic.find("GL_ARB_gl_spirv") ==
                        std::string::npos ||
                    fake_shader_binary_calls != 0 ||
                    fake_dispatch_calls != 0) {
                    return 4;
                }

                context.supports_arb_gl_spirv = 1u;
                const size_t arb_event_begin = fake_event_count;
                result = EXECUTE_SYMBOL(&execution_request, &adapter);
                if (!result.succeeded ||
                    result.error.code !=
                        CROSSTL_NATIVE_LOADER_CODE_OK ||
                    fake_shader_binary_calls != 1 ||
                    fake_arb_specialize_calls != 1 ||
                    fake_core_specialize_calls != 0 ||
                    fake_dispatch_calls != 1 ||
                    !fake_contract_valid ||
                    fake_entry_point != "main" ||
                    fake_specialization_count != 4u) {
                    return 5;
                }
                const CrossTLOpenGL43UInt expected_ids[4] = {
                    10u, 20u, 30u, 40u};
                const CrossTLOpenGL43UInt expected_values[4] = {
                    1u, 0xfffffffeu, 0x80000000u, 0x3fc00000u};
                for (size_t index = 0u; index < 4u; ++index) {
                    if (fake_ids[index] != expected_ids[index] ||
                        fake_values[index] != expected_values[index]) {
                        return 6;
                    }
                }
                size_t binary_position = 0u;
                size_t specialization_position = 0u;
                size_t link_position = 0u;
                if (!event_position(
                        arb_event_begin, 20, &binary_position) ||
                    !event_position(
                        arb_event_begin,
                        30,
                        &specialization_position) ||
                    !event_position(
                        arb_event_begin, 50, &link_position) ||
                    !(binary_position < specialization_position &&
                      specialization_position < link_position)) {
                    return 7;
                }

                fake_minor_version = 6;
                context.supports_arb_gl_spirv = 0u;
                result = EXECUTE_SYMBOL(&execution_request, &adapter);
                if (!result.succeeded ||
                    fake_shader_binary_calls != 2 ||
                    fake_arb_specialize_calls != 1 ||
                    fake_core_specialize_calls != 1 ||
                    fake_dispatch_calls != 2 ||
                    !fake_contract_valid) {
                    return 8;
                }
                return 0;
            }
            """)
        .replace("EXECUTE_SYMBOL", execute_symbol)
        .replace("UNIT_SYMBOL", unit_symbol)
        .replace(
            "MISALIGNED_HASH",
            hashlib.sha256(misaligned_binary).hexdigest(),
        )
        .lstrip(),
        encoding="utf-8",
    )

    executable_path = tmp_path / (
        "native_opengl_spirv_test.exe"
        if os.name == "nt"
        else "native_opengl_spirv_test"
    )
    compile_result = subprocess.run(
        _compile_command(compiler, source_path, executable_path),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr

    run_result = subprocess.run(
        [str(executable_path), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, (
        f"exit={run_result.returncode}\n" f"{run_result.stdout}{run_result.stderr}"
    )
