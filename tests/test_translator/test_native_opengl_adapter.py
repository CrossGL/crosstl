import os
import re
import shlex
import shutil
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
            "hash": {"algorithm": "sha256", "value": "0" * 64},
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
                if (adapter.apply_specialization(
                        adapter.context, NULL, NULL, NULL) !=
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
                    return 9;
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
                    return 10;
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
                    return 11;
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
                    return 12;
                }
                const float expected[4] = {2.0f, 3.0f, 5.0f, 9.0f};
                for (size_t index = 0u; index < 4u; ++index) {
                    if (output_values[index] != expected[index]) {
                        return 13;
                    }
                }
                return 0;
            }
            """).replace("EXECUTE_SYMBOL", execute_symbol).lstrip(),
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
