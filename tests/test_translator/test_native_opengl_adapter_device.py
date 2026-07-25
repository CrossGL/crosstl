import hashlib
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import crosstl
from crosstl.project import (
    generate_native_loader_execution_abi,
    generate_native_loader_target_adapter,
)

_RUN_DEVICE_TEST = "CROSSTL_RUN_NATIVE_OPENGL_ADAPTER_DEVICE_TEST"
_SPECIALIZATION_ID = 7
_SPECIALIZATION_VALUE = 11
_ELEMENT_COUNT = 8
_CONTRACT_SPIRV = struct.pack("<5I", 0x07230203, 0x00010000, 0, 1, 0)
_SOURCE_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "native_loader"
    / "opengl"
    / "specialized_output.metal"
)


def _required_executable(name):
    executable = shutil.which(name)
    if executable is None:
        pytest.fail(f"{name} is required when {_RUN_DEVICE_TEST}=1")
    return executable


def _compiler_command():
    configured = os.environ.get("CXX")
    candidates = [shlex.split(configured)] if configured else []
    candidates.extend([[name] for name in ("clang++", "g++", "c++")])
    for command in candidates:
        if command and shutil.which(command[0]):
            command[0] = shutil.which(command[0])
            return command
    pytest.fail(f"A C++17 compiler is required when {_RUN_DEVICE_TEST}=1")


def _pkg_config_flags():
    pkg_config = _required_executable("pkg-config")
    result = subprocess.run(
        [pkg_config, "--cflags", "--libs", "egl", "gl"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            "EGL and OpenGL development packages are required when "
            f"{_RUN_DEVICE_TEST}=1:\n{result.stderr}"
        )
    return shlex.split(result.stdout)


def _descriptor(spirv):
    return {
        "schemaVersion": 1,
        "kind": "crosstl-native-loader-abi-descriptor",
        "abiVersion": 1,
        "unitId": "native-opengl-device-test",
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
                "value": hashlib.sha256(spirv).hexdigest(),
            },
            "sizeBytes": len(spirv),
        },
        "source": {
            "path": "specialized_output.metal",
            "artifactPath": None,
            "backend": "metal",
            "hash": None,
            "remap": None,
        },
        "bindings": [
            {
                "name": "input_values",
                "kind": "buffer",
                "type": "uint[]",
                "namespace": "storage-buffer",
                "coordinates": {"set": 0, "binding": 0},
                "access": "read",
                "scalarLayout": None,
                "provenance": {},
            },
            {
                "name": "output_values",
                "kind": "buffer",
                "type": "uint[]",
                "namespace": "storage-buffer",
                "coordinates": {"set": 0, "binding": 1},
                "access": "write",
                "scalarLayout": None,
                "provenance": {},
            },
        ],
        "scalarLayout": {"constants": [], "bindings": []},
        "specializationConstants": [
            {
                "id": _SPECIALIZATION_ID,
                "name": "multiplier",
                "dtype": "uint32",
            }
        ],
        "provenance": {},
    }


def _translate_fixture():
    return crosstl.translate(
        str(_SOURCE_FIXTURE),
        backend="opengl",
        format_output=False,
        source_backend="metal",
    )


def _generate_contract(spirv):
    descriptor = _descriptor(spirv)
    return (
        descriptor,
        generate_native_loader_execution_abi(descriptor),
        generate_native_loader_target_adapter("opengl"),
    )


def _generated_symbols(execution_header):
    execute = re.search(
        r"static inline CrossTLNativeLoaderExecutionResult\s+"
        r"([A-Za-z_]\w*_execute)\(",
        execution_header,
    )
    if execute is None:
        pytest.fail("The generated execution function was not found")
    return execute.group(1)


def _harness_source(execute_symbol):
    return textwrap.dedent(f"""\
        #include "native_loader_execution.h"
        #include "native_opengl_adapter.h"

        #include <EGL/egl.h>
        #include <EGL/eglext.h>

        #include <cstdint>
        #include <cstring>
        #include <iostream>
        #include <stdexcept>
        #include <string>

        namespace {{

        class EglContext {{
          public:
            EglContext() {{
                PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
                    load_egl<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
                        "eglGetPlatformDisplayEXT");
                display_ = get_platform_display(
                    EGL_PLATFORM_SURFACELESS_MESA,
                    EGL_DEFAULT_DISPLAY,
                    nullptr);
                require(display_ != EGL_NO_DISPLAY, "surfaceless EGL display");

                EGLint major = 0;
                EGLint minor = 0;
                require(
                    eglInitialize(display_, &major, &minor) == EGL_TRUE,
                    "eglInitialize");
                require(eglBindAPI(EGL_OPENGL_API) == EGL_TRUE, "eglBindAPI");

                const EGLint config_attributes[] = {{
                    EGL_RENDERABLE_TYPE,
                    EGL_OPENGL_BIT,
                    EGL_SURFACE_TYPE,
                    EGL_PBUFFER_BIT,
                    EGL_NONE}};
                EGLint config_count = 0;
                require(
                    eglChooseConfig(
                        display_,
                        config_attributes,
                        &config_,
                        1,
                        &config_count) == EGL_TRUE &&
                        config_count == 1,
                    "eglChooseConfig");

                const EGLint context_attributes[] = {{
                    EGL_CONTEXT_MAJOR_VERSION_KHR,
                    4,
                    EGL_CONTEXT_MINOR_VERSION_KHR,
                    3,
                    EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR,
                    EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR,
                    EGL_NONE}};
                context_ = eglCreateContext(
                    display_,
                    config_,
                    EGL_NO_CONTEXT,
                    context_attributes);
                require(context_ != EGL_NO_CONTEXT, "eglCreateContext");
                require(
                    eglMakeCurrent(
                        display_,
                        EGL_NO_SURFACE,
                        EGL_NO_SURFACE,
                        context_) == EGL_TRUE,
                    "eglMakeCurrent");
            }}

            EglContext(const EglContext &) = delete;
            EglContext &operator=(const EglContext &) = delete;

            ~EglContext() {{
                if (display_ != EGL_NO_DISPLAY) {{
                    eglMakeCurrent(
                        display_,
                        EGL_NO_SURFACE,
                        EGL_NO_SURFACE,
                        EGL_NO_CONTEXT);
                    if (context_ != EGL_NO_CONTEXT) {{
                        eglDestroyContext(display_, context_);
                    }}
                    eglTerminate(display_);
                }}
            }}

          private:
            template <typename Function>
            static Function load_egl(const char *name) {{
                __eglMustCastToProperFunctionPointerType address =
                    eglGetProcAddress(name);
                require(address != nullptr, name);
                Function function = nullptr;
                static_assert(sizeof(function) == sizeof(address));
                std::memcpy(&function, &address, sizeof(function));
                return function;
            }}

            static void require(bool condition, const char *operation) {{
                if (!condition) {{
                    throw std::runtime_error(
                        std::string(operation) +
                        " failed with EGL error " +
                        std::to_string(eglGetError()));
                }}
            }}

            EGLDisplay display_ = EGL_NO_DISPLAY;
            EGLConfig config_ = nullptr;
            EGLContext context_ = EGL_NO_CONTEXT;
        }};

        template <typename Function>
        Function load_gl(const char *name, bool required = true) {{
            __eglMustCastToProperFunctionPointerType address =
                eglGetProcAddress(name);
            if (address == nullptr) {{
                if (required) {{
                    throw std::runtime_error(
                        std::string("OpenGL entry point is unavailable: ") +
                        name);
                }}
                return nullptr;
            }}
            Function function = nullptr;
            static_assert(sizeof(function) == sizeof(address));
            std::memcpy(&function, &address, sizeof(function));
            return function;
        }}

        CrossTLOpenGL43Functions load_function_table() {{
            CrossTLOpenGL43Functions functions = {{}};
            functions.get_error =
                load_gl<CrossTLOpenGL43GetErrorFunction>("glGetError");
            functions.get_integerv =
                load_gl<CrossTLOpenGL43GetIntegervFunction>("glGetIntegerv");
            functions.get_integer_indexed =
                load_gl<CrossTLOpenGL43GetIntegerIndexedFunction>(
                    "glGetIntegeri_v");
            functions.create_shader =
                load_gl<CrossTLOpenGL43CreateShaderFunction>("glCreateShader");
            functions.shader_source =
                load_gl<CrossTLOpenGL43ShaderSourceFunction>("glShaderSource");
            functions.compile_shader =
                load_gl<CrossTLOpenGL43CompileShaderFunction>("glCompileShader");
            functions.shader_binary =
                load_gl<CrossTLOpenGL43ShaderBinaryFunction>("glShaderBinary");
            functions.specialize_shader =
                load_gl<CrossTLOpenGL43SpecializeShaderFunction>(
                    "glSpecializeShader",
                    false);
            functions.specialize_shader_arb =
                load_gl<CrossTLOpenGL43SpecializeShaderFunction>(
                    "glSpecializeShaderARB",
                    false);
            functions.get_shader_iv =
                load_gl<CrossTLOpenGL43GetShaderivFunction>("glGetShaderiv");
            functions.get_shader_info_log =
                load_gl<CrossTLOpenGL43GetShaderInfoLogFunction>(
                    "glGetShaderInfoLog");
            functions.delete_shader =
                load_gl<CrossTLOpenGL43DeleteShaderFunction>("glDeleteShader");
            functions.create_program =
                load_gl<CrossTLOpenGL43CreateProgramFunction>("glCreateProgram");
            functions.attach_shader =
                load_gl<CrossTLOpenGL43AttachShaderFunction>("glAttachShader");
            functions.link_program =
                load_gl<CrossTLOpenGL43LinkProgramFunction>("glLinkProgram");
            functions.get_program_iv =
                load_gl<CrossTLOpenGL43GetProgramivFunction>("glGetProgramiv");
            functions.get_program_info_log =
                load_gl<CrossTLOpenGL43GetProgramInfoLogFunction>(
                    "glGetProgramInfoLog");
            functions.delete_program =
                load_gl<CrossTLOpenGL43DeleteProgramFunction>("glDeleteProgram");
            functions.use_program =
                load_gl<CrossTLOpenGL43UseProgramFunction>("glUseProgram");
            functions.gen_buffers =
                load_gl<CrossTLOpenGL43GenBuffersFunction>("glGenBuffers");
            functions.bind_buffer =
                load_gl<CrossTLOpenGL43BindBufferFunction>("glBindBuffer");
            functions.buffer_data =
                load_gl<CrossTLOpenGL43BufferDataFunction>("glBufferData");
            functions.bind_buffer_base =
                load_gl<CrossTLOpenGL43BindBufferBaseFunction>(
                    "glBindBufferBase");
            functions.delete_buffers =
                load_gl<CrossTLOpenGL43DeleteBuffersFunction>("glDeleteBuffers");
            functions.dispatch_compute =
                load_gl<CrossTLOpenGL43DispatchComputeFunction>(
                    "glDispatchCompute");
            functions.memory_barrier =
                load_gl<CrossTLOpenGL43MemoryBarrierFunction>("glMemoryBarrier");
            functions.finish =
                load_gl<CrossTLOpenGL43FinishFunction>("glFinish");
            functions.get_buffer_sub_data =
                load_gl<CrossTLOpenGL43GetBufferSubDataFunction>(
                    "glGetBufferSubData");
            return functions;
        }}

        bool has_extension(const char *expected) {{
            using GetStringiFunction =
                const unsigned char *(*)(unsigned int, unsigned int);
            GetStringiFunction get_string_indexed =
                load_gl<GetStringiFunction>("glGetStringi");
            CrossTLOpenGL43GetIntegervFunction get_integer =
                load_gl<CrossTLOpenGL43GetIntegervFunction>("glGetIntegerv");
            constexpr unsigned int extension_name = 0x1F03u;
            constexpr unsigned int extension_count = 0x821Du;
            CrossTLOpenGL43Int count = 0;
            get_integer(extension_count, &count);
            for (CrossTLOpenGL43Int index = 0; index < count; ++index) {{
                const unsigned char *name = get_string_indexed(
                    extension_name,
                    static_cast<unsigned int>(index));
                if (name != nullptr &&
                    std::strcmp(
                        reinterpret_cast<const char *>(name),
                        expected) == 0) {{
                    return true;
                }}
            }}
            return false;
        }}

        int32_t context_is_current(void *) {{
            return eglGetCurrentContext() != EGL_NO_CONTEXT ? 1 : 0;
        }}

        void report_diagnostic(
            void *opaque,
            int32_t status,
            const char *phase,
            const char *message) {{
            std::string *diagnostic = static_cast<std::string *>(opaque);
            *diagnostic =
                std::string(phase != nullptr ? phase : "") + ": " +
                (message != nullptr ? message : "") + " (" +
                std::to_string(status) + ")";
        }}

        void require_execution(
            const CrossTLNativeLoaderExecutionResult &result,
            const std::string &diagnostic) {{
            if (result.succeeded == 0) {{
                throw std::runtime_error(
                    "native execution failed in phase " +
                    std::to_string(result.error.phase) +
                    " with code " + std::to_string(result.error.code) +
                    ", adapter status " +
                    std::to_string(result.error.adapter_status) + ": " +
                    diagnostic);
            }}
            if (result.cleanup_error.code !=
                CROSSTL_NATIVE_LOADER_CODE_OK) {{
                throw std::runtime_error(
                    "native cleanup failed with code " +
                    std::to_string(result.cleanup_error.code));
            }}
        }}

        }}  // namespace

        int main(int argc, char **argv) {{
            try {{
                if (argc != 2) {{
                    throw std::runtime_error("expected artifact root argument");
                }}
                EglContext egl_context;
                CrossTLOpenGL43Functions functions = load_function_table();

                CrossTLOpenGL43Int major = 0;
                CrossTLOpenGL43Int minor = 0;
                functions.get_integerv(
                    CROSSTL_OPENGL43_MAJOR_VERSION,
                    &major);
                functions.get_integerv(
                    CROSSTL_OPENGL43_MINOR_VERSION,
                    &minor);
                const bool core_spirv =
                    major > 4 || (major == 4 && minor >= 6);
                const bool arb_spirv = has_extension("GL_ARB_gl_spirv");
                if (!core_spirv && !arb_spirv) {{
                    throw std::runtime_error(
                        "OpenGL SPIR-V specialization is unavailable");
                }}
                if (core_spirv && functions.specialize_shader == nullptr) {{
                    throw std::runtime_error(
                        "glSpecializeShader is unavailable");
                }}
                if (!core_spirv &&
                    (functions.specialize_shader_arb == nullptr ||
                     !arb_spirv)) {{
                    throw std::runtime_error(
                        "GL_ARB_gl_spirv entry points are unavailable");
                }}

                std::string diagnostic;
                CrossTLOpenGL43Context adapter_context = {{}};
                const int32_t initialize_status =
                    crosstl_opengl43_initialize_context(
                        &adapter_context,
                        &functions,
                        &diagnostic,
                        context_is_current);
                if (initialize_status != CROSSTL_OPENGL43_STATUS_OK) {{
                    throw std::runtime_error(
                        "adapter context initialization failed");
                }}
                char diagnostic_buffer[1024] = {{}};
                adapter_context.artifact_root = argv[1];
                adapter_context.report_diagnostic = report_diagnostic;
                adapter_context.diagnostic_buffer = diagnostic_buffer;
                adapter_context.diagnostic_buffer_capacity =
                    sizeof(diagnostic_buffer);
                adapter_context.supports_arb_gl_spirv =
                    arb_spirv ? 1u : 0u;
                CrossTLNativeLoaderAdapter adapter =
                    crosstl_opengl43_make_adapter(&adapter_context);

                std::uint32_t input_values[{_ELEMENT_COUNT}] = {{
                    2u, 5u, 9u, 14u, 23u, 31u, 47u, 61u}};
                std::uint32_t output_values[{_ELEMENT_COUNT}] = {{}};
                CrossTLNativeLoaderBindingRequest bindings[2] = {{
                    {{
                        "input_values",
                        "buffer",
                        "uint[]",
                        "storage-buffer",
                        0u,
                        0u,
                        CROSSTL_NATIVE_LOADER_ACCESS_READ,
                        input_values,
                        sizeof(input_values)}},
                    {{
                        "output_values",
                        "buffer",
                        "uint[]",
                        "storage-buffer",
                        0u,
                        1u,
                        CROSSTL_NATIVE_LOADER_ACCESS_WRITE,
                        output_values,
                        sizeof(output_values)}}}};
                const std::uint32_t multiplier =
                    {_SPECIALIZATION_VALUE}u;
                CrossTLNativeLoaderSpecializationRequest specialization = {{
                    1u,
                    {_SPECIALIZATION_ID}u,
                    "multiplier",
                    "uint32",
                    &multiplier,
                    sizeof(multiplier)}};
                CrossTLNativeLoaderDispatchGeometry dispatch = {{}};
                dispatch.workgroup_count[0] = {_ELEMENT_COUNT}u;
                dispatch.workgroup_count[1] = 1u;
                dispatch.workgroup_count[2] = 1u;
                dispatch.workgroup_size[0] = 1u;
                dispatch.workgroup_size[1] = 1u;
                dispatch.workgroup_size[2] = 1u;
                CrossTLNativeLoaderExecutionRequest request = {{
                    CROSSTL_NATIVE_LOADER_ABI_VERSION,
                    "opengl",
                    2u,
                    bindings,
                    1u,
                    &specialization,
                    dispatch}};

                CrossTLNativeLoaderExecutionResult result =
                    {execute_symbol}(&request, &adapter);
                require_execution(result, diagnostic);

                for (std::uint32_t index = 0u;
                     index < {_ELEMENT_COUNT}u;
                    ++index) {{
                    const std::uint32_t expected =
                        input_values[index] * {_SPECIALIZATION_VALUE}u;
                    const std::uint32_t unspecialized =
                        input_values[index] * 3u;
                    if (output_values[index] != expected ||
                        output_values[index] == unspecialized) {{
                        throw std::runtime_error(
                            "unexpected output at index " +
                            std::to_string(index) + ": " +
                            std::to_string(output_values[index]));
                    }}
                }}
                std::cout << "OpenGL numerical adapter execution passed"
                          << std::endl;
                return 0;
            }} catch (const std::exception &error) {{
                std::cerr << error.what() << std::endl;
                return 1;
            }}
        }}
        """)


def test_generated_opengl_adapter_contract_from_translated_fixture():
    translated_glsl = _translate_fixture()

    assert "layout(constant_id = 7) const uint multiplier = 3u;" in translated_glsl
    assert "output_values[index] = (input_values[index] * multiplier);" in (
        translated_glsl
    )
    assert (
        "layout(std430, binding = 0) readonly buffer input_valuesBuffer"
        in translated_glsl
    )
    assert "layout(std430, binding = 1) buffer output_valuesBuffer" in translated_glsl

    descriptor, execution_header, adapter_header = _generate_contract(_CONTRACT_SPIRV)
    expected_hash = hashlib.sha256(_CONTRACT_SPIRV).hexdigest()
    assert descriptor["artifact"]["hash"] == {
        "algorithm": "sha256",
        "value": expected_hash,
    }
    assert descriptor["artifact"]["sizeBytes"] == len(_CONTRACT_SPIRV)
    assert '"kernel.spv"' in execution_header
    assert '"SPIR-V binary"' in execution_header
    assert '"sha256"' in execution_header
    assert f'"{expected_hash}"' in execution_header
    assert _generated_symbols(execution_header).endswith("_execute")
    assert "crosstl_opengl43_make_adapter" in adapter_header


def test_generated_opengl_adapter_executes_specialized_spirv_on_device(tmp_path):
    if os.environ.get(_RUN_DEVICE_TEST) != "1":
        pytest.skip(f"Set {_RUN_DEVICE_TEST}=1 to run the native OpenGL device test")
    if not sys.platform.startswith("linux"):
        pytest.fail(f"{_RUN_DEVICE_TEST}=1 requires Linux")

    glslang = _required_executable("glslangValidator")
    compiler = _compiler_command()
    linker_flags = _pkg_config_flags()

    shader_path = tmp_path / "kernel.comp"
    spirv_path = tmp_path / "kernel.spv"
    translated_glsl = _translate_fixture()
    shader_path.write_text(translated_glsl, encoding="utf-8")
    subprocess.run(
        [
            glslang,
            "-G",
            "--target-env",
            "opengl",
            "-S",
            "comp",
            "-e",
            "main",
            "-o",
            str(spirv_path),
            str(shader_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    spirv = spirv_path.read_bytes()
    _descriptor_payload, execution_header, adapter_header = _generate_contract(spirv)

    (tmp_path / "native_loader_execution.h").write_text(
        execution_header, encoding="utf-8"
    )
    (tmp_path / "native_opengl_adapter.h").write_text(adapter_header, encoding="utf-8")
    source_path = tmp_path / "native_opengl_adapter_device.cpp"
    source_path.write_text(
        _harness_source(_generated_symbols(execution_header)),
        encoding="utf-8",
    )
    executable_path = tmp_path / "native_opengl_adapter_device"
    compile_result = subprocess.run(
        compiler
        + [
            "-std=c++17",
            "-O2",
            "-pedantic-errors",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source_path),
            "-o",
            str(executable_path),
        ]
        + linker_flags,
        check=False,
        capture_output=True,
        text=True,
    )
    if compile_result.returncode != 0:
        pytest.fail(
            "The native OpenGL device harness did not compile:\n"
            f"{compile_result.stdout}\n{compile_result.stderr}"
        )

    environment = os.environ.copy()
    environment.update(
        {
            "EGL_PLATFORM": "surfaceless",
            "LIBGL_ALWAYS_SOFTWARE": "true",
        }
    )
    execution_result = subprocess.run(
        [str(executable_path), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if execution_result.returncode != 0:
        pytest.fail(
            "The native OpenGL numerical device test failed:\n"
            f"{execution_result.stdout}\n{execution_result.stderr}"
        )
    assert "OpenGL numerical adapter execution passed" in execution_result.stdout
