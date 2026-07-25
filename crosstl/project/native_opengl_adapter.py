"""Generate the OpenGL 4.3 native loader execution adapter."""

from __future__ import annotations

import textwrap


def generate_opengl_native_loader_adapter() -> str:
    """Render the deterministic C++17 OpenGL 4.3 adapter header."""

    return textwrap.dedent(r"""
        #ifndef CROSSTL_NATIVE_LOADER_OPENGL43_ADAPTER_H
        #define CROSSTL_NATIVE_LOADER_OPENGL43_ADAPTER_H

        /*
         * Include a generated CrossTL native loader execution header first.
         *
         * The caller owns the OpenGL context and must keep an OpenGL 4.3 or
         * newer desktop context current on the calling thread for the complete
         * execution lifecycle. Populate CrossTLOpenGL43Functions through EGL,
         * GLX, epoxy, GLAD, or another context-compatible loader. Calls using
         * one CrossTLOpenGL43Context must be serialized. This adapter does not
         * create, destroy, or present a platform context and it consumes the
         * current OpenGL error state while reporting adapter diagnostics. It
         * restores the current program, generic buffer bindings, and indexed
         * buffer bindings changed by execution.
         *
         * The adapter accepts GLSL compute source with entry point "main",
         * OpenGL SPIR-V compute binaries, set-zero shader-storage buffers, and
         * set-zero uniform buffers. SPIR-V requires OpenGL 4.6 core support or
         * a caller-confirmed GL_ARB_gl_spirv capability plus matching loaded
         * entry points. Source specialization fails closed; binary
         * specialization uses glShaderBinary and glSpecializeShader or
         * glSpecializeShaderARB without source rewriting.
         *
         * Artifact bytes are checked against the descriptor's SHA-256 before
         * any OpenGL compilation callback. When artifact_root is set,
         * package paths must remain inside its canonical filesystem tree.
         * Texture, image, sampler, scalar-uniform, shared-allocation, and
         * nonzero descriptor-set contracts return an explicit non-success
         * status.
         */

        #ifndef __cplusplus
        #error "The CrossTL OpenGL 4.3 native loader adapter requires C++17"
        #endif

        #if defined(_MSVC_LANG)
        #if _MSVC_LANG < 201703L
        #error "The CrossTL OpenGL 4.3 native loader adapter requires C++17"
        #endif
        #elif __cplusplus < 201703L
        #error "The CrossTL OpenGL 4.3 native loader adapter requires C++17"
        #endif

        #ifndef CROSSTL_NATIVE_LOADER_EXECUTION_ABI_V1_TYPES
        #error "Include a generated CrossTL native loader execution header first"
        #endif

        #include <cmath>
        #include <cstddef>
        #include <cstdint>
        #include <cstdio>
        #include <cstring>
        #include <filesystem>
        #include <fstream>
        #include <limits>
        #include <new>
        #include <string>
        #include <vector>

        #ifndef CROSSTL_NATIVE_LOADER_TARGET_ADAPTER_VERSION
        #define CROSSTL_NATIVE_LOADER_TARGET_ADAPTER_VERSION 1u
        #elif CROSSTL_NATIVE_LOADER_TARGET_ADAPTER_VERSION != 1u
        #error "Conflicting CrossTL native loader target adapter version"
        #endif

        #if defined(_WIN32)
        #define CROSSTL_OPENGL43_APIENTRY __stdcall
        #else
        #define CROSSTL_OPENGL43_APIENTRY
        #endif

        typedef std::uint32_t CrossTLOpenGL43Enum;
        typedef std::uint32_t CrossTLOpenGL43Bitfield;
        typedef std::uint32_t CrossTLOpenGL43UInt;
        typedef std::int32_t CrossTLOpenGL43Int;
        typedef std::int32_t CrossTLOpenGL43Sizei;
        typedef std::ptrdiff_t CrossTLOpenGL43SizePtr;
        typedef char CrossTLOpenGL43Char;

        enum CrossTLOpenGL43Token : CrossTLOpenGL43Enum {
            CROSSTL_OPENGL43_NO_ERROR = 0x0000u,
            CROSSTL_OPENGL43_MAJOR_VERSION = 0x821Bu,
            CROSSTL_OPENGL43_MINOR_VERSION = 0x821Cu,
            CROSSTL_OPENGL43_COMPUTE_SHADER = 0x91B9u,
            CROSSTL_OPENGL43_SHADER_BINARY_FORMAT_SPIR_V = 0x9551u,
            CROSSTL_OPENGL43_COMPILE_STATUS = 0x8B81u,
            CROSSTL_OPENGL43_LINK_STATUS = 0x8B82u,
            CROSSTL_OPENGL43_INFO_LOG_LENGTH = 0x8B84u,
            CROSSTL_OPENGL43_CURRENT_PROGRAM = 0x8B8Du,
            CROSSTL_OPENGL43_SHADER_STORAGE_BUFFER = 0x90D2u,
            CROSSTL_OPENGL43_SHADER_STORAGE_BUFFER_BINDING = 0x90D3u,
            CROSSTL_OPENGL43_UNIFORM_BUFFER = 0x8A11u,
            CROSSTL_OPENGL43_UNIFORM_BUFFER_BINDING = 0x8A28u,
            CROSSTL_OPENGL43_DYNAMIC_COPY = 0x88EAu,
            CROSSTL_OPENGL43_UNIFORM_BARRIER_BIT = 0x00000004u,
            CROSSTL_OPENGL43_BUFFER_UPDATE_BARRIER_BIT = 0x00000200u,
            CROSSTL_OPENGL43_SHADER_STORAGE_BARRIER_BIT = 0x00002000u,
            CROSSTL_OPENGL43_MAX_COMPUTE_WORK_GROUP_COUNT = 0x91BEu
        };

        typedef enum CrossTLOpenGL43Status {
            CROSSTL_OPENGL43_STATUS_OK = 0,
            CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT = 1,
            CROSSTL_OPENGL43_STATUS_ABI_VERSION_MISMATCH = 2,
            CROSSTL_OPENGL43_STATUS_CONTEXT_CALLBACK_MISSING = 3,
            CROSSTL_OPENGL43_STATUS_CONTEXT_NOT_CURRENT = 4,
            CROSSTL_OPENGL43_STATUS_OPENGL_VERSION_UNSUPPORTED = 5,
            CROSSTL_OPENGL43_STATUS_ENTRY_POINT_MISSING = 6,
            CROSSTL_OPENGL43_STATUS_ARTIFACT_FORMAT_UNSUPPORTED = 7,
            CROSSTL_OPENGL43_STATUS_ARTIFACT_READ_FAILED = 8,
            CROSSTL_OPENGL43_STATUS_ARTIFACT_SIZE_MISMATCH = 9,
            CROSSTL_OPENGL43_STATUS_ARTIFACT_EMPTY = 10,
            CROSSTL_OPENGL43_STATUS_STAGE_UNSUPPORTED = 11,
            CROSSTL_OPENGL43_STATUS_ENTRY_POINT_UNSUPPORTED = 12,
            CROSSTL_OPENGL43_STATUS_SPECIALIZATION_UNSUPPORTED = 13,
            CROSSTL_OPENGL43_STATUS_RESOURCE_SET_UNSUPPORTED = 14,
            CROSSTL_OPENGL43_STATUS_RESOURCE_KIND_UNSUPPORTED = 15,
            CROSSTL_OPENGL43_STATUS_RESOURCE_ACCESS_UNSUPPORTED = 16,
            CROSSTL_OPENGL43_STATUS_RESOURCE_SIZE_UNSUPPORTED = 17,
            CROSSTL_OPENGL43_STATUS_OUT_OF_MEMORY = 18,
            CROSSTL_OPENGL43_STATUS_SHADER_CREATE_FAILED = 19,
            CROSSTL_OPENGL43_STATUS_SHADER_COMPILE_FAILED = 20,
            CROSSTL_OPENGL43_STATUS_PROGRAM_CREATE_FAILED = 21,
            CROSSTL_OPENGL43_STATUS_PROGRAM_LINK_FAILED = 22,
            CROSSTL_OPENGL43_STATUS_BUFFER_CREATE_FAILED = 23,
            CROSSTL_OPENGL43_STATUS_OPENGL_ERROR = 24,
            CROSSTL_OPENGL43_STATUS_DISPATCH_LIMIT_EXCEEDED = 25,
            CROSSTL_OPENGL43_STATUS_DISPATCH_REQUIRED = 26,
            CROSSTL_OPENGL43_STATUS_SYNCHRONIZATION_REQUIRED = 27,
            CROSSTL_OPENGL43_STATUS_CONTEXT_STRUCTURE_INVALID = 28,
            CROSSTL_OPENGL43_STATUS_TARGET_MISMATCH = 29,
            CROSSTL_OPENGL43_STATUS_INTERNAL_EXCEPTION = 30,
            CROSSTL_OPENGL43_STATUS_RESOURCE_SIZE_MISMATCH = 31,
            CROSSTL_OPENGL43_STATUS_ARTIFACT_SIZE_UNSUPPORTED = 32,
            CROSSTL_OPENGL43_STATUS_HASH_ALGORITHM_UNSUPPORTED = 33,
            CROSSTL_OPENGL43_STATUS_HASH_FAILED = 34,
            CROSSTL_OPENGL43_STATUS_HASH_MISMATCH = 35,
            CROSSTL_OPENGL43_STATUS_ARTIFACT_PATH_UNSAFE = 36,
            CROSSTL_OPENGL43_STATUS_ARTIFACT_PATH_RESOLUTION_FAILED = 37,
            CROSSTL_OPENGL43_STATUS_SPIRV_LAYOUT_INVALID = 38,
            CROSSTL_OPENGL43_STATUS_SPIRV_MAGIC_INVALID = 39,
            CROSSTL_OPENGL43_STATUS_SPIRV_CAPABILITY_UNSUPPORTED = 40,
            CROSSTL_OPENGL43_STATUS_SPIRV_ENTRY_POINT_MISSING = 41,
            CROSSTL_OPENGL43_STATUS_SPECIALIZATION_ID_REQUIRED = 42,
            CROSSTL_OPENGL43_STATUS_SPECIALIZATION_TYPE_UNSUPPORTED = 43,
            CROSSTL_OPENGL43_STATUS_SPECIALIZATION_PAYLOAD_INVALID = 44,
            CROSSTL_OPENGL43_STATUS_SPECIALIZATION_ID_DUPLICATE = 45,
            CROSSTL_OPENGL43_STATUS_SPIRV_BINARY_LOAD_FAILED = 46,
            CROSSTL_OPENGL43_STATUS_SPIRV_SPECIALIZATION_FAILED = 47
        } CrossTLOpenGL43Status;

        typedef CrossTLOpenGL43Enum(CROSSTL_OPENGL43_APIENTRY
                                        *CrossTLOpenGL43GetErrorFunction)(void);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43GetIntegervFunction)(
            CrossTLOpenGL43Enum, CrossTLOpenGL43Int *);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43GetIntegerIndexedFunction)(
            CrossTLOpenGL43Enum, CrossTLOpenGL43UInt, CrossTLOpenGL43Int *);
        typedef CrossTLOpenGL43UInt(CROSSTL_OPENGL43_APIENTRY
                                       *CrossTLOpenGL43CreateShaderFunction)(
            CrossTLOpenGL43Enum);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43ShaderSourceFunction)(
            CrossTLOpenGL43UInt,
            CrossTLOpenGL43Sizei,
            const CrossTLOpenGL43Char *const *,
            const CrossTLOpenGL43Int *);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43CompileShaderFunction)(
            CrossTLOpenGL43UInt);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43ShaderBinaryFunction)(
            CrossTLOpenGL43Sizei,
            const CrossTLOpenGL43UInt *,
            CrossTLOpenGL43Enum,
            const void *,
            CrossTLOpenGL43Sizei);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43SpecializeShaderFunction)(
            CrossTLOpenGL43UInt,
            const CrossTLOpenGL43Char *,
            CrossTLOpenGL43UInt,
            const CrossTLOpenGL43UInt *,
            const CrossTLOpenGL43UInt *);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43GetShaderivFunction)(
            CrossTLOpenGL43UInt, CrossTLOpenGL43Enum, CrossTLOpenGL43Int *);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43GetShaderInfoLogFunction)(
            CrossTLOpenGL43UInt,
            CrossTLOpenGL43Sizei,
            CrossTLOpenGL43Sizei *,
            CrossTLOpenGL43Char *);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43DeleteShaderFunction)(
            CrossTLOpenGL43UInt);
        typedef CrossTLOpenGL43UInt(CROSSTL_OPENGL43_APIENTRY
                                       *CrossTLOpenGL43CreateProgramFunction)(
            void);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43AttachShaderFunction)(
            CrossTLOpenGL43UInt, CrossTLOpenGL43UInt);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43LinkProgramFunction)(
            CrossTLOpenGL43UInt);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43GetProgramivFunction)(
            CrossTLOpenGL43UInt, CrossTLOpenGL43Enum, CrossTLOpenGL43Int *);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43GetProgramInfoLogFunction)(
            CrossTLOpenGL43UInt,
            CrossTLOpenGL43Sizei,
            CrossTLOpenGL43Sizei *,
            CrossTLOpenGL43Char *);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43DeleteProgramFunction)(
            CrossTLOpenGL43UInt);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43UseProgramFunction)(
            CrossTLOpenGL43UInt);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43GenBuffersFunction)(
            CrossTLOpenGL43Sizei, CrossTLOpenGL43UInt *);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43BindBufferFunction)(
            CrossTLOpenGL43Enum, CrossTLOpenGL43UInt);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43BufferDataFunction)(
            CrossTLOpenGL43Enum,
            CrossTLOpenGL43SizePtr,
            const void *,
            CrossTLOpenGL43Enum);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43BindBufferBaseFunction)(
            CrossTLOpenGL43Enum,
            CrossTLOpenGL43UInt,
            CrossTLOpenGL43UInt);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43DeleteBuffersFunction)(
            CrossTLOpenGL43Sizei, const CrossTLOpenGL43UInt *);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43DispatchComputeFunction)(
            CrossTLOpenGL43UInt,
            CrossTLOpenGL43UInt,
            CrossTLOpenGL43UInt);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43MemoryBarrierFunction)(
            CrossTLOpenGL43Bitfield);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43FinishFunction)(void);
        typedef void(CROSSTL_OPENGL43_APIENTRY
                         *CrossTLOpenGL43GetBufferSubDataFunction)(
            CrossTLOpenGL43Enum,
            CrossTLOpenGL43SizePtr,
            CrossTLOpenGL43SizePtr,
            void *);

        typedef struct CrossTLOpenGL43Functions {
            CrossTLOpenGL43GetErrorFunction get_error;
            CrossTLOpenGL43GetIntegervFunction get_integerv;
            CrossTLOpenGL43GetIntegerIndexedFunction get_integer_indexed;
            CrossTLOpenGL43CreateShaderFunction create_shader;
            CrossTLOpenGL43ShaderSourceFunction shader_source;
            CrossTLOpenGL43CompileShaderFunction compile_shader;
            CrossTLOpenGL43ShaderBinaryFunction shader_binary;
            CrossTLOpenGL43SpecializeShaderFunction specialize_shader;
            CrossTLOpenGL43SpecializeShaderFunction specialize_shader_arb;
            CrossTLOpenGL43GetShaderivFunction get_shader_iv;
            CrossTLOpenGL43GetShaderInfoLogFunction get_shader_info_log;
            CrossTLOpenGL43DeleteShaderFunction delete_shader;
            CrossTLOpenGL43CreateProgramFunction create_program;
            CrossTLOpenGL43AttachShaderFunction attach_shader;
            CrossTLOpenGL43LinkProgramFunction link_program;
            CrossTLOpenGL43GetProgramivFunction get_program_iv;
            CrossTLOpenGL43GetProgramInfoLogFunction get_program_info_log;
            CrossTLOpenGL43DeleteProgramFunction delete_program;
            CrossTLOpenGL43UseProgramFunction use_program;
            CrossTLOpenGL43GenBuffersFunction gen_buffers;
            CrossTLOpenGL43BindBufferFunction bind_buffer;
            CrossTLOpenGL43BufferDataFunction buffer_data;
            CrossTLOpenGL43BindBufferBaseFunction bind_buffer_base;
            CrossTLOpenGL43DeleteBuffersFunction delete_buffers;
            CrossTLOpenGL43DispatchComputeFunction dispatch_compute;
            CrossTLOpenGL43MemoryBarrierFunction memory_barrier;
            CrossTLOpenGL43FinishFunction finish;
            CrossTLOpenGL43GetBufferSubDataFunction get_buffer_sub_data;
        } CrossTLOpenGL43Functions;

        typedef int32_t (*CrossTLOpenGL43ContextCurrentFunction)(void *);
        typedef void (*CrossTLOpenGL43DiagnosticFunction)(
            void *, int32_t, const char *, const char *);

        typedef struct CrossTLOpenGL43Context {
            uint32_t adapter_version;
            size_t structure_size;
            CrossTLOpenGL43Functions functions;
            void *user_data;
            CrossTLOpenGL43ContextCurrentFunction is_context_current;
            CrossTLOpenGL43DiagnosticFunction report_diagnostic;
            const char *artifact_root;
            char *diagnostic_buffer;
            size_t diagnostic_buffer_capacity;
            uint32_t supports_arb_gl_spirv;
            uint64_t dispatch_serial;
            uint64_t synchronized_serial;
        } CrossTLOpenGL43Context;

        typedef enum CrossTLOpenGL43ArtifactKind {
            CROSSTL_OPENGL43_ARTIFACT_GLSL_SOURCE = 1,
            CROSSTL_OPENGL43_ARTIFACT_SPIRV_BINARY = 2
        } CrossTLOpenGL43ArtifactKind;

        typedef struct CrossTLOpenGL43Specialization {
            CrossTLOpenGL43UInt constant_id;
            CrossTLOpenGL43UInt encoded_value;
        } CrossTLOpenGL43Specialization;

        typedef struct CrossTLOpenGL43Artifact {
            CrossTLOpenGL43ArtifactKind kind;
            std::vector<uint8_t> bytes;
            std::vector<CrossTLOpenGL43Specialization> specializations;
        } CrossTLOpenGL43Artifact;

        typedef struct CrossTLOpenGL43Pipeline {
            CrossTLOpenGL43UInt shader;
            CrossTLOpenGL43UInt program;
            uint64_t dispatch_serial;
            int32_t dispatched;
        } CrossTLOpenGL43Pipeline;

        typedef struct CrossTLOpenGL43Resource {
            CrossTLOpenGL43Context *owner;
            CrossTLOpenGL43Enum target;
            CrossTLOpenGL43UInt buffer;
            CrossTLOpenGL43UInt binding;
            CrossTLOpenGL43UInt previous_indexed_buffer;
            size_t size_bytes;
        } CrossTLOpenGL43Resource;

        static inline const char *crosstl_opengl43_platform_requirements(void) {
            return "caller-owned current desktop OpenGL 4.3+ context; "
                   "context-compatible loaded core entry points; serialized "
                   "execution on the context-owning thread; OpenGL 4.6 or "
                   "GL_ARB_gl_spirv for SPIR-V specialization";
        }

        static inline const char *crosstl_opengl43_status_name(int32_t status) {
            switch (status) {
                case CROSSTL_OPENGL43_STATUS_OK:
                    return "ok";
                case CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT:
                    return "invalid-argument";
                case CROSSTL_OPENGL43_STATUS_ABI_VERSION_MISMATCH:
                    return "abi-version-mismatch";
                case CROSSTL_OPENGL43_STATUS_CONTEXT_CALLBACK_MISSING:
                    return "context-callback-missing";
                case CROSSTL_OPENGL43_STATUS_CONTEXT_NOT_CURRENT:
                    return "context-not-current";
                case CROSSTL_OPENGL43_STATUS_OPENGL_VERSION_UNSUPPORTED:
                    return "opengl-version-unsupported";
                case CROSSTL_OPENGL43_STATUS_ENTRY_POINT_MISSING:
                    return "entry-point-missing";
                case CROSSTL_OPENGL43_STATUS_ARTIFACT_FORMAT_UNSUPPORTED:
                    return "artifact-format-unsupported";
                case CROSSTL_OPENGL43_STATUS_ARTIFACT_READ_FAILED:
                    return "artifact-read-failed";
                case CROSSTL_OPENGL43_STATUS_ARTIFACT_SIZE_MISMATCH:
                    return "artifact-size-mismatch";
                case CROSSTL_OPENGL43_STATUS_ARTIFACT_EMPTY:
                    return "artifact-empty";
                case CROSSTL_OPENGL43_STATUS_STAGE_UNSUPPORTED:
                    return "stage-unsupported";
                case CROSSTL_OPENGL43_STATUS_ENTRY_POINT_UNSUPPORTED:
                    return "entry-point-unsupported";
                case CROSSTL_OPENGL43_STATUS_SPECIALIZATION_UNSUPPORTED:
                    return "specialization-unsupported";
                case CROSSTL_OPENGL43_STATUS_RESOURCE_SET_UNSUPPORTED:
                    return "resource-set-unsupported";
                case CROSSTL_OPENGL43_STATUS_RESOURCE_KIND_UNSUPPORTED:
                    return "resource-kind-unsupported";
                case CROSSTL_OPENGL43_STATUS_RESOURCE_ACCESS_UNSUPPORTED:
                    return "resource-access-unsupported";
                case CROSSTL_OPENGL43_STATUS_RESOURCE_SIZE_UNSUPPORTED:
                    return "resource-size-unsupported";
                case CROSSTL_OPENGL43_STATUS_OUT_OF_MEMORY:
                    return "out-of-memory";
                case CROSSTL_OPENGL43_STATUS_SHADER_CREATE_FAILED:
                    return "shader-create-failed";
                case CROSSTL_OPENGL43_STATUS_SHADER_COMPILE_FAILED:
                    return "shader-compile-failed";
                case CROSSTL_OPENGL43_STATUS_PROGRAM_CREATE_FAILED:
                    return "program-create-failed";
                case CROSSTL_OPENGL43_STATUS_PROGRAM_LINK_FAILED:
                    return "program-link-failed";
                case CROSSTL_OPENGL43_STATUS_BUFFER_CREATE_FAILED:
                    return "buffer-create-failed";
                case CROSSTL_OPENGL43_STATUS_OPENGL_ERROR:
                    return "opengl-error";
                case CROSSTL_OPENGL43_STATUS_DISPATCH_LIMIT_EXCEEDED:
                    return "dispatch-limit-exceeded";
                case CROSSTL_OPENGL43_STATUS_DISPATCH_REQUIRED:
                    return "dispatch-required";
                case CROSSTL_OPENGL43_STATUS_SYNCHRONIZATION_REQUIRED:
                    return "synchronization-required";
                case CROSSTL_OPENGL43_STATUS_CONTEXT_STRUCTURE_INVALID:
                    return "context-structure-invalid";
                case CROSSTL_OPENGL43_STATUS_TARGET_MISMATCH:
                    return "target-mismatch";
                case CROSSTL_OPENGL43_STATUS_INTERNAL_EXCEPTION:
                    return "internal-exception";
                case CROSSTL_OPENGL43_STATUS_RESOURCE_SIZE_MISMATCH:
                    return "resource-size-mismatch";
                case CROSSTL_OPENGL43_STATUS_ARTIFACT_SIZE_UNSUPPORTED:
                    return "artifact-size-unsupported";
                case CROSSTL_OPENGL43_STATUS_HASH_ALGORITHM_UNSUPPORTED:
                    return "hash-algorithm-unsupported";
                case CROSSTL_OPENGL43_STATUS_HASH_FAILED:
                    return "hash-failed";
                case CROSSTL_OPENGL43_STATUS_HASH_MISMATCH:
                    return "hash-mismatch";
                case CROSSTL_OPENGL43_STATUS_ARTIFACT_PATH_UNSAFE:
                    return "artifact-path-unsafe";
                case CROSSTL_OPENGL43_STATUS_ARTIFACT_PATH_RESOLUTION_FAILED:
                    return "artifact-path-resolution-failed";
                case CROSSTL_OPENGL43_STATUS_SPIRV_LAYOUT_INVALID:
                    return "spirv-layout-invalid";
                case CROSSTL_OPENGL43_STATUS_SPIRV_MAGIC_INVALID:
                    return "spirv-magic-invalid";
                case CROSSTL_OPENGL43_STATUS_SPIRV_CAPABILITY_UNSUPPORTED:
                    return "spirv-capability-unsupported";
                case CROSSTL_OPENGL43_STATUS_SPIRV_ENTRY_POINT_MISSING:
                    return "spirv-entry-point-missing";
                case CROSSTL_OPENGL43_STATUS_SPECIALIZATION_ID_REQUIRED:
                    return "specialization-id-required";
                case CROSSTL_OPENGL43_STATUS_SPECIALIZATION_TYPE_UNSUPPORTED:
                    return "specialization-type-unsupported";
                case CROSSTL_OPENGL43_STATUS_SPECIALIZATION_PAYLOAD_INVALID:
                    return "specialization-payload-invalid";
                case CROSSTL_OPENGL43_STATUS_SPECIALIZATION_ID_DUPLICATE:
                    return "specialization-id-duplicate";
                case CROSSTL_OPENGL43_STATUS_SPIRV_BINARY_LOAD_FAILED:
                    return "spirv-binary-load-failed";
                case CROSSTL_OPENGL43_STATUS_SPIRV_SPECIALIZATION_FAILED:
                    return "spirv-specialization-failed";
                default:
                    return "unknown";
            }
        }

        static inline int32_t crosstl_opengl43_report(
            CrossTLOpenGL43Context *context,
            int32_t status,
            const char *phase,
            const char *message) {
            if (context != NULL && context->diagnostic_buffer != NULL &&
                context->diagnostic_buffer_capacity != 0u) {
                const char *source = message != NULL ? message : "";
                std::snprintf(
                    context->diagnostic_buffer,
                    context->diagnostic_buffer_capacity,
                    "%s",
                    source);
            }
            if (context != NULL && context->report_diagnostic != NULL) {
                context->report_diagnostic(
                    context->user_data, status, phase, message != NULL ? message : "");
            }
            return status;
        }

        static inline int32_t crosstl_opengl43_initialize_context(
            CrossTLOpenGL43Context *context,
            const CrossTLOpenGL43Functions *functions,
            void *user_data,
            CrossTLOpenGL43ContextCurrentFunction is_context_current) {
            if (context == NULL || functions == NULL) {
                return CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT;
            }
            *context = CrossTLOpenGL43Context{};
            context->adapter_version =
                CROSSTL_NATIVE_LOADER_TARGET_ADAPTER_VERSION;
            context->structure_size = sizeof(CrossTLOpenGL43Context);
            context->functions = *functions;
            context->user_data = user_data;
            context->is_context_current = is_context_current;
            return CROSSTL_OPENGL43_STATUS_OK;
        }

        static inline int32_t crosstl_opengl43_require_function(
            CrossTLOpenGL43Context *context,
            int function_available,
            const char *name) {
            if (function_available != 0) {
                return CROSSTL_OPENGL43_STATUS_OK;
            }
            char message[192];
            std::snprintf(
                message,
                sizeof(message),
                "Required OpenGL entry point %s is unavailable.",
                name);
            return crosstl_opengl43_report(
                context,
                CROSSTL_OPENGL43_STATUS_ENTRY_POINT_MISSING,
                "validate-context",
                message);
        }

        static inline int32_t crosstl_opengl43_validate_function_table(
            CrossTLOpenGL43Context *context) {
            int32_t status = CROSSTL_OPENGL43_STATUS_OK;
        #define CROSSTL_OPENGL43_REQUIRE(member, name)                            \
            status = crosstl_opengl43_require_function(                         \
                context,                                                        \
                context->functions.member != NULL,                              \
                name);                                                          \
            if (status != CROSSTL_OPENGL43_STATUS_OK) {                         \
                return status;                                                  \
            }
            CROSSTL_OPENGL43_REQUIRE(get_error, "glGetError")
            CROSSTL_OPENGL43_REQUIRE(get_integerv, "glGetIntegerv")
            CROSSTL_OPENGL43_REQUIRE(get_integer_indexed, "glGetIntegeri_v")
            CROSSTL_OPENGL43_REQUIRE(create_shader, "glCreateShader")
            CROSSTL_OPENGL43_REQUIRE(get_shader_iv, "glGetShaderiv")
            CROSSTL_OPENGL43_REQUIRE(get_shader_info_log, "glGetShaderInfoLog")
            CROSSTL_OPENGL43_REQUIRE(delete_shader, "glDeleteShader")
            CROSSTL_OPENGL43_REQUIRE(create_program, "glCreateProgram")
            CROSSTL_OPENGL43_REQUIRE(attach_shader, "glAttachShader")
            CROSSTL_OPENGL43_REQUIRE(link_program, "glLinkProgram")
            CROSSTL_OPENGL43_REQUIRE(get_program_iv, "glGetProgramiv")
            CROSSTL_OPENGL43_REQUIRE(get_program_info_log, "glGetProgramInfoLog")
            CROSSTL_OPENGL43_REQUIRE(delete_program, "glDeleteProgram")
            CROSSTL_OPENGL43_REQUIRE(use_program, "glUseProgram")
            CROSSTL_OPENGL43_REQUIRE(gen_buffers, "glGenBuffers")
            CROSSTL_OPENGL43_REQUIRE(bind_buffer, "glBindBuffer")
            CROSSTL_OPENGL43_REQUIRE(buffer_data, "glBufferData")
            CROSSTL_OPENGL43_REQUIRE(bind_buffer_base, "glBindBufferBase")
            CROSSTL_OPENGL43_REQUIRE(delete_buffers, "glDeleteBuffers")
            CROSSTL_OPENGL43_REQUIRE(dispatch_compute, "glDispatchCompute")
            CROSSTL_OPENGL43_REQUIRE(memory_barrier, "glMemoryBarrier")
            CROSSTL_OPENGL43_REQUIRE(finish, "glFinish")
            CROSSTL_OPENGL43_REQUIRE(get_buffer_sub_data, "glGetBufferSubData")
        #undef CROSSTL_OPENGL43_REQUIRE
            return CROSSTL_OPENGL43_STATUS_OK;
        }

        static inline int32_t crosstl_opengl43_validate_source_functions(
            CrossTLOpenGL43Context *context) {
            int32_t status = crosstl_opengl43_require_function(
                context,
                context->functions.shader_source != NULL,
                "glShaderSource");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            return crosstl_opengl43_require_function(
                context,
                context->functions.compile_shader != NULL,
                "glCompileShader");
        }

        static inline void crosstl_opengl43_clear_errors(
            CrossTLOpenGL43Context *context) {
            for (size_t index = 0u; index < 32u; ++index) {
                if (context->functions.get_error() == CROSSTL_OPENGL43_NO_ERROR) {
                    return;
                }
            }
        }

        static inline int32_t crosstl_opengl43_check_error_status(
            CrossTLOpenGL43Context *context,
            int32_t status,
            const char *phase,
            const char *operation) {
            const CrossTLOpenGL43Enum error = context->functions.get_error();
            if (error == CROSSTL_OPENGL43_NO_ERROR) {
                return CROSSTL_OPENGL43_STATUS_OK;
            }
            char message[224];
            std::snprintf(
                message,
                sizeof(message),
                "%s failed with OpenGL error 0x%04X.",
                operation,
                static_cast<unsigned int>(error));
            return crosstl_opengl43_report(
                context, status, phase, message);
        }

        static inline int32_t crosstl_opengl43_check_error(
            CrossTLOpenGL43Context *context,
            const char *phase,
            const char *operation) {
            return crosstl_opengl43_check_error_status(
                context,
                CROSSTL_OPENGL43_STATUS_OPENGL_ERROR,
                phase,
                operation);
        }

        static inline int32_t crosstl_opengl43_validate_context(
            CrossTLOpenGL43Context *context) {
            if (context == NULL) {
                return CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT;
            }
            if (context->adapter_version !=
                    CROSSTL_NATIVE_LOADER_TARGET_ADAPTER_VERSION ||
                context->structure_size < sizeof(CrossTLOpenGL43Context)) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_CONTEXT_STRUCTURE_INVALID,
                    "validate-context",
                    "OpenGL adapter context structure or version is invalid.");
            }
            if (context->is_context_current == NULL) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_CONTEXT_CALLBACK_MISSING,
                    "validate-context",
                    "A context-current query callback is required.");
            }
            if (context->is_context_current(context->user_data) <= 0) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_CONTEXT_NOT_CURRENT,
                    "validate-context",
                    "The caller-owned OpenGL context is not current.");
            }
            int32_t status = crosstl_opengl43_validate_function_table(context);
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            CrossTLOpenGL43Int major = 0;
            CrossTLOpenGL43Int minor = 0;
            crosstl_opengl43_clear_errors(context);
            context->functions.get_integerv(
                CROSSTL_OPENGL43_MAJOR_VERSION, &major);
            context->functions.get_integerv(
                CROSSTL_OPENGL43_MINOR_VERSION, &minor);
            status = crosstl_opengl43_check_error(
                context, "validate-context", "glGetIntegerv");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            if (major < 4 || (major == 4 && minor < 3)) {
                char message[192];
                std::snprintf(
                    message,
                    sizeof(message),
                    "OpenGL 4.3 or newer is required; current context is %d.%d.",
                    static_cast<int>(major),
                    static_cast<int>(minor));
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_OPENGL_VERSION_UNSUPPORTED,
                    "validate-context",
                    message);
            }
            return CROSSTL_OPENGL43_STATUS_OK;
        }

        static inline int crosstl_opengl43_is_absolute_path(const char *path) {
            if (path == NULL || path[0] == '\0') {
                return 0;
            }
            return path[0] == '/' || path[0] == '\\' ||
                   (path[1] != '\0' && path[1] == ':');
        }

        static inline int crosstl_opengl43_path_is_contained(
            const std::filesystem::path &root,
            const std::filesystem::path &candidate) {
            std::filesystem::path::const_iterator root_part = root.begin();
            std::filesystem::path::const_iterator candidate_part =
                candidate.begin();
            for (; root_part != root.end(); ++root_part, ++candidate_part) {
                if (candidate_part == candidate.end() ||
                    *root_part != *candidate_part) {
                    return 0;
                }
            }
            return 1;
        }

        static inline int32_t crosstl_opengl43_artifact_path(
            CrossTLOpenGL43Context *context,
            const char *package_path,
            std::filesystem::path *path_out) {
            if (package_path == NULL || package_path[0] == '\0' ||
                path_out == NULL) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT,
                    "load-artifact",
                    "Artifact path resolution requires a package path and output.");
            }
            if (context->artifact_root == NULL ||
                context->artifact_root[0] == '\0') {
                *path_out = std::filesystem::path(package_path);
                return CROSSTL_OPENGL43_STATUS_OK;
            }
            const std::filesystem::path relative_path(package_path);
            if (crosstl_opengl43_is_absolute_path(package_path) ||
                relative_path.is_absolute() ||
                relative_path.has_root_name() ||
                relative_path.has_root_directory()) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_ARTIFACT_PATH_UNSAFE,
                    "load-artifact",
                    "Package artifact paths must be relative to artifact_root.");
            }
            std::error_code error;
            const std::filesystem::path root =
                std::filesystem::weakly_canonical(
                    std::filesystem::path(context->artifact_root), error);
            if (error) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_ARTIFACT_PATH_RESOLUTION_FAILED,
                    "load-artifact",
                    "artifact_root could not be resolved canonically.");
            }
            const std::filesystem::path candidate =
                std::filesystem::weakly_canonical(
                    root / relative_path, error);
            if (error) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_ARTIFACT_PATH_RESOLUTION_FAILED,
                    "load-artifact",
                    "The package artifact path could not be resolved canonically.");
            }
            if (!crosstl_opengl43_path_is_contained(root, candidate)) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_ARTIFACT_PATH_UNSAFE,
                    "load-artifact",
                    "The package artifact path escapes artifact_root.");
            }
            *path_out = candidate;
            return CROSSTL_OPENGL43_STATUS_OK;
        }

        static inline uint32_t crosstl_opengl43_sha256_rotate_right(
            uint32_t value, uint32_t count) {
            return (value >> count) | (value << (32u - count));
        }

        static inline void crosstl_opengl43_sha256_transform(
            uint32_t state[8], const uint8_t block[64]) {
            static const uint32_t constants[64] = {
                0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
                0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
                0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
                0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
                0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
                0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
                0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
                0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
                0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
                0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
                0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
                0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
                0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
                0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
                0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
                0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
            uint32_t words[64] = {};
            for (size_t index = 0u; index < 16u; ++index) {
                const size_t offset = index * 4u;
                words[index] =
                    (static_cast<uint32_t>(block[offset]) << 24u) |
                    (static_cast<uint32_t>(block[offset + 1u]) << 16u) |
                    (static_cast<uint32_t>(block[offset + 2u]) << 8u) |
                    static_cast<uint32_t>(block[offset + 3u]);
            }
            for (size_t index = 16u; index < 64u; ++index) {
                const uint32_t first =
                    crosstl_opengl43_sha256_rotate_right(
                        words[index - 15u], 7u) ^
                    crosstl_opengl43_sha256_rotate_right(
                        words[index - 15u], 18u) ^
                    (words[index - 15u] >> 3u);
                const uint32_t second =
                    crosstl_opengl43_sha256_rotate_right(
                        words[index - 2u], 17u) ^
                    crosstl_opengl43_sha256_rotate_right(
                        words[index - 2u], 19u) ^
                    (words[index - 2u] >> 10u);
                words[index] = words[index - 16u] + first +
                               words[index - 7u] + second;
            }
            uint32_t a = state[0];
            uint32_t b = state[1];
            uint32_t c = state[2];
            uint32_t d = state[3];
            uint32_t e = state[4];
            uint32_t f = state[5];
            uint32_t g = state[6];
            uint32_t h = state[7];
            for (size_t index = 0u; index < 64u; ++index) {
                const uint32_t upper_sigma_one =
                    crosstl_opengl43_sha256_rotate_right(e, 6u) ^
                    crosstl_opengl43_sha256_rotate_right(e, 11u) ^
                    crosstl_opengl43_sha256_rotate_right(e, 25u);
                const uint32_t choose = (e & f) ^ ((~e) & g);
                const uint32_t temporary_one =
                    h + upper_sigma_one + choose + constants[index] +
                    words[index];
                const uint32_t upper_sigma_zero =
                    crosstl_opengl43_sha256_rotate_right(a, 2u) ^
                    crosstl_opengl43_sha256_rotate_right(a, 13u) ^
                    crosstl_opengl43_sha256_rotate_right(a, 22u);
                const uint32_t majority =
                    (a & b) ^ (a & c) ^ (b & c);
                const uint32_t temporary_two =
                    upper_sigma_zero + majority;
                h = g;
                g = f;
                f = e;
                e = d + temporary_one;
                d = c;
                c = b;
                b = a;
                a = temporary_one + temporary_two;
            }
            state[0] += a;
            state[1] += b;
            state[2] += c;
            state[3] += d;
            state[4] += e;
            state[5] += f;
            state[6] += g;
            state[7] += h;
        }

        static inline int crosstl_opengl43_sha256_hex(
            const uint8_t *data, size_t size, char digest[65]) {
            if ((data == NULL && size != 0u) || digest == NULL ||
                size > std::numeric_limits<uint64_t>::max() / 8u) {
                return 0;
            }
            uint32_t state[8] = {
                0x6a09e667u,
                0xbb67ae85u,
                0x3c6ef372u,
                0xa54ff53au,
                0x510e527fu,
                0x9b05688cu,
                0x1f83d9abu,
                0x5be0cd19u};
            size_t offset = 0u;
            while (size - offset >= 64u) {
                crosstl_opengl43_sha256_transform(
                    state, data + offset);
                offset += 64u;
            }
            uint8_t final_blocks[128] = {};
            const size_t remainder = size - offset;
            if (remainder != 0u) {
                std::memcpy(final_blocks, data + offset, remainder);
            }
            final_blocks[remainder] = 0x80u;
            const size_t final_size = remainder < 56u ? 64u : 128u;
            const uint64_t bit_length = static_cast<uint64_t>(size) * 8u;
            for (size_t index = 0u; index < 8u; ++index) {
                final_blocks[final_size - 1u - index] =
                    static_cast<uint8_t>(bit_length >> (index * 8u));
            }
            crosstl_opengl43_sha256_transform(state, final_blocks);
            if (final_size == 128u) {
                crosstl_opengl43_sha256_transform(
                    state, final_blocks + 64u);
            }
            static const char hexadecimal[] = "0123456789abcdef";
            for (size_t index = 0u; index < 32u; ++index) {
                const uint8_t byte = static_cast<uint8_t>(
                    state[index / 4u] >>
                    (24u - static_cast<uint32_t>((index % 4u) * 8u)));
                digest[index * 2u] = hexadecimal[byte >> 4u];
                digest[index * 2u + 1u] = hexadecimal[byte & 0x0fu];
            }
            digest[64] = '\0';
            return 1;
        }

        static inline int crosstl_opengl43_hexadecimal_value(char value) {
            if (value >= '0' && value <= '9') {
                return value - '0';
            }
            if (value >= 'a' && value <= 'f') {
                return value - 'a' + 10;
            }
            if (value >= 'A' && value <= 'F') {
                return value - 'A' + 10;
            }
            return -1;
        }

        static inline int32_t crosstl_opengl43_verify_artifact_hash(
            CrossTLOpenGL43Context *context,
            const CrossTLNativeLoaderUnitDescriptor *unit,
            const uint8_t *data,
            size_t size) {
            if (!crosstl_native_loader_strings_equal(
                    unit->artifact_hash_algorithm, "sha256")) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_HASH_ALGORITHM_UNSUPPORTED,
                    "load-artifact",
                    "The artifact hash algorithm must be sha256.");
            }
            const char *expected = unit->artifact_hash_value;
            if (expected == NULL || std::strlen(expected) != 64u) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_HASH_FAILED,
                    "load-artifact",
                    "The descriptor SHA-256 value is malformed.");
            }
            for (size_t index = 0u; index < 64u; ++index) {
                if (crosstl_opengl43_hexadecimal_value(expected[index]) < 0) {
                    return crosstl_opengl43_report(
                        context,
                        CROSSTL_OPENGL43_STATUS_HASH_FAILED,
                        "load-artifact",
                        "The descriptor SHA-256 value is malformed.");
                }
            }
            char actual[65] = {};
            if (!crosstl_opengl43_sha256_hex(data, size, actual)) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_HASH_FAILED,
                    "load-artifact",
                    "The artifact SHA-256 calculation failed.");
            }
            uint32_t difference = 0u;
            for (size_t index = 0u; index < 64u; ++index) {
                const int expected_value =
                    crosstl_opengl43_hexadecimal_value(expected[index]);
                const int actual_value =
                    crosstl_opengl43_hexadecimal_value(actual[index]);
                difference |= static_cast<uint32_t>(
                    expected_value ^ actual_value);
            }
            if (difference != 0u) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_HASH_MISMATCH,
                    "load-artifact",
                    "The artifact SHA-256 does not match its descriptor.");
            }
            return CROSSTL_OPENGL43_STATUS_OK;
        }

        static inline int32_t crosstl_opengl43_load_artifact(
            void *opaque,
            const CrossTLNativeLoaderUnitDescriptor *unit,
            void **artifact_out) {
            CrossTLOpenGL43Context *context =
                static_cast<CrossTLOpenGL43Context *>(opaque);
            if (artifact_out != NULL) {
                *artifact_out = NULL;
            }
            if (unit == NULL || artifact_out == NULL) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT,
                    "load-artifact",
                    "Artifact loading requires a unit and output pointer.");
            }
            int32_t status = crosstl_opengl43_validate_context(context);
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            if (unit->abi_version != CROSSTL_NATIVE_LOADER_ABI_VERSION) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_ABI_VERSION_MISMATCH,
                    "load-artifact",
                    "The unit uses an unsupported native loader ABI version.");
            }
            if (!crosstl_native_loader_strings_equal(unit->target, "opengl")) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_TARGET_MISMATCH,
                    "load-artifact",
                    "The unit target is not OpenGL.");
            }
            CrossTLOpenGL43ArtifactKind kind =
                CROSSTL_OPENGL43_ARTIFACT_GLSL_SOURCE;
            if (crosstl_native_loader_strings_equal(
                    unit->artifact_format, "GLSL source")) {
                status = crosstl_opengl43_validate_source_functions(context);
                if (status != CROSSTL_OPENGL43_STATUS_OK) {
                    return status;
                }
            } else if (crosstl_native_loader_strings_equal(
                           unit->artifact_format, "SPIR-V binary")) {
                kind = CROSSTL_OPENGL43_ARTIFACT_SPIRV_BINARY;
            } else {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_ARTIFACT_FORMAT_UNSUPPORTED,
                    "load-artifact",
                    "OpenGL artifacts must use GLSL source or SPIR-V binary format.");
            }
            if (unit->artifact_path == NULL || unit->artifact_path[0] == '\0') {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT,
                    "load-artifact",
                    "The OpenGL artifact path is missing.");
            }
            try {
                std::filesystem::path path;
                status = crosstl_opengl43_artifact_path(
                    context, unit->artifact_path, &path);
                if (status != CROSSTL_OPENGL43_STATUS_OK) {
                    return status;
                }
                std::ifstream stream(path, std::ios::binary | std::ios::ate);
                if (!stream) {
                    return crosstl_opengl43_report(
                        context,
                        CROSSTL_OPENGL43_STATUS_ARTIFACT_READ_FAILED,
                        "load-artifact",
                        "The OpenGL artifact could not be opened.");
                }
                const std::ifstream::pos_type end = stream.tellg();
                if (end <= std::ifstream::pos_type(0)) {
                    return crosstl_opengl43_report(
                        context,
                        CROSSTL_OPENGL43_STATUS_ARTIFACT_EMPTY,
                        "load-artifact",
                        "The OpenGL artifact is empty.");
                }
                const uint64_t size = static_cast<uint64_t>(end);
                if (size != unit->artifact_size_bytes) {
                    return crosstl_opengl43_report(
                        context,
                        CROSSTL_OPENGL43_STATUS_ARTIFACT_SIZE_MISMATCH,
                        "load-artifact",
                        "The OpenGL artifact size does not match its descriptor.");
                }
                if (size >
                    static_cast<uint64_t>(
                        std::numeric_limits<CrossTLOpenGL43Sizei>::max())) {
                    return crosstl_opengl43_report(
                        context,
                        CROSSTL_OPENGL43_STATUS_ARTIFACT_SIZE_UNSUPPORTED,
                        "load-artifact",
                        "The artifact exceeds the OpenGL input length limit.");
                }
                CrossTLOpenGL43Artifact *artifact =
                    new (std::nothrow) CrossTLOpenGL43Artifact{
                        kind, {}, {}};
                if (artifact == NULL) {
                    return crosstl_opengl43_report(
                        context,
                        CROSSTL_OPENGL43_STATUS_OUT_OF_MEMORY,
                        "load-artifact",
                        "The OpenGL artifact allocation failed.");
                }
                try {
                    artifact->bytes.resize(static_cast<size_t>(size));
                } catch (...) {
                    delete artifact;
                    return crosstl_opengl43_report(
                        context,
                        CROSSTL_OPENGL43_STATUS_OUT_OF_MEMORY,
                        "load-artifact",
                        "The OpenGL artifact byte allocation failed.");
                }
                stream.seekg(0, std::ios::beg);
                stream.read(
                    reinterpret_cast<char *>(artifact->bytes.data()),
                    static_cast<std::streamsize>(artifact->bytes.size()));
                if (!stream) {
                    delete artifact;
                    return crosstl_opengl43_report(
                        context,
                        CROSSTL_OPENGL43_STATUS_ARTIFACT_READ_FAILED,
                        "load-artifact",
                        "The complete OpenGL artifact could not be read.");
                }
                status = crosstl_opengl43_verify_artifact_hash(
                    context,
                    unit,
                    artifact->bytes.data(),
                    artifact->bytes.size());
                if (status != CROSSTL_OPENGL43_STATUS_OK) {
                    delete artifact;
                    return status;
                }
                if (kind == CROSSTL_OPENGL43_ARTIFACT_SPIRV_BINARY) {
                    if (artifact->bytes.size() < 20u ||
                        artifact->bytes.size() % sizeof(uint32_t) != 0u) {
                        delete artifact;
                        return crosstl_opengl43_report(
                            context,
                            CROSSTL_OPENGL43_STATUS_SPIRV_LAYOUT_INVALID,
                            "load-artifact",
                            "SPIR-V must contain an aligned five-word header.");
                    }
                    if (artifact->bytes[0] != 0x03u ||
                        artifact->bytes[1] != 0x02u ||
                        artifact->bytes[2] != 0x23u ||
                        artifact->bytes[3] != 0x07u) {
                        delete artifact;
                        return crosstl_opengl43_report(
                            context,
                            CROSSTL_OPENGL43_STATUS_SPIRV_MAGIC_INVALID,
                            "load-artifact",
                            "The SPIR-V artifact has an invalid magic word.");
                    }
                }
                *artifact_out = artifact;
                return CROSSTL_OPENGL43_STATUS_OK;
            } catch (...) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_INTERNAL_EXCEPTION,
                    "load-artifact",
                    "An exception occurred while loading the OpenGL artifact.");
            }
        }

        static inline int32_t crosstl_opengl43_unload_artifact(
            void *opaque, void *artifact) {
            CrossTLOpenGL43Context *context =
                static_cast<CrossTLOpenGL43Context *>(opaque);
            if (artifact == NULL) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT,
                    "cleanup-artifact",
                    "Artifact cleanup received a null artifact.");
            }
            delete static_cast<CrossTLOpenGL43Artifact *>(artifact);
            return CROSSTL_OPENGL43_STATUS_OK;
        }

        static inline int32_t crosstl_opengl43_report_shader_log(
            CrossTLOpenGL43Context *context,
            CrossTLOpenGL43UInt shader,
            int32_t status,
            const char *phase,
            const char *fallback) {
            CrossTLOpenGL43Int length = 0;
            context->functions.get_shader_iv(
                shader, CROSSTL_OPENGL43_INFO_LOG_LENGTH, &length);
            if (length <= 1) {
                return crosstl_opengl43_report(
                    context, status, phase, fallback);
            }
            try {
                std::vector<CrossTLOpenGL43Char> log(
                    static_cast<size_t>(length), '\0');
                CrossTLOpenGL43Sizei written = 0;
                context->functions.get_shader_info_log(
                    shader, length, &written, log.data());
                return crosstl_opengl43_report(
                    context, status, phase, log.data());
            } catch (...) {
                return crosstl_opengl43_report(
                    context, status, phase, fallback);
            }
        }

        static inline int32_t crosstl_opengl43_report_program_log(
            CrossTLOpenGL43Context *context,
            CrossTLOpenGL43UInt program,
            int32_t status,
            const char *phase,
            const char *fallback) {
            CrossTLOpenGL43Int length = 0;
            context->functions.get_program_iv(
                program, CROSSTL_OPENGL43_INFO_LOG_LENGTH, &length);
            if (length <= 1) {
                return crosstl_opengl43_report(
                    context, status, phase, fallback);
            }
            try {
                std::vector<CrossTLOpenGL43Char> log(
                    static_cast<size_t>(length), '\0');
                CrossTLOpenGL43Sizei written = 0;
                context->functions.get_program_info_log(
                    program, length, &written, log.data());
                return crosstl_opengl43_report(
                    context, status, phase, log.data());
            } catch (...) {
                return crosstl_opengl43_report(
                    context, status, phase, fallback);
            }
        }

        static inline int32_t crosstl_opengl43_spirv_specializer(
            CrossTLOpenGL43Context *context,
            CrossTLOpenGL43SpecializeShaderFunction *specializer_out,
            const char **entry_point_name_out) {
            if (specializer_out == NULL || entry_point_name_out == NULL) {
                return CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT;
            }
            *specializer_out = NULL;
            *entry_point_name_out = NULL;
            if (context->functions.shader_binary == NULL) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_SPIRV_ENTRY_POINT_MISSING,
                    "create-pipeline",
                    "OpenGL SPIR-V requires glShaderBinary.");
            }
            CrossTLOpenGL43Int major = 0;
            CrossTLOpenGL43Int minor = 0;
            crosstl_opengl43_clear_errors(context);
            context->functions.get_integerv(
                CROSSTL_OPENGL43_MAJOR_VERSION, &major);
            context->functions.get_integerv(
                CROSSTL_OPENGL43_MINOR_VERSION, &minor);
            int32_t status = crosstl_opengl43_check_error(
                context, "create-pipeline", "OpenGL version query");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            const int core_capability =
                major > 4 || (major == 4 && minor >= 6);
            if (core_capability &&
                context->functions.specialize_shader != NULL) {
                *specializer_out =
                    context->functions.specialize_shader;
                *entry_point_name_out = "glSpecializeShader";
                return CROSSTL_OPENGL43_STATUS_OK;
            }
            if (context->supports_arb_gl_spirv != 0u &&
                context->functions.specialize_shader_arb != NULL) {
                *specializer_out =
                    context->functions.specialize_shader_arb;
                *entry_point_name_out = "glSpecializeShaderARB";
                return CROSSTL_OPENGL43_STATUS_OK;
            }
            if (!core_capability &&
                context->supports_arb_gl_spirv == 0u) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_SPIRV_CAPABILITY_UNSUPPORTED,
                    "create-pipeline",
                    "SPIR-V specialization requires OpenGL 4.6 or "
                    "caller-confirmed GL_ARB_gl_spirv support.");
            }
            return crosstl_opengl43_report(
                context,
                CROSSTL_OPENGL43_STATUS_SPIRV_ENTRY_POINT_MISSING,
                "create-pipeline",
                core_capability
                    ? "OpenGL 4.6 SPIR-V requires glSpecializeShader."
                    : "GL_ARB_gl_spirv requires glSpecializeShaderARB.");
        }

        static inline int32_t crosstl_opengl43_create_pipeline(
            void *opaque,
            void *artifact_opaque,
            const CrossTLNativeLoaderUnitDescriptor *unit,
            void **pipeline_out) {
            CrossTLOpenGL43Context *context =
                static_cast<CrossTLOpenGL43Context *>(opaque);
            if (pipeline_out != NULL) {
                *pipeline_out = NULL;
            }
            if (artifact_opaque == NULL || unit == NULL ||
                pipeline_out == NULL) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT,
                    "create-pipeline",
                    "Pipeline creation requires artifact, unit, and output.");
            }
            int32_t status = crosstl_opengl43_validate_context(context);
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            if (!crosstl_native_loader_strings_equal(unit->stage, "compute")) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_STAGE_UNSUPPORTED,
                    "create-pipeline",
                    "The OpenGL 4.3 adapter supports compute stages only.");
            }
            CrossTLOpenGL43Artifact *artifact =
                static_cast<CrossTLOpenGL43Artifact *>(artifact_opaque);
            if (artifact->kind ==
                    CROSSTL_OPENGL43_ARTIFACT_GLSL_SOURCE &&
                !crosstl_native_loader_strings_equal(
                    unit->entry_point, "main")) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_ENTRY_POINT_UNSUPPORTED,
                    "create-pipeline",
                    "GLSL source execution requires entry point main.");
            }
            if (artifact->kind ==
                    CROSSTL_OPENGL43_ARTIFACT_SPIRV_BINARY &&
                (unit->entry_point == NULL ||
                 unit->entry_point[0] == '\0')) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_ENTRY_POINT_UNSUPPORTED,
                    "create-pipeline",
                    "SPIR-V specialization requires an entry point name.");
            }
            CrossTLOpenGL43SpecializeShaderFunction specializer = NULL;
            const char *specializer_name = NULL;
            std::vector<CrossTLOpenGL43UInt> specialization_ids;
            std::vector<CrossTLOpenGL43UInt> specialization_values;
            if (artifact->kind ==
                CROSSTL_OPENGL43_ARTIFACT_SPIRV_BINARY) {
                status = crosstl_opengl43_spirv_specializer(
                    context, &specializer, &specializer_name);
                if (status != CROSSTL_OPENGL43_STATUS_OK) {
                    return status;
                }
                try {
                    specialization_ids.reserve(
                        artifact->specializations.size());
                    specialization_values.reserve(
                        artifact->specializations.size());
                    for (const CrossTLOpenGL43Specialization &item :
                         artifact->specializations) {
                        specialization_ids.push_back(item.constant_id);
                        specialization_values.push_back(item.encoded_value);
                    }
                } catch (...) {
                    return crosstl_opengl43_report(
                        context,
                        CROSSTL_OPENGL43_STATUS_OUT_OF_MEMORY,
                        "create-pipeline",
                        "SPIR-V specialization array allocation failed.");
                }
            } else {
                status = crosstl_opengl43_validate_source_functions(context);
                if (status != CROSSTL_OPENGL43_STATUS_OK) {
                    return status;
                }
            }
            CrossTLOpenGL43UInt shader = 0u;
            CrossTLOpenGL43UInt program = 0u;
            crosstl_opengl43_clear_errors(context);
            shader = context->functions.create_shader(
                CROSSTL_OPENGL43_COMPUTE_SHADER);
            status = crosstl_opengl43_check_error(
                context, "create-pipeline", "glCreateShader");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            if (shader == 0u) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_SHADER_CREATE_FAILED,
                    "create-pipeline",
                    "OpenGL did not create a compute shader.");
            }
            int32_t shader_failure_status =
                CROSSTL_OPENGL43_STATUS_SHADER_COMPILE_FAILED;
            const char *shader_failure_message =
                "GLSL compute shader compilation failed.";
            if (artifact->kind ==
                CROSSTL_OPENGL43_ARTIFACT_GLSL_SOURCE) {
                const CrossTLOpenGL43Char *source =
                    reinterpret_cast<const CrossTLOpenGL43Char *>(
                        artifact->bytes.data());
                const CrossTLOpenGL43Int length =
                    static_cast<CrossTLOpenGL43Int>(
                        artifact->bytes.size());
                context->functions.shader_source(
                    shader, 1, &source, &length);
                context->functions.compile_shader(shader);
                status = crosstl_opengl43_check_error(
                    context,
                    "create-pipeline",
                    "GLSL shader compilation");
            } else {
                context->functions.shader_binary(
                    1,
                    &shader,
                    CROSSTL_OPENGL43_SHADER_BINARY_FORMAT_SPIR_V,
                    artifact->bytes.data(),
                    static_cast<CrossTLOpenGL43Sizei>(
                        artifact->bytes.size()));
                status = crosstl_opengl43_check_error_status(
                    context,
                    CROSSTL_OPENGL43_STATUS_SPIRV_BINARY_LOAD_FAILED,
                    "create-pipeline",
                    "glShaderBinary");
                if (status == CROSSTL_OPENGL43_STATUS_OK) {
                    specializer(
                        shader,
                        unit->entry_point,
                        static_cast<CrossTLOpenGL43UInt>(
                            specialization_ids.size()),
                        specialization_ids.empty()
                            ? NULL
                            : specialization_ids.data(),
                        specialization_values.empty()
                            ? NULL
                            : specialization_values.data());
                    status = crosstl_opengl43_check_error_status(
                        context,
                        CROSSTL_OPENGL43_STATUS_SPIRV_SPECIALIZATION_FAILED,
                        "create-pipeline",
                        specializer_name);
                }
                shader_failure_status =
                    CROSSTL_OPENGL43_STATUS_SPIRV_SPECIALIZATION_FAILED;
                shader_failure_message =
                    "OpenGL SPIR-V shader specialization failed.";
            }
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                context->functions.delete_shader(shader);
                return status;
            }
            CrossTLOpenGL43Int compiled = 0;
            context->functions.get_shader_iv(
                shader, CROSSTL_OPENGL43_COMPILE_STATUS, &compiled);
            status = crosstl_opengl43_check_error(
                context, "create-pipeline", "glGetShaderiv");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                context->functions.delete_shader(shader);
                return status;
            }
            if (compiled == 0) {
                status = crosstl_opengl43_report_shader_log(
                    context,
                    shader,
                    shader_failure_status,
                    "create-pipeline",
                    shader_failure_message);
                context->functions.delete_shader(shader);
                return status;
            }
            program = context->functions.create_program();
            status = crosstl_opengl43_check_error(
                context, "create-pipeline", "glCreateProgram");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                context->functions.delete_shader(shader);
                return status;
            }
            if (program == 0u) {
                context->functions.delete_shader(shader);
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_PROGRAM_CREATE_FAILED,
                    "create-pipeline",
                    "OpenGL did not create a compute program.");
            }
            context->functions.attach_shader(program, shader);
            context->functions.link_program(program);
            status = crosstl_opengl43_check_error(
                context, "create-pipeline", "OpenGL program linking");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                context->functions.delete_program(program);
                context->functions.delete_shader(shader);
                return status;
            }
            CrossTLOpenGL43Int linked = 0;
            context->functions.get_program_iv(
                program, CROSSTL_OPENGL43_LINK_STATUS, &linked);
            status = crosstl_opengl43_check_error(
                context, "create-pipeline", "glGetProgramiv");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                context->functions.delete_program(program);
                context->functions.delete_shader(shader);
                return status;
            }
            if (linked == 0) {
                status = crosstl_opengl43_report_program_log(
                    context,
                    program,
                    CROSSTL_OPENGL43_STATUS_PROGRAM_LINK_FAILED,
                    "create-pipeline",
                    artifact->kind ==
                            CROSSTL_OPENGL43_ARTIFACT_GLSL_SOURCE
                        ? "GLSL compute program linking failed."
                        : "SPIR-V compute program linking failed.");
                context->functions.delete_program(program);
                context->functions.delete_shader(shader);
                return status;
            }
            CrossTLOpenGL43Pipeline *pipeline =
                new (std::nothrow) CrossTLOpenGL43Pipeline{
                    shader, program, 0u, 0};
            if (pipeline == NULL) {
                context->functions.delete_program(program);
                context->functions.delete_shader(shader);
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_OUT_OF_MEMORY,
                    "create-pipeline",
                    "The OpenGL pipeline allocation failed.");
            }
            *pipeline_out = pipeline;
            return CROSSTL_OPENGL43_STATUS_OK;
        }

        static inline int32_t crosstl_opengl43_destroy_pipeline(
            void *opaque, void *pipeline_opaque) {
            CrossTLOpenGL43Context *context =
                static_cast<CrossTLOpenGL43Context *>(opaque);
            if (pipeline_opaque == NULL) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT,
                    "cleanup-pipeline",
                    "Pipeline cleanup received a null pipeline.");
            }
            CrossTLOpenGL43Pipeline *pipeline =
                static_cast<CrossTLOpenGL43Pipeline *>(pipeline_opaque);
            int32_t status = crosstl_opengl43_validate_context(context);
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                delete pipeline;
                return status;
            }
            crosstl_opengl43_clear_errors(context);
            context->functions.delete_program(pipeline->program);
            context->functions.delete_shader(pipeline->shader);
            status = crosstl_opengl43_check_error(
                context, "cleanup-pipeline", "OpenGL pipeline deletion");
            delete pipeline;
            return status;
        }

        static inline int32_t crosstl_opengl43_apply_specialization(
            void *opaque,
            void *artifact_opaque,
            const CrossTLNativeLoaderSpecializationDescriptor *descriptor,
            const CrossTLNativeLoaderSpecializationRequest *request) {
            CrossTLOpenGL43Context *context =
                static_cast<CrossTLOpenGL43Context *>(opaque);
            if (artifact_opaque == NULL || descriptor == NULL ||
                request == NULL || request->payload == NULL) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT,
                    "apply-specialization",
                    "Specialization requires artifact, descriptor, and payload.");
            }
            CrossTLOpenGL43Artifact *artifact =
                static_cast<CrossTLOpenGL43Artifact *>(artifact_opaque);
            if (artifact->kind ==
                CROSSTL_OPENGL43_ARTIFACT_GLSL_SOURCE) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_SPECIALIZATION_UNSUPPORTED,
                    "apply-specialization",
                    "GLSL source specialization is unsupported; provide a "
                    "SPIR-V binary artifact.");
            }
            if (descriptor->has_id == 0u || request->has_id == 0u ||
                descriptor->id != request->id) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_SPECIALIZATION_ID_REQUIRED,
                    "apply-specialization",
                    "OpenGL SPIR-V specialization requires a matching numeric ID.");
            }
            uint32_t encoded_value = 0u;
            if (crosstl_native_loader_strings_equal(
                    descriptor->type_name, "bool")) {
                if (request->payload_size_bytes == sizeof(uint8_t)) {
                    uint8_t value = 0u;
                    std::memcpy(
                        &value, request->payload, sizeof(value));
                    encoded_value = value;
                } else if (
                    request->payload_size_bytes == sizeof(uint32_t)) {
                    std::memcpy(
                        &encoded_value,
                        request->payload,
                        sizeof(encoded_value));
                } else {
                    return crosstl_opengl43_report(
                        context,
                        CROSSTL_OPENGL43_STATUS_SPECIALIZATION_PAYLOAD_INVALID,
                        "apply-specialization",
                        "A bool specialization payload must be one or four bytes.");
                }
                if (encoded_value > 1u) {
                    return crosstl_opengl43_report(
                        context,
                        CROSSTL_OPENGL43_STATUS_SPECIALIZATION_PAYLOAD_INVALID,
                        "apply-specialization",
                        "A bool specialization payload must encode zero or one.");
                }
            } else if (
                crosstl_native_loader_strings_equal(
                    descriptor->type_name, "int32") ||
                crosstl_native_loader_strings_equal(
                    descriptor->type_name, "uint32") ||
                crosstl_native_loader_strings_equal(
                    descriptor->type_name, "float32")) {
                if (request->payload_size_bytes != sizeof(uint32_t)) {
                    return crosstl_opengl43_report(
                        context,
                        CROSSTL_OPENGL43_STATUS_SPECIALIZATION_PAYLOAD_INVALID,
                        "apply-specialization",
                        "A 32-bit specialization payload must contain four bytes.");
                }
                std::memcpy(
                    &encoded_value,
                    request->payload,
                    sizeof(encoded_value));
                if (crosstl_native_loader_strings_equal(
                        descriptor->type_name, "float32")) {
                    float value = 0.0f;
                    std::memcpy(
                        &value, request->payload, sizeof(value));
                    if (!std::isfinite(value)) {
                        return crosstl_opengl43_report(
                            context,
                            CROSSTL_OPENGL43_STATUS_SPECIALIZATION_PAYLOAD_INVALID,
                            "apply-specialization",
                            "A float32 specialization payload must be finite.");
                    }
                }
            } else {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_SPECIALIZATION_TYPE_UNSUPPORTED,
                    "apply-specialization",
                    "OpenGL SPIR-V specialization supports bool, int32, "
                    "uint32, and float32.");
            }
            size_t position = 0u;
            while (position < artifact->specializations.size() &&
                   artifact->specializations[position].constant_id <
                       descriptor->id) {
                ++position;
            }
            if (position < artifact->specializations.size() &&
                artifact->specializations[position].constant_id ==
                    descriptor->id) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_SPECIALIZATION_ID_DUPLICATE,
                    "apply-specialization",
                    "A specialization constant ID was applied more than once.");
            }
            try {
                artifact->specializations.insert(
                    artifact->specializations.begin() +
                        static_cast<std::ptrdiff_t>(position),
                    CrossTLOpenGL43Specialization{
                        descriptor->id, encoded_value});
            } catch (...) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_OUT_OF_MEMORY,
                    "apply-specialization",
                    "The specialization value could not be retained.");
            }
            return CROSSTL_OPENGL43_STATUS_OK;
        }

        static inline int32_t crosstl_opengl43_binding_target(
            CrossTLOpenGL43Context *context,
            const CrossTLNativeLoaderBindingDescriptor *descriptor,
            CrossTLOpenGL43Enum *target_out,
            CrossTLOpenGL43Enum *binding_token_out) {
            if (descriptor->set_index != 0u) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_RESOURCE_SET_UNSUPPORTED,
                    "bind-resource",
                    "OpenGL native bindings require descriptor set zero.");
            }
            if (crosstl_native_loader_strings_equal(
                    descriptor->binding_namespace, "storage-buffer") &&
                crosstl_native_loader_strings_equal(
                    descriptor->resource_kind, "buffer")) {
                *target_out = CROSSTL_OPENGL43_SHADER_STORAGE_BUFFER;
                *binding_token_out =
                    CROSSTL_OPENGL43_SHADER_STORAGE_BUFFER_BINDING;
                return CROSSTL_OPENGL43_STATUS_OK;
            }
            if (crosstl_native_loader_strings_equal(
                    descriptor->binding_namespace, "uniform-buffer") &&
                (crosstl_native_loader_strings_equal(
                     descriptor->resource_kind, "constant-buffer") ||
                 crosstl_native_loader_strings_equal(
                     descriptor->resource_kind, "uniform"))) {
                if (descriptor->access != CROSSTL_NATIVE_LOADER_ACCESS_READ) {
                    return crosstl_opengl43_report(
                        context,
                        CROSSTL_OPENGL43_STATUS_RESOURCE_ACCESS_UNSUPPORTED,
                        "bind-resource",
                        "OpenGL uniform buffers must use read access.");
                }
                *target_out = CROSSTL_OPENGL43_UNIFORM_BUFFER;
                *binding_token_out =
                    CROSSTL_OPENGL43_UNIFORM_BUFFER_BINDING;
                return CROSSTL_OPENGL43_STATUS_OK;
            }
            return crosstl_opengl43_report(
                context,
                CROSSTL_OPENGL43_STATUS_RESOURCE_KIND_UNSUPPORTED,
                "bind-resource",
                "The reflected resource kind is not an SSBO or uniform buffer.");
        }

        static inline int32_t crosstl_opengl43_bind_resource(
            void *opaque,
            void *pipeline_opaque,
            const CrossTLNativeLoaderBindingDescriptor *descriptor,
            const CrossTLNativeLoaderBindingRequest *request,
            void **resource_out) {
            CrossTLOpenGL43Context *context =
                static_cast<CrossTLOpenGL43Context *>(opaque);
            if (resource_out != NULL) {
                *resource_out = NULL;
            }
            if (pipeline_opaque == NULL || descriptor == NULL ||
                request == NULL || resource_out == NULL ||
                request->payload == NULL || request->payload_size_bytes == 0u) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT,
                    "bind-resource",
                    "Resource binding received an incomplete request.");
            }
            int32_t status = crosstl_opengl43_validate_context(context);
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            if (request->payload_size_bytes >
                static_cast<size_t>(
                    std::numeric_limits<CrossTLOpenGL43SizePtr>::max())) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_RESOURCE_SIZE_UNSUPPORTED,
                    "bind-resource",
                    "The resource exceeds the OpenGL buffer-size limit.");
            }
            CrossTLOpenGL43Enum target = 0u;
            CrossTLOpenGL43Enum binding_token = 0u;
            status = crosstl_opengl43_binding_target(
                context, descriptor, &target, &binding_token);
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            CrossTLOpenGL43Int previous_generic_buffer = 0;
            CrossTLOpenGL43Int previous_indexed_buffer = 0;
            crosstl_opengl43_clear_errors(context);
            context->functions.get_integerv(
                binding_token, &previous_generic_buffer);
            context->functions.get_integer_indexed(
                binding_token,
                descriptor->binding_index,
                &previous_indexed_buffer);
            status = crosstl_opengl43_check_error(
                context, "bind-resource", "OpenGL buffer binding query");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            CrossTLOpenGL43UInt buffer = 0u;
            crosstl_opengl43_clear_errors(context);
            context->functions.gen_buffers(1, &buffer);
            status = crosstl_opengl43_check_error(
                context, "bind-resource", "glGenBuffers");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            if (buffer == 0u) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_BUFFER_CREATE_FAILED,
                    "bind-resource",
                    "OpenGL did not create a buffer.");
            }
            context->functions.bind_buffer(target, buffer);
            context->functions.buffer_data(
                target,
                static_cast<CrossTLOpenGL43SizePtr>(
                    request->payload_size_bytes),
                request->payload,
                CROSSTL_OPENGL43_DYNAMIC_COPY);
            context->functions.bind_buffer_base(
                target, descriptor->binding_index, buffer);
            status = crosstl_opengl43_check_error(
                context, "bind-resource", "OpenGL buffer upload and binding");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                crosstl_opengl43_clear_errors(context);
                context->functions.bind_buffer_base(
                    target,
                    descriptor->binding_index,
                    static_cast<CrossTLOpenGL43UInt>(
                        previous_indexed_buffer));
                context->functions.bind_buffer(
                    target,
                    static_cast<CrossTLOpenGL43UInt>(
                        previous_generic_buffer));
                context->functions.delete_buffers(1, &buffer);
                return status;
            }
            context->functions.bind_buffer(
                target,
                static_cast<CrossTLOpenGL43UInt>(previous_generic_buffer));
            status = crosstl_opengl43_check_error(
                context,
                "bind-resource",
                "OpenGL generic buffer binding restoration");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                context->functions.bind_buffer_base(
                    target,
                    descriptor->binding_index,
                    static_cast<CrossTLOpenGL43UInt>(
                        previous_indexed_buffer));
                context->functions.delete_buffers(1, &buffer);
                return status;
            }
            CrossTLOpenGL43Resource *resource =
                new (std::nothrow) CrossTLOpenGL43Resource{
                    context,
                    target,
                    buffer,
                    descriptor->binding_index,
                    static_cast<CrossTLOpenGL43UInt>(
                        previous_indexed_buffer),
                    request->payload_size_bytes};
            if (resource == NULL) {
                context->functions.bind_buffer_base(
                    target,
                    descriptor->binding_index,
                    static_cast<CrossTLOpenGL43UInt>(
                        previous_indexed_buffer));
                context->functions.delete_buffers(1, &buffer);
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_OUT_OF_MEMORY,
                    "bind-resource",
                    "The OpenGL resource allocation failed.");
            }
            *resource_out = resource;
            return CROSSTL_OPENGL43_STATUS_OK;
        }

        static inline int32_t crosstl_opengl43_release_resource(
            void *opaque,
            void *resource_opaque,
            const CrossTLNativeLoaderBindingDescriptor *descriptor) {
            CrossTLOpenGL43Context *context =
                static_cast<CrossTLOpenGL43Context *>(opaque);
            (void)descriptor;
            if (resource_opaque == NULL) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT,
                    "cleanup-resource",
                    "Resource cleanup received a null resource.");
            }
            CrossTLOpenGL43Resource *resource =
                static_cast<CrossTLOpenGL43Resource *>(resource_opaque);
            int32_t status = crosstl_opengl43_validate_context(context);
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                delete resource;
                return status;
            }
            if (resource->owner != context) {
                status = crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT,
                    "cleanup-resource",
                    "The resource belongs to a different OpenGL context.");
                delete resource;
                return status;
            }
            crosstl_opengl43_clear_errors(context);
            context->functions.bind_buffer_base(
                resource->target,
                resource->binding,
                resource->previous_indexed_buffer);
            context->functions.delete_buffers(1, &resource->buffer);
            status = crosstl_opengl43_check_error(
                context, "cleanup-resource", "OpenGL buffer deletion");
            delete resource;
            return status;
        }

        static inline int32_t crosstl_opengl43_dispatch(
            void *opaque,
            void *pipeline_opaque,
            const CrossTLNativeLoaderDispatchGeometry *geometry) {
            CrossTLOpenGL43Context *context =
                static_cast<CrossTLOpenGL43Context *>(opaque);
            if (pipeline_opaque == NULL || geometry == NULL) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT,
                    "dispatch",
                    "Dispatch requires a pipeline and geometry.");
            }
            int32_t status = crosstl_opengl43_validate_context(context);
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            for (CrossTLOpenGL43UInt axis = 0u; axis < 3u; ++axis) {
                CrossTLOpenGL43Int maximum = 0;
                context->functions.get_integer_indexed(
                    CROSSTL_OPENGL43_MAX_COMPUTE_WORK_GROUP_COUNT,
                    axis,
                    &maximum);
                status = crosstl_opengl43_check_error(
                    context, "dispatch", "glGetIntegeri_v");
                if (status != CROSSTL_OPENGL43_STATUS_OK) {
                    return status;
                }
                if (maximum <= 0 ||
                    geometry->workgroup_count[axis] >
                        static_cast<uint32_t>(maximum)) {
                    return crosstl_opengl43_report(
                        context,
                        CROSSTL_OPENGL43_STATUS_DISPATCH_LIMIT_EXCEEDED,
                        "dispatch",
                        "Dispatch workgroup count exceeds the context limit.");
                }
            }
            CrossTLOpenGL43Pipeline *pipeline =
                static_cast<CrossTLOpenGL43Pipeline *>(pipeline_opaque);
            crosstl_opengl43_clear_errors(context);
            CrossTLOpenGL43Int previous_program = 0;
            context->functions.get_integerv(
                CROSSTL_OPENGL43_CURRENT_PROGRAM, &previous_program);
            status = crosstl_opengl43_check_error(
                context, "dispatch", "OpenGL current program query");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            context->functions.use_program(pipeline->program);
            context->functions.dispatch_compute(
                geometry->workgroup_count[0],
                geometry->workgroup_count[1],
                geometry->workgroup_count[2]);
            status = crosstl_opengl43_check_error(
                context, "dispatch", "glDispatchCompute");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                context->functions.use_program(
                    static_cast<CrossTLOpenGL43UInt>(previous_program));
                return status;
            }
            context->functions.memory_barrier(
                CROSSTL_OPENGL43_SHADER_STORAGE_BARRIER_BIT |
                CROSSTL_OPENGL43_BUFFER_UPDATE_BARRIER_BIT |
                CROSSTL_OPENGL43_UNIFORM_BARRIER_BIT);
            status = crosstl_opengl43_check_error(
                context, "dispatch", "glMemoryBarrier");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                context->functions.use_program(
                    static_cast<CrossTLOpenGL43UInt>(previous_program));
                return status;
            }
            context->functions.use_program(
                static_cast<CrossTLOpenGL43UInt>(previous_program));
            status = crosstl_opengl43_check_error(
                context, "dispatch", "OpenGL current program restoration");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            ++context->dispatch_serial;
            if (context->dispatch_serial == 0u) {
                ++context->dispatch_serial;
            }
            context->synchronized_serial = 0u;
            pipeline->dispatch_serial = context->dispatch_serial;
            pipeline->dispatched = 1;
            return CROSSTL_OPENGL43_STATUS_OK;
        }

        static inline int32_t crosstl_opengl43_synchronize(
            void *opaque, void *pipeline_opaque) {
            CrossTLOpenGL43Context *context =
                static_cast<CrossTLOpenGL43Context *>(opaque);
            if (pipeline_opaque == NULL) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT,
                    "synchronize",
                    "Synchronization requires a pipeline.");
            }
            int32_t status = crosstl_opengl43_validate_context(context);
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            CrossTLOpenGL43Pipeline *pipeline =
                static_cast<CrossTLOpenGL43Pipeline *>(pipeline_opaque);
            if (pipeline->dispatched == 0 || pipeline->dispatch_serial == 0u) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_DISPATCH_REQUIRED,
                    "synchronize",
                    "A successful compute dispatch is required before synchronization.");
            }
            crosstl_opengl43_clear_errors(context);
            context->functions.finish();
            status = crosstl_opengl43_check_error(
                context, "synchronize", "glFinish");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            context->synchronized_serial = pipeline->dispatch_serial;
            return CROSSTL_OPENGL43_STATUS_OK;
        }

        static inline int32_t crosstl_opengl43_readback(
            void *opaque,
            void *resource_opaque,
            const CrossTLNativeLoaderBindingDescriptor *descriptor,
            const CrossTLNativeLoaderBindingRequest *request) {
            CrossTLOpenGL43Context *context =
                static_cast<CrossTLOpenGL43Context *>(opaque);
            (void)descriptor;
            if (resource_opaque == NULL || request == NULL ||
                request->payload == NULL) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT,
                    "readback",
                    "Readback received an incomplete resource request.");
            }
            int32_t status = crosstl_opengl43_validate_context(context);
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            CrossTLOpenGL43Resource *resource =
                static_cast<CrossTLOpenGL43Resource *>(resource_opaque);
            if (resource->owner != context) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_INVALID_ARGUMENT,
                    "readback",
                    "The resource belongs to a different OpenGL context.");
            }
            if (request->payload_size_bytes != resource->size_bytes) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_RESOURCE_SIZE_MISMATCH,
                    "readback",
                    "Readback size does not match the bound OpenGL buffer.");
            }
            if (context->dispatch_serial == 0u ||
                context->synchronized_serial != context->dispatch_serial) {
                return crosstl_opengl43_report(
                    context,
                    CROSSTL_OPENGL43_STATUS_SYNCHRONIZATION_REQUIRED,
                    "readback",
                    "A successful synchronization is required before readback.");
            }
            crosstl_opengl43_clear_errors(context);
            CrossTLOpenGL43Enum binding_token =
                resource->target ==
                        CROSSTL_OPENGL43_SHADER_STORAGE_BUFFER
                    ? CROSSTL_OPENGL43_SHADER_STORAGE_BUFFER_BINDING
                    : CROSSTL_OPENGL43_UNIFORM_BUFFER_BINDING;
            CrossTLOpenGL43Int previous_generic_buffer = 0;
            context->functions.get_integerv(
                binding_token, &previous_generic_buffer);
            status = crosstl_opengl43_check_error(
                context, "readback", "OpenGL buffer binding query");
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            context->functions.bind_buffer(
                resource->target, resource->buffer);
            context->functions.get_buffer_sub_data(
                resource->target,
                0,
                static_cast<CrossTLOpenGL43SizePtr>(resource->size_bytes),
                request->payload);
            status = crosstl_opengl43_check_error(
                context, "readback", "glGetBufferSubData");
            context->functions.bind_buffer(
                resource->target,
                static_cast<CrossTLOpenGL43UInt>(previous_generic_buffer));
            if (status != CROSSTL_OPENGL43_STATUS_OK) {
                return status;
            }
            return crosstl_opengl43_check_error(
                context,
                "readback",
                "OpenGL generic buffer binding restoration");
        }

        static inline CrossTLNativeLoaderAdapter
        crosstl_opengl43_make_adapter(CrossTLOpenGL43Context *context) {
            CrossTLNativeLoaderAdapter adapter = {
                CROSSTL_NATIVE_LOADER_ABI_VERSION,
                "opengl",
                context,
                crosstl_opengl43_load_artifact,
                crosstl_opengl43_unload_artifact,
                crosstl_opengl43_create_pipeline,
                crosstl_opengl43_destroy_pipeline,
                crosstl_opengl43_apply_specialization,
                crosstl_opengl43_bind_resource,
                crosstl_opengl43_release_resource,
                crosstl_opengl43_dispatch,
                crosstl_opengl43_synchronize,
                crosstl_opengl43_readback};
            return adapter;
        }

        #undef CROSSTL_OPENGL43_APIENTRY

        #endif /* CROSSTL_NATIVE_LOADER_OPENGL43_ADAPTER_H */
        """).lstrip()
