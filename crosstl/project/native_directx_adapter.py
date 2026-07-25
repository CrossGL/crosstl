"""Generate the Direct3D 12 implementation of the native loader execution ABI."""

from __future__ import annotations


def generate_directx_native_loader_adapter() -> str:
    """Render the deterministic C++17 Direct3D 12 native loader adapter."""

    return _DIRECTX_NATIVE_LOADER_ADAPTER


_DIRECTX_NATIVE_LOADER_ADAPTER = r"""#ifndef CROSSTL_DIRECTX_NATIVE_LOADER_ADAPTER_V1_H
#define CROSSTL_DIRECTX_NATIVE_LOADER_ADAPTER_V1_H

#ifndef __cplusplus
#error "The CrossTL Direct3D 12 native loader adapter requires C++17"
#endif
#if defined(_MSVC_LANG)
#if _MSVC_LANG < 201703L
#error "The CrossTL Direct3D 12 native loader adapter requires C++17"
#endif
#elif __cplusplus < 201703L
#error "The CrossTL Direct3D 12 native loader adapter requires C++17"
#endif

#ifndef CROSSTL_NATIVE_LOADER_EXECUTION_ABI_V1_TYPES
#error "Include a generated CrossTL native loader execution header first"
#endif

#include <stddef.h>
#include <stdint.h>

typedef enum CrossTLDirectXNativeLoaderStatus {
    CROSSTL_DIRECTX_NATIVE_LOADER_OK = 0,
    CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT = 1001,
    CROSSTL_DIRECTX_NATIVE_LOADER_PLATFORM_UNAVAILABLE = 1002,
    CROSSTL_DIRECTX_NATIVE_LOADER_NOT_INITIALIZED = 1003,
    CROSSTL_DIRECTX_NATIVE_LOADER_ALREADY_INITIALIZED = 1004,
    CROSSTL_DIRECTX_NATIVE_LOADER_HOST_ALLOCATION_FAILED = 1005,
    CROSSTL_DIRECTX_NATIVE_LOADER_INTERNAL_FAILURE = 1006,
    CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_PATH_INVALID = 1101,
    CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_FORMAT_UNSUPPORTED = 1102,
    CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_OPEN_FAILED = 1103,
    CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_READ_FAILED = 1104,
    CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_EMPTY = 1105,
    CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_NOT_DXIL = 1106,
    CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_SIZE_MISMATCH = 1107,
    CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_HASH_UNSUPPORTED = 1108,
    CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_HASH_FAILED = 1109,
    CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_HASH_MISMATCH = 1110,
    CROSSTL_DIRECTX_NATIVE_LOADER_SPECIALIZATION_UNSUPPORTED = 1201,
    CROSSTL_DIRECTX_NATIVE_LOADER_STAGE_UNSUPPORTED = 1301,
    CROSSTL_DIRECTX_NATIVE_LOADER_RESOURCE_KIND_UNSUPPORTED = 1401,
    CROSSTL_DIRECTX_NATIVE_LOADER_BINDING_NAMESPACE_UNSUPPORTED = 1402,
    CROSSTL_DIRECTX_NATIVE_LOADER_BINDING_ACCESS_UNSUPPORTED = 1403,
    CROSSTL_DIRECTX_NATIVE_LOADER_BINDING_NOT_FOUND = 1404,
    CROSSTL_DIRECTX_NATIVE_LOADER_RESOURCE_SIZE_INVALID = 1405,
    CROSSTL_DIRECTX_NATIVE_LOADER_RESOURCE_ALREADY_BOUND = 1406,
    CROSSTL_DIRECTX_NATIVE_LOADER_SCALAR_LAYOUT_UNSUPPORTED = 1407,
    CROSSTL_DIRECTX_NATIVE_LOADER_CONSTANT_BUFFER_TOO_LARGE = 1408,
    CROSSTL_DIRECTX_NATIVE_LOADER_PIPELINE_MISMATCH = 1409,
    CROSSTL_DIRECTX_NATIVE_LOADER_BUFFER_TYPE_UNSUPPORTED = 1410,
    CROSSTL_DIRECTX_NATIVE_LOADER_DEVICE_CREATION_FAILED = 1501,
    CROSSTL_DIRECTX_NATIVE_LOADER_COMMAND_QUEUE_CREATION_FAILED = 1502,
    CROSSTL_DIRECTX_NATIVE_LOADER_FENCE_CREATION_FAILED = 1503,
    CROSSTL_DIRECTX_NATIVE_LOADER_COMMAND_ALLOCATOR_CREATION_FAILED = 1504,
    CROSSTL_DIRECTX_NATIVE_LOADER_COMMAND_LIST_CREATION_FAILED = 1505,
    CROSSTL_DIRECTX_NATIVE_LOADER_ROOT_SIGNATURE_SERIALIZATION_FAILED = 1506,
    CROSSTL_DIRECTX_NATIVE_LOADER_ROOT_SIGNATURE_CREATION_FAILED = 1507,
    CROSSTL_DIRECTX_NATIVE_LOADER_DESCRIPTOR_HEAP_CREATION_FAILED = 1508,
    CROSSTL_DIRECTX_NATIVE_LOADER_PIPELINE_CREATION_FAILED = 1509,
    CROSSTL_DIRECTX_NATIVE_LOADER_BUFFER_CREATION_FAILED = 1510,
    CROSSTL_DIRECTX_NATIVE_LOADER_BUFFER_MAP_FAILED = 1511,
    CROSSTL_DIRECTX_NATIVE_LOADER_COMMAND_RECORDING_FAILED = 1512,
    CROSSTL_DIRECTX_NATIVE_LOADER_COMMAND_SUBMISSION_FAILED = 1513,
    CROSSTL_DIRECTX_NATIVE_LOADER_FENCE_SIGNAL_FAILED = 1514,
    CROSSTL_DIRECTX_NATIVE_LOADER_FENCE_WAIT_FAILED = 1515,
    CROSSTL_DIRECTX_NATIVE_LOADER_DEVICE_REMOVED = 1516,
    CROSSTL_DIRECTX_NATIVE_LOADER_DISPATCH_INVALID = 1601,
    CROSSTL_DIRECTX_NATIVE_LOADER_BINDINGS_INCOMPLETE = 1602,
    CROSSTL_DIRECTX_NATIVE_LOADER_DISPATCH_ALREADY_RECORDED = 1603,
    CROSSTL_DIRECTX_NATIVE_LOADER_DISPATCH_NOT_SUBMITTED = 1604,
    CROSSTL_DIRECTX_NATIVE_LOADER_SYNCHRONIZE_BEFORE_DISPATCH = 1701,
    CROSSTL_DIRECTX_NATIVE_LOADER_READBACK_BEFORE_SYNCHRONIZE = 1801,
    CROSSTL_DIRECTX_NATIVE_LOADER_READBACK_DESTINATION_INVALID = 1802,
    CROSSTL_DIRECTX_NATIVE_LOADER_RESOURCE_NOT_WRITABLE = 1803
} CrossTLDirectXNativeLoaderStatus;

#if !defined(_WIN32)

typedef struct CrossTLDirectXNativeLoaderContext {
    int32_t last_status;
    int32_t last_hresult;
    int32_t initialized;
} CrossTLDirectXNativeLoaderContext;

static inline int32_t crosstl_directx_native_loader_platform_failure(
    CrossTLDirectXNativeLoaderContext *context) {
    if (context != NULL) {
        context->last_status = CROSSTL_DIRECTX_NATIVE_LOADER_PLATFORM_UNAVAILABLE;
        context->last_hresult = 0;
        context->initialized = 0;
    }
    return CROSSTL_DIRECTX_NATIVE_LOADER_PLATFORM_UNAVAILABLE;
}

static inline int32_t crosstl_directx_native_loader_context_initialize(
    CrossTLDirectXNativeLoaderContext *context,
    const char *package_root) {
    (void)package_root;
    if (context == NULL) {
        return CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT;
    }
    return crosstl_directx_native_loader_platform_failure(context);
}

static inline int32_t crosstl_directx_native_loader_context_shutdown(
    CrossTLDirectXNativeLoaderContext *context) {
    if (context == NULL) {
        return CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT;
    }
    return crosstl_directx_native_loader_platform_failure(context);
}

static inline int32_t crosstl_directx_native_loader_last_status(
    const CrossTLDirectXNativeLoaderContext *context) {
    return context == NULL
        ? CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT
        : context->last_status;
}

static inline int32_t crosstl_directx_native_loader_last_hresult(
    const CrossTLDirectXNativeLoaderContext *context) {
    return context == NULL ? 0 : context->last_hresult;
}

static inline int crosstl_directx_native_loader_is_available(void) {
    return 0;
}

static inline int32_t crosstl_directx_native_loader_load_artifact(
    void *context_value,
    const CrossTLNativeLoaderUnitDescriptor *unit,
    void **artifact_out) {
    (void)unit;
    if (artifact_out != NULL) {
        *artifact_out = NULL;
    }
    return crosstl_directx_native_loader_platform_failure(
        (CrossTLDirectXNativeLoaderContext *)context_value);
}

static inline int32_t crosstl_directx_native_loader_unload_artifact(
    void *context_value,
    void *artifact) {
    (void)artifact;
    return crosstl_directx_native_loader_platform_failure(
        (CrossTLDirectXNativeLoaderContext *)context_value);
}

static inline int32_t crosstl_directx_native_loader_create_pipeline(
    void *context_value,
    void *artifact,
    const CrossTLNativeLoaderUnitDescriptor *unit,
    void **pipeline_out) {
    (void)artifact;
    (void)unit;
    if (pipeline_out != NULL) {
        *pipeline_out = NULL;
    }
    return crosstl_directx_native_loader_platform_failure(
        (CrossTLDirectXNativeLoaderContext *)context_value);
}

static inline int32_t crosstl_directx_native_loader_destroy_pipeline(
    void *context_value,
    void *pipeline) {
    (void)pipeline;
    return crosstl_directx_native_loader_platform_failure(
        (CrossTLDirectXNativeLoaderContext *)context_value);
}

static inline int32_t crosstl_directx_native_loader_apply_specialization(
    void *context_value,
    void *artifact,
    const CrossTLNativeLoaderSpecializationDescriptor *descriptor,
    const CrossTLNativeLoaderSpecializationRequest *request) {
    (void)artifact;
    (void)descriptor;
    (void)request;
    return crosstl_directx_native_loader_platform_failure(
        (CrossTLDirectXNativeLoaderContext *)context_value);
}

static inline int32_t crosstl_directx_native_loader_bind_resource(
    void *context_value,
    void *pipeline,
    const CrossTLNativeLoaderBindingDescriptor *descriptor,
    const CrossTLNativeLoaderBindingRequest *request,
    void **resource_out) {
    (void)pipeline;
    (void)descriptor;
    (void)request;
    if (resource_out != NULL) {
        *resource_out = NULL;
    }
    return crosstl_directx_native_loader_platform_failure(
        (CrossTLDirectXNativeLoaderContext *)context_value);
}

static inline int32_t crosstl_directx_native_loader_release_resource(
    void *context_value,
    void *resource,
    const CrossTLNativeLoaderBindingDescriptor *descriptor) {
    (void)resource;
    (void)descriptor;
    return crosstl_directx_native_loader_platform_failure(
        (CrossTLDirectXNativeLoaderContext *)context_value);
}

static inline int32_t crosstl_directx_native_loader_dispatch(
    void *context_value,
    void *pipeline,
    const CrossTLNativeLoaderDispatchGeometry *geometry) {
    (void)pipeline;
    (void)geometry;
    return crosstl_directx_native_loader_platform_failure(
        (CrossTLDirectXNativeLoaderContext *)context_value);
}

static inline int32_t crosstl_directx_native_loader_synchronize(
    void *context_value,
    void *pipeline) {
    (void)pipeline;
    return crosstl_directx_native_loader_platform_failure(
        (CrossTLDirectXNativeLoaderContext *)context_value);
}

static inline int32_t crosstl_directx_native_loader_readback(
    void *context_value,
    void *resource,
    const CrossTLNativeLoaderBindingDescriptor *descriptor,
    const CrossTLNativeLoaderBindingRequest *request) {
    (void)resource;
    (void)descriptor;
    (void)request;
    return crosstl_directx_native_loader_platform_failure(
        (CrossTLDirectXNativeLoaderContext *)context_value);
}

#else

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

#pragma comment(lib, "bcrypt.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;

typedef struct CrossTLDirectXNativeLoaderContext {
    /*
     * The context owns one device, compute queue, and fence. Calls using a
     * context must be serialized. Shut it down only after every pipeline
     * created from its adapter has been destroyed.
     */
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = NULL;
    uint64_t next_fence_value = 1u;
    std::filesystem::path package_root;
    int32_t last_status = CROSSTL_DIRECTX_NATIVE_LOADER_OK;
    HRESULT last_hresult = S_OK;
    bool initialized = false;
} CrossTLDirectXNativeLoaderContext;

typedef struct CrossTLDirectXNativeLoaderArtifact {
    std::vector<uint8_t> dxil;
} CrossTLDirectXNativeLoaderArtifact;

struct CrossTLDirectXNativeLoaderPipeline;

typedef struct CrossTLDirectXNativeLoaderResource {
    CrossTLDirectXNativeLoaderPipeline *pipeline = NULL;
    const CrossTLNativeLoaderBindingDescriptor *descriptor = NULL;
    size_t binding_slot = 0u;
    size_t size_bytes = 0u;
    uint32_t structured_stride = 0u;
    D3D12_DESCRIPTOR_RANGE_TYPE range_type = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    bool writable = false;
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12Resource> device;
    ComPtr<ID3D12Resource> readback;
} CrossTLDirectXNativeLoaderResource;

typedef struct CrossTLDirectXNativeLoaderPipeline {
    CrossTLDirectXNativeLoaderContext *context = NULL;
    const CrossTLNativeLoaderUnitDescriptor *unit = NULL;
    ComPtr<ID3D12RootSignature> root_signature;
    ComPtr<ID3D12PipelineState> pipeline_state;
    ComPtr<ID3D12DescriptorHeap> descriptor_heap;
    ComPtr<ID3D12CommandAllocator> command_allocator;
    ComPtr<ID3D12GraphicsCommandList> command_list;
    std::vector<CrossTLDirectXNativeLoaderResource *> resources;
    uint32_t descriptor_stride = 0u;
    uint64_t fence_value = 0u;
    bool command_list_open = false;
    bool dispatch_recorded = false;
    bool submitted = false;
    bool synchronized = false;
} CrossTLDirectXNativeLoaderPipeline;

static inline int32_t crosstl_directx_native_loader_fail(
    CrossTLDirectXNativeLoaderContext *context,
    CrossTLDirectXNativeLoaderStatus status,
    HRESULT result = S_OK) {
    if (context != NULL) {
        context->last_status = (int32_t)status;
        context->last_hresult = result;
    }
    return (int32_t)status;
}

static inline int32_t crosstl_directx_native_loader_succeed(
    CrossTLDirectXNativeLoaderContext *context) {
    if (context != NULL) {
        context->last_status = CROSSTL_DIRECTX_NATIVE_LOADER_OK;
        context->last_hresult = S_OK;
    }
    return CROSSTL_DIRECTX_NATIVE_LOADER_OK;
}

static inline bool crosstl_directx_native_loader_ascii_equal(
    const char *left,
    const char *right) {
    if (left == NULL || right == NULL) {
        return left == right;
    }
    while (*left != '\0' && *right != '\0') {
        unsigned char left_character = (unsigned char)*left;
        unsigned char right_character = (unsigned char)*right;
        if (std::tolower(left_character) != std::tolower(right_character)) {
            return false;
        }
        ++left;
        ++right;
    }
    return *left == *right;
}

static inline bool crosstl_directx_native_loader_ascii_contains(
    const char *value,
    const char *fragment) {
    if (value == NULL || fragment == NULL || *fragment == '\0') {
        return false;
    }
    size_t value_size = std::strlen(value);
    size_t fragment_size = std::strlen(fragment);
    if (fragment_size > value_size) {
        return false;
    }
    for (size_t offset = 0u; offset + fragment_size <= value_size; ++offset) {
        size_t index = 0u;
        for (; index < fragment_size; ++index) {
            unsigned char value_character = (unsigned char)value[offset + index];
            unsigned char fragment_character = (unsigned char)fragment[index];
            if (std::tolower(value_character) !=
                std::tolower(fragment_character)) {
                break;
            }
        }
        if (index == fragment_size) {
            return true;
        }
    }
    return false;
}

static inline D3D12_HEAP_PROPERTIES
crosstl_directx_native_loader_heap_properties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES properties = {};
    properties.Type = type;
    properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    properties.CreationNodeMask = 1u;
    properties.VisibleNodeMask = 1u;
    return properties;
}

static inline D3D12_RESOURCE_DESC
crosstl_directx_native_loader_buffer_description(
    uint64_t size_bytes,
    D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC description = {};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Alignment = 0u;
    description.Width = size_bytes;
    description.Height = 1u;
    description.DepthOrArraySize = 1u;
    description.MipLevels = 1u;
    description.Format = DXGI_FORMAT_UNKNOWN;
    description.SampleDesc.Count = 1u;
    description.SampleDesc.Quality = 0u;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    description.Flags = flags;
    return description;
}

static inline void crosstl_directx_native_loader_transition(
    ID3D12GraphicsCommandList *command_list,
    ID3D12Resource *resource,
    D3D12_RESOURCE_STATES before,
    D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    command_list->ResourceBarrier(1u, &barrier);
}

static inline int32_t crosstl_directx_native_loader_wait_for_queue(
    CrossTLDirectXNativeLoaderContext *context,
    uint64_t *fence_value_out) {
    if (context == NULL || !context->initialized || context->queue == NULL ||
        context->fence == NULL || context->fence_event == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_NOT_INITIALIZED);
    }
    uint64_t fence_value = context->next_fence_value++;
    if (fence_value == 0u) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_FENCE_SIGNAL_FAILED);
    }
    HRESULT result = context->queue->Signal(context->fence.Get(), fence_value);
    if (FAILED(result)) {
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_FENCE_SIGNAL_FAILED,
            result);
    }
    result = context->fence->SetEventOnCompletion(
        fence_value, context->fence_event);
    if (FAILED(result)) {
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_FENCE_WAIT_FAILED,
            result);
    }
    DWORD wait_status = WaitForSingleObject(context->fence_event, INFINITE);
    if (wait_status != WAIT_OBJECT_0) {
        HRESULT wait_result =
            wait_status == WAIT_FAILED ? HRESULT_FROM_WIN32(GetLastError()) : E_FAIL;
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_FENCE_WAIT_FAILED,
            wait_result);
    }
    result = context->device->GetDeviceRemovedReason();
    if (FAILED(result)) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_DEVICE_REMOVED, result);
    }
    if (fence_value_out != NULL) {
        *fence_value_out = fence_value;
    }
    return crosstl_directx_native_loader_succeed(context);
}

static inline int32_t crosstl_directx_native_loader_context_initialize(
    CrossTLDirectXNativeLoaderContext *context,
    const char *package_root) {
    if (context == NULL) {
        return CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT;
    }
    if (context->initialized) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_ALREADY_INITIALIZED);
    }
    try {
        std::filesystem::path root = package_root == NULL || *package_root == '\0'
            ? std::filesystem::current_path()
            : std::filesystem::u8path(package_root);
        context->package_root = std::filesystem::absolute(root).lexically_normal();
    } catch (const std::bad_alloc &) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_HOST_ALLOCATION_FAILED);
    } catch (...) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_PATH_INVALID);
    }

    ComPtr<IDXGIFactory6> factory;
    HRESULT result = CreateDXGIFactory2(
        0u, IID_PPV_ARGS(factory.ReleaseAndGetAddressOf()));
    if (FAILED(result)) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_DEVICE_CREATION_FAILED, result);
    }

    for (uint32_t index = 0u; ; ++index) {
        ComPtr<IDXGIAdapter1> candidate;
        result = factory->EnumAdapterByGpuPreference(
            index,
            DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            IID_PPV_ARGS(candidate.ReleaseAndGetAddressOf()));
        if (result == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (FAILED(result)) {
            continue;
        }
        DXGI_ADAPTER_DESC1 description = {};
        if (FAILED(candidate->GetDesc1(&description)) ||
            (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0u) {
            continue;
        }
        result = D3D12CreateDevice(
            candidate.Get(),
            D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(context->device.ReleaseAndGetAddressOf()));
        if (SUCCEEDED(result)) {
            break;
        }
    }

    if (context->device == NULL) {
        ComPtr<IDXGIAdapter> warp_adapter;
        result = factory->EnumWarpAdapter(
            IID_PPV_ARGS(warp_adapter.ReleaseAndGetAddressOf()));
        if (SUCCEEDED(result)) {
            result = D3D12CreateDevice(
                warp_adapter.Get(),
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(context->device.ReleaseAndGetAddressOf()));
        }
    }
    if (context->device == NULL || FAILED(result)) {
        context->device.Reset();
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_DEVICE_CREATION_FAILED, result);
    }

    D3D12_COMMAND_QUEUE_DESC queue_description = {};
    queue_description.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queue_description.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queue_description.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queue_description.NodeMask = 0u;
    result = context->device->CreateCommandQueue(
        &queue_description,
        IID_PPV_ARGS(context->queue.ReleaseAndGetAddressOf()));
    if (FAILED(result)) {
        context->device.Reset();
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_COMMAND_QUEUE_CREATION_FAILED,
            result);
    }
    result = context->device->CreateFence(
        0u,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(context->fence.ReleaseAndGetAddressOf()));
    if (FAILED(result)) {
        context->queue.Reset();
        context->device.Reset();
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_FENCE_CREATION_FAILED, result);
    }
    context->fence_event = CreateEventW(NULL, FALSE, FALSE, NULL);
    if (context->fence_event == NULL) {
        HRESULT event_result = HRESULT_FROM_WIN32(GetLastError());
        context->fence.Reset();
        context->queue.Reset();
        context->device.Reset();
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_FENCE_CREATION_FAILED,
            event_result);
    }
    context->next_fence_value = 1u;
    context->initialized = true;
    return crosstl_directx_native_loader_succeed(context);
}

static inline int32_t crosstl_directx_native_loader_context_shutdown(
    CrossTLDirectXNativeLoaderContext *context) {
    if (context == NULL) {
        return CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT;
    }
    int32_t status = CROSSTL_DIRECTX_NATIVE_LOADER_OK;
    if (context->initialized && context->queue != NULL &&
        context->fence != NULL && context->fence_event != NULL) {
        status = crosstl_directx_native_loader_wait_for_queue(context, NULL);
    }
    if (context->fence_event != NULL) {
        CloseHandle(context->fence_event);
        context->fence_event = NULL;
    }
    context->fence.Reset();
    context->queue.Reset();
    context->device.Reset();
    context->initialized = false;
    context->next_fence_value = 1u;
    if (status == CROSSTL_DIRECTX_NATIVE_LOADER_OK) {
        return crosstl_directx_native_loader_succeed(context);
    }
    return status;
}

static inline int32_t crosstl_directx_native_loader_last_status(
    const CrossTLDirectXNativeLoaderContext *context) {
    return context == NULL
        ? CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT
        : context->last_status;
}

static inline int32_t crosstl_directx_native_loader_last_hresult(
    const CrossTLDirectXNativeLoaderContext *context) {
    return context == NULL ? 0 : (int32_t)context->last_hresult;
}

static inline int crosstl_directx_native_loader_is_available(void) {
    return 1;
}

static inline int32_t crosstl_directx_native_loader_verify_sha256(
    CrossTLDirectXNativeLoaderContext *context,
    const std::vector<uint8_t> &content,
    const char *algorithm,
    const char *expected_value) {
    if (!crosstl_directx_native_loader_ascii_equal(algorithm, "sha256") ||
        expected_value == NULL || std::strlen(expected_value) != 64u) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_HASH_UNSUPPORTED);
    }

    BCRYPT_ALG_HANDLE algorithm_handle = NULL;
    BCRYPT_HASH_HANDLE hash_handle = NULL;
    DWORD object_size = 0u;
    DWORD result_size = 0u;
    std::vector<uint8_t> hash_object;
    uint8_t digest[32] = {};
    NTSTATUS native_status = BCryptOpenAlgorithmProvider(
        &algorithm_handle, BCRYPT_SHA256_ALGORITHM, NULL, 0u);
    if (BCRYPT_SUCCESS(native_status)) {
        native_status = BCryptGetProperty(
            algorithm_handle,
            BCRYPT_OBJECT_LENGTH,
            (PUCHAR)&object_size,
            sizeof(object_size),
            &result_size,
            0u);
    }
    try {
        if (BCRYPT_SUCCESS(native_status)) {
            hash_object.resize(object_size);
        }
    } catch (const std::bad_alloc &) {
        if (algorithm_handle != NULL) {
            BCryptCloseAlgorithmProvider(algorithm_handle, 0u);
        }
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_HOST_ALLOCATION_FAILED);
    }
    if (BCRYPT_SUCCESS(native_status)) {
        native_status = BCryptCreateHash(
            algorithm_handle,
            &hash_handle,
            hash_object.data(),
            object_size,
            NULL,
            0u,
            0u);
    }
    size_t offset = 0u;
    while (BCRYPT_SUCCESS(native_status) && offset < content.size()) {
        size_t remaining = content.size() - offset;
        ULONG chunk_size = (ULONG)std::min<size_t>(
            remaining, (size_t)std::numeric_limits<ULONG>::max());
        native_status = BCryptHashData(
            hash_handle,
            (PUCHAR)(content.data() + offset),
            chunk_size,
            0u);
        offset += chunk_size;
    }
    if (BCRYPT_SUCCESS(native_status)) {
        native_status = BCryptFinishHash(
            hash_handle, digest, (ULONG)sizeof(digest), 0u);
    }
    if (hash_handle != NULL) {
        BCryptDestroyHash(hash_handle);
    }
    if (algorithm_handle != NULL) {
        BCryptCloseAlgorithmProvider(algorithm_handle, 0u);
    }
    if (!BCRYPT_SUCCESS(native_status)) {
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_HASH_FAILED,
            HRESULT_FROM_NT(native_status));
    }

    static const char hexadecimal[] = "0123456789abcdef";
    char actual_value[65] = {};
    for (size_t index = 0u; index < sizeof(digest); ++index) {
        actual_value[index * 2u] = hexadecimal[digest[index] >> 4u];
        actual_value[index * 2u + 1u] = hexadecimal[digest[index] & 0x0fu];
    }
    if (!crosstl_directx_native_loader_ascii_equal(
            actual_value, expected_value)) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_HASH_MISMATCH);
    }
    return crosstl_directx_native_loader_succeed(context);
}

static inline bool crosstl_directx_native_loader_safe_relative_path(
    const std::filesystem::path &path) {
    if (path.empty() || path.is_absolute() || path.has_root_name() ||
        path.has_root_directory()) {
        return false;
    }
    for (const std::filesystem::path &component : path) {
        if (component == "..") {
            return false;
        }
    }
    return true;
}

static inline int32_t crosstl_directx_native_loader_load_artifact(
    void *context_value,
    const CrossTLNativeLoaderUnitDescriptor *unit,
    void **artifact_out) {
    CrossTLDirectXNativeLoaderContext *context =
        (CrossTLDirectXNativeLoaderContext *)context_value;
    if (artifact_out != NULL) {
        *artifact_out = NULL;
    }
    if (context == NULL || unit == NULL || artifact_out == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT);
    }
    if (!context->initialized) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_NOT_INITIALIZED);
    }
    if (unit->artifact_path == NULL ||
        unit->artifact_hash_algorithm == NULL ||
        unit->artifact_hash_value == NULL) {
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_PATH_INVALID);
    }
    if (!crosstl_directx_native_loader_ascii_contains(
            unit->artifact_format, "dxil")) {
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_FORMAT_UNSUPPORTED);
    }

    try {
        std::filesystem::path relative_path =
            std::filesystem::u8path(unit->artifact_path);
        if (!crosstl_directx_native_loader_safe_relative_path(relative_path)) {
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_PATH_INVALID);
        }
        std::filesystem::path artifact_path =
            (context->package_root / relative_path).lexically_normal();
        std::ifstream stream(artifact_path, std::ios::binary | std::ios::ate);
        if (!stream.is_open()) {
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_OPEN_FAILED);
        }
        std::streamoff length = stream.tellg();
        if (length <= 0) {
            return crosstl_directx_native_loader_fail(
                context, CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_EMPTY);
        }
        if ((uint64_t)length != unit->artifact_size_bytes ||
            (uint64_t)length >
                (uint64_t)std::numeric_limits<size_t>::max()) {
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_SIZE_MISMATCH);
        }

        std::unique_ptr<CrossTLDirectXNativeLoaderArtifact> artifact(
            new (std::nothrow) CrossTLDirectXNativeLoaderArtifact());
        if (!artifact) {
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_HOST_ALLOCATION_FAILED);
        }
        artifact->dxil.resize((size_t)length);
        stream.seekg(0, std::ios::beg);
        if (!stream.read(
                (char *)artifact->dxil.data(),
                (std::streamsize)artifact->dxil.size())) {
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_READ_FAILED);
        }
        if (artifact->dxil.size() < 4u ||
            std::memcmp(artifact->dxil.data(), "DXBC", 4u) != 0) {
            return crosstl_directx_native_loader_fail(
                context, CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_NOT_DXIL);
        }
        int32_t status = crosstl_directx_native_loader_verify_sha256(
            context,
            artifact->dxil,
            unit->artifact_hash_algorithm,
            unit->artifact_hash_value);
        if (status != CROSSTL_DIRECTX_NATIVE_LOADER_OK) {
            return status;
        }
        *artifact_out = artifact.release();
        return crosstl_directx_native_loader_succeed(context);
    } catch (const std::bad_alloc &) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_HOST_ALLOCATION_FAILED);
    } catch (...) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_INTERNAL_FAILURE);
    }
}

static inline int32_t crosstl_directx_native_loader_unload_artifact(
    void *context_value,
    void *artifact_value) {
    CrossTLDirectXNativeLoaderContext *context =
        (CrossTLDirectXNativeLoaderContext *)context_value;
    if (context == NULL || artifact_value == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT);
    }
    delete (CrossTLDirectXNativeLoaderArtifact *)artifact_value;
    return crosstl_directx_native_loader_succeed(context);
}

static inline int32_t crosstl_directx_native_loader_binding_range_type(
    CrossTLDirectXNativeLoaderContext *context,
    const CrossTLNativeLoaderBindingDescriptor *binding,
    D3D12_DESCRIPTOR_RANGE_TYPE *range_type_out) {
    if (binding == NULL || range_type_out == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT);
    }
    bool is_buffer =
        crosstl_directx_native_loader_ascii_equal(
            binding->resource_kind, "buffer");
    bool is_constant_buffer =
        crosstl_directx_native_loader_ascii_equal(
            binding->resource_kind, "constant-buffer");
    if (!is_buffer && !is_constant_buffer) {
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_RESOURCE_KIND_UNSUPPORTED);
    }
    if (is_buffer &&
        !crosstl_directx_native_loader_ascii_contains(
            binding->type_name, "structuredbuffer")) {
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_BUFFER_TYPE_UNSUPPORTED);
    }
    if (crosstl_directx_native_loader_ascii_equal(
            binding->binding_namespace, "cbv")) {
        if (!is_constant_buffer ||
            binding->access != CROSSTL_NATIVE_LOADER_ACCESS_READ) {
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_BINDING_ACCESS_UNSUPPORTED);
        }
        *range_type_out = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
        return CROSSTL_DIRECTX_NATIVE_LOADER_OK;
    }
    if (crosstl_directx_native_loader_ascii_equal(
            binding->binding_namespace, "srv")) {
        if (!is_buffer ||
            binding->access != CROSSTL_NATIVE_LOADER_ACCESS_READ) {
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_BINDING_ACCESS_UNSUPPORTED);
        }
        *range_type_out = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        return CROSSTL_DIRECTX_NATIVE_LOADER_OK;
    }
    if (crosstl_directx_native_loader_ascii_equal(
            binding->binding_namespace, "uav")) {
        if (!is_buffer ||
            (binding->access != CROSSTL_NATIVE_LOADER_ACCESS_WRITE &&
             binding->access != CROSSTL_NATIVE_LOADER_ACCESS_READ_WRITE)) {
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_BINDING_ACCESS_UNSUPPORTED);
        }
        *range_type_out = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        return CROSSTL_DIRECTX_NATIVE_LOADER_OK;
    }
    return crosstl_directx_native_loader_fail(
        context,
        CROSSTL_DIRECTX_NATIVE_LOADER_BINDING_NAMESPACE_UNSUPPORTED);
}

static inline int32_t crosstl_directx_native_loader_create_pipeline(
    void *context_value,
    void *artifact_value,
    const CrossTLNativeLoaderUnitDescriptor *unit,
    void **pipeline_out) {
    CrossTLDirectXNativeLoaderContext *context =
        (CrossTLDirectXNativeLoaderContext *)context_value;
    if (pipeline_out != NULL) {
        *pipeline_out = NULL;
    }
    if (context == NULL || artifact_value == NULL || unit == NULL ||
        pipeline_out == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT);
    }
    if (!context->initialized || context->device == NULL ||
        context->queue == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_NOT_INITIALIZED);
    }
    if (!crosstl_directx_native_loader_ascii_equal(unit->target, "directx") ||
        !crosstl_directx_native_loader_ascii_equal(unit->stage, "compute") ||
        unit->entry_point == NULL || *unit->entry_point == '\0') {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_STAGE_UNSUPPORTED);
    }
    if (unit->binding_count > (size_t)std::numeric_limits<UINT>::max() ||
        (unit->binding_count != 0u && unit->bindings == NULL)) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT);
    }

    CrossTLDirectXNativeLoaderArtifact *artifact =
        (CrossTLDirectXNativeLoaderArtifact *)artifact_value;
    if (artifact->dxil.empty()) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_EMPTY);
    }
    CrossTLDirectXNativeLoaderPipeline *pipeline =
        new (std::nothrow) CrossTLDirectXNativeLoaderPipeline();
    if (pipeline == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_HOST_ALLOCATION_FAILED);
    }
    pipeline->context = context;
    pipeline->unit = unit;

    try {
        std::vector<D3D12_DESCRIPTOR_RANGE> ranges(unit->binding_count);
        std::vector<D3D12_ROOT_PARAMETER> parameters(unit->binding_count);
        pipeline->resources.resize(unit->binding_count, NULL);
        for (size_t index = 0u; index < unit->binding_count; ++index) {
            D3D12_DESCRIPTOR_RANGE_TYPE range_type;
            int32_t status = crosstl_directx_native_loader_binding_range_type(
                context, &unit->bindings[index], &range_type);
            if (status != CROSSTL_DIRECTX_NATIVE_LOADER_OK) {
                delete pipeline;
                return status;
            }
            ranges[index].RangeType = range_type;
            ranges[index].NumDescriptors = 1u;
            ranges[index].BaseShaderRegister =
                unit->bindings[index].binding_index;
            ranges[index].RegisterSpace = unit->bindings[index].set_index;
            ranges[index].OffsetInDescriptorsFromTableStart =
                D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

            parameters[index].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[index].DescriptorTable.NumDescriptorRanges = 1u;
            parameters[index].DescriptorTable.pDescriptorRanges = &ranges[index];
            parameters[index].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }

        D3D12_ROOT_SIGNATURE_DESC root_description = {};
        root_description.NumParameters = (UINT)parameters.size();
        root_description.pParameters =
            parameters.empty() ? NULL : parameters.data();
        root_description.NumStaticSamplers = 0u;
        root_description.pStaticSamplers = NULL;
        root_description.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

        ComPtr<ID3DBlob> serialized_root;
        ComPtr<ID3DBlob> root_errors;
        HRESULT result = D3D12SerializeRootSignature(
            &root_description,
            D3D_ROOT_SIGNATURE_VERSION_1,
            serialized_root.ReleaseAndGetAddressOf(),
            root_errors.ReleaseAndGetAddressOf());
        if (FAILED(result)) {
            delete pipeline;
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_ROOT_SIGNATURE_SERIALIZATION_FAILED,
                result);
        }
        result = context->device->CreateRootSignature(
            0u,
            serialized_root->GetBufferPointer(),
            serialized_root->GetBufferSize(),
            IID_PPV_ARGS(pipeline->root_signature.ReleaseAndGetAddressOf()));
        if (FAILED(result)) {
            delete pipeline;
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_ROOT_SIGNATURE_CREATION_FAILED,
                result);
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_description = {};
        pipeline_description.pRootSignature = pipeline->root_signature.Get();
        pipeline_description.CS.pShaderBytecode = artifact->dxil.data();
        pipeline_description.CS.BytecodeLength = artifact->dxil.size();
        pipeline_description.NodeMask = 0u;
        pipeline_description.CachedPSO.pCachedBlob = NULL;
        pipeline_description.CachedPSO.CachedBlobSizeInBytes = 0u;
        pipeline_description.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
        result = context->device->CreateComputePipelineState(
            &pipeline_description,
            IID_PPV_ARGS(pipeline->pipeline_state.ReleaseAndGetAddressOf()));
        if (FAILED(result)) {
            delete pipeline;
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_PIPELINE_CREATION_FAILED,
                result);
        }

        if (unit->binding_count != 0u) {
            D3D12_DESCRIPTOR_HEAP_DESC heap_description = {};
            heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            heap_description.NumDescriptors = (UINT)unit->binding_count;
            heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            heap_description.NodeMask = 0u;
            result = context->device->CreateDescriptorHeap(
                &heap_description,
                IID_PPV_ARGS(
                    pipeline->descriptor_heap.ReleaseAndGetAddressOf()));
            if (FAILED(result)) {
                delete pipeline;
                return crosstl_directx_native_loader_fail(
                    context,
                    CROSSTL_DIRECTX_NATIVE_LOADER_DESCRIPTOR_HEAP_CREATION_FAILED,
                    result);
            }
            pipeline->descriptor_stride =
                context->device->GetDescriptorHandleIncrementSize(
                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }

        result = context->device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_COMPUTE,
            IID_PPV_ARGS(
                pipeline->command_allocator.ReleaseAndGetAddressOf()));
        if (FAILED(result)) {
            delete pipeline;
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_COMMAND_ALLOCATOR_CREATION_FAILED,
                result);
        }
        result = context->device->CreateCommandList(
            0u,
            D3D12_COMMAND_LIST_TYPE_COMPUTE,
            pipeline->command_allocator.Get(),
            pipeline->pipeline_state.Get(),
            IID_PPV_ARGS(pipeline->command_list.ReleaseAndGetAddressOf()));
        if (FAILED(result)) {
            delete pipeline;
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_COMMAND_LIST_CREATION_FAILED,
                result);
        }
        pipeline->command_list_open = true;
        *pipeline_out = pipeline;
        return crosstl_directx_native_loader_succeed(context);
    } catch (const std::bad_alloc &) {
        delete pipeline;
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_HOST_ALLOCATION_FAILED);
    } catch (...) {
        delete pipeline;
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_INTERNAL_FAILURE);
    }
}

static inline int32_t crosstl_directx_native_loader_apply_specialization(
    void *context_value,
    void *artifact,
    const CrossTLNativeLoaderSpecializationDescriptor *descriptor,
    const CrossTLNativeLoaderSpecializationRequest *request) {
    CrossTLDirectXNativeLoaderContext *context =
        (CrossTLDirectXNativeLoaderContext *)context_value;
    if (context == NULL || artifact == NULL || descriptor == NULL ||
        request == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT);
    }
    return crosstl_directx_native_loader_fail(
        context,
        CROSSTL_DIRECTX_NATIVE_LOADER_SPECIALIZATION_UNSUPPORTED);
}

static inline size_t crosstl_directx_native_loader_binding_slot(
    const CrossTLDirectXNativeLoaderPipeline *pipeline,
    const CrossTLNativeLoaderBindingDescriptor *descriptor) {
    if (pipeline == NULL || pipeline->unit == NULL || descriptor == NULL) {
        return (size_t)-1;
    }
    for (size_t index = 0u; index < pipeline->unit->binding_count; ++index) {
        if (&pipeline->unit->bindings[index] == descriptor) {
            return index;
        }
    }
    return (size_t)-1;
}

static inline bool crosstl_directx_native_loader_json_uint32(
    const char *json,
    uint32_t *value_out) {
    if (json == NULL || value_out == NULL) {
        return false;
    }
    const char *key = "\"elementStrideBytes\"";
    const char *position = std::strstr(json, key);
    if (position == NULL) {
        return false;
    }
    position += std::strlen(key);
    while (*position != '\0' &&
           std::isspace((unsigned char)*position) != 0) {
        ++position;
    }
    if (*position++ != ':') {
        return false;
    }
    while (*position != '\0' &&
           std::isspace((unsigned char)*position) != 0) {
        ++position;
    }
    if (!std::isdigit((unsigned char)*position)) {
        return false;
    }
    uint64_t value = 0u;
    do {
        value = value * 10u + (uint64_t)(*position - '0');
        if (value > (uint64_t)std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        ++position;
    } while (std::isdigit((unsigned char)*position));
    *value_out = (uint32_t)value;
    return true;
}

static inline int32_t crosstl_directx_native_loader_create_committed_buffer(
    CrossTLDirectXNativeLoaderContext *context,
    uint64_t size_bytes,
    D3D12_HEAP_TYPE heap_type,
    D3D12_RESOURCE_FLAGS flags,
    D3D12_RESOURCE_STATES initial_state,
    ComPtr<ID3D12Resource> *resource_out) {
    D3D12_HEAP_PROPERTIES properties =
        crosstl_directx_native_loader_heap_properties(heap_type);
    D3D12_RESOURCE_DESC description =
        crosstl_directx_native_loader_buffer_description(size_bytes, flags);
    HRESULT result = context->device->CreateCommittedResource(
        &properties,
        D3D12_HEAP_FLAG_NONE,
        &description,
        initial_state,
        NULL,
        IID_PPV_ARGS(resource_out->ReleaseAndGetAddressOf()));
    if (FAILED(result)) {
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_BUFFER_CREATION_FAILED,
            result);
    }
    return CROSSTL_DIRECTX_NATIVE_LOADER_OK;
}

static inline int32_t crosstl_directx_native_loader_upload_payload(
    CrossTLDirectXNativeLoaderContext *context,
    ID3D12Resource *resource,
    size_t allocation_size,
    const void *payload,
    size_t payload_size) {
    void *mapped = NULL;
    D3D12_RANGE read_range = {0u, 0u};
    HRESULT result = resource->Map(0u, &read_range, &mapped);
    if (FAILED(result) || mapped == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_BUFFER_MAP_FAILED, result);
    }
    std::memset(mapped, 0, allocation_size);
    std::memcpy(mapped, payload, payload_size);
    D3D12_RANGE written_range = {0u, payload_size};
    resource->Unmap(0u, &written_range);
    return CROSSTL_DIRECTX_NATIVE_LOADER_OK;
}

static inline int32_t crosstl_directx_native_loader_bind_resource(
    void *context_value,
    void *pipeline_value,
    const CrossTLNativeLoaderBindingDescriptor *descriptor,
    const CrossTLNativeLoaderBindingRequest *request,
    void **resource_out) {
    CrossTLDirectXNativeLoaderContext *context =
        (CrossTLDirectXNativeLoaderContext *)context_value;
    CrossTLDirectXNativeLoaderPipeline *pipeline =
        (CrossTLDirectXNativeLoaderPipeline *)pipeline_value;
    if (resource_out != NULL) {
        *resource_out = NULL;
    }
    if (context == NULL || pipeline == NULL || descriptor == NULL ||
        request == NULL || resource_out == NULL || request->payload == NULL ||
        request->payload_size_bytes == 0u) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT);
    }
    if (pipeline->context != context || !pipeline->command_list_open ||
        pipeline->submitted) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_PIPELINE_MISMATCH);
    }
    size_t slot =
        crosstl_directx_native_loader_binding_slot(pipeline, descriptor);
    if (slot == (size_t)-1) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_BINDING_NOT_FOUND);
    }
    if (pipeline->resources[slot] != NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_RESOURCE_ALREADY_BOUND);
    }

    D3D12_DESCRIPTOR_RANGE_TYPE range_type;
    int32_t status = crosstl_directx_native_loader_binding_range_type(
        context, descriptor, &range_type);
    if (status != CROSSTL_DIRECTX_NATIVE_LOADER_OK) {
        return status;
    }
    CrossTLDirectXNativeLoaderResource *resource =
        new (std::nothrow) CrossTLDirectXNativeLoaderResource();
    if (resource == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_HOST_ALLOCATION_FAILED);
    }
    resource->pipeline = pipeline;
    resource->descriptor = descriptor;
    resource->binding_slot = slot;
    resource->size_bytes = request->payload_size_bytes;
    resource->range_type = range_type;
    resource->writable = range_type == D3D12_DESCRIPTOR_RANGE_TYPE_UAV;

    uint64_t allocation_size = (uint64_t)resource->size_bytes;
    if (range_type == D3D12_DESCRIPTOR_RANGE_TYPE_CBV) {
        allocation_size = (allocation_size + 255u) & ~UINT64_C(255);
        if (allocation_size > D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT * 16u) {
            delete resource;
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_CONSTANT_BUFFER_TOO_LARGE);
        }
        status = crosstl_directx_native_loader_create_committed_buffer(
            context,
            allocation_size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            &resource->device);
        if (status == CROSSTL_DIRECTX_NATIVE_LOADER_OK) {
            status = crosstl_directx_native_loader_upload_payload(
                context,
                resource->device.Get(),
                (size_t)allocation_size,
                request->payload,
                request->payload_size_bytes);
        }
    } else {
        uint32_t stride = 0u;
        if (!crosstl_directx_native_loader_json_uint32(
                descriptor->scalar_layout_json,
                &stride) ||
            stride < 4u || stride > D3D12_REQ_MULTI_ELEMENT_STRUCTURE_SIZE_IN_BYTES ||
            stride % 4u != 0u || resource->size_bytes % stride != 0u ||
            resource->size_bytes / stride >
                (size_t)std::numeric_limits<UINT>::max()) {
            delete resource;
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_SCALAR_LAYOUT_UNSUPPORTED);
        }
        resource->structured_stride = stride;
        status = crosstl_directx_native_loader_create_committed_buffer(
            context,
            allocation_size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            &resource->upload);
        if (status == CROSSTL_DIRECTX_NATIVE_LOADER_OK) {
            status = crosstl_directx_native_loader_upload_payload(
                context,
                resource->upload.Get(),
                resource->size_bytes,
                request->payload,
                request->payload_size_bytes);
        }
        D3D12_RESOURCE_FLAGS flags = resource->writable
            ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            : D3D12_RESOURCE_FLAG_NONE;
        if (status == CROSSTL_DIRECTX_NATIVE_LOADER_OK) {
            status = crosstl_directx_native_loader_create_committed_buffer(
                context,
                allocation_size,
                D3D12_HEAP_TYPE_DEFAULT,
                flags,
                D3D12_RESOURCE_STATE_COPY_DEST,
                &resource->device);
        }
        if (status == CROSSTL_DIRECTX_NATIVE_LOADER_OK &&
            resource->writable) {
            status = crosstl_directx_native_loader_create_committed_buffer(
                context,
                allocation_size,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
                &resource->readback);
        }
        if (status == CROSSTL_DIRECTX_NATIVE_LOADER_OK) {
            pipeline->command_list->CopyBufferRegion(
                resource->device.Get(),
                0u,
                resource->upload.Get(),
                0u,
                allocation_size);
            D3D12_RESOURCE_STATES target_state = resource->writable
                ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            crosstl_directx_native_loader_transition(
                pipeline->command_list.Get(),
                resource->device.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                target_state);
            resource->state = target_state;
        }
    }
    if (status != CROSSTL_DIRECTX_NATIVE_LOADER_OK) {
        delete resource;
        return status;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle =
        pipeline->descriptor_heap->GetCPUDescriptorHandleForHeapStart();
    cpu_handle.ptr += slot * pipeline->descriptor_stride;
    if (range_type == D3D12_DESCRIPTOR_RANGE_TYPE_CBV) {
        D3D12_CONSTANT_BUFFER_VIEW_DESC view = {};
        view.BufferLocation = resource->device->GetGPUVirtualAddress();
        view.SizeInBytes = (UINT)allocation_size;
        context->device->CreateConstantBufferView(&view, cpu_handle);
    } else if (range_type == D3D12_DESCRIPTOR_RANGE_TYPE_SRV) {
        D3D12_SHADER_RESOURCE_VIEW_DESC view = {};
        view.Format = DXGI_FORMAT_UNKNOWN;
        view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Buffer.FirstElement = 0u;
        view.Buffer.NumElements =
            (UINT)(resource->size_bytes / resource->structured_stride);
        view.Buffer.StructureByteStride = resource->structured_stride;
        view.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        context->device->CreateShaderResourceView(
            resource->device.Get(), &view, cpu_handle);
    } else {
        D3D12_UNORDERED_ACCESS_VIEW_DESC view = {};
        view.Format = DXGI_FORMAT_UNKNOWN;
        view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        view.Buffer.FirstElement = 0u;
        view.Buffer.NumElements =
            (UINT)(resource->size_bytes / resource->structured_stride);
        view.Buffer.StructureByteStride = resource->structured_stride;
        view.Buffer.CounterOffsetInBytes = 0u;
        view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
        context->device->CreateUnorderedAccessView(
            resource->device.Get(), NULL, &view, cpu_handle);
    }

    pipeline->resources[slot] = resource;
    *resource_out = resource;
    return crosstl_directx_native_loader_succeed(context);
}

static inline int32_t crosstl_directx_native_loader_dispatch(
    void *context_value,
    void *pipeline_value,
    const CrossTLNativeLoaderDispatchGeometry *geometry) {
    CrossTLDirectXNativeLoaderContext *context =
        (CrossTLDirectXNativeLoaderContext *)context_value;
    CrossTLDirectXNativeLoaderPipeline *pipeline =
        (CrossTLDirectXNativeLoaderPipeline *)pipeline_value;
    if (context == NULL || pipeline == NULL || geometry == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT);
    }
    if (pipeline->context != context || !pipeline->command_list_open ||
        pipeline->command_list == NULL || pipeline->pipeline_state == NULL ||
        pipeline->root_signature == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_PIPELINE_MISMATCH);
    }
    if (pipeline->dispatch_recorded || pipeline->submitted) {
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_DISPATCH_ALREADY_RECORDED);
    }
    for (size_t index = 0u; index < 3u; ++index) {
        if (geometry->workgroup_count[index] == 0u ||
            geometry->workgroup_count[index] >
                D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION) {
            return crosstl_directx_native_loader_fail(
                context, CROSSTL_DIRECTX_NATIVE_LOADER_DISPATCH_INVALID);
        }
    }
    for (CrossTLDirectXNativeLoaderResource *resource :
         pipeline->resources) {
        if (resource == NULL) {
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_BINDINGS_INCOMPLETE);
        }
    }

    pipeline->command_list->SetPipelineState(pipeline->pipeline_state.Get());
    pipeline->command_list->SetComputeRootSignature(
        pipeline->root_signature.Get());
    if (pipeline->descriptor_heap != NULL) {
        ID3D12DescriptorHeap *heaps[] = {pipeline->descriptor_heap.Get()};
        pipeline->command_list->SetDescriptorHeaps(1u, heaps);
        D3D12_GPU_DESCRIPTOR_HANDLE base =
            pipeline->descriptor_heap->GetGPUDescriptorHandleForHeapStart();
        for (size_t index = 0u; index < pipeline->resources.size(); ++index) {
            D3D12_GPU_DESCRIPTOR_HANDLE handle = base;
            handle.ptr += index * pipeline->descriptor_stride;
            pipeline->command_list->SetComputeRootDescriptorTable(
                (UINT)index, handle);
        }
    }

    pipeline->command_list->Dispatch(
        geometry->workgroup_count[0],
        geometry->workgroup_count[1],
        geometry->workgroup_count[2]);
    pipeline->dispatch_recorded = true;

    for (CrossTLDirectXNativeLoaderResource *resource :
         pipeline->resources) {
        if (!resource->writable) {
            continue;
        }
        D3D12_RESOURCE_BARRIER unordered_access_barrier = {};
        unordered_access_barrier.Type =
            D3D12_RESOURCE_BARRIER_TYPE_UAV;
        unordered_access_barrier.Flags =
            D3D12_RESOURCE_BARRIER_FLAG_NONE;
        unordered_access_barrier.UAV.pResource = resource->device.Get();
        pipeline->command_list->ResourceBarrier(
            1u, &unordered_access_barrier);
        crosstl_directx_native_loader_transition(
            pipeline->command_list.Get(),
            resource->device.Get(),
            resource->state,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        resource->state = D3D12_RESOURCE_STATE_COPY_SOURCE;
        pipeline->command_list->CopyBufferRegion(
            resource->readback.Get(),
            0u,
            resource->device.Get(),
            0u,
            (uint64_t)resource->size_bytes);
    }

    HRESULT result = pipeline->command_list->Close();
    if (FAILED(result)) {
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_COMMAND_RECORDING_FAILED,
            result);
    }
    pipeline->command_list_open = false;
    ID3D12CommandList *command_lists[] = {pipeline->command_list.Get()};
    context->queue->ExecuteCommandLists(1u, command_lists);
    pipeline->submitted = true;
    return crosstl_directx_native_loader_succeed(context);
}

static inline int32_t crosstl_directx_native_loader_synchronize(
    void *context_value,
    void *pipeline_value) {
    CrossTLDirectXNativeLoaderContext *context =
        (CrossTLDirectXNativeLoaderContext *)context_value;
    CrossTLDirectXNativeLoaderPipeline *pipeline =
        (CrossTLDirectXNativeLoaderPipeline *)pipeline_value;
    if (context == NULL || pipeline == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT);
    }
    if (pipeline->context != context) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_PIPELINE_MISMATCH);
    }
    if (!pipeline->dispatch_recorded || !pipeline->submitted) {
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_SYNCHRONIZE_BEFORE_DISPATCH);
    }
    if (pipeline->synchronized) {
        return crosstl_directx_native_loader_succeed(context);
    }
    int32_t status = crosstl_directx_native_loader_wait_for_queue(
        context, &pipeline->fence_value);
    if (status == CROSSTL_DIRECTX_NATIVE_LOADER_OK) {
        pipeline->synchronized = true;
    }
    return status;
}

static inline int32_t crosstl_directx_native_loader_readback(
    void *context_value,
    void *resource_value,
    const CrossTLNativeLoaderBindingDescriptor *descriptor,
    const CrossTLNativeLoaderBindingRequest *request) {
    CrossTLDirectXNativeLoaderContext *context =
        (CrossTLDirectXNativeLoaderContext *)context_value;
    CrossTLDirectXNativeLoaderResource *resource =
        (CrossTLDirectXNativeLoaderResource *)resource_value;
    if (context == NULL || resource == NULL || descriptor == NULL ||
        request == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT);
    }
    if (resource->pipeline == NULL ||
        resource->pipeline->context != context ||
        resource->descriptor != descriptor) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_PIPELINE_MISMATCH);
    }
    if (!resource->writable || resource->readback == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_RESOURCE_NOT_WRITABLE);
    }
    if (!resource->pipeline->synchronized) {
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_READBACK_BEFORE_SYNCHRONIZE);
    }
    if (request->payload == NULL ||
        request->payload_size_bytes < resource->size_bytes) {
        return crosstl_directx_native_loader_fail(
            context,
            CROSSTL_DIRECTX_NATIVE_LOADER_READBACK_DESTINATION_INVALID);
    }

    void *mapped = NULL;
    D3D12_RANGE read_range = {0u, resource->size_bytes};
    HRESULT result = resource->readback->Map(0u, &read_range, &mapped);
    if (FAILED(result) || mapped == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_BUFFER_MAP_FAILED, result);
    }
    std::memcpy(request->payload, mapped, resource->size_bytes);
    D3D12_RANGE written_range = {0u, 0u};
    resource->readback->Unmap(0u, &written_range);
    return crosstl_directx_native_loader_succeed(context);
}

static inline int32_t crosstl_directx_native_loader_release_resource(
    void *context_value,
    void *resource_value,
    const CrossTLNativeLoaderBindingDescriptor *descriptor) {
    CrossTLDirectXNativeLoaderContext *context =
        (CrossTLDirectXNativeLoaderContext *)context_value;
    CrossTLDirectXNativeLoaderResource *resource =
        (CrossTLDirectXNativeLoaderResource *)resource_value;
    if (context == NULL || resource == NULL || descriptor == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT);
    }
    CrossTLDirectXNativeLoaderPipeline *pipeline = resource->pipeline;
    if (pipeline == NULL || pipeline->context != context ||
        resource->descriptor != descriptor ||
        resource->binding_slot >= pipeline->resources.size() ||
        pipeline->resources[resource->binding_slot] != resource) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_PIPELINE_MISMATCH);
    }
    if (pipeline->submitted && !pipeline->synchronized) {
        int32_t status =
            crosstl_directx_native_loader_synchronize(context, pipeline);
        if (status != CROSSTL_DIRECTX_NATIVE_LOADER_OK) {
            return status;
        }
    }
    pipeline->resources[resource->binding_slot] = NULL;
    delete resource;
    return crosstl_directx_native_loader_succeed(context);
}

static inline int32_t crosstl_directx_native_loader_destroy_pipeline(
    void *context_value,
    void *pipeline_value) {
    CrossTLDirectXNativeLoaderContext *context =
        (CrossTLDirectXNativeLoaderContext *)context_value;
    CrossTLDirectXNativeLoaderPipeline *pipeline =
        (CrossTLDirectXNativeLoaderPipeline *)pipeline_value;
    if (context == NULL || pipeline == NULL) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_INVALID_ARGUMENT);
    }
    if (pipeline->context != context) {
        return crosstl_directx_native_loader_fail(
            context, CROSSTL_DIRECTX_NATIVE_LOADER_PIPELINE_MISMATCH);
    }
    if (pipeline->submitted && !pipeline->synchronized) {
        int32_t status =
            crosstl_directx_native_loader_synchronize(context, pipeline);
        if (status != CROSSTL_DIRECTX_NATIVE_LOADER_OK) {
            return status;
        }
    }
    for (CrossTLDirectXNativeLoaderResource *resource :
         pipeline->resources) {
        if (resource != NULL) {
            return crosstl_directx_native_loader_fail(
                context,
                CROSSTL_DIRECTX_NATIVE_LOADER_BINDINGS_INCOMPLETE);
        }
    }
    delete pipeline;
    return crosstl_directx_native_loader_succeed(context);
}

#endif

static inline CrossTLNativeLoaderAdapter
crosstl_directx_native_loader_adapter(
    CrossTLDirectXNativeLoaderContext *context) {
    CrossTLNativeLoaderAdapter adapter = {};
    adapter.abi_version = CROSSTL_NATIVE_LOADER_ABI_VERSION;
    adapter.target = "directx";
    adapter.context = context;
    adapter.load_artifact = crosstl_directx_native_loader_load_artifact;
    adapter.unload_artifact = crosstl_directx_native_loader_unload_artifact;
    adapter.create_pipeline = crosstl_directx_native_loader_create_pipeline;
    adapter.destroy_pipeline = crosstl_directx_native_loader_destroy_pipeline;
    adapter.apply_specialization =
        crosstl_directx_native_loader_apply_specialization;
    adapter.bind_resource = crosstl_directx_native_loader_bind_resource;
    adapter.release_resource = crosstl_directx_native_loader_release_resource;
    adapter.dispatch = crosstl_directx_native_loader_dispatch;
    adapter.synchronize = crosstl_directx_native_loader_synchronize;
    adapter.readback = crosstl_directx_native_loader_readback;
    return adapter;
}

#endif /* CROSSTL_DIRECTX_NATIVE_LOADER_ADAPTER_V1_H */
"""
