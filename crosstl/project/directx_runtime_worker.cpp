// Bounded Direct3D execution over shared allocations and explicit buffer views.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
class Progress {
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
public:
    void enter(const char* phase, uint32_t dispatch = UINT32_MAX) const {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started).count();
        std::cerr << "CROSSTL_DIRECTX_PROGRESS {\"phase\":\"" << phase
                  << "\",\"dispatchIndex\":";
        if (dispatch == UINT32_MAX) std::cerr << "null";
        else std::cerr << dispatch;
        std::cerr << ",\"elapsedMilliseconds\":" << elapsed << "}\n" << std::flush;
    }
};

// A view's kind holds the resource kind in bits 0-1 (0 constant buffer, 1 shader resource, 2 unordered access,
// 3 static sampler) and these flags: 4 a raw buffer view, 8 an extension record follows the view, 16 the view is
// the next descriptor of the previous view's range, as the elements of a resource array are.
//
// The extension record is eight 32-bit words: the DXGI format of a typed buffer or texture view (for a sampler,
// 1 when it is a comparison sampler); the bytes of one of its elements or texels; the texture dimension (0 for a
// buffer, then 1D, 1D array, 2D, 2D array, 3D, cube and cube array); the width, height and array size or depth of
// a texture (a cube's array size counts its faces); one more than the allocation that holds the UAV counter of a
// structured buffer, or 0; and a reserved zero. A texture's allocation holds its texels with no row padding, one
// subresource after another in array order, and it has one mip level.
struct View {
    uint32_t kind, slot, allocation, stride;
    uint64_t offset, length;
    uint32_t format = 0, element_bytes = 0, dimension = 0, width = 0, height = 0, layers = 0, counter = 0;
    uint32_t resource_kind() const { return kind & 3u; }
    bool raw() const { return (kind & 4u) != 0; }
    bool extended() const { return (kind & 8u) != 0; }
    bool continues() const { return (kind & 16u) != 0; }
    bool sampler() const { return resource_kind() == 3; }
    bool texture() const { return dimension != 0; }
    bool typed() const { return format != 0 && dimension == 0 && !sampler(); }
    uint32_t element_size() const { return raw() ? 4u : typed() ? element_bytes : stride; }
};
enum TextureDimension : uint32_t {
    Texture1D = 1, Texture1DArray, Texture2D, Texture2DArray, Texture3D, TextureCube, TextureCubeArray
};
struct Dispatch {
    std::vector<char> shader;
    uint32_t groups[3];
    std::vector<View> views;
};
struct Request {
    std::vector<std::vector<char>> allocations;
    std::vector<Dispatch> dispatches;
};

struct AdapterIdentity {
    std::wstring name;
    bool hardware;
    uint32_t vendor;
    uint64_t video_memory, dedicated_memory, shared_memory;
    bool matches(const AdapterIdentity& other) const {
        return name == other.name && hardware == other.hardware && vendor == other.vendor &&
            video_memory == other.video_memory && dedicated_memory == other.dedicated_memory &&
            shared_memory == other.shared_memory;
    }
};

uint64_t read_integer(std::istream& input, unsigned bytes) {
    uint64_t value = 0;
    for (unsigned index = 0; index < bytes; ++index) {
        int next = input.get();
        if (next == EOF) throw std::runtime_error("truncated request");
        value |= uint64_t(uint8_t(next)) << (index * 8);
    }
    return value;
}
uint32_t read_u32(std::istream& input) {
    return static_cast<uint32_t>(read_integer(input, 4));
}
[[maybe_unused]] void write_integer(std::ostream& output, uint64_t value, unsigned bytes) {
    for (unsigned index = 0; index < bytes; ++index)
        output.put(static_cast<char>((value >> (index * 8)) & 255));
}
std::vector<char> read_blob(std::istream& input, uint64_t size) {
    auto position = input.tellg();
    input.seekg(0, std::ios::end);
    auto remaining = input.tellg() - position;
    input.seekg(position);
    if (remaining < 0 || size > static_cast<uint64_t>(remaining) ||
        size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("invalid payload size");
    std::vector<char> result(static_cast<size_t>(size));
    if (size) input.read(result.data(), static_cast<std::streamsize>(size));
    if (!input) throw std::runtime_error("truncated payload");
    return result;
}
void check_buffer_view(const View& view, uint64_t size) {
    if (view.resource_kind() == 0) {
        if (view.raw() || view.extended() || view.continues() || view.offset % 256 ||
            view.length > 65536 || ((view.length + 255) & ~uint64_t(255)) > size - view.offset)
            throw std::runtime_error("invalid constant-buffer view alignment or extent");
    } else if (view.typed()) {
        auto bytes = view.element_bytes;
        if (view.raw() || view.stride || (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8 &&
            bytes != 12 && bytes != 16) || view.offset % bytes || view.length % bytes ||
            view.length / bytes > UINT32_MAX)
            throw std::runtime_error("invalid typed-buffer view");
    } else if (view.format || view.element_bytes) {
        throw std::runtime_error("invalid buffer view format");
    } else if (view.raw()) {
        if (view.stride || view.offset % 16 || view.length % 4 || view.length / 4 > UINT32_MAX)
            throw std::runtime_error("invalid raw-buffer view");
    } else if (!view.stride || view.stride > 2048 || view.offset % view.stride ||
               view.length % view.stride || view.length / view.stride > UINT32_MAX) {
        throw std::runtime_error("invalid structured-buffer view");
    }
}

void check_texture_view(const View& view) {
    auto dimension = view.dimension;
    auto bytes = view.element_bytes;
    bool cube = dimension == TextureCube || dimension == TextureCubeArray;
    if (dimension > TextureCubeArray || view.raw() || view.stride || view.counter || !view.format ||
        (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8 && bytes != 12 && bytes != 16) ||
        !view.width || view.width > 16384 || !view.height || view.height > 16384 ||
        !view.layers || view.layers > 2048 ||
        ((dimension == Texture1D || dimension == Texture1DArray) && view.height != 1) ||
        ((dimension == Texture1D || dimension == Texture2D) && view.layers != 1) ||
        (dimension == Texture3D && (view.width > 2048 || view.height > 2048)) ||
        (cube && (view.width != view.height || view.layers % 6)) ||
        (dimension == TextureCube && view.layers != 6) ||
        (view.resource_kind() == 2 && cube))
        throw std::runtime_error("invalid texture view");
    if (view.length != uint64_t(bytes) * view.width * view.height * view.layers)
        throw std::runtime_error("texture view length does not match its extent");
}

View read_view(std::istream& input, const Request& request, const std::vector<View>& previous_views) {
    View view;
    view.kind = read_u32(input);
    view.slot = read_u32(input);
    view.allocation = read_u32(input);
    view.stride = read_u32(input);
    view.offset = read_integer(input, 8);
    view.length = read_integer(input, 8);
    if (view.kind & ~31u) throw std::runtime_error("invalid buffer view");
    if (view.extended()) {
        view.format = read_u32(input);
        view.element_bytes = read_u32(input);
        view.dimension = read_u32(input);
        view.width = read_u32(input);
        view.height = read_u32(input);
        view.layers = read_u32(input);
        view.counter = read_u32(input);
        if (read_u32(input)) throw std::runtime_error("invalid view extension");
    }
    if (view.sampler()) {
        if (!view.extended() || view.raw() || view.continues() || view.allocation || view.stride ||
            view.offset || view.length || view.format > 1 || view.element_bytes || view.dimension ||
            view.width || view.height || view.layers || view.counter)
            throw std::runtime_error("invalid sampler");
    } else {
        if (view.allocation >= request.allocations.size() || !view.length)
            throw std::runtime_error("invalid buffer view");
        auto size = request.allocations[view.allocation].size();
        if (view.offset > size || view.length > size - view.offset)
            throw std::runtime_error("view exceeds allocation");
        if (view.texture()) check_texture_view(view);
        else check_buffer_view(view, size);
        if (view.counter && (view.resource_kind() != 2 || view.raw() || view.typed() || view.texture() ||
                             view.counter - 1 >= request.allocations.size() ||
                             view.counter - 1 == view.allocation ||
                             request.allocations[view.counter - 1].size() < 4))
            throw std::runtime_error("invalid UAV counter");
    }
    if (view.continues()) {
        if (previous_views.empty() || previous_views.back().resource_kind() != view.resource_kind() ||
            previous_views.back().slot + 1 != view.slot || view.resource_kind() == 0)
            throw std::runtime_error("a continued descriptor range needs the previous register");
    }
    for (const auto& previous : previous_views) {
        if (previous.resource_kind() == view.resource_kind() && previous.slot == view.slot)
            throw std::runtime_error("duplicate register");
        if (previous.sampler() || view.sampler()) continue;
        if ((previous.counter && (previous.counter - 1 == view.allocation || previous.counter == view.counter)) ||
            (view.counter && view.counter - 1 == previous.allocation))
            throw std::runtime_error("a UAV counter shares an allocation");
        if (previous.allocation != view.allocation) continue;
        if ((previous.resource_kind() == 2) != (view.resource_kind() == 2))
            throw std::runtime_error("simultaneous SRV/CBV and UAV access requires enhanced barriers");
        if (view.resource_kind() == 2 && previous.offset < view.offset + view.length &&
            view.offset < previous.offset + previous.length)
            throw std::runtime_error("overlapping writable views");
    }
    return view;
}

Request read_request(std::istream& input) {
    input.seekg(0, std::ios::end);
    auto request_size = input.tellg();
    if (request_size < 0 || request_size > 512 * 1024 * 1024)
        throw std::runtime_error("request exceeds execution limit");
    input.seekg(0);
    if (read_u32(input) != 0x31565844u)
        throw std::runtime_error("unsupported request version");
    Request request;
    uint32_t count = read_u32(input);
    if (count == 0 || count > 65536) throw std::runtime_error("invalid allocation count");
    for (uint32_t index = 0; index < count; ++index) {
        auto size = read_integer(input, 8);
        if (!size || size > 256 * 1024 * 1024)
            throw std::runtime_error("invalid allocation size");
        request.allocations.push_back(read_blob(input, size));
    }
    count = read_u32(input);
    if (!count || count > 65536) throw std::runtime_error("invalid dispatch count");
    for (uint32_t index = 0; index < count; ++index) {
        Dispatch dispatch;
        dispatch.shader = read_blob(input, read_integer(input, 8));
        if (dispatch.shader.size() < 4 || std::memcmp(dispatch.shader.data(), "DXBC", 4))
            throw std::runtime_error("shader is not a DXIL container");
        for (auto& dimension : dispatch.groups) {
            dimension = read_u32(input);
            if (!dimension || dimension > 65535) throw std::runtime_error("invalid dispatch size");
        }
        uint32_t view_count = read_u32(input);
        if (view_count > 64) throw std::runtime_error("descriptor table exceeds root signature budget");
        for (uint32_t view_index = 0; view_index < view_count; ++view_index)
            dispatch.views.push_back(read_view(input, request, dispatch.views));
        request.dispatches.push_back(std::move(dispatch));
    }
    if (input.peek() != EOF) throw std::runtime_error("trailing request data");
    return request;
}
} // namespace

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cstdlib>
#include <sstream>

#ifdef CROSSTL_D3D12_SDK_VERSION
extern "C" {
__declspec(dllexport) extern const UINT D3D12SDKVersion = CROSSTL_D3D12_SDK_VERSION;
__declspec(dllexport) const char* D3D12SDKPath = ".\\D3D12\\";
}
#endif

namespace {
using Microsoft::WRL::ComPtr;
// Older SDK headers name TOOL_DEBUG (0x1) DEBUG.
constexpr auto tool_debug_flag = static_cast<D3D12_PIPELINE_STATE_FLAGS>(0x1);
void check(HRESULT status, const char* operation) {
    if (FAILED(status)) {
        std::ostringstream message;
        message << operation << " failed: 0x" << std::hex << static_cast<uint32_t>(status);
        throw std::runtime_error(message.str());
    }
}
class DebugMessages {
    ComPtr<ID3D12InfoQueue> queue;
    std::filesystem::path path;
public:
    explicit DebugMessages(const std::filesystem::path& destination) : path(destination) {
        if (path.empty()) return;
        ComPtr<ID3D12Debug> debug;
        check(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)), "D3D12GetDebugInterface");
        debug->EnableDebugLayer();
    }
    void attach(ID3D12Device* device) {
        if (path.empty()) return;
        check(device->QueryInterface(IID_PPV_ARGS(&queue)), "QueryInterface ID3D12InfoQueue");
        check(queue->SetMessageCountLimit(64), "SetMessageCountLimit");
    }
    void save() const {
        if (!queue) return;
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open debug message output");
        auto stored = queue->GetNumStoredMessagesAllowedByRetrievalFilter();
        auto count = static_cast<uint32_t>(std::min<UINT64>(stored, 64));
        write_integer(output, 0x31475844u, 4);
        write_integer(output, stored, 8);
        write_integer(output, queue->GetNumMessagesDiscardedByMessageCountLimit(), 8);
        write_integer(output, count, 4);
        for (uint32_t index = 0; index < count; ++index) {
            SIZE_T size = 0;
            check(queue->GetMessage(index, nullptr, &size), "GetMessage size");
            if (size < sizeof(D3D12_MESSAGE) || size > 65536)
                throw std::runtime_error("invalid debug message size");
            std::vector<char> storage(size);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            check(queue->GetMessage(index, message, &size), "GetMessage");
            auto length = message->DescriptionByteLength;
            if (!length || length > storage.size() - sizeof(D3D12_MESSAGE))
                throw std::runtime_error("invalid debug description size");
            if (message->pDescription[length - 1] == '\0') --length;
            auto retained = std::min<SIZE_T>(length, 4096);
            write_integer(output, static_cast<uint32_t>(message->ID), 4);
            write_integer(output, static_cast<uint32_t>(message->Category), 4);
            write_integer(output, static_cast<uint32_t>(message->Severity), 4);
            write_integer(output, length > retained ? 1 : 0, 4);
            write_integer(output, retained, 4);
            output.write(message->pDescription, static_cast<std::streamsize>(retained));
        }
        output.close();
        if (!output) throw std::runtime_error("cannot write debug message output");
    }
    ~DebugMessages() {
        try { save(); }
        catch (const std::exception& error) { std::cerr << error.what() << '\n'; }
    }
};
void write_runtime_libraries(const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open runtime identity output");
    write_integer(output, 0x314c5844u, 4);
    write_integer(output, GetCurrentProcessId(), 4);
    write_integer(output, 3, 4);
    for (const auto* name : {L"d3d12.dll", L"D3D12Core.dll", L"d3d10warp.dll"}) {
        HMODULE module = GetModuleHandleW(name);
        std::vector<wchar_t> filename(32768);
        DWORD length = 0, error = 0;
        if (module) {
            length = GetModuleFileNameW(module, filename.data(),
                                        static_cast<DWORD>(filename.size()));
            if (!length || length >= filename.size()) {
                error = GetLastError();
                if (!error) error = ERROR_INSUFFICIENT_BUFFER;
                length = 0;
            }
        }
        write_integer(output, module ? 1 : 0, 4);
        write_integer(output, error, 4);
        write_integer(output, length, 4);
        for (DWORD index = 0; index < length; ++index)
            write_integer(output, static_cast<uint16_t>(filename[index]), 2);
    }
    output.close();
    if (!output) throw std::runtime_error("cannot write runtime identity output");
}
struct Device {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    HANDLE event = nullptr;
    uint64_t serial = 0;
    explicit Device(const AdapterIdentity& identity, bool tool_debug) {
        ComPtr<IDXGIFactory4> factory;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
        ComPtr<IDXGIAdapter1> selected;
        for (UINT index = 0;; ++index) {
            ComPtr<IDXGIAdapter1> candidate;
            HRESULT status = factory->EnumAdapters1(index, &candidate);
            if (status == DXGI_ERROR_NOT_FOUND) break;
            check(status, "EnumAdapters1");
            DXGI_ADAPTER_DESC1 description = {};
            check(candidate->GetDesc1(&description), "GetDesc1");
            if (!matches(identity, description)) continue;
            if (selected) {
                DXGI_ADAPTER_DESC1 previous = {};
                check(selected->GetDesc1(&previous), "GetDesc1 selected");
                if (previous.AdapterLuid.LowPart != description.AdapterLuid.LowPart ||
                    previous.AdapterLuid.HighPart != description.AdapterLuid.HighPart)
                    throw std::runtime_error("adapter identity is ambiguous");
            }
            selected = candidate;
        }
        if (!selected) {
            ComPtr<IDXGIAdapter1> warp;
            check(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)), "EnumWarpAdapter");
            DXGI_ADAPTER_DESC1 description = {};
            check(warp->GetDesc1(&description), "GetDesc1 WARP");
            if (!matches(identity, description))
                throw std::runtime_error("selected Direct3D adapter identity was not found");
            selected = warp;
        }
        if (tool_debug) {
            ComPtr<IDXGIAdapter1> warp;
            check(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)), "EnumWarpAdapter debug");
            DXGI_ADAPTER_DESC1 selected_desc = {}, warp_desc = {};
            check(selected->GetDesc1(&selected_desc), "GetDesc1 debug adapter");
            check(warp->GetDesc1(&warp_desc), "GetDesc1 debug WARP");
            if (selected_desc.AdapterLuid.LowPart != warp_desc.AdapterLuid.LowPart ||
                selected_desc.AdapterLuid.HighPart != warp_desc.AdapterLuid.HighPart)
                throw std::runtime_error("tool debugging requires the WARP adapter");
        }
        check(D3D12CreateDevice(selected.Get(), D3D_FEATURE_LEVEL_12_0,
                               IID_PPV_ARGS(&device)), "D3D12CreateDevice");
        D3D12_COMMAND_QUEUE_DESC description = {};
        description.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        check(device->CreateCommandQueue(&description, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
        check(device->CreateCommandAllocator(description.Type, IID_PPV_ARGS(&allocator)), "CreateCommandAllocator");
        check(device->CreateCommandList(0, description.Type, allocator.Get(), nullptr,
                                       IID_PPV_ARGS(&list)), "CreateCommandList");
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) throw std::runtime_error("CreateEvent failed");
    }
    static bool matches(const AdapterIdentity& identity, const DXGI_ADAPTER_DESC1& description) {
        return identity.matches({description.Description,
            (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0, description.VendorId,
            description.DedicatedVideoMemory, description.DedicatedSystemMemory,
            description.SharedSystemMemory});
    }
    ~Device() { if (event) CloseHandle(event); }
    void submit() {
        check(list->Close(), "Close command list");
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        check(queue->Signal(fence.Get(), ++serial), "Signal fence");
        if (fence->GetCompletedValue() < serial) {
            check(fence->SetEventOnCompletion(serial, event), "SetEventOnCompletion");
            if (WaitForSingleObject(event, 60000) != WAIT_OBJECT_0)
                throw std::runtime_error("GPU fence timeout");
        }
        check(device->GetDeviceRemovedReason(), "Device status");
        check(allocator->Reset(), "Reset allocator");
        check(list->Reset(allocator.Get(), nullptr), "Reset command list");
    }
    ComPtr<ID3D12Resource> buffer(uint64_t size, D3D12_HEAP_TYPE heap,
                                 D3D12_RESOURCE_STATES state) {
        D3D12_HEAP_PROPERTIES properties = {};
        properties.Type = heap;
        properties.CreationNodeMask = properties.VisibleNodeMask = 1;
        D3D12_RESOURCE_DESC description = {};
        description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        description.Width = size;
        description.Height = description.DepthOrArraySize = description.MipLevels = 1;
        description.SampleDesc.Count = 1;
        description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        description.Flags = heap == D3D12_HEAP_TYPE_DEFAULT
            ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        ComPtr<ID3D12Resource> resource;
        check(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &description,
                                             state, nullptr, IID_PPV_ARGS(&resource)), "CreateCommittedResource");
        return resource;
    }
    void transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                    D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER barrier = {};
        if (before == after) {
            if (after != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) return;
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.UAV.pResource = resource;
        } else {
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = resource;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = before;
            barrier.Transition.StateAfter = after;
        }
        list->ResourceBarrier(1, &barrier);
    }
};

// The root signature of one dispatch: a descriptor table per range of consecutive registers (one per view, or one
// per resource array), the views that take descriptors in heap order with each table's first descriptor, and a
// static sampler per sampler view.
struct Layout {
    std::vector<const View*> views;
    std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
    std::vector<D3D12_ROOT_PARAMETER> parameters;
    std::vector<UINT> first;
    std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;
};

D3D12_STATIC_SAMPLER_DESC static_sampler(const View& view) {
    // Normalized coordinates, point filtering, wrapping addresses and every mip level, the state the conformance
    // harness gives every sampler; a comparison sampler passes when the reference is at most the texel.
    D3D12_STATIC_SAMPLER_DESC sampler = {};
    bool comparison = view.format == 1;
    sampler.Filter = comparison ? D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT : D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.MaxAnisotropy = 1;
    sampler.ComparisonFunc = comparison ? D3D12_COMPARISON_FUNC_LESS_EQUAL : D3D12_COMPARISON_FUNC_NEVER;
    sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderRegister = view.slot;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    return sampler;
}

Layout describe(const Dispatch& dispatch) {
    Layout layout;
    layout.ranges.reserve(dispatch.views.size());
    for (const auto& view : dispatch.views) {
        if (view.sampler()) {
            layout.samplers.push_back(static_sampler(view));
            continue;
        }
        // read_view admits a continued range only right after the view of the previous register.
        if (view.continues()) {
            ++layout.ranges.back().NumDescriptors;
        } else {
            D3D12_DESCRIPTOR_RANGE range = {};
            range.RangeType = view.resource_kind() == 0 ? D3D12_DESCRIPTOR_RANGE_TYPE_CBV
                : view.resource_kind() == 1 ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV : D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            range.NumDescriptors = 1;
            range.BaseShaderRegister = view.slot;
            layout.ranges.push_back(range);
            layout.first.push_back(static_cast<UINT>(layout.views.size()));
        }
        layout.views.push_back(&view);
    }
    layout.parameters.resize(layout.ranges.size());
    for (size_t index = 0; index < layout.ranges.size(); ++index) {
        auto& parameter = layout.parameters[index];
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameter.DescriptorTable.NumDescriptorRanges = 1;
        parameter.DescriptorTable.pDescriptorRanges = &layout.ranges[index];
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    return layout;
}

// Create the texture of ``view``, fill it from its allocation and leave it in the state its view needs. The upload
// buffer goes to ``uploads``, which keeps it alive until the commands that read it have run.
ComPtr<ID3D12Resource> upload_texture(Device& runtime, const View& view, const std::vector<char>& data,
                                      std::vector<ComPtr<ID3D12Resource>>& uploads) {
    D3D12_RESOURCE_DESC description = {};
    auto dimension = view.dimension;
    description.Dimension = dimension <= Texture1DArray ? D3D12_RESOURCE_DIMENSION_TEXTURE1D
        : dimension == Texture3D ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    description.Width = view.width;
    description.Height = view.height;
    description.DepthOrArraySize = static_cast<UINT16>(view.layers);
    description.MipLevels = 1;
    description.Format = static_cast<DXGI_FORMAT>(view.format);
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    description.Flags = view.resource_kind() == 2 ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                                                  : D3D12_RESOURCE_FLAG_NONE;
    D3D12_HEAP_PROPERTIES properties = {};
    properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    properties.CreationNodeMask = properties.VisibleNodeMask = 1;
    ComPtr<ID3D12Resource> texture;
    check(runtime.device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &description,
                                                  D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                  IID_PPV_ARGS(&texture)), "CreateCommittedResource texture");
    UINT count = dimension == Texture3D ? 1 : view.layers;
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints(count);
    std::vector<UINT> rows(count);
    std::vector<UINT64> row_bytes(count);
    UINT64 total = 0;
    runtime.device->GetCopyableFootprints(&description, 0, count, 0, footprints.data(), rows.data(),
                                          row_bytes.data(), &total);
    uint64_t row = uint64_t(view.element_bytes) * view.width;
    if (row_bytes[0] != row || rows[0] != view.height)
        throw std::runtime_error("texture format does not match its texel size");
    auto upload = runtime.buffer(total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    char* mapped = nullptr;
    D3D12_RANGE no_reads = {0, 0};
    check(upload->Map(0, &no_reads, reinterpret_cast<void**>(&mapped)), "Map texture upload");
    const char* source = data.data() + view.offset;
    for (UINT index = 0; index < count; ++index) {
        const auto& footprint = footprints[index];
        for (UINT slice = 0; slice < footprint.Footprint.Depth; ++slice) {
            for (UINT line = 0; line < rows[index]; ++line) {
                std::memcpy(mapped + footprint.Offset +
                                (uint64_t(slice) * rows[index] + line) * footprint.Footprint.RowPitch,
                            source, static_cast<size_t>(row));
                source += row;
            }
        }
    }
    D3D12_RANGE written = {0, static_cast<SIZE_T>(total)};
    upload->Unmap(0, &written);
    for (UINT index = 0; index < count; ++index) {
        D3D12_TEXTURE_COPY_LOCATION target = {}, origin = {};
        target.pResource = texture.Get();
        target.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        target.SubresourceIndex = index;
        origin.pResource = upload.Get();
        origin.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        origin.PlacedFootprint = footprints[index];
        runtime.list->CopyTextureRegion(&target, 0, 0, 0, &origin, nullptr);
    }
    runtime.transition(texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                       view.resource_kind() == 2 ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                                 : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    uploads.push_back(std::move(upload));
    return texture;
}

void create_shader_resource_view(ID3D12Device* device, const View& view, ID3D12Resource* buffer,
                                 ID3D12Resource* texture, D3D12_CPU_DESCRIPTOR_HANDLE cpu) {
    D3D12_SHADER_RESOURCE_VIEW_DESC descriptor = {};
    descriptor.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (view.texture()) {
        descriptor.Format = static_cast<DXGI_FORMAT>(view.format);
        switch (view.dimension) {
        case Texture1D:
            descriptor.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            descriptor.Texture1D.MipLevels = 1;
            break;
        case Texture1DArray:
            descriptor.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
            descriptor.Texture1DArray.MipLevels = 1;
            descriptor.Texture1DArray.ArraySize = view.layers;
            break;
        case Texture2D:
            descriptor.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            descriptor.Texture2D.MipLevels = 1;
            break;
        case Texture2DArray:
            descriptor.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            descriptor.Texture2DArray.MipLevels = 1;
            descriptor.Texture2DArray.ArraySize = view.layers;
            break;
        case Texture3D:
            descriptor.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            descriptor.Texture3D.MipLevels = 1;
            break;
        case TextureCube:
            descriptor.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
            descriptor.TextureCube.MipLevels = 1;
            break;
        default:
            descriptor.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
            descriptor.TextureCubeArray.MipLevels = 1;
            descriptor.TextureCubeArray.NumCubes = view.layers / 6;
            break;
        }
        device->CreateShaderResourceView(texture, &descriptor, cpu);
        return;
    }
    descriptor.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    descriptor.Format = view.raw() ? DXGI_FORMAT_R32_TYPELESS : static_cast<DXGI_FORMAT>(view.format);
    descriptor.Buffer.Flags = view.raw() ? D3D12_BUFFER_SRV_FLAG_RAW : D3D12_BUFFER_SRV_FLAG_NONE;
    descriptor.Buffer.FirstElement = view.offset / view.element_size();
    descriptor.Buffer.NumElements = static_cast<UINT>(view.length / view.element_size());
    descriptor.Buffer.StructureByteStride = view.typed() ? 0 : view.stride;
    device->CreateShaderResourceView(buffer, &descriptor, cpu);
}

void create_unordered_access_view(ID3D12Device* device, const View& view, ID3D12Resource* buffer,
                                  ID3D12Resource* texture, ID3D12Resource* counter,
                                  D3D12_CPU_DESCRIPTOR_HANDLE cpu) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC descriptor = {};
    if (view.texture()) {
        // read_view admits no cube UAV.
        descriptor.Format = static_cast<DXGI_FORMAT>(view.format);
        switch (view.dimension) {
        case Texture1D:
            descriptor.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
            break;
        case Texture1DArray:
            descriptor.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            descriptor.Texture1DArray.ArraySize = view.layers;
            break;
        case Texture2D:
            descriptor.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            break;
        case Texture2DArray:
            descriptor.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            descriptor.Texture2DArray.ArraySize = view.layers;
            break;
        default:
            descriptor.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            descriptor.Texture3D.WSize = view.layers;
            break;
        }
        device->CreateUnorderedAccessView(texture, nullptr, &descriptor, cpu);
        return;
    }
    descriptor.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    descriptor.Format = view.raw() ? DXGI_FORMAT_R32_TYPELESS : static_cast<DXGI_FORMAT>(view.format);
    descriptor.Buffer.Flags = view.raw() ? D3D12_BUFFER_UAV_FLAG_RAW : D3D12_BUFFER_UAV_FLAG_NONE;
    descriptor.Buffer.FirstElement = view.offset / view.element_size();
    descriptor.Buffer.NumElements = static_cast<UINT>(view.length / view.element_size());
    descriptor.Buffer.StructureByteStride = view.typed() ? 0 : view.stride;
    device->CreateUnorderedAccessView(buffer, counter, &descriptor, cpu);
}

void create_view(ID3D12Device* device, const View& view, ID3D12Resource* buffer, ID3D12Resource* texture,
                 ID3D12Resource* counter, D3D12_CPU_DESCRIPTOR_HANDLE cpu) {
    if (view.resource_kind() == 0) {
        D3D12_CONSTANT_BUFFER_VIEW_DESC descriptor = {};
        descriptor.BufferLocation = buffer->GetGPUVirtualAddress() + view.offset;
        descriptor.SizeInBytes = static_cast<UINT>((view.length + 255) & ~uint64_t(255));
        device->CreateConstantBufferView(&descriptor, cpu);
    } else if (view.resource_kind() == 1) {
        create_shader_resource_view(device, view, buffer, texture, cpu);
    } else {
        create_unordered_access_view(device, view, buffer, texture, counter, cpu);
    }
}

void execute(const Request& request, std::ostream& output, const AdapterIdentity& identity,
             const std::filesystem::path& runtime_path, bool tool_debug,
             const std::filesystem::path& debug_path) {
    Progress progress;
    DebugMessages debug(debug_path);
    progress.enter("device-create");
    Device runtime(identity, tool_debug);
    debug.attach(runtime.device.Get());
    // Inspect this process after device creation, not the Python launcher.
    write_runtime_libraries(runtime_path);
    progress.enter("allocations");
    // Textures and their upload buffers stay alive until the dispatch that reads them completes.
    std::vector<ComPtr<ID3D12Resource>> buffers, uploads, textures;
    std::vector<D3D12_RESOURCE_STATES> states(request.allocations.size(), D3D12_RESOURCE_STATE_COPY_DEST);
    for (const auto& payload : request.allocations) {
        auto upload = runtime.buffer(payload.size(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto buffer = runtime.buffer(payload.size(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
        void* mapped = nullptr;
        D3D12_RANGE no_reads = {0, 0};
        check(upload->Map(0, &no_reads, &mapped), "Map upload");
        std::memcpy(mapped, payload.data(), payload.size());
        D3D12_RANGE written = {0, payload.size()};
        upload->Unmap(0, &written);
        runtime.list->CopyBufferRegion(buffer.Get(), 0, upload.Get(), 0, payload.size());
        uploads.push_back(std::move(upload));
        buffers.push_back(std::move(buffer));
    }
    progress.enter("upload-submit");
    runtime.submit();
    // Buffers decay to COMMON after an ExecuteCommandLists submission completes.
    std::fill(states.begin(), states.end(), D3D12_RESOURCE_STATE_COMMON);
    for (uint32_t dispatch_index = 0; dispatch_index < request.dispatches.size(); ++dispatch_index) {
        const auto& dispatch = request.dispatches[dispatch_index];
        progress.enter("root-signature", dispatch_index);
        Layout layout = describe(dispatch);
        std::vector<D3D12_RESOURCE_STATES> wanted(buffers.size(), D3D12_RESOURCE_STATE_COMMON);
        for (const auto* view : layout.views) {
            auto state = view->resource_kind() == 0 ? D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
                : view->resource_kind() == 1 ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            // A texture's allocation only fills the texture, which has a state of its own.
            if (!view->texture())
                wanted[view->allocation] = static_cast<D3D12_RESOURCE_STATES>(wanted[view->allocation] | state);
            if (view->counter)
                wanted[view->counter - 1] = static_cast<D3D12_RESOURCE_STATES>(
                    wanted[view->counter - 1] | D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        for (size_t index = 0; index < buffers.size(); ++index) {
            if (wanted[index] == D3D12_RESOURCE_STATE_COMMON) continue;
            runtime.transition(buffers[index].Get(), states[index], wanted[index]);
            states[index] = wanted[index];
        }
        D3D12_ROOT_SIGNATURE_DESC root_desc = {};
        root_desc.NumParameters = static_cast<UINT>(layout.parameters.size());
        root_desc.pParameters = layout.parameters.data();
        root_desc.NumStaticSamplers = static_cast<UINT>(layout.samplers.size());
        root_desc.pStaticSamplers = layout.samplers.data();
        ComPtr<ID3DBlob> root_blob, errors;
        check(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                         &root_blob, &errors), "SerializeRootSignature");
        ComPtr<ID3D12RootSignature> root;
        check(runtime.device->CreateRootSignature(0, root_blob->GetBufferPointer(),
                                                  root_blob->GetBufferSize(), IID_PPV_ARGS(&root)), "CreateRootSignature");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_desc = {};
        pipeline_desc.pRootSignature = root.Get();
        pipeline_desc.CS = {dispatch.shader.data(), dispatch.shader.size()};
        pipeline_desc.Flags = tool_debug ? tool_debug_flag
                                        : D3D12_PIPELINE_STATE_FLAG_NONE;
        ComPtr<ID3D12PipelineState> pipeline;
        progress.enter("pipeline-create", dispatch_index);
        check(runtime.device->CreateComputePipelineState(&pipeline_desc, IID_PPV_ARGS(&pipeline)), "CreateComputePipelineState");
        progress.enter("descriptors", dispatch_index);
        runtime.list->SetPipelineState(pipeline.Get());
        runtime.list->SetComputeRootSignature(root.Get());
        ComPtr<ID3D12DescriptorHeap> heap;
        if (!layout.views.empty()) {
            D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
            heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            heap_desc.NumDescriptors = static_cast<UINT>(layout.views.size());
            heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            check(runtime.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "CreateDescriptorHeap");
            auto stride = runtime.device->GetDescriptorHandleIncrementSize(heap_desc.Type);
            auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
            for (const auto* view : layout.views) {
                ID3D12Resource* texture = nullptr;
                if (view->texture()) {
                    textures.push_back(upload_texture(runtime, *view, request.allocations[view->allocation],
                                                      uploads));
                    texture = textures.back().Get();
                }
                create_view(runtime.device.Get(), *view, buffers[view->allocation].Get(), texture,
                            view->counter ? buffers[view->counter - 1].Get() : nullptr, cpu);
                cpu.ptr += stride;
            }
            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            runtime.list->SetDescriptorHeaps(1, heaps);
            auto gpu = heap->GetGPUDescriptorHandleForHeapStart();
            for (UINT index = 0; index < layout.parameters.size(); ++index) {
                auto table = gpu;
                table.ptr += uint64_t(stride) * layout.first[index];
                runtime.list->SetComputeRootDescriptorTable(index, table);
            }
        }
        progress.enter("dispatch-submit", dispatch_index);
        runtime.list->Dispatch(dispatch.groups[0], dispatch.groups[1], dispatch.groups[2]);
        runtime.submit();
        std::fill(states.begin(), states.end(), D3D12_RESOURCE_STATE_COMMON);
    }
    std::vector<ComPtr<ID3D12Resource>> readbacks;
    progress.enter("readback-submit");
    for (size_t index = 0; index < buffers.size(); ++index) {
        runtime.transition(buffers[index].Get(), states[index], D3D12_RESOURCE_STATE_COPY_SOURCE);
        auto readback = runtime.buffer(request.allocations[index].size(), D3D12_HEAP_TYPE_READBACK,
                                       D3D12_RESOURCE_STATE_COPY_DEST);
        runtime.list->CopyBufferRegion(readback.Get(), 0, buffers[index].Get(), 0,
                                       request.allocations[index].size());
        readbacks.push_back(std::move(readback));
    }
    runtime.submit();
    write_integer(output, 0x31525844u, 4);
    progress.enter("readback-map");
    write_integer(output, buffers.size(), 4);
    for (size_t index = 0; index < buffers.size(); ++index) {
        auto size = request.allocations[index].size();
        void* mapped = nullptr;
        D3D12_RANGE reads = {0, size};
        check(readbacks[index]->Map(0, &reads, &mapped), "Map readback");
        write_integer(output, buffers[index]->GetGPUVirtualAddress(), 8);
        write_integer(output, size, 8);
        output.write(static_cast<const char*>(mapped), static_cast<std::streamsize>(size));
        D3D12_RANGE no_writes = {0, 0};
        readbacks[index]->Unmap(0, &no_writes);
    }
    output.flush();
    if (!output) throw std::runtime_error("cannot write output");
    progress.enter("complete");
}

uint64_t adapter_number(const wchar_t* argument) {
    std::wstring text(argument);
    if (text.empty() || text.find_first_not_of(L"0123456789") != std::wstring::npos)
        throw std::runtime_error("invalid adapter identity number");
    return std::stoull(text);
}
} // namespace
#endif

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        if (argc != 3 && (argc < 9 || argc > 11))
            throw std::runtime_error("expected request, output and adapter identity");
        [[maybe_unused]] bool tool_debug = false, debug_layer = false;
        for (int index = 9; index < argc; ++index) {
            auto option = std::filesystem::path(argv[index]);
            if (option == "--warp-tool-debug" && !tool_debug) tool_debug = true;
            else if (option == "--debug-layer" && !debug_layer) debug_layer = true;
            else throw std::runtime_error("invalid pipeline debug option");
        }
        std::ifstream input(std::filesystem::path(argv[1]), std::ios::binary);
        if (!input) throw std::runtime_error("cannot open request");
        auto request = read_request(input);
        if (std::filesystem::path(argv[2]) == "--validate") {
            std::cout << request.allocations.size() << " allocations, "
                      << request.dispatches.size() << " dispatches\n";
            return 0;
        }
#ifdef _WIN32
        if (argc < 9) throw std::runtime_error("selected adapter identity is required");
        auto hardware = adapter_number(argv[4]);
        auto vendor = adapter_number(argv[5]);
        if (hardware > 1 || vendor > UINT32_MAX)
            throw std::runtime_error("invalid adapter identity flag or vendor");
        AdapterIdentity identity = {argv[3], hardware != 0, static_cast<uint32_t>(vendor),
            adapter_number(argv[6]), adapter_number(argv[7]), adapter_number(argv[8])};
        std::ofstream output(std::filesystem::path(argv[2]), std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open output");
        auto runtime_path = std::filesystem::path(argv[2]);
        runtime_path += ".runtime";
        std::filesystem::path debug_path;
        if (debug_layer) {
            debug_path = std::filesystem::path(argv[2]);
            debug_path += ".debug";
        }
        execute(request, output, identity, runtime_path, tool_debug, debug_path);
        if (!output) throw std::runtime_error("cannot write output");
        if (tool_debug) std::cout << "{\"pipelineStateFlags\":1}\n";
#else
        throw std::runtime_error("Direct3D 12 execution requires Windows");
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
