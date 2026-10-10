// Bounded Direct3D execution over shared allocations and explicit buffer views.
#include <algorithm>
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
struct View {
    uint32_t kind, slot, allocation, stride;
    uint64_t offset, length;
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
        for (uint32_t view_index = 0; view_index < view_count; ++view_index) {
            View view;
            view.kind = read_u32(input);
            view.slot = read_u32(input);
            view.allocation = read_u32(input);
            view.stride = read_u32(input);
            view.offset = read_integer(input, 8);
            view.length = read_integer(input, 8);
            if (view.kind > 2 || view.allocation >= request.allocations.size() || !view.length)
                throw std::runtime_error("invalid buffer view");
            auto size = request.allocations[view.allocation].size();
            if (view.offset > size || view.length > size - view.offset)
                throw std::runtime_error("view exceeds allocation");
            if (view.kind == 0) {
                if (view.offset % 256 || view.length > 65536 ||
                    ((view.length + 255) & ~uint64_t(255)) > size - view.offset)
                    throw std::runtime_error("invalid constant-buffer view alignment or extent");
            } else if (!view.stride || view.stride > 2048 || view.offset % view.stride ||
                       view.length % view.stride || view.length / view.stride > UINT32_MAX) {
                throw std::runtime_error("invalid structured-buffer view");
            }
            for (const auto& previous : dispatch.views) {
                if (previous.kind == view.kind && previous.slot == view.slot)
                    throw std::runtime_error("duplicate register");
                if (previous.allocation != view.allocation) continue;
                if ((previous.kind == 2) != (view.kind == 2))
                    throw std::runtime_error("simultaneous SRV/CBV and UAV access requires enhanced barriers");
                if (view.kind == 2 && previous.offset < view.offset + view.length &&
                    view.offset < previous.offset + previous.length)
                    throw std::runtime_error("overlapping writable views");
            }
            dispatch.views.push_back(view);
        }
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
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cstdlib>
#include <sstream>

namespace {
using Microsoft::WRL::ComPtr;
void check(HRESULT status, const char* operation) {
    if (FAILED(status)) {
        std::ostringstream message;
        message << operation << " failed: 0x" << std::hex << static_cast<uint32_t>(status);
        throw std::runtime_error(message.str());
    }
}
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
    explicit Device(const AdapterIdentity& identity) {
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

void execute(const Request& request, std::ostream& output, const AdapterIdentity& identity,
             const std::filesystem::path& runtime_path) {
    Device runtime(identity);
    // Inspect this process after device creation, not the Python launcher.
    write_runtime_libraries(runtime_path);
    std::vector<ComPtr<ID3D12Resource>> buffers, uploads;
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
    runtime.submit();
    // Buffers decay to COMMON after an ExecuteCommandLists submission completes.
    std::fill(states.begin(), states.end(), D3D12_RESOURCE_STATE_COMMON);
    for (const auto& dispatch : request.dispatches) {
        std::vector<D3D12_DESCRIPTOR_RANGE> ranges(dispatch.views.size());
        std::vector<D3D12_ROOT_PARAMETER> parameters(dispatch.views.size());
        std::vector<D3D12_RESOURCE_STATES> wanted(buffers.size(), D3D12_RESOURCE_STATE_COMMON);
        for (size_t index = 0; index < dispatch.views.size(); ++index) {
            const auto& view = dispatch.views[index];
            ranges[index].RangeType = view.kind == 0 ? D3D12_DESCRIPTOR_RANGE_TYPE_CBV
                : view.kind == 1 ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV : D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            ranges[index].NumDescriptors = 1;
            ranges[index].BaseShaderRegister = view.slot;
            parameters[index].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[index].DescriptorTable.NumDescriptorRanges = 1;
            parameters[index].DescriptorTable.pDescriptorRanges = &ranges[index];
            parameters[index].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            auto state = view.kind == 0 ? D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
                : view.kind == 1 ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            wanted[view.allocation] = static_cast<D3D12_RESOURCE_STATES>(wanted[view.allocation] | state);
        }
        for (size_t index = 0; index < buffers.size(); ++index) {
            if (wanted[index] == D3D12_RESOURCE_STATE_COMMON) continue;
            runtime.transition(buffers[index].Get(), states[index], wanted[index]);
            states[index] = wanted[index];
        }
        D3D12_ROOT_SIGNATURE_DESC root_desc = {};
        root_desc.NumParameters = static_cast<UINT>(parameters.size());
        root_desc.pParameters = parameters.data();
        ComPtr<ID3DBlob> root_blob, errors;
        check(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                         &root_blob, &errors), "SerializeRootSignature");
        ComPtr<ID3D12RootSignature> root;
        check(runtime.device->CreateRootSignature(0, root_blob->GetBufferPointer(),
                                                  root_blob->GetBufferSize(), IID_PPV_ARGS(&root)), "CreateRootSignature");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_desc = {};
        pipeline_desc.pRootSignature = root.Get();
        pipeline_desc.CS = {dispatch.shader.data(), dispatch.shader.size()};
        ComPtr<ID3D12PipelineState> pipeline;
        check(runtime.device->CreateComputePipelineState(&pipeline_desc, IID_PPV_ARGS(&pipeline)), "CreateComputePipelineState");
        runtime.list->SetPipelineState(pipeline.Get());
        runtime.list->SetComputeRootSignature(root.Get());
        ComPtr<ID3D12DescriptorHeap> heap;
        if (!dispatch.views.empty()) {
            D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
            heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            heap_desc.NumDescriptors = static_cast<UINT>(dispatch.views.size());
            heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            check(runtime.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "CreateDescriptorHeap");
            auto stride = runtime.device->GetDescriptorHandleIncrementSize(heap_desc.Type);
            auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
            for (const auto& view : dispatch.views) {
                auto buffer = buffers[view.allocation].Get();
                if (view.kind == 0) {
                    D3D12_CONSTANT_BUFFER_VIEW_DESC descriptor = {};
                    descriptor.BufferLocation = buffer->GetGPUVirtualAddress() + view.offset;
                    descriptor.SizeInBytes = static_cast<UINT>((view.length + 255) & ~uint64_t(255));
                    runtime.device->CreateConstantBufferView(&descriptor, cpu);
                } else if (view.kind == 1) {
                    D3D12_SHADER_RESOURCE_VIEW_DESC descriptor = {};
                    descriptor.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                    descriptor.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    descriptor.Buffer.FirstElement = view.offset / view.stride;
                    descriptor.Buffer.NumElements = static_cast<UINT>(view.length / view.stride);
                    descriptor.Buffer.StructureByteStride = view.stride;
                    runtime.device->CreateShaderResourceView(buffer, &descriptor, cpu);
                } else {
                    D3D12_UNORDERED_ACCESS_VIEW_DESC descriptor = {};
                    descriptor.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                    descriptor.Buffer.FirstElement = view.offset / view.stride;
                    descriptor.Buffer.NumElements = static_cast<UINT>(view.length / view.stride);
                    descriptor.Buffer.StructureByteStride = view.stride;
                    runtime.device->CreateUnorderedAccessView(buffer, nullptr, &descriptor, cpu);
                }
                cpu.ptr += stride;
            }
            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            runtime.list->SetDescriptorHeaps(1, heaps);
            auto gpu = heap->GetGPUDescriptorHandleForHeapStart();
            for (UINT index = 0; index < dispatch.views.size(); ++index) {
                runtime.list->SetComputeRootDescriptorTable(index, gpu);
                gpu.ptr += stride;
            }
        }
        runtime.list->Dispatch(dispatch.groups[0], dispatch.groups[1], dispatch.groups[2]);
        runtime.submit();
        std::fill(states.begin(), states.end(), D3D12_RESOURCE_STATE_COMMON);
    }
    std::vector<ComPtr<ID3D12Resource>> readbacks;
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
        if (argc != 3 && argc != 9) throw std::runtime_error("expected request, output and adapter identity");
        std::ifstream input(std::filesystem::path(argv[1]), std::ios::binary);
        if (!input) throw std::runtime_error("cannot open request");
        auto request = read_request(input);
        if (std::filesystem::path(argv[2]) == "--validate") {
            std::cout << request.allocations.size() << " allocations, "
                      << request.dispatches.size() << " dispatches\n";
            return 0;
        }
#ifdef _WIN32
        if (argc != 9) throw std::runtime_error("selected adapter identity is required");
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
        execute(request, output, identity, runtime_path);
        if (!output) throw std::runtime_error("cannot write output");
#else
        throw std::runtime_error("Direct3D 12 execution requires Windows");
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
