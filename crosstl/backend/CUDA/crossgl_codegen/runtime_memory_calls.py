"""CUDA host memory, array, texture and graphics interoperability calls in the CUDA to CrossGL converter, described as comments."""


class CudaImportRuntimeMemoryCallMixin:
    """Runtime memory call members of ``CudaToCrossGLConverter``."""

    def describe_cuda_memory_allocation_call(self, node, name, args):
        """Comments for a memory allocation, free, copy, set, pointer query or memory pool call; None for any other name."""
        if name in {"cudaMalloc", "cudaMallocManaged", "cudaMallocHost"}:
            if len(node.args) >= 2:
                target = self.format_runtime_pointer_target(node.args[0])
                size = self.visit(node.args[1])
                return [f"// CUDA memory allocate: {target}, bytes: {size}"]
        elif name in {"cudaFree", "cudaFreeHost"}:
            if args:
                return [f"// CUDA memory free: {args[0]}"]
        elif name in {"cudaMemcpy", "cudaMemcpyAsync"}:
            if len(args) >= 4:
                comment = (
                    f"// CUDA memory copy: {args[1]} -> {args[0]}, "
                    f"bytes: {args[2]}, kind: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", stream: {args[4]}"
                return [comment]
        elif name in {"cudaMemset", "cudaMemsetAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// CUDA memory set: {args[0]}, value: {args[1]}, "
                    f"bytes: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name == "cuMemAlloc":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver memory allocate: output: {output}, "
                    f"bytes: {args[1]}"
                ]
        elif name == "cuMemAllocManaged":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver memory allocate managed: "
                    f"output: {output}, bytes: {args[1]}, flags: {args[2]}"
                ]
        elif name in {"cuMemAllocHost", "cuMemHostAlloc"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                comment = (
                    f"// CUDA driver host memory allocate: output: {output}, "
                    f"bytes: {args[1]}"
                )
                if name == "cuMemHostAlloc" and len(args) >= 3:
                    comment += f", flags: {args[2]}"
                return [comment]
        elif name == "cuMemFree":
            if args:
                return [f"// CUDA driver memory free: {args[0]}"]
        elif name == "cuMemFreeHost":
            if args:
                return [f"// CUDA driver host memory free: {args[0]}"]
        elif name == "cuPointerGetAttribute":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver pointer get attribute: "
                    f"output: {output}, attribute: {args[1]}, pointer: {args[2]}"
                ]
        elif name == "cuPointerGetAttributes":
            if len(args) >= 4:
                return [
                    "// CUDA driver pointer get attributes: "
                    f"count: {args[0]}, attributes: {args[1]}, "
                    f"data: {args[2]}, pointer: {args[3]}"
                ]
        elif name == "cuMemGetAddressRange":
            if len(node.args) >= 3:
                base_output = self.format_runtime_pointer_target(node.args[0])
                size_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver memory get address range: "
                    f"base output: {base_output}, size output: {size_output}, "
                    f"pointer: {args[2]}"
                ]
        elif name in {"cuMemGetInfo", "cuMemGetInfo_v2"}:
            if len(node.args) >= 2:
                free_output = self.format_runtime_pointer_target(node.args[0])
                total_output = self.format_runtime_pointer_target(node.args[1])
                suffix = " v2" if name == "cuMemGetInfo_v2" else ""
                return [
                    f"// CUDA driver memory get info{suffix}: "
                    f"free output: {free_output}, total output: {total_output}"
                ]
        elif name == "cuMemAdvise":
            if len(args) >= 4:
                return [
                    "// CUDA driver memory advise: "
                    f"pointer: {args[0]}, bytes: {args[1]}, "
                    f"advice: {args[2]}, device: {args[3]}"
                ]
        elif name == "cuMemAllocAsync":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver memory allocate async: "
                    f"output: {output}, bytes: {args[1]}, stream: {args[2]}"
                ]
        elif name == "cuMemFreeAsync":
            if len(args) >= 2:
                return [
                    "// CUDA driver memory free async: "
                    f"pointer: {args[0]}, stream: {args[1]}"
                ]
        elif name == "cuMemPoolCreate":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver memory pool create: "
                    f"output: {output}, props: {args[1]}"
                ]
        elif name == "cuMemPoolDestroy":
            if args:
                return [f"// CUDA driver memory pool destroy: pool: {args[0]}"]
        elif name == "cuMemPoolTrimTo":
            if len(args) >= 2:
                return [
                    f"// CUDA driver memory pool trim: "
                    f"pool: {args[0]}, bytes: {args[1]}"
                ]
        elif name in {"cuDeviceGetDefaultMemPool", "cuDeviceGetMemPool"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                pool_kind = (
                    "default memory pool"
                    if name == "cuDeviceGetDefaultMemPool"
                    else "memory pool"
                )
                return [
                    f"// CUDA driver device get {pool_kind}: "
                    f"output: {output}, device: {args[1]}"
                ]
        elif name == "cuDeviceSetMemPool":
            if len(args) >= 2:
                return [
                    f"// CUDA driver device set memory pool: "
                    f"device: {args[0]}, pool: {args[1]}"
                ]
        elif name in {"cuMemPoolSetAttribute", "cuMemPoolGetAttribute"}:
            if len(node.args) >= 3:
                value = self.format_runtime_pointer_target(node.args[2])
                action = "set" if name == "cuMemPoolSetAttribute" else "get"
                value_label = "value" if action == "set" else "output"
                return [
                    f"// CUDA driver memory pool {action} attribute: "
                    f"pool: {args[0]}, attribute: {args[1]}, "
                    f"{value_label}: {value}"
                ]
        elif name == "cuMemPoolSetAccess":
            if len(args) >= 3:
                return [
                    f"// CUDA driver memory pool set access: "
                    f"pool: {args[0]}, descriptors: {args[1]}, count: {args[2]}"
                ]
        elif name == "cuMemPoolGetAccess":
            if len(node.args) >= 3:
                flags_output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver memory pool get access: "
                    f"flags output: {flags_output}, pool: {args[1]}, "
                    f"location: {args[2]}"
                ]
        elif name == "cuMemPoolExportToShareableHandle":
            if len(node.args) >= 4:
                handle_output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver memory pool export shareable handle: "
                    f"handle output: {handle_output}, pool: {args[1]}, "
                    f"handle type: {args[2]}, flags: {args[3]}"
                ]
        return None

    def describe_cuda_memory_sharing_call(self, node, name, args):
        """Comments for a memory pool sharing, virtual address, physical allocation, shareable handle or driver external memory and semaphore call; None for any other name."""
        if name == "cuMemPoolImportFromShareableHandle":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver memory pool import shareable handle: "
                    f"output: {output}, handle: {args[1]}, "
                    f"handle type: {args[2]}, flags: {args[3]}"
                ]
        elif name == "cuMemPoolExportPointer":
            if len(node.args) >= 2:
                share_data_output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver memory pool export pointer: "
                    f"share data output: {share_data_output}, pointer: {args[1]}"
                ]
        elif name == "cuMemPoolImportPointer":
            if len(node.args) >= 3:
                pointer_output = self.format_runtime_pointer_target(node.args[0])
                share_data = self.format_runtime_pointer_target(node.args[2])
                return [
                    f"// CUDA driver memory pool import pointer: "
                    f"pointer output: {pointer_output}, pool: {args[1]}, "
                    f"share data: {share_data}"
                ]
        elif name == "cuMemAddressReserve":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver virtual memory reserve: "
                    f"output: {output}, bytes: {args[1]}, "
                    f"alignment: {args[2]}, address: {args[3]}, "
                    f"flags: {args[4]}"
                ]
        elif name == "cuMemAddressFree":
            if len(args) >= 2:
                return [
                    f"// CUDA driver virtual memory free address: "
                    f"address: {args[0]}, bytes: {args[1]}"
                ]
        elif name == "cuMemCreate":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver virtual memory create allocation: "
                    f"output: {output}, bytes: {args[1]}, "
                    f"props: {args[2]}, flags: {args[3]}"
                ]
        elif name == "cuMemRelease":
            if args:
                return [
                    f"// CUDA driver virtual memory release allocation: "
                    f"allocation: {args[0]}"
                ]
        elif name == "cuMemMap":
            if len(args) >= 5:
                return [
                    f"// CUDA driver virtual memory map: "
                    f"address: {args[0]}, bytes: {args[1]}, "
                    f"offset: {args[2]}, allocation: {args[3]}, "
                    f"flags: {args[4]}"
                ]
        elif name == "cuMemUnmap":
            if len(args) >= 2:
                return [
                    f"// CUDA driver virtual memory unmap: "
                    f"address: {args[0]}, bytes: {args[1]}"
                ]
        elif name == "cuMemSetAccess":
            if len(args) >= 4:
                return [
                    f"// CUDA driver virtual memory set access: "
                    f"address: {args[0]}, bytes: {args[1]}, "
                    f"descriptors: {args[2]}, count: {args[3]}"
                ]
        elif name == "cuMemGetAccess":
            if len(node.args) >= 3:
                flags_output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver virtual memory get access: "
                    f"flags output: {flags_output}, location: {args[1]}, "
                    f"address: {args[2]}"
                ]
        elif name == "cuMemRetainAllocationHandle":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver virtual memory retain allocation handle: "
                    f"output: {output}, address: {args[1]}"
                ]
        elif name == "cuMemExportToShareableHandle":
            if len(node.args) >= 4:
                handle_output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver virtual memory export shareable handle: "
                    f"handle output: {handle_output}, allocation: {args[1]}, "
                    f"handle type: {args[2]}, flags: {args[3]}"
                ]
        elif name == "cuMemImportFromShareableHandle":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver virtual memory import shareable handle: "
                    f"output: {output}, handle: {args[1]}, "
                    f"handle type: {args[2]}"
                ]
        elif name == "cuImportExternalMemory":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver external memory import: "
                    f"output: {output}, handle: {args[1]}"
                ]
        elif name == "cuExternalMemoryGetMappedBuffer":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver external memory mapped buffer: "
                    f"{args[1]}, desc: {args[2]}, output: {output}"
                ]
        elif name == "cuExternalMemoryGetMappedMipmappedArray":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver external memory mapped mipmapped array: "
                    f"{args[1]}, desc: {args[2]}, output: {output}"
                ]
        elif name == "cuDestroyExternalMemory":
            if args:
                return [f"// CUDA driver external memory destroy: {args[0]}"]
        elif name == "cuImportExternalSemaphore":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver external semaphore import: "
                    f"output: {output}, handle: {args[1]}"
                ]
        elif name in {
            "cuSignalExternalSemaphoresAsync",
            "cuWaitExternalSemaphoresAsync",
        }:
            if len(args) >= 3:
                operation = (
                    "signal" if name == "cuSignalExternalSemaphoresAsync" else "wait"
                )
                comment = (
                    f"// CUDA driver external semaphore {operation}: "
                    f"semaphores: {args[0]}, params: {args[1]}, count: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name == "cuDestroyExternalSemaphore":
            if args:
                return [f"// CUDA driver external semaphore destroy: {args[0]}"]
        return None

    def describe_cuda_array_call(self, node, name, args):
        """Comments for an array, mipmapped array, texture object or surface object call; None for any other name."""
        if name in {"cuArrayCreate", "cuArrayCreate_v2"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver array create: output: {output}, "
                    f"desc: {args[1]}"
                ]
        elif name in {"cuArray3DCreate", "cuArray3DCreate_v2"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver array 3D create: output: {output}, "
                    f"desc: {args[1]}"
                ]
        elif name == "cuArrayDestroy":
            if args:
                return [f"// CUDA driver array destroy: array: {args[0]}"]
        elif name == "cuMipmappedArrayCreate":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver mipmapped array create: output: {output}, "
                    f"desc: {args[1]}, levels: {args[2]}"
                ]
        elif name == "cuMipmappedArrayGetLevel":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver mipmapped array get level: "
                    f"output: {output}, mipmapped array: {args[1]}, "
                    f"level: {args[2]}"
                ]
        elif name == "cuMipmappedArrayDestroy":
            if args:
                return [
                    f"// CUDA driver mipmapped array destroy: mipmapped array: {args[0]}"
                ]
        elif name in {"cuArrayGetDescriptor", "cuArrayGetDescriptor_v2"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver array get descriptor: "
                    f"output: {output}, array: {args[1]}"
                ]
        elif name in {"cuArray3DGetDescriptor", "cuArray3DGetDescriptor_v2"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver array 3D get descriptor: "
                    f"output: {output}, array: {args[1]}"
                ]
        elif name == "cuArrayGetMemoryRequirements":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver array get memory requirements: "
                    f"output: {output}, array: {args[1]}, device: {args[2]}"
                ]
        elif name == "cuMipmappedArrayGetMemoryRequirements":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver mipmapped array get memory requirements: "
                    f"output: {output}, mipmapped array: {args[1]}, "
                    f"device: {args[2]}"
                ]
        elif name == "cuArrayGetPlane":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver array get plane: output: {output}, "
                    f"array: {args[1]}, plane: {args[2]}"
                ]
        elif name == "cuArrayGetSparseProperties":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver array get sparse properties: "
                    f"output: {output}, array: {args[1]}"
                ]
        elif name == "cuMipmappedArrayGetSparseProperties":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver mipmapped array get sparse properties: "
                    f"output: {output}, mipmapped array: {args[1]}"
                ]
        elif name == "cuTexObjectCreate":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture object create: "
                    f"output: {output}, resource desc: {args[1]}, "
                    f"texture desc: {args[2]}, resource view desc: {args[3]}"
                ]
        elif name == "cuTexObjectDestroy":
            if args:
                return [
                    f"// CUDA driver texture object destroy: texture object: {args[0]}"
                ]
        elif name == "cuTexObjectGetResourceDesc":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture object get resource desc: "
                    f"output: {output}, texture object: {args[1]}"
                ]
        elif name == "cuTexObjectGetTextureDesc":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture object get texture desc: "
                    f"output: {output}, texture object: {args[1]}"
                ]
        elif name == "cuTexObjectGetResourceViewDesc":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture object get resource view desc: "
                    f"output: {output}, texture object: {args[1]}"
                ]
        elif name == "cuSurfObjectCreate":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver surface object create: "
                    f"output: {output}, resource desc: {args[1]}"
                ]
        elif name == "cuSurfObjectDestroy":
            if args:
                return [
                    f"// CUDA driver surface object destroy: surface object: {args[0]}"
                ]
        elif name == "cuSurfObjectGetResourceDesc":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver surface object get resource desc: "
                    f"output: {output}, surface object: {args[1]}"
                ]
        return None

    def describe_cuda_texture_reference_call(self, node, name, args):
        """Comments for a texture reference or surface reference call; None for any other name."""
        if name == "cuTexRefSetArray":
            if len(args) >= 3:
                return [
                    "// CUDA driver texture reference set array: "
                    f"texture ref: {args[0]}, array: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cuTexRefSetMipmappedArray":
            if len(args) >= 3:
                return [
                    "// CUDA driver texture reference set mipmapped array: "
                    f"texture ref: {args[0]}, mipmapped array: {args[1]}, "
                    f"flags: {args[2]}"
                ]
        elif name in {"cuTexRefSetAddress", "cuTexRefSetAddress_v2"}:
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference set address: "
                    f"byte offset output: {output}, texture ref: {args[1]}, "
                    f"pointer: {args[2]}, bytes: {args[3]}"
                ]
        elif name in {"cuTexRefSetAddress2D", "cuTexRefSetAddress2D_v2"}:
            if len(args) >= 4:
                return [
                    "// CUDA driver texture reference set address 2D: "
                    f"texture ref: {args[0]}, desc: {args[1]}, "
                    f"pointer: {args[2]}, pitch: {args[3]}"
                ]
        elif name == "cuTexRefSetFormat":
            if len(args) >= 3:
                return [
                    "// CUDA driver texture reference set format: "
                    f"texture ref: {args[0]}, format: {args[1]}, "
                    f"components: {args[2]}"
                ]
        elif name == "cuTexRefSetAddressMode":
            if len(args) >= 3:
                return [
                    "// CUDA driver texture reference set address mode: "
                    f"texture ref: {args[0]}, dimension: {args[1]}, "
                    f"mode: {args[2]}"
                ]
        elif name == "cuTexRefSetFilterMode":
            if len(args) >= 2:
                return [
                    "// CUDA driver texture reference set filter mode: "
                    f"texture ref: {args[0]}, mode: {args[1]}"
                ]
        elif name == "cuTexRefSetMipmapFilterMode":
            if len(args) >= 2:
                return [
                    "// CUDA driver texture reference set mipmap filter mode: "
                    f"texture ref: {args[0]}, mode: {args[1]}"
                ]
        elif name == "cuTexRefSetMipmapLevelBias":
            if len(args) >= 2:
                return [
                    "// CUDA driver texture reference set mipmap level bias: "
                    f"texture ref: {args[0]}, bias: {args[1]}"
                ]
        elif name == "cuTexRefSetMipmapLevelClamp":
            if len(args) >= 3:
                return [
                    "// CUDA driver texture reference set mipmap level clamp: "
                    f"texture ref: {args[0]}, min level: {args[1]}, "
                    f"max level: {args[2]}"
                ]
        elif name == "cuTexRefSetMaxAnisotropy":
            if len(args) >= 2:
                return [
                    "// CUDA driver texture reference set max anisotropy: "
                    f"texture ref: {args[0]}, anisotropy: {args[1]}"
                ]
        elif name == "cuTexRefSetBorderColor":
            if len(args) >= 2:
                return [
                    "// CUDA driver texture reference set border color: "
                    f"texture ref: {args[0]}, color: {args[1]}"
                ]
        elif name == "cuTexRefSetFlags":
            if len(args) >= 2:
                return [
                    "// CUDA driver texture reference set flags: "
                    f"texture ref: {args[0]}, flags: {args[1]}"
                ]
        elif name in {"cuTexRefGetAddress", "cuTexRefGetAddress_v2"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get address: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetArray":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get array: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetMipmappedArray":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get mipmapped array: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetAddressMode":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get address mode: "
                    f"output: {output}, texture ref: {args[1]}, "
                    f"dimension: {args[2]}"
                ]
        elif name == "cuTexRefGetFilterMode":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get filter mode: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetFormat":
            if len(node.args) >= 3:
                format_output = self.format_runtime_pointer_target(node.args[0])
                channel_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver texture reference get format: "
                    f"format output: {format_output}, "
                    f"channel output: {channel_output}, texture ref: {args[2]}"
                ]
        elif name == "cuTexRefGetMipmapFilterMode":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get mipmap filter mode: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetMipmapLevelBias":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get mipmap level bias: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetMipmapLevelClamp":
            if len(node.args) >= 3:
                min_output = self.format_runtime_pointer_target(node.args[0])
                max_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver texture reference get mipmap level clamp: "
                    f"min output: {min_output}, max output: {max_output}, "
                    f"texture ref: {args[2]}"
                ]
        elif name == "cuTexRefGetMaxAnisotropy":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get max anisotropy: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetBorderColor":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get border color: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetFlags":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get flags: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuSurfRefSetArray":
            if len(args) >= 3:
                return [
                    "// CUDA driver surface reference set array: "
                    f"surface ref: {args[0]}, array: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cuSurfRefGetArray":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver surface reference get array: "
                    f"output: {output}, surface ref: {args[1]}"
                ]
        return None

    def describe_cuda_graphics_interop_call(self, node, name, args):
        """Comments for an OpenGL or Direct3D interoperability or graphics resource call; None for any other name."""
        if name == "cuGLInit":
            return ["// CUDA driver OpenGL initialize"]
        elif name in {"cuGLCtxCreate", "cuGLCtxCreate_v2"}:
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver OpenGL context create: "
                    f"output: {output}, flags: {args[1]}, device: {args[2]}"
                ]
        elif name == "cuGLGetDevices":
            if len(node.args) >= 4:
                count_output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver OpenGL get devices: "
                    f"count output: {count_output}, devices: {args[1]}, "
                    f"max devices: {args[2]}, device list: {args[3]}"
                ]
        elif name == "cuGLRegisterBufferObject":
            if args:
                return [
                    "// CUDA driver OpenGL register buffer object: "
                    f"buffer object: {args[0]}"
                ]
        elif name == "cuGLSetBufferObjectMapFlags":
            if len(args) >= 2:
                return [
                    "// CUDA driver OpenGL set buffer object map flags: "
                    f"buffer object: {args[0]}, flags: {args[1]}"
                ]
        elif name in {"cuGLMapBufferObject", "cuGLMapBufferObject_v2"}:
            if len(node.args) >= 3:
                pointer_output = self.format_runtime_pointer_target(node.args[0])
                size_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver OpenGL map buffer object: "
                    f"pointer output: {pointer_output}, "
                    f"size output: {size_output}, buffer object: {args[2]}"
                ]
        elif name in {
            "cuGLMapBufferObjectAsync",
            "cuGLMapBufferObjectAsync_v2",
        }:
            if len(node.args) >= 4:
                pointer_output = self.format_runtime_pointer_target(node.args[0])
                size_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver OpenGL map buffer object async: "
                    f"pointer output: {pointer_output}, "
                    f"size output: {size_output}, buffer object: {args[2]}, "
                    f"stream: {args[3]}"
                ]
        elif name == "cuGLUnmapBufferObject":
            if args:
                return [
                    "// CUDA driver OpenGL unmap buffer object: "
                    f"buffer object: {args[0]}"
                ]
        elif name == "cuGLUnmapBufferObjectAsync":
            if len(args) >= 2:
                return [
                    "// CUDA driver OpenGL unmap buffer object async: "
                    f"buffer object: {args[0]}, stream: {args[1]}"
                ]
        elif name == "cuGLUnregisterBufferObject":
            if args:
                return [
                    "// CUDA driver OpenGL unregister buffer object: "
                    f"buffer object: {args[0]}"
                ]
        elif name == "cuGraphicsGLRegisterBuffer":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graphics GL register buffer: "
                    f"output: {output}, buffer: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cuGraphicsGLRegisterImage":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graphics GL register image: "
                    f"output: {output}, image: {args[1]}, "
                    f"target: {args[2]}, flags: {args[3]}"
                ]
        elif name in {
            "cuGraphicsD3D9RegisterResource",
            "cuGraphicsD3D10RegisterResource",
            "cuGraphicsD3D11RegisterResource",
        }:
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                api = name[len("cuGraphics") : -len("RegisterResource")]
                return [
                    f"// CUDA driver graphics {api} register resource: "
                    f"output: {output}, resource: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cuGraphicsUnregisterResource":
            if args:
                return [
                    f"// CUDA driver graphics unregister resource: resource: {args[0]}"
                ]
        elif name in {"cuGraphicsMapResources", "cuGraphicsUnmapResources"}:
            if len(args) >= 3:
                operation = "map" if name == "cuGraphicsMapResources" else "unmap"
                return [
                    f"// CUDA driver graphics {operation} resources: "
                    f"count: {args[0]}, resources: {args[1]}, stream: {args[2]}"
                ]
        elif name in {
            "cuGraphicsResourceSetMapFlags",
            "cuGraphicsResourceSetMapFlags_v2",
        }:
            if len(args) >= 2:
                return [
                    "// CUDA driver graphics resource set map flags: "
                    f"resource: {args[0]}, flags: {args[1]}"
                ]
        elif name in {
            "cuGraphicsResourceGetMappedPointer",
            "cuGraphicsResourceGetMappedPointer_v2",
        }:
            if len(node.args) >= 3:
                pointer_output = self.format_runtime_pointer_target(node.args[0])
                size_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver graphics mapped pointer: "
                    f"pointer output: {pointer_output}, "
                    f"size output: {size_output}, resource: {args[2]}"
                ]
        elif name == "cuGraphicsSubResourceGetMappedArray":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graphics subresource mapped array: "
                    f"output: {output}, resource: {args[1]}, "
                    f"array index: {args[2]}, mip level: {args[3]}"
                ]
        elif name == "cuGraphicsResourceGetMappedMipmappedArray":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graphics mapped mipmapped array: "
                    f"output: {output}, resource: {args[1]}"
                ]
        return None

    def describe_cuda_driver_memory_copy_call(self, node, name, args):
        """Comments for a driver memory copy or memory set call; None for any other name."""
        if name in {"cuMemcpy", "cuMemcpyAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// CUDA driver memory copy: {args[1]} -> {args[0]}, "
                    f"bytes: {args[2]}"
                )
                if name == "cuMemcpyAsync" and len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name in {
            "cuMemcpyHtoD",
            "cuMemcpyDtoH",
            "cuMemcpyDtoD",
            "cuMemcpyHtoDAsync",
            "cuMemcpyDtoHAsync",
            "cuMemcpyDtoDAsync",
        }:
            if len(args) >= 3:
                direction = name[len("cuMemcpy") :]
                if direction.endswith("Async"):
                    direction = direction[: -len("Async")]
                comment = (
                    f"// CUDA driver memory copy {direction}: "
                    f"{args[1]} -> {args[0]}, bytes: {args[2]}"
                )
                if name.endswith("Async") and len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name in {"cuMemcpy2D", "cuMemcpy2DAsync", "cuMemcpy2DUnaligned"}:
            if node.args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// CUDA driver memory copy 2D: params: {params}"
                if name == "cuMemcpy2DAsync" and len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name in {"cuMemcpy3D", "cuMemcpy3DAsync"}:
            if node.args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// CUDA driver memory copy 3D: params: {params}"
                if name == "cuMemcpy3DAsync" and len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name in {
            "cuMemsetD8",
            "cuMemsetD32",
            "cuMemsetD8Async",
            "cuMemsetD32Async",
        }:
            if len(args) >= 3:
                width = "D32" if "D32" in name else "D8"
                comment = (
                    f"// CUDA driver memory set {width}: {args[0]}, "
                    f"value: {args[1]}, count: {args[2]}"
                )
                if name.endswith("Async") and len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name in {
            "cuMemsetD2D8",
            "cuMemsetD2D32",
            "cuMemsetD2D8Async",
            "cuMemsetD2D32Async",
        }:
            if len(args) >= 5:
                width = "D32" if "D32" in name else "D8"
                comment = (
                    f"// CUDA driver memory set 2D {width}: {args[0]}, "
                    f"pitch: {args[1]}, value: {args[2]}, "
                    f"width: {args[3]}, height: {args[4]}"
                )
                if name.endswith("Async") and len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        return None
