"""HIP host memory allocation, copy, memory set and IPC memory calls in the HIP to CrossGL converter, described as comments."""


class HipRuntimeMemoryCallMixin:
    """Runtime memory call members of ``HipToCrossGLConverter``."""

    def describe_hip_memory_allocation_call(self, node, name, args):
        """Comments for a memory, host memory, array or mipmapped array allocation or free, an array query, or a memory pool call; None for any other name."""
        if name in {"hipMalloc", "hipMallocManaged"}:
            if len(node.args) >= 2:
                target = self.format_runtime_raw_output_target(node.args[0])
                size = self.visit(node.args[1])
                return [f"// HIP memory allocate: {target}, bytes: {size}"]
        elif name == "hipExtMallocWithFlags":
            if len(args) >= 3:
                target = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP extended memory allocate: {target}, "
                    f"bytes: {args[1]}, flags: {args[2]}"
                ]
        elif name == "hipMallocAsync":
            if len(args) >= 3:
                target = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP async memory allocate: {target}, bytes: {args[1]}, "
                    f"stream: {args[2]}"
                ]
        elif name == "hipMallocFromPoolAsync":
            if len(args) >= 4:
                target = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP async memory allocate from pool: {target}, "
                    f"bytes: {args[1]}, pool: {args[2]}, stream: {args[3]}"
                ]
        elif name in {"hipHostMalloc", "hipHostAlloc"}:
            if len(args) >= 2:
                target = self.format_runtime_raw_output_target(node.args[0])
                comment = f"// HIP host memory allocate: {target}, bytes: {args[1]}"
                if len(args) >= 3:
                    comment += f", flags: {args[2]}"
                return [comment]
        elif name == "hipMallocPitch":
            if len(args) >= 4:
                target = self.format_runtime_raw_output_target(node.args[0])
                pitch = self.format_runtime_raw_output_target(node.args[1])
                return [
                    f"// HIP pitched memory allocate: {target}, pitch: {pitch}, "
                    f"width: {args[2]}, height: {args[3]}"
                ]
        elif name == "hipMalloc3D":
            if len(args) >= 2:
                target = self.format_runtime_raw_output_target(node.args[0])
                return [f"// HIP 3D memory allocate: {target}, extent: {args[1]}"]
        elif name in {"hipMallocArray", "hipMalloc3DArray"}:
            if len(args) >= 4:
                target = self.format_runtime_raw_output_target(node.args[0])
                descriptor = self.format_runtime_pointer_target(node.args[1])
                if name == "hipMallocArray":
                    comment = (
                        f"// HIP array allocate: {target}, desc: {descriptor}, "
                        f"width: {args[2]}, height: {args[3]}"
                    )
                    if len(args) >= 5:
                        comment += f", flags: {args[4]}"
                else:
                    comment = (
                        f"// HIP 3D array allocate: {target}, desc: {descriptor}, "
                        f"extent: {args[2]}, flags: {args[3]}"
                    )
                return [comment]
        elif name == "hipMallocMipmappedArray":
            if len(args) >= 5:
                target = self.format_runtime_raw_output_target(node.args[0])
                descriptor = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// HIP mipmapped array allocate: output: {target}, "
                    f"desc: {descriptor}, extent: {args[2]}, levels: {args[3]}, "
                    f"flags: {args[4]}"
                ]
        elif name == "hipMipmappedArrayCreate":
            if len(args) >= 3:
                target = self.format_runtime_raw_output_target(node.args[0])
                descriptor = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// HIP mipmapped array create: output: {target}, "
                    f"descriptor: {descriptor}, levels: {args[2]}"
                ]
        elif name in {"hipGetMipmappedArrayLevel", "hipMipmappedArrayGetLevel"}:
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP mipmapped array get level: output: {output}, "
                    f"mipmapped array: {args[1]}, level: {args[2]}"
                ]
        elif name == "hipMipmappedArrayDestroy":
            if args:
                return [f"// HIP free mipmapped array: {args[0]}"]
        elif name in {"hipFree", "hipHostFree", "hipFreeHost"}:
            if args:
                return [f"// HIP memory free: {args[0]}"]
        elif name == "hipFreeAsync":
            if len(args) >= 2:
                return [f"// HIP async memory free: {args[0]}, stream: {args[1]}"]
        elif name in {"hipFreeArray", "hipArrayDestroy"}:
            if args:
                return [f"// HIP array free: {args[0]}"]
        elif name in {"hipArrayCreate", "hipArray3DCreate"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                descriptor = self.format_runtime_pointer_target(node.args[1])
                dimension = "3D " if name == "hipArray3DCreate" else ""
                return [
                    f"// HIP {dimension}array create: output: {output}, "
                    f"descriptor: {descriptor}"
                ]
        elif name in {"hipArrayGetDescriptor", "hipArray3DGetDescriptor"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                dimension = "3D " if name == "hipArray3DGetDescriptor" else ""
                if output_name is not None:
                    if name == "hipArray3DGetDescriptor":
                        members = self.HIP_ARRAY3D_DESCRIPTOR_MEMBERS
                        query_prefix = "array.descriptor3D"
                    else:
                        members = self.HIP_ARRAY_DESCRIPTOR_MEMBERS
                        query_prefix = "array.descriptor"
                    self.register_member_query_sources(
                        output_name,
                        {
                            member: f"{query_prefix}.{member}({args[1]})"
                            for member in members
                        },
                    )
                return [
                    f"// HIP array get {dimension}descriptor: output: {output}, "
                    f"array: {args[1]}"
                ]
        elif name == "hipArrayGetInfo":
            if len(args) >= 4:
                descriptor_output = self.format_runtime_raw_output_target(node.args[0])
                extent_output = self.format_runtime_raw_output_target(node.args[1])
                flags_output = self.format_runtime_raw_output_target(node.args[2])
                descriptor_output_name = self.get_runtime_pointer_target_name(
                    node.args[0]
                )
                extent_output_name = self.get_runtime_pointer_target_name(node.args[1])
                flags_output_name = self.get_runtime_pointer_target_name(node.args[2])
                if descriptor_output_name is not None:
                    self.register_member_query_sources(
                        descriptor_output_name,
                        {
                            member: f"array.info.channelDesc.{member}({args[3]})"
                            for member in self.HIP_CHANNEL_DESCRIPTOR_MEMBERS
                        },
                    )
                if extent_output_name is not None:
                    self.register_member_query_sources(
                        extent_output_name,
                        {
                            member: f"array.info.extent.{member}({args[3]})"
                            for member in self.HIP_EXTENT_MEMBERS
                        },
                    )
                if flags_output_name is not None:
                    self.register_device_query_source(
                        flags_output_name,
                        f"array.info.flags({args[3]})",
                    )
                return [
                    f"// HIP array get info: desc output: {descriptor_output}, "
                    f"extent output: {extent_output}, flags output: {flags_output}, "
                    f"array: {args[3]}"
                ]
        elif name == "hipMemPoolCreate":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory pool create: output: {output}, properties: {args[1]}"
                ]
        elif name == "hipMemPoolDestroy":
            if args:
                return [f"// HIP memory pool destroy: {args[0]}"]
        elif name == "hipMemPoolTrimTo":
            if len(args) >= 2:
                return [
                    f"// HIP memory pool trim: pool: {args[0]}, "
                    f"minimum bytes: {args[1]}"
                ]
        elif name == "hipMemPoolSetAttribute":
            if len(args) >= 3:
                value = self.format_runtime_pointer_target(node.args[2])
                return [
                    f"// HIP memory pool set attribute: pool: {args[0]}, "
                    f"attribute: {args[1]}, value: {value}"
                ]
        elif name == "hipMemPoolGetAttribute":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[2])
                output_name = self.get_runtime_pointer_target_name(node.args[2])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"memoryPool.attribute({args[0]}, {args[1]})",
                    )
                return [
                    f"// HIP memory pool get attribute: pool: {args[0]}, "
                    f"attribute: {args[1]}, output: {output}"
                ]
        elif name == "hipMemPoolSetAccess":
            if len(args) >= 3:
                return [
                    f"// HIP memory pool set access: pool: {args[0]}, "
                    f"descriptors: {args[1]}, count: {args[2]}"
                ]
        elif name == "hipMemPoolGetAccess":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                location = self.format_runtime_pointer_target(node.args[2])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"memoryPool.accessFlags({args[1]}, {location})",
                    )
                return [
                    f"// HIP memory pool get access: output: {output}, "
                    f"pool: {args[1]}, location: {args[2]}"
                ]
        elif name == "hipMemPoolExportToShareableHandle":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory pool export to shareable handle: output: {output}, "
                    f"pool: {args[1]}, handle type: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipMemPoolImportFromShareableHandle":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory pool import from shareable handle: output: {output}, "
                    f"handle: {args[1]}, handle type: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipMemPoolExportPointer":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory pool export pointer: output: {output}, "
                    f"pointer: {args[1]}"
                ]
        elif name == "hipMemPoolImportPointer":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory pool import pointer: output: {output}, "
                    f"pool: {args[1]}, export data: {args[2]}"
                ]
        elif name == "hipDeviceGetDefaultMemPool":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP get default memory pool: output: {output}, "
                    f"device: {args[1]}"
                ]
        elif name == "hipDeviceSetMemPool":
            if len(args) >= 2:
                return [
                    f"// HIP set device memory pool: device: {args[0]}, pool: {args[1]}"
                ]
        elif name == "hipDeviceGetMemPool":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP get device memory pool: output: {output}, "
                    f"device: {args[1]}"
                ]
        return None

    def describe_hip_virtual_memory_call(self, node, name, args):
        """Comments for a prefetch, memory advice, range attribute, virtual memory, shareable handle, host registration, linear copy or peer copy call; None for any other name."""
        if name == "hipMemPrefetchAsync":
            if len(args) >= 4:
                return [
                    f"// HIP memory prefetch: pointer: {args[0]}, bytes: {args[1]}, "
                    f"device: {args[2]}, stream: {args[3]}"
                ]
        elif name == "hipMemPrefetchAsync_v2":
            if len(args) >= 5:
                return [
                    f"// HIP memory prefetch v2: pointer: {args[0]}, "
                    f"bytes: {args[1]}, location: {args[2]}, flags: {args[3]}, "
                    f"stream: {args[4]}"
                ]
        elif name == "hipMemAdvise":
            if len(args) >= 4:
                return [
                    f"// HIP memory advise: pointer: {args[0]}, bytes: {args[1]}, "
                    f"advice: {args[2]}, device: {args[3]}"
                ]
        elif name == "hipMemAdvise_v2":
            if len(args) >= 4:
                return [
                    f"// HIP memory advise v2: pointer: {args[0]}, "
                    f"bytes: {args[1]}, advice: {args[2]}, location: {args[3]}"
                ]
        elif name == "hipMemRangeGetAttribute":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"memory.rangeAttribute({args[2]}, {args[3]}, {args[4]})",
                    )
                return [
                    f"// HIP memory range get attribute: output: {output}, "
                    f"output bytes: {args[1]}, attribute: {args[2]}, "
                    f"pointer: {args[3]}, range bytes: {args[4]}"
                ]
        elif name == "hipMemRangeGetAttributes":
            if len(args) >= 6:
                return [
                    f"// HIP memory range get attributes: outputs: {args[0]}, "
                    f"output sizes: {args[1]}, attributes: {args[2]}, "
                    f"attribute count: {args[3]}, pointer: {args[4]}, "
                    f"range bytes: {args[5]}"
                ]
        elif name == "hipStreamAttachMemAsync":
            if len(args) >= 4:
                return [
                    f"// HIP stream attach memory: stream: {args[0]}, "
                    f"pointer: {args[1]}, bytes: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipMemGetAllocationGranularity":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                properties = self.format_runtime_pointer_target(node.args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        "virtualMemory.allocationGranularity("
                        f"{properties}, {args[2]})",
                    )
                return [
                    f"// HIP virtual memory allocation granularity: output: {output}, "
                    f"properties: {args[1]}, option: {args[2]}"
                ]
        elif name == "hipMemCreate":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory create allocation: output: {output}, "
                    f"bytes: {args[1]}, properties: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipMemRelease":
            if args:
                return [f"// HIP virtual memory release allocation: {args[0]}"]
        elif name == "hipMemAddressReserve":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory reserve address: output: {output}, "
                    f"bytes: {args[1]}, alignment: {args[2]}, address: {args[3]}, "
                    f"flags: {args[4]}"
                ]
        elif name == "hipMemAddressFree":
            if len(args) >= 2:
                return [
                    f"// HIP virtual memory free address: pointer: {args[0]}, "
                    f"bytes: {args[1]}"
                ]
        elif name == "hipMemMap":
            if len(args) >= 5:
                return [
                    f"// HIP virtual memory map: pointer: {args[0]}, "
                    f"bytes: {args[1]}, offset: {args[2]}, handle: {args[3]}, "
                    f"flags: {args[4]}"
                ]
        elif name == "hipMemUnmap":
            if len(args) >= 2:
                return [
                    f"// HIP virtual memory unmap: pointer: {args[0]}, "
                    f"bytes: {args[1]}"
                ]
        elif name == "hipMemSetAccess":
            if len(args) >= 4:
                return [
                    f"// HIP virtual memory set access: pointer: {args[0]}, "
                    f"bytes: {args[1]}, descriptors: {args[2]}, count: {args[3]}"
                ]
        elif name == "hipMemGetAccess":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                location = self.format_runtime_pointer_target(node.args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"virtualMemory.accessFlags({location}, {args[2]})",
                    )
                return [
                    f"// HIP virtual memory get access: output: {output}, "
                    f"location: {args[1]}, pointer: {args[2]}"
                ]
        elif name == "hipMemGetAllocationPropertiesFromHandle":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory allocation properties: output: {output}, "
                    f"handle: {args[1]}"
                ]
        elif name == "hipMemRetainAllocationHandle":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory retain allocation handle: output: {output}, "
                    f"address: {args[1]}"
                ]
        elif name == "hipMemExportToShareableHandle":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory export shareable handle: output: {output}, "
                    f"handle: {args[1]}, handle type: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipMemImportFromShareableHandle":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory import shareable handle: output: {output}, "
                    f"shareable handle: {args[1]}, handle type: {args[2]}"
                ]
        elif name == "hipHostRegister":
            if len(args) >= 3:
                return [
                    f"// HIP host memory register: {args[0]}, bytes: {args[1]}, "
                    f"flags: {args[2]}"
                ]
        elif name == "hipHostUnregister":
            if args:
                return [f"// HIP host memory unregister: {args[0]}"]
        elif name == "hipHostGetDevicePointer":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP host device pointer: output: {output}, host: {args[1]}, "
                    f"flags: {args[2]}"
                ]
        elif name == "hipHostGetFlags":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"hostMemory.flags({args[1]})",
                    )
                return [f"// HIP host memory flags: output: {output}, host: {args[1]}"]
        elif name in {"hipMemcpy", "hipMemcpyAsync", "hipMemcpyWithStream"}:
            if len(args) >= 4:
                comment = (
                    f"// HIP memory copy: {args[1]} -> {args[0]}, "
                    f"bytes: {args[2]}, kind: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", stream: {args[4]}"
                return [comment]
        elif name in {"hipMemcpyPeer", "hipMemcpyPeerAsync"}:
            if len(args) >= 5:
                comment = (
                    f"// HIP peer memory copy: source: {args[2]}, "
                    f"source device: {args[3]}, destination: {args[0]}, "
                    f"destination device: {args[1]}, bytes: {args[4]}"
                )
                if len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name in {"hipMemcpy2D", "hipMemcpy2DAsync"}:
            if len(args) >= 7:
                comment = (
                    f"// HIP 2D memory copy: {args[2]} -> {args[0]}, "
                    f"dst pitch: {args[1]}, src pitch: {args[3]}, "
                    f"width: {args[4]}, height: {args[5]}, kind: {args[6]}"
                )
                if len(args) >= 8:
                    comment += f", stream: {args[7]}"
                return [comment]
        return None

    def describe_hip_memory_copy_call(self, node, name, args):
        """Comments for an array, 3D, parameter, batch or symbol copy, a symbol query, a memory set, or a driver allocation call; None for any other name."""
        if name in {"hipMemcpyToArray", "hipMemcpyToArrayAsync"}:
            if len(args) >= 6:
                comment = (
                    f"// HIP memory copy to array: source: {args[3]}, "
                    f"destination array: {args[0]}, w offset: {args[1]}, "
                    f"h offset: {args[2]}, bytes: {args[4]}, kind: {args[5]}"
                )
                if len(args) >= 7:
                    comment += f", stream: {args[6]}"
                return [comment]
        elif name in {"hipMemcpyFromArray", "hipMemcpyFromArrayAsync"}:
            if len(args) >= 6:
                comment = (
                    f"// HIP memory copy from array: source array: {args[1]}, "
                    f"w offset: {args[2]}, h offset: {args[3]}, "
                    f"destination: {args[0]}, bytes: {args[4]}, kind: {args[5]}"
                )
                if len(args) >= 7:
                    comment += f", stream: {args[6]}"
                return [comment]
        elif name in {"hipMemcpy2DToArray", "hipMemcpy2DToArrayAsync"}:
            if len(args) >= 8:
                comment = (
                    f"// HIP 2D memory copy to array: source: {args[3]}, "
                    f"source pitch: {args[4]}, destination array: {args[0]}, "
                    f"w offset: {args[1]}, h offset: {args[2]}, width: {args[5]}, "
                    f"height: {args[6]}, kind: {args[7]}"
                )
                if len(args) >= 9:
                    comment += f", stream: {args[8]}"
                return [comment]
        elif name in {"hipMemcpy2DFromArray", "hipMemcpy2DFromArrayAsync"}:
            if len(args) >= 8:
                comment = (
                    f"// HIP 2D memory copy from array: source array: {args[2]}, "
                    f"w offset: {args[3]}, h offset: {args[4]}, "
                    f"destination: {args[0]}, destination pitch: {args[1]}, "
                    f"width: {args[5]}, height: {args[6]}, kind: {args[7]}"
                )
                if len(args) >= 9:
                    comment += f", stream: {args[8]}"
                return [comment]
        elif name == "hipMemcpyArrayToArray":
            if len(args) >= 8:
                return [
                    f"// HIP memory copy array to array: source array: {args[3]}, "
                    f"source w offset: {args[4]}, source h offset: {args[5]}, "
                    f"destination array: {args[0]}, destination w offset: {args[1]}, "
                    f"destination h offset: {args[2]}, bytes: {args[6]}, "
                    f"kind: {args[7]}"
                ]
        elif name == "hipMemcpy2DArrayToArray":
            if len(args) >= 9:
                return [
                    f"// HIP 2D memory copy array to array: source array: {args[3]}, "
                    f"source w offset: {args[4]}, source h offset: {args[5]}, "
                    f"destination array: {args[0]}, destination w offset: {args[1]}, "
                    f"destination h offset: {args[2]}, width: {args[6]}, "
                    f"height: {args[7]}, kind: {args[8]}"
                ]
        elif name in {"hipMemcpy3D", "hipMemcpy3DAsync"}:
            if args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP 3D memory copy: params: {params}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name in {"hipMemcpy3DPeer", "hipMemcpy3DPeerAsync"}:
            if args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP 3D peer memory copy: params: {params}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name in {"hipMemcpyParam2D", "hipMemcpyParam2DAsync"}:
            if args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP 2D parameterized memory copy: params: {params}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name in {"hipDrvMemcpy3D", "hipDrvMemcpy3DAsync"}:
            if args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP driver 3D memory copy: params: {params}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name == "hipMemcpyBatchAsync":
            if len(args) >= 9:
                fail_output = self.format_runtime_raw_output_target(node.args[7])
                return [
                    f"// HIP batched memory copy: destinations: {args[0]}, "
                    f"sources: {args[1]}, sizes: {args[2]}, count: {args[3]}, "
                    f"attributes: {args[4]}, attribute indices: {args[5]}, "
                    f"attribute count: {args[6]}, fail index output: {fail_output}, "
                    f"stream: {args[8]}"
                ]
        elif name == "hipMemcpy3DBatchAsync":
            if len(args) >= 5:
                fail_output = self.format_runtime_raw_output_target(node.args[2])
                return [
                    f"// HIP batched 3D memory copy: count: {args[0]}, "
                    f"operations: {args[1]}, fail index output: {fail_output}, "
                    f"flags: {args[3]}, stream: {args[4]}"
                ]
        elif name == "hipGetSymbolAddress":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP get symbol address: output: {output}, "
                    f"symbol: {args[1]}"
                ]
        elif name == "hipGetSymbolSize":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"symbol.size({args[1]})",
                    )
                return [f"// HIP get symbol size: output: {output}, symbol: {args[1]}"]
        elif name in {"hipMemcpyToSymbol", "hipMemcpyToSymbolAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// HIP symbol copy to: {args[0]}, source: {args[1]}, "
                    f"bytes: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", offset: {args[3]}"
                if len(args) >= 5:
                    comment += f", kind: {args[4]}"
                if len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name in {"hipMemcpyFromSymbol", "hipMemcpyFromSymbolAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// HIP symbol copy from: {args[1]}, destination: {args[0]}, "
                    f"bytes: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", offset: {args[3]}"
                if len(args) >= 5:
                    comment += f", kind: {args[4]}"
                if len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name in {"hipMemset", "hipMemsetAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// HIP memory set: {args[0]}, value: {args[1]}, "
                    f"bytes: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name in {"hipMemset2D", "hipMemset2DAsync"}:
            if len(args) >= 5:
                comment = (
                    f"// HIP 2D memory set: {args[0]}, pitch: {args[1]}, "
                    f"value: {args[2]}, width: {args[3]}, height: {args[4]}"
                )
                if len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name in {"hipMemset3D", "hipMemset3DAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// HIP 3D memory set: {args[0]}, value: {args[1]}, "
                    f"extent: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name == "hipMemAlloc":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP driver memory allocate: output: {output}, "
                    f"bytes: {args[1]}"
                ]
        elif name == "hipMemAllocPitch":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[0])
                pitch = self.format_runtime_raw_output_target(node.args[1])
                return [
                    f"// HIP driver pitched memory allocate: output: {output}, "
                    f"pitch output: {pitch}, width: {args[2]}, height: {args[3]}, "
                    f"element bytes: {args[4]}"
                ]
        elif name == "hipMemFree":
            if args:
                return [f"// HIP driver memory free: {args[0]}"]
        elif name in {"hipMemAllocHost", "hipMemHostAlloc"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                comment = (
                    f"// HIP driver host memory allocate: output: {output}, "
                    f"bytes: {args[1]}"
                )
                if len(args) >= 3:
                    comment += f", flags: {args[2]}"
                return [comment]
        elif name == "hipMemFreeHost":
            if args:
                return [f"// HIP driver host memory free: {args[0]}"]
        return None

    def describe_hip_driver_memory_call(self, node, name, args):
        """Comments for a driver host pointer, address range, typed copy, typed memory set or IPC memory handle call; None for any other name."""
        if name == "hipMemHostGetDevicePointer":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP driver host device pointer: output: {output}, "
                    f"host: {args[1]}, flags: {args[2]}"
                ]
        elif name == "hipMemGetAddressRange":
            if len(args) >= 3:
                base_output = self.format_runtime_pointer_target(node.args[0])
                size_output = self.format_runtime_pointer_target(node.args[1])
                base_output_name = self.get_runtime_pointer_target_name(node.args[0])
                size_output_name = self.get_runtime_pointer_target_name(node.args[1])
                if base_output_name is not None:
                    self.register_device_query_source(
                        base_output_name,
                        f"memory.addressRange.base({args[2]})",
                    )
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"memory.addressRange.size({args[2]})",
                    )
                return [
                    f"// HIP driver memory address range: base output: {base_output}, "
                    f"size output: {size_output}, pointer: {args[2]}"
                ]
        elif name in {
            "hipMemcpyHtoD",
            "hipMemcpyHtoDAsync",
            "hipMemcpyDtoH",
            "hipMemcpyDtoHAsync",
            "hipMemcpyDtoD",
            "hipMemcpyDtoDAsync",
        }:
            if len(args) >= 3:
                copy_kind = {
                    "hipMemcpyHtoD": "host to device",
                    "hipMemcpyHtoDAsync": "host to device",
                    "hipMemcpyDtoH": "device to host",
                    "hipMemcpyDtoHAsync": "device to host",
                    "hipMemcpyDtoD": "device to device",
                    "hipMemcpyDtoDAsync": "device to device",
                }[name]
                comment = (
                    f"// HIP driver memory copy {copy_kind}: "
                    f"source: {args[1]}, destination: {args[0]}, bytes: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name in {"hipMemcpyAtoH", "hipMemcpyAtoHAsync"}:
            if len(args) >= 4:
                comment = (
                    "// HIP driver memory copy array to host: "
                    f"source array: {args[1]}, source offset: {args[2]}, "
                    f"destination host: {args[0]}, bytes: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", stream: {args[4]}"
                return [comment]
        elif name in {"hipMemcpyHtoA", "hipMemcpyHtoAAsync"}:
            if len(args) >= 4:
                comment = (
                    "// HIP driver memory copy host to array: "
                    f"source host: {args[2]}, destination array: {args[0]}, "
                    f"destination offset: {args[1]}, bytes: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", stream: {args[4]}"
                return [comment]
        elif name == "hipMemcpyAtoD":
            if len(args) >= 4:
                return [
                    "// HIP driver memory copy array to device: "
                    f"source array: {args[1]}, source offset: {args[2]}, "
                    f"destination device: {args[0]}, bytes: {args[3]}"
                ]
        elif name == "hipMemcpyDtoA":
            if len(args) >= 4:
                return [
                    "// HIP driver memory copy device to array: "
                    f"source device: {args[2]}, destination array: {args[0]}, "
                    f"destination offset: {args[1]}, bytes: {args[3]}"
                ]
        elif name == "hipMemcpyAtoA":
            if len(args) >= 5:
                return [
                    "// HIP driver memory copy array to array: "
                    f"source array: {args[2]}, source offset: {args[3]}, "
                    f"destination array: {args[0]}, destination offset: {args[1]}, "
                    f"bytes: {args[4]}"
                ]
        elif name in {
            "hipMemsetD2D8",
            "hipMemsetD2D8Async",
            "hipMemsetD2D16",
            "hipMemsetD2D16Async",
            "hipMemsetD2D32",
            "hipMemsetD2D32Async",
        }:
            if len(args) >= 5:
                width = {
                    "hipMemsetD2D8": "8-bit",
                    "hipMemsetD2D8Async": "8-bit",
                    "hipMemsetD2D16": "16-bit",
                    "hipMemsetD2D16Async": "16-bit",
                    "hipMemsetD2D32": "32-bit",
                    "hipMemsetD2D32Async": "32-bit",
                }[name]
                comment = (
                    f"// HIP driver 2D memory set {width}: pointer: {args[0]}, "
                    f"pitch: {args[1]}, value: {args[2]}, width: {args[3]}, "
                    f"height: {args[4]}"
                )
                if len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name in {
            "hipMemsetD8",
            "hipMemsetD8Async",
            "hipMemsetD16",
            "hipMemsetD16Async",
            "hipMemsetD32",
            "hipMemsetD32Async",
        }:
            if len(args) >= 3:
                width = {
                    "hipMemsetD8": "8-bit",
                    "hipMemsetD8Async": "8-bit",
                    "hipMemsetD16": "16-bit",
                    "hipMemsetD16Async": "16-bit",
                    "hipMemsetD32": "32-bit",
                    "hipMemsetD32Async": "32-bit",
                }[name]
                comment = (
                    f"// HIP driver memory set {width}: pointer: {args[0]}, "
                    f"value: {args[1]}, count: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name == "hipIpcGetMemHandle":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP IPC get memory handle: output: {output}, "
                    f"pointer: {args[1]}"
                ]
        elif name == "hipIpcOpenMemHandle":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP IPC open memory handle: output: {output}, "
                    f"handle: {args[1]}, flags: {args[2]}"
                ]
        elif name == "hipIpcCloseMemHandle":
            if args:
                return [f"// HIP IPC close memory handle: pointer: {args[0]}"]
        return None
