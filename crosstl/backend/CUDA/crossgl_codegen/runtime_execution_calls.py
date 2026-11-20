"""CUDA host module, device, context, stream, event and external resource calls in the CUDA to CrossGL converter, described as comments."""


class CudaImportRuntimeExecutionCallMixin:
    """Runtime execution call members of ``CudaToCrossGLConverter``."""

    def describe_cuda_module_call(self, node, name, args):
        """Comments for a module, linker, kernel launch, occupancy or function attribute call; None for any other name."""
        if name == "cuModuleLoad":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver module load: output: {output}, path: {args[1]}"
                ]
        elif name == "cuModuleLoadData":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver module load data: "
                    f"output: {output}, image: {args[1]}"
                ]
        elif name == "cuModuleLoadDataEx":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver module load data with options: "
                    f"output: {output}, image: {args[1]}, "
                    f"option count: {args[2]}, options: {args[3]}, "
                    f"option values: {args[4]}"
                ]
        elif name == "cuLinkCreate":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[3])
                return [
                    "// CUDA driver linker create: "
                    f"output: {output}, option count: {args[0]}, "
                    f"options: {args[1]}, option values: {args[2]}"
                ]
        elif name == "cuLinkAddData":
            if len(args) >= 8:
                return [
                    "// CUDA driver linker add data: "
                    f"state: {args[0]}, type: {args[1]}, data: {args[2]}, "
                    f"bytes: {args[3]}, name: {args[4]}, option count: {args[5]}, "
                    f"options: {args[6]}, option values: {args[7]}"
                ]
        elif name == "cuLinkAddFile":
            if len(args) >= 6:
                return [
                    "// CUDA driver linker add file: "
                    f"state: {args[0]}, type: {args[1]}, path: {args[2]}, "
                    f"option count: {args[3]}, options: {args[4]}, "
                    f"option values: {args[5]}"
                ]
        elif name == "cuLinkComplete":
            if len(node.args) >= 3:
                cubin_output = self.format_runtime_pointer_target(node.args[1])
                size_output = self.format_runtime_pointer_target(node.args[2])
                return [
                    "// CUDA driver linker complete: "
                    f"state: {args[0]}, cubin output: {cubin_output}, "
                    f"size output: {size_output}"
                ]
        elif name == "cuLinkDestroy":
            if args:
                return [f"// CUDA driver linker destroy: state: {args[0]}"]
        elif name == "cuModuleGetFunction":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver module get function: "
                    f"output: {output}, module: {args[1]}, name: {args[2]}"
                ]
        elif name == "cuModuleGetTexRef":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver module get texture reference: "
                    f"output: {output}, module: {args[1]}, name: {args[2]}"
                ]
        elif name == "cuModuleGetSurfRef":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver module get surface reference: "
                    f"output: {output}, module: {args[1]}, name: {args[2]}"
                ]
        elif name == "cuModuleGetGlobal":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                byte_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver module get global: "
                    f"output: {output}, bytes output: {byte_output}, "
                    f"module: {args[2]}, name: {args[3]}"
                ]
        elif name == "cuModuleUnload":
            if args:
                return [f"// CUDA driver module unload: {args[0]}"]
        elif name in {"cuLaunchKernel", "cuLaunchCooperativeKernel"}:
            if len(args) >= 10:
                launch_kind = (
                    "launch cooperative kernel"
                    if name == "cuLaunchCooperativeKernel"
                    else "launch kernel"
                )
                comment = (
                    f"// CUDA driver {launch_kind}: function: {args[0]}, "
                    f"grid: {args[1]} x {args[2]} x {args[3]}, "
                    f"block: {args[4]} x {args[5]} x {args[6]}, "
                    f"shared memory: {args[7]}, stream: {args[8]}, "
                    f"params: {args[9]}"
                )
                if name == "cuLaunchKernel" and len(args) >= 11:
                    comment += f", extra: {args[10]}"
                return [comment]
        elif name == "cuLaunchCooperativeKernelMultiDevice":
            if len(node.args) >= 3:
                params = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver launch cooperative kernel multi-device: "
                    f"params: {params}, count: {args[1]}, flags: {args[2]}"
                ]
        elif name in {
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        }:
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                comment = (
                    "// CUDA driver occupancy active blocks: "
                    f"output: {output}, function: {args[1]}, "
                    f"block size: {args[2]}, dynamic shared memory: {args[3]}"
                )
                if name.endswith("WithFlags") and len(args) >= 5:
                    comment += f", flags: {args[4]}"
                return [comment]
        elif name in {
            "cuOccupancyMaxPotentialBlockSize",
            "cuOccupancyMaxPotentialBlockSizeWithFlags",
        }:
            if len(node.args) >= 6:
                min_grid_output = self.format_runtime_pointer_target(node.args[0])
                block_size_output = self.format_runtime_pointer_target(node.args[1])
                flags = (
                    args[6] if name.endswith("WithFlags") and len(args) >= 7 else "0"
                )
                return [
                    "// CUDA driver occupancy potential block size: "
                    f"min grid output: {min_grid_output}, "
                    f"block size output: {block_size_output}, "
                    f"function: {args[2]}, dynamic shared memory callback: {args[3]}, "
                    f"dynamic shared memory: {args[4]}, block size limit: {args[5]}, "
                    f"flags: {flags}"
                ]
        elif name == "cuFuncGetAttribute":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver function get attribute: "
                    f"output: {output}, attribute: {args[1]}, function: {args[2]}"
                ]
        elif name == "cuFuncSetAttribute":
            if len(args) >= 3:
                return [
                    "// CUDA driver function set attribute: "
                    f"function: {args[0]}, attribute: {args[1]}, value: {args[2]}"
                ]
        elif name == "cuFuncSetCacheConfig":
            if len(args) >= 2:
                return [
                    "// CUDA driver function set cache config: "
                    f"function: {args[0]}, config: {args[1]}"
                ]
        elif name == "cuFuncSetSharedMemConfig":
            if len(args) >= 2:
                return [
                    "// CUDA driver function set shared memory config: "
                    f"function: {args[0]}, config: {args[1]}"
                ]
        return None

    def describe_cuda_device_call(self, node, name, args):
        """Comments for a driver initialization, device, context or primary context call; None for any other name."""
        if name == "cuInit":
            flags = args[0] if args else "0"
            return [f"// CUDA driver initialize: flags: {flags}"]
        elif name == "cuDriverGetVersion":
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA driver get version: output: {output}"]
        elif name == "cuDeviceGet":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver get device: output: {output}, "
                    f"ordinal: {args[1]}"
                ]
        elif name == "cuDeviceGetCount":
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA driver get device count: output: {output}"]
        elif name == "cuDeviceGetName":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver get device name: output: {output}, "
                    f"length: {args[1]}, device: {args[2]}"
                ]
        elif name == "cuDeviceGetUuid":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver get device UUID: output: {output}, "
                    f"device: {args[1]}"
                ]
        elif name in {"cuDeviceTotalMem", "cuDeviceTotalMem_v2"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver get device total memory: "
                    f"output: {output}, device: {args[1]}"
                ]
        elif name == "cuDeviceGetAttribute":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver get device attribute: "
                    f"output: {output}, attribute: {args[1]}, device: {args[2]}"
                ]
        elif name == "cuDeviceComputeCapability":
            if len(node.args) >= 3:
                major_output = self.format_runtime_pointer_target(node.args[0])
                minor_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver get device compute capability: "
                    f"major output: {major_output}, minor output: {minor_output}, "
                    f"device: {args[2]}"
                ]
        elif name == "cuDeviceGetProperties":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver get device properties: "
                    f"output: {output}, device: {args[1]}"
                ]
        elif name == "cuDeviceGetP2PAttribute":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver get device P2P attribute: "
                    f"output: {output}, attribute: {args[1]}, "
                    f"source device: {args[2]}, destination device: {args[3]}"
                ]
        elif name == "cuDeviceCanAccessPeer":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver device peer access query: "
                    f"output: {output}, device: {args[1]}, peer device: {args[2]}"
                ]
        elif name == "cuDeviceGetLuid":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                node_mask_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver get device LUID: "
                    f"output: {output}, node mask output: {node_mask_output}, "
                    f"device: {args[2]}"
                ]
        elif name == "cuDeviceGetByPCIBusId":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver get device by PCI bus ID: "
                    f"output: {output}, bus ID: {args[1]}"
                ]
        elif name == "cuDeviceGetPCIBusId":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver get device PCI bus ID: "
                    f"output: {output}, length: {args[1]}, device: {args[2]}"
                ]
        elif name in {"cuCtxCreate", "cuCtxCreate_v2"}:
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver context create: output: {output}, "
                    f"flags: {args[1]}, device: {args[2]}"
                ]
        elif name in {"cuCtxDestroy", "cuCtxDestroy_v2"}:
            if args:
                return [f"// CUDA driver context destroy: {args[0]}"]
        elif name == "cuCtxSetCurrent":
            if args:
                return [f"// CUDA driver context set current: {args[0]}"]
        elif name == "cuCtxGetCurrent":
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA driver context get current: output: {output}"]
        elif name in {"cuCtxPushCurrent", "cuCtxPushCurrent_v2"}:
            if args:
                return [f"// CUDA driver context push current: {args[0]}"]
        elif name in {"cuCtxPopCurrent", "cuCtxPopCurrent_v2"}:
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA driver context pop current: output: {output}"]
        elif name == "cuCtxGetDevice":
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA driver context get device: output: {output}"]
        elif name == "cuCtxGetFlags":
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA driver context get flags: output: {output}"]
        elif name == "cuCtxGetId":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA driver context get id: context: {args[0]}, "
                    f"output: {output}"
                ]
        elif name == "cuCtxGetLimit":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver context get limit: output: {output}, "
                    f"limit: {args[1]}"
                ]
        elif name == "cuCtxSetLimit":
            if len(args) >= 2:
                return [
                    f"// CUDA driver context set limit: limit: {args[0]}, "
                    f"value: {args[1]}"
                ]
        elif name == "cuCtxGetCacheConfig":
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA driver context get cache config: output: {output}"]
        elif name == "cuCtxSetCacheConfig":
            if args:
                return [f"// CUDA driver context set cache config: config: {args[0]}"]
        elif name == "cuCtxGetSharedMemConfig":
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver context get shared memory config: "
                    f"output: {output}"
                ]
        elif name == "cuCtxSetSharedMemConfig":
            if args:
                return [
                    "// CUDA driver context set shared memory config: "
                    f"config: {args[0]}"
                ]
        elif name == "cuCtxEnablePeerAccess":
            if args:
                flags = args[1] if len(args) >= 2 else "0"
                return [
                    f"// CUDA driver context enable peer access: "
                    f"peer: {args[0]}, flags: {flags}"
                ]
        elif name == "cuCtxDisablePeerAccess":
            if args:
                return [f"// CUDA driver context disable peer access: peer: {args[0]}"]
        elif name == "cuCtxSynchronize":
            return ["// CUDA driver context synchronize"]
        elif name == "cuDevicePrimaryCtxRetain":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver device primary context retain: "
                    f"output: {output}, device: {args[1]}"
                ]
        elif name == "cuDevicePrimaryCtxGetState":
            if len(node.args) >= 3:
                flags_output = self.format_runtime_pointer_target(node.args[1])
                active_output = self.format_runtime_pointer_target(node.args[2])
                return [
                    "// CUDA driver device primary context get state: "
                    f"device: {args[0]}, flags output: {flags_output}, "
                    f"active output: {active_output}"
                ]
        elif name in {"cuDevicePrimaryCtxSetFlags", "cuDevicePrimaryCtxSetFlags_v2"}:
            if len(args) >= 2:
                return [
                    "// CUDA driver device primary context set flags: "
                    f"device: {args[0]}, flags: {args[1]}"
                ]
        elif name in {"cuDevicePrimaryCtxRelease", "cuDevicePrimaryCtxRelease_v2"}:
            if args:
                return [
                    "// CUDA driver device primary context release: "
                    f"device: {args[0]}"
                ]
        elif name in {"cuDevicePrimaryCtxReset", "cuDevicePrimaryCtxReset_v2"}:
            if args:
                return [
                    f"// CUDA driver device primary context reset: device: {args[0]}"
                ]
        return None

    def describe_cuda_stream_call(self, node, name, args):
        """Comments for a synchronization, stream, event, profiler, host callback, stream capture or error query call; None for any other name."""
        if name in {"cudaDeviceSynchronize", "cudaStreamSynchronize"}:
            if args:
                return [f"// CUDA synchronize: {args[0]}"]
            return ["// CUDA device synchronize"]
        elif name in {
            "cudaStreamCreate",
            "cudaStreamCreateWithFlags",
            "cudaStreamCreateWithPriority",
            "cudaStreamDestroy",
        }:
            if args:
                action = "destroy" if name == "cudaStreamDestroy" else "create"
                stream = (
                    self.format_runtime_pointer_target(node.args[0])
                    if action == "create"
                    else args[0]
                )
                comment = f"// CUDA stream {action}: {stream}"
                if (
                    name in {
                        "cudaStreamCreateWithFlags",
                        "cudaStreamCreateWithPriority",
                    }
                    and len(args) >= 2
                ):
                    comment += f", flags: {args[1]}"
                if name == "cudaStreamCreateWithPriority" and len(args) >= 3:
                    comment += f", priority: {args[2]}"
                return [comment]
        elif name in {"cudaEventCreate", "cudaEventCreateWithFlags"}:
            if args:
                event = self.format_runtime_pointer_target(node.args[0])
                comment = f"// CUDA event create: {event}"
                if len(args) >= 2:
                    comment += f", flags: {args[1]}"
                return [comment]
        elif name == "cudaEventRecord":
            if args:
                comment = f"// CUDA event record: {args[0]}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name == "cudaEventSynchronize":
            if args:
                return [f"// CUDA event synchronize: {args[0]}"]
        elif name == "cudaEventElapsedTime":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA event elapsed time: {args[1]} -> {args[2]}, "
                    f"output: {output}"
                ]
        elif name == "cudaEventDestroy":
            if args:
                return [f"// CUDA event destroy: {args[0]}"]
        elif name == "cudaEventQuery":
            if args:
                return [f"// CUDA event query: {args[0]}"]
        elif name in {"cuEventCreate", "cuEventCreateWithFlags"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver event create: output: {output}, "
                    f"flags: {args[1]}"
                ]
        elif name in {"cuEventRecord", "cuEventRecordWithFlags"}:
            if len(args) >= 2:
                if name == "cuEventRecordWithFlags" and len(args) >= 3:
                    return [
                        "// CUDA driver event record with flags: "
                        f"event: {args[0]}, stream: {args[1]}, flags: {args[2]}"
                    ]
                return [
                    f"// CUDA driver event record: event: {args[0]}, "
                    f"stream: {args[1]}"
                ]
        elif name in {"cuEventQuery", "cuEventSynchronize", "cuEventDestroy"}:
            if args:
                action = {
                    "cuEventQuery": "query",
                    "cuEventSynchronize": "synchronize",
                    "cuEventDestroy": "destroy",
                }[name]
                return [f"// CUDA driver event {action}: event: {args[0]}"]
        elif name == "cuEventElapsedTime":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver event elapsed time: {args[1]} -> {args[2]}, "
                    f"output: {output}"
                ]
        elif name == "cuProfilerInitialize":
            if len(args) >= 3:
                return [
                    "// CUDA driver profiler initialize: "
                    f"config: {args[0]}, output: {args[1]}, mode: {args[2]}"
                ]
        elif name in {"cuProfilerStart", "cuProfilerStop"}:
            action = "start" if name == "cuProfilerStart" else "stop"
            return [f"// CUDA driver profiler {action}"]
        elif name == "cudaStreamWaitEvent":
            if len(args) >= 2:
                comment = f"// CUDA stream wait event: {args[0]} waits for {args[1]}"
                if len(args) >= 3:
                    comment += f", flags: {args[2]}"
                return [comment]
        elif name in {
            "cuStreamCreate",
            "cuStreamCreateWithFlags",
            "cuStreamCreateWithPriority",
        }:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                action = (
                    "create with priority"
                    if name == "cuStreamCreateWithPriority"
                    else "create"
                )
                comment = (
                    f"// CUDA driver stream {action}: output: {output}, "
                    f"flags: {args[1]}"
                )
                if name == "cuStreamCreateWithPriority" and len(args) >= 3:
                    comment += f", priority: {args[2]}"
                return [comment]
        elif name == "cuStreamDestroy":
            if args:
                return [f"// CUDA driver stream destroy: stream: {args[0]}"]
        elif name in {"cuStreamQuery", "cuStreamSynchronize"}:
            if args:
                action = "query" if name == "cuStreamQuery" else "synchronize"
                return [f"// CUDA driver stream {action}: stream: {args[0]}"]
        elif name == "cuStreamWaitEvent":
            if len(args) >= 2:
                flags = args[2] if len(args) >= 3 else "0"
                return [
                    f"// CUDA driver stream wait event: stream: {args[0]}, "
                    f"event: {args[1]}, flags: {flags}"
                ]
        elif name == "cuLaunchHostFunc":
            if len(args) >= 3:
                return [
                    "// CUDA driver stream launch host function: "
                    f"stream: {args[0]}, callback: {args[1]}, user data: {args[2]}"
                ]
        elif name == "cuStreamAddCallback":
            if len(args) >= 4:
                return [
                    "// CUDA driver stream add callback: "
                    f"stream: {args[0]}, callback: {args[1]}, "
                    f"user data: {args[2]}, flags: {args[3]}"
                ]
        elif name == "cuStreamAttachMemAsync":
            if len(args) >= 3:
                flags = args[3] if len(args) >= 4 else "0"
                return [
                    "// CUDA driver stream attach memory: "
                    f"stream: {args[0]}, pointer: {args[1]}, bytes: {args[2]}, "
                    f"flags: {flags}"
                ]
        elif name == "cuStreamBeginCapture":
            if len(args) >= 2:
                return [
                    "// CUDA driver stream begin capture: "
                    f"stream: {args[0]}, mode: {args[1]}"
                ]
        elif name == "cuStreamEndCapture":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver stream end capture: "
                    f"stream: {args[0]}, graph output: {output}"
                ]
        elif name in {
            "cuStreamWaitValue32",
            "cuStreamWaitValue64",
            "cuStreamWriteValue32",
            "cuStreamWriteValue64",
        }:
            if len(args) >= 4:
                operation = "wait" if "Wait" in name else "write"
                width = "32-bit" if name.endswith("32") else "64-bit"
                return [
                    f"// CUDA driver stream {operation} {width} value: "
                    f"stream: {args[0]}, address: {args[1]}, "
                    f"value: {args[2]}, flags: {args[3]}"
                ]
        elif name == "cuStreamBatchMemOp":
            if len(args) >= 4:
                return [
                    "// CUDA driver stream batch memory operation: "
                    f"stream: {args[0]}, count: {args[1]}, "
                    f"params: {args[2]}, flags: {args[3]}"
                ]
        elif name in {"cuStreamGetAttribute", "cuStreamSetAttribute"}:
            if len(node.args) >= 3:
                action = "get" if name == "cuStreamGetAttribute" else "set"
                target = self.format_runtime_pointer_target(node.args[2])
                target_label = "output" if action == "get" else "value"
                return [
                    f"// CUDA driver stream {action} attribute: "
                    f"stream: {args[0]}, attribute: {args[1]}, "
                    f"{target_label}: {target}"
                ]
        elif name == "cuStreamCopyAttributes":
            if len(args) >= 2:
                return [
                    "// CUDA driver stream copy attributes: "
                    f"destination: {args[0]}, source: {args[1]}"
                ]
        elif name in {"cuStreamGetCtx", "cuStreamGetFlags", "cuStreamGetPriority"}:
            if len(node.args) >= 2:
                query_kind = {
                    "cuStreamGetCtx": "context",
                    "cuStreamGetFlags": "flags",
                    "cuStreamGetPriority": "priority",
                }[name]
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA driver stream {query_kind} query: "
                    f"stream: {args[0]}, output: {output}"
                ]
        elif name == "cuStreamIsCapturing":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver stream capture status query: "
                    f"stream: {args[0]}, output: {output}"
                ]
        elif name == "cuStreamGetCaptureInfo_v2":
            if len(node.args) >= 3:
                fields = [
                    f"stream: {args[0]}",
                    f"status output: {self.format_runtime_pointer_target(node.args[1])}",
                    f"id output: {self.format_runtime_pointer_target(node.args[2])}",
                ]
                if len(node.args) >= 4:
                    fields.append(
                        "graph output: "
                        f"{self.format_runtime_pointer_target(node.args[3])}"
                    )
                if len(node.args) >= 5:
                    fields.append(
                        "dependencies output: "
                        f"{self.format_runtime_pointer_target(node.args[4])}"
                    )
                if len(node.args) >= 6:
                    fields.append(
                        "dependency count output: "
                        f"{self.format_runtime_pointer_target(node.args[5])}"
                    )
                return [
                    "// CUDA driver stream capture info query: " + ", ".join(fields)
                ]
        elif name == "cuStreamUpdateCaptureDependencies":
            if len(args) >= 4:
                return [
                    "// CUDA driver stream update capture dependencies: "
                    f"stream: {args[0]}, dependencies: {args[1]}, "
                    f"dependency count: {args[2]}, flags: {args[3]}"
                ]
        elif name == "cudaStreamBeginCapture":
            if len(args) >= 2:
                return [
                    f"// CUDA stream begin capture: stream: {args[0]}, mode: {args[1]}"
                ]
        elif name == "cudaStreamEndCapture":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA stream end capture: stream: {args[0]}, "
                    f"graph output: {output}"
                ]
        elif name == "cudaGetLastError":
            return ["// CUDA get last error"]
        elif name == "cudaPeekAtLastError":
            return ["// CUDA peek at last error"]
        return None

    def describe_cuda_external_resource_call(self, node, name, args):
        """Comments for a runtime texture or surface object descriptor query or runtime external memory and semaphore call; None for any other name."""
        if name in {
            "cudaGetTextureObjectResourceDesc",
            "cudaGetTextureObjectTextureDesc",
            "cudaGetTextureObjectResourceViewDesc",
        }:
            if len(node.args) >= 2:
                descriptor_kind = {
                    "cudaGetTextureObjectResourceDesc": "resource",
                    "cudaGetTextureObjectTextureDesc": "texture",
                    "cudaGetTextureObjectResourceViewDesc": "resource view",
                }[name]
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA texture object {descriptor_kind} descriptor query: "
                    f"{args[1]}, output: {output}"
                ]
        elif name == "cudaGetSurfaceObjectResourceDesc":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA surface object resource descriptor query: "
                    f"{args[1]}, output: {output}"
                ]
        elif name == "cudaImportExternalMemory":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA external memory import: output: {output}, "
                    f"handle: {args[1]}"
                ]
        elif name == "cudaExternalMemoryGetMappedBuffer":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA external memory mapped buffer: {args[1]}, "
                    f"desc: {args[2]}, output: {output}"
                ]
        elif name == "cudaExternalMemoryGetMappedMipmappedArray":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA external memory mapped mipmapped array: "
                    f"{args[1]}, desc: {args[2]}, output: {output}"
                ]
        elif name == "cudaDestroyExternalMemory":
            if args:
                return [f"// CUDA external memory destroy: {args[0]}"]
        elif name == "cudaImportExternalSemaphore":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA external semaphore import: output: {output}, "
                    f"handle: {args[1]}"
                ]
        elif name in {
            "cudaSignalExternalSemaphoresAsync",
            "cudaWaitExternalSemaphoresAsync",
        }:
            if len(args) >= 3:
                operation = (
                    "signal" if name == "cudaSignalExternalSemaphoresAsync" else "wait"
                )
                comment = (
                    f"// CUDA external semaphore {operation}: "
                    f"semaphores: {args[0]}, params: {args[1]}, count: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name == "cudaDestroyExternalSemaphore":
            if args:
                return [f"// CUDA external semaphore destroy: {args[0]}"]
        return None
