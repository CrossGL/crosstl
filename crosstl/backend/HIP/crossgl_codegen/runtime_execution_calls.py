"""HIP host launch, stream, event and device calls in the HIP to CrossGL converter, described as comments."""


class HipRuntimeExecutionCallMixin:
    """Runtime execution call members of ``HipToCrossGLConverter``."""

    def describe_hip_launch_call(self, node, name, args):
        """Comments for an IPC event handle, stream memory operation, stream callback, launch configuration or extended kernel launch call; None for any other name."""
        if name == "hipIpcGetEventHandle":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP IPC get event handle: output: {output}, event: {args[1]}"
                ]
        elif name == "hipIpcOpenEventHandle":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP IPC open event handle: output: {output}, "
                    f"handle: {args[1]}"
                ]
        elif name in {
            "hipStreamWaitValue32",
            "hipStreamWaitValue64",
            "hipStreamWriteValue32",
            "hipStreamWriteValue64",
        }:
            if len(args) >= 4:
                operation = "wait" if "Wait" in name else "write"
                width = "32" if name.endswith("32") else "64"
                return [
                    f"// HIP stream {operation} value{width}: stream: {args[0]}, "
                    f"address: {args[1]}, value: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipStreamBatchMemOp":
            if len(args) >= 4:
                return [
                    f"// HIP stream batch memory op: stream: {args[0]}, "
                    f"count: {args[1]}, params: {args[2]}, flags: {args[3]}"
                ]
        elif name in {"hipStreamSynchronize"}:
            if args:
                return [f"// HIP synchronize: {args[0]}"]
        elif name == "hipStreamAddCallback":
            if len(args) >= 4:
                return [
                    f"// HIP stream add callback: stream: {args[0]}, "
                    f"callback: {args[1]}, user data: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipLaunchHostFunc":
            if len(args) >= 3:
                return [
                    f"// HIP launch host function: stream: {args[0]}, "
                    f"function: {args[1]}, user data: {args[2]}"
                ]
        elif name in {"hipConfigureCall", "__hipPushCallConfiguration"}:
            if len(args) >= 4:
                action = (
                    "push call configuration"
                    if name.startswith("__")
                    else "configure call"
                )
                return [
                    f"// HIP {action}: grid: {args[0]}, block: {args[1]}, "
                    f"shared memory: {args[2]}, stream: {args[3]}"
                ]
        elif name == "__hipPopCallConfiguration":
            if len(args) >= 4:
                grid_output = self.format_runtime_raw_output_target(node.args[0])
                block_output = self.format_runtime_raw_output_target(node.args[1])
                shared_output = self.format_runtime_raw_output_target(node.args[2])
                stream_output = self.format_runtime_raw_output_target(node.args[3])
                return [
                    f"// HIP pop call configuration: grid output: {grid_output}, "
                    f"block output: {block_output}, shared memory output: "
                    f"{shared_output}, stream output: {stream_output}"
                ]
        elif name == "hipSetupArgument":
            if len(args) >= 3:
                return [
                    f"// HIP setup kernel argument: value: {args[0]}, "
                    f"bytes: {args[1]}, offset: {args[2]}"
                ]
        elif name == "hipLaunchByPtr":
            if args:
                return [f"// HIP launch by pointer: function: {args[0]}"]
        elif name == "hipLaunchKernelExC":
            if len(args) >= 3:
                return [
                    f"// HIP launch kernel ex: config: {args[0]}, "
                    f"function: {args[1]}, args: {args[2]}"
                ]
        elif name == "hipDrvLaunchKernelEx":
            if len(args) >= 4:
                return [
                    f"// HIP driver launch kernel ex: config: {args[0]}, "
                    f"function: {args[1]}, params: {args[2]}, extra: {args[3]}"
                ]
        elif name == "hipExtLaunchKernel":
            if len(args) >= 9:
                function = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// HIP extended kernel launch: function: {function}, "
                    f"grid: {args[1]}, block: {args[2]}, args: {args[3]}, "
                    f"shared memory: {args[4]}, stream: {args[5]}, "
                    f"start event: {args[6]}, stop event: {args[7]}, "
                    f"flags: {args[8]}"
                ]
        elif name == "hipExtLaunchKernelGGL":
            if len(args) >= 8:
                call_args = ", ".join(args[8:]) if len(args) > 8 else "<none>"
                return [
                    f"// HIP extended kernel launch GGL: function: {args[0]}, "
                    f"grid: {args[1]}, block: {args[2]}, "
                    f"shared memory: {args[3]}, stream: {args[4]}, "
                    f"start event: {args[5]}, stop event: {args[6]}, "
                    f"flags: {args[7]}, args: {call_args}"
                ]
        elif name == "hipExtLaunchMultiKernelMultiDevice":
            if len(args) >= 3:
                return [
                    f"// HIP extended multi-kernel multi-device launch: "
                    f"params: {args[0]}, devices: {args[1]}, flags: {args[2]}"
                ]
        return None

    def describe_hip_stream_call(self, node, name, args):
        """Comments for a stream priority, stream capture, synchronization, stream, event or error query call; None for any other name."""
        if name == "hipDeviceGetStreamPriorityRange":
            if len(args) >= 2:
                least_output = self.format_runtime_raw_output_target(node.args[0])
                greatest_output = self.format_runtime_raw_output_target(node.args[1])
                least_output_name = self.get_runtime_pointer_target_name(node.args[0])
                greatest_output_name = self.get_runtime_pointer_target_name(
                    node.args[1]
                )
                if least_output_name is not None:
                    self.register_device_query_source(
                        least_output_name, "streamPriorityRange.least"
                    )
                if greatest_output_name is not None:
                    self.register_device_query_source(
                        greatest_output_name, "streamPriorityRange.greatest"
                    )
                return [
                    f"// HIP get stream priority range: "
                    f"least output: {least_output}, greatest output: {greatest_output}"
                ]
        elif name == "hipStreamGetFlags":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[1])
                output_name = self.get_runtime_pointer_target_name(node.args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"stream.flags({args[0]})"
                    )
                return [f"// HIP get stream flags: stream: {args[0]}, output: {output}"]
        elif name == "hipStreamGetPriority":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[1])
                output_name = self.get_runtime_pointer_target_name(node.args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"stream.priority({args[0]})"
                    )
                return [
                    f"// HIP get stream priority: stream: {args[0]}, output: {output}"
                ]
        elif name == "hipStreamBeginCapture":
            if len(args) >= 2:
                return [
                    f"// HIP stream begin capture: stream: {args[0]}, mode: {args[1]}"
                ]
        elif name == "hipStreamEndCapture":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[1])
                return [
                    f"// HIP stream end capture: stream: {args[0]}, graph output: {output}"
                ]
        elif name == "hipStreamIsCapturing":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[1])
                output_name = self.get_runtime_pointer_target_name(node.args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"stream.captureStatus({args[0]})"
                    )
                return [
                    f"// HIP stream is capturing: stream: {args[0]}, output: {output}"
                ]
        elif name in {"hipStreamGetCaptureInfo", "hipStreamGetCaptureInfo_v2"}:
            if len(args) >= 3:
                status_output = self.format_runtime_raw_output_target(node.args[1])
                id_output = self.format_runtime_raw_output_target(node.args[2])
                status_output_name = self.get_runtime_pointer_target_name(node.args[1])
                id_output_name = self.get_runtime_pointer_target_name(node.args[2])
                if status_output_name is not None:
                    self.register_device_query_source(
                        status_output_name, f"stream.captureStatus({args[0]})"
                    )
                if id_output_name is not None:
                    self.register_device_query_source(
                        id_output_name, f"stream.captureId({args[0]})"
                    )
                comment = (
                    f"// HIP stream capture info: stream: {args[0]}, "
                    f"status output: {status_output}, id output: {id_output}"
                )
                if len(args) >= 6:
                    graph_output = self.format_runtime_raw_output_target(node.args[3])
                    dependencies_output = self.format_runtime_raw_output_target(
                        node.args[4]
                    )
                    count_output = self.format_runtime_raw_output_target(node.args[5])
                    count_output_name = self.get_runtime_pointer_target_name(
                        node.args[5]
                    )
                    if count_output_name is not None:
                        self.register_device_query_source(
                            count_output_name,
                            f"stream.captureDependencyCount({args[0]})",
                        )
                    comment += (
                        f", graph output: {graph_output}, "
                        f"dependencies output: {dependencies_output}, "
                        f"dependency count output: {count_output}"
                    )
                return [comment]
        elif name == "hipStreamUpdateCaptureDependencies":
            if len(args) >= 4:
                return [
                    f"// HIP stream update capture dependencies: stream: {args[0]}, "
                    f"dependencies: {args[1]}, count: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipDeviceSynchronize":
            return ["// HIP device synchronize"]
        elif name in {
            "hipStreamCreate",
            "hipStreamCreateWithFlags",
            "hipStreamCreateWithPriority",
            "hipStreamDestroy",
        }:
            if args:
                action = "destroy" if name == "hipStreamDestroy" else "create"
                stream = (
                    self.format_runtime_pointer_target(node.args[0])
                    if action == "create"
                    else args[0]
                )
                comment = f"// HIP stream {action}: {stream}"
                if (
                    name in {
                        "hipStreamCreateWithFlags",
                        "hipStreamCreateWithPriority",
                    }
                    and len(args) >= 2
                ):
                    comment += f", flags: {args[1]}"
                if name == "hipStreamCreateWithPriority" and len(args) >= 3:
                    comment += f", priority: {args[2]}"
                return [comment]
        elif name in {"hipEventCreate", "hipEventCreateWithFlags"}:
            if args:
                event = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP event create: {event}"
                if len(args) >= 2:
                    comment += f", flags: {args[1]}"
                return [comment]
        elif name == "hipEventRecord":
            if args:
                comment = f"// HIP event record: {args[0]}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name == "hipEventSynchronize":
            if args:
                return [f"// HIP event synchronize: {args[0]}"]
        elif name == "hipEventElapsedTime":
            if len(node.args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"event.elapsedTime({args[1]}, {args[2]})",
                    )
                return [
                    f"// HIP event elapsed time: {args[1]} -> {args[2]}, "
                    f"output: {output}"
                ]
        elif name == "hipEventDestroy":
            if args:
                return [f"// HIP event destroy: {args[0]}"]
        elif name == "hipEventQuery":
            if args:
                return [f"// HIP event query: {args[0]}"]
        elif name == "hipStreamWaitEvent":
            if len(args) >= 2:
                comment = f"// HIP stream wait event: {args[0]} waits for {args[1]}"
                if len(args) >= 3:
                    comment += f", flags: {args[2]}"
                return [comment]
        elif name == "hipGetLastError":
            return ["// HIP get last error"]
        elif name == "hipPeekAtLastError":
            return ["// HIP peek at last error"]
        return None

    def describe_hip_device_call(self, node, name, args):
        """Comments for a device selection, property, configuration, limit or flag call, a procedure address lookup, or a memory or pointer information query; None for any other name."""
        if name == "hipGetDevice":
            if args:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "currentDevice")
                return [f"// HIP get current device: output: {output}"]
        elif name == "hipGetDeviceCount":
            if args:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "deviceCount")
                return [f"// HIP get device count: output: {output}"]
        elif name == "hipSetDevice":
            if args:
                return [f"// HIP set device: {args[0]}"]
        elif name == "hipSetValidDevices":
            if len(args) >= 2:
                return [
                    f"// HIP set valid devices: devices: {args[0]}, count: {args[1]}"
                ]
        elif name == "hipGetDeviceProperties":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_property_source(output_name, args[1])
                return [f"// HIP get device properties: {output}, device: {args[1]}"]
        elif name == "hipDeviceGetAttribute":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_attribute_source(output_name, args[1], args[2])
                return [
                    f"// HIP get device attribute: output: {output}, "
                    f"attribute: {args[1]}, device: {args[2]}"
                ]
        elif name == "hipDeviceGetName":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP get device name: output: {output}, "
                    f"length: {args[1]}, device: {args[2]}"
                ]
        elif name == "hipDeviceGetUuid":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [f"// HIP get device UUID: output: {output}, device: {args[1]}"]
        elif name == "hipDeviceTotalMem":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "totalMem", args[1])
                return [
                    f"// HIP get device total memory: output: {output}, "
                    f"device: {args[1]}"
                ]
        elif name == "hipDeviceComputeCapability":
            if len(args) >= 3:
                major_output = self.format_runtime_raw_output_target(node.args[0])
                minor_output = self.format_runtime_raw_output_target(node.args[1])
                major_output_name = self.get_runtime_pointer_target_name(node.args[0])
                minor_output_name = self.get_runtime_pointer_target_name(node.args[1])
                if major_output_name is not None:
                    self.register_device_query_source(
                        major_output_name, "computeCapability.major", args[2]
                    )
                if minor_output_name is not None:
                    self.register_device_query_source(
                        minor_output_name, "computeCapability.minor", args[2]
                    )
                return [
                    f"// HIP get device compute capability: "
                    f"major output: {major_output}, minor output: {minor_output}, "
                    f"device: {args[2]}"
                ]
        elif name == "hipChooseDevice":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "selectedDevice")
                return [
                    f"// HIP choose device: output: {output}, properties: {args[1]}"
                ]
        elif name == "hipExtGetLinkTypeAndHopCount":
            if len(args) >= 4:
                link_type_output = self.format_runtime_raw_output_target(node.args[2])
                hop_count_output = self.format_runtime_raw_output_target(node.args[3])
                link_type_output_name = self.get_runtime_pointer_target_name(
                    node.args[2]
                )
                hop_count_output_name = self.get_runtime_pointer_target_name(
                    node.args[3]
                )
                if link_type_output_name is not None:
                    self.register_device_query_source(
                        link_type_output_name, f"linkType({args[0]}, {args[1]})"
                    )
                if hop_count_output_name is not None:
                    self.register_device_query_source(
                        hop_count_output_name, f"hopCount({args[0]}, {args[1]})"
                    )
                return [
                    f"// HIP get link type and hop count: device 1: {args[0]}, "
                    f"device 2: {args[1]}, link type output: {link_type_output}, "
                    f"hop count output: {hop_count_output}"
                ]
        elif name == "hipDeviceGetPCIBusId":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP get device PCI bus id: output: {output}, "
                    f"length: {args[1]}, device: {args[2]}"
                ]
        elif name == "hipDeviceGetByPCIBusId":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"deviceByPCIBusId({args[1]})"
                    )
                return [
                    f"// HIP get device by PCI bus id: output: {output}, "
                    f"bus id: {args[1]}"
                ]
        elif name == "hipDeviceGetCacheConfig":
            if args:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "cacheConfig")
                return [f"// HIP get device cache config: output: {output}"]
        elif name == "hipDeviceSetCacheConfig":
            if args:
                return [f"// HIP set device cache config: {args[0]}"]
        elif name == "hipDeviceGetSharedMemConfig":
            if args:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "sharedMemConfig")
                return [f"// HIP get device shared memory config: output: {output}"]
        elif name == "hipDeviceSetSharedMemConfig":
            if args:
                return [f"// HIP set device shared memory config: {args[0]}"]
        elif name == "hipDeviceGetLimit":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, f"limit.{args[1]}")
                return [f"// HIP get device limit: output: {output}, limit: {args[1]}"]
        elif name == "hipDeviceSetLimit":
            if len(args) >= 2:
                return [f"// HIP set device limit: limit: {args[0]}, value: {args[1]}"]
        elif name == "hipDeviceReset":
            return ["// HIP device reset"]
        elif name == "hipGetDeviceFlags":
            if args:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "deviceFlags")
                return [f"// HIP get device flags: output: {output}"]
        elif name == "hipSetDeviceFlags":
            if args:
                return [f"// HIP set device flags: {args[0]}"]
        elif name == "hipGetProcAddress":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[1])
                status_output = self.format_runtime_raw_output_target(node.args[4])
                return [
                    f"// HIP get proc address: symbol: {args[0]}, output: {output}, "
                    f"version: {args[2]}, flags: {args[3]}, "
                    f"status output: {status_output}"
                ]
        elif name == "hipMemGetInfo":
            if len(args) >= 2:
                free_output = self.format_runtime_raw_output_target(node.args[0])
                total_output = self.format_runtime_raw_output_target(node.args[1])
                free_output_name = self.get_runtime_pointer_target_name(node.args[0])
                total_output_name = self.get_runtime_pointer_target_name(node.args[1])
                if free_output_name is not None:
                    self.register_device_query_source(
                        free_output_name,
                        "memory.info.free",
                    )
                if total_output_name is not None:
                    self.register_device_query_source(
                        total_output_name,
                        "memory.info.total",
                    )
                return [
                    f"// HIP memory info: free output: {free_output}, "
                    f"total output: {total_output}"
                ]
        elif name == "hipPointerGetAttributes":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP pointer attributes: output: {output}, pointer: {args[1]}"
                ]
        elif name == "hipDrvPointerGetAttributes":
            if len(args) >= 4:
                return [
                    f"// HIP driver pointer attributes: count: {args[0]}, "
                    f"attributes: {args[1]}, data: {args[2]}, pointer: {args[3]}"
                ]
        elif name == "hipPointerGetAttribute":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"pointer.attribute({args[1]}, {args[2]})",
                    )
                return [
                    f"// HIP pointer attribute: output: {output}, "
                    f"attribute: {args[1]}, pointer: {args[2]}"
                ]
        elif name == "hipPointerSetAttribute":
            if len(args) >= 3:
                value = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// HIP pointer set attribute: value: {value}, "
                    f"attribute: {args[1]}, pointer: {args[2]}"
                ]
        elif name == "hipMemPtrGetInfo":
            if len(args) >= 2:
                size_output = self.format_runtime_raw_output_target(node.args[1])
                size_output_name = self.get_runtime_pointer_target_name(node.args[1])
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"memoryPointer.size({args[0]})",
                    )
                return [
                    f"// HIP memory pointer info: pointer: {args[0]}, "
                    f"size output: {size_output}"
                ]
        return None
