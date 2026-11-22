"""HIP runtime status, interop, RTC, driver and graph calls in the HIP to CrossGL converter."""

from ..HipAst import (
    AssignmentNode,
    CastNode,
    FunctionCallNode,
    UnaryOpNode,
    VariableNode,
)


class HipRuntimeCallMixin:
    """Runtime call members of ``HipToCrossGLConverter``."""

    def format_hip_runtime_status_expression(self, value):
        if not isinstance(value, FunctionCallNode):
            return None
        if self.is_user_defined_function(value.name):
            return None

        wrapped_runtime = self.format_hip_runtime_wrapper_expression(value)
        if wrapped_runtime is not None:
            return wrapped_runtime

        comments = self.format_hip_runtime_call(value)
        if comments is None:
            return None

        if value.name == "hipGetSurfaceObjectResourceDesc":
            return comments, "hipErrorNotSupported"

        success_value = (
            "HIPRTC_SUCCESS" if value.name.startswith("hiprtc") else "hipSuccess"
        )
        return comments, success_value

    def format_hip_runtime_wrapper_expression(self, value):
        if not isinstance(value, FunctionCallNode):
            return None
        if self.is_user_defined_function(value.name):
            return None
        if value.name not in self.HIP_RUNTIME_ERROR_WRAPPER_NAMES:
            return None
        if len(getattr(value, "args", []) or []) != 1:
            return None
        return self.format_hip_runtime_status_expression(value.args[0])

    def format_hip_interop_runtime_call(self, name, raw_args, args):
        if name == "hipProfilerStart":
            return ["// HIP profiler start"]
        if name == "hipProfilerStop":
            return ["// HIP profiler stop"]
        if name == "hipImportExternalMemory":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP import external memory: output: {output}, "
                    f"descriptor: {args[1]}"
                ]
        if name == "hipDestroyExternalMemory":
            if args:
                return [f"// HIP destroy external memory: {args[0]}"]
        if name == "hipExternalMemoryGetMappedBuffer":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP external memory mapped buffer: output: {output}, "
                    f"memory: {args[1]}, descriptor: {args[2]}"
                ]
        if name == "hipExternalMemoryGetMappedMipmappedArray":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP external memory mapped mipmapped array: output: {output}, "
                    f"memory: {args[1]}, descriptor: {args[2]}"
                ]
        if name == "hipFreeMipmappedArray":
            if args:
                return [f"// HIP free mipmapped array: {args[0]}"]
        if name == "hipImportExternalSemaphore":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP import external semaphore: output: {output}, "
                    f"descriptor: {args[1]}"
                ]
        if name == "hipDestroyExternalSemaphore":
            if args:
                return [f"// HIP destroy external semaphore: {args[0]}"]
        if name in {
            "hipSignalExternalSemaphoresAsync",
            "hipWaitExternalSemaphoresAsync",
        }:
            if len(args) >= 4:
                action = (
                    "signal" if name == "hipSignalExternalSemaphoresAsync" else "wait"
                )
                return [
                    f"// HIP {action} external semaphores: semaphores: {args[0]}, "
                    f"params: {args[1]}, count: {args[2]}, stream: {args[3]}"
                ]
        if name == "hipGraphicsGLRegisterBuffer":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP OpenGL register buffer: output: {output}, "
                    f"buffer: {args[1]}, flags: {args[2]}"
                ]
        if name == "hipGraphicsGLRegisterImage":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP OpenGL register image: output: {output}, "
                    f"image: {args[1]}, target: {args[2]}, flags: {args[3]}"
                ]
        if name == "hipGraphicsUnregisterResource":
            if args:
                return [f"// HIP graphics unregister resource: {args[0]}"]
        if name in {"hipGraphicsMapResources", "hipGraphicsUnmapResources"}:
            if len(args) >= 3:
                action = "map" if name == "hipGraphicsMapResources" else "unmap"
                return [
                    f"// HIP graphics {action} resources: count: {args[0]}, "
                    f"resources: {args[1]}, stream: {args[2]}"
                ]
        if name == "hipGraphicsResourceGetMappedPointer":
            if len(args) >= 3:
                pointer_output = self.format_runtime_raw_output_target(raw_args[0])
                size_output = self.format_runtime_raw_output_target(raw_args[1])
                return [
                    f"// HIP graphics mapped pointer: pointer output: {pointer_output}, "
                    f"size output: {size_output}, resource: {args[2]}"
                ]
        if name == "hipGraphicsSubResourceGetMappedArray":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graphics mapped subresource array: output: {output}, "
                    f"resource: {args[1]}, array index: {args[2]}, mip level: {args[3]}"
                ]
        return None

    def format_hip_rtc_runtime_call(self, name, raw_args, args):
        if name == "hiprtcVersion":
            if len(args) >= 2:
                major_output = self.format_runtime_raw_output_target(raw_args[0])
                minor_output = self.format_runtime_raw_output_target(raw_args[1])
                major_output_name = self.get_runtime_pointer_target_name(raw_args[0])
                minor_output_name = self.get_runtime_pointer_target_name(raw_args[1])
                if major_output_name is not None:
                    self.register_device_query_source(
                        major_output_name, "rtc.version.major"
                    )
                if minor_output_name is not None:
                    self.register_device_query_source(
                        minor_output_name, "rtc.version.minor"
                    )
                return [
                    f"// HIPRTC version: major output: {major_output}, "
                    f"minor output: {minor_output}"
                ]
        if name == "hiprtcCreateProgram":
            if len(args) >= 6:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIPRTC create program: output: {output}, source: {args[1]}, "
                    f"name: {args[2]}, headers: {args[3]}, header sources: {args[4]}, "
                    f"include names: {args[5]}"
                ]
        if name == "hiprtcDestroyProgram":
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [f"// HIPRTC destroy program: output: {output}"]
        if name == "hiprtcCompileProgram":
            if len(args) >= 3:
                return [
                    f"// HIPRTC compile program: program: {args[0]}, "
                    f"options: {args[1]}, option values: {args[2]}"
                ]
        if name in {
            "hiprtcGetCodeSize",
            "hiprtcGetBitcodeSize",
            "hiprtcGetProgramLogSize",
        }:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                output_name = self.get_runtime_pointer_target_name(raw_args[1])
                artifact = {
                    "hiprtcGetCodeSize": "code size",
                    "hiprtcGetBitcodeSize": "bitcode size",
                    "hiprtcGetProgramLogSize": "program log size",
                }[name]
                query = {
                    "hiprtcGetCodeSize": "rtc.program.code.size",
                    "hiprtcGetBitcodeSize": "rtc.program.bitcode.size",
                    "hiprtcGetProgramLogSize": "rtc.program.log.size",
                }[name]
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"{query}({args[0]})"
                    )
                return [
                    f"// HIPRTC get {artifact}: program: {args[0]}, output: {output}"
                ]
        if name in {"hiprtcGetCode", "hiprtcGetBitcode", "hiprtcGetProgramLog"}:
            if len(args) >= 2:
                self.clear_lvalue_metadata_source(raw_args[1])
                artifact = {
                    "hiprtcGetCode": "code",
                    "hiprtcGetBitcode": "bitcode",
                    "hiprtcGetProgramLog": "program log",
                }[name]
                return [
                    f"// HIPRTC get {artifact}: program: {args[0]}, output: {args[1]}"
                ]
        if name == "hiprtcAddNameExpression":
            if len(args) >= 2:
                return [
                    f"// HIPRTC add name expression: program: {args[0]}, "
                    f"expression: {args[1]}"
                ]
        if name == "hiprtcGetLoweredName":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[2])
                return [
                    f"// HIPRTC get lowered name: program: {args[0]}, "
                    f"expression: {args[1]}, output: {output}"
                ]
        if name == "hiprtcLinkCreate":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(raw_args[3])
                return [
                    f"// HIPRTC link create: options: {args[0]}, "
                    f"option keys: {args[1]}, option values: {args[2]}, "
                    f"state output: {output}"
                ]
        if name == "hiprtcLinkAddFile":
            if len(args) >= 6:
                return [
                    f"// HIPRTC link add file: state: {args[0]}, input type: {args[1]}, "
                    f"path: {args[2]}, options: {args[3]}, option keys: {args[4]}, "
                    f"option values: {args[5]}"
                ]
        if name == "hiprtcLinkAddData":
            if len(args) >= 8:
                return [
                    f"// HIPRTC link add data: state: {args[0]}, input type: {args[1]}, "
                    f"image: {args[2]}, bytes: {args[3]}, name: {args[4]}, "
                    f"options: {args[5]}, option keys: {args[6]}, "
                    f"option values: {args[7]}"
                ]
        if name == "hiprtcLinkComplete":
            if len(args) >= 3:
                binary_output = self.format_runtime_raw_output_target(raw_args[1])
                size_output = self.format_runtime_raw_output_target(raw_args[2])
                size_output_name = self.get_runtime_pointer_target_name(raw_args[2])
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"rtc.link.complete.size({args[0]})",
                    )
                return [
                    f"// HIPRTC link complete: state: {args[0]}, "
                    f"binary output: {binary_output}, size output: {size_output}"
                ]
        if name == "hiprtcLinkDestroy":
            if args:
                return [f"// HIPRTC link destroy: state: {args[0]}"]
        return None

    def format_hip_driver_runtime_call(self, name, raw_args, args):
        if name == "hipInit":
            if args:
                return [f"// HIP initialize runtime: flags: {args[0]}"]
        if name in {"hipDriverGetVersion", "hipRuntimeGetVersion"}:
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                version_kind = "driver" if name == "hipDriverGetVersion" else "runtime"
                output_name = self.get_runtime_pointer_target_name(raw_args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"{version_kind}.version"
                    )
                return [f"// HIP get {version_kind} version: output: {output}"]
        if name == "hipDeviceGet":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[0])
                output_name = self.get_runtime_pointer_target_name(raw_args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"deviceHandle({args[1]})"
                    )
                return [
                    f"// HIP get device handle: output: {output}, ordinal: {args[1]}"
                ]
        if name == "hipDeviceCanAccessPeer":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                output_name = self.get_runtime_pointer_target_name(raw_args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"canAccessPeer({args[1]}, {args[2]})"
                    )
                return [
                    f"// HIP device can access peer: output: {output}, "
                    f"device: {args[1]}, peer device: {args[2]}"
                ]
        if name == "hipDeviceGetP2PAttribute":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(raw_args[0])
                output_name = self.get_runtime_pointer_target_name(raw_args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"p2pAttribute.{args[1]}({args[2]}, {args[3]})",
                    )
                return [
                    f"// HIP get P2P attribute: output: {output}, "
                    f"attribute: {args[1]}, source device: {args[2]}, "
                    f"destination device: {args[3]}"
                ]
        if name in {"hipDeviceEnablePeerAccess", "hipDeviceDisablePeerAccess"}:
            if args:
                action = "enable" if name == "hipDeviceEnablePeerAccess" else "disable"
                comment = f"// HIP {action} peer access: peer device: {args[0]}"
                if len(args) >= 2:
                    comment += f", flags: {args[1]}"
                return [comment]
        if name == "hipCtxCreate":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP context create: output: {output}, flags: {args[1]}, "
                    f"device: {args[2]}"
                ]
        if name == "hipCtxDestroy":
            if args:
                return [f"// HIP context destroy: {args[0]}"]
        if name == "hipCtxPopCurrent":
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [f"// HIP context pop current: output: {output}"]
        if name == "hipCtxPushCurrent":
            if args:
                return [f"// HIP context push current: {args[0]}"]
        if name == "hipCtxSetCurrent":
            if args:
                return [f"// HIP context set current: {args[0]}"]
        if name == "hipCtxGetCurrent":
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [f"// HIP context get current: output: {output}"]
        if name == "hipCtxGetDevice":
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                output_name = self.get_runtime_pointer_target_name(raw_args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "context.device")
                return [f"// HIP context get device: output: {output}"]
        if name == "hipCtxGetApiVersion":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                output_name = self.get_runtime_pointer_target_name(raw_args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"context.apiVersion({args[0]})"
                    )
                return [
                    f"// HIP context get API version: context: {args[0]}, "
                    f"output: {output}"
                ]
        if name in {"hipCtxGetCacheConfig", "hipCtxGetSharedMemConfig"}:
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                config_kind = "cache" if "Cache" in name else "shared memory"
                output_name = self.get_runtime_pointer_target_name(raw_args[0])
                if output_name is not None:
                    query_name = (
                        "context.cacheConfig"
                        if "Cache" in name
                        else "context.sharedMemConfig"
                    )
                    self.register_device_query_source(output_name, query_name)
                return [f"// HIP context get {config_kind} config: output: {output}"]
        if name in {"hipCtxSetCacheConfig", "hipCtxSetSharedMemConfig"}:
            if args:
                config_kind = "cache" if "Cache" in name else "shared memory"
                return [f"// HIP context set {config_kind} config: {args[0]}"]
        if name == "hipCtxGetFlags":
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                output_name = self.get_runtime_pointer_target_name(raw_args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "context.flags")
                return [f"// HIP context get flags: output: {output}"]
        if name == "hipCtxSynchronize":
            return ["// HIP context synchronize"]
        if name == "hipDevicePrimaryCtxRetain":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP primary context retain: output: {output}, "
                    f"device: {args[1]}"
                ]
        if name in {"hipDevicePrimaryCtxRelease", "hipDevicePrimaryCtxReset"}:
            if args:
                action = "release" if name == "hipDevicePrimaryCtxRelease" else "reset"
                return [f"// HIP primary context {action}: device: {args[0]}"]
        if name == "hipDevicePrimaryCtxSetFlags":
            if len(args) >= 2:
                return [
                    f"// HIP primary context set flags: device: {args[0]}, "
                    f"flags: {args[1]}"
                ]
        if name == "hipDevicePrimaryCtxGetState":
            if len(args) >= 3:
                flags_output = self.format_runtime_raw_output_target(raw_args[1])
                active_output = self.format_runtime_raw_output_target(raw_args[2])
                flags_output_name = self.get_runtime_pointer_target_name(raw_args[1])
                active_output_name = self.get_runtime_pointer_target_name(raw_args[2])
                if flags_output_name is not None:
                    self.register_device_query_source(
                        flags_output_name, "primaryContext.flags", args[0]
                    )
                if active_output_name is not None:
                    self.register_device_query_source(
                        active_output_name, "primaryContext.active", args[0]
                    )
                return [
                    f"// HIP primary context get state: device: {args[0]}, "
                    f"flags output: {flags_output}, active output: {active_output}"
                ]
        return None

    def format_hip_graph_node_params_comment(self, name, args):
        node_kind = self.get_hip_graph_node_kind(name)
        action = "get" if "GetParams" in name else "set"
        return (
            f"// HIP graph {node_kind} node {action} params: "
            f"node: {args[0]}, params: {args[1]}"
        )

    def get_hip_graph_node_kind(self, name):
        if "Kernel" in name:
            return "kernel"
        if "Memcpy" in name:
            return "memcpy"
        if "Memset" in name:
            return "memset"
        if "Host" in name:
            return "host"
        if "ChildGraph" in name:
            return "child graph"
        return "unknown"

    def format_hip_runtime_status_inline_expression(self, node):
        runtime_status = self.format_hip_runtime_status_expression(node)
        if runtime_status is None:
            return None

        comments, value = runtime_status
        detail = "; ".join(
            self.format_runtime_expression_comment_text(comment) for comment in comments
        )
        return f"(/* {detail} */ {value})"

    def format_runtime_expression_comment_text(self, comment):
        text = str(comment).strip()
        if text.startswith("// "):
            text = text[3:]
        elif text.startswith("//"):
            text = text[2:].lstrip()
        return text.replace("*/", "* /")

    def format_runtime_pointer_target(self, arg):
        if isinstance(arg, CastNode):
            return self.format_runtime_pointer_target(arg.expression)
        if isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.visit_lvalue_expression(arg.operand)
        return self.visit(arg)

    def format_runtime_raw_output_target(self, arg):
        self.clear_lvalue_metadata_source(arg)
        return self.format_runtime_pointer_target(arg)

    def get_runtime_pointer_target_name(self, arg):
        if isinstance(arg, CastNode):
            return self.get_runtime_pointer_target_name(arg.expression)
        if isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.get_runtime_pointer_target_name(arg.operand)
        if isinstance(arg, str):
            return arg
        return None

    def format_statement_fragment(self, stmt):
        if stmt is None:
            return ""
        if isinstance(stmt, list):
            return ", ".join(self.format_statement_fragment(item) for item in stmt)
        if isinstance(stmt, VariableNode):
            raw_type = self.function_local_materialized_variable_type(stmt)
            var_type = self.convert_hip_variable_type_to_crossgl(raw_type, stmt.name)
            self.register_vector1_name(stmt.name, raw_type)
            self.register_variable_type(stmt.name, var_type)
            if hasattr(stmt, "value") and stmt.value:
                value = self.format_variable_initializer_value(stmt.value)
                return f"var {stmt.name}: {var_type} = {value}"
            return f"var {stmt.name}: {var_type}"
        if isinstance(stmt, AssignmentNode):
            left = self.visit_lvalue_expression(stmt.left)
            right = self.visit(stmt.right)
            operator = getattr(stmt, "operator", "=")
            self.clear_lvalue_metadata_source(stmt.left)
            return f"{left} {operator} {right}"

        result = self.visit(stmt)
        return result if isinstance(result, str) else ""
