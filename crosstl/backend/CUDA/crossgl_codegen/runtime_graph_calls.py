"""CUDA host graph calls of the runtime and driver APIs in the CUDA to CrossGL converter, described as comments."""


class CudaImportRuntimeGraphCallMixin:
    """Runtime graph call members of ``CudaToCrossGLConverter``."""

    def describe_cuda_graph_lifecycle_call(self, node, name, args):
        """Comments for a user object, graph memory, graph creation, instantiation, conditional, launch or executable graph destruction call; None for any other name."""
        if name == "cudaUserObjectCreate":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA user object create: output: {output}, "
                    f"payload: {args[1]}, destroy callback: {args[2]}, "
                    f"initial references: {args[3]}, flags: {args[4]}"
                ]
        elif name in {"cudaUserObjectRetain", "cudaUserObjectRelease"}:
            if args:
                operation = "retain" if name == "cudaUserObjectRetain" else "release"
                count = args[1] if len(args) >= 2 else "1"
                return [
                    f"// CUDA user object {operation}: object: {args[0]}, "
                    f"references: {count}"
                ]
        elif name in {"cudaGraphRetainUserObject", "cudaGraphReleaseUserObject"}:
            if len(args) >= 2:
                operation = (
                    "retain" if name == "cudaGraphRetainUserObject" else "release"
                )
                count = args[2] if len(args) >= 3 else "1"
                comment = (
                    f"// CUDA graph {operation} user object: graph: {args[0]}, "
                    f"object: {args[1]}, references: {count}"
                )
                if name == "cudaGraphRetainUserObject":
                    flags = args[3] if len(args) >= 4 else "0"
                    comment += f", flags: {flags}"
                return [comment]
        elif name in {
            "cudaDeviceGetGraphMemAttribute",
            "cudaDeviceSetGraphMemAttribute",
        }:
            if len(node.args) >= 3:
                value = self.format_runtime_pointer_target(node.args[2])
                action = "get" if name == "cudaDeviceGetGraphMemAttribute" else "set"
                value_label = "output" if action == "get" else "value"
                return [
                    f"// CUDA device {action} graph memory attribute: "
                    f"device: {args[0]}, attribute: {args[1]}, "
                    f"{value_label}: {value}"
                ]
        elif name == "cudaDeviceGraphMemTrim":
            if args:
                return [f"// CUDA device graph memory trim: device: {args[0]}"]
        elif name == "cudaGraphCreate":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA graph create: output: {output}, flags: {args[1]}"]
        elif name == "cudaGraphClone":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA graph clone: output: {output}, source: {args[1]}"]
        elif name == "cudaGraphInstantiate":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                error_output = self.format_runtime_pointer_target(node.args[2])
                return [
                    f"// CUDA graph instantiate: output: {output}, graph: {args[1]}, "
                    f"error node output: {error_output}, log buffer: {args[3]}, "
                    f"log bytes: {args[4]}"
                ]
        elif name == "cudaGraphInstantiateWithFlags":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph instantiate with flags: output: {output}, "
                    f"graph: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cudaGraphInstantiateWithParams":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph instantiate with params: output: {output}, "
                    f"graph: {args[1]}, params: {args[2]}"
                ]
        elif name == "cudaGraphConditionalHandleCreate":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                comment = (
                    f"// CUDA graph conditional handle create: output: {output}, "
                    f"graph: {args[1]}"
                )
                if len(args) >= 3:
                    comment += f", default launch value: {args[2]}"
                if len(args) >= 4:
                    comment += f", flags: {args[3]}"
                return [comment]
        elif name == "cudaGraphConditionalHandleCreate_v2":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                comment = (
                    f"// CUDA graph conditional handle create v2: output: {output}, "
                    f"graph: {args[1]}"
                )
                if len(args) >= 3:
                    comment += f", context: {args[2]}"
                if len(args) >= 4:
                    comment += f", default launch value: {args[3]}"
                if len(args) >= 5:
                    comment += f", flags: {args[4]}"
                return [comment]
        elif name == "cudaGraphSetConditional":
            if len(args) >= 2:
                return [
                    f"// CUDA graph set conditional: handle: {args[0]}, "
                    f"value: {args[1]}"
                ]
        elif name in {
            "cudaGraphKernelNodeSetEnabled",
            "cudaGraphKernelNodeSetGridDim",
        }:
            if len(args) >= 2:
                action = (
                    "set enabled"
                    if name == "cudaGraphKernelNodeSetEnabled"
                    else "set grid dim"
                )
                value_label = (
                    "enabled" if name == "cudaGraphKernelNodeSetEnabled" else "grid dim"
                )
                return [
                    f"// CUDA device graph kernel node {action}: "
                    f"node: {args[0]}, {value_label}: {args[1]}"
                ]
        elif name == "cudaGraphKernelNodeSetParam":
            if len(args) >= 3:
                comment = (
                    f"// CUDA device graph kernel node set param: "
                    f"node: {args[0]}, offset: {args[1]}, value: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", bytes: {args[3]}"
                return [comment]
        elif name == "cudaGraphKernelNodeUpdatesApply":
            if len(args) >= 2:
                return [
                    f"// CUDA device graph kernel node updates apply: "
                    f"updates: {args[0]}, count: {args[1]}"
                ]
        elif name == "cudaGraphDebugDotPrint":
            if len(args) >= 3:
                return [
                    f"// CUDA graph debug DOT print: graph: {args[0]}, "
                    f"path: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cudaGraphLaunch":
            if len(args) >= 2:
                device_launch_mode = self.CUDA_DEVICE_GRAPH_LAUNCH_MODES.get(args[1])
                if device_launch_mode is not None:
                    return [
                        f"// CUDA graph device launch: exec: {args[0]}, "
                        f"mode: {device_launch_mode}"
                    ]
                return [f"// CUDA graph launch: exec: {args[0]}, stream: {args[1]}"]
        elif name in {"cudaGraphUpload", "cudaGraphExecUpload"}:
            if len(args) >= 2:
                operation = "exec upload" if name == "cudaGraphExecUpload" else "upload"
                return [
                    f"// CUDA graph {operation}: exec: {args[0]}, stream: {args[1]}"
                ]
        elif name == "cudaGraphExecDestroy":
            if args:
                return [f"// CUDA graph exec destroy: {args[0]}"]
        elif name == "cuGraphCreate":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph create: output: {output}, "
                    f"flags: {args[1]}"
                ]
        elif name == "cuGraphClone":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph clone: output: {output}, "
                    f"source: {args[1]}"
                ]
        elif name == "cuGraphInstantiate":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                error_output = self.format_runtime_pointer_target(node.args[2])
                return [
                    f"// CUDA driver graph instantiate: output: {output}, "
                    f"graph: {args[1]}, error node output: {error_output}, "
                    f"log buffer: {args[3]}, log bytes: {args[4]}"
                ]
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph instantiate: output: {output}, "
                    f"graph: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cuGraphInstantiateWithFlags":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graph instantiate with flags: "
                    f"output: {output}, graph: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cuGraphInstantiateWithParams":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graph instantiate with params: "
                    f"output: {output}, graph: {args[1]}, params: {args[2]}"
                ]
        elif name == "cuGraphUpload":
            if len(args) >= 2:
                return [
                    f"// CUDA driver graph upload: exec: {args[0]}, "
                    f"stream: {args[1]}"
                ]
        elif name == "cuGraphLaunch":
            if len(args) >= 2:
                return [
                    f"// CUDA driver graph launch: exec: {args[0]}, "
                    f"stream: {args[1]}"
                ]
        elif name == "cuGraphExecDestroy":
            if args:
                return [f"// CUDA driver graph exec destroy: {args[0]}"]
        return None

    def describe_cuda_driver_graph_node_call(self, node, name, args):
        """Comments for a driver user object, graph memory, conditional handle, node creation, dependency or graph query call; None for any other name."""
        if name == "cuUserObjectCreate":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver user object create: output: {output}, "
                    f"payload: {args[1]}, destroy callback: {args[2]}, "
                    f"initial references: {args[3]}, flags: {args[4]}"
                ]
        elif name in {"cuUserObjectRetain", "cuUserObjectRelease"}:
            if args:
                operation = "retain" if name == "cuUserObjectRetain" else "release"
                count = args[1] if len(args) >= 2 else "1"
                return [
                    f"// CUDA driver user object {operation}: object: {args[0]}, "
                    f"references: {count}"
                ]
        elif name in {"cuGraphRetainUserObject", "cuGraphReleaseUserObject"}:
            if len(args) >= 2:
                operation = "retain" if name == "cuGraphRetainUserObject" else "release"
                count = args[2] if len(args) >= 3 else "1"
                comment = (
                    f"// CUDA driver graph {operation} user object: "
                    f"graph: {args[0]}, object: {args[1]}, references: {count}"
                )
                if name == "cuGraphRetainUserObject":
                    flags = args[3] if len(args) >= 4 else "0"
                    comment += f", flags: {flags}"
                return [comment]
        elif name in {
            "cuDeviceGetGraphMemAttribute",
            "cuDeviceSetGraphMemAttribute",
        }:
            if len(node.args) >= 3:
                value = self.format_runtime_pointer_target(node.args[2])
                action = "get" if name == "cuDeviceGetGraphMemAttribute" else "set"
                value_label = "output" if action == "get" else "value"
                return [
                    f"// CUDA driver device {action} graph memory attribute: "
                    f"device: {args[0]}, attribute: {args[1]}, "
                    f"{value_label}: {value}"
                ]
        elif name == "cuDeviceGraphMemTrim":
            if args:
                return [f"// CUDA driver device graph memory trim: device: {args[0]}"]
        elif name == "cuGraphConditionalHandleCreate":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graph conditional handle create: "
                    f"output: {output}, graph: {args[1]}, context: {args[2]}, "
                    f"default launch value: {args[3]}, flags: {args[4]}"
                ]
        elif name == "cuGraphAddNode":
            if len(node.args) >= 6:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph add generic node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"edge data: {args[3]}, dependency count: {args[4]}, "
                    f"params: {args[5]}"
                ]
        elif name in {
            "cuGraphAddKernelNode",
            "cuGraphAddMemcpyNode",
            "cuGraphAddMemsetNode",
            "cuGraphAddHostNode",
        }:
            if len(node.args) >= 5:
                node_kind = {
                    "cuGraphAddKernelNode": "kernel",
                    "cuGraphAddMemcpyNode": "memcpy",
                    "cuGraphAddMemsetNode": "memset",
                    "cuGraphAddHostNode": "host",
                }[name]
                output = self.format_runtime_pointer_target(node.args[0])
                comment = (
                    f"// CUDA driver graph add {node_kind} node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"params: {args[4]}"
                )
                if name in {"cuGraphAddMemcpyNode", "cuGraphAddMemsetNode"}:
                    if len(args) >= 6:
                        comment += f", context: {args[5]}"
                return [comment]
        elif name == "cuGraphAddChildGraphNode":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graph add child graph node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"child graph: {args[4]}"
                ]
        elif name == "cuGraphAddEmptyNode":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph add empty node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}"
                ]
        elif name in {"cuGraphAddEventRecordNode", "cuGraphAddEventWaitNode"}:
            if len(node.args) >= 5:
                node_kind = "record" if "Record" in name else "wait"
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph add event {node_kind} node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"event: {args[4]}"
                ]
        elif name in {
            "cuGraphAddExternalSemaphoresSignalNode",
            "cuGraphAddExternalSemaphoresWaitNode",
        }:
            if len(node.args) >= 5:
                node_kind = "signal" if "Signal" in name else "wait"
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph add external semaphore {node_kind} "
                    f"node: output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"params: {args[4]}"
                ]
        elif name == "cuGraphAddMemAllocNode":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graph add memory alloc node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"params: {args[4]}"
                ]
        elif name == "cuGraphAddMemFreeNode":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph add memory free node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}, pointer: {args[4]}"
                ]
        elif name in {"cuGraphAddDependencies", "cuGraphRemoveDependencies"}:
            if len(args) >= 5:
                operation = "add" if name == "cuGraphAddDependencies" else "remove"
                return [
                    f"// CUDA driver graph {operation} dependencies: "
                    f"graph: {args[0]}, from: {args[1]}, to: {args[2]}, "
                    f"edge data: {args[3]}, count: {args[4]}"
                ]
            if len(args) >= 4:
                operation = "add" if name == "cuGraphAddDependencies" else "remove"
                return [
                    f"// CUDA driver graph {operation} dependencies: "
                    f"graph: {args[0]}, from: {args[1]}, to: {args[2]}, "
                    f"count: {args[3]}"
                ]
        elif name == "cuGraphNodeFindInClone":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph node find in clone: output: {output}, "
                    f"original node: {args[1]}, clone graph: {args[2]}"
                ]
        elif name == "cuGraphNodeGetType":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA driver graph node get type: node: {args[0]}, "
                    f"output: {output}"
                ]
        elif name == "cuGraphDebugDotPrint":
            if len(args) >= 3:
                return [
                    f"// CUDA driver graph debug DOT print: graph: {args[0]}, "
                    f"path: {args[1]}, flags: {args[2]}"
                ]
        elif name in {"cuGraphExecGetFlags", "cuGraphExecGetId"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                if name == "cuGraphExecGetFlags":
                    return [
                        f"// CUDA driver graph exec get flags: exec: {args[0]}, "
                        f"output: {output}"
                    ]
                return [
                    f"// CUDA driver graph exec get id: exec: {args[0]}, "
                    f"output: {output}"
                ]
        elif name == "cuGraphGetId":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA driver graph get id: graph: {args[0]}, "
                    f"output: {output}"
                ]
        elif name == "cuGraphGetEdges":
            if len(node.args) >= 5:
                from_output = self.format_runtime_pointer_target(node.args[1])
                to_output = self.format_runtime_pointer_target(node.args[2])
                edge_data = self.format_runtime_pointer_target(node.args[3])
                count_output = self.format_runtime_pointer_target(node.args[4])
                return [
                    f"// CUDA driver graph get edges: graph: {args[0]}, "
                    f"from output: {from_output}, to output: {to_output}, "
                    f"edge data: {edge_data}, count output: {count_output}"
                ]
        elif name in {"cuGraphGetNodes", "cuGraphGetRootNodes"}:
            if len(node.args) >= 3:
                node_output = self.format_runtime_pointer_target(node.args[1])
                count_output = self.format_runtime_pointer_target(node.args[2])
                node_kind = "root nodes" if name == "cuGraphGetRootNodes" else "nodes"
                return [
                    f"// CUDA driver graph get {node_kind}: graph: {args[0]}, "
                    f"nodes output: {node_output}, count output: {count_output}"
                ]
        return None

    def describe_cuda_driver_graph_parameter_call(self, node, name, args):
        """Comments for a driver graph node query, node parameter or executable graph update call; None for any other name."""
        if name == "cuGraphNodeGetContainingGraph":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver graph node get containing graph: "
                    f"node: {args[0]}, output: {output}"
                ]
        elif name in {
            "cuGraphNodeGetDependencies",
            "cuGraphNodeGetDependentNodes",
        }:
            if len(node.args) >= 4:
                nodes_output = self.format_runtime_pointer_target(node.args[1])
                edge_data = self.format_runtime_pointer_target(node.args[2])
                count_output = self.format_runtime_pointer_target(node.args[3])
                if name == "cuGraphNodeGetDependencies":
                    return [
                        "// CUDA driver graph node get dependencies: "
                        f"node: {args[0]}, dependencies output: {nodes_output}, "
                        f"edge data: {edge_data}, count output: {count_output}"
                    ]
                return [
                    "// CUDA driver graph node get dependent nodes: "
                    f"node: {args[0]}, dependent nodes output: {nodes_output}, "
                    f"edge data: {edge_data}, count output: {count_output}"
                ]
        elif name in {"cuGraphNodeGetEnabled", "cuGraphNodeSetEnabled"}:
            if len(node.args) >= 3:
                if name == "cuGraphNodeGetEnabled":
                    output = self.format_runtime_pointer_target(node.args[2])
                    return [
                        f"// CUDA driver graph node get enabled: exec: {args[0]}, "
                        f"node: {args[1]}, output: {output}"
                    ]
                return [
                    f"// CUDA driver graph node set enabled: exec: {args[0]}, "
                    f"node: {args[1]}, enabled: {args[2]}"
                ]
        elif name in {"cuGraphNodeGetLocalId", "cuGraphNodeGetToolsId"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                id_kind = "local id" if name == "cuGraphNodeGetLocalId" else "tools id"
                return [
                    f"// CUDA driver graph node get {id_kind}: node: {args[0]}, "
                    f"output: {output}"
                ]
        elif name == "cuGraphChildGraphNodeGetGraph":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver graph child graph node get graph: "
                    f"node: {args[0]}, output: {output}"
                ]
        elif name == "cuGraphMemAllocNodeGetParams":
            if len(args) >= 2:
                return [
                    f"// CUDA driver graph memory alloc node get params: "
                    f"node: {args[0]}, params: {args[1]}"
                ]
        elif name == "cuGraphMemFreeNodeGetParams":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA driver graph memory free node get params: "
                    f"node: {args[0]}, output: {output}"
                ]
        elif name in {"cuGraphNodeGetParams", "cuGraphNodeSetParams"}:
            if len(args) >= 2:
                action = "get" if name == "cuGraphNodeGetParams" else "set"
                return [
                    f"// CUDA driver graph node {action} params: "
                    f"node: {args[0]}, params: {args[1]}"
                ]
        elif name == "cuGraphExecNodeSetParams":
            if len(args) >= 3:
                return [
                    f"// CUDA driver graph exec set node params: "
                    f"exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                ]
        elif name in {
            "cuGraphKernelNodeGetAttribute",
            "cuGraphKernelNodeSetAttribute",
        }:
            if len(node.args) >= 3:
                action = "get" if name.endswith("GetAttribute") else "set"
                value_label = "output" if action == "get" else "value"
                value = (
                    self.format_runtime_pointer_target(node.args[2])
                    if action == "get"
                    else args[2]
                )
                return [
                    f"// CUDA driver graph kernel node {action} attribute: "
                    f"node: {args[0]}, attribute: {args[1]}, "
                    f"{value_label}: {value}"
                ]
        elif name == "cuGraphKernelNodeCopyAttributes":
            if len(args) >= 2:
                return [
                    f"// CUDA driver graph kernel node copy attributes: "
                    f"source: {args[0]}, destination: {args[1]}"
                ]
        elif name in {
            "cuGraphKernelNodeGetParams",
            "cuGraphKernelNodeSetParams",
            "cuGraphMemcpyNodeGetParams",
            "cuGraphMemcpyNodeSetParams",
            "cuGraphMemsetNodeGetParams",
            "cuGraphMemsetNodeSetParams",
            "cuGraphHostNodeGetParams",
            "cuGraphHostNodeSetParams",
        }:
            if len(args) >= 2:
                node_kind = {
                    "cuGraphKernelNodeGetParams": "kernel",
                    "cuGraphKernelNodeSetParams": "kernel",
                    "cuGraphMemcpyNodeGetParams": "memcpy",
                    "cuGraphMemcpyNodeSetParams": "memcpy",
                    "cuGraphMemsetNodeGetParams": "memset",
                    "cuGraphMemsetNodeSetParams": "memset",
                    "cuGraphHostNodeGetParams": "host",
                    "cuGraphHostNodeSetParams": "host",
                }[name]
                action = "get" if name.endswith("GetParams") else "set"
                return [
                    f"// CUDA driver graph {node_kind} node {action} params: "
                    f"node: {args[0]}, params: {args[1]}"
                ]
        elif name in {
            "cuGraphExternalSemaphoresSignalNodeGetParams",
            "cuGraphExternalSemaphoresSignalNodeSetParams",
            "cuGraphExternalSemaphoresWaitNodeGetParams",
            "cuGraphExternalSemaphoresWaitNodeSetParams",
        }:
            if len(args) >= 2:
                node_kind = "signal" if "Signal" in name else "wait"
                action = "get" if name.endswith("GetParams") else "set"
                return [
                    f"// CUDA driver graph external semaphore {node_kind} "
                    f"node {action} params: node: {args[0]}, params: {args[1]}"
                ]
        elif name in {
            "cuGraphEventRecordNodeGetEvent",
            "cuGraphEventWaitNodeGetEvent",
        }:
            if len(node.args) >= 2:
                node_kind = "record" if "Record" in name else "wait"
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA driver graph event {node_kind} node get event: "
                    f"node: {args[0]}, output: {output}"
                ]
        elif name in {
            "cuGraphEventRecordNodeSetEvent",
            "cuGraphEventWaitNodeSetEvent",
        }:
            if len(args) >= 2:
                node_kind = "record" if "Record" in name else "wait"
                return [
                    f"// CUDA driver graph event {node_kind} node set event: "
                    f"node: {args[0]}, event: {args[1]}"
                ]
        elif name in {"cuGraphExecUpdate", "cuGraphExecUpdate_v2"}:
            if len(node.args) >= 4:
                error_output = self.format_runtime_pointer_target(node.args[2])
                result_output = self.format_runtime_pointer_target(node.args[3])
                return [
                    f"// CUDA driver graph exec update: exec: {args[0]}, "
                    f"graph: {args[1]}, error node output: {error_output}, "
                    f"result output: {result_output}"
                ]
            if len(node.args) >= 3:
                result_info = self.format_runtime_pointer_target(node.args[2])
                update_kind = " update v2" if name.endswith("_v2") else " update"
                return [
                    f"// CUDA driver graph exec{update_kind}: exec: {args[0]}, "
                    f"graph: {args[1]}, result info output: {result_info}"
                ]
        elif name in {
            "cuGraphExecKernelNodeSetParams",
            "cuGraphExecMemcpyNodeSetParams",
            "cuGraphExecMemsetNodeSetParams",
            "cuGraphExecHostNodeSetParams",
        }:
            if len(args) >= 3:
                node_kind = {
                    "cuGraphExecKernelNodeSetParams": "kernel",
                    "cuGraphExecMemcpyNodeSetParams": "memcpy",
                    "cuGraphExecMemsetNodeSetParams": "memset",
                    "cuGraphExecHostNodeSetParams": "host",
                }[name]
                comment = (
                    f"// CUDA driver graph exec set {node_kind} node params: "
                    f"exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                )
                if (
                    name in {
                        "cuGraphExecMemcpyNodeSetParams",
                        "cuGraphExecMemsetNodeSetParams",
                    }
                    and len(args) >= 4
                ):
                    comment += f", context: {args[3]}"
                return [comment]
        elif name in {
            "cuGraphExecExternalSemaphoresSignalNodeSetParams",
            "cuGraphExecExternalSemaphoresWaitNodeSetParams",
        }:
            if len(args) >= 3:
                node_kind = "signal" if "Signal" in name else "wait"
                return [
                    f"// CUDA driver graph exec set external semaphore "
                    f"{node_kind} node params: exec: {args[0]}, "
                    f"node: {args[1]}, params: {args[2]}"
                ]
        elif name in {
            "cuGraphExecEventRecordNodeSetEvent",
            "cuGraphExecEventWaitNodeSetEvent",
        }:
            if len(args) >= 3:
                node_kind = "record" if "Record" in name else "wait"
                return [
                    f"// CUDA driver graph exec set event {node_kind} "
                    f"node event: exec: {args[0]}, node: {args[1]}, "
                    f"event: {args[2]}"
                ]
        elif name == "cuGraphExecChildGraphNodeSetParams":
            if len(args) >= 3:
                return [
                    "// CUDA driver graph exec set child graph node params: "
                    f"exec: {args[0]}, node: {args[1]}, child graph: {args[2]}"
                ]
        return None

    def describe_cuda_runtime_graph_node_call(self, node, name, args):
        """Comments for a runtime graph node creation, dependency or graph query call; None for any other name."""
        if name in {
            "cudaGraphAddKernelNode",
            "cudaGraphAddMemcpyNode",
            "cudaGraphAddMemsetNode",
            "cudaGraphAddHostNode",
        }:
            if len(node.args) >= 5:
                node_kind = {
                    "cudaGraphAddKernelNode": "kernel",
                    "cudaGraphAddMemcpyNode": "memcpy",
                    "cudaGraphAddMemsetNode": "memset",
                    "cudaGraphAddHostNode": "host",
                }[name]
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add {node_kind} node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}, params: {args[4]}"
                ]
        elif name == "cuGraphAddBatchMemOpNode":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graph add batch memory operation node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"params: {args[4]}"
                ]
        elif name == "cudaGraphAddMemcpyNode1D":
            if len(node.args) >= 8:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add memcpy 1D node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}, dst: {args[4]}, src: {args[5]}, "
                    f"byte count: {args[6]}, kind: {args[7]}"
                ]
        elif name in {
            "cudaGraphAddMemcpyNodeFromSymbol",
            "cudaGraphAddMemcpyNodeToSymbol",
        }:
            if len(node.args) >= 9:
                output = self.format_runtime_pointer_target(node.args[0])
                if name == "cudaGraphAddMemcpyNodeFromSymbol":
                    return [
                        "// CUDA graph add memcpy-from-symbol node: "
                        f"output: {output}, graph: {args[1]}, "
                        f"dependencies: {args[2]}, dependency count: {args[3]}, "
                        f"dst: {args[4]}, symbol: {args[5]}, "
                        f"byte count: {args[6]}, offset: {args[7]}, "
                        f"kind: {args[8]}"
                    ]
                return [
                    "// CUDA graph add memcpy-to-symbol node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"symbol: {args[4]}, src: {args[5]}, "
                    f"byte count: {args[6]}, offset: {args[7]}, kind: {args[8]}"
                ]
        elif name == "cudaGraphAddNode":
            if len(node.args) >= 6:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add generic node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"edge data: {args[3]}, dependency count: {args[4]}, "
                    f"params: {args[5]}"
                ]
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add generic node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}, params: {args[4]}"
                ]
        elif name == "cudaGraphAddChildGraphNode":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add child graph node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}, child graph: {args[4]}"
                ]
        elif name == "cudaGraphAddEmptyNode":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add empty node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}"
                ]
        elif name in {"cudaGraphAddEventRecordNode", "cudaGraphAddEventWaitNode"}:
            if len(node.args) >= 5:
                operation = (
                    "record" if name == "cudaGraphAddEventRecordNode" else "wait"
                )
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add event {operation} node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"event: {args[4]}"
                ]
        elif name in {
            "cudaGraphAddExternalSemaphoresSignalNode",
            "cudaGraphAddExternalSemaphoresWaitNode",
        }:
            if len(node.args) >= 5:
                operation = (
                    "signal"
                    if name == "cudaGraphAddExternalSemaphoresSignalNode"
                    else "wait"
                )
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add external semaphore {operation} node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"params: {args[4]}"
                ]
        elif name == "cudaGraphAddMemAllocNode":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add memory alloc node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}, params: {args[4]}"
                ]
        elif name == "cudaGraphAddMemFreeNode":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add memory free node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}, pointer: {args[4]}"
                ]
        elif name == "cudaGraphMemAllocNodeGetParams":
            if len(args) >= 2:
                return [
                    f"// CUDA graph memory alloc node get params: "
                    f"node: {args[0]}, params: {args[1]}"
                ]
        elif name == "cudaGraphMemFreeNodeGetParams":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA graph memory free node get params: "
                    f"node: {args[0]}, output: {output}"
                ]
        elif name in {"cudaGraphAddDependencies", "cudaGraphRemoveDependencies"}:
            if len(args) >= 5:
                operation = "add" if name == "cudaGraphAddDependencies" else "remove"
                return [
                    f"// CUDA graph {operation} dependencies: graph: {args[0]}, "
                    f"from: {args[1]}, to: {args[2]}, edge data: {args[3]}, "
                    f"count: {args[4]}"
                ]
            if len(args) >= 4:
                operation = "add" if name == "cudaGraphAddDependencies" else "remove"
                return [
                    f"// CUDA graph {operation} dependencies: graph: {args[0]}, "
                    f"from: {args[1]}, to: {args[2]}, count: {args[3]}"
                ]
        elif name == "cudaGraphNodeFindInClone":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph node find in clone: output: {output}, "
                    f"original node: {args[1]}, clone graph: {args[2]}"
                ]
        elif name == "cudaGraphNodeGetType":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA graph node get type: node: {args[0]}, "
                    f"output: {output}"
                ]
        elif name == "cudaGraphGetId":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [f"// CUDA graph get id: graph: {args[0]}, output: {output}"]
        elif name == "cudaGraphGetEdges":
            if len(node.args) >= 5:
                from_output = self.format_runtime_pointer_target(node.args[1])
                to_output = self.format_runtime_pointer_target(node.args[2])
                edge_data = self.format_runtime_pointer_target(node.args[3])
                count_output = self.format_runtime_pointer_target(node.args[4])
                return [
                    f"// CUDA graph get edges: graph: {args[0]}, "
                    f"from output: {from_output}, to output: {to_output}, "
                    f"edge data: {edge_data}, count output: {count_output}"
                ]
        elif name in {"cudaGraphGetNodes", "cudaGraphGetRootNodes"}:
            if len(node.args) >= 3:
                node_output = self.format_runtime_pointer_target(node.args[1])
                count_output = self.format_runtime_pointer_target(node.args[2])
                node_kind = "root nodes" if name == "cudaGraphGetRootNodes" else "nodes"
                return [
                    f"// CUDA graph get {node_kind}: graph: {args[0]}, "
                    f"nodes output: {node_output}, count output: {count_output}"
                ]
        elif name == "cudaGraphNodeGetContainingGraph":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA graph node get containing graph: node: {args[0]}, "
                    f"output: {output}"
                ]
        return None

    def describe_cuda_runtime_graph_parameter_call(self, node, name, args):
        """Comments for a runtime graph node query or node parameter call; None for any other name."""
        if name in {
            "cudaGraphNodeGetDependencies",
            "cudaGraphNodeGetDependentNodes",
        }:
            if len(node.args) >= 4:
                nodes_output = self.format_runtime_pointer_target(node.args[1])
                edge_data = self.format_runtime_pointer_target(node.args[2])
                count_output = self.format_runtime_pointer_target(node.args[3])
                if name == "cudaGraphNodeGetDependencies":
                    return [
                        f"// CUDA graph node get dependencies: node: {args[0]}, "
                        f"dependencies output: {nodes_output}, "
                        f"edge data: {edge_data}, count output: {count_output}"
                    ]
                return [
                    f"// CUDA graph node get dependent nodes: node: {args[0]}, "
                    f"dependent nodes output: {nodes_output}, "
                    f"edge data: {edge_data}, count output: {count_output}"
                ]
        elif name in {"cudaGraphNodeGetEnabled", "cudaGraphNodeSetEnabled"}:
            if len(node.args) >= 3:
                if name == "cudaGraphNodeGetEnabled":
                    output = self.format_runtime_pointer_target(node.args[2])
                    return [
                        f"// CUDA graph node get enabled: exec: {args[0]}, "
                        f"node: {args[1]}, output: {output}"
                    ]
                return [
                    f"// CUDA graph node set enabled: exec: {args[0]}, "
                    f"node: {args[1]}, enabled: {args[2]}"
                ]
        elif name in {"cudaGraphNodeGetLocalId", "cudaGraphNodeGetToolsId"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                id_kind = (
                    "local id" if name == "cudaGraphNodeGetLocalId" else "tools id"
                )
                return [
                    f"// CUDA graph node get {id_kind}: node: {args[0]}, "
                    f"output: {output}"
                ]
        elif name in {
            "cudaGraphEventRecordNodeGetEvent",
            "cudaGraphEventWaitNodeGetEvent",
        }:
            if len(node.args) >= 2:
                node_kind = (
                    "record" if name == "cudaGraphEventRecordNodeGetEvent" else "wait"
                )
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA graph event {node_kind} node get event: "
                    f"node: {args[0]}, output: {output}"
                ]
        elif name in {
            "cudaGraphEventRecordNodeSetEvent",
            "cudaGraphEventWaitNodeSetEvent",
        }:
            if len(args) >= 2:
                node_kind = (
                    "record" if name == "cudaGraphEventRecordNodeSetEvent" else "wait"
                )
                return [
                    f"// CUDA graph event {node_kind} node set event: "
                    f"node: {args[0]}, event: {args[1]}"
                ]
        elif name == "cudaGraphChildGraphNodeGetGraph":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA graph child graph node get graph: "
                    f"node: {args[0]}, output: {output}"
                ]
        elif name in {"cudaGraphNodeGetParams", "cudaGraphNodeSetParams"}:
            if len(args) >= 2:
                action = "get" if name == "cudaGraphNodeGetParams" else "set"
                return [
                    f"// CUDA graph node {action} params: "
                    f"node: {args[0]}, params: {args[1]}"
                ]
        elif name in {
            "cuGraphBatchMemOpNodeGetParams",
            "cuGraphBatchMemOpNodeSetParams",
        }:
            if len(args) >= 2:
                action = "get" if name.endswith("GetParams") else "set"
                return [
                    f"// CUDA driver graph batch memory operation node {action} "
                    f"params: node: {args[0]}, params: {args[1]}"
                ]
        elif name == "cuGraphExecBatchMemOpNodeSetParams":
            if len(args) >= 3:
                return [
                    "// CUDA driver graph exec set batch memory operation node "
                    f"params: exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                ]
        elif name in {
            "cudaGraphKernelNodeGetAttribute",
            "cudaGraphKernelNodeSetAttribute",
        }:
            if len(node.args) >= 3:
                action = "get" if name.endswith("GetAttribute") else "set"
                value_label = "output" if action == "get" else "value"
                value = (
                    self.format_runtime_pointer_target(node.args[2])
                    if action == "get"
                    else args[2]
                )
                return [
                    f"// CUDA graph kernel node {action} attribute: "
                    f"node: {args[0]}, attribute: {args[1]}, "
                    f"{value_label}: {value}"
                ]
        elif name == "cudaGraphKernelNodeCopyAttributes":
            if len(args) >= 2:
                return [
                    f"// CUDA graph kernel node copy attributes: "
                    f"source: {args[0]}, destination: {args[1]}"
                ]
        elif name in {
            "cudaGraphKernelNodeGetParams",
            "cudaGraphKernelNodeSetParams",
            "cudaGraphMemcpyNodeGetParams",
            "cudaGraphMemcpyNodeSetParams",
            "cudaGraphMemsetNodeGetParams",
            "cudaGraphMemsetNodeSetParams",
            "cudaGraphHostNodeGetParams",
            "cudaGraphHostNodeSetParams",
        }:
            if len(args) >= 2:
                node_kind = {
                    "cudaGraphKernelNodeGetParams": "kernel",
                    "cudaGraphKernelNodeSetParams": "kernel",
                    "cudaGraphMemcpyNodeGetParams": "memcpy",
                    "cudaGraphMemcpyNodeSetParams": "memcpy",
                    "cudaGraphMemsetNodeGetParams": "memset",
                    "cudaGraphMemsetNodeSetParams": "memset",
                    "cudaGraphHostNodeGetParams": "host",
                    "cudaGraphHostNodeSetParams": "host",
                }[name]
                action = "get" if name.endswith("GetParams") else "set"
                return [
                    f"// CUDA graph {node_kind} node {action} params: "
                    f"node: {args[0]}, params: {args[1]}"
                ]
        elif name == "cudaGraphMemcpyNodeSetParams1D":
            if len(args) >= 5:
                return [
                    f"// CUDA graph set memcpy 1D node params: "
                    f"node: {args[0]}, dst: {args[1]}, src: {args[2]}, "
                    f"byte count: {args[3]}, kind: {args[4]}"
                ]
        elif name in {
            "cudaGraphMemcpyNodeSetParamsFromSymbol",
            "cudaGraphMemcpyNodeSetParamsToSymbol",
        }:
            if len(args) >= 6:
                if name == "cudaGraphMemcpyNodeSetParamsFromSymbol":
                    return [
                        "// CUDA graph set memcpy-from-symbol node params: "
                        f"node: {args[0]}, dst: {args[1]}, "
                        f"symbol: {args[2]}, byte count: {args[3]}, "
                        f"offset: {args[4]}, kind: {args[5]}"
                    ]
                return [
                    "// CUDA graph set memcpy-to-symbol node params: "
                    f"node: {args[0]}, symbol: {args[1]}, src: {args[2]}, "
                    f"byte count: {args[3]}, offset: {args[4]}, "
                    f"kind: {args[5]}"
                ]
        elif name in {
            "cudaGraphExternalSemaphoresSignalNodeGetParams",
            "cudaGraphExternalSemaphoresSignalNodeSetParams",
            "cudaGraphExternalSemaphoresWaitNodeGetParams",
            "cudaGraphExternalSemaphoresWaitNodeSetParams",
        }:
            if len(args) >= 2:
                node_kind = "signal" if "Signal" in name else "wait"
                action = "get" if name.endswith("GetParams") else "set"
                return [
                    f"// CUDA graph external semaphore {node_kind} "
                    f"node {action} params: node: {args[0]}, params: {args[1]}"
                ]
        return None

    def describe_cuda_graph_update_call(self, node, name, args):
        """Comments for an executable graph update, node or graph destruction, or extended kernel launch call; None for any other name."""
        if name == "cudaGraphExecUpdate":
            if len(node.args) >= 4:
                error_output = self.format_runtime_pointer_target(node.args[2])
                result_output = self.format_runtime_pointer_target(node.args[3])
                return [
                    f"// CUDA graph exec update: exec: {args[0]}, "
                    f"graph: {args[1]}, error node output: {error_output}, "
                    f"result output: {result_output}"
                ]
            if len(args) >= 3:
                return [
                    f"// CUDA graph exec update: exec: {args[0]}, "
                    f"graph: {args[1]}, result info: {args[2]}"
                ]
        elif name in {"cudaGraphExecGetFlags", "cudaGraphExecGetId"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                if name == "cudaGraphExecGetFlags":
                    return [
                        f"// CUDA graph exec get flags: exec: {args[0]}, "
                        f"output: {output}"
                    ]
                return [f"// CUDA graph exec get id: exec: {args[0]}, output: {output}"]
        elif name in {
            "cudaGraphExecKernelNodeSetParams",
            "cudaGraphExecMemcpyNodeSetParams",
            "cudaGraphExecMemsetNodeSetParams",
            "cudaGraphExecHostNodeSetParams",
        }:
            if len(args) >= 3:
                node_kind = {
                    "cudaGraphExecKernelNodeSetParams": "kernel",
                    "cudaGraphExecMemcpyNodeSetParams": "memcpy",
                    "cudaGraphExecMemsetNodeSetParams": "memset",
                    "cudaGraphExecHostNodeSetParams": "host",
                }[name]
                return [
                    f"// CUDA graph exec set {node_kind} node params: "
                    f"exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                ]
        elif name == "cudaGraphExecMemcpyNodeSetParams1D":
            if len(args) >= 6:
                return [
                    f"// CUDA graph exec set memcpy 1D node params: "
                    f"exec: {args[0]}, node: {args[1]}, dst: {args[2]}, "
                    f"src: {args[3]}, byte count: {args[4]}, kind: {args[5]}"
                ]
        elif name in {
            "cudaGraphExecMemcpyNodeSetParamsFromSymbol",
            "cudaGraphExecMemcpyNodeSetParamsToSymbol",
        }:
            if len(args) >= 7:
                if name == "cudaGraphExecMemcpyNodeSetParamsFromSymbol":
                    return [
                        "// CUDA graph exec set memcpy-from-symbol node params: "
                        f"exec: {args[0]}, node: {args[1]}, dst: {args[2]}, "
                        f"symbol: {args[3]}, byte count: {args[4]}, "
                        f"offset: {args[5]}, kind: {args[6]}"
                    ]
                return [
                    "// CUDA graph exec set memcpy-to-symbol node params: "
                    f"exec: {args[0]}, node: {args[1]}, symbol: {args[2]}, "
                    f"src: {args[3]}, byte count: {args[4]}, "
                    f"offset: {args[5]}, kind: {args[6]}"
                ]
        elif name == "cudaGraphExecNodeSetParams":
            if len(args) >= 3:
                return [
                    f"// CUDA graph exec set node params: "
                    f"exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                ]
        elif name in {
            "cudaGraphExecExternalSemaphoresSignalNodeSetParams",
            "cudaGraphExecExternalSemaphoresWaitNodeSetParams",
        }:
            if len(args) >= 3:
                node_kind = "signal" if "Signal" in name else "wait"
                return [
                    f"// CUDA graph exec set external semaphore {node_kind} "
                    f"node params: exec: {args[0]}, node: {args[1]}, "
                    f"params: {args[2]}"
                ]
        elif name in {
            "cudaGraphExecEventRecordNodeSetEvent",
            "cudaGraphExecEventWaitNodeSetEvent",
        }:
            if len(args) >= 3:
                node_kind = (
                    "record"
                    if name == "cudaGraphExecEventRecordNodeSetEvent"
                    else "wait"
                )
                return [
                    f"// CUDA graph exec set event {node_kind} node event: "
                    f"exec: {args[0]}, node: {args[1]}, event: {args[2]}"
                ]
        elif name == "cudaGraphExecChildGraphNodeSetParams":
            if len(args) >= 3:
                return [
                    f"// CUDA graph exec set child graph node params: "
                    f"exec: {args[0]}, node: {args[1]}, child graph: {args[2]}"
                ]
        elif name == "cudaGraphDestroyNode":
            if args:
                return [f"// CUDA graph destroy node: {args[0]}"]
        elif name == "cudaGraphDestroy":
            if args:
                return [f"// CUDA graph destroy: {args[0]}"]
        elif name in {"cudaLaunchKernelExC", "cudaLaunchKernelEx"}:
            if len(node.args) >= 2:
                config = self.format_runtime_pointer_target(node.args[0])
                function = self.format_cuda_kernel_function_reference(node.args[1])
                launch_args = node.args[2:]
                if name == "cudaLaunchKernelExC" and launch_args:
                    launch_args = self.resolve_packed_launch_args([launch_args[0]])

                comment = (
                    f"// CUDA launch kernel ex: config: {config}, "
                    f"function: {function}"
                )
                if launch_args:
                    args_str = ", ".join(
                        self.format_kernel_launch_arg(arg) for arg in launch_args
                    )
                    comment += f", args: {args_str}"
                return [comment]
        elif name == "cuGraphDestroyNode":
            if args:
                return [f"// CUDA driver graph destroy node: {args[0]}"]
        elif name == "cuGraphDestroy":
            if args:
                return [f"// CUDA driver graph destroy: {args[0]}"]
        return None
