"""HIP graph calls of the runtime and driver APIs in the HIP to CrossGL converter, described as comments."""


class HipRuntimeGraphCallMixin:
    """Runtime graph call members of ``HipToCrossGLConverter``."""

    def describe_hip_graph_capture_and_driver_node_call(self, name, raw_args, args):
        """Comments for a stream capture, graph create, destroy or clone, generic node or driver graph node call; None for any other name."""
        if name == "hipStreamBeginCaptureToGraph":
            if len(args) >= 6:
                return [
                    f"// HIP stream begin capture to graph: stream: {args[0]}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency data: {args[3]}, count: {args[4]}, mode: {args[5]}"
                ]
        if name == "hipThreadExchangeStreamCaptureMode":
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [f"// HIP exchange stream capture mode: output: {output}"]
        if name == "hipGraphCreate":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [f"// HIP graph create: output: {output}, flags: {args[1]}"]
        if name == "hipGraphDestroy":
            if args:
                return [f"// HIP graph destroy: {args[0]}"]
        if name == "hipGraphClone":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [f"// HIP graph clone: output: {output}, source: {args[1]}"]
        if name == "hipGraphAddNode":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph add generic node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, params: {args[4]}"
                ]
        if name == "hipDrvGraphAddMemcpyNode":
            if len(args) >= 6:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP driver graph add memcpy node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, params: {args[4]}, context: {args[5]}"
                ]
        if name == "hipDrvGraphAddMemsetNode":
            if len(args) >= 6:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP driver graph add memset node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, params: {args[4]}, context: {args[5]}"
                ]
        if name == "hipDrvGraphAddMemFreeNode":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP driver graph add memory free node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, pointer: {args[4]}"
                ]
        if name in {"hipDrvGraphMemcpyNodeGetParams", "hipDrvGraphMemcpyNodeSetParams"}:
            if len(args) >= 2:
                action = "get" if "GetParams" in name else "set"
                params = (
                    self.format_runtime_raw_output_target(raw_args[1])
                    if action == "get"
                    else args[1]
                )
                label = "params output" if action == "get" else "params"
                return [
                    f"// HIP driver graph memcpy node {action} params: "
                    f"node: {args[0]}, {label}: {params}"
                ]
        if name in {
            "hipDrvGraphExecMemcpyNodeSetParams",
            "hipDrvGraphExecMemsetNodeSetParams",
        }:
            if len(args) >= 4:
                node_kind = "memcpy" if "Memcpy" in name else "memset"
                return [
                    f"// HIP driver graph exec {node_kind} node set params: "
                    f"exec: {args[0]}, node: {args[1]}, params: {args[2]}, "
                    f"context: {args[3]}"
                ]
        return None

    def describe_hip_graph_node_creation_call(self, name, raw_args, args):
        """Comments for adding an empty, host, kernel, memcpy, memset, child graph, event or memory node, or for a 1D or symbol memcpy node and its parameter setters; None for any other name."""
        if name in {
            "hipGraphAddEmptyNode",
            "hipGraphAddHostNode",
            "hipGraphAddKernelNode",
            "hipGraphAddMemcpyNode",
            "hipGraphAddMemsetNode",
        }:
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(raw_args[0])
                node_kind = {
                    "hipGraphAddEmptyNode": "empty",
                    "hipGraphAddHostNode": "host",
                    "hipGraphAddKernelNode": "kernel",
                    "hipGraphAddMemcpyNode": "memcpy",
                    "hipGraphAddMemsetNode": "memset",
                }[name]
                comment = (
                    f"// HIP graph add {node_kind} node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, count: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", params: {args[4]}"
                return [comment]
        if name == "hipGraphAddChildGraphNode":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph add child graph node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, child graph: {args[4]}"
                ]
        if name in {"hipGraphAddEventRecordNode", "hipGraphAddEventWaitNode"}:
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                action = "record" if name == "hipGraphAddEventRecordNode" else "wait"
                return [
                    f"// HIP graph add event {action} node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, event: {args[4]}"
                ]
        if name in {
            "hipGraphAddMemAllocNode",
            "hipGraphAddMemFreeNode",
        }:
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                node_kind = "alloc" if name == "hipGraphAddMemAllocNode" else "free"
                detail_label = "params" if node_kind == "alloc" else "pointer"
                return [
                    f"// HIP graph add memory {node_kind} node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, {detail_label}: {args[4]}"
                ]
        if name in {"hipGraphMemAllocNodeGetParams", "hipGraphMemFreeNodeGetParams"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                node_kind = "alloc" if "Alloc" in name else "free"
                detail_label = (
                    "params output" if node_kind == "alloc" else "pointer output"
                )
                return [
                    f"// HIP graph memory {node_kind} node get params: "
                    f"node: {args[0]}, {detail_label}: {output}"
                ]
        if name in {
            "hipGraphAddMemcpyNode1D",
            "hipGraphAddMemcpyNodeFromSymbol",
            "hipGraphAddMemcpyNodeToSymbol",
        }:
            if name == "hipGraphAddMemcpyNode1D" and len(args) >= 8:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph add memcpy 1D node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, count: {args[3]}, "
                    f"destination: {args[4]}, source: {args[5]}, bytes: {args[6]}, "
                    f"kind: {args[7]}"
                ]
            if len(args) >= 9:
                output = self.format_runtime_raw_output_target(raw_args[0])
                copy_kind = {
                    "hipGraphAddMemcpyNodeFromSymbol": "from symbol",
                    "hipGraphAddMemcpyNodeToSymbol": "to symbol",
                }[name]
                return [
                    f"// HIP graph add memcpy {copy_kind} node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, count: {args[3]}, "
                    f"destination: {args[4]}, source: {args[5]}, bytes: {args[6]}, "
                    f"offset: {args[7]}, kind: {args[8]}"
                ]
        if name in {
            "hipGraphMemcpyNodeSetParams1D",
            "hipGraphMemcpyNodeSetParamsFromSymbol",
            "hipGraphMemcpyNodeSetParamsToSymbol",
        }:
            if name == "hipGraphMemcpyNodeSetParams1D" and len(args) >= 5:
                return [
                    f"// HIP graph memcpy 1D node set params: "
                    f"node: {args[0]}, destination: {args[1]}, source: {args[2]}, "
                    f"bytes: {args[3]}, kind: {args[4]}"
                ]
            if len(args) >= 6:
                copy_kind = {
                    "hipGraphMemcpyNodeSetParamsFromSymbol": "from symbol",
                    "hipGraphMemcpyNodeSetParamsToSymbol": "to symbol",
                }[name]
                return [
                    f"// HIP graph memcpy {copy_kind} node set params: "
                    f"node: {args[0]}, destination: {args[1]}, source: {args[2]}, "
                    f"bytes: {args[3]}, offset: {args[4]}, kind: {args[5]}"
                ]
        if name in {
            "hipGraphExecMemcpyNodeSetParams1D",
            "hipGraphExecMemcpyNodeSetParamsFromSymbol",
            "hipGraphExecMemcpyNodeSetParamsToSymbol",
        }:
            if name == "hipGraphExecMemcpyNodeSetParams1D" and len(args) >= 6:
                return [
                    f"// HIP graph exec memcpy 1D node set params: "
                    f"exec: {args[0]}, node: {args[1]}, destination: {args[2]}, "
                    f"source: {args[3]}, bytes: {args[4]}, kind: {args[5]}"
                ]
            if len(args) >= 7:
                copy_kind = {
                    "hipGraphExecMemcpyNodeSetParamsFromSymbol": "from symbol",
                    "hipGraphExecMemcpyNodeSetParamsToSymbol": "to symbol",
                }[name]
                return [
                    f"// HIP graph exec memcpy {copy_kind} node set params: "
                    f"exec: {args[0]}, node: {args[1]}, destination: {args[2]}, "
                    f"source: {args[3]}, bytes: {args[4]}, offset: {args[5]}, "
                    f"kind: {args[6]}"
                ]
        return None

    def describe_hip_graph_topology_call(self, name, raw_args, args):
        """Comments for a dependency, node or edge query, a clone lookup, the node type, a child graph, kernel node attributes, the enabled state or node destruction; None for any other name."""
        if name in {"hipGraphAddDependencies", "hipGraphRemoveDependencies"}:
            if len(args) >= 4:
                action = "add" if name == "hipGraphAddDependencies" else "remove"
                return [
                    f"// HIP graph {action} dependencies: graph: {args[0]}, "
                    f"from: {args[1]}, to: {args[2]}, count: {args[3]}"
                ]
        if name in {"hipGraphGetNodes", "hipGraphGetRootNodes"}:
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[2])
                node_set = "nodes" if name == "hipGraphGetNodes" else "root nodes"
                return [
                    f"// HIP graph get {node_set}: graph: {args[0]}, "
                    f"nodes output: {args[1]}, count output: {output}"
                ]
        if name == "hipGraphGetEdges":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(raw_args[3])
                return [
                    f"// HIP graph get edges: graph: {args[0]}, "
                    f"from output: {args[1]}, to output: {args[2]}, "
                    f"count output: {output}"
                ]
        if name in {"hipGraphNodeGetDependencies", "hipGraphNodeGetDependentNodes"}:
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[2])
                node_set = (
                    "dependencies"
                    if name == "hipGraphNodeGetDependencies"
                    else "dependent nodes"
                )
                return [
                    f"// HIP graph node get {node_set}: node: {args[0]}, "
                    f"nodes output: {args[1]}, count output: {output}"
                ]
        if name == "hipGraphNodeFindInClone":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph node find in clone: output: {output}, "
                    f"original: {args[1]}, clone graph: {args[2]}"
                ]
        if name == "hipGraphNodeGetType":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                return [
                    f"// HIP graph node get type: node: {args[0]}, output: {output}"
                ]
        if name == "hipGraphChildGraphNodeGetGraph":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                return [
                    f"// HIP graph child node get graph: "
                    f"node: {args[0]}, output: {output}"
                ]
        if name == "hipGraphKernelNodeCopyAttributes":
            if len(args) >= 2:
                return [
                    f"// HIP graph kernel node copy attributes: "
                    f"source: {args[0]}, destination: {args[1]}"
                ]
        if name in {"hipGraphKernelNodeGetAttribute", "hipGraphKernelNodeSetAttribute"}:
            if len(args) >= 3:
                action = "get" if name == "hipGraphKernelNodeGetAttribute" else "set"
                value = (
                    self.format_runtime_raw_output_target(raw_args[2])
                    if action == "get"
                    else self.format_runtime_pointer_target(raw_args[2])
                )
                label = "output" if action == "get" else "value"
                return [
                    f"// HIP graph kernel node {action} attribute: "
                    f"node: {args[0]}, attribute: {args[1]}, {label}: {value}"
                ]
        if name in {"hipGraphNodeSetEnabled", "hipGraphNodeGetEnabled"}:
            if len(args) >= 3:
                action = "get" if name == "hipGraphNodeGetEnabled" else "set"
                value = (
                    self.format_runtime_raw_output_target(raw_args[2])
                    if action == "get"
                    else args[2]
                )
                label = "output" if action == "get" else "value"
                return [
                    f"// HIP graph node {action} enabled: "
                    f"exec: {args[0]}, node: {args[1]}, {label}: {value}"
                ]
        if name == "hipGraphDestroyNode":
            if args:
                return [f"// HIP graph destroy node: {args[0]}"]
        return None

    def describe_hip_graph_execution_call(self, name, raw_args, args):
        """Comments for graph instantiation, upload, launch, executable graph destruction, update or flags, or a node parameter query or update; None for any other name."""
        if name == "hipGraphInstantiate":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                error_output = self.format_runtime_raw_output_target(raw_args[2])
                return [
                    f"// HIP graph instantiate: output: {output}, graph: {args[1]}, "
                    f"error node output: {error_output}, log buffer: {args[3]}, "
                    f"log bytes: {args[4]}"
                ]
        if name == "hipGraphInstantiateWithFlags":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph instantiate with flags: output: {output}, "
                    f"graph: {args[1]}, flags: {args[2]}"
                ]
        if name == "hipGraphInstantiateWithParams":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph instantiate with params: output: {output}, "
                    f"graph: {args[1]}, params: {args[2]}"
                ]
        if name == "hipGraphUpload":
            if len(args) >= 2:
                return [f"// HIP graph upload: exec: {args[0]}, stream: {args[1]}"]
        if name == "hipGraphLaunch":
            if len(args) >= 2:
                return [f"// HIP graph launch: exec: {args[0]}, stream: {args[1]}"]
        if name == "hipGraphExecDestroy":
            if args:
                return [f"// HIP graph exec destroy: {args[0]}"]
        if name == "hipGraphExecUpdate":
            if len(args) >= 4:
                error_output = self.format_runtime_raw_output_target(raw_args[2])
                result_output = self.format_runtime_raw_output_target(raw_args[3])
                return [
                    f"// HIP graph exec update: exec: {args[0]}, graph: {args[1]}, "
                    f"error node output: {error_output}, result output: {result_output}"
                ]
        if name == "hipGraphExecGetFlags":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                return [
                    f"// HIP graph exec get flags: exec: {args[0]}, output: {output}"
                ]
        if name in {"hipGraphNodeSetParams", "hipGraphExecNodeSetParams"}:
            if len(args) >= 2:
                if name == "hipGraphNodeSetParams":
                    return [
                        f"// HIP graph generic node set params: "
                        f"node: {args[0]}, params: {args[1]}"
                    ]
                if len(args) >= 3:
                    return [
                        f"// HIP graph exec generic node set params: "
                        f"exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                    ]
        if name in {
            "hipGraphKernelNodeGetParams",
            "hipGraphKernelNodeSetParams",
            "hipGraphMemcpyNodeGetParams",
            "hipGraphMemcpyNodeSetParams",
            "hipGraphMemsetNodeGetParams",
            "hipGraphMemsetNodeSetParams",
            "hipGraphHostNodeGetParams",
            "hipGraphHostNodeSetParams",
        }:
            if len(args) >= 2:
                if "GetParams" in name:
                    self.clear_lvalue_metadata_source(raw_args[1])
                return [self.format_hip_graph_node_params_comment(name, args)]
        if name in {
            "hipGraphExecKernelNodeSetParams",
            "hipGraphExecMemcpyNodeSetParams",
            "hipGraphExecMemsetNodeSetParams",
            "hipGraphExecHostNodeSetParams",
            "hipGraphExecChildGraphNodeSetParams",
        }:
            if len(args) >= 3:
                node_kind = self.get_hip_graph_node_kind(name)
                return [
                    f"// HIP graph exec set {node_kind} node params: exec: {args[0]}, "
                    f"node: {args[1]}, params: {args[2]}"
                ]
        return None

    def describe_hip_graph_event_and_resource_call(self, name, raw_args, args):
        """Comments for an event record or wait node, an external semaphore node, graph memory attributes or trim, a debug dump or a user object call; None for any other name."""
        if name in {
            "hipGraphEventRecordNodeGetEvent",
            "hipGraphEventWaitNodeGetEvent",
        }:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                action = (
                    "record" if name == "hipGraphEventRecordNodeGetEvent" else "wait"
                )
                return [
                    f"// HIP graph event {action} node get event: "
                    f"node: {args[0]}, output: {output}"
                ]
        if name in {
            "hipGraphEventRecordNodeSetEvent",
            "hipGraphEventWaitNodeSetEvent",
        }:
            if len(args) >= 2:
                action = (
                    "record" if name == "hipGraphEventRecordNodeSetEvent" else "wait"
                )
                return [
                    f"// HIP graph event {action} node set event: "
                    f"node: {args[0]}, event: {args[1]}"
                ]
        if name in {
            "hipGraphExecEventRecordNodeSetEvent",
            "hipGraphExecEventWaitNodeSetEvent",
        }:
            if len(args) >= 3:
                action = (
                    "record"
                    if name == "hipGraphExecEventRecordNodeSetEvent"
                    else "wait"
                )
                return [
                    f"// HIP graph exec event {action} node set event: "
                    f"exec: {args[0]}, node: {args[1]}, event: {args[2]}"
                ]
        if name in {
            "hipGraphAddExternalSemaphoresSignalNode",
            "hipGraphAddExternalSemaphoresWaitNode",
        }:
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                action = (
                    "signal"
                    if name == "hipGraphAddExternalSemaphoresSignalNode"
                    else "wait"
                )
                return [
                    f"// HIP graph add external semaphore {action} node: "
                    f"output: {output}, graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, params: {args[4]}"
                ]
        if name in {
            "hipGraphExternalSemaphoresSignalNodeGetParams",
            "hipGraphExternalSemaphoresSignalNodeSetParams",
            "hipGraphExternalSemaphoresWaitNodeGetParams",
            "hipGraphExternalSemaphoresWaitNodeSetParams",
        }:
            if len(args) >= 2:
                action = "signal" if "Signal" in name else "wait"
                direction = "get" if "GetParams" in name else "set"
                if direction == "get":
                    self.clear_lvalue_metadata_source(raw_args[1])
                return [
                    f"// HIP graph external semaphore {action} node {direction} params: "
                    f"node: {args[0]}, params: {args[1]}"
                ]
        if name in {
            "hipGraphExecExternalSemaphoresSignalNodeSetParams",
            "hipGraphExecExternalSemaphoresWaitNodeSetParams",
        }:
            if len(args) >= 3:
                action = "signal" if "Signal" in name else "wait"
                return [
                    f"// HIP graph exec external semaphore {action} node set params: "
                    f"exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                ]
        if name in {"hipDeviceGetGraphMemAttribute", "hipDeviceSetGraphMemAttribute"}:
            if len(args) >= 3:
                action = "get" if name == "hipDeviceGetGraphMemAttribute" else "set"
                value = (
                    self.format_runtime_raw_output_target(raw_args[2])
                    if action == "get"
                    else self.format_runtime_pointer_target(raw_args[2])
                )
                label = "output" if action == "get" else "value"
                return [
                    f"// HIP device graph memory {action} attribute: "
                    f"device: {args[0]}, attribute: {args[1]}, {label}: {value}"
                ]
        if name == "hipDeviceGraphMemTrim":
            if args:
                return [f"// HIP device graph memory trim: device: {args[0]}"]
        if name == "hipGraphDebugDotPrint":
            if len(args) >= 3:
                return [
                    f"// HIP graph debug dot print: graph: {args[0]}, "
                    f"path: {args[1]}, flags: {args[2]}"
                ]
        if name == "hipUserObjectCreate":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP user object create: output: {output}, "
                    f"resource: {args[1]}, destructor: {args[2]}, "
                    f"initial refcount: {args[3]}, flags: {args[4]}"
                ]
        if name in {"hipUserObjectRetain", "hipUserObjectRelease"}:
            if len(args) >= 2:
                action = "retain" if name == "hipUserObjectRetain" else "release"
                return [
                    f"// HIP user object {action}: object: {args[0]}, count: {args[1]}"
                ]
        if name in {"hipGraphRetainUserObject", "hipGraphReleaseUserObject"}:
            if len(args) >= 3:
                action = "retain" if name == "hipGraphRetainUserObject" else "release"
                comment = (
                    f"// HIP graph {action} user object: graph: {args[0]}, "
                    f"object: {args[1]}, count: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", flags: {args[3]}"
                return [comment]
        return None
