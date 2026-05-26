"""Optional hipcc-backed smoke tests for native HIP frontend samples."""

import shutil
import subprocess

import pytest

from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser


def compile_hip_if_hipcc_available(hip_code, tmp_path):
    """Compile native HIP source when hipcc is available."""
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")

    source_path = tmp_path / "native_smoke.hip"
    object_path = tmp_path / "native_smoke.o"
    source_path.write_text(hip_code, encoding="utf-8")

    result = subprocess.run(
        [hipcc, "-std=c++17", "-c", str(source_path), "-o", str(object_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + "\n\n" + hip_code


def convert_native_hip_to_crossgl(hip_code):
    tokens = HipLexer(hip_code).tokenize()
    ast = HipParser(tokens).parse()
    return HipToCrossGLConverter().generate(ast)


def test_native_hip_atomic_barrier_smoke_parses_and_compiles_if_available(tmp_path):
    """Smoke native HIP atomics, shared memory, barriers, and fences."""
    hip_code = """
    #include <hip/hip_runtime.h>

    __global__ void native_smoke(int* out, int* counter) {
        __shared__ int scratch[32];
        int lane = threadIdx.x;
        scratch[lane] = lane;
        __syncthreads();
        int old = atomicAdd(counter, 1);
        __threadfence();
        out[lane] = old + scratch[lane];
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Kernel: native_smoke" in crossgl
    assert "var<workgroup> scratch: array<i32, 32>;" in crossgl
    assert "var lane: i32 = gl_LocalInvocationID.x;" in crossgl
    assert "scratch[lane] = lane;" in crossgl
    assert "workgroupBarrier();" in crossgl
    assert "var old: i32 = atomicAdd(counter, 1);" in crossgl
    assert "memoryBarrier();" in crossgl
    assert "out[lane] = (old + scratch[lane]);" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_vector_constructor_smoke_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP vector constructors and vector member reads."""
    hip_code = """
    #include <hip/hip_runtime.h>

    __global__ void vector_smoke(float* out) {
        float2 uv = make_float2(1.0f, 2.0f);
        float4 color = make_float4(uv.x, uv.y, 3.0f, 4.0f);
        out[threadIdx.x] = color.w;
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Kernel: vector_smoke" in crossgl
    assert "var uv: vec2<f32> = vec2<f32>(1.0f, 2.0f);" in crossgl
    assert "var color: vec4<f32> = vec4<f32>(uv.x, uv.y, 3.0f, 4.0f);" in crossgl
    assert "out[gl_LocalInvocationID.x] = color.w;" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_texture_surface_smoke_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP texture and surface object access."""
    hip_code = """
    #include <hip/hip_runtime.h>

    __global__ void resource_smoke(hipTextureObject_t tex, hipSurfaceObject_t surf, float4* out) {
        float2 uv = make_float2(0.25f, 0.75f);
        int2 pixel = make_int2(threadIdx.x, 0);
        float4 sampled = tex2D<float4>(tex, uv.x, uv.y);
        float4 loaded = surf2Dread<float4>(
            surf,
            pixel.x * sizeof(float4),
            pixel.y
        );
        surf2Dwrite(sampled, surf, pixel.x * sizeof(float4), pixel.y);
        out[threadIdx.x] = loaded;
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Kernel: resource_smoke" in crossgl
    assert "sampler2D tex" in crossgl
    assert "image2D surf" in crossgl
    assert (
        "@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>"
        in crossgl
    )
    assert "var uv: vec2<f32> = vec2<f32>(0.25f, 0.75f);" in crossgl
    assert "var pixel: vec2<i32> = vec2<i32>(gl_LocalInvocationID.x, 0);" in crossgl
    assert "var sampled: vec4<f32> = texture(tex, vec2<f32>(uv.x, uv.y));" in crossgl
    assert (
        "var loaded: vec4<f32> = imageLoad(surf, vec2<i32>(pixel.x, pixel.y));"
        in crossgl
    )
    assert "imageStore(surf, vec2<i32>(pixel.x, pixel.y), sampled);" in crossgl
    assert "out[gl_LocalInvocationID.x] = loaded;" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_texture_surface_lifecycle_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP texture and surface object lifecycle APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void resource_lifecycle(
        hipResourceDesc* resourceDesc,
        hipTextureDesc* textureDesc
    ) {
        hipTextureObject_t tex = 0;
        hipSurfaceObject_t surf = 0;
        hipCreateTextureObject(&tex, resourceDesc, textureDesc, NULL);
        hipCreateSurfaceObject(&surf, resourceDesc);
        hipDestroyTextureObject(tex);
        hipDestroySurfaceObject(surf);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: resource_lifecycle" in crossgl
    assert (
        "void resource_lifecycle("
        "ptr<hipResourceDesc> resourceDesc, ptr<hipTextureDesc> textureDesc)" in crossgl
    )
    assert "var tex: sampler = 0;" in crossgl
    assert "var surf: image2D = 0;" in crossgl
    assert (
        "// HIP texture object create: tex, resource: resourceDesc, "
        "texture desc: textureDesc, resource view: NULL"
    ) in crossgl
    assert "// HIP surface object create: surf, resource: resourceDesc" in crossgl
    assert "// HIP texture object destroy: tex" in crossgl
    assert "// HIP surface object destroy: surf" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_memory_lifecycle_parses_and_compiles_if_available(tmp_path):
    """Smoke native HIP allocation, memset, copy, and free APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void memory_lifecycle(float* host, size_t n) {
        float* device = NULL;
        hipMalloc((void**)&device, n * sizeof(float));
        hipMemset(device, 0, n * sizeof(float));
        hipMemcpy(device, host, n * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(host, device, n * sizeof(float), hipMemcpyDeviceToHost);
        hipFree(device);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: memory_lifecycle" in crossgl
    assert "void memory_lifecycle(ptr<f32> host, u32 n)" in crossgl
    assert "var device: ptr<f32> = NULL;" in crossgl
    assert "// HIP memory allocate: device, bytes: (n * sizeof(float))" in crossgl
    assert "// HIP memory set: device, value: 0, bytes: (n * sizeof(float))" in crossgl
    assert (
        "// HIP memory copy: host -> device, bytes: (n * sizeof(float)), "
        "kind: hipMemcpyHostToDevice"
    ) in crossgl
    assert (
        "// HIP memory copy: device -> host, bytes: (n * sizeof(float)), "
        "kind: hipMemcpyDeviceToHost"
    ) in crossgl
    assert "// HIP memory free: device" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_managed_async_memory_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP managed, async, prefetch, and advisory memory APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void managed_async_memory(float* host, size_t n, hipStream_t stream) {
        float* managed = NULL;
        float* async_ptr = NULL;
        unsigned long long access_flags = 0;
        hipMemRangeAttribute attribute = hipMemRangeAttributeAccessedBy;

        hipMallocManaged((void**)&managed, n * sizeof(float));
        hipMallocAsync((void**)&async_ptr, n * sizeof(float), stream);
        hipMemPrefetchAsync(managed, n * sizeof(float), 0, stream);
        hipMemAdvise(managed, n * sizeof(float), hipMemAdviseSetReadMostly, 0);
        hipMemRangeGetAttribute(
            &access_flags, sizeof(access_flags), attribute,
            managed, n * sizeof(float)
        );
        hipStreamAttachMemAsync(
            stream, managed, n * sizeof(float), hipMemAttachSingle
        );
        hipMemcpyAsync(
            async_ptr, host, n * sizeof(float), hipMemcpyHostToDevice, stream
        );
        hipFreeAsync(async_ptr, stream);
        hipFree(managed);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: managed_async_memory" in crossgl
    assert (
        "void managed_async_memory(ptr<f32> host, u32 n, hipStream_t stream)" in crossgl
    )
    assert "var managed: ptr<f32> = NULL;" in crossgl
    assert "var async_ptr: ptr<f32> = NULL;" in crossgl
    assert "var access_flags: u64 = 0;" in crossgl
    assert "var attribute: hipMemRangeAttribute = hipMemRangeAttributeAccessedBy;" in (
        crossgl
    )
    assert "// HIP memory allocate: managed, bytes: (n * sizeof(float))" in crossgl
    assert (
        "// HIP async memory allocate: async_ptr, bytes: (n * sizeof(float)), "
        "stream: stream"
    ) in crossgl
    assert (
        "// HIP memory prefetch: pointer: managed, bytes: (n * sizeof(float)), "
        "device: 0, stream: stream"
    ) in crossgl
    assert (
        "// HIP memory advise: pointer: managed, bytes: (n * sizeof(float)), "
        "advice: hipMemAdviseSetReadMostly, device: 0"
    ) in crossgl
    assert (
        "// HIP memory range get attribute: output: access_flags, "
        "output bytes: sizeof(access_flags), attribute: attribute, "
        "pointer: managed, range bytes: (n * sizeof(float))"
    ) in crossgl
    assert (
        "// HIP stream attach memory: stream: stream, pointer: managed, "
        "bytes: (n * sizeof(float)), flags: hipMemAttachSingle"
    ) in crossgl
    assert (
        "// HIP memory copy: host -> async_ptr, bytes: (n * sizeof(float)), "
        "kind: hipMemcpyHostToDevice, stream: stream"
    ) in crossgl
    assert "// HIP async memory free: async_ptr, stream: stream" in crossgl
    assert "// HIP memory free: managed" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_stream_event_lifecycle_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP stream and event lifecycle APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void stream_event_lifecycle(float* elapsed_ms) {
        hipStream_t stream;
        hipEvent_t start;
        hipEvent_t stop;
        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, stream);
        hipStreamWaitEvent(stream, start, 0);
        hipEventRecord(stop, stream);
        hipStreamSynchronize(stream);
        hipEventSynchronize(stop);
        hipEventElapsedTime(elapsed_ms, start, stop);
        hipEventDestroy(start);
        hipEventDestroy(stop);
        hipStreamDestroy(stream);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: stream_event_lifecycle" in crossgl
    assert "void stream_event_lifecycle(ptr<f32> elapsed_ms)" in crossgl
    assert "var stream: hipStream_t;" in crossgl
    assert "var start: hipEvent_t;" in crossgl
    assert "var stop: hipEvent_t;" in crossgl
    assert "// HIP stream create: stream, flags: hipStreamNonBlocking" in crossgl
    assert "// HIP event create: start" in crossgl
    assert "// HIP event create: stop" in crossgl
    assert "// HIP event record: start, stream: stream" in crossgl
    assert "// HIP stream wait event: stream waits for start, flags: 0" in crossgl
    assert "// HIP event record: stop, stream: stream" in crossgl
    assert "// HIP synchronize: stream" in crossgl
    assert "// HIP event synchronize: stop" in crossgl
    assert "// HIP event elapsed time: start -> stop, output: elapsed_ms" in crossgl
    assert "// HIP event destroy: start" in crossgl
    assert "// HIP event destroy: stop" in crossgl
    assert "// HIP stream destroy: stream" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_lifecycle_parses_and_compiles_if_available(tmp_path):
    """Smoke native HIP graph capture, node creation, instantiate, and launch APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void graph_lifecycle(hipStream_t stream) {
        hipGraph_t graph;
        hipGraph_t captured_graph;
        hipGraphExec_t exec;
        hipGraphNode_t empty_node;
        hipGraphNode_t error_node;
        char log[128];
        hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
        hipStreamEndCapture(stream, &captured_graph);
        hipGraphDestroy(captured_graph);
        hipGraphCreate(&graph, 0);
        hipGraphAddEmptyNode(&empty_node, graph, NULL, 0);
        hipGraphInstantiate(&exec, graph, &error_node, log, 128);
        hipGraphLaunch(exec, stream);
        hipGraphExecDestroy(exec);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: graph_lifecycle" in crossgl
    assert "void graph_lifecycle(hipStream_t stream)" in crossgl
    assert "var graph: hipGraph_t;" in crossgl
    assert "var captured_graph: hipGraph_t;" in crossgl
    assert "var exec: hipGraphExec_t;" in crossgl
    assert "var empty_node: hipGraphNode_t;" in crossgl
    assert "var error_node: hipGraphNode_t;" in crossgl
    assert "var log: array<i8, 128>;" in crossgl
    assert (
        "// HIP stream begin capture: stream: stream, "
        "mode: hipStreamCaptureModeGlobal"
    ) in crossgl
    assert (
        "// HIP stream end capture: stream: stream, graph output: captured_graph"
        in crossgl
    )
    assert "// HIP graph destroy: captured_graph" in crossgl
    assert "// HIP graph create: output: graph, flags: 0" in crossgl
    assert (
        "// HIP graph add empty node: output: empty_node, graph: graph, "
        "dependencies: NULL, count: 0"
    ) in crossgl
    assert (
        "// HIP graph instantiate: output: exec, graph: graph, "
        "error node output: error_node, log buffer: log, log bytes: 128"
    ) in crossgl
    assert "// HIP graph launch: exec: exec, stream: stream" in crossgl
    assert "// HIP graph exec destroy: exec" in crossgl
    assert "// HIP graph destroy: graph" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_exec_update_parses_and_compiles_if_available(tmp_path):
    """Smoke native HIP graph exec update and kernel-node parameter APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    __global__ void graph_update_kernel(float* out, float value) {
        out[threadIdx.x] = value;
    }

    void graph_exec_update(float* device_ptr) {
        hipGraph_t graph;
        hipGraphExec_t exec;
        hipGraphNode_t kernel_node;
        hipGraphNode_t error_node;
        hipGraphExecUpdateResult update_result;
        hipKernelNodeParams kernel_params;
        hipKernelNodeParams fetched_params;
        char log[128];

        hipGraphCreate(&graph, 0);
        hipGraphAddKernelNode(&kernel_node, graph, NULL, 0, &kernel_params);
        hipGraphKernelNodeGetParams(kernel_node, &fetched_params);
        hipGraphKernelNodeSetParams(kernel_node, &kernel_params);
        hipGraphInstantiate(&exec, graph, &error_node, log, 128);
        hipGraphExecKernelNodeSetParams(exec, kernel_node, &kernel_params);
        hipGraphExecUpdate(exec, graph, &error_node, &update_result);
        hipGraphExecDestroy(exec);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Kernel: graph_update_kernel" in crossgl
    assert "void graph_exec_update(ptr<f32> device_ptr)" in crossgl
    assert "var graph: hipGraph_t;" in crossgl
    assert "var exec: hipGraphExec_t;" in crossgl
    assert "var kernel_node: hipGraphNode_t;" in crossgl
    assert "var error_node: hipGraphNode_t;" in crossgl
    assert "var update_result: hipGraphExecUpdateResult;" in crossgl
    assert "var kernel_params: hipKernelNodeParams;" in crossgl
    assert "var fetched_params: hipKernelNodeParams;" in crossgl
    assert "var log: array<i8, 128>;" in crossgl
    assert "// HIP graph create: output: graph, flags: 0" in crossgl
    assert (
        "// HIP graph add kernel node: output: kernel_node, graph: graph, "
        "dependencies: NULL, count: 0, params: (&kernel_params)"
    ) in crossgl
    assert (
        "// HIP graph kernel node get params: node: kernel_node, "
        "params: (&fetched_params)"
    ) in crossgl
    assert (
        "// HIP graph kernel node set params: node: kernel_node, "
        "params: (&kernel_params)"
    ) in crossgl
    assert (
        "// HIP graph instantiate: output: exec, graph: graph, "
        "error node output: error_node, log buffer: log, log bytes: 128"
    ) in crossgl
    assert (
        "// HIP graph exec set kernel node params: exec: exec, "
        "node: kernel_node, params: (&kernel_params)"
    ) in crossgl
    assert (
        "// HIP graph exec update: exec: exec, graph: graph, "
        "error node output: error_node, result output: update_result"
    ) in crossgl
    assert "// HIP graph exec destroy: exec" in crossgl
    assert "// HIP graph destroy: graph" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_dependency_queries_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP graph dependency and query APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void graph_dependency_queries() {
        hipGraph_t graph;
        hipGraph_t clone;
        hipGraphNode_t root;
        hipGraphNode_t child;
        hipGraphNode_t cloned_child;
        hipGraphNode_t nodes[4];
        hipGraphNode_t edge_from[4];
        hipGraphNode_t edge_to[4];
        hipGraphNodeType node_type;
        size_t num_nodes = 4;
        size_t num_roots = 4;
        size_t num_edges = 4;
        size_t num_dependencies = 4;
        size_t num_dependents = 4;

        hipGraphCreate(&graph, 0);
        hipGraphAddEmptyNode(&root, graph, NULL, 0);
        hipGraphAddEmptyNode(&child, graph, NULL, 0);
        hipGraphAddDependencies(graph, &root, &child, 1);
        hipGraphGetNodes(graph, nodes, &num_nodes);
        hipGraphGetRootNodes(graph, nodes, &num_roots);
        hipGraphGetEdges(graph, edge_from, edge_to, &num_edges);
        hipGraphNodeGetDependencies(child, nodes, &num_dependencies);
        hipGraphNodeGetDependentNodes(root, nodes, &num_dependents);
        hipGraphNodeGetType(child, &node_type);
        hipGraphClone(&clone, graph);
        hipGraphNodeFindInClone(&cloned_child, child, clone);
        hipGraphRemoveDependencies(graph, &root, &child, 1);
        hipGraphDestroyNode(child);
        hipGraphDestroy(clone);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: graph_dependency_queries" in crossgl
    assert "void graph_dependency_queries()" in crossgl
    assert "var graph: hipGraph_t;" in crossgl
    assert "var clone: hipGraph_t;" in crossgl
    assert "var root: hipGraphNode_t;" in crossgl
    assert "var child: hipGraphNode_t;" in crossgl
    assert "var cloned_child: hipGraphNode_t;" in crossgl
    assert "var nodes: array<hipGraphNode_t, 4>;" in crossgl
    assert "var edge_from: array<hipGraphNode_t, 4>;" in crossgl
    assert "var edge_to: array<hipGraphNode_t, 4>;" in crossgl
    assert "var node_type: hipGraphNodeType;" in crossgl
    assert "var num_nodes: u32 = 4;" in crossgl
    assert "var num_roots: u32 = 4;" in crossgl
    assert "var num_edges: u32 = 4;" in crossgl
    assert "var num_dependencies: u32 = 4;" in crossgl
    assert "var num_dependents: u32 = 4;" in crossgl
    assert "// HIP graph create: output: graph, flags: 0" in crossgl
    assert (
        "// HIP graph add empty node: output: root, graph: graph, "
        "dependencies: NULL, count: 0"
    ) in crossgl
    assert (
        "// HIP graph add empty node: output: child, graph: graph, "
        "dependencies: NULL, count: 0"
    ) in crossgl
    assert (
        "// HIP graph add dependencies: graph: graph, from: (&root), "
        "to: (&child), count: 1"
    ) in crossgl
    assert (
        "// HIP graph get nodes: graph: graph, nodes output: nodes, "
        "count output: num_nodes"
    ) in crossgl
    assert (
        "// HIP graph get root nodes: graph: graph, nodes output: nodes, "
        "count output: num_roots"
    ) in crossgl
    assert (
        "// HIP graph get edges: graph: graph, from output: edge_from, "
        "to output: edge_to, count output: num_edges"
    ) in crossgl
    assert (
        "// HIP graph node get dependencies: node: child, nodes output: nodes, "
        "count output: num_dependencies"
    ) in crossgl
    assert (
        "// HIP graph node get dependent nodes: node: root, nodes output: nodes, "
        "count output: num_dependents"
    ) in crossgl
    assert "// HIP graph node get type: node: child, output: node_type" in crossgl
    assert "// HIP graph clone: output: clone, source: graph" in crossgl
    assert (
        "// HIP graph node find in clone: output: cloned_child, "
        "original: child, clone graph: clone"
    ) in crossgl
    assert (
        "// HIP graph remove dependencies: graph: graph, from: (&root), "
        "to: (&child), count: 1"
    ) in crossgl
    assert "// HIP graph destroy node: child" in crossgl
    assert "// HIP graph destroy: clone" in crossgl
    assert "// HIP graph destroy: graph" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_memory_nodes_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP graph memory allocation and free node APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void graph_memory_nodes() {
        hipGraph_t graph;
        hipGraphNode_t alloc_node;
        hipGraphNode_t free_node;
        hipMemAllocNodeParams alloc_params;
        hipMemAllocNodeParams fetched_alloc_params;
        void* device_ptr = NULL;

        hipGraphCreate(&graph, 0);
        hipGraphAddMemAllocNode(&alloc_node, graph, NULL, 0, &alloc_params);
        hipGraphMemAllocNodeGetParams(alloc_node, &fetched_alloc_params);
        hipGraphAddMemFreeNode(&free_node, graph, NULL, 0, device_ptr);
        hipGraphMemFreeNodeGetParams(free_node, &device_ptr);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: graph_memory_nodes" in crossgl
    assert "void graph_memory_nodes()" in crossgl
    assert "var graph: hipGraph_t;" in crossgl
    assert "var alloc_node: hipGraphNode_t;" in crossgl
    assert "var free_node: hipGraphNode_t;" in crossgl
    assert "var alloc_params: hipMemAllocNodeParams;" in crossgl
    assert "var fetched_alloc_params: hipMemAllocNodeParams;" in crossgl
    assert "var device_ptr: ptr<void> = NULL;" in crossgl
    assert "// HIP graph create: output: graph, flags: 0" in crossgl
    assert (
        "// HIP graph add memory alloc node: output: alloc_node, "
        "graph: graph, dependencies: NULL, count: 0, params: (&alloc_params)"
    ) in crossgl
    assert (
        "// HIP graph memory alloc node get params: node: alloc_node, "
        "params output: fetched_alloc_params"
    ) in crossgl
    assert (
        "// HIP graph add memory free node: output: free_node, graph: graph, "
        "dependencies: NULL, count: 0, pointer: device_ptr"
    ) in crossgl
    assert (
        "// HIP graph memory free node get params: node: free_node, "
        "pointer output: device_ptr"
    ) in crossgl
    assert "// HIP graph destroy: graph" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_memcpy_nodes_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP graph memcpy node add, query, set, and exec patch APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void graph_memcpy_nodes() {
        hipGraph_t graph;
        hipGraphExec_t exec;
        hipGraphNode_t generic_node;
        hipGraphNode_t copy_1d_node;
        hipGraphNode_t error_node;
        hipMemcpy3DParms copy_params;
        hipMemcpy3DParms fetched_copy_params;
        char log[128];
        void* dst = NULL;
        void* src = NULL;
        size_t bytes = 64;

        hipGraphCreate(&graph, 0);
        hipGraphAddMemcpyNode(&generic_node, graph, NULL, 0, &copy_params);
        hipGraphMemcpyNodeGetParams(generic_node, &fetched_copy_params);
        hipGraphMemcpyNodeSetParams(generic_node, &copy_params);
        hipGraphAddMemcpyNode1D(
            &copy_1d_node, graph, NULL, 0, dst, src, bytes,
            hipMemcpyDeviceToDevice
        );
        hipGraphMemcpyNodeSetParams1D(
            copy_1d_node, dst, src, bytes, hipMemcpyDeviceToDevice
        );
        hipGraphInstantiate(&exec, graph, &error_node, log, 128);
        hipGraphExecMemcpyNodeSetParams(exec, generic_node, &copy_params);
        hipGraphExecMemcpyNodeSetParams1D(
            exec, copy_1d_node, dst, src, bytes, hipMemcpyDeviceToDevice
        );
        hipGraphExecDestroy(exec);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: graph_memcpy_nodes" in crossgl
    assert "void graph_memcpy_nodes()" in crossgl
    assert "var graph: hipGraph_t;" in crossgl
    assert "var exec: hipGraphExec_t;" in crossgl
    assert "var generic_node: hipGraphNode_t;" in crossgl
    assert "var copy_1d_node: hipGraphNode_t;" in crossgl
    assert "var error_node: hipGraphNode_t;" in crossgl
    assert "var copy_params: hipMemcpy3DParms;" in crossgl
    assert "var fetched_copy_params: hipMemcpy3DParms;" in crossgl
    assert "var log: array<i8, 128>;" in crossgl
    assert "var dst: ptr<void> = NULL;" in crossgl
    assert "var src: ptr<void> = NULL;" in crossgl
    assert "var bytes: u32 = 64;" in crossgl
    assert "// HIP graph create: output: graph, flags: 0" in crossgl
    assert (
        "// HIP graph add memcpy node: output: generic_node, graph: graph, "
        "dependencies: NULL, count: 0, params: (&copy_params)"
    ) in crossgl
    assert (
        "// HIP graph memcpy node get params: node: generic_node, "
        "params: (&fetched_copy_params)"
    ) in crossgl
    assert (
        "// HIP graph memcpy node set params: node: generic_node, "
        "params: (&copy_params)"
    ) in crossgl
    assert (
        "// HIP graph add memcpy 1D node: output: copy_1d_node, graph: graph, "
        "dependencies: NULL, count: 0, destination: dst, source: src, "
        "bytes: bytes, kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert (
        "// HIP graph memcpy 1D node set params: node: copy_1d_node, "
        "destination: dst, source: src, bytes: bytes, kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert (
        "// HIP graph instantiate: output: exec, graph: graph, "
        "error node output: error_node, log buffer: log, log bytes: 128"
    ) in crossgl
    assert (
        "// HIP graph exec set memcpy node params: exec: exec, "
        "node: generic_node, params: (&copy_params)"
    ) in crossgl
    assert (
        "// HIP graph exec memcpy 1D node set params: exec: exec, "
        "node: copy_1d_node, destination: dst, source: src, bytes: bytes, "
        "kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert "// HIP graph exec destroy: exec" in crossgl
    assert "// HIP graph destroy: graph" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_symbol_memcpy_nodes_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP graph symbol memcpy node add, set, and exec patch APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void graph_symbol_memcpy_nodes() {
        hipGraph_t graph;
        hipGraphExec_t exec;
        hipGraphNode_t from_symbol_node;
        hipGraphNode_t to_symbol_node;
        hipGraphNode_t error_node;
        void* dst = NULL;
        const void* symbol = NULL;
        const void* src = NULL;
        size_t bytes = 64;
        size_t offset = 4;
        char log[128];

        hipGraphCreate(&graph, 0);
        hipGraphAddMemcpyNodeFromSymbol(
            &from_symbol_node, graph, NULL, 0, dst, symbol, bytes, offset,
            hipMemcpyDeviceToDevice
        );
        hipGraphMemcpyNodeSetParamsFromSymbol(
            from_symbol_node, dst, symbol, bytes, offset, hipMemcpyDeviceToDevice
        );
        hipGraphAddMemcpyNodeToSymbol(
            &to_symbol_node, graph, NULL, 0, symbol, src, bytes, offset,
            hipMemcpyDeviceToDevice
        );
        hipGraphMemcpyNodeSetParamsToSymbol(
            to_symbol_node, symbol, src, bytes, offset, hipMemcpyDeviceToDevice
        );
        hipGraphInstantiate(&exec, graph, &error_node, log, 128);
        hipGraphExecMemcpyNodeSetParamsFromSymbol(
            exec, from_symbol_node, dst, symbol, bytes, offset,
            hipMemcpyDeviceToDevice
        );
        hipGraphExecMemcpyNodeSetParamsToSymbol(
            exec, to_symbol_node, symbol, src, bytes, offset,
            hipMemcpyDeviceToDevice
        );
        hipGraphExecDestroy(exec);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: graph_symbol_memcpy_nodes" in crossgl
    assert "void graph_symbol_memcpy_nodes()" in crossgl
    assert "var graph: hipGraph_t;" in crossgl
    assert "var exec: hipGraphExec_t;" in crossgl
    assert "var from_symbol_node: hipGraphNode_t;" in crossgl
    assert "var to_symbol_node: hipGraphNode_t;" in crossgl
    assert "var error_node: hipGraphNode_t;" in crossgl
    assert "var dst: ptr<void> = NULL;" in crossgl
    assert "var symbol: ptr<void> = NULL;" in crossgl
    assert "var src: ptr<void> = NULL;" in crossgl
    assert "var bytes: u32 = 64;" in crossgl
    assert "var offset: u32 = 4;" in crossgl
    assert "var log: array<i8, 128>;" in crossgl
    assert "// HIP graph create: output: graph, flags: 0" in crossgl
    assert (
        "// HIP graph add memcpy from symbol node: output: from_symbol_node, "
        "graph: graph, dependencies: NULL, count: 0, destination: dst, "
        "source: symbol, bytes: bytes, offset: offset, "
        "kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert (
        "// HIP graph memcpy from symbol node set params: "
        "node: from_symbol_node, destination: dst, source: symbol, "
        "bytes: bytes, offset: offset, kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert (
        "// HIP graph add memcpy to symbol node: output: to_symbol_node, "
        "graph: graph, dependencies: NULL, count: 0, destination: symbol, "
        "source: src, bytes: bytes, offset: offset, kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert (
        "// HIP graph memcpy to symbol node set params: node: to_symbol_node, "
        "destination: symbol, source: src, bytes: bytes, offset: offset, "
        "kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert (
        "// HIP graph instantiate: output: exec, graph: graph, "
        "error node output: error_node, log buffer: log, log bytes: 128"
    ) in crossgl
    assert (
        "// HIP graph exec memcpy from symbol node set params: exec: exec, "
        "node: from_symbol_node, destination: dst, source: symbol, "
        "bytes: bytes, offset: offset, kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert (
        "// HIP graph exec memcpy to symbol node set params: exec: exec, "
        "node: to_symbol_node, destination: symbol, source: src, bytes: bytes, "
        "offset: offset, kind: hipMemcpyDeviceToDevice"
    ) in crossgl
    assert "// HIP graph exec destroy: exec" in crossgl
    assert "// HIP graph destroy: graph" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_graph_child_kernel_attributes_parse_and_compile_if_available(
    tmp_path,
):
    """Smoke native HIP graph child, kernel attribute, and enabled-state APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void graph_child_kernel_attributes(float* out) {
        hipGraph_t graph;
        hipGraph_t child_graph;
        hipGraph_t fetched_child_graph;
        hipGraphExec_t exec;
        hipGraphNode_t child_node;
        hipGraphNode_t source_kernel_node;
        hipGraphNode_t kernel_node;
        hipGraphNode_t error_node;
        hipKernelNodeParams source_kernel_params;
        hipKernelNodeParams kernel_params;
        hipKernelNodeAttrValue attr_value;
        unsigned int enabled = 0;
        char log[128];

        hipGraphCreate(&graph, 0);
        hipGraphCreate(&child_graph, 0);
        hipGraphAddChildGraphNode(&child_node, graph, NULL, 0, child_graph);
        hipGraphChildGraphNodeGetGraph(child_node, &fetched_child_graph);
        hipGraphAddKernelNode(
            &source_kernel_node, graph, NULL, 0, &source_kernel_params
        );
        hipGraphAddKernelNode(&kernel_node, graph, NULL, 0, &kernel_params);
        hipGraphKernelNodeCopyAttributes(kernel_node, source_kernel_node);
        hipGraphKernelNodeSetAttribute(
            kernel_node, hipKernelNodeAttributeCooperative, &attr_value
        );
        hipGraphKernelNodeGetAttribute(
            kernel_node, hipKernelNodeAttributeCooperative, &attr_value
        );
        hipGraphInstantiate(&exec, graph, &error_node, log, 128);
        hipGraphNodeSetEnabled(exec, kernel_node, 1);
        hipGraphNodeGetEnabled(exec, kernel_node, &enabled);
        hipGraphExecDestroy(exec);
        hipGraphDestroy(child_graph);
        hipGraphDestroy(graph);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: graph_child_kernel_attributes" in crossgl
    assert "void graph_child_kernel_attributes(ptr<f32> out)" in crossgl
    assert "var graph: hipGraph_t;" in crossgl
    assert "var child_graph: hipGraph_t;" in crossgl
    assert "var fetched_child_graph: hipGraph_t;" in crossgl
    assert "var exec: hipGraphExec_t;" in crossgl
    assert "var child_node: hipGraphNode_t;" in crossgl
    assert "var source_kernel_node: hipGraphNode_t;" in crossgl
    assert "var kernel_node: hipGraphNode_t;" in crossgl
    assert "var error_node: hipGraphNode_t;" in crossgl
    assert "var source_kernel_params: hipKernelNodeParams;" in crossgl
    assert "var kernel_params: hipKernelNodeParams;" in crossgl
    assert "var attr_value: hipKernelNodeAttrValue;" in crossgl
    assert "var enabled: u32 = 0;" in crossgl
    assert "var log: array<i8, 128>;" in crossgl
    assert crossgl.count("// HIP graph create: output:") == 2
    assert (
        "// HIP graph add child graph node: output: child_node, "
        "graph: graph, dependencies: NULL, count: 0, child graph: child_graph"
    ) in crossgl
    assert (
        "// HIP graph child node get graph: node: child_node, "
        "output: fetched_child_graph"
    ) in crossgl
    assert (
        "// HIP graph add kernel node: output: source_kernel_node, "
        "graph: graph, dependencies: NULL, count: 0, "
        "params: (&source_kernel_params)"
    ) in crossgl
    assert (
        "// HIP graph add kernel node: output: kernel_node, graph: graph, "
        "dependencies: NULL, count: 0, params: (&kernel_params)"
    ) in crossgl
    assert (
        "// HIP graph kernel node copy attributes: source: kernel_node, "
        "destination: source_kernel_node"
    ) in crossgl
    assert (
        "// HIP graph kernel node set attribute: node: kernel_node, "
        "attribute: hipKernelNodeAttributeCooperative, value: attr_value"
    ) in crossgl
    assert (
        "// HIP graph kernel node get attribute: node: kernel_node, "
        "attribute: hipKernelNodeAttributeCooperative, output: attr_value"
    ) in crossgl
    assert (
        "// HIP graph instantiate: output: exec, graph: graph, "
        "error node output: error_node, log buffer: log, log bytes: 128"
    ) in crossgl
    assert (
        "// HIP graph node set enabled: exec: exec, node: kernel_node, value: 1"
        in crossgl
    )
    assert (
        "// HIP graph node get enabled: exec: exec, node: kernel_node, "
        "output: enabled"
    ) in crossgl
    assert "// HIP graph exec destroy: exec" in crossgl
    assert "// HIP graph destroy: child_graph" in crossgl
    assert "// HIP graph destroy: graph" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_device_error_runtime_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP device selection and error query APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void device_error_smoke(int* count_out) {
        int device = 0;
        int count = 0;
        hipDeviceProp_t props;
        hipGetDevice(&device);
        hipGetDeviceCount(&count);
        hipSetDevice(device);
        hipGetDeviceProperties(&props, device);
        hipDeviceSynchronize();
        hipError_t last = hipGetLastError();
        hipError_t peek = hipPeekAtLastError();
        const char* name = hipGetErrorName(last);
        const char* message = hipGetErrorString(peek);
        count_out[0] = count;
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: device_error_smoke" in crossgl
    assert "void device_error_smoke(ptr<i32> count_out)" in crossgl
    assert "var device: i32 = 0;" in crossgl
    assert "var count: i32 = 0;" in crossgl
    assert "var props: hipDeviceProp_t;" in crossgl
    assert "// HIP get current device: output: device" in crossgl
    assert "// HIP get device count: output: count" in crossgl
    assert "// HIP set device: device" in crossgl
    assert "// HIP get device properties: props, device: device" in crossgl
    assert "// HIP device synchronize" in crossgl
    assert "// HIP get last error" in crossgl
    assert "var last: hipError_t = hipSuccess;" in crossgl
    assert "// HIP peek at last error" in crossgl
    assert "var peek: hipError_t = hipSuccess;" in crossgl
    assert 'var name: ptr<i8> = /* HIP error name: last */ "";' in crossgl
    assert 'var message: ptr<i8> = /* HIP error string: peek */ "";' in crossgl
    assert "count_out[0] = count;" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_host_pinned_memory_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP host-pinned allocation and registration APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void host_pinned_memory(float* registered, size_t n) {
        float* host = NULL;
        float* device = NULL;
        unsigned int flags = 0;
        hipHostMalloc((void**)&host, n * sizeof(float), hipHostMallocMapped);
        hipHostGetDevicePointer((void**)&device, host, 0);
        hipHostGetFlags(&flags, host);
        hipHostRegister(registered, n * sizeof(float), hipHostRegisterMapped);
        hipHostUnregister(registered);
        hipHostFree(host);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: host_pinned_memory" in crossgl
    assert "void host_pinned_memory(ptr<f32> registered, u32 n)" in crossgl
    assert "var host: ptr<f32> = NULL;" in crossgl
    assert "var device: ptr<f32> = NULL;" in crossgl
    assert "var flags: u32 = 0;" in crossgl
    assert (
        "// HIP host memory allocate: host, bytes: (n * sizeof(float)), "
        "flags: hipHostMallocMapped"
    ) in crossgl
    assert "// HIP host device pointer: output: device, host: host, flags: 0" in crossgl
    assert "// HIP host memory flags: output: flags, host: host" in crossgl
    assert (
        "// HIP host memory register: registered, bytes: (n * sizeof(float)), "
        "flags: hipHostRegisterMapped"
    ) in crossgl
    assert "// HIP host memory unregister: registered" in crossgl
    assert "// HIP memory free: host" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)


def test_native_hip_ipc_handle_lifecycle_parses_and_compiles_if_available(
    tmp_path,
):
    """Smoke native HIP IPC memory and event handle APIs."""
    hip_code = """
    #include <hip/hip_runtime.h>

    void ipc_handle_lifecycle(float* device_ptr) {
        hipIpcMemHandle_t mem_handle;
        hipIpcEventHandle_t event_handle;
        void* opened = NULL;
        hipEvent_t event;
        hipEvent_t opened_event;
        hipIpcGetMemHandle(&mem_handle, device_ptr);
        hipIpcOpenMemHandle(&opened, mem_handle, hipIpcMemLazyEnablePeerAccess);
        hipIpcCloseMemHandle(opened);
        hipEventCreateWithFlags(&event, hipEventInterprocess);
        hipIpcGetEventHandle(&event_handle, event);
        hipIpcOpenEventHandle(&opened_event, event_handle);
        hipEventDestroy(event);
    }
    """

    crossgl = convert_native_hip_to_crossgl(hip_code)

    assert "// Function: ipc_handle_lifecycle" in crossgl
    assert "void ipc_handle_lifecycle(ptr<f32> device_ptr)" in crossgl
    assert "var mem_handle: hipIpcMemHandle_t;" in crossgl
    assert "var event_handle: hipIpcEventHandle_t;" in crossgl
    assert "var opened: ptr<void> = NULL;" in crossgl
    assert "var event: hipEvent_t;" in crossgl
    assert "var opened_event: hipEvent_t;" in crossgl
    assert (
        "// HIP IPC get memory handle: output: mem_handle, pointer: device_ptr"
        in crossgl
    )
    assert (
        "// HIP IPC open memory handle: output: opened, handle: mem_handle, "
        "flags: hipIpcMemLazyEnablePeerAccess"
    ) in crossgl
    assert "// HIP IPC close memory handle: pointer: opened" in crossgl
    assert "// HIP event create: event, flags: hipEventInterprocess" in crossgl
    assert "// HIP IPC get event handle: output: event_handle, event: event" in crossgl
    assert (
        "// HIP IPC open event handle: output: opened_event, handle: event_handle"
        in crossgl
    )
    assert "// HIP event destroy: event" in crossgl

    compile_hip_if_hipcc_available(hip_code, tmp_path)
