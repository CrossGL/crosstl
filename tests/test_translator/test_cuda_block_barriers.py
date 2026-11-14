"""CUDA and HIP block and warp barriers order global memory, and device code has no host runtime.

``__syncthreads()`` waits for the block and makes every global and shared memory access made before it visible to
the whole block; ``__syncwarp()`` does the same for the lanes it waits for (CUDA C++ Programming Guide,
"Synchronization Functions"). CrossGL's ``workgroupBarrier`` orders workgroup memory only (GLSL ``barrier()``, SPIR-V
``OpControlBarrier`` with ``WorkgroupMemory`` semantics, HLSL ``GroupMemoryBarrierWithGroupSync``, Metal
``threadgroup_barrier(mem_flags::mem_threadgroup)``), so the importers fence all memory at workgroup scope on each
side of it. A kernel cannot allocate, set or copy memory, launch kernels or graphs on any CrossGL target, so
device-side runtime calls are refused instead of becoming comments and success values.
"""

from __future__ import annotations

import re

import pytest

import crosstl
from crosstl.backend.CUDA.CudaCrossGLCodeGen import CudaToCrossGLConverter
from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.CUDA.CudaParser import CudaParser
from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser
from crosstl.translator.codegen.barrier_peephole import merge_adjacent_cuda_barriers
from crosstl.translator.semantics.warp import CudaWarpIntrinsicUnsupportedError
from tests.lowering_refusal_helpers import refused_cuda_import, refused_hip_import

FENCED_BARRIER = re.compile(
    r"groupMemoryBarrier\(\);\s*workgroupBarrier\(\);\s*groupMemoryBarrier\(\);"
)

BLOCK_BARRIER = """
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void exchange(int *data) {
    cg::thread_block block = cg::this_thread_block();
    unsigned int i = threadIdx.x;
    data[i] = (int)i;
    __syncthreads();
    int a = data[i ^ 1u];
    block.sync();
    data[i] = a + 1;
    cg::sync(block);
    data[i] += data[i ^ 2u];
}
"""


def cuda_to_crossgl(source):
    return CudaToCrossGLConverter().generate(
        CudaParser(CudaLexer(source).tokenize()).parse()
    )


def hip_to_crossgl(source):
    return HipToCrossGLConverter().generate(
        HipParser(HipLexer(source).tokenize()).parse()
    )


def translate_cuda(tmp_path, source, backend, **options):
    path = tmp_path / "kernel.cu"
    path.write_text(source)
    return crosstl.translate(
        str(path),
        backend=backend,
        format_output=False,
        source_options={"threadgroup_size": [64, 1, 1]},
        **options,
    )


def test_block_barriers_fence_all_memory_on_each_side():
    crossgl = cuda_to_crossgl(BLOCK_BARRIER)
    assert len(FENCED_BARRIER.findall(crossgl)) == 3
    assert crossgl.count("workgroupBarrier();") == 3


@pytest.mark.parametrize(
    ("backend", "barrier", "others"),
    (
        (
            "directx",
            "AllMemoryBarrierWithGroupSync();",
            ("GroupMemoryBarrierWithGroupSync", "AllMemoryBarrier();"),
        ),
        (
            "metal",
            "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | "
            "mem_flags::mem_texture);",
            ("threadgroup_barrier(mem_flags::mem_threadgroup);",),
        ),
        ("cuda", "__syncthreads();", ("__threadfence_block",)),
        ("hip", "__syncthreads();", ("__threadfence_block",)),
    ),
)
def test_fenced_block_barriers_are_one_target_barrier(
    tmp_path, backend, barrier, others
):
    generated = translate_cuda(tmp_path, BLOCK_BARRIER, backend)
    assert generated.count(barrier) == 3
    for other in others:
        assert other not in generated


def test_glsl_block_barrier_orders_buffer_memory(tmp_path):
    generated = translate_cuda(tmp_path, BLOCK_BARRIER, "opengl")
    assert (
        len(
            re.findall(
                r"groupMemoryBarrier\(\);\s*barrier\(\);\s*groupMemoryBarrier\(\);",
                generated,
            )
        )
        == 3
    )


def test_spirv_block_barrier_fences_uniform_memory(tmp_path):
    generated = translate_cuda(tmp_path, BLOCK_BARRIER, "vulkan")
    # UniformMemory | WorkgroupMemory | ImageMemory | AcquireRelease = 0x40 | 0x100 | 0x800 | 0x8.
    semantics = {
        match.group(1): int(match.group(2))
        for match in re.finditer(r"%(\w+) = OpConstant %\w+ (\d+)", generated)
    }
    fences = re.findall(r"OpMemoryBarrier %\w+ %(\w+)", generated)
    assert len(fences) == 6
    assert {semantics[fence] for fence in fences} == {0x948}
    assert generated.count("OpControlBarrier") == 3


def test_tile_sync_of_a_warp_tile_is_syncwarp(tmp_path):
    source = """
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void tiles(int *data) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    cg::thread_block_tile<16> half = cg::tiled_partition<16>(block);
    unsigned int i = threadIdx.x;
    data[i] = (int)i;
    warp.sync();
    int a = data[i ^ 1u];
    cg::sync(half);
    data[i] = a;
}
"""
    crossgl = cuda_to_crossgl(source)
    assert crossgl.count("__syncwarp();") == 2
    assert "workgroupBarrier" not in crossgl
    generated = translate_cuda(tmp_path, source, "opengl")
    assert "__syncwarp" not in generated
    assert (
        len(
            re.findall(
                r"groupMemoryBarrier\(\);\s*barrier\(\);\s*groupMemoryBarrier\(\);",
                generated,
            )
        )
        == 2
    )


TILE_REDUCTION = """
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void reduce(const float *input, float *output, unsigned int n) {
    __shared__ float partial[256];
    cg::thread_block block = cg::this_thread_block();
    float sum = 0.0f;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        sum += input[i];
    }
    partial[block.thread_rank()] = sum;
    cg::sync(block);
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);
    for (int i = tile.size() / 2; i > 0; i >>= 1) {
        if (tile.thread_rank() < i) {
            partial[block.thread_rank()] += partial[block.thread_rank() + i];
        }
        cg::sync(tile);
    }
    cg::sync(block);
    if (block.thread_rank() == 0) {
        float total = 0.0f;
        for (int i = 0; i < 256; i += 32) {
            total += partial[i];
        }
        output[blockIdx.x] = total;
    }
}
"""


@pytest.mark.parametrize("backend", ("opengl", "directx", "vulkan", "metal"))
def test_tile_sync_after_a_divergent_loop_is_emulated(tmp_path, backend):
    # Every invocation leaves the grid-stride loop, so the tile loop after it is uniform; its i is another variable
    # than the grid-stride loop's.
    path = tmp_path / "reduce.cu"
    path.write_text(TILE_REDUCTION)
    generated = crosstl.translate(
        str(path),
        backend=backend,
        format_output=False,
        source_options={"threadgroup_size": [256, 1, 1]},
    )
    assert "__syncwarp" not in generated
    assert "sync" not in generated.replace("Sync", "")


def test_tile_sync_in_divergent_control_flow_is_refused(tmp_path):
    # Only the lanes of the tile meet at its sync; the workgroup barrier that emulates it needs every invocation.
    source = """
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void tiles(int *data) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    unsigned int i = threadIdx.x;
    if (data[i] > 0) {
        data[i] = 1;
        warp.sync();
    }
}
"""
    with pytest.raises(CudaWarpIntrinsicUnsupportedError) as raised:
        translate_cuda(tmp_path, source, "opengl")
    assert raised.value.reason_code == "non-uniform-control-flow"


@pytest.mark.parametrize("tile", ("64", "TILE"))
def test_tile_sync_beyond_a_literal_warp_tile_is_refused(tile):
    # A tile of 64 spans two warps; a size named by a constant is not proved to fit in one.
    source = f"""
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
constexpr int TILE = 8;
__global__ void tiles(int *data) {{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<{tile}> group = cg::tiled_partition<{tile}>(block);
    group.sync();
}}
"""
    ast = CudaParser(CudaLexer(source).tokenize()).parse()
    message = refused_cuda_import(CudaToCrossGLConverter().generate, ast)
    assert (
        ".sync(): CrossGL has barriers over a workgroup and over a 32-lane warp only"
        in message
    )


def test_hip_block_barriers_fence_all_memory():
    source = """
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void exchange(int *data) {
    cg::thread_block block = cg::this_thread_block();
    unsigned int i = threadIdx.x;
    data[i] = (int)i;
    __syncthreads();
    int a = data[i ^ 1u];
    block.sync();
    data[i] = a;
}
"""
    crossgl = hip_to_crossgl(source)
    assert len(FENCED_BARRIER.findall(crossgl)) == 2


@pytest.mark.parametrize(
    ("statement", "name"),
    (
        ("cudaMalloc((void**)&scratch, 16);", "cudaMalloc"),
        ("cudaMemset(data, 0, 16);", "cudaMemset"),
        ("cudaFree(data);", "cudaFree"),
        (
            "cudaError_t status = cudaMemcpyAsync(data, data, 4, cudaMemcpyDeviceToDevice);",
            "cudaMemcpyAsync",
        ),
        (
            "cudaGraphExec_t current = cudaGetCurrentGraphExec();",
            "cudaGetCurrentGraphExec",
        ),
        ("child<<<1, 32>>>(data);", "child<<<...>>>"),
    ),
)
def test_device_runtime_calls_are_refused(statement, name):
    source = f"""
__global__ void child(float *data) {{ data[threadIdx.x] = 1.0f; }}
__device__ void helper(float *data) {{
    float *scratch;
    {statement}
}}
__global__ void parent(float *data) {{ helper(data); }}
"""
    ast = CudaParser(CudaLexer(source).tokenize()).parse()
    message = refused_cuda_import(CudaToCrossGLConverter().generate, ast)
    assert name in message
    assert "in device code" in message


def test_host_runtime_calls_stay_host_descriptions():
    source = """
__global__ void k(float *out) { out[threadIdx.x] = 1.0f; }
void launch(float *host, int n) {
    float *d;
    cudaMalloc((void**)&d, n * 4);
    k<<<1, 32>>>(d);
    cudaMemcpy(host, d, n * 4, cudaMemcpyDeviceToHost);
    cudaFree(d);
}
"""
    crossgl = cuda_to_crossgl(source)
    assert "// CUDA memory allocate: d, bytes: (n * 4)" in crossgl
    assert "// Kernel launch: k<<<1, 32>>>()" in crossgl
    assert "// CUDA memory free: d" in crossgl


@pytest.mark.parametrize(
    ("statement", "name"),
    (
        ("hipMalloc((void**)&scratch, 16);", "hipMalloc"),
        ("hipError_t status = hipMemset(data, 0, 16);", "hipMemset"),
        ("const char *text = hipGetErrorString(hipSuccess);", "hipGetErrorString"),
        ("hipDeviceSynchronize();", "hipDeviceSynchronize"),
        ("child<<<1, 32>>>(data);", "child<<<...>>>"),
    ),
)
def test_hip_device_runtime_calls_are_refused(statement, name):
    source = f"""
#include <hip/hip_runtime.h>
__global__ void child(float *data) {{ data[threadIdx.x] = 1.0f; }}
__global__ void parent(float *data) {{
    float *scratch;
    {statement}
}}
"""
    ast = HipParser(HipLexer(source).tokenize()).parse()
    message = refused_hip_import(HipToCrossGLConverter().generate, ast)
    assert name in message
    assert "in device code" in message


def test_hip_host_device_synchronize_stays_a_host_description():
    source = """
#include <hip/hip_runtime.h>
__global__ void k(float *out) { out[threadIdx.x] = 1.0f; }
void launch(float *d) {
    k<<<1, 32>>>(d);
    hipDeviceSynchronize();
}
"""
    crossgl = hip_to_crossgl(source)
    assert "// HIP device synchronize" in crossgl
    assert "// Kernel launch: k<<<1, 32>>>()" in crossgl


@pytest.mark.parametrize(
    ("code", "merged"),
    (
        (
            "  __threadfence_block();\n  __syncthreads();\n  __threadfence_block();\n  x = 1;",
            "  __syncthreads();\n  x = 1;",
        ),
        (
            "  __threadfence_block();\n  __threadfence_block();\n  __syncthreads();",
            "  __syncthreads();",
        ),
        # A device-scope fence orders more than __syncthreads and ends the run.
        (
            "  __syncthreads();\n  __threadfence();\n  __threadfence_block();\n  __syncthreads();",
            "  __syncthreads();\n  __threadfence();\n  __syncthreads();",
        ),
        # Statements at another indentation belong to another block.
        (
            "  __threadfence_block();\n    __syncthreads();",
            "  __threadfence_block();\n    __syncthreads();",
        ),
        ("  __threadfence_block();\n  x = 1;", "  __threadfence_block();\n  x = 1;"),
    ),
)
def test_cuda_barrier_runs_merge_into_syncthreads(code, merged):
    assert merge_adjacent_cuda_barriers(code) == merged
