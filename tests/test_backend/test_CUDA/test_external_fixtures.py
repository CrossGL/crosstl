from crosstl.backend.CUDA.CudaAst import (
    AtomicOperationNode,
    FunctionCallNode,
    SharedMemoryNode,
    StructNode,
    TypeAliasNode,
)
from crosstl.backend.CUDA.CudaCrossGLCodeGen import CudaToCrossGLConverter
from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.CUDA.CudaParser import CudaParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

EXTERNAL_SAMPLES = [
    {
        "repo": "https://github.com/NVIDIA/cuda-samples",
        "commit": "b7c5481c556c3fe98db060207ecaa41a4b9a9abc",
        "paths": [
            "cpp/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics_kernel.cuh",
            "cpp/0_Introduction/mergeSort/mergeSort.cu",
            "cpp/0_Introduction/simpleSurfaceWrite/simpleSurfaceWrite.cu",
            "cpp/0_Introduction/simpleTexture3D/simpleTexture3D_kernel.cu",
            "cpp/0_Introduction/simpleVoteIntrinsics/simpleVote_kernel.cuh",
            "cpp/2_Concepts_and_Techniques/MC_SingleAsianOptionP/src/pricingengine.cu",
            "cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu",
            "cpp/2_Concepts_and_Techniques/scan/scan.cu",
            "cpp/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu",
            "cpp/3_CUDA_Features/cudaCompressibleMemory/saxpy.cu",
            "cpp/3_CUDA_Features/cdpAdvancedQuicksort/cdpAdvancedQuicksort.cu",
            "cpp/3_CUDA_Features/simpleCudaGraphs/simpleCudaGraphs.cu",
            "cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu",
            "cpp/6_Performance/transpose/transpose.cu",
        ],
    },
    {
        "repo": "https://github.com/NVIDIA/cccl",
        "commit": "5a9ea633bfe63f113f4e99ecd505985ec2c38206",
        "paths": [
            "cudax/test/stf/examples/05-stencil.cu",
            "cub/examples/block/example_block_reduce_dyn_smem.cu",
            "cub/cub/device/dispatch/kernels/kernel_transform.cuh",
            "libcudacxx/include/cuda/std/__variant/comparison.h",
            "thrust/examples/cuda/async_reduce.cu",
        ],
    },
    {
        "repo": "https://github.com/NVIDIA/cutlass",
        "commit": "2599f2975b06a67d5ee25e4a7292afeda1475c9b",
        "paths": [
            "examples/cute/tutorial/sgemm_1.cu",
            "test/unit/cute/ampere/tiled_cp_async.cu",
        ],
    },
    {
        "repo": "https://github.com/NVIDIA/CUDALibrarySamples",
        "commit": "830c6b0e5b4bf44e6d487c8505dbe42ef243b8a9",
        "paths": [
            "MathDx/cuFFTDx/02_simple_fft_block/simple_fft_block.cu",
            "cuTENSOR/reduction.cu",
        ],
    },
    {
        "repo": "https://github.com/Madreag/turbo3-cuda",
        "commit": "ae6ee21b92bc3e0fb4e6a5ab7383497861e644cc",
        "paths": ["ggml/src/ggml-cuda/fattn-vec.cuh"],
    },
    {
        "repo": "https://github.com/LLNL/RAJA",
        "commit": "2b575f125fd37fdbd6dafdd84cd6c97a025321a1",
        "paths": ["include/RAJA/policy/cuda/intrinsics.hpp"],
    },
]


def parse_cuda(source):
    return CudaParser(CudaLexer(source).tokenize()).parse()


def cuda_to_crossgl(source):
    return CudaToCrossGLConverter().generate(parse_cuda(source))


def assert_crossgl_reparse(source):
    CrossGLParser(CrossGLLexer(source).tokens).parse()


def test_external_fixture_metadata_records_repositories_and_commits():
    assert all(
        sample["repo"].startswith("https://github.com/") for sample in EXTERNAL_SAMPLES
    )
    assert all(len(sample["commit"]) == 40 for sample in EXTERNAL_SAMPLES)
    assert all(sample["paths"] for sample in EXTERNAL_SAMPLES)


def test_cuda_samples_simple_vote_intrinsics_warp_vote_codegen_reparse():
    source = """
    __global__ void VoteAnyKernel1(unsigned int *input,
                                   unsigned int *result,
                                   int size) {
        int tx = threadIdx.x;

        int mask = 0xffffffff;
        result[tx] = __any_sync(mask, input[tx]);
    }

    __global__ void VoteAllKernel2(unsigned int *input,
                                   unsigned int *result,
                                   int size) {
        int tx = threadIdx.x;

        int mask = 0xffffffff;
        result[tx] = __all_sync(mask, input[tx]);
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "var tx: i32 = gl_LocalInvocationID.x;" in crossgl
    assert "cuda warp intrinsic __any_sync(mask, input[tx])" in crossgl
    assert "cuda warp intrinsic __all_sync(mask, input[tx])" in crossgl
    assert "__any_sync(mask, input[tx]);" not in crossgl
    assert "__all_sync(mask, input[tx]);" not in crossgl


def test_cuda_samples_monte_carlo_reserved_in_parameter_codegen_reparse():
    source = """
    namespace cg = cooperative_groups;

    template <typename Real>
    __device__ Real reduce_sum(Real in, cg::thread_block cta) {
        SharedMemory<Real> sdata;
        unsigned int ltid = threadIdx.x;

        sdata[ltid] = in;
        cg::sync(cta);

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (ltid < s) {
                sdata[ltid] += sdata[ltid + s];
            }

            cg::sync(cta);
        }

        return sdata[0];
    }
    """

    ast = parse_cuda(source)
    reduce_sum = ast.functions[0]
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert reduce_sum.params[0].name == "in"
    assert "Real reduce_sum(Real in_, cooperative_groups_thread_block cta)" in crossgl
    assert "sdata[ltid] = in_;" in crossgl
    assert "Real in," not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_simple_atomic_intrinsics_codegen_reparse():
    source = """
    __global__ void testKernel(int *g_odata) {
        const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

        atomicAdd(&g_odata[0], 10);
        atomicSub(&g_odata[1], 10);
        atomicExch(&g_odata[2], tid);
        atomicMax(&g_odata[3], tid);
        atomicMin(&g_odata[4], tid);
        atomicInc((unsigned int *)&g_odata[5], 17);
        atomicDec((unsigned int *)&g_odata[6], 137);
        atomicCAS(&g_odata[7], tid - 1, tid);
        atomicAnd(&g_odata[8], 2 * tid + 7);
        atomicOr(&g_odata[9], 1 << tid);
        atomicXor(&g_odata[10], tid);
    }
    """

    ast = parse_cuda(source)
    atomic_ops = [
        stmt.operation
        for stmt in ast.kernels[0].body
        if isinstance(stmt, AtomicOperationNode)
    ]
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert atomic_ops == [
        "atomicAdd",
        "atomicSub",
        "atomicExch",
        "atomicMax",
        "atomicMin",
        "atomicInc",
        "atomicDec",
        "atomicCAS",
        "atomicAnd",
        "atomicOr",
        "atomicXor",
    ]
    assert "atomicExchange(g_odata[2], tid);" in crossgl
    assert "atomicCompareExchange(g_odata[7], (tid - 1), tid);" in crossgl
    assert "atomicOr(g_odata[9], (1 << tid));" in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_scan_cooperative_group_parameter_codegen_reparse():
    source = """
    namespace cg = cooperative_groups;

    __device__ uint scan1Inclusive(uint idata,
                                   volatile uint *s_Data,
                                   uint size,
                                   cg::thread_block cta) {
        uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
        for (uint offset = 1; offset < size; offset <<= 1) {
            cg::sync(cta);
            s_Data[pos] = s_Data[pos] + s_Data[pos - offset];
        }
        return s_Data[pos];
    }

    __global__ void scanExclusiveShared(uint4 *d_Dst, uint4 *d_Src, uint size) {
        cg::thread_block cta = cg::this_thread_block();
        __shared__ uint s_Data[2 * 256];
        uint pos = blockIdx.x * blockDim.x + threadIdx.x;
        uint4 idata4 = d_Src[pos];
        idata4.x = scan1Inclusive(idata4.x, s_Data, size, cta);
        d_Dst[pos] = idata4;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "cooperative_groups_thread_block cta" in crossgl
    assert "cg::thread_block" not in crossgl
    assert "workgroupBarrier();" in crossgl
    assert "array<vec4<u32> >" in crossgl


def test_cutlass_cute_template_array_bound_in_shared_memory_codegen_reparse():
    source = """
    __global__ void gemm() {
        __shared__ TA smemA[cosize_v<ASmemLayout>];
        smemA[threadIdx.x] = smemA[threadIdx.x];
    }
    """

    ast = parse_cuda(source)
    shared = ast.kernels[0].body[0]
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert isinstance(shared, SharedMemoryNode)
    assert shared.vtype == "TA[cosize_v<ASmemLayout>]"
    assert "array<TA, cosize_v<ASmemLayout>>" in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_transpose_parenthesized_index_and_shared_tile_codegen_reparse():
    source = """
    namespace cg = cooperative_groups;
    #define TILE_DIM 32
    #define BLOCK_ROWS 16

    __global__ void transposeNoBankConflicts(float *odata,
                                             float *idata,
                                             int width,
                                             int height) {
        cg::thread_block cta = cg::this_thread_block();
        __shared__ float tile[TILE_DIM][TILE_DIM + 1];

        int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
        int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
        int index_in = xIndex + (yIndex) * width;

        xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
        yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
        int index_out = xIndex + (yIndex) * height;

        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
            tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
        }

        cg::sync(cta);

        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
            odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
    """

    ast = parse_cuda(source)
    shared = next(
        stmt for stmt in ast.kernels[0].body if isinstance(stmt, SharedMemoryNode)
    )
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert shared.vtype == "float[32][(32 + 1)]"
    assert "var<workgroup> tile: array<array<f32, (32 + 1)>, 32>;" in crossgl
    assert "var index_in: i32 = (xIndex + (yIndex * width));" in crossgl
    assert "var index_out: i32 = (xIndex + (yIndex * height));" in crossgl
    assert "yIndex((*width))" not in crossgl
    assert "yIndex((*height))" not in crossgl
    assert "workgroupBarrier();" in crossgl
    assert_crossgl_reparse(crossgl)


def test_cufftdx_decltype_using_alias_parses_and_codegen_runs():
    source = """
    void configure_fft() {
        using FFT = decltype(Block() + Size<128>() + Type<fft_type::c2c>());
        FFT().execute();
    }
    """

    ast = parse_cuda(source)
    alias = ast.functions[0].body[0]
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert isinstance(alias, TypeAliasNode)
    assert alias.name == "FFT"
    assert alias.alias_type == "decltype(Block()+Size<128>()+Type<fft_type::c2c>())"
    assert "typedef decltype(Block()+Size<128>()+Type<fft_type::c2c>()) FFT;" in crossgl


def test_empty_for_body_from_cuda_samples_does_not_emit_none_statement():
    source = """
    uint factorRadix2(uint log2L, uint L) {
        for (log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
        return L;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert "None;" not in crossgl
    assert "for (log2L = 0; ((L & 1) == 0); L >>= 1, (log2L++)) {" in crossgl
    assert_crossgl_reparse(crossgl)


def test_cccl_cub_local_union_aligned_dynamic_shared_codegen_reparse():
    source = """
    template <int BLOCK_THREADS>
    __global__ void BlockReduceKernel(int* d_in, int* d_out) {
        using BlockReduceT = cub::BlockReduce<int, BLOCK_THREADS>;
        using TempStorageT = typename BlockReduceT::TempStorage;

        union ShmemLayout {
            TempStorageT reduce;
            int aggregate;
        };

        extern __shared__ __align__(alignof(ShmemLayout)) char smem[];
        auto& temp_storage = reinterpret_cast<TempStorageT&>(smem);
        int data = d_in[threadIdx.x];
        int aggregate = BlockReduceT(temp_storage).Sum(data);
        __syncthreads();
        int* smem_integers = reinterpret_cast<int*>(smem);
        if (threadIdx.x == 0) {
            smem_integers[0] = aggregate;
        }
        __syncthreads();
        d_out[threadIdx.x] = smem_integers[0];
    }
    """

    ast = parse_cuda(source)
    kernel_body = ast.kernels[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert isinstance(kernel_body[2], StructNode)
    assert kernel_body[2].name == "ShmemLayout"
    assert isinstance(kernel_body[3], SharedMemoryNode)
    assert kernel_body[3].vtype == "char[]"
    assert kernel_body[3].name == "smem"
    assert getattr(kernel_body[3], "is_dynamic_shared_memory", False)
    assert "struct ShmemLayout" in crossgl
    assert "var<workgroup> smem: array<i8>;" in crossgl
    assert "workgroupBarrier();" in crossgl
    assert_crossgl_reparse(crossgl)


def test_cccl_libcudacxx_templated_lambda_codegen_reparse():
    source = """
    __global__ void compare_kernel(int* out) {
        auto __three_way = []<class _Type>(
            const _Type& __v, const _Type& __w) -> int {
            return __v < __w;
        };
        out[threadIdx.x] = __three_way(1, 2);
    }
    """

    ast = parse_cuda(source)
    templated_lambda = ast.kernels[0].body[0].value
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert isinstance(templated_lambda, FunctionCallNode)
    assert templated_lambda.name == "lambda"
    assert [(arg.vtype, arg.name) for arg in templated_lambda.args[:-1]] == [
        ("const _Type &", "__v"),
        ("const _Type &", "__w"),
    ]
    assert "lambda(_Type __v, _Type __w, (__v < __w))" in crossgl
    assert "out[gl_LocalInvocationID.x] = __three_way(1, 2);" in crossgl
    assert_crossgl_reparse(crossgl)


def test_cccl_kernel_transform_trailing_return_type_codegen_reparse():
    source = """
    template <typename T>
    _CCCL_HOST_DEVICE auto make_aligned_base_ptr(
        const T* ptr,
        int alignment) -> aligned_base_ptr<T> {
        const auto raw_ptr = reinterpret_cast<const char*>(ptr);
        const auto base_ptr = ::cuda::align_down(raw_ptr, alignment);
        return aligned_base_ptr<T>{
            base_ptr,
            static_cast<int>(raw_ptr - base_ptr)};
    }
    """

    ast = parse_cuda(source)
    function = ast.functions[0]
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert function.name == "make_aligned_base_ptr"
    assert function.return_type == "aligned_base_ptr<T>"
    assert function.params[0].vtype == "const T *"
    assert function.params[1].vtype == "int"
    assert (
        "aligned_base_ptr<T> make_aligned_base_ptr(ptr<T> ptr, i32 alignment)"
        in crossgl
    )
    assert "var raw_ptr: auto = ptr<i8>(ptr);" in crossgl
    assert "return aligned_base_ptr<T>(base_ptr, i32((raw_ptr - base_ptr)));" in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_compressible_memory_grid_stride_float4_codegen_reparse():
    source = """
    __global__ void saxpy(const float a,
                          const float4 *x,
                          const float4 *y,
                          float4 *z,
                          const size_t n) {
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
             i < n;
             i += gridDim.x * blockDim.x) {
            const float4 x4 = x[i];
            const float4 y4 = y[i];
            z[i] = make_float4(a * x4.x + y4.x,
                               a * x4.y + y4.y,
                               a * x4.z + y4.z,
                               a * x4.w + y4.w);
        }
    }

    __global__ void init(float4 *x, float4 *y, const float val, const size_t n) {
        const float4 val4 = make_float4(val, val, val, val);
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
             i < n;
             i += gridDim.x * blockDim.x) {
            x[i] = y[i] = val4;
        }
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert "array<vec4<f32> >" in crossgl
    assert "gl_NumWorkGroups.x * gl_WorkGroupSize.x" in crossgl
    assert (
        "z[i] = vec4<f32>(((a * x4.x) + y4.x), ((a * x4.y) + y4.y), "
        "((a * x4.z) + y4.z), ((a * x4.w) + y4.w));"
    ) in crossgl
    assert "x[i] = y[i] = val4;" in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_simple_surface_write_texture_object_codegen_reparse():
    source = """
    __global__ void surfaceWriteKernel(float *gIData,
                                       int width,
                                       int height,
                                       cudaSurfaceObject_t outputSurface) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        surf2Dwrite(gIData[y * width + x],
                    outputSurface,
                    x * 4,
                    y,
                    cudaBoundaryModeTrap);
    }

    __global__ void transformKernel(float *gOData,
                                    int width,
                                    int height,
                                    float theta,
                                    cudaTextureObject_t tex) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        float u = x / (float)width;
        float v = y / (float)height;

        u -= 0.5f;
        v -= 0.5f;
        float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
        float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

        gOData[y * width + x] = tex2D<float>(tex, tu, tv);
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "image2D outputSurface" in crossgl
    assert (
        "imageStore(outputSurface, vec2<i32>((x * 4), y), "
        "gIData[((y * width) + x)]);"
    ) in crossgl
    assert "sampler2D tex" in crossgl
    assert "gOData[((y * width) + x)] = texture(tex, vec2<f32>(tu, tv));" in crossgl
    assert "surf2Dwrite" not in crossgl
    assert "tex2D<float>" not in crossgl


def test_cuda_samples_simple_texture3d_umul24_codegen_reparse():
    source = """
    typedef unsigned int uint;

    __global__ void d_render(uint *d_output,
                             uint imageW,
                             uint imageH,
                             float w,
                             cudaTextureObject_t texObj) {
        uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

        float u = x / (float)imageW;
        float v = y / (float)imageH;
        float voxel = tex3D<float>(texObj, u, v, w);

        if ((x < imageW) && (y < imageH)) {
            uint i = __umul24(y, imageW) + x;
            d_output[i] = voxel * 255;
        }
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "sampler3D texObj" in crossgl
    assert "texture(texObj, vec3<f32>(u, v, w))" in crossgl
    assert "(gl_WorkGroupID.x & 0x00ffffffu)" in crossgl
    assert "(gl_WorkGroupSize.x & 0x00ffffffu)" in crossgl
    assert "__umul24" not in crossgl


def test_cuda_samples_tensor_core_gemm_2d_dynamic_shared_codegen_reparse():
    source = """
    __global__ void compute_gemm(const half *A) {
        extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];
        shmem[threadIdx.y][threadIdx.x] = A[threadIdx.x];
    }
    """

    ast = parse_cuda(source)
    shared = ast.kernels[0].body[0]
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert isinstance(shared, SharedMemoryNode)
    assert shared.vtype == "half[][((CHUNK_K * K) + SKEW_HALF)]"
    assert shared.is_extern_shared_memory is True
    assert shared.is_dynamic_shared_memory is True
    assert (
        "// CUDA dynamic shared memory: shmem uses launch-time shared memory size"
        in crossgl
    )
    assert (
        "var<workgroup> shmem: " "array<array<f16, ((CHUNK_K * K) + SKEW_HALF)>>;"
    ) in crossgl
    assert_crossgl_reparse(crossgl)


def test_external_turbo3_fast_exp_softmax_codegen_reparse():
    source = """
    __global__ void softmax_update(float* kq,
                                   float* kq_max,
                                   float* kq_sum,
                                   int nthreads) {
        int tid = threadIdx.x;
        int j = blockIdx.x;
        const float kq_max_scale = __expf(kq_max[j] - kq[tid]);
        float reg = __expf(kq[(j * nthreads) + tid] - kq_max[j]);
        kq_sum[j] = kq_sum[j] * kq_max_scale + reg;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "var kq_max_scale: f32 = exp((kq_max[j] - kq[tid]));" in crossgl
    assert "var reg: f32 = exp((kq[((j * nthreads) + tid)] - kq_max[j]));" in crossgl
    assert "__expf" not in crossgl


def test_external_raja_global_qualified_cuda_shuffle_codegen_reparse():
    source = """
    __device__ int shfl_sync(int var, int srcLane) {
        return ::__shfl_sync(0xffffffffu, var, srcLane);
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "i32 shfl_sync(i32 var_, i32 srcLane)" in crossgl
    assert "return WaveReadLaneAt(var_, srcLane);" in crossgl
    assert "::__shfl_sync" not in crossgl


def test_cuda_samples_cdp_advanced_quicksort_popc_codegen_reparse():
    source = """
    __global__ void qsort_warp(unsigned *outdata,
                               unsigned int gt_mask,
                               unsigned int lt_mask,
                               unsigned int lane_mask_lt) {
        unsigned int gt_count = __popc(gt_mask);
        unsigned int lt_count = __popc(lt_mask);
        unsigned int my_mask = gt_mask | lt_mask;
        unsigned int my_offset = __popc(my_mask & lane_mask_lt);
        outdata[threadIdx.x] = gt_count + lt_count + my_offset;
    }
    """

    ast = parse_cuda(source)
    body = ast.kernels[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert isinstance(body[0].value, FunctionCallNode)
    assert body[0].value.name == "__popc"
    assert "var gt_count: u32 = bitCount(gt_mask);" in crossgl
    assert "var lt_count: u32 = bitCount(lt_mask);" in crossgl
    assert "var my_offset: u32 = bitCount((my_mask & lane_mask_lt));" in crossgl
    assert "__popc" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_sobol_qrng_ffs_codegen_reparse():
    source = """
    __global__ void sobol(unsigned int *v,
                          unsigned int *out,
                          unsigned int stride) {
        for (unsigned int k = 0; k < __ffs(stride) - 1; k++) {
            out[k] = v[k];
        }

        unsigned int v_log2stridem1 = v[__ffs(stride) - 2];
        out[threadIdx.x] = v_log2stridem1;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "(k < ((findLSB(stride) + 1) - 1))" in crossgl
    assert "v[((findLSB(stride) + 1) - 2)]" in crossgl
    assert "__ffs" not in crossgl


def test_cuda_samples_merge_sort_clz_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/0_Introduction/mergeSort/mergeSort.cu
    source = """
    template <uint W>
    __device__ uint factorRadix2(uint x) {
        return 1U << (W - __clz(x - 1));
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "return (1U << (W - countLeadingZeros((x - 1))));" in crossgl
    assert "__clz" not in crossgl


def test_cuda_samples_reduction_reduce_add_sync_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu
    source = """
    __device__ int warpReduceSumFullMask(int mySum) {
        return __reduce_add_sync(0xffffffffu, mySum);
    }

    __device__ int warpReduceSumMasked(unsigned int mask, int mySum) {
        mySum = __reduce_add_sync(mask, mySum);
        return mySum;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "return WaveActiveSum(mySum);" in crossgl
    assert (
        "/* cuda warp intrinsic __reduce_add_sync(mask, mySum) not directly "
        "supported in CrossGL */ 0"
    ) in crossgl
    assert "__reduce_add_sync(0xffffffffu, mySum);" not in crossgl
    assert "__reduce_add_sync(mask, mySum);" not in crossgl


def test_cuda_samples_simple_cuda_graphs_tiled_partition_sync_codegen_reparse():
    source = """
    namespace cg = cooperative_groups;
    #define THREADS_PER_BLOCK 256

    __global__ void reduce(float *inputVec,
                           double *outputVec,
                           size_t inputSize,
                           size_t outputSize) {
        __shared__ double tmp[THREADS_PER_BLOCK];

        cg::thread_block cta = cg::this_thread_block();
        size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

        double temp_sum = 0.0;
        for (int i = globaltid; i < inputSize; i += gridDim.x * blockDim.x) {
            temp_sum += (double)inputVec[i];
        }
        tmp[cta.thread_rank()] = temp_sum;

        cg::sync(cta);

        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

        double beta = temp_sum;
        double temp;

        for (int i = tile32.size() / 2; i > 0; i >>= 1) {
            if (tile32.thread_rank() < i) {
                temp = tmp[cta.thread_rank() + i];
                beta += temp;
                tmp[cta.thread_rank()] = beta;
            }
            cg::sync(tile32);
        }
        cg::sync(cta);

        if (cta.thread_rank() == 0 && blockIdx.x < outputSize) {
            beta = 0.0;
            for (int i = 0; i < cta.size(); i += tile32.size()) {
                beta += tmp[i];
            }
            outputVec[blockIdx.x] = beta;
        }
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert (
        "// cooperative_groups thread_block_tile<32> tile32 maps to a tiled "
        "partition of the current workgroup"
    ) in crossgl
    assert crossgl.count("workgroupBarrier();") == 3
    assert (
        "var globaltid: u32 = ((gl_WorkGroupID.x * gl_WorkGroupSize.x) + "
        "gl_LocalInvocationID.x);"
    ) in crossgl
    assert "for (var i: i32 = (32 / 2); (i > 0); i >>= 1)" in crossgl
    assert "tmp[gl_LocalInvocationIndex] = beta;" in crossgl
    assert (
        "cooperative_groups thread_block_tile.sync not directly supported"
        not in crossgl
    )
