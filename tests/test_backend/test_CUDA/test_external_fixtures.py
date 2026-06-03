from crosstl.backend.CUDA.CudaAst import (
    AtomicOperationNode,
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
            "cpp/0_Introduction/simpleSurfaceWrite/simpleSurfaceWrite.cu",
            "cpp/0_Introduction/simpleTexture3D/simpleTexture3D_kernel.cu",
            "cpp/0_Introduction/simpleVoteIntrinsics/simpleVote_kernel.cuh",
            "cpp/2_Concepts_and_Techniques/scan/scan.cu",
            "cpp/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu",
            "cpp/3_CUDA_Features/cudaCompressibleMemory/saxpy.cu",
            "cpp/6_Performance/transpose/transpose.cu",
        ],
    },
    {
        "repo": "https://github.com/NVIDIA/cccl",
        "commit": "5a9ea633bfe63f113f4e99ecd505985ec2c38206",
        "paths": [
            "cudax/test/stf/examples/05-stencil.cu",
            "cub/examples/block/example_block_reduce_dyn_smem.cu",
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
