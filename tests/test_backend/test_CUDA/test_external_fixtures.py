from crosstl.backend.CUDA.CudaAst import SharedMemoryNode, StructNode, TypeAliasNode
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
            "cpp/2_Concepts_and_Techniques/scan/scan.cu",
            "cpp/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu",
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
