import re
import shutil
import subprocess

from crosstl.backend.CUDA.CudaAst import (
    AtomicOperationNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    SharedMemoryNode,
    StructNode,
    TypeAliasNode,
    VariableNode,
)
from crosstl.backend.CUDA.CudaCrossGLCodeGen import CudaToCrossGLConverter
from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.CUDA.CudaParser import CudaParser
from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

EXTERNAL_SAMPLES = [
    {
        "repo": "https://github.com/NVIDIA/cuda-samples",
        "commit": "b7c5481c556c3fe98db060207ecaa41a4b9a9abc",
        "paths": [
            "cpp/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics_kernel.cuh",
            "cpp/0_Introduction/fp16ScalarProduct/fp16ScalarProduct.cu",
            "cpp/0_Introduction/mergeSort/mergeSort.cu",
            "cpp/0_Introduction/simpleSurfaceWrite/simpleSurfaceWrite.cu",
            "cpp/0_Introduction/simpleTexture3D/simpleTexture3D_kernel.cu",
            "cpp/0_Introduction/simpleVoteIntrinsics/simpleVote_kernel.cuh",
            "cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_kernel.cu",
            "cpp/2_Concepts_and_Techniques/interval/cuda_interval_rounded_arith.h",
            "cpp/2_Concepts_and_Techniques/MC_SingleAsianOptionP/src/pricingengine.cu",
            "cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu",
            "cpp/2_Concepts_and_Techniques/scan/scan.cu",
            "cpp/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu",
            "cpp/3_CUDA_Features/cudaCompressibleMemory/saxpy.cu",
            "cpp/3_CUDA_Features/bf16TensorCoreGemm/bf16TensorCoreGemm.cu",
            "cpp/3_CUDA_Features/cdpAdvancedQuicksort/cdpAdvancedQuicksort.cu",
            "cpp/3_CUDA_Features/cdpQuadtree/cdpQuadtree.cu",
            "cpp/3_CUDA_Features/simpleCudaGraphs/simpleCudaGraphs.cu",
            "cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu",
            "cpp/4_CUDA_Libraries/batchCUBLAS/batchCUBLAS.h",
            "cpp/5_Domain_Specific/BlackScholes/BlackScholes_kernel.cuh",
            "cpp/5_Domain_Specific/dxtc/dxtc.cu",
            "cpp/5_Domain_Specific/stereoDisparity/stereoDisparity_kernel.cuh",
            "cpp/6_Performance/transpose/transpose.cu",
            "cpp/9_CUDA_Tile/Benchmark_Common/matmul_benchmark.h",
        ],
    },
    {
        "repo": "https://github.com/NVIDIA/cccl",
        "commit": "5a9ea633bfe63f113f4e99ecd505985ec2c38206",
        "paths": [
            "cudax/test/stf/examples/05-stencil.cu",
            "cub/examples/block/example_block_reduce_dyn_smem.cu",
            "cub/cub/device/dispatch/kernels/kernel_transform.cuh",
            "libcudacxx/include/cuda/__bit/bit_reverse.h",
            "libcudacxx/include/cuda/std/__variant/comparison.h",
            "thrust/examples/cuda/async_reduce.cu",
        ],
    },
    {
        "repo": "https://github.com/NVIDIA/cccl",
        "commit": "cea7dcd759b81d3059db8b74a3d5d4005ce4a398",
        "paths": [
            "README.md",
            "libcudacxx/include/cuda/__barrier/barrier_block_scope.h",
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
        "repo": "https://github.com/NVIDIA/cutlass",
        "commit": "d80a4e53b52b42550659a8696dab32705265e324",
        "paths": ["include/cute/stride.hpp"],
    },
    {
        "repo": "https://github.com/NVlabs/tiny-cuda-nn",
        "commit": "749dd70c5afc5a9dadb85e5652ed65d55e0ba187",
        "paths": ["src/fully_fused_mlp.cu"],
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
        "repo": "https://github.com/NVIDIA/nvidia-hpcg",
        "commit": "7dd63cd06c0620dddd5702ad7b4fca376c19813e",
        "paths": ["src/CudaKernels.cu"],
    },
    {
        "repo": "https://github.com/rapidsai/cudf",
        "commit": "d387ee637326739a00bb4825eebb2ad0c66bdd01",
        "paths": ["cpp/include/cudf/hashing/detail/hash_functions.cuh"],
    },
    {
        "repo": "https://github.com/Madreag/turbo3-cuda",
        "commit": "ae6ee21b92bc3e0fb4e6a5ab7383497861e644cc",
        "paths": ["ggml/src/ggml-cuda/fattn-vec.cuh"],
    },
    {
        "repo": "https://github.com/ggml-org/llama.cpp",
        "commit": "7c158fbb4aec1bdc9c81d6ca0e785139f4826fae",
        "paths": ["ggml/src/ggml-cuda/common.cuh"],
    },
    {
        "repo": "https://github.com/LLNL/RAJA",
        "commit": "2b575f125fd37fdbd6dafdd84cd6c97a025321a1",
        "paths": ["include/RAJA/policy/cuda/intrinsics.hpp"],
    },
    {
        "repo": "https://github.com/cupy/cupy",
        "commit": "ba594a4aebbfda022ba20575a161b88d9d54665a",
        "paths": ["cupy/_core/include/cupy/carray.cuh"],
    },
]


def parse_cuda(source):
    return CudaParser(CudaLexer(source).tokenize()).parse()


def cuda_to_crossgl(source):
    return CudaToCrossGLConverter().generate(parse_cuda(source))


def assert_crossgl_reparse(source):
    CrossGLParser(CrossGLLexer(source).tokens).parse()


def crossgl_to_cuda(source):
    return CudaCodeGen().generate(CrossGLParser(CrossGLLexer(source).tokens).parse())


def compile_cuda_if_nvcc_available(cuda_code, tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return False

    source_path = tmp_path / "roundtrip.cu"
    object_path = tmp_path / "roundtrip.o"
    source_path.write_text(cuda_code, encoding="utf-8")

    result = subprocess.run(
        [nvcc, "-std=c++17", "-c", str(source_path), "-o", str(object_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + "\n\n" + cuda_code
    return True


def test_external_fixture_metadata_records_repositories_and_commits():
    assert all(
        sample["repo"].startswith("https://github.com/") for sample in EXTERNAL_SAMPLES
    )
    assert all(len(sample["commit"]) == 40 for sample in EXTERNAL_SAMPLES)
    assert all(sample["paths"] for sample in EXTERNAL_SAMPLES)


def test_cccl_annotated_ptr_explicit_constexpr_constructor_is_skipped():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cccl
    # commit: 5a9ea633bfe63f113f4e99ecd505985ec2c38206
    # path: libcudacxx/include/cuda/__annotated_ptr/annotated_ptr.h
    source = """
    template <typename _Tp, typename _Property>
    class annotated_ptr : private ::cuda::__annotated_ptr_base<_Property>
    {
    private:
      pointer __repr = nullptr;

    public:
      _CCCL_HOST_DEVICE_API explicit constexpr annotated_ptr(pointer __p) noexcept
          : __repr{__p}
      {
        if constexpr (__is_smem) { return; }
      }
    };
    """

    ast = parse_cuda(source)
    crossgl = cuda_to_crossgl(source)

    assert ast.structs[0].name == "annotated_ptr"
    assert [(member.vtype, member.name) for member in ast.structs[0].members] == [
        ("pointer", "__repr")
    ]
    assert "struct annotated_ptr" in crossgl
    assert "pointer __repr;" in crossgl
    assert "explicit" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_interval_template_specializations_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/2_Concepts_and_Techniques/interval/cuda_interval_rounded_arith.h
    source = """
    template <class T> struct rounded_arith {};
    template <> struct rounded_arith<float> {};
    template <> struct rounded_arith<double> {};

    __device__ void use_specializations(rounded_arith<float>* f,
                                        rounded_arith<double>* d) {
        return;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert "struct rounded_arith_f32 {" in crossgl
    assert "struct rounded_arith_f64 {" in crossgl
    assert "ptr<rounded_arith_f32> f" in crossgl
    assert "ptr<rounded_arith_f64> d" in crossgl
    assert "rounded_arith<float>" not in crossgl
    assert "rounded_arith<double>" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_native_saxpy_round_trip_regenerates_native_cuda(tmp_path):
    source = """
    __global__ void saxpy(float* y, const float* x, float a, unsigned int n) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            y[idx] = a * x[idx] + y[idx];
        }
    }
    """

    crossgl = cuda_to_crossgl(source)
    regenerated_cuda = crossgl_to_cuda(crossgl)

    assert "f32 a" in crossgl
    assert "u32 n" in crossgl
    assert "gl_WorkGroupID.x" in crossgl
    assert "gl_WorkGroupSize.x" in crossgl
    assert "gl_LocalInvocationID.x" in crossgl

    assert 'extern "C" __global__ void saxpy(' in regenerated_cuda
    assert re.search(r"\bfloat\s+a\b", regenerated_cuda)
    assert re.search(r"\bunsigned\s+int\s+n\b", regenerated_cuda)
    assert re.search(r"\bunsigned\s+int\s+idx\b", regenerated_cuda)
    assert "threadIdx.x" in regenerated_cuda
    assert "blockIdx.x" in regenerated_cuda
    assert "blockDim.x" in regenerated_cuda
    assert "gl_GlobalInvocationID" not in regenerated_cuda
    assert "gl_WorkGroupID" not in regenerated_cuda
    assert "gl_WorkGroupSize" not in regenerated_cuda
    assert "gl_LocalInvocationID" not in regenerated_cuda
    assert not re.search(r"\bf32\b", regenerated_cuda)
    assert not re.search(r"\bu32\b", regenerated_cuda)

    compile_cuda_if_nvcc_available(regenerated_cuda, tmp_path)


def test_cupy_carray_post_return_device_struct_methods_are_skipped():
    # Upstream source:
    # repo: https://github.com/cupy/cupy
    # commit: ba594a4aebbfda022ba20575a161b88d9d54665a
    # path: cupy/_core/include/cupy/carray.cuh
    source = """
    class CIndexer {
      static unsigned int __device__ _log2(unsigned int x) {
        return __popc(x - 1);
      }
      static unsigned long long int __device__ _log2(unsigned long long int x) {
        return __popcll(x - 1);
      }
    };
    """

    ast = parse_cuda(source)
    crossgl = cuda_to_crossgl(source)

    assert ast.structs[0].name == "CIndexer"
    assert ast.structs[0].members == []
    assert "struct CIndexer" in crossgl
    assert "_log2" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_eglstream_elaborated_struct_types_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/eglstrm_common.h
    source = """
    static double getMicrosecond(struct timespec t) {
        return (t.tv_sec * 1000000.0);
    }

    static inline void getTime(struct timespec *t) {
        clock_gettime(clock_id, t);
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert "f64 getMicrosecond(timespec t)" in crossgl
    assert "void getTime(ptr<timespec> t)" in crossgl
    assert "struct timespec" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_simple_ipc_platform_error_guard_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/0_Introduction/simpleIPC/simpleIPC.cu
    source = """
    #if defined(__linux__)
    #define cpu_atomic_add32(a, x) __sync_add_and_fetch(a, x)
    #elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #define cpu_atomic_add32(a, x) InterlockedAdd((volatile LONG *)a, x)
    #else
    #error Unsupported system
    #endif

    static void barrierWait(volatile int *barrier, unsigned int n) {
        int count;
        count = cpu_atomic_add32(barrier, 1);
        if (count == n) {
            return;
        }
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert "__sync_add_and_fetch(barrier, 1)" in crossgl
    assert "InterlockedAdd" not in crossgl
    assert "Unsupported system" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_line_of_sight_anonymous_enum_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/4_CUDA_Libraries/lineOfSight/lineOfSight.cu
    source = """
    typedef unsigned char Bool;
    enum { False = 0, True = 1 };

    __global__ void computeVisibilities_kernel(Bool *visibilities) {
        visibilities[threadIdx.x] = True;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert "enum anonymous_enum_0 {" in crossgl
    assert "False = 0," in crossgl
    assert "True = 1," in crossgl
    assert "enum  {" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_nested_scoped_template_return_type_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/2_Concepts_and_Techniques/streamOrderedAllocationP2P/streamOrderedAllocationP2P.cu
    source = """
    std::multimap<std::pair<int, int>, int> getIdenticalGPUs() {
        return identicalGpus;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert "std_multimap<std_pair<i32, i32>, i32> getIdenticalGPUs()" in crossgl
    assert "std::pair" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_interval_buffer_parameter_keyword_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/2_Concepts_and_Techniques/interval/cuda_interval.h
    source = """
    template <typename T>
    __global__ void bisect(T* buffer, int index) {
        buffer[threadIdx.x] = index;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert "buffer_: array<T>" in crossgl
    assert "buffer_[gl_LocalInvocationID.x] = index;" in crossgl
    assert " buffer:" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_nvrtc_char_escape_literals_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_nvrtc.h
    source = r"""
    void clear(char* memBlock, unsigned int inputSize) {
        memBlock[inputSize] = '\x0';
        memBlock[0] = '\101';
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert "memBlock[inputSize] = 0;" in crossgl
    assert "memBlock[0] = 65;" in crossgl
    assert "'\\x0'" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_batch_cublas_unsigned_inline_union_and_comma_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/4_CUDA_Libraries/batchCUBLAS/batchCUBLAS.h
    source = """
    static __inline__ unsigned floatAsUInt(float x) {
        volatile union {
            float f;
            unsigned i;
        } xx;
        xx.f = x;
        return xx.i;
    }

    static __inline__ unsigned cuRand(void) {
        static unsigned int cuda_jsr = 123456789;
        return (cuda_jsr = cuda_jsr ^ (cuda_jsr << 17),
                cuda_jsr = cuda_jsr ^ (cuda_jsr >> 13),
                cuda_jsr = cuda_jsr ^ (cuda_jsr << 5));
    }
    """

    ast = parse_cuda(source)
    crossgl = cuda_to_crossgl(source)

    assert ast.functions[0].return_type == "unsigned int"
    assert ast.functions[0].name == "floatAsUInt"
    assert isinstance(ast.functions[0].body[0], StructNode)
    assert ast.functions[0].body[0].name == "anonymous_xx_layout"
    assert isinstance(ast.functions[0].body[1], VariableNode)
    assert ast.functions[0].body[1].vtype == "anonymous_xx_layout"

    assert "u32 floatAsUInt(f32 x)" in crossgl
    assert "struct anonymous_xx_layout" in crossgl
    assert "var xx: anonymous_xx_layout;" in crossgl
    assert "CUDA comma expression" in crossgl
    assert_crossgl_reparse(crossgl)


def test_cutlass_static_member_value_expression_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cutlass
    # commit: d80a4e53b52b42550659a8696dab32705265e324
    # path: include/cute/numeric/integral_constant.hpp
    source = """
    template <class T>
    constexpr bool is_integral_v = is_integral<T>::value;
    """

    crossgl = cuda_to_crossgl(source)

    assert "var is_integral_v: bool = is_integral_T_value;" in crossgl
    assert "::value" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cutlass_empty_initializer_list_return_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cutlass
    # commit: d80a4e53b52b42550659a8696dab32705265e324
    # path: include/cute/numeric/integral_constant.hpp
    source = """
    template <auto v> struct C { static constexpr auto value = v; };

    template <auto t>
    C<(+t)> operator+(C<t>) {
        return {};
    }

    int zero() {
        return {0};
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert "C_t operator_add(C<t> _unused_param_0)" in crossgl
    assert "return 0 /* CUDA initializer list return */;" in crossgl
    assert "i32 zero()" in crossgl
    assert "return {};" not in crossgl
    assert "return {0};" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_tiny_cuda_nn_unroll_macro_markers_before_for_loops_parse_and_codegen():
    # Upstream source:
    # repo: https://github.com/NVlabs/tiny-cuda-nn
    # commit: 749dd70c5afc5a9dadb85e5652ed65d55e0ba187
    # path: src/fully_fused_mlp.cu
    source = """
    template <uint32_t N_ITERS>
    __device__ void threadblock_load_input_static(float* out,
                                                  const float* input) {
        __syncthreads();

        TCNN_PRAGMA_UNROLL
        for (uint32_t i = 0; i < N_ITERS; ++i) {
            out[i] = input[i];
        }

        _Pragma("unroll")
        for (uint32_t j = 0; j < N_ITERS; ++j) {
            out[j] = out[j];
        }
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    loops = [stmt for stmt in body if isinstance(stmt, ForNode)]
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert len(loops) == 2
    assert body[0].sync_type == "__syncthreads"
    assert "TCNN_PRAGMA_UNROLL" not in crossgl
    assert "_Pragma" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_tiny_cuda_nn_enable_if_return_type_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVlabs/tiny-cuda-nn
    # commit: 749dd70c5afc5a9dadb85e5652ed65d55e0ba187
    # path: src/fully_fused_mlp.cu
    source = """
    template <typename T>
    std::enable_if_t<!std::is_same<__half, T>::value>
    mlp_fused_backward(cudaStream_t stream) {
        return;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert "void mlp_fused_backward(cudaStream_t stream)" in crossgl
    assert "std::enable_if_t" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cufftdx_scoped_storage_types_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/CUDALibrarySamples
    # commit: 830c6b0e5b4bf44e6d487c8505dbe42ef243b8a9
    # path: MathDx/cuFFTDx/02_simple_fft_block/simple_fft_block.cu
    source = """
    __global__ void block_fft_kernel(FFT::value_type* data) {
        FFT::value_type thread_data[FFT::storage_size];
        data[threadIdx.x] = thread_data[0];
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert "array<FFT_value_type>" in crossgl
    assert "array<FFT::value_type, FFT_storage_size>" in crossgl
    assert "data: array<FFT::value_type>" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_dependent_scoped_and_enum_types_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # paths:
    # - cpp/5_Domain_Specific/nbody/bodysystemcuda.cu
    # - cpp/5_Domain_Specific/SobelFilter/SobelFilter_kernels.cu
    source = """
    enum SobelDisplayMode { SOBELDISPLAY_IMAGE };

    template <typename T>
    __device__ typename vec3<T>::Type bodyBodyInteraction(
        typename vec3<T>::Type ai,
        enum SobelDisplayMode mode) {
        typename vec3<T>::Type r = ai;
        return r;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert (
        "vec3_T_Type bodyBodyInteraction(vec3_T_Type ai, SobelDisplayMode mode)"
        in crossgl
    )
    assert "var r: vec3_T_Type = ai;" in crossgl
    assert "vec3<T>::Type" not in crossgl
    assert "enum SobelDisplayMode mode" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_tiny_cuda_nn_wmma_type_positions_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVlabs/tiny-cuda-nn
    # commit: 749dd70c5afc5a9dadb85e5652ed65d55e0ba187
    # path: src/fully_fused_mlp.cu
    source = """
    template <uint32_t WIDTH>
    __device__ void threadblock_last_layer_forward(
        nvcuda::wmma::layout_t output_layout) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> result_frag[2];
        if (output_layout == wmma::mem_row_major) {
            return;
        }
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert "nvcuda_wmma_layout_t output_layout" in crossgl
    assert "var act_frag: wmma_fragment_wmma_matrix_a_" in crossgl
    assert "var result_frag: wmma_fragment_wmma_accumulator_" in crossgl
    assert "nvcuda::wmma::layout_t" not in crossgl
    assert "wmma::fragment" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_programming_guide_syncthreads_predicate_variants_codegen_reparse():
    # Upstream source:
    # NVIDIA CUDA Programming Guide v13.3, section 5.4.4.1
    # Thread Block Synchronization Functions.
    source = """
    __global__ void vote_block(int *out, int limit) {
        int pred = threadIdx.x < limit;
        int count = __syncthreads_count(pred);
        int all_set = __syncthreads_and(pred);
        int any_set = __syncthreads_or(pred);
        out[threadIdx.x] = count + all_set + any_set;
    }
    """

    ast = parse_cuda(source)
    body = ast.kernels[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert isinstance(body[1].value, FunctionCallNode)
    assert body[1].value.name == "__syncthreads_count"
    assert body[2].value.name == "__syncthreads_and"
    assert body[3].value.name == "__syncthreads_or"
    assert (
        "var count: i32 = (/* cuda thread block sync vote "
        "__syncthreads_count(pred) not directly supported in CrossGL */ 0);"
    ) in crossgl
    assert (
        "var all_set: i32 = (/* cuda thread block sync vote "
        "__syncthreads_and(pred) not directly supported in CrossGL */ 0);"
    ) in crossgl
    assert (
        "var any_set: i32 = (/* cuda thread block sync vote "
        "__syncthreads_or(pred) not directly supported in CrossGL */ 0);"
    ) in crossgl
    assert "__syncthreads_count(pred);" not in crossgl
    assert "__syncthreads_and(pred);" not in crossgl
    assert "__syncthreads_or(pred);" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_programming_guide_one_component_vectors_codegen_reparse():
    # Upstream source:
    # NVIDIA CUDA Programming Guide v12.4, section 7.3.1 Built-in Vector Types.
    source = """
    struct Pair {
        float x;
    };

    float unpack(Pair pair, float1 packed) {
        float1 local = make_float1(2.0f);
        int1 count(3);
        return pair.x + packed.x + local.x + count.x;
    }
    """

    ast = parse_cuda(source)
    function = ast.functions[0]
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert function.params[1].vtype == "float1"
    assert isinstance(function.body[0], VariableNode)
    assert function.body[0].vtype == "float1"
    assert isinstance(function.body[0].value, FunctionCallNode)
    assert function.body[0].value.name == "make_float1"
    assert "f32 unpack(Pair pair, f32 packed)" in crossgl
    assert "var local: f32 = f32(2.0f);" in crossgl
    assert "var count: i32 = i32(3);" in crossgl
    assert "pair.x" in crossgl
    assert "packed.x" not in crossgl
    assert "local.x" not in crossgl
    assert "count.x" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_simple_vote_intrinsics_warp_vote_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/0_Introduction/simpleVoteIntrinsics/simpleVote_kernel.cuh
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
    assert "result[tx] = (WaveActiveAnyTrue((input[tx] != 0)) ? 1 : 0);" in crossgl
    assert "result[tx] = (WaveActiveAllTrue((input[tx] != 0)) ? 1 : 0);" in crossgl
    assert "cuda warp intrinsic __any_sync(mask, input[tx])" not in crossgl
    assert "cuda warp intrinsic __all_sync(mask, input[tx])" not in crossgl
    assert "__any_sync(mask, input[tx]);" not in crossgl
    assert "__all_sync(mask, input[tx]);" not in crossgl


def test_cuda_samples_fp16_scalar_product_high2float_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/0_Introduction/fp16ScalarProduct/fp16ScalarProduct.cu
    source = """
    __global__ void scalarProductKernel_intrinsics(half2 const *const a,
                                                   half2 const *const b,
                                                   float *const results) {
        half2 result = a[threadIdx.x];
        float f_result = __low2float(result) + __high2float(result);
        results[blockIdx.x] = f_result;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "var f_result: f32 = (f32(result.x) + f32(result.y));" in crossgl
    assert "__high2float" not in crossgl


def test_cuda_math_api_floats2half2_rn_codegen_reparse():
    # Source inspiration:
    # NVIDIA CUDA Math API, Half Precision Conversion and Data Movement.
    # cuda-samples fp16ScalarProduct covers half2 lane extraction; the CUDA
    # Math API documents the related two-float half2 constructor.
    source = """
    __global__ void pack_half2(float *a, float *b, half2 *out) {
        int idx = threadIdx.x;
        out[idx] = __floats2half2_rn(a[idx], b[idx]);
    }
    """

    ast = parse_cuda(source)
    assignment = ast.kernels[0].body[1]
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert assignment.right.name == "__floats2half2_rn"
    assert "out[idx] = vec2<f16>(a[idx], b[idx]);" in crossgl
    assert "__floats2half2_rn" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_math_api_half2_float2_vector_conversions_codegen_reparse():
    # Source inspiration:
    # NVIDIA CUDA Math API v13.3, Half Precision Conversion and Data Movement.
    # URL:
    # https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html
    source = """
    __device__ float2 convert_half2_vectors(float2 input, half2 packed) {
        half2 converted = __float22half2_rn(input);
        float2 unpacked = ::__half22float2(packed);
        return make_float2(unpacked.x + input.x, unpacked.y + input.y);
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert body[0].value.name == "__float22half2_rn"
    assert body[1].value.name == "::__half22float2"
    assert "var converted: vec2<f16> = vec2<f16>(input.x, input.y);" in crossgl
    assert "var unpacked: vec2<f32> = vec2<f32>(packed.x, packed.y);" in crossgl
    assert (
        "return vec2<f32>((unpacked.x + input.x), (unpacked.y + input.y));" in crossgl
    )
    assert "__float22half2_rn" not in crossgl
    assert "__half22float2" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_tile_matmul_half2float_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/9_CUDA_Tile/Benchmark_Common/matmul_benchmark.h
    source = """
    void matmul_cpu(float* C,
                    const __half* A,
                    const __half* B,
                    int i,
                    int j,
                    int k,
                    int K,
                    int N) {
        float sum = 0.0f;
        sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
        C[i * N + j] = sum;
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert body[1].right.left.name == "__half2float"
    assert body[1].right.right.name == "__half2float"
    assert "sum += (f32(A[((i * K) + k)]) * f32(B[((k * N) + j)]));" in crossgl
    assert "__half2float" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_tile_matmul_float2half_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/9_CUDA_Tile/tileMatmul/tileMatmul.cu
    # Semantics source:
    # NVIDIA CUDA Math API, Half Precision Conversion and Data Movement.
    source = """
    void init_half_inputs(half* h_A, float value) {
        h_A[0] = __float2half(value - 0.5f);
        h_A[1] = ::__float2half_rn(value + 0.5f);
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert body[0].right.name == "__float2half"
    assert body[1].right.name == "::__float2half_rn"
    assert "h_A[0] = f16((value - 0.5f));" in crossgl
    assert "h_A[1] = f16((value + 0.5f));" in crossgl
    assert "__float2half" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_interval_bit_reinterpret_intrinsics_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/2_Concepts_and_Techniques/interval/cuda_interval_rounded_arith.h
    # Semantics source:
    # NVIDIA CUDA Math API v13.3, section 11 Type Casting Intrinsics.
    source = """
    __device__ float neg_inf() {
        return __int_as_float(0xff800000);
    }

    __device__ float pos_inf() {
        return ::__int_as_float(0x7f800000);
    }

    __device__ void roundtrip_bits(float value,
                                   int signedBits,
                                   unsigned int unsignedBits,
                                   int *outSigned,
                                   unsigned int *outUnsigned,
                                   float *outFloat) {
        outSigned[0] = __float_as_int(value);
        outUnsigned[0] = __float_as_uint(value);
        outFloat[0] = __int_as_float(signedBits);
        outFloat[1] = __uint_as_float(unsignedBits);
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "return intBitsToFloat(0xff800000);" in crossgl
    assert "return intBitsToFloat(0x7f800000);" in crossgl
    assert "outSigned[0] = floatBitsToInt(value);" in crossgl
    assert "outUnsigned[0] = floatBitsToUint(value);" in crossgl
    assert "outFloat[0] = intBitsToFloat(signedBits);" in crossgl
    assert "outFloat[1] = uintBitsToFloat(unsignedBits);" in crossgl
    assert "__int_as_float" not in crossgl
    assert "__uint_as_float" not in crossgl
    assert "__float_as_int" not in crossgl
    assert "__float_as_uint" not in crossgl


def test_cupy_double_longlong_bit_reinterpret_intrinsics_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/cupy/cupy
    # commit: 554b3835f905ceae3dd1335f42a16563510b9812
    # path: cupy/_core/include/cupy/atomics.cuh
    source = """
    __device__ double cupy_double_bits(double val,
                                       unsigned long long assumed) {
        unsigned long long bits =
            __double_as_longlong(val + __longlong_as_double(assumed));
        return __longlong_as_double(bits);
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert body[0].value.name == "__double_as_longlong"
    assert body[0].value.args[0].right.name == "__longlong_as_double"
    assert body[1].value.name == "__longlong_as_double"
    assert (
        "var bits: u64 = doubleBitsToLong((val + longBitsToDouble(assumed)));"
        in crossgl
    )
    assert "return longBitsToDouble(bits);" in crossgl
    assert "__double_as_longlong" not in crossgl
    assert "__longlong_as_double" not in crossgl
    assert_crossgl_reparse(crossgl)


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


def test_cuda_samples_dxtc_thread_group_parameter_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/5_Domain_Specific/dxtc/dxtc.cu
    source = """
    namespace cg = cooperative_groups;

    __device__ void sortColors(const float *values,
                               int *ranks,
                               cg::thread_group tile) {
        cg::sync(tile);
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert "cooperative_groups_thread_group tile" in crossgl
    assert "cg::thread_group" not in crossgl
    assert "cooperative_groups thread_group.sync not directly supported" in crossgl
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


def test_cuda_programming_guide_scoped_atomics_codegen_reparse():
    # Upstream source:
    # NVIDIA CUDA Programming Guide, atomic functions section. CUDA samples
    # simpleAtomicIntrinsics covers the unscoped atomic family; the programming
    # guide documents _block and _system scoped variants.
    source = """
    __global__ void scopedAtomicKernel(int *g_odata, unsigned int *flags) {
        int old = atomicAdd_system(&g_odata[0], 10);
        atomicMax_block(&g_odata[threadIdx.x], old);
        unsigned int exchanged = atomicCAS_system(flags, 0u, 1u);
        flags[threadIdx.x] = exchanged;
    }
    """

    ast = parse_cuda(source)
    body = ast.kernels[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert isinstance(body[0].value, AtomicOperationNode)
    assert body[0].value.operation == "atomicAdd_system"
    assert isinstance(body[1], AtomicOperationNode)
    assert body[1].operation == "atomicMax_block"
    assert isinstance(body[2].value, AtomicOperationNode)
    assert body[2].value.operation == "atomicCAS_system"
    assert "cuda system-scope atomic atomicAdd_system lowered to atomicAdd" in crossgl
    assert "cuda block-scope atomic atomicMax_block lowered to atomicMax" in crossgl
    assert (
        "cuda system-scope atomic atomicCAS_system lowered to " "atomicCompareExchange"
    ) in crossgl
    assert "atomicAdd(g_odata[0], 10)" in crossgl
    assert "atomicMax(g_odata[gl_LocalInvocationID.x], old)" in crossgl
    assert "atomicCompareExchange(flags, 0u, 1u)" in crossgl
    assert "atomicAdd_system(" not in crossgl
    assert "atomicMax_block(" not in crossgl
    assert "atomicCAS_system(" not in crossgl
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
    assert "array<TA, cosize_v_ASmemLayout>" in crossgl
    assert_crossgl_reparse(crossgl)


def test_cutlass_cute_dependent_template_type_alias_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cutlass
    # commit: d80a4e53b52b42550659a8696dab32705265e324
    # path: include/cute/stride.hpp
    # line: 299
    source = """
    template <class Shape>
    __device__ void compact_like() {
        using Lambda = CompactLambda<Major>;
        using Seq = typename Lambda::template seq<Shape>;
        Seq order;
    }
    """

    ast = parse_cuda(source)
    aliases = ast.functions[0].body[:2]
    order = ast.functions[0].body[2]
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert all(isinstance(alias, TypeAliasNode) for alias in aliases)
    assert aliases[1].alias_type == "typename Lambda::seq<Shape>"
    assert isinstance(order, VariableNode)
    assert order.vtype == "Seq"
    assert "typedef Lambda::seq<Shape> Seq;" in crossgl
    assert "::template" not in crossgl
    assert "typename Lambda::seq" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cutlass_named_alignas_struct_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cutlass
    # commit: 2599f2975b06a67d5ee25e4a7292afeda1475c9b
    # path: include/cutlass/half.h
    source = """
    struct alignas(2) half_t {
        unsigned short storage;
    };

    __global__ void write_half_storage(half_t *out) {
        out[threadIdx.x].storage = 0;
    }
    """

    ast = parse_cuda(source)
    half_t = ast.structs[0]
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert isinstance(half_t, StructNode)
    assert half_t.name == "half_t"
    assert half_t.attributes == ["alignas(2)"]
    assert ast.global_variables == []
    assert "struct half_t {" in crossgl
    assert "u16 storage;" in crossgl
    assert "out[gl_LocalInvocationID.x].storage = 0;" in crossgl
    assert "alignas" not in crossgl
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
    assert (
        "var<workgroup> tile: " "array<array<f32, cuda_array_extent_32_plus_1>, 32>;"
    ) in crossgl
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


def test_cccl_cub_shared_union_temp_storage_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cccl
    # commit: 6383b911c29ba6e8dc24ae3d33694fb2c54fde00
    # path: cub/examples/block/example_block_scan.cu
    source = """
    using BlockLoad = cub::BlockLoad<int, 128, 4>;
    using BlockScan = cub::BlockScan<int, 128>;

    __global__ void scan_kernel(int* d_in, int* d_out) {
        __shared__ union TempStorage {
            typename BlockLoad::TempStorage load;
            typename BlockScan::TempStorage scan;
            int aggregate;
        } temp_storage;

        int data = d_in[threadIdx.x];
        BlockLoad(temp_storage.load).Load(d_in, data);
        int aggregate = BlockScan(temp_storage.scan).Sum(data);
        temp_storage.aggregate = aggregate;
        d_out[threadIdx.x] = temp_storage.aggregate;
    }
    """

    ast = parse_cuda(source)
    body = ast.kernels[0].body
    crossgl = cuda_to_crossgl(source)

    assert isinstance(body[0], StructNode)
    assert body[0].name == "TempStorage"
    assert [member.name for member in body[0].members] == ["load", "scan", "aggregate"]
    assert isinstance(body[1], SharedMemoryNode)
    assert body[1].vtype == "TempStorage"
    assert body[1].name == "temp_storage"
    assert "struct TempStorage {" in crossgl
    assert "var<workgroup> temp_storage: TempStorage;" in crossgl
    assert "BlockScan(temp_storage.scan).Sum(data)" in crossgl
    assert_crossgl_reparse(crossgl)


def test_cccl_cub_parenthesized_std_max_call_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cccl
    # commit: 5a9ea633bfe63f113f4e99ecd505985ec2c38206
    # path: cub/examples/block/example_block_reduce_dyn_smem.cu
    source = """
    void configure(int block_reduce_temp_bytes, int n) {
        auto smem_size =
            (std::max)(1 * sizeof(int), block_reduce_temp_bytes);
        std::size_t count = (std::size_t)(n);
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert isinstance(body[0].value, FunctionCallNode)
    assert body[0].value.name == "std::max"
    assert body[0].value.args[1] == "block_reduce_temp_bytes"
    assert body[1].value.target_type == "std::size_t"
    expected_call = "std::max((1 * sizeof(int)), block_reduce_temp_bytes);"
    assert f"var smem_size: auto = {expected_call}" in crossgl
    assert "var count: std::size_t = std::size_t(n);" in crossgl
    assert_crossgl_reparse(crossgl)


def test_cccl_readme_span_reduce_kernel_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cccl
    # commit: cea7dcd759b81d3059db8b74a3d5d4005ce4a398
    # path: README.md
    # lines: 274-289
    source = """
    template <int block_size>
    __global__ void reduce(cuda::std::span<int const> data,
                           cuda::std::span<int> result) {
      using BlockReduce = cub::BlockReduce<int, block_size>;
      __shared__ typename BlockReduce::TempStorage temp_storage;

      int const index = threadIdx.x + blockIdx.x * blockDim.x;
      int sum = 0;
      if (index < data.size()) {
        sum += data[index];
      }
      sum = BlockReduce(temp_storage).Sum(sum);
      if (threadIdx.x == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> atomic_result(result.front());
        atomic_result.fetch_add(sum, cuda::memory_order_relaxed);
      }
    }
    """

    ast = parse_cuda(source)
    kernel = ast.kernels[0]
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert [(param.vtype, param.name) for param in kernel.params] == [
        ("cuda::std::span<int const>", "data"),
        ("cuda::std::span<int>", "result"),
    ]
    assert "array<i32> data" in crossgl
    assert "array<i32> result" in crossgl
    assert "var<workgroup> temp_storage: BlockReduce::TempStorage;" in crossgl
    assert "if ((index < data.size())) {" in crossgl
    assert "var atomic_result: cuda::atomic_ref<int, cuda::thread_scope_device>" in (
        crossgl
    )
    assert "cuda::std::span" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cccl_barrier_match_any_sync_active_mask_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cccl
    # commit: cea7dcd759b81d3059db8b74a3d5d4005ce4a398
    # path: libcudacxx/include/cuda/__barrier/barrier_block_scope.h
    # lines: 170-172
    source = """
    __device__ unsigned int match_barrier_update(unsigned int update,
                                                 unsigned int barrier_key) {
        unsigned int mask = ::__activemask();
        unsigned int activeA = ::__match_any_sync(mask, update);
        unsigned int activeB = __match_any_sync(mask, barrier_key);
        return activeA & activeB;
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert body[0].value.name == "::__activemask"
    assert body[1].value.name == "::__match_any_sync"
    assert body[2].value.name == "__match_any_sync"
    assert "var mask: u32 = WaveActiveBallot(true).x;" in crossgl
    assert "var activeA: u32 = WaveMatch(update).x;" in crossgl
    assert "var activeB: u32 = WaveMatch(barrier_key).x;" in crossgl
    assert "__match_any_sync" not in crossgl
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
        "imageStore(outputSurface, vec2<i32>(x, y), " "gIData[((y * width) + x)]);"
    ) in crossgl
    assert "sampler2D tex" in crossgl
    assert "gOData[((y * width) + x)] = texture(tex, vec2<f32>(tu, tv));" in crossgl
    assert "surf2Dwrite" not in crossgl
    assert "tex2D<float>" not in crossgl


def test_cuda_samples_scalar_prod_mul24_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/2_Concepts_and_Techniques/scalarProd/scalarProd_kernel.cuh
    source = """
    #define IMUL(a, b) __mul24(a, b)

    __global__ void scalarProdGPU(float *d_C, int elementN) {
        int vectorBase = IMUL(blockIdx.x, elementN);
        int vectorEnd = vectorBase + elementN;
        d_C[threadIdx.x] = vectorEnd;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert (
        "var vectorBase: i32 = "
        "(((gl_WorkGroupID.x << 8) >> 8) * ((elementN << 8) >> 8));"
    ) in crossgl
    assert "__mul24" not in crossgl


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


def test_llama_cpp_dp4a_integer_dot_product_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/ggml-org/llama.cpp
    # commit: 7c158fbb4aec1bdc9c81d6ca0e785139f4826fae
    # path: ggml/src/ggml-cuda/common.cuh
    # Reduced from ggml_cuda_dp4a, which returns __dp4a(a, b, c)
    # when the CUDA architecture supports packed int8 dot products.
    source = """
    __device__ int ggml_cuda_dp4a(int a, int b, int c) {
        return __dp4a(a, b, c);
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert body[0].value.name == "__dp4a"
    assert_crossgl_reparse(crossgl)
    assert "return (dot4I8Packed(a, b) + c);" in crossgl
    assert "__dp4a" not in crossgl


def test_cuda_samples_box_filter_saturatef_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_kernel.cu
    source = """
    __device__ unsigned int rgbaFloatToInt(float4 rgba) {
        rgba.x = __saturatef(rgba.x);
        rgba.y = __saturatef(rgba.y);
        return (unsigned int)(rgba.x * 255.0f);
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert body[0].right.name == "__saturatef"
    assert body[1].right.name == "__saturatef"
    assert "rgba.x = clamp(rgba.x, 0.0f, 1.0f);" in crossgl
    assert "rgba.y = clamp(rgba.y, 0.0f, 1.0f);" in crossgl
    assert "__saturatef" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_black_scholes_fdividef_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/5_Domain_Specific/BlackScholes/BlackScholes_kernel.cuh
    source = """
    __device__ float cndGPU(float d) {
        float K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));
        return K;
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert isinstance(body[0].value, FunctionCallNode)
    assert body[0].value.name == "__fdividef"
    assert "var K: f32 = (1.0f / (1.0f + (0.2316419f * abs(d))));" in crossgl
    assert "__fdividef" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_device_time_functions_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # paths:
    # cpp/0_Introduction/clock/clock.cu
    # cpp/6_Performance/cudaGraphsPerfScaling/cudaGraphPerfScaling.cu
    # Semantics source:
    # NVIDIA CUDA C++ Programming Guide v13.1.1, section 5.3.6.1.
    source = """
    __global__ void timedReduction(clock_t *timer, long long ticks) {
        int tid = threadIdx.x;
        long long endTime = ::clock64() + ticks;
        while (clock64() < endTime) {
        }
        if (tid == 0) {
            timer[blockIdx.x] = clock();
        }
    }
    """

    ast = parse_cuda(source)
    body = ast.kernels[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert body[1].value.left.name == "::clock64"
    assert body[2].condition.left.name == "clock64"
    assert body[3].if_body[0].right.name == "clock"
    assert (
        "var endTime: i64 = ((/* cuda time function clock64() "
        "not directly supported in CrossGL */ 0) + ticks);"
    ) in crossgl
    assert (
        "while (((/* cuda time function clock64() not directly supported in CrossGL "
        "*/ 0) < endTime)) {"
    ) in crossgl
    assert (
        "timer[gl_WorkGroupID.x] = (/* cuda time function clock() "
        "not directly supported in CrossGL */ 0);"
    ) in crossgl
    assert "clock64() + ticks" not in crossgl
    assert "= clock();" not in crossgl
    assert_crossgl_reparse(crossgl)


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
        "var<workgroup> shmem: " "array<array<f16, CHUNK_K_mul_K_plus_SKEW_HALF>>;"
    ) in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_bf16_tensor_core_pointer_declarations_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/3_CUDA_Features/bf16TensorCoreGemm/bf16TensorCoreGemm.cu
    source = """
    __global__ void compute_bf16gemm(const __nv_bfloat16 *A,
                                     __nv_bfloat16 *D) {
        extern __shared__ __nv_bfloat16 shmem[][CHUNK_K * K + SKEW_BF16];
        __nv_bfloat16 *local = D;
        const __nv_bfloat16 *tile_ptr =
            &shmem[threadIdx.y][threadIdx.x];
        __nv_bfloat16 value = A[threadIdx.x];
        local[threadIdx.x] = value;
        D[threadIdx.x] = tile_ptr[0];
    }
    """

    ast = parse_cuda(source)
    body = ast.kernels[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert ast.kernels[0].params[0].vtype == "const __nv_bfloat16 *"
    assert ast.kernels[0].params[1].vtype == "__nv_bfloat16 *"
    assert isinstance(body[0], SharedMemoryNode)
    assert body[0].vtype == "__nv_bfloat16[][((CHUNK_K * K) + SKEW_BF16)]"
    assert body[1].vtype == "__nv_bfloat16 *"
    assert body[1].name == "local"
    assert body[2].vtype == "const __nv_bfloat16 *"
    assert body[2].name == "tile_ptr"
    assert body[3].vtype == "__nv_bfloat16"
    assert "array<f16>" in crossgl
    assert (
        "var<workgroup> shmem: " "array<array<f16, CHUNK_K_mul_K_plus_SKEW_BF16>>;"
    ) in crossgl
    assert "var local: ptr<f16> = D;" in crossgl
    assert (
        "var tile_ptr: ptr<f16> = "
        "(&shmem[gl_LocalInvocationID.y][gl_LocalInvocationID.x]);"
    ) in crossgl
    assert "var value: f16 = A[gl_LocalInvocationID.x];" in crossgl
    assert "__nv_bfloat16" not in crossgl
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


def test_cuda_math_api_fast_single_precision_intrinsics_codegen_reparse():
    # Upstream source:
    # NVIDIA CUDA Math API Reference Manual v13.1, section 7
    # Single Precision Intrinsics.
    # URL:
    # https://docs.nvidia.com/cuda/archive/13.1.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html
    source = """
    __device__ float fast_intrinsics(float x, float y) {
        float trig = __sinf(x) + ::__cosf(x) + __tanf(x);
        float logs = __logf(y) + __log2f(y);
        float powered = __powf(x, y);
        float shaped = __tanhf(x) + ::__expf(y);
        return trig + logs + powered + shaped;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "var trig: f32 = ((sin(x) + cos(x)) + tan(x));" in crossgl
    assert "var logs: f32 = (log(y) + log2(y));" in crossgl
    assert "var powered: f32 = pow(x, y);" in crossgl
    assert "var shaped: f32 = (tanh(x) + exp(y));" in crossgl
    assert "__sinf" not in crossgl
    assert "__cosf" not in crossgl
    assert "__tanf" not in crossgl
    assert "__logf" not in crossgl
    assert "__log2f" not in crossgl
    assert "__powf" not in crossgl
    assert "__tanhf" not in crossgl
    assert "__expf" not in crossgl


def test_cuda_math_api_rounding_single_precision_intrinsics_codegen_reparse():
    # Upstream source:
    # NVIDIA CUDA Math API Reference Manual v13.1, section 7
    # Single Precision Intrinsics.
    # URL:
    # https://docs.nvidia.com/cuda/archive/13.1.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html
    source = """
    __device__ float rounded_intrinsics(float x, float y, float z) {
        float ops = __fadd_rn(x, y) + __fsub_rz(x, y) + __fmul_ru(x, y);
        float ratios = __fdiv_rd(x, y) + __frcp_rn(y);
        float roots = __fsqrt_rn(x) + __frsqrt_rn(y);
        float fused = __fmaf_rn(x, y, z) + __fmaf_ieee_rz(x, y, z);
        float base10 = __exp10f(x);
        float standard_fma = fmaf(x, y, z);
        return ops + ratios + roots + fused + base10 + standard_fma;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "var ops: f32 = (((x + y) + (x - y)) + (x * y));" in crossgl
    assert "var ratios: f32 = ((x / y) + (1.0f / y));" in crossgl
    assert "var roots: f32 = (sqrt(x) + inversesqrt(y));" in crossgl
    assert "var fused: f32 = (fma(x, y, z) + fma(x, y, z));" in crossgl
    assert "var base10: f32 = pow(10.0f, x);" in crossgl
    assert "var standard_fma: f32 = fma(x, y, z);" in crossgl
    for raw_name in {
        "__fadd_rn",
        "__fsub_rz",
        "__fmul_ru",
        "__fdiv_rd",
        "__frcp_rn",
        "__fsqrt_rn",
        "__frsqrt_rn",
        "__fmaf_rn",
        "__fmaf_ieee_rz",
        "__exp10f",
        "fmaf",
    }:
        assert raw_name not in crossgl


def test_cuda_math_api_float_to_integer_conversion_intrinsics_codegen_reparse():
    # Upstream source:
    # NVIDIA CUDA Math API Reference Manual v13.3, Type Casting Intrinsics.
    # URL:
    # https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html
    source = """
    __device__ unsigned long long convert_rounding(float x, double y) {
        int nearest = __float2int_rn(x);
        int down = ::__float2int_rd(x);
        unsigned int trunc = __float2uint_rz(x);
        unsigned int up = __double2uint_ru(y);
        long long wide = __double2ll_rn(y);
        unsigned long long uwide = __float2ull_rd(x);
        return (unsigned long long)(nearest + down) + trunc + up
               + (unsigned long long)wide + uwide;
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert body[0].value.name == "__float2int_rn"
    assert body[1].value.name == "::__float2int_rd"
    assert body[2].value.name == "__float2uint_rz"
    assert body[3].value.name == "__double2uint_ru"
    assert body[4].value.name == "__double2ll_rn"
    assert body[5].value.name == "__float2ull_rd"
    assert_crossgl_reparse(crossgl)
    assert "var nearest: i32 = i32(round(x));" in crossgl
    assert "var down: i32 = i32(floor(x));" in crossgl
    assert "var trunc: u32 = u32(x);" in crossgl
    assert "var up: u32 = u32(ceil(y));" in crossgl
    assert "var wide: i64 = i64(round(y));" in crossgl
    assert "var uwide: u64 = u64(floor(x));" in crossgl
    for raw_name in {
        "__float2int_rn",
        "__float2int_rd",
        "__float2uint_rz",
        "__double2uint_ru",
        "__double2ll_rn",
        "__float2ull_rd",
    }:
        assert raw_name not in crossgl


def test_cuda_math_api_qualified_scalar_math_aliases_codegen_reparse():
    source = """
    __device__ float qualified_aliases(float x, float y, float z) {
        float roots = ::sqrtf(x) + ::rsqrtf(y);
        float extrema = std::fminf(x, y) + std::fmaxf(x, y);
        float magnitude = ::cuda::std::fabs(x);
        float fused = ::std::fmaf(x, y, z);
        return roots + extrema + magnitude + fused;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "var roots: f32 = (sqrt(x) + inversesqrt(y));" in crossgl
    assert "var extrema: f32 = (min(x, y) + max(x, y));" in crossgl
    assert "var magnitude: f32 = abs(x);" in crossgl
    assert "var fused: f32 = fma(x, y, z);" in crossgl
    for raw_name in {
        "::sqrtf",
        "::rsqrtf",
        "std::fminf",
        "std::fmaxf",
        "::cuda::std::fabs",
        "::std::fmaf",
        "cuda::std::fabs",
        "std::fmaf",
        "sqrtf",
        "rsqrtf",
        "fminf",
        "fmaxf",
        "fmaf",
    }:
        assert raw_name not in crossgl


def test_cuda_math_api_namespace_alias_scalar_math_codegen_reparse():
    source = """
    namespace cstd = cuda::std;

    __device__ float namespace_alias_math(float x, float y, float z) {
        float extrema = cstd::fminf(x, y) + cstd::fmaxf(x, y);
        float fused = cstd::fmaf(x, y, z);
        return extrema + fused;
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "var extrema: f32 = (min(x, y) + max(x, y));" in crossgl
    assert "var fused: f32 = fma(x, y, z);" in crossgl
    for raw_name in {
        "cstd::",
        "cuda::std::fminf",
        "cuda::std::fmaxf",
        "cuda::std::fmaf",
        "fminf",
        "fmaxf",
        "fmaf",
    }:
        assert raw_name not in crossgl


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


def test_cuda_samples_cdp_quadtree_tile_collectives_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/3_CUDA_Features/cdpQuadtree/cdpQuadtree.cu
    source = """
    namespace cg = cooperative_groups;

    __global__ void qtree_warp(unsigned int *out,
                               bool pred,
                               unsigned int lane_mask_lt,
                               int limit) {
        cg::thread_block cta = cg::this_thread_block();
        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

        for (int range_it = threadIdx.x;
             tile32.any(range_it < limit);
             range_it += warpSize) {
            unsigned int vote = tile32.ballot(pred);
            unsigned int dest = __popc(vote & lane_mask_lt);
            out[threadIdx.x] = tile32.shfl(dest, 0);
        }
    }
    """

    ast = parse_cuda(source)
    loop = ast.kernels[0].body[2]
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert isinstance(loop, ForNode)
    assert isinstance(loop.condition, FunctionCallNode)
    assert isinstance(loop.condition.name, MemberAccessNode)
    assert loop.condition.name.member == "any"
    assert loop.body[0].value.name.member == "ballot"
    assert loop.body[2].right.name.member == "shfl"
    assert (
        "for (var range_it: i32 = gl_LocalInvocationID.x; "
        "WaveActiveAnyTrue(((range_it < limit) != 0)); range_it += 32)" in crossgl
    )
    assert "var vote: u32 = WaveActiveBallot((pred != 0)).x;" in crossgl
    assert "var dest: u32 = bitCount((vote & lane_mask_lt));" in crossgl
    assert "out[gl_LocalInvocationID.x] = WaveReadLaneAt(dest, 0);" in crossgl
    assert "tile32.any" not in crossgl
    assert "tile32.ballot" not in crossgl
    assert "tile32.shfl" not in crossgl
    assert "thread_block_tile.any not directly supported" not in crossgl
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


def test_cuda_math_api_long_long_integer_intrinsics_codegen_reparse():
    # Upstream source:
    # NVIDIA CUDA Math API v13.3, section 13 Integer Intrinsics.
    source = """
    __device__ int long_long_intrinsics(unsigned long long x) {
        int leading = __clzll(x);
        int first = __ffsll(x);
        return leading + first;
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert body[0].value.name == "__clzll"
    assert body[1].value.name == "__ffsll"
    assert_crossgl_reparse(crossgl)
    assert "var leading: i32 = countLeadingZeros(x);" in crossgl
    assert "var first: i32 = (findLSB(x) + 1);" in crossgl
    assert "__clzll" not in crossgl
    assert "__ffsll" not in crossgl


def test_cuda_cccl_integer_average_intrinsics_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cccl
    # commit: bfea3082381a656d9a36fc3dc39f2057f4c31615
    # path: c/parallel.v2/src/hostjit/include/hostjit/cuda_minimal/
    #       __clang_cuda_device_functions.h
    source = """
    __device__ int signed_average(int x, int y) {
        int floor_avg = ::__hadd(x, y);
        int rounded_avg = __rhadd(x, y);
        return floor_avg + rounded_avg;
    }

    __device__ unsigned int unsigned_average(unsigned int x, unsigned int y) {
        unsigned int floor_avg = __uhadd(x, y);
        unsigned int rounded_avg = __urhadd(x, y);
        return floor_avg + rounded_avg;
    }
    """

    ast = parse_cuda(source)
    signed_body = ast.functions[0].body
    unsigned_body = ast.functions[1].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert signed_body[0].value.name == "::__hadd"
    assert signed_body[1].value.name == "__rhadd"
    assert unsigned_body[0].value.name == "__uhadd"
    assert unsigned_body[1].value.name == "__urhadd"
    assert "var floor_avg: i32 = ((x & y) + ((x ^ y) >> 1));" in crossgl
    assert "var rounded_avg: i32 = ((x | y) - ((x ^ y) >> 1));" in crossgl
    assert "var floor_avg: u32 = ((x & y) + ((x ^ y) >> 1));" in crossgl
    assert "var rounded_avg: u32 = ((x | y) - ((x ^ y) >> 1));" in crossgl
    assert "__hadd" not in crossgl
    assert "__rhadd" not in crossgl
    assert "__uhadd" not in crossgl
    assert "__urhadd" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_math_api_funnelshift_integer_intrinsics_emit_diagnostics():
    # Upstream source:
    # NVIDIA CUDA Math API v13.2, section 13 Integer Intrinsics.
    # URL:
    # https://docs.nvidia.com/cuda/archive/13.2.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html
    source = """
    __device__ unsigned int funnel_intrinsics(unsigned int lo,
                                              unsigned int hi,
                                              unsigned int shift) {
        unsigned int left = __funnelshift_l(lo, hi, shift);
        unsigned int left_clamped = __funnelshift_lc(lo, hi, shift);
        unsigned int right = __funnelshift_r(lo, hi, shift);
        unsigned int right_clamped = ::__funnelshift_rc(lo, hi, shift);
        return left + left_clamped + right + right_clamped;
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert body[0].value.name == "__funnelshift_l"
    assert body[1].value.name == "__funnelshift_lc"
    assert body[2].value.name == "__funnelshift_r"
    assert body[3].value.name == "::__funnelshift_rc"
    assert_crossgl_reparse(crossgl)
    for name in {
        "__funnelshift_l",
        "__funnelshift_lc",
        "__funnelshift_r",
        "__funnelshift_rc",
    }:
        assert f"cuda integer intrinsic {name}(lo, hi, shift)" in crossgl
        assert f"= {name}(lo, hi, shift);" not in crossgl


def test_nvidia_hpcg_cuda_kernels_brev_hash_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/nvidia-hpcg
    # commit: 7dd63cd06c0620dddd5702ad7b4fca376c19813e
    # path: src/CudaKernels.cu
    source = """
    __global__ void color_kernel(unsigned int *out,
                                 int *color,
                                 int next_color) {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (color[i] != -1) {
            return;
        }

        unsigned int i_rand = __brev(i) /* hash function*/;
        unsigned int j = i + 1;
        unsigned int j_rand = __brev(j) /* hash function*/;

        if (i_rand <= j_rand) {
            out[threadIdx.x] = j_rand;
        }
    }
    """

    crossgl = cuda_to_crossgl(source)
    regenerated_cuda = crossgl_to_cuda(crossgl)

    assert_crossgl_reparse(crossgl)
    assert "var i_rand: u32 = bitfieldReverse(i);" in crossgl
    assert "var j_rand: u32 = bitfieldReverse(j);" in crossgl
    assert "__brev" not in crossgl
    assert "unsigned int i_rand = __brev(i);" in regenerated_cuda
    assert "unsigned int j_rand = __brev(j);" in regenerated_cuda
    assert "reverseBits(" not in regenerated_cuda


def test_nvidia_hpcg_cuda_kernels_ldcs_cached_load_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/nvidia-hpcg
    # commit: 7dd63cd06c0620dddd5702ad7b4fca376c19813e
    # path: src/CudaKernels.cu
    source = """
    __global__ void lower_symmetric_cached_loads(int *ell_columns,
                                                 double *ell_values,
                                                 int *out) {
        int i = threadIdx.x;
        int col = __ldcs(&ell_columns[i]);
        double val = __ldcs(&ell_values[i]);
        out[i] = col + (int)val;
    }
    """

    ast = parse_cuda(source)
    body = ast.kernels[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert body[1].value.name == "__ldcs"
    assert body[2].value.name == "__ldcs"
    assert_crossgl_reparse(crossgl)
    assert "var col: i32 = ell_columns[i];" in crossgl
    assert "var val: f64 = ell_values[i];" in crossgl
    assert "__ldcs" not in crossgl


def test_external_cccl_bit_reverse_brevll_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cccl
    # commit: 5a9ea633bfe63f113f4e99ecd505985ec2c38206
    # path: libcudacxx/include/cuda/__bit/bit_reverse.h
    source = """
    __device__ unsigned long long bit_reverse_device(unsigned long long value) {
        return ::__brevll(value);
    }
    """

    crossgl = cuda_to_crossgl(source)
    regenerated_cuda = crossgl_to_cuda(crossgl)

    assert_crossgl_reparse(crossgl)
    assert "return bitfieldReverse(value);" in crossgl
    assert "__brevll" not in crossgl
    assert "return __brevll(value);" in regenerated_cuda
    assert "reverseBits(" not in regenerated_cuda


def test_external_cccl_bit_reverse_macro_block_is_skipped():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cccl
    # commit: 5a9ea633bfe63f113f4e99ecd505985ec2c38206
    # path: libcudacxx/include/cuda/__bit/bit_reverse.h
    source = """
    template <typename _Tp>
    [[nodiscard]] _CCCL_API constexpr _Tp bit_reverse(_Tp __value) noexcept
    {
      static_assert(::cuda::std::__cccl_is_cv_unsigned_integer_v<_Tp>,
                    "bit_reverse() requires unsigned integer types");
      _CCCL_IF_NOT_CONSTEVAL_DEFAULT
      {
        NV_IF_TARGET(
            NV_IS_DEVICE,
            (return ::cuda::__bit_reverse_device(__value);))
      }
      return ::cuda::__bit_reverse_generic(__value);
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body

    assert ast.functions[0].name == "bit_reverse"
    assert isinstance(body[0], FunctionCallNode)
    assert body[0].name == "static_assert"
    assert len(body) == 2
    assert isinstance(body[1], ReturnNode)
    assert body[1].value.name == "::cuda::__bit_reverse_generic"


def test_cuda_math_api_sad_intrinsics_codegen_reparse():
    # Upstream source:
    # NVIDIA CUDA Math API 12.6.1, Integer Intrinsics.
    source = """
    __global__ void sad_kernel(unsigned int *out,
                               int x,
                               int y,
                               unsigned int ux,
                               unsigned int uy,
                               unsigned int bias) {
        out[0] = __sad(x, y, bias);
        out[1] = __usad(ux, uy, bias);
    }
    """

    crossgl = cuda_to_crossgl(source)

    assert_crossgl_reparse(crossgl)
    assert "out[0] = (abs(x - y) + bias);" in crossgl
    assert "out[1] = (((ux > uy) ? (ux - uy) : (uy - ux)) + bias);" in crossgl
    assert "__sad" not in crossgl
    assert "__usad" not in crossgl


def test_rapids_cudf_hash_byte_perm_endian_swap_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/rapidsai/cudf
    # commit: d387ee637326739a00bb4825eebb2ad0c66bdd01
    # path: cpp/include/cudf/hashing/detail/hash_functions.cuh
    source = """
    __device__ inline uint32_t swap_endian(uint32_t x)
    {
        return __byte_perm(x, 0, 0x0123);
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert body[0].value.name == "__byte_perm"
    assert_crossgl_reparse(crossgl)
    assert (
        "return ((((x & 0x000000ffu) << 24) | ((x & 0x0000ff00u) << 8)) | "
        "(((x & 0x00ff0000u) >> 8) | ((x & 0xff000000u) >> 24)));"
    ) in crossgl
    assert "__byte_perm" not in crossgl


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


def test_cuda_samples_reduction_multiblock_cg_tile_reduce_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/2_Concepts_and_Techniques/reductionMultiBlockCG/reductionMultiBlockCG.cu
    source = """
    namespace cg = cooperative_groups;

    __device__ void reduceBlock(double *sdata, const cg::thread_block &cta) {
        const unsigned int tid = cta.thread_rank();
        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
        sdata[tid] = cg::reduce(tile32, sdata[tid], cg::plus<double>());
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert isinstance(body[2].right, FunctionCallNode)
    assert body[2].right.name == "cg::reduce"
    assert_crossgl_reparse(crossgl)
    assert "sdata[tid] = WaveActiveSum(sdata[tid]);" in crossgl
    assert "cg::reduce" not in crossgl
    assert "cooperative_groups::reduce" not in crossgl


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


def test_public_cuda_kernel_if_init_statement_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/InternLM/lmdeploy
    # commit: 51334f09560a3432c434ad39c5ea67cc379ad995
    # path: src/turbomind/kernels/quantization.cu
    source = """
    __global__ void guarded_store(int* out,
                                  int num,
                                  int dim,
                                  int rows,
                                  int row) {
        int di = threadIdx.x;
        int ti = blockIdx.x;
        for (int s = 0; s < 2; ++s) {
            if (auto r = ti + s * rows + row; r < num && di < dim) {
                out[r] = di;
            }
        }
    }
    """

    ast = parse_cuda(source)
    loop_body = ast.kernels[0].body[2].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert isinstance(loop_body[0], VariableNode)
    assert loop_body[0].name == "r"
    assert loop_body[0].vtype == "auto"
    assert isinstance(loop_body[1], IfNode)
    assert "var r: auto = ((ti + (s * rows)) + row);" in crossgl
    assert "if (((r < num) && (di < dim))) {" in crossgl
    assert "out[r] = di;" in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_programming_guide_nanosleep_mutex_codegen_reparse():
    # Upstream source:
    # NVIDIA CUDA C++ Programming Guide v12.5.1, section 7.23.3.
    # URL:
    # https://docs.nvidia.com/cuda/archive/12.5.1/cuda-c-programming-guide/index.html#nanosleep-function
    source = """
    __device__ void mutex_lock(unsigned int *mutex) {
        unsigned int ns = 8;
        while (atomicCAS(mutex, 0, 1) == 1) {
            __nanosleep(ns);
            if (ns < 256) {
                ns *= 2;
            }
        }
    }
    """

    ast = parse_cuda(source)
    loop = ast.functions[0].body[1]
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert loop.body[0].name == "__nanosleep"
    assert (
        "/* cuda nanosleep __nanosleep(ns) not directly supported in CrossGL */ 0"
        in crossgl
    )
    assert "__nanosleep(ns);" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_dxtc_bitfield_struct_members_parse_and_codegen():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/5_Domain_Specific/dxtc/dxtc.cu
    source = """
    union Color16
    {
        struct
        {
            unsigned short b : 5;
            unsigned short g : 6;
            unsigned short r : 5;
        };
        unsigned short u;
    };

    struct BlockDXT1
    {
        Color16 col0;
        Color16 col1;
        union
        {
            unsigned char row[4];
            unsigned int  indices;
        };
    };
    """

    ast = parse_cuda(source)
    color16, block = ast.structs
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert [(member.vtype, member.name) for member in color16.members] == [
        ("unsigned short", "b"),
        ("unsigned short", "g"),
        ("unsigned short", "r"),
        ("unsigned short", "u"),
    ]
    assert [(member.vtype, member.name) for member in block.members] == [
        ("Color16", "col0"),
        ("Color16", "col1"),
        ("unsigned char[4]", "row"),
        ("unsigned int", "indices"),
    ]
    assert "u16 b;" in crossgl
    assert "u16 g;" in crossgl
    assert "u16 r;" in crossgl
    assert "u16 u;" in crossgl
    assert " : 5" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_stereo_disparity_usad4_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/5_Domain_Specific/stereoDisparity/stereoDisparity_kernel.cuh
    source = """
    __device__ unsigned int __usad4(unsigned int A,
                                    unsigned int B,
                                    unsigned int C = 0) {
        unsigned int result;

        asm("vabsdiff4.u32.u32.u32.add"
            " %0, %1, %2, %3;"
            : "=r"(result)
            : "r"(A), "r"(B), "r"(C));

        return result;
    }

    __global__ void stereoDisparityKernel(unsigned int *g_odata,
                                          unsigned int imLeft,
                                          unsigned int imRight,
                                          unsigned int seed) {
        unsigned int cost = __usad4(imLeft, imRight);
        unsigned int biasedCost = __usad4(imLeft, imRight, seed);
        g_odata[threadIdx.x] = cost + biasedCost;
    }
    """

    ast = parse_cuda(source)
    body = ast.kernels[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert ast.functions[0].name == "__usad4"
    assert body[0].value.name == "__usad4"
    assert body[1].value.name == "__usad4"
    assert_crossgl_reparse(crossgl)
    assert 'CUDA inline PTX: "vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;"' in crossgl
    assert "((imLeft & 0xffu) > (imRight & 0xffu))" in crossgl
    assert "(((imLeft >> 24) & 0xffu) > ((imRight >> 24) & 0xffu))" in crossgl
    assert "+ seed);" in crossgl
    assert "__usad4(imLeft, imRight)" not in crossgl
    assert "__usad4(imLeft, imRight, seed)" not in crossgl


def test_cuda_samples_dxtc_rintf_color_quantization_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/5_Domain_Specific/dxtc/dxtc.cu
    source = """
    __device__ unsigned int packRgb565(float4 v) {
        v.x = rintf(__saturatef(v.x) * 31.0f);
        v.y = rintf(__saturatef(v.y) * 63.0f);
        v.z = rintf(__saturatef(v.z) * 31.0f);
        return ((unsigned int)v.x << 11) |
               ((unsigned int)v.y << 5) |
               (unsigned int)v.z;
    }
    """

    ast = parse_cuda(source)
    body = ast.functions[0].body
    crossgl = CudaToCrossGLConverter().generate(ast)

    assert body[0].right.name == "rintf"
    assert_crossgl_reparse(crossgl)
    assert "v.x = round((clamp(v.x, 0.0f, 1.0f) * 31.0f));" in crossgl
    assert "v.y = round((clamp(v.y, 0.0f, 1.0f) * 63.0f));" in crossgl
    assert "v.z = round((clamp(v.z, 0.0f, 1.0f) * 31.0f));" in crossgl
    assert "rintf" not in crossgl


def test_cuda_samples_tile_spmv_trailing_unsigned_parameter_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/9_CUDA_Tile/tileSpMV/tileSpMV.cu
    source = """
    struct SellMatrix {};

    static SellMatrix generateRandom(int num_rows, int num_cols,
                                     int avg_nnz_per_row, unsigned seed) {
      return SellMatrix();
    }
    """

    ast = parse_cuda(source)
    params = ast.functions[0].params
    crossgl = cuda_to_crossgl(source)

    assert [(param.vtype, param.name) for param in params] == [
        ("int", "num_rows"),
        ("int", "num_cols"),
        ("int", "avg_nnz_per_row"),
        ("unsigned int", "seed"),
    ]
    assert "u32 seed" in crossgl
    assert "unsigned seed _unused_param" not in crossgl
    assert_crossgl_reparse(crossgl)


def test_cuda_samples_vector_of_const_char_pointer_type_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # paths:
    # - cpp/5_Domain_Specific/vulkanImageCUDA/vulkanImageCUDA.cu
    # - cpp/6_Performance/cudaGraphsPerfScaling/cudaGraphPerfScaling.cu
    source = """
    const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
    std::vector<const char *> metricName;
    """

    crossgl = cuda_to_crossgl(source)

    assert "var validationLayers: std_vector_ptr_i8" in crossgl
    assert "var metricName: std_vector_ptr_i8;" in crossgl
    assert "ptr<std::vector<const char>>" not in crossgl
    assert_crossgl_reparse(crossgl)
