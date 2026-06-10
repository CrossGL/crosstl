from crosstl.backend.HIP.HipAst import (
    AtomicOperationNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    FunctionNode,
    HipAsmNode,
    IfNode,
    KernelLaunchNode,
    KernelNode,
    StructNode,
    SwitchNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
)
from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

EXTERNAL_FIXTURE_SOURCES = {
    "rocm_examples": {
        "url": "https://github.com/ROCm/rocm-examples",
        "commit": "cf369da68f209c315074204bd0eb61d1a5c015d1",
        "paths": [
            "Applications/convolution/main.hip",
            "Common/rocjpeg_utils.hpp",
            "HIP-Basic/bit_extract/main.hip",
            "HIP-Basic/cooperative_groups/main.hip",
            "HIP-Basic/device_globals/main.hip",
            "HIP-Basic/dynamic_shared/main.hip",
            "HIP-Basic/texture_management/main.hip",
            "HIP-Basic/warp_shuffle/main.hip",
            "HIP-Doc/Reference/HIP-Complex-Math-API/complex_math/main.hip",
            "HIP-Doc/Tutorials/graph_api/src/filtering.hip",
            "HIP-Doc/Tutorials/graph_api/src/phantom.hip",
            "HIP-Doc/Tutorials/Programming-Patterns/image_convolution/main.hip",
            "HIP-Doc/Programming-Guide/HIP-C++-Language-Extensions/warp_size_reduction/popcount.hpp",
        ],
    },
    "rocm_examples_inline_assembly": {
        "url": "https://github.com/ROCm/rocm-examples",
        "commit": "b4ee9992e851a078c99d93de59d6142a51f5e3a1",
        "paths": ["HIP-Basic/inline_assembly/main.hip"],
    },
    "rocm_examples_rocdecode": {
        "url": "https://github.com/ROCm/rocm-examples",
        "commit": "d3ad835e46ff50412cf51086df7400fb3bbd1649",
        "paths": ["Common/rocdecode_utils.hpp"],
    },
    "rocm_examples_runtime_compilation": {
        "url": "https://github.com/ROCm/rocm-examples",
        "commit": "d3ad835e46ff50412cf51086df7400fb3bbd1649",
        "paths": ["HIP-Basic/runtime_compilation/main.hip"],
    },
    "rocm_examples_stb_image_write": {
        "url": "https://github.com/ROCm/rocm-examples",
        "commit": "d3ad835e46ff50412cf51086df7400fb3bbd1649",
        "paths": [
            "HIP-Doc/Tutorials/Programming-Patterns/image_convolution/"
            "stb_image_write.h"
        ],
    },
    "hip_examples": {
        "url": "https://github.com/ROCm/HIP-Examples",
        "commit": "cdf9d101acd9a3fc89ee750f73c1f1958cbd5cc3",
        "paths": ["HIP-Examples-Applications/Histogram/Histogram.cpp"],
    },
    "rocwmma": {
        "url": "https://github.com/ROCm/rocWMMA",
        "commit": "1e3bed23f981d8ca0ce1b8634e6b91b5ccf91cfb",
        "paths": [
            "samples/simple_sgemm.cpp",
            "samples/simple_hgemm.cpp",
            "samples/simple_dgemm.cpp",
        ],
    },
    "hpc_training": {
        "url": "https://github.com/amd/HPCTrainingExamples",
        "commit": "56b903adadb113097ed4333d0bdc3e3bc537c8a2",
    },
    "hip": {
        "url": "https://github.com/ROCm/HIP",
        "commit": "0447ec8e079d9cd0a2bc966124977a0b92fac472",
        "paths": ["docs/tools/example_codes/low_precision_float_fp16.hip"],
    },
    "hip_tests": {
        "url": "https://github.com/ROCm/hip-tests",
        "commit": "d01e1f96059edc25600eb13434d7e2b71c09af01",
        "paths": [
            "catch/unit/deviceLib/funnelshift.cc",
            "catch/unit/math/integer_intrinsics.cc",
            "catch/unit/deviceLib/popc.cc",
        ],
    },
    "hip_tests_nested_sections": {
        "url": "https://github.com/ROCm/hip-tests",
        "commit": "8889ba5c7a89a85d5262dadcfbde17589a53ccfb",
        "paths": ["catch/multiproc/hipIpcEventHandle.cc"],
    },
    "hip_tests_cooperative_groups": {
        "url": "https://github.com/ROCm/hip-tests",
        "commit": "8889ba5c7a89a85d5262dadcfbde17589a53ccfb",
        "paths": [
            "catch/unit/cooperativeGrps/hipCGThreadBlockTileTypeShfl_old.cc",
        ],
    },
    "hipblas": {
        "url": "https://github.com/ROCm/hipBLAS",
        "commit": "23b26a0093345264e7387481cbe01d1e1ae55fda",
        "paths": ["clients/gtest/blas3/gemm_gtest.cpp"],
    },
    "rocblas": {
        "url": "https://github.com/ROCm/rocBLAS",
        "commit": "defce200a69e5346eeadd7ac1e199238758add61",
        "paths": [
            "clients/common/blas3/common_gemm.cpp",
            "clients/common/common_helpers.hpp",
        ],
    },
    "llvm_project": {
        "url": "https://github.com/llvm/llvm-project",
        "commit": "3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff",
        "paths": ["clang/test/SemaCUDA/amdgpu-attrs.cu"],
    },
    "llvm_cuda_consteval": {
        "url": "https://github.com/llvm/llvm-project",
        "commit": "146abed5cd072a27c4240c7cf53b764571e62462",
        "paths": ["clang/test/SemaCUDA/consteval-func.cu"],
    },
    "cccl": {
        "url": "https://github.com/NVIDIA/cccl",
        "commit": "17869e1d46314036843a9ff9a0cb726de560e94e",
        "paths": ["libcudacxx/include/cuda/std/__utility/pair.h"],
    },
    "hip_kittens": {
        "url": "https://github.com/HazyResearch/HipKittens",
        "commit": "cd090ae98ee4e7b8d3d5291fc62cfd716aecb946",
        "paths": ["include/common/base_ops.cuh"],
    },
}


def parse_hip_source(source):
    tokens = HipLexer(source).tokenize()
    return HipParser(tokens).parse()


def generate_crossgl_from_hip(source):
    ast = parse_hip_source(source)
    return ast, HipToCrossGLConverter().generate(ast)


def assert_crossgl_reparses(source):
    ast, crossgl = generate_crossgl_from_hip(source)
    CrossGLParser(CrossGLLexer(crossgl).tokens).parse()
    return ast, crossgl


def generate_hip_from_crossgl(source):
    crossgl_ast = CrossGLParser(CrossGLLexer(source).tokens).parse()
    return HipCodeGen().generate(crossgl_ast)


COMPILER_MANUAL_SAMPLER_FIXTURE = """
shader DirectXMixedManualSamplerUsageUnsupportedShader {
  compute {
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
    layout(set = 0, binding = 0) buffer vec4* values;
    layout(set = 0, binding = 2) uniform sampler2DShadow shadowMap;
    layout(set = 0, binding = 3) uniform sampler2DArrayShadow shadowAtlas;
    layout(set = 0, binding = 5) sampler sharedShadowSampler;

    void main() {
      float visibility =
          textureCompare(shadowMap, sharedShadowSampler,
                         vec2(0.5, 0.5), 0.25);
      float manualVisibility =
          textureCompareLodManual(shadowAtlas, sharedShadowSampler,
                                  vec3(0.25, 0.5, 1.0), 0.33, 2.0,
                                  less_equal);
      values[0] = vec4(visibility, manualVisibility, 0.0, 1.0);
      return;
    }
  }
}
"""


COMPILER_MATRIX_CONSTRUCTOR_FIXTURE = """
shader MatrixConstructorComputeShader {
  compute {
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
    layout(set = 0, binding = 0) buffer float* values;

    void keep(mat2 flattened, mat2 diagonal, mat3 expanded, mat3 basis) {
      values[0] = 1.0;
      return;
    }

    void main() {
      mat2 flattened = mat2(1.0, 2.0, 3.0, 4.0);
      mat2 diagonal = mat2(5.0);
      mat3 expanded = mat3(flattened);
      vec3 c0 = vec3(1.0, 2.0, 3.0);
      vec3 c1 = vec3(4.0, 5.0, 6.0);
      vec3 c2 = vec3(7.0, 8.0, 9.0);
      mat3 basis = mat3(c0, c1, c2);
      keep(flattened, diagonal, expanded, basis);
      return;
    }
  }
}
"""


def assert_hip_fixture_reverse_codegen_prunes_generated_matrix_helpers(source):
    hip_code = generate_hip_from_crossgl(source)
    assert "struct float2x2" in hip_code

    _, crossgl = generate_crossgl_from_hip(hip_code)

    assert "struct float2x2" not in crossgl
    assert "operator_mul" not in crossgl
    assert "CGL_COLUMNS" not in crossgl
    CrossGLParser(CrossGLLexer(crossgl).tokens).parse()
    return crossgl


def test_compiler_manual_sampler_fixture_reverse_codegen_prunes_hip_helpers():
    # Reduced from CrossGL-Compiler/tests/fixtures/
    # DirectXMixedManualSamplerUsageUnsupportedShader.cgl.
    crossgl = assert_hip_fixture_reverse_codegen_prunes_generated_matrix_helpers(
        COMPILER_MANUAL_SAMPLER_FIXTURE
    )

    assert len(crossgl) < 2_000
    assert "textureCompareLodManual(" in crossgl
    assert "var visibility: f32 = 0.0f;" in crossgl


def test_compiler_matrix_constructor_fixture_reverse_codegen_prunes_hip_helpers():
    # Reduced from CrossGL-Compiler/tests/fixtures/MatrixConstructorComputeShader.cgl.
    crossgl = assert_hip_fixture_reverse_codegen_prunes_generated_matrix_helpers(
        COMPILER_MATRIX_CONSTRUCTOR_FIXTURE
    )

    assert len(crossgl) < 2_000
    assert (
        "void keep(mat2 flattened, mat2 diagonal, mat3 expanded, mat3 basis)" in crossgl
    )
    assert "var flattened: mat2 = mat2(1.0, 2.0, 3.0, 4.0);" in crossgl
    assert "var basis: mat3 = mat3(c0, c1, c2);" in crossgl


def test_external_hipblas_gtest_parameterized_block_macro_codegen_reparse():
    # Reduced from ROCm/hipBLAS@23b26a0093345264e7387481cbe01d1e1ae55fda,
    # clients/gtest/blas3/gemm_gtest.cpp.
    source = """
    using gemm = gemm_template<gemm_testing, GEMM>;
    TEST_P(gemm, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<gemm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm);
    """

    ast, crossgl = assert_crossgl_reparses(source)
    alias = ast.statements[0]
    test_case = ast.statements[1]
    instantiate = ast.statements[2]

    assert isinstance(alias, TypeAliasNode)
    assert alias.name == "gemm"
    assert isinstance(test_case, FunctionNode)
    assert test_case.name == "gemm_blas3"
    assert test_case.qualifiers == ["TEST_P"]
    assert isinstance(test_case.body[0], FunctionCallNode)
    assert test_case.body[0].name == "CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES"
    assert isinstance(instantiate, FunctionCallNode)
    assert instantiate.name == "INSTANTIATE_TEST_CATEGORIES"
    assert "// Function: gemm_blas3" in crossgl
    assert "hipblas_simple_dispatch<gemm_testing>(GetParam())" in crossgl
    assert "TEST_P" not in crossgl


def test_external_rocblas_repeated_instantiate_tests_macro_chain_parse():
    # Reduced from ROCm/rocBLAS@defce200a69e5346eeadd7ac1e199238758add61,
    # clients/common/blas3/common_gemm.cpp plus clients/common/common_helpers.hpp.
    source = """
    #define INSTANTIATE(T_)                 \\
        INSTANTIATE_TESTS(gemm, T_)         \\
        INSTANTIATE_TESTS(gemm_batched, T_) \\
        INSTANTIATE_TESTS(gemm_strided_batched, T_)

    INSTANTIATE(rocblas_half)
    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(rocblas_float_complex)
    """

    ast = parse_hip_source(source)
    calls = ast.statements

    assert len(calls) == 12
    assert all(isinstance(call, FunctionCallNode) for call in calls)
    assert {call.name for call in calls} == {"INSTANTIATE_TESTS"}
    assert [call.args for call in calls[:3]] == [
        ["gemm", "rocblas_half"],
        ["gemm_batched", "rocblas_half"],
        ["gemm_strided_batched", "rocblas_half"],
    ]
    assert calls[-1].args == ["gemm_strided_batched", "rocblas_float_complex"]


def test_external_rocthrust_rocprim_namespace_macros_codegen_reparse():
    # Reduced from ROCm/rocThrust@8c061ed4f0628254578a3de28df775bea765f89d
    # and ROCm/rocPRIM@14cd5e3c27a4b9ae7d510823a450723a03985ac0.
    source = """
    THRUST_NAMESPACE_BEGIN
    THRUST_EXEC_CHECK_DISABLE
    THRUST_HOST_DEVICE THRUST_FORCEINLINE int thrust_identity(int value) {
        return value;
    }
    THRUST_NAMESPACE_END

    BEGIN_ROCPRIM_NAMESPACE
    THRUST_HIP_DEVICE_FUNCTION int rocprim_identity(int value) {
        return value;
    }
    END_ROCPRIM_NAMESPACE
    """

    ast, crossgl = assert_crossgl_reparses(source)

    assert [statement.name for statement in ast.statements] == [
        "thrust_identity",
        "rocprim_identity",
    ]
    assert ast.statements[0].qualifiers == [
        "THRUST_HOST_DEVICE",
        "THRUST_FORCEINLINE",
    ]
    assert ast.statements[1].qualifiers == ["THRUST_HIP_DEVICE_FUNCTION"]
    assert "i32 thrust_identity(i32 value)" in crossgl
    assert "i32 rocprim_identity(i32 value)" in crossgl
    assert "THRUST_" not in crossgl
    assert "ROCPRIM_NAMESPACE" not in crossgl


def test_external_rocprim_enable_if_trailing_return_codegen_reparse():
    # Reduced from ROCm/rocPRIM@14cd5e3c27a4b9ae7d510823a450723a03985ac0,
    # test/rocprim/test_block_histogram.kernels.hpp.
    source = """
    template<class T>
    auto get_safe_maxval(size_t maxval)
        -> std::enable_if_t<rocprim::is_floating_point<T>::value, bool> {
        return maxval > 0;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    assert isinstance(ast.statements[0], FunctionNode)
    assert ast.statements[0].return_type == (
        "std::enable_if_t<rocprim::is_floating_point<T>::value, bool>"
    )
    assert "void get_safe_maxval(u32 maxval)" in crossgl
    assert "std::enable_if_t" not in crossgl


def test_external_rocthrust_static_templated_constructor_declaration_codegen_reparse():
    # Reduced from ROCm/rocThrust@8c061ed4f0628254578a3de28df775bea765f89d,
    # examples/sum.cu and examples/lexicographical_sort.cu.
    source = """
    int my_rand(void)
    {
        static thrust::uniform_int_distribution<int> dist(0, 9999);
        return dist(rng);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    function = ast.statements[0]
    declaration = function.body[0]

    assert isinstance(declaration, VariableNode)
    assert declaration.qualifiers == ["static"]
    assert declaration.vtype == "thrust::uniform_int_distribution<int>"
    assert declaration.name == "dist"
    assert isinstance(declaration.value, FunctionCallNode)
    assert declaration.value.name == "thrust::uniform_int_distribution<int>"
    assert declaration.value.args == ["0", "9999"]
    assert (
        "var dist: thrust::uniform_int_distribution<int> = "
        "thrust::uniform_int_distribution<int>(0, 9999);" in crossgl
    )


def test_external_rocthrust_return_type_followed_by_hip_qualifiers_codegen_reparse():
    # Reduced from ROCm/rocThrust@8c061ed4f0628254578a3de28df775bea765f89d,
    # examples/cuda/range_view.cu.
    source = """
    template <class Iterator, class Size>
    range_view<Iterator> __host__ __device__
    make_range_view(Iterator first, Size n)
    {
        return range_view<Iterator>(first, first + n);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    function = ast.statements[0]

    assert isinstance(function, FunctionNode)
    assert function.return_type == "range_view<Iterator>"
    assert function.name == "make_range_view"
    assert function.qualifiers == ["__host__", "__device__"]
    assert "range_view<Iterator> make_range_view(Iterator first, Size n)" in crossgl
    assert "__host__" not in crossgl
    assert "__device__" not in crossgl


def test_external_rocm_device_globals_symbol_api_codegen_reparse():
    source = """
    constexpr unsigned int device_array_size = 16;
    __device__ float global;
    __device__ float global_array[device_array_size];

    __global__ void test_globals_kernel(float* out,
                                        const float* in,
                                        const size_t size) {
        const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if(tid < size) {
            out[tid] = in[tid] + global + global_array[tid % device_array_size];
        }
    }

    void host(float* d_out, float* d_in, size_t size, size_t size_bytes) {
        void* d_global{};
        size_t global_size_bytes{};
        HIP_CHECK(hipGetSymbolAddress(&d_global, HIP_SYMBOL(global)));
        HIP_CHECK(hipGetSymbolSize(&global_size_bytes, HIP_SYMBOL(global)));
        HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(global_array), d_in, size_bytes));
        test_globals_kernel<<<dim3(64), dim3(1), 0, hipStreamDefault>>>(
            d_out,
            d_in,
            size);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    global_var = ast.statements[1]
    global_array = ast.statements[2]
    assert isinstance(global_var, VariableNode)
    assert set(global_var.qualifiers) == {"__device__"}
    assert global_array.vtype == "float[device_array_size]"
    assert "var global_array: array<f32, device_array_size>;" in crossgl
    assert (
        "var tid: u32 = ((gl_WorkGroupSize.x * gl_WorkGroupID.x) + "
        "gl_LocalInvocationID.x);"
    ) in crossgl
    assert "in_[tid] + global" in crossgl
    assert (
        "// HIP get symbol address: output: d_global, symbol: HIP_SYMBOL(global)"
        in crossgl
    )
    assert (
        "// HIP get symbol size: output: global_size_bytes, symbol: HIP_SYMBOL(global)"
        in crossgl
    )
    assert (
        "// HIP symbol copy to: HIP_SYMBOL(global_array), source: d_in, "
        "bytes: size_bytes"
    ) in crossgl
    assert (
        "// Kernel launch: test_globals_kernel<<<vec3<u32>(64, 1, 1), "
        "vec3<u32>(1, 1, 1), 0, hipStreamDefault>>>()"
    ) in crossgl


def test_external_rocm_host_memory_size_multiword_integer_cast_codegen_reparse():
    # CUDA-compatible HIP host code uses the same C++ decl-specifier grammar.
    # This alternate scalar spelling appears in NVIDIA cuda-samples:
    # repo: https://github.com/NVIDIA/cuda-samples
    # commit: b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    # path: cpp/0_Introduction/matrixMulDrv/matrixMulDrv.cpp
    source = """
    void host(size_t total_global_mem) {
        double total_global_mem_in_gbytes =
            (double)(long long unsigned int)total_global_mem
            / (1024.0 * 1024.0 * 1024.0);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    outer_cast = ast.statements[0].body[0].value.left

    assert isinstance(outer_cast, CastNode)
    assert outer_cast.target_type == "double"
    assert isinstance(outer_cast.expression, CastNode)
    assert outer_cast.expression.target_type == "long long unsigned int"
    assert "f64(u64(total_global_mem))" in crossgl


def test_external_rocm_cooperative_groups_thread_group_parameter_codegen_reparse():
    source = """
    using namespace cooperative_groups;

    __device__ unsigned int reduce_sum(thread_group g,
                                       unsigned int* x,
                                       unsigned int val) {
        const unsigned int group_thread_id = g.thread_rank();

        for(unsigned int i = g.size() / 2; i > 0; i /= 2) {
            x[group_thread_id] = val;
            g.sync();

            if(group_thread_id < i) {
                val += x[group_thread_id + i];
            }

            g.sync();
        }

        if(g.thread_rank() == 0)
            return val;
        else
            return 0;
    }

    template<unsigned int PartitionSize>
    __global__ void vector_reduce_kernel(const unsigned int* d_vector,
                                         unsigned int* d_block_reduced_vector,
                                         unsigned int* d_partition_reduced_vector) {
        thread_block thread_block_group = this_thread_block();
        __shared__ unsigned int workspace[2048];
        unsigned int output;

        const unsigned int input = d_vector[thread_block_group.thread_rank()];
        output = reduce_sum(thread_block_group, workspace, input);

        if(thread_block_group.thread_rank() == 0) {
            d_block_reduced_vector[0] = output;
        }

        thread_block_tile<PartitionSize> custom_partition
            = tiled_partition<PartitionSize>(thread_block_group);
        const unsigned int group_offset
            = thread_block_group.thread_rank() - custom_partition.thread_rank();
        output = reduce_sum(custom_partition, &workspace[group_offset], input);

        if(custom_partition.thread_rank() == 0) {
            const unsigned int partition_id
                = thread_block_group.thread_rank() / PartitionSize;
            d_partition_reduced_vector[partition_id] = output;
        }
        return;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    reduce_sum = ast.statements[0]

    assert reduce_sum.params[0]["type"] == "thread_group"
    assert (
        "u32 reduce_sum(cooperative_groups_thread_group g, ptr<u32> x, u32 val)"
        in crossgl
    )
    assert (
        "cooperative_groups thread_group.thread_rank not directly supported" in crossgl
    )
    assert "cooperative_groups thread_group.size not directly supported" in crossgl
    assert "cooperative_groups thread_group.sync not directly supported" in crossgl
    assert (
        "cooperative_groups thread_block thread_block_group maps to the current "
        "workgroup"
    ) in crossgl
    assert (
        "cooperative_groups thread_block_tile<PartitionSize> custom_partition"
        in crossgl
    )
    assert "var<workgroup> workspace: array<u32, 2048>;" in crossgl
    assert "g.sync()" not in crossgl
    assert "g.thread_rank()" not in crossgl


def test_external_hip_tests_thread_block_tile_shuffle_codegen_reparse():
    # Upstream: ROCm/hip-tests@8889ba5c7a89a85d5262dadcfbde17589a53ccfb,
    # catch/unit/cooperativeGrps/hipCGThreadBlockTileTypeShfl_old.cc.
    source = """
    namespace cg = cooperative_groups;

    template <unsigned int tileSz>
    __device__ int reduction_kernel_shfl(cg::thread_block_tile<tileSz> const& g,
                                         int val) {
        int sz = g.size();

        for(int i = sz / 2; i > 0; i >>= 1) {
            val += g.shfl_down(val, i);
            val += g.shfl_xor(val, i);
            val += g.shfl_up(val, i);
        }

        return val;
    }

    template <unsigned int tile_size>
    static __global__ void kernel_cg_group_partition_static(int* result) {
        cg::thread_block thread_block_CG_ty = cg::this_thread_block();
        cg::thread_block_tile<tile_size> tiled_part =
            cg::tiled_partition<tile_size>(thread_block_CG_ty);

        int input = tiled_part.thread_rank();
        result[thread_block_CG_ty.thread_rank()] =
            reduction_kernel_shfl(tiled_part, input);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    helper = ast.statements[0]

    assert helper.params[0]["type"] == "cg::thread_block_tile<tileSz> const &"
    assert "cooperative_groups_thread_block_tile_tileSz g" in crossgl
    assert "var sz: i32 = tileSz;" in crossgl
    assert "WaveReadLaneAt(val, (WaveGetLaneIndex() + (i)))" in crossgl
    assert (
        "WaveReadLaneAt(val, ((WaveGetLaneIndex() - "
        "(WaveGetLaneIndex() % tileSz)) + ((WaveGetLaneIndex() % tileSz) ^ (i))))"
        in crossgl
    )
    assert "WaveReadLaneAt(val, (WaveGetLaneIndex() - (i)))" in crossgl
    assert (
        "cooperative_groups thread_block_tile.shfl_down not directly supported"
        not in crossgl
    )
    assert (
        "cooperative_groups thread_block_tile.shfl_xor not directly supported"
        not in crossgl
    )
    assert (
        "cooperative_groups thread_block_tile.shfl_up not directly supported"
        not in crossgl
    )


def test_external_hip_tests_parenthesized_scoped_enum_case_codegen_reparse():
    # Upstream: ROCm/hip-tests@8889ba5c7a89a85d5262dadcfbde17589a53ccfb,
    # catch/unit/cooperativeGrps/hipCGThreadBlockTileTypeShfl_old.cc.
    source = """
    enum class TiledGroupShflTests { shflDown, shflXor };

    __global__ void kernel(TiledGroupShflTests shfl_test, int* out) {
        switch (shfl_test) {
            case (TiledGroupShflTests::shflDown):
                out[0] = 1;
                break;
            case (TiledGroupShflTests::shflXor):
                out[0] = 2;
                break;
        }
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    switch = ast.statements[1].body[0]

    assert isinstance(switch, SwitchNode)
    assert switch.cases[0].value == "TiledGroupShflTests::shflDown"
    assert "case TiledGroupShflTests::shflDown:" in crossgl
    assert "case TiledGroupShflTests::shflXor:" in crossgl


def test_external_rocm_saxpy_kernel_launch_crossgl_reparse():
    source = """
    __global__ void saxpy_kernel(
        const float a,
        const float* d_x,
        float* d_y,
        const unsigned int size) {
        const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(global_idx < size) {
            d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
        }
    }

    void host(float a, float* d_x, float* d_y, unsigned int size) {
        saxpy_kernel<<<dim3(ceiling_div(size, 256)),
                       dim3(256),
                       0,
                       hipStreamDefault>>>(a, d_x, d_y, size);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    launch = ast.statements[1].body[0]
    assert isinstance(launch, KernelLaunchNode)
    assert launch.kernel_name == "saxpy_kernel"
    assert "fn saxpy_kernel(" in crossgl
    assert "Kernel launch: saxpy_kernel<<<" in crossgl


def test_external_rocm_platform_error_guards_default_codegen_reparse():
    # Reduced from ROCm/rocm-examples@cf369da68f209c315074204bd0eb61d1a5c015d1:
    # HIP-Basic/vulkan_interop/main.hip,
    # HIP-Basic/vulkan_interop_mipmap/main.hip, and
    # HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/
    # identifying_compilation_target_platform/main.cpp.
    source = """
    #if defined(__HIP_PLATFORM_AMD__)
    static int selected_hip_platform() { return 1; }
    #elif defined(__HIP_PLATFORM_NVIDIA__)
    static int selected_hip_platform() { return 2; }
    #else
    #error unsupported platform
    #endif

    #if defined(_WIN32)
    static int selected_host_platform() { return 3; }
    #elif defined(__linux__)
    static int selected_host_platform() { return 4; }
    #else
    #error unsupported host
    #endif

    void host(int* out) {
        out[0] = selected_hip_platform() + selected_host_platform();
    }
    """

    _, crossgl = assert_crossgl_reparses(source)

    assert "i32 selected_hip_platform()" in crossgl
    assert "return 1;" in crossgl
    assert "return 2;" not in crossgl
    assert "i32 selected_host_platform()" in crossgl
    assert "return 4;" in crossgl
    assert "return 3;" not in crossgl
    assert "unsupported" not in crossgl


def test_external_rocm_runtime_compilation_adjacent_raw_string_codegen_reparse():
    # Upstream: ROCm/rocm-examples@d3ad835e46ff50412cf51086df7400fb3bbd1649,
    # HIP-Basic/runtime_compilation/main.hip.
    source = r"""
    static constexpr auto saxpy_kernel{
        "#include \"test_header.h\"\n"
        "#include \"test_header1.h\"\n"
        R"(
extern "C" __global__ void saxpy_kernel()
{
}
)"};

    int main() {
        hiprtcProgram prog;
        hiprtcCreateProgram(&prog,
                            saxpy_kernel,
                            "saxpy_kernel.cu",
                            0,
                            nullptr,
                            nullptr);
        return 0;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    declaration = ast.statements[0]
    assert isinstance(declaration, VariableNode)
    assert "test_header1.h" in declaration.value.elements[0]
    assert 'extern "C" __global__ void saxpy_kernel' in declaration.value.elements[0]
    assert 'R"(' not in crossgl
    assert ')"' not in crossgl
    assert "var saxpy_kernel: auto = " in crossgl
    assert 'extern \\"C\\" __global__ void saxpy_kernel' in crossgl
    assert "HIPRTC create program: output: prog, source: saxpy_kernel" in crossgl
    assert 'name: "saxpy_kernel.cu"' in crossgl


def test_external_rocm_hiprtc_linker_commented_macro_call_codegen_reparse():
    # Upstream shape: ROCm/rocm-examples, Programming-for-HIP-Runtime-Compiler
    # linker API examples. The HIPRTC_CHECK invocation keeps line comments on
    # split arguments, which must not truncate the preprocessed macro call.
    source = r"""
    #define CHECK_RET_CODE(call, ret_code)                                      \
    {                                                                           \
        if ((call) != ret_code)                                                 \
        {                                                                       \
            std::abort();                                                       \
        }                                                                       \
    }
    #define HIPRTC_CHECK(call) CHECK_RET_CODE(call, HIPRTC_SUCCESS)

    void linkWithComments(hiprtcJIT_option* options, void** option_vals) {
        auto num_options = 0u;
        auto rtc_link_state = hiprtcLinkState{};
        HIPRTC_CHECK(hiprtcLinkCreate(num_options,        // number of options
                                      options,            // options
                                      option_vals,        // option values
                                      &rtc_link_state));  // state output
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    link_function = ast.statements[0]
    assert isinstance(link_function, FunctionNode)
    assert len(link_function.body) == 3
    assert (
        "HIPRTC link create: options: num_options, option keys: options, "
        "option values: option_vals, state output: rtc_link_state"
    ) in crossgl


def test_external_rocm_convolution_std_array_template_extent_codegen_reparse():
    source = """
    const constexpr std::array<float, 5 * 5> convolution_filter_5x5 = {
        1.0f, 3.0f, 0.0f, -2.0f, -0.0f
    };

    __constant__ float d_mask[5 * 5];

    template<size_t MaskWidth = 5>
    __global__ void convolution(const float* input,
                                float* output,
                                const uint2 input_dimensions) {
        const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const size_t width = input_dimensions.x;
        if(x < width) {
            output[x] = input[x] * d_mask[x % MaskWidth];
        }
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    host_filter = ast.statements[0]
    assert isinstance(host_filter, VariableNode)
    assert host_filter.vtype == "const std::array<float, 5*5>"
    assert set(host_filter.qualifiers) == {"constexpr"}
    assert (
        "var convolution_filter_5x5: array<f32, hip_array_extent_5_mul_5> = "
        "{1.0f, 3.0f, 0.0f, (-2.0f), (-0.0f)};"
    ) in crossgl
    assert "ptr<std::array" not in crossgl
    assert (
        "@group(0) @binding(0) var<uniform> d_mask: "
        "array<f32, hip_array_extent_5_mul_5>;"
    ) in crossgl
    assert "fn convolution(" in crossgl


def test_external_rocm_stb_image_trailing_aligned_attribute_codegen_reparse():
    # Upstream: ROCm/rocm-examples image convolution and histogram_atomics
    # vendored stb_image.h define STBI_SIMD_ALIGN as:
    # type name __attribute__((aligned(16))).
    source = """
    void stbi_filter_tile() {
        unsigned char temp[16] __attribute__((aligned(16)));
        temp[0] = 1;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    temp = ast.statements[0].body[0]

    assert isinstance(temp, VariableNode)
    assert temp.vtype == "unsigned char[16]"
    assert "var temp: array<u8, 16>;" in crossgl
    assert "temp[0] = 1;" in crossgl
    assert "__attribute__" not in crossgl
    assert "aligned" not in crossgl


def test_external_rocm_stb_image_write_function_type_typedef_codegen_reparse():
    # Upstream: ROCm/rocm-examples@d3ad835e46ff50412cf51086df7400fb3bbd1649,
    # HIP-Doc/Tutorials/Programming-Patterns/image_convolution/stb_image_write.h.
    source = """
    typedef void stbi_write_func(void *context, void *data, int size);

    extern int stbi_write_png_to_func(stbi_write_func *func,
                                      void *context,
                                      int w,
                                      int h);
    """

    ast, crossgl = assert_crossgl_reparses(source)
    alias = ast.statements[0]
    function = ast.statements[1]

    assert isinstance(alias, TypeAliasNode)
    assert alias.name == "stbi_write_func"
    assert alias.alias_type == "void"
    assert function.params[0]["type"] == "stbi_write_func *"
    assert "typedef void stbi_write_func;" in crossgl
    assert "i32 stbi_write_png_to_func(ptr<stbi_write_func> func" in crossgl


def test_external_hip_examples_function_pointer_parameter_codegen_reparse():
    # Upstream: ROCm/HIP-Examples@cdf9d101acd9a3fc89ee750f73c1f1958cbd5cc3,
    # HIP-Examples-Applications/common/SDKUtil.hpp.
    source = """
    template <class T>
    std::string toString(T t,
                         std::ios_base& (*r)(std::ios_base&) = std::dec) {
        std::ostringstream output;
        output << r << t;
        return output.str();
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    function = ast.statements[0]

    assert isinstance(function, FunctionNode)
    assert function.params[1]["type"] == "std::ios_base & (*)"
    assert "std::string toString(T t, ptr<std::ios_base> r)" in crossgl
    assert "ptr<std::ios_base ()>" not in crossgl
    assert "var output: std::ostringstream;" in crossgl


def test_external_hip_one_component_vectors_codegen_reparse():
    # HIP exposes CUDA-compatible one-component vector types through
    # hip_vector_types.h; they should lower to scalar CrossGL values.
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

    ast, crossgl = assert_crossgl_reparses(source)
    function = ast.statements[1]

    assert function.params[1]["type"] == "float1"
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


def test_external_hip_tests_catch_test_case_block_codegen_reparse():
    # Upstream: ROCm/hip-tests@d01e1f96059edc25600eb13434d7e2b71c09af01,
    # catch/unit/deviceLib/funnelshift.cc.
    source = """
    #define NUM_TESTS 65

    __global__ void funnelshift_kernel(unsigned int* l_out) {
        for(int i = 0; i < NUM_TESTS; i++) {
            l_out[i] = __funnelshift_l(0xdeadbeef, 0xfacefeed, i);
        }
    }

    HIP_TEST_CASE(Unit_funnelshift) {
        unsigned int* host_l_output;
        unsigned int* device_l_output;

        host_l_output = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));
        HIP_CHECK(hipMalloc((void**)&device_l_output,
                            NUM_TESTS * sizeof(unsigned int)));
        hipLaunchKernelGGL(funnelshift_kernel,
                           dim3(1),
                           dim3(1),
                           0,
                           0,
                           device_l_output);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    test_case = ast.statements[1]

    assert isinstance(test_case, FunctionNode)
    assert test_case.name == "Unit_funnelshift"
    assert test_case.qualifiers == ["HIP_TEST_CASE"]
    assert isinstance(test_case.body[0], VariableNode)
    assert test_case.body[0].vtype == "unsigned int *"
    assert "void Unit_funnelshift()" in crossgl
    assert "var host_l_output: ptr<u32>;" in crossgl
    assert (
        "// Kernel launch: funnelshift_kernel<<<vec3<u32>(1, 1, 1), "
        "vec3<u32>(1, 1, 1), 0, 0>>>()"
    ) in crossgl


def test_external_hip_tests_nested_section_block_codegen_reparse():
    # Upstream: ROCm/hip-tests@8889ba5c7a89a85d5262dadcfbde17589a53ccfb,
    # catch/multiproc/hipIpcEventHandle.cc.
    source = """
    HIP_TEST_CASE(Unit_hipIpcEventHandle_ParameterValidation) {
        hipIpcEventHandle_t eventHandle;
        hipError_t ret;

        SECTION("Get event handle with event(nullptr)") {
            ret = hipIpcGetEventHandle(&eventHandle, nullptr);
            REQUIRE(ret == hipErrorInvalidValue);
        }
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    test_case = ast.statements[0]

    assert isinstance(test_case, FunctionNode)
    assert test_case.name == "Unit_hipIpcEventHandle_ParameterValidation"
    assert test_case.qualifiers == ["HIP_TEST_CASE"]
    assert len(test_case.body) == 4
    assert isinstance(test_case.body[0], VariableNode)
    assert test_case.body[0].vtype == "hipIpcEventHandle_t"
    assert isinstance(test_case.body[1], VariableNode)
    assert test_case.body[1].vtype == "hipError_t"
    assert "SECTION" not in crossgl
    assert "HIP IPC get event handle: output: eventHandle, event: nullptr" in crossgl
    assert "REQUIRE((ret == hipErrorInvalidValue));" in crossgl


def test_external_hip_tests_inline_constant_global_codegen_reparse():
    # Upstream: ROCm/hip-tests@d01e1f96059edc25600eb13434d7e2b71c09af01,
    # catch/unit/memory/memoryGlobal.hh and catch/unit/memory/memoryCommon.cc.
    source = """
    inline __constant__ int globalVar;

    void set_value(int const value) {
        HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(globalVar), &value, sizeof(value)));
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    global_var = ast.statements[0]

    assert isinstance(global_var, VariableNode)
    assert global_var.name == "globalVar"
    assert global_var.vtype == "int"
    assert global_var.qualifiers == ["inline", "__constant__"]
    assert "@group(0) @binding(0) var<uniform> globalVar: i32;" in crossgl
    assert "HIP symbol copy to: HIP_SYMBOL(globalVar)" in crossgl


def test_external_hip_tests_bare_check_image_support_macro_codegen_reparse():
    # Upstream: ROCm/hip-tests@d01e1f96059edc25600eb13434d7e2b71c09af01,
    # catch/unit/memory/hipFree.cc.
    source = """
    HIP_TEMPLATE_TEST_CASE(Unit_hipFreeImplicitSyncArray, char, float, float2, float4) {
        CHECK_IMAGE_SUPPORT

        hipArray_t arrayPtr{};
        hipExtent extent{};
        extent.width = GENERATE(32, 128, 256, 512, 1024);
        hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

        HIP_CHECK(hipMallocArray(&arrayPtr, &desc, extent.width, extent.height, hipArrayDefault));
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    test_case = ast.statements[0]

    assert isinstance(test_case, FunctionNode)
    assert test_case.name == "Unit_hipFreeImplicitSyncArray"
    assert test_case.qualifiers == ["HIP_TEMPLATE_TEST_CASE"]
    assert isinstance(test_case.body[0], VariableNode)
    assert test_case.body[0].vtype == "hipArray_t"
    assert test_case.body[0].name == "arrayPtr"
    assert "CHECK_IMAGE_SUPPORT" not in crossgl
    assert "var arrayPtr: ptr<void> = {};" in crossgl
    assert (
        "HIP array allocate: arrayPtr, desc: desc, width: extent.width, "
        "height: extent.height, flags: hipArrayDefault"
    ) in crossgl


def test_external_rocm_inline_assembly_kernel_codegen_reparse():
    source = """
    __global__ void matrix_transpose_kernel(float* out,
                                            const float* in,
                                            const unsigned int width) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;

        asm volatile("v_mov_b32_e32 %0, %1"
                     : "=v"(out[x * width + y])
                     : "v"(in[y * width + x]));
    }

    void host(float* d_transposed_matrix, float* d_matrix, unsigned int width) {
        matrix_transpose_kernel<<<dim3(width / 8, width / 8),
                                  dim3(8, 8),
                                  0,
                                  hipStreamDefault>>>(
            d_transposed_matrix,
            d_matrix,
            width);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    asm_stmt = ast.statements[0].body[2]
    launch = ast.statements[1].body[0]

    assert isinstance(asm_stmt, HipAsmNode)
    assert asm_stmt.is_volatile is True
    assert asm_stmt.template == '"v_mov_b32_e32 %0, %1"'
    assert asm_stmt.outputs[0].constraint == '"=v"'
    assert asm_stmt.outputs[0].expression.array == "out"
    assert asm_stmt.inputs[0].constraint == '"v"'
    assert asm_stmt.inputs[0].expression.array == "in"
    assert isinstance(launch, KernelLaunchNode)
    assert '// HIP inline assembly volatile: "v_mov_b32_e32 %0, %1"' in crossgl
    assert '// HIP inline assembly outputs: "=v"(out[((x * width) + y)])' in crossgl
    assert (
        "// Kernel launch: matrix_transpose_kernel<<<vec3<u32>((width / 8), "
        "(width / 8), 1), vec3<u32>(8, 8, 1), 0, hipStreamDefault>>>()" in crossgl
    )


def test_external_rocm_reduction_nested_template_static_for_crossgl_reparse():
    source = """
    template<uint32_t WarpCount, uint32_t WarpSize>
    __global__ void kernel(unsigned* out) {
        unsigned res = 0;
        tmp::static_for<WarpCount,
                        tmp::not_equal<0>,
                        tmp::select<tmp::not_equal<1>,
                                    tmp::divide_ceil<WarpSize>,
                                    tmp::constant<0>>>(
            [&]<uint32_t ActiveWarps>()
            {
                if(threadIdx.x < ActiveWarps) {
                    res = max(res, __shfl_down(res, 1));
                }
            });
        out[threadIdx.x] = res;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    static_for = ast.statements[0].body[1]
    assert isinstance(static_for, FunctionCallNode)
    assert static_for.name == (
        "tmp::static_for<WarpCount, tmp::not_equal<0>, "
        "tmp::select<tmp::not_equal<1>, tmp::divide_ceil<WarpSize>, "
        "tmp::constant<0>>>"
    )
    assert "tmp::static_for<WarpCount, tmp::not_equal<0>" in crossgl


def test_external_rocm_reduction_digit_separator_literals_crossgl_reparse():
    source = """
    __global__ void kernel(unsigned* out) {
        constexpr unsigned input_count = 100'000'000;
        out[0] = input_count;
    }
    """

    _, crossgl = assert_crossgl_reparses(source)

    assert "100000000" in crossgl
    assert "100'000'000" not in crossgl


def test_external_rocm_occupancy_unsigned_long_suffix_literals_crossgl_reparse():
    # Upstream: https://github.com/ROCm/rocm-examples
    # Commit: adaf64a066eecb4ad90036dfd1838fc95bed9914
    # Path: Tools/rocprof-compute/occupancy.hip
    source = """
    constexpr size_t fully_allocate_lds = 64ul * 1024ul / sizeof(double);

    __launch_bounds__(256)
    __global__ void ldsbound(double* ptr) {
        __shared__ double intermediates[fully_allocate_lds];
        double x = ptr[threadIdx.x];
        ptr[threadIdx.x] = x;
    }
    """

    _, crossgl = assert_crossgl_reparses(source)

    assert "var fully_allocate_lds: u32 = ((64u * 1024u) / sizeof(double));" in crossgl
    assert "64ul" not in crossgl
    assert "1024ul" not in crossgl


def test_external_rocm_vulkan_interop_numeric_limits_max_codegen_reparse():
    # Upstream: https://github.com/ROCm/rocm-examples
    # Commit: adaf64a066eecb4ad90036dfd1838fc95bed9914
    # Path: HIP-Basic/vulkan_interop/main.hip
    source = """
    constexpr uint64_t frame_timeout = std::numeric_limits<uint64_t>::max();
    """

    ast, crossgl = assert_crossgl_reparses(source)
    declaration = ast.statements[0]

    assert isinstance(declaration, VariableNode)
    assert declaration.name == "frame_timeout"
    assert declaration.vtype == "uint64_t"
    assert declaration.value.name == "std::numeric_limits<uint64_t>::max"
    assert "var frame_timeout: u64 = numeric_limits_max<u64>();" in crossgl
    assert "std::numeric_limits<uint64_t>::max()" not in crossgl


def test_external_rocm_graph_api_variable_template_constant_crossgl_reparse():
    source = """
    __global__ void filter_creation_kernel(float* __restrict__ r,
                                           int N_hFFT,
                                           float tau) {
        constexpr auto pi = std::numbers::pi_v<float>;
        auto const i = blockDim.x * blockIdx.x + threadIdx.x;
        if(i < static_cast<unsigned int>(N_hFFT)) {
            auto const x = pi * i / N_hFFT;
            r[i] = x / tau;
        }
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    pi = ast.statements[0].body[0]

    assert isinstance(pi, VariableNode)
    assert pi.value == "std::numbers::pi_v<float>"
    assert "var pi: auto = std::numbers::pi_v<float>;" in crossgl
    assert "var x: auto = ((pi * i) / N_hFFT);" in crossgl


def test_external_rocm_graph_api_variable_template_arithmetic_crossgl_reparse():
    # Upstream: ROCm/rocm-examples@cf369da68f209c315074204bd0eb61d1a5c015d1,
    # HIP-Doc/Tutorials/graph_api/src/phantom.hip.
    source = """
    struct spheroid {
        float theta;
    };

    constexpr __device__ spheroid shepp_logan_phantom[] = {{0.f}};

    __global__ void create_phantom_kernel(float* out) {
        auto theta_rad =
            -shepp_logan_phantom[0].theta * std::numbers::pi_v<float> / 180.0f;
        out[0] = theta_rad;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    theta_rad = ast.statements[2].body[0]

    assert isinstance(theta_rad, VariableNode)
    assert isinstance(theta_rad.value, BinaryOpNode)
    assert theta_rad.value.op == "/"
    assert isinstance(theta_rad.value.left, BinaryOpNode)
    assert theta_rad.value.left.op == "*"
    assert theta_rad.value.left.right == "std::numbers::pi_v<float>"
    assert (
        "var theta_rad: auto = (((-shepp_logan_phantom[0].theta) * "
        "std::numbers::pi_v<float>) / 180.0f);"
    ) in crossgl


def test_external_rocm_docs_system_scope_atomics_codegen_reparse():
    # Upstream source:
    # ROCm HIP C++ language extensions atomic functions table. ROCm examples
    # exercise ordinary atomics in kernels; HIP docs list the _system variants.
    source = """
    __global__ void scoped_atomic_kernel(unsigned int* bins,
                                         unsigned int* flags) {
        unsigned int old = atomicAdd_system(&bins[threadIdx.x], 1u);
        atomicOr_system(flags, old);
        unsigned int exchanged = atomicCAS_system(flags, 0u, 1u);
        bins[threadIdx.x] = exchanged;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    body = ast.statements[0].body

    assert isinstance(body[0].value, AtomicOperationNode)
    assert body[0].value.operation == "atomicAdd_system"
    assert isinstance(body[1], AtomicOperationNode)
    assert body[1].operation == "atomicOr_system"
    assert isinstance(body[2].value, AtomicOperationNode)
    assert body[2].value.operation == "atomicCAS_system"
    assert "hip system-scope atomic atomicAdd_system lowered to atomicAdd" in crossgl
    assert "hip system-scope atomic atomicOr_system lowered to atomicOr" in crossgl
    assert (
        "hip system-scope atomic atomicCAS_system lowered to " "atomicCompareExchange"
    ) in crossgl
    assert "atomicAdd(bins[gl_LocalInvocationID.x], 1u)" in crossgl
    assert "atomicOr(flags, old)" in crossgl
    assert "atomicCompareExchange(flags, 0u, 1u)" in crossgl
    assert "atomicAdd_system(" not in crossgl
    assert "atomicOr_system(" not in crossgl
    assert "atomicCAS_system(" not in crossgl


def test_external_hip_tests_bare_atomic_compat_block_macro_codegen_reparse():
    # Upstream: https://github.com/ROCm/hip-tests
    # Commit: 8889ba5c7a89a85d5262dadcfbde17589a53ccfb
    # Path: catch/unit/atomics/arithmetic_common.hh
    source = """
    __device__ int perform_atomic_add(int* mem, int val) {
        if(val > 0) {
            HIP_TEST_ATOMIC_BACKWARD_COMPAT_MEMORY {
                return __hip_atomic_fetch_add(
                    mem,
                    val,
                    __ATOMIC_RELAXED,
                    __HIP_MEMORY_SCOPE_AGENT);
            }
        }
        return 0;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    branch = ast.statements[0].body[0]

    assert isinstance(branch, IfNode)
    assert len(branch.if_body) == 1
    assert "HIP_TEST_ATOMIC_BACKWARD_COMPAT_MEMORY" not in crossgl
    assert "__hip_atomic_fetch_add" in crossgl
    assert "return 0;" in crossgl


def test_external_hip_tests_newline_split_range_for_initializer_codegen_reparse():
    # Upstream: https://github.com/ROCm/hip-tests
    # Commit: 8889ba5c7a89a85d5262dadcfbde17589a53ccfb
    # Path: catch/unit/atomics/arithmetic_common.hh
    source = """
    void host() {
        using LA = LinearAllocs;
        for (const auto alloc_type :
             {LA::hipMalloc}) {
            sink(alloc_type);
        }
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    loop = ast.statements[0].body[1]

    assert loop.vtype == "const auto"
    assert loop.name == "alloc_type"
    assert "for alloc_type in {LA::hipMalloc}" in crossgl


def test_external_hip_tests_user_function_template_call_codegen_reparse():
    # Upstream: https://github.com/ROCm/hip-tests
    # Commit: 8889ba5c7a89a85d5262dadcfbde17589a53ccfb
    # Path: catch/unit/atomics/arithmetic_common.hh
    source = """
    struct TestParams {
        int value;
    };

    template <typename TestType, int operation, bool use_shared_mem, int memory_scope>
    void TestCore(TestParams params) {
        sink(params.value);
    }

    template <typename TestType, int operation, int memory_scope>
    void host(TestParams params) {
        TestCore<TestType, operation, false, memory_scope>(params);
    }
    """

    _, crossgl = assert_crossgl_reparses(source)

    assert "TestCore(params);" in crossgl
    assert "TestCore<TestType" not in crossgl


def test_external_hip_tests_user_function_template_reference_codegen_reparse():
    # Upstream: https://github.com/ROCm/hip-tests
    # Commit: 8889ba5c7a89a85d5262dadcfbde17589a53ccfb
    # Path: catch/unit/atomics/arithmetic_common.hh
    source = """
    template <typename TestType, int operation>
    void HostAtomicOperation(int iterations) {
        sink(iterations);
    }

    template <typename TestType, int operation>
    void host(int iterations) {
        std::thread(HostAtomicOperation<TestType, operation>, iterations);
    }
    """

    _, crossgl = assert_crossgl_reparses(source)

    assert "std::thread(HostAtomicOperation, iterations);" in crossgl
    assert "HostAtomicOperation<TestType" not in crossgl


def test_external_hip_tests_tuple_of_vectors_return_type_codegen_reparse():
    # Upstream: https://github.com/ROCm/hip-tests
    # Commit: 8889ba5c7a89a85d5262dadcfbde17589a53ccfb
    # Path: catch/unit/atomics/arithmetic_common.hh
    source = """
    template <typename TestType>
    std::tuple<std::vector<TestType>, std::vector<TestType>>
    TestKernelHostRef(TestParams p) {
        std::vector<TestType> old_vals(p.width);
        std::vector<TestType> res_vals(p.width);
        return std::make_tuple(res_vals, old_vals);
    }
    """

    _, crossgl = assert_crossgl_reparses(source)

    assert "tuple<array<TestType>, array<TestType>> TestKernelHostRef" in crossgl
    assert "std::tuple" not in crossgl


def test_external_hip_tests_pair_of_vectors_return_type_codegen_reparse():
    # Reduced from ROCm/hip-tests@a618b48f0a29cfbd8c990fa72ab772483f61d381:
    # catch/unit/deviceLib/hadd.cc.
    source = """
    static auto get_hadd_inputs() -> std::pair<std::vector<int>, std::vector<int>> {
        std::vector<int> a, b;
        return std::make_pair(a, b);
    }
    """

    _, crossgl = assert_crossgl_reparses(source)

    assert "pair<array<i32>, array<i32>> get_hadd_inputs()" in crossgl
    assert "std::pair" not in crossgl
    assert "std::vector" not in crossgl


def test_external_hipblas_template_template_parameter_codegen_reparse():
    # Upstream: https://github.com/ROCm/hipBLAS
    # Commit: 4a1f902127ba378e1e22600e17de5894327e28d5
    # Path: clients/gtest/auxil/set_get_mode_gtest.cpp
    source = """
    enum aux_mode_test_type { SG_POINTER, SG_ATOMICS };

    template <template <typename...> class FILTER, aux_mode_test_type AUX_TYPE>
    void host(aux_mode_test_type value) {
        if(value == AUX_TYPE) {
            sink();
        }
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    function = ast.statements[1]

    assert isinstance(function, FunctionNode)
    assert function.name == "host"
    assert "void host(aux_mode_test_type value)" in crossgl
    assert "if ((value == AUX_TYPE))" in crossgl


def test_external_rocm_rocdecode_typedef_enum_codegen_reparse():
    # Upstream: ROCm/rocm-examples@d3ad835e46ff50412cf51086df7400fb3bbd1649,
    # Common/rocdecode_utils.hpp.
    source = """
    typedef enum reconfigure_flush_mode_enum {
        RECONFIG_FLUSH_MODE_NONE = 0x0,
        RECONFIG_FLUSH_MODE_DUMP_TO_FILE = 0x1,
        RECONFIG_FLUSH_MODE_CALCULATE_MD5 = (0x1 << 1),
    } reconfigure_flush_mode;

    bool should_calculate_md5(reconfigure_flush_mode flush_mode) {
        return flush_mode
            == reconfigure_flush_mode::RECONFIG_FLUSH_MODE_CALCULATE_MD5;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    enum = ast.statements[0]
    function = ast.statements[1]

    assert enum.name == "reconfigure_flush_mode"
    assert enum.members[0] == ("RECONFIG_FLUSH_MODE_NONE", "0x0")
    assert enum.members[1] == ("RECONFIG_FLUSH_MODE_DUMP_TO_FILE", "0x1")
    assert isinstance(enum.members[2][1], BinaryOpNode)
    assert function.params[0]["type"] == "reconfigure_flush_mode"
    assert "enum reconfigure_flush_mode {" in crossgl
    assert "RECONFIG_FLUSH_MODE_CALCULATE_MD5 = (0x1 << 1)," in crossgl
    assert "bool should_calculate_md5(reconfigure_flush_mode flush_mode)" in crossgl


def test_external_hip_examples_histogram_dynamic_shared_crossgl_reparse():
    source = """
    #define BIN_SIZE 256
    __global__ void histogram256(unsigned int* data, unsigned int* binResult) {
        HIP_DYNAMIC_SHARED(unsigned char, sharedArray);
        size_t localId = hipThreadIdx_x;
        uchar4* input = (uchar4*)sharedArray;
        for(int i = 0; i < 64; ++i) {
            input[localId + i] = make_uchar4(0, 0, 0, 0);
        }
        __syncthreads();
        uint4 binCount = make_uint4(0, 0, 0, 0);
        uint result = binCount.x + binCount.y + binCount.z + binCount.w;
        binResult[localId] = result;
    }
    """

    _, crossgl = assert_crossgl_reparses(source)

    assert "var<workgroup> sharedArray: array<u8>;" in crossgl
    assert "var input: ptr<vec4<u8>> = ptr<vec4<u8>>(sharedArray);" in crossgl
    assert "workgroupBarrier();" in crossgl
    assert "var binCount: vec4<u32> = vec4<u32>(0, 0, 0, 0);" in crossgl


def test_external_hip_examples_histogram_newline_split_c_style_cast_reparse():
    source = """
    __global__ void histogram256(unsigned int* binResult) {
        HIP_DYNAMIC_SHARED(unsigned char, sharedArray);
        uchar4* input = (uchar4*)
            sharedArray;
        uint result = input[threadIdx.x].x;
        binResult[threadIdx.x] = result;
    }
    """

    _, crossgl = assert_crossgl_reparses(source)

    assert "var input: ptr<vec4<u8>> = ptr<vec4<u8>>(sharedArray);" in crossgl
    assert "sharedArray;" not in crossgl


def test_external_rocwmma_float_aliases_codegen_reparse():
    # Upstream: ROCm/rocWMMA@1e3bed23f981d8ca0ce1b8634e6b91b5ccf91cfb,
    # samples/simple_sgemm.cpp, samples/simple_hgemm.cpp, and
    # samples/simple_dgemm.cpp.
    source = """
    using rocwmma::float16_t;
    using rocwmma::float32_t;
    using rocwmma::float64_t;

    __global__ void sgemm_rocwmma_d(uint32_t m,
                                    float32_t const* a,
                                    float32_t* d,
                                    float32_t alpha) {
        d[threadIdx.x] = a[threadIdx.x] * alpha;
    }

    __host__ void gemm_test(float32_t alpha, float64_t beta) {
        float16_t* d_half;
        float32_t* d_a;
        float64_t elapsedTimeMs = 0.0;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    kernel = ast.statements[0]
    host = ast.statements[1]

    assert kernel.params[1]["type"] == "float32_t const *"
    assert kernel.params[2]["type"] == "float32_t *"
    assert kernel.params[3]["type"] == "float32_t"
    assert [declaration.vtype for declaration in host.body] == [
        "float16_t *",
        "float32_t *",
        "float64_t",
    ]
    assert "@group(0) @binding(1) var<storage, read_write> a: array<f32>" in crossgl
    assert "@group(0) @binding(2) var<storage, read_write> d: array<f32>" in crossgl
    assert "f32 alpha" in crossgl
    assert "f64 beta" in crossgl
    assert "var d_half: ptr<f16>;" in crossgl
    assert "var d_a: ptr<f32>;" in crossgl
    assert "var elapsedTimeMs: f64 = 0.0;" in crossgl
    assert "float16_t" not in crossgl
    assert "float32_t" not in crossgl
    assert "float64_t" not in crossgl


def test_external_rocwmma_device_macro_auto_helper_codegen_reparse():
    # Upstream: ROCm/rocWMMA@1e3bed23f981d8ca0ce1b8634e6b91b5ccf91cfb,
    # samples/perf_sgemm.cpp, transformGRFragAToLWFragA.
    source = """
    ROCWMMA_DEVICE auto transformGRFragAToLWFragA(GRFragA const& grFragA) {
        return apply_data_layout<DataLayoutLds>(grFragA);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    function = ast.statements[0]
    return_call = function.body[0].value

    assert isinstance(function, FunctionNode)
    assert function.return_type == "auto"
    assert function.name == "transformGRFragAToLWFragA"
    assert function.qualifiers == ["ROCWMMA_DEVICE"]
    assert function.params == [{"type": "GRFragA const &", "name": "grFragA"}]
    assert isinstance(return_call, FunctionCallNode)
    assert return_call.name == "apply_data_layout<DataLayoutLds>"
    assert "auto transformGRFragAToLWFragA(GRFragA grFragA)" in crossgl
    assert "return apply_data_layout<DataLayoutLds>(grFragA);" in crossgl
    assert "ROCWMMA_DEVICE" not in crossgl


def test_external_hpc_training_double2_launch_codegen():
    source = """
    __launch_bounds__(256,1)
    __global__ void kernel_5(
        double2* d_x,
        double2* d_y,
        double2* d_z,
        double a,
        size_t N) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= N) return;
        d_z[idx] = d_x[idx] * a + d_y[idx];
    }

    void host(double* d_x, double* d_y, double* d_z, double a, size_t N) {
        dim3 grid_5(((N/2)+255)/256, 1, 1);
        dim3 block_5(256, 1, 1);
        kernel_5 <<< grid_5, block_5 >>> (
            (double2*)d_x, (double2*)d_y, (double2*)d_z, a, N/2);
    }
    """

    ast, crossgl = generate_crossgl_from_hip(source)

    launch = ast.statements[1].body[2]
    assert isinstance(launch, KernelLaunchNode)
    assert launch.kernel_name == "kernel_5"
    assert "// HIP launch bounds: (256, 1)" in crossgl
    assert "array<vec2<f64>>" in crossgl
    assert "ptr<vec2<f64>>(d_x)" in crossgl


def test_external_llvm_amdgpu_attribute_before_global_codegen_reparse():
    # Upstream:
    # repo: https://github.com/llvm/llvm-project
    # commit: 3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff
    # path: clang/test/SemaCUDA/amdgpu-attrs.cu
    source = """
    __attribute__((amdgpu_flat_work_group_size(32, 64)))
    __global__ void flat_work_group_size_32_64(float* out) {
        out[threadIdx.x] = 1.0f;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    kernel = ast.statements[0]

    assert isinstance(kernel, KernelNode)
    assert kernel.name == "flat_work_group_size_32_64"
    assert kernel.attributes == ["__attribute__((amdgpu_flat_work_group_size(32,64)))"]
    assert "// HIP AMDGPU flat work group size: (32, 64)" in crossgl
    assert "fn flat_work_group_size_32_64(" in crossgl
    assert "out[gl_LocalInvocationID.x] = 1.0f;" in crossgl
    assert "__attribute__" not in crossgl


def test_external_llvm_cuda_consteval_device_function_hip_codegen_reparse():
    # Upstream:
    # repo: https://github.com/llvm/llvm-project
    # commit: 146abed5cd072a27c4240c7cf53b764571e62462
    # path: clang/test/SemaCUDA/consteval-func.cu
    source = """
    __device__ consteval int f() { return 0; }
    int main() { return f(); }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    function_signatures = [
        (function.name, function.return_type, function.qualifiers)
        for function in ast.statements
    ]
    assert function_signatures == [
        ("f", "int", ["__device__", "consteval"]),
        ("main", "int", []),
    ]
    assert "i32 f()" in crossgl
    assert "i32 main()" in crossgl
    assert "return f();" in crossgl
    assert "consteval" not in crossgl


def test_external_cccl_three_way_operator_hip_codegen_reparse():
    # Reduced after macro expansion from:
    # repo: https://github.com/NVIDIA/cccl
    # commit: 17869e1d46314036843a9ff9a0cb726de560e94e
    # path: libcudacxx/include/cuda/std/__utility/pair.h
    source = """
    template <class _T1, class _T2>
    __host__ __device__ int operator<=>(const pair<_T1, _T2>& __x,
                                        const pair<_T1, _T2>& __y)
    {
      return compare(__x.first, __y.first);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    function = ast.statements[0]
    assert function.name == "operator<=>"
    assert function.qualifiers == ["__host__", "__device__"]
    assert function.params == [
        {"type": "const pair<_T1, _T2> &", "name": "__x"},
        {"type": "const pair<_T1, _T2> &", "name": "__y"},
    ]
    assert "i32 operator_three_way(pair<_T1, _T2> __x, pair<_T1, _T2> __y)" in crossgl
    assert "i32 operator<=>" not in crossgl


def test_external_cccl_requires_requires_struct_specialization_hip_codegen_reparse():
    # Reduced from:
    # repo: https://github.com/NVIDIA/cccl
    # commit: 17869e1d46314036843a9ff9a0cb726de560e94e
    # path: libcudacxx/include/cuda/std/__utility/pair.h
    source = """
    template <class _T1, class _T2, class _U1, class _U2>
      requires requires {
        typename pair<common_type_t<_T1, _U1>, common_type_t<_T2, _U2>>;
      }
    struct common_type<pair<_T1, _T2>, pair<_U1, _U2>>
    {
      using type = pair<common_type_t<_T1, _U1>,
                        common_type_t<_T2, _U2>>;
    };

    __device__ int keep_after_requires(int value) {
        return value;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    record = ast.statements[0]
    function = ast.statements[1]
    assert isinstance(record, StructNode)
    assert record.name == "common_type<pair<_T1, _T2>, pair<_U1, _U2>>"
    assert function.name == "keep_after_requires"
    assert "struct common_type_pair__T1__T2_pair__U1__U2 {" in crossgl
    assert "i32 keep_after_requires(i32 value)" in crossgl
    assert "requires requires" not in crossgl
    assert "typename pair" not in crossgl
    assert "common_type<" not in crossgl


def test_external_rocm_dynamic_shared_extern_crossgl_reparse():
    source = """
    __global__ void matrix_transpose_kernel(
        float* out,
        const float* in_matrix,
        const unsigned int width) {
        extern __shared__ float shared_matrix_memory[];
        const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
        const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
        shared_matrix_memory[y * width + x] = in_matrix[x * width + y];
        __syncthreads();
        out[y * width + x] = shared_matrix_memory[y * width + x];
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    shared = ast.statements[0].body[0]

    assert isinstance(shared, VariableNode)
    assert shared.vtype == "float[]"
    assert shared.name == "shared_matrix_memory"
    assert set(shared.qualifiers) == {"extern", "__shared__"}
    assert (
        "// HIP dynamic shared memory: shared_matrix_memory uses launch-time "
        "shared memory size"
    ) in crossgl
    assert "var<workgroup> shared_matrix_memory: array<f32>;" in crossgl


def test_external_rocm_bit_extract_builtin_crossgl_reparse():
    source = """
    __global__ void bit_extract_kernel(
        uint32_t* d_output,
        const uint32_t* d_input,
        size_t size) {
        const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = blockDim.x * gridDim.x;
        for(size_t i = offset; i < size; i += stride) {
            d_output[i] = __bitextract_u32(d_input[i], 8, 4);
        }
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    bit_extract_call = ast.statements[0].body[2].body[0].right

    assert isinstance(bit_extract_call, FunctionCallNode)
    assert bit_extract_call.name == "__bitextract_u32"
    assert "gl_NumWorkGroups.x" in crossgl
    assert "d_output[i] = ((d_input[i] >> 8) & 0xfu);" in crossgl
    assert "__bitextract_u32" not in crossgl


def test_external_rocm_texture_management_tex2d_atomic_codegen_reparse():
    source = """
    __global__ void histogram_kernel(unsigned int* histogram,
                                     unsigned int size_x,
                                     unsigned int size_y,
                                     unsigned int hist_bin_count,
                                     hipTextureObject_t tex_obj) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x >= size_x || y >= size_y) {
            return;
        }

        float u = x / static_cast<float>(size_x) + .5f;
        float v = y / static_cast<float>(size_y) + .5f;
        unsigned char val = tex2D<unsigned char>(tex_obj, u, v);
        unsigned int bin_range = ceiling_div(256, hist_bin_count);
        unsigned int bin_idx = static_cast<unsigned int>(val) / bin_range;
        atomicAdd(&histogram[bin_idx], 1);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    body = ast.statements[0].body

    assert isinstance(body[5].value, FunctionCallNode)
    assert body[5].value.name == "tex2D<unsigned char>"
    assert isinstance(body[8], AtomicOperationNode)
    assert body[8].operation == "atomicAdd"
    assert "sampler2D tex_obj" in crossgl
    assert "var val: u8 = texture(tex_obj, vec2<f32>(u, v));" in crossgl
    assert "atomicAdd(histogram[bin_idx], 1);" in crossgl


def test_external_rocm_texture_management_lod_grad_gather_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/ROCm/rocm-examples
    # commit: cf369da68f209c315074204bd0eb61d1a5c015d1
    # path: HIP-Basic/texture_management/main.hip
    source = """
    __global__ void sample_texture_variants(float4* out,
                                            hipTextureObject_t tex_obj,
                                            float2 uv,
                                            float lod) {
        float2 dx = make_float2(1.0f, 0.0f);
        float2 dy = make_float2(0.0f, 1.0f);
        float4 sampled_lod = tex2DLod<float4>(tex_obj, uv.x, uv.y, lod);
        float4 sampled_grad = tex2DGrad<float4>(tex_obj, uv, dx, dy);
        float4 gathered = tex2Dgather<float4>(tex_obj, uv.x, uv.y, 2);
        out[threadIdx.x] = sampled_lod + sampled_grad + gathered;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    body = ast.statements[0].body

    assert body[2].value.name == "tex2DLod<float4>"
    assert body[3].value.name == "tex2DGrad<float4>"
    assert body[4].value.name == "tex2Dgather<float4>"
    assert "sampler2D tex_obj" in crossgl
    assert "var sampled_lod: vec4<f32> = textureLod(" in crossgl
    assert "tex_obj, vec2<f32>(uv.x, uv.y), lod" in crossgl
    assert "var sampled_grad: vec4<f32> = textureGrad(tex_obj, uv, dx, dy);" in crossgl
    assert (
        "var gathered: vec4<f32> = textureGather(tex_obj, vec2<f32>(uv.x, uv.y), 2);"
        in crossgl
    )
    assert (
        "out[gl_LocalInvocationID.x] = ((sampled_lod + sampled_grad) + gathered);"
        in (crossgl)
    )
    assert "tex2DLod" not in crossgl
    assert "tex2DGrad" not in crossgl
    assert "tex2Dgather" not in crossgl


def test_external_rocm_warp_shuffle_reserved_in_parameter_codegen_reparse():
    source = """
    __global__ void matrix_transpose_kernel(float* out,
                                            const float* in,
                                            const unsigned int width) {
        const unsigned int x = threadIdx.x;
        const unsigned int y = threadIdx.y;

        if(x < width && y < width) {
            const float val = in[y * width + x];
            out[x * width + y] = __shfl(val, y * width + x);
            out[x * width + y] = __shfl_sync(__activemask(), val, y * width + x);
        }
    }

    void host(float* d_transposed_matrix, float* d_matrix, unsigned int width) {
        const dim3 block_dim(width, width);
        const dim3 grid_dim(1);
        matrix_transpose_kernel<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
            d_transposed_matrix,
            d_matrix,
            width);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    kernel = ast.statements[0]

    assert kernel.params[1]["name"] == "in"
    assert "@group(0) @binding(1) var<storage, read_write> in_: array<f32>" in crossgl
    assert "var val: f32 = in_[((y * width) + x)];" in crossgl
    assert "hip warp intrinsic __shfl(val, ((y * width) + x))" in crossgl
    assert "hip warp intrinsic __shfl_sync" in crossgl
    assert (
        "// Kernel launch: matrix_transpose_kernel<<<grid_dim, block_dim, "
        "0, hipStreamDefault>>>()"
    ) in crossgl


def test_external_rocm_histogram_ffs_codegen_reparse():
    source = """
    __global__ void histogram(unsigned int* bins,
                              int tid,
                              int block_size) {
        const int b_bits_length = __ffs(block_size) - 3;
        const int sh_thread_id
            = (tid & (1 << b_bits_length) - 1) << 2
              | (tid >> b_bits_length);
        bins[threadIdx.x] = sh_thread_id;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    b_bits_length = ast.statements[0].body[0]

    assert isinstance(b_bits_length.value.left, FunctionCallNode)
    assert b_bits_length.value.left.name == "__ffs"
    assert "var b_bits_length: i32 = ((findLSB(block_size) + 1) - 3);" in crossgl
    assert "__ffs" not in crossgl


def test_rocm_hip_math_api_ffsll_intrinsic_codegen_reparse():
    # ROCm HIP math API lists __ffsll as the 64-bit sibling of __ffs.
    source = """
    __global__ void ffsll_kernel(unsigned int* out,
                                 unsigned long long mask) {
        out[threadIdx.x] = __ffsll(mask);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    assignment = ast.statements[0].body[0]

    assert isinstance(assignment.right, FunctionCallNode)
    assert assignment.right.name == "__ffsll"
    assert "out[gl_LocalInvocationID.x] = (findLSB(mask) + 1);" in crossgl
    assert "__ffsll" not in crossgl


def test_external_rocm_rocjpeg_alignment_bitwise_not_codegen_reparse():
    # Upstream: ROCm/rocm-examples@cf369da68f209c315074204bd0eb61d1a5c015d1,
    # Common/rocjpeg_utils.hpp.
    source = """
    static inline int align(int value, int alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    return_expr = ast.statements[0].body[0].value

    assert return_expr.op == "&"
    assert isinstance(return_expr.right, UnaryOpNode)
    assert return_expr.right.op == "~"
    assert "return (((value + alignment) - 1) & (~(alignment - 1)));" in crossgl


def test_external_rocm_hip_tests_clz_intrinsics_codegen_reparse():
    source = """
    __global__ void clz_kernel(unsigned int* out,
                               unsigned int x,
                               unsigned long long int y) {
        out[0] = __clz(x);
        out[1] = __clzll(y);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    body = ast.statements[0].body

    assert isinstance(body[0].right, FunctionCallNode)
    assert body[0].right.name == "__clz"
    assert isinstance(body[1].right, FunctionCallNode)
    assert body[1].right.name == "__clzll"
    assert "out[0] = countLeadingZeros(x);" in crossgl
    assert "out[1] = countLeadingZeros(y);" in crossgl
    assert "__clz" not in crossgl
    assert "__clzll" not in crossgl


def test_external_hip_tests_multiword_integer_spellings_codegen_reparse():
    # Reduced from ROCm/hip-tests@a618b48f0a29cfbd8c990fa72ab772483f61d381:
    # catch/unit/deviceLib/ldg.cc, plus long-long spellings used across HIP tests.
    source = """
    char2 make_vector2(signed char a) {
        return make_char2(a, a);
    }

    __device__ unsigned long long int ticks(long long int x) {
        unsigned long long int y = 1ull;
        return y + (unsigned long long int)x;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    assert ast.statements[0].params[0]["type"] == "signed char"
    assert "vec2<i8> make_vector2(i8 a)" in crossgl
    assert "u64 ticks(i64 x)" in crossgl
    assert "var y: u64 = 1u;" in crossgl
    assert "u64(x)" in crossgl


def test_external_rocm_hip_tests_brev_intrinsics_codegen_reparse():
    # Upstream: ROCm/hip-tests@d01e1f96059edc25600eb13434d7e2b71c09af01,
    # catch/unit/math/integer_intrinsics.cc.
    source = """
    __global__ void bit_reverse_kernel(unsigned int* out32,
                                       unsigned long long int* out64,
                                       unsigned int x,
                                       unsigned long long int y) {
        out32[0] = __brev(x);
        out64[0] = __brevll(y);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    body = ast.statements[0].body

    assert isinstance(body[0].right, FunctionCallNode)
    assert body[0].right.name == "__brev"
    assert isinstance(body[1].right, FunctionCallNode)
    assert body[1].right.name == "__brevll"
    assert "out32[0] = bitfieldReverse(x);" in crossgl
    assert "out64[0] = bitfieldReverse(y);" in crossgl
    assert "__brev" not in crossgl
    assert "__brevll" not in crossgl
    assert "reverseBits" not in crossgl

    roundtrip_hip = generate_hip_from_crossgl(crossgl)
    assert "out32[0] = __brev(x);" in roundtrip_hip
    assert "out64[0] = __brevll(y);" in roundtrip_hip
    assert "bitfieldReverse" not in roundtrip_hip
    assert "reverseBits" not in roundtrip_hip


def test_rocm_hip_math_api_bit_reinterpret_intrinsics_codegen_reparse():
    # ROCm HIP math API lists the CUDA-compatible type-cast intrinsics.
    source = """
    __device__ float neg_inf() {
        return __int_as_float(0xff800000);
    }

    __global__ void reinterpret_bits(float value,
                                     double wide,
                                     int signedBits,
                                     unsigned int unsignedBits,
                                     long long int signedWide,
                                     int* outSigned,
                                     unsigned int* outUnsigned,
                                     float* outFloat,
                                     long long int* outLong,
                                     double* outDouble) {
        outSigned[0] = __float_as_int(value);
        outUnsigned[0] = __float_as_uint(value);
        outFloat[0] = __int_as_float(signedBits);
        outFloat[1] = ::__uint_as_float(unsignedBits);
        outLong[0] = __double_as_longlong(wide);
        outDouble[0] = __longlong_as_double(signedWide);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    kernel_body = ast.statements[1].body

    assert kernel_body[0].right.name == "__float_as_int"
    assert kernel_body[1].right.name == "__float_as_uint"
    assert kernel_body[2].right.name == "__int_as_float"
    assert kernel_body[3].right.name == "::__uint_as_float"
    assert kernel_body[4].right.name == "__double_as_longlong"
    assert kernel_body[5].right.name == "__longlong_as_double"
    assert "return intBitsToFloat(0xff800000);" in crossgl
    assert "outSigned[0] = floatBitsToInt(value);" in crossgl
    assert "outUnsigned[0] = floatBitsToUint(value);" in crossgl
    assert "outFloat[0] = intBitsToFloat(signedBits);" in crossgl
    assert "outFloat[1] = uintBitsToFloat(unsignedBits);" in crossgl
    assert "outLong[0] = doubleBitsToLong(wide);" in crossgl
    assert "outDouble[0] = longBitsToDouble(signedWide);" in crossgl
    for raw_name in {
        "__double_as_longlong",
        "__float_as_int",
        "__float_as_uint",
        "__int_as_float",
        "__longlong_as_double",
        "__uint_as_float",
    }:
        assert raw_name not in crossgl


def test_rocm_hip_math_api_sad_intrinsics_codegen_reparse():
    # Upstream source:
    # ROCm HIP math API 6.2.41133, Integer intrinsics.
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

    ast, crossgl = assert_crossgl_reparses(source)
    body = ast.statements[0].body

    assert body[0].right.name == "__sad"
    assert body[1].right.name == "__usad"
    assert "out[0] = (abs(x - y) + bias);" in crossgl
    assert "out[1] = (((ux > uy) ? (ux - uy) : (uy - ux)) + bias);" in crossgl
    assert "__sad" not in crossgl
    assert "__usad" not in crossgl


def test_external_rocm_hip_tests_integer_average_intrinsics_codegen_reparse():
    # Upstream: ROCm/hip-tests@d01e1f96059edc25600eb13434d7e2b71c09af01,
    # catch/unit/math/integer_intrinsics.cc.
    source = """
    __global__ void average_kernel(int* signed_out,
                                   unsigned int* unsigned_out,
                                   int x,
                                   int y,
                                   unsigned int ux,
                                   unsigned int uy) {
        signed_out[0] = __hadd(x, y);
        signed_out[1] = __rhadd(x, y);
        unsigned_out[0] = __uhadd(ux, uy);
        unsigned_out[1] = __urhadd(ux, uy);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    body = ast.statements[0].body

    assert body[0].right.name == "__hadd"
    assert body[1].right.name == "__rhadd"
    assert body[2].right.name == "__uhadd"
    assert body[3].right.name == "__urhadd"
    assert "signed_out[0] = ((x & y) + ((x ^ y) >> 1));" in crossgl
    assert "signed_out[1] = ((x | y) - ((x ^ y) >> 1));" in crossgl
    assert "unsigned_out[0] = ((ux & uy) + ((ux ^ uy) >> 1));" in crossgl
    assert "unsigned_out[1] = ((ux | uy) - ((ux ^ uy) >> 1));" in crossgl
    assert "__hadd" not in crossgl
    assert "__rhadd" not in crossgl
    assert "__uhadd" not in crossgl
    assert "__urhadd" not in crossgl


def test_external_rocm_hip_tests_24_bit_multiply_intrinsics_codegen_reparse():
    # Upstream: ROCm/hip-tests@d01e1f96059edc25600eb13434d7e2b71c09af01,
    # catch/unit/math/integer_intrinsics.cc.
    source = """
    __global__ void mul24_kernel(int* signed_out,
                                 unsigned int* unsigned_out,
                                 int x,
                                 int y,
                                 unsigned int ux,
                                 unsigned int uy) {
        signed_out[0] = __mul24(x, y);
        unsigned_out[0] = ::__umul24(ux, uy);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    body = ast.statements[0].body

    assert body[0].right.name == "__mul24"
    assert body[1].right.name == "::__umul24"
    assert "signed_out[0] = (((x << 8) >> 8) * ((y << 8) >> 8));" in crossgl
    assert "unsigned_out[0] = ((ux & 0x00ffffffu) * (uy & 0x00ffffffu));" in crossgl
    assert "__mul24" not in crossgl
    assert "__umul24" not in crossgl


def test_external_rocm_hip_tests_multiply_high_intrinsics_codegen_reparse():
    # Upstream: ROCm/hip-tests@d01e1f96059edc25600eb13434d7e2b71c09af01,
    # catch/unit/math/integer_intrinsics.cc.
    source = """
    __global__ void multiply_high_kernel(int* signed_out,
                                         unsigned int* unsigned_out,
                                         int x,
                                         int y,
                                         unsigned int ux,
                                         unsigned int uy) {
        signed_out[0] = __mulhi(x, y);
        unsigned_out[0] = ::__umulhi(ux, uy);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    body = ast.statements[0].body

    assert body[0].right.name == "__mulhi"
    assert body[1].right.name == "::__umulhi"
    assert "signed_out[0] = i32((i64(x) * i64(y)) >> 32);" in crossgl
    assert "unsigned_out[0] = u32((u64(ux) * u64(uy)) >> 32u);" in crossgl
    assert "__mulhi" not in crossgl
    assert "__umulhi" not in crossgl


def test_external_rocm_hip_complex_math_codegen_reparse():
    # Upstream: ROCm/rocm-examples@cf369da68f209c315074204bd0eb61d1a5c015d1,
    # HIP-Doc/Reference/HIP-Complex-Math-API/complex_math/main.hip.
    source = """
    __device__ hipFloatComplex accumulateDFT(float sample,
                                            hipFloatComplex sum,
                                            int k,
                                            int n,
                                            int N) {
        float angle = -2.0f * M_PI * k * n / N;
        hipFloatComplex w = make_hipFloatComplex(cosf(angle), sinf(angle));
        hipFloatComplex x = make_hipFloatComplex(sample, 0.0f);
        return hipCaddf(sum, hipCmulf(x, w));
    }

    __device__ float complexResidual(hipFloatComplex gpu_output,
                                     hipFloatComplex cpu_output) {
        return (hipCrealf(gpu_output) - hipCrealf(cpu_output))
             + (hipCimagf(gpu_output) - hipCimagf(cpu_output));
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    w_value = ast.statements[0].body[1].value
    return_value = ast.statements[0].body[3].value

    assert isinstance(w_value, FunctionCallNode)
    assert w_value.name == "make_hipFloatComplex"
    assert isinstance(return_value, FunctionCallNode)
    assert return_value.name == "hipCaddf"
    assert return_value.args[1].name == "hipCmulf"
    assert (
        "vec2<f32> accumulateDFT(f32 sample, vec2<f32> sum, " "i32 k, i32 n, i32 N)"
    ) in crossgl
    assert "var w: vec2<f32> = vec2<f32>(cos(angle), sin(angle));" in crossgl
    assert (
        "return (sum + vec2<f32>(((x.x * w.x) - (x.y * w.y)), "
        "((x.x * w.y) + (x.y * w.x))));"
    ) in crossgl
    assert "f32 complexResidual(vec2<f32> gpu_output, vec2<f32> cpu_output)" in crossgl
    assert (
        "return ((gpu_output.x - cpu_output.x) + " "(gpu_output.y - cpu_output.y));"
    ) in crossgl
    for raw_name in (
        "hipFloatComplex",
        "make_hipFloatComplex",
        "hipCaddf",
        "hipCmulf",
        "hipCrealf",
        "hipCimagf",
    ):
        assert raw_name not in crossgl


def test_external_rocm_image_convolution_continue_codegen_reparse():
    source = """
    __global__ void conv2d(uint8_t *image,
                           uint8_t *output,
                           float *mask,
                           int image_width,
                           int image_height,
                           int channels,
                           int mask_width,
                           int mask_height) {
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;

      if (x >= image_width || y >= image_height) {
        return;
      }

      for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;

        for (int i = 0; i < mask_height; i++) {
          for (int j = 0; j < mask_width; j++) {
            int image_x = x + j - mask_width / 2;
            int image_y = y + i - mask_height / 2;

            if (image_x < 0 || image_x >= image_width || image_y < 0 ||
                image_y >= image_height) {
              continue;
            }

            int image_index = (image_y * image_width + image_x) * channels + c;
            int mask_index = i * mask_width + j;
            sum += image[image_index] / 255.0f * mask[mask_index];
          }
        }

        int output_index = (y * image_width + x) * channels + c;
        output[output_index] = static_cast<uint8_t>(sum * 255.0f);
      }
    }
    """

    _, crossgl = assert_crossgl_reparses(source)

    assert "@group(0) @binding(0) var<storage, read_write> image: array<u8>" in crossgl
    assert "for (var c: i32 = 0; (c < channels); (++c))" in crossgl
    assert "continue;" in crossgl
    assert "sum += ((image[image_index] / 255.0f) * mask[mask_index]);" in crossgl
    assert "output[output_index] = u8((sum * 255.0f));" in crossgl
    assert "static_cast" not in crossgl


def test_external_hip_kittens_fast_exp_vector_codegen_reparse():
    source = """
    __device__ float hk_exp(float x) {
        return __expf(x);
    }

    __device__ float2 hk_exp2(float2 x) {
        return float2{__expf(x.x), __expf(x.y)};
    }
    """

    _, crossgl = assert_crossgl_reparses(source)

    assert "return exp(x);" in crossgl
    assert "return vec2<f32>(exp(x.x), exp(x.y));" in crossgl
    assert "__expf" not in crossgl


def test_hip_scalar_brace_initializer_scoped_call_codegen_reparse():
    source = """
    template <class T>
    struct Info {
        int value;
    };

    template <class T>
    int make_info() {
        return ::hip::detail::make_value<T>();
    }

    template <class T>
    Info<T> info = {::hip::detail::make_info<T>()};
    """

    ast, crossgl = assert_crossgl_reparses(source)

    assert ast.statements[1].name == "make_info"
    assert ast.statements[2].name == "info"
    assert "return hip_detail_make_value();" in crossgl
    assert "var info: Info<T> = hip_detail_make_info();" in crossgl
    assert "{::hip::detail" not in crossgl
    assert "::hip::detail" not in crossgl


def test_rocm_math_api_fast_float_intrinsics_codegen_reparse():
    # HIP math API documents CUDA-compatible single-precision intrinsics such
    # as __fdividef, rounded arithmetic aliases, and __saturatef.
    source = """
    __device__ float normalized_weight(float weight, float normalizer, float bias) {
        float scaled = __fdividef(weight, normalizer);
        float rounded = __fdiv_rn(weight, normalizer);
        float adjusted = __fadd_rd(scaled, bias);
        float centered = __fsub_rz(adjusted, rounded);
        float product = __fmul_ru(centered, __fadd_rn(weight, bias));
        float reciprocal = __frcp_rn(__fadd_rz(normalizer, 1.0f));
        return __saturatef(__fmul_rn(product, reciprocal));
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    body = ast.statements[0].body

    assert body[0].value.name == "__fdividef"
    assert body[1].value.name == "__fdiv_rn"
    assert body[2].value.name == "__fadd_rd"
    assert body[3].value.name == "__fsub_rz"
    assert body[4].value.name == "__fmul_ru"
    assert body[5].value.name == "__frcp_rn"
    assert body[6].value.name == "__saturatef"
    assert "var scaled: f32 = (weight / normalizer);" in crossgl
    assert "var rounded: f32 = (weight / normalizer);" in crossgl
    assert "var adjusted: f32 = (scaled + bias);" in crossgl
    assert "var centered: f32 = (adjusted - rounded);" in crossgl
    assert "var product: f32 = (centered * (weight + bias));" in crossgl
    assert "var reciprocal: f32 = (1.0f / (normalizer + 1.0f));" in crossgl
    assert "return clamp((product * reciprocal), 0.0f, 1.0f);" in crossgl
    for raw_name in (
        "__fdividef",
        "__fdiv_rn",
        "__fadd_rd",
        "__fadd_rn",
        "__fadd_rz",
        "__fsub_rz",
        "__fmul_ru",
        "__fmul_rn",
        "__frcp_rn",
        "__saturatef",
    ):
        assert raw_name not in crossgl


def test_external_hip_fp16_scalar_conversion_codegen_reparse():
    source = """
    __global__ void add_half_precision(__half* in1,
                                       __half* in2,
                                       float* out,
                                       size_t size) {
        int idx = threadIdx.x;
        if(idx < size) {
            float sum = __half2float(in1[idx] + in2[idx]);
            out[idx] = sum;
        }
    }

    void host(float in) {
        __half value = __float2half(in);
    }
    """

    _, crossgl = assert_crossgl_reparses(source)

    assert "var sum: f32 = f32((in1[idx] + in2[idx]));" in crossgl
    assert "var value: f16 = f16(in_);" in crossgl
    assert "__half2float" not in crossgl
    assert "__float2half" not in crossgl


def test_hipify_half2_high_lane_and_two_float_constructor_codegen_reparse():
    # Source inspiration:
    # ROCm HIPIFY CUDA Device API supported-by-HIP table lists the CUDA half2
    # lane extraction and two-float constructor intrinsics as HIP-supported.
    source = """
    __device__ float pack_and_sum_half2(float a, float b) {
        half2 pair = __floats2half2_rn(a, b);
        return __low2float(pair) + __high2float(pair);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    body = ast.statements[0].body

    assert body[0].value.name == "__floats2half2_rn"
    assert body[1].value.left.name == "__low2float"
    assert body[1].value.right.name == "__high2float"
    assert "var pair: vec2<f16> = vec2<f16>(a, b);" in crossgl
    assert "return (f32(pair.x) + f32(pair.y));" in crossgl
    assert "__floats2half2_rn" not in crossgl
    assert "__high2float" not in crossgl


def test_rocm_hip_math_half2_lane_pack_codegen_reparse():
    # Source: ROCm HIP Math API half precision conversion/data movement.
    # URL: https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.2/reference/math_api.html
    source = """
    __device__ half2 repack_half2_lanes(half2 a, half2 b) {
        __half lo = __low2half(a);
        __half hi = __high2half(a);
        half2 packed = __halves2half2(lo, hi);
        half2 lows = __lows2half2(a, b);
        half2 highs = __highs2half2(a, b);
        half2 duplicatedLow = __low2half2(a);
        half2 duplicatedHigh = __high2half2(b);
        half2 swapped = __lowhigh2highlow(a);
        return packed + lows + highs + duplicatedLow + duplicatedHigh + swapped;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    body = ast.statements[0].body

    assert body[0].value.name == "__low2half"
    assert body[1].value.name == "__high2half"
    assert body[2].value.name == "__halves2half2"
    assert body[3].value.name == "__lows2half2"
    assert body[4].value.name == "__highs2half2"
    assert body[5].value.name == "__low2half2"
    assert body[6].value.name == "__high2half2"
    assert body[7].value.name == "__lowhigh2highlow"
    assert "var lo: f16 = a.x;" in crossgl
    assert "var hi: f16 = a.y;" in crossgl
    assert "var packed: vec2<f16> = vec2<f16>(lo, hi);" in crossgl
    assert "var lows: vec2<f16> = vec2<f16>(a.x, b.x);" in crossgl
    assert "var highs: vec2<f16> = vec2<f16>(a.y, b.y);" in crossgl
    assert "var duplicatedLow: vec2<f16> = vec2<f16>(a.x, a.x);" in crossgl
    assert "var duplicatedHigh: vec2<f16> = vec2<f16>(b.y, b.y);" in crossgl
    assert "var swapped: vec2<f16> = vec2<f16>(a.y, a.x);" in crossgl
    for raw_name in (
        "__low2half",
        "__high2half",
        "__halves2half2",
        "__lows2half2",
        "__highs2half2",
        "__low2half2",
        "__high2half2",
        "__lowhigh2highlow",
    ):
        assert raw_name not in crossgl


def test_external_rocm_warp_size_reduction_popcount_codegen_reparse():
    source = """
    inline auto popcount(unsigned int x) -> int {
        return __builtin_popcount(x);
    }

    inline auto popcount(unsigned long x) -> int {
        return __builtin_popcountl(x);
    }

    inline auto popcount(unsigned long long x) -> int {
        return __builtin_popcountll(x);
    }

    inline auto msvc_popcount16(unsigned short x) -> unsigned short {
        return __popcnt16(x);
    }

    inline auto msvc_popcount32(unsigned int x) -> unsigned int {
        return __popcnt(x);
    }

    inline auto msvc_popcount64(unsigned __int64 x) -> unsigned __int64 {
        return __popcnt64(x);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    msvc_popcount64 = ast.statements[-1]

    assert msvc_popcount64.return_type == "unsigned __int64"
    assert msvc_popcount64.params[0]["type"] == "unsigned __int64"
    assert crossgl.count("return bitCount(x);") == 6
    assert "u64 msvc_popcount64(u64 x)" in crossgl
    for raw_name in (
        "__builtin_popcount",
        "__builtin_popcountl",
        "__builtin_popcountll",
        "__popcnt16",
        "__popcnt(",
        "__popcnt64",
    ):
        assert raw_name not in crossgl


def test_external_rocm_hip_tests_popc_intrinsics_codegen_reparse():
    source = """
    __global__ void popc_kernel(
        unsigned int* a,
        unsigned int* b,
        unsigned long long int* c,
        unsigned long long int* d,
        int width,
        int height) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int i = y * width + x;
        if (i < (width * height)) {
            a[i] = __popc(b[i]);
            c[i] = __popcll(d[i]);
        }
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    branch = ast.statements[0].body[-1]

    assert branch.if_body[0].right.name == "__popc"
    assert branch.if_body[1].right.name == "__popcll"
    assert "a[i] = bitCount(b[i]);" in crossgl
    assert "c[i] = bitCount(d[i]);" in crossgl
    assert "__popc" not in crossgl
    assert "__popcll" not in crossgl


def test_external_hip_tests_unsigned_long_long_sizeof_codegen_reparse():
    # Upstream: ROCm/hip-tests@d01e1f96059edc25600eb13434d7e2b71c09af01,
    # catch/unit/math/integer_intrinsics.cc and catch/unit/deviceLib/popc.cc.
    source = """
    void allocate_popc_inputs() {
        unsigned long long int* hostD;
        hostD = (unsigned long long int*)malloc(
            NUM * sizeof(unsigned long long int));
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    assignment = ast.statements[0].body[1]
    malloc_call = assignment.right.expression
    sizeof_call = malloc_call.args[0].right

    assert assignment.right.target_type == "unsigned long long *"
    assert malloc_call.name == "malloc"
    assert sizeof_call.name == "sizeof"
    assert sizeof_call.args == ["unsigned long long"]
    assert "var hostD: ptr<u64>;" in crossgl
    assert "sizeof(unsigned long long)" in crossgl


def test_external_rocm_hip_tests_byte_perm_intrinsic_codegen_reparse():
    # Upstream: ROCm/hip-tests@d01e1f96059edc25600eb13434d7e2b71c09af01,
    # catch/unit/math/integer_intrinsics.cc.
    source = """
    __global__ void __byte_perm(
        unsigned int* y,
        unsigned int x1,
        unsigned int x2,
        unsigned int s) {
        y[0] = __byte_perm(x1, x2, s);
        y[1] = __byte_perm(x1, x2, 0x0123);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    first_assignment = ast.statements[0].body[0]

    assert first_assignment.right.name == "__byte_perm"
    assert "y[0] = " in crossgl
    assert "((s & 0xf) == 0)" in crossgl
    assert "((s >> 12) & 0xf)" in crossgl
    assert "y[1] = (((x1 >> 24) & 0xffu)" in crossgl
    assert "= __byte_perm(" not in crossgl


def test_external_rocm_hip_tests_funnelshift_intrinsics_codegen_reparse():
    # Reduced from ROCm/hip-tests@d01e1f96059edc25600eb13434d7e2b71c09af01,
    # catch/unit/deviceLib/funnelshift.cc.
    source = """
    __global__ void funnelshift_kernel(unsigned int* l_out,
                                       unsigned int* lc_out,
                                       unsigned int* r_out,
                                       unsigned int* rc_out) {
        unsigned int i = threadIdx.x;
        l_out[i] = __funnelshift_l(0xdeadbeefu, 0xfacefeedu, i);
        lc_out[i] = __funnelshift_lc(0xdeadbeefu, 0xfacefeedu, i);
        r_out[i] = __funnelshift_r(0xdeadbeefu, 0xfacefeedu, i);
        rc_out[i] = __funnelshift_rc(0xdeadbeefu, 0xfacefeedu, i);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    body = ast.statements[0].body

    assert body[1].right.name == "__funnelshift_l"
    assert body[2].right.name == "__funnelshift_lc"
    assert body[3].right.name == "__funnelshift_r"
    assert body[4].right.name == "__funnelshift_rc"
    assert (
        "l_out[i] = u32((((u64(0xfacefeedu) << 32u) | "
        "u64(0xdeadbeefu)) << (i & 31u)) >> 32u);"
    ) in crossgl
    assert (
        "lc_out[i] = u32((((u64(0xfacefeedu) << 32u) | "
        "u64(0xdeadbeefu)) << min(i, 32u)) >> 32u);"
    ) in crossgl
    assert (
        "r_out[i] = u32(((u64(0xfacefeedu) << 32u) | "
        "u64(0xdeadbeefu)) >> (i & 31u));"
    ) in crossgl
    assert (
        "rc_out[i] = u32(((u64(0xfacefeedu) << 32u) | "
        "u64(0xdeadbeefu)) >> min(i, 32u));"
    ) in crossgl
    assert "__funnelshift_" not in crossgl


def test_public_cuda_kernel_if_init_statement_hip_parity_codegen_reparse():
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

    ast, crossgl = assert_crossgl_reparses(source)
    loop_body = ast.statements[0].body[2].body

    assert isinstance(loop_body[0], VariableNode)
    assert loop_body[0].name == "r"
    assert loop_body[0].vtype == "auto"
    assert isinstance(loop_body[1], IfNode)
    assert "var r: auto = ((ti + (s * rows)) + row);" in crossgl
    assert "if (((r < num) && (di < dim))) {" in crossgl
    assert "out[r] = di;" in crossgl


def test_current_cccl_pair_namespace_visibility_and_alias_macros_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/NVIDIA/cccl
    # commit: 17869e1d46314036843a9ff9a0cb726de560e94e
    # path: libcudacxx/include/cuda/std/__utility/pair.h
    source = """
    _CCCL_BEGIN_NAMESPACE_CUDA_STD

    template <class _T1, class _T2>
    struct _CCCL_TYPE_VISIBILITY_DEFAULT pair {
      _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void swap(pair& __p) noexcept {
        using ::cuda::std::swap;
        swap(first, __p.first);
      }
      _T1 first;
    };

    template <class _Tp, class _Up>
    struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<0, pair<_Tp, _Up>> {
      using type _CCCL_NODEBUG_ALIAS = _Tp;
    };

    _CCCL_END_NAMESPACE_CUDA_STD
    _CCCL_BEGIN_NAMESPACE_STD
    _CCCL_END_NAMESPACE_STD
    """

    ast, crossgl = assert_crossgl_reparses(source)

    assert ast.statements[0].name == "pair"
    assert ast.statements[1].name == "tuple_element<0, pair<_Tp, _Up>>"
    assert "struct pair" in crossgl
    assert "struct tuple_element_type_0_pair__Tp__Up" in crossgl
    assert "_CCCL_TYPE_VISIBILITY_DEFAULT" not in crossgl
    assert "_CCCL_NODEBUG_ALIAS" not in crossgl


def test_external_llvm_amdgpu_numeric_template_kernel_name_codegen_reparse():
    # Upstream source:
    # repo: https://github.com/llvm/llvm-project
    # commit: 3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff
    # path: clang/test/SemaCUDA/amdgpu-attrs.cu
    source = """
    template <int X, int Y>
    __attribute__((amdgpu_flat_work_group_size(X, Y)))
    __global__ void template_flat_work_group_size_32_64() {}

    template __global__ void template_flat_work_group_size_32_64<32, 64>();
    """

    ast, crossgl = assert_crossgl_reparses(source)

    assert ast.statements[0].name == "template_flat_work_group_size_32_64"
    assert ast.statements[1].name == "template_flat_work_group_size_32_64<32, 64>"
    assert "fn template_flat_work_group_size_32_64(" in crossgl
    assert crossgl.count("fn template_flat_work_group_size_32_64(") == 2
    assert "fn template_flat_work_group_size_32_64<32, 64>" not in crossgl
