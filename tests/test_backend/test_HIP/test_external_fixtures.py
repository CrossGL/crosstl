from crosstl.backend.HIP.HipAst import (
    AtomicOperationNode,
    FunctionCallNode,
    HipAsmNode,
    KernelLaunchNode,
    VariableNode,
)
from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

EXTERNAL_FIXTURE_SOURCES = {
    "rocm_examples": {
        "url": "https://github.com/ROCm/rocm-examples",
        "commit": "cf369da68f209c315074204bd0eb61d1a5c015d1",
        "paths": [
            "HIP-Basic/bit_extract/main.hip",
            "HIP-Basic/cooperative_groups/main.hip",
            "HIP-Basic/device_globals/main.hip",
            "HIP-Basic/dynamic_shared/main.hip",
            "HIP-Basic/texture_management/main.hip",
            "HIP-Basic/warp_shuffle/main.hip",
            "HIP-Doc/Tutorials/graph_api/src/filtering.hip",
            "HIP-Doc/Tutorials/Programming-Patterns/image_convolution/main.hip",
            "HIP-Doc/Programming-Guide/HIP-C++-Language-Extensions/warp_size_reduction/popcount.hpp",
        ],
    },
    "rocm_examples_inline_assembly": {
        "url": "https://github.com/ROCm/rocm-examples",
        "commit": "b4ee9992e851a078c99d93de59d6142a51f5e3a1",
        "paths": ["HIP-Basic/inline_assembly/main.hip"],
    },
    "hip_examples": {
        "url": "https://github.com/ROCm/HIP-Examples",
        "commit": "cdf9d101acd9a3fc89ee750f73c1f1958cbd5cc3",
        "paths": ["HIP-Examples-Applications/Histogram/Histogram.cpp"],
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
        "// Kernel launch: test_globals_kernel<<<vec3<u32>(64), vec3<u32>(1), "
        "0, hipStreamDefault>>>()"
    ) in crossgl


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
        "(width / 8)), vec3<u32>(8, 8), 0, hipStreamDefault>>>()" in crossgl
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
    assert "__bitextract_u32(d_input[i], 8, 4)" in crossgl


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
    assert "atomicAdd((&histogram[bin_idx]), 1);" in crossgl


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
