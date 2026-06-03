from crosstl.backend.HIP.HipAst import (
    AtomicOperationNode,
    FunctionCallNode,
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
            "HIP-Basic/dynamic_shared/main.hip",
            "HIP-Basic/texture_management/main.hip",
            "HIP-Basic/warp_shuffle/main.hip",
        ],
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
