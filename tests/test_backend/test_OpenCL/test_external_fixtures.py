from crosstl.backend.OpenCL.OpenCLCrossGLCodeGen import OpenCLToCrossGLConverter
from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

EXTERNAL_FIXTURE_SOURCES = {
    "arrayfire": {
        "url": "https://github.com/arrayfire/arrayfire",
        "path": "src/backend/opencl/kernel/reduce_first.cl",
    },
    "pocl": {
        "url": "https://github.com/pocl/pocl",
        "path": "lib/CL/devices/level0/memfill.cl",
    },
    "clblast": {
        "url": "https://github.com/CNugteren/CLBlast",
        "path": "src/kernels/level3/xgemm_part2.opencl",
    },
    "darktable": {
        "url": "https://github.com/darktable-org/darktable",
        "path": "data/kernels/noise_generator.h",
    },
    "darktable_basic_qualified_vector_constructor": {
        "url": "https://github.com/darktable-org/darktable",
        "commit": "321ec599414e138be0232e24ab0d322eac073deb",
        "path": "data/kernels/basic.cl",
    },
    "darktable_blendop_comma_assignment_statement": {
        "url": "https://github.com/darktable-org/darktable",
        "commit": "f2565512667db3d4a7142ba3d3248a02f7be917d",
        "path": "data/kernels/blendop.cl",
    },
    "darktable_channelmixer_statement_expression_switch": {
        "url": "https://github.com/darktable-org/darktable",
        "commit": "f2565512667db3d4a7142ba3d3248a02f7be917d",
        "path": "data/kernels/channelmixer.cl",
    },
    "opencv_multiline_constant_address_space_array": {
        "url": "https://github.com/opencv/opencv",
        "commit": "6f29af625bb4617e2e061f8097b5f3e2ed341a82",
        "path": "modules/core/src/opencl/cvtclr_dx.cl",
    },
    "opencv_fft_leading_attribute_type_macro_constructor": {
        "url": "https://github.com/opencv/opencv",
        "commit": "6f29af625bb4617e2e061f8097b5f3e2ed341a82",
        "path": "modules/core/src/opencl/fft.cl",
    },
    "khronos_opencl_sdk_reduce_shared_parameter_name": {
        "url": "https://github.com/KhronosGroup/OpenCL-SDK",
        "commit": "e26922bdf54eaa9fcc31fe1f91d21b8d2bd6970f",
        "path": "samples/core/reduce/reduce.cl",
    },
    "khronos_opencl_sdk_histogram_newline_for_declaration": {
        "url": "https://github.com/KhronosGroup/OpenCL-SDK",
        "commit": "e26922bdf54eaa9fcc31fe1f91d21b8d2bd6970f",
        "path": "samples/extensions/khr/histogram/histogram.cl",
    },
    "arrayfire_nearest_neighbour_bare_unsigned_parameter": {
        "url": "https://github.com/arrayfire/arrayfire",
        "commit": "492718b5a256d4a9d5198fdce89d8fd21772bfda",
        "path": "src/backend/opencl/kernel/nearest_neighbour.cl",
    },
    "arrayfire_magma_opencl_api_handle_typedefs": {
        "url": "https://github.com/arrayfire/arrayfire",
        "commit": "492718b5a256d4a9d5198fdce89d8fd21772bfda",
        "path": "src/backend/opencl/magma/magma_common.h",
    },
    "arrayfire_magma_dependent_typename_inline": {
        "url": "https://github.com/arrayfire/arrayfire",
        "commit": "492718b5a256d4a9d5198fdce89d8fd21772bfda",
        "path": "src/backend/opencl/magma/magma_blas_clblast.h",
    },
    "opencv_optical_flow_opencl_const_parameter": {
        "url": "https://github.com/opencv/opencv",
        "commit": "6f29af625bb4617e2e061f8097b5f3e2ed341a82",
        "path": "modules/video/src/opencl/optical_flow_farneback.cl",
    },
    "opencv_batchnorm_macro_type_cast": {
        "url": "https://github.com/opencv/opencv",
        "commit": "6f29af625bb4617e2e061f8097b5f3e2ed341a82",
        "path": "modules/dnn/src/opencl/batchnorm.cl",
    },
    "opencv_filter_sep_col_macro_type_cast": {
        "url": "https://github.com/opencv/opencv",
        "commit": "6f29af625bb4617e2e061f8097b5f3e2ed341a82",
        "path": "modules/imgproc/src/opencl/filterSepCol.cl",
    },
    "opencv_morph3x3_vector_scalar_cast": {
        "url": "https://github.com/opencv/opencv",
        "commit": "6f29af625bb4617e2e061f8097b5f3e2ed341a82",
        "path": "modules/imgproc/src/opencl/morph3x3.cl",
    },
    "pocl_noinline_function_specifier": {
        "url": "https://github.com/pocl/pocl",
        "commit": "d11f27f3ba667456466cd935dacaf69e5cbf2598",
        "path": "tests/kernel/test_as_type.cl",
    },
}


def assert_crossgl_reparses(source):
    ast = OpenCLParser(OpenCLLexer(source).tokenize()).parse()
    crossgl = OpenCLToCrossGLConverter().generate(ast)
    CrossGLParser(CrossGLLexer(crossgl).tokens).parse()
    return ast, crossgl


def test_external_arrayfire_reduce_first_pattern_codegen_reparse():
    source = """
    kernel void reduce_first_kernel(global float *oData, const global float *iData) {
        const uint lidx = get_local_id(0);
        const uint groupId_x = get_group_id(0);
        local float s_val[256];
        s_val[lidx] = iData[lidx];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lidx == 0) {
            oData[groupId_x] = s_val[0];
        }
    }
    """

    _ast, crossgl = assert_crossgl_reparses(source)

    assert "gl_LocalInvocationID.x" in crossgl
    assert "gl_WorkGroupID.x" in crossgl
    assert "var<workgroup> s_val: array<f32, 256>;" in crossgl


def test_external_pocl_void_kernel_after_return_type_pattern_codegen_reparse():
    source = """
    void __kernel memfill_4(global uint *mem, const uint pattern) {
        size_t gid = get_global_id(0) * 4;
        mem[gid + 0] = pattern;
        mem[gid + 1] = pattern;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    assert ast.statements[0].name == "memfill_4"
    assert "fn memfill_4(" in crossgl
    assert "gl_GlobalInvocationID.x" in crossgl


def test_external_clblast_kernel_pattern_codegen_reparse():
    source = """
    R"(
    __kernel void XgemmBody(global float *agm, global float *bgm, global float *cgm,
                            const int kSizeM, const int kSizeN) {
        const int tidm = get_local_id(0);
        const int tidn = get_local_id(1);
        const int offset = get_group_id(0) * kSizeM + get_group_id(1) * kSizeN;
        cgm[offset + tidm + tidn] = agm[tidm] + bgm[tidn];
    }
    )"
    """

    _ast, crossgl = assert_crossgl_reparses(source)

    assert "gl_LocalInvocationID.x" in crossgl
    assert "gl_LocalInvocationID.y" in crossgl
    assert "gl_WorkGroupID.x" in crossgl


def test_external_darktable_hex_float_literal_codegen_reparse():
    source = """
    static inline float uniform_noise(uint state[4]) {
        uint result = state[0];
        return (float)(result >> 8) * 0x1.0p-24f;
    }

    kernel void noise_probe(global float *out, global uint *state) {
        uint local_state[4];
        local_state[0] = state[0];
        out[0] = uniform_noise(local_state);
    }
    """

    _ast, crossgl = assert_crossgl_reparses(source)

    assert "0x1.0p-24f" in crossgl


def test_external_darktable_qualified_vector_constructor_codegen_reparse():
    source_info = EXTERNAL_FIXTURE_SOURCES[
        "darktable_basic_qualified_vector_constructor"
    ]
    assert source_info["commit"] == "321ec599414e138be0232e24ab0d322eac073deb"
    assert source_info["path"] == "data/kernels/basic.cl"

    source = """
    kernel void rawprepare_4f_sample(global float *out,
                                     global const float *black,
                                     global const float *div) {
        const float4 black4 = (const float4)(black[0], black[1], black[2], black[3]);
        const float4 div4 = (const float4)(div[0], div[1], div[2], div[3]);
        out[0] = black4.x / div4.x;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    first_decl = ast.statements[0].body[0]
    assert first_decl.value.name == "float4"
    assert "vec4<f32>(" in crossgl
    assert "const float4(" not in crossgl


def test_external_darktable_comma_assignment_statement_codegen_reparse():
    source_info = EXTERNAL_FIXTURE_SOURCES[
        "darktable_blendop_comma_assignment_statement"
    ]
    assert source_info["commit"] == "f2565512667db3d4a7142ba3d3248a02f7be917d"
    assert source_info["path"] == "data/kernels/blendop.cl"

    source = """
    kernel void blendif_scale(global float *scaled,
                              global const float4 *input) {
        float4 pixel = input[0];
        scaled[0] = pixel.x / 100.0f,
        scaled[1] = pixel.y / 256.0f;
        scaled[2] = pixel.z / 256.0f;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)
    body = ast.statements[0].body

    assert len(body) == 4
    assert "scaled[0] = (pixel.x / 100.0f);" in crossgl
    assert "scaled[1] = (pixel.y / 256.0f);" in crossgl
    assert "scaled[0] = pixel.x / 100.0f, scaled[1]" not in crossgl


def test_external_darktable_channelmixer_statement_expression_codegen_reparse():
    source_info = EXTERNAL_FIXTURE_SOURCES[
        "darktable_channelmixer_statement_expression_switch"
    ]
    assert source_info["commit"] == "f2565512667db3d4a7142ba3d3248a02f7be917d"
    assert source_info["path"] == "data/kernels/channelmixer.cl"

    source = """
    #define unswitch_channelmixer(kind) \\
      ({ switch(kind) \\
        { \\
          case 0: \\
          { \\
            out[0] = 1.0f; \\
            break; \\
          } \\
          default: \\
          { \\
            out[0] = 0.0f; \\
            break; \\
          } \\
        }})

    kernel void channelmixer_probe(global float *out, const int kind) {
        unswitch_channelmixer(kind);
    }
    """

    _ast, crossgl = assert_crossgl_reparses(source)

    assert "switch (kind)" in crossgl
    assert "case 0:" in crossgl
    assert "out[0] = 1.0f;" in crossgl


def test_external_opencv_multiline_constant_address_space_array_codegen_reparse():
    source_info = EXTERNAL_FIXTURE_SOURCES[
        "opencv_multiline_constant_address_space_array"
    ]
    assert source_info["commit"] == "6f29af625bb4617e2e061f8097b5f3e2ed341a82"
    assert source_info["path"] == "modules/core/src/opencl/cvtclr_dx.cl"

    source = """
    static
    __constant
    float c_YUV2RGBCoeffs_420[5] =
    {
         1.163999557f,
         2.017999649f,
        -0.390999794f,
        -0.812999725f,
         1.5959997177f
    };

    kernel void yuv_coeff_probe(global float *out) {
        __constant float *coeffs = c_YUV2RGBCoeffs_420;
        out[0] = coeffs[0];
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    assert ast.statements[0].vtype == "__constant__ float[5]"
    assert "var c_YUV2RGBCoeffs_420: array<f32, 5>" in crossgl
    assert "var<uniform> coeffs: ptr<f32> = c_YUV2RGBCoeffs_420;" in crossgl


def test_external_opencv_fft_leading_attribute_type_macro_constructor_codegen_reparse():
    source_info = EXTERNAL_FIXTURE_SOURCES[
        "opencv_fft_leading_attribute_type_macro_constructor"
    ]
    assert source_info["commit"] == "6f29af625bb4617e2e061f8097b5f3e2ed341a82"
    assert source_info["path"] == "modules/core/src/opencl/fft.cl"

    source = """
    __attribute__((always_inline))
    CT mul_complex(CT a, CT b) {
        return (CT)(fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x));
    }

    kernel void fft_probe(global CT *out, global const CT *twiddles) {
        CT a = twiddles[0];
        out[0] = mul_complex(a, twiddles[1]);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    assert ast.statements[0].return_type == "CT"
    assert ast.statements[0].params[0] == {"type": "CT", "name": "a"}
    assert "CT mul_complex(CT a, CT b)" in crossgl
    assert "return CT(fma(a.x, b.x" in crossgl


def test_external_khronos_opencl_sdk_shared_parameter_name_codegen_reparse():
    source_info = EXTERNAL_FIXTURE_SOURCES[
        "khronos_opencl_sdk_reduce_shared_parameter_name"
    ]
    assert source_info["commit"] == "e26922bdf54eaa9fcc31fe1f91d21b8d2bd6970f"
    assert source_info["path"] == "samples/core/reduce/reduce.cl"

    source = """
    int read_local(local int *shared, size_t count, size_t i) {
        return i < count ? shared[i] : 0;
    }

    kernel void reduce(global int *front, local int *shared) {
        size_t lid = get_local_id(0);
        shared[lid] = read_local(shared, 1, lid) + front[lid];
    }
    """

    _ast, crossgl = assert_crossgl_reparses(source)

    assert "ptr<i32> shared_" in crossgl
    assert "shared_[i]" in crossgl
    assert "shared: array<i32>" not in crossgl


def test_external_khronos_opencl_sdk_histogram_newline_for_codegen_reparse():
    source_info = EXTERNAL_FIXTURE_SOURCES[
        "khronos_opencl_sdk_histogram_newline_for_declaration"
    ]
    assert source_info["commit"] == "e26922bdf54eaa9fcc31fe1f91d21b8d2bd6970f"
    assert source_info["path"] == "samples/extensions/khr/histogram/histogram.cl"

    source = """
    kernel void histogram_probe(global uint *out,
                                uint channel_per_thread,
                                uint lid,
                                uint bins) {
        for(
            uint channel = channel_per_thread * lid;
            channel < min(channel_per_thread * (lid + 1), bins);
            channel++
        ){
            out[channel] = channel;
        }
    }
    """

    _ast, crossgl = assert_crossgl_reparses(source)

    assert "for (var channel: u32 =" in crossgl
    assert "channel < min" in crossgl
    assert "channel++" in crossgl


def test_external_arrayfire_bare_unsigned_parameter_codegen_reparse():
    source_info = EXTERNAL_FIXTURE_SOURCES[
        "arrayfire_nearest_neighbour_bare_unsigned_parameter"
    ]
    assert source_info["commit"] == "492718b5a256d4a9d5198fdce89d8fd21772bfda"
    assert source_info["path"] == "src/backend/opencl/kernel/nearest_neighbour.cl"

    source = """
    __inline unsigned popcount(unsigned x) {
        x = x - ((x >> 1) & 0x55555555);
        return x;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    assert ast.statements[0].params[0] == {"type": "unsigned int", "name": "x"}
    assert "u32 popcount(u32 x)" in crossgl
    assert "unsigned x _param" not in crossgl


def test_external_arrayfire_magma_opencl_api_typedefs_codegen_reparse():
    source_info = EXTERNAL_FIXTURE_SOURCES["arrayfire_magma_opencl_api_handle_typedefs"]
    assert source_info["commit"] == "492718b5a256d4a9d5198fdce89d8fd21772bfda"
    assert source_info["path"] == "src/backend/opencl/magma/magma_common.h"

    source = """
    typedef cl_command_queue magma_queue_t;
    typedef cl_event magma_event_t;
    typedef cl_device_id magma_device_t;
    typedef cl_double2 magmaDoubleComplex;
    typedef cl_float2 magmaFloatComplex;
    typedef cl_mem magma_ptr;
    enum magma_uplo_t {
        MagmaUpper = 121,
        MagmaLower = 122,
    };
    typedef magma_uplo_t magma_type_t;
    struct cpu_blas_gemv_func<float> {
        int placeholder;
    };
    """

    _ast, crossgl = assert_crossgl_reparses(source)

    assert "typedef u64 magma_queue_t;" in crossgl
    assert "typedef u64 magma_event_t;" in crossgl
    assert "typedef u64 magma_device_t;" in crossgl
    assert "typedef vec2<f64> magmaDoubleComplex;" in crossgl
    assert "typedef vec2<f32> magmaFloatComplex;" in crossgl
    assert "typedef u64 magma_ptr;" in crossgl
    assert "typedef u32 magma_type_t;" in crossgl
    assert "struct cpu_blas_gemv_func_float" in crossgl
    assert "cl_command_queue" not in crossgl


def test_external_arrayfire_magma_dependent_typename_inline_codegen_reparse():
    source_info = EXTERNAL_FIXTURE_SOURCES["arrayfire_magma_dependent_typename_inline"]
    assert source_info["commit"] == "492718b5a256d4a9d5198fdce89d8fd21772bfda"
    assert source_info["path"] == "src/backend/opencl/magma/magma_blas_clblast.h"

    source = """
    typedef cl_float2 cfloat;

    template<typename T>
    struct CLBlastType {
        using Type = T;
    };

    template<typename T>
    typename CLBlastType<T>::Type inline toCLBlastConstant(const T val);

    kernel void arrayfire_magma_probe(global float *out, const float value) {
        out[0] = toCLBlastConstant(value);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    helper = ast.statements[2]
    assert helper.return_type == "CLBlastType<T>::Type"
    assert helper.qualifiers == ["inline"]
    assert "CLBlastType<T>::Type toCLBlastConstant(T val)" in crossgl


def test_external_opencv_opencl_const_parameter_codegen_reparse():
    source_info = EXTERNAL_FIXTURE_SOURCES["opencv_optical_flow_opencl_const_parameter"]
    assert source_info["commit"] == "6f29af625bb4617e2e061f8097b5f3e2ed341a82"
    assert source_info["path"] == "modules/video/src/opencl/optical_flow_farneback.cl"

    source = """
    __kernel void polynomialExpansion(__global __const float *src,
                                      __global float *dst,
                                      __local float *smem) {
        __local float *row = smem + get_local_id(0);
        dst[0] = src[0] + row[0];
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    assert ast.statements[0].params[0] == {
        "type": "__global__ const float *",
        "name": "src",
    }
    assert "var<storage, read_write> src: array<f32>" in crossgl
    assert "__const" not in crossgl


def test_external_opencv_batchnorm_macro_type_cast_codegen_reparse():
    source_info = EXTERNAL_FIXTURE_SOURCES["opencv_batchnorm_macro_type_cast"]
    assert source_info["commit"] == "6f29af625bb4617e2e061f8097b5f3e2ed341a82"
    assert source_info["path"] == "modules/dnn/src/opencl/batchnorm.cl"

    source = """
    __kernel void batch_norm1(__global const Dtype* src,
                              __global const float* weight,
                              __global const float* bias,
                              __global Dtype* dst) {
        int index = get_global_id(0);
        float w = weight[index];
        float b = bias[index];
        float_type src_vec = convert_f(src[index]);
        float_type dst_vec = src_vec * w + (float_type)b;
        dst[index] = convert_T(dst_vec);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    cast = ast.statements[0].body[4].value.right
    assert cast.target_type == "float_type"
    assert "float_type(b)" in crossgl


def test_external_opencv_filter_sep_col_macro_type_cast_codegen_reparse():
    source_info = EXTERNAL_FIXTURE_SOURCES["opencv_filter_sep_col_macro_type_cast"]
    assert source_info["commit"] == "6f29af625bb4617e2e061f8097b5f3e2ed341a82"
    assert source_info["path"] == "modules/imgproc/src/opencl/filterSepCol.cl"

    source = """
    kernel void opencv_filter_probe(global float *dst, float delta) {
        srcT sum = (srcT)delta;
        dst[0] = sum;
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    cast = ast.statements[0].body[0].value
    assert cast.target_type == "srcT"
    assert "var sum: srcT = srcT(delta);" in crossgl


def test_external_opencv_morph3x3_vector_scalar_cast_codegen_reparse():
    source_info = EXTERNAL_FIXTURE_SOURCES["opencv_morph3x3_vector_scalar_cast"]
    assert source_info["commit"] == "6f29af625bb4617e2e061f8097b5f3e2ed341a82"
    assert source_info["path"] == "modules/imgproc/src/opencl/morph3x3.cl"

    source = """
    #define VAL 0
    kernel void opencv_morph3x3_probe(global uchar16 *out,
                                      int y,
                                      uchar16 line) {
        out[0] = (y == 0) ? (uchar16)VAL: as_uchar16(line);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    ternary = ast.statements[0].body[0].right
    assert ternary.true_expr.target_type == "uchar16"
    assert "array<u8, 16>(0)" in crossgl


def test_external_pocl_noinline_function_specifier_codegen_reparse():
    source_info = EXTERNAL_FIXTURE_SOURCES["pocl_noinline_function_specifier"]
    assert source_info["commit"] == "d11f27f3ba667456466cd935dacaf69e5cbf2598"
    assert source_info["path"] == "tests/kernel/test_as_type.cl"

    source = """
    _CL_NOINLINE
    void clear_bytes(uchar* p, uchar c, size_t n)
    {
      for (size_t i = 0; i < n; ++i) {
        p[i] = c;
      }
    }

    kernel void test_as_type(global uchar *out, uchar value) {
      clear_bytes(out, value, 1);
    }
    """

    ast, crossgl = assert_crossgl_reparses(source)

    helper = ast.statements[0]
    assert helper.name == "clear_bytes"
    assert helper.qualifiers == ["_CL_NOINLINE"]
    assert "void clear_bytes(ptr<u8> p, u8 c, u32 n)" in crossgl
