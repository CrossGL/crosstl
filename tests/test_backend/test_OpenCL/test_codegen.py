from crosstl.backend.OpenCL.OpenCLCrossGLCodeGen import OpenCLToCrossGLConverter
from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser


def generate_crossgl(source):
    tokens = OpenCLLexer(source).tokenize()
    ast = OpenCLParser(tokens).parse()
    crossgl = OpenCLToCrossGLConverter().generate(ast)
    CrossGLParser(CrossGLLexer(crossgl).tokens).parse()
    return crossgl


def test_opencl_kernel_codegen_reparses_and_lowers_builtin_ids():
    crossgl = generate_crossgl("""
        kernel void saxpy(global float *out, const global float *x, float a) {
            const uint gid = get_global_id(0);
            const uint lid = get_local_id(0);
            out[gid] = a * x[gid] + (float)lid;
        }
        """)

    assert "// OpenCL to CrossGL conversion" in crossgl
    assert "fn saxpy(" in crossgl
    assert "@group(0) @binding(0) var<storage, read_write> out: array<f32>" in crossgl
    assert "var gid: u32 = gl_GlobalInvocationID.x;" in crossgl
    assert "var lid: u32 = gl_LocalInvocationID.x;" in crossgl


def test_opencl_local_memory_and_barrier_codegen_reparse():
    crossgl = generate_crossgl("""
        __attribute__((reqd_work_group_size(256, 1, 1)))
        kernel void reduce_first_kernel(global float *out, const global float *in) {
            const uint lid = get_local_id(0);
            local float scratch[256];
            scratch[lid] = in[lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid == 0) {
                out[get_group_id(0)] = scratch[0];
            }
        }
        """)

    assert "@workgroup_size(256, 1, 1)" in crossgl
    assert "var<workgroup> scratch: array<f32, 256>;" in crossgl
    assert "workgroupBarrier();" in crossgl
    assert "out[gl_WorkGroupID.x] = scratch[0];" in crossgl


def test_opencl_vector_constructor_cast_codegen_reparse():
    crossgl = generate_crossgl("""
        __kernel void gaussian_filter(__read_only image2d_t srcImg,
                                      __write_only image2d_t dstImg) {
            int2 coord = (int2)(get_global_id(0), get_global_id(1));
            float4 color = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            write_imagef(dstImg, coord, color);
        }
        """)

    assert "var coord: vec2<i32> = vec2<i32>(" in crossgl
    assert "var color: vec4<f32> = vec4<f32>(" in crossgl
