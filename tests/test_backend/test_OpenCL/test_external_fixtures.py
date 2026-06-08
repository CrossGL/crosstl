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
