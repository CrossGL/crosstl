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
