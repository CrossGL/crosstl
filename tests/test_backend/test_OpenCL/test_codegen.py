import pytest

import crosstl
from crosstl.backend.common_ast import UnaryOpNode
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.backend.OpenCL.OpenCLCrossGLCodeGen import OpenCLToCrossGLConverter
from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser
from crosstl.backend.OpenCL.target_lowering import (
    OpenCLTargetUnsupportedError,
    validate_opencl_intermediate_for_target,
)
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser


def parse_crossgl_from_opencl(source):
    tokens = OpenCLLexer(source).tokenize()
    ast = OpenCLParser(tokens).parse()
    crossgl = OpenCLToCrossGLConverter().generate(ast)
    cgl_ast = CrossGLParser(CrossGLLexer(crossgl).tokens).parse()
    return cgl_ast, crossgl


def generate_crossgl(source):
    _cgl_ast, crossgl = parse_crossgl_from_opencl(source)
    return crossgl


def test_opencl_event_token_extraction_handles_identifier_spellings():
    converter = OpenCLToCrossGLConverter()
    assert converter.opencl_event_token_name("read") == "read"
    assert converter.opencl_event_token_name(UnaryOpNode("&", "read")) == "read"

    ast = OpenCLParser(OpenCLLexer("""
            kernel void reduce(global int *front, local int *shared) {
                event_t read;
                async_work_group_copy(shared, front, 1, read);
                wait_group_events(1, &read);
            }
            """).tokenize()).parse()

    assert converter.collect_opencl_event_tokens(ast) == {"read"}


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


def test_opencl_saxpy_signed_global_id_lowers_to_wgsl_casted_local(tmp_path):
    source = """
        kernel void saxpy(global float *out, const global float *x, float a) {
            const int gid = get_global_id(0);
            out[gid] = a * x[gid];
        }
        """
    crossgl = generate_crossgl(source)
    opencl_path = tmp_path / "saxpy.cl"
    opencl_path.write_text(source, encoding="utf-8")
    wgsl = crosstl.translate(str(opencl_path), backend="wgsl", format_output=False)

    assert "var gid: i32 = gl_GlobalInvocationID.x;" in crossgl
    assert "var gid: i32 = i32(global_invocation_id.x);" in wgsl
    assert "out[gid] = (_saxpy_Args.a * x[gid]);" in wgsl


def test_opencl_saxpy_fma_lowers_to_directx_mad(tmp_path):
    source = """
        kernel void saxpy(global float *y, global const float *x, const float a) {
            const uint gid = get_global_id(0);
            y[gid] = fma(a, x[gid], y[gid]);
        }
        """
    opencl_path = tmp_path / "saxpy.cl"
    opencl_path.write_text(source, encoding="utf-8")

    hlsl = crosstl.translate(str(opencl_path), backend="directx", format_output=False)

    assert "y[gid] = mad(a, x[gid], y[gid]);" in hlsl
    assert "fma(" not in hlsl
    HLSLParser(HLSLLexer(hlsl).tokenize()).parse()


def test_hashcat_kernel_specifier_macros_codegen_reparse():
    crossgl = generate_crossgl("""
        KERNEL_FQ KERNEL_FA void amp(global ulong *pws, const ulong gid_max) {
            const ulong gid = get_global_id(0);
            if (gid >= gid_max) return;
            pws[gid] = gid;
        }
        """)

    assert "@compute" in crossgl
    assert "fn amp(" in crossgl
    assert "var gid: u64 = gl_GlobalInvocationID.x;" in crossgl


def test_hashcat_opaque_kernel_parameter_pack_macro_codegen_reparse():
    crossgl = generate_crossgl("""
        KERNEL_FQ KERNEL_FA void m00000_m04(KERN_ATTR_RULES ()) {
            const ulong gid = get_global_id(0);
            if (gid >= GID_CNT) return;
            pws[gid] = gid;
        }
        """)

    assert "@compute" in crossgl
    assert "fn m00000_m04(" in crossgl
    assert "GID_CNT" in crossgl


def test_hashcat_parenthesized_array_access_shift_index_codegen_reparse():
    crossgl = generate_crossgl("""
        kernel void cast_lookup(global uint *out, global uint *x, global uint *table) {
            out[0] = table[(uint)(uchar)(((x[0xD / 4]) >> (8 * (3 - 0xD % 4))))];
        }
        """)

    assert "fn cast_lookup(" in crossgl
    assert ">>" in crossgl
    assert "table[" in crossgl


def test_hashcat_address_space_macro_parameters_codegen_reparse():
    crossgl = generate_crossgl("""
        kernel void copy_macro_spaces(GLOBAL_AS const uint *in_buf,
                                      GLOBAL_AS uint *out,
                                      LOCAL_AS uint *scratch) {
            uint lid = get_local_id(0);
            scratch[lid] = in_buf[lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            out[lid] = scratch[lid];
        }
        """)

    assert "@binding(0) var<storage, read> in_buf: array<u32>" in crossgl
    assert "@binding(1) var<storage, read_write> out: array<u32>" in crossgl
    assert "var<workgroup> scratch: array<u32>" in crossgl


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


def test_opencl_local_pointer_kernel_param_uses_workgroup_storage():
    crossgl = generate_crossgl("""
        kernel void local_arg_first(uint scale, local float *scratch, global float *out) {
            uint lid = get_local_id(0);
            scratch[lid] = out[lid] * scale;
            barrier(CLK_LOCAL_MEM_FENCE);
            out[lid] = scratch[lid];
        }
        """)

    assert "var<workgroup> scratch: array<f32>" in crossgl
    assert "@group(0) @binding(0) var<storage, read_write> out: array<f32>" in crossgl
    assert "var<storage, read_write> scratch" not in crossgl
    assert "@binding(1) var<storage, read_write> out" not in crossgl


def test_opencl_reduce_helpers_report_structured_target_contracts():
    cgl_ast, _crossgl = parse_crossgl_from_opencl("""
        int op(int lhs, int rhs);
        int work_group_reduce_op(int val);
        int sub_group_reduce_op(int val);

        int read_local(local int* shared, size_t i) {
            return shared[i];
        }

        kernel void reduce(global int* front, global int* back, local int* shared) {
            event_t read;
            async_work_group_copy(shared, front, 1, read);
            wait_group_events(1, &read);
            int temp = work_group_reduce_op(
                op(read_local(shared, 0), read_local(shared, 1))
            );
            back[0] = sub_group_reduce_op(temp);
        }
        """)

    with pytest.raises(OpenCLTargetUnsupportedError) as error:
        validate_opencl_intermediate_for_target(cgl_ast, "opengl")

    contracts = [contract.to_json() for contract in error.value.contracts]

    assert str(error.value).startswith(
        "opencl.target.unsupported: cannot lower OpenCL source to opengl"
    )
    assert {
        contract["signature"]
        for contract in contracts
        if contract["code"] == "opencl.target.unresolved-helper"
    } == {
        "op(lhs: i32, rhs: i32) -> i32",
        "work_group_reduce_op(val: i32) -> i32",
        "sub_group_reduce_op(val: i32) -> i32",
    }
    assert {
        contract["signature"]
        for contract in contracts
        if contract["code"] == "opencl.target.pointer-helper-parameter"
    } == {"read_local(shared_: ptr<i32>)"}
    assert {
        contract["signature"]
        for contract in contracts
        if contract["code"] == "opencl.target.unsupported-builtin"
    } == {"async_work_group_copy(...)", "wait_group_events(...)"}
    assert all(contract["action"] for contract in contracts)


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
    assert "image2D srcImg @binding(0) @readonly" in crossgl
    assert "image2D dstImg @binding(1) @writeonly" in crossgl


def test_opencl_image_and_sampler_kernel_params_keep_resource_metadata():
    crossgl = generate_crossgl("""
        __kernel void sample_and_store(__read_only image2d_t srcImg,
                                       sampler_t smp,
                                       __write_only image2d_t dstImg) {
            int2 coord = (int2)(get_global_id(0), get_global_id(1));
            float4 color = read_imagef(srcImg, smp, coord);
            write_imagef(dstImg, coord, color);
        }
        """)

    assert "image2D srcImg @binding(0) @readonly" in crossgl
    assert "sampler smp @binding(1)" in crossgl
    assert "image2D dstImg @binding(2) @writeonly" in crossgl


def test_opencl_signed_char_pointer_lowers_to_crossgl_i8_array():
    crossgl = generate_crossgl("""
        kernel void pocl_add_i8(global signed char *a,
                                global signed char *b,
                                global signed char *out) {
            uint gid = get_global_id(0);
            out[gid] = a[gid] + b[gid];
        }
        """)

    assert "a: array<i8>" in crossgl
    assert "b: array<i8>" in crossgl
    assert "out: array<i8>" in crossgl


def test_opencl_typedef_alias_chain_lowers_to_reparseable_crossgl():
    crossgl = generate_crossgl("""
        typedef float real;
        typedef real real_arg;

        kernel void scale(global real *out, real_arg alpha) {
            uint gid = get_global_id(0);
            out[gid] = alpha;
        }
        """)

    assert "typedef f32 real;" in crossgl
    assert "typedef f32 real_arg;" in crossgl
    assert "out: array<f32>" in crossgl
    assert "f32 alpha" in crossgl


def test_opencl_host_embedded_source_string_is_skipped_in_crossgl():
    crossgl = generate_crossgl("""
        static const char *embedded_kernel = "
        __kernel void not_a_top_level_kernel(__global float *out) {
            out[0] = 1.0f;
        }
        ";
        """)

    assert "// skipped host OpenCL source string: embedded_kernel" in crossgl
    assert "not_a_top_level_kernel" not in crossgl


def test_opencl_host_string_template_preprocessor_errors_are_inert():
    crossgl = generate_crossgl("""
        const char *SYMM_C_KERNEL = "
        #if !defined(__SYMM_UPPER__)
        #error Upper or Lower must be defined
        #endif
        __kernel void not_a_top_level_kernel(__global float *out) {
            out[0] = 1.0f;
        }
        ";
        """)

    assert "// skipped host OpenCL source string: SYMM_C_KERNEL" in crossgl
    assert "not_a_top_level_kernel" not in crossgl


def test_opencl_standalone_real_alias_falls_back_to_f32():
    crossgl = generate_crossgl("""
        typedef real realV;

        INLINE_FUNC realV multiply(realV value, const real scale) {
            return value * scale;
        }
        """)

    assert "typedef f32 realV;" in crossgl
    assert "f32 multiply(f32 value, f32 scale)" in crossgl


def test_opencv_anonymous_enum_and_constant_array_codegen_reparse():
    crossgl = generate_crossgl("""
        enum
        {
            hsv_shift = 12
        };

        __constant int sector_data[][3] = { { 1, 3, 0 },
                                            { 1, 0, 2 } };

        __kernel void RGB2HSV(__global uchar *dst) {
            int h = hsv_shift;
            dst[sector_data[0][0]] = (uchar)h;
        }
        """)

    assert "enum  {" not in crossgl
    assert "const i32 hsv_shift = 12;" in crossgl
    assert "sector_data: array<array<i32, 3>>" in crossgl


def test_darktable_unsigned_char_pointer_to_array_kernel_param_codegen_reparse():
    crossgl = generate_crossgl("""
        static inline int FCxtrans(const int row,
                                  const int col,
                                  global const unsigned char (*const xtrans)[6]) {
            return xtrans[row][col];
        }

        kernel void capture_xtrans(global float *out,
                                   const int row,
                                   const int col,
                                   global const unsigned char (*const xtrans)[6]) {
            out[0] = (float)FCxtrans(row, col, xtrans);
        }
        """)

    assert "xtrans: array<array<u8, 6>>" in crossgl
    assert "unsigned char ()" not in crossgl


def test_darktable_float_pointer_to_array_kernel_param_codegen_reparse():
    crossgl = generate_crossgl("""
        kernel void rgblevels_sample(global float *out,
                                     const float (*const levels)[3]) {
            out[0] = levels[0][0];
        }
        """)

    assert "levels: array<array<f32, 3>>" in crossgl
    assert "float ()" not in crossgl


def test_clspv_unnamed_local_pointer_kernel_param_codegen_reparse():
    crossgl = generate_crossgl("""
        kernel void k0(int v, local int *, global int *b) {
        }
        """)

    assert "i32 v" in crossgl
    assert "var<workgroup> _param1: array<i32>" in crossgl
    assert "@group(0) @binding(0) var<storage, read_write> b: array<i32>" in crossgl


def test_clspv_struct_return_and_local_elaborated_type_codegen_reparse():
    crossgl = generate_crossgl("""
        struct T {
            global int *ptr;
        };

        struct T bar(global int *out) {
            struct T t;
            t.ptr = out;
            return t;
        }

        kernel void foo(global int *out) {
            struct T t = bar(out);
            *(t.ptr) = 42;
        }
        """)

    assert "T bar(ptr<i32> out)" in crossgl
    assert "var t: T;" in crossgl
    assert "var t: T = bar(out);" in crossgl


def test_pocl_statement_expression_macro_block_codegen_reparse():
    crossgl = generate_crossgl("""
        DEFINE_BODY_G
        (test_rotate,
         ({
           int patterns[] = {0x01, 0x80};
         })
        )
        """)

    assert "// OpenCL macro block: DEFINE_BODY_G(" in crossgl
    assert "test_rotate" in crossgl
