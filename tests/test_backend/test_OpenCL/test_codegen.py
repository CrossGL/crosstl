import pytest

import crosstl
from crosstl.backend.common_ast import UnaryOpNode
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.backend.OpenCL.OpenCLCrossGLCodeGen import (
    OpenCLAtomicOrderUnsupportedError,
    OpenCLFenceScopeUnsupportedError,
    OpenCLImageReadUnsupportedError,
    OpenCLImageWriteUnsupportedError,
    OpenCLToCrossGLConverter,
)
from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser
from crosstl.backend.OpenCL.target_lowering import (
    OpenCLTargetUnsupportedError,
    validate_opencl_intermediate_for_target,
)
from crosstl.translator.diagnostics import TranslationDiagnostic
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


def test_opencl_explicit_atomics_drop_memory_order_and_scope():
    crossgl = generate_crossgl("""
        kernel void probe(global int *iin, global atomic_int *outp) {
            uint i = get_global_id(0);
            atomic_fetch_sub_explicit(&outp[i], iin[i], memory_order_relaxed);
            atomic_fetch_add_explicit(
                &outp[i], iin[i], memory_order_relaxed, memory_scope_device);
            atomic_exchange_explicit(&outp[i], iin[i], memory_order_relaxed);
            atomic_fetch_or_explicit(
                &outp[i], iin[i], memory_order_relaxed, memory_scope_work_group);
        }
        """)

    assert "memory_order" not in crossgl
    assert "memory_scope" not in crossgl
    assert "atomicSub(outp[i], iin[i]);" in crossgl
    assert "atomicAdd(outp[i], iin[i]);" in crossgl
    assert "atomicExchange(outp[i], iin[i]);" in crossgl
    assert "atomicOr(outp[i], iin[i]);" in crossgl


@pytest.mark.parametrize(
    ("call", "reason"),
    [
        (
            "atomic_fetch_add_explicit(&outp[i], 1, memory_order_seq_cst)",
            "memory_order_seq_cst: CrossGL atomics are relaxed",
        ),
        (
            "atomic_exchange_explicit(&outp[i], 1, memory_order_acq_rel)",
            "memory_order_acq_rel: CrossGL atomics are relaxed",
        ),
        (
            "atomic_fetch_or_explicit(&outp[i], 1, order)",
            "with order: CrossGL atomics are relaxed",
        ),
        (
            "atomic_compare_exchange_strong_explicit(&outp[i], &expected, 1, "
            "memory_order_relaxed, memory_order_acquire)",
            "memory_order_acquire: CrossGL atomics are relaxed",
        ),
        (
            "atomic_fetch_max_explicit(&outp[i], 1, memory_order_relaxed, "
            "memory_scope_all_svm_devices)",
            "memory_scope_all_svm_devices: CrossGL atomics are atomic at device scope",
        ),
        (
            "atomic_fetch_min_explicit(&outp[i], 1)",
            "takes 3 or 4 arguments, got 2",
        ),
    ],
)
def test_opencl_explicit_atomics_refuse_orders_and_scopes_crossgl_cannot_keep(
    call, reason
):
    source = f"""
        kernel void probe(global atomic_int *outp, memory_order order) {{
            uint i = get_global_id(0);
            int expected = 0;
            {call};
        }}
        """

    with pytest.raises(OpenCLAtomicOrderUnsupportedError, match=reason) as raised:
        generate_crossgl(source)
    assert raised.value.code == "project.translate.opencl-atomic-order-unsupported"


def test_opencl_get_local_linear_id_lowers_to_local_invocation_index(tmp_path):
    source = """
        kernel __attribute__((reqd_work_group_size(1, 1, 1)))
        void probe(global uint *outp) {
            uint i = get_global_id(0);
            ulong r = get_local_linear_id();
            outp[i] = (uint)r;
        }
        """
    crossgl = generate_crossgl(source)

    assert "get_local_linear_id" not in crossgl
    assert "gl_LocalInvocationIndex" in crossgl

    opencl_path = tmp_path / "probe.cl"
    opencl_path.write_text(source, encoding="utf-8")
    hlsl = crosstl.translate(str(opencl_path), backend="directx", format_output=False)

    assert "get_local_linear_id" not in hlsl
    assert "SV_GroupIndex" in hlsl


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


def test_opencl_prefetch_is_dropped_as_a_hint():
    crossgl = generate_crossgl("""
        kernel void probe(global uint *uin, global uint *outp) {
            uint i = get_global_id(0);
            ulong n = (ulong)uin[i];
            prefetch((const global float *)outp, n);
            prefetch(outp + i, 4);
            prefetch(&outp[get_global_id(0)], 1);
            prefetch(outp, i > 0u ? 4 : 8);
            prefetch(outp, sizeof(uint));
            outp[i] = 0u;
        }
        """)

    assert "prefetch" not in crossgl
    assert "ptr<" not in crossgl
    assert "&outp" not in crossgl
    assert "?" not in crossgl
    assert "4;" not in [line.strip() for line in crossgl.splitlines()]
    assert "outp[i] = 0u;" in crossgl


def test_opencl_prefetch_keeps_argument_side_effects():
    crossgl = generate_crossgl("""
        kernel void probe(global float *outp) {
            uint i = get_global_id(0);
            uint j = i;
            uint n = 0u;
            prefetch(outp, i++);
            prefetch((const global float4 *)(outp + j++), n = 4u);
            outp[i + j] = (float)n;
        }
        """)

    assert "prefetch" not in crossgl
    assert "ptr<" not in crossgl
    assert "outp +" not in crossgl
    effects = [crossgl.index(effect) for effect in ("i++", "j++", "n = 4u;")]
    assert effects == sorted(effects)
    assert effects[-1] < crossgl.index("outp[(i + j)]")


def test_opencl_prefetch_translates_to_targets(tmp_path):
    opencl_path = tmp_path / "prefetch.cl"
    opencl_path.write_text("""
        __attribute__((reqd_work_group_size(1, 1, 1)))
        kernel void probe(global uint *uin, global uint *outp) {
            uint i = get_global_id(0);
            uint n = uin[i];
            prefetch((const global float *)outp, n);
            outp[i] = 0u;
        }
        """)

    for backend in ("directx", "metal", "wgsl"):
        output = crosstl.translate(
            str(opencl_path), backend=backend, format_output=False
        )
        assert "prefetch" not in output
        assert "outp[i] = 0u" in output


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


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_opencl_device_enqueue_event_builtins_are_refused(target):
    cgl_ast, _crossgl = parse_crossgl_from_opencl("""
        kernel void events(global uint* outp, global int* status) {
            uint i = get_global_id(0);
            clk_event_t e = create_user_event();
            retain_event(e);
            if (is_valid_event(e)) {
                set_user_event_status(e, status[i]);
                capture_event_profiling_info(
                    e, CLK_PROFILING_COMMAND_EXEC_TIME, (global void*)outp
                );
            }
            release_event(e);
            outp[i] = 0u;
        }
        """)

    with pytest.raises(OpenCLTargetUnsupportedError) as error:
        validate_opencl_intermediate_for_target(cgl_ast, target)

    contracts = [contract.to_json() for contract in error.value.contracts]

    assert "device-side enqueue event builtins have no target equivalent" in str(
        error.value
    )
    assert {contract["signature"] for contract in contracts} == {
        "capture_event_profiling_info(...)",
        "create_user_event(...)",
        "is_valid_event(...)",
        "release_event(...)",
        "retain_event(...)",
        "set_user_event_status(...)",
    }
    assert {contract["code"] for contract in contracts} == {
        "opencl.target.unsupported-builtin"
    }
    assert all(contract["action"] for contract in contracts)


def test_opencl_helper_named_like_device_enqueue_event_builtin_is_not_refused():
    # OpenCL C 1.2 does not reserve the device-side enqueue names.
    cgl_ast, crossgl = parse_crossgl_from_opencl("""
        uint retain_event(uint x) {
            return x + 1u;
        }

        kernel void events(global uint* outp) {
            uint i = get_global_id(0);
            outp[i] = retain_event(i);
        }
        """)

    validate_opencl_intermediate_for_target(cgl_ast, "opengl")
    assert "u32 retain_event(u32 x)" in crossgl


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


def test_opencl_write_image_builtins_lower_to_image_store():
    crossgl = generate_crossgl("""
        kernel void k(global float4 *src, write_only image2d_t tex,
                      write_only image2d_t itex, read_write image2d_t utex,
                      write_only image1d_t line, write_only image1d_buffer_t buf,
                      write_only image1d_array_t lines, write_only image3d_t vol,
                      write_only image2d_array_t layers) {
            int i = get_global_id(0);
            write_imagef(tex, (int2)(i, 0), src[i]);
            write_imagei(itex, (int2)(i, 1), (int4)(1, 2, 3, 4));
            write_imageui(utex, (int2)(i, 2), (uint4)(1u, 2u, 3u, 4u));
            write_imagef(line, i, src[i]);
            write_imagef(buf, i, src[i]);
            write_imagef(lines, (int2)(i, 1), src[i]);
            write_imagef(vol, (int4)(i, 0, 0, 0), src[i]);
            write_imagef(layers, (int4)(i, 0, 1, 0), src[i]);
        }
        """)

    assert "write_image" not in crossgl
    # The write function gives the image its element type.
    assert " image2D tex," in crossgl
    assert " iimage2D itex," in crossgl
    assert " uimage2D utex," in crossgl
    assert " imageBuffer buf," in crossgl
    assert "imageStore(tex, vec2<i32>(i, 0), src[i]);" in crossgl
    assert "imageStore(itex, vec2<i32>(i, 1), vec4<i32>(1, 2, 3, 4));" in crossgl
    assert "imageStore(utex, vec2<i32>(i, 2), vec4<u32>(1u, 2u, 3u, 4u));" in crossgl
    assert "imageStore(line, i, src[i]);" in crossgl
    assert "imageStore(buf, i, src[i]);" in crossgl
    assert "imageStore(lines, vec2<i32>(i, 1), src[i]);" in crossgl
    # 3D images and 2D image arrays take an int4 coordinate in OpenCL.
    assert "imageStore(vol, (vec4<i32>(i, 0, 0, 0)).xyz, src[i]);" in crossgl
    assert "imageStore(layers, (vec4<i32>(i, 0, 1, 0)).xyz, src[i]);" in crossgl


def test_opencl_write_image_translates_to_hlsl_texture_store(tmp_path):
    source = """
        kernel void k(global float *fin, write_only image2d_t dst,
                      write_only image3d_t idst, write_only image1d_buffer_t udst) {
            int i = get_global_id(0);
            write_imagef(dst, (int2)(i, 0), (float4)(fin[i], 0.0f, 0.0f, 1.0f));
            write_imagei(idst, (int4)(i, 0, 1, 0), (int4)(i));
            write_imageui(udst, i, (uint4)(1u));
        }
        """
    opencl_path = tmp_path / "write_image.cl"
    opencl_path.write_text(source, encoding="utf-8")

    hlsl = crosstl.translate(str(opencl_path), backend="directx", format_output=False)

    assert "write_image" not in hlsl
    # Every image, integer ones included, is a bound resource, not a k_Args member.
    assert "k_Args" not in hlsl
    assert "RWTexture2D<float4> dst : register(u" in hlsl
    assert "RWTexture3D<int4> idst : register(u" in hlsl
    assert "RWBuffer<uint4> udst : register(u" in hlsl
    assert "dst[int2(i, 0)] = float4(fin[i], 0.0, 0.0, 1.0);" in hlsl
    assert "idst[int4(i, 0, 1, 0).xyz] = int4(i, i, i, i);" in hlsl
    assert "udst[i] = uint4(1u, 1u, 1u, 1u);" in hlsl
    HLSLParser(HLSLLexer(hlsl).tokenize()).parse()


@pytest.mark.parametrize(
    "source, reason",
    [
        (
            """
            kernel void k(global float4 *v, write_only image2d_t tex) {
                write_imagef(tex, (int2)(0, 0), 1, v[0]);
            }
            """,
            "write_imagef with 4 arguments",
        ),
        (
            """
            kernel void k(global float *v, write_only image2d_depth_t tex) {
                write_imagef(tex, (int2)(0, 0), v[0]);
            }
            """,
            "writing to image2d_depth_t: CrossGL has no depth storage image",
        ),
        (
            """
            kernel void k(global float4 *v, write_only image2d_t tex) {
                write_imagef(tex, (int2)(0, 0), v[0]);
                write_imagei(tex, (int2)(0, 1), (int4)(1));
            }
            """,
            "write_imagei on tex of type image2D",
        ),
        (
            """
            void put(write_only image2d_t t, int4 v) {
                write_imagei(t, (int2)(0, 0), v);
            }
            kernel void k(global int4 *v, write_only image2d_t tex) {
                put(tex, v[0]);
            }
            """,
            "write_imagei on parameter t of helper function put",
        ),
    ],
)
def test_opencl_write_image_refuses_writes_imagestore_cannot_keep(source, reason):
    with pytest.raises(OpenCLImageWriteUnsupportedError, match=reason):
        generate_crossgl(source)


def test_opencl_read_imagef_lowers_to_texture_reads():
    crossgl = generate_crossgl("""
        __constant sampler_t smp = CLK_NORMALIZED_COORDS_TRUE |
            CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
        __constant sampler_t pix = CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
        float4 read_scaled(image2d_t img, sampler_t s, float2 c) {
            return read_imagef(img, s, c * 2.0f);
        }
        kernel void k(global const float *fin, global float *outp,
                      read_only image2d_t tex, image3d_t vol,
                      read_write image2d_t rw, sampler_t hs) {
            int i = get_global_id(0);
            float2 uv = (float2)(fin[i], 0.5f);
            outp[i] = read_imagef(tex, smp, (float2)(fin[i], 0.5f)).x;
            outp[i] += read_imagef(tex, pix, uv).y;
            outp[i] += read_imagef(tex, pix, (int2)(i, 0)).z;
            outp[i] += read_imagef(tex, (int2)(i, 0)).w;
            outp[i] += read_imagef(vol, smp, (float4)(fin[i], 0.5f, 0.5f, 0.0f)).x;
            outp[i] += read_imagef(vol, (int4)(i, 0, 0, 0)).x;
            outp[i] += read_imagef(rw, (int2)(i, 0)).x;
            outp[i] += read_imagef(tex, hs, uv).x;
            outp[i] += read_imagef(tex, CLK_FILTER_NEAREST, (int2)(i, 0)).x;
            outp[i] += read_scaled(tex, smp, uv).x;
        }
        """)

    assert "read_imagef" not in crossgl
    # A sampler_t constant fixes its CLK_* state, as a sampler(...) does.
    assert "CLK_" not in crossgl
    assert (
        "const sampler smp = sampler(coord::normalized, address::clamp_to_edge, "
        "filter::linear);" in crossgl
    )
    assert (
        "const sampler pix = sampler(coord::pixel, address::clamp_to_zero, "
        "filter::nearest);" in crossgl
    )
    # Images are read_only by default, so only read_write stays a storage image.
    assert "texture2D tex" in crossgl
    assert "texture3D vol" in crossgl
    assert "image2D rw" in crossgl
    assert "read_scaled(texture2D img, sampler s, vec2<f32> c)" in crossgl
    assert "textureLod(img, s, (c * 2.0f), 0.0)" in crossgl
    assert "textureLod(tex, smp, vec2<f32>(fin[i], 0.5f), 0.0).x" in crossgl
    assert "textureLod(tex, pix, uv, 0.0).y" in crossgl
    # An integer coordinate goes through the sampler as a pixel coordinate.
    assert "textureLod(tex, pix, vec2(vec2<i32>(i, 0)), 0.0).z" in crossgl
    assert "texelFetch(tex, vec2<i32>(i, 0), 0).w" in crossgl
    assert (
        "textureLod(vol, smp, (vec4<f32>(fin[i], 0.5f, 0.5f, 0.0f)).xyz, 0.0).x"
        in crossgl
    )
    assert "texelFetch(vol, (vec4<i32>(i, 0, 0, 0)).xyz, 0).x" in crossgl
    assert "imageLoad(rw, vec2<i32>(i, 0)).x" in crossgl
    assert "textureLod(tex, hs, uv, 0.0).x" in crossgl
    assert (
        "textureLod(tex, sampler(coord::pixel, address::clamp_to_edge, "
        "filter::nearest), vec2(vec2<i32>(i, 0)), 0.0).x" in crossgl
    )


PIXEL_SAMPLER_SOURCE = """
    __constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    __kernel void k(__global const float *fin, __global float4 *outp,
                    __read_only image2d_t tex) {
        int i = get_global_id(0);
        outp[i] = read_imagef(tex, smp, (float2)(fin[i], 0.5f));
    }
    """


def test_opencl_unnormalized_sampler_keeps_pixel_coordinates_for_metal(tmp_path):
    opencl_path = tmp_path / "pixel.cl"
    opencl_path.write_text(PIXEL_SAMPLER_SOURCE, encoding="utf-8")

    metal = crosstl.translate(str(opencl_path), backend="metal", format_output=False)

    assert (
        "sampler smp = sampler(coord::pixel, address::clamp_to_edge, "
        "filter::nearest);" in metal
    )
    assert "tex.sample(smp, float2(fin[i], 0.5), level(0.0))" in metal


@pytest.mark.parametrize("backend", ["directx", "vulkan", "opengl", "wgsl"])
def test_opencl_unnormalized_sampler_is_refused_where_targets_cannot_fix_it(
    tmp_path, backend
):
    opencl_path = tmp_path / "pixel.cl"
    opencl_path.write_text(PIXEL_SAMPLER_SOURCE, encoding="utf-8")

    with pytest.raises(TranslationDiagnostic) as raised:
        crosstl.translate(str(opencl_path), backend=backend)

    # The target refuses pixel coordinates, or fixed sampler state at all.
    code = raised.value.project_diagnostic_code
    assert code.startswith(f"project.translate.{backend}-")
    assert "sampler" in code


@pytest.mark.parametrize(
    ("read", "reason"),
    [
        ("read_imagef(tex, (float2)(0.5f))", "without a sampler takes integer"),
        ("read_imagef(rw, pix, (int2)(0))", "read_write image is read without"),
        ("read_imagef(tex, pix, (float2)(0.5f), 1.0f)", "4-argument form"),
        ("read_imagef(dep, pix, (float2)(0.5f))", "image2d_depth_t has no"),
        ("read_imagef(buf, 0)", "image1d_buffer_t has no"),
        ("read_imagef(vol, pix, (float4)(0.5f))", "pixel coordinates on a 3D"),
        ("read_imagef(arr, smp, (int4)(0))", "pixel coordinates on a 2DArray"),
        ("read_imagef(tex, pix, floor((float2)(0.5f)))", "element type of"),
        (
            "read_imagef(tex, CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR, (float2)(0.5f))",
            "repeats unnormalized",
        ),
        ("read_imagef(tex, CLK_ADDRESS_CLAMP, (int2)(0))", "no CLK_FILTER_* flag"),
        (
            "read_imagef(tex, CLK_FILTER_NEAREST | CLK_FILTER_LINEAR, (int2)(0))",
            "two filter",
        ),
        ("read_imagef(tex, 18, (int2)(0))", "only CLK_* sampler flags"),
        (
            "read_imagef(rw, (int2)(0)); write_imagei(rw, (int2)(0), (int4)(0))",
            "makes it an integer image",
        ),
    ],
)
def test_opencl_read_imagef_refuses_reads_crossgl_cannot_carry(read, reason):
    source = f"""
        __constant sampler_t smp = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;
        __constant sampler_t pix = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST;
        kernel void k(global float4 *outp, read_only image2d_t tex,
                      read_write image2d_t rw, read_only image2d_depth_t dep,
                      read_only image1d_buffer_t buf, read_only image3d_t vol,
                      read_only image2d_array_t arr) {{
            outp[0] = {read};
        }}
        """

    with pytest.raises(OpenCLImageReadUnsupportedError) as raised:
        generate_crossgl(source)

    assert reason in str(raised.value)
    assert (
        raised.value.project_diagnostic_code
        == "project.translate.opencl-image-read-unsupported"
    )


def test_opencl_sampled_read_imagef_translates_to_hlsl_sample_level(tmp_path):
    source = """
        __constant sampler_t smp = CLK_NORMALIZED_COORDS_TRUE |
            CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
        __kernel void k(__global const float *fin, __global float *outp,
                        __read_only image2d_t tex) {
            int i = get_global_id(0);
            outp[i] = read_imagef(tex, smp, (float2)(fin[i], 0.5f)).x;
        }
        """
    opencl_path = tmp_path / "read_image.cl"
    opencl_path.write_text(source, encoding="utf-8")

    hlsl = crosstl.translate(str(opencl_path), backend="directx", format_output=False)

    assert "read_imagef" not in hlsl
    # The host creates the fixed sampler from the header record.
    assert hlsl.startswith("// crosstl-fixed-samplers:")
    assert '"mag_filter":"linear"' in hlsl
    assert "SamplerState smp : register(s0);" in hlsl
    # The texture is a bound resource, not a member of the k_Args cbuffer.
    assert "k_Args" not in hlsl
    assert "Texture2D tex : register(t" in hlsl
    assert "outp[i] = tex.SampleLevel(smp, float2(fin[i], 0.5), 0.0).x;" in hlsl
    HLSLParser(HLSLLexer(hlsl).tokenize()).parse()


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


def test_opencl_nested_anonymous_records_materialize_named_crossgl_types():
    cases = (
        (
            "inner",
            "struct",
            """
            typedef struct Outer {
              struct { int value; } inner;
            } Outer;
            kernel void probe(global const int* input, global int* output) {
              Outer state;
              state.inner.value = input[0];
              output[0] = state.inner.value;
            }
            """,
        ),
        (
            "payload",
            "union",
            """
            typedef struct Outer {
              union { int value; uint bits; } payload;
            } Outer;
            kernel void probe(global const int* input, global int* output) {
              Outer state;
              state.payload.value = input[0];
              output[0] = state.payload.value;
            }
            """,
        ),
    )

    for member_name, keyword, source in cases:
        crossgl = generate_crossgl(source)
        assert crossgl == generate_crossgl(source)
        synthetic_prefix = f"CrossGLAnonymous_Outer_{member_name}_{keyword}_"
        assert f"struct {synthetic_prefix}" in crossgl
        assert f"{synthetic_prefix}" in crossgl
        assert f" {member_name};" in crossgl
        assert "hip_anonymous_union" not in crossgl
        assert "struct Outer {\n};" not in crossgl
        assert f"state.{member_name}.value" in crossgl


def test_opencl_nested_anonymous_struct_declarators_share_materialized_type():
    crossgl = generate_crossgl("""
        typedef struct Outer {
          struct { int value; } left, right;
        } Outer;
        kernel void probe(global const int* input, global int* output) {
          Outer state;
          state.left.value = input[0];
          state.right.value = input[1];
          output[0] = state.left.value + state.right.value;
        }
        """)

    declaration_lines = [
        line.strip()
        for line in crossgl.splitlines()
        if "CrossGLAnonymous_Outer_left_struct_" in line and "struct " not in line
    ]
    assert len(declaration_lines) == 2
    assert declaration_lines[0].split()[0] == declaration_lines[1].split()[0]
    assert declaration_lines[0].endswith(" left;")
    assert declaration_lines[1].endswith(" right;")


def test_opencl_nested_record_materialization_handles_named_injected_and_arrays():
    crossgl = generate_crossgl("""
        typedef struct Outer {
          struct Named { int named_value; } named;
          struct { int injected_value; };
          union { int selected; uint bits; };
          struct { int item_value; } items[2];
        } Outer;
        kernel void probe(global const int* input, global int* output) {
          Outer state;
          state.named.named_value = input[0];
          state.injected_value = input[1];
          state.selected = input[2];
          state.items[1].item_value = input[3];
          output[0] = state.named.named_value + state.injected_value
              + state.selected + state.items[1].item_value;
        }
        """)

    assert "struct Named {" in crossgl
    assert "Named named;" in crossgl
    assert "i32 injected_value;" in crossgl
    assert "i32 selected;" in crossgl
    assert "u32 bits;" in crossgl
    assert "array<CrossGLAnonymous_Outer_items_struct_" in crossgl
    assert ", 2> items;" in crossgl
    assert "struct <anonymous>" not in crossgl
    assert "hip_anonymous_union" not in crossgl
    assert "state.named.named_value" in crossgl
    assert "state.injected_value" in crossgl
    assert "state.selected" in crossgl
    assert "state.items[1].item_value" in crossgl


def test_opencl_function_local_anonymous_records_materialize_named_types():
    source = """
        kernel void probe(global const int* input, global int* output) {
          struct { int value; } left, right;
          union { int selected; uint bits; } payloads[2];
          left.value = input[0];
          right.value = input[1];
          payloads[1].selected = left.value + right.value;
          output[0] = payloads[1].selected;
        }
        """

    crossgl = generate_crossgl(source)
    assert crossgl == generate_crossgl(source)

    definitions = {
        line.split()[1]: line
        for line in crossgl.splitlines()
        if line.startswith("struct CrossGLFunctionLocal_probe_")
    }
    struct_name = next(name for name in definitions if "_left_struct_" in name)
    union_name = next(name for name in definitions if "_payloads_union_" in name)

    assert crossgl.index(f"struct {struct_name} {{") < crossgl.index("fn probe(")
    assert crossgl.index(f"struct {union_name} {{") < crossgl.index("fn probe(")
    assert f"var left: {struct_name};" in crossgl
    assert f"var right: {struct_name};" in crossgl
    assert f"var payloads: array<{union_name}, 2>;" in crossgl
    assert "struct None" not in crossgl
    assert "struct <anonymous>" not in crossgl
    assert "union <anonymous>" not in crossgl


def test_opencl_memory_fences_lower_to_crossgl_fences_by_flag():
    crossgl = generate_crossgl("""
        kernel void probe(global uint *outp) {
            uint i = get_global_id(0);
            read_mem_fence(CLK_LOCAL_MEM_FENCE);
            write_mem_fence(CLK_GLOBAL_MEM_FENCE);
            atomic_work_item_fence(
                CLK_IMAGE_MEM_FENCE, memory_order_acq_rel, memory_scope_work_group);
            read_mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
            outp[i] = 0u;
        }
        """)

    assert "CLK_" not in crossgl
    assert "mem_fence" not in crossgl
    assert "memoryBarrierShared();" in crossgl
    assert "memoryBarrierBuffer();" in crossgl
    assert "memoryBarrierImage();" in crossgl
    assert "memoryBarrier();" in crossgl
    assert "subgroupBarrier();" in crossgl


def test_opencl_mem_fence_and_scoped_fences_lower_to_crossgl_fences():
    crossgl = generate_crossgl("""
        kernel void probe(global uint *outp) {
            uint i = get_global_id(0);
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            atomic_work_item_fence(
                CLK_LOCAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
            sub_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_sub_group);
            outp[i] = 0u;
        }
        """)

    assert "mem_fence" not in crossgl
    assert "memory_" not in crossgl
    assert "memoryBarrierBuffer();" in crossgl
    assert "memoryBarrierShared();" in crossgl
    assert "subgroupBarrier();" in crossgl


@pytest.mark.parametrize(
    "call",
    [
        "atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acq_rel, "
        "memory_scope_all_svm_devices)",
        "sub_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_work_group)",
        "sub_group_barrier(CLK_LOCAL_MEM_FENCE, memory_scope_device)",
    ],
)
def test_opencl_fences_refuse_scopes_no_crossgl_fence_orders(call):
    with pytest.raises(OpenCLFenceScopeUnsupportedError, match="memory_scope_"):
        generate_crossgl(f"""
            kernel void probe(global uint *outp) {{
                {call};
                outp[get_global_id(0)] = 0u;
            }}
            """)
