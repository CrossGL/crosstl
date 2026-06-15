from crosstl.backend.OpenCL.OpenCLAst import (
    ForNode,
    FunctionCallNode,
    KernelNode,
    OpenCLStatementExpressionNode,
    SwitchNode,
    SyncNode,
)
from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser, OpenCLProgramNode


def parse_code(source):
    tokens = OpenCLLexer(source).tokenize()
    return OpenCLParser(tokens).parse()


def test_kernel_qualifier_before_return_type_parses():
    ast = parse_code("""
        kernel void saxpy(global float *out, const global float *x, float a) {
            const uint gid = get_global_id(0);
            out[gid] = a * x[gid];
        }
        """)

    assert isinstance(ast, OpenCLProgramNode)
    kernel = ast.statements[0]
    assert isinstance(kernel, KernelNode)
    assert kernel.name == "saxpy"
    assert kernel.params[0]["type"] == "__global__ float *"
    assert kernel.params[1]["type"] == "const __global__ float *"


def test_pocl_style_kernel_qualifier_after_return_type_parses():
    ast = parse_code("""
        void __kernel memfill(global uint *mem, const uint pattern) {
            size_t gid = get_global_id(0);
            mem[gid] = pattern;
        }
        """)

    kernel = ast.statements[0]
    assert isinstance(kernel, KernelNode)
    assert kernel.name == "memfill"


def test_hashcat_kernel_specifier_macros_parse_as_kernel():
    ast = parse_code("""
        KERNEL_FQ KERNEL_FA void amp(global ulong *pws, const ulong gid_max) {
            const ulong gid = get_global_id(0);
            if (gid >= gid_max) return;
            pws[gid] = gid;
        }
        """)

    kernel = ast.statements[0]
    assert isinstance(kernel, KernelNode)
    assert kernel.name == "amp"
    assert kernel.params[0]["type"] == "__global__ ulong *"
    assert kernel.params[1]["type"] == "const ulong"


def test_hashcat_opaque_kernel_parameter_pack_macro_parses():
    ast = parse_code("""
        KERNEL_FQ KERNEL_FA void m00000_m04(KERN_ATTR_RULES ()) {
            const ulong gid = get_global_id(0);
            if (gid >= GID_CNT) return;
            pws[gid] = gid;
        }
        """)

    kernel = ast.statements[0]
    assert isinstance(kernel, KernelNode)
    assert kernel.name == "m00000_m04"
    assert kernel.params == []


def test_hashcat_declspec_function_specifier_macro_parses():
    ast = parse_code("""
        DECLSPEC u32 MurmurHash64A_truncated(PRIVATE_AS const u32 *data,
                                             const u32 len) {
            u64 hash = len * 0xc6a4a7935bd1e995;
            return (u32)(hash >> 32);
        }
        """)

    helper = ast.statements[0]
    assert helper.name == "MurmurHash64A_truncated"
    assert helper.return_type == "u32"
    assert helper.qualifiers == ["DECLSPEC"]
    assert helper.params[0]["type"] == "__private__ const u32 *"


def test_hashcat_parenthesized_array_access_shift_index_parses():
    ast = parse_code("""
        kernel void cast_lookup(global uint *out, global uint *x, global uint *table) {
            out[0] = table[(uint)(uchar)(((x[0xD / 4]) >> (8 * (3 - 0xD % 4))))];
        }
        """)

    kernel = ast.statements[0]
    assert isinstance(kernel, KernelNode)
    assert kernel.name == "cast_lookup"


def test_hashcat_address_space_macro_declaration_parses():
    ast = parse_code("""
        CONSTANT_VK uint table[2] = { 1, 2 };

        kernel void compare(global uint *out, GLOBAL_AS const digest_t *digests_buf) {
            for (int digest_pos = 0; digest_pos < DIGESTS_CNT; digest_pos++) {
                GLOBAL_AS const digest_t *digest = digests_buf + digest_pos;
                out[digest_pos] = digest->digest_buf[0];
            }
        }
        """)

    assert ast.statements[0].qualifiers == ["__constant__"]
    kernel = ast.statements[1]
    assert isinstance(kernel, KernelNode)
    assert kernel.params[1]["type"] == "__global__ const digest_t *"


def test_local_memory_and_barrier_parse_from_arrayfire_pattern():
    ast = parse_code("""
        kernel void reduce_first_kernel(global float *out, const global float *in) {
            const uint lid = get_local_id(0);
            local float scratch[256];
            scratch[lid] = in[lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            out[get_group_id(0)] = scratch[0];
        }
        """)

    body = ast.statements[0].body
    assert body[1].qualifiers == ["__shared__"]
    assert isinstance(body[3], SyncNode)
    assert isinstance(body[0].value, FunctionCallNode)
    assert body[0].value.name == "get_local_id"


def test_macro_style_type_alias_cast_parses():
    ast = parse_code("""
        void load2LocalMem(global const inType* in) {
            outType value = (outType)in[0];
        }
        """)

    value = ast.statements[0].body[0]
    assert value.vtype == "outType"
    assert value.name == "value"


def test_underscored_image_access_qualifiers_parse():
    ast = parse_code("""
        __kernel void gaussian_filter(__read_only image2d_t srcImg,
                                      __write_only image2d_t dstImg) {
        }
        """)

    kernel = ast.statements[0]
    assert isinstance(kernel, KernelNode)
    assert kernel.params[0]["type"] == "read_only image2d_t"
    assert kernel.params[1]["type"] == "write_only image2d_t"


def test_parameter_attribute_between_pointer_and_name_parses():
    ast = parse_code("""
        void aligned_copy(global uchar * __attribute__((align_value(16))) dst,
                          const global uchar * __attribute__((align_value(16))) src,
                          uint n) {
            for (uint i = 0; i < n; i++) {
                dst[i] = src[i];
            }
        }
        """)

    helper = ast.statements[0]
    assert helper.name == "aligned_copy"
    assert helper.params[0] == {"type": "__global__ uchar *", "name": "dst"}
    assert helper.params[1] == {"type": "const __global__ uchar *", "name": "src"}
    assert helper.params[2] == {"type": "uint", "name": "n"}


def test_volatile_global_unsigned_pointer_cast_parses():
    ast = parse_code("""
        void atomicAdd(volatile __global T *ptr, T val) {
            current.u = atomic_cmpxchg((volatile __global unsigned *)ptr,
                                       expected.u,
                                       next.u);
        }
        """)

    assignment = ast.statements[0].body[0]
    assert assignment.right.operation == "atomic_cmpxchg"
    assert assignment.right.args[0].target_type == "volatile __global__ unsigned int *"


def test_macro_style_casts_in_ternary_branches_parse():
    ast = parse_code("""
        kernel void identity_kernel(global T *out) {
            unsigned idx = get_global_id(0);
            unsigned idy = get_global_id(1);
            T val = (idx == idy) ? (T)(ONE) : (T)(ZERO);
            out[idx] = val;
        }
        """)

    value = ast.statements[0].body[2].value
    assert value.true_expr.target_type == "T"
    assert value.false_expr.target_type == "T"


def test_macro_expanded_parenthesized_lvalue_assignment_parses():
    ast = parse_code("""
        kernel void expanded_lvalue(local T *scratch) {
            int tid_y = get_local_id(1);
            int tid_x = get_local_id(0);
            (scratch[(tid_y) * 81 + (0) * 9 + (tid_x * 2)]) = 0.0f;
        }
        """)

    assignment = ast.statements[0].body[2]
    assert assignment.operator == "="


def test_public_inline_function_specifier_macros_parse():
    ast = parse_code("""
        __inline unsigned popcount(unsigned x) { return x; }
        INLINE_FUNC void helper(const __global real *src, LOCAL_PTR real *dst) {
            dst[0] = src[0];
        }
        """)

    assert [stmt.name for stmt in ast.statements] == ["popcount", "helper"]


def test_line_break_after_binary_operator_parses():
    ast = parse_code("""
        kernel void line_broken(global float *out, global const float *center) {
            float value = (center[0] - center[1] -
                           center[2] + center[3]);
            out[0] = value;
        }
        """)

    value = ast.statements[0].body[0].value
    assert value.op == "+"


def test_khronos_histogram_newline_after_for_lparen_declaration_parses():
    ast = parse_code("""
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
        """)

    loop = ast.statements[0].body[0]
    assert isinstance(loop, ForNode)
    assert loop.init.name == "channel"
    assert loop.init.vtype == "uint"
    assert loop.condition.op == "<"
    assert loop.update.op == "++_POST"


def test_darktable_pointer_to_array_const_declarator_parses():
    ast = parse_code("""
        static inline int FCxtrans(const int row,
                                  const int col,
                                  global const unsigned char (*const xtrans)[6]) {
            return xtrans[row][col];
        }
        """)

    helper = ast.statements[0]
    assert helper.name == "FCxtrans"
    assert helper.params[2]["name"] == "xtrans"
    assert helper.params[2]["type"] == "__global__ const unsigned char (*)[6]"


def test_darktable_gnu_statement_expression_switch_macro_parses():
    ast = parse_code("""
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
        """)

    statement_expr = ast.statements[0].body[0]
    assert isinstance(statement_expr, OpenCLStatementExpressionNode)
    assert isinstance(statement_expr.statements[0], SwitchNode)


def test_opencl_generic_address_space_pointer_parameter_parses():
    ast = parse_code("""
        int read_generic(generic int *p) {
            return *p;
        }

        kernel void generic_probe(global int *out) {
            out[0] = read_generic(out);
        }
        """)

    helper = ast.statements[0]
    assert helper.params[0] == {"type": "__generic__ int *", "name": "p"}
    assert ast.statements[1].name == "generic_probe"


def test_clspv_postfix_volatile_pointer_declarator_parses():
    ast = parse_code("""
        kernel void volatile_pointer_probe(global uint *results) {
            __private float val;
            val = 0.1f;
            float *volatile ptr = &val;
            results[0] = ptr != 0;
        }
        """)

    ptr = ast.statements[0].body[2]
    assert ptr.vtype == "float * volatile"
    assert ptr.name == "ptr"


def test_clspv_line_broken_private_qualifier_declaration_parses():
    ast = parse_code("""
        void kernel private_array_probe(global int *out) {
        private
          unsigned char tab[128];
          out[0] = tab[0];
        }
        """)

    tab = ast.statements[0].body[0]
    assert tab.qualifiers == ["__private__"]
    assert tab.vtype == "unsigned char[128]"
    assert tab.name == "tab"
