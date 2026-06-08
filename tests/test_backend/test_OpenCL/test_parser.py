from crosstl.backend.OpenCL.OpenCLAst import FunctionCallNode, KernelNode, SyncNode
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
