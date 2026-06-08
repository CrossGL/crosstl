from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer


def token_pairs(source):
    return [(token.type, token.value) for token in OpenCLLexer(source).tokenize()]


def test_opencl_kernel_and_address_space_tokens_are_normalized():
    tokens = token_pairs(
        "__kernel void k(__global const float *in, local float *scratch, "
        "LOCAL_PTR float *macro_scratch) { "
        "barrier(CLK_LOCAL_MEM_FENCE); }"
    )

    assert ("__GLOBAL__", "__kernel") in tokens
    assert ("__DEVICE__", "__global__") in tokens
    assert ("__SHARED__", "__shared__") in tokens
    assert ("SYNCTHREADS", "barrier") in tokens


def test_opencl_access_qualifier_tokens():
    tokens = token_pairs("read_only image2d_t src; write_only image2d_t dst;")

    assert ("READ_ONLY", "read_only") in tokens
    assert ("WRITE_ONLY", "write_only") in tokens


def test_cpp_raw_opencl_helper_literal_without_kernel_is_unwrapped():
    tokens = token_pairs("""
        R"(
        INLINE_FUNC void helper(const __global float *src, __local float *dst) {
            dst[0] = src[0];
        }
        )"
        """)

    assert ("IDENTIFIER", "INLINE_FUNC") in tokens
    assert ("IDENTIFIER", "helper") in tokens
    assert all(value != 'R"(' for _kind, value in tokens)
