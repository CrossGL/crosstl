from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer


def token_pairs(source):
    return [(token.type, token.value) for token in OpenCLLexer(source).tokenize()]


def test_opencl_kernel_and_address_space_tokens_are_normalized():
    tokens = token_pairs(
        "__kernel void k(__global const float *in, local float *scratch) { "
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

