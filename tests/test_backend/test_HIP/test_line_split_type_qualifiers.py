"""HIP input accepts a kernel parameter with a line break after a type qualifier (#3754).

``const`` / ``float* in`` is the same as ``const float* in``: the HIP lexer keeps line breaks as
``NEWLINE`` tokens, and the declaration parser used to stop at that break. The handling that
OpenCL input already had for the same shape now lives on ``HipParser``.
"""

import pytest

from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser


def generate_crossgl(source):
    ast = HipParser(HipLexer(source).tokenize()).parse()
    return HipToCrossGLConverter().generate(ast)


@pytest.mark.parametrize(
    ("first_line", "second_line", "expected"),
    [
        (
            "const",
            "float* in",
            "@group(0) @binding(0) var<storage, read> in_: array<f32>",
        ),
        (
            "const volatile",
            "float* in",
            "@group(0) @binding(0) var<storage, read> in_: array<f32>",
        ),
        (
            "const",
            "float* __restrict__ in",
            "@group(0) @binding(0) var<storage, read> in_: array<f32>",
        ),
        (
            "volatile",
            "float* in",
            "@group(0) @binding(0) var<storage, read_write> in_: array<f32>",
        ),
    ],
)
def test_hip_kernel_parameter_qualifiers_split_across_lines(
    first_line, second_line, expected
):
    split_source = f"""
        #include <hip/hip_runtime.h>
        __global__ void first({first_line}
            {second_line}, float* out) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            out[i] = in[i];
        }}
        """
    one_line_source = split_source.replace(f"{first_line}\n", f"{first_line} ")

    crossgl = generate_crossgl(split_source)

    assert expected in crossgl
    assert "@group(0) @binding(1) var<storage, read_write> out: array<f32>" in crossgl
    assert crossgl == generate_crossgl(one_line_source)


def test_hip_function_parameter_qualifier_split_across_lines():
    split_source = """
        #include <hip/hip_runtime.h>
        __device__ float twice(const
            float v) {
            return v * 2.0f;
        }

        __global__ void first(const float* in, float* out) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            out[i] = twice(in[i]);
        }
        """
    one_line_source = split_source.replace("const\n", "const ")

    crossgl = generate_crossgl(split_source)

    assert "f32 twice(f32 v)" in crossgl
    assert crossgl == generate_crossgl(one_line_source)


def test_hip_line_split_qualifiers_at_eof_without_crash():
    # The qualifier scan used to read past the last token when blanks followed the line break (#4371).
    def parse_type(source):
        return HipParser(HipLexer(source).tokenize()).parse_type()

    assert parse_type("const\n ") == parse_type("const\n")
