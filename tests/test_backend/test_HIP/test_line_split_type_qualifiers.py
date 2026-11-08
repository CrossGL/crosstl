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


@pytest.mark.parametrize(
    ("first_line", "second_line"),
    [
        ("const float", "* in"),
        ("float const", "* in"),
        ("const float*", "__restrict__ in"),
    ],
)
def test_hip_kernel_parameter_type_split_after_base_type(first_line, second_line):
    # A line break after the base type, a ``*`` or a postfix qualifier is ignored (#3753).
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

    assert "@group(0) @binding(0) var<storage, read> in_: array<f32>" in crossgl
    assert crossgl == generate_crossgl(one_line_source)


@pytest.mark.parametrize(
    ("split_line", "rest"),
    [
        ("__device__ float widen(unsigned", "v) { return (float)v; }"),
        ("__device__ float widen(unsigned v) { const", "float f = (float)v; return f; }"),
        ("__device__ float widen(unsigned v) { return (const", "float)v; }"),
    ],
)
def test_hip_unsigned_local_and_cast_split_across_lines(split_line, rest):
    split_source = f"""
        #include <hip/hip_runtime.h>
        {split_line}
            {rest}

        __global__ void first(float* out) {{
            out[0] = widen(1u);
        }}
        """
    one_line_source = split_source.replace(f"{split_line}\n", f"{split_line} ")

    crossgl = generate_crossgl(split_source)

    assert "f32 widen(u32 v)" in crossgl
    assert crossgl == generate_crossgl(one_line_source)
