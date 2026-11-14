"""HIP input accepts a type that a line break splits after its base type, a ``*`` or a qualifier (#3753).

``const float`` / ``* in`` is the same as ``const float* in``. OpenCL input shares ``HipParser``'s type
reader, so the OpenCL cases in ``tests/test_backend/test_OpenCL/test_codegen.py`` cover the same code.
"""

import pytest

from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser


def generate_crossgl(source):
    ast = HipParser(HipLexer(source).tokenize()).parse()
    return HipToCrossGLConverter().generate(ast)


@pytest.mark.parametrize(
    ("first_line", "second_line"),
    [
        ("const float", "* in"),
        ("float const", "* in"),
        ("const float*", "__restrict__ in"),
    ],
)
def test_hip_kernel_parameter_type_split_after_base_type(first_line, second_line):
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
        (
            "__device__ float widen(unsigned v) { const",
            "float f = (float)v; return f; }",
        ),
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
