import pytest

import crosstl
from crosstl.backend.OpenCL.OpenCLCrossGLCodeGen import OpenCLToCrossGLConverter
from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser


def generate_crossgl(source):
    tokens = OpenCLLexer(source).tokenize()
    ast = OpenCLParser(tokens).parse()
    crossgl = OpenCLToCrossGLConverter().generate(ast)
    CrossGLParser(CrossGLLexer(crossgl).tokens).parse()
    return crossgl


def kernel_with(query, dimension):
    return f"""
        kernel void probe(global uint *outp) {{
            uint i = get_global_id(0);
            outp[i] = {query}({dimension});
        }}
        """


# The six work-item queries and the CrossGL builtin each maps to.
WORK_ITEM_QUERIES = [
    ("get_global_id", "gl_GlobalInvocationID"),
    ("get_local_id", "gl_LocalInvocationID"),
    ("get_group_id", "gl_WorkGroupID"),
    ("get_local_size", "gl_WorkGroupSize"),
    ("get_num_groups", "gl_NumWorkGroups"),
]


@pytest.mark.parametrize("query,base", WORK_ITEM_QUERIES)
def test_opencl_work_item_queries_accept_suffixed_dimensions(query, base):
    crossgl = generate_crossgl(kernel_with(query, "1u"))

    assert f"{base}.y" in crossgl
    assert f"{query}(" not in crossgl


@pytest.mark.parametrize("query,base", WORK_ITEM_QUERIES)
def test_opencl_work_item_queries_accept_cast_dimensions(query, base):
    crossgl = generate_crossgl(kernel_with(query, "(uint)1"))

    assert f"{base}.y" in crossgl
    assert f"{query}(" not in crossgl


def test_opencl_global_size_accepts_suffixed_and_cast_dimensions():
    crossgl = generate_crossgl(kernel_with("get_global_size", "1u"))
    assert "(gl_NumWorkGroups.y * gl_WorkGroupSize.y)" in crossgl
    assert "get_global_size(" not in crossgl

    crossgl = generate_crossgl(kernel_with("get_global_size", "(uint)1"))
    assert "(gl_NumWorkGroups.y * gl_WorkGroupSize.y)" in crossgl
    assert "get_global_size(" not in crossgl


@pytest.mark.parametrize(
    "dimension,component",
    [
        ("0u", "x"),
        ("2U", "z"),
        ("1ul", "y"),
        ("1UL", "y"),
        ("(unsigned int)2", "z"),
        ("(int)0", "x"),
        ("(uint)(int)1", "y"),
        ("(size_t)1", "y"),
    ],
)
def test_opencl_work_item_queries_accept_other_constant_spellings(dimension, component):
    crossgl = generate_crossgl(kernel_with("get_global_id", dimension))

    assert f"gl_GlobalInvocationID.{component}" in crossgl
    assert "get_global_id(" not in crossgl


@pytest.mark.parametrize(
    "query,dimension,base,component",
    [
        # A cast to a narrow integer type applies that type's width first:
        # (uchar)256 is 0 and (ushort)65537 is 1 in C.
        ("get_local_size", "(uchar)256", "gl_WorkGroupSize", "x"),
        ("get_group_id", "(ushort)65537", "gl_WorkGroupID", "y"),
        # Hex literals spell a constant dimension too.
        ("get_global_id", "0x1", "gl_GlobalInvocationID", "y"),
        ("get_global_id", "(uint)0x2", "gl_GlobalInvocationID", "z"),
    ],
)
def test_opencl_work_item_queries_apply_cast_width_and_hex_literals(
    query, dimension, base, component
):
    crossgl = generate_crossgl(kernel_with(query, dimension))

    assert f"{base}.{component}" in crossgl
    assert f"{query}(" not in crossgl


def test_opencl_suffixed_dimension_lowers_to_directx(tmp_path):
    source = """
        __kernel __attribute__((reqd_work_group_size(1, 1, 1)))
        void probe(__global const int* iin, __global float* outp) {
            uint i = get_global_id(0);
            outp[i] = (float)(get_global_id(1u));
        }
        """
    opencl_path = tmp_path / "probe.cl"
    opencl_path.write_text(source, encoding="utf-8")
    hlsl = crosstl.translate(str(opencl_path), backend="directx", format_output=False)

    assert "get_global_id" not in hlsl
    assert "dispatchThreadID.y" in hlsl
