"""OpenCL C import of work-item functions (OpenCL C 3.0, 6.15.1) beyond a literal dimension 0, 1 or 2 (#4820)."""

import pytest

import crosstl
from crosstl.backend.OpenCL.OpenCLCrossGLCodeGen import OpenCLToCrossGLConverter
from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser
from crosstl.backend.OpenCL.work_items import OpenCLWorkItemUnsupportedError
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

KERNEL = """
__kernel __attribute__((reqd_work_group_size(4, 2, 1)))
void k(__global const int* iin, __global uint* o) {{
    int d = iin[0];
    uint e = (uint)iin[1];
    o[0] = {query};
}}
"""


def to_crossgl(source):
    ast = OpenCLParser(OpenCLLexer(source).tokenize()).parse()
    crossgl = OpenCLToCrossGLConverter().generate(ast)
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()
    return crossgl


def stored(query):
    """The CrossGL value the kernel stores for the OpenCL C expression ``query``."""
    crossgl = to_crossgl(KERNEL.format(query=query))
    values = [
        line.strip()[len("o[0] = ") : -1]
        for line in crossgl.splitlines()
        if line.strip().startswith("o[0] = ")
    ]
    assert len(values) == 1, crossgl
    return values[0]


@pytest.mark.parametrize(
    "query, value",
    [
        # 6.15.1: a dimension outside 0 to get_work_dim() - 1 reads 0 for an ID and 1 for a size.
        ("get_global_id(d)", "(u32(d) < 3u ? gl_GlobalInvocationID[u32(d)] : 0u)"),
        ("get_local_id(e)", "(u32(e) < 3u ? gl_LocalInvocationID[u32(e)] : 0u)"),
        (
            "get_group_id(e + 1)",
            "(u32((e + 1)) < 3u ? gl_WorkGroupID[u32((e + 1))] : 0u)",
        ),
        ("get_local_size(d)", "(u32(d) < 3u ? gl_WorkGroupSize[u32(d)] : 1u)"),
        ("get_num_groups(e)", "(u32(e) < 3u ? gl_NumWorkGroups[u32(e)] : 1u)"),
        (
            "get_global_size(d)",
            "(u32(d) < 3u ? (gl_NumWorkGroups[u32(d)] * gl_WorkGroupSize[u32(d)]) : 1u)",
        ),
        # A cast to uint is the conversion the dimension parameter makes; it is not repeated.
        (
            "get_global_id((uint)iin[0])",
            "(u32(iin[0]) < 3u ? gl_GlobalInvocationID[u32(iin[0])] : 0u)",
        ),
    ],
)
def test_a_runtime_dimension_reads_the_component_or_the_out_of_range_value(
    query, value
):
    assert stored(query) == value


@pytest.mark.parametrize(
    "query, value",
    [
        ("get_global_id(3)", "0u"),
        ("get_local_id(7)", "0u"),
        ("get_group_id(-1)", "0u"),
        ("get_local_size(3u)", "1u"),
        ("get_num_groups(0x10)", "1u"),
        ("get_global_size((uint)5)", "1u"),
        ("get_enqueued_local_size(4)", "1u"),
    ],
)
def test_a_constant_dimension_of_three_or_more_folds(query, value):
    assert stored(query) == value


@pytest.mark.parametrize(
    "query, value",
    [
        ("get_enqueued_local_size(0)", "gl_WorkGroupSize.x"),
        ("get_enqueued_local_size(2)", "gl_WorkGroupSize.z"),
        ("get_enqueued_local_size(e)", "(u32(e) < 3u ? gl_WorkGroupSize[u32(e)] : 1u)"),
    ],
)
def test_the_enqueued_local_size_is_the_local_size_of_uniform_work_groups(query, value):
    assert stored(query) == value


def test_the_global_offset_is_zero_like_the_global_id_mapping_assumes():
    assert stored("get_global_offset(0)") == "0u"
    assert stored("get_global_offset(d)") == "0u"


def test_the_global_linear_id_follows_the_definition_with_a_zero_offset():
    size_x = "(gl_NumWorkGroups.x * gl_WorkGroupSize.x)"
    size_y = "(gl_NumWorkGroups.y * gl_WorkGroupSize.y)"
    assert stored("get_global_linear_id()") == (
        f"(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * {size_x}"
        f" + gl_GlobalInvocationID.z * {size_x} * {size_y})"
    )


def test_get_work_dim_is_refused_with_an_opencl_diagnostic():
    with pytest.raises(OpenCLWorkItemUnsupportedError, match="get_work_dim"):
        stored("get_work_dim()")


@pytest.mark.parametrize("query", ["get_global_id(d++)", "get_global_offset(e--)"])
def test_a_dimension_with_a_side_effect_is_refused(query):
    with pytest.raises(OpenCLWorkItemUnsupportedError, match="side effect"):
        stored(query)


def test_a_program_function_with_a_work_item_name_keeps_its_call():
    crossgl = to_crossgl(
        "uint get_work_dim() { return 2u; }\n"
        "__kernel void k(__global uint* o) { o[0] = get_work_dim(); }"
    )
    assert "o[0] = get_work_dim();" in crossgl


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal", "vulkan", "wgsl"])
def test_runtime_dimensions_translate_to_every_compute_target(tmp_path, backend):
    source = tmp_path / "probe.cl"
    source.write_text(
        KERNEL.format(
            query="get_global_id(d) + get_local_size(e) + get_global_size(d)"
            " + get_global_linear_id() + get_global_offset(e) + get_enqueued_local_size(d)"
        ),
        encoding="utf-8",
    )
    translated = crosstl.translate(str(source), backend=backend, format_output=False)
    assert "get_global_id" not in translated
    assert "get_global_linear_id" not in translated
