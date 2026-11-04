"""The cooperative-matrix diagonal: Metal's one-value ``simdgroup_matrix``.

``simdgroup_matrix<T, Cols, Rows>(T value)`` builds a matrix with ``value`` on
its diagonal and zero elsewhere (Metal Shading Language specification,
"SIMD-group matrix functions"); ``make_filled_simdgroup_matrix`` is the fill.
CrossGL spells it ``cooperative_matrix_diagonal(value)`` or, like a GLSL
``matN(s)`` constructor, ``CooperativeMatrix<...>(value)``.
"""

import pytest

from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.translator import parse
from crosstl.translator.ast import CooperativeMatrixOpNode
from crosstl.translator.codegen.ast_walk import walk_ast_values
from crosstl.translator.codegen.cooperative_matrix_diagonal import (
    registered_mapping_closed_form,
)
from crosstl.translator.codegen.directx_codegen import (
    DirectXCooperativeMatrixUnsupportedError,
    HLSLCodeGen,
)
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLCooperativeMatrixError,
)
from crosstl.translator.codegen.metal_codegen import (
    MetalCodeGen,
    UnsupportedMetalFeatureError,
)
from crosstl.translator.codegen.SPIRV_codegen import (
    UnsupportedSPIRVFeatureError,
    VulkanSPIRVCodeGen,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from tests.test_translator.test_matrix_transfers import MATRIX, _dx, _dx_compile, _gl
from tests.test_translator.test_metal_builtin_ownership import _compile

PLAIN = "CooperativeMatrix<float, 8, 8, subgroup, unspecified, unspecified>"

SHADER = f"""shader Diagonals {{
    compute {{
        @numthreads(32, 4, 1)
        void main(StructuredBuffer<float> src @register(t0),
                  RWStructuredBuffer<float> dst @register(u1),
                  uint invocation @gl_LocalInvocationIndex) {{
            int counter = 0;
            {{MATRIX}} A = cooperative_matrix_diagonal(float(counter++));
            {{MATRIX}} B;
            B = {{MATRIX}}(src[0]);
            cooperative_matrix_store(dst, A, 8);
            cooperative_matrix_store(dst + 64, B, 8);
        }}
    }}
}}"""


def _shader(matrix=MATRIX):
    return SHADER.replace("{MATRIX}", matrix)


def _tokens(code):
    return Lexer(code).get_tokens()


def _diagonals(tree):
    return [
        node
        for node in walk_ast_values(tree)
        if isinstance(node, CooperativeMatrixOpNode) and node.operation == "diagonal"
    ]


def test_parser_types_both_spellings_from_context_and_constructor():
    diagonals = _diagonals(parse(_shader(PLAIN)))
    assert len(diagonals) == 2
    for node in diagonals:
        assert node.result_type is not None
        assert (node.result_type.rows, node.result_type.cols) == (8, 8)


@pytest.mark.parametrize(
    "replacement, message",
    [
        ("cooperative_matrix_diagonal(float(counter++), 1.0)", "1 argument, got 2"),
        (f"{PLAIN}(src[0], 1.0)", "1 diagonal value, got 2"),
        (f"{PLAIN}()", "1 diagonal value, got 0"),
    ],
)
def test_parser_rejects_other_arities(replacement, message):
    source = _shader(PLAIN)
    if replacement.startswith("cooperative"):
        source = source.replace(
            "cooperative_matrix_diagonal(float(counter++))", replacement
        )
    else:
        source = source.replace(f"{PLAIN}(src[0])", replacement)
    with pytest.raises(SyntaxError, match=message):
        Parser(_tokens(source), strict_function_bodies=True).parse()


def test_metal_emits_the_one_value_constructor(tmp_path):
    metal = MetalCodeGen().generate(parse(_shader(PLAIN)))
    assert "simdgroup_matrix<float, 8, 8>(float((counter++)))" in metal or (
        "simdgroup_matrix<float, 8, 8>(float(counter++))" in metal
    )
    assert "B = simdgroup_matrix<float, 8, 8>(" in metal
    assert "CooperativeMatrix" not in metal
    assert "cooperative_matrix_diagonal" not in metal
    _compile(metal, "metal", tmp_path)


def test_metal_rejects_a_diagonal_without_a_matrix_context():
    source = _shader(PLAIN).replace(
        "cooperative_matrix_store(dst, A, 8);",
        "cooperative_matrix_diagonal(1.0);\n            cooperative_matrix_store(dst, A, 8);",
    )
    with pytest.raises(UnsupportedMetalFeatureError, match="diagonal requires"):
        MetalCodeGen().generate(parse(source))


METAL_SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void diagonals(device float* out [[buffer(0)]],
                      const device float* scale [[buffer(1)]],
                      uint lane [[thread_index_in_simdgroup]]) {
  simdgroup_float8x8 a(scale[0]);
  simdgroup_matrix<float, 8, 8> b = simdgroup_matrix<float, 8, 8>(2.0f);
  simdgroup_float8x8 c = make_filled_simdgroup_matrix<float, 8, 8>(1.0f);
  c = simdgroup_float8x8(scale[1]);
  simdgroup_multiply_accumulate(c, a, b, c);
  simdgroup_store(c, out, 8);
}
"""


def _crossgl(source=METAL_SOURCE):
    ast = MetalParser(MetalLexer(source).tokenize()).parse()
    return MetalToCrossGLConverter().generate(ast)


def test_metal_frontend_imports_the_constructor_as_a_diagonal():
    crossgl = _crossgl()
    assert "cooperative_matrix_diagonal(float(2.0f))" in crossgl
    assert "c = cooperative_matrix_diagonal(float(buffer_load(scale, 1)));" in crossgl
    assert "cooperative_matrix_fill(float(1.0f))" in crossgl
    # The mangled library spelling of the constructor is never a call.
    assert "simdgroup_matrix_" not in crossgl


def test_metal_round_trip_keeps_diagonals_and_fills_apart(tmp_path):
    metal = MetalCodeGen().generate(parse(_crossgl()))
    assert "simdgroup_matrix<float, 8, 8> a = simdgroup_matrix<float, 8, 8>(" in metal
    assert "simdgroup_matrix<float, 8, 8>(float(2.0))" in metal
    assert "make_filled_simdgroup_matrix<float, 8, 8>(float(1.0))" in metal
    assert "c = simdgroup_matrix<float, 8, 8>(float(scale[1]));" in metal
    assert "CooperativeMatrix" not in metal
    _compile(metal, "metal", tmp_path, metal_compile_flags=("-fno-fast-math",))


def test_registered_mapping_closed_form_requires_the_verified_table():
    coordinate = GLSLCodeGen.glsl_cooperative_matrix_transfer_coordinate
    closed = registered_mapping_closed_form(
        "tile_4x4_row_pair", (8, 8), (32, 2), coordinate
    )
    assert (
        closed == GLSLCodeGen.GLSL_COOPERATIVE_MATRIX_TRANSFER_MAPPINGS[
            "tile_4x4_row_pair"
        ]
    )
    assert (
        registered_mapping_closed_form("unknown", (8, 8), (32, 2), coordinate) is None
    )
    assert (
        registered_mapping_closed_form("tile_4x4_row_pair", (8, 8), (32, 2), None)
        is None
    )
    swapped = lambda lane, element: coordinate(lane, element)[::-1]  # noqa: E731
    assert (
        registered_mapping_closed_form("tile_4x4_row_pair", (8, 8), (32, 2), swapped)
        is None
    )


def test_the_mapping_places_each_diagonal_element_once():
    # The lowering writes ``value`` where row == column: that must be the
    # eight diagonal elements, each held by exactly one lane element.
    coordinate = GLSLCodeGen.glsl_cooperative_matrix_transfer_coordinate
    on_diagonal = [
        coordinate(lane, element)
        for lane in range(32)
        for element in range(2)
        if coordinate(lane, element)[0] == coordinate(lane, element)[1]
    ]
    assert sorted(on_diagonal) == [(index, index) for index in range(8)]


def test_opengl_lowers_a_diagonal_per_lane_element(tmp_path):
    generated = _gl().generate_stage(parse(_shader()), "compute")
    helper = (
        "crossgl_cooperative_matrix_fragment_float_8x8_32x2_tile_4x4_row_pair_diagonal"
    )
    assert f"{helper}(float value) {{" in generated
    assert "result.elements[element] = row == column ? value : float(0);" in generated
    assert "uint lane = gl_LocalInvocationIndex % 32u;" in generated
    assert f"{helper}(float(float((counter++))))" in generated
    assert generated.count("counter++") == 1
    assert f"B = {helper}(float(src[0]));" in generated
    assert "cooperative_matrix_diagonal" not in generated
    _compile(generated, "opengl", tmp_path)


def test_directx_lowers_a_diagonal_per_lane_element(tmp_path):
    generated = _dx().generate_stage(parse(_shader()), "compute")
    assert (
        "float2 __crossgl_software_matrix_diagonal_float2(float value, uint lane) {"
        in (generated)
    )
    assert "result[element] = row == column ? value : (float)0;" in generated
    assert generated.count("counter++") == 1
    assert (
        "B = __crossgl_software_matrix_diagonal_float2((float)(src[0]), "
        "uint(__crossgl_software_subgroup_invocation) % 32u);"
    ) in generated
    assert "cooperative_matrix_diagonal" not in generated
    _dx_compile(generated, tmp_path)


def test_software_lowerings_refuse_an_unregistered_fragment_mapping():
    source = SHADER.replace(
        "{MATRIX}", MATRIX.replace("tile_4x4_row_pair", "column_pair_tiles")
    )
    with pytest.raises(OpenGLCooperativeMatrixError) as gl_error:
        _gl().generate_stage(parse(source), "compute")
    assert gl_error.value.reason == "unregistered-fragment-mapping"
    with pytest.raises(DirectXCooperativeMatrixUnsupportedError) as dx_error:
        _dx().generate_stage(parse(source), "compute")
    assert dx_error.value.reason == "unregistered-fragment-mapping"


def test_software_lowerings_need_32_lane_subgroups():
    with pytest.raises(OpenGLCooperativeMatrixError) as gl_error:
        GLSLCodeGen(cooperative_matrix_software_lowering=True).generate_stage(
            parse(_shader()), "compute"
        )
    assert gl_error.value.reason == "exact-subgroup-width-required"
    assert "diagonal" in str(gl_error.value)
    with pytest.raises(DirectXCooperativeMatrixUnsupportedError) as dx_error:
        HLSLCodeGen(cooperative_matrix_software_lowering=True).generate_stage(
            parse(_shader()), "compute"
        )
    assert dx_error.value.reason == "exact-subgroup-width-required"
    assert "diagonal" in str(dx_error.value)


def test_spirv_refuses_a_diagonal_with_a_typed_error():
    with pytest.raises(UnsupportedSPIRVFeatureError):
        VulkanSPIRVCodeGen().generate(parse(_shader(PLAIN)))
