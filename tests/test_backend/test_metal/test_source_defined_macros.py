"""Macros and alias templates that a Metal library declares are read from the source, not from built-in tables.

The importer knows one library macro by name: ``METAL_FUNC``, which the Metal standard library headers define and
the importer does not read. Any other qualifier or alignment macro is only meaningful through its ``#define``: a
defined one translates like its expansion, and an undefined one is a syntax error, as it is for the Metal compiler.
"""

import pytest

from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser, MetalParserError
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

KERNEL = """
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    o[i] = float(count) * scale(o[i]);
}
"""


def translate(source):
    preprocessed = MetalPreprocessor().preprocess(source)
    ast = MetalParser(MetalLexer(preprocessed).tokenize()).parse()
    return MetalToCrossGLConverter().generate(ast)


def test_defined_qualifier_macros_translate_like_their_expansion():
    expanded = translate(
        "#include <metal_stdlib>\n"
        "static constant constexpr const int count = 4;\n"
        "inline float scale(float v) { return v * 2.0; }\n" + KERNEL
    )
    defined = translate(
        "#include <metal_stdlib>\n"
        "#define LIB_CONST static constant constexpr const\n"
        "#define LIB_FUNC inline\n"
        "LIB_CONST int count = 4;\n"
        "LIB_FUNC float scale(float v) { return v * 2.0; }\n" + KERNEL
    )
    assert defined == expanded
    assert "LIB_" not in defined


@pytest.mark.parametrize(
    "declaration",
    [
        "LIB_CONST int count = 4;\ninline float scale(float v) { return v; }\n",
        "constant int count = 4;\nLIB_FUNC float scale(float v) { return v; }\n",
    ],
)
def test_undefined_qualifier_macros_are_refused(declaration):
    with pytest.raises(MetalParserError):
        translate("#include <metal_stdlib>\n" + declaration + KERNEL)


def test_metal_standard_library_function_macro_is_known():
    crossgl = translate(
        "#include <metal_stdlib>\n"
        "constant int count = 4;\n"
        "METAL_FUNC float scale(float v) { return v * 2.0; }\n" + KERNEL
    )
    assert "float scale(float v)" in crossgl


def test_defined_alignment_macro_keeps_the_struct_alignment():
    crossgl = translate(
        "#include <metal_stdlib>\n"
        "#define LIB_ALIGN_AS(n) alignas(n)\n"
        "#define LIB_ALIGN LIB_ALIGN_AS(16)\n"
        "struct LIB_ALIGN Block { float x; };\n"
    )
    assert "@metal_alignas(16)\n    struct Block {" in crossgl


def test_undefined_alignment_macro_is_refused():
    with pytest.raises(MetalParserError):
        translate("#include <metal_stdlib>\nstruct LIB_ALIGN Block { float x; };\n")


NESTED_METAL_ALIAS = """
#include <metal_stdlib>
namespace lib {
namespace metal {
template <typename T, unsigned N>
using array = ::metal::array<T, N>;
} // namespace metal
} // namespace lib
"""


def test_globally_qualified_alias_target_names_the_global_namespace():
    # Inside lib::metal, ``metal::array`` would name lib::metal::array itself; ``::metal::array`` is the Metal one.
    crossgl = translate(
        NESTED_METAL_ALIAS
        + "struct Shape { lib::metal::array<uint, 3> sizes; };\n"
        + "kernel void k(constant Shape& s [[buffer(0)]], device float* o [[buffer(1)]],\n"
        + "              uint i [[thread_position_in_grid]]) {\n"
        + "    lib::metal::array<float, 4> v = {1.0, 2.0, 3.0, 4.0};\n"
        + "    o[i] = v[i % 4] * float(s.sizes[1]);\n"
        + "}\n"
    )
    assert "uint[3] sizes;" in crossgl
    assert "float[4] v = float[4](1.0, 2.0, 3.0, 4.0);" in crossgl
    assert "lib::metal::array" not in crossgl


@pytest.mark.parametrize(
    "use, expected",
    [
        ("pair_t<lib::idx> v = {1, 2};", "int[2] v = int[2](1, 2);"),
        ("pair_t<metal::int2> v;", "ivec2[2] v;"),
        ("arr_t<float, lib::n> v = {1.0, 2.0, 3.0};", "float["),
        ("arr_t<float, ::lib::n> v = {1.0, 2.0, 3.0};", "float["),
    ],
)
def test_namespace_qualified_alias_template_arguments_are_not_dependent_members(
    use, expected
):
    crossgl = translate(
        "#include <metal_stdlib>\n"
        "namespace lib { using idx = int; constant constexpr unsigned n = 3; }\n"
        "template <typename T> using pair_t = metal::array<T, 2>;\n"
        "template <typename T, unsigned N> using arr_t = metal::array<T, N>;\n"
        "kernel void k(device int* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n"
        f"    {use}\n"
        "    o[i] = int(v[1]);\n"
        "}\n"
    )
    assert expected in crossgl
