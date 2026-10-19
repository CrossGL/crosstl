"""Metal frontend support that MLX's Winograd convolution transforms rely on."""

import re

import pytest

import crosstl
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalStaticConstantResolutionError,
    MetalToCrossGLConverter,
)
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.preprocessor import MetalStructMethodError

MAPPED_MATRIX = (
    "CooperativeMatrix<float,8,8,subgroup,unspecified,unspecified,"
    "metal_thread_elements,32,2,metal_thread_elements_reference_view,"
    "tile_4x4_row_pair,winograd_lane_coordinates>"
)
UNSPECIFIED_MATRIX = "CooperativeMatrix<float,8,8,subgroup,unspecified,unspecified>"

# conv.metal's shape: a struct method in the same translation unit, a table in
# an explicit specialization with its C++17 out-of-class redeclaration, and
# simdgroup_matrix lanes written through indexed thread_elements().
SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <int M>
struct Transforms {};

template <>
struct Transforms<2> {
  static constant constexpr const float table[2][2] = {{1.0f, -2.0 / 9.0}, {0.25f}};
};

constant constexpr const float Transforms<2>::table[2][2];

struct Scale {
  float factor;
  float apply(float value) const {
    return value * factor;
  }
};

kernel void indexed_fragments(
    device float* out [[buffer(0)]],
    uint lane [[thread_index_in_simdgroup]]) {
  Scale scale{2.0f};
  simdgroup_matrix<float, 8, 8> left;
  left.thread_elements()[0] = Transforms<2>::table[lane % 2][0];
  left.thread_elements()[1] = scale.apply(Transforms<2>::table[1][lane % 2]);
  simdgroup_matrix<float, 8, 8> product = left * left;
  out[lane * 2] = product.thread_elements()[0];
  out[lane * 2 + 1] = product.thread_elements()[1];
}
"""


def _crossgl(source, **options):
    ast = MetalParser(MetalLexer(source).tokenize()).parse()
    return MetalToCrossGLConverter(**options).generate(ast)


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def test_out_of_class_redeclaration_keeps_the_in_class_table():
    crossgl = _crossgl(SOURCE)
    assert (
        "constant float[2][2] Transforms_u3c2_u3e_u3a_u3atable = "
        "{{1.0f, (-2.0) / 9.0}, {0.25f}};"
    ) in crossgl
    assert "Transforms_u3c2_u3e_u3a_u3atable[lane % 2][0]" in crossgl


TABLE_SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <int M>
struct Transforms {};

template <>
struct Transforms<2> {
  static constant constexpr const float table[2][2] = {{1.0f, -2.0 / 9.0}, {0.25f}};
};

constant constexpr const float Transforms<2>::table[2][2];

kernel void read_table(device float* out [[buffer(0)]],
                       uint lane [[thread_position_in_grid]]) {
  out[lane] = Transforms<2>::table[lane % 2][lane / 2 % 2];
}
"""


def test_out_of_class_table_stays_a_constant_on_opengl(tmp_path):
    generated = " ".join(_translate(tmp_path, TABLE_SOURCE, "opengl").split())
    # The rows keep their declared extent: missing elements are zero.
    assert (
        "const float Transforms_u3c2_u3e_u3a_u3atable[2][2] = "
        "float[2][2](float[2](1.0, ((-2.0) / 9.0)), float[2](0.25, 0.0));"
    ) in generated
    assert "uniform float Transforms" not in generated


@pytest.mark.parametrize(
    "initializer",
    ["{float(SIZE), 0.5f}", "{SIZE * 1.0f, 0.5f}"],
)
def test_out_of_class_redeclaration_without_literal_initializer_fails_closed(
    initializer,
):
    source = f"""
#include <metal_stdlib>
using namespace metal;

struct Table {{
  static constant constexpr const int SIZE = 2;
  static constant constexpr const float values[2] = {initializer};
}};

constant constexpr const float Table::values[2];

kernel void read_table(device float* out [[buffer(0)]],
                       uint lane [[thread_position_in_grid]]) {{
  out[lane] = Table::values[lane % 2];
}}
"""
    with pytest.raises(MetalStaticConstantResolutionError, match="literal"):
        _crossgl(source)


def test_library_template_receivers_bypass_struct_method_lowering():
    crossgl = _crossgl(SOURCE)
    applied = "Scale__apply(scale, Transforms_u3c2_u3e_u3a_u3atable[1][lane % 2])"
    product = f"{UNSPECIFIED_MATRIX} product = cooperative_matrix_multiply(left, left);"
    assert applied in crossgl
    assert "cooperative_matrix_element(left, 0) = " in crossgl
    assert product in crossgl


def test_texture_receivers_bypass_struct_method_lowering():
    source = """
#include <metal_stdlib>
using namespace metal;

struct Scale {
  float factor;
  float apply(float value) const {
    return value * factor;
  }
};

kernel void read_texture(texture2d<float, access::read> image [[texture(0)]],
                         device float* out [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
  Scale scale{2.0f};
  out[gid.y * 8 + gid.x] = scale.apply(image.read(gid).x);
}
"""
    crossgl = _crossgl(source)
    assert "Scale__apply(scale, " in crossgl
    assert "image" in crossgl.split("Scale__apply(scale, ", 1)[1].split(";", 1)[0]


def test_plain_receiver_shadowing_a_library_template_still_fails_closed():
    source = """
#include <metal_stdlib>
using namespace metal;

struct Scale {
  float factor;
  float apply(float value) const {
    return value * factor;
  }
};

struct Opaque;

kernel void shadowed(device float* out [[buffer(0)]],
                     uint lane [[thread_index_in_simdgroup]]) {
  Scale scale{2.0f};
  simdgroup_matrix<float, 8, 8> left;
  {
    thread Opaque& left = *reinterpret_cast<thread Opaque*>(out);
    out[lane] = scale.apply(left.thread_elements()[0]);
  }
}
"""
    with pytest.raises(MetalStructMethodError):
        _crossgl(source)


def test_indexed_thread_elements_take_the_configured_fragment_contract():
    crossgl = _crossgl(
        SOURCE,
        cooperative_matrix_fragment_mapping="tile_4x4_row_pair",
        cooperative_matrix_fragment_mapping_provenance="winograd_lane_coordinates",
    )
    emitted = set(re.findall(r"CooperativeMatrix<float,8,8,[^>]+>", crossgl))
    assert emitted == {MAPPED_MATRIX}
    product = f"{MAPPED_MATRIX} product = cooperative_matrix_multiply(left, left);"
    assert product in crossgl


def test_indexed_thread_elements_without_a_mapping_stay_unspecified():
    crossgl = _crossgl(SOURCE)
    emitted = set(re.findall(r"CooperativeMatrix<float,8,8,[^>]+>", crossgl))
    assert emitted == {UNSPECIFIED_MATRIX}
    assert "metal_thread_elements" not in crossgl
