"""Concrete namespace-scope Metal free operators bind their expressions to the source definition."""

import re

import pytest

import crosstl
from crosstl.backend.Metal.concrete_free_operators import (
    materialize_concrete_free_operators,
)
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

# `operator+` deliberately multiplies the imaginary parts: a translator that
# falls back to a built-in addition cannot produce `a.imag * b.imag`.
SOURCE = """#include <metal_stdlib>
using namespace metal;

struct cplx {
  float real;
  float imag;
};

constexpr cplx operator+(cplx a, cplx b) {
  return cplx{a.real + b.real, a.imag * b.imag};
}

inline cplx operator-(cplx a) {
  return cplx{-a.real, a.imag};
}

inline bool operator>(cplx a, cplx b) {
  return a.real > b.real;
}

inline bool operator<=(cplx a, cplx b) {
  return !operator>(a, b);
}

thread cplx& operator*=(thread cplx& a, cplx b) {
  a.real = a.real * b.real;
  a.imag = a.imag - b.imag;
  return a;
}

kernel void k(device cplx* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  cplx v = out[i];
  cplx w = v + out[i + 1];
  w *= -v;
  out[i] = (v <= w) ? w : v;
}
"""

ADD = "crosstl_metal_operator_add__cplx__cplx"
NEGATE = "crosstl_metal_operator_negate__cplx"
LESS_EQUAL = "crosstl_metal_operator_less_equal__cplx__cplx"
MULTIPLY_ASSIGN = "crosstl_metal_operator_multiply_assign__cplx__cplx"


def _translate(tmp_path, source, backend):
    path = tmp_path / "unit.metal"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_concrete_operators_become_named_helpers():
    output = MetalPreprocessor().preprocess(SOURCE)

    assert f"inline cplx {ADD}(cplx a, cplx b)" in output
    assert f"inline cplx {NEGATE}(cplx a)" in output
    # The direct call names the `>` helper defined before it.
    assert "return !crosstl_metal_operator_greater__cplx__cplx(a, b);" in output
    # The compound operator updates and returns a copy.
    assert f"inline cplx {MULTIPLY_ASSIGN}(cplx a, cplx b)" in output
    assert not re.search(r"\boperator\s*[-+*/<>=!]+\s*\(", output)


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_expressions_call_the_source_operators(tmp_path, backend):
    output = _translate(tmp_path, SOURCE, backend)

    if backend == "opengl":
        assert "crossgl_ieee_product_float(a.imag, b.imag)" in output
    else:
        assert "a.imag * b.imag" in output
    for helper in (ADD, NEGATE, LESS_EQUAL, MULTIPLY_ASSIGN):
        name = helper if backend != "opengl" else helper.replace("__", "_")
        assert re.search(rf"\b{name}\(", output), (backend, helper)


def test_spirv_calls_the_source_operators(tmp_path):
    output = _translate(tmp_path, SOURCE, "vulkan")

    # add, negate, less_equal (which calls greater) and multiply_assign.
    assert output.count("OpFunctionCall") == 5


def test_member_template_friend_and_qualified_operators_are_left_alone():
    source = """struct box {
  float v;
  box operator+(box o) const { return box{v + o.v}; }
  friend box operator-(box a, box b) { return box{a.v - b.v}; }
};
template <typename T> T operator*(T a, T b) { return a; }
box box_operator_plus(box a, box b);
box box::operator/(box o) const { return o; }
constant box origin = {0.0f};
"""
    output = materialize_concrete_free_operators(MetalPreprocessor(), source)

    assert output == source


def test_namespace_operators_keep_their_namespace_and_linkage():
    source = """namespace geom {
struct p { float x; };
p operator+(p a, p b) { return p{a.x + b.x}; }
}
namespace {
geom::p operator-(geom::p a, geom::p b) { return geom::p{a.x - b.x}; }
}
static geom::p operator*(geom::p a, geom::p b) { return geom::p{a.x * b.x}; }
"""
    output = materialize_concrete_free_operators(MetalPreprocessor(), source)

    assert (
        "namespace geom {\nstruct p { float x; };\ninline p crosstl_metal_operator_add__p__p(p a, p b)"
        in output
    )
    assert (
        "static inline geom::p crosstl_metal_operator_subtract__geom_p__geom_p("
        in output
    )
    assert (
        "static inline geom::p crosstl_metal_operator_multiply__geom_p__geom_p("
        in output
    )


def test_rewritten_definitions_keep_body_lines():
    source = "struct s { float v; };\nconstexpr METAL_FUNC\ns operator+(s a,\n            s b)\n{\n  return s{a.v + b.v};\n}\n"
    output = materialize_concrete_free_operators(MetalPreprocessor(), source)

    assert output.count("\n") == source.count("\n")
    assert output.splitlines()[-2:] == source.splitlines()[-2:]


def test_device_reference_compound_operator_is_not_given_a_value_form():
    source = """struct s { float v; };
device s& operator+=(device s& a, s b) {
  a.v = a.v + b.v;
  return a;
}
"""
    output = materialize_concrete_free_operators(MetalPreprocessor(), source)

    assert output == source
