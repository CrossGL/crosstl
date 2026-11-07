"""Converting constructors in copy-initialization outside declarations.

A non-explicit Metal constructor callable with one argument converts implicitly ([class.conv.ctor]) wherever a value
copy-initializes an object of the class: an assignment's right operand, a stored buffer element and a value or
``const`` reference argument. CrossGL structs have no constructors, so the importer spells the conversion through
the constructor factory, as it already does for ``T t = v;``.
"""

import pytest

import crosstl
from crosstl.backend.Metal.implicit_constructor_conversions import (
    MetalAssignmentOperatorError,
)
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalConstructorContractError

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

COMPLEX = """
struct cpx {
  float real;
  float imag;
  cpx(float2 v) thread : real(v.x), imag(v.y) {}
  cpx(float r, float i) thread : real(r), imag(i) {}
};
"""

ASSIGNMENTS = COMPLEX + """
kernel void k(device cpx* out [[buffer(0)]], device float2* io [[buffer(1)]],
              constant bool& inv [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  float2 value = io[i];
  cpx c = value;
  thread cpx* p = &c;
  *p = value;
  c = value;
  out[i] = inv ? float2(value.x, -value.y) : value;
  out[i + 1] = c;
}
"""

ARGUMENTS = COMPLEX + """
cpx take(cpx c) { return c; }
float mag(const thread cpx& c) { return c.real * c.real + c.imag * c.imag; }
struct Holder {
  float scale;
  float apply(cpx c) const thread { return c.real * scale; }
  float run(float2 v) const thread { return apply(v); }
};

kernel void k(device float2* io [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  float2 value = io[i];
  cpx b = take(value);
  float m = mag(value);
  Holder h{2.0f};
  io[i] = float2(b.real + m, h.run(value));
}
"""


def _translate(tmp_path, source, backend="cgl"):
    path = tmp_path / "kernel.metal"
    path.write_text(PREFIX + source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def test_assignments_and_stores_convert_through_the_constructor(tmp_path):
    cgl = _translate(tmp_path, ASSIGNMENTS)
    assert "cpx c = crosstl_ctor_cpx_1(value);" in cgl
    assert "(*p) = crosstl_ctor_cpx_1(value);" in cgl
    assert "c = crosstl_ctor_cpx_1(value);" in cgl
    assert (
        "buffer_store(out_, i, crosstl_ctor_cpx_1(inv ? vec2(value.x, (-value.y)) : value));"
        in cgl
    )
    # A value of the class type is copied, not converted.
    assert "buffer_store(out_, i + 1, c);" in cgl


def test_arguments_convert_through_the_constructor(tmp_path):
    cgl = _translate(tmp_path, ARGUMENTS)
    assert "cpx b = take(crosstl_ctor_cpx_1(value));" in cgl
    assert "float m = mag(crosstl_ctor_cpx_1(value));" in cgl
    assert "crosstl_ctor_cpx_1(v)" in cgl


def test_converted_store_reaches_directx_half_storage(tmp_path):
    source = """
struct hcpx {
  half real;
  half imag;
  hcpx(half2 v) thread : real(v.x), imag(v.y) {}
};
kernel void k(device hcpx* out [[buffer(0)]], constant bool& inv [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
  half2 value = half2(1.0h, 2.0h);
  out[i] = inv ? half2(value.x, -value.y) : value;
}
"""
    hlsl = _translate(tmp_path, source, "directx")
    assert (
        "__crossgl_half_store_hcpx(crosstl_ctor_hcpx_1((k_inv ? "
        "float16_t2(value.x, -value.y) : value)))" in hlsl
    )


def test_user_assignment_operator_is_refused(tmp_path):
    source = """
struct cpx {
  float real;
  float imag;
  cpx(float2 v) thread : real(v.x), imag(v.y) {}
  thread cpx& operator=(float2 v) thread { real = v.y; imag = v.x; return *this; }
};
kernel void k(device float2* io [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  cpx c = io[i];
  c = io[i + 1];
  io[i] = float2(c.real, c.imag);
}
"""
    with pytest.raises(MetalAssignmentOperatorError) as raised:
        _translate(tmp_path, source)
    assert raised.value.project_diagnostic_code == (
        "project.translate.metal-assignment-operator-unrepresentable"
    )
    assert "cpx" in str(raised.value)


def test_explicit_constructor_does_not_convert(tmp_path):
    source = """
struct cpx {
  float real;
  float imag;
  explicit cpx(float2 v) thread : real(v.x), imag(v.y) {}
};
kernel void k(device cpx* out [[buffer(0)]], device float2* io [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
  out[i] = io[i];
}
"""
    with pytest.raises(MetalConstructorContractError, match="non-explicit"):
        _translate(tmp_path, source)


def test_parser_records_member_assignment_operators():
    from crosstl.backend.Metal.MetalLexer import MetalLexer
    from crosstl.backend.Metal.MetalParser import MetalParser

    source = """
struct a { float x; thread a& operator=(float2 v) thread { return *this; } };
struct b { float x; template <typename U> thread b& operator=(U v) thread { return *this; } };
struct c { float x; bool operator==(c o) const { return true; } };
"""
    ast = MetalParser(MetalLexer(source).tokenize()).parse()
    flags = {node.name: node.declares_assignment_operator for node in ast.structs}
    assert flags == {"a": True, "b": True, "c": False}
