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


# Two scale formats, each constructible from float and converting back to it.
SCALES = """
struct e8 {
  e8(float x) thread { bits = uint8_t(x); }
  operator float() thread { return float(bits); }
  uint8_t bits;
};
struct e4 {
  e4(float x) thread { bits = uint8_t(x * 2.0f); }
  operator float() thread { return float(bits) * 0.5f; }
  uint8_t bits;
};
"""


def test_conditional_alias_value_of_the_class_type_is_not_converted(tmp_path):
    # ``S`` selects ``e8`` or ``e4`` through a local constexpr condition, so a loaded ``S``
    # already has the type ``operator float`` takes as its object; no constructor applies.
    source = SCALES + """
template <int group_size>
[[kernel]] void k(const device uint8_t* scales [[buffer(0)]],
                  device float* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  constexpr bool use_e8 = group_size == 32;
  using S = metal::conditional_t<use_e8, e8, e4>;
  auto q = ((device S*)(scales))[i];
  out[i] = float(q);
}
template [[host_name("k32")]] [[kernel]] void k<32>(
    const device uint8_t*, device float*, uint);
template [[host_name("k16")]] [[kernel]] void k<16>(
    const device uint8_t*, device float*, uint);
"""
    cgl = _translate(tmp_path, source)
    assert "e8__operator_float(q)" in cgl
    assert "e4__operator_float(q)" in cgl
    assert "crosstl_ctor_e8_1(q)" not in cgl
    assert "crosstl_ctor_e4_1(q)" not in cgl


SELECTED_LOADERS = """
template <typename T, int N>
struct SmallLoader {
  const device T* src;
  int n;
  SmallLoader(const device T* src_, int n_) : src(src_), n(n_ * N) {}
  T load() const thread { return src[n]; }
};
template <typename T, int N>
struct LargeLoader {
  const device T* src;
  LargeLoader(const device T* src_, int n_) : src(src_ + n_) {}
  T load() const thread { return src[0]; }
};
template <typename T, int N>
T take(SmallLoader<T, N> loader) {
  return loader.load();
}
template <typename T, int N>
[[kernel]] void copy_k(const device T* in [[buffer(0)]], device T* out [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {
  using loader_t = typename metal::conditional_t<(N <= 4), SmallLoader<T, N>, LargeLoader<T, N>>;
  loader_t loader(in, int(tid));
  out[tid] = {use};
}
template [[host_name("copy_float_2")]] [[kernel]] void copy_k<float, 2>(
    const device float*, device float*, uint);
"""


@pytest.mark.parametrize("backend", ("cgl", "directx", "vulkan"))
def test_a_conditional_alias_receiver_is_not_converted(tmp_path, backend):
    # `loader` has the type `conditional_t` selects, SmallLoader<float, 2>; its
    # method call passes it as the implicit object argument, which no
    # converting constructor initializes ([over.match.funcs]/5).
    output = _translate(
        tmp_path, SELECTED_LOADERS.replace("{use}", "loader.load()"), backend
    )
    if backend == "cgl":
        assert "SmallLoader_float_2__load(loader, in_)" in output
        assert "crosstl_ctor_SmallLoader_float_2_1(loader" not in output


@pytest.mark.parametrize("backend", ("directx", "vulkan"))
def test_a_conditional_alias_argument_has_the_selected_type(tmp_path, backend):
    # A value argument of the selected class type is copied, not converted.
    output = _translate(
        tmp_path, SELECTED_LOADERS.replace("{use}", "take(loader)"), backend
    )
    if backend == "directx":
        assert "take_float_2(loader," in output


SELECTED_SCALE_LOAD = """
struct Wide {
  uint8_t bits;
  Wide(float v) thread { bits = uint8_t(v); }
  operator float() thread { return float(bits); }
};
struct Narrow {
  uint8_t bits;
  Narrow(float v) thread { bits = uint8_t(v * 2.0f); }
  operator float() thread { return float(bits) * 0.5f; }
};
template <typename T, int group_size>
[[kernel]] void decode(const device uint8_t* scales [[buffer(0)]], device T* out [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {
  constexpr bool wide = group_size == 32;
  using Scale = metal::conditional_t<wide, Wide, Narrow>;
  auto q = ((const device Scale*)(scales))[tid / group_size];
  out[tid] = T(float(q));
}
template [[host_name("decode_float_32")]] [[kernel]] void decode<float, 32>(
    const device uint8_t*, device float*, uint);
"""


@pytest.mark.parametrize("backend", ("cgl", "directx"))
def test_a_conditional_alias_conversion_operator_receiver_is_not_converted(
    tmp_path, backend
):
    # `float(q)` runs Wide::operator float on `q`, whose type is the selected
    # Wide; the receiver is passed as it is.
    output = _translate(tmp_path, SELECTED_SCALE_LOAD, backend)
    if backend == "cgl":
        assert "Wide__operator_float(q)" in output
        assert "crosstl_ctor_Wide_1(q)" not in output
