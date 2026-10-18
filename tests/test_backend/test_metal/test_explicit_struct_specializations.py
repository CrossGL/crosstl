"""Full explicit struct specializations used as concrete method receivers."""

import pytest

import crosstl
from tests.test_translator.test_metal_builtin_ownership import _compile

# scan.h's shape: CumProd<bool> replaces the primary's methods, and a kernel
# template receives it as `Op op; op.method(...)` plus `Op::init`.
SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <typename U>
struct Combine {
  static constexpr constant U init = static_cast<U>(1);
  U apply(U a, U b) thread {
    return a * b;
  }
};

template <>
struct Combine<bool> {
  static constexpr constant bool init = true;
  bool apply(bool a, bool b) thread {
    return a & b;
  }
};

template <typename T>
struct Limit {
  static constexpr constant T low = static_cast<T>(0);
};

template <>
struct Limit<bool> {
  static constexpr constant bool low = false;
};

template <typename T, typename Op>
[[kernel]] void combine_values(
    device const T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  Op op;
  T low = Limit<T>::low;
  out[index] = in[index] == low ? low : op.apply(in[index], Op::init);
}

template [[host_name("combine_bool")]] [[kernel]] void
combine_values<bool, Combine<bool>>(device const bool*, device bool*, uint);
template [[host_name("combine_int")]] [[kernel]] void
combine_values<int, Combine<int>>(device const int*, device int*, uint);
"""


def _translate(tmp_path, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(SOURCE, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def test_method_bearing_explicit_specialization_becomes_a_concrete_receiver(
    tmp_path,
):
    crossgl = " ".join(_translate(tmp_path, "cgl").split())
    # The specialization keeps its own body; the primary still serves int.
    assert "struct Combine_bool {" in crossgl
    assert "Combine_bool op;" in crossgl and "Combine_int op;" in crossgl
    assert "Combine_bool__apply(op, " in crossgl
    assert "return int(a) & int(b);" in crossgl
    assert "return a * b;" in crossgl
    # Data-only explicit specializations keep their static-member path.
    assert "struct Limit_bool" not in crossgl


@pytest.mark.parametrize("backend", ("metal", "opengl", "directx"))
def test_explicit_specialization_receivers_compile_natively(tmp_path, backend):
    generated = _translate(tmp_path, backend)
    assert "Combine<bool>" not in generated
    _compile(generated, backend, tmp_path)
