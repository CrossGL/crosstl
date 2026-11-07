"""Constrained free-function calls in concrete helpers that no kernel calls.

Every concrete function is translated, so a constrained call in an unused ``inline`` helper must bind to the
template it selects. It must not be left for codegen, whose only remaining source candidate may be an explicit
specialization for another type. A call there that cannot be resolved is not a program error, because no entry
point evaluates it. A call that a kernel reaches keeps its strict resolution.
"""

import textwrap

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalTemplateSpecializationError,
)

MAX_HELPERS = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    namespace util {
    template <typename T>
    ::metal::enable_if_t<::metal::is_floating_point_v<T>, T> larger(T a, T b) {
      return ::metal::isunordered(a, b) ? NAN : ::metal::max(b, a);
    }
    template <>
    inline bfloat larger(bfloat a, bfloat b) {
      return bfloat(::metal::max(float(b), float(a)));
    }
    inline float widest(float a, float b) {
      auto m = larger(a, b);
      return m;
    }
    } // namespace util
    kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
      out[i] = out[i] + 1.0f;
    }
""")

AMBIGUOUS = textwrap.dedent("""
    #include <metal_stdlib>
    template <typename T>
    metal::enable_if_t<metal::is_floating_point_v<T>, T> pick(T v) {
      return v;
    }
    template <typename T>
    metal::enable_if_t<metal::is_arithmetic_v<T>, T> pick(T v) {
      return v + v;
    }
    template <typename T>
    metal::enable_if_t<metal::is_floating_point_v<T>, T> half_of(T v) {
      return v * T(0.5);
    }
    inline float unused(float v) {
      return pick(v);
    }
    kernel void k(device float* f [[buffer(0)]], uint id [[thread_position_in_grid]]) {
      f[id] = half_of(f[id]);
      CALL;
    }
""")


def test_unreachable_helper_binds_the_constrained_template():
    output = MetalPreprocessor().preprocess(MAX_HELPERS)
    assert "auto m = ::util::larger_float(a, b);" in output
    assert "inline float larger_float(float a, float b)" in output


@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan", "metal"])
def test_unreachable_helper_with_constrained_call_translates(tmp_path, backend):
    source = tmp_path / "helpers.metal"
    source.write_text(MAX_HELPERS)
    output = crosstl.translate(
        str(source), backend=backend, format_output=False, source_backend="metal"
    )
    if backend != "vulkan":
        assert "larger_float(a, b)" in output
        assert "larger_float(float a, float b)" in output


def test_unresolvable_call_in_unreachable_helper_is_left_unbound():
    output = MetalPreprocessor().preprocess(AMBIGUOUS.replace("CALL;", ""))
    # The kernel's call still binds; the helper keeps its ambiguous call untouched.
    assert "half_of_float(f[id])" in output
    assert "return pick(v);" in output


def test_unresolvable_call_in_a_kernel_is_still_refused():
    with pytest.raises(MetalTemplateSpecializationError, match="pick"):
        MetalPreprocessor().preprocess(
            AMBIGUOUS.replace("CALL;", "f[id] = pick(f[id]);")
        )


CHAINED = textwrap.dedent("""
    #include <metal_stdlib>
    template <typename T>
    metal::enable_if_t<metal::is_floating_point_v<T>, T> larger(T a, T b) {
      return metal::max(a, b);
    }
    template <typename T, metal::enable_if_t<metal::is_floating_point_v<T>, bool> = true>
    inline T halve(const T v) {
      return v * T(0.5);
    }
    kernel void k(device float* f [[buffer(0)]], uint id [[thread_position_in_grid]]) {
      auto a = larger(f[id], 1.0f);
      f[id] = halve(a);
    }
""")


def test_argument_typed_by_an_earlier_binding_binds_in_a_later_round():
    # ``a`` has a type only once ``larger`` is bound to ``larger_float``, whose enabler names ``float``.
    output = MetalPreprocessor().preprocess(CHAINED)
    assert "auto a = larger_float(f[id], 1.0f);" in output
    assert "f[id] = halve_float(a);" in output
    assert "inline float larger_float(float a, float b)" in output


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_chained_constrained_calls_translate(tmp_path, backend):
    source = tmp_path / "chained.metal"
    source.write_text(CHAINED)
    output = crosstl.translate(
        str(source), backend=backend, format_output=False, source_backend="metal"
    )
    assert "halve_float(a)" in output


def test_argument_that_never_gets_a_type_is_still_refused():
    source = CHAINED.replace(
        "auto a = larger(f[id], 1.0f);", "auto a = unknown(f[id]);"
    )
    with pytest.raises(MetalTemplateSpecializationError, match="halve"):
        MetalPreprocessor().preprocess(source)


def test_later_round_reuses_an_earlier_instantiation():
    # Round 1 binds ``halve(f[id])`` and ``larger``; round 2 binds ``halve(a)`` to the same ``halve<float>``.
    source = CHAINED.replace("f[id] = halve(a);", "f[id] = halve(a) + halve(f[id]);")
    output = MetalPreprocessor().preprocess(source)
    assert "f[id] = halve_float(a) + halve_float(f[id]);" in output
    assert output.count("inline float halve_float(") == 1
    assert "halve_float_1" not in output
