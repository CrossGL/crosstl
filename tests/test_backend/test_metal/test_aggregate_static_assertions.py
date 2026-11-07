"""Static assertions over constexpr aggregates outside template materialization.

MLX's complex.h asserts ``(complex_t<half>{1, 2} * complex_t<half>{3, 4}).real
== -5``. Template materialization (DirectX, OpenGL, ...) decides it with the
Metal constexpr evaluator, but the Metal and CrossGL targets left it to the
translator, whose evaluator covers integral constant expressions only, so
every kernel including complex.h was refused for them. The importer now
decides aggregate conditions while the source still spells the struct
templates and their operators, and leaves integral and undecided ones to the
translator, which keeps CrossGL's declarations as before.
"""

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import MetalStaticAssertionError
from crosstl.translator.semantics.static_assertions import (
    StaticAssertionUnresolvedError,
)

COMPLEX = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
struct complex_t {
  T real;
  T imag;
  constexpr complex_t(T real, T imag) thread : real(real), imag(imag) {};
};

template <typename T>
constexpr complex_t<T> operator*(complex_t<T> a, complex_t<T> b) {
  return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

static_assert(sizeof(complex_t<float>) == 8, "two floats");
static_assert((complex_t<half>{1.0h, 2.0h} * complex_t<half>{3.0h, 4.0h}).real == EXPECTED);

kernel void k(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  complex_t<float> a(1.0f, 2.0f);
  out[tid] = (a * a).real;
}
"""


def _translate(tmp_path, code, backend):
    source = tmp_path / "complex.metal"
    source.write_text(code)
    return crosstl.translate(
        str(source), backend=backend, source_backend="metal", format_output=False
    )


@pytest.mark.parametrize("backend", ["metal", "cgl", "directx", "opengl"])
def test_true_aggregate_assertion_is_decided_for_every_target(tmp_path, backend):
    output = _translate(tmp_path, COMPLEX.replace("EXPECTED", "-5.0h"), backend)

    assert "complex_t" in output
    # The aggregate assertion is proven and removed; only CrossGL keeps the integral one.
    assertions = [line for line in output.splitlines() if "static_assert" in line]
    assert assertions == (
        ['    static_assert(8 == 8, "two floats");'] if backend == "cgl" else []
    )


def test_integral_assertion_is_still_carried_into_crossgl(tmp_path):
    output = _translate(tmp_path, COMPLEX.replace("EXPECTED", "-5.0h"), "cgl")

    # The translator folds the sizeof and keeps the declaration and its message.
    assert 'static_assert(8 == 8, "two floats");' in output


@pytest.mark.parametrize("backend", ["metal", "cgl"])
def test_false_aggregate_assertion_is_refused(tmp_path, backend):
    with pytest.raises(MetalStaticAssertionError) as caught:
        _translate(tmp_path, COMPLEX.replace("EXPECTED", "-6.0h"), backend)

    assert caught.value.reason == "assertion-failed"


def test_undecided_aggregate_assertion_is_left_to_the_translator(tmp_path):
    # Without a constexpr constructor the evaluator cannot build the operands.
    code = COMPLEX.replace(
        "  constexpr complex_t(T real, T imag) thread : real(real), imag(imag) {};\n",
        "",
    ).replace("complex_t<float> a(1.0f, 2.0f);", "complex_t<float> a = {1.0f, 2.0f};")

    with pytest.raises(StaticAssertionUnresolvedError):
        _translate(tmp_path, code.replace("EXPECTED", "-5.0h"), "metal")
