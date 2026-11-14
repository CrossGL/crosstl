"""Explicit Metal template calls that leave trailing parameters to deduction.

``offset<false>(ptr, n)`` against ``template <bool scaled, typename T> void offset(const device T*& p, int n)``
supplies ``scaled`` and deduces ``T`` from ``ptr`` ([temp.deduct.call]). A deduced argument takes precedence over a
declaration default ([temp.deduct]/5), and two arguments that deduce different types make the call ill-formed.
"""

import pytest

import crosstl
from crosstl.backend.Metal.call_template_deduction import (
    TemplateDeductionConflict,
    deduce_omitted_template_arguments,
)
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalTemplateArgumentResolutionError,
)

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"


def _translate(tmp_path, source, backend):
    path = tmp_path / "unit.metal"
    path.write_text(PREFIX + source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


POINTER_SOURCE = """
template <typename T>
void offset_ptr(const device T*& x, int s) {
  x += s;
}

template <bool scaled, typename T>
void offset_ptr(const device T*& x, int s) {
  x += scaled ? 2 * s : s;
}

template <bool scaled, typename T>
void run_impl(const device T* x, device T* y, uint i) {
  offset_ptr<scaled>(x, int(i));
  y[i] = x[0];
}

kernel void run(const device float* x [[buffer(0)]], device float* y [[buffer(1)]],
                uint i [[thread_position_in_grid]]) {
  run_impl<false>(x, y, i);
}
"""


@pytest.mark.parametrize("backend", ["metal", "cgl"])
def test_pointer_parameter_deduces_the_omitted_type(tmp_path, backend):
    code = _translate(tmp_path, POINTER_SOURCE, backend)

    assert "run_impl_false_float(x, y, i);" in code
    assert "offset_ptr_false_float(x, int(i));" in code
    assert "x += false ? 2 * s : s;" in code


def test_pointer_specializations_keep_metal_parameter_types(tmp_path):
    code = _translate(tmp_path, POINTER_SOURCE, "metal")

    assert "void offset_ptr_false_float(const device float* thread& x, int s)" in code
    assert (
        "void run_impl_false_float(const device float* x, device float* y, uint i)"
        in code
    )


def test_vector_shorthand_argument_deduces_the_element_type(tmp_path):
    code = _translate(
        tmp_path,
        """
template <int lane, typename T>
T pick(vec<T, 4> v) {
  return v[lane];
}

kernel void run(device float4* x [[buffer(0)]], device float* y [[buffer(1)]],
                uint i [[thread_position_in_grid]]) {
  float4 v = x[i];
  y[i] = pick<2>(v);
}
""",
        "metal",
    )

    assert "inline float pick_2_float(float4 v)" in code
    assert "y[i] = pick_2_float(v);" in code


def test_deduced_argument_takes_precedence_over_the_default(tmp_path):
    code = _translate(
        tmp_path,
        """
template <int factor, typename T = int>
T scale(T x) {
  return x * T(factor);
}

kernel void run(device float* y [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  float v = y[i];
  y[i] = scale<3>(v);
}
""",
        "metal",
    )

    assert "inline float scale_3_float(float x)" in code
    assert "y[i] = scale_3_float(v);" in code


def test_conflicting_deductions_are_refused(tmp_path):
    with pytest.raises(MetalTemplateArgumentResolutionError) as raised:
        _translate(
            tmp_path,
            """
template <bool first, typename T>
T choose(T a, T b) {
  return first ? a : b;
}

kernel void run(device float* y [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  float a = y[i];
  int b = int(i);
  y[i] = choose<true>(a, b);
}
""",
            "metal",
        )

    message = str(raised.value)
    assert "choose" in message
    assert "conflicting" in message


def test_undeducible_parameter_still_requires_an_argument(tmp_path):
    with pytest.raises(MetalTemplateArgumentResolutionError) as raised:
        _translate(
            tmp_path,
            """
template <int factor, typename T>
float widen(float x) {
  return x * float(factor) + float(sizeof(T));
}

kernel void run(device float* y [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  y[i] = widen<2>(y[i]);
}
""",
            "metal",
        )

    assert "was not supplied and has no declaration default" in str(raised.value)


def test_deduction_matches_parameter_shapes():
    parameters = [("value", "scaled"), ("typename", "T"), ("typename", "S")]

    bindings = deduce_omitted_template_arguments(
        parameters,
        1,
        ["const device T*&", "thread S&", "int"],
        ["const device float*", "half", "int"],
    )

    assert bindings == {"S": "half", "T": "float"}


def test_deduction_ignores_unknown_and_mismatched_arguments():
    parameters = [("value", "N"), ("typename", "T")]

    assert deduce_omitted_template_arguments(
        parameters, 1, ["vec<T, 2>", "T"], [None, "float3"]
    ) == {"T": "float3"}
    assert deduce_omitted_template_arguments(parameters, 1, ["T*"], ["float"]) == {}
    assert deduce_omitted_template_arguments(parameters, 2, ["T"], ["float"]) == {}


def test_deduction_binds_value_parameters_from_vector_extents():
    bindings = deduce_omitted_template_arguments(
        [("typename", "T"), ("value", "N")],
        0,
        ["metal::vec<T, N>"],
        ["half3"],
    )

    assert bindings == {"N": "3", "T": "half"}


def test_conflicting_deductions_raise():
    with pytest.raises(TemplateDeductionConflict) as raised:
        deduce_omitted_template_arguments(
            [("value", "B"), ("typename", "T")],
            1,
            ["T", "T"],
            ["float", "int"],
        )

    assert raised.value.parameter_name == "T"
    assert raised.value.values == ("float", "int")
