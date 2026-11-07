"""An exact-match constrained template outranks a visible overload that needs a conversion."""

import pytest

import crosstl
from crosstl.backend.Metal.overload_precedence import exact_template_outranks
from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalTemplateSpecializationError,
)

SOURCE = """#include <metal_stdlib>
using namespace metal;
template <typename T, enable_if_t<!is_same_v<T, long>, bool> = true>
inline T total(T val) {{ return val + T(1); }}
inline {competitor_return} total({competitor}) {{ return {competitor_body}; }}
kernel void k(device float* out [[buffer(0)]], device float2* out2 [[buffer(1)]],
              device int* ints [[buffer(2)]], uint i [[thread_position_in_grid]]) {{
  {call}
}}
"""


def _translate(tmp_path, competitor, call, competitor_return="float2", body="val"):
    path = tmp_path / "total.metal"
    path.write_text(
        SOURCE.format(
            competitor=competitor,
            competitor_return=competitor_return,
            competitor_body=body,
            call=call,
        )
    )
    return crosstl.translate(str(path), backend="directx", format_output=False)


def test_scalar_call_selects_the_template_over_a_vector_overload(tmp_path):
    output = _translate(tmp_path, "float2 val", "float x = out[i]; out[i] = total(x);")
    assert "out_[i] = total_float(x);" in output
    assert "float total_float(float val)" in output


def test_integer_call_selects_the_template_over_a_float_overload(tmp_path):
    output = _translate(
        tmp_path,
        "float val",
        "int n = ints[i]; ints[i] = total(n);",
        competitor_return="float",
    )
    assert "total_int(n)" in output


@pytest.mark.parametrize(
    "competitor,competitor_return,call",
    (
        # The non-template matches exactly too, so it wins; not ranked here.
        ("float2 val", "float2", "float2 v = out2[i]; out2[i] = total(v);"),
        # A reference parameter of the argument's type is also an exact match.
        ("thread const float& val", "float", "float x = out[i]; out[i] = total(x);"),
        # Another spelling of the argument's type.
        ("int32_t val", "int", "int n = ints[i]; ints[i] = total(n);"),
        # A competitor type the proof cannot read.
        ("Custom val", "float", "float x = out[i]; out[i] = total(x);"),
    ),
)
def test_unproven_precedence_is_still_refused(
    tmp_path, competitor, competitor_return, call
):
    source_competitor = competitor
    prelude = "struct Custom { float v; };\n" if "Custom" in competitor else ""
    with pytest.raises(MetalTemplateSpecializationError, match="unproven precedence"):
        path = tmp_path / "total.metal"
        path.write_text(
            prelude
            + SOURCE.format(
                competitor=source_competitor,
                competitor_return=competitor_return,
                competitor_body=(
                    "Custom{0.0f}.v"
                    if "Custom" in competitor
                    else competitor_return + "(val)"
                ),
                call=call,
            )
        )
        crosstl.translate(str(path), backend="directx", format_output=False)


@pytest.mark.parametrize(
    "template_parameters,bindings,arguments,competitor,expected",
    (
        ("T val", {"T": "float"}, ["float"], ["float2 val"], True),
        ("T val", {"T": "int"}, ["int"], ["uint val"], True),
        ("T a, T b", {"T": "half"}, ["half", "half"], ["half a", "float b"], True),
        ("T val", {"T": "float2"}, ["float2"], ["float2 val"], False),
        ("T val", {"T": "int"}, ["int"], ["int32_t val"], False),
        ("thread const T& val", {"T": "float"}, ["float"], ["float2 val"], False),
        ("T val", {"T": "float"}, ["float"], ["thread float2& val"], False),
        ("T val", {"T": "float"}, [None], ["float2 val"], False),
        ("T val", {"T": "float"}, ["Custom"], ["float2 val"], False),
        # The template does not match its argument exactly.
        ("float val", {}, ["half"], ["float2 val"], False),
        ("T val", {"T": "float"}, ["float"], [], False),
    ),
)
def test_exact_template_outranks(
    template_parameters, bindings, arguments, competitor, expected
):
    assert (
        exact_template_outranks(
            MetalPreprocessor(), template_parameters, bindings, arguments, competitor
        )
        is expected
    )
