"""Swizzles of typed postfix expressions type constrained-call arguments."""

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

LOCALS = {
    "v": "float4",
    "h": "half4",
    "s": "Custom",
    "i": "int",
    "a": "float",
    "b": "float2",
}


@pytest.mark.parametrize(
    "expression,expected",
    (
        ("v.xy.x", "float"),
        ("(v + v).zw", "float2"),
        ("int2(1, 2).y", "int"),
        ("float4(1.0f).w", "float"),
        ("h.rgb", "half3"),
        ("a + b.x", "float"),
    ),
)
def test_swizzle_of_typed_base(expression, expected):
    assert MetalPreprocessor()._infer_argument_type(expression, {}, LOCALS) == expected


@pytest.mark.parametrize(
    "expression",
    (
        "b.z",  # out of range for a float2
        "v.xg",  # mixed component sets
        "s.x",  # a struct member, not a swizzle
        "unknown(v).x",  # untyped base
        "v.xyzwx",  # five components
    ),
)
def test_untyped_swizzles(expression):
    assert MetalPreprocessor()._infer_argument_type(expression, {}, LOCALS) is None


SOURCE = """#include <metal_stdlib>
using namespace metal;
template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
inline T larger(T a, T b) {{ return a < b ? b : a; }}
kernel void k(device const float2* partials [[buffer(0)]], device float* out [[buffer(1)]],
              uint tid [[thread_position_in_grid]]) {{
  float best = 0.0f;
  for (uint i = 0; i < 4; i++) {{
    best = larger({call});
  }}
  out[tid] = best;
}}
"""


@pytest.mark.parametrize(
    "call",
    (
        "best, partials[i].x",
        "partials[i].x, partials[i].y",
        "best, (partials[i] * 2.0f).y",
    ),
)
def test_constrained_call_with_component_arguments_resolves(tmp_path, call):
    path = tmp_path / "components.metal"
    path.write_text(SOURCE.format(call=call))
    output = crosstl.translate(str(path), backend="directx", format_output=False)
    assert "larger_float(" in output
