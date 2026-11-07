"""Implicit template materialization does not type a call by another overload's return type."""

import pytest

import crosstl
from crosstl.project.pipeline import MetalSourceMaterializationError

SOURCE = """#include <metal_stdlib>
using namespace metal;
float f(float x) {{ return x * 2.0f; }}
{other} f({other_parameter} x) {{ return {other}(x); }}
template <typename T>
T g(T v) {{ return v + T(1); }}
kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {{
  float x = out[i];
  out[i] = g(f(x));
}}
"""


def _translate(tmp_path, other, other_parameter, backend="directx"):
    path = tmp_path / "overloads.metal"
    path.write_text(SOURCE.format(other=other, other_parameter=other_parameter))
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", ("directx", "vulkan", "opengl"))
def test_overloads_with_different_return_types_do_not_type_the_call(tmp_path, backend):
    # f(float) is selected, so g is g<float>. The later f(half) definition used to
    # type the call as half, which materialized g<half> and rounded the value.
    try:
        output = _translate(tmp_path, "half", "half", backend)
    except MetalSourceMaterializationError as error:
        assert "g missing T" in str(error)
    else:
        assert "g_half" not in output
        assert "g_float" in output


def test_overloads_with_one_return_type_still_type_the_call(tmp_path):
    output = _translate(tmp_path, "float", "half")
    assert "g_float(" in output
    assert "g_half" not in output
