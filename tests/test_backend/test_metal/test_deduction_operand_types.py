"""Operand shapes that type call arguments for template deduction in project materialization.

Materializing ``outer<float>`` deduces the template arguments of the calls in its body from their argument types
([temp.deduct.call]). Arguments such as ``-x``, ``exp(-x)`` and ``x * M_SQRT1_2_F`` used to have no inferred type,
so every template-hostile target refused the program.
"""

import pytest

import crosstl
from crosstl.project.pipeline import _split_metal_expression_operator

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"


def _kernel(argument):
    return f"""
template <typename T> METAL_FUNC T inner(T in) {{ return in; }}
template <typename T> METAL_FUNC T outer(T x) {{ return static_cast<T>(inner({argument})); }}
kernel void k(device float* f [[buffer(0)]], device half* h [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {{
  f[i] = outer(f[i]);
  h[i] = outer(h[i]);
}}
"""


def _translate(tmp_path, source, backend="directx"):
    path = tmp_path / "kernel.metal"
    path.write_text(PREFIX + source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


@pytest.mark.parametrize(
    "argument",
    ["-x", "exp(-x)", "precise::tanh(x)", "x * -x", "1 + exp(-x)"],
)
def test_same_type_operands_deduce_the_operand_type(tmp_path, argument):
    hlsl = _translate(tmp_path, _kernel(argument))
    assert "float inner_float(float in_)" in hlsl
    assert "float16_t inner_half(float16_t in_)" in hlsl


def test_predefined_float_constant_promotes_half_operands(tmp_path):
    # half * float is float (usual arithmetic conversions), so both outers call inner<float>.
    hlsl = _translate(tmp_path, _kernel("x * M_SQRT1_2_F"))
    assert "float inner_float(float in_)" in hlsl
    assert "inner_half" not in hlsl
    # The half operand is widened before the float multiplication.
    assert (
        "inner_float((__crossgl_binary16_to_float(uint(asuint16(x))) * "
        "0.707106781186547524400844362104849039f))" in hlsl
    )


def test_unary_signs_arrows_and_updates_do_not_split_binary_operators():
    assert _split_metal_expression_operator("x * -y", "-") == []
    assert _split_metal_expression_operator("x * -y", "*") == ["x", "-y"]
    assert _split_metal_expression_operator("-x", "-") == []
    assert _split_metal_expression_operator("p->x - 1", "-") == ["p->x", "1"]
    assert _split_metal_expression_operator("i++ - j", "-") == ["i++", "j"]
    assert _split_metal_expression_operator("a+-b", "+") == ["a", "-b"]
    assert _split_metal_expression_operator("a - b", "-") == ["a", "b"]
