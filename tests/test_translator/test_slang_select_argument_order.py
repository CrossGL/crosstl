"""Slang's select takes the condition first, unlike CrossGL's select."""

from __future__ import annotations

import crosstl
from crosstl.translator.codegen.slang_codegen import SlangCodeGen

TEMPLATE = """
shader SelectOrder {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        HELPERvoid main(RWStructuredBuffer<float> output @ binding(0)) {
            float x = output[0];
            bool b = x > 1.0;
            STATEMENT
        }
    }
}
"""


def generate(statement, helper=""):
    source = TEMPLATE.replace("STATEMENT", statement).replace("HELPER", helper)
    return SlangCodeGen().generate(crosstl.translator.parse(source))


def test_scalar_condition_is_reordered():
    generated = generate("output[1] = select(10.0, 20.0, b);")
    assert "select(10.0, 20.0, b)" not in generated


def test_scalar_condition_with_vector_values_is_reordered():
    generated = generate(
        "vec3 v = select(vec3(1.0), vec3(2.0), b); output[1] = v.x;"
    )
    assert "select(float3(1.0), float3(2.0), b)" not in generated


def test_bvec_mask_is_reordered():
    generated = generate(
        "vec3 v = select(vec3(1.0), 2.0, bvec3(b, !b, b)); output[1] = v.x;"
    )
    assert "select(bool3(b, !b, b), 2.0, float3(1.0))" in generated


def test_any_condition_is_reordered():
    generated = generate(
        "output[1] = select(10.0, 20.0, any(lessThan(vec2(x), vec2(1.0))));"
    )
    assert "select(any(float2(x) < float2(1.0)), 20.0, 10.0)" in generated


def test_user_defined_select_is_unchanged():
    helper = (
        "float select(float first, float second, bool flag) "
        "{ if (flag) { return first; } return second; }\n        "
    )
    generated = generate("output[1] = select(10.0, 20.0, b);", helper)
    assert "select(10.0, 20.0, b)" in generated

