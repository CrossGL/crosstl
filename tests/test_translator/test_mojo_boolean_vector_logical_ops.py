"""Mojo output combines boolean vector masks per component under && and ||."""

import pytest

import crosstl


def translate(tmp_path, code):
    path = tmp_path / "boolean-vector-logic.cgl"
    path.write_text(code)
    return crosstl.translate(str(path), backend="mojo", format_output=False)


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("op,lane_op", [("&&", "&"), ("||", "|")])
def test_boolean_vector_operands_use_lane_wise_operators(tmp_path, width, op, lane_op):
    output = translate(
        tmp_path,
        f"shader P {{ bvec{width} f(bvec{width} a, bvec{width} b) "
        f"{{ return a {op} b; }} }}",
    )
    assert f"return (a {lane_op} b)" in output
    assert " and " not in output
    assert " or " not in output


def test_comparison_masks_combine_lane_wise(tmp_path):
    output = translate(
        tmp_path,
        "shader P { bvec3 cmp(vec3 x, vec3 y, vec3 z) "
        "{ return (x < y) && (y < z) || (x == z); } }",
    )
    assert "(((x).lt(y) & (y).lt(z)) | (x).eq(z))" in output


def test_reduced_mask_combination(tmp_path):
    output = translate(
        tmp_path,
        "shader P { bool f(bvec2 a, bvec2 b) { return any(a && b); } }",
    )
    assert "((a & b)).reduce_or()" in output


def test_scalar_bool_operands_keep_short_circuit_operators(tmp_path):
    output = translate(
        tmp_path,
        "shader P { bool f(bool a, bool b, float x) " "{ return a && b || x < 1.0; } }",
    )
    assert "((a and b) or (x < 1.0))" in output
    assert "&" not in output.split("fn f", 1)[1]
