"""Mojo output combines boolean vector masks per component under && and ||."""

import re
import subprocess

import pytest

import crosstl
from tools.coverage import validators


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
        "shader P { bool f(bool a, bool b, float x) { return a && b || x < 1.0; } }",
    )
    assert "((a and b) or (x < 1.0))" in output
    assert "&" not in output.split("fn f", 1)[1]


def test_masks_of_different_widths_are_refused(tmp_path):
    with pytest.raises(ValueError, match="masks of the same width"):
        translate(
            tmp_path,
            "shader P { bvec2 f(bvec2 a, bvec3 b) { return a && b; } }",
        )


@pytest.mark.parametrize("width", [2, 3, 4])
def test_boolean_vector_logical_ops_run_with_native_mojo(tmp_path, width):
    mojo = validators.command("mojo")
    if not mojo:
        pytest.skip("Mojo compiler not installed")
    output = translate(
        tmp_path,
        f"""
shader Probe {{
    bvec{width} both(bvec{width} a, bvec{width} b) {{ return a && b; }}
    bvec{width} either(bvec{width} a, bvec{width} b) {{ return a || b; }}
    bool any_both(bvec{width} a, bvec{width} b) {{ return any(a && b); }}
    bool all_either(bvec{width} a, bvec{width} b) {{ return all(a || b); }}
}}
""",
    )
    storage = 4 if width == 3 else width
    mask = f"SIMD[DType.bool, {storage}]"
    padding = ["False"] if width == 3 else []
    first = (["True", "False"] * 2)[:width] + padding
    second = (["False", "True"] * 2)[:width] + padding
    # As in test_mojo_vector_comparison_types.py, compile the generated return
    # expressions in current-syntax wrappers.
    expressions = {
        name: re.search(rf"fn {name}\([^\n]*\n\s+return (.+)", output).group(1)
        for name in ("both", "either", "any_both", "all_either")
    }
    program = f"""
def both(a: {mask}, b: {mask}) -> {mask}:
    return {expressions["both"]}

def either(a: {mask}, b: {mask}) -> {mask}:
    return {expressions["either"]}

def any_both(a: {mask}, b: {mask}) -> Bool:
    return {expressions["any_both"]}

def all_either(a: {mask}, b: {mask}) -> Bool:
    return {expressions["all_either"]}

def main():
    var a = {mask}({", ".join(first)})
    var b = {mask}({", ".join(second)})
    print(both(a, b))
    print(either(a, b))
    print(any_both(a, b))
    print(all_either(a, b))
"""
    path = tmp_path / "run-logical.mojo"
    path.write_text(program)
    result = subprocess.run(
        [*mojo, "run", str(path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    # `and` and `or` would return a whole operand: b for a && b and a for a || b,
    # since a has a true lane.
    either_lanes = ["True"] * width + padding
    assert result.stdout.splitlines() == [
        f"[{', '.join(['False'] * storage)}]",
        f"[{', '.join(either_lanes)}]",
        "False",
        "True",
    ]
