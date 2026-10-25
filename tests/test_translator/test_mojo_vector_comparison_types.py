"""Public Mojo translation preserves component-wise comparison masks."""

import re
import shutil
import subprocess

import pytest

import crosstl

OPS = [("<", "lt"), ("<=", "le"), (">", "gt"), (">=", "ge"), ("==", "eq"), ("!=", "ne")]


def source(width, prefix="vec", op="<"):
    vector = f"{prefix}{width}"
    return f"""
shader Probe {{
    bvec{width} compare({vector} a, {vector} b) {{ return a {op} b; }}
    bool either({vector} a, {vector} b) {{ return any(a {op} b); }}
    bool every({vector} a, {vector} b) {{ return all(a {op} b); }}
}}
"""


def translate(tmp_path, code):
    path = tmp_path / "vector-comparison.cgl"
    path.write_text(code)
    return crosstl.translate(str(path), backend="mojo", format_output=False)


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("prefix", ["vec", "ivec", "uvec"])
@pytest.mark.parametrize("op,method", OPS)
def test_vector_comparisons_emit_boolean_masks(tmp_path, width, prefix, op, method):
    output = translate(tmp_path, source(width, prefix, op))
    assert f"SIMD[DType.bool, {4 if width == 3 else width}]" in output
    assert f".{method}(" in output
    assert ".reduce_or()" in output
    assert ".reduce_and()" in output
    assert not re.search(r"\b(?:any|all)\(", output)


def test_scalar_comparison_remains_scalar(tmp_path):
    output = translate(
        tmp_path, "shader Probe { bool scalar(float a, float b) { return a < b; } }"
    )
    assert "(a < b)" in output
    assert ".lt(" not in output


@pytest.mark.parametrize("width", [2, 3, 4])
def test_boolean_reductions_run_with_native_mojo(tmp_path, width):
    mojo = shutil.which("mojo")
    if mojo is None:
        pytest.skip("Mojo compiler not installed")
    # Equal padding must not make a three-lane <= reduction true; unequal
    # logical lanes must not make a three-lane < all-reduction false.
    code = source(width, op="<=").replace("return all(a <= b)", "return all(a < b)")
    output = translate(tmp_path, code)
    storage = 4 if width == 3 else width
    low = ", ".join(["0.0"] * width + (["0.0"] if width == 3 else []))
    high = ", ".join(["1.0"] * width + (["0.0"] if width == 3 else []))
    # The repository still emits legacy `fn` headers, rejected by its pinned
    # compiler. Compile the unchanged generated return expressions in current
    # syntax wrappers; this validates mask semantics, not whole-module syntax.
    expressions = {
        name: re.search(rf"fn {name}\([^\n]*\n\s+return (.+)", output).group(1)
        for name in ("either", "every")
    }
    output = f"""
def either(a: SIMD[DType.float32, {storage}], b: SIMD[DType.float32, {storage}]) -> Bool:
    return {expressions['either']}

def every(a: SIMD[DType.float32, {storage}], b: SIMD[DType.float32, {storage}]) -> Bool:
    return {expressions['every']}

def main():
    var low = SIMD[DType.float32, {storage}]({low})
    var high = SIMD[DType.float32, {storage}]({high})
    print(either(high, low))
    print(every(low, high))
"""
    path = tmp_path / "run-comparison.mojo"
    path.write_text(output)
    result = subprocess.run(
        [mojo, "run", str(path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["False", "True"]


@pytest.mark.parametrize("scalar_left", [False, True])
def test_vector_scalar_comparison_preserves_mask_shape(tmp_path, scalar_left):
    parameters = "float a, vec2 b" if scalar_left else "vec2 a, float b"
    output = translate(
        tmp_path, f"shader Probe {{ bvec2 compare({parameters}) {{ return a < b; }} }}"
    )
    assert "SIMD[DType.bool, 2]" in output
    assert ".lt(" in output
    if scalar_left:
        assert "SIMD[DType.float32, 2](a)" in output


def test_user_defined_any_is_not_replaced(tmp_path):
    output = translate(
        tmp_path,
        "shader Probe { float any(float a) { return a; } float caller(float a) { return any(a); } }",
    )
    assert "any(a)" in output
    assert ".reduce_or()" not in output
