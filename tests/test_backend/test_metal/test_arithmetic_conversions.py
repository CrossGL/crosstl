"""Built-in arithmetic between different Metal scalar and vector types is typed like the Metal compiler."""

import platform
import shutil
import subprocess

import pytest

from crosstl.backend.Metal.arithmetic_conversions import builtin_arithmetic_conversion
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

# (left, operator, right, result). Each result is what `xcrun metal -std=metal3.1` gives for
# decltype(a op b) on non-constant operands.
WELL_FORMED = (
    ("half", "*", "float", "float"),
    ("float", "*", "half", "float"),
    ("float", "-", "half", "float"),
    ("int", "*", "float", "float"),
    ("uint", "*", "float", "float"),
    ("bool", "*", "float", "float"),
    ("int", "*", "half", "half"),
    ("long", "*", "half", "half"),
    ("char", "*", "half", "half"),
    ("half", "/", "int", "half"),
    ("bfloat", "*", "float", "float"),
    ("bfloat", "*", "int", "bfloat"),
    ("bfloat", "*", "uint", "bfloat"),
    ("half4", "*", "float", "half4"),
    ("float", "*", "half4", "half4"),
    ("float4", "*", "half", "float4"),
    ("float4", "*", "int", "float4"),
    ("float3", "+", "uint", "float3"),
    ("half2", "*", "long", "half2"),
    ("float4", "*", "bfloat", "float4"),
    ("int4", "*", "long", "int4"),
    ("int4", "*", "uint", "int4"),
    ("uint4", "*", "int", "uint4"),
    ("uchar4", "*", "int", "uchar4"),
    ("ushort2", "*", "int", "ushort2"),
    ("short4", "*", "int", "short4"),
    ("short2", "*", "int", "short2"),
    ("short", "*", "int2", "int2"),
    ("int4", "%", "int", "int4"),
    ("bfloat4", "*", "int", "bfloat4"),
    ("bfloat4", "*", "bfloat", "bfloat4"),
)

ILL_FORMED = (
    ("half", "+", "bfloat"),
    ("int4", "*", "float"),
    ("int2", "*", "half"),
    ("half4", "*", "bfloat"),
    ("bfloat4", "*", "float"),
    ("bfloat4", "*", "half"),
    ("float4", "*", "half4"),
    ("float", "%", "int"),
)


@pytest.mark.parametrize("left,operator,right,result", WELL_FORMED)
def test_mixed_operands_take_the_metal_result_type(left, operator, right, result):
    preprocessor = MetalPreprocessor()
    assert (
        preprocessor._infer_argument_type(
            f"a {operator} b", {}, {"a": left, "b": right}
        )
        == result
    )


@pytest.mark.parametrize("left,operator,right", ILL_FORMED)
def test_ill_formed_mixes_stay_untyped(left, operator, right):
    preprocessor = MetalPreprocessor()
    assert (
        preprocessor._infer_argument_type(
            f"a {operator} b", {}, {"a": left, "b": right}
        )
        is None
    )
    assert builtin_arithmetic_conversion(preprocessor, left, right, operator) == (
        True,
        None,
    )


@pytest.mark.parametrize(
    "expression,left,result",
    (
        # An unsuffixed floating literal is float, so it outranks half.
        ("a * 1.5", "half", "float"),
        ("a * 1.5f", "half", "float"),
        ("a * 2.0h", "half", "half"),
        ("a * 1.5", "int", "float"),
        ("a * 1.5h", "int", "half"),
        ("a * 1.5", "bfloat", "float"),
        ("a * 1.5", "half4", "half4"),
        ("a * 2", "half", "half"),
        ("a * 2", "float4", "float4"),
    ),
)
def test_literals_follow_the_same_conversions(expression, left, result):
    assert (
        MetalPreprocessor()._infer_argument_type(expression, {}, {"a": left}) == result
    )


@pytest.mark.parametrize(
    "expression,left",
    (("a * 1.5", "int4"), ("a * 2.0h", "bfloat"), ("a * 1.5", "bfloat4")),
)
def test_ill_formed_literal_mixes_stay_untyped(expression, left):
    assert MetalPreprocessor()._infer_argument_type(expression, {}, {"a": left}) is None


@pytest.mark.parametrize(
    "left,right",
    (
        ("bool4", "int"),  # left to the existing rules
        ("float2x2", "float"),
        ("Custom", "float"),
        ("device float*", "int"),
    ),
)
def test_other_operand_types_are_not_decided_here(left, right):
    assert builtin_arithmetic_conversion(MetalPreprocessor(), left, right, "*") == (
        False,
        None,
    )


def _metal_compiler():
    if platform.system() != "Darwin" or not shutil.which("xcrun"):
        return None
    probe = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return ["xcrun", "-sdk", "macosx", "metal"] if probe.returncode == 0 else None


def _compile(compiler, tmp_path, name, body):
    source = tmp_path / f"{name}.metal"
    source.write_text(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        + body
        + "\nkernel void probe_kernel() {}\n"
    )
    return subprocess.run(
        [
            *compiler,
            "-std=metal3.1",
            "-c",
            str(source),
            "-o",
            str(tmp_path / f"{name}.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.skipif(
    _metal_compiler() is None, reason="the Metal compiler is not installed"
)
def test_the_metal_compiler_agrees(tmp_path):
    compiler = _metal_compiler()
    checks = "\n".join(
        f"void check{index}(thread {left}& a, thread {right}& b) {{ "
        f'static_assert(is_same_v<decltype(a {operator} b), {result}>, "{left} {operator} {right}"); }}'
        for index, (left, operator, right, result) in enumerate(WELL_FORMED)
    )
    compiled = _compile(compiler, tmp_path, "well_formed", checks)
    assert compiled.returncode == 0, compiled.stderr[-2000:]
    for index, (left, operator, right) in enumerate(ILL_FORMED):
        compiled = _compile(
            compiler,
            tmp_path,
            f"ill_formed{index}",
            f"void check(thread {left}& a, thread {right}& b) {{ (void)(a {operator} b); }}",
        )
        assert compiled.returncode != 0, f"{left} {operator} {right} compiled"
