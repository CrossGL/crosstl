"""Vector equality retains lane-wise Boolean results in nested expressions."""

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLScalarConversionError,
)
from tests.test_translator.test_metal_builtin_ownership import _compile


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("operator,intrinsic", [("==", "equal"), ("!=", "notEqual")])
@pytest.mark.parametrize("kind", ["float", "int", "uint", "bool"])
def test_vector_equality_preserves_boolean_result_context(
    tmp_path, width, operator, intrinsic, kind
):
    source = f"""shader Comparison {{
        bool{width} compare({kind}{width} left, {kind}{width} right) {{
            bool{width} local = left {operator} right;
            return left {operator} right;
        }}
        compute {{ @numthreads(1, 1, 1) void main() {{
            bool{width} result = compare({kind}{width}(1), {kind}{width}(0));
        }} }}
    }}"""
    generated = GLSLCodeGen().generate_stage(parse(source), "compute")
    assert generated.count(f"{intrinsic}(left, right)") == 2
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize("operator", ["==", "!="])
def test_scalar_equality_is_not_changed(operator):
    source = f"shader Comparison {{ bool compare(float left, float right) {{ return left {operator} right; }} }}"
    generated = GLSLCodeGen().generate(parse(source))
    assert f"(left {operator} right)" in generated


@pytest.mark.parametrize("operator", ["==", "!="])
def test_implicit_boolean_vector_reduction_remains_rejected(operator):
    source = f"shader Comparison {{ bool compare(float3 left, float3 right) {{ return left {operator} right; }} }}"
    with pytest.raises(OpenGLScalarConversionError) as raised:
        GLSLCodeGen().generate(parse(source))
    assert raised.value.reason == "vector-to-scalar"


def test_comparison_operands_are_evaluated_once():
    source = """shader Comparison {
        float2 next_value(inout uint count) { ++count; return float2(count); }
        bool2 compare(inout uint count) { return next_value(count) == next_value(count); }
    }"""
    generated = GLSLCodeGen().generate(parse(source))
    assert "equal(next_value(count), next_value(count))" in generated


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("operator,intrinsic", [("==", "equal"), ("!=", "notEqual")])
@pytest.mark.parametrize("kind", ["float", "int", "uint", "bool"])
@pytest.mark.parametrize("reduction", ["any", "all"])
def test_nested_vector_equality_compiles(
    tmp_path, width, operator, intrinsic, kind, reduction
):
    source = f"""shader Comparison {{
        bool compare({kind}{width} left, {kind}{width} right) {{
            return {reduction}(left {operator} right);
        }}
        compute {{ @numthreads(1, 1, 1) void main() {{
            bool result = compare({kind}{width}(1), {kind}{width}(0));
        }} }}
    }}"""
    generated = GLSLCodeGen().generate_stage(parse(source), "compute")
    assert f"{reduction}({intrinsic}(left, right))" in generated
    _compile(generated, "opengl", tmp_path)


def test_nested_boolean_vector_comparisons_compile(tmp_path):
    source = """shader Comparison {
        bool compare(float3 left, float3 right, float3 third) {
            return !any((left == right) != (right == third));
        }
        compute { @numthreads(1, 1, 1) void main() {
            bool result = compare(float3(1), float3(0), float3(2));
        } }
    }"""
    generated = GLSLCodeGen().generate_stage(parse(source), "compute")
    assert "any(notEqual(equal(left, right), equal(right, third)))" in generated
    _compile(generated, "opengl", tmp_path)
