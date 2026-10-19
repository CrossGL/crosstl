"""Keep known builtin result widths during contextual aggregate validation."""

import pytest

from crosstl import translate
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import (
    DirectXAggregateInitializerError,
    HLSLCodeGen,
)
from tests.test_translator.test_project_translation import (
    assert_directx_compute_validates_if_available,
)


@pytest.mark.parametrize(
    "function", sorted(HLSLCodeGen.HLSL_COMPONENTWISE_FLOAT_UNARY_FUNCTIONS)
)
@pytest.mark.parametrize(
    "argument_type,result_type,target_type",
    [("float", "vec2", "float2"), ("vec2", "vec4", "float4")],
)
def test_builtin_width_in_vector_initializer(
    tmp_path, function, argument_type, result_type, target_type
):
    source = f"""shader BuiltinAggregate {{
        {result_type} build({argument_type} theta) {{
            {result_type} value = {{{function}(theta), {function}(theta)}};
            return value;
        }}
        compute {{ void main() {{}} }}
    }}"""
    generated = HLSLCodeGen().generate(parse(source))
    assert (
        f"{target_type} value = {target_type}({function}(theta), {function}(theta));"
        in generated
    )
    assert_directx_compute_validates_if_available(generated, tmp_path)


def test_nested_builtin_initializer_keeps_single_evaluation(tmp_path):
    source = """shader NestedBuiltinAggregate {
        float next(inout int count) { count += 1; return float(count); }
        vec2 build(inout int count) {
            vec2 value = {cos(sin(next(count))), exp(sqrt(next(count)))};
            return value;
        }
        compute { void main() {} }
    }"""
    generated = HLSLCodeGen().generate(parse(source))
    assert (
        "float2 value = float2(cos(sin(next(count))), exp(sqrt(next(count))));"
        in generated
    )
    assert generated.count("next(count)") == 2
    assert_directx_compute_validates_if_available(generated, tmp_path)


def test_user_overload_return_type_precedes_builtin(tmp_path):
    source = """shader UserBuiltinAggregate {
        struct Angle { int raw; };
        vec2 sin(Angle value) { return vec2(float(value.raw)); }
        vec4 build(Angle value) {
            vec4 result = {sin(value), sin(value)};
            return result;
        }
        compute { void main() {} }
    }"""
    generated = HLSLCodeGen().generate(parse(source))
    assert "float4 result = float4(sin(value), sin(value));" in generated
    assert_directx_compute_validates_if_available(generated, tmp_path)


@pytest.mark.parametrize(
    "argument_type,call,reason",
    [
        ("vec2", "sin(theta)", "element-count-mismatch"),
        ("float", "unknown_function(theta)", "vector-element-type-unresolved"),
        ("float", "sin(theta, theta)", "vector-element-type-unresolved"),
        ("float", "sin(unknown_function(theta))", "vector-element-type-unresolved"),
    ],
)
def test_unproven_initializer_retains_diagnostic(argument_type, call, reason):
    source = f"""shader InvalidBuiltinAggregate {{
        vec2 build({argument_type} theta) {{
            vec2 value = {{{call}, {call}}};
            return value;
        }}
    }}"""
    with pytest.raises(DirectXAggregateInitializerError) as error:
        HLSLCodeGen().generate(parse(source))
    assert error.value.reason == reason


def test_metal_builtin_initializer_matches_saved_intermediate(tmp_path):
    source = tmp_path / "twiddle.metal"
    source.write_text(
        """#include <metal_stdlib>
        using namespace metal;
        kernel void build(device float2* output [[buffer(0)]],
                          constant float& theta [[buffer(1)]],
                          uint index [[thread_position_in_grid]]) {
            float2 twiddle = {metal::fast::cos(theta), metal::fast::sin(theta)};
            output[index] = twiddle;
        }""",
        encoding="utf-8",
    )
    generated = translate(str(source), backend="directx", format_output=False)
    intermediate = tmp_path / "twiddle.cgl"
    intermediate.write_text(
        translate(str(source), backend="cgl", format_output=False), encoding="utf-8"
    )
    assert (
        translate(str(intermediate), backend="directx", format_output=False)
        == generated
    )
    assert "float2 twiddle = float2(cos(build_theta), sin(build_theta));" in generated
    assert_directx_compute_validates_if_available(generated, tmp_path)
