"""Base-ten logarithms under the base-two logarithm profiles.

``binary32_log2_operand_profile`` and ``binary32_log2_accuracy_profile`` also
govern Metal ``log10``: a profiled call becomes the base-two helper's result
times log10(2), so it keeps the helper's operand policy and finite accuracy
instead of the target's own base-two logarithm near one.
"""

import json
import os
import sys
from decimal import Decimal, localcontext
from functools import lru_cache, partial
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalPreciseMathLoweringError,
    MetalToCrossGLConverter,
)
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.translator.precise_log2 import LOG2_PROFILED_OPERATIONS, LOG10_OF_TWO
from tests.ci_helpers import assert_native_step_prerequisites
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_log import (
    FIELDS,
    GUARD,
    LANE_SIGNS,
    _float,
    _inputs,
    _round_decimal,
)

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_LOG2"
OPERANDS = (None, "preserve-subnormals", "flush-subnormals")
ACCURACY = (None, "portable-finite")
TARGETS = ("metal", "opengl", "directx")
# The base-two helper's four-ULP bound, plus the rounding of the product and
# of log10(2): a relative error of four binary32 ULPs of log2(x) can be up to
# eight ULPs of log10(x) when the two results sit at opposite ends of their
# binades.
MAXIMUM_ULPS = 9
SOURCE = """#include <metal_stdlib>
using namespace metal;
using Scalar = float;
using Quad = float4;
float record(thread uint& count, float value) { count += 1; return value; }
Scalar evaluate(Scalar value) { return metal::log10(value); }
kernel void logarithms(const device uint* values [[buffer(0)]],
                      device uint* results [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    float x = as_type<float>(values[i]);
    float negative = as_type<float>(values[i] ^ 0x80000000u);
    uint count = 0;
    float scalar = evaluate(x);
    float2 pair = metal::precise::log10(float2(x, negative));
    float3 triple = precise::log10(float3(x, negative, x));
    Quad quad = metal::log10(Quad(record(count, x), negative, x, negative));
    results[4u + 12u*i] = as_type<uint>(scalar);
    results[5u + 12u*i] = as_type<uint>(pair.x);
    results[6u + 12u*i] = as_type<uint>(pair.y);
    results[7u + 12u*i] = as_type<uint>(triple.x);
    results[8u + 12u*i] = as_type<uint>(triple.y);
    results[9u + 12u*i] = as_type<uint>(triple.z);
    results[10u + 12u*i] = as_type<uint>(quad.x);
    results[11u + 12u*i] = as_type<uint>(quad.y);
    results[12u + 12u*i] = as_type<uint>(quad.z);
    results[13u + 12u*i] = as_type<uint>(quad.w);
    results[14u + 12u*i] = count;
    results[15u + 12u*i] = as_type<uint>(x);
}
"""


def _translate(tmp_path, source=SOURCE, target="crossgl", operand=None, accuracy=None):
    path = tmp_path / "log10.metal"
    path.write_text(source, encoding="utf-8")
    return translate(
        str(path),
        backend=target,
        format_output=False,
        source_options={
            "binary32_log2_operand_profile": operand,
            "binary32_log2_accuracy_profile": accuracy,
        },
    )


def _convert(source, **options):
    converter = MetalToCrossGLConverter(**options)
    return converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())


def test_log10_shares_the_profiled_base_two_logarithm():
    assert LOG2_PROFILED_OPERATIONS == {"log2", "log10"}
    with localcontext() as context:
        context.prec = 60
        exact = Decimal(2).ln() / Decimal(10).ln()
    assert abs(Decimal(LOG10_OF_TWO) - exact) < Decimal("1e-20")


@pytest.mark.parametrize("operand", OPERANDS)
@pytest.mark.parametrize("accuracy", ACCURACY)
@pytest.mark.parametrize("target", TARGETS)
def test_log10_profiles_compile_and_survive_saved_crossgl(
    tmp_path, target, operand, accuracy
):
    generated = _translate(tmp_path, target=target, operand=operand, accuracy=accuracy)
    profiled = operand is not None or accuracy is not None
    assert ("metal_log2_float" in generated) == profiled
    saved = tmp_path / "saved.cgl"
    saved.write_text(_translate(tmp_path, operand=operand, accuracy=accuracy))
    assert translate(str(saved), backend=target, format_output=False) == generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-fno-fast-math",))


def test_log10_calls_multiply_the_helper_result_once():
    generated = _convert(
        SOURCE,
        binary32_log2_operand_profile="preserve-subnormals",
        binary32_log2_accuracy_profile="portable-finite",
    )
    product = f" * {LOG10_OF_TWO})"
    assert "(__crossgl_metal_log2_float(float(value))" + product in generated
    assert "(__crossgl_metal_log2_float2(vec2(vec2(x, negative)))" + product in (
        generated
    )
    assert generated.count(product) == 4
    assert "log10(" not in generated
    # The operand of the four-lane call keeps its single evaluation.
    assert generated.count("record(count, x)") == 1


def test_log10_without_profiles_is_unchanged(tmp_path):
    generated = _translate(tmp_path)
    assert "__crossgl_metal_log2" not in generated and LOG10_OF_TWO not in generated
    assert generated == _translate(
        tmp_path,
        operand=None,
        accuracy=None,
    )
    assert generated == translate(
        str(tmp_path / "log10.metal"),
        backend="crossgl",
        format_output=False,
        source_options={"binary32_log_profile": "flush-subnormals"},
    )


@pytest.mark.parametrize("operand", OPERANDS[1:])
@pytest.mark.parametrize("accuracy", ACCURACY)
def test_log10_preserves_fast_calls_and_user_overloads(tmp_path, operand, accuracy):
    source = """
float log10(float x) { return x + 3.0f; }
float user(float x) { return ::log10(x); }
float fast_mode(float x) { return metal::fast::log10(x); }
float explicit_mode(float x) { return metal::precise::log10(x); }
float default_mode(float x) { return metal::log10(x); }
"""
    generated = _translate(tmp_path, source, operand=operand, accuracy=accuracy)
    assert "return log10__metal_overload_1(x);" in generated
    # Explicit fast calls keep the target's own base-ten logarithm.
    fast = generated.split("float fast_mode(float x) {", 1)[1].split("}", 1)[0]
    assert fast.strip() == "return log10(x);"
    assert (
        generated.count(
            f"return (__crossgl_metal_log2_float(float(x)) * {LOG10_OF_TWO});"
        )
        == 2
    )


@pytest.mark.parametrize("namespace", ("", "precise", "fast"))
def test_log10_keeps_materialized_bfloat_wrappers(namespace):
    source = f"""
typedef bfloat bfloat16_t;
namespace metal {{
{('namespace ' + namespace + ' {') if namespace else ''}
METAL_FUNC bfloat16_t log10(bfloat16_t x) {{ return bfloat16_t(__metal_log10(float(x))); }}
{'}' if namespace else ''}
}}
bfloat16_t apply(bfloat16_t x) {{ return metal::{(namespace + '::') if namespace else ''}log10(x); }}
"""
    generated = _convert(
        source,
        binary32_log2_operand_profile="flush-subnormals",
        binary32_log2_accuracy_profile="portable-finite",
    )
    expected = f"bfloat16((__crossgl_metal_log2_float(float(x)) * {LOG10_OF_TWO}))"
    assert (expected in generated) == (namespace != "fast")


def test_log10_rejects_global_initializers_and_doubles(tmp_path):
    with pytest.raises(MetalPreciseMathLoweringError, match="log10.*global"):
        _translate(
            tmp_path,
            "constant float value = metal::log10(1.0f);",
            accuracy="portable-finite",
        )
    with pytest.raises(MetalPreciseMathLoweringError, match="binary32 computation"):
        _translate(
            tmp_path,
            "double f(double x) { return metal::log10(x); }",
            accuracy="portable-finite",
        )


@pytest.mark.parametrize("operand", ("half", "half2", "bfloat"))
@pytest.mark.parametrize("target", TARGETS)
def test_log10_narrow_conversions_compile(tmp_path, operand, target):
    name = "metal::precise::log10" if operand == "bfloat" else "metal::log10"
    component = "value.x" if operand == "half2" else "value"
    source = f"""#include <metal_stdlib>
using namespace metal;
using Narrow = {operand};
Narrow f(Narrow x) {{ return {name}(x); }}
kernel void logarithms(device float* results [[buffer(0)]]) {{
    Narrow value = f(Narrow(2.0f));
    results[0] = float({component});
}}
"""
    generated = _translate(tmp_path, source, target, OPERANDS[1], ACCURACY[1])
    assert "metal_log2_float" in generated
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


@lru_cache(maxsize=None)
def _oracle(word, operand):
    magnitude = word & 0x7FFFFFFF
    if magnitude == 0 or (operand == "flush-subnormals" and magnitude < 0x800000):
        return 0xFF800000
    if magnitude > 0x7F800000 or word >> 31:
        return 0x7FC00000
    if magnitude == 0x7F800000:
        return magnitude
    references = []
    for precision in (120, 180):
        with localcontext() as context:
            context.prec = precision
            references.append(
                _round_decimal(Decimal.from_float(_float(word)).ln() / Decimal(10).ln())
            )
    assert references[0] == references[1], hex(word)
    return references[0]


def _expected(inputs, operand):
    values = list(GUARD)
    for word in inputs:
        values.extend(_oracle(word ^ (sign << 31), operand) for sign in LANE_SIGNS)
        values.extend((1, word))
    return values + GUARD


def _check_value(got, want):
    magnitude = want & 0x7FFFFFFF
    if magnitude > 0x7F800000:
        assert got & 0x7FFFFFFF > 0x7F800000, "NaN classification"
        return 0
    if magnitude in (0, 0x7F800000):
        assert got == want, "zero or infinity"
        return 0
    assert got & 0x7FFFFFFF < 0x7F800000, "finite logarithm"
    assert got >> 31 == want >> 31, "logarithm sign"
    distance = abs(got - want)
    assert distance <= MAXIMUM_ULPS, (hex(got), hex(want), distance)
    return distance


def _check(actual, expected):
    assert len(actual) == len(expected), "output size"
    assert actual[:4] == actual[-4:] == GUARD, "output guards"
    maximum = 0
    for offset in range(4, len(actual) - 4, FIELDS):
        for lane in range(10):
            maximum = max(
                maximum, _check_value(actual[offset + lane], expected[offset + lane])
            )
        assert actual[offset + 10] == 1, "operand evaluation count"
        assert actual[offset + 11] == expected[offset + 11], "input bit preservation"
    return maximum


def test_log10_oracle_and_boundaries():
    assert _oracle(0x3F800000, OPERANDS[1]) == 0
    assert _oracle(0x41200000, OPERANDS[1]) == 0x3F800000  # log10(10) == 1
    assert _oracle(0x447A0000, OPERANDS[1]) == 0x40400000  # log10(1000) == 3
    assert _oracle(0x80000001, OPERANDS[2]) == 0xFF800000
    assert _oracle(0xBF800000, OPERANDS[1]) == 0x7FC00000
    assert _oracle(1, OPERANDS[1]) == _round_decimal(
        Decimal(-149) * Decimal(2).ln() / Decimal(10).ln()
    )


@pytest.mark.parametrize("fault", ("value", "input", "counter", "guard", "nan"))
def test_log10_verifier_rejects_corruption(fault):
    expected = _expected([0x3F7FFFFF, 0x7FC12345, 0x41200000], OPERANDS[1])
    actual = list(expected)
    index = {"value": 4, "input": 15, "counter": 14, "guard": 0, "nan": 16}[fault]
    actual[index] = (actual[index] + MAXIMUM_ULPS + 1) if fault == "value" else 0
    with pytest.raises(AssertionError):
        _check(actual, expected)


@pytest.mark.parametrize("operand", OPERANDS[1:])
def test_log10_profiles_execute(tmp_path, monkeypatch, operand):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native logarithm profiles")
    from tests.test_translator import test_fused_math as native

    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    monkeypatch.setattr(
        native, "_compile", partial(_compile, metal_compile_flags=("-fno-fast-math",))
    )
    inputs = _inputs()
    expected = _expected(inputs, operand)
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    shader = _translate(tmp_path, target=target, operand=operand, accuracy=ACCURACY[1])
    actual, evidence = native._dispatch(
        tmp_path,
        target,
        shader,
        [(word,) for word in inputs],
        len(expected),
        entry="logarithms" if target == "metal" else None,
        initial_output=GUARD + [0xDEADBEEF] * (FIELDS * len(inputs)) + GUARD,
    )
    maximum = _check(actual, expected)
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "operandProfile": operand,
                "accuracyProfile": ACCURACY[1],
                "inputCount": len(inputs),
                "oracle": "120/180-digit Decimal ln(x)/ln(10), nearest-even binary32",
                "maximumAllowedUlps": MAXIMUM_ULPS,
                "maximumFiniteUlps": maximum,
                **evidence,
            },
            indent=2,
        )
    )


def test_ci_requires_log10_profiles_in_existing_native_step():
    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned unary arithmetic"
    )
    selector = "tests/test_translator/test_metal_log10.py::test_log10_profiles_execute"
    assert workflow.count(selector) == 1 and selector in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert_native_step_prerequisites(step, job="portable-host", pinned=True)
    assert "continue-on-error" not in step
