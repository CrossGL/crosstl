"""Base-two logarithm policies, source ownership and native execution."""

import json
import math
import os
import struct
import sys
from copy import deepcopy
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
from crosstl.project import (
    build_runtime_artifact_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
    validate_project_report,
)
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_log import (
    FIELDS,
    GUARD,
    LANE_SIGNS,
    _bits,
    _check,
    _check_value,
    _float,
    _inputs,
    _narrow_source,
    _round_decimal,
)

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_LOG2"
OPERANDS = (None, "preserve-subnormals", "flush-subnormals")
ACCURACY = (None, "portable-finite")
SOURCE = """#include <metal_stdlib>
using namespace metal;
using Scalar = float;
using Quad = float4;
float record(thread uint& count, float value) { count += 1; return value; }
Scalar evaluate(Scalar value) { return metal::log2(value); }
kernel void logarithms(const device uint* values [[buffer(0)]],
                      device uint* results [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    float x = as_type<float>(values[i]);
    float negative = as_type<float>(values[i] ^ 0x80000000u);
    uint count = 0;
    float scalar = evaluate(x);
    float2 pair = metal::precise::log2(float2(x, negative));
    float3 triple = precise::log2(float3(x, negative, x));
    Quad quad = metal::log2(Quad(record(count, x), negative, x, negative));
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
    path = tmp_path / "log2.metal"
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


@pytest.mark.parametrize("operand", OPERANDS)
@pytest.mark.parametrize("accuracy", ACCURACY)
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_log2_profiles_compile_and_survive_saved_crossgl(
    tmp_path, target, operand, accuracy
):
    generated = _translate(tmp_path, target=target, operand=operand, accuracy=accuracy)
    assert ("metal_log2_float" in generated) == (
        operand is not None or accuracy is not None
    )
    saved = tmp_path / "saved.cgl"
    saved.write_text(_translate(tmp_path, operand=operand, accuracy=accuracy))
    assert translate(str(saved), backend=target, format_output=False) == generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-fno-fast-math",))


def test_log2_profiles_do_not_change_natural_logarithms(tmp_path):
    source = "float f(float x) { return metal::precise::log(x) + metal::log(x); }"
    assert _translate(tmp_path, source) == _translate(
        tmp_path, source, operand=OPERANDS[2], accuracy=ACCURACY[1]
    )
    source = "float f(float x) { return metal::log2(x); }"
    default = _translate(tmp_path, source)
    assert "__crossgl_metal_log2" not in default
    assert (
        translate(
            str(tmp_path / "log2.metal"),
            backend="crossgl",
            format_output=False,
            source_options={"binary32_log_profile": "flush-subnormals"},
        )
        == default
    )


@pytest.mark.parametrize("operand", OPERANDS)
@pytest.mark.parametrize("accuracy", ACCURACY)
def test_log2_preserves_fast_calls_and_user_overloads(tmp_path, operand, accuracy):
    source = """
float log2(float x) { return x + 3.0f; }
int log2(int x) { return x + 7; }
float user(float x) { return ::log2(x); }
int integer(int x) { return ::log2(x); }
float fast_mode(float x) { return metal::fast::log2(x); }
float explicit_mode(float x) { return metal::precise::log2(x); }
float default_mode(float x) { return metal::log2(x); }
"""
    generated = _translate(tmp_path, source, operand=operand, accuracy=accuracy)
    assert "return log2__metal_overload_1(x);" in generated
    assert "return log2__metal_overload_2(x);" in generated
    assert "return log2(x);" in generated
    assert generated.count("return __crossgl_metal_log2_float(float(x));") == (
        2 if operand or accuracy else 0
    )


@pytest.mark.parametrize("namespace", ("", "precise", "fast"))
@pytest.mark.parametrize("materialized", (False, True))
def test_log2_preserves_bfloat_wrapper_ownership(namespace, materialized):
    qualifier = "METAL_FUNC" if materialized else ""
    body = "__metal_log2(float(x))" if materialized else "float(x) + 3.0f"
    source = f"""
typedef bfloat bfloat16_t;
namespace metal {{
{('namespace ' + namespace + ' {') if namespace else ''}
{qualifier} bfloat16_t log2(bfloat16_t x) {{ return bfloat16_t({body}); }}
{'}' if namespace else ''}
}}
bfloat16_t apply(bfloat16_t x) {{ return metal::{(namespace + '::') if namespace else ''}log2(x); }}
"""
    converter = MetalToCrossGLConverter(
        binary32_log2_operand_profile="flush-subnormals",
        binary32_log2_accuracy_profile="portable-finite",
    )
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert ("__crossgl_metal_log2_float" in generated) == (
        materialized and namespace != "fast"
    )
    if materialized and namespace != "fast":
        assert "bfloat16(__crossgl_metal_log2_float(float(x)))" in generated


def test_log2_helpers_reset_and_avoid_source_names():
    converter = MetalToCrossGLConverter(
        binary32_log2_accuracy_profile="portable-finite"
    )
    source = """
float __crossgl_metal_log2_float(float x) { return x; }
float2 __crossgl_metal_log2_float2(float2 x) { return x; }
float2 f(float2 x) { return metal::log2(x); }
"""
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "float __crossgl_metal_log2_float_(float value)" in generated
    assert "__crossgl_metal_log2_float2_(vec2(x))" in generated
    generated = converter.generate(
        MetalParser(
            MetalLexer("float f(float x) { return metal::fast::log2(x); }").tokenize()
        ).parse()
    )
    assert "__crossgl_metal_log2" not in generated
    assert converter.binary32_log2_accuracy_profile == "portable-finite"


@pytest.mark.parametrize(
    "option", ("binary32_log2_operand_profile", "binary32_log2_accuracy_profile")
)
@pytest.mark.parametrize("invalid", (True, 1, "", "rne-flush", [], {}))
def test_log2_invalid_options_are_rejected(option, invalid):
    with pytest.raises(ValueError, match=option):
        MetalToCrossGLConverter(**{option: invalid})


def test_log2_rejects_global_runtime_initializers(tmp_path):
    with pytest.raises(MetalPreciseMathLoweringError, match="global initializers"):
        _translate(
            tmp_path,
            "constant float value = metal::log2(1.0f);",
            accuracy="portable-finite",
        )


def test_log2_rejects_unrepresentable_profiled_double(tmp_path):
    with pytest.raises(MetalPreciseMathLoweringError, match="binary32 computation"):
        _translate(
            tmp_path,
            "double f(double x) { return metal::log2(x); }",
            accuracy="portable-finite",
        )


@pytest.mark.parametrize("operand", ("half", "half2", "bfloat"))
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_log2_narrow_conversions_compile(tmp_path, operand, target):
    name = "metal::precise::log2" if operand == "bfloat" else "metal::log2"
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
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


def test_log2_report_and_package_retain_resolved_profiles(tmp_path):
    (tmp_path / "log2.metal").write_text(SOURCE)
    (tmp_path / "crosstl.toml").write_text("""[project]
targets = ["directx", "opengl"]
[project.source_options.metal]
binary32_log2_operand_profile = "preserve-subnormals"
binary32_log2_accuracy_profile = "portable-finite"
[project.source_options.metal.target_options.opengl.source_patterns."log2.metal"]
binary32_log2_operand_profile = "flush-subnormals"
""")
    report = translate_project(load_project_config(tmp_path), format_output=False)
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 2, data["diagnostics"]
    path = tmp_path / "report.json"
    report.write_json(path)
    assert validate_project_report(path)["success"]
    manifest = build_runtime_artifact_manifest(path)
    assert manifest["success"], manifest
    for collection in (data["artifacts"], manifest["artifacts"]):
        for artifact in collection:
            assert artifact["provenance"]["binary32Log2OperandProfile"] == (
                "flush-subnormals"
                if artifact["target"] == "opengl"
                else "preserve-subnormals"
            )
            assert (
                artifact["provenance"]["binary32Log2AccuracyProfile"]
                == "portable-finite"
            )
    manifest_path = tmp_path / "artifacts.json"
    manifest_path.write_text(json.dumps(manifest))
    assert build_runtime_package(manifest_path, tmp_path / "package")["success"]
    for field in ("binary32Log2OperandProfile", "binary32Log2AccuracyProfile"):
        for invalid in (None, "wrong", False, "flush-subnormals"):
            changed = deepcopy(data)
            if invalid is None:
                changed["artifacts"][0]["provenance"].pop(field)
            else:
                changed["artifacts"][0]["provenance"][field] = invalid
            path.write_text(json.dumps(changed))
            result = validate_project_report(path)
            assert not result["success"]
            assert field in json.dumps(result["diagnostics"])


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
                _round_decimal(Decimal.from_float(_float(word)).ln() / Decimal(2).ln())
            )
    assert references[0] == references[1], hex(word)
    return references[0]


def _expected(inputs, operand):
    values = list(GUARD)
    for word in inputs:
        values.extend(_oracle(word ^ (sign << 31), operand) for sign in LANE_SIGNS)
        values.extend((1, word))
    return values + GUARD


def test_log2_oracle_and_boundary_coverage():
    words = _inputs()
    assert len(words) == 28024
    assert {(word >> 23) & 255 for word in words} == set(range(256))
    for word in words:
        value = _float(word)
        if value > 0 and math.isfinite(value):
            assert _oracle(word, OPERANDS[1]) == _bits(math.log2(value))
    assert _oracle(0x3F7FFFFF, OPERANDS[1]) == 0xB3B8AA3C
    assert _oracle(0x3F800000, OPERANDS[1]) == 0
    assert _oracle(1, OPERANDS[1]) == _bits(-149.0)
    assert _oracle(0x80000001, OPERANDS[2]) == 0xFF800000


@pytest.mark.parametrize(
    "fault", ("value", "input", "counter", "guard", "nan", "flush")
)
def test_log2_verifier_rejects_corruption(fault):
    expected = _expected([0x3F7FFFFF, 0x7FC12345, 1], OPERANDS[1])
    actual = list(expected)
    index = {
        "value": 4,
        "input": 15,
        "counter": 14,
        "guard": 0,
        "nan": 16,
        "flush": 28,
    }[fault]
    actual[index] = (actual[index] + 5) if fault == "value" else 0
    with pytest.raises(AssertionError):
        _check(actual, expected)


@pytest.mark.parametrize("operand", OPERANDS[1:])
@pytest.mark.parametrize("accuracy", ACCURACY)
def test_log2_profiles_execute(tmp_path, monkeypatch, operand, accuracy):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native logarithm profiles")
    from tests.test_translator import test_fused_math as native

    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    monkeypatch.setattr(
        native, "_compile", partial(_compile, metal_compile_flags=("-fno-fast-math",))
    )
    # Operand-only controls use exact powers of two; finite-accuracy profiles
    # additionally cover dense cancellation, normalization and exponent cases.
    inputs = (
        _inputs()
        if accuracy
        else sorted(
            {0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC12345}
            | {
                word | sign
                for word in [1 << bit for bit in range(23)] + [
                    exponent << 23 for exponent in range(1, 255)
                ]
                for sign in (0, 0x80000000)
            }
        )
    )
    expected = _expected(inputs, operand)
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    variants = [
        (
            "generated",
            _translate(tmp_path, target=target, operand=operand, accuracy=accuracy),
        )
    ]
    if target == "metal" and operand == "flush-subnormals":
        variants.append(("original", SOURCE))
    records = {}
    for label, source in variants:
        work = tmp_path / label
        work.mkdir()
        actual, evidence = native._dispatch(
            work,
            target,
            source,
            [(word,) for word in inputs],
            len(expected),
            entry="logarithms" if target == "metal" else None,
            initial_output=GUARD + [0xDEADBEEF] * (FIELDS * len(inputs)) + GUARD,
        )
        records[label] = {**evidence, "maximumFiniteUlps": _check(actual, expected)}
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "operandProfile": operand,
                "accuracyProfile": accuracy,
                "inputCount": len(inputs),
                "oracle": "120/180-digit Decimal ln(x)/ln(2), nearest-even binary32",
                "maximumAllowedUlps": 4,
                "originalMetalControl": (
                    "Characterized source device with explicit flush policy; no buffer normalization"
                ),
                "records": records,
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("operand", ("half", "bfloat"))
def test_log2_narrow_promotion_executes_once(tmp_path, monkeypatch, operand):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native logarithm promotion")
    from tests.test_translator import test_fused_math as native

    source = _narrow_source(operand).replace("::log(", "::log2(")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    monkeypatch.setattr(
        native,
        "_compile",
        partial(
            _compile,
            metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
            directx_compile_flags=("-enable-16bit-types",),
        ),
    )
    inputs = [
        0,
        0x8000,
        1,
        0x8001,
        0x3BFF,
        0x3C00,
        0x3C01,
        0x3F7F,
        0x3F80,
        0x3F81,
        0x7BFF,
        0x7C00,
        0x7F80,
        0x7FC1,
        0xFC00,
        0xFF80,
    ]
    promoted = [
        (
            _bits(struct.unpack("<e", struct.pack("<H", word))[0])
            if operand == "half"
            else word << 16
        )
        for word in inputs
    ]
    expected = [_oracle(word, "preserve-subnormals") for word in promoted]
    generated = _translate(
        tmp_path, source, target, "preserve-subnormals", "portable-finite"
    )
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    actual, evidence = native._dispatch(
        tmp_path,
        target,
        generated,
        [(word,) for word in inputs],
        3 * len(inputs) + 8,
        entry="logarithms" if target == "metal" else None,
        initial_output=GUARD + [0xDEADBEEF] * (3 * len(inputs)) + GUARD,
    )
    assert actual[:4] == actual[-4:] == GUARD
    maximum = 0
    for index, want in enumerate(expected):
        implicit, explicit, count = actual[4 + 3 * index : 7 + 3 * index]
        assert implicit == explicit
        assert count == 1
        maximum = max(maximum, _check_value(implicit, want))
    (tmp_path / "evidence.json").write_text(
        json.dumps({**evidence, "maximumFiniteUlps": maximum}, indent=2)
    )


def test_log2_unspecified_operand_policy_keeps_native_subnormals(tmp_path, monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native logarithm controls")
    from tests.test_translator import test_fused_math as native

    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    monkeypatch.setattr(
        native, "_compile", partial(_compile, metal_compile_flags=("-fno-fast-math",))
    )
    inputs = [
        word | sign
        for word in (0, 1, 2, 0x400000, 0x7FFFFF)
        for sign in (0, 0x80000000)
    ]
    results, receipts = {}, {}
    for label, accuracy in (("native", None), ("profiled", "portable-finite")):
        work = tmp_path / label
        work.mkdir()
        shader = _translate(work, target=target, accuracy=accuracy)
        actual, evidence = native._dispatch(
            work,
            target,
            shader,
            [(word,) for word in inputs],
            FIELDS * len(inputs) + 8,
            entry="logarithms" if target == "metal" else None,
            initial_output=GUARD + [0xDEADBEEF] * (FIELDS * len(inputs)) + GUARD,
        )
        results[label], receipts[label] = actual, evidence
    for got, want in zip(results["profiled"], results["native"]):
        assert (
            (got & 0x7FFFFFFF > 0x7F800000)
            if want & 0x7FFFFFFF > 0x7F800000
            else got == want
        )
    for values in results.values():
        assert values[:4] == values[-4:] == GUARD
        for index, word in enumerate(inputs):
            assert values[14 + FIELDS * index : 16 + FIELDS * index] == [1, word]
    (tmp_path / "evidence.json").write_text(json.dumps(receipts, indent=2))


def test_ci_requires_log2_profiles_in_existing_native_step():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned unary arithmetic"
    )
    for name in (
        "test_log2_profiles_execute",
        "test_log2_narrow_promotion_executes_once",
        "test_log2_unspecified_operand_policy_keeps_native_subnormals",
    ):
        selector = f"tests/test_translator/test_metal_log2.py::{name}"
        assert workflow.count(selector) == 1 and selector in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "--timeout-seconds 300" in step
    assert "continue-on-error" not in step and "if:" not in step
