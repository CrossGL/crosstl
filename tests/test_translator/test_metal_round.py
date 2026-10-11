"""Preserve Metal halfway-away rounding through the portable representation."""

import json
import math
import os
import random
import struct
import sys
from decimal import ROUND_HALF_UP, Decimal
from functools import partial
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalBuiltinResultTypeResolutionError,
    MetalRoundLoweringError,
    MetalToCrossGLConverter,
)
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.project import load_project_config, translate_project
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_ROUND"
GUARD = [0x58A5B6C7] * 4
FIELDS = 14
LANE_SIGNS = (0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0)
SOURCE = """#include <metal_stdlib>
using namespace metal;
using Scalar = float;
using Quad = float4;
Scalar record(thread uint& count, Scalar x) { count += 1; return x; }
Scalar evaluate(Scalar x) { return metal::round(x); }
kernel void rounding(const device uint* values [[buffer(0)]],
                     device uint* results [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    Scalar x = as_type<float>(values[i]);
    float negative = as_type<float>(values[i] ^ 0x80000000u);
    uint count = 0;
    float2 pair = metal::round(float2(x, negative));
    float3 triple = round(float3(x, negative, x));
    Quad quad = metal::round(Quad(record(count, x), negative, x, negative));
    results[4u + 14u*i] = as_type<uint>(round(x));
    results[5u + 14u*i] = as_type<uint>(evaluate(x));
    results[6u + 14u*i] = as_type<uint>(pair.x);
    results[7u + 14u*i] = as_type<uint>(pair.y);
    results[8u + 14u*i] = as_type<uint>(triple.x);
    results[9u + 14u*i] = as_type<uint>(triple.y);
    results[10u + 14u*i] = as_type<uint>(triple.z);
    results[11u + 14u*i] = as_type<uint>(quad.x);
    results[12u + 14u*i] = as_type<uint>(quad.y);
    results[13u + 14u*i] = as_type<uint>(quad.z);
    results[14u + 14u*i] = as_type<uint>(quad.w);
    results[15u + 14u*i] = as_type<uint>(round(metal::round(x)));
    results[16u + 14u*i] = count;
    results[17u + 14u*i] = values[i];
}
"""


def _translate(tmp_path, source=SOURCE, target="crossgl"):
    path = tmp_path / "round.metal"
    path.write_text(source)
    return translate(str(path), backend=target, format_output=False)


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_round_helpers_compile(tmp_path, target):
    generated = _translate(tmp_path, target=target)
    for suffix in ("", "2", "3", "4"):
        assert f"metal_round_float{suffix}(" in generated
    assert "return round(" not in generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-fno-fast-math",))


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_round_survives_saved_crossgl(tmp_path, target):
    path = tmp_path / "saved.cgl"
    path.write_text(_translate(tmp_path))
    assert translate(str(path), backend=target, format_output=False) == _translate(
        tmp_path, target=target
    )


def test_round_keeps_user_functions_and_rint(tmp_path):
    generated = _translate(
        tmp_path,
        """
float round(float x) { return x + 3.0f; }
namespace custom { float round(float x) { return x - 3.0f; } }
float builtin_value(float x) { return metal::round(x); }
float user_value(float x) { return ::round(x); }
float custom_value(float x) { return custom::round(x); }
float even_value(float x) { return metal::rint(x); }
""",
    )
    assert "return __crossgl_metal_round_float(float(x));" in generated
    assert "return round__metal_overload_1(x);" in generated
    assert "return round__metal_overload_2(x);" in generated
    assert "return rint(x);" in generated


def test_hlsl_round_is_not_metal_round(tmp_path):
    path = tmp_path / "source.hlsl"
    path.write_text("float evaluate(float x) { return round(x); }")
    generated = translate(str(path), backend="crossgl", format_output=False)
    assert "round(x)" in generated
    assert "metal_round" not in generated


def test_round_helpers_reset_and_avoid_source_names():
    converter = MetalToCrossGLConverter()
    source = """
float __crossgl_metal_round_float(float x) { return x; }
float2 __crossgl_metal_round_float2(float2 x) { return x; }
float2 evaluate(float2 x) { return metal::round(x); }
"""
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "float __crossgl_metal_round_float_(float value)" in generated
    assert "return __crossgl_metal_round_float2_(vec2(x));" in generated
    generated = converter.generate(
        MetalParser(
            MetalLexer("float f(float x) { return rint(x); }").tokenize()
        ).parse()
    )
    assert "metal_round" not in generated


@pytest.mark.parametrize("operand", ("double", "double2"))
def test_round_diagnoses_unrepresentable_types(tmp_path, operand):
    with pytest.raises(MetalRoundLoweringError) as error:
        _translate(tmp_path, f"{operand} f({operand} x) {{ return round(x); }}")
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-round-unsupported"
    )


@pytest.mark.parametrize("operand", ("float8", "int", "Payload"))
def test_round_rejects_invalid_builtin_operands(tmp_path, operand):
    with pytest.raises(MetalBuiltinResultTypeResolutionError):
        _translate(
            tmp_path,
            f"struct Payload {{ float x; }}; {operand} f({operand} x) {{ return round(x); }}",
        )


def test_round_diagnoses_runtime_global_initializer(tmp_path):
    with pytest.raises(MetalRoundLoweringError, match="global initializers"):
        _translate(tmp_path, "constant float x = round(0.5f);")


def test_round_project_report_rejects_unsupported_lowering(tmp_path):
    (tmp_path / "source.metal").write_text("double f(double x) { return round(x); }")
    (tmp_path / "crosstl.toml").write_text(
        '[project]\ntargets = ["metal", "directx", "opengl"]\n'
    )
    report = translate_project(
        load_project_config(tmp_path), format_output=False
    ).to_json()
    assert report["summary"]["failedCount"] == 3
    assert report["summary"]["translatedCount"] == 0
    assert len(report["artifacts"]) == 3
    assert all(item["status"] == "failed" for item in report["artifacts"])
    assert all(not (tmp_path / item["path"]).exists() for item in report["artifacts"])
    errors = [item for item in report["diagnostics"] if item["severity"] == "error"]
    assert len(errors) == 3
    assert all(
        item["code"] == "project.translate.metal-round-unsupported" for item in errors
    )


@pytest.mark.parametrize("operand", ("half", "half2", "half3", "half4"))
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_round_preserves_half_result_type(tmp_path, operand, target):
    source = f"""#include <metal_stdlib>
using namespace metal;
using Narrow = {operand};
kernel void rounding(device float* results [[buffer(0)]], uint i [[thread_position_in_grid]]) {{
    Narrow value = Narrow(float(i) + 0.5f);
    auto result = metal::round(value);
    results[i] = float(result{'.x' if operand != 'half' else ''});
}}
"""
    canonical = _translate(tmp_path, source)
    mapped = "float16" if operand == "half" else "f16vec" + operand[-1]
    assert f"{mapped} result = {mapped}(__crossgl_metal_round_float" in canonical
    _compile(
        _translate(tmp_path, source, target),
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-fno-fast-math",),
    )


def test_round_bfloat_wrapper_retains_narrowing(tmp_path):
    source = """
typedef bfloat Narrow;
namespace metal {
METAL_FUNC Narrow round(Narrow value) { return Narrow(__metal_round(float(value))); }
}
float f(Narrow value) { auto result = metal::round(value); return float(result); }
"""
    canonical = _translate(tmp_path, source)
    assert (
        "bfloat16 result = bfloat16(__crossgl_metal_round_float(float(value)));"
        in canonical
    )
    assert "__metal_round" not in canonical


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _float(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _oracle(word):
    value = _float(word)
    if not math.isfinite(value):
        return word
    # Decimal provides an independent arithmetic reference for the bit helper.
    rounded = Decimal.from_float(value).to_integral_value(rounding=ROUND_HALF_UP)
    return _bits(float(rounded)) if rounded else 0


def _inputs():
    rng = random.Random(2138)
    words = {0, 1, 0x7FFFFF, 0x800000, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC12345}
    for integer in list(range(256)) + [
        (1 << exponent) - 1 for exponent in range(1, 24)
    ]:
        midpoint = _bits(integer + 0.5)
        words.update(range(midpoint - 2, midpoint + 3))
    for exponent in range(256):
        words.update((exponent << 23) | rng.randrange(1 << 23) for _ in range(8))
    return sorted(words | {word | 0x80000000 for word in words})


def _check(actual, expected):
    assert len(actual) == len(expected), "output size"
    assert actual[:4] == actual[-4:] == GUARD, "output guards"
    for index in range(4, len(expected) - 4, FIELDS):
        for lane in range(12):
            got, want = actual[index + lane], expected[index + lane]
            if want & 0x7FFFFFFF > 0x7F800000:
                assert got & 0x7FFFFFFF > 0x7F800000, "NaN classification"
            else:
                assert got == want, ("round result", index, lane, hex(got), hex(want))
        assert actual[index + 12] == 1, "operand evaluation count"
        assert actual[index + 13] == expected[index + 13], "input bits"


def _expected(words):
    expected = list(GUARD)
    for word in words:
        expected.extend(_oracle(word ^ (sign << 31)) for sign in LANE_SIGNS)
        expected.extend((1, word))
    return expected + GUARD


def test_round_oracle_boundaries():
    words = _inputs()
    assert {(word >> 23) & 255 for word in words} == set(range(256))
    for value, wanted in ((0.5, 1), (2.5, 3), (-0.5, -1), (-2.5, -3), (-0.25, 0)):
        assert _oracle(_bits(value)) == _bits(wanted)
    assert _oracle(0x3EFFFFFF) == 0
    assert _oracle(0x4AFFFFFF) == 0x4B000000
    assert _oracle(0x80000000) == 0


@pytest.mark.parametrize(
    "fault", ("size", "guard", "value", "zero", "nan", "count", "input")
)
def test_round_verifier_rejects_corruption(fault):
    expected = _expected([0x3F000000, 0x80000000, 0x7FC12345])
    actual = list(expected)
    if fault == "size":
        actual.pop()
    else:
        index = {
            "guard": 0,
            "value": 4,
            "zero": 4 + FIELDS,
            "nan": 4 + 2 * FIELDS,
            "count": 16,
            "input": 17,
        }[fault]
        actual[index] = 0x80000000 if fault == "zero" else 0
    with pytest.raises(AssertionError):
        _check(actual, expected)


def test_round_executes(tmp_path, monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native rounding")
    from tests.test_translator import test_fused_math as native

    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    monkeypatch.setattr(
        native, "_compile", partial(_compile, metal_compile_flags=("-fno-fast-math",))
    )
    words = _inputs()
    expected = _expected(words)
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    variants = [("generated", _translate(tmp_path, target=target))]
    if target == "metal":
        variants.append(("original", SOURCE))
    records = {}
    for label, source in variants:
        work = tmp_path / label
        work.mkdir()
        actual, evidence = native._dispatch(
            work,
            target,
            source,
            [(word,) for word in words],
            len(expected),
            entry="rounding" if target == "metal" else None,
            initial_output=GUARD + [0xDEADBEEF] * (FIELDS * len(words)) + GUARD,
        )
        _check(actual, expected)
        records[label] = evidence
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "inputCount": len(words),
                "oracle": "Decimal ROUND_HALF_UP, positive zero",
                "nanComparison": "classification-only",
                "records": records,
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("operand", ("half", "bfloat"))
def test_round_narrow_executes(tmp_path, monkeypatch, operand):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native rounding")
    from tests.test_translator import test_fused_math as native

    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    source = """#include <metal_stdlib>
using namespace metal;
using Narrow = $operand;
Narrow record(thread uint& count, Narrow x) { count += 1; return x; }
kernel void rounding(const device uint* values [[buffer(0)]], device uint* results [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    Narrow x = as_type<$operand>(ushort(values[i]));
    uint count = 0;
    auto rounded = round(record(count, x));
    results[4u + 3u*i] = as_type<uint>(float(rounded));
    results[5u + 3u*i] = count;
    results[6u + 3u*i] = values[i];
}
""".replace("$operand", operand)
    monkeypatch.setattr(
        native,
        "_compile",
        partial(
            _compile,
            metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
            directx_compile_flags=("-enable-16bit-types",),
        ),
    )
    words = list(range(65536))
    promoted = [
        (
            _bits(struct.unpack("<e", struct.pack("<H", word))[0])
            if operand == "half"
            else word << 16
        )
        for word in words
    ]
    expected = [_oracle(word) for word in promoted]
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    variants = [("generated", _translate(tmp_path, source, target))]
    if target == "metal":
        variants.append(("original", source))
    records = {}
    for label, variant in variants:
        records[label] = []
        # Keep each dispatch within the OpenGL/DirectX workgroup-count limit.
        for start in (0, 32768):
            batch = words[start : start + 32768]
            work = tmp_path / f"{label}-{start}"
            work.mkdir()
            actual, evidence = native._dispatch(
                work,
                target,
                variant,
                [(word,) for word in batch],
                3 * len(batch) + 8,
                entry="rounding" if target == "metal" else None,
                initial_output=GUARD + [0xDEADBEEF] * (3 * len(batch)) + GUARD,
            )
            assert len(actual) == 3 * len(batch) + 8
            assert actual[:4] == actual[-4:] == GUARD
            for i, want in enumerate(expected[start : start + len(batch)]):
                got, count, word = actual[4 + 3 * i : 7 + 3 * i]
                assert count == 1 and word == batch[i]
                if want & 0x7FFFFFFF > 0x7F800000:
                    assert got & 0x7FFFFFFF > 0x7F800000
                else:
                    assert got == want, (operand, label, hex(word), hex(got), hex(want))
            records[label].append(evidence)
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {"operand": operand, "inputCount": len(words), "records": records}, indent=2
        )
    )


def test_ci_requires_round_in_existing_native_step():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned unary arithmetic"
    )
    selector = "tests/test_translator/test_metal_round.py::test_round_executes"
    assert workflow.count(selector) == 1 and selector in step
    selector = "tests/test_translator/test_metal_round.py::test_round_narrow_executes"
    assert workflow.count(selector) == 1 and selector in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "--timeout-seconds 300" in step
    assert "continue-on-error" not in step and "if:" not in step
