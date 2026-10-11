"""Precise binary32 logarithms with explicit source underflow policy."""

import json
import math
import os
import random
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
from tests.test_translator.test_metal_precise_trig import (
    LANE_SIGNS,
    _bits,
    _float,
    _round_decimal,
)

PROFILES = ("preserve-subnormals", "flush-subnormals")
REQUIRE_ENV = "CROSTL_REQUIRE_METAL_PRECISE_LOG"
GUARD = [0x35A5B6C7] * 4
FIELDS = 12
SOURCE = """#include <metal_stdlib>
using namespace metal;
using Scalar = float;
using Quad = float4;
float record(thread uint& count, float value) { count += 1; return value; }
Scalar evaluate(Scalar value) { return metal::precise::log(value); }
kernel void logarithms(const device uint* values [[buffer(0)]],
                      device uint* results [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    float x = as_type<float>(values[i]);
    float negative = as_type<float>(values[i] ^ 0x80000000u);
    uint count = 0;
    float scalar = evaluate(x);
    float2 pair = metal::precise::log(float2(x, negative));
    float3 triple = precise::log(float3(x, negative, x));
    Quad quad = metal::precise::log(Quad(record(count, x), negative, x, negative));
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


def _translate(tmp_path, source=SOURCE, target="crossgl", profile=None):
    path = tmp_path / "log.metal"
    path.write_text(source, encoding="utf-8")
    return translate(
        str(path),
        backend=target,
        format_output=False,
        source_options={"binary32_log_profile": profile},
    )


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_precise_log_helpers_compile(tmp_path, target, profile):
    generated = _translate(tmp_path, target=target, profile=profile)
    for suffix in ("", "2", "3", "4"):
        assert f"metal_precise_log_float{suffix}(" in generated
    assert "return log(" not in generated
    assert "double" not in generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-fno-fast-math",))


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_precise_log_profile_survives_saved_crossgl(tmp_path, profile, target):
    intermediate = tmp_path / "saved.cgl"
    intermediate.write_text(_translate(tmp_path, profile=profile), encoding="utf-8")
    restored = translate(str(intermediate), backend=target, format_output=False)
    assert restored == _translate(tmp_path, target=target, profile=profile)


def test_precise_log_default_preserves_subnormals(tmp_path):
    assert _translate(tmp_path) == _translate(tmp_path, profile=PROFILES[0])


def test_precise_log_keeps_other_modes_and_source_functions(tmp_path):
    generated = _translate(
        tmp_path,
        """
        float log(float x) { return x + 3.0f; }
        float explicit_mode(float x) { return metal::precise::log(x); }
        float imported_mode(float x) { return precise::log(x); }
        float default_mode(float x) { return metal::log(x); }
        float fast_mode(float x) { return metal::fast::log(x); }
        float user_mode(float x) { return ::log(x); }
    """,
    )
    assert generated.count("return __crossgl_metal_precise_log_float(x);") == 2
    assert generated.count("return log(x);") == 2
    assert "return log__metal_overload_1(x);" in generated


def test_precise_log_helpers_reset_and_avoid_source_names():
    converter = MetalToCrossGLConverter(binary32_log_profile=PROFILES[1])
    source = """
        float __crossgl_metal_precise_log_float(float x) { return x; }
        float2 __crossgl_metal_precise_log_float2(float2 x) { return x; }
        float2 evaluate(float2 x) { return precise::log(x); }
    """
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "float __crossgl_metal_precise_log_float_(float value)" in generated
    assert "return __crossgl_metal_precise_log_float2_(x);" in generated
    assert "if (true && magnitude < 0x00800000u)" in generated
    source = "float evaluate(float x) { return metal::log(x); }"
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "__crossgl_metal_precise_log" not in generated
    assert converter.binary32_log_profile == PROFILES[1]


@pytest.mark.parametrize("profile", (True, 1, "", "rne-flush", [], {}))
def test_log_profile_rejects_invalid_configuration(profile):
    with pytest.raises(ValueError, match="binary32_log_profile"):
        MetalToCrossGLConverter(binary32_log_profile=profile)


@pytest.mark.parametrize("operand", ("int", "double", "float8", "half2", "Payload"))
def test_precise_log_diagnoses_unsupported_operands(tmp_path, operand):
    source = (
        "struct Payload { float x; };\n"
        + f"{operand} f({operand} x) {{ return precise::log(x); }}"
    )
    with pytest.raises(MetalPreciseMathLoweringError) as error:
        _translate(tmp_path, source)
    assert error.value.operation == "log"
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-precise-math-unsupported"
    )


@pytest.mark.parametrize("arguments", ("", "x, x"))
def test_precise_log_requires_one_operand(tmp_path, arguments):
    with pytest.raises(MetalPreciseMathLoweringError, match="exactly one operand"):
        _translate(
            tmp_path, f"float f(float x) {{ return precise::log({arguments}); }}"
        )


def test_precise_log_diagnoses_global_runtime_initialization(tmp_path):
    with pytest.raises(MetalPreciseMathLoweringError, match="global initializers"):
        _translate(tmp_path, "constant float value = precise::log(1.0f);")


def test_log_report_retains_resolved_profile_and_rejects_mutation(tmp_path):
    (tmp_path / "log.metal").write_text(SOURCE)
    (tmp_path / "crosstl.toml").write_text("""[project]
targets = ["directx", "opengl"]
[project.source_options.metal]
binary32_log_profile = "preserve-subnormals"
[project.source_options.metal.target_options.opengl.source_patterns."log.metal"]
binary32_log_profile = "flush-subnormals"
""")
    report = translate_project(load_project_config(tmp_path), format_output=False)
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 2, data["diagnostics"]
    expected = dict(zip(("directx", "opengl"), PROFILES))
    assert {
        a["target"]: a["provenance"]["binary32LogProfile"] for a in data["artifacts"]
    } == expected
    path = tmp_path / "report.json"
    report.write_json(path)
    assert validate_project_report(path)["success"]
    manifest = build_runtime_artifact_manifest(path)
    assert manifest["success"], manifest
    assert {
        a["target"]: a["provenance"]["binary32LogProfile"]
        for a in manifest["artifacts"]
    } == expected
    manifest_path = tmp_path / "artifacts.json"
    manifest_path.write_text(json.dumps(manifest))
    assert build_runtime_package(manifest_path, tmp_path / "package")["success"]
    for invalid in (None, PROFILES[1], "rne-flush", False):
        changed = deepcopy(data)
        if invalid is None:
            changed["artifacts"][0]["provenance"].pop("binary32LogProfile")
        else:
            changed["artifacts"][0]["provenance"]["binary32LogProfile"] = invalid
        path.write_text(json.dumps(changed))
        validation = validate_project_report(path)
        assert not validation["success"]
        assert "binary32LogProfile" in json.dumps(validation["diagnostics"])


@lru_cache(maxsize=None)
def _oracle(word, profile):
    magnitude = word & 0x7FFFFFFF
    if profile == "flush-subnormals" and magnitude < 0x800000:
        return 0xFF800000
    if magnitude == 0:
        return 0xFF800000
    if magnitude > 0x7F800000 or word & 0x80000000:
        return 0x7FC00000
    if magnitude == 0x7F800000:
        return 0x7F800000
    with localcontext() as context:
        context.prec = 120
        return _round_decimal(Decimal.from_float(_float(word)).ln())


def _inputs():
    words = set(range(0x3F800000 - 4096, 0x3F800000 + 4097))
    for center in (0x3FB504F3, 0x3F3504F3, 0x00800000):
        words.update(range(center - 32, center + 33))
    rng = random.Random(2123)
    for exponent in range(256):
        for mantissa in (0, 1, 0x3FFFFF, 0x7FFFFE, 0x7FFFFF, rng.getrandbits(23)):
            words.add(exponent << 23 | mantissa)
    words.update(rng.getrandbits(31) for _ in range(4096))
    return sorted(words | {word | 0x80000000 for word in words})


def _expected(inputs, profile):
    output = list(GUARD)
    for word in inputs:
        output.extend(_oracle(word ^ (sign << 31), profile) for sign in LANE_SIGNS)
        output.extend((1, word))
    return output + GUARD


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
    assert distance <= 4, (hex(got), hex(want), distance)
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


def test_precise_log_oracle_and_coverage():
    words = _inputs()
    assert len(words) < 65536
    assert {(word >> 23) & 255 for word in words} == set(range(256))
    assert all(word ^ 0x80000000 in words for word in words)
    for word in words:
        value = _float(word)
        if value > 0 and math.isfinite(value):
            assert _oracle(word, PROFILES[0]) == _bits(math.log(value))
    assert _oracle(0x3F7FFFFF, PROFILES[0]) == 0xB3800000
    assert _oracle(0x3F800000, PROFILES[0]) == 0
    assert _oracle(1, PROFILES[0]) == _bits(math.log(2.0**-149))
    for word in (1, 0x80000001, 0x7FFFFF, 0x807FFFFF):
        assert _oracle(word, PROFILES[1]) == 0xFF800000


@pytest.mark.parametrize(
    "fault",
    (
        "size",
        "first-guard",
        "last-guard",
        "count",
        "input",
        "value",
        "nan",
        "infinity",
        "zero",
        "flush",
    ),
)
def test_precise_log_verifier_rejects_corruption(fault):
    expected = _expected(
        [0x3F7FFFFF, 0x7FC12345, 0x7F800000, 0x3F800000, 1], PROFILES[0]
    )
    actual = list(expected)
    if fault == "size":
        actual.pop()
    elif fault == "first-guard":
        actual[0] ^= 1
    elif fault == "last-guard":
        actual[-1] ^= 1
    elif fault == "count":
        actual[14] = 2
    elif fault == "input":
        actual[15] ^= 1
    elif fault == "value":
        actual[4] += 5
    else:
        index = {"nan": 1, "infinity": 2, "zero": 3, "flush": 4}[fault]
        actual[4 + FIELDS * index] = 0xFF800000 if fault == "flush" else 0x3F800000
    with pytest.raises(AssertionError):
        _check(actual, expected)


@pytest.mark.parametrize("profile", PROFILES)
def test_precise_log_executes(tmp_path, monkeypatch, profile):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native logarithms")
    from tests.test_translator import test_fused_math as native

    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    monkeypatch.setattr(
        native, "_compile", partial(_compile, metal_compile_flags=("-fno-fast-math",))
    )
    inputs = _inputs()
    expected = _expected(inputs, profile)
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    variants = [("generated", _translate(tmp_path, target=target, profile=profile))]
    if target == "metal" and profile == "flush-subnormals":
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
                "profile": profile,
                "inputCount": len(inputs),
                "oracle": "120-digit Decimal ln, nearest-even binary32",
                "maximumAllowedUlps": 4,
                "originalMetalControl": (
                    "Explicit flush-subnormals source policy; no output normalization."
                ),
                "records": records,
            },
            indent=2,
        )
    )


def _narrow_source(operand):
    return """#include <metal_stdlib>
using namespace metal;
using Narrow = $operand;
Narrow record(thread uint& count, Narrow value) { count += 1; return value; }
kernel void logarithms(const device uint* values [[buffer(0)]],
                      device uint* results [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    Narrow value = as_type<$operand>(ushort(values[i]));
    uint count = 0;
    auto implicit_value = metal::precise::log(record(count, value));
    auto explicit_value = metal::precise::log(float(value));
    results[4u + 3u*i] = as_type<uint>(implicit_value);
    results[5u + 3u*i] = as_type<uint>(explicit_value);
    results[6u + 3u*i] = count;
}
""".replace("$operand", operand)


@pytest.mark.parametrize("operand", ("half", "bfloat"))
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_precise_log_narrow_alias_compiles(tmp_path, operand, target):
    generated = _translate(tmp_path, _narrow_source(operand), target)
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


@pytest.mark.parametrize("operand", ("half", "bfloat"))
def test_precise_log_promotes_narrow_alias_once(tmp_path, monkeypatch, operand):
    source = _narrow_source(operand)
    canonical = _translate(tmp_path, source)
    assert "__crossgl_metal_precise_log_float(record(count, value))" in canonical
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native logarithm promotion")
    from tests.test_translator import test_fused_math as native

    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    monkeypatch.setattr(
        native,
        "_compile",
        partial(
            _compile,
            directx_compile_flags=("-enable-16bit-types",),
            metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
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
    expected = []
    for word in inputs:
        promoted = (
            _bits(struct.unpack("<e", struct.pack("<H", word))[0])
            if operand == "half"
            else word << 16
        )
        expected.append(_oracle(promoted, PROFILES[0]))
    generated = _translate(tmp_path, source, target)
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
        assert implicit == explicit, "implicit and explicit promotion"
        assert count == 1, "operand evaluation count"
        maximum = max(maximum, _check_value(implicit, want))
    (tmp_path / "evidence.json").write_text(
        json.dumps({**evidence, "maximumFiniteUlps": maximum}, indent=2)
    )


def test_ci_requires_logarithm_in_existing_native_step():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned unary arithmetic"
    )
    for name in (
        "test_precise_log_executes",
        "test_precise_log_promotes_narrow_alias_once",
    ):
        selector = f"tests/test_translator/test_metal_precise_log.py::{name}"
        assert workflow.count(selector) == 1 and selector in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "--timeout-seconds 300" in step
    assert "continue-on-error" not in step and "if:" not in step
