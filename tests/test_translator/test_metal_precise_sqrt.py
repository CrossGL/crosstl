"""Correctly rounded binary32 square roots with explicit operand underflow."""

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
REQUIRE_ENV = "CROSTL_REQUIRE_METAL_PRECISE_SQRT"
GUARD = [0x58A5B6C7] * 4
FIELDS = 13
SOURCE = """#include <metal_stdlib>
using namespace metal;
using Scalar = float;
using Quad = float4;
float record(thread uint& count, float value) { count += 1; return value; }
Scalar evaluate(Scalar value) { return metal::precise::sqrt(value); }
kernel void roots(const device uint* values [[buffer(0)]],
                 device uint* results [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
    float x = as_type<float>(values[i]);
    float negative = as_type<float>(values[i] ^ 0x80000000u);
    float magnitude = as_type<float>(values[i] & 0x7fffffffu);
    uint count = 0;
    float scalar = evaluate(x);
    float2 pair = metal::precise::sqrt(float2(x, negative));
    float3 triple = precise::sqrt(float3(x, negative, x));
    Quad quad = metal::precise::sqrt(Quad(record(count, x), negative, x, negative));
    float composed = precise::sqrt(precise::sqrt(magnitude));
    results[4u + 13u*i] = as_type<uint>(scalar);
    results[5u + 13u*i] = as_type<uint>(pair.x);
    results[6u + 13u*i] = as_type<uint>(pair.y);
    results[7u + 13u*i] = as_type<uint>(triple.x);
    results[8u + 13u*i] = as_type<uint>(triple.y);
    results[9u + 13u*i] = as_type<uint>(triple.z);
    results[10u + 13u*i] = as_type<uint>(quad.x);
    results[11u + 13u*i] = as_type<uint>(quad.y);
    results[12u + 13u*i] = as_type<uint>(quad.z);
    results[13u + 13u*i] = as_type<uint>(quad.w);
    results[14u + 13u*i] = as_type<uint>(composed);
    results[15u + 13u*i] = count;
    results[16u + 13u*i] = as_type<uint>(x);
}
"""


def _translate(tmp_path, source=SOURCE, target="crossgl", profile=None):
    path = tmp_path / "sqrt.metal"
    path.write_text(source, encoding="utf-8")
    return translate(
        str(path),
        backend=target,
        format_output=False,
        source_options={"binary32_sqrt_profile": profile},
    )


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_precise_sqrt_helpers_compile(tmp_path, target, profile):
    generated = _translate(tmp_path, target=target, profile=profile)
    for suffix in ("", "2", "3", "4"):
        assert f"metal_precise_sqrt_float{suffix}(" in generated
    assert "return sqrt(" not in generated
    assert "double" not in generated and "uint64" not in generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-fno-fast-math",))


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_precise_sqrt_profile_survives_saved_crossgl(tmp_path, profile, target):
    path = tmp_path / "saved.cgl"
    path.write_text(_translate(tmp_path, profile=profile), encoding="utf-8")
    restored = translate(str(path), backend=target, format_output=False)
    assert restored == _translate(tmp_path, target=target, profile=profile)


def test_precise_sqrt_default_preserves_subnormals(tmp_path):
    assert _translate(tmp_path) == _translate(tmp_path, profile=PROFILES[0])


def test_precise_sqrt_keeps_other_modes_and_source_functions(tmp_path):
    generated = _translate(
        tmp_path,
        """
        float sqrt(float x) { return x + 3.0f; }
        float explicit_mode(float x) { return metal::precise::sqrt(x); }
        float imported_mode(float x) { return precise::sqrt(x); }
        float default_mode(float x) { return metal::sqrt(x); }
        float fast_mode(float x) { return metal::fast::sqrt(x); }
        float user_mode(float x) { return ::sqrt(x); }
    """,
    )
    assert generated.count("return __crossgl_metal_precise_sqrt_float(x);") == 2
    assert generated.count("return sqrt(x);") == 2
    assert "return sqrt__metal_overload_1(x);" in generated


def test_precise_sqrt_helpers_reset_and_avoid_source_names():
    converter = MetalToCrossGLConverter(binary32_sqrt_profile=PROFILES[1])
    source = """
        float __crossgl_metal_precise_sqrt_float(float x) { return x; }
        float2 __crossgl_metal_precise_sqrt_float2(float2 x) { return x; }
        float2 evaluate(float2 x) { return precise::sqrt(x); }
    """
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "float __crossgl_metal_precise_sqrt_float_(float value)" in generated
    assert "return __crossgl_metal_precise_sqrt_float2_(x);" in generated
    assert "if (true && magnitude < 0x00800000u)" in generated
    source = "float evaluate(float x) { return metal::sqrt(x); }"
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "__crossgl_metal_precise_sqrt" not in generated
    assert converter.binary32_sqrt_profile == PROFILES[1]


@pytest.mark.parametrize("profile", (True, 1, "", "rne-flush", [], {}))
def test_sqrt_profile_rejects_invalid_configuration(profile):
    with pytest.raises(ValueError, match="binary32_sqrt_profile"):
        MetalToCrossGLConverter(binary32_sqrt_profile=profile)


@pytest.mark.parametrize("operand", ("int", "double", "float8", "half2", "Payload"))
def test_precise_sqrt_diagnoses_unsupported_operands(tmp_path, operand):
    source = (
        "struct Payload { float x; };\n"
        + f"{operand} f({operand} x) {{ return precise::sqrt(x); }}"
    )
    with pytest.raises(MetalPreciseMathLoweringError) as error:
        _translate(tmp_path, source)
    assert error.value.operation == "sqrt"
    assert error.value.project_diagnostic_code == (
        "project.translate.metal-precise-math-unsupported"
    )


@pytest.mark.parametrize("arguments", ("", "x, x"))
def test_precise_sqrt_requires_one_operand(tmp_path, arguments):
    with pytest.raises(MetalPreciseMathLoweringError, match="exactly one operand"):
        _translate(
            tmp_path, f"float f(float x) {{ return precise::sqrt({arguments}); }}"
        )


def test_precise_sqrt_diagnoses_global_runtime_initialization(tmp_path):
    with pytest.raises(MetalPreciseMathLoweringError, match="global initializers"):
        _translate(tmp_path, "constant float value = precise::sqrt(1.0f);")


def test_sqrt_report_retains_resolved_profile_and_rejects_mutation(tmp_path):
    (tmp_path / "sqrt.metal").write_text(SOURCE)
    (tmp_path / "crosstl.toml").write_text("""[project]
targets = ["directx", "opengl"]
[project.source_options.metal]
binary32_sqrt_profile = "preserve-subnormals"
[project.source_options.metal.target_options.opengl.source_patterns."sqrt.metal"]
binary32_sqrt_profile = "flush-subnormals"
""")
    report = translate_project(load_project_config(tmp_path), format_output=False)
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 2, data["diagnostics"]
    expected = dict(zip(("directx", "opengl"), PROFILES))
    assert {
        a["target"]: a["provenance"]["binary32SqrtProfile"] for a in data["artifacts"]
    } == expected
    path = tmp_path / "report.json"
    report.write_json(path)
    assert validate_project_report(path)["success"]
    manifest = build_runtime_artifact_manifest(path)
    assert manifest["success"], manifest
    assert {
        a["target"]: a["provenance"]["binary32SqrtProfile"]
        for a in manifest["artifacts"]
    } == expected
    manifest_path = tmp_path / "artifacts.json"
    manifest_path.write_text(json.dumps(manifest))
    assert build_runtime_package(manifest_path, tmp_path / "package")["success"]
    for invalid in (None, PROFILES[1], "rne-flush", False):
        changed = deepcopy(data)
        if invalid is None:
            changed["artifacts"][0]["provenance"].pop("binary32SqrtProfile")
        else:
            changed["artifacts"][0]["provenance"]["binary32SqrtProfile"] = invalid
        path.write_text(json.dumps(changed))
        validation = validate_project_report(path)
        assert not validation["success"]
        assert "binary32SqrtProfile" in json.dumps(validation["diagnostics"])


@lru_cache(maxsize=None)
def _oracle(word, profile):
    magnitude = word & 0x7FFFFFFF
    if magnitude == 0 or (profile == "flush-subnormals" and magnitude < 0x800000):
        return word & 0x80000000
    if magnitude > 0x7F800000 or word & 0x80000000:
        return 0x7FC00000
    if magnitude == 0x7F800000:
        return magnitude
    with localcontext() as context:
        context.prec = 120
        return _round_decimal(Decimal.from_float(_float(word)).sqrt())


def _inputs():
    words = set(range(0x3F800000 - 512, 0x3F800000 + 513))
    rng = random.Random(2124)
    for exponent in range(256):
        for mantissa in (0, 1, 2, 0x3FFFFF, 0x400000, 0x7FFFFE, 0x7FFFFF):
            words.add(exponent << 23 | mantissa)
    words.update(rng.getrandbits(31) for _ in range(4096))
    words.update(range(512))
    words.update(range(0x800000 - 64, 0x800000 + 65))
    # Round inputs immediately around squared output midpoints; these expose
    # one-bit mistakes in either integer-root refinement or final rounding.
    with localcontext() as context:
        context.prec = 120
        for _ in range(1024):
            root = rng.randrange(0x20000000, 0x5F7FFFFF)
            midpoint = (
                Decimal.from_float(_float(root)) + Decimal.from_float(_float(root + 1))
            ) / 2
            center = _round_decimal(midpoint * midpoint)
            words.update((center - 1, center, center + 1))
    return sorted(words | {word | 0x80000000 for word in words})


def _expected(inputs, profile):
    output = list(GUARD)
    for word in inputs:
        output.extend(_oracle(word ^ (sign << 31), profile) for sign in LANE_SIGNS)
        output.append(_oracle(_oracle(word & 0x7FFFFFFF, profile), profile))
        output.extend((1, word))
    return output + GUARD


def _check_value(got, want):
    if want & 0x7FFFFFFF > 0x7F800000:
        assert got & 0x7FFFFFFF > 0x7F800000, "NaN classification"
    else:
        assert got == want, ("correctly rounded square root", hex(got), hex(want))


def _check(actual, expected):
    assert len(actual) == len(expected), "output size"
    assert actual[:4] == actual[-4:] == GUARD, "output guards"
    for offset in range(4, len(actual) - 4, FIELDS):
        for lane in range(11):
            _check_value(actual[offset + lane], expected[offset + lane])
        assert actual[offset + 11] == 1, "operand evaluation count"
        assert actual[offset + 12] == expected[offset + 12], "input bit preservation"


def test_precise_sqrt_oracle_and_coverage():
    words = _inputs()
    assert len(words) < 65536
    assert {(word >> 23) & 255 for word in words} == set(range(256))
    assert all(word ^ 0x80000000 in words for word in words)
    for word in words:
        value = _float(word)
        if value >= 0 and math.isfinite(value):
            assert _oracle(word, PROFILES[0]) == _bits(math.sqrt(value))
    assert _oracle(1, PROFILES[0]) == 0x1A3504F3
    assert _oracle(0x80000001, PROFILES[0]) == 0x7FC00000
    assert _oracle(0x7F7FFFFF, PROFILES[0]) == 0x5F7FFFFF
    for word in (1, 0x80000001, 0x7FFFFF, 0x807FFFFF):
        assert _oracle(word, PROFILES[1]) == word & 0x80000000


@pytest.mark.parametrize(
    "fault",
    (
        "size",
        "first-guard",
        "last-guard",
        "count",
        "input",
        "value",
        "composed",
        "nan",
        "infinity",
        "zero",
        "flush",
    ),
)
def test_precise_sqrt_verifier_rejects_corruption(fault):
    expected = _expected(
        [0x40000000, 0xBF800000, 0x7F800000, 0x80000000, 1], PROFILES[0]
    )
    actual = list(expected)
    if fault == "size":
        actual.pop()
    elif fault == "first-guard":
        actual[0] ^= 1
    elif fault == "last-guard":
        actual[-1] ^= 1
    elif fault == "count":
        actual[15] = 2
    elif fault == "input":
        actual[16] ^= 1
    elif fault == "value":
        actual[4] += 1
    elif fault == "composed":
        actual[14] += 1
    else:
        index = {"nan": 1, "infinity": 2, "zero": 3, "flush": 4}[fault]
        actual[4 + FIELDS * index] = 0 if fault in ("flush", "zero") else 0x3F800000
    with pytest.raises(AssertionError):
        _check(actual, expected)


@pytest.mark.parametrize("profile", PROFILES)
def test_precise_sqrt_executes(tmp_path, monkeypatch, profile):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native square roots")
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
            entry="roots" if target == "metal" else None,
            initial_output=GUARD + [0xDEADBEEF] * (FIELDS * len(inputs)) + GUARD,
        )
        _check(actual, expected)
        records[label] = evidence
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "profile": profile,
                "inputCount": len(inputs),
                "oracle": "120-digit Decimal sqrt, nearest-even binary32",
                "finiteComparison": "bit-exact",
                "nanComparison": "classification-only",
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
kernel void roots(const device uint* values [[buffer(0)]],
                 device uint* results [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
    Narrow value = as_type<$operand>(ushort(values[i]));
    uint count = 0;
    auto implicit_value = metal::precise::sqrt(record(count, value));
    auto explicit_value = metal::precise::sqrt(float(value));
    results[4u + 3u*i] = as_type<uint>(implicit_value);
    results[5u + 3u*i] = as_type<uint>(explicit_value);
    results[6u + 3u*i] = count;
}
""".replace("$operand", operand)


@pytest.mark.parametrize("operand", ("half", "bfloat"))
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_precise_sqrt_narrow_alias_compiles(tmp_path, operand, target):
    generated = _translate(tmp_path, _narrow_source(operand), target)
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


@pytest.mark.parametrize("operand", ("half", "bfloat"))
def test_precise_sqrt_promotes_narrow_alias_once(tmp_path, monkeypatch, operand):
    source = _narrow_source(operand)
    canonical = _translate(tmp_path, source)
    assert "__crossgl_metal_precise_sqrt_float(record(count, value))" in canonical
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native square-root promotion")
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
    infinity = 0x7C00 if operand == "half" else 0x7F80
    inputs = [word for word in range(65536) if word & 0x7FFF <= infinity]
    inputs.extend((infinity + 1, infinity + 0x8001))
    expected = []
    for word in inputs:
        promoted = (
            _bits(struct.unpack("<e", struct.pack("<H", word))[0])
            if operand == "half"
            else word << 16
        )
        expected.append(_oracle(promoted, PROFILES[0]))
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    actual, evidence = native._dispatch(
        tmp_path,
        target,
        _translate(tmp_path, source, target),
        [(word,) for word in inputs],
        3 * len(inputs) + 8,
        entry="roots" if target == "metal" else None,
        initial_output=GUARD + [0xDEADBEEF] * (3 * len(inputs)) + GUARD,
    )
    assert len(actual) == 3 * len(inputs) + 8
    assert actual[:4] == actual[-4:] == GUARD
    for index, want in enumerate(expected):
        implicit, explicit, count = actual[4 + 3 * index : 7 + 3 * index]
        assert implicit == explicit, "implicit and explicit promotion"
        assert count == 1, "operand evaluation count"
        _check_value(implicit, want)
    (tmp_path / "evidence.json").write_text(
        json.dumps({**evidence, "inputCount": len(inputs)}, indent=2)
    )


def test_ci_requires_square_root_in_existing_native_step():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned unary arithmetic"
    )
    for name in (
        "test_precise_sqrt_executes",
        "test_precise_sqrt_promotes_narrow_alias_once",
    ):
        selector = f"tests/test_translator/test_metal_precise_sqrt.py::{name}"
        assert workflow.count(selector) == 1 and selector in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "--timeout-seconds 300" in step
    assert "continue-on-error" not in step and "if:" not in step
