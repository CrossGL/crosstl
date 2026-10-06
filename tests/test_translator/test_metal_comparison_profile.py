"""Source comparison profiles keep subnormal policy separate from conversion."""

import json
import operator
import os
import random
import struct
import sys
from functools import partial
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalAst import BinaryOpNode
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalComparisonProfileError,
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
from tests.test_translator.test_fused_math import _dispatch
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_division import _bfloat

REQUIRE_ENV = "CROSTL_REQUIRE_COMPARISON_PROFILE"
PROFILES = ("preserve-subnormals", "flush-subnormals")
GUARD = 0xDEADBEEF
FIELDS = 20
SOURCE = """#include <metal_stdlib>
using namespace metal;
using Real = float;
uint asuint(float value) { return uint(value) + 200u; }
static float record(thread uint& count, float x) { count += 1u; return x; }
kernel void compare_profile(const device uint* values [[buffer(0)]],
                            device uint* results [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    Real a = as_type<float>(values[2u * i]);
    float b = as_type<float>(values[2u * i + 1u]);
    uint count = 0u;
    bool equal = record(count, a) == b;
    bool unused = false && (record(count, a) > b);
    bool4 less = float4(a, b, a, b) < float4(b, a, b, a);
    bool2 greater = float2(a, b) >= a;
    bool3 same = float3(a, b, a) == float3(b, a, a);
    results[4u + 20u * i] = uint(equal);
    results[5u + 20u * i] = uint(a != b);
    results[6u + 20u * i] = uint(a < b);
    results[7u + 20u * i] = uint(a <= b);
    results[8u + 20u * i] = uint(a > b);
    results[9u + 20u * i] = uint(a >= b);
    results[10u + 20u * i] = uint(less.x);
    results[11u + 20u * i] = uint(less.y);
    results[12u + 20u * i] = uint(less.z);
    results[13u + 20u * i] = uint(less.w);
    results[14u + 20u * i] = uint(greater.x);
    results[15u + 20u * i] = uint(greater.y);
    results[16u + 20u * i] = uint(same.x);
    results[17u + 20u * i] = uint(same.y);
    results[18u + 20u * i] = uint(same.z);
    results[19u + 20u * i] = uint(a != 0 && b < 0);
    results[20u + 20u * i] = as_type<uint>(b);
    results[21u + 20u * i] = as_type<uint>(a);
    results[22u + 20u * i] = uint(bfloat(a) < bfloat(b));
    results[23u + 20u * i] = count + uint(unused) + ::asuint(0.0f);
}
"""


def _translate(tmp_path, source=SOURCE, target="crossgl", profile="flush-subnormals"):
    path = tmp_path / "comparison.metal"
    path.write_text(source, encoding="utf-8")
    return translate(
        str(path),
        backend=target,
        format_output=False,
        source_options={"binary32_comparison_profile": profile},
    )


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_comparison_profiles_compile(tmp_path, target, profile):
    generated = _translate(tmp_path, target=target, profile=profile)
    assert "crossgl_compare_bits" in generated
    _compile(
        generated,
        target,
        tmp_path,
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


@pytest.mark.parametrize("profile", (True, 1, "", "rne-flush", [], {}))
def test_comparison_profile_rejects_invalid_configuration(profile):
    with pytest.raises(ValueError, match="binary32_comparison_profile"):
        MetalToCrossGLConverter(binary32_comparison_profile=profile)


def test_comparison_profile_is_explicit_and_resets(tmp_path):
    assert "__crossgl_compare_bits" not in _translate(tmp_path, profile=None)
    converter = MetalToCrossGLConverter(binary32_comparison_profile=PROFILES[0])
    for source, expected in (
        (SOURCE, True),
        ("bool value(int a, int b) { return a < b; }", False),
    ):
        tree = MetalParser(MetalLexer(source).tokenize()).parse()
        assert ("__crossgl_compare_bits" in converter.generate(tree)) == expected


def test_comparison_helpers_avoid_source_names(tmp_path):
    source = """
uint __crossgl_compare_bits(uint a) { return a; }
bool __crossgl_metal_compare_less_float(float a) { return true; }
bool compare(float a, float b) { return a < b; }
"""
    generated = _translate(tmp_path, source)
    assert "uint __crossgl_compare_bits_(uint a, uint b)" in generated
    assert "bool __crossgl_metal_compare_less_float_(float a, float b)" in generated


def test_comparison_keeps_user_operators_and_other_types(tmp_path):
    source = """
struct Value { int x; };
bool crosstl_metal_operator_less__Value__Value(Value a, Value b) { return a.x < b.x; }
bool custom(Value a, Value b) { return a < b; }
bool integral(int a, int b) { return a < b; }
bool narrow(half a, half b) { return a < b; }
bool wider(double a, double b) { return a < b; }
"""
    generated = _translate(tmp_path, source)
    assert "__crossgl_compare_bits" not in generated
    assert "operator_less" in generated
    assert generated.count("return a < b;") == 3


def test_comparison_profile_does_not_replace_boolean_conversions(tmp_path):
    source = "bool converted(float a) { return bool(a); }"
    ordinary = _translate(tmp_path, source, profile=None)
    for profile in PROFILES:
        assert _translate(tmp_path, source, profile=profile) == ordinary


def test_comparison_profile_preserves_narrow_operand_conversion(tmp_path):
    source = """
bool mixed(bfloat a, int b) { return a < b; }
bool promoted(half a, float b) { return a >= b; }
bool2 vector(float2 a, int b) { return a == b; }
"""
    generated = _translate(tmp_path, source)
    assert "float(bfloat16(b))" in generated
    assert "compare_greater_equal_float(float(a), float(b))" in generated
    assert "compare_equal_float2(vec2(a), vec2(b))" in generated


def test_comparison_profile_diagnoses_unknown_type_and_constant_context(tmp_path):
    converter = MetalToCrossGLConverter(binary32_comparison_profile=PROFILES[0])
    with pytest.raises(MetalComparisonProfileError, match="resolved"):
        converter.generate_expression(BinaryOpNode("unknown", "<", "1.0f"))
    with pytest.raises(MetalComparisonProfileError, match="require a function"):
        _translate(tmp_path, "bool x = 1.0f < 2.0f;")


def test_comparison_profile_resolves_static_member_types(tmp_path):
    source = """
template <typename T> struct Bound;
template <> struct Bound<float> { static constexpr constant float max = 1.0f; };
struct Count { static constexpr constant int max = 3; };
bool floating(float value) {
    using Limit = Bound<float>;
    return value < Limit::max;
}
bool integral(int value) { return value < Count::max; }
"""
    generated = _translate(tmp_path, source)
    assert "compare_less_float(float(value), float((1.0f)))" in generated
    assert "return value < 3;" in generated


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_comparison_profile_survives_saved_intermediate(tmp_path, target):
    intermediate = tmp_path / "saved.cgl"
    intermediate.write_text(_translate(tmp_path), encoding="utf-8")
    generated = translate(str(intermediate), backend=target, format_output=False)
    assert "crossgl_compare_bits" in generated
    _compile(
        generated,
        target,
        tmp_path,
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


def test_comparison_profile_report_and_package_provenance(tmp_path):
    (tmp_path / "comparison.metal").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "crosstl.toml").write_text(
        """[project]
targets = ["directx", "opengl"]
[project.source_options.metal]
binary32_comparison_profile = "preserve-subnormals"
[project.source_options.metal.target_options.opengl.source_patterns."comparison.metal"]
binary32_comparison_profile = "flush-subnormals"
""",
        encoding="utf-8",
    )
    report = translate_project(
        load_project_config(tmp_path),
        validate=False,
        run_toolchains=False,
        format_output=False,
    )
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 2, data["diagnostics"]
    expected = {"directx": PROFILES[0], "opengl": PROFILES[1]}
    assert {
        a["target"]: a["provenance"]["binary32ComparisonProfile"]
        for a in data["artifacts"]
    } == expected
    path = tmp_path / "report.json"
    report.write_json(path)
    assert validate_project_report(path)["success"]
    manifest = build_runtime_artifact_manifest(path)
    assert manifest["success"], manifest
    assert {
        a["target"]: a["provenance"]["binary32ComparisonProfile"]
        for a in manifest["artifacts"]
    } == expected
    manifest_path = tmp_path / "artifacts.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert build_runtime_package(manifest_path, tmp_path / "package")["success"]
    for invalid in (None, PROFILES[1], "rne-flush"):
        if invalid is None:
            data["artifacts"][0]["provenance"].pop("binary32ComparisonProfile", None)
        else:
            data["artifacts"][0]["provenance"]["binary32ComparisonProfile"] = invalid
        path.write_text(json.dumps(data), encoding="utf-8")
        validation = validate_project_report(path)
        assert not validation["success"]
        assert "binary32ComparisonProfile" in json.dumps(validation["diagnostics"])


def _oracle(a, b, operation, flush):
    if flush:
        a, b = (
            word & 0x80000000 if word & 0x7F800000 == 0 else word for word in (a, b)
        )
    left, right = (struct.unpack("<f", struct.pack("<I", word))[0] for word in (a, b))
    return int(getattr(operator, operation)(left, right))


def _pairs():
    edges = (
        0,
        0x80000000,
        1,
        0x80000001,
        0x10000,
        0x80010000,
        0x7FFFFF,
        0x807FFFFF,
        0x800000,
        0x80800000,
        0x3F800000,
        0xBF800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0x7F800001,
    )
    pairs = [(a, b) for a in edges for b in edges]
    random_source = random.Random(2000)
    pairs.extend(
        (random_source.getrandbits(32), random_source.getrandbits(32))
        for _ in range(2048)
    )
    return pairs


@pytest.mark.parametrize("profile", (*PROFILES, "source"))
def test_comparison_profile_executes(tmp_path, profile, monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native comparison profiles")
    from tests.test_translator import test_fused_math as native_math

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    if profile == "source" and target != "metal":
        pytest.skip("The unchanged source control requires Metal")
    monkeypatch.setattr(
        native_math,
        "_compile",
        partial(_compile, metal_compile_flags=("-std=metal3.1", "-fno-fast-math")),
    )
    pairs = _pairs()
    flush = profile != PROFILES[0]
    expected = [GUARD] * 4
    for a, b in pairs:
        comparison = lambda x, y, op: _oracle(x, y, op, flush)
        expected.extend(
            [
                *[comparison(a, b, op) for op in ("eq", "ne", "lt", "le", "gt", "ge")],
                comparison(a, b, "lt"),
                comparison(b, a, "lt"),
                comparison(a, b, "lt"),
                comparison(b, a, "lt"),
                comparison(a, a, "ge"),
                comparison(b, a, "ge"),
                comparison(a, b, "eq"),
                comparison(b, a, "eq"),
                comparison(a, a, "eq"),
                int(comparison(a, 0, "ne") and comparison(b, 0, "lt")),
                b,
                a,
                comparison(_bfloat(a), _bfloat(b), "lt"),
                201,
            ]
        )
    expected.extend([GUARD] * 4)
    assert len(expected) == FIELDS * len(pairs) + 8
    generated = (
        SOURCE
        if profile == "source"
        else _translate(tmp_path, target=target, profile=profile)
    )
    (tmp_path / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    actual, evidence = _dispatch(
        tmp_path,
        target,
        generated,
        pairs,
        len(expected),
        entry="compare_profile" if target == "metal" else None,
        initial_output=[GUARD] * len(expected),
    )
    mismatches = [
        {"index": i, "expected": want, "actual": got}
        for i, (want, got) in enumerate(zip(expected, actual))
        if want != got
    ]
    evidence.update(
        profile=profile,
        pairCount=len(pairs),
        guardCount=8,
        mismatchCount=len(mismatches),
        mismatches=mismatches,
    )
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    assert len(actual) == len(expected)
    assert not mismatches, mismatches[:10]


def test_ci_requires_comparison_profiles_once_per_native_target():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate binary32 arithmetic"
    )
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert workflow.count("tests/test_translator/test_metal_comparison_profile.py") == 1
    assert "tests/test_translator/test_metal_comparison_profile.py" in step
    assert "--timeout-seconds 120" in step
    assert "pytest -q -n auto" in step and "--junitxml=" in step
