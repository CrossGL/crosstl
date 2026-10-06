"""Explicit binary32 remainder policies with independent native references."""

import json
import os
import sys
from functools import partial
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalRemainderProfileError,
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
from tests.test_translator.test_floating_remainder_math import _oracle as _exact
from tests.test_translator.test_floating_remainder_math import _pairs
from tests.test_translator.test_fused_math import _dispatch
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_division import _bfloat

PROFILES = ("preserve-subnormals", "flush-arithmetic-subnormals")
REQUIRE_ENV = "CROSTL_REQUIRE_REMAINDER_PROFILE"
GUARD = 0x1937A5C3
FIELDS = 14
SOURCE = """#include <metal_stdlib>
using namespace metal;
uint asuint(float value) { return uint(value) + 200u; }
float asfloat(uint value) { return float(value) + 20.0f; }
static float record(thread uint& count, float value) { count += 1u; return value; }
kernel void remainder_profile(const device uint* values [[buffer(0)]],
                              device uint* results [[buffer(1)]],
                              uint i [[thread_position_in_grid]]) {
    float a = as_type<float>(values[2u * i]);
    float b = as_type<float>(values[2u * i + 1u]);
    float negative_a = as_type<float>(values[2u * i] ^ 0x80000000u);
    float negative_b = as_type<float>(values[2u * i + 1u] ^ 0x80000000u);
    uint count = 0u;
    float scalar = fmod(record(count, a), b);
    float2 pair = metal::fmod(float2(a, negative_a), float2(b, negative_b));
    float3 triple = fmod(float3(a, negative_a, a), b);
    float4 quad = metal::fmod(a, float4(b, negative_b, b, negative_b));
    bfloat narrow = bfloat(fmod(bfloat(a), bfloat(b)));
    results[4u + 14u * i] = as_type<uint>(scalar);
    results[5u + 14u * i] = as_type<uint>(pair.x);
    results[6u + 14u * i] = as_type<uint>(pair.y);
    results[7u + 14u * i] = as_type<uint>(triple.x);
    results[8u + 14u * i] = as_type<uint>(triple.y);
    results[9u + 14u * i] = as_type<uint>(triple.z);
    results[10u + 14u * i] = as_type<uint>(quad.x);
    results[11u + 14u * i] = as_type<uint>(quad.y);
    results[12u + 14u * i] = as_type<uint>(quad.z);
    results[13u + 14u * i] = as_type<uint>(quad.w);
    results[14u + 14u * i] = uint(as_type<ushort>(narrow)) << 16u;
    results[15u + 14u * i] = as_type<uint>(a);
    results[16u + 14u * i] = as_type<uint>(b);
    results[17u + 14u * i] = count + ::asuint(0.0f) + uint(::asfloat(0u));
}
"""


def _translate(tmp_path, source=SOURCE, target="crossgl", profile=PROFILES[1]):
    path = tmp_path / "remainder.metal"
    path.write_text(source, encoding="utf-8")
    return translate(
        str(path),
        backend=target,
        format_output=False,
        source_options={"binary32_remainder_profile": profile},
    )


def _oracle(a, b, flush):
    if flush and b & 0x7FFFFFFF < 0x00800000:
        return 0x7FC00000
    result = _exact(a, b, 32)
    if (
        flush
        and (a & 0x7FFFFFFF) >= (b & 0x7FFFFFFF)
        and result & 0x7FFFFFFF < 0x00800000
    ):
        return result & 0x80000000
    return result


@pytest.mark.parametrize(
    "a,b,preserved,flushed",
    [
        (1, 0xC0000000, 1, 1),
        (0x80000001, 0x40000000, 0x80000001, 0x80000001),
        (0, 0x807FFFFF, 0, 0x7FC00000),
        (0x00800001, 0x00800000, 1, 0),
        (0x80800001, 0x00800000, 0x80000001, 0x80000000),
    ],
)
def test_remainder_profile_oracle_distinguishes_shortcut_and_arithmetic(
    a, b, preserved, flushed
):
    assert _oracle(a, b, False) == preserved
    assert _oracle(a, b, True) == flushed


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_remainder_profiles_compile(tmp_path, profile, target):
    generated = _translate(tmp_path, profile=profile, target=target)
    assert "crossgl_remainder_bits" in generated
    _compile(
        generated,
        target,
        tmp_path,
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


@pytest.mark.parametrize("profile", (True, 1, "", "flush-subnormals", [], {}))
def test_remainder_profile_rejects_invalid_configuration(profile):
    with pytest.raises(ValueError, match="binary32_remainder_profile"):
        MetalToCrossGLConverter(binary32_remainder_profile=profile)


def test_remainder_profile_is_explicit_and_resets(tmp_path):
    assert "__crossgl_remainder_bits" not in _translate(tmp_path, profile=None)
    converter = MetalToCrossGLConverter(binary32_remainder_profile=PROFILES[0])
    for source, expected in (
        (SOURCE, True),
        ("float add(float x) { return x + 1; }", False),
    ):
        tree = MetalParser(MetalLexer(source).tokenize()).parse()
        assert ("__crossgl_remainder_bits" in converter.generate(tree)) == expected


def test_remainder_profile_preserves_user_overloads_and_other_widths(tmp_path):
    source = """
float fmod(float x, float y) { return x + y; }
float user(float x, float y) { return ::fmod(x, y); }
half narrow(half x, half y) { return metal::fmod(x, y); }
double wide(double x, double y) { return metal::fmod(x, y); }
"""
    assert _translate(tmp_path, source) == _translate(tmp_path, source, profile=None)


@pytest.mark.parametrize("namespace", ("", "precise", "fast"))
@pytest.mark.parametrize("materialized", (False, True))
def test_remainder_profile_preserves_bfloat_wrapper_narrowing(namespace, materialized):
    qualifier = "METAL_FUNC" if materialized else ""
    body = "__metal_fmod(float(x), float(y))" if materialized else "float(x) + float(y)"
    source = f"""
typedef bfloat bfloat16_t;
namespace metal {{
{('namespace ' + namespace + ' {') if namespace else ''}
{qualifier} bfloat16_t fmod(bfloat16_t x, bfloat16_t y) {{
    return bfloat16_t({body});
}}
{'}' if namespace else ''}
}}
bfloat16_t apply(bfloat16_t x, bfloat16_t y) {{
    return metal::{(namespace + '::') if namespace else ''}fmod(x, y);
}}
"""
    converter = MetalToCrossGLConverter(binary32_remainder_profile=PROFILES[1])
    tree = MetalParser(MetalLexer(source).tokenize()).parse()
    generated = converter.generate(tree)
    profiled = materialized and namespace != "fast"
    assert ("__crossgl_remainder_bits" in generated) == profiled
    if profiled:
        assert (
            "bfloat16(__crossgl_metal_remainder_float(float(x), float(y)))" in generated
        )
    elif materialized:
        assert "bfloat16(fmod(float(x), float(y)))" in generated


def test_remainder_helpers_avoid_source_names(tmp_path):
    source = """
uint __crossgl_remainder_bits(uint x) { return x; }
float __crossgl_metal_remainder_float(float x) { return x; }
float apply(float x, float y) { return metal::fmod(x, y); }
"""
    generated = _translate(tmp_path, source)
    assert "uint __crossgl_remainder_bits_(uint a, uint b" in generated
    assert "float __crossgl_metal_remainder_float_(float a, float b)" in generated


def test_remainder_profile_diagnoses_constant_context(tmp_path):
    with pytest.raises(MetalRemainderProfileError, match="require.*function"):
        _translate(tmp_path, "float value = metal::fmod(1.0f, 2.0f);")


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_remainder_profile_survives_saved_intermediate(tmp_path, target):
    path = tmp_path / "saved.cgl"
    path.write_text(_translate(tmp_path), encoding="utf-8")
    generated = translate(str(path), backend=target, format_output=False)
    assert "crossgl_remainder_bits" in generated
    _compile(
        generated,
        target,
        tmp_path,
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


def test_remainder_profile_report_provenance(tmp_path):
    (tmp_path / "remainder.metal").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "crosstl.toml").write_text(
        """[project]
targets = ["directx", "opengl"]
[project.source_options.metal]
binary32_remainder_profile = "preserve-subnormals"
[project.source_options.metal.target_options.opengl.source_patterns."remainder.metal"]
binary32_remainder_profile = "flush-arithmetic-subnormals"
""",
        encoding="utf-8",
    )
    report = translate_project(load_project_config(tmp_path), format_output=False)
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 2, data["diagnostics"]
    expected = dict(zip(("directx", "opengl"), PROFILES))
    assert {
        a["target"]: a["provenance"]["binary32RemainderProfile"]
        for a in data["artifacts"]
    } == expected
    path = tmp_path / "report.json"
    report.write_json(path)
    assert validate_project_report(path)["success"]
    manifest = build_runtime_artifact_manifest(path)
    assert manifest["success"], manifest
    assert {
        a["target"]: a["provenance"]["binary32RemainderProfile"]
        for a in manifest["artifacts"]
    } == expected
    manifest_path = tmp_path / "artifacts.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert build_runtime_package(manifest_path, tmp_path / "package")["success"]
    for invalid in (None, PROFILES[1], "flush-subnormals"):
        if invalid is None:
            data["artifacts"][0]["provenance"].pop("binary32RemainderProfile", None)
        else:
            data["artifacts"][0]["provenance"]["binary32RemainderProfile"] = invalid
        path.write_text(json.dumps(data), encoding="utf-8")
        result = validate_project_report(path)
        assert not result["success"]
        assert "binary32RemainderProfile" in json.dumps(result["diagnostics"])


@pytest.mark.parametrize("profile", (*PROFILES, "source"))
def test_remainder_profile_executes(tmp_path, profile, monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native remainder policies")
    from tests.test_translator import test_fused_math as native

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    if profile == "source" and target != "metal":
        pytest.skip("The unchanged source control requires Metal")
    monkeypatch.setattr(
        native,
        "_compile",
        partial(_compile, metal_compile_flags=("-std=metal3.1", "-fno-fast-math")),
    )
    pairs = _pairs(32)
    expected = [GUARD] * 4
    for a, b in pairs:
        r = lambda x, y: _oracle(x, y, profile != PROFILES[0])
        expected.extend(
            [
                r(a, b),
                r(a, b),
                r(a ^ 0x80000000, b ^ 0x80000000),
                r(a, b),
                r(a ^ 0x80000000, b),
                r(a, b),
                r(a, b),
                r(a, b ^ 0x80000000),
                r(a, b),
                r(a, b ^ 0x80000000),
                _bfloat(r(_bfloat(a), _bfloat(b))),
                a,
                b,
                221,
            ]
        )
    expected.extend([GUARD] * 4)
    source = (
        SOURCE
        if profile == "source"
        else _translate(tmp_path, profile=profile, target=target)
    )
    (tmp_path / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    actual, evidence = _dispatch(
        tmp_path,
        target,
        source,
        pairs,
        len(expected),
        entry="remainder_profile" if target == "metal" else None,
        initial_output=[GUARD] * len(expected),
    )
    assert len(actual) == len(expected)
    differences = []
    for i, (want, got) in enumerate(zip(expected, actual)):
        arithmetic = 4 <= i < len(expected) - 4 and (i - 4) % FIELDS <= 10
        both_nan = want & 0x7FFFFFFF > 0x7F800000 and got & 0x7FFFFFFF > 0x7F800000
        if want != got and not (arithmetic and both_nan):
            differences.append({"index": i, "expected": want, "actual": got})
    evidence.update(
        profile=profile,
        pairCount=len(pairs),
        guardCount=8,
        mismatchCount=len(differences),
        mismatches=differences,
        nanComparison="arithmetic results only; raw operand words exact",
    )
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    assert not differences, differences[:20]


def test_ci_requires_remainder_profiles_in_existing_native_job():
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
    assert workflow.count("tests/test_translator/test_metal_remainder_profile.py") == 1
    assert "tests/test_translator/test_metal_remainder_profile.py" in step
    assert "--timeout-seconds 120" in step
