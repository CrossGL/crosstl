"""Explicit binary32 addition and subtraction policies across native targets."""

import json
import os
import sys
from functools import partial
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalAst import BinaryOpNode
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalAdditiveProfileError,
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
from tests.test_translator.test_floating_remainder_math import _pairs
from tests.test_translator.test_fused_math import _dispatch
from tests.test_translator.test_fused_math import _oracle as _exact_fma
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_division import _bfloat

PROFILES = ("rne-gradual", "rne-flush")
REQUIRE_ENV = "CROSTL_REQUIRE_ADDITIVE_PROFILE"
GUARD = 0x1937A5C3
FIELDS = 24
SOURCE = """#include <metal_stdlib>
using namespace metal;
using Real = float;
struct Cell { Real value; };
uint asuint(float value) { return uint(value) + 200u; }
float asfloat(uint value) { return float(value) + 20.0f; }
static float record(thread uint& count, float value) { count += 1u; return value; }
kernel void additive_profile(const device uint* values [[buffer(0)]],
                             device uint* results [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
    Real a = as_type<float>(values[2u * i]);
    float b = as_type<float>(values[2u * i + 1u]);
    uint count = 0u;
    float added = record(count, a) + b;
    float subtracted = a - record(count, b);
    float2 pair = float2(a, b) + float2(b, a);
    float3 triple = float3(a, b, a) - b;
    float4 quad = a + float4(b, a, b, a);
    float local = a; float returned = (local += b);
    Cell cell; cell.value = a; cell.value -= b;
    float3 vector(a); vector += b;
    float3 component(a); component[1] -= b;
    bfloat narrow_a = bfloat(a); bfloat narrow_b = bfloat(b);
    bfloat narrow_add = narrow_a + narrow_b;
    bfloat narrow_subtract = narrow_a - narrow_b;
    narrow_a += narrow_b;
    results[4u + 24u * i] = as_type<uint>(added);
    results[5u + 24u * i] = as_type<uint>(subtracted);
    results[6u + 24u * i] = as_type<uint>(pair.x);
    results[7u + 24u * i] = as_type<uint>(pair.y);
    results[8u + 24u * i] = as_type<uint>(triple.x);
    results[9u + 24u * i] = as_type<uint>(triple.y);
    results[10u + 24u * i] = as_type<uint>(triple.z);
    results[11u + 24u * i] = as_type<uint>(quad.x);
    results[12u + 24u * i] = as_type<uint>(quad.y);
    results[13u + 24u * i] = as_type<uint>(quad.z);
    results[14u + 24u * i] = as_type<uint>(quad.w);
    results[15u + 24u * i] = as_type<uint>(local);
    results[16u + 24u * i] = as_type<uint>(returned);
    results[17u + 24u * i] = as_type<uint>(cell.value);
    results[18u + 24u * i] = as_type<uint>(vector.z);
    results[19u + 24u * i] = as_type<uint>(component[1]);
    results[20u + 24u * i] = as_type<uint>(component[0]);
    results[21u + 24u * i] = uint(as_type<ushort>(narrow_add)) << 16u;
    results[22u + 24u * i] = uint(as_type<ushort>(narrow_subtract)) << 16u;
    results[23u + 24u * i] = uint(as_type<ushort>(narrow_a)) << 16u;
    results[24u + 24u * i] = as_type<uint>(a);
    results[25u + 24u * i] = as_type<uint>(b);
    results[26u + 24u * i] = count + ::asuint(0.0f) + uint(::asfloat(0u));
    results[27u + 24u * i] = as_type<uint>((a + b) - a);
}
"""

BUFFER_SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void additive_profile(const device uint* values [[buffer(0)]],
                             device float* results [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
    float a = as_type<float>(values[2u * i]);
    float b = as_type<float>(values[2u * i + 1u]);
    results[4u + 4u * i] = a;
    float added = (results[4u + 4u * i] += b);
    results[5u + 4u * i] = added;
    results[6u + 4u * i] = a;
    float subtracted = (results[6u + 4u * i] -= b);
    results[7u + 4u * i] = subtracted;
}
"""


def _translate(tmp_path, source=SOURCE, target="crossgl", profile=PROFILES[1]):
    path = tmp_path / "additive.metal"
    path.write_text(source, encoding="utf-8")
    return translate(
        str(path),
        backend=target,
        format_output=False,
        source_options={"binary32_additive_profile": profile},
    )


def _oracle(a, b, flush, subtract=False):
    # Arbitrary-precision integer reference, independent of shader word arithmetic.
    return _exact_fma(a, 0x3F800000, b ^ (0x80000000 if subtract else 0), flush)


@pytest.mark.parametrize(
    "a,b,preserved,flushed",
    [
        (0x00800000, 0x80800001, 0x80000001, 0x80000000),
        (0x80800000, 0x00800001, 1, 0),
        (0x00800000, 0x807FFFFF, 1, 0x00800000),
        (0x81E657A8, 0x01F010C6, 0x0026E478, 0),
        (0x80000000, 0x80000000, 0x80000000, 0x80000000),
        (0x3F800000, 0x33800000, 0x3F800000, 0x3F800000),
        (0x3F800001, 0x33800000, 0x3F800002, 0x3F800002),
    ],
)
def test_additive_oracle_boundaries(a, b, preserved, flushed):
    assert _oracle(a, b, False) == preserved
    assert _oracle(a, b, True) == flushed
    assert _oracle(a, b ^ 0x80000000, False, True) == preserved
    assert _oracle(a, b ^ 0x80000000, True, True) == flushed


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_additive_profiles_compile(tmp_path, profile, target):
    generated = _translate(tmp_path, profile=profile, target=target)
    assert "crossgl_metal_add_float" in generated
    assert "crossgl_metal_subtract_float" in generated
    assert "crossgl_metal_add_assign" in generated
    assert "crossgl_metal_subtract_assign" in generated
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


@pytest.mark.parametrize("profile", (True, 1, "", "flush-subnormals", [], {}))
def test_additive_profile_rejects_invalid_configuration(profile):
    with pytest.raises(ValueError, match="binary32_additive_profile"):
        MetalToCrossGLConverter(binary32_additive_profile=profile)


def test_additive_profile_is_explicit_and_resets(tmp_path):
    assert "__crossgl_metal_add_float" not in _translate(tmp_path, profile=None)
    converter = MetalToCrossGLConverter(binary32_additive_profile=PROFILES[0])
    for source, expected in (
        (SOURCE, True),
        ("float value(float a) { return a; }", False),
    ):
        tree = MetalParser(MetalLexer(source).tokenize()).parse()
        assert ("__crossgl_fma_bits" in converter.generate(tree)) == expected


def test_additive_profile_retains_operator_ownership_and_other_types(tmp_path):
    source = """
struct Value { float x; };
Value crosstl_metal_operator_add__Value__Value(Value a, Value b) {
    Value c; c.x = a.x * b.x; return c;
}
Value custom(Value a, Value b) { return a + b; }
int integral(int a, int b) { return a + b; }
half narrow(half a, half b) { return a - b; }
double wide(double a, double b) { return a + b; }
device float* offset(device float* a, int b) { return a + b; }
"""
    assert _translate(tmp_path, source) == _translate(tmp_path, source, profile=None)


def test_additive_helpers_avoid_source_names(tmp_path):
    source = """
uint __crossgl_fma_bits(uint a) { return a; }
float __crossgl_metal_add_float(float a) { return a; }
float __crossgl_metal_subtract_float(float a) { return a; }
float value(float a, float b) { return (a + b) - b; }
"""
    generated = _translate(tmp_path, source)
    assert "uint __crossgl_fma_bits_(uint a, uint b" in generated
    for name in ("add", "subtract"):
        assert f"float __crossgl_metal_{name}_float_(float a, float b)" in generated


@pytest.mark.parametrize(
    "source",
    (
        "float value = 1.0f + 3.0f;",
        "float value = 1.0f - 3.0f;",
        "float value(float a, float b) { a += b++; return a; }",
        "float value(float a, float b) { float2 c(a); c.x -= b; return c.x; }",
    ),
)
def test_additive_profile_diagnoses_unrepresentable_contexts(tmp_path, source):
    with pytest.raises(MetalAdditiveProfileError) as error:
        _translate(tmp_path, source)
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-additive-profile-unsupported"
    )
    assert error.value.profile == PROFILES[1]


def test_additive_profile_diagnoses_unresolved_type():
    converter = MetalToCrossGLConverter(binary32_additive_profile=PROFILES[1])
    with pytest.raises(MetalAdditiveProfileError, match="unresolved"):
        converter.generate_expression(BinaryOpNode("unresolved", "+", "1.0f"))


@pytest.mark.parametrize("width", (8, 16))
@pytest.mark.parametrize("operator,name", (("+", "add"), ("-", "subtract")))
def test_additive_wide_vectors_use_per_lane_helpers(tmp_path, width, operator, name):
    generated = _translate(
        tmp_path,
        f"""using Wide = metal::vec<float, {width}>;
Wide value(Wide a, Wide b) {{ Wide c = a {operator} b; c {operator}= b; return c; }}
""",
    )
    assert generated.count(f"__crossgl_metal_{name}_float(float(") == 2 * width


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_additive_profile_survives_saved_intermediate(tmp_path, target):
    path = tmp_path / "saved.cgl"
    path.write_text(_translate(tmp_path), encoding="utf-8")
    generated = translate(str(path), backend=target, format_output=False)
    assert "crossgl_metal_add_float" in generated
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


def test_additive_profile_is_independent_of_fused_profile():
    source = "float value(float a, float b) { return fma(a, b, a) + b; }"
    tree = MetalParser(MetalLexer(source).tokenize()).parse()
    generated = MetalToCrossGLConverter(
        binary32_additive_profile=PROFILES[0], binary32_fma_profile=PROFILES[1]
    ).generate(tree)
    assert "asuint(a), 0x3f800000u, asuint(b), false" in generated
    assert "asuint(a), asuint(b), asuint(c), true" in generated


def test_additive_profile_report_provenance(tmp_path):
    (tmp_path / "additive.metal").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "crosstl.toml").write_text(
        """[project]
targets = ["directx", "opengl"]
[project.source_options.metal]
binary32_additive_profile = "rne-gradual"
[project.source_options.metal.target_options.opengl.source_patterns."additive.metal"]
binary32_additive_profile = "rne-flush"
""",
        encoding="utf-8",
    )
    report = translate_project(load_project_config(tmp_path), format_output=False)
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 2, data["diagnostics"]
    expected = dict(zip(("directx", "opengl"), PROFILES))
    assert {
        a["target"]: a["provenance"]["binary32AdditiveProfile"]
        for a in data["artifacts"]
    } == expected
    path = tmp_path / "report.json"
    report.write_json(path)
    assert validate_project_report(path)["success"]
    manifest = build_runtime_artifact_manifest(path)
    assert manifest["success"], manifest
    assert {
        a["target"]: a["provenance"]["binary32AdditiveProfile"]
        for a in manifest["artifacts"]
    } == expected
    manifest_path = tmp_path / "artifacts.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert build_runtime_package(manifest_path, tmp_path / "package")["success"]
    for invalid in (None, PROFILES[1], "flush-subnormals"):
        if invalid is None:
            data["artifacts"][0]["provenance"].pop("binary32AdditiveProfile", None)
        else:
            data["artifacts"][0]["provenance"]["binary32AdditiveProfile"] = invalid
        path.write_text(json.dumps(data), encoding="utf-8")
        result = validate_project_report(path)
        assert not result["success"]
        assert "binary32AdditiveProfile" in json.dumps(result["diagnostics"])


@pytest.mark.parametrize("profile", (*PROFILES, "source"))
@pytest.mark.parametrize("buffer", (False, True))
def test_additive_profiles_execute(tmp_path, profile, buffer, monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native additive policies")
    from tests.test_translator import test_fused_math as native

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    if profile == "source" and target != "metal":
        pytest.skip("The unchanged source control requires Metal")
    monkeypatch.setattr(
        native,
        "_compile",
        partial(
            _compile,
            directx_compile_flags=("-enable-16bit-types",),
            metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
        ),
    )
    pairs = _pairs(32) + [
        (0x00800000, 0x80800001),
        (0x80800000, 0x00800001),
        (0x81E657A8, 0x01F010C6),
        (0x00800000, 0x807FFFFF),
    ]
    # Adjacent opposite-signed normal values cancel into binary32 subnormals.
    pairs.extend((a << 16, (a + 1) << 16 ^ 0x80000000) for a in range(0x80, 0x180))
    if buffer:
        pairs = pairs[::17] + pairs[-256:]
    expected = [GUARD] * 4
    for a, b in pairs:
        flush = profile != PROFILES[0]
        add = _oracle(a, b, flush)
        sub = _oracle(a, b, flush, True)
        if buffer:
            expected.extend([add, add, sub, sub])
            continue
        narrow_add = _bfloat(_oracle(_bfloat(a), _bfloat(b), flush))
        narrow_sub = _bfloat(_oracle(_bfloat(a), _bfloat(b), flush, True))
        expected.extend(
            [
                add,
                sub,
                add,
                _oracle(b, a, flush),
                sub,
                _oracle(b, b, flush, True),
                sub,
                add,
                _oracle(a, a, flush),
                add,
                _oracle(a, a, flush),
                add,
                add,
                sub,
                add,
                sub,
                a,
                narrow_add,
                narrow_sub,
                narrow_add,
                a,
                b,
                222,
                _oracle(add, a, flush, True),
            ]
        )
    expected.extend([GUARD] * 4)
    original = BUFFER_SOURCE if buffer else SOURCE
    source = (
        original
        if profile == "source"
        else _translate(tmp_path, original, profile=profile, target=target)
    )
    (tmp_path / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    actual, evidence = _dispatch(
        tmp_path,
        target,
        source,
        pairs,
        len(expected),
        entry="additive_profile" if target == "metal" else None,
        initial_output=[GUARD] * len(expected),
        output_dtype="float32" if buffer else "uint32",
    )
    assert len(actual) == len(expected)
    differences = []
    for i, (want, got) in enumerate(zip(expected, actual)):
        arithmetic = 4 <= i < len(expected) - 4 and (
            buffer or (i - 4) % FIELDS not in {16, 20, 21, 22}
        )
        both_nan = want & 0x7FFFFFFF > 0x7F800000 and got & 0x7FFFFFFF > 0x7F800000
        classification_only = profile == "source" or (
            not buffer and (i - 4) % FIELDS in {17, 18, 19}
        )
        if want != got and not (arithmetic and classification_only and both_nan):
            differences.append({"index": i, "expected": want, "actual": got})
    evidence.update(
        profile=profile,
        pairCount=len(pairs),
        guardCount=8,
        mismatchCount=len(differences),
        mismatches=differences,
        nanComparison="exact helper results; classification after bfloat narrowing and in original source",
    )
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    assert not differences, differences[:20]


def test_ci_requires_additive_profiles_in_existing_native_job():
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
    assert workflow.count("tests/test_translator/test_metal_additive_profile.py") == 1
    assert "tests/test_translator/test_metal_additive_profile.py" in step
    assert "--timeout-seconds 120" in step
