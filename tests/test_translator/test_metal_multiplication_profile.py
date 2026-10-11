"""Explicit binary32 products, including subnormals and constant operands."""

import json
import os
import sys
from functools import partial
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalAst import BinaryOpNode
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalMultiplicationProfileError,
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
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_division import _bfloat

PROFILES = ("rne-gradual", "rne-flush")
REQUIRE_ENV = "CROSTL_REQUIRE_MULTIPLICATION_PROFILE"
GUARD = 0x1937A5C3
FIELDS = 28
SOURCE = """#include <metal_stdlib>
using namespace metal;
using Real = float;
struct Cell { Real value; };
struct Converted { float value; operator float() const thread { return value; } };
uint asuint(float value) { return uint(value) + 200u; }
float asfloat(uint value) { return float(value) + 20.0f; }
static float record(thread uint& count, float value) { count += 1u; return value; }
kernel void multiplication_profile(const device uint* values [[buffer(0)]],
                                   device uint* results [[buffer(1)]],
                                   uint i [[thread_position_in_grid]]) {
    Real a = as_type<float>(values[2u * i]);
    float b = as_type<float>(values[2u * i + 1u]);
    uint count = 0u;
    float product = record(count, a) * b;
    float reverse = b * record(count, a);
    float2 pair = float2(a, b) * float2(b, a);
    float3 triple = float3(a, b, a) * b;
    float4 quad = a * float4(b, a, b, a);
    Converted right; right.value = b;
    float local = a; float returned = (local *= right);
    Cell cell; cell.value = a; cell.value *= b;
    float3 vector(a); vector *= b;
    float3 component(a); component[1] *= b;
    bfloat narrow_a = bfloat(a); bfloat narrow_b = bfloat(b);
    bfloat narrow_product = narrow_a * narrow_b;
    narrow_a *= narrow_b;
    results[4u + 28u * i] = as_type<uint>(product);
    results[5u + 28u * i] = as_type<uint>(reverse);
    results[6u + 28u * i] = as_type<uint>(pair.x);
    results[7u + 28u * i] = as_type<uint>(pair.y);
    results[8u + 28u * i] = as_type<uint>(triple.x);
    results[9u + 28u * i] = as_type<uint>(triple.y);
    results[10u + 28u * i] = as_type<uint>(triple.z);
    results[11u + 28u * i] = as_type<uint>(quad.x);
    results[12u + 28u * i] = as_type<uint>(quad.y);
    results[13u + 28u * i] = as_type<uint>(quad.z);
    results[14u + 28u * i] = as_type<uint>(quad.w);
    results[15u + 28u * i] = as_type<uint>(local);
    results[16u + 28u * i] = as_type<uint>(returned);
    results[17u + 28u * i] = as_type<uint>(cell.value);
    results[18u + 28u * i] = as_type<uint>(vector.z);
    results[19u + 28u * i] = as_type<uint>(component[1]);
    results[20u + 28u * i] = as_type<uint>(component[0]);
    results[21u + 28u * i] = uint(as_type<ushort>(narrow_product)) << 16u;
    results[22u + 28u * i] = uint(as_type<ushort>(narrow_a)) << 16u;
    results[23u + 28u * i] = as_type<uint>(0.0f * a);
    results[24u + 28u * i] = as_type<uint>(-0.0f * a);
    results[25u + 28u * i] = as_type<uint>(a * 0.0f);
    results[26u + 28u * i] = as_type<uint>(a * -0.0f);
    results[27u + 28u * i] = as_type<uint>(a);
    results[28u + 28u * i] = as_type<uint>(b);
    results[29u + 28u * i] = count + ::asuint(0.0f) + uint(::asfloat(0u));
    Converted converted; converted.value = a;
    results[30u + 28u * i] = as_type<uint>(2 * converted);
    results[31u + 28u * i] = as_type<uint>((a * b) * b);
}
"""

BUFFER_SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void multiplication_profile(const device uint* values [[buffer(0)]],
                                   device float* results [[buffer(1)]],
                                   uint i [[thread_position_in_grid]]) {
    float a = as_type<float>(values[2u * i]);
    float b = as_type<float>(values[2u * i + 1u]);
    results[4u + 2u * i] = a;
    float product = (results[4u + 2u * i] *= b);
    results[5u + 2u * i] = product;
}
"""


def _translate(tmp_path, source=SOURCE, target="crossgl", profile=PROFILES[1]):
    path = tmp_path / "multiplication.metal"
    path.write_text(source, encoding="utf-8")
    return translate(
        str(path),
        backend=target,
        format_output=False,
        source_options={"binary32_multiplication_profile": profile},
    )


def _oracle(a, b, flush):
    # Exact integer product and independent ties-to-even quantization; no host FP.
    if flush:
        a, b = (v & 0x80000000 if v & 0x7F800000 == 0 else v for v in (a, b))
    sign = (a ^ b) & 0x80000000
    aa, bb = a & 0x7FFFFFFF, b & 0x7FFFFFFF
    if max(aa, bb) > 0x7F800000 or (max(aa, bb) == 0x7F800000 and min(aa, bb) == 0):
        return 0x7FC00000
    if max(aa, bb) == 0x7F800000:
        return sign | 0x7F800000
    if min(aa, bb) == 0:
        return sign
    ea, eb = aa >> 23, bb >> 23
    product = ((aa & 0x7FFFFF) | (0x800000 if ea else 0)) * (
        (bb & 0x7FFFFF) | (0x800000 if eb else 0)
    )
    base = max(ea, 1) + max(eb, 1) - 300
    exponent = product.bit_length() - 1 + base
    if flush and exponent < -126:
        return sign
    quantum = max(-149, exponent - 23)
    shift = quantum - base
    if shift > 0:
        rounded, remainder = divmod(product, 1 << shift)
        half = 1 << (shift - 1)
        rounded += remainder > half or (remainder == half and rounded & 1)
    else:
        rounded = product << -shift
    if not rounded:
        return sign
    exponent = rounded.bit_length() - 1 + quantum
    if exponent > 127:
        return sign | 0x7F800000
    if exponent < -126:
        return sign | rounded
    if rounded >= 1 << 24:
        rounded >>= 1
    return sign | ((exponent + 127) << 23) | (rounded & 0x7FFFFF)


@pytest.mark.parametrize(
    "a,b,preserved,flushed",
    [
        (0x00800000, 0x3F000000, 0x00400000, 0),
        (0x80800000, 0x3F000000, 0x80400000, 0x80000000),
        (1, 0x7F000000, 0x34800000, 0),
        (0x007FFFFF, 0x40000000, 0x00FFFFFE, 0),
        (0x00800000, 0x3F7FFFFF, 0x00800000, 0),
        (0x80000000, 0x3F800000, 0x80000000, 0x80000000),
        (0x80000000, 0xBF800000, 0, 0),
        (0, 0x7F800000, 0x7FC00000, 0x7FC00000),
        (0x7F7FFFFF, 0x40000000, 0x7F800000, 0x7F800000),
        (0x3F800001, 0x3FC00000, 0x3FC00002, 0x3FC00002),
        (0x3F800003, 0x3FC00000, 0x3FC00004, 0x3FC00004),
    ],
)
def test_multiplication_oracle_boundaries(a, b, preserved, flushed):
    assert _oracle(a, b, False) == preserved
    assert _oracle(a, b, True) == flushed
    assert _oracle(b, a, False) == preserved


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_multiplication_profiles_compile(tmp_path, profile, target):
    generated = _translate(tmp_path, profile=profile, target=target)
    assert "crossgl_metal_multiply_float" in generated
    assert "crossgl_metal_multiply_assign" in generated
    assert "Berkeley SoftFloat" in generated
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


@pytest.mark.parametrize("profile", (True, 1, "", "flush-subnormals", [], {}))
def test_multiplication_profile_rejects_invalid_configuration(profile):
    with pytest.raises(ValueError, match="binary32_multiplication_profile"):
        MetalToCrossGLConverter(binary32_multiplication_profile=profile)


def test_multiplication_profile_is_explicit_and_resets(tmp_path):
    assert "__crossgl_metal_multiply_float" not in _translate(tmp_path, profile=None)
    converter = MetalToCrossGLConverter(binary32_multiplication_profile=PROFILES[0])
    for source, expected in (
        (SOURCE, True),
        ("float value(float a) { return a; }", False),
    ):
        tree = MetalParser(MetalLexer(source).tokenize()).parse()
        assert ("__crossgl_fma_bits" in converter.generate(tree)) == expected


def test_multiplication_keeps_operator_ownership_and_other_types(tmp_path):
    source = """
struct Value { float x; };
Value crosstl_metal_operator_multiply__Value__Value(Value a, Value b) {
    Value c; c.x = a.x + b.x; return c;
}
Value custom(Value a, Value b) { return a * b; }
int integral(int a, int b) { return a * b; }
half narrow(half a, half b) { return a * b; }
double wide(double a, double b) { return a * b; }
"""
    assert _translate(tmp_path, source) == _translate(tmp_path, source, profile=None)


def test_multiplication_helpers_avoid_source_names(tmp_path):
    source = """
uint __crossgl_fma_bits(uint a) { return a; }
float __crossgl_metal_multiply_float(float a) { return a; }
float value(float a, float b) { return a * b; }
"""
    generated = _translate(tmp_path, source)
    assert "uint __crossgl_fma_bits_(uint a, uint b" in generated
    assert "float __crossgl_metal_multiply_float_(float a, float b)" in generated


@pytest.mark.parametrize(
    "source",
    (
        "float value = 1.0f * 3.0f;",
        "float value(float a, float b) { a *= b++; return a; }",
        "float value(float a, float b) { float2 c(a); c.x *= b; return c.x; }",
    ),
)
def test_multiplication_diagnoses_unrepresentable_contexts(tmp_path, source):
    with pytest.raises(MetalMultiplicationProfileError) as error:
        _translate(tmp_path, source)
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-multiplication-profile-unsupported"
    )
    assert error.value.profile == PROFILES[1]


def test_multiplication_diagnoses_unresolved_type():
    converter = MetalToCrossGLConverter(binary32_multiplication_profile=PROFILES[1])
    with pytest.raises(MetalMultiplicationProfileError, match="unresolved"):
        converter.generate_expression(BinaryOpNode("unresolved", "*", "1.0f"))


@pytest.mark.parametrize("width", (8, 16))
def test_multiplication_wide_vectors_use_per_lane_helpers(tmp_path, width):
    generated = _translate(
        tmp_path,
        f"""using Wide = metal::vec<float, {width}>;
Wide value(Wide a, Wide b) {{ Wide c = a * b; c *= b; return c; }}
""",
    )
    assert generated.count("__crossgl_metal_multiply_float(float(") == 2 * width


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_multiplication_profile_survives_saved_intermediate(tmp_path, target):
    path = tmp_path / "saved.cgl"
    path.write_text(_translate(tmp_path), encoding="utf-8")
    generated = translate(str(path), backend=target, format_output=False)
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


def test_multiplication_profile_is_independent_of_fused_profile():
    source = "float value(float a, float b) { return fma(a, b, a) * b; }"
    tree = MetalParser(MetalLexer(source).tokenize()).parse()
    generated = MetalToCrossGLConverter(
        binary32_multiplication_profile=PROFILES[0], binary32_fma_profile=PROFILES[1]
    ).generate(tree)
    assert "left, right, zero, false" in generated
    assert "asuint(a), asuint(b), asuint(c), true" in generated
    assert generated.count("uint __crossgl_fma_bits(") == 1


def test_multiplication_profile_report_provenance(tmp_path):
    (tmp_path / "multiplication.metal").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "crosstl.toml").write_text(
        """[project]
targets = ["directx", "opengl"]
[project.source_options.metal]
binary32_multiplication_profile = "rne-gradual"
[project.source_options.metal.target_options.opengl.source_patterns."multiplication.metal"]
binary32_multiplication_profile = "rne-flush"
""",
        encoding="utf-8",
    )
    report = translate_project(load_project_config(tmp_path), format_output=False)
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 2, data["diagnostics"]
    expected = dict(zip(("directx", "opengl"), PROFILES))
    assert {
        a["target"]: a["provenance"]["binary32MultiplicationProfile"]
        for a in data["artifacts"]
    } == expected
    path = tmp_path / "report.json"
    report.write_json(path)
    assert validate_project_report(path)["success"]
    manifest = build_runtime_artifact_manifest(path)
    assert manifest["success"], manifest
    assert {
        a["target"]: a["provenance"]["binary32MultiplicationProfile"]
        for a in manifest["artifacts"]
    } == expected
    manifest_path = tmp_path / "artifacts.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert build_runtime_package(manifest_path, tmp_path / "package")["success"]
    for invalid in (None, PROFILES[1], "flush-subnormals"):
        if invalid is None:
            data["artifacts"][0]["provenance"].pop(
                "binary32MultiplicationProfile", None
            )
        else:
            data["artifacts"][0]["provenance"][
                "binary32MultiplicationProfile"
            ] = invalid
        path.write_text(json.dumps(data), encoding="utf-8")
        result = validate_project_report(path)
        assert not result["success"]
        assert "binary32MultiplicationProfile" in json.dumps(result["diagnostics"])


@pytest.mark.parametrize("profile", (*PROFILES, "source"))
@pytest.mark.parametrize("buffer", (False, True))
def test_multiplication_profiles_execute(tmp_path, profile, buffer, monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native multiplication policies")
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
    pairs = _pairs(32) + [(0x00800000, 0x3F000000), (1, 0x7F000000)]
    if buffer:
        pairs = pairs[::17]
    expected = [GUARD] * 4
    for a, b in pairs:
        flush = profile != PROFILES[0]
        product = _oracle(a, b, flush)
        if buffer:
            expected.extend([product, product])
            continue
        narrow = _bfloat(_oracle(_bfloat(a), _bfloat(b), flush))
        zero = _oracle(0, a, flush)
        negative_zero = _oracle(0x80000000, a, flush)
        expected.extend(
            [
                product,
                product,
                product,
                product,
                product,
                _oracle(b, b, flush),
                product,
                product,
                _oracle(a, a, flush),
                product,
                _oracle(a, a, flush),
                product,
                product,
                product,
                product,
                product,
                a,
                narrow,
                narrow,
                zero,
                negative_zero,
                zero,
                negative_zero,
                a,
                b,
                222,
                _oracle(0x40000000, a, flush),
                _oracle(product, b, flush),
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
        entry="multiplication_profile" if target == "metal" else None,
        initial_output=[GUARD] * len(expected),
        output_dtype="float32" if buffer else "uint32",
    )
    assert len(actual) == len(expected)
    differences = []
    for i, (want, got) in enumerate(zip(expected, actual)):
        arithmetic = 4 <= i < len(expected) - 4 and (
            buffer or (i - 4) % FIELDS not in {16, 23, 24, 25}
        )
        both_nan = want & 0x7FFFFFFF > 0x7F800000 and got & 0x7FFFFFFF > 0x7F800000
        classification_only = profile == "source" or (
            not buffer and (i - 4) % FIELDS in {17, 18}
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


def test_ci_requires_multiplication_profiles_in_existing_native_job():
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
    assert (
        workflow.count("tests/test_translator/test_metal_multiplication_profile.py")
        == 1
    )
    assert "tests/test_translator/test_metal_multiplication_profile.py" in step
    assert "--timeout-seconds 120" in step
