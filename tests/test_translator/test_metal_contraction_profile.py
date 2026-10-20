"""Metal's expression-level multiply-add contraction, reproduced by translation."""

import json
import math
import os
import random
import struct
import sys
from fractions import Fraction
from functools import partial
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalContractionProfileError,
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
from tests.ci_helpers import assert_native_step_prerequisites
from tests.test_translator.test_fused_math import _dispatch
from tests.test_translator.test_fused_math import _oracle as _fma
from tests.test_translator.test_metal_builtin_ownership import _compile

PROFILES = ("rne-gradual", "rne-flush")
REQUIRE_ENV = "CROSTL_REQUIRE_CONTRACTION_PROFILE"
GUARD = 0x2F6B41D9
FIELDS = 20
ONE = 0x3F800000
NEGATIVE_ZERO = 0x80000000
SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void contraction_profile(const device uint* values [[buffer(0)]],
                                device uint* results [[buffer(1)]],
                                uint i [[thread_position_in_grid]]) {
    float a = as_type<float>(values[4u * i]);
    float b = as_type<float>(values[4u * i + 1u]);
    float c = as_type<float>(values[4u * i + 2u]);
    float d = as_type<float>(values[4u * i + 3u]);
    uint base = 4u + 20u * i;
    results[base] = as_type<uint>(a * b + c);
    results[base + 1u] = as_type<uint>(c + a * b);
    results[base + 2u] = as_type<uint>(a * b - c);
    results[base + 3u] = as_type<uint>(c - a * b);
    results[base + 4u] = as_type<uint>(a * b + c * d);
    results[base + 5u] = as_type<uint>(-(a * b) + c);
    float product = a * b;
    results[base + 6u] = as_type<uint>(product + c);
    float increased = c;
    increased += a * b;
    results[base + 7u] = as_type<uint>(increased);
    float decreased = c;
    decreased -= a * b;
    results[base + 8u] = as_type<uint>(decreased);
    float2 pair = float2(a, c) * float2(b, d) + float2(c, a);
    results[base + 9u] = as_type<uint>(pair.y);
    float4 lanes = float4(a) * b + float4(c);
    results[base + 10u] = as_type<uint>(lanes.w);
    float2 splat = a * b + float2(c);
    results[base + 11u] = as_type<uint>(splat.x);
    results[base + 12u] = as_type<uint>(c + a * b * d);
    results[base + 13u] = as_type<uint>(0.1f * 3.0f + c);
    half narrow = half(a) * half(b) + half(c);
    results[base + 14u] = uint(as_type<ushort>(narrow));
    bfloat brain = bfloat(a) * bfloat(b) + bfloat(c);
    results[base + 15u] = uint(as_type<ushort>(brain));
    half accumulated = half(c);
    accumulated += half(a) * half(b);
    results[base + 16u] = uint(as_type<ushort>(accumulated));
    results[base + 17u] = as_type<uint>(c + i * a);
    // Casts to the product's own type emit nothing, so Metal still fuses.
    results[base + 18u] = as_type<uint>(static_cast<float>(a * b) + c);
    results[base + 19u] = as_type<uint>(c - float(a * d));
}
"""


def _source_options(profile):
    return {
        "floating_contraction_profile": profile,
        # Exact separate rounding keeps uncontracted pairs out of the target
        # compiler's own contraction, so every field has one exact result.
        "binary32_multiplication_profile": profile,
        "binary32_additive_profile": profile,
    }


def _translate(tmp_path, source=SOURCE, target="crossgl", options=None):
    path = tmp_path / "contraction.metal"
    path.write_text(source, encoding="utf-8")
    return translate(
        str(path),
        backend=target,
        format_output=False,
        source_options=(
            {"floating_contraction_profile": PROFILES[1]}
            if options is None
            else options
        ),
    )


def _function_bodies(generated):
    return {
        line.strip()
        for line in generated.splitlines()
        if line.strip().startswith(("return", "acc"))
    }


SELECTION = """
float f_expr(float a, float b, float c) { return a * b + c; }
float f_right(float a, float b, float c) { return c + a * b; }
float f_sub_left(float a, float b, float c) { return a * b - c; }
float f_sub_right(float a, float b, float c) { return c - a * b; }
float f_two(float a, float b, float c, float d) { return a * b + c * d; }
float f_neg(float a, float b, float c) { return -(a * b) + c; }
float f_stmt(float a, float b, float c) { float p = a * b; return p + c; }
float f_compound(float a, float b, float c) { float acc = c; acc += a * b; acc -= b * c; return acc; }
float f_const(float c) { return 0.1f * 3.0f + c; }
float4 f_vec(float4 a, float b, float4 c) { return a * b + c; }
float2 f_splat(float a, float b, float2 c) { return a * b + c; }
float f_int(int a, int b, float c) { return a * b + c; }
float f_uint(uint i, float s, float c) { return c + i * s; }
half h_expr(half a, half b, half c) { return a * b + c; }
half h_uint(uint i, half s, half c) { return c + i * s; }
bfloat b_expr(bfloat a, bfloat b, bfloat c) { return a * b + c; }
bfloat b_uint(uint i, bfloat s, bfloat c) { return c + i * s; }
float m_half(half a, half b, float c) { return a * b + c; }
float f_chain(float a, float b, float c, float d) { return c + a * b * d; }
float f_cast(float a, float b, float c) { return static_cast<float>(a * b) + c; }
float f_functional(float a, float b, float c) { return c - float(a * b); }
float f_widened(half a, half b, float c) { return float(a * b) + c; }
"""


def test_contraction_selects_the_pairs_metal_fuses(tmp_path):
    bodies = _function_bodies(_translate(tmp_path, SELECTION))
    fused = "__crossgl_metal_fmuladd_float"
    for expected in (
        f"return {fused}(float(a), float(b), float(c));",
        f"return {fused}(float(a), float(b), -float(c));",
        f"return {fused}(-float(a), float(b), float(c));",
        # The left product is fused; the right product is rounded first.
        f"return {fused}(float(a), float(b), float(c * d));",
        f"acc = {fused}(float(a), float(b), float(acc));",
        f"acc = {fused}(-float(b), float(c), float(acc));",
        f"return {fused}4(vec4(a), vec4(b), vec4(c));",
        f"return {fused}(float(i), float(s), float(c));",
        f"return {fused}(float(a * b), float(d), float(c));",
        f"return {fused}(-float(a), float(b), float(c));",
        # binary16 products are exact in binary32 and round once.
        "return float16(float(a) * float(b) + float(c));",
        "return float16(float(float16(i)) * float(s) + float(c));",
        # Metal fuses bfloat arithmetic in binary32 before one bfloat rounding.
        f"return bfloat16({fused}(float(a), float(b), float(c)));",
        f"return bfloat16({fused}(float(bfloat16(i)), float(s), float(c)));",
    ):
        assert expected in bodies, expected
    for unchanged in (
        "return (-(a * b)) + c;",
        "return p + c;",
        "return 0.1f * 3.0f + c;",
        "return a * b + c;",
    ):
        assert unchanged in bodies, unchanged
    # Integer products, widened half products and splatted scalar products stay
    # separate operations, as in Metal.
    generated = _translate(tmp_path, SELECTION)
    for name in ("f_int", "m_half", "f_splat"):
        body = generated[generated.index(name) :].split("}", 1)[0]
        assert "return a * b + c;" in body, name
    for name in ("f_cast", "f_functional"):
        body = generated[generated.index(name) :].split("}", 1)[0]
        assert fused + "(" in body, name
    widened = generated[generated.index("f_widened") :].split("}", 1)[0]
    assert "fmuladd" not in widened


def test_contraction_is_explicit_and_resets(tmp_path):
    assert "fmuladd" not in _translate(tmp_path, SELECTION, options={})
    converter = MetalToCrossGLConverter(floating_contraction_profile=PROFILES[0])
    for source, expected in (
        (SELECTION, True),
        ("float value(float a, float b) { return a * b; }", False),
    ):
        tree = MetalParser(MetalLexer(source).tokenize()).parse()
        assert ("__crossgl_fma_bits" in converter.generate(tree)) is expected


@pytest.mark.parametrize("profile", (True, 1, "", "flush-subnormals", "fast", [], {}))
def test_contraction_profile_rejects_invalid_configuration(profile):
    with pytest.raises(ValueError, match="floating_contraction_profile"):
        MetalToCrossGLConverter(floating_contraction_profile=profile)


@pytest.mark.parametrize("profile", PROFILES)
def test_contraction_helper_selects_subnormal_mode(tmp_path, profile):
    generated = _translate(
        tmp_path, SELECTION, options={"floating_contraction_profile": profile}
    )
    flush = "true" if profile == "rne-flush" else "false"
    assert f"asuint(a), asuint(b), asuint(c), {flush}));" in generated
    assert generated.count("uint __crossgl_fma_bits(") == 1


def test_contraction_is_independent_of_explicit_fma_profile(tmp_path):
    generated = _translate(
        tmp_path,
        "float value(float a, float b, float c) { return fma(a, b, c) + a * b; }",
        options={
            "floating_contraction_profile": PROFILES[0],
            "binary32_fma_profile": PROFILES[1],
        },
    )
    assert "__crossgl_metal_fma_float(float(a), float(b), float(c))" in generated
    assert "asuint(a), asuint(b), asuint(c), true));" in generated
    assert "asuint(a), asuint(b), asuint(c), false));" in generated
    assert generated.count("uint __crossgl_fma_bits(") == 1


@pytest.mark.parametrize(
    "source,reason",
    (
        (
            "float value(device float* v, uint i, float a, float b) "
            "{ v[i++] += a * b; return v[0]; }",
            "side-effect-free",
        ),
        (
            "half value(float a, float b) { half h = 0.0h; h += a * b; return h; }",
            "conversion",
        ),
        (
            "using Wide = metal::vec<float, 8>;\n"
            "Wide value(Wide a, Wide b, Wide c) { return a * b + c; }",
            "between one and four lanes",
        ),
    ),
)
def test_contraction_diagnoses_unrepresentable_contexts(tmp_path, source, reason):
    with pytest.raises(MetalContractionProfileError, match=reason) as error:
        _translate(tmp_path, source)
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-contraction-profile-unsupported"
    )
    assert error.value.profile == PROFILES[1]


def test_contraction_keeps_operator_ownership(tmp_path):
    source = """
struct Value { float x; };
Value crosstl_metal_operator_multiply__Value__Value(Value a, Value b) {
    Value c; c.x = a.x * b.x; return c;
}
Value crosstl_metal_operator_add__Value__Value(Value a, Value b) {
    Value c; c.x = a.x - b.x; return c;
}
Value custom(Value a, Value b, Value c) { return a * b + c; }
float global_scale = 2.0f * 3.0f + 1.0f;
"""
    generated = _translate(tmp_path, source)
    # Only the arithmetic inside the user-defined operators uses binary32.
    assert "fmuladd" not in generated
    assert generated == _translate(tmp_path, source, options={})


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_contraction_profiles_compile(tmp_path, profile, target):
    generated = _translate(tmp_path, options=_source_options(profile), target=target)
    assert "crossgl_metal_fmuladd_float" in generated
    assert "Berkeley SoftFloat" in generated
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


def test_contraction_profile_report_provenance(tmp_path):
    (tmp_path / "contraction.metal").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "crosstl.toml").write_text(
        """[project]
targets = ["directx", "opengl", "metal"]
[project.source_options.metal.target_options.directx]
floating_contraction_profile = "rne-gradual"
[project.source_options.metal.target_options.opengl.source_patterns."contraction.metal"]
floating_contraction_profile = "rne-flush"
""",
        encoding="utf-8",
    )
    report = translate_project(load_project_config(tmp_path), format_output=False)
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 3, data["diagnostics"]
    expected = {"directx": PROFILES[0], "opengl": PROFILES[1], "metal": None}
    assert {
        a["target"]: a["provenance"].get("floatingContractionProfile")
        for a in data["artifacts"]
    } == expected
    path = tmp_path / "report.json"
    report.write_json(path)
    assert validate_project_report(path)["success"]
    manifest = build_runtime_artifact_manifest(path)
    assert manifest["success"], manifest
    assert {
        a["target"]: a["provenance"].get("floatingContractionProfile")
        for a in manifest["artifacts"]
    } == expected
    manifest_path = tmp_path / "artifacts.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert build_runtime_package(manifest_path, tmp_path / "package")["success"]
    directx = next(
        index
        for index, artifact in enumerate(data["artifacts"])
        if artifact["target"] == "directx"
    )
    for invalid in (None, PROFILES[1], "fast"):
        provenance = data["artifacts"][directx]["provenance"]
        if invalid is None:
            provenance.pop("floatingContractionProfile", None)
        else:
            provenance["floatingContractionProfile"] = invalid
        path.write_text(json.dumps(data), encoding="utf-8")
        result = validate_project_report(path)
        assert not result["success"]
        assert "floatingContractionProfile" in json.dumps(result["diagnostics"])


def _flush(bits):
    return bits & 0x80000000 if bits & 0x7F800000 == 0 else bits


def _multiply(a, b, flush):
    # Adding negative zero preserves every product, including signed zeros.
    return _fma(a, b, NEGATIVE_ZERO, flush)


def _add(a, b, flush):
    return _fma(a, ONE, b, flush)


def _negate(bits):
    return bits ^ 0x80000000


def _f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _half_from_f32(bits):
    """Round binary32 bits to binary16 bits with ties to even."""
    value = _f32(bits)
    if math.isnan(value):
        return 0x7E00
    return _half_round(value)


def _half_round(value):
    sign = 0x8000 if math.copysign(1.0, value) < 0 else 0
    if math.isinf(value):
        return sign | 0x7C00
    if math.isnan(value):
        return 0x7E00
    magnitude = Fraction(abs(value))
    if magnitude == 0:
        return sign
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    quantum = Fraction(2) ** (max(exponent, -14) - 10)
    scaled = magnitude / quantum
    rounded, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    if twice > scaled.denominator or (twice == scaled.denominator and rounded & 1):
        rounded += 1
    result = rounded * quantum
    if result >= 65536:
        return sign | 0x7C00
    return sign | struct.unpack("<H", struct.pack("<e", float(result)))[0]


def _half_value(word):
    return struct.unpack("<e", struct.pack("<H", word))[0]


def _half_fused(a, b, c):
    # The binary16 product is exact in binary64, and one binary64 rounding of
    # the sum before binary16 rounding cannot change the result.
    x, y, z = (_half_value(_half_from_f32(value)) for value in (a, b, c))
    with_product = x * y
    if math.isnan(with_product) or math.isnan(z):
        return 0x7E00
    if math.isinf(with_product) and math.isinf(z) and with_product != z:
        return 0x7E00
    return _half_round(with_product + z)


def _bfloat(bits):
    if (bits & 0x7F800000) == 0x7F800000 and bits & 0x007FFFFF:
        return (bits >> 16) | 0x40
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def _expected_fields(a, b, c, d, index, flush):
    product = _multiply(a, b, flush)
    narrow = _half_fused(a, b, c)
    return [
        _fma(a, b, c, flush),
        _fma(a, b, c, flush),
        _fma(a, b, _negate(c), flush),
        _fma(_negate(a), b, c, flush),
        _fma(a, b, _multiply(c, d, flush), flush),
        _add(_negate(product), c, flush),
        _add(product, c, flush),
        _fma(a, b, c, flush),
        _fma(_negate(a), b, c, flush),
        _fma(c, d, a, flush),
        _fma(a, b, c, flush),
        _add(product, c, flush),
        _fma(product, d, c, flush),
        _add(_multiply(_f32_bits(0.1), _f32_bits(3.0), flush), c, flush),
        narrow,
        _bfloat(_fma(_bfloat(a) << 16, _bfloat(b) << 16, _bfloat(c) << 16, flush)),
        narrow,
        _fma(_f32_bits(float(index)), a, c, flush),
        _fma(a, b, c, flush),
        _fma(_negate(a), d, c, flush),
    ]


def _inputs():
    generator = random.Random(2193)
    edges = (0, NEGATIVE_ZERO, 1, 0x807FFFFF, 0x800000, 0x80800000, ONE, 0xBF800000)
    values = [
        (a, b, c, d) for a in edges for b in edges for c in edges for d in edges[::3]
    ]

    def finite(low, high):
        exponent = generator.randint(low, high) + 127
        sign = generator.getrandbits(1) << 31
        if exponent <= 0:
            # Subnormal binary32 values carry their scale in the fraction.
            return sign | max(1, generator.getrandbits(23) >> (1 - exponent))
        return sign | (exponent << 23) | generator.getrandbits(23)

    for _ in range(1024):
        values.append(tuple(finite(-20, 20) for _ in range(4)))
    for _ in range(512):
        # Products near the addend magnitude exercise cancellation.
        a, b = finite(-6, 6), finite(-6, 6)
        exact = _f32(a) * _f32(b)
        c = _f32_bits(-exact) ^ generator.getrandbits(3)
        values.append((a, b, c, finite(-6, 6)))
    for _ in range(512):
        # Exact results near the binary32 subnormal boundary separate modes.
        values.append(
            (finite(-70, -60), finite(-70, -60), finite(-130, -120), finite(-4, 4))
        )
    for _ in range(128):
        values.append((finite(100, 127), finite(100, 127), finite(-4, 4), ONE))
    return values


def _is_nan(bits, width):
    if width == 16:
        return (bits & 0x7FFF) > 0x7C00
    return (bits & 0x7FFFFFFF) > 0x7F800000


@pytest.mark.parametrize("profile", (*PROFILES, "source"))
def test_contraction_profiles_execute(tmp_path, profile, monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native contraction policies")
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
    inputs = _inputs()
    flush = profile != PROFILES[0]
    expected = [GUARD] * 4
    for index, values in enumerate(inputs):
        expected.extend(_expected_fields(*values, index, flush))
    expected.extend([GUARD] * 4)
    source = (
        SOURCE
        if profile == "source"
        else _translate(tmp_path, options=_source_options(profile), target=target)
    )
    (tmp_path / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    actual, evidence = _dispatch(
        tmp_path,
        target,
        source,
        inputs,
        len(expected),
        entry="contraction_profile" if target == "metal" else None,
        initial_output=[GUARD] * len(expected),
    )
    assert len(actual) == len(expected)
    differences = []
    for position, (want, got) in enumerate(zip(expected, actual)):
        field = (position - 4) % FIELDS
        width = (
            16 if 4 <= position < len(expected) - 4 and field in {14, 15, 16} else 32
        )
        if field == 15 and width == 16:
            both_nan = (want & 0x7FFF) > 0x7F80 and (got & 0x7FFF) > 0x7F80
        else:
            both_nan = _is_nan(want, width) and _is_nan(got, width)
        if want != got and not both_nan:
            differences.append(
                {
                    "index": position,
                    "field": field,
                    "input": (
                        inputs[(position - 4) // FIELDS]
                        if 4 <= position < len(expected) - 4
                        else None
                    ),
                    "expected": want,
                    "actual": got,
                }
            )
    evidence.update(
        profile=profile,
        inputCount=len(inputs),
        fieldCount=FIELDS,
        guardCount=8,
        mismatchCount=len(differences),
        mismatches=differences[:200],
        nanComparison="classification only; payloads are not represented",
    )
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    assert not differences, differences[:20]


def test_ci_requires_contraction_profiles_in_existing_native_job():
    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate binary32 arithmetic"
    )
    assert_native_step_prerequisites(step, job="portable-host")
    assert f'{REQUIRE_ENV}: "1"' in step
    assert (
        workflow.count("tests/test_translator/test_metal_contraction_profile.py") == 1
    )
    assert "tests/test_translator/test_metal_contraction_profile.py" in step
    # The step budget covers the slowest Windows runner observed (about 1.9x).
    assert "--timeout-seconds 480" in step
