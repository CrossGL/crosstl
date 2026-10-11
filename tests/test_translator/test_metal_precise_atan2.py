"""Precise two-argument arctangent with explicit source underflow policy."""

import json
import math
import os
import random
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
    _bits,
    _float,
    _pi,
    _round_decimal,
)

PROFILES = ("preserve-subnormals", "flush-subnormals")
REQUIRE_ENV = "CROSTL_REQUIRE_METAL_PRECISE_ATAN2"
GUARD = 0x35A5B6C7
FIELDS = 23
SOURCE = """#include <metal_stdlib>
using namespace metal;
float record(thread uint& count, float value) { count += 1; return value; }
float atan2(float y, float x) { return y; }
kernel void angles(const device uint* values [[buffer(0)]],
                   device uint* results [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    float y = as_type<float>(values[2u * i]);
    float x = as_type<float>(values[2u * i + 1u]);
    uint count = 0;
    float angle = metal::precise::atan2(record(count, y), record(count, x));
    float2 pair = metal::precise::atan2(float2(y, x), float2(x, y));
    float3 triple = precise::atan2(float3(y, y, -y), float3(x, -x, x));
    float4 quad = metal::precise::atan2(float4(y, x, -y, -x), float4(x, y, x, y));
    float2 broadcast2 = precise::atan2(record(count, y), float2(x, -x));
    float3 broadcast3 = precise::atan2(float3(y, y, -y), record(count, x));
    float4 broadcast4 = precise::atan2(record(count, y), float4(x, -x, x, -x));
    results[4u + 23u * i] = as_type<uint>(y);
    results[5u + 23u * i] = as_type<uint>(x);
    results[6u + 23u * i] = as_type<uint>(angle);
    results[7u + 23u * i] = as_type<uint>(pair.x);
    results[8u + 23u * i] = as_type<uint>(pair.y);
    results[9u + 23u * i] = as_type<uint>(triple.x);
    results[10u + 23u * i] = as_type<uint>(triple.y);
    results[11u + 23u * i] = as_type<uint>(triple.z);
    results[12u + 23u * i] = as_type<uint>(quad.x);
    results[13u + 23u * i] = as_type<uint>(quad.y);
    results[14u + 23u * i] = as_type<uint>(quad.z);
    results[15u + 23u * i] = as_type<uint>(quad.w);
    results[16u + 23u * i] = as_type<uint>(broadcast2.x);
    results[17u + 23u * i] = as_type<uint>(broadcast2.y);
    results[18u + 23u * i] = as_type<uint>(broadcast3.x);
    results[19u + 23u * i] = as_type<uint>(broadcast3.y);
    results[20u + 23u * i] = as_type<uint>(broadcast3.z);
    results[21u + 23u * i] = as_type<uint>(broadcast4.x);
    results[22u + 23u * i] = as_type<uint>(broadcast4.y);
    results[23u + 23u * i] = as_type<uint>(broadcast4.z);
    results[24u + 23u * i] = as_type<uint>(broadcast4.w);
    results[25u + 23u * i] = count;
    results[26u + 23u * i] = as_type<uint>(::atan2(y, x));
}
"""

PRECISION_SCOPE_SOURCE = """#include <metal_stdlib>
using namespace metal;
float product(float a, float b) { return a * b; }
float difference(float a, float b, float c, float d) {
    return product(a, b) - product(c, d);
}
kernel void scope(const device uint* values [[buffer(0)]],
                 device uint* results [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
    float a = as_type<float>(values[4u*i]);
    float b = as_type<float>(values[4u*i+1u]);
    float c = as_type<float>(values[4u*i+2u]);
    float d = as_type<float>(values[4u*i+3u]);
    results[4u+2u*i] = as_type<uint>(difference(a, b, c, d));
    results[5u+2u*i] = as_type<uint>(metal::precise::atan2(a, b));
}
"""


def _translate(tmp_path, source=SOURCE, target="crossgl", profile=None, **options):
    path = tmp_path / "atan2.metal"
    path.write_text(source, encoding="utf-8")
    return translate(
        str(path),
        backend=target,
        format_output=False,
        source_options={"binary32_atan2_profile": profile, **options},
    )


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_precise_atan2_helpers_compile(tmp_path, target, profile):
    generated = _translate(tmp_path, target=target, profile=profile)
    for suffix in ("", "2", "3", "4"):
        assert f"metal_precise_atan2_float{suffix}(" in generated
    assert "metal_precise_atan_float(" in generated
    assert "crossgl_divide_bits(" in generated
    assert "double" not in generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-fno-fast-math",))


def test_precise_atan2_does_not_set_file_scope_contraction(tmp_path):
    generated = _translate(tmp_path, source=PRECISION_SCOPE_SOURCE, target="metal")
    assert generated.count("    #pragma clang fp contract(off)") == 2
    assert "\n#pragma clang fp contract(" not in generated
    assert "#pragma clang fp contract(fast)" not in generated
    product = generated.split("float product(", 1)[1].split("}", 1)[0]
    assert "#pragma" not in product


def test_ci_requires_metal_contraction_scope_in_existing_native_step():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned arctangent"
    )
    selector = "tests/test_translator/test_metal_precise_atan2.py::test_precise_helpers_preserve_enclosing_metal_contraction"
    assert workflow.count(selector) == 1 and selector in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "--timeout-seconds 180" in step
    assert "continue-on-error" not in step and "if:" not in step


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_precise_atan2_profile_survives_saved_crossgl(tmp_path, profile, target):
    intermediate = tmp_path / "saved.cgl"
    intermediate.write_text(_translate(tmp_path, profile=profile), encoding="utf-8")
    restored = translate(str(intermediate), backend=target, format_output=False)
    assert restored == _translate(tmp_path, target=target, profile=profile)


def test_precise_atan2_default_preserves_subnormals(tmp_path):
    assert _translate(tmp_path) == _translate(tmp_path, profile=PROFILES[0])


def test_precise_atan2_modes_and_source_functions_stay_separate(tmp_path):
    generated = _translate(
        tmp_path,
        """
        float atan2(float y, float x) { return y; }
        float explicit_mode(float y, float x) { return metal::precise::atan2(y, x); }
        float imported_mode(float y, float x) { return precise::atan2(y, x); }
        float default_mode(float y, float x) { return metal::atan2(y, x); }
        float fast_mode(float y, float x) { return metal::fast::atan2(y, x); }
        float user_mode(float y, float x) { return ::atan2(y, x); }
    """,
    )
    assert generated.count("return __crossgl_metal_precise_atan2_float(y, x);") == 2
    assert generated.count("return atan2(y, x);") == 2
    assert "return atan2__metal_overload_1(y, x);" in generated


def test_precise_atan2_helpers_reset_and_avoid_source_names():
    converter = MetalToCrossGLConverter()
    source = """
        float __crossgl_metal_precise_atan2_float(float y) { return y; }
        float2 __crossgl_metal_precise_atan2_float2(float2 y) { return y; }
        uint __crossgl_divide_bits(uint y) { return y; }
        float2 evaluate(float2 y, float2 x) { return precise::atan2(y, x); }
    """
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "float __crossgl_metal_precise_atan2_float_(float y, float x)" in generated
    assert "return __crossgl_metal_precise_atan2_float2_(y, x);" in generated
    assert (
        "uint __crossgl_divide_bits_(uint a, uint b, bool flush_denormals)" in generated
    )
    source = "float value(float y, float x) { return metal::atan2(y, x); }"
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "__crossgl_metal_precise_atan" not in generated
    assert "__crossgl_divide_bits" not in generated


@pytest.mark.parametrize("profile", (True, 1, "", "rne-flush", [], {}))
def test_atan2_profile_rejects_invalid_configuration(profile):
    with pytest.raises(ValueError, match="binary32_atan2_profile"):
        MetalToCrossGLConverter(binary32_atan2_profile=profile)


@pytest.mark.parametrize("operand", ("int", "double", "float8", "half2", "Payload"))
def test_precise_atan2_diagnoses_unsupported_operands(tmp_path, operand):
    source = (
        "struct Payload { float x; };\n"
        + f"{operand} f({operand} y, {operand} x) {{ return precise::atan2(y, x); }}"
    )
    with pytest.raises(MetalPreciseMathLoweringError) as error:
        _translate(tmp_path, source)
    assert error.value.operation == "atan2"
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-precise-math-unsupported"
    )


@pytest.mark.parametrize(
    "parameters,arguments", (("float y", "y"), ("float2 y, float3 x", "y, x"))
)
def test_precise_atan2_requires_two_matching_shapes(tmp_path, parameters, arguments):
    with pytest.raises(MetalPreciseMathLoweringError):
        _translate(
            tmp_path, f"float f({parameters}) {{ return precise::atan2({arguments}); }}"
        )


def test_precise_atan2_diagnoses_global_runtime_initialization(tmp_path):
    with pytest.raises(MetalPreciseMathLoweringError, match="global initializers"):
        _translate(tmp_path, "constant float angle = precise::atan2(1.0f, 1.0f);")


@pytest.mark.parametrize("operand", ("half", "bfloat"))
def test_precise_atan2_promotes_narrow_scalars(tmp_path, operand):
    generated = _translate(
        tmp_path,
        f"""
        float evaluate({operand} y, {operand} x) {{ return precise::atan2(y, x); }}
    """,
    )
    assert "__crossgl_metal_precise_atan2_float(" in generated
    assert "float evaluate" in generated


@pytest.mark.parametrize("width", (2, 3, 4))
@pytest.mark.parametrize("scalar_index", (0, 1))
def test_precise_atan2_broadcasts_scalar_arguments(tmp_path, width, scalar_index):
    types = [f"float{width}"] * 2
    types[scalar_index] = "float"
    generated = _translate(
        tmp_path,
        f"""
        float{width} evaluate({types[0]} y, {types[1]} x) {{
            return precise::atan2(y, x);
        }}
    """,
    )
    operands = ["y", "x"]
    operands[scalar_index] = f"vec{width}({operands[scalar_index]})"
    assert (
        f"return __crossgl_metal_precise_atan2_float{width}({', '.join(operands)});"
        in generated
    )


def test_atan2_report_retains_resolved_profile_and_rejects_mutation(tmp_path):
    (tmp_path / "atan2.metal").write_text(SOURCE)
    (tmp_path / "crosstl.toml").write_text("""[project]
targets = ["directx", "opengl"]
[project.source_options.metal]
binary32_atan2_profile = "preserve-subnormals"
[project.source_options.metal.target_options.opengl.source_patterns."atan2.metal"]
binary32_atan2_profile = "flush-subnormals"
""")
    report = translate_project(load_project_config(tmp_path), format_output=False)
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 2, data["diagnostics"]
    expected = dict(zip(("directx", "opengl"), PROFILES))
    assert {
        a["target"]: a["provenance"]["binary32Atan2Profile"] for a in data["artifacts"]
    } == expected
    path = tmp_path / "report.json"
    report.write_json(path)
    assert validate_project_report(path)["success"]
    manifest = build_runtime_artifact_manifest(path)
    assert manifest["success"], manifest
    assert {
        a["target"]: a["provenance"]["binary32Atan2Profile"]
        for a in manifest["artifacts"]
    } == expected
    manifest_path = tmp_path / "artifacts.json"
    manifest_path.write_text(json.dumps(manifest))
    assert build_runtime_package(manifest_path, tmp_path / "package")["success"]
    for invalid in (None, PROFILES[1], "rne-flush", False):
        changed = deepcopy(data)
        if invalid is None:
            changed["artifacts"][0]["provenance"].pop("binary32Atan2Profile")
        else:
            changed["artifacts"][0]["provenance"]["binary32Atan2Profile"] = invalid
        path.write_text(json.dumps(changed))
        validation = validate_project_report(path)
        assert not validation["success"]
        assert "binary32Atan2Profile" in json.dumps(validation["diagnostics"])


def _profile_word(word, profile):
    if profile == "flush-subnormals" and word & 0x7F800000 == 0:
        return word & 0x80000000
    return word


@lru_cache(maxsize=None)
def _oracle(y, x, profile):
    y, x = (_profile_word(word, profile) for word in (y, x))
    my, mx = y & 0x7FFFFFFF, x & 0x7FFFFFFF
    sign = y & 0x80000000
    negative_x = x >> 31
    if max(my, mx) > 0x7F800000:
        return 0x7FC00000
    if my == 0:
        return sign | (0x40490FDB if negative_x else 0)
    if mx == 0:
        return sign | 0x3FC90FDB
    if my == 0x7F800000:
        return sign | (
            (0x4016CBE4 if negative_x else 0x3F490FDB) if mx == my else 0x3FC90FDB
        )
    if mx == 0x7F800000:
        return sign | (0x40490FDB if negative_x else 0)
    with localcontext() as context:
        context.prec = 180
        ratio = Decimal.from_float(_float(my)) / Decimal.from_float(_float(mx))
        invert = ratio > 1
        if invert:
            ratio = 1 / ratio
        # Half-angle identities provide an independent reference to emitted pi/4 reduction.
        for _ in range(2):
            ratio = ratio / (1 + (1 + ratio * ratio).sqrt())
        term = total = ratio
        for index in range(1, 256):
            term *= -ratio * ratio
            addend = term / (2 * index + 1)
            total += addend
            if abs(addend) < Decimal("1e-160"):
                break
        else:
            raise AssertionError("atan2 reference did not converge")
        angle = 4 * total
        if invert:
            angle = _pi() / 2 - angle
        if negative_x:
            angle = _pi() - angle
        result = _round_decimal(angle) | sign
        if profile == "flush-subnormals" and result & 0x7F800000 == 0:
            return 0
        return result


def _underflow_boundary_pairs():
    for exponent in range(1, 128):
        y = exponent << 23 | 0x7FFFFF
        x = (exponent + 127) << 23
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for sign in (0, 0x80000000):
                    yield (y + dy) | sign, x + dx


def _pairs():
    edges = (
        0,
        1,
        2,
        0x7FFFFF,
        0x800000,
        0x3F000000,
        0x3F800000,
        0x7F7FFFFF,
        0x7F800000,
        0x7FC01234,
    )
    edges = tuple(word | sign for word in edges for sign in (0, 0x80000000))
    pairs = [(y, x) for y in edges for x in edges]
    rng = random.Random(2118)
    for exponent in range(255):
        word = exponent << 23 | rng.randrange(1 << 23)
        for other in (0x3F800000, 0x3F800001):
            for sign in (0, 0x80000000):
                pairs.extend(((word | sign, other), (other, word | sign)))
    center = _bits(math.sqrt(2) - 1)
    pairs.extend((word, 0x3F800000) for word in range(center - 16, center + 17))
    pairs.extend((rng.getrandbits(32), rng.getrandbits(32)) for _ in range(512))
    pairs.extend(_underflow_boundary_pairs())
    return pairs


def _operands(y, x):
    ny, nx = y ^ 0x80000000, x ^ 0x80000000
    return (
        (y, x),
        (y, x),
        (x, y),
        (y, x),
        (y, nx),
        (ny, x),
        (y, x),
        (x, y),
        (ny, x),
        (nx, y),
        (y, x),
        (y, nx),
        (y, x),
        (y, x),
        (ny, x),
        (y, x),
        (y, nx),
        (y, x),
        (y, nx),
    )


def _expected(pairs, profile):
    words = [GUARD] * 4
    for y, x in pairs:
        words.extend((y, x))
        words.extend(_oracle(a, b, profile) for a, b in _operands(y, x))
        words.extend((5, y))
    return words + [GUARD] * 4


def _check(actual, pairs, profile):
    assert len(actual) == FIELDS * len(pairs) + 8, "output size"
    assert all(type(word) is int and 0 <= word <= 0xFFFFFFFF for word in actual)
    assert actual[:4] == actual[-4:] == [GUARD] * 4, "output guards"
    maximum = 0
    for i, (y, x) in enumerate(pairs):
        row = actual[4 + FIELDS * i : 4 + FIELDS * (i + 1)]
        assert row[:2] == [y, x], "operand bits"
        assert row[-2] == 5, "operand evaluation count"
        assert row[-1] == y, "source overload result"
        for got, (a, b) in zip(row[2:-2], _operands(y, x)):
            want = _oracle(a, b, profile)
            if want & 0x7FFFFFFF > 0x7F800000:
                assert got & 0x7FFFFFFF > 0x7F800000, "NaN classification"
                continue
            assert got >> 31 == want >> 31, "result sign"
            magnitudes = [_profile_word(word, profile) & 0x7FFFFFFF for word in (a, b)]
            if want & 0x7FFFFFFF == 0 or any(
                value in (0, 0x7F800000) for value in magnitudes
            ):
                assert got == want, "axis or zero result"
            error = abs(got - want)
            # MSL specification, Table 8.1: binary32 atan2 has a six-ULP contract.
            assert error <= 6, (i, hex(a), hex(b), hex(got), hex(want), error)
            maximum = max(maximum, error)
    return maximum


def test_atan2_oracle_profiles_and_input_coverage():
    pairs = _pairs()
    assert len(pairs) == 5271
    assert {y >> 23 & 255 for y, _ in pairs} == set(range(256))
    assert _oracle(1, 1, PROFILES[0]) == 0x3F490FDB
    assert _oracle(1, 1, PROFILES[1]) == 0
    assert _oracle(0x80000001, 1, PROFILES[1]) == 0x80000000
    assert _oracle(0x800000, 0x40000000, PROFILES[0]) == 0x400000
    assert _oracle(0x800000, 0x40000000, PROFILES[1]) == 0
    assert _oracle(0x80800000, 0x40000000, PROFILES[1]) == 0
    assert _oracle(0x80000000, 0x40000000, PROFILES[1]) == 0x80000000
    for a, b in pairs[:400]:
        if not math.isnan(_float(a)) and not math.isnan(_float(b)):
            assert _oracle(a, b, PROFILES[0]) == _bits(math.atan2(_float(a), _float(b)))


def test_atan2_underflow_midpoints_and_adjacent_normal_values():
    from fractions import Fraction

    midpoint = Fraction(2**24 - 1, 2**150)
    pairs = list(_underflow_boundary_pairs())
    assert len(pairs) == len(set(pairs)) == 2286
    for exponent in range(1, 128):
        y = exponent << 23 | 0x7FFFFF
        x = (exponent + 127) << 23
        assert Fraction(_float(y)) / Fraction(_float(x)) == midpoint
        for sign in (0, 0x80000000):
            assert _oracle(y | sign, x, PROFILES[0]) == sign | 0x7FFFFF
            assert _oracle(y | sign, x, PROFILES[1]) == 0
            assert _oracle((y + 1) | sign, x, PROFILES[1]) == sign | 0x800000
            assert _oracle(y | sign, x - 1, PROFILES[1]) == sign | 0x800000


def test_atan2_verifier_rejects_rounding_the_underflow_midpoint_upward():
    pairs = [(0x00FFFFFF, 0x40000000), (0x80FFFFFF, 0x40000000)]
    expected = _expected(pairs, PROFILES[1])
    assert _check(expected, pairs, PROFILES[1]) == 0
    for index, word in ((6, 0x800000), (6 + FIELDS, 0x80800000)):
        changed = list(expected)
        changed[index] = word
        with pytest.raises(AssertionError):
            _check(changed, pairs, PROFILES[1])


@pytest.mark.parametrize(
    "fault",
    (
        "size",
        "guard",
        "count",
        "input",
        "overload",
        "sign",
        "value",
        "nan",
        "subnormal",
    ),
)
def test_atan2_verifier_rejects_corruption(fault):
    pairs = [
        (0x80000000, 0x3F800000),
        (0x3F800000, 0x3F800000),
        (0x7FC00000, 0),
        (0x1000, 0x3F800000),
    ]
    actual = _expected(pairs, PROFILES[0])
    assert _check(actual, pairs, PROFILES[0]) == 0
    if fault == "size":
        actual.pop()
    elif fault == "guard":
        actual[-1] ^= 1
    elif fault == "count":
        actual[4 + FIELDS - 2] = 6
    elif fault == "input":
        actual[4] ^= 1
    elif fault == "overload":
        actual[4 + FIELDS - 1] = 0
    elif fault == "sign":
        actual[6] = 0
    elif fault == "value":
        actual[6 + FIELDS] += 7
    elif fault == "nan":
        actual[6 + 2 * FIELDS] = 0
    else:
        actual[6 + 3 * FIELDS] = 0
    with pytest.raises(AssertionError):
        _check(actual, pairs, PROFILES[0])


def test_atan2_flush_profile_keeps_axis_and_underflow_zero_checks_distinct():
    pairs = [(0x80000000, 0x40000000), (0x80800000, 0x40000000)]
    expected = _expected(pairs, PROFILES[1])
    assert expected[6] == 0x80000000 and expected[6 + FIELDS] == 0
    for index in (6, 6 + FIELDS):
        changed = list(expected)
        changed[index] ^= 0x80000000
        with pytest.raises(AssertionError, match="result sign"):
            _check(changed, pairs, PROFILES[1])


@pytest.mark.parametrize("profile", (*PROFILES, "source"))
def test_precise_atan2_executes(tmp_path, profile, monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native precise atan2")
    from tests.test_translator import test_fused_math as native_math

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    if profile == "source" and target != "metal":
        pytest.skip("The unchanged source control requires Metal")
    monkeypatch.setattr(
        native_math,
        "_compile",
        partial(_compile, metal_compile_flags=("-fno-fast-math",)),
    )
    policy = PROFILES[1] if profile == "source" else profile
    pairs = _pairs()
    expected = _expected(pairs, policy)
    source = (
        SOURCE
        if profile == "source"
        else _translate(tmp_path, target=target, profile=profile)
    )
    actual, evidence = native_math._dispatch(
        tmp_path,
        target,
        source,
        pairs,
        len(expected),
        entry="angles" if target == "metal" else None,
        initial_output=[GUARD] * len(expected),
    )
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    maximum = _check(actual, pairs, policy)
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                **evidence,
                "profile": profile,
                "comparisonProfile": policy,
                "pairCount": len(pairs),
                "angleCount": (FIELDS - 4) * len(pairs),
                "guardCount": 8,
                "maximumUlpError": maximum,
                "ulpLimit": 6,
                "oracle": "180-digit Decimal half-angle atan2, binary32 nearest-even",
                "unchangedSourceControl": profile == "source",
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("contraction", ("off", "on", "fast"))
def test_precise_helpers_preserve_enclosing_metal_contraction(
    tmp_path, contraction, monkeypatch
):
    if os.environ.get(REQUIRE_ENV) != "1" or sys.platform != "darwin":
        pytest.skip(f"set {REQUIRE_ENV}=1 on macOS for contraction scope execution")
    from tests.test_translator import test_fused_math as native_math

    flags = ("-fno-fast-math", f"-ffp-contract={contraction}")
    monkeypatch.setattr(
        native_math, "_compile", partial(_compile, metal_compile_flags=flags)
    )
    pairs = [
        (0x3F800001, 0x3F800001, 0x3F800002, 0x3F800000),
        (0x3F800001, 0x3F7FFFFE, 0x3F800000, 0x3F800000),
        (0x3F800002, 0x3F800002, 0x3F800004, 0x3F800000),
    ]
    products = (
        [0x28800000, 0xA8800000, 0x29800000] if contraction == "fast" else [0] * 3
    )
    angles = [_oracle(a, b, "preserve-subnormals") for a, b, _, _ in pairs]
    expected = [GUARD] * 4
    for product, angle in zip(products, angles):
        expected.extend((product, angle))
    expected.extend([GUARD] * 4)
    generated = _translate(tmp_path, source=PRECISION_SCOPE_SOURCE, target="metal")
    for label, source in (
        ("original", PRECISION_SCOPE_SOURCE),
        ("translated", generated),
    ):
        work = tmp_path / label
        work.mkdir()
        actual, evidence = native_math._dispatch(
            work,
            "metal",
            source,
            pairs,
            len(expected),
            entry="scope",
            initial_output=[GUARD] * len(expected),
        )
        (work / "expected.json").write_text(json.dumps(expected))
        assert len(actual) == len(expected)
        assert actual[:4] == actual[-4:] == [GUARD] * 4
        assert actual[4:-4:2] == products
        maximum_angle_error = max(
            abs(got - want) for got, want in zip(actual[5:-4:2], angles)
        )
        (work / "evidence.json").write_text(
            json.dumps(
                {
                    **evidence,
                    "compilerFlags": flags,
                    "contraction": contraction,
                    "unchangedSourceControl": label == "original",
                    "guardCount": 8,
                    "maximumAngleUlpError": maximum_angle_error,
                    "angleUlpLimit": 6,
                },
                indent=2,
            )
        )
        assert maximum_angle_error <= 6
