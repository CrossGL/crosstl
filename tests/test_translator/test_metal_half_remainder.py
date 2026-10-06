"""Explicit binary32-quotient profile for source half floating remainder."""

import json
import math
import os
import random
import struct
import sys
from functools import partial
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalHalfRemainderProfileError,
    MetalToCrossGLConverter,
)
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.project import (
    ProjectConfig,
    build_runtime_artifact_manifest,
    translate_project,
    validate_project_report,
)
from tests.test_translator.test_fused_math import _dispatch
from tests.test_translator.test_metal_builtin_ownership import _compile

PROFILE = "binary32-quotient"
REQUIRE_ENV = "CROSTL_REQUIRE_HALF_REMAINDER"
GUARD = 0x1937A5C3
SOURCE = """#include <metal_stdlib>
using namespace metal;
using Real = half;
static half record(thread uint& count, half value) { count += 1; return value; }
kernel void computeMain(device const uint* values [[buffer(0)]],
                        device uint* results [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    Real a = as_type<half>(ushort(values[2*i]));
    half b = as_type<half>(ushort(values[2*i+1]));
    uint count = 0;
    half scalar = fmod(record(count, a), b);
    half2 pair = metal::fmod(half2(a, -a), half2(b, -b));
    half3 triple = fmod(half3(a, -a, a), b);
    half4 quad = metal::fmod(a, half4(b, -b, b, -b));
    results[4 + 11*i] = uint(as_type<ushort>(scalar));
    results[4 + 11*i+1] = uint(as_type<ushort>(pair.x));
    results[4 + 11*i+2] = uint(as_type<ushort>(pair.y));
    results[4 + 11*i+3] = uint(as_type<ushort>(triple.x));
    results[4 + 11*i+4] = uint(as_type<ushort>(triple.y));
    results[4 + 11*i+5] = uint(as_type<ushort>(triple.z));
    results[4 + 11*i+6] = uint(as_type<ushort>(quad.x));
    results[4 + 11*i+7] = uint(as_type<ushort>(quad.y));
    results[4 + 11*i+8] = uint(as_type<ushort>(quad.z));
    results[4 + 11*i+9] = uint(as_type<ushort>(quad.w));
    results[4 + 11*i+10] = count;
}
"""


def _translate(tmp_path, source=SOURCE, target="crossgl", profile=PROFILE):
    path = tmp_path / "remainder.metal"
    path.write_text(source, encoding="utf-8")
    return translate(
        str(path),
        backend=target,
        format_output=False,
        source_options={"binary16_remainder_profile": profile},
    )


def _oracle(a, b):
    x, y = (struct.unpack("<e", struct.pack("<H", word))[0] for word in (a, b))
    if not math.isfinite(x) or not math.isfinite(y) or y == 0:
        return 0x7E00

    def binary32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]

    # Half operands have at most eleven significant bits. Binary64 suffices
    # for these binary32 rounding steps, including the unrounded product.
    quotient = binary32(x / y)
    integral = math.copysign(float(math.trunc(quotient)), quotient)
    product = binary32(integral * y)
    difference = binary32(x - product)
    return struct.unpack("<H", struct.pack("<e", difference))[0]


def _pairs():
    edges = [
        0,
        1,
        2,
        3,
        0x3FF,
        0x400,
        0x401,
        0x3BFF,
        0x3C00,
        0x3C01,
        0x4000,
        0x7BFF,
        0x7C00,
        0x7C01,
        0x7E00,
    ]
    signed = [value | sign for value in edges for sign in (0, 0x8000)]
    pairs = [(a, b) for a in signed for b in signed]
    rng = random.Random(2113)
    pairs.extend((rng.getrandbits(16), rng.getrandbits(16)) for _ in range(4096))
    pairs.extend(
        ((exponent << 10) | fraction, divisor)
        for exponent in range(1, 31)
        for fraction in (0, 1, 1023)
        for divisor in (0x3BFF, 0x3C01)
    )
    return pairs


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (0xC580, 0x4000, 0xBE00),
        (0x4580, 0xC000, 0x3E00),
        (0x8000, 0x4000, 0),
        (0x3C00, 0x7C00, 0x7E00),
        (0x7C00, 0x4000, 0x7E00),
        (0x3C00, 0, 0x7E00),
        (3, 2, 1),
        (0x7BFF, 0x0401, 0),
        (0x0400, 0x03FF, 1),
        (0x8400, 0x03FF, 0x8001),
    ],
)
def test_half_remainder_profile_oracle(a, b, expected):
    assert _oracle(a, b) == expected


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_half_remainder_profile_compiles(tmp_path, target):
    generated = _translate(tmp_path, target=target)
    for width in ("", "2", "3", "4"):
        assert f"metal_remainder_half{width}(" in generated
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


@pytest.mark.parametrize("profile", [True, 1, "", "exact", [], {}])
def test_half_remainder_profile_rejects_invalid_configuration(profile):
    with pytest.raises(ValueError, match="binary16_remainder_profile"):
        MetalToCrossGLConverter(binary16_remainder_profile=profile)


def test_half_remainder_profile_is_explicit_and_resets(tmp_path):
    assert "metal_remainder_half" not in _translate(tmp_path, profile=None)
    converter = MetalToCrossGLConverter(binary16_remainder_profile=PROFILE)
    for source, expected in (
        (SOURCE, True),
        ("float apply(float a, float b) { return fmod(a, b); }", False),
    ):
        generated = converter.generate(
            MetalParser(MetalLexer(source).tokenize()).parse()
        )
        assert ("metal_remainder_half" in generated) == expected
        assert ("__crossgl_fma_bits" in generated) == expected
        assert ("__crossgl_divide_bits" in generated) == expected


def test_half_remainder_profile_preserves_overloads(tmp_path):
    source = """
    half fmod(half a, half b) { return a + b; }
    half custom(half a, half b) { return ::fmod(a, b); }
    half builtin(half a, half b) { return metal::fmod(a, b); }
    float wide(float a, float b) { return metal::fmod(a, b); }
    """
    generated = _translate(tmp_path, source)
    assert "return fmod__metal_overload_1(a, b);" in generated
    assert "return __crossgl_metal_remainder_half(float16(a), float16(b));" in generated
    assert "return fmod(a, b);" in generated


def test_half_remainder_helpers_do_not_capture_source_names(tmp_path):
    source = """
    uint __crossgl_fma_bits(uint a) { return a; }
    uint __crossgl_divide_bits(uint a) { return a; }
    half __crossgl_metal_remainder_half(half a) { return a; }
    half apply(half a, half b) { return fmod(a, b); }
    """
    generated = _translate(tmp_path, source)
    assert "uint __crossgl_fma_bits_(uint a, uint b" in generated
    assert "uint __crossgl_divide_bits_(uint a, uint b" in generated
    assert "float16 __crossgl_metal_remainder_half_(float16 a, float16 b)" in generated


def test_half_remainder_profile_shares_helpers_without_changing_other_profiles(
    tmp_path,
):
    path = tmp_path / "profiles.metal"
    path.write_text(
        "half narrow(half a, half b) { return fmod(a, b); }\n"
        "float wide(float a, float b) { return fma(a, b, a) / b; }\n",
        encoding="utf-8",
    )
    generated = translate(
        str(path),
        backend="crossgl",
        format_output=False,
        source_options={
            "binary16_remainder_profile": PROFILE,
            "binary32_division_profile": "rne-flush",
            "binary32_fma_profile": "rne-flush",
        },
    )
    assert generated.count("uint __crossgl_divide_bits(uint a, uint b") == 1
    assert generated.count("uint __crossgl_fma_bits(uint a, uint b") == 1
    assert "__crossgl_divide_bits(x, y, false)" in generated
    assert "__crossgl_divide_bits(asuint(a), asuint(b), true)" in generated
    assert "__crossgl_fma_bits(asuint(a), asuint(b), asuint(c), true)" in generated
    assert (
        "__crossgl_fma_bits(product ^ 0x80000000u, 0x3f800000u, x, false)" in generated
    )


def test_half_remainder_profile_rejects_global_initializer(tmp_path):
    with pytest.raises(MetalHalfRemainderProfileError) as error:
        _translate(tmp_path, "half value = fmod(half(1.5), half(0.7));")
    assert error.value.profile == PROFILE
    assert error.value.project_diagnostic_code.endswith(
        "half-remainder-profile-unsupported"
    )


@pytest.mark.parametrize("builtin", ["asfloat", "asuint"])
def test_half_remainder_profile_preserves_source_bitcast_names(tmp_path, builtin):
    declaration = (
        "float asfloat(uint x) { return float(x) + 100.0f; }"
        if builtin == "asfloat"
        else "uint asuint(float x) { return uint(x) + 200u; }"
    )
    source = declaration + "\nhalf apply(half a, half b) { return fmod(a, b); }"
    generated = _translate(tmp_path, source)
    assert f"{builtin}__metal_overload_1(" in generated
    assert "uint x = asuint(float(a));" in generated
    assert "return float16(asfloat(difference));" in generated


def test_project_retains_half_remainder_profile(tmp_path):
    (tmp_path / "remainder.metal").write_text(SOURCE, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=("opengl",),
            source_options={"metal": {"binary16_remainder_profile": PROFILE}},
        ),
        format_output=False,
    )
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 1, data["diagnostics"]
    provenance = data["artifacts"][0]["provenance"]
    assert provenance["binary16RemainderProfile"] == PROFILE
    path = tmp_path / "report.json"
    report.write_json(path)
    assert validate_project_report(path)["success"]
    assert build_runtime_artifact_manifest(path)["success"]
    for invalid in (None, "exact", 1, []):
        if invalid is None:
            provenance.pop("binary16RemainderProfile", None)
        else:
            provenance["binary16RemainderProfile"] = invalid
        path.write_text(json.dumps(data), encoding="utf-8")
        result = validate_project_report(path)
        assert not result["success"]
        assert "binary16RemainderProfile" in json.dumps(result["diagnostics"])


@pytest.mark.parametrize("original", [False, True], ids=["generated", "original"])
def test_half_remainder_profile_executes(tmp_path, monkeypatch, original):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native half remainder")
    if original and sys.platform != "darwin":
        pytest.skip("The unchanged Metal control requires macOS")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    source = SOURCE if original else _translate(tmp_path, target=target)
    monkeypatch.setattr(
        "tests.test_translator.test_fused_math._compile",
        partial(
            _compile,
            directx_compile_flags=("-enable-16bit-types",),
            metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
        ),
    )
    pairs = _pairs()
    expected = [GUARD] * 4
    for a, b in pairs:
        normal, negative = _oracle(a, b), _oracle(a ^ 0x8000, b)
        expected.extend(
            [
                normal,
                normal,
                negative,
                normal,
                negative,
                normal,
                normal,
                normal,
                normal,
                normal,
                1,
            ]
        )
    expected.extend([GUARD] * 4)
    (tmp_path / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    actual, evidence = _dispatch(
        tmp_path,
        target,
        source,
        pairs,
        len(expected),
        initial_output=[GUARD] * len(expected),
    )
    assert len(actual) == len(expected)
    assert actual[:4] == actual[-4:] == [GUARD] * 4
    mismatches = []
    for index, (want, got) in enumerate(zip(expected[4:-4], actual[4:-4])):
        both_nan = (
            index % 11 != 10
            and 0 <= got <= 0xFFFF
            and want & 0x7FFF > 0x7C00
            and got & 0x7FFF > 0x7C00
        )
        if want != got and not both_nan:
            mismatches.append({"index": index, "expected": want, "actual": got})
    evidence.update(
        profile=PROFILE,
        originalSource=original,
        pairCount=len(pairs),
        numericalResultCount=10 * len(pairs),
        sideEffectCount=len(pairs),
        guardCount=8,
        nanComparison="classification-only",
        mismatchCount=len(mismatches),
        mismatches=mismatches,
    )
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    assert not mismatches, mismatches[:10]


def test_ci_requires_half_remainder_in_existing_math_job():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Validate binary32 arithmetic"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert (
        "tests/test_translator/test_metal_half_remainder.py::test_half_remainder_profile_executes"
        in step
    )
    assert "pytest -q -n auto" in step
    assert workflow.count("tests/test_translator/test_metal_half_remainder.py") == 1
