"""Explicit source policy for numeric half-to-float NaN conversion."""

import json
import os
import sys
from functools import partial
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalHalfWideningProfileError,
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
from tests.test_translator.test_half_copy_identity import CASES, _source, _widen
from tests.test_translator.test_metal_builtin_ownership import _compile

PROFILE = "canonical-nan"
REQUIRE_ENV = "CROSTL_REQUIRE_HALF_WIDENING"
GUARD = 0x1937A5C3
SOURCE = """#include <metal_stdlib>
using namespace metal;
using Narrow = half;
using Wide = float;
struct Cell { float value; };
static half record(thread uint& count, half value) { count += 1; return value; }
static float widen_return(Narrow value) { return value; }
static uint float_argument(Wide value) { return as_type<uint>(value); }
kernel void computeMain(device const uint* values [[buffer(0)]],
                        device uint* results [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    Narrow value = as_type<half>(ushort(values[i]));
    uint count = 0;
    float initialized = value;
    float assigned = 0.0f;
    assigned = value;
    Cell cell = {value};
    float local[2] = {value, value};
    half2 pair = half2(value, value);
    half3 triple = half3(value, value, value);
    half4 quad = half4(value, value, value, value);
    results[4 + 16*i] = as_type<uint>(float(record(count, value)));
    results[4 + 16*i+1] = as_type<uint>((float)value);
    results[4 + 16*i+2] = as_type<uint>(initialized);
    results[4 + 16*i+3] = as_type<uint>(assigned);
    results[4 + 16*i+4] = as_type<uint>(widen_return(value));
    results[4 + 16*i+5] = float_argument(value);
    results[4 + 16*i+6] = as_type<uint>(float2(value).x);
    results[4 + 16*i+7] = as_type<uint>(float2(pair).y);
    results[4 + 16*i+8] = as_type<uint>(float3(triple).z);
    results[4 + 16*i+9] = as_type<uint>(float4(quad).w);
    results[4 + 16*i+10] = as_type<uint>(float4(float2(pair), value, 1.0f).z);
    results[4 + 16*i+11] = as_type<uint>(cell.value);
    results[4 + 16*i+12] = as_type<uint>(local[1]);
    results[4 + 16*i+13] = as_type<uint>(i % 2u != 0u ? record(count, value) : 1.0f);
    results[4 + 16*i+14] = uint(as_type<ushort>(half(value)));
    results[4 + 16*i+15] = count;
}
"""


def _translate(tmp_path, source=SOURCE, target="crossgl", profile=PROFILE):
    path = tmp_path / "widen.metal"
    path.write_text(source, encoding="utf-8")
    return translate(
        str(path),
        backend=target,
        format_output=False,
        source_options={"binary16_widening_profile": profile},
    )


@pytest.mark.parametrize("profile", [True, 1, "", "native", [], {}])
def test_half_widening_profile_rejects_invalid_configuration(profile):
    with pytest.raises(ValueError, match="binary16_widening_profile"):
        MetalToCrossGLConverter(binary16_widening_profile=profile)


def test_half_widening_profile_is_explicit_and_resets(tmp_path):
    assert "__crossgl_metal_widen_half" not in _translate(tmp_path, profile=None)
    converter = MetalToCrossGLConverter(binary16_widening_profile=PROFILE)
    for source, expected in (
        (SOURCE, True),
        ("half identity(half value) { return value; }", False),
    ):
        generated = converter.generate(
            MetalParser(MetalLexer(source).tokenize()).parse()
        )
        assert ("__crossgl_metal_widen_half" in generated) == expected


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_half_widening_profile_compiles(tmp_path, target):
    generated = _translate(tmp_path, target=target)
    for width in ("", "2", "3", "4"):
        assert f"metal_widen_half{width}(" in generated
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


@pytest.mark.parametrize("case", CASES)
def test_half_widening_profile_preserves_same_type_copies(tmp_path, case):
    generated = _translate(tmp_path, _source(case), target="opengl")
    assert "__crossgl_metal_widen_half" not in generated
    assert "crossgl_round_half" not in generated


@pytest.mark.parametrize(
    "source",
    [
        "float apply(half value) { return value; }",
        "float apply(half value) { float output = value; return output; }",
        "float apply(half value) { float output; output = value; return output; }",
        "void apply(half value, device float* output) { output[0] = value; }",
        "struct Cell { float value; }; void apply(half value, thread Cell& output) { output.value = value; }",
        "float identity(float value) { return value; } float apply(half value) { return identity(value); }",
        "float apply(half value, bool test) { return test ? value : 1.0f; }",
        "float apply(half value) { return value + 1.0f; }",
        "float2 apply(half value, float2 other) { return other + value; }",
    ],
)
def test_half_widening_profile_retains_implicit_conversions(tmp_path, source):
    generated = _translate(tmp_path, source)
    assert "__crossgl_metal_widen_half(value)" in generated


def test_half_widening_helpers_do_not_capture_source_names(tmp_path):
    source = """
    float __crossgl_metal_widen_half(half value) { return 42.0f; }
    float asfloat(uint value) { return 42.0f; }
    float apply(half value) { return float(value); }
    """
    generated = _translate(tmp_path, source)
    assert "float __crossgl_metal_widen_half_(float16 value)" in generated
    assert "asfloat__metal_overload_1" in generated


def test_half_widening_profile_rejects_global_initializer(tmp_path):
    with pytest.raises(MetalHalfWideningProfileError):
        _translate(
            tmp_path, "constant half value = half(1.0f); float result = float(value);"
        )


@pytest.mark.parametrize(
    "options",
    [
        {"binary32_additive_profile": "rne-gradual"},
        {"binary32_multiplication_profile": "rne-gradual"},
        {"binary32_division_profile": "rne-gradual"},
    ],
)
def test_half_widening_precedes_profiled_arithmetic(tmp_path, options):
    path = tmp_path / "arithmetic.metal"
    path.write_text(
        "float apply(half value, float other) { return value + other * value / other; }",
        encoding="utf-8",
    )
    generated = translate(
        str(path),
        backend="crossgl",
        format_output=False,
        source_options={"binary16_widening_profile": PROFILE, **options},
    )
    assert "__crossgl_metal_widen_half(value)" in generated


def test_ci_requires_half_widening_in_existing_math_job():
    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Validate binary32 arithmetic"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert (
        "tests/test_translator/test_metal_half_widening.py::test_half_widening_profile_executes"
        in step
    )
    assert "pytest -q -n auto" in step
    assert workflow.count("tests/test_translator/test_metal_half_widening.py") == 1


def test_project_retains_half_widening_profile(tmp_path):
    (tmp_path / "widen.metal").write_text(SOURCE, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=("opengl",),
            source_options={"metal": {"binary16_widening_profile": PROFILE}},
        ),
        format_output=False,
    )
    data = report.to_json()
    assert data["summary"]["translatedCount"] == 1, data["diagnostics"]
    provenance = data["artifacts"][0]["provenance"]
    assert provenance["binary16WideningProfile"] == PROFILE
    path = tmp_path / "report.json"
    report.write_json(path)
    assert validate_project_report(path)["success"]
    manifest = build_runtime_artifact_manifest(path)
    assert manifest["success"]
    assert PROFILE in json.dumps(manifest)
    for invalid in (None, "native", 1, []):
        if invalid is None:
            provenance.pop("binary16WideningProfile", None)
        else:
            provenance["binary16WideningProfile"] = invalid
        path.write_text(json.dumps(data), encoding="utf-8")
        result = validate_project_report(path)
        assert not result["success"]
        assert "binary16WideningProfile" in json.dumps(result["diagnostics"])


@pytest.mark.parametrize("original", [False, True], ids=["generated", "original"])
def test_half_widening_profile_executes(tmp_path, monkeypatch, original):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native half widening")
    if original and sys.platform != "darwin":
        pytest.skip("The unchanged Metal control requires macOS")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    generated = SOURCE if original else _translate(tmp_path, target=target)
    monkeypatch.setattr(
        "tests.test_translator.test_fused_math._compile",
        partial(
            _compile,
            directx_compile_flags=("-enable-16bit-types",),
            metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
        ),
    )
    inputs = [(word,) for word in range(65536)]
    expected = []
    for word in range(65536):
        widened = 0x7FC00000 if word & 0x7FFF > 0x7C00 else _widen(word)
        expected.extend([widened] * 13)
        expected.extend([widened if word % 2 else 0x3F800000, word, 1 + word % 2])
    trials = (
        (fill, start) for fill in (0x3EAAAAAB, 0xBE4CCCCD) for start in (0, 32768)
    )
    for fill, start in trials:
        work = tmp_path / f"fill-{fill:08x}-{start}"
        work.mkdir()
        selected = inputs[start : start + 32768]
        selected_expected = expected[16 * start : 16 * (start + 32768)]
        initial = [GUARD] * 4 + [fill] * len(selected_expected) + [GUARD] * 4
        actual, evidence = _dispatch(
            work, target, generated, selected, len(initial), initial_output=initial
        )
        want = [GUARD] * 4 + selected_expected + [GUARD] * 4
        mismatches = [
            {"index": index, "expected": left, "actual": right}
            for index, (left, right) in enumerate(zip(want, actual))
            if left != right
        ]
        evidence.update(
            profile=PROFILE,
            originalSource=original,
            inputWordCount=len(selected),
            firstInputWord=start,
            outputWordCount=len(want),
            guardWordCount=8,
            mismatchCount=len(mismatches),
            mismatches=mismatches[:32],
        )
        (work / "evidence.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )
        assert len(actual) == len(want)
        assert not mismatches, mismatches[:16]
