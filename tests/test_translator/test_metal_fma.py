"""Explicit binary32 FMA source profiles across native targets."""

import json
import os
import sys

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from tests.test_translator.test_fused_math import (
    REQUIRE_ENV,
    _dispatch,
    _oracle,
    _triples,
)
from tests.test_translator.test_metal_builtin_ownership import _compile, _run

SOURCE = """#include <metal_stdlib>
using namespace metal;
static float record(thread uint& count, float value) { count += 1; return value; }
kernel void fused(device const uint* values [[buffer(0)]],
                  device uint* results [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
    float a = as_type<float>(values[3 * i]);
    float b = as_type<float>(values[3 * i + 1]);
    float c = as_type<float>(values[3 * i + 2]);
    uint count = 0;
    float2 pair = metal::fma(float2(a, b), float2(b, a), float2(c));
    float3 triple = metal::precise::fma(float3(a, b, a), float3(b, a, b), float3(c));
    float4 quad = precise::fma(float4(record(count, a), b, a, b), float4(b, a, b, a), float4(c));
    results[11 * i] = as_type<uint>(fma(a, b, c));
    results[11 * i + 1] = as_type<uint>(pair.x);
    results[11 * i + 2] = as_type<uint>(pair.y);
    results[11 * i + 3] = as_type<uint>(triple.x);
    results[11 * i + 4] = as_type<uint>(triple.y);
    results[11 * i + 5] = as_type<uint>(triple.z);
    results[11 * i + 6] = as_type<uint>(quad.x);
    results[11 * i + 7] = as_type<uint>(quad.y);
    results[11 * i + 8] = as_type<uint>(quad.z);
    results[11 * i + 9] = as_type<uint>(quad.w);
    results[11 * i + 10] = count;
}
"""


def _translate(tmp_path, source, target, profile="rne-flush"):
    path = tmp_path / "fused.metal"
    path.write_text(source, encoding="utf-8")
    return translate(
        str(path),
        backend=target,
        format_output=False,
        source_options={"binary32_fma_profile": profile},
    )


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
@pytest.mark.parametrize("profile", ["rne-flush", "rne-gradual"])
def test_metal_fma_helpers_compile(tmp_path, target, profile):
    generated = _translate(tmp_path, SOURCE, target, profile)
    for suffix in ("", "2", "3", "4"):
        assert f"metal_fma_float{suffix}(" in generated
    assert "Berkeley SoftFloat" in generated
    assert "uint64" not in generated and "double" not in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("profile", ["", "toward-zero", True, 1, [], {}])
def test_metal_fma_rejects_unknown_profile(profile):
    with pytest.raises(ValueError, match="binary32_fma_profile"):
        MetalToCrossGLConverter(binary32_fma_profile=profile)


def test_metal_fma_requires_explicit_profile(tmp_path):
    generated = _translate(tmp_path, SOURCE, "crossgl", None)
    assert "__crossgl_fma_" not in generated
    assert "__crossgl_metal_fma_" not in generated


def test_metal_fma_keeps_user_overloads_fast_mode_and_half(tmp_path):
    source = """
    float fma(float a, float b, float c) { return a + b + c; }
    float builtin(float a) { return metal::fma(a, a, a); }
    float user(float a) { return ::fma(a, a, a); }
    float fast_mode(float a) { return metal::fast::fma(a, a, a); }
    half narrow(half a) { return metal::fma(a, a, a); }
    """
    generated = _translate(tmp_path, source, "crossgl")
    assert (
        "return __crossgl_metal_fma_float(float(a), float(a), float(a));" in generated
    )
    assert "return fma__metal_overload_1(a, a, a);" in generated
    assert generated.count("return fma(a, a, a);") == 2


def test_metal_fma_helper_names_do_not_collide(tmp_path):
    source = """
    float __crossgl_metal_fma_float(float a) { return a; }
    uint __crossgl_fma_bits(uint a) { return a; }
    float builtin(float a) { return metal::fma(a, a, a); }
    """
    generated = _translate(tmp_path, source, "crossgl")
    assert "float __crossgl_metal_fma_float_(float a, float b, float c)" in generated
    assert "uint __crossgl_fma_bits_(uint a, uint b, uint c," in generated


def test_metal_fma_helpers_have_private_linkage(tmp_path):
    intermediate = _translate(tmp_path, SOURCE, "crossgl")
    ast = Parser(Lexer(intermediate).get_tokens()).parse()
    helpers = [f for f in ast.functions if f.name.startswith("__crossgl_")]
    assert len(helpers) == 11
    assert all(f.linkage == "internal" for f in helpers)


def test_metal_fma_modules_link_together(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native fused rounding")
    if sys.platform != "darwin":
        pytest.skip("Metal library linking requires macOS")
    objects = []
    for index, profile in enumerate(("rne-gradual", "rne-flush")):
        directory = tmp_path / str(index)
        directory.mkdir()
        source = SOURCE.replace("kernel void fused(", f"kernel void fused{index}(")
        _compile(_translate(directory, source, "metal", profile), "metal", directory)
        objects.append(str(directory / "translated.air"))
    library = tmp_path / "combined.metallib"
    _run(["xcrun", "--sdk", "macosx", "metallib", *objects, "-o", str(library)])
    assert library.stat().st_size > 0


@pytest.mark.parametrize("profile", ["rne-gradual", "rne-flush", "source"])
def test_metal_fma_executes(tmp_path, profile):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native fused rounding")
    original = profile == "source"
    if original and sys.platform != "darwin":
        pytest.skip("The original Metal kernel requires macOS")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    generated = SOURCE if original else _translate(tmp_path, SOURCE, target, profile)
    triples = _triples()
    expected = [
        want
        for triple in triples
        for want in [_oracle(*triple, flush=profile != "rne-gradual")] * 10 + [1]
    ]
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    actual, evidence = _dispatch(
        tmp_path,
        target,
        generated,
        triples,
        len(expected),
        entry="fused" if target == "metal" else None,
    )
    mismatches = []
    for index, (got, want) in enumerate(zip(actual, expected)):
        both_nan = (got & 0x7FFFFFFF) > 0x7F800000 and (want & 0x7FFFFFFF) > 0x7F800000
        if got != want and not (original and index % 11 != 10 and both_nan):
            mismatches.append(
                {
                    "index": index,
                    "inputBits": triples[index // 11],
                    "expected": want,
                    "actual": got,
                }
            )
    evidence.update(
        profile=profile,
        original=original,
        mismatchCount=len(mismatches),
        mismatches=mismatches,
    )
    (tmp_path / "evidence.json").write_text(json.dumps(evidence, indent=2))
    assert len(actual) == len(expected)
    assert not mismatches, mismatches[:10]
