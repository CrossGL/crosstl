"""Lossless binary16 bitcasts through widened OpenGL storage."""

import json
import os
import sys
from functools import partial

import pytest

from crosstl import translate
from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from tests.test_translator.test_fused_math import _dispatch
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_HALF_WIDENING"
SOURCE = """#include <metal_stdlib>
using namespace metal;
half2 record(thread uint& count, half2 value) { count += 1; return value; }
kernel void computeMain(device const uint* values [[buffer(0)]],
                        device uint* results [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    uint word = values[i];
    uint count = 0;
    half2 pair = as_type<half2>(word);
    half scalar = as_type<half>(ushort(word));
    half signed_value = as_type<half>(short(word >> 16u));
    results[4 + 5*i] = as_type<uint>(record(count, pair));
    results[4 + 5*i+1] = uint(as_type<ushort>(scalar));
    results[4 + 5*i+2] = uint(as_type<ushort>(pair.x));
    results[4 + 5*i+3] = uint(as_type<short>(signed_value));
    results[4 + 5*i+4] = count;
}
"""


@pytest.mark.parametrize("builtin", ["uintBitsToFloat", "floatBitsToUint", "findMSB"])
def test_half_bitcasts_reject_shadowed_target_builtins(builtin):
    source = f"""shader Bits {{
        uint {builtin}(uint value) {{ return value; }}
        half decode(uint16 value) {{ return as_type<half>(value); }}
    }}"""
    with pytest.raises(ValueError, match=f"unshadowed {builtin}"):
        GLSLCodeGen().generate(parse(source))


def test_half_bitcast_helpers_reset_and_avoid_source_names():
    generator = GLSLCodeGen()
    source = """shader Bits {
        float crossgl_half_bits_decode(uint value) { return float(value); }
        half decode(uint16 value) { return as_type<half>(value); }
    }"""
    generated = generator.generate(parse(source))
    helper = generator.glsl_half_bitcast_helper_names["decode"]
    assert helper != "crossgl_half_bits_decode"
    assert f"return {helper}(value);" in generated
    assert "crossgl_half_bits_" not in generator.generate(parse("shader Empty {}"))


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_half_bitcasts_compile(tmp_path, target):
    path = tmp_path / "bits.metal"
    path.write_text(SOURCE, encoding="utf-8")
    generated = translate(str(path), backend=target, format_output=False)
    if target == "opengl":
        assert "packHalf2x16(" not in generated
        assert "unpackHalf2x16(" not in generated
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


@pytest.mark.parametrize("original", [False, True], ids=["generated", "original"])
def test_half_bitcasts_execute(tmp_path, monkeypatch, original):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native bitcast controls")
    if original and sys.platform != "darwin":
        pytest.skip("The unchanged Metal control requires macOS")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    path = tmp_path / "bits.metal"
    path.write_text(SOURCE, encoding="utf-8")
    generated = (
        SOURCE
        if original
        else translate(str(path), backend=target, format_output=False)
    )
    monkeypatch.setattr(
        "tests.test_translator.test_fused_math._compile",
        partial(
            _compile,
            directx_compile_flags=("-enable-16bit-types",),
            metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
        ),
    )
    for fill in (0x1937A5C3, 0xC5937A31):
        for start in (0, 32768):
            work = tmp_path / f"fill-{fill:08x}-{start}"
            work.mkdir()
            words = [
                word | ((65535 - word) << 16) for word in range(start, start + 32768)
            ]
            expected = []
            for word in words:
                low, high = word & 65535, word >> 16
                signed_high = high if high < 32768 else high | 0xFFFF0000
                expected.extend([word, low, low, signed_high, 1])
            guards = [0xC94A1375] * 4
            initial = guards + [fill] * len(expected) + guards
            actual, evidence = _dispatch(
                work,
                target,
                generated,
                [(word,) for word in words],
                len(initial),
                initial_output=initial,
            )
            want = guards + expected + guards
            mismatches = [
                {"index": index, "expected": left, "actual": right}
                for index, (left, right) in enumerate(zip(want, actual))
                if left != right
            ]
            evidence.update(
                originalSource=original,
                firstLowWord=start,
                inputWordCount=len(words),
                guardWordCount=8,
                mismatchCount=len(mismatches),
                mismatches=mismatches[:32],
            )
            (work / "evidence.json").write_text(
                json.dumps(evidence, indent=2), encoding="utf-8"
            )
            assert len(actual) == len(want)
            assert not mismatches, mismatches[:16]


def test_half_buffer_bitcasts_execute(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native half buffer bitcasts")
    from tests.test_translator.test_boolean_buffer_runtime import (
        _bound_values,
        _request,
    )
    from tests.test_translator.test_half_buffer_runtime import _validate_half
    from tests.test_translator.test_half_copy_identity import _widen
    from tests.test_translator.test_loop_updates import _execute
    from tests.test_translator.test_software_subgroup_product import _package

    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    source = """#include <metal_stdlib>
using namespace metal;
kernel void half_bits(const device half* values [[buffer(0)]],
                      device uint* results [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    results[i + 4u] = uint(as_type<ushort>(values[i]));
}
"""
    _, descriptor, package = _package(
        tmp_path, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    binding = next(
        row
        for row in descriptor["bindings"]
        if row["scalarLayout"].get("memberName", row["name"]) == "values"
    )
    dtype = binding["scalarLayout"]["elementType"]
    assert dtype in {"float32", "float16", "uint16"}
    for fill in (0x1937A5C3, 0xC5937A31):
        for start in (0, 32768):
            work = tmp_path / f"buffer-{fill:08x}-{start}"
            work.mkdir()
            words = list(range(start, start + 32768))
            data = [_widen(word) for word in words] if dtype == "float32" else words
            encoding = {
                "float32": "ieee754-binary32",
                "float16": "ieee754-binary16",
            }.get(dtype)
            guard = [0xC94A1375] * 4
            inputs = {
                "values": {
                    "dtype": dtype,
                    "shape": [len(words)],
                    "values": data,
                    **({"encoding": encoding} if encoding else {}),
                },
                "results": {
                    "dtype": "uint32",
                    "shape": [len(words) + 8],
                    "values": guard + [fill] * len(words) + guard,
                },
            }
            outputs = {
                "results": {**inputs["results"], "values": guard + words + guard}
            }
            request = _request(descriptor, package, inputs, outputs, len(words))
            _execute(
                request,
                _bound_values(descriptor, outputs),
                work,
                validate=_validate_half,
            )
