"""Source binary16 nearest-even conversions with native DirectX storage."""

import json
import math
import os
import shutil
import struct
import sys
from pathlib import Path

import pytest
import yaml

from crosstl import translate
from crosstl.translator.codegen.directx_codegen import (
    DirectXContextualConversionError,
    HLSLCodeGen,
)
from tests.test_translator.test_directx_float_atomics import _compile
from tests.test_translator.test_opengl_half_conversion import (
    SOURCE,
    TARGET_ENV,
    _shader,
)
from tests.test_translator.test_opengl_half_conversion import (
    test_half_rounding_native as _native_rounding,
)


def _compile_half(generated, tmp_path):
    return _compile(
        generated, tmp_path, profile="cs_6_2", flags=("-enable-16bit-types",)
    )


@pytest.mark.parametrize(
    "expression", ["half(input[i])", "static_cast<half>(input[i])", "(half)input[i]"]
)
def test_public_metal_half_conversion(tmp_path, expression):
    source = tmp_path / "kernel.metal"
    source.write_text(
        SOURCE.replace("BODY", f"output[i] = float({expression});"), encoding="utf-8"
    )
    generated = translate(str(source), backend="directx", format_output=False)
    assert "__crossgl_round_half1(float(input.Load(i)))" in generated
    assert "return asfloat16(uint16_t(sign | result));" in generated
    _compile_half(generated, tmp_path)


@pytest.mark.parametrize("dtype", ["half", "float16_t", "half2", "half3", "half4"])
def test_half_constructors_and_boundaries(tmp_path, dtype):
    width = int(dtype[-1]) if dtype[-1] in "234" else 1
    floating = "float" if width == 1 else f"float{width}"
    generated = HLSLCodeGen().generate(
        _shader(
            f"{dtype} x = {floating}(output[0]); {dtype} y = {dtype}(output[1]); "
            "output[0] = float(narrow(output[2])); output[1] = widen(output[3]);",
            "half narrow(float x) { return x; } float widen(half x) { return float(x); }",
        )
    )
    assert f"__crossgl_round_half{width}(" in generated
    assert "return __crossgl_round_half1(x);" in generated
    assert "widen(__crossgl_round_half1(output[3]))" in generated
    _compile_half(generated, tmp_path)


def test_half_vector_side_effects_are_evaluated_once(tmp_path):
    generated = HLSLCodeGen().generate(
        _shader(
            "float calls = 0.0; half4 value = half4(record(calls)); output[0] = calls;",
            "float record(inout float calls) { calls += 1.0; return 1.00146484375; }",
        )
    )
    assert generated.count("record(calls)") == 1
    assert "__crossgl_round_half4(((float4)(record(calls))))" in generated
    _compile_half(generated, tmp_path)


def test_half_helpers_are_collision_safe_and_reset(tmp_path):
    generator = HLSLCodeGen()
    generated = generator.generate(
        _shader(
            "float __crossgl_round_half1 = output[0]; half x = __crossgl_round_half1;"
        )
    )
    assert "float16_t __crossgl_round_half1_(float value)" in generated
    _compile_half(generated, tmp_path)
    assert "__crossgl_round_half" not in generator.generate(_shader("output[0] = 1.0;"))


@pytest.mark.parametrize(
    "body",
    [
        "half value = double(1.0);",
        "output[0] = float(half(double(1.0)));",
        "half3 value = half3(double2(1.0));",
        "half value = half(1.0); value += double(0.1);",
    ],
)
def test_half_double_rounding_is_rejected(body):
    with pytest.raises(DirectXContextualConversionError) as error:
        HLSLCodeGen().generate(_shader(body))
    assert error.value.reason == "half-double-rounding"


@pytest.mark.parametrize("name", ["asuint", "asfloat16"])
def test_half_shadowed_intrinsics_are_rejected(name):
    with pytest.raises(DirectXContextualConversionError) as error:
        HLSLCodeGen().generate(
            _shader("half value = output[0];", f"float {name}(float x) {{ return x; }}")
        )
    assert error.value.reason == "half-target-intrinsic-shadowed"


def test_half_bitcasts_are_not_numeric_conversions(tmp_path):
    generated = HLSLCodeGen().generate(
        _shader(
            "half value = as_type<half>(uint16_t(0x7e01)); "
            "output[0] = float(as_type<ushort>(value));"
        )
    )
    assert "__crossgl_round_half" not in generated
    assert "asfloat16(" in generated
    _compile_half(generated, tmp_path)


def test_half_global_constants_compile(tmp_path):
    generated = HLSLCodeGen().generate(
        _shader(
            "output[0] = float(narrowed);",
            "const half narrowed = half(0.1);",
        )
    )
    _compile_half(generated, tmp_path)


def test_half_mixed_vector_widening_decodes_each_payload(tmp_path):
    generated = HLSLCodeGen().generate(
        _shader(
            "half4 value = half4(output[0]); float4 wide = float4(value.xy, value.z, value.w); "
            "output[0] = wide.x;"
        )
    )
    assert "__crossgl_binary16_to_float(uint2(asuint16(value.xy)))" in generated
    assert "__crossgl_binary16_to_float(uint(asuint16(value.z)))" in generated
    assert "__crossgl_binary16_to_float(uint(asuint16(value.w)))" in generated
    _compile_half(generated, tmp_path)


@pytest.mark.parametrize(
    "name", ["asuint16", "asuint", "asfloat", "__crossgl_binary16_to_float"]
)
def test_half_widening_rejects_shadowed_helpers(name):
    with pytest.raises(DirectXContextualConversionError) as error:
        HLSLCodeGen().generate(
            _shader(
                "half value = half(1.0); output[0] = float(value);",
                f"float {name}(float x) {{ return x; }}",
            )
        )
    assert error.value.reason == "half-widening-intrinsic-shadowed"


@pytest.mark.parametrize("mode", ["rounding", "boundaries", "vectors"])
def test_directx_half_cases_compile(tmp_path, monkeypatch, mode):
    if shutil.which("dxc") is None:
        pytest.skip("DXC is not installed")
    monkeypatch.setenv(TARGET_ENV, "directx")
    _native_rounding(tmp_path, mode, compile_only=True)


@pytest.mark.parametrize("mode", ["rounding", "boundaries", "vectors"])
def test_directx_half_rounding_native(tmp_path, monkeypatch, mode):
    monkeypatch.setenv(TARGET_ENV, "directx")
    _native_rounding(tmp_path, mode)


def _half_product_cases():
    pairs = {(value, 0x3C01) for value in range(65536) if value & 0x7C00 != 0x7C00}
    boundaries = (
        0,
        1,
        3,
        0x03FF,
        0x0400,
        0x3800,
        0x3801,
        0x3BFF,
        0x3C00,
        0x3C01,
        0x7BFE,
        0x7BFF,
    )
    for left in boundaries:
        for right in boundaries:
            for sign in (0, 0x8000):
                pairs.update(((left | sign, right), (right, left | sign)))
    return sorted(pairs)


def _half_product_bits(left, right):
    x, y = (struct.unpack("<e", struct.pack("<H", word))[0] for word in (left, right))
    # Products of finite binary16 operands are exact in Python's binary64 value.
    product = x * y
    try:
        return struct.unpack("<H", struct.pack("<e", product))[0]
    except OverflowError:
        return 0xFC00 if math.copysign(1.0, product) < 0 else 0x7C00


def _half_product_control(mode):
    generator = HLSLCodeGen()
    rounded = generator.hlsl_half_conversion_expression("product", 1, ("float",))
    decoded = "__crossgl_binary16_to_float(uint(asuint16({})))"
    bodies = {
        "native": "return x * y;",
        "precise-native": "precise float16_t product = x * y; return product;",
        "widened-cast": (
            "precise float product = float(x) * float(y); return float16_t(product);"
        ),
        "widened-rounded": (
            f"precise float product = float(x) * float(y); return {rounded};"
        ),
        "decoded-rounded": (
            f"precise float product = {decoded.format('x')} * {decoded.format('y')}; "
            f"return {rounded};"
        ),
    }
    helper = generator.generate_hlsl_half_helpers() if mode.endswith("-rounded") else ""
    if mode == "decoded-rounded":
        generator.require_hlsl_explicit_bitcast_helper("binary16_to_float")
        helper += generator.generate_hlsl_explicit_bitcast_helpers()
    return helper + f"""
StructuredBuffer<uint> inputWords : register(t0);
RWStructuredBuffer<uint> outputWords : register(u1);
float16_t multiply_values(float16_t x, float16_t y) {{
    {bodies[mode]}
}}
[numthreads(1, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {{
    uint pair = inputWords[32u + tid.x];
    float16_t x = asfloat16(uint16_t(pair & 0xffffu));
    float16_t y = asfloat16(uint16_t(pair >> 16u));
    outputWords[32u + tid.x * 2u] = uint(asuint16(multiply_values(x, y)));
    outputWords[33u + tid.x * 2u] = pair;
}}
"""


def test_half_product_reference_covers_overflow_and_underflow_boundaries():
    cases = _half_product_cases()
    assert 63488 <= len(cases) <= 65535
    assert {(word, 0x3C01) for word in range(65536) if word & 0x7C00 != 0x7C00} <= set(
        cases
    )
    assert {(1, 0x3800), (3, 0x3800), (0x8001, 0x3800)} <= set(cases)
    assert _half_product_bits(0x7BFE, 0x3C01) == 0x7C00
    assert _half_product_bits(0xFBFE, 0x3C01) == 0xFC00
    assert _half_product_bits(0x0001, 0x3800) == 0x0000
    assert _half_product_bits(0x0003, 0x3800) == 0x0002
    assert _half_product_bits(0x8001, 0x3800) == 0x8000


@pytest.mark.parametrize("profile", ["cs_6_2", "cs_6_6"])
@pytest.mark.parametrize(
    "mode",
    ["native", "precise-native", "widened-cast", "widened-rounded", "decoded-rounded"],
)
def test_half_product_controls_compile(tmp_path, profile, mode):
    listing = tmp_path / "product.dxil.txt"
    _compile(
        _half_product_control(mode),
        tmp_path,
        profile=profile,
        flags=("-enable-16bit-types", "-Fc", str(listing)),
    )
    arithmetic = listing.read_text(encoding="utf-8")
    if mode.endswith("-rounded"):
        assert "fmul float" in arithmetic
        assert "fmul half" not in arithmetic
    else:
        assert "fmul" in arithmetic


@pytest.mark.parametrize("profile", ["cs_6_2", "cs_6_6"])
@pytest.mark.parametrize(
    "mode",
    ["native", "precise-native", "widened-cast", "widened-rounded", "decoded-rounded"],
)
def test_half_product_controls_execute(tmp_path, profile, mode):
    if os.environ.get("CROSTL_REQUIRE_HALF_CONVERSION_RUNTIME") != "1":
        pytest.skip("Set CROSTL_REQUIRE_HALF_CONVERSION_RUNTIME=1 for native controls")
    assert sys.platform == "win32", "DirectX half controls require Windows"
    assert shutil.which("dxc"), "DXC is required for native half controls"
    from tests.test_translator.test_directx_software_reductions import _execute_words

    cases = _half_product_cases()
    words = [left | (right << 16) for left, right in cases]
    products = [_half_product_bits(left, right) for left, right in cases]
    # The shared runner initializes unwritten output words with this sentinel.
    guard = [0xDEADBEEF] * 32
    expected = guard + [value for pair in zip(products, words) for value in pair]

    def compile_control(generated, directory, *, flags):
        return _compile(generated, directory, profile=profile, flags=flags)

    actual = _execute_words(
        tmp_path,
        _half_product_control(mode),
        guard + words,
        expected,
        workgroup_size=(1, 1, 1),
        workgroup_count=(len(cases), 1, 1),
        compile_flags=("-enable-16bit-types",),
        module_compiler=compile_control,
    )
    assert actual[:32] == guard
    assert actual[33::2] == words
    mismatches = [
        {
            "index": index,
            "operands": list(pair),
            "expectedBits": want,
            "actualBits": got,
        }
        for index, (pair, want, got) in enumerate(zip(cases, products, actual[32::2]))
        if want != got
    ]
    (tmp_path / "comparison.json").write_text(
        json.dumps(
            {
                "controlOnly": True,
                "replacesRequiredGate": False,
                "mode": mode,
                "profile": profile,
                "count": len(cases),
                "requiresParity": mode == "decoded-rounded",
                "mismatches": mismatches,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Native casts characterize the runtime; bit-preserving conversions must be exact.
    if mode == "decoded-rounded":
        assert not mismatches, mismatches[:10]


def test_half_product_controls_keep_required_native_gate():
    from tests import workflow_text
    from tests.ci_helpers import assert_paths_covered

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_step_section(
        workflow, "Validate DirectX half conversions"
    )
    assert "test_directx_half_conversion.py" in step
    assert 'CROSTL_REQUIRE_HALF_CONVERSION_RUNTIME: "1"' in step
    assert "--timeout-seconds" in step and "--junitxml" in step and "-n auto" in step
    (definition,) = yaml.safe_load(step)
    assert " ".join(definition["if"].split()) == (
        "${{ !cancelled() && runner.os == 'Windows' && "
        "steps.host-warp-identity.outcome == 'success' }}"
    )
    assert "continue-on-error" not in step
    for event in ("push", "pull_request"):
        assert_paths_covered(
            workflow_text.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_directx_half_conversion.py",
        )
