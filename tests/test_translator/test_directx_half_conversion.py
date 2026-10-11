"""Source binary16 nearest-even conversions with native DirectX storage."""

import shutil

import pytest

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
