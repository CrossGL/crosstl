"""HLSL ``half`` is binary32 unless the program is compiled with native 16-bit types (crosstl/backend/DirectX/half_width.py)."""

import textwrap

import pytest

import crosstl
from crosstl.backend.DirectX.DirectxCrossGLCodeGen import HLSLToCrossGLConverter
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.backend.DirectX.half_width import (
    binary32_half_token,
    resolve_native_16_bit_types,
)
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

# The coverage probe of the HLSL type ``half``: DXC compiles it without -enable-16bit-types, so v * v is a binary32
# product, and Direct3D 12 WARP writes 0x3f99ab6e for fin[0] = 1.0956935 where binary16 arithmetic gives 0x3f99a000.
HALF_PROBE = textwrap.dedent("""
    RWStructuredBuffer<float> fin : register(u0);
    RWStructuredBuffer<float> outp : register(u1);
    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {
        uint i = id.x;
        half v = half(fin[i]);
        outp[i] = float(v * v);
    }
""")


def to_crossgl(source, **lexer_options):
    tokens = HLSLLexer(source, **lexer_options).tokenize()
    output = HLSLToCrossGLConverter().generate(HLSLParser(tokens).parse())
    CrossGLParser(CrossGLLexer(output).get_tokens()).parse()
    return output


def test_half_is_binary32_without_native_16_bit_types():
    crossgl = to_crossgl(HALF_PROBE)

    assert "float v = float(fin[i]);" in crossgl
    assert "float16" not in crossgl


@pytest.mark.parametrize(
    "declaration, expected",
    [
        ("half a = 1.5h;", "float a = 1.5;"),
        ("half3 b = half3(1, 2, 3);", "vec3 b = vec3(1, 2, 3);"),
        ("half2x3 m = (half2x3)0;", "mat3x2 m"),
        ("vector<half, 2> c = vector<half, 2>(1, 2);", "vec2 c = vec2(1, 2);"),
        ("half1 d = (half1)2;", "float d"),
    ],
)
def test_every_half_spelling_is_binary32_without_native_16_bit_types(
    declaration, expected
):
    crossgl = to_crossgl(f"[numthreads(1, 1, 1)] void main() {{ {declaration} }}")

    assert expected in crossgl
    assert "f16" not in crossgl and "float16" not in crossgl


def test_half_is_binary16_where_the_program_names_a_native_16_bit_type():
    # float16_t, int16_t and uint16_t compile only with -enable-16bit-types, under which half is float16_t.
    crossgl = to_crossgl(HALF_PROBE.replace("uint i = id.x;", "uint16_t i = id.x;"))

    assert "float16 v = float16(fin[i]);" in crossgl


def test_the_source_option_states_native_16_bit_types():
    assert "float16 v = float16(fin[i]);" in to_crossgl(
        HALF_PROBE, enable_16bit_types=True
    )
    assert "float v = float(fin[i]);" in to_crossgl(
        HALF_PROBE.replace("uint i = id.x;", "uint16_t i = id.x;"),
        enable_16bit_types=False,
    )


def test_the_source_option_defines_dxcs_16_bit_macro():
    source = textwrap.dedent("""
        #if __HLSL_ENABLE_16_BIT
        typedef half real;
        #else
        typedef double real;
        #endif
        [numthreads(1, 1, 1)] void main() { real r = 1; }
    """)

    assert "float16 r" in to_crossgl(source, enable_16bit_types=True)
    assert "double r" in to_crossgl(source)


def test_translate_passes_the_source_option_to_the_importer(tmp_path):
    path = tmp_path / "half.hlsl"
    path.write_text(HALF_PROBE)

    assert "float v = float(fin[i]);" in crosstl.translate(str(path), backend="cgl")
    assert "float16 v = float16(fin[i]);" in crosstl.translate(
        str(path), backend="cgl", source_options={"enable_16bit_types": True}
    )


def test_coverage_imports_hlsl_probes_with_native_16_bit_types():
    # The coverage validators compile every HLSL probe with -enable-16bit-types, so its importer is told the same.
    from tools.coverage import runner

    crossgl = runner.translate(HALF_PROBE, ".hlsl", "cgl", "directx")

    assert "float16 v = float16(fin[i]);" in crossgl


def test_hlsl_output_of_a_binary32_half_needs_no_16_bit_types(tmp_path):
    path = tmp_path / "half.hlsl"
    path.write_text(HALF_PROBE)

    hlsl = crosstl.translate(str(path), backend="directx")

    assert "float16_t" not in hlsl
    assert "float v = float(fin[i]);" in hlsl


@pytest.mark.parametrize(
    "token, expected",
    [
        (("HALF", "half"), ("FLOAT", "float")),
        (("FVECTOR", "half4"), ("FVECTOR", "float4")),
        (("FVECTOR", "half1"), ("FVECTOR", "float1")),
        (("MATRIX", "half3x2"), ("MATRIX", "float3x2")),
        (("FVECTOR", "float4"), None),
        (("IDENTIFIER", "halfWidth"), None),
        (("MATRIX", "min16float2x2"), None),
    ],
)
def test_binary32_half_token(token, expected):
    assert binary32_half_token(*token) == expected


@pytest.mark.parametrize(
    "code, option, expected",
    [
        ("half h;", None, False),
        ("float16_t2 h;", None, True),
        ("int16_t h;", None, True),
        ("uint16_t3x3 h;", None, True),
        ("min16float h; float16_tx;", None, False),
        ("half h;", True, True),
        ("float16_t h;", False, False),
    ],
)
def test_resolve_native_16_bit_types(code, option, expected):
    assert resolve_native_16_bit_types(code, option) is expected
