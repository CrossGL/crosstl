from typing import Iterable, List

import pytest

from crosstl.backend.DirectX.DirectxLexer import HLSLLexer


def tokenize_code(code: str) -> List:
    """Helper function to tokenize HLSL code."""
    lexer = HLSLLexer(code)
    return lexer.tokenize()


def token_values(tokens: List) -> List[str]:
    values = []
    for token in tokens:
        if isinstance(token, tuple) and len(token) >= 2:
            values.append(token[1])
    return [value for value in values if value is not None]


def assert_values_present(
    values: Iterable[str], expected: Iterable[str], case_insensitive: bool = False
):
    normalized = [str(value) for value in values]
    if case_insensitive:
        normalized_lower = [value.lower() for value in normalized]
        missing = [value for value in expected if value.lower() not in normalized_lower]
    else:
        missing = [value for value in expected if value not in normalized]

    assert not missing, f"Missing tokens: {missing}"


def test_keywords_and_types_tokenization():
    code = """
    cbuffer CameraBuffer : register(b0) {
        float4x4 viewProj;
        float3 eyePos;
        float padding;
    };

    Texture2D tex0 : register(t0);
    SamplerState samp0 : register(s0);

    struct VSInput {
        float3 position : POSITION;
        float2 uv : TEXCOORD0;
    };

    struct VSOutput {
        float4 position : SV_Position;
        float2 uv : TEXCOORD0;
    };

    double d = 1.0;
    half h = 0.5h;
    bool flag = true;
    int i = -1;
    uint u = 2u;

    float4 VSMain(VSInput input) : SV_Position {
        return float4(input.position, 1.0);
    }
    """
    tokens = tokenize_code(code)
    values = token_values(tokens)

    expected = [
        "struct",
        "cbuffer",
        "register",
        "texture2d",
        "samplerstate",
        "float4x4",
        "float3",
        "float2",
        "float4",
        "double",
        "half",
        "bool",
        "int",
        "uint",
        "return",
        "sv_position",
        "texcoord0",
    ]
    assert_values_present(values, expected, case_insensitive=True)


def test_one_row_column_matrix_aliases_from_dxc_matrix_syntax_tokenize_as_matrices():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/SemaHLSL/matrix-syntax.hlsl
    tokens = tokenize_code("""
    void matrix_on_demand() {
        float1x2 f12;
        bool2x1 boolMatrix;
        unsigned int4x2 unsignedMatrix;
    }
    """)

    assert ("MATRIX", "float1x2") in tokens
    assert ("MATRIX", "bool2x1") in tokens
    assert ("MATRIX", "int4x2") in tokens


def test_control_flow_keywords_tokenization():
    code = """
    int ControlFlow(int a, int b) {
        int result = 0;
        if (a > b) {
            result = a;
        } else if (a == b) {
            result = 0;
        } else {
            result = b;
        }

        for (int i = 0; i < 4; ++i) {
            if (i == 2) {
                continue;
            }
            result += i;
        }

        int j = 0;
        while (j < 2) {
            j++;
        }

        do {
            result--;
        } while (result > 0);

        switch (a) {
            case 0:
                result += 1;
                break;
            default:
                result += 2;
                break;
        }

        return result;
    }
    """
    tokens = tokenize_code(code)
    values = token_values(tokens)

    expected = [
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "default",
        "break",
        "continue",
        "return",
    ]
    assert_values_present(values, expected, case_insensitive=True)


def test_operator_tokenization():
    code = """
    int a = 1;
    int b = 2;
    int c = 0;
    c = a + b;
    c = a - b;
    c = a * b;
    c = a / b;
    c = a % b;
    c += a;
    c -= b;
    c *= a;
    c /= b;
    c %= b;
    c <<= 1;
    c >>= 1;
    c &= a;
    c |= b;
    c ^= a;
    bool ok = (a == b) || (a != b) && (a < b) && (a <= b) && (a > b) && (a >= b);
    int d = (a & b) | (a ^ b);
    int e = ~a;
    int f = (a << 1) + (b >> 1);
    int g = a > b ? a : b;
    a++;
    --b;
    """
    tokens = tokenize_code(code)
    values = token_values(tokens)

    expected = [
        "+",
        "-",
        "*",
        "/",
        "%",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "<<",
        ">>",
        "<<=",
        ">>=",
        "&",
        "|",
        "^",
        "~",
        "&&",
        "||",
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        "?",
        ":",
        "++",
        "--",
    ]
    assert_values_present(values, expected)


def test_numeric_literals_tokenization():
    code = """
    float a = 1.0;
    float b = 1.0f;
    float c = 0.5;
    float d = 1e-3;
    float e = 6.02e23;
    half h = 2.5h;
    int i = 42;
    uint u0 = 0u;
    uint u1 = 123u;
    int hexVal = 0xFF;
    """
    tokens = tokenize_code(code)
    values = token_values(tokens)

    expected = [
        "1.0",
        "1.0f",
        "0.5",
        "1e-3",
        "6.02e23",
        "2.5h",
        "42",
        "0u",
        "123u",
        "0xFF",
    ]
    assert_values_present(values, expected)


def test_scalar_literal_swizzle_keeps_dot_token():
    tokens = tokenize_code("float4 clear = 0.xxxx; float a = 1.; float b = 1.f;")

    assert tokens[3:6] == [("NUMBER", "0"), ("DOT", "."), ("IDENTIFIER", "xxxx")]
    assert ("NUMBER", "0.") not in tokens
    assert ("NUMBER", "1.") in tokens
    assert ("NUMBER", "1.f") in tokens


def test_legacy_special_float_literal_from_directx_graphics_samples():
    tokens = tokenize_code("const float FLT_INFINITY = 1.#INF;")

    assert ("NUMBER", "1.#INF") in tokens
    assert ("PREPROCESSOR", "#INF;") not in tokens


def test_unsigned_long_long_integer_suffixes_from_directx_samples():
    tokens = tokenize_code(
        "uint64_t a = 1ull; uint64_t b = 1llu; uint64_t c = 0xffull;"
    )

    assert ("NUMBER", "1ull") in tokens
    assert ("NUMBER", "1llu") in tokens
    assert ("HEX_NUMBER", "0xffull") in tokens
    assert ("IDENTIFIER", "l") not in tokens


def test_min_precision_types_tokenization():
    code = "min16float a; min10float b; min16int c; min12int d; min16uint e;"
    tokens = tokenize_code(code)
    values = token_values(tokens)
    expected = ["min16float", "min10float", "min16int", "min12int", "min16uint"]
    assert_values_present(values, expected, case_insensitive=True)


def test_exact_16_bit_scalar_types_tokenization_from_hlsl_docs():
    # Source: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-scalar
    tokens = tokenize_code("float16_t f; int16_t i; uint16_t u;")

    assert ("FLOAT16_T", "float16_t") in tokens
    assert ("INT16_T", "int16_t") in tokens
    assert ("UINT16_T", "uint16_t") in tokens
    assert ("IDENTIFIER", "float16_t") not in tokens
    assert ("IDENTIFIER", "int16_t") not in tokens
    assert ("IDENTIFIER", "uint16_t") not in tokens


def test_interpolation_modifiers_tokenization():
    code = """
    struct PSInput {
        linear float2 uv0 : TEXCOORD0;
        centroid noperspective float2 uv1 : TEXCOORD1;
        nointerpolation uint id : TEXCOORD2;
        sample float4 color : COLOR0;
    };
    """
    tokens = tokenize_code(code)

    assert ("LINEAR", "linear") in tokens
    assert ("CENTROID", "centroid") in tokens
    assert ("NOPERSPECTIVE", "noperspective") in tokens
    assert ("NOINTERPOLATION", "nointerpolation") in tokens
    assert ("SAMPLE", "sample") in tokens
    assert ("IDENTIFIER", "noperspective") not in tokens


def test_min_precision_vector_and_matrix_types_tokenization():
    code = """
    min16float3 hdr;
    min10float2 uv;
    min16int4 signedIndex;
    min12int2 smallSigned;
    min16uint4 mask;
    min16float2x3 colorMatrix;
    min10float4x4 transform;
    """
    tokens = tokenize_code(code)

    assert ("FVECTOR", "min16float3") in tokens
    assert ("FVECTOR", "min10float2") in tokens
    assert ("IVECTOR", "min16int4") in tokens
    assert ("IVECTOR", "min12int2") in tokens
    assert ("UVECTOR", "min16uint4") in tokens
    assert ("MATRIX", "min16float2x3") in tokens
    assert ("MATRIX", "min10float4x4") in tokens
    assert ("IDENTIFIER", "min16float3") not in tokens


def test_swizzle_and_member_access_tokenization():
    code = """
    struct V {
        float4 color : TEXCOORD0;
        float3 pos : POSITION;
    };

    float3 UseSwizzle(V input) {
        float3 p = input.pos.xyz;
        float4 c = float4(p, 1.0).rgba;
        float2 uv = input.color.xy;
        float2 zw = input.color.zw;
        return p + float3(uv, 0.0);
    }
    """
    tokens = tokenize_code(code)
    values = token_values(tokens)

    expected = ["xyz", "rgba", "xy", "zw"]
    assert_values_present(values, expected)


def test_comments_are_ignored():
    code = """
    // COMMENT_SHOULD_NOT_APPEAR
    float4 main() : SV_Target0 {
        /* BLOCK_COMMENT_SHOULD_NOT_APPEAR */
        return float4(1.0, 1.0, 1.0, 1.0); // TRAILING_COMMENT_SHOULD_NOT_APPEAR
    }
    """
    tokens = tokenize_code(code)
    values = token_values(tokens)
    joined = " ".join(str(value) for value in values)

    assert "COMMENT_SHOULD_NOT_APPEAR" not in joined
    assert "BLOCK_COMMENT_SHOULD_NOT_APPEAR" not in joined
    assert "TRAILING_COMMENT_SHOULD_NOT_APPEAR" not in joined


def test_unterminated_comment_raises():
    code = "float4 main() : SV_Target0 { /* unterminated "
    with pytest.raises(SyntaxError):
        tokenize_code(code)


def test_invalid_character_raises():
    code = "int a = 0; `"
    with pytest.raises(SyntaxError):
        tokenize_code(code)


def test_hex_escape_char_literals_tokenization_from_dxc_rewriter():
    code = r"""
    void expressions() {
        int local_i;
        local_i = 'c';
        local_i = '\xff';
        local_i = '\x94';
        local_i = '\123';
    }
    """
    tokens = tokenize_code(code)

    assert ("CHAR_LITERAL", "'c'") in tokens
    assert ("CHAR_LITERAL", r"'\xff'") in tokens
    assert ("CHAR_LITERAL", r"'\x94'") in tokens
    assert ("CHAR_LITERAL", r"'\123'") in tokens


def test_preprocessor_directives_tokenization():
    code = """
    #define USE_LIGHTING 1
    #if USE_LIGHTING
    float3 Lighting(float3 n) { return n; }
    #endif
    #pragma pack_matrix(row_major)
    #include "common.hlsl"

    float4 main() : SV_Target0 {
        return float4(1.0, 1.0, 1.0, 1.0);
    }
    """
    tokens = tokenize_code(code)
    values = token_values(tokens)
    assert_values_present(values, ["float4"], case_insensitive=True)


def test_resource_types_tokenization():
    code = """
    Texture2DMS texms;
    Texture2DMSArray texmsArray;
    FeedbackTexture2D feedbackTex;
    FeedbackTexture2DArray feedbackTexArray;
    RWTexture1DArray rwTex1DArray;
    RasterizerOrderedTexture1D<float4> rovTex1D;
    RasterizerOrderedTexture1DArray<uint> rovTex1DArray;
    RasterizerOrderedTexture2D<uint> rovTex2D;
    RasterizerOrderedTexture2DArray<float4> rovTex2DArray;
    RasterizerOrderedTexture3D<int> rovTex3D;
    RasterizerOrderedBuffer<uint> rovBuffer;
    RasterizerOrderedStructuredBuffer<int> rovStructured;
    RasterizerOrderedByteAddressBuffer rovBytes;
    RaytracingAccelerationStructure accel;
    RayQuery rayQuery;
    InputPatch<VSInput, 3> patch;
    OutputPatch<VSOutput, 3> outPatch;
    PointStream<GSOut> pointStream;
    LineStream<GSOut> lineStream;
    TriangleStream<GSOut> triStream;
    AppendStructuredBuffer<float4> appendBuf;
    ConsumeStructuredBuffer<float4> consumeBuf;
    """
    tokens = tokenize_code(code)
    values = token_values(tokens)
    expected = [
        "Texture2DMS",
        "Texture2DMSArray",
        "FeedbackTexture2D",
        "FeedbackTexture2DArray",
        "RWTexture1DArray",
        "RasterizerOrderedTexture1D",
        "RasterizerOrderedTexture1DArray",
        "RasterizerOrderedTexture2D",
        "RasterizerOrderedTexture2DArray",
        "RasterizerOrderedTexture3D",
        "RasterizerOrderedBuffer",
        "RasterizerOrderedStructuredBuffer",
        "RasterizerOrderedByteAddressBuffer",
        "RaytracingAccelerationStructure",
        "RayQuery",
        "InputPatch",
        "OutputPatch",
        "PointStream",
        "LineStream",
        "TriangleStream",
        "AppendStructuredBuffer",
        "ConsumeStructuredBuffer",
    ]
    assert_values_present(values, expected, case_insensitive=False)

    token_types_by_value = {
        token_value: token_type for token_type, token_value in tokens if token_value
    }
    assert (
        token_types_by_value["RasterizerOrderedTexture2D"]
        == "RASTERIZERORDEREDTEXTURE2D"
    )
    assert (
        token_types_by_value["RasterizerOrderedStructuredBuffer"]
        == "RASTERIZERORDEREDSTRUCTUREDBUFFER"
    )
    assert (
        token_types_by_value["RasterizerOrderedByteAddressBuffer"]
        == "RASTERIZERORDEREDBYTEADDRESSBUFFER"
    )


def test_register_and_packoffset_tokenization():
    code = """
    cbuffer FrameData : register(b0, space1) {
        float4x4 viewProj : packoffset(c0);
        float4 color : packoffset(c1.y);
    };
    Texture2D tex0 : register(t0, space2);
    SamplerState samp0 : register(s0, space2);
    """
    tokens = tokenize_code(code)
    values = token_values(tokens)
    expected = ["register", "packoffset", "b0", "space1", "t0", "space2", "s0", "c0"]
    assert_values_present(values, expected, case_insensitive=True)


def test_hlsl_define_is_enabled_for_default_preprocessing():
    code = """
    #ifndef HLSL
    struct float2 { float x, y; };
    #endif
    float main() { return 0.0; }
    """

    tokens = tokenize_code(code)
    values = token_values(tokens)

    assert "struct" not in values
    assert_values_present(values, ["float", "main", "return"])


def test_from_file_decodes_legacy_comment_bytes_with_replacement(tmp_path):
    source = tmp_path / "legacy_comment.hlsl"
    source.write_bytes(b"// Matthias M\xfcller\nfloat main() { return 0.0; }\n")

    tokens = HLSLLexer.from_file(str(source)).tokenize()
    values = token_values(tokens)

    assert_values_present(values, ["float", "main", "return"])


if __name__ == "__main__":
    pytest.main()
