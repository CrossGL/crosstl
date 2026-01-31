import pytest
from typing import Iterable, List

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


def test_min_precision_types_tokenization():
    code = "min16float a; min10float b; min16int c; min12int d; min16uint e;"
    tokens = tokenize_code(code)
    values = token_values(tokens)
    expected = ["min16float", "min10float", "min16int", "min12int", "min16uint"]
    assert_values_present(values, expected, case_insensitive=True)


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


if __name__ == "__main__":
    pytest.main()
