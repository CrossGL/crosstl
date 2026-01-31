import pytest
from typing import Iterable, List, Optional
from crosstl.backend.Metal.MetalLexer import MetalLexer


def tokenize_code(code: str) -> List:
    """Tokenize Metal code."""
    lexer = MetalLexer(code)
    return lexer.tokenize()


def token_values(tokens: List) -> List[str]:
    """Return token values as strings for flexible assertions."""
    values = []

    def split_top_level(text: str) -> List[str]:
        parts = []
        buf = ""
        depth = 0
        for ch in text:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                if buf.strip():
                    parts.append(buf.strip())
                buf = ""
                continue
            buf += ch
        if buf.strip():
            parts.append(buf.strip())
        return parts

    for token in tokens:
        if isinstance(token, tuple) and len(token) >= 2:
            token_type, token_value = token[0], str(token[1])
            values.append(token_value)
            if token_type == "ATTRIBUTE":
                content = token_value[2:-2].strip()
                for part in split_top_level(content):
                    name = part.strip().split("(", 1)[0].strip()
                    if name:
                        values.append(name)
        else:
            values.append(str(token))
    return values


def assert_contains(values: List[str], expected: Iterable[str]) -> None:
    missing = [item for item in expected if item not in values]
    assert not missing, f"Missing tokens: {missing}"


def assert_literal_present(
    values: List[str], raw: str, numeric_alt: Optional[int] = None
) -> None:
    if raw in values:
        return
    if numeric_alt is not None and str(numeric_alt) in values:
        return
    assert False, f"Literal {raw} not found in tokens"


def test_tokenizes_structs_functions_and_attributes():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct VertexInput {
        float3 position [[attribute(0)]];
        float3 normal [[attribute(1)]];
        float2 uv [[attribute(5)]];
    };

    struct VertexOutput {
        float4 position [[position]];
        float2 uv;
    };

    vertex VertexOutput vertex_main(VertexInput in [[stage_in]],
                                    constant float4x4& model [[buffer(0)]]) {
        VertexOutput out;
        out.position = model * float4(in.position, 1.0);
        out.uv = in.uv;
        return out;
    }

    fragment float4 fragment_main(VertexOutput in [[stage_in]],
                                  texture2d<float> albedo [[texture(0)]],
                                  sampler samp [[sampler(0)]]) {
        return albedo.sample(samp, in.uv);
    }
    """
    tokens = tokenize_code(code)
    values = token_values(tokens)

    assert_contains(
        values,
        [
            "struct",
            "vertex",
            "fragment",
            "attribute",
            "stage_in",
            "position",
            "buffer",
            "texture",
            "sampler",
            "return",
            "texture2d",
        ],
    )


def test_tokenizes_control_flow_keywords():
    code = """
    void main() {
        int i = 0;
        if (i == 0) {
            i = 1;
        } else if (i == 1) {
            i = 2;
        } else {
            i = 3;
        }

        for (int j = 0; j < 4; j++) {
            if (j == 2) {
                continue;
            }
        }

        while (i < 10) {
            i++;
        }

        do {
            i--;
        } while (i > 0);

        switch (i) {
            case 0:
                i = 1;
                break;
            default:
                i = 2;
                break;
        }

        return;
    }
    """
    tokens = tokenize_code(code)
    values = token_values(tokens)

    assert_contains(
        values,
        [
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
        ],
    )


def test_tokenizes_operators():
    code = """
    void main() {
        int a = 1;
        int b = 2;
        a += b;
        a -= b;
        a *= b;
        a /= b;
        a %= b;

        bool c = (a == b) || (a != b) && (a < b) && (a <= b) && (a > b) && (a >= b);
        bool d = !c;

        a = (a & b) | (a ^ b);
        a = ~a;

        a = a << 1;
        a = a >> 1;
        a <<= 1;
        a >>= 1;
        a &= b;
        a |= b;
        a ^= b;

        a++;
        b--;

        int e = (a > b) ? a : b;
    }
    """
    tokens = tokenize_code(code)
    values = token_values(tokens)

    assert_contains(
        values,
        [
            "+=",
            "-=",
            "*=",
            "/=",
            "%=",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
            "&&",
            "||",
            "!",
            "&",
            "|",
            "^",
            "~",
            "<<",
            ">>",
            "<<=",
            ">>=",
            "&=",
            "|=",
            "^=",
            "++",
            "--",
            "?",
            ":",
        ],
    )


def test_tokenizes_numeric_literals_and_suffixes():
    code = """
    void main() {
        float a = 1.0f;
        half b = 2.5h;
        int c = 42;
        uint d = 17u;
        int e = 0x1A;
        int f = 0b101011;
        float g = 3.14159;
        float h = 0.125;
    }
    """
    values = token_values(tokenize_code(code))

    assert_literal_present(values, "1.0f")
    assert_literal_present(values, "2.5h")
    assert_literal_present(values, "42")
    assert_literal_present(values, "17u")
    assert_literal_present(values, "0x1A", numeric_alt=26)
    assert_literal_present(values, "0b101011", numeric_alt=43)
    assert_literal_present(values, "3.14159")
    assert_literal_present(values, "0.125")


def test_tokenizes_advanced_types_and_qualifiers():
    code = """
    struct Types {
        packed_float4 p4;
        simd_float3 s3;
        simd_float4x4 s44;
        atomic_int counter;
    };

    kernel void main(threadgroup_imageblock float4* img [[threadgroup(0)]],
                     texturecube_array<float> cubeArr [[texture(1)]],
                     texture2d_ms<float> msTex [[texture(2)]],
                     depth2d_array<float> depthArr [[texture(3)]],
                     device int* buffer [[buffer(0)]],
                     read_write int rw) {
        int x = 0;
    }
    """
    values = token_values(tokenize_code(code))
    assert_contains(
        values,
        [
            "packed_float4",
            "simd_float3",
            "simd_float4x4",
            "atomic_int",
            "threadgroup_imageblock",
            "texturecube_array",
            "texture2d_ms",
            "depth2d_array",
            "read_write",
        ],
    )


def test_tokenizes_scoped_identifiers_and_access():
    code = """
    float main(float x) {
        return metal::fast::sin(x);
    }

    texture2d<float, access::read_write> tex;
    """
    values = token_values(tokenize_code(code))
    assert_contains(values, ["metal", "::", "fast", "sin", "access", "read_write"])


def test_tokenizes_attribute_with_multiple_args():
    code = """
    struct Out {
        float4 color [[color(0, rgba8unorm)]];
    };
    """
    values = token_values(tokenize_code(code))
    assert_contains(values, ["color"])


def test_tokenizes_raytracing_types():
    code = """
    acceleration_structure accel;
    intersection_function_table<RayFunc> ift;
    visible_function_table<HitFunc> vft;
    """
    values = token_values(tokenize_code(code))
    assert_contains(
        values,
        [
            "acceleration_structure",
            "intersection_function_table",
            "visible_function_table",
        ],
    )


def test_tokenizes_raytracing_qualifiers():
    code = """
    intersection void isect();
    anyhit void any_hit();
    closesthit void closest_hit();
    miss void miss_main();
    callable void callable_main();
    mesh void mesh_main();
    object void object_main();
    amplification void ampl_main();
    """
    values = token_values(tokenize_code(code))
    assert_contains(
        values,
        [
            "intersection",
            "anyhit",
            "closesthit",
            "miss",
            "callable",
            "mesh",
            "object",
            "amplification",
        ],
    )


def test_tokenizes_enum_and_typedef():
    code = """
    typedef int32_t MyInt;
    enum Mode { Off, On = 2, Auto };
    """
    values = token_values(tokenize_code(code))
    assert_contains(values, ["typedef", "enum", "int32_t"])


def test_tokenizes_sizeof_and_alignof():
    code = """
    void main() {
        int a = sizeof(int);
        int b = alignof(float4);
    }
    """
    values = token_values(tokenize_code(code))
    assert_contains(values, ["sizeof", "alignof"])


def test_tokenizes_alignas_and_static_assert():
    code = """
    alignas(16) float4 alignedValue;
    static_assert(1 == 1, "ok");
    """
    values = token_values(tokenize_code(code))
    assert_contains(values, ["alignas", "static_assert"])


def test_tokenizes_indirect_command_buffer():
    code = """
    indirect_command_buffer icb;
    """
    values = token_values(tokenize_code(code))
    assert_contains(values, ["indirect_command_buffer"])


def test_comments_are_ignored():
    code = """
    void main() {
        int keep = 1; // COMMENT_TOKEN_123
        /* BLOCK_TOKEN_456 */
        int keep2 = 2;
    }
    """
    values = token_values(tokenize_code(code))
    assert "COMMENT_TOKEN_123" not in values
    assert "BLOCK_TOKEN_456" not in values
    assert_contains(values, ["int", "keep", "keep2"])


def test_preprocessor_does_not_break_tokenization():
    code = """
    #include <metal_stdlib>
    #define MY_MACRO 42
    using namespace metal;

    struct S {
        float x;
    };
    """
    values = token_values(tokenize_code(code))
    assert_contains(values, ["struct", "S", "float", "x"])


def test_invalid_character_raises():
    code = "void main() { int a = 1 @ 2; }"
    with pytest.raises(SyntaxError):
        tokenize_code(code)


def test_unterminated_block_comment_raises():
    code = "void main() { /* unterminated comment "
    with pytest.raises(SyntaxError):
        tokenize_code(code)


if __name__ == "__main__":
    pytest.main()
