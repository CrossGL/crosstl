import pytest
from typing import List, Tuple

from crosstl.backend.GLSL.OpenglLexer import GLSLLexer

TokenPair = Tuple[str, object]
IGNORED_TOKEN_TYPES = {"EOF", "NEWLINE", "NL"}


def tokenize_code(code: str):
    """Helper function to tokenize GLSL code."""
    lexer = GLSLLexer(code)
    return lexer.tokenize()


def normalize_tokens(tokens) -> List[TokenPair]:
    normalized = []
    for token in tokens:
        if isinstance(token, tuple):
            if len(token) >= 2:
                token_type, token_value = token[0], token[1]
            elif len(token) == 1:
                token_type, token_value = token[0], None
            else:
                raise AssertionError("Empty token tuple encountered")
        elif hasattr(token, "type"):
            token_type = token.type
            token_value = getattr(token, "value", None)
        else:
            raise AssertionError(f"Unexpected token format: {token!r}")
        normalized.append((token_type, token_value))
    return normalized


def strip_ignored(tokens: List[TokenPair]) -> List[TokenPair]:
    return [token for token in tokens if token[0] not in IGNORED_TOKEN_TYPES]


def token_values(tokens: List[TokenPair]) -> List[object]:
    return [value for _, value in tokens if value is not None]


def first_token_type(code: str) -> str:
    tokens = strip_ignored(normalize_tokens(tokenize_code(code)))
    assert tokens, f"No tokens produced for input: {code!r}"
    return tokens[0][0]


def identifier_token_type() -> str:
    return first_token_type("identifier")


def token_type_for_value(tokens: List[TokenPair], value: str) -> str:
    for token_type, token_value in tokens:
        if token_value == value:
            return token_type
    raise AssertionError(f"Token value {value!r} not found in {tokens}")


def assert_contains_value(tokens: List[TokenPair], value: str):
    values = token_values(tokens)
    assert value in values, f"Expected token value {value!r} in {values}"


def test_empty_input():
    tokens = strip_ignored(normalize_tokens(tokenize_code("")))
    assert tokens == []


def test_whitespace_only_input():
    tokens = strip_ignored(normalize_tokens(tokenize_code("   \n\t  \n")))
    assert tokens == []


@pytest.mark.parametrize(
    "keyword",
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
        "discard",
        "struct",
        "layout",
        "in",
        "out",
        "uniform",
        "const",
        "precision",
        "attribute",
        "varying",
        "inout",
    "flat",
    "smooth",
    "noperspective",
    "centroid",
    "sample",
    "patch",
    "invariant",
    "precise",
    "coherent",
    "volatile",
    "restrict",
    "readonly",
    "writeonly",
    "shared",
    "buffer",
    ],
)
def test_keywords_are_reserved(keyword):
    assert first_token_type(keyword) != identifier_token_type()


@pytest.mark.parametrize(
    "dtype",
    [
        "void",
        "bool",
        "int",
        "uint",
        "float",
        "double",
        "vec2",
        "vec3",
        "vec4",
        "ivec2",
        "ivec3",
        "uvec2",
        "uvec3",
        "bvec2",
        "bvec3",
        "mat2",
        "mat3",
        "mat4",
        "sampler2D",
        "samplerCube",
        "sampler1D",
        "sampler1DArray",
        "sampler2DArray",
        "samplerCubeArray",
        "sampler2DShadow",
        "sampler2DArrayShadow",
        "samplerCubeArrayShadow",
        "sampler2DRect",
        "samplerBuffer",
        "sampler2DMS",
        "sampler2DMSArray",
        "isampler2D",
        "usampler2D",
        "image2D",
        "image3D",
        "imageCube",
        "image2DArray",
        "imageBuffer",
        "iimage2D",
        "uimage2D",
        "atomic_uint",
        "subroutine",
    ],
)
def test_builtin_types_are_reserved(dtype):
    assert first_token_type(dtype) != identifier_token_type()


@pytest.mark.parametrize("literal", ["0", "42", "0xFF", "1.0", ".5", "1.", "1e-3", "2E+4"])
def test_numeric_literals_tokenize(literal):
    tokens = strip_ignored(normalize_tokens(tokenize_code(f"float x = {literal};")))
    assert_contains_value(tokens, literal)
    assert token_type_for_value(tokens, literal) != identifier_token_type()


@pytest.mark.parametrize("literal", ["true", "false"])
def test_boolean_literals_tokenize(literal):
    assert first_token_type(literal) != identifier_token_type()


@pytest.mark.parametrize(
    "operator, code",
    [
        ("==", "void main() { bool c = 1 == 2; }"),
        ("!=", "void main() { bool c = 1 != 2; }"),
        ("<=", "void main() { bool c = 1 <= 2; }"),
        (">=", "void main() { bool c = 1 >= 2; }"),
        ("&&", "void main() { bool c = true && false; }"),
        ("||", "void main() { bool c = true || false; }"),
        ("<<", "void main() { int a = 1 << 2; }"),
        (">>", "void main() { int a = 4 >> 1; }"),
        ("<<=", "void main() { int a = 1; a <<= 2; }"),
        (">>=", "void main() { int a = 1; a >>= 2; }"),
        ("++", "void main() { int a = 0; a++; }"),
        ("--", "void main() { int a = 0; --a; }"),
        ("+=", "void main() { int a = 0; a += 1; }"),
        ("-=", "void main() { int a = 0; a -= 1; }"),
        ("*=", "void main() { int a = 1; a *= 2; }"),
        ("/=", "void main() { int a = 4; a /= 2; }"),
        ("%=", "void main() { int a = 4; a %= 2; }"),
        ("&=", "void main() { int a = 1; a &= 3; }"),
        ("|=", "void main() { int a = 1; a |= 3; }"),
        ("^=", "void main() { int a = 1; a ^= 3; }"),
        ("?", "void main() { int a = 1; int b = 2; int c = (a > b) ? a : b; }"),
        (":", "void main() { int a = 1; int b = 2; int c = (a > b) ? a : b; }"),
    ],
)
def test_compound_operator_tokens(operator, code):
    tokens = strip_ignored(normalize_tokens(tokenize_code(code)))
    assert_contains_value(tokens, operator)


def test_assignment_and_equality_tokens_are_distinct():
    assign_tokens = strip_ignored(
        normalize_tokens(tokenize_code("void main() { int a = 1; }"))
    )
    eq_tokens = strip_ignored(
        normalize_tokens(tokenize_code("void main() { bool b = 1 == 1; }"))
    )
    assert token_type_for_value(assign_tokens, "=") != token_type_for_value(
        eq_tokens, "=="
    )


def test_comments_are_ignored():
    code = "void main() { int a = 1; }"
    with_comment = "void main() { int a = 1; } // comment"
    tokens = strip_ignored(normalize_tokens(tokenize_code(code)))
    tokens_with_comment = strip_ignored(normalize_tokens(tokenize_code(with_comment)))
    assert tokens_with_comment == tokens


def test_block_comments_are_ignored():
    code = "void main() { int a = 1; }"
    with_comment = "void main() { /* block comment */ int a = 1; }"
    tokens = strip_ignored(normalize_tokens(tokenize_code(code)))
    tokens_with_comment = strip_ignored(normalize_tokens(tokenize_code(with_comment)))
    assert tokens_with_comment == tokens


def test_unterminated_block_comment_raises():
    with pytest.raises(SyntaxError):
        tokenize_code("void main() { /* unterminated comment }")


def test_preprocessor_version_directive_tokens():
    code = "#version 450 core\nvoid main() { gl_Position = vec4(1.0); }"
    tokens = strip_ignored(normalize_tokens(tokenize_code(code)))
    assert_contains_value(tokens, "version")
    assert_contains_value(tokens, "450")
    assert_contains_value(tokens, "core")
    assert any(
        token_value == "#"
        or token_type in {"HASH", "PP_HASH", "PREPROCESSOR"}
        for token_type, token_value in tokens
    )


def test_layout_qualifier_tokens():
    code = "layout(location = 0) in vec3 position;"
    tokens = strip_ignored(normalize_tokens(tokenize_code(code)))
    assert_contains_value(tokens, "layout")
    assert_contains_value(tokens, "location")
    assert_contains_value(tokens, "0")
    assert_contains_value(tokens, "in")
    assert_contains_value(tokens, "vec3")
    assert_contains_value(tokens, "position")


def test_swizzle_and_member_access_tokens():
    code = "void main() { vec4 color = vec4(1.0); float r = color.rgb.r; }"
    tokens = strip_ignored(normalize_tokens(tokenize_code(code)))
    values = token_values(tokens)
    assert "color" in values
    assert "rgb" in values
    assert "r" in values
    assert "." in values


def test_mod_operator_tokenization():
    tokens = strip_ignored(
        normalize_tokens(tokenize_code("void main() { int a = 10 % 3; }"))
    )
    assert_contains_value(tokens, "%")


def test_bitwise_not_operator_tokenization():
    tokens = strip_ignored(
        normalize_tokens(tokenize_code("void main() { int a = ~5; }"))
    )
    assert_contains_value(tokens, "~")


def test_unsigned_int_type_tokenization():
    tokens = strip_ignored(
        normalize_tokens(tokenize_code("void main() { uint a = 3u; }"))
    )
    assert_contains_value(tokens, "uint")


def test_input_output_tokenization():
    code = """
    #version 330 core
    layout(location = 0) in vec3 position;
    out vec2 vUV;
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;
    """
    tokens = strip_ignored(normalize_tokens(tokenize_code(code)))
    assert_contains_value(tokens, "layout")
    assert_contains_value(tokens, "in")
    assert_contains_value(tokens, "out")
    assert_contains_value(tokens, "vec3")
    assert_contains_value(tokens, "vec4")


def test_control_flow_tokenization():
    code = """
    if (a > b) { return a; }
    else if (a < b) { return b; }
    else { return 0; }
    for (int i = 0; i < 10; i = i + 1) { sum += i; }
    """
    tokens = strip_ignored(normalize_tokens(tokenize_code(code)))
    values = token_values(tokens)
    assert "if" in values
    assert "else" in values
    assert "for" in values


def test_function_call_tokenization():
    code = """
    float perlinNoise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }
    void main() {
        float noise = perlinNoise(vec2(0.0));
    }
    """
    tokens = strip_ignored(normalize_tokens(tokenize_code(code)))
    assert_contains_value(tokens, "perlinNoise")
    assert_contains_value(tokens, "fract")
    assert_contains_value(tokens, "sin")
    assert_contains_value(tokens, "dot")


def test_illegal_character_raises():
    with pytest.raises(SyntaxError):
        tokenize_code("int a = 1 `")


if __name__ == "__main__":
    pytest.main()
