from hypothesis import given, settings
from hypothesis import strategies as st

from crosstl.translator.lexer import Lexer

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)

LONGEST_OPERATOR_CASES = (
    ("ASSIGN_SHIFT_LEFT", "<<="),
    ("ASSIGN_SHIFT_RIGHT", ">>="),
    ("SPACESHIP", "<=>"),
    ("LESS_EQUAL", "<="),
    ("GREATER_EQUAL", ">="),
    ("EQUAL", "=="),
    ("NOT_EQUAL", "!="),
    ("LOGICAL_AND", "&&"),
    ("LOGICAL_OR", "||"),
    ("BITWISE_SHIFT_LEFT", "<<"),
    ("BITWISE_SHIFT_RIGHT", ">>"),
    ("INCREMENT", "++"),
    ("DECREMENT", "--"),
    ("POWER", "**"),
    ("ARROW", "->"),
    ("FAT_ARROW", "=>"),
    ("DOUBLE_COLON", "::"),
    ("RANGE_INCLUSIVE", "..="),
    ("RANGE", ".."),
    ("ELVIS", "?:"),
    ("ASSIGN_ADD", "+="),
    ("ASSIGN_SUB", "-="),
    ("ASSIGN_MUL", "*="),
    ("ASSIGN_DIV", "/="),
    ("ASSIGN_MOD", "%="),
    ("ASSIGN_AND", "&="),
    ("ASSIGN_OR", "|="),
    ("ASSIGN_XOR", "^="),
)

RESOURCE_TOKEN_CASES = (
    ("TEXTURE1D", "texture1d"),
    ("TEXTURE2D", "texture2d"),
    ("TEXTURE3D", "texture3d"),
    ("TEXTURECUBE", "texturecube"),
    ("TEXTURE2DARRAY", "texture2darray"),
    ("SAMPLER", "sampler"),
    ("SAMPLER1D", "sampler1d"),
    ("SAMPLER1DARRAY", "sampler1DArray"),
    ("SAMPLER2D", "sampler2d"),
    ("SAMPLER3D", "sampler3d"),
    ("SAMPLERCUBE", "samplercube"),
    ("SAMPLER2DARRAY", "sampler2darray"),
    ("SAMPLER2DSHADOW", "sampler2dshadow"),
    ("SAMPLER2DARRAYSHADOW", "sampler2darrayshadow"),
    ("SAMPLERCUBESHADOW", "samplercubeshadow"),
    ("SAMPLERCUBEARRAY", "samplercubearray"),
    ("SAMPLERCUBEARRAYSHADOW", "samplercubearrayshadow"),
    ("SAMPLER2DMS", "sampler2dms"),
    ("SAMPLER2DMSARRAY", "sampler2dmsarray"),
    ("IIMAGE1D", "iimage1D"),
    ("IIMAGE1DARRAY", "iimage1DArray"),
    ("IIMAGE2D", "iimage2D"),
    ("IIMAGE3D", "iimage3D"),
    ("IIMAGE2DARRAY", "iimage2DArray"),
    ("IIMAGE2DMS", "iimage2DMS"),
    ("IIMAGE2DMSARRAY", "iimage2DMSArray"),
    ("UIMAGE1D", "uimage1D"),
    ("UIMAGE1DARRAY", "uimage1DArray"),
    ("UIMAGE2D", "uimage2D"),
    ("UIMAGE3D", "uimage3D"),
    ("UIMAGE2DARRAY", "uimage2DArray"),
    ("UIMAGE2DMS", "uimage2DMS"),
    ("UIMAGE2DMSARRAY", "uimage2DMSArray"),
    ("IMAGE1D", "image1D"),
    ("IMAGE1DARRAY", "image1DArray"),
    ("IMAGE2D", "image2D"),
    ("IMAGE3D", "image3D"),
    ("IMAGECUBE", "imageCube"),
    ("IMAGE2DARRAY", "image2DArray"),
    ("IMAGE2DMS", "image2DMS"),
    ("IMAGE2DMSARRAY", "image2DMSArray"),
)

STAGE_ALIAS_CASES = (
    ("VERTEX", "vertex"),
    ("FRAGMENT", "fragment"),
    ("COMPUTE", "compute"),
    ("GEOMETRY", "geometry"),
    ("TESSELLATION_CONTROL", "tessellation_control"),
    ("TESSELLATION_EVALUATION", "tessellation_evaluation"),
    ("TESSELLATION_CONTROL", "hull"),
    ("TESSELLATION_EVALUATION", "domain"),
    ("TASK", "task"),
    ("AMPLIFICATION", "amplification"),
    ("OBJECT", "object"),
    ("MESH", "mesh"),
    ("RAY_GENERATION", "ray_generation"),
    ("RAY_INTERSECTION", "ray_intersection"),
    ("RAY_INTERSECTION", "intersection"),
    ("RAY_CLOSEST_HIT", "ray_closest_hit"),
    ("RAY_CLOSEST_HIT", "closesthit"),
    ("RAY_MISS", "ray_miss"),
    ("RAY_MISS", "miss"),
    ("RAY_ANY_HIT", "ray_any_hit"),
    ("RAY_ANY_HIT", "anyhit"),
    ("RAY_CALLABLE", "ray_callable"),
    ("RAY_CALLABLE", "callable"),
)


def _tokens(code):
    return Lexer(code).get_tokens()


def _without_eof(tokens):
    return [token for token in tokens if token[0] != "EOF"]


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    value=st.integers(min_value=0, max_value=4096),
)
def test_generated_comments_and_whitespace_are_parser_trivia(suffix, value):
    variable_name = f"value_{suffix}"
    code = f"""
    // shader compute fragment {variable_name} {value}
    shader Generated_{suffix} {{
        /*
           ray_generation closesthit miss {variable_name}
        */
        compute {{
            void main() {{
                int {variable_name} = {value};
            }}
        }}
    }}
    """

    tokens = _without_eof(_tokens(code))
    token_types = [token_type for token_type, _ in tokens]
    token_values = [text for _, text in tokens]

    assert "COMMENT_SINGLE" not in token_types
    assert "COMMENT_MULTI" not in token_types
    assert token_types.count("SHADER") == 1
    assert token_types.count("COMPUTE") == 1
    assert "fragment" not in token_values
    assert "ray_generation" not in token_values
    assert ("IDENTIFIER", variable_name) in tokens
    assert ("NUMBER", str(value)) in tokens


@settings(max_examples=25, deadline=None)
@given(
    keyword_case=st.sampled_from(STAGE_ALIAS_CASES),
    suffix=IDENTIFIER_SUFFIXES,
)
def test_generated_keywords_only_match_at_identifier_boundaries(
    keyword_case,
    suffix,
):
    expected_type, keyword = keyword_case
    suffixed_identifier = f"{keyword}_{suffix}"
    prefixed_identifier = f"user_{keyword}"
    code = f"{keyword} {{ }} {suffixed_identifier}; {prefixed_identifier};"

    tokens = _without_eof(_tokens(code))

    assert (expected_type, keyword) in tokens
    assert ("IDENTIFIER", suffixed_identifier) in tokens
    assert ("IDENTIFIER", prefixed_identifier) in tokens
    assert (expected_type, suffixed_identifier) not in tokens
    assert (expected_type, prefixed_identifier) not in tokens


@settings(max_examples=25, deadline=None)
@given(
    operator_cases=st.lists(
        st.sampled_from(LONGEST_OPERATOR_CASES),
        min_size=1,
        max_size=12,
    )
)
def test_generated_operator_sequences_use_longest_token_matches(operator_cases):
    code = "\n".join(f"a {operator_text} b;" for _, operator_text in operator_cases)
    tokens = _without_eof(_tokens(code))
    operator_tokens = [
        token for token in tokens if token[0] not in {"IDENTIFIER", "SEMICOLON"}
    ]

    assert operator_tokens == operator_cases


@settings(max_examples=25, deadline=None)
@given(
    resource_cases=st.lists(
        st.sampled_from(RESOURCE_TOKEN_CASES),
        min_size=1,
        max_size=12,
        unique_by=lambda case: case[1],
    ),
    suffix=IDENTIFIER_SUFFIXES,
)
def test_generated_resource_type_spellings_are_single_semantic_tokens(
    resource_cases,
    suffix,
):
    code = "\n".join(
        f"{resource_text} resource_{index}_{suffix};"
        for index, (_, resource_text) in enumerate(resource_cases)
    )
    tokens = _without_eof(_tokens(code))

    for index, (expected_type, resource_text) in enumerate(resource_cases):
        assert (expected_type, resource_text) in tokens
        assert ("IDENTIFIER", resource_text) not in tokens
        assert ("IDENTIFIER", f"resource_{index}_{suffix}") in tokens
