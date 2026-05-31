from itertools import product

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)

ADDRESS_SPACE_GROUPS = (
    ("device", "global"),
    ("constant",),
    ("thread", "local", "private", "function"),
    ("threadgroup", "workgroup", "shared", "groupshared"),
    ("storage",),
)

ADDRESS_SPACE_CONFLICT_PAIRS = tuple(
    (left, right)
    for left_group, right_group in product(ADDRESS_SPACE_GROUPS, repeat=2)
    if left_group is not right_group
    for left in left_group
    for right in right_group
)

STORAGE_IMAGE_FORMATS = (
    "r32f",
    "r32i",
    "r32ui",
    "rgba16f",
    "rgba32f",
    "rgba32ui",
)


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    first=st.integers(min_value=0, max_value=63),
    second=st.integers(min_value=0, max_value=63),
)
def test_generated_conflicting_binding_metadata_is_rejected(
    suffix,
    first,
    second,
):
    assume(first != second)
    code = f"""
    shader ConflictingBinding_{suffix} {{
        sampler2D tex_{suffix} @binding({first}) @binding({second});
    }}
    """

    with pytest.raises(ValueError, match="Conflicting binding metadata"):
        parse_code(code)


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    first=st.sampled_from(STORAGE_IMAGE_FORMATS),
    second=st.sampled_from(STORAGE_IMAGE_FORMATS),
)
def test_generated_conflicting_storage_image_formats_are_rejected(
    suffix,
    first,
    second,
):
    assume(first != second)
    code = f"""
    shader ConflictingImageFormat_{suffix} {{
        image2D image_{suffix} @{first} @{second};
    }}
    """

    with pytest.raises(ValueError, match="Conflicting format metadata"):
        parse_code(code)


@settings(max_examples=20, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    image_format=st.sampled_from(STORAGE_IMAGE_FORMATS),
)
def test_generated_matching_storage_image_format_aliases_are_allowed(
    suffix,
    image_format,
):
    code = f"""
    shader MatchingImageFormat_{suffix} {{
        image2D image_{suffix} @{image_format} @format({image_format});
    }}
    """

    ast = parse_code(code)

    assert ast.global_variables[0].name == f"image_{suffix}"
    assert [attr.name for attr in ast.global_variables[0].attributes] == [
        image_format,
        "format",
    ]
    assert ast.global_variables[0].attributes[1].arguments[0].name == image_format


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    qualifiers=st.sampled_from(ADDRESS_SPACE_CONFLICT_PAIRS),
)
def test_generated_conflicting_parameter_address_spaces_are_rejected(
    suffix,
    qualifiers,
):
    first, second = qualifiers
    code = f"""
    shader ConflictingAddressSpace_{suffix} {{
        void consume_{suffix}({first} {second} float* values_{suffix}) {{
        }}
    }}
    """

    with pytest.raises(ValueError, match="Conflicting address space metadata"):
        parse_code(code)
