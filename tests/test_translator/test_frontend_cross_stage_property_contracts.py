from dataclasses import dataclass

import pytest
from hypothesis import given, settings, strategies as st

from crosstl.translator.ast import ShaderStage
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)
INTERFACE_TYPES = ("float", "vec2", "vec3", "vec4")
MATCHING_INTERPOLATION_ALIASES = (
    ("flat", "nointerpolation"),
    ("smooth", "linear"),
    ("noperspective", "linear_noperspective"),
)


@dataclass(frozen=True)
class InterfaceTypeConflict:
    producer_type: str
    consumer_type: str


@dataclass(frozen=True)
class InterpolationConflict:
    producer_qualifier: str
    consumer_qualifier: str
    message: str


TYPE_CONFLICTS = tuple(
    InterfaceTypeConflict(producer_type, consumer_type)
    for producer_type in INTERFACE_TYPES
    for consumer_type in INTERFACE_TYPES
    if producer_type != consumer_type
)
INTERPOLATION_CONFLICTS = (
    InterpolationConflict(
        "flat",
        "noperspective",
        "Conflicting cross-stage interface interpolation mode",
    ),
    InterpolationConflict(
        "smooth",
        "flat",
        "Conflicting cross-stage interface interpolation mode",
    ),
    InterpolationConflict(
        "centroid",
        "sample",
        "Conflicting cross-stage interface interpolation sampling",
    ),
    InterpolationConflict(
        "linear_centroid",
        "linear_sample",
        "Conflicting cross-stage interface interpolation sampling",
    ),
)


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


def shader_with_struct_interface(
    suffix,
    producer_member,
    consumer_member,
):
    return f"""
    shader CrossStageProperty_{suffix} {{
        struct VSOut_{suffix} {{
            {producer_member}
        }};

        struct FSIn_{suffix} {{
            {consumer_member}
        }};

        vertex {{
            VSOut_{suffix} main() {{
                VSOut_{suffix} output;
                return output;
            }}
        }}

        fragment {{
            void main(FSIn_{suffix} input) {{
            }}
        }}
    }}
    """


def stage_member_source(
    qualifier,
    type_name,
    member_name,
    location,
    component,
    semantic_slot,
):
    return (
        f"layout(location = {location}, component = {component}) "
        f"{qualifier} {type_name} {member_name} : TEXCOORD{semantic_slot};"
    )


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    location=st.integers(min_value=0, max_value=31),
    component=st.integers(min_value=0, max_value=3),
    semantic_slot=st.integers(min_value=0, max_value=15),
    type_name=st.sampled_from(INTERFACE_TYPES),
    interpolation_aliases=st.sampled_from(MATCHING_INTERPOLATION_ALIASES),
)
def test_generated_cross_stage_struct_interfaces_accept_matching_alias_metadata(
    suffix,
    location,
    component,
    semantic_slot,
    type_name,
    interpolation_aliases,
):
    producer_qualifier, consumer_qualifier = interpolation_aliases
    code = shader_with_struct_interface(
        suffix,
        stage_member_source(
            producer_qualifier,
            type_name,
            f"producer_{suffix}",
            location,
            component,
            semantic_slot,
        ),
        stage_member_source(
            consumer_qualifier,
            type_name,
            f"consumer_{suffix}",
            location,
            component,
            semantic_slot,
        ),
    )

    ast = parse_code(code)

    assert ast.stages[ShaderStage.VERTEX].entry_point.return_type.name == (
        f"VSOut_{suffix}"
    )
    assert ast.stages[ShaderStage.FRAGMENT].entry_point.parameters[
        0
    ].param_type.name == (f"FSIn_{suffix}")


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    location=st.integers(min_value=0, max_value=31),
    component=st.integers(min_value=0, max_value=3),
    semantic_slot=st.integers(min_value=0, max_value=15),
    conflict=st.sampled_from(TYPE_CONFLICTS),
)
def test_generated_cross_stage_struct_interfaces_reject_type_mismatches(
    suffix,
    location,
    component,
    semantic_slot,
    conflict,
):
    code = shader_with_struct_interface(
        suffix,
        stage_member_source(
            "flat",
            conflict.producer_type,
            f"producer_{suffix}",
            location,
            component,
            semantic_slot,
        ),
        stage_member_source(
            "flat",
            conflict.consumer_type,
            f"consumer_{suffix}",
            location,
            component,
            semantic_slot,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Conflicting cross-stage interface type.*location",
    ):
        parse_code(code)


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    producer_location=st.integers(min_value=0, max_value=30),
    component=st.integers(min_value=0, max_value=3),
    semantic_slot=st.integers(min_value=0, max_value=15),
    type_name=st.sampled_from(INTERFACE_TYPES),
)
def test_generated_cross_stage_struct_interfaces_reject_location_mismatches(
    suffix,
    producer_location,
    component,
    semantic_slot,
    type_name,
):
    code = shader_with_struct_interface(
        suffix,
        stage_member_source(
            "flat",
            type_name,
            f"producer_{suffix}",
            producer_location,
            component,
            semantic_slot,
        ),
        stage_member_source(
            "flat",
            type_name,
            f"consumer_{suffix}",
            producer_location + 1,
            component,
            semantic_slot,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Conflicting cross-stage interface location metadata",
    ):
        parse_code(code)


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    location=st.integers(min_value=0, max_value=31),
    component=st.integers(min_value=0, max_value=3),
    semantic_slot=st.integers(min_value=0, max_value=15),
    type_name=st.sampled_from(INTERFACE_TYPES),
    conflict=st.sampled_from(INTERPOLATION_CONFLICTS),
)
def test_generated_cross_stage_struct_interfaces_reject_interpolation_mismatches(
    suffix,
    location,
    component,
    semantic_slot,
    type_name,
    conflict,
):
    code = shader_with_struct_interface(
        suffix,
        stage_member_source(
            conflict.producer_qualifier,
            type_name,
            f"producer_{suffix}",
            location,
            component,
            semantic_slot,
        ),
        stage_member_source(
            conflict.consumer_qualifier,
            type_name,
            f"consumer_{suffix}",
            location,
            component,
            semantic_slot,
        ),
    )

    with pytest.raises(ValueError, match=conflict.message):
        parse_code(code)
