from dataclasses import dataclass
from typing import Optional, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from crosstl.translator.ast import (
    ArrayAccessNode,
    BinaryOpNode,
    IdentifierNode,
    MemberAccessNode,
    UnaryOpNode,
)
from crosstl.translator.validation import (
    expression_debug_name,
    floating_coordinate_dimension,
    integer_coordinate_dimension,
    texture_bias_argument_index,
    texture_compare_argument_index,
    texture_gather_component_argument_index,
    texture_gradient_argument_indices,
    texture_intrinsic_allowed_argument_counts,
    texture_intrinsic_max_argument_count,
    texture_intrinsic_min_argument_count,
    texture_lod_argument_index,
    texture_mip_level_argument_index,
    texture_offset_argument_indices,
    texture_query_lod_coordinate_argument_index,
    texture_sample_index_argument_index,
)

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


@dataclass(frozen=True)
class ArgumentCountCase:
    name: str
    minimum: int
    maximum: int
    supports_explicit_sampler: bool
    allowed_counts: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class RoleIndexCase:
    helper_name: str
    function_name: str
    implicit_argument_count: int
    implicit_indices: Tuple[int, ...]
    supports_explicit_sampler: bool = True


@dataclass(frozen=True)
class SampleIndexCase:
    function_name: str
    argument_count: Optional[int]
    expected_index: Optional[int]


ARGUMENT_COUNT_CASES = (
    ArgumentCountCase("texture", 2, 3, True),
    ArgumentCountCase("textureGrad", 4, 4, True),
    ArgumentCountCase("textureCompareLod", 4, 4, True),
    ArgumentCountCase("textureQueryLod", 2, 2, True),
    ArgumentCountCase("textureGatherOffsets", 3, 7, True, (3, 4, 6, 7)),
    ArgumentCountCase("texelFetch", 3, 3, False),
    ArgumentCountCase("imageLoad", 2, 3, False),
    ArgumentCountCase("imageAtomicCompSwap", 4, 5, False),
)

ROLE_INDEX_CASES = (
    RoleIndexCase("offset", "textureOffset", 3, (2,)),
    RoleIndexCase("offset", "textureGatherOffsets", 7, (2, 3, 4, 5)),
    RoleIndexCase("gradient", "textureGrad", 4, (2, 3)),
    RoleIndexCase("gradient", "textureCompareGrad", 5, (3, 4)),
    RoleIndexCase("compare", "textureCompare", 3, (2,)),
    RoleIndexCase("compare", "textureGatherCompareOffset", 4, (2,)),
    RoleIndexCase("lod", "textureLod", 3, (2,)),
    RoleIndexCase("lod", "textureCompareLod", 4, (3,)),
    RoleIndexCase("bias", "texture", 3, (2,)),
    RoleIndexCase("query_lod_coord", "textureQueryLod", 2, (1,)),
    RoleIndexCase("mip_level", "textureSize", 2, (1,), False),
    RoleIndexCase("mip_level", "texelFetch", 3, (2,), False),
    RoleIndexCase("gather_component", "textureGather", 3, (2,)),
    RoleIndexCase("gather_component", "textureGatherOffset", 4, (3,)),
    RoleIndexCase("gather_component", "textureGatherOffsets", 7, (6,)),
)

SAMPLE_INDEX_CASES = (
    SampleIndexCase("texelFetch", None, 2),
    SampleIndexCase("texelFetch", 3, 2),
)

NO_SAMPLE_INDEX_CASES = (
    SampleIndexCase("texelFetch", 0, None),
    SampleIndexCase("texelFetch", 1, None),
    SampleIndexCase("texelFetch", 2, None),
    SampleIndexCase("texture", None, None),
    SampleIndexCase("texture", 3, None),
    SampleIndexCase("textureLod", 3, None),
    SampleIndexCase("textureSize", 2, None),
    SampleIndexCase("texelFetchOffset", 4, None),
    SampleIndexCase("imageLoad", 3, None),
    SampleIndexCase("textureSamplePosition", 2, None),
)

INTEGER_COORDINATE_TYPES = (
    ("int", 1),
    ("uint", 1),
    ("ivec2", 2),
    ("ivec3", 3),
    ("ivec4", 4),
    ("uvec2", 2),
    ("uvec3", 3),
    ("uvec4", 4),
    ("int2", 2),
    ("int3", 3),
    ("int4", 4),
    ("uint2", 2),
    ("uint3", 3),
    ("uint4", 4),
)

FLOATING_COORDINATE_TYPES = (
    ("float", 1),
    ("half", 1),
    ("double", 1),
    ("vec2", 2),
    ("vec3", 3),
    ("vec4", 4),
    ("dvec2", 2),
    ("dvec3", 3),
    ("dvec4", 4),
    ("float2", 2),
    ("float3", 3),
    ("float4", 4),
    ("half2", 2),
    ("half3", 3),
    ("half4", 4),
    ("double2", 2),
    ("double3", 3),
    ("double4", 4),
)


def role_indices(case, has_explicit_sampler, argument_count):
    if case.helper_name == "offset":
        return texture_offset_argument_indices(
            case.function_name,
            has_explicit_sampler=has_explicit_sampler,
            argument_count=argument_count,
        )
    if case.helper_name == "gradient":
        return texture_gradient_argument_indices(
            case.function_name,
            has_explicit_sampler=has_explicit_sampler,
            argument_count=argument_count,
        )
    if case.helper_name == "compare":
        index = texture_compare_argument_index(
            case.function_name,
            has_explicit_sampler=has_explicit_sampler,
            argument_count=argument_count,
        )
        return [] if index is None else [index]
    if case.helper_name == "lod":
        index = texture_lod_argument_index(
            case.function_name,
            has_explicit_sampler=has_explicit_sampler,
            argument_count=argument_count,
        )
        return [] if index is None else [index]
    if case.helper_name == "bias":
        index = texture_bias_argument_index(
            case.function_name,
            has_explicit_sampler=has_explicit_sampler,
            argument_count=argument_count,
        )
        return [] if index is None else [index]
    if case.helper_name == "query_lod_coord":
        index = texture_query_lod_coordinate_argument_index(
            case.function_name,
            has_explicit_sampler=has_explicit_sampler,
            argument_count=argument_count,
        )
        return [] if index is None else [index]
    if case.helper_name == "mip_level":
        index = texture_mip_level_argument_index(
            case.function_name,
            argument_count=argument_count,
        )
        return [] if index is None else [index]

    index = texture_gather_component_argument_index(
        case.function_name,
        has_explicit_sampler=has_explicit_sampler,
        argument_count=argument_count,
    )
    return [] if index is None else [index]


@settings(max_examples=40, deadline=None)
@given(case=st.sampled_from(ARGUMENT_COUNT_CASES))
def test_generated_texture_intrinsic_argument_count_helpers_shift_explicit_samplers(
    case,
):
    assert texture_intrinsic_min_argument_count(case.name) == case.minimum
    assert texture_intrinsic_max_argument_count(case.name) == case.maximum
    assert texture_intrinsic_allowed_argument_counts(case.name) == case.allowed_counts

    expected_shift = 1 if case.supports_explicit_sampler else 0
    assert texture_intrinsic_min_argument_count(
        case.name, has_explicit_sampler=True
    ) == (case.minimum + expected_shift)
    assert texture_intrinsic_max_argument_count(
        case.name, has_explicit_sampler=True
    ) == (case.maximum + expected_shift)

    allowed_counts = texture_intrinsic_allowed_argument_counts(
        case.name, has_explicit_sampler=True
    )
    if case.allowed_counts is None:
        assert allowed_counts is None
    else:
        assert allowed_counts == tuple(
            count + expected_shift for count in case.allowed_counts
        )


@settings(max_examples=50, deadline=None)
@given(case=st.sampled_from(ROLE_INDEX_CASES))
def test_generated_texture_intrinsic_role_indices_shift_with_explicit_samplers(case):
    assert role_indices(
        case,
        has_explicit_sampler=False,
        argument_count=case.implicit_argument_count,
    ) == list(case.implicit_indices)

    explicit_argument_count = case.implicit_argument_count + (
        1 if case.supports_explicit_sampler else 0
    )
    expected_indices = [
        index + (1 if case.supports_explicit_sampler else 0)
        for index in case.implicit_indices
    ]
    assert (
        role_indices(
            case,
            has_explicit_sampler=True,
            argument_count=explicit_argument_count,
        )
        == expected_indices
    )

    assert (
        role_indices(
            case,
            has_explicit_sampler=False,
            argument_count=min(case.implicit_indices),
        )
        == []
    )


@settings(max_examples=10, deadline=None)
@given(case=st.sampled_from(SAMPLE_INDEX_CASES))
def test_generated_texture_intrinsic_sample_index_metadata_marks_multisample_fetches(
    case,
):
    assert (
        texture_sample_index_argument_index(
            case.function_name, argument_count=case.argument_count
        )
        == case.expected_index
    )


@settings(max_examples=20, deadline=None)
@given(case=st.sampled_from(NO_SAMPLE_INDEX_CASES))
def test_generated_texture_intrinsic_sample_index_metadata_ignores_non_sample_calls(
    case,
):
    assert (
        texture_sample_index_argument_index(
            case.function_name, argument_count=case.argument_count
        )
        == case.expected_index
    )


@settings(max_examples=40, deadline=None)
@given(
    integer_case=st.sampled_from(INTEGER_COORDINATE_TYPES),
    floating_case=st.sampled_from(FLOATING_COORDINATE_TYPES),
    array_size=st.integers(min_value=1, max_value=64),
)
def test_generated_coordinate_dimension_helpers_strip_array_suffixes(
    integer_case,
    floating_case,
    array_size,
):
    integer_type, integer_dimension = integer_case
    floating_type, floating_dimension = floating_case

    assert integer_coordinate_dimension(integer_type) == integer_dimension
    assert integer_coordinate_dimension(f"{integer_type}[{array_size}]") == (
        integer_dimension
    )
    assert floating_coordinate_dimension(floating_type) == floating_dimension
    assert floating_coordinate_dimension(f"{floating_type}[{array_size}]") == (
        floating_dimension
    )


@settings(max_examples=25, deadline=None)
@given(suffix=IDENTIFIER_SUFFIXES)
def test_generated_expression_debug_names_are_source_like_for_nested_access(suffix):
    base = IdentifierNode(f"resources_{suffix}")
    index = BinaryOpNode(
        IdentifierNode(f"base_{suffix}"),
        "+",
        IdentifierNode(f"lane_{suffix}"),
    )
    access = MemberAccessNode(ArrayAccessNode(base, index), f"color_{suffix}")

    assert expression_debug_name(access) == (
        f"resources_{suffix}[base_{suffix} + lane_{suffix}].color_{suffix}"
    )
    assert expression_debug_name(
        UnaryOpNode("-", IdentifierNode(f"offset_{suffix}"))
    ) == (f"-offset_{suffix}")
    assert (
        expression_debug_name(
            UnaryOpNode("++", IdentifierNode(f"index_{suffix}"), is_postfix=True)
        )
        == f"index_{suffix}++"
    )
