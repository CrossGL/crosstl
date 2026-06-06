from types import SimpleNamespace

import pytest

from crosstl.translator.ast import AttributeNode
from crosstl.translator.codegen.image_access_contracts import (
    INTEGER_COORDINATE_TYPE_NAMES,
    PROJECTED_TEXTURE_BASIC_INTRINSIC_NAMES,
    PROJECTED_TEXTURE_BASIC_OFFSET_INTRINSIC_NAMES,
    PROJECTED_TEXTURE_COMPARE_INTRINSIC_NAMES,
    PROJECTED_TEXTURE_GRAD_INTRINSIC_NAMES,
    PROJECTED_TEXTURE_GRAD_OFFSET_INTRINSIC_NAMES,
    PROJECTED_TEXTURE_INTRINSIC_NAMES,
    PROJECTED_TEXTURE_LOD_INTRINSIC_NAMES,
    PROJECTED_TEXTURE_LOD_OFFSET_INTRINSIC_NAMES,
    PROJECTED_TEXTURE_OFFSET_INTRINSIC_NAMES,
    RESOURCE_QUERY_SAMPLES_INTRINSIC_NAMES,
    RESOURCE_QUERY_SIZE_INTRINSIC_NAMES,
    TEXTURE_COMPARE_BASIC_INTRINSIC_NAMES,
    TEXTURE_COMPARE_GRAD_INTRINSIC_NAMES,
    TEXTURE_COMPARE_GRAD_OFFSET_INTRINSIC_NAMES,
    TEXTURE_COMPARE_INTRINSIC_NAMES,
    TEXTURE_COMPARE_LOD_INTRINSIC_NAMES,
    TEXTURE_COMPARE_LOD_OFFSET_INTRINSIC_NAMES,
    TEXTURE_COMPARE_NON_PROJECTED_INTRINSIC_NAMES,
    TEXTURE_COMPARE_NON_PROJECTED_OFFSET_OPERATION_NAMES,
    TEXTURE_COMPARE_OFFSET_INTRINSIC_NAMES,
    TEXTURE_COMPARE_OFFSET_OPERATION_NAMES,
    TEXTURE_GATHER_BASIC_INTRINSIC_NAMES,
    TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES,
    TEXTURE_GATHER_COMPARE_OFFSET_INTRINSIC_NAMES,
    TEXTURE_GATHER_INTRINSIC_NAMES,
    TEXTURE_GATHER_MULTI_OFFSET_INTRINSIC_NAMES,
    TEXTURE_GATHER_OFFSET_INTRINSIC_NAMES,
    TEXTURE_GATHER_SINGLE_OFFSET_INTRINSIC_NAMES,
    TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES,
    TEXTURE_OFFSET_INTRINSIC_NAMES,
    TEXTURE_QUERY_INTRINSIC_NAMES,
    TEXTURE_QUERY_LEVELS_INTRINSIC_NAMES,
    TEXTURE_QUERY_LOD_INTRINSIC_NAMES,
    TEXTURE_QUERY_SAMPLES_INTRINSIC_NAMES,
    TEXTURE_QUERY_SIZE_INTRINSIC_NAMES,
    TEXTURE_RESOURCE_INTRINSIC_NAMES,
    TEXTURE_SAMPLE_BASIC_INTRINSIC_NAMES,
    TEXTURE_SAMPLE_BASIC_OFFSET_INTRINSIC_NAMES,
    TEXTURE_SAMPLE_GRAD_INTRINSIC_NAMES,
    TEXTURE_SAMPLE_GRAD_OFFSET_INTRINSIC_NAMES,
    TEXTURE_SAMPLE_INTRINSIC_NAMES,
    TEXTURE_SAMPLE_LOD_INTRINSIC_NAMES,
    TEXTURE_SAMPLE_LOD_OFFSET_INTRINSIC_NAMES,
    TEXTURE_SAMPLE_OFFSET_INTRINSIC_NAMES,
    TEXTURE_SAMPLING_INTRINSIC_NAMES,
    TEXTURE_SAMPLING_OFFSET_INTRINSIC_NAMES,
    TEXTURE_TEXEL_FETCH_BASIC_INTRINSIC_NAMES,
    TEXTURE_TEXEL_FETCH_INTRINSIC_NAMES,
    TEXTURE_TEXEL_FETCH_OFFSET_INTRINSIC_NAMES,
    component_count_mismatch,
    component_kind_mismatch,
    component_shape_requirement,
    default_storage_image_channel_count,
    explicit_image_access,
    explicit_image_format,
    floating_coordinate_dimension_from_type_name,
    image_access_diagnostic_name,
    image_access_requirement_label,
    image_atomic_explicit_format_component_kind,
    image_atomic_format_allowed_names,
    image_atomic_format_error,
    image_atomic_format_requirement,
    image_atomic_helper_descriptor_fields,
    image_atomic_helper_resource_metadata,
    image_atomic_resource_type_error,
    image_atomic_result_kind_error,
    image_atomic_result_kind_mismatch,
    image_atomic_storage_component_kind,
    image_atomic_value_arguments,
    image_atomic_value_kind_error,
    image_atomic_value_kind_mismatch,
    image_format_channel_count,
    image_format_component_kind,
    image_format_component_type,
    image_format_or_default_channel_count,
    image_format_result_type,
    image_format_vector_type,
    image_load_result_kind_error,
    image_load_result_kind_mismatch,
    image_load_result_shape_error,
    image_load_result_shape_mismatch,
    image_multisample_sample_argument_index,
    image_multisample_sample_type_error,
    image_multisample_sample_type_mismatch,
    image_resource_metadata,
    image_store_value_kind_error,
    image_store_value_kind_mismatch,
    image_store_value_shape_error,
    image_store_value_shape_mismatch,
    integer_coordinate_dimension_from_type_name,
    is_floating_scalar_type_name,
    is_glsl_float_image_resource,
    is_glsl_integer_image_type,
    is_glsl_storage_image_type,
    is_image_atomic_operation,
    is_image_format_attribute,
    is_image_resource_operation,
    is_integer_coordinate_type_name,
    is_integer_scalar_type_name,
    is_metal_float_image_resource,
    is_metal_integer_image_type,
    is_metal_storage_image_resource,
    is_numeric_scalar_type_name,
    is_projected_texture_basic_offset_operation,
    is_projected_texture_basic_operation,
    is_projected_texture_compare_operation,
    is_projected_texture_grad_offset_operation,
    is_projected_texture_grad_operation,
    is_projected_texture_lod_offset_operation,
    is_projected_texture_lod_operation,
    is_projected_texture_operation,
    is_resource_access_attribute,
    is_resource_samples_query_operation,
    is_resource_size_query_operation,
    is_scalar_image_format,
    is_storage_image_texture_comparison_operation,
    is_storage_image_texture_operation,
    is_texel_fetch_basic_operation,
    is_texel_fetch_offset_operation,
    is_texel_fetch_operation,
    is_texture_compare_any_offset_operation,
    is_texture_compare_basic_operation,
    is_texture_compare_grad_offset_operation,
    is_texture_compare_grad_operation,
    is_texture_compare_lod_offset_operation,
    is_texture_compare_lod_operation,
    is_texture_compare_non_projected_offset_operation,
    is_texture_compare_non_projected_operation,
    is_texture_compare_offset_operation,
    is_texture_compare_operation,
    is_texture_gather_basic_operation,
    is_texture_gather_compare_offset_operation,
    is_texture_gather_compare_operation,
    is_texture_gather_multi_offset_operation,
    is_texture_gather_offset_operation,
    is_texture_gather_operation,
    is_texture_gather_single_offset_operation,
    is_texture_implicit_sampler_operation,
    is_texture_offset_operation,
    is_texture_query_levels_operation,
    is_texture_query_lod_operation,
    is_texture_query_operation,
    is_texture_resource_operation,
    is_texture_sample_basic_offset_operation,
    is_texture_sample_basic_operation,
    is_texture_sample_grad_offset_operation,
    is_texture_sample_grad_operation,
    is_texture_sample_lod_offset_operation,
    is_texture_sample_lod_operation,
    is_texture_sample_offset_operation,
    is_texture_sample_operation,
    is_texture_samples_query_operation,
    is_texture_sampling_offset_operation,
    is_texture_sampling_operation,
    is_texture_size_query_operation,
    is_two_component_image_format,
    literal_numeric_component_count,
    literal_numeric_component_kind,
    metal_storage_image_access_agnostic_type,
    metal_storage_image_component_type,
    normalized_image_access,
    numeric_component_count_from_type,
    numeric_component_kind_from_type,
    numeric_expression_component_count,
    numeric_expression_component_kind,
    numeric_scalar_expression_kind,
    numeric_scalar_type_kind,
    numeric_type_component_count,
    numeric_type_component_kind,
    operation_argument_type_error,
    operation_dimension_argument_error,
    projected_texture_extra_argument_count_error,
    projected_texture_offset_capability_error,
    record_explicit_image_metadata,
    requires_integer_coordinate,
    resolve_image_atomic_component_kind,
    resource_query_get_dimensions_descriptor,
    resource_query_method_size_descriptor,
    resource_query_scalar_constant_helper_descriptor,
    resource_query_scalar_helper_descriptor,
    resource_query_size_components_descriptor,
    resource_query_size_helper_descriptor,
    should_validate_image_atomic_component_kind,
    should_validate_image_load_result_shape,
    storage_image_atomic_zero_value,
    storage_image_format_store_constructor,
    storage_image_load_component_suffix,
    storage_image_store_constructors,
    storage_image_store_value_expression,
    storage_image_store_vector_constructor,
    storage_image_two_component_store_expression,
    storage_image_zero_values,
    supported_image_formats,
    texel_fetch_offset_multisample_reason,
    texel_fetch_zero_value,
    texture_argument_diagnostic_type,
    texture_compare_argument_error,
    texture_compare_coordinate_error,
    texture_compare_extra_argument_count_error,
    texture_compare_offset_capability_error,
    texture_compare_projected_coordinate_error,
    texture_compare_projected_lod_array_error,
    texture_coordinate_arguments_error,
    texture_gather_capability_error,
    texture_gather_compare_extra_argument_count_error,
    texture_gather_component_count_error,
    texture_gather_component_literal_error,
    texture_gather_offset_argument_count_error,
    texture_gather_offset_capability_error,
    texture_gather_offsets_argument_count_error,
    texture_gather_operation_error,
    texture_image_resource_operation_names,
    texture_multisample_sample_type_error,
    texture_query_levels_multisample_expression,
    texture_query_levels_zero_value,
    texture_query_lod_coordinate_dimension_error,
    texture_query_lod_coordinate_swizzle,
    texture_query_lod_coordinate_type_error,
    texture_query_lod_zero_value,
    texture_resource_dimension_descriptor,
    texture_resource_offset_dimension_key,
    texture_sample_offset_capability_error,
    texture_sample_offset_extra_argument_count_error,
    texture_samples_query_expression,
    texture_samples_query_requirement_name,
    texture_scalar_zero_value,
    texture_vector_zero_value,
    unsupported_cube_texel_fetch_expression,
    unsupported_image_atomic_expression,
    unsupported_multisample_image_atomic_expression,
    unsupported_multisample_image_store_expression,
    unsupported_multisample_texel_fetch_offset_expression,
    unsupported_multisample_texture_call_expression,
    unsupported_multisample_texture_call_vector_expression,
    unsupported_multisample_texture_compare_expression,
    unsupported_multisample_texture_compare_scalar_expression,
    unsupported_multisample_texture_gather_compare_expression,
    unsupported_multisample_texture_gather_compare_vector_expression,
    unsupported_multisample_texture_query_expression,
    unsupported_multisample_texture_query_lod_expression,
    unsupported_projected_texture_call_expression,
    unsupported_projected_texture_expression,
    unsupported_projected_texture_operation_error,
    unsupported_storage_image_texture_comparison_expression,
    unsupported_storage_image_texture_comparison_scalar_expression,
    unsupported_storage_image_texture_operation_expression,
    unsupported_storage_image_texture_operation_vector_expression,
    unsupported_texel_fetch_expression,
    unsupported_texel_fetch_offset_expression,
    unsupported_texture_compare_expression,
    unsupported_texture_compare_operation_error,
    unsupported_texture_compare_scalar_expression,
    unsupported_texture_gather_call_expression,
    unsupported_texture_gather_compare_call_expression,
    unsupported_texture_gather_compare_expression,
    unsupported_texture_gather_expression,
    unsupported_texture_offset_call_expression,
    unsupported_texture_offset_expression,
    unsupported_texture_offset_operation_error,
    unsupported_texture_query_expression,
    unsupported_texture_query_levels_expression,
    unsupported_texture_query_lod_expression,
    unsupported_texture_samples_query_call_expression,
    unsupported_texture_samples_query_expression,
    validate_texture_operation_arity,
)
from crosstl.translator.codegen.resource_query import ResourceQueryMixin


def attribute_value_to_string(value):
    return getattr(value, "value", value)


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("read", "read"),
        ("readonly", "read"),
        ("read-only", "read"),
        ("read_only", "read"),
        ("write", "write"),
        ("writeonly", "write"),
        ("access::write", "write"),
        ("readwrite", "read_write"),
        ("read_write", "read_write"),
        ("access::read_write", "read_write"),
        ("semantic", None),
    ],
)
def test_normalized_image_access_accepts_common_spellings(spelling, expected):
    assert normalized_image_access(spelling) == expected


def test_image_access_diagnostic_labels_match_backend_conventions():
    assert image_access_requirement_label("read") == "read-capable"
    assert image_access_requirement_label("write") == "write-capable"
    assert image_access_requirement_label("read_write") == "read-write"
    assert (
        image_access_requirement_label("read_write", read_write_label="read_write")
        == "read_write"
    )
    assert image_access_requirement_label("custom") == "custom"

    assert image_access_diagnostic_name("read") == "readonly"
    assert image_access_diagnostic_name("write") == "writeonly"
    assert image_access_diagnostic_name("read_write") == "readwrite"
    assert (
        image_access_diagnostic_name("read_write", read_write_label="read_write")
        == "read_write"
    )
    assert image_access_diagnostic_name("custom") == "custom"


def test_operation_argument_type_error_builds_backend_diagnostic():
    assert (
        operation_argument_type_error(
            "DirectX",
            "texture compare",
            "textureCompare",
            "a scalar floating",
            "compare",
            "depthRef",
            "int",
        )
        == "DirectX texture compare operation 'textureCompare' requires a "
        "scalar floating compare argument: depthRef has type int"
    )
    assert (
        operation_argument_type_error(
            "OpenGL",
            "resource",
            "imageLoad",
            "an integer",
            "coordinate",
            "uv",
            "vec2",
        )
        == "OpenGL resource operation 'imageLoad' requires an integer "
        "coordinate argument: uv has type vec2"
    )


def test_operation_dimension_argument_error_builds_backend_diagnostic():
    assert (
        operation_dimension_argument_error(
            "Metal",
            "resource",
            "textureGrad",
            2,
            "floating",
            "gradient",
            "texture2d<float>",
            "ddx",
            "float3",
        )
        == "Metal resource operation 'textureGrad' requires a 2D floating "
        "gradient for texture2d<float>: ddx has type float3"
    )


def test_texture_query_lod_coordinate_errors_match_shared_diagnostics():
    assert (
        texture_query_lod_coordinate_type_error(
            "OpenGL", "textureQueryLod", "pixel", "ivec2"
        )
        == "OpenGL texture query operation 'textureQueryLod' requires a "
        "floating coordinate argument: pixel has type ivec2"
    )
    assert (
        texture_query_lod_coordinate_dimension_error(
            "DirectX",
            "textureQueryLod",
            3,
            "TextureCube",
            "uv",
            "float2",
        )
        == "DirectX texture query operation 'textureQueryLod' requires a "
        "3D floating coordinate for TextureCube: uv has type float2"
    )


def test_texture_argument_diagnostic_type_prefers_resource_then_sampler_then_result():
    resource_types = {"textureArg": "Texture2D"}
    expression_names = {
        "textureArg": "textureArg",
        "samplerArg": "samplerArg",
        "valueArg": "valueArg",
    }
    result_types = {"samplerArg": "SamplerState", "valueArg": "float2"}

    assert (
        texture_argument_diagnostic_type(
            "textureArg",
            resource_types.get,
            expression_names.get,
            result_types.get,
            {"samplerArg"},
        )
        == "Texture2D"
    )
    assert (
        texture_argument_diagnostic_type(
            "samplerArg",
            resource_types.get,
            expression_names.get,
            result_types.get,
            {"samplerArg"},
        )
        == "sampler"
    )
    assert (
        texture_argument_diagnostic_type(
            "valueArg",
            resource_types.get,
            expression_names.get,
            result_types.get,
            {"samplerArg"},
        )
        == "float2"
    )


def test_validate_texture_operation_arity_ignores_non_resource_operations():
    def unexpected_explicit_sampler_check(args):
        raise AssertionError("sampler mode should not be checked")

    validate_texture_operation_arity(
        "OpenGL",
        "helper",
        [],
        {"texture"},
        unexpected_explicit_sampler_check,
    )


def test_validate_texture_operation_arity_rejects_too_few_arguments():
    with pytest.raises(
        ValueError,
        match=(
            "DirectX texture operation 'textureLod' requires at least 3 "
            "argument\\(s\\), got 2"
        ),
    ):
        validate_texture_operation_arity(
            "DirectX",
            "textureLod",
            ["tex", "uv"],
            {"textureLod"},
            lambda args: False,
        )


def test_validate_texture_operation_arity_rejects_disallowed_counts():
    with pytest.raises(
        ValueError,
        match=(
            "Metal texture operation 'textureGatherOffsets' accepts "
            "3, 4, 6, 7 argument\\(s\\), got 5"
        ),
    ):
        validate_texture_operation_arity(
            "Metal",
            "textureGatherOffsets",
            ["tex", "uv", "o0", "o1", "o2"],
            {"textureGatherOffsets"},
            lambda args: False,
        )


def test_validate_texture_operation_arity_rejects_too_many_arguments():
    with pytest.raises(
        ValueError,
        match=(
            "OpenGL texture operation 'texture' accepts at most 3 "
            "argument\\(s\\), got 4"
        ),
    ):
        validate_texture_operation_arity(
            "OpenGL",
            "texture",
            ["tex", "uv", "bias", "extra"],
            {"texture"},
            lambda args: False,
        )


def test_texture_compare_extra_argument_count_error_matches_operation_shapes():
    assert texture_compare_extra_argument_count_error("textureCompare", 1) is None
    assert (
        texture_compare_extra_argument_count_error("textureCompare", 2)
        == "accepts no extra arguments"
    )
    assert texture_compare_extra_argument_count_error("textureCompareOffset", 2) is None
    assert (
        texture_compare_extra_argument_count_error("textureCompareOffset", 1)
        == "requires compare and offset arguments"
    )
    assert texture_compare_extra_argument_count_error("textureCompareLod", 2) is None
    assert (
        texture_compare_extra_argument_count_error("textureCompareLod", 1)
        == "requires compare and lod arguments"
    )
    assert (
        texture_compare_extra_argument_count_error("textureCompareLodOffset", 3) is None
    )
    assert (
        texture_compare_extra_argument_count_error("textureCompareLodOffset", 2)
        == "requires compare, lod, and offset arguments"
    )
    assert texture_compare_extra_argument_count_error("textureCompareGrad", 3) is None
    assert (
        texture_compare_extra_argument_count_error("textureCompareGrad", 2)
        == "requires compare, gradient x, and gradient y arguments"
    )
    assert (
        texture_compare_extra_argument_count_error("textureCompareGradOffset", 4)
        is None
    )
    assert (
        texture_compare_extra_argument_count_error("textureCompareGradOffset", 3)
        == "requires compare, gradient x, gradient y, and offset arguments"
    )
    assert texture_compare_extra_argument_count_error("texture", 2) is None


def test_texture_gather_compare_extra_argument_count_error_matches_shapes():
    assert (
        texture_gather_compare_extra_argument_count_error("textureGatherCompare", 1)
        is None
    )
    assert (
        texture_gather_compare_extra_argument_count_error("textureGatherCompare", 2)
        == "accepts no extra arguments"
    )
    assert (
        texture_gather_compare_extra_argument_count_error(
            "textureGatherCompareOffset", 2
        )
        is None
    )
    assert (
        texture_gather_compare_extra_argument_count_error(
            "textureGatherCompareOffset", 1
        )
        == "requires compare and offset arguments"
    )
    assert texture_gather_compare_extra_argument_count_error("textureGather", 1) is None


def test_texture_compare_projected_coordinate_error_matches_backend_terms():
    assert (
        texture_compare_projected_coordinate_error("GLSL")
        == "requires sampler2DShadow vec3/vec4 or sampler2DArrayShadow "
        "vec4 projection coordinates"
    )
    assert (
        texture_compare_projected_coordinate_error("DirectX")
        == "requires Texture2D vec3/vec4 or Texture2DArray "
        "vec4 projection coordinates"
    )
    assert (
        texture_compare_projected_coordinate_error("Metal")
        == "requires depth2d vec3/vec4 or depth2d_array "
        "vec4 projection coordinates"
    )


def test_texture_compare_coordinate_error_matches_shared_reason():
    assert (
        texture_compare_coordinate_error()
        == "requires supported shadow texture coordinates"
    )


def test_texture_argument_presence_errors_match_shared_reasons():
    assert (
        texture_coordinate_arguments_error()
        == "requires texture and coordinate arguments"
    )
    assert texture_compare_argument_error() == "requires a compare argument"


def test_texture_gather_diagnostic_errors_match_shared_reasons():
    assert (
        texture_gather_capability_error()
        == "requires 2D, 2D-array, cube, or cube-array textures"
    )
    assert (
        texture_gather_offset_capability_error()
        == "offsets require 2D or 2D-array textures"
    )
    assert (
        texture_gather_component_count_error()
        == "accepts at most one component argument"
    )
    assert (
        texture_gather_offset_argument_count_error()
        == "requires offset and optional component arguments"
    )
    assert (
        texture_gather_offsets_argument_count_error()
        == "requires a typed offsets array or four offset arguments"
    )
    assert texture_gather_operation_error() == "requires a gather operation"
    assert (
        texture_gather_component_literal_error()
        == "component literal must be 0, 1, 2, or 3"
    )


def test_texture_compare_projected_lod_array_error_matches_shared_reason():
    assert (
        texture_compare_projected_lod_array_error()
        == "projected explicit LOD is not supported for sampler2DArrayShadow"
    )


def test_unsupported_texture_compare_operation_error_matches_projection_mode():
    assert (
        unsupported_texture_compare_operation_error(projected=True)
        == "is not a supported projected shadow compare operation"
    )
    assert (
        unsupported_texture_compare_operation_error()
        == "is not a supported shadow compare operation"
    )


def test_unsupported_texture_offset_operation_errors_match_shared_reasons():
    assert (
        unsupported_texture_offset_operation_error()
        == "is not a supported texture offset operation"
    )
    assert (
        unsupported_projected_texture_operation_error()
        == "unsupported projected texture operation"
    )


def test_texture_compare_offset_capability_error_matches_backend_terms():
    assert (
        texture_compare_offset_capability_error("GLSL")
        == "offsets require 2D or 2D-array shadow samplers"
    )
    assert (
        texture_compare_offset_capability_error("DirectX")
        == "offsets require 2D or 2D-array textures"
    )
    assert (
        texture_compare_offset_capability_error("Metal")
        == "offsets require 2D or 2D-array depth textures"
    )


def test_texture_sample_offset_capability_error_matches_backend_terms():
    assert (
        texture_sample_offset_capability_error("GLSL")
        == "offsets require 1D, 2D, 2D-array, 3D, "
        "or planar shadow samplers"
    )
    assert (
        texture_sample_offset_capability_error("DirectX")
        == "offsets require 1D, 2D, 2D-array, or 3D textures"
    )
    assert (
        texture_sample_offset_capability_error("Metal")
        == "offsets require 2D, 2D-array, or 3D textures"
    )


def test_projected_texture_offset_capability_error_matches_shared_reason():
    assert (
        projected_texture_offset_capability_error()
        == "offsets require 2D, 2D-array, or 3D textures"
    )


def test_texture_sample_offset_extra_argument_count_error_matches_shapes():
    assert texture_sample_offset_extra_argument_count_error("textureOffset", 1) is None
    assert texture_sample_offset_extra_argument_count_error("textureOffset", 2) is None
    assert (
        texture_sample_offset_extra_argument_count_error("textureOffset", 0)
        == "requires offset and optional bias arguments"
    )
    assert (
        texture_sample_offset_extra_argument_count_error("textureLodOffset", 2) is None
    )
    assert (
        texture_sample_offset_extra_argument_count_error("textureLodOffset", 1)
        == "requires lod and offset arguments"
    )
    assert (
        texture_sample_offset_extra_argument_count_error("textureGradOffset", 3) is None
    )
    assert (
        texture_sample_offset_extra_argument_count_error("textureGradOffset", 2)
        == "requires gradient x, gradient y, and offset arguments"
    )
    assert texture_sample_offset_extra_argument_count_error("texture", 2) is None


def test_projected_texture_extra_argument_count_error_matches_shapes():
    assert projected_texture_extra_argument_count_error("textureProj", 0) is None
    assert projected_texture_extra_argument_count_error("textureProj", 1) is None
    assert (
        projected_texture_extra_argument_count_error("textureProj", 2)
        == "accepts at most one bias argument"
    )
    assert projected_texture_extra_argument_count_error("textureProjOffset", 1) is None
    assert projected_texture_extra_argument_count_error("textureProjOffset", 2) is None
    assert (
        projected_texture_extra_argument_count_error("textureProjOffset", 0)
        == "requires offset and optional bias arguments"
    )
    assert projected_texture_extra_argument_count_error("textureProjLod", 1) is None
    assert (
        projected_texture_extra_argument_count_error("textureProjLod", 0)
        == "requires one lod argument"
    )
    assert (
        projected_texture_extra_argument_count_error("textureProjLodOffset", 2) is None
    )
    assert (
        projected_texture_extra_argument_count_error("textureProjLodOffset", 1)
        == "requires lod and offset arguments"
    )
    assert projected_texture_extra_argument_count_error("textureProjGrad", 2) is None
    assert (
        projected_texture_extra_argument_count_error("textureProjGrad", 1)
        == "requires gradient x and gradient y arguments"
    )
    assert (
        projected_texture_extra_argument_count_error("textureProjGradOffset", 3) is None
    )
    assert (
        projected_texture_extra_argument_count_error("textureProjGradOffset", 2)
        == "requires gradient x, gradient y, and offset arguments"
    )
    assert projected_texture_extra_argument_count_error("texture", 2) is None


def test_requires_integer_coordinate_matches_intrinsic_contract_set():
    integer_coordinate_intrinsics = {"texelFetch", "imageLoad", "imageStore"}

    assert requires_integer_coordinate("texelFetch", integer_coordinate_intrinsics)
    assert requires_integer_coordinate("imageLoad", integer_coordinate_intrinsics)
    assert not requires_integer_coordinate("texture", integer_coordinate_intrinsics)
    assert not requires_integer_coordinate("textureSize", integer_coordinate_intrinsics)


def test_integer_coordinate_type_names_match_source_and_backend_spellings():
    assert INTEGER_COORDINATE_TYPE_NAMES == frozenset(
        {
            "int",
            "uint",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "int2",
            "int3",
            "int4",
            "uint2",
            "uint3",
            "uint4",
        }
    )

    assert is_integer_coordinate_type_name("int")
    assert is_integer_coordinate_type_name("uint")
    assert is_integer_coordinate_type_name("ivec2")
    assert is_integer_coordinate_type_name("uvec4")
    assert is_integer_coordinate_type_name("int2")
    assert is_integer_coordinate_type_name("uint4")
    assert is_integer_coordinate_type_name("vec2", "int2")
    assert is_integer_coordinate_type_name("float2", "uint3")
    assert not is_integer_coordinate_type_name("float")
    assert not is_integer_coordinate_type_name("vec2", "float2")


def test_scalar_type_name_helpers_accept_source_and_backend_spellings():
    def map_type(type_name):
        return {
            "floatAlias": "float",
            "numericAlias": "double",
            "uintAlias": "uint",
        }.get(type_name, type_name)

    assert is_floating_scalar_type_name("float", map_type)
    assert is_floating_scalar_type_name("half", map_type)
    assert is_floating_scalar_type_name("floatAlias", map_type)
    assert not is_floating_scalar_type_name("int", map_type)

    assert is_numeric_scalar_type_name("double", map_type)
    assert is_numeric_scalar_type_name("numericAlias", map_type)
    assert is_numeric_scalar_type_name("uint", map_type)
    assert not is_numeric_scalar_type_name("bool", map_type)

    assert is_integer_scalar_type_name("int", map_type)
    assert is_integer_scalar_type_name("uintAlias", map_type)
    assert not is_integer_scalar_type_name("float", map_type)


def test_scalar_type_name_helpers_reject_unknown_and_array_spellings():
    def unexpected_map_type(type_name):
        raise AssertionError(f"should not map {type_name}")

    assert not is_floating_scalar_type_name(None, unexpected_map_type)
    assert not is_numeric_scalar_type_name("", unexpected_map_type)
    assert not is_integer_scalar_type_name("int[4]", unexpected_map_type)


def test_floating_coordinate_dimension_from_type_name_uses_source_and_mapped_names():
    def map_type(type_name):
        return {
            "float2Alias": "float2",
            "vec3Alias": "vec3",
        }.get(type_name, type_name)

    assert floating_coordinate_dimension_from_type_name("float", map_type) == 1
    assert floating_coordinate_dimension_from_type_name("vec2", map_type) == 2
    assert floating_coordinate_dimension_from_type_name("float3", map_type) == 3
    assert floating_coordinate_dimension_from_type_name("float2Alias", map_type) == 2
    assert floating_coordinate_dimension_from_type_name("vec3Alias", map_type) == 3
    assert floating_coordinate_dimension_from_type_name("int2", map_type) is None
    assert floating_coordinate_dimension_from_type_name(None, map_type) is None


def test_integer_coordinate_dimension_from_type_name_uses_source_and_mapped_names():
    def map_type(type_name):
        return {
            "ivec2Alias": "ivec2",
            "uint3Alias": "uint3",
        }.get(type_name, type_name)

    assert integer_coordinate_dimension_from_type_name("int", map_type) == 1
    assert integer_coordinate_dimension_from_type_name("ivec2", map_type) == 2
    assert integer_coordinate_dimension_from_type_name("uint4", map_type) == 4
    assert integer_coordinate_dimension_from_type_name("ivec2Alias", map_type) == 2
    assert integer_coordinate_dimension_from_type_name("uint3Alias", map_type) == 3
    assert integer_coordinate_dimension_from_type_name("vec2", map_type) is None
    assert integer_coordinate_dimension_from_type_name(None, map_type) is None


def test_texture_resource_dimension_descriptor_builds_canonical_shape():
    descriptor = texture_resource_dimension_descriptor(
        "Texture2DArray",
        {
            "sample_offset": True,
            "gather_offset": True,
            "compare_offset": True,
            "gather_compare_offset": True,
        },
        coordinate_dimension=3,
        offset_dimension=2,
        gradient_dimension=2,
        query_lod_coordinate_dimension=3,
    )

    assert descriptor == {
        "texture_type": "Texture2DArray",
        "coordinate_dimension": 3,
        "offset_dimension": 2,
        "sample_offset_dimension": 2,
        "texel_fetch_offset_dimension": 2,
        "gather_offset_dimension": 2,
        "compare_offset_dimension": 2,
        "compare_lod_offset_dimension": 2,
        "compare_grad_offset_dimension": 2,
        "gather_compare_offset_dimension": 2,
        "gradient_dimension": 2,
        "query_lod_coordinate_dimension": 3,
    }


def test_texture_resource_dimension_descriptor_honors_explicit_capability_keys():
    descriptor = texture_resource_dimension_descriptor(
        "sampler2DArrayShadow",
        {
            "sample_offset": True,
            "gather_offset": False,
            "compare_offset": True,
            "compare_lod_offset": False,
            "compare_grad_offset": True,
            "gather_compare_offset": True,
        },
        coordinate_dimension=3,
        offset_dimension=2,
    )

    assert descriptor["sample_offset_dimension"] == 2
    assert descriptor["gather_offset_dimension"] is None
    assert descriptor["compare_offset_dimension"] == 2
    assert descriptor["compare_lod_offset_dimension"] is None
    assert descriptor["compare_grad_offset_dimension"] == 2
    assert descriptor["gather_compare_offset_dimension"] == 2


def test_texture_resource_dimension_descriptor_honors_multisample_texel_fetch():
    descriptor = texture_resource_dimension_descriptor(
        "Texture2DMS",
        {"sample_offset": True},
        coordinate_dimension=2,
        offset_dimension=2,
        is_multisample=True,
    )

    assert descriptor["sample_offset_dimension"] == 2
    assert descriptor["texel_fetch_offset_dimension"] is None


def test_explicit_image_access_prefers_qualifier_before_attribute():
    node = SimpleNamespace(
        qualifiers=["writeonly"],
        attributes=[AttributeNode("access", [SimpleNamespace(value="read")])],
    )

    assert explicit_image_access(node, attribute_value_to_string) == "write"


def test_explicit_image_access_reads_access_attribute_argument():
    node = SimpleNamespace(
        qualifiers=[],
        attributes=[AttributeNode("access", [SimpleNamespace(value="access::read")])],
    )

    assert explicit_image_access(node, attribute_value_to_string) == "read"


def test_explicit_image_access_reads_named_access_attribute():
    node = SimpleNamespace(qualifiers=[], attributes=[AttributeNode("readwrite")])

    assert explicit_image_access(node, attribute_value_to_string) == "read_write"


def test_is_resource_access_attribute_only_matches_access_metadata():
    assert is_resource_access_attribute(AttributeNode("access"))
    assert is_resource_access_attribute(AttributeNode("readonly"))
    assert not is_resource_access_attribute(AttributeNode("position"))


def test_supported_image_formats_contains_common_typed_storage_formats():
    formats = supported_image_formats()

    assert "rgba32f" in formats
    assert "r32i" in formats
    assert "r32ui" in formats


def test_explicit_image_format_reads_named_format_attribute():
    node = SimpleNamespace(qualifiers=[], attributes=[AttributeNode("r32ui")])

    assert explicit_image_format(node, attribute_value_to_string) == "r32ui"


def test_explicit_image_format_reads_format_attribute_argument():
    node = SimpleNamespace(
        qualifiers=[],
        attributes=[AttributeNode("format", [SimpleNamespace(value="rgba32f")])],
    )

    assert explicit_image_format(node, attribute_value_to_string) == "rgba32f"


def test_explicit_image_format_ignores_unsupported_formats():
    node = SimpleNamespace(
        qualifiers=[],
        attributes=[AttributeNode("format", [SimpleNamespace(value="rgba64f")])],
    )

    assert explicit_image_format(node, attribute_value_to_string) is None


def test_record_explicit_image_metadata_writes_selected_maps():
    node = SimpleNamespace(
        qualifiers=["readonly"],
        attributes=[AttributeNode("format", [SimpleNamespace(value="rg32f")])],
    )
    formats = {}
    accesses = {}

    assert record_explicit_image_metadata(
        "img",
        node,
        attribute_value_to_string,
        image_formats=formats,
        image_accesses=accesses,
    ) == ("rg32f", "read")
    assert formats == {"img": "rg32f"}
    assert accesses == {"img": "read"}


def test_record_explicit_image_metadata_allows_format_only_maps():
    node = SimpleNamespace(
        qualifiers=["writeonly"],
        attributes=[AttributeNode("rgba16f")],
    )
    formats = {}

    assert record_explicit_image_metadata(
        "img",
        node,
        attribute_value_to_string,
        image_formats=formats,
    ) == ("rgba16f", "write")
    assert formats == {"img": "rgba16f"}


def test_image_resource_metadata_prefers_parameters_over_globals():
    def expression_name(value):
        return getattr(value, "name", None)

    assert (
        image_resource_metadata(
            SimpleNamespace(name="imageParam"),
            expression_name,
            {"imageParam": "r32ui"},
            {"imageParam": "rgba32f"},
        )
        == "r32ui"
    )
    assert (
        image_resource_metadata(
            SimpleNamespace(name="globalImage"),
            expression_name,
            {},
            {"globalImage": "rgba16f"},
        )
        == "rgba16f"
    )
    assert image_resource_metadata(object(), expression_name, {}, {}) is None


def test_is_image_format_attribute_only_matches_format_metadata():
    assert is_image_format_attribute(AttributeNode("format"))
    assert is_image_format_attribute(AttributeNode("rgba16f"))
    assert not is_image_format_attribute(AttributeNode("position"))


@pytest.mark.parametrize(
    ("image_format", "channels", "kind"),
    [
        ("r32f", 1, "float"),
        ("rg16i", 2, "int"),
        ("rgba8ui", 4, "uint"),
    ],
)
def test_image_format_metadata_reports_channels_and_component_kind(
    image_format, channels, kind
):
    assert image_format_channel_count(image_format) == channels
    assert image_format_component_kind(image_format) == kind


def test_numeric_type_component_kind_uses_scalar_or_vector_kind():
    assert numeric_type_component_kind("float") == "float"
    assert numeric_type_component_kind("float2", "float") == "float"
    assert numeric_type_component_kind("vec2", "float") == "float"
    assert numeric_type_component_kind("double", "double") is None
    assert numeric_type_component_kind("bool2", "bool") is None
    assert numeric_type_component_kind("Texture2D") is None


def test_numeric_type_component_count_handles_scalars_vectors_and_exclusions():
    assert numeric_type_component_count("float", scalar_types={"float", "double"}) == 1
    assert numeric_type_component_count("double", scalar_types={"float", "double"}) == 1
    assert numeric_type_component_count("float2", "float") == 2
    assert numeric_type_component_count("uvec4", "uint") == 4
    assert (
        numeric_type_component_count("float4x4", "float", excluded_type_markers=("x",))
        is None
    )
    assert (
        numeric_type_component_count("mat4", "float", excluded_type_markers=("mat",))
        is None
    )
    assert numeric_type_component_count("Texture2D", None) is None


def test_literal_numeric_component_helpers_only_accept_numeric_kinds():
    def literal(kind):
        return SimpleNamespace(literal_type=SimpleNamespace(name=kind))

    assert literal_numeric_component_kind(literal("uint")) == "uint"
    assert literal_numeric_component_count(literal("uint")) == 1
    assert literal_numeric_component_kind(literal("bool")) is None
    assert literal_numeric_component_count(literal("bool")) is None
    assert literal_numeric_component_kind(object()) is None
    assert literal_numeric_component_count(object()) is None


def test_numeric_scalar_type_kind_uses_backend_mapping():
    def type_name_string(vtype):
        return vtype

    def map_type(vtype):
        return {"half": "float", "short": "int"}.get(vtype, vtype)

    assert numeric_scalar_type_kind("half", type_name_string, map_type) == "float"
    assert numeric_scalar_type_kind("uint", type_name_string, map_type) == "uint"
    assert numeric_scalar_type_kind("float2", type_name_string, map_type) is None
    assert numeric_scalar_type_kind(None, type_name_string, map_type) is None


def test_numeric_scalar_expression_kind_prefers_literal_then_inferred_type():
    def expression_result_type(expr):
        return getattr(expr, "expr_type", None)

    def type_name_string(vtype):
        return vtype

    def map_type(vtype):
        return {"half": "float", "short": "int"}.get(vtype, vtype)

    literal = SimpleNamespace(
        literal_type=SimpleNamespace(name="int"), expr_type="half"
    )
    assert (
        numeric_scalar_expression_kind(
            literal, expression_result_type, type_name_string, map_type
        )
        == "int"
    )
    inferred = SimpleNamespace(expr_type="short")
    assert (
        numeric_scalar_expression_kind(
            inferred, expression_result_type, type_name_string, map_type
        )
        == "int"
    )
    vector = SimpleNamespace(expr_type="float2")
    assert (
        numeric_scalar_expression_kind(
            vector, expression_result_type, type_name_string, map_type
        )
        is None
    )


def test_numeric_component_kind_from_type_uses_backend_vector_mapping():
    def type_name_string(vtype):
        return vtype

    def map_type(vtype):
        return {"vec2": "float2", "uintPair": "uint2"}.get(vtype, vtype)

    def vector_component_type(vtype):
        return {"vec2": "float", "uintPair": "uint"}.get(vtype)

    assert (
        numeric_component_kind_from_type(
            "vec2", type_name_string, map_type, vector_component_type
        )
        == "float"
    )
    assert (
        numeric_component_kind_from_type(
            "uintPair", type_name_string, map_type, vector_component_type
        )
        == "uint"
    )
    assert (
        numeric_component_kind_from_type(
            "matrix", type_name_string, map_type, vector_component_type
        )
        is None
    )


def test_numeric_expression_component_kind_prefers_literal_then_inferred_type():
    def expression_result_type(expr):
        return getattr(expr, "expr_type", None)

    def type_name_string(vtype):
        return vtype

    def map_type(vtype):
        return {"vec2": "float2"}.get(vtype, vtype)

    def vector_component_type(vtype):
        return {"vec2": "float"}.get(vtype)

    literal = SimpleNamespace(
        literal_type=SimpleNamespace(name="uint"), expr_type="vec2"
    )
    assert (
        numeric_expression_component_kind(
            literal,
            expression_result_type,
            type_name_string,
            map_type,
            vector_component_type,
        )
        == "uint"
    )
    inferred = SimpleNamespace(expr_type="vec2")
    assert (
        numeric_expression_component_kind(
            inferred,
            expression_result_type,
            type_name_string,
            map_type,
            vector_component_type,
        )
        == "float"
    )


def test_numeric_component_count_from_type_uses_backend_shape_rules():
    def type_name_string(vtype):
        return vtype

    def map_type(vtype):
        return {"vec2": "float2"}.get(vtype, vtype)

    def vector_component_type(vtype):
        return {"vec2": "float", "mat2": "float"}.get(vtype)

    assert (
        numeric_component_count_from_type(
            "vec2",
            type_name_string,
            map_type,
            vector_component_type,
            scalar_types={"float", "double"},
        )
        == 2
    )
    assert (
        numeric_component_count_from_type(
            "float",
            type_name_string,
            map_type,
            vector_component_type,
            scalar_types={"float", "double"},
        )
        == 1
    )
    assert (
        numeric_component_count_from_type(
            "mat2",
            type_name_string,
            map_type,
            vector_component_type,
            excluded_type_markers=("mat",),
        )
        is None
    )


def test_numeric_expression_component_count_prefers_literal_then_inferred_type():
    def expression_result_type(expr):
        return getattr(expr, "expr_type", None)

    def type_name_string(vtype):
        return vtype

    def map_type(vtype):
        return {"vec3": "float3"}.get(vtype, vtype)

    def vector_component_type(vtype):
        return {"vec3": "float"}.get(vtype)

    literal = SimpleNamespace(literal_type=SimpleNamespace(name="float"))
    assert (
        numeric_expression_component_count(
            literal,
            expression_result_type,
            type_name_string,
            map_type,
            vector_component_type,
        )
        == 1
    )
    inferred = SimpleNamespace(expr_type="vec3")
    assert (
        numeric_expression_component_count(
            inferred,
            expression_result_type,
            type_name_string,
            map_type,
            vector_component_type,
        )
        == 3
    )


def test_image_atomic_value_kind_mismatch_reports_first_concrete_mismatch():
    kinds = {"unknown": None, "bad": "int", "later": "float"}

    assert is_image_atomic_operation("imageAtomicAdd")
    assert is_image_atomic_operation("imageAtomicCompSwap")
    assert not is_image_atomic_operation("imageLoad")
    assert not is_image_atomic_operation("imageSamples")

    assert image_atomic_value_kind_mismatch(
        "imageAtomicAdd",
        ["unknown", "bad", "later"],
        "uint",
        kinds.get,
    ) == ("bad", "int")
    assert (
        image_atomic_value_kind_mismatch("imageAtomicAdd", ["bad"], "float", kinds.get)
        is None
    )
    assert image_atomic_value_kind_mismatch(
        "imageAtomicExchange", ["bad"], "float", kinds.get
    ) == ("bad", "int")


def test_image_atomic_result_kind_mismatch_reports_expected_kind():
    assert image_atomic_result_kind_mismatch(None, "uint") is None
    assert image_atomic_result_kind_mismatch("uint", "uint") is None
    assert image_atomic_result_kind_mismatch("float", "double") is None
    assert image_atomic_result_kind_mismatch("int", "uint") == "int"


def test_image_atomic_storage_component_kind_allows_scalar_integer_and_float_exchange():
    assert image_atomic_storage_component_kind("imageAtomicAdd", "int") == "int"
    assert image_atomic_storage_component_kind("imageAtomicAdd", "uint") == "uint"
    assert (
        image_atomic_storage_component_kind("imageAtomicExchange", "float") == "float"
    )
    assert image_atomic_storage_component_kind("imageAtomicAdd", "float") is None
    assert image_atomic_storage_component_kind("imageAtomicAdd", "uint4") is None


def test_image_atomic_error_helpers_build_backend_diagnostics():
    assert (
        image_atomic_format_error("DirectX", "imageAtomicAdd", "rgba32ui")
        == "DirectX image atomic operation 'imageAtomicAdd' requires r32i "
        "or r32ui image format, got rgba32ui"
    )
    assert (
        image_atomic_value_kind_error(
            "OpenGL", "imageAtomicAdd", "r32ui", "uint", "value", "int"
        )
        == "OpenGL image atomic operation 'imageAtomicAdd' requires uint "
        "data argument for r32ui images: value has type int"
    )
    assert (
        image_atomic_result_kind_error(
            "Metal", "imageAtomicExchange", "r32f", "float", "uint"
        )
        == "Metal image atomic operation 'imageAtomicExchange' requires float "
        "result context for r32f images: expected uint"
    )
    assert (
        image_atomic_resource_type_error(
            "DirectX", "imageAtomicAdd", "RWTexture2D<float4>"
        )
        == "DirectX image atomic operation 'imageAtomicAdd' requires a scalar "
        "r32i or r32ui integer storage image, got RWTexture2D<float4>"
    )


def test_resolve_image_atomic_component_kind_validates_format_or_component_type():
    assert (
        resolve_image_atomic_component_kind(
            "imageAtomicAdd",
            "r32ui",
            None,
            "DirectX",
            "RWTexture2D<uint>",
        )
        == "uint"
    )
    assert (
        resolve_image_atomic_component_kind(
            "imageAtomicAdd",
            None,
            "int",
            "Metal",
            "texture2d<int, access::read_write>",
        )
        == "int"
    )
    with pytest.raises(ValueError, match="requires r32i or r32ui image format"):
        resolve_image_atomic_component_kind(
            "imageAtomicAdd",
            "rgba32ui",
            None,
            "OpenGL",
            "uimage2D",
        )
    with pytest.raises(ValueError, match="requires a scalar r32i or r32ui"):
        resolve_image_atomic_component_kind(
            "imageAtomicAdd",
            None,
            "float4",
            "DirectX",
            "RWTexture2D<float4>",
        )


def test_unsupported_image_atomic_fallback_helpers_build_backend_expressions():
    assert storage_image_atomic_zero_value("uint") == "0u"
    assert storage_image_atomic_zero_value("float") == "0.0"
    assert storage_image_atomic_zero_value("int") == "0"
    assert (
        unsupported_image_atomic_expression(
            "Metal",
            "imageAtomicExchange",
            "texture2d<float, access::read_write>",
            "0.0",
        )
        == "/* unsupported Metal image atomic resource call: "
        "imageAtomicExchange on texture2d<float, access::read_write> */ 0.0"
    )
    assert (
        unsupported_multisample_image_atomic_expression(
            "DirectX", "imageAtomicAdd", "RWTexture2DMS<uint>", "0u"
        )
        == "/* unsupported DirectX multisample image atomic: imageAtomicAdd "
        "on RWTexture2DMS<uint> */ 0u"
    )
    assert (
        unsupported_multisample_image_store_expression(
            "DirectX", "RWTexture2DMS<float4>"
        )
        == "/* unsupported DirectX multisample image store: imageStore on "
        "RWTexture2DMS<float4> */ ((void)0)"
    )


def test_image_atomic_helper_descriptor_fields_build_backend_helper_names():
    assert image_atomic_helper_descriptor_fields(
        "imageAtomicAdd", "uint", "image2D", "int2"
    ) == {
        "helper_name": "imageAtomicAdd_uimage2D",
        "return_type": "uint",
        "coord_type": "int2",
    }
    assert image_atomic_helper_descriptor_fields(
        "imageAtomicCompSwap", "int", "image2DArray", "int3"
    ) == {
        "helper_name": "imageAtomicCompSwap_iimage2DArray",
        "return_type": "int",
        "coord_type": "int3",
    }
    assert (
        image_atomic_helper_descriptor_fields(
            "imageAtomicAdd", "float", "image2D", "int2"
        )
        is None
    )
    assert (
        image_atomic_helper_descriptor_fields("imageAtomicAdd", "uint", None, "int2")
        is None
    )


def test_image_atomic_helper_resource_metadata_reports_backend_family_contract():
    assert image_atomic_helper_resource_metadata(
        "RWTexture2DMS",
        {"RWTexture2DMS": "image2DMS"},
        {"RWTexture2DMS": "int2"},
        sample_families={"RWTexture2DMS"},
    ) == {
        "suffix_family": "image2DMS",
        "coord_type": "int2",
        "has_sample": True,
    }
    assert image_atomic_helper_resource_metadata(
        "texture2d_array",
        {"texture2d_array": "image2DArray"},
        {"texture2d_array": "int3"},
        extra_fields_by_family={
            "texture2d_array": {
                "exchange_expr": "image.atomic_compare_exchange_weak(...)"
            }
        },
    ) == {
        "suffix_family": "image2DArray",
        "coord_type": "int3",
        "exchange_expr": "image.atomic_compare_exchange_weak(...)",
    }
    assert (
        image_atomic_helper_resource_metadata(
            "texturecube", {"texture2d": "image2D"}, {"texture2d": "int2"}
        )
        is None
    )
    assert (
        image_atomic_helper_resource_metadata("texture2d", {"texture2d": "image2D"}, {})
        is None
    )
    assert (
        image_atomic_helper_resource_metadata(
            "texture2d",
            {"texture2d": "image2D"},
            {"texture2d": "int2"},
            extra_fields_by_family={},
        )
        is None
    )


def test_resource_query_get_dimensions_descriptor_normalizes_query_metadata():
    assert resource_query_get_dimensions_descriptor(
        "int3",
        ["width", "height", "elements"],
        "int3(width, height, elements)",
        function_params="int lod",
        get_dimensions_args=["lod", "width", "height", "elements", "levels"],
    ) == {
        "size_return_type": "int3",
        "function_params": "int lod",
        "dimensions": ("width", "height", "elements"),
        "get_dimensions_args": ("lod", "width", "height", "elements", "levels"),
        "size_return_expr": "int3(width, height, elements)",
    }
    assert resource_query_get_dimensions_descriptor(
        "int2",
        ("width", "height"),
        "int2(width, height)",
    ) == {
        "size_return_type": "int2",
        "function_params": "",
        "dimensions": ("width", "height"),
        "get_dimensions_args": ("width", "height"),
        "size_return_expr": "int2(width, height)",
    }


def test_resource_query_size_components_descriptor_builds_query_metadata():
    assert resource_query_size_components_descriptor(
        "int3",
        ("width", "height", "elements"),
        tail_dimensions=("levels",),
        function_params="int lod",
        get_dimensions_prefix=("lod",),
    ) == {
        "size_return_type": "int3",
        "function_params": "int lod",
        "dimensions": ("width", "height", "elements", "levels"),
        "get_dimensions_args": ("lod", "width", "height", "elements", "levels"),
        "size_return_expr": "int3(width, height, elements)",
    }
    assert resource_query_size_components_descriptor(
        "int",
        ("width",),
        tail_dimensions=("samples",),
    ) == {
        "size_return_type": "int",
        "function_params": "",
        "dimensions": ("width", "samples"),
        "get_dimensions_args": ("width", "samples"),
        "size_return_expr": "int(width)",
    }


def test_resource_query_size_helper_descriptor_derives_public_size_metadata():
    query_descriptor = resource_query_get_dimensions_descriptor(
        "int3",
        ("width", "height", "elements", "levels"),
        "int3(width, height, elements)",
        function_params="int lod",
        get_dimensions_args=("lod", "width", "height", "elements", "levels"),
    )

    assert resource_query_size_helper_descriptor(query_descriptor) == {
        "return_type": "int3",
        "function_params": "int lod",
        "dimensions": ("width", "height", "elements", "levels"),
        "get_dimensions_args": ("lod", "width", "height", "elements", "levels"),
        "return_expr": "int3(width, height, elements)",
    }
    assert resource_query_size_helper_descriptor(
        query_descriptor, include_function_fields=False
    ) == {
        "return_type": "int3",
        "dimensions": ("width", "height", "elements", "levels"),
        "return_expr": "int3(width, height, elements)",
    }
    assert resource_query_size_helper_descriptor(None) is None


def test_resource_query_scalar_helper_descriptor_derives_public_scalar_metadata():
    query_descriptor = resource_query_get_dimensions_descriptor(
        "int3",
        ("width", "height", "elements", "levels"),
        "int3(width, height, elements)",
        function_params="int lod",
        get_dimensions_args=("0", "width", "height", "elements", "levels"),
    )

    assert resource_query_scalar_helper_descriptor(query_descriptor, "int(levels)") == {
        "return_type": "int",
        "function_params": "",
        "dimensions": ("width", "height", "elements", "levels"),
        "get_dimensions_args": ("0", "width", "height", "elements", "levels"),
        "return_expr": "int(levels)",
    }
    assert resource_query_scalar_helper_descriptor(
        query_descriptor, "uint(samples)", return_type="uint"
    ) == {
        "return_type": "uint",
        "function_params": "",
        "dimensions": ("width", "height", "elements", "levels"),
        "get_dimensions_args": ("0", "width", "height", "elements", "levels"),
        "return_expr": "uint(samples)",
    }
    assert resource_query_scalar_helper_descriptor(None, "int(levels)") is None


def test_resource_query_scalar_constant_helper_descriptor_builds_scalar_metadata():
    assert resource_query_scalar_constant_helper_descriptor("1") == {
        "return_type": "int",
        "function_params": "",
        "dimensions": (),
        "get_dimensions_args": (),
        "return_expr": "1",
    }
    assert resource_query_scalar_constant_helper_descriptor(
        "uint(4)", return_type="uint"
    ) == {
        "return_type": "uint",
        "function_params": "",
        "dimensions": (),
        "get_dimensions_args": (),
        "return_expr": "uint(4)",
    }


def test_resource_query_method_size_descriptor_normalizes_method_metadata():
    assert resource_query_method_size_descriptor(
        "int3",
        [
            ("get_width", True),
            ("get_height", True),
            ("get_array_size", False),
        ],
    ) == {
        "return_type": "int3",
        "dimensions": (
            ("get_width", True),
            ("get_height", True),
            ("get_array_size", False),
        ),
    }


@pytest.mark.parametrize(
    ("resource_type", "expected_dimensions", "mip", "samples"),
    [
        ("image1D", ("width",), False, False),
        ("iimage1D", ("width",), False, False),
        ("uimage1D", ("width",), False, False),
        ("image1DArray", ("width", "elements"), False, False),
        ("iimage1DArray", ("width", "elements"), False, False),
        ("uimage1DArray", ("width", "elements"), False, False),
        ("imageCube", ("width", "height"), False, False),
        ("iimageCube", ("width", "height"), False, False),
        ("uimageCube", ("width", "height"), False, False),
        ("imageCubeArray", ("width", "height", "elements"), False, False),
        ("iimageCubeArray", ("width", "height", "elements"), False, False),
        ("uimageCubeArray", ("width", "height", "elements"), False, False),
        ("image2DMSArray", ("width", "height", "elements"), False, True),
        ("samplerCubeArrayShadow", ("width", "height", "elements"), True, False),
    ],
)
def test_shared_resource_query_specs_cover_array_ms_and_cube_resources(
    resource_type, expected_dimensions, mip, samples
):
    spec = ResourceQueryMixin().dimension_query_spec(resource_type)

    assert spec == {
        "dimensions": expected_dimensions,
        "mip": mip,
        "samples": samples,
    }


def test_texture_query_levels_multisample_expression_matches_single_level():
    assert texture_query_levels_multisample_expression() == "1"
    assert texture_query_levels_zero_value() == "0"


def test_texture_query_lod_coordinate_swizzle_drops_array_layers():
    assert texture_query_lod_coordinate_swizzle("GLSL", "sampler1DArray") == "x"
    assert texture_query_lod_coordinate_swizzle("GLSL", "isampler2DArray") == "xy"
    assert (
        texture_query_lod_coordinate_swizzle("GLSL", "samplerCubeArrayShadow") == "xyz"
    )
    assert texture_query_lod_coordinate_swizzle("DirectX", "Texture1DArray") == "x"
    assert texture_query_lod_coordinate_swizzle("DirectX", "Texture2DArray") == "xy"
    assert texture_query_lod_coordinate_swizzle("DirectX", "TextureCubeArray") == "xyz"
    assert (
        texture_query_lod_coordinate_swizzle("Metal", "texture1d_array<float>") == "x"
    )
    assert texture_query_lod_coordinate_swizzle("Metal", "depth2d_array<float>") == "xy"
    assert (
        texture_query_lod_coordinate_swizzle("Metal", "depthcube_array<float>") == "xyz"
    )
    assert texture_query_lod_coordinate_swizzle("GLSL", "sampler2D") is None


def test_texture_samples_query_expression_matches_backend_syntax():
    assert texture_samples_query_expression("GLSL", "msTex") == "textureSamples(msTex)"
    assert (
        texture_samples_query_expression("DirectX", "textures[2]")
        == "textureSamples(textures[2])"
    )
    assert (
        texture_samples_query_expression("Metal", "arrays[layer]")
        == "int(arrays[layer].get_num_samples())"
    )


def test_texture_samples_query_requirement_name_matches_backend_terms():
    assert texture_samples_query_requirement_name("GLSL") == "sampler"
    assert texture_samples_query_requirement_name("DirectX") == "texture"
    assert texture_samples_query_requirement_name("Metal") == "texture"


def test_unsupported_texture_query_fallback_helpers_build_backend_expressions():
    assert (
        unsupported_texture_query_expression(
            "GLSL", "textureQueryLevels", "image2D", "0"
        )
        == "/* unsupported GLSL texture query: textureQueryLevels on image2D */ 0"
    )
    assert (
        unsupported_texture_query_levels_expression("GLSL", "image2D")
        == "/* unsupported GLSL texture query: textureQueryLevels on image2D */ 0"
    )
    assert (
        unsupported_texture_query_levels_expression(
            "Metal", "texture2d<float, access::read_write>"
        )
        == "/* unsupported Metal texture query: textureQueryLevels on "
        "texture2d<float, access::read_write> */ 0"
    )
    assert (
        unsupported_texture_query_expression(
            "Metal",
            "textureQueryLod",
            "texture2d<float, access::read_write>",
            "float2(0.0)",
        )
        == "/* unsupported Metal texture query: textureQueryLod on "
        "texture2d<float, access::read_write> */ float2(0.0)"
    )
    assert (
        unsupported_multisample_texture_query_expression(
            "DirectX", "textureQueryLod", "Texture2DMS<float4>", "float2(0.0)"
        )
        == "/* unsupported DirectX multisample texture query: "
        "textureQueryLod on Texture2DMS<float4> */ float2(0.0)"
    )
    assert texture_query_lod_zero_value("GLSL") == "vec2(0.0)"
    assert texture_query_lod_zero_value("DirectX") == "float2(0.0, 0.0)"
    assert texture_query_lod_zero_value("Metal") == "float2(0.0)"
    assert (
        unsupported_texture_query_lod_expression("GLSL", "image2D")
        == "/* unsupported GLSL texture query: textureQueryLod on "
        "image2D */ vec2(0.0)"
    )
    assert (
        unsupported_texture_query_lod_expression("DirectX", "RWTexture2D<float4>")
        == "/* unsupported DirectX texture query: textureQueryLod on "
        "RWTexture2D<float4> */ float2(0.0, 0.0)"
    )
    assert (
        unsupported_multisample_texture_query_lod_expression("GLSL", "sampler2DMS")
        == "/* unsupported GLSL multisample texture query: "
        "textureQueryLod on sampler2DMS */ vec2(0.0)"
    )
    assert (
        unsupported_multisample_texture_query_lod_expression(
            "Metal", "texture2d_ms_array<float>"
        )
        == "/* unsupported Metal multisample texture query: "
        "textureQueryLod on texture2d_ms_array<float> */ float2(0.0)"
    )
    assert (
        unsupported_texture_samples_query_expression("GLSL", "sampler")
        == "/* unsupported GLSL texture samples query: "
        "requires multisample sampler */ 0"
    )
    assert (
        unsupported_texture_samples_query_call_expression("DirectX")
        == "/* unsupported DirectX texture samples query: "
        "requires multisample texture */ 0"
    )
    assert (
        unsupported_texture_samples_query_call_expression("Metal")
        == "/* unsupported Metal texture samples query: "
        "requires multisample texture */ 0"
    )
    assert (
        unsupported_texture_samples_query_expression("Metal", "texture")
        == "/* unsupported Metal texture samples query: "
        "requires multisample texture */ 0"
    )


def test_unsupported_multisample_texture_fallback_helpers_build_backend_expressions():
    assert (
        unsupported_multisample_texture_call_expression(
            "GLSL", "texture", "sampler2DMS", "vec4(0.0)"
        )
        == "/* unsupported GLSL multisample texture call: "
        "texture on sampler2DMS */ vec4(0.0)"
    )
    assert (
        unsupported_multisample_texture_call_vector_expression(
            "DirectX", "textureGrad", "Texture2DMS<float4>"
        )
        == "/* unsupported DirectX multisample texture call: "
        "textureGrad on Texture2DMS<float4> */ float4(0.0, 0.0, 0.0, 0.0)"
    )
    assert (
        unsupported_multisample_texture_call_vector_expression(
            "Metal", "texture", "texture2d_ms_array<float>"
        )
        == "/* unsupported Metal multisample texture call: "
        "texture on texture2d_ms_array<float> */ float4(0.0)"
    )
    assert (
        unsupported_multisample_texture_compare_expression(
            "Metal", "textureCompare", "texture2d_ms<float>", "0.0"
        )
        == "/* unsupported Metal multisample texture comparison: "
        "textureCompare on texture2d_ms<float> */ 0.0"
    )
    assert (
        unsupported_multisample_texture_compare_scalar_expression(
            "GLSL", "textureCompareGrad", "sampler2DMS"
        )
        == "/* unsupported GLSL multisample texture comparison: "
        "textureCompareGrad on sampler2DMS */ 0.0"
    )
    assert (
        unsupported_multisample_texture_compare_scalar_expression(
            "DirectX", "textureCompare", "Texture2DMSArray<float4>"
        )
        == "/* unsupported DirectX multisample texture comparison: "
        "textureCompare on Texture2DMSArray<float4> */ 0.0"
    )
    assert (
        unsupported_multisample_texture_gather_compare_expression(
            "DirectX",
            "textureGatherCompare",
            "Texture2DMSArray<float4>",
            "float4(0.0)",
        )
        == "/* unsupported DirectX multisample texture gather comparison: "
        "textureGatherCompare on Texture2DMSArray<float4> */ float4(0.0)"
    )
    assert (
        unsupported_multisample_texture_gather_compare_vector_expression(
            "GLSL", "textureGatherCompare", "sampler2DMS"
        )
        == "/* unsupported GLSL multisample texture gather comparison: "
        "textureGatherCompare on sampler2DMS */ vec4(0.0)"
    )
    assert (
        unsupported_multisample_texture_gather_compare_vector_expression(
            "Metal", "textureGatherCompareOffset", "texture2d_ms_array<float>"
        )
        == "/* unsupported Metal multisample texture gather comparison: "
        "textureGatherCompareOffset on texture2d_ms_array<float> */ float4(0.0)"
    )


def test_unsupported_storage_image_texture_helpers_build_backend_expressions():
    assert (
        unsupported_storage_image_texture_comparison_expression(
            "GLSL", "textureCompare", "image2D", "0.0"
        )
        == "/* unsupported GLSL storage image texture comparison: "
        "textureCompare on image2D */ 0.0"
    )
    assert texture_scalar_zero_value("GLSL") == "0.0"
    assert texture_scalar_zero_value("DirectX") == "0.0"
    assert texture_scalar_zero_value("Metal") == "0.0"
    assert (
        unsupported_storage_image_texture_comparison_scalar_expression(
            "Metal", "textureCompare", "texture2d<float, access::read_write>"
        )
        == "/* unsupported Metal storage image texture comparison: "
        "textureCompare on texture2d<float, access::read_write> */ 0.0"
    )
    assert (
        unsupported_storage_image_texture_comparison_scalar_expression(
            "DirectX", "textureGatherCompare", "RWTexture2D<float4>"
        )
        == "/* unsupported DirectX storage image texture comparison: "
        "textureGatherCompare on RWTexture2D<float4> */ 0.0"
    )
    assert (
        unsupported_storage_image_texture_operation_expression(
            "DirectX", "textureGather", "RWTexture2D<float4>", "float4(0.0)"
        )
        == "/* unsupported DirectX storage image texture operation: "
        "textureGather on RWTexture2D<float4> */ float4(0.0)"
    )
    assert (
        unsupported_storage_image_texture_operation_expression(
            "Metal",
            "texelFetch",
            "texture2d<float, access::read_write>",
            "float4(0.0)",
        )
        == "/* unsupported Metal storage image texture operation: texelFetch "
        "on texture2d<float, access::read_write> */ float4(0.0)"
    )
    assert (
        unsupported_storage_image_texture_operation_vector_expression(
            "GLSL", "texture", "image2D"
        )
        == "/* unsupported GLSL storage image texture operation: texture on "
        "image2D */ vec4(0.0)"
    )
    assert (
        unsupported_storage_image_texture_operation_vector_expression(
            "DirectX", "textureGather", "RWTexture2D<float4>"
        )
        == "/* unsupported DirectX storage image texture operation: "
        "textureGather on RWTexture2D<float4> */ float4(0.0, 0.0, 0.0, 0.0)"
    )


def test_storage_image_texture_operation_classifiers_match_intrinsic_groups():
    assert is_storage_image_texture_comparison_operation("textureCompare")
    assert is_storage_image_texture_comparison_operation("textureCompareProjGrad")
    assert is_storage_image_texture_operation("texture")
    assert is_storage_image_texture_operation("textureGatherCompare")
    assert is_storage_image_texture_operation("texelFetchOffset")
    assert not is_storage_image_texture_comparison_operation("texture")
    assert not is_storage_image_texture_operation("textureCompare")
    assert not is_storage_image_texture_operation("imageLoad")


def test_projected_texture_classifiers_match_intrinsic_groups():
    assert is_projected_texture_operation("textureProj")
    assert is_projected_texture_operation("textureProjGradOffset")
    assert is_projected_texture_compare_operation("textureCompareProj")
    assert is_projected_texture_compare_operation("textureCompareProjGradOffset")
    assert not is_projected_texture_operation("texture")
    assert not is_projected_texture_operation("textureCompareProj")
    assert not is_projected_texture_compare_operation("textureCompare")
    assert not is_projected_texture_compare_operation("textureProj")
    assert PROJECTED_TEXTURE_COMPARE_INTRINSIC_NAMES <= TEXTURE_COMPARE_INTRINSIC_NAMES
    assert not PROJECTED_TEXTURE_INTRINSIC_NAMES & TEXTURE_COMPARE_INTRINSIC_NAMES


def test_projected_texture_shape_classifiers_match_intrinsic_groups():
    assert is_projected_texture_basic_operation("textureProj")
    assert is_projected_texture_basic_offset_operation("textureProjOffset")
    assert is_projected_texture_lod_operation("textureProjLod")
    assert is_projected_texture_lod_offset_operation("textureProjLodOffset")
    assert is_projected_texture_grad_operation("textureProjGrad")
    assert is_projected_texture_grad_offset_operation("textureProjGradOffset")

    assert not is_projected_texture_basic_operation("textureProjOffset")
    assert not is_projected_texture_basic_offset_operation("textureProj")
    assert not is_projected_texture_lod_operation("textureProjLodOffset")
    assert not is_projected_texture_lod_offset_operation("textureProjLod")
    assert not is_projected_texture_grad_operation("textureProjGradOffset")
    assert not is_projected_texture_grad_offset_operation("textureProjGrad")

    assert PROJECTED_TEXTURE_INTRINSIC_NAMES == (
        PROJECTED_TEXTURE_BASIC_INTRINSIC_NAMES
        | PROJECTED_TEXTURE_BASIC_OFFSET_INTRINSIC_NAMES
        | PROJECTED_TEXTURE_LOD_INTRINSIC_NAMES
        | PROJECTED_TEXTURE_LOD_OFFSET_INTRINSIC_NAMES
        | PROJECTED_TEXTURE_GRAD_INTRINSIC_NAMES
        | PROJECTED_TEXTURE_GRAD_OFFSET_INTRINSIC_NAMES
    )
    assert PROJECTED_TEXTURE_OFFSET_INTRINSIC_NAMES == (
        PROJECTED_TEXTURE_BASIC_OFFSET_INTRINSIC_NAMES
        | PROJECTED_TEXTURE_LOD_OFFSET_INTRINSIC_NAMES
        | PROJECTED_TEXTURE_GRAD_OFFSET_INTRINSIC_NAMES
    )


def test_texture_operation_classifiers_match_intrinsic_groups():
    assert is_texture_sample_operation("texture")
    assert is_texture_sample_operation("textureGrad")
    assert is_texture_sample_offset_operation("textureOffset")
    assert is_texture_sample_offset_operation("textureGradOffset")
    assert is_texture_sampling_operation("texture")
    assert is_texture_sampling_operation("textureOffset")
    assert is_texture_sampling_operation("textureProjGradOffset")
    assert is_texture_gather_operation("textureGather")
    assert is_texture_gather_operation("textureGatherOffsets")
    assert is_texture_gather_compare_operation("textureGatherCompare")
    assert is_texture_gather_compare_operation("textureGatherCompareOffset")
    assert is_texture_query_operation("textureQueryLod")
    assert is_texture_query_operation("textureSamples")
    assert not is_texture_sample_operation("textureOffset")
    assert not is_texture_sample_offset_operation("texture")
    assert not is_texture_sampling_operation("textureCompare")
    assert not is_texture_sampling_operation("textureGather")
    assert not is_texture_sampling_operation("texelFetch")
    assert not is_texture_gather_operation("textureGatherCompare")
    assert not is_texture_gather_compare_operation("textureGather")
    assert not is_texture_query_operation("texture")
    assert TEXTURE_SAMPLING_INTRINSIC_NAMES == (
        TEXTURE_SAMPLE_INTRINSIC_NAMES
        | TEXTURE_SAMPLE_OFFSET_INTRINSIC_NAMES
        | PROJECTED_TEXTURE_INTRINSIC_NAMES
    )
    assert not TEXTURE_GATHER_INTRINSIC_NAMES & TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES
    assert not TEXTURE_QUERY_INTRINSIC_NAMES & TEXTURE_SAMPLING_INTRINSIC_NAMES
    assert not TEXTURE_TEXEL_FETCH_INTRINSIC_NAMES & TEXTURE_QUERY_INTRINSIC_NAMES


def test_texture_sample_shape_classifiers_match_intrinsic_groups():
    assert is_texture_sample_basic_operation("texture")
    assert is_texture_sample_lod_operation("textureLod")
    assert is_texture_sample_grad_operation("textureGrad")
    assert is_texture_sample_basic_offset_operation("textureOffset")
    assert is_texture_sample_lod_offset_operation("textureLodOffset")
    assert is_texture_sample_grad_offset_operation("textureGradOffset")

    assert not is_texture_sample_basic_operation("textureLod")
    assert not is_texture_sample_lod_operation("textureLodOffset")
    assert not is_texture_sample_grad_operation("textureGradOffset")
    assert not is_texture_sample_basic_offset_operation("texture")
    assert not is_texture_sample_lod_offset_operation("textureLod")
    assert not is_texture_sample_grad_offset_operation("textureGrad")
    assert TEXTURE_SAMPLE_INTRINSIC_NAMES == (
        TEXTURE_SAMPLE_BASIC_INTRINSIC_NAMES
        | TEXTURE_SAMPLE_LOD_INTRINSIC_NAMES
        | TEXTURE_SAMPLE_GRAD_INTRINSIC_NAMES
    )
    assert TEXTURE_SAMPLE_OFFSET_INTRINSIC_NAMES == (
        TEXTURE_SAMPLE_BASIC_OFFSET_INTRINSIC_NAMES
        | TEXTURE_SAMPLE_LOD_OFFSET_INTRINSIC_NAMES
        | TEXTURE_SAMPLE_GRAD_OFFSET_INTRINSIC_NAMES
    )
    assert not TEXTURE_SAMPLE_INTRINSIC_NAMES & TEXTURE_SAMPLE_OFFSET_INTRINSIC_NAMES


def test_texture_query_shape_classifiers_match_intrinsic_groups():
    assert is_texture_query_lod_operation("textureQueryLod")
    assert is_texture_query_levels_operation("textureQueryLevels")
    assert is_texture_size_query_operation("textureSize")
    assert is_texture_samples_query_operation("textureSamples")
    assert is_resource_size_query_operation("imageSize")
    assert is_resource_size_query_operation("textureSize")
    assert is_resource_samples_query_operation("imageSamples")
    assert is_resource_samples_query_operation("textureSamples")

    assert not is_texture_query_lod_operation("textureQueryLevels")
    assert not is_texture_query_levels_operation("textureQueryLod")
    assert not is_texture_size_query_operation("imageSize")
    assert not is_texture_samples_query_operation("imageSamples")
    assert TEXTURE_QUERY_INTRINSIC_NAMES == (
        TEXTURE_QUERY_LOD_INTRINSIC_NAMES
        | TEXTURE_QUERY_LEVELS_INTRINSIC_NAMES
        | TEXTURE_QUERY_SIZE_INTRINSIC_NAMES
        | TEXTURE_QUERY_SAMPLES_INTRINSIC_NAMES
    )
    assert RESOURCE_QUERY_SIZE_INTRINSIC_NAMES == {"textureSize", "imageSize"}
    assert RESOURCE_QUERY_SAMPLES_INTRINSIC_NAMES == {
        "textureSamples",
        "imageSamples",
    }


def test_texture_compare_shape_classifiers_match_intrinsic_groups():
    assert is_texture_compare_operation("textureCompare")
    assert is_texture_compare_operation("textureCompareProjGradOffset")
    assert is_texture_compare_basic_operation("textureCompare")
    assert is_texture_compare_basic_operation("textureCompareProj")
    assert is_texture_compare_offset_operation("textureCompareProjOffset")
    assert is_texture_compare_lod_operation("textureCompareProjLod")
    assert is_texture_compare_lod_offset_operation("textureCompareLodOffset")
    assert is_texture_compare_grad_operation("textureCompareProjGrad")
    assert is_texture_compare_grad_offset_operation("textureCompareGradOffset")
    assert is_texture_compare_non_projected_operation("textureCompareGradOffset")
    assert is_texture_compare_non_projected_offset_operation("textureCompareGradOffset")

    assert not is_texture_compare_basic_operation("textureCompareOffset")
    assert not is_texture_compare_lod_operation("textureCompareLodOffset")
    assert not is_texture_compare_grad_operation("textureCompareGradOffset")
    assert not is_texture_compare_operation("texture")
    assert not is_texture_compare_operation("textureProj")
    assert not is_texture_compare_non_projected_operation("textureCompareProj")
    assert not is_texture_compare_non_projected_offset_operation(
        "textureCompareProjOffset"
    )
    assert TEXTURE_COMPARE_INTRINSIC_NAMES == (
        TEXTURE_COMPARE_BASIC_INTRINSIC_NAMES
        | TEXTURE_COMPARE_OFFSET_INTRINSIC_NAMES
        | TEXTURE_COMPARE_LOD_INTRINSIC_NAMES
        | TEXTURE_COMPARE_LOD_OFFSET_INTRINSIC_NAMES
        | TEXTURE_COMPARE_GRAD_INTRINSIC_NAMES
        | TEXTURE_COMPARE_GRAD_OFFSET_INTRINSIC_NAMES
    )
    assert TEXTURE_COMPARE_NON_PROJECTED_INTRINSIC_NAMES == (
        TEXTURE_COMPARE_INTRINSIC_NAMES - PROJECTED_TEXTURE_COMPARE_INTRINSIC_NAMES
    )
    assert TEXTURE_COMPARE_NON_PROJECTED_OFFSET_OPERATION_NAMES == (
        TEXTURE_COMPARE_OFFSET_OPERATION_NAMES
        - PROJECTED_TEXTURE_COMPARE_INTRINSIC_NAMES
    )
    assert PROJECTED_TEXTURE_COMPARE_INTRINSIC_NAMES <= TEXTURE_COMPARE_INTRINSIC_NAMES


def test_texture_gather_shape_classifiers_match_intrinsic_groups():
    assert is_texture_gather_basic_operation("textureGather")
    assert is_texture_gather_single_offset_operation("textureGatherOffset")
    assert is_texture_gather_multi_offset_operation("textureGatherOffsets")
    assert is_texture_gather_offset_operation("textureGatherOffset")
    assert is_texture_gather_offset_operation("textureGatherOffsets")

    assert not is_texture_gather_basic_operation("textureGatherOffset")
    assert not is_texture_gather_single_offset_operation("textureGather")
    assert not is_texture_gather_multi_offset_operation("textureGatherOffset")
    assert not is_texture_gather_offset_operation("textureGather")
    assert TEXTURE_GATHER_OFFSET_INTRINSIC_NAMES == (
        TEXTURE_GATHER_SINGLE_OFFSET_INTRINSIC_NAMES
        | TEXTURE_GATHER_MULTI_OFFSET_INTRINSIC_NAMES
    )
    assert TEXTURE_GATHER_INTRINSIC_NAMES == (
        TEXTURE_GATHER_BASIC_INTRINSIC_NAMES | TEXTURE_GATHER_OFFSET_INTRINSIC_NAMES
    )
    assert not (
        TEXTURE_GATHER_BASIC_INTRINSIC_NAMES & TEXTURE_GATHER_OFFSET_INTRINSIC_NAMES
    )
    assert not (
        TEXTURE_GATHER_SINGLE_OFFSET_INTRINSIC_NAMES
        & TEXTURE_GATHER_MULTI_OFFSET_INTRINSIC_NAMES
    )


def test_texture_resource_and_sampler_intrinsic_groups_are_shared_unions():
    assert TEXTURE_RESOURCE_INTRINSIC_NAMES == (
        TEXTURE_SAMPLING_INTRINSIC_NAMES
        | TEXTURE_COMPARE_INTRINSIC_NAMES
        | TEXTURE_GATHER_INTRINSIC_NAMES
        | TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES
        | TEXTURE_QUERY_INTRINSIC_NAMES
        | TEXTURE_TEXEL_FETCH_INTRINSIC_NAMES
    )
    assert TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES == (
        TEXTURE_SAMPLING_INTRINSIC_NAMES
        | TEXTURE_COMPARE_INTRINSIC_NAMES
        | TEXTURE_GATHER_INTRINSIC_NAMES
        | TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES
        | TEXTURE_QUERY_LOD_INTRINSIC_NAMES
    )
    for intrinsic in TEXTURE_RESOURCE_INTRINSIC_NAMES:
        assert is_texture_resource_operation(intrinsic)
    for intrinsic in TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES:
        assert is_texture_implicit_sampler_operation(intrinsic)

    assert is_texture_resource_operation("textureSamples")
    assert is_texture_resource_operation("texelFetch")
    assert is_texture_implicit_sampler_operation("textureGatherCompare")
    assert is_texture_implicit_sampler_operation("textureQueryLod")
    assert not is_texture_implicit_sampler_operation("textureSize")
    assert not is_texture_implicit_sampler_operation("texelFetch")
    assert not is_texture_resource_operation("imageLoad")
    assert is_image_resource_operation("imageLoad", {"imageLoad", "imageSize"})
    assert is_image_resource_operation("imageSize", {"imageLoad", "imageSize"})
    assert not is_image_resource_operation("imageSamples", {"imageLoad", "imageSize"})
    assert not is_image_resource_operation("textureSize", {"imageLoad", "imageSize"})
    assert texture_image_resource_operation_names({"imageLoad", "imageSize"}) == (
        TEXTURE_RESOURCE_INTRINSIC_NAMES | {"imageLoad", "imageSize", "imageSamples"}
    )


def test_texture_offset_classifiers_match_intrinsic_groups():
    assert is_texture_sampling_offset_operation("textureOffset")
    assert is_texture_sampling_offset_operation("textureProjGradOffset")
    assert is_texture_gather_offset_operation("textureGatherOffsets")
    assert is_texture_gather_compare_offset_operation("textureGatherCompareOffset")
    assert is_texture_compare_offset_operation("textureCompareProjOffset")
    assert is_texture_compare_lod_offset_operation("textureCompareProjLodOffset")
    assert is_texture_compare_grad_offset_operation("textureCompareProjGradOffset")
    assert is_texture_compare_any_offset_operation("textureCompareGradOffset")
    assert is_texel_fetch_offset_operation("texelFetchOffset")

    for intrinsic in TEXTURE_OFFSET_INTRINSIC_NAMES:
        assert is_texture_offset_operation(intrinsic)

    assert not is_texture_offset_operation("texture")
    assert not is_texture_sampling_offset_operation("textureCompareOffset")
    assert not is_texture_compare_any_offset_operation("textureProjOffset")
    assert TEXTURE_SAMPLING_OFFSET_INTRINSIC_NAMES == (
        TEXTURE_SAMPLE_OFFSET_INTRINSIC_NAMES | PROJECTED_TEXTURE_OFFSET_INTRINSIC_NAMES
    )
    assert TEXTURE_COMPARE_OFFSET_OPERATION_NAMES == (
        TEXTURE_COMPARE_OFFSET_INTRINSIC_NAMES
        | TEXTURE_COMPARE_LOD_OFFSET_INTRINSIC_NAMES
        | TEXTURE_COMPARE_GRAD_OFFSET_INTRINSIC_NAMES
    )
    assert TEXTURE_GATHER_OFFSET_INTRINSIC_NAMES <= TEXTURE_GATHER_INTRINSIC_NAMES
    assert (
        TEXTURE_GATHER_COMPARE_OFFSET_INTRINSIC_NAMES
        <= TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES
    )
    assert (
        TEXTURE_TEXEL_FETCH_OFFSET_INTRINSIC_NAMES
        <= TEXTURE_TEXEL_FETCH_INTRINSIC_NAMES
    )


def test_texture_resource_offset_dimension_key_routes_operation_shapes():
    assert (
        texture_resource_offset_dimension_key("textureGatherOffset")
        == "gather_offset_dimension"
    )
    assert (
        texture_resource_offset_dimension_key("textureGatherCompareOffset")
        == "gather_compare_offset_dimension"
    )
    assert (
        texture_resource_offset_dimension_key("textureCompareOffset")
        == "compare_offset_dimension"
    )
    assert (
        texture_resource_offset_dimension_key("textureCompareLodOffset")
        == "compare_lod_offset_dimension"
    )
    assert (
        texture_resource_offset_dimension_key("textureCompareGradOffset")
        == "compare_grad_offset_dimension"
    )
    assert (
        texture_resource_offset_dimension_key(
            "textureCompareGradOffset", collapse_compare_offsets=True
        )
        == "compare_offset_dimension"
    )
    assert (
        texture_resource_offset_dimension_key("texelFetchOffset")
        == "texel_fetch_offset_dimension"
    )
    assert (
        texture_resource_offset_dimension_key("textureProjLodOffset")
        == "sample_offset_dimension"
    )
    assert texture_resource_offset_dimension_key("imageLoad") == "offset_dimension"


def test_texel_fetch_classifiers_match_intrinsic_groups():
    assert is_texel_fetch_operation("texelFetch")
    assert is_texel_fetch_operation("texelFetchOffset")
    assert is_texel_fetch_basic_operation("texelFetch")
    assert is_texel_fetch_offset_operation("texelFetchOffset")

    assert not is_texel_fetch_operation("texture")
    assert not is_texel_fetch_basic_operation("texelFetchOffset")
    assert not is_texel_fetch_offset_operation("texelFetch")

    assert TEXTURE_TEXEL_FETCH_INTRINSIC_NAMES == (
        TEXTURE_TEXEL_FETCH_BASIC_INTRINSIC_NAMES
        | TEXTURE_TEXEL_FETCH_OFFSET_INTRINSIC_NAMES
    )
    assert not TEXTURE_TEXEL_FETCH_INTRINSIC_NAMES & TEXTURE_QUERY_INTRINSIC_NAMES


def test_unsupported_texture_gather_and_texel_fetch_helpers_build_backend_expressions():
    assert (
        unsupported_texture_gather_expression(
            "GLSL",
            "textureGather",
            "requires 2D, 2D-array, cube, or cube-array textures",
            "vec4(0.0)",
        )
        == "/* unsupported GLSL texture gather: textureGather requires 2D, "
        "2D-array, cube, or cube-array textures */ vec4(0.0)"
    )
    assert (
        unsupported_texture_gather_call_expression(
            "Metal",
            "textureGatherOffsets",
            "offsets require 2D or 2D-array textures",
        )
        == "/* unsupported Metal texture gather: textureGatherOffsets offsets "
        "require 2D or 2D-array textures */ float4(0.0)"
    )
    assert (
        unsupported_texture_gather_compare_expression(
            "DirectX",
            "textureGatherCompareOffset",
            "offsets require 2D or 2D-array depth textures",
            "float4(0.0)",
        )
        == "/* unsupported DirectX texture gather compare: "
        "textureGatherCompareOffset offsets require 2D or 2D-array depth "
        "textures */ float4(0.0)"
    )
    assert (
        unsupported_texture_gather_compare_call_expression(
            "GLSL",
            "textureGatherCompareOffset",
            "offsets require 2D or 2D-array depth textures",
        )
        == "/* unsupported GLSL texture gather compare: "
        "textureGatherCompareOffset offsets require 2D or 2D-array depth "
        "textures */ vec4(0.0)"
    )
    assert (
        unsupported_texel_fetch_expression(
            "Metal", "texelFetch", "texturecube<float>", "float4(0.0)"
        )
        == "/* unsupported Metal texel fetch: texelFetch on "
        "texturecube<float> */ float4(0.0)"
    )
    assert texel_fetch_zero_value("GLSL") == "vec4(0.0)"
    assert texel_fetch_zero_value("DirectX") == "float4(0.0, 0.0, 0.0, 0.0)"
    assert texel_fetch_zero_value("Metal") == "float4(0.0)"
    assert (
        unsupported_cube_texel_fetch_expression(
            "GLSL", "texelFetchOffset", "samplerCube"
        )
        == "/* unsupported GLSL texel fetch: texelFetchOffset on "
        "samplerCube */ vec4(0.0)"
    )
    assert (
        unsupported_cube_texel_fetch_expression(
            "DirectX", "texelFetch", "TextureCubeArray"
        )
        == "/* unsupported DirectX texel fetch: texelFetch on "
        "TextureCubeArray */ float4(0.0, 0.0, 0.0, 0.0)"
    )
    assert (
        unsupported_cube_texel_fetch_expression(
            "Metal", "texelFetch", "texturecube<float>"
        )
        == "/* unsupported Metal texel fetch: texelFetch on "
        "texturecube<float> */ float4(0.0)"
    )


def test_unsupported_texture_offset_helpers_build_backend_expressions():
    assert (
        texel_fetch_offset_multisample_reason("GLSL", "sampler2DMS")
        == "multisample texture sampler2DMS does not support offsets"
    )
    assert (
        texel_fetch_offset_multisample_reason("DirectX")
        == "multisample textures do not support offsets"
    )
    assert (
        texel_fetch_offset_multisample_reason("Metal")
        == "multisample textures do not support offsets"
    )
    assert (
        unsupported_texture_offset_expression(
            "GLSL",
            "textureOffset",
            "requires offset and optional bias arguments",
            "vec4(0.0)",
        )
        == "/* unsupported GLSL texture offset: textureOffset requires "
        "offset and optional bias arguments */ vec4(0.0)"
    )
    assert texture_vector_zero_value("GLSL") == "vec4(0.0)"
    assert texture_vector_zero_value("DirectX") == "float4(0.0, 0.0, 0.0, 0.0)"
    assert texture_vector_zero_value("Metal") == "float4(0.0)"
    assert (
        unsupported_texture_offset_call_expression(
            "DirectX",
            "textureGradOffset",
            "offsets require 1D, 2D, 2D-array, or 3D textures",
        )
        == "/* unsupported DirectX texture offset: textureGradOffset "
        "offsets require 1D, 2D, 2D-array, or 3D textures */ "
        "float4(0.0, 0.0, 0.0, 0.0)"
    )
    assert (
        unsupported_texture_offset_call_expression(
            "Metal", "textureOffset", "offsets require 2D, 2D-array, or 3D textures"
        )
        == "/* unsupported Metal texture offset: textureOffset offsets "
        "require 2D, 2D-array, or 3D textures */ float4(0.0)"
    )
    assert (
        unsupported_texel_fetch_offset_expression(
            "GLSL",
            "multisample texture sampler2DMS does not support offsets",
            "vec4(0.0)",
        )
        == "/* unsupported GLSL texel fetch offset: multisample texture "
        "sampler2DMS does not support offsets */ vec4(0.0)"
    )
    assert (
        unsupported_texel_fetch_offset_expression(
            "DirectX",
            "multisample textures do not support offsets",
            "float4(0.0)",
        )
        == "/* unsupported DirectX texel fetch offset: multisample "
        "textures do not support offsets */ float4(0.0)"
    )
    assert (
        unsupported_multisample_texel_fetch_offset_expression(
            "GLSL", "sampler2DMSArray"
        )
        == "/* unsupported GLSL texel fetch offset: multisample texture "
        "sampler2DMSArray does not support offsets */ vec4(0.0)"
    )
    assert (
        unsupported_multisample_texel_fetch_offset_expression("DirectX")
        == "/* unsupported DirectX texel fetch offset: multisample "
        "textures do not support offsets */ float4(0.0, 0.0, 0.0, 0.0)"
    )
    assert (
        unsupported_multisample_texel_fetch_offset_expression("Metal")
        == "/* unsupported Metal texel fetch offset: multisample textures "
        "do not support offsets */ float4(0.0)"
    )


def test_unsupported_projected_texture_helper_builds_backend_expressions():
    assert (
        unsupported_projected_texture_expression(
            "GLSL",
            "textureProjGradOffset",
            "requires 1D, 2D, 2D-array, or 3D projection coordinates",
            "vec4(0.0)",
        )
        == "/* unsupported GLSL projected texture: textureProjGradOffset "
        "requires 1D, 2D, 2D-array, or 3D projection coordinates */ vec4(0.0)"
    )
    assert (
        unsupported_projected_texture_expression(
            "Metal",
            "textureProj",
            "requires 1D, 2D, or 3D projection coordinates",
            "float4(0.0)",
        )
        == "/* unsupported Metal projected texture: textureProj requires "
        "1D, 2D, or 3D projection coordinates */ float4(0.0)"
    )
    assert (
        unsupported_projected_texture_call_expression(
            "DirectX",
            "textureProj",
            "requires 1D, 2D, or 3D projection coordinates",
        )
        == "/* unsupported DirectX projected texture: textureProj requires "
        "1D, 2D, or 3D projection coordinates */ float4(0.0, 0.0, 0.0, 0.0)"
    )
    assert (
        unsupported_projected_texture_call_expression(
            "GLSL",
            "textureProjGradOffset",
            "requires 1D, 2D, 2D-array, or 3D projection coordinates",
        )
        == "/* unsupported GLSL projected texture: textureProjGradOffset "
        "requires 1D, 2D, 2D-array, or 3D projection coordinates */ vec4(0.0)"
    )


def test_unsupported_texture_compare_helper_builds_backend_expressions():
    assert (
        unsupported_texture_compare_expression(
            "GLSL",
            "textureCompareProjLodOffset",
            "projected explicit LOD is not supported for sampler2DArrayShadow",
            "0.0",
        )
        == "/* unsupported GLSL texture compare: textureCompareProjLodOffset "
        "projected explicit LOD is not supported for sampler2DArrayShadow */ 0.0"
    )
    assert (
        unsupported_texture_compare_expression(
            "DirectX",
            "textureCompareProjGradOffset",
            "requires Texture2D vec3/vec4 or Texture2DArray vec4 projection coordinates",
            "0.0",
        )
        == "/* unsupported DirectX texture compare: textureCompareProjGradOffset "
        "requires Texture2D vec3/vec4 or Texture2DArray vec4 projection coordinates */ 0.0"
    )
    assert (
        unsupported_texture_compare_expression(
            "Metal",
            "textureCompareProjGradOffset",
            "requires depth2d vec3/vec4 or depth2d_array vec4 projection coordinates",
            "0.0",
        )
        == "/* unsupported Metal texture compare: textureCompareProjGradOffset "
        "requires depth2d vec3/vec4 or depth2d_array vec4 projection coordinates */ 0.0"
    )
    assert (
        unsupported_texture_compare_scalar_expression(
            "GLSL",
            "textureCompareLodOffset",
            "explicit LOD offsets require 2D shadow samplers",
        )
        == "/* unsupported GLSL texture compare: textureCompareLodOffset "
        "explicit LOD offsets require 2D shadow samplers */ 0.0"
    )
    assert (
        unsupported_texture_compare_scalar_expression(
            "Metal",
            "textureCompareOffset",
            "offsets require 2D or 2D-array depth textures",
        )
        == "/* unsupported Metal texture compare: textureCompareOffset "
        "offsets require 2D or 2D-array depth textures */ 0.0"
    )


def test_component_count_mismatch_allows_unknown_scalar_and_exact_counts():
    assert component_count_mismatch(None, 4) is None
    assert component_count_mismatch(4, None) is None
    assert component_count_mismatch(4, 1) is None
    assert component_count_mismatch(4, 4) is None
    assert component_count_mismatch(4, 2) == 2
    assert component_count_mismatch(4, 1, allow_scalar=False) == 1


def test_component_kind_mismatch_reports_concrete_conflicts():
    assert component_kind_mismatch(None, "float") is None
    assert component_kind_mismatch("float", None) is None
    assert component_kind_mismatch("float", "float") is None
    assert component_kind_mismatch("float", "int") == "int"


def test_image_store_value_mismatch_helpers_match_component_rules():
    assert image_store_value_shape_mismatch(4, 1) is None
    assert image_store_value_shape_mismatch(4, 2) == 2
    assert image_store_value_kind_mismatch("uint", "uint") is None
    assert image_store_value_kind_mismatch("uint", "float") == "float"


def test_image_store_value_error_helpers_build_backend_diagnostics():
    assert (
        image_store_value_shape_error("OpenGL", "rgba32f", "value", 4, 2)
        == "OpenGL image store operation 'imageStore' requires scalar or "
        "4-component value for rgba32f images: value has 2 components"
    )
    assert (
        image_store_value_kind_error("Metal", "rg32ui", "value", "uint", "float")
        == "Metal image store operation 'imageStore' requires uint value for "
        "rg32ui images: value has type float"
    )


def test_image_load_result_kind_mismatch_requires_numeric_component_kind():
    assert image_load_result_kind_mismatch(None, "float") is None
    assert image_load_result_kind_mismatch("float", "double") is None
    assert image_load_result_kind_mismatch("float", "float") is None
    assert image_load_result_kind_mismatch("float", "int") == "float"


def test_image_load_result_shape_mismatch_matches_component_rules():
    assert image_load_result_shape_mismatch(4, 1) is None
    assert image_load_result_shape_mismatch(4, 4) is None
    assert image_load_result_shape_mismatch(2, 4) == 4


def test_image_load_result_error_helpers_build_backend_diagnostics():
    assert (
        image_load_result_kind_error("DirectX", "r32ui", "uint", "int")
        == "DirectX image load operation 'imageLoad' requires uint result "
        "context for r32ui images: expected int"
    )
    assert (
        image_load_result_shape_error("OpenGL", "rgba32f", 4, 2)
        == "OpenGL image load operation 'imageLoad' requires scalar or "
        "4-component result context for rgba32f images: expected 2-component"
    )


def test_should_validate_image_load_result_shape_requires_type_signal():
    assert not should_validate_image_load_result_shape(None, "float")
    assert not should_validate_image_load_result_shape("float", "double")
    assert should_validate_image_load_result_shape("float", "int")


def test_image_format_or_default_channel_count_prefers_explicit_format():
    assert image_format_or_default_channel_count("rg32f", 4) == 2
    assert image_format_or_default_channel_count("rgba32ui", 1) == 4
    assert image_format_or_default_channel_count(None, 3) == 3


def test_default_storage_image_channel_count_accepts_numeric_component_kinds():
    assert default_storage_image_channel_count("float") == 4
    assert default_storage_image_channel_count("int") == 4
    assert default_storage_image_channel_count("uint") == 4
    assert default_storage_image_channel_count("float2") is None
    assert default_storage_image_channel_count(None) is None


def test_component_shape_requirement_describes_scalar_or_exact_width():
    assert component_shape_requirement(1, "value") == "scalar value"
    assert component_shape_requirement(2, "value") == "scalar or 2-component value"
    assert (
        component_shape_requirement(4, "result context")
        == "scalar or 4-component result context"
    )


def test_image_atomic_format_helpers_allow_float_only_for_exchange():
    assert image_atomic_format_allowed_names("imageAtomicAdd") == ("r32i", "r32ui")
    assert image_atomic_format_requirement("imageAtomicAdd") == "r32i or r32ui"
    assert (
        image_atomic_explicit_format_component_kind("imageAtomicAdd", "r32ui") == "uint"
    )
    assert image_atomic_explicit_format_component_kind("imageAtomicAdd", "r32f") is None
    assert image_atomic_format_allowed_names("imageAtomicExchange") == (
        "r32i",
        "r32ui",
        "r32f",
    )
    assert (
        image_atomic_explicit_format_component_kind("imageAtomicExchange", "r32f")
        == "float"
    )
    assert (
        image_atomic_format_requirement("imageAtomicExchange") == "r32i, r32ui, or r32f"
    )


def test_image_atomic_value_arguments_accounts_for_samples_and_compare_swap():
    args = [0, 1, 2, 3, 4, 5]

    assert image_atomic_value_arguments("imageAtomicAdd", args, False) == [2]
    assert image_atomic_value_arguments("imageAtomicAdd", args, True) == [3]
    assert image_atomic_value_arguments("imageAtomicCompSwap", args, False) == [2, 3]
    assert image_atomic_value_arguments("imageAtomicCompSwap", args, True) == [3, 4]


@pytest.mark.parametrize(
    ("func_name", "component_kind", "expected"),
    [
        ("imageAtomicAdd", "int", True),
        ("imageAtomicAdd", "uint", True),
        ("imageAtomicAdd", "float", False),
        ("imageAtomicExchange", "float", True),
        ("imageAtomicExchange", "double", False),
        ("imageAtomicExchange", None, False),
    ],
)
def test_should_validate_image_atomic_component_kind(
    func_name, component_kind, expected
):
    assert (
        should_validate_image_atomic_component_kind(func_name, component_kind)
        is expected
    )


@pytest.mark.parametrize(
    ("func_name", "argument_count"),
    [
        ("imageLoad", 3),
        ("imageStore", 4),
        ("imageAtomicAdd", 4),
        ("imageAtomicExchange", 4),
        ("imageAtomicCompSwap", 5),
    ],
)
def test_image_multisample_sample_argument_index_reports_sample_slot(
    func_name, argument_count
):
    assert (
        image_multisample_sample_argument_index(
            func_name, argument_count, True, "OpenGL"
        )
        == 2
    )


@pytest.mark.parametrize(
    ("func_name", "argument_count"),
    [
        ("imageLoad", 2),
        ("imageStore", 3),
        ("imageAtomicAdd", 3),
        ("imageAtomicCompSwap", 4),
        ("texture", 3),
    ],
)
def test_image_multisample_sample_argument_index_ignores_regular_image_calls(
    func_name, argument_count
):
    assert (
        image_multisample_sample_argument_index(
            func_name, argument_count, False, "DirectX"
        )
        is None
    )


@pytest.mark.parametrize(
    ("func_name", "argument_count", "message"),
    [
        (
            "imageLoad",
            2,
            "Metal multisample image operation 'imageLoad' requires image, "
            "coordinate, and sample index arguments, got 2",
        ),
        (
            "imageStore",
            3,
            "Metal multisample image operation 'imageStore' requires image, "
            "coordinate, sample index, and value arguments, got 3",
        ),
        (
            "imageAtomicAdd",
            3,
            "Metal multisample image atomic operation 'imageAtomicAdd' "
            "requires image, coordinate, sample index, and value arguments, got 3",
        ),
        (
            "imageAtomicCompSwap",
            4,
            "Metal multisample image atomic operation 'imageAtomicCompSwap' "
            "requires image, coordinate, sample index, compare, and value "
            "arguments, got 4",
        ),
    ],
)
def test_image_multisample_sample_argument_index_rejects_missing_sample_argument(
    func_name, argument_count, message
):
    with pytest.raises(ValueError, match=message):
        image_multisample_sample_argument_index(
            func_name, argument_count, True, "Metal"
        )


@pytest.mark.parametrize(
    ("func_name", "argument_count", "message"),
    [
        (
            "imageLoad",
            3,
            "OpenGL texture operation 'imageLoad' accepts at most 2 "
            "argument\\(s\\), got 3",
        ),
        (
            "imageStore",
            4,
            "OpenGL texture operation 'imageStore' accepts at most 3 "
            "argument\\(s\\), got 4",
        ),
        (
            "imageAtomicAdd",
            4,
            "OpenGL texture operation 'imageAtomicAdd' accepts at most 3 "
            "argument\\(s\\), got 4",
        ),
        (
            "imageAtomicCompSwap",
            5,
            "OpenGL texture operation 'imageAtomicCompSwap' accepts at most "
            "4 argument\\(s\\), got 5",
        ),
    ],
)
def test_image_multisample_sample_argument_index_rejects_extra_regular_image_args(
    func_name, argument_count, message
):
    with pytest.raises(ValueError, match=message):
        image_multisample_sample_argument_index(
            func_name, argument_count, False, "OpenGL"
        )


def test_image_multisample_sample_type_helpers_validate_scalar_integer_samples():
    def is_scalar_integer_type(type_name):
        return type_name == "int"

    assert image_multisample_sample_type_mismatch(None, is_scalar_integer_type) is None
    assert image_multisample_sample_type_mismatch("int", is_scalar_integer_type) is None
    assert (
        image_multisample_sample_type_mismatch("float", is_scalar_integer_type)
        == "float"
    )
    assert (
        image_multisample_sample_type_error(
            "Metal", "imageLoad", "sampleValue", "float"
        )
        == "Metal multisample image operation 'imageLoad' requires a scalar "
        "integer sample index argument: sampleValue has type float"
    )


def test_texture_multisample_sample_type_error_matches_texel_fetch_diagnostic():
    assert (
        texture_multisample_sample_type_error(
            "DirectX", "texelFetch", "sampleValue", "float"
        )
        == "DirectX multisample texel fetch operation 'texelFetch' requires a "
        "scalar integer sample index argument: sampleValue has type float"
    )


def test_image_format_type_helpers_build_backend_type_names():
    assert image_format_component_type("r32ui") == "uint"
    assert image_format_vector_type("rg32i") == "int2"
    assert image_format_vector_type("rgba16f") == "float4"


def test_image_format_result_type_supports_backend_vector_prefixes():
    assert image_format_result_type("r32f") == "float"
    assert image_format_result_type("rg32i") == "int2"
    assert (
        image_format_result_type(
            "rgba32ui",
            vector_prefixes={"float": "vec", "int": "ivec", "uint": "uvec"},
        )
        == "uvec4"
    )
    assert (
        image_format_result_type(
            "r32ui",
            vector_prefixes={"float": "vec", "int": "ivec", "uint": "uvec"},
        )
        == "uint"
    )


def test_image_format_channel_predicates():
    assert is_scalar_image_format("r16f")
    assert is_two_component_image_format("rg8ui")
    assert not is_two_component_image_format("rgba8")


def test_storage_image_load_component_suffix_uses_format_and_context():
    assert (
        storage_image_load_component_suffix(
            None, expected_scalar=False, scalar_integer_resource=True
        )
        == ".x"
    )
    assert (
        storage_image_load_component_suffix(
            "rgba32f", expected_scalar=True, float_resource=True
        )
        == ".x"
    )
    assert storage_image_load_component_suffix("rg32f", expected_scalar=False) == ".xy"
    assert storage_image_load_component_suffix("rg32f", expected_scalar=True) == ".x"
    assert storage_image_load_component_suffix("rgba32f", expected_scalar=False) == ""


def test_storage_image_store_constructor_uses_scalar_format_kind():
    constructors = storage_image_store_constructors("vec4", "ivec4", "uvec4")

    assert storage_image_format_store_constructor("r32ui", constructors) == "uvec4"
    assert storage_image_format_store_constructor("rg32ui", constructors) is None


def test_storage_image_store_vector_constructor_uses_component_width_and_kind():
    zeros = storage_image_zero_values()

    assert storage_image_store_vector_constructor("float4", 4, "float") == "float4"
    assert storage_image_store_vector_constructor(
        "uint2", 2, "uint", zero_values_by_kind=zeros
    ) == ("uint2", "0u")
    assert storage_image_store_vector_constructor("float2", 4, "float") is None
    assert storage_image_store_vector_constructor("bool2", 2, None) is None


def test_storage_image_two_component_store_expression_pads_values():
    constructors = storage_image_store_constructors("vec4", "ivec4", "uvec4")
    zeros = storage_image_zero_values()

    assert (
        storage_image_two_component_store_expression(
            "rg32i", "value", False, constructors, zeros
        )
        == "ivec4(value, 0, 0)"
    )
    assert (
        storage_image_two_component_store_expression(
            "rg32i", "value", True, constructors, zeros
        )
        == "ivec4(value, 0, 0, 0)"
    )


def test_storage_image_store_value_expression_packs_required_values():
    constructors = storage_image_store_constructors("vec4", "ivec4", "uvec4")
    zeros = storage_image_zero_values()

    assert (
        storage_image_store_value_expression(
            "rg32ui",
            "value",
            False,
            constructors_by_kind=constructors,
            zero_values_by_kind=zeros,
        )
        == "uvec4(value, 0u, 0u)"
    )
    assert (
        storage_image_store_value_expression(
            "r32f",
            "value",
            True,
            scalar_integer_resource=True,
            constructors_by_kind=constructors,
        )
        == "vec4(value)"
    )


def test_glsl_storage_image_type_helpers_match_expected_families():
    assert is_glsl_storage_image_type("imageCube")
    assert is_glsl_integer_image_type("iimage2DArray")
    assert is_glsl_integer_image_type("uimage1D")
    assert is_glsl_float_image_resource("image3D")
    assert not is_glsl_float_image_resource("imageCube")
    assert not is_glsl_integer_image_type("image2D")


def test_metal_storage_image_type_helpers_ignore_access_mode():
    readonly_texture = "texture2d<int, access::read>"
    readwrite_texture = "texture2d<int, access::read_write>"
    sampled_texture = "texture2d<float>"

    assert is_metal_storage_image_resource(readonly_texture)
    assert (
        metal_storage_image_access_agnostic_type(readonly_texture) == readwrite_texture
    )
    assert metal_storage_image_component_type(readonly_texture) == "int"
    assert is_metal_integer_image_type(readonly_texture)
    assert not is_metal_storage_image_resource(sampled_texture)


def test_metal_float_image_resource_helper_matches_storage_textures():
    assert is_metal_float_image_resource("texture1d_array<float, access::write>")
    assert not is_metal_float_image_resource("texture2d<uint, access::read_write>")
    assert (
        storage_image_store_value_expression(
            None,
            "value",
            False,
            scalar_integer_resource=True,
            integer_constructor="ivec4",
        )
        == "ivec4(value)"
    )
    assert (
        storage_image_store_value_expression(
            None,
            "value",
            True,
            float_resource=True,
            float_constructor="vec4",
        )
        == "vec4(value)"
    )
