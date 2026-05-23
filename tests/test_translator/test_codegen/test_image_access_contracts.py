from types import SimpleNamespace

import pytest

from crosstl.translator.ast import AttributeNode
from crosstl.translator.codegen.image_access_contracts import (
    component_count_mismatch,
    component_kind_mismatch,
    component_shape_requirement,
    default_storage_image_channel_count,
    explicit_image_access,
    explicit_image_format,
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
    image_load_result_kind_error,
    image_load_result_kind_mismatch,
    image_load_result_shape_error,
    image_load_result_shape_mismatch,
    image_format_or_default_channel_count,
    image_format_channel_count,
    image_format_component_kind,
    image_format_component_type,
    image_format_result_type,
    image_format_vector_type,
    image_multisample_sample_argument_index,
    image_multisample_sample_type_error,
    image_multisample_sample_type_mismatch,
    image_resource_metadata,
    image_store_value_kind_error,
    image_store_value_kind_mismatch,
    image_store_value_shape_error,
    image_store_value_shape_mismatch,
    is_glsl_float_image_resource,
    is_glsl_integer_image_type,
    is_glsl_storage_image_type,
    is_image_format_attribute,
    is_metal_float_image_resource,
    is_metal_integer_image_type,
    is_metal_storage_image_resource,
    is_resource_access_attribute,
    is_scalar_image_format,
    is_storage_image_texture_comparison_operation,
    is_storage_image_texture_operation,
    is_two_component_image_format,
    literal_numeric_component_count,
    literal_numeric_component_kind,
    metal_storage_image_access_agnostic_type,
    metal_storage_image_component_type,
    numeric_component_count_from_type,
    numeric_component_kind_from_type,
    numeric_expression_component_count,
    numeric_expression_component_kind,
    numeric_scalar_expression_kind,
    numeric_scalar_type_kind,
    numeric_type_component_count,
    numeric_type_component_kind,
    normalized_image_access,
    record_explicit_image_metadata,
    resolve_image_atomic_component_kind,
    resource_query_get_dimensions_descriptor,
    resource_query_scalar_helper_descriptor,
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
    unsupported_multisample_texture_call_expression,
    unsupported_multisample_texture_compare_expression,
    unsupported_multisample_texture_gather_compare_expression,
    unsupported_multisample_texture_query_expression,
    unsupported_storage_image_texture_comparison_expression,
    unsupported_storage_image_texture_operation_expression,
    unsupported_image_atomic_expression,
    unsupported_multisample_image_atomic_expression,
    unsupported_multisample_image_store_expression,
    unsupported_texture_query_expression,
    unsupported_texture_samples_query_expression,
)


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


def test_unsupported_texture_query_fallback_helpers_build_backend_expressions():
    assert (
        unsupported_texture_query_expression(
            "GLSL", "textureQueryLevels", "image2D", "0"
        )
        == "/* unsupported GLSL texture query: textureQueryLevels on image2D */ 0"
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
    assert (
        unsupported_texture_samples_query_expression("GLSL", "sampler")
        == "/* unsupported GLSL texture samples query: "
        "requires multisample sampler */ 0"
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
        unsupported_multisample_texture_compare_expression(
            "Metal", "textureCompare", "texture2d_ms<float>", "0.0"
        )
        == "/* unsupported Metal multisample texture comparison: "
        "textureCompare on texture2d_ms<float> */ 0.0"
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


def test_unsupported_storage_image_texture_helpers_build_backend_expressions():
    assert (
        unsupported_storage_image_texture_comparison_expression(
            "GLSL", "textureCompare", "image2D", "0.0"
        )
        == "/* unsupported GLSL storage image texture comparison: "
        "textureCompare on image2D */ 0.0"
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


def test_storage_image_texture_operation_classifiers_match_intrinsic_groups():
    assert is_storage_image_texture_comparison_operation("textureCompare")
    assert is_storage_image_texture_comparison_operation("textureCompareProjGrad")
    assert is_storage_image_texture_operation("texture")
    assert is_storage_image_texture_operation("textureGatherCompare")
    assert is_storage_image_texture_operation("texelFetchOffset")
    assert not is_storage_image_texture_comparison_operation("texture")
    assert not is_storage_image_texture_operation("textureCompare")
    assert not is_storage_image_texture_operation("imageLoad")


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
