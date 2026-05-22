from types import SimpleNamespace

import pytest

from crosstl.translator.ast import AttributeNode
from crosstl.translator.codegen.image_access_contracts import (
    explicit_image_access,
    explicit_image_format,
    image_format_channel_count,
    image_format_component_kind,
    image_format_component_type,
    image_format_vector_type,
    is_glsl_float_image_resource,
    is_glsl_integer_image_type,
    is_glsl_storage_image_type,
    is_image_format_attribute,
    is_metal_float_image_resource,
    is_metal_integer_image_type,
    is_metal_storage_image_resource,
    is_resource_access_attribute,
    is_scalar_image_format,
    is_two_component_image_format,
    metal_storage_image_access_agnostic_type,
    metal_storage_image_component_type,
    normalized_image_access,
    storage_image_format_store_constructor,
    storage_image_load_component_suffix,
    storage_image_store_value_expression,
    storage_image_two_component_store_expression,
    supported_image_formats,
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


def test_image_format_type_helpers_build_backend_type_names():
    assert image_format_component_type("r32ui") == "uint"
    assert image_format_vector_type("rg32i") == "int2"
    assert image_format_vector_type("rgba16f") == "float4"


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
    constructors = {"float": "vec4", "int": "ivec4", "uint": "uvec4"}

    assert storage_image_format_store_constructor("r32ui", constructors) == "uvec4"
    assert storage_image_format_store_constructor("rg32ui", constructors) is None


def test_storage_image_two_component_store_expression_pads_values():
    constructors = {"float": "vec4", "int": "ivec4", "uint": "uvec4"}
    zeros = {"float": "0.0", "int": "0", "uint": "0u"}

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
    constructors = {"float": "vec4", "int": "ivec4", "uint": "uvec4"}
    zeros = {"float": "0.0", "int": "0", "uint": "0u"}

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
        metal_storage_image_access_agnostic_type(readonly_texture)
        == readwrite_texture
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
