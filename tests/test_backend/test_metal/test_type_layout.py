import pytest

from crosstl.backend.Metal.type_layout import metal_type_layout, metal_type_size


@pytest.mark.parametrize(
    ("type_text", "expected"),
    [
        ("bool", 1),
        ("char", 1),
        ("signed char", 1),
        ("int8_t", 1),
        ("unsigned char", 1),
        ("uint8_t", 1),
        ("short", 2),
        ("signed short", 2),
        ("int16_t", 2),
        ("unsigned short", 2),
        ("uint16_t", 2),
        ("half", 2),
        ("bfloat", 2),
        ("bfloat16", 2),
        ("bfloat16_t", 2),
        ("float16_t", 2),
        ("int", 4),
        ("signed", 4),
        ("int32_t", 4),
        ("unsigned", 4),
        ("uint32_t", 4),
        ("float", 4),
        ("float32_t", 4),
        ("long", 8),
        ("signed long", 8),
        ("int64_t", 8),
        ("unsigned long", 8),
        ("uint64_t", 8),
        ("double", 8),
        ("float64_t", 8),
        ("size_t", 8),
        ("ptrdiff_t", 8),
    ],
)
def test_metal_type_size_supports_scalar_spelling_and_fixed_width_aliases(
    type_text, expected
):
    assert metal_type_size(type_text) == expected


@pytest.mark.parametrize(
    ("type_text", "expected"),
    [
        ("bool3", 4),
        ("char3", 4),
        ("short3", 8),
        ("int3", 16),
        ("long3", 32),
        ("half3", 8),
        ("bfloat3", 8),
        ("float2", 8),
        ("float3", 16),
        ("float4", 16),
        ("vector<int8_t, 3>", 4),
        ("vector<uint16_t, 3>", 8),
        ("metal::vector<uint32_t, 1 + 2>", 16),
        ("::metal::vec<int64_t, 3>", 32),
    ],
)
def test_metal_type_size_uses_padded_three_lane_vector_layout(type_text, expected):
    assert metal_type_size(type_text) == expected


@pytest.mark.parametrize(
    ("type_text", "expected"),
    [
        ("packed_char3", 3),
        ("packed_short3", 6),
        ("packed_int3", 12),
        ("packed_half3", 6),
        ("packed_bfloat3", 6),
        ("metal::packed_float3", 12),
        ("packed_long3", 24),
        ("packed_vec<uint16_t, 3>", 6),
    ],
)
def test_metal_type_size_keeps_packed_vectors_tightly_packed(type_text, expected):
    assert metal_type_size(type_text) == expected


@pytest.mark.parametrize(
    ("type_text", "expected"),
    [
        ("half2x3", 16),
        ("half3x2", 12),
        ("float2x3", 32),
        ("float3x2", 24),
        ("float3x3", 48),
        ("float4x3", 64),
        ("matrix<float, 3, 3>", 48),
        ("metal::matrix<half, 4, 3>", 32),
    ],
)
def test_metal_type_size_treats_matrices_as_aligned_columns(type_text, expected):
    assert metal_type_size(type_text) == expected


@pytest.mark.parametrize(
    ("type_text", "expected"),
    [
        ("float3[2]", 32),
        ("packed_float3[2]", 24),
        ("float[2][3]", 24),
        ("array<float3, 2>", 32),
        ("metal::array<array<half3, 2>, 3>", 48),
        ("array<packed_float3, 2 + 1>", 36),
        ("vector<uint16_t, 3>[0x2u]", 16),
        ("array<float, (1 << 2)>", 16),
    ],
)
def test_metal_type_size_composes_fixed_array_layout(type_text, expected):
    assert metal_type_size(type_text) == expected


@pytest.mark.parametrize(
    ("type_text", "expected"),
    [
        ("const float", 4),
        ("volatile metal::float3", 16),
        ("threadgroup const metal::array<half3, 2>", 16),
        ("float const[2]", 8),
        ("array<const uint32_t, 4>", 16),
    ],
)
def test_metal_type_size_strips_only_layout_neutral_qualifiers(type_text, expected):
    assert metal_type_size(type_text) == expected


@pytest.mark.parametrize(
    ("type_text", "expected"),
    [
        ("float", (4, 4)),
        ("float3", (16, 16)),
        ("packed_float3", (12, 4)),
        ("half3", (8, 8)),
        ("float3[2]", (32, 16)),
        ("array<packed_float3, 2>", (24, 4)),
    ],
)
def test_metal_type_layout_reports_object_alignment(type_text, expected):
    assert metal_type_layout(type_text) == expected


@pytest.mark.parametrize(
    "type_text",
    [
        None,
        "",
        "void",
        "atomic_int",
        "sampler",
        "Widget",
        "struct Widget",
        "foo::float",
        "std::array<float, 4>",
        "c10::metal::array<float, 4>",
        "float*",
        "device float *",
        "float&",
        "array<device float, 4>",
        "packed_vec<bool, 3>",
        "vector<float, 1>",
        "vector<float, 5>",
        "vector<float, N>",
        "vector<float3, 2>",
        "matrix<int, 3, 3>",
        "matrix<bfloat, 3, 3>",
        "matrix<bfloat16_t, 3, 3>",
        "bfloat3x3",
        "matrix<float, 3>",
        "matrix<float, 3, Rows>",
        "float[]",
        "float[N]",
        "float[0]",
        "array<float, 0>",
        "array<float, 4 / 0>",
        "array<float>",
        "array<float, 2, 3>",
        "static float",
        "uniform float",
        "const const float",
        "metal::::float",
        "float value",
        "float[2] trailing",
        "array<float, 1 << 62>",
        "array<float, 1 << 1000000>",
        f"float[{'9' * 4100}]",
    ],
)
def test_metal_type_size_rejects_non_object_dependent_and_malformed_types(type_text):
    assert metal_type_size(type_text) is None
    assert metal_type_layout(type_text) is None
