import pytest

from crosstl.project.directx_toolchain import (
    directx_target_profiles_for_source,
    dxc_compiler_arguments_for_source,
    dxc_profile_for_source,
    hlsl_requires_native_16bit_types,
)

NATIVE_16_BIT_SOURCE = "RWStructuredBuffer<uint16_t> values;"


@pytest.mark.parametrize(
    "source",
    (
        "float16_t value;",
        "int16_t2 value;",
        "uint16_t4 value;",
        "vector<float16_t, 3> value;",
    ),
)
def test_native_16_bit_scalar_and_vector_types_are_detected(source):
    assert hlsl_requires_native_16bit_types(source) is True


@pytest.mark.parametrize(
    "source",
    (
        "min16float value; min16int2 pair; min16uint4 lanes;",
        "float16_type value;",
        "my_int16_t value;",
        "uint16_t_values value;",
        "float16_/* do not join tokens across comments */t value;",
    ),
)
def test_non_native_or_embedded_type_names_are_not_detected(source):
    assert hlsl_requires_native_16bit_types(source) is False


@pytest.mark.parametrize(
    "source",
    (
        "// uint16_t4 ignored;\nfloat4 value;",
        "/* float16_t ignored;\nint16_t2 also_ignored; */ float value;",
        "float value; /* unterminated uint16_t4 comment",
        'string type_name = "uint16_t4"; float value;',
        r'string type_name = "escaped \" // int16_t2"; float value;',
    ),
)
def test_native_type_names_in_comments_and_literals_are_ignored(source):
    assert hlsl_requires_native_16bit_types(source) is False


@pytest.mark.parametrize(
    "source",
    (
        'string marker = "// uint16_t4"; int16_t value;',
        'string marker = "/* float16_t */"; uint16_t2 value;',
    ),
)
def test_comment_markers_in_literals_do_not_hide_following_code(source):
    assert hlsl_requires_native_16bit_types(source) is True


@pytest.mark.parametrize(
    ("profile", "expected"),
    (
        ("vs_6_0", "vs_6_2"),
        ("ps_6_1", "ps_6_2"),
        ("cs_6_2", "cs_6_2"),
        ("lib_6_3", "lib_6_3"),
        ("ms_6_6", "ms_6_6"),
        ("cs_6_10", "cs_6_10"),
        ("cs_7_0", "cs_7_0"),
    ),
)
def test_native_types_raise_profiles_to_6_2_without_lowering(profile, expected):
    assert dxc_profile_for_source(profile, NATIVE_16_BIT_SOURCE) == expected


def test_profiles_without_native_types_and_unrecognized_profiles_are_unchanged():
    assert dxc_profile_for_source("cs_6_0", "float value;") == "cs_6_0"
    assert dxc_profile_for_source("not-a-profile", NATIVE_16_BIT_SOURCE) == (
        "not-a-profile"
    )


@pytest.mark.parametrize(
    "source",
    (
        "float16_t a; int16_t2 b; uint16_t4 c;",
        "uint16_t4 c; float16_t a; int16_t2 b;",
    ),
)
def test_native_types_have_deterministic_compiler_and_target_requirements(source):
    assert dxc_compiler_arguments_for_source(source) == ("-enable-16bit-types",)
    assert directx_target_profiles_for_source(source) == ("directx-12",)


def test_non_native_types_keep_default_compiler_and_target_requirements():
    source = "min16float4 value; // uint16_t is only documentation"

    assert dxc_compiler_arguments_for_source(source) == ()
    assert directx_target_profiles_for_source(source) == (
        "directx-11",
        "directx-12",
    )
