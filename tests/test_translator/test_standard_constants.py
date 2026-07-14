from dataclasses import FrozenInstanceError, fields

import pytest

from crosstl.translator.standard_constants import (
    STANDARD_MATH_CONSTANTS,
    StandardMathConstant,
    render_standard_math_constant,
    standard_math_constant,
)

STANDARD_NAMES = {
    "M_E",
    "M_LOG2E",
    "M_LOG10E",
    "M_LN2",
    "M_LN10",
    "M_PI",
    "M_PI_2",
    "M_PI_4",
    "M_1_PI",
    "M_2_PI",
    "M_2_SQRTPI",
    "M_SQRT2",
    "M_SQRT1_2",
}


def test_pi_records_preserve_exact_decimal_spellings():
    pi = standard_math_constant("M_PI")
    pi_float = standard_math_constant("M_PI_F")

    assert pi == StandardMathConstant(
        name="M_PI",
        scalar_type="double",
        decimal="3.14159265358979323846264338327950288",
        source_environments=frozenset({"c", "cpp", "opencl"}),
    )
    assert pi_float == StandardMathConstant(
        name="M_PI_F",
        scalar_type="float",
        decimal="3.14159265358979323846264338327950288",
        source_environments=frozenset({"metal", "opencl"}),
    )
    assert pi.decimal_spelling == pi.decimal
    assert pi_float.decimal_spelling == pi_float.decimal


def test_registry_has_complete_typed_constant_pairs():
    expected_names = STANDARD_NAMES | {f"{name}_F" for name in STANDARD_NAMES}

    assert set(STANDARD_MATH_CONSTANTS) == expected_names
    assert len(STANDARD_MATH_CONSTANTS) == 26
    assert {field.name for field in fields(StandardMathConstant)} == {
        "name",
        "scalar_type",
        "decimal",
        "source_environments",
    }

    for name in STANDARD_NAMES:
        double_constant = standard_math_constant(name)
        float_constant = standard_math_constant(f"{name}_F")

        assert double_constant is STANDARD_MATH_CONSTANTS[name]
        assert float_constant is STANDARD_MATH_CONSTANTS[f"{name}_F"]
        assert double_constant.name == name
        assert double_constant.scalar_type == "double"
        assert double_constant.source_environments == frozenset({"c", "cpp", "opencl"})
        assert float_constant.name == f"{name}_F"
        assert float_constant.scalar_type == "float"
        assert float_constant.source_environments == frozenset({"metal", "opencl"})
        assert double_constant.decimal == float_constant.decimal


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("directx", "3.14159265358979323846264338327950288f"),
        ("hlsl", "3.14159265358979323846264338327950288f"),
        (" DirectX ", "3.14159265358979323846264338327950288f"),
        ("HLSL", "3.14159265358979323846264338327950288f"),
        ("opengl", "3.14159265358979323846264338327950288f"),
        ("glsl", "3.14159265358979323846264338327950288f"),
        (" OpenGL ", "3.14159265358979323846264338327950288f"),
        ("GLSL", "3.14159265358979323846264338327950288f"),
    ],
)
def test_render_normalizes_supported_target_aliases(target, expected):
    assert render_standard_math_constant("M_PI_F", target) == expected


@pytest.mark.parametrize(
    ("name", "target", "suffix"),
    [
        ("M_E_F", "hlsl", "f"),
        ("M_E", "hlsl", "L"),
        ("M_E_F", "glsl", "f"),
        ("M_E", "glsl", "LF"),
    ],
)
def test_render_uses_target_and_scalar_type_suffixes(name, target, suffix):
    constant = standard_math_constant(name)

    assert render_standard_math_constant(name, target) == f"{constant.decimal}{suffix}"


def test_unknown_constants_and_unsupported_targets_return_none():
    assert standard_math_constant("M_TAU") is None
    assert standard_math_constant("m_pi") is None
    assert standard_math_constant(None) is None
    assert render_standard_math_constant("M_TAU", "hlsl") is None
    assert render_standard_math_constant("M_PI", "metal") is None
    assert render_standard_math_constant("M_PI", "spirv") is None
    assert render_standard_math_constant("M_PI", "") is None
    assert render_standard_math_constant("M_PI", None) is None


def test_records_and_registry_are_immutable():
    pi = standard_math_constant("M_PI")

    with pytest.raises(FrozenInstanceError):
        pi.name = "M_TAU"
    with pytest.raises(AttributeError):
        pi.source_environments.add("metal")
    with pytest.raises(TypeError):
        STANDARD_MATH_CONSTANTS["M_PI"] = StandardMathConstant(
            name="M_PI",
            scalar_type="double",
            decimal="3.0",
            source_environments=frozenset(),
        )
