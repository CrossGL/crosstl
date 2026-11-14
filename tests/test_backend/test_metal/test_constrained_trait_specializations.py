"""Trait specializations constrained by enable_if, defaulted trait parameters and Metal's SIMD-group type trait."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

TRAITS = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
struct wrapper {
    uint i;
    T val;
};

template <typename T>
struct pair_value {
    T a;
    T b;
};

template <typename T>
constexpr constant bool is_simd_t = __is_valid_simdgroup_type<T>::value;

template <typename T, typename _E = void>
struct is_valid_simd_type {
    static constant constexpr bool value = false;
};

template <typename T>
constexpr constant bool is_valid_simd_t = is_valid_simd_type<T>::value;

template <typename T>
struct is_valid_simd_type<T, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};

template <typename T>
struct is_valid_simd_type<wrapper<T>, typename metal::enable_if_t<is_valid_simd_t<T>>> {
    static constant constexpr bool value = true;
};

template <>
struct is_valid_simd_type<int64_t> {
    static constant constexpr bool value = true;
};
"""

REDUCERS = """
template <typename T, typename _E = void>
struct reducer;

template <typename T>
struct reducer<T, typename metal::enable_if_t<is_valid_simd_t<T>>> {
    uint kind() const { return 1u; }
};

template <typename T>
struct reducer<T, typename metal::enable_if_t<!is_valid_simd_t<T>>> {
    uint kind() const { return 2u; }
};

kernel void reducer_kinds(device uint* out [[buffer(0)]],
                          uint gid [[thread_position_in_grid]]) {
    reducer<float> scalar;
    reducer<wrapper<half>> wrapped;
    reducer<pair_value<float>> paired;
    reducer<int64_t> wide;
    out[gid] = scalar.kind() + 10u * wrapped.kind() + 100u * paired.kind()
        + 1000u * wide.kind();
}
"""


def _evaluate(expression: str, code: str = TRAITS) -> bool:
    preprocessor = MetalPreprocessor()
    return preprocessor._evaluate_boolean_constraint(
        expression,
        {},
        boolean_templates=preprocessor._find_boolean_variable_templates(code),
        position=len(code),
    )


@pytest.mark.parametrize(
    ("type_name", "expected"),
    [
        ("float", True),
        ("wrapper<half>", True),
        ("wrapper<wrapper<int>>", True),
        ("int64_t", True),
        ("pair_value<float>", False),
        ("wrapper<pair_value<float>>", False),
        ("bool", False),
    ],
)
def test_enable_if_constrained_trait_specializations(type_name, expected):
    assert _evaluate(f"is_valid_simd_t<{type_name}>") is expected
    assert _evaluate(f"is_valid_simd_type<{type_name}, void>::value") is expected


def test_trait_arguments_take_the_primary_defaults():
    code = TRAITS + """
template <typename T, typename U = T>
struct same_pair { static constexpr bool value = false; };

template <typename T>
struct same_pair<T, T> { static constexpr bool value = true; };
"""

    assert _evaluate("same_pair<float>::value", code)
    assert not _evaluate("same_pair<float, int>::value", code)


def test_constrained_reducers_select_each_specialization(tmp_path):
    path = tmp_path / "reducers.metal"
    path.write_text(TRAITS + REDUCERS)

    output = crosstl.translate(str(path), backend="cgl", format_output=False)

    kinds = {
        name: int(value)
        for name, value in re.findall(
            r"(reducer_\w+?)__kind\([^)]*\)[^{;]*\{\s*return (\d)u;", output
        )
    }
    assert kinds == {
        "reducer_float_void": 1,
        "reducer_wrapper_half_void": 1,
        "reducer_pair_value_float_void": 2,
        "reducer_int64_t_void": 1,
    }


SIMDGROUP_TYPES = {
    "float": True,
    "half4": True,
    "packed_float3": True,
    "char2": True,
    "ushort3": True,
    "uint8_t": True,
    "const float": True,
    "int64_t": False,
    "long": False,
    "bfloat": False,
    "bool": False,
    "float2x2": False,
    "pair_value<float>": False,
}


@pytest.mark.parametrize("type_name", sorted(SIMDGROUP_TYPES))
def test_simdgroup_type_trait_follows_metal_simdgroup(type_name):
    assert (
        _evaluate(f"__is_valid_simdgroup_type<{type_name}>::value")
        is SIMDGROUP_TYPES[type_name]
    )


@pytest.mark.parametrize("type_name", ["vec<float, 8>", "unknown_type"])
def test_simdgroup_type_trait_refuses_undecided_operands(type_name):
    with pytest.raises(MetalPreprocessor._UnrecognizedConstraint):
        _evaluate(f"__is_valid_simdgroup_type<{type_name}>::value")


def test_simdgroup_type_table_matches_the_metal_compiler(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None or (
        subprocess.run(
            [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True
        ).returncode
        != 0
    ):
        pytest.skip("the Metal compiler is not available")
    assertions = "\n".join(
        f"static_assert({'' if valid else '!'}__is_valid_simdgroup_type<{name}>::value);"
        for name, valid in sorted(SIMDGROUP_TYPES.items())
    )
    path = tmp_path / "simdgroup_types.metal"
    path.write_text(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "template <typename T> struct pair_value { T a; T b; };\n"
        f"{assertions}\n"
        "kernel void k(device float* out [[buffer(0)]]) { out[0] = 1.0f; }\n"
    )
    completed = subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(path), "-o", str(tmp_path / "a")],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
