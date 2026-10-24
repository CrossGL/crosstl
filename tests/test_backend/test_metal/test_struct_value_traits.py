"""enable_if constraints may read the Boolean ``value`` of a user struct-template trait."""

import re

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalTemplateSpecializationError,
)

PRELUDE = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
struct indexed {
    uint i;
    T val;
};

template <typename T>
struct plain {
    T val;
};
"""

VALUE_MEMBER_TRAIT = """
template <typename T>
struct is_indexed_type {
    static constant constexpr bool value = false;
};

template <typename T>
constexpr constant bool is_indexed_t = is_indexed_type<T>::value;

template <typename T>
struct is_indexed_type<indexed<T>> {
    static constant constexpr bool value = true;
};

template <typename T>
constexpr constant bool not_indexed_t = !is_indexed_t<T>;
"""

BASE_CLASS_TRAIT = """
template <typename T>
struct is_indexed_type : metal::false_type {};

template <typename T>
struct is_indexed_type<indexed<T>> : metal::true_type {};

template <typename T>
struct is_indexed_alias : is_indexed_type<T> {};

template <typename T>
constexpr constant bool is_indexed_t = is_indexed_alias<T>::value;

template <typename T>
constexpr constant bool not_indexed_t = !is_indexed_t<T>;
"""

LOADERS = """
template <typename T, typename R, typename _E = void>
struct loader;

template <typename T, typename R>
struct loader<T, R, typename metal::enable_if_t<NOT_INDEXED>> {
    R load(T x) const { R r; r.val = x; return r; }
};

template <typename T, typename R>
struct loader<T, R, typename metal::enable_if_t<INDEXED>> {
    R load(T x) const { R r; r.i = 7; r.val = x; return r; }
};

kernel void load_values(device float* in [[buffer(0)]],
                        device plain<float>* out [[buffer(1)]],
                        device indexed<float>* indexed_out [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    loader<float, plain<float>> a;
    out[gid] = a.load(in[gid]);
    loader<float, indexed<float>> b;
    indexed_out[gid] = b.load(in[gid]);
}
"""

CASES = {
    "variable-templates": (VALUE_MEMBER_TRAIT, "not_indexed_t<R>", "is_indexed_t<R>"),
    "value-member": (
        VALUE_MEMBER_TRAIT,
        "!is_indexed_type<R>::value",
        "is_indexed_type<R>::value",
    ),
    "base-classes": (BASE_CLASS_TRAIT, "not_indexed_t<R>", "is_indexed_t<R>"),
}


def _source(trait: str, not_indexed: str, indexed: str) -> str:
    loaders = LOADERS.replace("NOT_INDEXED", not_indexed).replace("INDEXED", indexed)
    return PRELUDE + trait + loaders


def _load_body(output: str, struct: str) -> str:
    definitions = re.findall(rf"{struct}__load\([^)]*\)[^{{;]*\{{([^}}]*)\}}", output)
    assert len(definitions) == 1, output
    return definitions[0]


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("backend", ["directx", "cgl"])
def test_trait_selects_each_constrained_partial_specialization(tmp_path, case, backend):
    path = tmp_path / "loaders.metal"
    path.write_text(_source(*CASES[case]))

    output = crosstl.translate(str(path), backend=backend, format_output=False)

    plain_body = _load_body(output, "loader_float_plain_float_void")
    indexed_body = _load_body(output, "loader_float_indexed_float_void")
    assert "r.i = 7" not in plain_body
    assert "r.i = 7" in indexed_body


def test_full_specialization_and_bool_constant_values():
    preprocessor = MetalPreprocessor()
    code = PRELUDE + """
template <typename T>
struct is_wide : metal::bool_constant<(sizeof(T) > 4)> {};

template <typename T>
struct is_special { static constexpr bool value = false; };

template <>
struct is_special<int> { static constexpr bool value = true; };
"""
    templates = preprocessor._find_boolean_variable_templates(code)

    def evaluate(expression):
        return preprocessor._evaluate_boolean_constraint(
            expression, {}, boolean_templates=templates, position=len(code)
        )

    assert evaluate("is_special<int>::value")
    assert not evaluate("is_special<float>::value")
    assert evaluate("is_wide<float4>::value")
    assert not evaluate("is_wide<float>::value")


def test_trait_names_are_not_variable_templates():
    preprocessor = MetalPreprocessor()
    code = PRELUDE + VALUE_MEMBER_TRAIT
    templates = preprocessor._find_boolean_variable_templates(code)

    with pytest.raises(MetalPreprocessor._UnrecognizedConstraint):
        preprocessor._evaluate_boolean_constraint(
            "is_indexed_type<float>", {}, boolean_templates=templates
        )
    with pytest.raises(MetalPreprocessor._UnrecognizedConstraint):
        preprocessor._evaluate_boolean_constraint(
            "is_indexed_t<float>::value", {}, boolean_templates=templates
        )


def test_unreadable_trait_value_still_refuses_the_specialization(tmp_path):
    trait = """
template <typename T>
constexpr bool compute_indexed() { return false; }

template <typename T>
struct is_indexed_type {
    static constexpr bool value = compute_indexed<T>();
};
"""
    path = tmp_path / "loaders.metal"
    path.write_text(
        _source(trait, "!is_indexed_type<R>::value", "is_indexed_type<R>::value")
    )

    with pytest.raises(MetalTemplateSpecializationError, match="enable_if"):
        crosstl.translate(str(path), backend="directx", format_output=False)


def test_member_templates_are_not_namespace_scope_traits():
    preprocessor = MetalPreprocessor()
    code = PRELUDE + """
struct Holder {
    template <typename T>
    struct nested { static constexpr bool value = true; };
};
"""
    templates = preprocessor._find_boolean_variable_templates(code)

    assert "nested" not in templates
