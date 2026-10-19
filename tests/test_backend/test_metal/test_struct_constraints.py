import pytest

from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalTemplateSpecializationError,
)


def materialize(source):
    return MetalPreprocessor()._materialize_explicit_template_struct_instantiations(
        source, allow_partial_object_specializations=True
    )


@pytest.mark.parametrize("declarator", ["typename", "class", "typename Enable"])
@pytest.mark.parametrize(
    "pattern",
    [
        "enable_if_t<accepts<T>>",
        "metal::enable_if_t<is_same_v<T, float>>",
        "typename metal::enable_if<is_same<T, float>::value>::type",
    ],
)
@pytest.mark.parametrize("scalar,field", [("float", "float"), ("uint", "uint")])
def test_constrained_partial_struct_preserves_selected_field(
    declarator, pattern, scalar, field
):
    source = f"""
    template <typename T> constexpr constant bool accepts = is_same_v<T, float>;
    template <typename T, {declarator} = void> struct Cell {{ uint value; }};
    template <typename T> struct Cell<T, {pattern}> {{ float value; }};
    kernel void write_value(device Cell<{scalar}>* out [[buffer(0)]]) {{}}
    """
    output = materialize(source)
    name = f"Cell_{scalar}_void" if scalar == "float" else f"Cell_{scalar}"
    assert f"struct {name} {{ {field} value; }}" in output
    assert f"device {name}* out" in output


def test_anonymous_type_parameters_retain_positions_and_dependent_defaults():
    source = """
    template <typename, typename T, class = T> struct Cell { T value; };
    kernel void write_value(device Cell<uint, float>* out [[buffer(0)]]) {}
    """
    preprocessor = MetalPreprocessor()
    template = preprocessor._find_template_structs(source)[0]
    assert len(template.template_parameters) == 3
    assert template.template_parameter_defaults == {
        template.template_parameters[2]: "T"
    }
    assert preprocessor._template_arguments_with_resolved_defaults(
        template, ["uint", "float"]
    ) == ["uint", "float", "float"]
    output = materialize(source)
    assert "struct Cell_uint_float { float value; }" in output


def test_anonymous_parameter_identity_cannot_replace_source_identifiers():
    source = """
    template <typename T, typename = void> struct Cell { T crosstl_unnamed_type_1; };
    kernel void write_value(device Cell<float>* out [[buffer(0)]]) {}
    """
    assert "struct Cell_float { float crosstl_unnamed_type_1; }" in materialize(source)


@pytest.mark.parametrize("argument,field", [("float", "float"), ("uint", "uint")])
def test_constraint_result_type_must_match_explicit_argument(argument, field):
    source = f"""
    template <typename T, typename = void> struct Cell {{ uint value; }};
    template <typename T> struct Cell<T, enable_if_t<true, T>> {{ float value; }};
    kernel void write_value(device Cell<float, {argument}>* out [[buffer(0)]]) {{}}
    """
    assert f"struct Cell_float_{argument} {{ {field} value; }}" in materialize(source)


def test_constraint_uses_declaration_alias_not_call_site_shadow():
    source = """
    using Scalar = float;
    template <typename T, typename = void> struct Cell { uint value; };
    template <typename T> struct Cell<T, enable_if_t<is_same_v<T, Scalar>>> { float value; };
    kernel void write_value(device float* out [[buffer(0)]]) {
      using Scalar = uint;
      Cell<float> value;
      out[0] = value.value;
    }
    """
    assert "struct Cell_float_void { float value; }" in materialize(source)


def test_constraint_uses_named_predicate_namespace_and_late_specialization():
    source = """
    template <typename T> constexpr constant bool accepts = false;
    namespace traits {
      template <typename T> constexpr constant bool accepts = false;
      template <typename T, typename = void> struct Cell { uint value; };
      template <typename T> struct Cell<T, enable_if_t<accepts<T>>> { float value; };
      template <> constexpr constant bool accepts<float> = true;
    }
    kernel void write_value(device traits::Cell<float>* out [[buffer(0)]]) {}
    """
    assert "struct Cell_float_void { float value; }" in materialize(source)


def test_constrained_partial_struct_type_alias_is_resolved():
    source = """
    template <typename T, typename = void> struct Cell { using type = uint; };
    template <typename T> struct Cell<T, enable_if_t<is_same_v<T, float>>> { using type = float; };
    template <typename T> struct Buffer { T value; };
    kernel void write_value(device Buffer<Cell<float>::type>* out [[buffer(0)]]) {}
    """
    assert "struct Buffer_float { float value; }" in materialize(source)


@pytest.mark.parametrize("result", ["typename T::type", "Unknown<T>", "Unknown"])
def test_unresolved_constraint_result_cannot_discard_partial(result):
    source = f"""
    template <typename T, typename = void> struct Cell {{ uint value; }};
    template <typename T> struct Cell<T, enable_if_t<true, {result}>> {{ float value; }};
    kernel void write_value(device Cell<float>* out [[buffer(0)]]) {{}}
    """
    with pytest.raises(MetalTemplateSpecializationError, match="storage layout"):
        materialize(source)


def test_constraint_result_resolves_alias_and_standard_cv_removal():
    source = """
    using namespace metal;
    using Scalar = float;
    template <typename T, typename = Scalar> struct Cell { uint value; };
    template <typename T> struct Cell<T, enable_if_t<true, remove_cv_t<const T>>> { float value; };
    kernel void write_value(device Cell<float>* out [[buffer(0)]]) {}
    """
    assert "{ float value; }" in materialize(source).rsplit("struct Cell_", 1)[1]


def test_structural_mismatch_does_not_evaluate_unknown_constraint():
    source = """
    template <typename T, typename U, typename = void> struct Cell { uint value; };
    template <typename T> struct Cell<T, float, enable_if_t<missing<T>>> { float value; };
    kernel void write_value(device Cell<float, uint>* out [[buffer(0)]]) {}
    """
    assert "struct Cell_float_uint { uint value; }" in materialize(source)


def test_false_constraint_does_not_evaluate_ill_formed_result():
    source = """
    template <typename T, typename = void> struct Cell { uint value; };
    template <typename T> struct Cell<T, enable_if_t<false, typename T::type>> { float value; };
    kernel void write_value(device Cell<float>* out [[buffer(0)]]) {}
    """
    assert "struct Cell_float { uint value; }" in materialize(source)


@pytest.mark.parametrize("qualified_alias", [False, True])
def test_multiple_viable_partials_cannot_fall_back_to_primary(qualified_alias):
    source = """
    template <typename T, typename U, typename = void> struct Cell { uint value; using type = uint; };
    template <typename T, typename U> struct Cell<T, U, enable_if_t<true>> { int value; using type = int; };
    template <typename T> struct Cell<T, uint, void> { float value; using type = float; };
    template <typename T> struct Buffer { T value; };
    kernel void write_value(device TYPE* out [[buffer(0)]]) {}
    """.replace(
        "TYPE",
        "Buffer<Cell<float, uint>::type>" if qualified_alias else "Cell<float, uint>",
    )
    with pytest.raises(
        MetalTemplateSpecializationError, match="ordering is not supported"
    ):
        materialize(source)
