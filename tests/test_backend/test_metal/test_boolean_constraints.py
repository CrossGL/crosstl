import pytest

from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalTemplateSpecializationError,
)


def constrained_source(predicate, *, condition="accepts<T>", namespace=""):
    functions = f"""
    template <typename T, enable_if_t<{condition}, bool> = true>
    T choose(T value) {{ return value + T(1); }}
    template <typename T, enable_if_t<!({condition}), bool> = true>
    T choose(T value) {{ return value + T(2); }}
    """
    if namespace:
        functions = f"namespace {namespace} {{ {functions} }}"
    return f"""#include <metal_stdlib>
    using namespace metal;
    {predicate}
    {functions}
    kernel void select_values(device uint* out [[buffer(0)]],
                              uint tid [[thread_position_in_grid]]) {{
        out[2 * tid] = {namespace + '::' if namespace else ''}choose(tid);
        out[2 * tid + 1] = uint({namespace + '::' if namespace else ''}choose(float(tid)));
    }}
    """


@pytest.mark.parametrize(
    "predicate,condition",
    [
        ("is_same_v<T, uint>", "accepts<T>"),
        ("is_same<T, uint>::value", "accepts<T>"),
        ("!is_same_v<T, float> && is_integral_v<T>", "accepts<T>"),
        ("_disjunction<is_same<T, uint>, is_same<T, int>>::value", "accepts<T>"),
        ("_conjunction<is_integral<T>, is_unsigned<T>>::value", "accepts<T>"),
        ("!_disjunction<is_same<T, float>, is_same<T, half>>::value", "accepts<T>"),
    ],
)
def test_named_boolean_constraint_selects_both_overloads(predicate, condition):
    source = constrained_source(
        f"template <typename T> constexpr constant bool accepts = {predicate};",
        condition=condition,
    )
    output = MetalPreprocessor().preprocess(source)
    assert "return value + uint(1);" in output
    assert "return value + float(2);" in output
    assert "return value + uint(2);" not in output
    assert "return value + float(1);" not in output


@pytest.mark.parametrize(
    "condition", ["accepts<T>", "traits::accepts<T>", "::traits::accepts<T>"]
)
def test_named_boolean_constraint_uses_declaration_namespace(condition):
    source = constrained_source(
        """
        template <typename T> constexpr constant bool accepts = false;
        namespace traits {
          template <typename T> constexpr constant bool accepts = is_same_v<T, uint>;
        }
        """,
        namespace="traits",
        condition=condition,
    )
    output = MetalPreprocessor().preprocess(source)
    assert "return value + uint(1);" in output
    assert "return value + float(2);" in output


def test_named_boolean_constraint_resolves_nested_predicate_in_its_own_namespace():
    source = constrained_source(
        """
        namespace traits {
          template <typename T> constexpr constant bool base = is_same_v<T, uint>;
          template <typename T> constexpr constant bool accepts = base<T>;
        }
        template <typename T> constexpr constant bool base = false;
        """,
        condition="traits::accepts<T>",
    )
    output = MetalPreprocessor().preprocess(source)
    assert "return value + uint(1);" in output
    assert "return value + float(2);" in output


@pytest.mark.parametrize("after_functions", [False, True])
def test_named_boolean_constraint_applies_explicit_specialization(after_functions):
    primary = "template <typename T> constexpr constant bool accepts = false;"
    specialization = "template <> constexpr constant bool accepts<uint> = true;"
    source = constrained_source(primary + ("" if after_functions else specialization))
    if after_functions:
        source = source.replace("kernel void", specialization + "\nkernel void")
    output = MetalPreprocessor().preprocess(source)
    assert "return value + uint(1);" in output
    assert "return value + float(2);" in output


@pytest.mark.parametrize(
    "predicate",
    [
        "template <typename T> constexpr constant bool accepts = missing<T>;",
        "template <typename T> constexpr constant bool accepts = accepts<T>;",
        "template <typename T> constexpr constant bool accepts = unknown(T(0));",
        "namespace hidden { template <typename T> constexpr constant bool accepts = true; }",
        """namespace first { template <typename T> constexpr constant bool accepts = true; }
        namespace second { template <typename T> constexpr constant bool accepts = false; }
        using namespace first; using namespace second;""",
    ],
)
def test_named_boolean_constraints_fail_closed(predicate):
    with pytest.raises(MetalTemplateSpecializationError, match="SFINAE constraint"):
        MetalPreprocessor().preprocess(constrained_source(predicate))


@pytest.mark.parametrize(
    "expression,expected",
    [("!false && false", False), ("!true || true", True), ("!(true || false)", False)],
)
def test_boolean_constraint_negation_precedence(expression, expected):
    assert MetalPreprocessor()._evaluate_boolean_constraint(expression, {}) is expected


def test_named_boolean_constraint_reuse_does_not_leak_previous_source():
    preprocessor = MetalPreprocessor()
    for predicate, increment in (("true", 1), ("false", 2)):
        source = constrained_source(
            f"template <typename T> constexpr constant bool accepts = {predicate};"
        )
        output = preprocessor.preprocess(source)
        assert f"return value + uint({increment});" in output
        assert f"return value + float({increment});" in output


@pytest.mark.parametrize("specialized_type", ["uint", "unsigned int", "Index"])
def test_named_boolean_specialization_preserves_type_aliases(specialized_type):
    source = constrained_source(
        "using Index = uint;\n"
        "template <typename T> constexpr constant bool accepts = false;\n"
        f"template <> constexpr constant bool accepts<{specialized_type}> = true;"
    )
    output = MetalPreprocessor().preprocess(source)
    assert "return value + uint(1);" in output
    assert "return value + float(2);" in output


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("(true)", True),
        ("!((false))", True),
        ("false && missing<uint>", False),
        ("true || missing<uint>", True),
        ("_conjunction<>::value", True),
        ("_disjunction<>::value", False),
        ("_disjunction<is_same<uint, uint>, missing<uint>>::value", True),
        ("_conjunction<is_same<uint, float>, missing<uint>>::value", False),
        (
            "_disjunction<_conjunction<is_integral<uint>, is_unsigned<uint>>, is_same<uint, float>>::value",
            True,
        ),
    ],
)
def test_boolean_constraint_composition(expression, expected):
    assert MetalPreprocessor()._evaluate_boolean_constraint(expression, {}) is expected


def test_boolean_variable_partial_specialization_preserves_fixed_arguments():
    source = """
    template <typename A, typename B> constexpr constant bool accepts = false;
    template <typename T> constexpr constant bool accepts<T, uint> = true;
    """
    preprocessor = MetalPreprocessor()
    definitions = preprocessor._find_boolean_variable_templates(source)
    for expression, expected in (
        ("accepts<float, uint>", True),
        ("accepts<float, int>", False),
    ):
        assert (
            preprocessor._evaluate_boolean_constraint(
                expression, {}, boolean_templates=definitions, position=len(source)
            )
            is expected
        )


def test_boolean_variable_ambiguous_partial_specializations_fail_closed():
    source = """
    template <typename A, typename B> constexpr constant bool accepts = false;
    template <typename T> constexpr constant bool accepts<T, uint> = true;
    template <typename T> constexpr constant bool accepts<uint, T> = false;
    """
    preprocessor = MetalPreprocessor()
    with pytest.raises(preprocessor._UnrecognizedConstraint):
        preprocessor._evaluate_boolean_constraint(
            "accepts<uint, uint>",
            {},
            boolean_templates=preprocessor._find_boolean_variable_templates(source),
            position=len(source),
        )


def test_boolean_variable_discovery_excludes_function_and_member_declarations():
    source = """
    template <typename T> constexpr bool predicate(T value = T(1)) { return false; }
    struct Traits {
      template <typename T> static constexpr constant bool predicate = true;
    };
    // template <typename T> constexpr constant bool predicate = true;
    """
    assert MetalPreprocessor()._find_boolean_variable_templates(source) == {}


def test_named_boolean_constraint_cannot_see_a_later_primary_declaration():
    source = constrained_source("").replace(
        "kernel void",
        "template <typename T> constexpr constant bool accepts = true;\nkernel void",
    )
    with pytest.raises(MetalTemplateSpecializationError, match="SFINAE constraint"):
        MetalPreprocessor().preprocess(source)


def test_named_boolean_predicate_resolves_alias_at_its_declaration():
    source = constrained_source(
        "using Scalar = uint;\n"
        "template <typename T> constexpr constant bool accepts = is_same_v<T, Scalar>;"
    ).replace("out[2 * tid]", "using Scalar = float;\n out[2 * tid]", 1)
    output = MetalPreprocessor().preprocess(source)
    assert "return value + uint(1);" in output
    assert "return value + float(2);" in output


@pytest.mark.parametrize(
    "pattern", ["enable_if_t<accepts<T>>", "enable_if_t<is_same_v<T, float>>"]
)
def test_unproven_constrained_struct_specialization_does_not_select_primary(pattern):
    source = f"""
    template <typename T> constexpr constant bool accepts = is_same_v<T, float>;
    template <typename T, typename = void> struct Cell {{ uint value; }};
    template <typename T> struct Cell<T, {pattern}> {{ float value; }};
    kernel void write_value(device Cell<float>* out [[buffer(0)]]) {{
      out[0].value = 1.5f;
    }}
    """
    with pytest.raises(
        MetalTemplateSpecializationError, match="field types or storage layout"
    ) as error:
        MetalPreprocessor().preprocess(source)
    assert error.value.callee_template == "Cell"
    assert error.value.requested_arguments == ("float",)
    assert error.value.suggested_action


def test_free_operator_uses_predicate_specialization_available_at_instantiation():
    source = """
    template <typename T> struct Box { T value; };
    template <typename T> constexpr constant bool accepts = false;
    template <typename T, typename U, enable_if_t<accepts<U>, bool> = true>
    constexpr Box<T> operator+(U scalar, Box<T> value) {
      return {static_cast<T>(scalar) + value.value};
    }
    template <> constexpr constant bool accepts<float> = true;
    kernel void add_scalar(device float* out [[buffer(0)]], float scalar) {
      Box<float> value{1.0f};
      auto result = scalar + value;
      out[0] = result.value;
    }
    """
    output = MetalPreprocessor().preprocess(source)
    assert "crosstl_metal_operator_add__float__Box_float" in output
    assert "return {static_cast<float>(scalar) + value.value};" in output
