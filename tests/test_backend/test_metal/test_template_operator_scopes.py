import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor


@pytest.mark.parametrize("operator", ["<", "<=", ">", ">=", "<<", ">>", "<<=", ">>="])
@pytest.mark.parametrize("spacing", ["", " ", "/* angle: < */", "// angle: <\n"])
def test_operator_template_body_keeps_dependent_struct_references(operator, spacing):
    declaration = (
        "template <typename T, T a, typename U, U b>\n"
        f"constexpr auto operator{spacing}{operator}(Value<T, a>, Value<U, b>) {{\n"
        f"    constexpr auto result = a {operator} b;\n"
        "    return Value<decltype(result), result>{};\n"
        "}"
    )
    source = (
        "template <typename T, T v> struct Value { static constexpr T value = v; };\n"
        + declaration
        + "\nValue<int, 3> concrete;\n"
        "template <typename T> T identity(T x) { return x; }\n"
    )
    preprocessor = MetalPreprocessor()

    spans = preprocessor._find_template_declaration_spans(source)

    assert len(spans) == 3
    assert source[slice(*spans[1])] == declaration
    assert source[slice(*spans[2])] == (
        "template <typename T> T identity(T x) { return x; }"
    )
    output = preprocessor._materialize_explicit_template_struct_instantiations(source)
    assert declaration in output
    assert "Value_int_3 concrete;" in output
    assert "struct Value_int_3" in output
    assert "Value_decltype_result_result" not in output


@pytest.mark.parametrize("operator", ["<", "<=", ">", ">=", "<<", ">>", "<<=", ">>="])
def test_operator_template_prototype_ends_before_next_definition(operator):
    declaration = (
        "template <typename T> " f"bool operator{operator}(Box<T> a, Box<T> b);"
    )
    next_declaration = "template <typename T> struct Box { T value; };"
    source = declaration + "\n" + next_declaration

    spans = MetalPreprocessor()._find_template_declaration_spans(source)

    assert source[slice(*spans[0])] == declaration
    assert len(spans) == 2
    assert source[slice(*spans[1])] == next_declaration[:-1]


@pytest.mark.parametrize(
    "source",
    [
        "void f(myoperator<decltype(Value<int>{})> value) { return; }",
        "Wrapper<decltype(Value<int>{})> operator<(Value<int> left) { return {}; }",
        'void f(const char* value = "operator<") { return; }',
        "void f() /* operator< */ { return; }",
        "void f() // operator<\n { return; }",
        "operator Wrapper<decltype(Value<int>{})>() { return {}; }",
    ],
)
def test_operator_name_scan_preserves_actual_template_arguments_and_literals(source):
    body = MetalPreprocessor()._find_next_top_level_char(source, 0, "{")

    assert body == source.index("{ return")


def test_unused_comparison_template_does_not_emit_dependent_specialization():
    source = """
    template <typename T, T v>
    struct Constant {
      static constexpr constant T value = v;
      using value_type = T;
      constexpr operator value_type() const noexcept { return value; }
    };
    template <typename T, T a, typename U, U b>
    constexpr auto operator<(Constant<T, a>, Constant<U, b>) {
      constexpr auto result = a < b;
      return Constant<decltype(result), result>{};
    }
    kernel void copy_values(device float* output [[buffer(0)]],
                            uint index [[thread_position_in_grid]]) {
      output[index] = 3.0f;
    }
    """

    output = MetalPreprocessor().preprocess(source)

    assert "Constant_decltype_result_result" not in output
    assert "output[index] = 3.0f;" in output
