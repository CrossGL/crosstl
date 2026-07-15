import re

import pytest

import crosstl
from crosstl.backend.common_ast import AssignmentNode
from crosstl.backend.Metal.MetalAst import ConstructorNode
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalConstructorContractError,
    MetalToCrossGLConverter,
)
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser


def preprocess_and_parse(source, file_path="constructors.metal"):
    preprocessed = MetalPreprocessor().preprocess(source, file_path=file_path)
    tokens = MetalLexer(
        preprocessed,
        preprocess=False,
        file_path=file_path,
    ).tokenize()
    return preprocessed, MetalParser(tokens, file_path=file_path).parse()


def convert(source, file_path="constructors.metal"):
    preprocessed, ast = preprocess_and_parse(source, file_path)
    return preprocessed, MetalToCrossGLConverter().generate(ast)


def assert_crossgl_parses(source):
    CrossGLParser(CrossGLLexer(source).get_tokens()).parse()


def test_parser_retains_templated_constructor_contract_and_initializers():
    source = """
    struct Pair {
      int x = 4;
      int y;
      template <typename T>
      explicit Pair(T a, int b = 2) : x(a), y{b} { x ^= y; }
      Pair() = default;
    };
    """

    _preprocessed, ast = preprocess_and_parse(source, "contracts.metal")
    constructors = ast.structs[0].constructors

    assert len(constructors) == 2
    templated, defaulted = constructors
    assert isinstance(templated, ConstructorNode)
    assert templated.owner_name == "Pair"
    assert templated.template_parameters == [("typename", "T")]
    assert templated.qualifiers == ["explicit"]
    assert [parameter.vtype for parameter in templated.params] == ["T", "int"]
    assert templated.params[1].default_value == "2"
    assert [initializer.target for initializer in templated.initializers] == [
        "x",
        "y",
    ]
    assert [initializer.style for initializer in templated.initializers] == [
        "paren",
        "brace",
    ]
    assert isinstance(templated.body[0], AssignmentNode)
    assert templated.body[0].operator == "^="
    assert templated.source_location["file"] == "contracts.metal"
    assert templated.source_location["line"] == 6
    assert defaulted.declaration_kind == "default"
    assert defaulted.body is None


def test_preprocessor_keeps_member_template_constructor_in_its_owner():
    source = """
    struct fp8_like {
      uint bits;
      template <typename T>
      fp8_like(T f) {
        uint f_bits = as_type<uint>(f);
        f_bits ^= 1;
        bits = f_bits;
      }
    };
    uint encode(float f) { return fp8_like(f).bits; }
    """

    preprocessed, ast = preprocess_and_parse(source)

    assert "fp8_like(T f)" in preprocessed
    assert not re.search(r"(?m)^\s*fp8_like_[A-Za-z0-9_]+\s*\(", preprocessed)
    assert len(ast.structs[0].constructors) == 1
    body = ast.structs[0].constructors[0].body
    assert any(
        isinstance(statement, AssignmentNode) and statement.operator == "^="
        for statement in body
    )


def test_concrete_template_struct_renames_and_lowers_its_constructor():
    source = """
    template <typename T>
    struct Box {
      T value;
      Box(T input) : value(input) { value += T(1); }
    };

    kernel void compute(device float* output [[buffer(0)]]) {
      Box<float> box(2.0f);
      output[0] = box.value;
    }
    """

    preprocessed, generated = convert(source)

    assert "struct Box_float" in preprocessed
    assert "Box_float(float input)" in preprocessed
    assert "Box(float input)" not in preprocessed.split("struct Box_float", 1)[1]
    assert "Box_float crosstl_ctor_Box_float_1(float input)" in generated
    assert "crosstl_ctor_value.value = float(input);" in generated
    assert "crosstl_ctor_value.value += float(1);" in generated
    assert "Box_float box = crosstl_ctor_Box_float_1(2.0f);" in generated
    assert_crossgl_parses(generated)


def test_constructor_factories_preserve_argument_body_and_field_side_effects():
    source = """
    int observe(int value) { return value + 1; }

    struct Audit {
      Audit(int value) { observe(value); }
    };

    struct Pair {
      int first = 5;
      int second;
      Pair(int seed, int delta = 2) : second(seed) {
        first += delta;
        this->second ^= first;
      }
    };

    kernel void compute(device int* output [[buffer(0)]], int seed) {
      Audit audit(observe(seed));
      Pair pair(seed);
      output[0] = pair.first + pair.second;
    }
    """

    _preprocessed, generated = convert(source)

    assert re.search(
        r"Audit crosstl_ctor_Audit_1\(int value\).*?"
        r"observe\(value\);.*?return crosstl_ctor_value;",
        generated,
        re.DOTALL,
    )
    assert "Audit audit = crosstl_ctor_Audit_1(observe(seed));" in generated
    first_default = generated.index("crosstl_ctor_value.first = 5;")
    second_initializer = generated.index("crosstl_ctor_value.second = int(seed);")
    body_write = generated.index("crosstl_ctor_value.first += delta;")
    explicit_this_write = generated.index(
        "crosstl_ctor_value.second ^= crosstl_ctor_value.first;"
    )
    assert first_default < second_initializer < body_write < explicit_this_write
    assert "Pair pair = crosstl_ctor_Pair_1(seed, 2);" in generated
    assert_crossgl_parses(generated)


def test_default_constructor_runs_for_uninitialized_local():
    source = """
    struct State {
      int value = 5;
      State() { value += 1; }
    };

    kernel void compute(device int* output [[buffer(0)]]) {
      State state;
      output[0] = state.value;
    }
    """

    _preprocessed, generated = convert(source)

    assert "State state = crosstl_ctor_State_1();" in generated
    assert "crosstl_ctor_value.value = 5;" in generated
    assert "crosstl_ctor_value.value += 1;" in generated
    assert_crossgl_parses(generated)


def test_constructor_body_calls_lowered_sibling_method_with_owned_receiver():
    source = """
    struct Counter {
      int value;
      void assign(int input) { value = input; }
      Counter(int input) { assign(input); }
    };

    kernel void compute(device int* output [[buffer(0)]]) {
      Counter counter(7);
      output[0] = counter.value;
    }
    """

    preprocessed, generated = convert(source)

    assert "void Counter__assign(thread Counter& self, int input)" in preprocessed
    assert "Counter__assign(crosstl_ctor_value, input);" in generated
    assert "Counter counter = crosstl_ctor_Counter_1(7);" in generated
    assert_crossgl_parses(generated)


def test_overloaded_templated_and_address_space_constructors_remain_distinct():
    source = """
    struct Scalar {
      float value;
      Scalar(int input) : value(float(input)) {}
      Scalar(float input) : value(input) {}
      template <typename T>
      Scalar(T input) : value(float(input) + 1.0f) {}
    };

    struct Addressed {
      int value;
      template <typename T>
      Addressed(T input) thread : value(int(input)) {}
      template <typename T>
      Addressed(T input) threadgroup : value(int(input) + 1) {}
    };

    kernel void compute(device float* output [[buffer(0)]]) {
      Scalar integer(1);
      Scalar real(1.0f);
      Scalar narrow(short(2));
      Scalar copied(integer);
      Addressed local(3);
      threadgroup Addressed shared(4);
      output[0] = integer.value + real.value + narrow.value + copied.value +
          float(local.value + shared.value);
    }
    """

    _preprocessed, generated = convert(source)

    assert "Scalar integer = crosstl_ctor_Scalar_1(1);" in generated
    assert "Scalar real = crosstl_ctor_Scalar_2(1.0f);" in generated
    assert "Scalar narrow = crosstl_ctor_Scalar_3_short(int16(2));" in generated
    assert "Scalar copied = integer;" in generated
    assert "Addressed local = crosstl_ctor_Addressed_1_int(3);" in generated
    assert (
        "threadgroup Addressed shared = crosstl_ctor_Addressed_2_int(4);" in generated
    )
    assert_crossgl_parses(generated)


def test_unrepresentable_constructor_reports_source_located_contract_error():
    source = """
    struct Parent { int value; };
    struct Child : Parent {
      Child(int value) : Parent() {}
    };
    kernel void compute() { Child child(1); }
    """

    _preprocessed, ast = preprocess_and_parse(source, "unsupported.metal")
    with pytest.raises(MetalConstructorContractError) as raised:
        MetalToCrossGLConverter().generate(ast)

    error = raised.value
    assert error.project_diagnostic_code == (
        "project.translate.metal-constructor-unrepresentable"
    )
    assert error.missing_capabilities == ("metal.explicit-constructor-lowering",)
    assert error.owner == "Child"
    assert "base-class construction" in error.reason
    assert error.source_location["file"] == "unsupported.metal"
    assert error.source_location["line"] == 4
    assert error.source_location["column"] == 7


def test_pointer_reinterpret_cast_does_not_invoke_constructor():
    source = """
    struct fp8_like {
      template <typename T>
      fp8_like(T value) { bits = uint8_t(value); }
      uint8_t bits;
    };

    float decode(uint8_t value) {
      return float(*(thread fp8_like*)(&value));
    }
    """

    _preprocessed, generated = convert(source)

    assert "crosstl_ctor_fp8_like" not in generated
    assert "(*(fp8_like*)(&value))" in generated
    assert_crossgl_parses(generated)


def test_plain_aggregate_initialization_keeps_existing_lowering():
    source = """
    struct Plain {
      int count;
      float scale;
    };

    kernel void compute(device float* output [[buffer(0)]]) {
      Plain value(2, 3.0f);
      output[0] = float(value.count) * value.scale;
    }
    """

    _preprocessed, generated = convert(source)

    assert "crosstl_ctor_Plain" not in generated
    assert "Plain value = Plain(2, 3.0f);" in generated
    assert_crossgl_parses(generated)


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_fp8_constructor_boundary_translates_to_native_targets(tmp_path, backend):
    source = """
    struct fp8_e4m3 {
      template <typename T>
      fp8_e4m3(T value) {
        uint f_bits = as_type<uint>(float(value));
        uint sign = f_bits & 0x80000000;
        f_bits ^= sign;
        bits = uint8_t(f_bits >> 24);
      }
      uint8_t bits;
    };

    kernel void unary(
        device uint8_t* output [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
      fp8_e4m3 value(float(index));
      output[index] = value.bits;
    }
    """
    source_path = tmp_path / "unary.metal"
    source_path.write_text(source, encoding="utf-8")

    generated = crosstl.translate(
        str(source_path),
        backend=backend,
        format_output=False,
    )

    assert "crosstl_ctor_fp8_e4m3_1_float" in generated
    assert "f_bits ^= sign;" in generated
    assert "crosstl_ctor_value.bits" in generated
