import re
import shutil
import subprocess

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


def test_stateful_global_constructor_reports_contract_error():
    source = """
    struct State {
      int value;
      constexpr State() constant : value(5) { value += 1; }
    };

    constant State state;
    kernel void compute(device int* output [[buffer(0)]]) {
      output[0] = state.value;
    }
    """

    _preprocessed, ast = preprocess_and_parse(source, "constructor-global.metal")
    with pytest.raises(MetalConstructorContractError) as raised:
        MetalToCrossGLConverter().generate(ast)

    error = raised.value
    assert error.owner == "State"
    assert "global object construction" in error.reason
    assert error.source_location["file"] == "constructor-global.metal"
    assert error.source_location["line"] == 7


def test_braced_return_and_assignment_use_explicit_constructor_contract():
    source = """
    struct Value {
      int value;
      Value(int input) : value(input) { value += 1; }
    };

    Value make_value() { return {2}; }

    kernel void compute(device int* output [[buffer(0)]]) {
      Value current(1);
      current = {3};
      output[0] = current.value;
    }
    """

    _preprocessed, generated = convert(source)

    assert "return crosstl_ctor_Value_1(2);" in generated
    assert "current = crosstl_ctor_Value_1(3);" in generated
    assert_crossgl_parses(generated)


def test_copy_initialization_uses_declared_copy_constructor():
    source = """
    struct Copyable {
      int value;
      Copyable(int input) : value(input) {}
      Copyable(const thread Copyable& other) : value(other.value + 1) {}
    };

    kernel void compute(device int* output [[buffer(0)]]) {
      Copyable source(1);
      Copyable copy = source;
      output[0] = copy.value;
    }
    """

    _preprocessed, generated = convert(source)

    assert "Copyable copy = crosstl_ctor_Copyable_2(source);" in generated
    assert "crosstl_ctor_value.value = int(other.value + 1);" in generated
    assert_crossgl_parses(generated)


def test_fully_initialized_constructor_array_lowers_each_element():
    source = """
    struct Value {
      int value;
      explicit Value(int input) : value(input) {}
    };

    kernel void compute(device int* output [[buffer(0)]]) {
      Value values[2] = {Value(1), Value(2)};
      output[0] = values[0].value + values[1].value;
    }
    """

    _preprocessed, generated = convert(source)

    assert re.search(
        r"Value\[2\] values;\s*"
        r"values\[0\] = crosstl_ctor_Value_1\(1\);\s*"
        r"values\[1\] = crosstl_ctor_Value_1\(2\);",
        generated,
    )
    assert "Value[2] values =" not in generated
    assert_crossgl_parses(generated)


def test_partially_initialized_constructor_array_default_constructs_tail():
    source = """
    struct Value {
      int value;
      Value(int input) : value(input) {}
      Value() : value(0) {}
    };

    kernel void compute(device int* output [[buffer(0)]]) {
      Value values[3] = {0};
      output[0] = values[0].value + values[1].value + values[2].value;
    }
    """

    _preprocessed, generated = convert(source)

    assert re.search(
        r"Value\[3\] values;\s*"
        r"values\[0\] = crosstl_ctor_Value_1\(0\);\s*"
        r"for \(int (?P<index>\w+) = 1; (?P=index) < 3; "
        r"(?P=index)\+\+\) \{\s*"
        r"values\[(?P=index)\] = crosstl_ctor_Value_2\(\);\s*\}",
        generated,
    )
    assert generated.count("for (int _crosstl_constructor_index") == 1
    assert "values[1] =" not in generated
    assert "values[2] =" not in generated
    assert_crossgl_parses(generated)


def test_uninitialized_constructor_array_default_constructs_each_element():
    source = """
    struct Value {
      int value;
      Value() : value(0) {}
      explicit Value(int input) : value(input) {}
    };

    kernel void compute(device int* output [[buffer(0)]]) {
      Value values[2];
      output[0] = values[0].value + values[1].value;
    }
    """

    _preprocessed, generated = convert(source)

    assert re.search(
        r"Value\[2\] values;\s*"
        r"for \(int (?P<index>\w+) = 0; (?P=index) < 2; "
        r"(?P=index)\+\+\) \{\s*"
        r"values\[(?P=index)\] = crosstl_ctor_Value_1\(\);\s*\}",
        generated,
    )
    assert generated.count("for (int _crosstl_constructor_index") == 1
    assert "values[0] =" not in generated
    assert "values[1] =" not in generated
    assert_crossgl_parses(generated)


def test_constructor_array_loop_index_avoids_source_identifier_collision():
    source = """
    struct Value {
      int value;
      Value() : value(0) {}
    };

    kernel void compute(device int* output [[buffer(0)]]) {
      int _crosstl_constructor_index = 7;
      Value values[2];
      output[0] = _crosstl_constructor_index + values[0].value;
    }
    """

    _preprocessed, generated = convert(source)

    assert "int _crosstl_constructor_index = 7;" in generated
    assert re.search(
        r"for \(int _crosstl_constructor_index_2 = 0; "
        r"_crosstl_constructor_index_2 < 2; "
        r"_crosstl_constructor_index_2\+\+\)",
        generated,
    )
    assert "values[_crosstl_constructor_index_2]" in generated
    assert_crossgl_parses(generated)


def test_constructor_array_loop_uses_evaluated_conditional_extent():
    source = """
    struct Value {
      int value;
      Value() : value(0) {}
    };

    kernel void compute(device int* output [[buffer(0)]]) {
      Value values[2 == 0 ? 1 : 2];
      output[0] = values[1].value;
    }
    """

    _preprocessed, generated = convert(source)

    assert "Value[2 == 0 ? 1 : 2] values;" in generated
    assert re.search(
        r"for \(int (?P<index>\w+) = 0; (?P=index) < 2; " r"(?P=index)\+\+\)",
        generated,
    )
    assert "< 2 == 0 ? 1 : 2" not in generated
    assert_crossgl_parses(generated)


def test_multidimensional_constructor_array_reports_contract_error():
    source = """
    struct Value {
      int value;
      explicit Value(int input) : value(input) {}
    };

    kernel void compute() {
      Value values[2][2] = {
        {Value(1), Value(2)},
        {Value(3), Value(4)}
      };
    }
    """

    _preprocessed, ast = preprocess_and_parse(
        source, "constructor-multidimensional-array.metal"
    )
    with pytest.raises(MetalConstructorContractError) as raised:
        MetalToCrossGLConverter().generate(ast)

    error = raised.value
    assert error.owner == "Value"
    assert error.reason == (
        "only one-dimensional local constructor arrays can be lowered"
    )
    assert error.source_location["file"] == ("constructor-multidimensional-array.metal")
    assert error.source_location["line"] == 8


def test_const_constructor_array_reports_source_located_contract_error():
    source = """
    struct Value {
      int value;
      Value(int input) : value(input) {}
    };

    kernel void compute() {
      const Value values[2] = {Value(1), Value(2)};
    }
    """

    _preprocessed, ast = preprocess_and_parse(
        source, "constructor-qualified-array.metal"
    )
    with pytest.raises(MetalConstructorContractError) as raised:
        MetalToCrossGLConverter().generate(ast)

    error = raised.value
    assert error.owner == "Value"
    assert error.reason == (
        "local constructor arrays with storage or immutability qualifier(s) const "
        "cannot be lowered through runtime element assignments"
    )
    assert error.source_location["file"] == "constructor-qualified-array.metal"
    assert error.source_location["line"] == 8
    assert error.source_location["column"] == 31


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


@pytest.mark.parametrize(
    ("backend", "extension", "compiler_name"),
    [
        pytest.param("directx", "hlsl", "dxc", id="directx"),
        pytest.param("opengl", "comp", "glslangValidator", id="opengl"),
    ],
)
def test_fixed_size_constructor_array_survives_crossgl_target_generation(
    tmp_path, backend, extension, compiler_name
):
    source = """
    struct Value {
      int value;
      Value(int input) : value(input) {}
      Value() : value(0) {}
    };

    kernel void compute(
        device int* output [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
      Value values[3] = {0};
      output[index] = values[0].value + values[1].value + values[2].value;
    }
    """

    _preprocessed, crossgl = convert(source, "constructor-array-target.metal")
    crossgl_path = tmp_path / "constructor-array.cgl"
    crossgl_path.write_text(crossgl, encoding="utf-8")

    generated = crosstl.translate(
        str(crossgl_path), backend=backend, format_output=False
    )

    assert "Value crosstl_ctor_Value_1(" in generated
    assert "Value crosstl_ctor_Value_2()" in generated
    assert "values[0] = crosstl_ctor_Value_1(0);" in generated
    assert re.search(
        r"for \(int (?P<index>_crosstl_constructor_index) = 1;[^{]+\{\s*"
        r"values\[(?P=index)\] = crosstl_ctor_Value_2\(\);",
        generated,
    )

    generated_path = tmp_path / f"constructor-array.{extension}"
    generated_path.write_text(generated, encoding="utf-8")
    compiler = shutil.which(compiler_name)
    if compiler is None:
        pytest.skip(f"{compiler_name} is not installed")

    binary_path = tmp_path / f"constructor-array.{backend}.bin"
    if backend == "directx":
        command = [
            compiler,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(generated_path),
            "-Fo",
            str(binary_path),
        ]
    else:
        command = [
            compiler,
            "--target-env",
            "opengl",
            "-S",
            "comp",
            str(generated_path),
            "-o",
            str(binary_path),
        ]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert binary_path.stat().st_size > 0
    if backend == "opengl":
        spirv_val = shutil.which("spirv-val")
        if spirv_val is not None:
            result = subprocess.run(
                [
                    spirv_val,
                    "--target-env",
                    "opengl4.5",
                    str(binary_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan"])
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

    if backend == "vulkan":
        assert "OpFunctionCall" in generated
        assert "OpBitwiseXor" in generated
        assert "crosstl_ctor_value" in generated
    else:
        assert "crosstl_ctor_fp8_e4m3_1_float" in generated
        assert "f_bits ^= sign;" in generated
        assert "crosstl_ctor_value.bits" in generated
