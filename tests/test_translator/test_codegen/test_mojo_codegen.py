import shutil
import subprocess
from typing import List

import pytest
import crosstl.translator
from crosstl.translator.parser import Parser
from crosstl.translator.lexer import Lexer
from crosstl.translator.ast import LiteralNode, PrimitiveType
from crosstl.translator.codegen.mojo_codegen import MojoCodeGen


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = Lexer(code)
    return lexer.get_tokens()


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST.

    Args:
        tokens (List): The list of tokens to parse
    Returns:
        AST: The abstract syntax tree generated from the parser
    """
    parser = Parser(tokens)
    return parser.parse()


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = MojoCodeGen()
    return codegen.generate(ast_node)


def test_struct():
    code = """
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "struct VSInput:" in generated_code
        assert "struct VSOutput:" in generated_code
        assert "var texCoord: SIMD[DType.float32, 2]" in generated_code
        assert "var color: SIMD[DType.float32, 4]" in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Mojo struct codegen not implemented.")


def test_basic_shader():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert generated_code is not None
        assert "fn main(" in generated_code
        assert "@vertex_shader" in generated_code
        assert "@fragment_shader" in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Mojo basic shader codegen not implemented.")


def test_if_statement():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                if (input.texCoord.x > 0.5) {
                    output.color = vec4(1.0, 1.0, 1.0, 1.0);
                } else {
                    output.color = vec4(0.0, 0.0, 0.0, 1.0);
                }
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float brightness = texture(iChannel0, input.color.xy).r;
                float bloom = max(0.0, brightness - 0.5);
                if (bloom > 0.5) {
                    bloom = 0.5;
                } else {
                    bloom = 0.0;
                }
                vec3 texColor = texture(iChannel0, input.color.xy).rgb;
                vec3 colorWithBloom = texColor + vec3(bloom);
                return vec4(colorWithBloom, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "if " in generated_code
        assert "else:" in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("If statement codegen not implemented.")


def test_for_statement():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                for (int i = 0; i < 10; i++) {
                    output.color = vec4(1.0, 1.0, 1.0, 1.0);
                }
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "while " in generated_code  # Mojo uses while for C-style for loops
        print(generated_code)
    except SyntaxError:
        pytest.fail("For statement codegen not implemented.")


def test_increment_and_decrement_emit_mojo_assignment_updates():
    code = """
    shader main {
        compute {
            void main() {
                int i = 0;
                i++;
                ++i;
                i--;
                --i;
                for (int j = 0; j < 2; j++) {
                    i++;
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "i += 1" in generated_code
    assert "i -= 1" in generated_code
    assert "j += 1" in generated_code
    assert "++i" not in generated_code
    assert "i++" not in generated_code
    assert "--i" not in generated_code
    assert "i--" not in generated_code
    assert "++j" not in generated_code


def test_bool_string_and_char_literals_emit_mojo_syntax():
    code = """
    shader main {
        compute {
            void main() {
                bool enabled = true;
                bool disabled = false;
                string label = "debug";
                char marker = 'x';
                if (enabled && !disabled) {
                    label = "active";
                    marker = 'y';
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    entry_point = next(iter(ast.stages.values())).entry_point

    assert entry_point.body.statements[0].initial_value.value is True
    assert entry_point.body.statements[1].initial_value.value is False

    generated_code = generate_code(ast)

    assert "var enabled: Bool = True" in generated_code
    assert "var disabled: Bool = False" in generated_code
    assert 'var label: String = "debug"' in generated_code
    assert 'var marker: String = "x"' in generated_code
    assert 'label = "active"' in generated_code
    assert 'marker = "y"' in generated_code


def test_direct_literal_nodes_emit_mojo_escaping():
    codegen = MojoCodeGen()

    assert (
        codegen.generate_expression(LiteralNode(True, PrimitiveType("bool"))) == "True"
    )
    assert (
        codegen.generate_expression(LiteralNode('debug"name', PrimitiveType("string")))
        == '"debug\\"name"'
    )


def test_else_if_statement():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                if (input.texCoord.x > 0.5) {
                    output.color = vec4(1.0, 1.0, 1.0, 1.0);
                } else if (input.texCoord.x < 0.5) {
                    output.color = vec4(0.0, 0.0, 0.0, 1.0);
                } else {
                    output.color = vec4(0.5, 0.5, 0.5, 1.0);
                }
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float brightness = texture(iChannel0, input.color.xy).r;
                float bloom = max(0.0, brightness - 0.5);
                if (bloom > 0.5) {
                    bloom = 0.5;
                } else if (bloom < 0.5) {
                    bloom = 0.0;
                } else {
                    bloom = 0.5;
                }
                vec3 texColor = texture(iChannel0, input.color.xy).rgb;
                vec3 colorWithBloom = texColor + vec3(bloom);
                return vec4(colorWithBloom, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "elif " in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Else if codegen not implemented.")


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    float result = add(1.0, 2.0);
                }
                
                float add(float a, float b) {
                    return a + b;
                }
            }
            """,
            "add(1.0, 2.0)",
        )
    ],
)
def test_function_call(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = MojoCodeGen()
    generated_code = code_gen.generate(ast)
    assert expected_output in generated_code


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    a |= 2;
                }
            }
            """,
            "a |= 2",
        )
    ],
)
def test_assignment_or_operator(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = MojoCodeGen()
    generated_code = code_gen.generate(ast)
    assert expected_output in generated_code


def test_assignment_modulus_operator():
    code = """
    shader main {
        vertex {
            void main() {
                int a = 10;
                a %= 3;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "a %= 3" in generated_code or "a = a % 3" in generated_code
    except SyntaxError:
        pytest.fail("Assignment modulus operator codegen not implemented.")


def test_assignment_xor_operator():
    code = """
    shader main {
        vertex {
            void main() {
                int a = 5;
                a ^= 3;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "a ^= 3" in generated_code or "a = a ^ 3" in generated_code
    except SyntaxError:
        pytest.fail("Assignment XOR operator codegen not implemented.")


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    a <<= 2;
                    a >>= 1;
                }
            }
            """,
            ["a <<= 2", "a >>= 1"],
        )
    ],
)
def test_assignment_shift_operators(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = MojoCodeGen()
    generated_code = code_gen.generate(ast)
    for output in expected_output:
        assert output in generated_code


@pytest.mark.parametrize(
    "shader, expected_outputs",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    int b = 2;
                    int c = a | b;
                    int d = a & b;
                    int e = a ^ b;
                }
            }
            """,
            ["a | b", "a & b", "a ^ b"],
        )
    ],
)
def test_bitwise_operators(shader, expected_outputs):
    ast = crosstl.translator.parse(shader)
    code_gen = MojoCodeGen()
    generated_code = code_gen.generate(ast)
    for expected in expected_outputs:
        assert expected in generated_code


def test_bitwise_and_operator():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(float(int(input.texCoord.x * 100.0) & 15), 
                                    float(int(input.texCoord.y * 100.0) & 15), 
                                    0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.color.rgb, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise AND codegen not implemented")


def test_double_data_type():
    code = """
    shader DoubleShader {
        struct VSInput {
            double texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            double color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = input.texCoord * 2.0;
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.color, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "Float64" in generated_code  # Mojo uses Float64 for double
        print(generated_code)
    except SyntaxError:
        pytest.fail("Double data type not supported.")


@pytest.mark.parametrize(
    "shader, expected_outputs",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    int b = 2;
                    int c = a << b;
                    int d = a >> b;
                }
            }
            """,
            ["a << b", "a >> b"],
        )
    ],
)
def test_shift_operators(shader, expected_outputs):
    ast = crosstl.translator.parse(shader)
    code_gen = MojoCodeGen()
    generated_code = code_gen.generate(ast)
    for expected in expected_outputs:
        assert expected in generated_code


def test_bitwise_or_operator():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        sampler2D iChannel0;
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(float(int(input.texCoord.x * 100.0) | 15), 
                                    float(int(input.texCoord.y * 100.0) | 15), 
                                    0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.color.rgb, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise OR codegen not implemented")


def test_ternary_operator():
    code = """
    shader main {
        vertex {
            vec4 main() {
                float x = 0.5;
                float result = x > 0.0 ? 1.0 : 0.0;
                return vec4(result, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert (
            "if" in generated_code and "else" in generated_code
        )  # Mojo ternary syntax
        print(generated_code)
    except SyntaxError:
        pytest.fail("Ternary operator codegen not implemented")


def test_vector_constructor():
    code = """
    shader main {
        vertex {
            vec4 main() {
                vec2 uv = vec2(0.5, 0.5);
                vec4 color = vec4(uv, 0.0, 1.0);
                return color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "SIMD[DType.float32, 2]" in generated_code
        assert "SIMD[DType.float32, 4]" in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Vector constructor codegen not implemented")


def test_double_vector_and_matrix_types_emit_mojo_names():
    code = """
    shader main {
        compute {
            void main() {
                dvec2 preciseUV = dvec2(1.0, 2.0);
                bvec2 mask = bvec2(true, false);
                bvec3 flags;
                mat2 transform = mat2(1.0, 0.0, 0.0, 1.0);
                mat3x4 affine;
                dmat2 precise = dmat2(1.0, 0.0, 0.0, 1.0);
                dmat4x3 jacobian;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "var preciseUV: SIMD[DType.float64, 2] = " "SIMD[DType.float64, 2](1.0, 2.0)"
    ) in generated_code
    assert (
        "var mask: SIMD[DType.bool, 2] = " "SIMD[DType.bool, 2](True, False)"
    ) in generated_code
    assert "var flags: SIMD[DType.bool, 4]" in generated_code
    assert (
        "var transform: Matrix[DType.float32, 2, 2] = "
        "Matrix[DType.float32, 2, 2](1.0, 0.0, 0.0, 1.0)"
    ) in generated_code
    assert "var affine: Matrix[DType.float32, 3, 4]" in generated_code
    assert (
        "var precise: Matrix[DType.float64, 2, 2] = "
        "Matrix[DType.float64, 2, 2](1.0, 0.0, 0.0, 1.0)"
    ) in generated_code
    assert "var jacobian: Matrix[DType.float64, 4, 3]" in generated_code
    assert "dvec2(" not in generated_code
    assert "bvec2(" not in generated_code
    assert "bool2" not in generated_code
    assert "dmat2(" not in generated_code
    assert "MatrixType(" not in generated_code


def test_generic_vector_constructors_emit_mojo_names():
    code = """
    shader main {
        compute {
            void main() {
                vec2<f64> precise = vec2<f64>(1.0, 2.0);
                vec3<i32> index = vec3<i32>(1, 2, 3);
                vec4<u32> mask = vec4<u32>(1, 2, 3, 4);
                vec2<bool> flags = vec2<bool>(true, false);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "var precise: SIMD[DType.float64, 2] = " "SIMD[DType.float64, 2](1.0, 2.0)"
    ) in generated_code
    assert (
        "var index: SIMD[DType.int32, 4] = " "SIMD[DType.int32, 4](1, 2, 3, 0)"
    ) in generated_code
    assert (
        "var mask: SIMD[DType.uint32, 4] = " "SIMD[DType.uint32, 4](1, 2, 3, 4)"
    ) in generated_code
    assert (
        "var flags: SIMD[DType.bool, 2] = " "SIMD[DType.bool, 2](True, False)"
    ) in generated_code
    assert "vec2<" not in generated_code
    assert "vec3<" not in generated_code
    assert "vec4<" not in generated_code


def test_three_component_vectors_emit_power_of_two_simd_storage():
    code = """
    vec3 buildColor() {
        return vec3(1.0, 2.0, 3.0);
    }

    dvec3 buildPrecise() {
        return dvec3(1.0, 2.0, 3.0);
    }

    ivec3 buildIndex() {
        return ivec3(1, 2, 3);
    }

    uvec3 buildMask() {
        return uvec3(1, 2, 3);
    }

    bvec3 buildFlags() {
        return bvec3(true, false, true);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn buildColor() -> SIMD[DType.float32, 4]:" in generated_code
    assert (
        "return SIMD[DType.float32, 4](1.0, 2.0, 3.0, 0.0)" in generated_code
    )
    assert "fn buildPrecise() -> SIMD[DType.float64, 4]:" in generated_code
    assert (
        "return SIMD[DType.float64, 4](1.0, 2.0, 3.0, 0.0)" in generated_code
    )
    assert "fn buildIndex() -> SIMD[DType.int32, 4]:" in generated_code
    assert "return SIMD[DType.int32, 4](1, 2, 3, 0)" in generated_code
    assert "fn buildMask() -> SIMD[DType.uint32, 4]:" in generated_code
    assert "return SIMD[DType.uint32, 4](1, 2, 3, 0)" in generated_code
    assert "fn buildFlags() -> SIMD[DType.bool, 4]:" in generated_code
    assert (
        "return SIMD[DType.bool, 4](True, False, True, False)" in generated_code
    )
    assert ", 3]" not in generated_code


def test_three_component_vector_codegen_compiles_with_mojo(tmp_path):
    mojo = shutil.which("mojo")
    if mojo is None:
        pytest.skip("mojo compiler is not installed")

    code = """
    vec3 buildColor() {
        return vec3(1.0, 2.0, 3.0);
    }

    ivec3 buildIndex() {
        return ivec3(1, 2, 3);
    }

    bvec3 buildFlags() {
        return bvec3(true, false, true);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    var color = buildColor()
    var index = buildIndex()
    var flags = buildFlags()
    print(color)
    print(index)
    print(flags)
"""

    source_path = tmp_path / "three_component_vectors.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[1.0, 2.0, 3.0, 0.0]" in result.stdout
    assert "[1, 2, 3, 0]" in result.stdout
    assert "[True, False, True, False]" in result.stdout


def test_swizzles_and_composite_vector_constructors_emit_indexed_lanes():
    code = """
    struct Input {
        vec4 color;
        vec2 uv;
    };

    vec4 widen(vec2 uv, vec4 color) {
        return vec4(uv, color.z, 1.0);
    }

    vec4 convert(Input input) {
        return vec4(input.color.rgb, 1.0);
    }

    vec2 narrow(Input input) {
        return input.color.xy;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "return SIMD[DType.float32, 4](uv[0], uv[1], color[2], 1.0)"
        in generated_code
    )
    assert (
        "return SIMD[DType.float32, 4](input.color[0], input.color[1], "
        "input.color[2], 1.0)"
    ) in generated_code
    assert (
        "return SIMD[DType.float32, 2](input.color[0], input.color[1])"
        in generated_code
    )
    assert ".rgb" not in generated_code
    assert ".xy" not in generated_code
    assert ".z" not in generated_code


def test_swizzles_and_composite_vector_constructors_compile_with_mojo(tmp_path):
    mojo = shutil.which("mojo")
    if mojo is None:
        pytest.skip("mojo compiler is not installed")

    code = """
    struct Input {
        vec4 color;
        vec2 uv;
    };

    vec4 widen(vec2 uv, vec4 color) {
        return vec4(uv, color.z, 1.0);
    }

    vec4 convert(Input input) {
        return vec4(input.color.rgb, 1.0);
    }

    vec2 narrow(Input input) {
        return input.color.xy;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "swizzles_and_composites.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_scalar_three_component_constructors_pad_hidden_lane():
    code = """
    vec3 tint(float bloom) {
        return vec3(bloom);
    }

    dvec3 precise(double value) {
        return dvec3(value);
    }

    ivec3 index(int value) {
        return ivec3(value);
    }

    uvec3 mask(uint value) {
        return uvec3(value);
    }

    bvec3 flags(bool enabled) {
        return bvec3(enabled);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "return SIMD[DType.float32, 4](bloom, bloom, bloom, 0.0)"
        in generated_code
    )
    assert (
        "return SIMD[DType.float64, 4](value, value, value, 0.0)"
        in generated_code
    )
    assert "return SIMD[DType.int32, 4](value, value, value, 0)" in generated_code
    assert "return SIMD[DType.uint32, 4](value, value, value, 0)" in generated_code
    assert (
        "return SIMD[DType.bool, 4](enabled, enabled, enabled, False)"
        in generated_code
    )


def test_later_function_call_swizzles_use_precollected_return_types():
    code = """
    ivec2 useLaterIndex() {
        return laterIndex().xy;
    }

    dvec2 useLaterPrecise() {
        return laterPrecise().xy;
    }

    ivec3 laterIndex() {
        return ivec3(1, 2, 3);
    }

    dvec3 laterPrecise() {
        return dvec3(1.0, 2.0, 3.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn _crossgl_swizzle_i32_4_xy" in generated_code
    assert "fn _crossgl_swizzle_f64_4_xy" in generated_code
    assert (
        "return _crossgl_swizzle_i32_4_xy(laterIndex())"
        in generated_code
    )
    assert (
        "return _crossgl_swizzle_f64_4_xy(laterPrecise())"
        in generated_code
    )
    assert "laterIndex()[0]" not in generated_code
    assert "laterPrecise()[0]" not in generated_code
    assert "SIMD[DType.float32, 2](laterIndex()" not in generated_code
    assert "SIMD[DType.float32, 2](laterPrecise()" not in generated_code


def test_duplicate_sensitive_splats_and_swizzles_use_helpers():
    code = """
    float expensive() {
        return 0.25;
    }

    vec3 tint() {
        return vec3(expensive());
    }

    ivec2 indexPair() {
        return laterIndex().xy;
    }

    dvec3 preciseRgb() {
        return laterPrecise().rgb;
    }

    ivec3 laterIndex() {
        return ivec3(1, 2, 3);
    }

    dvec3 laterPrecise() {
        return dvec3(1.0, 2.0, 3.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn _crossgl_vec3_splat_f32" in generated_code
    assert "fn _crossgl_swizzle_i32_4_xy" in generated_code
    assert "fn _crossgl_swizzle_f64_4_rgb" in generated_code
    assert "return _crossgl_vec3_splat_f32(expensive())" in generated_code
    assert "return _crossgl_swizzle_i32_4_xy(laterIndex())" in generated_code
    assert "return _crossgl_swizzle_f64_4_rgb(laterPrecise())" in generated_code
    assert "SIMD[DType.float32, 4](expensive(), expensive()" not in generated_code
    assert "laterIndex()[0]" not in generated_code
    assert "laterPrecise()[0]" not in generated_code


def test_scalar_vec3_splats_and_later_function_swizzles_compile_with_mojo(tmp_path):
    mojo = shutil.which("mojo")
    if mojo is None:
        pytest.skip("mojo compiler is not installed")

    code = """
    float expensive() {
        return 0.25;
    }

    vec3 tint() {
        return vec3(expensive());
    }

    ivec2 useLaterIndex() {
        return laterIndex().xy;
    }

    dvec2 useLaterPrecise() {
        return laterPrecise().xy;
    }

    ivec3 laterIndex() {
        return ivec3(1, 2, 3);
    }

    dvec3 laterPrecise() {
        return dvec3(1.0, 2.0, 3.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(tint())
    print(useLaterIndex())
    print(useLaterPrecise())
"""

    source_path = tmp_path / "scalar_splats_and_later_swizzles.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[0.25, 0.25, 0.25, 0.0]" in result.stdout
    assert "[1, 2]" in result.stdout
    assert "[1.0, 2.0]" in result.stdout


def test_duplicate_sensitive_composite_constructors_use_helpers():
    code = """
    vec2 makeUv() {
        return vec2(0.25, 0.75);
    }

    vec3 makeColor() {
        return vec3(1.0, 2.0, 3.0);
    }

    vec4 fromUv(float z, float w) {
        return vec4(makeUv(), z, w);
    }

    vec4 fromRgb(float alpha) {
        return vec4(makeColor().rgb, alpha);
    }

    vec3 narrowColor() {
        return vec3(makeColor().rgb);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn _crossgl_construct_f32_4_vf322_01_s_s" in generated_code
    assert "fn _crossgl_construct_f32_4_vf324_012_s" in generated_code
    assert "fn _crossgl_construct_f32_4_vf324_012" in generated_code
    assert (
        "return _crossgl_construct_f32_4_vf322_01_s_s(makeUv(), z, w)"
        in generated_code
    )
    assert (
        "return _crossgl_construct_f32_4_vf324_012_s(makeColor(), alpha)"
        in generated_code
    )
    assert "return _crossgl_construct_f32_4_vf324_012(makeColor())" in generated_code
    assert "makeUv()[0]" not in generated_code
    assert "makeColor()[0]" not in generated_code


def test_duplicate_sensitive_composite_constructors_compile_with_mojo(tmp_path):
    mojo = shutil.which("mojo")
    if mojo is None:
        pytest.skip("mojo compiler is not installed")

    code = """
    vec2 makeUv() {
        return vec2(0.25, 0.75);
    }

    vec3 makeColor() {
        return vec3(1.0, 2.0, 3.0);
    }

    vec4 fromUv(float z, float w) {
        return vec4(makeUv(), z, w);
    }

    vec4 fromRgb(float alpha) {
        return vec4(makeColor().rgb, alpha);
    }

    vec3 narrowColor() {
        return vec3(makeColor().rgb);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(fromUv(0.5, 1.0))
    print(fromRgb(1.0))
    print(narrowColor())
"""

    source_path = tmp_path / "duplicate_sensitive_composite_constructors.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[0.25, 0.75, 0.5, 1.0]" in result.stdout
    assert "[1.0, 2.0, 3.0, 1.0]" in result.stdout
    assert "[1.0, 2.0, 3.0, 0.0]" in result.stdout


def test_duplicate_sensitive_bool_vector_helpers_use_bool_dtype():
    code = """
    bool flag() {
        return true;
    }

    bvec3 makeFlags() {
        return bvec3(true, false, true);
    }

    bvec3 boolSplat() {
        return bvec3(flag());
    }

    bvec2 flagPair() {
        return makeFlags().xy;
    }

    bvec4 packFlags(bool tail) {
        return bvec4(makeFlags().rgb, tail);
    }

    vec3<bool> genericBoolSplat() {
        return vec3<bool>(flag());
    }

    vec4<bool> genericPack(bool tail) {
        return vec4<bool>(makeFlags().rgb, tail);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn _crossgl_vec3_splat_bool(s: Bool)" in generated_code
    assert "fn _crossgl_swizzle_bool_4_xy" in generated_code
    assert "fn _crossgl_construct_bool_4_vbool4_012_s" in generated_code
    assert "return _crossgl_vec3_splat_bool(flag())" in generated_code
    assert "return _crossgl_swizzle_bool_4_xy(makeFlags())" in generated_code
    assert (
        "return _crossgl_construct_bool_4_vbool4_012_s(makeFlags(), tail)"
        in generated_code
    )
    assert "SIMD[DType.float32, 2](makeFlags()" not in generated_code
    assert "makeFlags()[0]" not in generated_code
    assert "flag(), flag(), flag()" not in generated_code


def test_duplicate_sensitive_bool_vector_helpers_compile_with_mojo(tmp_path):
    mojo = shutil.which("mojo")
    if mojo is None:
        pytest.skip("mojo compiler is not installed")

    code = """
    bool flag() {
        return true;
    }

    bvec3 makeFlags() {
        return bvec3(true, false, true);
    }

    bvec3 boolSplat() {
        return bvec3(flag());
    }

    bvec2 flagPair() {
        return makeFlags().xy;
    }

    bvec4 packFlags(bool tail) {
        return bvec4(makeFlags().rgb, tail);
    }

    vec3<bool> genericBoolSplat() {
        return vec3<bool>(flag());
    }

    vec4<bool> genericPack(bool tail) {
        return vec4<bool>(makeFlags().rgb, tail);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(boolSplat())
    print(flagPair())
    print(packFlags(False))
    print(genericBoolSplat())
    print(genericPack(True))
"""

    source_path = tmp_path / "duplicate_sensitive_bool_vector_helpers.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[True, True, True, False]" in result.stdout
    assert "[True, False]" in result.stdout
    assert "[True, False, True, False]" in result.stdout
    assert "[True, False, True, True]" in result.stdout


def test_duplicate_sensitive_integer_vector_helpers_use_typed_dtypes():
    code = """
    int pickIndex() {
        return 7;
    }

    uint pickMask() {
        return 9;
    }

    ivec3 makeIndex() {
        return ivec3(1, 2, 3);
    }

    uvec3 makeMask() {
        return uvec3(4, 5, 6);
    }

    ivec3 indexSplat() {
        return ivec3(pickIndex());
    }

    uvec3 maskSplat() {
        return uvec3(pickMask());
    }

    ivec2 indexPair() {
        return makeIndex().xy;
    }

    uvec2 maskPair() {
        return makeMask().xy;
    }

    ivec4 packIndex(int tail) {
        return ivec4(makeIndex().rgb, tail);
    }

    uvec4 packMask(uint tail) {
        return uvec4(makeMask().rgb, tail);
    }

    vec4<i32> genericIndex(int tail) {
        return vec4<i32>(makeIndex().rgb, tail);
    }

    vec4<u32> genericMask(uint tail) {
        return vec4<u32>(makeMask().rgb, tail);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn _crossgl_vec3_splat_i32(s: Int32)" in generated_code
    assert "fn _crossgl_vec3_splat_u32(s: UInt32)" in generated_code
    assert "fn _crossgl_swizzle_i32_4_xy" in generated_code
    assert "fn _crossgl_swizzle_u32_4_xy" in generated_code
    assert "fn _crossgl_construct_i32_4_vi324_012_s" in generated_code
    assert "fn _crossgl_construct_u32_4_vu324_012_s" in generated_code
    assert "return _crossgl_vec3_splat_i32(pickIndex())" in generated_code
    assert "return _crossgl_vec3_splat_u32(pickMask())" in generated_code
    assert "return _crossgl_swizzle_i32_4_xy(makeIndex())" in generated_code
    assert "return _crossgl_swizzle_u32_4_xy(makeMask())" in generated_code
    assert (
        "return _crossgl_construct_i32_4_vi324_012_s(makeIndex(), tail)"
        in generated_code
    )
    assert (
        "return _crossgl_construct_u32_4_vu324_012_s(makeMask(), tail)"
        in generated_code
    )
    assert "SIMD[DType.float32, 2](makeIndex()" not in generated_code
    assert "SIMD[DType.float32, 2](makeMask()" not in generated_code
    assert "pickIndex(), pickIndex(), pickIndex()" not in generated_code
    assert "pickMask(), pickMask(), pickMask()" not in generated_code


def test_duplicate_sensitive_integer_vector_helpers_compile_with_mojo(tmp_path):
    mojo = shutil.which("mojo")
    if mojo is None:
        pytest.skip("mojo compiler is not installed")

    code = """
    int pickIndex() {
        return 7;
    }

    uint pickMask() {
        return 9;
    }

    ivec3 makeIndex() {
        return ivec3(1, 2, 3);
    }

    uvec3 makeMask() {
        return uvec3(4, 5, 6);
    }

    ivec3 indexSplat() {
        return ivec3(pickIndex());
    }

    uvec3 maskSplat() {
        return uvec3(pickMask());
    }

    ivec2 indexPair() {
        return makeIndex().xy;
    }

    uvec2 maskPair() {
        return makeMask().xy;
    }

    ivec4 packIndex(int tail) {
        return ivec4(makeIndex().rgb, tail);
    }

    uvec4 packMask(uint tail) {
        return uvec4(makeMask().rgb, tail);
    }

    vec4<i32> genericIndex(int tail) {
        return vec4<i32>(makeIndex().rgb, tail);
    }

    vec4<u32> genericMask(uint tail) {
        return vec4<u32>(makeMask().rgb, tail);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(indexSplat())
    print(maskSplat())
    print(indexPair())
    print(maskPair())
    print(packIndex(8))
    print(packMask(10))
    print(genericIndex(11))
    print(genericMask(12))
"""

    source_path = tmp_path / "duplicate_sensitive_integer_vector_helpers.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[7, 7, 7, 0]" in result.stdout
    assert "[9, 9, 9, 0]" in result.stdout
    assert "[1, 2]" in result.stdout
    assert "[4, 5]" in result.stdout
    assert "[1, 2, 3, 8]" in result.stdout
    assert "[4, 5, 6, 10]" in result.stdout
    assert "[1, 2, 3, 11]" in result.stdout
    assert "[4, 5, 6, 12]" in result.stdout


def test_vec3_arithmetic_helpers_preserve_hidden_lane():
    code = """
    vec3 addScalar(vec3 color, float bloom) {
        return color + bloom;
    }

    vec3 addSplat(vec3 color, float bloom) {
        return color + vec3(bloom);
    }

    vec3 scalarSubtract(vec3 color, float base) {
        return base - color;
    }

    vec3 divideScalar(vec3 color, float exposure) {
        return color / exposure;
    }

    vec4 pack(vec3 color, float bloom, float base) {
        vec3 mixed = (color + bloom) + (base - color);
        return vec4(mixed, 1.0);
    }

    ivec3 offsetIndex(ivec3 index, int offset) {
        return index + offset;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn _crossgl_vec3_add_f32_vs" in generated_code
    assert "fn _crossgl_vec3_add_f32_vv" in generated_code
    assert "fn _crossgl_vec3_sub_f32_sv" in generated_code
    assert "fn _crossgl_vec3_div_f32_vs" in generated_code
    assert "fn _crossgl_vec3_add_i32_vs" in generated_code
    assert "return _crossgl_vec3_add_f32_vs(color, bloom)" in generated_code
    assert (
        "return _crossgl_vec3_add_f32_vv(color, "
        "SIMD[DType.float32, 4](bloom, bloom, bloom, 0.0))"
    ) in generated_code
    assert "return _crossgl_vec3_sub_f32_sv(base, color)" in generated_code
    assert "return _crossgl_vec3_div_f32_vs(color, exposure)" in generated_code
    assert (
        "var mixed: SIMD[DType.float32, 4] = "
        "_crossgl_vec3_add_f32_vv(_crossgl_vec3_add_f32_vs(color, bloom), "
        "_crossgl_vec3_sub_f32_sv(base, color))"
    ) in generated_code
    assert "return _crossgl_vec3_add_i32_vs(index, offset)" in generated_code
    assert "(color + bloom)" not in generated_code
    assert "(base - color)" not in generated_code


def test_vec3_arithmetic_helpers_compile_with_mojo(tmp_path):
    mojo = shutil.which("mojo")
    if mojo is None:
        pytest.skip("mojo compiler is not installed")

    code = """
    vec3 addScalar(vec3 color, float bloom) {
        return color + bloom;
    }

    vec3 addSplat(vec3 color, float bloom) {
        return color + vec3(bloom);
    }

    vec3 scalarSubtract(vec3 color, float base) {
        return base - color;
    }

    vec3 divideScalar(vec3 color, float exposure) {
        return color / exposure;
    }

    vec4 pack(vec3 color, float bloom, float base) {
        vec3 mixed = (color + bloom) + (base - color);
        return vec4(mixed, 1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    var color = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 0.0)
    print(addScalar(color, 0.25))
    print(addSplat(color, 0.25))
    print(scalarSubtract(color, 10.0))
    print(divideScalar(color, 2.0))
    print(pack(color, 0.25, 10.0))
"""

    source_path = tmp_path / "vec3_arithmetic_helpers.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[1.25, 2.25, 3.25, 0.0]" in result.stdout
    assert "[9.0, 8.0, 7.0, 0.0]" in result.stdout
    assert "[0.5, 1.0, 1.5, 0.0]" in result.stdout
    assert "[10.25, 10.25, 10.25, 1.0]" in result.stdout


def test_generic_vector_composite_types_emit_mojo_names():
    code = """
    shader GenericComposite {
        struct Packed {
            vec2<f64> precise;
            vec3<i32> index;
            vec4<u32> mask;
            vec2<bool> flags;
        };

        vec2<f64> passthrough(vec2<f64> value, vec3<i32> index, vec4<u32> mask, vec2<bool> flags) {
            vec2<f64> localValues[2];
            localValues[0] = value;
            return localValues[0];
        }

        compute {
            void main() {
                Packed p;
                vec2<f64> values[2];
                values[0] = vec2<f64>(1.0, 2.0);
                vec2<f64> result = passthrough(values[0], vec3<i32>(1, 2, 3), vec4<u32>(1, 2, 3, 4), vec2<bool>(true, false));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "var precise: SIMD[DType.float64, 2]" in generated_code
    assert "var index: SIMD[DType.int32, 4]" in generated_code
    assert "var mask: SIMD[DType.uint32, 4]" in generated_code
    assert "var flags: SIMD[DType.bool, 2]" in generated_code
    assert (
        "fn passthrough(value: SIMD[DType.float64, 2], "
        "index: SIMD[DType.int32, 4], mask: SIMD[DType.uint32, 4], "
        "flags: SIMD[DType.bool, 2]) -> SIMD[DType.float64, 2]:"
    ) in generated_code
    assert "var localValues: StaticTuple[SIMD[DType.float64, 2], 2]" in generated_code
    assert "var values: StaticTuple[SIMD[DType.float64, 2], 2]" in generated_code
    assert "LiteralNode(" not in generated_code
    assert "vec2<" not in generated_code


def test_array_access():
    code = """
    shader main {
        vertex {
            vec4 main() {
                float values[4];
                values[0] = 1.0;
                values[1] = 2.0;
                return vec4(values[0], values[1], 0.0, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "values[0]" in generated_code and "values[1]" in generated_code
    except SyntaxError:
        pytest.fail("Array access code generation not implemented.")


def test_mojo_imports():
    code = """
    shader main {
        vertex {
            vec4 main() {
                return vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "from math import *" in generated_code
        assert "from simd import *" in generated_code
        assert "from gpu import *" in generated_code
        print(generated_code)
    except SyntaxError:
        pytest.fail("Mojo imports not generated")


if __name__ == "__main__":
    pytest.main()
