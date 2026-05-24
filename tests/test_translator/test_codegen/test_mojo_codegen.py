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


def find_mojo_compiler():
    mojo = shutil.which("mojo")
    if mojo is None:
        pytest.skip("mojo compiler is not installed")

    try:
        result = subprocess.run(
            [mojo, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"mojo compiler is not executable: {exc}")

    if result.returncode != 0:
        output = (result.stderr or result.stdout).strip()
        pytest.skip(
            f"mojo compiler is not installed or not executable ({mojo}): {output}"
        )

    return mojo


def test_find_mojo_compiler_skips_non_compiler_on_path(monkeypatch):
    class FailedProbe:
        returncode = 29
        stdout = ""
        stderr = "Can't find C:\\Strawberry\\perl\\bin\\mojo.BAT on PATH"

    monkeypatch.setattr(
        shutil, "which", lambda name: r"C:\Strawberry\perl\bin\mojo.BAT"
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: FailedProbe())

    with pytest.raises(pytest.skip.Exception):
        find_mojo_compiler()


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


def test_compute_stage_local_helper_functions_emit_before_entry_point():
    code = """
    shader StageLocalMojo {
        compute {
            float helper(float value) {
                return value + 1.0;
            }

            void main() {
                float y = helper(1.0);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert len(next(iter(ast.stages.values())).local_functions) == 1
    helper_signature = "fn helper(value: Float32) -> Float32:"
    entry_signature = "fn main() -> None:"
    assert helper_signature in generated_code
    assert entry_signature in generated_code
    assert generated_code.index(helper_signature) < generated_code.index(
        entry_signature
    )
    assert "var y: Float32 = helper(1.0)" in generated_code


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


def test_sampler3d_type_maps_to_texture3d():
    code = """
    shader VolumeShader {
        sampler3D volumeMap;
        fragment {
            vec4 sampleVolume(sampler3D tex, vec3 uvw) {
                return texture(tex, uvw);
            }
            vec4 main() @ gl_FragColor {
                vec3 uvw = vec3(0.25, 0.5, 0.75);
                return sampleVolume(volumeMap, uvw);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "var volumeMap: Texture3D" in generated_code
    assert "fn sampleVolume(tex: Texture3D, uvw:" in generated_code
    assert "return sample(tex, uvw)" in generated_code
    assert "sampler3D" not in generated_code


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


def test_for_continue_emits_update_before_continue_in_mojo():
    code = """
    shader main {
        compute {
            void main() {
                int total = 0;
                for (int i = 0; i < 4; i++) {
                    if (i == 1) {
                        continue;
                    }
                    for (int j = 0; j < 2; j++) {
                        if (j == 0) {
                            continue;
                        }
                        total += j;
                    }
                    total += i;
                }
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "if (i == 1):\n            i += 1\n            continue" in generated_code
    assert (
        "if (j == 0):\n                j += 1\n                continue"
        in generated_code
    )
    assert (
        "if (j == 0):\n                i += 1\n                continue"
        not in generated_code
    )
    assert "DoWhileNode" not in generated_code


def test_for_in_statement_lowers_to_mojo_ranges_and_scopes_loop_contexts():
    code = """
    shader main {
        compute {
            void main() {
                int total = 0;
                for i in 4 {
                    if (i == 1) {
                        continue;
                    }
                    total += i;
                }
                for j in 2..5 {
                    total += j;
                }
                for k in 1..=4 {
                    total += k;
                }
                for (int outer = 0; outer < 2; outer++) {
                    for inner in 3 {
                        if (inner == 1) {
                            continue;
                        }
                        total += inner;
                    }
                }
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for i in range(4):" in generated_code
    assert "for j in range(2, 5):" in generated_code
    assert "for k in range(1, (4 + 1)):" in generated_code
    assert "total += i" in generated_code
    assert "total += j" in generated_code
    assert "total += k" in generated_code
    assert "total += inner" in generated_code
    assert "if (inner == 1):\n                continue" in generated_code
    assert "outer += 1\n                continue" not in generated_code
    assert "ForInNode" not in generated_code
    assert "RangeNode" not in generated_code


def test_while_statement_lowers_to_mojo_while_and_scopes_loop_contexts():
    code = """
    shader main {
        compute {
            void main() {
                int value = 0;
                while (value < 4) {
                    if (value == 2) {
                        continue;
                    }
                    value += 1;
                }
                for (int i = 0; i < 3; i++) {
                    int j = 0;
                    while (j < 2) {
                        if (j == 1) {
                            continue;
                        }
                        j += 1;
                    }
                }
                do {
                    int k = 0;
                    while (k < 2) {
                        if (k == 1) {
                            break;
                        }
                        k += 1;
                    }
                    value += 1;
                } while (value < 8);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "while (value < 4):" in generated_code
    assert "while (j < 2):" in generated_code
    assert "if (value == 2):\n            continue" in generated_code
    assert "if (j == 1):\n                continue" in generated_code
    assert (
        "if (j == 1):\n                i += 1\n                continue"
        not in generated_code
    )
    assert "__cgl_do_break_0 = True" not in generated_code
    assert "WhileNode" not in generated_code


def test_loop_statement_lowers_to_mojo_while_true_and_scopes_loop_contexts():
    code = """
    shader main {
        compute {
            void main() {
                int value = 0;
                loop {
                    value += 1;
                    if (value == 2) {
                        continue;
                    }
                    if (value > 3) {
                        break;
                    }
                }
                for (int i = 0; i < 3; i++) {
                    loop {
                        if (i == 1) {
                            continue;
                        }
                        break;
                    }
                }
                do {
                    loop {
                        if (value == 4) {
                            break;
                        }
                        continue;
                    }
                    value += 1;
                } while (value < 8);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "while True:" in generated_code
    assert "value += 1" in generated_code
    assert "if (value == 2):\n            continue" in generated_code
    assert "if (i == 1):\n                continue" in generated_code
    assert (
        "if (i == 1):\n                i += 1\n                continue"
        not in generated_code
    )
    assert "__cgl_do_break_0 = True" not in generated_code
    assert "LoopNode" not in generated_code


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


def test_increment_and_decrement_initializers_preserve_mojo_value_order():
    code = """
    shader main {
        compute {
            void main() {
                int i = 0;
                int pre = ++i;
                int post = i++;
                int pre_dec = --i;
                int post_dec = i--;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    lines = [line.strip() for line in generated_code.splitlines()]

    def contains_adjacent(first, second):
        return any(
            current == first and following == second
            for current, following in zip(lines, lines[1:])
        )

    assert contains_adjacent("i += 1", "var pre: Int32 = i")
    assert contains_adjacent("var post: Int32 = i", "i += 1")
    assert contains_adjacent("i -= 1", "var pre_dec: Int32 = i")
    assert contains_adjacent("var post_dec: Int32 = i", "i -= 1")
    assert "var pre: Int32 = i += 1" not in generated_code
    assert "var post: Int32 = i += 1" not in generated_code
    assert "var pre_dec: Int32 = i -= 1" not in generated_code
    assert "var post_dec: Int32 = i -= 1" not in generated_code


def test_do_while_statement_lowers_to_mojo_loop_with_condition_after_body():
    code = """
    shader main {
        compute {
            void main() {
                int value = 0;
                do {
                    value += 1;
                    if (value == 2) {
                        continue;
                    }
                    if (value == 4) {
                        break;
                    }
                    value += 2;
                } while (value < 8);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "var __cgl_do_break_0: Bool = False" in generated_code
    assert "while True:\n        while True:" in generated_code
    assert "value += 1" in generated_code
    assert "if (value == 2):\n                break" in generated_code
    assert (
        "if (value == 4):\n"
        "                __cgl_do_break_0 = True\n"
        "                break"
    ) in generated_code
    assert "if not (value < 8):" in generated_code
    assert "DoWhileNode" not in generated_code


def test_bool_string_and_char_literals_emit_mojo_syntax():
    code = """
    shader main {
        compute {
            void main() {
                bool enabled = true;
                bool disabled = false;
                bool inverted = !disabled;
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
    assert "var inverted: Bool = (not disabled)" in generated_code
    assert 'var label: String = "debug"' in generated_code
    assert 'var marker: String = "x"' in generated_code
    assert "if (enabled and (not disabled)):" in generated_code
    assert "!disabled" not in generated_code
    assert 'label = "active"' in generated_code
    assert 'marker = "y"' in generated_code


def test_inferred_let_declarations_do_not_emit_none_type():
    code = """
    shader main {
        fragment {
            float4 main() {
                let weight = 1;
                let mask = true;
                let tint = float4(1.0, 0.0, 0.0, 1.0);
                return tint;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "var weight = 1" in generated_code
    assert "var mask = True" in generated_code
    assert "var tint = float4(1.0, 0.0, 0.0, 1.0)" in generated_code
    assert ": None" not in generated_code


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


def test_builtin_function_call_names_are_mapped():
    code = """
    shader main {
        compute {
            void main() {
                float x = mix(0.0, 1.0, 0.25);
                float y = dot(vec3(1.0), vec3(2.0));
                float z = pow(2.0, 3.0);
                float w = mod(5.0, 2.0);
                float sat = saturate(1.25);
                float satNested = saturate(frac(1.75));
                vec2 satVec = saturate(vec2(-1.0, 2.0));
                float inv = inversesqrt(x);
                vec2 invVec = inversesqrt(vec2(4.0, 9.0));
                vec3 v = vec3(5.0, 7.0, 9.0);
                vec3 wrappedVec = mod(v, vec3(2.0));
                int n = 5;
                int i = mod(n, 2);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "lerp(0.0, 1.0, 0.25)" in generated_code
    assert "dot_product(" in generated_code
    assert "power(2.0, 3.0)" in generated_code
    assert "var sat: Float32 = clamp(1.25, 0.0, 1.0)" in generated_code
    assert (
        "var satNested: Float32 = "
        "clamp(_crossgl_fract_f32(1.75), 0.0, 1.0)" in generated_code
    )
    assert (
        "var satVec: SIMD[DType.float32, 2] = "
        "_crossgl_saturate_f32_2_2("
        "SIMD[DType.float32, 2]((-1.0), 2.0))" in generated_code
    )
    assert "var inv: Float32 = rsqrt(x)" in generated_code
    assert (
        "var invVec: SIMD[DType.float32, 2] = "
        "rsqrt(SIMD[DType.float32, 2](4.0, 9.0))" in generated_code
    )
    assert "var w: Float32 = fmod(5.0, 2.0)" in generated_code
    assert (
        "var wrappedVec: SIMD[DType.float32, 4] = "
        "fmod(v, SIMD[DType.float32, 4](2.0, 2.0, 2.0, 0.0))" in generated_code
    )
    assert "var i: Int32 = (n % 2)" in generated_code
    assert "mix(0.0, 1.0, 0.25)" not in generated_code
    assert "dot(" not in generated_code
    assert "pow(2.0, 3.0)" not in generated_code
    assert "saturate(" not in generated_code
    assert "frac(" not in generated_code
    assert "inversesqrt(" not in generated_code
    assert "var w: Float32 = mod(" not in generated_code
    assert "var i: Int32 = fmod(" not in generated_code


def test_user_defined_mix_function_is_not_lowered_to_lerp():
    code = """
    shader main {
        compute {
            float mix(float x, float y, float t) {
                return x + y + t;
            }

            void main() {
                float adjusted = mix(0.0, 1.0, 0.25);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn mix(x: Float32, y: Float32, t: Float32) -> Float32:" in generated_code
    assert "var adjusted: Float32 = mix(0.0, 1.0, 0.25)" in generated_code
    assert "var adjusted: Float32 = lerp(0.0, 1.0, 0.25)" not in generated_code


def test_fract_scalar_builtin_lowers_to_helper():
    code = """
    shader main {
        compute {
            void main() {
                float x = fract(1.25);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn _crossgl_fract_f32(x: Float32) -> Float32:" in generated_code
    assert "return x - floor(x)" in generated_code
    assert "var x: Float32 = _crossgl_fract_f32(1.25)" in generated_code
    assert "fract(" not in generated_code


def test_frac_scalar_builtin_alias_lowers_to_fract_helper():
    code = """
    shader main {
        compute {
            void main() {
                float x = frac(1.25);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn _crossgl_fract_f32(x: Float32) -> Float32:" in generated_code
    assert "return x - floor(x)" in generated_code
    assert "var x: Float32 = _crossgl_fract_f32(1.25)" in generated_code
    assert "frac(" not in generated_code


def test_saturate_integer_input_is_left_unmapped():
    code = """
    shader main {
        compute {
            void main() {
                int n = 5;
                int x = saturate(n);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "var x: Int32 = saturate(n)" in generated_code
    assert "clamp(n, 0.0, 1.0)" not in generated_code


def test_saturate_vec3_builtin_lowers_componentwise_and_preserves_padding():
    code = """
    shader main {
        compute {
            void main() {
                vec3 v = saturate(vec3(-1.0, 0.5, 2.0));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "fn _crossgl_saturate_f32_3_4(v: SIMD[DType.float32, 4]) "
        "-> SIMD[DType.float32, 4]:"
    ) in generated_code
    assert (
        "return SIMD[DType.float32, 4]("
        "clamp(v[0], 0.0, 1.0), clamp(v[1], 0.0, 1.0), "
        "clamp(v[2], 0.0, 1.0), 0.0)"
    ) in generated_code
    assert (
        "var v: SIMD[DType.float32, 4] = "
        "_crossgl_saturate_f32_3_4("
        "SIMD[DType.float32, 4]((-1.0), 0.5, 2.0, 0.0))"
    ) in generated_code
    assert "saturate(" not in generated_code


def test_saturate_integer_vector_input_is_left_unmapped():
    code = """
    shader main {
        compute {
            void main() {
                ivec2 n = ivec2(1, 2);
                ivec2 x = saturate(n);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "var x: SIMD[DType.int32, 2] = saturate(n)" in generated_code
    assert "_crossgl_saturate" not in generated_code
    assert "clamp(n, 0.0, 1.0)" not in generated_code


def test_fract_vec3_builtin_lowers_componentwise_and_preserves_padding():
    code = """
    shader main {
        compute {
            void main() {
                vec3 v = fract(vec3(1.25, 2.5, 3.75));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "fn _crossgl_fract_f32_3_4(v: SIMD[DType.float32, 4]) "
        "-> SIMD[DType.float32, 4]:"
    ) in generated_code
    assert (
        "return SIMD[DType.float32, 4]("
        "v[0] - floor(v[0]), v[1] - floor(v[1]), "
        "v[2] - floor(v[2]), 0.0)"
    ) in generated_code
    assert (
        "var v: SIMD[DType.float32, 4] = "
        "_crossgl_fract_f32_3_4("
        "SIMD[DType.float32, 4](1.25, 2.5, 3.75, 0.0))"
    ) in generated_code
    assert "fract(" not in generated_code


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
    assert "struct CrossGLMatrixF32C2R2:" in generated_code
    assert "struct CrossGLMatrixF32C3R4:" in generated_code
    assert "struct CrossGLMatrixF64C2R2:" in generated_code
    assert "struct CrossGLMatrixF64C4R3:" in generated_code
    assert (
        "var transform: CrossGLMatrixF32C2R2 = "
        "CrossGLMatrixF32C2R2(SIMD[DType.float32, 2](1.0, 0.0), "
        "SIMD[DType.float32, 2](0.0, 1.0))"
    ) in generated_code
    assert "var affine: CrossGLMatrixF32C3R4" in generated_code
    assert (
        "var precise: CrossGLMatrixF64C2R2 = "
        "CrossGLMatrixF64C2R2(SIMD[DType.float64, 2](1.0, 0.0), "
        "SIMD[DType.float64, 2](0.0, 1.0))"
    ) in generated_code
    assert "var jacobian: CrossGLMatrixF64C4R3" in generated_code
    assert "dvec2(" not in generated_code
    assert "bvec2(" not in generated_code
    assert "bool2" not in generated_code
    assert "dmat2(" not in generated_code
    assert "Matrix[DType" not in generated_code
    assert "MatrixType(" not in generated_code


def test_matrix_constructors_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    mat2 makeMat2() {
        return mat2(1.0, 2.0, 3.0, 4.0);
    }

    dmat2 makeDMat2() {
        return dmat2(1.0, 2.0, 3.0, 4.0);
    }

    mat3x4 makeMat3x4() {
        return mat3x4(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        );
    }

    mat4x3 makeMat4x3() {
        return mat4x3(
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0
        );
    }

    mat3 diagonal(float value) {
        return mat3(value);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(makeMat2().c0)
    print(makeMat2().c1)
    print(makeDMat2().c1)
    print(makeMat3x4().c2)
    print(makeMat4x3().c3)
    print(diagonal(2.0).c0)
    print(diagonal(2.0).c1)
    print(diagonal(2.0).c2)
"""

    source_path = tmp_path / "matrix_constructors.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[1.0, 2.0]" in result.stdout
    assert "[3.0, 4.0]" in result.stdout
    assert "[9.0, 10.0, 11.0, 12.0]" in result.stdout
    assert "[10.0, 11.0, 12.0, 0.0]" in result.stdout
    assert "[2.0, 0.0, 0.0, 0.0]" in result.stdout
    assert "[0.0, 2.0, 0.0, 0.0]" in result.stdout
    assert "[0.0, 0.0, 2.0, 0.0]" in result.stdout


def test_vector_fed_matrix_constructors_use_helpers_and_index_fields():
    code = """
    vec2 makeUv() {
        return vec2(1.0, 2.0);
    }

    ivec2 makeIndex() {
        return ivec2(3, 4);
    }

    vec3 makeColor() {
        return vec3(5.0, 6.0, 7.0);
    }

    mat2 fromPairs() {
        return mat2(makeUv(), makeIndex());
    }

    mat3 fromColumns() {
        return mat3(makeColor(), makeColor(), makeColor());
    }

    vec2 firstColumn(mat2 value) {
        return value[0];
    }

    float firstElement(mat2 value) {
        return value[0][1];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn _crossgl_construct_matrix_f32_c2_r2" in generated_code
    assert "fn _crossgl_construct_matrix_f32_c3_r3" in generated_code
    assert "var v1_cast = v1.cast[DType.float32]()" in generated_code
    assert (
        "return _crossgl_construct_matrix_f32_c2_r2_2_vf322_01_vi322_01("
        "makeUv(), makeIndex())"
    ) in generated_code
    assert (
        "return _crossgl_construct_matrix_f32_c3_r3_3_vf324_012_vf324_012_vf324_012("
        "makeColor(), makeColor(), makeColor())"
    ) in generated_code
    assert "return value.c0" in generated_code
    assert "return value.c0[1]" in generated_code
    assert "makeUv()[0]" not in generated_code
    assert "makeIndex()[0]" not in generated_code
    assert "makeColor()[0]" not in generated_code
    assert "return value[0]" not in generated_code


def test_vector_fed_matrix_constructors_and_indexing_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    vec2 makeUv() {
        return vec2(1.0, 2.0);
    }

    ivec2 makeIndex() {
        return ivec2(3, 4);
    }

    vec3 makeColor() {
        return vec3(5.0, 6.0, 7.0);
    }

    mat2 fromPairs() {
        return mat2(makeUv(), makeIndex());
    }

    mat3 fromColumns() {
        return mat3(makeColor(), makeColor(), makeColor());
    }

    vec2 firstColumn(mat2 value) {
        return value[0];
    }

    float firstElement(mat2 value) {
        return value[0][1];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(fromPairs().c0)
    print(fromPairs().c1)
    print(fromColumns().c2)
    var matrix = fromPairs()
    print(firstColumn(matrix))
    print(firstElement(matrix))
"""

    source_path = tmp_path / "vector_fed_matrix_constructors.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[1.0, 2.0]" in result.stdout
    assert "[3.0, 4.0]" in result.stdout
    assert "[5.0, 6.0, 7.0, 0.0]" in result.stdout
    assert "2.0" in result.stdout


def test_dynamic_matrix_indexing_emits_getitem_and_vector_index_casts():
    code = """
    mat2 mutateLocal(int column, int row) {
        mat2 value = mat2(1.0, 2.0, 3.0, 4.0);
        vec2 current = value[column];
        value[column] = vec2(current.x + 1.0, current.y + 2.0);
        value[0][1] = 9.0;
        float selected = value[column][row];
        return value;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn __getitem__(self, index: Int32)" in generated_code
    assert (
        "fn __setitem__(inout self, index: Int32, " "value: SIMD[DType.float32, 2])"
    ) in generated_code
    assert "var current: SIMD[DType.float32, 2] = value[column]" in generated_code
    assert "value[column] = SIMD[DType.float32, 2]" in generated_code
    assert "value.c0[1] = 9.0" in generated_code
    assert "var selected: Float32 = value[column][int(row)]" in generated_code


def test_dynamic_matrix_indexing_and_assignment_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    mat2 mutateLocal(int column, int row) {
        mat2 value = mat2(1.0, 2.0, 3.0, 4.0);
        vec2 current = value[column];
        value[column] = vec2(current.x + 1.0, current.y + 2.0);
        value[0][1] = 9.0;
        float selected = value[column][row];
        value[0][0] = selected;
        return value;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    var matrix = mutateLocal(1, 0)
    print(matrix.c0)
    print(matrix.c1)
"""

    source_path = tmp_path / "dynamic_matrix_indexing.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[4.0, 9.0]" in result.stdout
    assert "[4.0, 6.0]" in result.stdout


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
    assert "return SIMD[DType.float32, 4](1.0, 2.0, 3.0, 0.0)" in generated_code
    assert "fn buildPrecise() -> SIMD[DType.float64, 4]:" in generated_code
    assert "return SIMD[DType.float64, 4](1.0, 2.0, 3.0, 0.0)" in generated_code
    assert "fn buildIndex() -> SIMD[DType.int32, 4]:" in generated_code
    assert "return SIMD[DType.int32, 4](1, 2, 3, 0)" in generated_code
    assert "fn buildMask() -> SIMD[DType.uint32, 4]:" in generated_code
    assert "return SIMD[DType.uint32, 4](1, 2, 3, 0)" in generated_code
    assert "fn buildFlags() -> SIMD[DType.bool, 4]:" in generated_code
    assert "return SIMD[DType.bool, 4](True, False, True, False)" in generated_code
    assert ", 3]" not in generated_code


def test_three_component_vector_codegen_compiles_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

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
        "return SIMD[DType.float32, 4](uv[0], uv[1], color[2], 1.0)" in generated_code
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
    mojo = find_mojo_compiler()

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

    assert "return SIMD[DType.float32, 4](bloom, bloom, bloom, 0.0)" in generated_code
    assert "return SIMD[DType.float64, 4](value, value, value, 0.0)" in generated_code
    assert "return SIMD[DType.int32, 4](value, value, value, 0)" in generated_code
    assert "return SIMD[DType.uint32, 4](value, value, value, 0)" in generated_code
    assert (
        "return SIMD[DType.bool, 4](enabled, enabled, enabled, False)" in generated_code
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
    assert "return _crossgl_swizzle_i32_4_xy(laterIndex())" in generated_code
    assert "return _crossgl_swizzle_f64_4_xy(laterPrecise())" in generated_code
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
    mojo = find_mojo_compiler()

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
        "return _crossgl_construct_f32_4_vf322_01_s_s(makeUv(), z, w)" in generated_code
    )
    assert (
        "return _crossgl_construct_f32_4_vf324_012_s(makeColor(), alpha)"
        in generated_code
    )
    assert "return _crossgl_construct_f32_4_vf324_012(makeColor())" in generated_code
    assert "makeUv()[0]" not in generated_code
    assert "makeColor()[0]" not in generated_code


def test_duplicate_sensitive_composite_constructors_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

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
    mojo = find_mojo_compiler()

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
    mojo = find_mojo_compiler()

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


def test_mixed_dtype_vector_constructors_cast_and_preserve_single_eval():
    code = """
    int nextIndex() {
        return 7;
    }

    ivec3 makeIndex() {
        return ivec3(1, 2, 3);
    }

    uvec3 makeMask() {
        return uvec3(4, 5, 6);
    }

    vec3 makeWeight() {
        return vec3(1.25, 2.25, 3.25);
    }

    vec3 fromIndex() {
        return vec3(makeIndex());
    }

    ivec3 fromWeight() {
        return ivec3(makeWeight());
    }

    vec4 packFromIndex(float tail) {
        return vec4(makeIndex().rgb, tail);
    }

    vec4 packFromIndexScalar() {
        return vec4(makeIndex().rgb, nextIndex());
    }

    ivec4 packFromMask(int tail) {
        return ivec4(makeMask().rgb, tail);
    }

    uvec4 packFromIndexUnsigned(uint tail) {
        return uvec4(makeIndex().rgb, tail);
    }

    vec3 fromLocal(ivec3 index) {
        return vec3(index);
    }

    vec3 splatIndex() {
        return vec3(nextIndex());
    }

    vec3 splatLocal(int value) {
        return vec3(value);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn _crossgl_construct_f32_4_vi324_012" in generated_code
    assert "fn _crossgl_construct_i32_4_vf324_012" in generated_code
    assert "fn _crossgl_construct_f32_4_vi324_012_si32" in generated_code
    assert "fn _crossgl_construct_i32_4_vu324_012_s" in generated_code
    assert "fn _crossgl_construct_u32_4_vi324_012_s" in generated_code
    assert "var v0_cast = v0.cast[DType.float32]()" in generated_code
    assert "var v0_cast = v0.cast[DType.int32]()" in generated_code
    assert "var v0_cast = v0.cast[DType.uint32]()" in generated_code
    assert "return _crossgl_construct_f32_4_vi324_012(makeIndex())" in generated_code
    assert "return _crossgl_construct_i32_4_vf324_012(makeWeight())" in generated_code
    assert (
        "return _crossgl_construct_f32_4_vi324_012_s(makeIndex(), tail)"
        in generated_code
    )
    assert (
        "return _crossgl_construct_f32_4_vi324_012_si32(makeIndex(), nextIndex())"
        in generated_code
    )
    assert (
        "return _crossgl_construct_i32_4_vu324_012_s(makeMask(), tail)"
        in generated_code
    )
    assert (
        "return _crossgl_construct_u32_4_vi324_012_s(makeIndex(), tail)"
        in generated_code
    )
    assert (
        "return SIMD[DType.float32, 4](index[0].cast[DType.float32](), "
        "index[1].cast[DType.float32](), index[2].cast[DType.float32](), 0.0)"
    ) in generated_code
    assert (
        "return _crossgl_vec3_splat_f32((nextIndex()).cast[DType.float32]())"
        in generated_code
    )
    assert (
        "return SIMD[DType.float32, 4]((value).cast[DType.float32](), "
        "(value).cast[DType.float32](), (value).cast[DType.float32](), 0.0)"
    ) in generated_code
    assert "makeIndex()[0]" not in generated_code
    assert "makeWeight()[0]" not in generated_code
    assert "makeMask()[0]" not in generated_code


def test_mixed_dtype_vector_constructors_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    int nextIndex() {
        return 7;
    }

    ivec3 makeIndex() {
        return ivec3(1, 2, 3);
    }

    uvec3 makeMask() {
        return uvec3(4, 5, 6);
    }

    vec3 makeWeight() {
        return vec3(1.25, 2.25, 3.25);
    }

    vec3 fromIndex() {
        return vec3(makeIndex());
    }

    ivec3 fromWeight() {
        return ivec3(makeWeight());
    }

    vec4 packFromIndex(float tail) {
        return vec4(makeIndex().rgb, tail);
    }

    vec4 packFromIndexScalar() {
        return vec4(makeIndex().rgb, nextIndex());
    }

    ivec4 packFromMask(int tail) {
        return ivec4(makeMask().rgb, tail);
    }

    uvec4 packFromIndexUnsigned(uint tail) {
        return uvec4(makeIndex().rgb, tail);
    }

    vec3 fromLocal(ivec3 index) {
        return vec3(index);
    }

    vec3 splatIndex() {
        return vec3(nextIndex());
    }

    vec3 splatLocal(int value) {
        return vec3(value);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(fromIndex())
    print(fromWeight())
    print(packFromIndex(9.0))
    print(packFromIndexScalar())
    print(packFromMask(10))
    print(packFromIndexUnsigned(11))
    print(fromLocal(SIMD[DType.int32, 4](8, 9, 10, 0)))
    print(splatIndex())
    print(splatLocal(12))
"""

    source_path = tmp_path / "mixed_dtype_vector_constructors.mojo"
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
    assert "[1.0, 2.0, 3.0, 9.0]" in result.stdout
    assert "[1.0, 2.0, 3.0, 7.0]" in result.stdout
    assert "[4, 5, 6, 10]" in result.stdout
    assert "[1, 2, 3, 11]" in result.stdout
    assert "[8.0, 9.0, 10.0, 0.0]" in result.stdout
    assert "[7.0, 7.0, 7.0, 0.0]" in result.stdout
    assert "[12.0, 12.0, 12.0, 0.0]" in result.stdout


def test_scalar_vec2_vec4_constructors_emit_splat_form():
    code = """
    float nextFloat() {
        return 1.25;
    }

    double nextDouble() {
        return 2.5;
    }

    int nextInt() {
        return 3;
    }

    bool nextBool() {
        return true;
    }

    vec2 splatVec2Float() {
        return vec2(nextFloat());
    }

    vec4 splatVec4Float() {
        return vec4(nextFloat());
    }

    vec2 splatVec2Int() {
        return vec2(nextInt());
    }

    vec4 splatVec4Int() {
        return vec4(nextInt());
    }

    dvec2 splatDvec2Float() {
        return dvec2(nextFloat());
    }

    dvec4 splatDvec4Float() {
        return dvec4(nextFloat());
    }

    vec2 splatVec2Double() {
        return vec2(nextDouble());
    }

    vec4 splatVec4Double() {
        return vec4(nextDouble());
    }

    ivec2 splatIvec2Float() {
        return ivec2(nextFloat());
    }

    ivec4 splatIvec4Float() {
        return ivec4(nextFloat());
    }

    bvec2 splatBvec2() {
        return bvec2(nextBool());
    }

    bvec4 splatBvec4() {
        return bvec4(nextBool());
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "return SIMD[DType.float32, 2](nextFloat())" in generated_code
    assert "return SIMD[DType.float32, 4](nextFloat())" in generated_code
    assert (
        "return SIMD[DType.float32, 2]((nextInt()).cast[DType.float32]())"
        in generated_code
    )
    assert (
        "return SIMD[DType.float32, 4]((nextInt()).cast[DType.float32]())"
        in generated_code
    )
    assert (
        "return SIMD[DType.float64, 2]((nextFloat()).cast[DType.float64]())"
        in generated_code
    )
    assert (
        "return SIMD[DType.float64, 4]((nextFloat()).cast[DType.float64]())"
        in generated_code
    )
    assert (
        "return SIMD[DType.float32, 2]((nextDouble()).cast[DType.float32]())"
        in generated_code
    )
    assert (
        "return SIMD[DType.float32, 4]((nextDouble()).cast[DType.float32]())"
        in generated_code
    )
    assert (
        "return SIMD[DType.int32, 2]((nextFloat()).cast[DType.int32]())"
        in generated_code
    )
    assert (
        "return SIMD[DType.int32, 4]((nextFloat()).cast[DType.int32]())"
        in generated_code
    )
    assert "return SIMD[DType.bool, 2](nextBool())" in generated_code
    assert "return SIMD[DType.bool, 4](nextBool())" in generated_code
    assert "fn _crossgl_vec3_splat" not in generated_code


def test_scalar_vec2_vec4_constructors_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    float nextFloat() {
        return 1.25;
    }

    double nextDouble() {
        return 2.5;
    }

    int nextInt() {
        return 3;
    }

    bool nextBool() {
        return true;
    }

    vec2 splatVec2Float() {
        return vec2(nextFloat());
    }

    vec4 splatVec4Float() {
        return vec4(nextFloat());
    }

    vec2 splatVec2Int() {
        return vec2(nextInt());
    }

    vec4 splatVec4Int() {
        return vec4(nextInt());
    }

    dvec2 splatDvec2Float() {
        return dvec2(nextFloat());
    }

    dvec4 splatDvec4Float() {
        return dvec4(nextFloat());
    }

    vec2 splatVec2Double() {
        return vec2(nextDouble());
    }

    vec4 splatVec4Double() {
        return vec4(nextDouble());
    }

    ivec2 splatIvec2Float() {
        return ivec2(nextFloat());
    }

    ivec4 splatIvec4Float() {
        return ivec4(nextFloat());
    }

    bvec2 splatBvec2() {
        return bvec2(nextBool());
    }

    bvec4 splatBvec4() {
        return bvec4(nextBool());
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(splatVec2Float())
    print(splatVec4Float())
    print(splatVec2Int())
    print(splatVec4Int())
    print(splatDvec2Float())
    print(splatDvec4Float())
    print(splatVec2Double())
    print(splatVec4Double())
    print(splatIvec2Float())
    print(splatIvec4Float())
    print(splatBvec2())
    print(splatBvec4())
"""

    source_path = tmp_path / "scalar_vec2_vec4_constructors.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[1.25, 1.25]" in result.stdout
    assert "[1.25, 1.25, 1.25, 1.25]" in result.stdout
    assert "[3.0, 3.0]" in result.stdout
    assert "[3.0, 3.0, 3.0, 3.0]" in result.stdout
    assert "[2.5, 2.5]" in result.stdout
    assert "[2.5, 2.5, 2.5, 2.5]" in result.stdout
    assert "[1, 1]" in result.stdout
    assert "[1, 1, 1, 1]" in result.stdout
    assert "[True, True]" in result.stdout
    assert "[True, True, True, True]" in result.stdout


def test_nested_composite_vector_constructors_use_helpers_and_casts():
    code = """
    float nextFloat() {
        return 1.25;
    }

    double nextDouble() {
        return 2.5;
    }

    int nextInt() {
        return 3;
    }

    uint nextUint() {
        return 4;
    }

    bool nextBool() {
        return true;
    }

    vec2 makeUv() {
        return vec2(0.25, 0.75);
    }

    ivec2 makeIndexPair() {
        return ivec2(1, 2);
    }

    uvec2 makeMaskPair() {
        return uvec2(3, 4);
    }

    dvec2 makePrecisePair() {
        return dvec2(5.5, 6.5);
    }

    bvec2 makeFlagPair() {
        return bvec2(true, false);
    }

    vec4 packTwoNestedVec2() {
        return vec4(vec2(nextFloat()), vec2(nextDouble()));
    }

    vec4 packNestedVec2Scalars() {
        return vec4(vec2(nextFloat()), nextInt(), nextDouble());
    }

    vec4 packMixedIvec2() {
        return vec4(ivec2(nextInt()), nextFloat(), nextInt());
    }

    ivec4 packMixedUvec2() {
        return ivec4(uvec2(nextUint()), nextFloat(), nextInt());
    }

    dvec4 packMixedVec2() {
        return dvec4(vec2(nextFloat()), nextDouble(), nextInt());
    }

    bvec4 packNestedBvec2() {
        return bvec4(bvec2(nextBool()), false, nextBool());
    }

    vec4 packLocal(vec2 uv, ivec2 index) {
        return vec4(uv, index);
    }

    vec4 packFunctionPairs() {
        return vec4(makeUv(), makeIndexPair());
    }

    ivec4 packUnsignedPair() {
        return ivec4(makeMaskPair(), makeIndexPair());
    }

    dvec4 packPrecisePair() {
        return dvec4(makePrecisePair(), makeUv());
    }

    bvec4 packFlagPair() {
        return bvec4(makeFlagPair(), nextBool(), false);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn _crossgl_construct_f32_4_vf322_01_vf322_01" in generated_code
    assert "fn _crossgl_construct_f32_4_vf322_01_si32_sf64" in generated_code
    assert "fn _crossgl_construct_f32_4_vi322_01_s_si32" in generated_code
    assert "fn _crossgl_construct_i32_4_vu322_01_sf32_s" in generated_code
    assert "fn _crossgl_construct_f64_4_vf322_01_s_si32" in generated_code
    assert "fn _crossgl_construct_bool_4_vbool2_01_s_s" in generated_code
    assert "fn _crossgl_construct_f32_4_vf322_01_vi322_01" in generated_code
    assert "fn _crossgl_construct_i32_4_vu322_01_vi322_01" in generated_code
    assert "fn _crossgl_construct_f64_4_vf642_01_vf322_01" in generated_code
    assert "var v0_cast = v0.cast[DType.float32]()" in generated_code
    assert "var v0_cast = v0.cast[DType.int32]()" in generated_code
    assert "var v1_cast = v1.cast[DType.float64]()" in generated_code
    assert (
        "return _crossgl_construct_f32_4_vf322_01_vf322_01("
        "SIMD[DType.float32, 2](nextFloat()), "
        "SIMD[DType.float32, 2]((nextDouble()).cast[DType.float32]()))"
    ) in generated_code
    assert (
        "return _crossgl_construct_f32_4_vf322_01_si32_sf64("
        "SIMD[DType.float32, 2](nextFloat()), nextInt(), nextDouble())"
    ) in generated_code
    assert (
        "return _crossgl_construct_f32_4_vi322_01_s_si32("
        "SIMD[DType.int32, 2](nextInt()), nextFloat(), nextInt())"
    ) in generated_code
    assert (
        "return _crossgl_construct_i32_4_vu322_01_sf32_s("
        "SIMD[DType.uint32, 2](nextUint()), nextFloat(), nextInt())"
    ) in generated_code
    assert (
        "return SIMD[DType.float32, 4](uv[0], uv[1], "
        "index[0].cast[DType.float32](), index[1].cast[DType.float32]())"
    ) in generated_code
    assert (
        "return _crossgl_construct_f32_4_vf322_01_vi322_01("
        "makeUv(), makeIndexPair())"
    ) in generated_code
    assert "SIMD[DType.float32, 2](nextFloat())[0]" not in generated_code
    assert "makeUv()[0]" not in generated_code
    assert "makeIndexPair()[0]" not in generated_code
    assert "makeMaskPair()[0]" not in generated_code


def test_nested_composite_vector_constructors_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    float nextFloat() {
        return 1.25;
    }

    double nextDouble() {
        return 2.5;
    }

    int nextInt() {
        return 3;
    }

    uint nextUint() {
        return 4;
    }

    bool nextBool() {
        return true;
    }

    vec2 makeUv() {
        return vec2(0.25, 0.75);
    }

    ivec2 makeIndexPair() {
        return ivec2(1, 2);
    }

    uvec2 makeMaskPair() {
        return uvec2(3, 4);
    }

    dvec2 makePrecisePair() {
        return dvec2(5.5, 6.5);
    }

    bvec2 makeFlagPair() {
        return bvec2(true, false);
    }

    vec4 packTwoNestedVec2() {
        return vec4(vec2(nextFloat()), vec2(nextDouble()));
    }

    vec4 packNestedVec2Scalars() {
        return vec4(vec2(nextFloat()), nextInt(), nextDouble());
    }

    vec4 packMixedIvec2() {
        return vec4(ivec2(nextInt()), nextFloat(), nextInt());
    }

    ivec4 packMixedUvec2() {
        return ivec4(uvec2(nextUint()), nextFloat(), nextInt());
    }

    dvec4 packMixedVec2() {
        return dvec4(vec2(nextFloat()), nextDouble(), nextInt());
    }

    bvec4 packNestedBvec2() {
        return bvec4(bvec2(nextBool()), false, nextBool());
    }

    vec4 packLocal(vec2 uv, ivec2 index) {
        return vec4(uv, index);
    }

    vec4 packFunctionPairs() {
        return vec4(makeUv(), makeIndexPair());
    }

    ivec4 packUnsignedPair() {
        return ivec4(makeMaskPair(), makeIndexPair());
    }

    dvec4 packPrecisePair() {
        return dvec4(makePrecisePair(), makeUv());
    }

    bvec4 packFlagPair() {
        return bvec4(makeFlagPair(), nextBool(), false);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(packTwoNestedVec2())
    print(packNestedVec2Scalars())
    print(packMixedIvec2())
    print(packMixedUvec2())
    print(packMixedVec2())
    print(packNestedBvec2())
    print(packFunctionPairs())
    print(packUnsignedPair())
    print(packPrecisePair())
    print(packFlagPair())
    print(packLocal(SIMD[DType.float32, 2](8.0, 9.0), SIMD[DType.int32, 2](10, 11)))
"""

    source_path = tmp_path / "nested_composite_vector_constructors.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[1.25, 1.25, 2.5, 2.5]" in result.stdout
    assert "[1.25, 1.25, 3.0, 2.5]" in result.stdout
    assert "[3.0, 3.0, 1.25, 3.0]" in result.stdout
    assert "[4, 4, 1, 3]" in result.stdout
    assert "[1.25, 1.25, 2.5, 3.0]" in result.stdout
    assert "[True, True, False, True]" in result.stdout
    assert "[0.25, 0.75, 1.0, 2.0]" in result.stdout
    assert "[3, 4, 1, 2]" in result.stdout
    assert "[5.5, 6.5, 0.25, 0.75]" in result.stdout
    assert "[True, False, True, False]" in result.stdout
    assert "[8.0, 9.0, 10.0, 11.0]" in result.stdout


def test_truncating_vector_constructors_use_helpers_and_pad_hidden_lane():
    code = """
    vec4 makeColor() {
        return vec4(1.0, 2.0, 3.0, 4.0);
    }

    ivec4 makeIndex() {
        return ivec4(5, 6, 7, 8);
    }

    uvec4 makeMask() {
        return uvec4(9, 10, 11, 12);
    }

    bvec4 makeFlags() {
        return bvec4(true, false, true, false);
    }

    vec2 narrowColor() {
        return vec2(makeColor());
    }

    vec3 narrowColor3() {
        return vec3(makeColor());
    }

    vec2 narrowIndexToFloat() {
        return vec2(makeIndex());
    }

    ivec3 narrowMaskToInt() {
        return ivec3(makeMask());
    }

    bvec3 narrowFlags() {
        return bvec3(makeFlags());
    }

    vec3 localNarrow(vec4 color) {
        return vec3(color);
    }

    vec3 overfull(vec2 xy, vec2 zw) {
        return vec3(xy, zw);
    }

    vec3 overfullDuplicate() {
        return vec3(vec2(1.0, 2.0), vec2(3.0, 4.0));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn _crossgl_construct_f32_2_vf324_01" in generated_code
    assert "fn _crossgl_construct_f32_2_vi324_01" in generated_code
    assert "fn _crossgl_construct_f32_4_vf324_012" in generated_code
    assert "fn _crossgl_construct_i32_4_vu324_012" in generated_code
    assert "fn _crossgl_construct_bool_4_vbool4_012" in generated_code
    assert "fn _crossgl_construct_f32_4_vf322_01_vf322_0" in generated_code
    assert "var v0_cast = v0.cast[DType.float32]()" in generated_code
    assert "var v0_cast = v0.cast[DType.int32]()" in generated_code
    assert "return _crossgl_construct_f32_2_vf324_01(makeColor())" in generated_code
    assert "return _crossgl_construct_f32_4_vf324_012(makeColor())" in generated_code
    assert "return _crossgl_construct_f32_2_vi324_01(makeIndex())" in generated_code
    assert "return _crossgl_construct_i32_4_vu324_012(makeMask())" in generated_code
    assert "return _crossgl_construct_bool_4_vbool4_012(makeFlags())" in generated_code
    assert (
        "return SIMD[DType.float32, 4](color[0], color[1], color[2], 0.0)"
        in generated_code
    )
    assert "return SIMD[DType.float32, 4](xy[0], xy[1], zw[0], 0.0)" in generated_code
    assert (
        "return _crossgl_construct_f32_4_vf322_01_vf322_0("
        "SIMD[DType.float32, 2](1.0, 2.0), "
        "SIMD[DType.float32, 2](3.0, 4.0))"
    ) in generated_code
    assert "makeColor()[0]" not in generated_code
    assert "makeIndex()[0]" not in generated_code
    assert "makeMask()[0]" not in generated_code
    assert "makeFlags()[0]" not in generated_code
    assert "zw[1]" not in generated_code


def test_truncating_vector_constructors_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    vec4 makeColor() {
        return vec4(1.0, 2.0, 3.0, 4.0);
    }

    ivec4 makeIndex() {
        return ivec4(5, 6, 7, 8);
    }

    uvec4 makeMask() {
        return uvec4(9, 10, 11, 12);
    }

    bvec4 makeFlags() {
        return bvec4(true, false, true, false);
    }

    vec2 narrowColor() {
        return vec2(makeColor());
    }

    vec3 narrowColor3() {
        return vec3(makeColor());
    }

    vec2 narrowIndexToFloat() {
        return vec2(makeIndex());
    }

    ivec3 narrowMaskToInt() {
        return ivec3(makeMask());
    }

    bvec3 narrowFlags() {
        return bvec3(makeFlags());
    }

    vec3 localNarrow(vec4 color) {
        return vec3(color);
    }

    vec3 overfull(vec2 xy, vec2 zw) {
        return vec3(xy, zw);
    }

    vec3 overfullDuplicate() {
        return vec3(vec2(1.0, 2.0), vec2(3.0, 4.0));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(narrowColor())
    print(narrowColor3())
    print(narrowIndexToFloat())
    print(narrowMaskToInt())
    print(narrowFlags())
    print(localNarrow(SIMD[DType.float32, 4](13.0, 14.0, 15.0, 16.0)))
    print(overfull(SIMD[DType.float32, 2](17.0, 18.0), SIMD[DType.float32, 2](19.0, 20.0)))
    print(overfullDuplicate())
"""

    source_path = tmp_path / "truncating_vector_constructors.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[1.0, 2.0]" in result.stdout
    assert "[1.0, 2.0, 3.0, 0.0]" in result.stdout
    assert "[5.0, 6.0]" in result.stdout
    assert "[9, 10, 11, 0]" in result.stdout
    assert "[True, False, True, False]" in result.stdout
    assert "[13.0, 14.0, 15.0, 0.0]" in result.stdout
    assert "[17.0, 18.0, 19.0, 0.0]" in result.stdout
    assert "[1.0, 2.0, 3.0, 0.0]" in result.stdout


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
    mojo = find_mojo_compiler()

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
    assert (
        "var localValues = InlineArray[SIMD[DType.float64, 2], 2]"
        "(unsafe_uninitialized=True)"
    ) in generated_code
    assert (
        "var values = InlineArray[SIMD[DType.float64, 2], 2]"
        "(unsafe_uninitialized=True)"
    ) in generated_code
    assert "StaticTuple" not in generated_code
    assert "LiteralNode(" not in generated_code
    assert "vec2<" not in generated_code


def test_fixed_size_arrays_emit_inlinearray_and_cast_dynamic_indices():
    code = """
    struct Packed {
        mat2 transforms[2];
        vec2 samples[2];
    };

    vec2 vectorArray(vec2 value, int index) {
        vec2 values[2];
        values[0] = vec2(1.0, 2.0);
        values[index] = value;
        return values[index];
    }

    mat2 matrixArray(int index) {
        mat2 values[2];
        values[0] = mat2(1.0, 2.0, 3.0, 4.0);
        values[index] = mat2(5.0, 6.0, 7.0, 8.0);
        return values[index];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "var transforms: InlineArray[CrossGLMatrixF32C2R2, 2]" in generated_code
    assert "var samples: InlineArray[SIMD[DType.float32, 2], 2]" in generated_code
    assert (
        "var values = InlineArray[SIMD[DType.float32, 2], 2]"
        "(unsafe_uninitialized=True)"
    ) in generated_code
    assert (
        "var values = InlineArray[CrossGLMatrixF32C2R2, 2]"
        "(unsafe_uninitialized=True)"
    ) in generated_code
    assert "values[int(index)] = value" in generated_code
    assert "return values[int(index)]" in generated_code
    assert "StaticTuple" not in generated_code
    assert "DynamicVector" not in generated_code


def test_fixed_size_arrays_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    vec4 scalarArray(int index) {
        float values[4];
        values[0] = 1.0;
        values[1] = 2.0;
        values[2] = 4.0;
        values[3] = 5.0;
        values[index] = 3.0;
        return vec4(values[0], values[index], values[2], values[3]);
    }

    vec2 vectorArray(vec2 value, int index) {
        vec2 values[2];
        values[0] = vec2(1.0, 2.0);
        values[index] = value;
        return values[index];
    }

    mat2 matrixArray(int index) {
        mat2 values[2];
        values[0] = mat2(1.0, 2.0, 3.0, 4.0);
        values[index] = mat2(5.0, 6.0, 7.0, 8.0);
        return values[index];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(scalarArray(1))
    print(vectorArray(SIMD[DType.float32, 2](8.0, 9.0), 1))
    print(matrixArray(1).c1)
"""

    source_path = tmp_path / "fixed_size_arrays.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[1.0, 3.0, 4.0, 5.0]" in result.stdout
    assert "[8.0, 9.0]" in result.stdout
    assert "[7.0, 8.0]" in result.stdout


def test_struct_array_fields_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct Packed {
        vec2 samples[2];
        mat2 transforms[2];
    };

    Packed makePacked() {
        vec2 samples[2] = {vec2(1.0, 2.0), vec2(3.0, 4.0)};
        mat2 transforms[2] = {
            mat2(1.0, 2.0, 3.0, 4.0),
            mat2(5.0, 6.0, 7.0, 8.0)
        };
        return Packed(samples, transforms);
    }

    vec2 readSample(Packed packed, int index) {
        return packed.samples[index];
    }

    vec2 readColumn(Packed packed, int index) {
        return packed.transforms[index][1];
    }

    float readElement(Packed packed, int matrixIndex, int row) {
        return packed.transforms[matrixIndex][1][row];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    var packed = makePacked()
    print(readSample(packed, 1))
    print(readColumn(packed, 1))
    print(readElement(packed, 1, 0))
"""

    source_path = tmp_path / "struct_array_fields.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[3.0, 4.0]" in result.stdout
    assert "[7.0, 8.0]" in result.stdout
    assert "7.0" in result.stdout


def test_local_struct_array_fields_default_initialize_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct Packed {
        float weights[4];
        vec2 sample;
        mat2 transforms[2];
    };

    Packed buildPacked(int index) {
        Packed packed;
        packed.weights = {1.0, 2.0};
        packed.weights[index] = 5.0;
        packed.sample = vec2(3.0, 4.0);
        packed.transforms = {mat2(6.0, 7.0, 8.0, 9.0)};
        return packed;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "var packed: Packed" not in generated_code
    assert "var packed = Packed(" in generated_code
    assert "InlineArray[Float32, 4](0.0, 0.0, 0.0, 0.0)" in generated_code

    generated_code += """
fn main():
    var packed = buildPacked(1)
    print(packed.weights[0])
    print(packed.weights[1])
    print(packed.weights[3])
    print(packed.sample)
    print(packed.transforms[1].c1)
"""

    source_path = tmp_path / "local_struct_array_fields.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "1.0" in result.stdout
    assert "5.0" in result.stdout
    assert "0.0" in result.stdout
    assert "[3.0, 4.0]" in result.stdout
    assert "[0.0, 0.0]" in result.stdout


def test_nested_struct_arrays_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct Inner {
        vec2 sample;
        float weights[2];
    };

    struct Outer {
        Inner primary;
        Inner entries[2];
    };

    Inner makeInner(float base) {
        Inner inner;
        inner.sample = vec2(base, base + 1.0);
        inner.weights = {base + 2.0, base + 3.0};
        return inner;
    }

    Outer buildOuter(int index) {
        Outer outer;
        outer.primary = makeInner(1.0);
        outer.entries = {makeInner(5.0)};
        outer.entries[index] = makeInner(9.0);
        outer.entries[index].sample = vec2(13.0, 14.0);
        outer.entries[index].weights[1] = 15.0;
        return outer;
    }

    vec4 readOuter(Outer outer, int index) {
        return vec4(outer.primary.sample, outer.entries[index].sample);
    }

    float readNestedWeight(Outer outer, int index) {
        return outer.entries[index].weights[1];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "var entries: InlineArray[Inner, 2]" in generated_code
    assert "var inner = Inner(" in generated_code
    assert "var outer = Outer(" in generated_code
    assert "outer.entries[int(index)].weights[1] = 15.0" in generated_code

    generated_code += """
fn main():
    var outer = buildOuter(1)
    print(readOuter(outer, 1))
    print(readNestedWeight(outer, 1))
    print(outer.entries[0].sample)
    print(outer.entries[0].weights[1])
"""

    source_path = tmp_path / "nested_struct_arrays.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[1.0, 2.0, 13.0, 14.0]" in result.stdout
    assert "15.0" in result.stdout
    assert "[5.0, 6.0]" in result.stdout
    assert "8.0" in result.stdout


def test_array_literals_emit_inlinearray_and_zero_padding():
    code = """
    float globalWeights[4] = {1.0, 2.0};

    float pickScalar(int index) {
        float values[4] = {1.0, 2.0};
        return values[index] + values[3];
    }

    vec2 pickVector(int index) {
        vec2 values[2] = {vec2(1.0, 2.0), vec2(3.0, 4.0)};
        return values[index];
    }

    float[4] makeValues() {
        return {1.0, 2.0, 3.0, 4.0};
    }

    mat2 pickMatrix() {
        mat2 values[2] = {mat2(1.0, 2.0, 3.0, 4.0)};
        return values[1];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "var globalWeights: InlineArray[Float32, 4] = "
        "InlineArray[Float32, 4](1.0, 2.0, 0.0, 0.0)"
    ) in generated_code
    assert (
        "var values: InlineArray[Float32, 4] = "
        "InlineArray[Float32, 4](1.0, 2.0, 0.0, 0.0)"
    ) in generated_code
    assert (
        "InlineArray[SIMD[DType.float32, 2], 2]("
        "SIMD[DType.float32, 2](1.0, 2.0), "
        "SIMD[DType.float32, 2](3.0, 4.0))"
    ) in generated_code
    assert "return InlineArray[Float32, 4](1.0, 2.0, 3.0, 4.0)" in generated_code
    assert (
        "CrossGLMatrixF32C2R2(SIMD[DType.float32, 2](0.0, 0.0), "
        "SIMD[DType.float32, 2](0.0, 0.0))"
    ) in generated_code
    assert "IdentifierNode(name={)" not in generated_code


def test_array_literals_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    float globalWeights[4] = {1.0, 2.0};

    float pickGlobal(int index) {
        return globalWeights[index];
    }

    float pickScalar(int index) {
        float values[4] = {1.0, 2.0};
        return values[index] + values[3];
    }

    vec2 pickVector(int index) {
        vec2 values[2] = {vec2(1.0, 2.0), vec2(3.0, 4.0)};
        return values[index];
    }

    float[4] makeValues() {
        return {1.0, 2.0, 3.0, 4.0};
    }

    mat2 pickMatrix() {
        mat2 values[2] = {mat2(1.0, 2.0, 3.0, 4.0)};
        return values[1];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(pickGlobal(3))
    print(pickScalar(1))
    print(pickScalar(3))
    print(pickVector(1))
    print(makeValues()[2])
    print(pickMatrix().c1)
"""

    source_path = tmp_path / "array_literals.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "0.0" in result.stdout
    assert "2.0" in result.stdout
    assert "[3.0, 4.0]" in result.stdout
    assert "3.0" in result.stdout
    assert "[0.0, 0.0]" in result.stdout


def test_mutated_parameters_emit_owned_only_when_needed():
    code = """
    float writeArray(float values[4], int index) {
        values[index] = 9.0;
        return values[index];
    }

    mat2 writeMatrix(mat2 value, int column) {
        value[column] = vec2(9.0, 10.0);
        return value;
    }

    float bumpScalar(float value) {
        value += 1.0;
        return value;
    }

    vec2 tweakVector(vec2 value) {
        value[0] = 5.0;
        return value;
    }

    float readArray(float values[4], int index) {
        return values[index];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "fn writeArray(owned values: InlineArray[Float32, 4], index: Int32)"
        in generated_code
    )
    assert (
        "fn writeMatrix(owned value: CrossGLMatrixF32C2R2, column: Int32)"
        in generated_code
    )
    assert "fn bumpScalar(owned value: Float32)" in generated_code
    assert "fn tweakVector(owned value: SIMD[DType.float32, 2])" in generated_code
    assert (
        "fn readArray(values: InlineArray[Float32, 4], index: Int32)" in generated_code
    )
    assert "fn readArray(owned values" not in generated_code


def test_mutated_array_matrix_vector_parameters_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    float writeArray(float values[4], int index) {
        values[index] = 9.0;
        return values[index];
    }

    mat2 writeMatrix(mat2 value, int column) {
        value[column] = vec2(9.0, 10.0);
        return value;
    }

    float bumpScalar(float value) {
        value += 1.0;
        return value;
    }

    vec2 tweakVector(vec2 value) {
        value[0] = 5.0;
        return value;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    var values = InlineArray[Float32, 4](unsafe_uninitialized=True)
    values[0] = 1.0
    values[1] = 2.0
    print(writeArray(values, 1))
    print(values[1])

    var matrix = CrossGLMatrixF32C2R2(
        SIMD[DType.float32, 2](1.0, 2.0),
        SIMD[DType.float32, 2](3.0, 4.0),
    )
    var changed = writeMatrix(matrix, 1)
    print(changed.c1)
    print(matrix.c1)

    print(bumpScalar(2.0))
    print(tweakVector(SIMD[DType.float32, 2](1.0, 2.0)))
"""

    source_path = tmp_path / "mutated_parameters.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "9.0" in result.stdout
    assert "2.0" in result.stdout
    assert "[9.0, 10.0]" in result.stdout
    assert "[3.0, 4.0]" in result.stdout
    assert "3.0" in result.stdout
    assert "[5.0, 2.0]" in result.stdout


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


def test_switch_statement_lowers_to_mojo_condition_chain():
    code = """
    shader TestShader {
        compute {
            void main() {
                int x = 1;
                switch (x) {
                    case 1:
                        x = 2;
                        break;
                    case 2:
                        x = 4;
                        break;
                    default:
                        x = 3;
                }
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "if x == 1:" in generated_code
    assert "elif x == 2:" in generated_code
    assert "else:" in generated_code
    assert "x = 2" in generated_code
    assert "x = 4" in generated_code
    assert "x = 3" in generated_code
    assert "SwitchNode" not in generated_code
    assert "CaseNode" not in generated_code
    assert "break" not in generated_code


def test_match_statement_lowers_to_mojo_condition_chain():
    code = """
    shader main {
        compute {
            int main(int mode) {
                int value = 0;
                match mode {
                    0 => {
                        value = 1;
                    }
                    _ => {
                        value = 2;
                    }
                }
                return value;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "if mode == 0:" in generated_code
    assert "else:" in generated_code
    assert "value = 1" in generated_code
    assert "value = 2" in generated_code
    assert "MatchNode" not in generated_code
    assert "MatchArmNode" not in generated_code


def test_match_guarded_arm_is_rejected_for_mojo_codegen():
    code = """
    shader main {
        compute {
            int main(int mode) {
                int value = 0;
                match mode {
                    0 if mode > 0 => {
                        value = 1;
                    }
                    _ => {
                        value = 2;
                    }
                }
                return value;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    with pytest.raises(ValueError, match="Unsupported match arm for Mojo"):
        generate_code(ast)


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
