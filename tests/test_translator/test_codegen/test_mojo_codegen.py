import shutil
import subprocess
from typing import List

import pytest
import crosstl.translator
from crosstl.translator.parser import Parser
from crosstl.translator.lexer import Lexer
from crosstl.translator.ast import (
    AtomicOpNode,
    ArrayLiteralNode,
    AssignmentNode,
    BlockNode,
    BufferOpNode,
    BuiltinVariableNode,
    CastNode,
    ConstructorNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralPatternNode,
    LiteralNode,
    MatchArmNode,
    MatchNode,
    MeshOpNode,
    PrimitiveType,
    RayQueryOpNode,
    RayTracingOpNode,
    SwizzleNode,
    SyncNode,
    TextureNode,
    TextureOpNode,
    VariableNode,
    VectorType,
    WaveOpNode,
    WildcardPatternNode,
)
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


def test_cbuffer_type_nodes_emit_mojo_storage_types():
    code = """
    cbuffer Camera {
        mat4 viewProj;
        vec4 tint;
        float weights[4];
    };
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "var viewProj: CrossGLMatrixF32C4R4" in generated_code
    assert "var tint: SIMD[DType.float32, 4]" in generated_code
    assert "var weights: InlineArray[Float32, 4]" in generated_code
    assert "MatrixType(" not in generated_code
    assert "VectorType(" not in generated_code
    assert "ArrayType(" not in generated_code
    assert "PrimitiveType(" not in generated_code


def test_cbuffer_type_nodes_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    cbuffer Camera {
        mat4 viewProj;
        vec4 tint;
        float weights[4];
    };
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "cbuffer_type_nodes.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_cbuffer_structs_register_for_mojo_value_usage(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    cbuffer Camera {
        mat2 transform;
        vec4 tint;
        float weights[2];
    };

    Camera makeCamera(int index) {
        Camera camera;
        camera.tint = vec4(1.0);
        camera.weights[index] = 2.0;
        return camera;
    }

    vec4 readTint(Camera camera) {
        return camera.tint;
    }

    float readWeight(Camera camera, int index) {
        return camera.weights[index];
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "# CrossGL resource metadata: name=Camera kind=cbuffer" in generated_code
    assert "struct Camera:" in generated_code
    assert "var transform: CrossGLMatrixF32C2R2" in generated_code
    assert "var tint: SIMD[DType.float32, 4]" in generated_code
    assert "var weights: InlineArray[Float32, 2]" in generated_code
    assert "var camera = Camera(" in generated_code
    assert "camera.weights[int(index)] = 2.0" in generated_code
    assert "return camera.weights[int(index)]" in generated_code
    assert "var camera: Camera" not in generated_code

    generated_code += """
fn main():
    var camera = makeCamera(1)
    print(readTint(camera))
    print(readWeight(camera, 1))
"""

    source_path = tmp_path / "cbuffer_struct_value_usage.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[1.0, 1.0, 1.0, 1.0]" in result.stdout
    assert "2.0" in result.stdout


def test_braced_struct_constructors_emit_mojo_initializers():
    code = """
    struct Particle {
        vec4 position;
        float mass;
    };

    Particle makeNamed(vec4 position, float mass) {
        return Particle { position: position, mass: mass };
    }

    Particle makeShorthand(vec4 position, float mass) {
        return Particle { position, mass };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "return Particle(position=position, mass=mass)" in generated_code
    assert "ConstructorNode(" not in generated_code


def test_braced_struct_constructors_zero_fill_missing_fields():
    code = """
    struct Pair {
        float x;
        float y;
    };

    struct Mixed {
        float value;
        float2 vectorValue;
    };

    Pair makePartialNamed() {
        return Pair { y: 2.0 };
    }

    Mixed makePartialMixed() {
        return Mixed { value: 3.0 };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "return Pair(x=0.0, y=2.0)" in generated_code
    assert (
        "return Mixed(value=3.0, vectorValue=SIMD[DType.float32, 2](0.0, 0.0))"
        in generated_code
    )
    assert "Pair(y=2.0)" not in generated_code


def test_braced_struct_constructors_zero_fill_nested_struct_and_array_fields():
    code = """
    struct Inner {
        float value;
        float2 offset;
    };

    struct Outer {
        Inner inner;
        float weights[2];
        float scalar;
    };

    Outer makePartialOuter() {
        return Outer { scalar: 5.0 };
    }

    Outer makePartialInner() {
        return Outer { inner: Inner { value: 1.0 } };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "return Outer(inner=Inner(0.0, SIMD[DType.float32, 2](0.0, 0.0)), "
        "weights=InlineArray[Float32, 2](0.0, 0.0), scalar=5.0)" in generated_code
    )
    assert (
        "return Outer(inner=Inner(value=1.0, "
        "offset=SIMD[DType.float32, 2](0.0, 0.0)), "
        "weights=InlineArray[Float32, 2](0.0, 0.0), scalar=0.0)" in generated_code
    )
    assert "Outer(scalar=5.0)" not in generated_code
    assert (
        "return Outer(inner=Inner(value=1.0, "
        "offset=SIMD[DType.float32, 2](0.0, 0.0)))" not in generated_code
    )


def _braced_struct_matrix_and_array_struct_zero_fill_source():
    return """
    struct Leaf {
        float weight;
        float2 offset;
    };

    struct MatrixBundle {
        mat2 transform;
        Leaf leaves[2];
        float scalar;
    };

    MatrixBundle makeMatrixBundle() {
        return MatrixBundle { scalar: 7.0 };
    }

    MatrixBundle makePartialLeaves() {
        return MatrixBundle { leaves: { Leaf { weight: 3.0 } } };
    }
    """


def test_braced_struct_constructors_zero_fill_matrix_and_array_struct_fields():
    generated_code = generate_code(
        parse_code(
            tokenize_code(_braced_struct_matrix_and_array_struct_zero_fill_source())
        )
    )

    matrix_zero = (
        "CrossGLMatrixF32C2R2(SIMD[DType.float32, 2](0.0, 0.0), "
        "SIMD[DType.float32, 2](0.0, 0.0))"
    )
    leaf_zero = "Leaf(0.0, SIMD[DType.float32, 2](0.0, 0.0))"
    leaf_partial = "Leaf(weight=3.0, offset=SIMD[DType.float32, 2](0.0, 0.0))"

    assert (
        f"return MatrixBundle(transform={matrix_zero}, "
        f"leaves=InlineArray[Leaf, 2]({leaf_zero}, {leaf_zero}), scalar=7.0)"
        in generated_code
    )
    assert (
        f"return MatrixBundle(transform={matrix_zero}, "
        f"leaves=InlineArray[Leaf, 2]({leaf_partial}, {leaf_zero}), scalar=0.0)"
        in generated_code
    )
    assert "MatrixBundle(scalar=7.0)" not in generated_code
    assert "leaves=InlineArray[Leaf, 2](Leaf(weight=3.0))" not in generated_code


def test_braced_struct_positional_missing_fields_zero_fill_direct_ast():
    codegen = MojoCodeGen()
    codegen.struct_types["Pair"] = {"x": "float", "y": "float"}
    constructor = ConstructorNode(
        PrimitiveType("Pair"),
        [LiteralNode(1.0, PrimitiveType("float"))],
    )

    assert (
        codegen.generate_constructor_node(constructor, "Pair", "return value")
        == "Pair(x=1.0, y=0.0)"
    )


def test_braced_struct_constructors_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct Pair {
        float x;
        float y;
    };

    Pair makePair(float x, float y) {
        return Pair { x: x, y: y };
    }

    Pair makePositionalPair(float x, float y) {
        return Pair { x + 0.0, y + 0.0 };
    }

    float sumPair() {
        Pair pair = Pair { x: 1.0, y: 2.0 };
        return pair.x + pair.y;
    }

    float sumPartialPairs() {
        Pair positional = Pair { x: 8.0 };
        Pair named = Pair { y: 9.0 };
        return positional.x + positional.y + named.x + named.y;
    }

    struct Inner {
        float value;
        float2 offset;
    };

    struct Outer {
        Inner inner;
        float weights[2];
        float scalar;
    };

    Outer makePartialOuter() {
        return Outer { scalar: 5.0 };
    }

    Outer makePartialInner() {
        return Outer { inner: Inner { value: 1.0 } };
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += """
fn main():
    print(makePair(4.0, 5.0).y)
    print(makePositionalPair(6.0, 7.0).x)
    print(sumPair())
    print(sumPartialPairs())
    print(makePartialOuter().inner.value + makePartialOuter().weights[0] + makePartialOuter().scalar)
    print(makePartialInner().inner.value + makePartialInner().inner.offset[0] + makePartialInner().weights[1] + makePartialInner().scalar)
"""

    source_path = tmp_path / "braced_struct_constructors.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "5.0" in result.stdout
    assert "6.0" in result.stdout
    assert "3.0" in result.stdout
    assert "17.0" in result.stdout
    assert "5.0" in result.stdout
    assert "1.0" in result.stdout


def test_braced_struct_matrix_and_array_struct_zero_fill_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    generated_code = generate_code(
        parse_code(
            tokenize_code(_braced_struct_matrix_and_array_struct_zero_fill_source())
        )
    )
    generated_code += """
fn main():
    print(makeMatrixBundle().transform.c0)
    print(makeMatrixBundle().leaves[1].offset)
    print(makeMatrixBundle().scalar)
    print(makePartialLeaves().leaves[0].weight)
    print(makePartialLeaves().leaves[1].weight)
"""

    source_path = tmp_path / "braced_struct_matrix_array_zero_fill.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[0.0, 0.0]" in result.stdout
    assert "7.0" in result.stdout
    assert "3.0" in result.stdout


def test_unit_enums_lower_to_mojo_aliases():
    code = """
    enum Mode {
        Add,
        Multiply = 4,
        Screen
    };

    int choose(Mode mode) {
        if (mode == Mode::Multiply) {
            return int(Mode::Screen);
        }
        return int(Mode::Add);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "alias Mode = Int32" in generated_code
    assert "alias Mode_Add = 0" in generated_code
    assert "alias Mode_Multiply = 4" in generated_code
    assert "alias Mode_Screen = 5" in generated_code
    assert "fn choose(mode: Mode) -> Int32:" in generated_code
    assert "if (mode == Mode_Multiply):" in generated_code
    assert "return Int32(Mode_Screen)" in generated_code
    assert "Mode::" not in generated_code


def test_unit_enums_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    enum Mode {
        Add,
        Multiply = 4,
        Screen
    };

    int choose(Mode mode) {
        if (mode == Mode::Multiply) {
            return int(Mode::Screen);
        }
        return int(Mode::Add);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    print(choose(Mode_Multiply))\n"

    source_path = tmp_path / "unit_enums.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "5" in result.stdout


def test_enum_flag_values_reference_previous_aliases_and_auto_increment():
    code = """
    enum Flags : uint {
        Read = 1 << 0,
        Write = 1 << 1,
        ReadWrite = Read | Write,
        Next
    };

    uint mask() {
        return uint(Flags::ReadWrite | Flags::Next);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "alias Flags = UInt32" in generated_code
    assert "alias Flags_Read = (1 << 0)" in generated_code
    assert "alias Flags_Write = (1 << 1)" in generated_code
    assert "alias Flags_ReadWrite = (Flags_Read | Flags_Write)" in generated_code
    assert "alias Flags_Next = 4" in generated_code
    assert "return UInt32((Flags_ReadWrite | Flags_Next))" in generated_code
    assert "Read | Write" not in generated_code
    assert "Flags::" not in generated_code


def test_enum_flag_values_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    enum Flags : uint {
        Read = 1 << 0,
        Write = 1 << 1,
        ReadWrite = Read | Write,
        Next
    };

    uint mask() {
        return uint(Flags::ReadWrite | Flags::Next);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    print(mask())\n"

    source_path = tmp_path / "enum_flag_values.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "7" in result.stdout


def test_integer_enum_underlying_types_are_preserved():
    code = """
    enum Small : ushort {
        Off,
        On = 2,
        Auto
    };

    ushort pick() {
        return ushort(Small::Auto);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "alias Small = UInt16" in generated_code
    assert "alias Small_Auto = 3" in generated_code
    assert "fn pick() -> UInt16:" in generated_code
    assert "return UInt16(Small_Auto)" in generated_code


def test_payload_enum_variants_lower_to_tagged_mojo_struct_constructors():
    code = """
    enum MaybeInt {
        Some(int),
        Named { value: float, flag: bool },
        None
    };

    MaybeInt makeSome(int value) {
        return MaybeInt::Some(value);
    }

    MaybeInt makeNamed(float value) {
        return MaybeInt::Named(value, true);
    }

    MaybeInt makeNone() {
        return MaybeInt::None;
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "@value\nstruct MaybeInt:" in generated_code
    assert "var variant: Int32" in generated_code
    assert "var Some_0: Int32" in generated_code
    assert "var value: Float32" in generated_code
    assert "var flag: Bool" in generated_code
    assert "alias MaybeInt_Some = 0" in generated_code
    assert "alias MaybeInt_Named = 1" in generated_code
    assert "alias MaybeInt_None = 2" in generated_code
    assert "fn MaybeInt_Some_make(payload0: Int32) -> MaybeInt:" in generated_code
    assert (
        "return MaybeInt(variant=MaybeInt_Some, Some_0=payload0, "
        "value=0.0, flag=False)"
    ) in generated_code
    assert (
        "return MaybeInt(variant=MaybeInt_Named, Some_0=0, "
        "value=payload0, flag=payload1)"
    ) in generated_code
    assert (
        "return MaybeInt(variant=MaybeInt_None, Some_0=0, value=0.0, " "flag=False)"
    ) in generated_code
    assert "return MaybeInt_Some_make(value)" in generated_code
    assert "return MaybeInt_Named_make(value, True)" in generated_code
    assert "return MaybeInt_None_make()" in generated_code


def test_payload_enum_variants_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    enum MaybeInt {
        Some(int),
        Named { count: int, flag: bool },
        None
    };

    MaybeInt makeSome(int value) {
        return MaybeInt::Some(value);
    }

    MaybeInt makeNamed(int value) {
        return MaybeInt::Named(value, true);
    }

    MaybeInt makeNone() {
        return MaybeInt::None;
    }

    int inspect(MaybeInt value) {
        return match value {
            MaybeInt::Some => value.Some_0,
            MaybeInt::Named => value.count,
            _ => -1
        };
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += (
        "\nfn main():\n"
        "    print(inspect(makeSome(7)))\n"
        "    print(inspect(makeNamed(3)))\n"
        "    print(inspect(makeNone()))\n"
    )

    source_path = tmp_path / "payload_enums.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["7", "3", "-1"]


def test_payload_enum_invalid_shapes_are_rejected_for_mojo_codegen():
    code = """
    enum Bad {
        IntValue { value: int },
        FloatValue { value: float }
    };
    """

    with pytest.raises(ValueError, match="payload fields must"):
        generate_code(parse_code(tokenize_code(code)))


def test_payload_enum_constructor_arity_is_validated_for_mojo_codegen():
    code = """
    enum MaybeInt {
        Some(int),
        None
    };

    MaybeInt broken() {
        return MaybeInt::Some();
    }
    """

    with pytest.raises(ValueError, match="MaybeInt::Some expects 1 arguments, got 0"):
        generate_code(parse_code(tokenize_code(code)))


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


def test_return_semantic_metadata_and_validation_for_mojo_codegen():
    vertex_code = """
    shader ReturnPosition {
        vertex {
            vec4 main() @ gl_Position {
                return vec4(0.0);
            }
        }
    }
    """
    vertex_output = generate_code(parse_code(tokenize_code(vertex_code)))

    assert (
        "# CrossGL return semantic: stage=vertex semantic=position "
        "source=gl_Position" in vertex_output
    )

    color_code = """
    shader ReturnColor {
        fragment {
            vec4 main() @ SV_Target1 {
                return vec4(1.0);
            }
        }
    }
    """
    color_output = generate_code(parse_code(tokenize_code(color_code)))

    assert (
        "# CrossGL return semantic: stage=fragment semantic=color(1) "
        "source=SV_Target1" in color_output
    )

    depth_code = """
    shader ReturnDepth {
        fragment {
            float main() @ gl_FragDepth {
                return 0.5;
            }
        }
    }
    """
    depth_output = generate_code(parse_code(tokenize_code(depth_code)))

    assert (
        "# CrossGL return semantic: stage=fragment semantic=depth(any) "
        "source=gl_FragDepth" in depth_output
    )

    invalid_cases = [
        (
            """
            shader BadPositionType {
                vertex {
                    float main() @ gl_Position {
                        return 0.0;
                    }
                }
            }
            """,
            "gl_Position.*vec4",
        ),
        (
            """
            shader BadColorType {
                fragment {
                    vec3 main() @ gl_FragColor {
                        return vec3(0.0);
                    }
                }
            }
            """,
            "gl_FragColor.*vec4",
        ),
        (
            """
            shader BadDepthType {
                fragment {
                    vec4 main() @ gl_FragDepth {
                        return vec4(0.0);
                    }
                }
            }
            """,
            "gl_FragDepth.*float",
        ),
        (
            """
            shader BadStage {
                vertex {
                    vec4 main() @ gl_FragColor {
                        return vec4(0.0);
                    }
                }
            }
            """,
            "gl_FragColor.*vertex stage",
        ),
        (
            """
            shader BadVoidSemantic {
                fragment {
                    void main() @ SV_Target0 {
                    }
                }
            }
            """,
            "SV_Target0.*void return type",
        ),
        (
            """
            shader BadInputOnlyReturn {
                vertex {
                    uint main() @ gl_VertexID {
                        return uint(0);
                    }
                }
            }
            """,
            "gl_VertexID.*input-only",
        ),
    ]

    for shader, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            generate_code(parse_code(tokenize_code(shader)))


def test_struct_semantics_validate_builtin_types_and_stage_context_for_mojo():
    valid_code = """
    shader StructSemantics {
        struct FSOutput {
            vec4 color @ SV_Target1;
            float depth @ gl_FragDepth;
            vec2 uv @ TEXCOORD0;
        };

        fragment {
            FSOutput main() {
                FSOutput output;
                return output;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(valid_code)))

    assert "var color: SIMD[DType.float32, 4]  # color(1)" in generated_code
    assert "var depth: Float32  # depth(any)" in generated_code
    assert "var uv: SIMD[DType.float32, 2]  # texcoord0" in generated_code

    invalid_type = """
    shader BadStructSemanticType {
        struct FSOutput {
            vec3 color @ gl_FragColor;
        };
    }
    """
    with pytest.raises(ValueError, match="gl_FragColor.*vec4"):
        generate_code(parse_code(tokenize_code(invalid_type)))

    invalid_stage = """
    shader BadStructSemanticStage {
        struct FSOutput {
            vec4 position @ gl_Position;
        };

        fragment {
            FSOutput main() {
                FSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="gl_Position.*fragment stage"):
        generate_code(parse_code(tokenize_code(invalid_stage)))


def test_stage_parameter_semantic_validation_for_mojo_codegen():
    vertex_code = """
    shader VertexParameterSemantics {
        vertex {
            vec4 main(
                uint vertexId @ gl_VertexID,
                uint instanceId @ SV_InstanceID,
                vec3 position @ POSITION,
                vec2 uv @ TEXCOORD0
            ) @ gl_Position {
                return vec4(position, 1.0);
            }
        }
    }
    """
    vertex_output = generate_code(parse_code(tokenize_code(vertex_code)))

    assert "vertexId: UInt32  # vertex_id" in vertex_output
    assert "instanceId: UInt32  # instance_id" in vertex_output
    assert "position: SIMD[DType.float32, 4]  # position" in vertex_output
    assert "uv: SIMD[DType.float32, 2]  # texcoord0" in vertex_output

    fragment_code = """
    shader FragmentParameterSemantics {
        fragment {
            vec4 main(
                vec4 coord @ gl_FragCoord,
                bool front @ gl_FrontFacing,
                vec2 point @ gl_PointCoord,
                vec2 uv @ TEXCOORD0
            ) @ gl_FragColor {
                return vec4(uv, point.x, front ? 1.0 : 0.0);
            }
        }
    }
    """
    fragment_output = generate_code(parse_code(tokenize_code(fragment_code)))

    assert "coord: SIMD[DType.float32, 4]  # position" in fragment_output
    assert "front: Bool  # front_facing" in fragment_output
    assert "point: SIMD[DType.float32, 2]  # point_coord" in fragment_output
    assert "uv: SIMD[DType.float32, 2]  # texcoord0" in fragment_output

    fragment_position_code = """
    shader FragmentPositionParameterSemantic {
        fragment {
            vec4 main(vec4 pos @ gl_Position) @ gl_FragColor {
                return pos;
            }
        }
    }
    """
    fragment_position_output = generate_code(
        parse_code(tokenize_code(fragment_position_code))
    )

    assert "pos: SIMD[DType.float32, 4]  # position" in fragment_position_output

    compute_code = """
    shader ComputeParameterSemantics {
        compute {
            void main(
                uvec3 globalId @ gl_GlobalInvocationID,
                uvec3 localId @ SV_GroupThreadID,
                uint groupIndex @ SV_GroupIndex
            ) {
            }
        }
    }
    """
    compute_output = generate_code(parse_code(tokenize_code(compute_code)))

    assert "globalId: SIMD[DType.uint32, 4]  # global_invocation_id" in compute_output
    assert "localId: SIMD[DType.uint32, 4]  # local_invocation_id" in compute_output
    assert "groupIndex: UInt32  # group_index" in compute_output

    invalid_cases = [
        (
            """
            shader DuplicateParameterSemantic {
                fragment {
                    vec4 main(vec2 uv @ TEXCOORD0, vec2 uv2 @ TEXCOORD0) @ gl_FragColor {
                        return vec4(uv + uv2, 0.0, 1.0);
                    }
                }
            }
            """,
            "Conflicting Mojo fragment parameter semantic.*uv2.*texcoord0.*uv",
        ),
        (
            """
            shader OutputOnlyParameterSemantic {
                fragment {
                    vec4 main(vec4 color @ gl_FragColor) @ gl_FragColor {
                        return color;
                    }
                }
            }
            """,
            "gl_FragColor.*output-only",
        ),
        (
            """
            shader BadParameterStage {
                fragment {
                    vec4 main(uint vertexId @ gl_VertexID) @ gl_FragColor {
                        return vec4(float(vertexId));
                    }
                }
            }
            """,
            "gl_VertexID.*fragment stage.*vertex",
        ),
        (
            """
            shader BadFrontFacingType {
                fragment {
                    vec4 main(float front @ gl_FrontFacing) @ gl_FragColor {
                        return vec4(front);
                    }
                }
            }
            """,
            "gl_FrontFacing.*bool",
        ),
        (
            """
            shader BadComputeVectorType {
                compute {
                    void main(vec3 globalId @ gl_GlobalInvocationID) {
                    }
                }
            }
            """,
            "gl_GlobalInvocationID.*integer vec3",
        ),
        (
            """
            shader BadComputeVarying {
                compute {
                    void main(vec2 uv @ TEXCOORD0) {
                    }
                }
            }
            """,
            "TEXCOORD0.*compute stage.*fragment, vertex",
        ),
    ]

    for shader, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            generate_code(parse_code(tokenize_code(shader)))


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


def test_stage_local_resources_are_visible_to_mojo_stage_helpers():
    code = """
    shader StageLocalResourceMojo {
        fragment {
            uniform sampler2D shadowMap;

            vec4 sampleShadow(vec2 uv) {
                return texture(shadowMap, uv);
            }

            vec4 main() @ gl_FragColor {
                return sampleShadow(vec2(0.5, 0.5));
            }
        }

        compute {
            uniform sampler2D shadowMap;

            void main() {
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "# CrossGL resource metadata: name=shadowMap kind=texture" in generated_code
    assert generated_code.count("var shadowMap: Texture2D = Texture2D()") == 1
    assert "fn sampleShadow(uv: SIMD[DType.float32, 2])" in generated_code
    assert "return sample(shadowMap, uv)" in generated_code
    assert "fn sample(tex: Texture2D, coord: SIMD[DType.float32, 2])" in generated_code


@pytest.mark.parametrize(
    "stage_name",
    [
        "geometry",
        "tessellation_control",
        "tessellation_evaluation",
        "task",
        "amplification",
        "object",
        "mesh",
        "ray_generation",
        "ray_intersection",
        "ray_closest_hit",
        "ray_any_hit",
        "ray_miss",
        "ray_callable",
    ],
)
def test_unsupported_shader_stages_are_rejected_for_mojo_codegen(stage_name):
    code = f"""
    shader UnsupportedStage {{
        {stage_name} {{
            void main() {{}}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=rf"Unsupported {stage_name} shader stage for Mojo codegen",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_synchronization_builtins_lower_to_compile_smoke_helpers(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    void sync() {
        barrier();
        workgroupBarrier();
        memoryBarrier();
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "fn _crossgl_workgroup_barrier():" in generated_code
    assert "fn _crossgl_memory_barrier():" in generated_code
    assert generated_code.count("_crossgl_workgroup_barrier()") == 3
    assert generated_code.count("_crossgl_memory_barrier()") == 2
    assert "\n    barrier()" not in generated_code
    assert "workgroupBarrier()" not in generated_code
    assert "memoryBarrier()" not in generated_code
    generated_code += "\nfn main():\n    sync()\n"

    source_path = tmp_path / "synchronization_builtins.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_user_defined_synchronization_names_are_not_lowered():
    code = """
    void barrier() {
    }

    void memoryBarrier() {
    }

    shader SyncMojo {
        compute {
            void main() {
                barrier();
                memoryBarrier();
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "fn barrier() -> None:" in generated_code
    assert "fn memoryBarrier() -> None:" in generated_code
    assert "_crossgl_workgroup_barrier" not in generated_code
    assert "_crossgl_memory_barrier" not in generated_code


def test_direct_sync_nodes_emit_mojo_helpers_without_ast_repr():
    codegen = MojoCodeGen()

    workgroup_stmt = codegen.generate_statement(SyncNode("barrier"), indent=1)
    memory_stmt = codegen.generate_statement(SyncNode("memoryBarrier"))

    assert workgroup_stmt == "    _crossgl_workgroup_barrier()\n"
    assert memory_stmt == "_crossgl_memory_barrier()\n"
    assert "SyncNode(" not in workgroup_stmt
    assert "_crossgl_workgroup_barrier" in codegen.required_sync_helpers
    assert "_crossgl_memory_barrier" in codegen.required_sync_helpers


def test_direct_sync_nodes_reject_unknown_or_argument_forms():
    codegen = MojoCodeGen()

    with pytest.raises(
        ValueError, match="Unsupported Mojo synchronization operation deviceBarrier"
    ):
        codegen.generate_statement(SyncNode("deviceBarrier"))

    with pytest.raises(ValueError, match="arguments are not supported"):
        codegen.generate_statement(
            SyncNode("barrier", [LiteralNode(1, PrimitiveType("int"))])
        )


def test_direct_atomic_op_nodes_emit_mojo_image_atomic_helpers_without_ast_repr():
    codegen = MojoCodeGen()
    image = VariableNode("image", PrimitiveType("uimage2D"))
    signed_image = VariableNode("signedImage", PrimitiveType("iimage2D"))
    pixel = VariableNode("pixel", VectorType(PrimitiveType("int"), 2))
    value = VariableNode("value", PrimitiveType("uint"))
    replacement = VariableNode("replacement", PrimitiveType("uint"))
    signed_value = VariableNode("signedValue", PrimitiveType("int"))

    codegen.register_variable_type("image", "uimage2D")
    codegen.register_variable_type("signedImage", "iimage2D")
    codegen.register_variable_type("pixel", "ivec2")
    codegen.register_variable_type("value", "uint")
    codegen.register_variable_type("replacement", "uint")
    codegen.register_variable_type("signedValue", "int")

    add_call = codegen.generate_expression(
        AtomicOpNode("imageAtomicAdd", image, [pixel, value])
    )
    compare_call = codegen.generate_expression(
        AtomicOpNode("atomicCompareExchange", image, [pixel, replacement, value])
    )
    signed_call = codegen.generate_expression(
        AtomicOpNode("Add", signed_image, [pixel, signed_value])
    )

    assert add_call.startswith("_crossgl_image_atomic_add_UImage2D")
    assert add_call.endswith("(image, pixel, value)")
    assert compare_call.startswith("_crossgl_image_atomic_comp_swap_UImage2D")
    assert compare_call.endswith("(image, pixel, replacement, value)")
    assert signed_call.startswith("_crossgl_image_atomic_add_IImage2D")
    assert signed_call.endswith("(signedImage, pixel, signedValue)")
    assert "AtomicOpNode(" not in add_call
    assert (
        codegen.expression_result_type(
            AtomicOpNode("imageAtomicAdd", image, [pixel, value])
        )
        == "uint"
    )
    assert (
        codegen.expression_result_type(
            AtomicOpNode("Add", signed_image, [pixel, signed_value])
        )
        == "int"
    )
    helper_names = {
        helper["name"] for helper in codegen.required_resource_builtin_helpers.values()
    }
    assert any(
        name.startswith("_crossgl_image_atomic_add_UImage2D") for name in helper_names
    )
    assert any(
        name.startswith("_crossgl_image_atomic_comp_swap_UImage2D")
        for name in helper_names
    )
    assert any(
        name.startswith("_crossgl_image_atomic_add_IImage2D") for name in helper_names
    )


def test_direct_atomic_op_nodes_reject_unsupported_targets_and_operations():
    codegen = MojoCodeGen()
    color_image = VariableNode("colorImage", PrimitiveType("image2D"))
    image = VariableNode("image", PrimitiveType("uimage2D"))
    pixel = VariableNode("pixel", VectorType(PrimitiveType("int"), 2))
    value = VariableNode("value", PrimitiveType("uint"))

    codegen.register_variable_type("colorImage", "image2D")
    codegen.register_variable_type("image", "uimage2D")
    codegen.register_variable_type("pixel", "ivec2")
    codegen.register_variable_type("value", "uint")

    with pytest.raises(ValueError, match="integer image required"):
        codegen.generate_expression(
            AtomicOpNode("atomicAdd", color_image, [pixel, value])
        )

    with pytest.raises(ValueError, match="Unsupported Mojo atomic operation atomicInc"):
        codegen.generate_expression(AtomicOpNode("atomicInc", image, [pixel, value]))


def test_direct_texture_op_nodes_emit_mojo_image_atomic_alias_helpers_without_ast_repr():
    codegen = MojoCodeGen()
    image = VariableNode("image", PrimitiveType("uimage2D"))
    signed_image = VariableNode("signedImage", PrimitiveType("iimage2D"))
    tex = VariableNode("tex", PrimitiveType("sampler2D"))
    pixel = VariableNode("pixel", VectorType(PrimitiveType("int"), 2))
    value = VariableNode("value", PrimitiveType("uint"))
    replacement = VariableNode("replacement", PrimitiveType("uint"))
    signed_value = VariableNode("signedValue", PrimitiveType("int"))

    codegen.register_variable_type("image", "uimage2D")
    codegen.register_variable_type("signedImage", "iimage2D")
    codegen.register_variable_type("tex", "sampler2D")
    codegen.register_variable_type("pixel", "ivec2")
    codegen.register_variable_type("value", "uint")
    codegen.register_variable_type("replacement", "uint")
    codegen.register_variable_type("signedValue", "int")

    add_call = codegen.generate_expression(TextureOpNode("Add", image, [pixel, value]))
    compare_call = codegen.generate_expression(
        TextureOpNode("InterlockedCompareExchange", image, [pixel, replacement, value])
    )
    signed_call = codegen.generate_expression(
        TextureOpNode("atomicMin", signed_image, [pixel, signed_value])
    )

    assert add_call.startswith("_crossgl_image_atomic_add_UImage2D")
    assert add_call.endswith("(image, pixel, value)")
    assert compare_call.startswith("_crossgl_image_atomic_comp_swap_UImage2D")
    assert compare_call.endswith("(image, pixel, replacement, value)")
    assert signed_call.startswith("_crossgl_image_atomic_min_IImage2D")
    assert signed_call.endswith("(signedImage, pixel, signedValue)")
    assert "TextureOpNode(" not in add_call
    assert (
        codegen.expression_result_type(TextureOpNode("Add", image, [pixel, value]))
        == "uint"
    )
    assert (
        codegen.expression_result_type(
            TextureOpNode("atomicMin", signed_image, [pixel, signed_value])
        )
        == "int"
    )
    helper_names = {
        helper["name"] for helper in codegen.required_resource_builtin_helpers.values()
    }
    assert any(
        name.startswith("_crossgl_image_atomic_add_UImage2D") for name in helper_names
    )
    assert any(
        name.startswith("_crossgl_image_atomic_comp_swap_UImage2D")
        for name in helper_names
    )
    assert any(
        name.startswith("_crossgl_image_atomic_min_IImage2D") for name in helper_names
    )

    with pytest.raises(ValueError, match="Unsupported Mojo texture operation Add"):
        codegen.generate_expression(TextureOpNode("Add", tex, [pixel, value]))


def test_direct_buffer_op_nodes_emit_mojo_helpers_without_ast_repr():
    codegen = MojoCodeGen()
    values = VariableNode("values", PrimitiveType("StructuredBuffer<int>"))
    output = VariableNode("output", PrimitiveType("RWStructuredBuffer<int>"))
    appended = VariableNode("appended", PrimitiveType("AppendStructuredBuffer<int>"))
    consumed = VariableNode("consumed", PrimitiveType("ConsumeStructuredBuffer<int>"))
    raw = VariableNode("raw", PrimitiveType("RWByteAddressBuffer"))
    index = VariableNode("index", PrimitiveType("uint"))
    value = VariableNode("value", PrimitiveType("int"))
    count = VariableNode("count", PrimitiveType("uint"))
    raw_vec = VariableNode("rawVec", PrimitiveType("uvec4"))

    codegen.register_variable_type("values", "StructuredBuffer<int>")
    codegen.register_variable_type("output", "RWStructuredBuffer<int>")
    codegen.register_variable_type("appended", "AppendStructuredBuffer<int>")
    codegen.register_variable_type("consumed", "ConsumeStructuredBuffer<int>")
    codegen.register_variable_type("raw", "RWByteAddressBuffer")
    codegen.register_variable_type("index", "uint")
    codegen.register_variable_type("value", "int")
    codegen.register_variable_type("count", "uint")
    codegen.register_variable_type("rawVec", "uvec4")

    load_call = codegen.generate_expression(BufferOpNode("Load", values, [index]))
    store_call = codegen.generate_expression(
        BufferOpNode("Store", output, [index, value])
    )
    append_call = codegen.generate_expression(BufferOpNode("Append", appended, [value]))
    consume_call = codegen.generate_expression(BufferOpNode("Consume", consumed, []))
    dimensions_call = codegen.generate_expression(
        BufferOpNode("GetDimensions", values, [count])
    )
    vector_load_call = codegen.generate_expression(BufferOpNode("Load3", raw, [index]))
    vector_store_call = codegen.generate_expression(
        BufferOpNode("Store4", raw, [index, raw_vec])
    )

    assert load_call == "buffer_load(values, index)"
    assert store_call == "buffer_store(output, index, value)"
    assert append_call == "buffer_append(appended, value)"
    assert consume_call == "buffer_consume(consumed)"
    assert dimensions_call == "buffer_dimensions(values, count)"
    assert vector_load_call == "buffer_load3(raw, index)"
    assert vector_store_call == "buffer_store4(raw, index, rawVec)"
    assert "BufferOpNode(" not in "".join(
        [
            load_call,
            store_call,
            append_call,
            consume_call,
            dimensions_call,
            vector_load_call,
            vector_store_call,
        ]
    )
    assert ("StructuredBuffer", "int") in codegen.required_buffer_load_helpers
    assert ("RWStructuredBuffer", "int") in codegen.required_buffer_store_helpers
    assert ("AppendStructuredBuffer", "int") in codegen.required_buffer_append_helpers
    assert ("ConsumeStructuredBuffer", "int") in codegen.required_buffer_consume_helpers
    assert ("StructuredBuffer", "int") in codegen.required_buffer_dimensions_helpers
    assert (
        "RWByteAddressBuffer",
        3,
    ) in codegen.required_byte_address_vector_load_helpers
    assert (
        "RWByteAddressBuffer",
        4,
    ) in codegen.required_byte_address_vector_store_helpers
    assert (
        codegen.expression_result_type(BufferOpNode("Load", values, [index])) == "int"
    )
    assert (
        codegen.expression_result_type(BufferOpNode("Load3", raw, [index])) == "uvec3"
    )
    assert (
        codegen.expression_result_type(BufferOpNode("Consume", consumed, [])) == "int"
    )
    assert (
        codegen.expression_result_type(BufferOpNode("Store", output, [index, value]))
        == "void"
    )
    assert (
        codegen.expression_result_type(BufferOpNode("Append", appended, [value]))
        == "void"
    )
    assert (
        codegen.expression_result_type(BufferOpNode("GetDimensions", values, [count]))
        == "void"
    )


def test_direct_buffer_op_constructor_contexts_use_single_evaluation_helpers():
    codegen = MojoCodeGen()
    pairs = VariableNode("pairs", PrimitiveType("StructuredBuffer<float2>"))
    consumed = VariableNode(
        "consumed", PrimitiveType("ConsumeStructuredBuffer<float2>")
    )
    scalar_consumed = VariableNode(
        "scalarConsumed", PrimitiveType("ConsumeStructuredBuffer<int>")
    )
    raw = VariableNode("raw", PrimitiveType("RWByteAddressBuffer"))
    index = VariableNode("index", PrimitiveType("uint"))
    uv = VariableNode("uv", PrimitiveType("float2"))

    codegen.register_variable_type("pairs", "StructuredBuffer<float2>")
    codegen.register_variable_type("consumed", "ConsumeStructuredBuffer<float2>")
    codegen.register_variable_type("scalarConsumed", "ConsumeStructuredBuffer<int>")
    codegen.register_variable_type("raw", "RWByteAddressBuffer")
    codegen.register_variable_type("index", "uint")
    codegen.register_variable_type("uv", "float2")

    load = BufferOpNode("Load", pairs, [index])
    consume = BufferOpNode("Consume", consumed, [])
    scalar_consume = BufferOpNode("Consume", scalar_consumed, [])
    reinterpret = FunctionCallNode("asfloat", [BufferOpNode("Load2", raw, [index])])

    load_vector = codegen.generate_expression(FunctionCallNode("float4", [load, uv]))
    consume_vector = codegen.generate_expression(
        FunctionCallNode("float4", [consume, uv])
    )
    reinterpret_vector = codegen.generate_expression(
        FunctionCallNode("float4", [reinterpret, uv])
    )
    load_matrix = codegen.generate_expression(FunctionCallNode("float2x2", [load, uv]))
    consume_diagonal = codegen.generate_expression(
        FunctionCallNode("float2x2", [scalar_consume])
    )

    vector_helper = "_crossgl_construct_f32_4_vf322_01_vf322_01"
    matrix_helper = "_crossgl_construct_matrix_f32_c2_r2_2_vf322_01_vf322_01"
    diagonal_helper = "_crossgl_matrix_diagonal_f32_c2_r2"

    assert load_vector == f"{vector_helper}(buffer_load(pairs, index), uv)"
    assert consume_vector == f"{vector_helper}(buffer_consume(consumed), uv)"
    assert (
        reinterpret_vector == f"{vector_helper}(asfloat(buffer_load2(raw, index)), uv)"
    )
    assert load_matrix == f"{matrix_helper}(buffer_load(pairs, index), uv)"
    assert consume_diagonal.startswith(f"{diagonal_helper}(")
    assert consume_diagonal.count("buffer_consume(scalarConsumed)") == 1

    combined = "\n".join([load_vector, consume_vector, reinterpret_vector, load_matrix])
    assert "buffer_load(pairs, index)[0]" not in combined
    assert "buffer_consume(consumed)[0]" not in combined
    assert "asfloat(buffer_load2(raw, index))[0]" not in combined
    assert ("StructuredBuffer", "float2") in codegen.required_buffer_load_helpers
    assert (
        "ConsumeStructuredBuffer",
        "float2",
    ) in codegen.required_buffer_consume_helpers
    assert (
        "RWByteAddressBuffer",
        2,
    ) in codegen.required_byte_address_vector_load_helpers
    assert any(
        key[0] == "asfloat" and key[2] == "SIMD[DType.float32, 2]"
        for key in codegen.required_reinterpret_helpers
    )
    assert any(
        helper["key"][0] == "DType.float32"
        for helper in codegen.required_constructor_helpers.values()
    )
    assert codegen.required_matrix_constructor_helpers
    assert ("DType.float32", 2, 2) in codegen.required_matrix_diagonal_helpers


def test_direct_buffer_op_nodes_reject_unsupported_operations_and_targets():
    codegen = MojoCodeGen()
    values = VariableNode("values", PrimitiveType("StructuredBuffer<int>"))
    output = VariableNode("output", PrimitiveType("RWStructuredBuffer<int>"))
    raw = VariableNode("raw", PrimitiveType("ByteAddressBuffer"))
    index = VariableNode("index", PrimitiveType("uint"))
    value = VariableNode("value", PrimitiveType("int"))

    codegen.register_variable_type("values", "StructuredBuffer<int>")
    codegen.register_variable_type("output", "RWStructuredBuffer<int>")
    codegen.register_variable_type("raw", "ByteAddressBuffer")
    codegen.register_variable_type("index", "uint")
    codegen.register_variable_type("value", "int")

    with pytest.raises(ValueError, match="Invalid Load2.*byte-address buffer"):
        codegen.generate_expression(BufferOpNode("Load2", values, [index]))

    with pytest.raises(ValueError, match="Unsupported buffer_store.*ByteAddressBuffer"):
        codegen.generate_expression(BufferOpNode("Store", raw, [index, value]))

    with pytest.raises(
        ValueError, match="Unsupported buffer_append.*RWStructuredBuffer"
    ):
        codegen.generate_expression(BufferOpNode("Append", output, [value]))

    with pytest.raises(
        ValueError, match="Unsupported Mojo buffer operation IncrementCounter"
    ):
        codegen.generate_expression(BufferOpNode("IncrementCounter", values, [index]))


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


def test_resource_placeholders_emit_for_sampler_types():
    code = """
    sampler1d ramp;
    sampler1DArray lineArray;
    sampler2D colorMap;
    sampler2darray layerMap;
    sampler3D volumeMap;
    samplerCube cubeMap;
    samplercubearray probeArray;
    sampler linearSampler;

    vec4 sample1D(sampler1D tex, float u) {
        return texture(tex, u);
    }
    vec4 sample1DArray(sampler1DArray tex, vec2 uLayer) {
        return texture(tex, uLayer);
    }
    vec4 sample2D(sampler2D tex, vec2 uv) {
        return texture(tex, uv);
    }
    vec4 sample2DArray(sampler2DArray tex, vec3 uvLayer) {
        return texture(tex, uvLayer);
    }
    vec4 sample3D(sampler3D tex, vec3 uvw) {
        return texture(tex, uvw);
    }
    vec4 sampleCube(samplerCube tex, vec3 dir) {
        return texture(tex, dir);
    }
    vec4 sampleCubeArray(samplerCubeArray tex, vec4 dirLayer) {
        return texture(tex, dirLayer);
    }
    vec4 sampleMip(sampler2D tex, vec2 uv) {
        return textureLod(tex, uv, 1.0);
    }
    vec4 sampleGrad(sampler2D tex, vec2 uv) {
        return textureGrad(tex, uv, uv, uv);
    }
    vec4 sampleSplit(sampler2D tex, sampler state, vec2 uv) {
        return texture(tex, state, uv) +
            textureLod(tex, state, uv, 1.0) +
            textureGrad(tex, state, uv, uv, uv);
    }
    void noop() {}
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct Texture1D:" in generated_code
    assert "struct Texture1DArray:" in generated_code
    assert "struct Texture2D:" in generated_code
    assert "struct Texture2DArray:" in generated_code
    assert "struct Texture3D:" in generated_code
    assert "struct TextureCube:" in generated_code
    assert "struct TextureCubeArray:" in generated_code
    assert "struct Sampler:" in generated_code
    assert "var ramp: Texture1D = Texture1D()" in generated_code
    assert "var lineArray: Texture1DArray = Texture1DArray()" in generated_code
    assert "var colorMap: Texture2D = Texture2D()" in generated_code
    assert "var layerMap: Texture2DArray = Texture2DArray()" in generated_code
    assert "var volumeMap: Texture3D = Texture3D()" in generated_code
    assert "var cubeMap: TextureCube = TextureCube()" in generated_code
    assert "var probeArray: TextureCubeArray = TextureCubeArray()" in generated_code
    assert "var linearSampler: Sampler = Sampler()" in generated_code
    assert "fn sample(tex: Texture1D, coord: Float32)" in generated_code
    assert (
        "fn sample(tex: Texture1DArray, coord: SIMD[DType.float32, 2])"
        in generated_code
    )
    assert "fn sample(tex: Texture2D, coord: SIMD[DType.float32, 2])" in generated_code
    assert (
        "fn sample(tex: Texture2DArray, coord: SIMD[DType.float32, 4])"
        in generated_code
    )
    assert "fn sample(tex: Texture3D, coord: SIMD[DType.float32, 4])" in generated_code
    assert (
        "fn sample(tex: TextureCube, coord: SIMD[DType.float32, 4])" in generated_code
    )
    assert (
        "fn sample(tex: TextureCubeArray, coord: SIMD[DType.float32, 4])"
        in generated_code
    )
    assert (
        "fn sample_lod(tex: Texture2D, coord: SIMD[DType.float32, 2], lod: Float32)"
        in generated_code
    )
    assert (
        "fn sample_grad(tex: Texture2D, coord: SIMD[DType.float32, 2], "
        "ddx: SIMD[DType.float32, 2], ddy: SIMD[DType.float32, 2])" in generated_code
    )
    assert "return sample_lod(tex, uv, 1.0)" in generated_code
    assert "return sample_grad(tex, uv, uv, uv)" in generated_code
    assert "sample(tex, uv)" in generated_code
    assert "sample_lod(tex, uv, 1.0)" in generated_code
    assert "sample_grad(tex, uv, uv, uv)" in generated_code
    assert "sample(tex, state, uv)" not in generated_code
    assert "sample_lod(tex, state, uv, 1.0)" not in generated_code
    assert "sample_grad(tex, state, uv, uv, uv)" not in generated_code
    assert "fn noop() -> None:\n    pass" in generated_code
    assert "sampler1D" not in generated_code
    assert "sampler1DArray" not in generated_code
    assert "sampler2D" not in generated_code
    assert "sampler2DArray" not in generated_code
    assert "sampler3D" not in generated_code
    assert "samplerCube" not in generated_code
    assert "samplerCubeArray" not in generated_code
    assert "sampler linearSampler" not in generated_code
    assert "textureLod" not in generated_code
    assert "textureGrad" not in generated_code


def test_resource_placeholders_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    sampler1d ramp;
    sampler1DArray lineArray;
    sampler2D colorMap;
    sampler2darray layerMap;
    sampler3D volumeMap;
    samplerCube cubeMap;
    samplercubearray probeArray;
    sampler linearSampler;

    vec4 sample1D(sampler1D tex, float u) {
        return texture(tex, u);
    }
    vec4 sample1DArray(sampler1DArray tex, vec2 uLayer) {
        return texture(tex, uLayer);
    }
    vec4 sample2D(sampler2D tex, vec2 uv) {
        return texture(tex, uv);
    }
    vec4 sample2DArray(sampler2DArray tex, vec3 uvLayer) {
        return texture(tex, uvLayer);
    }
    vec4 sample3D(sampler3D tex, vec3 uvw) {
        return texture(tex, uvw);
    }
    vec4 sampleCube(samplerCube tex, vec3 dir) {
        return texture(tex, dir);
    }
    vec4 sampleCubeArray(samplerCubeArray tex, vec4 dirLayer) {
        return texture(tex, dirLayer);
    }
    vec4 sampleMip(sampler2D tex, vec2 uv) {
        return textureLod(tex, uv, 1.0);
    }
    vec4 sampleGrad(sampler2D tex, vec2 uv) {
        return textureGrad(tex, uv, uv, uv);
    }
    vec4 sampleSplit(sampler2D tex, sampler state, vec2 uv) {
        return texture(tex, state, uv) +
            textureLod(tex, state, uv, 1.0) +
            textureGrad(tex, state, uv, uv, uv);
    }
    void noop() {}
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "resource_placeholders.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_hlsl_sampled_texture_aliases_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    Texture2D<float4> colorMap : register(t0, space1);
    Texture2DMS<float4> samples;
    SamplerState linearSampler : register(s0);
    SamplerComparisonState compareSampler : register(s1);
    sampler2DShadow shadowMap;

    vec4 sampleSplit(vec2 uv) {
        return colorMap.Sample(linearSampler, uv);
    }
    vec4 sampleSplitLevel(vec2 uv) {
        return colorMap.SampleLevel(linearSampler, uv, 1.0);
    }
    vec4 sampleSplitGrad(vec2 uv) {
        return colorMap.SampleGrad(linearSampler, uv, uv, uv);
    }
    ivec2 querySplit() {
        return colorMap.GetDimensions();
    }
    float queryMethodLod(vec2 uv) {
        return colorMap.CalculateLevelOfDetail(linearSampler, uv) +
            colorMap.CalculateLevelOfDetailUnclamped(linearSampler, uv);
    }
    vec4 fetchSplit(ivec2 pixel, int sampleIndex) {
        return samples.Load(pixel, sampleIndex);
    }
    float sampleShadow(vec3 uvDepth) {
        return texture(shadowMap, compareSampler, uvDepth);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct Sampler:" in generated_code
    assert "struct Texture2D:" in generated_code
    assert "struct Texture2DMS:" in generated_code
    assert "var colorMap: Texture2D = Texture2D()" in generated_code
    assert "var samples: Texture2DMS = Texture2DMS()" in generated_code
    assert "var linearSampler: Sampler = Sampler()" in generated_code
    assert "var compareSampler: Sampler = Sampler()" in generated_code
    assert (
        "# CrossGL resource metadata: name=linearSampler kind=sampler set=0 "
        "binding=0 binding_source=explicit register=s0" in generated_code
    )
    assert "return sample(colorMap, uv)" in generated_code
    assert "return sample_lod(colorMap, uv, 1.0)" in generated_code
    assert "return sample_grad(colorMap, uv, uv, uv)" in generated_code
    assert "return texture_size(colorMap)" in generated_code
    assert "_crossgl_calculate_level_of_detail_Texture2D" in generated_code
    assert "_crossgl_calculate_level_of_detail_unclamped_Texture2D" in generated_code
    assert "return texel_fetch(samples, pixel, sampleIndex)" in generated_code
    assert "return sample(shadowMap, uvDepth)" in generated_code
    assert "Texture2D<float4>" not in generated_code
    assert "Texture2DMS<float4>" not in generated_code
    assert "SamplerState" not in generated_code
    assert "SamplerComparisonState" not in generated_code
    assert ".Sample(" not in generated_code
    assert ".SampleLevel(" not in generated_code
    assert ".SampleGrad(" not in generated_code
    assert ".GetDimensions(" not in generated_code
    assert ".CalculateLevelOfDetail(" not in generated_code
    assert ".CalculateLevelOfDetailUnclamped(" not in generated_code
    assert ".Load(" not in generated_code

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "hlsl_sampled_texture_aliases.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_hlsl_rw_texture_aliases_emit_mojo_image_helpers():
    code = """
    RWTexture2D<float4> outColor : register(u0, space2);
    RWTexture2D<int> signedImage;
    RWTexture2D<uint> counters;

    float4 readColor(int2 pixel) {
        return outColor.Load(pixel);
    }
    void writeColor(int2 pixel, float4 value) {
        outColor.Store(pixel, value);
    }
    int readSigned(int2 pixel) {
        return signedImage.Load(pixel);
    }
    void writeSigned(int2 pixel, int value) {
        signedImage.Store(pixel, value);
    }
    uint readCounter(int2 pixel) {
        return counters.Load(pixel);
    }
    void writeCounter(int2 pixel, uint value) {
        counters.Store(pixel, value);
    }
    int2 queryOut() {
        return outColor.GetDimensions();
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct Image2DFloat4:" in generated_code
    assert "struct IImage2D:" in generated_code
    assert "struct UImage2D:" in generated_code
    assert "var outColor: Image2DFloat4 = Image2DFloat4()" in generated_code
    assert "var signedImage: IImage2D = IImage2D()" in generated_code
    assert "var counters: UImage2D = UImage2D()" in generated_code
    assert (
        "# CrossGL resource metadata: name=outColor kind=image set=2 "
        "binding=0 binding_source=explicit register=u0" in generated_code
    )
    assert "return image_load(outColor, pixel)" in generated_code
    assert "image_store(outColor, pixel, value)" in generated_code
    assert "return image_load(signedImage, pixel)" in generated_code
    assert "image_store(signedImage, pixel, value)" in generated_code
    assert "return image_load(counters, pixel)" in generated_code
    assert "image_store(counters, pixel, value)" in generated_code
    assert "return image_size(outColor)" in generated_code
    assert "RWTexture2D<float4>" not in generated_code
    assert "RWTexture2D<int>" not in generated_code
    assert "RWTexture2D<uint>" not in generated_code
    assert ".Load(" not in generated_code
    assert ".Store(" not in generated_code
    assert ".GetDimensions(" not in generated_code


def test_hlsl_rw_texture_aliases_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    RWTexture2D<float4> outColor : register(u0, space2);
    RWTexture2D<int> signedImage;
    RWTexture2D<uint> counters;

    float4 readColor(int2 pixel) {
        return outColor.Load(pixel);
    }
    void writeColor(int2 pixel, float4 value) {
        outColor.Store(pixel, value);
    }
    int readSigned(int2 pixel) {
        return signedImage.Load(pixel);
    }
    void writeSigned(int2 pixel, int value) {
        signedImage.Store(pixel, value);
    }
    uint readCounter(int2 pixel) {
        return counters.Load(pixel);
    }
    void writeCounter(int2 pixel, uint value) {
        counters.Store(pixel, value);
    }
    int2 queryOut() {
        return outColor.GetDimensions();
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "hlsl_rw_texture_aliases.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_hlsl_rasterizer_ordered_texture_aliases_emit_mojo_image_helpers():
    code = """
    RasterizerOrderedTexture1D<float> row;
    RasterizerOrderedTexture1DArray<int> signedRows;
    RasterizerOrderedTexture2D<uint> counters : register(u0, space3);
    RasterizerOrderedTexture2DArray<float4> layers;
    RasterizerOrderedTexture3D<int> volume;

    float readRow(int pixel) {
        return row.Load(pixel);
    }
    void writeRow(int pixel, float value) {
        row.Store(pixel, value);
    }
    int readSigned(int2 pixelLayer) {
        return signedRows.Load(pixelLayer);
    }
    void writeSigned(int2 pixelLayer, int value) {
        signedRows.Store(pixelLayer, value);
    }
    uint readCounter(int2 pixel) {
        return counters.Load(pixel);
    }
    void writeCounter(int2 pixel, uint value) {
        counters.Store(pixel, value);
    }
    float4 readLayer(int4 pixelLayer) {
        return layers.Load(pixelLayer);
    }
    void writeLayer(int4 pixelLayer, float4 value) {
        layers.Store(pixelLayer, value);
    }
    int readVolume(int4 voxel) {
        return volume.Load(voxel);
    }
    void writeVolume(int4 voxel, int value) {
        volume.Store(voxel, value);
    }
    int2 queryCounters() {
        return counters.GetDimensions();
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct Image1DFloat:" in generated_code
    assert "struct IImage1DArray:" in generated_code
    assert "struct UImage2D:" in generated_code
    assert "struct Image2DArrayFloat4:" in generated_code
    assert "struct IImage3D:" in generated_code
    assert "var row: Image1DFloat = Image1DFloat()" in generated_code
    assert "var signedRows: IImage1DArray = IImage1DArray()" in generated_code
    assert "var counters: UImage2D = UImage2D()" in generated_code
    assert "var layers: Image2DArrayFloat4 = Image2DArrayFloat4()" in generated_code
    assert "var volume: IImage3D = IImage3D()" in generated_code
    assert (
        "# CrossGL resource metadata: name=counters kind=image set=3 "
        "binding=0 binding_source=explicit register=u0" in generated_code
    )
    assert "return image_load(row, pixel)" in generated_code
    assert "image_store(row, pixel, value)" in generated_code
    assert "return image_load(signedRows, pixelLayer)" in generated_code
    assert "image_store(signedRows, pixelLayer, value)" in generated_code
    assert "return image_load(counters, pixel)" in generated_code
    assert "image_store(counters, pixel, value)" in generated_code
    assert "return image_load(layers, pixelLayer)" in generated_code
    assert "image_store(layers, pixelLayer, value)" in generated_code
    assert "return image_load(volume, voxel)" in generated_code
    assert "image_store(volume, voxel, value)" in generated_code
    assert "return image_size(counters)" in generated_code
    assert "RasterizerOrderedTexture" not in generated_code
    assert ".Load(" not in generated_code
    assert ".Store(" not in generated_code
    assert ".GetDimensions(" not in generated_code


def test_hlsl_typed_float_image_aliases_preserve_mojo_value_signatures():
    code = """
    RWTexture1D<float> scalarImage;
    RWTexture2D<float2> rgImage;
    RWTexture2D<float3> normalImage;
    RasterizerOrderedTexture2DArray<float4> layers;

    float readScalar(int pixel) {
        return scalarImage.Load(pixel);
    }
    void writeScalar(int pixel, float value) {
        scalarImage.Store(pixel, value);
    }
    float2 readRg(int2 pixel) {
        return rgImage.Load(pixel);
    }
    void writeRg(int2 pixel, float2 value) {
        rgImage.Store(pixel, value);
    }
    float3 readNormal(int2 pixel) {
        return normalImage.Load(pixel);
    }
    void writeNormal(int2 pixel, float3 value) {
        normalImage.Store(pixel, value);
    }
    float4 readLayer(int4 pixelLayer) {
        return layers.Load(pixelLayer);
    }
    void writeLayer(int4 pixelLayer, float4 value) {
        layers.Store(pixelLayer, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct Image1DFloat:" in generated_code
    assert "struct Image2DFloat2:" in generated_code
    assert "struct Image2DFloat3:" in generated_code
    assert "struct Image2DArrayFloat4:" in generated_code
    assert "var scalarImage: Image1DFloat = Image1DFloat()" in generated_code
    assert "var rgImage: Image2DFloat2 = Image2DFloat2()" in generated_code
    assert "var normalImage: Image2DFloat3 = Image2DFloat3()" in generated_code
    assert "var layers: Image2DArrayFloat4 = Image2DArrayFloat4()" in generated_code
    assert "struct Image1D:\n" not in generated_code
    assert "struct Image2D:\n" not in generated_code
    assert (
        "fn image_load(image: Image1DFloat, coord: Int32) -> Float32:" in generated_code
    )
    assert (
        "fn image_store(image: Image1DFloat, coord: Int32, value: Float32):"
        in generated_code
    )
    assert (
        "fn image_load(image: Image2DFloat2, coord: SIMD[DType.int32, 2]) -> "
        "SIMD[DType.float32, 2]:" in generated_code
    )
    assert (
        "fn image_store(image: Image2DFloat2, coord: SIMD[DType.int32, 2], "
        "value: SIMD[DType.float32, 2]):" in generated_code
    )
    assert (
        "fn image_load(image: Image2DFloat3, coord: SIMD[DType.int32, 2]) -> "
        "SIMD[DType.float32, 4]:" in generated_code
    )
    assert (
        "fn image_store(image: Image2DFloat3, coord: SIMD[DType.int32, 2], "
        "value: SIMD[DType.float32, 4]):" in generated_code
    )
    assert (
        "fn image_store(image: Image2DArrayFloat4, coord: SIMD[DType.int32, 4], "
        "value: SIMD[DType.float32, 4]):" in generated_code
    )


def test_hlsl_typed_float_image_store_rejects_mismatched_value_width():
    code = """
    RWTexture2D<float2> rgImage;

    void invalidStore(int2 pixel, float4 value) {
        rgImage.Store(pixel, value);
    }
    """

    with pytest.raises(ValueError, match="image_store.*value.*Image2DFloat2"):
        generate_code(parse_code(tokenize_code(code)))


def test_hlsl_typed_integer_image_aliases_preserve_vector_value_signatures():
    code = """
    RWTexture2D<int2> signedPairs;
    RWTexture2D<uint4> counters;

    int2 readSigned(int2 pixel) {
        return signedPairs.Load(pixel);
    }
    void writeSigned(int2 pixel, int2 value) {
        signedPairs.Store(pixel, value);
    }
    uint4 readCounters(int2 pixel) {
        return counters.Load(pixel);
    }
    void writeCounters(int2 pixel, uint4 value) {
        counters.Store(pixel, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct IImage2DInt2:" in generated_code
    assert "struct UImage2DUInt4:" in generated_code
    assert "var signedPairs: IImage2DInt2 = IImage2DInt2()" in generated_code
    assert "var counters: UImage2DUInt4 = UImage2DUInt4()" in generated_code
    assert (
        "fn image_load(image: IImage2DInt2, coord: SIMD[DType.int32, 2]) -> "
        "SIMD[DType.int32, 2]:" in generated_code
    )
    assert (
        "fn image_store(image: IImage2DInt2, coord: SIMD[DType.int32, 2], "
        "value: SIMD[DType.int32, 2]):" in generated_code
    )
    assert (
        "fn image_load(image: UImage2DUInt4, coord: SIMD[DType.int32, 2]) -> "
        "SIMD[DType.uint32, 4]:" in generated_code
    )
    assert (
        "fn image_store(image: UImage2DUInt4, coord: SIMD[DType.int32, 2], "
        "value: SIMD[DType.uint32, 4]):" in generated_code
    )


def test_hlsl_16bit_typed_image_aliases_preserve_mojo_value_signatures():
    code = """
    RWTexture2D<half2> halfPairs;
    RWTexture2D<min16float3> minNormals;
    RWTexture2D<min16int> smallSigned;
    RasterizerOrderedTexture3D<min16uint4> smallCounters;

    half2 readHalf(int2 pixel) {
        return halfPairs.Load(pixel);
    }
    void writeHalf(int2 pixel, half2 value) {
        halfPairs.Store(pixel, value);
    }
    min16float3 readMinNormal(int2 pixel) {
        return minNormals.Load(pixel);
    }
    void writeMinNormal(int2 pixel, min16float3 value) {
        minNormals.Store(pixel, value);
    }
    min16int readSigned(int2 pixel) {
        return smallSigned.Load(pixel);
    }
    void writeSigned(int2 pixel, min16int value) {
        smallSigned.Store(pixel, value);
    }
    min16uint4 readCounters(int4 voxel) {
        return smallCounters.Load(voxel);
    }
    void writeCounters(int4 voxel, min16uint4 value) {
        smallCounters.Store(voxel, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct Image2DHalf2:" in generated_code
    assert "struct Image2DHalf3:" in generated_code
    assert "struct IImage2DInt16:" in generated_code
    assert "struct UImage3DUInt164:" in generated_code
    assert "var halfPairs: Image2DHalf2 = Image2DHalf2()" in generated_code
    assert "var minNormals: Image2DHalf3 = Image2DHalf3()" in generated_code
    assert "var smallSigned: IImage2DInt16 = IImage2DInt16()" in generated_code
    assert "var smallCounters: UImage3DUInt164 = UImage3DUInt164()" in generated_code
    assert (
        "fn image_store(image: Image2DHalf2, coord: SIMD[DType.int32, 2], "
        "value: SIMD[DType.float16, 2]):" in generated_code
    )
    assert (
        "fn image_load(image: Image2DHalf3, coord: SIMD[DType.int32, 2]) -> "
        "SIMD[DType.float16, 4]:" in generated_code
    )
    assert (
        "fn image_store(image: IImage2DInt16, coord: SIMD[DType.int32, 2], "
        "value: Int16):" in generated_code
    )
    assert (
        "fn image_store(image: UImage3DUInt164, coord: SIMD[DType.int32, 4], "
        "value: SIMD[DType.uint16, 4]):" in generated_code
    )


def test_typed_vector_image_atomics_require_scalar_integer_diagnostics():
    code = """
    RWTexture2D<int2> signedPairs;

    int2 invalidAtomic(int2 pixel, int2 value) {
        return signedPairs.InterlockedAdd(pixel, value);
    }
    """

    with pytest.raises(
        ValueError,
        match="image atomic.*scalar integer image required.*IImage2DInt2",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_hlsl_typed_writable_image_arrays_preserve_mojo_element_placeholders():
    code = """
    RWTexture2D<float2> rgImages[2];
    RasterizerOrderedTexture2DArray<uint4> counters[3];

    float2 readRg(int slot, int2 pixel) {
        return rgImages[slot].Load(pixel);
    }
    void writeRg(int slot, int2 pixel, float2 value) {
        rgImages[slot].Store(pixel, value);
    }
    uint4 readCounter(int slot, int4 pixelLayer) {
        return counters[slot].Load(pixelLayer);
    }
    void writeCounter(int slot, int4 pixelLayer, uint4 value) {
        counters[slot].Store(pixelLayer, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct Image2DFloat2:" in generated_code
    assert "struct UImage2DArrayUInt4:" in generated_code
    assert "var rgImages = InlineArray[Image2DFloat2, 2]" in generated_code
    assert "var counters = InlineArray[UImage2DArrayUInt4, 3]" in generated_code
    assert (
        "fn image_load(image: Image2DFloat2, coord: SIMD[DType.int32, 2]) -> "
        "SIMD[DType.float32, 2]:" in generated_code
    )
    assert (
        "fn image_store(image: UImage2DArrayUInt4, coord: SIMD[DType.int32, 4], "
        "value: SIMD[DType.uint32, 4]):" in generated_code
    )
    assert "return image_load(rgImages[int(slot)], pixel)" in generated_code
    assert "image_store(rgImages[int(slot)], pixel, value)" in generated_code
    assert "return image_load(counters[int(slot)], pixelLayer)" in generated_code
    assert "image_store(counters[int(slot)], pixelLayer, value)" in generated_code
    assert "RWTexture2D<float2>" not in generated_code
    assert "RasterizerOrderedTexture2DArray<uint4>" not in generated_code
    assert ".Load(" not in generated_code
    assert ".Store(" not in generated_code


def test_hlsl_typed_writable_image_arrays_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    RWTexture2D<float2> rgImages[2];
    RasterizerOrderedTexture2DArray<uint4> counters[3];

    float2 readRg(int slot, int2 pixel) {
        return rgImages[slot].Load(pixel);
    }
    void writeRg(int slot, int2 pixel, float2 value) {
        rgImages[slot].Store(pixel, value);
    }
    uint4 readCounter(int slot, int4 pixelLayer) {
        return counters[slot].Load(pixelLayer);
    }
    void writeCounter(int slot, int4 pixelLayer, uint4 value) {
        counters[slot].Store(pixelLayer, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "hlsl_typed_writable_image_arrays.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_hlsl_typed_writable_image_array_struct_members_preserve_mojo_placeholders():
    code = """
    struct ResourceGroup {
        RWTexture2D<float2> rgImages[2];
        RasterizerOrderedTexture2DArray<uint4> counters[3];
    };

    float2 readStruct(ResourceGroup group, int slot, int2 pixel) {
        return group.rgImages[slot].Load(pixel);
    }
    void writeStruct(ResourceGroup group, int slot, int2 pixel, float2 value) {
        group.rgImages[slot].Store(pixel, value);
    }
    uint4 readCounterStruct(ResourceGroup group, int slot, int4 pixelLayer) {
        return group.counters[slot].Load(pixelLayer);
    }
    void writeCounterStruct(ResourceGroup group, int slot, int4 pixelLayer, uint4 value) {
        group.counters[slot].Store(pixelLayer, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct ResourceGroup:" in generated_code
    assert "var rgImages: InlineArray[Image2DFloat2, 2]" in generated_code
    assert "var counters: InlineArray[UImage2DArrayUInt4, 3]" in generated_code
    assert (
        "fn image_load(image: Image2DFloat2, coord: SIMD[DType.int32, 2]) -> "
        "SIMD[DType.float32, 2]:" in generated_code
    )
    assert (
        "fn image_store(image: UImage2DArrayUInt4, coord: SIMD[DType.int32, 4], "
        "value: SIMD[DType.uint32, 4]):" in generated_code
    )
    assert "return image_load(group.rgImages[int(slot)], pixel)" in generated_code
    assert "image_store(group.rgImages[int(slot)], pixel, value)" in generated_code
    assert "return image_load(group.counters[int(slot)], pixelLayer)" in generated_code
    assert "image_store(group.counters[int(slot)], pixelLayer, value)" in generated_code
    assert "RWTexture2D<float2>" not in generated_code
    assert "RasterizerOrderedTexture2DArray<uint4>" not in generated_code
    assert ".Load(" not in generated_code
    assert ".Store(" not in generated_code


def test_hlsl_typed_writable_image_array_parameters_preserve_mojo_placeholders():
    code = """
    float2 readParam(RWTexture2D<float2> images[2], int slot, int2 pixel) {
        return images[slot].Load(pixel);
    }
    void writeParam(RWTexture2D<float2> images[2], int slot, int2 pixel, float2 value) {
        images[slot].Store(pixel, value);
    }
    uint4 readCounterParam(RasterizerOrderedTexture2DArray<uint4> counters[3], int slot, int4 pixelLayer) {
        return counters[slot].Load(pixelLayer);
    }
    void writeCounterParam(RasterizerOrderedTexture2DArray<uint4> counters[3], int slot, int4 pixelLayer, uint4 value) {
        counters[slot].Store(pixelLayer, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "fn readParam(images: InlineArray[Image2DFloat2, 2], slot: Int32, "
        "pixel: SIMD[DType.int32, 2]) -> SIMD[DType.float32, 2]:" in generated_code
    )
    assert (
        "fn writeParam(images: InlineArray[Image2DFloat2, 2], slot: Int32, "
        "pixel: SIMD[DType.int32, 2], value: SIMD[DType.float32, 2]) -> None:"
        in generated_code
    )
    assert (
        "fn readCounterParam(counters: InlineArray[UImage2DArrayUInt4, 3], "
        "slot: Int32, pixelLayer: SIMD[DType.int32, 4]) -> "
        "SIMD[DType.uint32, 4]:" in generated_code
    )
    assert "return image_load(images[int(slot)], pixel)" in generated_code
    assert "image_store(images[int(slot)], pixel, value)" in generated_code
    assert "return image_load(counters[int(slot)], pixelLayer)" in generated_code
    assert "image_store(counters[int(slot)], pixelLayer, value)" in generated_code
    assert "RWTexture2D<float2>" not in generated_code
    assert "RasterizerOrderedTexture2DArray<uint4>" not in generated_code
    assert ".Load(" not in generated_code
    assert ".Store(" not in generated_code


@pytest.mark.parametrize(
    ("code", "pattern"),
    [
        (
            """
            float2 invalidRead(writeonly RWTexture2D<float2> images[2], int slot, int2 pixel) {
                return images[slot].Load(pixel);
            }
            """,
            r"imageLoad.*images.*writeonly",
        ),
        (
            """
            void invalidStore(readonly RWTexture2D<float2> images[2], int slot, int2 pixel, float2 value) {
                images[slot].Store(pixel, value);
            }
            """,
            r"imageStore.*images.*readonly",
        ),
        (
            """
            struct Resources {
                writeonly RWTexture2D<float2> images[2];
            };

            Resources resources;

            float2 invalidRead(int slot, int2 pixel) {
                return resources.images[slot].Load(pixel);
            }
            """,
            r"imageLoad.*resources\.images.*writeonly",
        ),
        (
            """
            struct Resources {
                readonly RWTexture2D<float2> images[2];
            };

            Resources resources;

            void invalidStore(int slot, int2 pixel, float2 value) {
                resources.images[slot].Store(pixel, value);
            }
            """,
            r"imageStore.*resources\.images.*readonly",
        ),
    ],
)
def test_typed_writable_image_array_access_qualifiers_apply_to_members_and_params(
    code, pattern
):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(code)))


def test_hlsl_typed_writable_image_array_members_and_parameters_compile_with_mojo(
    tmp_path,
):
    mojo = find_mojo_compiler()

    code = """
    struct ResourceGroup {
        RWTexture2D<float2> rgImages[2];
        RasterizerOrderedTexture2DArray<uint4> counters[3];
    };

    float2 readStruct(ResourceGroup group, int slot, int2 pixel) {
        return group.rgImages[slot].Load(pixel);
    }
    void writeStruct(ResourceGroup group, int slot, int2 pixel, float2 value) {
        group.rgImages[slot].Store(pixel, value);
    }
    uint4 readCounterStruct(ResourceGroup group, int slot, int4 pixelLayer) {
        return group.counters[slot].Load(pixelLayer);
    }
    void writeCounterStruct(ResourceGroup group, int slot, int4 pixelLayer, uint4 value) {
        group.counters[slot].Store(pixelLayer, value);
    }
    float2 readParam(RWTexture2D<float2> images[2], int slot, int2 pixel) {
        return images[slot].Load(pixel);
    }
    void writeParam(RWTexture2D<float2> images[2], int slot, int2 pixel, float2 value) {
        images[slot].Store(pixel, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "hlsl_typed_writable_image_array_members_params.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _typed_image_array_branch_context_source():
    return """
    struct ResourceGroup {
        RWTexture2D<float2> rgImages[2];
        RasterizerOrderedTexture2DArray<uint> counters[2];
    };

    float2 readConditional(
        ResourceGroup group,
        bool useFirst,
        int slot,
        int2 pixel
    ) {
        return useFirst
            ? group.rgImages[0].Load(pixel)
            : group.rgImages[slot].Load(pixel);
    }

    float2 readMatch(
        ResourceGroup group,
        int mode,
        int slot,
        int2 pixel
    ) {
        return match mode {
            0 => group.rgImages[0].Load(pixel),
            _ => group.rgImages[slot].Load(pixel)
        };
    }

    void storeConditional(
        ResourceGroup group,
        bool useFirst,
        int slot,
        int2 pixel,
        float2 value
    ) {
        if (useFirst) {
            group.rgImages[0].Store(pixel, value);
        } else {
            group.rgImages[slot].Store(pixel, value);
        }
    }

    uint updateConditional(
        ResourceGroup group,
        bool useFirst,
        int slot,
        int4 pixelLayer,
        uint value
    ) {
        return useFirst
            ? group.counters[0].InterlockedAdd(pixelLayer, value)
            : group.counters[slot].InterlockedMax(pixelLayer, value);
    }

    uint updateMatch(
        ResourceGroup group,
        int mode,
        int slot,
        int4 pixelLayer,
        uint value
    ) {
        return match mode {
            0 => group.counters[0].InterlockedAdd(pixelLayer, value),
            _ => group.counters[slot].InterlockedExchange(pixelLayer, value)
        };
    }
    """


def test_typed_image_array_branch_contexts_preserve_mojo_placeholders():
    generated_code = generate_code(
        parse_code(tokenize_code(_typed_image_array_branch_context_source()))
    )

    atomic_add = "_crossgl_image_atomic_add_UImage2DArray_SIMD_DType_int32_4_UInt32"
    atomic_max = "_crossgl_image_atomic_max_UImage2DArray_SIMD_DType_int32_4_UInt32"
    atomic_exchange = (
        "_crossgl_image_atomic_exchange_UImage2DArray_SIMD_DType_int32_4_UInt32"
    )

    assert "struct ResourceGroup:" in generated_code
    assert "var rgImages: InlineArray[Image2DFloat2, 2]" in generated_code
    assert "var counters: InlineArray[UImage2DArray, 2]" in generated_code
    assert (
        "fn image_load(image: Image2DFloat2, coord: SIMD[DType.int32, 2]) -> "
        "SIMD[DType.float32, 2]:" in generated_code
    )
    assert (
        "fn image_store(image: Image2DFloat2, coord: SIMD[DType.int32, 2], "
        "value: SIMD[DType.float32, 2]):" in generated_code
    )
    assert f"fn {atomic_add}" in generated_code
    assert f"fn {atomic_max}" in generated_code
    assert f"fn {atomic_exchange}" in generated_code
    assert (
        "return (image_load(group.rgImages[0], pixel) if useFirst else "
        "image_load(group.rgImages[int(slot)], pixel))" in generated_code
    )
    assert (
        "var __cgl_match_value_0: SIMD[DType.float32, 2] = "
        "SIMD[DType.float32, 2](0.0, 0.0)" in generated_code
    )
    assert "__cgl_match_value_0 = image_load(group.rgImages[0], pixel)" in (
        generated_code
    )
    assert (
        "__cgl_match_value_0 = image_load(group.rgImages[int(slot)], pixel)"
        in generated_code
    )
    assert "image_store(group.rgImages[0], pixel, value)" in generated_code
    assert "image_store(group.rgImages[int(slot)], pixel, value)" in generated_code
    assert (
        f"return ({atomic_add}(group.counters[0], pixelLayer, value) if useFirst "
        f"else {atomic_max}(group.counters[int(slot)], pixelLayer, value))"
        in generated_code
    )
    assert "var __cgl_match_value_1: UInt32 = 0" in generated_code
    assert (
        f"__cgl_match_value_1 = {atomic_add}(group.counters[0], pixelLayer, value)"
        in generated_code
    )
    assert (
        f"__cgl_match_value_1 = {atomic_exchange}("
        "group.counters[int(slot)], pixelLayer, value)" in generated_code
    )
    assert ".Load(" not in generated_code
    assert ".Store(" not in generated_code
    assert ".Interlocked" not in generated_code
    assert "MatchNode" not in generated_code


def test_typed_image_array_branch_contexts_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    generated_code = generate_code(
        parse_code(tokenize_code(_typed_image_array_branch_context_source()))
    )
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "typed_image_array_branch_contexts.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            struct ResourceGroup {
                RWTexture2D<float2> rgImages[2];
            };

            float2 invalidLoad(
                ResourceGroup group,
                bool useFirst,
                int slot,
                float2 pixel
            ) {
                return useFirst
                    ? group.rgImages[0].Load(pixel)
                    : group.rgImages[slot].Load(pixel);
            }
            """,
            r"image_load.*coordinate.*Image2DFloat2",
        ),
        (
            """
            struct ResourceGroup {
                RWTexture2D<float2> rgImages[2];
            };

            void invalidStore(
                ResourceGroup group,
                int mode,
                int slot,
                int2 pixel,
                float4 value
            ) {
                match mode {
                    0 => {
                        group.rgImages[0].Store(pixel, value);
                    }
                    _ => {
                        group.rgImages[slot].Store(pixel, value);
                    }
                }
            }
            """,
            r"image_store.*value.*Image2DFloat2",
        ),
        (
            """
            struct ResourceGroup {
                RWTexture2D<uint4> counters[2];
            };

            uint4 invalidAtomic(
                ResourceGroup group,
                int mode,
                int slot,
                int2 pixel,
                uint4 value
            ) {
                return match mode {
                    0 => group.counters[0].InterlockedAdd(pixel, value),
                    _ => group.counters[slot].InterlockedAdd(pixel, value)
                };
            }
            """,
            r"image atomic.*scalar integer image required.*UImage2DUInt4",
        ),
    ],
)
def test_typed_image_array_branch_context_diagnostics_for_mojo_codegen(source, pattern):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            RWTexture2D<float2> image;

            float4 invalidReturn(int2 pixel) {
                return image.Load(pixel);
            }
            """,
            r"image_load target.*return value.*expects vec2.*Image2DFloat2.*got float4",
        ),
        (
            """
            RWTexture2D<float2> image;

            float4 invalidDeclaration(int2 pixel) {
                float4 color = image.Load(pixel);
                return color;
            }
            """,
            r"image_load target.*declaration color.*expects vec2.*got float4",
        ),
        (
            """
            RWTexture2D<float2> image;

            void invalidAssignment(int2 pixel) {
                float4 color;
                color = image.Load(pixel);
            }
            """,
            r"image_load target.*assignment target.*expects vec2.*got float4",
        ),
        (
            """
            RWTexture2D<float2> image;

            float4 invalidTernary(bool useFirst, int2 pixel) {
                return useFirst ? image.Load(pixel) : image.Load(pixel);
            }
            """,
            r"image_load target.*return value true branch.*expects vec2.*got float4",
        ),
        (
            """
            RWTexture2D<float2> image;

            float4 invalidMatch(int mode, int2 pixel) {
                return match mode {
                    0 => image.Load(pixel),
                    _ => image.Load(pixel)
                };
            }
            """,
            r"image_load target.*return value match arm 1.*expects vec2.*got float4",
        ),
        (
            """
            RWTexture2D<float2> image;

            void acceptColor(float4 color) {
            }

            void invalidArgument(int2 pixel) {
                acceptColor(image.Load(pixel));
            }
            """,
            r"image_load target.*argument 1 for acceptColor.*expects vec2.*got float4",
        ),
        (
            """
            struct Pixel {
                float4 color;
            };

            RWTexture2D<float2> image;

            Pixel invalidStructField(int2 pixel) {
                return Pixel { color: image.Load(pixel) };
            }
            """,
            (
                r"image_load target.*return value field Pixel\.color"
                r".*expects vec2.*got float4"
            ),
        ),
        (
            """
            RWTexture2D<float2> image;

            float4 invalidVectorConstructor(int2 pixel) {
                return float4(image.Load(pixel));
            }
            """,
            (
                r"image_load target.*vector constructor float4 argument 1"
                r".*expects vec2.*got float4"
            ),
        ),
        (
            """
            RWTexture2D<uint> image;

            float invalidAtomicReturn(int2 pixel, uint value) {
                return image.InterlockedAdd(pixel, value);
            }
            """,
            r"image_atomic_add target.*return value.*expects uint.*UImage2D.*got float",
        ),
        (
            """
            RWTexture2D<uint> image;

            float invalidAtomicDeclaration(int2 pixel, uint value) {
                float old = image.InterlockedAdd(pixel, value);
                return old;
            }
            """,
            r"image_atomic_add target.*declaration old.*expects uint.*got float",
        ),
    ],
)
def test_image_result_target_context_diagnostics_for_mojo_codegen(source, pattern):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def test_image_atomic_explicit_scalar_cast_remains_allowed_for_mojo_codegen():
    code = """
    RWTexture2D<uint> image;

    float castAtomic(int2 pixel, uint value) {
        return float(image.InterlockedAdd(pixel, value));
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "return Float32(_crossgl_image_atomic_add_UImage2D_SIMD_DType_int32_2_UInt32"
        in generated_code
    )


@pytest.mark.parametrize(
    ("resource_type", "value_type"),
    [
        ("RWTexture2DMS<float4>", "float4"),
        ("RWTexture2DMSArray<uint>", "uint"),
        ("RasterizerOrderedTexture2DMS<float4>", "float4"),
        ("RasterizerOrderedTexture2DMSArray<int>", "int"),
    ],
)
def test_hlsl_multisample_writable_texture_aliases_emit_unsupported_diagnostic(
    resource_type, value_type
):
    code = f"""
    {resource_type} msImage;

    {value_type} invalidRead(int2 pixel, int sampleIndex) {{
        return msImage.Load(pixel, sampleIndex);
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported writable texture resource for Mojo codegen; "
            "multisample writable texture alias is not supported"
        ),
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_hlsl_rasterizer_ordered_texture_aliases_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    RasterizerOrderedTexture1D<float> row;
    RasterizerOrderedTexture1DArray<int> signedRows;
    RasterizerOrderedTexture2D<uint> counters : register(u0, space3);
    RasterizerOrderedTexture2DArray<float4> layers;
    RasterizerOrderedTexture3D<int> volume;

    float readRow(int pixel) {
        return row.Load(pixel);
    }
    void writeRow(int pixel, float value) {
        row.Store(pixel, value);
    }
    int readSigned(int2 pixelLayer) {
        return signedRows.Load(pixelLayer);
    }
    void writeSigned(int2 pixelLayer, int value) {
        signedRows.Store(pixelLayer, value);
    }
    uint readCounter(int2 pixel) {
        return counters.Load(pixel);
    }
    void writeCounter(int2 pixel, uint value) {
        counters.Store(pixel, value);
    }
    float4 readLayer(int4 pixelLayer) {
        return layers.Load(pixelLayer);
    }
    void writeLayer(int4 pixelLayer, float4 value) {
        layers.Store(pixelLayer, value);
    }
    int readVolume(int4 voxel) {
        return volume.Load(voxel);
    }
    void writeVolume(int4 voxel, int value) {
        volume.Store(voxel, value);
    }
    int2 queryCounters() {
        return counters.GetDimensions();
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "hlsl_rasterizer_ordered_texture_aliases.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_hlsl_rasterizer_ordered_buffer_aliases_emit_mojo_buffer_helpers():
    code = """
    RasterizerOrderedBuffer<uint> bins : register(u2, space1);
    RasterizerOrderedStructuredBuffer<int> values;
    RasterizerOrderedByteAddressBuffer rawBytes;
    RasterizerOrderedByteAddressBuffer rawArray[2];

    uint readBin(uint index) {
        return bins.Load(index);
    }
    void writeBin(uint index, uint value) {
        bins.Store(index, value);
    }
    int readValue(uint index) {
        return values.Load(index);
    }
    void writeValue(uint index, int value) {
        values.Store(index, value);
    }
    uint readRaw(uint offset) {
        return rawBytes.Load(offset);
    }
    void writeRaw(uint offset, uint value) {
        rawBytes.Store(offset, value);
    }
    uint3 readRaw3(uint offset) {
        return rawBytes.Load3(offset);
    }
    void writeRaw4(uint offset, uint4 value) {
        rawBytes.Store4(offset, value);
    }
    void queryBins(uint count) {
        bins.GetDimensions(count);
    }
    uint updateRawArray(int slot, uint offset) {
        uint previous = rawArray[slot].Load(offset);
        rawArray[slot].Store(offset, previous + uint(1));
        return previous;
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct RWBuffer[T: AnyType]:" in generated_code
    assert "struct RWStructuredBuffer[T: AnyType]:" in generated_code
    assert "struct RWByteAddressBuffer:" in generated_code
    assert "var bins: RWBuffer[UInt32] = RWBuffer[UInt32]()" in generated_code
    assert (
        "var values: RWStructuredBuffer[Int32] = RWStructuredBuffer[Int32]()"
        in generated_code
    )
    assert "var rawBytes: RWByteAddressBuffer = RWByteAddressBuffer()" in (
        generated_code
    )
    assert "InlineArray[RWByteAddressBuffer, 2]" in generated_code
    assert (
        "# CrossGL resource metadata: name=bins kind=buffer set=1 "
        "binding=2 binding_source=explicit register=u2" in generated_code
    )
    assert (
        "fn buffer_load(buffer: RWBuffer[UInt32], index: UInt32) -> UInt32:"
        in generated_code
    )
    assert (
        "fn buffer_store(buffer: RWBuffer[UInt32], index: UInt32, value: UInt32):"
        in generated_code
    )
    assert (
        "fn buffer_load(buffer: RWStructuredBuffer[Int32], index: UInt32) -> Int32:"
        in generated_code
    )
    assert (
        "fn buffer_store(buffer: RWStructuredBuffer[Int32], "
        "index: UInt32, value: Int32):"
    ) in generated_code
    assert (
        "fn buffer_load(buffer: RWByteAddressBuffer, index: UInt32) -> UInt32:"
        in generated_code
    )
    assert (
        "fn buffer_store(buffer: RWByteAddressBuffer, index: UInt32, value: UInt32):"
        in generated_code
    )
    assert (
        "fn buffer_load3(buffer: RWByteAddressBuffer, "
        "index: UInt32) -> SIMD[DType.uint32, 4]:"
    ) in generated_code
    assert (
        "fn buffer_store4(buffer: RWByteAddressBuffer, "
        "index: UInt32, value: SIMD[DType.uint32, 4]):"
    ) in generated_code
    assert (
        "fn buffer_dimensions(buffer: RWBuffer[UInt32], dimensions: UInt32):"
        in generated_code
    )
    assert "return buffer_load(bins, index)" in generated_code
    assert "buffer_store(bins, index, value)" in generated_code
    assert "return buffer_load(values, index)" in generated_code
    assert "buffer_store(values, index, value)" in generated_code
    assert "return buffer_load(rawBytes, offset)" in generated_code
    assert "buffer_store(rawBytes, offset, value)" in generated_code
    assert "return buffer_load3(rawBytes, offset)" in generated_code
    assert "buffer_store4(rawBytes, offset, value)" in generated_code
    assert "var __cgl_dimensions_0 = buffer_dimensions(bins)" in generated_code
    assert "count = UInt32(__cgl_dimensions_0[0])" in generated_code
    assert "var previous: UInt32 = buffer_load(rawArray[int(slot)], offset)" in (
        generated_code
    )
    assert "buffer_store(rawArray[int(slot)], offset, (previous + UInt32(1)))" in (
        generated_code
    )
    assert "RasterizerOrdered" not in generated_code
    assert ".Load(" not in generated_code
    assert ".Store(" not in generated_code
    assert ".GetDimensions(" not in generated_code


def test_hlsl_rasterizer_ordered_buffer_aliases_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    RasterizerOrderedBuffer<uint> bins : register(u2, space1);
    RasterizerOrderedStructuredBuffer<int> values;
    RasterizerOrderedByteAddressBuffer rawBytes;
    RasterizerOrderedByteAddressBuffer rawArray[2];

    uint readBin(uint index) {
        return bins.Load(index);
    }
    void writeBin(uint index, uint value) {
        bins.Store(index, value);
    }
    int readValue(uint index) {
        return values.Load(index);
    }
    void writeValue(uint index, int value) {
        values.Store(index, value);
    }
    uint readRaw(uint offset) {
        return rawBytes.Load(offset);
    }
    void writeRaw(uint offset, uint value) {
        rawBytes.Store(offset, value);
    }
    uint3 readRaw3(uint offset) {
        return rawBytes.Load3(offset);
    }
    void writeRaw4(uint offset, uint4 value) {
        rawBytes.Store4(offset, value);
    }
    void queryBins(uint count) {
        bins.GetDimensions(count);
    }
    uint updateRawArray(int slot, uint offset) {
        uint previous = rawArray[slot].Load(offset);
        rawArray[slot].Store(offset, previous + uint(1));
        return previous;
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "hlsl_rasterizer_ordered_buffer_aliases.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_hlsl_typed_buffer_aliases_emit_mojo_buffer_helpers():
    code = """
    Buffer<float> inputValues;
    RWBuffer<float> outputValues : register(u1, space2);

    float readTyped(uint index) {
        return inputValues.Load(index);
    }
    void writeTyped(uint index, float value) {
        outputValues.Store(index, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct Buffer[T: AnyType]:" in generated_code
    assert "struct RWBuffer[T: AnyType]:" in generated_code
    assert "var inputValues: Buffer[Float32] = Buffer[Float32]()" in generated_code
    assert "var outputValues: RWBuffer[Float32] = RWBuffer[Float32]()" in generated_code
    assert (
        "# CrossGL resource metadata: name=outputValues kind=buffer set=2 "
        "binding=1 binding_source=explicit register=u1" in generated_code
    )
    assert (
        "fn buffer_load(buffer: Buffer[Float32], index: UInt32) -> Float32:"
        in generated_code
    )
    assert (
        "fn buffer_store(buffer: RWBuffer[Float32], index: UInt32, value: Float32):"
        in generated_code
    )
    assert "return buffer_load(inputValues, index)" in generated_code
    assert "buffer_store(outputValues, index, value)" in generated_code
    assert "Buffer<" not in generated_code
    assert "RWBuffer<" not in generated_code
    assert ".Load(" not in generated_code
    assert ".Store(" not in generated_code


def test_hlsl_typed_buffer_aliases_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    Buffer<float> inputValues;
    RWBuffer<float> outputValues : register(u1, space2);

    float readTyped(uint index) {
        return inputValues.Load(index);
    }
    void writeTyped(uint index, float value) {
        outputValues.Store(index, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "hlsl_typed_buffer_aliases.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_resource_query_and_image_placeholders_emit_mojo_helpers():
    code = """
    sampler2D colorMap;
    sampler2DMS msTex;
    image2D colorImage;
    uimage2DMS msCounterImage;
    iimage2D signedImage;
    uimage2D counterImage;

    ivec2 texSize(sampler2D tex) {
        return textureSize(tex, 0);
    }
    int texLevels(sampler2D tex) {
        return textureQueryLevels(tex);
    }
    int texSamples(sampler2DMS tex) {
        return textureSamples(tex);
    }
    int imageMsCount(uimage2DMS image) {
        return imageSamples(image);
    }
    vec4 fetchTex(sampler2D tex, ivec2 pixel) {
        return texelFetch(tex, pixel, 0);
    }
    ivec2 imgSize(image2D image) {
        return imageSize(image);
    }
    vec4 readColor(image2D image, ivec2 pixel) {
        return imageLoad(image, pixel);
    }
    void writeColor(image2D image, ivec2 pixel, vec4 value) {
        imageStore(image, pixel, value);
    }
    int readSigned(iimage2D image, ivec2 pixel) {
        return imageLoad(image, pixel);
    }
    void writeSigned(iimage2D image, ivec2 pixel, int value) {
        imageStore(image, pixel, value);
    }
    uint readCounter(uimage2D image, ivec2 pixel) {
        return imageLoad(image, pixel);
    }
    void writeCounter(uimage2D image, ivec2 pixel, uint value) {
        imageStore(image, pixel, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct Image2D:" in generated_code
    assert "struct IImage2D:" in generated_code
    assert "struct Texture2DMS:" in generated_code
    assert "struct UImage2D:" in generated_code
    assert "struct UImage2DMS:" in generated_code
    assert "var colorImage: Image2D = Image2D()" in generated_code
    assert "var msTex: Texture2DMS = Texture2DMS()" in generated_code
    assert "var msCounterImage: UImage2DMS = UImage2DMS()" in generated_code
    assert "var signedImage: IImage2D = IImage2D()" in generated_code
    assert "var counterImage: UImage2D = UImage2D()" in generated_code
    assert (
        "fn texture_size(tex: Texture2D, lod: Int32) -> SIMD[DType.int32, 2]:"
        in generated_code
    )
    assert "fn texture_query_levels(tex: Texture2D) -> Int32:" in generated_code
    assert "fn texture_samples(tex: Texture2DMS) -> Int32:" in generated_code
    assert "fn image_samples(image: UImage2DMS) -> Int32:" in generated_code
    assert (
        "fn texel_fetch(tex: Texture2D, coord: SIMD[DType.int32, 2], lod: Int32)"
        in generated_code
    )
    assert "fn image_size(image: Image2D) -> SIMD[DType.int32, 2]:" in generated_code
    assert (
        "fn image_load(image: Image2D, coord: SIMD[DType.int32, 2]) -> "
        "SIMD[DType.float32, 4]:" in generated_code
    )
    assert (
        "fn image_load(image: IImage2D, coord: SIMD[DType.int32, 2]) -> Int32:"
        in generated_code
    )
    assert (
        "fn image_load(image: UImage2D, coord: SIMD[DType.int32, 2]) -> UInt32:"
        in generated_code
    )
    assert "return texture_size(tex, 0)" in generated_code
    assert "return texture_query_levels(tex)" in generated_code
    assert "return texture_samples(tex)" in generated_code
    assert "return image_samples(image)" in generated_code
    assert "return texel_fetch(tex, pixel, 0)" in generated_code
    assert "return image_size(image)" in generated_code
    assert "return image_load(image, pixel)" in generated_code
    assert "image_store(image, pixel, value)" in generated_code
    assert "textureSize" not in generated_code
    assert "textureQueryLevels" not in generated_code
    assert "textureSamples" not in generated_code
    assert "imageSamples" not in generated_code
    assert "texelFetch" not in generated_code
    assert "imageLoad" not in generated_code
    assert "imageStore" not in generated_code


def _resource_size_target_shape_source():
    return """
    sampler1D strip;
    sampler2D colorMap;
    sampler2DArray layers;
    image2D colorImage;

    void accept2(ivec2 dims) {
    }

    void accept3(int3 dims) {
    }

    int scalarSize(sampler1D tex) {
        int width = textureSize(tex, 0);
        return width;
    }

    ivec2 assignSize(sampler2D tex, image2D image) {
        ivec2 dims = textureSize(tex, 0);
        dims = imageSize(image);
        accept2(textureSize(tex, 0));
        return dims;
    }

    int3 arraySize(sampler2DArray tex) {
        int3 dims = textureSize(tex, 0);
        accept3(textureSize(tex, 0));
        return dims;
    }
    """


def _nested_resource_size_target_shape_source():
    return """
    struct SizeInfo {
        ivec2 dims;
        int3 layers;
    };

    sampler2D colorMap;
    sampler2DArray layerMap;
    image2D colorImage;

    void acceptDims(ivec2 dims) {
    }

    void acceptInfo(SizeInfo info) {
    }

    ivec2 chooseDims(bool useTexture, sampler2D tex, image2D image) {
        return useTexture ? textureSize(tex, 0) : imageSize(image);
    }

    ivec2[2] collectDims(sampler2D tex, image2D image) {
        return {textureSize(tex, 0), imageSize(image)};
    }

    SizeInfo makeInfo(sampler2D tex, image2D image, sampler2DArray layers) {
        SizeInfo info = SizeInfo{
            dims: imageSize(image),
            layers: textureSize(layers, 0),
        };
        acceptDims(textureSize(tex, 0));
        acceptDims(true ? textureSize(tex, 0) : imageSize(image));
        acceptInfo(SizeInfo{
            dims: textureSize(tex, 0),
            layers: textureSize(layers, 0),
        });
        return info;
    }

    SizeInfo makePositionalInfo(image2D image, sampler2DArray layers) {
        return SizeInfo{imageSize(image), textureSize(layers, 0)};
    }
    """


def _nested_array_resource_size_target_shape_source():
    return """
    sampler2D colorMap;
    image2D colorImage;

    ivec2[2] rowDims(sampler2D tex, image2D image) {
        return {textureSize(tex, 0), imageSize(image)};
    }

    ivec2[2][2] nestedDims(sampler2D tex, image2D image) {
        ivec2[2][2] dims = {
            rowDims(tex, image),
            {imageSize(image), textureSize(tex, 0)}
        };
        return dims;
    }

    ivec2[2][2] directNestedDims(sampler2D tex, image2D image) {
        return {
            {textureSize(tex, 0), imageSize(image)},
            rowDims(tex, image)
        };
    }
    """


def _value_expression_resource_size_target_shape_source():
    return """
    sampler2D colorMap;
    image2D colorImage;

    void acceptGrid(ivec2[2][2] grid) {
    }

    ivec2 chooseByMatch(int mode, sampler2D tex, image2D image) {
        return match mode {
            0 => textureSize(tex, 0),
            _ => imageSize(image)
        };
    }

    void passNestedArrayArgument(sampler2D tex, image2D image) {
        acceptGrid({
            {textureSize(tex, 0)},
            {imageSize(image), textureSize(tex, 0)}
        });
    }
    """


def _nested_match_resource_size_target_shape_source():
    return """
    struct SizeInfo {
        ivec2 dims;
        ivec2 alt;
    };

    sampler2D colorMap;
    image2D colorImage;

    ivec2[2] collectMatchDims(int mode, sampler2D tex, image2D image) {
        return {
            match mode {
                0 => textureSize(tex, 0),
                _ => imageSize(image)
            },
            match mode {
                1 => imageSize(image),
                _ => textureSize(tex, 0)
            }
        };
    }

    ivec2[2] updateMatchDims(int mode, sampler2D tex, image2D image) {
        ivec2[2] dims = {
            match mode {
                0 => textureSize(tex, 0),
                _ => imageSize(image)
            },
            imageSize(image)
        };
        dims = {
            textureSize(tex, 0),
            match mode {
                1 => imageSize(image),
                _ => textureSize(tex, 0)
            }
        };
        return dims;
    }

    SizeInfo makeMatchInfo(int mode, sampler2D tex, image2D image) {
        return SizeInfo{
            dims: match mode {
                0 => textureSize(tex, 0),
                _ => imageSize(image)
            },
            alt: match mode {
                1 => imageSize(image),
                _ => textureSize(tex, 0)
            }
        };
    }
    """


def _constructor_match_resource_size_target_shape_source():
    return """
    sampler1D widthMap;
    sampler2D colorMap;
    sampler2DMS msColorMap;
    image2D colorImage;
    uimage1D rowImage;

    ivec2 vectorMatchConstructor(int mode, sampler2D tex, image2D image) {
        return ivec2(match mode {
            0 => textureSize(tex, 0),
            _ => imageSize(image)
        });
    }

    vec2 scalarMatchConstructor(int mode) {
        return vec2(match mode {
            0 => 1,
            _ => 2
        });
    }

    float scalarCastConstructor(int mode) {
        return float(match mode {
            0 => 1,
            _ => 2
        });
    }

    float scalarResourceMatchConstructor(int mode, sampler1D tex) {
        return float(match mode {
            0 => textureSize(tex, 0),
            _ => 7
        });
    }

    float scalarResourceTernaryConstructor(bool highMip, sampler1D tex) {
        return float(highMip ? textureSize(tex, 1) : 7);
    }

    float scalarImageTernaryConstructor(bool useWidth, uimage1D image) {
        return float(useWidth ? imageSize(image) : 3);
    }

    float scalarQueryLevelsMatchConstructor(int mode, sampler2D tex) {
        return float(match mode {
            0 => textureQueryLevels(tex),
            _ => 1
        });
    }

    float scalarSamplesTernaryConstructor(bool useSamples, sampler2DMS tex) {
        return float(useSamples ? textureSamples(tex) : 1);
    }

    float scalarMemberQueryConstructors(
        Texture1D<float4> strip,
        Texture2D<float4> queryTex,
        Texture2DMS<float4> msTex
    ) {
        return float(strip.GetDimensions()) +
            float(queryTex.textureQueryLevels()) +
            float(msTex.textureSamples());
    }

    mat2 matrixMatchConstructor(int mode) {
        return mat2(match mode {
            0 => 1.0,
            _ => 2.0
        });
    }
    """


def test_resource_size_target_shape_contexts_for_mojo_codegen():
    generated_code = generate_code(
        parse_code(tokenize_code(_resource_size_target_shape_source()))
    )

    assert "fn texture_size(tex: Texture1D, lod: Int32) -> Int32:" in generated_code
    assert (
        "fn texture_size(tex: Texture2D, lod: Int32) -> SIMD[DType.int32, 2]:"
        in generated_code
    )
    assert (
        "fn texture_size(tex: Texture2DArray, lod: Int32) -> " "SIMD[DType.int32, 4]:"
    ) in generated_code
    assert "fn image_size(image: Image2D) -> SIMD[DType.int32, 2]:" in generated_code
    assert "var width: Int32 = texture_size(tex, 0)" in generated_code
    assert "var dims: SIMD[DType.int32, 2] = texture_size(tex, 0)" in generated_code
    assert "dims = image_size(image)" in generated_code
    assert "accept2(texture_size(tex, 0))" in generated_code
    assert "var dims: SIMD[DType.int32, 4] = texture_size(tex, 0)" in generated_code
    assert "accept3(texture_size(tex, 0))" in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code


def test_nested_resource_size_target_shape_contexts_for_mojo_codegen():
    generated_code = generate_code(
        parse_code(tokenize_code(_nested_resource_size_target_shape_source()))
    )

    assert "fn chooseDims" in generated_code
    assert (
        "return (texture_size(tex, 0) if useTexture else image_size(image))"
        in generated_code
    )
    assert (
        "return InlineArray[SIMD[DType.int32, 2], 2]("
        "texture_size(tex, 0), image_size(image))"
    ) in generated_code
    assert "var info: SizeInfo = SizeInfo(" in generated_code
    assert "dims=image_size(image)" in generated_code
    assert "layers=texture_size(layers, 0)" in generated_code
    assert (
        "return SizeInfo(image_size(image), texture_size(layers, 0))" in generated_code
    )
    assert (
        "acceptDims((texture_size(tex, 0) if True else image_size(image)))"
        in generated_code
    )
    assert "acceptInfo(SizeInfo(" in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code


def test_nested_array_resource_size_target_shape_contexts_for_mojo_codegen():
    generated_code = generate_code(
        parse_code(tokenize_code(_nested_array_resource_size_target_shape_source()))
    )

    row_type = "InlineArray[SIMD[DType.int32, 2], 2]"
    nested_type = f"InlineArray[{row_type}, 2]"
    assert (
        f"fn rowDims(tex: Texture2D, image: Image2D) -> {row_type}:" in generated_code
    )
    assert (
        f"fn nestedDims(tex: Texture2D, image: Image2D) -> {nested_type}:"
        in generated_code
    )
    assert f"var dims: {nested_type} = {nested_type}(" in generated_code
    assert (
        f"fn directNestedDims(tex: Texture2D, image: Image2D) -> {nested_type}:"
        in generated_code
    )
    assert (
        f"return {nested_type}({row_type}(texture_size(tex, 0), image_size(image)), "
        "rowDims(tex, image))"
    ) in generated_code
    assert "List[SIMD[DType.int32, 2]]" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code


def test_value_expression_resource_size_target_shape_contexts_for_mojo_codegen():
    generated_code = generate_code(
        parse_code(tokenize_code(_value_expression_resource_size_target_shape_source()))
    )

    row_type = "InlineArray[SIMD[DType.int32, 2], 2]"
    nested_type = f"InlineArray[{row_type}, 2]"
    assert "var __cgl_match_value_0: SIMD[DType.int32, 2]" in generated_code
    assert "__cgl_match_value_0 = texture_size(tex, 0)" in generated_code
    assert "__cgl_match_value_0 = image_size(image)" in generated_code
    assert "return __cgl_match_value_0" in generated_code
    assert (
        f"acceptGrid({nested_type}("
        f"{row_type}(texture_size(tex, 0), SIMD[DType.int32, 2](0, 0)), "
        f"{row_type}(image_size(image), texture_size(tex, 0))))"
    ) in generated_code
    assert "MatchNode" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code


def test_nested_match_resource_size_target_shape_contexts_for_mojo_codegen():
    generated_code = generate_code(
        parse_code(tokenize_code(_nested_match_resource_size_target_shape_source()))
    )

    row_type = "InlineArray[SIMD[DType.int32, 2], 2]"
    assert (
        "fn collectMatchDims(mode: Int32, tex: Texture2D, image: Image2D) "
        f"-> {row_type}:"
    ) in generated_code
    assert "var __cgl_match_value_0: SIMD[DType.int32, 2]" in generated_code
    assert "__cgl_match_value_0 = texture_size(tex, 0)" in generated_code
    assert "__cgl_match_value_0 = image_size(image)" in generated_code
    assert "var __cgl_match_value_1: SIMD[DType.int32, 2]" in generated_code
    assert (
        f"return {row_type}(__cgl_match_value_0, __cgl_match_value_1)" in generated_code
    )
    assert (
        f"var dims: {row_type} = {row_type}(__cgl_match_value_2, " "image_size(image))"
    ) in generated_code
    assert (
        f"dims = {row_type}(texture_size(tex, 0), __cgl_match_value_3)"
        in generated_code
    )
    assert (
        "return SizeInfo(dims=__cgl_match_value_4, alt=__cgl_match_value_5)"
        in generated_code
    )
    assert "MatchNode" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code


def test_constructor_match_resource_size_target_shape_contexts_for_mojo_codegen():
    generated_code = generate_code(
        parse_code(
            tokenize_code(_constructor_match_resource_size_target_shape_source())
        )
    )

    assert "fn vectorMatchConstructor" in generated_code
    assert "var __cgl_match_value_0: SIMD[DType.int32, 2]" in generated_code
    assert "__cgl_match_value_0 = texture_size(tex, 0)" in generated_code
    assert "__cgl_match_value_0 = image_size(image)" in generated_code
    assert (
        "return SIMD[DType.int32, 2](" "__cgl_match_value_0[0], __cgl_match_value_0[1])"
    ) in generated_code
    assert "var __cgl_match_value_1: Int32" in generated_code
    assert (
        "return SIMD[DType.float32, 2](" "(__cgl_match_value_1).cast[DType.float32]())"
    ) in generated_code
    assert "var __cgl_match_value_2: Int32" in generated_code
    assert "return Float32(__cgl_match_value_2)" in generated_code
    assert "var __cgl_match_value_3: Int32" in generated_code
    assert "__cgl_match_value_3 = texture_size(tex, 0)" in generated_code
    assert "__cgl_match_value_3 = 7" in generated_code
    assert "return Float32(__cgl_match_value_3)" in generated_code
    assert "return Float32((texture_size(tex, 1) if highMip else 7))" in generated_code
    assert "return Float32((image_size(image) if useWidth else 3))" in generated_code
    assert "var __cgl_match_value_4: Int32" in generated_code
    assert "__cgl_match_value_4 = texture_query_levels(tex)" in generated_code
    assert "__cgl_match_value_4 = 1" in generated_code
    assert "return Float32(__cgl_match_value_4)" in generated_code
    assert (
        "return Float32((texture_samples(tex) if useSamples else 1))" in generated_code
    )
    assert (
        "return ((Float32(texture_size(strip)) + "
        "Float32(texture_query_levels(queryTex))) + "
        "Float32(texture_samples(msTex)))"
    ) in generated_code
    assert "var __cgl_match_value_5: Float32" in generated_code
    assert "return CrossGLMatrixF32C2R2(" in generated_code
    assert generated_code.count("var __cgl_match_value_") == 6
    assert "MatchNode" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "textureQueryLevels(" not in generated_code
    assert "textureSamples(" not in generated_code


def test_match_expression_requires_final_wildcard_for_mojo_codegen():
    code = """
    shader main {
        compute {
            int main(int mode) {
                return match mode {
                    0 => 1
                    1 => 2
                };
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    with pytest.raises(
        ValueError, match="match expressions must include a final wildcard arm"
    ):
        generate_code(ast)


def test_match_expression_statement_arm_requires_tail_value_for_mojo_codegen():
    code = """
    shader main {
        compute {
            int main(int mode) {
                return match mode {
                    0 => {
                        int selected = 1;
                        selected = selected + 1;
                    }
                    _ => 2
                };
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    with pytest.raises(ValueError, match="every arm must end with a value expression"):
        generate_code(ast)


@pytest.mark.parametrize(
    ("arms_source", "message"),
    [
        (
            """
            _ => 0
            1 => 1
            """,
            "wildcard arm must be final",
        ),
        (
            """
            _ => 0
            _ => 1
            """,
            "multiple wildcard arms cannot be lowered",
        ),
        (
            """
            other => other
            _ => 0
            """,
            "identifier binding patterns cannot be lowered",
        ),
        (
            """
            Some(value) => 1
            _ => 0
            """,
            "constructor patterns cannot be lowered",
        ),
        (
            """
            Point { x } => 1
            _ => 0
            """,
            "struct destructuring patterns cannot be lowered",
        ),
        (
            """
            .. => 1
            _ => 0
            """,
            "range-style patterns cannot be lowered",
        ),
    ],
)
def test_match_expression_unsupported_patterns_are_rejected_for_mojo_codegen(
    arms_source, message
):
    code = f"""
    shader main {{
        compute {{
            int main(int mode) {{
                return match mode {{
                    {arms_source}
                }};
            }}
        }}
    }}
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    with pytest.raises(ValueError, match=message):
        generate_code(ast)


def test_match_expression_lowers_guarded_bindings_and_enum_identifier_patterns():
    code = """
    enum Mode {
        Add,
        Multiply
    };

    int guardedValue(int value) {
        return match value {
            zero if zero == 0 => zero + 1,
            _ => 2
        };
    }

    int guardedStatement(int value) {
        int output = 0;
        match value {
            bound if bound > 0 => {
                output = bound;
            }
            _ => {
                output = -1;
            }
        }
        return output;
    }

    int enumValue(Mode mode) {
        return match mode {
            Mode::Add => 10,
            Mode::Multiply => 20,
            _ => 0
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "if (value == 0):" in generated_code
    assert "__cgl_match_value_0 = (value + 1)" in generated_code
    assert "if (value > 0):" in generated_code
    assert "output = value" in generated_code
    assert "zero +" not in generated_code
    assert "bound" not in generated_code
    assert "if mode == Mode_Add:" in generated_code
    assert "elif mode == Mode_Multiply:" in generated_code
    assert "identifier binding patterns cannot be lowered" not in generated_code
    assert "MatchNode" not in generated_code


def test_match_expression_lowers_payload_enum_destructuring_patterns():
    code = """
    enum MaybeInt {
        Some(int),
        Named { count: int, flag: bool },
        None
    };

    int inspect(MaybeInt value) {
        return match value {
            MaybeInt::Some(payload) if payload > 4 => payload + 1,
            MaybeInt::Named { count, flag } if flag => count,
            MaybeInt::None => -1,
            _ => -2
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "if (value.variant == MaybeInt_Some) and ((value.Some_0 > 4)):"
        in generated_code
    )
    assert "__cgl_match_value_0 = (value.Some_0 + 1)" in generated_code
    assert "elif (value.variant == MaybeInt_Named) and (value.flag):" in generated_code
    assert "__cgl_match_value_0 = value.count" in generated_code
    assert "elif value.variant == MaybeInt_None:" in generated_code
    assert "payload + 1" not in generated_code
    assert "MatchNode" not in generated_code


def test_match_expression_payload_enum_destructuring_patterns_compile_with_mojo(
    tmp_path,
):
    mojo = find_mojo_compiler()

    code = """
    enum MaybeInt {
        Some(int),
        Named { count: int, flag: bool },
        None
    };

    int inspect(MaybeInt value) {
        return match value {
            MaybeInt::Some(payload) if payload > 4 => payload + 1,
            MaybeInt::Named { count, flag } if flag => count,
            MaybeInt::None => -1,
            _ => -2
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += (
        "\nfn main():\n"
        "    print(inspect(MaybeInt_Some_make(5)))\n"
        "    print(inspect(MaybeInt_Some_make(3)))\n"
        "    print(inspect(MaybeInt_Named_make(8, True)))\n"
        "    print(inspect(MaybeInt_None_make()))\n"
    )

    source_path = tmp_path / "payload_enum_destructuring.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["6", "-2", "8", "-1"]


def test_match_expression_allows_exhaustive_payload_enum_patterns_without_wildcard():
    code = """
    enum MaybeInt {
        Some(int),
        Named { count: int, flag: bool },
        None
    };

    int inspect(MaybeInt value) {
        return match value {
            MaybeInt::Some(payload) => payload,
            MaybeInt::Named { count, flag } => count,
            MaybeInt::None => -1
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "if value.variant == MaybeInt_Some:" in generated_code
    assert "elif value.variant == MaybeInt_Named:" in generated_code
    assert "else:\n        __cgl_match_value_0 = (-1)" in generated_code
    assert "match expressions must include a final wildcard arm" not in generated_code


def test_match_expression_rejects_constrained_payload_patterns_without_wildcard():
    code = """
    enum MaybeInt {
        Some(int),
        None
    };

    int inspect(MaybeInt value) {
        return match value {
            MaybeInt::Some(1) => 1,
            MaybeInt::None => 0
        };
    }
    """

    with pytest.raises(
        ValueError,
        match="match expressions must include a final wildcard arm",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_match_expression_allows_final_struct_destructuring_without_wildcard():
    code = """
    struct Point {
        x: int;
        y: int;
    };

    int sum(Point point) {
        return match point {
            Point { x, y } => x + y
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "__cgl_match_value_0 = (point.x + point.y)" in generated_code
    assert "if True:" not in generated_code
    assert "match expressions must include a final wildcard arm" not in generated_code


def test_match_expression_rejects_nonfinal_irrefutable_struct_pattern():
    code = """
    struct Point {
        x: int;
        y: int;
    };

    int pick(Point point) {
        return match point {
            Point { x, y } => x,
            _ => y
        };
    }
    """

    with pytest.raises(ValueError, match="irrefutable pattern must be final"):
        generate_code(parse_code(tokenize_code(code)))


def test_match_expression_guarded_struct_destructuring_is_refutable():
    code = """
    struct Point {
        x: int;
        y: int;
    };

    int pick(Point point) {
        return match point {
            Point { x, y } if x > 0 => x,
            _ => point.y
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "if (point.x > 0):" in generated_code
    assert "__cgl_match_value_0 = point.x" in generated_code
    assert "else:" in generated_code
    assert "__cgl_match_value_0 = point.y" in generated_code
    assert "(True) and" not in generated_code
    assert "Point {" not in generated_code


def test_match_expression_guarded_struct_destructuring_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct Point {
        x: int;
        y: int;
    };

    int pick(Point point) {
        return match point {
            Point { x, y } if x > 0 => x,
            _ => point.y
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += (
        "\nfn main():\n"
        "    print(pick(Point(x=4, y=9)))\n"
        "    print(pick(Point(x=0, y=9)))\n"
    )

    source_path = tmp_path / "guarded_struct_destructuring.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["4", "9"]


def test_match_expression_struct_destructuring_patterns_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct Point {
        x: int;
        y: int;
    };

    int sum(Point point) {
        return match point {
            Point { x, y } => x + y
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += (
        "\nfn main():\n" "    var point = Point(x=4, y=9)\n" "    print(sum(point))\n"
    )

    source_path = tmp_path / "struct_destructuring.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["13"]


def test_generic_payload_enum_variants_lower_to_specialized_mojo_structs():
    code = """
    generic<T, E> struct Result {
        enum ResultType {
            Ok(T),
            Err(E)
        }
        ResultType variant;
    }

    Result<int, int> makeOk(int value) {
        return Result::Ok(value);
    }

    Result<int, int> makeErr(int value) {
        return Result::Err(value);
    }

    int inspect(Result<int, int> value) {
        return match value {
            Result::Ok(ok) => ok,
            Result::Err(err) => err + 10
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "@value\nstruct Result_int_int:" in generated_code
    assert "alias Result_Ok = 0" in generated_code
    assert "alias Result_Err = 1" in generated_code
    assert (
        "fn Result_int_int_Ok_make(payload0: Int32) -> Result_int_int:"
        in generated_code
    )
    assert (
        "fn Result_int_int_Err_make(payload0: Int32) -> Result_int_int:"
        in generated_code
    )
    assert "fn makeOk(value: Int32) -> Result_int_int:" in generated_code
    assert "return Result_int_int_Ok_make(value)" in generated_code
    assert "if value.variant == Result_Ok:" in generated_code
    assert "else:\n        __cgl_match_value_0 = (value.Err_0 + 10)" in generated_code
    assert "Result::" not in generated_code


def test_generic_payload_enum_variants_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    generic<T, E> struct Result {
        enum ResultType {
            Ok(T),
            Err(E)
        }
        ResultType variant;
    }

    Result<int, int> makeOk(int value) {
        return Result::Ok(value);
    }

    Result<int, int> makeErr(int value) {
        return Result::Err(value);
    }

    int inspect(Result<int, int> value) {
        return match value {
            Result::Ok(ok) => ok,
            Result::Err(err) => err + 10
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += (
        "\nfn main():\n"
        "    print(inspect(makeOk(7)))\n"
        "    print(inspect(makeErr(3)))\n"
    )

    source_path = tmp_path / "generic_payload_enum.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["7", "13"]


def _generic_payload_enum_struct_array_payload_source():
    return """
    generic<T, E> struct Result {
        enum ResultType {
            Ok(T),
            Err(E)
        }
        ResultType variant;
    }

    struct Payload {
        int x;
        int values[2];
    };

    Payload makePayload(int x, int first, int second) {
        return Payload { x: x, values: {first, second} };
    }

    Result<Payload, int> makePayloadResult(int x, int first, int second) {
        return Result::Ok(makePayload(x, first, second));
    }

    int inspectPayload(Result<Payload, int> value, int index) {
        return match value {
            Result::Ok(payload) => payload.x + payload.values[index],
            Result::Err(err) => err
        };
    }

    Result<int[2], int> makeArray(int first, int second) {
        return Result::Ok({first, second});
    }

    int inspectArray(Result<int[2], int> value, int index) {
        return match value {
            Result::Ok(values) => values[index],
            Result::Err(err) => err
        };
    }
    """


def test_generic_payload_enum_struct_and_array_payload_bindings_for_mojo_codegen():
    generated_code = generate_code(
        parse_code(tokenize_code(_generic_payload_enum_struct_array_payload_source()))
    )

    assert "struct Result_Payload_int:" in generated_code
    assert "var Ok_0: Payload" in generated_code
    assert "struct Result_int_2_int:" in generated_code
    assert "var Ok_0: InlineArray[Int32, 2]" in generated_code
    assert (
        "Ok_0=Payload(0, InlineArray[Int32, 2](0, 0)), Err_0=payload0)"
        in generated_code
    )
    assert "value.Ok_0.x + value.Ok_0.values[int(index)]" in generated_code
    assert "value.Ok_0[int(index)]" in generated_code
    assert "Ok_0=Payload()" not in generated_code
    assert "value.Ok_0[0]" not in generated_code
    assert "value.Ok_0.values[index]" not in generated_code


def test_generic_payload_enum_struct_and_array_payloads_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    generated_code = generate_code(
        parse_code(tokenize_code(_generic_payload_enum_struct_array_payload_source()))
    )
    generated_code += (
        "\nfn main():\n"
        "    print(inspectPayload(makePayloadResult(4, 5, 6), 1))\n"
        "    print(inspectArray(makeArray(7, 8), 1))\n"
    )

    source_path = tmp_path / "generic_payload_struct_array_payloads.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["10", "8"]


def _generic_payload_enum_resource_struct_payload_source():
    return """
    generic<T, E> struct Result {
        enum ResultType {
            Ok(T),
            Err(E)
        }
        ResultType variant;
    }

    struct ResourceSet {
        sampler2D textures[2];
        sampler state;
        readonly image2D inputs[2];
        writeonly image2D outputs[2];
        RWStructuredBuffer<int> values[2];
        RWByteAddressBuffer rawBuffers[2];
        sampler2D singleTexture;
        readonly image2D singleInput;
        writeonly image2D singleOutput;
        RWStructuredBuffer<int> singleValues;
        RWByteAddressBuffer singleRaw;
    };

    Result<ResourceSet, int> makeResources(ResourceSet resources) {
        return Result::Ok(resources);
    }

    vec4 sampleResult(Result<ResourceSet, int> value, int slot, vec2 uv) {
        return match value {
            Result::Ok(resources) => texture(
                resources.textures[slot],
                resources.state,
                uv
            ),
            Result::Err(_) => vec4(0.0)
        };
    }

    vec4 sampleDirectResult(Result<ResourceSet, int> value, vec2 uv) {
        return match value {
            Result::Ok(resources) => texture(
                resources.singleTexture,
                resources.state,
                uv
            ),
            Result::Err(_) => vec4(0.0)
        };
    }

    vec4 loadImageResult(Result<ResourceSet, int> value, int slot, ivec2 pixel) {
        return match value {
            Result::Ok(resources) => imageLoad(resources.inputs[slot], pixel),
            Result::Err(_) => vec4(0.0)
        };
    }

    vec4 loadDirectImageResult(Result<ResourceSet, int> value, ivec2 pixel) {
        return match value {
            Result::Ok(resources) => imageLoad(resources.singleInput, pixel),
            Result::Err(_) => vec4(0.0)
        };
    }

    int loadBufferResult(Result<ResourceSet, int> value, int slot, uint index) {
        return match value {
            Result::Ok(resources) => resources.values[slot].Load(index),
            Result::Err(err) => err
        };
    }

    int loadDirectBufferResult(Result<ResourceSet, int> value, uint index) {
        return match value {
            Result::Ok(resources) => resources.singleValues.Load(index),
            Result::Err(err) => err
        };
    }

    void storeResult(
        Result<ResourceSet, int> value,
        int slot,
        ivec2 pixel,
        vec4 color,
        uint index,
        int scalar,
        uint offset,
        uint4 data
    ) {
        match value {
            Result::Ok(resources) => {
                imageStore(resources.outputs[slot], pixel, color);
                resources.values[slot].Store(index, scalar);
                resources.rawBuffers[slot].Store4(offset, data);
            }
            Result::Err(_) => {
            }
        }
    }

    void storeDirectResult(
        Result<ResourceSet, int> value,
        ivec2 pixel,
        vec4 color,
        uint index,
        int scalar,
        uint offset,
        uint4 data
    ) {
        match value {
            Result::Ok(resources) => {
                imageStore(resources.singleOutput, pixel, color);
                resources.singleValues.Store(index, scalar);
                resources.singleRaw.Store4(offset, data);
            }
            Result::Err(_) => {
            }
        }
    }

    Result<ResourceSet[2], int> makeResourceArray(ResourceSet sets[2]) {
        return Result::Ok(sets);
    }

    vec4 sampleArrayResult(
        Result<ResourceSet[2], int> value,
        int index,
        int slot,
        vec2 uv
    ) {
        return match value {
            Result::Ok(sets) => texture(
                sets[index].textures[slot],
                sets[index].state,
                uv
            ),
            Result::Err(_) => vec4(0.0)
        };
    }

    int loadArrayBufferResult(
        Result<ResourceSet[2], int> value,
        int index,
        int slot,
        uint bufferIndex
    ) {
        return match value {
            Result::Ok(sets) => sets[index].values[slot].Load(bufferIndex),
            Result::Err(err) => err
        };
    }
    """


def test_generic_payload_enum_resource_struct_payloads_for_mojo_codegen():
    generated_code = generate_code(
        parse_code(
            tokenize_code(_generic_payload_enum_resource_struct_payload_source())
        )
    )

    assert "struct Result_ResourceSet_int:" in generated_code
    assert "var Ok_0: ResourceSet" in generated_code
    assert "struct Result_ResourceSet_2_int:" in generated_code
    assert "var Ok_0: InlineArray[ResourceSet, 2]" in generated_code
    assert (
        "Ok_0=ResourceSet(InlineArray[Texture2D, 2](Texture2D(), Texture2D()), "
        "Sampler(), InlineArray[Image2D, 2](Image2D(), Image2D())" in generated_code
    )
    assert "Ok_0=ResourceSet()" not in generated_code
    assert "sample(value.Ok_0.textures[int(slot)], uv)" in generated_code
    assert "sample(value.Ok_0.singleTexture, uv)" in generated_code
    assert "image_load(value.Ok_0.inputs[int(slot)], pixel)" in generated_code
    assert "image_load(value.Ok_0.singleInput, pixel)" in generated_code
    assert "buffer_load(value.Ok_0.values[int(slot)], index)" in generated_code
    assert "buffer_load(value.Ok_0.singleValues, index)" in generated_code
    assert "image_store(value.Ok_0.outputs[int(slot)], pixel, color)" in generated_code
    assert "image_store(value.Ok_0.singleOutput, pixel, color)" in generated_code
    assert "buffer_store(value.Ok_0.values[int(slot)], index, scalar)" in generated_code
    assert "buffer_store(value.Ok_0.singleValues, index, scalar)" in generated_code
    assert (
        "buffer_store4(value.Ok_0.rawBuffers[int(slot)], offset, data)"
        in generated_code
    )
    assert "buffer_store4(value.Ok_0.singleRaw, offset, data)" in generated_code
    assert "sample(value.Ok_0[int(index)].textures[int(slot)], uv)" in generated_code
    assert (
        "buffer_load(value.Ok_0[int(index)].values[int(slot)], bufferIndex)"
        in generated_code
    )
    assert "fn sample(tex: Texture2D, coord: SIMD[DType.float32, 2])" in (
        generated_code
    )
    assert "fn image_load(image: Image2D, coord: SIMD[DType.int32, 2])" in (
        generated_code
    )
    assert "fn buffer_load(buffer: RWStructuredBuffer[Int32], " in generated_code
    assert "fn buffer_store4(buffer: RWByteAddressBuffer, " in generated_code
    assert "Result<" not in generated_code
    assert ".Load(" not in generated_code
    assert ".Store(" not in generated_code
    assert ".Store4(" not in generated_code


def test_generic_payload_enum_resource_struct_payloads_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    generated_code = generate_code(
        parse_code(
            tokenize_code(_generic_payload_enum_resource_struct_payload_source())
        )
    )
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "generic_payload_resource_struct_payloads.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("code", "pattern"),
    [
        (
            """
            generic<T, E> struct Result {
                enum ResultType {
                    Ok(T),
                    Err(E)
                }
                ResultType variant;
            }

            struct ResourceSet {
                writeonly image2D outputImage;
            };

            vec4 invalidRead(Result<ResourceSet, int> value, ivec2 pixel) {
                return match value {
                    Result::Ok(resources) => imageLoad(resources.outputImage, pixel),
                    Result::Err(_) => vec4(0.0)
                };
            }
            """,
            r"imageLoad.*resources\.outputImage.*writeonly",
        ),
        (
            """
            generic<T, E> struct Result {
                enum ResultType {
                    Ok(T),
                    Err(E)
                }
                ResultType variant;
            }

            struct ResourceSet {
                readonly image2D inputImage;
            };

            void invalidStore(
                Result<ResourceSet, int> value,
                ivec2 pixel,
                vec4 color
            ) {
                match value {
                    Result::Ok(resources) => {
                        imageStore(resources.inputImage, pixel, color);
                    }
                    Result::Err(_) => {
                    }
                }
            }
            """,
            r"imageStore.*resources\.inputImage.*readonly",
        ),
    ],
)
def test_generic_payload_enum_direct_resource_fields_preserve_access_diagnostics(
    code, pattern
):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(code)))


def _generic_payload_enum_direct_resource_payload_source():
    return """
    generic<T, E> struct Result {
        enum ResultType {
            Ok(T),
            Err(E)
        }
        ResultType variant;
    }

    struct ResourceBox {
        image2D image;
        RWStructuredBuffer<int> values;
        int tag;
    };

    readonly image2D readonlyImages[2];

    Result<sampler2D, int> makeTexture(sampler2D texture) {
        return Result::Ok(texture);
    }

    Result<image2D, int> makeImage(image2D image) {
        return Result::Ok(image);
    }

    Result<RWStructuredBuffer<int>, int> makeBuffer(
        RWStructuredBuffer<int> values
    ) {
        return Result::Ok(values);
    }

    Result<sampler2D[2], int> makeTextureArray(sampler2D textures[2]) {
        return Result::Ok(textures);
    }

    Result<image2D[2], int> makeImageArray(image2D images[2]) {
        return Result::Ok(images);
    }

    Result<RWStructuredBuffer<int>[2], int> makeBufferArray(
        RWStructuredBuffer<int> values[2]
    ) {
        return Result::Ok(values);
    }

    Result<ResourceBox, int> makeBox(ResourceBox payload) {
        return Result::Ok(payload);
    }

    ivec2 queryDirectTexture(Result<sampler2D, int> value) {
        return match value {
            Result::Ok(texture) => textureSize(texture, 0),
            Result::Err(_) => ivec2(0)
        };
    }

    ivec2 queryDirectImage(Result<image2D, int> value) {
        return match value {
            Result::Ok(image) => imageSize(image),
            Result::Err(_) => ivec2(0)
        };
    }

    vec4 sampleDirectPayload(
        Result<sampler2D, int> value,
        sampler state,
        vec2 uv
    ) {
        return match value {
            Result::Ok(texture) => texture(texture, state, uv),
            Result::Err(_) => vec4(0.0)
        };
    }

    vec4 loadDirectPayload(Result<image2D, int> value, ivec2 pixel) {
        return match value {
            Result::Ok(image) => imageLoad(image, pixel),
            Result::Err(_) => vec4(0.0)
        };
    }

    void storeDirectImagePayload(
        Result<image2D, int> value,
        ivec2 pixel,
        vec4 color
    ) {
        match value {
            Result::Ok(image) => {
                imageStore(image, pixel, color);
            }
            Result::Err(_) => {
            }
        }
    }

    int loadBufferPayload(Result<RWStructuredBuffer<int>, int> value, uint index) {
        return match value {
            Result::Ok(values) => values.Load(index),
            Result::Err(err) => err
        };
    }

    void storeBufferPayload(
        Result<RWStructuredBuffer<int>, int> value,
        uint index,
        int scalar
    ) {
        match value {
            Result::Ok(values) => {
                values.Store(index, scalar);
            }
            Result::Err(_) => {
            }
        }
    }

    vec4 sampleTextureArrayPayload(
        Result<sampler2D[2], int> value,
        sampler state,
        int slot,
        vec2 uv
    ) {
        return match value {
            Result::Ok(textures) => texture(textures[slot], state, uv),
            Result::Err(_) => vec4(0.0)
        };
    }

    vec4 loadImageArrayPayload(
        Result<image2D[2], int> value,
        int slot,
        ivec2 pixel
    ) {
        return match value {
            Result::Ok(images) => imageLoad(images[slot], pixel),
            Result::Err(_) => vec4(0.0)
        };
    }

    void storeImageArrayPayload(
        Result<image2D[2], int> value,
        int slot,
        ivec2 pixel,
        vec4 color
    ) {
        match value {
            Result::Ok(images) => {
                imageStore(images[slot], pixel, color);
            }
            Result::Err(_) => {
            }
        }
    }

    int loadBufferArrayPayload(
        Result<RWStructuredBuffer<int>[2], int> value,
        int slot,
        uint index
    ) {
        return match value {
            Result::Ok(values) => values[slot].Load(index),
            Result::Err(err) => err
        };
    }

    void storeBufferArrayPayload(
        Result<RWStructuredBuffer<int>[2], int> value,
        int slot,
        uint index,
        int scalar
    ) {
        match value {
            Result::Ok(values) => {
                values[slot].Store(index, scalar);
            }
            Result::Err(_) => {
            }
        }
    }

    void reassignDirectPayloadWrapper() {
        Result<image2D, int> value = Result::Ok(readonlyImages[0]);
        value = Result::Err(0);
    }

    int loadNestedPayload(Result<ResourceBox, int> value, uint index) {
        return match value {
            Result::Ok(ResourceBox { image, values, tag }) => values.Load(index) + tag,
            Result::Err(err) => err
        };
    }

    vec4 loadNestedImagePayload(Result<ResourceBox, int> value, ivec2 pixel) {
        return match value {
            Result::Ok(ResourceBox { image, values, tag }) => imageLoad(image, pixel),
            Result::Err(_) => vec4(0.0)
        };
    }
    """


def test_generic_payload_enum_direct_resource_payloads_for_mojo_codegen():
    generated_code = generate_code(
        parse_code(
            tokenize_code(_generic_payload_enum_direct_resource_payload_source())
        )
    )

    assert "struct Result_sampler2D_int:" in generated_code
    assert "var Ok_0: Texture2D" in generated_code
    assert "struct Result_sampler2D_2_int:" in generated_code
    assert "var Ok_0: InlineArray[Texture2D, 2]" in generated_code
    assert "struct Result_image2D_int:" in generated_code
    assert "var Ok_0: Image2D" in generated_code
    assert "struct Result_image2D_2_int:" in generated_code
    assert "var Ok_0: InlineArray[Image2D, 2]" in generated_code
    assert "struct Result_RWStructuredBuffer_int_int:" in generated_code
    assert "var Ok_0: RWStructuredBuffer[Int32]" in generated_code
    assert "struct Result_RWStructuredBuffer_int_2_int:" in generated_code
    assert "var Ok_0: InlineArray[RWStructuredBuffer[Int32], 2]" in generated_code
    assert "texture_size(value.Ok_0, 0)" in generated_code
    assert "image_size(value.Ok_0)" in generated_code
    assert "sample(value.Ok_0, uv)" in generated_code
    assert "image_load(value.Ok_0, pixel)" in generated_code
    assert "image_store(value.Ok_0, pixel, color)" in generated_code
    assert "buffer_load(value.Ok_0, index)" in generated_code
    assert "buffer_store(value.Ok_0, index, scalar)" in generated_code
    assert "sample(value.Ok_0[int(slot)], uv)" in generated_code
    assert "image_load(value.Ok_0[int(slot)], pixel)" in generated_code
    assert "image_store(value.Ok_0[int(slot)], pixel, color)" in generated_code
    assert "buffer_load(value.Ok_0[int(slot)], index)" in generated_code
    assert "buffer_store(value.Ok_0[int(slot)], index, scalar)" in generated_code
    assert "buffer_load(value.Ok_0.values, index) + value.Ok_0.tag" in generated_code
    assert "image_load(value.Ok_0.image, pixel)" in generated_code
    assert "fn reassignDirectPayloadWrapper() -> None:" in generated_code
    assert "value = Result_image2D_int_Err_make(0)" in generated_code
    assert "fn texture_size(tex: Texture2D, lod: Int32)" in generated_code
    assert "fn image_size(image: Image2D)" in generated_code
    assert "fn image_store(image: Image2D, " in generated_code
    assert "fn buffer_store(buffer: RWStructuredBuffer[Int32], " in generated_code
    assert "Result<" not in generated_code
    assert ".Load(" not in generated_code
    assert ".Store(" not in generated_code


def test_generic_payload_enum_direct_resource_payloads_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    generated_code = generate_code(
        parse_code(
            tokenize_code(_generic_payload_enum_direct_resource_payload_source())
        )
    )
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "generic_payload_direct_resource_payloads.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            generic<T, E> struct Result {
                enum ResultType { Ok(T), Err(E) }
                ResultType variant;
            }
            readonly image2D inputs[2];
            void invalidAssignedStore(ivec2 pixel, vec4 color) {
                Result<image2D, int> value = Result::Err(0);
                value = Result::Ok(inputs[0]);
                match value {
                    Result::Ok(image) => {
                        imageStore(image, pixel, color);
                    }
                    Result::Err(_) => {
                    }
                }
            }
            """,
            r"imageStore.*inputs.*readonly",
        ),
        (
            """
            generic<T, E> struct Result {
                enum ResultType { Ok(T), Err(E) }
                ResultType variant;
            }
            writeonly image2D outputs[2];
            vec4 invalidArrayRead(int slot, ivec2 pixel) {
                Result<image2D[2], int> value = Result::Ok(outputs);
                return match value {
                    Result::Ok(images) => imageLoad(images[slot], pixel),
                    Result::Err(_) => vec4(0.0)
                };
            }
            """,
            r"imageLoad.*outputs.*writeonly",
        ),
        (
            """
            generic<T, E> struct Result {
                enum ResultType { Ok(T), Err(E) }
                ResultType variant;
            }
            readonly RWStructuredBuffer<int> values[2];
            void invalidArrayBufferStore(int slot, uint index, int scalar) {
                Result<RWStructuredBuffer<int>[2], int> value = Result::Ok(values);
                match value {
                    Result::Ok(buffers) => {
                        buffers[slot].Store(index, scalar);
                    }
                    Result::Err(_) => {
                    }
                }
            }
            """,
            r"Store.*values.*readonly",
        ),
    ],
)
def test_generic_payload_enum_local_direct_resource_payload_access_diagnostics(
    source, pattern
):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def _generic_payload_enum_resource_query_source():
    return """
    generic<T, E> struct Result {
        enum ResultType {
            Ok(T),
            Err(E)
        }
        ResultType variant;
    }

    struct QueryResources {
        sampler2D textures[2];
        sampler2DMS msTextures[2];
        readonly image2D images[2];
        readonly image2DMS msImages[2];
        sampler state;
    };

    ivec2 queryTextureSize(Result<QueryResources, int> value, int slot) {
        return match value {
            Result::Ok(resources) => textureSize(resources.textures[slot], 0),
            Result::Err(_) => ivec2(0)
        };
    }

    int queryTextureLevels(Result<QueryResources, int> value, int slot) {
        return match value {
            Result::Ok(resources) => textureQueryLevels(resources.textures[slot]),
            Result::Err(err) => err
        };
    }

    int queryTextureSamples(Result<QueryResources, int> value, int slot) {
        return match value {
            Result::Ok(resources) => textureSamples(resources.msTextures[slot]),
            Result::Err(err) => err
        };
    }

    ivec2 queryImageSize(Result<QueryResources, int> value, int slot) {
        return match value {
            Result::Ok(resources) => imageSize(resources.images[slot]),
            Result::Err(_) => ivec2(0)
        };
    }

    int queryImageSamples(Result<QueryResources, int> value, int slot) {
        return match value {
            Result::Ok(resources) => imageSamples(resources.msImages[slot]),
            Result::Err(err) => err
        };
    }

    vec4 fetchTextureSample(
        Result<QueryResources, int> value,
        int slot,
        ivec2 pixel,
        int sampleIndex
    ) {
        return match value {
            Result::Ok(resources) => texelFetch(
                resources.msTextures[slot],
                pixel,
                sampleIndex
            ),
            Result::Err(_) => vec4(0.0)
        };
    }

    vec2 queryTextureLod(
        Result<QueryResources, int> value,
        int slot,
        vec2 uv
    ) {
        return match value {
            Result::Ok(resources) => textureQueryLod(
                resources.textures[slot],
                resources.state,
                uv
            ),
            Result::Err(_) => vec2(0.0)
        };
    }

    ivec2 queryArrayTextureSize(
        Result<QueryResources[2], int> value,
        int index,
        int slot
    ) {
        return match value {
            Result::Ok(sets) => textureSize(sets[index].textures[slot], 0),
            Result::Err(_) => ivec2(0)
        };
    }

    int queryArrayImageSamples(
        Result<QueryResources[2], int> value,
        int index,
        int slot
    ) {
        return match value {
            Result::Ok(sets) => imageSamples(sets[index].msImages[slot]),
            Result::Err(err) => err
        };
    }
    """


def test_generic_payload_enum_resource_query_bindings_for_mojo_codegen():
    generated_code = generate_code(
        parse_code(tokenize_code(_generic_payload_enum_resource_query_source()))
    )

    assert "struct Result_QueryResources_int:" in generated_code
    assert "var Ok_0: QueryResources" in generated_code
    assert "struct Result_QueryResources_2_int:" in generated_code
    assert "var Ok_0: InlineArray[QueryResources, 2]" in generated_code
    assert (
        "fn texture_size(tex: Texture2D, lod: Int32) -> SIMD[DType.int32, 2]:"
        in generated_code
    )
    assert "fn image_size(image: Image2D) -> SIMD[DType.int32, 2]:" in generated_code
    assert "fn texture_query_levels(tex: Texture2D) -> Int32:" in generated_code
    assert "fn texture_samples(tex: Texture2DMS) -> Int32:" in generated_code
    assert "fn image_samples(image: Image2DMS) -> Int32:" in generated_code
    assert "_crossgl_texture_query_lod_Texture2D" in generated_code
    assert "texture_size(value.Ok_0.textures[int(slot)], 0)" in generated_code
    assert "texture_query_levels(value.Ok_0.textures[int(slot)])" in generated_code
    assert "texture_samples(value.Ok_0.msTextures[int(slot)])" in generated_code
    assert "image_size(value.Ok_0.images[int(slot)])" in generated_code
    assert "image_samples(value.Ok_0.msImages[int(slot)])" in generated_code
    assert (
        "texel_fetch(value.Ok_0.msTextures[int(slot)], pixel, sampleIndex)"
        in generated_code
    )
    assert (
        "_crossgl_texture_query_lod_Texture2D_SIMD_DType_float32_2("
        "value.Ok_0.textures[int(slot)], uv)" in generated_code
    )
    assert (
        "texture_size(value.Ok_0[int(index)].textures[int(slot)], 0)" in generated_code
    )
    assert "image_samples(value.Ok_0[int(index)].msImages[int(slot)])" in generated_code
    assert "textureSize(" not in generated_code
    assert "textureQueryLevels(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageSamples(" not in generated_code
    assert "textureQueryLod(" not in generated_code


def test_generic_payload_enum_resource_query_bindings_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    generated_code = generate_code(
        parse_code(tokenize_code(_generic_payload_enum_resource_query_source()))
    )
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "generic_payload_resource_query_bindings.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            generic<T, E> struct Result {
                enum ResultType { Ok(T), Err(E) }
                ResultType variant;
            }
            struct ResourceSet {
                writeonly image2D outputs[2];
            };
            vec4 invalidRead(
                Result<ResourceSet, int> value,
                int slot,
                ivec2 pixel
            ) {
                return match value {
                    Result::Ok(resources) => imageLoad(
                        resources.outputs[slot],
                        pixel
                    ),
                    Result::Err(_) => vec4(0.0)
                };
            }
            """,
            r"imageLoad.*resources\.outputs.*writeonly",
        ),
        (
            """
            generic<T, E> struct Result {
                enum ResultType { Ok(T), Err(E) }
                ResultType variant;
            }
            struct ResourceSet {
                readonly image2D inputs[2];
            };
            void invalidWrite(
                Result<ResourceSet, int> value,
                int slot,
                ivec2 pixel,
                vec4 color
            ) {
                match value {
                    Result::Ok(resources) => {
                        imageStore(resources.inputs[slot], pixel, color);
                    }
                    Result::Err(_) => {
                    }
                }
            }
            """,
            r"imageStore.*resources\.inputs.*readonly",
        ),
        (
            """
            generic<T, E> struct Result {
                enum ResultType { Ok(T), Err(E) }
                ResultType variant;
            }
            struct ResourceSet {
                readonly RWByteAddressBuffer rawBuffers[2];
            };
            void invalidRawStore(
                Result<ResourceSet, int> value,
                int slot,
                uint offset,
                uint4 data
            ) {
                match value {
                    Result::Ok(resources) => {
                        resources.rawBuffers[slot].Store4(offset, data);
                    }
                    Result::Err(_) => {
                    }
                }
            }
            """,
            r"Store4.*resources\.rawBuffers.*readonly",
        ),
    ],
)
def test_generic_payload_enum_resource_struct_diagnostics_for_mojo_codegen(
    source, pattern
):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            generic<T, E> struct Result {
                enum ResultType { Ok(T), Err(E) }
                ResultType variant;
            }
            struct QueryResources {
                sampler2DMS msTextures[2];
            };
            int invalidLevels(Result<QueryResources, int> value, int slot) {
                return match value {
                    Result::Ok(resources) => textureQueryLevels(
                        resources.msTextures[slot]
                    ),
                    Result::Err(err) => err
                };
            }
            """,
            r"texture_query_levels.*non-multisample texture required: Texture2DMS",
        ),
        (
            """
            generic<T, E> struct Result {
                enum ResultType { Ok(T), Err(E) }
                ResultType variant;
            }
            struct QueryResources {
                image2D images[2];
            };
            int invalidSamples(Result<QueryResources, int> value, int slot) {
                return match value {
                    Result::Ok(resources) => imageSamples(resources.images[slot]),
                    Result::Err(err) => err
                };
            }
            """,
            r"image_samples.*multisample image required: Image2D",
        ),
        (
            """
            generic<T, E> struct Result {
                enum ResultType { Ok(T), Err(E) }
                ResultType variant;
            }
            struct QueryResources {
                sampler2DShadow shadows[2];
                sampler state;
            };
            vec2 invalidQueryLod(
                Result<QueryResources, int> value,
                int slot,
                vec2 uv
            ) {
                return match value {
                    Result::Ok(resources) => textureQueryLod(
                        resources.shadows[slot],
                        resources.state,
                        uv
                    ),
                    Result::Err(_) => vec2(0.0)
                };
            }
            """,
            r"texture_query_lod.*non-shadow texture required: Texture2DShadow",
        ),
    ],
)
def test_generic_payload_enum_resource_query_diagnostics_for_mojo_codegen(
    source, pattern
):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def _helper_returned_resource_alias_source():
    return """
    readonly image2D inputs[2];
    writeonly image2D outputs[2];
    sampler2D textures[2];

    image2D chooseInput(readonly image2D inputImage) {
        return inputImage;
    }

    image2D chooseViaNestedHelper(writeonly image2D outputImage) {
        return chooseOutput(outputImage);
    }

    image2D chooseOutput(writeonly image2D outputImage) {
        return outputImage;
    }

    sampler2D chooseTexture(sampler2D tex) {
        return tex;
    }

    vec4 readReturnedAlias(int slot, ivec2 pixel) {
        image2D selected = chooseInput(inputs[slot]);
        return imageLoad(selected, pixel);
    }

    void writeReturnedAlias(int slot, ivec2 pixel, vec4 color) {
        image2D selected = chooseViaNestedHelper(outputs[slot]);
        imageStore(selected, pixel, color);
    }

    ivec2 queryReturnedTexture(int slot) {
        return textureSize(chooseTexture(textures[slot]), 0);
    }
    """


def test_helper_returned_resource_aliases_preserve_mojo_qualifiers_and_compile(
    tmp_path,
):
    mojo = find_mojo_compiler()

    generated_code = generate_code(
        parse_code(tokenize_code(_helper_returned_resource_alias_source()))
    )

    assert "fn chooseInput(inputImage: Image2D) -> Image2D:" in generated_code
    assert "fn chooseViaNestedHelper(outputImage: Image2D) -> Image2D:" in (
        generated_code
    )
    assert "var selected: Image2D = chooseInput(inputs[int(slot)])" in generated_code
    assert "return image_load(selected, pixel)" in generated_code
    assert (
        "var selected: Image2D = chooseViaNestedHelper(outputs[int(slot)])"
        in generated_code
    )
    assert "image_store(selected, pixel, color)" in generated_code
    assert "return texture_size(chooseTexture(textures[int(slot)]), 0)" in (
        generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code
    assert "textureSize(" not in generated_code

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "helper_returned_resource_aliases.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            writeonly image2D outputs[2];

            image2D chooseOutput(writeonly image2D outputImage) {
                return outputImage;
            }

            vec4 invalidRead(int slot, ivec2 pixel) {
                image2D selected = chooseOutput(outputs[slot]);
                return imageLoad(selected, pixel);
            }
            """,
            r"imageLoad.*outputs.*writeonly",
        ),
        (
            """
            writeonly image2D outputs[2];

            image2D chooseOutput(writeonly image2D outputImage) {
                return outputImage;
            }

            vec4 invalidDirectRead(int slot, ivec2 pixel) {
                return imageLoad(chooseOutput(outputs[slot]), pixel);
            }
            """,
            r"imageLoad.*outputs.*writeonly",
        ),
        (
            """
            readonly image2D inputs[2];

            image2D chooseInput(readonly image2D inputImage) {
                return inputImage;
            }

            void invalidWrite(int slot, ivec2 pixel, vec4 color) {
                image2D selected = chooseInput(inputs[slot]);
                imageStore(selected, pixel, color);
            }
            """,
            r"imageStore.*inputs.*readonly",
        ),
    ],
)
def test_helper_returned_resource_alias_diagnostics_for_mojo_codegen(source, pattern):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def test_generic_payload_enum_parameterized_specializations_are_rejected_for_mojo_codegen():
    code = """
    generic<T, E> struct Result {
        enum ResultType {
            Ok(T),
            Err(E)
        }
        ResultType variant;
    }

    enum MathError {
        DivisionByZero
    };

    generic<T> fn pass(Result<T, MathError> value) -> Result<T, MathError> {
        return match value {
            Result::Ok(ok) => Result::Ok(ok),
            Result::Err(err) => Result::Err(err)
        };
    }
    """

    with pytest.raises(ValueError, match="must be concrete"):
        generate_code(parse_code(tokenize_code(code)))


def test_generic_payload_enum_match_expression_uses_single_subject_call():
    code = """
    generic<T, E> struct Result {
        enum ResultType {
            Ok(T),
            Err(E)
        }
        ResultType variant;
    }

    enum MathError {
        DivisionByZero
    };

    Result<vec3, MathError> makeResult(bool ok) {
        match ok {
            true => { return Result::Ok(vec3(1.0, 2.0, 3.0)); },
            _ => { return Result::Err(MathError::DivisionByZero); }
        }
    }

    vec3 read(bool ok, vec3 fallback) {
        let value = match makeResult(ok) {
            Result::Ok(actual) => actual,
            Result::Err(_) => fallback
        };
        return value;
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "var __cgl_match_subject_0: Result_vec3_MathError = makeResult(ok)"
        in generated_code
    )
    assert generated_code.count("makeResult(ok)") == 1
    assert "__cgl_match_value_0 = __cgl_match_subject_0.Ok_0" in generated_code
    assert "else:" in generated_code
    assert "Result<" not in generated_code


def test_nested_generic_payload_enum_field_patterns_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    generic<T> struct Option {
        enum OptionType {
            Some(T),
            None
        }
        OptionType variant;
    }

    enum RenderCommand {
        Draw {
            material: Option<int>,
            value: int
        },
        Clear
    };

    int inspect(RenderCommand command) {
        return match command {
            RenderCommand::Draw { material: Option::Some(mat), value } => mat + value,
            RenderCommand::Draw { material: Option::None, value } => value,
            RenderCommand::Clear => 0
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += (
        "\nfn main():\n"
        "    print(inspect(RenderCommand_Draw_make(Option_int_Some_make(3), 4)))\n"
        "    print(inspect(RenderCommand_Draw_make(Option_int_None_make(), 4)))\n"
        "    print(inspect(RenderCommand_Clear_make()))\n"
    )

    source_path = tmp_path / "nested_generic_payload_enum_patterns.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["7", "4", "0"]


def test_match_expression_guarded_bindings_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    enum Mode {
        Add,
        Multiply
    };

    int guardedValue(int value) {
        return match value {
            zero if zero == 0 => zero + 1,
            _ => 2
        };
    }

    int guardedStatement(int value) {
        int output = 0;
        match value {
            bound if bound > 0 => {
                output = bound;
            }
            _ => {
                output = -1;
            }
        }
        return output;
    }

    int enumValue(Mode mode) {
        return match mode {
            Mode::Add => 10,
            Mode::Multiply => 20,
            _ => 0
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += (
        "\nfn main():\n"
        "    print(guardedValue(0))\n"
        "    print(guardedValue(4))\n"
        "    print(guardedStatement(5))\n"
        "    print(enumValue(Mode_Multiply))\n"
    )

    source_path = tmp_path / "guarded_match_patterns.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["1", "2", "5", "20"]


def test_match_argument_nested_array_resource_size_target_shape_direct_ir_for_mojo_codegen():
    source = """
    sampler2D colorMap;
    image2D colorImage;

    void acceptGrid(ivec2[2][2] grid) {
    }

    void passMatchNestedArrayArgument(int mode, sampler2D tex, image2D image) {
    }
    """
    ast = parse_code(tokenize_code(source))

    def int_literal(value):
        return LiteralNode(value, PrimitiveType("int"))

    def call(name, *args):
        return FunctionCallNode(IdentifierNode(name), list(args))

    def texture_size():
        return call("textureSize", IdentifierNode("tex"), int_literal(0))

    def image_size():
        return call("imageSize", IdentifierNode("image"))

    match_expr = MatchNode(
        IdentifierNode("mode"),
        [
            MatchArmNode(
                LiteralPatternNode(int_literal(0)),
                None,
                ArrayLiteralNode(
                    [
                        ArrayLiteralNode([texture_size(), image_size()]),
                        ArrayLiteralNode([image_size(), texture_size()]),
                    ]
                ),
            ),
            MatchArmNode(
                WildcardPatternNode(),
                None,
                ArrayLiteralNode(
                    [
                        ArrayLiteralNode([image_size()]),
                        ArrayLiteralNode([texture_size()]),
                    ]
                ),
            ),
        ],
    )
    ast.functions[-1].body = BlockNode(
        [
            ExpressionStatementNode(
                call("acceptGrid", match_expr),
                is_tail_expression=False,
            )
        ]
    )

    generated_code = generate_code(ast)

    row_type = "InlineArray[SIMD[DType.int32, 2], 2]"
    nested_type = f"InlineArray[{row_type}, 2]"
    assert f"var __cgl_match_value_0: {nested_type} = {nested_type}(" in generated_code
    assert f"__cgl_match_value_0 = {nested_type}(" in generated_code
    assert (
        f"{row_type}(image_size(image), SIMD[DType.int32, 2](0, 0))" in generated_code
    )
    assert "acceptGrid(__cgl_match_value_0)" in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            sampler2D colorMap;

            int invalidDeclaration(sampler2D tex) {
                int dims = textureSize(tex, 0);
                return dims;
            }
            """,
            r"texture_size target.*declaration dims.*expects ivec2.*got int",
        ),
        (
            """
            sampler2D colorMap;

            void invalidAssignment(sampler2D tex) {
                int dims;
                dims = textureSize(tex, 0);
            }
            """,
            r"texture_size target.*assignment target.*expects ivec2.*got int",
        ),
        (
            """
            sampler2DMSArray multisampled;

            ivec2 invalidReturn(sampler2DMSArray tex) {
                return textureSize(tex);
            }
            """,
            r"texture_size target.*return value.*expects ivec3.*got ivec2",
        ),
        (
            """
            sampler2D colorMap;

            void takeScalar(int dims) {
            }

            void invalidArgument(sampler2D tex) {
                takeScalar(textureSize(tex, 0));
            }
            """,
            r"texture_size target.*argument 1 for takeScalar.*expects ivec2.*got int",
        ),
        (
            """
            samplerCubeArray skyLayers;

            ivec4 invalidWideReturn(samplerCubeArray tex) {
                return textureSize(tex, 0);
            }
            """,
            r"texture_size target.*return value.*expects ivec3.*got ivec4",
        ),
        (
            """
            image2D colorImage;

            uvec2 invalidUnsignedReturn(image2D image) {
                return imageSize(image);
            }
            """,
            r"image_size target.*return value.*expects ivec2.*got uvec2",
        ),
        (
            """
            sampler2D colorMap;
            sampler2DArray layerMap;

            ivec2 invalidTernary(bool choose, sampler2D tex, sampler2DArray layers) {
                return choose ? textureSize(layers, 0) : textureSize(tex, 0);
            }
            """,
            r"texture_size target.*return value true branch.*expects ivec3.*got ivec2",
        ),
        (
            """
            sampler2D colorMap;
            sampler2DArray layerMap;

            ivec2[2] invalidArray(sampler2D tex, sampler2DArray layers) {
                return {textureSize(tex, 0), textureSize(layers, 0)};
            }
            """,
            r"texture_size target.*array literal element 2.*expects ivec3.*got ivec2",
        ),
        (
            """
            struct SizeInfo {
                ivec2 dims;
            };

            sampler2DArray layerMap;

            SizeInfo invalidStruct(sampler2DArray layers) {
                return SizeInfo{dims: textureSize(layers, 0)};
            }
            """,
            r"texture_size target.*field SizeInfo\.dims.*expects ivec3.*got ivec2",
        ),
        (
            """
            struct SizeInfo {
                ivec2 dims;
            };

            sampler2DArray layerMap;

            SizeInfo invalidPositionalStruct(sampler2DArray layers) {
                return SizeInfo{textureSize(layers, 0)};
            }
            """,
            r"texture_size target.*field SizeInfo\.dims.*expects ivec3.*got ivec2",
        ),
        (
            """
            struct SizeInfo {
                ivec2 dims;
            };

            sampler2D colorMap;
            sampler2DArray layerMap;

            void acceptInfo(SizeInfo info) {
            }

            void invalidStructArgument(sampler2D tex, sampler2DArray layers) {
                acceptInfo(SizeInfo{dims: true ? textureSize(tex, 0) : textureSize(layers, 0)});
            }
            """,
            r"texture_size target.*argument 1 for acceptInfo field SizeInfo\.dims false branch.*expects ivec3.*got ivec2",
        ),
        (
            """
            sampler2D colorMap;
            sampler2DArray layerMap;

            ivec2[2][2] invalidNestedArray(sampler2D tex, sampler2DArray layers) {
                return {
                    {textureSize(tex, 0), textureSize(tex, 0)},
                    {textureSize(tex, 0), textureSize(layers, 0)}
                };
            }
            """,
            r"texture_size target.*array literal element 2 array literal element 2.*expects ivec3.*got ivec2",
        ),
        (
            """
            sampler2D colorMap;
            sampler2DArray layerMap;

            ivec2 invalidMatch(int mode, sampler2D tex, sampler2DArray layers) {
                return match mode {
                    0 => textureSize(layers, 0),
                    _ => textureSize(tex, 0)
                };
            }
            """,
            r"texture_size target.*return value match arm 1.*expects ivec3.*got ivec2",
        ),
        (
            """
            sampler2D colorMap;
            sampler2DArray layerMap;

            vec4 invalidDirectVectorConstructor(
                bool useLayer,
                sampler2D tex,
                sampler2DArray layers
            ) {
                return vec4(
                    useLayer ? textureSize(layers, 0) : textureSize(tex, 0),
                    1.0,
                    2.0,
                    3.0
                );
            }
            """,
            r"texture_size target.*vector constructor vec4 argument 1 true branch.*expects ivec3.*got float",
        ),
        (
            """
            sampler2D colorMap;
            sampler2DArray layerMap;

            mat2 invalidDirectMatrixConstructor(
                bool useLayer,
                sampler2D tex,
                sampler2DArray layers
            ) {
                return mat2(
                    1.0,
                    useLayer ? textureSize(layers, 0) : textureSize(tex, 0),
                    2.0,
                    3.0
                );
            }
            """,
            r"texture_size target.*matrix constructor mat2 argument 2 true branch.*expects ivec3.*got float",
        ),
        (
            """
            sampler2D colorMap;
            sampler2DArray layerMap;

            float invalidScalarConstructorTernary(
                bool useLayer,
                sampler2DArray layers
            ) {
                return float(useLayer ? textureSize(layers, 0) : 0);
            }
            """,
            r"texture_size target.*scalar constructor float argument 1 true branch.*expects ivec3.*got float",
        ),
        (
            """
            sampler2D colorMap;
            sampler2DArray layerMap;

            float invalidScalarConstructorMatch(
                int mode,
                sampler2DArray layers
            ) {
                return float(match mode {
                    0 => textureSize(layers, 0),
                    _ => 0
                });
            }
            """,
            r"texture_size target.*scalar constructor float argument 1 match arm 1.*expects ivec3.*got float",
        ),
        (
            """
            sampler2D colorMap;
            sampler2DArray layerMap;

            float invalidScalarImageConstructor(
                bool useWide,
                image2D image
            ) {
                return float(useWide ? imageSize(image) : 0);
            }
            """,
            r"image_size target.*scalar constructor float argument 1 true branch.*expects ivec2.*got float",
        ),
        (
            """
            sampler2D colorMap;
            sampler2DArray layerMap;

            void acceptGrid(ivec2[2][2] grid) {
            }

            void invalidNestedArgument(sampler2D tex, sampler2DArray layers) {
                acceptGrid({
                    {textureSize(tex, 0), textureSize(layers, 0)},
                    {textureSize(tex, 0), textureSize(tex, 0)}
                });
            }
            """,
            r"texture_size target.*argument 1 for acceptGrid array literal element 1 array literal element 2.*expects ivec3.*got ivec2",
        ),
        (
            """
            sampler2D colorMap;
            sampler2DArray layerMap;

            ivec2[2] invalidArrayMatch(
                int mode,
                sampler2D tex,
                sampler2DArray layers
            ) {
                return {
                    match mode {
                        0 => textureSize(layers, 0),
                        _ => textureSize(tex, 0)
                    },
                    textureSize(tex, 0)
                };
            }
            """,
            r"texture_size target.*return value array literal element 1 match arm 1.*expects ivec3.*got ivec2",
        ),
        (
            """
            struct SizeInfo {
                ivec2 dims;
            };

            sampler2D colorMap;
            sampler2DArray layerMap;

            SizeInfo invalidStructMatch(
                int mode,
                sampler2D tex,
                sampler2DArray layers
            ) {
                return SizeInfo{
                    dims: match mode {
                        0 => textureSize(layers, 0),
                        _ => textureSize(tex, 0)
                    }
                };
            }
            """,
            r"texture_size target.*return value field SizeInfo\.dims match arm 1.*expects ivec3.*got ivec2",
        ),
    ],
)
def test_resource_size_target_shape_diagnostics_for_mojo_codegen(source, pattern):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            sampler2D colorMap;
            image2D colorImage;

            ivec2[2] invalidArrayMatch(int mode, sampler2D tex, image2D image) {
                return {
                    match mode {
                        0 => textureSize(tex, 0)
                    },
                    imageSize(image)
                };
            }
            """,
            "match expressions must include a final wildcard arm",
        ),
        (
            """
            struct SizeInfo {
                ivec2 dims;
            };

            sampler2D colorMap;

            SizeInfo invalidStructMatch(int mode, sampler2D tex) {
                return SizeInfo{
                    dims: match mode {
                        0 => textureSize(tex, 0)
                    }
                };
            }
            """,
            "match expressions must include a final wildcard arm",
        ),
    ],
)
def test_nested_match_expression_exhaustiveness_diagnostics_for_mojo_codegen(
    source, pattern
):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


@pytest.mark.parametrize(
    "source",
    [
        """
        vec2 invalidVectorConstructorMatch(int mode) {
            return vec2(match mode {
                0 => 1
            });
        }
        """,
        """
        float invalidScalarConstructorMatch(int mode) {
            return float(match mode {
                0 => 1
            });
        }
        """,
        """
        mat2 invalidMatrixConstructorMatch(int mode) {
            return mat2(match mode {
                0 => 1.0
            });
        }
        """,
    ],
)
def test_constructor_match_expression_exhaustiveness_diagnostics_for_mojo_codegen(
    source,
):
    with pytest.raises(
        ValueError, match="match expressions must include a final wildcard arm"
    ):
        generate_code(parse_code(tokenize_code(source)))


def test_resource_size_target_shape_contexts_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()
    generated_code = generate_code(
        parse_code(tokenize_code(_resource_size_target_shape_source()))
    )
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "resource_size_target_shapes.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_nested_resource_size_target_shape_contexts_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()
    generated_code = generate_code(
        parse_code(tokenize_code(_nested_resource_size_target_shape_source()))
    )
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "nested_resource_size_target_shapes.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_nested_array_resource_size_target_shape_contexts_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()
    generated_code = generate_code(
        parse_code(tokenize_code(_nested_array_resource_size_target_shape_source()))
    )
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "nested_array_resource_size_target_shapes.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_value_expression_resource_size_target_shape_contexts_compile_with_mojo(
    tmp_path,
):
    mojo = find_mojo_compiler()
    generated_code = generate_code(
        parse_code(tokenize_code(_value_expression_resource_size_target_shape_source()))
    )
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "value_expression_resource_size_target_shapes.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_nested_match_resource_size_target_shape_contexts_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()
    generated_code = generate_code(
        parse_code(tokenize_code(_nested_match_resource_size_target_shape_source()))
    )
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "nested_match_resource_size_target_shapes.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_constructor_match_resource_size_target_shape_contexts_compile_with_mojo(
    tmp_path,
):
    mojo = find_mojo_compiler()
    generated_code = generate_code(
        parse_code(
            tokenize_code(_constructor_match_resource_size_target_shape_source())
        )
    )
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "constructor_match_resource_size_target_shapes.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_resource_query_and_image_placeholders_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    sampler2D colorMap;
    sampler2DMS msTex;
    image2D colorImage;
    uimage2DMS msCounterImage;
    iimage2D signedImage;
    uimage2D counterImage;

    ivec2 texSize(sampler2D tex) {
        return textureSize(tex, 0);
    }
    int texLevels(sampler2D tex) {
        return textureQueryLevels(tex);
    }
    int texSamples(sampler2DMS tex) {
        return textureSamples(tex);
    }
    int imageMsCount(uimage2DMS image) {
        return imageSamples(image);
    }
    vec4 fetchTex(sampler2D tex, ivec2 pixel) {
        return texelFetch(tex, pixel, 0);
    }
    ivec2 imgSize(image2D image) {
        return imageSize(image);
    }
    vec4 readColor(image2D image, ivec2 pixel) {
        return imageLoad(image, pixel);
    }
    void writeColor(image2D image, ivec2 pixel, vec4 value) {
        imageStore(image, pixel, value);
    }
    int readSigned(iimage2D image, ivec2 pixel) {
        return imageLoad(image, pixel);
    }
    void writeSigned(iimage2D image, ivec2 pixel, int value) {
        imageStore(image, pixel, value);
    }
    uint readCounter(uimage2D image, ivec2 pixel) {
        return imageLoad(image, pixel);
    }
    void writeCounter(uimage2D image, ivec2 pixel, uint value) {
        imageStore(image, pixel, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "resource_query_image_placeholders.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_resource_sample_count_and_query_level_diagnostics_for_mojo_codegen():
    non_multisample_texture_samples = """
    sampler2D colorMap;

    int invalidSamples(sampler2D tex) {
        return textureSamples(tex);
    }
    """
    with pytest.raises(
        ValueError, match="texture_samples.*multisample texture required"
    ):
        generate_code(parse_code(tokenize_code(non_multisample_texture_samples)))

    non_multisample_image_samples = """
    uimage2D image;

    int invalidSamples(uimage2D image) {
        return imageSamples(image);
    }
    """
    with pytest.raises(ValueError, match="image_samples.*multisample image required"):
        generate_code(parse_code(tokenize_code(non_multisample_image_samples)))

    wrong_resource_for_texture_samples = """
    image2D image;

    int invalidSamples(image2D image) {
        return textureSamples(image);
    }
    """
    with pytest.raises(ValueError, match="texture_samples.*texture resource required"):
        generate_code(parse_code(tokenize_code(wrong_resource_for_texture_samples)))

    multisample_query_levels = """
    sampler2DMS samples;

    int invalidLevels(sampler2DMS tex) {
        return textureQueryLevels(tex);
    }
    """
    with pytest.raises(
        ValueError, match="texture_query_levels.*non-multisample texture required"
    ):
        generate_code(parse_code(tokenize_code(multisample_query_levels)))

    image_query_levels = """
    image2D image;

    int invalidLevels(image2D image) {
        return textureQueryLevels(image);
    }
    """
    with pytest.raises(
        ValueError, match="texture_query_levels.*texture resource required"
    ):
        generate_code(parse_code(tokenize_code(image_query_levels)))


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            float invalidGetDimensions(int count) {
                return float(count.GetDimensions());
            }
            """,
            r"GetDimensions.*texture, image, or buffer resource receiver required: Int32",
        ),
        (
            """
            vec2 invalidTextureSize(bool useQuery, int count) {
                return vec2(useQuery ? count.textureSize() : 1);
            }
            """,
            r"texture_size.*texture resource receiver required: Int32",
        ),
        (
            """
            vec2 invalidQueryLevels(bool useQuery, int count) {
                return vec2(useQuery ? count.textureQueryLevels() : 1);
            }
            """,
            r"texture_query_levels.*texture resource receiver required: Int32",
        ),
        (
            """
            float invalidTextureSamples(int mode, int count) {
                return float(match mode {
                    0 => count.textureSamples(),
                    _ => 1
                });
            }
            """,
            r"texture_samples.*texture resource receiver required: Int32",
        ),
        (
            """
            vec2 invalidImageSamples(bool useQuery, int count) {
                return vec2(useQuery ? count.imageSamples() : 1);
            }
            """,
            r"image_samples.*image resource receiver required: Int32",
        ),
        (
            """
            sampler2DMS msTex;

            vec2 invalidImageSamplesOnTexture(bool useQuery, sampler2DMS tex) {
                return vec2(useQuery ? tex.imageSamples() : 1);
            }
            """,
            r"image_samples.*image resource required: Texture2DMS",
        ),
        (
            """
            image2DMS msImage;

            float invalidTextureSamplesOnImage(image2DMS image) {
                return float(image.textureSamples());
            }
            """,
            r"texture_samples.*texture resource required: Image2DMS",
        ),
        (
            """
            sampler2D colorMap;

            float invalidNonMsTextureSamples(sampler2D tex) {
                return float(tex.textureSamples());
            }
            """,
            r"texture_samples.*multisample texture required: Texture2D",
        ),
        (
            """
            image2D colorImage;

            vec2 invalidNonMsImageSamples(bool useQuery, image2D image) {
                return vec2(useQuery ? image.imageSamples() : 1);
            }
            """,
            r"image_samples.*multisample image required: Image2D",
        ),
        (
            """
            image2D colorImage;

            float invalidQueryLevelsOnImage(image2D image) {
                return float(image.textureQueryLevels());
            }
            """,
            r"texture_query_levels.*texture resource required: Image2D",
        ),
    ],
)
def test_member_resource_query_receiver_diagnostics_for_mojo_codegen(source, pattern):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def test_multisample_resource_placeholders_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    sampler2DMS msTex;
    sampler2DMSArray msLayers;
    image2DMS msImage;
    uimage2DMSArray msCounters;

    ivec2 msTexSize(sampler2DMS tex) {
        return textureSize(tex);
    }
    ivec3 msLayerSize(sampler2DMSArray tex) {
        return textureSize(tex);
    }
    vec4 fetchMs(sampler2DMS tex, ivec2 pixel, int sampleIndex) {
        return texelFetch(tex, pixel, sampleIndex);
    }
    vec4 fetchLayerMs(sampler2DMSArray tex, ivec3 pixelLayer, int sampleIndex) {
        return texelFetch(tex, pixelLayer, sampleIndex);
    }
    vec4 readMs(image2DMS image, ivec2 pixel, int sampleIndex) {
        return imageLoad(image, pixel, sampleIndex);
    }
    void writeMs(image2DMS image, ivec2 pixel, int sampleIndex, vec4 value) {
        imageStore(image, pixel, sampleIndex, value);
    }
    uint readLayerMs(uimage2DMSArray image, ivec3 pixelLayer, int sampleIndex) {
        return imageLoad(image, pixelLayer, sampleIndex);
    }
    void writeLayerMs(
        uimage2DMSArray image,
        ivec3 pixelLayer,
        int sampleIndex,
        uint value
    ) {
        imageStore(image, pixelLayer, sampleIndex, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct Texture2DMS:" in generated_code
    assert "struct Texture2DMSArray:" in generated_code
    assert "struct Image2DMS:" in generated_code
    assert "struct UImage2DMSArray:" in generated_code
    assert (
        "fn texture_size(tex: Texture2DMS) -> SIMD[DType.int32, 2]:" in generated_code
    )
    assert (
        "fn texture_size(tex: Texture2DMSArray) -> SIMD[DType.int32, 4]:"
        in generated_code
    )
    assert (
        "fn texel_fetch(tex: Texture2DMS, coord: SIMD[DType.int32, 2], lod: Int32)"
        in generated_code
    )
    assert (
        "fn texel_fetch(tex: Texture2DMSArray, coord: SIMD[DType.int32, 4], "
        "lod: Int32)" in generated_code
    )
    assert (
        "fn image_load(image: Image2DMS, coord: SIMD[DType.int32, 2], "
        "sample: Int32) -> SIMD[DType.float32, 4]:" in generated_code
    )
    assert (
        "fn image_store(image: UImage2DMSArray, coord: SIMD[DType.int32, 4], "
        "sample: Int32, value: UInt32):" in generated_code
    )
    assert "return texture_size(tex)" in generated_code
    assert "return texel_fetch(tex, pixel, sampleIndex)" in generated_code
    assert "return image_load(image, pixel, sampleIndex)" in generated_code
    assert "image_store(image, pixelLayer, sampleIndex, value)" in generated_code

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "multisample_resource_placeholders.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_resource_arrays_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    sampler2D textures[2];
    image2D images[2];

    vec4 sampleArray(int index, vec2 uv) {
        return texture(textures[index], uv);
    }
    vec4 readArray(int index, ivec2 pixel) {
        return imageLoad(images[index], pixel);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "var textures = InlineArray[Texture2D, 2]" in generated_code
    assert "var images = InlineArray[Image2D, 2]" in generated_code
    assert "return sample(textures[int(index)], uv)" in generated_code
    assert "return image_load(images[int(index)], pixel)" in generated_code

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "resource_arrays.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_resource_binding_metadata_comments_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    @set(2) @binding(5) sampler2D colorMap;
    sampler linearSampler @binding(1);
    @binding(3) RWStructuredBuffer<int> counters[2];
    Texture2D hlslTexture : register(t7, space3);

    @binding(6)
    cbuffer Camera {
        float exposure;
    }

    image2D outImage;
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "# CrossGL resource metadata: name=colorMap kind=texture set=2 "
        "binding=5 binding_source=explicit" in generated_code
    )
    assert (
        "# CrossGL resource metadata: name=linearSampler kind=sampler set=0 "
        "binding=1 binding_source=explicit" in generated_code
    )
    assert (
        "# CrossGL resource metadata: name=counters kind=buffer set=0 "
        "binding=3 binding_source=explicit count=2" in generated_code
    )
    assert (
        "# CrossGL resource metadata: name=hlslTexture kind=texture set=3 "
        "binding=7 binding_source=explicit register=t7,space3" in generated_code
    )
    assert (
        "# CrossGL resource metadata: name=outImage kind=image set=0 "
        "binding=0 binding_source=automatic" in generated_code
    )
    assert (
        "# CrossGL resource metadata: name=Camera kind=cbuffer set=0 "
        "binding=6 binding_source=explicit" in generated_code
    )

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "resource_binding_metadata.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_duplicate_resource_bindings_are_rejected_for_mojo_codegen():
    duplicate_texture_binding = """
    @binding(2) sampler2D firstTexture;
    @binding(2) sampler2D secondTexture;
    """

    with pytest.raises(ValueError, match="Conflicting Mojo resource binding"):
        generate_code(parse_code(tokenize_code(duplicate_texture_binding)))

    overlapping_buffer_range = """
    @binding(3) RWStructuredBuffer<int> counters[2];
    @binding(4)
    cbuffer Camera {
        float exposure;
    }
    """

    with pytest.raises(ValueError, match="Conflicting Mojo resource binding"):
        generate_code(parse_code(tokenize_code(overlapping_buffer_range)))


def test_resource_memory_qualifier_metadata_and_access_validation_for_mojo_codegen():
    code = """
    coherent readonly uniform uimage2D counters;
    volatile writeonly uniform image2D outImage;

    uint readCounter(ivec2 pixel) {
        return imageLoad(counters, pixel);
    }

    void writeOutput(ivec2 pixel, vec4 value) {
        imageStore(outImage, pixel, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "# CrossGL resource metadata: name=counters kind=image set=0 binding=0 "
        "binding_source=automatic access=readonly memory=coherent" in generated_code
    )
    assert (
        "# CrossGL resource metadata: name=outImage kind=image set=0 binding=1 "
        "binding_source=automatic access=writeonly memory=volatile" in generated_code
    )
    assert "return image_load(counters, pixel)" in generated_code
    assert "image_store(outImage, pixel, value)" in generated_code

    invalid_read = """
    writeonly uniform image2D outImage;

    vec4 invalidRead(ivec2 pixel) {
        return imageLoad(outImage, pixel);
    }
    """
    with pytest.raises(ValueError, match="imageLoad.*writeonly"):
        generate_code(parse_code(tokenize_code(invalid_read)))

    invalid_write = """
    readonly uniform image2D source;

    void invalidWrite(ivec2 pixel, vec4 value) {
        imageStore(source, pixel, value);
    }
    """
    with pytest.raises(ValueError, match="imageStore.*readonly"):
        generate_code(parse_code(tokenize_code(invalid_write)))


def test_resource_array_access_qualifiers_apply_to_indexed_mojo_resources():
    invalid_image_read = """
    writeonly uniform image2D images[2];

    vec4 invalidRead(int slot, ivec2 pixel) {
        return imageLoad(images[slot], pixel);
    }
    """
    with pytest.raises(ValueError, match="imageLoad.*images.*writeonly"):
        generate_code(parse_code(tokenize_code(invalid_image_read)))

    invalid_image_write = """
    readonly uniform image2D images[2];

    void invalidWrite(int slot, ivec2 pixel, vec4 value) {
        imageStore(images[slot], pixel, value);
    }
    """
    with pytest.raises(ValueError, match="imageStore.*images.*readonly"):
        generate_code(parse_code(tokenize_code(invalid_image_write)))

    invalid_buffer_read = """
    writeonly RWStructuredBuffer<int> buffers[2];

    int invalidRead(int slot, uint index) {
        return buffer_load(buffers[slot], index);
    }
    """
    with pytest.raises(ValueError, match="buffer_load.*buffers.*writeonly"):
        generate_code(parse_code(tokenize_code(invalid_buffer_read)))

    invalid_buffer_write = """
    readonly RWStructuredBuffer<int> buffers[2];

    void invalidWrite(int slot, uint index, int value) {
        buffer_store(buffers[slot], index, value);
    }
    """
    with pytest.raises(ValueError, match="buffer_store.*buffers.*readonly"):
        generate_code(parse_code(tokenize_code(invalid_buffer_write)))

    invalid_raw_method_write = """
    readonly RWByteAddressBuffer rawBuffers[2];

    void invalidMethodWrite(int slot, uint offset, uint value) {
        rawBuffers[slot].Store(offset, value);
    }
    """
    with pytest.raises(ValueError, match="Store.*rawBuffers.*readonly"):
        generate_code(parse_code(tokenize_code(invalid_raw_method_write)))


def test_struct_resource_field_access_qualifiers_are_enforced_for_mojo_codegen():
    invalid_structured_member_read = """
    struct Resources {
        writeonly RWStructuredBuffer<int> values[2];
    };

    Resources resources;

    int invalidRead(int slot, uint index) {
        return resources.values[slot].Load(index);
    }
    """
    with pytest.raises(ValueError, match=r"Load.*resources\.values.*writeonly"):
        generate_code(parse_code(tokenize_code(invalid_structured_member_read)))

    invalid_structured_free_store = """
    struct Resources {
        readonly RWStructuredBuffer<int> values[2];
    };

    Resources resources;

    void invalidStore(int slot, uint index, int value) {
        buffer_store(resources.values[slot], index, value);
    }
    """
    with pytest.raises(ValueError, match=r"buffer_store.*resources\.values.*readonly"):
        generate_code(parse_code(tokenize_code(invalid_structured_free_store)))

    invalid_raw_member_store = """
    struct Resources {
        readonly RWByteAddressBuffer rawBuffers[2];
    };

    Resources resources;

    void invalidRawStore(int slot, uint offset, uint4 value) {
        resources.rawBuffers[slot].Store4(offset, value);
    }
    """
    with pytest.raises(ValueError, match=r"Store4.*resources\.rawBuffers.*readonly"):
        generate_code(parse_code(tokenize_code(invalid_raw_member_store)))

    invalid_raw_free_read = """
    struct Resources {
        writeonly RWByteAddressBuffer rawBuffers[2];
    };

    Resources resources;

    uint invalidRawRead(int slot, uint offset) {
        return buffer_load(resources.rawBuffers[slot], offset);
    }
    """
    with pytest.raises(
        ValueError, match=r"buffer_load.*resources\.rawBuffers.*writeonly"
    ):
        generate_code(parse_code(tokenize_code(invalid_raw_free_read)))


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            writeonly image2D writeImages[2];
            readonly image2D readImages[2];

            vec4 invalidRead(bool useWrite, int slot, ivec2 pixel) {
                return imageLoad(
                    (useWrite ? writeImages : readImages)[slot],
                    pixel
                );
            }
            """,
            r"imageLoad.*writeImages.*writeonly",
        ),
        (
            """
            readonly image2D readImages[2];
            writeonly image2D writeImages[2];

            void invalidWrite(bool useRead, int slot, ivec2 pixel, vec4 value) {
                imageStore(
                    (useRead ? readImages : writeImages)[slot],
                    pixel,
                    value
                );
            }
            """,
            r"imageStore.*readImages.*readonly",
        ),
        (
            """
            writeonly RWStructuredBuffer<int> writeBuffers[2];
            RWStructuredBuffer<int> values[2];

            int invalidLoad(bool useWrite, int slot, uint index) {
                return (useWrite ? writeBuffers : values)[slot].Load(index);
            }
            """,
            r"Load.*writeBuffers.*writeonly",
        ),
        (
            """
            readonly RWStructuredBuffer<int> readBuffers[2];
            RWStructuredBuffer<int> values[2];

            void invalidStore(int mode, int slot, uint index, int value) {
                (match mode { 0 => readBuffers, _ => values })[slot].Store(
                    index,
                    value
                );
            }
            """,
            r"Store.*readBuffers.*readonly",
        ),
        (
            """
            readonly RWByteAddressBuffer readRaw[2];
            RWByteAddressBuffer rawBuffers[2];

            void invalidRawStore(int mode, int slot, uint offset, uint4 data) {
                (match mode { 0 => readRaw, _ => rawBuffers })[slot].Store4(
                    offset,
                    data
                );
            }
            """,
            r"Store4.*readRaw.*readonly",
        ),
        (
            """
            struct Resources {
                writeonly image2D images[2];
            };

            vec4 invalidStructRead(
                bool useLeft,
                Resources left,
                Resources right,
                int slot,
                ivec2 pixel
            ) {
                return imageLoad((useLeft ? left : right).images[slot], pixel);
            }
            """,
            r"imageLoad.*images.*writeonly",
        ),
    ],
)
def test_branch_selected_resource_access_qualifiers_apply_for_mojo_codegen(
    source, pattern
):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            writeonly image2D writeImage;
            image2D colorImage;

            vec4 invalidAliasRead(bool useWrite, ivec2 pixel) {
                image2D alias = useWrite ? writeImage : colorImage;
                return imageLoad(alias, pixel);
            }
            """,
            r"imageLoad.*writeImage.*writeonly",
        ),
        (
            """
            readonly uniform uimage2D readCounters;
            uimage2D counters;

            uint invalidAliasAtomic(bool useRead, ivec2 pixel, uint value) {
                uimage2D alias = useRead ? readCounters : counters;
                return alias.InterlockedAdd(pixel, value);
            }
            """,
            r"image_atomic_add.*readCounters.*readonly",
        ),
        (
            """
            writeonly uniform uimage2D writeCounters;
            uimage2D counters;

            uint invalidAliasAtomic(int mode, ivec2 pixel, uint value) {
                uimage2D alias = match mode {
                    0 => writeCounters,
                    _ => counters
                };
                return imageAtomicAdd(alias, pixel, value);
            }
            """,
            r"image_atomic_add.*writeCounters.*writeonly",
        ),
        (
            """
            readonly RWStructuredBuffer<int> readValues;
            RWStructuredBuffer<int> values;

            void invalidAliasStore(bool useRead, uint index, int value) {
                RWStructuredBuffer<int> alias = useRead ? readValues : values;
                alias.Store(index, value);
            }
            """,
            r"Store.*readValues.*readonly",
        ),
        (
            """
            readonly RWStructuredBuffer<int> readValues;
            RWStructuredBuffer<int> values;

            void invalidAssignedAliasStore(bool useRead, uint index, int value) {
                RWStructuredBuffer<int> alias;
                alias = useRead ? readValues : values;
                alias.Store(index, value);
            }
            """,
            r"Store.*readValues.*readonly",
        ),
        (
            """
            readonly RWByteAddressBuffer readRaw;
            RWByteAddressBuffer rawBytes;

            void invalidAliasAtomic(
                int mode,
                uint offset,
                uint value,
                uint original
            ) {
                RWByteAddressBuffer alias = match mode {
                    0 => readRaw,
                    _ => rawBytes
                };
                alias.InterlockedAdd(offset, value, original);
            }
            """,
            r"InterlockedAdd.*readRaw.*readonly",
        ),
        (
            """
            writeonly uniform uimage2D writeCounters;
            uimage2D counters;

            uint invalidAssignedAliasAtomic(bool useWrite, ivec2 pixel, uint value) {
                uimage2D alias;
                alias = useWrite ? writeCounters : counters;
                return alias.InterlockedAdd(pixel, value);
            }
            """,
            r"image_atomic_add.*writeCounters.*writeonly",
        ),
    ],
)
def test_resource_alias_access_qualifiers_propagate_for_mojo_codegen(source, pattern):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def test_unsupported_buffer_counter_and_byte_address_atomic_methods_are_diagnostic():
    invalid_byte_address_atomic = """
    RWByteAddressBuffer rawBuffer;

    void invalidAtomic(uint offset, uint value) {
        rawBuffer.InterlockedAdd(offset, value);
    }
    """
    with pytest.raises(
        ValueError, match="byte-address buffer atomic method InterlockedAdd"
    ):
        generate_code(parse_code(tokenize_code(invalid_byte_address_atomic)))

    invalid_compare_store = """
    RWByteAddressBuffer rawBuffer;

    void invalidCompareStore(uint offset, uint compare, uint value) {
        rawBuffer.InterlockedCompareStore(offset, compare, value);
    }
    """
    with pytest.raises(
        ValueError, match="byte-address buffer atomic method InterlockedCompareStore"
    ):
        generate_code(parse_code(tokenize_code(invalid_compare_store)))

    invalid_readonly_atomic = """
    readonly RWByteAddressBuffer rawBuffer;

    void invalidReadonlyAtomic(uint offset, uint value) {
        rawBuffer.InterlockedOr(offset, value);
    }
    """
    with pytest.raises(ValueError, match="InterlockedOr.*rawBuffer.*readonly"):
        generate_code(parse_code(tokenize_code(invalid_readonly_atomic)))

    invalid_counter = """
    RWStructuredBuffer<int> values;

    uint invalidCounter() {
        return values.IncrementCounter();
    }
    """
    with pytest.raises(ValueError, match="buffer counter method IncrementCounter"):
        generate_code(parse_code(tokenize_code(invalid_counter)))

    invalid_readonly_counter = """
    readonly RWStructuredBuffer<int> values;

    uint invalidReadonlyCounter() {
        return values.DecrementCounter();
    }
    """
    with pytest.raises(ValueError, match="DecrementCounter.*values.*readonly"):
        generate_code(parse_code(tokenize_code(invalid_readonly_counter)))


def test_struct_resource_field_buffer_access_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct Resources {
        RWStructuredBuffer<int> values[2];
        RWByteAddressBuffer rawBuffers[2];
    };

    Resources resources;

    int update(int slot, uint index, uint offset, uint4 data) {
        int previous = resources.values[slot].Load(index);
        resources.rawBuffers[slot].Store4(offset, data);
        return previous;
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "fn buffer_load(buffer: RWStructuredBuffer[Int32], " in generated_code
    assert "fn buffer_store4(buffer: RWByteAddressBuffer, " in generated_code
    assert "buffer_load(resources.values[int(slot)], index)" in generated_code
    assert "buffer_store4(resources.rawBuffers[int(slot)], offset, data)" in (
        generated_code
    )
    assert ".Load(" not in generated_code
    assert ".Store4(" not in generated_code

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "struct_resource_field_buffer_access.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_nested_struct_resource_array_containers_preserve_mojo_placeholders():
    code = """
    struct ResourceSet {
        readonly RWStructuredBuffer<int> inputValues[2];
        RWStructuredBuffer<int> outputValues[2];
        RWByteAddressBuffer rawBuffers[2];
    };

    struct ResourceBundle {
        ResourceSet resources;
    };

    ResourceBundle bundle;
    ResourceBundle bundles[2];

    int readNested(uint outer, int slot, uint index) {
        return bundles[outer].resources.inputValues[slot].Load(index);
    }

    void writeNested(int slot, uint index, int value, uint offset, uint4 data) {
        bundle.resources.outputValues[slot].Store(index, value);
        bundle.resources.rawBuffers[slot].Store4(offset, data);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct ResourceSet:" in generated_code
    assert "var inputValues: InlineArray[RWStructuredBuffer[Int32], 2]" in (
        generated_code
    )
    assert "var outputValues: InlineArray[RWStructuredBuffer[Int32], 2]" in (
        generated_code
    )
    assert "var rawBuffers: InlineArray[RWByteAddressBuffer, 2]" in generated_code
    assert "struct ResourceBundle:" in generated_code
    assert "var resources: ResourceSet" in generated_code
    assert "var bundle = ResourceBundle(ResourceSet(" in generated_code
    assert "var bundles = InlineArray[ResourceBundle, 2]" in generated_code
    assert (
        "buffer_load(bundles[int(outer)].resources.inputValues[int(slot)], index)"
        in generated_code
    )
    assert (
        "buffer_store(bundle.resources.outputValues[int(slot)], index, value)"
        in generated_code
    )
    assert (
        "buffer_store4(bundle.resources.rawBuffers[int(slot)], offset, data)"
        in generated_code
    )
    assert (
        "fn buffer_load(buffer: RWStructuredBuffer[Int32], index: UInt32) -> Int32:"
        in generated_code
    )
    assert (
        "fn buffer_store(buffer: RWStructuredBuffer[Int32], "
        "index: UInt32, value: Int32):" in generated_code
    )
    assert (
        "fn buffer_store4(buffer: RWByteAddressBuffer, "
        "index: UInt32, value: SIMD[DType.uint32, 4]):" in generated_code
    )
    assert ".Load(" not in generated_code
    assert ".Store(" not in generated_code
    assert ".Store4(" not in generated_code
    assert "RWStructuredBuffer<" not in generated_code


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            struct Resources {
                writeonly RWStructuredBuffer<int> values[2];
            };
            struct Bundle {
                Resources resources;
            };
            int invalidRead(Bundle bundle, int slot, uint index) {
                return bundle.resources.values[slot].Load(index);
            }
            """,
            r"Load.*bundle\.resources\.values.*writeonly",
        ),
        (
            """
            struct Resources {
                readonly RWStructuredBuffer<int> values[2];
            };
            struct Bundle {
                Resources resources;
            };
            void invalidStore(Bundle bundle, int slot, uint index, int value) {
                bundle.resources.values[slot].Store(index, value);
            }
            """,
            r"Store.*bundle\.resources\.values.*readonly",
        ),
        (
            """
            struct Resources {
                readonly RWByteAddressBuffer rawBuffers[2];
            };
            struct Bundle {
                Resources resources;
            };
            void invalidRawStore(
                Bundle bundles[2],
                uint outer,
                int slot,
                uint offset,
                uint4 value
            ) {
                bundles[outer].resources.rawBuffers[slot].Store4(offset, value);
            }
            """,
            r"Store4.*bundles\.resources\.rawBuffers.*readonly",
        ),
        (
            """
            struct Resources {
                writeonly RWByteAddressBuffer rawBuffers[2];
            };
            struct Bundle {
                Resources resources;
            };
            uint invalidRawRead(Bundle bundle, int slot, uint offset) {
                return buffer_load(bundle.resources.rawBuffers[slot], offset);
            }
            """,
            r"buffer_load.*bundle\.resources\.rawBuffers.*writeonly",
        ),
    ],
)
def test_nested_struct_resource_array_access_qualifiers_apply_for_mojo_codegen(
    source, pattern
):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def test_nested_struct_resource_array_containers_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct ResourceSet {
        readonly RWStructuredBuffer<int> inputValues[2];
        RWStructuredBuffer<int> outputValues[2];
        RWByteAddressBuffer rawBuffers[2];
    };

    struct ResourceBundle {
        ResourceSet resources;
    };

    ResourceBundle bundle;
    ResourceBundle bundles[2];

    int readNested(uint outer, int slot, uint index) {
        return bundles[outer].resources.inputValues[slot].Load(index);
    }

    void writeNested(int slot, uint index, int value, uint offset, uint4 data) {
        bundle.resources.outputValues[slot].Store(index, value);
        bundle.resources.rawBuffers[slot].Store4(offset, data);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "nested_struct_resource_array_containers.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_resource_struct_fields_default_initialize_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct ResourceSet {
        sampler2D texture;
        sampler state;
        image2D inputImage;
        sampler2D textures[2];
        image2D images[2];
        RWStructuredBuffer<int> values[2];
        RWByteAddressBuffer rawBuffers[2];
    };

    struct ResourceBundle {
        ResourceSet sets[2];
        ResourceSet primary;
    };

    ResourceBundle makeBundle() {
        ResourceBundle bundle;
        return bundle;
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct ResourceSet:" in generated_code
    assert "struct ResourceBundle:" in generated_code
    assert "var bundle = ResourceBundle(" in generated_code
    assert "InlineArray[ResourceSet, 2](ResourceSet(" in generated_code
    assert "InlineArray[Texture2D, 2](Texture2D(), Texture2D())" in generated_code
    assert "InlineArray[Image2D, 2](Image2D(), Image2D())" in generated_code
    assert (
        "InlineArray[RWStructuredBuffer[Int32], 2]"
        "(RWStructuredBuffer[Int32](), RWStructuredBuffer[Int32]())" in generated_code
    )
    assert (
        "InlineArray[RWByteAddressBuffer, 2]"
        "(RWByteAddressBuffer(), RWByteAddressBuffer())" in generated_code
    )

    generated_code += "\nfn main():\n    var bundle = makeBundle()\n    _ = bundle\n"

    source_path = tmp_path / "resource_struct_field_defaults.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_resource_struct_arrays_default_initialize_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct ResourceSet {
        sampler2D texture;
        sampler state;
        image2D inputImage;
        sampler2D textures[2];
        image2D images[2];
        RWStructuredBuffer<int> values[2];
        RWByteAddressBuffer rawBuffers[2];
    };

    ResourceSet chooseSet(ResourceSet sets[2], int index) {
        return sets[index];
    }

    ResourceSet buildSet(
        sampler2D texture,
        sampler state,
        image2D image,
        RWStructuredBuffer<int> values[2],
        RWByteAddressBuffer rawBuffers[2]
    ) {
        ResourceSet partials[2] = {
            ResourceSet {
                texture: texture,
                state: state,
                inputImage: image,
                values: values,
                rawBuffers: rawBuffers
            }
        };
        ResourceSet locals[2];
        locals[0] = partials[0];
        return chooseSet(locals, 0);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "fn chooseSet(sets: InlineArray[ResourceSet, 2], index: Int32)" in (
        generated_code
    )
    assert "var locals = InlineArray[ResourceSet, 2](ResourceSet(" in generated_code
    assert (
        "var locals = InlineArray[ResourceSet, 2](unsafe_uninitialized=True)"
        not in generated_code
    )
    assert (
        "textures=InlineArray[Texture2D, 2](Texture2D(), Texture2D())" in generated_code
    )
    assert "images=InlineArray[Image2D, 2](Image2D(), Image2D())" in generated_code
    assert (
        "InlineArray[RWStructuredBuffer[Int32], 2]"
        "(RWStructuredBuffer[Int32](), RWStructuredBuffer[Int32]())" in generated_code
    )
    assert (
        "InlineArray[RWByteAddressBuffer, 2]"
        "(RWByteAddressBuffer(), RWByteAddressBuffer())" in generated_code
    )

    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "resource_struct_array_defaults.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_resource_struct_array_function_boundaries_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct ResourceSet {
        sampler2D texture;
        sampler state;
        image2D inputImage;
        sampler2D textures[2];
        image2D images[2];
        RWStructuredBuffer<int> values[2];
        RWByteAddressBuffer rawBuffers[2];
    };

    ResourceSet[2] cloneSets(ResourceSet sets[2]) {
        return sets;
    }

    ResourceSet[2] replaceSets(
        ResourceSet sets[2],
        ResourceSet replacement,
        int index
    ) {
        sets[index] = replacement;
        return sets;
    }

    ResourceSet forwardSet(ResourceSet sets[2], int index) {
        ResourceSet copied[2] = cloneSets(sets);
        return copied[index];
    }

    ResourceSet buildReplacement() {
        ResourceSet set;
        return set;
    }

    ResourceSet callBoundaries(int index) {
        ResourceSet sets[2];
        ResourceSet replacement = buildReplacement();
        ResourceSet changedSets[2] = replaceSets(sets, replacement, index);
        ResourceSet forwarded = forwardSet(changedSets, index);
        return forwarded;
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "fn cloneSets(sets: InlineArray[ResourceSet, 2]) -> "
        "InlineArray[ResourceSet, 2]:" in generated_code
    )
    assert (
        "fn replaceSets(owned sets: InlineArray[ResourceSet, 2], "
        "replacement: ResourceSet, index: Int32) -> InlineArray[ResourceSet, 2]:"
        in generated_code
    )
    assert (
        "var changedSets: InlineArray[ResourceSet, 2] = "
        "replaceSets(sets, replacement, index)" in generated_code
    )
    assert "var sets = InlineArray[ResourceSet, 2](ResourceSet(" in generated_code
    assert "unsafe_uninitialized=True" not in generated_code

    generated_code += (
        "\nfn main():\n    var resourceSet = callBoundaries(1)\n    _ = resourceSet\n"
    )

    source_path = tmp_path / "resource_struct_array_function_boundaries.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_resource_struct_array_member_operations_after_copy_compile_with_mojo(
    tmp_path,
):
    mojo = find_mojo_compiler()

    code = """
    struct ResourceSet {
        sampler2D texture;
        sampler state;
        readonly image2D inputs[2];
        writeonly image2D outputs[2];
        sampler2D textures[2];
        RWStructuredBuffer<int> values[2];
        RWByteAddressBuffer rawBuffers[2];
    };

    ResourceSet[2] cloneSets(ResourceSet sets[2]) {
        return sets;
    }

    vec4 sampleCopied(ResourceSet sets[2], int slot, vec2 uv) {
        ResourceSet copied[2] = cloneSets(sets);
        return texture(copied[0].textures[slot], copied[0].state, uv);
    }

    vec4 readCopied(ResourceSet sets[2], int slot, ivec2 pixel) {
        ResourceSet copied[2] = cloneSets(sets);
        return imageLoad(copied[0].inputs[slot], pixel);
    }

    void writeCopied(ResourceSet sets[2], int slot, ivec2 pixel, vec4 value) {
        ResourceSet copied[2] = cloneSets(sets);
        imageStore(copied[0].outputs[slot], pixel, value);
    }

    int loadBufferCopied(ResourceSet sets[2], int slot, uint index) {
        ResourceSet copied[2] = cloneSets(sets);
        return copied[0].values[slot].Load(index);
    }

    void storeBufferCopied(ResourceSet sets[2], int slot, uint index, int value) {
        ResourceSet copied[2] = cloneSets(sets);
        copied[0].values[slot].Store(index, value);
    }

    uint loadRawCopied(ResourceSet sets[2], int slot, uint offset) {
        ResourceSet copied[2] = cloneSets(sets);
        return copied[0].rawBuffers[slot].Load(offset);
    }

    void storeRawCopied(ResourceSet sets[2], int slot, uint offset, uint4 data) {
        ResourceSet copied[2] = cloneSets(sets);
        copied[0].rawBuffers[slot].Store4(offset, data);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "fn cloneSets(sets: InlineArray[ResourceSet, 2]) -> "
        "InlineArray[ResourceSet, 2]:" in generated_code
    )
    assert "return sample(copied[0].textures[int(slot)], uv)" in generated_code
    assert "return image_load(copied[0].inputs[int(slot)], pixel)" in generated_code
    assert "image_store(copied[0].outputs[int(slot)], pixel, value)" in (generated_code)
    assert "return buffer_load(copied[0].values[int(slot)], index)" in generated_code
    assert "buffer_store(copied[0].values[int(slot)], index, value)" in generated_code
    assert "return buffer_load(copied[0].rawBuffers[int(slot)], offset)" in (
        generated_code
    )
    assert "buffer_store4(copied[0].rawBuffers[int(slot)], offset, data)" in (
        generated_code
    )
    assert "fn sample(tex: Texture2D, coord: SIMD[DType.float32, 2])" in (
        generated_code
    )
    assert "fn image_load(image: Image2D, coord: SIMD[DType.int32, 2])" in (
        generated_code
    )
    assert (
        "fn buffer_load(buffer: RWStructuredBuffer[Int32], index: UInt32) -> Int32:"
        in generated_code
    )
    assert (
        "fn buffer_load(buffer: RWByteAddressBuffer, index: UInt32) -> UInt32:"
        in generated_code
    )
    assert (
        "fn buffer_store4(buffer: RWByteAddressBuffer, "
        "index: UInt32, value: SIMD[DType.uint32, 4]):" in generated_code
    )
    assert "unsafe_uninitialized=True" not in generated_code
    assert ".Load(" not in generated_code
    assert ".Store(" not in generated_code
    assert ".Store4(" not in generated_code

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "resource_struct_array_member_operations.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_resource_struct_array_branch_value_operations_compile_with_mojo(
    tmp_path,
):
    mojo = find_mojo_compiler()

    code = """
    struct ResourceSet {
        sampler2D texture;
        sampler state;
        readonly image2D inputs[2];
        writeonly image2D outputs[2];
        sampler2D textures[2];
        RWStructuredBuffer<int> values[2];
        RWByteAddressBuffer rawBuffers[2];
    };

    vec4 sampleTernaryDirect(
        bool useLeft,
        ResourceSet left,
        ResourceSet right,
        int slot,
        vec2 uv
    ) {
        return texture(
            (useLeft ? left : right).textures[slot],
            (useLeft ? left : right).state,
            uv
        );
    }

    vec4 readTernaryArrayDirect(
        bool useLeft,
        ResourceSet left[2],
        ResourceSet right[2],
        int slot,
        ivec2 pixel
    ) {
        return imageLoad((useLeft ? left : right)[0].inputs[slot], pixel);
    }

    void writeTernaryArrayDirect(
        bool useLeft,
        ResourceSet left[2],
        ResourceSet right[2],
        int slot,
        ivec2 pixel,
        vec4 value
    ) {
        imageStore((useLeft ? left : right)[0].outputs[slot], pixel, value);
    }

    int loadTernaryBufferDirect(
        bool useLeft,
        ResourceSet left,
        ResourceSet right,
        int slot,
        uint index
    ) {
        return (useLeft ? left : right).values[slot].Load(index);
    }

    void storeTernaryBufferDirect(
        bool useLeft,
        ResourceSet left,
        ResourceSet right,
        int slot,
        uint index,
        int value
    ) {
        (useLeft ? left : right).values[slot].Store(index, value);
    }

    uint loadTernaryRawDirect(
        bool useLeft,
        ResourceSet left,
        ResourceSet right,
        int slot,
        uint offset
    ) {
        return (useLeft ? left : right).rawBuffers[slot].Load(offset);
    }

    void storeTernaryRawDirect(
        bool useLeft,
        ResourceSet left,
        ResourceSet right,
        int slot,
        uint offset,
        uint4 data
    ) {
        (useLeft ? left : right).rawBuffers[slot].Store4(offset, data);
    }

    vec4 readMatchArrayDirect(
        int mode,
        ResourceSet left[2],
        ResourceSet right[2],
        int slot,
        ivec2 pixel
    ) {
        return imageLoad(
            (match mode { 0 => left, _ => right })[0].inputs[slot],
            pixel
        );
    }

    int loadMatchBufferDirect(
        int mode,
        ResourceSet left,
        ResourceSet right,
        int slot,
        uint index
    ) {
        return (match mode { 0 => left, _ => right }).values[slot].Load(index);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "return sample((left if useLeft else right).textures[int(slot)], uv)" in (
        generated_code
    )
    assert (
        "return image_load((left if useLeft else right)[0].inputs[int(slot)], pixel)"
        in generated_code
    )
    assert (
        "image_store((left if useLeft else right)[0].outputs[int(slot)], pixel, value)"
        in generated_code
    )
    assert (
        "return buffer_load((left if useLeft else right).values[int(slot)], index)"
        in generated_code
    )
    assert (
        "buffer_store((left if useLeft else right).values[int(slot)], index, value)"
        in generated_code
    )
    assert (
        "return buffer_load((left if useLeft else right).rawBuffers[int(slot)], offset)"
        in generated_code
    )
    assert (
        "buffer_store4((left if useLeft else right).rawBuffers[int(slot)], "
        "offset, data)" in generated_code
    )
    assert "return image_load(__cgl_match_value_0[0].inputs[int(slot)], pixel)" in (
        generated_code
    )
    assert "return buffer_load(__cgl_match_value_1.values[int(slot)], index)" in (
        generated_code
    )
    assert "fn sample(tex: Texture2D, coord: SIMD[DType.float32, 2])" in (
        generated_code
    )
    assert "fn image_load(image: Image2D, coord: SIMD[DType.int32, 2])" in (
        generated_code
    )
    assert (
        "fn buffer_load(buffer: RWStructuredBuffer[Int32], index: UInt32) -> Int32:"
        in generated_code
    )
    assert (
        "fn buffer_store4(buffer: RWByteAddressBuffer, "
        "index: UInt32, value: SIMD[DType.uint32, 4]):" in generated_code
    )
    assert "unsafe_uninitialized=True" not in generated_code
    assert "MatchNode" not in generated_code
    assert ".Load(" not in generated_code
    assert ".Store(" not in generated_code
    assert ".Store4(" not in generated_code

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "resource_struct_array_branch_values.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_nested_sampled_image_resource_array_containers_preserve_mojo_placeholders():
    code = """
    struct SampleSet {
        sampler2D textures[2];
        sampler states[2];
        readonly image2D inputs[2];
        writeonly image2D outputs[2];
    };

    struct SampleBundle {
        SampleSet resources;
    };

    SampleBundle bundle;
    SampleBundle bundles[2];

    vec4 sampleNested(uint outer, int slot, vec2 uv) {
        return texture(
            bundles[outer].resources.textures[slot],
            bundles[outer].resources.states[slot],
            uv
        );
    }

    vec4 readNested(int slot, ivec2 pixel) {
        return imageLoad(bundle.resources.inputs[slot], pixel);
    }

    void writeNested(int slot, ivec2 pixel, vec4 value) {
        imageStore(bundle.resources.outputs[slot], pixel, value);
    }

    vec4 sampleSetParam(SampleSet resources, int slot, vec2 uv) {
        return texture(resources.textures[slot], resources.states[slot], uv);
    }

    void writeSetParam(SampleSet resources, int slot, ivec2 pixel, vec4 value) {
        imageStore(resources.outputs[slot], pixel, value);
    }

    vec4 sampleBundleParam(
        SampleBundle bundles[2],
        uint outer,
        int slot,
        vec2 uv
    ) {
        return texture(bundles[outer].resources.textures[slot], uv);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct SampleSet:" in generated_code
    assert "var textures: InlineArray[Texture2D, 2]" in generated_code
    assert "var states: InlineArray[Sampler, 2]" in generated_code
    assert "var inputs: InlineArray[Image2D, 2]" in generated_code
    assert "var outputs: InlineArray[Image2D, 2]" in generated_code
    assert "struct SampleBundle:" in generated_code
    assert "var resources: SampleSet" in generated_code
    assert "var bundle = SampleBundle(SampleSet(" in generated_code
    assert "var bundles = InlineArray[SampleBundle, 2]" in generated_code
    assert (
        "fn sampleSetParam(resources: SampleSet, slot: Int32, "
        "uv: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 4]:" in generated_code
    )
    assert (
        "fn sampleBundleParam(bundles: InlineArray[SampleBundle, 2], "
        "outer: UInt32, slot: Int32, uv: SIMD[DType.float32, 2]) -> "
        "SIMD[DType.float32, 4]:" in generated_code
    )
    assert (
        "return sample(bundles[int(outer)].resources.textures[int(slot)], uv)"
        in generated_code
    )
    assert "return image_load(bundle.resources.inputs[int(slot)], pixel)" in (
        generated_code
    )
    assert "image_store(bundle.resources.outputs[int(slot)], pixel, value)" in (
        generated_code
    )
    assert "return sample(resources.textures[int(slot)], uv)" in generated_code
    assert "image_store(resources.outputs[int(slot)], pixel, value)" in generated_code
    assert "fn sample(tex: Texture2D, coord: SIMD[DType.float32, 2])" in (
        generated_code
    )
    assert "fn image_load(image: Image2D, coord: SIMD[DType.int32, 2])" in (
        generated_code
    )
    assert (
        "fn image_store(image: Image2D, coord: SIMD[DType.int32, 2], "
        "value: SIMD[DType.float32, 4]):" in generated_code
    )
    assert "sampler2D" not in generated_code
    assert "image2D" not in generated_code
    assert "texture(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            struct SampleSet {
                sampler2D textures[2];
            };
            struct SampleBundle {
                SampleSet resources;
            };
            vec4 invalidSample(SampleBundle bundle, int slot, vec4 uv) {
                return texture(bundle.resources.textures[slot], uv);
            }
            """,
            r"sample.*coordinate.*Texture2D",
        ),
        (
            """
            struct SampleSet {
                readonly image2D inputs[2];
            };
            struct SampleBundle {
                SampleSet resources;
            };
            vec4 invalidRead(SampleBundle bundle, int slot, int pixel) {
                return imageLoad(bundle.resources.inputs[slot], pixel);
            }
            """,
            r"image_load.*coordinate.*Image2D",
        ),
        (
            """
            struct SampleSet {
                writeonly image2D images[2];
            };
            struct SampleBundle {
                SampleSet resources;
            };
            vec4 invalidRead(SampleBundle bundles[2], uint outer, int slot, ivec2 pixel) {
                return imageLoad(bundles[outer].resources.images[slot], pixel);
            }
            """,
            r"imageLoad.*bundles\.resources\.images.*writeonly",
        ),
        (
            """
            struct SampleSet {
                readonly image2D images[2];
            };
            struct SampleBundle {
                SampleSet resources;
            };
            void invalidWrite(
                SampleBundle bundle,
                int slot,
                ivec2 pixel,
                vec4 value
            ) {
                imageStore(bundle.resources.images[slot], pixel, value);
            }
            """,
            r"imageStore.*bundle\.resources\.images.*readonly",
        ),
    ],
)
def test_nested_sampled_image_resource_array_diagnostics_for_mojo_codegen(
    source, pattern
):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def test_nested_sampled_image_resource_array_containers_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct SampleSet {
        sampler2D textures[2];
        sampler states[2];
        readonly image2D inputs[2];
        writeonly image2D outputs[2];
    };

    struct SampleBundle {
        SampleSet resources;
    };

    SampleBundle bundle;
    SampleBundle bundles[2];

    vec4 sampleNested(uint outer, int slot, vec2 uv) {
        return texture(
            bundles[outer].resources.textures[slot],
            bundles[outer].resources.states[slot],
            uv
        );
    }

    vec4 readNested(int slot, ivec2 pixel) {
        return imageLoad(bundle.resources.inputs[slot], pixel);
    }

    void writeNested(int slot, ivec2 pixel, vec4 value) {
        imageStore(bundle.resources.outputs[slot], pixel, value);
    }

    vec4 sampleSetParam(SampleSet resources, int slot, vec2 uv) {
        return texture(resources.textures[slot], resources.states[slot], uv);
    }

    void writeSetParam(SampleSet resources, int slot, ivec2 pixel, vec4 value) {
        imageStore(resources.outputs[slot], pixel, value);
    }

    vec4 sampleBundleParam(
        SampleBundle bundles[2],
        uint outer,
        int slot,
        vec2 uv
    ) {
        return texture(bundles[outer].resources.textures[slot], uv);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "nested_sampled_image_resource_array_containers.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_nested_sampled_image_query_resource_array_containers_for_mojo_codegen():
    code = """
    struct QuerySet {
        sampler2D textures[2];
        sampler2DMS multisampled[2];
        readonly image2D images[2];
        readonly image2DMS msImages[2];
        writeonly image2DMS msOutputs[2];
    };

    struct QueryBundle {
        QuerySet resources;
    };

    QueryBundle bundle;
    QueryBundle bundles[2];

    ivec2 nestedTexSize(uint outer, int slot) {
        return textureSize(bundles[outer].resources.textures[slot], 0);
    }

    int nestedLevels(int slot) {
        return textureQueryLevels(bundle.resources.textures[slot]);
    }

    int nestedTexSamples(int slot) {
        return textureSamples(bundle.resources.multisampled[slot]);
    }

    ivec2 nestedImageSize(uint outer, int slot) {
        return imageSize(bundles[outer].resources.images[slot]);
    }

    int nestedImageSamples(int slot) {
        return imageSamples(bundle.resources.msImages[slot]);
    }

    vec4 fetchNested(int slot, ivec2 pixel, int sampleIndex) {
        return texelFetch(
            bundle.resources.multisampled[slot],
            pixel,
            sampleIndex
        );
    }

    vec4 loadNested(int slot, ivec2 pixel, int sampleIndex) {
        return imageLoad(bundle.resources.msImages[slot], pixel, sampleIndex);
    }

    void storeNested(int slot, ivec2 pixel, int sampleIndex, vec4 value) {
        imageStore(bundle.resources.msOutputs[slot], pixel, sampleIndex, value);
    }

    ivec2 paramTexSize(QuerySet resources, int slot) {
        return textureSize(resources.textures[slot], 0);
    }

    int paramImageSamples(QueryBundle bundles[2], uint outer, int slot) {
        return imageSamples(bundles[outer].resources.msImages[slot]);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct QuerySet:" in generated_code
    assert "var textures: InlineArray[Texture2D, 2]" in generated_code
    assert "var multisampled: InlineArray[Texture2DMS, 2]" in generated_code
    assert "var images: InlineArray[Image2D, 2]" in generated_code
    assert "var msImages: InlineArray[Image2DMS, 2]" in generated_code
    assert "var msOutputs: InlineArray[Image2DMS, 2]" in generated_code
    assert "var bundles = InlineArray[QueryBundle, 2]" in generated_code
    assert (
        "fn paramImageSamples(bundles: InlineArray[QueryBundle, 2], "
        "outer: UInt32, slot: Int32) -> Int32:" in generated_code
    )
    assert (
        "fn texture_size(tex: Texture2D, lod: Int32) -> SIMD[DType.int32, 2]:"
        in generated_code
    )
    assert "fn image_size(image: Image2D) -> SIMD[DType.int32, 2]:" in generated_code
    assert "fn texture_query_levels(tex: Texture2D) -> Int32:" in generated_code
    assert "fn texture_samples(tex: Texture2DMS) -> Int32:" in generated_code
    assert "fn image_samples(image: Image2DMS) -> Int32:" in generated_code
    assert (
        "fn texel_fetch(tex: Texture2DMS, coord: SIMD[DType.int32, 2], "
        "lod: Int32)" in generated_code
    )
    assert (
        "fn image_load(image: Image2DMS, coord: SIMD[DType.int32, 2], "
        "sample: Int32) -> SIMD[DType.float32, 4]:" in generated_code
    )
    assert (
        "fn image_store(image: Image2DMS, coord: SIMD[DType.int32, 2], "
        "sample: Int32, value: SIMD[DType.float32, 4]):" in generated_code
    )
    assert "fn image_size(image: Texture2D)" not in generated_code
    assert "fn image_size(image: Texture2DMS)" not in generated_code
    assert "fn texture_size(tex: Image2D)" not in generated_code
    assert "fn texture_size(tex: Image2DMS)" not in generated_code
    assert (
        "return texture_size(bundles[int(outer)].resources.textures[int(slot)], 0)"
        in generated_code
    )
    assert (
        "return texture_query_levels(bundle.resources.textures[int(slot)])"
        in generated_code
    )
    assert (
        "return texture_samples(bundle.resources.multisampled[int(slot)])"
        in generated_code
    )
    assert (
        "return image_size(bundles[int(outer)].resources.images[int(slot)])"
        in generated_code
    )
    assert (
        "return image_samples(bundle.resources.msImages[int(slot)])" in generated_code
    )
    assert (
        "return texel_fetch("
        "bundle.resources.multisampled[int(slot)], pixel, sampleIndex)"
        in generated_code
    )
    assert (
        "return image_load(bundle.resources.msImages[int(slot)], pixel, sampleIndex)"
        in generated_code
    )
    assert (
        "image_store("
        "bundle.resources.msOutputs[int(slot)], pixel, sampleIndex, value)"
        in generated_code
    )
    assert "return texture_size(resources.textures[int(slot)], 0)" in generated_code
    assert (
        "return image_samples("
        "bundles[int(outer)].resources.msImages[int(slot)])" in generated_code
    )
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "textureQueryLevels(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "imageSamples(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            ivec2 invalidSize() {
                return textureSize();
            }
            """,
            r"texture_size.*texture resource required",
        ),
        (
            """
            ivec2 invalidSize() {
                return imageSize();
            }
            """,
            r"image_size.*image resource required",
        ),
        (
            """
            struct QuerySet {
                image2D images[2];
            };
            struct QueryBundle {
                QuerySet resources;
            };
            ivec2 invalidSize(QueryBundle bundle, int slot) {
                return textureSize(bundle.resources.images[slot], 0);
            }
            """,
            r"texture_size.*texture resource required",
        ),
        (
            """
            struct QuerySet {
                sampler2D textures[2];
            };
            struct QueryBundle {
                QuerySet resources;
            };
            ivec2 invalidSize(QueryBundle bundle, int slot) {
                return imageSize(bundle.resources.textures[slot]);
            }
            """,
            r"image_size.*image resource required",
        ),
        (
            """
            struct QuerySet {
                image2D images[2];
            };
            struct QueryBundle {
                QuerySet resources;
            };
            ivec2 invalidSize(QueryBundle bundle, int slot) {
                return imageSize(bundle.resources.images[slot], 0);
            }
            """,
            r"image_size.*expected 0 argument.*got 1",
        ),
        (
            """
            struct QuerySet {
                sampler2D textures[2];
            };
            struct QueryBundle {
                QuerySet resources;
            };
            ivec2 invalidSize(QueryBundle bundle, int slot, float lod) {
                return textureSize(bundle.resources.textures[slot], lod);
            }
            """,
            r"texture_size.*lod.*Texture2D.*Int32",
        ),
        (
            """
            struct QuerySet {
                sampler2DMS multisampled[2];
            };
            struct QueryBundle {
                QuerySet resources;
            };
            int invalidSamples(QueryBundle bundle, int slot) {
                return textureQueryLevels(bundle.resources.multisampled[slot]);
            }
            """,
            r"texture_query_levels.*non-multisample texture required",
        ),
        (
            """
            struct QuerySet {
                sampler2D textures[2];
            };
            struct QueryBundle {
                QuerySet resources;
            };
            int invalidSamples(QueryBundle bundle, int slot) {
                return textureSamples(bundle.resources.textures[slot]);
            }
            """,
            r"texture_samples.*multisample texture required",
        ),
        (
            """
            struct QuerySet {
                image2D images[2];
            };
            struct QueryBundle {
                QuerySet resources;
            };
            int invalidSamples(QueryBundle bundle, int slot) {
                return imageSamples(bundle.resources.images[slot]);
            }
            """,
            r"image_samples.*multisample image required",
        ),
        (
            """
            struct QuerySet {
                sampler2DMS multisampled[2];
            };
            struct QueryBundle {
                QuerySet resources;
            };
            vec4 invalidFetch(
                QueryBundle bundle,
                int slot,
                vec3 pixel,
                int sampleIndex
            ) {
                return texelFetch(
                    bundle.resources.multisampled[slot],
                    pixel,
                    sampleIndex
                );
            }
            """,
            r"texel_fetch.*coordinate.*Texture2DMS",
        ),
        (
            """
            struct QuerySet {
                image2DMS images[2];
            };
            struct QueryBundle {
                QuerySet resources;
            };
            vec4 invalidLoad(QueryBundle bundle, int slot, ivec2 pixel) {
                return imageLoad(bundle.resources.images[slot], pixel);
            }
            """,
            r"image_load.*expected 2 argument.*got 1",
        ),
        (
            """
            struct QuerySet {
                writeonly image2DMS images[2];
            };
            struct QueryBundle {
                QuerySet resources;
            };
            void invalidStore(
                QueryBundle bundle,
                int slot,
                ivec2 pixel,
                float sampleIndex,
                vec4 value
            ) {
                imageStore(
                    bundle.resources.images[slot],
                    pixel,
                    sampleIndex,
                    value
                );
            }
            """,
            r"image_store.*sample.*Image2DMS.*Int32",
        ),
    ],
)
def test_nested_sampled_image_query_resource_array_diagnostics_for_mojo_codegen(
    source, pattern
):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def test_nested_sampled_image_query_resource_array_containers_compile_with_mojo(
    tmp_path,
):
    mojo = find_mojo_compiler()

    code = """
    struct QuerySet {
        sampler2D textures[2];
        sampler2DMS multisampled[2];
        readonly image2D images[2];
        readonly image2DMS msImages[2];
        writeonly image2DMS msOutputs[2];
    };

    struct QueryBundle {
        QuerySet resources;
    };

    QueryBundle bundle;
    QueryBundle bundles[2];

    ivec2 nestedTexSize(uint outer, int slot) {
        return textureSize(bundles[outer].resources.textures[slot], 0);
    }

    int nestedLevels(int slot) {
        return textureQueryLevels(bundle.resources.textures[slot]);
    }

    int nestedTexSamples(int slot) {
        return textureSamples(bundle.resources.multisampled[slot]);
    }

    ivec2 nestedImageSize(uint outer, int slot) {
        return imageSize(bundles[outer].resources.images[slot]);
    }

    int nestedImageSamples(int slot) {
        return imageSamples(bundle.resources.msImages[slot]);
    }

    vec4 fetchNested(int slot, ivec2 pixel, int sampleIndex) {
        return texelFetch(
            bundle.resources.multisampled[slot],
            pixel,
            sampleIndex
        );
    }

    vec4 loadNested(int slot, ivec2 pixel, int sampleIndex) {
        return imageLoad(bundle.resources.msImages[slot], pixel, sampleIndex);
    }

    void storeNested(int slot, ivec2 pixel, int sampleIndex, vec4 value) {
        imageStore(bundle.resources.msOutputs[slot], pixel, sampleIndex, value);
    }

    ivec2 paramTexSize(QuerySet resources, int slot) {
        return textureSize(resources.textures[slot], 0);
    }

    int paramImageSamples(QueryBundle bundles[2], uint outer, int slot) {
        return imageSamples(bundles[outer].resources.msImages[slot]);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "nested_sampled_image_query_resource_array_containers.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_nested_hlsl_member_resource_query_containers_for_mojo_codegen():
    code = """
    struct MemberSet {
        Texture2D<float4> textures[2];
        Texture2DMS<float4> multisampled[2];
        RWTexture2D<float2> images[2];
    };

    struct MemberBundle {
        MemberSet resources;
    };

    MemberBundle bundle;
    MemberBundle bundles[2];

    int2 nestedTextureDimensions(uint outer, int slot) {
        return bundles[outer].resources.textures[slot].GetDimensions();
    }

    float4 nestedTextureLoad(int slot, int2 pixel, int sampleIndex) {
        return bundle.resources.multisampled[slot].Load(pixel, sampleIndex);
    }

    int2 nestedImageDimensions(int slot) {
        return bundle.resources.images[slot].GetDimensions();
    }

    float2 nestedImageLoad(int slot, int2 pixel) {
        return bundle.resources.images[slot].Load(pixel);
    }

    void nestedImageStore(MemberBundle bundle, int slot, int2 pixel, float2 value) {
        bundle.resources.images[slot].Store(pixel, value);
    }

    int2 paramTextureDimensions(MemberSet resources, int slot) {
        return resources.textures[slot].GetDimensions();
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct MemberSet:" in generated_code
    assert "var textures: InlineArray[Texture2D, 2]" in generated_code
    assert "var multisampled: InlineArray[Texture2DMS, 2]" in generated_code
    assert "var images: InlineArray[Image2DFloat2, 2]" in generated_code
    assert "var bundles = InlineArray[MemberBundle, 2]" in generated_code
    assert "fn texture_size(tex: Texture2D) -> SIMD[DType.int32, 2]:" in generated_code
    assert (
        "fn texel_fetch(tex: Texture2DMS, coord: SIMD[DType.int32, 2], "
        "lod: Int32)" in generated_code
    )
    assert (
        "fn image_size(image: Image2DFloat2) -> SIMD[DType.int32, 2]:" in generated_code
    )
    assert (
        "fn image_load(image: Image2DFloat2, coord: SIMD[DType.int32, 2]) -> "
        "SIMD[DType.float32, 2]:" in generated_code
    )
    assert (
        "fn image_store(image: Image2DFloat2, coord: SIMD[DType.int32, 2], "
        "value: SIMD[DType.float32, 2]):" in generated_code
    )
    assert (
        "return texture_size(bundles[int(outer)].resources.textures[int(slot)])"
        in generated_code
    )
    assert (
        "return texel_fetch("
        "bundle.resources.multisampled[int(slot)], pixel, sampleIndex)"
        in generated_code
    )
    assert "return image_size(bundle.resources.images[int(slot)])" in generated_code
    assert (
        "return image_load(bundle.resources.images[int(slot)], pixel)" in generated_code
    )
    assert (
        "image_store(bundle.resources.images[int(slot)], pixel, value)"
        in generated_code
    )
    assert "return texture_size(resources.textures[int(slot)])" in generated_code
    assert "Texture2D<float4>" not in generated_code
    assert "Texture2DMS<float4>" not in generated_code
    assert "RWTexture2D<float2>" not in generated_code
    assert ".GetDimensions(" not in generated_code
    assert ".Load(" not in generated_code
    assert ".Store(" not in generated_code


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            struct MemberSet {
                Texture2D<float4> textures[2];
            };
            struct MemberBundle {
                MemberSet resources;
            };
            void invalidStore(
                MemberBundle bundle,
                int slot,
                int2 pixel,
                float4 value
            ) {
                bundle.resources.textures[slot].Store(pixel, value);
            }
            """,
            r"image_store.*image resource required.*Texture2D",
        ),
        (
            """
            struct MemberSet {
                RWTexture2D<float2> images[2];
            };
            struct MemberBundle {
                MemberSet resources;
            };
            float2 invalidLoad(
                MemberBundle bundle,
                int slot,
                int2 pixel,
                int sampleIndex
            ) {
                return bundle.resources.images[slot].Load(pixel, sampleIndex);
            }
            """,
            r"image_load.*expected 1 argument.*got 2",
        ),
        (
            """
            struct MemberSet {
                Texture2D<float4> textures[2];
            };
            struct MemberBundle {
                MemberSet resources;
            };
            int2 invalidDimensions(MemberBundle bundle, int slot, float lod) {
                return bundle.resources.textures[slot].GetDimensions(lod);
            }
            """,
            r"texture_size.*lod.*Texture2D.*Int32",
        ),
        (
            """
            struct MemberSet {
                RWTexture2D<float2> images[2];
            };
            struct MemberBundle {
                MemberSet resources;
            };
            int2 invalidDimensions(MemberBundle bundle, int slot) {
                return bundle.resources.images[slot].GetDimensions(0);
            }
            """,
            r"image_size.*expected 0 argument.*got 1",
        ),
        (
            """
            struct MemberSet {
                Texture2DMS<float4> multisampled[2];
            };
            struct MemberBundle {
                MemberSet resources;
            };
            float4 invalidLoad(
                MemberBundle bundle,
                int slot,
                float2 pixel,
                int sampleIndex
            ) {
                return bundle.resources.multisampled[slot].Load(pixel, sampleIndex);
            }
            """,
            r"texel_fetch.*coordinate.*Texture2DMS.*SIMD\[DType\.int32, 2\]",
        ),
    ],
)
def test_nested_hlsl_member_resource_query_diagnostics_for_mojo_codegen(
    source, pattern
):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def test_nested_hlsl_member_resource_query_containers_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct MemberSet {
        Texture2D<float4> textures[2];
        Texture2DMS<float4> multisampled[2];
        RWTexture2D<float2> images[2];
    };

    struct MemberBundle {
        MemberSet resources;
    };

    MemberBundle bundle;
    MemberBundle bundles[2];

    int2 nestedTextureDimensions(uint outer, int slot) {
        return bundles[outer].resources.textures[slot].GetDimensions();
    }

    float4 nestedTextureLoad(int slot, int2 pixel, int sampleIndex) {
        return bundle.resources.multisampled[slot].Load(pixel, sampleIndex);
    }

    int2 nestedImageDimensions(int slot) {
        return bundle.resources.images[slot].GetDimensions();
    }

    float2 nestedImageLoad(int slot, int2 pixel) {
        return bundle.resources.images[slot].Load(pixel);
    }

    void nestedImageStore(MemberBundle bundle, int slot, int2 pixel, float2 value) {
        bundle.resources.images[slot].Store(pixel, value);
    }

    int2 paramTextureDimensions(MemberSet resources, int slot) {
        return resources.textures[slot].GetDimensions();
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "nested_hlsl_member_resource_query_containers.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_hlsl_get_dimensions_out_parameter_statements_for_mojo_codegen():
    code = """
    struct DimensionSet {
        Texture2DArray<float4> textures[2];
        RWTexture3D<float4> volumes[2];
        StructuredBuffer<float4> buffers[2];
    };

    Texture2D<float4> colorMap;
    Texture2DMS<float4> multisampled;
    RWTexture2D<uint> counters;
    DimensionSet resources;

    void queryDimensions(uint lod, int slot) {
        uint width;
        uint height;
        uint depth;
        uint elements;
        uint levels;
        uint samples;
        uint stride;

        colorMap.GetDimensions(width, height, levels);
        resources.textures[slot].GetDimensions(
            lod,
            width,
            height,
            elements,
            levels
        );
        multisampled.GetDimensions(width, height, samples);
        counters.GetDimensions(width, height);
        resources.volumes[slot].GetDimensions(width, height, depth);
        resources.buffers[slot].GetDimensions(elements, stride);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "fn texture_size(tex: Texture2D) -> SIMD[DType.int32, 2]:" in (
        generated_code
    )
    assert (
        "fn texture_size(tex: Texture2DArray, lod: Int32) -> " "SIMD[DType.int32, 4]:"
    ) in generated_code
    assert (
        "fn texture_size(tex: Texture2DMS) -> SIMD[DType.int32, 2]:" in generated_code
    )
    assert "fn image_size(image: UImage2D) -> SIMD[DType.int32, 2]:" in generated_code
    assert (
        "fn image_size(image: Image3DFloat4) -> SIMD[DType.int32, 4]:" in generated_code
    )
    assert "fn texture_query_levels(tex: Texture2D) -> Int32:" in generated_code
    assert "fn texture_query_levels(tex: Texture2DArray) -> Int32:" in generated_code
    assert "fn texture_samples(tex: Texture2DMS) -> Int32:" in generated_code
    assert (
        "fn buffer_dimensions("
        "buffer: StructuredBuffer[SIMD[DType.float32, 4]], "
        "dimensions: UInt32, stride: UInt32):"
    ) in generated_code
    assert "var __cgl_dimensions_0 = texture_size(colorMap)" in generated_code
    assert "width = UInt32(__cgl_dimensions_0[0])" in generated_code
    assert "height = UInt32(__cgl_dimensions_0[1])" in generated_code
    assert "levels = UInt32(texture_query_levels(colorMap))" in generated_code
    assert (
        "var __cgl_dimensions_1 = "
        "texture_size(resources.textures[int(slot)], Int32(lod))"
    ) in generated_code
    assert "elements = UInt32(__cgl_dimensions_1[2])" in generated_code
    assert (
        "levels = UInt32(texture_query_levels(resources.textures[int(slot)]))"
        in generated_code
    )
    assert "samples = UInt32(texture_samples(multisampled))" in generated_code
    assert "var __cgl_dimensions_3 = image_size(counters)" in generated_code
    assert "var __cgl_dimensions_4 = image_size(resources.volumes[int(slot)])" in (
        generated_code
    )
    assert "depth = UInt32(__cgl_dimensions_4[2])" in generated_code
    assert (
        "var __cgl_dimensions_5 = buffer_dimensions(resources.buffers[int(slot)])"
        in generated_code
    )
    assert "elements = UInt32(__cgl_dimensions_5[0])" in generated_code
    assert "stride = UInt32(__cgl_dimensions_5[1])" in generated_code
    assert ".GetDimensions(" not in generated_code


def _hlsl_get_dimensions_edge_variant_source():
    return """
    Texture1D<float4> strip;
    Texture1DArray<float4> stripArray;
    TextureCube<float4> sky;
    TextureCubeArray<float4> skyArray;
    Texture2DMSArray<float4> msaaArray;
    RWTexture1D<uint> row;
    RWTexture1DArray<uint> rowArray;
    RWTexture2DArray<uint> atlas;

    void queryEdges(
        uint lod,
        uint width,
        uint height,
        uint elements,
        uint levels,
        uint samples
    ) {
        strip.GetDimensions(lod, width, levels);
        strip.GetDimensions(width, levels);
        stripArray.GetDimensions(lod, width, elements, levels);
        sky.GetDimensions(width, height, levels);
        skyArray.GetDimensions(lod, width, height, elements, levels);
        msaaArray.GetDimensions(width, height, elements, samples);
        row.GetDimensions(width);
        rowArray.GetDimensions(width, elements);
        atlas.GetDimensions(width, height, elements);
    }

    void queryParameterTexture(
        Texture1DArray<float4> tex,
        uint lod,
        uint width,
        uint elements,
        uint levels
    ) {
        tex.GetDimensions(lod, width, elements, levels);
    }
    """


def test_hlsl_get_dimensions_out_parameter_edge_variants_for_mojo_codegen():
    generated_code = generate_code(
        parse_code(tokenize_code(_hlsl_get_dimensions_edge_variant_source()))
    )

    assert "fn texture_size(tex: Texture1D) -> Int32:" in generated_code
    assert (
        "fn texture_size(tex: Texture1DArray, lod: Int32) -> " "SIMD[DType.int32, 2]:"
    ) in generated_code
    assert "fn texture_size(tex: TextureCube) -> SIMD[DType.int32, 2]:" in (
        generated_code
    )
    assert (
        "fn texture_size(tex: TextureCubeArray, lod: Int32) -> " "SIMD[DType.int32, 4]:"
    ) in generated_code
    assert (
        "fn texture_size(tex: Texture2DMSArray) -> SIMD[DType.int32, 4]:"
        in generated_code
    )
    assert "fn texture_samples(tex: Texture2DMSArray) -> Int32:" in generated_code
    assert "fn image_size(image: UImage1D) -> Int32:" in generated_code
    assert "fn image_size(image: UImage1DArray) -> SIMD[DType.int32, 2]:" in (
        generated_code
    )
    assert "fn image_size(image: UImage2DArray) -> SIMD[DType.int32, 4]:" in (
        generated_code
    )

    assert (
        "fn queryEdges(lod: UInt32, owned width: UInt32, "
        "owned height: UInt32, owned elements: UInt32, "
        "owned levels: UInt32, owned samples: UInt32) -> None:"
    ) in generated_code
    assert "owned lod" not in generated_code
    assert "var __cgl_dimensions_0 = texture_size(strip, Int32(lod))" in (
        generated_code
    )
    assert "width = UInt32(__cgl_dimensions_0)" in generated_code
    assert "var __cgl_dimensions_1 = texture_size(strip)" in generated_code
    assert "var __cgl_dimensions_2 = texture_size(stripArray, Int32(lod))" in (
        generated_code
    )
    assert "elements = UInt32(__cgl_dimensions_2[1])" in generated_code
    assert "var __cgl_dimensions_3 = texture_size(sky)" in generated_code
    assert "height = UInt32(__cgl_dimensions_3[1])" in generated_code
    assert "var __cgl_dimensions_4 = texture_size(skyArray, Int32(lod))" in (
        generated_code
    )
    assert "elements = UInt32(__cgl_dimensions_4[2])" in generated_code
    assert "var __cgl_dimensions_5 = texture_size(msaaArray)" in generated_code
    assert "samples = UInt32(texture_samples(msaaArray))" in generated_code
    assert "width = UInt32(image_size(row))" in generated_code
    assert "var __cgl_dimensions_6 = image_size(rowArray)" in generated_code
    assert "var __cgl_dimensions_7 = image_size(atlas)" in generated_code
    assert (
        "fn queryParameterTexture(tex: Texture1DArray, lod: UInt32, "
        "owned width: UInt32, owned elements: UInt32, "
        "owned levels: UInt32) -> None:"
    ) in generated_code
    assert "var __cgl_dimensions_8 = texture_size(tex, Int32(lod))" in (generated_code)
    assert ".GetDimensions(" not in generated_code


def test_hlsl_get_dimensions_out_parameter_edge_variants_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()
    generated_code = generate_code(
        parse_code(tokenize_code(_hlsl_get_dimensions_edge_variant_source()))
    )
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "hlsl_get_dimensions_edge_variants.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        (
            """
            Texture2D<float4> colorMap;

            void invalidOverload() {
                uint width;
                uint height;
                uint levels;
                uint extra;
                uint overflow;
                colorMap.GetDimensions(width, height, levels, extra, overflow);
            }
            """,
            r"GetDimensions overload.*Texture2D.*expected 2, 3, or 4.*got 5",
        ),
        (
            """
            Texture2D<float4> colorMap;

            void invalidLod(float lod) {
                uint width;
                uint height;
                uint levels;
                colorMap.GetDimensions(lod, width, height, levels);
            }
            """,
            r"GetDimensions overload.*lod argument.*Texture2D.*integer scalar",
        ),
        (
            """
            RWTexture2D<float4> output;

            void invalidImageOverload() {
                uint width;
                uint height;
                uint levels;
                output.GetDimensions(width, height, levels);
            }
            """,
            r"GetDimensions overload.*Image2DFloat4.*expected 2.*got 3",
        ),
    ],
)
def test_hlsl_get_dimensions_out_parameter_diagnostics_for_mojo_codegen(
    source, pattern
):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def test_hlsl_get_dimensions_out_parameter_statements_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct DimensionSet {
        Texture2DArray<float4> textures[2];
        RWTexture3D<float4> volumes[2];
        StructuredBuffer<float4> buffers[2];
    };

    Texture2D<float4> colorMap;
    Texture2DMS<float4> multisampled;
    RWTexture2D<uint> counters;
    DimensionSet resources;

    void queryDimensions(uint lod, int slot) {
        uint width;
        uint height;
        uint depth;
        uint elements;
        uint levels;
        uint samples;
        uint stride;

        colorMap.GetDimensions(width, height, levels);
        resources.textures[slot].GetDimensions(
            lod,
            width,
            height,
            elements,
            levels
        );
        multisampled.GetDimensions(width, height, samples);
        counters.GetDimensions(width, height);
        resources.volumes[slot].GetDimensions(width, height, depth);
        resources.buffers[slot].GetDimensions(elements, stride);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "hlsl_get_dimensions_out_parameters.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_glsl_buffer_block_metadata_comments_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    layout(std430, set = 1, binding = 2) readonly buffer ParticleBlock {
        vec4 positions[2];
        uint count;
    } particles;

    layout(std430, binding = 5) writeonly buffer OutputBlock {
        vec4 colors[2];
    } outputs;

    vec4 readParticle(uint index) {
        return particles.positions[index];
    }

    void writeOutput(uint index, vec4 value) {
        outputs.colors[index] = value;
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "# CrossGL resource metadata: name=particles kind=glsl_buffer_block "
        "set=1 binding=2 binding_source=explicit layout=std430 access=readonly"
        in generated_code
    )
    assert (
        "# CrossGL resource metadata: name=outputs kind=glsl_buffer_block "
        "set=0 binding=5 binding_source=explicit layout=std430 access=writeonly"
        in generated_code
    )
    assert "return particles.positions[int(index)]" in generated_code
    assert "outputs.colors[int(index)] = value" in generated_code

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "glsl_buffer_block_metadata.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_glsl_buffer_block_access_and_binding_diagnostics_for_mojo_codegen():
    direct_array_block = """
    layout(std430, binding = 3) readonly buffer float values[];
    """
    generated_code = generate_code(parse_code(tokenize_code(direct_array_block)))

    assert (
        "# CrossGL resource metadata: name=values kind=glsl_buffer_block "
        "set=0 binding=3 binding_source=explicit layout=std430 access=readonly"
        in generated_code
    )
    assert "var values = List[Float32]()" in generated_code

    invalid_write = """
    layout(std430, binding = 2) readonly buffer ParticleBlock {
        vec4 positions[2];
    } particles;

    void writeParticle(uint index, vec4 value) {
        particles.positions[index] = value;
    }
    """
    with pytest.raises(ValueError, match="assignment.*readonly"):
        generate_code(parse_code(tokenize_code(invalid_write)))

    overlapping_binding = """
    layout(std430, binding = 4) buffer ParticleBlock {
        vec4 positions[2];
    } particles;

    @binding(4)
    cbuffer Camera {
        float exposure;
    }
    """
    with pytest.raises(ValueError, match="Conflicting Mojo resource binding"):
        generate_code(parse_code(tokenize_code(overlapping_binding)))


def test_structured_buffer_placeholders_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct Particle {
        vec4 position;
        float mass;
    };

    StructuredBuffer<Particle> inputParticles;
    RWStructuredBuffer<Particle> outputParticles;
    RWStructuredBuffer<int> counters[2];
    AppendStructuredBuffer<int> appended;
    ConsumeStructuredBuffer<int> consumed;

    float scale(
        RWStructuredBuffer<float> outputValues,
        StructuredBuffer<float> inputValues,
        uint index
    ) {
        float value = buffer_load(inputValues, index);
        buffer_store(outputValues, index, value * 2.0);
        return value;
    }

    Particle copyParticle(
        RWStructuredBuffer<Particle> outputBuffer,
        StructuredBuffer<Particle> inputBuffer,
        uint index
    ) {
        Particle value = buffer_load(inputBuffer, index);
        buffer_store(outputBuffer, index, value);
        return value;
    }

    int updateCounter(int slot, uint index, int value) {
        int previous = buffer_load(counters[slot], index);
        buffer_store(counters[slot], index, previous + value);
        buffer_append(appended, value);
        return buffer_consume(consumed);
    }

    void queryCount(StructuredBuffer<float> inputValues) {
        uint count = 0;
        buffer_dimensions(inputValues, count);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct StructuredBuffer[T: AnyType]:" in generated_code
    assert "struct RWStructuredBuffer[T: AnyType]:" in generated_code
    assert "struct AppendStructuredBuffer[T: AnyType]:" in generated_code
    assert "struct ConsumeStructuredBuffer[T: AnyType]:" in generated_code
    assert "var inputParticles: StructuredBuffer[Particle]" in generated_code
    assert "var outputParticles: RWStructuredBuffer[Particle]" in generated_code
    assert "InlineArray[RWStructuredBuffer[Int32], 2]" in generated_code
    assert (
        "fn buffer_load(buffer: StructuredBuffer[Float32], "
        "index: UInt32) -> Float32:" in generated_code
    )
    assert (
        "fn buffer_store(buffer: RWStructuredBuffer[Particle], "
        "index: UInt32, value: Particle):" in generated_code
    )
    assert (
        "fn buffer_append(buffer: AppendStructuredBuffer[Int32], value: Int32):"
        in generated_code
    )
    assert (
        "fn buffer_consume(buffer: ConsumeStructuredBuffer[Int32]) -> Int32:"
        in generated_code
    )
    assert (
        "fn buffer_dimensions(buffer: StructuredBuffer[Float32], "
        "dimensions: UInt32):" in generated_code
    )
    assert "RWStructuredBuffer<" not in generated_code
    assert "StructuredBuffer<" not in generated_code

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "structured_buffer_placeholders.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_typed_buffer_vector_alias_placeholders_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    StructuredBuffer<half3> halfVectors;
    RWStructuredBuffer<short4> intVectors;
    AppendStructuredBuffer<ushort3> appended;
    ConsumeStructuredBuffer<ushort3> consumed;

    half3 readHalf(uint index) {
        return halfVectors.Load(index);
    }

    void writeInt(uint index, short4 value) {
        intVectors.Store(index, value);
    }

    ushort3 roundTrip(ushort3 value) {
        appended.Append(value);
        return consumed.Consume();
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "StructuredBuffer[SIMD[DType.float16, 4]]" in generated_code
    assert "RWStructuredBuffer[SIMD[DType.int16, 4]]" in generated_code
    assert "AppendStructuredBuffer[SIMD[DType.uint16, 4]]" in generated_code
    assert "ConsumeStructuredBuffer[SIMD[DType.uint16, 4]]" in generated_code
    assert (
        "fn buffer_load(buffer: StructuredBuffer[SIMD[DType.float16, 4]], "
        "index: UInt32) -> SIMD[DType.float16, 4]:"
    ) in generated_code
    assert (
        "fn buffer_store(buffer: RWStructuredBuffer[SIMD[DType.int16, 4]], "
        "index: UInt32, value: SIMD[DType.int16, 4]):"
    ) in generated_code
    assert (
        "fn buffer_append(buffer: AppendStructuredBuffer[SIMD[DType.uint16, 4]], "
        "value: SIMD[DType.uint16, 4]):"
    ) in generated_code
    assert (
        "fn buffer_consume(buffer: ConsumeStructuredBuffer[SIMD[DType.uint16, 4]]) "
        "-> SIMD[DType.uint16, 4]:"
    ) in generated_code
    assert "half3" not in generated_code
    assert "short4" not in generated_code
    assert "ushort3" not in generated_code

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "typed_buffer_vector_alias_placeholders.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_byte_address_buffer_placeholders_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    ByteAddressBuffer rawBytes;
    RWByteAddressBuffer rawOutput;
    RWByteAddressBuffer rawArray[2];

    uint readRaw(ByteAddressBuffer input, uint offset) {
        return buffer_load(input, offset);
    }

    float readFloat(ByteAddressBuffer input, uint offset) {
        return asfloat(buffer_load(input, offset));
    }

    void writeRaw(RWByteAddressBuffer output, uint offset, uint value, float weight) {
        buffer_store(output, offset, value);
        buffer_store(output, offset + uint(4), asuint(weight));
    }

    uint updateArray(int slot, uint offset) {
        uint value = buffer_load(rawArray[slot], offset);
        buffer_store(rawArray[slot], offset, value + uint(1));
        return value;
    }

    void queryRaw(ByteAddressBuffer input) {
        uint count = 0;
        buffer_dimensions(input, count);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct ByteAddressBuffer:" in generated_code
    assert "struct RWByteAddressBuffer:" in generated_code
    assert "var rawBytes: ByteAddressBuffer = ByteAddressBuffer()" in generated_code
    assert (
        "var rawOutput: RWByteAddressBuffer = RWByteAddressBuffer()" in generated_code
    )
    assert "InlineArray[RWByteAddressBuffer, 2]" in generated_code
    assert (
        "fn buffer_load(buffer: ByteAddressBuffer, index: UInt32) -> UInt32:"
        in generated_code
    )
    assert (
        "fn buffer_load(buffer: RWByteAddressBuffer, index: UInt32) -> UInt32:"
        in generated_code
    )
    assert (
        "fn buffer_store(buffer: RWByteAddressBuffer, index: UInt32, value: UInt32):"
        in generated_code
    )
    assert (
        "fn buffer_dimensions(buffer: ByteAddressBuffer, dimensions: UInt32):"
        in generated_code
    )
    assert "fn asfloat(value: UInt32) -> Float32:" in generated_code
    assert "fn asuint(value: Float32) -> UInt32:" in generated_code
    assert "return asfloat(buffer_load(input, offset))" in generated_code
    assert (
        "buffer_store(output, (offset + UInt32(4)), asuint(weight))" in generated_code
    )

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "byte_address_buffer_placeholders.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_byte_address_vector_methods_and_reinterpret_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    RWByteAddressBuffer rawVectors;

    vec2 readPair(uint offset) {
        return asfloat(rawVectors.Load2(offset));
    }

    vec3 readTriple(uint offset) {
        return asfloat(rawVectors.Load3(offset));
    }

    vec4 readQuad(uint offset) {
        return asfloat(rawVectors.Load4(offset));
    }

    int readSigned(uint offset) {
        return asint(rawVectors.Load(offset));
    }

    void writeVectors(uint offset, vec2 pair, vec3 normal, vec4 color) {
        rawVectors.Store2(offset, asuint(pair));
        rawVectors.Store3(offset + uint(16), asuint(normal));
        rawVectors.Store4(offset + uint(32), asuint(color));
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "None(" not in generated_code
    assert (
        "fn buffer_load2(buffer: RWByteAddressBuffer, "
        "index: UInt32) -> SIMD[DType.uint32, 2]:" in generated_code
    )
    assert (
        "fn buffer_load3(buffer: RWByteAddressBuffer, "
        "index: UInt32) -> SIMD[DType.uint32, 4]:" in generated_code
    )
    assert (
        "fn buffer_load4(buffer: RWByteAddressBuffer, "
        "index: UInt32) -> SIMD[DType.uint32, 4]:" in generated_code
    )
    assert (
        "fn buffer_store2(buffer: RWByteAddressBuffer, "
        "index: UInt32, value: SIMD[DType.uint32, 2]):" in generated_code
    )
    assert (
        "fn buffer_store3(buffer: RWByteAddressBuffer, "
        "index: UInt32, value: SIMD[DType.uint32, 4]):" in generated_code
    )
    assert (
        "fn asfloat(value: SIMD[DType.uint32, 2]) -> SIMD[DType.float32, 2]:"
        in generated_code
    )
    assert (
        "fn asfloat(value: SIMD[DType.uint32, 4]) -> SIMD[DType.float32, 4]:"
        in generated_code
    )
    assert (
        "fn asuint(value: SIMD[DType.float32, 2]) -> SIMD[DType.uint32, 2]:"
        in generated_code
    )
    assert (
        "fn asuint(value: SIMD[DType.float32, 4]) -> SIMD[DType.uint32, 4]:"
        in generated_code
    )
    assert "fn asint(value: UInt32) -> Int32:" in generated_code
    assert "return asfloat(buffer_load2(rawVectors, offset))" in generated_code
    assert "return asfloat(buffer_load3(rawVectors, offset))" in generated_code
    assert (
        "buffer_store4(rawVectors, (offset + UInt32(32)), asuint(color))"
        in generated_code
    )

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "byte_address_vector_methods.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_invalid_buffer_operations_are_rejected_for_mojo_codegen():
    code = """
    ByteAddressBuffer rawBytes;
    StructuredBuffer<int> readonlyValues;

    void invalidRaw(ByteAddressBuffer input, uint offset, uint value) {
        buffer_store(input, offset, value);
    }

    void invalidStructured(StructuredBuffer<int> input, uint index, int value) {
        buffer_store(input, index, value);
    }
    """

    with pytest.raises(ValueError, match="Unsupported buffer_store"):
        generate_code(parse_code(tokenize_code(code)))


def test_wave_intrinsics_are_rejected_for_mojo_codegen():
    code = """
    shader ComputeWaveDiagnostics {
        compute {
            void main() {
                uint lane = WaveGetLaneIndex();
                uint sumValue = WaveActiveSum(lane);
                bool anyLane = WaveActiveAnyTrue(sumValue > 0u);
                uvec4 ballot = WaveActiveBallot(anyLane);
                uint broadcast = WaveReadLaneAt(sumValue, 0u);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Unsupported Mojo wave intrinsic Wave"):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        ("TraceRay(1, 2, 3, 4, 5, 6, 7, 8);", "ray tracing intrinsic TraceRay"),
        ("ReportHit(1.0, 0);", "ray tracing intrinsic ReportHit"),
        ("SetMeshOutputCounts(1, 1);", "mesh intrinsic SetMeshOutputCounts"),
        ("DispatchMesh(1, 1, 1);", "mesh intrinsic DispatchMesh"),
    ],
)
def test_ray_and_mesh_intrinsics_are_rejected_for_mojo_codegen(statement, message):
    code = f"""
    shader UnsupportedPipelineIntrinsic {{
        compute {{
            void main() {{
                {statement}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=f"Unsupported Mojo {message}"):
        generate_code(parse_code(tokenize_code(code)))


def test_ray_query_methods_are_rejected_for_mojo_codegen():
    code = """
    void queryRay(RayQuery query) {
        bool active = query.Proceed();
    }
    """

    with pytest.raises(ValueError, match="Unsupported Mojo ray query method Proceed"):
        generate_code(parse_code(tokenize_code(code)))


def test_direct_pipeline_ir_nodes_are_rejected_for_mojo_codegen_without_ast_repr():
    codegen = MojoCodeGen()
    query = VariableNode("query", PrimitiveType("RayQuery"))
    scene = VariableNode("scene", PrimitiveType("RaytracingAccelerationStructure"))
    ray = VariableNode("ray", PrimitiveType("RayDesc"))

    cases = [
        (
            WaveOpNode("WaveActiveSum", [LiteralNode(1, PrimitiveType("uint"))]),
            "Unsupported Mojo wave intrinsic WaveActiveSum",
            "WaveOpNode(",
        ),
        (
            RayTracingOpNode(
                "TraceRay",
                [
                    scene,
                    LiteralNode(0, PrimitiveType("uint")),
                    LiteralNode(255, PrimitiveType("uint")),
                    LiteralNode(0, PrimitiveType("uint")),
                    LiteralNode(0, PrimitiveType("uint")),
                    LiteralNode(0, PrimitiveType("uint")),
                    ray,
                    LiteralNode(0, PrimitiveType("uint")),
                ],
            ),
            "Unsupported Mojo ray tracing intrinsic TraceRay",
            "RayTracingOpNode(",
        ),
        (
            RayQueryOpNode("TraceRayInline", query, [scene, 0, 255, ray]),
            "Unsupported Mojo ray query method TraceRayInline",
            "RayQueryOpNode(",
        ),
        (
            RayQueryOpNode("CandidateRayT", query, []),
            "Unsupported Mojo ray query method CandidateRayT",
            "RayQueryOpNode(",
        ),
        (
            MeshOpNode("SetMeshOutputCounts", [1, 1]),
            "Unsupported Mojo mesh intrinsic SetMeshOutputCounts",
            "MeshOpNode(",
        ),
        (
            MeshOpNode("DispatchMesh", [1, 1, 1]),
            "Unsupported Mojo mesh intrinsic DispatchMesh",
            "MeshOpNode(",
        ),
    ]

    for expr, message, repr_marker in cases:
        with pytest.raises(ValueError, match=message) as excinfo:
            codegen.generate_expression(expr)
        assert repr_marker not in str(excinfo.value)


def test_invalid_structured_buffer_member_methods_are_rejected_for_mojo_codegen():
    invalid_member_load = """
    AppendStructuredBuffer<int> values;

    int invalidLoad(uint index) {
        return values.Load(index);
    }
    """
    with pytest.raises(ValueError, match="Unsupported Load.*AppendStructuredBuffer"):
        generate_code(parse_code(tokenize_code(invalid_member_load)))

    invalid_free_load = """
    ConsumeStructuredBuffer<int> values;

    int invalidFreeLoad(uint index) {
        return buffer_load(values, index);
    }
    """
    with pytest.raises(
        ValueError, match="Unsupported buffer_load.*ConsumeStructuredBuffer"
    ):
        generate_code(parse_code(tokenize_code(invalid_free_load)))

    invalid_append = """
    RWStructuredBuffer<int> values;

    void invalidAppend(int value) {
        values.Append(value);
    }
    """
    with pytest.raises(ValueError, match="Unsupported Append.*RWStructuredBuffer"):
        generate_code(parse_code(tokenize_code(invalid_append)))

    invalid_consume = """
    ByteAddressBuffer rawBytes;

    uint invalidConsume() {
        return rawBytes.Consume();
    }
    """
    with pytest.raises(ValueError, match="Unsupported Consume.*ByteAddressBuffer"):
        generate_code(parse_code(tokenize_code(invalid_consume)))


def test_unsupported_buffer_counter_methods_are_rejected_for_mojo_codegen():
    invalid_increment = """
    RWStructuredBuffer<int> values;

    uint invalidIncrement() {
        return values.IncrementCounter();
    }
    """
    with pytest.raises(ValueError, match="buffer counter method IncrementCounter"):
        generate_code(parse_code(tokenize_code(invalid_increment)))

    invalid_decrement = """
    AppendStructuredBuffer<int> values;

    uint invalidDecrement() {
        return values.DecrementCounter();
    }
    """
    with pytest.raises(ValueError, match="buffer counter method DecrementCounter"):
        generate_code(parse_code(tokenize_code(invalid_decrement)))

    readonly_increment = """
    readonly RWStructuredBuffer<int> values;

    uint invalidReadonlyIncrement() {
        return values.IncrementCounter();
    }
    """
    with pytest.raises(ValueError, match="IncrementCounter.*values.*readonly"):
        generate_code(parse_code(tokenize_code(readonly_increment)))


def test_unsupported_byte_address_atomic_methods_are_rejected_for_mojo_codegen():
    invalid_interlocked_add = """
    RWByteAddressBuffer rawBytes;

    void invalidAtomic(uint offset, uint value, uint original) {
        rawBytes.InterlockedAdd(offset, value, original);
    }
    """
    with pytest.raises(
        ValueError, match="byte-address buffer atomic method InterlockedAdd"
    ):
        generate_code(parse_code(tokenize_code(invalid_interlocked_add)))

    invalid_interlocked_compare_exchange = """
    RWByteAddressBuffer rawBytes[2];

    void invalidAtomic(
        int slot,
        uint offset,
        uint compare,
        uint value,
        uint original
    ) {
        rawBytes[slot].InterlockedCompareExchange(
            offset,
            compare,
            value,
            original
        );
    }
    """
    with pytest.raises(
        ValueError,
        match="byte-address buffer atomic method InterlockedCompareExchange",
    ):
        generate_code(parse_code(tokenize_code(invalid_interlocked_compare_exchange)))

    readonly_interlocked_add = """
    readonly RWByteAddressBuffer rawBytes;

    void invalidReadonlyAtomic(uint offset, uint value, uint original) {
        rawBytes.InterlockedAdd(offset, value, original);
    }
    """
    with pytest.raises(ValueError, match="InterlockedAdd.*rawBytes.*readonly"):
        generate_code(parse_code(tokenize_code(readonly_interlocked_add)))


def test_advanced_texture_placeholder_builtins_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    sampler2D colorMap;
    sampler2DArray arrayMap;
    sampler2DShadow shadowMap;
    sampler2DArrayShadow shadowLayers;
    sampler linearSampler;

    vec4 sampleAdvanced(
        sampler2D tex,
        sampler state,
        vec2 uv,
        ivec2 offset,
        vec2 ddx,
        vec2 ddy
    ) {
        return textureOffset(tex, state, uv, offset) +
            textureLodOffset(tex, state, uv, 1.0, offset) +
            textureGradOffset(tex, state, uv, ddx, ddy, offset) +
            textureGather(tex, state, uv, 1) +
            textureGatherOffset(tex, state, uv, offset, 2) +
            textureProj(tex, state, vec3(uv, 1.0));
    }

    vec2 queryLod(sampler2D tex, sampler state, vec2 uv) {
        return textureQueryLod(tex, state, uv);
    }

    vec2 queryArrayLod(sampler2DArray tex, sampler state, vec3 uvLayer) {
        return textureQueryLod(tex, state, uvLayer);
    }

    float compareShadow(
        sampler2DShadow shadow,
        sampler state,
        vec2 uv,
        float depth,
        ivec2 offset
    ) {
        return textureCompare(shadow, state, uv, depth) +
            textureCompareOffset(shadow, state, uv, depth, offset) +
            textureCompareLod(shadow, state, uv, depth, 1.0);
    }

    vec4 gatherShadow(
        sampler2DShadow shadow,
        sampler state,
        vec2 uv,
        float depth,
        ivec2 offset
    ) {
        return textureGatherCompare(shadow, state, uv, depth) +
            textureGatherCompareOffset(shadow, state, uv, depth, offset);
    }

    float compareShadowArray(
        sampler2DArrayShadow shadow,
        sampler state,
        vec3 uvLayer,
        float depth,
        ivec2 offset
    ) {
        return textureCompare(shadow, state, uvLayer, depth) +
            textureCompareOffset(shadow, state, uvLayer, depth, offset) +
            textureCompareLod(shadow, state, uvLayer, depth, 1.0);
    }

    vec4 gatherShadowArray(
        sampler2DArrayShadow shadow,
        sampler state,
        vec3 uvLayer,
        float depth,
        ivec2 offset
    ) {
        return textureGatherCompare(shadow, state, uvLayer, depth) +
            textureGatherCompareOffset(shadow, state, uvLayer, depth, offset);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct Texture2DShadow:" in generated_code
    assert "struct Texture2DArray:" in generated_code
    assert "struct Texture2DArrayShadow:" in generated_code
    assert "_crossgl_sample_offset_Texture2D" in generated_code
    assert "_crossgl_texture_gather_Texture2D" in generated_code
    assert "_crossgl_sample_proj_Texture2D" in generated_code
    assert "_crossgl_texture_query_lod_Texture2D" in generated_code
    assert "_crossgl_texture_query_lod_Texture2DArray" in generated_code
    assert "_crossgl_texture_compare_Texture2DShadow" in generated_code
    assert "_crossgl_texture_gather_compare_Texture2DShadow" in generated_code
    assert "_crossgl_texture_compare_Texture2DArrayShadow" in generated_code
    assert "_crossgl_texture_gather_compare_Texture2DArrayShadow" in generated_code
    assert "textureOffset" not in generated_code
    assert "textureLodOffset" not in generated_code
    assert "textureGradOffset" not in generated_code
    assert "textureGather" not in generated_code
    assert "textureProj" not in generated_code
    assert "textureQueryLod" not in generated_code
    assert "textureCompare" not in generated_code

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "advanced_texture_placeholders.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_advanced_texture_builtins_reject_invalid_mojo_resources():
    image_resource = """
    image2D colorImage;

    vec4 invalidImage(image2D image, vec2 uv) {
        return textureGather(image, uv, 0);
    }
    """
    with pytest.raises(ValueError, match="texture_gather.*texture resource required"):
        generate_code(parse_code(tokenize_code(image_resource)))

    non_shadow_compare = """
    sampler2D colorMap;

    float invalidCompare(sampler2D tex, vec2 uv, float depth) {
        return textureCompare(tex, uv, depth);
    }
    """
    with pytest.raises(ValueError, match="texture_compare.*shadow texture required"):
        generate_code(parse_code(tokenize_code(non_shadow_compare)))

    multisample_resource = """
    sampler2DMS msTex;

    vec4 invalidMultisample(sampler2DMS tex, vec2 uv) {
        return textureGather(tex, uv, 0);
    }
    """
    with pytest.raises(
        ValueError, match="texture_gather.*non-multisample texture required"
    ):
        generate_code(parse_code(tokenize_code(multisample_resource)))

    shadow_query_lod = """
    sampler2DShadow shadowMap;
    sampler linearSampler;

    vec2 invalidShadowQuery(sampler2DShadow shadow, sampler state, vec2 uv) {
        return textureQueryLod(shadow, state, uv);
    }
    """
    with pytest.raises(
        ValueError, match="texture_query_lod.*non-shadow texture required"
    ):
        generate_code(parse_code(tokenize_code(shadow_query_lod)))


def test_texture_coordinate_and_argument_diagnostics_for_mojo_codegen():
    scalar_sample_coord = """
    sampler2D colorMap;

    vec4 invalidSample(sampler2D tex, float u) {
        return texture(tex, u);
    }
    """
    with pytest.raises(ValueError, match="sample.*coordinate.*Texture2D"):
        generate_code(parse_code(tokenize_code(scalar_sample_coord)))

    missing_lod = """
    sampler2D colorMap;

    vec4 invalidLod(sampler2D tex, vec2 uv) {
        return textureLod(tex, uv);
    }
    """
    with pytest.raises(ValueError, match="sample_lod.*expected 2 argument"):
        generate_code(parse_code(tokenize_code(missing_lod)))

    bad_grad_rank = """
    sampler2D colorMap;

    vec4 invalidGrad(sampler2D tex, vec2 uv, vec3 ddx, vec2 ddy) {
        return textureGrad(tex, uv, ddx, ddy);
    }
    """
    with pytest.raises(ValueError, match="sample_grad.*ddx.*Texture2D"):
        generate_code(parse_code(tokenize_code(bad_grad_rank)))

    missing_texel_lod = """
    sampler2D colorMap;

    vec4 invalidFetch(sampler2D tex, ivec2 pixel) {
        return texelFetch(tex, pixel);
    }
    """
    with pytest.raises(ValueError, match="texel_fetch.*expected 2 argument"):
        generate_code(parse_code(tokenize_code(missing_texel_lod)))

    scalar_texel_coord = """
    sampler2D colorMap;

    vec4 invalidFetch(sampler2D tex, int pixel) {
        return texelFetch(tex, pixel, 0);
    }
    """
    with pytest.raises(ValueError, match="texel_fetch.*coordinate.*Texture2D"):
        generate_code(parse_code(tokenize_code(scalar_texel_coord)))

    image_texel_fetch = """
    image2D colorImage;

    vec4 invalidFetch(image2D image, ivec2 pixel) {
        return texelFetch(image, pixel, 0);
    }
    """
    with pytest.raises(ValueError, match="texel_fetch.*texture resource required"):
        generate_code(parse_code(tokenize_code(image_texel_fetch)))


def test_advanced_texture_placeholder_argument_diagnostics_for_mojo_codegen():
    missing_proj_coord = """
    sampler2D colorMap;

    vec4 invalidProj(sampler2D tex) {
        return textureProj(tex);
    }
    """
    with pytest.raises(ValueError, match="sample_proj.*expected 1 argument"):
        generate_code(parse_code(tokenize_code(missing_proj_coord)))

    scalar_proj_coord = """
    sampler2D colorMap;

    vec4 invalidProj(sampler2D tex, vec2 uv) {
        return textureProj(tex, uv);
    }
    """
    with pytest.raises(ValueError, match="sample_proj.*coordinate.*Texture2D"):
        generate_code(parse_code(tokenize_code(scalar_proj_coord)))

    missing_gather_offset = """
    sampler2D colorMap;

    vec4 invalidGatherOffsets(sampler2D tex, vec2 uv, ivec2 offset) {
        return textureGatherOffsets(tex, uv, offset, offset, offset);
    }
    """
    with pytest.raises(
        ValueError, match="texture_gather_offsets.*expected 5 or 6 argument"
    ):
        generate_code(parse_code(tokenize_code(missing_gather_offset)))

    scalar_gather_offset = """
    sampler2D colorMap;

    vec4 invalidGatherOffset(sampler2D tex, vec2 uv, int offset) {
        return textureGatherOffset(tex, uv, offset, 0);
    }
    """
    with pytest.raises(ValueError, match="texture_gather_offset.*offset.*Texture2D"):
        generate_code(parse_code(tokenize_code(scalar_gather_offset)))

    missing_compare_offset = """
    sampler2DShadow shadowMap;

    float invalidCompareOffset(sampler2DShadow shadow, vec2 uv, float depth) {
        return textureCompareOffset(shadow, uv, depth);
    }
    """
    with pytest.raises(ValueError, match="texture_compare_offset.*expected 3 argument"):
        generate_code(parse_code(tokenize_code(missing_compare_offset)))

    bad_compare_grad = """
    sampler2DShadow shadowMap;

    float invalidCompareGrad(
        sampler2DShadow shadow,
        vec2 uv,
        float depth,
        vec3 ddx,
        vec2 ddy
    ) {
        return textureCompareGrad(shadow, uv, depth, ddx, ddy);
    }
    """
    with pytest.raises(ValueError, match="texture_compare_grad.*ddx.*Texture2DShadow"):
        generate_code(parse_code(tokenize_code(bad_compare_grad)))


def test_image_load_store_argument_diagnostics_for_mojo_codegen():
    missing_multisample_load_index = """
    uimage2DMS counters;

    uint invalidLoad(uimage2DMS image, ivec2 pixel) {
        return imageLoad(image, pixel);
    }
    """
    with pytest.raises(ValueError, match="image_load.*expected 2 argument"):
        generate_code(parse_code(tokenize_code(missing_multisample_load_index)))

    scalar_multisample_load_coord = """
    uimage2DMS counters;

    uint invalidLoad(uimage2DMS image, int pixel, int sampleIndex) {
        return imageLoad(image, pixel, sampleIndex);
    }
    """
    with pytest.raises(ValueError, match="image_load.*coordinate.*UImage2DMS"):
        generate_code(parse_code(tokenize_code(scalar_multisample_load_coord)))

    missing_multisample_store_index = """
    uimage2DMS counters;

    void invalidStore(uimage2DMS image, ivec2 pixel, uint value) {
        imageStore(image, pixel, value);
    }
    """
    with pytest.raises(ValueError, match="image_store.*expected 3 argument"):
        generate_code(parse_code(tokenize_code(missing_multisample_store_index)))

    wrong_store_value_type = """
    uimage2D counters;

    void invalidStore(uimage2D image, ivec2 pixel, float value) {
        imageStore(image, pixel, value);
    }
    """
    with pytest.raises(ValueError, match="image_store.*value.*UImage2D"):
        generate_code(parse_code(tokenize_code(wrong_store_value_type)))

    texture_image_load = """
    sampler2D colorMap;

    vec4 invalidLoad(sampler2D tex, ivec2 pixel) {
        return imageLoad(tex, pixel);
    }
    """
    with pytest.raises(ValueError, match="image_load.*image resource required"):
        generate_code(parse_code(tokenize_code(texture_image_load)))


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            """
            float invalidLoad(int count, ivec2 pixel) {
                return float(count.Load(pixel));
            }
            """,
            r"Load.*texture, image, or buffer resource receiver required: Int32",
        ),
        (
            """
            vec2 invalidAtomic(int count, ivec2 pixel, uint value) {
                return vec2(count.Add(pixel, value), 1.0);
            }
            """,
            r"image_atomic_add.*image resource receiver required: Int32",
        ),
        (
            """
            uint passValue(uint value) {
                return value;
            }

            uint invalidInterlocked(int count, ivec2 pixel, uint value) {
                return passValue(count.InterlockedAdd(pixel, value));
            }
            """,
            (
                r"image_atomic_add.*image or byte-address buffer resource "
                r"receiver required: Int32"
            ),
        ),
        (
            """
            uint passValue(uint value) {
                return value;
            }

            uint invalidStore(int count, ivec2 pixel, uint value) {
                return passValue(count.Store(pixel, value));
            }
            """,
            r"Store.*image or buffer resource receiver required: Int32",
        ),
        (
            """
            float invalidImageLoad(uimage2D image, int pixel) {
                return float(image.Load(pixel));
            }
            """,
            r"image_load.*coordinate.*UImage2D",
        ),
        (
            """
            uint invalidAtomicCoord(uimage2D image, int pixel, uint value) {
                return image.Add(pixel, value);
            }
            """,
            r"image_atomic_add.*coordinate.*UImage2D",
        ),
        (
            """
            uint invalidAtomicValue(uimage2D image, ivec2 pixel, int value) {
                return image.Add(pixel, value);
            }
            """,
            r"image_atomic_add.*value.*UImage2D",
        ),
        (
            """
            uint passValue(uint value) {
                return value;
            }

            uint invalidImageStore(uimage2D image, ivec2 pixel, float value) {
                return passValue(image.Store(pixel, value));
            }
            """,
            r"image_store.*value.*UImage2D",
        ),
    ],
)
def test_image_member_operation_diagnostics_in_nested_contexts_for_mojo_codegen(
    source, message
):
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(source)))


def test_shadow_sampler_texture_calls_return_float_and_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    sampler2DShadow shadowMap;
    sampler linearSampler;

    float sampleShadow(
        sampler2DShadow shadow,
        sampler state,
        vec2 uv,
        float depth
    ) {
        return texture(shadow, state, vec3(uv, depth)) +
            textureLod(shadow, state, vec3(uv, depth), 1.0);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "fn sample(tex: Texture2DShadow, "
        "coord: SIMD[DType.float32, 4]) -> Float32:" in generated_code
    )
    assert (
        "fn sample_lod(tex: Texture2DShadow, "
        "coord: SIMD[DType.float32, 4], lod: Float32) -> Float32:" in generated_code
    )
    assert "fn sampleShadow(" in generated_code
    assert "-> Float32:" in generated_code

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "shadow_sampler_texture_calls.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_integer_image_atomic_placeholders_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    uimage2D counters;
    iimage2D signedCounters;

    uint updateCounter(uimage2D image, ivec2 pixel, uint value, uint replacement) {
        uint added = imageAtomicAdd(image, pixel, value);
        uint minValue = imageAtomicMin(image, pixel, value);
        uint maxValue = imageAtomicMax(image, pixel, value);
        uint andValue = imageAtomicAnd(image, pixel, value);
        uint orValue = imageAtomicOr(image, pixel, value);
        uint xorValue = imageAtomicXor(image, pixel, value);
        uint exchanged = imageAtomicExchange(image, pixel, replacement);
        return imageAtomicCompSwap(image, pixel, exchanged, value) + added +
            minValue + maxValue + andValue + orValue + xorValue;
    }

    int updateSigned(iimage2D image, ivec2 pixel, int value, int replacement) {
        int added = imageAtomicAdd(image, pixel, value);
        int minValue = imageAtomicMin(image, pixel, value);
        int maxValue = imageAtomicMax(image, pixel, value);
        int exchanged = imageAtomicExchange(image, pixel, replacement);
        return imageAtomicCompSwap(image, pixel, exchanged, value) + added +
            minValue + maxValue;
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct UImage2D:" in generated_code
    assert "struct IImage2D:" in generated_code
    assert "_crossgl_image_atomic_add_UImage2D" in generated_code
    assert "_crossgl_image_atomic_min_UImage2D" in generated_code
    assert "_crossgl_image_atomic_max_UImage2D" in generated_code
    assert "_crossgl_image_atomic_and_UImage2D" in generated_code
    assert "_crossgl_image_atomic_or_UImage2D" in generated_code
    assert "_crossgl_image_atomic_xor_UImage2D" in generated_code
    assert "_crossgl_image_atomic_exchange_UImage2D" in generated_code
    assert "_crossgl_image_atomic_comp_swap_UImage2D" in generated_code
    assert "_crossgl_image_atomic_add_IImage2D" in generated_code
    assert "imageAtomic" not in generated_code

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "integer_image_atomic_placeholders.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_multisample_image_atomic_placeholders_preserve_sample_indices_for_mojo():
    code = """
    uimage2DMS counters;
    iimage2DMSArray signedLayers;

    uint updateCounter(
        uimage2DMS image,
        ivec2 pixel,
        int sampleIndex,
        uint value,
        uint replacement
    ) {
        uint added = imageAtomicAdd(image, pixel, sampleIndex, value);
        return image.InterlockedCompareExchange(
            pixel,
            sampleIndex,
            added,
            replacement
        );
    }

    int updateLayer(
        iimage2DMSArray image,
        ivec4 pixelLayer,
        int sampleIndex,
        int value
    ) {
        return image.InterlockedAdd(pixelLayer, sampleIndex, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct UImage2DMS:" in generated_code
    assert "struct IImage2DMSArray:" in generated_code
    assert (
        "fn _crossgl_image_atomic_add_UImage2DMS_SIMD_DType_int32_2_Int32_UInt32"
        "(arg0: UImage2DMS, arg1: SIMD[DType.int32, 2], arg2: Int32, "
        "arg3: UInt32) -> UInt32:" in generated_code
    )
    assert (
        "fn _crossgl_image_atomic_comp_swap_UImage2DMS_SIMD_DType_int32_2_"
        "Int32_UInt32_UInt32"
        "(arg0: UImage2DMS, arg1: SIMD[DType.int32, 2], arg2: Int32, "
        "arg3: UInt32, arg4: UInt32) -> UInt32:" in generated_code
    )
    assert (
        "fn _crossgl_image_atomic_add_IImage2DMSArray_SIMD_DType_int32_4_"
        "Int32_Int32"
        "(arg0: IImage2DMSArray, arg1: SIMD[DType.int32, 4], arg2: Int32, "
        "arg3: Int32) -> Int32:" in generated_code
    )
    assert (
        "_crossgl_image_atomic_add_UImage2DMS_SIMD_DType_int32_2_Int32_UInt32"
        "(image, pixel, sampleIndex, value)" in generated_code
    )
    assert (
        "_crossgl_image_atomic_comp_swap_UImage2DMS_SIMD_DType_int32_2_"
        "Int32_UInt32_UInt32(image, pixel, sampleIndex, added, replacement)"
        in generated_code
    )
    assert (
        "_crossgl_image_atomic_add_IImage2DMSArray_SIMD_DType_int32_4_Int32_Int32"
        "(image, pixelLayer, sampleIndex, value)" in generated_code
    )
    assert "imageAtomic" not in generated_code
    assert ".Interlocked" not in generated_code


def test_multisample_image_atomic_placeholders_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    uimage2DMS counters;
    iimage2DMSArray signedLayers;

    uint updateCounter(
        uimage2DMS image,
        ivec2 pixel,
        int sampleIndex,
        uint value,
        uint replacement
    ) {
        uint added = imageAtomicAdd(image, pixel, sampleIndex, value);
        return image.InterlockedCompareExchange(
            pixel,
            sampleIndex,
            added,
            replacement
        );
    }

    int updateLayer(
        iimage2DMSArray image,
        ivec4 pixelLayer,
        int sampleIndex,
        int value
    ) {
        return image.InterlockedAdd(pixelLayer, sampleIndex, value);
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += "\nfn main():\n    pass\n"

    source_path = tmp_path / "multisample_image_atomic_placeholders.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            """
            uimage2DMS counters;

            uint invalidMissingSample(uimage2DMS image, ivec2 pixel, uint value) {
                return imageAtomicAdd(image, pixel, value);
            }
            """,
            r"image_atomic_add.*expected 3 argument.*got 2",
        ),
        (
            """
            uimage2DMS counters;

            uint invalidMemberMissingSample(
                uimage2DMS image,
                ivec2 pixel,
                uint compare,
                uint value
            ) {
                return image.CompareExchange(pixel, compare, value);
            }
            """,
            r"image_atomic_comp_swap.*expected 4 argument.*got 3",
        ),
        (
            """
            uimage2DMS counters;

            uint invalidSampleType(
                uimage2DMS image,
                ivec2 pixel,
                float sampleIndex,
                uint value
            ) {
                return image.InterlockedAdd(pixel, sampleIndex, value);
            }
            """,
            r"image_atomic_add.*sample.*UImage2DMS.*Int32",
        ),
    ],
)
def test_multisample_image_atomic_argument_diagnostics_for_mojo_codegen(
    source, message
):
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(source)))


def test_float_image_atomics_are_rejected_for_mojo_codegen():
    code = """
    image2D colorImage;

    int invalidAtomic(image2D image, ivec2 pixel, int value) {
        return imageAtomicAdd(image, pixel, value);
    }
    """

    with pytest.raises(ValueError, match="integer image required"):
        generate_code(parse_code(tokenize_code(code)))


def test_image_atomics_require_read_write_access_for_mojo_codegen():
    readonly_code = """
    readonly uniform uimage2D counters;

    uint invalidReadonly(ivec2 pixel, uint value) {
        return imageAtomicAdd(counters, pixel, value);
    }
    """

    with pytest.raises(ValueError, match="image_atomic_add.*readonly"):
        generate_code(parse_code(tokenize_code(readonly_code)))

    writeonly_code = """
    writeonly uniform uimage2D counters;

    uint invalidWriteonly(ivec2 pixel, uint value) {
        return imageAtomicAdd(counters, pixel, value);
    }
    """

    with pytest.raises(ValueError, match="image_atomic_add.*writeonly"):
        generate_code(parse_code(tokenize_code(writeonly_code)))


def test_image_atomic_member_aliases_emit_mojo_helpers_and_compile_with_mojo(
    tmp_path,
):
    mojo = find_mojo_compiler()

    code = """
    uimage2D counters;
    iimage2D signedCounters;

    uint updateCounter(uimage2D image, ivec2 pixel, uint value, uint replacement) {
        uint added = image.Add(pixel, value);
        uint andValue = image.atomicAnd(pixel, value);
        uint exchanged = image.InterlockedExchange(pixel, replacement);
        uint swapped = image.CompareExchange(pixel, exchanged, value);
        return swapped + added + andValue;
    }

    int updateSigned(iimage2D image, ivec2 pixel, int value, int replacement) {
        int minValue = image.Min(pixel, value);
        int maxValue = image.InterlockedMax(pixel, value);
        int swapped = image.atomicCompareExchange(pixel, minValue, replacement);
        return swapped + maxValue;
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "_crossgl_image_atomic_add_UImage2D" in generated_code
    assert "_crossgl_image_atomic_and_UImage2D" in generated_code
    assert "_crossgl_image_atomic_exchange_UImage2D" in generated_code
    assert "_crossgl_image_atomic_comp_swap_UImage2D" in generated_code
    assert "_crossgl_image_atomic_min_IImage2D" in generated_code
    assert "_crossgl_image_atomic_max_IImage2D" in generated_code
    assert "_crossgl_image_atomic_comp_swap_IImage2D" in generated_code
    assert "image.Add" not in generated_code
    assert "image.atomicAnd" not in generated_code
    assert "image.InterlockedExchange" not in generated_code
    assert "image.CompareExchange" not in generated_code
    assert "image.atomicCompareExchange" not in generated_code

    generated_code += "\nfn main():\n    pass\n"
    source_path = tmp_path / "image_atomic_member_aliases.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_image_atomic_member_aliases_preserve_mojo_diagnostics():
    readonly_code = """
    readonly uniform uimage2D counters;

    uint invalidReadonly(ivec2 pixel, uint value) {
        return counters.Add(pixel, value);
    }
    """

    with pytest.raises(ValueError, match="image_atomic_add.*readonly"):
        generate_code(parse_code(tokenize_code(readonly_code)))

    writeonly_code = """
    writeonly uniform uimage2D counters;

    uint invalidWriteonly(ivec2 pixel, uint value) {
        return counters.InterlockedAdd(pixel, value);
    }
    """

    with pytest.raises(ValueError, match="image_atomic_add.*writeonly"):
        generate_code(parse_code(tokenize_code(writeonly_code)))

    float_image_code = """
    image2D colorImage;

    int invalidFloatImage(ivec2 pixel, int value) {
        return colorImage.atomicAdd(pixel, value);
    }
    """

    with pytest.raises(ValueError, match="integer image required"):
        generate_code(parse_code(tokenize_code(float_image_code)))


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


def _for_update_assignment_prelude_source():
    return """
    int updateArrayLoop(int mode) {
        int values[2] = {0, 0};
        for (int step = 0; step < 2; values = {
            match mode {
                0 => 1,
                _ => 2
            },
            step
        }) {
            step++;
            if (step == 1) {
                continue;
            }
        }
        return values[0] + values[1];
    }
    """


def test_for_update_assignment_uses_expression_preludes_for_mojo_codegen():
    generated_code = generate_code(
        parse_code(tokenize_code(_for_update_assignment_prelude_source()))
    )

    assert "while (step < 2):" in generated_code
    assert "if (step == 1):\n            var __cgl_match_value_0: Int32 = 0" in (
        generated_code
    )
    assert "__cgl_match_value_0 = 1" in generated_code
    assert "__cgl_match_value_0 = 2" in generated_code
    assert "values = InlineArray[Int32, 2](__cgl_match_value_0, step)" in (
        generated_code
    )
    assert (
        "values = InlineArray[Int32, 2](__cgl_match_value_0, step)\n"
        "            continue"
    ) in generated_code
    assert "var __cgl_match_value_1: Int32 = 0" in generated_code
    assert "values = InlineArray[Int32, 2](__cgl_match_value_1, step)" in (
        generated_code
    )
    assert "values = match mode" not in generated_code
    assert "MatchNode" not in generated_code


def test_for_update_assignment_contexts_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    generated_code = generate_code(
        parse_code(tokenize_code(_for_update_assignment_prelude_source()))
    )
    generated_code += """
fn main():
    print(updateArrayLoop(0))
"""

    source_path = tmp_path / "for_update_assignment_contexts.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "3" in result.stdout


def test_for_update_match_assignment_diagnostics_preserve_target_context():
    code = """
    Texture2D<float4> tex;

    float invalidForUpdate(int mode) {
        float scalar = 0.0;
        for (int i = 0; i < 1; scalar = match mode {
            0 => textureSize(tex, 0),
            _ => 1.0
        }) {
            i++;
        }
        return scalar;
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            r"texture_size target.*assignment target match arm 1"
            r".*expects ivec2.*got float"
        ),
    ):
        generate_code(parse_code(tokenize_code(code)))


def _expression_statement_assignment_wrapper_ast():
    source = """
    int wrappedAssign(int mode) {
        int values[2] = {0, 0};
        return values[0] + values[1];
    }
    """
    ast = parse_code(tokenize_code(source))

    int_type = PrimitiveType("int")

    def int_literal(value):
        return LiteralNode(value, int_type)

    match_expr = MatchNode(
        IdentifierNode("mode"),
        [
            MatchArmNode(
                LiteralPatternNode(int_literal(0)),
                None,
                int_literal(1),
            ),
            MatchArmNode(WildcardPatternNode(), None, int_literal(2)),
        ],
    )
    assignment = AssignmentNode(
        IdentifierNode("values"),
        ArrayLiteralNode([match_expr, int_literal(3)]),
    )
    ast.functions[-1].body.statements.insert(
        1,
        ExpressionStatementNode(assignment, is_tail_expression=False),
    )
    return ast


def test_expression_statement_assignment_wrapper_uses_mojo_statement_preludes():
    generated_code = generate_code(_expression_statement_assignment_wrapper_ast())

    assert "var __cgl_match_value_0: Int32 = 0" in generated_code
    assert "__cgl_match_value_0 = 1" in generated_code
    assert "__cgl_match_value_0 = 2" in generated_code
    assert "values = InlineArray[Int32, 2](__cgl_match_value_0, 3)" in generated_code
    assert "ExpressionStatementNode" not in generated_code
    assert "MatchNode" not in generated_code


def test_expression_statement_assignment_wrapper_compile_smoke_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    generated_code = generate_code(_expression_statement_assignment_wrapper_ast())
    generated_code += """
fn main():
    print(wrappedAssign(0))
"""

    source_path = tmp_path / "expression_statement_assignment_wrapper.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "4" in result.stdout


def test_expression_statement_compound_assignment_diagnostics_preserve_target_context():
    source = """
    Texture2D<float4> tex;

    float invalidWrappedCompoundAssignment(int mode) {
        float scalar = 0.0;
        return scalar;
    }
    """
    ast = parse_code(tokenize_code(source))
    int_type = PrimitiveType("int")

    def int_literal(value):
        return LiteralNode(value, int_type)

    texture_size = FunctionCallNode(
        IdentifierNode("textureSize"),
        [IdentifierNode("tex"), int_literal(0)],
    )
    match_expr = MatchNode(
        IdentifierNode("mode"),
        [
            MatchArmNode(
                LiteralPatternNode(int_literal(0)),
                None,
                texture_size,
            ),
            MatchArmNode(
                WildcardPatternNode(), None, LiteralNode(1.0, PrimitiveType("float"))
            ),
        ],
    )
    ast.functions[-1].body.statements.insert(
        1,
        ExpressionStatementNode(
            AssignmentNode(IdentifierNode("scalar"), match_expr, "+="),
            is_tail_expression=False,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"texture_size target.*assignment target match arm 1"
            r".*expects ivec2.*got float"
        ),
    ):
        generate_code(ast)


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
    assert "var tint = SIMD[DType.float32, 4](1.0, 0.0, 0.0, 1.0)" in generated_code
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


def test_direct_cast_nodes_emit_mojo_casts_without_ast_repr():
    codegen = MojoCodeGen()
    scalar_expr = CastNode(
        VariableNode("index", PrimitiveType("int")),
        PrimitiveType("float"),
    )

    codegen.register_variable_type("index", "int")

    assert codegen.generate_expression(scalar_expr) == "Float32(index)"
    assert codegen.expression_result_type(scalar_expr) == "float"

    resource_codegen = MojoCodeGen()
    resource_codegen.register_variable_type("mode", "int")
    resource_codegen.register_variable_type("strip", "sampler1D")
    resource_codegen.register_variable_type("queryTex", "sampler2D")
    resource_codegen.register_variable_type("msTex", "sampler2DMS")
    resource_codegen.register_variable_type("msImage", "uimage2DMS")
    resource_codegen.register_variable_type("rowImage", "uimage1D")
    resource_codegen.register_variable_type("layers", "sampler2DArray")
    resource_codegen.register_variable_type("wideImage", "uimage2D")

    valid_resource_match = MatchNode(
        IdentifierNode("mode"),
        [
            MatchArmNode(
                LiteralPatternNode(LiteralNode(0, PrimitiveType("int"))),
                None,
                FunctionCallNode(
                    IdentifierNode("textureSize"),
                    [
                        IdentifierNode("strip"),
                        LiteralNode(0, PrimitiveType("int")),
                    ],
                ),
            ),
            MatchArmNode(
                WildcardPatternNode(),
                None,
                LiteralNode(7, PrimitiveType("int")),
            ),
        ],
    )
    prelude, expression = resource_codegen.generate_expression_with_prelude(
        CastNode(valid_resource_match, PrimitiveType("float")),
        1,
        "float",
        "return value",
    )

    assert "var __cgl_match_value_0: Int32" in prelude
    assert "__cgl_match_value_0 = texture_size(strip, 0)" in prelude
    assert "__cgl_match_value_0 = 7" in prelude
    assert expression == "Float32(__cgl_match_value_0)"
    assert (
        resource_codegen.generate_expression(
            CastNode(
                TextureOpNode(
                    "textureQueryLevels",
                    VariableNode("queryTex", PrimitiveType("sampler2D")),
                    [],
                ),
                PrimitiveType("float"),
            )
        )
        == "Float32(texture_query_levels(queryTex))"
    )
    assert (
        resource_codegen.generate_expression(
            CastNode(
                TextureOpNode(
                    "textureSamples",
                    VariableNode("msTex", PrimitiveType("sampler2DMS")),
                    [],
                ),
                PrimitiveType("float"),
            )
        )
        == "Float32(texture_samples(msTex))"
    )
    assert (
        resource_codegen.generate_expression(
            CastNode(
                TextureOpNode(
                    "imageSamples",
                    VariableNode("msImage", PrimitiveType("uimage2DMS")),
                    [],
                ),
                PrimitiveType("float"),
            )
        )
        == "Float32(image_samples(msImage))"
    )
    assert (
        resource_codegen.generate_expression(
            CastNode(
                TextureOpNode(
                    "GetDimensions",
                    VariableNode("rowImage", PrimitiveType("uimage1D")),
                    [],
                ),
                PrimitiveType("float"),
            )
        )
        == "Float32(image_size(rowImage))"
    )

    invalid_resource_match = MatchNode(
        IdentifierNode("mode"),
        [
            MatchArmNode(
                LiteralPatternNode(LiteralNode(0, PrimitiveType("int"))),
                None,
                FunctionCallNode(
                    IdentifierNode("textureSize"),
                    [
                        IdentifierNode("layers"),
                        LiteralNode(0, PrimitiveType("int")),
                    ],
                ),
            ),
            MatchArmNode(
                WildcardPatternNode(),
                None,
                LiteralNode(0, PrimitiveType("int")),
            ),
        ],
    )
    with pytest.raises(
        ValueError,
        match=(
            r"texture_size target.*cast to float match arm 1"
            r".*expects ivec3.*got float"
        ),
    ):
        resource_codegen.generate_expression_with_prelude(
            CastNode(invalid_resource_match, PrimitiveType("float")),
            1,
            "float",
            "return value",
        )

    invalid_image_match = MatchNode(
        IdentifierNode("mode"),
        [
            MatchArmNode(
                LiteralPatternNode(LiteralNode(0, PrimitiveType("int"))),
                None,
                FunctionCallNode(
                    IdentifierNode("imageSize"),
                    [IdentifierNode("wideImage")],
                ),
            ),
            MatchArmNode(
                WildcardPatternNode(),
                None,
                LiteralNode(0, PrimitiveType("int")),
            ),
        ],
    )
    with pytest.raises(
        ValueError,
        match=(
            r"image_size target.*cast to float match arm 1"
            r".*expects ivec2.*got float"
        ),
    ):
        resource_codegen.generate_expression_with_prelude(
            CastNode(invalid_image_match, PrimitiveType("float")),
            1,
            "float",
            "return value",
        )


def test_direct_vector_cast_nodes_emit_mojo_simd_casts():
    codegen = MojoCodeGen()
    vector_expr = CastNode(
        VariableNode("mask", VectorType(PrimitiveType("uint"), 4)),
        VectorType(PrimitiveType("float"), 4),
    )

    codegen.register_variable_type("mask", "uvec4")

    assert codegen.generate_expression(vector_expr) == "mask.cast[DType.float32]()"
    assert codegen.expression_result_type(vector_expr) == "vec4"


def _direct_scalar_match_for_mojo_codegen():
    def int_literal(value):
        return LiteralNode(value, PrimitiveType("int"))

    return MatchNode(
        IdentifierNode("mode"),
        [
            MatchArmNode(
                LiteralPatternNode(int_literal(0)),
                None,
                IdentifierNode("left"),
            ),
            MatchArmNode(WildcardPatternNode(), None, IdentifierNode("right")),
        ],
    )


def _register_direct_match_variables_for_mojo_codegen(codegen):
    codegen.register_variable_type("mode", "int")
    codegen.register_variable_type("left", "int")
    codegen.register_variable_type("right", "int")


def test_direct_vector_cast_match_emits_single_mojo_prelude_temp():
    codegen = MojoCodeGen()
    _register_direct_match_variables_for_mojo_codegen(codegen)
    vector_cast = CastNode(
        _direct_scalar_match_for_mojo_codegen(),
        VectorType(PrimitiveType("float"), 2),
    )

    prelude, expression = codegen.generate_expression_with_prelude(
        vector_cast,
        1,
        "vec2",
        "return value",
    )

    assert prelude.count("var __cgl_match_value_") == 1
    assert "var __cgl_match_value_0: Int32 = 0" in prelude
    assert "__cgl_match_value_1" not in prelude
    assert (
        expression
        == "SIMD[DType.float32, 2]((__cgl_match_value_0).cast[DType.float32]())"
    )


def test_direct_constructor_and_cast_match_require_mojo_expression_prelude():
    codegen = MojoCodeGen()
    _register_direct_match_variables_for_mojo_codegen(codegen)
    vector_constructor = FunctionCallNode(
        IdentifierNode("vec2"), [_direct_scalar_match_for_mojo_codegen()]
    )

    with pytest.raises(ValueError, match="expression prelude context is required"):
        codegen.generate_expression(vector_constructor)

    vector_cast = CastNode(
        _direct_scalar_match_for_mojo_codegen(),
        VectorType(PrimitiveType("float"), 2),
    )
    with pytest.raises(ValueError, match="expression prelude context is required"):
        codegen.generate_expression(vector_cast)


def test_direct_swizzle_nodes_emit_mojo_components_without_ast_repr():
    codegen = MojoCodeGen()
    swizzle_expr = SwizzleNode(
        VariableNode("color", VectorType(PrimitiveType("float"), 4)),
        "zyx",
    )

    codegen.register_variable_type("color", "vec4")

    assert (
        codegen.generate_expression(swizzle_expr)
        == "SIMD[DType.float32, 4](color[2], color[1], color[0], 0.0)"
    )
    assert codegen.expression_result_type(swizzle_expr) == "vec3"


def test_direct_builtin_variable_nodes_emit_mojo_names_without_ast_repr():
    codegen = MojoCodeGen()

    position = BuiltinVariableNode("gl_Position")
    dispatch_x = BuiltinVariableNode("SV_DispatchThreadID", "x")

    assert codegen.generate_expression(position) == "position"
    assert codegen.expression_result_type(position) == "vec4"
    assert codegen.generate_expression(dispatch_x) == "global_invocation_id[0]"
    assert codegen.expression_result_type(dispatch_x) == "uint"


def test_direct_texture_nodes_emit_mojo_sample_helpers_without_ast_repr():
    codegen = MojoCodeGen()
    tex = VariableNode("tex", PrimitiveType("sampler2D"))
    shadow = VariableNode("shadow", PrimitiveType("sampler2DShadow"))
    state = VariableNode("state", PrimitiveType("sampler"))
    uv = VariableNode("uv", VectorType(PrimitiveType("float"), 2))
    lod = VariableNode("lod", PrimitiveType("float"))

    codegen.register_variable_type("tex", "sampler2D")
    codegen.register_variable_type("shadow", "sampler2DShadow")
    codegen.register_variable_type("state", "sampler")
    codegen.register_variable_type("uv", "vec2")
    codegen.register_variable_type("lod", "float")

    sample_expr = TextureNode(tex, state, uv)
    no_sampler_expr = TextureNode(tex, None, uv)
    lod_expr = TextureNode(tex, state, uv, level=lod)
    shadow_expr = TextureNode(shadow, state, uv)

    assert codegen.generate_expression(sample_expr) == "sample(tex, uv)"
    assert codegen.generate_expression(no_sampler_expr) == "sample(tex, uv)"
    assert codegen.generate_expression(lod_expr) == "sample_lod(tex, uv, lod)"
    assert "TextureNode(" not in codegen.generate_expression(sample_expr)
    assert codegen.expression_result_type(sample_expr) == "vec4"
    assert codegen.expression_result_type(shadow_expr) == "float"
    assert "Texture2D" in codegen.required_resource_sample_types
    assert "Texture2D" in codegen.required_resource_lod_types


def test_direct_texture_nodes_emit_mojo_offset_helpers_without_ast_repr():
    codegen = MojoCodeGen()
    tex = VariableNode("tex", PrimitiveType("sampler2D"))
    state = VariableNode("state", PrimitiveType("sampler"))
    uv = VariableNode("uv", VectorType(PrimitiveType("float"), 2))
    lod = VariableNode("lod", PrimitiveType("float"))
    offset = VariableNode("offset", VectorType(PrimitiveType("int"), 2))

    codegen.register_variable_type("tex", "sampler2D")
    codegen.register_variable_type("state", "sampler")
    codegen.register_variable_type("uv", "vec2")
    codegen.register_variable_type("lod", "float")
    codegen.register_variable_type("offset", "ivec2")

    offset_call = codegen.generate_expression(
        TextureNode(tex, state, uv, offset=offset)
    )
    lod_offset_call = codegen.generate_expression(
        TextureNode(tex, state, uv, level=lod, offset=offset)
    )

    assert offset_call.startswith("_crossgl_sample_offset_Texture2D")
    assert offset_call.endswith("(tex, uv, offset)")
    assert lod_offset_call.startswith("_crossgl_sample_lod_offset_Texture2D")
    assert lod_offset_call.endswith("(tex, uv, lod, offset)")
    assert "TextureNode(" not in offset_call
    assert "state" not in offset_call
    assert (
        codegen.expression_result_type(TextureNode(tex, state, uv, offset=offset))
        == "vec4"
    )
    helper_names = {
        helper["name"] for helper in codegen.required_resource_builtin_helpers.values()
    }
    assert any(
        name.startswith("_crossgl_sample_offset_Texture2D") for name in helper_names
    )
    assert any(
        name.startswith("_crossgl_sample_lod_offset_Texture2D") for name in helper_names
    )


def test_direct_texture_op_nodes_emit_mojo_sample_helpers_without_ast_repr():
    codegen = MojoCodeGen()
    tex = VariableNode("tex", PrimitiveType("sampler2D"))
    state = VariableNode("state", PrimitiveType("sampler"))
    uv = VariableNode("uv", VectorType(PrimitiveType("float"), 2))
    lod = VariableNode("lod", PrimitiveType("float"))
    ddx = VariableNode("ddx", VectorType(PrimitiveType("float"), 2))
    ddy = VariableNode("ddy", VectorType(PrimitiveType("float"), 2))

    codegen.register_variable_type("tex", "sampler2D")
    codegen.register_variable_type("state", "sampler")
    codegen.register_variable_type("uv", "vec2")
    codegen.register_variable_type("lod", "float")
    codegen.register_variable_type("ddx", "vec2")
    codegen.register_variable_type("ddy", "vec2")

    sample_call = codegen.generate_expression(
        TextureOpNode("Sample", tex, [uv], sampler_expr=state)
    )
    lod_call = codegen.generate_expression(
        TextureOpNode("SampleLevel", tex, [uv, lod], sampler_expr=state)
    )
    grad_call = codegen.generate_expression(
        TextureOpNode("SampleGrad", tex, [uv, ddx, ddy], sampler_expr=state)
    )

    assert sample_call == "sample(tex, uv)"
    assert lod_call == "sample_lod(tex, uv, lod)"
    assert grad_call == "sample_grad(tex, uv, ddx, ddy)"
    assert "TextureOpNode(" not in sample_call
    assert (
        codegen.expression_result_type(
            TextureOpNode("Sample", tex, [uv], sampler_expr=state)
        )
        == "vec4"
    )
    assert "Texture2D" in codegen.required_resource_sample_types
    assert "Texture2D" in codegen.required_resource_lod_types
    assert "Texture2D" in codegen.required_resource_grad_types


def test_direct_texture_op_nodes_emit_mojo_resource_helpers_without_ast_repr():
    codegen = MojoCodeGen()
    tex = VariableNode("tex", PrimitiveType("sampler2D"))
    ms_tex = VariableNode("msTex", PrimitiveType("sampler2DMS"))
    ms_image = VariableNode("msImage", PrimitiveType("uimage2DMS"))
    shadow = VariableNode("shadow", PrimitiveType("sampler2DShadow"))
    state = VariableNode("state", PrimitiveType("sampler"))
    uv = VariableNode("uv", VectorType(PrimitiveType("float"), 2))
    coord = VariableNode("coord", VectorType(PrimitiveType("int"), 2))
    depth = VariableNode("depth", PrimitiveType("float"))

    codegen.register_variable_type("tex", "sampler2D")
    codegen.register_variable_type("msTex", "sampler2DMS")
    codegen.register_variable_type("msImage", "uimage2DMS")
    codegen.register_variable_type("shadow", "sampler2DShadow")
    codegen.register_variable_type("state", "sampler")
    codegen.register_variable_type("uv", "vec2")
    codegen.register_variable_type("coord", "ivec2")
    codegen.register_variable_type("depth", "float")

    load_call = codegen.generate_expression(TextureOpNode("Load", tex, [coord]))
    size_call = codegen.generate_expression(TextureOpNode("GetDimensions", tex, []))
    gather_call = codegen.generate_expression(
        TextureOpNode("Gather", tex, [uv], sampler_expr=state)
    )
    compare_call = codegen.generate_expression(
        TextureOpNode("SampleCmp", shadow, [uv, depth], sampler_expr=state)
    )
    texture_samples_call = codegen.generate_expression(
        TextureOpNode("textureSamples", ms_tex, [])
    )
    image_samples_call = codegen.generate_expression(
        TextureOpNode("imageSamples", ms_image, [])
    )

    assert load_call == "texel_fetch(tex, coord, 0)"
    assert size_call == "texture_size(tex)"
    assert gather_call.startswith("_crossgl_texture_gather_Texture2D")
    assert gather_call.endswith("(tex, uv)")
    assert compare_call.startswith("_crossgl_texture_compare_Texture2DShadow")
    assert compare_call.endswith("(shadow, uv, depth)")
    assert texture_samples_call == "texture_samples(msTex)"
    assert image_samples_call == "image_samples(msImage)"
    assert "TextureOpNode(" not in gather_call
    assert codegen.expression_result_type(TextureOpNode("Load", tex, [coord])) == "vec4"
    assert (
        codegen.expression_result_type(TextureOpNode("GetDimensions", tex, []))
        == "ivec2"
    )
    assert (
        codegen.expression_result_type(
            TextureOpNode("SampleCmp", shadow, [uv, depth], sampler_expr=state)
        )
        == "float"
    )
    assert (
        codegen.expression_result_type(TextureOpNode("textureSamples", ms_tex, []))
        == "int"
    )
    assert (
        codegen.expression_result_type(TextureOpNode("imageSamples", ms_image, []))
        == "int"
    )
    helper_names = {
        helper["name"] for helper in codegen.required_resource_builtin_helpers.values()
    }
    assert any(
        name.startswith("_crossgl_texture_gather_Texture2D") for name in helper_names
    )
    assert any(
        name.startswith("_crossgl_texture_compare_Texture2DShadow")
        for name in helper_names
    )
    assert ("texture_samples", "Texture2DMS") in (
        codegen.required_resource_sample_count_helpers
    )
    assert ("image_samples", "UImage2DMS") in (
        codegen.required_resource_sample_count_helpers
    )


def test_direct_texture_op_nodes_emit_advanced_mojo_resource_helpers_without_ast_repr():
    codegen = MojoCodeGen()
    tex = VariableNode("tex", PrimitiveType("sampler2D"))
    array_tex = VariableNode("arrayTex", PrimitiveType("sampler2DArray"))
    shadow = VariableNode("shadow", PrimitiveType("sampler2DShadow"))
    array_shadow = VariableNode("arrayShadow", PrimitiveType("sampler2DArrayShadow"))
    state = VariableNode("state", PrimitiveType("sampler"))
    uv = VariableNode("uv", VectorType(PrimitiveType("float"), 2))
    uv_layer = VariableNode("uvLayer", VectorType(PrimitiveType("float"), 3))
    wide_uv = VariableNode("wideUv", VectorType(PrimitiveType("float"), 3))
    scalar_pixel = VariableNode("scalarPixel", PrimitiveType("int"))
    bias = VariableNode("bias", PrimitiveType("float"))
    lod = VariableNode("lod", PrimitiveType("float"))
    depth = VariableNode("depth", PrimitiveType("float"))
    offset = VariableNode("offset", VectorType(PrimitiveType("int"), 2))
    ddx = VariableNode("ddx", VectorType(PrimitiveType("float"), 2))
    ddy = VariableNode("ddy", VectorType(PrimitiveType("float"), 2))

    codegen.register_variable_type("tex", "sampler2D")
    codegen.register_variable_type("arrayTex", "sampler2DArray")
    codegen.register_variable_type("shadow", "sampler2DShadow")
    codegen.register_variable_type("arrayShadow", "sampler2DArrayShadow")
    codegen.register_variable_type("state", "sampler")
    codegen.register_variable_type("uv", "vec2")
    codegen.register_variable_type("uvLayer", "vec3")
    codegen.register_variable_type("wideUv", "vec3")
    codegen.register_variable_type("scalarPixel", "int")
    codegen.register_variable_type("bias", "float")
    codegen.register_variable_type("lod", "float")
    codegen.register_variable_type("depth", "float")
    codegen.register_variable_type("offset", "ivec2")
    codegen.register_variable_type("ddx", "vec2")
    codegen.register_variable_type("ddy", "vec2")

    bias_call = codegen.generate_expression(
        TextureOpNode("SampleBias", tex, [uv, bias], sampler_expr=state)
    )
    bias_offset_call = codegen.generate_expression(
        TextureOpNode("SampleBias", tex, [uv, bias, offset], sampler_expr=state)
    )
    compare_lod_call = codegen.generate_expression(
        TextureOpNode("SampleCmpLevel", shadow, [uv, depth, lod], sampler_expr=state)
    )
    compare_lod_offset_call = codegen.generate_expression(
        TextureOpNode(
            "SampleCmpLevel",
            shadow,
            [uv, depth, lod, offset],
            sampler_expr=state,
        )
    )
    compare_grad_call = codegen.generate_expression(
        TextureOpNode(
            "SampleCmpGrad",
            shadow,
            [uv, depth, ddx, ddy],
            sampler_expr=state,
        )
    )
    compare_grad_offset_call = codegen.generate_expression(
        TextureOpNode(
            "SampleCmpGrad",
            shadow,
            [uv, depth, ddx, ddy, offset],
            sampler_expr=state,
        )
    )
    gather_offsets_call = codegen.generate_expression(
        TextureOpNode(
            "textureGatherOffsets",
            tex,
            [uv, offset, offset, offset, offset],
            sampler_expr=state,
        )
    )
    compare_proj_grad_offset_call = codegen.generate_expression(
        TextureOpNode(
            "textureCompareProjGradOffset",
            shadow,
            [uv, depth, ddx, ddy, offset],
            sampler_expr=state,
        )
    )
    query_lod_call = codegen.generate_expression(
        TextureOpNode("textureQueryLod", array_tex, [uv_layer], sampler_expr=state)
    )
    calculate_lod_call = codegen.generate_expression(
        TextureOpNode(
            "CalculateLevelOfDetail", array_tex, [uv_layer], sampler_expr=state
        )
    )
    calculate_lod_unclamped_call = codegen.generate_expression(
        TextureOpNode(
            "CalculateLevelOfDetailUnclamped",
            array_tex,
            [uv_layer],
            sampler_expr=state,
        )
    )
    gather_cmp_red_call = codegen.generate_expression(
        TextureOpNode(
            "GatherCmpRed", array_shadow, [uv_layer, depth], sampler_expr=state
        )
    )
    gather_cmp_alpha_offset_call = codegen.generate_expression(
        TextureOpNode(
            "GatherCmpAlpha",
            array_shadow,
            [uv_layer, depth, offset],
            sampler_expr=state,
        )
    )

    assert bias_call.startswith("_crossgl_sample_bias_Texture2D")
    assert bias_call.endswith("(tex, uv, bias)")
    assert bias_offset_call.startswith("_crossgl_sample_bias_offset_Texture2D")
    assert bias_offset_call.endswith("(tex, uv, bias, offset)")
    assert compare_lod_call.startswith("_crossgl_texture_compare_lod_Texture2DShadow")
    assert compare_lod_call.endswith("(shadow, uv, depth, lod)")
    assert compare_lod_offset_call.startswith(
        "_crossgl_texture_compare_lod_offset_Texture2DShadow"
    )
    assert compare_lod_offset_call.endswith("(shadow, uv, depth, lod, offset)")
    assert compare_grad_call.startswith("_crossgl_texture_compare_grad_Texture2DShadow")
    assert compare_grad_call.endswith("(shadow, uv, depth, ddx, ddy)")
    assert compare_grad_offset_call.startswith(
        "_crossgl_texture_compare_grad_offset_Texture2DShadow"
    )
    assert compare_grad_offset_call.endswith("(shadow, uv, depth, ddx, ddy, offset)")
    assert gather_offsets_call.startswith("_crossgl_texture_gather_offsets_Texture2D")
    assert gather_offsets_call.endswith("(tex, uv, offset, offset, offset, offset)")
    assert compare_proj_grad_offset_call.startswith(
        "_crossgl_texture_compare_proj_grad_offset_Texture2DShadow"
    )
    assert compare_proj_grad_offset_call.endswith(
        "(shadow, uv, depth, ddx, ddy, offset)"
    )
    assert query_lod_call.startswith("_crossgl_texture_query_lod_Texture2DArray")
    assert query_lod_call.endswith("(arrayTex, uvLayer)")
    assert calculate_lod_call.startswith(
        "_crossgl_calculate_level_of_detail_Texture2DArray"
    )
    assert calculate_lod_call.endswith("(arrayTex, uvLayer)")
    assert calculate_lod_unclamped_call.startswith(
        "_crossgl_calculate_level_of_detail_unclamped_Texture2DArray"
    )
    assert calculate_lod_unclamped_call.endswith("(arrayTex, uvLayer)")
    assert gather_cmp_red_call.startswith(
        "_crossgl_texture_gather_compare_Texture2DArrayShadow"
    )
    assert gather_cmp_red_call.endswith("(arrayShadow, uvLayer, depth)")
    assert gather_cmp_alpha_offset_call.startswith(
        "_crossgl_texture_gather_compare_offset_Texture2DArrayShadow"
    )
    assert gather_cmp_alpha_offset_call.endswith(
        "(arrayShadow, uvLayer, depth, offset)"
    )
    assert "TextureOpNode(" not in "\n".join(
        [
            bias_call,
            bias_offset_call,
            compare_lod_call,
            compare_lod_offset_call,
            compare_grad_call,
            compare_grad_offset_call,
            gather_offsets_call,
            compare_proj_grad_offset_call,
            query_lod_call,
            calculate_lod_call,
            calculate_lod_unclamped_call,
            gather_cmp_red_call,
            gather_cmp_alpha_offset_call,
        ]
    )
    assert (
        codegen.expression_result_type(
            TextureOpNode("SampleBias", tex, [uv, bias], sampler_expr=state)
        )
        == "vec4"
    )
    assert (
        codegen.expression_result_type(
            TextureOpNode(
                "SampleCmpLevel", shadow, [uv, depth, lod], sampler_expr=state
            )
        )
        == "float"
    )
    assert (
        codegen.expression_result_type(
            TextureOpNode(
                "SampleCmpGrad", shadow, [uv, depth, ddx, ddy], sampler_expr=state
            )
        )
        == "float"
    )
    assert (
        codegen.expression_result_type(
            TextureOpNode("textureQueryLod", array_tex, [uv_layer], sampler_expr=state)
        )
        == "vec2"
    )
    assert (
        codegen.expression_result_type(
            TextureOpNode(
                "CalculateLevelOfDetail",
                array_tex,
                [uv_layer],
                sampler_expr=state,
            )
        )
        == "float"
    )
    helper_names = {
        helper["name"] for helper in codegen.required_resource_builtin_helpers.values()
    }
    assert any(
        name.startswith("_crossgl_sample_bias_Texture2D") for name in helper_names
    )
    assert any(
        name.startswith("_crossgl_sample_bias_offset_Texture2D")
        for name in helper_names
    )
    assert any(
        name.startswith("_crossgl_texture_compare_lod_Texture2DShadow")
        for name in helper_names
    )
    assert any(
        name.startswith("_crossgl_texture_compare_grad_Texture2DShadow")
        for name in helper_names
    )
    assert any(
        name.startswith("_crossgl_texture_gather_offsets_Texture2D")
        for name in helper_names
    )
    assert any(
        name.startswith("_crossgl_texture_compare_proj_grad_offset_Texture2DShadow")
        for name in helper_names
    )
    assert any(
        name.startswith("_crossgl_texture_query_lod_Texture2DArray")
        for name in helper_names
    )
    assert any(
        name.startswith("_crossgl_calculate_level_of_detail_Texture2DArray")
        for name in helper_names
    )
    assert any(
        name.startswith("_crossgl_calculate_level_of_detail_unclamped_Texture2DArray")
        for name in helper_names
    )
    assert any(
        name.startswith("_crossgl_texture_gather_compare_Texture2DArrayShadow")
        for name in helper_names
    )

    with pytest.raises(
        ValueError, match="texture_compare_grad.*shadow texture required"
    ):
        codegen.generate_expression(
            TextureOpNode("SampleCmpGrad", tex, [uv, depth, ddx, ddy])
        )
    with pytest.raises(
        ValueError, match="texture_query_lod.*non-shadow texture required"
    ):
        codegen.generate_expression(
            TextureOpNode("textureQueryLod", shadow, [uv], sampler_expr=state)
        )
    with pytest.raises(
        ValueError, match="calculate_level_of_detail.*non-shadow texture required"
    ):
        codegen.generate_expression(
            TextureOpNode("CalculateLevelOfDetail", shadow, [uv], sampler_expr=state)
        )
    with pytest.raises(ValueError, match="sample.*coordinate.*Texture2D"):
        codegen.generate_expression(
            TextureOpNode("Sample", tex, [wide_uv], sampler_expr=state)
        )
    with pytest.raises(ValueError, match="texel_fetch.*coordinate.*Texture2D"):
        codegen.generate_expression(TextureOpNode("Load", tex, [scalar_pixel]))


def test_direct_texture_op_nodes_reject_unknown_operations():
    codegen = MojoCodeGen()
    tex = VariableNode("tex", PrimitiveType("sampler2D"))
    uv = VariableNode("uv", VectorType(PrimitiveType("float"), 2))

    codegen.register_variable_type("tex", "sampler2D")
    codegen.register_variable_type("uv", "vec2")

    with pytest.raises(
        ValueError, match="Unsupported Mojo texture operation SampleClamp"
    ):
        codegen.generate_expression(TextureOpNode("SampleClamp", tex, [uv]))


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


def test_lambda_call_materializes_local_mojo_functions_for_typed_parameters():
    code = """
    shader LambdaShader {
        compute {
            void main() {
                let folded = fold(values, 0, lambda(int acc, int x, (acc + x)));
                let mapped = map(colors, lambda(vec3 color, { return color; }));
                let genericVector = map(colors, lambda(vec3<f32> color, color));
                let always = lambda(true);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn _crossgl_lambda_0(acc: Int32, x: Int32) -> Int32:" in generated_code
    assert "return (acc + x)" in generated_code
    assert "var folded = fold(values, 0, _crossgl_lambda_0)" in generated_code
    assert (
        "fn _crossgl_lambda_1(color: SIMD[DType.float32, 4]) -> "
        "SIMD[DType.float32, 4]:" in generated_code
    )
    assert "var mapped = map(colors, _crossgl_lambda_1)" in generated_code
    assert (
        "fn _crossgl_lambda_2(color: SIMD[DType.float32, 4]) -> "
        "SIMD[DType.float32, 4]:" in generated_code
    )
    assert "var genericVector = map(colors, _crossgl_lambda_2)" in generated_code
    assert "fn _crossgl_lambda_3() -> Bool:" in generated_code
    assert "return True" in generated_code
    assert "var always = _crossgl_lambda_3" in generated_code
    assert "lambda(" not in generated_code


def test_lambda_call_preserves_unsupported_mojo_lambda_shapes():
    code = """
    shader LambdaFallbackShader {
        compute {
            void main() {
                let mapped = map(values, lambda(x, (x + 1)));
                let generic = map(values, lambda(Result<i32, i32> value, { return value; }));
                let tupled = map(values, lambda((i32, i32) pair, { return pair; }));
                let qualified = map(values, lambda(const int value, value));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "lambda(x, (x + 1))" in generated_code
    assert "lambda(Result<i32, i32> value, { return value; })" in generated_code
    assert "lambda((i32, i32) pair, { return pair; })" in generated_code
    assert "lambda(const int value, value)" in generated_code
    assert "_crossgl_lambda_" not in generated_code


def test_lambda_call_materializes_before_mojo_assignment_statement():
    code = """
    shader LambdaAssignmentShader {
        compute {
            void main() {
                int folded = 0;
                folded = fold(values, 0, lambda(int acc, int x, (acc + x)));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    helper = "fn _crossgl_lambda_0(acc: Int32, x: Int32) -> Int32:"
    assignment = "folded = fold(values, 0, _crossgl_lambda_0)"
    assert helper in generated_code
    assert assignment in generated_code
    assert generated_code.index(helper) < generated_code.index(assignment)
    assert "lambda(" not in generated_code


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


def test_bool_scalar_mix_lowers_to_mojo_select():
    code = """
    shader main {
        compute {
            float nextX() {
                return 0.0;
            }

            float nextY() {
                return 1.0;
            }

            bool nextFlag() {
                return true;
            }

            void main() {
                bool flag = true;
                float literal = mix(0.0, 1.0, flag);
                float complexValue = mix(nextX(), nextY(), nextFlag());
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "var literal: Float32 = (1.0 if flag else 0.0)" in generated_code
    assert (
        "var complexValue: Float32 = "
        "(nextY() if nextFlag() else nextX())" in generated_code
    )
    assert "lerp(0.0, 1.0, flag)" not in generated_code
    assert "lerp(nextX(), nextY(), nextFlag())" not in generated_code
    assert "mix(0.0, 1.0, flag)" not in generated_code
    assert "mix(nextX(), nextY(), nextFlag())" not in generated_code


def test_bool_vector_mix_and_ternary_lower_to_mojo_lane_select():
    code = """
    bvec3 makeMask() {
        return bvec3(true, false, true);
    }

    vec3 makeA() {
        return vec3(1.0, 2.0, 3.0);
    }

    vec3 makeB() {
        return vec3(4.0, 5.0, 6.0);
    }

    void probe(bvec3 mask, vec3 a, vec3 b) {
        vec3 selected = mix(a, b, mask);
        vec3 ternarySelected = mask ? b : a;
        vec3 complexSelected = mix(makeA(), makeB(), makeMask());
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "fn _crossgl_select_f32_3_4(mask: SIMD[DType.bool, 4], "
        "true_value: SIMD[DType.float32, 4], "
        "false_value: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:"
        in generated_code
    )
    assert (
        "return SIMD[DType.float32, 4]("
        "true_value[0] if mask[0] else false_value[0], "
        "true_value[1] if mask[1] else false_value[1], "
        "true_value[2] if mask[2] else false_value[2], 0.0)" in generated_code
    )
    assert (
        "var selected: SIMD[DType.float32, 4] = "
        "_crossgl_select_f32_3_4(mask, b, a)" in generated_code
    )
    assert (
        "var ternarySelected: SIMD[DType.float32, 4] = "
        "_crossgl_select_f32_3_4(mask, b, a)" in generated_code
    )
    assert (
        "var complexSelected: SIMD[DType.float32, 4] = "
        "_crossgl_select_f32_3_4(makeMask(), makeB(), makeA())" in generated_code
    )
    assert "lerp(a, b, mask)" not in generated_code
    assert "lerp(makeA(), makeB(), makeMask())" not in generated_code
    assert "(b if mask else a)" not in generated_code
    assert "mix(a, b, mask)" not in generated_code


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


def test_hlsl_style_matrix_aliases_emit_mojo_matrix_names():
    code = """
    shader main {
        compute {
            void main() {
                float3x2 hlslFloat = float3x2(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
                simd_float4x2 simdFloat;
                double2x4 precise = double2x4(
                    1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0
                );
                half3x2 halfMatrix;
                min16float2x3 minMatrix;
                f16mat2x3 f16Matrix = f16mat2x3(
                    1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct CrossGLMatrixF32C2R3:" in generated_code
    assert "struct CrossGLMatrixF32C2R4:" in generated_code
    assert "struct CrossGLMatrixF64C4R2:" in generated_code
    assert "struct CrossGLMatrixF16C2R3:" in generated_code
    assert "struct CrossGLMatrixF16C3R2:" in generated_code
    assert "var hlslFloat: CrossGLMatrixF32C2R3" in generated_code
    assert "var simdFloat: CrossGLMatrixF32C2R4" in generated_code
    assert "var precise: CrossGLMatrixF64C4R2" in generated_code
    assert "var halfMatrix: CrossGLMatrixF16C2R3" in generated_code
    assert "var minMatrix: CrossGLMatrixF16C3R2" in generated_code
    assert "var f16Matrix: CrossGLMatrixF16C2R3" in generated_code
    assert (
        "CrossGLMatrixF32C2R3(SIMD[DType.float32, 4](1.0, 2.0, 3.0, 0.0), "
        "SIMD[DType.float32, 4](4.0, 5.0, 6.0, 0.0))" in generated_code
    )
    assert (
        "CrossGLMatrixF64C4R2(SIMD[DType.float64, 2](1.0, 2.0), "
        "SIMD[DType.float64, 2](3.0, 4.0), "
        "SIMD[DType.float64, 2](5.0, 6.0), "
        "SIMD[DType.float64, 2](7.0, 8.0))" in generated_code
    )
    assert (
        "CrossGLMatrixF16C2R3(SIMD[DType.float16, 4](1.0, 2.0, 3.0, 0.0), "
        "SIMD[DType.float16, 4](4.0, 5.0, 6.0, 0.0))" in generated_code
    )
    for raw_alias in (
        "float3x2(",
        "simd_float4x2(",
        "double2x4(",
        "half3x2(",
        "min16float2x3(",
        "f16mat2x3(",
    ):
        assert raw_alias not in generated_code


def test_hlsl_style_matrix_aliases_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    float3x2 makeFloat() {
        return float3x2(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    }

    double2x4 makeDouble() {
        return double2x4(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0
        );
    }

    half2x2 makeHalf() {
        return half2x2(1.0, 2.0, 3.0, 4.0);
    }

    min16float2x3 makeMin() {
        return min16float2x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    }

    f16mat2x3 makeF16Mat() {
        return f16mat2x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(makeFloat().c0)
    print(makeFloat().c1)
    print(makeDouble().c3)
    print(makeHalf().c1)
    print(makeMin().c2)
    print(makeF16Mat().c1)
"""

    source_path = tmp_path / "hlsl_style_matrix_aliases.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[1.0, 2.0, 3.0, 0.0]" in result.stdout
    assert "[4.0, 5.0, 6.0, 0.0]" in result.stdout
    assert "[7.0, 8.0]" in result.stdout


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


def test_mixed_vector_scalar_matrix_constructors_use_helpers():
    code = """
    vec2 makeUv() {
        return vec2(1.0, 2.0);
    }

    ivec2 makeIndexPair() {
        return ivec2(3, 4);
    }

    int nextIndex() {
        return 5;
    }

    float nextWeight() {
        return 6.0;
    }

    double nextPrecise() {
        return 7.0;
    }

    mat2 fromUvAndScalars() {
        return mat2(makeUv(), nextIndex(), nextPrecise());
    }

    mat2 fromIndexAndScalars() {
        return mat2(makeIndexPair(), nextWeight(), nextIndex());
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "fn _crossgl_construct_matrix_f32_c2_r2_2_vf322_01_si32_sf64" in generated_code
    )
    assert "fn _crossgl_construct_matrix_f32_c2_r2_2_vi322_01_s_si32" in generated_code
    assert "var v0_cast = v0.cast[DType.float32]()" in generated_code
    assert (
        "return _crossgl_construct_matrix_f32_c2_r2_2_vf322_01_si32_sf64("
        "makeUv(), nextIndex(), nextPrecise())"
    ) in generated_code
    assert (
        "return _crossgl_construct_matrix_f32_c2_r2_2_vi322_01_s_si32("
        "makeIndexPair(), nextWeight(), nextIndex())"
    ) in generated_code
    assert "makeUv()[0]" not in generated_code
    assert "makeIndexPair()[0]" not in generated_code


def test_mixed_vector_scalar_matrix_constructors_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    vec2 makeUv() {
        return vec2(1.0, 2.0);
    }

    ivec2 makeIndexPair() {
        return ivec2(3, 4);
    }

    int nextIndex() {
        return 5;
    }

    float nextWeight() {
        return 6.0;
    }

    double nextPrecise() {
        return 7.0;
    }

    mat2 fromUvAndScalars() {
        return mat2(makeUv(), nextIndex(), nextPrecise());
    }

    mat2 fromIndexAndScalars() {
        return mat2(makeIndexPair(), nextWeight(), nextIndex());
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(fromUvAndScalars().c0)
    print(fromUvAndScalars().c1)
    print(fromIndexAndScalars().c0)
    print(fromIndexAndScalars().c1)
"""

    source_path = tmp_path / "mixed_vector_scalar_matrix_constructors.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[1.0, 2.0]" in result.stdout
    assert "[5.0, 7.0]" in result.stdout
    assert "[3.0, 4.0]" in result.stdout
    assert "[6.0, 5.0]" in result.stdout


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
                vec3<f16> lowp = vec3<f16>(1.0, 2.0, 3.0);
                vec3<i32> index = vec3<i32>(1, 2, 3);
                vec4<i16> smallIndex = vec4<i16>(1, 2, 3, 4);
                vec4<u32> mask = vec4<u32>(1, 2, 3, 4);
                vec3<u16> smallMask = vec3<u16>(1, 2, 3);
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
        "var lowp: SIMD[DType.float16, 4] = "
        "SIMD[DType.float16, 4](1.0, 2.0, 3.0, 0.0)"
    ) in generated_code
    assert (
        "var index: SIMD[DType.int32, 4] = " "SIMD[DType.int32, 4](1, 2, 3, 0)"
    ) in generated_code
    assert (
        "var smallIndex: SIMD[DType.int16, 4] = " "SIMD[DType.int16, 4](1, 2, 3, 4)"
    ) in generated_code
    assert (
        "var mask: SIMD[DType.uint32, 4] = " "SIMD[DType.uint32, 4](1, 2, 3, 4)"
    ) in generated_code
    assert (
        "var smallMask: SIMD[DType.uint16, 4] = " "SIMD[DType.uint16, 4](1, 2, 3, 0)"
    ) in generated_code
    assert (
        "var flags: SIMD[DType.bool, 2] = " "SIMD[DType.bool, 2](True, False)"
    ) in generated_code
    assert "vec2<" not in generated_code
    assert "vec3<" not in generated_code
    assert "vec4<" not in generated_code


def test_generic_min_precision_vector_aliases_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    vec3<f16> makeF16(vec2<f16> pair, f16 z) {
        return vec3<f16>(pair, z);
    }

    vec4<i16> makeI16(vec2<i16> pair, i16 z, i16 w) {
        return vec4<i16>(pair, z, w);
    }

    vec3<u16> makeU16(vec2<u16> pair, u16 z) {
        return vec3<u16>(pair, z);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(makeF16(SIMD[DType.float16, 2](1.0, 2.0), Float16(3.0)))
    print(makeI16(SIMD[DType.int16, 2](4, 5), Int16(6), Int16(7)))
    print(makeU16(SIMD[DType.uint16, 2](8, 9), UInt16(10)))
"""

    source_path = tmp_path / "generic_min_precision_vector_aliases.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[4, 5, 6, 7]" in result.stdout
    assert "[8, 9, 10, 0]" in result.stdout


def test_generic_scalar_alias_vector_constructors_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    vec3<half> makeHalf(vec2<half> pair, half z) {
        return vec3<half>(pair, z);
    }

    vec4<min16float> makeMinFloat(
        vec2<min16float> pair, min16float z, min16float w
    ) {
        return vec4<min16float>(pair, z, w);
    }

    vec3<int16> makeInt16(vec2<int16> pair, int16 z) {
        return vec3<int16>(pair, z);
    }

    vec4<uint16> makeUInt16(vec2<uint16> pair, uint16 z, uint16 w) {
        return vec4<uint16>(pair, z, w);
    }

    vec3<min12int> makeMinInt(vec2<min12int> pair, min12int z) {
        return vec3<min12int>(pair, z);
    }

    vec3<min16uint> makeMinUInt(vec2<min16uint> pair, min16uint z) {
        return vec3<min16uint>(pair, z);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn makeHalf(pair: SIMD[DType.float16, 2], z: Float16)" in generated_code
    assert "fn makeInt16(pair: SIMD[DType.int16, 2], z: Int16)" in generated_code
    assert "fn makeUInt16(pair: SIMD[DType.uint16, 2], z: UInt16" in generated_code
    assert "return SIMD[DType.float16, 4](pair[0], pair[1], z, 0.0)" in generated_code
    assert "return SIMD[DType.int16, 4](pair[0], pair[1], z, 0)" in generated_code
    assert "return SIMD[DType.uint16, 4](pair[0], pair[1], z, 0)" in generated_code
    for raw_generic in (
        "vec3<half>",
        "vec4<min16float>",
        "vec3<int16>",
        "vec4<uint16>",
        "vec3<min12int>",
        "vec3<min16uint>",
    ):
        assert raw_generic not in generated_code

    generated_code += """
fn main():
    print(makeHalf(SIMD[DType.float16, 2](1.0, 2.0), Float16(3.0)))
    print(makeMinFloat(SIMD[DType.float16, 2](4.0, 5.0), Float16(6.0), Float16(7.0)))
    print(makeInt16(SIMD[DType.int16, 2](8, 9), Int16(10)))
    print(makeUInt16(SIMD[DType.uint16, 2](11, 12), UInt16(13), UInt16(14)))
    print(makeMinInt(SIMD[DType.int16, 2](15, 16), Int16(17)))
    print(makeMinUInt(SIMD[DType.uint16, 2](18, 19), UInt16(20)))
"""

    source_path = tmp_path / "generic_scalar_alias_vector_constructors.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[8, 9, 10, 0]" in result.stdout
    assert "[11, 12, 13, 14]" in result.stdout
    assert "[15, 16, 17, 0]" in result.stdout
    assert "[18, 19, 20, 0]" in result.stdout


def test_nested_generic_scalar_alias_vector_constructors_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    vec4<half> packHalf(half tail) {
        return vec4<half>(vec3<half>(1.0, 2.0, 3.0), tail);
    }

    vec4<int16> packInt16(int16 tail) {
        return vec4<int16>(vec3<int16>(1, 2, 3), tail);
    }

    vec4<uint16> packUInt16(uint16 tail) {
        return vec4<uint16>(vec3<uint16>(4, 5, 6), tail);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "_crossgl_construct_f16_4_vf164_012_s" in generated_code
    assert "_crossgl_construct_i16_4_vi164_012_s" in generated_code
    assert "_crossgl_construct_u16_4_vu164_012_s" in generated_code
    assert "SIMD[DType.float16, 4](SIMD[DType.float16" not in generated_code
    assert "SIMD[DType.int16, 4](SIMD[DType.int16" not in generated_code
    assert "SIMD[DType.uint16, 4](SIMD[DType.uint16" not in generated_code

    generated_code += """
fn main():
    print(packHalf(Float16(4.0)))
    print(packInt16(Int16(7)))
    print(packUInt16(UInt16(8)))
"""

    source_path = tmp_path / "nested_generic_scalar_alias_vectors.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[1, 2, 3, 7]" in result.stdout
    assert "[4, 5, 6, 8]" in result.stdout


def test_simd_and_packed_precision_vector_aliases_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct ExtraAliases {
        simd_half3 halfValue;
        simd_double2 preciseValue;
        packed_short3 packedSigned;
        simd_short4 simdSigned;
        packed_ushort2 packedUnsigned;
        simd_ushort3 simdUnsigned;
    };

    simd_half3 buildSimdHalf(simd_half2 pair, half z) {
        return simd_half3(pair, z);
    }

    simd_double3 buildSimdDouble(simd_double2 pair, double z) {
        return simd_double3(pair, z);
    }

    packed_short3 buildPackedShort(packed_short2 pair, short z) {
        return packed_short3(pair, z);
    }

    simd_short4 buildSimdShort(simd_short2 pair, short z, short w) {
        return simd_short4(pair, z, w);
    }

    packed_ushort3 buildPackedUShort(packed_ushort2 pair, ushort z) {
        return packed_ushort3(pair, z);
    }

    simd_ushort4 buildSimdUShort(simd_ushort2 pair, ushort z, ushort w) {
        return simd_ushort4(pair, z, w);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "var halfValue: SIMD[DType.float16, 4]" in generated_code
    assert "var preciseValue: SIMD[DType.float64, 2]" in generated_code
    assert "var packedSigned: SIMD[DType.int16, 4]" in generated_code
    assert "var simdSigned: SIMD[DType.int16, 4]" in generated_code
    assert "var packedUnsigned: SIMD[DType.uint16, 2]" in generated_code
    assert "var simdUnsigned: SIMD[DType.uint16, 4]" in generated_code
    assert "return SIMD[DType.float16, 4](pair[0], pair[1], z, 0.0)" in generated_code
    assert "return SIMD[DType.float64, 4](pair[0], pair[1], z, 0.0)" in generated_code
    assert "return SIMD[DType.int16, 4](pair[0], pair[1], z, 0)" in generated_code
    assert "return SIMD[DType.uint16, 4](pair[0], pair[1], z, 0)" in generated_code
    for raw_alias in (
        "simd_half2(",
        "simd_half3(",
        "simd_double2(",
        "simd_double3(",
        "packed_short2(",
        "packed_short3(",
        "simd_short2(",
        "simd_short4(",
        "packed_ushort2(",
        "packed_ushort3(",
        "simd_ushort2(",
        "simd_ushort4(",
    ):
        assert raw_alias not in generated_code

    generated_code += """
fn main():
    print(buildSimdHalf(SIMD[DType.float16, 2](1.0, 2.0), Float16(3.0)))
    print(buildSimdDouble(SIMD[DType.float64, 2](4.0, 5.0), 6.0))
    print(buildPackedShort(SIMD[DType.int16, 2](7, 8), Int16(9)))
    print(buildSimdShort(SIMD[DType.int16, 2](10, 11), Int16(12), Int16(13)))
    print(buildPackedUShort(SIMD[DType.uint16, 2](14, 15), UInt16(16)))
    print(buildSimdUShort(SIMD[DType.uint16, 2](17, 18), UInt16(19), UInt16(20)))
"""

    source_path = tmp_path / "simd_and_packed_precision_vector_aliases.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[7, 8, 9, 0]" in result.stdout
    assert "[10, 11, 12, 13]" in result.stdout
    assert "[14, 15, 16, 0]" in result.stdout
    assert "[17, 18, 19, 20]" in result.stdout


def test_simd_precision_matrix_aliases_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    simd_half2x2 makeSimdHalfMatrix() {
        return simd_half2x2(1.0, 2.0, 3.0, 4.0);
    }

    simd_double3x2 makeSimdDoubleMatrix() {
        return simd_double3x2(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn makeSimdHalfMatrix() -> CrossGLMatrixF16C2R2:" in generated_code
    assert "fn makeSimdDoubleMatrix() -> CrossGLMatrixF64C2R3:" in generated_code
    assert "simd_half2x2(" not in generated_code
    assert "simd_double3x2(" not in generated_code

    generated_code += """
fn main():
    print(makeSimdHalfMatrix().c1)
    print(makeSimdDoubleMatrix().c0)
    print(makeSimdDoubleMatrix().c1)
"""

    source_path = tmp_path / "simd_precision_matrix_aliases.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[3.0, 4.0]" in result.stdout
    assert "[1.0, 2.0, 3.0, 0.0]" in result.stdout
    assert "[4.0, 5.0, 6.0, 0.0]" in result.stdout


def test_integer_matrix_aliases_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    int2x2 makeIntMatrix() {
        return int2x2(1, 2, 3, 4);
    }

    uint2x3 makeUIntMatrix() {
        return uint2x3(1, 2, 3, 4, 5, 6);
    }

    short3x2 makeShortMatrix() {
        return short3x2(1, 2, 3, 4, 5, 6);
    }

    min12int2x2 makeMinSignedMatrix() {
        return min12int2x2(7, 8, 9, 10);
    }

    ushort2x2 makeUShortMatrix() {
        return ushort2x2(11, 12, 13, 14);
    }

    min16uint3x2 makeMinUnsignedMatrix() {
        return min16uint3x2(15, 16, 17, 18, 19, 20);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn makeIntMatrix() -> CrossGLMatrixI32C2R2:" in generated_code
    assert "fn makeUIntMatrix() -> CrossGLMatrixU32C3R2:" in generated_code
    assert "fn makeShortMatrix() -> CrossGLMatrixI16C2R3:" in generated_code
    assert "fn makeMinSignedMatrix() -> CrossGLMatrixI16C2R2:" in generated_code
    assert "fn makeUShortMatrix() -> CrossGLMatrixU16C2R2:" in generated_code
    assert "fn makeMinUnsignedMatrix() -> CrossGLMatrixU16C2R3:" in generated_code
    for raw_alias in (
        "int2x2(",
        "uint2x3(",
        "short3x2(",
        "min12int2x2(",
        "ushort2x2(",
        "min16uint3x2(",
    ):
        assert raw_alias not in generated_code

    generated_code += """
fn main():
    print(makeIntMatrix().c1)
    print(makeUIntMatrix().c2)
    print(makeShortMatrix().c1)
    print(makeMinSignedMatrix().c1)
    print(makeUShortMatrix().c1)
    print(makeMinUnsignedMatrix().c1)
"""

    source_path = tmp_path / "integer_matrix_aliases.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[3, 4]" in result.stdout
    assert "[5, 6]" in result.stdout
    assert "[4, 5, 6, 0]" in result.stdout
    assert "[9, 10]" in result.stdout
    assert "[13, 14]" in result.stdout
    assert "[18, 19, 20, 0]" in result.stdout


def test_integer_matrix_aliases_in_aggregates_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    struct PackedIntMatrices {
        int2x2 signedMatrix;
        short3x2 signedArray[2];
        min16uint3x2 unsignedMatrix;
    };

    cbuffer MatrixParams {
        min12int2x2 smallSigned;
        ushort2x2 smallUnsigned[2];
    };

    PackedIntMatrices buildPacked(min16int index) {
        PackedIntMatrices packed;
        packed.signedMatrix = int2x2(1, 2, 3, 4);
        packed.signedArray = {short3x2(5, 6, 7, 8, 9, 10)};
        packed.signedArray[index] = short3x2(11, 12, 13, 14, 15, 16);
        packed.unsignedMatrix = min16uint3x2(17, 18, 19, 20, 21, 22);
        return packed;
    }

    short3x2 mutateMatrix(short3x2 value, min16uint column) {
        value[column] = short3(23, 24, 25);
        return value;
    }

    short columnValue(PackedIntMatrices packed, min16uint arrayIndex, min16uint row) {
        return packed.signedArray[arrayIndex][1][row];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "var signedMatrix: CrossGLMatrixI32C2R2" in generated_code
    assert "var signedArray: InlineArray[CrossGLMatrixI16C2R3, 2]" in generated_code
    assert "var unsignedMatrix: CrossGLMatrixU16C2R3" in generated_code
    assert "var smallSigned: CrossGLMatrixI16C2R2" in generated_code
    assert "var smallUnsigned: InlineArray[CrossGLMatrixU16C2R2, 2]" in generated_code
    assert "packed.signedArray[int(index)]" in generated_code
    assert "value[int(column)] = SIMD[DType.int16, 4](23, 24, 25, 0)" in generated_code
    assert "packed.signedArray[int(arrayIndex)].c1[int(row)]" in generated_code

    generated_code += """
fn main():
    var packed = buildPacked(Int16(1))
    print(packed.signedMatrix.c1)
    print(packed.signedArray[0].c1)
    print(packed.signedArray[1].c1)
    print(packed.unsignedMatrix.c1)
    print(mutateMatrix(packed.signedArray[0], UInt16(1)).c1)
    print(columnValue(packed, UInt16(1), UInt16(2)))
"""

    source_path = tmp_path / "integer_matrix_alias_aggregates.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[3, 4]" in result.stdout
    assert "[8, 9, 10, 0]" in result.stdout
    assert "[14, 15, 16, 0]" in result.stdout
    assert "[20, 21, 22, 0]" in result.stdout
    assert "[23, 24, 25, 0]" in result.stdout
    assert "16" in result.stdout


def test_hlsl_style_vector_aliases_emit_mojo_simd_names():
    code = """
    struct HlslVectors {
        float2 uv;
        float3 normal;
        float4 color;
        packed_float4 packedColor;
        simd_float3 simdNormal;
        half3 hdr;
        min16float3 minHdr;
        min10float2 minUv;
        double2 precise;
        int3 index;
        uint4 mask;
        bool2 flags;
    };

    float4 buildColor(float2 uv, float z) {
        return float4(uv, z, 1.0);
    }

    packed_float4 buildPackedFloat(packed_float2 pair, float z, float w) {
        return packed_float4(pair, z, w);
    }

    simd_float3 buildSimdFloat(simd_float2 pair, float value) {
        return simd_float3(pair, value);
    }

    half3 buildHalf(half2 pair, half value) {
        return half3(pair, value);
    }

    packed_half4 buildPacked(packed_half2 pair, half z, half w) {
        return packed_half4(pair, z, w);
    }

    f16vec3 buildF16(f16vec2 pair, half value) {
        return f16vec3(pair, value);
    }

    min16float3 buildMin16(min16float2 pair, min16float value) {
        return min16float3(pair, value);
    }

    min10float4 buildMin10(min10float2 pair, min10float z, min10float w) {
        return min10float4(pair, z, w);
    }

    double3 buildPrecise(double2 pair, double value) {
        return double3(pair, value);
    }

    int3 buildIndex(int2 pair, int z) {
        return int3(pair, z);
    }

    uint3 buildMask(uint2 pair, uint z) {
        return uint3(pair, z);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "var uv: SIMD[DType.float32, 2]" in generated_code
    assert "var normal: SIMD[DType.float32, 4]" in generated_code
    assert "var color: SIMD[DType.float32, 4]" in generated_code
    assert "var packedColor: SIMD[DType.float32, 4]" in generated_code
    assert "var simdNormal: SIMD[DType.float32, 4]" in generated_code
    assert "var hdr: SIMD[DType.float16, 4]" in generated_code
    assert "var minHdr: SIMD[DType.float16, 4]" in generated_code
    assert "var minUv: SIMD[DType.float16, 2]" in generated_code
    assert "var precise: SIMD[DType.float64, 2]" in generated_code
    assert "var index: SIMD[DType.int32, 4]" in generated_code
    assert "var mask: SIMD[DType.uint32, 4]" in generated_code
    assert "var flags: SIMD[DType.bool, 2]" in generated_code
    assert "return SIMD[DType.float32, 4](uv[0], uv[1], z, 1.0)" in generated_code
    assert "return SIMD[DType.float32, 4](pair[0], pair[1], z, w)" in generated_code
    assert (
        "return SIMD[DType.float32, 4](pair[0], pair[1], value, 0.0)" in generated_code
    )
    assert (
        "return SIMD[DType.float16, 4](pair[0], pair[1], value, 0.0)" in generated_code
    )
    assert "return SIMD[DType.float16, 4](pair[0], pair[1], z, w)" in generated_code
    assert (
        "return SIMD[DType.float64, 4](pair[0], pair[1], value, 0.0)" in generated_code
    )
    assert "return SIMD[DType.int32, 4](pair[0], pair[1], z, 0)" in generated_code
    assert "return SIMD[DType.uint32, 4](pair[0], pair[1], z, 0)" in generated_code
    for raw_alias in (
        "float2(",
        "float3(",
        "float4(",
        "packed_float2(",
        "packed_float4(",
        "simd_float2(",
        "simd_float3(",
        "half2(",
        "half3(",
        "packed_half2(",
        "packed_half4(",
        "f16vec2(",
        "f16vec3(",
        "min16float2(",
        "min16float3(",
        "min10float2(",
        "min10float4(",
        "double2(",
        "double3(",
        "int2(",
        "int3(",
        "uint2(",
        "uint3(",
    ):
        assert raw_alias not in generated_code


def test_hlsl_style_vector_aliases_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    float4 makeColor(float2 uv, float z, float w) {
        return float4(uv, z, w);
    }

    packed_float4 makePackedFloat(packed_float2 pair, float z, float w) {
        return packed_float4(pair, z, w);
    }

    simd_float3 makeSimdFloat(simd_float2 pair, float z) {
        return simd_float3(pair, z);
    }

    half3 makeHalf(half2 pair, half z) {
        return half3(pair, z);
    }

    min16float3 makeMin16(min16float2 pair, min16float z) {
        return min16float3(pair, z);
    }

    min10float4 makeMin10(min10float2 pair, min10float z, min10float w) {
        return min10float4(pair, z, w);
    }

    double3 makePrecise(double2 pair, double z) {
        return double3(pair, z);
    }

    int3 makeIndex(int2 pair, int z) {
        return int3(pair, z);
    }

    uint4 makeMask(uint2 pair, uint z, uint w) {
        return uint4(pair, z, w);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(makeColor(SIMD[DType.float32, 2](1.0, 2.0), 3.0, 4.0))
    print(makePackedFloat(SIMD[DType.float32, 2](5.0, 6.0), 7.0, 8.0))
    print(makeSimdFloat(SIMD[DType.float32, 2](9.0, 10.0), 11.0))
    print(makeHalf(SIMD[DType.float16, 2](1.0, 2.0), Float16(3.0)))
    print(makeMin16(SIMD[DType.float16, 2](4.0, 5.0), Float16(6.0)))
    print(makeMin10(SIMD[DType.float16, 2](7.0, 8.0), Float16(9.0), Float16(10.0)))
    print(makePrecise(SIMD[DType.float64, 2](1.0, 2.0), 3.0))
    print(makeIndex(SIMD[DType.int32, 2](1, 2), 3))
    print(makeMask(SIMD[DType.uint32, 2](1, 2), UInt32(3), UInt32(4)))
"""

    source_path = tmp_path / "hlsl_style_vector_aliases.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[1.0, 2.0, 3.0, 4.0]" in result.stdout
    assert "[5.0, 6.0, 7.0, 8.0]" in result.stdout
    assert "[9.0, 10.0, 11.0, 0.0]" in result.stdout
    assert "[1, 2, 3, 0]" in result.stdout


def test_hlsl_style_integer_vector_aliases_emit_mojo_simd_names():
    code = """
    struct IntegerVectors {
        packed_int3 packedIndex;
        simd_int4 simdIndex;
        packed_uint2 packedMask;
        simd_uint3 simdMask;
        short3 shortIndex;
        ushort2 ushortMask;
        i16vec3 i16Index;
        u16vec4 u16Mask;
        min16int3 minSigned;
        min12int2 min12Signed;
        min16uint4 minUnsigned;
    };

    packed_int3 buildPackedInt(packed_int2 pair, int z) {
        return packed_int3(pair, z);
    }

    simd_int4 buildSimdInt(simd_int2 pair, int z, int w) {
        return simd_int4(pair, z, w);
    }

    packed_uint3 buildPackedUint(packed_uint2 pair, uint z) {
        return packed_uint3(pair, z);
    }

    simd_uint4 buildSimdUint(simd_uint2 pair, uint z, uint w) {
        return simd_uint4(pair, z, w);
    }

    min16int3 buildMin16(min16int2 pair, min16int z) {
        return min16int3(pair, z);
    }

    min12int4 buildMin12(min12int2 pair, min12int z, min12int w) {
        return min12int4(pair, z, w);
    }

    min16uint3 buildMinUint(min16uint2 pair, min16uint z) {
        return min16uint3(pair, z);
    }

    short3 buildShort(short2 pair, short z) {
        return short3(pair, z);
    }

    ushort4 buildUShort(ushort2 pair, ushort z, ushort w) {
        return ushort4(pair, z, w);
    }

    i16vec3 buildI16(i16vec2 pair, i16 z) {
        return i16vec3(pair, z);
    }

    u16vec4 buildU16(u16vec2 pair, u16 z, u16 w) {
        return u16vec4(pair, z, w);
    }

    min12int passSmall(min12int value) {
        return min12int(value);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "var packedIndex: SIMD[DType.int32, 4]" in generated_code
    assert "var simdIndex: SIMD[DType.int32, 4]" in generated_code
    assert "var packedMask: SIMD[DType.uint32, 2]" in generated_code
    assert "var simdMask: SIMD[DType.uint32, 4]" in generated_code
    assert "var shortIndex: SIMD[DType.int16, 4]" in generated_code
    assert "var ushortMask: SIMD[DType.uint16, 2]" in generated_code
    assert "var i16Index: SIMD[DType.int16, 4]" in generated_code
    assert "var u16Mask: SIMD[DType.uint16, 4]" in generated_code
    assert "var minSigned: SIMD[DType.int16, 4]" in generated_code
    assert "var min12Signed: SIMD[DType.int16, 2]" in generated_code
    assert "var minUnsigned: SIMD[DType.uint16, 4]" in generated_code
    assert "fn passSmall(value: Int16) -> Int16:" in generated_code
    assert "return Int16(value)" in generated_code
    assert "return SIMD[DType.int32, 4](pair[0], pair[1], z, 0)" in generated_code
    assert "return SIMD[DType.uint32, 4](pair[0], pair[1], z, 0)" in generated_code
    assert "return SIMD[DType.int16, 4](pair[0], pair[1], z, 0)" in generated_code
    assert "return SIMD[DType.uint16, 4](pair[0], pair[1], z, 0)" in generated_code
    for raw_alias in (
        "packed_int2(",
        "packed_int3(",
        "simd_int2(",
        "simd_int4(",
        "packed_uint2(",
        "packed_uint3(",
        "simd_uint2(",
        "simd_uint4(",
        "min16int2(",
        "min16int3(",
        "min12int2(",
        "min12int4(",
        "min16uint2(",
        "min16uint3(",
        "short2(",
        "short3(",
        "ushort2(",
        "ushort4(",
        "i16vec2(",
        "i16vec3(",
        "u16vec2(",
        "u16vec4(",
    ):
        assert raw_alias not in generated_code


def test_hlsl_style_integer_vector_aliases_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    packed_int3 makePackedInt(packed_int2 pair, int z) {
        return packed_int3(pair, z);
    }

    simd_uint4 makeSimdUint(simd_uint2 pair, uint z, uint w) {
        return simd_uint4(pair, z, w);
    }

    min16int3 makeMin16(min16int2 pair, min16int z) {
        return min16int3(pair, z);
    }

    min12int4 makeMin12(min12int2 pair, min12int z, min12int w) {
        return min12int4(pair, z, w);
    }

    min16uint3 makeMinUint(min16uint2 pair, min16uint z) {
        return min16uint3(pair, z);
    }

    i16vec3 makeI16(i16vec2 pair, i16 z) {
        return i16vec3(pair, z);
    }

    u16vec4 makeU16(u16vec2 pair, u16 z, u16 w) {
        return u16vec4(pair, z, w);
    }

    min12int passSmall(min12int value) {
        return min12int(value);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(makePackedInt(SIMD[DType.int32, 2](1, 2), 3))
    print(makeSimdUint(SIMD[DType.uint32, 2](4, 5), UInt32(6), UInt32(7)))
    print(makeMin16(SIMD[DType.int16, 2](8, 9), Int16(10)))
    print(makeMin12(SIMD[DType.int16, 2](11, 12), Int16(13), Int16(14)))
    print(makeMinUint(SIMD[DType.uint16, 2](15, 16), UInt16(17)))
    print(makeI16(SIMD[DType.int16, 2](18, 19), Int16(20)))
    print(makeU16(SIMD[DType.uint16, 2](21, 22), UInt16(23), UInt16(24)))
    print(passSmall(Int16(25)))
"""

    source_path = tmp_path / "hlsl_style_integer_vector_aliases.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[1, 2, 3, 0]" in result.stdout
    assert "[4, 5, 6, 7]" in result.stdout
    assert "[8, 9, 10, 0]" in result.stdout
    assert "[11, 12, 13, 14]" in result.stdout
    assert "[15, 16, 17, 0]" in result.stdout
    assert "[21, 22, 23, 24]" in result.stdout
    assert "25" in result.stdout


def test_min_precision_integer_mod_and_indices_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    min12int signedMod(min12int a, min12int b) {
        return mod(a, b);
    }

    min16uint unsignedMod(min16uint a, min16uint b) {
        return mod(a, b);
    }

    i32 signedI32Mod(i32 a, i32 b) {
        return mod(a, b);
    }

    u32 unsignedU32Mod(u32 a, u32 b) {
        return mod(a, b);
    }

    float pickWithSignedIndex(min16int index) {
        float values[2] = {1.0, 2.0};
        return values[index];
    }

    float pickWithUnsignedIndex(uint16 index) {
        float values[2] = {3.0, 4.0};
        return values[index];
    }

    float pickWithI32Index(i32 index) {
        float values[2] = {5.0, 6.0};
        return values[index];
    }

    float pickWithU32Index(u32 index) {
        float values[2] = {7.0, 8.0};
        return values[index];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "return (a % b)" in generated_code
    assert "fmod(a, b)" not in generated_code
    assert "fn signedI32Mod(a: Int32, b: Int32) -> Int32:" in generated_code
    assert "fn unsignedU32Mod(a: UInt32, b: UInt32) -> UInt32:" in generated_code
    assert "return values[int(index)]" in generated_code
    assert "return values[index]" not in generated_code

    generated_code += """
fn main():
    print(signedMod(Int16(7), Int16(3)))
    print(unsignedMod(UInt16(8), UInt16(3)))
    print(signedI32Mod(Int32(9), Int32(4)))
    print(unsignedU32Mod(UInt32(10), UInt32(4)))
    print(pickWithSignedIndex(Int16(1)))
    print(pickWithUnsignedIndex(UInt16(1)))
    print(pickWithI32Index(Int32(1)))
    print(pickWithU32Index(UInt32(1)))
"""

    source_path = tmp_path / "min_precision_integer_mod_indices.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "1" in result.stdout
    assert "2.0" in result.stdout
    assert "4.0" in result.stdout
    assert "6.0" in result.stdout
    assert "8.0" in result.stdout


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


def _duplicate_sensitive_image_constructor_source():
    return """
    RWTexture2D<float4> colors;
    RWTexture2D<uint> counters;

    float2 pixel() {
        return float2(1.0, 2.0);
    }

    int2 coord() {
        return int2(0, 1);
    }

    float4 packImageVec() {
        return float4(colors.Load(coord()).xy, pixel());
    }

    float2x2 packImageMatrix() {
        return float2x2(colors.Load(coord()).xy, pixel());
    }

    float3 splatAtomic() {
        return float3(float(counters.InterlockedAdd(coord(), 1u)));
    }

    float2x2 diagonalAtomic() {
        return float2x2(float(counters.InterlockedAdd(coord(), 2u)));
    }
    """


def test_duplicate_sensitive_image_constructors_emit_single_evaluation_helpers():
    generated_code = generate_code(
        parse_code(tokenize_code(_duplicate_sensitive_image_constructor_source()))
    )

    vector_helper = "_crossgl_construct_f32_4_vf324_01_vf322_01"
    matrix_helper = "_crossgl_construct_matrix_f32_c2_r2_2_vf324_01_vf322_01"
    diagonal_helper = "_crossgl_matrix_diagonal_f32_c2_r2"
    atomic_helper = "_crossgl_image_atomic_add_UImage2D_SIMD_DType_int32_2_UInt32"
    atomic_one = f"{atomic_helper}(counters, coord(), 1)"
    atomic_two = f"{atomic_helper}(counters, coord(), 2)"

    assert f"fn {vector_helper}" in generated_code
    assert f"fn {matrix_helper}" in generated_code
    assert f"fn {diagonal_helper}" in generated_code
    assert (
        f"return {vector_helper}(image_load(colors, coord()), pixel())"
        in generated_code
    )
    assert (
        f"return {matrix_helper}(image_load(colors, coord()), pixel())"
        in generated_code
    )
    assert f"return _crossgl_vec3_splat_f32(Float32({atomic_one}))" in generated_code
    assert f"return {diagonal_helper}(Float32({atomic_two}))" in generated_code
    assert generated_code.count(atomic_one) == 1
    assert generated_code.count(atomic_two) == 1
    assert "image_load(colors, coord())[0]" not in generated_code


def test_duplicate_sensitive_image_constructors_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    generated_code = generate_code(
        parse_code(tokenize_code(_duplicate_sensitive_image_constructor_source()))
    )
    generated_code += """
fn main():
    print(packImageVec())
    print(packImageMatrix().c0)
    print(splatAtomic())
    print(diagonalAtomic().c0)
"""

    source_path = tmp_path / "duplicate_sensitive_image_constructors.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "source, pattern",
    [
        (
            """
            RWTexture2D<uint> counters;
            int2 coord() { return int2(0, 1); }
            float4 invalidVector() {
                return float4(
                    counters.InterlockedAdd(coord(), 1u),
                    0.0,
                    0.0,
                    0.0
                );
            }
            """,
            r"image_atomic_add target.*vector constructor float4 argument 1"
            r".*expects uint.*got float",
        ),
        (
            """
            RWTexture2D<uint> counters;
            int2 coord() { return int2(0, 1); }
            float2 makeUv() { return float2(1.0, 2.0); }
            float4 invalidHelperVector() {
                return float4(
                    makeUv(),
                    counters.InterlockedAdd(coord(), 1u),
                    0.0
                );
            }
            """,
            r"image_atomic_add target.*constructor helper argument 2"
            r".*expects uint.*got float",
        ),
        (
            """
            RWTexture2D<uint> counters;
            int2 coord() { return int2(0, 1); }
            float2x2 invalidMatrix() {
                return float2x2(counters.InterlockedAdd(coord(), 1u));
            }
            """,
            r"image_atomic_add target.*matrix constructor float2x2 argument 1"
            r".*expects uint.*got float",
        ),
    ],
)
def test_image_atomic_constructor_scalar_targets_require_explicit_casts(
    source, pattern
):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def _duplicate_sensitive_buffer_constructor_source():
    return """
    StructuredBuffer<float2> pairs;
    ConsumeStructuredBuffer<float2> consumed;
    ConsumeStructuredBuffer<int> scalarConsumed;
    RWByteAddressBuffer raw;

    uint index() {
        return 0u;
    }

    float2 uv() {
        return float2(1.0, 2.0);
    }

    float4 packLoad() {
        return float4(pairs.Load(index()), uv());
    }

    float4 packConsume() {
        return float4(consumed.Consume(), uv());
    }

    float4 packReinterpret() {
        return float4(asfloat(raw.Load2(index())), uv());
    }

    float2x2 packMatrix() {
        return float2x2(pairs.Load(index()), uv());
    }

    float2x2 diagonalConsume() {
        return float2x2(float(scalarConsumed.Consume()));
    }
    """


def test_duplicate_sensitive_buffer_constructors_emit_single_evaluation_helpers():
    generated_code = generate_code(
        parse_code(tokenize_code(_duplicate_sensitive_buffer_constructor_source()))
    )

    vector_helper = "_crossgl_construct_f32_4_vf322_01_vf322_01"
    matrix_helper = "_crossgl_construct_matrix_f32_c2_r2_2_vf322_01_vf322_01"
    diagonal_helper = "_crossgl_matrix_diagonal_f32_c2_r2"

    assert (
        f"return {vector_helper}(buffer_load(pairs, index()), uv())" in generated_code
    )
    assert f"return {vector_helper}(buffer_consume(consumed), uv())" in generated_code
    assert (
        f"return {vector_helper}(asfloat(buffer_load2(raw, index())), uv())"
        in generated_code
    )
    assert (
        f"return {matrix_helper}(buffer_load(pairs, index()), uv())" in generated_code
    )
    assert (
        f"return {diagonal_helper}(Float32(buffer_consume(scalarConsumed)))"
        in generated_code
    )
    assert "buffer_load(pairs, index())[0]" not in generated_code
    assert "buffer_consume(consumed)[0]" not in generated_code
    assert "asfloat(buffer_load2(raw, index()))[0]" not in generated_code
    assert generated_code.count("buffer_consume(scalarConsumed)") == 1


def test_duplicate_sensitive_buffer_constructors_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    generated_code = generate_code(
        parse_code(tokenize_code(_duplicate_sensitive_buffer_constructor_source()))
    )
    generated_code += """
fn main():
    print(packLoad())
    print(packConsume())
    print(packReinterpret())
    print(packMatrix().c0)
    print(diagonalConsume().c0)
"""

    source_path = tmp_path / "duplicate_sensitive_buffer_constructors.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "source, pattern",
    [
        (
            """
            StructuredBuffer<float2> pairs;
            uint index() { return 0u; }
            float invalidLoad() {
                return float(pairs.Load(index()));
            }
            """,
            r"buffer_load target.*scalar constructor float argument 1"
            r".*expects float2.*got float",
        ),
        (
            """
            ConsumeStructuredBuffer<float2> consumed;
            float invalidConsume() {
                return float(consumed.Consume());
            }
            """,
            r"buffer_consume target.*scalar constructor float argument 1"
            r".*expects float2.*got float",
        ),
        (
            """
            RWByteAddressBuffer raw;
            uint index() { return 0u; }
            float invalidReinterpret() {
                return float(asfloat(raw.Load2(index())));
            }
            """,
            r"asfloat target.*scalar constructor float argument 1"
            r".*expects vec2.*got float",
        ),
    ],
)
def test_buffer_and_reinterpret_constructor_scalar_targets_reject_vectors(
    source, pattern
):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def _aggregate_buffer_reinterpret_target_source():
    return """
    struct PairBox {
        float2 value;
        float2 alt;
    };

    StructuredBuffer<float2> pairs;
    ConsumeStructuredBuffer<float2> consumed;
    RWByteAddressBuffer raw;

    uint index() {
        return 0u;
    }

    void acceptPairs(float2[2] values) {
    }

    void acceptBox(PairBox pairBox) {
    }

    float2 chooseBuffer(bool useRaw) {
        return useRaw ? asfloat(raw.Load2(index())) : pairs.Load(index());
    }

    float2[2] collectBufferValues(int mode, bool useRaw) {
        return {
            useRaw ? asfloat(raw.Load2(index())) : pairs.Load(index()),
            match mode {
                0 => consumed.Consume(),
                _ => asfloat(raw.Load2(index()))
            }
        };
    }

    PairBox makeBufferBox(int mode, bool useRaw) {
        return PairBox{
            value: useRaw ? asfloat(raw.Load2(index())) : pairs.Load(index()),
            alt: match mode {
                0 => consumed.Consume(),
                _ => asfloat(raw.Load2(index()))
            }
        };
    }

    PairBox makePositionalBufferBox(int mode, bool useRaw) {
        return PairBox(
            useRaw ? asfloat(raw.Load2(index())) : pairs.Load(index()),
            match mode {
                0 => consumed.Consume(),
                _ => asfloat(raw.Load2(index()))
            }
        );
    }

    void passBufferArray(int mode, bool useRaw) {
        acceptPairs({
            useRaw ? pairs.Load(index()) : asfloat(raw.Load2(index())),
            match mode {
                0 => asfloat(raw.Load2(index())),
                _ => consumed.Consume()
            }
        });
    }

    void passBufferBox(int mode, bool useRaw) {
        acceptBox(PairBox{
            value: useRaw ? pairs.Load(index()) : asfloat(raw.Load2(index())),
            alt: match mode {
                0 => asfloat(raw.Load2(index())),
                _ => consumed.Consume()
            }
        });
    }
    """


def test_aggregate_buffer_reinterpret_targets_preserve_contexts_for_mojo_codegen():
    generated_code = generate_code(
        parse_code(tokenize_code(_aggregate_buffer_reinterpret_target_source()))
    )

    array_type = "InlineArray[SIMD[DType.float32, 2], 2]"
    assert "fn chooseBuffer(useRaw: Bool) -> SIMD[DType.float32, 2]:" in (
        generated_code
    )
    assert (
        "return (asfloat(buffer_load2(raw, index())) if useRaw else "
        "buffer_load(pairs, index()))"
    ) in generated_code
    assert f"fn collectBufferValues(mode: Int32, useRaw: Bool) -> {array_type}:" in (
        generated_code
    )
    assert "var __cgl_match_value_0: SIMD[DType.float32, 2]" in generated_code
    assert "__cgl_match_value_0 = buffer_consume(consumed)" in generated_code
    assert "__cgl_match_value_0 = asfloat(buffer_load2(raw, index()))" in (
        generated_code
    )
    assert (
        f"return {array_type}((asfloat(buffer_load2(raw, index())) if useRaw "
        "else buffer_load(pairs, index())), __cgl_match_value_0)"
    ) in generated_code
    assert (
        "return PairBox(value=(asfloat(buffer_load2(raw, index())) if useRaw else"
        in generated_code
    )
    assert "return PairBox((asfloat(buffer_load2(raw, index())) if useRaw else" in (
        generated_code
    )
    assert f"acceptPairs({array_type}(" in generated_code
    assert "acceptBox(PairBox(value=(buffer_load(pairs, index()) if useRaw else" in (
        generated_code
    )
    assert "pairs.Load(" not in generated_code
    assert "raw.Load2(" not in generated_code
    assert "consumed.Consume(" not in generated_code


def test_aggregate_buffer_reinterpret_targets_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    generated_code = generate_code(
        parse_code(tokenize_code(_aggregate_buffer_reinterpret_target_source()))
    )
    generated_code += """
fn main():
    print(chooseBuffer(True))
    print(collectBufferValues(0, False)[0])
    print(makeBufferBox(0, True).value)
    print(makePositionalBufferBox(1, False).alt)
    passBufferArray(0, True)
    passBufferBox(1, False)
"""

    source_path = tmp_path / "aggregate_buffer_reinterpret_targets.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "source, pattern",
    [
        (
            """
            StructuredBuffer<float2> pairs;
            uint index() { return 0u; }
            float[2] invalidArray(bool choose) {
                return {choose ? pairs.Load(index()) : 1.0, 2.0};
            }
            """,
            r"buffer_load target.*return value array literal element 1 true branch"
            r".*expects float2.*got float",
        ),
        (
            """
            RWByteAddressBuffer raw;
            uint index() { return 0u; }
            float[2] invalidArray(bool choose) {
                return {choose ? asfloat(raw.Load2(index())) : 1.0, 2.0};
            }
            """,
            r"asfloat target.*return value array literal element 1 true branch"
            r".*expects vec2.*got float",
        ),
        (
            """
            struct ScalarBox {
                float value;
            };
            StructuredBuffer<float2> pairs;
            uint index() { return 0u; }
            ScalarBox invalidBox(bool choose) {
                return ScalarBox{
                    value: choose ? pairs.Load(index()) : float2(1.0, 2.0)
                };
            }
            """,
            r"buffer_load target.*return value field ScalarBox\.value true branch"
            r".*expects float2.*got float",
        ),
        (
            """
            struct ScalarBox {
                float value;
            };
            RWByteAddressBuffer raw;
            uint index() { return 0u; }
            ScalarBox invalidBox(bool choose) {
                return ScalarBox(
                    choose ? asfloat(raw.Load2(index())) : float2(1.0, 2.0)
                );
            }
            """,
            r"asfloat target.*return value field ScalarBox\.value true branch"
            r".*expects vec2.*got float",
        ),
        (
            """
            ConsumeStructuredBuffer<float2> consumed;
            float invalidMatch(int mode) {
                return match mode {
                    0 => consumed.Consume(),
                    _ => float2(1.0, 2.0)
                };
            }
            """,
            r"buffer_consume target.*return value match arm 1"
            r".*expects float2.*got float",
        ),
    ],
)
def test_aggregate_buffer_reinterpret_target_shape_diagnostics(source, pattern):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def _function_style_struct_assignment_target_source():
    return """
    struct MixedBox {
        float2 loaded;
        float2 imageValue;
        float2 rawValue;
    };

    StructuredBuffer<float2> pairs;
    ConsumeStructuredBuffer<float2> consumed;
    RWByteAddressBuffer raw;
    RWTexture2D<float2> image;

    uint index() {
        return 0u;
    }

    int2 pixel() {
        return int2(0, 0);
    }

    void acceptMixed(MixedBox value) {
    }

    void useBoxes(int mode, bool choose) {
        let inferred = MixedBox(
            choose ? pairs.Load(index()) : consumed.Consume(),
            image.Load(pixel()),
            match mode {
                0 => asfloat(raw.Load2(index())),
                _ => pairs.Load(index())
            }
        );
        MixedBox explicitValue = MixedBox(
            consumed.Consume(),
            choose ? image.Load(pixel()) : pairs.Load(index()),
            asfloat(raw.Load2(index()))
        );
        MixedBox assigned;
        assigned = MixedBox(
            match mode {
                0 => pairs.Load(index()),
                _ => consumed.Consume()
            },
            image.Load(pixel()),
            choose ? asfloat(raw.Load2(index())) : pairs.Load(index())
        );
        inferred = MixedBox(
            consumed.Consume(),
            image.Load(pixel()),
            asfloat(raw.Load2(index()))
        );
        acceptMixed(MixedBox(
            pairs.Load(index()),
            image.Load(pixel()),
            asfloat(raw.Load2(index()))
        ));
    }
    """


def test_function_style_struct_constructors_preserve_assignment_target_contexts():
    generated_code = generate_code(
        parse_code(tokenize_code(_function_style_struct_assignment_target_source()))
    )

    assert "fn useBoxes(mode: Int32, choose: Bool) -> None:" in generated_code
    assert "var __cgl_match_value_0: SIMD[DType.float32, 2]" in generated_code
    assert "__cgl_match_value_0 = asfloat(buffer_load2(raw, index()))" in (
        generated_code
    )
    assert "__cgl_match_value_0 = buffer_load(pairs, index())" in generated_code
    assert (
        "var inferred = MixedBox((buffer_load(pairs, index()) if choose else "
        "buffer_consume(consumed)), image_load(image, pixel()), __cgl_match_value_0)"
    ) in generated_code
    assert (
        "var explicitValue: MixedBox = MixedBox(buffer_consume(consumed), "
        "(image_load(image, pixel()) if choose else buffer_load(pairs, index())), "
        "asfloat(buffer_load2(raw, index())))"
    ) in generated_code
    assert "var assigned = MixedBox(" in generated_code
    assert "var __cgl_match_value_1: SIMD[DType.float32, 2]" in generated_code
    assert "__cgl_match_value_1 = buffer_load(pairs, index())" in generated_code
    assert "__cgl_match_value_1 = buffer_consume(consumed)" in generated_code
    assert (
        "assigned = MixedBox(__cgl_match_value_1, image_load(image, pixel()), "
        "(asfloat(buffer_load2(raw, index())) if choose else "
        "buffer_load(pairs, index())))"
    ) in generated_code
    assert (
        "inferred = MixedBox(buffer_consume(consumed), image_load(image, pixel()), "
        "asfloat(buffer_load2(raw, index())))"
    ) in generated_code
    assert (
        "acceptMixed(MixedBox(buffer_load(pairs, index()), image_load(image, "
        "pixel()), asfloat(buffer_load2(raw, index()))))"
    ) in generated_code


def test_function_style_struct_assignment_contexts_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    generated_code = generate_code(
        parse_code(tokenize_code(_function_style_struct_assignment_target_source()))
    )
    generated_code += """
fn main():
    useBoxes(0, True)
"""

    source_path = tmp_path / "function_style_struct_assignment_contexts.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "source, pattern",
    [
        (
            """
            struct ScalarBox {
                float value;
            };
            StructuredBuffer<float2> pairs;
            uint index() { return 0u; }
            void invalidInferred() {
                let badValue = ScalarBox(pairs.Load(index()));
            }
            """,
            r"buffer_load target.*declaration badValue field ScalarBox\.value"
            r".*expects float2.*got float",
        ),
        (
            """
            struct ScalarBox {
                float value;
            };
            RWTexture2D<float2> image;
            int2 pixel() { return int2(0, 0); }
            void invalidExplicit() {
                ScalarBox badValue = ScalarBox(image.Load(pixel()));
            }
            """,
            r"image_load target.*declaration badValue field ScalarBox\.value"
            r".*expects vec2.*got float",
        ),
        (
            """
            struct ScalarBox {
                float value;
            };
            RWByteAddressBuffer raw;
            uint index() { return 0u; }
            void invalidAssignment() {
                ScalarBox assigned;
                assigned = ScalarBox(asfloat(raw.Load2(index())));
            }
            """,
            r"asfloat target.*assignment target field ScalarBox\.value"
            r".*expects vec2.*got float",
        ),
        (
            """
            struct ScalarBox {
                float value;
            };
            ConsumeStructuredBuffer<float2> consumed;
            void invalidInferredAssignment() {
                let assigned = ScalarBox(1.0);
                assigned = ScalarBox(consumed.Consume());
            }
            """,
            r"buffer_consume target.*assignment target field ScalarBox\.value"
            r".*expects float2.*got float",
        ),
    ],
)
def test_function_style_struct_constructor_assignment_diagnostics(source, pattern):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def test_function_style_struct_constructor_field_diagnostics():
    source = """
    struct PairBox {
        float value;
        float alt;
    };
    void invalidExtraPositional() {
        let badValue = PairBox(1.0, 2.0, 3.0);
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            r"too many positional fields.*declaration badValue" r".*field 3.*PairBox"
        ),
    ):
        generate_code(parse_code(tokenize_code(source)))


def test_function_style_struct_constructor_missing_field_diagnostics():
    source = """
    struct PairBox {
        float value;
        float alt;
    };
    void invalidMissingField() {
        let badValue = PairBox(1.0);
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            r"missing field PairBox\.alt.*declaration badValue"
            r".*expected fields: value, alt"
        ),
    ):
        generate_code(parse_code(tokenize_code(source)))


def _braced_struct_assignment_target_source():
    return """
    struct MixedBox {
        float2 loaded;
        float2 imageValue;
        float2 rawValue;
    };

    StructuredBuffer<float2> pairs;
    ConsumeStructuredBuffer<float2> consumed;
    RWByteAddressBuffer raw;
    RWTexture2D<float2> image;

    uint index() {
        return 0u;
    }

    int2 pixel() {
        return int2(0, 0);
    }

    void acceptMixed(MixedBox value) {
    }

    void useBracedBoxes(int mode, bool choose) {
        let inferred = MixedBox {
            loaded: choose ? pairs.Load(index()) : consumed.Consume(),
            imageValue: image.Load(pixel()),
            rawValue: match mode {
                0 => asfloat(raw.Load2(index())),
                _ => pairs.Load(index())
            }
        };
        MixedBox explicitValue = MixedBox {
            loaded: consumed.Consume(),
            imageValue: choose ? image.Load(pixel()) : pairs.Load(index()),
            rawValue: asfloat(raw.Load2(index()))
        };
        MixedBox assigned;
        assigned = MixedBox {
            loaded: match mode {
                0 => pairs.Load(index()),
                _ => consumed.Consume()
            },
            imageValue: image.Load(pixel()),
            rawValue: choose ? asfloat(raw.Load2(index())) : pairs.Load(index())
        };
        inferred = MixedBox {
            loaded: consumed.Consume(),
            imageValue: image.Load(pixel()),
            rawValue: asfloat(raw.Load2(index()))
        };
        acceptMixed(MixedBox {
            loaded: pairs.Load(index()),
            imageValue: image.Load(pixel()),
            rawValue: asfloat(raw.Load2(index()))
        });
    }
    """


def test_braced_struct_constructors_preserve_assignment_target_contexts():
    generated_code = generate_code(
        parse_code(tokenize_code(_braced_struct_assignment_target_source()))
    )

    assert "fn useBracedBoxes(mode: Int32, choose: Bool) -> None:" in generated_code
    assert "var __cgl_match_value_0: SIMD[DType.float32, 2]" in generated_code
    assert "__cgl_match_value_0 = asfloat(buffer_load2(raw, index()))" in (
        generated_code
    )
    assert "__cgl_match_value_0 = buffer_load(pairs, index())" in generated_code
    assert (
        "var inferred = MixedBox(loaded=(buffer_load(pairs, index()) if choose "
        "else buffer_consume(consumed)), imageValue=image_load(image, pixel()), "
        "rawValue=__cgl_match_value_0)"
    ) in generated_code
    assert (
        "var explicitValue: MixedBox = MixedBox(loaded=buffer_consume(consumed), "
        "imageValue=(image_load(image, pixel()) if choose else "
        "buffer_load(pairs, index())), rawValue=asfloat(buffer_load2(raw, index())))"
    ) in generated_code
    assert "var assigned = MixedBox(" in generated_code
    assert "var __cgl_match_value_1: SIMD[DType.float32, 2]" in generated_code
    assert "__cgl_match_value_1 = buffer_load(pairs, index())" in generated_code
    assert "__cgl_match_value_1 = buffer_consume(consumed)" in generated_code
    assert (
        "assigned = MixedBox(loaded=__cgl_match_value_1, "
        "imageValue=image_load(image, pixel()), rawValue=(asfloat(buffer_load2(raw, "
        "index())) if choose else buffer_load(pairs, index())))"
    ) in generated_code
    assert (
        "inferred = MixedBox(loaded=buffer_consume(consumed), "
        "imageValue=image_load(image, pixel()), rawValue=asfloat(buffer_load2(raw, "
        "index())))"
    ) in generated_code
    assert (
        "acceptMixed(MixedBox(loaded=buffer_load(pairs, index()), "
        "imageValue=image_load(image, pixel()), rawValue=asfloat(buffer_load2(raw, "
        "index()))))"
    ) in generated_code
    assert "MatchNode" not in generated_code


def test_braced_struct_assignment_contexts_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    generated_code = generate_code(
        parse_code(tokenize_code(_braced_struct_assignment_target_source()))
    )
    generated_code += """
fn main():
    useBracedBoxes(0, True)
"""

    source_path = tmp_path / "braced_struct_assignment_contexts.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "source, pattern",
    [
        (
            """
            struct ScalarBox {
                float value;
            };
            StructuredBuffer<float2> pairs;
            uint index() { return 0u; }
            void invalidInferred() {
                let badValue = ScalarBox { value: pairs.Load(index()) };
            }
            """,
            r"buffer_load target.*declaration badValue field ScalarBox\.value"
            r".*expects float2.*got float",
        ),
        (
            """
            struct ScalarBox {
                float value;
            };
            RWTexture2D<float2> image;
            int2 pixel() { return int2(0, 0); }
            void invalidInferredImage() {
                let badValue = ScalarBox { value: image.Load(pixel()) };
            }
            """,
            r"image_load target.*declaration badValue field ScalarBox\.value"
            r".*expects vec2.*got float",
        ),
        (
            """
            struct ScalarBox {
                float value;
            };
            RWByteAddressBuffer raw;
            uint index() { return 0u; }
            void invalidInferredReinterpret() {
                let badValue = ScalarBox { value: asfloat(raw.Load2(index())) };
            }
            """,
            r"asfloat target.*declaration badValue field ScalarBox\.value"
            r".*expects vec2.*got float",
        ),
        (
            """
            struct ScalarBox {
                float value;
            };
            ConsumeStructuredBuffer<float2> consumed;
            void invalidInferredMatch(int mode) {
                let badValue = ScalarBox {
                    value: match mode {
                        0 => consumed.Consume(),
                        _ => 1.0
                    }
                };
            }
            """,
            r"buffer_consume target.*declaration badValue field ScalarBox\.value "
            r"match arm 1.*expects float2.*got float",
        ),
    ],
)
def test_braced_struct_constructor_assignment_diagnostics(source, pattern):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


@pytest.mark.parametrize(
    "source, pattern",
    [
        (
            """
            struct ScalarBox {
                float value;
            };
            void invalidUnknownField() {
                let badValue = ScalarBox { missing: 1.0 };
            }
            """,
            r"unknown field ScalarBox\.missing.*declaration badValue"
            r".*expected one of: value",
        ),
        (
            """
            struct ScalarBox {
                float value;
            };
            void invalidExtraPositional() {
                let badValue = ScalarBox { 1.0, 2.0 };
            }
            """,
            r"too many positional fields.*declaration badValue" r".*field 2.*ScalarBox",
        ),
        (
            """
            struct PairBox {
                float value;
                float alt;
            };
            void invalidDuplicateField() {
                let badValue = PairBox { 1.0, value: 2.0 };
            }
            """,
            r"duplicate field PairBox\.value.*declaration badValue",
        ),
    ],
)
def test_braced_struct_constructor_field_diagnostics(source, pattern):
    with pytest.raises(ValueError, match=pattern):
        generate_code(parse_code(tokenize_code(source)))


def _constructor_helper_match_and_ternary_source():
    return """
    vec2 makeUv() {
        return vec2(0.25, 0.75);
    }

    vec2 makeAltUv() {
        return vec2(1.25, 1.75);
    }

    ivec2 makeIndexPair() {
        return ivec2(2, 3);
    }

    vec4 packTernaryVec(bool useAlt) {
        return vec4(useAlt ? makeAltUv() : makeUv(), makeIndexPair());
    }

    mat2 packTernaryMatrix(bool useAlt) {
        return mat2(useAlt ? makeAltUv() : makeUv(), makeIndexPair());
    }

    vec4 packMatchVec(int mode) {
        return vec4(match mode {
            0 => makeUv(),
            _ => makeAltUv()
        }, makeIndexPair());
    }

    mat2 packMatchMatrix(int mode) {
        return mat2(match mode {
            0 => makeUv(),
            _ => makeAltUv()
        }, makeIndexPair());
    }
    """


def test_constructor_helper_match_and_ternary_operands_emit_single_preludes():
    generated_code = generate_code(
        parse_code(tokenize_code(_constructor_helper_match_and_ternary_source()))
    )

    vector_helper = "_crossgl_construct_f32_4_vf322_01_vi322_01"
    matrix_helper = "_crossgl_construct_matrix_f32_c2_r2_2_vf322_01_vi322_01"
    assert f"fn {vector_helper}" in generated_code
    assert f"fn {matrix_helper}" in generated_code
    assert (
        f"return {vector_helper}((makeAltUv() if useAlt else makeUv()), "
        "makeIndexPair())"
    ) in generated_code
    assert (
        f"return {matrix_helper}((makeAltUv() if useAlt else makeUv()), "
        "makeIndexPair())"
    ) in generated_code
    assert "var __cgl_match_value_0: SIMD[DType.float32, 2]" in generated_code
    assert "__cgl_match_value_0 = makeUv()" in generated_code
    assert "__cgl_match_value_0 = makeAltUv()" in generated_code
    assert f"return {vector_helper}(__cgl_match_value_0, makeIndexPair())" in (
        generated_code
    )
    assert "var __cgl_match_value_1: SIMD[DType.float32, 2]" in generated_code
    assert f"return {matrix_helper}(__cgl_match_value_1, makeIndexPair())" in (
        generated_code
    )
    assert generated_code.count("var __cgl_match_value_") == 2
    assert "MatchNode" not in generated_code
    assert "makeIndexPair()[0]" not in generated_code
    assert "makeIndexPair()[1]" not in generated_code
    assert "SIMD[DType.float32, 4]((makeAltUv() if useAlt else makeUv())" not in (
        generated_code
    )
    assert "CrossGLMatrixF32C2R2((makeAltUv() if useAlt else makeUv())" not in (
        generated_code
    )


def test_constructor_helper_match_and_ternary_operands_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    generated_code = generate_code(
        parse_code(tokenize_code(_constructor_helper_match_and_ternary_source()))
    )
    generated_code += """
fn main():
    print(packTernaryVec(False))
    print(packTernaryVec(True))
    print(packTernaryMatrix(False).c0)
    print(packTernaryMatrix(False).c1)
    print(packTernaryMatrix(True).c0)
    print(packMatchVec(0))
    print(packMatchVec(1))
    print(packMatchMatrix(0).c0)
    print(packMatchMatrix(0).c1)
    print(packMatchMatrix(1).c0)
"""

    source_path = tmp_path / "constructor_helper_match_and_ternary.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[0.25, 0.75, 2.0, 3.0]" in result.stdout
    assert "[1.25, 1.75, 2.0, 3.0]" in result.stdout
    assert "[2.0, 3.0]" in result.stdout
    assert "[0.25, 0.75]" in result.stdout
    assert "[1.25, 1.75]" in result.stdout


def _constructor_helper_resource_query_branch_source():
    return """
    sampler1D widthMap;
    sampler2D colorMap;
    image2D colorImage;

    ivec2 makeIndexPair() {
        return ivec2(2, 3);
    }

    float nextFloat() {
        return 9.0;
    }

    vec4 packScalarTernaryResource(bool highMip, sampler1D tex) {
        return vec4(
            makeIndexPair(),
            highMip ? textureSize(tex, 1) : textureSize(tex, 0),
            nextFloat()
        );
    }

    vec4 packScalarMatchResource(int mode, sampler1D tex) {
        return vec4(makeIndexPair(), match mode {
            0 => textureSize(tex, 0),
            _ => textureSize(tex, 1)
        }, nextFloat());
    }

    vec4 packVectorTernaryResource(bool useTex, sampler2D tex, image2D image) {
        return vec4(
            useTex ? textureSize(tex, 0) : imageSize(image),
            makeIndexPair()
        );
    }

    mat2 packMatrixVectorTernaryResource(
        bool useTex,
        sampler2D tex,
        image2D image
    ) {
        return mat2(
            useTex ? textureSize(tex, 0) : imageSize(image),
            makeIndexPair()
        );
    }
    """


def test_constructor_helper_resource_query_branches_preserve_target_shapes():
    generated_code = generate_code(
        parse_code(tokenize_code(_constructor_helper_resource_query_branch_source()))
    )

    scalar_helper = "_crossgl_construct_f32_4_vi322_01_si32_s"
    vector_helper = "_crossgl_construct_f32_4_vi322_01_vi322_01"
    matrix_helper = "_crossgl_construct_matrix_f32_c2_r2_2_vi322_01_vi322_01"
    assert f"fn {scalar_helper}" in generated_code
    assert f"fn {vector_helper}" in generated_code
    assert f"fn {matrix_helper}" in generated_code
    assert (
        f"return {scalar_helper}(makeIndexPair(), "
        "(texture_size(tex, 1) if highMip else texture_size(tex, 0)), "
        "nextFloat())"
    ) in generated_code
    assert "var __cgl_match_value_0: Int32" in generated_code
    assert "__cgl_match_value_0 = texture_size(tex, 0)" in generated_code
    assert "__cgl_match_value_0 = texture_size(tex, 1)" in generated_code
    assert (
        f"return {scalar_helper}(makeIndexPair(), __cgl_match_value_0, nextFloat())"
        in generated_code
    )
    assert (
        f"return {vector_helper}((texture_size(tex, 0) if useTex else "
        "image_size(image)), makeIndexPair())"
    ) in generated_code
    assert (
        f"return {matrix_helper}((texture_size(tex, 0) if useTex else "
        "image_size(image)), makeIndexPair())"
    ) in generated_code
    assert generated_code.count("var __cgl_match_value_") == 1
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "MatchNode" not in generated_code


def test_constructor_helper_resource_query_ternary_shape_diagnostics():
    code = """
    sampler2D colorMap;
    sampler2DArray layerMap;

    ivec2 makeIndexPair() {
        return ivec2(2, 3);
    }

    float nextFloat() {
        return 9.0;
    }

    vec4 invalidScalarHelperTernary(
        bool useLayer,
        sampler2D tex,
        sampler2DArray layers
    ) {
        return vec4(
            makeIndexPair(),
            useLayer ? textureSize(layers, 0) : textureSize(tex, 0),
            nextFloat()
        );
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            r"texture_size target.*constructor helper argument 2 true branch"
            r".*expects ivec3.*got float"
        ),
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_constructor_helper_resource_query_branches_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    generated_code = generate_code(
        parse_code(tokenize_code(_constructor_helper_resource_query_branch_source()))
    )
    generated_code += """
fn main():
    print(packScalarTernaryResource(False, Texture1D()))
    print(packScalarTernaryResource(True, Texture1D()))
    print(packScalarMatchResource(0, Texture1D()))
    print(packVectorTernaryResource(False, Texture2D(), Image2D()))
    print(packMatrixVectorTernaryResource(True, Texture2D(), Image2D()).c0)
"""

    source_path = tmp_path / "constructor_helper_resource_query_branches.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[2.0, 3.0, 0.0, 9.0]" in result.stdout
    assert "[0.0, 0.0, 2.0, 3.0]" in result.stdout


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


def test_min_precision_truncating_vector_constructors_use_helpers():
    code = """
    half4 makeHalf4() {
        return half4(1.0, 2.0, 3.0, 4.0);
    }

    min16float4 makeMinFloat4() {
        return min16float4(1.5, 2.5, 3.5, 4.5);
    }

    short4 makeShort4() {
        return short4(5, 6, 7, 8);
    }

    min16int4 makeMinInt4() {
        return min16int4(9, 10, 11, 12);
    }

    ushort4 makeUShort4() {
        return ushort4(13, 14, 15, 16);
    }

    min16uint4 makeMinUInt4() {
        return min16uint4(17, 18, 19, 20);
    }

    half2 narrowHalf() {
        return half2(makeHalf4());
    }

    min16float3 narrowMinFloat() {
        return min16float3(makeMinFloat4());
    }

    short2 narrowShort() {
        return short2(makeShort4());
    }

    min16int3 narrowMinInt() {
        return min16int3(makeMinInt4());
    }

    ushort2 narrowUShort() {
        return ushort2(makeUShort4());
    }

    min16uint3 narrowMinUInt() {
        return min16uint3(makeMinUInt4());
    }

    half3 overfullHalf(half2 lo, half2 hi) {
        return half3(lo, hi);
    }

    half3 overfullHalfDuplicate() {
        return half3(half2(1.0, 2.0), half2(3.0, 4.0));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fn _crossgl_construct_f16_2_vf164_01" in generated_code
    assert "fn _crossgl_construct_f16_4_vf164_012" in generated_code
    assert "fn _crossgl_construct_i16_2_vi164_01" in generated_code
    assert "fn _crossgl_construct_i16_4_vi164_012" in generated_code
    assert "fn _crossgl_construct_u16_2_vu164_01" in generated_code
    assert "fn _crossgl_construct_u16_4_vu164_012" in generated_code
    assert "fn _crossgl_construct_f16_4_vf162_01_vf162_0" in generated_code
    assert "return _crossgl_construct_f16_2_vf164_01(makeHalf4())" in generated_code
    assert (
        "return _crossgl_construct_f16_4_vf164_012(makeMinFloat4())" in generated_code
    )
    assert "return _crossgl_construct_i16_2_vi164_01(makeShort4())" in generated_code
    assert "return _crossgl_construct_i16_4_vi164_012(makeMinInt4())" in generated_code
    assert "return _crossgl_construct_u16_2_vu164_01(makeUShort4())" in generated_code
    assert "return _crossgl_construct_u16_4_vu164_012(makeMinUInt4())" in generated_code
    assert "return SIMD[DType.float16, 4](lo[0], lo[1], hi[0], 0.0)" in generated_code
    assert (
        "return _crossgl_construct_f16_4_vf162_01_vf162_0("
        "SIMD[DType.float16, 2](1.0, 2.0), "
        "SIMD[DType.float16, 2](3.0, 4.0))"
    ) in generated_code
    assert "makeHalf4()[0]" not in generated_code
    assert "makeMinFloat4()[0]" not in generated_code
    assert "makeShort4()[0]" not in generated_code
    assert "makeMinInt4()[0]" not in generated_code
    assert "makeUShort4()[0]" not in generated_code
    assert "makeMinUInt4()[0]" not in generated_code
    assert "hi[1]" not in generated_code


def test_min_precision_truncating_vector_constructors_compile_with_mojo(tmp_path):
    mojo = find_mojo_compiler()

    code = """
    half4 makeHalf4() {
        return half4(1.0, 2.0, 3.0, 4.0);
    }

    min16float4 makeMinFloat4() {
        return min16float4(1.5, 2.5, 3.5, 4.5);
    }

    short4 makeShort4() {
        return short4(5, 6, 7, 8);
    }

    min16int4 makeMinInt4() {
        return min16int4(9, 10, 11, 12);
    }

    ushort4 makeUShort4() {
        return ushort4(13, 14, 15, 16);
    }

    min16uint4 makeMinUInt4() {
        return min16uint4(17, 18, 19, 20);
    }

    half2 narrowHalf() {
        return half2(makeHalf4());
    }

    min16float3 narrowMinFloat() {
        return min16float3(makeMinFloat4());
    }

    short2 narrowShort() {
        return short2(makeShort4());
    }

    min16int3 narrowMinInt() {
        return min16int3(makeMinInt4());
    }

    ushort2 narrowUShort() {
        return ushort2(makeUShort4());
    }

    min16uint3 narrowMinUInt() {
        return min16uint3(makeMinUInt4());
    }

    half3 overfullHalf(half2 lo, half2 hi) {
        return half3(lo, hi);
    }

    half3 overfullHalfDuplicate() {
        return half3(half2(1.0, 2.0), half2(3.0, 4.0));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)
    generated_code += """
fn main():
    print(narrowHalf())
    print(narrowMinFloat())
    print(narrowShort())
    print(narrowMinInt())
    print(narrowUShort())
    print(narrowMinUInt())
    print(overfullHalf(SIMD[DType.float16, 2](21.0, 22.0), SIMD[DType.float16, 2](23.0, 24.0)))
    print(overfullHalfDuplicate())
"""

    source_path = tmp_path / "min_precision_truncating_vector_constructors.mojo"
    source_path.write_text(generated_code)
    result = subprocess.run(
        [mojo, "run", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[5, 6]" in result.stdout
    assert "[9, 10, 11, 0]" in result.stdout
    assert "[13, 14]" in result.stdout
    assert "[17, 18, 19, 0]" in result.stdout


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


def test_match_statement_nonfinal_wildcard_is_rejected_for_mojo_codegen():
    code = """
    shader main {
        compute {
            int main(int mode) {
                int value = 0;
                match mode {
                    _ => {
                        value = 1;
                    }
                    1 => {
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

    with pytest.raises(ValueError, match="wildcard arm must be final"):
        generate_code(ast)


def test_match_guarded_literal_arm_lowers_for_mojo_codegen():
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

    generated_code = generate_code(ast)

    assert "if (mode == 0) and ((mode > 0)):" in generated_code
    assert "value = 1" in generated_code
    assert "else:\n        value = 2" in generated_code
    assert "Unsupported match arm" not in generated_code


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
