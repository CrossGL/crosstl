import shutil
import subprocess
from typing import List

import pytest
import crosstl.translator
from crosstl.translator.parser import Parser
from crosstl.translator.lexer import Lexer
from crosstl.translator.ast import (
    AtomicOpNode,
    BufferOpNode,
    BuiltinVariableNode,
    CastNode,
    LiteralNode,
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
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))
    generated_code += """
fn main():
    print(makePair(4.0, 5.0).y)
    print(makePositionalPair(6.0, 7.0).x)
    print(sumPair())
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


def test_payload_enums_are_rejected_for_mojo_codegen():
    code = """
    enum MaybeInt {
        Some(int),
        None
    };
    """

    with pytest.raises(ValueError, match="Unsupported enum payload for Mojo"):
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

    assert "struct Image2D:" in generated_code
    assert "struct IImage2D:" in generated_code
    assert "struct UImage2D:" in generated_code
    assert "var outColor: Image2D = Image2D()" in generated_code
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

    assert "struct Image1D:" in generated_code
    assert "struct IImage1DArray:" in generated_code
    assert "struct UImage2D:" in generated_code
    assert "struct Image2DArray:" in generated_code
    assert "struct IImage3D:" in generated_code
    assert "var row: Image1D = Image1D()" in generated_code
    assert "var signedRows: IImage1DArray = IImage1DArray()" in generated_code
    assert "var counters: UImage2D = UImage2D()" in generated_code
    assert "var layers: Image2DArray = Image2DArray()" in generated_code
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
    assert "buffer_dimensions(bins, count)" in generated_code
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


def test_direct_vector_cast_nodes_emit_mojo_simd_casts():
    codegen = MojoCodeGen()
    vector_expr = CastNode(
        VariableNode("mask", VectorType(PrimitiveType("uint"), 4)),
        VectorType(PrimitiveType("float"), 4),
    )

    codegen.register_variable_type("mask", "uvec4")

    assert codegen.generate_expression(vector_expr) == "mask.cast[DType.float32]()"
    assert codegen.expression_result_type(vector_expr) == "vec4"


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
